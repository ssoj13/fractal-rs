//! The fractal path tracer's CUDA kernels (Rust compiled by cuda-oxide), the single source
//! shared by WarpBro (built in through `#[cuda_module]`, loaded with [`kernels::load`]) and
//! ofx-rs `ofx-fractal` (embedded PTX/cubin artifacts regenerated from this crate).
//!
//! Originally a port of ofx-rs `ofx-fractal`: `fractal3d.wgsl` (the eight distance estimates,
//! march, normals, colour, sky) and `pathtrace.wgsl` + render-rs `pt-integrator` (path tracing
//! with sun/sky NEE and MIS), with the render-rs `standard-surface-bsdf` crate called directly as
//! the full material model. The parameter ABI and the host-shared helpers live in
//! [`fractal_core`].
//!
//! Kernel dispatch:
//! - Family-specific Fast/Full kernels serve legacy single-fractal renders.
//! - `world` evaluates node worlds with per-object material selection. `world_fast` removes
//!   unused Standard Surface branches in homogeneous Fast worlds; `world_fast_bulb` also
//!   specializes the DE for one evaluated Mandelbulb while retaining world transforms/lights.
//! - `tonemap`: running mean -> exposure / saturation -> scene-linear RGBA32F for vfx-ocio.
//!
//! Pixels are traced in 8x4 tiles (one warp each) so neighbouring rays share their march paths.

use cuda_device::{
    ConstantMemory, DisjointSlice, constant, kernel, launch_bounds, launch_contract, thread,
};
use cuda_host::cuda_module;
use fractal_core::params::*;

#[cuda_module]
#[expect(
    clippy::too_many_arguments,
    reason = "#[kernel] entry points take the launch ABI: each device buffer the host binds is \
              one parameter. Device helpers keep grouped arguments (PixelOut, Context)."
)]
pub mod kernels {
    use super::*;
    use fractal_core::path_sampling::{
        allows_surface_vertex, fast_ggx_pdf, fast_ggx_sample_local, fast_spec_probability,
    };
    use fractal_core::transmission::{
        AIR, exit_distance, medium_after_scatter, volume_transmittance,
    };
    // Every angle here is bounded far below sin_cos's 105615 (polar angles <= pi, sampled
    // azimuths 2 pi u, Mandelbulb angles <= 64 * 8 * pi + 4 pi with the UI's power, angle-scale
    // and phase ranges), so it gives libdevice's bits without the local-memory slow path.
    use cuda_device::math::sin_cos;
    use standard_surface_bsdf::ThinFilmEnergy;
    use standard_surface_bsdf::sample::{pdf_prepared, sample_prepared};
    use standard_surface_bsdf::surface::{
        ShadingFrame, SurfaceInputs, eval_emission, eval_light_prepared, layers,
        lobe_weights_prepared,
    };

    // =========================================================================
    // vector math
    // =========================================================================

    pub type V3 = [f32; 3];

    #[inline(always)]
    fn add(a: V3, b: V3) -> V3 {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
    }
    #[inline(always)]
    fn sub(a: V3, b: V3) -> V3 {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }
    #[inline(always)]
    fn mul(a: V3, s: f32) -> V3 {
        [a[0] * s, a[1] * s, a[2] * s]
    }
    #[inline(always)]
    fn had(a: V3, b: V3) -> V3 {
        [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
    }
    #[inline(always)]
    fn dot(a: V3, b: V3) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }
    #[inline(always)]
    fn length(a: V3) -> f32 {
        dot(a, a).sqrt()
    }
    #[inline(always)]
    fn normalize(a: V3) -> V3 {
        mul(a, 1.0 / length(a))
    }
    #[inline(always)]
    fn max3(a: V3) -> f32 {
        a[0].max(a[1].max(a[2]))
    }
    #[inline(always)]
    fn neg(a: V3) -> V3 {
        [-a[0], -a[1], -a[2]]
    }
    /// Luminance in the host's working space: the weights it wrote to `P_LUMA`
    /// (`fractal_core::luma::Luma`: ACEScg for WarpBro, Rec. 709 for OFX), one constant-cache
    /// broadcast per component.
    #[inline(always)]
    fn luminance(a: V3) -> f32 {
        dot(a, [global(P_LUMA), global(P_LUMA + 1), global(P_LUMA + 2)])
    }
    /// The per-launch parameter block (`params.rs` slots), set by the host before each launch.
    /// Every thread reads the same slot, so the constant cache broadcasts it to the warp.
    #[constant]
    static PARAMS: ConstantMemory<[f32; P_COUNT]> = ConstantMemory::UNINIT;

    #[inline(always)]
    fn pr(ctx: Context<'_>, i: usize) -> f32 {
        if ctx.world
            && ((P_FAMILY..P_LIGHT_DIR).contains(&i)
                || (P_BASE..P_EXPOSURE).contains(&i)
                || i == P_MATERIAL_MODEL
                || (P_TRANSMISSION..P_COUNT).contains(&i))
        {
            ctx.objects[ctx.object * OBJECT_STRIDE + i]
        } else {
            global(i)
        }
    }
    #[inline(always)]
    fn pv3(ctx: Context<'_>, i: usize) -> V3 {
        [pr(ctx, i), pr(ctx, i + 1), pr(ctx, i + 2)]
    }
    /// Row-major 3x3 at slot `i` times `v`.
    #[inline(always)]
    fn mat3(ctx: Context<'_>, i: usize, v: V3) -> V3 {
        [
            pr(ctx, i) * v[0] + pr(ctx, i + 1) * v[1] + pr(ctx, i + 2) * v[2],
            pr(ctx, i + 3) * v[0] + pr(ctx, i + 4) * v[1] + pr(ctx, i + 5) * v[2],
            pr(ctx, i + 6) * v[0] + pr(ctx, i + 7) * v[1] + pr(ctx, i + 8) * v[2],
        ]
    }

    #[derive(Clone, Copy)]
    struct Context<'a> {
        world: bool,
        objects: &'a [f32],
        lights: &'a [f32],
        object: usize,
    }

    #[inline(always)]
    fn global(i: usize) -> f32 {
        PARAMS.get_ref()[i]
    }

    const PI: f32 = core::f32::consts::PI;
    const TRAP_START: f32 = 3.402_823_5e38;
    const DERIVATIVE_LIMIT: f32 = 1.0e38;

    // =========================================================================
    // RNG: Owen-scrambled Sobol per pixel (`standard_surface_bsdf::sampling::sobol`, shared with
    // the CPU and WGSL path tracers of render-rs pt-integrator), dimensions in groups of four
    // =========================================================================

    pub struct Rng {
        px: u32,
        py: u32,
        sample: u32,
        dim: u32,
    }

    #[inline(always)]
    fn rand(r: &mut Rng) -> f32 {
        let u = standard_surface_bsdf::sampling::sobol(r.px, r.py, r.sample, r.dim);
        r.dim += 1;
        u
    }

    // =========================================================================
    // distance estimates (fractal3d.wgsl), each -> (DE, trap minimum)
    // =========================================================================

    #[inline(always)]
    fn trap3(ctx: Context<'_>, z: V3, mode: u32) -> f32 {
        if mode == 1 {
            return length(z);
        }
        let offset = sub(z, pv3(ctx, P_TRAP_POINT));
        if mode == 2 {
            return dot(offset, pv3(ctx, P_TRAP_NORMAL)).abs();
        }
        length(offset)
    }

    #[inline(always)]
    fn iter_rotate(ctx: Context<'_>, z: V3) -> V3 {
        if pr(ctx, P_ITER_ROTATE) == 0.0 {
            z
        } else {
            mat3(ctx, P_ITER_ROT, z)
        }
    }

    /// The Mandelbulb / Mandelbox orbit constant c at point `p` and its derivative dc/dp: the
    /// Julia mix t (`P_JULIA`) morphs c = mix(p, J, t) continuously from the Mandelbrot form
    /// (t = 0: c = p, dc/dp = 1) to the Julia form (t = 1: c = J, dc/dp = 0); the offset k
    /// (`P_ORBIT_OFFSET`) shifts c without changing dc/dp. Both ends and a zero offset take no
    /// arithmetic, so they keep the former Mandelbrot / Julia constants bit for bit.
    #[inline(always)]
    fn orbit_constant(ctx: Context<'_>, p: V3) -> (V3, f32) {
        let t = pr(ctx, P_JULIA);
        let c = if t == 0.0 {
            p
        } else if t == 1.0 {
            pv3(ctx, P_JULIA_C)
        } else {
            add(mul(p, 1.0 - t), mul(pv3(ctx, P_JULIA_C), t))
        };
        let k = pv3(ctx, P_ORBIT_OFFSET);
        let c = if k[0] == 0.0 && k[1] == 0.0 && k[2] == 0.0 {
            c
        } else {
            add(c, k)
        };
        (c, 1.0 - t)
    }

    /// A fractional iteration count n = whole + frac (`P_ITERATIONS`) for a family's loop: it
    /// runs `run` iterations (whole, one more when frac > 0), records its estimate and trap
    /// before the extra one ([`fraction_due`] / [`fraction_save`]) and blends them with the
    /// final pair by frac ([`fraction_blend`]). The result is continuous in n, both lerped
    /// fields are the same family's estimates; an integer n runs exactly the former loop.
    #[derive(Clone, Copy)]
    struct Fraction {
        run: u32,
        frac: f32,
        saved: Option<(f32, f32)>,
    }

    #[inline(always)]
    fn fraction(ctx: Context<'_>) -> Fraction {
        let n = pr(ctx, P_ITERATIONS).max(0.0);
        let whole = n as u32;
        let frac = n - whole as f32;
        Fraction {
            run: whole + (frac > 0.0) as u32,
            frac,
            saved: None,
        }
    }

    /// Whether iteration `i` (about to run) is the extra one: the caller saves its estimate of
    /// the state after `whole` iterations first.
    #[inline(always)]
    fn fraction_due(f: &Fraction, i: u32) -> bool {
        f.frac > 0.0 && i + 1 == f.run
    }

    #[inline(always)]
    fn fraction_save(f: &mut Fraction, distance: f32, trap: f32) {
        f.saved = Some((distance, trap));
    }

    /// The final (distance, trap), blended with the saved pair when the extra iteration ran. An
    /// orbit that stopped earlier has no saved pair: its state after `whole` and `whole + 1`
    /// iterations is the same.
    #[inline(always)]
    fn fraction_blend(f: &Fraction, distance: f32, trap: f32) -> (f32, f32) {
        match f.saved {
            Some((d, t)) => (d + (distance - d) * f.frac, t + (trap - t) * f.frac),
            None => (distance, trap),
        }
    }

    /// One Mandelbulb step z <- z^p + c, dr <- g p r^(p-1) dr + k. Returns false to stop.
    #[inline(always)]
    fn bulb_step(
        ctx: Context<'_>,
        z: &mut V3,
        dr: &mut f32,
        r: f32,
        c: V3,
        dr_constant: f32,
    ) -> bool {
        if pr(ctx, P_BULB_FAST8) != 0.0 {
            // Power 8, unit angle scales, no phase, no rotation: z^8 with three angle doublings
            // instead of acos / atan2 / pow (same set as the angular form, ~3x cheaper).
            let r2 = r * r;
            let r4 = r2 * r2;
            let growth = 8.0 * r4 * r2 * r;
            if *dr > DERIVATIVE_LIMIT / growth {
                return false;
            }
            *dr = growth * *dr + dr_constant;
            let (x, y, w) = (z[0], z[1], z[2]);
            let rxy2 = x * x + y * y;
            let r8 = r4 * r4;
            if rxy2 < 1.0e-20 {
                *z = add([0.0, 0.0, r8], c);
            } else {
                let inv_r2 = 1.0 / r2;
                let mut ct = 2.0 * w * w * inv_r2 - 1.0;
                let mut st = 2.0 * w * rxy2.sqrt() * inv_r2;
                let c2 = ct * ct - st * st;
                st = 2.0 * st * ct;
                ct = c2;
                let c2 = ct * ct - st * st;
                st = 2.0 * st * ct;
                ct = c2;
                let inv_rxy2 = 1.0 / rxy2;
                let mut cp = (x * x - y * y) * inv_rxy2;
                let mut sp = 2.0 * x * y * inv_rxy2;
                let c2 = cp * cp - sp * sp;
                sp = 2.0 * sp * cp;
                cp = c2;
                let c2 = cp * cp - sp * sp;
                sp = 2.0 * sp * cp;
                cp = c2;
                *z = add([st * cp * r8, st * sp * r8, ct * r8], c);
            }
            return true;
        }
        let power = pr(ctx, P_BULB_POWER);
        let turned = iter_rotate(ctx, *z);
        let theta = (turned[2] / r).clamp(-1.0, 1.0).acos();
        let phi = turned[1].atan2(turned[0]);
        let lr = r.ln();
        let growth = power * ((power - 1.0) * lr).exp() * pr(ctx, P_BULB_GROWTH);
        if *dr > DERIVATIVE_LIMIT / growth {
            return false;
        }
        let magnitude = (power * lr).exp();
        *dr = growth * *dr + dr_constant;
        let a = theta * pr(ctx, P_BULB_THETA_POWER) + pr(ctx, P_BULB_THETA_PHASE);
        let b = phi * pr(ctx, P_BULB_PHI_POWER) + pr(ctx, P_BULB_PHI_PHASE);
        let (sa, ca) = sin_cos(a);
        let (sb, cb) = sin_cos(b);
        *z = add(mul([sa * cb, sa * sb, ca], magnitude), c);
        true
    }

    #[inline(always)]
    fn bulb_distance(ctx: Context<'_>, q: V3, trap_mode: u32) -> (f32, f32) {
        let bailout = pr(ctx, P_BAILOUT);
        let (c, dr_constant) = orbit_constant(ctx, q);
        let mut fraction = fraction(ctx);
        let estimate = |r: f32, dr: f32| {
            if r <= bailout || dr <= 0.0 {
                0.0
            } else {
                0.5 * r.ln() * r / dr
            }
        };
        let mut z = q;
        let mut dr = 1.0f32;
        let mut r = length(z);
        let mut trap = TRAP_START;
        let mut i = 0u32;
        while i < fraction.run {
            if fraction_due(&fraction, i) {
                fraction_save(&mut fraction, estimate(r, dr), trap);
            }
            if r > bailout {
                break;
            }
            // Inside after `whole` iterations too (r <= bailout): both blended estimates are 0.
            if r < 1.0e-8 {
                return (0.0, trap);
            }
            if !bulb_step(ctx, &mut z, &mut dr, r, c, dr_constant) {
                break;
            }
            r = length(z);
            if trap_mode != 0 {
                trap = trap.min(trap3(ctx, z, trap_mode));
            }
            i += 1;
        }
        fraction_blend(&fraction, estimate(r, dr), trap)
    }

    #[inline(always)]
    fn box_fold(v: f32, limit: f32) -> f32 {
        if v > limit {
            2.0 * limit - v
        } else if v < -limit {
            -2.0 * limit - v
        } else {
            v
        }
    }

    /// One Mandelbox step. Returns false to stop.
    #[inline(always)]
    fn box_step(ctx: Context<'_>, z: &mut V3, dr: &mut f32, c: V3, dr_constant: f32) -> bool {
        let signed_scale = pr(ctx, P_BOX_SCALE);
        let magnitude = signed_scale.abs();
        let fold = pr(ctx, P_BOX_FOLD);
        let min_r2 = pr(ctx, P_BOX_MIN_R2);
        let fixed_r2 = pr(ctx, P_BOX_FIXED_R2);
        let t = iter_rotate(ctx, *z);
        let t = [
            box_fold(t[0], fold),
            box_fold(t[1], fold),
            box_fold(t[2], fold),
        ];
        let r2 = dot(t, t);
        let factor = if r2 < min_r2 {
            fixed_r2 / min_r2
        } else if r2 < fixed_r2 {
            fixed_r2 / r2
        } else {
            1.0
        };
        if *dr > DERIVATIVE_LIMIT / (factor * magnitude) {
            return false;
        }
        *z = add(mul(t, factor * signed_scale), c);
        *dr = *dr * factor * magnitude + dr_constant;
        true
    }

    #[inline(always)]
    fn box_distance(ctx: Context<'_>, q: V3, trap_mode: u32) -> (f32, f32) {
        let (c, dr_constant) = orbit_constant(ctx, q);
        let mut fraction = fraction(ctx);
        let mut z = q;
        let mut dr = 1.0f32;
        let mut trap = TRAP_START;
        let mut i = 0u32;
        while i < fraction.run {
            if fraction_due(&fraction, i) {
                fraction_save(&mut fraction, length(z) / dr, trap);
            }
            if !box_step(ctx, &mut z, &mut dr, c, dr_constant) {
                break;
            }
            if trap_mode != 0 {
                trap = trap.min(trap3(ctx, z, trap_mode));
            }
            i += 1;
        }
        fraction_blend(&fraction, length(z) / dr, trap)
    }

    #[inline(always)]
    fn quat_distance(ctx: Context<'_>, v: V3, trap_mode: u32) -> (f32, f32) {
        let bailout = pr(ctx, P_BAILOUT);
        let mut fraction = fraction(ctx);
        let estimate = |r: f32, dr: f32| {
            if r <= bailout || dr <= 0.0 {
                0.0
            } else {
                0.5 * r.ln() * r / dr
            }
        };
        let c = [
            pr(ctx, P_QUAT_C),
            pr(ctx, P_QUAT_C + 1),
            pr(ctx, P_QUAT_C + 2),
            pr(ctx, P_QUAT_C + 3),
        ];
        let row = |k: usize| P_QUAT_ROWS + 3 * k;
        let mut q = [
            pr(ctx, row(0)) * v[0]
                + pr(ctx, row(0) + 1) * v[1]
                + pr(ctx, row(0) + 2) * v[2]
                + pr(ctx, P_QUAT_OFFSET),
            pr(ctx, row(1)) * v[0]
                + pr(ctx, row(1) + 1) * v[1]
                + pr(ctx, row(1) + 2) * v[2]
                + pr(ctx, P_QUAT_OFFSET + 1),
            pr(ctx, row(2)) * v[0]
                + pr(ctx, row(2) + 1) * v[1]
                + pr(ctx, row(2) + 2) * v[2]
                + pr(ctx, P_QUAT_OFFSET + 2),
            pr(ctx, row(3)) * v[0]
                + pr(ctx, row(3) + 1) * v[1]
                + pr(ctx, row(3) + 2) * v[2]
                + pr(ctx, P_QUAT_OFFSET + 3),
        ];
        let len4 = |q: [f32; 4]| (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        let mut dr = 1.0f32;
        let mut r = len4(q);
        let mut trap = TRAP_START;
        let mut i = 0u32;
        while i < fraction.run {
            if fraction_due(&fraction, i) {
                fraction_save(&mut fraction, estimate(r, dr), trap);
            }
            if r > bailout {
                break;
            }
            if r < 1.0e-8 {
                return (0.0, trap);
            }
            let growth = 2.0 * r;
            if dr > DERIVATIVE_LIMIT / growth {
                break;
            }
            dr *= growth;
            q = [
                q[0] * q[0] - q[1] * q[1] - q[2] * q[2] - q[3] * q[3] + c[0],
                2.0 * q[0] * q[1] + c[1],
                2.0 * q[0] * q[2] + c[2],
                2.0 * q[0] * q[3] + c[3],
            ];
            r = len4(q);
            if trap_mode != 0 {
                // trap4: the trap geometry embedded at w = 0
                let t = if trap_mode == 1 {
                    r
                } else {
                    let o = sub([q[0], q[1], q[2]], pv3(ctx, P_TRAP_POINT));
                    if trap_mode == 2 {
                        dot(o, pv3(ctx, P_TRAP_NORMAL)).abs()
                    } else {
                        (dot(o, o) + q[3] * q[3]).sqrt()
                    }
                };
                trap = trap.min(t);
            }
            i += 1;
        }
        fraction_blend(&fraction, estimate(r, dr), trap)
    }

    #[inline(always)]
    fn kifs_fold(ctx: Context<'_>, v: V3) -> V3 {
        let (mut x, mut y, mut z) = (v[0], v[1], v[2]);
        let kind = pr(ctx, P_KIFS_KIND) as u32;
        if kind == 0 {
            if x + y < 0.0 {
                let t = x;
                x = -y;
                y = -t;
            }
            if x + z < 0.0 {
                let t = x;
                x = -z;
                z = -t;
            }
            if y + z < 0.0 {
                let t = y;
                y = -z;
                z = -t;
            }
        } else {
            x = x.abs();
            y = y.abs();
            z = z.abs();
            if x < y {
                core::mem::swap(&mut x, &mut y);
            }
            if x < z {
                core::mem::swap(&mut x, &mut z);
            }
            if y < z {
                core::mem::swap(&mut y, &mut z);
            }
            if kind == 2 {
                let h = pr(ctx, P_KIFS_FOLD_HEIGHT);
                z = h - (z - h).abs();
            }
        }
        [x, y, z]
    }

    #[inline(always)]
    fn kifs_step(ctx: Context<'_>, z: &mut V3, dr: &mut f32) -> bool {
        let s = pr(ctx, P_KIFS_SCALE);
        if *dr > DERIVATIVE_LIMIT / s {
            return false;
        }
        *z = sub(
            mul(kifs_fold(ctx, iter_rotate(ctx, *z)), s),
            pv3(ctx, P_KIFS_SHIFT),
        );
        *dr *= s;
        true
    }

    #[inline(always)]
    fn kifs_distance(ctx: Context<'_>, q: V3, trap_mode: u32) -> (f32, f32) {
        let mut fraction = fraction(ctx);
        let estimate = |z: V3, dr: f32| {
            let bound = if pr(ctx, P_KIFS_KIND) as u32 == 2 {
                // exact signed distance to [-1, 1]^3
                let d = [z[0].abs() - 1.0, z[1].abs() - 1.0, z[2].abs() - 1.0];
                length([d[0].max(0.0), d[1].max(0.0), d[2].max(0.0)])
                    + d[0].max(d[1].max(d[2])).min(0.0)
            } else {
                length(z) - pr(ctx, P_KIFS_BOUND)
            };
            bound / dr
        };
        let mut z = q;
        let mut dr = 1.0f32;
        let mut trap = TRAP_START;
        let mut i = 0u32;
        while i < fraction.run {
            if fraction_due(&fraction, i) {
                fraction_save(&mut fraction, estimate(z, dr), trap);
            }
            if !kifs_step(ctx, &mut z, &mut dr) {
                break;
            }
            if trap_mode != 0 {
                trap = trap.min(trap3(ctx, z, trap_mode));
            }
            i += 1;
        }
        fraction_blend(&fraction, estimate(z, dr), trap)
    }

    #[inline(always)]
    fn exact_sign(v: f32) -> f32 {
        if v > 0.0 {
            1.0
        } else if v < 0.0 {
            -1.0
        } else {
            0.0
        }
    }

    #[inline(always)]
    fn kleinian_wrap(x: f32, width: f32, left: f32) -> f32 {
        let shifted = x - left;
        shifted - width * (shifted / width).floor() + left
    }

    #[inline(always)]
    fn kleinian_distance(ctx: Context<'_>, q: V3, trap_mode: u32) -> (f32, f32) {
        let a = pr(ctx, P_KLEIN_A);
        let b = pr(ctx, P_KLEIN_B);
        let skew = pr(ctx, P_KLEIN_SKEW);
        let line_amp = pr(ctx, P_KLEIN_LINE_AMP);
        let line_rate = pr(ctx, P_KLEIN_LINE_RATE);
        let mut fraction = fraction(ctx);
        let estimate = |z: V3, dr: f32| z[1].min(0.05) / dr.max(1.0);
        let half_a = 0.5 * a;
        let half_b = 0.5 * b;
        let sign_b = exact_sign(b);
        // face_camera: (x, z, -y), then half a cell up.
        let mut z = [q[0], q[2] + half_a, -q[1]];
        let mut dr = 1.0f32;
        let mut previous = z;
        let mut before = z;
        let mut trap = TRAP_START;
        let mut n = 0u32;
        while n < fraction.run {
            if fraction_due(&fraction, n) {
                fraction_save(&mut fraction, estimate(z, dr), trap);
            }
            let (mut x, mut y, mut w) = (z[0], z[1], z[2]);
            let s = skew * y;
            x = kleinian_wrap(x + s, 2.0, -1.0) - s;
            w = kleinian_wrap(w, 2.0, -1.0);
            let u = x + half_b;
            let line =
                half_a + sign_b * line_amp * exact_sign(u) * (1.0 - (-line_rate * u.abs()).exp());
            if y >= line {
                x = -b - x;
                y = a - y;
                w = -w;
            }
            let r2 = x * x + y * y + w * w;
            if r2 < 1.0e-30 {
                break;
            }
            let inverse = 1.0 / r2;
            if inverse > 1.0 && dr > DERIVATIVE_LIMIT / inverse {
                break;
            }
            z = [x * inverse - b, a - y * inverse, -w * inverse];
            dr *= inverse;
            if trap_mode != 0 {
                trap = trap.min(trap3(ctx, z, trap_mode));
            }
            if dot(z, z) > 100.0 {
                break;
            }
            if n >= 1 {
                let gap = sub(z, before);
                if dot(gap, gap) < 1.0e-12 {
                    break;
                }
            }
            before = previous;
            previous = z;
            n += 1;
        }
        fraction_blend(&fraction, estimate(z, dr), trap)
    }

    #[inline(always)]
    fn pseudo_kleinian_distance(ctx: Context<'_>, q: V3, trap_mode: u32) -> (f32, f32) {
        let bx = pv3(ctx, P_PK_BOX);
        let size = pr(ctx, P_PK_SIZE);
        let c = pv3(ctx, P_PK_C);
        let mut fraction = fraction(ctx);
        let estimate = |z: V3, dr: f32| {
            let o = sub(z, pv3(ctx, P_PK_OFFSET));
            let t = pr(ctx, P_PK_THICKNESS);
            let thingy = ((o[0] * o[0] + o[1] * o[1]).sqrt() * o[2]).abs() - t;
            0.5 * thingy / (dot(o, o) + t.abs()).sqrt().max(1.0e-30) / dr
        };
        let mut z = q;
        let mut dr = 1.0f32;
        let mut trap = TRAP_START;
        let mut i = 0u32;
        while i < fraction.run {
            if fraction_due(&fraction, i) {
                fraction_save(&mut fraction, estimate(z, dr), trap);
            }
            z = [
                2.0 * z[0].clamp(-bx[0], bx[0]) - z[0],
                2.0 * z[1].clamp(-bx[1], bx[1]) - z[1],
                2.0 * z[2].clamp(-bx[2], bx[2]) - z[2],
            ];
            let r2 = dot(z, z);
            if r2 < 1.0e-30 {
                break;
            }
            let k = (size / r2).max(1.0);
            if dr > DERIVATIVE_LIMIT / k {
                break;
            }
            z = add(mul(z, k), c);
            dr *= k;
            if trap_mode != 0 {
                trap = trap.min(trap3(ctx, z, trap_mode));
            }
            i += 1;
        }
        fraction_blend(&fraction, estimate(z, dr), trap)
    }

    /// Apollonian wrap + inversion k = max(s / r2, 0.1). Returns false to stop.
    #[inline(always)]
    fn apollonian_step(ctx: Context<'_>, z: &mut V3, dr: &mut f32) -> bool {
        let h = [0.5 * z[0] + 0.5, 0.5 * z[1] + 0.5, 0.5 * z[2] + 0.5];
        let w = [
            -1.0 + 2.0 * (h[0] - h[0].floor()),
            -1.0 + 2.0 * (h[1] - h[1].floor()),
            -1.0 + 2.0 * (h[2] - h[2].floor()),
        ];
        let r2 = dot(w, w);
        if r2 < 1.0e-30 {
            return false;
        }
        let k = (pr(ctx, P_APOLLO_SCALE) / r2).max(0.1);
        if *dr > DERIVATIVE_LIMIT / k {
            return false;
        }
        *z = mul(w, k);
        *dr *= k;
        true
    }

    #[inline(always)]
    fn apollonian_distance(ctx: Context<'_>, q: V3, trap_mode: u32) -> (f32, f32) {
        let mut fraction = fraction(ctx);
        let estimate = |z: V3, dr: f32| 0.25 * z[1].abs() / dr;
        let mut z = add(q, [0.0, 1.0, 0.0]);
        let mut dr = 1.0f32;
        let mut trap = TRAP_START;
        let mut i = 0u32;
        while i < fraction.run {
            if fraction_due(&fraction, i) {
                fraction_save(&mut fraction, estimate(z, dr), trap);
            }
            if !apollonian_step(ctx, &mut z, &mut dr) {
                break;
            }
            if trap_mode != 0 {
                trap = trap.min(trap3(ctx, z, trap_mode));
            }
            i += 1;
        }
        fraction_blend(&fraction, estimate(z, dr), trap)
    }

    #[inline(always)]
    fn hybrid_distance(ctx: Context<'_>, q: V3, trap_mode: u32) -> (f32, f32) {
        let bailout = pr(ctx, P_BAILOUT);
        let mut fraction = fraction(ctx);
        let steps = pr(ctx, P_HYBRID_STEPS) as u32;
        let count = (pr(ctx, P_HYBRID_COUNT) as u32).max(1);
        let c = q;
        let mut z = q;
        let mut dr = 1.0f32;
        let mut trap = TRAP_START;
        let mut i = 0u32;
        while i < fraction.run {
            let r = length(z);
            if fraction_due(&fraction, i) {
                fraction_save(&mut fraction, 0.5 * r / dr, trap);
            }
            if r > bailout {
                break;
            }
            let step = (steps >> (3 * (i % count))) & 7;
            let ok = if step == 1 {
                r >= 1.0e-8 && bulb_step(ctx, &mut z, &mut dr, r, c, 1.0)
            } else if step == 2 {
                box_step(ctx, &mut z, &mut dr, c, 1.0)
            } else if step == 3 {
                kifs_step(ctx, &mut z, &mut dr)
            } else if step == 4 {
                apollonian_step(ctx, &mut z, &mut dr)
            } else {
                true
            };
            if !ok {
                break;
            }
            if trap_mode != 0 {
                trap = trap.min(trap3(ctx, z, trap_mode));
            }
            i += 1;
        }
        fraction_blend(&fraction, 0.5 * length(z) / dr, trap)
    }

    /// Scene point -> object space: R^T (p - offset) / scale.
    #[inline(always)]
    fn object_point(ctx: Context<'_>, x: V3) -> V3 {
        if !ctx.world {
            return mul(
                mat3(ctx, P_OBJ_AXES, sub(x, pv3(ctx, P_OBJ_OFFSET))),
                1.0 / pr(ctx, P_OBJ_SCALE),
            );
        }
        let b = ctx.object * OBJECT_STRIDE + O_INVERSE;
        inverse_affine(&ctx.objects[b..b + 12], x)
    }

    /// A world object's inverse affine (row-major 3x4) applied to `x`. Out of line so every
    /// route (specialized, mixed-world) contracts this stencil into the same FMAs: the oracle
    /// compares the routes bit for bit. It takes only the 12 matrix values, so the call costs
    /// a pointer, not the whole `Context`.
    #[inline(never)]
    fn inverse_affine(m: &[f32], x: V3) -> V3 {
        [
            m[0] * x[0] + m[1] * x[1] + m[2] * x[2] + m[3],
            m[4] * x[0] + m[5] * x[1] + m[6] * x[2] + m[7],
            m[8] * x[0] + m[9] * x[1] + m[10] * x[2] + m[11],
        ]
    }

    /// The largest integer not above `v` (|v| < 2^31): the device has no `f32::floor`.
    #[inline(always)]
    fn floor(v: f32) -> f32 {
        let i = v as i32 as f32;
        if i > v { i - 1.0 } else { i }
    }

    /// Whether a Cells mask removes the cell of hash `h`: its top 24 bits as a unit value fall
    /// below `density`. A cell's hash is `lowbias32(x ^ lowbias32(y ^ lowbias32(z ^ seed)))`.
    #[inline(always)]
    fn cell_removed(h: u32, density: f32) -> bool {
        ((h >> 8) as f32 * (1.0 / 16_777_216.0)) < density
    }

    /// A scene mask's distance at world point `x` (a lower bound in world units, positive where
    /// the mask removes): the shape's signed distance in mask space, scaled conservatively, and
    /// negated when inverted. Cells: the edge `size.x` grid, a cell removed when its hash falls
    /// below the density. The distance is to the nearest of the 26 neighbour cells in the other
    /// state (exact within the 3x3x3 block), else the block's boundary as the lower bound: the
    /// zero set is exactly the border between removed and kept cells, never a face two cells of
    /// one state share.
    #[inline(never)]
    fn mask_distance(m: &[f32], x: V3) -> f32 {
        let p = inverse_affine(&m[M_INVERSE..M_INVERSE + 12], x);
        let s = [m[M_SIZE], m[M_SIZE + 1], m[M_SIZE + 2]];
        let local = match m[M_SHAPE] as u32 {
            MASK_SPHERE => length(p) - s[0],
            MASK_BOX => {
                let q = [p[0].abs() - s[0], p[1].abs() - s[1], p[2].abs() - s[2]];
                length([q[0].max(0.0), q[1].max(0.0), q[2].max(0.0)]) + max3(q).min(0.0)
            }
            _ => {
                let edge = s[0];
                let f = [p[0] / edge, p[1] / edge, p[2] / edge];
                let cell = [floor(f[0]), floor(f[1]), floor(f[2])];
                let t = [f[0] - cell[0], f[1] - cell[1], f[2] - cell[2]];
                let c = [cell[0] as i32, cell[1] as i32, cell[2] as i32];
                let (seed, density) = (m[M_SEED] as u32, m[M_DENSITY]);
                // The hash's inner levels of the 3x3x3 block, shared by its cells: z (3), then
                // y and z (9); each cell then costs one outer hash, and only when it is near
                // enough to matter.
                use standard_surface_bsdf::sampling::lowbias32;
                let mut zy = [0u32; 9];
                let mut k = 0;
                while k < 3 {
                    let z = lowbias32((c[2] + k as i32 - 1) as u32 ^ seed);
                    let mut j = 0;
                    while j < 3 {
                        zy[j * 3 + k] = lowbias32((c[1] + j as i32 - 1) as u32 ^ z);
                        j += 1;
                    }
                    k += 1;
                }
                let removed = cell_removed(lowbias32(c[0] as u32 ^ zy[4]), density);
                // Per axis, the distance to the neighbour at offset -1 / +1 (0 for offset 0).
                let gap = |k: usize, o: i32| match o {
                    -1 => t[k],
                    1 => 1.0 - t[k],
                    _ => 0.0,
                };
                let mut best = 1.0
                    + t[0]
                        .min(1.0 - t[0])
                        .min(t[1].min(1.0 - t[1]))
                        .min(t[2].min(1.0 - t[2]));
                let mut n = 0;
                while n < 27 {
                    let o = [n / 9 - 1, (n / 3) % 3 - 1, n % 3 - 1];
                    // A cell no nearer than the best so far cannot lower it: its hash is skipped.
                    let near = length([gap(0, o[0]), gap(1, o[1]), gap(2, o[2])]);
                    if n != 13
                        && near < best
                        && cell_removed(
                            lowbias32((c[0] + o[0]) as u32 ^ zy[(n % 9) as usize]),
                            density,
                        ) != removed
                    {
                        best = near;
                    }
                    n += 1;
                }
                if removed { edge * best } else { -edge * best }
            }
        };
        let d = local * m[M_SCALE];
        if m[M_INVERT] != 0.0 { -d } else { d }
    }

    /// `d` intersected with every mask of the current world object.
    #[inline(always)]
    fn masked(ctx: Context<'_>, x: V3, d: f32) -> f32 {
        let row = ctx.object * OBJECT_STRIDE;
        let count = ctx.objects[row + O_MASK_COUNT] as usize;
        let mut d = d;
        let mut i = 0;
        while i < count {
            let m = &ctx.objects
                [row + O_MASKS + i * MASK_STRIDE..row + O_MASKS + (i + 1) * MASK_STRIDE];
            // A Cells distance never exceeds 1.5 cells (`mask_distance`): where the estimate is
            // already that far, `max` keeps it and the mask need not be evaluated.
            if !(m[M_SHAPE] as u32 == MASK_CELLS && d >= 1.5 * m[M_SIZE] * m[M_SCALE]) {
                d = d.max(mask_distance(m, x));
            }
            i += 1;
        }
        d
    }

    /// The estimate of formula `family` at its own point `q` (object space, before the
    /// object's clip and scale).
    #[inline(always)]
    fn family_distance(ctx: Context<'_>, family: u32, q: V3, trap_mode: u32) -> (f32, f32) {
        match family {
            FAMILY_BULB => bulb_distance(ctx, q, trap_mode),
            FAMILY_BOX => box_distance(ctx, q, trap_mode),
            FAMILY_QUAT => quat_distance(ctx, q, trap_mode),
            FAMILY_KIFS => kifs_distance(ctx, q, trap_mode),
            FAMILY_KLEINIAN => kleinian_distance(ctx, q, trap_mode),
            FAMILY_PSEUDO_KLEINIAN => pseudo_kleinian_distance(ctx, q, trap_mode),
            FAMILY_APOLLONIAN => apollonian_distance(ctx, q, trap_mode),
            _ => hybrid_distance(ctx, q, trap_mode),
        }
    }

    /// `d` intersected with the formula's own ball (`P_BOUND_RADIUS`, zero for none) around its
    /// point `p`.
    #[inline(always)]
    fn formula_bound(ctx: Context<'_>, p: V3, d: f32) -> f32 {
        let radius = pr(ctx, P_BOUND_RADIUS);
        if radius > 0.0 {
            d.max(length(p) - radius)
        } else {
            d
        }
    }

    /// The estimate of `family` at object point `q` behind the spherical inversion
    /// pre-transform (`P_INVERT_*`, `fractal_core::inversion`): the formula and its own ball
    /// are evaluated at the transformed point, and the distance is divided by the map's
    /// largest local stretch (the first-order correction derived there).
    ///
    /// Out of line, and taken only at a nonzero strength: the inverted route keeps one shared
    /// copy of the eight estimates, and the uninverted route's code stays exactly the former
    /// (measured: routing it through a shared wrapper changed its inlining and float
    /// contractions, and so the bits of every Mandelbulb render).
    #[inline(never)]
    fn inverted_estimate(ctx: Context<'_>, q: V3, family: u32, trap_mode: u32) -> (f32, f32) {
        let (p, stretch) = fractal_core::inversion::apply(
            q,
            pv3(ctx, P_INVERT_CENTER),
            pr(ctx, P_INVERT_RADIUS),
            pr(ctx, P_INVERT_STRENGTH),
        );
        let (d, trap) = family_distance(ctx, family, p, trap_mode);
        (formula_bound(ctx, p, d) / stretch, trap)
    }

    /// The object-space estimate `d` at object point `q` (scene point `x`) clipped by the
    /// object's sphere, then scaled to the scene (or world) and intersected with the masks.
    #[inline(always)]
    fn clipped_distance_and_trap(
        ctx: Context<'_>,
        x: V3,
        q: V3,
        d: f32,
        trap: f32,
        world_clip: bool,
    ) -> (f32, f32) {
        // Marching and every normal stencil must sample the same clipped field.
        let d = if world_clip {
            d.max(length(q) - ctx.objects[ctx.object * OBJECT_STRIDE + O_CLIP_RADIUS])
        } else {
            d
        };
        let d = if ctx.world {
            // Scene masks are world shapes: they intersect the world-scaled estimate.
            masked(
                ctx,
                x,
                d * ctx.objects[ctx.object * OBJECT_STRIDE + O_DISTANCE_SCALE],
            )
        } else {
            d * pr(ctx, P_OBJ_SCALE)
        };
        (d, trap)
    }

    #[inline(always)]
    fn bounded_distance_and_trap(
        ctx: Context<'_>,
        x: V3,
        q: V3,
        d: f32,
        trap: f32,
        world_clip: bool,
    ) -> (f32, f32) {
        clipped_distance_and_trap(ctx, x, q, formula_bound(ctx, q, d), trap, world_clip)
    }

    /// The scene estimate of family `family` at scene point `x`, through the inversion when its
    /// strength is nonzero.
    #[inline(always)]
    fn inverted_scene_estimate(
        ctx: Context<'_>,
        x: V3,
        family: u32,
        trap_mode: u32,
        world_clip: bool,
    ) -> (f32, f32) {
        let q = object_point(ctx, x);
        let (d, trap) = inverted_estimate(ctx, q, family, trap_mode);
        clipped_distance_and_trap(ctx, x, q, d, trap, world_clip)
    }

    /// Share transformed Mandelbulb arithmetic across the runtime and specialized routes.
    /// Inlined: the hot march pays no call; `inverse_affine` alone keeps a call boundary.
    #[inline(always)]
    fn bulb_estimate(ctx: Context<'_>, x: V3, trap_mode: u32, world_clip: bool) -> (f32, f32) {
        if pr(ctx, P_INVERT_STRENGTH) != 0.0 {
            return inverted_scene_estimate(ctx, x, FAMILY_BULB, trap_mode, world_clip);
        }
        let q = object_point(ctx, x);
        let (d, trap) = bulb_distance(ctx, q, trap_mode);
        bounded_distance_and_trap(ctx, x, q, d, trap, world_clip)
    }

    /// Signed transformed estimate and trap (fractal3d.wgsl signed_distance_and_trap).
    #[inline(always)]
    fn signed_distance_and_trap<const F: u32>(
        ctx: Context<'_>,
        x: V3,
        trap_mode: u32,
    ) -> (f32, f32) {
        let family = if F == FAMILY_WORLD {
            pr(ctx, P_FAMILY) as u32
        } else {
            F
        };
        let world_clip = F == FAMILY_WORLD || ctx.world;
        if family == FAMILY_BULB {
            return bulb_estimate(ctx, x, trap_mode, world_clip);
        }
        // The world route checks the inversion before its dispatch (`world_estimate`).
        if F != FAMILY_WORLD && pr(ctx, P_INVERT_STRENGTH) != 0.0 {
            return inverted_scene_estimate(ctx, x, family, trap_mode, world_clip);
        }
        let q = object_point(ctx, x);
        let (d, trap) = match family {
            FAMILY_BOX => box_distance(ctx, q, trap_mode),
            FAMILY_QUAT => quat_distance(ctx, q, trap_mode),
            FAMILY_KIFS => kifs_distance(ctx, q, trap_mode),
            FAMILY_KLEINIAN => kleinian_distance(ctx, q, trap_mode),
            FAMILY_PSEUDO_KLEINIAN => pseudo_kleinian_distance(ctx, q, trap_mode),
            FAMILY_APOLLONIAN => apollonian_distance(ctx, q, trap_mode),
            _ => hybrid_distance(ctx, q, trap_mode),
        };
        bounded_distance_and_trap(ctx, x, q, d, trap, world_clip)
    }

    #[inline(always)]
    fn family_signed<const F: u32>(ctx: Context<'_>) -> bool {
        let f = if F == FAMILY_WORLD {
            pr(ctx, P_FAMILY) as u32
        } else {
            F
        };
        f == FAMILY_KIFS || f == FAMILY_KLEINIAN || f == FAMILY_PSEUDO_KLEINIAN
    }

    // Keep the runtime family dispatcher out of line. Inlining eight DEs into
    // primary/shadow marches and four normal stencils makes cold driver JIT very costly.
    #[inline(never)]
    fn world_estimate(ctx: Context<'_>, x: V3, trap_mode: u32) -> (f32, f32) {
        if pr(ctx, P_INVERT_STRENGTH) != 0.0 {
            return inverted_scene_estimate(ctx, x, pr(ctx, P_FAMILY) as u32, trap_mode, true);
        }
        signed_distance_and_trap::<FAMILY_WORLD>(ctx, x, trap_mode)
    }

    #[inline(always)]
    fn scene_signed_distance<const F: u32>(ctx: Context<'_>, x: V3) -> f32 {
        if F == FAMILY_WORLD {
            world_estimate(ctx, x, 0).0
        } else {
            signed_distance_and_trap::<F>(ctx, x, 0).0
        }
    }

    // =========================================================================
    // march (fractal3d.wgsl march)
    // =========================================================================

    const RAY_PRIMARY: u32 = 0;
    const RAY_BOUNCE: u32 = 1;
    /// The OFX Direct preview's camera ray: a primary ray whose unresolved result may stand in
    /// for the surface (Keinert, see `trace_direct`).
    const RAY_DIRECT: u32 = 3;
    /// The longest sphere-tracing step is `2 (t_exit - t_enter) / STEP_CAP_STEPS`, a fixed share
    /// of the ray's interval, independent of the step budget (Mandelbulber's step limits are
    /// too). It used to divide by `max_steps`, so a larger budget also shortened every step.
    /// Measured on BUG1 frame 27 (PLAN.md, "Findings worth keeping", phantom hits): with a budget large enough to converge,
    /// divisors 256 and 8192 give 69 531 and 69 537 hits of 147 456 rays - the cap barely
    /// matters once rays converge; 256 is the former default's cap.
    const STEP_CAP_STEPS: f32 = 256.0;

    /// The cone march's own step budget ([`cone_start`]). Independent of `P_MAX_STEPS` on purpose:
    /// the start is an optimization, so raising or lowering the quality budget must not move it
    /// (and with it every camera hit inside its footprint band). Steps are capped like the march's
    /// (`2 span / STEP_CAP_STEPS`), so this budget crosses the span twice; a march that runs out of
    /// it only starts lower, which is always safe.
    const CONE_START_STEPS: u32 = 256;
    const HIT_REFINEMENTS: u32 = 16;
    const FOOTPRINT_RELATIVE_FLOOR: f32 = 0.000_001_907_348_6;

    /// How a march ended. The march only reports; the integrator applies one policy per ray
    /// kind (`trace_path`, `trace_direct`).
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum Outcome {
        /// The ray reached the surface (within its footprint).
        Hit,
        /// The ray left its interval: nothing to hit along it.
        Miss,
        /// The step budget ran out inside the interval: not converged. `Hit::point` / `ratio`
        /// hold the closest sample, which is not a surface point (BUG1: 52% of such samples are
        /// rays passing the surface, 48% lie 0.3 units before the real hit).
        Unresolved,
    }

    #[derive(Clone, Copy)]
    pub struct Hit {
        outcome: Outcome,
        point: V3,
        trap: f32,
        eps: f32,
        // Each evaluated object owns one packed material, so this is also its material index.
        object: usize,
        /// `Unresolved`: the closest sample's estimate in footprints (`d / eps`).
        ratio: f32,
        /// `Hit` on the march's very first sample: the ray started in a hit, the case the leave
        /// restart retries ([`march_leaving`]).
        first: bool,
    }
    impl Hit {
        #[inline(always)]
        fn hit(&self) -> bool {
            self.outcome == Outcome::Hit
        }
    }

    /// One camera sample: its radiance, whether the camera ray hit, the primary guides
    /// (albedo, normal) and whether any march of the path (camera, bounce, visibility) was
    /// unresolved, i.e. ran out of steps.
    #[derive(Clone, Copy)]
    pub struct PathSample {
        radiance: V3,
        hit: bool,
        albedo: V3,
        normal: V3,
        unresolved: bool,
    }

    #[inline(always)]
    fn footprint(base: f32, slope: f32, t: f32, x: V3) -> f32 {
        (base + slope * t).max(length(x) * FOOTPRINT_RELATIVE_FLOOR)
    }

    #[inline(always)]
    fn march_sample<const F: u32>(ctx: Context<'_>, x: V3, trap_mode: u32) -> (f32, f32, usize) {
        if F != FAMILY_WORLD {
            let (d, t) = signed_distance_and_trap::<F>(ctx, x, trap_mode);
            return (d.max(0.0), t, 0);
        }
        let mut best = (f32::MAX, TRAP_START, 0usize);
        let mut object = 0usize;
        while object < global(P_OBJECT_COUNT) as usize {
            let c = Context { object, ..ctx };
            // Every world estimate is clipped by its object's sphere (`bounded_distance_and_trap`)
            // and scaled by a lower bound of the inverse transform's stretch, so the distance to
            // that sphere bounds the object's estimate from below. An object whose sphere is no
            // nearer than the best estimate so far cannot win: its full estimate is skipped, and
            // the result is the same, bit for bit.
            let row = object * OBJECT_STRIDE;
            let reach = (length(object_point(c, x)) - ctx.objects[row + O_CLIP_RADIUS])
                * ctx.objects[row + O_DISTANCE_SCALE];
            if reach >= best.0 {
                object += 1;
                continue;
            }
            let mode = if trap_mode == 0 {
                0
            } else {
                pr(c, P_COLOR_MODE) as u32
            };
            let (d, trap) = world_estimate(c, x, mode);
            let d = d.max(0.0);
            if d < best.0 {
                best = (d, trap, object);
            }
            object += 1;
        }
        best
    }

    /// The interval a ray marches over: `(t_enter, max_distance)`, or `None` when it misses the
    /// bounding sphere. Outside the sphere every estimate is an exact "outside" (escape radius /
    /// ball bound), so marching there only burns steps; Direct previews and OFX renders that
    /// disable clipping march `[0, P_MAX_DISTANCE]`. Shared by [`march`] and
    /// [`march_visibility`].
    #[inline(always)]
    fn march_span(ctx: Context<'_>, origin: V3, dir: V3) -> Option<(f32, f32)> {
        let max_distance = pr(ctx, P_MAX_DISTANCE);
        if global(P_DIRECT) != 0.0 || (global(P_OFX) != 0.0 && global(P_OFX_NO_CLIP) != 0.0) {
            return Some((0.0, max_distance));
        }
        let oc = sub(origin, pv3(ctx, P_CLIP_CENTER));
        let radius = pr(ctx, P_CLIP_RADIUS);
        let b = dot(oc, dir);
        let disc = b * b - (dot(oc, oc) - radius * radius);
        if disc <= 0.0 {
            return None;
        }
        let root = disc.sqrt();
        let t_exit = -b + root;
        if t_exit <= 0.0 {
            return None;
        }
        Some(((-b - root).max(0.0), t_exit.min(max_distance)))
    }

    /// The ray length below which no camera ray of the pixel with centre ray `dir` can hit: a
    /// cone march along the centre ray with the pixel cone `0.5 sqrt(w^2 + h^2)` of one pixel on
    /// the unit image plane, exact for an exact distance. A centre ray that misses the reach ball
    /// says nothing about the pixel's other rays: 0. Valid for a pinhole camera only (a lens ray
    /// starts off the cone's apex). OFX's CPU and WGSL twins: `fractal3d::cone_start`,
    /// `cone_start`.
    #[inline(always)]
    fn cone_start<const F: u32>(ctx: Context<'_>, origin: V3, dir: V3) -> f32 {
        let Some((t_enter, t_exit)) = march_span(ctx, origin, dir) else {
            return 0.0;
        };
        let step_cap = 2.0 * (t_exit - t_enter) / STEP_CAP_STEPS;
        let slope = pr(ctx, P_FOOTPRINT);
        let w = 2.0 * pr(ctx, P_HALF_W) / pr(ctx, P_WIDTH);
        let h = 2.0 * pr(ctx, P_HALF_H) / pr(ctx, P_HEIGHT);
        let cone = 0.5 * (w * w + h * h).sqrt();
        let mut t = t_enter;
        let mut i = 0u32;
        while i < CONE_START_STEPS {
            if t > t_exit {
                break;
            }
            let point = add(origin, mul(dir, t));
            let d = march_sample::<F>(ctx, point, 0).0;
            let eps = footprint(0.0, slope, t, point);
            let step = ((0.5 * d - cone * t - eps) / (1.0 + cone)).min(step_cap);
            if step <= 0.5 * eps {
                break;
            }
            t += step;
            i += 1;
        }
        t
    }

    /// A shadow ray's march: only whether the segment toward the light is blocked.
    ///
    /// The same steps, footprint and step cap as a secondary [`march`], but it carries no hit
    /// record: no orbit trap (estimates run with trap mode 0), no closest sample, no refinement
    /// bracket. A hit ends it at once; `Unresolved` means the step budget ran out inside the
    /// interval after at least one sample, exactly when [`march`] would report it. Keeping that
    /// state out of the hot loop leaves its registers to the distance estimate.
    #[inline(always)]
    fn march_visibility<const F: u32>(
        ctx: Context<'_>,
        origin: V3,
        dir: V3,
        base: f32,
        slope: f32,
    ) -> (Outcome, bool) {
        let max_steps = pr(ctx, P_SECONDARY_STEPS) as u32;
        // Secondary rays resolve a coarser surface (their footprint times P_SECONDARY_EPS).
        let eps_scale = pr(ctx, P_SECONDARY_EPS);
        let base = base * eps_scale;
        let slope = slope * eps_scale;
        let Some((t_enter, max_distance)) = march_span(ctx, origin, dir) else {
            return (Outcome::Miss, false);
        };
        let step_cap = 2.0 * (max_distance - t_enter) / STEP_CAP_STEPS;
        let step_factor = pr(ctx, P_STEP_FACTOR);
        let mut t = t_enter;
        let mut sampled = false;
        let mut i = 0u32;
        while i < max_steps {
            if t > max_distance {
                break;
            }
            let point = add(origin, mul(dir, t));
            let (d, _, _) = march_sample::<F>(ctx, point, 0);
            let eps = footprint(base, slope, t, point);
            if d < eps {
                return (Outcome::Hit, i == 0);
            }
            sampled = true;
            t += (step_factor * d).max(eps * 0.5).min(step_cap);
            i += 1;
        }
        let outcome = if sampled && t <= max_distance {
            Outcome::Unresolved
        } else {
            Outcome::Miss
        };
        (outcome, false)
    }

    /// The origin and restart point of the ray leaving the vertex at `point` (the march's hit
    /// point) with geometric normal `geo_n` (facing the arriving ray `dir_in`) along `dir_out`.
    /// The one rule of every engine (OFX's CPU `Traced::leave`, WGSL `geo_leave`): `eps` off the
    /// surface along the normal on the side `dir_out` leaves on; a ray on the normal's side
    /// restarts `eps` back along `dir_in`, a ray on the far side (into a medium) has no restart
    /// point (the origin again).
    #[inline(always)]
    fn leave_ray(point: V3, geo_n: V3, dir_in: V3, dir_out: V3, eps: f32) -> (V3, V3) {
        if dot(dir_out, geo_n) < 0.0 {
            let origin = sub(point, mul(geo_n, eps));
            return (origin, origin);
        }
        (add(point, mul(geo_n, eps)), sub(point, mul(dir_in, eps)))
    }

    /// [`march`] of a ray that left a surface at `origin`. Leave restart (OFX
    /// `fractal3d::march_leaving`): when the march starts in a hit (the offset origin is still
    /// inside a fold of the set) and the ray has a restart point (`leave_ray`), it is marched
    /// again from `retreat`, once. One call site of [`march`], so its loop keeps no restart state.
    #[inline(always)]
    fn march_leaving<const F: u32>(
        ctx: Context<'_>,
        origin: V3,
        retreat: V3,
        dir: V3,
        kind: u32,
        base: f32,
        slope: f32,
        start: f32,
    ) -> Hit {
        let m = march::<F>(ctx, origin, dir, kind, base, slope, start);
        if retreat != origin && m.hit() && m.first {
            return march::<F>(ctx, retreat, dir, kind, base, slope, start);
        }
        m
    }

    /// [`march_visibility`] of a shadow ray leaving a surface, with the same leave restart as
    /// [`march_leaving`].
    #[inline(always)]
    fn visibility_leaving<const F: u32>(
        ctx: Context<'_>,
        origin: V3,
        retreat: V3,
        dir: V3,
        base: f32,
        slope: f32,
    ) -> Outcome {
        let (outcome, first) = march_visibility::<F>(ctx, origin, dir, base, slope);
        if retreat != origin && outcome == Outcome::Hit && first {
            return march_visibility::<F>(ctx, retreat, dir, base, slope).0;
        }
        outcome
    }

    /// Sphere-trace from `origin` along `dir` and report how it ended (`Outcome`); the caller
    /// decides what an unresolved march means for its ray. Shadow rays use
    /// [`march_visibility`]. `start` is a camera ray's pixel-cone start ([`cone_start`]; 0 for
    /// every other ray): the march begins at `max(t_enter, start)`.
    #[inline(always)]
    fn march<const F: u32>(
        ctx: Context<'_>,
        origin: V3,
        dir: V3,
        kind: u32,
        base: f32,
        slope: f32,
        start: f32,
    ) -> Hit {
        let miss = Hit {
            outcome: Outcome::Miss,
            point: origin,
            trap: TRAP_START,
            eps: 0.0,
            object: 0,
            ratio: 0.0,
            first: false,
        };
        let primary = kind == RAY_PRIMARY || kind == RAY_DIRECT;
        let max_steps = if primary {
            pr(ctx, P_MAX_STEPS)
        } else {
            pr(ctx, P_SECONDARY_STEPS)
        } as u32;
        // Secondary rays resolve a coarser surface (their footprint times P_SECONDARY_EPS).
        let eps_scale = if primary {
            1.0
        } else {
            pr(ctx, P_SECONDARY_EPS)
        };
        let base = base * eps_scale;
        let slope = slope * eps_scale;
        let Some((t_enter, max_distance)) = march_span(ctx, origin, dir) else {
            return miss;
        };
        let trap_mode = if F == FAMILY_WORLD {
            1
        } else {
            pr(ctx, P_COLOR_MODE) as u32
        };
        let step_cap = 2.0 * (max_distance - t_enter) / STEP_CAP_STEPS;
        let step_factor = pr(ctx, P_STEP_FACTOR);
        let sample = |point: V3, trap: f32, eps: f32, object: usize, ratio: f32| Hit {
            outcome: Outcome::Hit,
            point,
            trap,
            eps,
            object,
            ratio,
            first: false,
        };
        let mut t = t_enter.max(start);
        // The last sample outside the surface and its ray length (the refinement bracket).
        let mut outside: Option<(Hit, f32)> = None;
        let mut closest: Option<Hit> = None;
        let mut i = 0u32;
        while i < max_steps {
            if t > max_distance {
                break;
            }
            let point = add(origin, mul(dir, t));
            let (d, trap, object) = march_sample::<F>(ctx, point, trap_mode);
            // Start guard (OFX `fractal3d::march_with`, the rule of every engine): a first sample
            // inside the set at a start beyond `t_enter` means the start overshot; restart from
            // `t_enter`. It covers estimates that read `d <= 0` inside the set.
            if d <= 0.0 && outside.is_none() && t > t_enter {
                t = t_enter;
                i += 1;
                continue;
            }
            let eps = footprint(base, slope, t, point);
            let hit = if primary { d <= eps } else { d < eps };
            if hit {
                let bracket = match outside {
                    Some(bracket) if d <= 0.0 => bracket,
                    _ => {
                        return Hit {
                            first: i == 0,
                            ..sample(point, trap, eps, object, d / eps)
                        };
                    }
                };
                // Stepped inside: bisect the bracket from the last outside sample.
                let (mut best, mut lo) = bracket;
                let mut hi = t;
                let mut k = 0u32;
                while k < HIT_REFINEMENTS {
                    if hi - lo <= 0.5 * best.eps {
                        break;
                    }
                    let mid = 0.5 * (lo + hi);
                    let inner = add(origin, mul(dir, mid));
                    let (rd, rt, object) = march_sample::<F>(ctx, inner, trap_mode);
                    if rd > 0.0 {
                        lo = mid;
                        let eps = footprint(base, slope, mid, inner);
                        best = sample(inner, rt, eps, object, rd / eps);
                    } else {
                        hi = mid;
                    }
                    k += 1;
                }
                return best;
            }
            let here = sample(point, trap, eps, object, d / eps);
            if closest.is_none_or(|c| here.ratio < c.ratio) {
                closest = Some(here);
            }
            outside = Some((here, t));
            t += (step_factor * d).max(eps * 0.5).min(step_cap);
            i += 1;
        }
        match closest {
            Some(closest) if t <= max_distance => Hit {
                outcome: Outcome::Unresolved,
                ..closest
            },
            _ => miss,
        }
    }

    /// Trace the exit of the occupied dielectric using its own field, not the union.
    /// An exit that cannot be found is `Unresolved`: absorbed by the integrator (never a sky
    /// miss) and counted like any march out of steps.
    /// The transformed local sphere bounds the probe interval even with nonuniform TRS.
    #[inline(always)]
    fn march_exit<const F: u32>(ctx: Context<'_>, origin: V3, dir: V3, eps: f32) -> Hit {
        let q = object_point(ctx, origin);
        let v = sub(object_point(ctx, add(origin, dir)), q);
        let radius = if ctx.world {
            ctx.objects[ctx.object * OBJECT_STRIDE + O_CLIP_RADIUS]
        } else {
            pr(ctx, P_CLIP_RADIUS) / pr(ctx, P_OBJ_SCALE)
        };
        let a = dot(v, v);
        let b = dot(q, v);
        let disc = b * b - a * (dot(q, q) - radius * radius);
        let bound = if disc > 0.0 && a > 0.0 {
            ((-b + disc.sqrt()) / a).max(0.0) + 2.0 * eps
        } else {
            2.0 * eps
        };
        let extent = bound.min(pr(ctx, P_MAX_DISTANCE));
        let distance = exit_distance(
            |t| {
                let point = add(origin, mul(dir, t));
                if F == FAMILY_WORLD {
                    world_estimate(ctx, point, 0).0
                } else {
                    signed_distance_and_trap::<F>(ctx, point, 0).0
                }
            },
            extent,
            eps,
            // A resolution of its own (`Render::glass_probes`), not the march's step budget.
            pr(ctx, P_GLASS_PROBES) as u32,
            pr(ctx, P_STEP_FACTOR),
            family_signed::<F>(ctx),
        );
        // No exit found (probe count spent, or a non-finite estimate): unresolved, so the
        // integrator absorbs the path and the status bar counts it.
        let Some(t) = distance else {
            return Hit {
                outcome: Outcome::Unresolved,
                point: origin,
                trap: TRAP_START,
                eps,
                object: ctx.object,
                ratio: 0.0,
                first: false,
            };
        };
        let point = add(origin, mul(dir, t));
        let trap_mode = pr(ctx, P_COLOR_MODE) as u32;
        let trap = if F == FAMILY_WORLD {
            world_estimate(ctx, point, trap_mode).1
        } else {
            signed_distance_and_trap::<F>(ctx, point, trap_mode).1
        };
        Hit {
            outcome: Outcome::Hit,
            point,
            trap,
            eps,
            object: ctx.object,
            ratio: 0.0,
            first: false,
        }
    }

    // =========================================================================
    // normals: central differences, with exterior step halving
    // =========================================================================

    const NORMAL_STEP_HALVINGS: u32 = 3;

    #[inline(always)]
    fn surface_normal<const F: u32>(ctx: Context<'_>, point: V3, dir: V3, eps: f32) -> V3 {
        // For world objects the stencil samples f(A^-1 x). Its world gradient is
        // A^-T grad(f), so nonuniform scale and parent shear transform normals correctly.
        let signed = family_signed::<F>(ctx);
        let center = if signed {
            point
        } else {
            sub(point, mul(dir, eps))
        };
        // Opposite axis samples cancel the mixed-derivative bias of a tetrahedral
        // stencil. Keep the probes within a quarter of the hit footprint to resolve
        // curved DE fields without sampling distant folds.
        let mut h = 0.25 * eps;
        let mut k = 0u32;
        while k <= NORMAL_STEP_HALVINGS {
            let xp = scene_signed_distance::<F>(ctx, add(center, [h, 0.0, 0.0]));
            let xm = scene_signed_distance::<F>(ctx, sub(center, [h, 0.0, 0.0]));
            let yp = scene_signed_distance::<F>(ctx, add(center, [0.0, h, 0.0]));
            let ym = scene_signed_distance::<F>(ctx, sub(center, [0.0, h, 0.0]));
            let zp = scene_signed_distance::<F>(ctx, add(center, [0.0, 0.0, h]));
            let zm = scene_signed_distance::<F>(ctx, sub(center, [0.0, 0.0, h]));
            if signed || (xp > 0.0 && xm > 0.0 && yp > 0.0 && ym > 0.0 && zp > 0.0 && zm > 0.0) {
                let g = [xp - xm, yp - ym, zp - zm];
                // Normalize by the largest component first: tiny but nonzero
                // DE gradients retain their direction without an absolute cutoff.
                let scale = g[0].abs().max(g[1].abs()).max(g[2].abs());
                // Every component finite, as OFX's CPU and WGSL (`max` ignores a NaN, so
                // `scale` alone would pass one through to `normalize`).
                if scale > 0.0
                    && g[0].abs() <= f32::MAX
                    && g[1].abs() <= f32::MAX
                    && g[2].abs() <= f32::MAX
                {
                    return normalize([g[0] / scale, g[1] / scale, g[2] / scale]);
                }
            }
            h *= 0.5;
            k += 1;
        }
        neg(dir)
    }

    // =========================================================================
    // colour (fractal3d.wgsl palette_color / hit_palette)
    // =========================================================================

    #[inline(always)]
    fn palette_color(ctx: Context<'_>, lut: &[[f32; 4]], t: f32) -> V3 {
        let index = t.clamp(0.0, 1.0) * (PALETTE_SAMPLES - 1) as f32;
        let low = index as usize;
        let high = (low + 1).min(PALETTE_SAMPLES - 1);
        let blend = index - low as f32;
        // SAFETY: low, high <= PALETTE_SAMPLES - 1 and the LUT holds PALETTE_SAMPLES + 1 rows.
        let offset = if ctx.world {
            (ctx.object + 1) * (PALETTE_SAMPLES + 1)
        } else {
            0
        };
        let (a, b) = unsafe {
            (
                *lut.get_unchecked(offset + low),
                *lut.get_unchecked(offset + high),
            )
        };
        [
            a[0] + (b[0] - a[0]) * blend,
            a[1] + (b[1] - a[1]) * blend,
            a[2] + (b[2] - a[2]) * blend,
        ]
    }

    #[inline(always)]
    fn hit_palette(ctx: Context<'_>, lut: &[[f32; 4]], point: V3, n: V3, trap: f32) -> V3 {
        let position = if pr(ctx, P_COLOR_MODE) == 0.0 {
            (0.4 + 0.1 * length(object_point(ctx, point)) + 0.18 * n[1]).clamp(0.0, 1.0)
        } else {
            (1.0 - (-trap * pr(ctx, P_TRAP_SCALE)).exp()).clamp(0.0, 1.0)
        };
        palette_color(ctx, lut, position)
    }

    // =========================================================================
    // lighting: sun cone + gradient sky (pathtrace.wgsl SunSky)
    // =========================================================================

    #[inline(always)]
    fn sky_radiance(ctx: Context<'_>, d: V3) -> V3 {
        let t = (0.5 + 0.5 * d[1]).clamp(0.0, 1.0);
        let horizon = pv3(ctx, P_SKY_HORIZON);
        let zenith = pv3(ctx, P_SKY_ZENITH);
        mul(
            add(horizon, mul(sub(zenith, horizon), t)),
            pr(ctx, P_SKY_INTENSITY),
        )
    }

    // =========================================================================
    // HDRI environment (ofx-fractal environment.rs; the CDF scheme of pt-integrator's EquirectMap and
    // its shared equirect.wgsl, statement for statement). `lut` after the palette rows holds
    // EnvironmentMap::gpu_entries: level-0 texels with the row-conditional CDF in .w, the marginal CDF
    // in .x, the 9 spherical-harmonic coefficients of the irradiance over pi, then mip levels 1..
    // P_ENV_WIDTH 0 selects the sky gradient.
    // =========================================================================

    #[inline(always)]
    fn env_on(ctx: Context<'_>) -> bool {
        pr(ctx, P_ENV_WIDTH) > 0.0
    }

    #[inline(always)]
    fn env_entry(ctx: Context<'_>, lut: &[[f32; 4]], i: usize) -> [f32; 4] {
        // SAFETY: the host appends exactly the map's block after the palette rows (`cuda::render`
        // validates its length against P_ENV_WIDTH / P_ENV_HEIGHT / P_ENV_LEVELS), and every caller
        // indexes inside the level, CDF or coefficient range of that block.
        unsafe {
            *lut.get_unchecked(
                (1 + if ctx.world {
                    global(P_OBJECT_COUNT) as usize
                } else {
                    0
                }) * (PALETTE_SAMPLES + 1)
                    + i,
            )
        }
    }

    /// environment.rs Environment3d::env_to_world as pt-integrator's mat_vec: world into the map.
    #[inline(always)]
    fn env_to_map(ctx: Context<'_>, d: V3) -> V3 {
        let c = pr(ctx, P_ENV_COS);
        let s = pr(ctx, P_ENV_SIN);
        [c * d[0] - s * d[2], d[1], s * d[0] + c * d[2]]
    }

    /// Its transpose (pt-integrator's vec_mat): the map into the world.
    #[inline(always)]
    fn env_to_world(ctx: Context<'_>, e: V3) -> V3 {
        let c = pr(ctx, P_ENV_COS);
        let s = pr(ctx, P_ENV_SIN);
        [c * e[0] + s * e[2], e[1], -s * e[0] + c * e[2]]
    }

    /// pt-integrator dir_to_equirect_uv: u from atan2(z, x), v 0 at the zenith +Y.
    #[inline(always)]
    fn equirect_uv(d: V3) -> [f32; 2] {
        let theta = d[2].atan2(d[0]);
        let phi = d[1].clamp(-1.0, 1.0).asin();
        let mut u = theta / (2.0 * PI) + 0.5;
        let v = 0.5 - phi / PI;
        u -= u.floor();
        [u, v]
    }

    /// pt-integrator equirect_uv_to_dir.
    #[inline(always)]
    fn equirect_dir(uv: [f32; 2]) -> V3 {
        let (sin_theta, cos_theta) = sin_cos((uv[0] - 0.5) * 2.0 * PI);
        let (sin_phi, cos_phi) = sin_cos((0.5 - uv[1]) * PI);
        [cos_phi * cos_theta, sin_phi, cos_phi * sin_theta]
    }

    /// First entry, width and height of mip level `level` (environment.rs mips: each level halves
    /// rounding up; levels 1.. follow the coefficients).
    #[inline(always)]
    fn env_level(ctx: Context<'_>, level: u32) -> (usize, usize, usize) {
        let mut w = pr(ctx, P_ENV_WIDTH) as usize;
        let mut h = pr(ctx, P_ENV_HEIGHT) as usize;
        if level == 0 {
            return (0, w, h);
        }
        let mut at = w * h + h + 9;
        w = w.div_ceil(2);
        h = h.div_ceil(2);
        let mut l = 1u32;
        while l < level {
            at += w * h;
            w = w.div_ceil(2);
            h = h.div_ceil(2);
            l += 1;
        }
        (at, w, h)
    }

    #[inline(always)]
    fn mix1(a: f32, b: f32, t: f32) -> f32 {
        a * (1.0 - t) + b * t
    }

    /// pt-integrator lighting::bilinear on mip level `level`: columns wrap, rows clamp.
    #[inline(always)]
    fn env_bilinear(ctx: Context<'_>, lut: &[[f32; 4]], level: u32, uv: [f32; 2]) -> V3 {
        let (at, w, h) = env_level(ctx, level);
        let (wi, hi) = (w as i32, h as i32);
        let texel = |x: i32, y: i32| {
            let column = x.rem_euclid(wi);
            let row = y.clamp(0, hi - 1);
            env_entry(ctx, lut, at + (row * wi + column) as usize)
        };
        let fx = uv[0] * w as f32 - 0.5;
        let fy = uv[1] * h as f32 - 0.5;
        let x0 = fx.floor();
        let y0 = fy.floor();
        let tx = fx - x0;
        let ty = fy - y0;
        let (x0, y0) = (x0 as i32, y0 as i32);
        let (a, b, c, d) = (
            texel(x0, y0),
            texel(x0 + 1, y0),
            texel(x0, y0 + 1),
            texel(x0 + 1, y0 + 1),
        );
        let mut color = [0.0; 3];
        let mut channel = 0;
        while channel < 3 {
            let top = mix1(a[channel], b[channel], tx);
            let bottom = mix1(c[channel], d[channel], tx);
            color[channel] = mix1(top, bottom, ty);
            channel += 1;
        }
        color
    }

    /// environment.rs MapSky::radiance: level 0 toward the world direction `d`, times the intensity.
    #[inline(always)]
    fn env_map_radiance(ctx: Context<'_>, lut: &[[f32; 4]], d: V3) -> V3 {
        mul(
            env_bilinear(ctx, lut, 0, equirect_uv(env_to_map(ctx, d))),
            pr(ctx, P_ENV_INTENSITY),
        )
    }

    /// The background of a camera miss: the HDRI or the sky gradient, never a sun disc
    /// (fractal3d.rs Sky::radiance).
    #[inline(always)]
    fn background(ctx: Context<'_>, lut: &[[f32; 4]], d: V3) -> V3 {
        if env_on(ctx) {
            env_map_radiance(ctx, lut, d)
        } else {
            sky_radiance(ctx, d)
        }
    }

    /// environment.rs EnvironmentMap::irradiance: irradiance over pi about the world normal `n` from
    /// the L2 spherical harmonics, times the intensity.
    #[cfg(feature = "direct")]
    #[inline(always)]
    fn env_irradiance(ctx: Context<'_>, lut: &[[f32; 4]], n: V3) -> V3 {
        let [x, y, z] = env_to_map(ctx, n);
        let basis = [
            0.282_094_8,
            0.488_602_5 * y,
            0.488_602_5 * z,
            0.488_602_5 * x,
            1.092_548_4 * x * y,
            1.092_548_4 * y * z,
            0.315_391_57 * (3.0 * z * z - 1.0),
            1.092_548_4 * x * z,
            0.546_274_2 * (x * x - y * y),
        ];
        let w = pr(ctx, P_ENV_WIDTH) as usize;
        let h = pr(ctx, P_ENV_HEIGHT) as usize;
        let at = w * h + h;
        let mut e = [0.0f32; 3];
        let mut i = 0;
        while i < 9 {
            let coefficient = env_entry(ctx, lut, at + i);
            let mut c = 0;
            while c < 3 {
                e[c] += coefficient[c] * basis[i];
                c += 1;
            }
            i += 1;
        }
        // Clamped at 0: the L2 sum of a sun-dominated map rings below 0 away from the sun
        // (environment.rs EnvironmentMap::irradiance).
        mul(
            [e[0].max(0.0), e[1].max(0.0), e[2].max(0.0)],
            pr(ctx, P_ENV_INTENSITY),
        )
    }

    /// environment.rs EnvironmentMap::specular: the map toward the world direction `d` filtered for a
    /// GGX lobe of `alpha` (the two levels around log2(max(alpha h / pi, 1)) blended), times the
    /// intensity.
    #[cfg(feature = "direct")]
    #[inline(always)]
    fn env_specular(ctx: Context<'_>, lut: &[[f32; 4]], d: V3, alpha: f32) -> V3 {
        let texels = (alpha * pr(ctx, P_ENV_HEIGHT) / PI).max(1.0);
        let lod = texels.log2().min(pr(ctx, P_ENV_LEVELS) - 1.0);
        let uv = equirect_uv(env_to_map(ctx, d));
        let base = lod.floor();
        let t = lod - base;
        let lower = env_bilinear(ctx, lut, base as u32, uv);
        let filtered = if t == 0.0 {
            lower
        } else {
            let upper = env_bilinear(ctx, lut, base as u32 + 1, uv);
            [
                lower[0] + (upper[0] - lower[0]) * t,
                lower[1] + (upper[1] - lower[1]) * t,
                lower[2] + (upper[2] - lower[2]) * t,
            ]
        };
        mul(filtered, pr(ctx, P_ENV_INTENSITY))
    }

    /// Direct's diffuse and sheen light (fractal3d.rs Sky::irradiance): the HDRI's harmonics or the
    /// gradient's exact convolution.
    #[cfg(feature = "direct")]
    #[inline(always)]
    fn sky_irradiance(ctx: Context<'_>, lut: &[[f32; 4]], n: V3) -> V3 {
        if env_on(ctx) {
            return env_irradiance(ctx, lut, n);
        }
        let t = 0.5 + n[1] / 3.0;
        mul(
            add(
                pv3(ctx, P_SKY_HORIZON),
                mul(sub(pv3(ctx, P_SKY_ZENITH), pv3(ctx, P_SKY_HORIZON)), t),
            ),
            global(P_SKY_INTENSITY),
        )
    }

    /// Direct's specular and coat light (fractal3d.rs Sky::specular): the HDRI's level for `alpha`
    /// or the gradient at `d`.
    #[cfg(feature = "direct")]
    #[inline(always)]
    fn sky_specular(ctx: Context<'_>, lut: &[[f32; 4]], d: V3, alpha: f32) -> V3 {
        if env_on(ctx) {
            env_specular(ctx, lut, d, alpha)
        } else {
            sky_radiance(ctx, d)
        }
    }

    #[inline(always)]
    fn env_marginal(ctx: Context<'_>, lut: &[[f32; 4]], i: usize) -> f32 {
        let w = pr(ctx, P_ENV_WIDTH) as usize;
        let h = pr(ctx, P_ENV_HEIGHT) as usize;
        env_entry(ctx, lut, w * h + i)[0]
    }

    #[inline(always)]
    fn env_conditional(ctx: Context<'_>, lut: &[[f32; 4]], i: usize) -> f32 {
        env_entry(ctx, lut, i)[3]
    }

    #[inline(always)]
    fn env_marginal_below(ctx: Context<'_>, lut: &[[f32; 4]], i: usize) -> f32 {
        if i == 0 {
            0.0
        } else {
            env_marginal(ctx, lut, i - 1)
        }
    }

    #[inline(always)]
    fn env_conditional_below(ctx: Context<'_>, lut: &[[f32; 4]], row: usize, x: usize) -> f32 {
        if x == 0 {
            0.0
        } else {
            env_conditional(ctx, lut, row + x - 1)
        }
    }

    /// pt-integrator step_offset: the position of `xi` inside the CDF step [lo, hi).
    #[inline(always)]
    fn env_step_offset(xi: f32, lo: f32, hi: f32) -> f32 {
        if hi > lo {
            ((xi - lo) / (hi - lo)).clamp(0.0, 1.0)
        } else {
            0.5
        }
    }

    /// pt-integrator EquirectMap::texel_pdf: the solid-angle density inside texel (x, y) at `v`.
    #[inline(always)]
    fn env_texel_pdf(ctx: Context<'_>, lut: &[[f32; 4]], x: usize, y: usize, v: f32) -> f32 {
        let w = pr(ctx, P_ENV_WIDTH) as usize;
        let h = pr(ctx, P_ENV_HEIGHT) as usize;
        let row = y * w;
        let marginal_pdf = env_marginal(ctx, lut, y) - env_marginal_below(ctx, lut, y);
        let conditional_pdf =
            env_conditional(ctx, lut, row + x) - env_conditional_below(ctx, lut, row, x);
        let sin_theta = sin_cos(PI * v).0;
        if sin_theta <= 0.0 {
            return 0.0;
        }
        (marginal_pdf * conditional_pdf).max(0.0) * w as f32 * h as f32
            / (2.0 * PI * PI * sin_theta)
    }

    /// pt-integrator EquirectMap::sample then Equirect's rotation: one world direction.
    #[inline(always)]
    fn env_map_sample(ctx: Context<'_>, lut: &[[f32; 4]], r1: f32, r2: f32) -> V3 {
        let w = pr(ctx, P_ENV_WIDTH) as usize;
        let h = pr(ctx, P_ENV_HEIGHT) as usize;
        // First row whose marginal CDF is not below r2.
        let (mut lo, mut hi) = (0usize, h);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if env_marginal(ctx, lut, mid) < r2 {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let y = lo.min(h - 1);
        let row = y * w;
        // First texel of the row whose conditional CDF is not below r1.
        let (mut lo, mut hi) = (0usize, w);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if env_conditional(ctx, lut, row + mid) < r1 {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let x = lo.min(w - 1);
        let dv = env_step_offset(
            r2,
            env_marginal_below(ctx, lut, y),
            env_marginal(ctx, lut, y),
        );
        let du = env_step_offset(
            r1,
            env_conditional_below(ctx, lut, row, x),
            env_conditional(ctx, lut, row + x),
        );
        let u = (x as f32 + du) / w as f32;
        let v = (y as f32 + dv) / h as f32;
        env_to_world(ctx, equirect_dir([u, v]))
    }

    /// pt-integrator Equirect::pdf: the density of env_map_sample for the world direction `d`.
    #[inline(always)]
    fn env_map_pdf(ctx: Context<'_>, lut: &[[f32; 4]], d: V3) -> f32 {
        let w = pr(ctx, P_ENV_WIDTH) as u32;
        let h = pr(ctx, P_ENV_HEIGHT) as u32;
        let uv = equirect_uv(env_to_map(ctx, d));
        let x = ((uv[0] * w as f32) as u32).min(w - 1) as usize;
        let y = ((uv[1] * h as f32) as u32).min(h - 1) as usize;
        env_texel_pdf(ctx, lut, x, y, uv[1])
    }

    #[inline(always)]
    fn light_slot(ctx: Context<'_>, light: usize, slot: usize) -> f32 {
        ctx.lights[light * LIGHT_STRIDE + slot - P_LIGHT_DIR]
    }
    #[inline(always)]
    fn light_v3(ctx: Context<'_>, light: usize, slot: usize) -> V3 {
        [
            light_slot(ctx, light, slot),
            light_slot(ctx, light, slot + 1),
            light_slot(ctx, light, slot + 2),
        ]
    }
    #[inline(always)]
    fn world_sun_contains(ctx: Context<'_>, light: usize, dir: V3) -> bool {
        let d = sub(dir, light_v3(ctx, light, P_LIGHT_DIR));
        0.5 * dot(d, d) <= light_slot(ctx, light, P_SUN_ONE_MINUS_COS)
    }
    #[inline(always)]
    fn world_sun_weight(ctx: Context<'_>, light: usize) -> f32 {
        luminance(mul(
            light_v3(ctx, light, P_LIGHT_COLOR),
            light_slot(ctx, light, P_LIGHT_INTENSITY),
        ))
        .max(0.0)
    }
    #[inline(always)]
    fn world_sky_weight(ctx: Context<'_>) -> f32 {
        let mean = if env_on(ctx) {
            pr(ctx, P_ENV_MEAN) * pr(ctx, P_ENV_INTENSITY)
        } else {
            luminance(mul(
                add(pv3(ctx, P_SKY_HORIZON), pv3(ctx, P_SKY_ZENITH)),
                0.5 * pr(ctx, P_SKY_INTENSITY),
            ))
        };
        (PI * mean).max(0.000001)
    }
    #[inline(always)]
    fn world_light_total(ctx: Context<'_>) -> f32 {
        let mut total = world_sky_weight(ctx);
        let mut i = 0usize;
        while i < global(P_LIGHT_COUNT) as usize {
            total += world_sun_weight(ctx, i);
            i += 1;
        }
        total
    }

    /// pathtrace.rs SunSky::p_sun: the gradient's sun_select, or the host's map_sun_select against
    /// the HDRI (P_ENV_SUN_SELECT, 0 for a sun of no light).
    #[inline(always)]
    fn sun_select(ctx: Context<'_>) -> f32 {
        if env_on(ctx) {
            return pr(ctx, P_ENV_SUN_SELECT);
        }
        let sun = luminance(mul(pv3(ctx, P_LIGHT_COLOR), pr(ctx, P_LIGHT_INTENSITY)));
        let sky = luminance(mul(
            add(pv3(ctx, P_SKY_HORIZON), pv3(ctx, P_SKY_ZENITH)),
            0.5 * pr(ctx, P_SKY_INTENSITY),
        ));
        let total = sun + PI * sky;
        let s = if total > 0.0 { sun / total } else { 0.5 };
        s.clamp(0.1, 0.9)
    }

    #[inline(always)]
    fn sun_contains(ctx: Context<'_>, wi: V3) -> bool {
        let d = sub(wi, pv3(ctx, P_LIGHT_DIR));
        0.5 * dot(d, d) <= pr(ctx, P_SUN_ONE_MINUS_COS)
    }

    /// pathtrace.rs SunSky::radiance: the gradient (the sun disc inside its cone, the sky elsewhere)
    /// or the HDRI (the map everywhere plus the sun disc inside its cone).
    #[inline(always)]
    fn env_radiance(ctx: Context<'_>, lut: &[[f32; 4]], dir: V3) -> V3 {
        if ctx.world {
            let mut radiance = background(ctx, lut, dir);
            let mut i = 0usize;
            while i < global(P_LIGHT_COUNT) as usize {
                if world_sun_contains(ctx, i, dir) {
                    radiance = add(
                        radiance,
                        mul(
                            light_v3(ctx, i, P_LIGHT_COLOR),
                            light_slot(ctx, i, P_LIGHT_INTENSITY)
                                * light_slot(ctx, i, P_SUN_CONE_PDF),
                        ),
                    );
                }
                i += 1;
            }
            return radiance;
        }

        if env_on(ctx) {
            let map = env_map_radiance(ctx, lut, dir);
            return if sun_contains(ctx, dir) {
                add(
                    map,
                    mul(
                        pv3(ctx, P_LIGHT_COLOR),
                        pr(ctx, P_LIGHT_INTENSITY) * pr(ctx, P_SUN_CONE_PDF),
                    ),
                )
            } else {
                map
            };
        }
        if sun_contains(ctx, dir) {
            return mul(
                pv3(ctx, P_LIGHT_COLOR),
                pr(ctx, P_LIGHT_INTENSITY) * pr(ctx, P_SUN_CONE_PDF),
            );
        }
        sky_radiance(ctx, dir)
    }

    /// pathtrace.rs SunSky::pdf: the sun-cone and sky mixture density.
    #[inline(always)]
    fn env_pdf(ctx: Context<'_>, lut: &[[f32; 4]], dir: V3) -> f32 {
        if ctx.world {
            let total = world_light_total(ctx);
            let mut pdf = world_sky_weight(ctx) / total
                * if env_on(ctx) {
                    env_map_pdf(ctx, lut, dir)
                } else {
                    0.079_577_47
                };
            let mut i = 0usize;
            while i < global(P_LIGHT_COUNT) as usize {
                if world_sun_contains(ctx, i, dir) {
                    pdf += world_sun_weight(ctx, i) / total * light_slot(ctx, i, P_SUN_CONE_PDF);
                }
                i += 1;
            }
            return pdf;
        }

        let s = sun_select(ctx);
        let mut pdf = (1.0 - s)
            * if env_on(ctx) {
                env_map_pdf(ctx, lut, dir)
            } else {
                0.079_577_47
            };
        if sun_contains(ctx, dir) {
            pdf += s * pr(ctx, P_SUN_CONE_PDF);
        }
        pdf
    }

    #[inline(always)]
    fn basis(n: V3) -> (V3, V3) {
        let s = if n[2] >= 0.0 { 1.0 } else { -1.0 };
        let a = -1.0 / (s + n[2]);
        let b = n[0] * n[1] * a;
        (
            [1.0 + s * n[0] * n[0] * a, s * b, -s * n[0]],
            [b, s + n[1] * n[1] * a, -n[1]],
        )
    }

    /// pathtrace.rs SunSky::sample: the sun cone with probability p, else the sky (uniform sphere or
    /// the HDRI's CDFs).
    #[inline(always)]
    fn env_sample(ctx: Context<'_>, lut: &[[f32; 4]], r1: f32, r2: f32) -> (V3, f32) {
        if ctx.world {
            let mut selector = r1 * world_light_total(ctx);
            let mut i = 0usize;
            while i < global(P_LIGHT_COUNT) as usize {
                let weight = world_sun_weight(ctx, i);
                if selector < weight {
                    let d = light_v3(ctx, i, P_LIGHT_DIR);
                    let (t, b) = basis(d);
                    let cos_t = 1.0 - (selector / weight) * light_slot(ctx, i, P_SUN_ONE_MINUS_COS);
                    let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
                    let (sp, cp) = sin_cos(2.0 * PI * r2);
                    let dir = add(add(mul(t, sin_t * cp), mul(b, sin_t * sp)), mul(d, cos_t));
                    return (dir, env_pdf(ctx, lut, dir));
                }
                selector -= weight;
                i += 1;
            }
            let u = (selector / world_sky_weight(ctx)).clamp(0.0, 0.99999994);
            let dir = if env_on(ctx) {
                env_map_sample(ctx, lut, u, r2)
            } else {
                let z = 1.0 - 2.0 * u;
                let radius = (1.0 - z * z).max(0.0).sqrt();
                let (sp, cp) = sin_cos(2.0 * PI * r2);
                [radius * cp, radius * sp, z]
            };
            return (dir, env_pdf(ctx, lut, dir));
        }

        let s = sun_select(ctx);
        let dir = if r1 < s {
            let d = pv3(ctx, P_LIGHT_DIR);
            let (t, b) = basis(d);
            let cos_t = 1.0 - (r1 / s) * pr(ctx, P_SUN_ONE_MINUS_COS);
            let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
            let (sp, cp) = sin_cos(2.0 * PI * r2);
            add(add(mul(t, sin_t * cp), mul(b, sin_t * sp)), mul(d, cos_t))
        } else if env_on(ctx) {
            env_map_sample(ctx, lut, (r1 - s) / (1.0 - s), r2)
        } else {
            let z = 1.0 - 2.0 * ((r1 - s) / (1.0 - s));
            let r = (1.0 - z * z).max(0.0).sqrt();
            let (sp, cp) = sin_cos(2.0 * PI * r2);
            [r * cp, r * sp, z]
        };
        (dir, env_pdf(ctx, lut, dir))
    }

    #[inline(always)]
    fn power_heuristic(a: f32, b: f32) -> f32 {
        let (a2, b2) = (a * a, b * b);
        if a2 + b2 > 0.0 { a2 / (a2 + b2) } else { 0.0 }
    }

    // =========================================================================
    // materials
    // =========================================================================

    /// The fast model: Lambert base + one GGX lobe (dielectric F0 0.04 * specular or the metal's
    /// base colour), Schlick Fresnel, height-correlated-free Smith G1 product.
    #[derive(Clone, Copy)]
    pub struct Fast {
        albedo: V3,
        f0: V3,
        alpha: f32,
        diffuse_w: f32,
        spec_prob: f32,
    }

    #[inline(always)]
    fn fast_material(ctx: Context<'_>, base_color: V3, roughness: f32, metal: f32) -> Fast {
        let base = had(mul(base_color, pr(ctx, P_BASE)), pv3(ctx, P_BASE_TINT));
        let ior = pr(ctx, P_SPECULAR_IOR);
        let f0d = ((ior - 1.0) / (ior + 1.0)) * ((ior - 1.0) / (ior + 1.0)) * pr(ctx, P_SPECULAR);
        let f0 = add(
            mul(
                had([f0d, f0d, f0d], pv3(ctx, P_SPECULAR_COLOR)),
                1.0 - metal,
            ),
            mul(base, metal),
        );
        let rough = roughness.max(0.03);
        let diffuse_w = 1.0 - metal;
        let lf = luminance(f0);
        let spec_prob = fast_spec_probability(lf, diffuse_w * luminance(base), diffuse_w);
        Fast {
            albedo: base,
            f0,
            alpha: rough * rough,
            diffuse_w,
            spec_prob,
        }
    }

    #[inline(always)]
    fn schlick(f0: V3, cos: f32) -> V3 {
        let m = (1.0 - cos).clamp(0.0, 1.0);
        let m5 = m * m * m * m * m;
        add(f0, mul(sub([1.0, 1.0, 1.0], f0), m5))
    }

    #[inline(always)]
    fn smith_g1(a2: f32, nv: f32) -> f32 {
        2.0 * nv / (nv + (a2 + (1.0 - a2) * nv * nv).sqrt())
    }

    /// (f cos, pdf)
    #[inline(always)]
    fn fast_eval(m: &Fast, n: V3, wo: V3, wi: V3) -> (V3, f32) {
        let nl = dot(n, wi);
        let nv = dot(n, wo);
        if nl <= 0.0 || nv <= 0.0 {
            return ([0.0; 3], 0.0);
        }
        let h = normalize(add(wo, wi));
        let nh = dot(n, h).max(0.0);
        let vh = dot(wo, h).max(1.0e-6);
        let a2 = m.alpha * m.alpha;
        let dd = nh * nh * (a2 - 1.0) + 1.0;
        let d = a2 / (PI * dd * dd);
        let g1v = smith_g1(a2, nv);
        let g = g1v * smith_g1(a2, nl);
        let f = schlick(m.f0, vh);
        let spec = mul(f, d * g / (4.0 * nv));
        let diff = mul(
            had(mul(sub([1.0, 1.0, 1.0], f), m.diffuse_w), m.albedo),
            nl / PI,
        );
        let pdf = m.spec_prob * fast_ggx_pdf(d, g1v, nv) + (1.0 - m.spec_prob) * nl / PI;
        (add(spec, diff), pdf)
    }

    /// (wi, weight, pdf, valid)
    #[inline(always)]
    fn fast_sample(m: &Fast, n: V3, wo: V3, u: V3) -> (V3, V3, f32, bool) {
        let (t, b) = basis(n);
        let wi = if u[0] < m.spec_prob {
            let local =
                fast_ggx_sample_local([dot(wo, t), dot(wo, b), dot(wo, n)], m.alpha, [u[1], u[2]]);
            add(add(mul(t, local[0]), mul(b, local[1])), mul(n, local[2]))
        } else {
            let r = u[1].sqrt();
            let (sp, cp) = sin_cos(2.0 * PI * u[2]);
            add(
                add(mul(t, r * cp), mul(b, r * sp)),
                mul(n, (1.0 - u[1]).max(0.0).sqrt()),
            )
        };
        let (f, pdf) = fast_eval(m, n, wo, wi);
        if pdf <= 0.0 {
            return (wi, [0.0; 3], 0.0, false);
        }
        (wi, mul(f, 1.0 / pdf), pdf, true)
    }

    /// The full model's inputs (fractal3d.wgsl surface_inputs): palette * base_tint as base colour;
    /// Transmission tint is applied at interfaces only when volume absorption is disabled.
    #[inline(always)]
    fn surface_inputs(
        ctx: Context<'_>,
        base_color: V3,
        roughness: f32,
        metal: f32,
    ) -> SurfaceInputs {
        SurfaceInputs {
            base: pr(ctx, P_BASE),
            base_color: had(base_color, pv3(ctx, P_BASE_TINT)),
            diffuse_roughness: pr(ctx, P_DIFFUSE_ROUGHNESS),
            metalness: metal,
            specular: pr(ctx, P_SPECULAR),
            specular_color: pv3(ctx, P_SPECULAR_COLOR),
            specular_roughness: roughness,
            specular_ior: pr(ctx, P_SPECULAR_IOR),
            specular_anisotropy: pr(ctx, P_SPECULAR_ANISOTROPY),
            specular_rotation: pr(ctx, P_SPECULAR_ROTATION),
            transmission: pr(ctx, P_TRANSMISSION),
            transmission_color: if pr(ctx, P_TRANSMISSION_DEPTH) > 0.0 {
                [1.0; 3]
            } else {
                pv3(ctx, P_TRANSMISSION_COLOR)
            },
            transmission_extra_roughness: pr(ctx, P_TRANSMISSION_EXTRA_ROUGHNESS),
            sheen: pr(ctx, P_SHEEN),
            sheen_color: pv3(ctx, P_SHEEN_COLOR),
            sheen_roughness: pr(ctx, P_SHEEN_ROUGHNESS),
            coat: pr(ctx, P_COAT),
            coat_color: pv3(ctx, P_COAT_COLOR),
            coat_roughness: pr(ctx, P_COAT_ROUGHNESS),
            coat_ior: pr(ctx, P_COAT_IOR),
            coat_affect_color: pr(ctx, P_COAT_AFFECT_COLOR),
            coat_affect_roughness: pr(ctx, P_COAT_AFFECT_ROUGHNESS),
            thin_film_thickness: pr(ctx, P_THIN_FILM_THICKNESS),
            thin_film_ior: pr(ctx, P_THIN_FILM_IOR),
            emission: pr(ctx, P_EMISSION),
            emission_color: pv3(ctx, P_EMISSION_COLOR),
            thin_film_energy: if global(P_OFX) != 0.0 && global(P_OFX_THIN_FILM_ENERGY) != 0.0 {
                ThinFilmEnergy::Conserving
            } else {
                ThinFilmEnergy::MaterialX
            },
            ..SurfaceInputs::MATERIALX_DEFAULT
        }
    }

    /// What one path vertex needs from its BSDF: the value and density toward the NEE light
    /// sample (zero unless `light`), and the next direction with its weight.
    ///
    /// The material phase ([`full_scatter`], [`fast_scatter`]) produces it before the shadow
    /// ray marches. The material's inputs, layers and lobe weights are dead once it returns,
    /// so the march (the register-heaviest code of the kernel) never shares the register file
    /// with them; only these few values cross it.
    #[derive(Clone, Copy)]
    struct Scatter {
        /// `f(wo, ld) |cos|` toward the light sample.
        light_f: V3,
        /// The BSDF's density toward the light sample, for the MIS weight.
        light_pdf: f32,
        /// Next direction.
        wi: V3,
        /// `f |cos| / pdf` of `wi`.
        weight: V3,
        /// Density of `wi` (a probability for a delta lobe).
        pdf: f32,
        /// `wi` is a usable direction; otherwise the path ends.
        ok: bool,
        /// `wi` came from a delta lobe (no MIS).
        delta: bool,
        /// `wi` refracts through a transmissive surface.
        crossed: bool,
    }

    /// The full (MaterialX Standard Surface) model's [`Scatter`]: the layers and lobe weights
    /// are derived once and shared by the light evaluation, its pdf and the BSDF sample, the
    /// same values the `_with` entry points would re-derive per call.
    #[inline(always)]
    fn full_scatter(
        inputs: &SurfaceInputs,
        frame: &ShadingFrame,
        wo: V3,
        ld: V3,
        light: bool,
        u: V3,
    ) -> Scatter {
        let shading = layers(inputs, frame, wo);
        let weights = lobe_weights_prepared(inputs, &shading, wo);
        let (light_f, light_pdf) = if light {
            let lobes = eval_light_prepared(inputs, frame, &shading, wo, ld);
            let f = add(add(lobes.base, lobes.specular), lobes.transmission);
            let pdf = if max3(f) > 0.0 {
                pdf_prepared(&shading, wo, ld, &weights)
            } else {
                0.0
            };
            (f, pdf)
        } else {
            ([0.0; 3], 0.0)
        };
        let s = sample_prepared(inputs, frame, &shading, wo, u, &weights);
        Scatter {
            light_f,
            light_pdf,
            wi: s.wi,
            weight: s.weight,
            pdf: s.pdf,
            ok: s.valid,
            delta: s.delta,
            crossed: inputs.transmission > 0.0 && dot(s.wi, frame.n) < 0.0,
        }
    }

    /// The fast model's [`Scatter`]: Lambert + one GGX lobe, never delta, never transmissive.
    #[inline(always)]
    fn fast_scatter(m: &Fast, n: V3, wo: V3, ld: V3, light: bool, u: V3) -> Scatter {
        let (light_f, light_pdf) = if light {
            fast_eval(m, n, wo, ld)
        } else {
            ([0.0; 3], 0.0)
        };
        let (wi, weight, pdf, ok) = fast_sample(m, n, wo, u);
        Scatter {
            light_f,
            light_pdf,
            wi,
            weight,
            pdf,
            ok,
            delta: false,
            crossed: false,
        }
    }

    // =========================================================================

    // The CUDA macro must see kernel declarations inline to generate launchers.
    // The optional Direct implementation itself lives in its own source module.
    #[cfg(feature = "direct")]
    mod direct;

    // integrator (render-rs pt-integrator pt_trace_path)
    // =========================================================================

    /// One camera path. `FULL` picks the material model at compile time; the camera ray marches
    /// from `start` ([`cone_start`], 0 when unknown).
    #[inline(always)]
    fn trace_path<const FULL: bool, const F: u32, const MIXED: bool>(
        ctx: Context<'_>,
        lut: &[[f32; 4]],
        origin: V3,
        dir0: V3,
        start: f32,
        r: &mut Rng,
    ) -> PathSample {
        let cam = pv3(ctx, P_CAM_ORIGIN);
        let slope = pr(ctx, P_FOOTPRINT);
        let max_bounces = pr(ctx, P_MAX_BOUNCES) as u32;

        let mut ro = origin;
        // The restart point of the ray `ro`, `rd` (`leave_ray`); a camera ray has none.
        let mut rr = origin;
        let mut rd = dir0;
        let mut throughput = [1.0f32; 3];
        let mut radiance = [0.0f32; 3];
        let mut mis_bsdf_pdf = 0.0f32;
        let mut mis_env = false;
        let mut primary_hit = false;
        let mut unresolved = false;
        let mut primary_albedo = [0.0; 3];
        let mut primary_normal = [0.0; 3];
        let mut bounce = 0u32;
        let mut medium = AIR;
        let mut medium_eps = 0.0f32;
        // The last exact interface point includes the numerical ray offset in absorption.
        let mut interface_point = origin;
        loop {
            let base = if bounce == 0 {
                0.0
            } else {
                slope * length(sub(ro, cam))
            };
            let kind = if bounce == 0 { RAY_PRIMARY } else { RAY_BOUNCE };
            let inside = medium != AIR;
            let m = if inside {
                let occupied = Context {
                    object: medium as usize,
                    ..ctx
                };
                march_exit::<F>(occupied, ro, rd, medium_eps)
            } else {
                march_leaving::<F>(
                    ctx,
                    ro,
                    rr,
                    rd,
                    kind,
                    base,
                    slope,
                    if bounce == 0 { start } else { 0.0 },
                )
            };
            // One policy per ray kind for an unresolved march (`Outcome::Unresolved`): the camera
            // ray shows the background (Mandelbulber, Fragmentarium), a bounce ray ends the
            // path - an unknown direction brings no light (its environment would leak into
            // crevices) - and every case is counted for the status bar.
            if m.outcome == Outcome::Unresolved {
                unresolved = true;
                if bounce > 0 {
                    break;
                }
            }
            // An interior exit that cannot be found ends the path, absorbed (`march_exit`).
            if inside && !m.hit() {
                break;
            }
            if !m.hit() {
                // A missed camera ray shows the sun disc only when the host asks for it
                // (`P_CAMERA_SUN`, WarpBro); OFX's CPU and WGSL engines show the sky or map alone
                // (`pathtrace.rs` Tracer::trace_from). It has no competing strategy: weight one.
                if bounce == 0 && global(P_CAMERA_SUN) == 0.0 {
                    let seen = background(ctx, lut, rd);
                    // A missed camera ray's albedo guide is what it shows, clamped to [0, 1]
                    // (ofx-fractal `pathtrace.rs`: the denoiser keeps the sky as detail).
                    primary_albedo = seen.map(|v| v.max(0.0).min(1.0));
                    radiance = add(radiance, had(throughput, seen));
                    break;
                }
                if bounce == 0 {
                    // The camera ray shows the environment and the sun disc (`P_CAMERA_SUN`).
                    primary_albedo = env_radiance(ctx, lut, rd).map(|v| v.max(0.0).min(1.0));
                }
                let w = if mis_env {
                    power_heuristic(mis_bsdf_pdf, env_pdf(ctx, lut, rd))
                } else {
                    1.0
                };
                radiance = add(
                    radiance,
                    mul(had(throughput, env_radiance(ctx, lut, rd)), w),
                );
                break;
            }
            // The final BSDF ray still supplies its complementary MIS estimate
            // when it sees the environment. A surface beyond the depth limit must
            // not contribute emission or shading; testing depth before a miss used
            // to discard the environment estimate and bias the final NEE dark.
            if !allows_surface_vertex(bounce, max_bounces) {
                break;
            }
            if bounce == 0 {
                primary_hit = true;
            }
            let ctx = Context {
                object: m.object,
                ..ctx
            };
            let full = if MIXED && F == FAMILY_WORLD {
                pr(ctx, P_MATERIAL_MODEL) != 0.0
            } else {
                FULL
            };
            let up = if F == FAMILY_WORLD || ctx.world {
                let b = ctx.object * OBJECT_STRIDE + O_TANGENT;
                [ctx.objects[b], ctx.objects[b + 1], ctx.objects[b + 2]]
            } else {
                mat3(ctx, P_OBJ_AXES, [0.0, 1.0, 0.0])
            };
            if inside {
                throughput = had(
                    throughput,
                    volume_transmittance(
                        pv3(ctx, P_TRANSMISSION_COLOR),
                        pr(ctx, P_TRANSMISSION_DEPTH),
                        length(sub(m.point, interface_point)),
                    ),
                );
            }
            let n = surface_normal::<F>(ctx, m.point, if inside { neg(rd) } else { rd }, m.eps);
            let geo_n = if dot(n, rd) > 0.0 { neg(n) } else { n };
            let eps = 4.0 * m.eps;
            let wo = neg(rd);
            // Base colour: the palette (escape / trap colouring) or the material's solid colour.
            let mut color = if pr(ctx, P_COLOR_SOURCE) != 0.0 {
                pv3(ctx, P_BASE_COLOR)
            } else {
                hit_palette(ctx, lut, m.point, n, m.trap)
            };
            let mut roughness = pr(ctx, P_SPECULAR_ROUGHNESS);
            let mut metal = pr(ctx, P_METALNESS);
            // usd-rs pt-material-ext facing mix: toward material B at grazing angles.
            let facing_exp = pr(ctx, P_FACING_EXPONENT);
            if facing_exp > 0.0 {
                let f = (1.0 - dot(geo_n, wo).abs()).max(0.0).powf(facing_exp);
                let b = pv3(ctx, P_FACING_COLOR);
                color = add(color, mul(sub(b, color), f));
                roughness += (pr(ctx, P_FACING_ROUGHNESS) - roughness) * f;
                metal += (pr(ctx, P_FACING_METALLIC) - metal) * f;
            }

            if bounce == 0 {
                // The denoiser's albedo guide: the one definition (render-rs
                // `SurfaceInputs::guide_albedo`), on the inputs `surface_inputs` gives these fields.
                primary_albedo = SurfaceInputs {
                    base: pr(ctx, P_BASE),
                    base_color: had(color, pv3(ctx, P_BASE_TINT)),
                    emission: pr(ctx, P_EMISSION),
                    emission_color: pv3(ctx, P_EMISSION_COLOR),
                    ..SurfaceInputs::MATERIALX_DEFAULT
                }
                .guide_albedo();
                primary_normal = geo_n;
            }

            // Emission, then Russian roulette: both read only this vertex and the path.
            let full_inputs = if full {
                surface_inputs(ctx, color, roughness, metal)
            } else {
                SurfaceInputs::MATERIALX_DEFAULT
            };
            let frame = ShadingFrame {
                n: geo_n,
                tangent: up,
                inside,
                curvature: 0.0,
            };
            if full {
                radiance = add(
                    radiance,
                    had(throughput, eval_emission(&full_inputs, &frame, wo)),
                );
            } else if pr(ctx, P_EMISSION) > 0.0 {
                radiance = add(
                    radiance,
                    had(
                        throughput,
                        mul(pv3(ctx, P_EMISSION_COLOR), pr(ctx, P_EMISSION)),
                    ),
                );
            }

            // Russian roulette after the first bounce.
            if bounce > 0 {
                let p_continue = max3(throughput).min(1.0);
                if rand(r) >= p_continue {
                    break;
                }
                throughput = mul(throughput, 1.0 / p_continue);
            }

            // NEE samples one sun/sky direction, then the BSDF its next direction: the random
            // numbers are drawn in this order (r1, r2, u). The material phase answers both
            // before the shadow ray marches, so only `Scatter`'s few values cross that march.
            let r1 = rand(r);
            let r2 = rand(r);
            // A point sun (no cone: `P_SUN_CONE_PDF` 0) is a delta light, as on OFX's CPU and in
            // WGSL (`Nee::Delta`): its irradiance along its direction when the shadow ray is
            // clear, no MIS, no sky sample (escaping BSDF rays see the sky at weight one). An HDRI
            // whose sun has no light (`P_ENV_SUN_SELECT` 0) is the map alone, sampled by its CDFs.
            let point_sun = !ctx.world
                && pr(ctx, P_SUN_CONE_PDF) <= 0.0
                && !(env_on(ctx) && pr(ctx, P_ENV_SUN_SELECT) == 0.0);
            let (ld, lpdf) = if point_sun {
                (pv3(ctx, P_LIGHT_DIR), 1.0)
            } else {
                env_sample(ctx, lut, r1, r2)
            };
            let light = lpdf > 0.0 && !inside && dot(ld, geo_n) > 0.0;
            let u = [rand(r), rand(r), rand(r)];
            let scatter = if full {
                full_scatter(&full_inputs, &frame, wo, ld, light, u)
            } else {
                fast_scatter(
                    &fast_material(ctx, color, roughness, metal),
                    geo_n,
                    wo,
                    ld,
                    light,
                    u,
                )
            };
            // NEE, MIS-weighted against BSDF sampling.
            if light && max3(scatter.light_f) > 0.0 {
                let (so, sr) = leave_ray(m.point, geo_n, rd, ld, eps);
                let sb = slope * length(sub(so, cam));
                // An unresolved shadow ray counts as blocked: no light from an unknown path.
                let shadow = visibility_leaving::<F>(ctx, so, sr, ld, sb, 0.0);
                unresolved |= shadow == Outcome::Unresolved;
                if shadow == Outcome::Miss {
                    let light = if point_sun {
                        mul(pv3(ctx, P_LIGHT_COLOR), pr(ctx, P_LIGHT_INTENSITY))
                    } else {
                        let w = power_heuristic(lpdf, scatter.light_pdf);
                        mul(env_radiance(ctx, lut, ld), w / lpdf)
                    };
                    radiance = add(radiance, had(throughput, had(scatter.light_f, light)));
                }
            }

            let Scatter {
                wi,
                weight,
                pdf,
                ok,
                delta,
                crossed,
                ..
            } = scatter;
            if !ok {
                break;
            }
            throughput = had(throughput, weight);
            mis_bsdf_pdf = pdf;
            // NEE never connects through an interface, so a transmitted escape has no
            // competing light-sampling strategy and must retain weight one.
            // Nor does a point sun: NEE then never samples the sky.
            mis_env = !delta && !crossed && !point_sun;
            medium = medium_after_scatter(medium, m.object as u32, crossed);
            if crossed && medium != AIR {
                medium_eps = m.eps;
            }
            interface_point = m.point;
            (ro, rr) = leave_ray(m.point, geo_n, rd, wi, eps);
            rd = wi;
            bounce += 1;
        }
        PathSample {
            radiance,
            hit: primary_hit,
            albedo: primary_albedo,
            normal: primary_normal,
            unresolved,
        }
    }

    /// Linear thread index -> pixel, in 8x4 tiles so a warp traces a compact patch.
    #[inline(always)]
    fn tile_pixel(i: u32, width: u32) -> (u32, u32) {
        let tiles_x = width.div_ceil(8);
        let tile = i / 32;
        let lane = i % 32;
        (
            (tile % tiles_x) * 8 + lane % 8,
            (tile / tiles_x) * 4 + lane / 8,
        )
    }

    /// One pixel's accumulators, written by `trace_pixel`: (rgb sum, sample count) radiance,
    /// the albedo and normal guides, and stats: (sum of squared luminance, unresolved march
    /// count, the camera-ray start, 0) for the adaptive variance, the march-limit indicator and
    /// the pixel-cone start every later launch reuses.
    struct PixelOut<'a> {
        acc: &'a mut [f32; 4],
        albedo: &'a mut [f32; 4],
        normal: &'a mut [f32; 4],
        stats: &'a mut [f32; 4],
    }

    #[inline(always)]
    fn trace_pixel<const FULL: bool, const F: u32, const MIXED: bool, const DIRECT: bool>(
        ctx: Context<'_>,
        lut: &[[f32; 4]],
        i: u32,
        active: &[u32],
        out: PixelOut<'_>,
    ) {
        let PixelOut {
            acc,
            albedo,
            normal,
            stats,
        } = out;
        // Adaptive sampling: a converged 8x4 tile (one warp) is skipped as a whole.
        if active.get((i / 32) as usize).is_some_and(|&a| a == 0) {
            return;
        }
        let width = pr(ctx, P_WIDTH) as u32;
        let height = pr(ctx, P_HEIGHT) as u32;
        let ofx = global(P_OFX) != 0.0;
        let tile_width = if ofx {
            global(P_TILE_WIDTH) as u32
        } else {
            width
        };
        let tile_height = if ofx {
            global(P_TILE_HEIGHT) as u32
        } else {
            height
        };
        let (local_x, local_y) = tile_pixel(i, tile_width);
        if local_x >= tile_width || local_y >= tile_height {
            return;
        }
        let x = local_x + if ofx { global(P_TILE_X) as u32 } else { 0 };
        let y = local_y + if ofx { global(P_TILE_Y) as u32 } else { 0 };
        if x >= width || y >= height {
            return;
        }
        let begin = pr(ctx, P_SAMPLE_BEGIN) as u32;
        let spp = pr(ctx, P_SPP) as u32;
        let seed = if ofx {
            (global(P_OFX_SEED_HIGH) as u32) << 16 | (pr(ctx, P_SEED) as u32)
        } else {
            pr(ctx, P_SEED) as u32
        };
        let origin = pv3(ctx, P_CAM_ORIGIN);
        let fwd = pv3(ctx, P_CAM_FORWARD);
        let right = pv3(ctx, P_CAM_RIGHT);
        let up = pv3(ctx, P_CAM_UP);
        let aperture = pr(ctx, P_APERTURE);
        let background = pr(ctx, P_BACKGROUND) != 0.0;
        // The pixel's camera-ray start, the same for every sample and launch: found once by the
        // launch of sample 0 and kept in `stats[2]` (OFX keeps it in the albedo guide's w; here
        // that w is the guide's sample count the denoiser divides by). INVARIANT: an
        // accumulation's launches begin at sample 0 on zeroed stats, as the hosts reset them;
        // a later launch on a fresh buffer reads 0, i.e. no start. A lens ray does not start at
        // the cone's apex, so depth of field (and Direct) march from `t_enter`.
        let start = if DIRECT || aperture > 0.0 {
            0.0
        } else if begin != 0 {
            stats[2]
        } else {
            let cx = 2.0 * (x as f32 + 0.5) / width as f32 - 1.0;
            let cy = 1.0 - 2.0 * (y as f32 + 0.5) / height as f32;
            let centre = normalize(add(
                fwd,
                add(
                    mul(right, cx * pr(ctx, P_HALF_W)),
                    mul(up, cy * pr(ctx, P_HALF_H)),
                ),
            ));
            cone_start::<F>(ctx, origin, centre)
        };
        let mut sum = [acc[0], acc[1], acc[2]];
        let mut albedo_sum = [albedo[0], albedo[1], albedo[2]];
        let mut normal_sum = [normal[0], normal[1], normal[2]];
        let mut primary_hits = 0.0;
        // Sum of squared sample luminance: with `acc` it gives the pixel's sample variance.
        let mut luma2 = 0.0f32;
        // Samples with an unresolved march (out of steps; see `Outcome`), for the status bar.
        let mut unresolved = 0.0f32;
        let mut s = 0u32;
        while s < spp {
            let mut r = Rng {
                px: x ^ seed,
                py: y,
                sample: begin + s,
                dim: 0,
            };
            let grid = global(P_DIRECT_GRID) as u32;
            let index = begin + s;
            let fx = x as f32
                + if DIRECT {
                    ((index % grid) as f32 + 0.5) / grid as f32
                } else {
                    rand(&mut r)
                };
            let fy = y as f32
                + if DIRECT {
                    ((index / grid) as f32 + 0.5) / grid as f32
                } else {
                    rand(&mut r)
                };
            let nx = 2.0 * fx / width as f32 - 1.0;
            let ny = 1.0 - 2.0 * fy / height as f32;
            let mut ro = origin;
            let mut dir = normalize(add(
                fwd,
                add(
                    mul(right, nx * pr(ctx, P_HALF_W)),
                    mul(up, ny * pr(ctx, P_HALF_H)),
                ),
            ));
            if !DIRECT && aperture > 0.0 {
                // Thin lens: focus plane at P_FOCUS_DISTANCE along the view axis.
                let focus = add(origin, mul(dir, pr(ctx, P_FOCUS_DISTANCE) / dot(dir, fwd)));
                let (sa, ca) = sin_cos(2.0 * PI * rand(&mut r));
                let rr = aperture * rand(&mut r).sqrt();
                ro = add(origin, add(mul(right, rr * ca), mul(up, rr * sa)));
                dir = normalize(sub(focus, ro));
            }
            #[cfg(feature = "direct")]
            let sample = if DIRECT {
                direct::trace_direct::<F>(ctx, lut, ro, dir)
            } else {
                trace_path::<FULL, F, MIXED>(ctx, lut, ro, dir, start, &mut r)
            };
            #[cfg(not(feature = "direct"))]
            let sample = trace_path::<FULL, F, MIXED>(ctx, lut, ro, dir, start, &mut r);
            let PathSample {
                radiance: l,
                hit,
                albedo: a,
                normal: n,
                ..
            } = sample;
            if sample.unresolved {
                unresolved += 1.0;
            }
            primary_hits += if hit {
                1.0
            } else {
                if ofx { global(P_OFX_SKY_ALPHA) } else { 0.0 }
            };
            albedo_sum = add(albedo_sum, a);
            normal_sum = add(normal_sum, n);
            let l = if ofx {
                if hit {
                    l
                } else {
                    mul(l, global(P_OFX_SKY_ALPHA))
                }
            } else if hit || background {
                l
            } else {
                [0.0; 3]
            };
            sum = add(sum, l);
            let y = luminance(l);
            luma2 += y * y;
            s += 1;
        }
        *acc = [sum[0], sum[1], sum[2], acc[3] + spp as f32];
        *stats = [stats[0] + luma2, stats[1] + unresolved, start, 0.0];
        *albedo = [
            albedo_sum[0],
            albedo_sum[1],
            albedo_sum[2],
            albedo[3] + spp as f32,
        ];
        *normal = [
            normal_sum[0],
            normal_sum[1],
            normal_sum[2],
            normal[3] + if ofx { primary_hits } else { spp as f32 },
        ];
    }

    // One kernel per (family, material model): the family and the model are compile-time
    // constants, so each kernel carries only its own estimate and BSDF (fewer registers, no
    // dispatch). Accumulator rows: (rgb sum, sample count), tile order.

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn fast_bulb(
        lut: &[[f32; 4]],
        objects: &[f32],
        lights: &[f32],
        mut accum: DisjointSlice<[f32; 4]>,
        mut albedo: DisjointSlice<[f32; 4]>,
        mut normal: DisjointSlice<[f32; 4]>,
        active: &[u32],
        mut stats: DisjointSlice<[f32; 4]>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if let (Some(acc), Some(albedo), Some(normal), Some(stats)) = (
            accum.get_mut(idx),
            albedo.get_mut(thread::index_1d()),
            normal.get_mut(thread::index_1d()),
            stats.get_mut(thread::index_1d()),
        ) {
            trace_pixel::<false, FAMILY_BULB, false, false>(
                Context {
                    world: false,
                    objects,
                    lights,
                    object: 0,
                },
                lut,
                i,
                active,
                PixelOut {
                    acc,
                    albedo,
                    normal,
                    stats,
                },
            );
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn fast_box(
        lut: &[[f32; 4]],
        objects: &[f32],
        lights: &[f32],
        mut accum: DisjointSlice<[f32; 4]>,
        mut albedo: DisjointSlice<[f32; 4]>,
        mut normal: DisjointSlice<[f32; 4]>,
        active: &[u32],
        mut stats: DisjointSlice<[f32; 4]>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if let (Some(acc), Some(albedo), Some(normal), Some(stats)) = (
            accum.get_mut(idx),
            albedo.get_mut(thread::index_1d()),
            normal.get_mut(thread::index_1d()),
            stats.get_mut(thread::index_1d()),
        ) {
            trace_pixel::<false, FAMILY_BOX, false, false>(
                Context {
                    world: false,
                    objects,
                    lights,
                    object: 0,
                },
                lut,
                i,
                active,
                PixelOut {
                    acc,
                    albedo,
                    normal,
                    stats,
                },
            );
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn fast_quat(
        lut: &[[f32; 4]],
        objects: &[f32],
        lights: &[f32],
        mut accum: DisjointSlice<[f32; 4]>,
        mut albedo: DisjointSlice<[f32; 4]>,
        mut normal: DisjointSlice<[f32; 4]>,
        active: &[u32],
        mut stats: DisjointSlice<[f32; 4]>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if let (Some(acc), Some(albedo), Some(normal), Some(stats)) = (
            accum.get_mut(idx),
            albedo.get_mut(thread::index_1d()),
            normal.get_mut(thread::index_1d()),
            stats.get_mut(thread::index_1d()),
        ) {
            trace_pixel::<false, FAMILY_QUAT, false, false>(
                Context {
                    world: false,
                    objects,
                    lights,
                    object: 0,
                },
                lut,
                i,
                active,
                PixelOut {
                    acc,
                    albedo,
                    normal,
                    stats,
                },
            );
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn fast_kifs(
        lut: &[[f32; 4]],
        objects: &[f32],
        lights: &[f32],
        mut accum: DisjointSlice<[f32; 4]>,
        mut albedo: DisjointSlice<[f32; 4]>,
        mut normal: DisjointSlice<[f32; 4]>,
        active: &[u32],
        mut stats: DisjointSlice<[f32; 4]>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if let (Some(acc), Some(albedo), Some(normal), Some(stats)) = (
            accum.get_mut(idx),
            albedo.get_mut(thread::index_1d()),
            normal.get_mut(thread::index_1d()),
            stats.get_mut(thread::index_1d()),
        ) {
            trace_pixel::<false, FAMILY_KIFS, false, false>(
                Context {
                    world: false,
                    objects,
                    lights,
                    object: 0,
                },
                lut,
                i,
                active,
                PixelOut {
                    acc,
                    albedo,
                    normal,
                    stats,
                },
            );
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn fast_kleinian(
        lut: &[[f32; 4]],
        objects: &[f32],
        lights: &[f32],
        mut accum: DisjointSlice<[f32; 4]>,
        mut albedo: DisjointSlice<[f32; 4]>,
        mut normal: DisjointSlice<[f32; 4]>,
        active: &[u32],
        mut stats: DisjointSlice<[f32; 4]>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if let (Some(acc), Some(albedo), Some(normal), Some(stats)) = (
            accum.get_mut(idx),
            albedo.get_mut(thread::index_1d()),
            normal.get_mut(thread::index_1d()),
            stats.get_mut(thread::index_1d()),
        ) {
            trace_pixel::<false, FAMILY_KLEINIAN, false, false>(
                Context {
                    world: false,
                    objects,
                    lights,
                    object: 0,
                },
                lut,
                i,
                active,
                PixelOut {
                    acc,
                    albedo,
                    normal,
                    stats,
                },
            );
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn fast_pseudo_kleinian(
        lut: &[[f32; 4]],
        objects: &[f32],
        lights: &[f32],
        mut accum: DisjointSlice<[f32; 4]>,
        mut albedo: DisjointSlice<[f32; 4]>,
        mut normal: DisjointSlice<[f32; 4]>,
        active: &[u32],
        mut stats: DisjointSlice<[f32; 4]>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if let (Some(acc), Some(albedo), Some(normal), Some(stats)) = (
            accum.get_mut(idx),
            albedo.get_mut(thread::index_1d()),
            normal.get_mut(thread::index_1d()),
            stats.get_mut(thread::index_1d()),
        ) {
            trace_pixel::<false, FAMILY_PSEUDO_KLEINIAN, false, false>(
                Context {
                    world: false,
                    objects,
                    lights,
                    object: 0,
                },
                lut,
                i,
                active,
                PixelOut {
                    acc,
                    albedo,
                    normal,
                    stats,
                },
            );
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn fast_apollonian(
        lut: &[[f32; 4]],
        objects: &[f32],
        lights: &[f32],
        mut accum: DisjointSlice<[f32; 4]>,
        mut albedo: DisjointSlice<[f32; 4]>,
        mut normal: DisjointSlice<[f32; 4]>,
        active: &[u32],
        mut stats: DisjointSlice<[f32; 4]>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if let (Some(acc), Some(albedo), Some(normal), Some(stats)) = (
            accum.get_mut(idx),
            albedo.get_mut(thread::index_1d()),
            normal.get_mut(thread::index_1d()),
            stats.get_mut(thread::index_1d()),
        ) {
            trace_pixel::<false, FAMILY_APOLLONIAN, false, false>(
                Context {
                    world: false,
                    objects,
                    lights,
                    object: 0,
                },
                lut,
                i,
                active,
                PixelOut {
                    acc,
                    albedo,
                    normal,
                    stats,
                },
            );
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn fast_hybrid(
        lut: &[[f32; 4]],
        objects: &[f32],
        lights: &[f32],
        mut accum: DisjointSlice<[f32; 4]>,
        mut albedo: DisjointSlice<[f32; 4]>,
        mut normal: DisjointSlice<[f32; 4]>,
        active: &[u32],
        mut stats: DisjointSlice<[f32; 4]>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if let (Some(acc), Some(albedo), Some(normal), Some(stats)) = (
            accum.get_mut(idx),
            albedo.get_mut(thread::index_1d()),
            normal.get_mut(thread::index_1d()),
            stats.get_mut(thread::index_1d()),
        ) {
            trace_pixel::<false, FAMILY_HYBRID, false, false>(
                Context {
                    world: false,
                    objects,
                    lights,
                    object: 0,
                },
                lut,
                i,
                active,
                PixelOut {
                    acc,
                    albedo,
                    normal,
                    stats,
                },
            );
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn full_bulb(
        lut: &[[f32; 4]],
        objects: &[f32],
        lights: &[f32],
        mut accum: DisjointSlice<[f32; 4]>,
        mut albedo: DisjointSlice<[f32; 4]>,
        mut normal: DisjointSlice<[f32; 4]>,
        active: &[u32],
        mut stats: DisjointSlice<[f32; 4]>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if let (Some(acc), Some(albedo), Some(normal), Some(stats)) = (
            accum.get_mut(idx),
            albedo.get_mut(thread::index_1d()),
            normal.get_mut(thread::index_1d()),
            stats.get_mut(thread::index_1d()),
        ) {
            trace_pixel::<true, FAMILY_BULB, false, false>(
                Context {
                    world: false,
                    objects,
                    lights,
                    object: 0,
                },
                lut,
                i,
                active,
                PixelOut {
                    acc,
                    albedo,
                    normal,
                    stats,
                },
            );
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn full_box(
        lut: &[[f32; 4]],
        objects: &[f32],
        lights: &[f32],
        mut accum: DisjointSlice<[f32; 4]>,
        mut albedo: DisjointSlice<[f32; 4]>,
        mut normal: DisjointSlice<[f32; 4]>,
        active: &[u32],
        mut stats: DisjointSlice<[f32; 4]>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if let (Some(acc), Some(albedo), Some(normal), Some(stats)) = (
            accum.get_mut(idx),
            albedo.get_mut(thread::index_1d()),
            normal.get_mut(thread::index_1d()),
            stats.get_mut(thread::index_1d()),
        ) {
            trace_pixel::<true, FAMILY_BOX, false, false>(
                Context {
                    world: false,
                    objects,
                    lights,
                    object: 0,
                },
                lut,
                i,
                active,
                PixelOut {
                    acc,
                    albedo,
                    normal,
                    stats,
                },
            );
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn full_quat(
        lut: &[[f32; 4]],
        objects: &[f32],
        lights: &[f32],
        mut accum: DisjointSlice<[f32; 4]>,
        mut albedo: DisjointSlice<[f32; 4]>,
        mut normal: DisjointSlice<[f32; 4]>,
        active: &[u32],
        mut stats: DisjointSlice<[f32; 4]>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if let (Some(acc), Some(albedo), Some(normal), Some(stats)) = (
            accum.get_mut(idx),
            albedo.get_mut(thread::index_1d()),
            normal.get_mut(thread::index_1d()),
            stats.get_mut(thread::index_1d()),
        ) {
            trace_pixel::<true, FAMILY_QUAT, false, false>(
                Context {
                    world: false,
                    objects,
                    lights,
                    object: 0,
                },
                lut,
                i,
                active,
                PixelOut {
                    acc,
                    albedo,
                    normal,
                    stats,
                },
            );
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn full_kifs(
        lut: &[[f32; 4]],
        objects: &[f32],
        lights: &[f32],
        mut accum: DisjointSlice<[f32; 4]>,
        mut albedo: DisjointSlice<[f32; 4]>,
        mut normal: DisjointSlice<[f32; 4]>,
        active: &[u32],
        mut stats: DisjointSlice<[f32; 4]>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if let (Some(acc), Some(albedo), Some(normal), Some(stats)) = (
            accum.get_mut(idx),
            albedo.get_mut(thread::index_1d()),
            normal.get_mut(thread::index_1d()),
            stats.get_mut(thread::index_1d()),
        ) {
            trace_pixel::<true, FAMILY_KIFS, false, false>(
                Context {
                    world: false,
                    objects,
                    lights,
                    object: 0,
                },
                lut,
                i,
                active,
                PixelOut {
                    acc,
                    albedo,
                    normal,
                    stats,
                },
            );
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn full_kleinian(
        lut: &[[f32; 4]],
        objects: &[f32],
        lights: &[f32],
        mut accum: DisjointSlice<[f32; 4]>,
        mut albedo: DisjointSlice<[f32; 4]>,
        mut normal: DisjointSlice<[f32; 4]>,
        active: &[u32],
        mut stats: DisjointSlice<[f32; 4]>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if let (Some(acc), Some(albedo), Some(normal), Some(stats)) = (
            accum.get_mut(idx),
            albedo.get_mut(thread::index_1d()),
            normal.get_mut(thread::index_1d()),
            stats.get_mut(thread::index_1d()),
        ) {
            trace_pixel::<true, FAMILY_KLEINIAN, false, false>(
                Context {
                    world: false,
                    objects,
                    lights,
                    object: 0,
                },
                lut,
                i,
                active,
                PixelOut {
                    acc,
                    albedo,
                    normal,
                    stats,
                },
            );
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn full_pseudo_kleinian(
        lut: &[[f32; 4]],
        objects: &[f32],
        lights: &[f32],
        mut accum: DisjointSlice<[f32; 4]>,
        mut albedo: DisjointSlice<[f32; 4]>,
        mut normal: DisjointSlice<[f32; 4]>,
        active: &[u32],
        mut stats: DisjointSlice<[f32; 4]>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if let (Some(acc), Some(albedo), Some(normal), Some(stats)) = (
            accum.get_mut(idx),
            albedo.get_mut(thread::index_1d()),
            normal.get_mut(thread::index_1d()),
            stats.get_mut(thread::index_1d()),
        ) {
            trace_pixel::<true, FAMILY_PSEUDO_KLEINIAN, false, false>(
                Context {
                    world: false,
                    objects,
                    lights,
                    object: 0,
                },
                lut,
                i,
                active,
                PixelOut {
                    acc,
                    albedo,
                    normal,
                    stats,
                },
            );
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn full_apollonian(
        lut: &[[f32; 4]],
        objects: &[f32],
        lights: &[f32],
        mut accum: DisjointSlice<[f32; 4]>,
        mut albedo: DisjointSlice<[f32; 4]>,
        mut normal: DisjointSlice<[f32; 4]>,
        active: &[u32],
        mut stats: DisjointSlice<[f32; 4]>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if let (Some(acc), Some(albedo), Some(normal), Some(stats)) = (
            accum.get_mut(idx),
            albedo.get_mut(thread::index_1d()),
            normal.get_mut(thread::index_1d()),
            stats.get_mut(thread::index_1d()),
        ) {
            trace_pixel::<true, FAMILY_APOLLONIAN, false, false>(
                Context {
                    world: false,
                    objects,
                    lights,
                    object: 0,
                },
                lut,
                i,
                active,
                PixelOut {
                    acc,
                    albedo,
                    normal,
                    stats,
                },
            );
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn full_hybrid(
        lut: &[[f32; 4]],
        objects: &[f32],
        lights: &[f32],
        mut accum: DisjointSlice<[f32; 4]>,
        mut albedo: DisjointSlice<[f32; 4]>,
        mut normal: DisjointSlice<[f32; 4]>,
        active: &[u32],
        mut stats: DisjointSlice<[f32; 4]>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if let (Some(acc), Some(albedo), Some(normal), Some(stats)) = (
            accum.get_mut(idx),
            albedo.get_mut(thread::index_1d()),
            normal.get_mut(thread::index_1d()),
            stats.get_mut(thread::index_1d()),
        ) {
            trace_pixel::<true, FAMILY_HYBRID, false, false>(
                Context {
                    world: false,
                    objects,
                    lights,
                    object: 0,
                },
                lut,
                i,
                active,
                PixelOut {
                    acc,
                    albedo,
                    normal,
                    stats,
                },
            );
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn world(
        lut: &[[f32; 4]],
        objects: &[f32],
        lights: &[f32],
        mut accum: DisjointSlice<[f32; 4]>,
        mut albedo: DisjointSlice<[f32; 4]>,
        mut normal: DisjointSlice<[f32; 4]>,
        active: &[u32],
        mut stats: DisjointSlice<[f32; 4]>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if let (Some(acc), Some(albedo), Some(normal), Some(stats)) = (
            accum.get_mut(idx),
            albedo.get_mut(thread::index_1d()),
            normal.get_mut(thread::index_1d()),
            stats.get_mut(thread::index_1d()),
        ) {
            trace_pixel::<false, FAMILY_WORLD, true, false>(
                Context {
                    world: true,
                    objects,
                    lights,
                    object: 0,
                },
                lut,
                i,
                active,
                PixelOut {
                    acc,
                    albedo,
                    normal,
                    stats,
                },
            );
        }
    }

    /// Viewport picking: the world's camera ray through each query point (`[x, y, _, _]` in
    /// normalized device coordinates, -1..1, y up), without a lens, marched as a camera sample's
    /// first ray (the same estimators, transforms and masks, [`cone_start`]). `out` gets
    /// `[object + 1 (0: nothing hit), distance along the ray, depth along the view axis, 0]`.
    /// Reads the `PARAMS` the world launch uploads: the host sets them first.
    #[kernel]
    #[launch_contract(domain = 1, block = (32, 1, 1))]
    pub fn pick(
        objects: &[f32],
        lights: &[f32],
        query: &[[f32; 4]],
        mut out: DisjointSlice<[f32; 4]>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let (Some(q), Some(o)) = (query.get(i), out.get_mut(idx)) {
            let ctx = Context {
                world: true,
                objects,
                lights,
                object: 0,
            };
            let origin = pv3(ctx, P_CAM_ORIGIN);
            let fwd = pv3(ctx, P_CAM_FORWARD);
            let dir = normalize(add(
                fwd,
                add(
                    mul(pv3(ctx, P_CAM_RIGHT), q[0] * pr(ctx, P_HALF_W)),
                    mul(pv3(ctx, P_CAM_UP), q[1] * pr(ctx, P_HALF_H)),
                ),
            ));
            let start = cone_start::<FAMILY_WORLD>(ctx, origin, dir);
            let m = march::<FAMILY_WORLD>(
                ctx,
                origin,
                dir,
                RAY_PRIMARY,
                0.0,
                pr(ctx, P_FOOTPRINT),
                start,
            );
            *o = if m.hit() {
                let t = length(sub(m.point, origin));
                [(m.object + 1) as f32, t, t * dot(dir, fwd), 0.0]
            } else {
                [0.0; 4]
            };
        }
    }

    // Uniform-model worlds omit the unused BSDF at compile time, reducing register
    // pressure without bypassing world transforms, clipping, materials or lights.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn world_fast(
        lut: &[[f32; 4]],
        objects: &[f32],
        lights: &[f32],
        mut accum: DisjointSlice<[f32; 4]>,
        mut albedo: DisjointSlice<[f32; 4]>,
        mut normal: DisjointSlice<[f32; 4]>,
        active: &[u32],
        mut stats: DisjointSlice<[f32; 4]>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if let (Some(acc), Some(albedo), Some(normal), Some(stats)) = (
            accum.get_mut(idx),
            albedo.get_mut(thread::index_1d()),
            normal.get_mut(thread::index_1d()),
            stats.get_mut(thread::index_1d()),
        ) {
            trace_pixel::<false, FAMILY_WORLD, false, false>(
                Context {
                    world: true,
                    objects,
                    lights,
                    object: 0,
                },
                lut,
                i,
                active,
                PixelOut {
                    acc,
                    albedo,
                    normal,
                    stats,
                },
            );
        }
    }

    /// One evaluated Fast Mandelbulb can use its compile-time DE without the eight-family
    /// dispatcher. Context::world remains true: full affine inverse, conservative distance
    /// scale, world clipping, palette offsets and world illumination are still authoritative.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn world_fast_bulb(
        lut: &[[f32; 4]],
        objects: &[f32],
        lights: &[f32],
        mut accum: DisjointSlice<[f32; 4]>,
        mut albedo: DisjointSlice<[f32; 4]>,
        mut normal: DisjointSlice<[f32; 4]>,
        active: &[u32],
        mut stats: DisjointSlice<[f32; 4]>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if let (Some(acc), Some(albedo), Some(normal), Some(stats)) = (
            accum.get_mut(idx),
            albedo.get_mut(thread::index_1d()),
            normal.get_mut(thread::index_1d()),
            stats.get_mut(thread::index_1d()),
        ) {
            trace_pixel::<false, FAMILY_BULB, false, false>(
                Context {
                    world: true,
                    objects,
                    lights,
                    object: 0,
                },
                lut,
                i,
                active,
                PixelOut {
                    acc,
                    albedo,
                    normal,
                    stats,
                },
            );
        }
    }

    #[cfg(feature = "direct")]
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn direct_bulb(
        lut: &[[f32; 4]],
        objects: &[f32],
        lights: &[f32],
        mut accum: DisjointSlice<[f32; 4]>,
        mut albedo: DisjointSlice<[f32; 4]>,
        mut normal: DisjointSlice<[f32; 4]>,
        active: &[u32],
        mut stats: DisjointSlice<[f32; 4]>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if let (Some(acc), Some(albedo), Some(normal), Some(stats)) = (
            accum.get_mut(idx),
            albedo.get_mut(thread::index_1d()),
            normal.get_mut(thread::index_1d()),
            stats.get_mut(thread::index_1d()),
        ) {
            trace_pixel::<true, FAMILY_BULB, false, true>(
                Context {
                    world: false,
                    objects,
                    lights,
                    object: 0,
                },
                lut,
                i,
                active,
                PixelOut {
                    acc,
                    albedo,
                    normal,
                    stats,
                },
            );
        }
    }

    #[cfg(feature = "direct")]
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn direct_box(
        lut: &[[f32; 4]],
        objects: &[f32],
        lights: &[f32],
        mut accum: DisjointSlice<[f32; 4]>,
        mut albedo: DisjointSlice<[f32; 4]>,
        mut normal: DisjointSlice<[f32; 4]>,
        active: &[u32],
        mut stats: DisjointSlice<[f32; 4]>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if let (Some(acc), Some(albedo), Some(normal), Some(stats)) = (
            accum.get_mut(idx),
            albedo.get_mut(thread::index_1d()),
            normal.get_mut(thread::index_1d()),
            stats.get_mut(thread::index_1d()),
        ) {
            trace_pixel::<true, FAMILY_BOX, false, true>(
                Context {
                    world: false,
                    objects,
                    lights,
                    object: 0,
                },
                lut,
                i,
                active,
                PixelOut {
                    acc,
                    albedo,
                    normal,
                    stats,
                },
            );
        }
    }

    #[cfg(feature = "direct")]
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn direct_quat(
        lut: &[[f32; 4]],
        objects: &[f32],
        lights: &[f32],
        mut accum: DisjointSlice<[f32; 4]>,
        mut albedo: DisjointSlice<[f32; 4]>,
        mut normal: DisjointSlice<[f32; 4]>,
        active: &[u32],
        mut stats: DisjointSlice<[f32; 4]>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if let (Some(acc), Some(albedo), Some(normal), Some(stats)) = (
            accum.get_mut(idx),
            albedo.get_mut(thread::index_1d()),
            normal.get_mut(thread::index_1d()),
            stats.get_mut(thread::index_1d()),
        ) {
            trace_pixel::<true, FAMILY_QUAT, false, true>(
                Context {
                    world: false,
                    objects,
                    lights,
                    object: 0,
                },
                lut,
                i,
                active,
                PixelOut {
                    acc,
                    albedo,
                    normal,
                    stats,
                },
            );
        }
    }

    #[cfg(feature = "direct")]
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn direct_kifs(
        lut: &[[f32; 4]],
        objects: &[f32],
        lights: &[f32],
        mut accum: DisjointSlice<[f32; 4]>,
        mut albedo: DisjointSlice<[f32; 4]>,
        mut normal: DisjointSlice<[f32; 4]>,
        active: &[u32],
        mut stats: DisjointSlice<[f32; 4]>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if let (Some(acc), Some(albedo), Some(normal), Some(stats)) = (
            accum.get_mut(idx),
            albedo.get_mut(thread::index_1d()),
            normal.get_mut(thread::index_1d()),
            stats.get_mut(thread::index_1d()),
        ) {
            trace_pixel::<true, FAMILY_KIFS, false, true>(
                Context {
                    world: false,
                    objects,
                    lights,
                    object: 0,
                },
                lut,
                i,
                active,
                PixelOut {
                    acc,
                    albedo,
                    normal,
                    stats,
                },
            );
        }
    }

    #[cfg(feature = "direct")]
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn direct_kleinian(
        lut: &[[f32; 4]],
        objects: &[f32],
        lights: &[f32],
        mut accum: DisjointSlice<[f32; 4]>,
        mut albedo: DisjointSlice<[f32; 4]>,
        mut normal: DisjointSlice<[f32; 4]>,
        active: &[u32],
        mut stats: DisjointSlice<[f32; 4]>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if let (Some(acc), Some(albedo), Some(normal), Some(stats)) = (
            accum.get_mut(idx),
            albedo.get_mut(thread::index_1d()),
            normal.get_mut(thread::index_1d()),
            stats.get_mut(thread::index_1d()),
        ) {
            trace_pixel::<true, FAMILY_KLEINIAN, false, true>(
                Context {
                    world: false,
                    objects,
                    lights,
                    object: 0,
                },
                lut,
                i,
                active,
                PixelOut {
                    acc,
                    albedo,
                    normal,
                    stats,
                },
            );
        }
    }

    #[cfg(feature = "direct")]
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn direct_pseudo_kleinian(
        lut: &[[f32; 4]],
        objects: &[f32],
        lights: &[f32],
        mut accum: DisjointSlice<[f32; 4]>,
        mut albedo: DisjointSlice<[f32; 4]>,
        mut normal: DisjointSlice<[f32; 4]>,
        active: &[u32],
        mut stats: DisjointSlice<[f32; 4]>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if let (Some(acc), Some(albedo), Some(normal), Some(stats)) = (
            accum.get_mut(idx),
            albedo.get_mut(thread::index_1d()),
            normal.get_mut(thread::index_1d()),
            stats.get_mut(thread::index_1d()),
        ) {
            trace_pixel::<true, FAMILY_PSEUDO_KLEINIAN, false, true>(
                Context {
                    world: false,
                    objects,
                    lights,
                    object: 0,
                },
                lut,
                i,
                active,
                PixelOut {
                    acc,
                    albedo,
                    normal,
                    stats,
                },
            );
        }
    }

    #[cfg(feature = "direct")]
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn direct_apollonian(
        lut: &[[f32; 4]],
        objects: &[f32],
        lights: &[f32],
        mut accum: DisjointSlice<[f32; 4]>,
        mut albedo: DisjointSlice<[f32; 4]>,
        mut normal: DisjointSlice<[f32; 4]>,
        active: &[u32],
        mut stats: DisjointSlice<[f32; 4]>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if let (Some(acc), Some(albedo), Some(normal), Some(stats)) = (
            accum.get_mut(idx),
            albedo.get_mut(thread::index_1d()),
            normal.get_mut(thread::index_1d()),
            stats.get_mut(thread::index_1d()),
        ) {
            trace_pixel::<true, FAMILY_APOLLONIAN, false, true>(
                Context {
                    world: false,
                    objects,
                    lights,
                    object: 0,
                },
                lut,
                i,
                active,
                PixelOut {
                    acc,
                    albedo,
                    normal,
                    stats,
                },
            );
        }
    }

    #[cfg(feature = "direct")]
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn direct_hybrid(
        lut: &[[f32; 4]],
        objects: &[f32],
        lights: &[f32],
        mut accum: DisjointSlice<[f32; 4]>,
        mut albedo: DisjointSlice<[f32; 4]>,
        mut normal: DisjointSlice<[f32; 4]>,
        active: &[u32],
        mut stats: DisjointSlice<[f32; 4]>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if let (Some(acc), Some(albedo), Some(normal), Some(stats)) = (
            accum.get_mut(idx),
            albedo.get_mut(thread::index_1d()),
            normal.get_mut(thread::index_1d()),
            stats.get_mut(thread::index_1d()),
        ) {
            trace_pixel::<true, FAMILY_HYBRID, false, true>(
                Context {
                    world: false,
                    objects,
                    lights,
                    object: 0,
                },
                lut,
                i,
                active,
                PixelOut {
                    acc,
                    albedo,
                    normal,
                    stats,
                },
            );
        }
    }

    /// Untile the running mean and apply exposure/saturation in scene-linear ACEScg.
    /// Full ACES 2.0 runs through vfx-ocio on the shared wgpu device afterwards.
    /// Adaptive sampling (one thread per 8x4 tile): a tile stays active until every image pixel in
    /// it has `P_ADAPT_MIN` samples and a relative standard error of its mean luminance below
    /// `P_ADAPT_THRESHOLD`. The error is `se / sqrt(mean)` (Cycles-style): roughly perceptual for
    /// HDR, so bright and dark regions converge to a comparable visible noise.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn adapt(accum: &[[f32; 4]], stats: &[[f32; 4]], mut active: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let t = idx.get() as u32;
        if let Some(flag) = active.get_mut(idx) {
            let width = global(P_WIDTH) as u32;
            let height = global(P_HEIGHT) as u32;
            let threshold = global(P_ADAPT_THRESHOLD);
            let min = global(P_ADAPT_MIN);
            let tiles_x = width.div_ceil(8);
            let (tx, ty) = (t % tiles_x, t / tiles_x);
            let mut done = true;
            let mut p = 0u32;
            while p < 32 {
                let (x, y) = (tx * 8 + p % 8, ty * 4 + p / 8);
                let k = (t * 32 + p) as usize;
                if x < width && y < height {
                    let a = accum[k];
                    let n = a[3];
                    if n < min {
                        done = false;
                    } else {
                        let mean = luminance([a[0], a[1], a[2]]) / n;
                        let var = (stats[k][0] / n - mean * mean).max(0.0);
                        let error = (var / n).sqrt() / (mean.max(0.0) + 1.0e-4).sqrt();
                        if error > threshold {
                            done = false;
                        }
                    }
                }
                p += 1;
            }
            *flag = if done { 0 } else { 1 };
        }
    }

    /// Partial sums of [samples whose camera or bounce march ran out of steps, samples taken]:
    /// thread `i` of `sums.len()` adds pixels `i, i + len, ..` (coalesced), so the host reads a
    /// fixed small buffer whatever the resolution. Samples come from `accum`: adaptive sampling
    /// stops converged tiles, so pixels differ in sample count. f64 keeps the counts exact.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn tally(accum: &[[f32; 4]], stats: &[[f32; 4]], mut sums: DisjointSlice<[f64; 2]>) {
        let idx = thread::index_1d();
        let stride = sums.len();
        let mut k = idx.get();
        if let Some(sum) = sums.get_mut(idx) {
            let (mut unresolved, mut samples) = (0.0f64, 0.0f64);
            while let (Some(s), Some(a)) = (stats.get(k), accum.get(k)) {
                unresolved += f64::from(s[1]);
                samples += f64::from(a[3]);
                k += stride;
            }
            *sum = [unresolved, samples];
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn tonemap(accum: &[[f32; 4]], mut out: DisjointSlice<[f32; 4]>) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        let width = global(P_WIDTH) as u32;
        if let Some(px) = out.get_mut(idx) {
            let x = i % width;
            let y = i / width;
            let tile = (y / 4) * width.div_ceil(8) + x / 8;
            let a = accum[(tile * 32 + (y % 4) * 8 + x % 8) as usize];
            let inv = if a[3] > 0.0 {
                global(P_EXPOSURE) / a[3]
            } else {
                0.0
            };
            let c = [a[0] * inv, a[1] * inv, a[2] * inv];
            let l = luminance(c);
            let sat = global(P_SATURATION);
            *px = [
                l + (c[0] - l) * sat,
                l + (c[1] - l) * sat,
                l + (c[2] - l) * sat,
                1.0,
            ];
        }
    }
}
