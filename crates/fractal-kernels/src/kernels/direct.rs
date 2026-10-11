//! Optional deterministic Direct shading (OFX's Direct mode, the twin of its CPU and WGSL
//! `fractal3d::shade`): one primary march, the key light with a soft shadow, five-probe AO and the
//! sky or HDRI through its harmonics and mips. WarpBro's path tracer does not use it.
//! Keeping this module behind the `direct` feature excludes its BSDF and eight entry points
//! from normal device compilation, rather than merely skipping their dispatch.

use super::*;
use standard_surface_bsdf::microfacet::mx_average_alpha;
use standard_surface_bsdf::surface::{coat_alpha, main_alpha};
use standard_surface_bsdf::{Environment, dominant_dir, eval_environment, eval_light_disc};

// Deterministic Direct: evaluated Standard Surface, analytic sky, soft shadow and five-probe AO.
#[inline(always)]
fn direct_stencil<const F: u32>(ctx: Context<'_>, center: V3, mut h: f32, fallback: V3) -> V3 {
    let mut k = 0;
    while k <= NORMAL_STEP_HALVINGS {
        let ax = scene_signed_distance::<F>(ctx, add(center, [h, 0.0, 0.0]));
        let bx = scene_signed_distance::<F>(ctx, add(center, [-h, 0.0, 0.0]));
        let ay = scene_signed_distance::<F>(ctx, add(center, [0.0, h, 0.0]));
        let by = scene_signed_distance::<F>(ctx, add(center, [0.0, -h, 0.0]));
        let az = scene_signed_distance::<F>(ctx, add(center, [0.0, 0.0, h]));
        let bz = scene_signed_distance::<F>(ctx, add(center, [0.0, 0.0, -h]));
        if family_signed::<F>(ctx)
            || (ax > 0.0 && bx > 0.0 && ay > 0.0 && by > 0.0 && az > 0.0 && bz > 0.0)
        {
            let g = [ax - bx, ay - by, az - bz];
            return if length(g) >= 1.0e-7 {
                normalize(g)
            } else {
                fallback
            };
        }
        h *= 0.5;
        k += 1;
    }
    fallback
}
#[inline(always)]
fn direct_cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
#[inline(always)]
fn direct_shadow<const F: u32>(ctx: Context<'_>, p: V3, n: V3, l: V3, eps: f32) -> f32 {
    let strength = global(P_SHADOW_STRENGTH);
    let steps = global(P_SHADOW_STEPS) as u32;
    if strength == 0.0 || steps == 0 {
        return 1.0;
    }
    let max_distance = global(P_MAX_DISTANCE);
    let mut t = 4.0 * eps;
    let origin = add(p, mul(n, t));
    let cap = 2.0 * max_distance / steps as f32;
    let angle = global(P_LIGHT_HALF_ANGLE);
    let sharpness = if angle > 0.0 { 1.0 / angle.tan() } else { 0.0 };
    let mut visibility = 1.0f32;
    let mut i = 0;
    while i < steps && t <= max_distance {
        let d = march_sample::<F>(ctx, add(origin, mul(l, t)), 0).0;
        if d < eps {
            return (1.0 - strength).max(0.0);
        }
        if angle > 0.0 {
            visibility = visibility.min((sharpness * d / t).clamp(0.0, 1.0));
        }
        t += (0.5 * d).max(2.0 * eps).min(cap);
        i += 1;
    }
    (1.0 - strength * (1.0 - visibility)).max(0.0)
}
#[inline(always)]
fn direct_ao<const F: u32>(ctx: Context<'_>, p: V3, n: V3) -> f32 {
    let strength = global(P_AO_STRENGTH);
    let steps = global(P_AO_STEPS) as u32;
    if strength == 0.0 || steps == 0 {
        return 1.0;
    }
    let axis = if n[1].abs() < 0.9 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let tangent = normalize(direct_cross(axis, n));
    let bitangent = direct_cross(n, tangent);
    let directions = [
        n,
        add(mul(n, 0.6), mul(tangent, 0.8)),
        add(mul(n, 0.6), mul(tangent, -0.8)),
        add(mul(n, 0.6), mul(bitangent, 0.8)),
        add(mul(n, 0.6), mul(bitangent, -0.8)),
    ];
    let mut sum = 0.0;
    let mut j = 0;
    while j < 5 {
        let component = if j == 0 { 1.0 } else { 0.6 };
        let mut i = 1;
        while i <= steps {
            let t = global(P_AO_RADIUS) * i as f32 / steps as f32;
            let clearance = t * component;
            let d = march_sample::<F>(ctx, add(p, mul(directions[j], t)), 0).0;
            sum += ((clearance - d) / clearance).clamp(0.0, 1.0);
            i += 1;
        }
        j += 1;
    }
    (1.0 - strength * sum / (steps * 5) as f32).max(0.0)
}
#[inline(always)]
fn direct_roughness(r: f32, v: f32) -> f32 {
    if v == 0.0 {
        r
    } else {
        (r * r * r * r + v).sqrt().sqrt().min(1.0)
    }
}
#[inline(always)]
pub(super) fn trace_direct<const F: u32>(
    ctx: Context<'_>,
    lut: &[[f32; 4]],
    origin: V3,
    dir: V3,
) -> PathSample {
    let m = march::<F>(ctx, origin, dir, RAY_DIRECT, 0.0, global(P_FOOTPRINT), 0.0);
    let unresolved = m.outcome == Outcome::Unresolved;
    // Keinert et al. 2014 (Enhanced Sphere Tracing, 3.2): one ray per pixel, so an unresolved
    // ray's closest sample stands in for the surface when its screen-space error is under half
    // a pixel (`P_SAMPLE_CONE` footprints is a pixel). A sample that is not a hit is over one
    // footprint away, so with a footprint of half a pixel or more nothing qualifies - right:
    // such an error is visible.
    let accepted = m.hit() || (unresolved && m.ratio <= 0.5 * global(P_SAMPLE_CONE));
    if !accepted {
        return PathSample {
            radiance: background(ctx, lut, dir),
            hit: false,
            albedo: [0.0; 3],
            normal: [0.0; 3],
            unresolved,
        };
    }
    let center = if family_signed::<F>(ctx) {
        m.point
    } else {
        sub(m.point, mul(dir, m.eps))
    };
    let fine = direct_stencil::<F>(ctx, center, 0.5 * m.eps, neg(dir));
    let cone = global(P_SAMPLE_CONE);
    let n = if cone > 1.0 {
        direct_stencil::<F>(
            ctx,
            add(m.point, mul(fine, m.eps * cone)),
            0.5 * m.eps * cone,
            fine,
        )
    } else {
        fine
    };
    let delta = sub(n, fine);
    let variance = dot(delta, delta).min(0.18);
    let view = neg(dir);
    let facing = if dot(n, view) > 0.0 { n } else { neg(n) };
    let frame = ShadingFrame {
        n: facing,
        tangent: mat3(ctx, P_OBJ_AXES, [0.0, 1.0, 0.0]),
        inside: false,
        curvature: 0.0,
    };
    let mut color = if global(P_COLOR_SOURCE) != 0.0 {
        pv3(ctx, P_BASE_COLOR)
    } else {
        hit_palette(ctx, lut, m.point, n, m.trap)
    };
    let mut roughness = global(P_SPECULAR_ROUGHNESS);
    let mut metal = global(P_METALNESS);
    let exponent = global(P_FACING_EXPONENT);
    if exponent > 0.0 {
        let f = (1.0 - dot(facing, view).abs()).max(0.0).powf(exponent);
        color = add(color, mul(sub(pv3(ctx, P_FACING_COLOR), color), f));
        roughness += (global(P_FACING_ROUGHNESS) - roughness) * f;
        metal += (global(P_FACING_METALLIC) - metal) * f;
    }
    let mut inputs = surface_inputs(ctx, color, roughness, metal);
    inputs.specular_roughness = direct_roughness(inputs.specular_roughness, variance);
    inputs.coat_roughness = direct_roughness(inputs.coat_roughness, variance);
    let l = pv3(ctx, P_LIGHT_DIR);
    let sun = eval_light_disc(&inputs, &frame, view, l, global(P_LIGHT_HALF_ANGLE)).sum();
    let visibility = direct_shadow::<F>(ctx, m.point, n, l, m.eps);
    let ao = direct_ao::<F>(ctx, m.point, n);
    let specular_alpha = mx_average_alpha(main_alpha(&inputs));
    let coat_lobe_alpha = mx_average_alpha(coat_alpha(&inputs));
    let specular_dir = dominant_dir(&frame, view, specular_alpha);
    let coat_dir = dominant_dir(&frame, view, coat_lobe_alpha);
    // The sky light of Direct (OFX's CPU and WGSL `fractal3d::shade`): the HDRI's harmonics and
    // its mip level matched to each lobe's width, or the gradient's exact convolution.
    let environment = Environment {
        irradiance: mul(sky_irradiance(ctx, lut, facing), ao),
        specular_radiance: mul(sky_specular(ctx, lut, specular_dir, specular_alpha), ao),
        coat_radiance: mul(sky_specular(ctx, lut, coat_dir, coat_lobe_alpha), ao),
        transmission_radiance: [0.0; 3],
    };
    let reflected = eval_environment(&inputs, &frame, view, &environment).sum();
    let emitted = eval_emission(&inputs, &frame, view);
    let key = mul(
        had(pv3(ctx, P_LIGHT_COLOR), sun),
        global(P_LIGHT_INTENSITY) * visibility,
    );
    PathSample {
        radiance: add(add(key, reflected), emitted),
        hit: true,
        albedo: color,
        normal: n,
        unresolved,
    }
}
