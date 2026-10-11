//! Small sampling contracts shared by the CUDA integrator and its independent CPU oracles.
//! Changing a sampling distribution must change its density too; the shading BRDF stays fixed.

/// Bound the number of shaded surface vertices, not the final ray's environment visibility.
/// Call this after handling a miss: the last BSDF sample must still reach the environment so
/// its MIS contribution complements next-event estimation at the last allowed surface.
#[inline(always)]
pub fn allows_surface_vertex(bounce: u32, max_bounces: u32) -> bool {
    bounce <= max_bounces
}

/// Pure metals have no diffuse lobe, so reserve no samples for a cosine distribution.
/// Mixed materials retain the existing conservative probabilities and therefore their support.
#[inline(always)]
pub fn fast_spec_probability(luminance_f0: f32, diffuse_energy: f32, diffuse_weight: f32) -> f32 {
    if diffuse_weight <= 0.0 {
        1.0
    } else {
        (luminance_f0 / (luminance_f0 + diffuse_energy + 1.0e-4)).clamp(0.15, 0.9)
    }
}

/// Isotropic GGX reflection in a tangent frame, using the dependency's visible-normal sampler.
/// Directions below the surface are zero-contribution events, not samples to redraw.
#[inline(always)]
pub fn fast_ggx_sample_local(wo_local: [f32; 3], alpha: f32, xi: [f32; 2]) -> [f32; 3] {
    let h = standard_surface_bsdf::sampling::mx_ggx_importance_sample_vndf(
        xi,
        wo_local,
        [alpha, alpha],
    );
    let vh = wo_local[0] * h[0] + wo_local[1] * h[1] + wo_local[2] * h[2];
    [
        2.0 * vh * h[0] - wo_local[0],
        2.0 * vh * h[1] - wo_local[1],
        2.0 * vh * h[2] - wo_local[2],
    ]
}

/// Density of the visible-normal reflection sampler, in solid angle.
/// The caller already needs the same NDF and Smith G1(view) to evaluate the unchanged BRDF.
#[inline(always)]
pub fn fast_ggx_pdf(ndf: f32, g1_view: f32, n_dot_view: f32) -> f32 {
    if n_dot_view > 0.0 {
        ndf * g1_view / (4.0 * n_dot_view)
    } else {
        0.0
    }
}

/// Lat-long (equirectangular) lookup shared by the CUDA kernel and its CPU oracles.
///
/// Convention (identical to ofx-fractal's `dir_to_equirect_uv`, so both apps show one environment
/// orientation): longitude `atan2(z, x)` runs counter-clockwise seen from +Y, so with the camera
/// basis (forward -Z, right +X, up +Y) `u` INCREASES as the view turns right; `u = 0.25` looks
/// along -Z, `0.5` along +X, `0.75` along +Z. `v = acos(y) / pi` (0 at +Y). `rotation` is the
/// user's Environment rotation in radians: positive values shift `u` up, so the image content
/// appears to turn left in the view. Returns `(u, v)` with `u` wrapped to `[0, 1)`.
#[inline(always)]
pub fn env_uv(dir: [f32; 3], rotation: f32) -> [f32; 2] {
    use core::f32::consts::PI;
    let u = (0.5 + (dir[2].atan2(dir[0]) + rotation) / (2.0 * PI)).rem_euclid(1.0);
    [u, dir[1].clamp(-1.0, 1.0).acos() / PI]
}

/// Exact inverse of [`env_uv`] for importance sampling: the unit direction at horizontal
/// position `u` (any real, wrapped by the trig) and height `y = cos(polar angle)`.
#[inline(always)]
pub fn env_dir(u: f32, y: f32, rotation: f32) -> [f32; 3] {
    use core::f32::consts::PI;
    let radius = (1.0 - y * y).max(0.0).sqrt();
    let (st, ct) = (2.0 * PI * (u - 0.5) - rotation).sin_cos();
    [radius * ct, y, radius * st]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn final_lambert_nee_and_bsdf_visibility_complement_to_unit_irradiance() {
        // Independent hemisphere quadrature: f*cos = mu/pi; uniform sphere PDF = 1/(4pi).
        // NEE alone has ln(17)/16 of the energy. The final BSDF ray must supply the rest,
        // even when max_bounces=0 disallows shading another surface vertex.
        let count = 16_384;
        let mut nee = 0.0;
        let mut bsdf = 0.0;
        for i in 0..count {
            let mu = (i as f64 + 0.5) / count as f64;
            let light_pdf = 1.0 / (4.0 * PI);
            let bsdf_pdf = mu / PI;
            let denominator = light_pdf.powi(2) + bsdf_pdf.powi(2);
            nee += 2.0 * mu * light_pdf.powi(2) / denominator;
            bsdf += 2.0 * mu * bsdf_pdf.powi(2) / denominator;
        }
        nee /= count as f64;
        bsdf /= count as f64;
        let expected_nee = 17.0_f64.ln() / 16.0;
        assert!((nee - expected_nee).abs() < 1e-8);
        assert!((bsdf - (1.0 - expected_nee)).abs() < 1e-8);
        assert!((nee + bsdf - 1.0).abs() < 1e-12);
        assert!(allows_surface_vertex(0, 0));
        assert!(!allows_surface_vertex(1, 0));
        assert!(allows_surface_vertex(6, 6));
        assert!(!allows_surface_vertex(7, 6));
    }

    #[test]
    fn mis_is_complementary_and_metals_have_no_cosine_reservation() {
        use standard_surface_bsdf::sampling::power_heuristic;
        for (a, b) in [(0.25, 0.75), (1e-5, 2e-5), (1.0, 0.0)] {
            assert!((power_heuristic(a, b) + power_heuristic(b, a) - 1.0).abs() < 1e-6);
        }
        assert_eq!(power_heuristic(0.0, 0.0), 0.0);
        assert_eq!(fast_spec_probability(0.8, 0.0, 0.0), 1.0);
        assert_eq!(fast_spec_probability(0.0, 0.0, 0.0), 1.0);
        assert_eq!(fast_spec_probability(0.0, 1.0, 1.0), 0.15);
        assert_eq!(fast_spec_probability(1.0, 0.0, 1.0), 0.9);
        let expected = 0.04_f32 / (0.04 + 0.4 + 1.0e-4);
        assert_eq!(
            fast_spec_probability(0.04, 0.4, 0.5),
            expected.clamp(0.15, 0.9)
        );
        assert_eq!(fast_ggx_pdf(1.0, 1.0, 0.0), 0.0);
    }

    fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    // Independent f64 GGX/Smith oracle, deliberately not the production PDF helper.
    fn terms(wo: [f64; 3], wi: [f64; 3], alpha: f64) -> (f64, f64, f64) {
        let mut h = [wo[0] + wi[0], wo[1] + wi[1], wo[2] + wi[2]];
        let length = dot(h, h).sqrt();
        for v in &mut h {
            *v /= length;
        }
        let nh = h[2].max(0.0);
        let vh = dot(wo, h).max(0.0);
        let a2 = alpha * alpha;
        let d = a2 / (PI * (nh * nh * (a2 - 1.0) + 1.0).powi(2));
        let g1 = |mu: f64| 2.0 * mu / (mu + (a2 + (1.0 - a2) * mu * mu).sqrt());
        let f_cos = d * g1(wo[2]) * g1(wi[2]) / (4.0 * wo[2]);
        let density = d * g1(wo[2]) / (4.0 * wo[2]);
        (f_cos, density, d * nh / (4.0 * vh.max(1e-30)))
    }

    fn integrate(wo: [f64; 3], alpha: f64) -> (f64, f64) {
        let (rows, columns) = (384, 768);
        let mut energy = 0.0;
        let mut mass = 0.0;
        for row in 0..rows {
            let mu = (row as f64 + 0.5) / rows as f64;
            let radius = (1.0 - mu * mu).sqrt();
            for column in 0..columns {
                let phi = 2.0 * PI * (column as f64 + 0.5) / columns as f64;
                let (s, c) = phi.sin_cos();
                let (f, pdf, _) = terms(wo, [radius * c, radius * s, mu], alpha);
                energy += f;
                mass += pdf;
            }
        }
        let omega = 2.0 * PI / (rows * columns) as f64;
        (energy * omega, mass * omega)
    }

    fn uniform(state: &mut u64) -> f32 {
        // SplitMix64 keeps the test sequence independent of the renderer's counter RNG.
        *state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        (((z ^ (z >> 31)) >> 40) as f32) * (1.0 / 16_777_216.0)
    }

    #[test]
    fn grazing_vndf_matches_independent_pdf_and_energy_with_less_variance() {
        let nv = 0.08_f64;
        let alpha = 0.35_f64;
        let wo = [(1.0 - nv * nv).sqrt(), 0.0, nv];
        let wo_f = wo.map(|v| v as f32);
        let (reference, expected_acceptance) = integrate(wo, alpha);
        let samples = 131_072;
        let mut state = 0x71359a52;
        let mut moments = [[0.0_f64; 2]; 2];
        let mut accepted = [0; 2];
        for _ in 0..samples {
            let xi = [uniform(&mut state), uniform(&mut state)];
            let vndf = fast_ggx_sample_local(wo_f, alpha as f32, xi).map(f64::from);
            // Historical NDF sampler, in f64, with exactly the same BRDF and uniforms.
            let cos_h = ((1.0 - f64::from(xi[0]))
                / (1.0 + (alpha * alpha - 1.0) * f64::from(xi[0])))
            .sqrt();
            let sin_h = (1.0 - cos_h * cos_h).sqrt();
            let (s, c) = (2.0 * PI * f64::from(xi[1])).sin_cos();
            let h = [sin_h * c, sin_h * s, cos_h];
            let vh = dot(wo, h);
            let ndf = [
                2.0 * vh * h[0] - wo[0],
                2.0 * vh * h[1] - wo[1],
                2.0 * vh * h[2] - wo[2],
            ];
            for (kind, wi) in [vndf, ndf].into_iter().enumerate() {
                let weight = if wi[2] > 0.0 {
                    accepted[kind] += 1;
                    let (f, visible_pdf, old_pdf) = terms(wo, wi, alpha);
                    if kind == 0 {
                        // The shared shader PDF is checked against independent f64 arithmetic.
                        let g1 = 2.0 * nv
                            / (nv + (alpha * alpha + (1.0 - alpha * alpha) * nv * nv).sqrt());
                        let d = visible_pdf * (4.0 * nv) / g1;
                        let actual = f64::from(fast_ggx_pdf(d as f32, g1 as f32, nv as f32));
                        assert!((actual - visible_pdf).abs() < 2e-6 * visible_pdf.max(1.0));
                        f / actual
                    } else {
                        f / old_pdf
                    }
                } else {
                    0.0
                };
                moments[kind][0] += weight;
                moments[kind][1] += weight * weight;
            }
        }
        let mean = moments.map(|m| m[0] / samples as f64);
        let variance =
            std::array::from_fn::<_, 2, _>(|i| moments[i][1] / samples as f64 - mean[i] * mean[i]);
        assert!(
            (mean[0] - reference).abs() < 0.006,
            "VNDF {mean:?}, reference {reference}"
        );
        assert!(
            (mean[1] - reference).abs() < 0.02,
            "NDF {mean:?}, reference {reference}"
        );
        assert!((accepted[0] as f64 / samples as f64 - expected_acceptance).abs() < 0.006);
        assert!(accepted[0] > accepted[1], "acceptance {accepted:?}");
        assert!(variance[0] < variance[1] * 0.5, "variance {variance:?}");
    }

    /// Camera basis of scene.rs `pack`: forward -Z, right +X, up +Y, no environment rotation.
    #[test]
    fn env_uv_increases_as_the_view_turns_right() {
        let near =
            |a: [f32; 2], b: [f32; 2]| (a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6;
        assert!(near(env_uv([0.0, 0.0, -1.0], 0.0), [0.25, 0.5]), "forward");
        assert!(
            near(env_uv([1.0, 0.0, 0.0], 0.0), [0.5, 0.5]),
            "right 90 deg"
        );
        assert!(near(env_uv([0.0, 0.0, 1.0], 0.0), [0.75, 0.5]), "back");
        assert!(
            near(env_uv([-1.0, 0.0, 0.0], 0.0), [0.0, 0.5]),
            "left 90 deg"
        );
        assert!(env_uv([0.0, 1.0, 0.0], 0.0)[1].abs() < 1e-6, "up is v = 0");
        // Sweep the view from forward to the right: u must rise monotonically.
        let mut last = env_uv([0.0, 0.0, -1.0], 0.0)[0];
        for k in 1..=90 {
            let a = (k as f32).to_radians();
            let u = env_uv([a.sin(), 0.0, -a.cos()], 0.0)[0];
            assert!(u > last, "u fell at {k} deg: {last} -> {u}");
            last = u;
        }
        // Positive rotation shifts u up by rotation / 2pi.
        let shifted = env_uv([0.0, 0.0, -1.0], std::f32::consts::FRAC_PI_2)[0];
        assert!((shifted - 0.5).abs() < 1e-6, "rotation shift {shifted}");
    }

    #[test]
    fn env_dir_inverts_env_uv() {
        for rotation in [0.0_f32, 0.7, -2.1, 5.0] {
            for iu in 0..64 {
                for iv in 1..32 {
                    let (u, v) = ((iu as f32 + 0.5) / 64.0, iv as f32 / 32.0);
                    let d = env_dir(u, (v * std::f32::consts::PI).cos(), rotation);
                    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                    assert!((len - 1.0).abs() < 1e-5);
                    let [u2, v2] = env_uv(d, rotation);
                    let du = (u2 - u).abs().min(1.0 - (u2 - u).abs());
                    assert!(
                        du < 1e-4 && (v2 - v).abs() < 1e-4,
                        "rot {rotation}: {u},{v} -> {u2},{v2}"
                    );
                }
            }
        }
    }

    /// `map_sample` draws a direction inside texel `i` and `map_pdf` reads texel `map_index(dir)`:
    /// both must name the same texel, otherwise the sampled density is not the lookup density.
    #[test]
    fn sampled_direction_lands_in_the_texel_whose_pdf_it_uses() {
        let (w, h) = (16usize, 8usize);
        for rotation in [0.0_f32, 1.3, -0.9] {
            for i in 0..w * h {
                for (jitter, vv) in [(0.1_f32, 0.2_f32), (0.5, 0.5), (0.9, 0.8)] {
                    let pi = std::f32::consts::PI;
                    let u = ((i % w) as f32 + jitter) / w as f32;
                    let t0 = pi * (i / w) as f32 / h as f32;
                    let t1 = pi * (i / w + 1) as f32 / h as f32;
                    let y = t0.cos() + (t1.cos() - t0.cos()) * vv;
                    let [lu, lv] = env_uv(env_dir(u, y, rotation), rotation);
                    let found = ((lv * h as f32) as usize).min(h - 1) * w
                        + ((lu * w as f32) as usize).min(w - 1);
                    assert_eq!(found, i, "rot {rotation} jitter {jitter} v {vv}");
                }
            }
        }
    }
}
