//! Spherical inversion pre-transform of a fractal's object point (Mandelbulber's "T>Spherical
//! Invert", Luca GN's M3D transform), blended by a strength so it can be animated in from the
//! identity. One definition for the kernels (the map and its distance correction, [`apply`]) and
//! the hosts (the bounding sphere of the transformed set, [`reach`]).
//!
//! With `u = p - centre`, `r = |u|`, radius `rho` and strength `s`, the map is radial about the
//! centre:
//!
//! ```text
//! T(p) = centre + g(r) u,    g(r) = (1 - s) + s rho^2 / r^2
//! ```
//!
//! `s = 1` is the inversion in the sphere (centre, rho): `T(p) = centre + rho^2 u / r^2`; `s = 0`
//! the identity. The Jacobian `g I + r g'(r) u u^T / r^2` has the eigenvalue `g` on the two
//! tangential directions and `(1 - s) - s rho^2 / r^2` on the radial one, so for `s` in [0, 1] its
//! largest stretch is `g`. A distance `d'` estimated in the transformed space therefore bounds the
//! distance at `p` by `d' / g` to first order: the same correction Mandelbulber applies by
//! scaling its running derivative by `1 / r^2`.

/// `T(p)` and the map's largest local stretch `g` (module docs). `strength` in (0, 1]; a host
/// writes 0 for "off" and the kernels then skip the call, so off stays the former estimate.
#[inline(always)]
pub fn apply(p: [f32; 3], centre: [f32; 3], radius: f32, strength: f32) -> ([f32; 3], f32) {
    let u = [p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]];
    // The centre itself maps to infinity; the floor keeps that single point finite (and far).
    let r2 = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).max(1.0e-30);
    let g = (1.0 - strength) + strength * radius * radius / r2;
    (
        [centre[0] + g * u[0], centre[1] + g * u[1], centre[2] + g * u[2]],
        g,
    )
}

/// The radius of a ball about the origin that holds every point `p` with `|T(p)| <= ball`: the
/// object-space bounding sphere of a set bounded by `ball` in the transformed space (the escape
/// radius, a formula's bound). `None` when that set is unbounded: the full inversion (`strength`
/// 1) with its centre inside the ball turns the ball inside out.
///
/// `|T(p)| <= ball` needs the radial magnitude `m(r) = r g(r) = (1 - s) r + s rho^2 / r` within
/// `ball` of `D = |centre|` (triangle inequality), so `m(r) <= D + ball`. For `s < 1`, `m` grows
/// without bound and the largest such `r` is the larger root of
/// `(1 - s) r^2 - (D + ball) r + s rho^2 = 0` (no root: nothing reaches the ball). For `s = 1`,
/// `m = rho^2 / r` falls, and `m >= D - ball` bounds `r` only when `D > ball`.
pub fn reach(centre: [f32; 3], radius: f32, strength: f32, ball: f32) -> Option<f32> {
    if strength == 0.0 {
        return Some(ball);
    }
    let d = f64::from(centre[0]).hypot(f64::from(centre[1])).hypot(f64::from(centre[2]));
    let (s, rho2, ball) = (
        f64::from(strength),
        f64::from(radius) * f64::from(radius),
        f64::from(ball),
    );
    let r = if s >= 1.0 {
        if d <= ball {
            return None;
        }
        rho2 / (d - ball)
    } else {
        let far = d + ball;
        let disc = far * far - 4.0 * (1.0 - s) * s * rho2;
        if disc < 0.0 {
            return Some(0.0);
        }
        (far + disc.sqrt()) / (2.0 * (1.0 - s))
    };
    Some((d + r) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn length(v: [f32; 3]) -> f32 {
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    #[test]
    fn full_strength_is_the_sphere_inversion_and_an_involution() {
        let (c, rho) = ([0.3, -0.2, 0.5], 1.7);
        let p = [1.1, 0.4, -0.9];
        let (q, g) = apply(p, c, rho, 1.0);
        let u = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
        let v = [q[0] - c[0], q[1] - c[1], q[2] - c[2]];
        // |u| |v| = rho^2 along the same ray, and g = rho^2 / |u|^2.
        assert!((length(u) * length(v) - rho * rho).abs() < 1.0e-5);
        assert!((g - rho * rho / (length(u) * length(u))).abs() < 1.0e-5);
        let (back, _) = apply(q, c, rho, 1.0);
        assert!(length([back[0] - p[0], back[1] - p[1], back[2] - p[2]]) < 1.0e-5);
    }

    /// `g` is the largest stretch: no small displacement grows by more than `g` (plus the
    /// finite-difference slack), and a tangential one grows by about `g`.
    #[test]
    fn stretch_bounds_the_local_scale() {
        let (c, rho) = ([0.0, 0.0, 0.2], 0.8);
        for s in [0.25, 0.5, 0.9, 1.0] {
            for p in [[0.5, 0.1, 0.3], [-1.3, 0.7, 2.0], [0.05, -0.02, 0.21]] {
                let (q, g) = apply(p, c, rho, s);
                let h = 1.0e-3 * length([p[0] - c[0], p[1] - c[1], p[2] - c[2]]);
                for axis in 0..3 {
                    let mut e = p;
                    e[axis] += h;
                    let (qe, _) = apply(e, c, rho, s);
                    let moved = length([qe[0] - q[0], qe[1] - q[1], qe[2] - q[2]]);
                    assert!(moved <= g * h * 1.01, "s {s} p {p:?} axis {axis}");
                }
            }
        }
    }

    /// Points sampled along rays from the centre: wherever `|T(p)| <= ball`, `|p| <= reach`.
    #[test]
    fn reach_holds_every_point_that_maps_into_the_ball() {
        let ball = 1.2;
        for (c, rho, s) in [
            ([0.0, 0.0, 0.0], 1.0, 0.5),
            ([0.0, 0.0, 3.0], 1.0, 1.0),
            ([1.0, -0.5, 2.5], 0.7, 0.8),
            ([0.2, 0.0, 0.1], 2.0, 0.999),
        ] {
            let bound = reach(c, rho, s, ball).expect("bounded");
            let mut inside = 0;
            for k in 0..64 {
                let a = k as f32 * 0.7;
                let b = k as f32 * 1.3;
                let dir = [a.cos() * b.sin(), a.sin() * b.sin(), b.cos()];
                for i in 1..4000 {
                    let r = 1.0e-3 * i as f32 * (1.0 + i as f32 * 0.01);
                    let p = [c[0] + dir[0] * r, c[1] + dir[1] * r, c[2] + dir[2] * r];
                    if length(apply(p, c, rho, s).0) <= ball {
                        inside += 1;
                        assert!(length(p) <= bound * 1.0001, "{c:?} {rho} {s}: {p:?} > {bound}");
                    }
                }
            }
            assert!(inside > 0, "{c:?} {rho} {s}: the probe must find the set");
        }
        assert_eq!(reach([0.0; 3], 1.0, 1.0, ball), None);
        assert_eq!(reach([0.0, 0.0, 5.0], 1.0, 0.0, ball), Some(ball));
    }
}
