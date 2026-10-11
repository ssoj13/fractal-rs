//! Allocation-free interface tracking and interior marching, shared by CUDA and CPU tests.
//!
//! Exterior-only DEs report zero throughout a solid. Reusing the exterior marcher there
//! immediately hits the entrance again; an absolute-distance step also cannot advance.
//! Probe a bounded interval instead, then refine the first inside-to-outside bracket.
//! Unsigned fields use the entrance pixel tolerance as their resolved solid boundary.
//! Probe spacing is the interval over the caller's probe count (`Render::glass_probes`):
//! sub-probe cavities are not guaranteed to be found. This models one connected medium at a time, not nested dielectrics.

pub const AIR: u32 = u32::MAX;

/// Only an actual hemisphere crossing changes the medium; reflection and TIR retain it.
#[inline(always)]
pub fn medium_after_scatter(current: u32, object: u32, crossed: bool) -> u32 {
    if !crossed {
        current
    } else if current == object {
        AIR
    } else {
        object
    }
}

/// Locate the exit of the currently occupied object. None means unresolved/invalid,
// not environment visibility; the integrator must terminate that path without a sky leak.
#[inline(always)]
pub fn exit_distance(
    mut estimate: impl FnMut(f32) -> f32,
    extent: f32,
    tolerance: f32,
    steps: u32,
    step_factor: f32,
    signed: bool,
) -> Option<f32> {
    if !(extent > 0.0 && tolerance > 0.0) || steps == 0 {
        return None;
    }
    let threshold = if signed { 0.0 } else { tolerance };
    let minimum_step = (extent / steps as f32).max(tolerance * 0.5);
    let mut t = 0.0;
    let mut previous = 0.0;
    let mut i = 0;
    while i <= steps {
        let d = estimate(t);
        if !d.is_finite() {
            return None;
        }
        if d > threshold {
            if i == 0 {
                return Some(0.0);
            }
            let mut lo = previous;
            let mut hi = t;
            let mut refinement = 0;
            while refinement < 24 && hi - lo > tolerance * 0.25 {
                let mid = (lo + hi) * 0.5;
                let value = estimate(mid);
                if !value.is_finite() {
                    return None;
                }
                if value > threshold {
                    hi = mid;
                } else {
                    lo = mid;
                }
                refinement += 1;
            }
            return Some((lo + hi) * 0.5);
        }
        if t >= extent {
            break;
        }
        previous = t;
        let advance = if signed {
            (-d * step_factor).max(minimum_step)
        } else {
            minimum_step
        };
        t = (t + advance).min(extent);
        i += 1;
    }
    None
}

/// Beer-Lambert transmittance: color is the fraction of light remaining after
/// traveling reference_depth world units. Depth zero selects interface tinting instead.
#[inline(always)]
pub fn volume_transmittance(color: [f32; 3], reference_depth: f32, distance: f32) -> [f32; 3] {
    if reference_depth <= 0.0 || distance <= 0.0 {
        return [1.0; 3];
    }
    let exponent = distance / reference_depth;
    [
        color[0].clamp(0.0, 1.0).powf(exponent),
        color[1].clamp(0.0, 1.0).powf(exponent),
        color[2].clamp(0.0, 1.0).powf(exponent),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bottle_green_absorption_depends_on_distance_not_interface_count() {
        let color = [0.12, 0.82, 0.25];
        assert_eq!(volume_transmittance(color, 0.5, 0.0), [1.0; 3]);
        assert_eq!(volume_transmittance(color, 0.5, 0.5), color);
        let thick = volume_transmittance(color, 0.5, 1.0);
        let half = volume_transmittance(color, 0.5, 0.25);
        for channel in 0..3 {
            assert!((thick[channel] - color[channel] * color[channel]).abs() < 1e-6);
            assert!((half[channel] * half[channel] - color[channel]).abs() < 1e-6);
        }
        assert_eq!(
            volume_transmittance([0.0, 1.0, 0.0], 1.0, 2.0),
            [0.0, 1.0, 0.0]
        );
        assert_eq!(volume_transmittance(color, 0.0, 100.0), [1.0; 3]);
    }

    #[test]
    fn unsigned_solid_exits_at_far_interface_instead_of_rehitting_entrance() {
        let exit = exit_distance(|t| (t - 1.8).max(0.0), 2.0, 0.001, 256, 0.85, false).unwrap();
        assert!((exit - 1.801).abs() < 0.00025, "{exit}");
    }
    #[test]
    fn signed_solid_and_internal_reflection_find_the_next_interface() {
        let exit = exit_distance(|t| t - 1.8, 2.0, 0.001, 256, 0.85, true).unwrap();
        assert!((exit - 1.8).abs() < 0.00025);
        let entered = medium_after_scatter(AIR, 7, true);
        assert_eq!(entered, 7);
        assert_eq!(medium_after_scatter(entered, 7, false), entered);
        assert_eq!(medium_after_scatter(entered, 7, true), AIR);
    }
    #[test]
    fn first_resolved_cavity_is_the_exit_not_the_bounding_sphere() {
        let exit = exit_distance(
            |t| if t > 0.6 && t < 0.8 { 0.1 } else { 0.0 },
            2.0,
            0.001,
            256,
            0.85,
            false,
        )
        .unwrap();
        assert!((exit - 0.6).abs() < 0.00025);
    }
    #[test]
    fn exhausted_or_invalid_search_never_claims_environment_visibility() {
        assert_eq!(exit_distance(|_| 0.0, 2.0, 0.001, 256, 0.85, false), None);
        assert_eq!(
            exit_distance(|_| f32::NAN, 2.0, 0.001, 256, 0.85, false),
            None
        );
        assert_eq!(exit_distance(|_| 0.0, 2.0, 0.001, 0, 0.85, false), None);
    }
}
