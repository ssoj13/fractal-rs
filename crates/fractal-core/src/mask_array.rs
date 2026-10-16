//! Placement of a mask's array copies (`params::M_ARRAY*`): one definition for the kernels (the
//! union of copy distances) and the hosts (wires and gizmos drawn per copy).
//!
//! A linear copy `i` sits at `offset * i`. A radial copy `i` of `count` sits on a ring of
//! `radius` about an axis through the mask origin, at [`angle`]; an aligned radial box turns into
//! the copy's [`ring_basis`] so its local X faces out from the ring (Blender's Array Circle).
//! Sine and cosine are the caller's: the kernels use their device `sin_cos`, the hosts `std`.

use crate::params::ARRAY_MAX;

/// The packed copy count clamped to what the kernels union: at least the shape itself, at most
/// [`ARRAY_MAX`].
#[inline(always)]
pub fn count(packed: u32) -> u32 {
    packed.clamp(1, ARRAY_MAX)
}

/// Angle of radial copy `index` of `count`, in radians. A `sweep` of 0 is a full circle (no copy
/// on top of the first); otherwise the copies span `sweep` end to end.
#[inline(always)]
pub fn angle(index: u32, count: u32, sweep: f32) -> f32 {
    let i = index as f32;
    if sweep == 0.0 {
        i * core::f32::consts::TAU / count as f32
    } else if count <= 1 {
        0.0
    } else {
        i * sweep / (count as f32 - 1.0)
    }
}

/// Centre of the radial copy at angle (`sn`, `cs`) on a ring of `radius` about `axis`
/// (0 X, 1 Y, 2 Z; larger codes read as Z).
#[inline(always)]
pub fn ring_point(axis: u32, radius: f32, sn: f32, cs: f32) -> [f32; 3] {
    match axis {
        0 => [0.0, radius * cs, radius * sn],
        1 => [radius * cs, 0.0, radius * sn],
        _ => [radius * cs, radius * sn, 0.0],
    }
}

/// The aligned copy's local axes as rows (outward, up = the ring axis, tangent): a right-handed
/// rotation, so `local = rows * (p - centre)` and its transpose places the copy in mask space.
#[inline(always)]
pub fn ring_basis(axis: u32, sn: f32, cs: f32) -> [[f32; 3]; 3] {
    match axis {
        0 => [[0.0, cs, sn], [1.0, 0.0, 0.0], [0.0, sn, -cs]],
        1 => [[cs, 0.0, sn], [0.0, 1.0, 0.0], [-sn, 0.0, cs]],
        _ => [[cs, sn, 0.0], [0.0, 0.0, 1.0], [sn, -cs, 0.0]],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    #[test]
    fn a_full_circle_spaces_copies_evenly_and_a_sweep_spans_its_ends() {
        let tau = core::f32::consts::TAU;
        assert_eq!(angle(0, 4, 0.0), 0.0);
        assert!((angle(1, 4, 0.0) - tau / 4.0).abs() < 1e-6);
        assert!((angle(3, 4, 1.5) - 1.5).abs() < 1e-6);
        assert_eq!(angle(0, 1, 1.5), 0.0);
        assert_eq!((count(0), count(5), count(1000)), (1, 5, ARRAY_MAX));
    }

    #[test]
    fn the_ring_basis_is_right_handed_faces_out_and_keeps_the_axis_up() {
        let (sn, cs) = 0.7_f32.sin_cos();
        for axis in 0..3 {
            let [out, up, tan] = ring_basis(axis, sn, cs);
            let centre = ring_point(axis, 2.0, sn, cs);
            let along: f32 = (0..3).map(|k| out[k] * centre[k]).sum();
            assert!((along - 2.0).abs() < 1e-6, "outward points at the copy");
            assert_eq!(up[axis as usize], 1.0);
            let c = cross(out, up);
            assert!((0..3).all(|k| (c[k] - tan[k]).abs() < 1e-6), "axis {axis}");
        }
    }
}
