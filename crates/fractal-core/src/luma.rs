//! Luminance weights of the host's working space. Every place the tracer weighs colours against
//! each other (sun against sky, light selection, the Fast model's specular probability, adaptive
//! error, tonemap saturation) uses the weights of the space its colours are in: WarpBro renders
//! ACEScg, ofx-rs scene-linear Rec. 709. The host writes them to `P_LUMA`.

/// The working space whose luminance weights the tracer uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Luma {
    /// Linear Rec. 709 / sRGB primaries, D65 (ITU-R BT.709 Y row).
    Rec709,
    /// ACEScg (AP1 primaries): the Y row of AP1 to XYZ, the BSDF's own `SS_LUMA`.
    AcesCg,
}

impl Luma {
    /// The `[r, g, b]` weights; each set sums to one.
    pub const fn weights(self) -> [f32; 3] {
        match self {
            Self::Rec709 => [0.2126, 0.7152, 0.0722],
            Self::AcesCg => standard_surface_bsdf::consts::SS_LUMA,
        }
    }

    /// Luminance of `c` in this space.
    pub fn of(self, c: [f32; 3]) -> f32 {
        let w = self.weights();
        w[0] * c[0] + w[1] * c[1] + w[2] * c[2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_sum_to_one_and_acescg_is_the_bsdf_row() {
        for luma in [Luma::Rec709, Luma::AcesCg] {
            assert!((luma.of([1.0; 3]) - 1.0).abs() < 1e-6, "{luma:?}");
        }
        assert_eq!(
            Luma::AcesCg.weights(),
            standard_surface_bsdf::consts::SS_LUMA
        );
    }
}
