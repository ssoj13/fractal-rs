//! Low-discrepancy sampler shared by the CUDA integrator and its CPU tests: Owen-scrambled Sobol
//! (Burley 2020, "Practical Hash-based Owen Scrambling", JCGT 9(4)). Device-safe: no tables in
//! memory (direction numbers are compile-time constants), no allocation, integer arithmetic only.
//!
//! Each pixel gets an independently scrambled and shuffled Sobol sequence over its sample index,
//! so the estimator stays unbiased while the samples of one pixel are stratified (lower error
//! than white noise at equal sample counts, most at low counts and early bounces). Dimensions are
//! consumed in groups of four: the four dimensions of a group are jointly stratified (the first
//! four Sobol dimensions), different groups are decorrelated by their own seed (padding).

/// Sobol direction numbers of one dimension from its Joe-Kuo parameters
/// (`s` degree, `a` coefficients, `m` initial numbers).
const fn directions(s: usize, a: u32, m: [u32; 3]) -> [u32; 32] {
    let mut v = [0u32; 32];
    if s == 0 {
        // Dimension 0: van der Corput.
        let mut i = 0;
        while i < 32 {
            v[i] = 1 << (31 - i);
            i += 1;
        }
        return v;
    }
    let mut i = 0;
    while i < 32 {
        if i < s {
            v[i] = m[i] << (31 - i);
        } else {
            let mut x = v[i - s] ^ (v[i - s] >> s);
            let mut k = 1;
            while k < s {
                if (a >> (s - 1 - k)) & 1 == 1 {
                    x ^= v[i - k];
                }
                k += 1;
            }
            v[i] = x;
        }
        i += 1;
    }
    v
}

/// The first four Sobol dimensions (new-joe-kuo-6.21201, dimensions 2..4 after van der Corput).
const SOBOL: [[u32; 32]; 4] = [
    directions(0, 0, [0; 3]),
    directions(1, 0, [1, 0, 0]),
    directions(2, 1, [1, 3, 0]),
    directions(3, 1, [1, 3, 1]),
];

/// Sobol point `index` in dimension `dim` (0..4), as 32 fraction bits.
#[inline(always)]
fn sobol(mut index: u32, dim: usize) -> u32 {
    let v = &SOBOL[dim];
    let mut x = 0u32;
    let mut bit = 0;
    while index != 0 {
        if index & 1 == 1 {
            x ^= v[bit];
        }
        index >>= 1;
        bit += 1;
    }
    x
}

/// Laine-Karras style hash permutation (Burley 2020, listing 3): a nested uniform scramble of
/// the bit-reversed value.
#[inline(always)]
fn lk_permutation(mut x: u32, seed: u32) -> u32 {
    x = x.wrapping_add(seed);
    x ^= x.wrapping_mul(0x6c50_b47c);
    x ^= x.wrapping_mul(0xb82f_1e52);
    x ^= x.wrapping_mul(0xc7af_e638);
    x ^= x.wrapping_mul(0x8d22_f6e6);
    x
}

/// Owen scramble of a 32-bit fraction (nested uniform scrambling).
#[inline(always)]
fn owen(x: u32, seed: u32) -> u32 {
    lk_permutation(x.reverse_bits(), seed).reverse_bits()
}

/// Integer hash (lowbias32, Ellis) used to derive independent seeds.
#[inline(always)]
fn hash(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x21f0_aaad);
    x ^= x >> 15;
    x = x.wrapping_mul(0x735a_2d97);
    x ^= x >> 15;
    x
}

#[inline(always)]
fn hash3(a: u32, b: u32, c: u32) -> u32 {
    hash(a ^ hash(b ^ hash(c)))
}

/// Sample `index` of pixel (`px`, `py`) in dimension `dim`, uniform in [0, 1).
/// `px` already carries the frame seed (as the white-noise generator did).
#[inline(always)]
pub fn sample(px: u32, py: u32, index: u32, dim: u32) -> f32 {
    let group = dim >> 2;
    let seed = hash3(px, py, group);
    // Shuffle the sample order per pixel and group, then scramble each dimension.
    let shuffled = owen(index, seed);
    let bits = owen(sobol(shuffled, (dim & 3) as usize), hash(seed ^ dim));
    // 24 bits keep the result exactly representable below 1.0.
    (bits >> 8) as f32 * (1.0 / 16_777_216.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_dimensions_are_the_published_sobol_points() {
        // Unscrambled Sobol (0,2)-sequence: index 1..4 in dimensions 0 and 1.
        let point = |i, d| sobol(i, d) as f64 / 4_294_967_296.0;
        assert_eq!([point(1, 0), point(2, 0), point(3, 0)], [0.5, 0.25, 0.75]);
        assert_eq!([point(1, 1), point(2, 1), point(3, 1)], [0.5, 0.75, 0.25]);
        assert_eq!(point(1, 2), 0.5);
    }

    #[test]
    fn samples_are_in_range_and_stratified_per_pixel() {
        // Any 16 consecutive samples of one pixel hit every 1/4 x 1/4 cell of a dimension pair once
        // (scrambled and shuffled (0,2)-net), unlike white noise.
        for (px, py) in [(0, 0), (17, 3), (1023, 511)] {
            for dims in [(0, 1), (4, 5), (2, 3)] {
                let mut cells = [0u32; 16];
                for i in 0..16 {
                    let (u, v) = (sample(px, py, i, dims.0), sample(px, py, i, dims.1));
                    assert!((0.0..1.0).contains(&u) && (0.0..1.0).contains(&v));
                    cells[(u * 4.0) as usize * 4 + (v * 4.0) as usize] += 1;
                }
                assert!(
                    cells.iter().all(|&c| c == 1),
                    "pixel {px},{py} dims {dims:?}: {cells:?}"
                );
            }
        }
    }

    #[test]
    fn pixels_and_groups_are_decorrelated_and_uniform() {
        let n = 1 << 14;
        let mut mean = [0.0f64; 3];
        let mut equal = 0;
        for i in 0..n {
            let a = sample(i, 7, 3, 0);
            let b = sample(i, 7, 3, 4);
            mean[0] += f64::from(a);
            mean[1] += f64::from(b);
            mean[2] += f64::from(a) * f64::from(b);
            equal += u32::from(a == sample(i + 1, 7, 3, 0));
        }
        let [ma, mb, mab] = mean.map(|m| m / f64::from(n));
        assert!(
            (ma - 0.5).abs() < 0.01 && (mb - 0.5).abs() < 0.01,
            "{ma} {mb}"
        );
        assert!(
            (mab - ma * mb).abs() < 0.01,
            "groups must be independent: {mab}"
        );
        assert!(equal < 4, "neighbouring pixels must not share samples");
    }
}
