use super::*;
use fractal_core::params::P_COUNT;

/// The kernels' sky gradient (`sky_radiance`): horizon to zenith by `0.5 + 0.5 y`.
fn gradient(horizon: [f32; 3], zenith: [f32; 3], intensity: f32, d: [f32; 3]) -> [f32; 3] {
    let t = (0.5 + 0.5 * d[1]).clamp(0.0, 1.0);
    std::array::from_fn(|c| (horizon[c] + (zenith[c] - horizon[c]) * t) * intensity)
}

/// The exact cosine convolution of that gradient over pi (the kernels' `sky_irradiance`).
fn gradient_irradiance(
    horizon: [f32; 3],
    zenith: [f32; 3],
    intensity: f32,
    n: [f32; 3],
) -> [f32; 3] {
    let t = 0.5 + n[1] / 3.0;
    std::array::from_fn(|c| (horizon[c] + (zenith[c] - horizon[c]) * t) * intensity)
}

fn texel_dir(x: u32, y: u32, w: u32, h: u32) -> [f32; 3] {
    equirect_uv_to_dir([(x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32])
}

/// An affine radiance has no band above 1, so the L2 harmonics reproduce its exact convolution
/// up to the map's discretisation; its solid-angle mean is its value on the horizon.
#[test]
fn harmonics_of_the_gradient_are_the_exact_convolution() {
    let (horizon, zenith, intensity) = ([0.2, 0.4, 0.9], [1.5, 1.1, 0.3], 2.0);
    let (w, h) = (256, 128);
    let texels = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .map(|(x, y)| gradient(horizon, zenith, intensity, texel_dir(x, y, w, h)))
        .collect();
    let map = EnvMap::new(w, h, texels, Luma::AcesCg).unwrap();
    for n in [
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.48, 0.6, 0.64],
        [-0.6, -0.48, 0.64],
    ] {
        let sh = map.irradiance(n);
        let exact = gradient_irradiance(horizon, zenith, intensity, n);
        for c in 0..3 {
            assert!(
                (sh[c] - exact[c]).abs() <= 2e-3 * exact[c].max(0.1),
                "n {n:?} channel {c}: {} vs {}",
                sh[c],
                exact[c]
            );
        }
    }
    let mid = Luma::AcesCg.of(gradient(horizon, zenith, intensity, [1.0, 0.0, 0.0]));
    assert!(
        (map.mean() - mid).abs() < 1e-3 * mid,
        "{} vs {mid}",
        map.mean()
    );
}

/// A mirror reads level 0, a rough lobe blurred levels; the mips halve to 1 x 1; the GPU block
/// holds level 0 with the conditional CDF, the marginal, the harmonics, then the levels.
#[test]
fn specular_levels_follow_the_lobe_width_and_the_block_layout_holds() {
    let (w, h) = (64u32, 32u32);
    let mut texels = vec![[0.5f32, 0.25, 0.125]; (w * h) as usize];
    texels[(8 * w + 20) as usize] = [400.0, 300.0, 200.0];
    let map = EnvMap::new(w, h, texels, Luma::AcesCg).unwrap();
    assert_eq!(map.level_count(), 7);
    assert_eq!(map.lod(0.0), 0.0);
    assert_eq!(map.lod(10.0), 6.0);
    assert!((map.lod(0.5) - (0.5 * 32.0 / std::f32::consts::PI).log2()).abs() < 1e-6);
    let spot = texel_dir(20, 8, w, h);
    let sharp = map.specular(spot, 0.0);
    assert_eq!(sharp, map.equirect().lookup(dir_to_equirect_uv(spot)));
    let blurred = map.specular(spot, 0.5);
    assert!(
        blurred[0] < 0.25 * sharp[0] && blurred[0] > 0.5,
        "{blurred:?}"
    );
    let last = map.levels.last().unwrap();
    assert_eq!((last.width, last.height), (1, 1));
    let mean = (0.5 * (w * h - 1) as f32 + 400.0) / (w * h) as f32;
    assert!((last.texels[0][0] - mean).abs() < 1e-4 * mean);
    let block: Vec<[f32; 4]> = map.gpu_entries().collect();
    assert_eq!(block.len(), map.gpu_len());
    let level_entries: usize = map.levels.iter().map(|l| l.texels.len()).sum();
    assert_eq!(block.len(), (w * h + h + 9) as usize + level_entries);
    assert_eq!(
        block[(8 * w + 20) as usize][..3],
        map.equirect().texels()[(8 * w + 20) as usize]
    );
    assert_eq!(
        block[(w * h - 1) as usize][3],
        map.equirect().conditional()[(w * h - 1) as usize]
    );
    assert_eq!(block[(w * h) as usize][0], map.equirect().marginal()[0]);
    assert_eq!(block[(w * h + h) as usize][0], map.sh[0][0]);
}

/// Invalid texels are refused with their count, never clamped; radiance beyond the f16 range (an
/// unclipped sun) is kept exact.
#[test]
fn invalid_texels_are_refused() {
    let error = |texels: Vec<[f32; 3]>| EnvMap::new(2, 1, texels, Luma::AcesCg).unwrap_err();
    assert_eq!(
        error(vec![[f32::NAN, 0.0, 0.0], [1.0; 3]]),
        EnvError::NonFinite { count: 1 }
    );
    assert_eq!(
        error(vec![[-0.1, 0.0, 0.0], [-1.0; 3]]),
        EnvError::Negative { count: 2 }
    );
    let sun = EnvMap::new(
        2,
        1,
        vec![[150_000.0, 90_000.0, 1.0], [1.0; 3]],
        Luma::AcesCg,
    )
    .unwrap();
    assert_eq!(sun.equirect().texels()[0], [150_000.0, 90_000.0, 1.0]);
    assert_eq!(
        EnvMap::new(0, 1, Vec::new(), Luma::AcesCg).unwrap_err(),
        EnvError::Size {
            width: 0,
            height: 1
        }
    );
    assert!(matches!(
        EnvMap::new(2, 2, vec![[1.0; 3]; 3], Luma::AcesCg),
        Err(EnvError::Size { .. })
    ));
}

/// The L2 sum of a sun-dominated map rings below 0 about 122 degrees from the sun; the irradiance
/// Direct reads is never negative and matches the exact convolution facing the sun.
#[test]
fn irradiance_of_a_sun_map_is_never_negative() {
    let (w, h) = (128u32, 64u32);
    let sun = (40u32, 12u32);
    let mut texels = vec![[0.01f32; 3]; (w * h) as usize];
    texels[(sun.1 * w + sun.0) as usize] = [5000.0; 3];
    let map = EnvMap::new(w, h, texels, Luma::AcesCg).unwrap();
    let sun_dir = texel_dir(sun.0, sun.1, w, h);
    let side = {
        let d = sun_dir[1];
        let t = [-d * sun_dir[0], 1.0 - d * sun_dir[1], -d * sun_dir[2]];
        let l = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
        t.map(|v| v / l)
    };
    let angle = 122f32.to_radians();
    let n: [f32; 3] = std::array::from_fn(|c| angle.cos() * sun_dir[c] + angle.sin() * side[c]);
    assert!(
        map.irradiance_unclamped(n)[0] < 0.0,
        "{:?}",
        map.irradiance_unclamped(n)
    );
    assert_eq!(map.irradiance(n), [0.0; 3]);
    for i in 0..400 {
        let z = 1.0 - 2.0 * (i as f32 + 0.5) / 400.0;
        let r = (1.0 - z * z).sqrt();
        let phi = 2.399_963 * i as f32;
        let n = [r * phi.cos(), z, r * phi.sin()];
        assert!(map.irradiance(n).iter().all(|&v| v >= 0.0), "{n:?}");
    }
    let lit = map.irradiance(sun_dir)[0];
    let t0 = std::f32::consts::PI * sun.1 as f32 / h as f32;
    let t1 = std::f32::consts::PI * (sun.1 + 1) as f32 / h as f32;
    let omega = 2.0 * std::f32::consts::PI / w as f32 * (t0.cos() - t1.cos());
    let exact = 5000.0 * omega / std::f32::consts::PI + 0.01;
    assert!((lit - exact).abs() < 0.25 * exact, "{lit} vs {exact}");
}

/// The orientation both apps show (camera forward -Z, right +X, up +Y): `u` rises as the view
/// turns right, forward is `u = 0.25`, up is `v = 0`, and a positive turn of `r` adds `r / 2 pi`
/// to `u` (the content turns left), as WarpBro's former `env_uv` did.
#[test]
fn map_lookup_orientation_and_turn_sign() {
    let uv = |d: [f32; 3], degrees: f32| dir_to_equirect_uv(to_map(turn(degrees), d));
    let near = |a: [f32; 2], b: [f32; 2]| (a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6;
    assert!(near(uv([0.0, 0.0, -1.0], 0.0), [0.25, 0.5]), "forward");
    assert!(near(uv([1.0, 0.0, 0.0], 0.0), [0.5, 0.5]), "right");
    assert!(near(uv([0.0, 0.0, 1.0], 0.0), [0.75, 0.5]), "back");
    assert!(uv([0.0, 1.0, 0.0], 0.0)[1].abs() < 1e-6, "up");
    let mut last = uv([0.0, 0.0, -1.0], 0.0)[0];
    for k in 1..=90 {
        let a = (k as f32).to_radians();
        let u = uv([a.sin(), 0.0, -a.cos()], 0.0)[0];
        assert!(u > last, "u fell at {k} deg: {last} -> {u}");
        last = u;
    }
    let shifted = uv([0.0, 0.0, -1.0], 90.0)[0];
    assert!((shifted - 0.5).abs() < 1e-6, "turn shift {shifted}");
}

/// `pack` describes the map (or the gradient) in the parameter slots; a sun of no light leaves
/// the map alone to be sampled; otherwise the sun's share stays inside the clamp range.
#[test]
fn pack_fills_the_environment_slots() {
    let map = EnvMap::new(8, 4, vec![[2.0; 3]; 32], Luma::Rec709).unwrap();
    assert!((map.mean() - 2.0).abs() < 1e-5);
    let mut p = vec![-1.0f32; P_COUNT];
    pack(&mut p, Some(&map), 3.0, 90.0, [10.0, 10.0, 10.0]);
    assert_eq!(
        [p[P_ENV_WIDTH], p[P_ENV_HEIGHT], p[P_ENV_LEVELS]],
        [8.0, 4.0, map.level_count() as f32]
    );
    assert_eq!([p[P_ENV_MEAN], p[P_ENV_INTENSITY]], [map.mean(), 3.0]);
    assert!(p[P_ENV_COS].abs() < 1e-7 && (p[P_ENV_SIN] - 1.0).abs() < 1e-7);
    let expected = 10.0 / (10.0 + std::f32::consts::PI * 6.0);
    assert!((p[P_ENV_SUN_SELECT] - expected).abs() < 1e-6);
    assert_eq!(sun_select([0.0; 3], 6.0, Luma::Rec709), 0.0);
    assert_eq!(sun_select([1e6; 3], 1e-3, Luma::Rec709), SUN_SELECT_RANGE.1);
    assert_eq!(sun_select([1e-6; 3], 1e3, Luma::AcesCg), SUN_SELECT_RANGE.0);
    pack(&mut p, None, 1.0, 0.0, [1.0; 3]);
    assert_eq!(
        [p[P_ENV_WIDTH], p[P_ENV_LEVELS], p[P_ENV_SUN_SELECT]],
        [0.0; 3]
    );
}
