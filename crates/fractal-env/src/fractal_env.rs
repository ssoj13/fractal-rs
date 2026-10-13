//! The HDRI environment of the fractal path tracer on the host: one implementation of the map the
//! kernels sample, shared by WarpBro and ofx-rs `ofx-fractal` (and by OFX's CPU and WGSL engines,
//! which read the same map). Decoding files and caching maps stay with each application: this
//! crate starts from linear texels.
//!
//! [`EnvMap`] builds once from the texels (row 0 = zenith, in the host's working space):
//! - the importance CDFs (render-env `EnvCdfs`: a marginal CDF over rows, a conditional CDF per
//!   row) inside pt-integrator's [`EquirectMap`], the scheme the kernels' `env_map_sample` ports;
//! - the solid-angle mean luminance, which weighs the map against the sun ([`sun_select`]);
//! - the L2 spherical harmonics of the irradiance over pi (Ramamoorthi and Hanrahan 2001) and a
//!   2 x 2 box mip pyramid, which Direct shading reads for diffuse and glossy light.
//!
//! [`EnvMap::gpu_entries`] is the block the kernels read after the palette rows, and [`pack`]
//! fills the parameter slots that describe it; both hosts go through them.
//!
//! Luminance uses the working space's weights the map is built with ([`Luma`]), the ones the host
//! also writes to `P_LUMA` for the kernels, so the sun and the map are weighed with one rule.

pub use fractal_core::luma::Luma;
use fractal_core::params::{
    P_ENV_COS, P_ENV_HEIGHT, P_ENV_INTENSITY, P_ENV_LEVELS, P_ENV_MEAN, P_ENV_SIN,
    P_ENV_SUN_SELECT, P_ENV_WIDTH,
};
use pt_integrator::lighting::{
    EquirectError, EquirectMap, TexelPrecision, bilinear, dir_to_equirect_uv, equirect_uv_to_dir,
};

/// Most texels of a map (8192 x 4096): about 670 MB built (f32 texels, the conditional CDF and
/// the mips). Hosts refuse larger files from the header, before decoding.
pub const MAX_TEXELS: u64 = 8192 * 4096;

/// What a scene's environment path names a map embedded in the binary with: `builtin:<name>`,
/// one of [`BUILTINS`].
pub const BUILTIN_PREFIX: &str = "builtin:";

/// Maps embedded in the binary (name, Radiance `.hdr` bytes), so a scene needs no file for its
/// environment, in every host: Poly Haven's CC0 "Venice Sunset" 2K (`assets/hdri/NOTICE.md`),
/// about 5.6 MB.
pub const BUILTINS: &[(&str, &[u8])] = &[(
    "venice_sunset_2k",
    include_bytes!("../assets/hdri/venice_sunset_2k.hdr"),
)];

/// The default environment of a new scene: the first built-in map, as a path.
pub fn default_path() -> String {
    format!("{BUILTIN_PREFIX}{}", BUILTINS[0].0)
}

/// The Radiance `.hdr` bytes a `builtin:<name>` path names; None for any other path (a file) and
/// for an unknown name.
pub fn builtin(path: &str) -> Option<&'static [u8]> {
    let name = path.strip_prefix(BUILTIN_PREFIX)?;
    BUILTINS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, bytes)| *bytes)
}

/// Range of the probability of sampling the sun cone rather than the sky: neither strategy is
/// ever starved (the kernels' gradient `sun_select` clamps to the same range).
pub const SUN_SELECT_RANGE: (f32, f32) = (0.1, 0.9);

/// Why texels do not make a map. Hosts add the file name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EnvError {
    /// No texels, more than [`MAX_TEXELS`], or a texel count that is not `width * height`.
    Size { width: u32, height: u32 },
    /// Texels with a NaN or infinite component.
    NonFinite { count: usize },
    /// Texels with a negative component (radiance cannot be negative).
    Negative { count: usize },
}

impl std::fmt::Display for EnvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Size { width, height } => {
                write!(f, "{width} x {height} texels, expected 1 to {MAX_TEXELS}")
            }
            Self::NonFinite { count } => write!(f, "{count} texels are NaN or infinite"),
            Self::Negative { count } => write!(f, "{count} texels have negative radiance"),
        }
    }
}

impl std::error::Error for EnvError {}

/// One mip level above level 0: `width x height` rgb texels.
#[derive(Clone, Debug)]
struct Level {
    width: u32,
    height: u32,
    texels: Vec<[f32; 3]>,
}

/// A validated lat-long environment and everything derived from it once (crate docs).
#[derive(Debug)]
pub struct EnvMap {
    luma: Luma,
    equirect: EquirectMap,
    mean: f32,
    sh: [[f32; 3]; 9],
    levels: Vec<Level>,
}

impl EnvMap {
    /// The map of `width * height` linear `texels` (row 0 = zenith) in the working space whose
    /// weights are `luma`: validated (finite, non-negative, at most [`MAX_TEXELS`], never
    /// clamped), CDFs, mean, harmonics and mips built.
    pub fn new(
        width: u32,
        height: u32,
        texels: Vec<[f32; 3]>,
        luma: Luma,
    ) -> Result<Self, EnvError> {
        let n = u64::from(width) * u64::from(height);
        if n == 0 || n > MAX_TEXELS || texels.len() as u64 != n {
            return Err(EnvError::Size { width, height });
        }
        let (mut non_finite, mut negative) = (0, 0);
        for t in &texels {
            if !t.iter().all(|c| c.is_finite()) {
                non_finite += 1;
            } else if t.iter().any(|&c| c < 0.0) {
                negative += 1;
            }
        }
        if non_finite > 0 {
            return Err(EnvError::NonFinite { count: non_finite });
        }
        if negative > 0 {
            return Err(EnvError::Negative { count: negative });
        }
        let cdfs = render_env::EnvCdfs::build(width, height, &texels);
        let equirect = EquirectMap::new(
            width,
            height,
            texels,
            cdfs.marginal,
            cdfs.conditional,
            TexelPrecision::F32,
        )
        .map_err(|error| match error {
            EquirectError::Size => EnvError::Size { width, height },
            // Validated above: every component is finite.
            EquirectError::Range => EnvError::NonFinite { count: 0 },
        })?;
        let (mean, sh) = integrate(&equirect, luma);
        let levels = mips(&equirect);
        Ok(Self {
            luma,
            equirect,
            mean,
            sh,
            levels,
        })
    }

    /// Texels per row.
    pub fn width(&self) -> u32 {
        self.equirect.width()
    }

    /// Rows.
    pub fn height(&self) -> u32 {
        self.equirect.height()
    }

    /// The texels and CDFs (pt-integrator's CPU sampling, `Equirect`).
    pub fn equirect(&self) -> &EquirectMap {
        &self.equirect
    }

    /// The working space the map was built in.
    pub fn luma(&self) -> Luma {
        self.luma
    }

    /// Mean luminance over the sphere (solid-angle weighted), before the intensity.
    pub fn mean(&self) -> f32 {
        self.mean
    }

    /// Mip levels, level 0 included.
    pub fn level_count(&self) -> u32 {
        self.levels.len() as u32 + 1
    }

    /// Irradiance over pi (the `Environment::irradiance` convention) about the unit normal `n` of
    /// the map's frame, from the L2 harmonics, clamped at 0. A band-limited convolution of a
    /// sun-dominated map rings below 0 away from the sun (about -0.16 of the sun's `P / 4 pi` near
    /// 122 degrees from it) where the exact irradiance is small and positive; the clamp keeps
    /// Direct from subtracting light there. Kernel twin: `env_irradiance`.
    pub fn irradiance(&self, n: [f32; 3]) -> [f32; 3] {
        self.irradiance_unclamped(n).map(|v| v.max(0.0))
    }

    /// The harmonic sum of [`Self::irradiance`] before its clamp.
    fn irradiance_unclamped(&self, n: [f32; 3]) -> [f32; 3] {
        let mut e = [0.0f32; 3];
        for (coefficient, y) in self.sh.iter().zip(sh_basis(n)) {
            for c in 0..3 {
                e[c] += coefficient[c] * y;
            }
        }
        e
    }

    /// The fractional mip level whose texel height matches a GGX lobe of `alpha`:
    /// `log2(max(alpha * height / pi, 1))`, at most the last level.
    pub fn lod(&self, alpha: f32) -> f32 {
        let texels = (alpha * self.height() as f32 / std::f32::consts::PI).max(1.0);
        texels.log2().min((self.level_count() - 1) as f32)
    }

    /// Radiance at `uv` from mip level `level` (bilinear: columns wrap, rows clamp).
    fn level_lookup(&self, level: u32, uv: [f32; 2]) -> [f32; 3] {
        match level
            .checked_sub(1)
            .and_then(|l| self.levels.get(l as usize))
        {
            Some(l) => bilinear(&l.texels, l.width, l.height, uv),
            None => self.equirect.lookup(uv),
        }
    }

    /// Radiance along the map's frame direction `dir` filtered for a GGX lobe of `alpha`: the two
    /// mip levels around [`Self::lod`] blended linearly, a box-filtered stand-in for the
    /// GGX-prefiltered map (Karis 2013), sharp for a mirror. Kernel twin: `env_specular`.
    pub fn specular(&self, dir: [f32; 3], alpha: f32) -> [f32; 3] {
        let lod = self.lod(alpha);
        let uv = dir_to_equirect_uv(dir);
        let base = lod.floor();
        let t = lod - base;
        let lower = self.level_lookup(base as u32, uv);
        if t == 0.0 {
            return lower;
        }
        let upper = self.level_lookup(base as u32 + 1, uv);
        std::array::from_fn(|c| lower[c] + (upper[c] - lower[c]) * t)
    }

    /// Entries of [`Self::gpu_entries`].
    pub fn gpu_len(&self) -> usize {
        self.equirect.texels().len()
            + self.height() as usize
            + self.sh.len()
            + self.levels.iter().map(|l| l.texels.len()).sum::<usize>()
    }

    /// The block the kernels read after the palette rows, in order: `[0, w*h)` level-0 texels with
    /// the row-conditional CDF in `.w`; `[w*h, w*h + h)` the marginal CDF in `.x`; the 9 harmonic
    /// coefficients (rgb); mip levels 1.. (rgb). An iterator, so uploads stream it in runs and no
    /// host copy of the block is kept.
    pub fn gpu_entries(&self) -> impl Iterator<Item = [f32; 4]> + '_ {
        let level0 = self
            .equirect
            .texels()
            .iter()
            .zip(self.equirect.conditional())
            .map(|(t, &c)| [t[0], t[1], t[2], c]);
        let marginal = self.equirect.marginal().iter().map(|&m| [m, 0.0, 0.0, 0.0]);
        let sh = self.sh.iter().map(|c| [c[0], c[1], c[2], 0.0]);
        let levels = self
            .levels
            .iter()
            .flat_map(|level| level.texels.iter().map(|t| [t[0], t[1], t[2], 0.0]));
        level0.chain(marginal).chain(sh).chain(levels)
    }
}

/// `(cos, sin)` of a turn about +Y of `degrees`, computed once in f64. Positive turns the map
/// counter-clockwise seen from above, so its content appears to turn left in the view.
pub fn turn(degrees: f32) -> (f32, f32) {
    let radians = f64::from(degrees).to_radians();
    (radians.cos() as f32, radians.sin() as f32)
}

/// The world direction `d` in the map's frame for a [`turn`] `(c, s)`: `[c x - s z, y, s x + c z]`
/// (pt-integrator's `to_env`). Kernel twin: `env_to_map`.
pub fn to_map((c, s): (f32, f32), d: [f32; 3]) -> [f32; 3] {
    [c * d[0] - s * d[2], d[1], s * d[0] + c * d[2]]
}

/// The probability of sampling the sun cone against a map of mean luminance `sky_mean`
/// (intensity applied): 0 for a sun of no light (the map alone is the environment then), else the
/// sun's share of `sun + pi sky_mean` clamped to [`SUN_SELECT_RANGE`]. Computed on the host for
/// every engine (`P_ENV_SUN_SELECT`), with the map's working-space weights `luma`.
pub fn sun_select(sun_radiance: [f32; 3], sky_mean: f32, luma: Luma) -> f32 {
    let sun = luma.of(sun_radiance);
    if sun <= 0.0 {
        return 0.0;
    }
    let total = sun + std::f32::consts::PI * sky_mean;
    (sun / total).clamp(SUN_SELECT_RANGE.0, SUN_SELECT_RANGE.1)
}

/// Write the slots that describe `map` lit at `intensity`, turned by `rotation_degrees`, next to a
/// sun of radiance `sun_radiance` (colour times intensity); `None` writes width 0 (the gradient).
/// The one place both hosts fill the environment block's parameters.
pub fn pack(
    p: &mut [f32],
    map: Option<&EnvMap>,
    intensity: f32,
    rotation_degrees: f32,
    sun_radiance: [f32; 3],
) {
    let (c, s) = turn(rotation_degrees);
    p[P_ENV_INTENSITY] = intensity;
    p[P_ENV_COS] = c;
    p[P_ENV_SIN] = s;
    match map {
        Some(map) => {
            p[P_ENV_WIDTH] = map.width() as f32;
            p[P_ENV_HEIGHT] = map.height() as f32;
            p[P_ENV_LEVELS] = map.level_count() as f32;
            p[P_ENV_MEAN] = map.mean();
            p[P_ENV_SUN_SELECT] = sun_select(sun_radiance, map.mean() * intensity, map.luma());
        }
        None => {
            p[P_ENV_WIDTH] = 0.0;
            p[P_ENV_HEIGHT] = 0.0;
            p[P_ENV_LEVELS] = 0.0;
            p[P_ENV_MEAN] = 0.0;
            p[P_ENV_SUN_SELECT] = 0.0;
        }
    }
}

/// The L2 real spherical-harmonic basis at the unit direction `d` (the order of `EnvMap::sh`).
/// Kernel twin: the basis in `env_irradiance`.
fn sh_basis(d: [f32; 3]) -> [f32; 9] {
    let [x, y, z] = d;
    [
        0.282_094_8,
        0.488_602_5 * y,
        0.488_602_5 * z,
        0.488_602_5 * x,
        1.092_548_4 * x * y,
        1.092_548_4 * y * z,
        0.315_391_57 * (3.0 * z * z - 1.0),
        1.092_548_4 * x * z,
        0.546_274_2 * (x * x - y * y),
    ]
}

/// Solid-angle mean luminance and the irradiance-over-pi harmonics of the map, in f64: each texel
/// is its centre direction with its exact solid angle `(2 pi / w) (cos t0 - cos t1)`; the radiance
/// coefficients are convolved with the clamped cosine (`A_l / pi` = 1, 2/3, 1/4).
fn integrate(map: &EquirectMap, luma: Luma) -> (f32, [[f32; 3]; 9]) {
    use std::f64::consts::PI;
    let (w, h) = (map.width() as usize, map.height() as usize);
    let mut sh = [[0.0f64; 3]; 9];
    let mut mean = 0.0f64;
    for y in 0..h {
        let t0 = PI * y as f64 / h as f64;
        let t1 = PI * (y + 1) as f64 / h as f64;
        let omega = 2.0 * PI / w as f64 * (t0.cos() - t1.cos());
        for x in 0..w {
            let texel = map.texels()[y * w + x];
            let d = equirect_uv_to_dir([(x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32]);
            mean += f64::from(luma.of(texel)) * omega;
            for (coefficient, basis) in sh.iter_mut().zip(sh_basis(d)) {
                for c in 0..3 {
                    coefficient[c] += f64::from(texel[c]) * f64::from(basis) * omega;
                }
            }
        }
    }
    let band = [
        1.0,
        2.0 / 3.0,
        2.0 / 3.0,
        2.0 / 3.0,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
    ];
    let sh = std::array::from_fn(|i| sh[i].map(|v| (v * band[i]) as f32));
    ((mean / (4.0 * PI)) as f32, sh)
}

/// Mip levels 1.. of a map: each the 2 x 2 box average of the level below (columns wrap, rows
/// clamp, a size of 1 stays 1), down to 1 x 1.
fn mips(map: &EquirectMap) -> Vec<Level> {
    let mut levels: Vec<Level> = Vec::new();
    let (mut w, mut h) = (map.width() as usize, map.height() as usize);
    while w > 1 || h > 1 {
        let source = levels.last().map_or(map.texels(), |l| &l.texels);
        let (nw, nh) = (w.div_ceil(2), h.div_ceil(2));
        let mut texels = Vec::with_capacity(nw * nh);
        for y in 0..nh {
            for x in 0..nw {
                let mut sum = [0.0f32; 3];
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let t = source[(2 * y + dy).min(h - 1) * w + (2 * x + dx) % w];
                    for c in 0..3 {
                        sum[c] += t[c];
                    }
                }
                texels.push(sum.map(|v| 0.25 * v));
            }
        }
        levels.push(Level {
            width: nw as u32,
            height: nh as u32,
            texels,
        });
        (w, h) = (nw, nh);
    }
    levels
}

#[cfg(test)]
mod tests;
