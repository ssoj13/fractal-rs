//! Slots of the f32 parameter block shared by the hosts (WarpBro `scene.rs`, OFX `cuda_scene.rs`
//! pack it by name) and the kernels (fractal-kernels reads it): the role of ofx-fractal's
//! uniform3d LAYOUT. Integers are stored as exact f32
//! (all below 2^24). Vectors take 3 consecutive slots, 3x3 matrices 9 (row major).

/// Lays the slots out back to back. A slot's attributes (e.g. `#[cfg(..)]`) apply to its name
/// only: the slot always occupies its place, so a feature never shifts the offsets after it.
macro_rules! slots {
    ($($(#[$meta:meta])* $name:ident = $len:expr),* $(,)?) => {
        slots!(@acc 0usize; $($(#[$meta])* $name = $len,)*);
    };
    (@acc $at:expr; $(#[$meta:meta])* $name:ident = $len:expr, $($rest:tt)*) => {
        $(#[$meta])*
        pub const $name: usize = $at;
        slots!(@acc $at + $len; $($rest)*);
    };
    (@acc $at:expr;) => {
        pub const P_COUNT: usize = $at;
    };
}

slots! {
    // World ABI v1: object/light buffers are immutable for each worker launch.
    P_WORLD = 1, P_OBJECT_COUNT = 1, P_LIGHT_COUNT = 1,
    // frame
    P_WIDTH = 1, P_HEIGHT = 1, P_SAMPLE_BEGIN = 1, P_SPP = 1, P_SEED = 1,
    // camera (thin lens)
    P_CAM_ORIGIN = 3, P_CAM_FORWARD = 3, P_CAM_RIGHT = 3, P_CAM_UP = 3,
    P_HALF_W = 1, P_HALF_H = 1, P_APERTURE = 1, P_FOCUS_DISTANCE = 1,
    // march
    P_MAX_STEPS = 1, P_MAX_DISTANCE = 1, P_FOOTPRINT = 1, P_SAMPLE_CONE = 1, P_STEP_FACTOR = 1, P_SECONDARY_STEPS = 1, P_SECONDARY_EPS = 1, P_GLASS_PROBES = 1,
    // bounding sphere of the set in scene space (centre, radius): rays only march inside it
    P_CLIP_CENTER = 3, P_CLIP_RADIUS = 1,
    // integrator
    P_MAX_BOUNCES = 1, P_MATERIAL_MODEL = 1,
    // formula: common
    P_FAMILY = 1, P_ITERATIONS = 1, P_BAILOUT = 1, P_JULIA = 1, P_JULIA_C = 3,
    P_ITER_ROTATE = 1, P_ITER_ROT = 9,
    P_OBJ_OFFSET = 3, P_OBJ_AXES = 9, P_OBJ_SCALE = 1, P_BOUND_RADIUS = 1,
    // bulb
    P_BULB_POWER = 1, P_BULB_THETA_POWER = 1, P_BULB_PHI_POWER = 1, P_BULB_THETA_PHASE = 1,
    P_BULB_PHI_PHASE = 1, P_BULB_GROWTH = 1, P_BULB_FAST8 = 1,
    // box
    P_BOX_SCALE = 1, P_BOX_FOLD = 1, P_BOX_MIN_R2 = 1, P_BOX_FIXED_R2 = 1,
    // quaternion julia
    P_QUAT_C = 4, P_QUAT_ROWS = 12, P_QUAT_OFFSET = 4,
    // kifs
    P_KIFS_KIND = 1, P_KIFS_SCALE = 1, P_KIFS_SHIFT = 3, P_KIFS_FOLD_HEIGHT = 1, P_KIFS_BOUND = 1,
    // kleinian
    P_KLEIN_A = 1, P_KLEIN_B = 1, P_KLEIN_SKEW = 1, P_KLEIN_LINE_AMP = 1, P_KLEIN_LINE_RATE = 1,
    // pseudo-kleinian
    P_PK_BOX = 3, P_PK_SIZE = 1, P_PK_C = 3, P_PK_OFFSET = 3, P_PK_THICKNESS = 1,
    // apollonian
    P_APOLLO_SCALE = 1,
    // hybrid
    P_HYBRID_STEPS = 1, P_HYBRID_COUNT = 1,
    // colour
    P_COLOR_MODE = 1, P_TRAP_POINT = 3, P_TRAP_NORMAL = 3, P_TRAP_SCALE = 1,
    // light
    P_LIGHT_DIR = 3, P_LIGHT_COLOR = 3, P_LIGHT_INTENSITY = 1, P_SUN_ONE_MINUS_COS = 1,
    P_SUN_CONE_PDF = 1, P_SKY_INTENSITY = 1, P_SKY_HORIZON = 3, P_SKY_ZENITH = 3,
    P_BACKGROUND = 1,
    // 1 shows the sun disc in the background of a camera miss (WarpBro), 0 shows the sky or map
    // alone there (OFX, as its CPU and WGSL engines). Bounced rays always see the disc.
    P_CAMERA_SUN = 1,
    // Lat-long HDR map appended to the palette buffer (fractal-env `pack` fills the block):
    // level-0 size, mip levels, mean luminance, intensity, the turn about +Y as cos/sin, and the
    // probability of sampling the sun cone against the map (0: the map alone). Width 0 = gradient.
    P_ENV_WIDTH = 1, P_ENV_HEIGHT = 1, P_ENV_LEVELS = 1, P_ENV_MEAN = 1, P_ENV_INTENSITY = 1,
    P_ENV_COS = 1, P_ENV_SIN = 1, P_ENV_SUN_SELECT = 1,
    // material (Standard Surface inputs; the fast model reads base/metal/specular/roughness)
    P_BASE = 1, P_BASE_TINT = 3, P_DIFFUSE_ROUGHNESS = 1, P_METALNESS = 1,
    P_SPECULAR = 1, P_SPECULAR_COLOR = 3, P_SPECULAR_ROUGHNESS = 1, P_SPECULAR_IOR = 1,
    P_SPECULAR_ANISOTROPY = 1, P_SPECULAR_ROTATION = 1,
    P_SHEEN = 1, P_SHEEN_COLOR = 3, P_SHEEN_ROUGHNESS = 1,
    P_COAT = 1, P_COAT_COLOR = 3, P_COAT_ROUGHNESS = 1, P_COAT_IOR = 1,
    P_COAT_AFFECT_COLOR = 1, P_COAT_AFFECT_ROUGHNESS = 1,
    P_THIN_FILM_THICKNESS = 1, P_THIN_FILM_IOR = 1,
    P_EMISSION = 1, P_EMISSION_COLOR = 3,
    // base colour source (0 palette, 1 P_BASE_COLOR) and the facing mix (exponent 0 = off)
    P_COLOR_SOURCE = 1, P_BASE_COLOR = 3,
    P_FACING_EXPONENT = 1, P_FACING_COLOR = 3, P_FACING_ROUGHNESS = 1, P_FACING_METALLIC = 1,
    // tonemap
    P_EXPOSURE = 1, P_SATURATION = 1, P_TONEMAP = 1,
    // Luminance weights of the host's working space (`luma::Luma::weights`); every host writes
    // them, zeros would weigh every colour as black.
    P_LUMA = 3,
    // Optional OFX ABI: zeros retain WarpBro's full-frame guide sample counts.
    P_OFX = 1, P_TILE_X = 1, P_TILE_Y = 1, P_TILE_WIDTH = 1, P_TILE_HEIGHT = 1,
    P_OFX_NO_CLIP = 1, P_OFX_THIN_FILM_ENERGY = 1, P_OFX_SEED_HIGH = 1, P_OFX_SKY_ALPHA = 1,
    // Dedicated deterministic Direct shading; PT defaults remain zero. The OFX host writes the
    // shading controls and only the `direct` kernels read them, so their names exist there.
    P_DIRECT = 1, P_DIRECT_GRID = 1,
    #[cfg(feature = "direct")] P_SHADOW_STRENGTH = 1,
    #[cfg(feature = "direct")] P_SHADOW_STEPS = 1,
    #[cfg(feature = "direct")] P_AO_STRENGTH = 1,
    #[cfg(feature = "direct")] P_AO_STEPS = 1,
    #[cfg(feature = "direct")] P_AO_RADIUS = 1,
    #[cfg(feature = "direct")] P_LIGHT_HALF_ANGLE = 1,
    // Dielectric transmission (glass): weight, tint, extra roughness, Beer-Lambert depth.
    P_TRANSMISSION = 1, P_TRANSMISSION_COLOR = 3,
    P_TRANSMISSION_EXTRA_ROUGHNESS = 1, P_TRANSMISSION_DEPTH = 1,
    // Adaptive sampling (`adapt` kernel): relative error threshold and minimum samples per pixel.
    P_ADAPT_THRESHOLD = 1, P_ADAPT_MIN = 1,
}

/// Formula families (ofx-fractal Formula3d codes).
pub const FAMILY_BULB: u32 = 0;
pub const FAMILY_BOX: u32 = 1;
pub const FAMILY_QUAT: u32 = 2;
pub const FAMILY_KIFS: u32 = 3;
pub const FAMILY_KLEINIAN: u32 = 4;
pub const FAMILY_PSEUDO_KLEINIAN: u32 = 5;
pub const FAMILY_APOLLONIAN: u32 = 6;
pub const FAMILY_HYBRID: u32 = 7;
pub const FAMILY_WORLD: u32 = 8;

/// Each object starts with the existing parameter layout, then a row-major inverse affine.
/// Version 4: scene masks per object (`O_MASK_COUNT`, `O_MASKS`).
pub const WORLD_ABI_VERSION: u32 = 4;
pub const O_INVERSE: usize = P_COUNT;
pub const O_DISTANCE_SCALE: usize = O_INVERSE + 12;
pub const O_CLIP_RADIUS: usize = O_DISTANCE_SCALE + 1;
pub const O_TANGENT: usize = O_CLIP_RADIUS + 1;
/// Masks the object's estimate is intersected with (0..=`MAX_MASKS`), then their rows.
pub const O_MASK_COUNT: usize = O_TANGENT + 3;
pub const O_MASKS: usize = O_MASK_COUNT + 1;
pub const OBJECT_STRIDE: usize = O_MASKS + MAX_MASKS * MASK_STRIDE;

/// Masks per world object: a scene mask node (sphere, box or cell grid) intersects the estimate of
/// the objects in its scope, `d = max(d, mask)`, the mask negated when inverted.
pub const MAX_MASKS: usize = 4;
/// A mask row: row-major world-to-mask inverse affine (12), the distance scale (a lower bound of
/// that inverse's stretch, so mask-space distances stay conservative in world space), shape,
/// invert (0 / 1), size (sphere radius in x; box half extents; cell edge in x), cell density
/// (removed share, 0..1) and seed.
pub const M_INVERSE: usize = 0;
pub const M_SCALE: usize = 12;
pub const M_SHAPE: usize = 13;
pub const M_INVERT: usize = 14;
pub const M_SIZE: usize = 15;
pub const M_DENSITY: usize = 18;
pub const M_SEED: usize = 19;
pub const MASK_STRIDE: usize = 20;
pub const MASK_SPHERE: u32 = 0;
pub const MASK_BOX: u32 = 1;
pub const MASK_CELLS: u32 = 2;
pub const LIGHT_STRIDE: usize = P_SKY_INTENSITY - P_LIGHT_DIR;

/// Palette LUT entries (ofx-gen PALETTE_SAMPLES) plus the interior colour.
pub const PALETTE_SAMPLES: usize = 1024;

// The six Direct shading slots are named only with the `direct` feature, but always reserved, so
// kernels built with and without it read one layout (hosts pack by name against either build).
const _: () = assert!(P_TRANSMISSION == P_DIRECT_GRID + 7);
