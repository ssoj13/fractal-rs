//! The plain-Rust core of the fractal path tracer, shared by its CUDA kernels
//! (`fractal-kernels`) and every host that drives them (WarpBro, ofx-rs `ofx-fractal`).
//!
//! Nothing here depends on CUDA or on cuda-oxide, so hosts build it with ordinary Cargo;
//! the kernels compile the same functions for the device.
//!
//! - [`params`]: the parameter ABI, the float slots the host packs and the kernels read.
//! - [`luma`]: luminance weights of the host's working space (Rec. 709 or ACEScg).
//! - [`path_sampling`]: fast-material lobe sampling and path-depth rules.
//! - [`transmission`]: dielectric media along a path (exit distance, absorption, medium stack).

// `params::slots!` assigns the ABI's slot indices by recursion, one level per slot.
#![recursion_limit = "256"]

pub mod luma;
pub mod params;
pub mod path_sampling;
pub mod transmission;
