# fractal-rs

A GPU path tracer for 3D fractals, written in Rust and compiled to CUDA with
[cuda-oxide](https://github.com/ssoj13/cuda-rust-windows). It is the shared core of two
applications:

- [WarpBro](https://github.com/ssoj13/warpbro-rs), a standalone fractal explorer and renderer;
- ofx-rs `ofx-fractal`, an OpenFX plugin with CPU, WGPU and CUDA backends.

Both used to carry their own copy of the kernels, and the copies drifted apart. This repository
is now the single source for the kernels, their parameter layout and the material presets.

## What it renders

- Eight distance-estimated fractal families (Mandelbulb and others) and node-graph worlds of
  several transformed objects with per-object materials and lights.
- Sphere tracing with bounded steps, bisection refinement and finite-difference normals.
- Unidirectional path tracing with next-event estimation and MIS for the sun, sky and HDRI
  environment, and an Owen-scrambled Sobol sampler (Burley 2020).
- Materials through the MaterialX Standard Surface BSDF
  ([render-rs](https://github.com/ssoj13/render-rs) `standard-surface-bsdf`), plus a cheaper Fast
  model; dielectric transmission with Beer-Lambert absorption.
- Adaptive sampling per 8x4 tile, and an optional deterministic Direct shading mode.

## Crates

| crate | what it holds | builds with |
|---|---|---|
| `fractal-core` | the parameter ABI (`params`), sampler, path sampling and transmission helpers | plain Cargo |
| `fractal-kernels` | the CUDA kernels in one `#[cuda_module] kernels` module; feature `direct` adds the Direct entry points | `cargo oxide` |
| `fractal-materials` | the material preset catalog; append-only, because hosts store preset indices | plain Cargo |

`fractal-core` has no CUDA dependency: a host packs parameters with the same slot constants the
kernels read, and the device compiles the same helper functions the host tests run on the CPU.

## Using it

A host links `fractal-kernels` and builds itself with `cargo oxide`; the kernels become part of its
binary and load through the generated typed loader:

```rust
use fractal_kernels::kernels;

let module = kernels::load(&ctx)?; // cuda_core::CudaContext
```

A host that cannot use the cuda-oxide backend (an OpenFX plugin, for example) embeds PTX or cubin
built from `fractal-kernels` instead and launches the entry points by name.

## Building

- NVIDIA GPU, CUDA Toolkit 13.x, LLVM/`llc` 21 or newer, clang.
- The stable Rust toolchain from `rust-toolchain.toml` (with `rustc-dev`, `rust-src`, `llvm-tools`).
- `cargo-oxide` from the cuda-oxide fork, at the revision the consuming application pins.

```sh
cargo test -p fractal-core -p fractal-materials   # host-side crates, plain Cargo
cargo oxide build                                 # the kernels -> target/oxide-ptx/fractal_kernels.ptx
```

`cargo oxide build` builds in release. A debug build of the kernels is currently rejected by the
cuda-oxide backend (see AGENTS.md), so tests of a consuming app also run with `--release`.

`standard-surface-bsdf` comes from render-rs, which is not public yet, so the kernel crate can only
be built with access to it.

## Status

The kernels are WarpBro's. Merging in the OFX branch (HDRI importance sampling with SH irradiance
and mips, the leave-and-restart rule, the pixel-cone camera start, the delta sun and the newer
Direct body) is in progress; AGENTS.md tracks the plan and the decisions.

## License

BSD 3-Clause, see [LICENSE](LICENSE).
