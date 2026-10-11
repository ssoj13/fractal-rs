# fractal-rs

The fractal path tracer shared by [WarpBro](https://github.com/ssoj13/warpbro-rs) and
[ofx-rs](https://github.com/ssoj13/ofx-rs) `ofx-fractal`: one source for the CUDA kernels, their
parameter ABI and the material presets, instead of a copy in each application.

| crate | what | builds with |
|---|---|---|
| `fractal-core` | parameter ABI (`params`), Owen-scrambled Sobol sampler, path sampling, transmission | plain Cargo (stable) |
| `fractal-kernels` | the CUDA kernels (`#[cuda_module] kernels`), feature `direct` for Direct shading | `cargo oxide` (cuda-oxide fork) |
| `fractal-materials` | material presets shared by both applications | plain Cargo (stable) |

WarpBro links `fractal-kernels` and loads it with `fractal_kernels::kernels::load`. ofx-fractal
embeds PTX/cubin artifacts regenerated from `fractal-kernels` and packs parameters through
`fractal_core::params`.
