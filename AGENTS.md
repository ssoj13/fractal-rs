# fractal-rs: working notes

Keep this file current: what the crates are, the merge state, decisions, traps. Update it with every
significant change; compress stale entries into the changelog at the end.

## What and why

One source for the fractal path tracer shared by WarpBro (`cglibs/warpbro-rs`) and ofx-rs
`ofx-fractal` (`cglibs/ofx-rs/crates/ofx-fractal`). Before this repo each app kept its own copy of
the CUDA kernels, the parameter ABI and the material catalog, and they diverged (24 kernel items;
map: `C:\projects\projects.rust.cg\nv\prs\kernels-divergence.md`).

| crate | content | consumers |
|---|---|---|
| `fractal-core` | `params` (parameter ABI), `sampler` (Owen-Sobol), `path_sampling`, `transmission` | kernels (device), WarpBro host, OFX host |
| `fractal-kernels` | `#[cuda_module] kernels` (cuda-host flavour: typed loader `kernels::load`), `direct` feature | WarpBro links it; OFX regenerates embedded PTX/cubin from it |
| `fractal-materials` | material presets (append-only: hosts store indices) | WarpBro `materials.rs`, OFX `materials.rs` |

## Build facts (verified)

- Kernels in a dependency crate work: WarpBro's `cargo oxide build` compiles `fractal-kernels` and
  writes `target/oxide-ptx/fractal_kernels.ptx`; `#[cuda_module]` expands in the library and needs
  `cuda-core` as its direct dependency (the loader names its types).
- Tests and builds go through `--release` (`cargo oxide test -- --release`). A dev-profile build of
  `fractal-kernels` fails in the cuda-oxide verifier: `MirCallOp argument 0 type does not match
  callee signature` on `ConstantMemory::get_ref` (addrspace 4 receiver). Compiler bug, see TODO.
- `fractal-core` needs `#![recursion_limit = "256"]` for `params::slots!`.

## Merge decisions (user, 2026-10-10)

- HDRI: OFX's implementation (pt-integrator `EquirectMap` port: marginal/conditional CDF, SH9,
  mips; ENV slots `P_ENV_COS/SIN/LEVELS/SUN_SELECT`). WarpBro's host moves to the render-env LUT.
- OFX fixes for both apps: leave rule with restart (`Hit.first`), pixel-cone camera start with the
  start guard (own per-pixel storage, not `albedo.w`), delta point sun.
- Luma ACEScg (`SS_LUMA`) and Owen-Sobol for both apps.
- Sun disc on camera miss: a scene parameter (WarpBro default on, OFX default off).
- Path-normal finiteness: OFX's per-component check (WarpBro's lets NaN through).
- Direct: OFX's body (HDRI, SH irradiance, mip specular) in `kernels/direct.rs`, returning `PathSample`.

## TODO

- [x] Repo + three crates from WarpBro c999bfd, WarpBro switched to them
- [ ] WarpBro CUDA tests on the shared crate (bit-identical)
- [ ] K11 finiteness, K15 delta sun, K23 Direct body
- [ ] HDRI (K12/K13) + camera sun parameter (K14); freeze the merged `params` layout
- [ ] Cone start + guard (K10), leave restart (K9) over `march_visibility` and glass media
- [ ] OFX: launch ABI (active/stats), `cuda_scene.rs` slots, regenerate.py from this crate
      (`--features direct`, computed float count), provenance, artifacts, parity suites
- [ ] fractal-materials: OFX's `publish = false`/AGENTS rules, 67 -> 69 asserts in ofx-effects
- [ ] cuda-oxide: dev-profile verifier failure on cross-crate `ConstantMemory::get_ref`
- [ ] Side: sun half angle below ~3.5e-4 rad gives `P_SUN_CONE_PDF = +inf` in f32 (WarpBro scene.rs)
