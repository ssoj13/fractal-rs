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

## Landing in ofx-rs (agreed with the ofx-rs session "ofx-rs-be", 2026-10-10)

- ofx-rs keeps `crates/ofx-fractal` and `crates/fractal-materials` hands-off until we land; tell
  ofx-rs-be when it is in. The 67 -> 69 preset assert in `ofx-effects/src/fractal/mod.rs` changes in
  the same landing.
- Keep the leave rule exactly: `Traced::leave` on CPU/WGSL, CUDA `leave_ray` / `march_leaving`,
  the `Hit.first` restart, the shadow ray on the CPU's side of the normal. Gates:
  `crates/ofx-fractal/src/pathtrace/tests.rs` (pepper on CPU/WGPU/CUDA + self-check, restart-share
  bounds, `PARITY_RESTARTS`, over-tau chaos-floor bound, CUDA/CPU agreement with the 3-reseed noise
  floor).
- Run: ofx-fractal `--features wgpu,cuda -- --include-ignored` (skip `sample_hdris_load_and_cache`);
  ofx-effects `--test fractal_scenes --test fractal_split --test fractal_environment --test
  descriptors --release`.
- Owen-Sobol + ACEScg luma change CUDA output: re-measure the CUDA/CPU noise floors; never loosen a
  threshold without an attribution measurement in the test comment.
- `regenerate.py --record-only` must leave provenance byte-identical on a clean tree.
- Land only via `python scripts/land.py` from a branch merged with origin/main. ofx-rs's dev
  profile is line-tables-only.

## Resume point (2026-10-11, branch `merge-ofx`, WIP; main is untouched and still what WarpBro uses)

Done on the branch:
- `fractal-core::luma::Luma` {Rec709, AcesCg} + slot `P_LUMA` (3); kernels' `luminance` reads it
  (user: luma is a working-space option; ACEScg for both WarpBro and OFX, the option set where a
  host's colours are in another space, 2026-10-11).
- `fractal-env` crate: OFX's EnvironmentMap core (CDFs, mean, SH9, mips, gpu_entries, irradiance,
  specular) + `turn`, `to_map`, `sun_select`, `pack` (fills P_ENV_*); 6 tests pass. Hosts keep
  their own decode and caches. Orientation verified: OFX `env_to_map` = WarpBro's former
  `atan2(z,x) + rotation` (same sign).
- params: OFX ENV block (P_ENV_WIDTH/HEIGHT/LEVELS/MEAN/INTENSITY/COS/SIN/SUN_SELECT),
  `P_CAMERA_SUN`, `WORLD_ABI_VERSION` 3; `env_uv`/`env_dir` removed from path_sampling.
- kernels: K11 finiteness; OFX env block (env_on .. env_map_pdf, background, sky_irradiance,
  sky_specular, sun_select, env_radiance/pdf/sample) replaced WarpBro's map_*; camera miss
  honours `P_CAMERA_SUN`; point sun (K15) with `mis_env` gate.
- K23 Direct body (HDRI helpers behind `direct`); builds with and without the feature.
- K10 cone start + start guard (pinhole only; cache in `stats[2]`, stats is `[f32; 4]`), K9 leave
  rule + restart (`leave_ray`, `march_leaving`, `visibility_leaving`).
- WarpBro branch `fractal-merge` (local `[patch]` to this checkout, never committed): environment
  on `fractal_env`, `pack`, `P_LUMA` AcesCg, `P_CAMERA_SUN` = 1 (UI setting after the takes
  branch lands: world.rs is shared with it), `scene::sun_cone` (zero angle = point sun; world
  lights keep a 1e-4 rad minimum), stats `[f32; 4]`. CUDA tests after the HDRI step: 17/18 (the
  export cancel test timed out under load).

Next:
1. WarpBro CUDA tests after K9/K10; `bootstrap.py k` register/spill report vs before.
2. Merge merge-ofx -> main, drop the WarpBro `[patch]`, update its lock, land WarpBro.
3. The OFX switch (see "Landing in ofx-rs"), with OFX on ACEScg luma.
- ofx-rs main e366730 (exr-rs session): render-env no longer reads EXR itself; the host registers
  an EXR reader. fractal-env only uses `render_env::EnvCdfs`, so it is unaffected; the OFX switch
  must keep the reader registration working.
- OFX CPU and WGPU engines move to Owen-Sobol (shared `fractal_core::sampler`, WGSL port) in the
  OFX switch, so its three engines stay consistent (user, 2026-10-11).
- cuda-oxide fork main a11d4cce8 on Linux (WSL): test/test-cuda/check-guards/clippy pass,
  math_sin_cos SUCCESS; smoketest 232/235 (abi_hmm, disjoint_slice_len, host_closure fail),
  fmt-check exit 1, doc-check exit 101, check-intrinsics exit 1 (known nightly-llc evidence
  mismatch). Logs: WSL ~/pr-logs/allocas-main, `.smoketest-logs/`. To investigate.

## Resume point (2026-10-11 morning)

- All cargo runs go through `C:/projects/projects.rust.cg/cglibs/.buildlock/buildlock.py` (machine-wide
  build queue shared by the sessions).
- WarpBro main pinned to fractal-rs 919f4c7 (`cf0f91e`) until `fractal-merge` lands; it switches back
  to `branch = "main"`.
- ofx-rs worktree `cglibs/ofx-rs-fractal`, branch `fractal-rs`, commit `fdb6f08` (local, not pushed):
  the whole host/kernel switch; artifacts regenerated (fractal-rs 33fab96, 234 floats, 30 entries);
  record-only idempotent; clippy (ofx-fractal + ofx-effects, wgpu+cuda) clean. regenerate.py sets
  CUDA_OXIDE_PTX_DIR (a dependency's PTX otherwise lands in its ~/.cargo checkout).
- Open question (step 1): WarpBro `cuda_march_out_of_steps_is_a_reported_miss_not_a_phantom_hit`
  fails because `cone_start`'s loop is bounded by P_MAX_STEPS (start depends on the budget);
  `cuda_curved_hybrid_normals_match_converged_field_gradient` fails at the default budget: check
  whether the cone march (0.5 d safety) skips thin hybrid features (non-exact DE) -> geometry change.

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

## Resume (2026-10-11, usage limit)
- ofx-rs worktree C:\projects\projects.rust.cg\cglibs\ofx-rs-fractal, branch fractal-rs (off origin/main e366730): kernels/source is a thin crate on fractal-kernels (lock pins fractal-rs 574d83c, cuda-oxide a11d4cce, render-rs 65de24a). NEXT: regenerate.py (CRATE fractal_kernels, files Cargo.toml/Cargo.lock/src/lib.rs, no params copy, parameter count from _PARAMS[n], revisions from Cargo.lock, check host fractal-core rev == kernel rev); cuda/mod.rs (params from fractal-core with feature direct, 16-word launch: active empty + stats [f32;4] zeroed, validate comment); cuda_scene.rs (P_LUMA AcesCg, P_CAMERA_SUN 0); environment.rs on fractal_env::EnvMap (Luma::AcesCg) + uniform3d env words via fractal_env::turn/sun_select; fractal-materials -> git crate, 67->69 assert; regenerate artifacts; OFX tests per 'Landing in ofx-rs'.
- OFX CPU/WGSL already use SS_LUMA (ACEScg) via the BSDF; only the old CUDA kernel used Rec.709.
- OFX CPU/WGSL RNG is render-rs pt-integrator PathRng/PtRng: Owen-Sobol for them = move the sampler into render-rs (fractal-core re-exports it).
- WarpBro branch fractal-merge bdd2f13 (not on main): CUDA tests 15/18 after K9/K10; fails: cuda_march_out_of_steps_is_a_reported_miss_not_a_phantom_hit, cuda_curved_hybrid_normals_match_converged_field_gradient (likely cone_start's loop bounded by P_MAX_STEPS makes starts step-budget dependent), export cancel test (timeout under load). Then bootstrap.py k register report.

- [ ] OFX: cone_start budget = CONE_START_STEPS (256) on its CPU twin (fractal3d.rs cone_start) and WGSL (pathtrace.wgsl cone_start) when ofx-rs regenerates past 33fab96; tell ofx-rs-be first.

## State (2026-10-11 midday)

- DONE: fractal-rs main dada09d; WarpBro main 89c2491 on it (CUDA 18/18, non-GPU 245, clippy clean;
  register report unchanged vs before the OFX merge: sm_86 full_* frame 656-728, world 672, fast_* 128-184);
  ofx-rs main 9d1035a on fractal-rs 33fab96 (all OFX gates green).
- NEXT: OFX twins of the fixed cone budget + regenerate past 33fab96; OFX CPU/WGSL Owen-Sobol (sampler into
  render-rs pt-integrator, coordinate with exr-rs-ee's render-env B2); P_CAMERA_SUN as a WarpBro scene setting
  (after the takes branch lands); WarpBro takes (branch takes, WIP 4d78d97), File > New, template conversion.
