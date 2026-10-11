# AGENTS — fractal-materials

The material catalog shared by WarpBro and ofx-rs `ofx-fractal`: 69 presets in 12 categories, no
application, GPU, filesystem or Git dependencies.

`MaterialPreset::apply<T: MaterialTarget>` delegates to each renderer's adapter: WarpBro
`src/materials.rs` (opacity maps to dielectric transmission, `transmission_depth` to Beer-Lambert
absorption) and ofx-rs `crates/ofx-fractal/src/materials.rs` (opacity maps to a coated dielectric;
its CPU and WGSL engines have no transmission). OFX Studio applies the chosen values once, on a
user-edited parameter event.

Keep preset names, order and values stable: hosts save numeric indices (OFX Studio choices, WarpBro
scenes). Append new entries only; OFX keeps Custom at its panel's index 0.

Checks:

```text
cargo test -p fractal-materials
ofx-rs: cargo test -p ofx-fractal --lib materials::
ofx-rs: cargo test -p ofx-effects --test fractal_split
```
