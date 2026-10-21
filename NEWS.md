# yunque 0.0.0.9002

- Requires anvl >= 0.5.1. anvl 0.5 reshapes column-major and broadcasts
  left-aligned, like base R; yunque keeps torch semantics by reshaping
  row-major internally where axes split or merge (`group_norm()`,
  `upsample_nearest2d()`, `repeat_kv()`, `rope_apply()`, `rel_shift()`) and
  right-aligning lower-rank weights, biases and masks before broadcasting.
  Follows anvl's renames (`naxes()`, `nv_sum()`, `nv_max()`, `nv_plogis()`,
  `nv_pmax()`/`nv_pmin()`, `nv_rev()`, `end_indices`, `axis`, and
  `nv_pad()`'s `low`/`high`/`interior`).
- The `anvl-development` skill is rewritten for anvl 0.5, with an upgrade
  table from 0.3 and the layout rules torch ports need.

# yunque 0.0.0.9001

- Ship the `anvl-development` skill under `inst/skills/anvl-development/`:
  agent instructions for JAX-style array computing, JIT compilation, and
  autodiff with anvl, including tracing and recompilation pitfalls, static
  shapes, RNG threading, and porting torch models to the XLA backend. Moved
  here from the personal skill hub so it versions with the package.
