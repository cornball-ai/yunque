# yunque 0.0.0.9003

- Broadcasting arithmetic: `badd()`, `bsub()`, `bmul()`, `bdiv()` apply
  numpy-style broadcasting before the op, so `[B, S, D]` combines with
  `[D]`, `[1, D]` or `[B, 1, D]` directly.
- DiT building blocks: `modulate()` (adaLN `x * (1 + scale) + shift`) and
  `gated_residual()` take per-batch `[B, D]` vectors and align them to
  `[B, S, D]` activations themselves; `timestep_embedding()` is the diffusers
  sinusoidal embedding with the DDPM and FLUX conventions.
- `rope_apply()` and `rope_split()` accept `[S, D]` tables shared across
  batch and heads, or per-batch `[B, S, D]` tables, in addition to tables
  already shaped like `x`.
- Static-shape helpers: `pad_to()` pads an axis up to a bucket multiple so
  variable lengths compile once per bucket; `seq_mask()` and `causal_mask()`
  build the additive biases `sdpa()` takes.
- Declarative weight loading: describe a checkpoint as a nested tree of
  `w_linear()`, `w_tensor()`, `w_cat()` (fused projections) and `w_layers()`
  specs and `load_weights()` reads it into `AnvlArray`s, erroring on missing
  keys and reporting the checkpoint keys the spec never used.

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
