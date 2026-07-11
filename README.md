# yunque

<!-- badges: start -->
[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
<!-- badges: end -->

Model-agnostic neural-network building blocks for
[{anvl}](https://github.com/r-xla/anvl), the R interface to XLA.
*Yunque* is Spanish for anvil — this is the forge accessory.

anvl gives you XLA primitives and `jit()`; yunque gives you the
composable layers you reach for when porting a model: broadcast-aware
norms and activations, attention, rotary embeddings, and a base-R
safetensors reader. It is deliberately model-free — the FLUX.2 port that
motivated it lives in [diffuseR](https://github.com/cornball-ai/diffuseR)
(the `anvl_*` files), built on top of yunque.

## Installation

```r
install.packages("anvl", repos = c("https://r-xla.r-universe.dev", getOption("repos")))
pak::pak("cornball-ai/yunque")
```

## What's here

Every function works eagerly and inside `anvl::jit()`.

| Function | Purpose |
|---|---|
| `yq_softmax()`, `yq_layer_norm()`, `yq_rms_norm()`, `yq_group_norm()` | Reductions with the explicit-broadcast dance done for you |
| `yq_silu()` | SiLU / swish activation |
| `yq_linear()` | Bias-free/biased linear on pre-transposed weights, `precision =` aware |
| `yq_sdpa()` | Scaled dot-product attention (optional additive mask) |
| `yq_rope_apply()`, `yq_rope_split()` | Rotary embeddings — FLUX interleaved-pair and Llama/Qwen split-half |
| `yq_repeat_kv()` | Grouped-query KV head expansion |
| `yq_upsample_nearest2d()` | Nearest-2× upsampling over NCHW |
| `yq_slice_lastdim()`, `yq_slice_seq()` | Static slicing for fused-projection and sequence splits |
| `yq_st_open()`/`yq_st_open_sharded()`/`yq_st_read()`/`yq_st_close()`, `yq_read_safetensors()` | Base-R safetensors reader (BF16→f32, sharded-aware, partial reads) — no torch |

## Why explicit broadcasting

anvl binary ops broadcast **scalars only** — there is no implicit
numpy/torch-style shape broadcasting, so `(B, S, D) + (B, 1, D)` is an
error. Every reduction here that feeds a binary op wraps the result in
`anvl::nv_broadcast_to()` first. That single gotcha is the most common
porting error coming from torch, and packaging it behind these helpers
is most of what yunque is for.

## License

MIT © cornball.ai
