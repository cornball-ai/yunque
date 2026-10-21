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
| `softmax()`, `layer_norm()`, `rms_norm()`, `group_norm()` | Reductions with the explicit-broadcast dance done for you |
| `silu()` | SiLU / swish activation |
| `linear()` | Bias-free/biased linear on pre-transposed weights, `precision =` aware |
| `sdpa()` | Scaled dot-product attention (optional additive mask) |
| `rope_apply()`, `rope_split()` | Rotary embeddings — FLUX interleaved-pair and Llama/Qwen split-half; tables may be `[S, D]` or per-batch `[B, S, D]` |
| `repeat_kv()` | Grouped-query KV head expansion |
| `badd()`, `bsub()`, `bmul()`, `bdiv()` | numpy-style broadcasting arithmetic: `[B, S, D]` with `[D]`, `[1, D]` or `[B, 1, D]` |
| `modulate()`, `gated_residual()` | adaLN modulation and gated residuals from per-batch `[B, D]` vectors, no reshaping |
| `timestep_embedding()` | diffusers sinusoidal timestep embedding (DDPM and FLUX conventions) |
| `pad_to()`, `seq_mask()`, `causal_mask()` | Shape bucketing and additive attention masks for static-shape compilation |
| `w_linear()`, `w_tensor()`, `w_cat()`, `w_layers()`, `load_weights()` | Declarative weight tree: describe the checkpoint once, get `AnvlArray`s back, unused keys reported |
| `upsample_nearest2d()` | Nearest-2× upsampling over NCHW |
| `slice_lastdim()`, `slice_seq()` | Static slicing for fused-projection and sequence splits |
| `st_open()`/`st_open_sharded()`/`st_read()`/`st_close()`, `read_safetensors()` | Base-R safetensors reader (BF16→f32, sharded-aware, partial reads) — no torch |

## Why explicit broadcasting

anvl binary ops broadcast **scalars only** — there is no implicit
numpy/torch-style shape broadcasting, so `(B, S, D) + (B, 1, D)` is an
error. anvl's own `nv_broadcast_arrays()` and `nv_broadcast_to()` align
ranks on the left, like base R, so a `[D]` weight would meet the batch
axis. yunque applies the numpy rules for you: the reductions here wrap
their results before any binary op, `badd()` and friends broadcast two
arrays the numpy way, and `modulate()`, `gated_residual()` and the RoPE
functions know the one case numpy gets wrong in model code, a per-batch
`[B, D]` vector against `[B, S, D]` activations, which needs its axis
inserted at position 2 rather than on the left. That single gotcha is the
most common porting error coming from torch, and packaging it behind these
helpers is most of what yunque is for.

anvl's `nv_reshape()` is also column-major, like base R's `array()`.
Wherever a torch model splits or merges axes (attention heads, RoPE pairs,
GroupNorm groups, GQA expansion, upsampling), yunque reshapes in torch's
row-major order, so ports can follow the torch code line by line.

## License

MIT © cornball.ai
