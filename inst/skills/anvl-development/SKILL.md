---
name: anvl-development
description: >
  Develop with anvl (r-xla): JAX-style array computing, JIT compilation via
  XLA/StableHLO/PJRT, and reverse-mode autodiff in R. Use when writing or
  porting numerical/deep-learning code to anvl, debugging tracing and
  recompilation, working around static shapes, threading RNG state, or
  re-implementing torch models on the XLA backend.
---

# anvl (r-xla) Development

anvl is the R answer to JAX: trace R functions into a graph, lower to
StableHLO, compile with XLA, run on CPU or CUDA GPU. Reverse-mode
autodiff returns the gradient of a function as another R function. The
design borrows from JAX's autodidax tutorial. The package was renamed
from `anvil` to `anvl` — old links say anvil, the package is `{anvl}`.

Authors: Sebastian Fischer, Daniel Falbel, Tomasz Kalinowski (Posit),
Nikolai German. MIT license. Facts in this skill verified against
anvl 0.3.0.9000 source (July 2026).

## Ecosystem

| Package | Role |
|---|---|
| `anvl` | User-facing API: arrays, ops, `jit()`, `gradient()` |
| `stablehlo` | Builds StableHLO programs in R (anvl's lowering target) |
| `pjrt` | R interface to PJRT (the runtime that executes XLA binaries) |
| `tengen` | Shared S3 generics: `shape()`, `dtype()`, `device()`, `as_array()` |
| `xlamisc` | Helpers |
| `quickr` | Experimental Fortran backend (CPU only, Suggests) |

Docs: https://r-xla.github.io/anvl/ · Repo: https://github.com/r-xla/anvl
Benchmarks vs PyTorch and R torch: https://github.com/r-xla/benchmarks

## When to Use What

task: Install / set up CUDA
use: "Installation" below

task: Understand why a function recompiles or returns stale results
use: "The jit() Contract" — cache keys, closures, branch baking

task: Express data-dependent logic (filtering, `which`, boolean masks)
use: "Static Shapes and the Masking Pattern"

task: Port an R torch / PyTorch model to anvl
use: "Porting torch Models" — op mapping, gaps, params-as-list pattern

task: Debug wrong numbers after a port
use: "Validation" — parity fixtures + treesitR AST audit

task: Benchmark or speed up anvl code
use: "Performance" — await(), donate, bucketing

## Installation

```r
# Prebuilt binary from r-universe
install.packages("anvl",
  repos = c("https://r-xla.r-universe.dev", getOption("repos")))

# CUDA (Linux x86-64 only; WSL2 experimental)
install.packages("cuda12.8", repos = "https://mlverse.r-universe.dev")
```

- Runtime needs `libprotobuf`; source builds need C++20 and `protoc`.
- GPU platforms: Linux amd64 or WSL2. No macOS GPU. No Intel macOS at all.
- Debug device bring-up: set `PJRT_DEBUG=1` and `TF_CPP_MIN_LOG_LEVEL=0`
  in a fresh session, then `anvl::nv_scalar(1, device = "cuda")`.
- Conflicts with other cudatoolkit R packages in one process — isolate
  via {mirai} daemons if both torch-cuda and anvl-cuda are needed.
- Docker: `ghcr.io/r-xla/anvl-cuda:latest` (needs NVIDIA Container
  Toolkit, `--gpus all`).

## Core Model

`AnvlArray` is the data structure: dtype + static shape + device,
**value semantics** (never modified in place; eager subset-assign always
copies, even when R would optimize). 0-d arrays are scalars
(`nv_scalar()`). Convert with `as_array()` / `as.double()`; construct
with `nv_array(x, shape =, dtype =, device =)`.

Two execution modes:
- **Eager**: each op compiles and runs a tiny XLA program. Correct but
  slow — use for debugging intermediate values.
- **jit()**: trace once with placeholders, compile the whole function
  as one XLA executable, cache it. This is the production path.

```r
f <- function(a, b, x) a * x + b
f_jit <- jit(f)
f_jit(nv_scalar(1.0), nv_scalar(-2.0), nv_scalar(3.0))

g <- jit(gradient(f, wrt = c("a", "b")))   # named list of gradients
vg <- jit(value_and_gradient(f, wrt = "a"))
```

`gradient()` is reverse-mode, first-order only.

## The jit() Contract

Signature: `jit(f, static = character(), cache_size = 100L, backend =
NULL, device = NULL, ...)`. `donate =` passes through `...` to the XLA
backend.

**Cache key** = (shape, dtype) of each dynamic input + `identical()`
value of each static input + target device. Cache misses re-trace and
recompile. Consequences:

- Call `jit()` **once** and reuse the wrapper. `jit()` inside a loop
  creates a fresh cache every iteration.
- Every new input shape recompiles. Bucket/pad varying dims (see
  Performance).
- Device is part of the key unless pinned via `device = "cuda:0"`
  (pinning auto-copies inputs to that device).

**Tracing rules** (violations produce silently-wrong cached results,
not errors):

1. **Functions must be pure.** R-level side effects (assignments to
   enclosing env, message/print, RNG) fire once during tracing, never
   again on cached calls.
2. **R `for`/`while` loops unroll at trace time.** n iterations = n
   copies of the ops in the graph. Fine for a distilled 4–8 step
   denoise loop; catastrophic for n in the thousands — use `nv_while()`.
3. **R `if` bakes one branch** — the one taken at trace time. Safe only
   when the condition depends on `static` arguments. A condition on a
   closed-over variable is the classic stale-cache bug: make it a
   static argument instead.
4. **Closed-over values become compile-time constants.** Changing them
   later does nothing. Pass everything that varies as an argument.
5. Only `prim_*` calls (and `nv_*` wrappers that delegate to them) are
   recorded. Plain R arithmetic on R doubles happens at trace time and
   is constant-folded into the graph.

**Structured control flow** (traced as single ops, no unrolling):

```r
# State is a NAMED list; cond/body take the state elements as args
nv_while(
  list(i = nv_scalar(1L), total = nv_scalar(0L)),
  cond = function(i, total) i <= 100L,
  body = function(i, total) list(i = i + 1L, total = total + i)
)
nv_if(pred, true_fn, false_fn)     # value-level branch
nv_ifelse(pred, x, y)              # elementwise select
```

## Static Shapes and the Masking Pattern

XLA requires every intermediate shape known at compile time. Inside
`jit()` there is **no** `unique()`, `which()`, `x[x > 0]`, boolean-mask
subsetting, or dynamic-length ranges.

Standard workaround — keep the shape, mask the values:

```r
# sum(x[x > 0]) becomes: replace non-matches with the neutral element
nv_reduce_sum(nv_ifelse(x > 0, x, nv_fill_like(x, 0)), dims = 1L)

# mean over a mask: sum masked values / sum mask
mask <- x > 0
total <- nv_reduce_sum(nv_ifelse(mask, x, 0), dims = 1L)
total / nv_reduce_sum(mask, dims = 1L)
```

Neutral elements: 0 (sum), 1 (prod), -Inf (max), +Inf (min), FALSE
(any), TRUE (all). Genuinely dynamic ops (`which`, `unique`) must exit
to R: `as_array()`, compute, re-enter.

## Dtypes and Promotion

12 dtypes: `i1` (bool), `i8/i16/i32/i64`, `ui8/ui16/ui32/ui64`,
`f32`, `f64`. **No f16, no bf16, no fp8, no complex.** Half-precision
checkpoints must be upcast to f32, and a model's memory footprint is
2x its bf16 torch footprint. Plan VRAM accordingly.

R literals enter as *ambiguous* types (`f32?` for doubles, `i32?` for
integers, `i1?` for logicals): they defer to a known dtype on the other
operand instead of promoting it. `1.0 + x_i16` stays `i16`-adjacent
where torch would promote. Pin dtypes explicitly at boundaries
(`nv_scalar(1.0, "f32")`) when promotion matters; `common_dtype()`
answers "what would these combine to".

## Broadcasting Is Explicit (Scalars Excepted)

Binary ops (`+`, `*`, `nv_add`, ...) broadcast **only 0-d scalars**
automatically (`nv_broadcast_scalars`). There is NO implicit
numpy/torch-style shape broadcasting: `(B, S, D) + (B, 1, D)` is an
error, not a broadcast. Broadcast explicitly:

```r
e / nv_broadcast_to(s, shape(e))     # numpy-rule expansion to a shape
nv_broadcast_arrays(x, y)            # mutual broadcast, returns a list
```

This is the single most frequent porting error from torch. Every
reduction with `drop = FALSE` that feeds a binary op needs an explicit
`nv_broadcast_to`.

## Subsetting

1-based, like R. Static indices (R values) vs dynamic indices
(AnvlArrays) behave differently:

| Form | Static | Dynamic |
|---|---|---|
| Single index `x[2]` / `x[nv_scalar(2L)]` | yes | yes |
| Multiple `x[arr(2, 4, 6)]` / `x[nv_array(c(2L,4L,6L))]` | yes | yes |
| Range `x[2:5]` | yes | no — size must be static (`x[nv_seq(2, 5)]` ok: static size, dynamic start) |
| Boolean mask | no | no |
| Negative index | no | no |

Gotchas:
- **Static** out-of-bounds indices error at compile time. **Dynamic**
  OOB indices are silently **clamped** on read and silently **dropped**
  on write. No runtime bounds errors — validate index math yourself.
- Duplicate indices in one subset-assign: which write wins is
  backend-defined (CPU and GPU can differ).
- Eager subset-assign always copies the full array. Put updates inside
  `jit()` where XLA fuses them.

## RNG: Explicit State Threading

No global `.Random.seed`. State is a `ui64[2]` array; every sampler
takes it and returns `list(new_state, values)`:

```r
state <- nv_rng_state(42L)
r1 <- nv_runif(shape = 3L, initial_state = state)
state <- r1[[1L]]                    # THREAD IT — reusing the old
r2 <- nv_rnorm(shape = 3L, initial_state = state)  # state repeats values
```

Samplers: `nv_runif`, `nv_rnorm`, `nv_rbinom`, `nv_rdunif` (use named
arguments — the signature is `(shape, initial_state, ...)` and has
changed across 0.x). The generator is XLA's `rng_bit_generator` —
**no seed parity with R, torch, or diffusers.** For output parity with
a torch port, generate noise in the reference system and pass it in as
an input (same trick as parity fixtures).

## Weight I/O

`nv_save(arrays, path)` / `nv_read(path, device =)` speak safetensors
natively (a named list of AnvlArrays per file). To bring over a
HuggingFace / torch checkpoint: load with `safetensors::safe_load_file`
or torch, upcast f16/bf16 to f32, re-save, `nv_read` onto the target
device. There is no lazy/streamed loading — the file lands in memory.

## Porting torch Models

### Op mapping

| torch | anvl |
|---|---|
| `torch_matmul` (batched) | `nv_matmul(lhs, rhs)` — leading dims are batch dims, via `prim_dot_general` |
| `$transpose(i, j)` / `$permute(p)` | `nv_transpose(x, permutation = p)` (full permutation; default reverses all dims) |
| `$reshape` / `$view` | `nv_reshape(x, shape)` |
| `$unsqueeze` / `$squeeze` | `nv_unsqueeze` / `nv_squeeze` |
| `torch_cat(list, dim)` | `nv_concatenate` / `nv_rbind` / `nv_cbind` |
| `torch_where` | `nv_ifelse(pred, x, y)` |
| `$clamp` | `nv_clamp` |
| `nnf_sigmoid` | `nv_logistic` |
| `torch_rsqrt`, `tanh`, `erf`, ... | `nv_rsqrt`, `nv_tanh`, `nv_erf`, ... (98 primitives; most elementwise math exists) |
| `torch_sort` / `topk` / `argmax` | `nv_sort` / `nv_top_k` / `nv_argmax` |
| `torch_cumsum` | `nv_cumsum` (lowers to `reduce_window`) |
| index_select / gather / scatter | `nv_select`, `prim_gather`, `prim_scatter` |
| `chol/qr/svd/solve/eigh/lu` | `nv_chol/nv_qr/nv_svd/nv_solve/nv_eigh/nv_lu` (CPU perf depends on R's BLAS/LAPACK) |

### What does NOT exist (verified against 0.3.0.9000)

- **Convolution exists.** `prim_convolution` and `nv_conv1d`/`nv_conv2d`/
  `nv_conv3d` shipped in anvl 0.3.0.9000 (PR #385, on the
  `hlo_convolution` op from stablehlo #161). Handles stride, symmetric +
  asymmetric/causal padding, dilation, and groups (2D/3D), all parity-
  clean vs torch. No branch install is needed.
- **No softmax / layer_norm / group_norm / gelu / attention builtins in
  anvl itself** — but the **yunque** package provides them all
  (`yunque::yq_softmax`, `yq_layer_norm`, `yq_rms_norm`, `yq_group_norm`,
  `yq_silu`, `yq_linear`, `yq_sdpa`, `yq_rope_apply`/`yq_rope_split`,
  `yq_repeat_kv`, `yq_upsample_nearest2d`, + a bf16→f32 safetensors
  reader). Reach for yunque before composing from scratch. Only GELU
  isn't there yet — compose `0.5 * x * (1 + nv_erf(x / sqrt(2)))`.
- **No einsum.** Reformulate as `nv_transpose` + `nv_matmul`
  (`prim_dot_general` covers arbitrary contract/batch dims if needed).
- **No f16/bf16** (above; issue #379). No quantized dtypes; `ui8` +
  `nv_convert` + shift/and ops could express NF4-style dequant by hand.
  bf16 does NOT make 12B/22B models fit a 16GB GPU (still 24/44GB) — that
  needs NF4; bf16's payoff is 4B-class speed and lighter CPU loads.
- **No nn_module.** Models are pure functions; parameters travel as a
  named list argument (JAX pytree style — anvl traces through nested
  lists in inputs).

### Composed building blocks

```r
softmax_lastdim <- function(x) {
  d <- ndims(x)
  m <- nv_reduce_max(x, dims = d, drop = FALSE)
  e <- nv_exp(x - nv_broadcast_to(m, shape(x)))
  e / nv_broadcast_to(nv_reduce_sum(e, dims = d, drop = FALSE), shape(x))
}

# q, k, v: (B, H, S, D); attention as batched matmul
attn <- function(q, k, v, scale) {
  scores <- nv_matmul(q, nv_transpose(k, c(1L, 2L, 4L, 3L))) * scale
  nv_matmul(softmax_lastdim(scores), v)
}
```

Materializes the full (B, H, S, S) score matrix — same memory story as
pre-SDPA torch. XLA may fuse, but budget for the worst case at long
sequence lengths.

### Structure of a ported model

```r
# params: named list mirroring the checkpoint key tree (same 1:1
# census discipline as torch ports — every key lands, every slot fills)
model_step <- function(params, latents, t_emb) {
  h <- nv_matmul(latents, params$in_proj$weight)  # ...
  h
}
step <- jit(model_step)          # params traced as dynamic inputs
```

Inference needs no `with_no_grad` equivalent — gradients only exist
where you ask for them with `gradient()`.

### The porting playbook (proven on FLUX.2, reused across the model loop)

Three layers, kept separate:
- **anvl** — XLA primitives (`nv_*`, `jit`, convolution).
- **yunque** (github.com/cornball-ai/yunque) — model-agnostic NN
  building blocks composed from anvl: `yq_softmax`, `yq_layer_norm`,
  `yq_rms_norm`, `yq_group_norm`, `yq_silu`, `yq_linear`, `yq_sdpa`,
  `yq_rope_apply`/`yq_rope_split`, `yq_repeat_kv`,
  `yq_upsample_nearest2d`, and a base-R F16/BF16/F32 safetensors reader
  (`yq_st_open`/`yq_st_read`). Reusable across every model.
- **the model package** (e.g. diffuseR's `anvl_*.R` files) — the
  specific architecture, sitting beside its torch reference so parity
  tests can `library()` both. Functions call `yunque::` / `anvl::`
  qualified (both are Suggests, so the torch path is untouched).

Per-model recipe, one component at a time (block → encoder → VAE →
loop → e2e):
1. **Read the torch reference fully**, mirror its module tree 1:1.
   Each block/module becomes a closure over static config returning
   `function(activations..., w)`; weights travel as a nested named
   list (pytree) mirroring the checkpoint key tree — anvl traces
   through it.
2. **Weight loader**: `yq_st_open` the checkpoint, read per tensor and
   `nv_array` it straight to device (freeing the R copy) — never hold
   the whole model as R doubles twice. Conv weights load raw
   `[out,in,kH,kW]`; linear/attention weights transpose to `[in,out]`.
   Detect optional sub-modules (ResNet shortcut convs) by key existence.
3. **Parity fixture from torch**: load the SAME checkpoint into the
   torch reference, run on fixed random inputs (feed the reference's
   noise as an input — never chase RNG), save inputs + output (not
   weights — the anvl side reloads them). `$contiguous()`/`$clone()`
   every saved tensor (the view-save trap). Run torch with
   `TORCH_CUDATOOLKIT=FALSE` if a broken cuda toolkit is installed.
4. **Parity test**: jit the anvl component, load the same weights,
   compare. Judge by correlation + max-abs-diff **relative to the
   tensor scale** (deep residual streams run sd~35, so a 1e-3 abs diff
   is still 2.8e-5 relative — f32 rounding, not a bug). Target
   correlation 1.000000. Matching-sd-but-low-correlation ⇒ suspect a
   fixture view-save bug before the port.
5. **Debug divergence** by dumping per-stage intermediates from both
   sides and binary-searching the first mismatch — exactly localizes
   the bug (wrong axis, missing residual, transposed weight).

Random-init weights validate architecture parity just as well as real
weights and skip the checkpoint download; use real weights when you
also want a runnable end-to-end result. All parity is CPU f32 —
independent of GPU availability.

## Validation

Same discipline as torch ports, plus the standing treesitR rule:

1. **Parity fixtures**: run the reference (torch or Python) on fixed
   inputs, save inputs+outputs to safetensors, compare the anvl port
   at f32 tolerances (max diff < 1e-5 elementwise for a single block;
   looser end-to-end).
2. **Noise as input**: never try to match RNG streams — feed the
   reference's noise tensors in as arguments.
3. **treesitR AST audit** (mandatory finish step for any port): diff
   numeric literals and callee names per paired scope between the
   reference and the anvl R code. Pattern:
   `r tools/compare_translation.R` in diffuseR — extend its pairings.
   Parity tests prove the tested paths; the AST diff catches unported
   branches, dropped constants, and missing ops the fixtures never hit.
   R-torch-to-anvl redos parse both sides with `ts_language_r()`.
4. Eager mode is the layer-by-layer debugger: run the un-jitted
   function and print/`as_array()` intermediates. Then confirm the
   jitted output matches eager output exactly before trusting the
   cache.

## Performance

- **Everything is async.** Calls return before the computation
  finishes; results await implicitly on print/`as_array()`. Benchmarks
  MUST call `await(result)` inside the timed region or they measure
  dispatch, not compute. XLA runs small ops synchronously and large
  ones in the background, so wrong benchmarks can look plausible.
- **Exploit the async gap**: in a training/denoise loop, do R-side prep
  for step i+1 while the device chews on step i. Printing per step
  forces a sync and idles the GPU.
- **`donate`**: `jit(function(w, g) w - lr * g, donate = "w")` lets XLA
  reuse the input buffer for the output — the update-in-place idiom for
  loops. Touching a donated array afterwards errors.
- **Shape bucketing**: pad varying dims to 2^n buckets so calls share
  compiled executables; mask the padding (neutral values).
- **Keep data on device**; CPU↔GPU transfers have high fixed cost.
  Small ops on small arrays can be slower on GPU than CPU.
- Eager mode is inherently slow (one XLA program per op) — never
  benchmark eager and conclude anything about anvl.
- CPU linear algebra (`nv_svd` etc.) inherits R's BLAS — OpenBLAS/MKL
  matters. Thread control is OS-level: `taskset -c 0-3 Rscript ...`.
- Compilation cost is real for big graphs. A fully-unrolled 48-block
  transformer step is one giant program — expect a long first call and
  measure whether `nv_while` over blocks (weights stacked along a
  leading dim) compiles faster at equal runtime.

## Resources

All under https://r-xla.github.io/anvl/articles/:
`anvl` (get started), `jit` (deep dive), `static_shapes`, `efficiency`,
`type-promotion`, `subsetting`, `random-numbers`, `installation`,
`extending_api`, `extending_primitive` (add your own primitive),
`primitives` (the 98-primitive table), `internals`, `faq`, plus worked
examples: `gaussian-process`, `logistic-regression`,
`metropolis-hastings`.

## Lessons Learned

<!-- Updated as the diffuseR-on-anvl redo hits real problems. -->

### Layout semantics: R in, torch out (July 2026)

Verified by probe: `nv_array(1:6, shape = c(2, 3))` round-trips to
`matrix(1:6, 2, 3)` (R column-major logical indexing preserved), but
`nv_reshape` reinterprets in **row-major logical order** — exactly
torch/numpy `reshape` semantics on the logical array, NOT R's `dim<-`.
Consequence: port torch reshapes 1:1 without axis gymnastics, and
never reason about anvl reshapes with R's column-major intuition.
`nv_read` on a torch-written safetensors file gives arrays whose
logical content matches the torch tensors (pjrt buffers are row-major
like the file), so weights and fixtures cross over cleanly.

### First real port: FLUX.2 Klein single-stream block (July 2026)

Full block (modulated LayerNorm, fused QKV+MLP projection, RMS-normed
q/k, RoPE, SDPA, SwiGLU, gated residual) composed from nv_* primitives
matched the R torch reference on real checkpoint weights at
**max 3e-06** (S = 512, f32). One Klein block, S = 4608, batch 1,
RTX 5060 Ti, steady-state:

| Config | ms/iter |
|---|---|
| torch CUDA bf16 (production) | 53.0 |
| **anvl CUDA f32 (jit)** | **81.7** |
| torch CUDA f32 | 161.8 |
| torch CPU f32 | 1473.6 |
| anvl CPU f32 (jit) | 1553.9 |

XLA fusion beats eager torch by 2x at equal (f32) precision; anvl is
within 1.6x of torch bf16 while moving twice the bytes. CPU is a wash.
jit compile+first-call was ~4s (GPU) / ~1.6s (CPU) for the one-block
graph. bf16 support would likely flip the GPU ranking.

Correction after measuring precision properly: anvl 0.3.0's CUDA dots
run **TF32** with no user control (scaled matmul error 1.4e-03 vs
2.3e-06 for strict f32). The 81.7ms above was TF32-assisted. The dev
branch adds `nv_matmul(precision =)` defaulting to `"highest"`; at
matched strict-f32 the block is 126.6ms — still 1.28x faster than
torch f32 (161.8), with TF32 as a real extra gear torch-R doesn't
expose. Always verify TF32 vs strict-f32 before claiming "equal
precision" in a benchmark.

### Full FLUX.2 Klein DiT ported (July 2026)

The whole conv-free transformer (5 double-stream MMDiT blocks with
joint attention + 20 single-stream parallel blocks + embedders, shared
modulation, adaLN-continuous output norm) matched the R torch reference
end-to-end at **max 2.5e-05, correlation 1.000000** on real 3.876B-param
Klein weights. Patterns that worked:

- **Weights as a nested named list (pytree).** jit traces straight
  through nested lists of AnvlArrays passed as one argument — pass the
  whole model's weights as `w$double[[i]]$to_q` etc. instead of
  hundreds of positional args. This is the JAX-idiomatic shape and anvl
  supports it cleanly. Block builders are closures over static config
  returning `function(activations..., w)`.
- **Precompute parameter-free transforms host-side, pass as inputs.**
  The timestep sinusoid and the RoPE cos/sin tables have no weights and
  are deterministic per (timestep, resolution); computing them in base
  R and passing AnvlArrays keeps the jit boundary clean, mirrors how
  the diffusers pipeline precomputes `image_rotary_emb` outside the
  model, and sidesteps iota/dtype fiddliness. Fold into jit later only
  if a shape actually varies per call.
- **`nv_iota` is 1-based** (returns 1..n, R-style), so an `arange(0,
  n-1)` needs a `- 1`.
- The full f32 model (~15.5 GB weights) runs the parity test on **CPU**
  because it doesn't fit resident on a 16 GB GPU alongside activations
  — the concrete bf16-storage motivation (issue #379). Load each tensor
  to device and free the R copy immediately, or peak host memory is the
  full model twice.

### Qwen3-4B text encoder ported (July 2026)

The FLUX.2 klein text encoder (Qwen3-4B decoder stack, 27 of 36 layers
run) matched the torch reference at correlation 1.000000, relative
error 2.8e-05 (max 9.8e-04 at hidden-state sd ~35 — judge parity
relative to the tensor scale, not by a fixed absolute threshold; deep
residual streams are 100x the magnitude of the DiT's outputs). New
mechanics beyond the DiT, all composable from primitives:

- **Split-half RoPE** (Llama/Qwen `rotate_half`: pair element i with
  i + D/2) is a different kernel from FLUX's interleaved pairs — slice
  the head dim in half, `first*cos - second*sin | second*cos +
  first*sin`. Keep both in the toolkit.
- **GQA KV expansion** by interleave: `unsqueeze -> broadcast_to ->
  reshape` gives `repeat_interleave` order (each KV head repeated
  `groups` times consecutively). The naive tile order is wrong (same
  trap as the torch port).
- **Additive attention mask**: build the `[B,1,S,S]` causal+padding bias
  host-side and add it to scores inside SDPA — but anvl won't broadcast
  `[B,1,S,S]` against `[B,H,S,S]` (scalar-only), so `nv_broadcast_to`
  the mask to the full score shape first.
- **Embedding lookup host-side**: gather the token rows in R and pass
  `[B,S,hidden]` in. Keeps the 1.5 GB vocab table off the device and
  out of the jit graph — the lookup is parameter-free indexing, same
  boundary logic as precomputed RoPE/timestep.
- **Sharded checkpoints**: read `model.safetensors.index.json`, open
  each shard, dispatch reads by the key→shard map; klein needs only
  layers 0–26, so load to `max(out_layers)` and skip the rest plus the
  tied LM head.

The text encoder's concatenated mid-stack output (states 9/18/27 →
3×2560 = 7680) is exactly the DiT's `joint_attention_dim`, so the whole
conditioning → denoising path now runs on anvl.

### End-to-end sampling loop (July 2026)

Encoder → DiT → 4-step guidance-free FlowMatch Euler wired into one
text-to-latent pipeline, matching the torch reference at correlation
1.000000 (max 6.4e-05). Shape:

- **Host-driven step loop, jitted model called each step.** The DiT is
  `jit()`ed once and invoked `n_steps` times; latents stay AnvlArrays
  across steps (no host round-trip). The sigma schedule and per-step
  timestep sinusoid are host-computed scalars/arrays. `latents <-
  latents + velocity * nv_scalar(dt, "f32")` — the Euler update is a
  scalar-broadcast add, done eagerly between jitted DiT calls. No need
  to `nv_while` a 4-step loop; the host loop is simpler and the
  per-step compile is amortized after step 1.
- **Feed the reference's initial noise as a fixture input** (don't chase
  RNG): both sides start from identical packed noise, so a 4-step
  integration stays exact end-to-end rather than diverging.

**The view-save trap bit a second time, and it's the first thing to
suspect when random inputs pass but the fixture's own tensors fail.**
The packed initial latents were a `permute` view of a `reshape`;
`safetensors::safe_save_file` wrote it from the wrong strides, so the
saved fixture input was corrupted while the torch reference (computed
from the in-memory view) was correct. Signature: a per-op parity sweep
with random inputs is perfect, but the end-to-end fixture decorrelates
(cor ~0.15) — because only the fixture path reads the mangled tensor.
`$contiguous()` (or `$clone()`) every non-contiguous tensor before
`safe_save_file`; `permute`/`transpose`/`chunk`/narrow all return
views. (Unlike the size-1-dim chunk case, a genuine permute is
non-contiguous so `$contiguous()` does copy here.)

Remaining for pixels: only the VAE decoder, blocked on convolution
(stablehlo #161 / the anvl conv primitive).

### Convolution: rebasing stablehlo #161 + the anvl primitive (July 2026)

Built `prim_convolution` + `nv_conv2d`/`nv_conv3d` against Fischer's
open stablehlo conv PR (#161). All VAE-relevant cases match torch and a
base-R reference at f32 tolerance, eager and jitted: 2D stride 1/2,
symmetric padding, depthwise groups, 3D, and **causal asymmetric
padding** (the video-VAE case). Lessons:

- **Rebasing a stale PR across a refactor.** #161 predated the #168 op
  refactor that removed per-Op `repr.Op*` S3 methods in favour of a
  `render = function(ctx)` callback on `new_Op`. Cherry-pick onto main,
  resolve NAMESPACE to HEAD (drop the re-added `repr,Op*` exports), and
  convert the op's `repr.OpConvolution` into a `render_convolution(ctx)`
  reading `ctx$outputs_str`/`values_str`/`attrs`/`custom_attrs`/`sig_str`
  — model it on the sibling `render_dot_general`. The dimension-number
  reprs (`repr,ConvDimensionNumbers`) auto-merge; only the op-render
  block conflicts.
- **anvl primitive pattern for an op with dimension numbers + const
  attrs.** `new_primitive("convolution", impl, static = 3:10)` where the
  impl's `infer_fn` builds the StableHLO `ConvDimensionNumbers` (subtract
  1 from every anvl 1-based dim), wraps each window/padding/dilation arg
  with `r_to_constant(v, dtype = "i64", shape = ...)`, and calls
  `stablehlo::infer_types_convolution(at2vt(lhs), at2vt(rhs), ...)`. The
  `padding` const's `$data` must stay a matrix (`infer` calls
  `nrow`/`pad[i,]`), so pass the matrix, not a flattened vector. The
  `prim_X[["stablehlo"]]` lowering rule calls `hlo_convolution` directly
  (auto-imported via anvl's `@evalNamespace` over `hlo_*`). Reverse rule
  deferred (conv gradient is transposed conv; fine for inference).
- **Unexported constructor gap → fix upstream, not with `:::`.**
  `infer_types_convolution` required a `PrecisionConfig` object whose
  constructor stablehlo doesn't export. The clean fix is in stablehlo
  (make the infer fn also accept a character `precision_config` and
  coerce it, matching `hlo_convolution`), not a `stablehlo:::` reach from
  anvl. Belongs in the #161 PR patch, not the anvl PR.

### Full FLUX.2 Klein text-to-image on anvl (July 2026)

With convolution in hand, the VAE decoder (AutoencoderKLFlux2) ported
and the whole pipeline closed: text → Qwen3 encoder → DiT → FlowMatch
loop → VAE → RGB, matching torch end-to-end at correlation 1.000000
(full text→pixels max 1.3e-05). VAE-specific composition:

- **GroupNorm** = reshape `[B,C,H,W]` → `[B, groups, (C/g)*H*W]`,
  mean/var over the last axis, normalize, reshape back, then a
  per-channel affine broadcast from `[C]`.
- **Nearest-2× upsample** = reshape `[B,C,H,1,W,1]` → `broadcast_to`
  `[B,C,H,2,W,2]` → reshape `[B,C,2H,2W]` (row-major merge duplicates
  each pixel into a 2×2 block — no gather needed).
- **Conv weights load raw** `[out,in,kH,kW]` (what `nv_conv2d` wants);
  only the attention/linear weights transpose to `[in,out]`. Detect the
  optional ResNet `conv_shortcut` by checking key existence, not by
  recomputing channel arithmetic.
- **Normalization stats live where the data is packed.** FLUX.2's
  BatchNorm `running_mean/var` are **128 channels** — the denorm is on
  the DiT's packed representation, applied *before* the 2×2 unpatchify
  to 32 channels, not after. Read the stat's own shape to place it; a
  32-vs-128 guess crashes (or worse, silently mis-scales).
- Reshape/permute glue (token→grid unpack, unpatchify) is parameter-
  free, so it's cheapest host-side in base R — but R is column-major and
  torch reshape is row-major, so wrap array reshapes in an
  `aperm`-based torch-semantic reshape helper, matching `nv_reshape`.

### Install gotchas (Ubuntu 24.04, July 2026)

- pjrt source build needs `libprotobuf-dev` + `protobuf-compiler`
  (runtime libprotobuf is not enough; configure fails on
  `google/protobuf/message.h`).
- `install.packages()` from littler died with an empty `Error:` (no
  message, NULL call) for r-universe repos on this box; `curl` the
  tarballs and `R CMD INSTALL` in dependency order (tengen, xlamisc,
  stablehlo, pjrt, anvl) works fine.
- CUDA: the PJRT CUDA plugin (146MB, zml/pjrt-artifacts) auto-downloads
  to `~/.cache/R/pjrt/cuda` on first `device = "cuda"` use, but needs
  the mlverse `cuda12.8` package for toolkit libs (`libnvshmem_host`
  etc.); the package itself is a 56KB stub. Driver-only systems fail
  with a clear message naming the missing .so.

### Fixture trap: `safe_save_file` overflows int32 past 2 GB

`safetensors::safe_save_file` (the R writer) computes data offsets in
**int32**, so a fixture whose total tensor bytes exceed ~2 GB gets a
silently corrupted header (offsets go NaN / non-contiguous) — fatal for
large-model state-dict fixtures (a 2.6B-param UNet is 10 GB f32; even a
0.7B bigG text encoder is 2.8 GB). yunque's *reader* is fine (jsonlite
parses offsets as doubles, `seek()` takes doubles). Fix: write the
fixture with a small int64-safe, row-major F32 writer (compute offsets
as doubles, write the JSON header + raw little-endian f32 yourself). The
diffuseR-sdxl checkout that held a copy of that writer is gone; rewrite it
from this description when needed. Only affects large ports; sub-100 MB
fixtures (SD21 blocks, VAE) never hit it.

### Fixture trap: torch view tensors corrupt safetensors saves

When generating parity fixtures from torch: `$chunk()`/slice views
passed to `safetensors::safe_save_file` are written **from storage
offset 0** — every chunk silently saves as a copy of the first.
`$contiguous()` does NOT fix it (size-1 dims make view strides look
packed, so no copy happens); `$clone()` does. Symptom downstream:
parity fails with matching sd but correlation well below 1, and the
torch reference itself can't reproduce the fixture output. Check
suspiciously-equal saved tensors before blaming the port.
