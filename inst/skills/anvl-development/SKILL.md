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
from `anvil` to `anvl`; old links say anvil.

Authors: Sebastian Fischer, Daniel Falbel, Tomasz Kalinowski (Posit),
Nikolai German. MIT license. API facts in this skill were checked against
anvl 0.5.1 (October 2026). anvl 0.5 was a large breaking release: code
written for 0.3 needs porting (see "Upgrading from anvl 0.3").

## Ecosystem

| Package | Role |
|---|---|
| `anvl` | User-facing API: arrays, ops, `jit()`, `gradient()` |
| `stablehlo` | Builds StableHLO programs in R (anvl's lowering target) |
| `pjrt` | R interface to PJRT (the runtime that executes XLA binaries) |
| `xlamisc` | Shared generics and data types: `shape()`, `dtype()`, `as_array()`. Replaced `tengen` |
| `pjrt.cuda` | CUDA runtime libraries for pjrt |
| `quickr` | Experimental Fortran backend (CPU only, Suggests) |

Docs: https://r-xla.github.io/anvl/ · Repo: https://github.com/r-xla/anvl

## When to Use What

task: Install / set up CUDA
use: "Installation" below

task: Upgrade code written for anvl 0.3
use: "Upgrading from anvl 0.3"

task: Understand why a function recompiles or returns stale results
use: "The jit() Contract": cache keys, closures, branch baking

task: Express data-dependent logic (filtering, `which`, boolean masks)
use: "Static Shapes and the Masking Pattern"

task: Port an R torch / PyTorch model to anvl
use: "Layout: R Semantics, Not torch" first, then "Porting torch Models"

task: Debug wrong numbers after a port
use: "Validation": parity fixtures + treesitR AST audit

task: Benchmark or speed up anvl code
use: "Performance": await(), donate, bucketing

## Installation

```r
install.packages("anvl",
  repos = c("https://cloud.r-project.org", "https://r-xla.r-universe.dev"))
anvl::install_anvl()   # backend extras: the CUDA plugin and pjrt.cuda
```

- anvl is not on CRAN. Source builds need C++20, `protoc` and
  `libprotobuf-dev`.
- GPU platforms: Linux amd64 or WSL2. No macOS GPU.
- Debug device bring-up: set `PJRT_DEBUG=1` and `TF_CPP_MIN_LOG_LEVEL=0`
  in a fresh session, then `anvl::nv_scalar(1, device = "cuda")`.
- Other packages that load a different CUDA toolkit (torch) can conflict
  in one process. Isolate with {mirai} daemons if both are needed.
- Docker: `ghcr.io/r-xla/anvl-cuda:latest` (needs the NVIDIA Container
  Toolkit and `--gpus all`).
- `install.packages()` under littler (`r -e`) can fail with an empty
  `Error:`. Use `Rscript` for installs.

## Core Model

`AnvlArray` is the data structure: dtype + static shape + device, with
**value semantics** (never modified in place; eager subset-assign always
copies). 0-d arrays are scalars (`nv_scalar()`). Convert back with
`as_array()` / `as.double()`; construct with
`nv_array(x, shape =, dtype =, device =, byrow =)`. AnvlArrays survive
`saveRDS()` and load onto the default device.

Two execution modes:
- **Eager**: each op compiles and runs a tiny XLA program. Correct but
  slow; use it for debugging intermediate values.
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

Signature: `jit(f, static = character(), cache_size = 100L, device =
NULL, ...)`. `donate =` passes through `...` (pjrt backend only).

The backend is chosen when the jitted function is *called*
(`active_backend()`, set with `with_backend()` / `local_backend()`), so
one wrapper serves every backend, each with its own cache. `device =
"cuda:0"` pins the device and copies inputs to it; the default infers
the device from the inputs and falls back to `default_device()`.

**Cache key** = (shape, dtype) of each dynamic input + `identical()`
value of each static input + device. Cache misses re-trace and
recompile. Consequences:

- Call `jit()` **once** and reuse the wrapper. `jit()` inside a loop
  creates a fresh cache every iteration.
- Every new input shape recompiles. Bucket/pad varying axes (see
  Performance).

**Tracing rules** (violations produce silently wrong cached results,
not errors):

1. **Functions must be pure.** R-level side effects (assignments to
   the enclosing env, message/print, R's RNG) fire once during tracing,
   never again on cached calls.
2. **R `for`/`while` loops unroll at trace time.** n iterations = n
   copies of the ops in the graph. Fine for a 4 to 8 step denoise loop;
   catastrophic for thousands of steps. Use `nv_while()`.
3. **R `if` bakes one branch**, the one taken at trace time. Safe only
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
nv_if(pred, true, false)       # value-level branch
nv_ifelse(test, yes, no)       # elementwise select
```

## Static Shapes and the Masking Pattern

XLA requires every intermediate shape known at compile time. Inside
`jit()` there is **no** `unique()`, `which()`, `x[x > 0]`, boolean-mask
subsetting, or dynamic-length ranges.

Standard workaround: keep the shape, mask the values.

```r
# sum(x[x > 0]) becomes: replace non-matches with the neutral element
nv_sum(nv_ifelse(x > 0, x, 0), axes = 1L)

# mean over a mask: sum masked values / count
mask <- x > 0
nv_sum(nv_ifelse(mask, x, 0), axes = 1L) /
  nv_sum(nv_convert(mask, "f32"), axes = 1L)
```

Neutral elements: 0 (sum), 1 (prod), -Inf (max), +Inf (min), FALSE
(any), TRUE (all). Genuinely dynamic ops (`which`, `unique`) must exit
to R: `as_array()`, compute, re-enter.

## Dtypes and Promotion

Dtypes: `bool`, `i8/i16/i32/i64`, `ui8/ui16/ui32/ui64`, `f32`, `f64`.
**anvl has no f16 or bf16 yet** (pjrt has bf16 buffers since #241 and f16
is in review as #288; anvl integration is r-xla/anvl#379). No fp8, no
complex. Half-precision checkpoints must be upcast to f32, so a model
needs 2x its bf16 torch footprint. Plan VRAM accordingly.

`default_dtypes()` gives the defaults: R doubles become `f32`, R
integers `i32`, logicals `bool` (option `anvl.default_dtypes`).

R values have no dtype of their own: an R number combined with an array
is *built at* the dtype the arrays agree on, so `pi` against an `f64`
array keeps full precision and against `f32` stays fast. The category
still counts: an R double against an integer array gives a float.

```r
x <- nv_array(1:3, dtype = "i16")
dtype(x + 1L)   # i16
dtype(x + 1)    # f32: the double literal pulls in the default float
```

A value the target dtype cannot hold is an error (`x + (-2L)` on `ui8`),
while `nv_convert()` keeps XLA's wraparound. `common_dtype()` answers
"what would these combine to". Arrays are checked for `NA` on creation:
anvl has no NA.

## Layout: R Semantics, Not torch

Since anvl 0.5 the package follows base R everywhere. **This is the
first thing to get right in any torch port.**

- **`nv_reshape()` is column-major**, like `array()` / `dim<-`. torch's
  `reshape`/`view` is row-major. A reshape that only adds or drops
  size-1 axes is the same in both; one that splits or merges real axes
  (head split `[B, S, H*D] -> [B, S, H, D]`, interleaved pairs, GroupNorm
  groups, GQA expansion, pixel shuffles) regroups elements differently
  and silently gives wrong numbers.
- **`nv_array(vec, shape =)` fills column-major**; pass `byrow = TRUE`
  for data that arrives in torch (row-major) order.
- **Broadcasting aligns on the left.** `nv_broadcast_to()` and
  `nv_broadcast_arrays()` match axis 1 with axis 1, like base R, so a
  `[D]` weight against `[B, S, D]` lines up with `B` (an error, or a
  silent mismatch when `B == D`). numpy aligns on the right.

The torch-order reshape, by reversing axes (row-major order of `x` is
column-major order of `x` with its axes reversed):

```r
reshape_rm <- function(x, shape) {
  n <- length(shape(x))
  if (n > 1L) x <- nv_transpose(x)            # default perm reverses
  x <- nv_reshape(x, rev(shape))
  if (length(shape) > 1L) x <- nv_transpose(x)
  x
}
```

XLA folds the transposes into the surrounding ops. numpy-style
broadcasting: pad the rank with leading size-1 axes (an order-safe
reshape), then broadcast:

```r
bcast_to <- function(x, shape) {
  s <- shape(x)
  k <- length(shape) - length(s)
  if (k > 0L) x <- nv_reshape(x, c(rep(1L, k), s))
  nv_broadcast_to(x, shape)
}
```

yunque does both internally, and `yunque::badd()` / `bsub()` / `bmul()` /
`bdiv()` apply numpy broadcasting to two arrays; use its building blocks
rather than re-deriving them. Weights are unaffected: yunque's `st_read()` returns
R arrays whose `[i, j, ...]` indexing matches the torch tensor, and
`nv_array()` of an R array keeps that indexing.

## Broadcasting Is Explicit (Scalars Excepted)

Binary ops (`+`, `*`, `nv_add`, ...) broadcast **only 0-d scalars**
automatically. `(B, S, D) + (B, 1, D)` is an error. Expand first:

```r
e / nv_broadcast_to(s, shape(e))     # same rank, size-1 axes expand
nv_broadcast_arrays(x, y)            # mutual broadcast, returns a list
```

Both align ranks on the left (see Layout). Every reduction with
`drop = FALSE` that feeds a binary op needs an explicit
`nv_broadcast_to`. This is the most frequent porting error from torch.

## Subsetting

1-based, like R. Static indices (R values) vs dynamic indices
(AnvlArrays) behave differently:

| Form | Static | Dynamic |
|---|---|---|
| Single index `x[2]` / `x[nv_scalar(2L)]` | yes | yes |
| Multiple `x[arr(2, 4, 6)]` / `x[nv_array(c(2L, 4L, 6L))]` | yes | yes |
| Range `x[2:5]` | yes | static size only |
| Boolean mask | no | no |
| Negative index | no | no |

`x[, , 2L]` drops the indexed axis. Static slices with strides:
`nv_static_slice(x, start_indices, end_indices, strides)`, 1-based and
inclusive.

Gotchas:
- **Static** out-of-bounds indices error at compile time. **Dynamic**
  out-of-bounds indices are silently **clamped** on read and silently
  **dropped** on write. There are no runtime bounds errors, so validate
  index math yourself.
- Duplicate indices in one subset-assign: which write wins is
  backend-defined (CPU and GPU can differ).
- Eager subset-assign always copies the full array. Put updates inside
  `jit()` where XLA fuses them.

## RNG: Explicit State Threading

No global `.Random.seed`. State comes from `nv_rng_state(seed)`; every
sampler takes it and returns `list(state =, values =)`:

```r
state <- nv_rng_state(42L)
r1 <- nv_runif(shape = 3L, state = state)
state <- r1$state                    # THREAD IT: reusing the old
r2 <- nv_rnorm(shape = 3L, state = state)   # state repeats values
```

Samplers: `nv_runif(min, max)`, `nv_rnorm(mean, sd)`,
`nv_rbinom(size, prob)`, `nv_sample_int(n)`, named like base R. Use
named arguments. The generator is XLA's `rng_bit_generator`, with **no
seed parity with R, torch, or diffusers.** For output parity with a
torch port, generate noise in the reference system and pass it in as an
input (same trick as parity fixtures).

## Weight I/O

`nv_save(arrays, path)` / `nv_read(path, device =)` speak safetensors (a
named list of AnvlArrays per file). For HuggingFace / torch checkpoints,
yunque's base-R reader (`st_open()` / `st_read()` / `st_open_sharded()`,
BF16 and F16 upcast to f32, partial and sharded reads) avoids a torch
dependency. Read one tensor at a time and `nv_array()` it to the device,
freeing the R copy, or peak host memory is the whole model twice.

## Porting torch Models

### Op mapping

| torch | anvl |
|---|---|
| `torch_matmul` (batched) | `nv_matmul(x, y, precision = "highest")`; leading axes are batch axes |
| `$transpose(i, j)` / `$permute(p)` | `nv_transpose(x, perm = p)` (default reverses all axes) |
| `$reshape` / `$view` | **row-major**: not `nv_reshape` directly, see Layout |
| `$unsqueeze` / `$squeeze` | `nv_unsqueeze(x, axes)` / `nv_squeeze(x, axes)` |
| `torch_cat(list, dim)` | `nv_concatenate(..., axis =)` / `nv_rbind` / `nv_cbind` |
| `$flip(dims)` | `nv_rev(x, axes)` |
| `nnf_pad` (constant) | `nv_pad(x, value, low, high, interior)` |
| `torch_where` | `nv_ifelse(test, yes, no)` |
| `$clamp` | `nv_clamp(x, min, max)` |
| `torch_maximum` / `minimum` | `nv_pmax` / `nv_pmin` (`nv_max`/`nv_min` are reductions) |
| `$sum(dim)` / `$amax(dim)` / `$mean(dim)` | `nv_sum` / `nv_max` / `nv_mean` with `axes =`, `drop =` |
| `x$dim()` | `naxes(x)` |
| `nnf_sigmoid` | `nv_plogis` |
| `torch_rsqrt`, `tanh`, `erf`, ... | `nv_rsqrt`, `nv_tanh`, `nv_erf`, ... (101 primitives; most elementwise math exists) |
| `torch_sort` / `topk` / `argmax` | `nv_sort` / `nv_top_k` / `nv_which_max` |
| `torch_cumsum` | `nv_cumsum(x, axis)` |
| `index_select` / gather / scatter | `nv_select(x, axis, index)`, `prim_gather`, `prim_scatter` |
| `conv1d/2d/3d` | `nv_conv1d` / `nv_conv2d` / `nv_conv3d` (NC... layout, torch weight layout) |
| `chol/qr/svd/solve/eigh/lu` | `nv_chol/nv_qr/nv_svd/nv_solve/nv_eigh/nv_lu` (`nv_chol` uses LAPACK / cuSOLVER `potrf`) |

### What does NOT exist (checked against 0.5.1)

- **No softmax / layer_norm / group_norm / gelu / attention builtins in
  anvl.** The **yunque** package provides them: `softmax`, `layer_norm`,
  `rms_norm`, `group_norm`, `batch_norm`, `silu`, `gelu`, `mish`, `elu`,
  `snake`, `linear`, `sdpa`, `rope_apply` / `rope_split`, `repeat_kv`,
  `rel_position_attention`, `upsample_nearest2d`, `conv1d/2d/3d` with
  bias, `conv_transpose1d`, `lstm`, `stft` / `istft`, numpy-style
  `badd` / `bsub` / `bmul` / `bdiv`, the DiT pieces `modulate`,
  `gated_residual` and `timestep_embedding`, `pad_to` / `seq_mask` /
  `causal_mask` for static-shape batching, the declarative weight tree
  (`w_linear`, `w_tensor`, `w_cat`, `w_layers`, `load_weights`), and the
  safetensors reader. Reach for yunque before composing from scratch.
- **No transposed convolution.** yunque's `conv_transpose1d` builds it
  from interior padding plus a flipped-kernel conv.
- **No einsum.** Reformulate as `nv_transpose` + `nv_matmul`
  (`prim_dot_general` covers arbitrary contract/batch axes).
- **No f16/bf16** in anvl yet (see Dtypes). No quantized dtypes; `ui8` +
  `nv_convert` + shift/and ops could express NF4-style dequant by hand.
  bf16 does not make 12B/22B models fit a 16GB GPU (still 24/44GB); that
  needs NF4. bf16's payoff is speed for 4B-class models and lighter CPU
  loads.
- **No nn_module.** Models are pure functions; parameters travel as a
  named list argument (JAX pytree style; anvl traces through nested
  lists in inputs).

### Composed building blocks

```r
softmax_lastdim <- function(x) {
  d <- naxes(x)
  m <- nv_max(x, axes = d, drop = FALSE)
  e <- nv_exp(x - nv_broadcast_to(m, shape(x)))
  e / nv_broadcast_to(nv_sum(e, axes = d, drop = FALSE), shape(x))
}

# q, k, v: (B, H, S, D); attention as batched matmul
attn <- function(q, k, v, scale) {
  scores <- nv_matmul(q, nv_transpose(k, c(1L, 2L, 4L, 3L))) * scale
  nv_matmul(softmax_lastdim(scores), v)
}
```

This materializes the full (B, H, S, S) score matrix, the same memory
story as pre-SDPA torch. XLA may fuse, but budget for the worst case at
long sequence lengths.

### Structure of a ported model

```r
# params: named list mirroring the checkpoint key tree (same 1:1
# census discipline as torch ports: every key lands, every slot fills)
model_step <- function(params, latents, t_emb) {
  h <- nv_matmul(latents, params$in_proj$weight)  # ...
  h
}
step <- jit(model_step)          # params traced as dynamic inputs
```

Inference needs no `with_no_grad` equivalent: gradients only exist
where you ask for them with `gradient()`.

### The porting playbook (proven on FLUX.2, reused across the model loop)

Three layers, kept separate:
- **anvl**: XLA primitives (`nv_*`, `jit`, convolution).
- **yunque** (github.com/cornball-ai/yunque): model-agnostic NN building
  blocks composed from anvl with torch semantics (row-major reshapes,
  right-aligned broadcasting), plus the base-R safetensors reader.
- **the model package** (e.g. diffuseR's `anvl_*.R` files): the specific
  architecture, beside its torch reference so parity tests can
  `library()` both. Functions call `yunque::` / `anvl::` qualified (both
  are Suggests, so the torch path is untouched).

Per-model recipe, one component at a time (block, encoder, VAE, loop,
end to end):
1. **Read the torch reference fully** and mirror its module tree 1:1.
   Each block/module becomes a closure over static config returning
   `function(activations..., w)`; weights travel as a nested named list
   (pytree) mirroring the checkpoint key tree.
2. **Weight loader**: `st_open` the checkpoint, read per tensor and
   `nv_array` it straight to the device. Conv weights load raw
   `[out, in, kH, kW]`; linear/attention weights transpose to
   `[in, out]` (yunque's `linear()` takes the transposed weight). Detect
   optional sub-modules (ResNet shortcut convs) by key existence.
3. **Mark every torch reshape/view.** Each one that splits or merges
   real axes needs the row-major reshape (see Layout). This is where
   0.5-era ports go wrong silently.
4. **Parity fixture from torch**: load the SAME checkpoint into the
   torch reference, run on fixed random inputs (feed the reference's
   noise as an input), save inputs + output (not weights; the anvl side
   reloads them). `$clone()` every saved tensor (the view-save trap).
   Run torch with `TORCH_CUDATOOLKIT=FALSE` if a broken CUDA toolkit is
   installed.
5. **Parity test**: jit the anvl component, load the same weights,
   compare. Judge by correlation + max-abs-diff **relative to the tensor
   scale** (deep residual streams run sd ~35, so a 1e-3 absolute diff is
   still 2.8e-5 relative: f32 rounding, not a bug). Target correlation
   1.000000. Matching sd but low correlation means suspect a fixture
   view-save bug, or a column-major reshape, before anything else.
6. **Debug divergence** by dumping per-stage intermediates from both
   sides and binary-searching the first mismatch. That localizes the bug
   (wrong axis, missing residual, transposed weight, reshape order).

Random-init weights validate architecture parity as well as real weights
and skip the checkpoint download; use real weights when you also want a
runnable end-to-end result. All parity is CPU f32, independent of GPU
availability.

## Validation

Same discipline as torch ports, plus the standing treesitR rule:

1. **Parity fixtures**: run the reference (torch or Python) on fixed
   inputs, save inputs + outputs to safetensors, compare the anvl port
   at f32 tolerances (max diff < 1e-5 elementwise for a single block;
   looser end to end).
2. **Noise as input**: never try to match RNG streams; feed the
   reference's noise tensors in as arguments.
3. **treesitR AST audit** (mandatory finish step for any port): diff
   numeric literals and callee names per paired scope between the
   reference and the anvl R code. Pattern: `r tools/compare_translation.R`
   in diffuseR; extend its pairings. Parity tests prove the tested
   paths; the AST diff catches unported branches, dropped constants, and
   missing ops the fixtures never hit. R-torch-to-anvl redos parse both
   sides with `ts_language_r()`.
4. Eager mode is the layer-by-layer debugger: run the un-jitted function
   and `as_array()` intermediates. Then confirm the jitted output
   matches eager output exactly before trusting the cache.
5. **Test reshape-heavy helpers against a loop reference.** A base-R
   reference that computes each output element from its definition (no
   reshapes) catches order bugs that a reference built with `array()`
   would share.

## Performance

- **Everything is async.** Calls return before the computation
  finishes; results await implicitly on print / `as_array()`. Benchmarks
  MUST call `await(result)` inside the timed region or they measure
  dispatch, not compute. XLA runs small ops synchronously and large ones
  in the background, so wrong benchmarks can look plausible.
- **Exploit the async gap**: in a training/denoise loop, do R-side prep
  for step i+1 while the device works on step i. Printing per step
  forces a sync and idles the GPU.
- **`donate`**: `jit(function(w, g) w - lr * g, donate = "w")` lets XLA
  reuse the input buffer for the output, the update-in-place idiom for
  loops. Touching a donated array afterwards errors.
- **Matmul precision**: `nv_matmul()` and the convs default to
  `precision = "highest"` (strict f32). Lower it only on purpose: TF32
  dots measured 1.4e-03 scaled error against 2.3e-06.
- **Shape bucketing**: pad varying axes to bucket multiples so calls
  share compiled executables; mask the padding (neutral values).
- **Keep data on device**; CPU-GPU transfers have high fixed cost. Small
  ops on small arrays can be slower on GPU than CPU.
- Eager mode is inherently slow (one XLA program per op); never
  benchmark eager and conclude anything about anvl.
- CPU linear algebra inherits R's BLAS / LAPACK, so OpenBLAS or MKL
  matters. Thread control is OS-level: `taskset -c 0-3 Rscript ...`.
- Compilation cost is real for big graphs. A fully unrolled 48-block
  transformer step is one giant program: expect a long first call and
  measure whether `nv_while` over blocks (weights stacked along a leading
  axis) compiles faster at equal runtime.

## Upgrading from anvl 0.3

anvl 0.4 and 0.5 renamed most of the API toward base R and switched to
column-major semantics. The changes that broke yunque's port:

| anvl 0.3 | anvl 0.5 |
|---|---|
| `nv_reshape` row-major | column-major; use a row-major helper for torch reshapes |
| `nv_broadcast_to([D], [B, S, D])` | left-aligned; pad leading 1s first |
| `ndims(x)` | `naxes(x)` |
| `dim =`, `dims =`, `dimension =` | `axis =` / `axes =` |
| `nv_reduce_sum/max/...` | `nv_sum/nv_max/...` (`nv_max`/`nv_min` used to be elementwise) |
| `nv_max(x, 0)` elementwise | `nv_pmax(x, 0)` |
| `nv_logistic` | `nv_plogis` |
| `nv_reverse(x, dims)` | `nv_rev(x, axes)` |
| `nv_static_slice(limit_indices =)` | `end_indices =` |
| `nv_pad(edge_padding_low/high, interior_padding)` | `low`, `high`, `interior` |
| `nv_rdunif`, `lower/upper`, `mu/sigma` | `nv_sample_int`, `min/max`, `mean/sd` |
| `initial_state =`, `r[[1]]` | `state =`, `r$state` |
| `jit(backend =, device_arg =)` | `with_backend()`, `jit(device =)` |
| `"i1"`, ambiguous literal types | `"bool"`, R values take the arrays' dtype |
| `tengen` | `xlamisc` |

Old calls mostly fail loudly (unused argument, missing export). The two
that can fail silently are reshape order and left-aligned broadcasting
when sizes happen to match, so test those paths against loop references.

## Resources

All under https://r-xla.github.io/anvl/articles/:
`anvl` (get started), `jit`, `static_shapes`, `subsetting`,
`type-promotion`, `random-numbers`, `autodiff`, `efficiency`,
`gotchas`, `installation`, `extending_api`, `extending_primitive`,
`primitives` (the primitive table), `internals`, `faq`, `next_steps`,
plus worked examples: `gaussian-process`, `metropolis-hastings`.

## Lessons Learned

<!-- Updated as the diffuseR-on-anvl redo hits real problems. Entries
are dated; API names in them are from the anvl version of that date. -->

### anvl 0.5 switched to R layout semantics (October 2026)

yunque, written against anvl 0.3, broke in 8 of 9 test files on
upgrading to 0.5.1. Most were loud renames. The dangerous part was
semantic: `nv_reshape` became column-major and `nv_broadcast_to`
left-aligned. Head splits, RoPE pair interleaving, GroupNorm grouping,
GQA expansion and the nearest-2x upsample all needed a row-major
reshape helper, and five of those functions had no test that would have
caught the wrong order. Lesson: after any anvl upgrade, grep every
`nv_reshape` and `nv_broadcast_to` and ask whether it splits or merges
real axes or aligns lower-rank inputs. A reshape that only adds or drops
size-1 axes is safe either way.

### First real port: FLUX.2 Klein single-stream block (July 2026, anvl 0.3)

Full block (modulated LayerNorm, fused QKV+MLP projection, RMS-normed
q/k, RoPE, SDPA, SwiGLU, gated residual) composed from nv_* primitives
matched the R torch reference on real checkpoint weights at
**max 3e-06** (S = 512, f32). One Klein block, S = 4608, batch 1,
RTX 5060 Ti, steady state:

| Config | ms/iter |
|---|---|
| torch CUDA bf16 (production) | 53.0 |
| anvl CUDA f32 (jit, TF32 dots) | 81.7 |
| anvl CUDA f32 (jit, strict f32) | 126.6 |
| torch CUDA f32 | 161.8 |
| torch CPU f32 | 1473.6 |
| anvl CPU f32 (jit) | 1553.9 |

XLA fusion beat eager torch by 1.28x at matched strict f32; TF32 is an
extra gear torch-R doesn't expose. CPU is a wash. jit compile + first
call was ~4s (GPU) / ~1.6s (CPU) for the one-block graph. Always verify
TF32 vs strict f32 before claiming "equal precision" in a benchmark.

### Full FLUX.2 Klein DiT ported (July 2026, anvl 0.3)

The whole conv-free transformer (5 double-stream MMDiT blocks with joint
attention + 20 single-stream parallel blocks + embedders, shared
modulation, adaLN-continuous output norm) matched the R torch reference
end to end at **max 2.5e-05, correlation 1.000000** on real 3.876B-param
Klein weights. Patterns that worked:

- **Weights as a nested named list (pytree).** jit traces straight
  through nested lists of AnvlArrays passed as one argument: pass the
  whole model's weights as `w$double[[i]]$to_q` etc. instead of hundreds
  of positional args. Block builders are closures over static config
  returning `function(activations..., w)`.
- **Precompute parameter-free transforms host-side, pass as inputs.**
  The timestep sinusoid and the RoPE cos/sin tables have no weights and
  are deterministic per (timestep, resolution); computing them in base R
  and passing AnvlArrays keeps the jit boundary clean and mirrors how
  the diffusers pipeline precomputes `image_rotary_emb`. Fold into jit
  later only if a shape actually varies per call.
- **`nv_iota` is 1-based** (returns 1..n), so an `arange(0, n - 1)`
  needs a `- 1`.
- The full f32 model (~15.5 GB weights) runs the parity test on **CPU**
  because it doesn't fit resident on a 16 GB GPU alongside activations,
  the concrete bf16 motivation (anvl#379).

### Qwen3-4B text encoder ported (July 2026, anvl 0.3)

The FLUX.2 klein text encoder (Qwen3-4B decoder stack, 27 of 36 layers
run) matched the torch reference at correlation 1.000000, relative error
2.8e-05 (max 9.8e-04 at hidden-state sd ~35; judge parity relative to
the tensor scale). Mechanics beyond the DiT:

- **Split-half RoPE** (Llama/Qwen `rotate_half`: pair element i with
  i + D/2) is a different kernel from FLUX's interleaved pairs. Keep
  both (`rope_split`, `rope_apply`).
- **GQA KV expansion** by interleave: `unsqueeze -> broadcast_to ->
  row-major reshape` gives `repeat_interleave` order (each KV head
  repeated `groups` times consecutively). The naive tile order is wrong.
- **Additive attention mask**: build the `[B, 1, S, S]` causal+padding
  bias host-side and broadcast it to the full score shape before adding.
- **Embedding lookup host-side**: gather the token rows in R and pass
  `[B, S, hidden]` in. Keeps the 1.5 GB vocab table off the device and
  out of the jit graph.
- **Sharded checkpoints**: read `model.safetensors.index.json` and
  dispatch reads by the key-to-shard map; load only the layers you run.

### End-to-end sampling loop (July 2026, anvl 0.3)

Encoder, DiT and a 4-step guidance-free FlowMatch Euler loop wired into
one text-to-latent pipeline, matching torch at correlation 1.000000
(max 6.4e-05):

- **Host-driven step loop, jitted model called each step.** The DiT is
  `jit()`ed once and invoked `n_steps` times; latents stay AnvlArrays
  across steps. The Euler update is a scalar-broadcast add done eagerly
  between jitted calls. No need to `nv_while` a 4-step loop.
- **Feed the reference's initial noise as a fixture input** so a 4-step
  integration stays exact end to end.

### Convolution: rebasing stablehlo #161 + the anvl primitive (July 2026)

Built `prim_convolution` + `nv_conv2d`/`nv_conv3d` against Fischer's
stablehlo conv PR (#161); merged as anvl #385. All VAE-relevant cases
matched torch: 2D stride 1/2, symmetric padding, depthwise groups, 3D,
and causal asymmetric padding. Lessons:

- **Rebasing a stale PR across a refactor.** #161 predated the op
  refactor that replaced per-Op `repr.Op*` methods with a
  `render = function(ctx)` callback on `new_Op`. Convert the op's repr
  into a render function modeled on the sibling `render_dot_general`.
- **anvl primitive pattern for an op with dimension numbers + constant
  attrs.** `new_primitive("convolution", impl, static = 3:10)`; the
  `infer_fn` builds the StableHLO `ConvDimensionNumbers` (subtract 1
  from every 1-based axis) and wraps window/padding/dilation args as
  `i64` constants. The `padding` constant's `$data` must stay a matrix.
- **Unexported constructor gap: fix upstream, not with `:::`.**

### Full FLUX.2 Klein text-to-image on anvl (July 2026, anvl 0.3)

The VAE decoder ported and the whole pipeline closed: text, Qwen3
encoder, DiT, FlowMatch loop, VAE, RGB, matching torch end to end at
correlation 1.000000 (max 1.3e-05). VAE-specific composition:

- **GroupNorm** = row-major reshape `[B, C, H, W] -> [B, groups,
  (C/g)*H*W]`, mean/var over the last axis, normalize, reshape back,
  per-channel affine.
- **Nearest-2x upsample** = reshape `[B, C, H, 1, W, 1]`, broadcast to
  `[B, C, H, 2, W, 2]`, row-major reshape to `[B, C, 2H, 2W]`.
- **Normalization stats live where the data is packed.** FLUX.2's
  BatchNorm `running_mean/var` are **128 channels**: the denorm applies
  to the DiT's packed representation before the 2x2 unpatchify to 32
  channels. Read the stat's own shape to place it.
- Host-side reshape/permute glue in base R also needs torch order: wrap
  `array()` reshapes in an `aperm`-based row-major helper.

### Fixture trap: `safe_save_file` overflows int32 past 2 GB

`safetensors::safe_save_file` (the R writer) computes data offsets in
**int32**, so a fixture whose tensor bytes exceed ~2 GB gets a silently
corrupted header. yunque's *reader* is fine (offsets parse as doubles).
Fix: write large fixtures with a small int64-safe, row-major F32 writer
(offsets as doubles, write the JSON header + raw little-endian f32
yourself). Sub-100 MB fixtures never hit it.

### Fixture trap: torch view tensors corrupt safetensors saves

`$chunk()`/slice/`permute` views passed to `safetensors::safe_save_file`
can be written from the wrong storage offset or strides: every chunk
silently saves as a copy of the first, or a permuted latent saves
scrambled. `$contiguous()` does NOT always fix it (size-1 axes make view
strides look packed, so no copy happens); `$clone()` does. Symptom: a
per-op parity sweep with random inputs is perfect, but the fixture
decorrelates (matching sd, correlation well below 1). Check suspiciously
equal or scrambled saved tensors before blaming the port.
