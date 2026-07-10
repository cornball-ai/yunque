# yunque

<!-- badges: start -->
[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
<!-- badges: end -->

Neural-network building blocks for [{anvl}](https://github.com/r-xla/anvl):
broadcast-aware ops, composed attention, rotary embeddings, and ports of
diffusion-model transformer blocks. *Yunque* is Spanish for anvil — this
is the forge accessory.

yunque is a downstream, diffusion-focused layer built to port
[diffuseR](https://github.com/cornball-ai/diffuseR) models onto XLA.
[{alloy}](https://github.com/r-xla/alloy) is intended as the anvl
ecosystem's broader modeling framework; we'd be glad to coordinate on
reusable operators and conventions.

## Installation

```r
install.packages("anvl", repos = c("https://r-xla.r-universe.dev", getOption("repos")))
pak::pak("cornball-ai/yunque")
```

## What's here

| Function | Purpose |
|---|---|
| `yq_softmax()`, `yq_layer_norm()`, `yq_rms_norm()`, `yq_silu()` | Composed ops with the explicit-broadcast dance done for you |
| `yq_linear()` | Bias-free linear on pre-transposed weights, `precision = "highest"` |
| `yq_sdpa()` | Scaled dot-product attention from batched matmul + softmax |
| `yq_rope_apply()`, `yq_flux2_rope()` | Rotary embeddings (FLUX interleaved-pair) and the FLUX.2 4-axis table builder |
| `yq_slice_lastdim()`, `yq_slice_seq()` | Static slicing for fused-projection and text/image splits |
| `yq_flux2_single_block()`, `yq_flux2_double_block()` | FLUX.2 Klein single- and double-stream blocks as jit-ready closures |
| `yq_flux2_transformer()` | The full FLUX.2 Klein DiT forward (5 double + 20 single blocks) |
| `yq_qwen3_encoder()` | The FLUX.2 Klein text encoder: Qwen3-4B decoder stack (GQA, split RoPE, causal+padding mask), mid-stack states concatenated |
| `yq_flux2_sample()`, `yq_flux2_sigmas()` | The FlowMatch Euler denoising loop and its dynamic-shift sigma schedule (guidance-free, 4 steps) |
| `yq_rope_split()`, `yq_repeat_kv()` | Llama/Qwen split-half RoPE and grouped-query KV expansion |
| `yq_flux2_vae_decode()`, `yq_flux2_vae_prepare()` | AutoencoderKLFlux2 decoder (ResNet/attention/upsample blocks) and the unpack + BN-denorm + unpatchify glue |
| `yq_group_norm()`, `yq_upsample_nearest2d()` | GroupNorm and nearest-2× upsampling over NCHW |
| `yq_flux2_load_weights()`, `yq_qwen3_load_weights()`, `yq_flux2_load_vae()`, `yq_read_safetensors()` | bf16→f32 checkpoint loaders (base R, no torch; sharded-aware) into weights pytrees |

anvl binary ops broadcast scalars only — no implicit numpy/torch-style
shape broadcasting. These helpers wrap every reduction-feeds-binary-op
site in `nv_broadcast_to()`, which is the single biggest porting hazard
coming from torch.

## FLUX.2 Klein single block: parity and benchmark

One `single_transformer_blocks.0` from
[black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
(Apache-2.0), revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`,
3.876B parameters total (7.8 GB bf16 / 15.5 GB f32). Weights are read
from your local HuggingFace cache; nothing is redistributed here.

**Parity**, jitted anvl vs the R torch reference on real weights (f32):

| Unit | Max abs diff | Correlation |
|---|---|---|
| DiT single-stream block (S = 512) | 2.97e-06 | — |
| DiT double-stream block (S = 320) | 2.86e-06 | — |
| **full DiT forward** (25 blocks, 256 img + 64 txt tokens) | **2.52e-05** | **1.000000** |
| **full Qwen3-4B text encoder** (27 layers, S = 32) | **9.8e-04** (rel 2.8e-05) | **1.000000** |
| **end-to-end text → latent** (encoder → DiT → 4-step loop) | **6.4e-05** | **1.000000** |
| **VAE decoder** (AutoencoderKLFlux2, 32-ch → RGB) | **4.7e-06** | **1.000000** |
| **full text → pixels** (encoder → DiT → loop → VAE) | **1.3e-05** | **1.000000** |

**The entire FLUX.2 Klein text-to-image pipeline runs on anvl** and
matches the diffuseR torch reference to f32 tolerance end to end:
tokens through the Qwen3-4B encoder (mid-stack states 9/18/27
concatenated → 3 × 2560 = 7680, the DiT's `joint_attention_dim`), the
DiT under a 4-step guidance-free FlowMatch Euler loop, then the
AutoencoderKLFlux2 decoder (via `nv_conv2d`) to RGB pixels.

The VAE needs convolution, which is not yet in a released anvl — it
depends on the `hlo_convolution` op
([r-xla/stablehlo#161](https://github.com/r-xla/stablehlo/pull/161))
plus a downstream `prim_convolution`/`nv_conv2d` in anvl. Until those
land, the VAE stage requires those branches installed.

Both full-model tests load their checkpoints (DiT 3.876B params ~15.5 GB
f32; text encoder 27 layers) and run on CPU — at f32 the weights don't
fit resident on a 16 GB GPU alongside activations, which is exactly the
bf16 storage motivation ([r-xla/anvl#379](https://github.com/r-xla/anvl/issues/379)).
Per-block GPU timing is below.

**Benchmark** (S = 4608, batch 1, steady-state ms/iter, RTX 5060 Ti
16GB, driver 595.71.05):

| Config | Dot precision | ms/iter |
|---|---|---|
| torch CUDA bf16 | bf16 | 53.0 |
| anvl CUDA f32, `precision = "default"` (jit) | TF32 | 81.9 |
| anvl CUDA f32, `precision = "highest"` (jit) | strict f32 | 126.6 |
| torch CUDA f32 | strict f32 | 161.8 |
| torch CPU f32 | strict f32 | 1473.6 |
| anvl CPU f32 (jit) | strict f32 | 1553.9 |

The equal-precision comparison is the two strict-f32 CUDA rows: anvl is
1.28x faster than torch. The TF32 row is a real extra gear (mlverse
torch exposes no TF32 toggle) but it is not equal precision.

Dot precision was verified empirically (512x3072 @ 3072x512 vs a
float64 reference, error scaled by sd): strict f32 rows measure
~2.3e-06 on both frameworks; TF32 measures 1.4e-03. **anvl 0.3.0
(current release) has no precision parameter and its CUDA dots run
TF32 regardless** — these numbers use the dev branch, where
`nv_matmul(precision =)` exists and defaults to `"highest"`. libtorch's
matmul TF32 default is off.

Versions: anvl 0.3.0.9000 (e124cec), stablehlo 0.3.0.901 (3ca67f5),
pjrt 0.4.0.9000 (c4fb30d), tengen 0.2.0.9000 (4adc1c2), xlamisc 0.3.0,
R torch 0.17.0, safetensors 0.2.0.9000, R 4.6.0.
Timing: 20 iterations; torch warms up 3 calls and `cuda_synchronize()`s
before and after the timed loop under `with_no_grad()`; anvl's
compile+first call is `await()`ed before the timed loop starts and the
final result is `await()`ed inside the timed region.

### Reproduce

```sh
# 1. Fixture: reads 4 real bf16 tensors from the checkpoint (partial
#    read), runs the diffuseR torch reference, writes a 504 MB f32
#    safetensors to ~/.local/share/R/yunque/fixtures/.
#    Expected sha256: 028cb8a0a47b51d669789dfe4a2146fdb4fcc35806bf4d587c115038b0384281
r tools/gen_fixture_flux2_block.R

# 2. Parity: single block, double block, and the full DiT forward.
#    The forward test needs the full checkpoint in your HF cache and
#    tools/gen_fixture_flux2_forward.R run once first.
r -l yunque,tinytest -e 'Sys.setenv(TT_AT_HOME = "TRUE"); for (t in c("test_flux2_block.R","test_flux2_double.R","test_flux2_forward.R")) tinytest::run_test_file(file.path("inst/tinytest", t))'

# 3. Benchmarks
r tools/bench_flux2_block_torch.R float32 4608 cuda
r tools/bench_flux2_block_torch.R bfloat16 4608 cuda
r tools/bench_flux2_block_anvl.R cuda 4608 highest
r tools/bench_flux2_block_anvl.R cuda 4608 default
```

The fixture generator needs the checkpoint in your HF cache
(`hfhub::hub_snapshot("black-forest-labs/FLUX.2-klein-4B")` or any HF
download) and [diffuseR](https://github.com/cornball-ai/diffuseR) for
the torch reference. The torch scripts need R torch with CUDA.

### A trap worth knowing

When generating fixtures from torch: tensor **views** (`$chunk()`,
slices) passed to `safetensors::safe_save_file()` are written from
storage offset 0 — every chunk silently saves as a copy of the first.
`$contiguous()` does not fix it (size-1 dims make view strides look
packed, so no copy happens); `$clone()` does. Symptom: parity fails
with matching sd but correlation well below 1, and the torch reference
itself can't reproduce the fixture.

## License

MIT © cornball.ai
