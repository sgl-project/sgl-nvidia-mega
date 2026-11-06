# Muse Spark kernels on GB300 (sm_103)

This line serves GB300 (sm_103) only: `spec.REQUIREMENTS.isas` is `{"sm_103"}`. It carries the components of the
Muse Spark kernel half that a GB300 TP4 DFLASH server with NVFP4 KV engages (listed below); components written for
other configurations are not included.

## sm_103 limits, and how the kernels meet them

| sm_103 (GB300) | These kernels |
|---|---|
| 227 KiB dynamic shared memory per block (232448 B) | `draft_gemv` token tiles 32 and 64 exceed it and are guarded; every other kernel fits |
| 512 tensor-memory columns per SM | every kernel allocates at most 512 columns |
| clusters of at most 16 CTAs (non-portable) | every cluster launch is at most 16 |
| 152 SMs | grids are sized from the device's SM count at run time |

`draft_gemv` / `draft_boundary`: the default SwiGLU tactic's shared-memory ring is 240 KiB at token tile 32 and 298 KiB
at 64. `draft_gemv.fits()` compares the ring with the device's per-block opt-in, and `draft_boundary` keeps the stock
projections for a row block whose tile does not fit (counted as `draft_gemv: token tile exceeds the device's shared
memory`). At the default `MUSE_VR_DRAFT_GEMV_MAX_M=16` (tiles 8 and 16) nothing changes.

## Components

Engaged on a GB300 TP4 server (`--tp-size 4 --ep-size 4 --quantization modelopt_fp4 --kv-cache-dtype nvfp4
--attention-backend trtllm_mha --speculative-algorithm DFLASH --speculative-draft-attention-backend fa4
--speculative-num-draft-tokens 8 --enable-layernorm-sp --moe-a2a-backend flashinfer --enable-hierarchical-cache`):
`nvfp4_kv` (with `nvfp4_kv_verify`, `nvfp4_prefill_dq`, `kv4_dec`, `kv_draftwin`), `verify_attn` (`swa_verify_attn`,
`full_verify_attn`, `full_attn_wide`, `attn_qk_prologue`), `prefill_qkn`, `xattn_b`, `ar_attn_boundary`,
`ctx_proj_overlap`, `ctx_proj_prefill`, `draft_boundary`, `draft_candidates`, `draft_kv_append`, `final_norm`,
`input_norm`, `spec_sample_split`, `verify_topk_sample`, `gemm_pick`, `gemm_lt`, `sp_gate`, `sp_combine_inplace`,
`sp_fp4_dispatch`, `pfin`.

`pull_bounded` stays off on SGLang `306ccf6` (its source pin does not match `pull_gather_bounded.cuh`).

## Checked on a GB300

NVIDIA GB300 (152 SMs, CUDA 13.4), image `lmsysorg/sglang@sha256:9e6f6097...` (SGLang `306ccf6`), TP4, Muse Spark
Open 816B-A42B NVFP4 with the DFlash2 draft, real draft acceptance:

| GSM8K, 5-shot, 1319 questions, greedy (`sglang.test.few_shot_gsm8k`) | Accuracy | Invalid | Accept length |
|---|---|---|---|
| this package, `MUSE_VR_*` components on, NVFP4 KV | 0.953 | 0.000 | ~5.1 |
| stock SGLang, no components, FP8 KV | 0.951 | 0.000 | ~3.8 |

The stock run uses FP8 KV because stock SGLang rejects DFLASH over native NVFP4 KV (`nvfp4_kv`'s resolution step is
what accepts it). `bench_serving` (random, ISL 8192, OSL 512, concurrency 112, 448 prompts) with the components on: 448
of 448 completed, accept length 6.47, output 2643 tok/s, median TPOT 14.5 ms (not tuned).

Each decode step runs the DFLASH draft CUDA graph (about 1 ms) and then the target-verify CUDA graph (about 8 ms), with
the verify sampling and the draft KV append eager between them; prefill runs SGLang's breakable prefill graph.

## Open

- `--enable-nvidia-megakernel` cannot be used with SGLang `306ccf6`: `muse_spark.spec` declares
  `NumericsKind.TOLERANCE` with `tolerances={}`, which SGLang's `Numerics` rejects at import. The components register
  through the plugin entry point either way.
- SGLang's all-reduce logs `multicast = False` at TP4 on a GB300: its sm_100-family table has no multicast block count
  for 4 ranks. Multicast itself works on these nodes.
- `gemm_pick` / `gemm_lt` rules were measured on another architecture and are not tuned for GB300.
