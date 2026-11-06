# Muse Spark kernels on GB300 (sm_103)

The kernel half in this package was first written and measured on another architecture. This line serves GB300
(sm_103) only: `spec.REQUIREMENTS.isas` is `{"sm_103"}`. These notes record the sm_103 limits the kernels must
respect, what was changed for sm_103, what was checked on a GB300, and what is still open.

## sm_103 limits, and how the kernels meet them

| sm_103 (GB300) | These kernels |
|---|---|
| 227 KiB dynamic shared memory per block (232448 B, read on a GB300) | `draft_gemv` token tiles 32 and 64 exceed it (see below); every other kernel fits |
| 512 tensor-memory columns per SM | every kernel allocates at most 512 columns |
| clusters of at most 16 CTAs (non-portable) | every cluster launch is at most 16 |
| 152 SMs (GB300) | grids are sized from the device's SM count at run time |

The kernels use sm_100-family instructions only; `tcgen05.ld.red` (in `prefill_attn`) and the three-input `max.f32`
compile for `sm_103a`.

## Changes for sm_103

- `spec.py`: `isas` is `{"sm_103"}`.
- `draft_gemv` / `draft_boundary`: the default SwiGLU tactic's shared-memory ring is 240 KiB at token tile 32 and
  298 KiB at 64 (the convolution tactic's is 298 KiB at 64), above sm_103's 227 KiB. `draft_gemv.fits()` compares the
  ring with the device's per-block opt-in, and `draft_boundary` keeps the stock projections for a row block whose tile
  does not fit (counted as `draft_gemv: token tile exceeds the device's shared memory`) instead of failing at the
  launch. At the default `MUSE_VR_DRAFT_GEMV_MAX_M=16` (tiles 8 and 16, at most 211 KiB) nothing changes. Checked on a
  GB300: `fits()` is true for tiles 8 and 16 and false for 32 and 64.
- `moe_x_kernels/arch.py`: the MMA helper module (`MUSE_VR_MOE_X_MMA_HELPERS`) defaults to
  `cutlass.utils.blackwell_helpers`, which serves the sm_100 family including sm_103, instead of failing when the
  variable is unset. A set variable still wins. Checked on a GB300 image: the module has
  `make_blockscaled_trivial_tiled_mma` and `compute_epilogue_tile_shape`.

## Checked on a GB300

NVIDIA GB300 (152 SMs, CUDA 13.4), image `lmsysorg/sglang@sha256:9e6f6097...` (SGLang `306ccf6`), one GPU:

- Every JIT CUDA module under `components/ops`, `bal`, `final_norm` and `draft_kv_append` builds for `sm_103a`.
- The `preload` paths, which launch on zeros with each kernel's shared-memory attribute set, run for
  `nvfp4_verify_attn`, `kv4_attn_wide`, `full_attn_wide`, `full_verify_attn` (1-64 requests), `swa_verify_attn`
  (windows 511 and 1023, 1-64 requests), `swa_t`, `prefill_attn`, `prefill_qkn`, `nvfp4_kv_store`,
  `norm_fp4_quant_early`, `moe_ldw`, `moe_ldwb` and `bal`.
- `swa_t`'s occupancy query gives 76, 46, 36, 28, 23, 15, 15, 15, 13 resident clusters for kC = 2..10, so its kC
  choice adapts to the GB300's SM count.

## Not checked yet

- No multi-GPU run: the NVLink multicast / all-reduce kernels (`mc_push*`, `uc_push_dense`, `a2a_pull`, `c3_hops`,
  `c3_hb0`, `lb_hop`, the vendored all-reduce + norm) built but did not run across ranks.
- No served-model run on a GB300, so no accuracy or speed numbers.
- Not exercised: `unorm` (built only with the template arguments the boundary picks at run time), `proj_publish`,
  `wfc_k1s`, `moe_ld_ops`, and the CuTe DSL kernels (`opp_tgv_gemm`, `moe_x_kernels`, `fin2_cutedsl`). They pick
  their architecture from the device or use the sm_100 shared-memory capacity, which is the same on sm_103.
- Per-device tuning tables have no GB300 entry (device key `af3c79acd7d1e85f`). `moe_tactic` has no
  `data/moe_tactic/af3c79acd7d1e85f.json`, so its table is empty on a GB300 and FlashInfer's own tactic choice
  stands. `car_table` keeps the stock dispatch outside `SWEPT_DEVICES`. Both need the sweeps their docstrings name,
  run on a GB300. `vendor/sglang_pr47174` already carries a GB300 TP4 launch override.
