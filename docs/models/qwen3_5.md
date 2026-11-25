# Qwen3.5-397B-A17B (NVFP4)

| GPU | Tensor parallel | Coverage |
| --- | --- | --- |
| GB300 (SM103); GB200 (SM100) | 8 (two trays, one NVLink domain) | MTP target verify steps of 7 rows (batch 1) |

The draft spec provides the EAGLE entries. Qwen3.5 also offers the explicit GDN prefill slot below.

## Requirements

- An SGLang with `sglang.srt.kernel_plugins.nvidia.eagle_binding`
- 8 x GB300 or 8 x GB200, two trays of one NVLink domain

## Launch

On each of the two trays (`--node-rank 0` and `1`):

```bash
python -m sglang.launch_server \
  --model-path nvidia/Qwen3.5-397B-A17B-NVFP4-V2 \
  --tp 8 --nnodes 2 --node-rank 0 --dist-init-addr <tray-0 host>:<port> \
  --kv-cache-dtype fp8_e4m3 \
  --speculative-algorithm NEXTN --speculative-num-steps 6 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 7 \
  --attention-backend trtllm_mha \
  --linear-attn-decode-backend triton \
  --enable-linear-replayssm-spec --mamba-ssm-dtype bfloat16 \
  --moe-runner-backend flashinfer_trtllm \
  --flashinfer-allreduce-fusion-backend cutedsl \
  --enable-nvidia-megakernel
```

## What a covered step runs

Qwen3.5 (`qwen3_5/runtime`) runs the same way: per layer the stage boundary (`boundary/`), the GatedDeltaNet or the attention (`gdn/`, `attention/`) and the MoE (`moe/`), each with a stock implementation that reproduces SGLang's verify step bit for bit. It reads the running forward's metadata and pools through SGLang's EAGLE binding (`sglang.srt.kernel_plugins.nvidia.eagle_binding`). With the megakernel enabled, its shape-bounded bitwise paths run by default. `SGLANG_NVIDIA_MEGAKERNEL_SLOTS` overrides that policy for testing and bisecting; an empty value runs the stock paths.

| Slot | Rows | Kernel |
| --- | --- | --- |
| `route_topk`, `mkm_m32` … `mkm_m512` | 1–896 | Materialized stock routing and the shape-selected fused wide routed-expert engines |
| `hop`, `hop_fp8` | 1–15 | The layer boundary's fused all-reduce + residual add + RMSNorm, also writing the next static-FP8 input |
| `multi_request` | 2–128 requests | The target graphs keep every offered request count instead of padding to a coarser width |
| `draft_loop`, `draft_extend`, `draft_multi`, `draft_embed` | 1–128 requests | EAGLE draft entries, exact sharded top-1 and embedding exchange; the pre-FC norms write directly into the FC input |

The remaining kernel families are opt-in through that one slot policy. These change the arithmetic within the tolerances `spec.py` declares:

| Slot | Rows | Kernel |
| --- | --- | --- |
| `route`, `route_gemm`, `mkm`, `mkm_wide` | 1–32 | Fused router fronts and the small routed-expert engines |
| `gdn_verify`, `gdn_pdl`, `gdn_conv` | 7–896 | Explicit numeric verify chain and dependent variants; FP8 norm uses the v2 relative-L2 bound of 2e-2 |
| `gdn_core` | 7–224 | ReplaySSM verify, gated norm and FP8 output in one launch |
| `hop_mid` (with `hop`) | 16–28 | Extends the one-shot boundary mailbox through four requests instead of returning those widths to the stock boundary |
| `gdn_push` (with `gdn_verify`, `hop`) | 7 | The GatedDeltaNet `out_proj` as the hop's GEMM, whose epilogue pushes the partial into the hop's mailbox |
| `gdn_front`, `gdn_trt_front` | 7–28 | Fused GatedDeltaNet projections and conv before the dependent verify |
| `route_shared` (with `route`) | 1–8 | The shared expert's static-FP8 gate/up in the router front's launch, so the shared expert on the alternate stream waits for that launch only |
| `route_lin`, `skv2`, `mka`, `mka_push` | shape-bounded | Router/attention overlap, split-KV attention and fused attention boundaries |
| `lowm_in`, `lowm_out`, `lowm_se`, `lowm_push` | shape-bounded | Validated low-row projection and shared-expert launches |
| `draft_moe`, `draft_q1`, `draft_xchg`, `draft_hop`, `draft_fin`, `draft_front`, `draft_skv2` | shape-bounded | Fused MTP draft kernels and boundaries |

The tuned Qwen3.5 TP8/TEP8 serving policy explicitly includes `hop_mid` whenever it
includes `hop`.  It remains opt-in rather than part of the bitwise default policy;
an explicit `SGLANG_NVIDIA_MEGAKERNEL_SLOTS` value replaces, rather than extends,
that default list.

The TP8 plan (`topology.plan`, aggregated and PD decode) also runs `route_lin`
whenever the policy holds the slots it runs through (`route`, `mkm`, `hop`,
`gdn_push`, `mka`, `mka_push`, and `lowm_se` with `route_shared`): its router logits
stay within 1.05x stock's RMS error against FP64 on the same quantized operands.
The same TP8 plan adds `mkm_wide_bal` for `mkm_m256` / `mkm_m512`: balanced
CTA assignment at 129–512 verify rows, with bitwise-identical outputs.
TEP8, DEP8, TP2 and TP4 run exactly the requested slots their topology supports.

TEP8's routed body takes its NVFP4 MoE tactic per token bucket from a fixed table
(`moe/tactics.py`) on FlashInfer 0.7.1, SM103 and the routed experts' MoERunner
signature; any other configuration or bucket asks the autotuner, and the log names
each width's source. `SGLANG_NVIDIA_MEGAKERNEL_MOE_TACTICS` switches the table; unset,
it follows `SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION`.

The router-only `route`, `route_topk` and `route_gemm` slots support TP2, TP4, TP8 and TEP8 with the TRTLLM runner and no all-to-all. PD decode TP2 uses the same local row counts: batch 6/10 gives 42/70 verify rows. These fronts leave the expert GEMMs with the stock runner, so TP2 needs no fused expert engine. `route_shared` keeps its TP8/TEP8 projection bounds; DEP8 keeps stock MoE dispatch and routing.

`mka` uses the fused projection kernels through 14 verify rows on TP8 and 8 on
TP4. At wider rows, and at every TP2 width, it keeps SGLang's projection GEMMs
and fuses Q/K normalization, RoPE and the FP8 KV write, then the output gate
and static FP8 quantization. TP2 has 16 query heads and one KV head per rank;
TP4 has 8/1 and TP8 has 4/1. The same front path serves PD decode TP2 and DEP8 (32/2 per rank).
`multi_request` coverage is required for more than one request. `skv2` keeps
its one-request, one-KV-head scope; `skv2_cp` additionally requires replicated
KV heads across ranks.

The sampling and acceptance after the verify stay SGLang's own code. `gdn_prefill_split_conv` is an explicit bitwise slot for ordinary text prefill on TP1/PP4 with the data-only prefill binding: it splits the native conv by Q/K/V channels above 8192 channels, retaining native recurrence, norm and state writes. Unsupported layouts use SGLang; an empty slot policy calls the original prefill forward directly. In an SGLang that has the switches, `SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH=1` runs the sampling and acceptance in CUDA graphs, and `SGLANG_ENABLE_GDN_PREFILL_GRAPH_EXTEND=1` captures the GatedDeltaNet extend inside the breakable prefill CUDA graphs.

## Manual tests

Qwen3.5 runs TP8, which spans two GB300 trays: its e2e tests read a server launched on them through `SGLANG_NVIDIA_MEGAKERNEL_E2E_URL` and the files its ranks log to (`SGLANG_NVIDIA_MEGAKERNEL_E2E_LOGS`, comma-separated).
