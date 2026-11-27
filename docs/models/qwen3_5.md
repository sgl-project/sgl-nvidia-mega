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
| `gdn_conv_core` (with `gdn_core`, `gdn_conv`) | 29–224 | `gdn_conv`'s conv inside `gdn_core`'s launch, bit for bit the two launches |
| `hop_mid` (with `hop`) | 16–28 | Extends the one-shot boundary mailbox through four requests instead of returning those widths to the stock boundary |
| `gdn_push` (with `gdn_verify`, `hop`) | 7 | The GatedDeltaNet `out_proj` as the hop's GEMM, whose epilogue pushes the partial into the hop's mailbox |
| `gdn_front`, `gdn_trt_front` | 7–28 | Fused GatedDeltaNet projections and conv before the dependent verify |
| `route_shared` (with `route`) | 1–8 | The shared expert's static-FP8 gate/up in the router front's launch, so the shared expert on the alternate stream waits for that launch only |
| `route_lin`, `skv2`, `mka`, `mka_push` | shape-bounded | Router/attention overlap, split-KV attention and fused attention boundaries |
| `lowm_in`, `lowm_out`, `lowm_se`, `lowm_push` | shape-bounded | Validated low-row projection and shared-expert launches |
| `lowm_ba` (with `gdn_verify`; TP8, TEP8) | 84–112 | GatedDeltaNet `in_proj_ba` on one low-M launch whose cluster splits K, in place of SGLang's linear (cuBLAS); numeric (`runtime.NUMERICS_CHANGE`), with no bound stated in `spec.py` |
| `draft_moe`, `draft_q1`, `draft_xchg`, `draft_hop`, `draft_fin`, `draft_front` | shape-bounded | Fused MTP draft kernels and boundaries |
| `draft_skv2` (bitwise) | 1 request | The draft extend's KV-length graph variants, in which `draft_skv2_cp` serves one long request; below its length the extend's rows run the stock decode |
| `bt_tail` (bitwise) | 16–703 | The layer and MoE inputs' fused all-reduce + residual add + RMSNorm past the hop's rows, on FlashInfer's balanced (BT) protocol with its last launch replaced by one that also writes the next static-FP8 input; FlashInfer's outputs bit for bit. With `bt_tail_early` / `bt_tail_early_fin` the MoE input's / layer input's last launch releases its dependents at its wait. It needs the FlashInfer its launches were checked against (flashinfer-python 0.7.4, `FLASHINFER_SOURCES`) and a BF16-output workspace; with another FlashInfer, or at a width where either point does not route the three-launch BT (with the TP8 / TEP8 routes, the layer input below 24 rows), it stays stock |
| `gemm_ar` (bitwise, with `bt_tail`; TP8, TEP8) | 33–128 (none at TP4: the kernel takes K up to 1024, TP4's `out_proj` / `o_proj` have 2048) | The mixer's `out_proj` / `o_proj` runs inside the MoE input's all-reduce launch on `bt_tail`'s point: the FP8 projection, a two-shot all-reduce, the residual add, the norm and the MoE's static-FP8 input in one launch, at the BT path's rounding points (cuBLAS + FlashInfer's BT outputs bit for bit). A projection the kernel refuses, or one whose all-reduce does not run on `bt_tail`, runs on cuBLAS |
| `fc2ar` (bitwise, with `bt_tail` and `moe_engine`; TEP8) | 35–128 | The MoE engine writes its FC2 rows into a Lamport buffer and releases the next boundary point early; that point's one-shot all-reduce polls the rows while the engine runs and folds the finalize, the residual add, the norm and the next static-FP8 input in one launch, at the BT path's rounding points (FlashInfer's BT outputs bit for bit). Capture-mode steps only; a width without a BT route stays on BT. The Lamport buffer is made at load, for the widest covered call at FlashInfer's widest FP4 `tile_n`, and every `tile_n`'s call shares it |
| `draft_select` (with `draft_extend`) | 1–128 requests | The draft extend's selected rows in one row-copy launch at TP4, TP8 and TEP8; bitwise `torch.index_select` on in-range indices |
| `gumbel_fast` (with `sharded_gumbel`) | sampled verify rows | `sharded_gumbel`'s noise drawn four per Philox call (another stream, the same law), as one CUDA launch where the rows are aligned quads |
| `verify_head` (with `sharded_gumbel`) | 70–128 | The vocabulary-sharded verify's LM head on the tensor cores in place of the GEMM, through SGLang's `verify_shard_head` hook; numeric (`runtime.NUMERICS_CHANGE`), with no bound stated in `spec.py` |
| `verify_head_keys` (with `verify_head`, `gumbel_fast`) | 70–128 | The head also leaves the T = 1 and greedy keys of its logits, which `gumbel_fast`'s launch takes for rows at temperature 1 or 0 |

The MTP draft's norms in `draft_hop` and `draft_fin` take the rounding of SGLang's
GemmaRMSNorm (`gemma_fused_add_rmsnorm`, multiplier 1 + the raw weight in fp32), as the
stock draft does: the post-attention norm on the draft hop (the add-norm consume), the
final norm after a deferred MoE finalize on the hop, and, past the hop at TEP8, both norms
after SGLang's MNNVL CuTe DSL workspace, which then only sums. Past that workspace's LL
and BT routes (`draft.layer.STOCK_SUM_MAX_ROWS`, 703 rows), which add the ranks' rows in
rank order, those steps run SGLang's all-reduce and GemmaRMSNorm.

Opt-in variants of the CP slots (`skv2_cp`, `draft_q1_cp`, `draft_skv2_cp`) at TP8 and TEP8; their kernel tests in `test/kernels` compare the outputs with the default build's, byte for byte:

| Slot | Rows | Kernel |
| --- | --- | --- |
| `cp_smv2`, `cp_ptxas_o1` (with a CP slot) | as its CP slot | Every CP-lite partial built with softmax v2 (on unmasked tiles the row max from `tcgen05.ld.red`, then each 32-column chunk reloaded from tensor memory and exponentiated two lanes per `fma.rn.f32x2` / `add.rn.f32x2`) / with ptxas `-O1` |
| `cp_merge_plain_wait` (with `skv2_cp`) | `skv2_cp`'s multi-request verify | The multi-request target verify's exact per-head CP-lite merge polls its sources with the plain merge's volatile-load wait instead of the per-thread spin |
| `cp_q_push` (with `skv2_cp` and `mka`) | `skv2_cp`'s multi-request verify | The front's q writer stores the FP8 q rows to the CP-lite group's members as the q gather's low-latency words, so the ungated exact call's gather only waits for them |
| `cp_qw_entry` (with `cp_q_push`) | `skv2_cp`'s multi-request verify | The pushing q writer releases its dependents at its entry, so the gather's wait half and the partial launch under it; each waits before reading its outputs |
| `cp_merge_cg` (with `cp_q_push`) | the CP slots' early merges | Every early per-head CP-lite merge reads its lengths, epochs, the step's q / K / V and the gate through L2 (`ld.global.cg`) |
| `cp_lam_q` (with `cp_q_push`) | `skv2_cp`'s multi-request verify | The pushed q words move as Lamport words (plain 32-bit data in a sentinel-filled region of their own, one writer a word; the wait half stores the sentinel back over the words it polled) instead of 8-byte low-latency words |
| `cp_q_gather` (with `cp_q_push` and `cp_lam_q`) | `skv2_cp`'s multi-request verify through 63 rows | The pushing q writer also polls the group's Lamport q words of each row it pushed and lands CP-lite's `q_all`, prefix lengths and epochs itself, so the ungated exact call launches no gather and its partial launches under the writer |
| `cp_o_push` (with `cp_q_push` and `cp_merge_plain_wait`) | `skv2_cp`'s multi-request verify, up to 64 requests | The fused CP-lite partial writes each finalized (row, head) partial and its LSE into the per-head merges' mailboxes as low-latency words and releases the merge once its stream is done; the merge only polls every source's words in one pass, with its dependency wait moved to its end |

The tuned Qwen3.5 TP8/TEP8 serving policy explicitly includes `hop_mid` whenever it
includes `hop`.  It remains opt-in rather than part of the bitwise default policy;
an explicit `SGLANG_NVIDIA_MEGAKERNEL_SLOTS` value replaces, rather than extends,
that default list.

The TP8 plan (`topology.plan`, aggregated and PD decode) also runs `route_lin`
whenever the policy holds the slots it runs through (`route`, `mkm`, `hop`,
`gdn_push`, `mka`, `mka_push`, and `lowm_se` with `route_shared`): its router logits
stay within 1.05x stock's RMS error against FP64 on the same quantized operands.
No router operand is rounded to BF16, and the logits' RMSNorm scale comes from the
FP32 pre-norm row with each hidden partial's largest BF16 rounding residual per
32-channel group added back (one 4-byte cell per rank, row and group in the router
mailbox). Its packing checks each rank's own weights (a BF16 router, an FP16 scale
for the folded `out_proj` core, BF16-exact gamma times that scale); when any rank
declines, every rank keeps the stock router and logs a warning naming each declining
rank's reason.
The same TP8 plan adds `mkm_wide_bal` for `mkm_m256` / `mkm_m512`: balanced
CTA assignment at 129–512 verify rows, with bitwise-identical outputs.
TEP8, DEP8, TP2 and TP4 run exactly the requested slots their topology supports.

`gdn_conv_core` runs `gdn_conv`'s causal conv update inside `gdn_core`'s launch
(`sm103/gdn/core.py` with `conv`), with the arithmetic of SGLang's Triton conv, so its
conv output, rolling conv states, windows and the core's outputs equal the two launches'
bit for bit. The last CTA of a key head to read the prior q / k state rolls it, through
int32 counters per request and key head of each GDN layer (made at build), so a layer's
launches must not overlap: there is one verify stream, and the spec does not accept
SGLang's batch overlap. The launch releases `out_proj` at its entry at every width.

With `SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION=1` and no slot list (the measured
stack), TP8 and TEP8 on GB300 (compute capability 10.3, 152 SMs) run the measured plan
(`runtime.MEASURED_PLAN`): the shipped slots plus the 31 opt-in slots of
`runtime.MEASURED_OPT_IN` (not `route_gemm` or `draft_xchg`), the BT tail with its
two early releases (`runtime.MEASURED_BT_TAIL`; stock without flashinfer-python 0.7.4) and
the CP-lite variants of `runtime.MEASURED_CP_LITE`, the draft slots of
`runtime.MEASURED_DRAFT` and the verify keys provider's options `runtime.MEASURED_HOOKS`
(`gumbel_fast`, `verify_head`, `verify_head_keys`), and at TEP8 also `gemm_ar` and
`fc2ar` on the BT tail's points (`runtime.MEASURED_BT_AR`). The topology filters and
the TP8 plan then apply as for a slot list. The plan also runs options, which are no
slots (`runtime.MEASURED_OPTIONS`; the draft's are in the table below): `gdn_value_split`,
where `gdn_core` runs each
(request, value head) of the TP8 shard's 8 value heads on a cluster of two CTAs at
7–112 rows (the norm's sums are combined in the one-CTA order, so the same bits), and
`gemm_tactics`, where the FP8 projection GEMM of `in_proj_qkvz` and `qkv_proj` takes a
cuBLASLt algorithm per verify width from a table keyed by topology, GPU and cuBLASLt
version (`gemm_tactics.py`). Each table algorithm must equal FlashInfer's heuristic
algorithm bit for bit on a check GEMM of the layer's weights, else the width runs that
heuristic algorithm; another cuBLASLt version, or a FlashInfer whose cuBLAS FP8 runner
lacks a name the module reads, runs FlashInfer's BMM; each with one WARNING. Unlike the
rest of the build, a width resolves at its first call outside a CUDA graph capture, and
each rank resolves on its own (a rank whose check fails runs the heuristic algorithm,
whose bits the others' algorithm equals). Per row, of
those it turns on, `runtime.MEASURED_BITWISE` (`mkm`, `mkm_wide`, `mkm_wide_bal`,
`gdn_pdl`, `gdn_conv`, `gdn_conv_core`, `mka_push`, `bt_tail`,
`bt_tail_early`, `bt_tail_early_fin`, `gemm_ar`, `fc2ar`, `cp_smv2`, `cp_ptxas_o1`,
`cp_merge_plain_wait`, `cp_q_push`, `cp_qw_entry`, `cp_merge_cg`, `cp_lam_q`,
`cp_q_gather`, `cp_o_push`, `draft_skv2`, `draft_select`, `gdn_value_split`, `gemm_tactics`,
`gdn_prefill_front`, `gdn_prefill_track_snapshot`, `moe_prefill_input_nvfp4`,
`moe_prefill_add_norm`, `moe_prefill_noclone`, `prefill_gate_fp8`, `prefill_act_fp8`,
`gdn_prefill_norm_fp8`, `prefill_norm_fp8` and the bitwise draft rows below) keep the
step's bytes given the slots they need, and
`runtime.NUMERICS_CHANGE` change the arithmetic; each has a row in
`spec.NUMERIC_TOLERANCES` (empty where no bound is stated) or a tolerance claim in the
draft spec, except `hop_mid` (`runtime.NUMERICS_UNBOUNDED`).
Numerics exceptions (`spec.NUMERIC_EXCEPTIONS`): at TP8 / TEP8 (CP 4 x 4 query heads a
rank, CP-lite's LL path), `skv2_cp`'s target verify keeps two E4M3 terms
of P from one request (BF16 + residual partials)
through 64 (FP16 partials). At 65–128 requests, on CP-lite's two-slot
plans, it keeps one E4M3 term of P and BF16 partials. At TP4 (CP 2 x 8 query heads a
rank, whose rows do not fit the LL words) every width runs CP-lite's fold path as
before, with BF16 partials. In both, the exception declares a single request's error
unbounded (`worst_request`).
Every other topology or GPU runs the shipped slots, and an explicit slot list keeps the
policy above. The startup log names the plan's key and its slots. Where the measured plan
runs a CP slot, every CP-lite partial on the one-slot balanced plans also builds with a
lean fixed phase and an LSE-first combine, on the one-request-per-lane plan (up to 32
requests) with a per-CTA plan cache, and with `cp_o_push` reads its push epochs right after
its dependency wait (`q35_cp_fused.set_cp_phase`): data movement and issue order only,
compared byte for byte with the default build in `test/kernels`; elsewhere it builds as
before.

The rows that join the shipped slots per key (`runtime.MEASURED_PLAN`);
`test/unit/test_qwen3_5_plan_snapshots.py` pins each key's resolved slots, and that
the megakernel alone or an explicit slot list resolves every key as upstream's build:

| Key (GB300) | Rows that join the shipped slots |
| --- | --- |
| `tp8` | `MEASURED_OPT_IN`, `MEASURED_BT_TAIL`, `MEASURED_CP_LITE`, `MEASURED_DRAFT`, `MEASURED_HOOKS` and the options `MEASURED_OPTIONS`; the TP8 plan adds `route_lin` and `mkm_wide_bal` |
| `tep8` | The same and `MEASURED_TEP8`, `MEASURED_BT_AR`, `MEASURED_DRAFT_TEP8` and the options `MEASURED_TEP8_OPTIONS`, less `EP_STOCK_SLOTS` (expert parallelism keeps SGLang's MoE there) |
| `dep4` | `MEASURED_DEP4`; DEP4 declares none of the shipped slots |
| `p_tp1_pp2` | `MEASURED_PREFILL` (`gdn_prefill_front`, `gdn_prefill_track_snapshot`, `moe_prefill_input_nvfp4`, `moe_prefill_add_norm`, `moe_prefill_noclone`, `prefill_gate_fp8`, `prefill_act_fp8`, `gdn_prefill_norm_fp8`, `prefill_norm_fp8`, `attn_prefill_qk_rows`), the prefill slots declared at the key |
| Any other key or GPU | None: the shipped slots (the prefill body seam is still offered: see below) |

At TEP8 on GB300 the measured plan also runs the routed body's own rows
(`runtime.MEASURED_TEP8`; TEP8 only, each with the slots it needs). As with the other
opt-in slots, an explicit slot list that names them runs them at TEP8 on any GPU the
spec admits.

| Slot | Rows | Kernel |
| --- | --- | --- |
| `moe_engine` | every TEP8 width | The routed body on a persistent NVFP4 MoE engine (FC1, SwiGLU, NVFP4, FC2 in one launch) in place of FlashInfer's two GEMMs; the same FC2 rows bit for bit |
| `moe_ring` (with `moe_engine`, `route_lean`) | 64–128 | The engine streams the first FC1 stages of the experts the layer used at its previous call before its dependency wait; bitwise |
| `route_big_gemm` (numeric) | 33–192 (from 17 with `route_big_mid`) | Route-big's router logits from a split-K tcgen05 GEMM in a fixed, batch-independent sum order |
| `route_lean` | 17–448 | Route-big's lean variant: the routing cluster's counts by bulk copies onto mbarriers, the engine released at route-big's start; bitwise |
| `moe_engine_route` (with `moe_engine`, `route_big_gemm`) | 33–63 | The router GEMM keeps each token's top-10 per expert tile and the engine writes the routing tables itself, in place of route-big; bitwise |
| `route_big_mid` (numeric) | 17–32 | Route-big writes the routed body's tables at tier M's widths in place of route_topk's per-token launch, bit for bit on the same logits; with `route_big_gemm` the router GEMM runs from 17 rows |
| `lowm_se_tep8` (numeric, with `lowm_se`) | 33–128 | `lowm_se`'s two low-M launches for the shared expert at the routed body's widths |
| `lowm_se_wide` (numeric, with `lowm_se_tep8`) | 129–160 | The same launches over a 128-row and a 32-row chunk; each row the 128-row launch's bit for bit |
| `lowm_se_coresident` (with `lowm_se_tep8`) | 84–128 | The shared expert's gate_up releases its down launch at its end instead of after its weight loads; bitwise |

The MTP draft's own rows of the measured plan (slots of `runtime.MEASURED_DRAFT` at TP8
and TEP8 and `runtime.MEASURED_DRAFT_TEP8` at TEP8; options, which are no slots, of
`runtime.MEASURED_OPTIONS` and, TEP8 only, `runtime.MEASURED_TEP8_OPTIONS`):

| Row | Kind | Rows | What it runs |
| --- | --- | --- | --- |
| `embed_local` | option, bitwise | every verify width | The verify's embedding rows from a replicated BF16 table (the vocabulary-parallel shards all-gathered at build) in one gather launch, in place of the vocabulary-parallel lookup and its all-reduce; steps aside under `SGLANG_ENABLE_EMBED_REPLICATION` |
| `draft_embed_local` (with `draft_loop`) | slot, bitwise | 1–128 requests | The draft's embedding rows from the same table (one table a rank when the draft shares the target's embedding), in place of `draft_embed`'s exchange; steps aside as `embed_local` |
| `draft_fc_input` (with `draft_embed_local`) | slot, numeric | 1–128 requests | The fc input's table gather and both pre-fc Gemma RMSNorms in one launch |
| `draft_top1_cfg` | option, bitwise | 2–128 requests | `draft_multi`'s sharded top-1 sends its row keys with plain stores to every peer's buffer and shortens its last CTA's chain; the same tokens |
| `draft_head_lowm` (with `draft_multi`) | slot, numeric | 1–8 requests | The multi-request steps' LM head on the low-M GEMM at the widths where it beat cuBLAS in a build-time race (rank 0's picks on every rank) |
| `draft_head_keys` (with `draft_head_lowm`) | option, numeric | 9–16 requests | The multi-request steps' LM head on the tensor-core head with the row argmax in its epilogue (no logits written) and one key exchange over unicast stores, in place of the cuBLAS head and `draft_multi`'s sharded top-1; the same tokens on the same bf16 logits |
| `draft_router_lowm` | slot, numeric | 1–16 rows | The draft MoE block's router GEMM on the low-M GEMM at the widths its build-time race took |
| `draft_moe_engine` | TEP8 option, numeric | 1–32 rows | The draft's routed BF16 experts in one persistent launch (routing in every CTA's prologue, FlashInfer's Renormalize arithmetic) in place of FlashInfer's `trtllm_bf16_moe`, through the draft MoE block's deferred forward |
| `draft_moe_wide` (with `draft_moe_engine`) | TEP8 option, numeric | 33–128 rows | The engine's wide build: four token words a local expert, segments of 64 rows |
| `draft_shared_lowm` | TEP8 option, numeric | 2–16 rows | The draft's shared expert on two low-M launches (gate_up with the SwiGLU in its epilogue, then down) |
| `draft_fc_tp` (with `draft_fc_input`) | TEP8 slot, numeric | 1–128 rows | The steps' and extends' fc as the column-parallel fc plus one all-reduce and the layer's input norm (the draft hop's add-norm, else SGLang's MNNVL CuTe DSL sum) |
| `draft_fc_push` (with `draft_fc_tp`) | TEP8 option, numeric | the hop's widths | The fc window GEMM's epilogue pushes its columns into the draft hop, whose consume reads each column from its owner's slot |
| `draft_oproj_push` (with `draft_hop`) | option, numeric | 2–16 requests | The multi-request steps' o_proj on the low-M GEMM pushing its partial into the draft hop, so the publish launch goes |
| `draft_early_hops` | option, bitwise | 1–16 rows | The fused fc input, the draft hop's layer-input consume and its FFN-input consume release the next GEMM (low-M fc window, TGV qkv, low-M router) at entry; launch order only |
| `draft_q_push` (with `draft_front`, `draft_multi` and `cp_q_push`) | slot, bitwise | the multi-request calls CP-lite takes | The draft front's q writer pushes the steps' and extends' FP8 q rows to the CP-lite group as the q gather's words, so CP-lite's gather only waits; the same `q_all` bytes |
| `draft_q_gather` (with `draft_q_push` and `cp_lam_q`) | slot, bitwise | the multi-request calls CP-lite takes | The pushing draft front also polls the group's Lamport q words of each row it pushed and lands CP-lite's `q_all`, prefix lengths and epochs itself, so the call launches no gather and its partial launches under the front |
| `draft_cp` (with a CP slot) | slot, numeric | 2–64 requests | SGLang's draft worker offers its TRT-LLM MHA decode and draft-extend calls on the FP8 KV cache through its `draft_attention` hook; they run as CP-lite's precise call (two-term E4M3 P, BF16 + residual partials), and the stock attention past 64 requests. Where it runs, the megakernel draft's own CP-lite calls (`draft_q1_cp`, `draft_skv2_cp`, `draft_multi`) take the same precise call and cutover; without it they run as before |
| `draft_gate_fold` (with `draft_multi` and `draft_cp`) | slot, bitwise | 2–64 requests | The precise multi-request call's per-head merge applies the attention output gate in `fused_sigmoid_mul`'s arithmetic, so no gate launch follows; the same bytes |
| `draft_o_push` (with `draft_multi` and `draft_cp`) | slot, bitwise | 2–64 requests | The precise multi-request call's fused partial pushes its (O, LSE) and residual words into the members' per-head merge mailboxes, and the merge, still launched after its wait, only polls them; the same words |
| `draft_merge_early` (with `draft_o_push`) | option, bitwise | 2–64 requests | That polling merge runs its causal block and polls under the partial, its dependency wait moved to its end (before the gate read and every store), and the partial releases it after its stream; launch order only |
| `draft_hilo_trig` (with `draft_merge_early`) | option, bitwise | 2–64 requests, draft steps (1 row a request) | On the draft steps that early merge also releases its dependents at entry, so the draft o_proj (with `draft_oproj_push`, the draft hop's pushing GEMM) loads its weights under the merge; each reads the merge's output only after its own dependency wait; the draft extend's merge keeps the early form without it; launch order only |

With the plan's option `gdn_commit`, the TP8 or TEP8 build offers SGLang's
`SupportsRecurrentCommit`: after each chain verify it runs SGLang's ReplaySSM commit
(`gdn/commit.py`, the cursor advance and the conv rollback as SGLang's), with the
circular ring's compact commit as a CUDA kernel (`sm103/gdn/commit.py`). Its
checkpoints can differ in rounding from SGLang's Triton commit's (numeric, in
`runtime.NUMERICS_CHANGE`). It takes the pools at the first covered step: the compact
BF16 ring of length 16 with int32 cursors; on other pools it logs why and SGLang
commits as without it.

DEP4 (TP4, attention DP4, MoE EP4) is admitted only under the measured stack, which
lifts the `moe_a2a_backend` and `flashinfer_allreduce_fusion_backend` pins and keeps the
`flashinfer_trtllm` runner pin; the megakernel alone refuses it as before. On GB300 its
plan (`runtime.MEASURED_DEP4`) runs `gdn_verify` and `gdn_core` with three numeric rows
of its own, and the rest of the step (MoE, its exchange, the boundary and the draft)
stays SGLang's:

| Slot | Rows | Kernel |
| --- | --- | --- |
| `gdn_core_tc` (with `gdn_core`) | 7–896 | `gdn_core`'s launch on tcgen05 (`sm103/gdn/core_tc.py`): persistent CTAs over (request, value head) pairs |
| `gdn_commit_fold` (with `gdn_core_tc`) | 7–896 | That launch first folds a flush row's committed ring entries into its checkpoint; the commit after the verify (`SupportsRecurrentCommit`) then only advances the ring cursors and rolls back the conv state. It runs only with `--disable-radix-cache`, without unified memory or the parity check, and where every offered verify width runs `gdn_core`; it is decided before the build compiles. A retracted request is not backed up to the host (`SupportsStateSlotReset.backs_up_state`) |
| `gathered_cdf` | gathered verify rows | Temperature-only rows of the gathered verify logits draw their target tokens by inverse CDF (`sm103/gathered_cdf.py`) through SGLang's gathered-logits sampler policy: the same law as the stock sampler on another random stream. Its kernels and seed counter are built at load |

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

The sampling and acceptance after the verify stay SGLang's own code. `gdn_prefill_split_conv` is an explicit bitwise slot for ordinary text prefill on TP1/PP4 with the data-only prefill binding: it splits the native conv by Q/K/V channels above 8192 channels, retaining native recurrence, norm and state writes. Unsupported layouts use SGLang; an empty slot policy calls the original prefill forward directly. The measured stack also admits the prefill role at TP1/PP2 (key `p_tp1_pp2`, `spec.pd_reasons`). On GB300 its measured plan (`runtime.MEASURED_PREFILL`) runs `gdn_prefill_front`, declared at TP1/PP2 only: on SGLang's FlashInfer prefill route (`--linear-attn-prefill-backend flashinfer`), each GDN layer's causal conv, Q/K/V split, Q/K L2 norm and gates come from one CUDA launch and two small Triton launches (`sm103/gdn/prefill_front.py`) with the bytes of SGLang's `causal_conv1d_fn`, `gdn_prefill_qkv_prepare_fwd` and `fused_gdn_gating`; the layer's state read, the FlashInfer chunked prefill, the state gather and write-back, the prefix-track copies (with SGLang's one-launch state copies where its predicates take them), the gated norm and the output projection keep SGLang's order (`gdn/prefill_fi.py`). Under the measured stack the spec offers SGLang's prefill body seam (`prefill_graph`, `SupportsPrefillBody`), so the slot runs in the layer module's forward that SGLang's breakable prefill CUDA graphs capture per token bucket (`--cuda-graph-backend-prefill breakable`), or eagerly without them: each GDN layer's core is a graph break, as SGLang's own linear-attention core is, and runs on the live batch's real rows; the rest of the body is captured as SGLang's. The build checks SGLang's seams and every GDN layer's shape and weights, compiles the CUDA kernel and runs the Triton launches once; a layer it cannot take keeps SGLang's forward for the whole prefill. A step on another prefill kernel, a multi-item or strided state layout or recomputed checkpoints, and a layer whose tensors the front does not take, run SGLang's own core there before any state is written, each reason logged once. The plan's `gdn_prefill_track_snapshot` (bitwise, in the front's extend) has the FlashInfer chunked prefill store only the checkpoints the prefix-track copies read: FlashInfer's `chunk_gated_delta_rule` takes a destination row per checkpoint entry (`checkpoint_state_indices`, -1 skips the store), so the tracked requests' entries land in a buffer of one row each, which SGLang's track copy reads, and no other entry is written (`gdn/fi_track_snapshot.py`). The kernel and the states it stores are SGLang's call's; FlashInfer runs destination rows on its native (non context-parallel) kernels only, and the slot runs on the SM100 family only, so a step whose batch FlashInfer's own heuristic sends to its context-parallel kernel keeps SGLang's all-row checkpoints. The FlashInfer files the slot rests on are pinned by sha256 (flashinfer-python 0.7.4); another FlashInfer keeps SGLang's checkpoints, with one startup line. The slot-mode kernel is its own FlashInfer build: it compiles on the first step that runs it, as the measured stack runs it, and the slot adds no prewarm. With the body offered, SGLang waits for a layer-wise host load-back to finish before a prefill graph replays, where its stock body waits per layer. The spec decides the offer at import, before the role, the key and the GPU are known, so under the measured stack every process offers the body (the decode builds and every other key plan none): any of them that replays prefill graphs (a TP1/PP4 prefill, an aggregated server with breakable prefill graphs) waits that way too. What the wait costs on prefills with host-loaded prefix hits is not measured. In that body, slot `moe_prefill_input_nvfp4` (declared at TP1/PP2 only) serves the MoE layers on SGLang's CuTe DSL NVFP4 MoE (`--moe-runner-backend flashinfer_cutedsl`, no all-to-all, a static activation scale): the post-attention Gemma fused add-RMSNorm (FlashInfer's `FusedAddRMSNormKernel`, which it extends, `sm103/moe/prefill_input_nvfp4.py`) also writes the routed experts' NVFP4 input with `fp4_quantize`'s bytes, and the routed experts take it in place of their own quantization (`moe/prefill.py`); the norm output and the residual keep the stock bytes. The build checks every MoE layer's FFN input path and block (a layer it cannot take keeps SGLang's, with one startup line) and compiles the kernel; a batch the mirrored path does not cover runs SGLang's FFN input. Slot `moe_prefill_add_norm` (declared at TP1/PP2 only, in the same body) leaves each covered MoE block's shared-expert gated add (SGLang's Triton `fused_gate_sigmoid_mul_add`) to the next layer's input norm on the stage: the block launches the same Triton kernel in its `GATE_ONLY` mode with the stock launch shape, and the next layer's input Gemma fused add-RMSNorm (`sm103/moe/prefill_output_add_norm.py`, which extends FlashInfer's `FusedAddRMSNormKernel`) does the gated add in its load step; the norm output and the residual keep the stock bytes. A block defers only where its FFN exit passes the output through unchanged and the next layer takes it on the mirrored attention input; anything else (the stage's last layer, a capture, another batch) completes the gated add with the stock kernel first. Slot `moe_prefill_noclone` (declared at TP1/PP2 only, in the same body) gives a covered block's shared expert the block's input itself in the dual-stream order (under CUDA graph capture), where SGLang gives it a clone: the clone guards a routed path that writes into its input, and the CuTe DSL runner's standard path, which the build requires, only reads it. In that body, the measured plan's static FP8 hand-offs (`runtime/prefill_fp8.py`) have the op before a ModelOpt static per-tensor FP8 linear write that linear's E4M3 input, which the linear takes through its prequantized-input path instead of quantizing it again; each writes the bytes of SGLang's `static_quant_fp8` on the op's stock output, so the linear's output is the stock one: `prefill_gate_fp8`, a full-attention layer's output gate writing `o_proj`'s input (`sm103/attention/gate_quant.py`: the verify step's gate kernel on SGLang's launch of the gate); `prefill_act_fp8`, an MoE layer's shared expert (or a dense MLP) writing `down_proj`'s input from its SiLU-and-mul (`sm103/moe/prefill_act_fp8.py`, built through SGLang's JIT loader on SGLang's activation header; SGLang's MoE block calls the shared expert itself, so the build gives such an MLP the package's `Qwen2MoeMLP.forward`, which takes the fused path only while the body runs); `gdn_prefill_norm_fp8`, a GDN layer's gated RMSNorm writing `out_proj`'s input in the front's step (`gdn/norm_fp8.py`: the verify step's gated-norm kernel on SGLang's prefill launch, with an E4M3 store that keeps the BF16 store's register layout); `prefill_norm_fp8`, a layer's input Gemma add-RMSNorm also writing the static FP8 input of `in_proj_qkvz` or `qkv_proj` (`sm103/gemma_norm_fp8.py`: FlashInfer's CuTe DSL fused add-RMSNorm, the kernel SGLang's `GemmaRMSNorm` runs, with an E4M3 epilogue; the build gives the norm and its linear the package's forwards: the norm's runs the dual kernel where `GemmaRMSNorm._forward_impl` runs the fused add-RMSNorm alone, as on the call SGLang's attention read makes, `norm(x, residual, None)`, reading the linear's input scale at the call, and the linear takes the E4M3 copy only when its input is the norm's output itself, unmodified since; where `moe_prefill_add_norm` does the previous block's gated add in that norm, its kernel writes the copy, `sm103/moe/prefill_output_add_norm.py`'s E4M3 epilogue). A layer whose linear is not a static per-tensor FP8 linear keeps SGLang's forward, logged once at the build. The plan also runs `attn_prefill_qk_rows`, declared at TP1/PP2 only: each full-attention layer's Q / K GemmaRMSNorm + partial RoPE runs over tiles of 8 (token, head) rows of one kind, Q and K in one launch (`sm103/attention/qk_norm_rope_rows.py`), where SGLang's `fused_qk_gemma_rmsnorm_rope_gate` runs one program per (token, head), and `fused_sigmoid_mul` reads the output gate in place as a strided view of the QKV projection instead of SGLang's copy (where `prefill_gate_fp8` took the layer, its FP8 gate reads that view and writes `o_proj`'s input); the rest of the layer is SGLang's, in its order (`attention/prefill_rows.py`). It is numeric (`runtime.NUMERICS_CHANGE`): only each head's FP32 sum of squares runs in another order, so an element can round to the neighbouring BF16 value; given Q and K, the gated output's bytes are SGLang's. If the tiles do not take one full-attention layer's norms or head shape, or the model is not BF16, every full-attention layer keeps SGLang's for the whole prefill, with one startup line. Either slot runs the prefill body on its own; without the front, the GDN layers run SGLang's layer forward there. With the megakernel alone or an explicit slot list, TP1/PP4 stays the only prefill shape. In an SGLang that has the switches, `SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH=1` runs the sampling and acceptance in CUDA graphs, and `SGLANG_ENABLE_GDN_PREFILL_GRAPH_EXTEND=1` captures the GatedDeltaNet extend inside the breakable prefill CUDA graphs.

## Design

Where Qwen3.5 departs from or adds to the dev guide's [invariants](../../AGENTS.md#invariants) and its
[fallback rule](../../AGENTS.md#where-a-path-runs):

- **Built at load.** Exceptions, each made at a width's first eager forward (normally SGLang's eager warm-up run before it captures the width), after the KV cache is sized, and never under CUDA-graph capture: Qwen3.5's TEP8 routed body (`moe/body.py`), each width's runner, tactic, routing tables and body workspace, and with slot `moe_engine` the engine's buffers, call state and ring records (the engine's pieces refuse to be made under capture); the router-GEMM compiles of slots `route_big_gemm` and `moe_engine_route` (a width first seen under capture does not take these slots; `route_big_gemm`'s then runs SGLang's router GEMM); slot `moe_engine_route`'s per-CTA tables (refused under capture); and `gemm_tactics`' per-width algorithm check, which runs at a width's first call outside capture (its paragraph above).
- **Fail-stop waits.** Qwen3.5's hop compiles the build deadline in, and CP-lite's waits (the CP slots' q gathers, including the q writer's, their flag barriers and merges) take the same deadline from each wait's start; the waits of slots `bt_tail`, `gemm_ar` and `fc2ar` have none, as FlashInfer's BT owner reduce at the same points has none.
- **No silent fallback.** A Qwen3.5 slot whose packing the loaded weights do not fit runs stock on every rank instead, with the reason in the startup log; where the check reads rank-local weights (`route_lin`), the ranks agree first and each logs the decline as a warning.

## Manual tests

Qwen3.5 runs TP8, which spans two GB300 trays: its e2e tests read a server launched on them through `SGLANG_NVIDIA_MEGAKERNEL_E2E_URL` and the files its ranks log to (`SGLANG_NVIDIA_MEGAKERNEL_E2E_LOGS`, comma-separated).
