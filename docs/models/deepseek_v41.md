# DeepSeek-V4.1-Flash

| GPU | Tensor parallel | Coverage |
| --- | --- | --- |
| GB300 (SM103); 148-SM B300 (SM103) with `sm148` | 4 | DSpark target verify steps of 1 to 384 rows (batch 1 to 64 at 6 rows a request); on B300, speed was measured at one request only, serving passed a GSM8K canary at client concurrency 12 and 16 with decode CUDA graphs up to batch 16 (DSpark block 5), and larger batches and other block sizes rely on the kernel tests |
| GB300 (SM103) | 8, with the MoE at TP4 x EP2 (two trays, one NVLink domain) | The same verify widths and the DSpark draft step; served and measured at one request (DSpark block 5); larger batches rely on the world-8 kernel tests. See [TP8 + EP2](#tp8--ep2) |

Larger verify steps, prefill and extend run SGLang's standard path. The draft spec provides the DSpark proposal.

## Requirements

- 4 x GB300, or 4 x 148-SM B300 (SM103) with `sm148` in `SGLANG_NVIDIA_MEGAKERNEL_SLOTS`; at TP8 + EP2, 8 x GB300 on two trays of one NVLink domain
- FlashInfer with its low-latency block-scaled GEMM as in 0.7.0 to 0.7.3
- Waits on another GPU have a deadline below SGLang's watchdog. A rank that misses it faults at address 0x0, so no token from a failed step is returned.

## Launch

```bash
SGLANG_SPEC_TP_SYNC=rng,-dspark-accept-sample \
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V4.1-Flash \
  --tp 4 \
  --ep-size 1 \
  --kv-cache-dtype fp8_e4m3 \
  --dsv4-attn-backend flashmla \
  --engram-config '{"placement": "cpu"}' \
  --speculative-algorithm DSPARK \
  --speculative-dspark-block-size 5 \
  --enable-nvidia-megakernel
```

`SGLANG_SPEC_TP_SYNC` and `--engram-config` are required by the DeepSeek-V4.1 spec; without them the server stops at startup, naming each.

## What a covered step runs

The step is the plugin's own code over the weights SGLang loaded (`deepseek_v41/runtime`): the embedding and the Engram hashes, then per layer the mHC boundary (`boundary/`), the attention (`attention/`), the MoE (`moe/`) and Engram (`engram/`), then the head and the logits (`logits/`). Each package has a stock implementation that reproduces SGLang's decode step bit for bit, and puts this package's kernels in where the slot plan (`plan.py`) gives them a width:

| Slot | Rows | Kernel |
| --- | --- | --- |
| `wo_b` + attention all-reduce + mHC post + next norm (`hop_a`, `hop_a_batch2`) | 1–16 | `wo_b` GEMM whose epilogue stores its rows into a Lamport mailbox on every rank, and a Lamport consumer that reduces them and runs the mHC post, combine and RMSNorm |
| MoE all-reduce + mHC post + next norm (`hop_b`, `hop_b_wide`) | 1–48 | Hop B: the top-k finalize and the shared-expert add on a Lamport mailbox, whose consumer runs the mHC post, combine and RMSNorm; past 8 rows it writes the next input's MXFP8 scales per 8-row tile, so the next layer's `wqkv_a` runs on the low-latency GEMM. Hop A's widths use hop A's mailbox, the wider ones a second one |
| Routed experts (`moe`) | 1–48 | A route kernel and a pair of block-scaled tcgen05 GEMMs with 8-row tiles; an expert chosen by more than 8 rows takes several tiles |
| Router top-k (`gate`, `gate_wide`) | 1–48 | One CTA per row, bit for bit SGLang's fused gate (including the vision-token routing) |
| Shared expert gate_up (`shared_ll`) | 1–48 | FlashInfer's low-latency MXFP8 GEMM with 8-row tiles on the alternate stream from the layer's fork, after a one-pass MXFP8 quant into its row-tile layout (bit for bit the shared linear's own quant); SGLang's clamped SwiGLU and `down_proj` follow |
| Replicated embedding (`embed`) | 1–48 | SGLang's fused lookup on the whole table, all-gathered once at build; no all-reduce |
| Indexer top-k (`attn_indexer`) | 30–48 (batch 5–8) | The decode indexer's top-k in the plugin's forward, reading SGLang's paged indexer metadata and writing its selection into SGLang's buffer; the index query and head weights run on the layer's second stream, under the compressor. At 6–24 rows, chunked metadata or without paged metadata SGLang's indexer runs |
| Engram prefetch (`engram`) | 1–8 (batch 1) | Every Engram layer's lookup and WKV quarter on a side stream; their two all-reduces on the main stream (SGLang's TP group) |
| Attention and MoE all-reduce + mHC post + next norm (`two_shot`) | 49–384 | Two-shot exchange: each rank sends row t only to its owner rank (t mod the TP size), which sums the partials in rank order and sends the row back; the MoE post also folds the top-k finalize and the shared-expert add, and quantizes the next input to MXFP8 where SGLang's fused post does |

With the megakernel on, the default 70 slots run (`runtime.SHIPPED_SLOTS`): the ones above, the attention chain, hop B's triggers, the DSpark draft step's slots (`draft_hop` among them, which acts at TP8 only), and the 30 slots in the first table below (`runtime.DEFAULT_ON_SLOTS`). Each of those 30 measured faster served at batch 1 (a 95 % interval below zero) and passed the accuracy checks; the ones marked **numerics change** are not bitwise with the path they replace (`runtime.NUMERICS_CHANGE_SLOTS`) and are checked with evals instead. On a default start one of them with nothing to serve stays off (and `attn_q_cluster` and `wo_b_publish_split` stay off at TP4, where they measured slower), and the startup summary names it under "default slots left off". `markov_shard_logits` and `moe_balanced` act only when SGLang asks for them through a switch that is not in SGLang main yet (see the table); without it they leave the step unchanged.

`SGLANG_NVIDIA_MEGAKERNEL_SLOTS` (comma-separated) replaces the default list, for turning slots off, tests and bisecting; empty runs every step function's stock code. To turn one default slot off, list the default slots without it, for example:

```bash
export SGLANG_NVIDIA_MEGAKERNEL_SLOTS=$(python -c "from sgl_nvidia_mega.deepseek_v41.runtime import SHIPPED_SLOTS as s; print(','.join(x for x in s if x != 'wo_b_publish_split'))")
```

To run only the bitwise default slots, also leave out every name in `runtime.NUMERICS_CHANGE_SLOTS`. A slot listed explicitly must build, and a listed slot whose prerequisite is missing refuses the start.

Default-on slots (DeepSeek-V4.1-Flash):

| Slot | Numerics | What it does | Origin |
| --- | --- | --- | --- |
| `gate_route` | bitwise | The router's top-k and the routed pair's route record in one cluster launch at 1-8 rows, target and draft | own |
| `gate_route_quant` | bitwise | With `gate_route`: that launch also writes the routed input's MXFP8 quant (FlashInfer's linear-layout `mxfp8_quantize` bit for bit), so the MoE half forks no routed-quant stream and the top-k launches as the router's programmatic dependent | own |
| `gate_route_early` | bitwise | With `gate_route`: SGLang's tensor-core router GEMM, rebuilt from its `tiny_mma` kernel on SGLang's own schedule, releases the top-k once its weight loads are issued instead of after its cluster reduction; checked against `tiny_gemm_bf16` at build. The top-k then releases GEMM1 after the row's top-k, not right after its own wait, so GEMM1's waiting CTAs no longer take the SMs ahead of the layer's shared expert (`SGL_NVIDIA_MEGA_GATE_ROUTE_EARLY_RELEASE`: 1 after the wait, 2 at completion, 0 at entry) | SGLang kernel; release point as TensorRT-LLM's router GEMM |
| `gemm2_pipe` | bitwise | The routed GEMM2, target and draft, on a pipelined prelude build of the same kernel | derived from muLLM |
| `shared_fold`, `shared_fold_quant`, `shared_fold_early_release` | bitwise | With `shared_ll`: the shared expert's gate_up epilogue writes the clamped SwiGLU and `down_proj`'s MXFP8 input, and releases `down_proj`'s GEMM at its loads | own |
| `hop_mbar` | bitwise | The Lamport consumers' RMSNorm sums cross the cluster by `st.async` onto mbarriers | own |
| `attn_indexer_narrow`, `attn_indexer_fused` | bitwise (ties: in-order pick) | Slot `attn_indexer` at the swap-AB widths too, and the candidate consumers' logits + top-k at 1-8 rows in one cluster launch | own |
| `prologue_fork`, `hc_coeff_fused` | bitwise | The Engram hashes forked at the step start; the prenorm's mHC coefficients in one launch | own |
| `attn_q_norm_gemm` | bitwise | q_norm + MXFP8 quant inside the low-latency wq_b GEMM at 1-8 rows where no indexer reads q_lora | own |
| `attn_compressor_priority` | bitwise | The compressor layers' compressor on a stream at the device's highest priority | derived from muLLM |
| `markov_lowlat` | bitwise | The draft's Markov chain with converged pollers and acquire loads in its exchange, the direct exchange to 4 rows at TP4 and 2 at TP8 | own; acquire idiom as muLLM's |
| `markov_tail` | bitwise | The Markov chain also advances the draft-noise offset, and writes the verify ids `[anchor \| draft tokens]` when SGLang hands them over (an SGLang-side switch that is not upstream yet; without it only the offset add goes) | own |
| `draft_stage0_table` | bitwise | The first draft stage's query and K/V rows gathered in one launch from a per-token table, filled by the live kernels at the first eager forward and checked bit for bit per width; about 1.1 GiB per GPU at TP8 and about 20 s at startup | design derived from TensorRT-LLM |
| `engram_publish` | bitwise | With the Engram prefetch, each Engram layer's WKV columns go into every rank's mailbox as tagged words and the main stream collects them at the layer, in place of the all-reduce of zero-padded rows at layer 1 | derived from muLLM |
| `attn_indexer_logits_q6` | bitwise | With `attn_indexer`: the full paged index logits of 5-7 rows a request (the candidate source and full top-k layers) on DeepGEMM's paged MQA logits kernel with all of a request's rows in one Q block, so the index K streams once per step instead of once per 4 rows; the 6-row, page-128 variant is built at load, any other shape keeps SGLang's call | adapted from DeepGEMM |
| `gemm2_reg_pool` | bitwise | With `gemm2_pipe`: that GEMM2, target and draft, built at 96 registers per thread (`__maxnreg__`) instead of 128, so a co-running kernel's CTA can land on its SM | own; register pool as muLLM's |
| `markov_shard_logits` | bitwise | When SGLang's accept reads only each rank's own draft-logit columns (it passes `step_logits_shard_only` to the draft sampler; not in SGLang main yet), the Markov chain keeps only those columns: no peer staging stores, no publishing fence, no epilogue copy. Without it the chain is built as before | own; sharded draft logits as muLLM's |
| `target_lm_head_xload` | bitwise | With `target_lm_head_mxfp8`: that stream brings the rows of x into shared memory as 16-byte loads all issued before their stores (rows not 16-byte aligned keep the 8-byte loop), so its first MMAs wait one L2 round trip for x instead of several | own |
| `swapab_h8`, `wo_a_g1` | **numerics change** at TP8 | At attention TP8: swap-AB on the 8 local heads and the fused wo_a on the 1 local group, target and draft, in place of SGLang's FlashMLA and einsum (bitwise with the TP4 kernels on the same heads, not with SGLang's TP8 path); no effect at TP4 | own |
| `target_lm_head`, `target_lm_head_mxfp8` | **numerics change** | The target's LM head at 1-8 rows on the low-M kernel, streaming an MXFP8 copy of the vocabulary shard (E4M3 codes, UE8M0 scales of 32 columns, quantized at load) | own |
| `attn_q_cluster` | **numerics change** | q_lora norm + MXFP8 quant + wq_b in one cluster launch at 1-8 rows where no indexer reads q_lora (a different RMS order, an FP8 `mma.sync` k-block fold); a default start takes it at attention TP8 only (8 local heads; at TP4 it measured +21 us per step slower), listed it runs at TP4 too | derived from muLLM |
| `wo_b_publish_split` | **numerics change** | wo_b at 1-8 rows as a cluster split-K FP8 projection whose epilogue publishes into hop A's mailbox (and, with `draft_hop`, the draft hop's), in place of the publishing low-latency GEMM (a different FP32 summation order); a default start takes it at attention TP8 only (1024 wo_b inputs per rank; at TP4 it measured +94 us per step slower), listed it runs at TP4 too | derived from muLLM |
| `draft_shared_ll` | **numerics change** (draft only) | The draft stages' shared expert on the low-latency GEMM with the folded epilogue, as `shared_ll` runs the target's; the draft's proposals may change, the target's verify does not | design as muLLM's draft MoE |
| `moe_balanced` | **numerics change** | At attention TP8 + MoE TP4 x EP2 with SGLang's balanced expert storage (every rank holds its MoE-TP slice of all 384 routed target experts, about 47 GB more weights per GPU; an SGLang-side switch that is not in SGLang main yet): the route record deals each step's distinct experts, in id order, to the two expert-parallel groups in turn, so each group runs half of them whatever the routing (the experts' partial sums meet at hop B in another grouping). Without that storage it has nothing to serve and a default start leaves it off | rule as muLLM's balanced expert groups |

Opt-in slots (run only when listed):

| Slot | Numerics | What it does | Why it is not on by default |
| --- | --- | --- | --- |
| `attn_core_lean` | **numerics change** | At attention TP8, 1-8 rows: a one-launch sparse attention core (cluster merge with the sink and the inverse RoPE in the kernel) in place of swap-AB + combine, target and draft, on SGLang's `v41_new` KV layout (V4.1 KV encodings, P in BF16, its own split and merge order); needs `swapab_h8` and `wo_a_g1`; derived from muLLM | It needs `SGLANG_DSV4_KV_LAYOUT=v41_new`, a server-wide SGLang setting the plugin does not choose. Under `v41_new` swap-AB declines, so TP4 and verify widths above 8 rows run SGLang's attention, which is much slower there; set the variable only for attention-TP8 batch-1 serving |
| `draft_final_collapse` | bitwise | With `draft_hop`: the last draft stage's MoE post also writes the head's mHC collapse | No resolved served gain |
| `draft_stage0_first` | bitwise | With `draft_stage0_table`: the first draft stage's attention is issued from the token ids ahead of the embedding, whose lookup, hc broadcast and the stage's mHC coefficients run on a side stream joined at the stage's attention post | No resolved served gain alone; on current SGLang the draft's metadata preparation still binds the first stage |
| `hop_final_collapse` | bitwise at TP4; **not bitwise at TP8** | The last layer's hop B also writes the head's mHC collapse, so the head runs only its RMSNorm | At attention TP8 + EP2 it changed one DSpark proposal in 4183 verify steps: target outputs identical, one prompt took one extra verify step |
| `attn_cluster_merge`, `attn_index_q_first` | bitwise | Swap-AB's tile merge in the CTA cluster; the index query ahead of wq_b | Small gains not yet confirmed served: `attn_cluster_merge` -3.3 us per step in a 2 + 2-launch A/B; `attn_index_q_first` faster in the one-GPU index-chain bench, not resolved served |
| `attn_wo_a_fp8`, `attn_index_weights_bf16` | bitwise (`attn_wo_a_fp8`: where its FP8 recovery is exact, checked per layer) | wo_a streamed as E4M3 (32x32 UE8M0 scales, half the bytes) and widened exactly in shared memory; BF16 index head weights made beside the head weights, not cast before the logits | In the full batch-1 stack at attention TP8 + MoE TP4 x EP2 (real-text prompts, temperature 1, real routing, acceptance forced to 3.51), server step p50: `attn_wo_a_fp8` -13.1 us [-15.5, -10.7], `attn_index_weights_bf16` -7.7 us [-10.2, -5.1] (one launch per arm, pass-level interval); alone on an earlier stack the gain was not resolved |
| `lm_head_pinned_tactic` | untuned tactics | Both low-M LM heads on one fixed tactic per row count | Measurement aid |
| `draft_lm_head_mxfp8` | **measurement only** | With `draft_lm_head`: the draft's LM head on the MXFP8 copy | The checkpoint ships one BF16 `head.weight` shared by the target and the draft, and InferenceX's review rules forbid a draft head below the checkpoint's precision (`plan.MEASUREMENT_ONLY_SLOTS`) |
| `hop_b_quad_trigger`, `hop_b_cap16`, `draft_embed` | bitwise | Hop B's consumer launched right after the finalize-publish's wait; hop B's row tiles at 9-16 rows only; the draft's tokens from the target's gathered table | No served gain shown (`hop_b_cap16`: +16 us on the batch-1 verify graph) |

Listing `sm148` admits the 148 SMs of a B300 instead of the 152 required otherwise.

## TP8 + EP2

On two GB300 trays of one NVLink domain the spec admits one more layout: attention TP8 with the MoE at TP4 x EP2 (`--tp 8 --ep-size 2`). Each EP half holds 192 of the 384 routed experts and the shared expert is split over all 8 ranks, as SGLang loads them; every other (tp, ep) pair, TP4 over several nodes, a non-trivial expert placement and `SGLANG_SHARED_EXPERT_TP1` are refused at startup. The recommended launch, on each tray (`--node-rank 0` and `1`):

```bash
SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION=1 \
SGLANG_SPEC_TP_SYNC=rng,-dspark-accept-sample \
NCCL_CUMEM_ENABLE=1 NCCL_MNNVL_ENABLE=1 \
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V4.1-Flash \
  --tp 8 --ep-size 2 --nnodes 2 --node-rank 0 --dist-init-addr <tray-0 host>:<port> \
  --moe-a2a-backend none --moe-runner-backend flashinfer_mxfp4 \
  --kv-cache-dtype fp8_e4m3 \
  --dsv4-attn-backend flashmla \
  --engram-config '{"placement": "cpu"}' \
  --speculative-algorithm DSPARK \
  --speculative-dspark-block-size 5 \
  --enable-nvidia-megakernel
```

The default slot list serves this layout; `SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION=1` turns on SGLang's own DSpark runtime optimizations for megakernel serving (fused accept tail, deferred draft KV store and others in SGLang's `environ.py`); the TP8 numbers on this page were measured with it. Each rank keeps its own Engram host table (`per_rank`), as at TP4.

`swapab_h8` and `wo_a_g1` change the arithmetic against stock SGLang at TP8, which runs the decode attention as FlashMLA with the 8 local heads padded to 64 and wo_a as inverse RoPE, a BF16 einsum and wo_b's MXFP8 quantize. The two slots run TP4's kernels instead: swap-AB on the 8 local heads and the fused wo_a on the 1 local group, each bit for bit the TP4 kernel on the same heads or group. The fused wo_a writes the same MXFP8 codes as stock TP8 at 1 and 6 rows; at 16 rows 0.012 % of codes move by at most one eighth of a block step. GSM8K with the two slots matched stock TP8. Without them TP8 still serves, on SGLang's FlashMLA and einsum, at about 4.0 ms per verify step instead of 3.2 ms (one request, DSpark acceptance length forced to 3.51).

For attention-TP8 serving at batch 1 the opt-in `attn_core_lean` saves about 130 us per verify step more. It needs SGLang's `v41_new` KV layout, a server-wide setting the plugin does not choose, so add both to the launch above:

```bash
SGLANG_DSV4_KV_LAYOUT=v41_new \
SGLANG_NVIDIA_MEGAKERNEL_SLOTS=$(python -c "from sgl_nvidia_mega.deepseek_v41.runtime import SHIPPED_SLOTS as s; print(','.join(s + ('attn_core_lean',)))") \
```

Under `v41_new` swap-AB declines (it reads only the V4 layout), so verify steps above 8 rows run SGLang's attention and are much slower; do not set it for batched serving or at TP4.

SGLang fuses its all-reduce + mHC posts only at TP2 and TP4. At TP8 the target's posts run on the plugin's hops (hop A, hop B and the two-shot) and the draft's posts on a Lamport mailbox of their own (slot `draft_hop`, shipped; it builds nothing at TP4). At startup the draft logs each requested draft slot as built, declined (with the reason) or not needed.

## Manual tests

The DeepSeek-V4.1-Flash TP8 + EP2 kernel tests (`test_deepseek_v41_moe_ep.py`, `test_deepseek_v41_hop_b.py`, `test_deepseek_v41_hops.py`, `test_deepseek_v41_draft_hop.py`, and `test_deepseek_v41_draft.py -k ranks`) run on both trays under a two-node torchrun (`--nnodes 2 --nproc-per-node 4 -m pytest <file> -s`).
