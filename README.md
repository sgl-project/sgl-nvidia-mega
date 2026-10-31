# SGLang-NVIDIA-Mega

## Introduction

**SGLang-NVIDIA-Mega** holds the NVIDIA megakernels of the [SGLang](https://github.com/sgl-project/sglang) framework: for each model it covers, the code of its small-batch decode step and the kernels that only make sense inside such a step, which hand their outputs to each other through mailboxes and programmatic dependent launch instead of through per-op boundaries.

The package loads as an SGLang [general plugin](https://github.com/sgl-project/sglang/blob/main/docs/docs/hardware-platforms/plugin.mdx) (the `sglang.srt.plugins` entry point) and is turned on with SGLang's `--enable-nvidia-megakernel` server flag. The package registers each model's spec with SGLang's megakernels framework (`sglang.kernels.plugin_kernels.nvidia`, `sglang.srt.megakernels` before sgl-project/sglang#45389). With the flag, SGLang checks the setup at argument resolution, before the weights load. After the model runner has prepared a covered model (before the KV cache is sized), SGLang builds the megakernel and replaces that model instance's `forward`, which runs each covered step itself and leaves every other step to the model's own forward. Every rank then logs `NVIDIA megakernel: active (…)`. Without the flag, SGLang runs untouched.

> **Experimental:** the kernels, the models and setups they cover, and the switch may change.

## Features

### Supported Models

| Model | GPU | Tensor parallel | Coverage |
| --- | --- | --- | --- |
| DeepSeek-V4.1-Flash | GB300 (SM103); 148-SM B300 (SM103) with `sm148` | 4 | DSpark target verify steps of 1 to 384 rows (batch 1 to 64 at 6 rows a request); on B300, speed was measured at one request only, serving passed a GSM8K canary at client concurrency 12 and 16 with decode CUDA graphs up to batch 16 (DSpark block 5), and larger batches and other block sizes rely on the kernel tests |
| Qwen3.5-397B-A17B (NVFP4) | GB300 (SM103) | 8 (two trays, one NVLink domain) | MTP target verify steps of 7 rows (batch 1) |
| Kimi K3 (packed MXFP4) | GB300 (SM103) | 8 | One-token decode steps of 1–8, 16, 32 and 64 rows |
| GLM-5.3, GLM-5.2 (NVFP4) | B300 (SM103) | 8 | MTP target verify steps of 4 to 16 rows (batch 1 to 4 at 4 rows a request, batch 1 to 2 at 6); the NextN draft step in its captured graphs |

For DeepSeek-V4.1-Flash and Qwen3.5, larger verify steps, prefill and extend run
SGLang's standard path. Their draft specs provide the DSpark proposal and the
Qwen3.5 EAGLE entries, respectively. For Kimi K3, the plugin covers the 93-layer
language model (24 MLA and 69 KDA layers) at the listed widths; every other call
runs the standard path. For GLM-5.3, larger verify steps, prefill and extend run
SGLang's standard path, and the NextN draft step runs on its draft spec.

GLM-5.2 has GLM-5.3's architecture and the same value of every config field the
GLM-5.3 specs pin (`HF_CONFIG` in `glm5_3/__init__.py`), so the same specs serve it,
and what this README says of GLM-5.3 holds for GLM-5.2. On `nvidia/GLM-5.2-NVFP4`
(8 x B300) the megakernel engages, a test-only step-by-step comparison with SGLang's
own target verify and draft steps shows no difference, and GSM8K-200 (strict match)
scores 0.975 with the megakernel and 0.960 without it.

The Kimi K3 adapter is experimental. On the pinned GB300/TP8 configuration
below, every listed CUDA-graph width and padding boundary passed dummy-weight
capture/replay, B1 and B16 passed stock-versus-plugin real-weight canaries, and a
specialized-route/captured-replay canary passed, and the GSM8K quality protocol
matched its frozen stock-path reference score. This is bounded functional and
quality evidence; it does not claim performance, unlisted configurations, production
readiness, or publication readiness.

### What a covered DeepSeek-V4.1-Flash step runs

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
| Attention and MoE all-reduce + mHC post + next norm (`two_shot`) | 49–384 | Two-shot exchange: each rank sends row t only to its owner rank (t mod 4), which sums the four partials in rank order and sends the row back; the MoE post also folds the top-k finalize and the shared-expert add, and quantizes the next input to MXFP8 where SGLang's fused post does |

With the megakernel on, the default 39 slots run: the ones above, the attention chain, hop B's triggers, and the DSpark draft step's 16 slots. `SGLANG_NVIDIA_MEGAKERNEL_SLOTS` (comma-separated) overrides that list for tests and bisecting; empty runs every step function's stock code. The opt-in slots run only when listed there: `target_lm_head` (the target's LM head on the low-M kernel at 1-8 rows), `lm_head_pinned_tactic` (both low-M LM heads on one fixed tactic per row count, untuned), `hop_b_quad_trigger` (hop B's finalize-publish launches its consumer right after its own wait, batch 1 only), `hop_b_cap16` and `draft_embed`; listing `sm148` there admits the 148 SMs of a B300 instead of the 152 required otherwise.

Qwen3.5 (`qwen3_5/runtime`) runs the same way: per layer the stage boundary (`boundary/`), the GatedDeltaNet or the attention (`gdn/`, `attention/`) and the MoE (`moe/`), each with a stock implementation that reproduces SGLang's verify step bit for bit. It reads the running forward's metadata and pools through SGLang's EAGLE binding (`sglang.srt.kernel_plugins.nvidia.eagle_binding`). Its slots run only when listed in `SGLANG_NVIDIA_MEGAKERNEL_SLOTS`, and each is bit for bit SGLang's:

| Slot | Rows | Kernel |
| --- | --- | --- |
| `route` | 7 | The router GEMM and the NVFP4 quantization of the MoE input in one launch |
| `mkm` (with `route`) | 7 | On `route`'s outputs, the top-10, FC1, SwiGLU, the NVFP4 requantization and FC2 in one launch |
| `gdn_verify` | 7 | The GatedDeltaNet verify chain (conv update, split, ReplaySSM verify, gated norm with the FP8 `out_proj` input) with its launch choices changed |
| `gdn_pdl` (with `gdn_verify`) | 7 | Its ReplaySSM verify launched as a programmatic dependent of the conv |
| `hop` | 7 | The layer boundary's fused all-reduce + residual add + RMSNorm points on an NVLink multicast hop |

These slots change the arithmetic (numeric: each sums in another order than SGLang's op, within the tolerances `spec.py` declares):

| Slot | Rows | Kernel |
| --- | --- | --- |
| `gdn_push` (with `gdn_verify`, `hop`) | 7 | The GatedDeltaNet `out_proj` as the hop's GEMM, whose epilogue pushes the partial into the hop's mailbox |
| `gdn_front` (with `gdn_verify`, `gdn_pdl`) | 7 (one request) | The GatedDeltaNet `in_proj_qkvz` (static FP8) and `in_proj_ba` (BF16) projections, the conv update and an L2 prefetch of the verify's state in one launch, then the ReplaySSM verify and the gated norm writing `out_proj`'s FP8 input, each launched dependent on the one before |
| `route_shared` (with `route`) | 1–8 | The shared expert's static-FP8 gate/up in the router front's launch, so the shared expert on the alternate stream waits for that launch only |

The sampling and acceptance after the verify, and the GatedDeltaNet extend of prefill, stay SGLang's own code. In an SGLang that has the switches, `SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH=1` runs the sampling and acceptance in CUDA graphs, and `SGLANG_ENABLE_GDN_PREFILL_GRAPH_EXTEND=1` captures the GatedDeltaNet extend inside the breakable prefill CUDA graphs.

GLM-5.3 (`glm5_3/`) follows the same layout: the executor's layer loop runs the hops between the sublayers (`boundary/`), the attention (`attention/`) and the MoE (`moe/`), then the logits (`logits/`), each with a stock path that reproduces SGLang's verify step bit for bit, and CuTe DSL kernels (`sm103/`) where a slot serves its width:

| Slot | Rows | Kernel |
| --- | --- | --- |
| Attention (`attention`) | 4–16 | The q_a / kv_a projection on a small-M GEMM (with the indexer's key / head-gate projection, up to 8 rows), the MLA prologue (norms, q_b, W_UK absorption, RoPE and FP8 query, the MLA and index-K cache rows), the indexer query, the sparse MLA over the fused top-k page table and the value up-projection; the indexer's top-k stays SGLang's |
| Routed and shared experts (`moe`) | 4–16 | One small-M NVFP4 engine launch for the routed experts and the BF16 shared expert, for 1, 2 or 3–4 groups of up to 4 rows (a request's rows with 4 draft tokens; 6 rows run as 4 + 2, 12 as 4 + 4 + 4) |
| All-reduce + residual + RMSNorm (`hops`) | 4–16 | One hop per tensor-parallel sum over symmetric memory: `o_proj` with the router on MoE layers, the MoE finalize folded into the MLP-output all-reduce, and the final norm |
| Shared-top-k q_a / kv_a (`qkv_a_small_m_shared`) | 4–8 | The small-M projection on the layers that reuse the previous layer's top-k (one packed weight copy each) |
| Deferred input norm (`dnorm`) | 4–8 | Those layers' input RMSNorm applied in the projection's epilogue from the fold hop's block sums (moves the rounding points; its tolerances are in `glm5_3/spec.py`) |
| NextN draft step (`draft`) | every captured draft-decode and draft-extend graph | The draft's embedding gather + eh norms in one kernel on the replicated table (any row count), the column-sharded `eh_proj` (1–4 rows, and 8, 12 and 16 with 4 draft tokens or 6 and 12 with 6; gathered by the hops' all-reduce on the target's mailbox), the distributed top-1 in place of the logits all-gather (up to 8 rows), and the draft layer's MLA prologue (up to 16 rows); each kernel's other row counts run SGLang's code |

With the megakernel on, all six slots run; none needs to be listed. `SGLANG_NVIDIA_MEGAKERNEL_SLOTS` (comma-separated) overrides that list for tests and bisecting only; a list without a slot another needs (`dnorm` needs `qkv_a_small_m_shared` and `hops`, `qkv_a_small_m_shared` needs `attention`) stops the server at startup, naming it. Slot `draft` needs embedding replication, which the flag turns on (SGLang sets `SGLANG_ENABLE_EMBED_REPLICATION` when it is unset), greedy draft acceptance and an SGLang whose EAGLE draft graph runners run a draft spec's capture check; the server stops at startup, naming each one that is missing (an explicit `SGLANG_ENABLE_EMBED_REPLICATION=0` too).

### Design

- **Built at load.** Buffers, kernels and tactics for every covered width are allocated and compiled when the model is loaded, before the KV cache is sized. Nothing is compiled or allocated inside a step.
- **Graph-safe coverage.** A step is covered by its forward mode and row count alone, so every rank decides the same and a captured graph replays the decision made at capture.
- **Fail-stop synchronization.** DeepSeek-V4.1-Flash waits on another GPU have
  a deadline below SGLang's watchdog. A rank that misses it faults at address
  0x0, so no token from a failed step is returned. Kimi K3 validates SGLang's
  finite build deadline and compiles it into all six reachable TP8 remote-wait
  paths; expiry likewise stores through address 0x0 and traps.
- **No silent fallback.** With `--enable-nvidia-megakernel`, an unsupported setup stops the server at startup, before the weights load, listing every reason; a model the loaded weights do not fit stops every rank after the load.

## Quick Start

### Requirements

- An SGLang with the megakernels framework (`sglang.kernels.plugin_kernels.nvidia`, or `sglang.srt.megakernels` before sgl-project/sglang#45389; `--enable-nvidia-megakernel`); Qwen3.5 also needs `sglang.srt.kernel_plugins.nvidia.eagle_binding`; GLM-5.3 also needs its DSA binding view (`ForwardView.dsa` in `sglang/srt/kernel_plugins/nvidia/binding.py`, the SGLang PR filed with this model), and refuses at startup without it; with that PR the flag also turns on the embedding replication slot `draft` needs (`FLAG_SETS_ENV` in `sglang/kernels/plugin_kernels/nvidia/spec.py`)
- SGLang's CUDA 13 image: FlashInfer (its low-latency block-scaled GEMM as in 0.7.0 to 0.7.2), CuTe DSL (`nvidia-cutlass-dsl`) and the CUDA Python bindings
- DeepSeek-V4.1-Flash: 4 x GB300, or 4 x 148-SM B300 (SM103) with `sm148` in `SGLANG_NVIDIA_MEGAKERNEL_SLOTS`
- Qwen3.5: 8 x GB300, two trays of one NVLink domain
- Kimi K3: 8 x GB300 (SM103), BF16, context length 8192, TP8 and DP1,
  FP8 E4M3 KV cache with page size 64, `trtllm_mla` for prefill and decode,
  `flashinfer_mxfp4` with default precision, and full decode CUDA graphs
- GLM-5.3, GLM-5.2: 8 x B300

### Installation

```bash
git clone https://github.com/sgl-project/sgl-nvidia-mega.git
cd sgl-nvidia-mega
pip install -e .
```

### Usage

DeepSeek-V4.1-Flash:

```bash
SGLANG_SPEC_TP_SYNC=rng,-dspark-accept-sample \
SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank \
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V4.1-Flash \
  --tp 4 \
  --ep-size 1 \
  --kv-cache-dtype fp8_e4m3 \
  --dsv4-attn-backend flashmla \
  --speculative-algorithm DSPARK \
  --speculative-dspark-block-size 5 \
  --enable-nvidia-megakernel
```

The three environment variables are required by the DeepSeek-V4.1 spec; without them the server stops at startup, naming each.

Qwen3.5, on each of the two trays (`--node-rank 0` and `1`):

```bash
SGLANG_NVIDIA_MEGAKERNEL_SLOTS=route,mkm,gdn_verify,gdn_pdl,hop \
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

GLM-5.3 or GLM-5.2 (NVFP4) on 8 x B300, with an NVFP4 GLM-5.3 or GLM-5.2 checkpoint (for example `nvidia/GLM-5.2-NVFP4`) whose config has the shapes in `glm5_3/__init__.py` (`HF_CONFIG`; the spec checks them at startup):

```bash
python -m sglang.launch_server \
  --model-path <NVFP4 GLM-5.3 or GLM-5.2 checkpoint> \
  --trust-remote-code \
  --tp-size 8 \
  --kv-cache-dtype fp8_e4m3 --page-size 64 \
  --attention-backend dsa --dsa-decode-backend trtllm \
  --moe-runner-backend flashinfer_trtllm \
  --disable-shared-experts-fusion \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --enable-nvidia-megakernel
```

Every GLM-5.3 slot runs with `--enable-nvidia-megakernel` alone; no slot list and no environment variable are needed. The GLM-5.3 NextN draft step (slot `draft`) needs embedding replication, which the flag turns on: SGLang sets `SGLANG_ENABLE_EMBED_REPLICATION` when it is unset and logs it, and keeps a value you set (`0` stops the server at startup, naming it). The draft step also needs greedy draft acceptance (no `--speculative-use-rejection-sampling`, SGLang's default); without it the server stops at startup, naming it. SGLang's defaults give the rest of the environment the spec needs; a setup it does not serve stops at startup, naming each reason. Serving with HiCache (`--enable-hierarchical-cache`) is supported: the covered steps read only the running requests' device KV pages.

The GLM-5.3 megakernel serves MTP with 3 draft steps (`--speculative-num-steps 3 --speculative-num-draft-tokens 4`, as above) or 5 (`--speculative-num-steps 5 --speculative-num-draft-tokens 6`); any other draft length stops the server at startup, naming the served values.

The GLM-5.3 megakernel covers verify steps of up to 16 rows: 1 to 4 requests with 4 draft tokens, 1 or 2 with 6; larger steps run SGLang's path while its derived weight copies (about 20 GiB per GPU, logged at startup) still take KV-cache memory, so it helps little on high-concurrency servers, and the server warns when `--max-running-requests` is set above 8.

## Repository Structure

```text
sgl-nvidia-mega/
├── sgl_nvidia_mega/
│   ├── plugin.py          # registers the specs with SGLang's megakernels framework
│   ├── common/            # NVRTC builds and PDL launches; CuTe DSL sync primitives
│   ├── deepseek_v41/
│   │   ├── spec.py          # the spec: the setups it serves and its build
│   │   ├── plan.py          # which kernel fills each slot, per width
│   │   ├── runtime/         # the build, the executor and its per-step state, the embedding
│   │   ├── boundary/        # the mHC between sublayers
│   │   ├── attention/       # the attention step
│   │   ├── moe/             # the MoE step
│   │   ├── engram/          # the Engram layers
│   │   ├── logits/          # the head and the logits
│   │   └── sm103/           # kernels and the build-time runner; no SGLang imports
│   ├── qwen3_5/              # spec, plan, runtime, boundary, GDN, attention, MoE and SM103 kernels
│   ├── glm5_3/            # GLM-5.3 and GLM-5.2: the same layout (spec, plan, runtime, boundary, attention, moe, logits, sm103)
│   └── kimi_k3/              # the Kimi K3 spec, adapter, executor and CuTe DSL kernels
├── benchmark/             # the static decode table behind the speed numbers
└── test/
    ├── unit/              # CPU: the plugin, the setup checks, the step loop and the SGLang contract
    ├── kernels/           # GPU: each kernel against SGLang's or FlashInfer's op, fail-stop
    └── e2e/               # GPU: serving accuracy, KL, passkey and fail-stop, switch on and off
```

## Testing

```bash
pytest test/unit               # CPU only
pytest test/kernels            # DeepSeek-V4.1-Flash: 4 x GB300
pytest test/e2e                # DeepSeek-V4.1-Flash: 4 x GB300, with SGLANG_NVIDIA_MEGAKERNEL_E2E_MODEL_PATH set
```

Qwen3.5 runs TP8, which spans two GB300 trays: its e2e tests read a server launched on them through `SGLANG_NVIDIA_MEGAKERNEL_E2E_URL` and the files its ranks log to (`SGLANG_NVIDIA_MEGAKERNEL_E2E_LOGS`, comma-separated).

GLM-5.3 and Kimi K3 GPU and end-to-end suites are not yet part of these commands.
The GLM-5.3 CPU config and SGLang contract checks are included in `test/unit`.

## Contributing

Open a pull request against `main`. The PR title follows `[Type] scope: title` (see `.github/pull_request_template.md`), and `pre-commit run --all-files` must pass.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
