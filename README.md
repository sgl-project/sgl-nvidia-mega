# SGLang-NVIDIA-Mega

## Introduction

**SGLang-NVIDIA-Mega** holds the NVIDIA megakernels of the [SGLang](https://github.com/sgl-project/sglang) framework: for each model it covers, the code of its small-batch decode step and the kernels that only make sense inside such a step, which hand their outputs to each other through mailboxes and programmatic dependent launch instead of through per-op boundaries.

The package loads as an SGLang [general plugin](https://github.com/sgl-project/sglang/blob/main/docs/docs/hardware-platforms/plugin.mdx) (the `sglang.srt.plugins` entry point) and is turned on with SGLang's `--enable-nvidia-megakernel` server flag. The package registers each model's spec with SGLang's megakernels framework (`sglang.srt.megakernels`). With the flag, SGLang checks the setup at argument resolution, before the weights load. After the model runner has prepared a covered model (before the KV cache is sized), SGLang builds the megakernel and replaces that model instance's `forward`, which runs each covered step itself and leaves every other step to the model's own forward. Every rank then logs `NVIDIA megakernel: active (…)`. Without the flag, SGLang runs untouched.

> **Experimental:** the kernels, the models and setups they cover, and the switch may change.

## Features

### Supported Models

| Model | GPU | Tensor parallel | Coverage |
| --- | --- | --- | --- |
| DeepSeek-V4.1-Flash | GB300 (SM103) | 4 | DSpark target verify steps of 1 to 384 rows (batch 1 to 64 at 6 rows a request) |
| GLM-5 DSA (NVFP4) | GB300 (SM103), Rubin (SM107) | 8 | EAGLE topk-1 target verify steps of whole requests up to 16 rows (batch 1 to 4 at 4 rows a request, 1 to 2 at 6), NextN draft and draft-extend steps of 1 to 16 rows |

Larger verify steps, prefill and extend run SGLang's standard path. The DSpark draft step runs on the plugin's draft spec.

GLM-5 DSA needs `--kv-cache-dtype fp8_e4m3 --page-size 64 --attention-backend dsa --dsa-decode-backend trtllm --moe-runner-backend flashinfer_trtllm --speculative-algorithm EAGLE --speculative-eagle-topk 1 --speculative-num-steps 3 --speculative-num-draft-tokens 4` (or `--speculative-num-steps 5 --speculative-num-draft-tokens 6`) and `SGLANG_ENABLE_EMBED_REPLICATION=1`. Without NVLink multicast (no all-reduce fusion), set `SGLANG_NVIDIA_MEGAKERNEL_GLM_ALLOW_NO_AR_FUSION=1`.

### Speculative step graph (any EAGLE v2 model)

Off by default; each switch installs its SGLang hooks only when set to `1`.

| Switch | Effect |
| --- | --- |
| `SGLANG_NVIDIA_MEGAKERNEL_SPEC_STEP_GRAPH` | One CUDA graph per topk-1 decode step (draft, target verify, draft extend and the work between them), batch size 1 to 8 |
| `SGLANG_SPEC_STEP_GRAPH_FUSED_INPUT_COPY` | The step graph's input copies in one launch |
| `SGLANG_SPEC_PACKED_VERIFY_BROADCAST` | One TP broadcast of the verify result instead of three |
| `SGLANG_SPEC_LEAN_SAMPLED_VERIFY` | Chain sampled verify without the expanded temperatures and zero-filled draft probs |
| `SGLANG_SPEC_CHUNKED_VERIFY_SAMPLING` | With the lean verify: the sampling passes split over vocabulary chunks |
| `SGLANG_SPEC_FUSED_DRAFT_EXTEND_TAIL` | The topk-1 draft-extend argmax, probs and DSA seed gather in two launches |
| `SGLANG_SPEC_FUSED_RELAY` | The relay stash in one scatter, row views for a single request |

Every fusion's outputs are bitwise identical to the path it replaces.

### What a covered step runs

The step is the plugin's own code over the weights SGLang loaded (`deepseek_v41/runtime`): the embedding and the Engram hashes, then per layer the mHC boundary (`boundary/`), the attention (`attention/`), the MoE (`moe/`) and Engram (`engram/`), then the head and the logits (`logits/`). Each package has a stock implementation that reproduces SGLang's decode step bit for bit, and puts this package's kernels in where the slot plan (`plan.py`) gives them a width:

| Slot | Rows | Kernel |
| --- | --- | --- |
| `wo_b` + attention all-reduce + mHC post + next norm (`hop_a`, `hop_a_batch2`) | 1–16 | `wo_b` GEMM whose epilogue stores its rows into a Lamport mailbox on every rank, and a Lamport consumer that reduces them and runs the mHC post, combine and RMSNorm |
| Routed experts (`moe`) | 1–48 | A route kernel and a pair of block-scaled tcgen05 GEMMs with 8-row tiles; an expert chosen by more than 8 rows takes several tiles |
| Router top-k (`gate`, `gate_wide`) | 1–48 | One CTA per row, bit for bit SGLang's fused gate (including the vision-token routing) |
| Replicated embedding (`embed`) | 1–48 | SGLang's fused lookup on the whole table, all-gathered once at build; no all-reduce |
| Engram prefetch (`engram`) | 1–8 (batch 1) | Every Engram layer's lookup and WKV quarter on a side stream; their two all-reduces on the main stream (SGLang's TP group) |
| Attention and MoE all-reduce + mHC post + next norm (`two_shot`) | 49–384 | Two-shot exchange: each rank sends row t only to its owner rank (t mod 4), which sums the four partials in rank order and sends the row back; the MoE post also folds the top-k finalize and the shared-expert add, and quantizes the next input to MXFP8 where SGLang's fused post does |

With the megakernel on, the default 36 slots run: the ones above, the attention chain, `hop_b` (the MoE post on hop A's Lamport mailbox) with its triggers, and the DSpark draft step's 16 slots. `SGLANG_NVIDIA_MEGAKERNEL_SLOTS` (comma-separated) overrides that list for tests and bisecting; empty runs every step function's stock code. `draft_embed` runs only when listed there.

### Design

- **Built at load.** Buffers, kernels and tactics for every covered width are allocated and compiled when the model is loaded, before the KV cache is sized. Nothing is compiled or allocated inside a step.
- **Graph-safe coverage.** A step is covered by its forward mode and row count alone, so every rank decides the same and a captured graph replays the decision made at capture.
- **Fail-stop synchronization.** Every wait on another GPU has a deadline below SGLang's watchdog. A rank that misses it faults at address 0x0, so no token from a failed step is returned.
- **No silent fallback.** With `--enable-nvidia-megakernel`, an unsupported setup stops the server at startup, before the weights load, listing every reason; a model the loaded weights do not fit stops every rank after the load.

## Quick Start

### Requirements

- An SGLang with the megakernels framework (`sglang.srt.megakernels`, `--enable-nvidia-megakernel`)
- SGLang's CUDA 13 image: FlashInfer (its low-latency block-scaled GEMM as in 0.7.0 to 0.7.2), CuTe DSL (`nvidia-cutlass-dsl`) and the CUDA Python bindings
- 4 x GB300

### Installation

```bash
git clone https://github.com/sgl-project/sgl-nvidia-mega.git
cd sgl-nvidia-mega
pip install -e .
```

### Usage

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

## Repository Structure

```text
sgl-nvidia-mega/
├── sgl_nvidia_mega/
│   ├── plugin.py          # registers the specs with SGLang's megakernels framework
│   ├── common/            # NVRTC builds and PDL launches; CuTe DSL sync primitives
│   └── deepseek_v41/
│       ├── spec.py          # the spec: the setups it serves and its build
│       ├── plan.py          # which kernel fills each slot, per width
│       ├── runtime/         # the build, the executor and its per-step state, the embedding
│       ├── boundary/        # the mHC between sublayers
│       ├── attention/       # the attention step
│       ├── moe/             # the MoE step
│       ├── engram/          # the Engram layers
│       ├── logits/          # the head and the logits
│       └── sm103/           # kernels and the build-time runner; no SGLang imports
├── benchmark/             # the static decode table behind the speed numbers
└── test/
    ├── unit/              # CPU: the plugin, the setup checks, the step loop and the SGLang contract
    ├── kernels/           # GPU: each kernel against SGLang's or FlashInfer's op, fail-stop
    └── e2e/               # GPU: serving accuracy, KL, passkey and fail-stop, switch on and off
```

## Testing

```bash
pytest test/unit               # CPU only
pytest test/kernels            # 4 x GB300
pytest test/e2e                # 4 x GB300, with SGLANG_NVIDIA_MEGAKERNEL_E2E_MODEL_PATH set
```

## Contributing

Open a pull request against `main`. The PR title follows `[Type] scope: title` (see `.github/pull_request_template.md`), and `pre-commit run --all-files` must pass.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
