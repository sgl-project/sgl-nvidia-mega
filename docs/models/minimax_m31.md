# MiniMax-M3.1-Flash (NVFP4)

| GPU | Tensor parallel | Coverage |
| --- | --- | --- |
| GB300 (SM103) | 4 | DSpark target verify steps of 1 to 16 requests (8 to 128 rows, 8 a request) in one server: one request on the plugin's kernels, 2 to 16 on SGLang's step code with two bitwise MoE slots; one request's 7-row draft block; and the draft proposal's Markov chain at 1 to 16 requests |

Prefill, plain decode and verify steps of more than 16 requests run SGLang's standard path.

## Requirements

- 4 x GB300 on one host
- The MSA kernels SGLang's MiniMax-M3.1 support imports (SGLang's cu134 image carries them at SGLang's pinned commit) and Triton (slot `markov`'s noise kernel)
- A CUDA toolkit (`nvcc`) >= 13.4 at `CUDA_HOME`, while slot `mega` or `markov` is requested

The package's kernels build with `nvcc` (`minimax_m31/sm103/nvcc.py`), not with NVRTC as the other models' kernels do (`common/jit.py`), for two reasons:

- they are whole CUDA translation units (MegaAttention and the token-parallel MoE from PR #34, the draft backbone and the Markov chain), with host code beside the kernels (tensor-map encoding, the cooperative-launch helpers) that NVRTC does not compile; `nvcc` compiles it and the plugin keeps only the cubin, which it launches from Python;
- MegaAttention dequantizes NVFP4 with the QMUL4 instruction, which needs the CUDA 13.4 toolkit, the toolkit SGLang's own MiniMax-M3.1 path on Blackwell requires too.

Each cubin is built once per node at load, cached on disk by the digest of its sources, options and compiler. Moving the launchers to NVRTC is left to a follow-up.

## Launch

```bash
CUDA_HOME=/usr/local/cuda-13.4 SGLANG_FORCE_COARSE_WAR_BARRIER=1 \
SGLANG_PLUGINS=nvidia_megakernel \
python -m sglang.launch_server \
  --model-path MiniMaxAI/MiniMax-M3.1-Flash --trust-remote-code \
  --tp 4 \
  --moe-runner-backend flashinfer_trtllm_routed \
  --page-size 128 \
  --speculative-algorithm DSPARK --speculative-use-block-verification \
  --speculative-num-steps 1 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 8 --speculative-dspark-block-size 7 \
  --disable-attn-tp-gather \
  --max-running-requests 16 --cuda-graph-max-bs-decode 16 \
  --mem-fraction-static 0.8 \
  --enable-nvidia-megakernel
```

`SGLANG_PLUGINS=nvidia_megakernel` (SGLang's general-plugin whitelist) is needed where MSA's own SGLang plugin is installed too, as in SGLang's cu134 image: SGLang gives `--enable-nvidia-megakernel` to that plugin for this architecture when it is loaded, and the whitelist keeps it unloaded, so this package's specs serve the flag.

One server serves 1 to 16 concurrent requests: the decode CUDA graphs of batch 1 run slot `mega`, those of batches 2 to 16 the stock executor with `moe_nofill` and `moe_swap`; graphs of larger batches, if captured, run SGLang's forward. MegaAttention keeps all 148 of its CTAs resident: `SGLANG_FORCE_COARSE_WAR_BARRIER=1` (or `--disable-overlap-schedule`) keeps the next batch's preparation kernels from running beside it, and the GPUs must not be shared.

Every slot the package ships (`mega`, `moe_nofill`, `moe_swap` and the draft's `markov`) runs with the flag alone. `SGLANG_NVIDIA_MEGAKERNEL_SLOTS` (comma-separated) replaces that list for tests and bisecting: `mega,moe_nofill,moe_swap` keeps SGLang's draft proposal, `moe_nofill,moe_swap` serves every width bitwise without the routed-expert copy, and an empty value runs SGLang's step code at every width; an unknown slot or `moe_swap` without `moe_nofill` stops the server at startup, naming it. Slot `markov` replaces SGLang's folded DSpark proposal: the server stops at startup with `SGLANG_DSPARK_FOLDED_PROPOSAL=0` or `SGLANG_DSPARK_OPT_INT8_MARKOV_WALK=1`, and at its first step after graph capture if the proposal never asked the draft spec for its sampler. The checks of slot `mega`'s kernels (the CUDA toolkit, the config fields and the context length they are built for, `SGLANG_FORCE_COARSE_WAR_BARRIER`) apply only while it is requested.

## What a covered step runs

One spec covers the DSpark verify steps of 1 to 16 requests, each width with its own slots (`minimax_m31/plan.py`); the startup line names them, e.g. `slots by width 8: mega; 16-128: moe_nofill, moe_swap`:

| Slot | Widths | What it runs | Numerics |
| --- | --- | --- | --- |
| `mega` | 8 rows (one request) | The embedding and the three dense layers run SGLang's modules. Each of the 57 MoE layers is two persistent launches (`minimax_m31/sm103`): MegaAttention (input RMSNorm, the MXFP8 Q/K/V/index projection, head norms and RoPE, the NVFP4 cache writes, the indexer's top-16 pages, the sparse attention, the output projection with its TP4 all-reduce in an NVLink multicast mailbox, the residual add and the post-attention norm), then the token-parallel MoE (the router, the MXFP8 shared expert, the NVFP4 top-4 of 128 routed experts and the TP4 all-reduce). SGLang's final norm and logits processor close the step | tolerance (`minimax_m31/spec.py`) |
| `moe_nofill` | 16 to 128 rows (8 too without `mega`) | The stock executor (SGLang's modules in SGLang's order, never through a model or layer `forward`); each MoE layer's routed experts quantize their per-token NVFP4 input into buffers built once, without the chain's fill launch and allocations | bitwise |
| `moe_swap` | `moe_nofill`'s widths | `moe_nofill`'s MoE with the routed chain on the main stream (the FFN-input boundary kernel's PDL successor) and the shared expert on the alternate stream | bitwise |
| `markov` | draft blocks of 1 to 16 requests | A slot of the DSpark draft, built by its spec (`minimax_m31/draft_step.py`, `sampler`): SGLang's folded proposal, the tail of the draft's CUDA graph, asks the draft spec for its `draft_sampler`. The draft LM head stays SGLang's projection; the 7-step Markov chain of every draft row is one cooperative, vocab-sharded kernel per rank (`sm103/draft/markov.py`): this rank's 16-row head tiles held in registers and shared memory for the whole chain, one 64-bit key exchange per row and step through a symmetric-memory mailbox with a wait deadline, SGLang's draft noise drawn on a side stream under the LM head | tolerance (`minimax_m31/draft.py`): the chain's bias sums in another order than SGLang's GEMM, so a kept step logit can differ by at most two BF16 ulps of \|logit\| + \|base logit\| past one ulp of the step's largest \|bias\|, and a draft token only where two candidates are that close. The target's distribution is kept, but a seeded sampled request can return other tokens than with SGLang's proposal; greedy requests return the target's argmax tokens |

With no slot at a width the stock executor runs SGLang's forward call for call. The DSpark draft's block at one request runs the draft backbone: the embedding, the five dense layers and the final norm in one cooperative launch; larger draft batches run SGLang's draft forward. Either way the proposal that samples the block is slot `markov`'s up to 16 requests (SGLang's above, and at every size without the slot); the draft spec's startup line names it, and each rank logs which proposal it serves.

The kernels read SGLang's weights, KV pages and step data in place, except the routed experts of slot `mega`: the plugin keeps a copy (54.1 GiB a GPU) in its own layout, recovered from the `flashinfer_trtllm_routed` layout SGLang keeps for its own path, built only while `mega` is requested.

Every wait on another GPU (MegaAttention's mailbox, the MoE's exchange, the draft backbone's sums, the Markov chain's key exchange) carries SGLang's build deadline and traps when it expires, so no token from a failed step is returned.

## Tests

The e2e tests run on 4 x GB300 on one host in SGLang's cu134 image (`test/e2e/ci_models.yaml`). `test_minimax_m31_acc.py` and `test_minimax_m31_perf.py` are a matched pair against native SGLang, the same launch without `--enable-nvidia-megakernel`: each launches the baseline, runs the workload and tears it down, then launches the candidate on the same GPUs with the same arguments and environment and runs the same workload. Both fail unless the candidate's log shows the megakernel active on every rank and running its decode graphs, and the baseline's log no megakernel line.

- `test_minimax_m31_acc.py` (accuracy): chat GSM8K, greedy, real speculative acceptance, the same questions one at a time (the fused path) and 16 at a time (the 16-128-row steps), and three pass keys at 13K tokens. With b the questions only the baseline answers correctly and c those only the candidate does, the candidate passes when b - c <= max(2, ceil(2 * sqrt(b + c))); it must also retrieve every pass key the baseline retrieves.
- `test_minimax_m31_perf.py` (performance): `benchmark/static_table.py` at 8K / 1K with `SGLANG_SIMULATE_ACC_LEN=3.95` on both arms (measurement only: it fixes the accepted length), at concurrency 1 and the fallback widths 2 and 16, one untimed pass and then 3 timed passes an arm. At concurrency 1 the candidate's slowest pass must beat the baseline's fastest; at a fallback width the candidate's mean must not be below the baseline's by more than the larger of 3% and the two arms' pass-to-pass spreads.
- `test_minimax_m31_parity.py` (a supplement): with `SGLANG_NVIDIA_MEGAKERNEL_PARITY=1` and eager decode steps, each covered step against SGLang's forward on the same step: slot `mega`'s logits within the spec's tolerances, every other width bitwise.

`test/unit/test_minimax_m31_e2e_verdicts.py` checks the pair's rules on CPU. On a server with CUDA graphs, `SGLANG_NVIDIA_MEGAKERNEL_MARKOV_PARITY=1` runs SGLang's proposal beside the Markov chain inside each captured draft graph and logs, per rank every 30 s, the tokens and kept logits that differ against the declared bound (`NVIDIA megakernel: Markov parity rank N: ...`).
