# GLM-5.3, GLM-5.2 (NVFP4)

| GPU | Tensor parallel | Coverage |
| --- | --- | --- |
| B300 / GB300 (SM103), Rubin (SM107) | 8 (one node, or two trays of one NVLink domain); 4 (one GB300 tray or one Rubin node) | TP8: MTP target verify steps of 4 to 16 rows (batch 1 to 4 at 4 rows a request, batch 1 to 2 at 6); TP4: verify steps of up to 48 rows at 4 to 6 rows a request (batch 1 to 8 at 6); the NextN draft step in its captured graphs |
| Rubin (SM107), GLM-5.2 | 16 (four trays of one NVLink domain) | One request, EAGLE 7 steps / topk 1 / 8 draft tokens: the 75-layer decoder chain, dense layers 0-2, the fused NextN draft and the fused verify tail. See [TP16 on Rubin](#tp16-on-rubin) |

Larger verify steps, prefill and extend run SGLang's standard path, and the NextN draft step runs on its draft spec.

GLM-5.2 has GLM-5.3's architecture and the same value of every config field the
GLM-5.3 specs pin (`HF_CONFIG` in `glm5_3/__init__.py`), so the same specs serve it,
and what this page says of GLM-5.3 holds for GLM-5.2. On `nvidia/GLM-5.2-NVFP4`
(8 x B300) the megakernel engages, a test-only step-by-step comparison with SGLang's
own target verify and draft steps shows no difference, and GSM8K-200 (strict match)
scores 0.975 with the megakernel and 0.960 without it.

## Requirements

- An SGLang with GLM-5.3's DSA binding view (`ForwardView.dsa` in `sglang/srt/kernel_plugins/nvidia/binding.py`); without it the spec refuses at startup. Slot `draft` also needs `SGLANG_ENABLE_EMBED_REPLICATION=1`, set by the user: SGLang's `FLAG_SETS_ENV` (`sglang/kernels/plugin_kernels/nvidia/spec.py`) sets only the variables of a spec's `Requirements.env`, and this spec checks its environment per TP size instead
- 8 x B300, or 8 GPUs over two trays of one NVLink domain (GB300 NVL72, Rubin VR NVL72), or 4 x GB300 or 4 x Rubin (TP4, one tray or node)

## Launch

GLM-5.3 or GLM-5.2 (NVFP4) on 8 x B300, with an NVFP4 GLM-5.3 or GLM-5.2 checkpoint (for example `nvidia/GLM-5.2-NVFP4`) whose config has the shapes in `glm5_3/__init__.py` (`HF_CONFIG`; the spec checks them at startup):

```bash
SGLANG_ENABLE_EMBED_REPLICATION=1 \
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

Every GLM-5.3 slot runs with `--enable-nvidia-megakernel` and `SGLANG_ENABLE_EMBED_REPLICATION=1`; no slot list is needed. The GLM-5.3 NextN draft step (slot `draft`) reads the replicated embedding table, so it needs `SGLANG_ENABLE_EMBED_REPLICATION=1`; the flag does not set it (the spec checks its environment per TP size, so SGLang's resolution step does not set the variable for it), and without it the server stops at startup, naming it. The draft step also needs greedy draft acceptance (no `--speculative-use-rejection-sampling`, SGLang's default); without it the server stops at startup, naming it. SGLang's defaults give the rest of the environment the spec needs; a setup it does not serve stops at startup, naming each reason. Serving with HiCache (`--enable-hierarchical-cache`) is supported: the covered steps read only the running requests' device KV pages. At TP4 so is the decode role of a PD-disaggregated deployment (`--disaggregation-mode decode`): the KV arrives through the transfer backend before a request's first verify step, outside the covered step; the prefill role runs without the flag.

On two trays (GB300 NVL72, Rubin VR NVL72), launch the same command on each tray with `--nnodes 2 --node-rank <0|1> --dist-init-addr <tray-0 host>:<port>`. The covered steps need SGLang's FlashInfer all-reduce + RMSNorm fusion at verify sizes, which needs NVLink multicast; on nodes without it, set `SGLANG_NVIDIA_MEGAKERNEL_GLM_ALLOW_NO_AR_FUSION=1`: the covered steps then run anyway, the stock residual steps on SGLang's unfused all-reduce and slot `hops` on its own symmetric-memory all-reduce.

The GLM-5.3 megakernel serves MTP with 3 draft steps (`--speculative-num-steps 3 --speculative-num-draft-tokens 4`, as above) or 5 (`--speculative-num-steps 5 --speculative-num-draft-tokens 6`) at TP8, and 2 to 5 draft steps at TP4 (`--tp-size 4`); any other draft length stops the server at startup, naming the served values.

At TP8 the GLM-5.3 megakernel covers verify steps of up to 16 rows: 1 to 4 requests with 4 draft tokens, 1 or 2 with 6 (at TP4 up to 48 rows); larger steps run SGLang's path while its derived weight copies (about 20 GiB per GPU, logged at startup) still take KV-cache memory, so it helps little on high-concurrency servers, and the server warns when `--max-running-requests` is set above 8.

## What a covered step runs

GLM-5.3 (`glm5_3/`) follows the same layout: the executor's layer loop runs the hops between the sublayers (`boundary/`), the attention (`attention/`) and the MoE (`moe/`), then the logits (`logits/`), each with a stock path that reproduces SGLang's verify step bit for bit, and CuTe DSL kernels (`sm103/`) where a slot serves its width:

| Slot | Rows | Kernel |
| --- | --- | --- |
| Attention (`attention`) | 4–16 | The q_a / kv_a projection on a small-M GEMM (with the indexer's key / head-gate projection, up to 8 rows), the MLA prologue (norms, q_b, W_UK absorption, RoPE and FP8 query, the MLA and index-K cache rows), the indexer query, the sparse MLA over the fused top-k page table and the value up-projection; the indexer's top-k stays SGLang's |
| Routed and shared experts (`moe`) | 4–16 | One small-M NVFP4 engine launch for the routed experts and the BF16 shared expert, for 1, 2 or 3–4 groups of up to 4 rows (a request's rows with 4 draft tokens; 6 rows run as 4 + 2, 12 as 4 + 4 + 4) |
| All-reduce + residual + RMSNorm (`hops`) | 4–16 | One hop per tensor-parallel sum over symmetric memory: `o_proj` with the router on MoE layers, the MoE finalize folded into the MLP-output all-reduce, and the final norm |
| Shared-top-k q_a / kv_a (`qkv_a_small_m_shared`) | 4–8 | The small-M projection on the layers that reuse the previous layer's top-k (one packed weight copy each) |
| Deferred input norm (`dnorm`) | 4–8 | Those layers' input RMSNorm applied in the projection's epilogue from the fold hop's block sums (moves the rounding points; its tolerances are in `glm5_3/spec.py`) |
| NextN draft step (`draft`) | every captured draft-decode and draft-extend graph | The draft's embedding gather + eh norms in one kernel on the replicated table (any row count), the column-sharded `eh_proj` (1–4 rows, and 8, 12 and 16 with 4 draft tokens or 6 and 12 with 6; gathered by the hops' all-reduce on the target's mailbox), the distributed top-1 in place of the logits all-gather (up to 8 rows), and the draft layer's MLA prologue (up to 16 rows); each kernel's other row counts run SGLang's code |
| NextN draft-extend row select (`draft_rowselect`) | every captured draft-extend graph | After the draft layer's attention (which writes every row's KV and indexer key), the MLP, its all-reduce and the final norm run on the rows the draft's LM head reads (`select_index`, one per request) instead of every verify row; the kept rows are the full-row stream's bit for bit, the draft MoE then picks its kernels by the smaller row count (tolerances in `glm5_3/draft.py`) |

At TP4 (one GB300 tray or one Rubin node, 16 attention heads per rank) the verify step's groups are composed per width from kernel units (`glm5_3/ll/`; `glm5_3/units.py` orders them per step, `glm5_3/registry.py` composes them, `glm5_3/runtime/composed.py` runs them in the executor's layer loop): the sparse MLA as flash decoding over the top-k rows (`ll.sparse_mla_fd`, up to 7 rows a wave), the attention front on the low-latency fused-A GEMM and a one-launch MLA KV half (`ll.attention`), the small-M BF16 GEMMs on PDL split-K and TGV (`ll.gemm.<role>`), the all-reduce + residual + RMSNorm hops on a plugin-owned Lamport buffer with the MoE finalize folded in (`ll.hops`), FlashInfer's TRT-LLM FP4 MoE with that deferred finalize (`ll.moe`), and in the NextN draft the row-parallel `eh_proj` summed in the target's hop (`ll.draft_eh`). Each step runs the first unit that covers its width, SGLang's step body elsewhere; a width no unit covers runs SGLang's forward, and the startup log names why ("stock at N rows: ..."). Each unit is a slot (`runtime.LL_SLOTS`); `ll.moe.alt_stream_priority` runs only when listed. A unit that covers a width but fails to build stops the server on every rank, naming its slots.

The units' plans (GEMM ranges, hop caps, row caps) were measured on GB300 (`glm5_3/ll/target.py`). On Rubin (SM107) the units run the same kernels on plans derived from GB300's, which live in `glm5_3/sm107/tp4.py`; `glm5_3/targets.py` picks one of the two from the GPU's architecture when the build's profile is resolved, and `glm5_3/ll/` itself never checks the architecture. On Rubin the cluster geometry and the sparse MLA's row caps follow the device's SM count and occupancy, queried at build (a width Rubin cannot launch in one wave runs TRTLLM-Gen), split-K tactics are re-checked for one wave of 212 SMs, and the router logits keep SGLang's GEMM. Each unit logs once that its plan is derived.

The TP4 split-K GEMMs (`ll.gemm`) run FlashInfer's split-K kernel (`mm_bf16(..., tactic=)`) when the installed FlashInfer has it with the synchronization fixes (`SplitKTactic.SYNC_FIXED`); otherwise, and on Rubin (SM107), where FlashInfer's explicit tactics do not run, they run the plugin's vendored copy (`ll/kernels/cute/splitk.py`). The startup log names the source. `SGLANG_NVIDIA_MEGAKERNEL_GLM_GEMM_SPLITK` switches it: `auto` (the default), `flashinfer` (FlashInfer's kernel even without the fixes; the server stops if FlashInfer has no such API) or `vendored`.

With the megakernel on, all shipped slots run; none needs to be listed. `SGLANG_NVIDIA_MEGAKERNEL_SLOTS` (comma-separated) overrides that list for tests and bisecting only; a list without a slot another needs (`dnorm` needs `qkv_a_small_m_shared` and `hops`, `qkv_a_small_m_shared` needs `attention`, `draft_rowselect` needs `draft`) stops the server at startup, naming it. Slot `draft` needs `SGLANG_ENABLE_EMBED_REPLICATION=1` (set it yourself; the flag does not), greedy draft acceptance and an SGLang whose EAGLE draft graph runners run a draft spec's capture check; the server stops at startup, naming each one that is missing.

## TP16 on Rubin

At TP16 on Rubin (VR200, SM107, four trays), one request, EAGLE 7 steps / topk 1 /
8 draft tokens, the same flag runs `sgl_nvidia_mega/glm5_3/sm107` for GLM-5.2: the
75-layer decoder chain, dense layers 0-2, the fused NextN draft and the fused verify
tail, launched from prebuilt cubins. The GLM-5.3 specs select it by the TP size, and
refuse TP16 on any other GPU. `--enable-nvidia-megakernel` is the only switch. It needs
`SGLANG_NVIDIA_MEGAKERNEL_GLM_SM107_CUBINS` set to the cubin directory
(`{chain,d012,draft,tail}/{entry.cubin,CALL.json}`), `SGLANG_DSA_FUSE_TOPK=0`,
`--kv-cache-dtype fp8_e4m3 --dsa-decode-backend trtllm
--moe-runner-backend flashinfer_trtllm --speculative-algorithm EAGLE
--speculative-num-steps 7 --speculative-eagle-topk 1 --speculative-num-draft-tokens 8`;
a setup that lacks any of them stops the server at startup, naming each reason.
It serves one request at a time. A target verify step it does not cover (more than one
request, grammar, logprobs, penalties, logit bias, a sampling seed, top-k / min-p
sampling) raises instead of falling back (`sm107/verify_tail.py`), so serve it at
concurrency 1 with greedy or temperature / top-p sampling.

## Tests

GLM-5.3 GPU and end-to-end suites are not yet part of the initial CI workflow.
The GLM-5.3 CPU config and SGLang contract checks are included in `test/unit`.
