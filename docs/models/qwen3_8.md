# Qwen3.8-Flash-Next (NVFP4)

| GPU | Tensor parallel | Coverage |
| --- | --- | --- |
| Rubin (SM107) | 1; 4 with every op on SGLang's module | Target verify and decode steps and MTP draft-extend steps of 1 to 1024 rows in CUDA graphs, prefill and extend eagerly |

## What a covered step runs

The Qwen3.8 executor (`qwen3_8/runtime`) owns the forward of the target and of its MTP
draft: target verify, decode, draft extend, and prefill / extend. It runs the layer loop
itself. Per layer, that is the HC read, the mixer (QSA attention or GatedDeltaNet), the HC
write-back, and the MoE. SGLang keeps the scheduling, the speculative orchestration, the
caches, the sampling and the pools. The executor reads the running forward's metadata and
pools through SGLang's EAGLE hybrid binding (`binding="eagle_hybrid"`).

Each op runs its slot's kernel where that kernel serves the step's width, and SGLang's own
module everywhere else. The log records which implementation each op ran, once per new set
for each (mode, width): `NVIDIA megakernel engagement: ...`.

Every slot below runs by default, so `--enable-nvidia-megakernel` is the only setting the
model needs. A slot whose kernel cannot serve a setup (for example `qsa_pool_fp8` needs an
FP8 KV pool) leaves that step to SGLang's module and logs the reason. At TP1 the executor
runs each layer's HC read, mixer, HC write-back and MoE itself (the S1 layers, which `hc`,
`hc_wide`, `moe_finalize` and the PLE add need); at larger TP the layers go through SGLang's
stage boundaries.

`SGLANG_NVIDIA_MEGAKERNEL_SLOTS` (comma-separated) replaces the shipped list, for tests and
bisecting. An empty value runs every op on SGLang's module.

| Slot | Runs | Numerics against SGLang's module |
| --- | --- | --- |
| `hc` | HC write-backs of 1 to 32 rows; with `SGLANG_HC_MACRO` (on by default) at TP1, the HC reads the executor makes itself at 9 to 32 rows (the PLE layer's, and every layer's with the BF16 pin table) on the source's row-aware split-K tactics | rounding |
| `hc_wide` | HC reads and write-backs of 33 to 1024 rows; at 33 to 320 rows, a write-back and the next read in one launch | rounding |
| `gdn_native_pdl` | the GDN target verify on the circular ReplaySSM rings, as a PDL launch | within SGLang's own run-to-run variation |
| `qsa_indexer` | the QSA indexer's fold, scorer and top-k in verify, decode and draft extend | the same blocks, except on ties |
| `qsa_pool` | TP1 QSA sparse attention of up to 64 verify / decode rows on split-K pool kernels | rounding |
| `qsa_pool_fp8` | TP1 QSA sparse attention of 56 to 576 target-verify rows on an FP8 E4M3 KV pool: the new K/V cast in the pool write, each request's token union, the warp-specialized attention (PDL) and the merge | rounding |
| `moe_finalize` | the TP1 MoE finalize with the gated shared expert in one ordered kernel; from 1024 rows (prefill and extend, in breakable prefill graphs too) the extend-size stream order and a register-capped PDL launch, plus a FlashInfer tuning pass of the routed experts at prefill sizes that applies the prefill table of `SGLANG_FLASHINFER_TACTIC_PIN_FILE` | rounding |
| `gdn_prefill` | the GDN prefill on FlashInfer's chunked kernel: split conv, retiled gating, l2norm and norm, one-launch state copies | bitwise |
| `qsa_prefill` | QSA prefill: row-group union attention, row pack, fast top-k select, TMA prefill | union: rounding; the rest bitwise |
| `ple` | the PLE target-verify layout, host-table gather, gate and short conv | bitwise |

Switches:

- **The GDN u-cache target verify** (a deferred-fold ReplaySSM) is chosen from the server
  arguments; there is no switch.
  - It runs when its needs hold: `--mamba-ssm-dtype float16`, a linear NEXTN / EAGLE chain of
    1 to 8 draft tokens, SM107, and no `--enable-linear-replayssm-spec`. Without
    `--disable-radix-cache` it also needs `--mamba-radix-cache-strategy extra_buffer` or
    `extra_buffer_lazy`.
  - The default configuration (BF16 state, ReplaySSM) does not meet them and is unchanged.
  - The startup log says which verify runs and, when the u-cache is not used, every reason.
  - The executor owns the rings, keyed by request row with an fp16 key ring. It allocates
    them in the spec's `prepare`, resets them, and commits each verify, through SGLang's
    `SupportsStateSlotReset` and `SupportsRecurrentCommit`.
  - The published FP16 high-concurrency recipes use it. FP16 state with Triton verify and no
    u-cache has no recipe and has not been tested.
- **`SGLANG_NVIDIA_MEGAKERNEL_PARITY=layer|control`** is test only. It runs SGLang's forward
  beside the executor on every covered step, restoring the recurrent state between the two,
  and logs the steps, layers and logits that differ (`qwen3_8/tools/parity.py`).
- **`SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_OPCHECK=1`** is test only. On eager steps it also runs
  SGLang's attention on the same q / k / v and selected indices, and logs how far the pool
  kernel's output is from it, per row.
