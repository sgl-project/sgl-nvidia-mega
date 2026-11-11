<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

# Benchmarks

## Static decode table (switch on vs off)

`static_table.py` measures tokens per second per user at each batch size: ISL 8K, OSL 1K,
batch 1–64, one fresh server launch per arm. The two arms differ only by
`--enable-nvidia-megakernel` (every shipped slot; no `SGLANG_NVIDIA_MEGAKERNEL_SLOTS`).

Server (GB300, TP4, DSpark). Both arms run with the measured launch's environment:
the Engram prefetch needs SGLang's per-rank Engram host table (without the two host-table
variables the plugin is refused at startup with that reason). SGLang places the tables
only through its generic process NUMA membind (`SGLANG_NUMA_BIND_V2`), with no per-table
placement; the numbers were measured on nodes gated for free huge pages, with the page
cache evicted and the huge pages warmed before each launch. The TP sync setting was part
of every measured launch, and its share of the numbers was not measured separately.

```
SGLANG_SPEC_TP_SYNC=rng,-dspark-accept-sample \
SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank \
SGLANG_SIMULATE_ACC_LEN=3.51 SGLANG_SIMULATE_ACC_METHOD=match-expected \
SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token \
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V4.1-Flash --trust-remote-code \
  --tp 4 --ep-size 1 --kv-cache-dtype fp8_e4m3 --dsv4-attn-backend flashmla --mem-fraction-static 0.8 \
  --speculative-algorithm DSPARK --speculative-dspark-block-size 5 \
  --cuda-graph-max-bs-decode 64 --max-running-requests 256 --chunked-prefill-size 8192 \
  --enable-decoder-swa-bounded-replay \
  [--enable-nvidia-megakernel]
```

The three `SGLANG_SIMULATE_ACC_*` variables fix the acceptance length at 3.51, so both arms
run the same number of verify steps; never set them for an accuracy run. With DSpark
block 5 a verify step has 6 rows per request. The on arm runs the plugin's step at every
captured batch: batch 1–8 (6–48 rows) with the routed-expert GEMMs, and batch 10–64
(60–384 rows, 19 captured widths) with the two-shot attention and MoE posts (`plan.py`:
two-shot from 49 rows up to `MAX_ROWS`, 384). The off arm runs SGLang's standard path at
every batch.

Client, two passes per launch:

```
python benchmark/static_table.py --base-url http://127.0.0.1:30000 --salt r1 --out on-1
python benchmark/static_table.py --base-url http://127.0.0.1:30000 --salt r2 --reverse --out on-1
```

Report the mean of the two passes per batch.

Protocol:
- Both arms on one node.
- At least 2 fresh launches per arm, in palindrome order (on, off, off, on).
- Diff the two launch commands and environments before reading a number.
- Check the startup line `NVIDIA megakernel: active (DeepSeek-V4.1: …; widths [6, 12, …, 384]; …)` on every rank in the switch-on arm, and its absence in the switch-off arm.
- A difference inside the run-to-run band (≈ 1.5 % at batch 1 and 4, ≈ 1 % elsewhere) is not a result.

## Accuracy

`test/e2e/test_deepseek_v41_server.py` runs GSM8K-200 past the largest covered width, KL
of covered steps against the normal prefill path, and a passkey past the indexer's
candidate span. With `SGLANG_NVIDIA_MEGAKERNEL_E2E_URL` set it tests a running server, so the
same checks run against the switch-off arm; judge the switch-on numbers against that
arm's own band.

## GLM-5.2 (GLM DSA) decode A/B

The GLM rows run at the InferenceX AgentX point configurations (`tp4-416`:
aggregated TP4, EAGLE 4/1/5, HiCache; `tp8-516`: aggregated TP8 on two trays,
EAGLE 5/1/6, HiCache; `tp4-516`: the disaggregated decode role, TP4, EAGLE 5/1/6,
run standalone). Both arms set
`SGLANG_ENABLE_EMBED_REPLICATION=1` and differ only by `--enable-nvidia-megakernel`.
Acceptance is pinned at the golden length (3.33 at EAGLE 4/1/5, 3.61 at 5/1/6), so both
arms run the same verify steps:

```
SGLANG_ENABLE_EMBED_REPLICATION=1 \
SGLANG_SIMULATE_ACC_LEN=3.61 SGLANG_SIMULATE_ACC_METHOD=match-expected \
SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token \
python3 -m sglang.launch_server --model-path nvidia/GLM-5.2-NVFP4 --trust-remote-code \
  --quantization modelopt_fp4 --tp 4 --fp4-gemm-backend flashinfer_cutedsl \
  --speculative-algorithm EAGLE --speculative-eagle-topk 1 \
  --speculative-num-steps 5 --speculative-num-draft-tokens 6 \
  --max-running-requests 24 --cuda-graph-max-bs-decode 24 --mem-fraction-static 0.9 \
  [--enable-nvidia-megakernel]
```

Client: `python3 -m sglang.benchmark.serving --dataset-name random --random-input-len 8192
--random-output-len 1024 --random-range-ratio 1 --max-concurrency C --num-prompts 8C
--temperature 1 --top-p 0.95` (the AgentX sampling). Read the server's decode step at
batch C (`SGLANG_RECORD_STEP_TIME=1`, `/server_info` `step_time_dict`). Protocol as
above: palindrome order (off, on, on, off), two launches per arm, two passes each.

## Served A/B from step stamps (benchmark only)

`ab_step_times.py` resolves step-time differences of a few microseconds on a served
model, where the static table's client-side gaps cannot (routing follows the generated
tokens, so the MoE halves alone move by up to ±110 µs per step between identical runs).
Two benchmark-only switches of `runtime/bench.py`, both off by default:

- `SGLANG_NVIDIA_MEGAKERNEL_BENCH_FIXED_ROUTING=synth[:d=25,half0=13,...]` or
  `table:<file.npz>`: every covered MoE layer (target and draft, fast-gate and hash
  layers) routes to a fixed expert table, so two arms do the same expert GEMMs every
  step; everything downstream runs unchanged on the real weights. Outputs are wrong:
  the server refuses it without `SGLANG_SIMULATE_ACC_LEN`, and the accuracy test
  refuses to run with it set.
- `SGLANG_NVIDIA_MEGAKERNEL_BENCH_STEP_TIMES=<dir>`: a one-CTA kernel forked at the start
  of every covered verify and draft forward stamps `%globaltimer` and the position into a
  pinned ring (plus, on rank 0, each MoE layer's distinct-expert count of the previous
  replay); a host thread drains it to `<dir>`.

```
# login node, inside an allocation (one node for tp4, two trays for tp8: --node 0,1)
python3 benchmark/ab_step_times.py run --job <jobid> --topology tp4 --node 0 \
    --image <sqsh> --model <model dir> --src <this checkout> --out <dir> \
    --arm off "" --arm pf "AB_SLOTS_ADD=prologue_fork" --launches 2 --passes 4
python3 benchmark/ab_step_times.py analyze <dir>          # numpy; --fixed off runs use the covariate
```

Each launch: `ab_serve.sh` (fresh server, forced acceptance 3.51, batch 1,
`SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION=1` unless `--runtime-opt off`) and
`ab_client.sh` (a warm-up pass, then static_table.py passes). `analyze` takes per request
the median verify-step period (after dropping host-stall steps and slow-episode
requests, both counted), a launch as the mean over its requests, and reports B - A with
a pooled two-sample t over launch means; with real routing (`--fixed off`) it first
regresses the period on the step's distinct-expert count within each launch. Pass and
step intervals are printed for information only: launches differ by a few microseconds
(GB300 TP4, fixed routing: about 2.6 us sd per launch mean, against 0.3-0.8 us between
passes of one launch), so the launch is the unit. The client's step p50 per launch is
printed alongside.

The stamps cost about 18 us per step (client step p50 with them on vs off, 2 + 2
launches, 95 % [12, 24]), the same in both arms; an A/B that changes the boundary between
the plugin's forward and SGLang's code may interact with it, so check such a result on
the client's p50 too (`--no-stamps`).
