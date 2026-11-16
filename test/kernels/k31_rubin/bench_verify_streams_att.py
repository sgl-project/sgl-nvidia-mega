# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Timing (after bench_l4_verify_streams.py): the sparse MLA verify layer (Breq 1 x Q8,
K31_TEST_HEADS heads) per arm ``name:MODE[p]``: MODE = K31MK_INDEX_Q_OFF_WRITER (0/1/2), suffix ``p`` = the index-K
LayerNorm launched with PDL (K31MK_IDXK_LN_PDL=1). One GPU per process.

Per layer, as K31VerifyExecutor runs it: a ~30 us spin (stands in for the previous layer's MoE), the layer-input
producer (an MK split-K GEMM writing the 8 x 7168 rows; it triggers its dependents early like AttnRes), the index-K
writer on the index stream, the MK MLA input projection (SM107 tactic) on main, PreparedSparseVerifyMla.run.
``--layers`` layers with distinct weights and pools (cold L2) in one CUDA graph per arm; arms replayed in ABBA blocks.
Kernel timestamps come from the torch profiler (CUPTI), the same source as the round traces. Metric:
end of index_q - end of the producer; also logits / top-k / FMHA / value ends and the projection / store ends.
Writes OUT/rows_<gpu>.jsonl (one row per layer per replay) and OUT/summary_<gpu>.json (per-arm medians).
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import statistics as st
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as H  # noqa: E402
from test_k31_mla import HEADS, fakes, index_pool, kv_pool, make_weights, num_sms  # noqa: E402

ROWS, Q = 8, 8
GATE_END = 2112 + 128 * HEADS
KEYS = {  # metric -> kernel-name substring (first occurrence after the producer, per layer)
    "proj": "SplitKGemm",
    "ln": "layernorm",
    "store": "fused_store_indexer_cache",
    "index_q": "index_q_kernel",
    "logits": "paged_mqa_logits",
    "topk": "topk",
    "fmha": "fmhaSm100",
    "value": "MlaValue",
}


def build(args):
    H.load_plugin({})
    import deep_gemm
    from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.gemm.projection import PreparedProjection, projection_tactic
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.mla.prepared import PreparedMla
    from sgl_nvidia_mega.kimi_k31.dsa import verify as V
    from sgl_nvidia_mega.kimi_k31.dsa.indexer import IndexWeights
    from sgl_nvidia_mega.kimi_k31.verify_indexer import IndexKeyWeights, PreparedVerifyIndexKey

    dev = torch.device("cuda", torch.cuda.current_device())
    ctx = args.context
    g = H.generator(args.seed)
    tactic = projection_tactic("mla", ROWS, dev)
    npg = math.ceil(ctx / 64)
    pages = npg + 2
    causal = (ctx - Q + 1 + torch.arange(Q, device="cuda")).to(torch.int32)
    contexts = torch.full((1, Q), ctx, dtype=torch.int32, device="cuda")
    step = SimpleNamespace(
        sparse=1, batch_requests=1, query_width=Q, causal_lengths=causal, score_contexts=contexts,
        score_schedule=deep_gemm.get_paged_mqa_logits_metadata(contexts, 64, num_sms()), topk_plan=plan_topk_v2(causal),
    )
    first = torch.empty((ROWS, 7168), dtype=torch.bfloat16, device="cuda")
    projected = torch.empty((ROWS, GATE_END), dtype=torch.bfloat16, device="cuda")
    x_in = H.randn((ROWS, 1536), 1.0, g)
    lengths = torch.full((1,), ctx, dtype=torch.int32, device="cuda")
    layers = []
    for i in range(args.layers):
        w = make_weights(args.seed + 100 + i)
        wk = H.randn((128, 7168), 1 / math.sqrt(7168), g)
        producer = PreparedProjection(H.randn((7168, 1536), 1 / math.sqrt(1536), g), {ROWS: tactic}).prepare()
        projection = PreparedProjection(H.randn((GATE_END, 7168), 0.02, g), {ROWS: tactic}).prepare()
        table = (torch.randperm(pages - 1, generator=g, device="cuda")[:npg] + 1).to(torch.int32)[None]
        pos = torch.arange(ctx - Q, ctx, device="cuda")
        slots = table[0, pos // 64].long() * 64 + pos % 64
        kv0, idx0 = kv_pool(pages, g), index_pool(pages, g)
        attention, _ = fakes(w)
        scratch = torch.empty((ROWS, HEADS, 576), dtype=torch.uint8, device="cuda")
        base = PreparedMla(attention, {ROWS: GATE_END}, scratch, max_context=1 << 17, core="trtllm", queries=(Q,))
        arms = {}
        for name, spec in args.arm_modes.items():
            mode, ln_pdl = int(spec.rstrip("p")), spec.endswith("p")
            default, default_ln = V.INDEX_Q_OFF_WRITER, V.IDXK_LN_PDL
            V.INDEX_Q_OFF_WRITER, V.IDXK_LN_PDL = mode, ln_pdl
            try:
                core = V.PreparedSparseVerifyMla(base, IndexWeights(w.wq_b, w.w_proj, w.w_qa, 1e-5), ROWS, 1 << 17)
                kv, idx = kv0.clone(), idx0.clone()
                core.prepare_cache(kv, prepare_base=not arms)
                writer = V.SparseVerifyIndexKey(
                    PreparedVerifyIndexKey(IndexKeyWeights(wk, w.gamma, w.beta, w.k_eps)), core
                )
            finally:
                V.INDEX_Q_OFF_WRITER, V.IDXK_LN_PDL = default, default_ln
            assert core.index_q_mode == mode, "arm does not run the path it names"
            assert (writer.key is not None) == ln_pdl, "arm does not run the LayerNorm path it names"
            writer.bind(idx, 64, rows=(ROWS,))
            core.step = step
            out = torch.empty((ROWS, HEADS * 128), dtype=torch.bfloat16, device="cuda")
            arms[name] = SimpleNamespace(core=core, writer=writer, kv=kv, out=out)
        layers.append(SimpleNamespace(producer=producer, projection=projection, table=table, slots=slots, arms=arms))

    def forward(name):
        for L in layers:
            torch.cuda._sleep(args.spacer_cycles)
            L.producer.run(x_in, first)
            a = L.arms[name]
            a.writer.run(first, L.slots)
            L.projection.run(first, projected)
            a.core.run(projected, a.out, cache=a.kv, slots=L.slots, block_table=L.table, lengths=lengths, q=Q)

    graphs = {}
    for name in args.arm_modes:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            forward(name)
            forward(name)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            forward(name)
        graphs[name] = gr
    return graphs, tactic


def kernels(trace_path):
    d = json.load(gzip.open(trace_path) if str(trace_path).endswith(".gz") else open(trace_path))
    ev = [e for e in d["traceEvents"] if e.get("cat") == "kernel"]
    return sorted(ev, key=lambda e: e["ts"])


def per_layer(ev, nlayers):
    """Split one replay's kernels at the spin kernels; metrics relative to the producer's end."""
    spins = [i for i, e in enumerate(ev) if "spin" in e["name"].lower() or "sleep" in e["name"].lower()]
    rows = []
    for j, s in enumerate(spins):
        stop = spins[j + 1] if j + 1 < len(spins) else len(ev)
        seg = ev[s + 1 : stop]
        gemms = [e for e in seg if "SplitKGemm" in e["name"]]
        if len(gemms) < 2:
            continue
        prod = gemms[0]
        t0 = prod["ts"] + prod["dur"]
        rest = [e for e in seg if e is not prod]
        row = {"layer": j, "producer_dur": prod["dur"]}
        for key, sub in KEYS.items():
            hit = [e for e in rest if sub in e["name"]]
            if hit:
                e = hit[0]
                row[key + "_end"] = e["ts"] + e["dur"] - t0
                row[key + "_start"] = e["ts"] - t0
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--context", type=int, default=8704)
    ap.add_argument("--order", default="CDDC")
    ap.add_argument("--arms", default="C:2,D:2p", help="arm name:K31MK_INDEX_Q_OFF_WRITER mode[p = LayerNorm PDL]")
    ap.add_argument("--spacer-cycles", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    args.arm_modes = {a.split(":")[0]: a.split(":")[1] for a in args.arms.split(",")}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    H.hold_fp8_clock()
    graphs, tactic = build(args)
    from torch.profiler import ProfilerActivity, profile

    for name in args.arm_modes:  # warm replays
        for _ in range(5):
            graphs[name].replay()
    torch.cuda.synchronize()
    all_rows = []
    for block, name in enumerate(args.order):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(args.reps):
                graphs[name].replay()
            torch.cuda.synchronize()
        path = out / f"trace_{gpu}_{block}{name}.json"
        prof.export_chrome_trace(str(path))
        ev = kernels(path)
        spins = [i for i, e in enumerate(ev) if "spin" in e["name"].lower() or "sleep" in e["name"].lower()]
        per_rep = len(spins) // args.reps
        assert per_rep == args.layers, f"found {len(spins)} spin kernels for {args.reps} x {args.layers} layers"
        for r in range(args.reps):
            lo = spins[r * args.layers]
            hi = spins[(r + 1) * args.layers] if r + 1 < args.reps else len(ev)
            for row in per_layer(ev[lo:hi], args.layers):
                row.update(arm=name, block=block, rep=r, gpu=gpu)
                all_rows.append(row)
        os.remove(path) if os.environ.get("ATT_KEEP_TRACES") != "1" else None
    with open(out / f"rows_{gpu}.jsonl", "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")
    summary = {"gpu": gpu, "tactic": tactic, "device": H.device_facts(), "args": vars(args)}
    for name in args.arm_modes:
        rows = [r for r in all_rows if r["arm"] == name and r["rep"] >= 2]  # first replays of a block dropped
        summary[name] = {
            k: st.median([r[k] for r in rows if k in r]) for k in sorted({k for r in rows for k in r if k.endswith(("_end", "_start"))})
        }
        summary[name]["n"] = len(rows)
    (out / f"summary_{gpu}.json").write_text(json.dumps(summary, indent=1))
    print("ATTSUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
