# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""L38 timing (after bench_verify_streams_att.py): the sparse MLA verify layer at Breq x Q8 rows,
6 heads (TP16), per arm ``name:T`` with T = 0 (stock: trtllm-gen 16-head-padded sparse MLA + MlaValue) or the
K31MK_SPARSE_MLA6 core's tokens per split (64 / 128 / 256) + MlaValueFold. All arms use the base-H index-query
schedule (K31MK_INDEX_Q_OFF_WRITER=2). One GPU per process; arms replayed in ABBA blocks on that GPU.

Per layer, as K31VerifyExecutor runs it: a spin (stands in for the previous layer's MoE), the layer-input producer (an
MK split-K GEMM writing the rows; it triggers its dependents early like AttnRes), the index-K writer on the index
stream, the MK MLA input projection on main, PreparedSparseVerifyMla.run. ``--layers`` layers with distinct weights
and pools (cold L2) in one CUDA graph per arm. Kernel timestamps from the torch profiler (CUPTI). Metric: end of the
value kernel (MlaValue / MlaValueFold) - end of the producer, per layer; also the core / top-k ends.
Writes OUT/rows_<tag>.jsonl and OUT/summary_<tag>.json; prints L38SUMMARY.
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

Q = 8
GATE_END = 2112 + 128 * HEADS
KEYS = {  # metric -> kernel-name substrings (first occurrence after the producer, per layer)
    "proj": ("SplitKGemm",),
    "store": ("fused_store_indexer_cache",),
    "index_q": ("index_q_kernel",),
    "logits": ("paged_mqa_logits",),
    "topk": ("topk",),
    "merge": ("MlaFoldMerge",),  # K31MK_FOLD_SPLIT_ROWS arms only
    "core": ("fmhaSm100", "sparse_mla6_kernel"),
    "value": ("MlaValue",),
}


graphs_keep = []


def build(args, rows, ctx):
    H.load_plugin({})
    import deep_gemm
    from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.gemm.projection import PreparedProjection, projection_tactic
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.mla.prepared import PreparedMla
    from sgl_nvidia_mega.kimi_k31.dsa import sparse_mla6 as S6
    from sgl_nvidia_mega.kimi_k31.dsa import verify as V
    from sgl_nvidia_mega.kimi_k31.dsa.indexer import IndexWeights
    from sgl_nvidia_mega.kimi_k31.verify_indexer import IndexKeyWeights, PreparedVerifyIndexKey

    breq = rows // Q
    g = H.generator(args.seed + rows + ctx % 1009)
    tactic = projection_tactic("mla", rows, torch.device("cuda", torch.cuda.current_device()))
    npg = math.ceil(ctx / 64)
    pages = breq * npg + 2
    causal = (ctx - Q + 1 + torch.arange(Q, device="cuda")).repeat(breq).to(torch.int32)
    contexts = torch.full((breq, Q), ctx, dtype=torch.int32, device="cuda")
    step = SimpleNamespace(
        sparse=1, batch_requests=breq, query_width=Q, causal_lengths=causal, score_contexts=contexts,
        score_schedule=deep_gemm.get_paged_mqa_logits_metadata(contexts, 64, num_sms()), topk_plan=plan_topk_v2(causal),
    )
    first = torch.empty((rows, 7168), dtype=torch.bfloat16, device="cuda")
    projected = torch.empty((rows, GATE_END), dtype=torch.bfloat16, device="cuda")
    x_in = H.randn((rows, 1536), 1.0, g)
    lengths = torch.full((breq,), ctx, dtype=torch.int32, device="cuda")
    max_ctx = max(1 << 17, 1 << math.ceil(math.log2(ctx)))
    layers = []
    for i in range(args.layers):
        w = make_weights(args.seed + 100 + i)
        wk = H.randn((128, 7168), 1 / math.sqrt(7168), g)
        producer = PreparedProjection(H.randn((7168, 1536), 1 / math.sqrt(1536), g), {rows: tactic}).prepare()
        projection = PreparedProjection(H.randn((GATE_END, 7168), 0.02, g), {rows: tactic}).prepare()
        table = (torch.randperm(pages - 1, generator=g, device="cuda")[: breq * npg] + 1).to(torch.int32).view(breq, npg)
        pos = torch.arange(ctx - Q, ctx, device="cuda")
        slots = (table[:, pos // 64].long() * 64 + pos % 64).reshape(-1)
        kv0, idx0 = kv_pool(pages, g), index_pool(pages, g)
        attention, _ = fakes(w)
        scratch = torch.empty((rows, HEADS, 576), dtype=torch.uint8, device="cuda")
        base = PreparedMla(attention, {rows: GATE_END}, scratch, max_context=max_ctx, core="trtllm", queries=(Q,))
        arms = {}
        for name, tokens in args.arm_tokens.items():
            split = args.arm_split.get(name, 0)
            early = args.arm_early.get(name, False)
            saved = (V.INDEX_Q_OFF_WRITER, V.SPARSE_MLA6, S6.TOKENS, V.FOLD_SPLIT_ROWS, V.MLAVALUE_FOLD_EARLY)
            V.INDEX_Q_OFF_WRITER, V.SPARSE_MLA6, V.FOLD_SPLIT_ROWS, V.MLAVALUE_FOLD_EARLY = 2, tokens > 0, split, early
            S6.TOKENS = tokens if tokens > 0 else S6.TOKENS
            try:
                core = V.PreparedSparseVerifyMla(base, IndexWeights(w.wq_b, w.w_proj, w.w_qa, 1e-5), rows, max_ctx)
                kv, idx = kv0.clone(), idx0.clone()
                core.prepare_cache(kv, prepare_base=not arms)
                writer = V.SparseVerifyIndexKey(
                    PreparedVerifyIndexKey(IndexKeyWeights(wk, w.gamma, w.beta, w.k_eps)), core
                )
            finally:
                V.INDEX_Q_OFF_WRITER, V.SPARSE_MLA6, S6.TOKENS, V.FOLD_SPLIT_ROWS, V.MLAVALUE_FOLD_EARLY = saved
            assert (core.mla6 is not None) == (tokens > 0), "arm does not run the core it names"
            assert core.mla6 is None or core.mla6.tokens == tokens, "arm does not run the split it names"
            want = "split" if tokens > 0 and split > 0 and rows >= split else None
            got = getattr(core.sparse_values.get((rows, Q)), "kind", None)
            assert got == want, f"arm {name}: value form {got!r}, expected {want!r} (K31MK_FOLD_SPLIT_ROWS {split})"
            writer.bind(idx, 64, rows=(rows,))
            core.step = step
            out = torch.empty((rows, HEADS * 128), dtype=torch.bfloat16, device="cuda")
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
    for name in args.arm_tokens:
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
    # The captured graphs read and write these tensors' addresses: keep them alive with the graphs, or later allocations
    # reuse their blocks and the replays run on clobbered inputs.
    graphs_keep.append((x_in, first, projected, lengths, causal, contexts))
    return graphs, layers, tactic


def kernels(trace_path):
    d = json.load(gzip.open(trace_path) if str(trace_path).endswith(".gz") else open(trace_path))
    ev = [e for e in d["traceEvents"] if e.get("cat") == "kernel"]
    return sorted(ev, key=lambda e: e["ts"])


def per_layer(ev):
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
        for key, subs in KEYS.items():
            hit = [e for e in rest if any(sub in e["name"] for sub in subs)]
            if hit:
                e = hit[0]
                row[key + "_end"] = e["ts"] + e["dur"] - t0
                row[key + "_start"] = e["ts"] - t0
                row[key + "_dur"] = e["dur"]
                row[key + "_name"] = e["name"][:60]
                row[key + "_stream"] = e.get("args", {}).get("stream")
        rows.append(row)
    return rows


def run_config(args, rows, ctx, out, gpu):
    from torch.profiler import ProfilerActivity, profile

    graphs, layers, tactic = build(args, rows, ctx)
    names = list(args.arm_tokens)
    for name in names:
        for _ in range(5):
            graphs[name].replay()
    torch.cuda.synchronize()
    # Output agreement between arms on the last layer (recorded; the test file gates correctness).
    ref = layers[-1].arms[names[0]].out.float()
    agree = {}
    for name in names[1:]:
        o = layers[-1].arms[name].out.float()
        agree[name] = float(((o - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-30)).max())
    order = args.order.replace("A", names[0]).split(",") if "," in args.order else None
    order = order or [names[0]] + names[1:] + names[1:][::-1] + [names[0]]
    order = order * args.cycles
    tag = f"r{rows}_c{ctx}_{gpu}"
    all_rows = []
    for block, name in enumerate(order):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(args.reps):
                graphs[name].replay()
            torch.cuda.synchronize()
        path = out / f"trace_{tag}_{block}{name}.json"
        prof.export_chrome_trace(str(path))
        ev = kernels(path)
        spins = [i for i, e in enumerate(ev) if "spin" in e["name"].lower() or "sleep" in e["name"].lower()]
        assert len(spins) == args.reps * args.layers, f"found {len(spins)} spin kernels for {args.reps} x {args.layers}"
        for r in range(args.reps):
            lo = spins[r * args.layers]
            hi = spins[(r + 1) * args.layers] if r + 1 < args.reps else len(ev)
            for row in per_layer(ev[lo:hi]):
                row.update(arm=name, block=block, rep=r, gpu=gpu, rows=rows, ctx=ctx)
                all_rows.append(row)
        if os.environ.get("L38_KEEP_TRACES") != "1":
            os.remove(path)
    with open(out / f"rows_{tag}.jsonl", "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")
    summary = {"rows": rows, "ctx": ctx, "gpu": gpu, "tactic": tactic, "order": order, "agree_max_relL2": agree}
    for name in names:
        sel = [r for r in all_rows if r["arm"] == name and r["rep"] >= 2]  # first replays of a block dropped
        blocks = sorted({r["block"] for r in sel})
        bmed = [st.median([r["value_end"] for r in sel if r["block"] == b and "value_end" in r]) for b in blocks]
        summary[name] = {
            k: st.median([r[k] for r in sel if k in r])
            for k in sorted({k for r in sel for k in r if k.endswith(("_end", "_start", "_dur"))})
        }
        summary[name].update(n=len(sel), value_end_block_medians=bmed, spread=max(bmed) - min(bmed),
                             core_name=next((r.get("core_name") for r in sel if "core_name" in r), None),
                             # the core and the value kernel must share a stream (the fold's PDL edge to the core)
                             core_value_same_stream=all(r.get("core_stream") == r.get("value_stream") for r in sel))
    base = names[0]
    for name in names[1:]:
        delta = summary[name]["value_end"] - summary[base]["value_end"]
        spread = max(summary[name]["spread"], summary[base]["spread"])
        summary[name]["delta_value_end"] = delta
        summary[name]["resolved"] = abs(delta) > 2 * spread
    (out / f"summary_{tag}.json").write_text(json.dumps(summary, indent=1))
    print("L38SUMMARY", json.dumps(summary), flush=True)
    del graphs, layers
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--rows", default="8,16")
    ap.add_argument("--contexts", default="8704,131072,598016")
    ap.add_argument("--arms", default="B:0,L:128,L64:64,L80:80,L96:96,L256:256",
                    help="name:tokens per split (0 = stock core)[:K31MK_FOLD_SPLIT_ROWS[:K31MK_MLAVALUE_FOLD_EARLY]]")
    ap.add_argument("--order", default="", help="comma list of arm names; default A B.. ..B A (ABBA)")
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--spacer-cycles", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    args.arm_tokens = {a.split(":")[0]: int(a.split(":")[1]) for a in args.arms.split(",")}
    # optional third field: K31MK_FOLD_SPLIT_ROWS for that arm (e.g. S:128:16); default 0 = MlaValueFold
    args.arm_split = {a.split(":")[0]: int(a.split(":")[2]) for a in args.arms.split(",") if a.count(":") >= 2}
    # optional fourth field: K31MK_MLAVALUE_FOLD_EARLY for that arm, 0 / 1 / f (e.g. Sf:128:16:f); default 0
    args.arm_early = {a.split(":")[0]: {"0": False, "1": True, "f": "fence"}[a.split(":")[3]]
                      for a in args.arms.split(",") if a.count(":") == 3}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    H.hold_fp8_clock()
    for ctx in [int(c) for c in args.contexts.split(",")]:
        for rows in [int(r) for r in args.rows.split(",")]:
            run_config(args, rows, ctx, out, gpu)


if __name__ == "__main__":
    main()
