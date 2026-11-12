# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K31MK_VERIFY_TOPK=x47960: VerifySelection's <= 16-row path (vendored #47960 exact mode, dsa/x47960) is an exact top-2048.

Run as a script on one GPU (exit code 0 = pass):  K31MK_VERIFY_TOPK=x47960 python test_k31_verify_topk_x47960.py --inp DIR
DIR holds real_400k_rows.npy / real_400k_lens.npy (real K3.1 index logits, 393K keys; not included).

Exactness rule (independent reference, float64 torch.topk): as test_k31_verify_topk_gvr.py (k = min(2048, n), every key > v_k
selected, none < v_k, k distinct raw indices in [0, n) decoded through the page table, slots >= k are -1; columns >= n hold 1e4).
A: the served call shape (per-row causal lengths P + j + 1, per-row page table, stride-0 row table at 1 request) at 8 / 16 rows on
   real 8K-393K rows, randn and integer-tie rows, a non-contiguous request table; the stock topk_v2 module is loaded and run in the
   same process first (the two JIT modules must coexist).
B: CUDA-graph capture after one eager call, replayed on new scores and new lengths.
C: 64 rows route to gvr (stack-r7) and are exact.
D: routing table (hand-derived): rows <= 16 -> x47; > 16 -> gvr2 (r7).
Must-fail controls (the run fails if any passes): (1) stock topk_transform_paged_v2 on real 393K rows (known inexact);
(2) the same vendored module in mode 0 (fast; inexact on real 262K / 393K rows at 8 / 16 rows); (3) a causal-length
mutant: every row of a request given the request's last-row length (rows j < q - 1 then reach poisoned columns).
"""
import argparse
import os
import sys

os.environ.setdefault("K31MK_VERIFY_TOPK", "x47960")
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_k31_verify_topk_gvr import K, PAGE, DEV, check, fill, make_rows, step_of, tables  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--width", type=int, default=1 << 20)
    a = ap.parse_args()
    import numpy as np
    from sgl_nvidia_mega.kimi_k31.dsa import verify as V

    assert V.VERIFY_TOPK == "x47960", V.VERIFY_TOPK
    fails, results = [], []
    dev = torch.device(DEV)
    # D: routing table
    s8, s16, s64 = V.VerifySelection(dev, 8), V.VerifySelection(dev, 16), V.VerifySelection(dev, 64)
    want = {8: (True, None), 16: (True, None), 64: (False, "gvr2")}
    for rows, sel in ((8, s8), (16, s16), (64, s64)):
        got = (sel.x47 is not None, sel.gvr.__name__.split(".")[-2] if sel.gvr is not None else None)
        results.append(("route", rows, got))
        if got != want[rows]:
            fails.append(f"route rows {rows}: got {got}, want {want[rows]}")
    # Print routing failures now: a mis-routed selector has x47 None, and the A / B cases below then skip it (recorded as a
    # failure) instead of crashing; printing here keeps the 'FAIL route' line even if a later step raises.
    for f in fails:
        print("FAIL", f, flush=True)
    printed = len(fails)

    inp = {"r400": np.load(f"{a.inp}/real_400k_rows.npy", mmap_mode="r"), "l400": np.load(f"{a.inp}/real_400k_lens.npy")}
    W = a.width
    pages = W // PAGE
    from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2, topk_transform_paged_v2

    def stock(scores, lens, table, q):
        cl = torch.tensor(lens, dtype=torch.int32, device=DEV)
        out = torch.full((len(lens), K), -7, dtype=torch.int32, device=DEV)
        b = table.shape[0]
        rt = table.expand(q, pages) if b == 1 else table.repeat_interleave(q, dim=0)
        topk_transform_paged_v2(scores, cl, rt, out, PAGE, plan_topk_v2(cl), deterministic=False)
        torch.cuda.synchronize()
        return check(out, scores, lens, q, table.contiguous())

    # control 1 (stock, loaded first): real 393K rows
    b, q, L = 1, 8, 393216
    table = tables(b, pages)
    scores = torch.empty(b * q, W, dtype=torch.float32, device=DEV)
    lens, rows = make_rows("real", b, q, L, inp, seed=11)
    fill(scores, lens, rows)
    nb_stock = stock(scores, lens, table, q)
    results.append(("stock_v2_control", b, q, L, nb_stock))
    if nb_stock == 0:
        fails.append("CONTROL NOT CAUGHT: stock topk_v2 exact on the real 393K rows (expected >= 1 bad row)")
    # A: the served call shape
    cases = [  # (b, q, last_len, kind, contiguous request table)
        (1, 8, 8200, "real", True), (1, 8, 32768, "real", True), (1, 8, 131072, "real", True), (1, 8, 262144, "real", True),
        (1, 8, 393216, "real", True), (1, 8, 131072, "randn", True), (1, 8, 131072, "tie1", True), (1, 8, 393216, "tie1", True),
        (2, 8, 131072, "real", False), (2, 8, 262144, "real", True), (2, 8, 393216, "real", True), (2, 8, 32768, "tie1", True),
    ]
    for (b, q, L, kind, contig) in cases:
        sel = s8 if b * q == 8 else s16
        if sel.x47 is None:
            fails.append(f"x47 b{b} q{q} L{L} {kind}: selector has no x47 module (mis-routed)")
            continue
        table = tables(b, pages, contig)
        scores = torch.empty(b * q, W, dtype=torch.float32, device=DEV)
        lens, rows = make_rows(kind, b, q, L, inp, seed=11)
        fill(scores, lens, rows)
        st = step_of(lens, b, q)
        sel.selected.fill_(-7)
        out = sel.x47.run(scores, st.causal_lengths, sel.row_table(table, q), sel.selected)
        torch.cuda.synchronize()
        nb = check(out, scores, lens, q, table.contiguous())
        results.append((b, q, L, kind, contig, nb, "x47"))
        if nb:
            fails.append(f"x47 b{b} q{q} L{L} {kind}: {nb} bad rows")
    # B: CUDA graph (eager first, then capture; replay on new scores and new lengths)
    for (b, q, L, L2) in [(1, 8, 393216, 131072), (2, 8, 131072, 262144)]:
        sel = s8 if b * q == 8 else s16
        if sel.x47 is None:
            fails.append(f"x47 graph b{b} q{q}: selector has no x47 module (mis-routed)")
            continue
        table = tables(b, pages)
        scores = torch.empty(b * q, W, dtype=torch.float32, device=DEV)
        lens, rows = make_rows("real", b, q, L, inp, seed=11)
        fill(scores, lens, rows)
        st = step_of(lens, b, q)
        rt = sel.row_table(table, q)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            sel.x47.run(scores, st.causal_lengths, rt, sel.selected)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            sel.x47.run(scores, st.causal_lengths, rt, sel.selected)
        lens2, rows2 = make_rows("real", b, q, L2, inp, seed=29)
        fill(scores, lens2, rows2)
        st.causal_lengths.copy_(torch.tensor(lens2, dtype=torch.int32, device=DEV))
        if b > 1:
            rt.view(b, q, pages).copy_(table.unsqueeze(1).expand(b, q, pages))
        sel.selected.fill_(-7)
        g.replay()
        torch.cuda.synchronize()
        nb = check(sel.selected, scores, lens2, q, table)
        results.append((b, q, f"{L}->{L2}", "real_graph", True, nb, "x47"))
        if nb:
            fails.append(f"x47 graph b{b} q{q} L{L}->{L2}: {nb} bad rows")
    # C: 64 rows -> gvr (stack-r7)
    b, q, L = 8, 8, 131072
    table = tables(b, pages)
    scores = torch.empty(b * q, W, dtype=torch.float32, device=DEV)
    lens, rows = make_rows("real", b, q, L, inp, seed=11)
    fill(scores, lens, rows)
    s64.selected.fill_(-7)
    out = s64.gvr_select(scores, step_of(lens, b, q), table)
    torch.cuda.synchronize()
    nb = check(out, scores, lens, q, table)
    results.append((b, q, L, "real", True, nb, s64.gvr.__name__))
    if nb:
        fails.append(f"gvr 64 rows L{L}: {nb} bad rows")
    # control 2: mode 0 (fast) of the same vendored module on real 262K / 393K rows at 8 / 16 rows
    from sgl_nvidia_mega.kimi_k31.dsa.x47960 import X47960Select

    m0_bad = 0
    for (b, q, L) in [(1, 8, 262144), (1, 8, 393216), (2, 8, 262144), (2, 8, 393216)]:
        m0 = X47960Select(dev, b * q, mode=0)
        table = tables(b, pages)
        scores = torch.empty(b * q, W, dtype=torch.float32, device=DEV)
        lens, rows = make_rows("real", b, q, L, inp, seed=11)
        fill(scores, lens, rows)
        out = torch.full((b * q, K), -7, dtype=torch.int32, device=DEV)
        rt = table.expand(q, pages) if b == 1 else table.repeat_interleave(q, dim=0)
        m0.run(scores, torch.tensor(lens, dtype=torch.int32, device=DEV), rt, out)
        torch.cuda.synchronize()
        m0_bad += check(out, scores, lens, q, table)
    results.append(("mode0_control", m0_bad))
    if m0_bad == 0:
        fails.append("CONTROL NOT CAUGHT: mode 0 (fast) exact on the real 262K / 393K rows")
    # control 3: causal-length mutant (every row given its request's last-row length)
    b, q, L = 1, 8, 131072
    table = tables(b, pages)
    scores = torch.empty(b * q, W, dtype=torch.float32, device=DEV)
    lens, rows = make_rows("real", b, q, L, inp, seed=11)
    fill(scores, lens, rows)
    mut = torch.full((b * q,), max(lens), dtype=torch.int32, device=DEV)
    s8.selected.fill_(-7)
    s8.x47.run(scores, mut, s8.row_table(table, q), s8.selected)
    torch.cuda.synchronize()
    nb_mut = check(s8.selected, scores, lens, q, table)
    results.append(("len_mutant", b, q, L, nb_mut))
    if nb_mut == 0:
        fails.append("CONTROL NOT CAUGHT: last-row-length mutant passed")
    for r in results:
        print("RESULT", r, flush=True)
    for f in fails[printed:]:
        print("FAIL", f, flush=True)
    print("VERIFY_TOPK_X47960_TEST", "PASS" if not fails else "FAIL", flush=True)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
