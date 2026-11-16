# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K31MK_VERIFY_TOPK=x47960 above 16 rows: VerifySelection.gvr_select (vendored FlashInfer gvr_2, next_n form, paged output) is an
exact top-2048 at the served 24-64-row shapes (3-8 requests x Q8).

Run as a script on one GPU (exit code 0 = pass):  K31MK_VERIFY_TOPK=x47960 python test_k31_verify_topk_gvr.py --inp DIR
DIR holds real_400k_rows.npy / real_400k_lens.npy (real K3.1 index logits, 393K keys; not included).

Exactness rule (independent reference, float64 torch.topk): for row r with n keys, k = min(2048, n), v_k = k-th largest of row[:n].
Pass iff slots [0, k) hold k distinct raw indices in [0, n) (decoded through the page table), slots [k, 2048) are -1, every key
> v_k is selected and no selected key is < v_k (ties at v_k: any members). Columns >= n hold 1e4, so any read past n is caught.
Must-fail controls (the run fails if either passes): (1) stock topk_transform_paged_v2 (deterministic=False, the served kernel)
on real 393K rows: known inexact there; (2) the next_n contract mutant: kv_len taken from each request's FIRST
row instead of its last (shifts every row's length by up to 7 keys into the poisoned columns).
Also: CUDA-graph capture after one eager call, replayed on new scores; a non-contiguous request page table.
"""
import argparse
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("K31MK_VERIFY_TOPK", "x47960")
import numpy as np
import torch

K, PAGE, DEV = 2048, 64, "cuda"


def make_rows(kind, b, q, last_len, inp, seed):
    """Per-row lengths P + j + 1 (P = last_len - q) and score rows of that length."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    lens, rows = [], []
    for r in range(b):
        for j in range(q):
            n = last_len - q + j + 1
            if kind == "real":
                s = (r * 5 + j * 3 + seed) % inp["r400"].shape[0]
                src = np.array(inp["r400"][s, : int(inp["l400"][s])])
                x = torch.from_numpy(src[:n].copy()) if n <= src.shape[0] else None
                if x is None:
                    raise ValueError(f"real row too short for n={n}")
                x = x.to(DEV)
            else:
                x = torch.randn(n, generator=g, device=DEV)
                if kind == "tie1":
                    x = torch.floor(x)
            lens.append(n)
            rows.append(x)
    return lens, rows


def fill(scores, lens, rows):
    scores.fill_(1.0e4)
    for i, (n, x) in enumerate(zip(lens, rows)):
        scores[i, :n] = x


def check(out, scores, lens, q, table):
    """Number of bad rows under the exactness rule."""
    bad = 0
    pages = table.shape[1]
    for i, n in enumerate(lens):
        kk = min(K, n)
        o = out[i]
        sel, tail = o[:kk].long(), o[kk:]
        if (tail != -1).any() or (sel < 0).any():
            bad += 1
            continue
        req = i // q
        page, off = sel // PAGE, sel % PAGE
        # invert the request's page table (pages are unique per request in this test)
        inv = torch.full((int(table.max()) + 1,), -1, dtype=torch.long, device=DEV)
        inv[table[req].long()] = torch.arange(pages, device=DEV)
        lp = inv[page]
        raw = lp * PAGE + off
        if (lp < 0).any() or int(raw.min()) < 0 or int(raw.max()) >= n or torch.unique(raw).numel() != kk:
            bad += 1
            continue
        x = scores[i, :n].double()
        kth = torch.topk(x, kk).values[-1]
        xs = x[raw]
        if int((xs < kth).sum()) or int((x > kth).sum()) != int((xs > kth).sum()):
            bad += 1
    return bad


def step_of(lens, b, q):
    return SimpleNamespace(batch_requests=b, query_width=q, causal_lengths=torch.tensor(lens, dtype=torch.int32, device=DEV))


def tables(b, pages, contiguous=True):
    # request r owns pages [r * pages, (r + 1) * pages) in a shuffled order (decoding must go through the table)
    perm = torch.randperm(b * pages, generator=torch.Generator().manual_seed(7)).to(torch.int32).view(b, pages).to(DEV)
    if contiguous:
        return perm.contiguous()
    wide = torch.zeros(b, pages + 16, dtype=torch.int32, device=DEV)
    wide[:, :pages] = perm
    return wide[:, :pages]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--width", type=int, default=1 << 20)
    a = ap.parse_args()
    from sgl_nvidia_mega.kimi_k31.dsa import verify as V

    assert V.VERIFY_TOPK == "x47960", V.VERIFY_TOPK
    print("KNOB", V.VERIFY_TOPK, flush=True)
    inp = {"r400": np.load(f"{a.inp}/real_400k_rows.npy", mmap_mode="r"), "l400": np.load(f"{a.inp}/real_400k_lens.npy")}
    W = a.width
    pages = W // PAGE
    fails = []
    results = []
    cases = [  # (b, q, last_len, kind, contiguous table); gvr_2 serves > 16 rows (3-8 requests x Q8)
        (3, 8, 8200, "real", True), (3, 8, 32768, "real", True), (3, 8, 131072, "real", True), (3, 8, 262144, "real", True),
        (3, 8, 131072, "randn", True), (3, 8, 131072, "tie1", True), (3, 8, 393216, "tie1", True), (4, 8, 131072, "real", True),
        (3, 8, 131072, "real", False), (8, 8, 131072, "real", True), (8, 8, 262144, "real", True), (8, 8, 32768, "tie1", True),
    ]
    for (b, q, L, kind, contig) in cases:
        sel = V.VerifySelection(torch.device(DEV), b * q)
        table = tables(b, pages, contig)
        scores = torch.empty(b * q, W, dtype=torch.float32, device=DEV)
        lens, rows = make_rows(kind, b, q, L, inp, seed=11)
        fill(scores, lens, rows)
        st = step_of(lens, b, q)
        sel.selected.fill_(-7)
        out = sel.gvr_select(scores, st, table)
        torch.cuda.synchronize()
        nb = check(out, scores, lens, q, table.contiguous())
        results.append((b, q, L, kind, contig, nb, sel.gvr.__name__))
        if nb:
            fails.append(f"gvr b{b} q{q} L{L} {kind}: {nb} bad rows")
    # CUDA graph: one eager call (above, same geometry) then capture and replay on new scores
    for (b, q, L, kind) in [(3, 8, 262144, "real"), (8, 8, 131072, "real")]:
        sel = V.VerifySelection(torch.device(DEV), b * q)
        table = tables(b, pages)
        scores = torch.empty(b * q, W, dtype=torch.float32, device=DEV)
        lens, rows = make_rows(kind, b, q, L, inp, seed=11)
        fill(scores, lens, rows)
        st = step_of(lens, b, q)
        sel.gvr_select(scores, st, table)
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            sel.gvr_select(scores, st, table)  # eager on the capture stream (gvr_2's workspace slab is per stream)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            sel.gvr_select(scores, st, table)
        lens2, rows2 = make_rows(kind, b, q, L, inp, seed=29)
        fill(scores, lens2, rows2)
        st.causal_lengths.copy_(torch.tensor(lens2, dtype=torch.int32, device=DEV))
        sel.selected.fill_(-7)
        g.replay()
        torch.cuda.synchronize()
        nb = check(sel.selected, scores, lens2, q, table)
        results.append((b, q, L, kind + "_graph", True, nb, sel.gvr.__name__))
        if nb:
            fails.append(f"gvr graph b{b} q{q} L{L} {kind}: {nb} bad rows")
    # must-fail control 1: stock topk_v2 on real 393K rows
    from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2, topk_transform_paged_v2

    b, q, L = 1, 8, 393216
    table = tables(b, pages)
    scores = torch.empty(b * q, W, dtype=torch.float32, device=DEV)
    lens, rows = make_rows("real", b, q, L, inp, seed=11)
    fill(scores, lens, rows)
    cl = torch.tensor(lens, dtype=torch.int32, device=DEV)
    out = torch.full((b * q, K), -7, dtype=torch.int32, device=DEV)
    plan = plan_topk_v2(cl)
    topk_transform_paged_v2(scores, cl, table.expand(q, pages), out, PAGE, plan, deterministic=False)
    torch.cuda.synchronize()
    nb_stock = check(out, scores, lens, q, table)
    results.append(("stock_v2_control", b, q, L, nb_stock))
    if nb_stock == 0:
        fails.append("CONTROL NOT CAUGHT: stock topk_v2 exact on the real 393K rows (expected >= 1 bad row)")
    # must-fail control 2: kv_len from the first row of each request (contract mutant), at 3 requests (24 rows)
    b, q, L = 3, 8, 262144
    table = tables(b, pages)
    scores = torch.empty(b * q, W, dtype=torch.float32, device=DEV)
    lens, rows = make_rows("real", b, q, L, inp, seed=11)
    fill(scores, lens, rows)
    cl = torch.tensor(lens, dtype=torch.int32, device=DEV)
    sel = V.VerifySelection(torch.device(DEV), b * q)
    kv_bad = cl.view(b, q)[:, 0].contiguous()
    sel.selected.fill_(-7)
    sel.gvr.run_varlen(scores, None, kv_bad, sel.selected, next_n=q, top_k=K, max_seq_len=W, page_table=table, page_size=PAGE)
    torch.cuda.synchronize()
    nb_mut = check(sel.selected, scores, lens, q, table)
    results.append(("kv_first_row_mutant", b, q, L, nb_mut, sel.gvr.__name__))
    if nb_mut == 0:
        fails.append("CONTROL NOT CAUGHT: kv_len-from-first-row mutant passed")
    for r in results:
        print("RESULT", r, flush=True)
    for f in fails:
        print("FAIL", f, flush=True)
    print("VERIFY_TOPK_GVR_TEST", "PASS" if not fails else "FAIL", flush=True)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
