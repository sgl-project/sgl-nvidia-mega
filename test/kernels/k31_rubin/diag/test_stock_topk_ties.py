# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Repro/controls (no gate on our code): stock topk_transform_paged_v2(deterministic=False) exactness on its own input
scores for tie-heavy rows (index-K from 64 distinct rows). scan4 saw rows 0-1 of an 8-row causal call at 70K inexact.
Controls: repeat the call (determinism), each row alone (1-row call), and the same 8 rows with deterministic=True."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _harness as H  # noqa: E402
import _ref_mla as R  # noqa: E402
import test_k31_dsa_scan as T  # noqa: E402


@pytest.mark.parametrize("ctx", [70000, 131072])
def test_stock_ties(ctx):
    H.load_plugin(T.MUTATIONS)
    from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2, topk_transform_paged_v2

    rows = 8
    g = H.generator(300 + rows)
    pages, cache, table, q8, weights = T.scan_case(ctx, rows, g, distinct=64)
    dg, _ = T.stock_select(q8, weights, cache, table, ctx, rows)
    lens = T.lengths_for(ctx, rows)
    pos = R.slot_positions(table[0], pages)
    tab = table.expand(rows, -1).contiguous()

    def run(scores, ln, tb, stable):
        out = torch.full((scores.shape[0], 2048), -7, dtype=torch.int32, device="cuda")
        raw = torch.empty_like(out)
        topk_transform_paged_v2(scores, ln, tb, out, 64, plan_topk_v2(ln), raw, deterministic=stable)
        torch.cuda.synchronize()
        return out

    res = {}
    for name, fn in (("rep0", lambda: run(dg, lens, tab, False)), ("rep1", lambda: run(dg, lens, tab, False)),
                     ("stable", lambda: run(dg, lens, tab, True))):
        out = fn()
        res[name] = []
        for j in range(rows):
            try:
                T.check_exact(f"{name}.row{j}", out[j], dg[j], int(lens[j]), pos)
                res[name].append("ok")
            except AssertionError as e:
                res[name].append(str(e).split("\n")[0][:120])
    alone = []
    for j in range(rows):
        sc = dg[j:j + 1].contiguous()
        out = run(sc, lens[j:j + 1].contiguous(), table, False)
        try:
            T.check_exact(f"alone.row{j}", out[0], dg[j], int(lens[j]), pos)
            alone.append("ok")
        except AssertionError as e:
            alone.append(str(e).split("\n")[0][:120])
    res["alone"] = alone
    for k, v in res.items():
        print("STOCK_TIES", ctx, k, v, flush=True)
    H.record({"kind": "stock_ties", "ctx": ctx, **res})
