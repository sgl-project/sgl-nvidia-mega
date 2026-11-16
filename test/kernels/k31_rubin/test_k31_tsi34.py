# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K31MK_TSI_WIDE: TSI at Breq 2-4 (16 / 24 / 32 verify rows) with a SUM context gate (kimi_k31/dsa/verify.py) and the TSIM
serving selection at 3 and 4 requests.

Gate (CPU; expected values written from the spec in TSI34.md s4, not from the code): knob off = today's rule (rows 8 / 16, every
request >= K31MK_TSI_MIN_CONTEXT); knob on: one request keeps that rule; 2-4 requests (rows 16 / 24 / 32 at query width 8) go TSI
iff sum(contexts) >= K31MK_TSI_SUM_CONTEXT (default 393,216); 5+ requests never.

Selection (one GPU, the served TSI env: K31MK_TSI_MERGE=tsim + S1R; 16 ranks emulated as in test_k31_tsim_tsi.py, whose layer
procedure this file reuses with a flag check that also admits data-dependent fallback rows): 3 and 4 requests per step, mixed
contexts, including the padded 4th request of a Breq 3 step replayed in the capacity-4 graph (DSA graph fill seq_len 1 -> context
1 + Q = 9 tokens, causal 2..9: 15 of 16 ranks hold no local token of it), a request below 2048 tokens, the serving table width
(16,400 pages), a planted certificate failure on request 2, and a tie-heavy case. Reference: numpy over the FULL DCP1 DeepGEMM
logits (score desc, position asc), source-major slot order, mapped through the page table (test_k31_tsim_tsi.reference;
independent of the TSI / TSIM code). Exact, no tolerance.

Mutations (K31_MUTATION=<name>; each must FAIL this file):
  wide_gate_max   the sum gate reads the longest request          -> test_gate_wide
  wide_gate_b1    the sum gate also applies to one request        -> test_gate_wide
  wide_rows24     rows 24 (Breq 3) left out of TSI_WIDE_ROWS      -> test_gate_wide
  req_clamp       TSIM emission maps rows of requests >= 2 through request 1's page table -> test_select_wide (b3pad, b4mix)
  no_fallback     TSIM mutation 4 (emits from the fast path where the certificate failed) -> test_select_planted
"""

from __future__ import annotations

import os

os.environ["K31MK_TSI_MERGE"] = "tsim"  # read by tsi.py at import (one module build per boot)
os.environ.setdefault("K31MK_TSI_S1R", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import _harness as H  # noqa: E402
import _hook_variants as V  # noqa: E402

V_FILE = "kimi_k31/dsa/verify.py"
M_FILE = "kimi_k31/dsa/tsim.cuh"
MUTATIONS = {
    "wide_gate_max": (V_FILE, "int(seq_lens.sum()) >= TSI_SUM_CONTEXT", "int(seq_lens.max()) >= TSI_SUM_CONTEXT"),
    "wide_gate_b1": (V_FILE, "if TSI_WIDE and seq_lens.numel() >= 2:", "if TSI_WIDE and seq_lens.numel() >= 1:"),
    "wide_rows24": (V_FILE, "TSI_WIDE_ROWS = (16, 24, 32)", "TSI_WIDE_ROWS = (16, 32)"),
    "req_clamp": (
        M_FILE,
        "  // ---- KV-slot gathers for the keys that can be emitted (consumed at emission only) ----\n"
        "  const int32_t* const trow = p.table + static_cast<int64_t>(row / p.rows_per_request) * p.table_stride;",
        "  // ---- KV-slot gathers for the keys that can be emitted (consumed at emission only) ----\n"
        "  const int32_t* const trow = p.table + static_cast<int64_t>(min(row / p.rows_per_request, 1)) * p.table_stride;",
    ),
    "no_fallback": V.edits(M_FILE, 4),
}
H.load_plugin(MUTATIONS)  # before the imports below (they call load_plugin too; the first call wins)

import test_k31_tsi as TT  # noqa: E402
import test_k31_tsim_tsi as S  # noqa: E402

needs_gpu = S.needs_gpu
W, PAGE, TOPK, Q = S.W, S.PAGE, S.TOPK, S.Q
PAD = 1 + Q  # the padded request's context: DSA graph seq_len fill 1 (dsa_backend get_cuda_graph_seq_len_fill_value) + Q


def _verify():
    from sgl_nvidia_mega.kimi_k31.dsa import verify as V

    return V


# ------------------------------------------------------------------------------------------------ gate (CPU)
def test_gate_wide(monkeypatch):
    V = _verify()
    t = lambda *ctx: torch.tensor(ctx, dtype=torch.int64)  # noqa: E731
    assert V.TSI_SUM_CONTEXT == 393216, "default sum gate"
    monkeypatch.setattr(V, "TSI", True)
    monkeypatch.setattr(V, "TSI_MIN_CONTEXT", 262144)  # the served min gate (K31MK_TSI_MIN_CONTEXT=262144)
    # knob off: today's rule
    monkeypatch.setattr(V, "TSI_WIDE", False)
    assert V.tsi_graph_label(t(262144, 262144), 8) and not V.tsi_graph_label(t(700000, 10000), 8), "off: rows 16 min rule"
    assert not V.tsi_graph_label(t(400000, 400000, 400000), 8), "off: rows 24 never"
    assert not V.tsi_graph_label(t(400000, 400000, 400000, 400000), 8), "off: rows 32 never"
    # knob on
    monkeypatch.setattr(V, "TSI_WIDE", True)
    assert V.tsi_graph_label(t(262144), 8) and not V.tsi_graph_label(t(262143), 8), "on: Breq 1 keeps the min rule"
    # sum gate below the min gate: only a sum rule applied at one request would admit 200000
    monkeypatch.setattr(V, "TSI_SUM_CONTEXT", 100000)
    assert not V.tsi_graph_label(t(200000), 8), "on: Breq 1 never uses the sum rule"
    monkeypatch.setattr(V, "TSI_SUM_CONTEXT", 393216)
    assert V.tsi_graph_label(t(200000, 193216), 8), "on: Breq 2 sum == gate (min below the min gate)"
    assert not V.tsi_graph_label(t(200000, 193215), 8), "on: Breq 2 sum one below the gate"
    assert V.tsi_graph_label(t(131072, 131072, 131072), 8), "on: Breq 3 sum == gate"
    assert not V.tsi_graph_label(t(131072, 131072, 131071), 8), "on: Breq 3 sum one below"
    assert V.tsi_graph_label(t(380000, 4000, 4000, 5216), 8), "on: Breq 4 sum == gate, skewed"
    assert not V.tsi_graph_label(t(98304, 98304, 98304, 98303), 8), "on: Breq 4 sum one below"
    assert not V.tsi_graph_label(t(*([400000] * 5)), 8), "on: Breq 5 (rows 40) never"
    assert not V.tsi_graph_label(t(*([400000] * 8)), 8), "on: Breq 8 (rows 64) never"
    assert not V.tsi_graph_label(None, 8) and not V.tsi_graph_label(t(), 8)
    monkeypatch.setattr(V, "TSI", False)
    assert not V.tsi_graph_label(t(400000, 400000, 400000), 8), "TSI off: never"


# ------------------------------------------------------------------------------------------------ selection (GPU)
def layer(tag, states, case, ordinal, R, *, prepare, planted=None, any_flags=False):
    """= test_k31_tsim_tsi.layer (emulate 16 ranks, erase R's records, run R's serving select, compare with the reference,
    check R rewrote exactly its records); ``any_flags``: the emulation may set fallback flags on any row (R must match them)."""
    e = S.emu()
    if prepare:
        for st in states:
            st.prepare(case.table, case.contexts, case.causal)
    case.new_query()
    ref = S.reference(case, S.full_logits(case))
    ep = states[0].comm.epoch
    assert all(torch.equal(st.comm.epoch, ep) for st in states), "ranks' epochs differ"
    scores = [st.scan(case.query, case.gates, case.cache) for st in states]
    ncols = states[0].width * PAGE
    X = torch.stack([s[:, :ncols] for s in scores]).contiguous()
    LC = torch.stack([st.lcausal for st in states]).contiguous()
    for t in e.regions:
        t.zero_()
    e.epoch.copy_(ep)
    rows = case.rows
    out16 = torch.full((W, rows, TOPK), S.POISON, dtype=torch.int32, device=S.DEV)
    fb16 = torch.full((W, rows), -9, dtype=torch.int32, device=S.DEV)
    scr16 = torch.zeros((W, rows, W * TOPK), dtype=torch.int64, device=S.DEV)
    e.select(ordinal, X, LC, case.causal, case.table, Q, out16, fb16, scr16, ncols)
    torch.cuda.synchronize()
    o16, f16 = out16.cpu().numpy().astype(np.int64), fb16.cpu().numpy()
    for r in range(W):
        S.first_diff(f"{tag}: EMULATION rank {r}", o16[r], ref)
    assert not (f16 & 8).any(), f"{tag}: a poll hit its deadline in the emulation"
    if planted is not None:
        want = np.array([1 if (row // Q) in planted else 0 for row in range(rows)])
        assert ((f16 != 0) == want[None, :]).all(), f"{tag}: planted flags not as designed: {f16[0].tolist()}"
    elif not any_flags:
        assert (f16 == 0).all(), f"{tag}: emulation flags {sorted(set(f16.reshape(-1).tolist()))}: fast path expected"
    snap = [t.clone() for t in e.regions]
    parity = ordinal & 1
    for t in e.regions:
        S.cand_view(t)[parity, :rows, R].zero_()
    S.full_view(e.regions[R])[parity, :rows].zero_()
    st = states[R]
    out = torch.full((rows, TOPK), S.POISON, dtype=torch.int32, device=S.DEV)
    st._fallback.fill_(7)
    got = st.select(ordinal, case.query, case.gates, case.cache, out)
    torch.cuda.synchronize()
    assert got.data_ptr() == out.data_ptr()
    S.first_diff(f"{tag}: SERVING rank {R}", out.cpu().numpy().astype(np.int64), ref)
    fb = st.fallback.cpu().numpy()
    assert np.array_equal(fb, f16[R]), f"{tag}: rank {R} flags {fb.tolist()} want {f16[R].tolist()}"
    for i, (t, s) in enumerate(zip(e.regions, snap)):
        if not torch.equal(t, s):
            d = (t != s).nonzero()
            raise AssertionError(f"{tag}: region {i} differs from the emulated state after rank {R}'s launch at "
                                 f"{int(d.shape[0])} int64 elements, first flat index {int(d[0, 0])}")
    nfb = int((f16[R] != 0).sum())
    H.record({"kind": "check", "item": tag, "rank": R, "rows": rows, "ncols": ncols, "cells": int(rows * TOPK),
              "violations": 0, "fallback_rows": nfb, "epoch": int(ep[0])})
    print("TSI34", tag, "rows", rows, "fallback_rows", nfb, flush=True)


SELECT = {  # name: (contexts, table width or None, seed)
    "b3pad": ([131072, 300000, 70000, PAD], None, 31),  # Breq 3 in the capacity-4 graph: 32 rows, the 4th padded
    "b3": ([300000, 131072, 70000], None, 32),  # Breq 3 in its own graph (L3): 24 rows
    "b4mix": ([600000, 131072, 8192, 2047], None, 33),  # one request below 2048 tokens
    "b4serve": ([393216, 262144, 131072, PAD], 16400, 34),  # serving width (context 1,048,576): tsim_kernel<16> at 32 rows
}


@needs_gpu
@pytest.mark.parametrize("name", list(SELECT))
def test_select_wide(name):
    """Forward 1: ordinals 0 (rank 0) and 1 (rank 7); forward 2: ordinal 0 under a new epoch (rank 15)."""
    ctx, width, seed = SELECT[name]
    case = S.Case(ctx, seed=seed, width=width)
    states = S.ranks(case)
    for fwd, ordinal, R in ((0, 0, 0), (0, 1, 7), (1, 0, 15)):
        layer(f"{name}_f{fwd}_o{ordinal}", states, case, ordinal, R, prepare=ordinal == 0)


@needs_gpu
def test_select_planted():
    """Request 2's rank-3 pages carry 8x index-K scales: every rank takes the exact fallback on request 2's rows only."""
    case = S.Case([300000, 131072, 180017, 70000], seed=29, boost=3, boost_requests=(2,))
    states = S.ranks(case)
    layer("planted_q2_R3", states, case, 0, 3, prepare=True, planted=(2,))
    layer("planted_q2_R12", states, case, 1, 12, prepare=False, planted=(2,))


@needs_gpu
def test_select_ties():
    """Index-K rows from 37 templates (test_k31_tsi.Case tie=True): each row's cut splits a group of equal scores; the slots must
    follow (score desc, position asc). Fallback rows may occur (data-dependent); R must match the emulation's flags."""
    case = TT.Case([131072, 70000, 300000], seed=37, tie=True)
    states = S.ranks(case)
    layer("ties_b3_R4", states, case, 0, 4, prepare=True, any_flags=True)
