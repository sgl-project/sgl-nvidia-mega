# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""TSI (K31MK_TSI, TSIM merge) on MXFP4 index-K pages (K31MK_DSA_IDX_FP4=1), for the toggle setting in K31_TEST_IDX_Q
(fp8 = FP8 query x MXFP4 index-K, the SA form; fp4 = MXFP4 query), on ONE GPU: test_k31_tsim_tsi.py with the pool, query and DCP1
reference swapped to the MXFP4 forms; every other line of its procedure is unchanged.

Bundle-entry correctness (fixed before the first run). What runs: 16 TsiState objects (one per TP16 rank: real prepare + scan,
the scan now idx_fp4_q.paged_logits over each rank's local pages of an MXFP4 pool written by idx_fp4.store_index_k from random BF16 keys);
their local logits feed one tsim.Emulator launch; then ONE rank R runs TsiState.select exactly as verify.py calls it, after R's own records
were erased from every region.
Checks (exact, no tolerance; the reference is independent of TSI):
- reference: numpy over the FULL DCP1 logits of each row from the verify path's MXFP4-page logits call (idx_fp4_q.paged_logits on the
  request page table, the call VerifySelection makes at DCP1): min(n, 2048) keys of largest score, ties by position, source-major;
- validity: the emulated group's 16 outputs equal the reference (holds only if every rank's local MXFP4 logits equal the DCP1 logits at the
  same positions);
- R's output == the reference, R's fallback flags == the emulator's, regions unchanged after R's launch.
Cases: L 70,000 x Breq 2, 300,000 x 1, 700,000 x 2 (ordinals 0 / 1 / 0 on ranks 0 / 5 / 15); the planted case (request 0's rank-3 pages
carry 8x keys -> fallback rows on request 0).
Mutants (each must FAIL; K31_MUTATION=<name>): tsi_q4_table (the scan's MXFP4 logits read the request page table instead of the rank's
local table), tsi_q4_off (the scan keeps the FP8 DeepGEMM call on the MXFP4 pool).
"""

from __future__ import annotations

import os

os.environ["K31MK_TSI_MERGE"] = "tsim"  # read by tsi.py at import (one module build per boot)
os.environ["K31MK_DSA_IDX_FP4"] = "1"
os.environ["K31MK_DSA_IDX_Q"] = os.environ.get("K31_TEST_IDX_Q", "fp8")
os.environ["K31MK_DSA_DG_VENDOR"] = "1"
os.environ.pop("K31MK_DSA_IDX_FP4_FP8Q", None)
os.environ.setdefault("K31MK_TSI_S1R", "1")  # the served TSI config (Run K) beside it; TSIM does not use the merge

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import _harness as H  # noqa: E402

MUTATIONS = {
    "tsi_q4_table": ("kimi_k31/dsa/tsi.py", "scores = paged_logits(query, gates, cache, self.lctx, self.ltab, self.schedule,",
                     "scores = paged_logits(query, gates, cache, self.lctx, self.table[:, : self.width], self.schedule,"),
    "tsi_q4_off": ("kimi_k31/dsa/tsi.py", "        if IDX_FP4:  # MXFP4 index-K pages: the verify path's",
                   "        if False:  # MXFP4 index-K pages: the verify path's"),
}
H.load_plugin(MUTATIONS)

from sgl_nvidia_mega.kimi_k31.dsa import idx_fp4 as I4  # noqa: E402
from sgl_nvidia_mega.kimi_k31.dsa import idx_fp4_q as IQ  # noqa: E402
from sgl_nvidia_mega.kimi_k31.dsa import tsi as T  # noqa: E402
from sgl_nvidia_mega.kimi_k31.dsa import tsim as M  # noqa: E402

FP4Q = os.environ["K31MK_DSA_IDX_Q"] == "fp4"

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)
DEV = "cuda"
W, PAGE, TOPK, Q = 16, 64, 2048, 8
POISON = -7
_EMU = None


def emu():
    global _EMU
    if _EMU is None:
        _EMU = M.Emulator(DEV, cap=32, layers=24)
    return _EMU


def gen(seed):
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    return g


class Case:
    """Requests with random page tables over one FP8 index-K pool (= test_k31_tsi.py Case, plus a table width)."""

    def __init__(self, contexts, seed, *, width=None, boost=None, boost_requests=(0,)):
        g = self.g = gen(seed)
        self.b, self.q, self.rows = len(contexts), Q, len(contexts) * Q
        self.ctx = list(contexts)
        pages = [-(-c // PAGE) for c in contexts]
        self.width = width or max(pages) + 3
        assert self.width >= max(pages)
        pool = sum(pages) + 5
        perm = torch.randperm(pool, generator=g, device=DEV).to(torch.int32)
        table = torch.randint(0, pool, (self.b, self.width), generator=g, device=DEV, dtype=torch.int32)
        at = 0
        for i, n in enumerate(pages):
            table[i, :n] = perm[at : at + n]
            at += n
        self.table = table
        keys = torch.randn(pool, PAGE, 128, generator=g, device=DEV) * 0.5 * (torch.rand(pool, PAGE, 1, generator=g, device=DEV) + 0.5)
        if boost is not None:  # rank ``boost`` holds most of these requests' top-2048 -> > 384 keys from one source
            for i in boost_requests:
                p = torch.arange(pages[i], device=DEV)
                phys = table[i, : pages[i]][(p % W) == boost].long()
                keys[phys] *= 8.0
        self.cache = torch.zeros((pool, I4.PAGE_B), dtype=torch.uint8, device=DEV)  # MXFP4 pages [pool, 64 * 68]
        I4.store_index_k(keys.reshape(-1, 128).to(torch.bfloat16), self.cache, torch.arange(pool * PAGE, device=DEV))
        self.contexts = torch.tensor([[c] * Q for c in contexts], dtype=torch.int32, device=DEV)
        self.causal = torch.tensor([c - Q + j + 1 for c in contexts for j in range(Q)], dtype=torch.int32, device=DEV)
        self.new_query()

    def new_query(self):
        if FP4Q:  # the MXFP4 query and its head weights from the fp4 setting's quantizer of record (gates > 0: logits >= +0)
            q = torch.randn(self.rows, 32, 128, generator=self.g, device=DEV).to(torch.bfloat16)
            gate = (torch.rand(self.rows, 32, generator=self.g, device=DEV) + 0.05).to(torch.bfloat16)
            self.buf = I4.QueryBuffers(self.rows, DEV)
            self.query, w = I4.quantize_query(q, gate, self.buf)
            self.gates = w.reshape(self.rows, 32)
        else:
            self.query = torch.randn(self.rows, 32, 128, generator=self.g, device=DEV).to(torch.float8_e4m3fn)
            self.gates = torch.rand(self.rows, 32, generator=self.g, device=DEV) + 0.05  # > 0: logits >= +0, no -0


def full_logits(case):
    """The verify path's MXFP4-page logits over the FULL context (the DCP1 call) [rows, width * 64]."""
    import deep_gemm
    from sglang.kernels.ops.attention.dsa.paged_mqa_logits import _restore_row_stride

    sms = torch.cuda.get_device_properties(0).multi_processor_count
    schedule = deep_gemm.get_paged_mqa_logits_metadata(case.contexts, PAGE, sms)
    s = IQ.paged_logits(case.query, case.gates, case.cache, case.contexts, case.table, schedule, case.width * PAGE, case.b, Q)
    return _restore_row_stride(s)


def reference(case, logits):
    """[rows, 2048] int64 slots: exact top-min(n, 2048) (score desc, position asc), source-major, -1 after."""
    F = logits.float().cpu().numpy()
    table = case.table.cpu().numpy().astype(np.int64)
    causal = case.causal.cpu().numpy()
    ref = np.full((case.rows, TOPK), -1, np.int64)
    for row in range(case.rows):
        n, b = int(causal[row]), row // Q
        s = F[row, :n].astype(np.float64)
        assert np.isfinite(s).all(), "non-finite logits"
        assert not (np.signbit(s) & (s == 0)).any(), "-0 logits"
        want = min(n, TOPK)
        pos = np.arange(n, dtype=np.int64)
        sel = np.lexsort((pos, -s))[:want]
        e = sel[np.lexsort((sel, (sel // PAGE) % W))]
        ref[row, :want] = table[b, e // PAGE] * PAGE + e % PAGE
    return ref


def ranks(case):
    comms = [T.TsiComm(DEV, W, r, cap=32, deadline_ns=10_000_000_000) for r in range(W)]
    states = [T.TsiState(c, case.rows, case.width) for c in comms]
    regions = emu().regions
    for c in comms:  # the serving path takes these from the symmetric-memory rendezvous (TsiComm.from_group)
        c.tsim_pointers = [int(t.data_ptr()) for t in regions]
        c.tsim_mc = 0
    return states


def cand_view(region, cap=32):
    n = 2 * cap * W * M.CAP_SLOTS * 2
    return region[:n].view(2, cap, W, M.CAP_SLOTS, 2)


def full_view(region, cap=32):
    n = 2 * cap * W * M.CAP_SLOTS * 2
    return region[n:].view(2, cap, TOPK, 2)


def first_diff(tag, got, ref):
    bad = np.nonzero((got != ref).any(axis=1))[0]
    if bad.size:
        i = int(bad[0])
        col = int(np.nonzero(got[i] != ref[i])[0][0])
        same = np.array_equal(np.sort(got[i]), np.sort(ref[i]))
        raise AssertionError(f"{tag}: {bad.size}/{ref.shape[0]} rows differ ({'order' if same else 'set'}); row {i} "
                             f"first at {col}: got {got[i, col:col + 4].tolist()} want {ref[i, col:col + 4].tolist()}")


def layer(tag, states, case, ordinal, R, *, prepare, planted=None):
    """One DSA layer: emulate the 16 ranks from their real local logits, erase R's records, run R's serving select."""
    e = emu()
    if prepare:
        for st in states:
            st.prepare(case.table, case.contexts, case.causal)
    case.new_query()
    ref = reference(case, full_logits(case))
    ep = states[0].comm.epoch
    assert all(torch.equal(st.comm.epoch, ep) for st in states), "ranks' epochs differ"
    scores = [st.scan(case.query, case.gates, case.cache) for st in states]
    ncols = states[0].width * PAGE
    assert all(st.width * PAGE == ncols for st in states)
    X = torch.stack([s[:, :ncols] for s in scores]).contiguous()
    LC = torch.stack([st.lcausal for st in states]).contiguous()
    for t in e.regions:
        t.zero_()
    e.epoch.copy_(ep)
    rows = case.rows
    out16 = torch.full((W, rows, TOPK), POISON, dtype=torch.int32, device=DEV)
    fb16 = torch.full((W, rows), -9, dtype=torch.int32, device=DEV)
    scr16 = torch.zeros((W, rows, W * TOPK), dtype=torch.int64, device=DEV)
    e.select(ordinal, X, LC, case.causal, case.table, Q, out16, fb16, scr16, ncols)
    torch.cuda.synchronize()
    o16, f16 = out16.cpu().numpy().astype(np.int64), fb16.cpu().numpy()
    for r in range(W):  # validity: the emulated group reproduces the reference
        first_diff(f"{tag}: EMULATION rank {r}", o16[r], ref)
    if planted is None:
        assert (f16 == 0).all(), f"{tag}: emulation flags {sorted(set(f16.reshape(-1).tolist()))}: fast path expected"
    else:
        want = np.array([1 if (row // Q) in planted else 0 for row in range(rows)])
        assert ((f16 != 0) == want[None, :]).all(), f"{tag}: planted flags not as designed: {f16[0].tolist()}"
    snap = [t.clone() for t in e.regions]
    parity = ordinal & 1
    for t in e.regions:  # erase R's pushes (every region) and R's own FULL list
        cand_view(t)[parity, :rows, R].zero_()
    full_view(e.regions[R])[parity, :rows].zero_()
    st = states[R]
    out = torch.full((rows, TOPK), POISON, dtype=torch.int32, device=DEV)
    st._fallback.fill_(7)
    got = st.select(ordinal, case.query, case.gates, case.cache, out)
    torch.cuda.synchronize()
    assert got.data_ptr() == out.data_ptr()
    first_diff(f"{tag}: SERVING rank {R}", out.cpu().numpy().astype(np.int64), ref)
    fb = st.fallback.cpu().numpy()
    assert np.array_equal(fb, f16[R]), f"{tag}: rank {R} flags {fb.tolist()} want {f16[R].tolist()}"
    for i, (t, s) in enumerate(zip(e.regions, snap)):
        if not torch.equal(t, s):
            d = (t != s).nonzero()
            raise AssertionError(f"{tag}: region {i} differs from the emulated state after rank {R}'s launch at "
                                 f"{int(d.shape[0])} int64 elements, first flat index {int(d[0, 0])}")
    H.record({"kind": "check", "item": tag, "rank": R, "ncols": ncols, "groups": M.groups(ncols),
              "cells": int(rows * TOPK), "violations": 0, "flags": sorted(set(fb.tolist())),
              "epoch": int(ep[0])})


@needs_gpu
def test_import_branch():
    assert T.TSIM, "K31MK_TSI_MERGE=tsim did not select the TSIM branch"
    assert T.IDX_FP4, "K31MK_DSA_IDX_FP4=1 did not reach tsi.py"


@needs_gpu
@pytest.mark.parametrize("L,breq", [(70000, 2), (300000, 1), (700000, 2)])
def test_tsi_path_exact(L, breq):
    """Forward 1: ordinals 0 (rank 0) and 1 (rank 5); forward 2: ordinal 0 again under a new epoch (rank 15)."""
    case = Case([L] if breq == 1 else [L, (L * 3) // 5 + 17], seed=11 + L + breq)
    states = ranks(case)
    for fwd, ordinal, R in ((0, 0, 0), (0, 1, 5), (1, 0, 15)):
        layer(f"exact_L{L}_b{breq}_f{fwd}_o{ordinal}", states, case, ordinal, R, prepare=ordinal == 0)


@needs_gpu
def test_tsi_path_serving_width():
    """The serving table width at context 1,048,576 (verify.py: ceil(ctx / 64) + 16 = 16,400 pages -> 1025 local pages
    = 65,600 columns = 17 groups): tsim_kernel<16> over real DeepGEMM rows, parity 1."""
    case = Case([393216, 524288], seed=5, width=16400)
    states = ranks(case)
    assert M.groups(states[0].max_width * PAGE) == 17, "serving width not as designed"
    layer("serve_w16400_o0", states, case, 0, 9, prepare=True)
    layer("serve_w16400_o1", states, case, 1, 2, prepare=False)


@needs_gpu
def test_tsi_path_planted():
    """Request 0's rank-3 pages carry 8x index-K scales: rank 3 sends > 384 candidates -> every rank takes the exact
    FULL-list fallback on request 0's rows (flags != 0); request 1 stays on the fast path. R = 3 (the overflowing rank)
    and R = 12."""
    case = Case([300000, 180017], seed=23, boost=3, boost_requests=(0,))
    states = ranks(case)
    layer("planted_R3", states, case, 0, 3, prepare=True, planted=(0,))
    layer("planted_R12", states, case, 1, 12, prepare=False, planted=(0,))
