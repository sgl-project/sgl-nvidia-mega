# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 MK token-split indexer (kimi_k31/dsa/tsi.cuh + tsi.py) on ONE GPU with all 16 ranks emulated.

Every rank has its own receive region and epoch on this GPU; each DSA layer runs the scan (DeepGEMM logits over the
rank's pages + topk_v2) and the push half of the merge for every rank, then the poll half for every rank (``phases``),
so the 16-rank protocol (tags, parity, certificate, fallback, sort) runs with no cross-GPU dependency. Serving runs
push + poll in one launch. Any SM100-family GPU works.

References (independent of the code under test; exact comparisons, no tolerance):
- local lengths: brute-force count of the positions p < n with (p // 64) % 16 == r; local table: table[b, 16 i + r].
- logits: DeepGEMM over the FULL context (the DCP1 call verify.py makes); every rank's local logits gathered back to
  their global positions (brute-force position list) must equal it bitwise.
- selection: torch stable sort of the full logits over each row's causal prefix (score descending, position
  ascending), first min(n, 2048), sorted by position, mapped through the page table, -1 after. Every rank's output
  must equal it exactly. Under K31MK_TSI_S1R=1 (the s1r merge, tsi.cuh K31_TSI_EMIT 3) the same positions are
  ordered SOURCE-MAJOR on rows whose expected fallback flag is 0: source q = (pos // 64) % 16 ascending, position
  ascending inside a source; rows with flag 1 stay position-sorted (s1r keeps the radix sort on fallback rows).
  The order is derived from the reference positions only (the spec in tsi.py S1R), not from the kernel.
- fallback flag (closed form, see expected_flags): a row falls back iff some source q has more than C local keys and
  holds >= C of the row's global top-2048.
- the poll deadline (a source that never pushes or holds stale records must stop the reader) is tested in
  test_k31_tsi_trap.py, which deliberately faults the GPU and therefore runs on a dedicated node only.

Mutations (``K31_MUTATION=<name>`` through _harness / the fixed run_suite.sh; each must FAIL this file):
  drop_source  a dropped source (treated as empty)  -> test_select_exact (fallback flags)
  pos_page     position remap off by one page        -> test_select_exact, test_planted_fallback, test_ties
  stale_tag    any nonzero tag accepted              -> judged by test_k31_tsi_trap.py only
  no_cert      certificate never fails               -> test_planted_fallback
  unsorted     output not sorted by position         -> test_select_exact, test_ties (S1R: test_planted_fallback)
  local_len    local_len off by one                  -> test_prep_local_len, test_select_exact
  sender_sort  s1r sender's position ranking skipped -> S1R only: test_select_exact, test_ties (no effect on the
               default merge: tsi.cuh compiles it only under K31_TSI_EMIT 3)
  dg_scale_map vendored logits' index-K scale map 64 B off -> K31MK_DSA_DG_VENDOR=1 only: test_local_logits_bitwise,
               test_select_exact (with the switch off the scan runs DeepGEMM and never reads that map)
  t1l_full_short  T1L: the lazily published FULL list holds one key too few -> K31MK_TSI_T1L=1 only:
               test_planted_fallback (the only rows that publish)
  t1l_no_publish  T1L: the FULL list is never published -> readers trap at the deadline: judged like stale_tag
  t1l_raw_off  T1L: the sender reads every candidate one local token off -> K31MK_TSI_T1L=1 only: test_select_exact,
               test_ties, test_planted_fallback
Under K31MK_TSI_T1L=1 (with K31MK_TSI_S1R=1) the scan keeps the local top-256 and a layer runs three merge launches
per rank in turn (push, poll + certificate + FULL publish, fallback + output); the references are unchanged.
Run: K31_PLUGIN=<tree> bash run_suite.sh OUT test/kernels/k31_rubin/test_k31_tsi.py \
       drop_source pos_page no_cert unsorted local_len
     K31MK_TSI_S1R=1 K31_PLUGIN=<tree> bash ... drop_source pos_page no_cert unsorted local_len sender_sort
     K31MK_TSI_S1R=1 K31MK_TSI_T1L=1 K31_PLUGIN=<tree> bash ... (the S1R list) t1l_full_short t1l_raw_off
"""

from __future__ import annotations

import pytest
import torch

import _harness as H
import _hook_variants as V

MUTATIONS = {
    name: V.edits("kimi_k31/dsa/tsi.cuh", n)
    for n, name in enumerate(
        ("drop_source", "pos_page", "stale_tag", "no_cert", "unsorted", "local_len", "sender_sort", "t1l_full_short",
         "t1l_no_publish", "t1l_raw_off"),
        start=1,
    )
}
MUTATIONS["dg_scale_map"] = ("kimi_k31/dsa/dg_logits.py", "base + PAGE * HEAD_DIM,", "base + PAGE * HEAD_DIM + 64,")
H.load_plugin(MUTATIONS)

from sgl_nvidia_mega.kimi_k31.dsa import tsi as T  # noqa: E402

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)
DEV = "cuda"
W = T.WORLD
PAGE = 64
Q = 8
POISON = -7


def gen(seed):
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    return g


def owner_counts(n):
    """[W] brute-force count of the positions p < n owned by each rank ((p // 64) % W)."""
    if n <= 0:
        return torch.zeros(W, dtype=torch.int64)
    owners = (torch.arange(n, device=DEV) // PAGE) % W
    return torch.bincount(owners, minlength=W).cpu()


def owned_positions(n, r):
    """Ascending global positions p < n of rank r (local index l = index in this list)."""
    p = torch.arange(n, device=DEV)
    return p[(p // PAGE) % W == r]


class Case:
    """Requests with random (non-identity) page tables over one index-K pool, FP8 queries and FP32 head gates."""

    def __init__(self, contexts, seed, *, tie=False, boost=None, boost_requests=(0,)):
        g = self.g = gen(seed)
        self.b, self.q, self.rows = len(contexts), Q, len(contexts) * Q
        self.ctx = list(contexts)
        pages = [-(-c // PAGE) for c in contexts]
        self.width = max(pages) + 3  # wider than needed and not a multiple of 16: ltab zero-fill path
        pool = sum(pages) + 5
        perm = torch.randperm(pool, generator=g, device=DEV).to(torch.int32)
        table = torch.randint(0, pool, (self.b, self.width), generator=g, device=DEV, dtype=torch.int32)
        at = 0
        for i, n in enumerate(pages):
            table[i, :n] = perm[at : at + n]
            at += n
        self.table = table
        if tie:  # every index-K row is one of 37 templates: logits take 37 values per query row
            tv = (torch.randn(37, 128, generator=g, device=DEV) * 0.5).to(torch.float8_e4m3fn).view(torch.uint8)
            ts = torch.rand(37, generator=g, device=DEV) * 0.02 + 0.01
            pick = torch.randint(0, 37, (pool * PAGE,), generator=g, device=DEV)
            vals = tv[pick].view(pool, PAGE * 128)
            scales = ts[pick].view(pool, PAGE).contiguous()
        else:
            vals = (torch.randn(pool, PAGE * 128, generator=g, device=DEV) * 0.5).to(torch.float8_e4m3fn)
            vals = vals.view(torch.uint8)
            scales = torch.rand(pool, PAGE, generator=g, device=DEV) * 0.02 + 0.01
        if boost is not None:  # the pages of one rank get 8x scales: that rank holds most of the top-2048
            for i in boost_requests:
                p = torch.arange(pages[i], device=DEV)
                phys = table[i, : pages[i]][(p % W) == boost].long()
                scales[phys] *= 8.0
        self.cache = torch.cat([vals, scales.view(torch.uint8).view(pool, PAGE * 4)], dim=1).reshape(-1).contiguous()
        self.contexts = torch.tensor([[c] * Q for c in contexts], dtype=torch.int32, device=DEV)
        self.causal = torch.tensor([c - Q + j + 1 for c in contexts for j in range(Q)], dtype=torch.int32, device=DEV)
        self.new_query()

    def new_query(self):
        self.query = torch.randn(self.rows, 32, 128, generator=self.g, device=DEV).to(torch.float8_e4m3fn)
        self.gates = torch.rand(self.rows, 32, generator=self.g, device=DEV) + 0.05  # > 0: logits >= +0, no -0


def full_logits(case):
    """DeepGEMM paged MQA logits over the full context (the DCP1 call verify.py makes) [rows, width * 64]."""
    import deep_gemm
    from sglang.kernels.ops.attention.dsa.paged_mqa_logits import _restore_row_stride

    sms = torch.cuda.get_device_properties(0).multi_processor_count
    schedule = deep_gemm.get_paged_mqa_logits_metadata(case.contexts, PAGE, sms)
    s = deep_gemm.fp8_paged_mqa_logits(
        case.query.view(case.b, Q, 32, 128), case.cache.view(-1, PAGE, 1, T.INDEX_BYTES), case.gates,
        case.contexts, case.table, schedule, case.width * PAGE, clean_logits=False,
    )
    return _restore_row_stride(s)


def expected_flags(top_positions, n, cand=T.CAND):
    """Closed form of the certificate outcome: a row falls back iff some source q has more than C local keys and holds
    >= C of the row's global top-k (it then sent C keys that all reach t*; otherwise its C-th key is below t*)."""
    held = torch.bincount((top_positions // PAGE) % W, minlength=W).cpu()
    ll = owner_counts(n)
    return int(any(int(ll[q]) > cand and int(held[q]) >= cand for q in range(W)))


def emission_order(pos, flag):
    """The slot order the merge must emit for one row's selected positions (ascending): position order, or under
    T.S1R on a row whose certificate holds, source-major ((pos // 64) % W ascending, then position)."""
    if not T.S1R or flag:
        return pos
    key = ((pos // PAGE) % W) * (1 << 40) + pos
    return pos[torch.sort(key).indices]


def reference(case, logits):
    """[rows, 2048] int32 slots of the exact top-2048 under (score desc, position asc), sorted by position (S1R:
    emission_order), -1 after; [rows] expected fallback flags; [rows] tie count at the cut (ties beyond the cut
    included)."""
    ref = torch.full((case.rows, T.TOPK), -1, dtype=torch.int32, device=DEV)
    flags, ties = [], []
    for row in range(case.rows):
        n, b = int(case.causal[row]), row // Q
        s = logits[row, :n].float()
        assert bool(torch.isfinite(s).all()), "non-finite logits"
        assert not bool((torch.signbit(s) & (s == 0)).any()), "-0 logits (the key order differs from IEEE equality)"
        k = min(n, T.TOPK)
        order = torch.sort(s, descending=True, stable=True)
        top = order.indices[:k]
        pos = torch.sort(top).values
        flags.append(expected_flags(pos, n))
        pos = emission_order(pos, flags[-1])
        ref[row, :k] = (case.table[b, pos // PAGE].long() * PAGE + pos % PAGE).to(torch.int32)
        ties.append(int((s == order.values[k - 1]).sum()) if k else 0)
    return ref, torch.tensor(flags, dtype=torch.int32), ties


def ranks(max_rows, max_pages, *, deadline_s=10.0):
    records = T.region_records(W, 32, T.CAND)
    regions = [torch.zeros(records, dtype=torch.int64, device=DEV) for _ in range(W)]
    comms = [T.TsiComm(DEV, W, r, cap=32, peers=regions, deadline_ns=int(deadline_s * 1e9)) for r in range(W)]
    states = [T.TsiState(c, max_rows, max_pages) for c in comms]
    return comms, states, regions


def prepare_all(states, case):
    for st in states:
        st.prepare(case.table, case.contexts, case.causal)


def run_layer(states, case, ordinal, outs):
    for st, out in zip(states, outs):
        out.fill_(POISON)
        st._fallback.fill_(7)
    for st, out in zip(states, outs):
        st.select(ordinal, case.query, case.gates, case.cache, out, phases=T.PUSH)
    for st, out in zip(states, outs):
        st.select(ordinal, case.query, case.gates, case.cache, out, phases=T.POLL)
    if states[0].comm.t1l:  # T1L: every FULL list a fallback reads is published in the POLL launches above
        for st, out in zip(states, outs):
            st.select(ordinal, case.query, case.gates, case.cache, out, phases=T.FINISH)
    torch.cuda.synchronize()


def assert_rank_outputs(tag, states, outs, ref, flags):
    for r, (st, out) in enumerate(zip(states, outs)):
        bad = (out != ref).any(dim=1)
        if bool(bad.any()):
            row = int(bad.nonzero()[0])
            col = int((out[row] != ref[row]).nonzero()[0])
            raise AssertionError(
                f"{tag}: rank {r}: {int(bad.sum())}/{out.shape[0]} rows differ; row {row} first at {col}: "
                f"got {out[row, col:col + 4].tolist()} want {ref[row, col:col + 4].tolist()}; "
                f"cells {int((out != ref).sum())}"
            )
        got = st.fallback.cpu()
        assert torch.equal(got, flags), f"{tag}: rank {r}: fallback flags {got.tolist()} want {flags.tolist()}"


# ------------------------------------------------------------------------------------------------ prep
@needs_gpu
def test_prep_local_len():
    lengths = [0, 1, 63, 64, 65, 127, 1023, 1024, 1025, 2047, 3000, 4095, 4096, 4097, 65536, 70000, 300001,
               700000, 786465, 1048576]
    g = gen(11)
    lengths += torch.randint(1, 1 << 20, (4,), generator=g, device=DEV).tolist()
    b = len(lengths) // Q
    lengths = lengths[: b * Q]
    pages = 12345
    table = torch.randint(0, 1 << 20, (b, pages), generator=gen(12), device=DEV, dtype=torch.int32)
    contexts = torch.tensor(lengths, dtype=torch.int32, device=DEV).view(b, Q)
    causal = torch.tensor(list(reversed(lengths)), dtype=torch.int32, device=DEV)
    width = -(-pages // W)
    counts = {n: owner_counts(n) for n in set(lengths)}
    checked = 0
    for r in range(W):
        comm = T.TsiComm(DEV, W, r, peers=None)
        ltab = torch.full((b, width), POISON, dtype=torch.int32, device=DEV)
        lctx = torch.full((b, Q), POISON, dtype=torch.int32, device=DEV)
        lcausal = torch.full((b * Q,), POISON, dtype=torch.int32, device=DEV)
        before = int(comm.epoch[0])
        comm.prep(table, contexts, causal, ltab, lctx, lcausal)
        torch.cuda.synchronize()
        assert int(comm.epoch[0]) == before + 1
        want_ctx = torch.tensor([int(counts[n][r]) for n in lengths], dtype=torch.int32).view(b, Q)
        want_causal = torch.tensor([int(counts[n][r]) for n in reversed(lengths)], dtype=torch.int32)
        assert torch.equal(lctx.cpu(), want_ctx), (r, lctx.cpu().tolist(), want_ctx.tolist())
        assert torch.equal(lcausal.cpu(), want_causal), (r, lcausal.cpu().tolist(), want_causal.tolist())
        idx = torch.arange(width, device=DEV) * W + r
        want_tab = torch.zeros((b, width), dtype=torch.int32, device=DEV)
        ok = idx < pages
        want_tab[:, ok] = table[:, idx[ok]]
        assert torch.equal(ltab, want_tab), r
        checked += lctx.numel() + lcausal.numel() + ltab.numel()
    H.record({"kind": "check", "item": "prep_local_len", "cells": checked, "violations": 0})


# ------------------------------------------------------------------------------------------------ logits + local stage
CASES = [(3000, 1), (3000, 2), (70000, 1), (70000, 2), (300000, 1), (300000, 2), (700000, 1), (700000, 2)]


def contexts_of(L, breq):
    return [L] if breq == 1 else [L, (L * 3) // 5 + 17]


@needs_gpu
@pytest.mark.parametrize("L,breq", CASES)
def test_local_logits_bitwise(L, breq):
    """Each rank's local logits == the full DCP1 logits at their global positions, bitwise; the local top-2048 (stock
    topk_v2, stable ties) == the exact local top-2048 under (score desc, local index asc)."""
    case = Case(contexts_of(L, breq), seed=100 + L + breq)
    full = full_logits(case)
    _, states, _ = ranks(case.rows, case.width)
    cells = 0
    for r, st in enumerate(states):
        st.prepare(case.table, case.contexts, case.causal)
        scores = st.scan(case.query, case.gates, case.cache)
        torch.cuda.synchronize()
        for row in range(case.rows):
            b = row // Q
            pos = owned_positions(case.ctx[b], r)
            local = scores[row, : pos.numel()]
            want = full[row, pos]
            assert torch.equal(local.view(torch.int32), want.view(torch.int32)), (r, row)
            cells += pos.numel()
            n_l = int(owner_counts(int(case.causal[row]))[r])
            k = min(n_l, st.raw.shape[1])  # the local top-2048 (T1L: top-256)
            exact = torch.sort(scores[row, :n_l].float(), descending=True, stable=True).indices[:k]
            got = st.raw[row][st.raw[row] >= 0].long()
            assert got.numel() == k and torch.equal(torch.sort(got).values, torch.sort(exact).values), (
                f"local top-k of rank {r} row {row} (local length {n_l}) differs from the exact one: "
                f"{len(set(got.tolist()) ^ set(exact.tolist()))} indices")
    total = sum(case.ctx[row // Q] for row in range(case.rows))
    assert cells == total, (cells, total)  # every position of every row compared exactly once
    H.record({"kind": "check", "item": f"logits_L{L}_b{breq}", "cells": cells, "violations": 0})


# ------------------------------------------------------------------------------------------------ selection
@needs_gpu
@pytest.mark.parametrize("L,breq", CASES)
def test_select_exact(L, breq):
    """Three layers (forward 1: ordinals 0, 1; forward 2: ordinal 0, reusing parity 0 under a new tag), new queries
    each: every rank's slots == the exact reference, fallback flags == the closed form (0 on these data)."""
    case = Case(contexts_of(L, breq), seed=L + breq)
    _, states, _ = ranks(case.rows, case.width)
    outs = [torch.empty(case.rows, T.TOPK, dtype=torch.int32, device=DEV) for _ in range(W)]
    checked = 0
    for fwd, ordinal in ((0, 0), (0, 1), (1, 0)):
        if ordinal == 0:
            prepare_all(states, case)
        case.new_query()
        ref, flags, _ = reference(case, full_logits(case))
        run_layer(states, case, ordinal, outs)
        assert_rank_outputs(f"L={L} breq={breq} fwd={fwd} ord={ordinal}", states, outs, ref, flags)
        checked += W * case.rows * T.TOPK
    print("TSI_SELECT", L, breq, "cells", checked, flush=True)
    H.record({"kind": "check", "item": f"select_L{L}_b{breq}", "cells": checked, "violations": 0})


@needs_gpu
@pytest.mark.parametrize("L", [70000, 300000, 700000])
def test_planted_fallback(L):
    """Request 0's rank-3 pages carry 8x index-K scales: rank 3 holds most of its top-2048, so its certificate fails
    on request 0's rows (flag 1 on every rank) and the fallback still returns the exact set; request 1 stays 0."""
    case = Case([L, (L * 3) // 5 + 17], seed=7 + L, boost=3, boost_requests=(0,))
    _, states, _ = ranks(case.rows, case.width)
    outs = [torch.empty(case.rows, T.TOPK, dtype=torch.int32, device=DEV) for _ in range(W)]
    prepare_all(states, case)
    ref, flags, _ = reference(case, full_logits(case))
    assert flags[:Q].tolist() == [1] * Q and flags[Q:].tolist() == [0] * Q, f"planted case not as designed: {flags}"
    run_layer(states, case, 1, outs)
    assert_rank_outputs(f"planted L={L}", states, outs, ref, flags)
    H.record({"kind": "check", "item": f"planted_L{L}", "cells": W * case.rows * T.TOPK, "violations": 0})


@needs_gpu
@pytest.mark.parametrize("L", [70000, 300000])
def test_ties(L):
    """Index-K rows from 37 templates: every row's cut splits a group of equal scores; the slots must follow
    (score desc, position asc). L <= 300K keeps the local rows off topk_v2's cluster path (
    stock truncates an overflowing tie bin there)."""
    case = Case([L], seed=3 + L, tie=True)
    _, states, _ = ranks(case.rows, case.width)
    outs = [torch.empty(case.rows, T.TOPK, dtype=torch.int32, device=DEV) for _ in range(W)]
    prepare_all(states, case)
    ref, flags, ties = reference(case, full_logits(case))
    assert min(ties) > 100, f"tie construction failed: ties at the cut {ties}"
    run_layer(states, case, 0, outs)
    assert_rank_outputs(f"ties L={L}", states, outs, ref, flags)
    H.record({"kind": "check", "item": f"ties_L{L}", "cells": W * case.rows * T.TOPK, "violations": 0,
              "ties_at_cut": ties})
