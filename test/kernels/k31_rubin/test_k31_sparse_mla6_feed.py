# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""TMA-fed sparse core (dsa/sparse_mla6.cuh kFeed; K31MK_SPARSE_MLA6_FEED=1), one GPU.

Correctness (bitwise; the feed changes only how the same KV / query bytes reach SMEM): on the same rows the feed core must equal
the served core (SparseMla6 without the feed) word for word in partials and stats, untagged and on the fused route (tagged rows,
fence 1 and fence 0). Cells: real K3.1 selections (8K / 131K prefix, K31_REAL_ROWS; 16, 96, 128 rows) and synthetic cells (-1
entries, slots past a row's length, contexts < 2048 so whole splits are empty -> the zero-filled rows). Engagement: the feed
core launches sparse_mla6_kernel<6, 128, 4, false, true, true> and not the served instantiation (profiler kernel names).
Mutants (K31_MUTATION): feed_row_off (each row's copy lands one SMEM row down) must fail the bitwise test; feed_off (the feed
module builds the served kernels) must fail the engagement test.

Timing (K31_TIME=1, exploratory; TMA-FEED): cold rig as test_k31_sparse_mla6_tagnf.py, ABBA,
served vs feed (untagged) and fused fence-free vs fused fence-free + feed, 96 / 128 rows (also 16), 8K / 131K."""

import os

import pytest
import torch

import _harness as H
from test_k31_sparse_mla6_tagnf import BMM1, DEADLINE, ROWS_DIR, _make_cell, _pool, _words

SM = "kimi_k31/dsa/sparse_mla6.cuh"
MUTATIONS = {
    "feed_row_off": (SM, "bulk_g2s(&s.kv[r][0], p.kv + static_cast<int64_t>(slot) * kDim, kDim, bar);",
                     "bulk_g2s(&s.kv[(r + 1) % T][0], p.kv + static_cast<int64_t>(slot) * kDim, kDim, bar);"),
    "feed_off": (SM, "constexpr bool kFeedBuild = K31_SMLA6_FEED != 0;", "constexpr bool kFeedBuild = false;"),
}
FEED_KERNEL = "sparse_mla6_kernel<6, 128, 4, false, true, true>"
SERVED_KERNEL = "sparse_mla6_kernel<6, 128, 4, false, true, false>"


def _mods():
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.dsa import req_shard, sparse_mla6

    return sparse_mla6, req_shard


def _diff(a, b):
    return int((a.view(torch.int32) != b.view(torch.int32)).sum())


CELLS = [("8k", 12, 8), ("131k", 12, 8), ("131k", 16, 8), ("8k", 2, 8), ("synth", 3, 5), ("synth", 1, 7), ("synth", 12, 8)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU test")
@pytest.mark.parametrize("kind,requests,qw", CELLS)
def test_feed_bitwise_equal_served(kind, requests, qw):
    if kind != "synth" and not ROWS_DIR:
        pytest.skip("K31_REAL_ROWS not set")
    S6, _ = _mods()
    pages = {"8k": 4096, "131k": 40000, "synth": 4096}[kind]
    pool = _pool(pages)
    query, selected, lengths = _make_cell(kind, requests, qw, pool_pages=pages)
    rows = requests * qw
    ref = S6.SparseMla6(torch.device("cuda"), rows, tokens=128, feed=False)
    rp, rs_ = (t.clone() for t in ref.run(query, pool, selected, lengths, (BMM1, 1.0)))
    core = S6.SparseMla6(torch.device("cuda"), rows, tokens=128, feed=True)
    assert core.feed, "feed core did not take the feed module"
    bad = []
    for rep in range(2):  # twice: a launch must not depend on SMEM left by the previous one
        H.poison(core.partial, core.stats)
        p, s = core.run(query, pool, selected, lengths, (BMM1, 1.0))
        torch.cuda.synchronize()
        bad.append((_diff(p, rp), _diff(s, rs_)))
    H.record(dict(test="feed_bitwise", kind=kind, requests=requests, qw=qw, rows=rows, bad=bad, mutation=os.environ.get("K31_MUTATION", "")))
    for bp, bs in bad:
        assert bp == 0 and bs == 0, f"feed core differs from the served core: {bp} partial / {bs} stats words"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU test")
@pytest.mark.parametrize("kind,requests,qw", [("8k", 12, 8), ("131k", 16, 8), ("synth", 3, 5)])
def test_feed_tagged_bitwise_equal_served(kind, requests, qw):
    if kind != "synth" and not ROWS_DIR:
        pytest.skip("K31_REAL_ROWS not set")
    S6, RS = _mods()
    pages = {"8k": 4096, "131k": 40000, "synth": 4096}[kind]
    pool = _pool(pages)
    query, selected, lengths = _make_cell(kind, requests, qw, pool_pages=pages)
    rows = requests * qw
    ref = S6.SparseMla6(torch.device("cuda"), rows, tokens=128, feed=False)
    rp, rs_ = (t.clone() for t in ref.run(query, pool, selected, lengths, (BMM1, 1.0)))
    words = _words(selected)
    out = {}
    for fence in (True, False):
        core = S6.SparseMla6(torch.device("cuda"), rows, tokens=128, feed=True)
        H.poison(core.partial, core.stats)
        epochs = torch.zeros((rows, core.splits), dtype=torch.int32, device="cuda")
        p, s = core.run_tagged(query, pool, RS.TaggedRows(words, epochs, DEADLINE, rows, 1, fence=fence), lengths, (BMM1, 1.0))
        torch.cuda.synchronize()
        out[fence] = (_diff(p, rp), _diff(s, rs_), int((epochs != 1).sum()))
    H.record(dict(test="feed_tagged_bitwise", kind=kind, requests=requests, qw=qw, rows=rows, fence1=out[True], fence0=out[False],
                  mutation=os.environ.get("K31_MUTATION", "")))
    for fence, (bp, bs, be) in out.items():
        assert bp == 0 and bs == 0, f"fence {int(fence)}: tagged feed core differs from the served core: {bp} / {bs} words"
        assert be == 0, f"fence {int(fence)}: {be} use counters not advanced to 1"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU test")
def test_feed_engaged():
    S6, _ = _mods()
    pool = _pool(4096)
    query, selected, lengths = _make_cell("synth", 3, 8, pool_pages=4096)
    names = {}
    for feed in (False, True):
        core = S6.SparseMla6(torch.device("cuda"), 24, tokens=128, feed=feed)
        core.run(query, pool, selected, lengths, (BMM1, 1.0))
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            core.run(query, pool, selected, lengths, (BMM1, 1.0))
            torch.cuda.synchronize()
        names[feed] = sorted({e.name for e in prof.events() if "sparse_mla6_kernel" in e.name})
    H.record(dict(test="feed_engaged", names={str(k): v for k, v in names.items()}, mutation=os.environ.get("K31_MUTATION", "")))
    assert any(SERVED_KERNEL in n for n in names[False]), f"served core kernels: {names[False]}"
    assert any(FEED_KERNEL in n for n in names[True]) and not any(SERVED_KERNEL in n for n in names[True]), f"feed core kernels: {names[True]}"


def test_feed_declines_other_tokens():
    S6, _ = _mods()
    if not torch.cuda.is_available():
        pytest.skip("GPU test")
    assert not S6.SparseMla6(torch.device("cuda"), 16, tokens=64, feed=True).feed
    assert S6.SparseMla6(torch.device("cuda"), 16, tokens=128, feed=True).feed


@pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("K31_TIME") != "1", reason="timing: K31_TIME=1")
@pytest.mark.parametrize("kind,requests", [("8k", 12), ("131k", 12), ("8k", 16), ("131k", 16), ("8k", 2)])
def test_feed_timing(kind, requests):
    if not ROWS_DIR:
        pytest.skip("K31_REAL_ROWS not set")
    S6, RS = _mods()
    pages = {"8k": 32768, "131k": 40000}[kind]
    pool = _pool(pages)
    rows, keep = requests * 8, []
    cells = [_make_cell(kind, requests, 8, pool_pages=pages, seed=100 + c, page_seed=1000 + c) for c in range(16)]

    def calls_for(feed, tagged):
        lst = []
        for query, selected, lengths in cells:
            core = S6.SparseMla6(torch.device("cuda"), rows, tokens=128, feed=feed)
            if tagged:
                ep = torch.zeros((rows, core.splits), dtype=torch.int32, device="cuda")
                tr = RS.TaggedRows(_words(selected), ep, DEADLINE, rows, 1, fence=False)
                keep.append((core, ep, tr))
                lst.append(lambda c=core, q=query, t=tr, l=lengths, e=ep: (e.zero_(), c.run_tagged(q, pool, t, l, (BMM1, 1.0))))
            else:
                dummy = torch.zeros((rows, core.splits), dtype=torch.int32, device="cuda")
                keep.append((core, dummy))
                lst.append(lambda c=core, q=query, s=selected, l=lengths, d=dummy: (d.zero_(), c.run(q, pool, s, l, (BMM1, 1.0))))
        for c in lst:
            c()
        torch.cuda.synchronize()
        return lst

    calls = {"served": calls_for(False, False), "feed": calls_for(True, False), "f0": calls_for(False, True), "f0_feed": calls_for(True, True)}
    order = ("served", "feed", "f0", "f0_feed")
    res = {}
    for m in order + order[::-1]:
        res.setdefault(m, []).append(H.cold_time_us(calls[m], reps=20, warmup=3)["us_median"])
    H.record(dict(test="feed_timing", kind=kind, requests=requests, rows=rows, us=res, sms=H.device_facts().get("sms")))
    print(kind, rows, {k: [round(x, 2) for x in v] for k, v in res.items()}, flush=True)
