# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""8-warp TMA-fed sparse core (dsa/sparse_mla6.cuh W = 8 with kFeed; K31MK_SPARSE_MLA6_FEED_W8=1 with K31MK_SPARSE_MLA6_FEED=1), one GPU.

Correctness (bitwise): each warp scores one 16-token m-tile and all 8 warps run P.V on 64 latent dims; every output dim keeps its K
order and the per-thread softmax sums of warps 2v / 2v + 1 are added in the 4-warp order (kPair), so partials and stats must equal the
served 4-warp core word for word. Cells as test_k31_sparse_mla6_feed.py (real K3.1 selections at 8K / 131K, 16 / 96 / 128 rows, and
synthetic cells with -1 entries, slots past a row's length and empty splits). The tagged route through the FEED_W8 module keeps the
4-warp kernels and must equal the served core too (fence 1 / 0). Engagement: the FEED_W8 core launches
sparse_mla6_kernel<6, 128, 8, false, true, true> and nothing else; FEED_W8 without the feed declines to the served kernel.
Mutants (K31_MUTATION): w8_pair_off (kPair = false: 8 warp sums added in a different grouping) must fail the bitwise test on the stats;
w8_row_off (the 8-warp issue lands each KV row one SMEM row down) must fail it on the partials.

Timing (K31_TIME=1): cold rig ABBA, served vs feed vs feed + 8 warps, 96 / 128 rows (also 16), 8K / 131K. FORM lines (K31_FORM=1): the
profiler's name | grid | block | smem | regs of each form, as the served boot's trace prints them."""

import json
import os
import tempfile

import pytest
import torch

import _harness as H
from test_k31_sparse_mla6_tagnf import BMM1, DEADLINE, ROWS_DIR, _make_cell, _pool, _words

SM = "kimi_k31/dsa/sparse_mla6.cuh"
MUTATIONS = {
    "w8_pair_off": (SM, "  constexpr bool kPair = W == 8;", "  constexpr bool kPair = false;"),
    "w8_row_off": (SM, "          bulk_g2s(&s.kv[r][0], p.kv + static_cast<int64_t>(slot) * kDim, kDim, bar);\n        } else {\n#pragma unroll\n"
                       "          for (int c = 0; c < kDim / 16; ++c) reinterpret_cast<uint4*>(&s.kv[r][0])[c] = make_uint4(0, 0, 0, 0);\n          mbar_arrive(bar);\n        }",
                   "          bulk_g2s(&s.kv[(r + 1) % T][0], p.kv + static_cast<int64_t>(slot) * kDim, kDim, bar);\n        } else {\n#pragma unroll\n"
                   "          for (int c = 0; c < kDim / 16; ++c) reinterpret_cast<uint4*>(&s.kv[(r + 1) % T][0])[c] = make_uint4(0, 0, 0, 0);\n          mbar_arrive(bar);\n        }"),
}
W8_KERNEL = "sparse_mla6_kernel<6, 128, 8, false, true, true>"
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
def test_feedw8_bitwise_equal_served(kind, requests, qw):
    if kind != "synth" and not ROWS_DIR:
        pytest.skip("K31_REAL_ROWS not set")
    S6, _ = _mods()
    pages = {"8k": 4096, "131k": 40000, "synth": 4096}[kind]
    pool = _pool(pages)
    query, selected, lengths = _make_cell(kind, requests, qw, pool_pages=pages)
    rows = requests * qw
    ref = S6.SparseMla6(torch.device("cuda"), rows, tokens=128, feed=False, feed_w8=False)
    rp, rs_ = (t.clone() for t in ref.run(query, pool, selected, lengths, (BMM1, 1.0)))
    out = {}
    for name, feed, w8 in (("feed", True, False), ("feed_w8", True, True)):
        core = S6.SparseMla6(torch.device("cuda"), rows, tokens=128, feed=feed, feed_w8=w8)
        assert core.feed and core.feed_w8 == w8
        bad = []
        for rep in range(2):  # twice: a launch must not depend on SMEM left by the previous one
            H.poison(core.partial, core.stats)
            p, s = core.run(query, pool, selected, lengths, (BMM1, 1.0))
            torch.cuda.synchronize()
            bad.append((_diff(p, rp), _diff(s, rs_)))
        out[name] = bad
    H.record(dict(test="feedw8_bitwise", kind=kind, requests=requests, qw=qw, rows=rows, bad=out, mutation=os.environ.get("K31_MUTATION", "")))
    for name, bad in out.items():
        for bp, bs in bad:
            assert bp == 0 and bs == 0, f"{name} core differs from the served core: {bp} partial / {bs} stats words"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU test")
@pytest.mark.parametrize("kind,requests,qw", [("8k", 12, 8), ("131k", 16, 8), ("synth", 3, 5)])
def test_feedw8_tagged_route_bitwise(kind, requests, qw):
    if kind != "synth" and not ROWS_DIR:
        pytest.skip("K31_REAL_ROWS not set")
    S6, RS = _mods()
    pages = {"8k": 4096, "131k": 40000, "synth": 4096}[kind]
    pool = _pool(pages)
    query, selected, lengths = _make_cell(kind, requests, qw, pool_pages=pages)
    rows = requests * qw
    ref = S6.SparseMla6(torch.device("cuda"), rows, tokens=128, feed=False, feed_w8=False)
    rp, rs_ = (t.clone() for t in ref.run(query, pool, selected, lengths, (BMM1, 1.0)))
    words = _words(selected)
    out = {}
    for fence in (True, False):
        core = S6.SparseMla6(torch.device("cuda"), rows, tokens=128, feed=True, feed_w8=True)
        H.poison(core.partial, core.stats)
        epochs = torch.zeros((rows, core.splits), dtype=torch.int32, device="cuda")
        p, s = core.run_tagged(query, pool, RS.TaggedRows(words, epochs, DEADLINE, rows, 1, fence=fence), lengths, (BMM1, 1.0))
        torch.cuda.synchronize()
        out[fence] = (_diff(p, rp), _diff(s, rs_), int((epochs != 1).sum()))
    H.record(dict(test="feedw8_tagged_bitwise", kind=kind, requests=requests, qw=qw, rows=rows, fence1=out[True], fence0=out[False],
                  mutation=os.environ.get("K31_MUTATION", "")))
    for fence, (bp, bs, be) in out.items():
        assert bp == 0 and bs == 0, f"fence {int(fence)}: tagged route of the FEED_W8 module differs from the served core: {bp} / {bs} words"
        assert be == 0, f"fence {int(fence)}: {be} use counters not advanced to 1"


def _kernel_names(core, query, pool, selected, lengths):
    core.run(query, pool, selected, lengths, (BMM1, 1.0))
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        core.run(query, pool, selected, lengths, (BMM1, 1.0))
        torch.cuda.synchronize()
    return sorted({e.name for e in prof.events() if "sparse_mla6_kernel" in e.name})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU test")
def test_feedw8_engaged():
    S6, _ = _mods()
    pool = _pool(4096)
    query, selected, lengths = _make_cell("synth", 3, 8, pool_pages=4096)
    names = _kernel_names(S6.SparseMla6(torch.device("cuda"), 24, tokens=128, feed=True, feed_w8=True), query, pool, selected, lengths)
    H.record(dict(test="feedw8_engaged", names=names, mutation=os.environ.get("K31_MUTATION", "")))
    assert len(names) == 1 and W8_KERNEL in names[0], f"FEED_W8 core kernels: {names}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU test")
def test_feedw8_declines_without_feed():
    S6, _ = _mods()
    pool = _pool(4096)
    query, selected, lengths = _make_cell("synth", 3, 8, pool_pages=4096)
    core = S6.SparseMla6(torch.device("cuda"), 24, tokens=128, feed=False, feed_w8=True)
    assert not core.feed_w8
    names = _kernel_names(core, query, pool, selected, lengths)
    assert len(names) == 1 and SERVED_KERNEL in names[0], f"FEED_W8 without the feed: {names}"
    assert not S6.SparseMla6(torch.device("cuda"), 16, tokens=64, feed=True, feed_w8=True).feed_w8


@pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("K31_FORM") != "1", reason="form lines: K31_FORM=1")
@pytest.mark.parametrize("kind,requests", [("131k", 12), ("8k", 12), ("131k", 16), ("8k", 2)])
def test_feedw8_form_lines(kind, requests):
    if not ROWS_DIR:
        pytest.skip("K31_REAL_ROWS not set")
    S6, _ = _mods()
    pages = {"8k": 32768, "131k": 40000}[kind]
    pool = _pool(pages)
    query, selected, lengths = _make_cell(kind, requests, 8, pool_pages=pages, seed=100, page_seed=1000)
    rows = requests * 8
    for form, feed, w8 in (("served", False, False), ("feed", True, False), ("feed_w8", True, True)):
        core = S6.SparseMla6(torch.device("cuda"), rows, tokens=128, feed=feed, feed_w8=w8)
        core.run(query, pool, selected, lengths, (BMM1, 1.0))
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            core.run(query, pool, selected, lengths, (BMM1, 1.0))
            torch.cuda.synchronize()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "t.json")
            prof.export_chrome_trace(path)
            ev = json.load(open(path))["traceEvents"]
        lines = sorted({f"{e['name']} | grid {tuple(e['args'].get('grid', ()))} | block {tuple(e['args'].get('block', ()))} | "
                        f"smem {e['args'].get('shared memory')} | regs {e['args'].get('registers per thread')}"
                        for e in ev if e.get("cat") == "kernel" and "sparse_mla6_kernel" in e.get("name", "")})
        for l in lines:
            print(f"FORM rig {kind} rows={rows} form={form} | {l}", flush=True)
        H.record(dict(test="feedw8_form", kind=kind, rows=rows, form=form, lines=lines))
        assert len(lines) == 1


@pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("K31_TIME") != "1", reason="timing: K31_TIME=1")
@pytest.mark.parametrize("kind,requests", [("8k", 12), ("131k", 12), ("8k", 16), ("131k", 16), ("8k", 2)])
def test_feedw8_timing(kind, requests):
    if not ROWS_DIR:
        pytest.skip("K31_REAL_ROWS not set")
    S6, _ = _mods()
    pages = {"8k": 32768, "131k": 40000}[kind]
    pool = _pool(pages)
    rows, keep = requests * 8, []
    cells = [_make_cell(kind, requests, 8, pool_pages=pages, seed=100 + c, page_seed=1000 + c) for c in range(16)]

    def calls_for(feed, w8):
        lst = []
        for query, selected, lengths in cells:
            core = S6.SparseMla6(torch.device("cuda"), rows, tokens=128, feed=feed, feed_w8=w8)
            dummy = torch.zeros((rows, core.splits), dtype=torch.int32, device="cuda")
            keep.append((core, dummy))
            lst.append(lambda c=core, q=query, s=selected, l=lengths, d=dummy: (d.zero_(), c.run(q, pool, s, l, (BMM1, 1.0))))
        for c in lst:
            c()
        torch.cuda.synchronize()
        return lst

    calls = {"served": calls_for(False, False), "feed": calls_for(True, False), "feed_w8": calls_for(True, True)}
    order = ("served", "feed", "feed_w8")
    res = {}
    for m in order + order[::-1]:
        res.setdefault(m, []).append(H.cold_time_us(calls[m], reps=20, warmup=3)["us_median"])
    H.record(dict(test="feedw8_timing", kind=kind, requests=requests, rows=rows, us=res, sms=H.device_facts().get("sms")))
    print(kind, rows, {k: [round(x, 2) for x in v] for k, v in res.items()}, flush=True)
