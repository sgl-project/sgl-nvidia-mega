# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fused-route sparse core without its acquire fence (dsa/sparse_mla6.cuh kFence = false; K31MK_DSA_REQ_SHARD_TAG_NOFENCE with
K31MK_DSA_REQ_SHARD_AG=fused, TAGNF), one GPU.

Correctness (bitwise; the fence orders memory only, so the claim is identity): on the same rows, the tagged core with the fence
(fence 1, today's fused form) and without it (fence 0) must equal the served core (SparseMla6.run on the decoded slots) word for
word in partials and stats. Words are written as the route writes them ((tag << 25) | (slot + 1), tag = the core's use counter
+ 1), one block holding every row (world 1). Cells: real K3.1 selections (8K / 131K prefix, K31_REAL_ROWS) and a synthetic cell
(-1 entries, slots past a row's length, contexts < 2048 so whole splits are empty).
Mutant (K31_MUTATION): tf_slot (the core decodes slot + 1 as the slot) must fail the bitwise test in both fence forms.

Timing (K31_TIME=1, exploratory; H-FUSED-FENCE): one CUDA graph of 16 calls with distinct random
page tables over a >= 1.2 GB pool (cold KV), per call [reset the use counters (a 6 KB memset), core]; the served core gets the
same memset on a dummy counter. ABBA over (served, fence 1, fence 0). Question: is the in-situ cost of the tagged
core its per-CTA fence? Expectation (written before data): fence 1 - fence 0 is most of that cost at 96 / 128 rows if
the fence is the cost; fence 0 - served small (a few us: the ld.relaxed.sys word loads + decode)."""

import os

import pytest
import torch

import _harness as H

SM = "kimi_k31/dsa/sparse_mla6.cuh"
MUTATIONS = {
    "tf_slot": (SM, "poll_tag(mb + j, tg, t0, p.deadline_ns) & 0x1FFFFFFu) - 1 : -1;", "poll_tag(mb + j, tg, t0, p.deadline_ns) & 0x1FFFFFFu) : -1;"),
}
BMM1 = 0.0721
ROWS_DIR = os.environ.get("K31_REAL_ROWS", "")
DEADLINE = 2_000_000_000


def _mods():
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.dsa import req_shard, sparse_mla6

    return sparse_mla6, req_shard


def _real_positions(kind, prefix=None):
    """[6 groups, 8 rows, 2048] int64 top-2048 positions and [6, 8] causal lengths from topk's captured rows."""
    import numpy as np

    name = {"8k": "8k", "131k": "400k"}[kind]
    rows = torch.from_numpy(np.load(f"{ROWS_DIR}/real_{name}_rows.npy")).cuda()
    lens = torch.from_numpy(np.load(f"{ROWS_DIR}/real_{name}_lens.npy").astype("int64")).cuda()
    if prefix is not None:
        lens = prefix + (lens - lens.view(6, 8)[:, :1].repeat(1, 8).view(-1))
    pos = torch.empty((48, 2048), dtype=torch.int64, device="cuda")
    for i in range(48):
        L = int(lens[i])
        pos[i] = torch.topk(rows[i, :L], 2048).indices
    return pos.view(6, 8, 2048), lens.view(6, 8)


def _make_cell(kind, requests, qw=8, pool_pages=32768, seed=0, page_seed=None):
    """query [rows, 6, 576] FP8 codes, selected [rows, 2048] int32 slots (-1 = none), lengths [rows] int32."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows = requests * qw
    if kind == "synth":
        lens = torch.randint(900, 1900, (rows,), generator=g, device="cuda", dtype=torch.int64)
        pos = torch.full((rows, 2048), -1, dtype=torch.int64, device="cuda")
        for q_ in range(requests):
            base = torch.randperm(4000, generator=g, device="cuda")[:2400]
            for i in range(q_ * qw, (q_ + 1) * qw):
                L = int(lens[i])
                perm = base[torch.randperm(2400, generator=g, device="cuda")]
                pos[i, :L] = perm[:L]
                pos[i, L:L + 64] = perm[L:L + 64]  # past the length: must not count
        pos[::7, 5] = -1
        ctx = 4000
    else:
        prefix = 131072 if kind == "131k" else None
        p6, l6 = _real_positions(kind, prefix)
        pos = torch.stack([p6[i % 6] for i in range(requests)])[:, :qw].reshape(rows, 2048)
        lens = torch.stack([l6[i % 6] for i in range(requests)])[:, :qw].reshape(rows)
        ctx = int(lens.max()) + 1
    pages_per_req = (ctx + 63) // 64
    pg = torch.Generator(device="cuda").manual_seed(seed + 1 if page_seed is None else page_seed)
    table = torch.randperm(pool_pages, generator=pg, device="cuda")[: requests * pages_per_req].view(requests, pages_per_req)
    req = torch.arange(rows, device="cuda") // qw
    p = pos.clamp(min=0)
    slot = table[req[:, None], p // 64] * 64 + p % 64
    selected = torch.where(pos >= 0, slot, torch.full_like(slot, -1)).to(torch.int32)
    query = (torch.randn((rows, 6, 576), generator=g, device="cuda")).to(torch.float8_e4m3fn).view(torch.uint8)
    return query, selected.contiguous(), lens.to(torch.int32).contiguous()


def _pool(pool_pages, seed=7):
    g = torch.Generator(device="cuda").manual_seed(seed)
    out = torch.empty((pool_pages, 64, 576), dtype=torch.uint8, device="cuda")
    for i in range(0, pool_pages, 2048):
        n = min(2048, pool_pages - i)
        out[i:i + n] = torch.randn((n, 64, 576), generator=g, device="cuda").to(torch.float8_e4m3fn).view(torch.uint8)
    return out


def _words(selected, tag=1):
    """The route's word format for one block holding every row: (tag << 25) | (slot + 1), slot -1 -> value 0."""
    return (((selected.to(torch.int64) + 1) & 0x1FFFFFF) | (tag << 25)).to(torch.int32).contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU test")
@pytest.mark.parametrize("kind,requests,qw", [("8k", 12, 8), ("131k", 12, 8), ("131k", 16, 8), ("synth", 3, 5), ("synth", 1, 7)])
def test_tagged_fence_forms_bitwise_equal_served(kind, requests, qw):
    if kind != "synth" and not ROWS_DIR:
        pytest.skip("K31_REAL_ROWS not set")
    S6, RS = _mods()
    pages = {"8k": 4096, "131k": 40000, "synth": 4096}[kind]  # 131K: 16 requests x 2,049 pages must fit
    pool = _pool(pages)
    query, selected, lengths = _make_cell(kind, requests, qw, pool_pages=pages)
    rows = requests * qw
    ref = S6.SparseMla6(torch.device("cuda"), rows, tokens=128)
    rp, rs_ = (t.clone() for t in ref.run(query, pool, selected, lengths, (BMM1, 1.0)))
    words = _words(selected)
    out = {}
    for fence in (True, False):
        core = S6.SparseMla6(torch.device("cuda"), rows, tokens=128)
        H.poison(core.partial, core.stats)
        epochs = torch.zeros((rows, core.splits), dtype=torch.int32, device="cuda")
        tagged = RS.TaggedRows(words, epochs, DEADLINE, rows, 1, fence=fence)
        p, s = core.run_tagged(query, pool, tagged, lengths, (BMM1, 1.0))
        torch.cuda.synchronize()
        out[fence] = (int((p.view(torch.int32) != rp.view(torch.int32)).sum()), int((s.view(torch.int32) != rs_.view(torch.int32)).sum()),
                      int((epochs != 1).sum()))
    H.record(dict(test="tagnf_bitwise", kind=kind, requests=requests, qw=qw, rows=rows, fence1=out[True], fence0=out[False],
                  mutation=os.environ.get("K31_MUTATION", "")))
    for fence, (bp, bs, be) in out.items():
        assert bp == 0 and bs == 0, f"fence {int(fence)}: tagged core differs from the served core: {bp} partial / {bs} stats words"
        assert be == 0, f"fence {int(fence)}: {be} use counters not advanced to 1"


def test_fence_free_rows_need_deadline():
    """TaggedRows refuses a fence-free form without an armed deadline (T2 of the TAGNF conditions)."""
    _, RS = _mods()
    with pytest.raises(ValueError):
        RS.TaggedRows(None, None, 0, 8, 1, fence=False)
    RS.TaggedRows(None, None, 0, 8, 1, fence=True)  # the fenced form keeps today's contract


@pytest.mark.skipif(os.environ.get("K31_TIME") != "1", reason="timing: K31_TIME=1")
@pytest.mark.parametrize("kind", ["8k", "131k"])
@pytest.mark.parametrize("requests", [12, 16])
def test_tagged_fence_timing(kind, requests):
    if not ROWS_DIR:
        pytest.skip("K31_REAL_ROWS not set")
    S6, RS = _mods()
    pool_pages = {"8k": 32768, "131k": 40000}[kind]
    pool = _pool(pool_pages)
    rows, keep = requests * 8, []
    cells = [_make_cell(kind, requests, 8, pool_pages=pool_pages, seed=100 + c, page_seed=1000 + c) for c in range(16)]

    def calls_for(mode):
        lst = []
        for query, selected, lengths in cells:
            core = S6.SparseMla6(torch.device("cuda"), rows, tokens=128)
            epochs = torch.zeros((rows, core.splits), dtype=torch.int32, device="cuda")
            tagged = RS.TaggedRows(_words(selected), epochs, DEADLINE, rows, 1, fence=(mode == "f1"))
            keep.append((core, epochs, tagged))
            if mode == "s":
                lst.append(lambda c=core, e=epochs, q=query, s=selected, l=lengths: (e.zero_(), c.run(q, pool, s, l, (BMM1, 1.0))))
            else:
                lst.append(lambda c=core, e=epochs, t=tagged, q=query, l=lengths: (e.zero_(), c.run_tagged(q, pool, t, l, (BMM1, 1.0))))
        for c in lst:
            c()
        torch.cuda.synchronize()
        return lst

    calls = {m: calls_for(m) for m in ("s", "f1", "f0")}
    res = {}
    for m in ("s", "f1", "f0", "f0", "f1", "s"):
        res.setdefault(m, []).append(H.cold_time_us(calls[m], reps=20, warmup=3)["us_median"])
    H.record(dict(test="tagnf_timing", kind=kind, requests=requests, rows=rows, us=res, sms=H.device_facts().get("sms")))
