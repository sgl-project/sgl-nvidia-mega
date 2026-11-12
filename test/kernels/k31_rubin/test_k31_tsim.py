# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 MK TSIM (kimi_k31/dsa/tsim.cuh + tsim.py) on ONE GPU with all 16 ranks emulated.

The kernel runs as one cooperative grid [rows, 16 ranks]: every rank's sender pushes into all 16 regions (unicast),
every rank's receiver polls its own region, and the exact fallback reads the 16 FULL lists, as on 16 GPUs (the
multicast push, the alias fence and the deadline trap need real GPUs: the 4-GPU probe and test_k31_tsim_trap, later).

Reference (independent of the code under test; exact comparison, no tolerance): numpy over each GLOBAL row: the
min(n, 2048) keys of largest score, ties by position ascending (float64 comparison: -0.0 == +0.0), emitted
SOURCE-MAJOR (source (pos // 64) % 16 ascending, position ascending inside a source), mapped through the page table
(slot = table[pos // 64] * 64 + pos % 64), -1 after. Every rank's output must equal it. For the real rows the stored
reference of the prep tool (pc2_rows.npz exp_*) must equal it too. Local rows: global positions p < n with
(p // 64) % 16 == r in order; the columns past the local length hold +3e38 POISON (a selection that reads them fails).
Fallback flags: 0 on real / random / short rows (the fast path must hold there); != 0 on the planted rows (constant
rows, all-signed-zero rows, one rank holding the whole top-2048), whose outputs must still be exact.

Mutations (K31_MUTATION=<name>; each must FAIL this file):
  drop_source   -> test_real (outputs)          tie_rank     -> test_real (flags: the fast path must hold)
  emit_reversed -> test_real (order)            no_fallback  -> test_planted (outputs)
  full_trunc    -> test_planted[boost]          no_canon     -> test_planted[signed_zero]
  pos_page      -> test_real                    stash_group  -> test_long (rows of 9-16 groups)
  skip_last     -> test_real / test_serve (the uniform group skip drops a partial last group)
  long_drop     -> test_too_long (a local row past 16 groups must take the exact fallback)
  push_dest15   -> test_real (one destination never receives: its receiver's poll deadline sets flag 8)
  push_src_shift -> test_real (records land in the next source's slots)
  disp_plus1    -> test_long[560000] (rows of exactly 9 groups run the 8-group body)
  san_race / san_heap_oob / san_sync / san_uninit: compute-sanitizer POSITIVE controls only (racecheck / memcheck / synccheck /
  initcheck must report); never run them plainly. san_uninit passes an uninitialised scratch (run with PYTORCH_NO_CUDA_MEMORY_CACHING=1).
Serving width: CUDA-graph serving passes the STATIC table width (65,536 local columns at context 1,048,576 =
16 groups), so tsim_kernel<16> runs every row length; test_serve runs real and short rows at that width.
Data: K31_TSIM_ROWS = pc2_rows.npz (captured index logits, not included).
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

import _harness as H
import _hook_variants as V

MUTATIONS = {
    name: V.edits("kimi_k31/dsa/tsim.cuh", n)
    for n, name in enumerate(
        ("drop_source", "tie_rank", "emit_reversed", "no_fallback", "full_trunc", "no_canon", "pos_page", "stash_group",
         "skip_last", "san_race", "san_heap_oob", "long_drop", "push_dest15", "push_src_shift", "disp_plus1", "san_sync",
         "san_uninit"),
        start=1,
    )
}
H.load_plugin(MUTATIONS)

from sgl_nvidia_mega.kimi_k31.dsa import tsim as T  # noqa: E402

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)
W, PAGE, TOPK, Q = 16, 64, 2048, 8
POISON = np.float32(3.0e38)
ROWS = os.environ.get("K31_TSIM_ROWS", "")
_EMU = None


def emu():
    global _EMU
    if _EMU is None:
        _EMU = T.Emulator("cuda", cap=32, layers=24)
    return _EMU


def local_len(n, r):
    pages, rem = divmod(n, PAGE)
    return (pages + W - 1 - r) // W * PAGE + (rem if pages % W == r else 0)


def reference(rows, table):
    """[rows, 2048] int64 KV slots (exact top-min(n, 2048), source-major emission, -1 after) and the positions."""
    out = np.full((len(rows), TOPK), -1, np.int64)
    poss = []
    for i, s in enumerate(rows):
        n = s.size
        want = min(n, TOPK)
        pos = np.arange(n, dtype=np.int64)
        sel = np.lexsort((pos, -s.astype(np.float64)))[:want]
        e = sel[np.lexsort((sel, (sel // PAGE) % W))]
        poss.append(e)
        out[i, :want] = table[e // PAGE].astype(np.int64) * PAGE + e % PAGE
    return out, poss


SERVE_WIDTH = 1024  # local pages of the serving table at context 1,048,576 (16,384 pages / 16 ranks): 65,536 columns


def split(rows, extra_pages=0, width=None):
    """Local rows [16, rows, ncols] float32 (+POISON past each local length), local lengths [16, rows], ncols."""
    nmax = max(s.size for s in rows)
    width = width or -(-(-(-nmax // PAGE)) // W) + extra_pages
    ncols = width * PAGE
    X = np.full((W, len(rows), ncols), POISON, np.float32)
    NL = np.zeros((W, len(rows)), np.int32)
    for i, s in enumerate(rows):
        buf = np.zeros(width * W * PAGE, np.float32)
        buf[: s.size] = s
        loc = buf.reshape(width, W, PAGE)
        for r in range(W):
            nl = local_len(s.size, r)
            X[r, i, :nl] = loc[:, r, :].reshape(-1)[:nl]
            NL[r, i] = nl
    return X, NL, ncols


def run(rows, table, ordinal=0, advance=True, extra_pages=0, width=None):
    """All 16 ranks select over ``rows`` (list of float32 global rows of one request); returns out [16, rows, 2048],
    fallback [16, rows]."""
    e = emu()
    if advance:
        e.advance()
    X, NL, ncols = split(rows, extra_pages, width)
    R = len(rows)
    dev = "cuda"
    scores = torch.from_numpy(X).to(dev)
    lcausal = torch.from_numpy(NL).to(dev)
    causal = torch.tensor([s.size for s in rows], dtype=torch.int32, device=dev)
    # 64 spare zero entries: a mutant that maps a position one page row too far still reads inside the tensor
    tab = torch.from_numpy(np.concatenate([np.asarray(table, np.int32), np.zeros(64, np.int32)])).view(1, -1).to(dev)
    out = torch.full((W, R, TOPK), -7, dtype=torch.int32, device=dev)
    fb = torch.full((W, R), -9, dtype=torch.int32, device=dev)
    if os.environ.get("K31_MUTATION", "") == "san_uninit":  # initcheck positive control only: never-written scratch
        scratch = torch.empty((W, R, W * TOPK), dtype=torch.int64, device=dev)
    else:
        scratch = torch.zeros((W, R, W * TOPK), dtype=torch.int64, device=dev)
    e.select(ordinal, scores, lcausal, causal, tab, R, out, fb, scratch, ncols)
    torch.cuda.synchronize()
    return out.cpu().numpy(), fb.cpu().numpy(), ncols


def check(tag, out, fb, ref, *, planted=False):
    for r in range(W):
        bad = np.nonzero((out[r].astype(np.int64) != ref).any(axis=1))[0]
        if bad.size:
            i = int(bad[0])
            col = int(np.nonzero(out[r, i].astype(np.int64) != ref[i])[0][0])
            same = np.array_equal(np.sort(out[r, i].astype(np.int64)), np.sort(ref[i]))
            raise AssertionError(f"{tag}: rank {r}: {bad.size}/{ref.shape[0]} rows differ ({'order' if same else 'set'}); "
                                 f"row {i} first at {col}: got {out[r, i, col:col + 4].tolist()} want "
                                 f"{ref[i, col:col + 4].tolist()}; flags {fb[r].tolist()}")
        if planted:
            assert (fb[r] != 0).all(), f"{tag}: rank {r}: planted rows must take the fallback, flags {fb[r].tolist()}"
        else:
            assert (fb[r] == 0).all(), f"{tag}: rank {r}: the fast path must hold, flags {fb[r].tolist()}"
    H.record({"kind": "check", "item": tag, "cells": int(W * ref.size), "violations": 0,
              "flags": sorted(set(fb.reshape(-1).tolist()))})


def real_rows(D, L, b):
    mx = D["rowsA"].shape[2]
    rows = []
    for i in range(Q):
        n = L - 7 + i
        if L > mx:
            s = np.concatenate([D["rowsA"][b, i, : D["lenA"][b, i]], D["rowsB"][b, i, : D["lenB"][b, i]]])
            while s.size < n:  # rows beyond two captured forwards repeat them (the 1M cell only)
                s = np.concatenate([s, s])
            s = s[:n]
        else:
            s = D["rowsA"][b, i, :n]
        assert s.size == n
        rows.append(np.ascontiguousarray(s, np.float32))
    return rows


def data():
    if not ROWS or not os.path.exists(ROWS):
        pytest.skip("K31_TSIM_ROWS (pc2_rows.npz) not set")
    return np.load(ROWS)


def rand_table(pages, seed):
    g = np.random.default_rng(seed)
    return g.permutation(pages + 37)[:pages].astype(np.int32)


# ------------------------------------------------------------------------------------------------ real rows
@needs_gpu
@pytest.mark.parametrize("L", [131072, 262144, 393216, 524288])
def test_real(L):
    """Real captured index logits, 2 requests x 8 rows; ordinals 0 / 1 (parity) and a new forward between the
    requests (tags of the same parity re-used under a new epoch). Fast path everywhere (flags 0), exact."""
    D = data()
    table = D[f"table_{L}"]
    for b, (ordinal, advance) in enumerate(((0, True), (1, False))):
        rows = real_rows(D, L, b)
        ref, poss = reference(rows, table)
        for i in range(Q):  # the prep tool's stored reference (positions in emission order) must agree
            assert np.array_equal(poss[i], D[f"exp_{L}_{b}"][i][: poss[i].size]), (L, b, i)
        out, fb, ncols = run(rows, table, ordinal, advance)
        check(f"real_L{L}_b{b}_ord{ordinal}_g{T.groups(ncols)}", out, fb, ref)
    # the same rows again in a new forward, parity 0, with extra (poisoned) columns: kernel<16> when > 8 groups
    rows = real_rows(D, L, 1)
    ref, _ = reference(rows, table)
    out, fb, ncols = run(rows, table, 0, True, extra_pages=3)
    check(f"real_L{L}_b1_ord0_pad_g{T.groups(ncols)}", out, fb, ref)


@needs_gpu
@pytest.mark.parametrize("L", [560000, 667820, 786432, 1000000])
def test_long(L):
    """Rows of 9-16 float4 groups per thread (560000: exactly 9 groups, the first row length past the 8-group body) (the SMEM-staged groups; long-context inputs reach this range): two captured
    forwards concatenated (repeated for 1M). Fast path, exact."""
    D = data()
    for b in range(2):
        rows = real_rows(D, L, b)
        table = rand_table(-(-L // PAGE), 900 + b + L)
        ref, _ = reference(rows, table)
        out, fb, ncols = run(rows, table, b, b == 0)
        check(f"long_L{L}_b{b}_g{T.groups(ncols)}", out, fb, ref)


# ------------------------------------------------------------------------------------------------ serving width
@needs_gpu
@pytest.mark.parametrize("L", [131072, 262144, 393216, 524288, 0])
def test_serve(L):
    """The serving launch shape: every row at the static table width (65,536 local columns, tsim_kernel<16>), so
    rows of 1-8 groups run the one 16-group body with the uniform skip. L = 0: the short rows of test_short."""
    if L:
        D = data()
        table = D[f"table_{L}"]
        rows = real_rows(D, L, 0)
    else:
        g = np.random.default_rng(5)
        rows = [g.standard_normal(n).astype(np.float32) * 3 for n in [1, 64, 65, 2047, 2048, 2049, 4097, 70000]]
        table = rand_table(-(-70000 // PAGE), 6)
    ref, _ = reference(rows, table)
    out, fb, ncols = run(rows, table, 1, True, width=SERVE_WIDTH)
    assert T.groups(ncols) == 16, ncols
    check(f"serve_L{L}_g{T.groups(ncols)}", out, fb, ref)


@needs_gpu
def test_wide_table():
    """A table wider than 16 groups (16,400 pages = 1025 local pages = 65,600 columns: TsiState sizes max_pages = ceil(ctx / 64) + 16
    at context 1,048,576) is accepted (kernel<16>); real 524K rows stay on the fast path, exact."""
    D = data()
    table = D["table_524288"]
    rows = real_rows(D, 524288, 1)
    ref, _ = reference(rows, table)
    out, fb, ncols = run(rows, table, 0, True, width=1025)
    assert T.groups(ncols) == 17, ncols
    check(f"wide_g{T.groups(ncols)}", out, fb, ref)


@needs_gpu
def test_too_long():
    """Rows longer than 1,048,576 tokens (local rows past 16 groups): every rank must take the exact fallback (sender overflow), never
    truncate the row. Random scores, 1,100,000 - j tokens."""
    g = np.random.default_rng(13)
    lens = [1_100_000 - 977 * j for j in range(Q)]
    rows = [g.standard_normal(n).astype(np.float32) for n in lens]
    table = rand_table(-(-max(lens) // PAGE), 14)
    ref, _ = reference(rows, table)
    out, fb, ncols = run(rows, table, 1, True)
    assert T.groups(ncols) == 17, ncols
    check(f"too_long_g{T.groups(ncols)}", out, fb, ref, planted=True)


@needs_gpu
def test_serving_refuses_test_mode():
    """Binding::run (the serving entry, tsi.py) must refuse meta[7] != 0: the no-trap test mode would turn a missing
    record into flag 8 instead of a trap (fail-open, G2V.2 point 3). Benchmarks use Binding::bench."""
    e = emu()
    meta = e.meta(0)
    assert int(meta[7]) == 2
    z32 = torch.zeros((1, 2048), dtype=torch.int32, device="cuda")
    with pytest.raises(Exception, match="test_mode must be 0"):
        e.module.run(meta, e.epoch, torch.zeros((1, 4096), device="cuda"), z32[:, :1], z32[0, :1], z32, z32,
                     z32[:, :1], torch.zeros((1, W * TOPK), dtype=torch.int64, device="cuda"), 8, 4096)


# ------------------------------------------------------------------------------------------------ synthetic rows
@needs_gpu
def test_short():
    """Rows of 1 .. 70,000 tokens (complete sources, the slow pass, partial last pages), random scores."""
    g = np.random.default_rng(5)
    lens = [1, 64, 65, 2047, 2048, 2049, 4097, 70000]
    rows = [g.standard_normal(n).astype(np.float32) * 3 for n in lens]
    table = rand_table(-(-max(lens) // PAGE), 6)
    ref, _ = reference(rows, table)
    out, fb, _ = run(rows, table, 1, True)
    check("short", out, fb, ref)


@needs_gpu
def test_ties():
    """Scores from 37 levels: every cut splits a group of equal scores; order by position. Exact (flags recorded:
    the boundary bin may exceed the tie list, which takes the fallback)."""
    g = np.random.default_rng(7)
    levels = g.standard_normal(37).astype(np.float32)
    rows = [levels[g.integers(0, 37, 300000 - j)] for j in range(Q)]
    table = rand_table(-(-300000 // PAGE), 8)
    ref, _ = reference(rows, table)
    out, fb, _ = run(rows, table, 0, True)
    for r in range(W):
        assert np.array_equal(out[r].astype(np.int64), ref), f"ties: rank {r} differs, flags {fb[r].tolist()}"
    H.record({"kind": "check", "item": "ties", "cells": int(W * ref.size), "violations": 0,
              "flags": sorted(set(fb.reshape(-1).tolist()))})


@needs_gpu
@pytest.mark.parametrize("kind", ["constant", "signed_zero", "boost"])
def test_planted(kind):
    """Rows that must take the exact fallback: constant rows (every source overflows), all-signed-zero rows (-0.0 and
    +0.0 are equal scores: the order is by position), and one rank (3) holding the whole top-2048 (its certificate
    fails; its FULL list supplies all 2048 keys)."""
    g = np.random.default_rng(11)
    n0 = 300000
    lens = [n0 - j for j in range(Q)]
    if kind == "constant":
        rows = [np.full(n, 1.5, np.float32) for n in lens]
    elif kind == "signed_zero":
        rows = [np.where(g.random(n) < 0.5, np.float32(-0.0), np.float32(0.0)).astype(np.float32) for n in lens]
    else:
        rows = []
        for n in lens:
            s = g.standard_normal(n).astype(np.float32)
            pos = np.arange(n)
            s[(pos // PAGE) % W == 3] += np.float32(100.0)
            rows.append(s)
    table = rand_table(-(-n0 // PAGE), 12)
    ref, _ = reference(rows, table)
    out, fb, _ = run(rows, table, 1, True)
    check(f"planted_{kind}", out, fb, ref, planted=True)
