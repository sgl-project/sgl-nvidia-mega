# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""TM8: ``SplitKTmemGemm8`` (``gemm/splitk_tmem8.py``, 7-field tactics) for the MLA in-projection at the K3.1
TP16 per-rank shape [3008, 7168] and 8 rows (Breq 1 x q 8), both trigger modes (2 = warp 3 after its wait, fenced;
0 = the weight warp once every A tile is requested).

Reference: ``x.double() @ W.double().T`` (float64, independent of every kernel). Tolerances (stated before any run):
(1) every cell |out - ref| <= 1e-3 + 2^-7 |ref| (as test_k31_projection.py: one BF16 rounding of an FP32 sum);
(2) BITWISE equal to SplitKGemm at (128, 8, 7, 5): the same K16 MMAs over the same K slices in the same order and the
same rank-order sum of the split partials, so no rounding may differ; (3) guard regions: rows past ``rows`` and columns
past N of the output buffer keep their poison; (4) graph replay == eager, bitwise.

Mutations (each must make a test fail): ``stage_col`` copies every staged A tile into tensor-memory tile 0;
``owner_row`` stores every owner's share at the tile's first rows; ``rank_sum`` drops rank 1's partial from the sum;
``tm8_kda`` serves TM8 for the KDA in-projection too (caught by the routing test).
"""

import pytest
import torch

import _harness as H

MUTATIONS = {
    "stage_col": (
        "kimi_k31/k3/cutedsl/gemm/splitk_tmem8.py",
        "_copy_to_tmem(base + self.n + columns * k + 8 * kb, source)",
        "_copy_to_tmem(base + self.n + columns * 0 + 8 * kb, source)",
    ),
    "owner_row": (
        "kimi_k31/k3/cutedsl/gemm/splitk_tmem8.py",
        "row = row0 + split_rank * share + r",
        "row = row0 + 0 * share + r",
    ),
    "rank_sum": (
        "kimi_k31/k3/cutedsl/gemm/splitk_tmem8.py",
        "for s in cutlass.range_constexpr(1, self.split_k):",
        "for s in cutlass.range_constexpr(2, self.split_k):",
    ),
    "tm8_kda": (
        "kimi_k31/k3/cutedsl/gemm/projection.py",
        'SM107_TMEM8 = {"mla": (128, 8, 7, 12, 15, 16, 4)}',
        'SM107_TMEM8 = {"mla": (128, 8, 7, 12, 15, 16, 4), "kda": (128, 8, 7, 12, 15, 16, 4)}',
    ),
}

N, K, ROWS = 3008, 7168, 8
TM8 = (128, 8, 7, 12, 15, 16, 4)
SAME_ORDER = (128, 8, 7, 5)  # SplitKGemm with the same split: the bitwise reference form

needs_rubin = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7), reason="TM8 is an SM107 kernel"
)


def _run(proj, x, rows, n, pad_rows=8, pad_cols=64):
    buf = torch.empty((rows + pad_rows, n + pad_cols), dtype=torch.bfloat16, device="cuda")
    H.poison(buf)
    keep = buf.clone()
    out = buf[:rows, :n]
    proj.run(x, out)
    torch.cuda.synchronize()
    guard_rows = int((buf[rows:].view(torch.int16) != keep[rows:].view(torch.int16)).sum())
    guard_cols = int((buf[:rows, n:].view(torch.int16) != keep[:rows, n:].view(torch.int16)).sum())
    return out.clone(), guard_rows + guard_cols


@needs_rubin
@pytest.mark.parametrize("fenced", [2, 0])
@pytest.mark.parametrize("seed", [0, 1])
def test_tm8_projection(seed, fenced):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.gemm.projection import PreparedProjection

    g = H.generator(2800 + 17 * seed)
    weight = H.randn((N, K), 0.02, g)
    x = H.randn((ROWS, K), 1.0, g)
    ref = x.double() @ weight.double().T
    tm = PreparedProjection(weight, {ROWS: TM8}, fenced_trigger=fenced).prepare()
    same = PreparedProjection(weight, {ROWS: SAME_ORDER}, fenced_trigger=fenced).prepare()
    out, guard = _run(tm, x, ROWS, N)
    base, _ = _run(same, x, ROWS, N)
    assert guard == 0, f"TM8 wrote {guard} cells outside its output"
    H.check(f"tm8.mla.s{seed}.f{fenced}", out, ref.float(), 1e-3, 2**-7)
    diff = int((out.view(torch.int16) != base.view(torch.int16)).sum())
    d = (out.double() - ref).abs()
    print(f"TM8_ERR s{seed} f{fenced} max {float(d.max()):.3e} mean {float(d.mean()):.3e} cells_vs_{SAME_ORDER} {diff}")
    assert diff == 0, f"TM8 differs from SplitKGemm {SAME_ORDER} in {diff} cells (same MMA order expected)"
    # graph replay == eager (bitwise)
    outs = torch.empty_like(out)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        tm.run(x, outs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tm.run(x, outs)
    H.poison(outs)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(outs.view(torch.int16), out.view(torch.int16)), "graph replay differs from eager"


@needs_rubin
def test_tm8_routing(monkeypatch):
    """K31MK_PROJ_TMEM8=1 serves TM8 for the MLA in-projection at 8 rows and the TP16 width only; unset, the tables are
    unchanged; a 7-field K31MK_PROJ_TACTICS entry parses; invalid TM8 entries raise."""
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.gemm import projection as P

    dev = torch.device("cuda", torch.cuda.current_device())
    monkeypatch.delenv("K31MK_PROJ_TACTICS", raising=False)
    monkeypatch.delenv("K31MK_PROJ_TMEM8", raising=False)
    monkeypatch.delenv("K31MK_PROJ_TMEM64", raising=False)
    kinds, rows = ("kda", "mla"), (1, 8, 16, 64)
    off = {(k, r): P.projection_tactic(k, r, dev) for k in kinds for r in rows}
    off_wide = P.projection_tactic("mla", 8, dev, width=6016)  # another width (TP8-like): must keep its table entry
    assert all(len(t) == 4 for t in off.values()) and len(off_wide) == 4, (off, off_wide)
    monkeypatch.setenv("K31MK_PROJ_TMEM8", "1")
    for k in kinds:
        for r in rows:
            got = P.projection_tactic(k, r, dev)
            want = TM8 if (k, r) == ("mla", 8) else off[k, r]
            assert got == want, (k, r, got, want)
    for w in (2880, 3008, 3040):  # verify build, decode build, verify build under K31MK_DSA_QF
        assert P.projection_tactic("mla", 8, dev, width=w) == TM8, w
    assert P.projection_tactic("mla", 8, dev, width=6016) == off_wide
    P.tactic_overrides.cache_clear()
    assert P.tactic_overrides("mla:8=128.8.7.12.15.16.4") == {("mla", 8): TM8}
    for bad in ((128, 16, 7, 12, 15, 16, 4), (128, 8, 6, 12, 15, 16, 4), (128, 8, 7, 12, 15, 8, 4), (128, 8, 7, 17, 15, 16, 4)):
        with pytest.raises(ValueError):
            P.tmem8_check(bad, K)
