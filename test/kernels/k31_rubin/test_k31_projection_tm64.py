# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""TM64: ``SplitKTmemGemm64`` projections (``gemm/splitk_tmem64.py``, 5-field tactics) at the
K3.1 TP16 per-rank shapes (KDA [3216, 7168] split 6, MLA [3008, 7168] split 7) and 16 / 24 / 32 / 64 rows, both
trigger modes (2 = warp 3 after its wait, fenced; 0 = the original weight-warp trigger).

Reference: ``x.double() @ W.double().T`` (float64, independent of every kernel). Tolerances (stated before any run):
(1) every cell |out - ref| <= 1e-3 + 2^-7 |ref| (as test_k31_projection.py: one BF16 rounding of an FP32 sum);
(2) outcome envelope vs the served path: TM64's max and mean |out - ref| at most the served SplitKGemm tactic's
(same inputs) plus one BF16 ulp of max |ref| (max) and x 1.25 + 1e-6 (mean) -- the K-slice reorder may move a
cell by one rounding, not more; (3) guard regions: rows past ``rows`` and columns past N of the output buffer keep
their poison (the kernel's bounded stores); (4) graph replay == eager, bitwise.

Mutations (each must make a test fail): ``k_tail`` drops the extra K tile of the ranks below K % split; ``owner_row``
stores every owner's share at the tile's first rows; ``acc_first`` accumulates into stale TMEM at the first MMA;
``tm64_rows`` serves TM64 at 9-64 rows again (the rows table) -- caught by the routing test.
"""

import os

import pytest
import torch

import _harness as H

MUTATIONS = {
    "k_tail": (
        "kimi_k31/k3/cutedsl/gemm/splitk_tmem64.py",
        "tiles = base_tiles + Int32(split_rank < self.extra)",
        "tiles = base_tiles + Int32(0)",
    ),
    "owner_row": (
        "kimi_k31/k3/cutedsl/gemm/splitk_tmem64.py",
        "row = row0 + split_rank * share",
        "row = row0 + 0 * share",
    ),
    "acc_first": (
        "kimi_k31/k3/cutedsl/gemm/splitk_tmem64.py",
        "accumulate = Int32(0 if k == 0 and kb == 0 else 1)",
        "accumulate = Int32(1)",
    ),
    "tm64_rows": (
        "kimi_k31/k3/cutedsl/gemm/projection.py",
        "TMEM64_ROWS = range(64, 65)",
        "TMEM64_ROWS = range(9, 65)",
    ),
}

SHAPES = {"kda": 3216, "mla": 3008}
TM64 = {"kda": (128, 64, 6, 14, 14), "mla": (128, 64, 7, 16, 14)}
SERVED = {("kda", 16): (64, 16, 2, 11), ("kda", 24): (64, 32, 2, 9), ("kda", 32): (64, 32, 2, 9), ("kda", 64): (64, 64, 2, 6),
          ("mla", 16): (128, 16, 4, 5), ("mla", 24): (64, 32, 2, 8), ("mla", 32): (64, 32, 2, 8), ("mla", 64): (64, 64, 2, 6)}
K = 7168

needs_rubin = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7), reason="TM64 is an SM107 kernel"
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


def _err(out, ref):
    d = (out.double() - ref).abs()
    return float(d.max()), float(d.mean())


@needs_rubin
@pytest.mark.parametrize("fenced", [2, 0])
@pytest.mark.parametrize("rows", [16, 24, 32, 64])
@pytest.mark.parametrize("kind", ["kda", "mla"])
def test_tm64_projection(kind, rows, fenced):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.gemm.projection import PreparedProjection

    n = SHAPES[kind]
    g = H.generator(1700 + rows + (n % 97))
    weight = H.randn((n, K), 0.02, g)
    x = H.randn((rows, K), 1.0, g)
    ref = x.double() @ weight.double().T
    tm = PreparedProjection(weight, {rows: TM64[kind]}, fenced_trigger=fenced).prepare()
    served = PreparedProjection(weight, {rows: SERVED[kind, rows]}).prepare()
    out, guard = _run(tm, x, rows, n)
    base, _ = _run(served, x, rows, n)
    assert guard == 0, f"TM64 wrote {guard} cells outside its output"
    H.check(f"tm64.{kind}.r{rows}.f{fenced}", out, ref.float(), 1e-3, 2**-7)
    (tmax, tmean), (smax, smean) = _err(out, ref), _err(base, ref)
    ulp = float(ref.abs().max()) * 2**-7  # one BF16 ulp at the largest |ref| (upper bound)
    print(f"TM64_ERR {kind} r{rows} f{fenced} max {tmax:.3e} mean {tmean:.3e} | served {SERVED[kind, rows]} max {smax:.3e} mean {smean:.3e}")
    assert tmax <= smax + ulp, (tmax, smax, ulp)
    assert tmean <= 1.25 * smean + 1e-6, (tmean, smean)
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
def test_tm64_routing(monkeypatch):
    """K31MK_PROJ_TMEM64=1 serves TM64 at 64 rows only; unset, the SplitKGemm tables are unchanged."""
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.gemm import projection as P

    dev = torch.device("cuda", torch.cuda.current_device())
    monkeypatch.delenv("K31MK_PROJ_TACTICS", raising=False)
    monkeypatch.delenv("K31MK_PROJ_TMEM64", raising=False)
    off = {(k, r): P.projection_tactic(k, r, dev) for k in SHAPES for r in (8, 16, 32, 64, 128)}
    assert all(len(t) == 4 for t in off.values()), off
    monkeypatch.setenv("K31MK_PROJ_TMEM64", "1")
    for k in SHAPES:
        for r in (8, 16, 32, 64, 128):
            got = P.projection_tactic(k, r, dev)
            want = TM64[k] if r == 64 else off[k, r]
            assert got == want, (k, r, got, want)
