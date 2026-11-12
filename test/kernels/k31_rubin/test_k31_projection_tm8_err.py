# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""TM8 error against float64, compared with the served MLA in-projection, and TM8 at the 3040-row width.

(1) ``test_tm8_error_vs_served``: at the K3.1 TP16 per-rank shapes [2880, 7168] (the TARGET_VERIFY build) and [3008, 7168]
(the decode build) and 8 rows, TM8's error against ``x.double() @ W.double().T`` must be no worse than the served tactic's
(``projection_tactic("mla", 8, width=n)`` with every projection knob unset). Tolerances (stated before any run): mean |TM8 - ref| <= 1.05 x mean |served - ref|, and
max |TM8 - ref| <= 1.5 x max |served - ref|. Both outputs are one BF16 rounding of an FP32 sum, so their errors should
match to within the summation-order term.
(2) ``test_tm8_width``: the same kernel at 2880 rows (23 M tiles) and 3040 rows (2880 + index-K 128 + head gate 32, as
packed under K31MK_DSA_QF): every cell within 1e-3 + 2^-7 |ref| of float64, BITWISE equal to SplitKGemm (128, 8, 7, 5) at that width,
and no write outside the output.

Mutations (each must make both tests fail): ``rank_sum`` drops rank 1's partial from the sum; ``acc_reset`` restarts
the accumulator at every tensor-memory-staged K tile (drops the earlier tiles' products).
"""

import pytest
import torch

import _harness as H

MUTATIONS = {
    "rank_sum": (
        "kimi_k31/k3/cutedsl/gemm/splitk_tmem8.py",
        "for s in cutlass.range_constexpr(1, self.split_k):",
        "for s in cutlass.range_constexpr(2, self.split_k):",
    ),
    "acc_reset": (
        "kimi_k31/k3/cutedsl/gemm/splitk_tmem8.py",
        "accumulate = Int32(0 if k == 0 and kb == 0 else 1)\n                        a = base + self.n + columns * k + 8 * kb",
        "accumulate = Int32(0 if kb == 0 else 1)\n                        a = base + self.n + columns * k + 8 * kb",
    ),
}

K, ROWS = 7168, 8
TM8 = (128, 8, 7, 12, 15, 16, 4)
SAME_ORDER = (128, 8, 7, 5)

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
@pytest.mark.parametrize("n", [2880, 3008])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tm8_error_vs_served(seed, n, monkeypatch):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.gemm import projection as P

    for knob in ("K31MK_PROJ_TACTICS", "K31MK_PROJ_TMEM8", "K31MK_PROJ_TMEM64", "K31MK_PROJ_SM107_ENT", "K31MK_PROJ_SM107_TP8"):
        monkeypatch.delenv(knob, raising=False)
    P.tactic_overrides.cache_clear()
    dev = torch.device("cuda", torch.cuda.current_device())
    served = P.projection_tactic("mla", ROWS, dev, width=n)
    assert len(served) == 4, served
    g = H.generator(3100 + 29 * seed)
    weight = H.randn((n, K), 0.02, g)
    x = H.randn((ROWS, K), 1.0, g)
    ref = x.double() @ weight.double().T
    out_tm, guard = _run(P.PreparedProjection(weight, {ROWS: TM8}, fenced_trigger=2).prepare(), x, ROWS, n)
    out_sv, _ = _run(P.PreparedProjection(weight, {ROWS: served}, fenced_trigger=2).prepare(), x, ROWS, n)
    assert guard == 0
    e_tm, e_sv = (out_tm.double() - ref).abs(), (out_sv.double() - ref).abs()
    print(
        f"TM8_VS_SERVED n{n} s{seed} served {served} mean {float(e_tm.mean()):.4e} vs {float(e_sv.mean()):.4e} "
        f"max {float(e_tm.max()):.4e} vs {float(e_sv.max()):.4e} cells_differ "
        f"{int((out_tm.view(torch.int16) != out_sv.view(torch.int16)).sum())}"
    )
    assert float(e_tm.mean()) <= 1.05 * float(e_sv.mean()), "TM8 mean error worse than the served tactic's"
    assert float(e_tm.max()) <= 1.5 * float(e_sv.max()), "TM8 max error worse than the served tactic's"


@needs_rubin
@pytest.mark.parametrize("fenced", [2, 0])
@pytest.mark.parametrize("n", [2880, 3040])
def test_tm8_width(n, fenced):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.gemm.projection import PreparedProjection

    g = H.generator(3200 + n + fenced)
    weight = H.randn((n, K), 0.02, g)
    x = H.randn((ROWS, K), 1.0, g)
    ref = x.double() @ weight.double().T
    out, guard = _run(PreparedProjection(weight, {ROWS: TM8}, fenced_trigger=fenced).prepare(), x, ROWS, n)
    base, _ = _run(PreparedProjection(weight, {ROWS: SAME_ORDER}, fenced_trigger=fenced).prepare(), x, ROWS, n)
    assert guard == 0, f"TM8 wrote {guard} cells outside its output at width {n}"
    H.check(f"tm8.mla{n}.f{fenced}", out, ref.float(), 1e-3, 2**-7)
    diff = int((out.view(torch.int16) != base.view(torch.int16)).sum())
    print(f"TM8_W{n} f{fenced} cells_vs_{SAME_ORDER} {diff}")
    assert diff == 0, f"TM8 differs from SplitKGemm {SAME_ORDER} at width {n} in {diff} cells"
