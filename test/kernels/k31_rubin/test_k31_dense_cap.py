# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The first dense MLP (moe/dense.py PreparedDenseMLP) at a 128-row verify batch (Breq 16).

test_dense_128_rows_equal_two_64_row_runs: built for batches (8, 64, 128), the 128-row run must give, row for row, the
bits of two 64-row runs over its halves (128 rows take the 64-row tactics over two row tiles), and every run must sit
within rel-L2 1e-2 of an FP64 reference of the same method (BF16 gate/up, SiTU, BF16 hidden, down; the GEMMs accumulate
in FP32, so 1e-2 bounds the BF16 roundings, not the method).
test_dense_ragged_rows_match_the_box: built for (8, 64, 72, 96, 128), a ragged 72- or 96-row run (a partial second row
tile under the 128-row tactic) must give the bits of the 128-row run's first rows (same tactic; GEMM rows are independent),
sit within rel-L2 1e-2 of the FP64 reference, and leave the caller's rows past b untouched (sentinel).
Mutations (must fail): dense_rows64 (scratch and tiles stop at 64 rows, as before CAP-128 #12); dense_full_tiles
(run() covers only whole 64-row tiles of a ragged batch: a no-op at 8 / 64 / 128 rows, so only the ragged test sees it).

Needs one GPU."""

import os
from types import SimpleNamespace as NS

import pytest
import torch

import _harness as H
from _ref_moe import situ

MUTATIONS = {
    "dense_rows64": [
        ("kimi_k31/k3/cutedsl/moe/dense.py", "rows = max(64, *batches)", "rows = 64"),
        ("kimi_k31/k3/cutedsl/moe/dense.py", "for n in (8, 16, 32, 64, 128) if b <= n", "for n in (8, 16, 32, 64) if b <= n"),
    ],
    "dense_full_tiles": ("kimi_k31/k3/cutedsl/moe/dense.py",
                         "        gate, hidden = self.gates[: x.shape[0]], self.hidden_rows[: x.shape[0]]",
                         "        m = x.shape[0] // 64 * 64 or x.shape[0]\n"
                         "        gate, hidden = self.gates[:m], self.hidden_rows[:m]"),
}
HIDDEN, MODEL = 4224, 7168  # k31_build_smoke.py's fake layer-0 dense MLP (TP8 geometry)


def _mlp(dev):
    g = torch.Generator(device=dev).manual_seed(1234)
    w = lambda o, i: (torch.randn((o, i), generator=g, device=dev) * i ** -0.5).to(torch.bfloat16)
    return NS(gate_up_proj=NS(weight=w(2 * HIDDEN, MODEL)), down_proj=NS(weight=w(MODEL, HIDDEN)),
              act_fn=NS(beta=4.0, linear_beta=25.0))


def _reference(mlp, x):
    gu = (x.double() @ mlp.gate_up_proj.weight.double().T).to(torch.bfloat16).double()
    h = situ(gu[:, :HIDDEN], gu[:, HIDDEN:]).to(torch.bfloat16).double()
    return h @ mlp.down_proj.weight.double().T


def test_dense_128_rows_equal_two_64_row_runs():
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import dense as D

    dev = torch.device("cuda", 0)
    mlp = _mlp(dev)
    prepared = D.PreparedDenseMLP(mlp, (8, 64, 128))
    x = (torch.randn((128, MODEL), generator=torch.Generator(device=dev).manual_seed(99), device=dev)).to(torch.bfloat16)
    out = torch.empty((128, MODEL), dtype=torch.bfloat16, device=dev)
    halves = torch.empty_like(out)
    prepared.run(x, out)
    prepared.run(x[:64], halves[:64])
    prepared.run(x[64:], halves[64:])
    torch.cuda.synchronize()
    differ = int((out.view(torch.int16) != halves.view(torch.int16)).sum())
    ref = _reference(mlp, x)
    rel = lambda y: float((y.double() - ref).norm() / ref.norm())
    print(f"DENSE128 differing_bits_cells={differ} rel_l2_128={rel(out):.3e} rel_l2_halves={rel(halves):.3e}", flush=True)
    assert rel(halves) < 1e-2, "the 64-row path itself is off the FP64 reference"
    assert rel(out) < 1e-2
    assert differ == 0, f"{differ} cells of the 128-row run differ from the two 64-row runs"


def test_dense_ragged_rows_match_the_box():
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import dense as D

    dev = torch.device("cuda", 0)
    mlp = _mlp(dev)
    prepared = D.PreparedDenseMLP(mlp, (8, 64, 72, 96, 128))
    x = (torch.randn((128, MODEL), generator=torch.Generator(device=dev).manual_seed(77), device=dev)).to(torch.bfloat16)
    full = torch.empty((128, MODEL), dtype=torch.bfloat16, device=dev)
    prepared.run(x, full)
    ref = _reference(mlp, x)
    sentinel = torch.tensor(-7.0, dtype=torch.bfloat16).view(torch.int16).item()
    bad = []
    for b in (72, 96):
        buf = torch.full((128, MODEL), -7.0, dtype=torch.bfloat16, device=dev)
        prepared.run(x[:b], buf[:b])
        torch.cuda.synchronize()
        differ = int((buf[:b].view(torch.int16) != full[:b].view(torch.int16)).sum())
        touched = int((buf[b:].view(torch.int16) != sentinel).sum())
        rel = float((buf[:b].double() - ref[:b]).norm() / ref[:b].norm())
        print(f"DENSE_RAGGED b={b} differing_vs_128_cells={differ} touched_past_b={touched} rel_l2={rel:.3e}", flush=True)
        if differ or touched or not rel < 1e-2:
            bad.append((b, differ, touched, rel))
    assert not bad, f"ragged dense runs (b, cells differing from the 128-row run, cells written past b, rel-L2): {bad}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"] + ([] if not os.environ.get("K31_MUTATION") else ["-x"])))
