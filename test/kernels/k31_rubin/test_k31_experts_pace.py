# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 persistent experts with the producer pacer (``K31MK_EXPERTS_PRODUCER_PACE=<cycles>``, default 0):
after each weight stage's copies the producer lane spins that many SM clocks. No memory or barrier operation changes, so the
outputs must be bit-identical. Applied to the packed K64 engine's TP16 B4-8 stream calls only.

Checks (stated before any run; never loosened):
  P1  paced vs unpaced PersistentExperts (same weights, x, ids, one process): 0 differing BF16 output cells on every call
      (2 eager + 2 CUDA-graph replays) at B4 and B8, packed K64 with 6 and 7 slots, K-17 lean on (the Run form) [SM107].
  P2  engagement: at B4 / B8 packed the paced engine runs a different compiled kernel; unpacked B8 and packed B1 / B16 are
      inert (the same compiled kernel object as the unpaced engine).
  P3  ``K31MK_EXPERTS_PRODUCER_PACE`` rejects negative values (no GPU).
  T1  paced rows vs the FP64 reference (``_ref_moe16.expert_rows``).

Mutations (fixed harness; each must make a base-passing test fail):
  pace_inert     prepare never passes the pace to the kernel                                  [P2]
  pace_unpacked  prepare also paces the unpacked engine                                       [P2 unpacked]
  pace_neg       the env check accepts negative values                                        [P3]
"""

import os

import pytest
import torch

import _harness as H
import _ref_moe as R
import _ref_moe16 as M
import test_k31_experts_tp16 as T16

RUNTIME = "kimi_k31/k3/cutedsl/moe/experts_runtime.py"
MUTATIONS = {
    "pace_inert": (RUNTIME, "situ_lean=self.situ_lean, strong=self.strong, x3_release=x3, producer_pace=pace,",
                   "situ_lean=self.situ_lean, strong=self.strong, x3_release=x3, producer_pace=0,"),
    "pace_unpacked": (RUNTIME,
                      "pace = self.pace if self.tp == 16 and batch <= 8 and streamed(batch) and packed else 0  # cand-08",
                      "pace = self.pace if self.tp == 16 and batch <= 8 and streamed(batch) else 0  # cand-08"),
    "pace_neg": (RUNTIME, "    if value < 0:\n        raise ValueError(f\"K31MK_EXPERTS_PRODUCER_PACE={value}: cycles >= 0\")\n", ""),
}
needs_sm107 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="packed K64 experts need SM107 (VR200)",
)


def _er():
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import experts_runtime as ER

    return ER


def _env(name, value):
    old = os.environ.get(name)
    os.environ[name] = value
    return old


def _restore(name, old):
    if old is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = old


def engine(prepared, batch, pace, packed):
    ER = _er()
    old = _env("K31MK_EXPERTS_PRODUCER_PACE", str(pace))
    try:
        ex = ER.PersistentExperts(*prepared, beta=R.BETA, linear_beta=R.LIN, activation_floor=None, packed=packed,
                                  strong=True)
    finally:
        _restore("K31MK_EXPERTS_PRODUCER_PACE", old)
    assert ex.pace == pace
    out = torch.empty((batch * 16, R.H_LAT), dtype=torch.bfloat16, device="cuda")
    ex.prepare(batch, ER.scratch(ex.tp, torch.device("cuda")), out)
    return ex, out


def run_pair(batch, packed, seed, pace=240):
    H.load_plugin(MUTATIONS)
    logical, prepared = T16.weights16()
    (ex_a, out_a), (ex_p, out_p) = engine(prepared, batch, 0, packed), engine(prepared, batch, pace, packed)
    same = ex_a.prepared[batch][3] is ex_p.prepared[batch][3]
    engaged = bool(packed) and 4 <= batch <= 8
    H.record({"kind": "check", "item": f"pace.engaged.b{batch}.p{packed}", "same_kernel": same, "engaged": engaged})
    assert same != engaged, f"b{batch} p{packed}: paced engine shares the unpaced kernel = {same} (engaged {engaged})"
    g = H.generator(seed + batch)
    x = torch.empty((batch, R.H_LAT), device="cuda")
    ids = torch.empty((batch, 16), dtype=torch.int32, device="cuda")

    def call():
        ex_a.run(x, ids)
        ex_p.run(x, ids)

    graph = None
    tag = f"pace.p{packed}.b{batch}"
    for n in range(4):
        xv = T16.latent(batch, g)
        i, _ = R.random_route(batch, g)
        x.copy_(xv)
        ids.copy_(i)
        H.poison(out_a, out_p)
        if n < 2:
            call()
        else:
            graph = graph or T16.graph_of(call)
            H.poison(out_a, out_p)
            graph.replay()
        torch.cuda.synchronize()
        differ = int((out_p.view(torch.int16) != out_a.view(torch.int16)).sum())
        H.record({"kind": "check", "item": f"{tag}.call{n}", "cells_differ": differ, "cells": out_a.numel()})
        assert differ == 0, f"{tag}.call{n}: paced and unpaced differ in {differ} cells"
        T16.check_rows(f"{tag}.call{n}", out_p.view(batch, 16, -1), M.expert_rows(logical, xv, i))


@needs_sm107
@pytest.mark.parametrize("batch", (4, 8), ids=lambda b: f"b{b}")
@pytest.mark.parametrize("packed", (6, 7), ids=lambda p: f"p{p}")
def test_pace_packed(packed, batch):
    """P1 + P2 + T1 on the packed engine (K-17 lean on)."""
    run_pair(batch, packed, 1770)


@needs_sm107
@pytest.mark.parametrize("batch,packed", ((8, 0), (1, 7), (16, 7)), ids=("b8p0", "b1p7", "b16p7"))
def test_pace_inert(batch, packed):
    """P2 inert side: unpacked or not a stream call -> the same kernel as unpaced."""
    run_pair(batch, packed, 1870)


def test_pace_env():
    """P3 (no GPU)."""
    H.load_plugin(MUTATIONS)
    ER = _er()
    old = _env("K31MK_EXPERTS_PRODUCER_PACE", "240")
    try:
        assert ER.producer_pace() == 240
        os.environ["K31MK_EXPERTS_PRODUCER_PACE"] = "-1"
        with pytest.raises(ValueError):
            ER.producer_pace()
    finally:
        _restore("K31MK_EXPERTS_PRODUCER_PACE", old)
