# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Split-K projection tactic sweep for VR200 (same kernel, same gate as
``test_k31_projection.py``; only the (M tile, N tile, split K, stages) policy differs).

The GB300 tactics launch 59 CTAs for the MLA projection at B1 (3776 / 64, no split) and 100
for KDA (50 x split 2), i.e. a minority of VR200's SMs, and run measurably above the HBM
floor. Candidates: M tile 64/128, N tile = the batch's tile (8/16/32/64), split
K in {1, 2, 4, 7, 8} (divisors of the 56 K tiles, cluster <= 8), stages filling the SMEM allocation,
keeping grids within one wave (one CTA per SM). Every candidate must pass the
projection gate (|out - ref| <= 1e-3 + 2^-7 |ref|, FP32 reference) and is timed cold-L2
(K31_TIME=1); the current tactic is timed in the same process as the baseline.
"""

import os

import pytest
import torch

import _harness as H
from test_k31_projection import KDA_CANDIDATES, SHAPES, needs_gpu, gb300_tactic as current_tactic

SMEM = 232448  # per-block opt-in shared-memory limit of CC 10.x (CUDA C++ Programming Guide)
SMS = torch.cuda.get_device_properties(0).multi_processor_count if torch.cuda.is_available() else 0
K_TILES = 7168 // 128


def stages_for(m, n, split):
    stage = (m + n) * 128 * 2
    mailbox = (split - 1) * m * n * 4
    return min(16, (SMEM - mailbox - 2048) // stage)


def candidates(kind, batch):
    rows = SHAPES[kind]
    tile = next(t for t in (8, 16, 32, 64) if batch <= t)
    out = {current_tactic(kind, batch)}
    for m in (64, 128):
        for split in (1, 2, 4, 7, 8):
            if K_TILES % split:
                continue
            ctas = -(-rows // m) * split
            if not 96 <= ctas <= SMS:
                continue
            stages = stages_for(m, tile, split)
            if stages >= 3:
                out.add((m, tile, split, stages))
    if kind == "kda" and batch == 8:  # stage counts below the maximum too
        out.update(KDA_CANDIDATES)
    return sorted(out)


CASES = [(k, b, t) for k in sorted(SHAPES) for b in (1, 2, 4, 8, 16, 32, 64) for t in candidates(k, b)]


@needs_gpu
@pytest.mark.parametrize("kind,batch,tac", CASES, ids=[f"{k}-b{b}-{'.'.join(map(str, t))}" for k, b, t in CASES])
def test_projection_tactic(kind, batch, tac):
    H.load_plugin()
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.gemm.projection import PreparedProjection

    n, k = SHAPES[kind], 7168
    g = H.generator(2000 + batch)
    weight = H.randn((n, k), 0.02, g)
    x = H.randn((batch, k), 1.0, g)
    proj = PreparedProjection(weight, {batch: tac}).prepare()
    out = torch.empty((batch, n), dtype=torch.bfloat16, device="cuda")
    H.poison(out)
    proj.run(x, out)
    torch.cuda.synchronize()
    ref = (x.float() @ weight.float().T).to(torch.bfloat16)
    item = f"projsweep.{kind}.b{batch}.{'.'.join(map(str, tac))}"
    H.check(item, out, ref, 1e-3, 2**-7)
    if os.environ.get("K31_TIME") == "1":
        copies = H.copies_for(n * k * 2)
        projs = [proj] + [PreparedProjection(H.randn((n, k), 0.02, g), {batch: tac}).prepare() for _ in range(copies - 1)]
        outs = [torch.empty((batch, n), dtype=torch.bfloat16, device="cuda") for _ in projs]
        t = H.cold_time_us([lambda p=p, o=o: p.run(x, o) for p, o in zip(projs, outs)])
        fl = H.floor_us(n * k * 2 + batch * (k + n) * 2)
        ctas = -(-n // tac[0]) * tac[2]
        H.record({"kind": "time", "item": item, "tactic": tac, "ctas": ctas, "current": tac == current_tactic(kind, batch), **t, **fl})
