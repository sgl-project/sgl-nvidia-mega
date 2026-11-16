# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""SM107 projection table entries at 16 rows (Breq 2) and 256 rows (Breq 32), TP16 per-rank shapes (KDA [3216, 7168],
MLA [3008, 7168]): the served tactic (``projection_tactic``) against the tactic it replaced, on the same inputs.

Reference: ``x.double() @ W.double().T`` (float64, independent of every kernel). Lines (stated before any run):
(1) every cell |out - ref| <= 1e-3 + 2^-7 |ref| (as test_k31_projection.py: one BF16 rounding of an FP32 sum);
(2) where the split is unchanged (KDA: split 2 -> 2), the output is bitwise the replaced tactic's (same K slices, same
    MMA order per output element); otherwise (MLA: split 4 -> 2) the outcome envelope vs the replaced tactic: max
    |out - ref| at most its max plus one BF16 ulp of max |ref|, mean at most 1.25 x its mean + 1e-6;
(3) the served table on SM107 (retune on, no K31MK_PROJ_TACTICS / TMEM64) routes these rows to the replaced tactic by
    default and to the new tactic with ``K31MK_PROJ_SM107_ENT=1``;
(4) a second call on new activations matches its own reference (no state carried between calls).

Mutations (each must make a test fail): ``k_slice`` gives every split rank K slice 0; ``acc_reset`` discards K tile 0;
``entry_rows`` keys the KDA 16-row entry at 17 rows (16 rows then fall back to GB300's tactic); ``ent_default_on`` turns
the entries on by default.
"""

import pytest
import torch

import _harness as H

MUTATIONS = {
    "k_slice": (
        "kimi_k31/k3/cutedsl/gemm/splitk.py",
        "k_tile_start = split_rank * k_tile_count",
        "k_tile_start = 0 * k_tile_count",
    ),
    "acc_reset": (
        "kimi_k31/k3/cutedsl/gemm/splitk.py",
        "accumulate = k_tile != 0 if k_block == 0 else True",
        "accumulate = k_tile != 1 if k_block == 0 else True",
    ),
    "entry_rows": (
        "kimi_k31/k3/cutedsl/gemm/projection.py",
        '"kda": {16: (64, 16, 2, 11),',
        '"kda": {17: (64, 16, 2, 11),',
    ),
    "ent_default_on": (
        "kimi_k31/k3/cutedsl/gemm/projection.py",
        'os.environ.get("K31MK_PROJ_SM107_ENT", "0") == "1"',
        'os.environ.get("K31MK_PROJ_SM107_ENT", "1") == "1"',
    ),
}

SHAPES = {"kda": 3216, "mla": 3008}
K = 7168
# (kind, rows): (new served tactic, the tactic it replaced, same split)
ENTRIES = {
    ("kda", 16): ((64, 16, 2, 11), (128, 16, 2, 6), True),
    ("mla", 16): ((64, 16, 2, 11), (128, 16, 4, 5), False),
    ("kda", 256): ((128, 64, 2, 4), (128, 8, 2, 6), True),
    ("mla", 256): ((128, 64, 2, 4), (64, 8, 4, 8), False),
}

needs_rubin = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7), reason="SM107 table entries"
)


def _run(proj, x, rows, n):
    out = torch.empty((rows, n), dtype=torch.bfloat16, device="cuda")
    H.poison(out)
    proj.run(x, out)
    torch.cuda.synchronize()
    return out


def _err(out, ref):
    d = (out.double() - ref).abs()
    return float(d.max()), float(d.mean())


@needs_rubin
@pytest.mark.parametrize("rows", [16, 256])
@pytest.mark.parametrize("kind", ["kda", "mla"])
def test_sm107_entry(kind, rows, monkeypatch):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.gemm.projection import PreparedProjection, projection_tactic

    new, old, same_split = ENTRIES[kind, rows]
    dev = torch.device("cuda", torch.cuda.current_device())
    for knob in ("K31MK_PROJ_TACTICS", "K31MK_PROJ_TMEM64", "K31MK_PROJ_SM107_RETUNE", "K31MK_PROJ_SM107_ENT"):
        monkeypatch.delenv(knob, raising=False)
    served = tuple(projection_tactic(kind, rows, dev))
    assert served == old, f"{kind} {rows} rows, knob unset: served {served}, want the replaced {old}"
    monkeypatch.setenv("K31MK_PROJ_SM107_ENT", "1")
    served = tuple(projection_tactic(kind, rows, dev))
    assert served == new, f"{kind} {rows} rows, K31MK_PROJ_SM107_ENT=1: served {served}, want {new}"
    n = SHAPES[kind]
    g = H.generator(1900 + rows + (n % 97))
    weight = H.randn((n, K), 0.02, g)
    x = H.randn((rows, K), 1.0, g)
    ref = x.double() @ weight.double().T
    pn = PreparedProjection(weight, {rows: new}).prepare()
    po = PreparedProjection(weight, {rows: old}).prepare()
    out, base = _run(pn, x, rows, n), _run(po, x, rows, n)
    item = f"proj_entry.{kind}.r{rows}"
    H.check(item, out, ref.float(), 1e-3, 2**-7)
    H.check(item + ".replaced", base, ref.float(), 1e-3, 2**-7)
    (nmax, nmean), (omax, omean) = _err(out, ref), _err(base, ref)
    changed = int((out.view(torch.int16) != base.view(torch.int16)).sum())
    H.record({"kind": "check", "item": item, "new": list(new), "old": list(old), "new_max": nmax, "new_mean": nmean,
              "old_max": omax, "old_mean": omean, "changed_vs_old": changed, "cells": out.numel()})
    print(f"ENTRY_ERR {kind} r{rows} new {new} max {nmax:.3e} mean {nmean:.3e} | old {old} max {omax:.3e} mean {omean:.3e} "
          f"| changed {changed}/{out.numel()}", flush=True)
    if same_split:
        assert changed == 0, f"{item}: {changed} cells differ from the replaced tactic (same split: bitwise expected)"
    else:
        ulp = float(ref.abs().max()) * 2**-7
        assert nmax <= omax + ulp, (nmax, omax, ulp)
        assert nmean <= 1.25 * omean + 1e-6, (nmean, omean)
    x2 = H.randn((rows, K), 1.0, g)
    out2 = _run(pn, x2, rows, n)
    H.check(item + ".second", out2, (x2.double() @ weight.double().T).float(), 1e-3, 2**-7)
