# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""SM107 TP8 projection entries (``K31MK_PROJ_SM107_TP8=1``) at 128 rows (Breq 16) and 256 rows (Breq 32), TP8 per-rank
shapes (KDA [6288, 7168], MLA [3776, 7168]): the TP8 tactic against the tactic it replaces, on the same inputs.

Reference: ``x.double() @ W.double().T`` (float64, independent of every kernel). Lines (stated before any run):
(1) every cell |out - ref| <= 1e-3 + 2^-7 |ref| (as test_k31_projection.py: one BF16 rounding of an FP32 sum);
(2) where the split is unchanged (KDA 128 rows: split 2 -> 2), the output is bitwise the replaced tactic's; otherwise
    (split 2 -> 1) the outcome envelope vs the replaced tactic: max |out - ref| at most its max plus 2^-7 max |ref|, mean at
    most 1.25 x its mean + 1e-6;
(3) routing on SM107 (retune on, no K31MK_PROJ_TACTICS / TMEM64, K31MK_SPLITK_OVERSIZED=1): knob unset -> the replaced
    tactic (128 rows: (64, 64, 2, 6); 256 rows: GB300's fallback, or (128, 64, 2, 4) with K31MK_PROJ_SM107_ENT=1); knob on
    with the TP8 width -> the TP8 tactic, with or without K31MK_PROJ_SM107_ENT; knob on with the TP16 width (KDA 3216, MLA
    3008) or no width -> exactly the knob-off tactic;
(4) a second call on new activations matches its own reference (no state carried between calls);
(5) knob on without K31MK_SPLITK_OVERSIZED=1 raises for the TP8 width (its tactics exceed the normal SMEM), not for TP16's.

Mutations (each must make a test fail): ``acc_reset`` discards K tile 0; ``k_slice`` gives every split rank K slice 0;
``tp8_width`` keys the KDA entries on the TP16 width; ``tp8_rows`` keys the KDA 128-row entry at 129 rows; ``tp8_default_on``
turns the entries on by default; ``tp8_precedence`` lets the shared tables win over the TP8 entries; ``tp8_ov_unchecked``
drops the oversized-mode check.
"""

import pytest
import torch

import _harness as H

MUTATIONS = {
    "acc_reset": (
        "kimi_k31/k3/cutedsl/gemm/splitk.py",
        "accumulate = k_tile != 0 if k_block == 0 else True",
        "accumulate = k_tile != 1 if k_block == 0 else True",
    ),
    "k_slice": (
        "kimi_k31/k3/cutedsl/gemm/splitk.py",
        "k_tile_start = split_rank * k_tile_count",
        "k_tile_start = 0 * k_tile_count",
    ),
    "tp8_width": (
        "kimi_k31/k3/cutedsl/gemm/projection.py",
        'SM107_TP8_WIDTHS = {"kda": 6288, "mla": 3776}',
        'SM107_TP8_WIDTHS = {"kda": 3216, "mla": 3776}',
    ),
    "tp8_rows": (
        "kimi_k31/k3/cutedsl/gemm/projection.py",
        '"kda": {128: (64, 128, 2, 6),',
        '"kda": {129: (64, 128, 2, 6),',
    ),
    "tp8_default_on": (
        "kimi_k31/k3/cutedsl/gemm/projection.py",
        'os.environ.get("K31MK_PROJ_SM107_TP8", "0") == "1"',
        'os.environ.get("K31MK_PROJ_SM107_TP8", "1") == "1"',
    ),
    "tp8_precedence": (
        "kimi_k31/k3/cutedsl/gemm/projection.py",
        "table = {**table, **SM107_TACTICS_TP8[kind]}",
        "table = {**SM107_TACTICS_TP8[kind], **table}",
    ),
    "tp8_ov_unchecked": (
        "kimi_k31/k3/cutedsl/gemm/projection.py",
        'if os.environ.get("K31MK_SPLITK_OVERSIZED", "0") != "1":\n                raise ValueError("K31MK_PROJ_SM107_TP8=1',
        'if False:\n                raise ValueError("K31MK_PROJ_SM107_TP8=1',
    ),
}

K = 7168
TP8 = {"kda": 6288, "mla": 3776}
TP16 = {"kda": 3216, "mla": 3008}
# (kind, rows): (TP8 tactic, replaced without K31MK_PROJ_SM107_ENT, replaced with it)
ENTRIES = {
    ("kda", 128): ((64, 128, 2, 6), (64, 64, 2, 6), (64, 64, 2, 6)),
    ("mla", 128): ((64, 64, 1, 10), (64, 64, 2, 6), (64, 64, 2, 6)),
    ("kda", 256): ((64, 128, 1, 6), (128, 8, 2, 6), (128, 64, 2, 4)),
    ("mla", 256): ((128, 64, 1, 6), (64, 8, 4, 8), (128, 64, 2, 4)),
}
KNOBS = ("K31MK_PROJ_TACTICS", "K31MK_PROJ_TMEM64", "K31MK_PROJ_SM107_RETUNE", "K31MK_PROJ_SM107_ENT", "K31MK_PROJ_SM107_TP8")

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


def _clear(monkeypatch):
    for knob in KNOBS:
        monkeypatch.delenv(knob, raising=False)
    monkeypatch.setenv("K31MK_SPLITK_OVERSIZED", "1")


@needs_rubin
@pytest.mark.parametrize("rows", [128, 256])
@pytest.mark.parametrize("kind", ["kda", "mla"])
def test_tp8_entry(kind, rows, monkeypatch):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.gemm.projection import PreparedProjection, projection_tactic

    new, old, old_ent = ENTRIES[kind, rows]
    dev = torch.device("cuda", torch.cuda.current_device())
    _clear(monkeypatch)
    route = {}
    for ent in ("0", "1"):
        monkeypatch.setenv("K31MK_PROJ_SM107_ENT", ent)
        monkeypatch.delenv("K31MK_PROJ_SM107_TP8", raising=False)
        off = tuple(projection_tactic(kind, rows, dev, width=TP8[kind]))
        monkeypatch.setenv("K31MK_PROJ_SM107_TP8", "1")
        route[ent] = (off, tuple(projection_tactic(kind, rows, dev, width=TP8[kind])),
                      tuple(projection_tactic(kind, rows, dev, width=TP16[kind])), tuple(projection_tactic(kind, rows, dev)))
        monkeypatch.delenv("K31MK_PROJ_SM107_TP8", raising=False)
        knob_off_tp16 = tuple(projection_tactic(kind, rows, dev, width=TP16[kind]))
        want = (old_ent if ent == "1" else old, new, knob_off_tp16, off)
        assert route[ent] == want, f"{kind} {rows} rows, ENT={ent}: (off, TP8, TP16, no width) = {route[ent]}, want {want}"
    monkeypatch.setenv("K31MK_PROJ_SM107_ENT", "1")
    n = TP8[kind]
    g = H.generator(2200 + rows + (n % 97))
    weight = H.randn((n, K), 0.02, g)
    x = H.randn((rows, K), 1.0, g)
    ref = x.double() @ weight.double().T
    pn = PreparedProjection(weight, {rows: new}).prepare()
    po = PreparedProjection(weight, {rows: old_ent}).prepare()
    out, base = _run(pn, x, rows, n), _run(po, x, rows, n)
    item = f"proj_tp8.{kind}.r{rows}"
    H.check(item, out, ref.float(), 1e-3, 2**-7)
    H.check(item + ".replaced", base, ref.float(), 1e-3, 2**-7)
    (nmax, nmean), (omax, omean) = _err(out, ref), _err(base, ref)
    changed = int((out.view(torch.int16) != base.view(torch.int16)).sum())
    H.record({"kind": "check", "item": item, "new": list(new), "old": list(old_ent), "new_max": nmax, "new_mean": nmean,
              "old_max": omax, "old_mean": omean, "changed_vs_old": changed, "cells": out.numel()})
    print(f"TP8_ERR {kind} r{rows} new {new} max {nmax:.3e} mean {nmean:.3e} | old {old_ent} max {omax:.3e} mean {omean:.3e} "
          f"| changed {changed}/{out.numel()}", flush=True)
    if new[2] == old_ent[2]:
        assert changed == 0, f"{item}: {changed} cells differ from the replaced tactic (same split: bitwise expected)"
    else:
        ulp = float(ref.abs().max()) * 2**-7
        assert nmax <= omax + ulp, (nmax, omax, ulp)
        assert nmean <= 1.25 * omean + 1e-6, (nmean, omean)
    x2 = H.randn((rows, K), 1.0, g)
    out2 = _run(pn, x2, rows, n)
    H.check(item + ".second", out2, (x2.double() @ weight.double().T).float(), 1e-3, 2**-7)


@needs_rubin
def test_tp8_needs_oversized(monkeypatch):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.gemm.projection import projection_tactic

    dev = torch.device("cuda", torch.cuda.current_device())
    _clear(monkeypatch)
    monkeypatch.delenv("K31MK_SPLITK_OVERSIZED", raising=False)
    monkeypatch.setenv("K31MK_PROJ_SM107_TP8", "1")
    for kind in ("kda", "mla"):
        with pytest.raises(ValueError):
            projection_tactic(kind, 128, dev, width=TP8[kind])
        assert tuple(projection_tactic(kind, 128, dev, width=TP16[kind])) == (64, 64, 2, 6)
