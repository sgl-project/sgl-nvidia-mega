# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Packed split-K BF16 input projection (``gemm.projection.PreparedProjection``) at the
K3.1 TP8 per-rank shapes and the per-batch tactics ``providers.build_resources`` uses.

Shapes: MLA [3776 = q_a+kv_a 2112 | g_proj 1536 | index wk 128, 7168], KDA [6288, 7168],
rows 1-8/16/32/64. Reference: FP32 ``x @ W.T`` in PyTorch (independent of the kernel),
rounded once to BF16 like the kernel's single cast.

Tolerance (stated before any run): |out - ref| <= 1e-3 + 2^-7 |ref| on every cell. The
kernel accumulates each K slice in FP32 and reduces the slices in FP32 before one BF16
rounding, so it differs from the FP32 reference by at most one BF16 rounding (2^-9
relative) plus FP32 reordering (~1e-6 relative); 2^-7 leaves 4x margin. A dropped K tile
or split-K peer moves cells by O(|ref| / sqrt(tiles)) >> 2^-7.

Mutations (each must make this test fail): ``drop_peer`` drops the last split-K peer's
partial in the owner's reduction; ``acc_reset`` resets the accumulator at K tile 1, which
discards K tile 0.
"""

import os

import pytest
import torch

import _harness as H

MUTATIONS = {
    # The served epilogue's reduction (epilog_warp_act repeats the loop: the anchor runs to epilog_warp's next comment,
    # since an anchor found twice aborts the mutated import before any test runs, which run_suite counts as CAUGHT).
    "drop_peer": (
        "kimi_k31/k3/cutedsl/gemm/splitk.py",
        "for peer in cutlass.range_constexpr(self.split_k - 1):\n"
        "                    for i in cutlass.range_constexpr(vpt):\n"
        "                        rAcc[i] = rAcc[i] + mailbox[peer * vpp + epi_tid * vpt + i]\n\n"
        "        if split_rank == OWNER_RANK:\n"
        "            rD.store(rAcc.load().to(c_dtype))\n"
        "            # Preserve TMEM coordinates",
        "for peer in cutlass.range_constexpr(self.split_k - 2):\n"
        "                    for i in cutlass.range_constexpr(vpt):\n"
        "                        rAcc[i] = rAcc[i] + mailbox[peer * vpp + epi_tid * vpt + i]\n\n"
        "        if split_rank == OWNER_RANK:\n"
        "            rD.store(rAcc.load().to(c_dtype))\n"
        "            # Preserve TMEM coordinates",
    ),
    "acc_reset": (
        "kimi_k31/k3/cutedsl/gemm/splitk.py",
        "accumulate = k_tile != 0 if k_block == 0 else True",
        "accumulate = k_tile != 1 if k_block == 0 else True",
    ),
    # K31MK_PROJ_TACTICS: an override that ignores its rows key, and a split that does
    # not divide the K tiles accepted (SplitKGemm would then drop the remainder tiles).
    "ovr_rows": (
        "kimi_k31/k3/cutedsl/gemm/projection.py",
        "table[kind, int(rows)] = tactic",
        "table[kind, 8] = tactic",
    ),
    "split_unchecked": (
        "kimi_k31/k3/cutedsl/gemm/projection.py",
        "or k_tiles % split or",
        "or",
    ),
    # K-42: an override whose splits hold no more K tiles than stages accepted.
    "k42_unchecked": (
        "kimi_k31/k3/cutedsl/gemm/projection.py",
        "if len(tactic) == 4 and 56 // max(tactic[2], 1) <= tactic[3]:",
        "if False:",
    ),
    # L3: widths 24 / 40 / 48 / 56 fall back to GB300's N-tile-8 default instead of the next width's tactic.
    "l3_roundup": (
        "kimi_k31/k3/cutedsl/gemm/projection.py",
        "if 8 < batch < max(TACTIC_WIDTHS) and batch not in TACTIC_WIDTHS:",
        "if False:",
    ),
    #  (wide tier): SplitKGemm's C is [weight rows, tokens] (M = weight rows, N = tokens), so the token axis is
    # grid y. proj_floor_tokens: grid y covers only whole N (token) tiles, so the partial last token tile of a ragged 72-127-row
    # call never launches. Must fail test_projection_wide at every ragged row and pass at 128 (whole 64-token tiles).
    # (wm0proj_r1 ran a grid-x form, proj_floor_rows, by mistake: it floors the weight-row axis, caught at KDA's 3216 rows only.)
    "proj_floor_tokens": (
        "kimi_k31/k3/cutedsl/gemm/splitk.py",
        "                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,\n",
        "                c.layout.shape[1] // self.cta_n * self.split_k,\n",
    ),
}

# providers.build_resources tactics: (M tile, N tile, split K, stages) per batch.
MLA_TACTICS = {1: (64, 8, 1, 12), 16: (64, 16, 2, 10), 32: (64, 32, 2, 8), 64: (64, 64, 2, 5)}
KDA_TACTICS = {16: (128, 16, 2, 6), 32: (128, 32, 2, 5), 64: (64, 64, 1, 6)}
BATCHES = (1, 2, 3, 4, 5, 8, 16, 32, 64)
# Packed rows per rank; K31_TEST_HEADS=6 is TP16 (MLA pitch(6) = 3008;
# KDA 4 x 768 + 128 + 6 = 3206, padded to 3216).
SHAPES = {12: {"mla": 3776, "kda": 6288}, 6: {"mla": 3008, "kda": 3216}}[int(os.environ.get("K31_TEST_HEADS", "12"))]
# Candidates for the KDA projection at the c=1 verify rows (Breq 1 x Q8 = 8): the shipped
# 128.8.2.6 runs ceil(3216 / 128) x 2 = 52 CTAs at TP16; M tile 64 at split 2 runs 102.
KDA_CANDIDATES = ((64, 8, 2, 8), (64, 8, 2, 11), (64, 8, 2, 12), (128, 8, 4, 6), (64, 8, 4, 8))


def tactic(kind, batch):
    """The tactic providers.build_resources uses on this device (SM107 overrides included)."""
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.gemm.projection import projection_tactic

    return projection_tactic(kind, batch, torch.device("cuda", torch.cuda.current_device()))


def gb300_tactic(kind, batch):
    if kind == "mla":
        return MLA_TACTICS.get(batch, (64, 8, 4, 8))
    return KDA_TACTICS.get(batch, (128, 8, 2, 6))


needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)


@needs_gpu
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("kind", sorted(SHAPES))
def test_projection(kind, batch):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.gemm.projection import PreparedProjection

    n, k = SHAPES[kind], 7168
    g = H.generator(1000 + batch + (7 if kind == "kda" else 0))
    weight = H.randn((n, k), 0.02, g)
    x = H.randn((64, k), 1.0, g)[:batch]
    proj = PreparedProjection(weight, {batch: tactic(kind, batch)}).prepare()
    out = torch.empty((batch, n), dtype=torch.bfloat16, device="cuda")
    H.poison(out)
    proj.run(x, out)
    torch.cuda.synchronize()
    ref = (x.float() @ weight.float().T).to(torch.bfloat16)
    res = H.function_resources(proj.kernels[batch])
    H.record({"kind": "resources", "item": f"projection.{kind}.b{batch}", "tactic": tactic(kind, batch), **res, **H.device_facts()})
    H.check(f"projection.{kind}.b{batch}", out, ref, 1e-3, 2**-7)

    # Replay: a second call on new activations must not depend on the first.
    x2 = H.randn((batch, k), 1.0, g)
    H.poison(out)
    proj.run(x2, out)
    torch.cuda.synchronize()
    H.check(f"projection.{kind}.b{batch}.second", out, (x2.float() @ weight.float().T).to(torch.bfloat16), 1e-3, 2**-7)

    if os.environ.get("K31_TIME") == "1" and not os.environ.get("K31_MUTATION"):
        wbytes = n * k * 2
        copies = H.copies_for(wbytes)
        projs = [proj] + [
            PreparedProjection(H.randn((n, k), 0.02, g), {batch: tactic(kind, batch)}).prepare()
            for _ in range(copies - 1)
        ]
        outs = [torch.empty((batch, n), dtype=torch.bfloat16, device="cuda") for _ in projs]
        calls = [lambda p=p, o=o: p.run(x, o) for p, o in zip(projs, outs)]
        t = H.cold_time_us(calls)
        fl = H.floor_us(wbytes + batch * (k + n) * 2)
        H.record({"kind": "time", "item": f"projection.{kind}.b{batch}", **t, **fl, "tactic": tactic(kind, batch)})
        print("TIME", kind, batch, t, fl, flush=True)


# L3: the K3.1 verify widths of Breq 3 / 5 / 6 / 7 (Q8) run their own graphs. Each must take the tactic its
# batch ran with when padded to the Breq 4 / 8 graph (32 / 64 rows). Expected tactics are the source tables' 32- / 64-row
# entries, copied here: SM107 mla 32 is not in SM107_TACTICS (GB300's (64, 32, 2, 8)), kda 32 (64, 32, 2, 9), both 64
# (64, 64, 2, 6); GB300 = MLA_TACTICS / KDA_TACTICS above. Then the same output gate as test_projection.
L3_PADDED_TO = {24: 32, 40: 64, 48: 64, 56: 64}
L3_SM107 = {("mla", 32): (64, 32, 2, 8), ("kda", 32): (64, 32, 2, 9), ("mla", 64): (64, 64, 2, 6), ("kda", 64): (64, 64, 2, 6)}


@needs_gpu
@pytest.mark.parametrize("batch", sorted(L3_PADDED_TO))
@pytest.mark.parametrize("kind", sorted(SHAPES))
def test_projection_l3(kind, batch, monkeypatch):
    monkeypatch.delenv("K31MK_PROJ_TACTICS", raising=False)
    padded = L3_PADDED_TO[batch]
    sm107 = torch.cuda.get_device_capability() == (10, 7) and os.environ.get("K31MK_PROJ_SM107_RETUNE", "1") != "0"
    want = L3_SM107[kind, padded] if sm107 else gb300_tactic(kind, padded)
    got = tuple(tactic(kind, batch))
    assert got == want, f"{kind} {batch} rows: tactic {got}, want the {padded}-row tactic {want}"
    test_projection(kind, batch)


@needs_gpu
def test_tactic_override(monkeypatch):
    """K31MK_PROJ_TACTICS: each (kind, rows) entry replaces exactly that policy; every
    other policy keeps the device table. A split that does not divide K's 56 tiles is refused."""
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.gemm import projection as P

    dev = torch.device("cuda", torch.cuda.current_device())
    keys = [(k, b) for k in ("mla", "kda") for b in (1, 4, 8)]
    monkeypatch.delenv("K31MK_PROJ_TACTICS", raising=False)
    shipped = {key: P.projection_tactic(*key, dev) for key in keys}
    monkeypatch.setenv("K31MK_PROJ_TACTICS", "kda:4=64.8.2.12,mla:1=128.8.4.6")
    want = {**shipped, ("kda", 4): (64, 8, 2, 12), ("mla", 1): (128, 8, 4, 6)}
    assert {key: P.projection_tactic(*key, dev) for key in keys} == want
    weight = torch.zeros((256, 7168), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError):
        P.PreparedProjection(weight, {8: (64, 8, 3, 8)})
    # K-42: an override with no more K tiles per split (56 // 8 = 7) than stages (7) is refused.
    monkeypatch.setenv("K31MK_PROJ_TACTICS", "kda:4=64.8.8.7")
    with pytest.raises(ValueError):
        P.projection_tactic("kda", 4, dev)


@needs_gpu
@pytest.mark.parametrize("tac", KDA_CANDIDATES, ids=lambda t: ".".join(map(str, t)))
def test_kda_candidate(tac, monkeypatch):
    """A KDA projection candidate tactic at 8 rows (the c=1 verify) against the FP32
    reference (gate above), and against the shipped tactic's output: bitwise when the split matches
    (each split's K slice accumulates in the same order at either M tile), and NOT bitwise for a
    different split (control: the comparison detects a reordered FP32 sum)."""
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.gemm.projection import PreparedProjection

    n, k, batch = SHAPES["kda"], 7168, 8
    monkeypatch.delenv("K31MK_PROJ_TACTICS", raising=False)
    shipped = tactic("kda", batch)
    g = H.generator(4100 + n)
    weight = H.randn((n, k), 0.02, g)
    x = H.randn((batch, k), 1.0, g)
    ref = (x.float() @ weight.float().T).to(torch.bfloat16)
    outs = {}
    for t in (shipped, tac):
        out = torch.empty((batch, n), dtype=torch.bfloat16, device="cuda")
        H.poison(out)
        PreparedProjection(weight, {batch: t}).prepare().run(x, out)
        torch.cuda.synchronize()
        H.check(f"kda_candidate.{n}.{'.'.join(map(str, t))}", out, ref, 1e-3, 2**-7)
        outs[t] = out
    differ = int((outs[shipped].view(torch.int16) != outs[tac].view(torch.int16)).sum())
    H.record({"kind": "bitwise", "item": f"kda_candidate.{n}", "shipped": shipped, "tactic": tac, "differing_cells": differ})
    if tac[2] == shipped[2]:
        assert differ == 0, f"{tac} vs shipped {shipped}: {differ} differing cells"
    else:
        assert differ > 0, f"control: split {tac[2]} vs {shipped[2]} reproduced the shipped bits"


#  (wide decode tier): Breq 9-16 (Q8) run their own graphs at 72-128 rows, and the packed MLA / KDA
# projections take the 128-row tactic there (TACTIC_WIDTHS round-up), so the last M tile is partial at 72-120 rows. Same
# gate as test_projection (FP32 reference rounded once to BF16, |d| <= 1e-3 + 2^-7 |ref|, outputs poisoned first, a second
# call on new activations) at the served wide rows 72 / 96 / 120 / 128 plus off-grid 100 / 127. A row the kernel never
# writes keeps the poison and fails the gate. Mutant proj_floor_tokens must fail every ragged row.
WIDE_ROWS = (72, 96, 100, 120, 127, 128)


@needs_gpu
@pytest.mark.parametrize("batch", WIDE_ROWS)
@pytest.mark.parametrize("kind", sorted(SHAPES))
def test_projection_wide(kind, batch):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.gemm.projection import PreparedProjection

    n, k = SHAPES[kind], 7168
    g = H.generator(5000 + batch + (7 if kind == "kda" else 0))
    weight = H.randn((n, k), 0.02, g)
    tac = tactic(kind, batch)
    proj = PreparedProjection(weight, {batch: tac}).prepare()
    out = torch.empty((batch, n), dtype=torch.bfloat16, device="cuda")
    for call in ("first", "second"):
        x = H.randn((batch, k), 1.0, g)
        H.poison(out)
        proj.run(x, out)
        torch.cuda.synchronize()
        H.check(f"projection_wide.{kind}.b{batch}.{call}", out, (x.float() @ weight.float().T).to(torch.bfloat16), 1e-3, 2**-7)
    H.record({"kind": "resources", "item": f"projection_wide.{kind}.b{batch}", "tactic": tac})
