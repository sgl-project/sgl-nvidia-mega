# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""QF-cross index query (K31MK_DSA_QFX; kimi_k31/dsa/index_qfx.cuh) on 33-128 rows vs QF-wide (index_qfw.cuh, already gated vs an
FP64 reference and in-boot KL at 16-32 rows), an independent FP64 reference and stock's functions. Packed rows as under QF:
[q_a | kv_a | gate | index-K (128) | head gate (32)]. Helpers (case, FP64 reference, stock chain) are test_k31_index_qfw's.

Gates (stated before any run; assertion order in test_qfx_bitwise later changed
so the bitwise asserts run before the FP64 code line):
- test_qfx_bitwise (rows 33 / 40 / 64 / 72 / 100 / 128 x seeds 1 / 2): codes and head weights BITWISE equal to QF-wide run on the
  same rows in chunks of 9-32 rows (QF-wide's rows are independent of each other and of its instance: test_k31_index_qfw rows),
  AND the code line vs the independent FP64 reference (share of differing codes <= 1.25 x stock's share + 0.001; stock = the
  served rmsnorm -> torch.mm -> act_quant), AND head weights vs stock's bf16_head_gate_q_scale >= 98 % bitwise with every ratio
  exactly 1, 2 or 1/2. No NaN / Inf.
- test_qfx_index_k (rows 40 / 100 / 128; nonzero-mean index-K, random FP32 LayerNorm affine, one negative slot, poisoned pool): the
  whole page pool BITWISE equal to QF-wide run on the same chunks into a copy of the pool; at least one row stored.
- test_qfx_instances: rows 72 at its own instance (9 tiles) vs forced 16 tiles, rows 33 (5 tiles) vs forced 8: codes, weights and
  pool BITWISE equal.
- test_qfx_soak (rows 72 / 128): 200 launches on one input, every launch BITWISE equal to the first; the first passes the code
  line. Under variant qfx_jitter (poisoned slots + random delays before the DSMEM stores) the whole module must still PASS.
- test_qfx_trigger (rows 72 / 128; added with the trig parameter, before its data): codes, weights and pool BITWISE equal
  across trig 0 / 1 / 2 (dependent-launch placement has no memory effect).
Mutations (each must FAIL the named test): qfx_order (own partial added first) -> bitwise; qfx_slice (rank 7's sum-of-squares partial
dropped) -> bitwise / FP64; qfx_store64 (no index-K store for rows >= 64) -> index_k; qfx_owner (pass-1 tiles to the wrong owner) ->
bitwise; qfx_race (jitter + slots read before the exchange's wait) -> soak / bitwise; qfx_gate (next head's gate column) -> bitwise;
qfx_own (own partial dropped) -> bitwise.
K31_TEST_HEADS (default 12): MLA heads per rank (TP8 12, TP16 6); sets the index-K / gate column offsets.

LOW-ROW EXTENSION (K31MK_DSA_QFX_LOW: the same kernel at 9-32 rows, kT 2-4, in place of QF-wide; written
before any run of this file):
- test_qfx_bitwise also at rows 9 / 16 / 17 / 24 / 25 / 32 (one QF-wide launch covers each): same asserts as above.
- test_qfx_index_k also at rows 9 / 16 / 24 / 32; test_qfx_instances also rows 16 at kT 2 vs forced 4, rows 25 at kT 4 vs forced 8;
  test_qfx_soak also at rows 16 / 32; test_qfx_trigger also at rows 16 / 32.
- Every mutant above except qfx_store64 (rows >= 64 only) must FAIL at least one rows <= 32 item; qfx_race must fail the module;
  qfx_jitter (variant) must PASS the whole module.
- (added after an earlier run FAILED on 'qfx_order not caught at rows <= 32': random inputs
  almost never let an FP32 reorder of 8 slice partials reach an FP8 code) test_qfx_order_low (rows 16 / 24 / 32): wq_b slices 0-3 equal and
  q_a slice 0 = +2^18 v, slice t = -2^18 v for the rows of n8 tile t >= 1 (owner rank t), so slice 0 and the owner's slice cancel exactly and
  the summation order decides the low bits: codes and head weights BITWISE equal to QF-wide (which sums slices 0..7 in order).
"""

import pytest
import torch

import _harness as H
import _hook_variants as V
import test_k31_index_qfw as QW

_CUH = "kimi_k31/dsa/index_qfx.cuh"
MUTATIONS = {
    "qfx_order": V.edits(_CUH, 1),
    "qfx_slice": V.edits(_CUH, 2),
    "qfx_store64": V.edits(_CUH, 3),
    "qfx_owner": V.edits(_CUH, 4),
    "qfx_race": V.edits(_CUH, 5),
    "qfx_jitter": V.edits(_CUH, 6),  # variant: must PASS
    "qfx_gate": V.edits(_CUH, 7),
    "qfx_own": V.edits(_CUH, 8),
}
K_COL, G_COL, PITCH, PAGES = QW.K_COL, QW.G_COL, QW.PITCH, 8
needs_gpu = QW.needs_gpu


def chunks(rows):
    """QF-wide launches covering [0, rows): balanced chunks of 9-32 rows."""
    n = -(-rows // 32)
    base, extra = divmod(rows, n)
    out, lo = [], 0
    for i in range(n):
        hi = lo + base + (1 if i < extra else 0)
        out.append((lo, hi))
        lo = hi
    assert all(9 <= hi - lo <= 32 for lo, hi in out) and lo == rows
    return out


def run_qfx(w, projected, slots, cache, norm, kt=0):
    from sgl_nvidia_mega.kimi_k31.dsa.index_qfx import FusedIndexQueryCross

    fused = FusedIndexQueryCross(w, projected.shape[0])
    H.poison(fused.query, fused.gates)
    q8, gates = fused.run_qf(projected, K_COL, G_COL, slots, cache, norm, kt)
    torch.cuda.synchronize()
    return q8.view(torch.uint8).reshape(-1, 32, 128).clone(), gates.reshape(-1, 32).clone()


def run_qfw_chunks(w, projected, slots, cache, norm):
    codes, weights = [], []
    for lo, hi in chunks(projected.shape[0]):
        c, g = QW.run_qfw(w, projected[lo:hi].contiguous(), slots[lo:hi].contiguous(), cache, norm)
        codes.append(c)
        weights.append(g)
    return torch.cat(codes), torch.cat(weights)


def code_line(tag, codes, scodes, ref):
    rc = QW.canon_zero(ref)
    mism = float((QW.canon_zero(codes) != rc).float().mean())
    smism = float((QW.canon_zero(scodes) != rc).float().mean())
    nan = int(((codes & 0x7F) == 0x7F).sum())
    row = {"kind": "qfx_fp64", "item": tag, "mism": mism, "stock_mism": smism, "line": 1.25 * smism + 0.001,
           "vs_stock": float((QW.canon_zero(codes) != QW.canon_zero(scodes)).float().mean()), "nan": nan}
    H.record(row)
    print("QFX_FP64", row, flush=True)
    assert nan == 0, f"{tag}: NaN codes {row}"
    assert mism <= row["line"], f"{tag}: codes further from FP64 than stock's: {row}"
    return row


@needs_gpu
@pytest.mark.parametrize("rows", (9, 16, 17, 24, 25, 32, 33, 40, 64, 72, 100, 128), ids=lambda r: f"b{r}")
@pytest.mark.parametrize("seed", (1, 2))
def test_qfx_bitwise(rows, seed):
    H.load_plugin(MUTATIONS)
    w, projected, scodes, sgates, _ = QW.case(rows, seed)
    slots = torch.full((rows,), -1, dtype=torch.int64, device="cuda")
    dummy = torch.zeros((1, 64 * 132), dtype=torch.uint8, device="cuda")
    norm = QW.key_norm(3)
    codes, weights = run_qfx(w, projected, slots, dummy, norm)
    rcodes, rweights = run_qfw_chunks(w, projected, slots, dummy, norm)
    tag = f"b{rows}.s{seed}"
    row = {"kind": "qfx_bitwise", "item": tag, "codes_ne": int((codes != rcodes).sum()),
           "w_ne": int((weights != rweights).sum()), "chunks": chunks(rows)}
    H.record(row)
    print("QFX_BITWISE", row, flush=True)
    # bitwise first (FILL_qfw-wide_qfx1: with the FP64 line first, mutants qfx_owner / qfx_own failed on 'further from FP64' and never
    # reached the bitwise asserts); both checks still gate.
    assert row["codes_ne"] == 0, f"{tag}: codes not bitwise QF-wide: {row}"
    assert row["w_ne"] == 0, f"{tag}: weights not bitwise QF-wide: {row}"
    code_line(tag, codes, scodes, QW.fp64_codes(projected, w))
    assert int(dummy.sum()) == 0, f"{tag}: a negative slot was stored"
    assert bool(torch.isfinite(weights).all()), f"{tag}: non-finite head weights"
    ratio = weights.double() / sgates.double()
    pow2 = (ratio == 1.0) | (ratio == 2.0) | (ratio == 0.5)
    wrow = {"kind": "qfx_w", "item": tag, "w_eq": float((weights == sgates).float().mean()), "w_not_pow2": int((~pow2).sum())}
    H.record(wrow)
    print("QFX_W", wrow, flush=True)
    assert wrow["w_not_pow2"] == 0 and wrow["w_eq"] >= 0.98, f"{tag}: head weights off stock's: {wrow}"


def _pool_case(rows, seed):
    w, projected, _, _, g = QW.case(rows, seed)
    slots = torch.randperm(PAGES * 64, generator=g, device="cuda")[:rows].to(torch.int64).contiguous()
    slots[rows // 2] = -1
    norm = QW.key_norm(17 + rows)
    pool = torch.full((PAGES, 64 * 132), 0xA5, dtype=torch.uint8, device="cuda")
    return w, projected, slots, norm, pool


@needs_gpu
@pytest.mark.parametrize("rows", (9, 16, 24, 32, 40, 100, 128), ids=lambda r: f"b{r}")
def test_qfx_index_k(rows):
    H.load_plugin(MUTATIONS)
    w, projected, slots, norm, pool = _pool_case(rows, 3)
    got, ref = pool.clone(), pool.clone()
    run_qfx(w, projected, slots, got, norm)
    run_qfw_chunks(w, projected, slots, ref, norm)
    row = {"kind": "qfx_k", "item": f"b{rows}", "pool_ne": int((got != ref).sum()), "pool_written": int((got != pool).sum()),
           "ref_written": int((ref != pool).sum())}
    H.record(row)
    print("QFX_K", row, flush=True)
    assert row["ref_written"] > 0 and row["pool_written"] > 0, f"no index-K row stored: {row}"
    assert row["pool_ne"] == 0, f"index-K pool != QF-wide on row chunks: {row}"


@needs_gpu
def test_qfx_instances():
    H.load_plugin(MUTATIONS)
    bad = {}
    for rows, kt in ((72, 16), (33, 8), (16, 4), (25, 8)):
        w, projected, slots, norm, pool = _pool_case(rows, 5)
        a_pool, b_pool = pool.clone(), pool.clone()
        a = run_qfx(w, projected, slots, a_pool, norm)
        b = run_qfx(w, projected, slots, b_pool, norm, kt)
        bad[f"b{rows}_kt{kt}"] = [int((a[0] != b[0]).sum()), int((a[1] != b[1]).sum()), int((a_pool != b_pool).sum())]
    row = {"kind": "qfx_inst", "item": "s5", "codes_w_pool_ne": bad}
    H.record(row)
    print("QFX_INST", row, flush=True)
    assert all(v == [0, 0, 0] for v in bad.values()), f"instances differ: {row}"


@needs_gpu
@pytest.mark.parametrize("rows", (16, 32, 72, 128), ids=lambda r: f"b{r}")
def test_qfx_soak(rows):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.dsa.index_qfx import FusedIndexQueryCross

    w, projected, scodes, _, _ = QW.case(rows, 5)
    slots = torch.full((rows,), -1, dtype=torch.int64, device="cuda")
    dummy = torch.zeros((1, 64 * 132), dtype=torch.uint8, device="cuda")
    norm = QW.key_norm(5)
    fused = FusedIndexQueryCross(w, rows)
    outs = []
    for _ in range(200):
        H.poison(fused.query, fused.gates)
        q8, gates = fused.run_qf(projected, K_COL, G_COL, slots, dummy, norm)
        outs.append((q8.view(torch.uint8).clone(), gates.clone()))
    torch.cuda.synchronize()
    first = outs[0]
    ne = sum(int(not (torch.equal(o[0], first[0]) and torch.equal(o[1], first[1]))) for o in outs[1:])
    row = {"kind": "qfx_soak", "item": f"b{rows}", "launches": len(outs), "launches_ne_first": ne}
    H.record(row)
    print("QFX_SOAK", row, flush=True)
    code_line(f"soak.b{rows}", first[0].reshape(rows, 32, 128), scodes, QW.fp64_codes(projected, w))
    assert ne == 0, f"launches differ from the first: {row}"


@needs_gpu
@pytest.mark.parametrize("rows", (16, 32, 72, 128), ids=lambda r: f"b{r}")
def test_qfx_trigger(rows):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.dsa.index_qfx import FusedIndexQueryCross

    w, projected, slots, norm, pool = _pool_case(rows, 9)
    outs = []
    for trig in (0, 1, 2):
        fused = FusedIndexQueryCross(w, rows)
        fused.trig = trig
        H.poison(fused.query, fused.gates)
        cache = pool.clone()
        q8, gates = fused.run_qf(projected, K_COL, G_COL, slots, cache, norm)
        torch.cuda.synchronize()
        outs.append((q8.view(torch.uint8).clone(), gates.clone(), cache))
    bad = {f"trig{i}": [int((outs[i][j] != outs[0][j]).sum()) for j in range(3)] for i in (1, 2)}
    row = {"kind": "qfx_trig", "item": f"b{rows}", "codes_w_pool_ne_vs_trig0": bad, "k_written": int((outs[0][2] != pool).sum())}
    H.record(row)
    print("QFX_TRIG", row, flush=True)
    assert row["k_written"] > 0, f"no index-K row stored: {row}"
    assert all(v == [0, 0, 0] for v in bad.values()), f"outputs differ across trigger placements: {row}"


@needs_gpu
@pytest.mark.parametrize("rows", (16, 24, 32), ids=lambda r: f"b{r}")
def test_qfx_order_low(rows):
    """Order-sensitive input (header, QFXLOW_G2): exact cancellation between slice 0 and the owner rank's slice."""
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.dsa.indexer import IndexWeights

    g = H.generator(900 + rows)
    rn = lambda *s: torch.randn(s, generator=g, device="cuda", dtype=torch.float32)  # noqa: E731
    bf = torch.bfloat16
    w0 = rn(4096, 192) / 1536 ** 0.5
    n0 = 1 + 0.1 * rn(192)
    w = IndexWeights(w0.repeat(1, 8).to(bf).contiguous(), (rn(32, 7168) / 7168 ** 0.5).to(bf), n0.repeat(8).to(bf).contiguous(), 1e-5)
    projected = torch.randn((rows, PITCH), generator=g, device="cuda").to(bf)
    big = 2.0 ** 18
    for b in range(8, rows):
        t = b // 8
        v = (big * rn(192)).to(bf)
        projected[b, 0:192] = v
        projected[b, 192 * t : 192 * t + 192] = -v
    slots = torch.full((rows,), -1, dtype=torch.int64, device="cuda")
    dummy = torch.zeros((1, 64 * 132), dtype=torch.uint8, device="cuda")
    norm = QW.key_norm(19)
    codes, weights = run_qfx(w, projected, slots, dummy, norm)
    rcodes, rweights = run_qfw_chunks(w, projected, slots, dummy, norm)
    nz = int((QW.canon_zero(rcodes[8:]) != 0).sum())
    row = {"kind": "qfx_order_low", "item": f"b{rows}", "codes_ne": int((codes != rcodes).sum()), "w_ne": int((weights != rweights).sum()),
           "nonzero_codes_tiles_ge1": nz}
    H.record(row)
    print("QFX_ORDER_LOW", row, flush=True)
    assert nz > 0, f"degenerate input (all codes zero): {row}"
    assert row["codes_ne"] == 0 and row["w_ne"] == 0, f"not bitwise QF-wide on the order-sensitive input: {row}"
