# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fused index query (kimi_k31/dsa/index_q.*, one PDL kernel) and the MK's stock-chain fallback (IndexProjection) vs
STOCK's index-q method: the functions stock K3.1 runs, with stock's arguments (stock_index_q below), plus the
independent torch reference R.index_q; timing against the MK's stock chain.

Stock: Indexer.forward_cuda quantizes the query with act_quant(query, self.block_size=128, self.scale_fmt) and
DeepseekV2AttentionMLA builds the Indexer with scale_fmt="ue8m0" (models/deepseek_v2.py), so the scale is
2^ceil(log2(max(amax, 1e-4) / 448)) per (token, head); bf16_head_gate_q_scale folds it into the head weights.

Gates (stated before any run):
- test_index_q (random K3.1-shaped tensors): FP8 codes within 1 code step in every cell and differing in <= 1% of
  cells (R.fp8_ulp_check); head weights within 2^-7 relative (one BF16 rounding of the gate). Fused and IndexProjection
  against stock's functions, fused against the reference. Only the GEMMs' FP32 summation orders differ between the
  arms, so a different scale method shows up as a factor in (1, 2] on the weights and as many-step code errors.
- test_index_q_scale_domain (exact-arithmetic tensors: the BF16 query and gate are exact in any summation order, which
  the test asserts first, so the comparison isolates the quantizer): codes and head weights BITWISE equal to stock's
  functions in every cell, for every BF16 amax in [2^-20, 2^44) (64 binades x 128 mantissas). This covers the 1e-4
  amax floor and amax = 1.75 * 2^k, where amax * (1/448) is exactly a power of two. The FP8 codes of -0 and +0 count
  as equal there (7 cells per head are exact GEMM zeros, whose sign is the GEMM's, not the quantizer's); the number
  of such cells is recorded.
Mutations (each must FAIL): iq_range (FP8 range 440), iq_softmax (128^-0.5 dropped), iq_round (log2 rounded to
nearest, not up), iq_floor (1e-4 amax floor dropped), iq_exact (an earlier release's exact amax/448 scale), mk_chain_exact
(IndexProjection's act_quant calls back to scale_fmt=None), iq_copy_heads12 (the query copy always runs the 12-head
template; fails test_index_q_query_copy at K31_TEST_HEADS=6).
K31_TEST_HEADS (default 12): the MLA heads per rank of the packed rows and the query copy (TP8 12, TP16 6).
"""

import math
import os

import pytest
import torch

import _harness as H
import _hook_variants as V
import _ref_mla as R

_CUH = "kimi_k31/dsa/index_q.cuh"
MUTATIONS = {
    "iq_range": V.edits(_CUH, 1),
    "iq_softmax": V.edits(_CUH, 2),
    "iq_round": V.edits(_CUH, 3),
    "iq_floor": V.edits(_CUH, 4),
    "iq_exact": V.edits(_CUH, 5),
    "iq_copy_heads12": ("kimi_k31/dsa/index_q.cuh",
                        "auto kernel = mq_heads == 6 ? index_q_kernel<kSplit, 6> : index_q_kernel<kSplit, 12>;",
                        "auto kernel = index_q_kernel<kSplit, 12>;"),
    "mk_chain_exact": ("kimi_k31/dsa/indexer.py", 'SCALE_FMT = "ue8m0"', "SCALE_FMT = None"),
}
STOCK_SCALE_FMT = "ue8m0"  # models/deepseek_v2.py: Indexer(..., scale_fmt="ue8m0", block_size=128)
HEADS = int(os.environ.get("K31_TEST_HEADS", "12"))  # MLA heads per rank: TP8 12, TP16 6
PITCH = 2112 + 128 * HEADS + 128  # packed [q_a | kv_a | gate | index-K]: TP8 3776, TP16 3008
TIMING = os.environ.get("K31_TIME") == "1" and not os.environ.get("K31_MUTATION")
needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)


def weights(seed):
    from sgl_nvidia_mega.kimi_k31.dsa.indexer import IndexWeights

    g = H.generator(seed)
    rn = lambda *s: torch.randn(s, generator=g, device="cuda", dtype=torch.float32)  # noqa: E731
    bf = torch.bfloat16
    return IndexWeights(
        (rn(4096, 1536) / math.sqrt(1536)).to(bf), (rn(32, 7168) / math.sqrt(7168)).to(bf),
        (1 + 0.1 * rn(1536)).to(bf), 1e-5)


def activations(rows, g, pitch=PITCH):
    projected = torch.randn((rows, pitch), generator=g, device="cuda").to(torch.bfloat16)
    hidden = torch.randn((rows, 7168), generator=g, device="cuda").to(torch.bfloat16)
    return projected, hidden


def stock_index_q(projected, hidden, w):
    """Stock's chain on these tensors: q_a RMSNorm, wq_b GEMM, act_quant(query, 128, "ue8m0") (dsa_indexer.py), BF16
    head gate GEMM and bf16_head_gate_q_scale(gate, q_scale, 128**-0.5). Returns (query, codes, scales, weights)."""
    from sgl_kernel import rmsnorm
    bf16_head_gate_q_scale = H.bf16_head_gate_q_scale  # the release host has no standalone head-gate function
    from sglang.kernels.ops.quantization.act_quant_triton import act_quant

    rows = projected.shape[0]
    query = torch.mm(rmsnorm(projected[:, :1536], w.query_norm, w.epsilon), w.query.T).view(rows, 32, 128)
    codes, scales = act_quant(query, 128, STOCK_SCALE_FMT)
    gates = bf16_head_gate_q_scale(torch.mm(hidden, w.gate.T), scales, 128**-0.5)
    return query, codes.view(torch.uint8), scales.view(rows, 32), gates.view(rows, 32)


@needs_gpu
@pytest.mark.parametrize("rows", (1, 2, 3, 5, 8), ids=lambda r: f"b{r}")
@pytest.mark.parametrize("seed", (1, 2))
def test_index_q(rows, seed):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.dsa.index_q import FusedIndexQuery
    from sgl_nvidia_mega.kimi_k31.dsa.indexer import IndexProjection

    w = weights(100 * seed + rows)
    g = H.generator(7 * seed + rows)
    projected, hidden = activations(rows, g)
    fused = FusedIndexQuery(w, rows)
    H.poison(fused.query, fused.gates)
    q8, gates = fused.run(hidden, projected)
    q8, gates = q8.view(torch.uint8).reshape(rows, 32, 128), gates.reshape(rows, 32)
    mk = IndexProjection(w, rows)
    mq8, mgates = mk.run(hidden, projected)
    mq8, mgates = mq8.view(torch.uint8).reshape(rows, 32, 128), mgates.reshape(rows, 32)
    _, scodes, _, sgates = stock_index_q(projected, hidden, w)
    torch.cuda.synchronize()
    codes_ref, w_ref = R.index_q(projected, hidden, w.query_norm, w.epsilon, w.query, w.gate)
    tag = f"b{rows}.s{seed}"
    for arm, codes, ws in (("fused", q8, gates), ("mk_chain", mq8, mgates)):
        H.record({"kind": "bitwise_vs_stock", "item": f"{arm}.{tag}",
                  "codes_exact": float((codes == scodes).float().mean()),
                  "weights_exact": float((ws == sgates).float().mean())})
    # Control first: does stock itself meet the reference gate on these inputs? (recorded, not asserted)
    errs = []
    for name, codes, ref in (("stock_vs_ref", scodes, codes_ref), ("iq_vs_stock", q8, scodes),
                             ("mk_vs_stock", mq8, scodes), ("iq_vs_ref", q8, codes_ref)):
        try:
            H.record({"kind": "check", **R.fp8_ulp_check(f"{name}.{tag}", codes, ref)})
        except AssertionError as e:
            H.record({"kind": "check_fail", "item": f"{name}.{tag}", "msg": str(e)})
            errs.append(str(e))
    H.check(f"iw_vs_stock.{tag}", gates, sgates, 0.0, 2.0**-7)
    H.check(f"mw_vs_stock.{tag}", mgates, sgates, 0.0, 2.0**-7)
    H.check(f"iw_vs_ref.{tag}", gates, w_ref, 0.0, 2.0**-7)
    assert not [e for e in errs if "stock_vs_ref" not in e], errs
    print("IQ", tag, float((q8 == scodes).float().mean()), float((gates == sgates).float().mean()), flush=True)


def canon_zero(codes):
    """FP8 E4M3 codes with -0 (0x80) mapped to +0."""
    return torch.where(codes == 0x80, torch.zeros_like(codes), codes)


def hadamard8(device):
    h = torch.ones((1, 1), device=device)
    for _ in range(3):
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


def domain_case(e0, part, g):
    """Exact-arithmetic inputs for 8 rows x 32 heads: (row r, head h) has amax |v[r, h]| with
    v = +-(1 + (8 part + r) / 128) * 2^(e0 + h). Row r's q_a normalizes to the +-1 row [H8[r], 1, ..., 1] (BF16
    rounding absorbs the RMSNorm epsilon), wq_b row h*128 + j (j < 8) is v[j, h] / 8 * H8[j] on columns 0-7, so
    q[r, h*128 + j] = v[r, h] if j == r else 0; rows h*128 + m (m >= 8) carry u[h, m] (|u| <= 2^(e0 + h)) on column 8,
    the same in every row. hidden[r] is one-hot at r and weights_proj[h, r] = gate[r, h]. Every product and partial sum
    is exact in FP32, so any summation order gives the same BF16 query and gate."""
    from sgl_nvidia_mega.kimi_k31.dsa.indexer import IndexWeights

    dev, bf = "cuda", torch.bfloat16
    rows, heads = 8, 32
    h8 = hadamard8(dev)
    exps = torch.arange(heads, device=dev, dtype=torch.float64) + e0
    mant = 1 + (8 * part + torch.arange(rows, device=dev, dtype=torch.float64)) / 128
    sign = 1 - 2 * ((torch.arange(rows, device=dev)[:, None] + torch.arange(heads, device=dev)[None, :]) % 2)
    v = (sign * mant[:, None] * torch.exp2(exps)[None, :]).to(bf)  # [rows, heads], exact in BF16
    wq_b = torch.zeros((heads, 128, 1536), dtype=torch.float64, device=dev)
    wq_b[:, :rows, :rows] = (v.double().T / 8)[:, :, None] * h8[None]  # [h, j, k] = v[j, h] / 8 * H8[j, k]
    u = (torch.rand((heads, 128 - rows), generator=g, device=dev, dtype=torch.float64) * 2 - 1) * torch.exp2(exps)[:, None]
    wq_b[:, rows:, rows] = u.to(bf).double()
    projected = torch.randn((rows, PITCH), generator=g, device=dev).to(bf)
    projected[:, :1536] = 1
    projected[:, :rows] = h8.to(bf)
    gate = torch.randn((rows, heads), generator=g, device=dev).to(bf)
    w_proj = torch.zeros((heads, 7168), dtype=bf, device=dev)
    w_proj[:, :rows] = gate.T
    hidden = torch.zeros((rows, 7168), dtype=bf, device=dev)
    hidden[:, :rows] = torch.eye(rows, device=dev, dtype=bf)
    w = IndexWeights(wq_b.view(4096, 1536).to(bf), w_proj, torch.ones(1536, dtype=bf, device=dev), 1e-6)
    q = torch.zeros((rows, heads, 128), dtype=bf, device=dev)
    q[:, :, rows:] = u.to(bf)[None]
    q[torch.arange(rows), :, torch.arange(rows)] = v
    return w, projected, hidden, q, gate


@needs_gpu
@pytest.mark.parametrize("half", (0, 1))
@pytest.mark.parametrize("e0", (-20, 12))
def test_index_q_scale_domain(e0, half):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.dsa.index_q import FusedIndexQuery
    from sgl_nvidia_mega.kimi_k31.dsa.indexer import IndexProjection

    g = H.generator(1000 + e0 + 50 * half)
    bad = {"fused_codes": 0, "fused_weights": 0, "mk_codes": 0, "mk_weights": 0, "ref_codes": 0, "ref_scales": 0,
           "zero_sign_only": 0}
    cells = heads = 0
    for part in range(8 * half, 8 * half + 8):
        w, projected, hidden, q, gate = domain_case(e0, part, g)
        squery, scodes, sscales, sgates = stock_index_q(projected, hidden, w)
        # Premise: the construction is exact, so stock's BF16 query and gate are the designed ones in every cell.
        assert torch.equal(squery, q), f"e0 {e0} part {part}: stock query != designed query (construction not exact)"
        assert torch.equal(torch.mm(hidden, w.gate.T), gate), f"e0 {e0} part {part}: gate construction not exact"
        fused = FusedIndexQuery(w, 8)
        H.poison(fused.query, fused.gates)
        q8, gates = fused.run(hidden, projected)
        mq8, mgates = IndexProjection(w, 8).run(hidden, projected)
        torch.cuda.synchronize()
        rcodes, rscales = R.index_q_quant(q)
        fcodes = q8.view(torch.uint8).reshape(8, 32, 128)
        mcodes = mq8.view(torch.uint8).reshape(8, 32, 128)
        for c in (fcodes, mcodes, rcodes):
            bad["zero_sign_only"] += int(((c != scodes) & (canon_zero(c) == canon_zero(scodes))).sum())
        scodes = canon_zero(scodes)
        bad["fused_codes"] += int((canon_zero(fcodes) != scodes).sum())
        bad["fused_weights"] += int((gates.reshape(8, 32) != sgates).sum())
        bad["mk_codes"] += int((canon_zero(mcodes) != scodes).sum())
        bad["mk_weights"] += int((mgates.reshape(8, 32) != sgates).sum())
        bad["ref_codes"] += int((canon_zero(rcodes) != scodes).sum())  # control: exact-math reference vs stock
        bad["ref_scales"] += int((rscales != sscales).sum())
        cells += scodes.numel()
        heads += sscales.numel()
    row = {"kind": "domain", "item": f"e0{e0}.half{half}", "binades": [e0, e0 + 31], "code_cells": cells,
           "scale_cells": heads, **bad}
    H.record(row)
    print("DOMAIN", row, flush=True)
    assert bad["fused_codes"] == 0 and bad["fused_weights"] == 0, f"fused index query != stock: {row}"
    assert bad["mk_codes"] == 0 and bad["mk_weights"] == 0, f"IndexProjection != stock: {row}"


@needs_gpu
@pytest.mark.parametrize("rows", (1, 8), ids=lambda r: f"b{r}")
def test_time_index_q(rows):
    if not TIMING:
        pytest.skip("timing runs only in the base run")
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.dsa.index_q import FusedIndexQuery
    from sgl_nvidia_mega.kimi_k31.dsa.indexer import IndexProjection

    g = H.generator(970 + rows)
    nbytes = (4096 * 1536 + 32 * 7168) * 2
    n = H.copies_for(nbytes)
    projected, hidden = activations(rows, g)
    ws = [weights(2000 + i) for i in range(n)]
    stock = [IndexProjection(w, rows) for w in ws]
    fused = [FusedIndexQuery(w, rows) for w in ws]
    out = {}
    for name, objs in (("stock_chain", stock), ("fused", fused)):
        t = H.cold_time_us([lambda o=o: o.run(hidden, projected) for o in objs])
        fl = H.floor_us(nbytes)
        H.record({"kind": "time", "item": f"index_q.{name}.b{rows}", **t, **fl})
        out[name] = t["us_median"]
        print("TIME", name, rows, t, fl, flush=True)
    H.record({"kind": "index_q_ab", "item": f"b{rows}", **out, "ratio": out["fused"] / out["stock_chain"]})


@needs_gpu
@pytest.mark.parametrize("rows", (1, 8), ids=lambda r: f"b{r}")
def test_index_q_query_copy(rows):
    """The folded H -> 16 head-padded MLA query copy (H = HEADS): heads 0..H-1 byte-identical to the source, pad heads untouched
    (zero), and the index outputs identical to the run without the copy."""
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.dsa.index_q import FusedIndexQuery

    w = weights(300 + rows)
    g = H.generator(31 + rows)
    projected, hidden = activations(rows, g)
    src = torch.randint(0, 256, (rows, HEADS, 576), generator=g, device="cuda", dtype=torch.int32).to(torch.uint8)
    dst = torch.zeros((rows, 1, 16, 576), dtype=torch.uint8, device="cuda")
    fused = FusedIndexQuery(w, rows)
    q8a, ga = (t.clone() for t in fused.run(hidden, projected))
    q8b, gb = fused.run(hidden, projected, src, dst)
    torch.cuda.synchronize()
    assert torch.equal(dst[:, 0, :HEADS], src), f"padded query heads 0-{HEADS - 1} differ from the MLA query"
    assert int(dst[:, 0, HEADS:].sum()) == 0, "pad heads written"
    assert torch.equal(q8a.view(torch.uint8), q8b.view(torch.uint8)) and torch.equal(ga, gb), "copy changed outputs"
