# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""QFN index query (K31MK_DSA_QFN; kimi_k31/dsa/index_qfn.py through indexer.IndexProjection.project_query) on 2-128 rows vs an
FP64 reference and the stock wide chain (rmsnorm -> torch.mm -> indexer_q_quant_head_gate, IndexProjection with the knob off).

Gates (stated before any run):
- test_qfn_fp64 (rows 2 / 9 / 33 / 40 / 64 / 72 / 96 / 128 x seeds 1 / 2; random tensors): FP8 codes vs an independent FP64
  reference (xn = x rstd w_qa, q = xn @ wq_b^T, ue8m0 scale of the FP64 amax, e4m3 RN of the clamped quotient): share of differing
  codes <= 1.25 x stock's share vs the same reference + 0.001 (stock = IndexProjection with the knob off = the served chain; its
  share is the control, the QF-wide line). Head weights vs stock's: >= 98 % bitwise equal and every ratio exactly 1, 2 or 1/2 (the
  gate path is stock's; only the ue8m0 scale may move by one binade). No NaN / Inf. ENGAGED: the knob-on projection holds the fold
  and its codes differ from stock's in at least one cell (QFN rounds at other points; P(no difference) at >= 2 x 4096 cells ~ 0).
- test_qfn_off_is_stock (rows 40 / 128): with the knob off, IndexProjection.project_query is BITWISE the stock functions
  (rmsnorm -> torch.mm -> act_quant + bf16_head_gate_q_scale) - the OFF position is the served path.
- test_qfn_soak (rows 128): 100 launches of the knob-on projection on one input, every launch BITWISE equal to the first.
Mutations (each must FAIL the module): qfn_rstd (rstd := 1), qfn_nofold (w_qa not folded into wq_b), qfn_gate (the next head's gate
column), qfn_eps (epsilon x 1e4).
"""

import math
import os

import pytest
import torch

import _harness as H

_QFN = "kimi_k31/dsa/index_qfn.py"
MUTATIONS = {
    "qfn_rstd": (_QFN, "    rstd = tl.rsqrt(tl.sum(x * x, 0) / LORA + EPS)\n", "    rstd = 1.0\n"),
    "qfn_gate": (_QFN, "tl.load(G + row * GATE_STRIDE + heads)", "tl.load(G + row * GATE_STRIDE + (heads + 1) % HEADS)"),
    "qfn_eps": (_QFN, "            float(eps),\n            1536,\n", "            float(eps) * 1.0e4,\n            1536,\n"),
    "qfn_nofold": ("kimi_k31/dsa/index_qfw.py", "    folded = (query.float() * norm.float()[None, :])", "    folded = (query.float() * 1.0)"),
}
PITCH = 3040  # served TP16 packed-projection pitch under K31MK_DSA_QF=1; q_a is columns [0, 1536)
needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)


def index_weights(seed):
    from sgl_nvidia_mega.kimi_k31.dsa.indexer import IndexWeights

    g = H.generator(seed)
    rn = lambda *s: torch.randn(s, generator=g, device="cuda", dtype=torch.float32)  # noqa: E731
    bf = torch.bfloat16
    return IndexWeights(
        (rn(4096, 1536) / math.sqrt(1536)).to(bf), (rn(32, 7168) / math.sqrt(7168)).to(bf),
        (1 + 0.1 * rn(1536)).to(bf), 1e-5)


def case(rows, seed):
    w = index_weights(900 + 10 * seed + rows)
    g = H.generator(71 * seed + rows)
    # q_a with a per-row scale spread (x 0.25 .. 4) so rstd matters row by row
    projected = (torch.randn((rows, PITCH), generator=g, device="cuda")
                 * torch.exp2(4 * torch.rand((rows, 1), generator=g, device="cuda") - 2)).to(torch.bfloat16)
    hidden = torch.randn((rows, 7168), generator=g, device="cuda").to(torch.bfloat16)
    return w, projected, hidden


def projection(w, rows, on):
    from sgl_nvidia_mega.kimi_k31.dsa import indexer

    saved = indexer.DSA_QFN
    indexer.DSA_QFN = on
    try:
        return indexer.IndexProjection(w, rows)
    finally:
        indexer.DSA_QFN = saved


def run(proj, projected, hidden):
    H.poison(proj.query, proj.gate)
    proj.prepare_gate(hidden)
    q8, gates = proj.project_query(projected)
    torch.cuda.synchronize()
    rows = projected.shape[0]
    return q8.view(torch.uint8).reshape(rows, 32, 128).clone(), gates.reshape(rows, 32).clone()


def fp64_codes(projected, w):
    """Independent reference: FP64 RMSNorm and product, ue8m0 scale of the FP64 amax, e4m3 RN of the clamped quotient."""
    x = projected[:, :1536].double()
    rstd = torch.rsqrt((x * x).mean(-1, keepdim=True) + w.epsilon)
    q = ((x * rstd * w.query_norm.double()[None, :]) @ w.query.double().t()).view(x.shape[0], 32, 128)
    amax = q.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    return (q / scale).clamp(-448.0, 448.0).float().to(torch.float8_e4m3fn).view(torch.uint8)


def canon_zero(codes):
    return torch.where(codes == 0x80, torch.zeros_like(codes), codes)


@needs_gpu
@pytest.mark.parametrize("rows", (2, 9, 33, 40, 64, 72, 96, 128), ids=lambda r: f"b{r}")
@pytest.mark.parametrize("seed", (1, 2))
def test_qfn_fp64(rows, seed):
    H.load_plugin(MUTATIONS)
    w, projected, hidden = case(rows, seed)
    stock = projection(w, rows, False)
    qfn = projection(w, rows, True)
    tag = f"b{rows}.s{seed}"
    assert stock.folded is None and qfn.folded is not None, f"{tag}: QFN not engaged (fold missing)"
    scodes, sgates = run(stock, projected, hidden)
    codes, weights = run(qfn, projected, hidden)
    rc = canon_zero(fp64_codes(projected, w))
    mism = float((canon_zero(codes) != rc).float().mean())
    smism = float((canon_zero(scodes) != rc).float().mean())
    vs_stock = float((canon_zero(codes) != canon_zero(scodes)).float().mean())
    nan = int(((codes & 0x7F) == 0x7F).sum())
    row = {"kind": "qfn_fp64", "item": tag, "mism": mism, "stock_mism": smism, "line": 1.25 * smism + 0.001,
           "vs_stock": vs_stock, "nan": nan}
    H.record(row)
    print("QFN_FP64", row, flush=True)
    assert nan == 0, f"{tag}: NaN codes {row}"
    assert mism <= row["line"], f"{tag}: codes further from FP64 than stock's: {row}"
    assert vs_stock > 0, f"{tag}: QFN not engaged (codes bitwise stock's): {row}"
    assert bool(torch.isfinite(weights).all()), f"{tag}: non-finite head weights"
    ratio = weights.double() / sgates.double()
    pow2 = (ratio == 1.0) | (ratio == 2.0) | (ratio == 0.5)
    eq = float((weights == sgates).float().mean())
    wrow = {"kind": "qfn_w", "item": tag, "w_eq": eq, "w_not_pow2": int((~pow2).sum())}
    H.record(wrow)
    print("QFN_W", wrow, flush=True)
    assert wrow["w_not_pow2"] == 0 and eq >= 0.98, f"{tag}: head weights off stock's: {wrow}"


@needs_gpu
@pytest.mark.parametrize("rows", (40, 128), ids=lambda r: f"b{r}")
def test_qfn_off_is_stock(rows):
    H.load_plugin(MUTATIONS)
    from sgl_kernel import rmsnorm
    bf16_head_gate_q_scale = H.bf16_head_gate_q_scale  # the release host has no standalone head-gate function
    from sglang.kernels.ops.quantization.act_quant_triton import act_quant

    w, projected, hidden = case(rows, 3)
    codes, weights = run(projection(w, rows, False), projected, hidden)
    query = torch.mm(rmsnorm(projected[:, :1536].contiguous(), w.query_norm, w.epsilon), w.query.T).view(rows, 32, 128)
    rcodes, scales = act_quant(query, 128, "ue8m0")
    rgates = bf16_head_gate_q_scale(torch.mm(hidden, w.gate.T), scales, 128**-0.5).view(rows, 32)
    row = {"kind": "qfn_off", "item": f"b{rows}", "codes_ne": int((codes != rcodes.view(torch.uint8)).sum()),
           "w_ne": int((weights != rgates).sum())}
    H.record(row)
    print("QFN_OFF", row, flush=True)
    assert row["codes_ne"] == 0 and row["w_ne"] == 0, f"knob-off projection is not the stock chain: {row}"


@needs_gpu
def test_qfn_soak():
    H.load_plugin(MUTATIONS)
    rows = 128
    w, projected, hidden = case(rows, 5)
    qfn = projection(w, rows, True)
    first = run(qfn, projected, hidden)
    rc = canon_zero(fp64_codes(projected, w))
    scodes, _ = run(projection(w, rows, False), projected, hidden)
    mism = float((canon_zero(first[0]) != rc).float().mean())
    smism = float((canon_zero(scodes) != rc).float().mean())
    differ = 0
    for _ in range(100):
        out = run(qfn, projected, hidden)
        differ += int(not (torch.equal(out[0], first[0]) and torch.equal(out[1], first[1])))
    row = {"kind": "qfn_soak", "item": "b128", "differ": differ, "mism": mism, "stock_mism": smism}
    H.record(row)
    print("QFN_SOAK", row, flush=True)
    assert differ == 0, f"launches differ: {row}"
    assert mism <= 1.25 * smism + 0.001, f"codes further from FP64 than stock's: {row}"
