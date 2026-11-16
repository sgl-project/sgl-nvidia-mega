# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Query-front index query (K31MK_DSA_QF; kimi_k31/dsa/index_q.cuh with kQF): the head gate is read from the input
projection's extra BF16 columns, and the same kernel stores the new tokens' index-K rows, vs STOCK's functions.

Packed rows under QF: [q_a | kv_a | gate | index-K (indexer.wk, 128) | head gate (indexer.weights_proj, 32)].

Gates (stated before any run):
- test_qf_domain (the exact-arithmetic tensors of test_k31_index_q.domain_case; the gate columns hold the exact BF16
  gate, so stock's GEMM gives the same gate in any order): FP8 codes and head weights BITWISE equal to stock's functions
  in every cell (FP8 -0 and +0 count as equal, as in test_k31_index_q).
- test_qf_query (random tensors; gate columns = stock's BF16 torch.mm gate): FP8 codes BITWISE equal to the non-QF fused
  kernel on the same q_a (that path is unchanged); codes within 1 step of stock in every cell and differing in <= 1 %;
  head weights within 2^-7 relative of stock's.
- test_qf_index_k (random index-K rows with a nonzero mean, random FP32 LayerNorm weight and bias, distinct slots in a
  poisoned page pool, one negative slot): every byte outside the stored rows unchanged; stored rows vs stock's
  flashinfer layernorm + fused_store_index_k_cache on a copy of the pool: FP8 codes within 1 step and <= 1 % differing,
  FP32 scales within 2^-7 relative; the same lines against an FP64 LayerNorm reference with stock's quantizer (stock vs
  that reference is recorded as the control).
- test_qf16_halves (K31MK_DSA_QF_MAX_ROWS=16; 9 / 12 / 16 rows, random tensors as test_qf_index_k, gate columns = stock's
  BF16 gate, one negative slot): FP8 codes, head weights and the whole index-K page pool BITWISE equal to two 8-row-kernel
  launches on the row halves (rows 0-7 and 8+; each row's arithmetic is the same); codes within 1 step of stock and
  differing in <= 1 %; weights within 2^-7 relative of stock's.
Mutations (each must FAIL): qf_gate (the next head's gate column), qf_k_pitch (index-K row pitch 132), qf_k_beta
(LayerNorm bias dropped), qf_k_mean (mean not subtracted); 16 rows only: qf16_tile (the second n8 tile reads rows 0-7),
qf16_norm (row w + 8 normalized with row w's rsqrt), qf16_store (no index-K store for rows >= 8).
K31_TEST_HEADS (default 12): MLA heads per rank (TP8 12, TP16 6); sets the index-K / gate column offsets.
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
    "qf_gate": V.edits(_CUH, 6),
    "qf_k_pitch": V.edits(_CUH, 7),
    "qf_k_beta": V.edits(_CUH, 8),
    "qf_k_mean": V.edits(_CUH, 9),
    "qf16_tile": V.edits(_CUH, 10),
    "qf16_norm": V.edits(_CUH, 11),
    "qf16_store": V.edits(_CUH, 12),
}
HEADS = int(os.environ.get("K31_TEST_HEADS", "12"))
K_COL = 2112 + 128 * HEADS  # TP16 2880, TP8 3648
G_COL = K_COL + 128  # TP16 3008
PITCH = G_COL + 32  # TP16 3040
PAGES = 6
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


def key_norm(seed):
    g = H.generator(seed)
    gamma = 1 + 0.2 * torch.randn(128, generator=g, device="cuda")
    beta = 0.5 * torch.randn(128, generator=g, device="cuda")
    return gamma.float().contiguous(), beta.float().contiguous(), 1e-6


def stock_query(projected, hidden, w):
    from sgl_kernel import rmsnorm
    bf16_head_gate_q_scale = H.bf16_head_gate_q_scale  # the release host has no standalone head-gate function
    from sglang.kernels.ops.quantization.act_quant_triton import act_quant

    rows = projected.shape[0]
    query = torch.mm(rmsnorm(projected[:, :1536].contiguous(), w.query_norm, w.epsilon), w.query.T).view(rows, 32, 128)
    codes, scales = act_quant(query, 128, "ue8m0")
    gate = torch.mm(hidden, w.gate.T)
    gates = bf16_head_gate_q_scale(gate, scales, 128**-0.5)
    return query, codes.view(torch.uint8), scales.view(rows, 32), gates.view(rows, 32), gate


def stock_store(projected, cache, slots, norm):
    from flashinfer.norm import layernorm
    from sglang.kernels.ops.kvcache.dsa_fused_store_index_cache import fused_store_index_k_cache

    gamma, beta, eps = norm
    key = layernorm(projected[:, K_COL : K_COL + 128], gamma, beta, eps)
    fused_store_index_k_cache(key, cache, slots, 64)


def ref_store(projected, cache, slots, norm):
    """FP64 LayerNorm -> BF16, then stock's quantizer (FP32 scale = max(amax, 1e-4) / 448, codes of k * (1 / scale))."""
    gamma, beta, eps = norm
    x = projected[:, K_COL : K_COL + 128].double()
    mean = x.mean(-1, keepdim=True)
    var = ((x - mean) ** 2).mean(-1, keepdim=True)
    k = ((x - mean) / torch.sqrt(var + eps) * gamma.double() + beta.double()).to(torch.bfloat16).float()
    scale = torch.clamp(k.abs().amax(-1, keepdim=True), min=1e-4) / 448.0
    codes = torch.clamp(k * (1.0 / scale), -448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    for r in range(projected.shape[0]):
        s = int(slots[r])
        if s < 0:
            continue
        page, off = s // 64, s % 64
        cache[page, off * 128 : off * 128 + 128] = codes[r]
        cache[page, 8192 + off * 4 : 8192 + off * 4 + 4] = scale[r].view(torch.uint8)


def rows_of(cache, slots):
    """(codes [n, 128] uint8, scales [n] FP32) of the non-negative slots."""
    out_c, out_s = [], []
    for s in slots.tolist():
        if s < 0:
            continue
        page, off = s // 64, s % 64
        out_c.append(cache[page, off * 128 : off * 128 + 128])
        out_s.append(cache[page, 8192 + off * 4 : 8192 + off * 4 + 4].view(torch.float32))
    return torch.stack(out_c), torch.cat(out_s)


def run_qf(w, projected, slots, cache, norm):
    from sgl_nvidia_mega.kimi_k31.dsa.index_q import FusedIndexQuery

    fused = FusedIndexQuery(w, projected.shape[0])
    H.poison(fused.query, fused.gates)
    q8, gates = fused.run_qf(projected, K_COL, G_COL, slots, cache, norm)
    torch.cuda.synchronize()
    return q8.view(torch.uint8).reshape(-1, 32, 128).clone(), gates.reshape(-1, 32).clone()


def canon_zero(codes):
    return torch.where(codes == 0x80, torch.zeros_like(codes), codes)


@needs_gpu
@pytest.mark.parametrize("half", (0, 1))
@pytest.mark.parametrize("e0", (-20, 12))
def test_qf_domain(e0, half):
    H.load_plugin(MUTATIONS)
    import test_k31_index_q as base

    g = H.generator(3000 + e0 + 50 * half)
    bad = {"codes": 0, "weights": 0}
    cells = 0
    dummy = torch.zeros((1, 64 * 132), dtype=torch.uint8, device="cuda")
    for part in range(8 * half, 8 * half + 8):
        w, projected0, hidden, q, gate = base.domain_case(e0, part, g)
        squery, scodes, _, sgates, sgate = stock_query(projected0, hidden, w)
        assert torch.equal(squery, q) and torch.equal(sgate, gate), "construction not exact"
        projected = torch.randn((8, PITCH), generator=g, device="cuda").to(torch.bfloat16)
        projected[:, :1536] = projected0[:, :1536]
        projected[:, G_COL : G_COL + 32] = gate
        slots = torch.full((8,), -1, dtype=torch.int64, device="cuda")
        codes, weights = run_qf(w, projected, slots, dummy, key_norm(1))
        bad["codes"] += int((canon_zero(codes) != canon_zero(scodes)).sum())
        bad["weights"] += int((weights != sgates).sum())
        cells += scodes.numel()
    row = {"kind": "qf_domain", "item": f"e0{e0}.half{half}", "code_cells": cells, **bad}
    H.record(row)
    print("QF_DOMAIN", row, flush=True)
    assert int(dummy.sum()) == 0, "a negative slot was stored"
    assert bad["codes"] == 0 and bad["weights"] == 0, f"QF index query != stock: {row}"


@needs_gpu
@pytest.mark.parametrize("rows", (1, 3, 8), ids=lambda r: f"b{r}")
@pytest.mark.parametrize("seed", (1, 2))
def test_qf_query(rows, seed):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.dsa.index_q import FusedIndexQuery

    w = index_weights(100 * seed + rows)
    g = H.generator(17 * seed + rows)
    projected = torch.randn((rows, PITCH), generator=g, device="cuda").to(torch.bfloat16)
    hidden = torch.randn((rows, 7168), generator=g, device="cuda").to(torch.bfloat16)
    _, scodes, _, sgates, sgate = stock_query(projected, hidden, w)
    projected[:, G_COL : G_COL + 32] = sgate
    fused = FusedIndexQuery(w, rows)
    fq8, _ = fused.run(hidden, projected)
    fq8 = fq8.view(torch.uint8).reshape(rows, 32, 128).clone()
    slots = torch.full((rows,), -1, dtype=torch.int64, device="cuda")
    dummy = torch.zeros((1, 64 * 132), dtype=torch.uint8, device="cuda")
    codes, weights = run_qf(w, projected, slots, dummy, key_norm(2))
    tag = f"b{rows}.s{seed}"
    assert torch.equal(codes, fq8), f"{tag}: QF codes != the fused kernel's on the same q_a"
    H.record({"kind": "check", **R.fp8_ulp_check(f"qf_vs_stock.{tag}", codes, scodes)})
    H.check(f"qfw_vs_stock.{tag}", weights, sgates, 0.0, 2.0**-7)


@needs_gpu
@pytest.mark.parametrize("rows", (1, 5, 8), ids=lambda r: f"b{r}")
@pytest.mark.parametrize("seed", (1, 2))
def test_qf_index_k(rows, seed):
    H.load_plugin(MUTATIONS)
    w = index_weights(500 + 10 * seed + rows)
    g = H.generator(41 * seed + rows)
    projected = torch.randn((rows, PITCH), generator=g, device="cuda").to(torch.bfloat16)
    key = 0.5 + 2.0 * torch.randn((rows, 128), generator=g, device="cuda")
    projected[:, K_COL : K_COL + 128] = key.to(torch.bfloat16)
    perm = torch.randperm(PAGES * 64, generator=g, device="cuda")[:rows]
    slots = perm.to(torch.int64).contiguous()
    if rows > 1:
        slots[rows // 2] = -1
    norm = key_norm(7 * seed + rows)
    pool = torch.full((PAGES, 64 * 132), 0xA5, dtype=torch.uint8, device="cuda")
    got, ref, stock = pool.clone(), pool.clone(), pool.clone()
    run_qf(w, projected, slots, got, norm)
    stock_store(projected, stock, slots, norm)
    ref_store(projected, ref, slots, norm)
    torch.cuda.synchronize()
    tag = f"b{rows}.s{seed}"
    # Bytes outside the stored rows: untouched (the stock pool shows which bytes a store owns).
    owned = stock != pool
    for s in slots.tolist():
        if s >= 0:
            page, off = s // 64, s % 64
            owned[page, off * 128 : off * 128 + 128] = True
            owned[page, 8192 + off * 4 : 8192 + off * 4 + 4] = True
    assert torch.equal(got[~owned], pool[~owned]), f"{tag}: QF wrote outside the stored rows"
    gc, gs = rows_of(got, slots)
    sc, ss = rows_of(stock, slots)
    rc, rs = rows_of(ref, slots)
    errs = []
    for name, codes, scales, rcodes, rscales in (("qf_vs_stock", gc, gs, sc, ss), ("qf_vs_ref", gc, gs, rc, rs),
                                                  ("stock_vs_ref", sc, ss, rc, rs)):
        try:
            H.record({"kind": "check", **R.fp8_ulp_check(f"{name}.{tag}", codes, rcodes)})
            H.check(f"{name}.scale.{tag}", scales, rscales, 0.0, 2.0**-7)
        except AssertionError as e:
            H.record({"kind": "check_fail", "item": f"{name}.{tag}", "msg": str(e)})
            errs.append(str(e))
    print("QF_K", tag, float((gc == sc).float().mean()), float((gs == ss).float().mean()), flush=True)
    assert not [e for e in errs if "stock_vs_ref" not in e], errs


@needs_gpu
@pytest.mark.parametrize("rows", (9, 12, 16), ids=lambda r: f"b{r}")
@pytest.mark.parametrize("seed", (1, 2))
def test_qf16_halves(rows, seed):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.dsa.index_q import FusedIndexQuery

    w = index_weights(900 + 10 * seed + rows)
    g = H.generator(53 * seed + rows)
    projected = torch.randn((rows, PITCH), generator=g, device="cuda").to(torch.bfloat16)
    hidden = torch.randn((rows, 7168), generator=g, device="cuda").to(torch.bfloat16)
    key = 0.5 + 2.0 * torch.randn((rows, 128), generator=g, device="cuda")
    projected[:, K_COL : K_COL + 128] = key.to(torch.bfloat16)
    _, scodes, _, sgates, sgate = stock_query(projected, hidden, w)
    projected[:, G_COL : G_COL + 32] = sgate
    slots = torch.randperm(PAGES * 64, generator=g, device="cuda")[:rows].to(torch.int64).contiguous()
    slots[rows // 2] = -1
    norm = key_norm(11 * seed + rows)
    pool = torch.full((PAGES, 64 * 132), 0xA5, dtype=torch.uint8, device="cuda")
    got, ref = pool.clone(), pool.clone()
    wide = FusedIndexQuery(w, rows, qf_only=True)
    H.poison(wide.query, wide.gates)
    q8, gates = wide.run_qf(projected, K_COL, G_COL, slots, got, norm)
    codes, weights = q8.view(torch.uint8).reshape(rows, 32, 128).clone(), gates.reshape(rows, 32).clone()
    halves = []
    for lo, hi in ((0, 8), (8, rows)):
        part = FusedIndexQuery(w, hi - lo)
        H.poison(part.query, part.gates)
        hq, hg = part.run_qf(projected[lo:hi], K_COL, G_COL, slots[lo:hi], ref, norm)
        halves.append((hq.view(torch.uint8).reshape(hi - lo, 32, 128).clone(), hg.reshape(hi - lo, 32).clone()))
    torch.cuda.synchronize()
    rcodes = torch.cat([h[0] for h in halves])
    rweights = torch.cat([h[1] for h in halves])
    tag = f"b{rows}.s{seed}"
    row = {"kind": "qf16", "item": tag, "codes_ne": int((codes != rcodes).sum()), "weights_ne": int((weights != rweights).sum()),
           "pool_ne": int((got != ref).sum()), "pool_written": int((got != pool).sum())}
    H.record(row)
    print("QF16", row, flush=True)
    assert row["pool_written"] > 0, f"{tag}: no index-K row stored"
    assert row["codes_ne"] == 0 and row["weights_ne"] == 0 and row["pool_ne"] == 0, f"{tag}: 16-row QF != two 8-row launches: {row}"
    H.record({"kind": "check", **R.fp8_ulp_check(f"qf16_vs_stock.{tag}", codes, scodes)})
    H.check(f"qf16w_vs_stock.{tag}", weights, sgates, 0.0, 2.0**-7)
