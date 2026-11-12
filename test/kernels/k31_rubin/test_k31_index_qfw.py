# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""QF-wide index query (K31MK_DSA_QFW; kimi_k31/dsa/index_qfw.cuh) on 9-32 rows vs an FP64 reference, stock's functions and the
served 8-row query front (index_q.cuh kQF). Packed rows as under QF: [q_a | kv_a | gate | index-K (128) | head gate (32)].

Gates (stated before any run):
- test_qfw_fp64 (rows 9 / 16 / 17 / 24 / 25 / 32 x seeds 1 / 2; random tensors, gate columns = stock's BF16 torch.mm gate):
  FP8 codes vs an independent FP64 reference (xn = x rstd w_qa, q = xn @ wq_b^T, ue8m0 scale of the FP64 amax, e4m3 RN):
  share of differing codes <= 1.25 x stock's share vs the same reference + 0.001 (stock = sgl_kernel rmsnorm -> torch.mm ->
  act_quant, the served chain's functions; its share is recorded as the control). Head weights vs stock's
  bf16_head_gate_q_scale: >= 98 % bitwise equal and every ratio exactly 1, 2 or 1/2 (the gate path is stock's; only the ue8m0
  scale may move by one binade). No NaN / Inf.
- test_qfw_rows (one 32-row input): each row's codes, weights and stored index-K are BITWISE the same in every instance and row
  count: 32 rows (kR 32), rows 0-23 (kR 24) and forced kR 32, rows 0-16 (kR 24), rows 0-8 (kR 16), rows 0-15 and rows 16-31
  as their own 16-row launches (kR 16).
- test_qfw_index_k (rows 9 / 24 / 32; nonzero-mean index-K, random FP32 LayerNorm affine, one negative slot, poisoned pool):
  the whole page pool BITWISE equal to the served 8-row query-front kernel (FusedIndexQuery.run_qf) run on 8-row chunks of
  the same rows into a copy of the pool; at least one row stored.
- test_qfw_soak (rows 24 / 32): 200 launches on one input, every launch BITWISE equal to the first; the first passes the
  test_qfw_fp64 code line. Under variant qfw_jitter (poisoned exchange slots + random delays before the DSMEM stores) the
  whole module must still PASS.
- test_qfw_fold_cache (added after a run found index_qfw.py's fold cache keyed by data_ptr alone
  returned an earlier test's freed weights' fold, i.e. alternate test_qfw_fp64 items read ~0.986 code mismatch): rows 24, run once,
  then copy new values into the SAME wq_b / w_qa memory; the fold must equal bf16(wq_b * w_qa) of the new values BITWISE, and the
  kernel's codes before and after must each pass the test_qfw_fp64 code line vs FP64 of the weights current at that launch.
Mutations (each must FAIL the module): qfw_slice (rank 7's ss partial dropped), qfw_rstd (rstd := 1), qfw_store16 (no
index-K store for rows >= 16), qfw_gate (the next head's gate column), qfw_tile (n8 tile 2 reads tile 0's rows), qfw_race
(jitter + the peers' partials read before the exchange's wait), qfw_nofold (w_qa not folded into wq_b at load), qfw_cachekey
(the qfw3 fold cache: data_ptr key only, no source reference / version check -> stale fold).
K31_TEST_HEADS (default 12): MLA heads per rank (TP8 12, TP16 6); sets the index-K / gate column offsets.
"""

import math
import os

import pytest
import torch

import _harness as H
import _hook_variants as V

_CUH = "kimi_k31/dsa/index_qfw.cuh"
MUTATIONS = {
    "qfw_slice": V.edits(_CUH, 1),
    "qfw_rstd": V.edits(_CUH, 2),
    "qfw_store16": V.edits(_CUH, 3),
    "qfw_gate": V.edits(_CUH, 4),
    "qfw_tile": V.edits(_CUH, 5),
    "qfw_race": V.edits(_CUH, 6),
    "qfw_jitter": V.edits(_CUH, 7),  # variant: must PASS
    "qfw_nofold": ("kimi_k31/dsa/index_qfw.py", "    folded = (query.float() * norm.float()[None, :])", "    folded = (query.float() * 1.0)"),
    "qfw_cachekey": ("kimi_k31/dsa/index_qfw.py", "if entry is not None and entry[2] == query._version and entry[3] == norm._version:", "if entry is not None:"),
}
HEADS = int(os.environ.get("K31_TEST_HEADS", "12"))
K_COL = 2112 + 128 * HEADS  # TP16 2880, TP8 3648
G_COL = K_COL + 128
PITCH = G_COL + 32
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


def case(rows, seed):
    """(weights, projected [rows, PITCH] with stock's BF16 gate at G_COL and nonzero-mean index-K at K_COL, stock outputs)."""
    w = index_weights(700 + 10 * seed + rows)
    g = H.generator(61 * seed + rows)
    projected = torch.randn((rows, PITCH), generator=g, device="cuda").to(torch.bfloat16)
    hidden = torch.randn((rows, 7168), generator=g, device="cuda").to(torch.bfloat16)
    key = 0.5 + 2.0 * torch.randn((rows, 128), generator=g, device="cuda")
    projected[:, K_COL : K_COL + 128] = key.to(torch.bfloat16)
    scodes, sgates, sgate = stock_query(projected, hidden, w)
    projected[:, G_COL : G_COL + 32] = sgate
    return w, projected, scodes, sgates, g


def stock_query(projected, hidden, w):
    from sgl_kernel import rmsnorm
    bf16_head_gate_q_scale = H.bf16_head_gate_q_scale  # the release host has no standalone head-gate function
    from sglang.kernels.ops.quantization.act_quant_triton import act_quant

    rows = projected.shape[0]
    query = torch.mm(rmsnorm(projected[:, :1536].contiguous(), w.query_norm, w.epsilon), w.query.T).view(rows, 32, 128)
    codes, scales = act_quant(query, 128, "ue8m0")
    gate = torch.mm(hidden, w.gate.T)
    gates = bf16_head_gate_q_scale(gate, scales, 128**-0.5)
    return codes.view(torch.uint8), gates.view(rows, 32), gate


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


def run_qfw(w, projected, slots, cache, norm, kr=0):
    from sgl_nvidia_mega.kimi_k31.dsa.index_qfw import FusedIndexQueryWide

    fused = FusedIndexQueryWide(w, projected.shape[0])
    H.poison(fused.query, fused.gates)
    q8, gates = fused.run_qf(projected, K_COL, G_COL, slots, cache, norm, kr)
    torch.cuda.synchronize()
    return q8.view(torch.uint8).reshape(-1, 32, 128).clone(), gates.reshape(-1, 32).clone()


def code_line(tag, codes, scodes, ref):
    """test_qfw_fp64's code line; returns the stats row."""
    rc = canon_zero(ref)
    mism = float((canon_zero(codes) != rc).float().mean())
    smism = float((canon_zero(scodes) != rc).float().mean())
    nan = int(((codes & 0x7F) == 0x7F).sum())
    row = {"kind": "qfw_fp64", "item": tag, "mism": mism, "stock_mism": smism, "line": 1.25 * smism + 0.001,
           "vs_stock": float((canon_zero(codes) != canon_zero(scodes)).float().mean()), "nan": nan}
    H.record(row)
    print("QFW_FP64", row, flush=True)
    assert nan == 0, f"{tag}: NaN codes {row}"
    assert mism <= row["line"], f"{tag}: codes further from FP64 than stock's: {row}"
    return row


@needs_gpu
@pytest.mark.parametrize("rows", (9, 16, 17, 24, 25, 32), ids=lambda r: f"b{r}")
@pytest.mark.parametrize("seed", (1, 2))
def test_qfw_fp64(rows, seed):
    H.load_plugin(MUTATIONS)
    w, projected, scodes, sgates, _ = case(rows, seed)
    slots = torch.full((rows,), -1, dtype=torch.int64, device="cuda")
    dummy = torch.zeros((1, 64 * 132), dtype=torch.uint8, device="cuda")
    codes, weights = run_qfw(w, projected, slots, dummy, key_norm(3))
    tag = f"b{rows}.s{seed}"
    code_line(tag, codes, scodes, fp64_codes(projected, w))
    assert int(dummy.sum()) == 0, f"{tag}: a negative slot was stored"
    assert bool(torch.isfinite(weights).all()), f"{tag}: non-finite head weights"
    ratio = weights.double() / sgates.double()
    pow2 = (ratio == 1.0) | (ratio == 2.0) | (ratio == 0.5)
    eq = float((weights == sgates).float().mean())
    row = {"kind": "qfw_w", "item": tag, "w_eq": eq, "w_not_pow2": int((~pow2).sum())}
    H.record(row)
    print("QFW_W", row, flush=True)
    assert row["w_not_pow2"] == 0 and eq >= 0.98, f"{tag}: head weights off stock's: {row}"


@needs_gpu
@pytest.mark.parametrize("seed", (1, 2))
def test_qfw_rows(seed):
    H.load_plugin(MUTATIONS)
    w, projected, _, _, g = case(32, seed)
    slots = torch.randperm(PAGES * 64, generator=g, device="cuda")[:32].to(torch.int64).contiguous()
    norm = key_norm(13 * seed)
    pool = torch.full((PAGES, 64 * 132), 0xA5, dtype=torch.uint8, device="cuda")

    def go(lo, hi, kr=0):
        cache = pool.clone()
        codes, weights = run_qfw(w, projected[lo:hi].contiguous(), slots[lo:hi].contiguous(), cache, norm, kr)
        rows_k = []
        for s in slots[lo:hi].tolist():
            page, off = s // 64, s % 64
            rows_k.append(torch.cat([cache[page, off * 128 : off * 128 + 128], cache[page, 8192 + off * 4 : 8192 + off * 4 + 4]]))
        return codes, weights, torch.stack(rows_k)

    full = go(0, 32)
    parts = {"b24": (go(0, 24), 0), "b24_kr32": (go(0, 24, 32), 0), "b17": (go(0, 17), 0), "b9": (go(0, 9), 0),
             "b16_lo": (go(0, 16), 0), "b16_hi": (go(16, 32), 16)}
    bad = {}
    for name, (out, lo) in parts.items():
        n = out[0].shape[0]
        bad[name] = [int((out[i] != full[i][lo : lo + n]).sum()) for i in range(3)]
    row = {"kind": "qfw_rows", "item": f"s{seed}", "codes_w_k_ne": bad,
           "k_written": int((full[2] != 0xA5).sum())}
    H.record(row)
    print("QFW_ROWS", row, flush=True)
    assert row["k_written"] > 0, "no index-K row stored"
    assert all(v == [0, 0, 0] for v in bad.values()), f"rows differ across instances / row counts: {row}"


@needs_gpu
@pytest.mark.parametrize("rows", (9, 24, 32), ids=lambda r: f"b{r}")
def test_qfw_index_k(rows):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.dsa.index_q import FusedIndexQuery

    w, projected, _, _, g = case(rows, 3)
    slots = torch.randperm(PAGES * 64, generator=g, device="cuda")[:rows].to(torch.int64).contiguous()
    slots[rows // 2] = -1
    norm = key_norm(17 + rows)
    pool = torch.full((PAGES, 64 * 132), 0xA5, dtype=torch.uint8, device="cuda")
    got, ref = pool.clone(), pool.clone()
    run_qfw(w, projected, slots, got, norm)
    for lo in range(0, rows, 8):
        hi = min(rows, lo + 8)
        part = FusedIndexQuery(w, hi - lo)
        part.run_qf(projected[lo:hi].contiguous(), K_COL, G_COL, slots[lo:hi].contiguous(), ref, norm)
    torch.cuda.synchronize()
    row = {"kind": "qfw_k", "item": f"b{rows}", "pool_ne": int((got != ref).sum()), "pool_written": int((got != pool).sum()),
           "ref_written": int((ref != pool).sum())}
    H.record(row)
    print("QFW_K", row, flush=True)
    assert row["ref_written"] > 0 and row["pool_written"] > 0, f"no index-K row stored: {row}"
    assert row["pool_ne"] == 0, f"index-K pool != the served 8-row query front on row chunks: {row}"


@needs_gpu
@pytest.mark.parametrize("rows", (24, 32), ids=lambda r: f"b{r}")
def test_qfw_soak(rows):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.dsa.index_qfw import FusedIndexQueryWide

    w, projected, scodes, _, _ = case(rows, 5)
    slots = torch.full((rows,), -1, dtype=torch.int64, device="cuda")
    dummy = torch.zeros((1, 64 * 132), dtype=torch.uint8, device="cuda")
    norm = key_norm(5)
    fused = FusedIndexQueryWide(w, rows)
    outs = []
    for _ in range(200):
        H.poison(fused.query, fused.gates)
        q8, gates = fused.run_qf(projected, K_COL, G_COL, slots, dummy, norm)
        outs.append((q8.view(torch.uint8).clone(), gates.clone()))
    torch.cuda.synchronize()
    first = outs[0]
    ne = sum(int(not (torch.equal(o[0], first[0]) and torch.equal(o[1], first[1]))) for o in outs[1:])
    row = {"kind": "qfw_soak", "item": f"b{rows}", "launches": len(outs), "launches_ne_first": ne}
    H.record(row)
    print("QFW_SOAK", row, flush=True)
    code_line(f"soak.b{rows}", first[0].reshape(rows, 32, 128), scodes, fp64_codes(projected, w))
    assert ne == 0, f"launches differ from the first: {row}"


@needs_gpu
def test_qfw_fold_cache():
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.dsa.index_qfw import folded_query

    rows = 24
    w, projected, scodes, _, _ = case(rows, 7)
    slots = torch.full((rows,), -1, dtype=torch.int64, device="cuda")
    dummy = torch.zeros((1, 64 * 132), dtype=torch.uint8, device="cuda")
    norm = key_norm(7)
    before, _ = run_qfw(w, projected, slots, dummy, norm)
    code_line("cache.before", before, scodes, fp64_codes(projected, w))
    fresh = index_weights(900)
    ptrs = (w.query.data_ptr(), w.query_norm.data_ptr())
    w.query.copy_(fresh.query)
    w.query_norm.copy_(fresh.query_norm)
    assert ptrs == (w.query.data_ptr(), w.query_norm.data_ptr()), "in-place update moved the weights"
    ref = (w.query.float() * w.query_norm.float()[None, :]).to(torch.bfloat16)
    fold_eq = bool(torch.equal(folded_query(w), ref))
    row = {"kind": "qfw_cache", "item": "b24", "fold_eq_ref": fold_eq}
    H.record(row)
    print("QFW_CACHE", row, flush=True)
    assert fold_eq, f"stale fold after an in-place weight update: {row}"
    hidden = torch.zeros((rows, 7168), dtype=torch.bfloat16, device="cuda")
    scodes2, _, _ = stock_query(projected, hidden, w)
    after, _ = run_qfw(w, projected, slots, dummy, norm)
    code_line("cache.after", after, scodes2, fp64_codes(projected, w))
