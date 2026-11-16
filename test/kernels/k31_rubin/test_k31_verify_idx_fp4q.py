#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""In-path wiring test of the indexer-query toggle's fp4 setting (K31MK_DSA_IDX_Q=fp4 with K31MK_DSA_IDX_FP4=1) under the MID query-front
words (K31MK_DSA_QF=1, K31MK_DSA_QFP=1 at >= 72 rows, K31MK_DSA_QFN=1 at >= 33 rows, K31MK_DSA_LOGITS_EF=2, K31MK_DSA_DG_VENDOR=1) on one GPU:
the REAL plugin verify objects on an MXFP4 index-K pool, TP16 per-rank MLA heads (K31_TEST_HEADS=6), the QF-packed projection pitch,
random weights, eager and CUDA graph. .

Bundle-entry correctness (fixed before the first run). Arms (Q8, context 12,000), each with its MXFP4 query producer:
  B1 = 8 rows: the query front, index_q.cuh kQ4 (Fp4FusedIndexQuery.run_qf); B2 = 16 rows: the stock chain (rmsnorm, wq_b GEMM,
  idx_fp4.quantize_query); B5 = 40 rows: QFN (folded GEMM + idx_fp4_q.qfn_quantize_query), gate GEMM; B12 = 96 rows: QFP (index-K and gate
  from the projection, QFN query).
Checks (fixed before the run):
  Q0 the profile's form: B1 qf + Fp4FusedIndexQuery; B2 no qf / qfp / fused index query / fold; B5 fold, no qfp; B12 qfp.
  C1 the step's index-K rows vs the host quantizer of the form's key (as test_k31_verify_idx_fp4k_qf: exact, or the F3 lines for QF).
  C2 VerifySelection receives an idx_fp4.Fp4Query (q4 int8 [rows, 32, 64], sf int32 [rows, 32]).
  C2q that query vs quantize_query(the form's reference BF16 query, the form's gate): codes <= 0.5 % differing within one step, exponents
      <= 1 % within 1; head weights within 2^-7 relative (BITWISE where the gate is read from the projection: B1, B12). References: B1 / B2
      rmsnorm(q_a) @ wq_b^T; B5 / B12 bf16(bf16(q_a @ fold^T) * rsqrt(mean(q_a^2) + eps)) in torch FP32; gates F.linear(hidden, w_proj) or the
      projection's gate columns.
  C2c selection vs the FP64 top-2048 of sum_h w_h relu(q_h . k) from the RECEIVED MXFP4 query (decoded) / weights and the decoded pool:
      overlap >= 0.99, no position outside the causal length.
  C3 CUDA-graph replay == eager (index-K rows, sorted selection).
  C4 engagement: dg_logits_fp4 (vendored Q8 MXFP4 logits) engaged; dg_logits_fp4k not; idx_fp4_q STATS query_qf > 0 and query_qfn > 0,
      idx_fp4 STATS query > 0 (B2), logits == 0 (no stock-DeepGEMM fallback at Q8).
Mutants (each must FAIL a check; K31_MUTATION=<name>): qfp_gate (QFP's MXFP4 query reads the index-K columns as its gate), qfn_norm (QFN
  rows lose the rstd: idx_fp4.quantize_query on the folded GEMM output), q_route (the fp4 setting builds the FP8 fused index query).
Exit 0 all pass, 1 any check fails, 2 setup error. JSON lines; DONE last.
"""
from __future__ import annotations

import json
import math
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("K31_TEST_HEADS", "6")
os.environ.update({"K31MK_DSA_IDX_FP4": "1", "K31MK_DSA_IDX_Q": "fp4", "K31MK_DSA_QF": "1", "K31MK_DSA_QFP": "1",
                   "K31MK_DSA_QFP_MIN_ROWS": "72", "K31MK_DSA_QFN": "1", "K31MK_DSA_QFN_MIN_ROWS": "33", "K31MK_INDEX_Q_OFF_WRITER_WIDE": "1",
                   "K31MK_DSA_DG_VENDOR": "1", "K31MK_DSA_LOGITS_EF": "2"})
os.environ.pop("K31MK_DSA_IDX_FP4_FP8Q", None)
os.environ.pop("K31MK_FUSED_INDEX_Q_VERIFY", None)

import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _harness as H  # noqa: E402

V_FILE = "kimi_k31/dsa/verify.py"
MUTATIONS = {
    "qfp_gate": (V_FILE, "out = qfn_quantize_query(ip.query, projected, projected[:, g_col : g_col + 32], ip.weights.epsilon, ip.fp4)",
                 "out = qfn_quantize_query(ip.query, projected, projected[:, k_col : k_col + 32], ip.weights.epsilon, ip.fp4)"),
    "qfn_norm": (V_FILE, "return qfn_quantize_query(self.query, projected, self.gate, self.weights.epsilon, self.fp4)",
                 "return idx_fp4.quantize_query(self.query, self.gate, self.fp4)"),
    "q_route": (V_FILE, "from .idx_fp4_q import Fp4FusedIndexQuery as FusedIndexQuery", "from .index_q import FusedIndexQuery"),
}

Q, CTX = 8, 12000
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)
OKS = []


def emit(**kw):
    print(json.dumps(kw), flush=True)


def check(name, ok, **kw):
    emit(kind="check", name=name, ok=bool(ok), **kw)
    OKS.append(bool(ok))


def decode(vals_u8, sf_u8):
    v = vals_u8.to(torch.int64).cpu()
    codes = torch.stack([v & 0xF, v >> 4], dim=-1).reshape(v.shape[0], 128)
    mag = E2M1[codes & 7]
    val = torch.where((codes & 8) != 0, -mag, mag)
    scale = torch.pow(2.0, (sf_u8.to(torch.int64).cpu() - 127).to(torch.float64))
    return (val.view(-1, 4, 32) * scale[:, :, None]).reshape(-1, 128)


def rows_at(pool, slots, page_b):
    s = slots.to(torch.int64)
    flat = pool.view(-1)
    base = (s // 64) * page_b
    vi = (base + (s % 64) * 64)[:, None] + torch.arange(64, device=pool.device)[None, :]
    si = (base + 4096 + (s % 64) * 4)[:, None] + torch.arange(4, device=pool.device)[None, :]
    return flat[vi], flat[si]


def tol_compare(got, ref_q4, ref_sf):
    def codes(q4):
        v = q4.view(torch.uint8).reshape(-1, 64).to(torch.int32)
        return torch.stack([v & 0xF, v >> 4], dim=-1).reshape(-1, 128)

    si = lambda c: torch.where((c & 8) != 0, -(c & 7), c & 7)  # noqa: E731
    gc, rc = codes(got.q4), codes(ref_q4)
    gs = got.sf.contiguous().view(torch.uint8).to(torch.int32)
    rs = ref_sf.contiguous().view(torch.uint8).to(torch.int32)
    kw = dict(code_diff_frac=float((gc != rc).float().mean()), code_max_step=int((si(gc) - si(rc)).abs().max()),
              sf_diff_frac=float((gs != rs).float().mean()), sf_max_diff=int((gs - rs).abs().max()))
    return kw["code_diff_frac"] <= 0.005 and kw["code_max_step"] <= 1 and kw["sf_diff_frac"] <= 0.01 and kw["sf_max_diff"] <= 1, kw


def decode_query(q4, sf, rows):
    v = q4.view(torch.uint8).reshape(rows, 32, 64).to(torch.int64).cpu()
    codes = torch.stack([v & 0xF, v >> 4], dim=-1).reshape(rows, 32, 128)
    mag = E2M1[codes & 7]
    val = torch.where((codes & 8) != 0, -mag, mag)
    e = sf.contiguous().view(torch.uint8).reshape(rows, 32, 4).to(torch.int64).cpu()
    return (val.view(rows, 32, 4, 32) * torch.pow(2.0, (e - 127).to(torch.float64))[..., None]).reshape(rows, 32, 128)


def run_arm(B, seed, V, fp4, qref):
    import deep_gemm
    import torch.nn.functional as F
    from flashinfer.norm import layernorm
    from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.mla.prepared import PreparedMla
    from sgl_nvidia_mega.kimi_k31.dsa.indexer import IndexWeights
    from sgl_nvidia_mega.kimi_k31.verify_indexer import IndexKeyWeights, PreparedVerifyIndexKey
    from test_k31_mla import HEADS, fakes, kv_pool, make_weights, num_sms

    tag = f"B{B}"
    rows = B * Q
    g = H.generator(seed)
    w = make_weights(seed)
    wk = (torch.randn((128, 7168), generator=g, device="cuda") / math.sqrt(7168)).to(torch.bfloat16)
    k_col, g_col, pitch = V.qf_columns(HEADS)
    npg = math.ceil(CTX / 64)
    pages = B * npg + 2
    perm = torch.randperm(pages - 1, generator=g, device="cuda")[: B * npg] + 1
    table = perm.view(B, npg).to(torch.int32).contiguous()
    pool = torch.full((pages, fp4.PAGE_B), 0xA5, dtype=torch.uint8, device="cuda")
    allslots = (table.to(torch.int64)[:, :, None] * 64 + torch.arange(64, device="cuda")[None, None, :]).reshape(B, -1)[:, :CTX]
    prefill = torch.randn((B * (CTX - Q), 128), generator=g, device="cuda").to(torch.bfloat16)
    fp4.store_index_k(prefill, pool, allslots[:, : CTX - Q].reshape(-1))
    step_slots = allslots[:, CTX - Q:].reshape(-1).contiguous()
    kv = kv_pool(pages, g)
    lengths = torch.full((B,), CTX, dtype=torch.int32, device="cuda")
    causal = torch.cat([(CTX - Q + 1 + torch.arange(Q, device="cuda")) for _ in range(B)]).to(torch.int32)
    contexts = torch.full((B, Q), CTX, dtype=torch.int32, device="cuda")
    step = SimpleNamespace(sparse=1, batch_requests=B, query_width=Q, causal_lengths=causal, score_contexts=contexts,
                           score_schedule=deep_gemm.get_paged_mqa_logits_metadata(contexts, 64, num_sms()),
                           topk_plan=plan_topk_v2(causal))
    scratch = torch.empty((rows, HEADS, 576), dtype=torch.uint8, device="cuda")
    attention, _ = fakes(w)
    base = PreparedMla(attention, {rows: pitch}, scratch, max_context=1 << 17, core="trtllm", queries=(Q,))
    core = V.PreparedSparseVerifyMla(base, IndexWeights(w.wq_b, w.w_proj, w.w_qa, 1e-5), rows, 1 << 17)
    key_weights = IndexKeyWeights(wk, w.gamma, w.beta, w.k_eps)
    V._enable_qf(SimpleNamespace(profiles={rows: core}), key_weights)
    form = {"qf": core.qf is not None, "qfp": core.qfp is not None, "fused_iq": type(core.fused_index_q).__name__}
    form["fold"] = getattr(core.index_projection, "folded", None) is not None
    want = {1: {"qf": True, "qfp": False, "fused_iq": "Fp4FusedIndexQuery", "fold": False},
            2: {"qf": False, "qfp": False, "fused_iq": "NoneType", "fold": False},
            5: {"qf": False, "qfp": False, "fused_iq": "NoneType", "fold": True},
            12: {"qf": False, "qfp": True, "fused_iq": "NoneType", "fold": True}}[B]
    check(f"Q0_form_{tag}", form == want, **form)
    core.prepare_cache(kv)
    writer = V.SparseVerifyIndexKey(PreparedVerifyIndexKey(key_weights), core)
    writer.bind(pool, 64, rows=(rows,))
    core.step = step
    hidden_src = torch.randn((rows, 7168), generator=g, device="cuda").to(torch.bfloat16)
    proj_src = torch.randn((rows, pitch), generator=g, device="cuda").to(torch.bfloat16)
    first = torch.empty_like(hidden_src)
    projected = torch.empty_like(proj_src)
    out = torch.empty((rows, HEADS * 128), dtype=torch.bfloat16, device="cuda")

    def chain():
        first.copy_(hidden_src)
        writer.run(first, step_slots)
        projected.copy_(proj_src)
        core.run(projected, out, cache=kv, slots=step_slots, block_table=table, lengths=lengths, q=Q)

    def snap():
        torch.cuda.synchronize()
        v, s = rows_at(pool, step_slots, fp4.PAGE_B)
        return v.clone(), s.clone(), torch.sort(core.selection.selected[:rows].clone(), dim=-1).values, out.clone()

    seen = {}
    sel = core.selection
    orig_run = sel.run

    def spy(query, gates, cache, tbl, stp):
        seen["q"] = fp4.Fp4Query(query.q4.clone(), query.sf.clone()) if isinstance(query, fp4.Fp4Query) else query.clone()
        seen["g"] = gates.clone()
        return orig_run(query, gates, cache, tbl, stp)

    sel.run = spy
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        chain()
    torch.cuda.current_stream().wait_stream(s)
    eager = snap()
    del sel.run

    # C1 writer rows by form
    if core.qfp is not None or core.qf is not None:
        key_ref = layernorm(proj_src[:, k_col : k_col + 128], w.gamma, w.beta, w.k_eps)
    elif writer.packed_weight is not None:
        packed = torch.empty((rows, writer.packed_weight.shape[0]), dtype=torch.bfloat16, device="cuda")
        torch.mm(hidden_src, writer.packed_weight.T, out=packed)
        key_ref = layernorm(packed[:, :128], w.gamma, w.beta, w.k_eps)
    else:
        key_ref = layernorm(F.linear(hidden_src, wk), w.gamma, w.beta, w.k_eps)
    k4, ksf = qref(key_ref)
    v, sfb, selected, outv = eager
    form = "qf" if core.qf is not None else "qfp" if core.qfp is not None else ("packed" if writer.packed_weight is not None else "stock")
    rsf = ksf.contiguous().view(torch.uint8).view(-1, 4)
    if form == "qf":  # the kernel's LayerNorm: F3 lines of test_k31_index_q_qf_fp4
        gc = torch.stack([v & 0xF, v >> 4], dim=-1).reshape(-1, 128).to(torch.int32)
        rv = k4.view(torch.uint8)
        rc = torch.stack([rv & 0xF, rv >> 4], dim=-1).reshape(-1, 128).to(torch.int32)
        si = lambda c: torch.where((c & 8) != 0, -(c & 7), c & 7)  # noqa: E731
        step = int((si(gc) - si(rc)).abs().max())
        cfrac = float((gc != rc).float().mean())
        sdiff = int((sfb.to(torch.int32) - rsf.to(torch.int32)).abs().max())
        sfrac = float((sfb != rsf).float().mean())
        check(f"C1_writer_rows_{tag}", cfrac <= 0.005 and step <= 1 and sfrac <= 0.01 and sdiff <= 1, form=form, code_diff_frac=cfrac,
              code_max_step=step, sf_diff_frac=sfrac, sf_max_diff=sdiff)
    else:
        check(f"C1_writer_rows_{tag}", torch.equal(v, k4.view(torch.uint8)) and torch.equal(sfb, rsf),
              val_mismatch=int((v != k4.view(torch.uint8)).sum()), form=form)
    qx, gx = seen.get("q"), seen.get("g")
    is4 = isinstance(qx, fp4.Fp4Query) and tuple(qx.q4.shape) == (rows, 32, 64) and qx.q4.dtype == torch.int8 and \
        tuple(qx.sf.shape) == (rows, 32) and qx.sf.dtype == torch.int32
    check(f"C2_fp4_query_{tag}", is4, type=type(qx).__name__)
    # C2q: the form's reference BF16 query and gate through idx_fp4.quantize_query
    from sgl_kernel import rmsnorm

    if form == "qf" or B == 2:
        qref_bf = torch.mm(rmsnorm(proj_src[:, :1536], w.w_qa, 1e-5), w.wq_b.T).view(rows, 32, 128)
    else:
        acc = torch.mm(proj_src[:, :1536], core.index_projection.folded.T).float()
        rstd = torch.rsqrt(proj_src[:, :1536].float().pow(2).sum(-1) / 1536 + 1e-5)
        qref_bf = (acc.to(torch.bfloat16).float() * rstd[:, None]).to(torch.bfloat16).view(rows, 32, 128)
    gate_from_proj = form in ("qf", "qfp")
    gref = proj_src[:, g_col : g_col + 32].contiguous() if gate_from_proj else F.linear(hidden_src, w.w_proj)
    rb = fp4.QueryBuffers(rows, qref_bf.device)
    rq, rw = fp4.quantize_query(qref_bf.contiguous(), gref, rb)
    torch.cuda.synchronize()
    ok_q, kw = tol_compare(qx, rq.q4, rq.sf)
    gxw = gx.reshape(rows, 32).float()
    rel = float(((gxw - rw.reshape(rows, 32)).abs() / rw.reshape(rows, 32).abs().clamp_min(1e-30)).max())
    wbit = torch.equal(gxw.view(torch.int32), rw.reshape(rows, 32).view(torch.int32))
    check(f"C2q_query_vs_reference_{tag}", ok_q and rel <= 2.0 ** -7 and (wbit or not gate_from_proj), w_max_rel=rel, w_bitwise=wbit, **kw)
    qd = decode_query(qx.q4, qx.sf, rows)
    wr = gx.reshape(rows, 32).to(torch.float64).cpu()
    worst, bad_pos = 1.0, 0
    for b in range(B):
        kvb, ksb = rows_at(pool, allslots[b], fp4.PAGE_B)
        kd = decode(kvb, ksb)
        slot_to_pos = {int(sl): p for p, sl in enumerate(allslots[b].tolist())}
        for j in range(Q):
            r = b * Q + j
            n = int(causal[r])
            sc = (wr[r][:, None] * (qd[r] @ kd[:n].T).clamp(min=0)).sum(0)
            want_set = set(torch.topk(sc, 2048).indices.tolist())
            got = [slot_to_pos.get(int(x), -1) for x in selected[r].tolist() if int(x) >= 0]
            bad_pos += sum(1 for p in got if p < 0 or p >= n)
            worst = min(worst, len(want_set & set(got)) / 2048)
    check(f"C2c_selection_vs_fp64_{tag}", worst >= 0.99 and bad_pos == 0, min_overlap=worst, outside_causal=bad_pos)
    gph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gph):
        chain()
    core.selection.selected.fill_(-1)
    sv = step_slots.to(torch.int64)
    vi = ((sv // 64) * fp4.PAGE_B + (sv % 64) * 64)[:, None] + torch.arange(64, device="cuda")[None, :]
    si = ((sv // 64) * fp4.PAGE_B + 4096 + (sv % 64) * 4)[:, None] + torch.arange(4, device="cuda")[None, :]
    pool.view(-1)[vi.reshape(-1)] = 0xA5
    pool.view(-1)[si.reshape(-1)] = 0xA5
    gph.replay()
    rep = snap()
    check(f"C3_graph_equals_eager_{tag}", torch.equal(rep[0], eager[0]) and torch.equal(rep[1], eager[1]) and torch.equal(rep[2], eager[2]),
          sel_differ=int((rep[2] != eager[2]).sum()))
    check(f"C5_output_finite_{tag}", bool(torch.isfinite(outv.float()).all().item()))


def main():
    try:
        H.load_plugin(MUTATIONS)
        from sgl_nvidia_mega.kimi_k31.dsa import dg_logits_fp4 as D4
        from sgl_nvidia_mega.kimi_k31.dsa import dg_logits_fp4k as K4
        from sgl_nvidia_mega.kimi_k31.dsa import idx_fp4 as fp4
        from sgl_nvidia_mega.kimi_k31.dsa import idx_fp4_q as fq
        from sgl_nvidia_mega.kimi_k31.dsa import verify as V
        from sglang.kernels.ops.quantization.fp4_indexer_quant import quantize_fp4_indexer_tensor as qref

        torch.zeros(1, device="cuda")
        assert V.IDX_FP4 and V.IDX_FP4_FP4Q and not V.IDX_FP4_FP8Q and V.DSA_QF and V.DSA_QFP == 1 and V.FUSED_INDEX_Q_VERIFY and V.IDX_FP4_Q8_VENDOR
    except Exception as ex:  # noqa: BLE001
        emit(kind="setup_error", err=repr(ex)[:500])
        raise
    for B, seed in ((1, 51), (2, 52), (5, 54), (12, 53)):
        try:
            run_arm(B, seed, V, fp4, qref)
        except Exception as ex:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            check(f"arm_ran_B{B}", False, err=repr(ex)[:400])
    check("C4_engagement", len(D4._engaged) == 1 and not getattr(K4, "_engaged", []) and fq.STATS["query_qf"] > 0 and fq.STATS["query_qfn"] > 0
          and fp4.STATS["query"] > 0 and fp4.STATS["writer"] > 0 and fp4.STATS["logits"] == 0,
          fp4_vendor_engaged=len(D4._engaged), fp4k_engaged=len(getattr(K4, "_engaged", [])), **fp4.STATS, **{f"q_{k}": v for k, v in fq.STATS.items()})
    emit(kind="DONE", all_ok=all(OKS), n=len(OKS), failed=sum(not o for o in OKS))
    return 0 if all(OKS) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        sys.exit(2)
