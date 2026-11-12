#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""In-path wiring test of the indexer-query toggle (K31MK_DSA_IDX_FP4=1, K31MK_DSA_IDX_Q = K31_TEST_IDX_Q) under the MID DSA words on the
The MID plugin form: the query front at 1-8 rows (K31MK_DSA_QF=1), the norm-folded query at >= 33
rows (K31MK_DSA_QFN=1, QFN_MIN_ROWS=33), QFP at >= 72 rows with the FORKED index-K LayerNorm + store (K31MK_DSA_QFP=2, QFP_MIN_ROWS=72,
K31MK_DSA_QFP_PAR_MAX_ROWS = K31_TEST_QFP_PAR, default 0 = fork at every QFP row), K31MK_INDEX_Q_OFF_WRITER=2 / _WIDE=1, K31MK_DSA_DG_VENDOR=1,
K31MK_DSA_LOGITS_EF=2, and K31MK_MLAQ_W=1 (MIN_ROWS 72) when K31_TEST_MLAQ_W=1; one GPU, the REAL plugin verify objects on an MXFP4 index-K
pool, TP16 per-rank MLA heads (K31_TEST_HEADS=6), random weights, eager + CUDA graph. RS (request sharding) needs a TP group: not here.

, bundle-entry correctness of the MID composition (does QFP=2's forked store write MXFP4 68 B rows?);
fixed before the first run.
Arms (Q8, context 12,000): B1 = 8 rows (QF, kKFP4 store; fp4: kQ4 query), B5 = 40 rows (QFN, stock writer), B9 = 72, B12 = 96, B16 = 128 rows (QFP;
forked iff PAR == 0 or rows <= PAR; MLAQ-W at >= 72 rows when on).
Amendment 1 (before any data with it): K31_TEST_QFP_MIN (default 72) sets K31MK_DSA_QFP_MIN_ROWS; with 33 (MID B3's
word) the arms are B1 / B5 / B8 / B12 (QFP fork at 40 / 64 / 96 rows) and Q0 expects QFP at rows >= K31_TEST_QFP_MIN. Checks unchanged.
Checks (fixed before the run):
  Q0 the profile's form (qf, qfp, qfp_par, fused index query type, folded query, mlaq_w) equals the words' specification above.
  C1 the step's index-K rows in the pool vs the host quantizer (sglang fp4_indexer_quant) of the form's key: QFP / QFN / stock EXACT
     (QFP key = flashinfer LayerNorm of the projection's index-K columns); QF within test_k31_index_q_qf_fp4's F3 lines.
  C2 the logits receive the setting's query (fp8: FP8 [rows, 32, 128]; fp4: idx_fp4.Fp4Query); C2q (fp4) that query vs
     idx_fp4.quantize_query of the form's reference BF16 query and gate (tolerances of test_k31_verify_idx_fp4_c8).
  C2c selection vs the FP64 top-2048 of sum_h w_h relu(q_h . k) from the received query (decoded) / weights and the decoded pool: overlap
      >= 0.99, no position outside the causal length.
  C2s the step's own rows (positions written by THIS step's store, which the fork must finish before scoring): each own position whose FP64
      score is above the FP64 2048th score by > 1e-3 relative is selected, and each one below it by > 1e-3 relative is not.
  C3 CUDA-graph replay == eager (index-K rows, sorted selection) after the step's rows are reset to 0xA5.
  C4 engagement: fp8 = dg_logits_fp4k once, vendored MXFP4 logits never, no MXFP4 query; fp4 = vendored dg_logits_fp4 once, dg_logits_fp4k
     never, idx_fp4_q query_qf > 0 and query_qfn > 0; both: idx_fp4 writer > 0, idx_fp4 logits == 0 (no stock-DeepGEMM fallback at Q8).
  C5 sparse MLA output finite.
Mutants (each must FAIL; K31_MUTATION=<name>): par_slot (the forked store writes at reversed slots), par_skip (the forked store writes nothing).
Exit 0 all pass, 1 any check fails, 2 setup error. JSON lines; DONE last.
"""
from __future__ import annotations

import json
import math
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("K31_TEST_HEADS", "6")
IDXQ = os.environ.get("K31_TEST_IDX_Q", "fp8")
PAR = int(os.environ.get("K31_TEST_QFP_PAR", "0"))
MLAQW = os.environ.get("K31_TEST_MLAQ_W", "0") == "1"
QFPMIN = int(os.environ.get("K31_TEST_QFP_MIN", "72"))
os.environ.update({"K31MK_DSA_IDX_FP4": "1", "K31MK_DSA_IDX_Q": IDXQ, "K31MK_DSA_QF": "1", "K31MK_DSA_QFN": "1", "K31MK_DSA_QFN_MIN_ROWS": "33",
                   "K31MK_DSA_QFP": "2", "K31MK_DSA_QFP_MIN_ROWS": str(QFPMIN), "K31MK_DSA_QFP_PAR_MAX_ROWS": str(PAR),
                   "K31MK_INDEX_Q_OFF_WRITER": "2", "K31MK_INDEX_Q_OFF_WRITER_WIDE": "1", "K31MK_DSA_DG_VENDOR": "1", "K31MK_DSA_LOGITS_EF": "2",
                   "K31MK_MLAQ_W": "1" if MLAQW else "0", "K31MK_MLAQ_W_MIN_ROWS": "72"})
FP4 = IDXQ == "fp4"
os.environ.pop("K31MK_DSA_IDX_FP4_FP8Q", None)
os.environ.pop("K31MK_FUSED_INDEX_Q_VERIFY", None)

import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _harness as H  # noqa: E402

V_FILE = "kimi_k31/dsa/verify.py"
_PAR_STORE = ("                if IDX_FP4:  # K31MK_DSA_IDX_FP4_FP8Q: MXFP4 index-K rows (the refusals keep the (k + q) form off QF / QFP)\n"
              "                    idx_fp4.store_index_k(key, self.index_cache, slots)\n")
MUTATIONS = {
    "par_slot": (V_FILE, _PAR_STORE, _PAR_STORE.replace("self.index_cache, slots)", "self.index_cache, slots.flip(0))")),
    "par_skip": (V_FILE, _PAR_STORE, _PAR_STORE.replace("idx_fp4.store_index_k(key, self.index_cache, slots)", "pass")),
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
    v = vals_u8.to(torch.int64)
    codes = torch.stack([v & 0xF, v >> 4], dim=-1).reshape(v.shape[0], 128)
    e2m1 = E2M1.to(v.device)
    mag = e2m1[codes & 7]
    val = torch.where((codes & 8) != 0, -mag, mag)
    scale = torch.pow(2.0, (sf_u8.to(torch.int64) - 127).to(torch.float64))
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
    v = q4.view(torch.uint8).reshape(rows, 32, 64).to(torch.int64)
    codes = torch.stack([v & 0xF, v >> 4], dim=-1).reshape(rows, 32, 128)
    e2m1 = E2M1.to(v.device)
    mag = e2m1[codes & 7]
    val = torch.where((codes & 8) != 0, -mag, mag)
    e = sf.contiguous().view(torch.uint8).reshape(rows, 32, 4).to(torch.int64)
    return (val.view(rows, 32, 4, 32) * torch.pow(2.0, (e - 127).to(torch.float64))[..., None]).reshape(rows, 32, 128)


def want_form(B):
    rows = B * Q
    qfp = rows >= QFPMIN
    return {"qf": rows <= 8, "qfp": qfp, "par": qfp and (PAR == 0 or rows <= PAR),
            "fused_iq": ("Fp4FusedIndexQuery" if FP4 else "FusedIndexQuery") if rows <= 8 else "NoneType",
            "fold": rows >= 33, "mlaq": MLAQW and rows >= 72}


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
    stand_in = SimpleNamespace(profiles={rows: core})
    V._enable_qf(stand_in, key_weights)
    if MLAQW:
        V._enable_mlaq_w(stand_in)
    form = {"qf": core.qf is not None, "qfp": core.qfp is not None, "par": bool(core.qfp_par),
            "fused_iq": type(core.fused_index_q).__name__, "fold": getattr(core.index_projection, "folded", None) is not None,
            "mlaq": core.mlaq_w is not None}
    check(f"Q0_form_{tag}", form == want_form(B), got=form, want=want_form(B))
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
    kform = "qf" if core.qf is not None else "qfp" if core.qfp is not None else ("packed" if writer.packed_weight is not None else "stock")
    rsf = ksf.contiguous().view(torch.uint8).view(-1, 4)
    if kform == "qf":  # the kernel's LayerNorm: F3 lines of test_k31_index_q_qf_fp4
        gc = torch.stack([v & 0xF, v >> 4], dim=-1).reshape(-1, 128).to(torch.int32)
        rv = k4.view(torch.uint8)
        rc = torch.stack([rv & 0xF, rv >> 4], dim=-1).reshape(-1, 128).to(torch.int32)
        si = lambda c: torch.where((c & 8) != 0, -(c & 7), c & 7)  # noqa: E731
        stepd = int((si(gc) - si(rc)).abs().max())
        cfrac = float((gc != rc).float().mean())
        sdiff = int((sfb.to(torch.int32) - rsf.to(torch.int32)).abs().max())
        sfrac = float((sfb != rsf).float().mean())
        check(f"C1_writer_rows_{tag}", cfrac <= 0.005 and stepd <= 1 and sfrac <= 0.01 and sdiff <= 1, form=kform, code_diff_frac=cfrac,
              code_max_step=stepd, sf_diff_frac=sfrac, sf_max_diff=sdiff)
    else:
        check(f"C1_writer_rows_{tag}", torch.equal(v, k4.view(torch.uint8)) and torch.equal(sfb, rsf),
              val_mismatch=int((v != k4.view(torch.uint8)).sum()), sf_mismatch=int((sfb != rsf).sum()), form=kform)
    qx, gx = seen.get("q"), seen.get("g")
    wr = gx.reshape(rows, 32).to(torch.float64)
    if not FP4:
        ok8 = isinstance(qx, torch.Tensor) and qx.dtype == torch.float8_e4m3fn and qx.shape[0] == rows and qx.numel() == rows * 4096
        check(f"C2_fp8_query_{tag}", ok8, type=type(qx).__name__)
        qd = qx.view(rows, 32, 128).to(torch.float64)
    else:
        is4 = isinstance(qx, fp4.Fp4Query) and tuple(qx.q4.shape) == (rows, 32, 64) and qx.q4.dtype == torch.int8 and \
            tuple(qx.sf.shape) == (rows, 32) and qx.sf.dtype == torch.int32
        check(f"C2_fp4_query_{tag}", is4, type=type(qx).__name__)
        from sgl_kernel import rmsnorm

        if kform == "qf" or getattr(core.index_projection, "folded", None) is None:
            qref_bf = torch.mm(rmsnorm(proj_src[:, :1536], w.w_qa, 1e-5), w.wq_b.T).view(rows, 32, 128)
        else:
            acc = torch.mm(proj_src[:, :1536], core.index_projection.folded.T).float()
            rstd = torch.rsqrt(proj_src[:, :1536].float().pow(2).sum(-1) / 1536 + 1e-5)
            qref_bf = (acc.to(torch.bfloat16).float() * rstd[:, None]).to(torch.bfloat16).view(rows, 32, 128)
        gate_from_proj = kform in ("qf", "qfp")
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
    worst, bad_pos, own_bad, own_sel = 1.0, 0, 0, 0
    for b in range(B):
        kvb, ksb = rows_at(pool, allslots[b], fp4.PAGE_B)
        kd = decode(kvb, ksb)
        slot_to_pos = {int(sl): p for p, sl in enumerate(allslots[b].tolist())}
        for j in range(Q):
            r = b * Q + j
            n = int(causal[r])
            sc = (wr[r][:, None] * (qd[r] @ kd[:n].T).clamp(min=0)).sum(0)
            top = torch.topk(sc, 2048)
            thr = float(top.values[-1])
            want_set = set(top.indices.tolist())
            got = set(slot_to_pos.get(int(x), -1) for x in selected[r].tolist() if int(x) >= 0)
            bad_pos += sum(1 for p in got if p < 0 or p >= n)
            worst = min(worst, len(want_set & got) / 2048)
            margin = 1e-3 * abs(thr) + 1e-12
            for p in range(CTX - Q, n):  # this step's own rows
                val = float(sc[p])
                if val > thr + margin and p not in got:
                    own_bad += 1
                elif val < thr - margin and p in got:
                    own_bad += 1
                own_sel += int(p in got)
    check(f"C2c_selection_vs_fp64_{tag}", worst >= 0.99 and bad_pos == 0, min_overlap=worst, outside_causal=bad_pos)
    check(f"C2s_own_rows_{tag}", own_bad == 0, own_misplaced=own_bad, own_selected=own_sel)
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
        assert V.IDX_FP4 and V.IDX_FP4_FP4Q == FP4 and V.IDX_FP4_FP8Q != FP4 and V.DSA_QF and V.DSA_QFP == 2 and V.DSA_QFP_PAR_MAX_ROWS == PAR
        assert V.FUSED_INDEX_Q_VERIFY and V.MLAQ_W == MLAQW
    except Exception as ex:  # noqa: BLE001
        emit(kind="setup_error", err=repr(ex)[:500])
        raise
    emit(kind="config", idx_q=IDXQ, qfp_par_max_rows=PAR, mlaq_w=MLAQW, qfp_min_rows=QFPMIN, mutation=os.environ.get("K31_MUTATION", ""))
    arms = ((1, 51), (5, 54), (9, 57), (12, 53), (16, 58)) if QFPMIN == 72 else ((1, 51), (5, 54), (8, 59), (12, 53))
    for B, seed in arms:
        try:
            run_arm(B, seed, V, fp4, qref)
        except Exception as ex:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            check(f"arm_ran_B{B}", False, err=repr(ex)[:400])
    if FP4:
        c4 = len(D4._engaged) == 1 and not K4._engaged and fq.STATS["query_qf"] > 0 and fq.STATS["query_qfn"] > 0
    else:
        c4 = len(K4._engaged) == 1 and not D4._engaged and fp4.STATS["query"] == 0 and sum(fq.STATS.values()) == 0
    check("C4_engagement", c4 and fp4.STATS["writer"] > 0 and fp4.STATS["logits"] == 0,
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
