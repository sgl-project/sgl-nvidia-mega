#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""In-path wiring test of K31MK_DSA_IDX_FP4 on one GPU: the REAL plugin verify objects (SparseVerifyIndexKey over
PreparedVerifyIndexKey, PreparedSparseVerifyMla.run with its Fp4IndexProjection and VerifySelection) on an MXFP4 index-K pool,
TP16 per-rank MLA heads (K31_TEST_HEADS=6), random weights, eager and CUDA graph.

, bundle-entry correctness of the wiring (fixed before the first run). Arms: Breq 1 x Q8 (8 rows: the
packed index-K writer + packed gate), Breq 2 x Q8 (16 rows) and Breq 12 x Q8 (96 rows, the MID B12; stock-chain writer on the index
stream, gate GEMM on the gate stream, K31MK_INDEX_Q_OFF_WRITER_WIDE=1 as X). Context 12,000 per request (> 2048: sparse steps); the
prefill index-K rows are written with idx_fp4.store_index_k (the host prefill writer's row math). Checks (fixed before the run):
  C1 the step's index-K rows in the pool EQUAL host-quantizer bytes of LayerNorm(hidden @ wk) computed with the writer's own GEMM form
     (the packed weight at 8 rows, F.linear otherwise) - exact.
  C2a the MXFP4 query bytes in the projection's buffers EQUAL the host quantizer applied to rmsnorm(projected[:, :1536]) @ wq_b (BF16) - exact.
  C2b head weights EQUAL float32(bfloat16(gate * 32**-0.5)) * 128**-0.5 with gate = hidden @ w_proj from the writer's GEMM form - exact.
  C2c each row's selected slots (top-k v2 over the MXFP4 logits) map to positions < the row's causal length and overlap the FP64 top-2048
     of the dequantized q / keys (sum_h w_h relu(q_h . k)) by >= 0.99 (the FP4 logits are exact to ~1e-8 of the abs-sum; residue = ties).
  C3 a CUDA-graph replay of the same chain on the same inputs gives bitwise the eager index-K rows and the eager selected rows (sorted).
  C4 engagement: idx_fp4.STATS writer / query / logits > 0.
  C5 the MLA output is finite.
  C6 (subprocess) importing dsa.verify with K31MK_DSA_IDX_FP4=1 and the default K31MK_FUSED_INDEX_Q_VERIFY raises ValueError naming it.
Mutants (each must FAIL at least one check; K31_MUTATION=<name> via _harness.load_plugin): wr_slot (stock writer stores at reversed slots),
  wr_packed (packed writer stores at reversed slots), sel_qflip (the logits read the query rows in reverse order), gate_scale (head gate
  doubled).
Exit 0 all pass, 1 any check fails, 2 setup error. JSON lines; DONE last.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from types import SimpleNamespace

os.environ.setdefault("K31_TEST_HEADS", "6")
os.environ["K31MK_DSA_IDX_FP4"] = "1"
os.environ["K31MK_FUSED_INDEX_Q_VERIFY"] = "0"
os.environ["K31MK_INDEX_Q_OFF_WRITER_WIDE"] = "1"

import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _harness as H  # noqa: E402

V_FILE = "kimi_k31/dsa/verify.py"
MUTATIONS = {
    "wr_slot": ("kimi_k31/verify_indexer.py", "idx_fp4.store_index_k(key, self.cache, slots)",
                "idx_fp4.store_index_k(key, self.cache, slots.flip(0))"),
    "wr_packed": (V_FILE, "idx_fp4.store_index_k(key, self.base.cache, slots)", "idx_fp4.store_index_k(key, self.base.cache, slots.flip(0))"),
    "sel_qflip": (V_FILE, "            scores = fp4_logits(\n                query, gates,",
                  "            scores = fp4_logits(\n                idx_fp4.Fp4Query(query.q4.flip(0), query.sf.flip(0)), gates,"),
    "gate_scale": (V_FILE, "return idx_fp4.quantize_query(self.query, self.gate, self.fp4)",
                   "return idx_fp4.quantize_query(self.query, self.gate * 2, self.fp4)"),
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


def run_arm(B, seed, V, fp4, qref):
    import deep_gemm
    import torch.nn.functional as F
    from flashinfer.norm import layernorm
    from sgl_kernel import rmsnorm
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
    gate_end = 2112 + 128 * HEADS
    npg = math.ceil(CTX / 64)
    pages = B * npg + 2
    perm = torch.randperm(pages - 1, generator=g, device="cuda")[: B * npg] + 1
    table = perm.view(B, npg).to(torch.int32).contiguous()
    pool = torch.full((pages, fp4.PAGE_B), 0xA5, dtype=torch.uint8, device="cuda")
    allslots = (table.to(torch.int64)[:, :, None] * 64 + torch.arange(64, device="cuda")[None, None, :]).reshape(B, -1)[:, :CTX]
    prefill = torch.randn((B * (CTX - Q), 128), generator=g, device="cuda").to(torch.bfloat16)
    fp4.store_index_k(prefill, pool, allslots[:, : CTX - Q].reshape(-1))
    step_slots = allslots[:, CTX - Q:].reshape(-1).contiguous()  # the verify rows' new tokens, request-major
    kv = kv_pool(pages, g)
    lengths = torch.full((B,), CTX, dtype=torch.int32, device="cuda")
    causal = torch.cat([(CTX - Q + 1 + torch.arange(Q, device="cuda")) for _ in range(B)]).to(torch.int32)
    contexts = torch.full((B, Q), CTX, dtype=torch.int32, device="cuda")
    step = SimpleNamespace(sparse=1, batch_requests=B, query_width=Q, causal_lengths=causal, score_contexts=contexts,
                           score_schedule=deep_gemm.get_paged_mqa_logits_metadata(contexts, 64, num_sms()),
                           topk_plan=plan_topk_v2(causal))
    scratch = torch.empty((rows, HEADS, 576), dtype=torch.uint8, device="cuda")
    attention, _ = fakes(w)
    base = PreparedMla(attention, {rows: gate_end}, scratch, max_context=1 << 17, core="trtllm", queries=(Q,))
    core = V.PreparedSparseVerifyMla(base, IndexWeights(w.wq_b, w.w_proj, w.w_qa, 1e-5), rows, 1 << 17)
    assert core.fused_index_q is None and type(core.index_projection).__name__ == "Fp4IndexProjection", "FP4 projection not engaged"
    core.prepare_cache(kv)
    writer = V.SparseVerifyIndexKey(PreparedVerifyIndexKey(IndexKeyWeights(wk, w.gamma, w.beta, w.k_eps)), core)
    writer.bind(pool, 64, rows=(rows,))
    core.step = step
    hidden_src = torch.randn((rows, 7168), generator=g, device="cuda").to(torch.bfloat16)
    proj_src = torch.randn((rows, gate_end), generator=g, device="cuda").to(torch.bfloat16)
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

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        chain()
    torch.cuda.current_stream().wait_stream(s)
    eager = snap()

    # C1 writer rows: the writer's own GEMM form, then LayerNorm, then the host quantizer
    if writer.packed_weight is not None:  # the writer's call forms: mm(out=) into its packed buffer / F.linear + prepare_gate's mm(out=)
        packed = torch.empty((rows, writer.packed_weight.shape[0]), dtype=torch.bfloat16, device="cuda")
        torch.mm(hidden_src, writer.packed_weight.T, out=packed)
        key_ref = layernorm(packed[:, :128], w.gamma, w.beta, w.k_eps)
        gate_ref = packed[:, 128:160]
    else:
        key_ref = layernorm(F.linear(hidden_src, wk), w.gamma, w.beta, w.k_eps)
        gate_ref = torch.empty((rows, 32), dtype=torch.bfloat16, device="cuda")
        torch.mm(hidden_src, w.w_proj.T, out=gate_ref)
    k4, ksf = qref(key_ref)
    v, sfb, sel, outv = eager
    check(f"C1_writer_rows_{tag}", torch.equal(v, k4.view(torch.uint8)) and torch.equal(sfb, ksf.contiguous().view(torch.uint8).view(-1, 4)),
          val_mismatch=int((v != k4.view(torch.uint8)).sum()), packed=writer.packed_weight is not None)
    # C2a / C2b query bytes and head weights
    qn = rmsnorm(proj_src[:, :1536], w.w_qa, 1e-5)
    q_bf16 = torch.empty((rows, 32, 128), dtype=torch.bfloat16, device="cuda")
    torch.mm(qn, w.wq_b.T, out=q_bf16.view(-1, 4096))  # the projection's call form
    r4, rsf = qref(q_bf16.reshape(-1, 128))
    fb = core.index_projection.fp4
    check(f"C2a_query_bytes_{tag}", torch.equal(fb.q4[:rows].reshape(-1, 64).view(torch.uint8), r4.view(torch.uint8)) and
          torch.equal(fb.sf[:rows].reshape(-1), rsf), q_mismatch=int((fb.q4[:rows].reshape(-1, 64).view(torch.uint8) != r4.view(torch.uint8)).sum()))
    w_ref = (gate_ref.float() * 32 ** -0.5).to(torch.bfloat16).float() * 128 ** -0.5
    check(f"C2b_head_weights_{tag}", torch.equal(fb.weights[:rows].view(rows, 32), w_ref),
          max_abs=float((fb.weights[:rows].view(rows, 32) - w_ref).abs().max().item()))
    # C2c selection vs the FP64 top-2048 of the dequantized q / keys (pool as written)
    qd = decode(r4.view(torch.uint8), rsf.contiguous().view(torch.uint8).view(-1, 4)).view(rows, 32, 128)
    wr = w_ref.cpu().to(torch.float64)
    worst, bad_pos = 1.0, 0
    for b in range(B):
        kvb, ksb = rows_at(pool, allslots[b], fp4.PAGE_B)
        kd = decode(kvb, ksb)  # [CTX, 128] positions 0..CTX-1
        slot_to_pos = {int(sl): p for p, sl in enumerate(allslots[b].tolist())}
        for j in range(Q):
            r = b * Q + j
            n = int(causal[r])
            sc = (wr[r][:, None] * (qd[r] @ kd[:n].T).clamp(min=0)).sum(0)
            want = set(torch.topk(sc, 2048).indices.tolist())
            got = [slot_to_pos.get(int(x), -1) for x in sel[r].tolist() if int(x) >= 0]
            bad_pos += sum(1 for p in got if p < 0 or p >= n)
            worst = min(worst, len(want & set(got)) / 2048)
    check(f"C2c_selection_vs_fp64_{tag}", worst >= 0.99 and bad_pos == 0, min_overlap=worst, outside_causal=bad_pos)
    # C3 CUDA graph replay on the same inputs
    gph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gph):
        chain()
    core.selection.selected.fill_(-1)  # poison what the replay must rewrite: selection and the step's index-K rows
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
        from sgl_nvidia_mega.kimi_k31.dsa import idx_fp4 as fp4
        from sgl_nvidia_mega.kimi_k31.dsa import verify as V
        from sglang.kernels.ops.quantization.fp4_indexer_quant import quantize_fp4_indexer_tensor as qref

        torch.zeros(1, device="cuda")
        assert V.IDX_FP4 and fp4.ENABLED
    except Exception as ex:  # noqa: BLE001
        emit(kind="setup_error", err=repr(ex)[:500])
        raise
    for B, seed in ((1, 31), (2, 32), (12, 33)):
        try:
            run_arm(B, seed, V, fp4, qref)
        except Exception as ex:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            check(f"arm_ran_B{B}", False, err=repr(ex)[:400])
    check("C4_engagement", fp4.STATS["writer"] > 0 and fp4.STATS["query"] > 0 and fp4.STATS["logits"] > 0, **fp4.STATS)
    if not os.environ.get("K31_MUTATION"):
        env = dict(os.environ)
        env.pop("K31MK_FUSED_INDEX_Q_VERIFY", None)
        r = subprocess.run([sys.executable, "-c", "import sgl_nvidia_mega.kimi_k31.dsa.verify"], env=env, capture_output=True, text=True,
                           timeout=600)
        check("C6_import_refuses_fused_index_q", r.returncode != 0 and "K31MK_FUSED_INDEX_Q_VERIFY" in r.stderr, rc=r.returncode,
              tail=r.stderr.strip().splitlines()[-1][:200] if r.stderr.strip() else "")
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
