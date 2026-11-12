#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""In-path wiring test of K31MK_DSA_IDX_FP4_FP8Q (FP8 index query x MXFP4 index-K) on one GPU: the REAL plugin verify objects
(SparseVerifyIndexKey over PreparedVerifyIndexKey, PreparedSparseVerifyMla.run with the served FP8 index query - the fused index query at
8 rows, the stock chain above - and VerifySelection through dg_logits_fp4k) on an MXFP4 index-K pool, TP16 per-rank MLA heads
(K31_TEST_HEADS=6), random weights, eager and CUDA graph. X-like env: K31MK_DSA_DG_VENDOR=1, K31MK_INDEX_Q_OFF_WRITER_WIDE=1.

, bundle-entry correctness of the wiring (fixed before the first run). Arms: Breq 1 x Q8 (8 rows: packed
writer, fused index query), Breq 2 x Q8 (16 rows) and Breq 12 x Q8 (96 rows; stock-chain FP8 query). Context 12,000 per request; the
prefill index-K rows are written with idx_fp4.store_index_k. Checks (fixed before the run):
  C1 the step's index-K rows in the pool EQUAL host-quantizer bytes of LayerNorm(hidden @ wk) in the writer's own GEMM form - exact.
  C2 the logits consume the served FP8 query: VerifySelection.run receives an FP8 [rows, 32, 128] query (recorded by a spy on the eager
     pass); the fused index query is the producer at 8 rows, IndexProjection above; no MXFP4 query is made (idx_fp4.STATS query == 0).
  C2c each row's selected slots map to positions < the row's causal length and overlap the FP64 top-2048 of sum_h w_h relu(q_h . k) by
     >= 0.99, with q / w = the recorded FP8 query bytes and gates and k = the pool's MXFP4 rows decoded by the MX spec.
  C3 a CUDA-graph replay of the same chain on the same inputs gives bitwise the eager index-K rows and selected rows (sorted).
  C4 engagement: dg_logits_fp4k logged its ENGAGED line once; idx_fp4.STATS writer > 0, logits == 0 (the (k + q) logits never run).
  C5 the MLA output is finite.
  C6 (subprocess) dsa.verify refuses K31MK_DSA_IDX_FP4_FP8Q=1 with K31MK_DSA_IDX_FP4_VENDOR=1, and K31MK_DSA_IDX_FP4_FP8Q=1 without
     K31MK_DSA_IDX_FP4=1.
Mutants (each must FAIL at least one check; K31_MUTATION=<name>): wr_packed (packed writer stores at reversed slots), q_flip (the logits
  read the query rows in reverse order), gate_hflip (the logits read the head gates in reverse head order), sfkv (index-K scale map
  base + 16 B).
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
os.environ["K31MK_DSA_IDX_FP4_FP8Q"] = "1"
os.environ.pop("K31MK_FUSED_INDEX_Q_VERIFY", None)  # default 1: the fused index query at <= 8 rows
os.environ["K31MK_INDEX_Q_OFF_WRITER_WIDE"] = "1"
os.environ["K31MK_DSA_DG_VENDOR"] = "1"

import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _harness as H  # noqa: E402

V_FILE = "kimi_k31/dsa/verify.py"
K_FILE = "kimi_k31/dsa/dg_logits_fp4k.py"
MUTATIONS = {
    "wr_packed": (V_FILE, "idx_fp4.store_index_k(key, self.base.cache, slots)", "idx_fp4.store_index_k(key, self.base.cache, slots.flip(0))"),
    "q_flip": (V_FILE, "            scores = fp4_logits(\n                query, gates,", "            scores = fp4_logits(\n                query.flip(0), gates,"),
    "gate_hflip": (K_FILE, "gates.reshape(rows, HEADS), context_lens", "gates.reshape(rows, HEADS).flip(1).contiguous(), context_lens"),
    "sfkv": (K_FILE, "tm_sf_kv = _encode(i32, base + PAGE * VAL_B, ", "tm_sf_kv = _encode(i32, base + PAGE * VAL_B + 16, "),
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
    base = PreparedMla(attention, {rows: gate_end}, scratch, max_context=1 << 17, core="trtllm", queries=(Q,))
    core = V.PreparedSparseVerifyMla(base, IndexWeights(w.wq_b, w.w_proj, w.w_qa, 1e-5), rows, 1 << 17)
    producer = "fused_index_q" if core.fused_index_q is not None else type(core.index_projection).__name__
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

    seen = {}
    sel = core.selection
    orig_run = sel.run

    def spy(query, gates, cache, tbl, stp):
        seen["q"], seen["g"] = query.clone(), gates.clone()
        return orig_run(query, gates, cache, tbl, stp)

    sel.run = spy  # eager pass only
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        chain()
    torch.cuda.current_stream().wait_stream(s)
    eager = snap()
    del sel.run  # back to the class method before capture

    # C1 writer rows
    if writer.packed_weight is not None:
        packed = torch.empty((rows, writer.packed_weight.shape[0]), dtype=torch.bfloat16, device="cuda")
        torch.mm(hidden_src, writer.packed_weight.T, out=packed)
        key_ref = layernorm(packed[:, :128], w.gamma, w.beta, w.k_eps)
    else:
        key_ref = layernorm(F.linear(hidden_src, wk), w.gamma, w.beta, w.k_eps)
    k4, ksf = qref(key_ref)
    v, sfb, selected, outv = eager
    check(f"C1_writer_rows_{tag}", torch.equal(v, k4.view(torch.uint8)) and torch.equal(sfb, ksf.contiguous().view(torch.uint8).view(-1, 4)),
          val_mismatch=int((v != k4.view(torch.uint8)).sum()), packed=writer.packed_weight is not None)
    # C2 the served FP8 query reached the logits
    qx, gx = seen.get("q"), seen.get("g")
    want_producer = "fused_index_q" if rows <= 8 else "IndexProjection"
    check(f"C2_fp8_query_{tag}", qx is not None and qx.dtype == torch.float8_e4m3fn and qx.shape[0] == rows and qx.numel() == rows * 4096
          and producer == want_producer, dtype=str(getattr(qx, "dtype", None)), shape=list(getattr(qx, "shape", [])), producer=producer)
    # C2c selection vs FP64 top-2048 from the recorded query / gates and the decoded pool
    qd = qx.view(rows, 32, 128).to(torch.float64).cpu()
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
            want = set(torch.topk(sc, 2048).indices.tolist())
            got = [slot_to_pos.get(int(x), -1) for x in selected[r].tolist() if int(x) >= 0]
            bad_pos += sum(1 for p in got if p < 0 or p >= n)
            worst = min(worst, len(want & set(got)) / 2048)
    check(f"C2c_selection_vs_fp64_{tag}", worst >= 0.99 and bad_pos == 0, min_overlap=worst, outside_causal=bad_pos)
    # C3 CUDA graph replay
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
        from sgl_nvidia_mega.kimi_k31.dsa import dg_logits_fp4k as K4
        from sgl_nvidia_mega.kimi_k31.dsa import idx_fp4 as fp4
        from sgl_nvidia_mega.kimi_k31.dsa import verify as V
        from sglang.kernels.ops.quantization.fp4_indexer_quant import quantize_fp4_indexer_tensor as qref

        torch.zeros(1, device="cuda")
        assert V.IDX_FP4 and V.IDX_FP4_FP8Q and V.FUSED_INDEX_Q_VERIFY
    except Exception as ex:  # noqa: BLE001
        emit(kind="setup_error", err=repr(ex)[:500])
        raise
    for B, seed in ((1, 41), (2, 42), (12, 43)):
        try:
            run_arm(B, seed, V, fp4, qref)
        except Exception as ex:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            check(f"arm_ran_B{B}", False, err=repr(ex)[:400])
    check("C4_engagement", len(K4._engaged) == 1 and fp4.STATS["writer"] > 0 and fp4.STATS["query"] == 0 and fp4.STATS["logits"] == 0,
          fp4k_engaged=len(K4._engaged), **fp4.STATS)
    if not os.environ.get("K31_MUTATION"):
        for label, extra, drop, needle in (("vendor", {"K31MK_DSA_IDX_FP4_VENDOR": "1"}, (), "exclusive"),
                                           ("no_idx_fp4", {}, ("K31MK_DSA_IDX_FP4",), "needs K31MK_DSA_IDX_FP4=1")):
            env = dict(os.environ)
            env.update(extra)
            for k in drop:
                env.pop(k, None)
            r = subprocess.run([sys.executable, "-c", "import sgl_nvidia_mega.kimi_k31.dsa.verify"], env=env, capture_output=True, text=True,
                               timeout=600)
            check(f"C6_import_refuses_{label}", r.returncode != 0 and needle in r.stderr, rc=r.returncode,
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
