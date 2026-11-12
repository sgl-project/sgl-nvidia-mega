#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MXFP4 index query in the fused index query / query front (index_q.cuh kQ4: run_q4, run_qf_q4; K31MK_DSA_IDX_Q=fp4 through
idx_fp4_q.Fp4FusedIndexQuery) vs idx_fp4.quantize_query, the fp4 setting's quantizer of record. .

Bundle-entry correctness (fixed before the first run). TP16 columns (6 MLA heads).
  D exact-arithmetic cells (test_k31_index_q.domain_case: BF16 query and gate exact in any FP32 summation order; premise asserted first:
    stock's rmsnorm + wq_b GEMM reproduce the designed query and gate), e0 in (-20, 12), parts 0-7 each (amax over 64 binades):
    D1 run_q4 (8 rows): packed e2m1 bytes, ue8m0 scale words and head weights BITWISE equal to quantize_query(q, gate);
    D2 run_qf_q4 (8 rows; the same q_a columns, gate in the QF gate columns, MXFP4 pool): the same three BITWISE;
    D3 run_qf_q4 at 16 rows (the 8 rows twice; the kR = 16 instance): both halves BITWISE equal to quantize_query.
  R random cells (test_k31_index_q's weights / activations form), seeds 1 / 2; run_q4 at rows 1 / 5 / 8, run_qf_q4 at 1 / 5 / 8 / 12 / 16:
    R1 vs quantize_query(stock chain's BF16 query, reference gate): e2m1 codes <= 0.5 % differing, each within one signed step; ue8m0 bytes
       <= 1 % differing, each within 1 (only FP32 summation orders differ, as in test_k31_index_q's FP8 gate); head weights: run_qf
       BITWISE (its gate is read from the projection), run within 2^-7 relative (gate dot summation order);
    R2 (run_qf_q4) index-K bytes == those the FP8-query MXFP4-store launch (index_q.FusedIndexQuery.run_qf, kKFP4) stores on the same
       inputs, BITWISE; every pool byte outside the stored rows unchanged.
    R3 (run_qf_q4) the binding refuses FP8 index-K pages [pages, 64 * 132] (RuntimeError).
Mutants (each must FAIL; K31_MUTATION=<name>): q4_pair (group amax from this CTA's 16 dims only), q4_nib (pair nibble order swapped),
  q4_sfbyte (exponent byte at 3 - g), q4_wscale (head weights keep the q scale), q4_div (amax / 5).
Exit 0 all pass, 1 any check fails, 2 setup error. JSON lines; DONE last.
"""
from __future__ import annotations

import json
import math
import os
import sys

os.environ.setdefault("K31_TEST_HEADS", "6")

import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _harness as H  # noqa: E402

_CUH = "kimi_k31/dsa/index_q.cuh"
# q4_nib / q4_sfbyte anchors re-pointed (kQ4 v2 epilogue / exponent word; same faults, before any v2 data).
MUTATIONS = {
    "q4_pair": (_CUH, "const float gmax = fmaxf(s.ex_amax[rank & ~1u][b], s.ex_amax[rank | 1u][b]);", "const float gmax = s.ex_amax[rank][b];"),
    "q4_nib": (_CUH, "static_cast<uint8_t>(code | (hi << 4));", "static_cast<uint8_t>(hi | (code << 4));"),
    "q4_sfbyte": (_CUH, "s.ex_amax[2 * g4 + 1][b])) << (8 * g4);", "s.ex_amax[2 * g4 + 1][b])) << (8 * (3 - g4));"),
    "q4_wscale": (_CUH, "p.w[b * kHeads + head] = g2 * 0.08838834764831845f;  // 128^-0.5; no q scale",
                  "p.w[b * kHeads + head] = g2 * s.scale[b] * 0.08838834764831845f;  // 128^-0.5; no q scale"),
    "q4_div": (_CUH, "const float gsf = fmaxf(div_full(gmax, 6.0f), 1.0e-4f);", "const float gsf = fmaxf(div_full(gmax, 5.0f), 1.0e-4f);"),
}
HEADS = int(os.environ["K31_TEST_HEADS"])
K_COL = 2112 + 128 * HEADS
G_COL = K_COL + 128
QF_PITCH = G_COL + 32
PAGES = 6
OKS = []


def emit(**kw):
    print(json.dumps(kw), flush=True)


def check(name, ok, **kw):
    emit(kind="check", name=name, ok=bool(ok), **kw)
    OKS.append(bool(ok))


def signed(c):
    c = c.to(torch.int32)
    return torch.where((c & 8) != 0, -(c & 7), c & 7)


def codes_of(q4):
    v = q4.view(torch.uint8).reshape(-1, 64).to(torch.int32)
    return torch.stack([v & 0xF, v >> 4], dim=-1).reshape(-1, 128)


def sf_bytes(sf):
    return sf.contiguous().view(torch.uint8).reshape(-1, 4)


def tol_compare(got_q, got_sf, ref_q, ref_sf):
    gc, rc = codes_of(got_q), codes_of(ref_q)
    gs, rs = sf_bytes(got_sf).to(torch.int32), sf_bytes(ref_sf).to(torch.int32)
    kw = dict(code_diff_frac=float((gc != rc).float().mean()), code_max_step=int((signed(gc) - signed(rc)).abs().max()),
              sf_diff_frac=float((gs != rs).float().mean()), sf_max_diff=int((gs - rs).abs().max()))
    ok = kw["code_diff_frac"] <= 0.005 and kw["code_max_step"] <= 1 and kw["sf_diff_frac"] <= 0.01 and kw["sf_max_diff"] <= 1
    return ok, kw


def bitwise(got, ref):
    (gq, gs, gw), (rq, rs, rw) = got, ref
    kw = dict(q4_diff=int((gq.view(torch.uint8) != rq.view(torch.uint8)).sum()), sf_diff=int((gs != rs).sum()),
              w_diff=int((gw.reshape(-1).view(torch.int32) != rw.reshape(-1).view(torch.int32)).sum()))
    return all(v == 0 for v in kw.values()), kw


def qref(q, gate):
    """idx_fp4.quantize_query on a BF16 query [rows, 32, 128] and BF16 gate [rows, 32]: (q4, sf, weights) clones."""
    from sgl_nvidia_mega.kimi_k31.dsa import idx_fp4

    buf = idx_fp4.QueryBuffers(q.shape[0], q.device)
    fq, w = idx_fp4.quantize_query(q.contiguous(), gate, buf)
    torch.cuda.synchronize()
    return fq.q4.clone(), fq.sf.clone(), w.clone()


def run4(fused, *args, qf=False):
    from sgl_nvidia_mega.kimi_k31.dsa import idx_fp4_q

    H.poison(fused.q4, fused.sf, fused.gates)
    fq, w = (fused.run_qf if qf else fused.run)(*args)
    torch.cuda.synchronize()
    assert isinstance(fq, idx_fp4_q.Fp4Query)
    return fq.q4.clone(), fq.sf.clone(), w.clone()


def key_norm(seed):
    g = H.generator(seed)
    return (1 + 0.2 * torch.randn(128, generator=g, device="cuda")).float().contiguous(), \
        (0.5 * torch.randn(128, generator=g, device="cuda")).float().contiguous(), 1e-6


def qf_inputs(q_a, gate, g, rows):
    projected = torch.randn((rows, QF_PITCH), generator=g, device="cuda").to(torch.bfloat16)
    projected[:, :1536] = q_a
    projected[:, G_COL : G_COL + 32] = gate
    projected[:, K_COL : K_COL + 128] = (0.5 + 2.0 * torch.randn((rows, 128), generator=g, device="cuda")).to(torch.bfloat16)
    slots = torch.randperm(PAGES * 64, generator=g, device="cuda")[:rows].to(torch.int64).contiguous()
    if rows > 1:
        slots[rows // 2] = -1
    return projected, slots


def domain(T, Fp4FusedIndexQuery, stock_index_q):
    for e0 in (-20, 12):
        bad = {"D1": 0, "D2": 0, "D3": 0}
        detail = {}
        g = H.generator(4000 + e0)
        for part in range(8):
            w, projected, hidden, q, gate = T.domain_case(e0, part, g)
            squery, _, _, _ = stock_index_q(projected, hidden, w)
            assert torch.equal(squery, q), f"e0 {e0} part {part}: stock query != designed query (construction not exact)"
            assert torch.equal(torch.mm(hidden, w.gate.T), gate), f"e0 {e0} part {part}: gate construction not exact"
            ref = qref(q, gate)
            ok, kw = bitwise(run4(Fp4FusedIndexQuery(w, 8), hidden, projected), ref)
            bad["D1"] += not ok
            detail.setdefault("D1", kw) if not ok else None
            pqf, slots = qf_inputs(projected[:, :1536], gate, g, 8)
            pool = torch.full((PAGES, 64 * 68), 0xA5, dtype=torch.uint8, device="cuda")
            ok, kw = bitwise(run4(Fp4FusedIndexQuery(w, 8), pqf, K_COL, G_COL, slots, pool, key_norm(5), qf=True), ref)
            bad["D2"] += not ok
            detail.setdefault("D2", kw) if not ok else None
            p16 = torch.cat([pqf, pqf]).contiguous()
            s16 = torch.randperm(PAGES * 64, generator=g, device="cuda")[:16].to(torch.int64).contiguous()
            pool16 = torch.full((PAGES, 64 * 68), 0xA5, dtype=torch.uint8, device="cuda")
            got = run4(Fp4FusedIndexQuery(w, 16, qf_only=True), p16, K_COL, G_COL, s16, pool16, key_norm(5), qf=True)
            ref16 = tuple(torch.cat([t, t]) for t in ref)
            ok, kw = bitwise(got, ref16)
            bad["D3"] += not ok
            detail.setdefault("D3", kw) if not ok else None
        for k in ("D1", "D2", "D3"):
            check(f"{k}_exact_bitwise.e0{e0}", bad[k] == 0, parts_bad=bad[k], first_bad=detail.get(k))


def random_cells(T, Fp4FusedIndexQuery, FusedIndexQuery, stock_index_q):
    for seed in (1, 2):
        for rows in (1, 5, 8, 12, 16):
            tag = f"b{rows}.s{seed}"
            try:
                w = T.weights(100 * seed + rows)
                g = H.generator(7 * seed + rows)
                projected, hidden = T.activations(rows, g)
                squery, _, _, _ = stock_index_q(projected, hidden, w)
                gate_ref = torch.mm(hidden, w.gate.T)
                rq, rs, rw = qref(squery, gate_ref)
                if rows <= 8:
                    gq, gs, gw = run4(Fp4FusedIndexQuery(w, rows), hidden, projected)
                    ok, kw = tol_compare(gq, gs, rq, rs)
                    rel = float(((gw.reshape(rows, 32) - rw.reshape(rows, 32)).abs() / rw.reshape(rows, 32).abs().clamp_min(1e-30)).max())
                    check(f"R1_run_q4.{tag}", ok and rel <= 2.0 ** -7, w_max_rel=rel, **kw)
                pqf, slots = qf_inputs(projected[:, :1536], gate_ref, g, rows)
                norm = key_norm(9 * seed + rows)
                pool4 = torch.full((PAGES, 64 * 68), 0xA5, dtype=torch.uint8, device="cuda")
                clean = pool4.clone()
                qf_only = rows > 8
                gq, gs, gw = run4(Fp4FusedIndexQuery(w, rows, qf_only=qf_only), pqf, K_COL, G_COL, slots, pool4, norm, qf=True)
                ok, kw = tol_compare(gq, gs, rq, rs)
                wbit = torch.equal(gw.reshape(-1).view(torch.int32), rw.reshape(-1).view(torch.int32))
                check(f"R1_run_qf_q4.{tag}", ok and wbit, w_bitwise=wbit, **kw)
                pool8q = torch.full((PAGES, 64 * 68), 0xA5, dtype=torch.uint8, device="cuda")
                f8 = FusedIndexQuery(w, rows, qf_only=qf_only)
                f8.run_qf(pqf, K_COL, G_COL, slots, pool8q, norm)
                torch.cuda.synchronize()
                owned = torch.zeros_like(pool4, dtype=torch.bool)
                for s in slots.tolist():
                    if s >= 0:
                        owned[s // 64, (s % 64) * 64 : (s % 64) * 64 + 64] = True
                        owned[s // 64, 4096 + (s % 64) * 4 : 4096 + (s % 64) * 4 + 4] = True
                check(f"R2_index_k_bitwise.{tag}", torch.equal(pool4, pool8q) and torch.equal(pool4[~owned], clean[~owned]),
                      diff_vs_fp8q_launch=int((pool4 != pool8q).sum()), written_outside=int((pool4[~owned] != clean[~owned]).sum()))
                if rows == 5:
                    pool132 = torch.zeros((PAGES, 64 * 132), dtype=torch.uint8, device="cuda")
                    fused = Fp4FusedIndexQuery(w, rows)
                    try:
                        fused.module.run_qf_q4(pqf, w.query_norm, w.query, fused.q4, fused.sf, fused.gates.view(-1, 32), float(w.epsilon),
                                               K_COL, G_COL, slots, pool132, norm[0], norm[1], float(norm[2]))
                        torch.cuda.synchronize()
                        refused = False
                    except Exception:  # noqa: BLE001
                        refused = True
                    check(f"R3_fp8_pages_refused.{tag}", refused)
            except Exception as ex:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                check(f"cell_ran.{tag}", False, err=repr(ex)[:300])


def main():
    try:
        H.load_plugin(MUTATIONS)
        import test_k31_index_q as T
        from sgl_nvidia_mega.kimi_k31.dsa.idx_fp4_q import Fp4FusedIndexQuery
        from sgl_nvidia_mega.kimi_k31.dsa.index_q import FusedIndexQuery

        torch.zeros(1, device="cuda")
    except Exception as ex:  # noqa: BLE001
        emit(kind="setup_error", err=repr(ex)[:500])
        raise
    try:
        domain(T, Fp4FusedIndexQuery, T.stock_index_q)
    except Exception as ex:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        check("domain_ran", False, err=repr(ex)[:300])
    random_cells(T, Fp4FusedIndexQuery, FusedIndexQuery, T.stock_index_q)
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
