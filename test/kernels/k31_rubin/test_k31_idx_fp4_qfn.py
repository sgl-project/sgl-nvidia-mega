#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MXFP4 QFN index query (idx_fp4_q.qfn_quantize_query; K31MK_DSA_IDX_Q=fp4 on K31MK_DSA_QFN / QFP rows). .

Bundle-entry correctness (fixed before the first run). Cells: rows 2 / 33 / 40 / 96 / 128, seeds 1 / 2. acc = bf16(x @ wq_bg^T)
as QFN computes it (random BF16 wq_b and w_qa, folded as index_qfw.folded_query does); raw q_a x = BF16 rows of scale 0.5 + 2^(s - 3)
per row (rstd far from 1), inside a wider projected tensor (row stride free); gate BF16 [rows, 32] as a column slice (row stride free).
  N1 the served launch (no test hook) and the hooked launch (debug_query) give BITWISE equal e2m1 bytes, scale words and head weights.
  N2 the hooked launch's outputs are BITWISE quantize_query(its own BF16 query, gate) (idx_fp4's quantizer = the fp4 setting's of record).
  N3 its BF16 query vs FP64 (acc * rsqrt(sum(x^2) / 1536 + eps)): every element within 1 BF16 ulp of the FP64 value (|d| <= 2^-7 |ref|,
     plus 1e-30 absolute), and >= 99 % of elements equal bf16(FP64 value) (round-to-nearest of the exact product).
  N4 the same query, against the FP8 QFN kernel (index_qfn.qfn_quant_head_gate) on the same inputs: decoded FP8 codes x scale within the
     e4m3 rounding of the query (|q8 * scale - q| <= 2^-4 |q| + scale * 2^-9), i.e. both kernels quantize the same rstd-scaled query.
Mutants (each must FAIL; K31_MUTATION=<name>): qfn4_rstd (rstd := 1), qfn4_gate (the next head's gate column), qfn4_nib (pair nibble order
  swapped), qfn4_eps (epsilon x 1e4).
Exit 0 all pass, 1 any check fails, 2 setup error. JSON lines; DONE last.
"""
from __future__ import annotations

import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _harness as H  # noqa: E402

_PY = "kimi_k31/dsa/idx_fp4_q.py"
MUTATIONS = {
    "qfn4_rstd": (_PY, "    rstd = tl.rsqrt(tl.sum(x * x, 0) / LORA + EPS)\n    grp =", "    rstd = 1.0 + 0.0 * tl.sum(x * x, 0)\n    grp ="),
    "qfn4_gate": (_PY, "gate = tl.load(G + row * GATE_STRIDE + heads).to(tl.float32)", "gate = tl.load(G + row * GATE_STRIDE + (heads + 1) % HEADS_).to(tl.float32)"),
    "qfn4_nib": (_PY, "packed = (_fp4_e2m1_code(v0) & 0x0F) | ((_fp4_e2m1_code(v1) & 0x0F) << 4)",
                 "packed = (_fp4_e2m1_code(v1) & 0x0F) | ((_fp4_e2m1_code(v0) & 0x0F) << 4)"),
    "qfn4_eps": (_PY, "float(eps), 1536, HEADS, HEADS ** -0.5", "float(eps) * 1e4, 1536, HEADS, HEADS ** -0.5"),
}
OKS = []
EPS = 1e-5


def emit(**kw):
    print(json.dumps(kw), flush=True)


def check(name, ok, **kw):
    emit(kind="check", name=name, ok=bool(ok), **kw)
    OKS.append(bool(ok))


def main():
    try:
        H.load_plugin(MUTATIONS)
        from sgl_nvidia_mega.kimi_k31.dsa import idx_fp4
        from sgl_nvidia_mega.kimi_k31.dsa.idx_fp4_q import qfn_quantize_query
        from sgl_nvidia_mega.kimi_k31.dsa.index_qfn import qfn_quant_head_gate

        torch.zeros(1, device="cuda")
    except Exception as ex:  # noqa: BLE001
        emit(kind="setup_error", err=repr(ex)[:500])
        raise
    for seed in (1, 2):
        for rows in (2, 33, 40, 96, 128):
            tag = f"b{rows}.s{seed}"
            try:
                g = H.generator(300 * seed + rows)
                wq_b = (torch.randn((4096, 1536), generator=g, device="cuda") / 1536 ** 0.5).to(torch.bfloat16)
                w_qa = (1 + 0.1 * torch.randn(1536, generator=g, device="cuda")).to(torch.bfloat16)
                wq_bg = (wq_b.float() * w_qa.float()[None, :]).to(torch.bfloat16)
                rs = 0.5 + torch.exp2(torch.arange(rows, device="cuda", dtype=torch.float32) % 8 - 3)
                wide = torch.randn((rows, 3040), generator=g, device="cuda")
                wide[:, :1536] *= rs[:, None]
                projected = wide.to(torch.bfloat16)
                gate = projected[:, 3008:3040]  # row stride 3040, as QFP's projected[:, g_col : g_col + 32]
                acc = torch.mm(projected[:, :1536], wq_bg.T).view(rows, 32, 128).contiguous()
                served = idx_fp4.QueryBuffers(rows, acc.device)
                hooked = idx_fp4.QueryBuffers(rows, acc.device)
                H.poison(served.q4, served.sf, served.weights, hooked.q4, hooked.sf, hooked.weights)
                dq = torch.empty((rows, 32, 128), dtype=torch.bfloat16, device="cuda")
                fs, ws = qfn_quantize_query(acc, projected, gate, EPS, served)
                fh, wh = qfn_quantize_query(acc, projected, gate, EPS, hooked, debug_query=dq)
                torch.cuda.synchronize()
                n1 = (torch.equal(fs.q4, fh.q4) and torch.equal(fs.sf, fh.sf) and torch.equal(ws.view(torch.int32), wh.view(torch.int32)))
                check(f"N1_served_eq_hooked.{tag}", n1, q4_diff=int((fs.q4 != fh.q4).sum()), sf_diff=int((fs.sf != fh.sf).sum()))
                ref_buf = idx_fp4.QueryBuffers(rows, acc.device)
                fr, wr = idx_fp4.quantize_query(dq, gate, ref_buf)
                torch.cuda.synchronize()
                n2 = torch.equal(fr.q4, fh.q4) and torch.equal(fr.sf, fh.sf) and torch.equal(wr.view(torch.int32), wh.view(torch.int32))
                check(f"N2_quantizer_bitwise.{tag}", n2, q4_diff=int((fr.q4 != fh.q4).sum()), sf_diff=int((fr.sf != fh.sf).sum()),
                      w_diff=int((wr != wh).sum()))
                x64 = projected[:, :1536].double()
                rstd = torch.rsqrt((x64 * x64).sum(-1) / 1536 + EPS)
                ref64 = acc.double() * rstd[:, None, None]
                d = (dq.double() - ref64).abs()
                within = bool((d <= ref64.abs() * 2.0 ** -7 + 1e-30).all())
                eq_frac = float((dq == ref64.to(torch.bfloat16)).float().mean())
                check(f"N3_query_vs_fp64.{tag}", within and eq_frac >= 0.99, all_within_1ulp=within, rn_equal_frac=eq_frac)
                q8, w8 = qfn_quant_head_gate(acc, projected, gate, EPS, 128 ** -0.5, "ue8m0")
                torch.cuda.synchronize()
                # FP8 path: weights = bf16(gate * 32^-0.5) * scale * 128^-0.5 and the FP4 path's weights lack the scale -> scale = w8 / w4
                w4d = wh.reshape(rows, 32).double()
                scale = torch.where(w4d == 0, torch.ones_like(w4d), w8.reshape(rows, 32).double() / torch.where(w4d == 0, torch.ones_like(w4d), w4d))
                dec = q8.view(rows, 32, 128).to(torch.float32).double() * scale[:, :, None]
                qd = dq.double()
                ok4 = bool(((dec - qd).abs() <= qd.abs() * 2.0 ** -4 + scale[:, :, None] * 2.0 ** -9).all())
                check(f"N4_same_query_as_fp8_qfn.{tag}", ok4, max_abs=float((dec - qd).abs().max()))
            except Exception as ex:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                check(f"cell_ran.{tag}", False, err=repr(ex)[:300])
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
