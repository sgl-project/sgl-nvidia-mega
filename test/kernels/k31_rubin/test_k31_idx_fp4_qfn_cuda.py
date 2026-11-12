#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CUDA MXFP4 QFN index query (idx_fp4_qfn.cuh via idx_fp4_q.qfn_quantize_query impl='cuda'; K31MK_DSA_IDX_Q=fp4 on K31MK_DSA_QFN / QFP
rows). . Bundle-entry correctness (fixed before the first run). The reference is the Triton kernel's
gate (test_k31_idx_fp4_qfn.py N1-N3) applied to the CUDA kernel, plus agreement with the Triton kernel itself.
Cells: rows 2 / 33 / 40 / 72 / 96 / 128, seeds 1 / 2; inputs as test_k31_idx_fp4_qfn.py (acc = bf16(x @ wq_bg^T), raw q_a rows of scale
0.5 + 2^(s - 3), projected row stride 3040, gate = a column slice).
  C1 the served CUDA launch (no hook) and the hooked one (debug_query) give BITWISE equal e2m1 bytes, scale words and head weights.
  C2 the hooked CUDA outputs are BITWISE idx_fp4.quantize_query(its own BF16 query, gate).
  C3 its BF16 query vs FP64 (acc * rsqrt(sum(x^2) / 1536 + eps)): every element within 1 BF16 ulp (|d| <= 2^-7 |ref| + 1e-30) and >= 99 %
     of elements equal bf16(FP64 value).
  C4 vs the Triton kernel on the same inputs: head weights BITWISE equal; the two BF16 queries within 1 BF16 ulp of each other everywhere
     (the rstd sums differ only in FP32 order); e2m1 bytes equal on >= 99 % of bytes and scale words on >= 99 % of words.
  C5 CUDA-graph replay of the served CUDA launch == its eager outputs (bitwise) after the outputs are poisoned.
  C6 dispatch: each impl='cuda' call reached idx_fp4_qfn.qfn exactly once and impl='triton' never did.
Mutants (each must FAIL; K31_MUTATION=<name>; q4c_rstd anchor re-pointed after the kRstd template): q4c_rstd (rstd := 1), q4c_nib (nibble order of each pair swapped), q4c_gate (the next head's
  gate), q4c_sfpos (a group's scale byte written to its neighbour's slot).
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

_CU = "kimi_k31/dsa/idx_fp4_qfn.cuh"
MUTATIONS = {
    "q4c_rstd": (_CU, "    rstd = rsqrtf(__fadd_rn(div_full(tot, static_cast<float>(kLora)), p.eps));",
                 "    rstd = 1.0f + 0.0f * tot;"),
    "q4c_nib": (_CU, "for (int j = 0; j < 8; ++j) lo |= e2m1_code(div_full(q[j], scale)) << (4 * j);",
                "for (int j = 0; j < 8; ++j) lo |= e2m1_code(div_full(q[j], scale)) << (4 * (j ^ 1));"),
    "q4c_gate": (_CU, "const float g = __bfloat162float(p.gate[row * p.g_stride + t]);",
                 "const float g = __bfloat162float(p.gate[row * p.g_stride + ((t + 1) & 31)]);"),
    "q4c_sfpos": (_CU, "p.qsf[row * (kHeads * 4) + (t >> 1)] = static_cast<uint8_t>(ex);",
                  "p.qsf[row * (kHeads * 4) + ((t >> 1) ^ 1)] = static_cast<uint8_t>(ex);"),
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
        from sgl_nvidia_mega.kimi_k31.dsa import idx_fp4, idx_fp4_q, idx_fp4_qfn

        torch.zeros(1, device="cuda")
        idx_fp4_qfn.build()
    except Exception as ex:  # noqa: BLE001
        emit(kind="setup_error", err=repr(ex)[:500])
        raise
    calls = {"n": 0}
    real_qfn = idx_fp4_qfn.qfn

    def counting_qfn(*a, **k):
        calls["n"] += 1
        return real_qfn(*a, **k)

    idx_fp4_qfn.qfn = counting_qfn
    qq = idx_fp4_q.qfn_quantize_query
    for seed in (1, 2):
        for rows in (2, 33, 40, 72, 96, 128):
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
                gate = projected[:, 3008:3040]
                acc = torch.mm(projected[:, :1536], wq_bg.T).view(rows, 32, 128).contiguous()
                served, hooked, trit = (idx_fp4.QueryBuffers(rows, acc.device) for _ in range(3))
                H.poison(served.q4, served.sf, served.weights, hooked.q4, hooked.sf, hooked.weights, trit.q4, trit.sf, trit.weights)
                dq = torch.empty((rows, 32, 128), dtype=torch.bfloat16, device="cuda")
                dqt = torch.empty((rows, 32, 128), dtype=torch.bfloat16, device="cuda")
                n0 = calls["n"]
                fs, ws = qq(acc, projected, gate, EPS, served, impl="cuda")
                fh, wh = qq(acc, projected, gate, EPS, hooked, debug_query=dq, impl="cuda")
                n1 = calls["n"]
                ft, wt = qq(acc, projected, gate, EPS, trit, debug_query=dqt, impl="triton")
                torch.cuda.synchronize()
                check(f"C6_dispatch.{tag}", n1 - n0 == 2 and calls["n"] == n1, cuda_calls=n1 - n0, triton_reached_cuda=calls["n"] - n1)
                c1 = torch.equal(fs.q4, fh.q4) and torch.equal(fs.sf, fh.sf) and torch.equal(ws.view(torch.int32), wh.view(torch.int32))
                check(f"C1_served_eq_hooked.{tag}", c1, q4_diff=int((fs.q4 != fh.q4).sum()), sf_diff=int((fs.sf != fh.sf).sum()))
                ref_buf = idx_fp4.QueryBuffers(rows, acc.device)
                fr, wr = idx_fp4.quantize_query(dq, gate, ref_buf)
                torch.cuda.synchronize()
                c2 = torch.equal(fr.q4, fh.q4) and torch.equal(fr.sf, fh.sf) and torch.equal(wr.view(torch.int32), wh.view(torch.int32))
                check(f"C2_quantizer_bitwise.{tag}", c2, q4_diff=int((fr.q4 != fh.q4).sum()), sf_diff=int((fr.sf != fh.sf).sum()),
                      w_diff=int((wr != wh).sum()))
                x64 = projected[:, :1536].double()
                rstd = torch.rsqrt((x64 * x64).sum(-1) / 1536 + EPS)
                ref64 = acc.double() * rstd[:, None, None]
                d = (dq.double() - ref64).abs()
                within = bool((d <= ref64.abs() * 2.0 ** -7 + 1e-30).all())
                eq_frac = float((dq == ref64.to(torch.bfloat16)).float().mean())
                check(f"C3_query_vs_fp64.{tag}", within and eq_frac >= 0.99, all_within_1ulp=within, rn_equal_frac=eq_frac)
                wbit = torch.equal(wt.view(torch.int32), wh.view(torch.int32))
                dd = (dq.double() - dqt.double()).abs()
                q1ulp = bool((dd <= torch.maximum(dq.double().abs(), dqt.double().abs()) * 2.0 ** -7 + 1e-30).all())
                qeq = float((dq == dqt).float().mean())
                b_eq = float((ft.q4 == fh.q4).float().mean())
                s_eq = float((ft.sf == fh.sf).float().mean())
                check(f"C4_vs_triton.{tag}", wbit and q1ulp and b_eq >= 0.99 and s_eq >= 0.99, w_bitwise=wbit, query_within_1ulp=q1ulp,
                      query_equal_frac=qeq, q4_equal_frac=b_eq, sf_equal_frac=s_eq)
                # C5: graph replay of the served launch
                gbuf = idx_fp4.QueryBuffers(rows, acc.device)
                st = torch.cuda.Stream()
                st.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(st):
                    qq(acc, projected, gate, EPS, gbuf, impl="cuda")
                torch.cuda.current_stream().wait_stream(st)
                torch.cuda.synchronize()
                eager = (gbuf.q4[:rows].clone(), gbuf.sf[:rows].clone(), gbuf.weights[:rows].clone())
                gph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gph):
                    qq(acc, projected, gate, EPS, gbuf, impl="cuda")
                H.poison(gbuf.q4, gbuf.sf, gbuf.weights)
                gph.replay()
                torch.cuda.synchronize()
                c5 = torch.equal(gbuf.q4[:rows], eager[0]) and torch.equal(gbuf.sf[:rows], eager[1]) and \
                    torch.equal(gbuf.weights[:rows].view(torch.int32), eager[2].view(torch.int32))
                check(f"C5_graph_eq_eager.{tag}", c5)
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
