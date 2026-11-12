#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CUDA stock-chain MXFP4 index query (idx_fp4_q.quantize_query_served impl='cuda' -> idx_fp4_qfn.cuh<kRstd=false>; K31MK_DSA_IDX_Q=fp4 on
rows without a folded query). . Bundle-entry correctness, BITWISE lever (fixed before the first run).
Reference = idx_fp4.quantize_query (the Triton kernel, unchanged) on the same inputs.
Cells: rows 1 / 8 / 16 / 24 / 32 / 40 / 128, seeds 1 / 2. query BF16 [rows, 32, 128] contiguous: N(0, 1) x 2^(r % 9 - 4) per row r, plus in
every cell one all-zero (row, head), one (row, head) of exact e2m1 midpoints x 2^k (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5 and negatives: the
strict-threshold ties) and one with a 1e6-scale outlier; gate = a BF16 column slice of a wider tensor (row stride 3040).
  Q1 BITWISE: e2m1 bytes, scale words and head weights equal the reference's.
  Q2 CUDA-graph replay of the CUDA launch == its eager outputs (bitwise) after the outputs are poisoned.
  Q3 dispatch: each impl='cuda' call reached idx_fp4_qfn.quantize exactly once, impl='triton' never; idx_fp4.STATS['query'] counts both.
Mutants (each must FAIL; K31_MUTATION=<name>): qc_val (the stock-chain query value scaled by 1 + 2^-7 before quantizing), q4c_nib (pair nibble
  order), q4c_gate (the next head's gate).
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
    "qc_val": (_CU, "kRstd ? __float2bfloat16_rn(__fmul_rn(__bfloat162float(a16[j]), rstd)) : a16[j];",
               "kRstd ? __float2bfloat16_rn(__fmul_rn(__bfloat162float(a16[j]), rstd)) : __float2bfloat16_rn(__bfloat162float(a16[j]) * 1.0078125f);"),
    "q4c_nib": (_CU, "for (int j = 0; j < 8; ++j) lo |= e2m1_code(div_full(q[j], scale)) << (4 * j);",
                "for (int j = 0; j < 8; ++j) lo |= e2m1_code(div_full(q[j], scale)) << (4 * (j ^ 1));"),
    "q4c_gate": (_CU, "const float g = __bfloat162float(p.gate[row * p.g_stride + t]);",
                 "const float g = __bfloat162float(p.gate[row * p.g_stride + ((t + 1) & 31)]);"),
}
OKS = []
TIES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


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
    real = idx_fp4_qfn.quantize

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    idx_fp4_qfn.quantize = counting
    qs = idx_fp4_q.quantize_query_served
    for seed in (1, 2):
        for rows in (1, 8, 16, 24, 32, 40, 128):
            tag = f"t{rows}.s{seed}"
            try:
                g = H.generator(700 * seed + rows)
                sc = torch.exp2((torch.arange(rows, device="cuda") % 9 - 4).float())
                q = (torch.randn((rows, 32, 128), generator=g, device="cuda") * sc[:, None, None])
                q[0, 1] = 0.0
                k = int(torch.randint(-3, 4, (1,), generator=g, device="cuda").item())
                tv = torch.tensor(TIES, device="cuda") * (2.0 ** k)
                q[rows - 1, 2] = torch.cat([tv, -tv]).repeat(10)[:128]
                q[rows // 2, 3, 7] = 1.0e6
                query = q.to(torch.bfloat16).contiguous()
                wide = torch.randn((rows, 3040), generator=g, device="cuda").to(torch.bfloat16)
                gate = wide[:, 3008:3040]
                ref, cud, trit = (idx_fp4.QueryBuffers(rows, query.device) for _ in range(3))
                H.poison(ref.q4, ref.sf, ref.weights, cud.q4, cud.sf, cud.weights, trit.q4, trit.sf, trit.weights)
                n0, s0 = calls["n"], idx_fp4.STATS["query"]
                fr, wr = idx_fp4.quantize_query(query, gate, ref)
                fc, wc = qs(query, gate, cud, impl="cuda")
                n1 = calls["n"]
                ft, wt = qs(query, gate, trit, impl="triton")
                torch.cuda.synchronize()
                check(f"Q3_dispatch.{tag}", n1 - n0 == 1 and calls["n"] == n1 and idx_fp4.STATS["query"] - s0 == 3,
                      cuda_calls=n1 - n0, triton_reached_cuda=calls["n"] - n1, stats_query=idx_fp4.STATS["query"] - s0)
                q1 = torch.equal(fc.q4, fr.q4) and torch.equal(fc.sf, fr.sf) and torch.equal(wc.view(torch.int32), wr.view(torch.int32))
                check(f"Q1_bitwise_vs_triton.{tag}", q1, q4_diff=int((fc.q4 != fr.q4).sum()), sf_diff=int((fc.sf != fr.sf).sum()),
                      w_diff=int((wc.view(torch.int32) != wr.view(torch.int32)).sum()),
                      triton_path_equal=bool(torch.equal(ft.q4, fr.q4) and torch.equal(ft.sf, fr.sf)))
                gbuf = idx_fp4.QueryBuffers(rows, query.device)
                st = torch.cuda.Stream()
                st.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(st):
                    qs(query, gate, gbuf, impl="cuda")
                torch.cuda.current_stream().wait_stream(st)
                torch.cuda.synchronize()
                eager = (gbuf.q4[:rows].clone(), gbuf.sf[:rows].clone(), gbuf.weights[:rows].clone())
                gph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gph):
                    qs(query, gate, gbuf, impl="cuda")
                H.poison(gbuf.q4, gbuf.sf, gbuf.weights)
                gph.replay()
                torch.cuda.synchronize()
                q2 = torch.equal(gbuf.q4[:rows], eager[0]) and torch.equal(gbuf.sf[:rows], eager[1]) and \
                    torch.equal(gbuf.weights[:rows].view(torch.int32), eager[2].view(torch.int32))
                check(f"Q2_graph_eq_eager.{tag}", q2)
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
