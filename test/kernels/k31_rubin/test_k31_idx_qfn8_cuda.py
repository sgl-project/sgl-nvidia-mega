#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CUDA FP8 QFN index query (idx_qfn8.cuh via index_qfn.qfn_quant_head_gate under K31MK_DSA_QFN_CUDA=1). .
Bundle-entry correctness, BITWISE class (fixed before the first run). Reference = the served Triton kernel
index_qfn._qfn_quant_head_gate_kernel (the code it replaces), plus an FP64 sanity bound.
Inputs: acc = bf16(x @ wq_bg^T) [rows, 32, 128], raw q_a rows x with per-row scales 2^-6..2^6, head gate = a column slice of projected.
Layouts: p3040 (projected [rows, 3040]: the served QF pitch at 6 heads, 16-byte aligned rows = Triton's vec-8 load), p3048 (row stride
% 16 = 8: Triton's 1-element layout), poff (row stride 3056 = 16 x 191 but the pointer 8 bytes off 16-byte alignment: 1-element layout). Rows 1 / 2 / 33 / 40 / 48 / 56 / 64 / 71 / 72 / 96 / 128,
seeds 1 / 2 (eps 1e-5 / 1e-6), ue8m0 scales (served) and scale_fmt None; bulk cells 4096 rows (p3040, p3048); targeted cell 'pow2' (x = ones,
eps 0 -> rstd 1, head maxima 1.75 * 2^k and 448 * 2^k * (1 + {-2^-7, 0, 2^-7}): scales at / next to powers of two).
  Q1 every cell: the CUDA kernel's q8 bytes and head-weight words BITWISE equal the Triton kernel's (0 differing bytes / words).
  Q2 dispatch: with index_qfn.QFN_CUDA True each qfn_quant_head_gate call reaches idx_qfn8 once (STATS calls +1) and returns Q1's CUDA
     bytes; with QFN_CUDA False it never does.
  Q3 out= (the OWNQ owner-row form): writing rows lo = 8 .. 8 + rows - 1 of poisoned fixed buffers gives Q1's bytes there and leaves every
     other row's bytes poisoned (0xFF).
  Q4 CUDA-graph replay of the dispatch == its eager outputs (bitwise) after the outputs are poisoned.
  Q5 FP64 sanity (ue8m0 cells): ref = the FP64 query acc * rsqrt(sum(x^2) / 1536 + eps), s64 = 2^ceil(log2(amax64 / 448)) per head;
     per head the implied scale s = the one of s64 / 2, s64, 2 s64 that minimizes max |q8 s - ref|; (a) every head whose amax64 / 448 is
     not within 2^-6 (relative) of a power of two has s == s64; (b) |q8 s - ref| <= (2^-4 + 2^-8) |ref| + s 2^-9 on >= 99.9 % of elements.
  Q6 capture decline: with the module marked unbuilt, a captured dispatch declines (STATS declined +1) and the replay gives the Triton bytes.
  Q7 sensitivity: at least one K31_QFN8_VARIANTS variant whose rstd differs from the served variant's on >= 1 % of the bulk rows shows >= 1
     q8 byte differing from Triton on the bulk cells (Q1 can see a one-ulp rstd summation-order change).
  REPORT (no gate): every variant's differing q8 bytes / weight words vs Triton and rows with rstd != served; the Triton PTX's instruction
     counts per load layout (K31_QFN8_PTX_DIR receives the PTX files).
Mutants (each must FAIL Q1; K31_MUTATION=<name>): m8_rstd (rstd := 1), m8_gate (the next head's gate), m8_bf16 (q not rounded to BF16),
  m8_eps (eps dropped).
Amendment 1 (AFTER l2mk41's data; checks Q1-Q7 and their bars UNCHANGED): l2mk41 failed Q1 (whole heads with a
  2x scale: my lg2.approx vs Triton's libdevice log2f; plus 34 / 33.5M bulk bytes: my div.full + add vs Triton's div.approx(1, 1536) + fma). The
  kernel now uses the instructions of the Triton PTX that l2mk41 dumped (header of idx_qfn8.cuh); variants gain a 4th field (rstd tail) and the
  m8_rstd / m8_eps anchors follow the new rstd lines (same mutations). The variant table is reported over the bulk cells as before.
Exit 0 all pass, 1 any check fails, 2 setup error. JSON lines; DONE last.
"""
from __future__ import annotations

import json
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _harness as H  # noqa: E402

_CU = "kimi_k31/dsa/idx_qfn8.cuh"
MUTATIONS = {
    "m8_rstd": (_CU, '  asm("rsqrt.approx.ftz.f32 %0, %1;\\n" : "=f"(rstd) : "f"(rin));', "  rstd = 1.0f + 0.0f * rin;"),
    "m8_gate": (_CU, "const float g = __bfloat162float(p.gate[row * p.g_stride + h]);",
                "const float g = __bfloat162float(p.gate[row * p.g_stride + ((h + 1) & 31)]);"),
    "m8_bf16": (_CU, "    q[j] = __bfloat162float(qb);", "    q[j] = __fmul_rn(__bfloat162float(a16[j]), rstd);"),
    "m8_eps": (_CU, "    rin = __fmaf_rn(c, tot, p.eps);", "    rin = __fmaf_rn(c, tot, 0.0f * p.eps);"),
}
OKS = []
VARIANTS = [(i, c, t, s) for i in range(4) for c in range(2) for t in range(2) for s in range(3)]


def emit(**kw):
    print(json.dumps(kw), flush=True)


def check(name, ok, **kw):
    emit(kind="check", name=name, ok=bool(ok), **kw)
    OKS.append(bool(ok))


def make_inputs(rows, seed, layout, bulk=False):
    g = H.generator(1000 * seed + rows + (7 if layout != "p3040" else 0))
    wq_b = (torch.randn((4096, 1536), generator=g, device="cuda") / 1536 ** 0.5).to(torch.bfloat16)
    w_qa = (1 + 0.1 * torch.randn(1536, generator=g, device="cuda")).to(torch.bfloat16)
    wq_bg = (wq_b.float() * w_qa.float()[None, :]).to(torch.bfloat16)
    pitch = {"p3040": 3040, "p3048": 3048, "poff": 3056}[layout]
    wide = torch.randn((rows, pitch), generator=g, device="cuda")
    rs = torch.exp2(torch.randint(-6, 7, (rows,), generator=g, device="cuda").float()) * (0.5 + torch.rand((rows,), generator=g, device="cuda"))
    wide[:, :1536] *= rs[:, None]
    if bulk:  # heavy-tailed rows too
        wide[: rows // 4, :1536] *= torch.exp(torch.randn((rows // 4, 1536), generator=g, device="cuda"))
    full = wide.to(torch.bfloat16)
    projected = full[:, 4:] if layout == "poff" else full
    gate = projected[:, 3008:3040] if layout == "p3040" else projected[:, 2900:2932]
    acc = torch.mm(projected[:, :1536], wq_bg.T).view(rows, 32, 128).contiguous()
    return acc, projected, gate


def make_pow2(rows=64):
    g = H.generator(77)
    projected = torch.ones((rows, 3040), dtype=torch.bfloat16, device="cuda")
    base = torch.rand((rows, 32, 128), generator=g, device="cuda") * 2 - 1  # |.| <= 1
    tops = []
    for r in range(rows):
        for h in range(32):
            k = (r + h) % 17 - 8
            kind = (r * 32 + h) % 4
            tops.append(1.75 * 2.0 ** k if kind == 0 else 448 * 2.0 ** (k - 6) * (1 + (kind - 2) * 2 ** -7))
    top = torch.tensor(tops, device="cuda").view(rows, 32, 1)
    acc = (base * top * 0.99).to(torch.bfloat16)
    acc[:, :, 5] = top[:, :, 0].to(torch.bfloat16) * torch.where(torch.arange(rows, device="cuda")[:, None] % 2 == 0, 1.0, -1.0)
    gate = projected[:, 3008:3040].clone()
    gate.copy_((torch.randn((rows, 32), generator=g, device="cuda")).to(torch.bfloat16))
    return acc.contiguous(), projected, gate


def triton_ref(qfn_mod, acc, projected, gate, eps, sfmt):
    saved = qfn_mod.QFN_CUDA
    qfn_mod.QFN_CUDA = False
    try:
        return qfn_mod.qfn_quant_head_gate(acc, projected, gate, eps, 128 ** -0.5, sfmt)
    finally:
        qfn_mod.QFN_CUDA = saved


def cuda_out(q8m, acc, projected, gate, eps, sfmt, variant=(0, 0, 0, 0), debug=False):
    rows = acc.shape[0]
    q8 = torch.empty_like(acc, dtype=torch.float8_e4m3fn)
    w = torch.empty((rows, 32, 1), device="cuda", dtype=torch.float32)
    H.poison(q8, w)
    dq = torch.empty_like(acc) if debug else None
    dr = torch.empty((rows,), device="cuda", dtype=torch.float32) if debug else None
    q8m.qfn8_test(acc, projected, gate, q8, w, eps, sfmt is not None, debug_query=dq, debug_rstd=dr, variant=variant)
    return q8, w, dr


def ndiff(a, b):
    return int((a.view(torch.uint8) != b.view(torch.uint8)).sum())


def wdiff(a, b):
    return int((a.view(torch.int32) != b.view(torch.int32)).sum())


def fp64_ok(acc, projected, eps, q8):
    x64 = projected[:, :1536].double()
    rstd = torch.rsqrt((x64 * x64).sum(-1) / 1536 + eps)
    ref = acc.double() * rstd[:, None, None]
    amax = ref.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s64 = torch.exp2(torch.ceil(torch.log2(amax / 448)))
    cand = torch.stack([s64 / 2, s64, s64 * 2])  # [3, rows, 32, 1]
    err = ((q8.double()[None] * cand - ref[None]).abs()).amax(-1, keepdim=True)  # [3, rows, 32, 1]
    s = torch.gather(cand, 0, err.argmin(0, keepdim=True))[0]
    l2 = torch.log2(amax / 448)
    borderline = (l2 - torch.round(l2)).abs() <= 2.0 ** -6 / 0.6931471805599453
    scale_ok = bool(((s == s64) | borderline).all())
    bound = ref.abs() * (2.0 ** -4 + 2.0 ** -8) + s * 2.0 ** -9
    frac = float(((q8.double() * s - ref).abs() <= bound).double().mean())
    return scale_ok, frac


def ptx_report(qfn_mod, acc, projected, gate, eps, label):
    """REPORT: launch the Triton kernel exactly as qfn_quant_head_gate does and count instructions in its PTX."""
    rows, heads = acc.shape[0], acc.shape[1]
    q8 = torch.empty_like(acc, dtype=torch.float8_e4m3fn)
    w = torch.empty((rows, heads, 1), device="cuda", dtype=torch.float32)
    k = qfn_mod._qfn_quant_head_gate_kernel[(rows,)](acc, projected, gate, q8, w, projected.stride(0), gate.stride(0), float(eps), 1536, heads,
                                                     heads ** -0.5, 128 ** -0.5, True, True, 0, 2048, num_warps=8, num_stages=0)
    ptx = k.asm["ptx"]
    d = os.environ.get("K31_QFN8_PTX_DIR", "")
    if d:
        with open(os.path.join(d, f"qfn_triton_{label}.ptx"), "w") as f:
            f.write(ptx)
    pats = {"fma_f32": r"\bfma\.rn\.f32", "add_f32": r"\badd\.(rn\.)?f32", "mul_f32": r"\bmul\.(rn\.)?f32", "shfl_bfly": r"shfl\.sync\.bfly",
            "lg2": r"lg2\.approx", "ex2": r"ex2\.approx", "div_full": r"div\.full\.f32", "div_rn": r"div\.rn\.f32", "cvt_rpi": r"cvt\.rpi",
            "rsqrt": r"rsqrt\.approx", "ld_v4": r"ld\.global(\.[a-z0-9]+)*\.v4", "ld_b16": r"ld\.global(\.[a-z0-9]+)*\.b16",
            "e4m3x2": r"e4m3x2", "bar": r"bar\.sync|barrier\.sync"}
    emit(kind="report", what="triton_ptx", layout=label, lines=ptx.count("\n"), **{k2: len(re.findall(v, ptx)) for k2, v in pats.items()})


def main():
    try:
        H.load_plugin(MUTATIONS)
        from sgl_nvidia_mega.kimi_k31.dsa import idx_qfn8 as q8m
        from sgl_nvidia_mega.kimi_k31.dsa import index_qfn as qfn_mod

        torch.zeros(1, device="cuda")
        q8m.build()
        variants_built = not os.environ.get("K31_MUTATION")
        if variants_built:
            try:
                q8m._load(True)
            except Exception as ex:  # noqa: BLE001
                variants_built = False
                emit(kind="setup_error", what="variants build", err=repr(ex)[:400])
    except Exception as ex:  # noqa: BLE001
        emit(kind="setup_error", err=repr(ex)[:500])
        raise
    mutated = bool(os.environ.get("K31_MUTATION"))
    cells = []
    for seed, eps in ((1, 1e-5), (2, 1e-6)):
        for layout in ("p3040", "p3048", "poff"):
            for rows in (1, 2, 33, 40, 48, 56, 64, 71, 72, 96, 128):
                if layout == "poff" and rows not in (2, 40, 64, 128):
                    continue
                for sfmt in ("ue8m0", None):
                    if sfmt is None and rows not in (2, 40, 72):
                        continue
                    cells.append((f"{layout}.b{rows}.s{seed}.{sfmt or 'none'}", rows, seed, layout, eps, sfmt, False))
    cells.append(("p3040.bulk4096.s3.ue8m0", 4096, 3, "p3040", 1e-6, "ue8m0", True))
    cells.append(("p3048.bulk4096.s4.ue8m0", 4096, 4, "p3048", 1e-6, "ue8m0", True))
    cells.append(("pow2.b64.ue8m0", 64, 0, "pow2", 0.0, "ue8m0", False))
    q1_fail = 0
    bulk = []
    for tag, rows, seed, layout, eps, sfmt, is_bulk in cells:
        try:
            acc, projected, gate = make_pow2(rows) if layout == "pow2" else make_inputs(rows, seed, layout, bulk=is_bulk)
            tq, tw = triton_ref(qfn_mod, acc, projected, gate, eps, sfmt)
            cq, cw, cr = cuda_out(q8m, acc, projected, gate, eps, sfmt, debug=True)
            pq, pw, _ = cuda_out(q8m, acc, projected, gate, eps, sfmt)  # served launch form (no debug hooks)
            torch.cuda.synchronize()
            nq, nw = ndiff(tq, pq), wdiff(tw, pw)
            hook_eq = ndiff(cq, pq) == 0 and wdiff(cw, pw) == 0
            ok = nq == 0 and nw == 0 and hook_eq
            q1_fail += not ok
            if not ok or not mutated:
                check(f"Q1_bitwise_vs_triton.{tag}", ok, q8_diff=nq, w_diff=nw, hooked_eq_served=hook_eq, rows=rows)
            if mutated:
                continue
            if sfmt is not None:
                scale_ok, frac = fp64_ok(acc, projected, eps, pq.float())
                check(f"Q5_fp64.{tag}", scale_ok and frac >= 0.999, scale_ok=scale_ok, within_frac=frac)
            if is_bulk:
                bulk.append((tag, acc, projected, gate, eps, sfmt, tq, tw, cr))
            if is_bulk or layout == "pow2" or rows not in (40, 64):
                continue
            # Q2 dispatch
            n0 = q8m.STATS["calls"]
            qfn_mod.QFN_CUDA = True
            dq_, dw_ = qfn_mod.qfn_quant_head_gate(acc, projected, gate, eps, 128 ** -0.5, sfmt)
            n1 = q8m.STATS["calls"]
            qfn_mod.QFN_CUDA = False
            qfn_mod.qfn_quant_head_gate(acc, projected, gate, eps, 128 ** -0.5, sfmt)
            n2 = q8m.STATS["calls"]
            torch.cuda.synchronize()
            check(f"Q2_dispatch.{tag}", n1 - n0 == 1 and n2 == n1 and ndiff(dq_, pq) == 0 and wdiff(dw_, pw) == 0, cuda_calls=n1 - n0,
                  off_calls=n2 - n1)
            # Q3 out= owner rows
            R, lo = rows + 16, 8
            qb = torch.empty((R, 32, 128), device="cuda", dtype=torch.float8_e4m3fn)
            wb = torch.empty((R, 32, 1), device="cuda", dtype=torch.float32)
            H.poison(qb, wb)
            qfn_mod.QFN_CUDA = True
            rq, rw = qfn_mod.qfn_quant_head_gate(acc, projected, gate, eps, 128 ** -0.5, sfmt, out=(qb[lo: lo + rows], wb[lo: lo + rows]))
            qfn_mod.QFN_CUDA = False
            torch.cuda.synchronize()
            inside = ndiff(qb[lo: lo + rows], pq) == 0 and wdiff(wb[lo: lo + rows], pw) == 0 and rq.data_ptr() == qb[lo].data_ptr()
            outside = bool((qb[:lo].view(torch.uint8) == 0xFF).all() and (qb[lo + rows:].view(torch.uint8) == 0xFF).all()
                           and (wb[:lo].view(torch.uint8) == 0xFF).all() and (wb[lo + rows:].view(torch.uint8) == 0xFF).all())
            check(f"Q3_out_rows.{tag}", inside and outside, inside=inside, outside_poisoned=outside)
            # Q4 graph replay
            gq = torch.empty_like(pq)
            gw = torch.empty_like(pw)
            qfn_mod.QFN_CUDA = True
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                qfn_mod.qfn_quant_head_gate(acc, projected, gate, eps, 128 ** -0.5, sfmt, out=(gq, gw))
            torch.cuda.current_stream().wait_stream(st)
            torch.cuda.synchronize()
            gph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gph):
                qfn_mod.qfn_quant_head_gate(acc, projected, gate, eps, 128 ** -0.5, sfmt, out=(gq, gw))
            H.poison(gq, gw)
            gph.replay()
            torch.cuda.synchronize()
            check(f"Q4_graph_eq_eager.{tag}", ndiff(gq, pq) == 0 and wdiff(gw, pw) == 0)
            # Q6 capture decline (module marked unbuilt)
            if rows == 64 and layout == "p3040":
                d0 = q8m.STATS["declined"]
                q8m._STATE["built"] = False
                xq = torch.empty_like(pq)
                xw = torch.empty_like(pw)
                g2 = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g2):
                    qfn_mod.qfn_quant_head_gate(acc, projected, gate, eps, 128 ** -0.5, sfmt, out=(xq, xw))
                q8m._STATE["built"] = True
                H.poison(xq, xw)
                g2.replay()
                torch.cuda.synchronize()
                check(f"Q6_capture_decline.{tag}", q8m.STATS["declined"] == d0 + 1 and ndiff(xq, tq) == 0 and wdiff(xw, tw) == 0,
                      declined=q8m.STATS["declined"] - d0)
            qfn_mod.QFN_CUDA = False
        except Exception as ex:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            check(f"cell_ran.{tag}", False, err=repr(ex)[:300])
            qfn_mod.QFN_CUDA = False
    if mutated:
        emit(kind="DONE", all_ok=all(OKS) and q1_fail == 0, n=len(OKS), failed=sum(not o for o in OKS), q1_failed_cells=q1_fail)
        return 0 if all(OKS) and q1_fail == 0 else 1
    # Q7 + REPORT: variants on the bulk cells
    try:
        seen_sensitive = False
        for v in VARIANTS:
            if not variants_built and v != (0, 0, 0, 0):
                break
            tot_q = tot_w = tot_r = nrows = 0
            for tag, acc, projected, gate, eps, sfmt, tq, tw, cr in bulk:
                vq, vw, vr = cuda_out(q8m, acc, projected, gate, eps, sfmt, variant=v, debug=True)
                torch.cuda.synchronize()
                tot_q += ndiff(vq, tq)
                tot_w += wdiff(vw, tw)
                tot_r += int((vr.view(torch.int32) != cr.view(torch.int32)).sum())
                nrows += acc.shape[0]
            emit(kind="report", what="variant", inner=v[0], cross=v[1], tail=v[2], scl=v[3], q8_diff=tot_q, w_diff=tot_w, rstd_rows_diff=tot_r,
                 rows=nrows)
            if tot_r >= 0.01 * nrows and tot_q >= 1:
                seen_sensitive = True
        check("Q7_sensitivity", seen_sensitive and variants_built, variants_built=variants_built)
    except Exception as ex:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        check("Q7_sensitivity", False, err=repr(ex)[:300])
    for label, layout in (("vec8_p3040", "p3040"), ("vec1_p3048", "p3048")):
        try:
            acc, projected, gate = make_inputs(40, 9, layout)
            ptx_report(qfn_mod, acc, projected, gate, 1e-6, label)
        except Exception as ex:  # noqa: BLE001
            emit(kind="report", what="triton_ptx", layout=label, err=repr(ex)[:300])
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
