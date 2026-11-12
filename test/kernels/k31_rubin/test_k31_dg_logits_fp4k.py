#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GPU test of dsa/dg_logits_fp4k.py (K31MK_DSA_IDX_FP4_FP8Q: FP8 index query x MXFP4 index-K verify logits, the vendored DeepGEMM
paged kernel instantiated with cutlass::detail::float_e2m1_unpacksmem_t).

Bundle-entry correctness (fixed before the first run):
  V1 BITWISE at every cell: the kernel's FP32 logits equal deep_gemm.fp8_fp4_paged_mqa_logits' FP8-MX form (stock DeepGEMM, e4m3 x e4m3
     MXF8F6F4) on the SAME values: the same FP8 query with unit MX scales, and index-K pages holding each e2m1 value as its exact e4m3
     byte with the same ue8m0 scales (132 B rows). Every valid position t < ctx[b, j] of every row.
  V2 FP64: |kernel - ref| <= 1e-5 x sum_h |w[h]| sum_d |q[h, d] k[d]| at every valid position, ref = FP64 sum_h w[h] relu(q[h] . k) from
     the FP8 query bytes and the MXFP4 pages decoded by the OCP MX spec in this file (independent of the writer and of DeepGEMM).
  Cells (Q8 unless noted): 1 x 131072, 2 x 131072, 12 x 131072, 16 x 8192, 3 ragged (4097 / 70 / 2049) on pages written by
     idx_fp4.store_index_k (the served writer), one strided pool (page stride 8448 B, MXFP4 rows in the first 4352 B), 3 x 20000 at Q6.
  PASS = V1 and V2 at every cell (r2: a NaN logit fails V2; r1 ignored NaN there, V1's finite check covered it). A V1 miss with V2 passing is NOT a pass (it would mean the datapath differs from my expectation or
  a layout bug; it gets a control before any use).
Report-only (EXPLORATORY, 1 GPU, not the served topology): per-call us (CUDA graph of 24 calls, 3 rotations of the order) of the served
  FP8 vendored kernel (dg_logits, 132 B FP8 pages), DeepGEMM FP8-MX, the MXFP4 (k + q) vendored kernel (dg_logits_fp4) and this kernel,
  at Q8 x {1, 2, 12, 16} x 131072 and {1, 12} x 8192.
Mutants (run by the runner on tree copies; each must FAIL V1): sfq (query unit scale 2^1), mmaA (header: index-K operand read as e4m3),
  noswz (index-K map without the 128-byte swizzle), sfkv (index-K scale map base + 16 B).
Exit: 0 all checks pass, 1 any check fails, 2 setup error. One JSON line per check; DONE last.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import sys
import types

import torch

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def emit(**kw):
    print(json.dumps(kw), flush=True)


def load(tree, host_dir):
    try:
        import sglang.srt.runtime_context  # noqa: F401
    except Exception:
        m = types.ModuleType("sglang.srt.runtime_context")

        class _P:
            is_blackwell = True

        m.get_platform = lambda: _P()
        sys.modules["sglang.srt.runtime_context"] = m
    if host_dir:
        spec = importlib.util.spec_from_file_location("sglang.kernels.ops.quantization.fp4_indexer_quant",
                                                      os.path.join(host_dir, "fp4_indexer_quant.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    sys.path.insert(0, tree)
    mods = {n: importlib.import_module(f"sgl_nvidia_mega.kimi_k31.dsa.{n}") for n in ("idx_fp4", "dg_logits", "dg_logits_fp4", "dg_logits_fp4k")}
    return mods


def decode_pages(pages_u8, page_b):
    """MXFP4 pages [P, page_b] (first 4352 B = 64 x 64 B e2m1, low nibble = even element, then 64 x 4 B ue8m0) -> FP64 keys [P, 64, 128]
    by the OCP MX spec: value = (-1)^s * E2M1[code & 7] * 2^(ue8m0 - 127)."""
    P = pages_u8.shape[0]
    vals = pages_u8[:, : 64 * 64].reshape(P, 64, 64).to(torch.int64)
    lo, hi = vals & 0xF, vals >> 4
    codes = torch.stack([lo, hi], dim=-1).reshape(P, 64, 128)
    lut = torch.tensor(E2M1, dtype=torch.float64, device=pages_u8.device)
    mag = lut[codes & 7]
    sign = torch.where((codes & 8) != 0, -1.0, 1.0).to(torch.float64)
    sf = pages_u8[:, 64 * 64: 64 * 68].reshape(P, 64, 4).to(torch.float64)
    scale = torch.pow(torch.tensor(2.0, dtype=torch.float64, device=pages_u8.device), sf - 127.0)
    return sign * mag * scale.repeat_interleave(32, dim=-1)


def to_mx8_pages(pages_u8, page_b):
    """The same MXFP4 values as FP8-MX pages [P, 64 x 132] (64 x 128 e4m3 bytes, then the same 64 x 4 ue8m0 bytes): e2m1 values are exact in e4m3."""
    P = pages_u8.shape[0]
    vals = pages_u8[:, : 64 * 64].reshape(P, 64, 64).to(torch.int64)
    codes = torch.stack([vals & 0xF, vals >> 4], dim=-1).reshape(P, 64, 128)
    lut = torch.tensor([v for v in E2M1] + [-v for v in E2M1], dtype=torch.float32, device=pages_u8.device)
    e4m3 = lut[codes].to(torch.float8_e4m3fn).view(torch.uint8)
    assert torch.equal(e4m3.view(torch.float8_e4m3fn).to(torch.float32), lut[codes]), "e2m1 -> e4m3 not exact"
    out = torch.empty((P, 64 * 132), dtype=torch.uint8, device=pages_u8.device)
    out[:, : 64 * 128] = e4m3.reshape(P, -1)
    out[:, 64 * 128:] = pages_u8[:, 64 * 64: 64 * 68]
    return out


def time_graph(fns, warm=3, reps=10):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns[:2]:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fns:
            f()
    for _ in range(warm):
        g.replay()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(reps):
        st.record()
        g.replay()
        en.record()
        en.synchronize()
        ts.append(st.elapsed_time(en) * 1000.0 / len(fns))
    return statistics.median(ts)


def make_cell(m, g, dev, ctxs, N, use_writer, strided):
    fp4 = m["idx_fp4"]
    B = len(ctxs)
    wpages = -(-max(ctxs) // 64)
    npool = B * wpages + 3
    table = torch.randperm(npool, generator=g, device=dev)[: B * wpages].view(B, wpages).to(torch.int32)
    page_b = 8448 if strided else fp4.PAGE_B
    raw = torch.randint(0, 256, (npool, page_b), generator=g, device=dev, dtype=torch.uint8)
    # realistic scales (as the writer makes them for O(1) keys): ue8m0 in [120, 130]
    sf = raw[:, 64 * 64: 64 * 68]
    sf.copy_(torch.randint(120, 131, sf.shape, generator=g, device=dev, dtype=torch.uint8))
    if use_writer:
        keys = torch.randn(B * wpages * 64, 128, generator=g, device=dev).to(torch.bfloat16)
        slot = (table.to(torch.int64)[:, :, None] * 64 + torch.arange(64, device=dev)[None, None, :]).reshape(-1)
        dense = torch.empty((npool, fp4.PAGE_B), dtype=torch.uint8, device=dev)
        dense.copy_(raw[:, : fp4.PAGE_B])
        fp4.store_index_k(keys, dense, slot)
        raw[:, : fp4.PAGE_B] = dense
    cache = raw[:, : fp4.PAGE_B].as_strided((npool, 64, 1, 68), (page_b, 68, 68, 1))
    qf = torch.randn(B * N, 32, 128, generator=g, device=dev) * 3.0
    q8 = qf.to(torch.float8_e4m3fn)
    w = (torch.randn(B * N, 32, generator=g, device=dev) * 0.05).contiguous()
    ctx2 = torch.tensor([[c - (N - 1 - j) for j in range(N)] for c in ctxs], dtype=torch.int32, device=dev)
    return dict(B=B, N=N, wpages=wpages, npool=npool, table=table, raw=raw, page_b=page_b, cache=cache, q8=q8, w=w, ctx2=ctx2,
                maxlen=wpages * 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--host-dir", default="")
    ap.add_argument("--no-timing", action="store_true")
    args = ap.parse_args()
    m = load(os.path.abspath(args.tree), args.host_dir)
    k4 = m["dg_logits_fp4k"]
    import deep_gemm

    dev = torch.device("cuda")
    torch.zeros(1, device=dev)  # primary context before the driver-API cubin load
    torch.cuda.synchronize()
    nsm = torch.cuda.get_device_properties(0).multi_processor_count
    g = torch.Generator(device="cuda").manual_seed(1108)
    oks = []
    cells = [("1x131072", (131072,), 8, False, False), ("2x131072", (131072,) * 2, 8, False, False),
             ("12x131072", (131072,) * 12, 8, False, False), ("16x8192", (8192,) * 16, 8, False, False),
             ("ragged3_writer", (4097, 70, 2049), 8, True, False), ("strided3", (3000, 64, 1025), 8, True, True),
             ("q6_3x20000", (20000,) * 3, 6, False, False)]
    for name, ctxs, N, use_writer, strided in cells:
        c = make_cell(m, g, dev, ctxs, N, use_writer, strided)
        B, table, ctx2, maxlen = c["B"], c["table"], c["ctx2"], c["maxlen"]
        sched = deep_gemm.get_paged_mqa_logits_metadata(ctx2, 64, nsm)
        q4d = c["q8"].view(B, N, 32, 128)
        got = k4.paged_mqa_logits(q4d, c["cache"], c["w"], ctx2, table, sched, maxlen)
        mx8 = to_mx8_pages(c["raw"][:, : 64 * 68].contiguous(), c["page_b"]).view(c["npool"], 64, 1, 132)
        sfq = torch.full((B, N, 32), k4.UNIT_SF, dtype=torch.int32, device=dev)
        ref = deep_gemm.fp8_fp4_paged_mqa_logits((q4d, sfq), mx8, c["w"], ctx2, table, sched, maxlen, False, logits_dtype=torch.float32)
        torch.cuda.synchronize()
        valid = torch.arange(maxlen, device=dev)[None, :] < ctx2.view(-1, 1)
        a, b = got.contiguous()[valid], ref.contiguous()[valid]
        same = torch.equal(a.view(torch.int32), b.view(torch.int32))
        ndiff = int((a.view(torch.int32) != b.view(torch.int32)).sum())
        maxabs = float((a - b).abs().max()) if a.numel() else 0.0
        oks.append(bool(same))
        emit(kind="check", name=f"V1_bitwise_vs_dg_fp8mx_{name}", ok=bool(same), differing=ndiff, valid=int(valid.sum()), max_abs_diff=maxabs,
             finite=bool(torch.isfinite(a).all().item()), ref_absmax=float(b.abs().max()))
        # V2: FP64 reference from the FP8 query bytes and the spec-decoded MXFP4 pages
        keys = decode_pages(c["raw"][:, : 64 * 68].contiguous(), c["page_b"])  # [P, 64, 128] fp64
        qd = c["q8"].to(torch.float64).view(B, N, 32, 128)
        wd = c["w"].to(torch.float64).view(B, N, 32)
        worst = 0.0
        gotc = got.contiguous().view(B, N, -1)
        for bi in range(B):
            L = int(ctx2[bi].max())
            k = keys[table[bi].to(torch.int64)].reshape(-1, 128)[:L]  # [L, 128]
            dots = torch.einsum("nhd,td->nht", qd[bi], k)  # [N, 32, L]
            absd = torch.einsum("nhd,td->nht", qd[bi].abs(), k.abs())
            refl = (wd[bi][:, :, None] * dots.clamp_min(0)).sum(1)  # [N, L]
            bound = (wd[bi].abs()[:, :, None] * absd).sum(1)
            vmask = torch.arange(L, device=dev)[None, :] < ctx2[bi].view(-1, 1)
            err = (gotc[bi, :, :L].to(torch.float64) - refl).abs()
            ratio = (err / bound.clamp_min(1e-30))[vmask]
            r = float(ratio.max())
            worst = r if r != r else max(worst, r)  # NaN stays NaN (a non-finite logit fails V2)
            if worst != worst:
                break
            del dots, absd, refl, bound, err
        ok2 = worst == worst and worst <= 1e-5
        oks.append(ok2)
        emit(kind="check", name=f"V2_fp64_{name}", ok=ok2, worst_err_over_abs_bound=worst, line=1e-5)
        del keys, mx8, got, ref
        torch.cuda.empty_cache()
    if not args.no_timing:
        fp4, dgv, ven4 = m["idx_fp4"], m["dg_logits"], m["dg_logits_fp4"]
        dgv.build(0, 8)
        ven4.build(0, 8)
        for name, ctxs in (("1x131072", (131072,)), ("2x131072", (131072,) * 2), ("12x131072", (131072,) * 12),
                           ("16x131072", (131072,) * 16), ("1x8192", (8192,)), ("12x8192", (8192,) * 12)):
            N = 8
            c = make_cell(m, g, dev, ctxs, N, False, False)
            B, table, ctx2, maxlen = c["B"], c["table"], c["ctx2"], c["maxlen"]
            sched = deep_gemm.get_paged_mqa_logits_metadata(ctx2, 64, nsm)
            q4d = c["q8"].view(B, N, 32, 128)
            mx8 = to_mx8_pages(c["raw"][:, : 64 * 68].contiguous(), c["page_b"])
            fp8 = mx8.clone()  # served FP8 pages: same e4m3 bytes, FP32 scale 1.0 per token
            fp8[:, 64 * 128:] = torch.ones(64, dtype=torch.float32, device=dev).view(torch.uint8).repeat(fp8.shape[0], 1)
            mx8 = mx8.view(c["npool"], 64, 1, 132)
            fp8 = fp8.view(c["npool"], 64, 1, 132)
            sfq = torch.full((B, N, 32), k4.UNIT_SF, dtype=torch.int32, device=dev)
            qq = (torch.randn(B * N, 32, 128, generator=g, device=dev) * 2).to(torch.bfloat16)
            gate = torch.randn(B * N, 32, generator=g, device=dev).to(torch.bfloat16)
            fq, w4 = fp4.quantize_query(qq, gate, fp4.QueryBuffers(B * N, dev))
            qv4 = (fq.q4.view(B, N, 32, 64), fq.sf.view(B, N, 32))
            w4 = w4.view(B * N, 32)
            fns = {
                "fp8_ven": lambda: dgv.paged_mqa_logits(q4d, fp8, c["w"], ctx2, table, sched, maxlen),
                "fp8mx_dg": lambda: deep_gemm.fp8_fp4_paged_mqa_logits((q4d, sfq), mx8, c["w"], ctx2, table, sched, maxlen, False,
                                                                     logits_dtype=torch.float32),
                "fp4kq_ven": lambda: ven4.paged_mqa_logits(qv4, c["cache"], w4, ctx2, table, sched, maxlen),
                "fp4k": lambda: k4.paged_mqa_logits(q4d, c["cache"], c["w"], ctx2, table, sched, maxlen),
            }
            order = list(fns)
            t = {k: [] for k in order}
            for r in range(3):
                seq = order if r % 2 == 0 else order[::-1]
                for k in seq:
                    t[k].append(time_graph([fns[k]] * 24))
            res = {k: round(statistics.median(v), 3) for k, v in t.items()}
            res["fp4k_over_fp8_ven"] = round(res["fp4k"] / res["fp8_ven"], 3)
            res["spread_fp4k"] = round(max(t["fp4k"]) - min(t["fp4k"]), 3)
            res["spread_fp8_ven"] = round(max(t["fp8_ven"]) - min(t["fp8_ven"]), 3)
            emit(kind="report", name=f"time_us_{name}", **res)
            del mx8, fp8, c
            torch.cuda.empty_cache()
    emit(kind="DONE", all_ok=all(oks), n=len(oks), failed=sum(not o for o in oks))
    return 0 if all(oks) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as ex:  # noqa: BLE001
        emit(kind="setup_error", err=repr(ex)[:500])
        import traceback

        traceback.print_exc()
        sys.exit(2)
