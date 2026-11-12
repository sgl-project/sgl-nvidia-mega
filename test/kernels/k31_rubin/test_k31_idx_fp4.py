#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GPU unit test of dsa/idx_fp4.py (K31MK_DSA_IDX_FP4: MXFP4 index-K writer, MXFP4 index query + head gate, fp8_fp4 paged logits).

, bundle-entry correctness for the lever class "numerics-changing, FP64-referenced" (fixed before the first run).
Checks and tolerances (fixed before the run):
  W1 writer bytes: every written slot's 64 value bytes + 4 scale bytes (page layout A of a [pages, 4352] pool) EQUAL the host prefill
     quantizer's bytes (fp4_indexer_quant.quantize_fp4_indexer_tensor of the same BF16 key) - exact.
  W2 writer vs the format spec (independent decoder written here from the MXFP4 definition, float64): each 32-group exponent byte equals
     127 + ceil(log2(max(fp32(amax / 6), 1e-4))) clamped to [1, 254]; each decoded element is within half the local e2m1 grid step x 2^(e-127)
     of the BF16 input (inputs above 6 x scale clamp to 6 x scale) - exact predicate.
  W3 writer locality: every byte outside the written slots (values and scales) keeps the 0xA5 fill; negative slots write nothing - exact.
  Q1 query bytes: q4 / sf of quantize_query EQUAL quantize_fp4_indexer_tensor of the query rows - exact.
  Q2 head weights: EQUAL float32(bfloat16(gate * 32**-0.5)) * 128**-0.5 computed in torch - exact.
  Q3 owner rows (request sharding): Fp4Query.index_select(rows) EQUALS the q4 / sf rows picked by advanced indexing - exact (added in r2).
  L1 logits vs FP64: on pages written by store_index_k and the query from quantize_query, for B in {1, 3, 12} requests x Q 8 rows with ragged
     contexts (not multiples of 64) and a permuted block table, |logit - ref| <= 1e-5 x bound + 1e-12 at every valid position t < ctx[b, j],
     where ref / bound are float64 sum_h w_h relu(q_h . k_t) / sum_h |w_h| sum_d |q_hd k_td| over the DEQUANTIZED q and k (decoder of W2).
     (FP32 accumulation of exact e2m1 x e2m1 x 2^k products over 128 dims and 32 heads is within ~1e-5 of the absolute sum.)
  R1 refusals (CPU): check_cache rejects a [pages, 8448] FP8 pool and a non-uint8 pool.
Report-only (no verdict): top-2048 overlap of the FP4 logits vs FP64 logits of the BF16 (unquantized) q and k per row.
Mutants (run by the runner on copies of idx_fp4.py; each must FAIL at least one check above):
  sfoff (scale address + 1 slot) -> W1 / W3 / L1; planar (value row pitch 68 instead of 64) -> W1 / W3 / L1;
  nibble (q nibble order swapped) -> Q1 / L1; gate (no 32**-0.5 on the head gate) -> Q2 / L1.
Exit: 0 all checks pass, 1 any check fails, 2 setup error. One JSON line per check; DONE last.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import types

import torch


def emit(**kw):
    print(json.dumps(kw), flush=True)


def load(pkg_dir, host_dir):
    try:
        import sglang.srt.runtime_context  # noqa: F401
    except Exception:  # the runner runs without the host tree: the quantizer only asks is_blackwell
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
    pkg = types.ModuleType("l2mkpkg")
    pkg.__path__ = [pkg_dir]
    sys.modules["l2mkpkg"] = pkg
    out = []
    for name in ("idx_fp4_knob", "idx_fp4"):
        spec = importlib.util.spec_from_file_location(f"l2mkpkg.{name}", os.path.join(pkg_dir, f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        out.append(mod)
    from sglang.kernels.ops.quantization.fp4_indexer_quant import quantize_fp4_indexer_tensor

    return out[1], quantize_fp4_indexer_tensor


E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)


def decode_rows(vals_u8, sf_u8):
    """Independent MXFP4 decoder: vals uint8 [T, 64] (low nibble = even element), sf uint8 [T, 4] (byte g = biased exponent of group g)
    -> float64 [T, 128]."""
    v = vals_u8.to(torch.int64).cpu()
    codes = torch.stack([v & 0xF, v >> 4], dim=-1).reshape(v.shape[0], 128)
    mag = E2M1[codes & 7]
    val = torch.where((codes & 8) != 0, -mag, mag)
    e = sf_u8.to(torch.int64).cpu()
    scale = torch.pow(2.0, (e - 127).to(torch.float64))
    return (val.view(-1, 4, 32) * scale[:, :, None]).reshape(-1, 128)


def spec_exponents(x_bf16):
    """127 + ceil(log2(max(fp32(amax / 6), 1e-4))) per 32-group, clamped [1, 254] (frexp on the fp32 value; independent of the kernel)."""
    a = x_bf16.float().view(-1, 4, 32).abs().amax(dim=-1)
    s = torch.clamp(a / 6.0, min=1.0e-4).cpu().to(torch.float64)  # fp32 division, then exact in float64
    m, e = torch.frexp(s)  # s = m * 2^e, m in [0.5, 1)
    ceil_log2 = torch.where(m == 0.5, e - 1, e)  # exact power of two -> log2 exact
    return torch.clamp(ceil_log2 + 127, 1, 254)


def grid_ok(x_bf16, dec, exps):
    """|dec - x| <= half the local e2m1 step x scale (x clamped to +-6 x scale)."""
    x = x_bf16.cpu().to(torch.float64).view(-1, 4, 32)
    scale = torch.pow(2.0, (exps - 127).to(torch.float64))[:, :, None]
    y = (x / scale).clamp(-6.0, 6.0)
    step = torch.where(y.abs() <= 2.0, 0.5, torch.where(y.abs() <= 4.0, 1.0, 2.0))
    err = (dec.view(-1, 4, 32) / scale - y).abs()
    return bool((err <= step / 2 + 1e-12).all().item()), float(err.max().item())


def page_bytes(cache, slots, m):
    """(vals uint8 [T, 64], sf uint8 [T, 4]) at token slots ``slots`` of a [pages, 4352] pool (layout A)."""
    s = slots.to(torch.int64)
    page, off = s // 64, s % 64
    flat = cache.view(-1)
    base = page * m.PAGE_B
    vi = (base + off * 64)[:, None] + torch.arange(64, device=cache.device)[None, :]
    si = (base + 64 * 64 + off * 4)[:, None] + torch.arange(4, device=cache.device)[None, :]
    return flat[vi], flat[si]


def check(name, ok, **kw):
    emit(kind="check", name=name, ok=bool(ok), **kw)
    return bool(ok)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--pkg-dir", default=os.path.join(here, "..", "..", "..", "sgl_nvidia_mega", "kimi_k31", "dsa"))
    ap.add_argument("--host-dir", default="")
    args = ap.parse_args()
    m, qref = load(os.path.abspath(args.pkg_dir), args.host_dir)
    dev = torch.device("cuda")
    g = torch.Generator(device="cuda").manual_seed(31)
    oks = []

    # ---------------- W1-W3 writer
    pages, T = 50, 200
    cache = torch.full((pages, m.PAGE_B), 0xA5, dtype=torch.uint8, device=dev)
    key = (torch.randn(T, 128, generator=g, device=dev) *
           torch.pow(2.0, torch.randint(-10, 7, (T, 4, 1), generator=g, device=dev).float()).repeat(1, 1, 32).view(T, 128))
    key[0] = 0.0
    key[1, 5] = 3.0e4  # one outlier group
    key[2] = -key[2].abs()
    key[3] = 1.0e-7  # below the 1e-4 x 6 exponent clamp
    key = key.to(torch.bfloat16)
    perm = torch.randperm(pages * 64, generator=g, device=dev)[:T]
    slots = perm.clone()
    skip = torch.arange(0, T, 10, device=dev)
    slots[skip] = -1
    m.store_index_k(key, cache, slots)
    torch.cuda.synchronize()
    live = slots >= 0
    vals, sf = page_bytes(cache, slots[live], m)
    q4, qsf = qref(key[live])
    ref_vals = q4.view(torch.uint8)
    ref_sf = qsf.contiguous().view(torch.uint8).view(-1, 4)
    oks.append(check("W1_writer_bytes_eq_host_quantizer", torch.equal(vals, ref_vals) and torch.equal(sf, ref_sf),
                     rows=int(live.sum()), val_mismatch=int((vals != ref_vals).sum()), sf_mismatch=int((sf != ref_sf).sum())))
    exps = spec_exponents(key[live])
    dec = decode_rows(vals, sf)
    g_ok, g_err = grid_ok(key[live], dec, exps)
    oks.append(check("W2_writer_vs_spec", torch.equal(sf.cpu().to(torch.int64), exps.to(torch.int64)) and g_ok,
                     exp_mismatch=int((sf.cpu().to(torch.int64) != exps.to(torch.int64)).sum()), grid_ok=g_ok, max_err_steps=g_err))
    touched = torch.zeros(pages * m.PAGE_B, dtype=torch.bool, device=dev)
    s = slots[live].to(torch.int64)
    vi = ((s // 64) * m.PAGE_B + (s % 64) * 64)[:, None] + torch.arange(64, device=dev)[None, :]
    si = ((s // 64) * m.PAGE_B + 4096 + (s % 64) * 4)[:, None] + torch.arange(4, device=dev)[None, :]
    touched[vi.reshape(-1)] = True
    touched[si.reshape(-1)] = True
    untouched = cache.view(-1)[~touched]
    oks.append(check("W3_writer_locality", bool((untouched == 0xA5).all().item()), changed=int((untouched != 0xA5).sum())))

    # ---------------- Q1-Q2 query + head gate
    R = 104
    q = (torch.randn(R, 32, 128, generator=g, device=dev) * 2).to(torch.bfloat16)
    gate_buf = torch.randn(R, 3040, generator=g, device=dev).to(torch.bfloat16)
    gate = gate_buf[:, 128:160]
    bufs = m.QueryBuffers(128, dev)
    fq, w = m.quantize_query(q, gate, bufs)
    torch.cuda.synchronize()
    r4, rsf = qref(q.view(-1, 128))
    oks.append(check("Q1_query_bytes_eq_host_quantizer",
                     torch.equal(fq.q4.reshape(-1, 64).view(torch.uint8), r4.view(torch.uint8)) and torch.equal(fq.sf.reshape(-1), rsf),
                     q_mismatch=int((fq.q4.reshape(-1, 64).view(torch.uint8) != r4.view(torch.uint8)).sum()),
                     sf_mismatch=int((fq.sf.reshape(-1) != rsf).sum())))
    wref = (gate.float() * 32 ** -0.5).to(torch.bfloat16).float() * 128 ** -0.5
    oks.append(check("Q2_head_weights", torch.equal(w.view(R, 32), wref), max_abs=float((w.view(R, 32) - wref).abs().max().item())))
    own = torch.tensor([3, 17, 17, 64, 103, 0], device=dev)
    sel = fq.index_select(own)
    oks.append(check("Q3_owner_rows", torch.equal(sel.q4, fq.q4[own]) and torch.equal(sel.sf, fq.sf[own]) and sel.q4.is_contiguous()
                     and sel.shape == (6, 32, 128)))

    # ---------------- L1 logits vs FP64 over pages written by the writer
    import deep_gemm

    nsm = torch.cuda.get_device_properties(0).multi_processor_count
    for B, ctxs in ((1, (1000,)), (3, (4097, 70, 2049)), (12, tuple(1500 + 613 * i for i in range(12)))):
        N = 8
        maxctx = max(ctxs)
        wpages = -(-maxctx // 64)
        npool = B * wpages + 5
        pool = torch.full((npool, m.PAGE_B), 0xA5, dtype=torch.uint8, device=dev)
        table = torch.randperm(npool, generator=g, device=dev)[: B * wpages].view(B, wpages).to(torch.int32)
        keys = (torch.randn(B, wpages * 64, 128, generator=g, device=dev)).to(torch.bfloat16)
        slot = (table.to(torch.int64)[:, :, None] * 64 + torch.arange(64, device=dev)[None, None, :]).reshape(B, -1)
        m.store_index_k(keys.view(-1, 128), pool, slot.view(-1))
        qq = (torch.randn(B * N, 32, 128, generator=g, device=dev) * 2).to(torch.bfloat16)
        gg = torch.randn(B * N, 32, generator=g, device=dev).to(torch.bfloat16)
        qb = m.QueryBuffers(B * N, dev)
        fq, ww = m.quantize_query(qq, gg, qb)
        ctx2 = torch.tensor([[c - (N - 1 - j) for j in range(N)] for c in ctxs], dtype=torch.int32, device=dev)
        sched = deep_gemm.get_paged_mqa_logits_metadata(ctx2, 64, nsm)
        out = m.paged_logits(fq, ww, pool, ctx2, table, sched, wpages * 64, B, N)
        torch.cuda.synchronize()
        out = out.cpu().to(torch.float64)
        qd = decode_rows(fq.q4.reshape(-1, 64).view(torch.uint8), fq.sf.reshape(-1).contiguous().view(torch.uint8).view(-1, 4)).view(B * N, 32, 128)
        worst, nbad, ov = 0.0, 0, []
        for b in range(B):
            kv, ks = page_bytes(pool, slot[b, : ctxs[b]], m)
            kd = decode_rows(kv, ks)  # [ctx, 128]
            kraw = keys[b, : ctxs[b]].cpu().to(torch.float64)
            for j in range(N):
                r = b * N + j
                L = int(ctx2[b, j])
                dots = qd[r] @ kd[:L].T  # [32, L]
                wr = ww.view(-1, 32)[r].cpu().to(torch.float64)
                ref = (wr[:, None] * dots.clamp(min=0)).sum(0)
                bound = (wr.abs()[:, None] * (qd[r].abs() @ kd[:L].abs().T)).sum(0)
                err = (out[r, :L] - ref).abs()
                bad = err > 1e-5 * bound + 1e-12
                nbad += int(bad.sum())
                worst = max(worst, float((err / (bound + 1e-30)).max()))
                if L > 2048:
                    raw = (wr[:, None] * (qq[r].cpu().to(torch.float64) @ kraw[:L].T).clamp(min=0)).sum(0)
                    a = set(torch.topk(out[r, :L], 2048).indices.tolist())
                    bset = set(torch.topk(raw, 2048).indices.tolist())
                    ov.append(len(a & bset) / 2048)
        oks.append(check(f"L1_logits_vs_fp64_B{B}", nbad == 0, bad=nbad, worst_err_over_bound=worst))
        if ov:
            emit(kind="report", name=f"top2048_overlap_fp4_vs_bf16_B{B}", min=min(ov), mean=sum(ov) / len(ov))

    # ---------------- R1 refusals
    r_ok = True
    for bad_cache in (torch.zeros(4, 8448, dtype=torch.uint8, device=dev), torch.zeros(4, m.PAGE_B, dtype=torch.int8, device=dev)):
        try:
            m.check_cache(bad_cache)
            r_ok = False
        except ValueError:
            pass
    oks.append(check("R1_check_cache_refuses_fp8_pool", r_ok))
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
