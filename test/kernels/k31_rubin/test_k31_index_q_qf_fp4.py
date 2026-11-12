#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Query-front index query with the MXFP4 index-K store (index_q.cuh kKFP4; K31MK_DSA_IDX_FP4_FP8Q hands run_qf the host's 68 B
pages [pages, 64 * 68]), vs the same kernel's FP8 store and vs the stock MXFP4 writer's bytes. .

Bundle-entry correctness (fixed before the first run). Cells: rows 1 / 5 / 8 (the 8-row instance) and 12 / 16
(the 16-row instance), seeds 1 / 2; random tensors as test_k31_index_q_qf.test_qf_index_k (index-K columns 0.5 + 2 N(0, 1),
random FP32 LayerNorm weight / bias), distinct slots in 0xA5-poisoned pools, one negative slot (rows > 1). TP16 columns (6 heads).
  F1 the FP8 index-query codes and head weights of the MXFP4-store launch are BITWISE those of the FP8-store launch on the same
     inputs (the query side does not depend on kKFP4).
  F2 every byte of the MXFP4 pool outside the stored rows (64 B codes + 4 B exponents per non-negative slot) is unchanged.
  F3 the stored rows vs sglang's quantize_fp4_indexer_tensor (RNE off: the host prefill writer's and idx_fp4.store_index_k's row
     math) applied to flashinfer layernorm(projected[:, k_col : k_col + 128]) (stock's writer input): e2m1 codes <= 0.5 % differing,
     each within one signed magnitude step; ue8m0 bytes <= 1 % differing, each within 1. (The kernel's own LayerNorm sums in another
     FP32 order than flashinfer's: the FP8 QF store is held to 'within 1 step, <= 1 %' vs stock for the same reason.)
  Report only: the same comparison against an FP64 LayerNorm -> BF16 reference (control for the stock writer itself).
Mutants (each must FAIL; K31_MUTATION=<name>): div6 (group scale amax / 5), nib (nibble order of a pair swapped), group (row amax
instead of the 32-group amax), sfpos (exponent bytes in reverse group order).
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
MUTATIONS = {
    "div6": (_CUH, "const float sf = fmaxf(div_full(amax, 6.0f), 1.0e-4f);", "const float sf = fmaxf(div_full(amax, 5.0f), 1.0e-4f);"),
    "nib": (_CUH, "packed |= e2m1_code(div_full(v[j], scale)) << (4 * j);", "packed |= e2m1_code(div_full(v[j], scale)) << (4 * (j ^ 1));"),
    "group": (_CUH, "    for (int d = 4; d; d >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));\n    if (slot < 0) return;\n    const float sf",
              "    for (int d = 16; d; d >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));\n    if (slot < 0) return;\n    const float sf"),
    "sfpos": (_CUH, "page[64 * 64 + off * 4 + (lane >> 3)] = static_cast<uint8_t>(e);", "page[64 * 64 + off * 4 + (3 - (lane >> 3))] = static_cast<uint8_t>(e);"),
}
HEADS = int(os.environ["K31_TEST_HEADS"])
K_COL = 2112 + 128 * HEADS
G_COL = K_COL + 128
PITCH = G_COL + 32
PAGES = 6
OKS = []


def emit(**kw):
    print(json.dumps(kw), flush=True)


def check(name, ok, **kw):
    emit(kind="check", name=name, ok=bool(ok), **kw)
    OKS.append(bool(ok))


def index_weights(seed):
    from sgl_nvidia_mega.kimi_k31.dsa.indexer import IndexWeights

    g = H.generator(seed)
    rn = lambda *s: torch.randn(s, generator=g, device="cuda", dtype=torch.float32)  # noqa: E731
    bf = torch.bfloat16
    return IndexWeights((rn(4096, 1536) / math.sqrt(1536)).to(bf), (rn(32, 7168) / math.sqrt(7168)).to(bf), (1 + 0.1 * rn(1536)).to(bf), 1e-5)


def key_norm(seed):
    g = H.generator(seed)
    gamma = 1 + 0.2 * torch.randn(128, generator=g, device="cuda")
    beta = 0.5 * torch.randn(128, generator=g, device="cuda")
    return gamma.float().contiguous(), beta.float().contiguous(), 1e-6


def run_qf(w, projected, slots, cache, norm):
    from sgl_nvidia_mega.kimi_k31.dsa.index_q import FusedIndexQuery

    fused = FusedIndexQuery(w, projected.shape[0], qf_only=projected.shape[0] > 8)
    H.poison(fused.query, fused.gates)
    q8, gates = fused.run_qf(projected, K_COL, G_COL, slots, cache, norm)
    torch.cuda.synchronize()
    return q8.view(torch.uint8).reshape(-1, 32, 128).clone(), gates.reshape(-1, 32).clone()


def signed_index(codes):
    c = codes.to(torch.int32)
    return torch.where((c & 8) != 0, -(c & 7), c & 7)


def rows4(pool, slots):
    vals, sfs = [], []
    for s in slots.tolist():
        if s < 0:
            continue
        page, off = s // 64, s % 64
        vals.append(pool[page, off * 64 : off * 64 + 64])
        sfs.append(pool[page, 4096 + off * 4 : 4096 + off * 4 + 4])
    return torch.stack(vals), torch.stack(sfs)


def compare(tag, got_v, got_sf, ref_v, ref_sf, record_only=False):
    gc = torch.stack([got_v & 0xF, got_v >> 4], dim=-1).reshape(got_v.shape[0], 128)
    rc = torch.stack([ref_v & 0xF, ref_v >> 4], dim=-1).reshape(ref_v.shape[0], 128)
    dcode = (signed_index(gc) - signed_index(rc)).abs()
    code_frac = float((gc != rc).float().mean())
    dsf = (got_sf.to(torch.int32) - ref_sf.to(torch.int32)).abs()
    sf_frac = float((got_sf != ref_sf).float().mean())
    ok = code_frac <= 0.005 and int(dcode.max()) <= 1 and sf_frac <= 0.01 and int(dsf.max()) <= 1
    kw = dict(code_diff_frac=code_frac, code_max_step=int(dcode.max()), sf_diff_frac=sf_frac, sf_max_diff=int(dsf.max()))
    if record_only:
        emit(kind="report", name=tag, within_lines=ok, **kw)
    else:
        check(tag, ok, **kw)


def main():
    try:
        H.load_plugin(MUTATIONS)
        from flashinfer.norm import layernorm
        from sglang.kernels.ops.quantization.fp4_indexer_quant import quantize_fp4_indexer_tensor as qref

        torch.zeros(1, device="cuda")
    except Exception as ex:  # noqa: BLE001
        emit(kind="setup_error", err=repr(ex)[:500])
        raise
    for rows in (1, 5, 8, 12, 16):
        for seed in (1, 2):
            tag = f"b{rows}.s{seed}"
            try:
                w = index_weights(700 + 10 * seed + rows)
                g = H.generator(61 * seed + rows)
                projected = torch.randn((rows, PITCH), generator=g, device="cuda").to(torch.bfloat16)
                key = 0.5 + 2.0 * torch.randn((rows, 128), generator=g, device="cuda")
                projected[:, K_COL : K_COL + 128] = key.to(torch.bfloat16)
                slots = torch.randperm(PAGES * 64, generator=g, device="cuda")[:rows].to(torch.int64).contiguous()
                if rows > 1:
                    slots[rows // 2] = -1
                norm = key_norm(9 * seed + rows)
                pool8 = torch.full((PAGES, 64 * 132), 0xA5, dtype=torch.uint8, device="cuda")
                pool4 = torch.full((PAGES, 64 * 68), 0xA5, dtype=torch.uint8, device="cuda")
                clean4 = pool4.clone()
                c8, w8 = run_qf(w, projected, slots, pool8, norm)
                c4, w4 = run_qf(w, projected, slots, pool4, norm)
                check(f"F1_query_side_bitwise.{tag}", torch.equal(c4, c8) and torch.equal(w4.view(torch.int32), w8.view(torch.int32)),
                      codes_diff=int((c4 != c8).sum()), weights_diff=int((w4 != w8).sum()))
                owned = torch.zeros_like(pool4, dtype=torch.bool)
                for s in slots.tolist():
                    if s >= 0:
                        page, off = s // 64, s % 64
                        owned[page, off * 64 : off * 64 + 64] = True
                        owned[page, 4096 + off * 4 : 4096 + off * 4 + 4] = True
                check(f"F2_untouched_outside_rows.{tag}", torch.equal(pool4[~owned], clean4[~owned]),
                      written_outside=int((pool4[~owned] != clean4[~owned]).sum()))
                gamma, beta, eps = norm
                keep = slots >= 0
                kst = layernorm(projected[:, K_COL : K_COL + 128], gamma, beta, eps)[keep]
                rv, rsf = qref(kst.contiguous())
                gv, gsf = rows4(pool4, slots)
                compare(f"F3_vs_stock_writer_bytes.{tag}", gv, gsf, rv.view(torch.uint8).reshape(-1, 64),
                        rsf.contiguous().view(torch.uint8).reshape(-1, 4))
                x = projected[:, K_COL : K_COL + 128].double()
                mean = x.mean(-1, keepdim=True)
                var = ((x - mean) ** 2).mean(-1, keepdim=True)
                k64 = ((x - mean) / torch.sqrt(var + eps) * gamma.double() + beta.double()).to(torch.bfloat16)[keep]
                fv, fsf = qref(k64.contiguous())
                compare(f"fp64ln_control.{tag}", gv, gsf, fv.view(torch.uint8).reshape(-1, 64), fsf.contiguous().view(torch.uint8).reshape(-1, 4),
                        record_only=True)
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
