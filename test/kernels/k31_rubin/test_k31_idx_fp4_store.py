#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MXFP4 index-K store CUDA kernel (dsa/idx_fp4_store.cuh through idx_fp4.store_index_k, K31MK_DSA_IDX_FP4_STORE=cuda) vs the Triton writer
(idx_fp4._store_kernel, impl='triton': the host prefill writer's row math). .

Bundle-entry correctness of a BITWISE lever (fixed before the first run). Cells: rows 1 / 8 / 40 / 96 / 128 / 1000 / 65,536, seeds
1 / 2; keys BF16 N(0, 1) scaled per row by 2^U(-30, 30) (exponents near both clamps), with planted rows: all zeros, one 32-group all zeros, -0.0
entries, values exactly on e2m1 thresholds x a power of two, and a row of 448s; distinct random slots over a pool with 25 % spare pages, every
7th slot negative. Pools start 0xA5.
  S1 after the CUDA store the WHOLE pool is byte-identical to the pool after the Triton store on the same inputs (stored rows, skipped slots and
     untouched bytes alike).
  S2 the same inside a CUDA graph, with the keys produced by a preceding kernel on the same stream (the store's PDL wait must see them): a graph of
     [key.copy_(src); store] replayed after src changes gives the Triton bytes of the NEW src.
  S3 dispatch: K31MK_DSA_IDX_FP4_STORE default = cuda; FP32 keys, a strided key view and int32 slots take the Triton path (same bytes as triton).
Mutants (each must FAIL; K31_MUTATION=<name>): st_group (row amax instead of the 32-group amax), st_nib (nibble order of a pair swapped),
  st_sfpos (exponent bytes in reverse group order), st_div (amax / 5), st_skip (a negative slot -k is stored at slot k).
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

_CUH = "kimi_k31/dsa/idx_fp4_store.cuh"
MUTATIONS = {
    "st_group": (_CUH, "for (int d = 4; d; d >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));  // the lane's 32-group",
                 "for (int d = 16; d; d >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));  // the lane's 32-group"),
    "st_nib": (_CUH, "packed |= e2m1_code(div_full(v[j], scale)) << (4 * j);", "packed |= e2m1_code(div_full(v[j], scale)) << (4 * (j ^ 1));"),
    "st_sfpos": (_CUH, "page[64 * 64 + off * 4 + (lane >> 3)] = static_cast<uint8_t>(ex);", "page[64 * 64 + off * 4 + (3 - (lane >> 3))] = static_cast<uint8_t>(ex);"),
    "st_div": (_CUH, "const float sf = fmaxf(div_full(amax, 6.0f), 1.0e-4f);", "const float sf = fmaxf(div_full(amax, 5.0f), 1.0e-4f);"),
    "st_skip": (_CUH, "const int64_t slot = p.slots[row];", "const int64_t slot = p.slots[row] < 0 ? -p.slots[row] : p.slots[row];"),
}
OKS = []


def emit(**kw):
    print(json.dumps(kw), flush=True)


def check(name, ok, **kw):
    emit(kind="check", name=name, ok=bool(ok), **kw)
    OKS.append(bool(ok))


def make(rows, seed):
    g = H.generator(1000 * seed + rows)
    keys = torch.randn((rows, 128), generator=g, device="cuda")
    keys *= torch.exp2(torch.randint(-30, 31, (rows, 1), generator=g, device="cuda").float())
    if rows >= 8:
        keys[0] = 0
        keys[1, 32:64] = 0
        keys[2, ::3] = -0.0
        thr = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0], device="cuda")
        keys[3] = (thr.repeat(16) * torch.where(torch.arange(128, device="cuda") % 2 == 0, 1.0, -1.0)) * 4.0
        keys[4] = 448.0
    keys = keys.to(torch.bfloat16).contiguous()
    pages = max(2, (rows * 5 + 255) // 256 + 1)
    slots = torch.randperm(pages * 64, generator=g, device="cuda")[:rows].to(torch.int64)
    slots[::7] = -1 - (slots[::7] % 5)
    return keys, slots.contiguous(), pages


def main():
    try:
        H.load_plugin(MUTATIONS)
        from sgl_nvidia_mega.kimi_k31.dsa import idx_fp4

        torch.zeros(1, device="cuda")
    except Exception as ex:  # noqa: BLE001
        emit(kind="setup_error", err=repr(ex)[:500])
        raise
    check("S3_default_is_cuda", idx_fp4.STORE_IMPL == "cuda", impl=idx_fp4.STORE_IMPL)
    for seed in (1, 2):
        for rows in (1, 8, 40, 96, 128, 1000, 65536):
            tag = f"r{rows}.s{seed}"
            try:
                keys, slots, pages = make(rows, seed)
                ref = torch.full((pages, idx_fp4.PAGE_B), 0xA5, dtype=torch.uint8, device="cuda")
                got = ref.clone()
                idx_fp4.store_index_k(keys, ref, slots, impl="triton")
                idx_fp4.store_index_k(keys, got, slots, impl="cuda")
                torch.cuda.synchronize()
                check(f"S1_pool_bitwise.{tag}", torch.equal(got, ref), bytes_differ=int((got != ref).sum()), stored=int((slots >= 0).sum()))
                if rows in (8, 96):
                    src = torch.empty_like(keys)
                    k2 = torch.empty_like(keys)
                    gpool = torch.full_like(ref, 0xA5)
                    gph = torch.cuda.CUDAGraph()
                    s = torch.cuda.Stream()
                    s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s):
                        k2.copy_(src)
                        idx_fp4.store_index_k(k2, gpool, slots, impl="cuda")
                    torch.cuda.current_stream().wait_stream(s)
                    torch.cuda.synchronize()
                    with torch.cuda.graph(gph):
                        k2.copy_(src)
                        idx_fp4.store_index_k(k2, gpool, slots, impl="cuda")
                    new = (keys.float() * -3.0).to(torch.bfloat16)
                    src.copy_(new)
                    gpool.fill_(0xA5)
                    gph.replay()
                    ref2 = torch.full_like(ref, 0xA5)
                    idx_fp4.store_index_k(new, ref2, slots, impl="triton")
                    torch.cuda.synchronize()
                    check(f"S2_graph_pdl_bitwise.{tag}", torch.equal(gpool, ref2), bytes_differ=int((gpool != ref2).sum()))
                if rows == 40:
                    variants = (("fp32_key", keys.float(), slots), ("strided_key", torch.cat([keys, keys], 1)[:, :128], slots),
                                ("int32_slots", keys, slots.to(torch.int32)))
                    for name, k, sl in variants:
                        a = torch.full_like(ref, 0xA5)
                        idx_fp4.store_index_k(k, a, sl)
                        torch.cuda.synchronize()
                        check(f"S3_fallback_{name}.{tag}", torch.equal(a, ref), bytes_differ=int((a != ref).sum()))
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
