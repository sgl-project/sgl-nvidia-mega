#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GPU test of dsa/dg_logits_fp4.py (K31MK_DSA_IDX_FP4_VENDOR: the MXFP4 instantiation of the vendored DeepGEMM paged logits, PDL launch).

, bundle-entry correctness for a BITWISE lever (fixed before the first run):
  V1 at every cell, the vendored kernel's logits are BITWISE equal to deep_gemm.fp8_fp4_paged_mqa_logits (same instantiation, same inputs)
     at every valid position t < ctx[b, j] of every row; cells: Breq x Q8 at 1 x 131072, 4 x 65536, 12 x 131072 (MID rows), 16 x 8192,
     3 ragged requests (4097 / 70 / 2049) on pages written by idx_fp4.store_index_k, and one cell on a strided pool (page stride 8448 B,
     MXFP4 rows in the first 4352 B) for the page-stride argument.
  V2 the query / gates come from idx_fp4.quantize_query (the served producer).
Report-only (EXPLORATORY): isolated per-call time of both launches (CUDA graph of 24 calls, ABBA x 3) at 12 x 131072 and 16 x 131072;
  the PDL overlap itself only shows in the served chain (verify step), not here.
Mutants (run by the runner on copies of dg_logits_fp4.py; each must FAIL V1): sfkv (index-K scale map base + 16 B: scales of the wrong
  tokens), noswz (index-K map without the 64-byte swizzle the kernel's smem layout expects; same bytes copied, wrong placement).
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
    fp4 = importlib.import_module("sgl_nvidia_mega.kimi_k31.dsa.idx_fp4")
    ven = importlib.import_module("sgl_nvidia_mega.kimi_k31.dsa.dg_logits_fp4")
    return fp4, ven


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--host-dir", default="")
    ap.add_argument("--no-timing", action="store_true")
    args = ap.parse_args()
    fp4, ven = load(os.path.abspath(args.tree), args.host_dir)
    import deep_gemm

    dev = torch.device("cuda")
    torch.zeros(1, device=dev)  # create the primary context: build() loads the cubin through the driver API (r1 failed here: 201)
    torch.cuda.synchronize()
    nsm = torch.cuda.get_device_properties(0).multi_processor_count
    g = torch.Generator(device="cuda").manual_seed(1107)
    ven.build(0, 8)
    oks = []
    N = 8
    cells = [("1x131072", (131072,), False, False), ("4x65536", (65536,) * 4, False, False), ("12x131072", (131072,) * 12, False, False),
             ("16x8192", (8192,) * 16, False, False), ("ragged3_writer", (4097, 70, 2049), True, False), ("strided3", (3000, 64, 1025), True, True)]
    timing = {}
    for name, ctxs, use_writer, strided in cells:
        B = len(ctxs)
        wpages = -(-max(ctxs) // 64)
        npool = B * wpages + 3
        table = torch.randperm(npool, generator=g, device=dev)[: B * wpages].view(B, wpages).to(torch.int32)
        page_b = 8448 if strided else fp4.PAGE_B
        raw = torch.randint(0, 120, (npool, page_b), generator=g, device=dev, dtype=torch.uint8)
        if use_writer:
            keys = torch.randn(B * wpages * 64, 128, generator=g, device=dev).to(torch.bfloat16)
            slot = (table.to(torch.int64)[:, :, None] * 64 + torch.arange(64, device=dev)[None, None, :]).reshape(-1)
            if strided:  # write through a dense [npool, 4352] view of the first 4352 B of each page
                dense = torch.empty((npool, fp4.PAGE_B), dtype=torch.uint8, device=dev)
                fp4.store_index_k(keys, dense, slot)
                raw[:, : fp4.PAGE_B] = dense
            else:
                fp4.store_index_k(keys, raw, slot)
        cache = raw[:, : fp4.PAGE_B].as_strided((npool, 64, 1, 68), (page_b, 68, 68, 1))
        q = (torch.randn(B * N, 32, 128, generator=g, device=dev) * 2).to(torch.bfloat16)
        gate = torch.randn(B * N, 32, generator=g, device=dev).to(torch.bfloat16)
        fq, w = fp4.quantize_query(q, gate, fp4.QueryBuffers(B * N, dev))
        ctx2 = torch.tensor([[c - (N - 1 - j) for j in range(N)] for c in ctxs], dtype=torch.int32, device=dev)
        sched = deep_gemm.get_paged_mqa_logits_metadata(ctx2, 64, nsm)
        maxlen = wpages * 64
        qv = (fq.q4.view(B, N, 32, 64), fq.sf.view(B, N, 32))
        w2 = w.view(B * N, 32)

        def f_dg():
            return deep_gemm.fp8_fp4_paged_mqa_logits(qv, cache, w2, ctx2, table, sched, maxlen, False, logits_dtype=torch.float32)

        def f_ven():
            return ven.paged_mqa_logits(qv, cache, w2, ctx2, table, sched, maxlen)

        a = f_dg()
        b = f_ven()
        torch.cuda.synchronize()
        valid = torch.arange(maxlen, device=dev)[None, :] < ctx2.view(-1, 1)
        same = torch.equal(a.contiguous()[valid].view(torch.int32), b.contiguous()[valid].view(torch.int32))
        ndiff = int((a.contiguous()[valid].view(torch.int32) != b.contiguous()[valid].view(torch.int32)).sum())
        oks.append(bool(same))
        emit(kind="check", name=f"V1_bitwise_{name}", ok=bool(same), differing=ndiff, valid=int(valid.sum()),
             finite=bool(torch.isfinite(b.contiguous()[valid]).all().item()))
        if not args.no_timing and name in ("12x131072",):
            t = {"dg": [], "ven": []}
            for r in range(3):
                for k in (("dg", "ven") if r % 2 == 0 else ("ven", "dg")):
                    t[k].append(time_graph([f_dg if k == "dg" else f_ven] * 24))
            timing[name] = {k: round(statistics.median(v), 3) for k, v in t.items()}
            emit(kind="report", name=f"time_us_{name}", **timing[name])
        del raw, cache
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
