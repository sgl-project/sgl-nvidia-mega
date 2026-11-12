# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot attn_q_cluster alone on one GPU: the cluster kernel's registers and local memory,
and its time against the chain's pair and the attn_q_norm_gemm launch, each behind a
wqkv_a low-latency GEMM (PDL) over a CUDA graph of 14 layers with distinct weights.

  python benchmark/bench_deepseek_v41_q_cluster.py --rows 1,6,8
"""

import argparse
import statistics

import torch

from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll
from sgl_nvidia_mega.deepseek_v41.sm103.attention import q_cluster, q_norm_gemm
from sgl_nvidia_mega.deepseek_v41.sm103.attention.q_norm_quant import rmsnorm_mxfp8

K, QKV_N, HIDDEN, N, EPS, LAYERS = 1280, 1792, 4096, 4096, 1e-6, 14


def _block_weight(n, k):
    w = (torch.randn(n, k, device="cuda") * 64).clamp(-448, 448).to(torch.float8_e4m3fn)
    e = torch.randint(118, 122, (n // 32, k // 32), device="cuda", dtype=torch.uint8)
    sf = q_cluster.swizzle_scales(e.repeat_interleave(32, 0))
    return (w, sf), q_cluster.pack(w, sf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1,6,8")
    ap.add_argument("--arms", default="pair,qn,qc,qcC")
    ap.add_argument("--no-qkv", action="store_true", help="no wqkv_a in front")
    a = ap.parse_args()
    from cuda.bindings import driver as cu

    for fit in (True, False):
        for et in (True, False):
            k = q_cluster.load_kernel(torch.cuda.current_device(), fit, et)
            regs = cu.cuFuncGetAttribute(
                cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, k.function
            )[1]
            local = cu.cuFuncGetAttribute(
                cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, k.function
            )[1]
            smem = cu.cuFuncGetAttribute(
                cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, k.function
            )[1]
            print(
                f"fit={fit} et={et}: regs={regs} local={local} B static smem={smem} B"
            )
    consumer = mxfp8_ll.WQ_B_TP8
    for rows in map(int, a.rows.split(",")):
        x = mxfp8_ll.quantize(torch.randn(rows, HIDDEN, device="cuda").bfloat16())
        layers = []
        for _ in range(LAYERS):
            wq = torch.randn(QKV_N, HIDDEN, device="cuda").bfloat16() / 64
            wqkv = mxfp8_ll.quantize(wq)
            (wb, sb), packed = _block_weight(N, K)
            nw = (1 + 0.1 * torch.randn(K, device="cuda")).bfloat16()
            layers.append((wqkv[0], wqkv[1].reshape(-1), wb, sb, packed, nw))
        mxfp8_ll.precompile(layers[0][0], [rows])
        mxfp8_ll.precompile(wb, [rows], consumer)
        q_norm_gemm.precompile(wb, [rows], EPS, consumer=consumer)
        static_qkv = torch.randn(rows, QKV_N, device="cuda").bfloat16()

        def run(arm):
            for wq, sq, wb, sb, packed, nw in layers:
                qkv = (
                    static_qkv
                    if a.no_qkv
                    else mxfp8_ll.linear(x[0], x[1].reshape(-1), wq, sq)
                )
                ql = qkv[:, :K]
                if arm == "pair":
                    _, q8, s8 = rmsnorm_mxfp8(ql, nw, EPS, pdl=True)
                    mxfp8_ll.linear(q8, s8, wb, sb, consumer=consumer)
                elif arm == "qn":
                    q_norm_gemm.q_norm_linear(ql, nw, EPS, wb, sb, consumer=consumer)
                else:
                    q_cluster.q_cluster(ql, nw, EPS, packed, early_trigger=arm == "qc")

        graphs = {}
        for arm in a.arms.split(","):
            run(arm)
            torch.cuda.synchronize()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                run(arm)
            graphs[arm] = g
        res = {arm: [] for arm in graphs}
        st, en = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for rep in range(9):
            for arm in graphs if rep % 2 == 0 else list(graphs)[::-1]:
                ts = []
                for _ in range(40):
                    st.record()
                    graphs[arm].replay()
                    en.record()
                    torch.cuda.synchronize()
                    ts.append(st.elapsed_time(en) * 1e3 / LAYERS)
                res[arm].append(statistics.median(ts))
        print(
            f"rows={rows} no_qkv={a.no_qkv} us/layer: "
            + " ".join(f"{k}={statistics.median(v):.2f}" for k, v in res.items())
        )


if __name__ == "__main__":
    main()
