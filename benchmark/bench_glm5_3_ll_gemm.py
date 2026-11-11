# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Crossover bench of ``ll.gemm``'s plan on one GB300 GPU: each preferred role's GEMM
between model-like neighbours, SGLang's kernel ("stock") against the plan's, per row
count. Re-checks on the running image that every planned range still wins (the plan's
ranges were measured on the 684fea8a image, notes/branches/04-pdl-splitk-gemms.md 11).

Contexts, each captured in a CUDA graph and replayed with PDL:
- chain: a PDL RMSNorm (triggers at its exit) produces the input, then the GEMM, then
  a PDL RMSNorm of its output (a copy for the FP32 router), as the hop after it;
- pair (wk_weights_proj): the indexer's wq_b (q-LoRA 2048 -> 4096, on SGLang's
  kernel: TGV up to 64 rows, cuBLAS above) on a second stream beside wk on the main
  stream, joined by an event.
Weights rotate through >= 384 MB of copies, so every GEMM streams from HBM; inputs
stay L2-hot. Stock = what SGLang runs for the module: TGV where
``use_cutedsl_bf16_gemm`` picks it, else cuBLAS; the router (served from 17 rows) cuBLAS
FP32.

    python benchmark/bench_glm5_3_ll_gemm.py [--rows 1,6,24] [--out results.csv]
    python benchmark/bench_glm5_3_ll_gemm.py --ab-rounds 4   # vendored vs fi-1 split-K
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys

import torch
import torch.nn.functional as F

from sgl_nvidia_mega.glm5_3.ll import gemm
from sgl_nvidia_mega.glm5_3.ll import gemm_plan as plan

ROWS = (1, 3, 6, 12, 16, 18, 24, 30, 36, 48, 60, 64, 72, 96)
ROTATE_BYTES = 384 << 20


def graph_us(launches, *, replays=20, reps=7) -> float:
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for fn in launches:
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for fn in launches:
            fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    samples = []
    for _ in range(reps):
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / replays)
    del graph
    return statistics.median(samples)


class _Mod(torch.nn.Module):
    def __init__(self, w, router):
        super().__init__()
        self.weight = torch.nn.Parameter(w, requires_grad=False)
        self.bias = None
        if router:
            self.tiny_router_gemm_max_tokens = 16


def stock_fn(role):
    """SGLang's kernel for the role's module, as its forward picks it."""
    from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import (
        cutedsl_bf16_gemm,
        use_cutedsl_bf16_gemm,
    )

    if role == "router":
        # above the tiny GEMM's 16 rows (the plan's rows): cuBLAS BF16 x BF16 -> FP32
        return lambda x, w: torch.mm(x, w.t(), out_dtype=torch.float32)

    def linear(x, w):
        if use_cutedsl_bf16_gemm(x.shape[0], *w.shape):
            return cutedsl_bf16_gemm(x, w)
        return F.linear(x, w)

    return linear


def _weights(e: plan.Entry):
    n, k = e.shape
    copies = max(2, min(128, -(-ROTATE_BYTES // (n * k * 2))))
    w0 = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / math.sqrt(k)
    return [
        _Mod(w0 if i == 0 else w0.clone(), e.role == "router") for i in range(copies)
    ]


def _context(e: plan.Entry, m: int, mods):
    """launches(fn) for the role's context at m rows; fn(module, x) runs the GEMM."""
    import flashinfer

    n, k = e.shape
    wq_b_gemm = stock_fn("q_b")
    pair = e.role == "wk_weights_proj"
    wq_b = torch.randn(4096, 2048, device="cuda", dtype=torch.bfloat16) / math.sqrt(
        2048
    )
    wq_bs = [wq_b] + [wq_b.clone() for _ in range(max(1, len(mods) // 8) - 1)]
    h = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(m, 2048, device="cuda", dtype=torch.bfloat16)
    g_in = torch.ones(k, device="cuda", dtype=torch.bfloat16)
    g_out = torch.ones(n, device="cuda", dtype=torch.bfloat16)
    x = torch.empty(m, k, device="cuda", dtype=torch.bfloat16)
    z = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    alt = torch.cuda.Stream()

    def launches(fn):
        out = []
        for i, mod in enumerate(mods):
            out.append(
                lambda: flashinfer.norm.rmsnorm(h, g_in, 1e-6, out=x, enable_pdl=True)
            )
            if pair:

                def step(mod=mod, i=i):
                    fork = torch.cuda.Event()
                    fork.record()
                    with torch.cuda.stream(alt):
                        alt.wait_event(fork)
                        wq_b_gemm(q, wq_bs[i % len(wq_bs)])
                        join = torch.cuda.Event()
                        join.record()
                    fn(mod, x)
                    torch.cuda.current_stream().wait_event(join)

            else:

                def step(mod=mod):
                    y = fn(mod, x)
                    if y.dtype == torch.bfloat16:
                        flashinfer.norm.rmsnorm(y, g_out, 1e-6, out=z, enable_pdl=True)
                    else:
                        z.copy_(y)

            out.append(step)
        return out

    return launches


def _time_source(e, m, mods, launches, source) -> float:
    os.environ[gemm.SPLITK_ENV] = source
    try:
        gemm.reset()
        gemm.build(e.role, mods[0], [m])
        return graph_us(launches(lambda mod, xx: gemm.linear(e.role, mod, xx))) / len(
            mods
        )
    finally:
        os.environ.pop(gemm.SPLITK_ENV, None)
        gemm.reset()


def bench_entry(e: plan.Entry, rows, writer, sources):
    n, k = e.shape
    mods = _weights(e)
    stock = stock_fn(e.role)
    for m in rows:
        r = plan.range_at(e, m)
        if r is None or m < gemm._min_rows(e.role, mods[0]):
            continue
        launches = _context(e, m, mods)
        row = dict(
            role=e.role, n=n, k=k, m=m, kernel=plan.kernel_name(r), copies=len(mods)
        )
        row["stock_first_us"] = graph_us(
            launches(lambda mod, xx: stock(xx, mod.weight))
        ) / len(mods)
        for source in sources:
            row[f"{source}_us"] = _time_source(e, m, mods, launches, source)
            if r.kernel == plan.TGV:
                break  # the source only changes split-K
        row["stock_last_us"] = graph_us(
            launches(lambda mod, xx: stock(xx, mod.weight))
        ) / len(mods)
        row["stock_us"] = min(row["stock_first_us"], row["stock_last_us"])
        row["noise_us"] = abs(row["stock_first_us"] - row["stock_last_us"])
        best = min(
            v
            for key, v in row.items()
            if key.endswith("_us") and key.split("_")[0] in sources
        )
        row["gain_us"] = row["stock_us"] - best
        writer.writerow(row)
        print(
            f"{e.role:16s} (N={n:5d}, K={k:5d}) M={m:3d} {row['kernel']:14s} stock {row['stock_us']:7.2f} "
            + " ".join(
                f"{s} {row[f'{s}_us']:7.2f}" for s in sources if f"{s}_us" in row
            )
            + f"  gain {row['gain_us']:+6.2f} (noise {row['noise_us']:.2f})",
            flush=True,
        )
    del mods
    torch.cuda.empty_cache()


# The split-K rows the synchronization-fix A/B times (router: chain, wk: pair context)
AB_ROWS = {"router": (17, 24, 48, 96), "wk_weights_proj": (1, 6, 16, 64)}


def ab_vendored_vs_flashinfer(rounds: int) -> None:
    """The vendored split-K (synchronization fixes) against FlashInfer's (fi-1; fixed
    when ``SplitKTactic.SYNC_FIXED``), same tactic and context, ABBA ``rounds`` times."""
    fi_gemm = gemm.flashinfer_tactic_api()
    fi_fixed = gemm.flashinfer_splitk_sync_fixed(fi_gemm)
    print(f"AB FlashInfer {gemm.FI1_SYNC_MARKER}={fi_fixed}", flush=True)
    for e in plan.PLAN:
        if e.role not in AB_ROWS or not e.preferred:
            continue
        mods = _weights(e)
        for m in AB_ROWS[e.role]:
            launches = _context(e, m, mods)
            t = {"vendored": [], "flashinfer": []}
            for i in range(rounds):
                order = (
                    ("vendored", "flashinfer")
                    if i % 2 == 0
                    else ("flashinfer", "vendored")
                )
                for source in order + order[::-1]:
                    t[source].append(_time_source(e, m, mods, launches, source))
            ven, fi = (
                statistics.median(t["vendored"]),
                statistics.median(t["flashinfer"]),
            )
            spread = max(max(v) - min(v) for v in t.values())
            print(
                f"AB {e.role:16s} M={m:3d} {plan.kernel_name(plan.range_at(e, m)):14s} "
                f"vendored {ven:7.3f} flashinfer {fi:7.3f} us  delta {ven - fi:+.3f} "
                f"({100 * (ven - fi) / fi:+.2f} %, spread {spread:.3f}, n={len(t['vendored'])})",
                flush=True,
            )
        del mods
        torch.cuda.empty_cache()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default=",".join(map(str, ROWS)))
    p.add_argument("--roles", default="")
    p.add_argument("--out", default="")
    p.add_argument(
        "--ab-rounds",
        type=int,
        default=0,
        help="only the vendored vs FlashInfer split-K A/B",
    )
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.ab_rounds:
        if gemm.flashinfer_tactic_api() is None:
            print("A/B needs FlashInfer's mm_bf16(..., tactic=) (fi-1)")
            return 1
        with torch.no_grad():
            ab_vendored_vs_flashinfer(args.ab_rounds)
        return 0
    rows = [int(v) for v in args.rows.split(",")]
    sources = ["vendored"] + (
        ["flashinfer"] if gemm.flashinfer_tactic_api() is not None else []
    )
    roles = set(filter(None, args.roles.split(",")))
    fields = [
        "role",
        "n",
        "k",
        "m",
        "kernel",
        "copies",
        "stock_first_us",
        "stock_last_us",
        "stock_us",
        "noise_us",
    ]
    fields += [f"{s}_us" for s in sources] + ["gain_us"]
    out = open(args.out, "w", newline="") if args.out else sys.stdout
    writer = csv.DictWriter(
        out if args.out else open(os.devnull, "w"),
        fieldnames=fields,
        extrasaction="ignore",
    )
    writer.writeheader()
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.no_grad():
        for e in plan.PLAN:
            if e.preferred and (not roles or e.role in roles):
                bench_entry(e, rows, writer, sources)
    if args.out:
        out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
