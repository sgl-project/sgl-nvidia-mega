# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot attn_core_lean in isolation: 30 attention launches back to back in one CUDA graph,
muLLM's one-launch core (v41_new pool) against the shipped swap-AB + narrow combine (V4
pool; cluster merge on and off), 8 heads, PDL chain form. Microseconds per layer, median
of repeats. Isolation only: the served step bounds the core differently (neighbours, the
KV-store join), so only the served A/B decides.

    python benchmark/bench_deepseek_v41_core_lean.py --rows 1 6 8
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "test", "kernels"))
from test_deepseek_v41_core_lean import Case  # noqa: E402

from sgl_nvidia_mega.deepseek_v41.sm103.attention import swapab  # noqa: E402


def _graph(fn, layers):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(layers):
            fn()
    return graph


def _time(graph, layers, repeats):
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    times = []
    for _ in range(repeats):
        graph.replay()
        start.record()
        for _ in range(10):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000 / 10 / layers)
    times.sort()
    return times[len(times) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 6, 8])
    ap.add_argument("--layers", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=21)
    ap.add_argument("--extra-tokens", type=int, default=8192 // 4)
    args = ap.parse_args()
    for b in args.rows:
        for window_only in (False, True):
            c = Case(b, 1, window_only=window_only, extra_tokens=args.extra_tokens)
            tail = (c.swa_v4, c.ids, c.lengths, c.sink)
            if not window_only:
                tail += (c.extra_v4, c.ei, c.el)

            def shipped(cm):
                return lambda: swapab.swapab_attention(
                    c.q,
                    *tail,
                    max_rows=24,
                    pdl=True,
                    narrow_combine=True,
                    combine_entry_trigger=True,
                    q_rope=(c.freqs, c.positions),
                    kv_before_wait=True,
                    cluster_merge=cm,
                )

            arms = {
                "swapab+combine": shipped(False),
                "swapab cluster merge": shipped(True),
                "core_lean": lambda: c.lean(early=False),
                "core_lean early idx": lambda: c.lean(early=True),
            }
            for name, fn in arms.items():
                us = _time(_graph(fn, args.layers), args.layers, args.repeats)
                kind = "window-only" if window_only else "sparse"
                print(f"rows={b} {kind:11s} {name:22s} {us:7.2f} us/layer", flush=True)


if __name__ == "__main__":
    main()
