# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""One DeepSeek-V4.1 routed MoE half at batch 1 (6 verify rows, top-6 of 384 experts):
route + GEMM1 + GEMM2 on cold weights, optionally beside a shared-expert-sized chain on
a second stream, replayed from a CUDA graph of 39 halves.

  python benchmark/bench_deepseek_v41_routed_moe.py --distinct 16 24 32 \
      --variants clc pipe_clc pool_clc --side 0 1 --repeats 5 [--check]

Reports us per half (graph time / halves, minus a graph of the spacers alone), variants
interleaved in alternating order per repeat. --check compares every variant's GEMM2 rows
bit for bit against the first.

--local-experts 192 --offset 0|192: one EP2 rank's window of the 384 experts (TP8's
MoE TP4 x EP2); the routing stays over all 384, so a rank computes ~D/2 of them.

Variants, each under cluster launch control: clc (the shipped pair), pipe_clc (GEMM2 on
the gemm2_pipe slot's build), pool_clc (that build at the gemm2_reg_pool registers);
preroute+<variant> takes the route kernel off the measured chain.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics

import torch

from sgl_nvidia_mega.common import jit
from sgl_nvidia_mega.deepseek_v41.sm103.moe import routed_expert_gemm as reg

ROWS, TOP_K, EXPERTS, LAYERS = 6, 6, 384, 39  # EXPERTS: the router's


SOURCES = {
    "gemm1": "gemm1.cu",
    "gemm2": "gemm2.cu",
    "gemm2_pipe": "gemm2_pipe.cu",
}


def pipe_sources() -> dict:
    """The gemm2_reg_pool build of the pipeline GEMM2."""
    g2 = (reg.SOURCE_DIR / "gemm2_pipe.cu").read_text()
    return {"gemm2_pipe_pool": reg.register_pool(g2, reg.GEMM2_POOL_REGISTERS)}


def load_kernels(device_index: int, extra: dict | None = None) -> dict:
    """Every GEMM source, plus the route kernel."""
    major, minor = torch.cuda.get_device_capability(device_index)
    arch = f"sm_{major}{minor}a"
    facts = json.loads((reg.SOURCE_DIR / "launch.json").read_text())
    kernels = {"route": reg.load_kernels(device_index)["route"]}
    sources = {k: (reg.SOURCE_DIR / f).read_text() for k, f in SOURCES.items()}
    sources.update(extra or {})
    for name, text in sources.items():
        fact = facts["gemm2_pipe" if name.startswith("gemm2_pipe") else name[:5]]
        kernels[name] = jit.load(
            reg.compile_source(text, name, arch),
            fact["symbol"],
            fact["block"],
            int(fact["shared_memory_bytes"]),
        )
    return kernels


# ------------------------------------------------------------------------------- layers


def random_layer(generator, device, experts=EXPERTS):
    e, h, p = experts, reg.HIDDEN, reg.PADDED

    def rand(shape, lo=0, hi=256):
        return torch.randint(lo, hi, shape, dtype=torch.uint8, generator=generator).to(
            device
        )

    w13 = rand((e, 2 * p, h // 2))
    w13_s = rand((e, 2 * p, h // 32), 118, 126)
    w2 = rand((e, h, p // 2))
    w2_s = rand((e, h, p // 32), 118, 126)
    clamp = torch.full((e,), 10.0, dtype=torch.float32, device=device)
    tensors = (w13, w13_s, w2, w2_s, clamp)
    maps1 = (reg._weight_map(w13, 2 * p, h, 2), reg._scale_map(w13_s, 4))
    maps2 = (reg._weight_map(w2, h, p, 2), reg._scale_map(w2_s, 1))
    return tensors, maps1, maps2


def routing(distinct: int, generator) -> torch.Tensor:
    """6 x 6 ids over exactly ``distinct`` experts (6 <= distinct <= 36)."""
    pool = torch.randperm(EXPERTS, generator=generator)[:distinct]
    while True:
        rows = [
            pool[torch.randperm(distinct, generator=generator)[:TOP_K]]
            for _ in range(ROWS)
        ]
        ids = torch.stack(rows)
        if ids.unique().numel() == distinct:
            return ids.to(torch.int32)
        # force coverage: put the missing experts into random free slots
        missing = [x for x in pool.tolist() if x not in set(ids.flatten().tolist())]
        flat = ids.flatten().tolist()
        counts = {x: flat.count(x) for x in flat}
        for x in missing:
            for i in torch.randperm(len(flat), generator=generator).tolist():
                r = i // TOP_K
                if counts[flat[i]] > 1 and x not in flat[r * TOP_K : (r + 1) * TOP_K]:
                    counts[flat[i]] -= 1
                    flat[i] = x
                    counts[x] = 1
                    break
        ids = torch.tensor(flat).view(ROWS, TOP_K)
        if ids.unique().numel() == distinct:
            return ids.to(torch.int32)


# ------------------------------------------------------------------------------ variants


def grids(variant: str, ctas: int, sms: int):
    """(gemm1 name, grid1), (gemm2 name, grid2) of a variant."""
    gemm2 = {"clc": "gemm2", "pipe_clc": "gemm2_pipe", "pool_clc": "gemm2_pipe_pool"}
    if variant not in gemm2:
        raise ValueError(variant)
    return ("gemm1", (reg.GEMM1_UNITS, ctas, 1)), (
        gemm2[variant],
        (reg.GEMM2_UNITS, ctas, 1),
    )


class Bench:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        g = torch.Generator().manual_seed(7)
        self.local = args.local_experts
        self.offset = args.offset
        self.layers = [random_layer(g, self.device, self.local) for _ in range(2)]
        self.kernels = load_kernels(self.device.index, pipe_sources())
        from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

        x = torch.randn(ROWS, reg.HIDDEN, generator=g).to(torch.bfloat16).cuda()
        self.x_bf16 = x
        x_q, x_s = flashinfer_mxfp8_quantize(x, False, alignment=reg.HIDDEN)
        self.x_q, self.x_s = x_q, x_s.view(torch.uint8).reshape(ROWS, -1)
        self.ctas = reg.max_ctas(ROWS, TOP_K, self.local)
        self.words = reg.cta_words(self.ctas)
        self.barrier = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.flag = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.side = SideChain(self.device, g) if 1 in args.side else None
        self.side_stream = torch.cuda.Stream(self.device)

    def buffers(self):
        padded = reg.TILE * self.ctas
        return dict(
            permuted=torch.empty((ROWS, TOP_K), dtype=torch.int32, device=self.device),
            record=torch.empty(
                reg.route_words(self.ctas), dtype=torch.int32, device=self.device
            ),
            h=torch.empty((padded, reg.PADDED), dtype=torch.uint8, device=self.device),
            hs=torch.empty(
                (padded + 127) // 128 * 128 * reg.PADDED // 32,
                dtype=torch.uint8,
                device=self.device,
            ),
            y=torch.empty(
                (ROWS * TOP_K * reg.TILE, reg.HIDDEN),
                dtype=torch.bfloat16,
                device=self.device,
            ),
        )

    def route(self, ids, buf):
        reg._launch(
            self.kernels["route"],
            (1, 1, 1),
            [
                reg._pointer(ids),
                reg._pointer(buf["permuted"]),
                reg._pointer(buf["record"]),
                reg._pointer(self.barrier),
                ctypes.c_int32(ROWS),
                ctypes.c_int32(TOP_K),
                ctypes.c_int32(self.local),
                ctypes.c_int32(self.offset),
            ],
            torch.cuda.current_stream().cuda_stream,
        )

    def half(self, layer, ids, buf, variant, x_q=None, x_s=None):
        stream = torch.cuda.current_stream().cuda_stream
        _, maps1, maps2 = layer
        if variant.startswith("preroute+"):
            variant = variant[len("preroute+") :]
        else:
            self.route(ids, buf)
        (n1, g1), (n2, g2) = grids(variant, self.ctas, self.sms)
        words = ctypes.c_int32(self.words)
        reg._launch(
            self.kernels[n1],
            g1,
            [
                *maps1,
                reg._pointer(self.x_q if x_q is None else x_q),
                reg._pointer(self.x_s if x_s is None else x_s),
                reg._pointer(buf["record"]),
                reg._pointer(layer[0][4]),
                reg._pointer(buf["h"]),
                reg._pointer(buf["hs"]),
                words,
            ],
            stream,
        )
        reg._launch(
            self.kernels[n2],
            g2,
            [
                *maps2,
                reg._activation_map(buf["h"])
                if n2.startswith("gemm2_pipe")
                else reg._pointer(buf["h"]),
                reg._pointer(buf["hs"]),
                reg._pointer(buf["record"]),
                reg._pointer(buf["y"]),
                words,
            ],
            stream,
        )

    def graph(self, distinct, variant, side, spacers_only=False):
        g = torch.Generator().manual_seed(1000 + distinct)
        ids = [routing(distinct, g).to(self.device) for _ in range(LAYERS)]
        bufs = [self.buffers() for _ in range(LAYERS)]

        def body():
            main = torch.cuda.current_stream()
            for i in range(LAYERS):
                if variant.startswith("preroute+") and not spacers_only:
                    self.route(ids[i], bufs[i])  # off the measured chain
                self.flag.add_(1)  # spacer: no PDL, the next half starts after this
                if spacers_only:
                    continue
                if side:
                    self.side_stream.wait_stream(main)
                    with torch.cuda.stream(self.side_stream):
                        self.side()
                self.half(self.layers[i % 2], ids[i], bufs[i], variant)
                if side:
                    main.wait_stream(self.side_stream)

        body()  # warm-up and compile outside capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        torch.cuda.synchronize()
        return graph, bufs, ids

    def time(self, graph, replays):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        graph.replay()
        torch.cuda.synchronize()
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1e3 / replays / LAYERS


class SideChain:
    """A shared-expert-sized chain (row-tile MXFP8 quant, 1152 x 5120 MXFP8 gate_up on the
    low-latency kernel, clamped SwiGLU, MXFP8 quant, 5120 x 640 down) on its own weights.
    """

    def __init__(self, device, g):
        from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll

        self.mxfp8_ll = mxfp8_ll

        def weight(n, k):
            w = (
                (torch.randn(n, k, generator=g) * 0.05)
                .to(device)
                .to(torch.float8_e4m3fn)
            )
            s = torch.randint(
                120, 128, (n * k // 32,), dtype=torch.uint8, generator=g
            ).to(device)
            return w, s

        self.gate_up = weight(2 * reg.SLICE, reg.HIDDEN)
        self.down = weight(reg.HIDDEN, reg.PADDED)
        self.x = (
            torch.randn(ROWS, reg.HIDDEN, generator=g).to(torch.bfloat16).to(device)
        )
        mxfp8_ll.precompile(self.gate_up[0], [ROWS])
        mxfp8_ll.precompile(self.down[0], [ROWS])

    def __call__(self):
        from sgl_nvidia_mega.deepseek_v41.sm103.attention.q_norm_quant import (
            mxfp8_row_tiles,
        )

        x_q, x_s = mxfp8_row_tiles(self.x)
        gu = self.mxfp8_ll.linear(x_q, x_s, *self.gate_up)
        from sgl_nvidia_mega.deepseek_v41 import sglang_ops

        mid = torch.empty(ROWS, reg.SLICE, dtype=torch.bfloat16, device=gu.device)
        sglang_ops.silu_and_mul_clamp(gu, mid, 10.0)
        h = torch.nn.functional.pad(mid, (0, reg.PADDED - reg.SLICE))
        h_q, h_s = self.mxfp8_ll.quantize(h)
        return self.mxfp8_ll.linear(h_q, h_s, *self.down)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--distinct", type=int, nargs="+", default=[16, 20, 24, 28, 32])
    p.add_argument("--variants", nargs="+", default=["clc"])
    p.add_argument("--side", type=int, nargs="+", default=[0, 1])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--replays", type=int, default=20)
    p.add_argument("--bandwidth", action="store_true")
    p.add_argument("--local-experts", type=int, default=EXPERTS)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument(
        "--check", action="store_true", help="bitwise rows vs the first variant"
    )
    args = p.parse_args()
    bench = Bench(args)
    print(f"sms {bench.sms} ctas {bench.ctas}", flush=True)
    if args.bandwidth:
        big = torch.empty(4 << 30, dtype=torch.uint8, device=bench.device).view(
            torch.float32
        )
        big.normal_()
        for _ in range(3):
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            big.sum()
            e.record()
            torch.cuda.synchronize()
            print(
                f"read bandwidth {big.numel() * 4 / s.elapsed_time(e) / 1e9:.2f} TB/s"
            )
        del big
    spacer, *_ = bench.graph(16, "clc", False, spacers_only=True)
    spacer_us = statistics.median(bench.time(spacer, args.replays) for _ in range(3))
    print(f"spacer {spacer_us:.2f} us/half", flush=True)
    if args.check:
        for distinct in args.distinct:
            outs = {}
            for v in args.variants:
                graph, bufs, _ = bench.graph(distinct, v, False)
                graph.replay()
                torch.cuda.synchronize()
                # the routed slots' rows (-1: a slot the other EP rank computes)
                outs[v] = [
                    (
                        b["y"][b["permuted"].flatten().clamp(min=0).long()][
                            b["permuted"].flatten() >= 0
                        ].clone(),
                        b["h"].clone(),
                    )
                    for b in bufs
                ]
                del graph
            ref = outs[args.variants[0]]
            for v in args.variants[1:]:
                same = all(
                    torch.equal(a[0].view(torch.int16), b[0].view(torch.int16))
                    for a, b in zip(ref, outs[v])
                )
                print(f"check D={distinct} {v}: {'bitwise' if same else 'MISMATCH'}")
    for distinct in args.distinct:
        for side in args.side:
            graphs = {v: bench.graph(distinct, v, side)[0] for v in args.variants}
            times = {v: [] for v in args.variants}
            for r in range(args.repeats):
                order = args.variants if r % 2 == 0 else args.variants[::-1]
                for v in order:
                    times[v].append(bench.time(graphs[v], args.replays) - spacer_us)
            for v in args.variants:
                t = times[v]
                row = dict(
                    distinct=distinct,
                    side=side,
                    variant=v,
                    us=statistics.median(t),
                    lo=min(t),
                    hi=max(t),
                )
                print(
                    f"D={distinct:2d} side={side} {v:40s} {row['us']:7.2f} us "
                    f"[{row['lo']:.2f} .. {row['hi']:.2f}]",
                    flush=True,
                )
            del graphs


if __name__ == "__main__":
    main()
