# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import argparse
import json
import os
import statistics
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

HIDDEN, TOP_K, EPS = 4096, 10, 1e-6
QKVZ_N, ROUTER_N = 2560, 512  # in_proj_qkvz's rows a rank at TP8, the router's experts


def _setup(max_m: int):
    from sglang.srt.distributed import parallel_state as ps
    from sglang.srt.layers.flashinfer_mnnvl_cutedsl import (
        FlashInferMNNVLCuteDSLARFusion,
    )
    from sglang.srt.runtime_context import get_spec
    from sglang.test.test_utils import publish_build_topology

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    ps.set_custom_all_reduce(False)
    ps.init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=int(os.environ["LOCAL_RANK"]),
        distributed_init_method="env://",
    )
    publish_build_topology(tp_size=world, world_rank=rank)
    ps.initialize_model_parallel()
    with get_spec().override(speculative_algorithm="EAGLE"):
        fusion = FlashInferMNNVLCuteDSLARFusion(
            hidden_size=HIDDEN,
            top_k=TOP_K,
            max_m=max_m,
            rms_epsilon=EPS,
            weight_bias=0.0,
            process_group=ps.get_tp_group().device_group,
            device=torch.device(f"cuda:{torch.cuda.current_device()}"),
        )
    return rank, world, fusion


def _tail(fusion, early):
    from sgl_nvidia_mega.qwen3_5 import plan
    from sgl_nvidia_mega.qwen3_5.boundary.bt_tail import BTTail

    service = SimpleNamespace(
        _workspace=fusion, hidden_size=HIDDEN, top_k=TOP_K, rms_epsilon=EPS
    )
    tail = BTTail(
        service=service,
        eps=EPS,
        device=torch.device(f"cuda:{torch.cuda.current_device()}"),
        min_rows=plan.BT_TAIL_MIN_ROWS,
        max_rows=plan.BT_TAIL_MAX_ROWS,
        early=frozenset(early),
    )
    if tail.bind():
        raise RuntimeError("the BT tail does not bind to this workspace")
    return tail


def _inputs(m: int, layers: int, rank: int, device):
    gen = torch.Generator(device=device).manual_seed(1234 + rank)
    common = torch.Generator(device=device).manual_seed(77)

    def randn(*shape, std=1.0, g=gen):
        return (torch.randn(*shape, device=device, generator=g) * std).bfloat16()

    rows = m * TOP_K
    per_layer = []
    for _ in range(layers):
        per_layer.append(
            dict(
                routed=randn(rows, HIDDEN, std=0.05),
                weights=torch.softmax(
                    torch.randn(m, TOP_K, device=device, generator=gen), -1
                ).bfloat16(),
                perm=torch.randperm(rows, device=device, generator=gen)
                .to(torch.int32)
                .view(m, TOP_K),
                shared=randn(m, HIDDEN, std=0.05),
                partial=randn(m, HIDDEN, std=0.05),
                gamma_in=(1 + 0.1 * randn(HIDDEN, g=common).float()).bfloat16(),
                gamma_post=(1 + 0.1 * randn(HIDDEN, g=common).float()).bfloat16(),
                # in_proj_qkvz [K, N] column-major E4M3 and the router [N, K] BF16
                qkvz=(torch.randn(QKVZ_N, HIDDEN, device=device, generator=common) * 4)
                .clamp(-440, 440)
                .to(torch.float8_e4m3fn)
                .t(),
                router=randn(ROUTER_N, HIDDEN, std=0.02, g=common),
            )
        )
    return per_layer, randn(m, HIDDEN, g=common)


def _layer_fn(arm: str, mode: str, fusion, tails: dict, m: int, layers, resid0):
    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_bmm_fp8

    device = resid0.device
    sx = torch.tensor([0.05], device=device)
    sw = torch.tensor([0.01], device=device)
    engine = fusion if arm == "stock" else tails[arm]
    fp8 = arm != "stock" and arm != "tail"
    state = {"res": resid0}

    def run():
        res = state["res"]
        for x in layers:
            out = engine.moe_finalize_all_reduce_rms_norm(
                routed_output=x["routed"],
                expert_weights=x["weights"],
                permuted_indices=x["perm"],
                gated_shared_output=x["shared"],
                residual=res,
                gamma=x["gamma_in"],
                **({"fp8_scale": sx} if fp8 else {}),
            )
            norm, res = out[0], out[1]
            if mode == "layer":
                if fp8:
                    x8 = out[2]
                else:
                    x8, _ = static_quant_fp8(norm, sx, repeat_scale=False)
                flashinfer_bmm_fp8(x8, x["qkvz"], sx, sw, torch.bfloat16)
            norm, res = engine.all_reduce_residual_rms_norm(
                local_contribution=x["partial"], residual=res, gamma=x["gamma_post"]
            )[:2]
            if mode == "layer":
                F.linear(norm, x["router"])
        state["res"] = res

    return run


def _time(run, reps: int) -> float:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    del graph
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rows", default="42,56,84")
    parser.add_argument("--layers", type=int, default=30)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--modes", default="hop,layer")
    parser.add_argument("--rounds", type=int, default=2, help="ABBA rounds per point")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = [int(r) for r in args.rows.split(",")]
    rank, world, fusion = _setup(max(rows))
    tails = {
        "tail": _tail(fusion, ()),
        "tail_fp8": _tail(fusion, ()),
        "tail_fp8_early": _tail(fusion, ("all_reduce",)),
        "tail_fp8_early2": _tail(fusion, ("all_reduce", "finalize")),
    }
    arms = {
        "hop": ("stock", "tail", "tail_fp8"),
        "layer": ("stock", "tail_fp8", "tail_fp8_early", "tail_fp8_early2"),
    }
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    results = []
    with torch.inference_mode():
        for m in rows:
            if not tails["tail"].supports(m):
                raise ValueError(f"{m} rows are not a BT width of both points")
            layers, resid0 = _inputs(m, args.layers, rank, device)
            for mode in args.modes.split(","):
                points = args.layers * (2 if mode == "hop" else 1)
                seen = {arm: [] for arm in arms[mode]}
                for r in range(args.rounds):
                    order = arms[mode] if r % 2 == 0 else arms[mode][::-1]
                    for arm in order:
                        run = _layer_fn(arm, mode, fusion, tails, m, layers, resid0)
                        seen[arm].append(_time(run, args.reps) / points)
                for arm, values in seen.items():
                    t = torch.tensor([statistics.median(values)], device=device)
                    dist.all_reduce(t, op=dist.ReduceOp.MAX)
                    results.append(
                        dict(mode=mode, rows=m, arm=arm, us=t.item(), rounds=values)
                    )
                if rank == 0:
                    base = next(
                        r["us"]
                        for r in results
                        if (r["mode"], r["rows"], r["arm"]) == (mode, m, "stock")
                    )
                    for r in results:
                        if (r["mode"], r["rows"]) == (mode, m):
                            print(
                                f"{mode:5s} rows {m:4d} {r['arm']:16s} {r['us']:7.3f} us "
                                f"({r['us'] - base:+.3f})",
                                flush=True,
                            )
    if rank == 0:
        with open(args.out, "w") as f:
            json.dump(dict(args=vars(args), world=world, results=results), f, indent=1)
    dist.barrier()


if __name__ == "__main__":
    main()
