# SPDX-License-Identifier: Apache-2.0
"""GLM5 single-operator hot runs or independent MLA/MoE mixed workloads.

GPU kernel durations use Kineto/CUPTI in eager or CUDA Graph execution.
JIT, native-weight binding, input preparation and warmup precede sampling.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile, supported_activities

from sgl_nvidia_mega.glm_5.attention.step import cache_views
from sgl_nvidia_mega.glm_5.plan import CONFIGS, ExecutionPlan
from sgl_nvidia_mega.glm_5.runtime import state as runtime_state
from sgl_nvidia_mega.glm_5.runtime.weights import MlaWeights, MoeWeights
from sgl_nvidia_mega.glm_5.sm103 import jit, mla, moe


def _fp8(shape, generator, device):
    return (
        torch.randn(shape, generator=generator, device=device)
        .mul_(0.05)
        .to(torch.float8_e4m3fn)
    )


def _mla_case(plan, device, rank, args, layer):
    t, h, heads = plan.rows, 6144, 64 // plan.tp_size
    capacity = args.max_seq_len * plan.batch_size
    seed = args.seed + layer
    common = torch.Generator(device=device).manual_seed(seed)
    local = torch.Generator(device=device).manual_seed(seed + 1000 + rank)
    x = torch.randn(plan.shape, generator=common, device=device, dtype=torch.bfloat16)
    residual = torch.randn(
        plan.shape, generator=common, device=device, dtype=torch.bfloat16
    )
    selected = torch.stack(
        [
            torch.randperm(
                args.max_seq_len,
                generator=common,
                device=device,
                dtype=torch.int32,
            )[:2048]
            + (token // plan.draft_tokens) * args.max_seq_len
            for token in range(t)
        ]
    )
    phase = torch.randn((t, 32), generator=common, device=device).mul_(0.01)
    positions = torch.cat(
        [
            torch.arange(
                (request + 1) * args.max_seq_len - plan.draft_tokens,
                (request + 1) * args.max_seq_len,
                dtype=torch.int32,
                device=device,
            )
            for request in range(plan.batch_size)
        ]
    )
    if args.layout == "sglang":
        storage = (_fp8((capacity, 576), common, device),)
        kv, pe = cache_views(storage[0])
        table = torch.randn((capacity, 32), generator=common, device=device).mul_(0.01)
        rope = torch.cat((table.cos(), table.sin()), dim=-1)
        rope_positions = positions.long()
    else:
        kv = _fp8((capacity, 512), common, device)
        pe = _fp8((capacity, 64), common, device)
        storage = (kv, pe)
        rope = torch.stack((phase.cos(), phase.sin()), dim=-1).reshape(t, 64)
        rope_positions = None
    ones = lambda shape: torch.ones(shape, device=device, dtype=torch.float32)
    weights = MlaWeights(
        ones(h),
        _fp8((2048, h), common, device),
        ones((16, 48)),
        _fp8((576, h), common, device),
        ones((5, 48)),
        ones(2048),
        _fp8((heads * 256, 2048), local, device),
        ones((heads * 2, 16)),
        ones(512),
        _fp8((heads * 448, 512), local, device),
        ones((heads * 7 // 2, 4)),
        _fp8((h, heads * 256), local, device),
        ones((48, heads * 2)),
    )
    return {
        "x": x,
        "weights": weights,
        "selected_indices": selected,
        "selected_counts": torch.full((t,), 2048, dtype=torch.int32, device=device),
        "kv_cache": kv,
        "pe_cache": pe,
        "storage": storage,
        "cache_positions": positions,
        "rope": rope,
        "rope_positions": rope_positions,
        "residual": residual,
        "local_output": torch.empty_like(x),
        "final_output": torch.empty_like(x),
    }


def _expert_weights(shape, generator, device):
    result = torch.empty(shape, dtype=torch.float8_e4m3fn, device=device)
    for begin in range(0, shape[0], 32):
        end = min(begin + 32, shape[0])
        values = torch.randn(
            (end - begin, *shape[1:]),
            dtype=torch.bfloat16,
            generator=generator,
            device=device,
        )
        result[begin:end].copy_(values.mul_(0.5).to(torch.float8_e4m3fn))
    return result


def _moe_case(plan, device, rank, args, layer):
    inter, seed = 2048 // plan.tp_size, args.seed + layer
    routing = torch.Generator(device=device).manual_seed(seed)
    local = torch.Generator(device=device).manual_seed(seed + rank * 1009)
    x = torch.randn(plan.shape, generator=routing, device=device, dtype=torch.bfloat16)
    gamma = torch.randn(6144, generator=routing, device=device).mul_(0.02)
    gamma.add_(1.0)
    router = torch.randn(
        (256, 6144),
        generator=routing,
        device=device,
        dtype=torch.bfloat16,
    )
    router.mul_(6144**-0.5)
    bias = torch.rand(256, generator=routing, device=device)
    bias.sub_(0.5).mul_(0.1)
    scale = lambda shape: (
        torch.rand(shape, generator=local, device=device).mul_(0.01).add_(0.015)
    )
    weights = MoeWeights(
        gamma,
        router,
        bias,
        _expert_weights((257, 2 * inter, 6144), local, device),
        scale((257, 2 * inter // 128, 48)),
        _expert_weights((257, 6144, inter), local, device),
        scale((257, 48, inter // 128)),
    )
    return {
        "x": x,
        "weights": weights,
        "residual": x.clone(),
        "local_output": None,
        "final_output": torch.empty_like(x),
        "selected_indices": torch.empty(
            (plan.batch_size, plan.draft_tokens, 8),
            dtype=torch.int32,
            device=device,
        ),
        "selected_probs": torch.empty(
            (plan.batch_size, plan.draft_tokens, 8),
            dtype=torch.float32,
            device=device,
        ),
    }


def _states(metadata, count, device):
    if not dist.is_initialized() or count == 1:
        return [
            runtime_state.KernelState.allocate(metadata, device) for _ in range(count)
        ]
    owners, peers = runtime_state._symmetric_peers(
        device, metadata.peer_bytes * count, dist.group.WORLD
    )
    states = []
    for index in range(count):
        workspace = torch.empty(
            metadata.workspace_bytes, dtype=torch.uint8, device=device
        )
        workspace.fill_(metadata.workspace_fill)
        epochs = torch.zeros(metadata.epoch_count, dtype=torch.int32, device=device)
        states.append(
            runtime_state.KernelState(
                metadata,
                workspace,
                epochs,
                tuple(peer + index * metadata.peer_bytes for peer in peers),
                dist.get_rank(),
                owners,
            )
        )
    return states


@dataclass
class Prepared:
    op: str
    case: dict
    bound: Any
    dynamic: dict

    def run(self):
        self.bound(**self.dynamic)


def _prepare(plan, device, rank, args):
    operators = ("mla", "moe") if args.op == "both" else (args.op,)
    prepared = {}
    for op in operators:
        count = (
            (args.layers if op == "mla" else min(args.layers, args.moe_states))
            if args.mode == "mixed"
            else 1
        )
        kernel = (mla if op == "mla" else moe).load(
            draft_tokens=plan.draft_tokens,
            tp_size=plan.tp_size,
            batch_size=plan.batch_size,
        )
        prepared[op] = []
        for index, state in enumerate(_states(kernel.metadata, count, device)):
            case = (
                _mla_case(plan, device, rank, args, index)
                if op == "mla"
                else _moe_case(plan, device, rank, args, index)
            )
            if dist.is_initialized():
                tensors = [case["x"], case["residual"]]
                if op == "mla":
                    tensors += [
                        case["selected_indices"],
                        case["selected_counts"],
                        case["cache_positions"],
                        case["rope"],
                        *case["storage"],
                    ]
                    if case["rope_positions"] is not None:
                        tensors.append(case["rope_positions"])
                else:
                    tensors += list(case["weights"].kernel_args()[:3])
                for tensor in tensors:
                    dist.broadcast(tensor, src=0)
            fixed = (
                dict(
                    kv_cache=case["kv_cache"],
                    pe_cache=case["pe_cache"],
                    layout=args.layout,
                )
                if op == "mla"
                else {}
            )
            bound = kernel.bind(case["weights"], state=state, **fixed)
            dynamic = {
                key: value
                for key, value in case.items()
                if key not in ("weights", "kv_cache", "pe_cache", "storage")
            }
            prepared[op].append(Prepared(op, case, bound, dynamic))
    if args.mode == "single":
        return prepared[args.op] * args.layers
    return [
        item
        for index, attention in enumerate(prepared["mla"])
        for item in (
            prepared["moe"][index % len(prepared["moe"])],
            attention,
        )
    ]


def _capture(calls):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for item in calls:
            item.run()
    torch.cuda.synchronize()
    return graph


def _kernel_statistics(profiler, op, expected, skip):
    events = sorted(
        (
            event
            for event in profiler.events()
            if event.device_type == DeviceType.CUDA
            and op in event.name.lower()
            and event.device_time_total > 0
        ),
        key=lambda event: event.time_range.start,
    )
    if len(events) != expected + skip:
        names = sorted({event.name for event in events})
        raise AssertionError(
            f"{op}: expected {expected + skip} kernels, got {len(events)}: {names}"
        )
    samples = [float(event.device_time_total) for event in events[skip:]]
    ordered = sorted(samples)
    return {
        "mean_us": statistics.fmean(samples),
        "median_us": statistics.median(samples),
        "p90_us": ordered[int(0.9 * (len(ordered) - 1))],
        "samples": len(samples),
        "discarded_startup_samples": skip,
    }


def _profile(calls, args, device, group):
    distributed = group is not None
    merged = args.execution == "graph" and distributed
    queued = calls * args.profile_replays if merged else calls
    counts = Counter(item.op for item in queued)
    unique = {id(item): item for item in queued}
    epochs = {key: item.bound.state.epochs.clone() for key, item in unique.items()}
    for item in queued:
        item.run()
    torch.cuda.synchronize(device)
    graph = _capture(queued) if args.execution == "graph" else None

    def replay():
        if graph is not None:
            graph.replay()
        else:
            for item in queued:
                item.run()

    warmup = (
        math.ceil(args.warmup_replays / args.profile_replays)
        if merged
        else args.warmup_replays
    )
    replays = 1 if merged else args.profile_replays
    prime = int(distributed)
    rounds = []
    for _ in range(args.rounds):
        for _ in range(warmup):
            replay()
        torch.cuda.synchronize(device)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
        ) as profiler:
            if distributed:
                dist.barrier(group=group)
            for _ in range(prime + replays):
                replay()
            torch.cuda.synchronize(device)
        rounds.append(
            {
                op: _kernel_statistics(profiler, op, count * replays, count * prime)
                for op, count in counts.items()
            }
        )
    executions = 1 + args.rounds * (warmup + prime + replays)
    occurrences = Counter(id(item) for item in queued)
    for key, item in unique.items():
        expected = epochs[key] + occurrences[key] * executions
        assert torch.equal(item.bound.state.epochs, expected), (
            "unexpected state epoch progression"
        )
        assert torch.isfinite(item.case["final_output"]).all(), (
            "non-finite kernel output"
        )
    return {
        "profiler": rounds,
        "mean_gpu_us": {
            op: statistics.fmean(row[op]["mean_us"] for row in rounds) for op in counts
        },
        "states": dict(Counter(item.op for item in unique.values())),
        "calls_per_sample_batch": dict(counts),
    }


@torch.inference_mode()
def _worker(rank, world, args):
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (10, 3):
        raise RuntimeError("GLM5 kernels target SM103")
    if ProfilerActivity.CUDA not in supported_activities():
        raise RuntimeError("GLM5 benchmark requires CUDA profiling support")
    group = None
    if world > 1:
        dist.init_process_group(
            "nccl",
            init_method=f"tcp://127.0.0.1:{args.port}",
            rank=rank,
            world_size=world,
            timeout=timedelta(seconds=120),
        )
        group = dist.new_group(list(range(world)), backend="gloo")
    try:
        plan = ExecutionPlan(args.draft_tokens, args.tp_size, args.batch_size)
        calls = _prepare(plan, device, rank, args)
        local = _profile(calls, args, device, group)
        reports = [None] * world
        if world > 1:
            dist.all_gather_object(reports, local, group=group)
        else:
            reports[0] = local
        if rank == 0:
            print(
                json.dumps(
                    {
                        "mode": args.mode,
                        "op": args.op,
                        "execution": args.execution,
                        "draft_tokens": args.draft_tokens,
                        "tp_size": args.tp_size,
                        "batch_size": args.batch_size,
                        "kernel_rows": plan.rows,
                        "mtp_label": plan.variant("mla").mtp_label,
                        "world": world,
                        "local_shard": args.local,
                        "layout": args.layout,
                        "max_seq_len": args.max_seq_len,
                        "steps_per_batch": args.layers,
                        "states": local["states"],
                        "moe_states": local["states"].get("moe", 0),
                        "seed": args.seed,
                        "warmup_replays": args.warmup_replays,
                        "profile_replays": args.profile_replays,
                        "rounds": args.rounds,
                        "cuda_version": jit.cuda_version(),
                        "measurement": (
                            "Kineto/CUPTI GPU kernel duration; "
                            "TP startup batch discarded"
                        ),
                        "max_rank_mean_gpu_us": {
                            op: max(report["mean_gpu_us"][op] for report in reports)
                            for op in local["mean_gpu_us"]
                        },
                        "ranks": reports,
                    }
                ),
                flush=True,
            )
    finally:
        if world > 1:
            dist.destroy_process_group()


def _inventory():
    records = []
    for draft, tp, batch in CONFIGS:
        plan = ExecutionPlan(draft, tp, batch)
        for op in ("mla", "moe"):
            variant = plan.variant(op)
            reasons = jit.source_reasons(variant)
            records.append(
                {
                    "op": op,
                    "draft_tokens": draft,
                    "tp_size": tp,
                    "batch_size": batch,
                    "kernel_rows": plan.rows,
                    "mtp_label": variant.mtp_label,
                    "sources_ready": not reasons,
                    "reasons": reasons,
                }
            )
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("single", "mixed"), default="single")
    parser.add_argument("--op", choices=("mla", "moe", "both"))
    parser.add_argument("--execution", choices=("eager", "graph"), default="graph")
    parser.add_argument("--draft-tokens", type=int, choices=(4, 8), default=4)
    parser.add_argument("--tp-size", type=int, choices=(4, 8), default=4)
    parser.add_argument("--batch-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--layout", choices=("sglang", "legacy"), default="sglang")
    parser.add_argument("--layers", type=int, default=56)
    parser.add_argument("--moe-states", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--warmup-replays", type=int)
    parser.add_argument("--profile-replays", type=int, default=4)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--list-configs", action="store_true")
    parser.add_argument("--check-sources", action="store_true")
    args = parser.parse_args(argv)
    if args.list_configs or args.check_sources:
        records = _inventory()
        print(json.dumps(records, indent=2))
        return int(
            args.check_sources and any(not item["sources_ready"] for item in records)
        )
    args.op = args.op or ("both" if args.mode == "mixed" else "mla")
    if args.mode == "single" and args.op == "both":
        parser.error("single mode requires --op mla or --op moe")
    if args.mode == "mixed" and args.op != "both":
        parser.error("mixed mode requires --op both")
    args.max_seq_len = (
        args.max_seq_len
        if args.max_seq_len is not None
        else (4096 if args.mode == "single" else 8192)
    )
    args.warmup_replays = (
        args.warmup_replays
        if args.warmup_replays is not None
        else (100 if args.mode == "single" else 60)
    )
    args.rounds = (
        args.rounds if args.rounds is not None else (2 if args.mode == "single" else 4)
    )
    if min(args.layers, args.moe_states, args.profile_replays, args.rounds) < 1:
        parser.error("layers, states, profile replays and rounds must be positive")
    if args.warmup_replays < 0:
        parser.error("warmup replays must be nonnegative")
    if args.max_seq_len < 2048 + args.draft_tokens:
        parser.error("max sequence length is too small for selected KV and writes")
    try:
        ExecutionPlan(args.draft_tokens, args.tp_size, args.batch_size)
    except ValueError as error:
        parser.error(str(error))
    world = 1 if args.local else args.tp_size
    if not torch.cuda.is_available() or torch.cuda.device_count() < world:
        parser.error(f"needs {world} CUDA GPU(s)")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        args.port = sock.getsockname()[1]
    if world == 1:
        _worker(0, world, args)
    else:
        torch.multiprocessing.spawn(
            _worker, args=(world, args), nprocs=world, join=True
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
