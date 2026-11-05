# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Prepared, read-only weight prefetch joined before each decode step returns.

Dormant unless explicitly requested at build time; OFF prepares no optional
ranges, kernels, streams or events. Policies count token rows, not requests.
Short q1/q6 correctness is bounded evidence, not a demonstrated activation benefit
or long-context/combined-ON qualification. Retained weight owners must outlive
captured graphs, and every side-stream fork must join before forward returns.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from functools import partial

import torch


@dataclass(frozen=True)
class PrefetchPolicy:
    byte_cap: int = 0
    grid: int = 32
    carveout: int | None = None
    priority: bool = False


def prefetch_policy(point: str, kind: str, rows: int) -> PrefetchPolicy | None:
    """TP8 policies use token rows, not the number of speculative requests."""
    if point not in ("oproj_down", "attention_moe", "next_attention"):
        raise ValueError(f"unknown weight prefetch point: {point}")
    if kind not in ("kda", "mla") or type(rows) is not int or rows < 1:
        raise ValueError("weight prefetch requires an attention kind and positive rows")
    if point == "next_attention" and rows <= 8:
        cap = 96 if rows <= 3 else 64 if rows <= 7 else 32
        return PrefetchPolicy(cap << 20, 32, 100 if rows == 1 else None)
    if point == "oproj_down" and 3 <= rows <= 8:
        return PrefetchPolicy(grid=8, carveout=100, priority=True)
    if point == "attention_moe" and (
        (kind == "kda" and 5 <= rows <= 8) or (kind == "mla" and rows == 8)
    ):
        return PrefetchPolicy(carveout=100, priority=True)
    return None


def range_plan(weights, byte_cap: int = 0):
    """Retain borrowed owners and select aligned, nonoverlapping storage ranges.

    Strided weights may include storage between their elements, but never bytes
    outside their allocation. Caps count unique bytes in consumption order.
    """
    if type(byte_cap) is not int or byte_cap < 0 or byte_cap % 16:
        raise ValueError("weight prefetch cap must be a nonnegative multiple of 16")
    owners, ranges = [], []
    identities = set()
    remaining = byte_cap
    for weight in weights:
        if not isinstance(weight, torch.Tensor):
            raise ValueError("weight prefetch ranges require tensors")
        if id(weight) not in identities:
            owners.append(weight)
            identities.add(id(weight))
        if not weight.numel():
            continue
        shape, strides = tuple(weight.shape), tuple(weight.stride())
        if any(stride < 0 for stride in strides):
            raise ValueError("weight prefetch does not support negative strides")
        pointer, storage = weight.data_ptr(), weight.untyped_storage()
        span = 1 + sum((n - 1) * s for n, s in zip(shape, strides))
        nbytes = span * weight.element_size()
        begin, end = storage.data_ptr(), storage.data_ptr() + storage.nbytes()
        if (
            pointer <= 0
            or pointer % 16
            or not begin <= pointer <= pointer + nbytes <= end
        ):
            raise ValueError(
                "weight prefetch range is unaligned or outside its storage"
            )
        segments = [(pointer, pointer + nbytes // 16 * 16)]
        for address, size in ranges:
            stop = address + size
            uncovered = []
            for first, last in segments:
                if stop <= first or address >= last:
                    uncovered.append((first, last))
                else:
                    if first < address:
                        uncovered.append((first, address))
                    if stop < last:
                        uncovered.append((stop, last))
            segments = uncovered
        for first, last in segments:
            size = last - first
            if byte_cap:
                size = min(size, remaining)
            if size:
                ranges.append((first, size))
                if byte_cap:
                    remaining -= size
            if byte_cap and remaining == 0:
                break
        if byte_cap and remaining == 0:
            break
    return tuple(owners), tuple(ranges)


def attention_weights(prepared, layer):
    """Use final prepared storage, including the normalized MLA key matrix."""
    weights = (prepared.projection_weight,)
    if prepared.kind == "mla":
        weights += tuple(prepared.core.weights[2:5])
    elif prepared.kind == "kda":
        weights += (layer.self_attn.f_b_proj.weight,)
    else:
        raise ValueError(f"unknown attention kind: {prepared.kind}")
    return (*weights, layer.self_attn.o_proj.weight)


class PreparedWeightRanges:
    """Own a persistent range table and its executable, without copying weights."""

    def __init__(self, weights, *, byte_cap=0, grid=32, carveout=None, priority=False):
        from cuda.bindings import driver as cuda

        from .cutedsl.common.host import _checked, resolve_kernel, to_cute
        from .cutedsl.common.prefetch import compile_kernel

        self.weights, self.ranges = range_plan(weights, byte_cap)
        if not self.ranges:
            raise ValueError("weight prefetch needs at least one aligned range")
        device = self.weights[0].device
        if device.type != "cuda" or any(w.device != device for w in self.weights):
            raise ValueError("weight prefetch requires weights on one CUDA device")
        if type(grid) is not int or grid < 1 or carveout not in (None, 100):
            raise ValueError("unsupported weight prefetch launch policy")
        if type(priority) is not bool:
            raise ValueError("weight prefetch priority must be boolean")
        self.policy = PrefetchPolicy(byte_cap, grid, carveout, priority)
        self.table = torch.tensor(self.ranges, dtype=torch.int64, device=device)
        self.code = compile_kernel(len(self.ranges), grid, carveout, device.index)
        self.config = None
        if not priority and carveout is None:
            self.views = to_cute(self.table, 16)
            return
        self.executor, self.function = resolve_kernel(self.code, device.index)
        self.attributes = []
        if priority:
            top = cuda.CUlaunchAttribute()
            top.id = cuda.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PRIORITY
            top.value.priority = _checked(cuda.cuCtxGetStreamPriorityRange())[1]
            self.attributes.append(top)
        if carveout is not None:
            smem = cuda.CUlaunchAttribute()
            smem.id = cuda.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
            smem.value.sharedMemCarveout = carveout
            self.attributes.append(smem)
        config = self.config = cuda.CUlaunchConfig()
        config.gridDimX, config.gridDimY, config.gridDimZ = grid, 1, 1
        config.blockDimX, config.blockDimY, config.blockDimZ = 32, 1, 1
        config.sharedMemBytes, config.attrs = 0, self.attributes
        config.numAttrs = len(self.attributes)
        self.pointer = ctypes.c_void_p(self.table.data_ptr())
        self.arguments = (ctypes.c_void_p * 1)(ctypes.addressof(self.pointer))
        self.address = ctypes.addressof(self.arguments)
        self._driver_launch = partial(
            cuda.cuLaunchKernelEx, config, self.function, self.address, 0
        )
        self._checked = _checked

    @property
    def compiled(self):
        return self.code

    def launch(self, stream):
        if self.config is None:
            self.code(self.views, stream)
        else:
            self.config.hStream = stream
            self._checked(self._driver_launch())


class PrefetchController:
    """Prepare TP8 policies once; every fork is rejoined before forward returns."""

    def __init__(self, attention, layers, moe_front_weights, *, device, batches):
        from cuda.bindings import driver as cuda

        self.device = torch.device(device)
        self.tables, self.events, self.pending = {}, {}, False
        layers, batches = tuple(layers), tuple(batches)
        with torch.cuda.device(self.device):
            self.stream = torch.cuda.Stream(device=self.device)
            self.launch_stream = cuda.CUstream(self.stream.cuda_stream)
            self.done = torch.cuda.Event()
            self.done.record(self.stream)
            for index, layer in enumerate(layers):
                if not layer._is_moe_layer:
                    continue
                layer_id = int(layer.layer_idx)
                prepared = attention[layer_id]
                shared = layer.mlp.shared_experts
                packed = moe_front_weights[layer_id]
                points = {
                    "oproj_down": (packed,),
                    "attention_moe": (
                        packed,
                        shared.gate_up_proj.weight,
                        shared.down_proj.weight,
                    ),
                }
                if index + 1 < len(layers):
                    following = layers[index + 1]
                    points["next_attention"] = attention_weights(
                        attention[int(following.layer_idx)], following
                    )
                for point, weights in points.items():
                    groups = {}
                    for rows in batches:
                        policy = prefetch_policy(point, prepared.kind, rows)
                        if policy is not None:
                            groups.setdefault(policy, []).append(rows)
                    for policy, widths in groups.items():
                        ranges = PreparedWeightRanges(
                            weights,
                            byte_cap=policy.byte_cap,
                            grid=policy.grid,
                            carveout=policy.carveout,
                            priority=policy.priority,
                        )
                        for rows in widths:
                            self.tables[point, layer_id, rows] = ranges
                    if groups:
                        event = torch.cuda.Event()
                        event.record(torch.cuda.current_stream(self.device))
                        self.events[point, layer_id] = event

    def fork(self, point, layer_id, rows):
        ranges = self.tables.get((point, layer_id, rows))
        if ranges is None:
            return False
        event = self.events[point, layer_id]
        event.record(torch.cuda.current_stream(self.device))
        self.stream.wait_event(event)
        ranges.launch(self.launch_stream)
        self.pending = True
        return True

    def join(self):
        if self.pending:
            self.done.record(self.stream)
            torch.cuda.current_stream(self.device).wait_event(self.done)
            self.pending = False
