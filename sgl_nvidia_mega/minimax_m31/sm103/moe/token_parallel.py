# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The token-parallel MoE of a MiniMax-M3.1 decode layer on TP4: eight rows, the FP32
router, the MXFP8 shared expert, the NVFP4 routed experts (top-4 of 128) and the TP4
all-reduce in one cooperative launch per rank (``kernel/src/moe_token_parallel.cu``).

The ranks exchange partial sums through a 16 MiB arena per rank, zeroed once; every
call advances the arena's generation, so no per-call clearing is needed. Calls sharing
an arena (every layer's) run in stream order."""

from __future__ import annotations

import ctypes
from functools import cache
from pathlib import Path

import torch

from sgl_nvidia_mega.minimax_m31.sm103 import host, nvcc

KERNEL_ROOT = Path(__file__).resolve().parent / "kernel"

T, H, I, E, K = 8, 6144, 768, 128, 4
R = T * K
GRID, THREADS, SMEM_BYTES = 144, 384, 232448
ARENA_BYTES = 16 << 20
# Arena offsets of the activation hand-offs (kernel/include/sm100/common/layout.hpp)
SHARED_INPUT = 12 << 20
SHARED_INPUT_SCALE = SHARED_INPUT + T * H
ROUTED_INPUT = SHARED_INPUT_SCALE + T * (H // 32)
ROUTED_INPUT_SCALE = ROUTED_INPUT + T * (H // 2)
SHARED_MID = ROUTED_INPUT_SCALE + T * (H // 16)
SHARED_MID_SCALE = SHARED_MID + T * I
ROUTED_MID = SHARED_MID_SCALE + T * (I // 32)
ROUTED_MID_SCALE = ROUTED_MID + R * (I // 2)
UPGATE_K, SHARED_GU_PAGES_PER_SLOT, SHARED_CTAS = 512, 3, 16

# The kernel's exact-IEEE contract (no fast math, no flush to zero), and the half and
# bfloat16 operator macros it was validated with.
OPTIONS = (
    "-O3",
    "-std=c++20",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "--expt-relaxed-constexpr",
)
# The kernel source, then the launch and arena constants the host checks against its own
SOURCE = """#include "src/moe_token_parallel.cu"
#include <cstddef>
namespace mega_moe::token_parallel {
extern "C" __device__ long long minimax_m31_moe_constants[] = {
    kGrid, kThreads, kSmemBytes, (long long)kArenaBytes, (long long)kSharedInput,
    (long long)kSharedInputScale, (long long)kRoutedInput, (long long)kRoutedInputScale,
    (long long)kSharedMid, (long long)kSharedMidScale, (long long)kRoutedMid,
    (long long)kRoutedMidScale, kUpgateK, kSharedGuPagesPerSlot, kShared,
    (long long)sizeof(Args), (long long)offsetof(KernelArgs, maps), (long long)sizeof(KernelArgs)};
}
"""
SYMBOL = "_ZN8mega_moe14token_parallel10moe_kernelENS0_10KernelArgsE"

# (name, dtype, shape) of a layer's tensors, in the order of Args' pointers; then its output
TENSORS = (
    ("hidden", torch.bfloat16, (T, H)),
    ("residual", torch.bfloat16, (T, H)),
    ("router_weight", torch.float32, (E, H)),
    ("router_correction_bias", torch.float32, (E,)),
    ("shared_fc1_weight", torch.float8_e4m3fn, (2 * I, H)),
    ("shared_fc1_weight_sf", torch.uint8, (2 * I, H // 32)),
    ("shared_fc2_weight", torch.float8_e4m3fn, (H, I)),
    ("shared_fc2_weight_sf", torch.uint8, (H, I // 32)),
    ("routed_fc1_weight", torch.uint8, (E, 2 * I, H // 2)),
    ("routed_fc1_weight_sf", torch.float8_e4m3fn, (E, 2 * I, H // 16)),
    ("routed_fc2_weight", torch.uint8, (E, H, I // 2)),
    ("routed_fc2_weight_sf", torch.float8_e4m3fn, (E, H, I // 16)),
    ("routed_fc1_alpha", torch.float32, (E,)),
    # FC2's dequant scale with the routing multiplier folded in
    ("routed_fc2_alpha", torch.float32, (E,)),
)
OUTPUT = ("output", torch.bfloat16, (T, H))


class Args(ctypes.Structure):
    """``mega_moe::Args`` (kernel/include/sm100/common/params.hpp)."""

    _fields_ = [
        (name, ctypes.c_void_p)
        for name in (
            "post_attention_norm",
            "attention_residual",
            "router_weight",
            "router_bias",
            "shared_fc1",
            "shared_fc1_sf",
            "shared_fc2",
            "shared_fc2_sf",
            "routed_fc1",
            "routed_fc1_sf",
            "routed_fc2",
            "routed_fc2_sf",
            "fc1_alpha",
            "fc2_alpha",
            "shared_input",
            "shared_input_sf",
            "routed_input",
            "routed_input_sf",
            "shared_mid",
            "shared_mid_sf",
            "routed_mid",
            "routed_mid_sf",
            "out",
        )
    ] + [("ready", ctypes.c_void_p * host.TP), ("rank", ctypes.c_int32)]


# Maps.map[MapId] then Maps.routed_fc2_full (kernel/include/sm100/common/layout.hpp)
(
    ROUTED_FC1_VALUE,
    ROUTED_FC1_SCALE,
    SHARED_FC1_VALUE,
    SHARED_FC2_VALUE,
    SHARED_INPUT_VALUE,
    SHARED_MID_VALUE,
    ROUTED_FC2_FULL,
) = range(7)
MAPS_OFFSET = (ctypes.sizeof(Args) + 63) // 64 * 64
KERNEL_ARGS_BYTES = MAPS_OFFSET + 128 * 7
# minimax_m31_moe_constants in SOURCE, as the host uses them
CONSTANTS = (
    GRID,
    THREADS,
    SMEM_BYTES,
    ARENA_BYTES,
    SHARED_INPUT,
    SHARED_INPUT_SCALE,
    ROUTED_INPUT,
    ROUTED_INPUT_SCALE,
    SHARED_MID,
    SHARED_MID_SCALE,
    ROUTED_MID,
    ROUTED_MID_SCALE,
    UPGATE_K,
    SHARED_GU_PAGES_PER_SLOT,
    SHARED_CTAS,
    ctypes.sizeof(Args),
    MAPS_OFFSET,
    KERNEL_ARGS_BYTES,
)


@cache
def load_module(device_index: int, wait_timeout_ns: int) -> host.Module:
    """Compile (or read from the cache) and load the kernel for this device; a poll on
    another rank's records traps after ``wait_timeout_ns`` (0: never)."""
    device = torch.device("cuda", device_index)
    with torch.cuda.device(device):
        torch.zeros(1, device=device)  # make the primary context current
        options = (*OPTIONS, f"-DMEGA_MOE_WAIT_TIMEOUT_NS={int(wait_timeout_ns)}ull")
        cubin = nvcc.compile_cubin(
            "minimax_m31_moe", SOURCE, (KERNEL_ROOT,), nvcc.device_arch(device), options
        )
        module = host.Module(cubin, SYMBOL)
        constants = module.array("minimax_m31_moe_constants", len(CONSTANTS))
        if constants != CONSTANTS or module.parameter_size(0) != KERNEL_ARGS_BYTES:
            raise RuntimeError(
                f"the kernel's launch and arena constants {constants} are not the host's "
                f"{CONSTANTS}"
            )
        module.set_shared_memory(SMEM_BYTES)
        if module.resident_ctas(device, THREADS, SMEM_BYTES) < GRID:
            raise RuntimeError(f"the device cannot co-schedule {GRID} MoE CTAs")
        return module


def encode_maps(args: Args) -> list:
    """The layer's tensor maps in Maps order (``encode_maps`` in the kernel source)."""
    maps = [None] * 7
    maps[ROUTED_FC1_VALUE] = host.encode(
        "UINT8",
        args.routed_fc1,
        (128, 8, H // 256, I // 8, 2 * E),
        (H // 2, 128, 8 * (H // 2), I * (H // 2)),
        (128, 8, UPGATE_K // 256, 8, 2),
        "128B",
        "L2_256B",
    )
    maps[ROUTED_FC1_SCALE] = host.encode(
        "UINT8",
        args.routed_fc1_sf,
        (32, I, 2 * E, H // 512),
        (H // 16, I * (H // 16), 32),
        (32, 64, 2, UPGATE_K // 512),
        "32B",
        "NONE",
    )
    maps[ROUTED_FC2_FULL] = host.encode(
        "UINT8",
        args.routed_fc2,
        (128, E * H, 3),
        (I // 2, 128),
        (128, 128, 3),
        "128B",
        "NONE",
    )
    maps[SHARED_FC1_VALUE] = host.encode(
        "UINT8",
        args.shared_fc1,
        (128, 48, 2, SHARED_CTAS, H // 128),
        (H, H * I, 48 * H, 128),
        (128, 64, 2, 1, SHARED_GU_PAGES_PER_SLOT),
        "128B",
        "L2_256B",
    )
    maps[SHARED_INPUT_VALUE] = host.encode(
        "UINT8",
        args.shared_input,
        (128, T, H // 128),
        (H, 128),
        (128, T, SHARED_GU_PAGES_PER_SLOT),
        "128B",
        "L2_256B",
    )
    maps[SHARED_FC2_VALUE] = host.encode(
        "UINT8",
        args.shared_fc2,
        (128, H, 6),
        (I, 128),
        (128, 128, 6),
        "128B",
        "L2_256B",
    )
    maps[SHARED_MID_VALUE] = host.encode(
        "UINT8", args.shared_mid, (128, T, 6), (I, 128), (128, 8, 6), "128B", "L2_256B"
    )
    return maps


def create_arena(group, device: torch.device) -> host.Arena:
    """The MoE's arena, shared by every layer; collective."""
    return host.Arena(group, device, ARENA_BYTES)


class TokenParallelMoe:
    """One MoE layer bound to its weights, its input rows (the attention's normalized
    output and its residual) and its output, fixed at construction."""

    def __init__(
        self, arena: host.Arena, *, wait_timeout_ns: int = 0, **tensors: torch.Tensor
    ):
        """``wait_timeout_ns``: a poll on another rank traps this long after its first slow
        poll (0: never)."""
        device = arena.device
        for name, dtype, shape in (*TENSORS, OUTPUT):
            host.check_tensor(tensors[name], name, dtype, shape, device)
        output = tensors["output"]
        for name, _, _ in TENSORS:
            if host.overlaps(output, tensors[name]):
                raise ValueError(f"output must be disjoint from {name}")
        if host.overlaps(tensors["hidden"], tensors["residual"]):
            raise ValueError("hidden and residual must be disjoint")
        self.module = load_module(device.index, wait_timeout_ns)
        self.tensors = tensors  # the maps and pointers hold their addresses
        self.arena = arena
        args = Args()
        pointers = [tensors[name].data_ptr() for name, _, _ in TENSORS]
        for (field, _), pointer in zip(Args._fields_, pointers):
            setattr(args, field, pointer)
        base = arena.base
        args.shared_input = base + SHARED_INPUT
        args.shared_input_sf = base + SHARED_INPUT_SCALE
        args.routed_input = base + ROUTED_INPUT
        args.routed_input_sf = base + ROUTED_INPUT_SCALE
        args.shared_mid = base + SHARED_MID
        args.shared_mid_sf = base + SHARED_MID_SCALE
        args.routed_mid = base + ROUTED_MID
        args.routed_mid_sf = base + ROUTED_MID_SCALE
        args.out = output.data_ptr()
        for rank, peer in enumerate(arena.peers):
            args.ready[rank] = peer
        args.rank = arena.rank
        blob = (ctypes.c_uint8 * KERNEL_ARGS_BYTES)()
        ctypes.memmove(blob, ctypes.addressof(args), ctypes.sizeof(args))
        for index, tensor_map in enumerate(encode_maps(args)):
            host.place_map(blob, MAPS_OFFSET + 128 * index, tensor_map)
        self.blob = blob

    def run(self) -> torch.Tensor:
        """Enqueue the layer on the current stream; graph-capturable."""
        host.launch(
            self.module.function,
            (GRID, 1, 1),
            (THREADS, 1, 1),
            SMEM_BYTES,
            [self.blob],
            torch.cuda.current_stream(self.arena.device).cuda_stream,
            cooperative=True,
        )
        return self.tensors["output"]
