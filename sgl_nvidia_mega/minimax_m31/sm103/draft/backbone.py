# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The MiniMax-M3.1 DSpark draft's backbone on TP4: the vocab-parallel embedding, the
five dense decoder layers and the final norm of one request's seven-row draft block in
one cooperative launch per rank (``kernel/src/draft_backbone.cu``).

The ranks exchange their TP4 partials through a 32 MiB arena per rank, zeroed once;
every call advances its generation. Calls sharing an arena run in stream order."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, fields
from functools import cache
from pathlib import Path

import torch

from sgl_nvidia_mega.minimax_m31.sm103 import host, nvcc

KERNEL_ROOT = Path(__file__).resolve().parent / "kernel"

LAYERS, ROWS, HIDDEN, HEAD_DIM, ROTARY_DIM = 5, 7, 6144, 128, 64
Q_SIZE, QKV_ROWS, INTER = 2048, 2048 + 2 * 128, 3072
GATE_UP_ROWS, SCALE_BLOCK = 2 * INTER, 32
ARENA_BYTES = 32 << 20
THREADS = 8 * 32 + 32  # eight consumer warps and the weight-streaming warp
GEMMS, MAPS_PER_GEMM = 4, 2
K_BLOCK, TALL_ROWS, PAIR_ROWS, STAGE_ROW = 128, 16, 8, 2048

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
# The kernel source, then the launch constants the host reads from the cubin
SOURCE = """#include "src/draft_backbone.cu"
extern "C" __device__ long long minimax_m31_draft_smem_bytes = m31_draft::kSmemBytes;
extern "C" __device__ long long minimax_m31_draft_args_bytes = sizeof(m31_draft::Args);
extern "C" __device__ long long minimax_m31_draft_maps_bytes = sizeof(m31_draft::WeightMaps);
"""

LAYER_TENSORS = (
    ("input_norm", torch.bfloat16, (HIDDEN,)),
    ("post_norm", torch.bfloat16, (HIDDEN,)),
    ("q_norm", torch.bfloat16, (HEAD_DIM,)),
    ("k_norm", torch.bfloat16, (HEAD_DIM,)),
    ("qkv", torch.float8_e4m3fn, (QKV_ROWS, HIDDEN)),
    ("qkv_sf", torch.uint8, (QKV_ROWS, HIDDEN // SCALE_BLOCK)),
    ("o", torch.float8_e4m3fn, (HIDDEN, Q_SIZE)),
    ("o_sf", torch.uint8, (HIDDEN, Q_SIZE // SCALE_BLOCK)),
    ("gate_up", torch.float8_e4m3fn, (GATE_UP_ROWS, HIDDEN)),
    ("gate_up_sf", torch.uint8, (GATE_UP_ROWS, HIDDEN // SCALE_BLOCK)),
    ("down", torch.float8_e4m3fn, (HIDDEN, INTER)),
    ("down_sf", torch.uint8, (HIDDEN, INTER // SCALE_BLOCK)),
    ("k_cache", torch.bfloat16, (-1, HEAD_DIM)),
    ("v_cache", torch.bfloat16, (-1, HEAD_DIM)),
)
STEP_TENSORS = (
    ("input_ids", torch.int64, (ROWS,)),
    ("embed", torch.bfloat16, (-1, HIDDEN)),
    ("out", torch.bfloat16, (ROWS, HIDDEN)),
    ("final_norm", torch.bfloat16, (HIDDEN,)),
    ("positions", torch.int64, (ROWS,)),
    ("slot_mapping", torch.int64, (ROWS,)),
    ("page_table", torch.int32, (-1,)),
    ("kv_len", torch.int32, (1,)),
    ("cos_sin", torch.float32, (-1, ROTARY_DIM)),
)


class LayerArgs(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name, _, _ in LAYER_TENSORS]


class Args(ctypes.Structure):
    """``m31_draft::Args`` (kernel/include/draft_args.hpp)."""

    _fields_ = (
        [("layer", LayerArgs * LAYERS)]
        + [(name, ctypes.c_void_p) for name, _, _ in STEP_TENSORS]
        + [
            ("arena", ctypes.c_void_p * host.TP),
            ("vocab_start", ctypes.c_int64),
            ("vocab_rows", ctypes.c_int64),
            ("rank", ctypes.c_int32),
            ("max_pages", ctypes.c_int32),
            ("eps", ctypes.c_float),
        ]
    )


MAPS_BYTES = 128 * LAYERS * GEMMS * MAPS_PER_GEMM
GEMM_QKV, GEMM_O, GEMM_GATE_UP, GEMM_DOWN = range(GEMMS)


@cache
def load_module(device_index: int, wait_timeout_ns: int) -> host.Module:
    """Compile (or read from the cache) and load the kernel for this device; a wait on
    another CTA or rank traps after ``wait_timeout_ns`` (0: never)."""
    device = torch.device("cuda", device_index)
    with torch.cuda.device(device):
        torch.zeros(1, device=device)  # make the primary context current
        bound = int(wait_timeout_ns) if wait_timeout_ns > 0 else (1 << 64) - 1
        options = (*OPTIONS, f"-DM31_DRAFT_WAIT_TIMEOUT_NS={bound}ull")
        cubin = nvcc.compile_cubin(
            "minimax_m31_draft",
            SOURCE,
            (KERNEL_ROOT,),
            nvcc.device_arch(device),
            options,
        )
        (symbol,) = [
            n for n in nvcc.kernel_names(cubin) if "draft_backbone_kernel" in n
        ]
        module = host.Module(cubin, symbol)
        sizes = (
            module.constant("minimax_m31_draft_args_bytes"),
            module.constant("minimax_m31_draft_maps_bytes"),
        )
        if sizes != (ctypes.sizeof(Args), MAPS_BYTES) or sizes != (
            module.parameter_size(0),
            module.parameter_size(1),
        ):
            raise RuntimeError(f"draft backbone parameters are {sizes} bytes")
        module.shared = module.constant("minimax_m31_draft_smem_bytes")
        module.set_shared_memory(module.shared)
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        if module.resident_ctas(device, THREADS, module.shared) < sms:
            raise RuntimeError("the device cannot hold one draft backbone CTA per SM")
        return module


def encode_maps(args: Args) -> ctypes.Array:
    """``WeightMaps``: each layer's GEMM weights (the ``encode`` of every
    ``op_gemm_*.cuh``), 16-row units of 2048 K bytes in 128-byte swizzled boxes."""
    maps = (ctypes.c_uint8 * MAPS_BYTES)()

    def put(
        layer: int,
        gemm: int,
        slot: int,
        pointer: int,
        rows: int,
        k: int,
        box_rows,
        box_k,
    ):
        tensor_map = host.encode(
            "UINT8",
            pointer,
            (K_BLOCK, rows, k // K_BLOCK),
            (k, K_BLOCK),
            (K_BLOCK, box_rows, box_k // K_BLOCK),
            "128B",
            "L2_256B",
        )
        offset = 128 * ((layer * GEMMS + gemm) * MAPS_PER_GEMM + slot)
        host.place_map(maps, offset, tensor_map)

    for index in range(LAYERS):
        layer = args.layer[index]
        put(index, GEMM_QKV, 0, layer.qkv, QKV_ROWS, HIDDEN, TALL_ROWS, STAGE_ROW)
        put(index, GEMM_O, 0, layer.o, HIDDEN, Q_SIZE, TALL_ROWS, STAGE_ROW)
        put(
            index,
            GEMM_GATE_UP,
            0,
            layer.gate_up,
            GATE_UP_ROWS,
            HIDDEN,
            PAIR_ROWS,
            STAGE_ROW,
        )
        put(index, GEMM_DOWN, 0, layer.down, HIDDEN, INTER, TALL_ROWS, STAGE_ROW)
        put(index, GEMM_DOWN, 1, layer.down, HIDDEN, INTER, TALL_ROWS, STAGE_ROW // 2)
    return maps


def create_arena(group, device: torch.device) -> host.Arena:
    """The draft backbone's arena, shared by its calls; collective."""
    return host.Arena(group, device, ARENA_BYTES)


@dataclass
class DraftLayerWeights:
    """One decoder layer's rank-local operands in the checkpoint's row-major layouts."""

    # bf16 [6144]: raw Gemma w of input_layernorm (applied as 1 + w)
    input_norm: torch.Tensor
    post_norm: torch.Tensor  # bf16 [6144]: raw Gemma w of post_attention_layernorm
    q_norm: torch.Tensor  # bf16 [128]
    k_norm: torch.Tensor  # bf16 [128]
    qkv: torch.Tensor  # e4m3 [2304, 6144]: q (16 heads) | k | v rows of this rank
    qkv_sf: torch.Tensor  # u8 [2304, 192]: UE8M0 per 32 K values
    o: torch.Tensor  # e4m3 [6144, 2048]
    o_sf: torch.Tensor  # u8 [6144, 64]
    gate_up: torch.Tensor  # e4m3 [6144, 6144]: gate 3072 | up 3072 rows of this rank
    gate_up_sf: torch.Tensor  # u8 [6144, 192]
    down: torch.Tensor  # e4m3 [6144, 3072]
    down_sf: torch.Tensor  # u8 [6144, 96]


@dataclass
class DraftBackboneWeights:
    layers: list  # LAYERS x DraftLayerWeights
    embed: torch.Tensor  # bf16 [vocab_rows, 6144]: this rank's vocab shard
    vocab_start: int  # first token id of the shard
    final_norm: torch.Tensor  # bf16 [6144]: raw Gemma w of the final norm
    eps: float


@dataclass
class DraftBackboneInputs:
    """Graph-stable step inputs, rewritten in place before every replay."""

    input_ids: torch.Tensor  # int64 [7]
    positions: torch.Tensor  # int64 [7]
    slot_mapping: torch.Tensor  # int64 [7]: cache slot of each row, -1 skips the write
    page_table: torch.Tensor  # int32 [max_pages]
    kv_len: torch.Tensor  # int32 [1]: cached tokens including the block
    cos_sin: torch.Tensor  # f32 [positions, 64]: cos | sin
    k_caches: list  # LAYERS x bf16 [slots, 128]
    v_caches: list  # LAYERS x bf16 [slots, 128]


class DraftBackbone:
    """The backbone bound to fixed weights, step buffers and output; ``run`` writes the
    final-normed hidden rows (bf16 [7, 6144], identical on every rank) and the block's
    K/V rows of every layer's cache."""

    def __init__(
        self,
        arena: host.Arena,
        weights: DraftBackboneWeights,
        inputs: DraftBackboneInputs,
        out: torch.Tensor,
        *,
        wait_timeout_ns: int = 10_000_000_000,
    ):
        """``wait_timeout_ns``: a wait on another CTA or rank traps after this long (0:
        never)."""
        device = arena.device
        if len(weights.layers) != LAYERS:
            raise ValueError(f"the draft backbone has {LAYERS} layers")
        if not weights.eps > 0:
            raise ValueError("eps must be positive")
        layers = []
        for index, layer in enumerate(weights.layers):
            tensors = [getattr(layer, f.name) for f in fields(layer)]
            tensors += [inputs.k_caches[index], inputs.v_caches[index]]
            for tensor, (name, dtype, shape) in zip(tensors, LAYER_TENSORS):
                host.check_tensor(tensor, f"layer {index} {name}", dtype, shape, device)
            if tensors[-1].shape[0] != tensors[-2].shape[0]:
                raise ValueError("K and V caches need the same slots")
            layers.append(tensors)
        step = [
            inputs.input_ids,
            weights.embed,
            out,
            weights.final_norm,
            inputs.positions,
            inputs.slot_mapping,
            inputs.page_table,
            inputs.kv_len,
            inputs.cos_sin,
        ]
        for tensor, (name, dtype, shape) in zip(step, STEP_TENSORS):
            host.check_tensor(tensor, name, dtype, shape, device)
        for tensor in [t for layer in layers for t in layer] + step:
            if tensor is not out and host.overlaps(out, tensor):
                raise ValueError("out must be disjoint from every operand")
        self.module = load_module(device.index, wait_timeout_ns)
        self.arena = arena
        self.tensors = (layers, step)  # the maps and pointers hold their addresses
        self.out = out
        args = Args()
        for index, tensors in enumerate(layers):
            for (name, _, _), tensor in zip(LAYER_TENSORS, tensors):
                setattr(args.layer[index], name, tensor.data_ptr())
        for (name, _, _), tensor in zip(STEP_TENSORS, step):
            setattr(args, name, tensor.data_ptr())
        for rank, peer in enumerate(arena.peers):
            args.arena[rank] = peer
        args.vocab_start = int(weights.vocab_start)
        args.vocab_rows = weights.embed.shape[0]
        args.rank = arena.rank
        args.max_pages = inputs.page_table.shape[0]
        args.eps = float(weights.eps)
        self.args = args
        self.maps = encode_maps(args)
        self.grid = torch.cuda.get_device_properties(device).multi_processor_count

    def run(self) -> torch.Tensor:
        """Enqueue the backbone on the current stream; graph-capturable."""
        host.launch(
            self.module.function,
            (self.grid, 1, 1),
            (THREADS, 1, 1),
            self.module.shared,
            [self.args, self.maps],
            torch.cuda.current_stream(self.arena.device).cuda_stream,
            cooperative=True,
        )
        return self.out
