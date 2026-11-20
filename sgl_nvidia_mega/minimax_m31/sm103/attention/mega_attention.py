# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MegaAttention: the attention of a MiniMax-M3.1 decode layer on TP4 in one persistent
148-CTA launch per rank (``kernel/mega_attention``): the input RMSNorm, the MXFP8 Q/K/V/QI/KI projection, the head
norms and RoPE, the new K/V/KI cache rows, the indexer's page scores and top-16 pages, the
sparse attention and its combine, the output projection, its TP4 all-reduce, the residual
add and the post-attention RMSNorm.

One request of eight physical rows (Q=1-8 as the last Q valid rows). Every CTA streams its
static item sequence through a shared-memory page ring and hands work to the others through
epoch-tagged records in the layer's arena, zeroed once. The TP4 reduction is a Lamport
mailbox in a multicast workspace every layer shares; calls sharing it run in stream order.
The kernel's parameter block is built here; its layout and launch constants are read from
the compiled kernel."""

from __future__ import annotations

import ctypes
import math
import struct
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import torch

from sgl_nvidia_mega.minimax_m31.sm103 import host, nvcc

KERNEL_ROOT = Path(__file__).resolve().parent / "kernel"
ROOTS = (KERNEL_ROOT / "mega_attention", KERNEL_ROOT)

ROWS, HIDDEN, HEAD_DIM, HEADS, PAGE, TOPK = 8, 6144, 128, 16, 128, 16
PROJECTION_ROWS = 2560  # 16 Q heads | K | V | indexer Q | indexer K per rank
O_PROJ_K = HEADS * HEAD_DIM
TILE_ROWS, TILE_K = 16, 2048
PAGE_DATA_BYTES, PAGE_SCALE_BYTES = PAGE * 64, PAGE * 8
MAX_PAGES = 8192  # 1M tokens of context

# The kernel's exact-IEEE contract (no FMA contraction, no flush to zero, IEEE division and
# square root) and the half and bfloat16 operator macros it was validated with
OPTIONS = (
    "-O3",
    "-std=c++20",
    "-lineinfo",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
    "--fmad=false",
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "--expt-relaxed-constexpr",
)

# Leaf fields of attn_ring::Params (kernel/mega_attention/include/sm100/common/ring_config.cuh)
# in declaration order: (name, kind); "desc" is a 128-byte CUtensorMap, "ptr4" the four head
# norm weights
FIELDS = (
    ("tm_proj", "desc"),
    ("tm_wo", "desc"),
    ("tm_ki", "desc"),
    ("tm_k", "desc"),
    ("layer_input", "ptr"),
    ("input_norm", "ptr"),
    ("head_norm", "ptr4"),
    ("post_norm", "ptr"),
    ("cos_sin", "ptr"),
    ("positions", "ptr"),
    ("slot_mapping", "ptr"),
    ("kv_len", "ptr"),
    ("proj_sf", "ptr"),
    ("wo_sf", "ptr"),
    ("k_data", "ptr"),
    ("k_scale", "ptr"),
    ("v_data", "ptr"),
    ("v_scale", "ptr"),
    ("ki_data", "ptr"),
    ("ki_scale", "ptr"),
    ("k_data_stride", "i64"),
    ("k_scale_stride", "i64"),
    ("v_data_stride", "i64"),
    ("v_scale_stride", "i64"),
    ("ki_data_stride", "i64"),
    ("ki_scale_stride", "i64"),
    ("kv_indices", "ptr"),
    ("indexer_table", "ptr"),
    ("q_out", "ptr"),
    ("qi_out", "ptr"),
    ("kv_block_indexes", "ptr"),
    ("o_out", "ptr"),
    ("o_sf_out", "ptr"),
    ("attention_residual", "ptr"),
    ("post_attention_norm", "ptr"),
    ("arena", "ptr"),
    ("tp_unicast", "ptr"),
    ("tp_multicast", "ptr"),
    ("tp_payload_offset", "u64"),
    ("tp_publish_offset", "u64"),
    ("tp_consume_offset", "u64"),
    ("tp_rank", "i32"),
    ("input_eps", "f32"),
    ("head_eps", "f32"),
    ("post_eps", "f32"),
    ("k_global", "f32"),
    ("v_global", "f32"),
    ("ki_global", "f32"),
    ("softmax_log2_scale", "f32"),
    ("v_out_scale", "f32"),
    ("wait_timeout_ns", "u64"),
)
PACK = {"ptr": "<Q", "i32": "<i", "u64": "<Q", "i64": "<q", "f32": "<f"}
MAILBOX = "mega_atten::m3_decode::tp4_mailbox"
# Launch, arena and mailbox constants read from the compiled kernel, in this order
CONSTANTS = (
    ("params_bytes", "sizeof(Params)"),
    ("grid", "kGrid"),
    ("threads", "kBlock"),
    ("smem_bytes", "kDynamicSmemAlloc"),
    ("arena_bytes", "arena::kBytes"),
    ("workspace_bytes", f"{MAILBOX}::kWorkspaceBytes"),
    ("payload_offset", f"{MAILBOX}::kPayloadOffset"),
    ("payload_bytes", f"{MAILBOX}::kPayloadBytes"),
    ("publish_offset", f"{MAILBOX}::kPublishEpochOffset"),
    ("consume_offset", f"{MAILBOX}::kConsumeEpochOffset"),
    ("mailbox_empty", f"{MAILBOX}::kEmpty"),
)
_LAYOUT = "\n".join(
    [f"    (long long)({expression})," for _, expression in CONSTANTS]
    + [f"    (long long)offsetof(Params, {name})," for name, _ in FIELDS]
)
# The kernel's production instantiation (the one with every wait bounded is not built), then
# the layout and constants the host reads, with the TP4 mailbox of tp4_mailbox_layout.hpp
SOURCE = f"""#include "include/sm100/kernel/attn_ring_kernel.cuh"
#include <cstddef>
#include "common/kernel/include/tp4_mailbox_layout.hpp"
namespace attn_ring {{
template __global__ void attn_ring_kernel<false>(const __grid_constant__ Params);
static_assert({MAILBOX}::kGenerationCount == mailbox::kGenerations);
static_assert({MAILBOX}::kClusterCtas == mailbox::kOwnersPerToken);
extern "C" __device__ long long minimax_m31_mega_attention_layout[] = {{
{_LAYOUT}
}};
}}
"""


@dataclass(frozen=True)
class Layout:
    """The compiled kernel's parameter-block offsets and launch constants."""

    offsets: dict
    params_bytes: int
    grid: int
    threads: int
    smem_bytes: int
    arena_bytes: int
    workspace_bytes: int
    payload_offset: int
    payload_bytes: int
    publish_offset: int
    consume_offset: int
    mailbox_empty: int


class Kernel:
    """The compiled kernel and its layout."""

    def __init__(self, device: torch.device, wait_timeout_ns: int):
        options = (
            *OPTIONS,
            f"-DMEGA_ATTENTION_WAIT_TIMEOUT_NS={int(wait_timeout_ns)}ull",
        )
        cubin = nvcc.compile_cubin(
            "minimax_m31_mega_attention",
            SOURCE,
            ROOTS,
            nvcc.device_arch(device),
            options,
            nvcc.QMUL4_VERSION,
        )
        (symbol,) = [n for n in nvcc.kernel_names(cubin) if "attn_ring_kernel" in n]
        self.module = module = host.Module(cubin, symbol)
        values = module.array(
            "minimax_m31_mega_attention_layout", len(CONSTANTS) + len(FIELDS)
        )
        constants = dict(zip((name for name, _ in CONSTANTS), values[: len(CONSTANTS)]))
        offsets = dict(zip((name for name, _ in FIELDS), values[len(CONSTANTS) :]))
        self.layout = layout = Layout(offsets=offsets, **constants)
        if module.parameter_size(0) != layout.params_bytes:
            raise RuntimeError("the kernel's parameter block does not match its layout")
        module.set_shared_memory(layout.smem_bytes)
        # the persistent schedule spins on cross-CTA counters: all CTAs must be resident
        if (
            module.resident_ctas(device, layout.threads, layout.smem_bytes)
            < layout.grid
        ):
            raise RuntimeError(
                f"the device cannot hold all {layout.grid} MegaAttention CTAs at once"
            )


@cache
def load_kernel(device_index: int, wait_timeout_ns: int = 0) -> Kernel:
    """Compile (or read from the cache) and load the kernel for this device; a wait on
    another rank traps after ``wait_timeout_ns`` (0: never)."""
    device = torch.device("cuda", device_index)
    with torch.cuda.device(device):
        torch.zeros(1, device=device)  # make the primary context current
        return Kernel(device, wait_timeout_ns)


def create_workspace(group, device, wait_timeout_ns: int = 0) -> host.Arena:
    """The TP4 multicast workspace every layer shares; collective. Zeroed, with the mailbox
    slots set to the Lamport sentinel (BF16 negative zero). ``wait_timeout_ns`` is the
    layers' (the layout is read from their kernel)."""
    device = host.cuda_device(device)
    layout = host.agree(
        group,
        lambda: load_kernel(device.index, wait_timeout_ns).layout,
        "building MegaAttention",
    )

    def fill(buffer: torch.Tensor) -> None:
        buffer.zero_()
        payload = buffer[
            layout.payload_offset : layout.payload_offset + layout.payload_bytes
        ]
        payload.view(torch.int16).fill_(layout.mailbox_empty - (1 << 16))

    arena = host.Arena(group, device, max(1 << 20, layout.workspace_bytes), fill=fill)
    if not arena.multicast or arena.multicast % 16:
        raise RuntimeError("NVLink multicast mapping is unavailable")
    return arena


def new_arena(device, wait_timeout_ns: int = 0) -> torch.Tensor:
    """One layer's zeroed hand-off records, kept by every MegaAttention of the layer;
    ``wait_timeout_ns`` is the layer's (the size is read from its kernel)."""
    device = host.cuda_device(device)
    layout = load_kernel(device.index, wait_timeout_ns).layout
    return torch.zeros(layout.arena_bytes, dtype=torch.uint8, device=device)


@dataclass
class AttentionWeights:
    """One layer's rank-local weights: MXFP8 GEMM weights with their UE8M0 scales in the
    128x4-swizzled layout, and the raw Gemma RMSNorm weights (applied as 1 + w)."""

    # e4m3 [2560, 6144]: q (16 heads) | k | v | indexer q | indexer k
    projection: torch.Tensor
    projection_sf: torch.Tensor  # u8 [2560 * 192]
    output_projection: torch.Tensor  # e4m3 [6144, 2048]
    output_projection_sf: torch.Tensor  # u8 [6144 * 64]
    input_norm: torch.Tensor  # bf16 [6144]
    q_norm: torch.Tensor  # bf16 [128]
    k_norm: torch.Tensor  # bf16 [128]
    qi_norm: torch.Tensor  # bf16 [128]
    ki_norm: torch.Tensor  # bf16 [128]
    post_attention_norm: torch.Tensor  # bf16 [6144]


@dataclass
class AttentionCache:
    """One layer's paged NVFP4 cache: ``k``/``v``/``ki`` u8 [P, 1, 128, 64] (or [P, 128, 64])
    and E4M3 ``k_scale``/``v_scale``/``ki_scale`` [P, 1, 128, 8] (or [P, 128, 8]); rows dense
    inside a page, ``stride(0)`` the page stride in bytes."""

    k: torch.Tensor
    v: torch.Tensor
    ki: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    ki_scale: torch.Tensor


@dataclass
class AttentionStep:
    """Graph-stable step inputs shared by every layer, rewritten before each replay."""

    positions: torch.Tensor  # int64 [8]
    # int64 [8]: cache slot (page * 128 + row), -1 for padding
    slot_mapping: torch.Tensor
    # int32 [1]: cached tokens including the step's rows, in [8, 128 * S]; the kernel
    # reads it on device (graph replays), so the caller keeps it inside the page tables
    kv_len: torch.Tensor
    kv_indices: torch.Tensor  # int32 [1, S]: logical -> physical page (attention)
    page_table: torch.Tensor  # int32 [>= S]: logical -> physical page (indexer)
    cos_sin: (
        torch.Tensor
    )  # bf16 [positions, 64]: cos (32) | sin (32) of the rotary dims


class AttentionScratch:
    """The attention's quantized Q / QI / output rows and selected pages; written by every
    call, so the layers, which run in stream order, may share one."""

    def __init__(self, device):
        device = host.cuda_device(device)
        fp8 = torch.float8_e4m3fn
        self.q = torch.empty((ROWS, HEADS, HEAD_DIM), dtype=fp8, device=device)
        self.qi = torch.empty((ROWS, HEAD_DIM), dtype=fp8, device=device)
        self.o = torch.empty((ROWS, HEADS, HEAD_DIM), dtype=fp8, device=device)
        # UE8M0 scale of (head, row, 32-element group) at head * 512 + row * 16 + group
        self.o_sf = torch.empty(HEADS * 512, dtype=torch.uint8, device=device)
        self.kv_block_indexes = torch.full(
            (ROWS, 1, TOPK), -1, dtype=torch.int32, device=device
        )

    def tensors(self) -> list[torch.Tensor]:
        return [self.q, self.qi, self.o, self.o_sf, self.kv_block_indexes]


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _check_cache(
    name: str, tensor: torch.Tensor, row_bytes: int, page_bytes: int, pages: int
):
    dtypes = (torch.uint8,) if row_bytes == 64 else (torch.uint8, torch.float8_e4m3fn)
    if tensor.dtype not in dtypes:
        raise ValueError(f"{name} must be {' or '.join(map(str, dtypes))} storage")
    tensor = tensor.view(torch.uint8)
    if not (
        tensor.dim() in (3, 4)
        and tensor.size(-1) == row_bytes
        and tensor.size(-2) == PAGE
        and (tensor.dim() == 3 or tensor.size(1) == 1)
        and tensor.size(0) == pages
        and tensor.stride(-1) == 1
        and tensor.stride(-2) == row_bytes
        and page_bytes <= tensor.stride(0) < 1 << 40
        and tensor.stride(0) % 16 == 0
        and tensor.data_ptr() % 16 == 0
    ):
        raise ValueError(
            f"{name} must hold {pages} pages of dense [(1,) 128, {row_bytes}] byte rows, its "
            "page stride a multiple of 16 bytes covering a page, 16-byte aligned"
        )


class MegaAttention:
    """One layer's attention bound to its weights, cache, the shared step inputs and scratch,
    its input rows and its outputs (the attention residual and the post-attention norm, BF16
    [8, 6144]); ``run`` enqueues it."""

    def __init__(
        self,
        workspace: host.Arena,
        weights: AttentionWeights,
        cache: AttentionCache,
        step: AttentionStep,
        scratch: AttentionScratch,
        layer_input: torch.Tensor,
        attention_residual: torch.Tensor,
        post_attention_norm: torch.Tensor,
        *,
        sm_scale: float = 1.0 / math.sqrt(HEAD_DIM),
        input_norm_eps: float = 1e-6,
        head_norm_eps: float = 1e-6,
        post_attention_norm_eps: float = 1e-6,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        ki_scale: float = 1.0,
        wait_timeout_ns: int = 0,
        arena: torch.Tensor | None = None,
    ):
        """``wait_timeout_ns``: a wait on another rank traps after this long (0: never).
        ``arena``: the layer's zeroed hand-off records (``new_arena``), else allocated."""
        device = workspace.device
        self.kernel = kernel = load_kernel(device.index, wait_timeout_ns)
        layout = kernel.layout
        _check(
            weights,
            cache,
            step,
            scratch,
            layer_input,
            attention_residual,
            post_attention_norm,
            device,
        )
        for value in (
            sm_scale,
            input_norm_eps,
            head_norm_eps,
            post_attention_norm_eps,
            k_scale,
            v_scale,
            ki_scale,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("scales and epsilons must be finite and positive")
        if workspace.buffer.numel() < layout.workspace_bytes:
            raise ValueError("the TP4 workspace is smaller than the mailbox layout")
        self.workspace = workspace
        # rank-private hand-off records; every call advances their epochs
        if arena is None:
            arena = new_arena(device, wait_timeout_ns)
        host.check_tensor(arena, "arena", torch.uint8, (layout.arena_bytes,), device)
        self.arena = arena
        # the descriptors and pointers hold these addresses
        self.tensors = (
            weights,
            cache,
            step,
            scratch,
            layer_input,
            attention_residual,
            post_attention_norm,
        )
        u8 = {
            name: getattr(cache, name).view(torch.uint8)
            for name in ("k", "v", "ki", "k_scale", "v_scale", "ki_scale")
        }
        head_norms = (weights.q_norm, weights.k_norm, weights.qi_norm, weights.ki_norm)
        values = {
            "layer_input": layer_input.data_ptr(),
            "input_norm": weights.input_norm.data_ptr(),
            "head_norm": [t.data_ptr() for t in head_norms],
            "post_norm": weights.post_attention_norm.data_ptr(),
            "cos_sin": step.cos_sin.data_ptr(),
            "positions": step.positions.data_ptr(),
            "slot_mapping": step.slot_mapping.data_ptr(),
            "kv_len": step.kv_len.data_ptr(),
            "proj_sf": weights.projection_sf.data_ptr(),
            "wo_sf": weights.output_projection_sf.data_ptr(),
            **{f"{n}_data": u8[n].data_ptr() for n in ("k", "v", "ki")},
            **{f"{n}_scale": u8[f"{n}_scale"].data_ptr() for n in ("k", "v", "ki")},
            **{f"{n}_data_stride": u8[n].stride(0) for n in ("k", "v", "ki")},
            **{
                f"{n}_scale_stride": u8[f"{n}_scale"].stride(0)
                for n in ("k", "v", "ki")
            },
            "kv_indices": step.kv_indices.data_ptr(),
            "indexer_table": step.page_table.data_ptr(),
            "q_out": scratch.q.data_ptr(),
            "qi_out": scratch.qi.data_ptr(),
            "kv_block_indexes": scratch.kv_block_indexes.data_ptr(),
            "o_out": scratch.o.data_ptr(),
            "o_sf_out": scratch.o_sf.data_ptr(),
            "attention_residual": attention_residual.data_ptr(),
            "post_attention_norm": post_attention_norm.data_ptr(),
            "arena": self.arena.data_ptr(),
            "tp_unicast": workspace.base,
            "tp_multicast": workspace.multicast,
            "tp_payload_offset": layout.payload_offset,
            "tp_publish_offset": layout.publish_offset,
            "tp_consume_offset": layout.consume_offset,
            "tp_rank": workspace.rank,
            "input_eps": input_norm_eps,
            "head_eps": head_norm_eps,
            "post_eps": post_attention_norm_eps,
            "k_global": k_scale,
            "v_global": v_scale,
            "ki_global": ki_scale,
            # scale_q * scale_k * sm_scale * log2(e), each product rounded to float (scale_q = 1)
            "softmax_log2_scale": _f32(
                _f32(_f32(k_scale) * _f32(sm_scale)) * _f32(1.4426950408889634)
            ),
            "v_out_scale": v_scale,
            "wait_timeout_ns": 0,  # the bound of the instantiation that is not built
        }
        maps = {
            "tm_proj": _weight_map(weights.projection, PROJECTION_ROWS, HIDDEN),
            "tm_wo": _weight_map(weights.output_projection, HIDDEN, O_PROJ_K),
            "tm_ki": _page_map(u8["ki"]),
            "tm_k": _page_map(u8["k"]),
        }
        blob = (ctypes.c_uint8 * layout.params_bytes)()
        for name, kind in FIELDS:
            offset = layout.offsets[name]
            if kind == "desc":
                host.place_map(blob, offset, maps[name])
                continue
            if kind == "ptr4":
                data = struct.pack("<4Q", *values[name])
            else:
                data = struct.pack(PACK[kind], values[name])
            ctypes.memmove(ctypes.addressof(blob) + offset, data, len(data))
        self.blob = blob
        self.outputs = (attention_residual, post_attention_norm)

    def run(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Enqueue the layer on the current stream; graph-capturable."""
        layout = self.kernel.layout
        host.launch(
            self.kernel.module.function,
            (layout.grid, 1, 1),
            (layout.threads, 1, 1),
            layout.smem_bytes,
            [self.blob],
            torch.cuda.current_stream(self.workspace.device).cuda_stream,
        )
        return self.outputs


def _weight_map(weight: torch.Tensor, rows: int, k: int):
    """Row-major [rows, K] bytes viewed as [K / 128][rows][128 B]: one box of 16 K chunks x
    16 rows, 128-byte swizzled, per weight tile."""
    return host.encode(
        "UINT8",
        weight.data_ptr(),
        (128, rows, k // 128),
        (k, 128),
        (128, TILE_ROWS, TILE_K // 128),
        "128B",
        "L2_256B",
    )


def _page_map(cache: torch.Tensor):
    """One packed E2M1 page [128 rows][64 B] viewed as [64][128 B], pages stride(0) apart."""
    return host.encode(
        "UINT8",
        cache.data_ptr(),
        (128, 64, cache.shape[0]),
        (128, cache.stride(0)),
        (128, 64, 1),
        "128B",
        "L2_256B",
    )


def _check(
    weights,
    cache,
    step,
    scratch,
    layer_input,
    attention_residual,
    post_attention_norm,
    device,
) -> None:
    """The operand contract the kernel relies on."""
    fp8, u8, bf16 = torch.float8_e4m3fn, torch.uint8, torch.bfloat16
    host.check_tensor(
        weights.projection, "projection", fp8, (PROJECTION_ROWS, HIDDEN), device
    )
    host.check_tensor(
        weights.projection_sf,
        "projection_sf",
        u8,
        (PROJECTION_ROWS * (HIDDEN // 32),),
        device,
    )
    host.check_tensor(
        weights.output_projection, "output_projection", fp8, (HIDDEN, O_PROJ_K), device
    )
    host.check_tensor(
        weights.output_projection_sf,
        "output_projection_sf",
        u8,
        (HIDDEN * (O_PROJ_K // 32),),
        device,
    )
    for name in ("input_norm", "post_attention_norm"):
        host.check_tensor(getattr(weights, name), name, bf16, (HIDDEN,), device)
    for name in ("q_norm", "k_norm", "qi_norm", "ki_norm"):
        host.check_tensor(getattr(weights, name), name, bf16, (HEAD_DIM,), device)
    pages = cache.k.shape[0]
    for name, row_bytes, page_bytes in (
        ("k", 64, PAGE_DATA_BYTES),
        ("v", 64, PAGE_DATA_BYTES),
        ("ki", 64, PAGE_DATA_BYTES),
        ("k_scale", 8, PAGE_SCALE_BYTES),
        ("v_scale", 8, PAGE_SCALE_BYTES),
        ("ki_scale", 8, PAGE_SCALE_BYTES),
    ):
        tensor = getattr(cache, name)
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}")
        _check_cache(name, tensor, row_bytes, page_bytes, pages)
    host.check_tensor(step.positions, "positions", torch.int64, (ROWS,), device)
    host.check_tensor(step.slot_mapping, "slot_mapping", torch.int64, (ROWS,), device)
    host.check_tensor(step.kv_len, "kv_len", torch.int32, (1,), device)
    host.check_tensor(step.kv_indices, "kv_indices", torch.int32, (1, -1), device)
    host.check_tensor(step.page_table, "page_table", torch.int32, (-1,), device)
    host.check_tensor(step.cos_sin, "cos_sin", bf16, (-1, 64), device)
    if not 0 < step.kv_indices.shape[1] <= MAX_PAGES:
        raise ValueError(f"kv_indices must cover 1 to {MAX_PAGES} pages")
    if step.page_table.numel() < step.kv_indices.shape[1]:
        raise ValueError("page_table must cover every logical KV page")
    host.check_tensor(scratch.q, "q", fp8, (ROWS, HEADS, HEAD_DIM), device)
    host.check_tensor(scratch.qi, "qi", fp8, (ROWS, HEAD_DIM), device)
    host.check_tensor(scratch.o, "output", fp8, (ROWS, HEADS, HEAD_DIM), device)
    host.check_tensor(
        scratch.kv_block_indexes,
        "kv_block_indexes",
        torch.int32,
        (ROWS, 1, TOPK),
        device,
    )
    if (
        scratch.o_sf.dtype != u8
        or not scratch.o_sf.is_contiguous()
        or scratch.o_sf.numel() < HEADS * 512
    ):
        raise ValueError(
            f"o_sf must be contiguous uint8 storage of at least {HEADS * 512} bytes"
        )
    for name, tensor in (
        ("layer_input", layer_input),
        ("attention_residual", attention_residual),
        ("post_attention_norm", post_attention_norm),
    ):
        host.check_tensor(tensor, name, bf16, (ROWS, HIDDEN), device)
    outputs = [*scratch.tensors(), attention_residual, post_attention_norm]
    inputs = [layer_input, step.positions, step.slot_mapping, step.cos_sin]
    if any(
        host.overlaps(t, o)
        for i, t in enumerate(outputs)
        for o in outputs[i + 1 :] + inputs
    ):
        raise ValueError(
            "the attention's outputs must not overlap each other or its inputs"
        )
