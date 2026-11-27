# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import ctypes
import math
from collections import OrderedDict
from functools import cache
from pathlib import Path
from typing import Optional

import torch

from sgl_nvidia_mega.common import jit

__all__ = [
    "MAX_ROWS",
    "SEED_BASE",
    "box_rows_for",
    "build",
    "epoch_seeds",
    "head_mma_keys",
    "programs",
    "variant",
]

SEED_BASE = 0x5EED6A3B << 20


def epoch_seeds(epoch: int, rows: int, device) -> torch.Tensor:
    return (
        SEED_BASE
        + int(epoch) * 65536
        + torch.arange(rows, dtype=torch.int64, device=device)
    )


SRC = Path(__file__).parent / "src" / "head_mma.cu"
HIDDEN = 4096
BLOCK_K = 64
ROWS_TILE = 128
THREADS = 192
MAX_ROWS = 512
SMEM_LIMIT = 220 * 1024
MAX_STAGES = 12
MAP_CACHE = 512

_maps: OrderedDict[tuple, object] = OrderedDict()


class HeadMmaParams(ctypes.Structure):
    _fields_ = [
        ("keys", ctypes.c_void_p),
        ("done", ctypes.c_void_p),
        ("logits", ctypes.c_void_p),
        ("logits_stride", ctypes.c_int64),
        ("m", ctypes.c_int32),
        ("valid", ctypes.c_int32),
        ("vocab_start", ctypes.c_int32),
        ("box_rows", ctypes.c_int32),
        ("nan_hi", ctypes.c_int32),
        ("keys_stride", ctypes.c_int64),
        ("seeds", ctypes.c_void_p),
        ("pre", ctypes.c_void_p),
        ("epoch", ctypes.c_void_p),
    ]


def smem_bytes(n_tile: int, chunks: int, stages: int, kinds: int = 1) -> int:
    tokens = n_tile * chunks
    raw = stages * ROWS_TILE * BLOCK_K * 2 + stages * chunks * n_tile * BLOCK_K * 2
    raw += 2 * kinds * 4 * tokens * 4 + 2 * stages * 8 + 8 + 4
    return math.ceil(raw / 1024) * 1024


def variant(
    m: int, cluster: Optional[int] = None, stages: Optional[int] = None
) -> tuple[int, int, int, int]:
    """(N tile, token tiles, stages, cluster) serving ``m`` rows."""
    if not 1 <= m <= MAX_ROWS:
        raise ValueError(f"head_mma serves 1-{MAX_ROWS} rows, got {m}")
    chunks = 1 if m <= 256 else 2
    n_tile = math.ceil(math.ceil(m / chunks) / 16) * 16
    if stages is None:
        stages = MAX_STAGES
        while stages > 2 and smem_bytes(n_tile, chunks, stages) > SMEM_LIMIT:
            stages -= 1
    if cluster is None:
        cluster = 1
    if cluster not in (1, 2, 4) or (n_tile // cluster) % 8:
        cluster = 1
    return n_tile, chunks, stages, cluster


def box_rows_for(rows: int, sms: int) -> int:
    return min(ROWS_TILE, math.ceil(math.ceil(rows / sms) / 8) * 8)


def programs(rows: int, sms: int, m: int = 1, cluster: Optional[int] = None) -> int:
    """CTAs of one launch over a ``rows``-row shard (the early-release ticket count)."""
    c = variant(m, cluster)[3]
    return math.ceil(math.ceil(rows / box_rows_for(rows, sms)) / c) * c


@cache
def _kernel(
    device_index: int,
    n_tile: int,
    chunks: int,
    stages: int,
    cluster: int,
    keys: bool = True,
    gumbel1: bool = False,
    noise: bool = True,
) -> jit.Kernel:
    device = torch.device("cuda", device_index)
    torch.empty(1, device=device)
    major, minor = torch.cuda.get_device_capability(device)
    smem = smem_bytes(n_tile, chunks, stages)
    cubin = jit.compile_source(
        SRC.read_text(),
        f"q35_head_mma_n{n_tile}x{chunks}_s{stages}_c{cluster}"
        + ("" if keys else "_logits")
        + ("_gumbel1" if gumbel1 else "")
        + ("" if noise else "_nonoise"),
        f"sm_{major}{minor}a",
        (
            "-std=c++17",
            f"-DHEAD_N_TILE={n_tile}",
            f"-DHEAD_N_CHUNKS={chunks}",
            f"-DHEAD_STAGES={stages}",
            f"-DHEAD_CLUSTER={cluster}",
            f"-DHEAD_SMEM_BYTES={smem}",
            f"-DHEAD_KEYS={int(keys)}",
            f"-DHEAD_GUMBEL1={int(gumbel1)}",
            f"-DHEAD_NOISE={int(noise)}",
            *(("--ftz=true",) if gumbel1 else ()),
        ),
    )
    return jit.load(cubin, "head_mma_keys", (THREADS, 1, 1), smem)


def build(device_index: int, rows: tuple[int, ...]) -> None:
    """Compile and load the variants serving each row count (no compile in a capture)."""
    for m in rows:
        _kernel(device_index, *variant(m))


def _map(base: int, rows: int, row_stride_bytes: int, box_rows: int):
    key = (base, rows, row_stride_bytes, box_rows)
    tensor_map = _maps.get(key)
    if tensor_map is not None:
        _maps.move_to_end(key)
        return tensor_map
    cu = jit.driver()
    u64, u32 = cu.cuuint64_t, cu.cuuint32_t
    (tensor_map,) = jit.check(
        cu.cuTensorMapEncodeTiled(
            cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            base,
            [u64(HIDDEN), u64(rows)],
            [u64(row_stride_bytes)],
            [u32(BLOCK_K), u32(box_rows)],
            [u32(1)] * 2,
            cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
            cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        ),
        "cuTensorMapEncodeTiled(head_mma)",
    )
    _maps[key] = tensor_map
    if len(_maps) > MAP_CACHE:
        _maps.popitem(last=False)
    return tensor_map


def head_mma_keys(
    x: torch.Tensor,
    w: torch.Tensor,
    keys: Optional[torch.Tensor],
    *,
    vocab_start: int,
    valid: int,
    nan_hi: int,
    logits: Optional[torch.Tensor] = None,
    done: Optional[torch.Tensor] = None,
    box_rows: Optional[int] = None,
    cluster: Optional[int] = None,
    stages: Optional[int] = None,
    seeds: Optional[torch.Tensor] = None,
    pre: Optional[torch.Tensor] = None,
    noise: bool = True,
    epoch: Optional[torch.Tensor] = None,
) -> int:
    if keys is None and (logits is None or done is not None):
        raise ValueError(
            "head_mma without keys writes logits only (logits given, no done ticket)"
        )
    gumbel1 = pre is not None
    if gumbel1 and (
        (seeds is None) == (epoch is None)
        or logits is None
        or keys is not None
        or done is not None
    ):
        raise ValueError(
            "the Gumbel build takes seeds or an epoch, pre and logits (no keys, no done ticket)"
        )
    m, rows = int(x.shape[0]), int(w.shape[0])
    device = x.device
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    box = box_rows_for(rows, sms) if box_rows is None else box_rows
    if box % 8 or not 8 <= box <= ROWS_TILE:
        raise ValueError(f"head_mma box rows {box}")
    if x.data_ptr() % 16 or (x.stride(0) * 2) % 16 or w.data_ptr() % 16:
        raise ValueError("head_mma needs 16-byte aligned x / w rows (TMA)")
    n_tile, chunks, stages, c = variant(m, cluster, stages)
    if gumbel1:
        if c != 1 or chunks != 1 or n_tile > 128:
            raise ValueError(
                f"the Gumbel build serves <= 128 rows at cluster 1, got {m} rows cluster {c}"
            )
        grid_g = math.ceil(rows / box)
        if (
            pre.dtype != torch.int64
            or not pre.is_contiguous()
            or pre.numel() < 2 * m * grid_g + 1
        ):
            raise ValueError(
                f"pre must be contiguous int64 [2 * {m} * {grid_g} + 1] (keys, epoch tag)"
            )
        if seeds is not None and (
            seeds.dtype != torch.int64 or not seeds.is_contiguous() or seeds.numel() < m
        ):
            raise ValueError(f"seeds must be contiguous int64 [>= {m}]")
        if epoch is not None and (
            epoch.dtype != torch.int32 or not epoch.is_contiguous() or epoch.numel() < 4
        ):
            raise ValueError(
                "epoch must be contiguous int32 [4] (epoch, head ticket 0, keys ticket 0, 0)"
            )
    kernel = _kernel(
        device.index,
        n_tile,
        chunks,
        stages,
        c,
        keys is not None or gumbel1,
        gumbel1,
        noise,
    )
    w_map = _map(w.data_ptr(), rows, HIDDEN * 2, box)
    x_map = _map(x.data_ptr(), m, x.stride(0) * 2, n_tile // c)
    p = HeadMmaParams(
        keys=0 if keys is None else keys.data_ptr(),
        done=0 if done is None else done.data_ptr(),
        logits=0 if logits is None else logits.data_ptr(),
        logits_stride=0 if logits is None else logits.stride(0),
        m=m,
        valid=valid,
        vocab_start=vocab_start,
        box_rows=box,
        nan_hi=nan_hi,
        keys_stride=0 if keys is None else keys.stride(0),
        seeds=0 if seeds is None else seeds.data_ptr(),
        pre=0 if pre is None else pre.data_ptr(),
        epoch=0 if epoch is None else epoch.data_ptr(),
    )
    grid = math.ceil(math.ceil(rows / box) / c) * c
    stream = torch.cuda.current_stream(device).cuda_stream
    jit.launch(
        kernel,
        (grid, 1, 1),
        [w_map, x_map, p],
        stream,
        cluster=(c, 1, 1) if c > 1 else None,
    )
    return grid
