# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import ctypes
import math
from functools import cache
from pathlib import Path

import torch

from sgl_nvidia_mega.common import jit

__all__ = [
    "HeadTop1Exchange",
    "MAX_ROWS",
    "MAX_WORLD",
    "build",
]

SRC = Path(__file__).parent / "src"
_DRAFT_SRC = Path(__file__).parent.parent / "draft" / "src"
# head_xchg::kSlots / kMaxWorld / kMaxRows / kRowsPerCta / kEpochWords
SLOTS = 3
MAX_WORLD = 32
MAX_ROWS = 512
ROWS_PER_CTA = 8
EPOCH_WORDS = 2 + MAX_ROWS // ROWS_PER_CTA
THREADS = ROWS_PER_CTA * 32
KEY_LINE = 16


def _epoch(instances: int, device) -> torch.Tensor:
    e = torch.ones((instances, EPOCH_WORDS), dtype=torch.int32, device=device)
    e[:, 1] = 0
    return e


class HeadXchgParams(ctypes.Structure):
    """sglang::HeadXchgParams."""

    _fields_ = [
        ("peers", ctypes.c_uint64 * MAX_WORLD),
        ("keys", ctypes.c_void_p),
        ("win_keys", ctypes.c_void_p),
        ("done", ctypes.c_void_p),
        ("epoch", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("topk_p", ctypes.c_void_p),
        ("positions", ctypes.c_void_p),
        ("mrope", ctypes.c_void_p),
        ("mrope_stride", ctypes.c_int64),
        ("keys_stride", ctypes.c_int64),
        ("key_row_stride", ctypes.c_int64),
        ("out_inst_stride", ctypes.c_int64),
        ("out_stride", ctypes.c_int64),
        ("m", ctypes.c_int32),
        ("world", ctypes.c_int32),
        ("rank", ctypes.c_int32),
        ("column", ctypes.c_int32),
        ("head_parts", ctypes.c_uint32),
    ]


@cache
def _kernel(device_index: int) -> jit.Kernel:
    major, minor = torch.cuda.get_device_capability(device_index)
    cubin = jit.compile_source(
        (SRC / "head_xchg.cu").read_text(),
        "q35_head_xchg",
        f"sm_{major}{minor}a",
        ("-std=c++17", "-default-device"),
        headers={"mkd_common.cuh": (_DRAFT_SRC / "mkd_common.cuh").read_text()},
    )
    return jit.load(cubin, "head_xchg", (THREADS, 1, 1), 0)


def build(device_index: int) -> None:
    torch.empty(1, device=torch.device("cuda", device_index))
    _kernel(device_index)


def buffer_words(world: int) -> int:
    return SLOTS * world * MAX_ROWS * 2


def _check_world(world: int) -> None:
    if not 1 <= world <= MAX_WORLD:
        raise ValueError(f"head top-1 exchange: world {world} (1-{MAX_WORLD})")


def _check_out(out: torch.Tensor, m: int, column: int) -> None:
    if (
        out.dtype != torch.int64
        or out.dim() != 2
        or out.shape[0] < m
        or not 0 <= column < out.shape[1]
    ):
        raise ValueError(
            f"head top-1 exchange: out {out.dtype} {tuple(out.shape)} column {column}"
        )


def _check_draft_outputs(m, topk_p, positions, mrope_positions) -> None:
    for name, t in (("topk_p", topk_p), ("positions", positions)):
        if t is not None and (t.numel() < m or not t.is_contiguous()):
            raise ValueError(f"head top-1 exchange: {name} {tuple(t.shape)}")
    if topk_p is not None and topk_p.dtype != torch.float32:
        raise ValueError(f"head top-1 exchange: topk_p {topk_p.dtype}")
    if positions is not None and positions.dtype != torch.int64:
        raise ValueError(f"head top-1 exchange: positions {positions.dtype}")
    if mrope_positions is not None and (
        mrope_positions.dtype != torch.int64
        or mrope_positions.shape[0] != 3
        or mrope_positions.shape[1] < m
        or mrope_positions.stride(1) != 1
    ):
        raise ValueError(f"head top-1 exchange: M-RoPE {tuple(mrope_positions.shape)}")


def _launch(
    params: HeadXchgParams, grid: tuple[int, int], device: torch.device
) -> None:
    stream = torch.cuda.current_stream(device).cuda_stream
    jit.launch(_kernel(device.index), (grid[0], grid[1], 1), [params], stream)


class HeadTop1Exchange:
    def __call__(
        self,
        m: int,
        out: torch.Tensor,
        *,
        column: int = 0,
        topk_p: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        mrope_positions: torch.Tensor | None = None,
    ) -> None:
        if not 1 <= m <= MAX_ROWS:
            raise ValueError(f"head top-1 exchange: {m} rows (1-{MAX_ROWS})")
        _check_out(out, m, column)
        _check_draft_outputs(m, topk_p, positions, mrope_positions)
        p = HeadXchgParams()
        for r, ptr in enumerate(self.peers):
            p.peers[r] = ptr
        p.keys = self.keys.data_ptr()
        p.win_keys = 0
        p.done = 0
        p.epoch = self.epoch.data_ptr()
        p.out = out.data_ptr()
        p.topk_p = 0 if topk_p is None else topk_p.data_ptr()
        p.positions = 0 if positions is None else positions.data_ptr()
        p.mrope = 0 if mrope_positions is None else mrope_positions.data_ptr()
        p.mrope_stride = 0 if mrope_positions is None else mrope_positions.stride(0)
        p.keys_stride = 0
        p.key_row_stride = self.keys.stride(-1)
        p.out_inst_stride = 0
        p.out_stride = out.stride(0)
        p.m, p.world, p.rank, p.column = m, self.world, self.rank, column
        p.head_parts = 0
        _launch(p, (math.ceil(m / ROWS_PER_CTA), 1), self.device)
