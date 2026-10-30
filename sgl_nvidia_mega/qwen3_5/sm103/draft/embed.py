# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The embedding rows of a multi-row MTP draft step (``src/draft_embed.cu``): the rank whose
vocabulary shard holds a token multicasts its table row, in one launch, in place of the
vocabulary-parallel embedding's masked lookup and all-reduce (the same rows: -0.0 as +0.0,
as that all-reduce adds the other ranks' zero rows)."""

from __future__ import annotations

import ctypes
from functools import cache

import torch

from sgl_nvidia_mega.common import jit
from sgl_nvidia_mega.qwen3_5.sm103.draft.top1 import _launch_2d
from sgl_nvidia_mega.qwen3_5.sm103.draft.xchg import SRC, WORLDS, symmetric_words

__all__ = ["EmbedExchange"]

# draft_embed::kSlots / kMaxRows / kThreads * kWords
SLOTS = 3
MAX_ROWS = 128
CHUNK_WORDS = 256 * 4


class DraftEmbedParams(ctypes.Structure):
    """sglang::DraftEmbedParams."""

    _fields_ = [
        (name, ctypes.c_void_p)
        for name in ("tokens", "table", "out", "epoch", "mc", "uc")
    ] + [
        ("tokens_stride", ctypes.c_int64),
        ("m", ctypes.c_int32),
        ("hidden", ctypes.c_int32),
        ("vocab_start", ctypes.c_int32),
        ("vocab_local", ctypes.c_int32),
        ("rank", ctypes.c_int32),
        ("world", ctypes.c_int32),
    ]


@cache
def _kernel(device_index: int) -> jit.Kernel:
    major, minor = torch.cuda.get_device_capability(device_index)
    cubin = jit.compile_source(
        (SRC / "draft_embed.cu").read_text(),
        "q35_draft_embed",
        f"sm_{major}{minor}a",
        ("-std=c++17", "-default-device"),
        headers={"mkd_common.cuh": (SRC / "mkd_common.cuh").read_text()},
    )
    return jit.load(cubin, "draft_embed_xchg", (256, 1, 1), 0)


class EmbedExchange:
    """The exchange buffer and epoch of one rank. Construct on every rank of ``group`` (a
    collective), before any capture."""

    def __init__(
        self,
        *,
        world: int,
        table: torch.Tensor,
        vocab_start: int,
        vocab_local: int,
        group,
        cpu_group,
        device: torch.device,
    ):
        import torch.distributed as dist

        if world not in WORLDS:
            raise ValueError(f"the embedding exchange serves TP {WORLDS}, not {world}")
        if (
            table.dtype != torch.bfloat16
            or table.dim() != 2
            or not table.is_contiguous()
        ):
            raise ValueError(
                f"embedding exchange: table {table.dtype} {tuple(table.shape)}"
            )
        if table.shape[1] % 2 or table.shape[0] < vocab_local:
            raise ValueError(f"embedding exchange: table {tuple(table.shape)}")
        _kernel(device.index)
        self.table, self.hidden = table, table.shape[1]
        self.rank, self.world = dist.get_rank(group), world
        self.vocab_start, self.vocab_local, self.device = (
            vocab_start,
            vocab_local,
            device,
        )
        self._buf, self._handle, self.mc = symmetric_words(
            SLOTS * MAX_ROWS * self.hidden // 2 + world, group, device
        )
        self.uc = self._buf.data_ptr()
        # [epoch from 1, CTA arrivals]
        self.epoch = torch.tensor([1, 0], dtype=torch.int32, device=device)
        torch.cuda.synchronize(device)
        dist.barrier(group=cpu_group)

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        """The embedding rows [m, hidden] bf16 of ``token_ids`` [m] (int64, any stride)."""
        m = int(token_ids.shape[0])
        if (
            not 1 <= m <= MAX_ROWS
            or token_ids.dim() != 1
            or token_ids.dtype != torch.int64
        ):
            raise ValueError(
                f"embedding exchange: tokens {token_ids.dtype} {tuple(token_ids.shape)}"
            )
        out = torch.empty(m, self.hidden, dtype=torch.bfloat16, device=self.device)
        p = DraftEmbedParams(
            tokens=token_ids.data_ptr(),
            table=self.table.data_ptr(),
            out=out.data_ptr(),
            epoch=self.epoch.data_ptr(),
            mc=self.mc,
            uc=self.uc,
            tokens_stride=token_ids.stride(0),
            m=m,
            hidden=self.hidden,
            vocab_start=self.vocab_start,
            vocab_local=self.vocab_local,
            rank=self.rank,
            world=self.world,
        )
        chunks = (self.hidden // 2 + CHUNK_WORDS - 1) // CHUNK_WORDS
        _launch_2d(_kernel(self.device.index), (chunks, m), p, self.device)
        return out
