# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The sharded top-1 of a multi-row MTP draft step (``src/draft_top1.cu``): the argmax of
every row over the TP group's vocabulary shards in one launch, without the logits'
all-gather. It writes the token into a column of the draft tokens, 1.0 into the top-k
probability, and advances the positions (and M-RoPE positions) by one.
"""

from __future__ import annotations

import ctypes
import logging
from functools import cache
from typing import Optional

import torch

from sgl_nvidia_mega.common import jit
from sgl_nvidia_mega.qwen3_5.sm103.draft.xchg import (
    NAN_KEY_SPLIT_ARGMAX,
    NAN_KEY_TORCH_ARGMAX,
    SRC,
    WORLDS,
    symmetric_words,
)

__all__ = ["NAN_KEY_SPLIT_ARGMAX", "NAN_KEY_TORCH_ARGMAX", "Top1Exchange"]

logger = logging.getLogger(__name__)

# draft_top1::kSlots / kMaxRows / kThreads * kVec
SLOTS = 3
MAX_ROWS = 128
THREADS = 256
CHUNK = THREADS * 8

OPTION = "draft_top1_cfg"
DEFAULT_CFG = (256, 8, True, True)
SHIPPED_CFG = (THREADS, 8, False, False)
_SAID: set = set()


def config() -> tuple:
    from sgl_nvidia_mega.qwen3_5 import runtime

    return DEFAULT_CFG if runtime.measured_option(OPTION) else SHIPPED_CFG


def peer_table(ptrs, uc: int, rank: int, world: int) -> tuple:
    ptrs = [int(v) for v in ptrs]
    if len(ptrs) != world:
        return None, f"buffer_ptrs {len(ptrs)} for world {world}"
    off = uc - ptrs[rank]
    if not 0 <= off < (1 << 40):
        return None, f"buffer offset {off}"
    if any(v == 0 for v in ptrs) or len(set(ptrs)) != world:
        return None, "buffer_ptrs has a null or repeated peer address"
    return [v + off for v in ptrs], ""


class DraftTop1Params(ctypes.Structure):
    """sglang::DraftTop1Params."""

    _fields_ = [
        (name, ctypes.c_void_p)
        for name in (
            "logits",
            "keys",
            "epoch",
            "mc",
            "uc",
            "tokens",
            "topk_p",
            "positions",
            "mrope",
        )
    ] + [
        ("tokens_stride", ctypes.c_int64),
        ("mrope_stride", ctypes.c_int64),
        ("row_stride", ctypes.c_int64),
        ("m", ctypes.c_int32),
        ("vocab_local", ctypes.c_int32),
        ("vocab_start", ctypes.c_int32),
        ("vocab_size", ctypes.c_int32),
        ("column", ctypes.c_int32),
        ("rank", ctypes.c_int32),
        ("nan_key", ctypes.c_uint32),
    ]


class DraftTop1P2PParams(ctypes.Structure):
    _fields_ = list(DraftTop1Params._fields_) + [("peers", ctypes.c_void_p * 8)]


@cache
def _kernel(device_index: int, world: int, cfg: tuple = SHIPPED_CFG) -> jit.Kernel:
    major, minor = torch.cuda.get_device_capability(device_index)
    threads, vec, p2p, lean = cfg
    options = ("-std=c++17", "-default-device", f"-DMKD_WORLD={world}")
    name = f"q35_draft_top1_{world}"
    if cfg != SHIPPED_CFG:
        options += (
            f"-DTOP1_THREADS={threads}",
            f"-DTOP1_VEC={vec}",
            f"-DTOP1_P2P={int(p2p)}",
            f"-DTOP1_LEAN={int(lean)}",
        )
        name += f"_{threads}_{vec}_{int(p2p)}_{int(lean)}"
    cubin = jit.compile_source(
        (SRC / "draft_top1.cu").read_text(),
        name,
        f"sm_{major}{minor}a",
        options,
        headers={"mkd_common.cuh": (SRC / "mkd_common.cuh").read_text()},
    )
    return jit.load(cubin, "draft_top1_xchg", (threads, 1, 1), 0)


def build(device_index: int, world: int, cfg: tuple = SHIPPED_CFG) -> None:
    """Compile the kernel (no compile inside a capture)."""
    if world not in WORLDS:
        raise ValueError(f"the sharded top-1 serves TP {WORLDS}, not {world}")
    _kernel(device_index, world, cfg)


class Top1Exchange:
    """The exchange buffer, epoch and per-row fold of one rank. Construct on every rank of
    ``group`` (a collective), before any capture."""

    def __init__(
        self,
        *,
        world: int,
        rank: int,
        vocab_local: int,
        vocab_start: int,
        vocab_size: int,
        group,
        cpu_group,
        device: torch.device,
        cfg: Optional[tuple] = None,
    ):
        import torch.distributed as dist

        cfg = config() if cfg is None else tuple(cfg)
        self.world, self.rank, self.device = world, rank, device
        self.vocab_local, self.vocab_start = vocab_local, vocab_start
        self.vocab_size = vocab_size
        self._buf, self._handle, self.mc = symmetric_words(
            SLOTS * world * MAX_ROWS * 2, group, device, slot="draft_multi"
        )
        self.uc = self._buf.data_ptr()
        self.peers = None
        if cfg[2]:
            why = ""
            try:
                self.peers, why = peer_table(
                    self._handle.buffer_ptrs, self.uc, rank, world
                )
            except Exception as exc:  # no buffer_ptrs on this handle
                why = f"{type(exc).__name__}: {exc}"
            if why:
                logger.warning(
                    "NVIDIA megakernel: %s: the draft top-1 exchange keeps the multicast "
                    "store (%s)",
                    OPTION,
                    why,
                )
                cfg = (cfg[0], cfg[1], False, cfg[3])
        self.cfg = cfg
        build(device.index, world, cfg)
        # [epoch from 1, CTA arrivals]
        self.epoch = torch.tensor([1, 0], dtype=torch.int32, device=device)
        self.keys = torch.zeros(MAX_ROWS, dtype=torch.int64, device=device)
        torch.cuda.synchronize(device)
        dist.barrier(group=cpu_group)

    def __call__(
        self,
        local_logits: torch.Tensor,
        tokens: torch.Tensor,
        column: int,
        *,
        nan_key: int,
        topk_p: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        mrope_positions: Optional[torch.Tensor] = None,
    ) -> None:
        """``tokens[:, column]`` = the global argmax of each row of ``local_logits``
        [m, V_local] (this rank's shard, bf16); ``topk_p`` [m] (any trailing 1) = 1.0;
        ``positions`` [m] and the three rows of ``mrope_positions`` [3, >= m] += 1."""
        m = int(local_logits.shape[0])
        if not 1 <= m <= MAX_ROWS:
            raise ValueError(f"sharded top-1: {m} rows (1-{MAX_ROWS})")
        if (
            local_logits.dtype != torch.bfloat16
            or local_logits.shape[1] != self.vocab_local
            or local_logits.stride(1) != 1
        ):
            raise ValueError(
                f"sharded top-1: logits {local_logits.dtype} "
                f"{tuple(local_logits.shape)} strides {tuple(local_logits.stride())} "
                f"(want bf16 [m, {self.vocab_local}], unit column stride)"
            )
        if tokens.dtype != torch.int64 or tokens.dim() != 2 or tokens.shape[0] < m:
            raise ValueError(
                f"sharded top-1: tokens {tokens.dtype} {tuple(tokens.shape)}"
            )
        for name, t in (("topk_p", topk_p), ("positions", positions)):
            if t is not None and (t.numel() < m or not t.is_contiguous()):
                raise ValueError(f"sharded top-1: {name} {tuple(t.shape)}")
        if positions is not None and positions.dtype != torch.int64:
            raise ValueError(f"sharded top-1: positions {positions.dtype}")
        if mrope_positions is not None and (
            mrope_positions.dtype != torch.int64
            or mrope_positions.shape[0] != 3
            or mrope_positions.stride(1) != 1
        ):
            raise ValueError(f"sharded top-1: M-RoPE {tuple(mrope_positions.shape)}")
        params_t = DraftTop1P2PParams if self.peers is not None else DraftTop1Params
        p = params_t(
            logits=local_logits.data_ptr(),
            keys=self.keys.data_ptr(),
            epoch=self.epoch.data_ptr(),
            mc=self.mc,
            uc=self.uc,
            tokens=tokens.data_ptr(),
            topk_p=0 if topk_p is None else topk_p.data_ptr(),
            positions=0 if positions is None else positions.data_ptr(),
            mrope=0 if mrope_positions is None else mrope_positions.data_ptr(),
            tokens_stride=tokens.stride(0),
            mrope_stride=0 if mrope_positions is None else mrope_positions.stride(0),
            row_stride=local_logits.stride(0),
            m=m,
            vocab_local=self.vocab_local,
            vocab_start=self.vocab_start,
            vocab_size=self.vocab_size,
            column=column,
            rank=self.rank,
            nan_key=nan_key,
        )
        if self.peers is not None:
            p.peers = (ctypes.c_void_p * 8)(*(self.peers + [0] * (8 - self.world)))
        kernel = _kernel(self.device.index, self.world, self.cfg)
        chunk = self.cfg[0] * self.cfg[1]
        chunks = (self.vocab_local + chunk - 1) // chunk
        if (
            self.cfg != SHIPPED_CFG
            and torch.cuda.is_current_stream_capturing()
            and m not in _SAID
        ):
            _SAID.add(m)
            logger.info(
                "NVIDIA megakernel: %s: draft top-1 exchange at %d rows: %d CTAs a row "
                "of %d threads, %s%s",
                OPTION,
                m,
                chunks,
                self.cfg[0],
                "peer stores" if self.peers is not None else "multicast store",
                ", lean last CTA" if self.cfg[3] else "",
            )
        _launch_2d(kernel, (chunks, m), p, self.device)


def _launch_2d(kernel: jit.Kernel, grid: tuple[int, int], params, device) -> None:
    """``jit.launch`` of a (x, y) grid on the current stream, as a programmatic dependent."""
    stream = torch.cuda.current_stream(device).cuda_stream
    jit.launch(kernel, (grid[0], grid[1], 1), [params], stream)
