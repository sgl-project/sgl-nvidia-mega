# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import logging
from typing import Optional

import torch

from sgl_nvidia_mega.qwen3_5.sm103.head.xchg import HeadTop1Exchange

logger = logging.getLogger(__name__)

SGLANG_READS = {
    "srt/models/qwen3_5_mtp.py": ("lm_head", "logits_processor"),
    "srt/layers/logits_processor.py": (
        "logit_scale",
        "final_logit_softcapping",
        "vocab_size",
    ),
    "srt/layers/vocab_parallel_embedding.py": (
        "shard_indices",
        "org_vocab_start_index",
    ),
    "srt/kernel_plugins/nvidia/eagle_binding.py|srt/megakernels/eagle_binding.py": (
        "tp_rank",
        "tp_size",
        "tp_cpu_group",
        "tp_device_group",
    ),
}

OPTION = "draft_head_keys"
SLOT = "draft_head_lowm"
MIN_ROWS = 9
MAX_ROWS = 16  # the multi-request draft steps the decode graphs serve on TEP8
STAGES = 5
HIDDEN = 4096
# the widths whose capture logged the engagement line
_ENGAGED: set = set()


def note(rows: int) -> None:
    if torch.cuda.is_current_stream_capturing() and rows not in _ENGAGED:
        _ENGAGED.add(rows)
        logger.info(
            "NVIDIA megakernel: option %s runs slot %s's head at %d rows (argmax keys)",
            OPTION,
            SLOT,
            rows,
        )


def nan_hi(nan_key: int) -> int:
    v = (int(nan_key) & 0xFFFFFFFF) ^ 0x80000000
    return v - (1 << 32) if v >= (1 << 31) else v


def head_rows(widths) -> tuple:
    """The draft widths this head takes."""
    return tuple(sorted({int(r) for r in widths if MIN_ROWS <= int(r) <= MAX_ROWS}))


class DraftHeadKeys(HeadTop1Exchange):
    def __init__(
        self,
        weight: torch.Tensor,
        *,
        rows: tuple,
        world: int,
        rank: int,
        vocab_start: int,
        group,
        cpu_group,
        device: torch.device,
    ):
        import torch.distributed as dist

        from sgl_nvidia_mega.qwen3_5.runtime import SlotUnavailable
        from sgl_nvidia_mega.qwen3_5.sm103.draft.top1 import peer_table
        from sgl_nvidia_mega.qwen3_5.sm103.draft.xchg import symmetric_words
        from sgl_nvidia_mega.qwen3_5.sm103.head import mma, xchg
        from sgl_nvidia_mega.qwen3_5.sm103.head.keys import KEY_FLOOR

        xchg._check_world(world)
        self.weight = weight
        self.rows = frozenset(int(r) for r in rows)
        self.vocab_start = int(vocab_start)
        self.valid = int(weight.shape[0])
        for m in sorted(self.rows):
            mma._kernel(device.index, *mma.variant(m, stages=STAGES), True, False, True)
        xchg.build(device.index)
        self.world, self.rank, self.device = world, rank, device
        self._buf, self._handle, _ = symmetric_words(
            xchg.buffer_words(world), group, device, slot=SLOT
        )
        why = ""
        try:
            peers, why = peer_table(
                self._handle.buffer_ptrs, self._buf.data_ptr(), rank, world
            )
        except Exception as exc:  # a handle without buffer_ptrs
            peers, why = None, f"{type(exc).__name__}: {exc}"
        ok = torch.tensor([0 if why else 1], dtype=torch.int32, device=device)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
        if not int(ok.item()):
            raise SlotUnavailable(
                SLOT,
                f"the key exchange needs every rank's buffer address ({why or 'a peer'})",
            )
        self.peers = peers
        self._keys = torch.full(
            (xchg.MAX_ROWS, xchg.KEY_LINE), KEY_FLOOR, dtype=torch.int64, device=device
        )
        self.keys = self._keys[:, 0]
        self.epoch = xchg._epoch(1, device)
        torch.cuda.synchronize(device)
        dist.barrier(group=cpu_group)

    def takes(self, rows: int) -> bool:
        return rows in self.rows

    def serves(self, hidden: torch.Tensor, tokens: torch.Tensor, outputs: dict) -> bool:
        m = int(hidden.shape[0])
        topk_p, pos, mrope = (
            outputs.get(k) for k in ("topk_p", "positions", "mrope_positions")
        )
        return (
            self.takes(m)
            and set(outputs) <= {"topk_p", "positions", "mrope_positions"}
            and hidden.dtype == torch.bfloat16
            and hidden.dim() == 2
            and hidden.shape[1] == HIDDEN
            and tokens.dtype == torch.int64
            and tokens.dim() == 2
            and tokens.shape[0] >= m
            and (
                topk_p is None
                or (
                    topk_p.dtype == torch.float32
                    and topk_p.is_contiguous()
                    and topk_p.numel() >= m
                )
            )
            and (
                pos is None
                or (
                    pos.dtype == torch.int64
                    and pos.is_contiguous()
                    and pos.numel() >= m
                )
            )
            and (
                mrope is None
                or (
                    mrope.dtype == torch.int64
                    and mrope.dim() == 2
                    and mrope.shape[0] == 3
                    and mrope.shape[1] >= m
                    and mrope.stride(1) == 1
                )
            )
        )

    def top1_into(
        self,
        hidden: torch.Tensor,
        tokens: torch.Tensor,
        column: int,
        *,
        nan_key: int,
        topk_p: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        mrope_positions: Optional[torch.Tensor] = None,
    ) -> None:
        from sgl_nvidia_mega.qwen3_5.sm103.head import mma

        m = int(hidden.shape[0])
        x = hidden
        if x.stride(1) != 1 or x.data_ptr() % 16 or (x.stride(0) * 2) % 16:
            x = x.contiguous()
        # ``serves`` and the build checked the inputs
        mma.head_mma_keys(
            x,
            self.weight,
            self.keys,
            vocab_start=self.vocab_start,
            valid=self.valid,
            nan_hi=nan_hi(nan_key),
            stages=STAGES,
        )
        self(
            m,
            tokens,
            column=column,
            topk_p=topk_p,
            positions=positions,
            mrope_positions=mrope_positions,
        )


def unsupported_reason(
    *,
    weight: torch.Tensor,
    vocab_start: int,
    vocab_size: int,
    rank: int,
    world: int,
    processor,
) -> str:
    if (
        weight.dtype != torch.bfloat16
        or weight.dim() != 2
        or not weight.is_contiguous()
    ):
        return f"LM head {weight.dtype} {tuple(weight.shape)}; needs a contiguous BF16 shard"
    if weight.shape[1] != HIDDEN:
        return f"hidden size {weight.shape[1]}"
    if processor.logit_scale is not None or getattr(
        processor, "final_logit_softcapping", None
    ):
        return "logit scaling or softcapping"
    rows = int(weight.shape[0])
    if vocab_start != rank * rows or rows * world != vocab_size:
        return (
            f"the {rows}-row shard at {vocab_start} is not rank {rank}'s slice of "
            f"{vocab_size} ids"
        )
    if weight.data_ptr() % 16:
        return "the LM head shard is not 16-byte aligned (TMA)"
    major, _ = torch.cuda.get_device_capability(weight.device)
    if major != 10:
        return f"the tensor-core head needs SM10x, not SM{major}x"
    return ""


def maybe_build(parts) -> Optional[DraftHeadKeys]:
    from sgl_nvidia_mega.qwen3_5.draft import MULTI_LOOP_BS
    from sgl_nvidia_mega.qwen3_5.runtime import (
        SlotUnavailable,
        measured_option,
        slot_supported,
        unavailable_slots,
    )

    if not measured_option(OPTION):
        return None
    rows = head_rows(MULTI_LOOP_BS)
    if not rows:
        return None
    lm, b = parts.lm, parts.binding
    w = lm.lm_head.weight
    reason = unsupported_reason(
        weight=w,
        vocab_start=int(lm.lm_head.shard_indices.org_vocab_start_index),
        vocab_size=int(lm.logits_processor.vocab_size),
        rank=b.tp_rank,
        world=b.tp_size,
        processor=lm.logits_processor,
    )
    if not slot_supported(
        SLOT,
        reason and f"argmax-key head at {MIN_ROWS}-{MAX_ROWS} rows: {reason}",
        cpu_group=getattr(b, "tp_cpu_group", None),
    ):
        return None
    try:
        head = DraftHeadKeys(
            w.detach(),
            rows=rows,
            world=b.tp_size,
            rank=b.tp_rank,
            vocab_start=int(lm.lm_head.shard_indices.org_vocab_start_index),
            group=b.tp_device_group,
            cpu_group=b.tp_cpu_group,
            device=w.device,
        )
    except SlotUnavailable as exc:
        unavailable_slots(SLOT, exc.reason)
        return None
    if b.tp_rank == 0:
        logger.info(
            "NVIDIA megakernel: option %s runs slot %s as the argmax-key head + key "
            "exchange at %s rows",
            OPTION,
            SLOT,
            list(rows),
        )
    return head
