# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""FlashMLA's split-KV schedules for a covered step (slot "sched_ahead"): built on
side streams at the step's head and joined before the first attention layer.

SGLang builds each ratio's schedule lazily in the first FlashMLA layer of that ratio,
on the critical path, and skips the build when the step's schedule object already has
one. This fills the same objects with the same kernel and the same inputs, so the
schedule is SGLang's bit for bit; the objects are fresh in every metadata build."""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

import torch

# Keep CPU-only test collection independent of SGLang's CUDA backend imports.
_CUDA_AVAILABLE = torch.cuda.is_available()
if _CUDA_AVAILABLE:
    from sglang.kernels.ops.attention.dsv4.decode_attention_sm100 import (
        can_use_swapab_attention,
    )
    from sglang.kernels.ops.attention.dsv4.flashmla_sched_meta import (
        META_INTS,
        flashmla_sched_meta,
    )
    from sglang.kernels.ops.attention.flash_mla_interface import SPLIT_KV_MIN_TOPK
    from sglang.srt.layers.attention.deepseek_v4_backend import (
        _FLASHMLA_SCHED_BLOCK_SIZE_N,
        _FLASHMLA_SCHED_FIXED_OVERHEAD,
        DSV4AttnMetadata,
        _flashmla_v41_splits_kv,
        _num_sms,
        _pad_tensor_to_size,
    )
    from sglang.srt.runtime_context import get_platform

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/layers/attention/deepseek_v4_backend.py": (
        "DSV4AttnMetadata",
        "_FLASHMLA_SCHED_BLOCK_SIZE_N",
        "_FLASHMLA_SCHED_FIXED_OVERHEAD",
        "_num_sms",
        "_pad_tensor_to_size",
        "core_attn_metadata",
        "get_flashmla_metadata",
        "sparse_page_indices",
        "sparse_topk_lengths",
        "swa_page_indices",
        "swa_topk_lengths",
        "present_ratios",
        "low_ratios",
        '"tile_scheduler_metadata"',
        "num_splits",
        "_flashmla_v41_splits_kv",
    ),
    "kernels/ops/attention/flash_mla_interface.py": ("SPLIT_KV_MIN_TOPK",),
    "kernels/ops/attention/dsv4/kv_layout.py": ("is_interleaved",),
    "kernels/ops/attention/dsv4/flashmla_sched_meta.py": (
        "META_INTS",
        "flashmla_sched_meta",
    ),
    "kernels/ops/attention/dsv4/decode_attention_sm100.py": (
        "can_use_swapab_attention",
        "can_use_swapab_attention(q)",
        "can_use_swapab_attention(kv)",
        "can_use_swapab_attention(extra_kv)",
        "can_use_swapab_attention(num_heads)",
        "can_use_swapab_attention(head_dim_v)",
        "can_use_swapab_attention(softmax_scale)",
    ),
    "srt/models/deepseek_v4.py": (
        "is_dsv41",
        "attn_mqa",
        "head_dim",
        "softmax_scale",
        "compress_ratio",
        "layer_id",
    ),
    "srt/layers/radix_attention.py": ("tp_q_head_num", "v_head_dim"),
    "srt/mem_cache/deepseek_v4_memory_pool.py": (
        "get_swa_key_bytes_per_token",
        "get_extra_key_bytes_per_token",
        "get_swa_key_layout",
    ),
    "srt/runtime_context.py": ("get_platform", '"is_blackwell"', '"is_sm100"'),
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/binding.py": (
        "ForwardView",
        "metadata",
        "kv_pool",
        "trtllm_attn",
        "request_window",
        "pool",
    ),
}

# One stream per job in flight; jobs beyond these share them round-robin
NUM_STREAMS = 4


class _Job(NamedTuple):
    fm: Any  # the step's FlashMLA schedule object of one ratio
    meta: torch.Tensor
    num_splits: torch.Tensor
    event: torch.cuda.Event
    # the side stream reads these: a padded length tensor is allocated on the step's
    # stream, so it must outlive the join, not start()
    inputs: tuple


class _Streams:
    streams: Optional[list] = None


def sglang_takes_swapab(attn, view, rows: int) -> bool:
    """Whether SGLang's backend runs swap-AB instead of FlashMLA for ``attn``'s layer
    at ``rows`` rows: its own gate on shape-only stand-ins for the query and the
    caches (start() checks the forward mode)."""
    if not _CUDA_AVAILABLE or not (attn.is_dsv41 and get_platform().is_sm100):
        return False
    pool, layer = view.kv_pool, attn.attn_mqa

    def stand_in(*shape, dtype=torch.uint8):
        return torch.empty(shape, dtype=dtype, device="meta")

    extra = None
    if attn.compress_ratio != 0:
        extra = stand_in(0, pool.get_extra_key_bytes_per_token(attn.layer_id))
    return can_use_swapab_attention(
        q=stand_in(rows, 1, layer.tp_q_head_num, attn.head_dim, dtype=torch.bfloat16),
        kv=stand_in(0, pool.get_swa_key_bytes_per_token()),
        extra_kv=extra,
        num_heads=layer.tp_q_head_num,
        head_dim_v=layer.v_head_dim,
        softmax_scale=attn.softmax_scale,
    )


def _inputs(core, rows: int, runs_flashmla, interleaved: bool) -> list:
    """(schedule object, SWA lengths, SWA width, extra lengths, extra width) per
    ratio whose layers run FlashMLA with a split-KV schedule, as SGLang's FlashMLA
    layer passes them (lengths padded to the rows). On the interleaved KV layout
    SGLang builds no schedule below FlashMLA's split-KV top-k."""

    def match(x):
        if x is None or x.shape[0] == rows:
            return x
        return x[:rows] if x.shape[0] > rows else _pad_tensor_to_size(x, rows, value=1)

    out = []
    for ratio in (0, *core.low_ratios, 4, 128):
        if ratio != 0 and ratio not in core.present_ratios:
            continue
        if not runs_flashmla(ratio):
            continue
        fm = core.get_flashmla_metadata(ratio)
        if fm is None or getattr(fm, "tile_scheduler_metadata", None) is not None:
            continue
        extra = None if ratio == 0 else core.sparse_page_indices(ratio)
        if interleaved and not _flashmla_v41_splits_kv(
            core.swa_page_indices, extra, SPLIT_KV_MIN_TOPK
        ):
            continue
        out.append(
            (
                fm,
                match(core.swa_topk_lengths),
                core.swa_page_indices.shape[-1],
                None if ratio == 0 else match(core.sparse_topk_lengths(ratio)),
                0 if extra is None else extra.shape[-1],
            )
        )
    return out


def start(view, fb, rows: int, runs_flashmla) -> list[_Job]:
    """Fork this step's schedules onto the side streams, for each compress ratio
    whose layers run FlashMLA's decode kernel (``runs_flashmla(ratio)``)."""
    if not _CUDA_AVAILABLE:
        return []
    core = getattr(view.metadata, "core_attn_metadata", None)
    if (
        not get_platform().is_blackwell
        or view.trtllm_attn
        or not fb.forward_mode.is_target_verify()
        or not isinstance(core, DSV4AttnMetadata)
        or view.kv_pool.request_window is not None
    ):
        return []
    # KvPoolView has no layout getter, and V41 / V41_NEW share their bytes per token
    interleaved = view.kv_pool.pool.get_swa_key_layout().is_interleaved
    device = fb.positions.device
    # one query per row (s_q 1), as SGLang's lazy build sizes it
    num_sm_parts = max(_num_sms(device.index), 1)
    if 4 * (5 * rows + 1 + num_sm_parts * META_INTS) > 48 * 1024:
        return []
    if _Streams.streams is None:
        _Streams.streams = [torch.cuda.Stream() for _ in range(NUM_STREAMS)]
    current = torch.cuda.current_stream()
    jobs = []
    for i, (fm, lengths, width, extra_lengths, extra_width) in enumerate(
        _inputs(core, rows, runs_flashmla, interleaved)
    ):
        meta = torch.empty((num_sm_parts, META_INTS), dtype=torch.int32, device=device)
        num_splits = torch.empty((rows + 1,), dtype=torch.int32, device=device)
        stream = _Streams.streams[i % NUM_STREAMS]
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            flashmla_sched_meta(
                meta,
                num_splits,
                topk_length=lengths,
                extra_topk_length=extra_lengths,
                block_size_n=_FLASHMLA_SCHED_BLOCK_SIZE_N,
                fixed_overhead_num_blocks=_FLASHMLA_SCHED_FIXED_OVERHEAD,
                topk=width,
                extra_topk=extra_width,
            )
            event = stream.record_event()
        jobs.append(_Job(fm, meta, num_splits, event, (lengths, extra_lengths)))
    return jobs


def join(jobs: list[_Job]) -> None:
    """Make the step's stream wait for every schedule (none may stay unjoined in a
    capture) and hand them to their schedule objects."""
    current = torch.cuda.current_stream()
    for job in jobs:
        current.wait_event(job.event)
        job.fm.tile_scheduler_metadata = job.meta
        job.fm.num_splits = job.num_splits
