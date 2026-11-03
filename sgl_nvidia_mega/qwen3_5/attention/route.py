# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang branch nvidia/qwen35-agentx-gb300:
# python/sglang/srt/layers/attention/graph_variants.py (SplitKVGraphVariants,
# _split_kv_min_length, use_split_kv_attention).
"""Which verify steps take split-KV attention: one request whose KV holds at least
``min_length`` tokens, by a host-side lower bound of its length, so a captured graph is
picked without reading the device.

``VERIFY`` (8192 tokens, as the TP8 serving configuration on SGLang branch
nvidia/qwen35-agentx-gb300 sets ``SGLANG_Q35MF_MKA_SKV2_VERIFY_MIN_LENGTH``) routes the
target verify and ``EXTEND`` (32768, that branch's ``SPLIT_KV_MIN_LENGTH``) the draft
extend. A built megakernel offers its variants per SGLang runner (``graph_variants``:
"target_verify", "draft_loop", "draft_extend"); a runner captures each of
``capture_labels`` in order and replays the one ``select`` picks. A step reads the label
being captured; where SGLang captures none (no runner takes the variants yet), every
covered step takes split-KV attention.
"""

from __future__ import annotations

from typing import ClassVar

import msgspec

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/model_executor/runner_utils/capture_mode.py": (
        "get_capture_attention_variant",
        "get_is_capture_mode",
    ),
    "srt/model_executor/forward_batch_info.py": ("batch_size", "seq_lens_cpu"),
}
# select also reads ForwardBatch.attention_kv_len_lower_bound (the request's committed
# KV length), which SGLang adds with the runners that take these variants

SPLIT_KV_STOCK = "split_kv_stock"
SPLIT_KV_LONG = "split_kv_long"


class SplitKVGraphVariants(msgspec.Struct, frozen=True):
    """SGLang's ``AttentionGraphVariants``: short sequences stay on SGLang's attention
    without a device-to-host copy."""

    min_length: int
    # capture order matters where the variants share a graph memory pool
    capture_labels: ClassVar[tuple[str, ...]] = (SPLIT_KV_STOCK, SPLIT_KV_LONG)

    def select(self, forward_batch) -> str:
        if forward_batch.batch_size != 1:
            return SPLIT_KV_STOCK
        length = forward_batch.attention_kv_len_lower_bound
        if length is None:
            lengths = forward_batch.seq_lens_cpu
            if (
                lengths is not None
                and lengths.device.type == "cpu"
                and lengths.numel() == 1
            ):
                length = int(lengths[0])
        return (
            SPLIT_KV_LONG
            if length is not None and length >= self.min_length
            else SPLIT_KV_STOCK
        )


VERIFY = SplitKVGraphVariants(min_length=8192)
EXTEND = SplitKVGraphVariants(min_length=32768)


def split_kv(variants: SplitKVGraphVariants, forward_batch) -> bool:
    """Whether this step takes split-KV attention: the label being captured, else (an
    eager step) ``variants.select``; every step where SGLang captures no label of
    ours or keeps no host lower bound."""
    from sglang.srt.model_executor.runner_utils.capture_mode import (
        get_capture_attention_variant,
        get_is_capture_mode,
    )

    label = get_capture_attention_variant()
    if label in SplitKVGraphVariants.capture_labels:
        return label == SPLIT_KV_LONG
    if get_is_capture_mode() or not hasattr(
        forward_batch, "attention_kv_len_lower_bound"
    ):
        return True
    return variants.select(forward_batch) == SPLIT_KV_LONG
