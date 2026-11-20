# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang branch nvidia/qwen35-agentx-gb300:
# python/sglang/srt/layers/attention/graph_variants.py (SplitKVGraphVariants,
# _split_kv_min_length, use_split_kv_attention).
"""Which verify steps take split-KV attention: one request whose KV holds at least
``min_length`` tokens, by a host-side lower bound of its length, so a captured graph is
picked without reading the device.

``VERIFY`` (8192 tokens, as the TP8 serving configuration on SGLang branch
nvidia/qwen35-agentx-gb300) routes the
target verify and ``EXTEND`` (32768, that branch's ``SPLIT_KV_MIN_LENGTH``) the draft
extend. A built megakernel offers its variants per SGLang runner (``graph_variants``:
"target_verify", "draft_loop", "draft_extend"); a runner captures each of
``capture_labels`` in order and replays the one ``select`` picks. A step reads the label
being captured; where SGLang captures none (no runner takes the variants yet), every
covered step takes split-KV attention.

Slots ``skv2_cp`` / ``draft_skv2_cp`` (and ``draft_q1_cp`` for the draft loop) add a
third label, ``SPLIT_KV_CP``: one request past the measured per-site crossover takes
CP-lite (SGLang's read-split attention over ranks that hold the same KV head) in place
of SKV2, in a graph of its own, so the shorter steps' graphs stay as they are.
"""

from __future__ import annotations

import functools
from typing import ClassVar, Optional

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
SPLIT_KV_CP = "split_kv_cp"
CP_SITES = {"verify": "skv2_cp", "draft": "draft_q1_cp", "extend": "draft_skv2_cp"}
# TP8 static A/B crossovers against SKV2 / SKV2-Q1: verify 136 Ki, draft step
# 180 Ki, draft extend 240 Ki committed KV tokens.
CP_MIN_KV = {"verify": 139264, "draft": 184320, "extend": 245760}


class SplitKVGraphVariants(msgspec.Struct, frozen=True):
    """SGLang's ``AttentionGraphVariants``: short sequences stay on SGLang's attention
    without a device-to-host copy."""

    min_length: int
    # capture order matters where the variants share a graph memory pool
    capture_labels: ClassVar[tuple[str, ...]] = (SPLIT_KV_STOCK, SPLIT_KV_LONG)
    # select picks the stock label past one request (and the steps read the label only
    # for one), so SGLang captures one graph at every larger batch size
    max_batch_size: ClassVar[int] = 1

    def select(self, forward_batch) -> str:
        length = one_request_length(forward_batch)
        return (
            SPLIT_KV_LONG
            if length is not None and length >= self.min_length
            else SPLIT_KV_STOCK
        )


class SplitKVCPGraphVariants(SplitKVGraphVariants, frozen=True):
    """``SplitKVGraphVariants`` with ``SPLIT_KV_CP`` for one request from
    ``cp_min_length`` tokens of KV."""

    cp_min_length: int = CP_MIN_KV["verify"]
    capture_labels: ClassVar[tuple[str, ...]] = (
        SPLIT_KV_STOCK,
        SPLIT_KV_LONG,
        SPLIT_KV_CP,
    )

    def select(self, forward_batch) -> str:
        length = one_request_length(forward_batch)
        if length is not None and length >= self.cp_min_length:
            return SPLIT_KV_CP
        return (
            SPLIT_KV_LONG
            if length is not None and length >= self.min_length
            else SPLIT_KV_STOCK
        )


def one_request_length(forward_batch) -> Optional[int]:
    """A host-side lower bound of the KV length of a one-request step, else None."""
    if forward_batch.batch_size != 1:
        return None
    length = forward_batch.attention_kv_len_lower_bound
    if length is None:
        lengths = forward_batch.seq_lens_cpu
        if (
            lengths is not None
            and lengths.device.type == "cpu"
            and lengths.numel() == 1
        ):
            length = int(lengths[0])
    return length


VERIFY = SplitKVGraphVariants(min_length=8192)
EXTEND = SplitKVGraphVariants(min_length=32768)


def cp_min_length(site: str) -> Optional[int]:
    """The KV length from which one request's ``site`` ("verify" / "draft" / "extend")
    takes CP-lite, or None when its slot is not requested."""
    from sgl_nvidia_mega.qwen3_5.runtime import SLOTS

    if CP_SITES[site] not in SLOTS:
        return None
    return CP_MIN_KV[site]


def verify_variants() -> SplitKVGraphVariants:
    """The target verify's variants: ``VERIFY``, with ``SPLIT_KV_CP`` under slot skv2_cp."""
    n = cp_min_length("verify")
    if n is None:
        return VERIFY
    return _with_cp(VERIFY.min_length, n)


def extend_variants() -> SplitKVGraphVariants:
    """The draft extend's variants: ``EXTEND``, with ``SPLIT_KV_CP`` under slot draft_skv2_cp."""
    n = cp_min_length("extend")
    if n is None:
        return EXTEND
    return _with_cp(EXTEND.min_length, n)


@functools.cache
def _with_cp(min_length: int, cp_length: int) -> SplitKVCPGraphVariants:
    return SplitKVCPGraphVariants(min_length=min_length, cp_min_length=cp_length)


def label(variants: SplitKVGraphVariants, forward_batch) -> str:
    """This step's label of ``variants``: the one being captured, else (an eager step)
    ``variants.select``; ``SPLIT_KV_LONG`` where SGLang captures no label of ours or
    keeps no host lower bound."""
    from sglang.srt.model_executor.runner_utils.capture_mode import (
        get_capture_attention_variant,
        get_is_capture_mode,
    )

    captured = get_capture_attention_variant()
    if captured in variants.capture_labels:
        return captured
    if get_is_capture_mode() or not hasattr(
        forward_batch, "attention_kv_len_lower_bound"
    ):
        return SPLIT_KV_LONG
    return variants.select(forward_batch)


def split_kv(variants: SplitKVGraphVariants, forward_batch) -> bool:
    """Whether this step takes SKV2: ``label`` is ``SPLIT_KV_LONG``."""
    return label(variants, forward_batch) == SPLIT_KV_LONG
