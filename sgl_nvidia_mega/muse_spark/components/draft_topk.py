# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``draft_topk`` (env ``MUSE_VR_DRAFT_TOPK=1``, default off).

The DFLASH candidate selector takes the top-k of the draft head's logits per draft position (``models/dflash.py``
``_radix_topk``: ``flashinfer.top_k(..., sorted=True, deterministic=True)``, a radix select plus a stable sort). This
component serves that call with one multi-CTA exact top-k kernel (``ops/draft_topk``, ``csrc/draft_topk.cuh``) for 2-D
contiguous CUDA bf16/fp16/fp32 inputs with k <= 32 and at most 65536 columns; other calls run stock.

Numerics: the selected set and order are those of the composite key (value descending, then ascending column). They
equal flashinfer's wherever the k-th and (k+1)-th values differ; where they tie, flashinfer's deterministic choice
follows its own work split, so the draft's candidates can differ there (draft proposals only; verification and the
committed tokens follow the target as before). tests/test_draft_topk.py checks the kernel against a composite-key
reference and against torch.topk on tie-free rows.

``MUSE_VR_DRAFT_TOPK_MIN_ROWS`` (default 0): calls with fewer rows than this run stock (counted
``stock:rows below min``), so the kernel can be limited to the row counts where it is measured faster.

``MUSE_VR_DRAFT_TOPK_BW=1`` (default off; only with this component): calls that ``ops/draft_topk_bw`` supports (bf16, k = 16,
contiguous, columns a multiple of 8 up to 65528, 16-byte aligned) run that kernel instead: one CTA per row, the row read from
memory once and staged in shared memory, a two-byte radix select, the same composite-key order and therefore the same output as
the multi-CTA kernel (tests/test_draft_topk_bw.py). Other calls take the multi-CTA kernel as above.

``MUSE_VR_DRAFT_TOPK_SB=1`` (default off; only with this component; takes precedence over ``_BW``): calls that ``ops/draft_topk_sb``
supports (the same conditions as ``_BW``) with at least ``MUSE_VR_DRAFT_TOPK_SB_MIN_ROWS`` rows (default 0) run that kernel: one CTA
per row, a lower bound on the 16th key from a sample of the row, one streaming filter pass, an exact select among the values that
pass, and a full-row select when too many pass. Same composite-key order and output as the multi-CTA kernel
(tests/test_draft_topk_sb.py). Smaller calls (at or above ``_MIN_ROWS``) take the multi-CTA kernel. Captured calls are tallied
``capture:sb:rows<N>``.

``MUSE_VR_DRAFT_TOPK_MAX_ROWS`` (default 0 = no limit): calls that would take the multi-CTA (or ``_BW``) kernel with more rows than
this run stock (counted ``stock:rows above max``), so the multi-CTA kernel can serve a band of row counts below ``_SB_MIN_ROWS``.
"""

from __future__ import annotations

import logging

import torch

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


logger = logging.getLogger(__name__)

NAME = "draft_topk"  # env MUSE_VR_DRAFT_TOPK
TARGET = "sglang.srt.models.dflash._radix_topk"

PINS = {
    "sglang/srt/models/dflash.py": (
        "59ae9eab87fd3416828604018c36ad6a57be07c4ff6d17f699bb9c0be8c719ff",
        "b7b541368cbbb072a679d08ba5bf0f952f4a43f02b6c078562b2ea7a9ddfd01b",
    ),
}


def enabled() -> bool:
    return bed.enabled(NAME)


def pins_match() -> tuple[bool, str]:
    for rel, want in PINS.items():
        if not bed.pinned(rel, want):
            return False, f"{rel} differs from the pinned revision"
    return True, "ok"


def around_radix_topk(original, scores, k, *args, **kwargs):
    from .ops import draft_topk as op

    if (
        args
        or kwargs
        or not isinstance(scores, torch.Tensor)
        or not op.supported(scores, int(k))
    ):
        return original(scores, k, *args, **kwargs)
    capturing = torch.cuda.is_current_stream_capturing()
    sb = None
    if bed.flag(NAME + "_sb"):
        from .ops import draft_topk_sb as sb

        if not capturing and not sb.loaded():
            sb.module()  # load at the first eager call of any row count, so every later graph capture finds it loaded
    if scores.shape[0] < bed.env_int("MUSE_VR_DRAFT_TOPK_MIN_ROWS", 0):
        if not capturing:
            bed.count(NAME, "stock:rows below min")
        return original(scores, k, *args, **kwargs)
    if (
        sb is not None
        and scores.shape[0] >= bed.env_int("MUSE_VR_DRAFT_TOPK_SB_MIN_ROWS", 0)
        and sb.supported(scores, int(k))
    ):
        if capturing and not sb.loaded():
            bed.count(NAME, "stock:sb kernel not loaded before capture")
            return original(scores, k, *args, **kwargs)
        if not capturing:
            bed.count(NAME, "sb")
        else:
            bed.count(NAME, f"capture:sb:rows{scores.shape[0]}")
        return sb.topk(scores, int(k))
    max_rows = bed.env_int("MUSE_VR_DRAFT_TOPK_MAX_ROWS", 0)
    if max_rows > 0 and scores.shape[0] > max_rows:
        if not capturing:
            bed.count(NAME, "stock:rows above max")
        return original(scores, k, *args, **kwargs)
    if op.counters(scores.device, create=not capturing) is None:
        bed.count(NAME, "stock:no counters before capture")
        return original(scores, k, *args, **kwargs)
    if bed.flag(NAME + "_bw"):
        from .ops import draft_topk_bw as bw

        if bw.supported(scores, int(k)):
            if capturing and not bw.loaded():
                bed.count(NAME, "stock:bw kernel not loaded before capture")
                return original(scores, k, *args, **kwargs)
            if not capturing:
                bed.count(NAME, "bw")
            else:
                bed.count(NAME, f"capture:bw:rows{scores.shape[0]}")
            return bw.topk(scores, int(k))
    if not capturing:
        bed.count(NAME, "topk")
    else:
        bed.count(
            NAME, f"capture:rows{scores.shape[0]}"
        )  # host-side tally per captured graph (engagement evidence)
    return op.topk(scores, int(k))


def register() -> bool:
    if not enabled():
        return False
    ok, why = pins_match()
    if not ok:
        bed.report(NAME, False, reason=why)
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(TARGET, around_radix_topk, HookType.AROUND)
    bed.report(NAME, True)
    return True
