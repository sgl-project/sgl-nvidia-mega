# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``draft_attn`` (env ``MUSE_VR_DRAFT_ATTN=1``, default off).

The DFlash draft attends, per request, its 8-token block (non-causal) to the paged bf16 draft KV cache within its
sliding window; SGLang's FlashAttention backend runs it as the FA4 ``flash_attn_with_kvcache`` call, which at these
sizes is a split-KV kernel followed by a separate combine kernel. This component replaces that call, when the shape
matches, by one kernel (``csrc/draft_attn.cuh``, ``ops/draft_attn``) that splits the keys across a thread-block cluster
and combines the partials through distributed shared memory. Everything around the call (the KV-cache write, the
backend's metadata and page table, the window) stays SGLang's.

Numerics: same operand precisions as the stock kernel (bf16 Q/K/V, fp32 scores and accumulation, bf16 probabilities
for the second product); the softmax and output reductions run in a different order (tests/test_draft_attn.py: error
against fp64 at most stock's).

``MUSE_VR_DRAFT_ATTN_NS1=1`` (default off; only with this component): calls of the same shape class that this
component leaves to the stock function because they have more requests than ``MUSE_VR_DRAFT_ATTN_MAX_REQS`` run it
with ``num_splits=1`` (one key split, no combine kernel) instead of the automatic split, from
``MUSE_VR_DRAFT_ATTN_NS1_MIN_REQS`` requests (default by KV heads per rank: 32 at 2, 48 at 1; others off), where one
split measured faster. Draft numerics: the key range is summed in one pass instead of per split and
merged, so the result can differ from the automatic split in the summation order (error against fp64 at most stock's,
tests/test_draft_attn.py); it changes no other call.

``MUSE_VR_DRAFT_ATTN_SHADOW=1`` (debug) also runs the stock call for every eager fused call and tallies whether the two
agree within the summation-order tolerance (bed.shadow_result).

Engaged (an AROUND hook on the FlashAttention backend module's ``flash_attn_with_kvcache``) only for: FA4, bf16, head
dim 128, 8 query heads per KV head, 8 query tokens per request (``max_seqlen_q == 8`` and 8 x requests rows), no causal
mask, no soft cap / sinks / score modifiers / descales / rotary / new keys / LSE output, at most
``MUSE_VR_DRAFT_ATTN_MAX_REQS`` requests (default 4; tunable, tests/test_draft_attn.py --bench). Every other call runs
the stock function. The kernel is built and loaded on the first eager call (warm-up before graph capture); a capture
that finds it unloaded keeps the stock path.
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

NAME = "draft_attn"  # env MUSE_VR_DRAFT_ATTN
TARGET = "sglang.srt.layers.attention.flashattention_backend.flash_attn_with_kvcache"
Q_TOKENS = 8

PINS = {
    "sglang/srt/layers/attention/flashattention_backend.py": (
        "53489175aace7b97498efb10ae8b01c489d35e91058842eee02b19aec18f1a3c",
        "a7b8d205742765e12bf0bad48b151007258736e31f6c97a03725cac6ca2c1d17",
    ),
    "sglang/kernels/ops/attention/flash_attention.py": (
        "9f03405f95d100b1989c7041b4dd27298374bb8b8fe152f16949dd0e73ad22bc",
        "294a491df03ca57166044107299c66f0e7c3f4163986725a68709417c31eb981",
    ),
    "sglang/kernels/ops/attention/flash_attention_v4.py": (
        "7518bff1277105d4b8c26d1e22d72c051fd3a6423fd72800ac5fbe08da1bd70b",
        "ee34fddd7e9080268ea1e3ee8f11f32f0674b5d7eda3ff15c6f188de70dc6c01",
    ),
    "sglang/srt/models/dflash.py": (
        "59ae9eab87fd3416828604018c36ad6a57be07c4ff6d17f699bb9c0be8c719ff",
        "b7b541368cbbb072a679d08ba5bf0f952f4a43f02b6c078562b2ea7a9ddfd01b",
    ),
}

_state: dict = {"loaded": set()}

# keyword arguments that must be absent (None / default) for the fused path
_MUST_BE_NONE = (
    "k",
    "v",
    "qv",
    "rotary_cos",
    "rotary_sin",
    "cache_batch_idx",
    "cache_leftpad",
    "rotary_seqlens",
    "q_descale",
    "k_descale",
    "v_descale",
    "attention_chunk",
    "scheduler_metadata",
    "sinks",
    "score_mod",
    "aux_tensors",
    "sfq",
    "sfk",
    "sfv",
    "rel_bias",
    "rel_bias_prep_cache",
    "mask_mod",
)


def enabled() -> bool:
    return bed.enabled(NAME)


def pins_match() -> tuple[bool, str]:
    for rel, want in PINS.items():
        if not bed.pinned(rel, want):
            return False, f"{rel} differs from the pinned revision"
    return True, "ok"


def _eligible(kw: dict) -> str | None:
    if kw.get("ver") != 4:
        return "not FA4"
    for name in _MUST_BE_NONE:
        if kw.get(name) is not None:
            return f"{name} given"
    if (
        kw.get("causal")
        or float(kw.get("softcap") or 0.0) != 0.0
        or kw.get("return_softmax_lse")
    ):
        return "causal/softcap/lse"
    if kw.get("only_qv") or kw.get("pack_gqa") is not None:
        return "qv/pack_gqa"
    q, kc, vc = kw.get("q"), kw.get("k_cache"), kw.get("v_cache")
    pt, seqlens = kw.get("page_table"), kw.get("cache_seqlens")
    if not all(
        isinstance(t, torch.Tensor) and t.is_cuda for t in (q, kc, vc, pt, seqlens)
    ):
        return "inputs"
    if (
        q.dtype != torch.bfloat16
        or kc.dtype != torch.bfloat16
        or vc.dtype != torch.bfloat16
    ):
        return "dtype"
    if (
        q.dim() != 3
        or q.shape[2] != 128
        or kc.dim() != 4
        or vc.dim() != 4
        or kc.shape[3] != 128
    ):
        return "layout"
    if (
        q.stride(2) != 1
        or kc.stride(3) != 1
        or vc.stride(3) != 1
        or kc.shape[:3] != vc.shape[:3]
    ):
        return "layout"
    if (
        pt.dtype != torch.int32
        or pt.dim() != 2
        or pt.stride(1) != 1
        or seqlens.dtype != torch.int32
        or seqlens.dim() != 1
    ):
        return "metadata"
    hkv = kc.shape[2]
    if q.shape[1] != 8 * hkv:
        return "group size"
    bs = seqlens.shape[0]
    if kw.get("max_seqlen_q") != Q_TOKENS or q.shape[0] != Q_TOKENS * bs:
        return "query tokens"
    if bs > bed.env_int("MUSE_VR_DRAFT_ATTN_MAX_REQS", 4):
        return "requests above max"
    out = kw.get("out")
    if out is not None and (
        out.shape != q.shape or out.dtype != q.dtype or out.stride(2) != 1
    ):
        return "out layout"
    win = kw.get("window_size") or (-1, -1)
    if len(win) != 2:
        return "window"
    from .ops import draft_attn as op

    key = (op.use_pdl(), q.device.index)
    if torch.cuda.is_current_stream_capturing():
        if key not in _state["loaded"]:
            return "kernel not loaded before graph capture"
    else:
        op.module(Q_TOKENS, 8, op.SPLITS, op.use_pdl())
        _state["loaded"].add(key)
    return None


_POS = ("q", "k_cache", "v_cache")
# requests from which num_splits=1 measured faster than the automatic split, by KV heads per rank
_NS1_MIN_REQS = {2: 32, 1: 48}


def _ns1_min_reqs(hkv: int) -> int | None:
    v = bed.env_int("MUSE_VR_DRAFT_ATTN_NS1_MIN_REQS", 0)
    return v if v > 0 else _NS1_MIN_REQS.get(hkv)


def attention(original, *args, **kwargs):
    kw = dict(zip(_POS, args))
    kw.update(kwargs)
    if len(args) > len(_POS):
        bed.count(NAME, "stock:positional call")
        return original(*args, **kwargs)
    reason = _eligible(kw)
    if (
        reason == "requests above max"
        and bed.flag(NAME + "_ns1")
        and not kw.get("num_splits")
    ):
        # same shape class, more requests than the fused kernel takes: the stock call with one key split
        bs, hkv = kw["cache_seqlens"].shape[0], kw["k_cache"].shape[2]
        lo = _ns1_min_reqs(hkv)
        if lo is not None and bs >= lo:
            bed.count(NAME, f"stock+ns1:reqs={bs}")
            return original(*args, **dict(kwargs, num_splits=1))
    if reason is not None:
        bed.count(NAME, "stock:" + reason)
        return original(*args, **kwargs)
    from .ops import draft_attn as op

    q = kw["q"]
    scale = kw.get("softmax_scale")
    if scale is None:
        scale = q.shape[-1] ** -0.5
    win = kw.get("window_size") or (-1, -1)
    bed.count(NAME, f"fused:reqs={kw['cache_seqlens'].shape[0]}")
    out = op.attention(
        q,
        kw["k_cache"],
        kw["v_cache"],
        kw["page_table"],
        kw["cache_seqlens"],
        float(scale),
        (int(win[0]), int(win[1])),
        Q_TOKENS,
        out=kw.get("out"),
    )
    if bed.flag(NAME + "_shadow") and not torch.cuda.is_current_stream_capturing():
        # debug: the stock call on a copy of the output, compared within the summation-order tolerance (host sync)
        ref_kw = dict(kw, out=None)
        ref = original(**ref_kw)
        ref = ref[0] if isinstance(ref, tuple) else ref
        tol = 2.0**-6 * float(ref.float().abs().max()) + 1e-6
        bed.shadow_result(NAME, bool((ref.float() - out.float()).abs().max() <= tol))
    return out


def register() -> bool:
    if not enabled():
        return False
    ok, why = pins_match()
    if not ok:
        bed.report(NAME, False, reason=why)
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(TARGET, attention, HookType.AROUND)
    bed.report(NAME, True)
    return True
