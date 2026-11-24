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

Engaged (an AROUND hook on the FlashAttention backend module's ``flash_attn_with_kvcache``) only for: FA4, bf16, head
dim 128, 8 query heads per KV head, 8 query tokens per request (``max_seqlen_q == 8`` and 8 x requests rows), no causal
mask, no soft cap / sinks / score modifiers / descales / rotary / new keys / LSE output, at most
``MUSE_VR_DRAFT_ATTN_MAX_REQS`` requests (default 4; tunable, tests/test_draft_attn.py --bench). Every other call runs
the stock function. Refused (not registered) when the backend module holds no ``flash_attn_with_kvcache`` global: SGLang
after 306ccf6 binds the function per backend instance, out of the hook's reach. The kernel is built and loaded on the first eager call (warm-up before graph capture); a capture
that finds it unloaded keeps the stock path.
"""

from __future__ import annotations

import logging

import torch

from sgl_nvidia_mega.muse_spark.components import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/layers/attention/flashattention_backend.py": ("flash_attn_with_kvcache",),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}
# The FA4 call this component serves for its shape class (``_eligible`` checks its keywords)
SGLANG_MIRRORS = {
    "kernels/ops/attention/flash_attention.py:flash_attn_with_kvcache": "b133c3df871e52d9",
    "kernels/ops/attention/flash_attention_v4.py:flash_attn_with_kvcache": "5acd5d833ff2e3c7",
}


logger = logging.getLogger(__name__)

NAME = "draft_attn"  # env MUSE_VR_DRAFT_ATTN
TARGET = "sglang.srt.layers.attention.flashattention_backend.flash_attn_with_kvcache"
Q_TOKENS = 8

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
    return out


def _target_reachable() -> bool:
    """Whether the FlashAttention backend module binds the FA4 function at module level, the hook's target (SGLang after
    306ccf6 binds it per backend instance, which the hook cannot reach); read from the module's source, which plugin
    registration must not import."""
    import ast
    import importlib.util
    from pathlib import Path

    module, attr = TARGET.rsplit(".", 1)
    try:
        root = Path(
            list(importlib.util.find_spec("sglang").submodule_search_locations)[0]
        )
        tree = ast.parse(
            (root / Path(*module.split(".")[1:])).with_suffix(".py").read_text()
        )
    except Exception:  # noqa: BLE001 - no readable backend: no call to hook
        return False
    todo = list(tree.body)
    while todo:
        node = todo.pop()
        if isinstance(node, (ast.If, ast.Try)):
            todo += node.body + node.orelse + getattr(node, "finalbody", [])
            todo += [s for h in getattr(node, "handlers", []) for s in h.body]
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if any((a.asname or a.name) == attr for a in node.names):
                return True
        elif isinstance(node, (ast.FunctionDef, ast.Assign)):
            names = (
                [node.name]
                if isinstance(node, ast.FunctionDef)
                else [getattr(t, "id", None) for t in node.targets]
            )
            if attr in names:
                return True
    return False


def register() -> bool:
    if not enabled():
        return False
    if not _target_reachable():
        bed.report(NAME, False, reason=f"no module-level hook target {TARGET}")
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(TARGET, attention, HookType.AROUND)
    bed.report(NAME, True)
    return True
