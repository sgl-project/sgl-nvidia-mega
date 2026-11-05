# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``draft_head_lt`` (env ``MUSE_VR_DRAFT_HEAD_LT=1``, default off; needs ``gemm_lt`` engaged).

The DFlash draft projects its hiddens onto the target head's vocabulary shard in ``models/dflash.py``
``_project_candidate_logits``: for a dense head, ``torch.matmul(hidden.to(w.dtype), w[:num_org].T)``. That GEMM does not
go through SGLang's bf16 dispatch, so ``gemm_lt``'s entries never reach it. For each entry of
``MUSE_VR_DRAFT_HEAD_LT_RULES`` (``gemm_lt``'s entry form ``N x K : lo-hi : c0/.../c8``, separated by ``;``; default none,
so nothing is taken without entries) this component runs that call through ``gemm_lt.checked_linear``: the entry's fixed
cuBLASLt configuration behind the same first-use bit-for-bit check against the stock expression (a mismatch keeps that
shape and row count on stock for the process; a row count first met inside a graph capture stays stock). The output has
the stock call's dtype, shape and layout. Every other call runs stock: a quantized head, non-bf16 operands, a layout the
entry form does not cover, SGLang's batch-invariant mode, no entry, or ``gemm_lt`` not engaged.

Taken calls are counted ``lt_draft_head:<n>x<k>:m<m>``, the others ``stock_draft_head:<reason>``.
"""

from __future__ import annotations

import os

import torch

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


NAME = "draft_head_lt"  # env MUSE_VR_DRAFT_HEAD_LT
TARGET = "sglang.srt.models.dflash._project_candidate_logits"
PINS = {
    "sglang/srt/models/dflash.py": (
        "59ae9eab87fd3416828604018c36ad6a57be07c4ff6d17f699bb9c0be8c719ff",
        "b7b541368cbbb072a679d08ba5bf0f952f4a43f02b6c078562b2ea7a9ddfd01b",
    ),
}
_state: dict = {"rules": ()}


def take(hidden, lm_head, num_org: int):
    """The component's output for a dense-head call (None: the stock call runs it)."""
    from . import gemm_lt

    w0 = getattr(lm_head, "weight", None)
    if not gemm_lt._state["engaged"]:
        why = "gemm_lt not engaged"
    elif not (
        isinstance(hidden, torch.Tensor)
        and isinstance(w0, torch.Tensor)
        and hidden.dim() == 2
        and w0.dim() == 2
    ):
        why = "layout"
    else:
        m, (rows, k) = hidden.shape[0], w0.shape
        cfg = (
            gemm_lt.entry(m, num_org, k, _state["rules"])
            if 0 < num_org <= rows
            else None
        )
        if cfg is None:
            why = "no entry"
        elif hidden.dtype != torch.bfloat16 or w0.dtype != torch.bfloat16:
            why = "dtype"
        elif gemm_lt._batch_invariant():
            why = "batch-invariant mode"
        elif not (
            hidden.is_cuda
            and hidden.is_contiguous()
            and w0.is_contiguous()
            and hidden.shape[1] == k
        ):
            why = "layout"
        else:
            w = w0[
                :num_org
            ]  # leading rows of a contiguous row-major weight: contiguous
            assert w.is_contiguous()
            out, why = gemm_lt.checked_linear(
                hidden, w, cfg, lambda: torch.matmul(hidden, w.T)
            )
            if out is not None:
                bed.count(NAME, f"lt_draft_head:{num_org}x{k}:m{m}")
                return out
    bed.count(NAME, f"stock_draft_head:{why}")
    return None


def around_project(original, *args, **kwargs):
    if (
        len(args) == 2
        and set(kwargs) == {"num_org", "use_quant_head"}
        and not kwargs["use_quant_head"]
        and _state["rules"]
    ):
        out = take(args[0], args[1], int(kwargs["num_org"]))
        if out is not None:
            return out
    return original(*args, **kwargs)


def configure(rules=None) -> None:
    """Set the entries (tests, harness); register() reads them from the environment."""
    if rules is not None:
        _state["rules"] = rules


def register() -> bool:
    if not bed.enabled(NAME):
        return False
    for rel, want in PINS.items():
        if not bed.pinned(rel, want):
            bed.report(NAME, False, reason=f"pin mismatch: {rel}")
            return False
    from . import gemm_lt

    try:
        rules = gemm_lt.parse_rules(os.environ.get("MUSE_VR_DRAFT_HEAD_LT_RULES", ""))
    except ValueError as e:
        bed.report(NAME, False, reason=f"bad rules ({e})")
        return False
    if not rules:
        bed.report(NAME, False, reason="no entries")
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(TARGET, around_project, HookType.AROUND)
    _state["rules"] = rules
    fmt = ";".join(
        f"{n}x{k}:{lo}-{hi}:{'/'.join(map(str, c))}" for n, k, lo, hi, c in rules
    )
    bed.report(NAME, True, rules=fmt)
    return True
