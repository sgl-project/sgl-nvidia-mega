# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Which kernel unit serves each step of the TP4 profiles: the preference table, and the
units' switches.

``PREFERENCE`` has one ordered list per (group, step); the registry takes, per width,
the first unit in it that is present, switched on, preferred here (``Pref.when``: a
TP / row / mode predicate) and covers the width (``registry`` module docstring). A
name whose module is absent is skipped. Each entry names the measurement behind it.
The TP8 envelope does not come here: ``plan.build_providers`` builds its families.

Switches: every unit is a slot (``runtime.LL_SLOTS``, shipped; ``runtime.OPT_IN_SLOTS``
only when listed), so ``SGLANG_NVIDIA_MEGAKERNEL_SLOTS`` selects the units as it
selects the TP8 families: unset runs every shipped slot, "a,b" exactly those, "" none
(parity: every width on the stock bodies). The ``draft`` slot carries the draft's fused
front (``draft.front``) and gates every unit of the draft's groups, as it gates the
TP8 draft step. A GEMM role whose measured plan entry is deferred runs only when its
unit is listed explicitly.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

from sgl_nvidia_mega.glm5_3.registry import (
    DRAFT_GROUPS,
    GROUPS,
    Candidate,
    Pref,
)


def tp(n: int):
    """``Pref.when``: at attention TP ``n``."""
    return lambda p, rows, mode: p.tp == n


def in_mode(*modes: str):
    """``Pref.when``: in these step modes."""
    return lambda p, rows, mode: mode in modes


def _when_any(*preds):
    return lambda p, rows, mode: any(f(p, rows, mode) for f in preds)


# The attention front's steps (ll.attention overrides qkv_a / prologue / index_k /
# rope_cache; the indexer query and the indexer stay stock) and the core's
FRONT_STEPS = (
    "qkv_a",
    "prologue",
    "index_k",
    "index_q",
    "indexer",
    "q_absorb",
    "rope_cache",
)
CORE_STEPS = ("sparse_mla", "uv_proj")
HOP_STEPS = GROUPS["hops"]

# (group, step) -> preference order
PREFERENCE: Dict[Tuple[str, str], Tuple[Pref, ...]] = {}


def _put(group: str, steps: Sequence[str], *prefs: Pref) -> None:
    for s in steps:
        if s not in GROUPS[group]:
            raise ValueError(f"{group} has no step {s}")
        PREFERENCE[(group, s)] = tuple(prefs)


# Target verify, TP4 (plan section A.2)
_put(
    "attention",
    FRONT_STEPS,
    Pref("ll.attention", note="TP4: P5 LL fused-A + P2 KV-half (WP2)"),
)
_put(
    "attention",
    CORE_STEPS,
    Pref("ll.sparse_mla_fd", note="TP4: P1 one-tile, 16 heads, <= 7 rows (WP1)"),
)
_put(
    "hops",
    HOP_STEPS,
    Pref("ll.hops", note="TP4: P4 one-shot / two-shot + P9 finalize (WP4)"),
)
_put(
    "mlp",
    ("forward",),
    # requires hops:finalize and mlp_input:nvfp4 (ll.hops writes NVFP4 only for it)
    Pref("ll.moe.nvfp4", note="P10: NVFP4 MoE input from the hop (D6: LOO decides)"),
    Pref("ll.moe", note="TP4: FlashInfer trtllm-gen FP4 MoE, deferred finalize (WP4)"),
)

# Draft, TP4 (plan section A.3)
_put(
    "draft_front",
    ("front",),
    Pref("draft.front", note="D1a: the draft step's fused front, TP-agnostic"),
)
_put(
    "draft_eh",
    ("eh_proj",),
    # E-D1 (WP5): at TP4 P6a 21.5-25.7 us cold vs stock ~40 vs the TP8 column shard +
    # NCCL gather 34-67
    Pref("ll.draft_eh", note="D1b TP4: P6a row-parallel + P4 hop sum (E-D1)"),
)
_put(
    "draft_attention",
    FRONT_STEPS,
    Pref("ll.attention", note="D2 TP4: the NextN layer on P5 / P2 (WP2)"),
)
_put(
    "draft_attention",
    CORE_STEPS,
    Pref("ll.sparse_mla_fd", note="D2 TP4: P1 one-tile (16 heads, <= 7 rows)"),
)
_put(
    "draft_mlp",
    ("forward",),
    Pref(
        "draft_rowselect",
        when=in_mode("draft_extend"),
        note="D3: the draft-extend MoE on 1 row per request; -26 / -74 us/iter at TP8",
    ),
)
# draft_top1: none at TP4. E-D2 (WP5): SGLang's one-tray multimem gather + top-1 is
# 2-3 us/step faster than the distributed argmax at 1-4 rows.

# Options: units that are not candidates of a step group but switch like one (shipped
# slots). The GEMM roles of ll.gemm (ll.linear):
GEMM_UNITS = tuple(
    f"ll.gemm.{role}"
    for role in (
        "wk_weights_proj",
        "o_proj",
        "router",
        "dense_down",
        "eh_proj",
        "lm_head",
        "q_b",
    )
)
OPTIONS: Tuple[str, ...] = (
    *GEMM_UNITS,
    # ll.attention: the kv_a RMSNorm folded into P2's KV-half kernel (WP2; LOO "fold off")
    "ll.attention.kv_norm_fold",
    # ll.moe: the shared expert on a stream of the greatest priority at <= 6 rows (faster at 5-6 rows, slower at 12-24 rows)
    "ll.moe.alt_stream_priority",
)
# every unit name (each a slot: runtime.LL_SLOTS, or draft / draft_rowselect)
KNOWN: Tuple[str, ...] = tuple(
    dict.fromkeys(
        [p.name for prefs in PREFERENCE.values() for p in prefs] + list(OPTIONS)
    )
)
# unit -> the slot that switches it, where the two names differ
SLOT_OF = {"draft.front": "draft"}

_DRAFT_NAMES = frozenset(
    p.name for (g, _), prefs in PREFERENCE.items() if g in DRAFT_GROUPS for p in prefs
)
_TARGET_NAMES = frozenset(
    p.name
    for (g, _), prefs in PREFERENCE.items()
    if g not in DRAFT_GROUPS
    for p in prefs
)


def slot(name: str) -> str:
    return SLOT_OF.get(name, name)


def enabled(name: str, group: Optional[str] = None) -> bool:
    """Whether unit ``name`` is switched on (in ``group``, when it serves both the
    target and the draft: e.g. ll.sparse_mla_fd)."""
    from sgl_nvidia_mega.glm5_3.runtime import SLOTS

    if slot(name) not in SLOTS:
        return False
    if group is None:
        draft = name in _DRAFT_NAMES and name not in _TARGET_NAMES
    else:
        draft = group in DRAFT_GROUPS
    # the draft's units run only with slot draft (the draft step), as the TP8 draft
    return not (draft and "draft" not in SLOTS)


def listed(name: str) -> bool:
    """Whether ``SGLANG_NVIDIA_MEGAKERNEL_SLOTS`` names unit ``name`` explicitly (the
    deferred GEMM plan entries run only then)."""
    import os

    from sgl_nvidia_mega.glm5_3.runtime import SLOTS, SLOTS_ENV

    return os.environ.get(SLOTS_ENV) is not None and slot(name) in SLOTS


def allowed(candidate: Candidate) -> bool:
    """``registry.select``'s switch check."""
    return enabled(candidate.name, candidate.group)


def parity() -> bool:
    """Parity mode (``SGLANG_NVIDIA_MEGAKERNEL_SLOTS=""``): no unit is on; every width
    the profile accepts runs the layer loop on the stock bodies."""
    import os

    from sgl_nvidia_mega.glm5_3.runtime import SLOTS, SLOTS_ENV

    return os.environ.get(SLOTS_ENV) is not None and not SLOTS
