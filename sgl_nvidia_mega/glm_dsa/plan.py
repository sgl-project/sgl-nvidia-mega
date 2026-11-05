# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team
"""Coordination layer: which step groups cover which slots and at which row counts.

``build_providers`` builds the kernel family objects and returns a
``Providers`` per verify row count; ``compose`` resolves dependency fallbacks
within one row count; ``families`` translates active slots into the kernel
family names the runtime passes to the build step.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from sgl_nvidia_mega.glm_dsa.eligibility import MAX_ROWS  # noqa: F401


# Steps of each group in call order (for logging, rank agreement, tests).
STEPS = {
    "attention": (
        "forward",
        "qkv_a",
        "prologue",
        "index_k",
        "index_q",
        "indexer",
        "q_absorb",
        "rope_cache",
        "sparse_mla",
        "uv_proj",
    ),
    "hops": ("pre_attention", "post_attention", "final"),
    "mlp": ("forward",),
}

# Maps a slot name to the kernel family group it populates.
FAMILY_SLOTS = {"attention": "attention", "hops": "hops", "moe": "mlp"}


def _attention():
    from sgl_nvidia_mega.glm_dsa.attention import AttentionGlue
    return AttentionGlue


def _hops():
    from sgl_nvidia_mega.glm_dsa.boundary import HopFamily
    return HopFamily


def _mlp():
    from sgl_nvidia_mega.glm_dsa.moe import MoeGlue
    return MoeGlue


def families(slots) -> Dict[str, str]:
    """Active kernel family names for the given set of slot names."""
    return {v: k for k, v in FAMILY_SLOTS.items() if k in slots}


def build_providers(model, rows: Tuple[int, ...], derived, slots=None):
    """(step groups per verify row count, built families).

    Each family module's ``build(model, derived)`` returns None (stay stock)
    or an object with ``compile(rows)`` and ``for_rows(m)``. When ``slots`` is
    given only the slot-enabled families are attempted.
    """
    from sgl_nvidia_mega.glm_dsa.providers import Providers, compose

    if slots is None:
        from sgl_nvidia_mega.glm_dsa.runtime import SLOTS
        slots = SLOTS

    glue_map = {
        "attention": ("attention", lambda m, d: _build_family("attention", m, d)),
        "hops": ("hops", lambda m, d: _build_family("hops", m, d)),
        "moe": ("mlp", lambda m, d: _build_family("mlp", m, d)),
    }

    family_objs: Dict[str, Any] = {}
    for slot_name, (group, builder) in glue_map.items():
        if slot_name not in slots:
            continue
        fam = builder(model, derived)
        if fam is not None:
            fam.compile(rows)
            family_objs[group] = fam

    out = {
        m: Providers(**compose(m, {g: f.for_rows(m) for g, f in family_objs.items()}))
        for m in rows
    }
    for prov in out.values():
        wire = getattr(prov.hops, "defer_input_norm", None)
        if wire is not None and getattr(prov.attention, "takes_deferred_norm", False):
            wire()
    return out, family_objs


def _build_family(slot_name: str, model, derived):
    if slot_name == "attention":
        from sgl_nvidia_mega.glm_dsa import attention
        return attention.build(model, derived)
    if slot_name == "hops":
        from sgl_nvidia_mega.glm_dsa import hops
        return hops.build(model, derived)
    if slot_name == "mlp":
        from sgl_nvidia_mega.glm_dsa import moe
        return moe.build(model, derived)
    return None
