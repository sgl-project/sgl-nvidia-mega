# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The step groups of one covered verify step, per width, fixed at build.

Per decoder layer the executor's loop calls ``HopSteps.pre_attention`` (the previous
MLP output's all-reduce, residual add and ``input_layernorm``),
``AttentionStep.forward`` (up to the ``o_proj`` input), ``HopSteps.post_attention``
(``o_proj``, its all-reduce, residual add and ``post_attention_layernorm``) and
``MlpSteps.forward`` (un-reduced output); then ``HopSteps.final``. The default bodies
reproduce SGLang's verify step bit for bit, so with every group on its default the
loop equals SGLang's forward. A kernel family subclasses a group and overrides only
the steps it replaces, at the widths it serves; ``compose`` gives each width the
groups that serve together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

# Verify rows a request: MTP / EAGLE with top-1 drafts verifies speculative_num_steps
# draft tokens + 1 (SGLang's speculative_num_draft_tokens). The values served, each
# tested at its widths: 3 and 5 steps.
SERVED_TOKENS_PER_REQUEST = (4, 6)
# The widest covered step (the kernels' row cap)
MAX_ROWS = 16


def supported_rows(tokens_per_request: int) -> Tuple[int, ...]:
    """The covered widths of verify steps of ``tokens_per_request`` rows a request:
    every whole number of requests up to ``MAX_ROWS`` rows (4 rows: 1-4 requests, 4 /
    8 / 12 / 16 rows; 6 rows: 1-2 requests, 6 / 12 rows); none for a value that is not
    served. Steps with more requests run SGLang's path."""
    if tokens_per_request not in SERVED_TOKENS_PER_REQUEST:
        return ()
    return tuple(range(tokens_per_request, MAX_ROWS + 1, tokens_per_request))


def draft_length_reason(tokens_per_request: int) -> str:
    """Why verify steps of ``tokens_per_request`` rows a request are not served."""
    tokens = " or ".join(str(t) for t in SERVED_TOKENS_PER_REQUEST)
    steps = " or ".join(str(t - 1) for t in SERVED_TOKENS_PER_REQUEST)
    return (
        f"speculative_num_draft_tokens {tokens_per_request} is not served: MTP / EAGLE "
        f"with top-1 drafts is served at speculative_num_draft_tokens {tokens} "
        f"(speculative_num_steps {steps})"
    )


class FamilyDeclined(Exception):
    """A kernel family cannot serve this model; with its slot requested every rank
    refuses, naming the reason."""


# The steps of each group, in call order (describe, rank agreement, logs).
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

# Each step group's family and the slot that requests it
FAMILY_SLOTS = {"attention": "attention", "hops": "hops", "mlp": "moe"}


def _attention():
    from sgl_nvidia_mega.glm5_3.attention.step import AttentionStep

    return AttentionStep()


def _hops():
    from sgl_nvidia_mega.glm5_3.boundary.boundaries import HopSteps

    return HopSteps()


def _mlp():
    from sgl_nvidia_mega.glm5_3.moe.step import MlpSteps

    return MlpSteps()


@dataclass(eq=False)
class Providers:
    """The three step groups served at one verify width."""

    attention: Any = field(default_factory=_attention)
    hops: Any = field(default_factory=_hops)
    mlp: Any = field(default_factory=_mlp)

    def describe(self) -> Dict[str, str]:
        """Which class serves each step (logs, rank agreement, tests)."""
        out = {}
        for g, names in STEPS.items():
            cls = type(getattr(self, g))
            for n in names:
                out[f"{g}.{n}"] = getattr(cls, n).__qualname__.rsplit(".", 1)[0]
        return out


def families(slots) -> Tuple[Tuple[str, Any], ...]:
    """(group, family glue module) of each requested kernel family, in build order."""
    from sgl_nvidia_mega.glm5_3 import boundary, moe
    from sgl_nvidia_mega.glm5_3.attention import fused

    glue = {"attention": fused, "hops": boundary, "mlp": moe}
    return tuple((g, glue[g]) for g in STEPS if FAMILY_SLOTS[g] in slots)


def build_providers(model, rows: Tuple[int, ...], derived, slots):
    """(step groups per verify width, built families, declined reasons). Each
    requested family's glue module has ``build(model, derived)``: an object with
    ``compile(rows)`` (through ``compile_cached``) and ``for_rows(m)`` (its step group
    at ``m`` rows, None for stock), None for a model without that family's layers, or
    a ``FamilyDeclined`` / ``ValueError`` naming why it cannot serve them."""
    built: Dict[str, Any] = {}
    declined: Dict[str, str] = {}
    for group, glue in families(slots):
        try:
            family = glue.build(model, derived)
            if family is not None:
                family.compile(rows)
        except (FamilyDeclined, ValueError) as e:
            declined[group] = str(e)
            continue
        if family is not None:
            built[group] = family
    out = {
        m: Providers(**compose(m, {g: f.for_rows(m) for g, f in built.items()}))
        for m in rows
    }
    for prov in out.values():
        # the deferred input norm needs its producer (the fold hop) and its consumer
        # (the fused attention's epilogue-scaled projection) at the same width
        wire = getattr(prov.hops, "defer_input_norm", None)
        if wire is not None and getattr(prov.attention, "takes_deferred_norm", False):
            wire()
    return out, built, declined


def compose(rows: int, groups: Dict[str, Any]) -> Dict[str, Any]:
    """The step groups served together at ``rows`` (stock where None). A group
    whose ``requires`` names a family that is stock here is replaced by its
    ``fallback`` (checked the same way), or by stock without one."""
    chosen = {g: v for g, v in groups.items() if v is not None}
    changed = True
    while changed:
        changed = False
        for g, v in list(chosen.items()):
            if any(r not in chosen for r in getattr(v, "requires", ())):
                chosen[g] = getattr(v, "fallback", None)
                if chosen[g] is None:
                    del chosen[g]
                changed = True
    return chosen
