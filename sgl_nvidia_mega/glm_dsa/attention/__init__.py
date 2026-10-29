# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team
"""Attention package: stock step bodies (step.py) and fused kernel family (fused.py)."""

from sgl_nvidia_mega.glm_dsa.attention.step import (  # noqa: F401
    AttentionDecode,
    IndexQ,
    PrologueOut,
    StepContext,
)

# AttentionGlue and build() are the fused kernel family for this slot.
from sgl_nvidia_mega.glm_dsa.attention.fused import AttentionGlue  # noqa: F401


def build(model, derived):
    """None if the slot is disabled; otherwise the built AttentionGlue."""
    from sgl_nvidia_mega.glm_dsa.runtime import SLOTS

    if "attention" not in SLOTS:
        return None
    return AttentionGlue.build(model, derived)
