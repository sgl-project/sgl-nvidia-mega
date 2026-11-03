# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The Muse Spark kernels, registered as hooks on SGLang's own forward.

Each component replaces one call inside SGLang's forward -- a norm, an
attention call, a MoE finalize, a boundary reduction -- through
``sglang.srt.plugins.hook_registry``. Every one is opt-in and default off, so a
server that sets none of the ``MUSE_VR_*`` switches runs stock SGLang.

ORDER IS LOAD-BEARING. Several components hook the same call and must wrap each
other in a fixed order; the comments on COMPONENTS say which and why. Do not
sort this list.

A component that cannot engage -- a switch unset, a pinned SGLang source that
has moved, a model that does not match -- reports why and leaves SGLang stock.
It never half-engages: that is the difference between a run that is slower and
a run whose numbers cannot be trusted.
"""

from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)

# Order matters; see the module docstring.
COMPONENTS = (
    "ar_attn_boundary",
    "pull_bounded",
    "verify_logits_gather",
    "ctx_proj_prefill",
    "draft_candidates",
    "draft_boundary",
    "draft_attn",
    "eager_commit",
    "final_norm",
    "input_norm",
    "verify_attn",
    "moe_tactic",
    "prefill_moe_tactic",
    # before sp_fp4_dispatch: that hook adds the per-token scales and calls this one
    "prefill_moe_x",
    "sp_combine_inplace",
    # wraps the finalize op's CuTe-DSL call (sp_combine_inplace only picks its destination)
    "pfin",
    "small_extend_attn",
    # own extend-attention op; refuses to engage next to extend_attn / small_extend_attn
    "xattn_b",
    "sp_fp4_dispatch",
    "prefill_qkn",
)

# Components whose entry point is not called ``register``.
ENTRY: dict[str, str] = {}

_registered: dict[str, bool] | None = None


def register_all() -> dict[str, bool]:
    """Register every component once; returns name -> engaged.

    A component that raises is recorded as not engaged and SGLang stays stock
    for it: a failed opt-in must never take the server down.
    """
    global _registered
    if _registered is not None:
        return _registered
    from sgl_nvidia_mega.muse_spark.components import bed

    _registered = {}
    for name in COMPONENTS:
        try:
            mod = importlib.import_module(f"{__name__}.{name}")
            ok = bool(getattr(mod, ENTRY.get(name, "register"))())
        except Exception as e:  # noqa: BLE001 - a failed opt-in leaves SGLang stock
            bed.report(name, False, reason=f"registration failed: {type(e).__name__}: {e}")
            ok = False
        _registered[name] = ok
    on = sorted(k for k, v in _registered.items() if v)
    logger.info("muse_spark: %d of %d components engaged: %s", len(on), len(COMPONENTS), on)
    return _registered
