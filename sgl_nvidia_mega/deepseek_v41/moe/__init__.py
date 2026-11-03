# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The MoE step over a layer's MoE block weights (owner: MoE lane): router gate,
top-k, the routed experts (the GEMM pair at 1-48 rows, else FlashInfer's MoE), the
shared expert on the side stream, and a deferred finalize.

``step`` returns the MoE output in the form ``boundary.post(layer, "ffn")`` takes
(SGLang's MoE pieces or deferred finalize, as with ``leave_output_sum=True``).
Today it is the stock step (``moe/step.py``).
"""

from __future__ import annotations

import logging

from sgl_nvidia_mega.deepseek_v41.moe import routed
from sgl_nvidia_mega.deepseek_v41.moe.step import step

logger = logging.getLogger(__name__)


def build(model, ctx, group) -> routed.Bound:
    """The routed-expert GEMM pair of every MoE layer (slot ``moe``; slot ``gate``:
    its fast top-k); the result is ``st.bound["moe"]``."""
    from sgl_nvidia_mega.deepseek_v41.runtime import SLOTS

    if "moe" not in SLOTS:
        return routed.Bound(reason="stock (SGLANG_NVIDIA_MEGAKERNEL_SLOTS)")
    bound = routed.build(
        model,
        group,
        gate_topk="gate" in SLOTS,
        shared_ll="shared_ll" in SLOTS,
        widths=ctx.widths,
    )
    logger.info(
        "NVIDIA megakernel: DeepSeek-V4.1 MoE: routed experts on the GEMM pair in %d "
        "layers, fast gate in %d, shared expert on the low-latency GEMM in %d (%s; "
        "shared_ll: %s)",
        len(bound.slots),
        sum(s.gate for s in bound.slots.values()),
        sum(s.shared is not None for s in bound.slots.values()),
        bound.reason or "built",
        bound.shared_reason or "built",
    )
    return bound


__all__ = ["build", "step"]
