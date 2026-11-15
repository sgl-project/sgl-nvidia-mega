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


def gemm2_options(slots) -> dict:
    """The routed GEMM2 build the slots ask for (slot gemm2_pipe; slot gemm2_reg_pool:
    that build capped at GEMM2_POOL_REGISTERS per thread)."""
    from sgl_nvidia_mega.deepseek_v41.sm103.moe.routed_expert_gemm import (
        GEMM2_POOL_REGISTERS,
    )

    pipe = "gemm2_pipe" in slots
    registers = GEMM2_POOL_REGISTERS if pipe and "gemm2_reg_pool" in slots else 0
    return dict(gemm2_pipe=pipe, gemm2_registers=registers)


def gemm2_pool_summary(bound: routed.Bound) -> str:
    """The register-pool GEMM2 as built: its cap, registers and spill bytes per thread."""
    from sgl_nvidia_mega.deepseek_v41.sm103.moe.routed_expert_gemm import (
        kernel_registers,
    )

    gemms = [s.gemm for s in bound.slots.values() if s.gemm.gemm2_registers]
    if not gemms:
        return "off"
    regs, local = kernel_registers(gemms[0].gemm2_pipe)
    return (
        f"cap {gemms[0].gemm2_registers} in {len(gemms)} layers, {regs} registers "
        f"and {local} spill bytes per thread"
    )


def build(model, ctx, group) -> routed.Bound:
    """The routed-expert GEMM pair of every MoE layer (slot ``moe``; slot ``gate``:
    its fast top-k); the result is ``st.bound["moe"]``."""
    from sgl_nvidia_mega.deepseek_v41.runtime import SLOTS
    from sgl_nvidia_mega.deepseek_v41.sm103.moe.router_mma import gate_route_release

    if "moe" not in SLOTS:
        return routed.Bound(reason="stock (SGLANG_NVIDIA_MEGAKERNEL_SLOTS)")
    bound = routed.build(
        model,
        group,
        gate_topk="gate" in SLOTS,
        shared_ll="shared_ll" in SLOTS,
        widths=ctx.widths,
        **gemm2_options(SLOTS),
    )
    logger.info(
        "NVIDIA megakernel: DeepSeek-V4.1 MoE: routed experts on the GEMM pair in %d "
        "layers, fast gate in %d, shared expert on the low-latency GEMM in %d (%s; "
        "shared_ll: %s); early router (gate_route_early, top-k releases GEMM1 at "
        "GATE_ROUTE_RELEASE %d) in %d (%s)",
        len(bound.slots),
        sum(s.gate for s in bound.slots.values()),
        sum(s.shared is not None for s in bound.slots.values()),
        bound.reason or "built",
        bound.shared_reason or "built",
        gate_route_release(),
        sum(s.router is not None for s in bound.slots.values()),
        bound.router_reason or "built",
    )
    if "gemm2_reg_pool" in SLOTS:
        logger.info(
            "NVIDIA megakernel: DeepSeek-V4.1 MoE: GEMM2 register pool "
            "(gemm2_reg_pool): %s",
            gemm2_pool_summary(bound),
        )
    return bound


__all__ = ["build", "gemm2_options", "gemm2_pool_summary", "step"]
