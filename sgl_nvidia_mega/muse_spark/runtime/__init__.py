# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The built Muse Spark megakernel: what it serves, and the step it runs.

SKELETON. ``build`` reports that nothing was built, so a server started with
--enable-nvidia-megakernel on this architecture keeps running SGLang's own
forward and says so. Each slot lands here one at a time, with an A/B against
the stock forward before it is turned on.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

try:
    from sglang.kernels.plugin_kernels.nvidia.spec import Step
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.megakernels.spec import Step

from sgl_nvidia_mega.muse_spark.plan import Facts, plan

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": ("Step",),
}


# Slots whose kernels are built and measured. Empty: nothing is offered yet.
SLOTS: frozenset[str] = frozenset()


def slot_reasons() -> list[str]:
    """Why a slot cannot serve this configuration; empty when all can."""
    return []


def model_disable_reasons(model: Any) -> list[str]:
    """Why this loaded model cannot be served; empty when it can."""
    return []


class BuiltMuseSpark:
    """What SGLang calls for a covered step."""

    speculative_runtime_hooks = None

    def __init__(self, *, widths: tuple[int, ...], facts: Facts):
        self.widths = tuple(sorted(widths))
        self.facts = facts
        self.plan_by_width = plan(self.widths, facts)
        self.draft_sampler: Optional[Callable] = None

    def summary(self) -> str:
        """What was built, for SGLang's startup line."""
        if not self.widths:
            return "no slot built"
        slots = self.plan_by_width[self.widths[0]]
        on = sorted(f for f in slots.__struct_fields__ if getattr(slots, f))
        return f"slots {on}" if on else "no slot built"

    def plan(
        self, input_ids, forward_batch, input_embeds=None, pp_proxy_tensors=None
    ) -> Optional[Step]:
        """The step if this forward is covered, else None (SGLang's forward runs).

        Target-verify only, and only as SGLang captures it: the prefill and
        extend kernels of the plugin this is ported from need a binding that
        dispatches those steps (PORT-NOTES.md).
        """
        from sgl_nvidia_mega.muse_spark.spec import serves_this_step

        if (
            pp_proxy_tensors is not None
            or not forward_batch.forward_mode.is_target_verify()
        ):
            return None
        if not serves_this_step():
            return None
        rows = int(input_ids.shape[0])
        return Step(mode="target_verify", rows=rows) if rows in self.widths else None

    def forward(self, step: Step, original: Callable, *args: Any, **kwargs: Any) -> Any:
        """The covered step. No slot is built yet, so SGLang's forward runs."""
        return original(*args, **kwargs)


def build(model: Any, ctx: Any) -> BuiltMuseSpark:
    """Build the kernels for ctx.widths. Nothing is built yet, so no width is
    offered and SGLang logs that the spec covers no step."""
    return BuiltMuseSpark(widths=(), facts=Facts())
