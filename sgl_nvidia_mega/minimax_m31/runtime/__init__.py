# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The megakernel runtime: the kernel slots, the per-rank checks of the loaded model
(``model_disable_reasons``) and the build (``build``): the widths the megakernel
serves, each slot's build result and the slot plan per width. The executor of a
covered step is ``executor``, over its per-step state (``state``); slot mega's step is
``mega``; ``agree`` holds the TP ranks' build-time agreement and ``parity`` the
composition check's log lines.

The kernel slots: which of the megakernel's kernels replace SGLang's in a covered step.
With every slot off the step functions run SGLang's own modules, so the covered step
is bit for bit SGLang's forward. A slot is requested by name (``SLOTS_ENV``) and built
per width; the plan (``plan.Slots``) records which slots serve each width. This module
stays light (``build`` imports the rest at build): the spec, which SGLang's launcher
imports, reads the slots here.
"""

from __future__ import annotations

import os
from typing import Any

from sgl_nvidia_mega.minimax_m31.runtime.parity import log_parity  # noqa: F401

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": (
        "widths",
        "binding",
    ),
    "srt/models/minimax_m3.py": (
        "MiniMaxM3DecoderLayer",
        "MiniMaxM3Model",
        "pp_group",
        "start_layer",
        "end_layer",
        "layers",
        "_norm_quant",
    ),
    "srt/models/minimax_m3_vl.py": ("model", "is_mrope_enabled", "logits_processor"),
    "srt/distributed/parallel_state.py": ("is_first_rank", "is_last_rank"),
}

# The slots this package ships; unset SLOTS_ENV runs all of them:
# - mega: one request's verify step (8 rows) on MegaAttention and the token-parallel
#   MoE (runtime/mega.py, sm103/), every MoE layer two persistent launches, with a
#   copy of the routed experts in the kernels' layout (about 1 GiB a layer, built only
#   with this slot). Not bitwise: spec.NUMERICS states its tolerances
# - moe_nofill: at 2-16 requests (16-128 rows; at 8 rows too where mega is off) the
#   routed experts' per-token NVFP4 input quantized into buffers built once, without
#   the chain's fill launch (its global scale's torch.full) and allocations per MoE
#   layer (moe/moe_nofill.py). Bitwise: SGLang's kernels with SGLang's arguments, the
#   quantize checked against FlashInfer's own call at build
# - moe_swap: moe_nofill's MoE with the routed chain on the main stream, where its
#   router GEMM is the FFN-input boundary kernel's PDL successor, and the shared expert
#   on the alternate stream, swapping SGLang's two streams (moe/moe_swap.py). The same
#   kernels on the same inputs: bitwise
SHIPPED_SLOTS: tuple[str, ...] = ("mega", "moe_nofill", "moe_swap")
# slot -> the slots it needs
NEEDS: dict[str, tuple[str, ...]] = {
    # it swaps the streams of moe_nofill's MoE
    "moe_swap": ("moe_nofill",),
}
# The slots whose kernels agree with SGLang's bit for bit; slot mega makes the spec's
# numerics TOLERANCE (spec.NUMERICS)
BITWISE_SLOTS: frozenset[str] = frozenset({"moe_nofill", "moe_swap"})

SLOTS_ENV = "SGLANG_NVIDIA_MEGAKERNEL_SLOTS"


def requested_slots(environ=None) -> frozenset[str]:
    """``SLOTS_ENV`` (comma separated; "" turns every slot off: SGLang's step code at
    every width, for tests and bisecting), else every shipped slot."""
    raw = (os.environ if environ is None else environ).get(SLOTS_ENV)
    if raw is None:
        return frozenset(SHIPPED_SLOTS)
    return frozenset(name.strip() for name in raw.split(",")) - {""}


SLOTS = requested_slots()


def slot_reasons(slots: frozenset[str] = SLOTS) -> list[str]:
    """Why the requested slots cannot be served ([] when they can)."""
    reasons = [
        f"{SLOTS_ENV}: unknown slot {name!r} (known: {', '.join(SHIPPED_SLOTS)})"
        for name in sorted(slots - set(SHIPPED_SLOTS))
    ]
    reasons += [
        f"{SLOTS_ENV}: slot {slot!r} needs slot {need!r}"
        for slot in sorted(slots)
        for need in NEEDS.get(slot, ())
        if need not in slots
    ]
    return reasons


def model_disable_reasons(model: Any, hf_config: Any, slots=None) -> list[str]:
    """What the server arguments do not show: the loaded classes the executor
    follows, per rank, and with slot mega its layer loop and operands."""
    from sglang.srt.models.minimax_m3 import MiniMaxM3DecoderLayer, MiniMaxM3Model

    slots = SLOTS if slots is None else slots
    m = getattr(model, "model", None)
    if not isinstance(m, MiniMaxM3Model):
        return [f"the language model is {type(m).__name__}, not MiniMaxM3Model"]
    reasons = []
    if getattr(model, "is_mrope_enabled", False):
        reasons.append("the text model uses mrope positions")
    if not (m.pp_group.is_first_rank and m.pp_group.is_last_rank):
        reasons.append("the model is split across pipeline ranks")
    if getattr(model, "logits_processor", None) is None:
        reasons.append("the model has no logits processor (encoder only)")
    other = [
        i
        for i in range(m.start_layer, m.end_layer)
        if not isinstance(m.layers[i], MiniMaxM3DecoderLayer)
    ]
    if other:
        reasons.append(f"layers {other} are not MiniMaxM3DecoderLayer")
    # gfx950's norms quantize for the GEMMs that read them (kwargs the executor omits)
    quantizing = [
        i
        for i in range(m.start_layer, m.end_layer)
        if getattr(m.layers[i], "_norm_quant", None) is not None
    ]
    if quantizing:
        reasons.append(f"layers {quantizing} quantize in their input norms")
    if "mega" in slots and not reasons:
        from sgl_nvidia_mega.minimax_m31.runtime import mega

        reasons += mega.model_disable_reasons(model, hf_config)
    return reasons


def build(model: Any, hf_config: Any, ctx: Any):
    """The executor of the covered steps; ``ctx`` is SGLang's BuildContext."""
    from sgl_nvidia_mega.minimax_m31 import MAX_REQUESTS, ROWS, moe
    from sgl_nvidia_mega.minimax_m31 import plan as slot_plan
    from sgl_nvidia_mega.minimax_m31.plan import Facts, served_widths
    from sgl_nvidia_mega.minimax_m31.runtime.agree import agree_tp
    from sgl_nvidia_mega.minimax_m31.runtime.executor import MegakernelDecode

    widths = served_widths(tuple(ctx.widths))
    if not widths:
        raise RuntimeError(
            f"NVIDIA megakernel: no width of {list(ctx.widths)} is a verify step of "
            f"1-{MAX_REQUESTS} requests ({ROWS} rows a request)"
        )
    mega = None
    if "mega" in SLOTS:
        if ROWS not in widths:
            raise RuntimeError(
                f"NVIDIA megakernel: slot mega serves one request's {ROWS} verify "
                f"rows, which the widths {list(widths)} miss"
            )
        from sgl_nvidia_mega.minimax_m31.runtime.mega import MegaStep

        mega = MegaStep(model, hf_config, ctx)
    # each slot's build result (None with its slot off) and the widths it serves
    bound: dict[str, Any] = {"moe": moe.build(model, widths, SLOTS)}
    bound["moe_swap"] = moe.build_swap(model, widths, SLOTS, bound["moe"])
    facts = Facts(
        mega=frozenset({ROWS}) if mega is not None else frozenset(),
        moe_nofill=frozenset(w for w in widths if moe.serves(bound["moe"], w)),
        moe_swap=frozenset(w for w in widths if moe.serves(bound["moe_swap"], w)),
    )
    # every rank must launch the same kernels at each width
    agree_tp(facts, what="slot plans (widths per slot)")
    plan = slot_plan.plan(widths, facts)
    # a requested slot whose build serves no width refuses the server (one whose
    # widths slot mega takes over is built, not failed)
    missing = sorted(name for name in SLOTS if not getattr(facts, name))
    if missing:
        why = {
            "mega": "built" if mega is not None else "not requested",
            "moe_nofill": _reason(bound["moe"]),
            "moe_swap": _reason(bound["moe_swap"]),
        }
        raise RuntimeError(
            f"NVIDIA megakernel: requested slots {missing} serve no width of "
            f"{list(widths)} ("
            + "; ".join(f"{name}: {why[name]}" for name in why)
            + ")"
        )
    summary = (
        f"verify steps of {widths[0]}-{widths[-1]} rows ({ROWS} a request), slots by "
        f"width {slot_plan.describe(plan)}"
    )
    if mega is not None:
        summary += f"; mega: {mega.summary()}"
    return MegakernelDecode(
        model,
        widths=widths,
        plan=plan,
        bound=bound,
        mega=mega,
        binding=getattr(ctx, "binding", None),
        summary=summary,
    )


def _reason(bound) -> str:
    """Why a slot's build serves no width: "not requested", its reason, or "built"."""
    if bound is None:
        return "not requested"
    return getattr(bound, "reason", "") or "built"
