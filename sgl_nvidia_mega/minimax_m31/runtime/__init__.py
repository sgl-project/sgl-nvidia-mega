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
per width; the plan (``plan.Slots``) records which slots serve each width. The DSpark
draft's slots (``DRAFT_SLOTS``) are requested in the same list and built by the draft
spec (``draft``, ``draft_step``); the target's build names them in its summary and
checks, from its executor, that the draft engaged them. This module stays light
(``build`` imports the rest at build): the spec, which SGLang's launcher imports, reads
the slots here.
"""

from __future__ import annotations

import os
from typing import Any

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py": ("widths",),
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
# - markov: a slot of the DSpark draft (DRAFT_SLOTS below)
SHIPPED_SLOTS: tuple[str, ...] = ("mega", "moe_nofill", "moe_swap", "markov")
# Slots that run only when SLOTS_ENV lists them:
# - draft_lm_head: a slot of the DSpark draft (DRAFT_SLOTS below)
# - lm_head_mxfp8: the step slot mega serves (8 rows) projects its final-normed hidden
#   rows through an MXFP8 copy of this rank's LM-head shard (E4M3 codes and UE8M0
#   scales of 32 columns, about 300 MiB a rank, made at load) on a weight-stream
#   kernel (logits, sm103/logits/lm_head_mxfp8.py), half the bytes of the BF16 shard
#   SGLang's GEMM reads; SGLang's cross-rank gather, FP32 cast and logits processing
#   follow. Not bitwise (E4M3 weights): spec.LM_HEAD_TOLERANCES states its bound
OPT_IN_SLOTS: tuple[str, ...] = ("draft_lm_head", "lm_head_mxfp8")
# The DSpark draft's slots: the draft spec builds them (draft.SPEC, draft_step), not
# the target's plan, and they replace SGLang's folded proposal, the tail of the draft's
# CUDA graph, at 1-16 requests. Neither is bitwise (DRAFT_TOLERANCES, in draft.NUMERICS):
# - markov: the proposal's Markov chain (7 steps of bias, sample and cross-rank
#   exchange) in one vocab-sharded kernel per rank (sampler, sm103/draft/markov.py),
#   through SGLang's draft_sampler hook. Its MMA sums each step's bias in another order
#   than SGLang's bias GEMM, so a kept step logit can differ. Under SGLang's sampling
#   draft (SGLANG_DSPARK_USE_GREEDY_DRAFT unset) the kept logits are the draft
#   probabilities the accept reads: the chain samples from the logits it keeps, so the
#   target's distribution is kept, but a seeded sampled request can accept other tokens
#   than with SGLang's proposal. Greedy requests return the target's argmax tokens,
#   identical while the draft tokens are; a batch's verify numerics depend on its
#   draft tokens
# - draft_lm_head: the draft's base logits from an MXFP8 copy of this rank's LM-head
#   shard (about 300 MiB a rank) on SGLang's FlashInfer MXFP8 linear, half the bytes of
#   the BF16 shard SGLang's projection reads. Not bitwise (E4M3 weights and draft
#   rows): its draft tokens differ, and with them the acceptance length and the verify
#   of the other rows of a batch
DRAFT_SLOTS: tuple[str, ...] = ("markov", "draft_lm_head")
# slot -> the slots it needs
NEEDS: dict[str, tuple[str, ...]] = {
    # it swaps the streams of moe_nofill's MoE
    "moe_swap": ("moe_nofill",),
    # the MXFP8 copy serves the chain's step
    "draft_lm_head": ("markov",),
    # the stream closes the one step slot mega serves
    "lm_head_mxfp8": ("mega",),
}
# The slots whose kernels agree with SGLang's bit for bit; slots mega and lm_head_mxfp8
# make the spec's numerics TOLERANCE (spec.NUMERICS)
BITWISE_SLOTS: frozenset[str] = frozenset({"moe_nofill", "moe_swap"})
# draft slot -> {quantity: its largest difference against SGLang's proposal}
DRAFT_TOLERANCES: dict[str, dict[str, float]] = {
    "markov": {
        # a kept step logit, per element: at most two BF16 ulps of |kept logit| + |base
        # logit| past one BF16 ulp of the step's largest |bias| (sampler.bound_ulps).
        # The step's 256 bias products are exact in FP32; the MMA sums them in another
        # order than SGLang's bias GEMM, so the BF16 biases agree or are one ulp apart.
        # That moves the FP32 sum base + bias by one ulp of |bias|, which is at most
        # |kept| + |base| plus half a kept ulp, so one binade above it at most; rounding
        # the sum adds at most one ulp of the kept logit. A dot product that cancels far
        # below its terms can differ by more ulps of its own small value, but by its
        # FP32 summation error only, far below one ulp of the step's largest |bias|.
        # The kernel test asserts it on Gaussian weights
        # (test/kernels/test_minimax_m31_markov.py); the parity probe
        # (SGLANG_NVIDIA_MEGAKERNEL_MARKOV_PARITY) measures it on each rank's shard
        "draft_markov.step_logits.bf16_ulps": 2.0,
        # draft tokens may differ only where two candidates are that close; bitwise
        # when the bias products sum exactly (the kernel test)
        "draft_markov.tokens.exact_sum_mismatches": 0.0,
    },
    "draft_lm_head": {
        # the base logits against SGLang's BF16 projection, over their largest
        # |value|: MXFP8 rounding of the weights and the draft rows (the kernel test
        # asserts it on Gaussian data)
        "draft_lm_head.logits.rel": 2**-4,
    },
}

SLOTS_ENV = "SGLANG_NVIDIA_MEGAKERNEL_SLOTS"


def requested_slots(environ=None) -> frozenset[str]:
    """``SLOTS_ENV`` (comma separated; "" turns every slot off: SGLang's step code at
    every width and SGLang's draft proposal, for tests and bisecting), else every
    shipped slot (the opt-in slots run only when listed)."""
    raw = (os.environ if environ is None else environ).get(SLOTS_ENV)
    if raw is None:
        return frozenset(SHIPPED_SLOTS)
    return frozenset(name.strip() for name in raw.split(",")) - {""}


SLOTS = requested_slots()


def target_slots(slots: frozenset[str]) -> frozenset[str]:
    """The requested slots of the target's covered steps (the draft spec builds the
    others)."""
    return frozenset(slots) - frozenset(DRAFT_SLOTS)


def draft_slots(slots: frozenset[str]) -> frozenset[str]:
    """The requested slots of the DSpark draft's proposal (draft.SPEC builds them)."""
    return frozenset(slots) & frozenset(DRAFT_SLOTS)


def draft_tolerances(slots: frozenset[str] = SLOTS) -> dict[str, float]:
    """The tolerances of the requested draft slots."""
    out: dict[str, float] = {}
    for slot in sorted(draft_slots(slots)):
        out.update(DRAFT_TOLERANCES[slot])
    return out


def slot_reasons(slots: frozenset[str] = SLOTS) -> list[str]:
    """Why the requested slots cannot be served ([] when they can)."""
    known = (*SHIPPED_SLOTS, *OPT_IN_SLOTS)
    reasons = [
        f"{SLOTS_ENV}: unknown slot {name!r} (known: {', '.join(known)})"
        for name in sorted(slots - set(known))
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
    if "lm_head_mxfp8" in slots and not reasons:
        from sgl_nvidia_mega.minimax_m31 import logits

        reasons += [f"slot lm_head_mxfp8: {r}" for r in logits.head_reasons(model)]
    return reasons


def build(model: Any, hf_config: Any, ctx: Any):
    """The executor of the covered steps; ``ctx`` is SGLang's BuildContext."""
    from sgl_nvidia_mega.minimax_m31 import MAX_REQUESTS, ROWS, logits, moe
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
    # it closes slot mega's step: without that step it is not built (and refused below)
    bound["lm_head_mxfp8"] = (
        logits.build(model, widths, SLOTS) if mega is not None else None
    )
    facts = Facts(
        mega=frozenset({ROWS}) if mega is not None else frozenset(),
        moe_nofill=frozenset(w for w in widths if moe.serves(bound["moe"], w)),
        moe_swap=frozenset(w for w in widths if moe.serves(bound["moe_swap"], w)),
        lm_head_mxfp8=(
            frozenset({ROWS}) if bound["lm_head_mxfp8"] is not None else frozenset()
        ),
    )
    # every rank must launch the same kernels at each width
    agree_tp(facts, what="slot plans (widths per slot)")
    plan = slot_plan.plan(widths, facts)
    # a requested slot whose build serves no width refuses the server (one whose
    # widths slot mega takes over is built, not failed); the draft's slots build with
    # the draft spec (draft.SPEC), which refuses on its own
    draft = sorted(draft_slots(SLOTS))
    missing = sorted(name for name in target_slots(SLOTS) if not getattr(facts, name))
    if missing:
        why = {
            "mega": "built" if mega is not None else "not requested",
            "moe_nofill": _reason(bound["moe"]),
            "moe_swap": _reason(bound["moe_swap"]),
            "lm_head_mxfp8": (
                "built"
                if bound["lm_head_mxfp8"] is not None
                else "not requested"
                if "lm_head_mxfp8" not in SLOTS
                else "slot mega's step is not built"
            ),
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
    if bound["lm_head_mxfp8"] is not None:
        summary += f"; lm_head_mxfp8: {bound['lm_head_mxfp8'].summary()}"
    engagement = None
    if draft:
        from sgl_nvidia_mega.minimax_m31.draft import (
            EngagementCheck,
            proposal_off_reason,
        )

        off = proposal_off_reason()
        if off:
            summary += f"; draft slots {draft} not engaged: {off}"
        else:
            summary += f"; draft slots {draft}, built by the DSpark draft's spec"
            # stops the server if the draft's folded proposal never asks for them
            engagement = EngagementCheck(draft)
    return MegakernelDecode(
        model,
        widths=widths,
        plan=plan,
        bound=bound,
        mega=mega,
        summary=summary,
        draft_check=engagement,
    )


def _reason(bound) -> str:
    """Why a slot's build serves no width: "not requested", its reason, or "built"."""
    if bound is None:
        return "not requested"
    return getattr(bound, "reason", "") or "built"
