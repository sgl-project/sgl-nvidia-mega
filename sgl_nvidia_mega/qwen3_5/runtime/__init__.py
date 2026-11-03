# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The megakernel runtime: the executor of a covered target-verify step (``executor``),
its per-step state (``state``), the embedding (``embedding``), the test-only parity
check (``parity``), and ``build``, which SGLang runs through the spec once the model is
prepared and before the KV cache is sized."""

from __future__ import annotations

import os

import msgspec

# The kernel slots that run by default, and the ones that run only when requested
# (until step-level timing and accuracy runs move them to the default list)
SHIPPED_SLOTS: tuple[str, ...] = ()
OPT_IN_SLOTS: tuple[str, ...] = (
    "route",
    "mkm",
    "mkm_wide",
    "gdn_verify",
    "gdn_pdl",
    "hop",
    "gdn_push",
    "gdn_front",
    "route_shared",
    "skv2",
    "mka",
    "mka_push",
)
# Steps of more than one request (TP8's decode graphs of batch sizes 2 and 4), covered only
# when requested in the same variable: covering them costs the one-request step about
# 0.3% at TP8
OPT_IN_COVERAGE: tuple[str, ...] = ("multi_request",)
# The MTP draft's EAGLE entries and kernel groups (qwen3_5/draft), requested in the same
# variable; the draft spec's build reads them, the target's ignores them
DRAFT_ENTRIES: tuple[str, ...] = ("draft_loop", "draft_extend")
DRAFT_SLOTS: tuple[str, ...] = (
    "draft_moe",
    "draft_q1",
    "draft_xchg",
    "draft_hop",
    "draft_front",
    "draft_skv2",
)
SLOTS_ENV = "SGLANG_NVIDIA_MEGAKERNEL_SLOTS"


def _requested_slots() -> frozenset:
    """``SLOTS_ENV`` (comma-separated; "" runs the stock step), else every shipped
    slot."""
    raw = os.environ.get(SLOTS_ENV)
    if raw is None:
        return frozenset(SHIPPED_SLOTS)
    return frozenset(name.strip() for name in raw.split(",")) - {""}


SLOTS = _requested_slots()


def slot_reasons() -> list[str]:
    """Why the requested slots cannot be served ([] when they can)."""
    known = (
        *SHIPPED_SLOTS,
        *OPT_IN_SLOTS,
        *OPT_IN_COVERAGE,
        *DRAFT_ENTRIES,
        *DRAFT_SLOTS,
    )
    reasons = [
        f"{SLOTS_ENV}: unknown slot {name!r} (known: {', '.join(known) or 'none'})"
        for name in sorted(SLOTS - set(known) - {"verify_sample"})
    ]
    if "verify_sample" in SLOTS:
        reasons.append(
            f"{SLOTS_ENV}: 'verify_sample' is no longer an entry: drop it; an SGLang "
            "with post-verify graphs turns them on with "
            "SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH=1"
        )
    if "mkm" in SLOTS and "route" not in SLOTS:
        reasons.append(f"{SLOTS_ENV}: slot 'mkm' runs on slot 'route''s outputs")
    if "mkm_wide" in SLOTS and "mkm" not in SLOTS:
        reasons.append(f"{SLOTS_ENV}: slot 'mkm_wide' is slot 'mkm''s 16-token engine")
    if "mkm_wide" in SLOTS and "multi_request" not in SLOTS:
        reasons.append(
            f"{SLOTS_ENV}: slot 'mkm_wide' serves the steps of more than one request "
            "('multi_request')"
        )
    if "route_shared" in SLOTS and "route" not in SLOTS:
        reasons.append(
            f"{SLOTS_ENV}: slot 'route_shared' runs in slot 'route''s launch"
        )
    if "gdn_push" in SLOTS and not {"gdn_verify", "hop"} <= SLOTS:
        reasons.append(
            f"{SLOTS_ENV}: slot 'gdn_push' pushes slot 'gdn_verify''s out_proj into "
            "slot 'hop'"
        )
    if "gdn_front" in SLOTS and not {"gdn_verify", "gdn_pdl"} <= SLOTS:
        reasons.append(
            f"{SLOTS_ENV}: slot 'gdn_front' runs slot 'gdn_pdl''s dependent verify"
        )
    if "gdn_pdl" in SLOTS and "gdn_verify" not in SLOTS:
        reasons.append(
            f"{SLOTS_ENV}: slot 'gdn_pdl' launches slot 'gdn_verify''s verify"
        )
    for name in sorted(SLOTS & set(DRAFT_SLOTS) - {"draft_skv2"}):
        if "draft_loop" not in SLOTS:
            reasons.append(f"{SLOTS_ENV}: slot {name!r} runs inside entry 'draft_loop'")
    if "draft_hop" in SLOTS and "hop" not in SLOTS:
        reasons.append(f"{SLOTS_ENV}: slot 'draft_hop' runs on slot 'hop''s mailbox")
    if "draft_skv2" in SLOTS and "draft_extend" not in SLOTS:
        reasons.append(
            f"{SLOTS_ENV}: slot 'draft_skv2' runs inside entry 'draft_extend'"
        )
    if "mka_push" in SLOTS and not {"mka", "hop"} <= SLOTS:
        reasons.append(
            f"{SLOTS_ENV}: slot 'mka_push' pushes slot 'mka''s epilogue into slot "
            "'hop''s mailbox"
        )
    return reasons


# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": (
        "widths",
        "tokens_per_request",
        "deadline_s",
        "binding",
    ),
    "srt/runtime_context.py": ("get_parallel", '"tp_size"'),
}


class PackageContext(msgspec.Struct, frozen=True):
    """What every package's ``build(model, ctx, group)`` gets."""

    widths: tuple[int, ...]  # the covered steps' rows (requests x rows a request)
    tokens_per_request: int
    deadline_s: float  # each cross-GPU wait's deadline


def build(model, ctx):
    """Each package's build, the slot plan per width and the executor; ``ctx`` is
    SGLang's BuildContext."""
    import torch

    from sgl_nvidia_mega.qwen3_5 import attention, boundary, gdn, moe
    from sgl_nvidia_mega.qwen3_5 import plan as slot_plan
    from sgl_nvidia_mega.qwen3_5.runtime import parity
    from sgl_nvidia_mega.qwen3_5.runtime.executor import Qwen35Verify

    multi = slot_plan.MULTI_REQUEST in SLOTS
    if multi:
        from sglang.srt.runtime_context import get_parallel

        multi = slot_plan.multi_request(get_parallel().tp_size, SLOTS)
    widths = slot_plan.candidates(ctx.widths, ctx.tokens_per_request, multi)
    if not widths:
        raise RuntimeError(
            f"NVIDIA megakernel: {ctx.tokens_per_request} rows a request leave no "
            "covered width"
        )
    pctx = PackageContext(
        widths=widths,
        tokens_per_request=ctx.tokens_per_request,
        deadline_s=ctx.deadline_s,
    )
    bound = {
        "moe": moe.build(model, pctx),
        "gdn": gdn.build(model, pctx),
        "boundary": boundary.build(model, pctx),
        "attention": attention.build(model, pctx),
    }
    facts = slot_plan.Facts(
        route=bound["moe"].route,
        mkm=bool(bound["moe"].mkm),
        mkm_wide=bound["moe"].mkm_wide,
        gdn_verify=bound["gdn"].gdn_verify,
        gdn_pdl=bound["gdn"].gdn_pdl,
        hop=bound["boundary"].hop is not None,
        gdn_push=bool(bound["gdn"].packed),
        gdn_front=bool(bound["gdn"].front),
        route_shared=bound["moe"].shared,
        skv2=bound["attention"].skv2,
        mka=bound["attention"].mka,
        mka_push=bound["attention"].mka_push,
    )
    covered = slot_plan.covered(pctx.widths, facts, pctx.tokens_per_request)
    slots = slot_plan.plan(covered, facts, pctx.tokens_per_request)
    # the slots of every width wider than one request, for the startup line
    wider = "".join(
        f"; {w} rows: {[n for n, ok in msgspec.structs.asdict(s).items() if ok]}"
        for w, s in slots.items()
        if w != pctx.tokens_per_request
    )
    built = msgspec.structs.asdict(facts)
    # every package's build agrees across ranks, so this raises on every rank alike
    target_slots = SLOTS - set(DRAFT_ENTRIES) - set(DRAFT_SLOTS) - set(OPT_IN_COVERAGE)
    # mkm's 16-token engine has nothing to serve where no covered step is wider than 7 rows
    vacuous = (
        {"mkm_wide"}
        if slot_plan.widest(pctx.widths, slot_plan.MKM_WIDE_MAX_ROWS)
        <= slot_plan.MKM_MAX_ROWS
        else set()
    )
    missing = sorted(name for name in target_slots - vacuous if not built[name])
    if missing:
        raise RuntimeError(
            f"NVIDIA megakernel: requested slots {missing} built no width (MoE: "
            f"{bound['moe'].reason or 'built'}; GDN: {bound['gdn'].reason or 'built'}; "
            f"boundary: {bound['boundary'].reason or 'built'}; "
            f"attention: {bound['attention'].reason or 'built'})"
        )
    checker = (
        parity.Parity(covered, torch.cuda.current_device(), model)
        if parity.ENABLED
        else None
    )
    on = sorted(name for name, ok in built.items() if ok)
    return Qwen35Verify(
        model,
        slots,
        bound=bound,
        widths=covered,
        binding=ctx.binding,
        summary=f"slots {on or 'none (stock)'}{wider}",
        parity=checker,
    )
