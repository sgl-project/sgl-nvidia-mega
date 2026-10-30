# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The megakernel runtime: the executor of a covered target-verify step (``executor``),
its per-step state (``state``), the embedding (``embedding``), the test-only parity
check (``parity``), and ``build``, which SGLang runs through the spec once the model is
prepared and before the KV cache is sized."""

from __future__ import annotations

import logging
import os

import msgspec

logger = logging.getLogger(__name__)

# Bitwise, shape-bounded paths run whenever the Qwen3.5 megakernel is enabled. Numeric
# paths stay behind the one comma-separated slot policy below; BAL is opt-in too.
SHIPPED_SLOTS: tuple[str, ...] = (
    "route_topk",
    "mkm_m32",
    "mkm_m64",
    "mkm_m128",
    "mkm_m256",
    "mkm_m512",
    "gdn_verify",
    "gdn_pdl",
    "gdn_conv",
    "hop",
    "hop_fp8",
    "multi_request",
    "draft_loop",
    "draft_extend",
    "draft_multi",
    "draft_embed",
)
OPT_IN_SLOTS: tuple[str, ...] = (
    "route",
    "route_gemm",
    "mkm",
    "mkm_wide",
    "gdn_core",
    "hop_mid",
    "gdn_push",
    "gdn_front",
    "gdn_trt_front",
    "route_shared",
    "route_lin",
    "skv2",
    "mka",
    "mka_push",
    "lowm_in",
    "lowm_out",
    "lowm_se",
    "lowm_push",
)
# Coverage names requested in the same variable have no kernel build result of their
# own. Multi-request coverage is shipped because every covered width keeps its own
# shape instead of padding.
OPT_IN_COVERAGE: tuple[str, ...] = ("multi_request",)
# Graph or build variants of a slot, requested in the same variable: no width of their
# own. BAL stays opt-in because its serving evidence covers TP8 only.
# skv2_cp: one long request past the measured crossover runs CP-lite in place of SKV2,
# in a graph variant of its own (attention/route.py)
OPT_IN_VARIANTS: tuple[str, ...] = ("skv2_cp", "mkm_wide_bal")
# The MTP draft's EAGLE entries and kernel groups (qwen3_5/draft), requested in the same
# variable; the draft spec's build reads them, the target's ignores them
DRAFT_ENTRIES: tuple[str, ...] = ("draft_loop", "draft_extend")
DRAFT_SLOTS: tuple[str, ...] = (
    "draft_moe",
    "draft_q1",
    "draft_xchg",
    "draft_hop",
    # numeric: hand deferred MoE rows to a fused final norm
    "draft_fin",
    "draft_front",
    "draft_skv2",
    # draft_loop and draft_extend also serve TP8's draft graphs of 2 to 128 requests,
    # and those of two or more take the sharded top-1 (with multi_request)
    "draft_multi",
    # Their embedding rows are multicast by the rank holding the token, in place of
    # the vocabulary-parallel embedding's all-reduce.
    "draft_embed",
    # One long request's steps / extend use CP-lite in graph variants of their own.
    "draft_q1_cp",
    "draft_skv2_cp",
)
CP_LITE_SLOTS = frozenset({"skv2_cp", "draft_q1_cp", "draft_skv2_cp"})
# Provider callbacks consume these names from the shared slot policy.  They do not
# build a target-verify width of their own.
HOOK_SLOTS: tuple[str, ...] = ("sharded_gumbel",)
SLOTS_ENV = "SGLANG_NVIDIA_MEGAKERNEL_SLOTS"
# These slots run all 512 experts themselves. Under expert parallelism, keep the
# route-only and boundary slots but let SGLang's EP path run the routed experts.
EP_STOCK_SLOTS = frozenset(
    {
        "mkm",
        "mkm_wide",
        "mkm_m32",
        "mkm_m64",
        "mkm_m128",
        "mkm_m256",
        "mkm_m512",
        "mkm_wide_bal",
        "draft_moe",
    }
)


def _requested_slots() -> frozenset:
    """``SLOTS_ENV`` (comma-separated; "" runs the stock step), else every shipped
    slot."""
    raw = os.environ.get(SLOTS_ENV)
    if raw is None:
        return frozenset(SHIPPED_SLOTS)
    return frozenset(name.strip() for name in raw.split(",")) - {""}


SLOTS = _requested_slots()


def apply_cp_lite_topology(attention_tp_size: int) -> frozenset:
    """Drop CP-lite slots when the attention-TP Q/KV topology cannot serve them."""
    from sgl_nvidia_mega.qwen3_5 import cp_lite_topology

    global SLOTS
    if cp_lite_topology(attention_tp_size) is not None:
        return frozenset()
    dropped = SLOTS & CP_LITE_SLOTS
    SLOTS = SLOTS - CP_LITE_SLOTS
    if dropped:
        logger.info(
            "NVIDIA megakernel: Qwen3.5 attention-TP %d: CP slots %s use stock "
            "(unsupported Q/KV-head topology)",
            attention_tp_size,
            sorted(dropped),
        )
    return dropped


def build_cp_lite(binding) -> None:
    """Collectively build the CP slots' process-local state before validation.

    The allocation and rendezvous order must be identical on every attention-TP
    rank, so this runs once from each Qwen3.5 plugin build, before any package checks
    ``CpLite._instance`` or CUDA graph variant is created.
    """
    try:
        from sglang.srt.kernel_plugins.nvidia.binding import megakernel_agree
    except ImportError:  # pragma: no cover - the pre-kernel_plugins SGLang path
        from sglang.srt.megakernels.runtime import megakernel_agree

    requested = tuple(sorted(SLOTS & CP_LITE_SLOTS))
    megakernel_agree(
        requested,
        what="Qwen3.5 CP slots requested",
    )
    if not requested:
        return
    megakernel_agree(
        (binding.page_size, binding.tp_size),
        what="Qwen3.5 CP page size and TP size",
    )
    if binding.page_size != 64:
        raise RuntimeError(
            "NVIDIA megakernel: Qwen3.5 CP slots need page size 64 "
            f"(got {binding.page_size})"
        )

    import torch

    from sgl_nvidia_mega.qwen3_5.sm103.attention.q35_cp_lite import CpLite

    cp = CpLite.maybe_create(binding, torch.device("cuda", torch.cuda.current_device()))
    if cp is None:
        raise RuntimeError(
            "NVIDIA megakernel: Qwen3.5 CP slots need a supported attention-TP "
            "Q/KV-head topology"
        )


def apply_expert_parallel(ep_size: int) -> frozenset:
    """Drop ``EP_STOCK_SLOTS`` from ``SLOTS`` when the MoE runs expert-parallel
    (``ep_size`` > 1); returns the dropped names. Every process calls it before reading
    ``SLOTS`` (config check, target build, draft build)."""
    global SLOTS
    if ep_size <= 1:
        return frozenset()
    dropped = SLOTS & EP_STOCK_SLOTS
    SLOTS = SLOTS - EP_STOCK_SLOTS
    return dropped


def slot_reasons() -> list[str]:
    """Why the requested slots cannot be served ([] when they can)."""
    known = tuple(
        dict.fromkeys(
            (
                *SHIPPED_SLOTS,
                *OPT_IN_SLOTS,
                *OPT_IN_COVERAGE,
                *OPT_IN_VARIANTS,
                *DRAFT_ENTRIES,
                *DRAFT_SLOTS,
                *HOOK_SLOTS,
            )
        )
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
    if "mkm_m32" in SLOTS and "route_topk" not in SLOTS:
        reasons.append(f"{SLOTS_ENV}: slot 'mkm_m32' takes slot 'route_topk''s rows")
    if "mkm_m32" in SLOTS and "multi_request" not in SLOTS:
        reasons.append(
            f"{SLOTS_ENV}: slot 'mkm_m32' serves the steps of more than two requests "
            "('multi_request')"
        )
    if "mkm_m64" in SLOTS and "route_topk" not in SLOTS:
        reasons.append(
            f"{SLOTS_ENV}: slot 'mkm_m64' runs on slot 'route_topk''s routing"
        )
    if "mkm_m64" in SLOTS and "multi_request" not in SLOTS:
        reasons.append(
            f"{SLOTS_ENV}: slot 'mkm_m64' serves the steps of more than four requests "
            "('multi_request')"
        )
    if "mkm_m128" in SLOTS and "route_topk" not in SLOTS:
        reasons.append(
            f"{SLOTS_ENV}: slot 'mkm_m128' runs on slot 'route_topk''s routing"
        )
    if "mkm_m128" in SLOTS and "multi_request" not in SLOTS:
        reasons.append(
            f"{SLOTS_ENV}: slot 'mkm_m128' serves the steps of more than nine requests "
            "('multi_request')"
        )
    for name, more in (("mkm_m256", "18"), ("mkm_m512", "36")):
        if name in SLOTS and "route_topk" not in SLOTS:
            reasons.append(
                f"{SLOTS_ENV}: slot '{name}' runs on slot 'route_topk''s routing"
            )
        if name in SLOTS and "multi_request" not in SLOTS:
            reasons.append(
                f"{SLOTS_ENV}: slot '{name}' serves the steps of more than {more} requests "
                "('multi_request')"
            )
    if (
        "mkm_wide_bal" in SLOTS
        and not {
            "mkm_m256",
            "mkm_m512",
        }
        & SLOTS
    ):
        reasons.append(
            f"{SLOTS_ENV}: slot 'mkm_wide_bal' changes the work split of slot "
            "'mkm_m256' or 'mkm_m512'"
        )
    if "route_shared" in SLOTS and "route" not in SLOTS:
        reasons.append(
            f"{SLOTS_ENV}: slot 'route_shared' runs in slot 'route''s launch"
        )
    if "route_gemm" in SLOTS and "route_topk" not in SLOTS:
        reasons.append(f"{SLOTS_ENV}: slot 'route_gemm' runs inside slot 'route_topk'")
    if "hop_mid" in SLOTS and "hop" not in SLOTS:
        reasons.append(f"{SLOTS_ENV}: slot 'hop_mid' widens slot 'hop'")
    if "gdn_push" in SLOTS and not {"gdn_verify", "hop"} <= SLOTS:
        reasons.append(
            f"{SLOTS_ENV}: slot 'gdn_push' pushes slot 'gdn_verify''s out_proj into "
            "slot 'hop'"
        )
    if "gdn_front" in SLOTS and not {"gdn_verify", "gdn_pdl"} <= SLOTS:
        reasons.append(
            f"{SLOTS_ENV}: slot 'gdn_front' runs slot 'gdn_pdl''s dependent verify"
        )
    if "gdn_trt_front" in SLOTS and not {"gdn_front", "gdn_core"} <= SLOTS:
        reasons.append(
            f"{SLOTS_ENV}: slot 'gdn_trt_front' replaces slot 'gdn_front' in front "
            "of slot 'gdn_core'"
        )
    if "gdn_core" in SLOTS and "gdn_verify" not in SLOTS:
        reasons.append(
            f"{SLOTS_ENV}: slot 'gdn_core' runs slot 'gdn_verify''s step in one launch"
        )
    if "gdn_conv" in SLOTS and "gdn_verify" not in SLOTS:
        reasons.append(f"{SLOTS_ENV}: slot 'gdn_conv' runs slot 'gdn_verify''s conv")
    if "gdn_pdl" in SLOTS and "gdn_verify" not in SLOTS:
        reasons.append(
            f"{SLOTS_ENV}: slot 'gdn_pdl' launches slot 'gdn_verify''s verify"
        )
    for name in sorted(SLOTS & set(DRAFT_SLOTS) - {"draft_skv2", "draft_skv2_cp"}):
        if "draft_loop" not in SLOTS:
            reasons.append(f"{SLOTS_ENV}: slot {name!r} runs inside entry 'draft_loop'")
    if "draft_hop" in SLOTS and "hop" not in SLOTS:
        reasons.append(f"{SLOTS_ENV}: slot 'draft_hop' runs on slot 'hop''s mailbox")
    if "draft_fin" in SLOTS and "draft_hop" not in SLOTS:
        reasons.append(
            f"{SLOTS_ENV}: slot 'draft_fin' extends slot 'draft_hop''s fused final norm"
        )
    if "draft_multi" in SLOTS and "multi_request" not in SLOTS:
        reasons.append(
            f"{SLOTS_ENV}: slot 'draft_multi' serves the draft steps of two or more "
            "requests ('multi_request')"
        )
    if "draft_embed" in SLOTS and "draft_multi" not in SLOTS:
        reasons.append(
            f"{SLOTS_ENV}: slot 'draft_embed' runs in slot 'draft_multi''s steps"
        )
    if "draft_skv2" in SLOTS and "draft_extend" not in SLOTS:
        reasons.append(
            f"{SLOTS_ENV}: slot 'draft_skv2' runs inside entry 'draft_extend'"
        )
    for name, base in (
        ("skv2_cp", "skv2"),
        ("draft_q1_cp", "draft_q1"),
        ("draft_skv2_cp", "draft_skv2"),
    ):
        if name in SLOTS and base not in SLOTS:
            reasons.append(
                f"{SLOTS_ENV}: slot {name!r} takes slot {base!r}'s long-KV graph "
                "variant's place from its length up"
            )
    if "hop_fp8" in SLOTS and "hop" not in SLOTS:
        reasons.append(f"{SLOTS_ENV}: slot 'hop_fp8' is an output of slot 'hop'")
    if "lowm_push" in SLOTS and not {"lowm_out", "hop"} <= SLOTS:
        reasons.append(
            f"{SLOTS_ENV}: slot 'lowm_push' pushes slot 'lowm_out''s rows into slot "
            "'hop''s mailbox"
        )
    if "mka_push" in SLOTS and not {"mka", "hop"} <= SLOTS:
        reasons.append(
            f"{SLOTS_ENV}: slot 'mka_push' pushes slot 'mka''s epilogue into slot "
            "'hop''s mailbox"
        )
    if (
        "route_lin" in SLOTS
        and not {
            "route",
            "mkm",
            "hop",
            "gdn_push",
            "mka",
            "mka_push",
        }
        <= SLOTS
    ):
        reasons.append(
            f"{SLOTS_ENV}: slot 'route_lin' runs the router before the all-reduce "
            "through slots 'route', 'mkm', 'hop', 'gdn_push', 'mka' and 'mka_push'"
        )
    if "route_lin" in SLOTS and "route_shared" in SLOTS and "lowm_se" not in SLOTS:
        reasons.append(
            f"{SLOTS_ENV}: slot 'route_lin' needs 'lowm_se' to replace "
            "'route_shared' at one request"
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
    "srt/runtime_context.py": (
        "get_parallel",
        '"tp_size"',
        '"attn_tp_size"',
        '"moe_ep_size"',
    ),
    "srt/kernel_plugins/nvidia/eagle_binding.py|srt/megakernels/eagle_binding.py": (
        "page_size",
        "tp_rank",
        "tp_size",
        "tp_device_group",
    ),
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/runtime.py": (
        "megakernel_agree",
    ),
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
    from sglang.srt.runtime_context import get_parallel

    from sgl_nvidia_mega.qwen3_5 import attention, boundary, gdn, lowm, moe
    from sgl_nvidia_mega.qwen3_5 import plan as slot_plan
    from sgl_nvidia_mega.qwen3_5.runtime import hooks, parity
    from sgl_nvidia_mega.qwen3_5.runtime.executor import Qwen35Verify

    parallel = get_parallel()
    dropped = apply_expert_parallel(parallel.moe_ep_size)
    if dropped:
        logger.info(
            "NVIDIA megakernel: Qwen3.5 under expert parallelism (moe_ep_size %d): "
            "slots %s run SGLang's EP path",
            parallel.moe_ep_size,
            sorted(dropped),
        )
    apply_cp_lite_topology(parallel.attn_tp_size)
    build_cp_lite(ctx.binding)
    multi = slot_plan.MULTI_REQUEST in SLOTS
    if multi:
        multi = slot_plan.multi_request(parallel.tp_size, SLOTS)
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
        "lowm": lowm.build(model, pctx),
    }
    parallel = get_parallel()
    tep8_moe = bound["moe"].pre and parallel.tp_size == 8 and parallel.moe_ep_size > 1
    facts = slot_plan.Facts(
        route=bound["moe"].route,
        route_topk=bound["moe"].topk,
        route_gemm=bound["moe"].gemm,
        tep8_moe=tep8_moe,
        mkm=bool(bound["moe"].mkm),
        mkm_wide=bound["moe"].mkm_wide,
        mkm_m32=bool(bound["moe"].m32) and "mkm_m32" in SLOTS,
        mkm_m64=bool(bound["moe"].m64),
        mkm_m128=bool(bound["moe"].m128),
        mkm_m256=bool(bound["moe"].m256),
        mkm_m512=bool(bound["moe"].m512),
        gdn_verify=bound["gdn"].gdn_verify,
        gdn_pdl=bound["gdn"].gdn_pdl,
        gdn_core=bound["gdn"].gdn_core,
        gdn_conv=bound["gdn"].gdn_conv,
        hop=bound["boundary"].hop is not None,
        hop_mid=bound["boundary"].mid,
        gdn_push=bool(bound["gdn"].packed),
        gdn_front=bool(bound["gdn"].front),
        gdn_trt_front=bound["gdn"].gdn_trt_front,
        route_shared=bound["moe"].shared,
        skv2=bound["attention"].skv2,
        mka=bound["attention"].mka,
        mka_wide=bound["attention"].mka_wide,
        mka_push=bound["attention"].mka_push,
        route_lin=bool(bound["gdn"].rlin)
        and bound["attention"].route_lin
        and bound["boundary"].hop is not None
        and bound["boundary"].hop.rl is not None,
        lowm_in=bound["lowm"].lowm_in,
        lowm_out=bound["lowm"].lowm_out,
        lowm_se=bound["lowm"].se,
        lowm_se_pre=bound["lowm"].se_pre,
        lowm_push=bound["lowm"].push,
        hop_fp8=bound["lowm"].hop_fp8,
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
    target_slots = (
        SLOTS
        - set(DRAFT_ENTRIES)
        - set(DRAFT_SLOTS)
        - set(OPT_IN_COVERAGE)
        - set(OPT_IN_VARIANTS)
        - set(HOOK_SLOTS)
    )
    # mkm's 16-token engine has nothing to serve where no covered step is wider than 7 rows
    vacuous = (
        {"mkm_wide"}
        if slot_plan.widest(pctx.widths, slot_plan.MKM_WIDE_MAX_ROWS)
        <= slot_plan.MKM_MAX_ROWS
        else set()
    )
    if (
        slot_plan.widest(pctx.widths, slot_plan.MKM_M32_MAX_ROWS)
        <= slot_plan.MKM_WIDE_MAX_ROWS
    ):
        vacuous.add("mkm_m32")
    if (
        slot_plan.widest(pctx.widths, slot_plan.MKM_M64_MAX_ROWS)
        <= slot_plan.MKM_M32_MAX_ROWS
    ):
        vacuous.add("mkm_m64")
    if (
        slot_plan.widest(pctx.widths, slot_plan.MKM_M128_MAX_ROWS)
        <= slot_plan.MKM_M64_MAX_ROWS
    ):
        vacuous.add("mkm_m128")
    if (
        slot_plan.widest(pctx.widths, slot_plan.MKM_M256_MAX_ROWS)
        <= slot_plan.MKM_M128_MAX_ROWS
    ):
        vacuous.add("mkm_m256")
    if (
        slot_plan.widest(pctx.widths, slot_plan.MKM_M512_MAX_ROWS)
        <= slot_plan.MKM_M256_MAX_ROWS
    ):
        vacuous.add("mkm_m512")
    # hop_mid has nothing to serve when no covered step is past the shipped bound.
    if max(pctx.widths) <= slot_plan.HOP_MAX_ROWS:
        vacuous.add("hop_mid")
    # route_topk has nothing to serve where slot route covers every width
    if "route" in SLOTS and max(pctx.widths) <= slot_plan.ROUTE_MAX_ROWS:
        vacuous.update({"route_topk", "route_gemm"})
    missing = sorted(name for name in target_slots - vacuous if not built[name])
    if missing:
        raise RuntimeError(
            f"NVIDIA megakernel: requested slots {missing} built no width (MoE: "
            f"{bound['moe'].reason or 'built'}; GDN: {bound['gdn'].reason or 'built'}; "
            f"boundary: {bound['boundary'].reason or 'built'}; "
            f"attention: {bound['attention'].reason or 'built'}; "
            f"low-M: {bound['lowm'].reason or 'built'})"
        )
    checker = (
        parity.Parity(covered, torch.cuda.current_device(), model)
        if parity.ENABLED
        else None
    )
    hook_factory = (
        hooks.build(model, ctx.binding) if "sharded_gumbel" in SLOTS else None
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
        speculative_runtime_hooks=hook_factory,
    )
