# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The megakernel runtime: the executor of a covered target-verify step (``executor``),
its per-step state (``state``), the embedding (``embedding``), the test-only parity
check (``parity``), and ``build``, which SGLang runs through the spec once the model is
prepared and before the KV cache is sized."""

from __future__ import annotations

import logging
import os
from typing import Optional

import msgspec

from sgl_nvidia_mega.qwen3_5 import topology

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
    "hop",
    "hop_fp8",
    "multi_request",
    "draft_loop",
    "draft_extend",
    "draft_multi",
    "draft_embed",
)
OPT_IN_SLOTS: tuple[str, ...] = (
    # The fused FP8 norm can select a different reduction layout from BF16.
    # Keep its verify chain and dependent variants explicitly requested.
    "gdn_verify",
    "gdn_pdl",
    "gdn_conv",
    "gdn_prefill_split_conv",
    "gdn_prefill_front",
    "gdn_prefill_track_snapshot",
    "moe_prefill_input_nvfp4",
    "moe_prefill_add_norm",
    "moe_prefill_noclone",
    "prefill_gate_fp8",
    "prefill_act_fp8",
    "gdn_prefill_norm_fp8",
    "prefill_norm_fp8",
    # the TP1/PP2 prefill's Q / K norm + RoPE over row tiles, gate in place (numeric)
    "attn_prefill_qk_rows",
    "route",
    "route_gemm",
    "mkm",
    "mkm_wide",
    "gdn_core",
    "gdn_conv_core",
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
    "bt_tail",
    "gemm_ar",
    "fc2ar",
    "lowm_ba",
    "gdn_core_tc",
    "gdn_commit_fold",
)
# Coverage names requested in the same variable have no kernel build result of their
# own. Multi-request coverage is shipped because every covered width keeps its own
# shape instead of padding.
OPT_IN_COVERAGE: tuple[str, ...] = ("multi_request",)
# Graph or build variants of a slot, requested in the same variable: no width of their
# own. The TP8 plan enables BAL for its wide engines; elsewhere it stays explicit.
# skv2_cp: one long request past the measured crossover runs CP-lite in place of SKV2,
# in a graph variant of its own (attention/route.py)
OPT_IN_VARIANTS: tuple[str, ...] = (
    "skv2_cp",
    "mkm_wide_bal",
    "bt_tail_early",
    "bt_tail_early_fin",
    "cp_smv2",
    "cp_ptxas_o1",
    "cp_merge_plain_wait",
    "cp_q_push",
    # with cp_q_push: the pushing q writer releases its dependents at entry
    "cp_qw_entry",
    "cp_merge_cg",
    "moe_engine",
    "moe_ring",
    "route_big_gemm",
    "route_lean",
    "moe_engine_route",
    "route_big_mid",
    "lowm_se_tep8",
    "lowm_se_wide",
    "lowm_se_coresident",
    "cp_lam_q",
    "cp_q_gather",
    "cp_o_push",
)
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
    "draft_select",
    "draft_embed_local",
    "draft_fc_input",
    "draft_head_lowm",
    "draft_router_lowm",
    "draft_fc_tp",
    "draft_q_push",
    "draft_q_gather",
    "draft_gate_fold",
    "draft_o_push",
)
CP_LITE_SLOTS = frozenset({"skv2_cp", "draft_q1_cp", "draft_skv2_cp"})
# Build options of every CP slot's partial (apply_cp_lite_build)
CP_LITE_BUILD_VARIANTS = frozenset({"cp_smv2", "cp_ptxas_o1"})
CP_LITE_VARIANTS = CP_LITE_BUILD_VARIANTS | {
    # not a kernel variant: the draft attention provider runs on the CP slots' CP-lite
    "draft_cp",
    # draft_cp's precise call's merge variants
    "draft_gate_fold",
    "draft_o_push",
    "cp_merge_plain_wait",
    "cp_q_push",
    "cp_qw_entry",
    "cp_merge_cg",
    "cp_lam_q",
    "cp_q_gather",
    "cp_o_push",
}
# Variants that act on slot cp_q_push's pushed CP-lite call
CP_PUSHED_CALL_SLOTS = frozenset(
    {"cp_qw_entry", "cp_merge_cg", "cp_lam_q", "cp_o_push"}
)
# Provider callbacks consume these names from the shared slot policy.  They do not
# build a target-verify width of their own.
HOOK_SLOTS: tuple[str, ...] = (
    "sharded_gumbel",
    "gathered_cdf",
    "gumbel_fast",
    "verify_head",
    # verify_head also leaves gumbel_fast's T = 1 and greedy keys of its logits
    "verify_head_keys",
    "draft_cp",
)
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
MEASURED_OPT_IN: tuple[str, ...] = (
    "route",
    "mkm",
    "mkm_wide",
    "route_shared",
    "gdn_verify",
    "gdn_pdl",
    "gdn_conv",
    "gdn_core",
    "gdn_conv_core",
    "gdn_push",
    "gdn_front",
    "gdn_trt_front",
    "hop_mid",
    "skv2",
    "skv2_cp",
    "mka",
    "mka_push",
    "lowm_in",
    "lowm_out",
    "lowm_se",
    "lowm_push",
    "lowm_ba",
    "sharded_gumbel",
    "draft_moe",
    "draft_q1",
    "draft_hop",
    "draft_fin",
    "draft_front",
    "draft_skv2",
    "draft_q1_cp",
    "draft_skv2_cp",
)
MEASURED_TEP8: tuple[str, ...] = (
    "moe_engine",
    "moe_ring",
    "route_big_gemm",
    "route_lean",
    "moe_engine_route",
    "route_big_mid",
    "lowm_se_tep8",
    "lowm_se_wide",
    "lowm_se_coresident",
)
MEASURED_BT_TAIL: tuple[str, ...] = ("bt_tail", "bt_tail_early", "bt_tail_early_fin")
MEASURED_BT_AR: tuple[str, ...] = ("gemm_ar", "fc2ar")
MEASURED_CP_LITE: tuple[str, ...] = (
    "cp_smv2",
    "cp_ptxas_o1",
    "cp_merge_plain_wait",
    "cp_q_push",
    "cp_qw_entry",
    "cp_merge_cg",
    "cp_lam_q",
    "cp_q_gather",
    "cp_o_push",
)
MEASURED_OPTIONS: tuple[str, ...] = (
    "gdn_value_split",
    "gemm_tactics",
    "gdn_commit",
    "embed_local",
    "draft_top1_cfg",
    "draft_oproj_push",
    "draft_early_hops",
    "draft_head_keys",
    "draft_merge_early",
    "draft_hilo_trig",
)
MEASURED_TEP8_OPTIONS: tuple[str, ...] = (
    "draft_moe_engine",
    "draft_moe_wide",
    "draft_shared_lowm",
    "draft_fc_push",
)
OPTION_ROWS = frozenset(MEASURED_OPTIONS + MEASURED_TEP8_OPTIONS)
MEASURED_DRAFT: tuple[str, ...] = (
    "draft_cp",
    "draft_select",
    "draft_embed_local",
    "draft_fc_input",
    "draft_head_lowm",
    "draft_router_lowm",
    "draft_q_push",
    "draft_q_gather",
    "draft_gate_fold",
    "draft_o_push",
)
MEASURED_DRAFT_TEP8: tuple[str, ...] = ("draft_fc_tp",)
MEASURED_HOOKS: tuple[str, ...] = ("gumbel_fast", "verify_head", "verify_head_keys")
MEASURED_DEP4: tuple[str, ...] = (
    "gdn_verify",
    "gdn_core",
    "gdn_core_tc",
    "gdn_commit_fold",
    "gathered_cdf",
)
PREFILL_SLOTS: tuple[str, ...] = (
    "gdn_prefill_split_conv",
    "gdn_prefill_front",
    "gdn_prefill_track_snapshot",
    "moe_prefill_input_nvfp4",
    "moe_prefill_add_norm",
    "moe_prefill_noclone",
    "prefill_gate_fp8",
    "prefill_act_fp8",
    "gdn_prefill_norm_fp8",
    "prefill_norm_fp8",
    "attn_prefill_qk_rows",
)
MEASURED_PREFILL: tuple[str, ...] = (
    "gdn_prefill_front",
    "gdn_prefill_track_snapshot",
    "moe_prefill_input_nvfp4",
    "moe_prefill_add_norm",
    "moe_prefill_noclone",
    "prefill_gate_fp8",
    "prefill_act_fp8",
    "gdn_prefill_norm_fp8",
    "prefill_norm_fp8",
    "attn_prefill_qk_rows",
)
MEASURED_PLAN: dict[tuple[str, topology.Sku], frozenset] = {
    ("tp8", topology.GB300): frozenset(
        SHIPPED_SLOTS
        + MEASURED_OPT_IN
        + MEASURED_BT_TAIL
        + MEASURED_CP_LITE
        + MEASURED_OPTIONS
        + MEASURED_DRAFT
        + MEASURED_HOOKS
    ),
    ("tep8", topology.GB300): frozenset(
        SHIPPED_SLOTS
        + MEASURED_OPT_IN
        + MEASURED_BT_TAIL
        + MEASURED_BT_AR
        + MEASURED_CP_LITE
        + MEASURED_OPTIONS
        + MEASURED_TEP8
        + MEASURED_TEP8_OPTIONS
        + MEASURED_DRAFT_TEP8
        + MEASURED_DRAFT
        + MEASURED_HOOKS
    ),
    ("dep4", topology.GB300): frozenset(SHIPPED_SLOTS + MEASURED_DEP4),
    ("p_tp1_pp2", topology.GB300): frozenset(SHIPPED_SLOTS + MEASURED_PREFILL),
}
MEASURED_BITWISE: tuple[str, ...] = (
    "draft_skv2",
    "mkm",
    "mkm_wide",
    "mkm_wide_bal",
    "gdn_pdl",
    "gdn_conv",
    "gdn_conv_core",
    "mka_push",
    "moe_engine",
    "moe_ring",
    "route_lean",
    "moe_engine_route",
    "lowm_se_coresident",
    "bt_tail",
    "bt_tail_early",
    "bt_tail_early_fin",
    "gemm_ar",
    "fc2ar",
    "cp_smv2",
    "cp_ptxas_o1",
    "cp_merge_plain_wait",
    "cp_q_push",
    "cp_qw_entry",
    "cp_merge_cg",
    "cp_lam_q",
    "cp_q_gather",
    "cp_o_push",
    "gdn_value_split",
    "gemm_tactics",
    "draft_select",
    "embed_local",
    "draft_embed_local",
    "draft_top1_cfg",
    "draft_early_hops",
    "draft_q_push",
    "draft_q_gather",
    "draft_gate_fold",
    "draft_o_push",
    "draft_merge_early",
    "draft_hilo_trig",
    "gdn_prefill_front",
    "gdn_prefill_track_snapshot",
    "moe_prefill_input_nvfp4",
    "moe_prefill_add_norm",
    "moe_prefill_noclone",
    "prefill_gate_fp8",
    "prefill_act_fp8",
    "gdn_prefill_norm_fp8",
    "prefill_norm_fp8",
)
NUMERICS_CHANGE: tuple[str, ...] = (
    "route",
    "route_big_gemm",
    "route_big_mid",
    "route_shared",
    "route_lin",
    "gdn_verify",
    "gdn_core",
    "gdn_push",
    "gdn_front",
    "gdn_trt_front",
    "hop_mid",
    "skv2",
    "skv2_cp",
    "mka",
    "lowm_in",
    "lowm_out",
    "lowm_se",
    "lowm_se_tep8",
    "lowm_se_wide",
    "lowm_push",
    "lowm_ba",
    "sharded_gumbel",
    "draft_moe",
    "draft_q1",
    "draft_hop",
    "draft_fin",
    "draft_front",
    "draft_q1_cp",
    "draft_skv2_cp",
    "draft_fc_input",
    "draft_head_lowm",
    "draft_router_lowm",
    "draft_cp",
    "draft_moe_engine",
    "draft_moe_wide",
    "draft_shared_lowm",
    "draft_fc_tp",
    "draft_fc_push",
    "draft_oproj_push",
    "draft_head_keys",
    "gumbel_fast",
    "verify_head",
    "verify_head_keys",
    "gdn_commit",
    "gdn_core_tc",
    "gdn_commit_fold",
    "gathered_cdf",
    "attn_prefill_qk_rows",
)
NUMERICS_UNBOUNDED: tuple[str, ...] = ("hop_mid",)
MEASURED_KEY: Optional[tuple[str, topology.Sku]] = None


def _measured_stack_on() -> bool:
    if os.environ.get(SLOTS_ENV) is not None:
        return False
    from sglang.srt.environ import envs

    return envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get()


def _measured_rows_on(key, row: str) -> bool:
    return row in MEASURED_PLAN.get(key, ()) and _measured_stack_on()


def _cp_lite_constants_on() -> bool:
    return MEASURED_KEY is not None and any(
        _measured_rows_on(MEASURED_KEY, row) for row in SLOTS & CP_LITE_SLOTS
    )


def _measured_rows() -> frozenset:
    return frozenset(
        row
        for key, rows in MEASURED_PLAN.items()
        for row in rows
        if row not in OPTION_ROWS and _measured_rows_on(key, row)
    )


def measured_options() -> frozenset:
    return frozenset(
        row
        for key, rows in MEASURED_PLAN.items()
        for row in rows
        if row in MEASURED_OPTIONS and _measured_rows_on(key, row)
    )


def measured_option(name: str) -> bool:
    return MEASURED_KEY is not None and _measured_rows_on(MEASURED_KEY, name)


def _requested_slots() -> frozenset:
    """``SLOTS_ENV`` (comma-separated; "" runs the stock step), else every shipped
    slot."""
    raw = os.environ.get(SLOTS_ENV)
    if raw is not None:
        return frozenset(name.strip() for name in raw.split(",")) - {""}
    return frozenset(SHIPPED_SLOTS) | _measured_rows()


SLOTS = _requested_slots()


def slot_failure(slots, reason: str) -> str:
    """A requested slot's failed capability, including the recovery action."""
    names = [slots] if isinstance(slots, str) else sorted(slots)
    label = f"slot {names[0]}" if len(names) == 1 else f"slots {names}"
    return (
        f"NVIDIA megakernel: {label} cannot run: {reason}; "
        f"remove it from {SLOTS_ENV} or change the topology"
    )


class SlotUnavailable(RuntimeError):
    """A build capability refusal, distinct from CUDA errors and kernel failures."""

    def __init__(self, slots, reason: str):
        self.slots, self.reason = slots, reason
        super().__init__(slot_failure(slots, reason))


def require_multicast(slots, pointer: int, group, device) -> None:
    """All ranks reject an unavailable mapping before any consumer starts."""
    import torch
    import torch.distributed as dist

    valid = torch.tensor(
        [bool(pointer) and pointer % 16 == 0], dtype=torch.int32, device=device
    )
    dist.all_reduce(valid, op=dist.ReduceOp.MIN, group=group)
    if not int(valid.item()):
        raise SlotUnavailable(
            slots, "needs an aligned NVLink multicast mapping on every rank"
        )


def unavailable_slots(slots, reason: str, *, config: bool = False) -> None:
    """Log a refused slot; the caller keeps its stock implementation / unserved entry.

    A refusal that depends on the weights, the model's shapes or the environment
    never stops the server, even for an explicit SLOTS list. Only a ``config``
    contradiction (a server argument the slot cannot serve) raises, and only for an
    explicitly requested slot.
    """
    names = [slots] if isinstance(slots, str) else sorted(slots)
    if config and names and os.environ.get(SLOTS_ENV) is not None:
        raise RuntimeError(slot_failure(slots, reason))
    logger.info("NVIDIA megakernel: requested slots %s use stock (%s)", names, reason)


def agreed_decline(slot: str, reason: str, *, cpu_group=None) -> str:
    if cpu_group is not None:
        import torch.distributed as dist

        if dist.is_initialized():
            reasons = [None] * dist.get_world_size(cpu_group)
            dist.all_gather_object(reasons, reason, group=cpu_group)
            held: dict = {}
            for rank, why in enumerate(reasons):
                if why:
                    held.setdefault(why, []).append(rank)
            reason = "; ".join(f"{why} (ranks {ranks})" for why, ranks in held.items())
    if reason:
        logger.warning(
            "NVIDIA megakernel: slot %s uses stock on every rank: %s", slot, reason
        )
    return reason


def slot_supported(slots, reason: str, *, cpu_group=None) -> bool:
    """Agree on rank-local shard checks before an exchange's collective allocation."""
    if cpu_group is not None:
        import torch.distributed as dist

        if dist.is_initialized():
            reasons = [None] * dist.get_world_size(cpu_group)
            dist.all_gather_object(reasons, reason, group=cpu_group)
            reason = "; ".join(dict.fromkeys(r for r in reasons if r))
    if not reason:
        return True
    unavailable_slots(slots, reason)
    return False


def _measured_plan_requested() -> bool:
    return bool(_measured_rows() - frozenset(SHIPPED_SLOTS)) or any(
        _measured_rows_on(key, row)
        for key, rows in MEASURED_PLAN.items()
        for row in rows & OPTION_ROWS
    )


def apply_measured_plan(parallel, sku: Optional[topology.Sku] = None) -> frozenset:
    global SLOTS, MEASURED_KEY
    if MEASURED_KEY is not None or not _measured_plan_requested():
        return frozenset()
    measured = _measured_rows()
    try:
        from sglang.srt.kernel_plugins.nvidia.binding import megakernel_agree
    except ImportError:  # pragma: no cover - the pre-kernel_plugins SGLang path
        from sglang.srt.megakernels.runtime import megakernel_agree

    sku = topology.device_sku() if sku is None else sku
    # ranks on different GPUs would build different slots
    megakernel_agree(
        msgspec.structs.astuple(sku), what="Qwen3.5 GPU (compute capability, SMs)"
    )
    key = MEASURED_KEY = (topology.plan_key_name(parallel), sku)
    rows_on = frozenset(
        row for row in MEASURED_PLAN.get(key, ()) if _measured_rows_on(key, row)
    )
    keep = frozenset(SHIPPED_SLOTS).union(rows_on - OPTION_ROWS)
    dropped = SLOTS & (measured - keep)
    SLOTS = SLOTS - dropped
    if key in MEASURED_PLAN:
        current = topology.from_parallel(parallel)
        if key[0] == current.name:
            name, planned = current.name, topology.plan(keep, current)
        else:
            # a PD prefill key: the build runs the prefill slots declared there
            name, planned = key[0], keep & set(declared_prefill_slots(parallel))
        logger.info(
            "NVIDIA megakernel: Qwen3.5 measured plan at %s on %s: slots %s; "
            "options %s; numeric with the %s plan: %s",
            *key,
            sorted(keep),
            sorted(rows_on & OPTION_ROWS),
            name,
            sorted((planned | (rows_on & OPTION_ROWS)) & set(NUMERICS_CHANGE)),
        )
    else:
        logger.info(
            "NVIDIA megakernel: Qwen3.5 has no measured plan at %s on %s: "
            "the shipped slots run",
            *key,
        )
    return dropped


def follow_measured_plan() -> frozenset:
    global SLOTS
    if MEASURED_KEY is not None or not _measured_plan_requested():
        return frozenset()
    dropped = SLOTS & (_measured_rows() - frozenset(SHIPPED_SLOTS))
    SLOTS = SLOTS - dropped
    if dropped:
        logger.info(
            "NVIDIA megakernel: Qwen3.5 draft: the target resolved no measured plan; "
            "the shipped draft slots run"
        )
    return dropped


def apply_cp_lite_topology(attention_tp_size: int) -> frozenset:
    """Drop CP-lite slots when the attention-TP Q/KV topology cannot serve them."""
    from sgl_nvidia_mega.qwen3_5 import cp_lite_topology

    global SLOTS
    if cp_lite_topology(attention_tp_size) is not None:
        return frozenset()
    dropped = SLOTS & (CP_LITE_SLOTS | CP_LITE_VARIANTS)
    SLOTS = SLOTS - CP_LITE_SLOTS - CP_LITE_VARIANTS
    if dropped:
        logger.info(
            "NVIDIA megakernel: Qwen3.5 attention-TP %d: CP slots %s use stock "
            "(unsupported Q/KV-head topology)",
            attention_tp_size,
            sorted(dropped),
        )
    return dropped


def build_recurrent_commit():
    if not measured_option("gdn_commit"):
        return None
    import torch

    from sgl_nvidia_mega.qwen3_5.gdn.commit import RecurrentCommit

    return RecurrentCommit(torch.cuda.current_device())


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
        unavailable_slots(
            requested,
            f"CP slots need page size 64 (got {binding.page_size})",
            config=True,
        )
        return

    import torch

    from sgl_nvidia_mega.qwen3_5.sm103.attention.q35_cp_lite import CpLite

    CpLite.lam_q = "cp_lam_q" in SLOTS
    cp = CpLite.maybe_create(binding, torch.device("cuda", torch.cuda.current_device()))
    if cp is None:
        unavailable_slots(
            requested, "CP slots need a supported attention-TP Q/KV-head topology"
        )
        return


def apply_cp_lite_build(deadline_s: float | None = None) -> None:
    if not SLOTS & CP_LITE_SLOTS:
        return
    from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_fused, q35_cp_lite
    from sgl_nvidia_mega.qwen3_5.sm103.attention.q35_cp_lite import CpLite

    slots = SLOTS if CpLite._instance is not None else frozenset()
    if slots:
        CpLite._instance.merge_cg = "cp_merge_cg" in slots
        # the region exists: build_cp_lite saw every slot that survives the filters
        CpLite._instance.lam_q = "cp_lam_q" in slots
        CpLite._instance.o_push = "cp_o_push" in slots
        if deadline_s:
            q35_cp_lite.set_wait_deadline(deadline_s)
    q35_cp_fused.set_partial_build(
        softmax_v2=2 if "cp_smv2" in slots else 0,
        ptxas="-O1" if "cp_ptxas_o1" in slots else "",
    )
    q35_cp_fused.set_cp_phase(bool(slots) and _cp_lite_constants_on())


def apply_expert_parallel(ep_size: int) -> frozenset:
    """Drop ``EP_STOCK_SLOTS`` from ``SLOTS`` when the MoE runs expert-parallel
    (``ep_size`` > 1); returns the dropped names. Every process calls it before reading
    ``SLOTS`` (config check, target build, draft build)."""
    global SLOTS
    if ep_size <= 1:
        return frozenset()
    dropped = SLOTS & EP_STOCK_SLOTS
    SLOTS = SLOTS - EP_STOCK_SLOTS
    if dropped:
        logger.info(
            "NVIDIA megakernel: slots %s use stock under expert parallelism "
            "(moe_ep_size %d)",
            sorted(dropped),
            ep_size,
        )
    return dropped


def declared_prefill_slots(parallel) -> list:
    key = topology.plan_key_name(parallel)
    return [
        name
        for name in PREFILL_SLOTS
        if key in topology.SLOT_TOPOLOGIES.get(name, frozenset())
    ]


def apply_topology(parallel) -> frozenset:
    """Keep only each slot's declared capabilities, plus the slots the topology's plan
    adds (``topology.plan``), before package builds run; returns the dropped names."""
    global SLOTS
    current = topology.from_parallel(parallel)
    selected = topology.plan(SLOTS, current)
    dropped, added = SLOTS - selected, selected - SLOTS
    SLOTS = selected
    if added:
        logger.info(
            "NVIDIA megakernel: Qwen3.5 %s plan: slots %s run with the requested ones",
            current.name,
            sorted(added),
        )
    if dropped:
        logger.info(
            "NVIDIA megakernel: slots %s use stock (topology %s)",
            sorted(dropped),
            current.name,
        )
    return dropped


def apply_commit_fold(
    offered: tuple[int, ...], candidates: tuple[int, ...]
) -> frozenset:
    global SLOTS
    if "gdn_commit_fold" not in SLOTS:
        return frozenset()
    reason, config = _commit_fold_reason(offered, candidates)
    if not reason:
        return frozenset()
    SLOTS = SLOTS - {"gdn_commit_fold"}
    unavailable_slots("gdn_commit_fold", reason, config=config)
    return frozenset({"gdn_commit_fold"})


def _commit_fold_reason(offered, candidates) -> tuple[str, bool]:
    from sgl_nvidia_mega.qwen3_5 import plan as slot_plan
    from sgl_nvidia_mega.qwen3_5.runtime import parity

    if not _offers_recurrent_commit():
        return "this SGLang has no SupportsRecurrentCommit", False
    if parity.ENABLED:
        return "the parity check runs SGLang's verify on the folded ring", False
    try:
        from sglang.srt.runtime_context import get_memory

        memory = get_memory()
        radix = not memory.disable_radix_cache
        unified = bool(memory.enable_unified_memory)
    except (ImportError, AttributeError, ValueError) as e:
        return f"the memory configuration is unavailable: {e!r}", False
    if radix:
        return "the radix cache is on (fold only with --disable-radix-cache)", True
    if unified:
        return "unified memory indexes the ring by request sub-pool unit", True
    uncovered = [
        w for w in offered if w not in candidates or w > slot_plan.GDN_CORE_MAX_ROWS
    ]
    if uncovered:
        return f"target verifies of {uncovered} rows run no gdn_core launch", True
    return "", False


def _offers_recurrent_commit() -> bool:
    """Whether this SGLang commits a verify through a build's commit_recurrent_state."""
    try:
        from sglang.kernels.plugin_kernels.nvidia.spec import (  # noqa: F401
            SupportsRecurrentCommit,
        )
    except ImportError:
        return False
    return True


def slot_reasons(requested=None) -> list[str]:
    """Why the requested slots cannot be served ([] when they can)."""
    slots = SLOTS if requested is None else requested
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
    if "mkm" in slots and "route" not in slots:
        reasons.append(f"{SLOTS_ENV}: slot 'mkm' runs on slot 'route''s outputs")
    if "gdn_prefill_track_snapshot" in slots and "gdn_prefill_front" not in slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'gdn_prefill_track_snapshot' runs in slot "
            "'gdn_prefill_front''s extend"
        )
    if "mkm_wide" in slots and "mkm" not in slots:
        reasons.append(f"{SLOTS_ENV}: slot 'mkm_wide' is slot 'mkm''s 16-token engine")
    if "mkm_wide" in slots and "multi_request" not in slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'mkm_wide' serves the steps of more than one request "
            "('multi_request')"
        )
    if "mkm_m32" in slots and "route_topk" not in slots:
        reasons.append(f"{SLOTS_ENV}: slot 'mkm_m32' takes slot 'route_topk''s rows")
    if "mkm_m32" in slots and "multi_request" not in slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'mkm_m32' serves the steps of more than two requests "
            "('multi_request')"
        )
    if "mkm_m64" in slots and "route_topk" not in slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'mkm_m64' runs on slot 'route_topk''s routing"
        )
    if "mkm_m64" in slots and "multi_request" not in slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'mkm_m64' serves the steps of more than four requests "
            "('multi_request')"
        )
    if "mkm_m128" in slots and "route_topk" not in slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'mkm_m128' runs on slot 'route_topk''s routing"
        )
    if "mkm_m128" in slots and "multi_request" not in slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'mkm_m128' serves the steps of more than nine requests "
            "('multi_request')"
        )
    for name, more in (("mkm_m256", "18"), ("mkm_m512", "36")):
        if name in slots and "route_topk" not in slots:
            reasons.append(
                f"{SLOTS_ENV}: slot '{name}' runs on slot 'route_topk''s routing"
            )
        if name in slots and "multi_request" not in slots:
            reasons.append(
                f"{SLOTS_ENV}: slot '{name}' serves the steps of more than {more} requests "
                "('multi_request')"
            )
    if (
        "mkm_wide_bal" in slots
        and not {
            "mkm_m256",
            "mkm_m512",
        }
        & slots
    ):
        reasons.append(
            f"{SLOTS_ENV}: slot 'mkm_wide_bal' changes the work split of slot "
            "'mkm_m256' or 'mkm_m512'"
        )
    if "route_shared" in slots and "route" not in slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'route_shared' runs in slot 'route''s launch"
        )
    if "route_gemm" in slots and "route_topk" not in slots:
        reasons.append(f"{SLOTS_ENV}: slot 'route_gemm' runs inside slot 'route_topk'")
    # TEP8's routed body runs on route's and route_topk's builds
    for name in ("moe_engine", "route_big_gemm", "route_lean", "route_big_mid"):
        if name in slots and not {"route", "route_topk"} <= slots:
            reasons.append(
                f"{SLOTS_ENV}: slot {name!r} serves TEP8's routed body on slot "
                "'route_topk''s routing (with slot 'route')"
            )
    if "moe_ring" in slots and not {"moe_engine", "route_lean"} <= slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'moe_ring' fills slot 'moe_engine''s ring while slot "
            "'route_lean''s route-big runs"
        )
    if "lowm_se_tep8" in slots and not {"lowm_se", "route", "route_topk"} <= slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'lowm_se_tep8' widens slot 'lowm_se' at TEP8's routed-body "
            "rows (with slots 'route' and 'route_topk')"
        )
    for name in ("lowm_se_wide", "lowm_se_coresident"):
        if name in slots and "lowm_se_tep8" not in slots:
            reasons.append(
                f"{SLOTS_ENV}: slot {name!r} changes slot 'lowm_se_tep8''s launches"
            )
    if "moe_engine_route" in slots and not {"moe_engine", "route_big_gemm"} <= slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'moe_engine_route' routes in slot 'moe_engine' from "
            "slot 'route_big_gemm''s logits"
        )
    if "hop_mid" in slots and "hop" not in slots:
        reasons.append(f"{SLOTS_ENV}: slot 'hop_mid' widens slot 'hop'")
    if "gdn_push" in slots and not {"gdn_verify", "hop"} <= slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'gdn_push' pushes slot 'gdn_verify''s out_proj into "
            "slot 'hop'"
        )
    if "gdn_front" in slots and not {"gdn_verify", "gdn_pdl"} <= slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'gdn_front' runs slot 'gdn_pdl''s dependent verify"
        )
    if "gdn_trt_front" in slots and not {"gdn_front", "gdn_core"} <= slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'gdn_trt_front' replaces slot 'gdn_front' in front "
            "of slot 'gdn_core'"
        )
    if "gdn_core" in slots and "gdn_verify" not in slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'gdn_core' runs slot 'gdn_verify''s step in one launch"
        )
    if "gdn_core_tc" in slots and "gdn_core" not in slots:
        reasons.append(f"{SLOTS_ENV}: slot 'gdn_core_tc' runs slot 'gdn_core''s launch")
    if "gdn_commit_fold" in slots and "gdn_core_tc" not in slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'gdn_commit_fold' folds the commit into slot "
            "'gdn_core_tc''s launch"
        )
    if "gdn_conv" in slots and "gdn_verify" not in slots:
        reasons.append(f"{SLOTS_ENV}: slot 'gdn_conv' runs slot 'gdn_verify''s conv")
    if "gdn_conv_core" in slots and not {"gdn_core", "gdn_conv"} <= slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'gdn_conv_core' runs slot 'gdn_conv''s conv in slot "
            "'gdn_core''s launch"
        )
    if "lowm_ba" in slots and "gdn_verify" not in slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'lowm_ba' runs slot 'gdn_verify''s in_proj_ba"
        )
    if "gdn_pdl" in slots and "gdn_verify" not in slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'gdn_pdl' launches slot 'gdn_verify''s verify"
        )
    for name in sorted(
        slots & set(DRAFT_SLOTS) - {"draft_skv2", "draft_skv2_cp", "draft_select"}
    ):
        if "draft_loop" not in slots:
            reasons.append(f"{SLOTS_ENV}: slot {name!r} runs inside entry 'draft_loop'")
    if "draft_hop" in slots and "hop" not in slots:
        reasons.append(f"{SLOTS_ENV}: slot 'draft_hop' runs on slot 'hop''s mailbox")
    if "draft_fin" in slots and "draft_hop" not in slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'draft_fin' extends slot 'draft_hop''s fused final norm"
        )
    if "draft_multi" in slots and "multi_request" not in slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'draft_multi' serves the draft steps of two or more "
            "requests ('multi_request')"
        )
    for name, needs in (
        ("draft_fc_input", "draft_embed_local"),
        ("draft_head_lowm", "draft_multi"),
        ("draft_fc_tp", "draft_fc_input"),
    ):
        if name in slots and needs not in slots:
            reasons.append(f"{SLOTS_ENV}: slot {name!r} needs slot {needs!r}")
    if "draft_embed" in slots and "draft_multi" not in slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'draft_embed' runs in slot 'draft_multi''s steps"
        )
    q_push_needs = {"draft_front", "draft_multi", "cp_q_push"}
    if "draft_q_push" in slots and not q_push_needs <= slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'draft_q_push' pushes slot 'draft_multi''s q rows from "
            "slot 'draft_front' over slot 'cp_q_push''s words"
        )
    if "draft_q_gather" in slots and not {"draft_q_push", "cp_lam_q"} <= slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'draft_q_gather' polls slot 'draft_q_push''s pushed q rows "
            "as slot 'cp_lam_q''s Lamport words"
        )
    for name in ("draft_skv2", "draft_select"):
        if name in slots and "draft_extend" not in slots:
            reasons.append(
                f"{SLOTS_ENV}: slot {name!r} runs inside entry 'draft_extend'"
            )
    for name in sorted(slots & CP_LITE_BUILD_VARIANTS):
        if not slots & CP_LITE_SLOTS:
            reasons.append(
                f"{SLOTS_ENV}: slot {name!r} builds the CP slots' partial "
                f"({', '.join(sorted(CP_LITE_SLOTS))})"
            )
    if "cp_merge_plain_wait" in slots and "skv2_cp" not in slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'cp_merge_plain_wait' polls slot 'skv2_cp''s "
            "multi-request merge"
        )
    if "cp_q_push" in slots and not {"skv2_cp", "mka"} <= slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'cp_q_push' moves slot 'skv2_cp''s q gather push into "
            "slot 'mka''s front"
        )
    for name in sorted(slots & CP_PUSHED_CALL_SLOTS):
        if "cp_q_push" not in slots:
            reasons.append(
                f"{SLOTS_ENV}: slot {name!r} acts on slot 'cp_q_push''s pushed CP-lite call"
            )
    if "cp_o_push" in slots and "cp_merge_plain_wait" not in slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'cp_o_push' feeds slot 'cp_merge_plain_wait''s per-head merge"
        )
    if "cp_q_gather" in slots and not {"cp_q_push", "cp_lam_q"} <= slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'cp_q_gather' polls slot 'cp_q_push''s pushed q rows as "
            "slot 'cp_lam_q''s Lamport words"
        )
    for name, base in (
        ("skv2_cp", "skv2"),
        ("draft_q1_cp", "draft_q1"),
        ("draft_skv2_cp", "draft_skv2"),
    ):
        if name in slots and base not in slots:
            reasons.append(
                f"{SLOTS_ENV}: slot {name!r} takes slot {base!r}'s long-KV graph "
                "variant's place from its length up"
            )
    if "hop_fp8" in slots and "hop" not in slots:
        reasons.append(f"{SLOTS_ENV}: slot 'hop_fp8' is an output of slot 'hop'")
    if "bt_tail_early" in slots and "bt_tail" not in slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'bt_tail_early' changes slot 'bt_tail''s MoE-input launch"
        )
    if "bt_tail_early_fin" in slots and "bt_tail" not in slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'bt_tail_early_fin' changes slot 'bt_tail''s layer-input "
            "launch"
        )
    if "gemm_ar" in slots and "bt_tail" not in slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'gemm_ar' runs in slot 'bt_tail''s MoE-input point"
        )
    if "fc2ar" in slots and not {"bt_tail", "moe_engine"} <= slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'fc2ar' folds slot 'moe_engine''s FC2 rows at slot "
            "'bt_tail''s points"
        )
    if "lowm_push" in slots and not {"lowm_out", "hop"} <= slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'lowm_push' pushes slot 'lowm_out''s rows into slot "
            "'hop''s mailbox"
        )
    if "mka_push" in slots and not {"mka", "hop"} <= slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'mka_push' pushes slot 'mka''s epilogue into slot "
            "'hop''s mailbox"
        )
    if (
        "route_lin" in slots
        and not {
            "route",
            "mkm",
            "hop",
            "gdn_push",
            "mka",
            "mka_push",
        }
        <= slots
    ):
        reasons.append(
            f"{SLOTS_ENV}: slot 'route_lin' runs the router before the all-reduce "
            "through slots 'route', 'mkm', 'hop', 'gdn_push', 'mka' and 'mka_push'"
        )
    if "gumbel_fast" in slots and "sharded_gumbel" not in slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'gumbel_fast' draws slot 'sharded_gumbel''s noise"
        )
    if "verify_head" in slots and "sharded_gumbel" not in slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'verify_head' is offered by slot 'sharded_gumbel''s "
            "provider"
        )
    if "draft_cp" in slots and not slots & CP_LITE_SLOTS:
        reasons.append(
            f"{SLOTS_ENV}: slot 'draft_cp' runs on CP-lite, which a CP slot builds "
            f"({', '.join(sorted(CP_LITE_SLOTS))})"
        )
    for name in ("draft_gate_fold", "draft_o_push"):
        if name in slots and not {"draft_multi", "draft_cp"} <= slots:
            reasons.append(
                f"{SLOTS_ENV}: slot {name!r} acts on slot 'draft_multi''s CP-lite call "
                "made precise by slot 'draft_cp'"
            )
    if "verify_head_keys" in slots and not {"verify_head", "gumbel_fast"} <= slots:
        reasons.append(
            f"{SLOTS_ENV}: slot 'verify_head_keys' hands slot 'verify_head''s keys to "
            "slot 'gumbel_fast''s launch"
        )
    if "route_lin" in slots and "route_shared" in slots and "lowm_se" not in slots:
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
        "SupportsRecurrentCommit",
    ),
    "srt/runtime_context.py": (
        "get_parallel",
        "get_memory",
        '"tp_size"',
        '"attn_tp_size"',
        '"moe_ep_size"',
    ),
    "srt/arg_groups/fields/*.py": ("disable_radix_cache", "enable_unified_memory"),
    "srt/kernel_plugins/nvidia/eagle_binding.py|srt/megakernels/eagle_binding.py": (
        "page_size",
        "tp_rank",
        "tp_size",
        "tp_device_group",
    ),
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/runtime.py": (
        "megakernel_agree",
    ),
    "srt/environ.py": ("envs", "SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION"),
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
    from sgl_nvidia_mega.qwen3_5.runtime.executor import Qwen35FoldVerify, Qwen35Verify

    parallel = get_parallel()
    apply_measured_plan(parallel)
    pp_size = getattr(parallel, "pp_size", 1)
    # only the PD prefill role runs PP (spec.pd_reasons)
    if pp_size > 1:
        from sgl_nvidia_mega.qwen3_5.runtime import prefill_fp8
        from sgl_nvidia_mega.qwen3_5.runtime.prefill import Qwen35Prefill

        label = f"TP{parallel.tp_size}/PP{pp_size}"
        declared = declared_prefill_slots(parallel)
        unsupported = SLOTS - set(declared)
        why = f"only supports {', '.join(declared)}" if declared else "runs no slot"
        if unsupported:
            logger.info(
                "NVIDIA megakernel: slots %s use stock (PD prefill %s %s)",
                sorted(unsupported),
                label,
                why,
            )
        return Qwen35Prefill(
            model,
            ctx,
            "gdn_prefill_split_conv" in SLOTS.intersection(declared),
            label=label,
            split_conv="gdn_prefill_split_conv" in declared,
            front="gdn_prefill_front" in SLOTS.intersection(declared),
            track_snapshot="gdn_prefill_track_snapshot" in SLOTS.intersection(declared),
            moe_input_nvfp4="moe_prefill_input_nvfp4" in SLOTS.intersection(declared),
            moe_add_norm="moe_prefill_add_norm" in SLOTS.intersection(declared),
            moe_noclone="moe_prefill_noclone" in SLOTS.intersection(declared),
            fp8_rows=[
                name
                for name in prefill_fp8.ROWS
                if name in SLOTS.intersection(declared)
            ],
            qk_rows="attn_prefill_qk_rows" in SLOTS.intersection(declared),
        )

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
    current = topology.from_parallel(parallel)
    unsupported = apply_topology(parallel)
    apply_cp_lite_build(ctx.deadline_s)
    if unsupported:
        logger.info(
            "NVIDIA megakernel: Qwen3.5 %s: slots %s use stock (topology)",
            current.name,
            sorted(unsupported),
        )
    multi = current.stock_widths or slot_plan.MULTI_REQUEST in SLOTS
    if multi and not current.stock_widths:
        multi = slot_plan.multi_request(parallel.tp_size, SLOTS)
    widths = slot_plan.candidates(ctx.widths, ctx.tokens_per_request, multi)
    apply_commit_fold(ctx.widths, widths)
    if not widths:
        unavailable_slots(
            SLOTS,
            f"{ctx.tokens_per_request} rows a request leave no covered width",
            config=True,
        )
        return Qwen35Verify(
            model, {}, {}, (), binding=ctx.binding, summary="none (stock)"
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
    from sgl_nvidia_mega.qwen3_5.runtime import embedding

    bound["embed_local"] = embedding.build_local(model, ctx.binding)
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
        gdn_conv_core=bool(bound["gdn"].arrive),
        lowm_ba=bool(bound["gdn"].ba_lowm),
        gdn_core_tc=bound["gdn"].gdn_core_tc,
        gdn_commit_fold=bound["gdn"].gdn_commit_fold,
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
        route_big_mid=tep8_moe and bound["moe"].big_mid,
        lowm_se_tep8=bound["lowm"].se_tep8,
        bt_tail=bound["boundary"].bt is not None,
        bt_tail_early=bound["boundary"].bt is not None and "bt_tail_early" in SLOTS,
        bt_tail_early_fin=bound["boundary"].bt is not None
        and "bt_tail_early_fin" in SLOTS,
        gemm_ar=bound["boundary"].bt is not None
        and bound["boundary"].bt.gemm_ar is not None,
        fc2ar=bound["boundary"].bt is not None
        and bound["boundary"].bt.fc2ar is not None,
    )
    covered = slot_plan.covered(
        pctx.widths, facts, pctx.tokens_per_request, stock=current.stock_widths
    )
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
    # bt_tail has nothing to serve when no covered step is inside its rows
    if not any(
        slot_plan.BT_TAIL_MIN_ROWS <= w <= slot_plan.BT_TAIL_MAX_ROWS
        for w in pctx.widths
    ):
        vacuous.add("bt_tail")
    for name, lo, hi in (
        ("gemm_ar", slot_plan.GEMM_AR_MIN_ROWS, slot_plan.GEMM_AR_MAX_ROWS),
        ("fc2ar", slot_plan.FC2AR_MIN_ROWS, slot_plan.FC2AR_MAX_ROWS),
    ):
        if not any(lo <= w <= hi for w in pctx.widths):
            vacuous.add(name)
    # gdn_conv_core has nothing to serve when no covered step is inside its rows
    if (
        slot_plan.widest(pctx.widths, slot_plan.GDN_CONV_CORE_MAX_ROWS)
        < slot_plan.GDN_CONV_CORE_MIN_ROWS
    ):
        vacuous.add("gdn_conv_core")
    # lowm_ba has nothing to serve when no covered step is inside its rows
    if not any(
        slot_plan.LOWM_BA_MIN_ROWS <= w <= slot_plan.LOWM_BA_MAX_ROWS
        for w in pctx.widths
    ):
        vacuous.add("lowm_ba")
    # route_topk has nothing to serve where slot route covers every width
    if "route" in SLOTS and max(pctx.widths) <= slot_plan.ROUTE_MAX_ROWS:
        vacuous.update({"route_topk", "route_gemm"})
    missing = sorted(name for name in target_slots - vacuous if not built[name])
    if missing:
        bt_why = (
            f"BT tail: {bound['boundary'].bt_reason or 'built'}; "
            if "bt_tail" in SLOTS
            else ""
        )
        if bound["boundary"].bt is not None:
            bt_why += "".join(
                f"{name}: {why}; "
                for name, why in bound["boundary"].bt.ar_reasons.items()
                if name in SLOTS
            )
        logger.info(
            f"NVIDIA megakernel: requested slots {missing} use stock (MoE: "
            f"{bound['moe'].reason or 'built'}; GDN: {bound['gdn'].reason or 'built'}; "
            f"boundary: {bound['boundary'].reason or 'built'}; {bt_why}"
            f"attention: {bound['attention'].reason or 'built'}; "
            f"low-M: {bound['lowm'].reason or 'built'})"
        )
    checker = (
        parity.Parity(covered, torch.cuda.current_device(), model)
        if parity.ENABLED
        else None
    )
    hook_factory = (
        hooks.build(
            model,
            ctx.binding,
            sharded="sharded_gumbel" in SLOTS,
            gathered="gathered_cdf" in SLOTS,
            draft="draft_cp" in SLOTS,
        )
        if {"sharded_gumbel", "gathered_cdf", "draft_cp"} & SLOTS
        else None
    )
    on = sorted(name for name, ok in built.items() if ok)
    return (Qwen35FoldVerify if facts.gdn_commit_fold else Qwen35Verify)(
        model,
        slots,
        bound=bound,
        widths=covered,
        binding=ctx.binding,
        summary=f"slots {on or 'none (stock)'}{wider}",
        parity=checker,
        speculative_runtime_hooks=hook_factory,
        recurrent_commit=build_recurrent_commit(),
    )
