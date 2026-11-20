# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot capabilities, resolved once before any package allocates or compiles.

Widths are local target-verify rows (requests times tokens_per_request). DEP8's
attention is DP8/TP1 and its MoE is EP8; it must not inherit TP8/TEP8 kernels just
because its global TP group also contains eight ranks.
"""

from __future__ import annotations

import msgspec

SGLANG_READS = {
    "srt/runtime_context.py": ('"tp_size"', '"attn_dp_size"', '"moe_ep_size"'),
    "srt/arg_groups/fields/*.py": (
        "tp_size",
        "attn_dp_size",
        "ep_size",
        "dp_size",
        "enable_dp_attention",
    ),
}

TOPOLOGIES = frozenset({"tp2", "tp4", "tp8", "tep8", "dep8"})
LEGACY = frozenset({"tp8", "tep8"})
PREFILL_TOPOLOGIES = frozenset({"p_tp1_pp4"})

# Extend a slot's entry only alongside its topology-specific build/shape checks
# and correctness evidence. A missing entry supports no topology.
SLOT_TOPOLOGIES = {
    name: LEGACY
    for name in (
        "route",
        "route_topk",
        "route_gemm",
        "mkm",
        "mkm_wide",
        "mkm_m32",
        "mkm_m64",
        "mkm_m128",
        "mkm_m256",
        "mkm_m512",
        "mkm_wide_bal",
        "gdn_verify",
        "gdn_pdl",
        "gdn_conv",
        "gdn_core",
        "gdn_push",
        "gdn_front",
        "gdn_trt_front",
        "hop",
        "hop_mid",
        "hop_fp8",
        "route_shared",
        "route_lin",
        "skv2",
        "skv2_cp",
        "mka",
        "mka_push",
        "lowm_in",
        "lowm_out",
        "lowm_se",
        "lowm_push",
        "draft_loop",
        "draft_extend",
        "draft_moe",
        "draft_q1",
        "draft_xchg",
        "draft_hop",
        "draft_fin",
        "draft_front",
        "draft_skv2",
        "draft_multi",
        "draft_embed",
        "draft_q1_cp",
        "draft_skv2_cp",
    )
}
# GDN's opened chain uses the layer's local head counts; the Triton verify
# supports the aggregate 16-entry ring and DEP8's 32-entry ring. The numeric
# CUDA core retains its measured two-tile (16-entry) history implementation.
SLOT_TOPOLOGIES.update(
    gdn_verify=TOPOLOGIES,
    gdn_pdl=TOPOLOGIES,
    gdn_conv=TOPOLOGIES,
    gdn_core=frozenset({"tp2", "tp4", "tp8", "tep8"}),
)
SLOT_TOPOLOGIES["gdn_prefill_split_conv"] = PREFILL_TOPOLOGIES
# a9212216c6 already serves these TP4 shapes. Preserve that baseline support:
# MoE's small engine accepts I=256, GDN uses the layer's local head counts,
# the hop serves world 4, attention serves 8 Q / 1 KV heads, and the draft's
# exchange/embedding kernels serve world 4. Each package keeps its shape checks.
# The I=128-only wide MoE engines, TP8 GDN fronts and TP8 router-lin stay excluded.
TP4_V2_SLOTS = frozenset(
    {
        "route",
        "route_topk",
        "route_gemm",
        "mkm",
        "gdn_verify",
        "gdn_pdl",
        "gdn_conv",
        "gdn_core",
        "gdn_push",
        "hop",
        "hop_mid",
        "hop_fp8",
        "route_shared",
        "skv2",
        "skv2_cp",
        "mka",
        "mka_push",
        "lowm_in",
        "lowm_out",
        "lowm_se",
        "lowm_push",
        "draft_loop",
        "draft_extend",
        "draft_moe",
        "draft_q1",
        "draft_xchg",
        "draft_hop",
        "draft_fin",
        "draft_front",
        "draft_skv2",
        "draft_multi",
        "draft_embed",
        "draft_q1_cp",
        "draft_skv2_cp",
    }
)
for name in TP4_V2_SLOTS:
    SLOT_TOPOLOGIES[name] = SLOT_TOPOLOGIES[name] | {"tp4"}

# These names select coverage or SGLang's own runtime policy, not a plugin kernel.
SLOT_TOPOLOGIES.update(
    multi_request=TOPOLOGIES,
    sharded_gumbel=TOPOLOGIES,
)
# Full-step AB+BA prefers v2's stock draft on TP2. Keep every draft entry
# filtered there, including PD decode TP2; TP4/TP8 retain the BF16 entries.
for _name in (
    "draft_loop",
    "draft_extend",
    "draft_multi",
    "draft_embed",
    "draft_front",
):
    SLOT_TOPOLOGIES[_name] = frozenset({"tp4", "tp8", "tep8"})
# TP4/TP8 retain the fused vocabulary exchange.
SLOT_TOPOLOGIES["draft_xchg"] = frozenset({"tp4", "tp8", "tep8"})
SLOT_TOPOLOGIES["draft_moe"] = frozenset({"tp4", "tp8"})
SLOT_TOPOLOGIES["draft_q1"] = frozenset({"tp4", "tp8", "tep8"})
# Router-only fronts leave expert GEMMs with the stock TRTLLM runner. Their
# 4096-hidden / 512-expert routing does not depend on TP's intermediate width.
SLOT_TOPOLOGIES.update(
    route=frozenset({"tp2", "tp4", "tp8", "tep8"}),
    route_topk=frozenset({"tp2", "tp4", "tp8", "tep8"}),
    route_gemm=frozenset({"tp2", "tp4", "tp8", "tep8"}),
)
# ATTN: the generic front uses stock projection GEMMs on TP2 and DEP8.
SLOT_TOPOLOGIES.update(
    mka=TOPOLOGIES,
    skv2=frozenset({"tp4", "tp8", "tep8"}),
    skv2_cp=frozenset({"tp4", "tp8", "tep8"}),
)


class Topology(msgspec.Struct, frozen=True):
    tp_size: int
    attn_dp_size: int = 1
    moe_ep_size: int = 1

    @property
    def name(self) -> str:
        if self.attn_dp_size > 1:
            return (
                "dep8"
                if (self.tp_size, self.attn_dp_size, self.moe_ep_size) == (8, 8, 8)
                else "unsupported"
            )
        if self.tp_size == 8 and self.moe_ep_size > 1:
            return "tep8"
        return f"tp{self.tp_size}"

    @property
    def stock_widths(self) -> bool:
        """New topologies cover all offered widths even with every slot off."""
        return self.name in {"tp2", "tp4", "dep8"}


def from_parallel(parallel) -> Topology:
    return Topology(
        parallel.tp_size,
        getattr(parallel, "attn_dp_size", 1),
        parallel.moe_ep_size,
    )


def from_config(cfg) -> Topology:
    # Startup guards may run before legacy --dp-size/--enable-dp-attention is
    # normalized to attn_dp_size; the published ParallelContext is already resolved.
    attention_dp = cfg.attn_dp_size or 1
    if cfg.enable_dp_attention:
        attention_dp = max(attention_dp, cfg.dp_size or 1)
    return Topology(cfg.tp_size, attention_dp, cfg.ep_size or 1)


def select(requested: frozenset, topology: Topology) -> frozenset:
    """Only explicitly declared support enables a kernel on this topology."""
    return frozenset(
        name
        for name in requested
        if topology.name in SLOT_TOPOLOGIES.get(name, frozenset())
    )


# The slots route_lin runs through (runtime.slot_reasons); with route_shared, lowm_se
# replaces it at one request.
ROUTE_LIN_NEEDS = frozenset({"route", "mkm", "hop", "gdn_push", "mka", "mka_push"})


def _route_lin_fits(slots: frozenset) -> bool:
    return ROUTE_LIN_NEEDS <= slots and (
        "route_shared" not in slots or "lowm_se" in slots
    )


def plan(requested: frozenset, topology: Topology) -> frozenset:
    """``select``, plus the slots a topology's plan turns on with the requested ones:
    the TP8 plan (aggregated and PD decode) adds route_lin wherever its slots are
    selected, and balances the mkm_m256/mkm_m512 work split when offered.
    Every other topology keeps exactly its selected slots."""
    selected = select(requested, topology)
    if topology.name == "tp8" and _route_lin_fits(selected):
        selected |= {"route_lin"}
    if topology.name == "tp8" and {"mkm_m256", "mkm_m512"} & selected:
        selected |= {"mkm_wide_bal"}
    return selected


def plannable(requested: frozenset) -> frozenset:
    """``requested`` plus every slot some topology's plan can add to it (the numerics
    claim made before the topology is known)."""
    planned = requested | ({"route_lin"} if _route_lin_fits(requested) else frozenset())
    if {"mkm_m256", "mkm_m512"} & requested:
        planned |= {"mkm_wide_bal"}
    return planned
