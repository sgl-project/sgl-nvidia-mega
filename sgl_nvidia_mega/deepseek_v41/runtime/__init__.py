# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The megakernel runtime: the executor of a covered decode step (``executor``), its
per-step state (``state``), the embedding (``embedding``), and ``build``, which SGLang
runs through the spec once the model is prepared and before the KV cache is sized."""

from __future__ import annotations

import os
from typing import Any

import msgspec

# The kernel slots this package ships; with the megakernel on, all of them run. The
# Engram prefetch needs SGLang's per-rank Engram host table (spec.REQUIREMENTS).
SHIPPED_SLOTS = (
    "hop_a",
    "moe",
    "gate",
    "attn",
    "embed",
    "two_shot",
    "markov",
    "draft_lm_head",
    # aux_mean_stats: the DSpark capture layers' streams mean on the mHC stats stream,
    # beside the attention instead of on the main stream before it (bitwise)
    "aux_mean_stats",
    # gate_wide: the fast gate top-k at every routed width (1-48), target and draft
    "gate_wide",
    # shared_ll: the shared expert's gate_up on the low-latency GEMM, quantized at the
    # MoE fork on the alternate stream (moe/shared_ll.py; not bitwise: summation order)
    "shared_ll",
    # hop_a_batch2: hop A at 9-16 rows (batch 2) on the fused wo_a's row-tile scales
    "hop_a_batch2",
    "engram",
    "attn_qkv_a",
    "attn_qkv_a_bf16",
    # index_q_mxfp8_in: the indexer wq_b reads the chain's q_lora quant (bitwise)
    "index_q_mxfp8_in",
    # attn_indexer: a covered step's indexer top-k in the plugin's forward, the index
    # query and head weights on the layer's second stream (attention/indexer.py; bitwise)
    "attn_indexer",
    # hc_prenorm: the mHC coefficients from DeepGEMM's TF32 prenorm (not bitwise)
    "hc_prenorm",
    "hop_b",
    # hop_b_wide: hop B (the MoE post) at 9-48 rows on a mailbox of its own, writing
    # row-tile scales for wqkv_a on the wide low-latency GEMM (not bitwise past 8 rows)
    "hop_b_wide",
    "early_trigger",
    # hop_b_start_trigger: the MoE post's Lamport consumer lets the next layer's
    # first GEMM launch at its start, so its weight loads overlap the poll (bitwise)
    "hop_b_start_trigger",
    "draft",
    "draft_moe",
    "draft_gate",
    "draft_tail",
    "draft_post",
    # draft_hop: at attention TP8 with MoE TP4 x EP2, where SGLang has no fused post,
    # draft_post's posts on the plugin's Lamport all-reduce + mHC (no effect at TP4)
    "draft_hop",
    "draft_seam",
    "draft_attn",
    "draft_qfront",
    "draft_wo_a",
    "draft_swapab",
    "draft_qkv_a",
    "draft_prenorm",
    "draft_qkv_a_ll",
    "draft_wo_b_ll",
    "sched_ahead",
    "req_pos_once",
    # kv_before_wait: on the low-ratio layers swap-AB gathers the window KV before its
    # PDL wait; a layer's own sparse range (its top-k's selection, its compressor's
    # rows) only after it, since a captured stream join is a programmatic edge and
    # SGLang's top-k releases its dependents before it writes (bitwise). The window KV
    # store also releases before it stores; its rows are still gathered before the wait
    "kv_before_wait",
)
# the draft's slots, which build with the draft spec
DRAFT_SLOTS = frozenset(
    {
        "markov",
        "draft_lm_head",
        "draft",
        "draft_moe",
        "draft_gate",
        "draft_tail",
        "draft_post",
        "draft_hop",
        "draft_seam",
        "draft_attn",
        "draft_qfront",
        "draft_wo_a",
        "draft_swapab",
        "draft_qkv_a",
        "draft_embed",
        "draft_prenorm",
        "draft_qkv_a_ll",
        "draft_wo_b_ll",
        "markov_pdl",
        "markov_lowlat",
    }
)
# Slots that run only when listed (draft_embed: the draft's tokens from the target's
# gathered table; hop_b_cap16: hop_b_wide's row tiles at 9-16 rows only, no 48-row
# mailbox, which measured +16 us on the batch-1 verify graph; target_lm_head: the
# target's LM head on FlashInfer's low-M kernel at 1-8 rows, not bitwise; hop_b_quad_trigger:
# hop B's finalize-publish lets the consumer launch right after its own wait, not at its end;
# lm_head_pinned_tactic: the low-M LM heads on one fixed tactic per row count, untuned;
# hop_mbar: the Lamport consumers' RMSNorm sums cross the cluster by st.async onto
# mbarriers, not a cluster barrier, bitwise; hop_final_collapse: the last layer's hop B
# also writes the head's mHC collapse, so the head runs only its RMSNorm, bitwise;
# attn_indexer_narrow: slot attn_indexer at the swap-AB widths too (1-24 rows);
# attn_indexer_fused: that, and the candidate consumers' logits + top-k at 1-8 rows in
# one cluster launch, sm103/attention/reindex_select.py;
# target_lm_head_mxfp8 / draft_lm_head_mxfp8: the target's / the draft's LM head at 1-8
# rows on an MXFP8 copy of the shard, a numerics change; the draft's is measurement only,
# plan.MEASUREMENT_ONLY_SLOTS;
# shared_fold: slot shared_ll's gate_up epilogue writes the clamped SwiGLU, bitwise;
# shared_fold_quant: it also writes down_proj's MXFP8 input, bitwise;
# shared_fold_early_release: that GEMM releases down_proj's GEMM at its loads, as
# FlashInfer's own gate_up does, not after its stores;
# prologue_fork: the Engram hashes forked at the step start and the embedding written
# into the streams, bitwise; hc_coeff_fused: the prenorm's mHC coefficients in one
# launch, bitwise with hc_prenorm's two;
# swapab_h8 / wo_a_g1: at attention TP8, swap-AB on the 8 local heads and the fused wo_a
# on the 1 local group, target and draft, in place of SGLang's FlashMLA and einsum, which
# TP8 runs (each bitwise to its TP4 kernel on the same heads / group, not to SGLang's
# TP8 arm); no effect at TP4;
# gate_route: the router's top-k and the routed pair's route
# record in one cluster launch at 1-8 rows, target and draft (sm103/moe/src/gate_route.cuh;
# bitwise with gate_topk + route); gate_route_quant: that launch also writes the routed
# input's MXFP8 quant (FlashInfer's linear-layout mxfp8_quantize bit for bit,
# sm103/moe/src/route_quant.cuh), so the MoE half forks no routed-quant stream and the
# top-k launches as the router's programmatic dependent, target and draft (bitwise);
# markov_pdl: the draft's Markov chain launched as a programmatic dependent of the draft
# LM head, loading its head tiles before the wait, the anchors staged before the LM head;
# markov_lowlat: the Markov chain's low-latency exchange, converged pollers and acquire
# loads, the direct exchange to 4 rows at world 4 and 2 at world 8, the narrow kernel to
# 8 rows at every world; bitwise);
# attn_q_norm_gemm: at 1-8 rows the q_lora norm + MXFP8 quant inside wq_b's GEMM in
# the layers without an indexer, target and draft (sm103/attention/q_norm_gemm.py;
# bitwise; 0-0.5 us a layer); attn_index_q_first: the plugin indexer's index query
# (wq_b, RoPE + pack) on the main stream and wq_b on the second, so the index chain does
# not wait for the SMs wq_b and the attention hold (bitwise); attn_index_weights_bf16:
# the index head weights the BF16 scorers read made beside the head weights, not cast
# before the logits (bitwise); attn_cluster_merge: swap-AB merges a row's tiles through its
# CTA cluster's shared memory instead of the separate combine (swapab.cluster_merge_takes:
# 8 heads, the low-ratio range, 1-8 rows), target and draft, bitwise; faster in the
# attention-half bench, no step gain served at TP8 batch 1; gemm2_pipe: the routed
# GEMM2, target and draft, on muLLM's reference-pipeline + prelude build of the same
# kernel (bitwise);
# attn_wo_a_fp8: the target's fused wo_a streams wo_a as E4M3 with 32x32 UE8M0 scales
# (half the bytes), recovered from SGLang's BF16 wo_a and widened exactly in shared memory;
# bitwise where the recovery is exact, which build checks per layer;
# scheduling switches derived from muLLM (NVIDIA Corporation, Apache-2.0), all bitwise (the same
# kernels on the same inputs): attn_compressor_priority: the compressor layers' compressor
# on a stream at the device's highest priority; attn_index_early_trigger: the fused indexer
# consumer lets the attention launch right after its PDL wait and the attention reads that
# selection after its own wait; attn_defer_block_select: the candidate source's block
# selection after the layer's attention, not beside it)
OPT_IN_SLOTS = (
    "draft_embed",
    "hop_b_cap16",
    "target_lm_head",
    "hop_b_quad_trigger",
    "lm_head_pinned_tactic",
    "hop_mbar",
    "hop_final_collapse",
    "attn_indexer_narrow",
    "attn_indexer_fused",
    "target_lm_head_mxfp8",
    "draft_lm_head_mxfp8",
    "shared_fold",
    "shared_fold_quant",
    "shared_fold_early_release",
    "prologue_fork",
    "hc_coeff_fused",
    "swapab_h8",
    "wo_a_g1",
    "gate_route",
    "gate_route_quant",
    "markov_pdl",
    "markov_lowlat",
    "attn_q_norm_gemm",
    "attn_index_q_first",
    "attn_index_weights_bf16",
    "attn_cluster_merge",
    "gemm2_pipe",
    "attn_wo_a_fp8",
    "attn_compressor_priority",
    "attn_index_early_trigger",
    "attn_defer_block_select",
)
# Hardware covered only when listed in the same variable (sm148: the 148 SMs of a B300,
# the target spec's resident blocks)
OPT_IN_COVERAGE = ("sm148",)
SLOTS_ENV = "SGLANG_NVIDIA_MEGAKERNEL_SLOTS"


def _requested_slots() -> frozenset:
    """``SLOTS_ENV`` (comma-separated; for tests and bisecting: "" runs every step
    function's stock code), else every shipped slot."""
    raw = os.environ.get(SLOTS_ENV)
    if raw is None:
        return frozenset(SHIPPED_SLOTS)
    return frozenset(name.strip() for name in raw.split(",")) - {""}


SLOTS = _requested_slots()


def row_tile_layers(attention: dict, slots: frozenset) -> frozenset:
    """Slot hop_b_wide: the layers whose chain runs wqkv_a on the low-latency GEMM, for
    which hop B writes row-tile scales past 8 rows (none without the slot)."""
    if "hop_b_wide" not in slots:
        return frozenset()
    return frozenset(
        lid
        for lid, step in attention.items()
        if step.chain is not None and step.chain[0].qkv_a_ll
    )


def slot_reasons() -> list[str]:
    """Why the requested slots cannot be served ([] when they can)."""
    known = (*SHIPPED_SLOTS, *OPT_IN_SLOTS, *OPT_IN_COVERAGE)
    reasons = [
        f"{SLOTS_ENV}: unknown slot {name!r} (known: {', '.join(known)})"
        for name in sorted(SLOTS - set(known))
    ]
    for slot, needs in (
        ("gate", "moe"),
        ("shared_ll", "moe"),
        ("shared_fold", "shared_ll"),
        ("shared_fold_quant", "shared_fold"),
        ("shared_fold_early_release", "shared_fold_quant"),
        ("gate_wide", "gate"),
        ("gate_route", "gate"),
        ("gate_route_quant", "gate_route"),
        ("hop_a_batch2", "hop_a"),
        ("attn_qkv_a", "attn"),
        ("attn_indexer", "attn"),
        ("attn_indexer_narrow", "attn_indexer"),
        ("attn_indexer_fused", "attn_indexer"),
        ("attn_qkv_a_bf16", "attn_qkv_a"),
        ("index_q_mxfp8_in", "attn"),
        ("hop_b", "hop_a"),
        ("early_trigger", "hop_a"),
        ("hop_b_start_trigger", "hop_b"),
        ("hop_b_wide", "hop_b"),
        ("hop_b_wide", "attn_qkv_a"),
        ("hop_b_cap16", "hop_b_wide"),
        ("hop_b_quad_trigger", "hop_b"),
        ("hop_mbar", "hop_a"),
        ("hop_final_collapse", "hop_b"),
        ("draft_lm_head", "markov"),
        ("target_lm_head_mxfp8", "target_lm_head"),
        ("draft_lm_head_mxfp8", "draft_lm_head"),
        ("markov_pdl", "markov"),
        ("markov_lowlat", "markov"),
        ("kv_before_wait", "attn"),
        ("draft_moe", "draft"),
        ("draft_gate", "draft_moe"),
        ("draft_tail", "draft"),
        ("draft_post", "draft"),
        ("draft_seam", "draft_post"),
        ("draft_hop", "draft_post"),
        ("draft_attn", "draft"),
        ("draft_qfront", "draft_attn"),
        ("draft_wo_a", "draft_attn"),
        ("draft_swapab", "draft_attn"),
        ("draft_qkv_a", "draft_attn"),
        ("draft_embed", "draft"),
        ("draft_embed", "embed"),
        ("draft_prenorm", "draft"),
        ("draft_qkv_a_ll", "draft_qkv_a"),
        ("draft_wo_b_ll", "draft_wo_a"),
        ("hc_coeff_fused", "hc_prenorm"),
        ("swapab_h8", "attn"),
        ("wo_a_g1", "attn"),
        ("attn_q_norm_gemm", "attn"),
        ("attn_index_q_first", "attn_indexer"),
        ("attn_index_weights_bf16", "attn_indexer"),
        ("attn_cluster_merge", "attn"),
        ("attn_wo_a_fp8", "attn"),
        ("attn_compressor_priority", "attn"),
        ("attn_index_early_trigger", "attn_indexer_fused"),
        ("attn_index_early_trigger", "kv_before_wait"),
        ("attn_defer_block_select", "attn_indexer"),
    ):
        if slot in SLOTS and needs not in SLOTS:
            reasons.append(f"{SLOTS_ENV}: slot {slot!r} needs slot {needs!r}")
    if "lm_head_pinned_tactic" in SLOTS and not SLOTS & {
        "draft_lm_head",
        "target_lm_head",
    }:
        reasons.append(
            f"{SLOTS_ENV}: slot 'lm_head_pinned_tactic' needs slot 'draft_lm_head' "
            "or 'target_lm_head'"
        )
    return reasons


# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/runtime_context.py": (
        "get_parallel",
        '"tp_group"',
        '"attn_dp_size"',
    ),
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": (
        "widths",
        "tokens_per_request",
        "deadline_s",
    ),
    "srt/models/deepseek_v4.py": (
        "DeepseekV4CudaDecoderLayer",
        "hc_pre_from_prev_sublayer",
        "model",
        "start_layer",
        "end_layer",
        "layers",
    ),
}


class PackageContext(msgspec.Struct, frozen=True):
    """What every package's ``build(model, ctx, group)`` gets."""

    widths: tuple[int, ...]  # the covered steps' rows (requests x rows a request)
    tokens_per_request: int
    deadline_s: float  # each cross-GPU wait's deadline
    copies: dict  # (kind, name) -> tensor: weight copies the packages keep
    # SGLang's runtime state for this model (megakernels/binding.py), or None
    binding: Any = None


def model_disable_reasons(model) -> list[str]:
    """What the server arguments do not show: the layer loop the executor follows."""
    from sglang.srt.models.deepseek_v4 import DeepseekV4CudaDecoderLayer
    from sglang.srt.runtime_context import get_parallel

    m = model.model
    reasons = []
    if not m.hc_pre_from_prev_sublayer:
        reasons.append("needs the mHC pre-mix carried from the previous sublayer")
    if get_parallel().attn_dp_size > 1:
        reasons.append("attention DP: the covered step reads replicated token ids")
    other = [
        i
        for i in range(m.start_layer, m.end_layer)
        if not isinstance(m.layers[i], DeepseekV4CudaDecoderLayer)
    ]
    if other:
        reasons.append(f"layers {other}: not SGLang's CUDA decoder layer")
    return reasons


def built_slots(slots: dict, bound: dict, facts: Any) -> dict[str, bool]:
    """Each shipped slot: whether its build serves at least one width."""
    folds = [
        getattr(slot.shared, "fold", None)
        for slot in getattr(bound["moe"], "slots", {}).values()
        if slot.shared is not None
    ]
    folds = [fold for fold in folds if fold is not None]
    return {
        # slots shared_fold*: a shared expert's gate_up folds the SwiGLU (and the quant,
        # and releases down_proj early); a fold that steps aside leaves them unbuilt
        "shared_fold": bool(folds),
        "shared_fold_quant": any(fold.buffers is not None for fold in folds),
        "shared_fold_early_release": any(fold.early_release for fold in folds),
        "hop_a": any(s.hop_a for s in slots.values()),
        "moe": facts.routed,
        # the MoE build's own result: a layer's fast top-k may fail to build alone
        "gate": any(s.fast_gate for s in slots.values())
        and any(slot.gate for slot in getattr(bound["moe"], "slots", {}).values()),
        "attn": any(step.chain is not None for step in bound["attention"].values()),
        "embed": bound["embedding"] is not None,
        "shared_ll": any(
            slot.shared is not None
            for slot in getattr(bound["moe"], "slots", {}).values()
        ),
        "engram": bound["engram"].prefetch is not None,
        # slot gate_route: a layer's fast top-k built with the route record in one launch
        "gate_route": any(
            getattr(slot.gemm, "gate_route_kernel", None) is not None
            for slot in getattr(bound["moe"], "slots", {}).values()
        ),
        # slot gate_route_quant: that launch's build that quantizes the routed input
        "gate_route_quant": any(
            getattr(slot.gemm, "gate_route_quant_kernel", None) is not None
            for slot in getattr(bound["moe"], "slots", {}).values()
        ),
        # slot gemm2_pipe: a layer's GEMM2 on the pipelined build
        "gemm2_pipe": any(
            getattr(slot.gemm, "gemm2_pipe", None) is not None
            for slot in getattr(bound["moe"], "slots", {}).values()
        ),
        "target_lm_head": bound.get("logits") is not None,
        "target_lm_head_mxfp8": getattr(
            getattr(bound.get("logits"), "quant_method", None), "mxfp8", None
        )
        is not None,
    }


def attn_qkv_a_built(attention: dict, widths) -> bool:
    """Slot attn_qkv_a serves a width: a layer's chain runs wqkv_a on the low-latency
    GEMM, which takes 1-8 rows."""
    from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll

    return any(w <= mxfp8_ll.MAX_ROWS for w in widths) and any(
        step.chain is not None and step.chain[0].qkv_a_ll for step in attention.values()
    )


def build(model, ctx):
    """Each package's build (weights bound, buffers allocated, kernels compiled),
    the slot plan per width, and the executor; ``ctx`` is SGLang's BuildContext."""
    from sglang.srt.runtime_context import get_parallel

    from sgl_nvidia_mega.deepseek_v41 import plan as slot_plan
    from sgl_nvidia_mega.deepseek_v41.attention.chain import SWAPAB_MAX_ROWS
    from sgl_nvidia_mega.deepseek_v41.boundary import prenorm
    from sgl_nvidia_mega.deepseek_v41.runtime import bench, embedding
    from sgl_nvidia_mega.deepseek_v41.runtime.executor import (
        PACKAGES,
        MegakernelDecode,
    )
    from sgl_nvidia_mega.deepseek_v41.sm103.attention.swapab import CLUSTER_HEADS

    # benchmark-only switches: refused without simulated acceptance, before any build
    bench.setup()
    group = get_parallel().tp_group
    if not ctx.widths:
        raise RuntimeError(
            f"NVIDIA megakernel: {ctx.tokens_per_request} rows a request leave no "
            "covered width"
        )
    pctx = PackageContext(
        widths=ctx.widths,
        tokens_per_request=ctx.tokens_per_request,
        deadline_s=ctx.deadline_s,
        copies={},
        binding=getattr(ctx, "binding", None),
    )
    bound = {name: pkg.build(model, pctx, group) for name, pkg in PACKAGES.items()}
    bound["boundary"].row_tile_layers = row_tile_layers(bound["attention"], SLOTS)
    table, embed_reason = None, "not requested"
    if "embed" in SLOTS:
        table, embed_reason = embedding.replicate(
            model.model.embed_tokens, pctx.copies, group
        )
    bound["embedding"] = table
    engram = bound["engram"]
    # the boundary's hop A (its compiled widths) and the MoE step's GEMM pair, as built
    hop = getattr(bound["boundary"], "hop_a", None)
    facts = slot_plan.Facts(
        hop_a=hop is not None,
        hop_a_reason=getattr(bound["boundary"], "hop_a_reason", ""),
        routed=bool(getattr(bound["moe"], "slots", None)),
        embed=table is not None,
        engram_prefetch=engram.prefetch is not None,
        two_shot=getattr(bound["boundary"], "two_shot", None) is not None,
        fast_gate_wide="gate_wide" in SLOTS,
        prologue_fork="prologue_fork" in SLOTS,
    )
    slots = {
        w: msgspec.structs.replace(s, hop_a=s.hop_a and w in hop.rows)
        for w, s in slot_plan.plan(pctx.widths, facts).items()
    }
    built = {
        **built_slots(slots, bound, facts),
        "two_shot": any(s.two_shot for s in slots.values()),
        "hop_b": bound["boundary"].hop_b,
        "attn_qkv_a": attn_qkv_a_built(bound["attention"], pctx.widths),
        "attn_qkv_a_bf16": attn_qkv_a_built(bound["attention"], pctx.widths)
        and any(
            step.chain is not None and step.chain[0].qkv_a_bf16
            for step in bound["attention"].values()
        ),
        "hc_prenorm": bound["boundary"].prenorm,
        "hc_coeff_fused": bound["boundary"].prenorm and bool(prenorm.fused()),
        "early_trigger": "early_trigger" in SLOTS and facts.hop_a,
        "hop_a_batch2": any(
            s.hop_a and w > slot_plan.HOP_A_BATCH1_ROWS for w, s in slots.items()
        ),
        "hop_b_start_trigger": "hop_b_start_trigger" in SLOTS
        and bound["boundary"].hop_b,
        # hop B past 8 rows, with a chain that reads its row-tile rows
        "hop_b_wide": bound["boundary"].hop_b
        and any(w > slot_plan.HOP_A_BATCH1_ROWS for w in hop.hop_b_rows)
        and bool(bound["boundary"].row_tile_layers),
        # hop B on hop A's widths only
        "hop_b_cap16": bound["boundary"].hop_b
        and all(w in hop.rows for w in hop.hop_b_rows),
        "hop_b_quad_trigger": bound["boundary"].hop_b,
        "hop_mbar": facts.hop_a,
        "hop_final_collapse": bound["boundary"].final_collapse,
        # step-head work that needs no build: on when requested
        "sched_ahead": "sched_ahead" in SLOTS,
        # swap-AB serves 1-24 rows on low-ratio layers: built when a covered width is
        "kv_before_wait": "kv_before_wait" in SLOTS
        and any(w <= SWAPAB_MAX_ROWS for w in pctx.widths)
        and any(a.swapab and a.low_ratio for a in bound["attention"].values()),
        "aux_mean_stats": "aux_mean_stats" in SLOTS,
        # built when the fast top-k serves a width past the default rows
        "gate_wide": any(
            s.fast_gate and w not in slot_plan.FAST_GATE_ROWS for w, s in slots.items()
        ),
        "req_pos_once": "req_pos_once" in SLOTS,
        "prologue_fork": "prologue_fork" in SLOTS,
        "attn_indexer": any(
            a.chain is not None and a.chain[0].indexer is not None
            for a in bound["attention"].values()
        ),
        "attn_indexer_narrow": any(
            a.chain is not None and a.chain[1].indexer is not None
            for a in bound["attention"].values()
        ),
        "attn_indexer_fused": any(
            a.chain is not None
            and a.chain[0].indexer is not None
            and a.chain[0].indexer.fused
            for a in bound["attention"].values()
        ),
        # the in-cluster merge takes swap-AB on 8 local heads only (attention TP8)
        "attn_cluster_merge": any(
            getattr(a, "cluster_merge", False) and a.attn.n_local_heads in CLUSTER_HEADS
            for a in bound["attention"].values()
        ),
        # the norm inside wq_b's GEMM: a chain with the low-latency wq_b, 1-8 rows
        "attn_q_norm_gemm": any(w <= 8 for w in pctx.widths)
        and any(
            a.chain is not None and a.chain[1].q_norm_gemm
            for a in bound["attention"].values()
        ),
        "attn_wo_a_fp8": any(
            getattr(a, "wo_a_fp8_weight", None) is not None
            for a in bound["attention"].values()
        ),
        "attn_index_q_first": any(
            a.chain is not None
            and a.chain[0].indexer is not None
            and a.chain[0].index_q_first
            for a in bound["attention"].values()
        ),
        # a compressor layer's chain runs its compressor on the high-priority stream
        "attn_compressor_priority": any(
            a.chain is not None and a.chain[0].compressor_stream is not None
            for a in bound["attention"].values()
        ),
        # the fused consumer built, and swap-AB reading its selection after the wait
        "attn_index_early_trigger": any(
            a.chain is not None
            and a.chain[0].indexer is not None
            and a.chain[0].indexer.fused
            and a.chain[0].indexer.early_trigger
            and a.index_after_wait
            for a in bound["attention"].values()
        ),
        # the plugin indexer takes a candidate source
        "attn_defer_block_select": any(
            a.chain is not None
            and a.chain[0].indexer is not None
            and a.chain[0].indexer.defer_blocks
            and getattr(a.attn.indexer, "is_candidate_source", False)
            for a in bound["attention"].values()
        ),
        "attn_index_weights_bf16": any(
            a.chain is not None
            and a.chain[0].indexer is not None
            and a.chain[0].index_weights_bf16
            for a in bound["attention"].values()
        ),
        # the indexer wq_b on the chain's q_lora quant: indexer layers, 1-8 rows
        "index_q_mxfp8_in": any(w <= 8 for w in pctx.widths)
        and any(
            a.chain is not None and a.chain[0].index_q_mxfp8_in
            for a in bound["attention"].values()
        ),
    }
    # kv_before_wait only moves swap-AB's KV gather: with no low-ratio layer on
    # swap-AB (attention TP8 without slot swapab_h8) it has nothing to serve, which is
    # not a failed build; it is left off the active slots
    moot = (
        set()
        if any(a.swapab and a.low_ratio for a in bound["attention"].values())
        else {"kv_before_wait"}
    )
    # likewise attn_cluster_merge without 8 local heads (attention TP4)
    if not any(
        a.attn.n_local_heads in CLUSTER_HEADS for a in bound["attention"].values()
    ):
        moot.add("attn_cluster_merge")
    # every package's build agrees across ranks, so this raises on every rank alike
    missing = sorted(name for name in (SLOTS & built.keys()) - moot if not built[name])
    if missing:
        raise RuntimeError(
            f"NVIDIA megakernel: requested slots {missing} built at no width "
            f"(hop A: {facts.hop_a_reason or 'built'}; MoE: "
            f"{getattr(bound['moe'], 'reason', '') or 'built'}; shared_ll: "
            f"{getattr(bound['moe'], 'shared_reason', '') or 'built'}; embedding: "
            f"{embed_reason or 'built'}; Engram prefetch: {engram.reason or 'built'})"
        )
    # the requested slots that serve: a slot built but not requested (hop_mbar,
    # hop_b_quad_trigger follow from hop A / hop B) does not run, so it is not listed
    active = sorted(name for name, on in built.items() if on and name in SLOTS)
    summary = (
        f"slots {active or 'none (stock)'}; "
        f"hop A at {[w for w, x in slots.items() if x.hop_a]} rows, fast gate at "
        f"{[w for w, x in slots.items() if x.fast_gate]} rows, Engram prefetch at "
        f"{[w for w, x in slots.items() if x.engram_prefetch]} rows "
        f"({engram.reason or 'built, checked bit for bit'}); boundary: "
        f"{bound['boundary'].summary()}; weight copies "
        + str(
            {
                k: f"{v.numel() * v.element_size() / 2**20:.1f} MiB"
                for k, v in pctx.copies.items()
            }
        )
        + bench.fixed_summary()
    )
    return MegakernelDecode(
        model, slots, bound, widths=pctx.widths, summary=summary, binding=ctx.binding
    )
