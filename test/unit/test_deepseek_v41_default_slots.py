# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The default-on slots: shipped, not opt-in, their dependencies shipped too, the
numerics-changing ones labelled, and the unresolved, measurement-only and
layout-dependent slots still opt-in."""

import pytest

from sgl_nvidia_mega.deepseek_v41 import plan, runtime

NUMERICS_DEFAULT_ON = {
    "swapab_h8",
    "wo_a_g1",
    "target_lm_head",
    "target_lm_head_mxfp8",
    "attn_q_cluster",
    "draft_shared_ll",
    "wo_b_publish_split",
}


def test_the_default_on_slots_are_shipped_and_not_opt_in():
    on = set(runtime.DEFAULT_ON_SLOTS)
    assert on <= set(runtime.SHIPPED_SLOTS)
    assert not on & set(runtime.OPT_IN_SLOTS)
    assert len(runtime.DEFAULT_ON_SLOTS) == len(on) == 25
    assert len(runtime.SHIPPED_SLOTS) == len(set(runtime.SHIPPED_SLOTS)) == 65


def test_the_shipped_list_needs_no_opt_in_slot(monkeypatch):
    monkeypatch.setattr(runtime, "SLOTS", frozenset(runtime.SHIPPED_SLOTS))
    assert runtime.slot_reasons() == []


def test_the_numerics_changing_default_slots_are_labelled():
    on = set(runtime.DEFAULT_ON_SLOTS)
    assert on & runtime.NUMERICS_CHANGE_SLOTS == NUMERICS_DEFAULT_ON
    # the bitwise default slots are not labelled
    for slot in (
        "gate_route_early",
        "markov_tail",
        "draft_stage0_table",
        "engram_publish",
        "gemm2_pipe",
    ):
        assert slot in on and slot not in runtime.NUMERICS_CHANGE_SLOTS
    assert runtime.NUMERICS_CHANGE_SLOTS <= set(
        (*runtime.SHIPPED_SLOTS, *runtime.OPT_IN_SLOTS)
    )


@pytest.mark.parametrize(
    "slot",
    [
        # needs SGLANG_DSV4_KV_LAYOUT=v41_new, which the plugin does not select
        "attn_core_lean",
        # no resolved served gain
        "draft_final_collapse",
        "attn_cluster_merge",
        "attn_index_q_first",
        "attn_wo_a_fp8",
        "attn_index_early_trigger",
        "attn_defer_block_select",
        "markov_pdl",
        "attn_index_weights_bf16",
        # not bitwise at attention TP8 + EP2, no resolved gain alone
        "hop_final_collapse",
        # untuned tactics; measurement only
        "lm_head_pinned_tactic",
        "draft_lm_head_mxfp8",
    ],
)
def test_the_remaining_slots_stay_opt_in(slot):
    assert slot in runtime.OPT_IN_SLOTS and slot not in runtime.SHIPPED_SLOTS


def test_no_measurement_only_slot_runs_by_default():
    assert not plan.MEASUREMENT_ONLY_SLOTS & set(runtime.SHIPPED_SLOTS)


def test_the_dropped_slots_are_unknown():
    for slot in ("attn_core_staged", "draft_mx_boundary"):
        assert slot not in (*runtime.SHIPPED_SLOTS, *runtime.OPT_IN_SLOTS)


@pytest.mark.parametrize("slot", sorted(NUMERICS_DEFAULT_ON | {"gate_route_early"}))
def test_a_default_slot_can_be_turned_off(monkeypatch, slot):
    """Listing the shipped slots without one is a valid request (how a user turns a
    default slot off), unless another listed slot needs it."""
    needed_by = {"target_lm_head": "target_lm_head_mxfp8"}
    rest = (
        set(runtime.SHIPPED_SLOTS)
        - {slot}
        - ({needed_by[slot]} if slot in needed_by else set())
    )
    monkeypatch.setattr(runtime, "SLOTS", frozenset(rest))
    assert runtime.slot_reasons() == []
