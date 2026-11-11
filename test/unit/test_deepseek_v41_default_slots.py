# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The default-on slots: shipped, not opt-in, their dependencies shipped too, and the
numerics-changing and unresolved slots still opt-in."""

from sgl_nvidia_mega.deepseek_v41 import runtime


def test_the_default_on_slots_are_shipped_and_not_opt_in():
    on = set(runtime.DEFAULT_ON_SLOTS)
    assert on <= set(runtime.SHIPPED_SLOTS)
    assert not on & set(runtime.OPT_IN_SLOTS)
    assert len(runtime.SHIPPED_SLOTS) == len(set(runtime.SHIPPED_SLOTS)) == 54


def test_the_shipped_list_needs_no_opt_in_slot(monkeypatch):
    monkeypatch.setattr(runtime, "SLOTS", frozenset(runtime.SHIPPED_SLOTS))
    assert runtime.slot_reasons() == []


def test_numerics_changing_and_unresolved_slots_stay_opt_in():
    for slot in (
        "target_lm_head",
        "target_lm_head_mxfp8",
        "hop_final_collapse",
        "attn_q_cluster",
        "swapab_h8",
        "wo_a_g1",
        "attn_cluster_merge",
        "attn_index_q_first",
        "attn_wo_a_fp8",
        "attn_index_early_trigger",
        "attn_defer_block_select",
        "markov_pdl",
        "attn_index_weights_bf16",
    ):
        assert slot in runtime.OPT_IN_SLOTS and slot not in runtime.SHIPPED_SLOTS
