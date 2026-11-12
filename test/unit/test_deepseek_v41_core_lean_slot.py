# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot attn_core_lean on CPU: opt-in (it needs SGLang's v41_new KV layout, which the
plugin does not select), and it needs attn, swapab_h8 and wo_a_g1."""

from sgl_nvidia_mega.deepseek_v41 import runtime


def test_the_slot_is_opt_in():
    assert "attn_core_lean" in runtime.OPT_IN_SLOTS
    assert "attn_core_lean" not in runtime.SHIPPED_SLOTS
    assert "attn_core_staged" not in runtime.OPT_IN_SLOTS


def test_the_slot_needs_the_tp8_attention_slots(monkeypatch):
    base = {"attn", "swapab_h8", "wo_a_g1", "attn_core_lean"}
    monkeypatch.setattr(runtime, "SLOTS", frozenset(base))
    assert runtime.slot_reasons() == []
    for needs in ("attn", "swapab_h8", "wo_a_g1"):
        monkeypatch.setattr(runtime, "SLOTS", frozenset(base - {needs}))
        assert any(
            f"'attn_core_lean' needs slot {needs!r}" in r
            for r in runtime.slot_reasons()
        ), needs
