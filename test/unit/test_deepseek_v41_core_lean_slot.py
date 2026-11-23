# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot attn_core_lean on CPU: default-on (a default start takes it on SGLang's v41_new
KV layout only, test_deepseek_v41_default_slots.py), and it needs attn, swapab_h8 and
wo_a_g1."""

from sgl_nvidia_mega.deepseek_v41 import runtime


def test_the_slot_is_default_on():
    assert "attn_core_lean" in runtime.DEFAULT_ON_SLOTS
    assert "attn_core_lean" in runtime.NUMERICS_CHANGE_SLOTS
    assert "attn_core_lean" not in runtime.OPT_IN_SLOTS
    assert runtime.CORE_LEAN_KV_LAYOUT == "v41_new"


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
