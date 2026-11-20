# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot attn_wo_a_fp8 on CPU: opt-in, needs attn, and the E4M3 wo_a used only where it
widens back to the BF16 wo_a exactly."""

from types import SimpleNamespace

import torch

from sgl_nvidia_mega.deepseek_v41.attention import chain


def test_the_wo_a_fp8_slot_is_opt_in_and_needs_attn(monkeypatch):
    from sgl_nvidia_mega.deepseek_v41 import runtime

    assert "attn_wo_a_fp8" in runtime.OPT_IN_SLOTS
    assert "attn_wo_a_fp8" not in runtime.SHIPPED_SLOTS
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"attn_wo_a_fp8"}))
    assert any("needs slot 'attn'" in r for r in runtime.slot_reasons())
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"attn", "attn_wo_a_fp8"}))
    assert runtime.slot_reasons() == []


def test_wo_a_fp8_weight_declines_an_inexact_wo_a():
    """The recovered E4M3 form is used only when it widens back to the BF16 wo_a exactly."""
    w = torch.zeros(1024, 4096)
    w[0, 0] = 0.1234  # not an E4M3 value times a power of two
    attn = SimpleNamespace(
        layer_id=5,
        n_local_groups=1,
        o_lora_rank=1024,
        wo_a=SimpleNamespace(weight=w.bfloat16()),
    )
    assert chain.wo_a_fp8_weight(attn) is None
    assert "exact" in attn._mega_wo_a_fp8_reason
    exact = SimpleNamespace(
        layer_id=6,
        n_local_groups=1,
        o_lora_rank=1024,
        wo_a=SimpleNamespace(weight=torch.full((1024, 4096), 0.5).bfloat16()),
    )
    got = chain.wo_a_fp8_weight(exact)
    assert got.codes.shape == (1024, 4096) and got.scales.shape == (32, 128)
    assert chain.wo_a_fp8_weight(exact) is got
