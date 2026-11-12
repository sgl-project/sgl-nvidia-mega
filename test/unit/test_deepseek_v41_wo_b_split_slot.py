# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot wo_b_publish_split on CPU: a known slot that needs hop_a. The kernel is tested in
test/kernels/test_deepseek_v41_wo_b_split.py (world 8) and test_deepseek_v41_fail_stop.py."""

from sgl_nvidia_mega.deepseek_v41 import runtime

SLOT = "wo_b_publish_split"


def test_the_slot_is_known():
    assert SLOT in (*runtime.SHIPPED_SLOTS, *runtime.OPT_IN_SLOTS)


def test_the_slot_needs_hop_a(monkeypatch):
    monkeypatch.setattr(runtime, "SLOTS", frozenset({SLOT}))
    assert any(f"'{SLOT}' needs slot 'hop_a'" in r for r in runtime.slot_reasons())
    monkeypatch.setattr(runtime, "SLOTS", frozenset({SLOT, "hop_a"}))
    assert not any(SLOT in r for r in runtime.slot_reasons())
