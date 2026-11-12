# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot wo_b_publish_split on CPU: a known slot that needs hop_a, and on a default start
taken at attention TP8's wo_b shape only. The kernel is tested in
test/kernels/test_deepseek_v41_wo_b_split.py (world 8) and test_deepseek_v41_fail_stop.py."""

from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41 import runtime

SLOT = "wo_b_publish_split"


def test_the_slot_is_known():
    assert SLOT in (*runtime.SHIPPED_SLOTS, *runtime.OPT_IN_SLOTS)


def test_the_slot_needs_hop_a(monkeypatch):
    monkeypatch.setattr(runtime, "SLOTS", frozenset({SLOT}))
    assert any(f"'{SLOT}' needs slot 'hop_a'" in r for r in runtime.slot_reasons())
    monkeypatch.setattr(runtime, "SLOTS", frozenset({SLOT, "hop_a"}))
    assert not any(SLOT in r for r in runtime.slot_reasons())


@pytest.mark.parametrize(
    "listed, inputs, takes",
    [
        (False, 1024, True),  # default start, attention TP8
        (False, 2048, False),  # default start, TP4: left off
        (True, 2048, True),  # listed: every shape the kernel takes
    ],
)
def test_a_default_start_takes_it_at_attention_tp8_only(
    monkeypatch, listed, inputs, takes
):
    from sgl_nvidia_mega.deepseek_v41 import boundary

    if listed:
        monkeypatch.setenv(runtime.SLOTS_ENV, f"hop_a,{SLOT}")
    else:
        monkeypatch.delenv(runtime.SLOTS_ENV, raising=False)
    wo_b = SimpleNamespace(weight=torch.empty(4096, inputs, dtype=torch.float8_e4m3fn))
    m = SimpleNamespace(
        start_layer=0, layers=[SimpleNamespace(self_attn=SimpleNamespace(wo_b=wo_b))]
    )
    assert boundary._wo_b_split_default_takes(m, runtime) is takes
