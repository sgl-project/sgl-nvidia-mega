# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot draft_shared_ll on CPU: an opt-in draft slot needing draft_moe and shared_ll, stated
in the startup summary from the draft MoE build's shared experts. The kernels are slot
shared_ll's, tested in test/kernels/test_deepseek_v41_shared_fold.py."""

from types import SimpleNamespace

import pytest

from sgl_nvidia_mega.deepseek_v41 import runtime

pytest.importorskip("sglang.srt.models.deepseek_v4_mhc")

from sgl_nvidia_mega.deepseek_v41 import draft_slots  # noqa: E402

SLOT = "draft_shared_ll"


def test_the_slot_is_an_opt_in_draft_slot():
    assert SLOT in runtime.OPT_IN_SLOTS
    assert SLOT in runtime.DRAFT_SLOTS
    assert SLOT not in runtime.SHIPPED_SLOTS


@pytest.mark.parametrize("needs", ["draft_moe", "shared_ll"])
def test_the_slot_needs_its_prerequisites(monkeypatch, needs):
    full = {SLOT, "draft", "draft_moe", "moe", "shared_ll"}
    monkeypatch.setattr(runtime, "SLOTS", frozenset(full - {needs}))
    assert any(f"'{SLOT}' needs slot '{needs}'" in r for r in runtime.slot_reasons())
    monkeypatch.setattr(runtime, "SLOTS", frozenset(full))
    assert not any(SLOT in r for r in runtime.slot_reasons())


def _built(slots, shared_reason=""):
    return SimpleNamespace(
        draft_sampler=None,
        low_m=False,
        widths=(5,),
        _view=SimpleNamespace(model=SimpleNamespace(layers=[])),
        _bound={"moe": SimpleNamespace(slots=slots, shared_reason=shared_reason)},
    )


def test_the_summary_states_the_slot():
    on = {
        i: SimpleNamespace(shared=SimpleNamespace(fold=object()), gate=False)
        for i in (43, 44)
    }
    got = draft_slots.states(_built(on), frozenset({"draft", SLOT}))[SLOT]
    assert got[0] == draft_slots.BUILT and "[43, 44]" in got[1] and "folded" in got[1]
    off = {43: SimpleNamespace(shared=None, gate=False)}
    got = draft_slots.states(_built(off, "layer 43: x"), frozenset({"draft", SLOT}))
    assert got[SLOT] == (draft_slots.DECLINED, "layer 43: x")
