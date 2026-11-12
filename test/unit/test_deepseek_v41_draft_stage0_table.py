# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot draft_stage0_table on CPU: an opt-in draft slot that needs draft_swapab, built only
where the first stage's attention is ported with swap-AB, stated in the startup summary;
the boundary opens a sublayer without its input only for the first attention of fresh
streams; a step takes the table only at a checked width where swap-AB takes it. The gather,
fill and check run on a GPU in test/kernels/test_deepseek_v41_draft_stage0_table.py."""

from types import SimpleNamespace
from unittest import mock

import pytest

from sgl_nvidia_mega.deepseek_v41 import runtime

pytest.importorskip("sglang.srt.models.deepseek_v4_mhc")

from sgl_nvidia_mega.deepseek_v41 import (  # noqa: E402
    draft_attention,
    draft_slots,
    draft_step,
)
from sgl_nvidia_mega.deepseek_v41.boundary import boundaries  # noqa: E402

SLOT = "draft_stage0_table"


def test_the_slot_is_an_opt_in_draft_slot():
    assert SLOT in runtime.OPT_IN_SLOTS
    assert SLOT in runtime.DRAFT_SLOTS
    assert SLOT not in runtime.SHIPPED_SLOTS


def test_the_slot_needs_draft_swapab(monkeypatch):
    monkeypatch.setattr(runtime, "SLOTS", frozenset({SLOT}))
    assert any(
        f"'{SLOT}' needs slot 'draft_swapab'" in r for r in runtime.slot_reasons()
    )
    monkeypatch.setattr(
        runtime, "SLOTS", frozenset({SLOT, "draft_swapab", "draft_attn", "draft"})
    )
    assert not any(SLOT in r for r in runtime.slot_reasons())


def _stages():
    return [SimpleNamespace(layer_id=i) for i in range(3)]


def test_reason_names_what_is_missing():
    stages = _stages()
    assert "not ported" in draft_step._stage0_reason(stages, {})
    no_swapab = {0: draft_attention.Kernels(swapab=False)}
    assert "swap-AB" in draft_step._stage0_reason(stages, no_swapab)
    assert (
        draft_step._stage0_reason(stages, {0: draft_attention.Kernels(swapab=True)})
        == ""
    )


def _built(wanted, reason=""):
    return SimpleNamespace(
        draft_sampler=None,
        low_m=False,
        widths=(5,),
        _view=SimpleNamespace(model=SimpleNamespace(layers=[])),
        _stage0_wanted=wanted,
        _stage0_reason=reason,
    )


def test_the_summary_states_the_slot():
    slots = frozenset({"draft", SLOT})
    assert draft_slots.states(_built(True), slots)[SLOT][0] == draft_slots.BUILT
    got = draft_slots.states(_built(False, "no swap-AB"), slots)[SLOT]
    assert got == (draft_slots.DECLINED, "no swap-AB")


class _State:
    def __init__(self, pre=None, input=None):
        self.residual, self.pre, self.input, self.stats = "streams", pre, input, None

    def release(self):
        pass


def _boundary():
    b = boundaries.Boundaries.__new__(boundaries.Boundaries)
    combined = []
    b._mhc = SimpleNamespace(
        use_stats_stream=lambda *a: False,
        fork_stats_stream=lambda s: None,
        combine=lambda hc, state, q: combined.append(hc) or "x",
    )
    b._attn_mxfp8 = {0: False}
    return b, combined


def test_pre_without_input_opens_a_fresh_attention_only():
    layer = SimpleNamespace(layer_id=0, hc_cfg=None, attn_hc="attn_hc", ffn_hc="ffn_hc")
    b, combined = _boundary()
    st = boundaries.BoundaryState(hc=_State())
    assert b.pre(layer, "attn", st, None, need_input=False) == (None, None)
    assert combined == [] and st.open is not None
    # a carried pre-mix, a built input, or the FFN: the input is needed
    for state, kind in ((_State(pre="p"), "attn"), (_State(input="i"), "attn")):
        with pytest.raises(RuntimeError):
            b.pre(
                layer, kind, boundaries.BoundaryState(hc=state), None, need_input=False
            )
    with pytest.raises(RuntimeError):
        b.pre(
            layer, "ffn", boundaries.BoundaryState(hc=_State()), None, need_input=False
        )
    assert b.pre(layer, "attn", boundaries.BoundaryState(hc=_State()), None) == (
        "x",
        None,
    )
    assert combined == ["attn_hc"]


def _decode(table, swapab_takes=True):
    d = draft_step.DraftDecode.__new__(draft_step.DraftDecode)
    import contextlib

    stage = SimpleNamespace(
        layer_id=0,
        self_attn=SimpleNamespace(
            maybe_use_decode_attn_tp=lambda fb: contextlib.nullcontext()
        ),
    )
    d.lm = SimpleNamespace(stages=[stage])
    d._attn = {0: draft_attention.Kernels(swapab=True)}
    d._stage0_wanted, d._stage0, d._stage0_logged = True, table, set()
    return d, mock.patch.object(
        draft_attention, "_swapab_takes", return_value=swapab_takes
    )


def test_a_step_takes_the_table_at_checked_widths_only():
    table = SimpleNamespace(checked=(5,))
    fb = SimpleNamespace(forward_mode=SimpleNamespace(is_idle=lambda: False))
    d, takes = _decode(table)
    with takes:
        assert d._stage0_table(5, None, fb, None) is table
        assert d._stage0_table(10, None, fb, None) is None
    d, declines = _decode(table, swapab_takes=False)
    with declines:
        assert d._stage0_table(5, None, fb, None) is None
    d._stage0_wanted = False
    assert d._stage0_table(5, None, fb, None) is None


def test_no_fill_inside_a_capture():
    fb = SimpleNamespace(forward_mode=SimpleNamespace(is_idle=lambda: False))
    d, takes = _decode(None)
    with (
        takes,
        mock.patch("torch.cuda.is_current_stream_capturing", return_value=True),
        mock.patch.object(d, "_fill_stage0") as fill,
    ):
        assert d._stage0_table(5, None, fb, None) is None
    fill.assert_not_called()
