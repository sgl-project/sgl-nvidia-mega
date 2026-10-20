# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Hop B at 9-48 rows (slot hop_b_wide): it writes the next input's scales per 8-row tile only for a next
layer whose chain runs wqkv_a on the wide low-latency GEMM; SGLang's wqkv_a never
receives them."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41 import plan
from sgl_nvidia_mega.deepseek_v41.attention import chain, ops
from sgl_nvidia_mega.deepseek_v41.boundary import boundaries
from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll

HIDDEN = 5120


def _rows(rows):
    return mxfp8_ll.RowTileMxfp8(
        torch.zeros(rows, HIDDEN, dtype=torch.uint8),
        torch.zeros(mxfp8_ll.scale_bytes(rows, HIDDEN), dtype=torch.uint8),
    )


def test_hop_b_spans_batch_2_to_8_below_the_two_shot():
    assert plan.HOP_A_MAX_ROWS < plan.HOP_B_MAX_ROWS < plan.TWO_SHOT_MIN_ROWS
    assert plan.HOP_B_MAX_ROWS == plan.ROUTED_MAX_ROWS


@pytest.mark.parametrize("hop_a_rows", [8, 16])
def test_hop_b_goes_past_hop_a_only_with_the_slot(hop_a_rows):
    assert plan.hop_b_max_rows(False, hop_a_rows) == hop_a_rows
    assert plan.hop_b_max_rows(True, hop_a_rows) == plan.HOP_B_MAX_ROWS


@pytest.mark.parametrize(
    "slots, wide",
    [({"hop_a", "hop_b", "hop_b_wide"}, True), ({"hop_a", "hop_b"}, False)],
)
def test_wide_wqkv_a_compiles_only_with_hop_b_wide(monkeypatch, slots, wide):
    from sgl_nvidia_mega.deepseek_v41 import runtime

    calls = []
    monkeypatch.setattr(runtime, "SLOTS", frozenset(slots))
    monkeypatch.setattr(chain, "qkv_a_ll_applies", lambda attn: True)
    monkeypatch.setattr(chain, "wq_b_ll_applies", lambda attn: False)
    monkeypatch.setattr(
        chain.mxfp8_ll, "precompile", lambda w, rows: calls.append(rows)
    )
    attn = SimpleNamespace(wqkv_a=SimpleNamespace(weight="w"), use_fused_wo_a=False)
    chain.precompile(attn, (6, 12, 48, 96), qkv_a_ll=True)
    assert calls == [[6, 12, 48] if wide else [6]]


def test_row_tile_layers_only_with_hop_b_wide():
    from sgl_nvidia_mega.deepseek_v41 import runtime

    ll, dense = (SimpleNamespace(qkv_a_ll=q) for q in (True, False))
    attention = {
        3: SimpleNamespace(chain=(ll,)),
        4: SimpleNamespace(chain=(dense,)),
        5: SimpleNamespace(chain=None),
    }
    assert runtime.row_tile_layers(attention, frozenset({"hop_b_wide"})) == {3}
    assert runtime.row_tile_layers(attention, frozenset({"hop_b"})) == frozenset()


def test_row_tiles_only_past_8_rows_for_a_reading_chain():
    b = boundaries.Boundaries.__new__(boundaries.Boundaries)
    b._next_layer, b.row_tile_layers = {3: 4, 4: 5}, frozenset({4})
    assert b._row_tiles(3, 12) and b._row_tiles(3, 48)
    assert not b._row_tiles(3, 8)
    assert not b._row_tiles(4, 12)  # layer 5's chain does not read them
    assert not b._row_tiles(39, 12)  # the last layer has no next


def test_the_quantized_rows_carry_their_layout():
    q, sf = torch.zeros(12, 4), torch.zeros(8)
    with mock.patch("sglang.srt.models.deepseek_v4_mhc.HcQuantized", lambda r, s: s):
        assert isinstance(
            boundaries._quantized(None, q, sf, True), mxfp8_ll.RowTileMxfp8
        )
        assert not isinstance(boundaries._quantized(None, q, sf), mxfp8_ll.RowTileMxfp8)


@pytest.mark.parametrize("rows", [12, 48])
def test_the_chain_runs_row_tile_rows_on_the_wide_gemm(rows):
    attn = SimpleNamespace(wqkv_a=mock.Mock(return_value=("sglang", None)), layer_id=0)
    attn.wqkv_a.weight, attn.wqkv_a.weight_scale_inv_swizzled = "w", "w_sf"
    cops = chain.ChainOps.__new__(chain.ChainOps)
    cops.qkv_a_ll, cops.qkv_a_bf16 = True, False
    x = _rows(rows)
    with mock.patch.object(chain.mxfp8_ll, "linear", return_value="ll") as ll:
        assert cops.qkv_a(attn, x) == "ll"
    assert ll.call_args.args == (x.quant, x.scale, "w", "w_sf")
    attn.wqkv_a.assert_not_called()


def test_sglangs_wqkv_a_never_takes_row_tile_scales():
    attn = SimpleNamespace(wqkv_a=mock.Mock(return_value=("sglang", None)))
    with pytest.raises(RuntimeError, match="row-tile MXFP8 scales"):
        ops.qkv_a(attn, _rows(12))
    attn.wqkv_a.assert_not_called()


WIDTHS = (6, 12, 18, 24, 48)


def test_hop_layout_keeps_hop_a_widths_on_the_small_mailbox():
    off = plan.hop_layout(WIDTHS, hop_b=True, batch2=True, hop_b_wide=False)
    assert (off.hop_a_rows, off.hop_b_rows) == ((6, 12), (6, 12))
    assert (off.capacity, off.wide_capacity) == (16, 0)
    assert (
        not off.row_tiles and not off.wide_row_tiles
    )  # no row-tile kernel without the slot
    on = plan.hop_layout(WIDTHS, hop_b=True, batch2=True, hop_b_wide=True)
    assert (on.hop_a_rows, on.hop_b_rows) == ((6, 12), WIDTHS)
    assert (on.capacity, on.wide_capacity) == (16, 48)
    assert (
        on.row_tiles and on.wide_row_tiles
    )  # 12 rows on the small one, 18-48 on the wide
    b1 = plan.hop_layout(WIDTHS, hop_b=True, batch2=False, hop_b_wide=True)
    assert (b1.capacity, b1.wide_capacity, b1.row_tiles) == (8, 48, False)


def test_hop_b_cap16_builds_no_wide_mailbox_and_keeps_the_12_row_tiles():
    """Slot hop_b_cap16: hop B only at hop A's widths, so the 48-row mailbox (+16 us on
    the batch-1 verify graph) is never built; 12 rows keep their row-tile scales."""
    cap = plan.hop_layout(WIDTHS, hop_b=True, batch2=True, hop_b_wide=True, cap16=True)
    assert (cap.hop_a_rows, cap.hop_b_rows) == ((6, 12), (6, 12))
    assert (cap.capacity, cap.wide_capacity, cap.wide_row_tiles) == (16, 0, False)
    assert cap.row_tiles


def test_batch_1_and_2_hops_touch_only_the_small_mailbox():
    """Hop B at hop A's widths runs on hop A's 16-row mailbox; only past it on the
    48-row one (a wider mailbox measured +11.6 us/step at batch 1)."""
    from sgl_nvidia_mega.deepseek_v41.boundary.hop_a import HopA
    from sgl_nvidia_mega.deepseek_v41.sm103.runner import Runner

    calls = []
    runner = Runner.__new__(Runner)
    runner.hop = SimpleNamespace(
        capacity=16, finalize=lambda *a, **k: calls.append("small")
    )
    runner.hop_wide = SimpleNamespace(
        capacity=48, finalize=lambda *a, **k: calls.append("wide")
    )
    hop = HopA(runner, (6, 12), WIDTHS)
    for rows in WIDTHS:
        hop.finalize("args", rows=rows)
    assert calls == ["small", "small", "wide", "wide", "wide"]
    runner.hop_wide = None
    assert runner.hop_for(48) is runner.hop
