# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The opt-in two-shot next-input handoff, with the slot-off boundary as control."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

mhc = pytest.importorskip("sglang.srt.models.deepseek_v4_mhc")

from sgl_nvidia_mega.deepseek_v41.boundary.boundaries import Boundaries  # noqa: E402

ROWS, HC, HIDDEN = 6, 4, 8
REAL_DEFERS_SUM = Boundaries.defers_sum


def _layer(layer_id, *, engram=None, mxfp8=True):
    norm = SimpleNamespace(weight=torch.ones(HIDDEN), variance_epsilon=1e-6)
    return SimpleNamespace(
        layer_id=layer_id,
        engram=engram,
        boundary_mhc_fusions=True,
        boundary_pre_from_prev=True,
        _can_fuse_attn_mhc=True,
        _can_fuse_ffn_mhc=True,
        hc_stats_stream=object(),
        hc_cfg=SimpleNamespace(hidden=HIDDEN),
        attn_hc=SimpleNamespace(name=f"attn{layer_id}"),
        ffn_hc=SimpleNamespace(name=f"ffn{layer_id}"),
        input_layernorm=SimpleNamespace(norm=norm, name=f"in{layer_id}"),
        post_attention_layernorm=SimpleNamespace(norm=norm, name=f"post{layer_id}"),
        self_attn=SimpleNamespace(
            attn_tp_size=4, accepts_mxfp8_swizzled_input=lambda: mxfp8
        ),
        mlp=SimpleNamespace(tp_size=4),
    )


@pytest.fixture
def fake():
    calls = []
    m = Mock()
    m.HcState = mhc.HcState
    m.AttnOutput = mhc.AttnOutput
    m.use_stats_stream.return_value = True
    m.fork_stats_stream.side_effect = lambda s: calls.append(("fork", s))

    def combine(hc, state, quantized):
        calls.append(("combine", hc.name, quantized is not None))
        if quantized is not None:
            quantized.append("q")
        return "x"

    m.combine.side_effect = combine
    m.run_attn_post.side_effect = lambda *a, **k: mhc.HcState(torch.ones(1), "pre")
    m.run_moe_post.side_effect = lambda *a, **k: mhc.HcState(torch.zeros(1), "pre")
    m.calls = calls
    # the fused collective's widths unless a test says otherwise
    with patch.object(Boundaries, "defers_sum", return_value=True):
        yield m


def _deferred(rows, *, replicated=False):
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        FlashInferTrtllmDeferredFinalizeOutput,
    )
    from sglang.srt.models.deepseek_v2 import MoEOutput

    routed = FlashInferTrtllmDeferredFinalizeOutput(
        torch.ones(rows * 6, HIDDEN), torch.ones(rows, 6), torch.zeros(rows, 6), 6
    )
    return MoEOutput(routed, torch.ones(rows, HIDDEN), None, 1.0, replicated)


def _hop_boundary(fake, rows, **kw):
    """A boundary at ``rows`` with SGLang's triplet and next-input types real."""
    triplet = (torch.ones(rows, HC), torch.ones(rows, HC), torch.ones(rows, HC, HC))
    fake.mix_stats.return_value = triplet
    fake.HcNormed, fake.HcQuantized = mhc.HcNormed, mhc.HcQuantized
    fake.is_deferred_finalize = mhc.is_deferred_finalize
    layers = [_layer(0), _layer(1)]
    with patch("sglang.srt.models.deepseek_v4_mhc", fake):
        b = Boundaries(layers, **kw)
    norm = SimpleNamespace(weight=torch.ones(HIDDEN), variance_epsilon=1e-6)
    b._ffn_next[0] = SimpleNamespace(norm=norm)
    return b, layers, triplet, norm


@pytest.mark.parametrize(
    "receiver,rows",
    [("closed", 64), ("unfusable", 64), ("normed", 64), ("mxfp8", 64), ("mxfp8", 192)],
)
def test_two_shot_runs_both_posts_and_builds_sglangs_next_input(fake, receiver, rows):
    """Both posts take the two-shot hop; the MoE post hands on what SGLang's
    post would: MXFP8 only where its fused quant runs (up to 128 rows), else the
    streams only, for the next combine to collapse."""
    updated, normed = torch.zeros(rows, HC, HIDDEN), torch.ones(rows, HIDDEN)
    hop = SimpleNamespace(
        fits=lambda r, h: r >= 49 and h == HIDDEN,
        attn_post=Mock(return_value=(updated, normed)),
        moe_post=Mock(return_value=(updated, normed)),
        moe_post_only=Mock(return_value=updated),
        scale_bytes=lambda r: 160 * 128,
    )
    with patch.object(Boundaries, "defers_sum", REAL_DEFERS_SUM):
        b, layers, triplet, norm = _hop_boundary(fake, rows, two_shot=hop)
        assert b.defers_sum(layers[0], "attn", rows)
        assert b.defers_sum(layers[0], "ffn", rows)
    pre, post_mix, comb = triplet
    streams = torch.randn(rows, HC, HIDDEN)
    st = b.start(streams)
    b.pre(layers[0], "attn", st, object())
    partial = torch.randn(rows, HIDDEN)
    with (
        patch.object(torch.cuda, "current_stream"),
        patch(
            "sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe.can_fuse_all_reduce",
            side_effect=lambda r, h: r <= 152,
        ),
    ):
        b.post(layers[0], "attn", partial, st)
    assert hop.attn_post.call_args.args == (
        partial,
        streams,
        post_mix,
        comb,
        pre,
        norm.weight,
        1e-6,
    )
    # past SGLang's fused collective its own combine norms the streams
    if rows <= 152:
        assert isinstance(st.hc.input, mhc.HcNormed)
    else:
        assert st.hc.input is None and st.hc.streams is updated
    fake.run_attn_post.assert_not_called()

    b._attn_next[0] = {
        "closed": None,
        "unfusable": SimpleNamespace(norm=norm, norm_fusable=False, accepts_mxfp8=True),
        "normed": SimpleNamespace(norm=norm, norm_fusable=True, accepts_mxfp8=False),
        "mxfp8": SimpleNamespace(norm=norm, norm_fusable=True, accepts_mxfp8=True),
    }[receiver]
    b.pre(layers[0], "ffn", st, object())
    out = _deferred(rows)
    with patch.object(torch.cuda, "current_stream"):
        b.post(layers[0], "ffn", out, st)
    fake.run_moe_post.assert_not_called()
    if receiver != "mxfp8" or rows > 128:
        assert hop.moe_post_only.call_args.args[4] is out.shared
        assert st.hc.input is None
        hop.moe_post.assert_not_called()
    else:
        q, sf = hop.moe_post.call_args.kwargs["mxfp8_out"]
        assert q.shape == (rows, HIDDEN) and sf.numel() == 160 * 128
        assert st.hc.input.pair.data is q


@pytest.mark.parametrize("receiver", ["closed", "unfusable", "normed", "mxfp8"])
@pytest.mark.parametrize("rows", [64, 129, 153, 384])
def test_two_shot_norm_wide_hands_on_its_input_past_sglangs_caps(
    fake, caplog, receiver, rows
):
    """Slot two_shot_norm_wide: past 152 rows the attention post still hands on its
    normed rows, past 128 the MoE post its MXFP8 (or normed) input; up to the caps
    and for a closed / unfusable receiver it builds what the slot-off post does."""
    updated, normed = torch.zeros(rows, HC, HIDDEN), torch.ones(rows, HIDDEN)
    hop = SimpleNamespace(
        fits=lambda r, h: 49 <= r <= 384 and h == HIDDEN,
        attn_post=Mock(return_value=(updated, normed)),
        moe_post=Mock(return_value=(updated, normed)),
        moe_post_only=Mock(return_value=updated),
        scale_bytes=lambda r: 160 * 128 * -(-r // 128),
    )
    b, layers, _, norm = _hop_boundary(
        fake, rows, two_shot=hop, two_shot_norm_wide=True
    )
    st = b.start(torch.randn(rows, HC, HIDDEN))
    b.pre(layers[0], "attn", st, object())
    with (
        patch.object(torch.cuda, "current_stream"),
        patch(
            "sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe.can_fuse_all_reduce",
            side_effect=lambda r, h: r <= 152,
        ),
        caplog.at_level("INFO"),
    ):
        b.post(layers[0], "attn", torch.randn(rows, HIDDEN), st)
    assert isinstance(st.hc.input, mhc.HcNormed) and st.hc.input.rows is normed
    assert ("attention post (normed, wide)" in caplog.text) is (rows > 152)

    b._attn_next[0] = {
        "closed": None,
        "unfusable": SimpleNamespace(norm=norm, norm_fusable=False, accepts_mxfp8=True),
        "normed": SimpleNamespace(norm=norm, norm_fusable=True, accepts_mxfp8=False),
        "mxfp8": SimpleNamespace(norm=norm, norm_fusable=True, accepts_mxfp8=True),
    }[receiver]
    b.pre(layers[0], "ffn", st, object())
    with patch.object(torch.cuda, "current_stream"), caplog.at_level("INFO"):
        b.post(layers[0], "ffn", _deferred(rows), st)
    wide = rows > 128 and receiver in ("normed", "mxfp8")
    if receiver == "mxfp8":
        q, sf = hop.moe_post.call_args.kwargs["mxfp8_out"]
        assert q.shape == (rows, HIDDEN) and sf.numel() == hop.scale_bytes(rows)
        assert st.hc.input.pair.data is q
    elif wide:
        assert "mxfp8_out" not in hop.moe_post.call_args.kwargs
        assert isinstance(st.hc.input, mhc.HcNormed) and st.hc.input.rows is normed
    else:
        hop.moe_post.assert_not_called()
        assert st.hc.input is None
    assert ("MoE post (" in caplog.text) is wide


def test_two_shot_norm_wide_requires_two_shot(monkeypatch):
    from sgl_nvidia_mega.deepseek_v41 import runtime

    assert "two_shot_norm_wide" in runtime.OPT_IN_SLOTS
    assert "two_shot_norm_wide" not in runtime.SHIPPED_SLOTS
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"two_shot_norm_wide"}))
    assert any("needs slot 'two_shot'" in reason for reason in runtime.slot_reasons())
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"two_shot_norm_wide", "two_shot"}))
    assert runtime.slot_reasons() == []
