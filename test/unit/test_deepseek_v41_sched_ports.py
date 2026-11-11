# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slots attn_compressor_priority, attn_index_early_trigger and attn_defer_block_select on
CPU: their gating, and where each moves a launch or a trigger (the kernels and their
inputs do not change)."""

import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41.attention import chain, indexer, prepare

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_deepseek_v41_attention_indexer import (  # noqa: E402
    _fb,
    _ops_modules,
    _paged,
    _plugin,
    _Stream,
    _view,
)

sys.path.pop(0)

NEW = (
    "attn_compressor_priority",
    "attn_index_early_trigger",
    "attn_defer_block_select",
)


def test_the_slots_are_opt_in_but_compressor_priority():
    from sgl_nvidia_mega.deepseek_v41 import runtime

    assert "attn_compressor_priority" in runtime.DEFAULT_ON_SLOTS
    assert "attn_compressor_priority" in runtime.SHIPPED_SLOTS
    for slot in NEW[1:]:
        assert slot in runtime.OPT_IN_SLOTS
        assert slot not in runtime.SHIPPED_SLOTS


@pytest.mark.parametrize(
    "slots, needs",
    [
        ({"attn_compressor_priority"}, "attn"),
        ({"attn", "attn_indexer", "attn_indexer_fused", "attn_index_early_trigger"}, "kv_before_wait"),
        ({"attn", "attn_indexer", "kv_before_wait", "attn_index_early_trigger"}, "attn_indexer_fused"),
        ({"attn", "attn_defer_block_select"}, "attn_indexer"),
    ],
)  # fmt: skip
def test_each_slot_needs_its_parents(monkeypatch, slots, needs):
    from sgl_nvidia_mega.deepseek_v41 import runtime

    monkeypatch.setattr(runtime, "SLOTS", frozenset(slots))
    reasons = runtime.slot_reasons()
    assert any(f"needs slot {needs!r}" in r for r in reasons), reasons


def test_the_slots_with_their_parents_are_served(monkeypatch):
    from sgl_nvidia_mega.deepseek_v41 import runtime

    monkeypatch.setattr(
        runtime,
        "SLOTS",
        frozenset(
            {"attn", "attn_indexer", "attn_indexer_fused", "kv_before_wait"} | set(NEW)
        ),
    )
    assert runtime.slot_reasons() == []


def _source_attn(layer_id=20):
    return SimpleNamespace(
        compress_ratio=1,
        layer_id=layer_id,
        indexer=SimpleNamespace(is_candidate_source=True, uses_candidates=False),
    )


def _patches():
    return (
        mock.patch(
            "sglang.srt.model_executor.runner_utils.capture_mode.skip_low_ratio_indexer",
            return_value=False,
        ),
        mock.patch(
            "sglang.srt.model_executor.runner_utils.capture_mode.skip_candidate_filtering",
            return_value=False,
        ),
        mock.patch.object(torch.cuda, "current_stream"),
        mock.patch.object(torch.cuda, "stream", lambda s: nullcontext()),
        mock.patch.object(torch.cuda, "Event"),
        mock.patch.object(torch.Tensor, "record_stream", lambda self, s: None),
    )


def _query():
    return indexer.Query(
        torch.zeros(6, 1, 4, 64), torch.zeros(6, 1, 4), torch.zeros(6, 4)
    )


@pytest.mark.parametrize("defer", (False, True))
def test_the_block_selection_waits_for_the_flush(defer):
    """Slot off: the source issues its block selection with its top-512. On: only the
    top-512; the block selection (the same four ops) at ``flush_blocks`` of that layer,
    which another layer's flush does not issue; a new forward drops an unflushed one."""
    log = []
    p = _plugin()
    p.defer_blocks = defer
    view = _view(_paged())
    blocks = ["row_lens", "blocks", "sort", "schedule"]
    p1, p2, p3, p4, p5, p6 = _patches()
    with _ops_modules(log), p1, p2, p3, p4, p5, p6:
        p.stream = mock.Mock()
        p.serves(view, 6, _fb())
        p.topk(_source_attn(), _query(), torch.zeros(6, dtype=torch.int64), view)
        assert log == ["k_cache", "paged_logits", "topk_paged"] + (
            [] if defer else blocks
        )
        assert (p._table is None) is defer
        p.flush_blocks(24)  # a consumer layer
        assert (p._table is None) is defer
        p.flush_blocks(20)
        assert log == ["k_cache", "paged_logits", "topk_paged"] + blocks
        assert p._table is not None and p._pending is None
        # flushed once: a second flush issues nothing
        p.flush_blocks(20)
        assert log.count("schedule") == 1
        if defer:
            p.topk(_source_attn(), _query(), torch.zeros(6, dtype=torch.int64), view)
            assert p._pending is not None
            p.serves(_view(_paged()), 6, _fb())
            assert p._pending is None and p._table is None


def test_the_deferred_selection_waits_for_the_attention_stream():
    """The side stream joins the stream current at the flush (the attention's), and
    the block selection then runs on the side stream."""
    p = _plugin()
    p.defer_blocks = True
    p.stream = mock.Mock()
    view = _view(_paged())
    log = []
    p1, p2, p3, p4, p5, p6 = _patches()
    with _ops_modules(log), p1, p2, p4, p5, p6:
        main_at_topk, main_at_flush = mock.Mock(name="sources"), mock.Mock(name="attn")
        with mock.patch.object(torch.cuda, "current_stream", return_value=main_at_topk):
            p.serves(view, 6, _fb())
            p.topk(_source_attn(), _query(), torch.zeros(6, dtype=torch.int64), view)
        p.stream.wait_stream.assert_not_called()
        with mock.patch.object(
            torch.cuda, "current_stream", return_value=main_at_flush
        ):
            p.flush_blocks(20)
    p.stream.wait_stream.assert_called_once_with(main_at_flush)


@pytest.mark.parametrize("early", (False, True))
def test_the_early_trigger_moves_the_fused_consumers_trigger(early):
    p = _plugin()
    p.topk_blocks, p.fused, p.early_trigger = 2048, True, early
    p._table = indexer.Table(torch.zeros(1, 4), "s", "pb", "vl", "ready")
    attn = SimpleNamespace(
        compress_ratio=1,
        layer_id=9,
        indexer=SimpleNamespace(is_candidate_source=False, uses_candidates=True),
    )
    seen = {}
    p1, p2, p3, p4, p5, p6 = _patches()
    with (
        _ops_modules([]),
        p1,
        p2,
        p3,
        (
            mock.patch.object(
                sys.modules["sglang.srt.layers.attention.dsv4.v41_indexer.scoring"],
                "get_index_k_cache",
                return_value=torch.zeros(1, 64, 1, 68, dtype=torch.uint8),
            )
        ),
        mock.patch.object(indexer.reindex_select, "reason", return_value=""),
        (
            mock.patch.object(
                indexer.reindex_select,
                "reindex_select",
                side_effect=lambda *a, **k: seen.update(k),
            )
        ),
    ):
        p.topk(attn, _query(), None, _view(_paged()))
    assert seen["trigger_after_wait"] is early
    assert seen["inputs_before_wait"] is True


@pytest.mark.parametrize("defer_flag", (False, True))
def test_the_step_flushes_after_the_attention(defer_flag):
    import importlib

    step_mod = importlib.import_module("sgl_nvidia_mega.deepseek_v41.attention.step")
    order = []
    attn = SimpleNamespace(
        layer_id=20, compress_ratio=1, indexer=object(), alt_streams=None
    )
    st = step_mod.AttentionStep.__new__(step_mod.AttentionStep)
    st.attn, st.kv_before_wait, st.index_after_wait = attn, True, True
    st.fused_wo_a, st.cluster_merge, st.freqs = False, False, None
    st.wo_a_fp8_weight = None
    plugin = SimpleNamespace(
        flush_blocks=lambda lid: order.append(("flush", lid)),
    )
    st.chain = (SimpleNamespace(indexer=plugin), SimpleNamespace(indexer=plugin))
    attn.maybe_use_decode_attn_tp = lambda fb: nullcontext()
    view = SimpleNamespace(kv_pool=SimpleNamespace(uniform_fp8=False))
    with (
        mock.patch.object(prepare, "low_ratio", return_value="q"),
        mock.patch.object(st, "takes_swapab", return_value=True),
        mock.patch.object(
            st, "_prepare_path", side_effect=lambda *a: prepare.low_ratio
        ),
        mock.patch.object(
            chain,
            "sparse_decode",
            side_effect=lambda *a, **k: order.append("attention"),
        ),
        mock.patch.object(
            chain, "wo_a", side_effect=lambda *a, **k: order.append("wo_a")
        ),
        mock.patch.object(step_mod.ops, "wo_b", side_effect=lambda *a, **k: "out"),
    ):
        st.run(torch.zeros(6, 4), None, None, view=view)
    assert order == ["attention", ("flush", 20), "wo_a"]


def _low_ratio_with_compressor(hi):
    log = []
    current = _Stream("main", log)
    streams = [_Stream(n, log) for n in ("kv", "query", "sources")]
    attn = SimpleNamespace(
        alt_streams=streams,
        compressor=object(),
        indexer=SimpleNamespace(),
        fuse_wqa_wkv=False,
        compress_ratio=2,
    )
    stream_of = {"cur": current}
    q_ops = SimpleNamespace(
        compressor_stream=_Stream("hi", log) if hi else None,
        compute_q_a=lambda attn, x, qkv_a: log.append("q_a") or ("q_lora", "q"),
        compute_q_b=lambda *a: log.append("q_b") or "q",
        index_q_input=lambda attn, q_lora, q: "index_input",
        kv_to_cache=lambda *a: log.append("kv"),
    )
    view = SimpleNamespace(
        metadata=SimpleNamespace(),
        low_ratio_sources=lambda **kw: log.append(
            (
                "compressor" if not kw.get("run_indexer", True) else "indexer",
                stream_of["cur"].name,
            )
        ),
    )

    def stream_ctx(s):
        prev = stream_of["cur"]
        stream_of["cur"] = s

        class Ctx:
            def __enter__(self):
                return s

            def __exit__(self, *a):
                stream_of["cur"] = prev

        return Ctx()

    with (
        mock.patch.object(torch.cuda, "current_stream", lambda: stream_of["cur"]),
        mock.patch.object(torch.cuda, "stream", stream_ctx),
    ):
        prepare.low_ratio(
            attn, torch.zeros(6, 4), torch.zeros(6), _fb(), view, None, None, q_ops
        )
    return log


@pytest.mark.parametrize("hi", (False, True))
def test_the_compressor_runs_on_the_priority_stream(hi):
    log = _low_ratio_with_compressor(hi)
    comp = "hi" if hi else "sources"
    assert ("compressor", comp) in log and ("indexer", "sources") in log
    # forked from the layer's input, before the compressor
    assert log.index(("wait", comp, "main")) < log.index(("compressor", comp))
    if hi:
        # the indexer (it reads the compressor's index keys) joins the priority stream
        assert (
            log.index(("compressor", "hi"))
            < log.index(("wait", "sources", "hi"))
            < log.index(("indexer", "sources"))
        )
    # the main stream joins the sources stream (and through it the compressor) at the end
    assert ("wait", "main", "sources") in log
