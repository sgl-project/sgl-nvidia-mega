# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot attn_compressor_priority on CPU: its gating, and the stream its compressor runs
on (the kernels and their inputs do not change)."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41.attention import prepare

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_deepseek_v41_attention_indexer import _fb, _Stream  # noqa: E402

sys.path.pop(0)


def test_the_slot_runs_by_default():
    from sgl_nvidia_mega.deepseek_v41 import runtime

    assert "attn_compressor_priority" in runtime.DEFAULT_ON_SLOTS
    assert "attn_compressor_priority" in runtime.SHIPPED_SLOTS


def test_the_slot_needs_attn(monkeypatch):
    from sgl_nvidia_mega.deepseek_v41 import runtime

    monkeypatch.setattr(runtime, "SLOTS", frozenset({"attn_compressor_priority"}))
    reasons = runtime.slot_reasons()
    assert any("needs slot 'attn'" in r for r in reasons), reasons
    monkeypatch.setattr(
        runtime, "SLOTS", frozenset({"attn", "attn_compressor_priority"})
    )
    assert runtime.slot_reasons() == []


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
