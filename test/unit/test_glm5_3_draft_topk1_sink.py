# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""SGLang's draft top-1 sink written by the distributed argmax (CPU, the launch faked):
the pointer arguments of the direct outputs, the partials path without them, and the
sink's layout check."""

import ctypes
from dataclasses import dataclass

import pytest
import torch

pytest.importorskip("cutlass")

from sgl_nvidia_mega.glm5_3 import draft_step as ds  # noqa: E402


@dataclass
class _Sink:  # SGLang's DraftTopk1Sink fields
    topk_p: torch.Tensor
    topk_index: torch.Tensor
    positions: torch.Tensor
    draft_tokens: torch.Tensor
    column: int
    filled: bool = False


def _sink(rows=1, steps=5, column=2):
    return _Sink(
        topk_p=torch.empty(rows, 1, dtype=torch.float32),
        topk_index=torch.empty(rows, 1, dtype=torch.int64),
        positions=torch.arange(rows, dtype=torch.int64) + 100,
        draft_tokens=torch.zeros(rows, steps, dtype=torch.int64),
        column=column,
    )


class _Kernel:
    max_rows = 8

    def check_shapes(self, rows, vloc, n_valid, world, rank):
        assert rows <= self.max_rows


def _ptr(arg) -> int:
    try:
        return int(arg)
    except TypeError:
        return int(arg.value)


def _draft(monkeypatch, calls):
    fd = object.__new__(ds.FusedDraft)
    fd.world, fd.rank = 8, 0
    fd.top1_timeout_ns = 1
    fd.error_record = None
    fd._sink_logged = False
    mailbox = torch.zeros(4, dtype=torch.int64)
    mailbox.peers = torch.zeros(8, dtype=torch.int64)
    ctrl = torch.zeros(4, dtype=torch.int32)

    def fn(*args):
        calls.append(args)
        rows = _ptr(args[12])
        ctypes.memset(_ptr(args[4]), 0, 8 * rows)  # ids: token 0

    fd._top1 = {name: (_Kernel(), fn, mailbox, ctrl) for name in ("decode", "extend")}
    monkeypatch.setattr(ds.FusedDraft, "engaged", lambda self: True)
    monkeypatch.setattr(ds, "cu_stream", lambda device=None: 0)
    return fd


def test_sink_pointers_and_fill(monkeypatch):
    calls = []
    fd = _draft(monkeypatch, calls)
    shard = torch.zeros(1, 19360, dtype=torch.bfloat16)
    sink = _sink()
    assert fd.top1_sink(shard, 19360, sink)
    assert sink.filled
    a = calls[-1]
    assert _ptr(a[4]) == sink.topk_index.data_ptr()  # ids are the sink's topk_index
    assert _ptr(a[6]) == 0  # no partials
    assert _ptr(a[7]) == sink.topk_p.data_ptr()
    assert _ptr(a[8]) == sink.draft_tokens.data_ptr()
    assert _ptr(a[9]) == sink.draft_tokens.stride(0) and _ptr(a[10]) == sink.column
    assert _ptr(a[11]) == sink.positions.data_ptr()
    assert _ptr(a[12]) == 1  # rows


def test_partials_path_has_no_direct_outputs(monkeypatch):
    calls = []
    fd = _draft(monkeypatch, calls)
    shard = torch.zeros(2, 19360, dtype=torch.bfloat16)
    partials = fd.top1_partials(shard, 19360, extend=False)
    a = calls[-1]
    assert _ptr(a[6]) == partials.data_ptr()
    assert [_ptr(a[i]) for i in (7, 8, 11)] == [0, 0, 0]


@pytest.mark.parametrize(
    "field, bad",
    [
        ("topk_index", torch.empty(1, 1, dtype=torch.int32)),
        ("topk_p", torch.empty(2, 1, dtype=torch.float32)),
        ("positions", torch.empty(1, dtype=torch.int32)),
        ("draft_tokens", torch.zeros(1, 5, dtype=torch.int32)),
    ],
)
def test_sink_layout_rejected(monkeypatch, field, bad):
    calls = []
    fd = _draft(monkeypatch, calls)
    sink = _sink()
    setattr(sink, field, bad)
    assert ds.sink_reason(sink, 1)
    assert not fd.top1_sink(torch.zeros(1, 19360, dtype=torch.bfloat16), 19360, sink)
    assert not sink.filled and not calls


def test_sink_column_bound():
    sink = _sink(column=5)
    assert ds.sink_reason(sink, 1)
    assert ds.sink_reason(_sink(column=4), 1) == ""
