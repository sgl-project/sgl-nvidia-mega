# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The draft top-1 as SGLang's argmax partials (CPU, the kernel launch faked): the
distributed argmax gets the partials row pointer as its 7th argument (0 for the
one-hot path), the partials are [rows, 4] fp32, and draft_logits asks for them only
under SGLang's draft_argmax_only flag; the lazy zero allocator makes no buffer until
its first allocate."""

import ctypes

import pytest
import torch

pytest.importorskip("cutlass")

from sgl_nvidia_mega.glm5_3 import draft_step as ds  # noqa: E402
from sgl_nvidia_mega.glm5_3.runtime import LazyZeroAllocator  # noqa: E402


class _Kernel:
    max_rows = 8

    def check_shapes(self, rows, vloc, n_valid, world, rank):
        assert rows <= self.max_rows


def _draft(monkeypatch, calls):
    fd = object.__new__(ds.FusedDraft)
    fd.world, fd.rank = 8, 0
    fd.top1_timeout_ns = 1
    fd.error_record = None
    mailbox = torch.zeros(4, dtype=torch.int64)
    mailbox.peers = torch.zeros(8, dtype=torch.int64)
    ctrl = torch.zeros(4, dtype=torch.int32)

    def fn(*args):
        calls.append(args)
        # the kernel's ids output: token 0 on every row
        ctypes.memset(_ptr(args[4]), 0, 8 * int(_ptr(args[12])))

    fd._top1 = {name: (_Kernel(), fn, mailbox, ctrl) for name in ("decode", "extend")}
    monkeypatch.setattr(ds.FusedDraft, "engaged", lambda self: True)
    monkeypatch.setattr(ds, "cu_stream", lambda device=None: 0)
    return fd


def _ptr(arg) -> int:
    try:
        return int(arg)
    except TypeError:
        return int(arg.value)


def test_partials_pointer_and_shape(monkeypatch):
    calls = []
    fd = _draft(monkeypatch, calls)
    shard = torch.zeros(2, 19360, dtype=torch.bfloat16)
    partials = fd.top1_partials(shard, 19360, extend=False)
    assert partials.shape == (2, ds.PARTIAL_LANES) and partials.dtype == torch.float32
    assert _ptr(calls[-1][6]) == partials.data_ptr()
    one_hot = fd.top1_one_hot(shard, 19360, 154880, extend=True)
    assert one_hot.shape == (2, 154880)
    assert _ptr(calls[-1][6]) == 0


def test_partials_requested_follows_sglang_flag():
    class LP:
        def __init__(self, on):
            self.on = on

        def _draft_argmax_requested(self):
            return self.on

    assert ds._argmax_partials_requested(LP(True))
    assert not ds._argmax_partials_requested(LP(False))
    assert not ds._argmax_partials_requested(object())


def test_lazy_zero_allocator():
    z = LazyZeroAllocator(buffer_size=3, dtype=torch.float32, device="cpu")
    assert z._buffer is None
    a, b = z.allocate(2), z.allocate(1)
    assert a.tolist() == [0.0, 0.0] and b.tolist() == [0.0]
    with pytest.raises(AssertionError):
        z.allocate(1)
