# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Swap-AB attention with a row's tiles merged inside its CTA cluster (slot
attn_cluster_merge) on one SM100-family GPU (GB300): bit for bit the separate narrow
combine at 8 heads (attention TP8), the low-ratio range (window + compressed, padded
indices, short lengths) and the window only, every launch form the chain uses and a
long-context cache; where the policy keeps the combine, the launch is the
current one.
"""

import os
import sys

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41.sm103.attention import swapab

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepseek_v41_kernel_helpers import bitwise  # noqa: E402


def _ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        from triton.experimental import gluon  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _ready(), reason="needs an SM100-family GPU")

HEADS = 8


def _cache(page, pages):
    stride = page * 584 + 256
    raw = torch.full((pages, stride), 0x7F, dtype=torch.uint8, device="cuda")
    fp8 = torch.randn(pages, page, 448, device="cuda").to(torch.float8_e4m3fn)
    exps = torch.randint(118, 134, (pages, page, 7), device="cuda", dtype=torch.uint8)
    rope = (torch.randn(pages, page, 64, device="cuda") * 4).to(torch.bfloat16)
    tokens = raw[:, : page * 576].view(pages, page, 576)
    tokens[..., :448].copy_(fp8.view(torch.uint8))
    tokens[..., 448:].copy_(rope.view(torch.uint8))
    raw[:, page * 576 : page * 584].view(pages, page, 8)[..., :7].copy_(exps)
    return raw.as_strided((pages, page, 1, 584), (stride, 584, 584, 1))


def _inputs(b, seed, extra_tokens=16 * 11, window=True):
    g = torch.Generator(device="cuda").manual_seed(seed)
    kv = _cache(64, 11)
    extra = _cache(16, -(-extra_tokens // 16))
    q = (torch.randn(b, HEADS, 512, device="cuda", generator=g) * 3).to(torch.bfloat16)
    ids = torch.randint(
        0, 64 * 11, (b, 1, 128), device="cuda", dtype=torch.int32, generator=g
    )
    lengths = torch.randint(1, 129, (b,), device="cuda", dtype=torch.int32, generator=g)
    lengths[0] = 128
    ei = torch.randint(
        0, extra_tokens, (b, 1, 512), device="cuda", dtype=torch.int32, generator=g
    )
    ei[:, :, -7:] = -1
    el = torch.randint(1, 513, (b,), device="cuda", dtype=torch.int32, generator=g)
    el[-1] = 512
    sink = torch.randn(HEADS, device="cuda", generator=g) * 4
    angles = torch.rand(70000, 32, device="cuda", generator=g) * 6.3
    freqs = torch.view_as_real(torch.polar(torch.ones_like(angles), angles)).flatten(-2)
    positions = torch.randint(
        70000, (b,), device="cuda", dtype=torch.int64, generator=g
    )
    if not window:
        return q, (kv, ids, lengths, sink), freqs, positions
    return q, (kv, ids, lengths, sink, extra, ei, el), freqs, positions


@pytest.fixture
def launches(monkeypatch):
    """Counts the cluster launches."""
    calls = []
    real = swapab._launch_cluster

    def counted(fn, rows, args, kwargs, tiles, exchange):
        calls.append((rows, tiles))
        return real(fn, rows, args, kwargs, tiles, exchange)

    monkeypatch.setattr(swapab, "_launch_cluster", counted)
    return calls


def _launch(q, tail, freqs, positions, chain=True, kv_before_wait=True, **cluster):
    kw = dict(max_rows=24, narrow_combine=True)
    if chain:
        kw.update(
            pdl=True,
            combine_entry_trigger=True,
            q_rope=(freqs, positions),
            kv_before_wait=kv_before_wait,
        )
    return swapab.swapab_attention(q, *tail, **kw, **cluster)


FORCED = dict(cluster_merge=True, force_cluster=True)


@pytest.mark.parametrize("b", [1, 6])
def test_cluster_merge_is_the_narrow_combine(b, launches):
    n = 0
    for window in (True, False):
        x = _inputs(b, 1000 * b, window=window)
        for chain, kvbw in ((False, False), (True, False), (True, True)):
            want = _launch(*x, chain, kvbw)
            bitwise(_launch(*x, chain, kvbw, **FORCED), want)
            n += 1
    x = _inputs(b, 11, extra_tokens=250_000)
    bitwise(_launch(*x, **FORCED), _launch(*x))
    assert len(launches) == n + 1


def test_cluster_merge_policy(launches):
    # only where it measured faster: 8 heads, the low-ratio range, 1-8 rows
    assert swapab.cluster_merge_takes(1, 10, 8) and swapab.cluster_merge_takes(8, 10, 8)
    assert not swapab.cluster_merge_takes(9, 10, 8)
    assert not swapab.cluster_merge_takes(6, 10, 16)
    assert not swapab.cluster_merge_takes(6, 2, 8)
    _launch(*_inputs(6, 3, window=False), cluster_merge=True)
    _launch(*_inputs(12, 3), cluster_merge=True)
    assert not launches
    x = _inputs(6, 3)
    bitwise(_launch(*x, cluster_merge=True), _launch(*x))
    assert launches == [(6, 10)]
