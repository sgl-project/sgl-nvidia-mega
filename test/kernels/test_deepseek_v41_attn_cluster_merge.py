# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Swap-AB attention with a row's tiles merged inside its CTA cluster (slot
attn_cluster_merge) on one SM100-family GPU (GB300): bit for bit the separate narrow
combine at 1-8 rows, 16 and 8 heads, the low-ratio range (window + compressed, padded
indices, short lengths) and the window only, every launch form the chain uses, graph
replay, a long-context cache; past 8 rows, and where the policy keeps the combine, the
launch is the current one.
"""

import os
import sys

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41.sm103.attention import swapab

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_deepseek_v41_attn_combine import (  # noqa: E402
    LAUNCHES,
    _bitwise,
    _inputs,
    _ready,
)

pytestmark = pytest.mark.skipif(not _ready(), reason="needs an SM100-family GPU")

ROWS = (1, 2, 3, 4, 5, 6, 7, 8)


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


def _launch(q, tail, freqs, positions, form, kv_before_wait=False, **cluster):
    kw = dict(LAUNCHES[form])
    if form == "chain":
        kw.update(q_rope=(freqs, positions), kv_before_wait=kv_before_wait)
    return swapab.swapab_attention(q, *tail, **kw, **cluster)


FORCED = dict(cluster_merge=True, force_cluster=True)


@pytest.mark.parametrize("heads", [16, 8])
@pytest.mark.parametrize("b", ROWS)
def test_cluster_merge_is_the_narrow_combine(b, heads, launches):
    n = 0
    for seed in range(3):
        for window_only in (False, True):
            q, tail, freqs, positions = _inputs(
                b, heads, 1000 * b + seed, window=not window_only
            )
            for form in ("plain", "chain"):
                for kvbw in (False, True) if form == "chain" else (False,):
                    want = _launch(q, tail, freqs, positions, form, kvbw)
                    for _ in range(2):
                        got = _launch(q, tail, freqs, positions, form, kvbw, **FORCED)
                        _bitwise(got, want)
                        n += 1
    assert len(launches) == n


@pytest.mark.parametrize("heads", [16, 8])
@pytest.mark.parametrize("b", [1, 6, 8])
def test_cluster_merge_graph_replay(b, heads, launches):
    inputs = [_inputs(b, heads, 7 + b + i) for i in range(3)]
    want = [_launch(*x, "chain", True) for x in inputs]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for x in inputs:
            _launch(*x, "chain", True, **FORCED)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        # three layers back to back, as the chain launches them
        outs = [_launch(*x, "chain", True, **FORCED) for x in inputs]
    for _ in range(20):
        for o in outs:
            o.zero_()
        graph.replay()
        torch.cuda.synchronize()
        for o, w in zip(outs, want):
            _bitwise(o, w)
    assert len(launches) == 6


@pytest.mark.parametrize("heads", [16, 8])
def test_cluster_merge_long_context(heads, launches):
    q, tail, freqs, positions = _inputs(6, heads, 11, extra_tokens=250_000)
    want = _launch(q, tail, freqs, positions, "chain", True)
    got = _launch(q, tail, freqs, positions, "chain", True, **FORCED)
    _bitwise(got, want)
    assert len(launches) == 1


@pytest.mark.parametrize("b", [12, 24])
def test_cluster_merge_past_one_wave_is_the_two_launches(b, launches):
    q, tail, freqs, positions = _inputs(b, 8, 5 + b)
    want = _launch(q, tail, freqs, positions, "chain", True)
    got = _launch(q, tail, freqs, positions, "chain", True, **FORCED)
    _bitwise(got, want)
    assert not launches


def test_cluster_merge_policy(launches):
    # only where it measured faster: 8 heads, the low-ratio range, 1-8 rows
    assert swapab.cluster_merge_takes(6, 10, 8)
    assert swapab.cluster_merge_takes(1, 10, 8) and swapab.cluster_merge_takes(8, 10, 8)
    assert not swapab.cluster_merge_takes(9, 10, 8)
    assert not swapab.cluster_merge_takes(6, 10, 16)
    assert not swapab.cluster_merge_takes(6, 2, 8)
    q, tail, freqs, positions = _inputs(6, 16, 3)
    _launch(q, tail, freqs, positions, "chain", True, cluster_merge=True)
    q, tail, freqs, positions = _inputs(6, 8, 3, window=False)
    _launch(q, tail, freqs, positions, "chain", True, cluster_merge=True)
    assert not launches
    q, tail, freqs, positions = _inputs(6, 8, 3)
    want = _launch(q, tail, freqs, positions, "chain", True)
    got = _launch(q, tail, freqs, positions, "chain", True, cluster_merge=True)
    _bitwise(got, want)
    assert launches == [(6, 10)]
