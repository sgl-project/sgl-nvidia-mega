# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The fused consumer's selection is written before its dependents launch.

- The premise: in a captured graph a PDL launch after a stream join depends on the
  joined stream's kernel by a programmatic edge, so it may start (and read) before that
  kernel's writes once the kernel triggers; eager, the join is a full dependency.
- The fused consumer with ``trigger_after_write`` (the default): a dependent that reads
  the selection before its own wait sees the written selection on every replay.
"""

import sys
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl

HERE = Path(__file__).resolve().parent


def _ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import deep_gemm  # noqa: F401
        from sglang.kernels.ops.attention.dsv4 import candidate_table  # noqa: F401
    except ImportError:
        return False
    return True


needs_gpu = pytest.mark.skipif(not _ready(), reason="needs an SM100-family GPU, SGLang")

SPIN_NS = 100_000


@triton.jit
def _late_copy(SRC, DST, N: tl.constexpr, BLOCK: tl.constexpr, spin_ns: tl.constexpr):
    # a writer that releases its dependents, then writes
    tl.extra.cuda.gdc_launch_dependents()
    t0 = tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1
    )
    t = t0
    while t - t0 < spin_ns:
        t = tl.inline_asm_elementwise(
            "mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1
        )
    i = tl.arange(0, BLOCK)
    tl.store(DST + i, tl.load(SRC + i, i < N), i < N)


@triton.jit
def _read_before_wait(SRC, DST, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(SRC + i, i < N, volatile=True)
    tl.extra.cuda.gdc_wait()
    tl.store(DST + i, v, i < N)


def _late_writer(src, dst, side):
    main = torch.cuda.current_stream()
    side.wait_stream(main)
    with torch.cuda.stream(side):
        n = src.numel()
        _late_copy[(1,)](
            src, dst, N=n, BLOCK=triton.next_power_of_2(n), spin_ns=SPIN_NS
        )
    main.wait_stream(side)


def _graph(fn):
    fn()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        out = fn()
    torch.cuda.synchronize()
    return g, out


@needs_gpu
def test_a_captured_join_is_a_programmatic_edge():
    src = torch.arange(64, dtype=torch.int32, device="cuda") + 1
    dst = torch.zeros_like(src)
    seen = torch.zeros_like(src)
    side = torch.cuda.Stream()

    def chain():
        _late_writer(src, dst, side)
        _read_before_wait[(1,)](dst, seen, N=64, BLOCK=64, launch_pdl=True)

    dst.zero_()
    chain()
    torch.cuda.synchronize()
    assert torch.equal(seen, src), "eager: the event join orders the write"
    g, _ = _graph(chain)
    stale = 0
    for _ in range(5):
        dst.zero_()
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        stale += int(not torch.equal(seen, src))
    assert stale == 5, "captured: the dependent ran before the joined kernel wrote"


def _reindex_case():
    sys.path.insert(0, str(HERE))
    try:
        from test_deepseek_v41_reindex_select import Case
    finally:
        sys.path.remove(str(HERE))
    return Case


@needs_gpu
@pytest.mark.parametrize("rows", (1, 6, 8))
def test_the_fused_consumer_writes_before_it_triggers(rows):
    from sgl_nvidia_mega.deepseek_v41.sm103.attention import reindex_select as rs

    Case = _reindex_case()
    case = Case([30000 + 11 * i for i in range(rows)], seed=rows)
    phys, valid_lens, _ = case.publish()
    out = torch.full((rows, 512), -7, dtype=torch.int32, device="cuda")
    seen = torch.zeros_like(out)
    side = torch.cuda.Stream()

    def chain():
        main = torch.cuda.current_stream()
        side.wait_stream(main)
        with torch.cuda.stream(side):
            rs.reindex_select(
                case.q_fp4, case.q_sf, case.weights, case.k_cache, phys, valid_lens, out
            )
        main.wait_stream(side)
        n = out.numel()
        _read_before_wait[(1,)](
            out, seen, N=n, BLOCK=triton.next_power_of_2(n), launch_pdl=True
        )

    g, _ = _graph(chain)
    for r in range(10):
        other = Case([30000 + 11 * i for i in range(rows)], seed=100 + r)
        case.q_fp4.copy_(other.q_fp4)
        case.weights.copy_(other.weights)
        out.fill_(-7)
        seen.zero_()
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        assert (out != -7).any()
        assert torch.equal(seen, out)
