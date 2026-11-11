# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot attn_compressor_priority (scheduling only; derived from the COMPRESSOR_PRIORITY
switch of muLLM v0.4.1, NVIDIA Corporation, Apache-2.0).

- The compressor's kernel node is captured at the priority stream's priority, the other
  nodes at the default, and a torch-instantiated graph dispatches a ready high-priority
  node ahead of a low-priority grid's pending CTAs.
"""

import statistics
from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl


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


@triton.jit
def _now():
    return tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1
    )


@triton.jit
def _spin(START, END, spin_ns):
    pid = tl.program_id(0)
    t0 = _now()
    t = t0
    while t - t0 < spin_ns:
        t = _now()
    tl.store(START + pid, t0)
    tl.store(END + pid, t)


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


def _kernel_nodes(graph_ptr):
    from cuda.bindings import driver as cu

    def chk(res):
        assert int(res[0]) == 0, res[0]
        return res[1] if len(res) == 2 else res[1:]

    g = cu.CUgraph(graph_ptr)
    _, n = chk(cu.cuGraphGetNodes(g, 0))
    nodes, _ = chk(cu.cuGraphGetNodes(g, n))
    out = []
    for node in nodes:
        if (
            chk(cu.cuGraphNodeGetType(node))
            != cu.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL
        ):
            continue
        params = chk(cu.cuGraphKernelNodeGetParams(node))
        attr = chk(
            cu.cuGraphKernelNodeGetAttribute(
                node, cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PRIORITY
            )
        )
        out.append((params.gridDimX, attr.priority))
    return out


@triton.jit
def _fill(X, M, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(X + i, tl.full((N,), 1, tl.int32) + tl.load(M))


@needs_gpu
@pytest.mark.parametrize("slot", (False, True))
def test_the_compressor_node_is_captured_at_high_priority(slot):
    """prepare.low_ratio with a stand-in compressor (grid 3) and indexer (grid 5): with
    the slot, only the compressor's node carries the high priority; the indexer reads the
    compressor's write on every replay."""
    from sgl_nvidia_mega.deepseek_v41.attention import prepare

    hi = torch.cuda.Stream.priority_range()[1]
    comp_out = torch.zeros(64, dtype=torch.int32, device="cuda")
    idx_out = torch.zeros(64, dtype=torch.int32, device="cuda")
    t = torch.zeros(2, dtype=torch.int64, device="cuda")
    marker = torch.zeros(1, dtype=torch.int32, device="cuda")

    def sources(run_indexer=True, run_compressor=True, **kw):
        if not run_indexer:  # the compressor: a late write
            _spin[(3,)](t, t[1:], 20_000)
            _fill[(3,)](comp_out, marker, N=64)
        else:
            idx_out.copy_(comp_out)

    attn = SimpleNamespace(
        alt_streams=[torch.cuda.Stream() for _ in range(3)],
        compressor=object(),
        indexer=object(),
        fuse_wqa_wkv=False,
        compress_ratio=2,
    )
    q_ops = SimpleNamespace(
        compressor_stream=torch.cuda.Stream(priority=hi) if slot else None,
        compute_q_a=lambda attn, x, qkv_a: (x, x),
        compute_q_b=lambda attn, q, pos, out: q * 2,
        index_q_input=lambda attn, q_lora, q: q_lora,
        kv_to_cache=lambda *a: None,
    )
    view = SimpleNamespace(metadata=None, low_ratio_sources=sources)
    fb = SimpleNamespace(forward_mode=None)
    x = torch.zeros(6, 4, device="cuda")

    def body():
        return prepare.low_ratio(attn, x, None, fb, view, None, None, q_ops)

    body()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g, stream=stream):
        body()
    g.instantiate()
    nodes = _kernel_nodes(g.raw_cuda_graph())
    comp = [p for grid, p in nodes if grid == 3]
    assert comp and all(p == (hi if slot else 0) for p in comp), nodes
    assert all(p == 0 for grid, p in nodes if grid != 3), nodes
    for r in range(5):
        marker.fill_(r + 10)
        comp_out.zero_()
        idx_out.zero_()
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        assert bool((idx_out == r + 11).all())


@needs_gpu
def test_a_captured_high_priority_node_is_dispatched_first():
    """The premise of attn_compressor_priority on this torch / driver: a node captured on
    a high-priority stream starts ahead of a low-priority grid's pending CTAs."""
    nsm = torch.cuda.get_device_properties(0).multi_processor_count
    grid_a = nsm * 4 * 20
    lo_prio, hi_prio = torch.cuda.Stream.priority_range()
    a_s = torch.zeros(grid_a, dtype=torch.int64, device="cuda")
    a_e = torch.zeros_like(a_s)
    b = torch.zeros(4, dtype=torch.int64, device="cuda")
    starts = {}
    for name, prio in (("equal", lo_prio), ("high", hi_prio)):
        lo = torch.cuda.Stream(priority=lo_prio)
        hi = torch.cuda.Stream(priority=prio)

        def body(lo=lo, hi=hi):
            cur = torch.cuda.current_stream()
            lo.wait_stream(cur)
            hi.wait_stream(cur)
            with torch.cuda.stream(lo):
                _spin[(grid_a,)](a_s, a_e, 3000, num_warps=16)
            with torch.cuda.stream(hi):
                _spin[(1,)](b, b[1:], 2000, num_warps=1)
                _spin[(1,)](b[2:], b[3:], 0, num_warps=1)
            cur.wait_stream(lo)
            cur.wait_stream(hi)

        g, _ = _graph(body)
        late = []
        for _ in range(20):
            g.replay()
            torch.cuda.synchronize()
            late.append((int(b[2]) - int(a_s.min())) / 1e3)
        starts[name] = statistics.median(late)
    assert starts["high"] < 0.3 * starts["equal"], starts
