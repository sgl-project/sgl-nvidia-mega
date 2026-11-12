# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slots attn_index_early_trigger, attn_defer_block_select and attn_compressor_priority
(scheduling only; derived from the POOL_TOPK_EARLY_TRIGGER, DEFER_BLOCK_SELECT and
COMPRESSOR_PRIORITY switches of muLLM v0.4.1, NVIDIA Corporation, Apache-2.0).

- The fused consumer with ``trigger_after_wait`` writes the same selection bit for bit at
  1-8 rows, eager and replayed, and its trigger really moves: a dependent that reads the
  selection before its own wait reads it stale on every replay (which is why the
  attention reads that selection after its own wait).
- The real index chain (benchmark/bench_deepseek_v41_index_chain.py) with either slot or
  both: every consumer layer's attention output equals the shipped arm's on every graph
  replay while the step input changes between replays (a stale selection or a stale
  candidate table would show as the previous replay's output), also with a 200 us spin
  injected at the start of the deferred block selection; and the candidate table itself
  (blocks, pool slots, lengths) equals the shipped arm's on every replay,
  also with SGLang's consumer kernels (their top-k is not deterministic, so their
  attention outputs differ between two graphs of the same arm).
- The compressor's kernel node is captured at the priority stream's priority, the other
  nodes at the default, and a torch-instantiated graph dispatches a ready high-priority
  node ahead of a low-priority grid's pending CTAs.
"""

import importlib.util
import statistics
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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


@triton.jit
def _read_before_wait(SRC, DST, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(SRC + i, i < N, volatile=True)
    tl.extra.cuda.gdc_wait()
    tl.store(DST + i, v, i < N)


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


def _case():
    sys.path.insert(0, str(HERE))
    try:
        from test_deepseek_v41_reindex_select import Case
    finally:
        sys.path.remove(str(HERE))
    return Case


@needs_gpu
@pytest.mark.parametrize("rows", range(1, 9))
def test_the_trigger_after_wait_writes_the_same_selection(rows):
    from sgl_nvidia_mega.deepseek_v41.sm103.attention import reindex_select as rs

    Case = _case()
    case = Case([30000 + 13 * i for i in range(rows)], seed=rows)
    phys, valid_lens, _ = case.publish()
    outs = {}
    for mode in ("after_write", "after_wait"):
        out = torch.full((rows, 512), -7, dtype=torch.int32, device="cuda")

        def call(out=out, mode=mode):
            rs.reindex_select(
                case.q_fp4,
                case.q_sf,
                case.weights,
                case.k_cache,
                phys,
                valid_lens,
                out,
                inputs_before_wait=True,
                trigger_after_wait=mode == "after_wait",
            )
            return out

        call()
        torch.cuda.synchronize()
        eager = out.clone()
        out.fill_(-7)
        g, _ = _graph(call)
        for _ in range(3):
            out.fill_(-7)
            g.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, eager)
        outs[mode] = eager
    assert torch.equal(outs["after_wait"], outs["after_write"])
    assert (outs["after_wait"] != -7).any()


@needs_gpu
@pytest.mark.parametrize("rows", (1, 6, 8))
def test_the_trigger_after_wait_releases_dependents_before_the_write(rows):
    """The point of the slot, and why it needs the attention's after-wait read: a
    dependent launched behind the join starts before the selection is written."""
    from sgl_nvidia_mega.deepseek_v41.sm103.attention import reindex_select as rs

    Case = _case()
    case = Case([30000 + 11 * i for i in range(rows)], seed=rows)
    phys, valid_lens, _ = case.publish()
    out = torch.full((rows, 512), -7, dtype=torch.int32, device="cuda")
    seen = torch.zeros_like(out)
    side = torch.cuda.Stream()
    results = {}
    for early in (False, True):

        def chain(early=early):
            main = torch.cuda.current_stream()
            side.wait_stream(main)
            with torch.cuda.stream(side):
                rs.reindex_select(
                    case.q_fp4,
                    case.q_sf,
                    case.weights,
                    case.k_cache,
                    phys,
                    valid_lens,
                    out,
                    inputs_before_wait=True,
                    trigger_after_wait=early,
                )
            main.wait_stream(side)
            n = out.numel()
            _read_before_wait[(1,)](
                out, seen, N=n, BLOCK=triton.next_power_of_2(n), launch_pdl=True
            )

        g, _ = _graph(chain)
        stale = 0
        for _ in range(8):
            out.fill_(-7)
            seen.zero_()
            torch.cuda.synchronize()
            g.replay()
            torch.cuda.synchronize()
            assert (out != -7).any()
            stale += int(not torch.equal(seen, out))
        results[early] = stale
    assert results[False] == 0, "the fenced trigger: written before the release"
    assert results[True] == 8, "the early trigger: released before the write"


def _bench():
    path = HERE.parent.parent / "benchmark" / "bench_deepseek_v41_index_chain.py"
    spec = importlib.util.spec_from_file_location("bench_index_chain_sched", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _spin_blocks(ns, t):
    """StepIndexer._blocks with a spin of ``ns`` on its side stream first: the deferred
    block selection (and so the candidate table) lands that much later. ``t``: the spin's
    timestamp scratch, allocated before anything the graphs hold."""
    from sgl_nvidia_mega.deepseek_v41.attention import indexer as indexer_mod

    orig = indexer_mod.StepIndexer._blocks

    def slow(self, *a, **k):
        with torch.cuda.stream(self.stream):
            _spin[(1,)](t, t[1:], ns)
        return orig(self, *a, **k)

    return mock.patch.object(indexer_mod.StepIndexer, "_blocks", slow)


@needs_gpu
@pytest.mark.parametrize(
    "stock, arms, spin",
    [
        (False, ("stock_safe", "stock_safe+t", "stock_safe+d", "stock_safe+t+d"), 0),
        (
            False,
            ("stock_safe", "stock_safe+a", "stock_safe+d", "stock_safe+t+d"),
            200_000,
        ),
        (True, ("stock_safe", "stock_safe+d"), 0),
        (True, ("stock_safe", "stock_safe+a", "stock_safe+d"), 200_000),
    ],
    ids=("fused", "fused-late-table", "stock-consumer", "stock-consumer-late-table"),
)
@pytest.mark.parametrize("heads", (8, 16))
def test_the_index_chain_reads_fresh_selections_and_tables(stock, arms, spin, heads):
    from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll
    from sgl_nvidia_mega.deepseek_v41.sm103.attention import reindex_select as rs

    spin_t = torch.zeros(2, dtype=torch.int64, device="cuda")
    bench = _bench()
    pattern = "R,S,R,C,R,C,C".split(",")
    step = bench.Step(pattern, heads, 6, 65536, stock_consumer=stock)
    graphs, tables = {}, {}
    try:
        for arm in arms:
            # the shipped arm without the spin: the reference; "+a" is the shipped arm
            # with the spin (the table late beside the attention, as without the slot)
            ctx = (
                _spin_blocks(spin, spin_t) if spin and arm != arms[0] else nullcontext()
            )
            with ctx:
                graphs[arm] = step.graph(arm)
            # the table the captured source published (graph-pool tensors)
            t = step.indexer._table
            # (not the sparse schedule: its bytes differ between two replays of the same
            # arm, with or without the slots)
            tables[arm] = (t.blocks, t.phys_blocks, t.valid_lens)
    finally:
        rs.reindex_select = getattr(rs, "_bench_orig", rs.reindex_select)
    index = [i for i, k in enumerate(pattern) if k == "C"]
    differ = {arm: 0 for arm in arms[1:]}
    table_differ = {arm: 0 for arm in arms[1:]}
    changed = table_changed = 0
    prev_table = None
    prev = None
    for r in range(10):
        # a new step input: every layer's query, index query and selection change
        x = torch.randn(6, bench.HIDDEN, device="cuda", generator=None).bfloat16()
        q, sf = mxfp8_ll.quantize(x)
        step.x[0].copy_(q)
        step.x[1].copy_(sf.view_as(step.x[1]))
        got, tab = {}, {}
        for arm in arms:
            graphs[arm][0].replay()
            torch.cuda.synchronize()
            got[arm] = [graphs[arm][1][i].clone() for i in index]
            tab[arm] = [t.clone() for t in tables[arm]]
        for arm in arms[1:]:
            table_differ[arm] += int(
                not all(torch.equal(u, v) for u, v in zip(tab[arms[0]], tab[arm]))
            )
        if prev is not None:
            changed += int(
                not all(torch.equal(u, v) for u, v in zip(prev, got[arms[0]]))
            )
            table_changed += int(
                not all(torch.equal(u, v) for u, v in zip(prev_table, tab[arms[0]]))
            )
        prev = got[arms[0]]
        prev_table = tab[arms[0]]
    assert changed == 9, "the input change must move the consumers' outputs"
    assert table_changed == 9, "the input change must move the candidate table"
    assert table_differ == {arm: 0 for arm in arms[1:]}
    if not stock:
        assert differ == {arm: 0 for arm in arms[1:]}


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
