# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Swap-AB reads an indexer layer's sparse range after its PDL wait.

- The premise: in a captured graph a PDL launch after a stream join depends on the
  joined stream's kernel by a programmatic edge, so it may start (and read) before that
  kernel's writes once the kernel triggers; eager, the join is a full dependency.
- swap-AB with ``extra_after_wait`` is bit for bit the current launch, and in a graph
  behind a writer that triggers before it writes (as SGLang's top-k kernels and the c2
  compressor do) it reads the new selection and the new compressed rows on every
  replay; with only ``kv_before_wait``, it reads the old ones.
"""

import pytest
import torch
import triton
import triton.language as tl

from sgl_nvidia_mega.deepseek_v41.sm103.attention import swapab


def _ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        from triton.experimental import gluon  # noqa: F401
    except ImportError:
        return False
    return True


needs_gpu = pytest.mark.skipif(not _ready(), reason="needs an SM100-family GPU, Gluon")

SPIN_NS = 100_000
COPY_BLOCK = 1024


@triton.jit
def _late_copy(SRC, DST, N, BLOCK: tl.constexpr, spin_ns: tl.constexpr):
    # SGLang's top-k shape of the hazard: release the dependents, then write
    tl.extra.cuda.gdc_launch_dependents()
    t0 = tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1
    )
    t = t0
    while t - t0 < spin_ns:
        t = tl.inline_asm_elementwise(
            "mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1
        )
    # a grid that fits in one wave: every program releases before any writes
    for start in range(tl.program_id(0) * BLOCK, N, tl.num_programs(0) * BLOCK):
        i = start + tl.arange(0, BLOCK)
        tl.store(DST + i, tl.load(SRC + i, i < N), i < N)


@triton.jit
def _read_before_wait(SRC, DST, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(SRC + i, i < N, volatile=True)
    tl.extra.cuda.gdc_wait()
    tl.store(DST + i, v, i < N)


def _late_writer(src, dst, side):
    """``dst.copy_(src)`` on ``side``, joined back by events, by a kernel that releases
    its dependents 100 us before it writes."""
    main = torch.cuda.current_stream()
    side.wait_stream(main)
    with torch.cuda.stream(side):
        src_, dst_ = src.view(-1).view(torch.int32), dst.view(-1).view(torch.int32)
        n = src_.numel()
        _late_copy[(min(triton.cdiv(n, COPY_BLOCK), 64),)](
            src_, dst_, n, BLOCK=COPY_BLOCK, spin_ns=SPIN_NS
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


def _cache(pages, page=256, seed=0):
    """A KV cache view over padded pages, and the raw pages under it."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    stride = page * swapab.LAYOUT.bytes_per_token + 256
    raw = torch.randint(
        0, 120, (pages, stride), dtype=torch.uint8, device="cuda", generator=g
    )
    view = raw.as_strided(
        (pages, page, 1, swapab.LAYOUT.bytes_per_token),
        (stride, swapab.LAYOUT.bytes_per_token, swapab.LAYOUT.bytes_per_token, 1),
    )
    return view, raw


class Attn:
    """One swap-AB call's operands: a 128-slot window and a 512-slot sparse range."""

    def __init__(self, rows, ctx=20000, seed=0):
        torch.manual_seed(seed)
        self.rows = rows
        self.q = (torch.randn(rows, 1, 16, 512, device="cuda") * 0.5).bfloat16()
        self.swa, self.swa_raw = _cache(4, seed=seed)
        self.extra, self.extra_raw = _cache(-(-ctx // 256), seed=seed + 1)
        base = torch.randint(0, 600, (rows, 1, 1), device="cuda", dtype=torch.int32)
        self.ids = base + torch.arange(128, device="cuda", dtype=torch.int32)
        self.lengths = torch.full((rows,), 128, device="cuda", dtype=torch.int32)
        self.el = torch.randint(400, 513, (rows,), device="cuda", dtype=torch.int32)
        self.sink = torch.randn(16, device="cuda")
        angles = torch.rand(ctx, 32, device="cuda") * 6.3
        self.freqs = (
            torch.view_as_real(torch.polar(torch.ones_like(angles), angles))
            .flatten(-2)
            .contiguous()
        )
        self.pos = torch.randint(0, ctx, (rows,), device="cuda")
        self.ctx = ctx

    def selection(self, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        return torch.stack(
            [
                torch.randperm(self.ctx, device="cuda", generator=g)[:512]
                for _ in range(self.rows)
            ]
        ).to(torch.int32)[:, None, :]

    def run(self, sel, **kw):
        return swapab.swapab_attention(
            self.q,
            self.swa,
            self.ids,
            self.lengths,
            self.sink,
            self.extra,
            sel,
            self.el,
            max_rows=24,
            narrow_combine=True,
            q_rope=(self.freqs, self.pos),
            combine_entry_trigger=True,
            pdl=True,
            **kw,
        )


def _bitwise(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


@needs_gpu
@pytest.mark.parametrize("rows", (1, 2, 6, 8, 12, 24))
def test_extra_after_wait_is_the_current_launch_bit_for_bit(rows):
    a = Attn(rows, seed=rows)
    sel = a.selection(rows)
    want = a.run(sel, kv_before_wait=True)
    got = a.run(sel, kv_before_wait=True, extra_after_wait=True)
    plain = a.run(sel)
    torch.cuda.synchronize()
    assert _bitwise(got, want)
    assert _bitwise(got, plain)
    g, out = _graph(lambda: a.run(sel, kv_before_wait=True, extra_after_wait=True))
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    assert _bitwise(out, plain)


@needs_gpu
@pytest.mark.parametrize("rows", (1, 6, 8))
def test_a_late_selection_is_read_after_the_wait(rows):
    a = Attn(rows, seed=7 + rows)
    sel = torch.empty(rows, 1, 512, dtype=torch.int32, device="cuda")
    fresh = torch.empty_like(sel)
    side = torch.cuda.Stream()
    graphs = {}
    for after in (False, True):

        def chain(after=after):
            _late_writer(fresh, sel, side)
            return a.run(sel, kv_before_wait=True, extra_after_wait=after)

        sel.copy_(a.selection(0))
        fresh.copy_(a.selection(1))
        graphs[after] = _graph(chain)
    stale = {False: 0, True: 0}
    for r in range(8):
        new = a.selection(100 + r)
        want = a.run(new)
        for after, (g, out) in graphs.items():
            sel.copy_(a.selection(200 + r))  # the previous layer's selection
            fresh.copy_(new)
            torch.cuda.synchronize()
            g.replay()
            torch.cuda.synchronize()
            stale[after] += int(not _bitwise(out, want))
    assert stale[True] == 0
    # the hazard: the selection gathered before the wait
    assert stale[False] > 0


@needs_gpu
@pytest.mark.parametrize("rows", (1, 6))
def test_late_compressed_rows_are_read_after_the_wait(rows):
    """The c2 compressor releases its dependents before it stores the compressed rows
    the layer's sparse range gathers."""
    a = Attn(rows, ctx=4096, seed=11 + rows)
    sel = a.selection(3)
    old, new = a.extra_raw.clone(), torch.empty_like(a.extra_raw)
    fresh = torch.empty_like(a.extra_raw)
    side = torch.cuda.Stream()
    graphs = {}
    for after in (False, True):
        graphs[after] = _graph(
            lambda after=after: (
                _late_writer(fresh, a.extra_raw, side),
                a.run(sel, kv_before_wait=True, extra_after_wait=after),
            )[1]
        )
    stale = {False: 0, True: 0}
    for r in range(8):
        new.copy_(_cache(a.extra_raw.shape[0], seed=300 + r)[1])
        a.extra_raw.copy_(new)
        want = a.run(sel)
        for after, (g, out) in graphs.items():
            a.extra_raw.copy_(old)
            fresh.copy_(new)
            torch.cuda.synchronize()
            g.replay()
            torch.cuda.synchronize()
            stale[after] += int(not _bitwise(out, want))
    assert stale[True] == 0
    assert stale[False] > 0
