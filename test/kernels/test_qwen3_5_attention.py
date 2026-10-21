# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The full-attention kernels of slots skv2 and mka, on one SM100-family GPU (GB300), at
Qwen3.5's TP8 verify shape (7 rows of 4 query heads over one KV head of 256, FP8 pages of
64):

- SKV2 against an FP64 reference of the masked attention: at most twice the error of
  SGLang's attention (TRT-LLM MHA's decode kernel) on the same FP8 inputs, at KV lengths
  from one page to past 128K;
- SKV2 replayed in a CUDA graph on new inputs, bit for bit its eager result;
- the MKA prologue and epilogue twice on the same inputs, and replayed in a CUDA graph on
  new ones, bit for bit their eager result (FP8 Q, bf16 gate, the written K / V rows and
  the o_proj output);
- the ATTN_FRONT op (slot draft_front's kernel, sm103/draft/front.py) without its cache
  write bit for bit SGLang's fused q / k norm + RoPE + gate split, and with it replayed
  in a CUDA graph bit for bit.

Their bits against the kernels of SGLang branch nvidia/qwen35-agentx-gb300 are checked
with that branch installed as ``sglang``, not here.
"""

import math

import pytest
import torch

ROWS, HEADS, DIM, PAGE = 7, 4, 256, 64
SCALE = DIM**-0.5


def _gpu_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import cutlass  # noqa: F401
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _gpu_ready(), reason="needs an SM100-family GPU")


def _hnd(pool):
    """The pool's NHD rows as HND pages of one KV head, with TRT-LLM MHA's canonical
    strides (as its decode kernel and SKV2 get them)."""
    pages = pool.shape[0] // PAGE
    return pool.as_strided((pages, 1, PAGE, DIM), (PAGE * DIM, PAGE * DIM, DIM, 1))


def _fill(q, pools, table, lengths, length, g):
    """New FP8 query rows and pool contents, the request's pages scattered over the
    pool, and its length, written in place."""
    q.copy_((torch.randn(q.shape, device="cuda", generator=g) * 2).to(q.dtype))
    for pool in pools:
        pool.copy_(
            (torch.randn(pool.shape, device="cuda", generator=g) * 2).to(q.dtype)
        )
    pages = math.ceil(length / PAGE)
    total = pools[0].shape[0] // PAGE
    table.zero_()
    table[0, :pages] = torch.randperm(total, device="cuda", generator=g)[:pages].int()
    lengths.fill_(length)


def _inputs(length, g, spare_pages=8):
    """FP8 query rows, a paged FP8 KV pool with the request's pages scattered, its page
    table and its length."""
    total = math.ceil(length / PAGE) + spare_pages
    q = torch.empty(ROWS, HEADS, DIM, dtype=torch.float8_e4m3fn, device="cuda")
    pools = [
        torch.empty(total * PAGE, 1, DIM, dtype=torch.float8_e4m3fn, device="cuda")
        for _ in range(2)
    ]
    table = torch.zeros(1, total + 4, dtype=torch.int32, device="cuda")
    lengths = torch.zeros(1, dtype=torch.int32, device="cuda")
    _fill(q, pools, table, lengths, length, g)
    return q, pools, table, lengths


def _reference(q, kv, table, length, qk_scale, v_scale):
    """FP64 attention of the 7 verify rows, row t seeing the first length - 7 + t + 1
    tokens."""
    k, v = kv
    pages = table[0, : math.ceil(length / PAGE)].long()
    keys = k[pages, 0].reshape(-1, DIM)[:length].double()
    values = v[pages, 0].reshape(-1, DIM)[:length].double()
    scores = q.double() @ keys.T * qk_scale  # [rows, heads, length]
    tokens = torch.arange(length, device="cuda")
    for t in range(ROWS):
        scores[t, :, tokens > length - ROWS + t] = -math.inf
    return torch.softmax(scores, dim=-1) @ values * v_scale


def _sglang(q, kv, table, lengths, qk_scale, v_scale):
    """SGLang's attention on the same inputs: TRT-LLM MHA's decode kernel at 7 rows."""
    import flashinfer

    workspace = torch.zeros(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    return flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        query=q,
        kv_cache=kv,
        workspace_buffer=workspace,
        block_tables=table,
        seq_lens=lengths,
        max_seq_len=table.shape[1] * PAGE,
        bmm1_scale=qk_scale,
        bmm2_scale=v_scale,
        out_dtype=torch.bfloat16,
        q_len_per_req=ROWS,
    )


@pytest.mark.parametrize("length", [7, 64, 2048, 8192, 32769, 131072, 150007])
def test_skv2_is_within_twice_sglangs_error(length):
    from sgl_nvidia_mega.qwen3_5.sm103.attention.skv2 import StockSplitPlan

    g = torch.Generator(device="cuda").manual_seed(length)
    q, pools, table, lengths = _inputs(length, g)
    kv = tuple(map(_hnd, pools))
    qk_scale, v_scale = 0.75 * SCALE, 0.5
    out = torch.empty(ROWS, HEADS, DIM, dtype=torch.bfloat16, device="cuda")
    plan = StockSplitPlan(q, kv, table, lengths)
    plan.run(q, kv, table, lengths, qk_scale=qk_scale, v_scale=v_scale, out=out)
    want = _reference(q, kv, table, length, qk_scale, v_scale)
    assert torch.isfinite(out).all()
    err = (out.double() - want).square().mean().sqrt().item()
    stock = _sglang(q, kv, table, lengths, qk_scale, v_scale)
    stock_err = (stock.double() - want).square().mean().sqrt().item()
    assert err <= 2 * stock_err + 1e-6, (length, err, stock_err)


def test_skv2_replays_in_a_cuda_graph_bit_for_bit():
    from sgl_nvidia_mega.qwen3_5.sm103.attention.skv2 import StockSplitPlan

    g = torch.Generator(device="cuda").manual_seed(1)
    q, pools, table, lengths = _inputs(150007, g)
    kv = tuple(map(_hnd, pools))
    qk_scale, v_scale = 0.75 * SCALE, 0.5
    out = torch.empty(ROWS, HEADS, DIM, dtype=torch.bfloat16, device="cuda")
    plan = StockSplitPlan(q, kv, table, lengths)
    plan.run(q, kv, table, lengths, qk_scale=qk_scale, v_scale=v_scale, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.run(q, kv, table, lengths, qk_scale=qk_scale, v_scale=v_scale, out=out)
    for length in (131072, 9000, 7, 150007):
        _fill(q, pools, table, lengths, length, g)
        graph.replay()
        want = torch.empty_like(out)
        plan.run(q, kv, table, lengths, qk_scale=qk_scale, v_scale=v_scale, out=want)
        assert torch.equal(out.view(torch.int16), want.view(torch.int16)), length


@pytest.mark.parametrize("length", [7, 2048, 32769, 131072, 150007])
def test_skv2_as_a_programmatic_dependent_gives_the_same_bytes(length):
    """The default plan launches the producer and the merge as programmatic dependents,
    the producer loading its leading K / V tiles before it waits: same bytes as the plan
    that launches them plainly."""
    from sgl_nvidia_mega.qwen3_5.sm103.attention.skv2 import StockSplitPlan

    g = torch.Generator(device="cuda").manual_seed(100 + length)
    q, pools, table, lengths = _inputs(length, g)
    kv = tuple(map(_hnd, pools))
    outs = []
    for pdl in (False, True):
        out = torch.empty(ROWS, HEADS, DIM, dtype=torch.bfloat16, device="cuda")
        plan = StockSplitPlan(q, kv, table, lengths, pdl=pdl)
        plan.run(q, kv, table, lengths, qk_scale=0.75 * SCALE, v_scale=0.5, out=out)
        outs.append(out)
    torch.cuda.synchronize()
    assert torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16))


HIDDEN, QKV_ROWS, POOL_SLOTS = 4096, (2 * HEADS + 2) * DIM, 1024


def _fp8(shape, g, scale=1.0):
    return (torch.randn(shape, device="cuda", generator=g) * scale).to(
        torch.float8_e4m3fn
    )


def _mka_operands(g):
    """The TP8 layer's static operands: packed FP8 qkv (+ gate) and o_proj weights and
    their static scales, the q / k norm weights, an fp32 M-RoPE cos / sin cache, its axis
    map and FP8 KV scales."""
    from sgl_nvidia_mega.qwen3_5.sm103.attention import mka

    def scale(v):
        return torch.full((1,), v, dtype=torch.float32, device="cuda")

    angles = torch.rand(4096, 32, device="cuda", generator=g) * 6.3
    return dict(
        w_qkv=mka.pack_qkv(_fp8((QKV_ROWS, HIDDEN), g), HEADS),
        w_o=mka.pack_o(_fp8((HIDDEN, HEADS * DIM), g)),
        qkv_sx=scale(0.02),
        qkv_sw=scale(0.003),
        o_sx=scale(0.05),
        o_sw=scale(0.002),
        qnw=(torch.randn(DIM, device="cuda", generator=g) * 0.1).to(torch.bfloat16),
        knw=(torch.randn(DIM, device="cuda", generator=g) * 0.1).to(torch.bfloat16),
        cos_sin=torch.cat([angles.cos(), angles.sin()], dim=1).contiguous(),
        axis=torch.randint(0, 3, (32,), device="cuda", generator=g),
        k_scale=scale(0.8),
        v_scale=scale(1.3),
    )


class _MkaStep:
    """One step's buffers, refilled in place: the normed rows, positions, cache slots and
    the attention output the epilogue reads; the FP8 pool and every output."""

    def __init__(self, m, g):
        self.x = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device="cuda")
        self.pos = torch.empty(3, m, dtype=torch.int64, device="cuda")
        self.loc = torch.empty(m, dtype=torch.int64, device="cuda")
        self.attn = torch.empty(m, HEADS * DIM, dtype=torch.bfloat16, device="cuda")
        self.k = _fp8((POOL_SLOTS, 1, DIM), g)
        self.v = _fp8((POOL_SLOTS, 1, DIM), g)
        self.q = torch.empty(m, HEADS * DIM, dtype=torch.float8_e4m3fn, device="cuda")
        self.gate = torch.empty(m, HEADS * DIM, dtype=torch.bfloat16, device="cuda")
        self.out = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device="cuda")
        self.fill(g)

    def fill(self, g):
        m = self.x.shape[0]
        self.x.copy_(torch.randn(self.x.shape, device="cuda", generator=g) * 3)
        self.pos.copy_(
            torch.randint(0, 4096, self.pos.shape, device="cuda", generator=g)
        )
        self.loc.copy_(torch.randperm(POOL_SLOTS, device="cuda", generator=g)[:m])
        self.attn.copy_(torch.randn(self.attn.shape, device="cuda", generator=g))

    def run(self, ops):
        from sgl_nvidia_mega.qwen3_5.sm103.attention import mka

        mka.prologue(
            x=self.x,
            w_packed=ops["w_qkv"],
            sx=ops["qkv_sx"],
            sw=ops["qkv_sw"],
            q_norm_w=ops["qnw"],
            k_norm_w=ops["knw"],
            cos_sin=ops["cos_sin"],
            positions=self.pos,
            axis_map=ops["axis"],
            cache_loc=self.loc,
            k_cache=self.k,
            v_cache=self.v,
            k_scale=ops["k_scale"],
            v_scale=ops["v_scale"],
            q_out=self.q,
            gate_out=self.gate,
            nq=HEADS,
            eps=1e-6,
        )
        mka.epilogue(
            attn=self.attn,
            gate=self.gate,
            w_packed=ops["w_o"],
            sx=ops["o_sx"],
            sw=ops["o_sw"],
            out=self.out,
            nq=HEADS,
        )

    def bits(self):
        torch.cuda.synchronize()
        rows = (self.k[self.loc], self.v[self.loc])
        return [
            t.clone().view(torch.uint8) for t in (self.q, self.gate, *rows, self.out)
        ]


def _same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))


def test_mka_replays_in_a_cuda_graph_bit_for_bit():
    g = torch.Generator(device="cuda").manual_seed(5)
    ops = _mka_operands(g)
    step = _MkaStep(7, g)
    step.run(ops)
    first = step.bits()
    step.run(ops)
    assert _same(first, step.bits())  # deterministic
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step.run(ops)
    for _ in range(3):
        step.fill(g)
        graph.replay()
        replayed = step.bits()
        step.run(ops)
        assert _same(replayed, step.bits())
        assert torch.isfinite(step.out.float()).all()


def _bits(a, b):
    """Whether two tensors (of any dtype, FP8 too) hold the same bytes."""
    if a is None or b is None:
        return a is b
    return a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def _front_operands(rows, g, mrope=True, slots=64 * 16):
    q_gate = (torch.randn(rows, HEADS * 2 * DIM, device="cuda", generator=g) * 3).to(
        torch.bfloat16
    )
    k, v = (
        (torch.randn(rows, DIM, device="cuda", generator=g) * 3).to(torch.bfloat16)
        for _ in range(2)
    )
    norms = [
        (torch.randn(DIM, device="cuda", generator=g) * 0.1).to(torch.bfloat16)
        for _ in range(2)
    ]
    angles = torch.rand(4096, 32, device="cuda", generator=g) * 6.3
    cos_sin = torch.cat([angles.cos(), angles.sin()], dim=1).contiguous()
    if mrope:
        pos = torch.randint(0, 4096, (3, rows), device="cuda", generator=g)
        axis = torch.randint(0, 3, (32,), device="cuda", generator=g)
    else:
        pos, axis = torch.randint(0, 4096, (rows,), device="cuda", generator=g), None
    pools = tuple(
        torch.randn(slots, 1, DIM, device="cuda", generator=g).to(torch.float8_e4m3fn)
        for _ in range(2)
    )
    loc = torch.randperm(slots, device="cuda", generator=g)[:rows]
    args = (q_gate, k, *norms, cos_sin, pos, 1e-6, HEADS, 1, DIM, 64)
    return args, axis, v, pools, loc


@pytest.mark.parametrize("rows", [1, 7])
@pytest.mark.parametrize("mrope", [False, True])
def test_front_without_the_cache_write_is_sglangs(rows, mrope):
    from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
        fused_qk_gemma_rmsnorm_rope_gate as sglang_op,
    )

    from sgl_nvidia_mega.qwen3_5.sm103.draft.front import (
        fused_qk_gemma_rmsnorm_rope_gate,
    )

    g = torch.Generator(device="cuda").manual_seed(rows + 10 * mrope)
    args, axis, *_ = _front_operands(rows, g, mrope)
    want = sglang_op(*args, has_gate=True, mrope_axis_map=axis)
    got = fused_qk_gemma_rmsnorm_rope_gate(*args, has_gate=True, mrope_axis_map=axis)
    assert all(_bits(a, b) for a, b in zip(want, got))


def test_front_with_the_cache_write_replays_in_a_cuda_graph_bit_for_bit():
    from sgl_nvidia_mega.qwen3_5.sm103.draft.front import (
        fused_qk_gemma_rmsnorm_rope_gate,
    )

    g = torch.Generator(device="cuda").manual_seed(5)
    args, axis, v, pools, loc = _front_operands(7, g)
    scales = [torch.full((1,), s, device="cuda") for s in (0.8, 1.3)]
    kw = dict(has_gate=True, mrope_axis_map=axis, v=v, kv_cache=pools, cache_loc=loc)

    def run():
        return fused_qk_gemma_rmsnorm_rope_gate(
            *args, **kw, k_scale=scales[0], v_scale=scales[1]
        )

    q8, _, gate = run()
    eager = [q8.clone(), gate.clone(), *(p.clone() for p in pools)]
    for p in pools:
        p.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q8, _, gate = run()
    graph.replay()
    torch.cuda.synchronize()
    rows = loc.long()
    for a, b in zip(eager[:2], (q8, gate)):
        assert _bits(a, b)
    for a, b in zip(eager[2:], pools):
        assert _bits(a[rows], b[rows])
