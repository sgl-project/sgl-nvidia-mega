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


@pytest.mark.parametrize("chunk", [False, True], ids=["exact", "fold-chunk"])
def test_cp_r11_plain_merge_matches_r11_at_b32(chunk):
    """The draft-multi B32 merge's causal math is independent of whether it runs
    before or after publishing the prefix partial.  The D2 plain kernel is the R11
    program (including its volatile-load wait); EARLY is retained here only as a
    numerical comparison, not as the reference.  The direct FP32 calculation is
    diagnostic-only because the fold path merges its chunks in a different order;
    the correctness gate is the plain kernel against the exact R11 program.

    CP=1 makes the low-latency exchange self-contained on one GPU: every word this
    kernel polls is published by the same launch. PDL liveness and the production
    CP=4 exchange remain covered by the no-simulation server gate.
    """
    import hashlib
    import importlib.util
    import os
    import sys
    from pathlib import Path

    from sgl_nvidia_mega.qwen3_5.sm103.attention.q35_cp_lite import (
        _cp_merge_h_ll as feature_merge,
        _cp_merge_h_ll_plain as plain_merge,
    )

    r11_path = os.environ.get("V3D_R11_Q35_CP_LITE")
    r11_sha = os.environ.get("V3D_R11_Q35_CP_LITE_SHA256")
    if r11_path is None and r11_sha is None:
        pytest.skip("exact R11 q35_cp_lite source was not provided")
    assert r11_path is not None and r11_sha is not None
    r11_source = Path(r11_path)
    assert r11_source.is_file()
    actual_sha = hashlib.sha256(r11_source.read_bytes()).hexdigest()
    assert actual_sha == r11_sha, (actual_sha, r11_sha)
    spec = importlib.util.spec_from_file_location("_v3d_r11_q35_cp_lite", r11_source)
    assert spec is not None and spec.loader is not None
    r11_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = r11_module
    spec.loader.exec_module(r11_module)
    r11_merge = r11_module._cp_merge_h_ll

    rows, q_len, heads, dim, page, words = 32 * 7, 7, 4, 256, 64, 520
    g = torch.Generator(device="cuda").manual_seed(20261029)
    q = (torch.randn(rows, heads, dim, device="cuda", generator=g) * 2).to(
        torch.float8_e4m3fn
    )
    po = (torch.randn(1, rows, heads, dim, device="cuda", generator=g) * 0.1).to(
        torch.bfloat16
    )
    plse = torch.randn(1, rows, heads, device="cuda", generator=g)
    chunks_per_request = 2
    po_v = (
        torch.randn(
            32 * chunks_per_request,
            q_len * heads,
            dim,
            device="cuda",
            generator=g,
        )
        * 0.1
    ).to(torch.bfloat16)
    pl_v = torch.randn(
        32 * chunks_per_request,
        q_len * heads,
        device="cuda",
        generator=g,
    )
    vstart = (
        torch.arange(32, dtype=torch.int32, device="cuda") * chunks_per_request
    )
    vcnt = torch.full(
        (32,), chunks_per_request, dtype=torch.int32, device="cuda"
    )
    k, v = (
        (torch.randn(32, 1, page, dim, device="cuda", generator=g) * 2).to(
            torch.float8_e4m3fn
        )
        for _ in range(2)
    )
    table = torch.zeros(32, 2, dtype=torch.int32, device="cuda")
    table[:, 1] = torch.arange(32, dtype=torch.int32, device="cuda")
    lengths = torch.full((32,), page + q_len, dtype=torch.int32, device="cuda")

    def run(kernel, early=None):
        out = torch.empty(rows, heads, dim, dtype=torch.bfloat16, device="cuda")
        ll = torch.empty(rows * words, dtype=torch.int64, device="cuda")
        peers = torch.tensor([ll.data_ptr()], dtype=torch.int64, device="cuda")
        counter = torch.zeros(rows * heads, dtype=torch.int32, device="cuda")
        options = {} if early is None else {"EARLY": early}
        if chunk:
            options.update(
                po_v=po_v.view(torch.int32),
                pl_v=pl_v,
                vstart=vstart,
                vcnt=vcnt,
                CHUNK=True,
            )
        kernel[(rows, heads)](
            po.view(torch.int32),
            plse.view(torch.int32),
            heads * dim // 2,
            po.stride(0) // 2,
            heads,
            plse.stride(0),
            q,
            q.stride(0),
            k,
            v,
            k.stride(0),
            k.stride(2),
            table,
            table.stride(0),
            lengths,
            out,
            out.stride(0),
            peers,
            0,
            0,
            counter,
            0.75 * dim**-0.5,
            0.5,
            32,
            0,
            QL=q_len,
            HQ=heads,
            CP=1,
            D=dim,
            PG=page,
            NWP=words,
            NB=32,
            GMAX=False,
            WAIT=False,
            num_warps=4,
            launch_pdl=False,
            **options,
        )
        torch.cuda.synchronize()
        assert torch.isfinite(out).all()
        assert bool((counter == 1).all())
        return out

    # Direct FP32 evaluation of the same causal block followed by its prefix partial.
    # Elementwise products and reductions avoid TF32 matmul in the reference.
    requests = rows // q_len
    qf = q.float().view(requests, q_len, heads, dim)
    page_ids = table[:, 1].long()
    kf = k[page_ids, 0, :q_len].float()
    vf = v[page_ids, 0, :q_len].float()
    scores = (qf.unsqueeze(3) * kf[:, None, None]).sum(-1) * (0.75 * dim**-0.5)
    causal = torch.arange(q_len, device="cuda")
    scores.masked_fill_(
        causal[None, None, None] > causal[None, :, None, None], -math.inf
    )
    peak = scores.max(-1).values
    weight = torch.exp(scores - peak.unsqueeze(-1))
    causal_sum = weight.sum(-1)
    causal_out = (weight.unsqueeze(-1) * vf[:, None, None]).sum(-2) * 0.5
    if chunk:
        chunk_lse = pl_v.view(
            requests, chunks_per_request, q_len, heads
        ).float()
        chunk_peak = chunk_lse.max(1).values
        chunk_weight = torch.exp2(chunk_lse - chunk_peak.unsqueeze(1))
        chunk_sum = chunk_weight.sum(1)
        prefix_out = (
            po_v.float().view(
                requests, chunks_per_request, q_len, heads, dim
            )
            * chunk_weight.unsqueeze(-1)
        ).sum(1) / chunk_sum.unsqueeze(-1)
        prefix_lse = (chunk_peak + torch.log2(chunk_sum)) * math.log(2.0)
    else:
        prefix_out = po[0].float().view(requests, q_len, heads, dim)
        prefix_lse = plse[0].view(requests, q_len, heads) * math.log(2.0)
    merged_peak = torch.maximum(peak, prefix_lse)
    causal_weight = torch.exp(peak - merged_peak)
    prefix_weight = torch.exp(prefix_lse - merged_peak)
    reference = (
        causal_out * causal_weight.unsqueeze(-1)
        + prefix_out * prefix_weight.unsqueeze(-1)
    ) / (causal_sum * causal_weight + prefix_weight).unsqueeze(-1)
    reference = reference.view(rows, heads, dim)

    rtol, atol = 2**-7, 1e-6
    relative_floor = atol / rtol
    rounded_reference = reference.to(torch.bfloat16)

    def stats(got):
        error = (got.float() - reference).abs()
        relative = error / reference.abs().clamp_min(relative_floor)
        violation = error > (atol + rtol * reference.abs())
        return {
            "max_abs": error.max().item(),
            "max_rel": relative.max().item(),
            "tolerance_violations": violation.count_nonzero().item(),
            "bf16_mismatch": (
                got.view(torch.int16) != rounded_reference.view(torch.int16)
            )
            .count_nonzero()
            .item(),
        }

    def pair_stats(got, baseline):
        error = (got.float() - baseline.float()).abs()
        relative = error / baseline.float().abs().clamp_min(relative_floor)
        return {
            "max_abs": error.max().item(),
            "max_rel": relative.max().item(),
            "bf16_mismatch": (
                got.view(torch.int16) != baseline.view(torch.int16)
            )
            .count_nonzero()
            .item(),
        }

    r11 = run(r11_merge)
    plain, early = run(plain_merge), run(feature_merge, True)
    r11_stats = stats(r11)
    plain_stats, early_stats = stats(plain), stats(early)
    plain_vs_early = pair_stats(plain, early)
    plain_vs_r11 = pair_stats(plain, r11)
    print(
        "DRAFT_MERGE_NUMERICS "
        f"chunk={chunk} rtol={rtol:.9g} atol={atol:.9g} "
        f"relative_floor={relative_floor:.9g} "
        f"r11={r11_stats} plain={plain_stats} early={early_stats} "
        f"plain_vs_early={plain_vs_early} plain_vs_r11={plain_vs_r11}"
    )
    assert torch.equal(plain.view(torch.int16), r11.view(torch.int16)), plain_vs_r11


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


@pytest.mark.parametrize("m", [9, 14, 28, 56, 112])
def test_wide_mka_is_the_narrow_mka_on_each_8_row_chunk(m):
    """The wide entries (several requests' verify rows in one launch) give the bytes of
    the narrow kernels called on rows [8t, 8t + 8), eagerly and replayed in a CUDA
    graph on new inputs."""
    from sgl_nvidia_mega.qwen3_5.sm103.attention import mka

    g = torch.Generator(device="cuda").manual_seed(m)
    ops = _mka_operands(g)
    step = _MkaStep(m, g)

    def chunked():
        k, v = step.k.clone(), step.v.clone()
        q, gate, out = (torch.empty_like(t) for t in (step.q, step.gate, step.out))
        for s in range(0, m, 8):
            e = min(s + 8, m)
            mka.prologue(
                x=step.x[s:e],
                w_packed=ops["w_qkv"],
                sx=ops["qkv_sx"],
                sw=ops["qkv_sw"],
                q_norm_w=ops["qnw"],
                k_norm_w=ops["knw"],
                cos_sin=ops["cos_sin"],
                positions=step.pos[:, s:e],
                axis_map=ops["axis"],
                cache_loc=step.loc[s:e],
                k_cache=k,
                v_cache=v,
                k_scale=ops["k_scale"],
                v_scale=ops["v_scale"],
                q_out=q[s:e],
                gate_out=gate[s:e],
                nq=HEADS,
                eps=1e-6,
            )
            mka.epilogue(
                attn=step.attn[s:e],
                gate=gate[s:e],
                w_packed=ops["w_o"],
                sx=ops["o_sx"],
                sw=ops["o_sw"],
                out=out[s:e],
                nq=HEADS,
            )
        torch.cuda.synchronize()
        rows = (k[step.loc], v[step.loc])
        return [t.clone().view(torch.uint8) for t in (q, gate, *rows, out)]

    want = chunked()
    step.run(ops)
    assert _same(want, step.bits())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step.run(ops)
    for _ in range(2):
        step.fill(g)
        graph.replay()
        replayed = step.bits()
        assert _same(chunked(), replayed)
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


@pytest.mark.parametrize("rows", [7, 14, 28, 56, 112, 224, 448, 896])
def test_front_with_the_cache_write_is_sglangs_front_then_fp8_kv_write(rows):
    """The front with its FP8 KV write (one launch) gives the bytes of SGLang's front
    followed by its fused FP8 q / k / v + KV write, at every covered width."""
    from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
        fused_qk_gemma_rmsnorm_rope_gate as sglang_front,
    )
    from sglang.kernels.ops.kvcache.fused_fp8_qkv_kv_cache import (
        fused_fp8_qkv_kv_cache,
    )

    from sgl_nvidia_mega.qwen3_5.sm103.draft.front import (
        fused_qk_gemma_rmsnorm_rope_gate,
    )

    g = torch.Generator(device="cuda").manual_seed(300 + rows)
    args, axis, v, pools, loc = _front_operands(rows, g, slots=64 * 32)
    scales = [torch.full((1,), s, device="cuda") for s in (0.8, 1.3)]
    want_pools = tuple(p.clone() for p in pools)
    q, k, gate = sglang_front(*args, has_gate=True, mrope_axis_map=axis)
    q8 = fused_fp8_qkv_kv_cache(
        q.reshape(rows, -1), k.reshape(rows, -1), v, *want_pools, loc, *scales
    )
    got8, _, got_gate = fused_qk_gemma_rmsnorm_rope_gate(
        *args,
        has_gate=True,
        mrope_axis_map=axis,
        v=v,
        kv_cache=pools,
        cache_loc=loc,
        k_scale=scales[0],
        v_scale=scales[1],
    )
    torch.cuda.synchronize()
    assert _bits(q8.reshape(rows, -1), got8.reshape(rows, -1))
    assert _bits(gate.reshape(rows, -1), got_gate.reshape(rows, -1))
    rows_written = loc.long()
    for a, b in zip(want_pools, pools):
        assert _bits(a[rows_written], b[rows_written])


@pytest.mark.parametrize("rows", [7, 14, 28, 56, 112, 224, 448, 896])
def test_gate_quant_is_sglangs_sigmoid_mul_then_static_quant(rows):
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul
    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8

    from sgl_nvidia_mega.qwen3_5.sm103.attention.gate_quant import gate_quant

    g = torch.Generator(device="cuda").manual_seed(400 + rows)
    attn = (torch.randn(rows, HEADS * DIM, device="cuda", generator=g) * 4).bfloat16()
    gate = (torch.randn(rows, HEADS * DIM, device="cuda", generator=g) * 3).bfloat16()
    scale = torch.full((1,), 0.013, device="cuda")
    want, _ = static_quant_fp8(fused_sigmoid_mul(attn, gate), scale)
    assert _bits(want, gate_quant(attn, gate, scale))
