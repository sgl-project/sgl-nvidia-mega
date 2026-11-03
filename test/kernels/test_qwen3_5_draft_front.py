# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot draft_front's attention front (sm103/draft/front.py, 2.2's ATTN_FRONT) against SGLang's
two launches it replaces, on one GPU: the fused q / k Gemma RMSNorm + RoPE + gate split,
then the fused FP8 q / k / v + paged KV write. The FP8 q, the gate and the K / V cache
rows must be the same bytes, at the draft's widths (1, 2 and 7 rows) and query heads a rank
(4 at TP8, 8 at TP4), in eager and in a CUDA graph replayed with new positions and
slots."""

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

KV_HEADS, HEAD_DIM, ROTARY = 1, 256, 64  # the draft layer's shapes at TP8 and TP4
SLOTS = 4096


def _inputs(rows, seed, heads):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g, device=dev) * scale).to(torch.bfloat16)

    q_gate = randn(rows, heads * 2 * HEAD_DIM, scale=2.0)
    k = randn(rows, KV_HEADS * HEAD_DIM, scale=2.0)
    v = randn(rows, KV_HEADS * HEAD_DIM, scale=3.0)
    q_gate[0, :8] = -0.0
    k[-1, 16:32] *= 40
    q_w, k_w = randn(HEAD_DIM, scale=0.1), randn(HEAD_DIM, scale=0.1)
    pos = torch.arange(4096, device=dev, dtype=torch.float32)[:, None]
    freq = 1.0 / (10000 ** (torch.arange(0, ROTARY, 2, device=dev) / ROTARY))
    cos_sin = torch.cat([torch.cos(pos * freq), torch.sin(pos * freq)], dim=-1)
    positions = torch.randint(0, 4000, (rows,), generator=g, device=dev)
    cache_loc = torch.randperm(SLOTS, generator=g, device=dev)[:rows].to(torch.int64)
    k_scale = torch.tensor([0.75 if seed % 2 else 1.0], device=dev)
    v_scale = torch.tensor([1.0], device=dev)
    return q_gate, k, v, q_w, k_w, cos_sin, positions, cache_loc, k_scale, v_scale


def _caches():
    shape = (SLOTS, KV_HEADS, HEAD_DIM)
    return (
        torch.zeros(shape, dtype=torch.float8_e4m3fn, device="cuda"),
        torch.zeros(shape, dtype=torch.float8_e4m3fn, device="cuda"),
    )


def _stock(inputs, kv):
    from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
        fused_qk_gemma_rmsnorm_rope_gate,
    )
    from sglang.kernels.ops.kvcache.fused_fp8_qkv_kv_cache import (
        fused_fp8_qkv_kv_cache,
    )

    q_gate, k, v, q_w, k_w, cos_sin, positions, cache_loc, k_scale, v_scale = inputs
    rows, heads = q_gate.shape[0], q_gate.shape[1] // (2 * HEAD_DIM)
    q, k_out, gate = fused_qk_gemma_rmsnorm_rope_gate(
        q_gate, k, q_w, k_w, cos_sin, positions, 1e-6, heads, KV_HEADS, HEAD_DIM, ROTARY
    )
    q8 = fused_fp8_qkv_kv_cache(
        q=q.view(rows, -1),
        k=k_out.view(rows, -1),
        v=v,
        k_cache=kv[0],
        v_cache=kv[1],
        cache_loc=cache_loc,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    return q8.view(rows, -1), gate.view(rows, -1)


def _plugin(inputs, kv):
    from sgl_nvidia_mega.qwen3_5.sm103.draft.front import (
        fused_qk_gemma_rmsnorm_rope_gate,
    )

    q_gate, k, v, q_w, k_w, cos_sin, positions, cache_loc, k_scale, v_scale = inputs
    rows, heads = q_gate.shape[0], q_gate.shape[1] // (2 * HEAD_DIM)
    q8, k_none, gate = fused_qk_gemma_rmsnorm_rope_gate(
        q_gate,
        k,
        q_w,
        k_w,
        cos_sin,
        positions,
        1e-6,
        heads,
        KV_HEADS,
        HEAD_DIM,
        ROTARY,
        v=v,
        kv_cache=kv,
        cache_loc=cache_loc,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    assert k_none is None
    return q8.view(rows, -1), gate.view(rows, -1)


def _same(a, b):
    return torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


@cuda
@pytest.mark.parametrize("heads", [4, 8])
@pytest.mark.parametrize("rows", [1, 2, 7])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_front_writes_what_sglangs_two_launches_write(rows, seed, heads):
    inputs = _inputs(rows, seed, heads)
    kv_s, kv_p = _caches(), _caches()
    q_s, gate_s = _stock(inputs, kv_s)
    q_p, gate_p = _plugin(inputs, kv_p)
    assert q_p.dtype == torch.float8_e4m3fn
    assert _same(q_s, q_p) and _same(gate_s, gate_p)
    assert _same(kv_s[0], kv_p[0]) and _same(kv_s[1], kv_p[1])


@cuda
@pytest.mark.parametrize("heads", [4, 8])
def test_the_front_replays_in_a_graph_with_new_positions_and_slots(heads):
    rows = 1
    inputs = _inputs(rows, 5, heads)
    kv_s, kv_p = _caches(), _caches()
    static = [t.clone() for t in inputs]
    _plugin(static, kv_p)  # compile outside capture
    _stock(static, kv_s)  # the same warm-up row in the stock caches
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q_p, gate_p = _plugin(static, kv_p)
    for seed in (11, 12, 13):
        fresh = _inputs(rows, seed, heads)
        for dst, src in zip(static, fresh):
            dst.copy_(src)
        graph.replay()
        q_s, gate_s = _stock(fresh, kv_s)
        assert _same(q_s, q_p) and _same(gate_s, gate_p)
        assert _same(kv_s[0], kv_p[0]) and _same(kv_s[1], kv_p[1])
