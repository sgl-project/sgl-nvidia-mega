# SPDX-License-Identifier: Apache-2.0
"""The QSA prefill kernels runtime/qsa_prefill.py runs, against SGLang main's stock
path on an SM10x GPU.

Bitwise where the source claims it: the row pack (``pack_rows``) against
main's ``pack_qsa_prefill_kv``, the row-tiled QK norm + RoPE + gate (``qk_rows``) against
main's ``fused_qk_gemma_rmsnorm_rope_gate``, the scale-folded relative scorer against
main's ``qsa_mqa_prefill_relative``, and the TMA packed prefill (passes stock runs on its
16-token tile) against main's chunk kernel. The fast prefill select picks the blocks
main's relative selection picks (as sets per row). The union attention matches main's
chunk and cold kernels to a tolerance (declared, not measured; record the errors). The
runtime's attention routing (``qsa_prefill.attention``) is checked end to end on a
minimal pool against main's kernels in main's order.
"""

import dataclasses
from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.qwen3_8.runtime import qsa_prefill as qp

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="the QSA prefill kernels are SM10x only",
)

BF16 = torch.bfloat16


def _bits(a, b):
    return a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def _relative_indices(extend_lens, prefixes, topk, seed=0):
    """Per row, ``topk`` logical positions among the row's visible tokens (relative to
    its request's first context token), -1 past them, as the indexer expands them."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    blocks = []
    for extend, before in zip(extend_lens, prefixes):
        visible = torch.arange(extend, device="cuda") + before + 1
        scores = torch.rand(extend, before + extend, generator=gen, device="cuda")
        cols = torch.arange(before + extend, device="cuda")
        scores = torch.where(cols[None, :] < visible[:, None], scores, -1.0)
        picked = scores.topk(min(topk, before + extend), dim=1).indices
        picked = torch.where(
            torch.arange(picked.shape[1], device="cuda")[None, :] < visible[:, None],
            picked,
            -1,
        )
        if picked.shape[1] < topk:
            pad = picked.new_full((extend, topk - picked.shape[1]), -1)
            picked = torch.cat([picked, pad], dim=1)
        blocks.append(picked)
    return torch.cat(blocks).to(torch.int32).contiguous()


def _cu(lengths):
    return torch.tensor([0, *accumulate(lengths)], device="cuda", dtype=torch.int32)


@pytest.mark.parametrize(
    "extend_lens,prefixes",
    [([1100], [0]), ([700, 600], [4096, 300]), ([1024, 1, 513], [0, 9000, 77])],
)
def test_union_prefill_matches_the_stock_chunk_kernel(extend_lens, prefixes):
    from sglang.srt.layers.attention.qsa.sparse_attn import (
        sparse_gqa_fwd_interface_triton_ck,
    )

    from sgl_nvidia_mega.qwen3_8.qsa import union_prefill

    if not union_prefill._GLUON:
        pytest.skip("this Triton has no Gluon Blackwell API")
    torch.manual_seed(23)
    heads, kv_heads, dim, topk = 24, 2, 256, 2051
    kv_lens = [e + p for e, p in zip(extend_lens, prefixes)]
    rows = sum(extend_lens)
    q = torch.randn(rows, heads, dim, device="cuda", dtype=BF16)
    k = torch.randn(sum(kv_lens), kv_heads, dim, device="cuda", dtype=BF16)
    v = torch.randn_like(k)
    indices = _relative_indices(extend_lens, prefixes, topk)
    cu_q, cu_k = _cu(extend_lens), _cu(kv_lens)
    lengths = torch.tensor(kv_lens, device="cuda", dtype=torch.int32)
    scale = dim**-0.5
    assert union_prefill.applies(
        q,
        indices,
        kv_heads=kv_heads,
        kv_dtype=BF16,
        extend_lens=extend_lens,
        kv_max=max(kv_lens),
        total_k=k.shape[0],
    )
    expected = sparse_gqa_fwd_interface_triton_ck(
        q, k, v, indices, cu_q, cu_k, lengths, scale
    )
    actual = union_prefill.union_prefill(
        q,
        k,
        v,
        indices,
        cu_q,
        cu_k,
        lengths,
        scale,
        extend_lens=extend_lens,
        kv_max=max(kv_lens),
    )
    report = union_prefill.compare(actual, expected)
    print("union vs stock chunk:", report)
    assert report["nonfinite"] == report["ref_nonfinite"] == 0
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_union_cold_prefill_matches_the_stock_kernel():
    from sglang.srt.layers.attention.qsa.sparse_attn import (
        sparse_gqa_fwd_interface_triton,
    )

    from sgl_nvidia_mega.qwen3_8.qsa import union_prefill

    if not union_prefill._GLUON:
        pytest.skip("this Triton has no Gluon Blackwell API")
    torch.manual_seed(29)
    extend_lens = [800, 400]
    heads, kv_heads, dim, topk = 24, 2, 256, 2051
    rows = sum(extend_lens)
    q = torch.randn(rows, heads, dim, device="cuda", dtype=BF16)
    k = torch.randn(rows, kv_heads, dim, device="cuda", dtype=BF16)
    v = torch.randn_like(k)
    indices = _relative_indices(extend_lens, [0, 0], topk, seed=1)
    cu_q = _cu(extend_lens)
    lens = torch.tensor(extend_lens, device="cuda", dtype=torch.int32)
    scale = dim**-0.5
    expected = sparse_gqa_fwd_interface_triton(
        q, k, v, max(extend_lens), indices, cu_q, scale
    )
    actual = union_prefill.union_prefill(
        q,
        k,
        v,
        indices,
        cu_q,
        cu_q,
        lens,
        scale,
        extend_lens=extend_lens,
        kv_max=max(extend_lens),
    )
    print("union vs stock cold:", union_prefill.compare(actual, expected))
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("dtype", [BF16, torch.float8_e4m3fn])
def test_pack_rows_is_bitwise_the_stock_pack(compact, dtype):
    from sglang.srt.layers.attention.qsa.sparse_attn import pack_qsa_prefill_kv

    from sgl_nvidia_mega.qwen3_8.qsa import prefill_pack

    torch.manual_seed(5)
    slots, heads, dim = 4096, 2, 256
    k = torch.randn(slots, heads, dim, device="cuda").to(dtype)
    v = torch.randn(slots, heads, dim, device="cuda").to(dtype)
    lens = [1500] + [3] * 4000 if compact else [700, 1, 333, 64]
    table = torch.randint(0, slots, (len(lens), max(lens)), device="cuda").to(
        torch.int32
    )
    kwargs = dict(
        k=k,
        v=v,
        req_to_token=table,
        req_indices=torch.arange(len(lens), device="cuda", dtype=torch.int64),
        cu_k=_cu(lens),
        total_k=sum(lens),
        max_k=max(lens),
        output_dtype=BF16,
    )
    assert prefill_pack.applies(k, v, table)
    ref = pack_qsa_prefill_kv(**kwargs)
    new = prefill_pack.pack_rows(**kwargs)
    assert _bits(new[0], ref[0]) and _bits(new[1], ref[1])


def _qk_inputs(tokens, heads=24, kv_heads=2, dim=256, rotary=64, mrope=False):
    torch.manual_seed(7)
    qkv = torch.randn(tokens, 2 * heads * dim + 2 * kv_heads * dim, device="cuda")
    qkv = qkv.to(BF16)
    q_gate, k, _ = qkv.split([2 * heads * dim, kv_heads * dim, kv_heads * dim], -1)
    cache = torch.randn(65536, rotary, device="cuda", dtype=BF16)
    if mrope:
        positions = torch.randint(0, 65536, (3, tokens), device="cuda")
        axis_map = torch.randint(0, 3, (rotary // 2,), device="cuda", dtype=torch.int32)
    else:
        positions = torch.randint(0, 65536, (tokens,), device="cuda")
        axis_map = None
    q_w = torch.randn(dim, device="cuda", dtype=BF16) * 0.1
    k_w = torch.randn(dim, device="cuda", dtype=BF16) * 0.1
    return q_gate, k, q_w, k_w, cache, positions, axis_map


@pytest.mark.parametrize("tokens,mrope", [(1024, False), (4097, False), (2048, True)])
def test_qk_rows_is_bitwise_the_stock_kernel(tokens, mrope):
    """Bitwise as the source's Triton compiles the stock kernel (its claim)."""
    from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
        fused_qk_gemma_rmsnorm_rope_gate,
    )

    from sgl_nvidia_mega.qwen3_8.kernels.ops.attention import (
        fused_qk_rmsnorm_rope_gate_rows as rows,
    )

    if not rows._HAS_GLUON:
        pytest.skip("this Triton has no Gluon")
    q_gate, k, q_w, k_w, cache, positions, axis_map = _qk_inputs(tokens, mrope=mrope)
    assert rows.applies(q_gate, k, positions, tokens, 256, 64, True)
    ref = fused_qk_gemma_rmsnorm_rope_gate(
        q_gate,
        k,
        q_w,
        k_w,
        cache,
        positions,
        1e-6,
        24,
        2,
        256,
        64,
        has_gate=True,
        mrope_axis_map=axis_map,
    )
    q_out = torch.empty(tokens, 24 * 256, device="cuda", dtype=BF16)
    k_out = torch.empty(tokens, 2 * 256, device="cuda", dtype=BF16)
    gate_out = torch.empty(tokens, 24, 256, device="cuda", dtype=BF16)
    rows.launch(
        q_gate=q_gate,
        k=k,
        q_out=q_out,
        k_out=k_out,
        gate_out=gate_out,
        q_weight=q_w,
        k_weight=k_w,
        cos_sin_cache=cache,
        positions=positions,
        mrope_axis_map=axis_map,
        eps=1e-6,
        num_q_heads=24,
        num_kv_heads=2,
    )
    assert _bits(q_out, ref[0]) and _bits(k_out, ref[1]) and _bits(gate_out, ref[2])


@pytest.mark.parametrize("rows,width", [(37, 600), (1000, 30000)])
def test_scale_fold_is_bitwise_the_stock_scorer(rows, width):
    from sglang.srt.layers.attention.qsa.mqa import qsa_mqa_prefill_relative

    from sgl_nvidia_mega.qwen3_8.qsa import prefill_scale_fold

    if not prefill_scale_fold.HAS_TILELANG:
        pytest.skip("TileLang is unavailable")
    torch.manual_seed(11)
    q = torch.randn(rows, 4, 128, device="cuda", dtype=BF16)
    keys = torch.randn(width + 64, 1, 128, device="cuda", dtype=BF16)
    starts = torch.randint(0, 64, (rows,), device="cuda", dtype=torch.int32)
    lengths = torch.randint(1, width + 1, (rows,), device="cuda", dtype=torch.int32)
    ends = starts + lengths
    ref = qsa_mqa_prefill_relative(q, keys, starts, ends, width)
    new = prefill_scale_fold.tilelang_qsa_mqa_prefill_relative(
        q, keys, starts, ends, width, scale_fold=True
    )
    cols = torch.arange(width, device="cuda")[None, :]
    own = cols < lengths[:, None]  # columns past a row's length are undefined
    assert _bits(new[own], ref[own])


def _indexer():
    return SimpleNamespace(
        layer_id=0,
        index_n_heads=4,
        index_kv_heads=1,
        index_head_dim=128,
        compress_ratio=4,
        block_topk=512,
        token_topk=2048,
    )


@pytest.mark.parametrize(
    "options",
    [
        qp.HIGH_CONCURRENCY,
        dataclasses.replace(qp.HIGH_CONCURRENCY, scorer_tc=False),
        qp.MID_CONCURRENCY,
    ],
    ids=["high", "high-fp32-scorer", "mid"],
)
def test_fast_select_picks_the_stock_blocks(options):
    """The plugin's select against main's relative selection: the same token sets per
    row. Positive queries and keys keep every score above the ReLU's zero, so no row's
    512th block score is tied (Gaussian inputs tie many blocks at zero, and both
    selections then pick among them run to run)."""
    from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer

    from sgl_nvidia_mega.qwen3_8.qsa import prefill_select

    if not prefill_select.HAS_TILELANG:
        pytest.skip("TileLang is unavailable")
    torch.manual_seed(13)
    ix = _indexer()
    requests = [(900, 1200), (300, 40)]  # (rows, prefix blocks)
    q = torch.rand(sum(r for r, _ in requests), 4, 128, device="cuda", dtype=BF16)
    total_keys = sum(p + r // 4 + 1 for r, p in requests)
    keys = torch.rand(total_keys, 1, 128, device="cuda", dtype=BF16)
    starts, ends, positions, lengths = [], [], [], []
    base = 0
    for rows, prefix in requests:
        for r in range(rows):
            position = prefix * 4 + r
            starts.append(base)
            ends.append(base + (position + 1) // 4)
            positions.append(position)
            lengths.append(prefix * 4 + rows)
        base += prefix + rows // 4 + 1
    as_i32 = lambda xs: torch.tensor(xs, device="cuda", dtype=torch.int32)  # noqa: E731
    starts, ends, positions, lengths = map(as_i32, (starts, ends, positions, lengths))
    width = int((ends - starts).max())
    ref = QSAIndexer.select_prefill_tokens(
        ix, q, keys, starts, ends, positions, lengths, max_row_keys=width
    )
    out = torch.empty_like(ref)
    capacity = qp.compressed_capacity(32768, 16)
    assert qp.fast_select_reason(ix, q, keys, out, width, capacity) == ""
    qp.fast_select(options, q, keys, starts, ends, positions, lengths, out, width)
    for a, b in zip(out.sort(dim=1).values, ref.sort(dim=1).values):
        assert torch.equal(a, b)


def test_tma_prefill_is_bitwise_the_stock_chunk_kernel():
    from sglang.srt.layers.attention.qsa.sparse_attn import (
        _get_best_config,
        sparse_gqa_fwd_interface_triton_ck,
    )

    torch.manual_seed(733)
    extend_lens, prefixes = [257, 513], [1024, 2048]
    rows = sum(extend_lens)
    assert _get_best_config(rows) == (16, 1, 2)
    kv_lens = [e + p for e, p in zip(extend_lens, prefixes)]
    q = torch.randn(rows, 24, 256, device="cuda", dtype=BF16)
    k = torch.randn(sum(kv_lens), 2, 256, device="cuda", dtype=BF16)
    v = torch.randn_like(k)
    indices = _relative_indices(extend_lens, prefixes, 2051, seed=3)
    cu_q, cu_k = _cu(extend_lens), _cu(kv_lens)
    lengths = torch.tensor(kv_lens, device="cuda", dtype=torch.int32)
    fb = SimpleNamespace(extend_seq_lens=torch.tensor(extend_lens))
    attn = SimpleNamespace(scaling=0.0625)
    ref = sparse_gqa_fwd_interface_triton_ck(
        q, k, v, indices, cu_q, cu_k, lengths, 0.0625, max_q=max(extend_lens)
    )
    out = qp.tma_prefill(
        qp.DEFAULT, q, attn, fb, k, v, indices, cu_q, cu_k, lengths, extend_lens, 4
    )
    assert out is not None and _bits(out, ref)


class _St:
    def __init__(self, fb):
        self.fb = fb
        self.ran_ops = []

    def ran(self, op, impl):
        self.ran_ops.append((op, impl))


def _pool_backend(slots, heads, dim, requests, seq_lens):
    """A KV pool, request table and the backend attributes qsa_prefill.attention reads."""
    k_cache = torch.zeros(slots, heads, dim, device="cuda", dtype=BF16)
    v_cache = torch.zeros_like(k_cache)
    table = torch.zeros(requests, max(seq_lens), device="cuda", dtype=torch.int32)
    perm = torch.randperm(slots - 1, device="cuda")[: sum(seq_lens)] + 1
    start = 0
    for r, n in enumerate(seq_lens):
        table[r, :n] = perm[start : start + n].to(torch.int32)
        start += n

    class Layer:
        def store(self, attn, loc, k, v):
            k_cache[loc] = k.to(k_cache.dtype)
            v_cache[loc] = v.to(v_cache.dtype)

    kv = SimpleNamespace(key=lambda: k_cache, value=lambda: v_cache)
    backend = SimpleNamespace(
        kv_layers=SimpleNamespace(layer=lambda layer_id: Layer()),
        token_to_kv_pool=SimpleNamespace(
            layer_router=SimpleNamespace(read=lambda layer_id: kv)
        ),
        req_to_token_pool=SimpleNamespace(req_to_token=table),
        compress_ratio=4,
        _pad_extend_output=lambda out, rows: out.reshape(out.shape[0], -1),
    )
    return backend, table, k_cache, v_cache


@pytest.mark.parametrize("options", [qp.HIGH_CONCURRENCY, qp.MID_CONCURRENCY])
def test_attention_routing_against_mains_order(options):
    """qsa_prefill.attention on a prefix-chunk pass: the store, then the pack and the
    union (rows >= 1,024) or TMA / chunk kernel, against main's store + pack + chunk
    kernel. Bitwise without the union; to a tolerance with it."""
    from sglang.srt.layers.attention.qsa.sparse_attn import (
        pack_qsa_prefill_kv,
        sparse_gqa_fwd_interface_triton_ck,
    )

    torch.manual_seed(41)
    extend_lens, prefixes = [800, 300], [1000, 64]
    seq_lens = [e + p for e, p in zip(extend_lens, prefixes)]
    rows, heads, kv_heads, dim = sum(extend_lens), 24, 2, 256
    backend, table, k_cache, v_cache = _pool_backend(8192, kv_heads, dim, 2, seq_lens)
    k_cache.normal_()
    v_cache.normal_()
    loc = torch.cat(
        [
            table[r, p : p + e].long()
            for r, (e, p) in enumerate(zip(extend_lens, prefixes))
        ]
    )
    fb = SimpleNamespace(
        out_cache_loc=loc,
        extend_seq_lens=torch.tensor(extend_lens, device="cuda", dtype=torch.int32),
        extend_seq_lens_cpu=extend_lens,
        seq_lens_cpu=seq_lens,
        req_pool_indices=torch.arange(2, device="cuda"),
        _attn_output=None,
    )
    attn = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=heads,
        tp_k_head_num=kv_heads,
        tp_v_head_num=kv_heads,
        qk_head_dim=dim,
        v_head_dim=dim,
        head_dim=dim,
        scaling=dim**-0.5,
    )
    q = torch.randn(rows, heads * dim, device="cuda", dtype=BF16)
    k = torch.randn(rows, kv_heads * dim, device="cuda", dtype=BF16)
    v = torch.randn_like(k)
    indices = _relative_indices(extend_lens, prefixes, 2051, seed=4)
    # main: the store, the pack, the chunk kernel
    k_ref, v_ref = k_cache.clone(), v_cache.clone()
    k_ref[loc] = k.view(rows, kv_heads, dim)
    v_ref[loc] = v.view(rows, kv_heads, dim)
    seq = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)
    packed = pack_qsa_prefill_kv(
        k=k_ref,
        v=v_ref,
        req_to_token=table,
        req_indices=fb.req_pool_indices,
        cu_k=_cu(seq_lens),
        total_k=sum(seq_lens),
        max_k=max(seq_lens),
        output_dtype=BF16,
    )
    ref = sparse_gqa_fwd_interface_triton_ck(
        q.view(rows, heads, dim),
        *packed,
        indices,
        _cu(extend_lens),
        _cu(seq_lens),
        seq,
        attn.scaling,
    ).reshape(rows, -1)
    st = _St(fb)
    out = qp.attention(options, attn, backend, q, k, v, indices, st)
    assert torch.equal(k_cache, k_ref) and torch.equal(v_cache, v_ref)
    print("attention routing:", st.ran_ops)
    if ("qsa", "union") in st.ran_ops:
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
    else:
        assert _bits(out, ref)
    without = qp.Options(union=False, pack_rows=options.pack_rows)
    out = qp.attention(without, attn, backend, q, k, v, indices, _St(fb))
    assert _bits(out, ref)
