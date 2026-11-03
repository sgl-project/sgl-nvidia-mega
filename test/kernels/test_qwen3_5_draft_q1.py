# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The MTP draft's q1 decode attention (SKV2-Q1, sm103/draft/q1.py) on one SM100-family
GPU (GB300): FP8 paged KV of one KV head, 4 query heads (TP8) and 8 (TP4), 64-token
pages, at KV lengths from 1 to 150K:
- against a float64 attention over the same FP8 values, no farther than trtllm-gen
  (FlashInfer's decode as SGLang serves the draft) is, and close to trtllm-gen
  itself (only the fp32 association differs);
- the same bits eager and replayed from a CUDA graph whose KV length changes on the
  device between replays;
- the cache view and BMM scales formed from a draft step's KvPoolBuffers and
  AttnStepMetadata are SGLang's own (paged_kv_view, canonicalize_stride).
The kernel of SGLang branch nvidia/qwen35-agentx-gb300 against this copy, bit for bit, is
checked with that branch installed as ``sglang``, not here."""

from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.qwen3_5.sm103.draft import q1

PAGE, D = q1.PAGE_SIZE, q1.HEAD_DIM
SCALING = D**-0.5  # Qwen3.5's attention scaling
LENGTHS = (1, 255, 256, 4096, 16384, 32768, 131072, 150000)
MAX_LEN = max(LENGTHS)


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


@pytest.fixture(scope="module")
def decode():
    """One rank's plans: every case below shares a layout per head count, so the
    producer compiles once for each (it compiles on a plan's first, eager, use)."""
    return q1.Q1Decode()


def _pool(g, pages: int):
    """K and V pool buffers of one KV head, slot-major [slots, 1, D] FP8, as the
    draft's MHATokenToKVPool keeps them (slot 0's page absorbs padding writes)."""
    slots = (pages + 1) * PAGE
    k = (torch.randn(slots, 1, D, device="cuda", generator=g) * 2.0).to(
        torch.float8_e4m3fn
    )
    v = (torch.randn(slots, 1, D, device="cuda", generator=g) * 2.0).to(
        torch.float8_e4m3fn
    )
    return k, v


def _case(seed: int, heads: int, length: int, k_scale: float = 1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    width = -(-MAX_LEN // PAGE)  # the graph page table's static width
    k, v = _pool(g, width)
    pages = -(-length // PAGE)
    table = torch.zeros(1, width, dtype=torch.int32, device="cuda")
    # pages scattered over the pool, never page 0
    table[0, :pages] = (
        torch.randperm(width, device="cuda", generator=g)[:pages].int() + 1
    )
    query = (torch.randn(1, heads, D, device="cuda", generator=g) * 2.0).to(
        torch.float8_e4m3fn
    )
    lengths = torch.tensor([length], dtype=torch.int32, device="cuda")
    kv = SimpleNamespace(
        layer_ids=(3,), k=(k,), v=(v,), k_scale=(k_scale,), v_scale=(1.0,)
    )
    step = SimpleNamespace(page_table=table, cache_seqlens=lengths)
    return query, kv, step


def _reference(query, kv, step, scale):
    """float64 attention over the FP8 values: softmax(q k^T * scale) v."""
    length = int(step.cache_seqlens[0])
    pages = step.page_table[0, : -(-length // PAGE)].long()
    slots = (pages[:, None] * PAGE + torch.arange(PAGE, device="cuda")).flatten()
    slots = slots[:length]

    def rows(buf):  # gather as bytes: FP8 tensors do not index on every torch
        raw = buf.view(torch.uint8).view(-1, D)[slots]
        return raw.view(torch.float8_e4m3fn).float().double()

    kk, vv = rows(kv.k[0]), rows(kv.v[0])
    s = query[0].float().double() @ kk.T * scale
    return (torch.softmax(s, dim=-1) @ vv).unsqueeze(0)


def _trtllm_gen(query, kv_cache, step, bmm1, bmm2):
    """FlashInfer's trtllm-gen decode with SGLang's arguments (the draft's stock arm)."""
    import flashinfer
    from sglang.srt.layers.attention.trtllm_mla_backend import (
        make_persistent_multi_ctas_kv_counter_buffer,
    )

    workspace = torch.zeros(128 * 2**20, dtype=torch.uint8, device="cuda")
    counter = make_persistent_multi_ctas_kv_counter_buffer(
        torch.device("cuda"), num_q_heads=32, max_batch_size=8
    )
    kwargs = dict(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace,
        block_tables=step.page_table,
        seq_lens=step.cache_seqlens,
        max_seq_len=step.page_table.shape[1] * PAGE,
        bmm1_scale=bmm1,
        bmm2_scale=bmm2,
        window_left=-1,
        sinks=None,
        out_dtype=torch.bfloat16,
    )
    try:
        return flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            **kwargs, multi_ctas_kv_counter_buffer=counter
        )
    except TypeError:
        return flashinfer.decode.trtllm_batch_decode_with_kv_cache(**kwargs)


@pytest.mark.parametrize("heads", [4, 8])
@pytest.mark.parametrize("length", LENGTHS)
def test_q1_against_float64_and_trtllm_gen(decode, heads, length):
    query, kv, step = _case(100 * heads + length % 97, heads, length, k_scale=0.75)
    got = decode.step(query, kv, 3, step, scaling=SCALING)
    assert got.dtype == torch.bfloat16 and got.shape == (1, heads, D)
    assert torch.isfinite(got).all()
    bmm1, bmm2 = q1.bmm_scales(k_scale=0.75, v_scale=1.0, scaling=SCALING)
    ref = _reference(query, kv, step, bmm1)
    stock = _trtllm_gen(query, q1.paged_kv_cache(kv.k[0], kv.v[0]), step, bmm1, bmm2)
    err = (got.double() - ref).abs().max().item()
    err_stock = (stock.double() - ref).abs().max().item()
    # the FP8 rounding of P dominates both errors; Q1 rounds where trtllm-gen does
    assert err <= 1.5 * err_stock + 1e-3, (err, err_stock)
    gap = (got.float() - stock.float()).abs().max().item()
    assert gap <= 2**-6 * max(1.0, ref.abs().max().item()), gap


def test_q1_graph_replay_follows_the_device_length(decode):
    """One capture, the KV length and pages rewritten on the device before each
    replay."""
    query, kv, step = _case(7, 4, 20000)
    out = torch.empty(1, 4, D, dtype=torch.bfloat16, device="cuda")
    decode.step(query, kv, 3, step, scaling=SCALING, out=out)  # compiles, eagerly
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode.step(query, kv, 3, step, scaling=SCALING, out=out)
    for length in (20000, 131072, 70001, 16385, 150000, 300):
        pages = -(-length // PAGE)
        g = torch.Generator(device="cuda").manual_seed(length)
        order = torch.randperm(step.page_table.shape[1], device="cuda", generator=g)
        step.page_table[0, :pages] = order[:pages].int() + 1
        step.cache_seqlens.fill_(length)
        graph.replay()
        torch.cuda.synchronize()
        replayed = out.clone()
        eager = decode.step(query, kv, 3, step, scaling=SCALING)
        assert torch.equal(replayed.view(torch.int16), eager.view(torch.int16)), length


def test_q1_repeats_bit_for_bit(decode):
    query, kv, step = _case(9, 4, 131072)
    a = decode.step(query, kv, 3, step, scaling=SCALING)
    b = decode.step(query, kv, 3, step, scaling=SCALING)
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("hnd", [False, True])
def test_cache_view_is_sglangs(hnd):
    """The view from a pool buffer equals the backend's _reshape_paged_kv_cache
    (SGLang's paged_kv_view, permute, canonicalize_stride), slot-major or HND."""
    from sglang.kernels.ops.attention.utils import canonicalize_stride
    from sglang.srt.mem_cache.layout.paged_view import paged_kv_view

    pages = 33
    if hnd:
        buf = torch.zeros(pages, 1, PAGE, D, dtype=torch.float8_e4m3fn, device="cuda")
    else:
        buf = torch.zeros(pages * PAGE, 1, D, dtype=torch.float8_e4m3fn, device="cuda")
    want = canonicalize_stride(paged_kv_view(buf, PAGE, 1, D).permute(0, 2, 1, 3))
    got, _ = q1.paged_kv_cache(buf, buf)
    assert (got.shape, got.stride(), got.data_ptr()) == (
        want.shape,
        want.stride(),
        want.data_ptr(),
    )
    assert (
        q1.unsupported_reason(
            torch.zeros(1, 4, D, dtype=torch.float8_e4m3fn, device="cuda"),
            (got, got),
            torch.zeros(1, 4, dtype=torch.int32, device="cuda"),
            torch.ones(1, dtype=torch.int32, device="cuda"),
            bmm1_scale=SCALING,
            bmm2_scale=1.0,
        )
        is None
    )


def test_unsupported_reasons(decode):
    query, kv, step = _case(3, 4, 4096)
    kv_cache = q1.paged_kv_cache(kv.k[0], kv.v[0])
    ok = dict(bmm1_scale=SCALING, bmm2_scale=1.0)
    args = (query, kv_cache, step.page_table, step.cache_seqlens)
    assert q1.unsupported_reason(*args, **ok) is None
    assert q1.unsupported_reason(*args, bmm1_scale=SCALING, bmm2_scale=0.5)
    assert q1.unsupported_reason(*args, **ok, page_size=32)
    assert q1.unsupported_reason(*args, **ok, window_left=128)
    assert q1.unsupported_reason(*args, **ok, attn_dcp_size=2)
    assert q1.unsupported_reason(query.view(torch.uint8), *args[1:], **ok)
    two = torch.ones(2, dtype=torch.int32, device="cuda")
    assert q1.unsupported_reason(query, kv_cache, step.page_table, two, **ok)
    with pytest.raises(ValueError):
        decode.step(
            query,
            SimpleNamespace(**{**vars(kv), "v_scale": (0.5,)}),
            3,
            step,
            scaling=SCALING,
        )


def test_build_and_enabled():
    device = torch.cuda.current_device()
    q1.build(device, 4)
    assert q1.enabled_reason(device_index=device, max_context_len=262144) is None
    assert q1.enabled_reason(device_index=device, max_context_len=4096)
