# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot attn_indexer_logits_q6: DeepGEMM's paged MXFP4 logits with a request's rows in one
Q block equal SGLang's DeepGEMM call bit for bit on every valid logit, at the built
6-row, page-128 variant (one request, short and long contexts whose lengths end mid-page
and mid-split), eager and replayed in a CUDA graph; other shapes are declined."""

import pytest
import torch


def _ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import deep_gemm  # noqa: F401
        from sglang.kernels.ops.attention.dsv4.index_logits import (  # noqa: F401
            deep_gemm_fp4_paged_mqa_logits,
        )
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _ready(), reason="needs SM10x, DeepGEMM and SGLang")

HEADS = 32


def _case(rows, batch, keys, page, seed=0):
    import deep_gemm

    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = torch.device("cuda")
    pages = batch * (keys // page + 4) + 8
    raw = torch.randint(
        0, 256, (pages, page * 68), dtype=torch.uint8, device=dev, generator=g
    )
    raw[:, page * 64 :] = torch.randint(
        118, 132, (pages, page * 4), dtype=torch.uint8, device=dev, generator=g
    )
    k_cache = raw.view(pages, page, 1, 68)
    q_fp4 = torch.randint(
        -128, 127, (batch, rows, HEADS, 64), dtype=torch.int8, device=dev, generator=g
    )
    q_sf = (
        torch.randint(
            118,
            132,
            (batch, rows, HEADS, 4),
            dtype=torch.uint8,
            device=dev,
            generator=g,
        )
        .view(torch.int32)
        .view(batch, rows, HEADS)
    )
    w = (torch.randn(batch * rows, HEADS, device=dev, generator=g) * 0.1).to(
        torch.bfloat16
    )
    lens = torch.tensor(
        [[keys - rows + 1 + i + 101 * b for i in range(rows)] for b in range(batch)],
        dtype=torch.int32,
        device=dev,
    )
    maxlen = int(lens.max())
    npages = -(-maxlen // page)
    perm = torch.randperm(pages, device=dev, generator=g).to(torch.int32)
    table = perm[: batch * npages].view(batch, npages).contiguous()
    meta = deep_gemm.get_paged_mqa_logits_metadata(lens, page, deep_gemm.get_num_sms())
    return (q_fp4, q_sf), k_cache, w, lens, table, meta, maxlen


def _assert_equal(ref, out, lens):
    valid = torch.arange(ref.shape[1], device=ref.device)[None, :] < lens.view(-1, 1)
    assert torch.equal(ref.view(torch.int16)[valid], out.view(torch.int16)[valid])


@pytest.mark.parametrize("keys", [300, 70000])
def test_bitwise_vs_deep_gemm(keys):
    from sglang.kernels.ops.attention.dsv4.index_logits import (
        deep_gemm_fp4_paged_mqa_logits,
    )

    from sgl_nvidia_mega.deepseek_v41.sm103.attention import paged_logits_q6 as q6

    args = _case(6, 1, keys, 128)
    lens = args[3]
    ref = deep_gemm_fp4_paged_mqa_logits(*args)
    _assert_equal(ref, q6.paged_logits(*args), lens)

    out = {}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        q6.paged_logits(*args)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out["l"] = q6.paged_logits(*args)
    args[0][0].copy_(torch.randint_like(args[0][0], -128, 127))
    graph.replay()
    torch.cuda.synchronize()
    _assert_equal(deep_gemm_fp4_paged_mqa_logits(*args), out["l"], lens)


def test_declines_outside_build():
    from sgl_nvidia_mega.deepseek_v41.sm103.attention import paged_logits_q6 as q6

    assert not q6.serves(4, 128) and not q6.serves(8, 128) and not q6.serves(6, 256)
    with pytest.raises(ValueError):
        q6.paged_logits(*_case(4, 1, 1000, 128))
