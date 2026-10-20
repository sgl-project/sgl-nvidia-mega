# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The DeepSeek-V4.1 routed-expert kernels and router on one SM100-family GPU (GB300).

- The route kernels + GEMM pair against FlashInfer's routed MoE (its deferred-finalize
  GEMM2 rows), as SGLang serves it, on the target layer (384 experts, top-6) at
  representative rows (1 / 6 / 16 / 48), including experts chosen by every row (several
  tiles); eager and replayed from a CUDA graph.
- gate_topk.cu against SGLang's moe_fused_gate, bit for bit (ids and weights) at 1-64
  rows, including ties, both softplus branches, NaN logits, padded rows and no
  padded-row count (every row valid).
"""

import ctypes

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41.sm103.moe import routed_expert_gemm as reg

# --------------------------------------------------- routed experts and the target gate

TARGET_EXPERTS, TARGET_TOP_K = 384, 6
SCALE = 1.5
EPS = 1e-20  # topk._RENORMALIZE_SUM_EPSILON, which vision routing passes


def _moe_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import flashinfer  # noqa: F401
        from cuda.bindings import nvrtc  # noqa: F401
    except ImportError:
        return False
    return True


needs_moe = pytest.mark.skipif(not _moe_ready(), reason="needs an SM100-family GPU")


def _shuffled_layer(generator: torch.Generator, experts: int):
    """Random MXFP4 expert weights, shuffled as Mxfp4FlashinferTrtllmMoEMethod does."""
    from flashinfer.fp4_quantization import block_scale_interleave
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    e, h, p, s = experts, reg.HIDDEN, reg.PADDED, reg.SLICE
    w13 = torch.randint(
        0, 256, (e, 2 * p, h // 2), dtype=torch.uint8, generator=generator
    )
    w13_s = torch.randint(
        118, 126, (e, 2 * p, h // 32), dtype=torch.uint8, generator=generator
    )
    w13.view(e, 2, p, -1)[:, :, s:] = 0
    w2 = torch.randint(0, 256, (e, h, p // 2), dtype=torch.uint8, generator=generator)
    w2_s = torch.randint(
        118, 126, (e, h, p // 32), dtype=torch.uint8, generator=generator
    )
    w2[:, :, s // 2 :] = 0
    w13, w13_s, w2, w2_s = (t.cuda() for t in (w13, w13_s, w2, w2_s))
    cache: dict = {}
    g1, g1s, g2, g2s = [], [], [], []
    for i in range(e):
        perm = _maybe_get_cached_w3_w1_permute_indices(cache, w13[i], 128)
        g1.append(w13[i][perm.cuda()].contiguous())
        perm = _maybe_get_cached_w3_w1_permute_indices(
            cache, w13_s[i], 128, num_elts_per_sf=16
        )
        g1s.append(block_scale_interleave(w13_s[i][perm.cuda()].contiguous()))
        perm = get_w2_permute_indices_with_cache(cache, w2[i], 128)
        g2.append(w2[i][perm.cuda()].contiguous())
        perm = get_w2_permute_indices_with_cache(
            cache, w2_s[i], 128, num_elts_per_sf=16
        )
        g2s.append(block_scale_interleave(w2_s[i][perm.cuda()].contiguous()))
    w13 = torch.stack(g1)
    w13_s = torch.stack(g1s).view(torch.float8_e4m3fn).reshape(e, 2 * p, -1)
    w2 = torch.stack(g2)
    w2_s = torch.stack(g2s).view(torch.float8_e4m3fn).reshape(e, h, -1)
    clamp = torch.full((e,), 10.0, dtype=torch.float32, device="cuda")
    return w13, w13_s, w2, w2_s, clamp


def _routing(rows, distinct, generator, experts, top_k):
    distinct = min(max(distinct, top_k), experts)
    pool = torch.randperm(experts, generator=generator)[:distinct]
    ids = [
        pool[torch.randperm(distinct, generator=generator)[:top_k]] for _ in range(rows)
    ]
    return torch.stack(ids).to(torch.int32)


def _flashinfer(x_q, x_s, ids, weights, layer):
    """FlashInfer's routed MoE as SGLang serves it: GEMM2 rows and the slot -> row map."""
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
    from sglang.srt.layers.moe.utils import RoutingMethodType

    w13, s13, w2, s2, clamp = layer
    experts = w13.size(0)
    result = trtllm_fp4_block_scale_routed_moe(
        topk_ids=(ids, weights),
        routing_bias=None,
        routing_method_type=int(RoutingMethodType.TopK),
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        hidden_states=x_q,
        hidden_states_scale=x_s,
        gemm1_weights=w13,
        gemm1_weights_scale=s13,
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=clamp,
        gemm2_weights=w2,
        gemm2_weights_scale=s2,
        gemm2_bias=None,
        output1_scale_scalar=None,
        output1_scale_gate_scalar=None,
        output2_scale_scalar=None,
        num_experts=experts,
        top_k=ids.size(1),
        intermediate_size=reg.PADDED,
        local_expert_offset=0,
        local_num_experts=experts,
        do_finalize=False,
        tune_max_num_tokens=8,
        enable_pdl=True,
    )
    return result[0], result[2].view(ids.shape)


def _gathered(rows_out, permuted):
    return rows_out[permuted.long()].float()


@pytest.fixture(scope="module")
def target_layer():
    return _shuffled_layer(torch.Generator().manual_seed(2), TARGET_EXPERTS)


def _check_pair(layer, *, top_k, rows, distinct, graph, seed, max_rows):
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

    experts = layer[0].size(0)
    generator = torch.Generator().manual_seed(seed)
    gemm = reg.RoutedExpertGemm(*layer, top_k=top_k, max_rows=max_rows)
    x = torch.randn(rows, reg.HIDDEN, generator=generator).to(torch.bfloat16).cuda()
    x_q, x_s = flashinfer_mxfp8_quantize(x, False, alignment=reg.HIDDEN)
    x_s = x_s.view(torch.float8_e4m3fn).reshape(rows, -1)
    ids = _routing(rows, distinct, generator, experts, top_k).cuda()
    weights = torch.softmax(torch.randn(rows, top_k, generator=generator), -1).cuda()

    if graph:
        gemm(x_q, x_s, ids, weights)  # warm-up outside capture
        torch.cuda.synchronize()
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            routed = gemm(x_q, x_s, ids, weights)
        routed[0].zero_()
        for _ in range(3):
            captured.replay()
    else:
        routed = gemm(x_q, x_s, ids, weights)
    torch.cuda.synchronize()

    rows_out, _, permuted = routed
    want_rows, want_permuted = _flashinfer(x_q, x_s, ids, weights, layer)
    torch.cuda.synchronize()
    # Same MMA order, SwiGLU and quantization as the TRT-LLM-gen tactics; one BF16 ulp.
    torch.testing.assert_close(
        _gathered(rows_out, permuted),
        _gathered(want_rows, want_permuted),
        rtol=2**-7,
        atol=1e-6,
    )


@needs_moe
@pytest.mark.parametrize("rows", [1, 6, 16, 48])
@pytest.mark.parametrize("distinct", [6, 384])  # 6: every expert takes all rows
@pytest.mark.parametrize("graph", [False, True])
def test_target_pair_matches_flashinfer(target_layer, rows, distinct, graph):
    _check_pair(
        target_layer,
        top_k=TARGET_TOP_K,
        rows=rows,
        distinct=distinct,
        graph=graph,
        seed=900 + rows + distinct,
        max_rows=reg.SERVED_MAX_ROWS,
    )


# ----------------------------------------------------------------------------- gate top-k


def _stock(logits, bias, num_valid, vl=None):
    from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate

    extra = {}
    if vl is not None:
        bias_alt, input_ids, token = vl
        extra = dict(
            bias_alt=bias_alt,
            input_ids=input_ids,
            bias_alt_token_id=token,
            renormalize_epsilon=EPS,
        )
    w, ids = moe_fused_gate(
        logits,
        bias,
        topk=TARGET_TOP_K,
        scoring_func="sqrtsoftplus",
        num_fused_shared_experts=0,
        renormalize=True,
        routed_scaling_factor=SCALE,
        apply_routed_scaling_factor_on_output=True,
        num_token_non_padded=num_valid,
        sqrtsoftplus_log1p=True,
        **extra,
    )
    return w, ids.to(torch.int32)


def _fast(logits, bias, num_valid, vl=None):
    rows = logits.size(0)
    weights = torch.empty(
        (rows, TARGET_TOP_K), dtype=torch.float32, device=logits.device
    )
    ids = torch.empty((rows, TARGET_TOP_K), dtype=torch.int32, device=logits.device)
    bias_alt, input_ids, token = vl if vl is not None else (None, None, -1)
    reg._launch(
        reg.load_gate_topk(logits.device.index),
        (rows, 1, 1),
        [
            reg._pointer(logits),
            reg._pointer(bias),
            reg._null_or_pointer(bias_alt),
            reg._null_or_pointer(input_ids),
            ctypes.c_int32(input_ids is not None and input_ids.dtype == torch.int64),
            ctypes.c_int64(token),
            reg._null_or_pointer(num_valid),
            reg._pointer(weights),
            reg._pointer(ids),
            ctypes.c_float(SCALE),
            ctypes.c_float(EPS if vl is not None else 0.0),
        ],
        torch.cuda.current_stream().cuda_stream,
    )
    return weights, ids


def _logits(kind, rows, gen):
    shape = (rows, TARGET_EXPERTS)
    if kind == "normal":
        return torch.randn(shape, device="cuda", generator=gen)
    if kind == "wide":  # both softplus branches (> 20 passes through)
        return torch.randn(shape, device="cuda", generator=gen) * 15
    if kind == "ties":  # few distinct values: ties broken by the lowest id
        return torch.randint(-3, 4, shape, device="cuda", generator=gen).float() * 0.5
    if kind == "correlated":  # the rows of one request choose overlapping experts
        base = torch.randn((1, TARGET_EXPERTS), device="cuda", generator=gen)
        return base + 0.2 * torch.randn(shape, device="cuda", generator=gen)
    if kind == "nan":
        x = torch.randn(shape, device="cuda", generator=gen)
        x[torch.rand(shape, device="cuda", generator=gen) < 0.01] = float("nan")
        return x
    raise ValueError(kind)


@needs_moe
@pytest.mark.parametrize("kind", ["normal", "wide", "ties", "nan"])
def test_gate_topk_matches_moe_fused_gate(kind):
    gen = torch.Generator(device="cuda").manual_seed(1234)
    for rows in [*range(1, 49), 64]:
        for trial in range(4):
            logits = _logits(kind, rows, gen).contiguous()
            bias = torch.randn(TARGET_EXPERTS, device="cuda", generator=gen) * 0.05
            valid = rows if trial % 2 == 0 else max(1, rows - 1 - trial)
            num_valid = torch.tensor([valid], dtype=torch.int32, device="cuda")
            if trial == 2:  # no num_token_non_padded: every row is valid
                num_valid = None
            sw, sid = _stock(logits, bias, num_valid)
            fw, fid = _fast(logits, bias, num_valid)
            torch.cuda.synchronize()
            where = f"{kind} rows={rows} valid={valid} trial={trial}"
            assert torch.equal(fid, sid), where
            assert torch.equal(fw.view(torch.int32), sw.view(torch.int32)), where
