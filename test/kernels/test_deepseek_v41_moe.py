# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The DeepSeek-V4.1 routed-expert kernels and routers on one SM100-family GPU (GB300).

- The route kernels + GEMM pair against FlashInfer's routed MoE (its deferred-finalize
  GEMM2 rows), as SGLang serves it, on the target layer (384 experts, top-6) at 1-48
  rows, including experts chosen by more than 8 rows (several tiles), and on the DSpark
  draft layer (128 experts, top-3) at 1-8 rows; eager and replayed from a CUDA graph.
- gate_topk.cu against SGLang's moe_fused_gate, bit for bit (ids and weights) at 1-64
  rows, including ties, both softplus branches, NaN logits, padded rows, no padded-row
  count (every row valid) and the vision-token routing.
- gate_topk.cuh's two routers, the target's (384 experts, top-6, gate_topk.cu) and the
  DSpark draft's (128 experts, top-3, gate_topk_draft.cu), against moe_fused_gate bit
  for bit at the fast-gate widths (1-16 and 48 rows) and both routed scales; the ids
  also against torch's top-k, and replayed from a CUDA graph.
"""

import ctypes

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41.sm103.moe import routed_expert_gemm as reg

# --------------------------------------------------- routed experts and the target gate

TARGET_EXPERTS, TARGET_TOP_K = 384, 6
DRAFT_EXPERTS, DRAFT_TOP_K = 128, 3
SCALE = 1.5
EPS = 1e-20  # topk._RENORMALIZE_SUM_EPSILON, which vision routing passes
IMAGE_TOKEN = 128815  # arbitrary; only equality with the token ids matters


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


@pytest.fixture(scope="module")
def draft_layer():
    return _shuffled_layer(torch.Generator().manual_seed(1), DRAFT_EXPERTS)


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
@pytest.mark.parametrize("rows", [1, 2, 6, 8, 9, 12, 16, 24, 40, 48])
@pytest.mark.parametrize("distinct", [6, 20, 384])  # 6: every expert takes all rows
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


@needs_moe
@pytest.mark.parametrize("rows", [1, 5, 8])
@pytest.mark.parametrize("graph", [False, True])
def test_draft_pair_matches_flashinfer(draft_layer, rows, graph):
    _check_pair(
        draft_layer,
        top_k=DRAFT_TOP_K,
        rows=rows,
        distinct=12,
        graph=graph,
        seed=500 + rows,
        max_rows=reg.MAX_ROWS,
    )


@needs_moe
def test_steps_past_the_bound_are_declined(target_layer):
    gemm = reg.RoutedExpertGemm(
        *target_layer, top_k=TARGET_TOP_K, max_rows=reg.MAX_ROWS
    )
    assert gemm.route_wide is None
    rows = 9
    x_q = torch.zeros(rows, reg.HIDDEN, dtype=torch.uint8, device="cuda")
    x_s = torch.zeros(rows, reg.HIDDEN // 32, dtype=torch.uint8, device="cuda")
    assert not gemm.supports_inputs(x_q, x_s)


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
@pytest.mark.parametrize("kind", ["normal", "wide", "ties", "correlated", "nan"])
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


@needs_moe
@pytest.mark.parametrize("rows", [6, 12, 16, 48])
@pytest.mark.parametrize("count", [True, False])
def test_gate_topk_follows_live_inputs_in_a_graph(rows, count):
    gen = torch.Generator(device="cuda").manual_seed(7 + rows)
    logits = torch.randn((rows, TARGET_EXPERTS), device="cuda", generator=gen)
    bias = torch.randn(TARGET_EXPERTS, device="cuda", generator=gen) * 0.05
    num_valid = (
        torch.tensor([rows], dtype=torch.int32, device="cuda") if count else None
    )
    _fast(logits, bias, num_valid)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _fast(logits, bias, num_valid)
    for _ in range(3):
        logits.copy_(torch.randn((rows, TARGET_EXPERTS), device="cuda", generator=gen))
        graph.replay()
        ref = _stock(logits, bias, num_valid)
        torch.cuda.synchronize()
        for got, want in zip(out, ref):
            assert torch.equal(got.view(torch.int32), want.view(torch.int32))


@needs_moe
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
def test_gate_topk_vision_routing_matches_moe_fused_gate(ids_dtype):
    gen = torch.Generator(device="cuda").manual_seed(99)
    for rows in [1, 6, 7, 12, 16, 24, 48, 64]:
        for trial in range(4):
            logits = _logits("normal", rows, gen).contiguous()
            bias = torch.randn(TARGET_EXPERTS, device="cuda", generator=gen) * 0.05
            bias_alt = torch.randn(TARGET_EXPERTS, device="cuda", generator=gen) * 0.5
            input_ids = torch.randint(0, 200000, (rows,), device="cuda", generator=gen)
            input_ids[torch.rand(rows, device="cuda", generator=gen) < 0.4] = (
                IMAGE_TOKEN
            )
            input_ids = input_ids.to(ids_dtype)
            valid = rows if trial % 2 == 0 else max(1, rows - 2)
            num_valid = torch.tensor([valid], dtype=torch.int32, device="cuda")
            if trial == 2:
                num_valid = None
            vl = (bias_alt, input_ids, IMAGE_TOKEN)
            sw, sid = _stock(logits, bias, num_valid, vl)
            fw, fid = _fast(logits, bias, num_valid, vl)
            torch.cuda.synchronize()
            where = f"rows={rows} valid={valid} {ids_dtype}"
            assert torch.equal(fid, sid), where
            assert torch.equal(fw.view(torch.int32), sw.view(torch.int32)), where


@needs_moe
def test_gate_topk_then_pair_matches_the_two_steps(target_layer):
    """gate_topk_and_gemms equals moe_fused_gate's routing fed to the pair."""
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

    gemm = reg.RoutedExpertGemm(
        *target_layer,
        top_k=TARGET_TOP_K,
        max_rows=reg.SERVED_MAX_ROWS,
        gate_topk=True,
    )
    gen = torch.Generator(device="cuda").manual_seed(3)
    for rows, count in ((1, True), (6, False), (12, False), (16, True), (48, False)):
        logits = torch.randn((rows, TARGET_EXPERTS), device="cuda", generator=gen)
        bias = torch.randn(TARGET_EXPERTS, device="cuda", generator=gen) * 0.05
        num_valid = (
            torch.tensor([rows], dtype=torch.int32, device="cuda") if count else None
        )
        x = torch.randn(rows, reg.HIDDEN, device="cuda", generator=gen).to(
            torch.bfloat16
        )
        x_q, x_s = flashinfer_mxfp8_quantize(x, False, alignment=reg.HIDDEN)
        x_s = x_s.view(torch.float8_e4m3fn).reshape(rows, -1)
        assert gemm.supports_gate_topk(logits, bias) and gemm.supports_inputs(x_q, x_s)
        rows_out, weights, permuted, ids = gemm.gate_topk_and_gemms(
            x_q, x_s, logits, bias, SCALE, num_valid
        )
        sw, sid = _stock(logits, bias, num_valid)
        want_rows, want_weights, want_permuted = gemm(x_q, x_s, sid, sw)
        torch.cuda.synchronize()
        assert torch.equal(ids, sid) and torch.equal(weights, sw)
        assert torch.equal(
            _gathered(rows_out, permuted), _gathered(want_rows, want_permuted)
        )


# ------------------------------------------------------------------------- both routers

ROUTERS = [(384, 6), (128, 3)]  # target, DSpark draft
ROUTER_ROWS = [*range(1, 17), 48]


def _router_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        from cuda.bindings import nvrtc  # noqa: F401
    except ImportError:
        return False
    return True


needs_router = pytest.mark.skipif(
    not _router_ready(), reason="needs an SM100-family GPU"
)


def _router_stock(logits, bias, num_valid, top_k, scale):
    from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate

    w, ids = moe_fused_gate(
        logits,
        bias,
        topk=top_k,
        scoring_func="sqrtsoftplus",
        num_fused_shared_experts=0,
        renormalize=True,
        routed_scaling_factor=scale,
        apply_routed_scaling_factor_on_output=True,
        num_token_non_padded=num_valid,
        sqrtsoftplus_log1p=True,
    )
    return w, ids.to(torch.int32)


def _router_fast(logits, bias, num_valid, top_k, scale):
    rows, experts = logits.shape
    weights = torch.empty((rows, top_k), dtype=torch.float32, device=logits.device)
    ids = torch.empty((rows, top_k), dtype=torch.int32, device=logits.device)
    reg._launch(
        reg.load_gate_topk(logits.device.index, experts, top_k),
        (rows, 1, 1),
        [
            reg._pointer(logits),
            reg._pointer(bias),
            reg._null_or_pointer(None),
            reg._null_or_pointer(None),
            ctypes.c_int32(0),
            ctypes.c_int64(-1),
            reg._null_or_pointer(num_valid),
            reg._pointer(weights),
            reg._pointer(ids),
            ctypes.c_float(scale),
            ctypes.c_float(0.0),
        ],
        torch.cuda.current_stream().cuda_stream,
    )
    return weights, ids


def _router_logits(kind, rows, experts, gen):
    shape = (rows, experts)
    if kind == "normal":
        return torch.randn(shape, device="cuda", generator=gen) * 2
    if kind == "wide":  # both softplus branches (> 20 passes through)
        return torch.randn(shape, device="cuda", generator=gen) * 15
    if kind == "ties":  # few distinct values: ties broken by the lowest id
        return torch.randint(-3, 4, shape, device="cuda", generator=gen).float() * 0.5
    if kind == "correlated":  # the rows of one request choose overlapping experts
        base = torch.randn((1, experts), device="cuda", generator=gen)
        return base + 0.2 * torch.randn(shape, device="cuda", generator=gen)
    if kind == "nan":
        x = torch.randn(shape, device="cuda", generator=gen)
        x[torch.rand(shape, device="cuda", generator=gen) < 0.01] = float("nan")
        return x
    raise ValueError(kind)


@needs_router
@pytest.mark.parametrize("router", ROUTERS, ids=lambda r: f"{r[0]}x{r[1]}")
@pytest.mark.parametrize("scale", [SCALE, 1.0])
@pytest.mark.parametrize("kind", ["normal", "wide", "ties", "correlated", "nan"])
def test_each_router_matches_moe_fused_gate(router, scale, kind):
    experts, top_k = router
    gen = torch.Generator(device="cuda").manual_seed(4321 + experts)
    for rows in ROUTER_ROWS:
        for trial in range(4):
            logits = _router_logits(kind, rows, experts, gen).contiguous()
            bias = torch.randn(experts, device="cuda", generator=gen) * 0.05
            valid = rows if trial % 2 == 0 else max(1, rows - 1 - trial)
            num_valid = torch.tensor([valid], dtype=torch.int32, device="cuda")
            if trial == 2:  # no num_token_non_padded: every row is valid
                num_valid = None
            sw, sid = _router_stock(logits, bias, num_valid, top_k, scale)
            fw, fid = _router_fast(logits, bias, num_valid, top_k, scale)
            torch.cuda.synchronize()
            where = f"{kind} rows={rows} valid={valid} trial={trial}"
            assert torch.equal(fid, sid), where
            assert torch.equal(fw.view(torch.int32), sw.view(torch.int32)), where


@needs_router
@pytest.mark.parametrize("router", ROUTERS, ids=lambda r: f"{r[0]}x{r[1]}")
def test_each_router_picks_torchs_top_k(router):
    """On well-separated scores the ids are torch's top-k of the biased sqrtsoftplus."""
    experts, top_k = router
    gen = torch.Generator(device="cuda").manual_seed(17)
    for rows in ROUTER_ROWS:
        logits = _router_logits("normal", rows, experts, gen).contiguous()
        bias = torch.randn(experts, device="cuda", generator=gen) * 0.05
        scores = torch.nn.functional.softplus(logits.double()).sqrt() + bias.double()
        _, fid = _router_fast(logits, bias, None, top_k, SCALE)
        assert torch.equal(fid.long(), scores.topk(top_k, dim=-1).indices), rows


@needs_router
@pytest.mark.parametrize("router", ROUTERS, ids=lambda r: f"{r[0]}x{r[1]}")
@pytest.mark.parametrize("rows", [3, 12, 48])
@pytest.mark.parametrize("count", [True, False])
def test_each_router_follows_live_inputs_in_a_graph(router, rows, count):
    experts, top_k = router
    gen = torch.Generator(device="cuda").manual_seed(11 + rows)
    logits = torch.randn((rows, experts), device="cuda", generator=gen)
    bias = torch.randn(experts, device="cuda", generator=gen) * 0.05
    num_valid = (
        torch.tensor([rows], dtype=torch.int32, device="cuda") if count else None
    )
    _router_fast(logits, bias, num_valid, top_k, SCALE)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _router_fast(logits, bias, num_valid, top_k, SCALE)
    for _ in range(3):
        logits.copy_(torch.randn((rows, experts), device="cuda", generator=gen))
        graph.replay()
        ref = _router_stock(logits, bias, num_valid, top_k, SCALE)
        torch.cuda.synchronize()
        for got, want in zip(out, ref):
            assert torch.equal(got.view(torch.int32), want.view(torch.int32))


# ------------------------------------------- the shared expert on the low-latency GEMM

SHARED_N, SHARED_K, SHARED_I = 1152, 5120, 576  # TP4 gate_up [2 x 576, 5120]
SWIGLU_LIMIT = 10.0  # text_config.swiglu_limit
# gate_up: FP32 summation order only (split-K, K tile), as the attention projections
SHARED_GATE_UP_REL = 5e-4
SHARED_OUT_REL = 1e-2


def _shared_ready() -> bool:
    if not _moe_ready():
        return False
    from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll

    return mxfp8_ll.supported()


@pytest.fixture(scope="module")
def shared_layer():
    """A TP4 shared expert with SGLang's MXFP8 view (codes, swizzled scales), its
    down_proj on SGLang's dense MXFP8 linear, and slot shared_ll's layer over it."""
    from types import SimpleNamespace

    from sglang.srt.layers.quantization.fp8_utils import (
        flashinfer_mxfp8_blockscaled_linear,
    )

    from sgl_nvidia_mega.deepseek_v41.moe import shared_ll
    from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll

    gen = torch.Generator(device="cuda").manual_seed(5)

    def weight(n, k):
        w = torch.randn(n, k, device="cuda", generator=gen) / k**0.5
        q, sf = mxfp8_ll.quantize(w.bfloat16())
        return q.contiguous(), sf.reshape(-1).contiguous()

    def dense(x, w):
        return flashinfer_mxfp8_blockscaled_linear(
            x, w[0], w[1], backend="cute-dsl", pin_tactic=True
        )

    gate_up, down = weight(SHARED_N, SHARED_K), weight(SHARED_K, SHARED_I)
    shared = SimpleNamespace(
        gate_up_proj=SimpleNamespace(
            weight=gate_up[0], weight_scale_inv_swizzled=gate_up[1]
        ),
        down_proj=lambda h: (dense(h, down), None),
        swiglu_limit=SWIGLU_LIMIT,
    )
    layer = shared_ll.SharedLL(SimpleNamespace(shared_experts=shared), max_rows=48)
    return layer, lambda x: dense(x, gate_up), shared.down_proj


def _shared_rows(rows, gen):
    """Post-norm rows with a few massive channels, BF16."""
    gain = torch.exp(0.75 * torch.randn(SHARED_K, device="cuda", generator=gen))
    gain[torch.randperm(SHARED_K, device="cuda", generator=gen)[:8]] *= 30.0
    h = torch.randn(rows, SHARED_K, device="cuda", generator=gen) * gain
    return (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)).bfloat16()


def _clamped_down(gate_up, down_proj):
    """SGLang's shared MLP past gate_up: the clamped SwiGLU, then down_proj."""
    from sglang.kernels.ops.moe.dsv4 import silu_and_mul_clamp

    h = gate_up.new_empty((gate_up.shape[0], SHARED_I))
    silu_and_mul_clamp(gate_up, h, SWIGLU_LIMIT)
    return down_proj(h)[0]


def _rel(out, ref):
    d = (out.float() - ref.float()).abs()
    return (d.norm() / ref.float().norm()).item()


@pytest.mark.skipif(not _shared_ready(), reason="needs an SM100-family GPU")
@pytest.mark.parametrize("rows", [1, 6, 8, 12, 24, 30, 40, 48])
@pytest.mark.parametrize("graph", [False, True])
def test_shared_ll_matches_sglangs_shared_expert(shared_layer, rows, graph):
    """Slot shared_ll against SGLang's shared expert on the same rows: its quant is the
    shared linear's own (codes, and scales moved into row tiles, bit for bit); its
    gate_up differs from SGLang's dense gate_up by summation order only; the clamped
    SwiGLU and down_proj are SGLang's, bit for bit on that gate_up. In a graph it
    follows live inputs."""
    from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll
    from sgl_nvidia_mega.deepseek_v41.sm103.attention.q_norm_quant import (
        mxfp8_row_tiles,
    )

    layer, gate_up_dense, down_proj = shared_layer
    layer.precompile([rows])
    gen = torch.Generator(device="cuda").manual_seed(100 + rows)
    x = _shared_rows(rows, gen)
    if graph:
        layer(x)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = layer(x)
    r = torch.arange(rows, device="cuda")[:, None]
    c = torch.arange(SHARED_K // 32, device="cuda")[None, :]
    tiled = ((r // 8) * (SHARED_K // 128) + c // 4) * 512 + (r % 8) * 16 + c % 4
    swizzled = (c // 4) * 512 + ((r % 32) * 4 + (r // 32) % 4) * 4 + c % 4
    for _ in range(3 if graph else 1):
        if graph:
            x.copy_(_shared_rows(rows, gen))
            g.replay()
        else:
            x = _shared_rows(rows, gen)
            out = layer(x)
        q, s = mxfp8_row_tiles(x)
        sg_q, sg_s = mxfp8_ll.quantize(x)
        assert torch.equal(q.view(torch.uint8), sg_q.view(torch.uint8)), rows
        assert torch.equal(s[tiled], sg_s.view(torch.uint8).flatten()[swizzled]), rows
        ref_gate_up = gate_up_dense(x)
        gate_up = mxfp8_ll.linear(q, s, layer.weight, layer.weight_scale)
        ref, mine = (_clamped_down(t, down_proj) for t in (ref_gate_up, gate_up))
        torch.cuda.synchronize()
        gate_up_rel, out_rel = _rel(gate_up, ref_gate_up), _rel(out, ref)
        print(
            f"SHARED_LL rows={rows} graph={graph} gate_up_rel={gate_up_rel:.3e} "
            f"out_rel={out_rel:.3e}"
        )
        assert gate_up_rel <= SHARED_GATE_UP_REL, rows
        # past the quant the slot is the low-latency GEMM, then SGLang's code
        assert torch.equal(out, mine), rows
        # down_proj's MXFP8 requant of h turns the gate_up rounding into a code flip
        # now and then (1.7e-3 at most in 32 draws)
        assert out_rel <= SHARED_OUT_REL, rows
