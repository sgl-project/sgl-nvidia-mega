# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The MoE package's kernels on one SM100-family GPU (GB300), bit for bit, eager and
replayed from a CUDA graph:
- the router front (slot route): router_gemm_quant's logits against the cuBLAS router
  GEMM SGLang runs (ReplicatedLinear, bf16), within eight BF16 representable steps,
  and its NVFP4 input against the trtllm runner's quantizer
  (flashinfer.fp4_quantize, linear scale layout), at 1-16 rows;
- the fused routed experts (slot mkm), against FlashInfer's trtllm-gen NVFP4 MoE as
  SGLang serves Qwen3.5 at TP8 and TP4 (intermediate 128 / 256 a rank; routing from the
  logits, RenormalizeNaive, deferred finalize): each token's 10 expert rows and their
  weights, at 1, 4 and 7 rows (the 2-CTA engines) and at TP8 also 8, 14 and 16 rows (the
  16-token engine), 17, 21, 28 and 32 rows (the 32-token engine), 33, 56 and 64 rows
  (the 64-token engine, also on route_topk's routing), 65, 84, 112 and 128 rows (the
  128-token engine, on route_topk's routing) and 129, 168, 224, 256, 257, 336, 448 and 512
  rows (the wide engines, on route_topk's routing), also with every token routed to the
  same experts; graph replay of each;
- the router front with the shared gate/up stage (slot route_shared, numeric): its
  logits against cuBLAS within eight BF16 representable steps, its NVFP4 input bit for
  bit the plain launch's, and its gate/up rows against fp64 on SGLang's static FP8
  quantization of the rows, at most 1.05 x the mean relative L2 and 2 x the max abs
  error of FlashInfer's FP8 BMM, at 1-8 rows and the TP8 / TP16 widths (256 / 128 rows);
  graph replay;
- the TEP8 routed path: the routing tables route_topk and route-big write for a rank's
  window of local experts, and the trtllm-gen MoE body on them against FlashInfer's op
  as SGLang calls it (on
  route_topk's materialized routing, from the logits up to 16 rows) at TP8 and at TEP16
  (MoE TP2 x EP8: 64 local experts, intermediate 512 a rank, ranks 0 and 5): the weights,
  the slots routed outside the rank (-1) and every local slot's FC2 row bit for bit."""

from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.qwen3_5.sm103.moe import mkm, router

E, H, TOP_K = mkm.EXPERTS, mkm.HIDDEN, mkm.TOP_K
X_SCALE = 448.0 * 6.0 / 12.0  # the static input scale of the routed experts
H_SCALE = 30.0 / (448.0 * 6.0)  # the static FC2 input scale
ROUTER_BF16_ULPS = 8  # R11's validated four-chain reduction versus cuBLAS's order


def _gpu_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import flashinfer  # noqa: F401
        from cuda.bindings import nvrtc  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _gpu_ready(), reason="needs an SM100-family GPU")


def _experts(g, rows, cols, std, n=E):
    """NVFP4 experts [n, rows, cols]: gaussian weights with row and block spread,
    quantized by FlashInfer with a global scale per expert."""
    from flashinfer import fp4_quantize

    packed = torch.empty(n, rows, cols // 2, dtype=torch.uint8, device="cuda")
    sf = torch.empty(n, rows, cols // 16, dtype=torch.uint8, device="cuda")
    scale = torch.empty(n, dtype=torch.float32, device="cuda")
    for e in range(n):
        rs = torch.exp(torch.randn(rows, 1, device="cuda", generator=g) * 0.5)
        cs = torch.exp(torch.randn(1, cols // 16, device="cuda", generator=g) * 0.5)
        w = torch.randn(rows, cols, device="cuda", generator=g) * std
        w = (w * rs * cs.repeat_interleave(16, dim=1)).to(torch.bfloat16)
        s = (448.0 * 6.0 / w.float().abs().amax()).reshape(1)
        q, q_sf = fp4_quantize(w, s, 16, False, False)
        packed[e] = q.view(torch.uint8).reshape(rows, cols // 2)
        sf[e] = q_sf.view(torch.uint8).reshape(rows, cols // 16)
        scale[e] = s
    return packed, sf, scale


@pytest.fixture(scope="module", params=mkm.INTERMEDIATES, ids=lambda i: f"I{i}")
def layer(request):
    """One TP8 (intermediate 128 a rank) or TP4 (256) Qwen3.5 MoE layer on the trtllm-gen
    weight layout, as SGLang loads it."""
    from sglang.srt.layers.quantization.utils import (
        prepare_static_weights_for_trtllm_fp4_moe,
    )

    inter = request.param  # intermediate per rank
    g = torch.Generator(device="cuda").manual_seed(11)
    w13, sf13, s13 = _experts(g, 2 * inter, H, 0.02)
    w2, sf2, s2 = _experts(g, H, inter, 0.02)
    w13, sf13, w2, sf2 = prepare_static_weights_for_trtllm_fp4_moe(
        w13, w2, sf13, sf2, H, inter, E, is_gated=True
    )
    g1 = (1.0 / (X_SCALE * s13)).float()
    return {
        "w13": w13,
        "w13_sf": sf13.view(torch.float8_e4m3fn).reshape(E, 2 * inter, H // 16),
        "w2": w2,
        "w2_sf": sf2.view(torch.float8_e4m3fn).reshape(E, H, inter // 16),
        "g1": g1,
        "g1c": (g1 / H_SCALE).float(),
        "g2": (H_SCALE / s2).float(),
        "maps": mkm.weight_maps(w13, w2),
        "inter": inter,
    }


def _inputs(m, seed):
    from flashinfer import fp4_quantize

    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, H, device="cuda", generator=g)
    x[:, torch.randint(0, H, (8,), device="cuda", generator=g)] *= (
        20.0  # outlier channels
    )
    q, q_sf = fp4_quantize(
        x.to(torch.bfloat16), torch.tensor([X_SCALE], device="cuda"), 16, False, False
    )
    logits = torch.randn(m, E, device="cuda", generator=g).to(torch.bfloat16)
    return (
        logits,
        q.view(torch.uint8).reshape(m, H // 2),
        q_sf.view(torch.uint8).reshape(m, H // 16),
    )


def _fused(layer, logits, x_fp4, x_sf, **kwargs):
    return mkm.mkm_moe(
        layer["maps"],
        logits,
        x_fp4,
        x_sf,
        layer["w13_sf"].view(torch.uint8),
        layer["w2_sf"].view(torch.uint8),
        layer["g1"],
        layer["g1c"],
        layer["g2"],
        **kwargs,
    )


def _stock(layer, logits, x_fp4, x_sf):
    """The rows and weights SGLang's trtllm runner hands its deferred finalize, row
    t * 10 + k for expert k of token t."""
    from flashinfer.fused_moe import trtllm_fp4_block_scale_moe
    from flashinfer.fused_moe.core import ActivationType
    from sglang.srt.layers.moe.utils import RoutingMethodType

    m = logits.shape[0]
    rows, weights, permuted = trtllm_fp4_block_scale_moe(
        routing_logits=logits,
        routing_bias=None,
        hidden_states=x_fp4,
        hidden_states_scale=x_sf.view(torch.float8_e4m3fn),
        gemm1_weights=layer["w13"],
        gemm1_weights_scale=layer["w13_sf"],
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=layer["w2"],
        gemm2_weights_scale=layer["w2_sf"],
        gemm2_bias=None,
        output1_scale_scalar=layer["g1c"],
        output1_scale_gate_scalar=layer["g1"],
        output2_scale_scalar=layer["g2"],
        per_token_scale=None,
        num_experts=E,
        top_k=TOP_K,
        n_group=None,
        topk_group=None,
        intermediate_size=layer["inter"],
        local_expert_offset=0,
        local_num_experts=E,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.RenormalizeNaive,
        do_finalize=False,
        activation_type=ActivationType.Swiglu,
        tune_max_num_tokens=max(8, m),
        enable_pdl=True,
    )[:3]
    return rows[permuted.reshape(-1).long()[: m * TOP_K]], weights


@pytest.mark.parametrize("m", [1, 4, 7, 8, 14, 16, 17, 21, 28, 32, 33, 56, 64])
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("shared_route", [False, True])
def test_mkm_is_the_trtllm_routed_experts(layer, m, seed, shared_route):
    if m > mkm.max_tokens(layer["inter"], m32=True, m64=True):
        pytest.skip("no engine for these rows at this intermediate")
    logits, x_fp4, x_sf = _inputs(m, seed)
    if shared_route:
        # every token routes to the same 10 experts: one slot computes each for all rows
        logits = logits[:1].expand(m, -1).contiguous()
    rows, weights = _fused(layer, logits, x_fp4, x_sf)
    want_rows, want_weights = _stock(layer, logits, x_fp4, x_sf)
    assert want_weights.dtype == torch.bfloat16
    assert torch.equal(weights.view(torch.int16), want_weights.view(torch.int16))
    differ = (rows.view(torch.int16) != want_rows.view(torch.int16)).any(dim=1)
    assert not differ.any(), f"rows {differ.nonzero().flatten().tolist()} differ"


@pytest.mark.parametrize("m", [13, 14, 16])
@pytest.mark.parametrize("seed", [0, 1])
def test_draft_32_token_engine_is_the_trtllm_routed_experts(layer, m, seed):
    """The draft's rows past 12 use the 32-token engine with identical values."""
    if layer["inter"] not in mkm.M32_ENGINES:
        pytest.skip("no 32-token engine at this intermediate")
    logits, x_fp4, x_sf = _inputs(m, seed)
    assert mkm.engine_rows(m, m16_max_rows=12) == mkm.M32_MAX_TOKENS
    rows, weights = _fused(layer, logits, x_fp4, x_sf, m16_max_rows=12)
    want_rows, want_weights = _stock(layer, logits, x_fp4, x_sf)
    assert torch.equal(weights.view(torch.int16), want_weights.view(torch.int16))
    differ = (rows.view(torch.int16) != want_rows.view(torch.int16)).any(dim=1)
    assert not differ.any(), f"rows {differ.nonzero().flatten().tolist()} differ"


@pytest.mark.parametrize("m", [7, 14, 28, 56])
def test_mkm_replays_from_a_cuda_graph(layer, m):
    if m > mkm.max_tokens(layer["inter"], m32=True, m64=True):
        pytest.skip("no engine for these rows at this intermediate")
    logits, x_fp4, x_sf = _inputs(m, 2)
    eager = _fused(layer, logits, x_fp4, x_sf)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _fused(layer, logits, x_fp4, x_sf)
    graph.replay()
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(eager, captured))


def _assert_mkm_on_route_topks_routing(layer, m, shared_route, wide_bal=False):
    if m > mkm.max_tokens(
        layer["inter"], m32=True, m64=True, m128=True, m256=True, m512=True
    ):
        pytest.skip("no engine for these rows at this intermediate")
    logits, x_fp4, x_sf = _inputs(m, 5)
    if shared_route:
        logits = logits[:1].expand(m, -1).contiguous()
    x = torch.randn(m, H, device="cuda").to(torch.bfloat16)
    ids, weights, _, _ = router.route_topk_quant(
        logits, x, torch.tensor([X_SCALE], device="cuda"), quantize=False
    )
    rows, got_weights = mkm.mkm_moe(
        layer["maps"],
        logits,
        x_fp4,
        x_sf,
        layer["w13_sf"].view(torch.uint8),
        layer["w2_sf"].view(torch.uint8),
        layer["g1"],
        layer["g1c"],
        layer["g2"],
        routing=(ids, weights),
        wide_bal=wide_bal,
    )
    want_rows, want_weights = _stock(layer, logits, x_fp4, x_sf)
    assert torch.equal(got_weights.view(torch.int16), want_weights.view(torch.int16))
    differ = (rows.view(torch.int16) != want_rows.view(torch.int16)).any(dim=1)
    assert not differ.any(), f"rows {differ.nonzero().flatten().tolist()} differ"


@pytest.mark.parametrize(
    "m", [33, 35, 56, 64, 65, 84, 112, 128, 129, 168, 224, 256, 257, 336, 448, 512]
)
@pytest.mark.parametrize("shared_route", [False, True])
def test_mkm_on_route_topks_routing_is_the_trtllm_routed_experts(
    layer, m, shared_route
):
    """Slots mkm_m64 / mkm_m128 / mkm_m256 / mkm_m512 give the runner's
    materialized-routing rows bit for bit."""
    _assert_mkm_on_route_topks_routing(layer, m, shared_route)


@pytest.mark.parametrize("m", [129, 168, 224, 256, 257, 336, 448, 512])
@pytest.mark.parametrize("shared_route", [False, True])
def test_wide_engines_balanced_split_is_the_trtllm_routed_experts(
    layer, m, shared_route
):
    """Opt-in BAL changes only assignment: rows stay bitwise stock."""
    _assert_mkm_on_route_topks_routing(layer, m, shared_route, wide_bal=True)


@pytest.mark.parametrize("m", [56, 112, 224, 448])
def test_mkm_on_route_topks_routing_replays_from_a_cuda_graph(layer, m):
    if m > mkm.max_tokens(
        layer["inter"], m32=True, m64=True, m128=True, m256=True, m512=True
    ):
        pytest.skip("no engine for these rows at this intermediate")
    logits, x_fp4, x_sf = _inputs(m, 6)
    x = torch.randn(m, H, device="cuda").to(torch.bfloat16)
    ids, weights, _, _ = router.route_topk_quant(
        logits, x, torch.tensor([X_SCALE], device="cuda"), quantize=False
    )

    def run():
        return mkm.mkm_moe(
            layer["maps"],
            logits,
            x_fp4,
            x_sf,
            layer["w13_sf"].view(torch.uint8),
            layer["w2_sf"].view(torch.uint8),
            layer["g1"],
            layer["g1c"],
            layer["g2"],
            routing=(ids, weights),
        )

    eager = run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    graph.replay()
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(eager, captured))


# ---------------------------------------------------------------- the router front


def _router_inputs(m: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, router.HIDDEN, device="cuda", generator=g).to(torch.bfloat16)
    x[0, :32] = 0  # an all-zero quantization block
    x[-1, 32:48] *= 1e4  # a block far above the others
    w = (
        torch.randn(router.EXPERTS, router.HIDDEN, device="cuda", generator=g) * 0.02
    ).to(torch.bfloat16)
    scale = (448.0 * 6.0 / x.float().abs().max()).reshape(1)
    return x, w, scale


def _router_stock(x, w, scale):
    from flashinfer import fp4_quantize

    logits = torch.nn.functional.linear(x, w)
    fp4, sf = fp4_quantize(x, scale, 16, False, False)
    return logits, fp4.view(torch.uint8).reshape(x.shape[0], -1), sf.view(torch.uint8)


def _bf16_ulps(a, b):
    """Maximum BF16 ULP distance using ordered sign-magnitude bits."""
    ia = a.contiguous().view(torch.int16).int()
    ib = b.contiguous().view(torch.int16).int()
    ia = torch.where(ia < 0, -(ia & 0x7FFF), ia)
    ib = torch.where(ib < 0, -(ib & 0x7FFF), ib)
    return int((ia - ib).abs().max())


@pytest.mark.parametrize("m", list(range(1, router.MAX_TOKENS + 1)))
def test_router_gemm_quant_is_the_stock_router_gemm_and_quantizer(m):
    x, w, scale = _router_inputs(m, seed=m)
    got = router.router_gemm_quant(x, w, scale)
    want = _router_stock(x, w, scale)
    assert _bf16_ulps(got[0], want[0]) <= ROUTER_BF16_ULPS
    for name, a, b in zip(("x_fp4", "x_sf"), got[1:], want[1:]):
        assert torch.equal(a.view(torch.uint8), b.reshape(a.shape).view(torch.uint8)), (
            name
        )


def test_router_gemm_quant_replays_from_a_cuda_graph():
    x, w, scale = _router_inputs(7, seed=0)
    router.build(x.device.index, 7)
    eager = router.router_gemm_quant(x, w, scale)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = router.router_gemm_quant(x, w, scale)
    graph.replay()
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(eager, captured))


def _shared_weights(seed: int, n: int = 256):
    """The shared expert's static-FP8 gate/up at TP8 (2 x 128 rows a rank): [N, 4096]
    e4m3 row-major, and its input and weight scales."""
    g = torch.Generator(device="cuda").manual_seed(1000 + seed)
    w = torch.randn(n, router.HIDDEN, device="cuda", generator=g) * 0.05
    return (
        w.clamp(-440, 440).to(torch.float8_e4m3fn),
        torch.tensor([0.01], device="cuda"),
        torch.tensor([0.002], device="cuda"),
    )


@pytest.mark.parametrize("m", list(range(1, router.SHARED_MAX_M + 1)))
def test_router_gemm_quant_shared_holds_its_gate_up_to_fp64(m):
    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_bmm_fp8

    errors = {"got": [], "stock": []}
    for seed in range(3):
        x, w, scale = _router_inputs(m, seed=100 * m + seed)
        x[-1, 32:48] /= 1e4  # keep the shared projection's input in e4m3's range
        sh_w, sh_in, sh_ws = _shared_weights(seed)
        *routed, gate_up = router.router_gemm_quant_shared(
            x, w, scale, sh_w, sh_in, sh_ws
        )
        want = _router_stock(x, w, scale)
        assert _bf16_ulps(routed[0], want[0]) <= ROUTER_BF16_ULPS
        for name, a, b in zip(("x_fp4", "x_sf"), routed[1:], want[1:]):
            assert torch.equal(
                a.view(torch.uint8), b.reshape(a.shape).view(torch.uint8)
            ), name
        xq, _ = static_quant_fp8(x, sh_in, repeat_scale=False)
        ref = (xq.double() @ sh_w.double().t()) * (sh_in.double() * sh_ws.double())
        stock = flashinfer_bmm_fp8(xq, sh_w.t(), sh_in, sh_ws, torch.bfloat16)
        for name, t in (("got", gate_up), ("stock", stock)):
            diff = t.double() - ref
            errors[name].append(
                ((diff.norm() / ref.norm()).item(), diff.abs().max().item())
            )
    got, stock = (
        torch.tensor(errors[n], dtype=torch.float64) for n in ("got", "stock")
    )
    print(
        f"T0 route_shared m={m}: mean rel-L2 {got[:, 0].mean():.4g} vs stock "
        f"{stock[:, 0].mean():.4g}, max abs {got[:, 1].max():.4g} vs {stock[:, 1].max():.4g}",
        flush=True,
    )
    assert got[:, 0].mean() <= 1.05 * stock[:, 0].mean()
    assert got[:, 1].max() <= 2 * stock[:, 1].max()


def test_router_gemm_quant_shared_replays_from_a_cuda_graph():
    x, w, scale = _router_inputs(7, seed=0)
    sh_w, sh_in, sh_ws = _shared_weights(0)
    router.build(x.device.index, 7, shared=True)
    eager = router.router_gemm_quant_shared(x, w, scale, sh_w, sh_in, sh_ws)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = router.router_gemm_quant_shared(x, w, scale, sh_w, sh_in, sh_ws)
    graph.replay()
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(eager, captured))


@pytest.mark.parametrize("m", [1, 7, 8])
@pytest.mark.parametrize("width", [128, 256])
def test_shared_swiglu_static_quant_is_sglangs_bytes(m, width):
    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8
    from sglang.srt.layers.activation import SiluAndMul

    from sgl_nvidia_mega.qwen3_5.sm103.moe.shared_act_quant import (
        silu_and_mul_static_quant_fp8,
    )

    g = torch.Generator(device="cuda").manual_seed(73 + m + width)
    gate_up = torch.randn(
        m, 2 * width, device="cuda", dtype=torch.bfloat16, generator=g
    )
    # Exercise both saturation edges as well as ordinary activation values.
    gate_up[0, :4] = torch.tensor(
        [-80.0, -8.0, 8.0, 80.0], device="cuda", dtype=torch.bfloat16
    )
    scale = torch.tensor([0.003], device="cuda")
    # Keep the reference independent of the fused migration, which runtime-v3
    # removes: this is the stock BF16 activation boundary followed by its static
    # per-tensor FP8 quantizer.
    want, _ = static_quant_fp8(SiluAndMul()(gate_up), scale)
    got = silu_and_mul_static_quant_fp8(gate_up, scale)
    assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))


def test_shared_swiglu_static_quant_replays_from_a_cuda_graph():
    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8
    from sglang.srt.layers.activation import SiluAndMul

    from sgl_nvidia_mega.qwen3_5.sm103.moe.shared_act_quant import (
        build,
        silu_and_mul_static_quant_fp8,
    )

    gate_up = torch.randn(8, 512, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([0.003], device="cuda")
    with pytest.raises(ValueError, match=r"FP32 \[1\]"):
        silu_and_mul_static_quant_fp8(gate_up, scale.view(1, 1))
    with pytest.raises(ValueError, match="input device"):
        silu_and_mul_static_quant_fp8(gate_up, scale.cpu())
    with pytest.raises(ValueError, match=r"\[rows, 2 \* width\]"):
        silu_and_mul_static_quant_fp8(gate_up.view(2, 4, 512), scale)
    build(gate_up.device.index, gate_up.shape[1])
    expected, _ = static_quant_fp8(SiluAndMul()(gate_up), scale)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = silu_and_mul_static_quant_fp8(gate_up, scale)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(expected.view(torch.uint8), captured.view(torch.uint8))


def test_shared_tail_normalizes_modelopt_scalar_scale_for_flashinfer_down_projection():
    from sglang.srt.layers.activation import SiluAndMul
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8LinearMethod

    from sgl_nvidia_mega.qwen3_5.moe import route

    generator = torch.Generator(device="cuda").manual_seed(91)
    rows, intermediate, hidden = 8, 256, 4096
    gate_up = (
        torch.randn(
            rows,
            2 * intermediate,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 2
    )
    weight = (
        torch.randn(hidden, intermediate, device="cuda", generator=generator) * 0.05
    ).to(torch.float8_e4m3fn)
    input_scale = torch.tensor(0.02, device="cuda")
    method = object.__new__(ModelOptFp8LinearMethod)
    method.use_marlin = method.use_sm120_fp8 = False

    class Down:
        use_flashinfer_bmm = True
        quant_method = method

        def __init__(self):
            self.weight = weight.t()
            self.weight_scale = torch.tensor(0.01, device="cuda")
            self.input_scale = input_scale

        def __call__(self, value):
            return method.apply(self, value), None

    down = Down()
    expert = SimpleNamespace(
        _enable_silu_fp4_quant_fusion=False,
        act_fn=SiluAndMul(),
        down_proj=down,
    )

    expected = method.apply(down, SiluAndMul()(gate_up))
    actual = route.shared_expert_tail(expert, gate_up)

    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


# ---------------------------------------------------------------- slot route_topk


def _routed_stock(layer, routing, x_fp4, x_sf, tune=None, offset=0, raw=False):
    """``_stock`` on a materialized routing (ids int32, weights bf16 [M, 10]): the
    runner's routed path (FlashInfer's routed MoE, as SGLang calls it; ``tune`` its
    tune_max_num_tokens, SGLang's next power of 2); for the local experts from ``offset``
    of a layer with ``count`` of them, with ``raw`` FlashInfer's three outputs."""
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
    from flashinfer.fused_moe.core import ActivationType

    m = x_fp4.shape[0]
    rows, weights, permuted = trtllm_fp4_block_scale_routed_moe(
        topk_ids=routing,
        routing_bias=None,
        hidden_states=x_fp4,
        hidden_states_scale=x_sf.view(torch.float8_e4m3fn),
        gemm1_weights=layer["w13"],
        gemm1_weights_scale=layer["w13_sf"],
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=layer["w2"],
        gemm2_weights_scale=layer["w2_sf"],
        gemm2_bias=None,
        output1_scale_scalar=layer["g1c"],
        output1_scale_gate_scalar=layer["g1"],
        output2_scale_scalar=layer["g2"],
        per_token_scale=None,
        num_experts=E,
        top_k=TOP_K,
        n_group=0,
        topk_group=0,
        intermediate_size=layer["inter"],
        local_expert_offset=offset,
        local_num_experts=layer.get("count", E),
        routed_scaling_factor=None,
        routing_method_type=1,
        do_finalize=False,
        activation_type=ActivationType.Swiglu,
        tune_max_num_tokens=tune or max(8, m),
        enable_pdl=True,
    )[:3]
    if raw:
        return rows, weights, permuted
    return rows[permuted.reshape(-1).long()[: m * TOP_K]], weights


def _topk_inputs(m, seed):
    """Router logits with ties (a bf16 grid) and a row with -inf / equal scores, the
    input rows and the routed experts' input scale."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = (torch.randn(m, E, device="cuda", generator=g) * 2).to(torch.bfloat16)
    logits[0, ::3] = logits[0, 1]  # many ties
    if m > 2:
        logits[1] = 0.5  # every score equal: the lowest expert ids
    x = torch.randn(m, H, device="cuda", generator=g).to(torch.bfloat16)
    x[0, :32] = 0
    x[-1, 32:48] *= 1e4
    return logits, x, torch.tensor([X_SCALE], device="cuda")


@pytest.mark.parametrize("m", [1, 7, 17, 21, 28, 56, 112, 224, 448, 896])
def test_route_topk_is_the_trtllm_routing_and_the_runners_quantizer(layer, m):
    from flashinfer import fp4_quantize

    for seed in range(2):
        logits, x, scale = _topk_inputs(m, seed=1000 * m + seed)
        ids, weights, x_fp4, x_sf = router.route_topk_quant(logits, x, scale)
        q, q_sf = fp4_quantize(x, scale, 16, False, False)
        assert torch.equal(x_fp4, q.view(torch.uint8).reshape(m, -1))
        assert torch.equal(x_sf, q_sf.view(torch.uint8).reshape(m, -1))
        # the routed experts on this routing equal the runner's own routing from the
        # logits, row for row, weights included
        want_rows, want_w = _stock(layer, logits, x_fp4, x_sf)
        got_rows, got_w = _routed_stock(layer, (ids, weights), x_fp4, x_sf)
        assert torch.equal(got_w.view(torch.int16), want_w.view(torch.int16))
        assert torch.equal(got_rows.view(torch.int16), want_rows.view(torch.int16))


def test_route_topk_replays_from_a_cuda_graph():
    logits, x, scale = _topk_inputs(56, seed=3)
    router.build_topk(x.device.index)
    eager = router.route_topk_quant(logits, x, scale)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = router.route_topk_quant(logits, x, scale)
    graph.replay()
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(eager, captured))


# -------------------------------------------------------- TEP8 routed-body primitives


def _tables_ref(ids, tile, window=(0, E)):
    """routingIndicesCluster's tables for top-10 ids [M, 10] and tile_n over the local
    experts ``window`` (first, count): per-local-expert counts, each one's first padded row
    (its CTA offset * tile), the CTA -> local expert and CTA -> row-limit maps of the live
    CTAs, and their count."""
    lo, count = window
    local = ids.flatten().long() - lo
    local = local[(local >= 0) & (local < count)]
    counts = torch.bincount(local, minlength=count)
    ctas = (counts + tile - 1) // tile
    offset = torch.cumsum(ctas, 0) - ctas
    live = int(ctas.sum())
    expert = torch.repeat_interleave(torch.arange(count, device=ids.device), ctas)
    first = torch.repeat_interleave(offset, ctas)
    cta = torch.arange(live, device=ids.device)
    limit = torch.minimum(
        (cta + 1) * tile, first * tile + torch.repeat_interleave(counts, ctas)
    )
    return counts, offset * tile, expert, limit, live


def _check_tables(tables, window, token_order=False):
    """``tables`` (written for its ids) against ``_tables_ref``: counts, CTA maps, live
    count, each local slot in its expert's rows once (in token order with
    ``token_order``), mapping back to its token, -1 for every other slot and padding row."""
    m, tile = tables.m, tables.tile
    lo, count = window
    counts, first, expert, limit, live = _tables_ref(tables.ids, tile, window)
    assert tables.live_ctas.item() == live
    assert tables.total_padded.item() == live * tile
    assert torch.equal(tables.tokens_per_expert[:count], counts.int())
    assert torch.equal(tables.cta_expert[:live], expert.int())
    assert torch.equal(tables.cta_row_limit[:live], limit.int())
    flat = tables.ids.flatten().long() - lo
    on = (flat >= 0) & (flat < count)
    pos = tables.expanded_to_permuted.long()
    assert (pos[~on] == -1).all()
    flat, pos = flat[on], pos[on]
    assert ((pos >= first[flat]) & (pos < first[flat] + counts[flat])).all()
    assert torch.unique(pos).numel() == pos.numel()
    token = (torch.arange(m * TOP_K, device=pos.device) // TOP_K)[on]
    assert torch.equal(tables.permuted_to_token[pos].long(), token)
    if token_order:  # an expert's rows ascend with the token (the single-block routing)
        rank = pos - first[flat]
        same = flat[:, None] == flat[None, :]
        before = same & (token[None, :] < token[:, None])
        assert torch.equal(rank, before.sum(1))
    rows = torch.zeros(live * tile, dtype=torch.bool, device=pos.device)
    rows[pos] = True
    assert (tables.permuted_to_token[: live * tile][~rows] == -1).all()


@pytest.mark.parametrize("window", [(0, E), (0, 64), (320, 64)], ids=str)
@pytest.mark.parametrize("tile", [8, 16, 32])
@pytest.mark.parametrize("m", [7, 14, 17, 28, 56, 112, 224])
def test_route_topk_writes_the_moe_bodys_routing_tables(m, tile, window):
    """With tables, route_topk routes as without them and writes
    the trtllm-gen tables of its top-10 for tile_n over the local experts ``window`` (all
    512, or TEP16's 64 of ranks 0 and 5), a second launch alike (the counter resets); each
    expert's rows are its own, in any order (atomics, as stock)."""
    if window[1] < E and m > 28:
        pytest.skip("the expert-parallel steps are up to four requests")
    logits, x, scale = _topk_inputs(m, seed=700 + m + tile)
    router.build_topk(x.device.index)
    want = router.route_topk_quant(logits, x, scale)
    tables = router.RouteTables(m, tile, x.device, *window)
    for _ in range(2):
        got = router.route_topk_quant(logits, x, scale, tables=tables)
        torch.cuda.synchronize()
        for a, b in zip(got, want):
            assert torch.equal(a, b)
        assert got[0].data_ptr() == tables.ids.data_ptr()
        assert tables.counter.item() == 0
        _check_tables(tables, window)


def _experts_of(layer, rank=0):
    """The FusedMoE attributes moe/body.py reads, from the layer fixture (EP rank
    ``rank`` of a layer with ``count`` local experts)."""
    from types import SimpleNamespace

    return SimpleNamespace(
        num_experts=E,
        num_local_experts=layer.get("count", E),
        moe_ep_rank=rank,
        top_k=TOP_K,
        intermediate_size_per_partition=layer["inter"],
        w13_weight=layer["w13"],
        w13_weight_scale=layer["w13_sf"],
        w2_weight=layer["w2"],
        w2_weight_scale=layer["w2_sf"],
        g1_scale_c=layer["g1c"],
        g1_alphas=layer["g1"],
        g2_alphas=layer["g2"],
        moe_runner_config=SimpleNamespace(activation="silu", is_gated=True),
    )


@pytest.mark.parametrize("tuned", [True, False], ids=["tuned", "default"])
@pytest.mark.parametrize("m", [17, 28, 56, 112, 224])
def test_moe_body_on_route_topks_tables_is_the_routed_op(layer, m, tuned, monkeypatch):
    """The MoE body (FlashInfer's DA entry, the routed op's tactic: the
    autotuner's, or without one FlashInfer's default) on the tables route_topk's launch
    writes gives the routed op's FC2 rows and weights, row for row by expanded index, bit
    for bit; replayed from a CUDA graph too."""
    from flashinfer.autotuner import AutoTuner, autotune
    from sglang.srt.utils import next_power_of_2

    from sgl_nvidia_mega.qwen3_5.moe import body

    logits, x, scale = _topk_inputs(m, seed=1300 + m)
    router.build_topk(x.device.index)
    ids, weights, x_fp4, x_sf = router.route_topk_quant(logits, x, scale)
    tune = next_power_of_2(m)
    if tuned:
        with autotune(True):
            _routed_stock(layer, (ids, weights), x_fp4, x_sf, tune)
    else:  # no tactic for the routed op or the body: both run FlashInfer's default
        monkeypatch.setattr(
            AutoTuner, "choose_one", lambda self, op, runners, *a, **k: (runners[0], -1)
        )
    want_rows, want_w = _routed_stock(layer, (ids, weights), x_fp4, x_sf, tune)
    experts = _experts_of(layer)
    body._WIDTHS.clear()
    width = body.width(experts, x)
    assert (width.tactic == [-1, -1]) != tuned

    def step():
        got = router.route_topk_quant(logits, x, scale, tables=width.tables)
        rows, w, permuted = body.run(
            width, experts, got[2], got[3].view(torch.float8_e4m3fn)
        )[:3]
        return rows[permuted.reshape(-1).long()], w

    got_rows, got_w = step()
    torch.cuda.synchronize()
    assert torch.equal(got_w.view(torch.int16), want_w.view(torch.int16))
    assert torch.equal(got_rows.view(torch.int16), want_rows.view(torch.int16))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured[0].view(torch.int16), want_rows.view(torch.int16))
    body._WIDTHS.clear()


# ------------------------------------------------------ body under expert parallelism

TEP16_EXPERTS, TEP16_INTER = (
    64,
    512,
)  # MoE TP2 x EP8: local experts and intermediate a rank


@pytest.fixture(scope="module")
def ep_layer():
    """One TEP16 Qwen3.5 MoE layer's local experts (64, intermediate 512 a rank) on the
    trtllm-gen weight layout, as SGLang loads them; any rank's window holds them."""
    return _local_layer(TEP16_EXPERTS, TEP16_INTER, 13)


def _local_layer(n, inter, seed):
    """A layer's ``n`` local experts (intermediate ``inter`` a rank) on the trtllm-gen
    weight layout, as SGLang loads them."""
    from sglang.srt.layers.quantization.utils import (
        prepare_static_weights_for_trtllm_fp4_moe,
    )

    g = torch.Generator(device="cuda").manual_seed(seed)
    w13, sf13, s13 = _experts(g, 2 * inter, H, 0.02, n)
    w2, sf2, s2 = _experts(g, H, inter, 0.02, n)
    w13, sf13, w2, sf2 = prepare_static_weights_for_trtllm_fp4_moe(
        w13, w2, sf13, sf2, H, inter, n, is_gated=True
    )
    g1 = (1.0 / (X_SCALE * s13)).float()
    return {
        "w13": w13,
        "w13_sf": sf13.view(torch.float8_e4m3fn).reshape(n, 2 * inter, H // 16),
        "w2": w2,
        "w2_sf": sf2.view(torch.float8_e4m3fn).reshape(n, H, inter // 16),
        "g1": g1,
        "g1c": (g1 / H_SCALE).float(),
        "g2": (H_SCALE / s2).float(),
        "inter": inter,
        "count": n,
    }


def _stock_ep(layer, logits, x_fp4, x_sf, offset):
    """FlashInfer's op from the logits as SGLang calls it at up to 16 rows (deferred
    finalize) for the local experts from ``offset``: (FC2 rows, weights, expanded ->
    permuted index, -1 for a slot routed elsewhere)."""
    from flashinfer.fused_moe import trtllm_fp4_block_scale_moe
    from flashinfer.fused_moe.core import ActivationType
    from sglang.srt.layers.moe.utils import RoutingMethodType
    from sglang.srt.utils import next_power_of_2

    return trtllm_fp4_block_scale_moe(
        routing_logits=logits,
        routing_bias=None,
        hidden_states=x_fp4,
        hidden_states_scale=x_sf.view(torch.float8_e4m3fn),
        gemm1_weights=layer["w13"],
        gemm1_weights_scale=layer["w13_sf"],
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=layer["w2"],
        gemm2_weights_scale=layer["w2_sf"],
        gemm2_bias=None,
        output1_scale_scalar=layer["g1c"],
        output1_scale_gate_scalar=layer["g1"],
        output2_scale_scalar=layer["g2"],
        per_token_scale=None,
        num_experts=E,
        top_k=TOP_K,
        n_group=None,
        topk_group=None,
        intermediate_size=layer["inter"],
        local_expert_offset=offset,
        local_num_experts=layer.get("count", E),
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.RenormalizeNaive,
        do_finalize=False,
        activation_type=ActivationType.Swiglu,
        tune_max_num_tokens=next_power_of_2(logits.shape[0]),
        enable_pdl=True,
    )[:3]


def _same_moe(got, want, m):
    """Two deferred-finalize outputs (FC2 rows, weights, expanded -> permuted) agree: the
    weights bit for bit, the same slots routed outside the rank (-1), and every local
    slot's FC2 row bit for bit. Returns the two permutations."""
    rows, w, perm = got[:3]
    want_rows, want_w, want_perm = want[:3]
    perm = perm.reshape(-1)[: m * TOP_K].long()
    want_perm = want_perm.reshape(-1)[: m * TOP_K].long()
    assert torch.equal(w.view(torch.int16), want_w.view(torch.int16))
    assert torch.equal(perm < 0, want_perm < 0)
    on = want_perm >= 0
    assert torch.equal(
        rows[perm[on]].view(torch.int16), want_rows[want_perm[on]].view(torch.int16)
    )
    return perm, want_perm


def _ep_router_inputs(m, seed, kind, window):
    """Rows, router weights and the routed input scale; ``kind`` "local" lifts the
    window's experts by ~1 sd of a logit (about half the slots local), "same" sends every
    token to the window's first 10 experts (several CTAs an expert past 8 rows), "ties"
    gives every third expert the same top logit (the lowest ids win, past a warp of
    candidates)."""
    lo, count = window
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, H, device="cuda", generator=g)
    x[0, :32] = 0
    w = torch.randn(E, H, device="cuda", generator=g) * 0.02
    mean = x.mean(0)
    up = mean / mean.pow(2).sum()  # x_t . up ~ 1 for every token t
    if kind == "local":
        w[lo : lo + count] += 1.5 * up
    elif kind == "same":
        w[lo : lo + TOP_K] += 12.0 * up
    elif kind == "ties":
        w[::3] = w[1] + 3.0 * up
    return (
        x.to(torch.bfloat16),
        w.to(torch.bfloat16),
        torch.tensor([X_SCALE], device="cuda"),
    )


EP_CASES = [  # (layout, rank): TP8 (all 512 experts local) and TEP16 ranks 0 and 5
    ("tp8", 0),
    ("tep16", 0),
    ("tep16", 5),
]


def _ep_case(request, layout, rank):
    layer = request.getfixturevalue("ep_layer" if layout == "tep16" else "tp8_layer")
    count = layer.get("count", E)
    return layer, (rank * count, count)


@pytest.fixture(scope="module")
def tp8_layer():
    """The TP8 layer of the ``layer`` fixture (intermediate 128 a rank, 512 experts)."""
    from sglang.srt.layers.quantization.utils import (
        prepare_static_weights_for_trtllm_fp4_moe,
    )

    inter = 128
    g = torch.Generator(device="cuda").manual_seed(11)
    w13, sf13, s13 = _experts(g, 2 * inter, H, 0.02)
    w2, sf2, s2 = _experts(g, H, inter, 0.02)
    w13, sf13, w2, sf2 = prepare_static_weights_for_trtllm_fp4_moe(
        w13, w2, sf13, sf2, H, inter, E, is_gated=True
    )
    g1 = (1.0 / (X_SCALE * s13)).float()
    return {
        "w13": w13,
        "w13_sf": sf13.view(torch.float8_e4m3fn).reshape(E, 2 * inter, H // 16),
        "w2": w2,
        "w2_sf": sf2.view(torch.float8_e4m3fn).reshape(E, H, inter // 16),
        "g1": g1,
        "g1c": (g1 / H_SCALE).float(),
        "g2": (H_SCALE / s2).float(),
        "inter": inter,
    }


@pytest.mark.parametrize("kind", ["random", "local", "same"])
@pytest.mark.parametrize("m", [7, 14, 28])
@pytest.mark.parametrize("layout,rank", EP_CASES)
def test_moe_body_on_route_topks_ep_tables_is_the_routed_op(
    request, layout, rank, m, kind
):
    """For a rank's local experts, the MoE body on route_topk's
    tables is FlashInfer's routed op on route_topk's routing (the runner's path past 16
    rows) and its op from the logits: weights, the slots routed elsewhere (-1) and every
    local slot's FC2 row bit for bit; replayed from a CUDA graph too."""
    from sgl_nvidia_mega.qwen3_5.moe import body

    layer, window = _ep_case(request, layout, rank)
    x, w, scale = _ep_router_inputs(m, 3000 + 31 * m + rank, kind, window)
    logits = torch.nn.functional.linear(x, w)
    router.build_topk(x.device.index)
    ids, wts, x_fp4, x_sf = router.route_topk_quant(logits, x, scale)
    experts = _experts_of(layer, rank)
    body._WIDTHS.clear()
    width = body.width(experts, x)

    def step():
        got = router.route_topk_quant(logits, x, scale, tables=width.tables)
        return body.run(width, experts, got[2], got[3].view(torch.float8_e4m3fn))[:3]

    got = step()
    torch.cuda.synchronize()
    _check_tables(width.tables, window)
    routed = _routed_stock(layer, (ids, wts), x_fp4, x_sf, None, window[0], raw=True)
    _same_moe(got, routed, m)
    _same_moe(got, _stock_ep(layer, logits, x_fp4, x_sf, window[0]), m)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        _same_moe(captured, routed, m)
    body._WIDTHS.clear()


@pytest.mark.parametrize("m", [17, 21, 28, 32])
def test_router_front_tiles_are_its_16_row_launches(m):
    """Slot route_gemm (numeric past 16 rows): the router front over 16-row token tiles in
    one launch equals the 16-row launch on each tile, bit for bit; its NVFP4 input is the
    runner's quantizer's; its logits stay within eight BF16 representable steps of
    cuBLAS's GEMM; and
    route_topk's routing-only launch routes as its quantizing launch."""
    from flashinfer import fp4_quantize

    x, w, scale = _router_inputs(m, seed=900 + m)
    router.build(x.device.index, 16)
    logits, x_fp4, x_sf = router.router_gemm_quant_tiled(x, w, scale)
    for t0 in range(0, m, 16):
        tile = router.router_gemm_quant(x[t0 : t0 + 16].contiguous(), w, scale)
        for a, b in zip((logits, x_fp4, x_sf), tile):
            assert torch.equal(a[t0 : t0 + 16], b)
    q, q_sf = fp4_quantize(x, scale, 16, False, False)
    assert torch.equal(x_fp4, q.view(torch.uint8).reshape(m, -1))
    assert torch.equal(x_sf, q_sf.view(torch.uint8).reshape(m, -1))
    stock = torch.nn.functional.linear(x, w)
    assert _bf16_ulps(logits, stock) <= ROUTER_BF16_ULPS
    routed = router.route_topk_quant(logits, x, scale)
    only = router.route_topk_quant(logits, x, scale, quantize=False)
    assert only[2] is None and only[3] is None
    assert torch.equal(routed[0], only[0]) and torch.equal(routed[1], only[1])


# ------------------------------------------- routed body at TEP8's wide request counts

TEP8_EXPERTS, TEP8_INTER = (
    128,
    512,
)  # MoE TP2 x EP4: local experts and intermediate a rank
TEP8_ROWS = [231, 448, 896]  # 33, 64 and 128 requests of 7 verify rows
# route_big's TEP8 tiers L1 / L2: 5, 16, 32, 33, 64 and 128 requests
TEP8_BIG_ROWS = [35, 112, 224, 231, 448, 518, 672, 784, 896]


@pytest.fixture(scope="module")
def tep8_layer():
    """One TEP8 Qwen3.5 MoE layer's local experts (128, intermediate 512 a rank)."""
    return _local_layer(TEP8_EXPERTS, TEP8_INTER, 17)


@pytest.mark.parametrize(
    "window", [(0, TEP8_EXPERTS), (384, TEP8_EXPERTS), (0, E)], ids=str
)
@pytest.mark.parametrize("tile", [8, 32, 128])
@pytest.mark.parametrize("m", TEP8_ROWS)
def test_route_topk_writes_the_tep8_routing_tables(m, tile, window):
    """At TEP8's widths, route_topk with tables routes as without
    them and writes the tables of the rank's 128 local experts (ranks 0 and 3; all 512)
    for tile_n, a second launch alike."""
    logits, x, scale = _topk_inputs(m, seed=4100 + m + tile)
    router.build_topk(x.device.index)
    want = router.route_topk_quant(logits, x, scale)
    tables = router.RouteTables(m, tile, x.device, *window)
    for _ in range(2):
        got = router.route_topk_quant(logits, x, scale, tables=tables)
        torch.cuda.synchronize()
        for a, b in zip(got, want):
            assert torch.equal(a, b)
        assert tables.counter.item() == 0
        _check_tables(tables, window)


@pytest.mark.parametrize("tuned", [True, False], ids=["tuned", "default"])
@pytest.mark.parametrize("kind", ["random", "local", "same"])
@pytest.mark.parametrize("m", TEP8_ROWS)
@pytest.mark.parametrize("rank", [0, 3])
def test_moe_body_on_route_topks_tep8_tables_is_the_routed_op(
    tep8_layer, rank, m, kind, tuned, monkeypatch
):
    """At TEP8's widths, the MoE body on route_topk's tables for a rank's
    128 local experts is FlashInfer's routed op on route_topk's routing (the runner's
    path there; the autotuner's tactic or FlashInfer's default for both): weights, the
    slots routed elsewhere (-1) and every local slot's FC2 row bit for bit; replayed from
    a CUDA graph too."""
    from flashinfer.autotuner import AutoTuner, autotune
    from sglang.srt.utils import next_power_of_2

    from sgl_nvidia_mega.qwen3_5.moe import body

    window = (rank * TEP8_EXPERTS, TEP8_EXPERTS)
    x, w, scale = _ep_router_inputs(m, 5000 + 31 * m + rank, kind, window)
    logits = torch.nn.functional.linear(x, w)
    router.build_topk(x.device.index)
    ids, wts, x_fp4, x_sf = router.route_topk_quant(logits, x, scale)
    tune = next_power_of_2(m)
    if tuned:
        with autotune(True):
            _routed_stock(tep8_layer, (ids, wts), x_fp4, x_sf, tune, window[0])
    else:  # no tactic for the routed op or the body: both run FlashInfer's default
        monkeypatch.setattr(
            AutoTuner, "choose_one", lambda self, op, runners, *a, **k: (runners[0], -1)
        )
    routed = _routed_stock(
        tep8_layer, (ids, wts), x_fp4, x_sf, tune, window[0], raw=True
    )
    experts = _experts_of(tep8_layer, rank)
    body._WIDTHS.clear()
    width = body.width(experts, x)
    assert (width.tactic == [-1, -1]) != tuned

    def step():
        got = router.route_topk_quant(logits, x, scale, tables=width.tables)
        return body.run(width, experts, got[2], got[3].view(torch.float8_e4m3fn))[:3]

    got = step()
    torch.cuda.synchronize()
    _check_tables(width.tables, window)
    _same_moe(got, routed, m)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        _same_moe(captured, routed, m)
    body._WIDTHS.clear()


@pytest.mark.parametrize(
    "window", [(0, TEP8_EXPERTS), (384, TEP8_EXPERTS), (0, E)], ids=str
)
@pytest.mark.parametrize("tile", [8, 32, 128])
@pytest.mark.parametrize("m", [35, 112, 168, 224, 231, 252, 259, 448, 512, 518, 896])
def test_route_big_is_route_topk_with_its_tables_in_token_order(m, tile, window):
    """Route-big's launch (one routing cluster of 8 / 16 CTAs, the rest
    quantizing) gives route_topk's top-10 and NVFP4 input bit for bit and the same routing
    tables, an expert's rows in token order; a second launch on the same tables alike; and from
    a CUDA graph."""
    logits, x, scale = _topk_inputs(m, seed=4300 + m + tile)
    router.build_topk(x.device.index)
    router.build_big(x.device.index, window[1])
    want = router.route_topk_quant(logits, x, scale)
    tables = router.RouteTables(m, tile, x.device, *window)
    for _ in range(2):
        got = router.route_big_quant(logits, x, scale, tables)
        torch.cuda.synchronize()
        for a, b in zip(got, want):
            assert torch.equal(a, b)
        _check_tables(tables, window, token_order=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = router.route_big_quant(logits, x, scale, tables)
    tables.expanded_to_permuted.fill_(7)
    tables.permuted_to_token.fill_(7)
    graph.replay()
    torch.cuda.synchronize()
    for a, b in zip(captured, want):
        assert torch.equal(a, b)
    _check_tables(tables, window, token_order=True)


@pytest.mark.parametrize("window", [(0, TEP8_EXPERTS), (384, TEP8_EXPERTS)], ids=str)
@pytest.mark.parametrize("tile", [32, 64])
@pytest.mark.parametrize("ctas", [32, 64])
@pytest.mark.parametrize("m", [672, 679, 728, 784, 840, 896])
def test_route_big_wide_launches_are_route_topk(m, ctas, tile, window):
    """Both validated L2 launch widths reproduce route_topk and its padded tables."""
    logits, x, scale = _topk_inputs(m, seed=7100 + m + ctas + tile)
    router.build_topk(x.device.index)
    router.build_big(x.device.index, window[1])
    want = router.route_topk_quant(logits, x, scale)
    tables = router.RouteTables(m, tile, x.device, *window)
    got = router.route_big_quant(logits, x, scale, tables, ctas=ctas)
    torch.cuda.synchronize()
    for actual, expected in zip(got, want):
        assert torch.equal(actual, expected)
    _check_tables(tables, window, token_order=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = router.route_big_quant(logits, x, scale, tables, ctas=ctas)
    tables.expanded_to_permuted.fill_(7)
    tables.permuted_to_token.fill_(7)
    graph.replay()
    torch.cuda.synchronize()
    for actual, expected in zip(captured, want):
        assert torch.equal(actual, expected)
    _check_tables(tables, window, token_order=True)


@pytest.mark.parametrize("kind", ["random", "local", "same"])
@pytest.mark.parametrize("m", TEP8_BIG_ROWS)
@pytest.mark.parametrize("rank", [0, 3])
def test_moe_body_on_route_bigs_tep8_tables_is_the_routed_op(tep8_layer, rank, m, kind):
    """The MoE body on route_big's tables (a rank's 128 local experts) is
    FlashInfer's routed op on route_topk's routing: weights, the slots routed elsewhere (-1) and
    every local slot's FC2 row bit for bit; replayed from a CUDA graph too."""
    from sgl_nvidia_mega.qwen3_5.moe import body

    window = (rank * TEP8_EXPERTS, TEP8_EXPERTS)
    x, w, scale = _ep_router_inputs(m, 5200 + 31 * m + rank, kind, window)
    logits = torch.nn.functional.linear(x, w)
    router.build_topk(x.device.index)
    router.build_big(x.device.index, TEP8_EXPERTS)
    ids, wts, x_fp4, x_sf = router.route_topk_quant(logits, x, scale)
    routed = _routed_stock(
        tep8_layer, (ids, wts), x_fp4, x_sf, None, window[0], raw=True
    )
    experts = _experts_of(tep8_layer, rank)
    body._WIDTHS.clear()
    width = body.width(experts, x)

    def step():
        got = router.route_big_quant(logits, x, scale, width.tables)
        return body.run(width, experts, got[2], got[3].view(torch.float8_e4m3fn))[:3]

    got = step()
    torch.cuda.synchronize()
    _check_tables(width.tables, window, token_order=True)
    _same_moe(got, routed, m)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        _same_moe(captured, routed, m)
    body._WIDTHS.clear()


def test_moe_body_widths_share_their_buffers(tep8_layer):
    """TEP8's widths run their bodies on views of one set of buffers (the widest asked
    for first, as SGLang captures its graphs), each still the routed op's, alternately."""
    from sgl_nvidia_mega.qwen3_5.moe import body

    window = (TEP8_EXPERTS, TEP8_EXPERTS)
    router.build_topk(torch.cuda.current_device())
    experts = _experts_of(tep8_layer, 1)
    body._WIDTHS.clear()
    body._SHARED.clear()
    cases = []
    for m in (896, 231):
        x, w, scale = _ep_router_inputs(m, 6000 + m, "local", window)
        logits = torch.nn.functional.linear(x, w)
        ids, wts, x_fp4, x_sf = router.route_topk_quant(logits, x, scale)
        want = _routed_stock(
            tep8_layer, (ids, wts), x_fp4, x_sf, None, window[0], raw=True
        )
        cases.append((m, x, logits, scale, body.width(experts, x), want))
    for _ in range(2):
        for m, x, logits, scale, width, want in cases:
            got = router.route_topk_quant(logits, x, scale, tables=width.tables)
            got = body.run(width, experts, got[2], got[3].view(torch.float8_e4m3fn))[:3]
            torch.cuda.synchronize()
            _same_moe(got, want, m)
    wide, narrow = cases[0][4].workspace, cases[1][4].workspace
    for a, b in zip(
        wide, narrow
    ):  # the narrow width's buffers that fit are the wide one's
        if b.numel() and b.shape[0] <= a.shape[0]:
            assert a.data_ptr() == b.data_ptr()
    body._WIDTHS.clear()
