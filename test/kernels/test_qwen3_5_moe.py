# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The MoE package's kernels on one SM100-family GPU (GB300), bit for bit, eager and
replayed from a CUDA graph:
- the router front (slot route): router_gemm_quant's logits against the cuBLAS router
  GEMM SGLang runs (ReplicatedLinear, bf16), and its NVFP4 input against the trtllm
  runner's quantizer (flashinfer.fp4_quantize, linear scale layout), at 1-16 rows;
- the fused routed experts (slot mkm), against FlashInfer's trtllm-gen NVFP4 MoE as
  SGLang serves Qwen3.5 at TP8 and TP4 (intermediate 128 / 256 a rank; routing from the
  logits, RenormalizeNaive, deferred finalize): each token's 10 expert rows and their
  weights, at 1, 4 and 7 rows (the 2-CTA engines) and at TP8 also 8, 14 and 16 rows (the
  16-token engine), also with every token routed to the same experts; graph replay of
  both;
- the router front with the shared gate/up stage (slot route_shared, numeric): its
  logits and NVFP4 input bit for bit the plain launch's, and its gate/up rows against
  fp64 on SGLang's static FP8 quantization of the rows, at most 1.05 x the mean relative
  L2 and 2 x the max abs error of FlashInfer's FP8 BMM, at 1-8 rows; graph replay."""

import pytest
import torch

from sgl_nvidia_mega.qwen3_5.sm103.moe import mkm, router

E, H, TOP_K = mkm.EXPERTS, mkm.HIDDEN, mkm.TOP_K
X_SCALE = 448.0 * 6.0 / 12.0  # the static input scale of the routed experts
H_SCALE = 30.0 / (448.0 * 6.0)  # the static FC2 input scale


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


def _experts(g, rows, cols, std):
    """NVFP4 experts [E, rows, cols]: gaussian weights with row and block spread,
    quantized by FlashInfer with a global scale per expert."""
    from flashinfer import fp4_quantize

    packed = torch.empty(E, rows, cols // 2, dtype=torch.uint8, device="cuda")
    sf = torch.empty(E, rows, cols // 16, dtype=torch.uint8, device="cuda")
    scale = torch.empty(E, dtype=torch.float32, device="cuda")
    for e in range(E):
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


def _fused(layer, logits, x_fp4, x_sf):
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


@pytest.mark.parametrize("m", [1, 4, 7, 8, 14, 16])
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("shared_route", [False, True])
def test_mkm_is_the_trtllm_routed_experts(layer, m, seed, shared_route):
    if m > mkm.max_tokens(layer["inter"]):
        pytest.skip("no 16-token engine at this intermediate")
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


@pytest.mark.parametrize("m", [7, 14])
def test_mkm_replays_from_a_cuda_graph(layer, m):
    if m > mkm.max_tokens(layer["inter"]):
        pytest.skip("no 16-token engine at this intermediate")
    logits, x_fp4, x_sf = _inputs(m, 2)
    eager = _fused(layer, logits, x_fp4, x_sf)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _fused(layer, logits, x_fp4, x_sf)
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


@pytest.mark.parametrize("m", list(range(1, router.MAX_TOKENS + 1)))
def test_router_gemm_quant_is_the_stock_router_gemm_and_quantizer(m):
    x, w, scale = _router_inputs(m, seed=m)
    got = router.router_gemm_quant(x, w, scale)
    want = _router_stock(x, w, scale)
    for name, a, b in zip(("logits", "x_fp4", "x_sf"), got, want):
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
        for name, a, b in zip(
            ("logits", "x_fp4", "x_sf"), routed, router.router_gemm_quant(x, w, scale)
        ):
            assert torch.equal(a, b), name
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
