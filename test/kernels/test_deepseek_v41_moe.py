# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The DeepSeek-V4.1 routed-expert kernels and router on one SM100-family GPU (GB300).

- The route kernels + GEMM pair against FlashInfer's routed MoE (its deferred-finalize
  GEMM2 rows), as SGLang serves it, on the target layer (384 experts, top-6) at
  representative rows (1 / 6 / 16 / 48), including experts chosen by every row (several
  tiles); eager and replayed from a CUDA graph.
- gate_topk.cu (target) and gate_topk_draft.cu (DSpark draft) against SGLang's
  moe_fused_gate, bit for bit (ids and weights) at 1-48 and 64 rows, on whichever kernel
  serves it in the installed SGLang (Triton or the CUDA warp router): ties (also at the
  top-k cutoff), the score function's whole range (subnormals, the polynomial's steps and
  underflows, large magnitudes, +-inf), NaN logits and NaN rows, padded rows and no
  padded-row count (every row valid), and vision routing.
- Expert parallelism (TP8's MoE TP4 x EP2): each EP rank's half of a layer's experts
  (target 192 of 384, top-6; DSpark draft 64 of 128, top-3) against FlashInfer's routed
  MoE with SGLang's ``local_expert_offset``: the slots routed to the other half map to
  row -1 alike, and every kept slot's GEMM2 row equals the all-experts pair's bit for
  bit; from the router's logits, the fast gate's ids and weights equal the
  all-experts gate's bit for bit.
- The shared expert's gate_up on the low-latency GEMM at the TP8 shard (N 576), whose
  last 128-row tile is half full, against the TP4 shard's kernel on the same rows.
"""

import ctypes
import os

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41.sm103.moe import routed_expert_gemm as reg

# SGL_NVIDIA_MEGA_TEST_GEMM2=pipe | pool: every pair below on the gemm2_pipe slot's GEMM2
# (pool: its 96-register build), so the same cases gate the slot
GEMM2_OPTIONS = {
    "": {},
    "pipe": {"gemm2_pipe": True},
    "pool": {"gemm2_pipe": True, "gemm2_registers": reg.GEMM2_POOL_REGISTERS},
}[os.environ.get("SGL_NVIDIA_MEGA_TEST_GEMM2", "")]

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


def _flashinfer_ep(x_q, x_s, ids, weights, layer, *, num_experts, offset):
    """``_flashinfer`` on one EP rank's experts, as SGLang's EP routed MoE calls it."""
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
    from sglang.srt.layers.moe.utils import RoutingMethodType

    w13, s13, w2, s2, clamp = layer
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
        num_experts=num_experts,
        top_k=ids.size(1),
        intermediate_size=reg.PADDED,
        local_expert_offset=offset,
        local_num_experts=w13.size(0),
        do_finalize=False,
        tune_max_num_tokens=8,
        enable_pdl=True,
    )
    return result[0], result[2].view(ids.shape)


def _ep_half(layer, ep_rank: int, ep: int = 2):
    """EP rank ``ep_rank``'s consecutive window of a layer's experts, as SGLang loads
    them (rank r holds experts [r * E / ep, (r + 1) * E / ep))."""
    local = layer[0].size(0) // ep
    window = slice(ep_rank * local, (ep_rank + 1) * local)
    return tuple(t[window].contiguous() for t in layer), ep_rank * local


def _check_pair(layer, *, top_k, rows, distinct, graph, seed, max_rows):
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

    experts = layer[0].size(0)
    generator = torch.Generator().manual_seed(seed)
    gemm = reg.RoutedExpertGemm(*layer, top_k=top_k, max_rows=max_rows, **GEMM2_OPTIONS)
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

ROUTERS = {  # (experts, top_k): the target's and the DSpark draft's
    "target": (TARGET_EXPERTS, TARGET_TOP_K),
    "draft": (reg.DRAFT_GATE_TOPK_EXPERTS, reg.DRAFT_GATE_TOPK_TOP_K),
}
GATE_ROWS = [*range(1, reg.SERVED_MAX_ROWS + 1), 64]


def _stock(logits, bias, num_valid, top_k, vl=None):
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
        topk=top_k,
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


def _fast(logits, bias, num_valid, top_k, vl=None):
    rows, experts = logits.shape
    weights = torch.empty((rows, top_k), dtype=torch.float32, device=logits.device)
    ids = torch.empty((rows, top_k), dtype=torch.int32, device=logits.device)
    bias_alt, input_ids, token = vl if vl is not None else (None, None, -1)
    reg._launch(
        reg.load_gate_topk(logits.device.index, experts, top_k),
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


# Inputs that walk sqrt(softplus(x)) over its whole domain: around 0 and the subnormals, the
# polynomial's n = rint(-|x| log2(e) / 2) steps, the x <= -174 underflow of z = w^2 and the
# -176 underflow of w, the -500 clamp, the Triton kernel's x > 20 branch, and fp32's extremes.
_SPECIAL = [
    sign * float(v)
    for v in (
        "0 1e-45 1e-40 1.1754942e-38 1e-30 1e-7 0.5 0.6931472 1 1.3862944 20 87 88.7 103 "
        "170 173.9 174 174.7 175.5 176 179 181 499 500 501 1e4 1e10 1e30 3.4028235e38 inf"
    ).split()
    for sign in (1.0, -1.0)
] + [19.999998, 20.000002]


def _logits(kind, rows, experts, gen):
    shape, dev = (rows, experts), "cuda"

    def rand():
        return torch.rand(shape, device=dev, generator=gen)

    def sign():
        return torch.randint(0, 2, shape, device=dev, generator=gen) * 2.0 - 1

    if kind == "normal":
        return torch.randn(shape, device=dev, generator=gen)
    if kind == "wide":  # both softplus branches (> 20 passes through)
        return torch.randn(shape, device=dev, generator=gen) * 15
    if kind == "ties":  # few distinct values: ties broken by the lowest id
        return torch.randint(-3, 4, shape, device=dev, generator=gen).float() * 0.5
    if kind == "correlated":  # the rows of one request choose overlapping experts
        base = torch.randn((1, experts), device=dev, generator=gen)
        return base + 0.2 * torch.randn(shape, device=dev, generator=gen)
    if kind == "nan":
        x = torch.randn(shape, device=dev, generator=gen)
        x[rand() < 0.01] = float("nan")
        return x
    if kind == "range":
        # sign * 2^u, u uniform over fp32's exponents, and the special values
        x = (sign() * torch.exp2((rand() * 277.0 - 149.0).double())).float()
        special = torch.tensor(_SPECIAL, device=dev)
        pick = torch.randint(0, len(_SPECIAL), shape, device=dev, generator=gen)
        return torch.where(rand() < 0.25, special[pick], x)
    if kind == "polynomial":  # dense over [-200, 40]: every n step and both underflows
        return rand() * 240.0 - 200.0
    if kind == "subnormal":
        # |x| down in the subnormals, where -|x| and the selects matter
        bits = torch.randint(1, 0x00800000, shape, device=dev, generator=gen)
        return bits.int().view(torch.float32) * sign()
    if kind == "large":
        # |x| from 1e2 to fp32's max: weights far from 1, sums that round
        return (sign() * torch.exp2((rand() * 121.0 + 7.0).double())).float()
    if kind == "underflow":
        # every score tiny (x <= -174): z and w underflow, sums tiny
        return -174.0 - rand() * 8.0
    if kind == "tie_cutoff":  # many experts equal to the k-th score, the rest below
        x = torch.full(shape, -2.0, device=dev)
        x[:, torch.randperm(experts, device=dev, generator=gen)[: experts // 4]] = 0.75
        top = torch.randint(0, experts, (rows, 2), device=dev, generator=gen)
        return x.scatter_(1, top, 1.5)
    if kind == "nan_rows":
        # rows all NaN, NaN mixed with +-inf, and NaN on a few experts
        x = torch.randn(shape, device=dev, generator=gen)
        x[0::3] = float("nan")
        x[1::3, ::5] = float("nan")
        x[1::3, 1::7] = float("inf")
        x[2::3, 2::11] = float("-inf")
        return x
    raise ValueError(kind)


def _bias(kind, experts, gen):
    if kind == "subnormal":  # keep score + bias in the subnormals too
        bits = torch.randint(0, 0x00800000, (experts,), device="cuda", generator=gen)
        sign = torch.rand(experts, device="cuda", generator=gen) < 0.5
        return bits.int().view(torch.float32) * torch.where(sign, -1.0, 1.0)
    if kind == "ties":  # equal biased scores
        return torch.zeros(experts, device="cuda")
    return torch.randn(experts, device="cuda", generator=gen) * 0.05


GATE_KINDS = (
    "normal wide ties correlated nan range polynomial subnormal large underflow "
    "tie_cutoff nan_rows"
).split()


def _equal(fw, fid, sw, sid, where):
    torch.cuda.synchronize()
    assert torch.equal(fid, sid), f"{where}: ids\n{fid}\n{sid}"
    bad = (fw.view(torch.int32) != sw.view(torch.int32)).nonzero()
    assert bad.numel() == 0, (
        f"{where}: {bad.size(0)} weights differ, first at {bad[0].tolist()}: "
        f"{fw[tuple(bad[0])].item()!r} vs {sw[tuple(bad[0])].item()!r}"
    )


@needs_moe
def test_gate_reference_is_known():
    """The installed SGLang's router is one gate_topk.cuh reproduces (else the fast gate
    steps aside): Triton before sgl-project/sglang#47641, the warp router after."""
    from importlib.util import find_spec

    has_warp = find_spec("sglang.kernels.ops.moe.moe_route_warp") is not None
    for experts, top_k in ROUTERS.values():
        ref = reg.gate_reference(experts, top_k)
        assert ref is not None, (experts, top_k)
        assert (ref != reg.GATE_REFERENCE_TRITON) == has_warp, (experts, top_k, ref)


@needs_moe
@pytest.mark.parametrize("router", sorted(ROUTERS))
@pytest.mark.parametrize("kind", GATE_KINDS)
def test_gate_topk_matches_moe_fused_gate(router, kind):
    experts, top_k = ROUTERS[router]
    gen = torch.Generator(device="cuda").manual_seed(1234)
    for rows in GATE_ROWS:
        for trial in range(4):
            logits = _logits(kind, rows, experts, gen).contiguous()
            bias = _bias(kind, experts, gen)
            valid = rows if trial % 2 == 0 else max(1, rows - 1 - trial)
            num_valid = torch.tensor([valid], dtype=torch.int32, device="cuda")
            if trial == 2:  # no num_token_non_padded: every row is valid
                num_valid = None
            sw, sid = _stock(logits, bias, num_valid, top_k)
            fw, fid = _fast(logits, bias, num_valid, top_k)
            _equal(
                fw,
                fid,
                sw,
                sid,
                f"{router} {kind} rows={rows} valid={valid} trial={trial}",
            )


@needs_moe
@pytest.mark.parametrize("router", sorted(ROUTERS))
@pytest.mark.parametrize("kind", ["normal", "nan_rows"])
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
def test_gate_topk_vision_routing_matches_moe_fused_gate(router, kind, ids_dtype):
    """Rows whose token is the image token rank with the vision bias (DSV4.1 VL routing),
    normalized by sum + eps."""
    experts, top_k = ROUTERS[router]
    token = 128815
    gen = torch.Generator(device="cuda").manual_seed(4321)
    for rows in GATE_ROWS:
        for trial in range(2):
            logits = _logits(kind, rows, experts, gen).contiguous()
            bias = _bias(kind, experts, gen)
            bias_alt = torch.randn(experts, device="cuda", generator=gen) * 0.5
            image = torch.rand(rows, device="cuda", generator=gen) < 0.5
            input_ids = torch.where(
                image,
                torch.full((rows,), token, device="cuda"),
                torch.randint(0, 1000, (rows,), device="cuda", generator=gen),
            ).to(ids_dtype)
            valid = rows if trial == 0 else max(1, rows - 2)
            num_valid = torch.tensor([valid], dtype=torch.int32, device="cuda")
            vl = (bias_alt, input_ids, token)
            sw, sid = _stock(logits, bias, num_valid, top_k, vl)
            fw, fid = _fast(logits, bias, num_valid, top_k, vl)
            _equal(fw, fid, sw, sid, f"{router} vl {kind} rows={rows} trial={trial}")


# --------------------------------------------------------- expert parallelism (EP2)

DRAFT_EXPERTS, DRAFT_TOP_K = 128, 3


@pytest.fixture(scope="module")
def draft_layer():
    return _shuffled_layer(torch.Generator().manual_seed(3), DRAFT_EXPERTS)


def _quantized_rows(rows, generator):
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

    x = torch.randn(rows, reg.HIDDEN, generator=generator).to(torch.bfloat16).cuda()
    x_q, x_s = flashinfer_mxfp8_quantize(x, False, alignment=reg.HIDDEN)
    return x_q, x_s.view(torch.float8_e4m3fn).reshape(rows, -1)


def _check_ep_halves(layer, *, top_k, rows, distinct, seed):
    experts = layer[0].size(0)
    generator = torch.Generator().manual_seed(seed)
    x_q, x_s = _quantized_rows(rows, generator)
    ids = _routing(rows, distinct, generator, experts, top_k).cuda()
    weights = torch.softmax(torch.randn(rows, top_k, generator=generator), -1).cuda()
    full = reg.RoutedExpertGemm(
        *layer, top_k=top_k, max_rows=reg.SERVED_MAX_ROWS, **GEMM2_OPTIONS
    )
    full_rows, _, full_permuted = full(x_q, x_s, ids, weights)
    kept_total = torch.zeros_like(ids, dtype=torch.int32)
    for ep_rank in range(2):
        half, offset = _ep_half(layer, ep_rank)
        gemm = reg.RoutedExpertGemm(
            *half,
            top_k=top_k,
            max_rows=reg.SERVED_MAX_ROWS,
            router_experts=experts,
            expert_offset=offset,
            **GEMM2_OPTIONS,
        )
        rows_out, got_weights, permuted = gemm(x_q, x_s, ids, weights)
        want_rows, want_permuted = _flashinfer_ep(
            x_q, x_s, ids, weights, half, num_experts=experts, offset=offset
        )
        torch.cuda.synchronize()
        where = f"ep_rank={ep_rank} rows={rows} distinct={distinct}"
        local = (ids >= offset) & (ids < offset + half[0].size(0))
        assert torch.equal(permuted >= 0, local), where
        assert torch.equal(want_permuted >= 0, local), where
        assert torch.equal(got_weights, weights), where
        kept = permuted.clamp(min=0).long()
        # each kept slot's row is a function of its expert and token only: the
        # all-experts pair's row, bit for bit
        got = rows_out[kept][local]
        assert torch.equal(
            got.view(torch.int16),
            full_rows[full_permuted.long()][local].view(torch.int16),
        ), where
        want = want_rows[want_permuted.clamp(min=0).long()][local]
        torch.testing.assert_close(got.float(), want.float(), rtol=2**-7, atol=1e-6)
        kept_total += local.int()
    assert torch.equal(kept_total, torch.ones_like(kept_total))


@needs_moe
@pytest.mark.parametrize("rows", [1, 6, 8, 16, 48])
@pytest.mark.parametrize("distinct", [6, 384])
def test_target_ep_halves_match_flashinfer_ep(target_layer, rows, distinct):
    _check_ep_halves(
        target_layer,
        top_k=TARGET_TOP_K,
        rows=rows,
        distinct=distinct,
        seed=1900 + rows + distinct,
    )


@needs_moe
@pytest.mark.parametrize("rows", [1, 6, 16, 48])
@pytest.mark.parametrize("distinct", [3, 128])
def test_draft_ep_halves_match_flashinfer_ep(draft_layer, rows, distinct):
    _check_ep_halves(
        draft_layer,
        top_k=DRAFT_TOP_K,
        rows=rows,
        distinct=distinct,
        seed=2900 + rows + distinct,
    )


@needs_moe
@pytest.mark.parametrize(
    "which, experts, top_k", [("target", 384, 6), ("draft", 128, 3)]
)
@pytest.mark.parametrize("rows", [1, 6, 12, 48])
def test_ep_fast_gate_routes_the_router_ids(request, which, experts, top_k, rows):
    """The fast gate on an EP half reads the router's logits over every expert: its
    ids and weights are the all-experts gate's bit for bit, and its rows are the
    all-experts pair's for the slots it holds."""
    layer = request.getfixturevalue(f"{which}_layer")
    generator = torch.Generator().manual_seed(4000 + rows)
    gen = torch.Generator(device="cuda").manual_seed(4000 + rows)
    x_q, x_s = _quantized_rows(rows, generator)
    logits = torch.randn(rows, experts, device="cuda", generator=gen).contiguous()
    bias = torch.randn(experts, device="cuda", generator=gen) * 0.05
    num_valid = torch.tensor([rows], dtype=torch.int32, device="cuda")
    full = reg.RoutedExpertGemm(
        *layer, top_k=top_k, max_rows=reg.SERVED_MAX_ROWS, gate_topk=True
    )
    full_rows, full_w, full_permuted, full_ids = full.gate_topk_and_gemms(
        x_q, x_s, logits, bias, SCALE, num_valid
    )
    for ep_rank in range(2):
        half, offset = _ep_half(layer, ep_rank)
        gemm = reg.RoutedExpertGemm(
            *half,
            top_k=top_k,
            max_rows=reg.SERVED_MAX_ROWS,
            gate_topk=True,
            router_experts=experts,
            expert_offset=offset,
        )
        assert gemm.supports_gate_topk(logits, bias)
        rows_out, w, permuted, ids = gemm.gate_topk_and_gemms(
            x_q, x_s, logits, bias, SCALE, num_valid
        )
        torch.cuda.synchronize()
        where = f"{which} ep_rank={ep_rank} rows={rows}"
        assert torch.equal(ids, full_ids), where
        assert torch.equal(w.view(torch.int32), full_w.view(torch.int32)), where
        local = (ids >= offset) & (ids < offset + half[0].size(0))
        assert torch.equal(permuted >= 0, local), where
        got = rows_out[permuted.clamp(min=0).long()][local]
        want = full_rows[full_permuted.long()][local]
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), where


def test_an_expert_window_outside_the_router_is_refused(target_layer):
    half, _ = _ep_half(target_layer, 1)
    with pytest.raises(ValueError, match="outside the router"):
        reg.RoutedExpertGemm(
            *half, top_k=TARGET_TOP_K, max_rows=8, router_experts=384, expert_offset=200
        )


# ------------------------------------------------------- shared expert gate_up (TP8)

needs_ll = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)


@needs_ll
@pytest.mark.parametrize("rows", [1, 6, 8, 12, 24, 48])
def test_shared_gate_up_at_the_tp8_shard(rows):
    """The TP8 shard's gate_up (N 576 = 4.5 tiles of 128) on the low-latency GEMM: with
    one tactic, each output column equals the TP4 shard's (N 1152) kernel on the same
    rows bit for bit (a column depends on its weight row only), and the shared expert
    slot accepts the shard."""
    from types import SimpleNamespace

    from flashinfer import mxfp8_quantize

    from sgl_nvidia_mega.deepseek_v41.moe import shared_ll
    from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll
    from sgl_nvidia_mega.deepseek_v41.sm103.attention.q_norm_quant import (
        mxfp8_row_tiles,
    )

    if not mxfp8_ll.supported():
        pytest.skip("no low-latency MXFP8 GEMM on this GPU")
    gen = torch.Generator(device="cuda").manual_seed(77)
    k = reg.HIDDEN
    w = (torch.randn(1152, k, device="cuda", generator=gen) / k**0.5).bfloat16()
    w_q, w_s = mxfp8_quantize(w, is_sf_swizzled_layout=True)
    shard_q, shard_s = mxfp8_quantize(w[:576].contiguous(), is_sf_swizzled_layout=True)
    assert torch.equal(shard_q.view(torch.uint8), w_q[:576].view(torch.uint8))
    x = torch.randn(rows, k, device="cuda", generator=gen).bfloat16()
    x_q, x_s = mxfp8_row_tiles(x)
    major, minor = torch.cuda.get_device_capability()
    bucket = mxfp8_ll.row_bucket(rows)
    tactic = mxfp8_ll.select_tactic(major * 10 + minor, 576, k, bucket)
    shard = mxfp8_ll.linear(
        x_q, x_s, shard_q.contiguous(), shard_s.reshape(-1), tactic=tactic
    )
    whole = mxfp8_ll.linear(x_q, x_s, w_q.contiguous(), w_s.reshape(-1), tactic=tactic)
    torch.cuda.synchronize()
    assert torch.equal(
        shard.view(torch.int16), whole[:, :576].contiguous().view(torch.int16)
    ), rows
    default = mxfp8_ll.linear(x_q, x_s, shard_q.contiguous(), shard_s.reshape(-1))
    # against BF16 operands: the MXFP8 rounding of both, a few percent of the norm
    ref = x.float() @ w[:576].float().t()
    assert (default.float() - ref).norm() <= 0.05 * ref.norm(), rows

    gate_up = SimpleNamespace(
        weight=shard_q.contiguous(), weight_scale_inv_swizzled=shard_s.reshape(-1)
    )
    shared = SimpleNamespace(
        gate_up_proj=gate_up,
        down_proj=SimpleNamespace(weight=None),
        swiglu_limit=10.0,
        use_fused_clamp_act_mul=False,
        allow_fused_down=False,
    )
    mlp = SimpleNamespace(shared_experts=shared, num_fused_shared_experts=0)
    assert shared_ll.why_not(mlp) == ""


# ------------------------- slots gate_route (top-k + route fused) and gate_route_quant


def _gate_route_logits(kind, rows, experts, gen):
    if kind == "one_expert":  # every row's first choice is the same expert
        x = torch.randn((rows, experts), device="cuda", generator=gen)
        x[:, 7] = 30.0
        return x
    if kind == "cutoff_ties":  # equal scores straddle the top-k cutoff
        x = torch.randn((rows, experts), device="cuda", generator=gen) - 4.0
        x[:, 3:12] = 2.0
        return x
    return _logits(kind, rows, experts, gen)


def _routed_input(kind, rows, seed):
    """BF16 routed-input rows; "special" puts every awkward scale block in each row."""
    gen = torch.Generator().manual_seed(seed)
    h = reg.HIDDEN
    x = torch.randn(rows, h, generator=gen)
    if kind == "wide":  # each 32-wide block at its own power of two, 2^-130 .. 2^126
        exps = torch.randint(-130, 127, (rows, h // 32), generator=gen).float()
        x = x * torch.exp2(exps).repeat_interleave(32, dim=1)
    elif kind == "special":
        blocks = x.view(rows, h // 32, 32)
        blocks[:, 0] = 0.0
        blocks[:, 1] = -0.0
        blocks[:, 2] = 1e-40  # BF16 subnormals
        blocks[:, 3, ::3] = -1e-39
        blocks[:, 4, 5] = float("inf")
        blocks[:, 5, 7] = float("-inf")
        blocks[:, 6, 1] = float("nan")
        blocks[:, 7] = float("nan")
        blocks[:, 8] = 448.0
        blocks[:, 9] = torch.linspace(-449.0, 449.0, 32)
        blocks[:, 10] = 3.0e38
        blocks[:, 11, 0] = 2.0**-127  # the amax at the UE8M0 subnormal boundary
        blocks[:, 12] = 2.0**-133
        blocks[:, 13] = 1.5 * 2.0**-120
    return x.bfloat16().cuda().contiguous()


def _quantize_routed_input(x):
    """SGLang's MXFP8 quant of the routed input (Mxfp4FlashinferTrtllmMoEMethod)."""
    from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
        Mxfp4FlashinferTrtllmMoEMethod,
    )

    return Mxfp4FlashinferTrtllmMoEMethod.quantize_routed_input(None, x, reg.HIDDEN)


def _gate_route_pair(layer, *, top_k, router_experts, offset, route=True, quant=False):
    return reg.RoutedExpertGemm(
        *layer,
        top_k=top_k,
        max_rows=reg.SERVED_MAX_ROWS,
        gate_topk=True,
        router_experts=router_experts,
        expert_offset=offset or 0,
        gate_route=route,
        gate_route_quant=quant,
        **GEMM2_OPTIONS,
    )


def _run_gate(gemm, graph, *args, **kw):
    """gate_topk_and_gemms, eager or captured and replayed into cleared outputs."""
    if not graph:
        out = gemm.gate_topk_and_gemms(*args, **kw)
    else:
        gemm.gate_topk_and_gemms(*args, **kw)
        torch.cuda.synchronize()
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            out = gemm.gate_topk_and_gemms(*args, **kw)
        for t in out:
            t.zero_()
        for _ in range(3):
            captured.replay()
    torch.cuda.synchronize()
    return out


def _assert_same_route(got, want, where):
    """ids, weights, slot -> row map and the kept slots' GEMM2 rows bit for bit."""
    (r1, w1, p1, i1), (r0, w0, p0, i0) = got, want
    assert torch.equal(i1, i0), where
    assert torch.equal(w1.view(torch.int32), w0.view(torch.int32)), where
    assert torch.equal(p1, p0), where
    kept = p0 >= 0
    a = r1[p1.clamp(min=0).long()][kept]
    b = r0[p0.clamp(min=0).long()][kept]
    assert torch.equal(a.view(torch.int16), b.view(torch.int16)), where


# (router, rows): the served batch-1 widths (target verify 6, DSpark draft 5) and a
# target width past gate_route's 8 rows, which keeps the separate kernels
GATE_ROUTE_CASES = [("target", 6), ("target", 12), ("draft", 5)]
# (vision routing, padded rows, CUDA graph)
GATE_ROUTE_VARIANTS = [(False, False, False), (True, True, False), (True, False, True)]


def _gate_route_case(request, which, rows, window, kind, seed):
    layer = request.getfixturevalue(f"{which}_layer")
    top_k = TARGET_TOP_K if which == "target" else DRAFT_TOP_K
    router_experts = layer[0].size(0)
    offset = None
    if window != "all":
        layer, offset = _ep_half(layer, int(window[-1]))
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = _routed_input("normal", rows, seed)
    logits = _gate_route_logits(kind, rows, router_experts, gen).contiguous()
    bias = torch.randn(router_experts, device="cuda", generator=gen) * 0.05
    return layer, top_k, router_experts, offset, x, logits, bias, gen


def _gate_args(rows, router_experts, gen, vision, padded):
    valid = max(1, rows - 2) if padded else rows
    num_valid = torch.tensor([valid], dtype=torch.int32, device="cuda")
    if not vision:
        return num_valid, dict(alt=None, renormalize_epsilon=0.0)
    bias_alt = torch.randn(router_experts, device="cuda", generator=gen) * 0.5
    input_ids = torch.randint(0, 50, (rows,), device="cuda", generator=gen)
    input_ids[::2] = 17  # every other row is an image token
    return num_valid, dict(alt=(bias_alt, input_ids, 17), renormalize_epsilon=EPS)


@needs_moe
@pytest.mark.parametrize("which, rows", GATE_ROUTE_CASES)
@pytest.mark.parametrize("kind", ["normal", "cutoff_ties", "one_expert", "nan"])
@pytest.mark.parametrize("window", ["all", "ep1"])
def test_gate_route_matches_gate_topk_and_route(request, which, rows, kind, window):
    """Slot gate_route against the gate_topk + route kernels on the same layer half,
    and against SGLang's gate on the same logits."""
    case = _gate_route_case(request, which, rows, window, kind, 5000 + rows)
    layer, top_k, router_experts, offset, _, logits, bias, gen = case
    x_q, x_s = _quantized_rows(rows, torch.Generator().manual_seed(5000 + rows))
    gemms = [
        _gate_route_pair(
            layer, top_k=top_k, router_experts=router_experts, offset=offset, route=r
        )
        for r in (False, True)
    ]
    assert gemms[1].serves_gate_route(rows) == (rows <= reg.MAX_ROWS)
    for vision, padded, graph in GATE_ROUTE_VARIANTS:
        num_valid, kw = _gate_args(rows, router_experts, gen, vision, padded)
        args = (x_q, x_s, logits, bias, SCALE, num_valid)
        want, got = (_run_gate(g, graph, *args, **kw) for g in gemms)
        where = f"{which} rows={rows} {kind} {window} vision={vision} graph={graph}"
        _assert_same_route(got, want, where)
        if not graph:  # the stock gate on the same logits: the router's choice
            sw, sid = _stock(logits, bias, num_valid, top_k, kw["alt"])
            assert torch.equal(got[3], sid), where
            assert torch.equal(got[1].view(torch.int32), sw.view(torch.int32)), where


@needs_moe
@pytest.mark.parametrize("kind", ["wide", "special"])
@pytest.mark.parametrize("which, rows", [("target", 6), ("draft", 5)])
def test_gate_route_quant_matches_quantize_routed_input(request, which, rows, kind):
    """The fused launch's MXFP8 values and linear scales equal SGLang's routed-input quant
    (FlashInfer's mxfp8_quantize) bit for bit, subnormal, infinite and NaN blocks too."""
    layer, top_k, router_experts, offset, _, logits, bias, _ = _gate_route_case(
        request, which, rows, "ep1", "normal", rows
    )
    gemm = _gate_route_pair(
        layer, top_k=top_k, router_experts=router_experts, offset=offset, quant=True
    )
    assert gemm.serves_gate_route_quant(rows)
    x = _routed_input(kind, rows, 900 + rows)
    num_valid = torch.tensor([rows], dtype=torch.int32, device="cuda")
    seen = {}
    real = gemm._gemms

    def capture(x_quant, x_scale, record, ctas):
        seen["q"], seen["s"] = x_quant, x_scale
        return real(x_quant, x_scale, record, ctas)

    gemm._gemms = capture
    gemm.gate_topk_and_gemms(None, None, logits, bias, SCALE, num_valid, x=x)
    want_q, want_s = _quantize_routed_input(x)
    torch.cuda.synchronize()
    assert torch.equal(seen["q"].view(torch.uint8), want_q.view(torch.uint8))
    assert torch.equal(
        seen["s"].view(torch.uint8).flatten(), want_s.view(torch.uint8).flatten()
    )


@needs_moe
@pytest.mark.parametrize("which, rows", [("target", 6), ("draft", 5)])
@pytest.mark.parametrize("kind", ["normal", "one_expert"])
@pytest.mark.parametrize("window", ["all", "ep1"])
def test_gate_route_quant_matches_gate_route(request, which, rows, kind, window):
    """Slot gate_route_quant on the BF16 rows against slot gate_route on SGLang's quant
    of them."""
    case = _gate_route_case(request, which, rows, window, kind, 7000 + rows)
    layer, top_k, router_experts, offset, x, logits, bias, gen = case
    x_q, x_s = _quantize_routed_input(x)
    gemm = _gate_route_pair(
        layer, top_k=top_k, router_experts=router_experts, offset=offset, quant=True
    )
    for vision, padded, graph in GATE_ROUTE_VARIANTS:
        num_valid, kw = _gate_args(rows, router_experts, gen, vision, padded)
        args = (logits, bias, SCALE, num_valid)
        want = _run_gate(gemm, graph, x_q, x_s, *args, **kw)
        got = _run_gate(gemm, graph, None, None, *args, x=x, **kw)
        _assert_same_route(got, want, f"{which} {kind} {window} {vision} {graph}")


@needs_moe
def test_gate_route_quant_refuses_what_it_does_not_serve(target_layer):
    half, offset = _ep_half(target_layer, 0)
    common = dict(top_k=TARGET_TOP_K, router_experts=TARGET_EXPERTS, offset=offset)
    gemm = _gate_route_pair(half, quant=True, **common)
    x = _routed_input("normal", 12, 1)
    assert not gemm.serves_gate_route_quant(12, None, x)  # past gate_route's 8 rows
    assert not gemm.serves_gate_route_quant(6, None, x[:6].float())
    assert not gemm.serves_gate_route_quant(6, None, x[:6, :4096])
    logits = torch.randn(12, TARGET_EXPERTS, device="cuda")
    bias = torch.zeros(TARGET_EXPERTS, device="cuda")
    with pytest.raises(ValueError, match="gate_route_quant"):
        gemm.gate_topk_and_gemms(None, None, logits, bias, SCALE, None, x=x)
    # the quant rides on gate_route's launch
    plain = _gate_route_pair(half, route=False, quant=True, **common)
    assert plain.gate_route_quant_kernel is None
    assert not plain.serves_gate_route_quant(6)
