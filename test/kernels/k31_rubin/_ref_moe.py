# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Independent references for the K3.1 routed MoE (stock method) at the TP8 per-rank
shapes, and the host's FlashInfer MXFP4 weight layout.

Stock method (host ``kimi_k3.py`` fused front + FlashInfer trtllm-gen MXFP4 runner,
rc3 method fix "FP32 latent before MXFP8"):
  x (FP32 latent [3584]) -> MXFP8 (per 32-block UE8M0 scale, E4M3 values)
  FC1 = dequant(W13 MXFP4) @ x_q   rows [gate 0..383 | up 384..767] of this rank
  h = SiTU(gate, up) = beta*tanh(g/beta)*sigmoid(g) * lb*tanh(u/lb), beta 4, lb 25
  h -> MXFP8 (hidden; an all-zero block uses scale 2^0)
  y_e = dequant(W2 MXFP4) @ h_q -> BF16 row per (token, expert)
  out_t = sum_j w_tj * y_{e_tj} (FP32, then BF16)

MXFP8 scale rule (independent source: FlashInfer / TRT-LLM's quantizer as used by the
kprobe reference that matched FlashInfer on SM107): scale = 2^ceil(log2(amax / 448)),
448 = E4M3 max, so every scaled value is <= 448; computed here in FP64. Inputs clamp
amax at 1e-10 first. E4M3 rounding: round-to-nearest-even (torch's float8_e4m3fn cast).

The weights are generated in the logical (checkpoint) layout and shuffled with the same
transform the host applies after loading (``mxfp4.py`` K3 latent-MoE branch, identical
on the host branches): w13 rows reordered into (up_i, gate_i) pairs, then
FlashInfer's device permute (epilogue tile 128) and ``nvfp4_block_scale_interleave`` of
the scales. The references always use the logical weights.
"""

import math
import os

import torch

H_LAT = 3584
I_RANK = 384
E_ALL = 896
TOPK = 16
BETA, LIN = 4.0, 25.0
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
EXPERT_BYTES = 768 * 1792 + 3584 * 192 + 768 * 112 + 3584 * 12  # 2,193,408 B per rank


def make_logical(experts, gen, scale_lo=119, scale_hi=123):
    """Random logical MXFP4 weights: w13 [E, 768, 1792] (rows: gate 0..383, up 384..767),
    s13 [E, 768, 112], w2 [E, 3584, 192], s2 [E, 3584, 12] (UE8M0 bytes)."""
    u8 = dict(dtype=torch.uint8, device="cuda", generator=gen)
    w13 = torch.randint(0, 256, (experts, 2 * I_RANK, H_LAT // 2), **u8)
    s13 = torch.randint(scale_lo, scale_hi, (experts, 2 * I_RANK, H_LAT // 32), **u8)
    w2 = torch.randint(0, 256, (experts, H_LAT, I_RANK // 2), **u8)
    s2 = torch.randint(scale_lo, scale_hi, (experts, H_LAT, I_RANK // 32), **u8)
    return w13, s13, w2, s2


def host_layout(w13, s13, w2, s2):
    """The host's post-load FlashInfer MXFP4 transform (K3 non-interleaved latent MoE).
    Returns uint8 (W13, S13, W2, S2) in the layout the MK and FlashInfer consume."""
    from flashinfer import nvfp4_block_scale_interleave
    from flashinfer.fused_moe.core import get_w2_permute_indices_with_cache

    experts, two_i, _ = w13.shape
    half = two_i // 2
    pair = torch.empty(two_i, dtype=torch.long)
    pair[0::2] = torch.arange(half) + half  # up (w3)
    pair[1::2] = torch.arange(half)  # gate (w1)
    pair = pair.cuda()
    w13 = w13[:, pair, :].contiguous()
    s13 = s13[:, pair, :].contiguous()
    caches = [{}, {}, {}, {}]
    p13 = get_w2_permute_indices_with_cache(caches[0], w13[0], 128).to(w13.device)
    p13s = get_w2_permute_indices_with_cache(caches[1], s13[0], 128, num_elts_per_sf=16).to(w13.device)
    p2 = get_w2_permute_indices_with_cache(caches[2], w2[0], 128).to(w13.device)
    p2s = get_w2_permute_indices_with_cache(caches[3], s2[0], 128, num_elts_per_sf=16).to(w13.device)
    W13 = w13[:, p13, :].contiguous()
    W2 = w2[:, p2, :].contiguous()
    S13 = torch.stack([nvfp4_block_scale_interleave(s13[e][p13s].contiguous()) for e in range(experts)])
    S2 = torch.stack([nvfp4_block_scale_interleave(s2[e][p2s].contiguous()) for e in range(experts)])
    S13 = S13.reshape(experts, two_i, H_LAT // 32).view(torch.uint8).contiguous()
    S2 = S2.reshape(experts, H_LAT, I_RANK // 32).view(torch.uint8).contiguous()
    return W13, S13, W2, S2


def deq_mxfp4(packed, scale, dtype=torch.float64):
    """[R, K/2] u8 (low nibble = even element), [R, K/32] UE8M0 -> [R, K]."""
    lut = torch.tensor(E2M1, dtype=dtype, device=packed.device)
    lo, hi = lut[(packed & 0xF).long()], lut[(packed >> 4).long()]
    v = torch.stack([lo, hi], dim=-1).reshape(packed.shape[0], -1)
    s = torch.exp2(scale.to(dtype) - 127.0).repeat_interleave(32, dim=1)
    return v * s


def mx8_exponent(amax, hidden):
    """Biased UE8M0 exponent of each block's scale (FP64 log2; see module doc)."""
    a = amax.double()
    if hidden:
        e = torch.ceil(torch.log2(a.clamp(min=1e-300) / 448.0)) + 127.0
        e = torch.where(a == 0, torch.full_like(e, 127.0), e)
    else:
        e = torch.ceil(torch.log2(a.clamp(min=1e-10) / 448.0)) + 127.0
    return e.clamp(0, 254)


def mx8_qdq(x, hidden=False):
    """MXFP8 quantize + dequantize along the last dim (blocks of 32), from FP32 values.
    Returns (dequantized FP64, scale exponents)."""
    shape = x.shape
    g = x.float().reshape(*shape[:-1], shape[-1] // 32, 32)
    e = mx8_exponent(g.abs().amax(-1, keepdim=True), hidden)
    s = torch.exp2(e - 127.0)
    q = (g.double() / s).float().to(torch.float8_e4m3fn)
    return (q.double() * s).reshape(shape), e.squeeze(-1)


def situ(g, u):
    return BETA * torch.tanh(g / BETA) * torch.sigmoid(g) * (LIN * torch.tanh(u / LIN))


def expert_rows(logical, x, ids):
    """Per-(token, slot) FP64 expert outputs [B, 16, 3584] of the stock method.
    ``x``: FP32 [B, 3584]; ``ids``: int [B, 16]."""
    w13, s13, w2, s2 = logical
    xq, _ = mx8_qdq(x)
    b = x.shape[0]
    out = torch.empty((b, TOPK, H_LAT), dtype=torch.float64, device=x.device)
    flat = ids.reshape(-1).long()
    for e in torch.unique(flat).tolist():
        where = (flat == e).nonzero().flatten()
        tokens = where // TOPK
        w1 = deq_mxfp4(w13[e], s13[e])  # [768, 3584]
        h1 = xq[tokens] @ w1.T  # [n, 768]
        a = situ(h1[:, :I_RANK], h1[:, I_RANK:])
        aq, _ = mx8_qdq(a.float(), hidden=True)
        y = aq @ deq_mxfp4(w2[e], s2[e]).T  # [n, 3584]
        out.view(-1, H_LAT)[where] = y
    return out


def weighted(rows, weights):
    """sum_j w_tj * rows[t, j] in FP64 (rows: [B, 16, 3584] any float dtype)."""
    return (rows.double() * weights.double().unsqueeze(-1)).sum(1)


def rel_l2(out, ref):
    o, r = out.double(), ref.double()
    return float((o - r).norm() / r.norm().clamp(min=1e-300))


def random_route(batch, gen, *, distinct=False, pool=None):
    """Top-16 ids [B, 16] int32 and positive weights summing to 1 per token. ``distinct``:
    all B*16 ids distinct; else each token picks 16 distinct experts uniformly. Both draw
    from ``pool`` (a 1-D id tensor) when given, else from all 896 experts."""
    pool = pool if pool is not None else torch.arange(E_ALL, device="cuda")
    n = pool.numel()
    if distinct:
        ids = pool[torch.randperm(n, generator=gen, device="cuda")[: batch * TOPK]].view(batch, TOPK)
    else:
        scores = torch.rand(batch, n, generator=gen, device="cuda")
        ids = pool[torch.topk(scores, TOPK, dim=-1).indices]
    w = torch.rand(batch, TOPK, generator=gen, device="cuda") + 0.1
    w = w / w.sum(-1, keepdim=True)
    return ids.int().contiguous(), w.float().contiguous()


def fi_routed(prepared, xq, xs, ids, w, out):
    """Stock FlashInfer trtllm-gen MXFP4 x MXFP8 routed MoE (the kprobe call, which is
    dev's runner call), output BF16 [B, 3584] = weighted sum over the 16 experts."""
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
    from flashinfer.tllm_enums import ActivationType, RoutingMethodType

    W13, S13, W2, S2 = prepared
    m = xq.shape[0]
    e = W13.shape[0]
    alpha = torch.full((e,), BETA, dtype=torch.float32, device="cuda")
    beta = torch.full((e,), LIN, dtype=torch.float32, device="cuda")
    kw = dict(
        topk_ids=(ids, w), routing_bias=None, hidden_states=xq, hidden_states_scale=xs,
        gemm1_weights=W13, gemm1_weights_scale=S13.view(torch.float8_e4m3fn), gemm1_bias=None,
        gemm1_alpha=alpha, gemm1_beta=beta, gemm1_clamp_limit=None, gemm2_weights=W2,
        gemm2_weights_scale=S2.view(torch.float8_e4m3fn), gemm2_bias=None,
        output1_scale_scalar=None, output1_scale_gate_scalar=None, output2_scale_scalar=None,
        num_experts=E_ALL, top_k=TOPK, n_group=None, topk_group=None, intermediate_size=I_RANK,
        local_expert_offset=0, local_num_experts=e, routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.TopK.value, activation_type=ActivationType.Situ.value,
        tune_max_num_tokens=1 << max(0, (m - 1).bit_length()), output=out, do_finalize=True,
        enable_pdl=True,
    )
    return lambda: trtllm_fp4_block_scale_routed_moe(**kw)


def fi_inputs(x):
    """MXFP8 of the FP32 latent for FlashInfer: (E4M3 [B, 3584], UE8M0 as fp8 view [B, 112])."""
    b = x.shape[0]
    g = x.float().reshape(b, H_LAT // 32, 32)
    e = mx8_exponent(g.abs().amax(-1, keepdim=True), False)
    q = (g.double() / torch.exp2(e - 127.0)).float().to(torch.float8_e4m3fn).reshape(b, H_LAT)
    xs = e.squeeze(-1).to(torch.uint8).contiguous().view(torch.float8_e4m3fn)
    return q.contiguous(), xs


def floor_us(distinct):
    nbytes = distinct * EXPERT_BYTES
    bps = {k: float(os.environ.get(f"K31_TEST_{k}", "nan")) for k in ("HBM_BPS", "HBM_LOCAL_BPS", "HBM_LOCAL_PEAK_BPS")}
    return {"bytes": nbytes, "floor_us": nbytes / bps["HBM_BPS"] * 1e6, "floor_local_us": nbytes / bps["HBM_LOCAL_BPS"] * 1e6,
            "floor_local_peak_us": nbytes / bps["HBM_LOCAL_PEAK_BPS"] * 1e6, "distinct": int(distinct)}


_WEIGHTS = {}


def shared_weights(seed=31):
    """(logical, host-layout) weights for all 896 experts, built once per process."""
    if seed not in _WEIGHTS:
        import _harness as H

        logical = make_logical(E_ALL, H.generator(seed))
        _WEIGHTS[seed] = logical, host_layout(*logical)
    return _WEIGHTS[seed]


def nan_safe(x):
    return x if math.isfinite(x) else float("inf")
