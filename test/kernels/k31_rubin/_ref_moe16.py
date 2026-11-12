# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Independent references for the K3.1 routed MoE at the TP16 per-rank shapes.

Same stock method as ``_ref_moe`` (FP32 latent -> MXFP8 -> FP4 x MXFP8 FC1 -> SiTU -> MXFP8 hidden -> FC2 -> BF16),
with the TP16 per-rank intermediate I = 3072 / 16 = 192. The host pads it to 256 for FlashInfer (``mxfp4.py``:
SM100 FlashInfer rounds the intermediate up to a multiple of 128; ``FusedMoE._load_w13`` / ``_load_w2`` load the
checkpoint slice at the start of each half and zero the rest, weights and scale bytes alike), then applies the same
K3 latent-MoE transform as TP8 ((up_i, gate_i) row pairs, FlashInfer permute with epilogue tile 128, block-scale
interleave). ``host_layout16`` reproduces that from the logical weights; the references always use the logical,
unpadded weights.

``tp8_halves`` splits one TP8 rank's logical weights (I = 384) into the two TP16 ranks that own its intermediate
halves (TP16 rank 2q + h owns TP8 rank q's rows [192 h, 192 h + 192) of gate and of up, and FC2's matching K slice),
for the TP8 cross-check: the TP8 rank's expert row equals the sum of the two TP16 rows up to rounding.
"""

import torch

import _ref_moe as R

I16, I16_PAD = 192, 256


def make_logical16(experts, gen, scale_lo=119, scale_hi=123):
    """Random logical TP16 MXFP4 weights: w13 [E, 384, 1792] (gate 0..191, up 192..383), s13 [E, 384, 112],
    w2 [E, 3584, 96], s2 [E, 3584, 6]."""
    u8 = dict(dtype=torch.uint8, device="cuda", generator=gen)
    w13 = torch.randint(0, 256, (experts, 2 * I16, R.H_LAT // 2), **u8)
    s13 = torch.randint(scale_lo, scale_hi, (experts, 2 * I16, R.H_LAT // 32), **u8)
    w2 = torch.randint(0, 256, (experts, R.H_LAT, I16 // 2), **u8)
    s2 = torch.randint(scale_lo, scale_hi, (experts, R.H_LAT, I16 // 32), **u8)
    return w13, s13, w2, s2


def pad16(w13, s13, w2, s2):
    """The host's padded load buffers: gate rows at [0, 192), up rows at [256, 448), FC2 K at [0, 192); zeros
    elsewhere (weights and scale bytes)."""
    e = w13.shape[0]
    w13p = w13.new_zeros((e, 2 * I16_PAD, w13.shape[2]))
    s13p = s13.new_zeros((e, 2 * I16_PAD, s13.shape[2]))
    for dst, src in ((w13p, w13), (s13p, s13)):
        dst[:, :I16] = src[:, :I16]
        dst[:, I16_PAD:I16_PAD + I16] = src[:, I16:]
    w2p = w2.new_zeros((e, R.H_LAT, I16_PAD // 2))
    s2p = s2.new_zeros((e, R.H_LAT, I16_PAD // 32))
    w2p[:, :, : I16 // 2] = w2
    s2p[:, :, : I16 // 32] = s2
    return w13p, s13p, w2p, s2p


def host_layout16(w13, s13, w2, s2):
    """uint8 (W13 [E, 512, 1792], S13 [E, 512, 112], W2 [E, 3584, 128], S2 [E, 3584, 8]) as the host prepares them:
    ``_ref_moe.host_layout``'s transform on the padded buffers (its last reshape hard-codes TP8's 12 FC2 scale
    columns, so the transform is repeated here with the padded widths)."""
    from flashinfer import nvfp4_block_scale_interleave
    from flashinfer.fused_moe.core import get_w2_permute_indices_with_cache

    w13, s13, w2, s2 = pad16(w13, s13, w2, s2)
    experts, two_i, _ = w13.shape
    half = two_i // 2
    pair = torch.empty(two_i, dtype=torch.long)
    pair[0::2] = torch.arange(half) + half  # up (w3)
    pair[1::2] = torch.arange(half)  # gate (w1)
    pair = pair.to(w13.device)
    w13, s13 = w13[:, pair, :].contiguous(), s13[:, pair, :].contiguous()
    caches = [{}, {}, {}, {}]
    p13 = get_w2_permute_indices_with_cache(caches[0], w13[0], 128).to(w13.device)
    p13s = get_w2_permute_indices_with_cache(caches[1], s13[0], 128, num_elts_per_sf=16).to(w13.device)
    p2 = get_w2_permute_indices_with_cache(caches[2], w2[0], 128).to(w13.device)
    p2s = get_w2_permute_indices_with_cache(caches[3], s2[0], 128, num_elts_per_sf=16).to(w13.device)
    W13 = w13[:, p13, :].contiguous()
    W2 = w2[:, p2, :].contiguous()
    S13 = torch.stack([nvfp4_block_scale_interleave(s13[e][p13s].contiguous()) for e in range(experts)])
    S2 = torch.stack([nvfp4_block_scale_interleave(s2[e][p2s].contiguous()) for e in range(experts)])
    S13 = S13.reshape(experts, two_i, s13.shape[2]).view(torch.uint8).contiguous()
    S2 = S2.reshape(experts, R.H_LAT, s2.shape[2]).view(torch.uint8).contiguous()
    return W13, S13, W2, S2


def expert_rows(logical, x, ids):
    """Per-(token, slot) FP64 expert outputs [B, 16, 3584] of the stock method for any per-rank intermediate
    (``w13`` rows = 2 I). ``x``: FP32 [B, 3584]; ``ids``: int [B, 16]."""
    w13, s13, w2, s2 = logical
    inter = w13.shape[1] // 2
    xq, _ = R.mx8_qdq(x)
    b = x.shape[0]
    out = torch.empty((b, R.TOPK, R.H_LAT), dtype=torch.float64, device=x.device)
    flat = ids.reshape(-1).long()
    for e in torch.unique(flat).tolist():
        where = (flat == e).nonzero().flatten()
        tokens = where // R.TOPK
        h1 = xq[tokens] @ R.deq_mxfp4(w13[e], s13[e]).T  # [n, 2 I]
        a = R.situ(h1[:, :inter], h1[:, inter:])
        aq, _ = R.mx8_qdq(a.float(), hidden=True)
        out.view(-1, R.H_LAT)[where] = aq @ R.deq_mxfp4(w2[e], s2[e]).T
    return out


def row_control(logical, x, token, expert):
    """Control for a row far from the FP64 reference (diagnostic only, no pass/fail): the same expert row with
    FC1 and SiTU in FP32 (as the kernels compute them) before the hidden MXFP8, and the smallest distance of a
    hidden block's log2(amax / 448) to an integer (a block scale boundary). If the kernel row sits close to the
    FP32 variant and a margin is tiny, the gap is a hidden block scale flip, inherent to FP32 SiTU."""
    w13, s13, w2, s2 = logical
    inter = w13.shape[1] // 2
    xq, _ = R.mx8_qdq(x[token: token + 1])
    h1 = xq @ R.deq_mxfp4(w13[expert], s13[expert]).T
    w2d = R.deq_mxfp4(w2[expert], s2[expert])
    rows, margin = {}, None
    for name, dtype in (("fp64", torch.float64), ("fp32", torch.float32)):
        h = h1.to(dtype)
        a = R.situ(h[:, :inter], h[:, inter:])
        aq, _ = R.mx8_qdq(a.float(), hidden=True)
        rows[name] = (aq @ w2d.T)[0]
        ratio = a.float().reshape(-1, 32).abs().amax(-1).double().clamp(min=1e-300) / 448.0
        dist = float((torch.log2(ratio) - torch.round(torch.log2(ratio))).abs().min())
        margin = dist if margin is None else min(margin, dist)
    return rows, margin


def tp8_halves(logical8):
    """TP8 rank weights (I = 384) -> the two TP16 ranks' logical weights (I = 192 each)."""
    w13, s13, w2, s2 = logical8
    halves = []
    for h in range(2):
        rows = torch.cat((torch.arange(192 * h, 192 * h + 192), 384 + torch.arange(192 * h, 192 * h + 192)))
        rows = rows.to(w13.device)
        halves.append((w13[:, rows].contiguous(), s13[:, rows].contiguous(),
                       w2[:, :, 96 * h: 96 * h + 96].contiguous(), s2[:, :, 6 * h: 6 * h + 6].contiguous()))
    return halves


def fi_routed16(prepared, xq, xs, ids, w, out):
    """Stock FlashInfer trtllm-gen MXFP4 x MXFP8 routed MoE on the padded TP16 weights (intermediate 256)."""
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
    from flashinfer.tllm_enums import ActivationType, RoutingMethodType

    W13, S13, W2, S2 = prepared
    m, e = xq.shape[0], W13.shape[0]
    alpha = torch.full((e,), R.BETA, dtype=torch.float32, device="cuda")
    beta = torch.full((e,), R.LIN, dtype=torch.float32, device="cuda")
    kw = dict(
        topk_ids=(ids, w), routing_bias=None, hidden_states=xq, hidden_states_scale=xs,
        gemm1_weights=W13, gemm1_weights_scale=S13.view(torch.float8_e4m3fn), gemm1_bias=None,
        gemm1_alpha=alpha, gemm1_beta=beta, gemm1_clamp_limit=None, gemm2_weights=W2,
        gemm2_weights_scale=S2.view(torch.float8_e4m3fn), gemm2_bias=None,
        output1_scale_scalar=None, output1_scale_gate_scalar=None, output2_scale_scalar=None,
        num_experts=R.E_ALL, top_k=R.TOPK, n_group=None, topk_group=None, intermediate_size=I16_PAD,
        local_expert_offset=0, local_num_experts=e, routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.TopK.value, activation_type=ActivationType.Situ.value,
        tune_max_num_tokens=1 << max(0, (m - 1).bit_length()), output=out, do_finalize=True,
        enable_pdl=True,
    )
    return lambda: trtllm_fp4_block_scale_routed_moe(**kw)


def mx8_exp(amax):
    """Biased UE8M0 exponent of the stock MXFP8 input scale 2^ceil(log2(amax / 448)) (amax clamped at 1e-10 in
    FP32), computed exactly: with amax = m 2^e, m in [0.5, 1), and 448 = 0.875 2^9, the exponent is e - 9 + (m >
    0.875). ``_ref_moe.mx8_exponent`` takes the FP64 log2, which on the GPU can miss by one at exact powers of two
    (a block maximum of 448 x 2^s, as on-grid inputs have)."""
    m, e = torch.frexp(amax.float().clamp(min=1e-10).double())
    return (e.double() - 9 + (m > 0.875).double() + 127).clamp(0, 254)


def mx8_qdq_exact(x):
    """``_ref_moe.mx8_qdq`` (input rule) with the exact exponent: (dequantized FP64, biased exponents)."""
    shape = x.shape
    g = x.float().reshape(*shape[:-1], shape[-1] // 32, 32)
    e = mx8_exp(g.abs().amax(-1, keepdim=True))
    s = torch.exp2(e - 127.0)
    q = (g.double() / s).float().to(torch.float8_e4m3fn)
    return (q.double() * s).reshape(shape), e.squeeze(-1)


def mx8_bytes(x):
    """Reference MXFP8 encoding of FP32 rows [B, 3584] (blocks of 32): (E4M3 bytes [B, 3584] uint8, UE8M0 scale
    bytes [B, 112] uint8), the stock quantizer's rule (``_ref_moe`` module doc) with the exact exponent."""
    b = x.shape[0]
    g = x.float().reshape(b, R.H_LAT // 32, 32)
    e = mx8_exp(g.abs().amax(-1, keepdim=True))
    q = (g.double() / torch.exp2(e - 127.0)).float().to(torch.float8_e4m3fn).reshape(b, R.H_LAT)
    return q.view(torch.uint8).contiguous(), e.squeeze(-1).to(torch.uint8).contiguous()


def keep_scale(x):
    """``x`` with the largest element of every 32-block whose scaled maximum rounds down to 224 in E4M3 (scaled
    maximum in (224, 232]; the scale puts it in (224, 448]) set to +-240 x 2^s. Without this, re-quantizing such a
    block's quantized values picks scale s - 1 (same values, codes x2); with it, ``on_grid(x)`` re-quantizes to
    the same scale bytes and code values."""
    b = x.shape[0]
    g = x.float().reshape(b, -1, 32).clone()
    amax, arg = g.abs().max(-1)
    e = mx8_exp(amax)
    low = (amax.double() / torch.exp2(e - 127.0)) <= 232
    big = (torch.exp2(e - 127.0) * 240).float()
    sign = torch.sign(g.gather(-1, arg.unsqueeze(-1)).squeeze(-1))
    new = torch.where(low, sign * big, g.gather(-1, arg.unsqueeze(-1)).squeeze(-1))
    g.scatter_(-1, arg.unsqueeze(-1), new.unsqueeze(-1))
    return g.reshape(x.shape).contiguous()


def on_grid(x):
    """FP32 values of the reference MXFP8 quantization of ``x`` (exactly representable), -0 made +0: quantizing
    them again with the same rule gives the same values, so a kernel run on them reproduces its run on ``x`` bit
    for bit iff it quantized ``x`` to the reference's values (``assert_on_grid`` checks the premise)."""
    xq, _ = mx8_qdq_exact(x)
    out = xq.float()
    out[out == 0] = 0.0
    return out.contiguous()


def assert_on_grid(x, xg):
    """Premise of the bitwise check: the stock rule gives ``x`` and ``xg`` the same scale bytes and code values."""
    q, s = mx8_bytes(x)
    qg, sg = mx8_bytes(xg)
    v = q.view(torch.float8_e4m3fn).float()
    vg = qg.view(torch.float8_e4m3fn).float()
    assert torch.equal(s, sg) and torch.equal(v, vg), "on_grid premise: scales or code values differ"


def canon(x):
    """FP32 with -0 made +0 (never the 0x80000000 mailbox sentinel)."""
    out = x.float().clone()
    out[out == 0] = 0.0
    return out.contiguous()
