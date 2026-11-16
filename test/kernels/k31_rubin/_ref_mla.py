# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Independent PyTorch references of the stock K3.1 MLA + DSA decode method (trtllm DSA backend),
written from the method (dtypes, rounding points, quantizers), not from the kernels under test.

Rounding points (stock): RMSNorm / LayerNorm outputs BF16; GEMM outputs BF16 with FP32 accumulation;
MLA query/KV FP8 E4M3 (scale 1, saturating); index-K FP8 with scale = max(amax, 1e-4) / 448 and value * (1/scale);
index-Q act_quant per 128 with the ue8m0 scale 2^ceil(log2(max(amax, 1e-4) * (1/448))) (stock's Indexer passes
scale_fmt="ue8m0"), value / scale; head gate
float(bf16(bf16(x @ Wg^T) * 32**-0.5)) * q_scale * 128**-0.5; logits sum_h w_h relu(q_h . k) * kscale.
"""

from __future__ import annotations

import math

import torch

FP8 = torch.float8_e4m3fn
PAGE = 64
TOPK = 2048


def bf16(x):
    return x.float().to(torch.bfloat16).float()


def fp8_bytes(x):
    """Saturating RN cast to E4M3 (cvt.rn.satfinite), returned as uint8 codes."""
    return x.float().clamp(-448.0, 448.0).to(FP8).view(torch.uint8)


def fp8_value(codes):
    return codes.view(FP8).float()


def rmsnorm(x, w, eps):
    x = x.float()
    rr = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return bf16(x * rr * w.float())


def layernorm(x, gamma, beta, eps):
    x = x.float()
    mean = x.mean(-1, keepdim=True)
    var = (x - mean).pow(2).mean(-1, keepdim=True)
    return bf16((x - mean) * torch.rsqrt(var + eps) * gamma.float() + beta.float())


def mla_query(projected, w_qa, w_qb, w_uk, eps):
    """[B, H, 576] FP8 codes: absorbed nope (512) | rope (64); q/k scales are 1."""
    B, H = projected.shape[0], w_uk.shape[0]
    qa = rmsnorm(projected[:, :1536], w_qa, eps)
    q = bf16(qa @ w_qb.float().T).view(B, H, 192)
    nope, rope = q[..., :128], q[..., 128:]
    absorbed = bf16(torch.einsum("bhp,hpl->bhl", nope, w_uk.float()))
    return torch.cat([fp8_bytes(absorbed), fp8_bytes(rope)], -1)


def kv_row(projected, w_kva, eps):
    """[B, 576] FP8 codes of the current token's MLA KV row."""
    latent = rmsnorm(projected[:, 1536:2048], w_kva, eps)
    return torch.cat([fp8_bytes(latent), fp8_bytes(bf16(projected[:, 2048:2112]))], -1)


def index_k(projected, gamma, beta, eps):
    """(FP8 codes [B,128], FP32 scales [B]) of the planar index-K row."""
    key = layernorm(projected[:, -128:], gamma, beta, eps)  # the last 128 packed rows (TP8 3648:3776)
    scale = key.abs().amax(-1).clamp_min(1e-4) / 448.0
    inv = 1.0 / scale
    return fp8_bytes(key * inv[:, None]), scale


def ue8m0(x):
    """2^ceil(log2(x)) of positive FP32 x, from the exponent bits (exact; no log2 approximation)."""
    mant, exp = torch.frexp(x.float())  # x = mant * 2^exp, mant in [0.5, 1)
    n = exp.to(torch.int32) - (mant == 0.5).to(torch.int32)
    return ((n + 127) << 23).view(torch.float32)  # 2^n built from its bits (normal range)


def index_q_quant(q):
    """(FP8 codes, FP32 scales [...]) of BF16-valued q [..., 128]: stock act_quant(q, 128, "ue8m0")."""
    amax = q.float().abs().amax(-1).clamp_min(1e-4)
    scale = ue8m0(amax * torch.tensor(1.0 / 448.0, dtype=torch.float32))
    return fp8_bytes(q.float() / scale[..., None]), scale


def index_q(projected, hidden, w_qa, eps, wq_b, w_proj):
    """(FP8 codes [B,32,128], weights FP32 [B,32]) of the index query and head gate."""
    B = projected.shape[0]
    qa = rmsnorm(projected[:, :1536], w_qa, eps)
    q = bf16(qa @ wq_b.float().T).view(B, 32, 128)
    codes, scale = index_q_quant(q)
    gate = bf16(hidden.float() @ w_proj.float().T)
    weights = bf16(gate * 32**-0.5) * scale * 128**-0.5
    return codes, weights


def attention(q, keys, scale):
    """q [H, 576] fp32, keys [n, 576] fp32 -> [H, 512] latent (fp64 softmax)."""
    s = (q.double() @ keys.double().T) * scale
    p = torch.softmax(s, -1)
    return (p @ keys[:, :512].double()).float()


def value(latent, w_uv, gate, lengths, q=1):
    """[B, H*128] from BF16 latent [B,H,512], W_UV [H,512,128], gate BF16 [B, H*128]."""
    B, H = latent.shape[:2]
    acc = bf16(torch.einsum("bhl,hlv->bhv", latent.float(), w_uv.float())).reshape(B, H * 128)
    out = bf16(acc * bf16(torch.sigmoid(gate.float())))
    live = (lengths.repeat_interleave(q) > 0).float()[:, None]
    return out * live


def gather_tokens(pages, table_row, n, width):
    """Rows of token positions 0..n-1 from a paged pool [pages, 64, width]."""
    npg = math.ceil(n / PAGE)
    return pages[table_row[:npg].long()].reshape(-1, width)[:n]


def index_scores(index_cache, table_row, q_codes, weights, n):
    """fp64 logits [n] of one query row over planar index-K pages [pages, 8448]."""
    npg = math.ceil(n / PAGE)
    pg = index_cache[table_row[:npg].long()]
    k = pg[:, : PAGE * 128].contiguous().view(FP8).double().reshape(-1, 128)[:n]
    s = pg[:, PAGE * 128 :].contiguous().view(torch.float32).double().reshape(-1)[:n]
    dots = torch.relu(k @ fp8_value(q_codes).double().T)  # [n, 32]
    return (dots * weights.double()[None, :]).sum(-1) * s


def slot_positions(table_row, num_pages):
    """Inverse map physical slot -> token position for one request (-1 if not in the table)."""
    pos = torch.full((num_pages * PAGE,), -1, dtype=torch.int64, device=table_row.device)
    t = table_row.long()
    p = torch.arange(t.numel() * PAGE, device=t.device)
    pos[t.repeat_interleave(PAGE) * PAGE + p % PAGE] = p
    return pos


def check_selection(name, selected, ref_scores, n, pos_of_slot, band_rel=2.0**-14):
    """Selected physical slots vs the fp64 exact top-k: count, distinctness, valid positions, and the
    symmetric difference only within band_rel * max|ref| of the fp64 2048th score. Returns stats."""
    sel = selected.long()
    k = min(n, TOPK)
    valid = sel[sel >= 0]
    stats = {"item": name, "n": int(n), "k": k}
    assert valid.numel() == k, f"{name}: {valid.numel()} valid entries, expected {k}"
    if n <= TOPK:
        assert bool((sel[:k] >= 0).all()) and bool((sel[k:] == -1).all()), f"{name}: trivial layout broken"
    pos = pos_of_slot[valid]
    assert bool((pos >= 0).all()) and bool((pos < n).all()), f"{name}: slot outside the request's first {n} tokens"
    assert torch.unique(pos).numel() == k, f"{name}: duplicate selections"
    ref = ref_scores[:n]
    top = torch.topk(ref, k).indices
    thr = float(ref[top].min())
    got = torch.zeros(n, dtype=torch.bool, device=ref.device)
    got[pos] = True
    want = torch.zeros_like(got)
    want[top] = True
    diff = (got ^ want).nonzero().flatten()
    band = band_rel * float(ref.abs().max())
    worst = float((ref[diff] - thr).abs().max()) if diff.numel() else 0.0
    stats.update(symdiff=int(diff.numel()), worst_gap=worst, band=band, thr=thr, ref_max=float(ref.abs().max()))
    assert worst <= band, f"{name}: symmetric difference outside the near-tie band: {stats}"
    return stats


def fp8_ulp_check(name, got, ref, *, max_frac=0.01):
    """FP8 codes: every cell within 1 code step of the reference (sign-aware), at most max_frac differing."""
    def order(c):
        c = c.view(torch.uint8).to(torch.int32)
        m = c & 0x7F
        return torch.where(c >= 0x80, -m, m)

    assert got.shape == ref.shape, (name, tuple(got.shape), tuple(ref.shape))
    gb, rb = got.view(torch.uint8), ref.view(torch.uint8)
    nan = (gb & 0x7F) == 0x7F
    d = (order(gb) - order(rb)).abs()
    stats = {
        "item": name,
        "cells": gb.numel(),
        "exact_frac": float((gb == rb).float().mean()),
        "max_steps": int(d.max()) if d.numel() else 0,
        "nan": int(nan.sum()),
    }
    differ = float((d > 0).float().mean())
    stats["differ_frac"] = differ
    assert stats["nan"] == 0, f"{name}: NaN codes {stats}"
    assert stats["max_steps"] <= 1 and differ <= max_frac, f"{name}: FP8 codes off reference {stats}"
    return stats
