# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 sparse verify MLA query for wide row counts (K31MK_MLAQ_W; dsa/verify.py).

MlaQuery (kimi_k31/k3/cutedsl/mla/query.py tiled()) runs one 8-CTA cluster per (head, 32-row tile): 192 CTAs at 128 rows and 6 heads,
one per SM, each staging and RMS-normalizing its whole 32-row x tile. In the sparse verify step it shares the SMs with the index
chain (query GEMM, logits, top-k), which is SM-throughput bound at 72-128 rows. MlaQueryWide computes the same outputs with few CTAs,
in three stream-ordered launches on the caller's stream:

    K1 norm      (rows programs): rstd = rsqrt(sum(x^2) / 1536 + eps) (div.rn, as MlaQuery); xq = bf16((x rstd) w_qa), and the current
                 token's KV row exactly as MlaQuery.insert: cache[slot, :512] = e4m3(bf16((kv rk) w_kva) k_scale_inv), rk = rsqrt(sum(kv^2)
                 / 512 + eps); cache[slot, 512:] = e4m3(k_pe k_scale_inv); rows whose slot is negative write no KV.
    K2 Q_B       torch.mm(xq, W_qb^T) -> q [rows, H x 192] BF16 (cuBLAS, FP32 accumulation)
    K3 absorb    (row blocks of 32 x H x 4 latent blocks of 128 programs; 96 at 128 rows and 6 heads): query[r, h, :512] =
                 e4m3(bf16(q nope [r, h] @ W_UK[h]) q_scale_inv) (FP32 tl.dot accumulation over the 128 nope dims); the first latent block's
                 programs also write query[r, h, 512:] = e4m3(q rope [r, h] q_scale_inv)

The roundings are MlaQuery's (bf16 xq, bf16 q, bf16 absorbed, e4m3 satfinite round-to-nearest); FP32 summation orders differ, so the
query bytes are not bitwise MlaQuery's (KL class). DCP (the query's mailbox publish) and query_type subclasses with a finish() side
output are not served here (verify.py refuses them at build). No cross-CTA or cross-stream synchronization: stream order only.
Scratch (xq, q) is shared by every profile and layer of a device: all of them run on the verify main stream, in order.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_DIV_RN = hasattr(tl.math, "div_rn")  # MlaQuery divides the square sum by 1536 with div.rn.f32
_SCRATCH: dict = {}


@triton.jit
def _mlaq_w_norm_insert(P, P_STRIDE, WQA, WKVA, XQ, CACHE, C_PAGE, C_PS, C_RS, SLOTS, KSI, EPS,
                        BLOCK: tl.constexpr, USE_DIV_RN: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    o = tl.arange(0, BLOCK)
    m = o < 1536
    x = tl.load(P + row * P_STRIDE + o, m, other=0).to(tl.float32)
    if USE_DIV_RN:
        mean = tl.math.div_rn(tl.sum(x * x, 0), 1536.0)
    else:
        mean = tl.sum(x * x, 0) / 1536.0
    rr = tl.rsqrt(mean + EPS)
    g = tl.load(WQA + o, m, other=0).to(tl.float32)
    tl.store(XQ + row * 1536 + o, ((x * rr) * g).to(tl.bfloat16), m)
    slot = tl.load(SLOTS + row).to(tl.int64)
    if slot >= 0:
        base = (slot // C_PAGE) * C_PS + (slot % C_PAGE) * C_RS
        ko = tl.arange(0, 512)
        kv = tl.load(P + row * P_STRIDE + 1536 + ko).to(tl.float32)
        rk = tl.rsqrt(tl.sum(kv * kv, 0) * (1.0 / 512.0) + EPS)
        wk = tl.load(WKVA + ko).to(tl.float32)
        ksi = tl.load(KSI)
        normed = ((kv * rk) * wk).to(tl.bfloat16).to(tl.float32)
        tl.store(CACHE + base + ko, normed * ksi)
        po = tl.arange(0, 64)
        pe = tl.load(P + row * P_STRIDE + 2048 + po).to(tl.float32)
        tl.store(CACHE + base + 512 + po, pe * ksi)


@triton.jit
def _mlaq_w_absorb_quant(Q, Q_R, WUK, OUT, QSI, ROWS, HEADS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pm = tl.program_id(0)
    h = tl.program_id(1).to(tl.int64)
    pn = tl.program_id(2)
    rm = (pm * BM + tl.arange(0, BM)).to(tl.int64)
    mrow = rm < ROWS
    kk = tl.arange(0, 128)
    a = tl.load(Q + rm[:, None] * Q_R + h * 192 + kk[None, :], mrow[:, None], other=0)
    rn = pn * BN + tl.arange(0, BN)
    b = tl.load(WUK + h * (128 * 512) + kk[:, None] * 512 + rn[None, :])
    acc = tl.dot(a, b)
    qsi = tl.load(QSI)
    v = acc.to(tl.bfloat16).to(tl.float32) * qsi
    tl.store(OUT + (rm[:, None] * HEADS + h) * 576 + rn[None, :], v, mrow[:, None])
    if pn == 0:
        ro = tl.arange(0, 64)
        r = tl.load(Q + rm[:, None] * Q_R + h * 192 + 128 + ro[None, :], mrow[:, None], other=0).to(tl.float32)
        tl.store(OUT + (rm[:, None] * HEADS + h) * 576 + 512 + ro[None, :], r * qsi, mrow[:, None])


def _scratch(device, heads, rows):
    """(xq [rows, 1536], q [rows, H*192]) views of this device's shared BF16 scratch (grown, never shrunk; sized before graph capture
    by MlaQueryWide's constructor)."""
    key = (device, heads)
    have = _SCRATCH.get(key)
    if have is None or have[0] < rows:
        bf = torch.bfloat16
        have = (rows, torch.empty(rows * 1536, dtype=bf, device=device), torch.empty(rows * heads * 192, dtype=bf, device=device))
        _SCRATCH[key] = have
    _, xq, q = have
    return xq[: rows * 1536].view(rows, 1536), q[: rows * heads * 192].view(rows, heads * 192)


def eps_value(eps) -> float:
    """The Python float of a PreparedMla's eps (a cutlass.Float32 built from a Python float, or a float)."""
    for attr in ("value", "_value"):
        v = getattr(eps, attr, None)
        if isinstance(v, (int, float)):
            return float(v)
    return float(eps)


class MlaQueryWide:
    """MlaQuery's outputs for one sparse verify profile of ``rows`` rows (see the module docstring). ``weights`` = a PreparedMla's
    (q_a norm, kv_a norm, W_qb [H*192, 1536], W_UK [H, 128, 512], W_UV, q_scale_inv [1], k_scale_inv [1])."""

    def __init__(self, weights, eps, rows):
        w_qa, w_kva, w_qb, w_uk, _, qsi, ksi = weights
        heads = w_uk.shape[0]
        if (w_qa.shape != (1536,) or w_kva.shape != (512,) or w_qb.shape != (heads * 192, 1536) or w_uk.shape != (heads, 128, 512)
                or not w_uk.is_contiguous() or not w_qb.is_contiguous()):
            raise ValueError("MlaQueryWide: K3 MLA shapes (q_a 1536, kv_a 512, Q_B H x 192 rows, W_UK [H, 128, 512] contiguous)")
        if any(t.dtype != torch.bfloat16 for t in (w_qa, w_kva, w_qb, w_uk)) or any(t.dtype != torch.float32 for t in (qsi, ksi)):
            raise ValueError("MlaQueryWide: BF16 weights and FP32 scales")
        self.w_qa, self.w_kva, self.w_qb, self.w_uk, self.qsi, self.ksi = w_qa, w_kva, w_qb, w_uk, qsi, ksi
        self.heads, self.rows, self.eps = heads, rows, float(eps)
        _scratch(w_qb.device, heads, rows)

    def run(self, projected, query, cache, slots):
        """projected [rows, >= 2112] BF16 (row stride free), query [rows, H, 576] uint8 (the query scratch), cache [pages, page, 576]
        FP8 bytes (any page / row strides, unit column stride), slots [rows] (negative: no KV row). Launches on the current stream."""
        rows, h = self.rows, self.heads
        if projected.shape[0] != rows or query.shape != (rows, h, 576) or cache.shape[-1] != 576 or cache.stride(-1) != 1:
            raise ValueError("MlaQueryWide: shapes differ from the prepared profile")
        xq, q = _scratch(projected.device, h, rows)
        c8 = cache.view(torch.uint8).view(torch.float8_e4m3fn)
        c3 = c8.view(c8.shape[0], -1, 576) if c8.dim() != 3 else c8
        _mlaq_w_norm_insert[(rows,)](projected, projected.stride(0), self.w_qa, self.w_kva, xq, c3, c3.shape[1], c3.stride(0),
                                     c3.stride(1), slots, self.ksi, self.eps, 2048, _DIV_RN, num_warps=8)
        torch.mm(xq, self.w_qb.t(), out=q)
        # 32-row blocks: small-footprint CTAs start beside the index logits kernel (an earlier layer end than
        # 64-row blocks at 80-128 rows)
        _mlaq_w_absorb_quant[((rows + 31) // 32, h, 4)](q, q.stride(0), self.w_uk, query.view(torch.float8_e4m3fn), self.qsi, rows, h,
                                                        32, 128, num_warps=4)
        return query
