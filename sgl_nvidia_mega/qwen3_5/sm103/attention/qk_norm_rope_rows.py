# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0):
# python/sglang/kernels/ops/attention/fused_qk_rmsnorm_rope_gate.py (its per-element
# arithmetic, on row tiles).
"""Q/K GemmaRMSNorm + partial NeoX RoPE over row tiles, gate left in place (Triton)."""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


def _pdl_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
        return major >= 9
    except Exception:
        return False


_ENABLE_PDL = _pdl_supported()

_ROWS = 8
_NUM_WARPS = 2


@triton.jit
def _norm_rope_rows(
    src_ptr,
    out_ptr,
    w_ptr,
    cos_sin_ptr,
    pos_ptr,
    axis_ptr,
    row0,
    n_rows,
    stride_src_t,
    stride_src_h,
    stride_out_t,
    stride_cos_t,
    stride_pos_axis,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROT: tl.constexpr,
    HALF: tl.constexpr,
    ROWS: tl.constexpr,
    EPS: tl.constexpr,
    WEIGHT_SHIFT: tl.constexpr,
    FP16: tl.constexpr,
    MROPE: tl.constexpr,
):
    OUT_DTYPE = tl.float16 if FP16 else tl.bfloat16
    rows = row0 + tl.arange(0, ROWS)
    rmask = rows < n_rows
    t = (rows // HEADS).to(tl.int64)
    h = (rows % HEADS).to(tl.int64)
    src_row = src_ptr + t * stride_src_t + h * stride_src_h
    out_row = out_ptr + t * stride_out_t + h * HEAD_DIM

    # Sum of squares over the whole head -> 1 / rms per row.
    d = tl.arange(0, HEAD_DIM)
    x = tl.load(src_row[:, None] + d[None, :], mask=rmask[:, None], other=0.0).to(
        tl.float32
    )
    w = tl.load(w_ptr + d).to(tl.float32)
    var = tl.sum(x * x, axis=1) / HEAD_DIM
    inv_rms = tl.rsqrt(var + EPS)

    # Pass-through tail [ROT, HEAD_DIM): the rounded normalised value.
    xn = (x * inv_rms[:, None] * (w[None, :] + WEIGHT_SHIFT)).to(OUT_DTYPE)
    tl.store(
        out_row[:, None] + d[None, :],
        xn,
        mask=rmask[:, None] & (d[None, :] >= ROT),
    )

    j = tl.arange(0, ROT)
    jp = j ^ HALF
    jc = j & (HALF - 1)
    xr = tl.load(src_row[:, None] + j[None, :], mask=rmask[:, None], other=0.0).to(
        tl.float32
    )
    xq = tl.load(src_row[:, None] + jp[None, :], mask=rmask[:, None], other=0.0).to(
        tl.float32
    )
    wr = tl.load(w_ptr + j).to(tl.float32)
    wq = tl.load(w_ptr + jp).to(tl.float32)
    xr = (xr * inv_rms[:, None] * (wr[None, :] + WEIGHT_SHIFT)).to(OUT_DTYPE)
    xq = (xq * inv_rms[:, None] * (wq[None, :] + WEIGHT_SHIFT)).to(OUT_DTYPE)
    xr = xr.to(tl.float32)
    xq = xq.to(tl.float32)

    if MROPE:
        axis = tl.load(axis_ptr + jc)
        pos = tl.load(
            pos_ptr + axis[None, :].to(tl.int64) * stride_pos_axis + t[:, None],
            mask=rmask[:, None],
            other=0,
        )
    else:
        pos = tl.load(pos_ptr + t, mask=rmask, other=0)[:, None]
    cs = cos_sin_ptr + pos.to(tl.int64) * stride_cos_t + jc[None, :]
    cos = tl.load(cs, mask=rmask[:, None], other=0.0).to(tl.float32)
    sin = tl.load(cs + HALF, mask=rmask[:, None], other=0.0).to(tl.float32)

    first = xr * cos - xq * sin
    second = xr * cos + xq * sin
    y = tl.where(j[None, :] < HALF, first, second)
    tl.store(out_row[:, None] + j[None, :], y.to(OUT_DTYPE), mask=rmask[:, None])


@triton.jit
def _qk_norm_rope_rows_kernel(
    q_src_ptr,
    k_src_ptr,
    q_out_ptr,
    k_out_ptr,
    q_w_ptr,
    k_w_ptr,
    cos_sin_ptr,
    pos_ptr,
    axis_ptr,
    num_tokens,
    q_progs,
    stride_q_src_t,
    stride_k_src_t,
    stride_q_out_t,
    stride_k_out_t,
    stride_cos_t,
    stride_pos_axis,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    Q_HEAD_STRIDE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROT: tl.constexpr,
    HALF: tl.constexpr,
    ROWS: tl.constexpr,
    EPS: tl.constexpr,
    WEIGHT_SHIFT: tl.constexpr,
    FP16: tl.constexpr,
    MROPE: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    if ENABLE_PDL:
        # the projection rows come from the launch before this one
        tl.extra.cuda.gdc_wait()
    pid = tl.program_id(0)
    if pid < q_progs:
        _norm_rope_rows(
            q_src_ptr,
            q_out_ptr,
            q_w_ptr,
            cos_sin_ptr,
            pos_ptr,
            axis_ptr,
            pid * ROWS,
            num_tokens * NUM_Q_HEADS,
            stride_q_src_t,
            Q_HEAD_STRIDE,
            stride_q_out_t,
            stride_cos_t,
            stride_pos_axis,
            HEADS=NUM_Q_HEADS,
            HEAD_DIM=HEAD_DIM,
            ROT=ROT,
            HALF=HALF,
            ROWS=ROWS,
            EPS=EPS,
            WEIGHT_SHIFT=WEIGHT_SHIFT,
            FP16=FP16,
            MROPE=MROPE,
        )
    else:
        _norm_rope_rows(
            k_src_ptr,
            k_out_ptr,
            k_w_ptr,
            cos_sin_ptr,
            pos_ptr,
            axis_ptr,
            (pid - q_progs) * ROWS,
            num_tokens * NUM_KV_HEADS,
            stride_k_src_t,
            HEAD_DIM,
            stride_k_out_t,
            stride_cos_t,
            stride_pos_axis,
            HEADS=NUM_KV_HEADS,
            HEAD_DIM=HEAD_DIM,
            ROT=ROT,
            HALF=HALF,
            ROWS=ROWS,
            EPS=EPS,
            WEIGHT_SHIFT=WEIGHT_SHIFT,
            FP16=FP16,
            MROPE=MROPE,
        )
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def shape_unsupported(head_dim: int, rotary_dim: int) -> str:
    half = rotary_dim // 2
    if head_dim < 16 or head_dim & (head_dim - 1):
        return f"head_dim {head_dim} is not a power of two >= 16"
    if rotary_dim % 2 or not 0 < rotary_dim <= head_dim or half & (half - 1):
        return f"rotary_dim {rotary_dim} is not twice a power of two <= head_dim"
    return ""


def qk_norm_rope_rows_supported(
    q_gate: torch.Tensor, k: torch.Tensor, head_dim: int, rotary_dim: int
) -> bool:
    """Shapes and dtypes this kernel takes; anything else keeps SGLang's kernel."""
    return (
        q_gate.is_cuda
        and q_gate.dtype in (torch.bfloat16, torch.float16)
        and k.dtype == q_gate.dtype
        and q_gate.dim() == 2
        and k.dim() == 2
        and q_gate.stride(1) == 1
        and k.stride(1) == 1
        and not shape_unsupported(head_dim, rotary_dim)
    )


def qk_gemma_rmsnorm_rope_rows(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    has_gate: bool = True,
    mrope_axis_map: Optional[torch.Tensor] = None,
    rows: int = _ROWS,
    num_warps: int = _NUM_WARPS,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    assert positions.dim() in (1, 2), f"want [T] or [3, T], got {positions.shape}"
    mrope = positions.dim() == 2
    assert mrope == (mrope_axis_map is not None), "mrope_axis_map needs [3, T]"
    if mrope:
        assert positions.shape[0] == 3 and positions.stride(1) == 1, (
            f"want [3, T] contiguous over T, got {positions.shape} "
            f"stride {positions.stride()}"
        )
        assert mrope_axis_map.shape == (rotary_dim // 2,)
    T = q_gate.shape[0]
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    q_out = torch.empty(T, q_size, dtype=q_gate.dtype, device=q_gate.device)
    k_out = torch.empty(T, kv_size, dtype=k.dtype, device=k.device)
    q_head_stride = (2 if has_gate else 1) * head_dim
    gate = (
        q_gate.view(T, num_q_heads, q_head_stride)[:, :, head_dim:]
        if has_gate
        else None
    )
    if T == 0:
        return q_out, k_out, gate

    q_progs = triton.cdiv(T * num_q_heads, rows)
    k_progs = triton.cdiv(T * num_kv_heads, rows)
    _qk_norm_rope_rows_kernel[(q_progs + k_progs,)](
        q_gate,
        k,
        q_out,
        k_out,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        mrope_axis_map if mrope else positions,
        T,
        q_progs,
        q_gate.stride(0),
        k.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        cos_sin_cache.stride(0),
        positions.stride(0) if mrope else 0,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        Q_HEAD_STRIDE=q_head_stride,
        HEAD_DIM=head_dim,
        ROT=rotary_dim,
        HALF=rotary_dim // 2,
        ROWS=rows,
        EPS=eps,
        WEIGHT_SHIFT=1.0,
        FP16=q_gate.dtype == torch.float16,
        MROPE=mrope,
        ENABLE_PDL=_ENABLE_PDL,
        num_warps=num_warps,
        **({"launch_pdl": True} if _ENABLE_PDL else {}),
    )
    return q_out, k_out, gate


WARM_TOKENS = (1, 4, 16, 17)


def warm(
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    mrope_axis_map: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    device = q_weight.device
    q_gate_size, kv_size = num_q_heads * 2 * head_dim, num_kv_heads * head_dim
    for tokens in WARM_TOKENS:
        qkv = torch.zeros(tokens, q_gate_size + 2 * kv_size, dtype=dtype, device=device)
        q_gate, k, _ = qkv.split([q_gate_size, kv_size, kv_size], dim=-1)
        flat = torch.zeros(tokens, dtype=torch.int64, device=device)
        forms = [(flat, None)]
        if mrope_axis_map is not None:
            forms.append((flat.expand(3, tokens).contiguous(), mrope_axis_map))
        for positions, axis_map in forms:
            qk_gemma_rmsnorm_rope_rows(
                q_gate,
                k,
                q_weight,
                k_weight,
                cos_sin_cache,
                positions,
                eps,
                num_q_heads,
                num_kv_heads,
                head_dim,
                rotary_dim,
                True,
                axis_map,
            )
