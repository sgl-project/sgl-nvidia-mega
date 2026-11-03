# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright (c) 2024, Tri Dao.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# This file has been modified from its original version (a static FP8 store).
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/kernels/ops/attention/fla/layernorm_gated.py (_layer_norm_fwd_1pass_kernel,
# _layer_norm_fwd), itself adapted from flash-linear-attention (MIT, Tri Dao).
"""The gated RMSNorm of a GDN verify step writing static FP8 for the FP8 ``out_proj``:
SGLang's kernel and launch, plus a store that rounds each output to the activation
dtype, scales it by the linear's static input scale and casts it to e4m3, as the
linear's own input quantization would.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/ops/attention/fla/layernorm_gated.py": ("calc_rows_per_block",),
    "kernels/jit/utils/__init__.py": ("is_arch_support_pdl",),
}

# The SGLang functions whose body this module mirrors, by the digest of their code
# (test/unit/test_contract.py recomputes them on the installed SGLang)
SGLANG_MIRRORS = {
    "kernels/ops/attention/fla/layernorm_gated.py:_layer_norm_fwd_1pass_kernel": "80c8989ca591478d",
    "kernels/ops/attention/fla/layernorm_gated.py:_layer_norm_fwd": "01a3c1896f273538",
}


@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X,  # pointer to the input
    Q,  # pointer to the FP8 output
    W,  # pointer to the weights
    Z,  # pointer to the other branch
    S,  # pointer to the static input scale of the consumer
    Rstd,  # pointer to the 1/std
    stride_x_row,
    stride_q_row,
    stride_z_row,
    stride_z_token,
    stride_z_head,
    M,  # number of rows in X
    N: tl.constexpr,  # number of columns in X
    eps,
    BLOCK_N: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    Z_IS_3D: tl.constexpr,
    Z_HEADS: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    ACTIVATION: tl.constexpr,
    OUT_DTYPE: tl.constexpr,  # the activation dtype each output is rounded to first
    USE_GDC: tl.constexpr = False,
    EARLY_DEPENDENTS: tl.constexpr = False,
):
    if USE_GDC:
        tl.extra.cuda.gdc_wait()
        # the next launch loads only static data until its own wait
        if EARLY_DEPENDENTS:
            tl.extra.cuda.gdc_launch_dependents()

    row_start = tl.program_id(0) * ROWS_PER_BLOCK
    group = tl.program_id(1)
    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, BLOCK_N)
    row_offsets = rows[:, None] * stride_x_row
    col_offsets = cols[None, :] + group * N
    X_base = X + row_offsets + col_offsets
    row_mask = rows[:, None] < M
    col_mask = cols[None, :] < N
    mask = row_mask & col_mask
    x = tl.load(X_base, mask=mask, other=0.0).to(tl.float32)

    if Z_IS_3D:
        z_row_offsets = (rows[:, None] // Z_HEADS) * stride_z_token + (
            rows[:, None] % Z_HEADS
        ) * stride_z_head
    else:
        z_row_offsets = rows[:, None] * stride_z_row
    if not NORM_BEFORE_GATE:
        z = tl.load(Z + z_row_offsets + col_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            x *= z * tl.sigmoid(z)
        elif ACTIVATION == "sigmoid":
            x *= tl.sigmoid(z)

    # RMS norm
    xbar = tl.where(mask, x, 0.0)
    var = tl.sum(xbar * xbar, axis=1) / N
    rstd = tl.rsqrt(var + eps)
    rstd_offsets = group * M + rows
    tl.store(Rstd + rstd_offsets, rstd, mask=rows < M)

    w_offsets = cols + group * N
    w = tl.load(W + w_offsets, mask=cols < N, other=0.0).to(tl.float32)
    x_hat = x * rstd[:, None]
    y = x_hat * w[None, :]

    if NORM_BEFORE_GATE:
        z = tl.load(Z + z_row_offsets + col_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            y *= z * tl.sigmoid(z)
        elif ACTIVATION == "sigmoid":
            y *= tl.sigmoid(z)

    # the consumer's static per-tensor quantization of the rounded activation
    y = y.to(OUT_DTYPE).to(tl.float32)
    y_s = tl.load(S).to(tl.float32)
    y_q = tl.clamp(y * (1.0 / y_s), -448.0, 448.0).to(Q.dtype.element_ty)
    tl.store(Q + rows[:, None] * stride_q_row + col_offsets, y_q, mask=mask)

    if USE_GDC and not EARLY_DEPENDENTS:
        tl.extra.cuda.gdc_launch_dependents()


def rms_norm_gated_fp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    z: torch.Tensor,
    scale: torch.Tensor,
    *,
    norm_before_gate: bool = True,
    activation: str = "swish",
    early_dependents: bool = False,
) -> torch.Tensor:
    """e4m3 of RMSNorm(x) * act(z) [M, N] (rounded to x's dtype first) over the static
    input ``scale`` [1] of the linear it feeds; z [M, N] or [tokens, heads, N]. With
    ``early_dependents`` the next launch may start (and load its weights) at once."""
    from sglang.kernels.jit.utils import is_arch_support_pdl
    from sglang.kernels.ops.attention.fla.layernorm_gated import calc_rows_per_block

    M, N = x.shape
    z_is_3d = z.ndim == 3
    q = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    block_n = min(65536 // x.element_size(), triton.next_power_of_2(N))
    num_warps = min(max(block_n // 256, 1), 8)
    rows_per_block = calc_rows_per_block(M, x.device)
    pdl_kwargs = (
        {"USE_GDC": True, "launch_pdl": True, "EARLY_DEPENDENTS": early_dependents}
        if is_arch_support_pdl()
        else {}
    )
    out_dtype = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}[x.dtype]
    _layer_norm_fwd_1pass_kernel[(triton.cdiv(M, rows_per_block), 1)](
        x,
        q,
        weight,
        z,
        scale,
        rstd,
        x.stride(0),
        q.stride(0),
        0 if z_is_3d else z.stride(0),
        z.stride(0) if z_is_3d else 0,
        z.stride(1) if z_is_3d else 0,
        M,
        N,
        eps,
        BLOCK_N=block_n,
        ROWS_PER_BLOCK=rows_per_block,
        Z_IS_3D=z_is_3d,
        Z_HEADS=z.shape[1] if z_is_3d else 1,
        NORM_BEFORE_GATE=norm_before_gate,
        ACTIVATION=activation,
        OUT_DTYPE=out_dtype,
        num_warps=num_warps,
        **pdl_kwargs,
    )
    return q
