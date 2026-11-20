# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/kernels/ops/quantization/fp8_kernel.py
"""The shared expert's BF16 SwiGLU boundary and static FP8 quantization.

This is the activation tail of slot ``route_shared``.  Keeping it in the plugin
means the slot no longer calls a Qwen3.5-specific SGLang kernel after the fused
router/shared projection.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_and_mul_static_quant_fp8(
    x_ptr,
    y_q_ptr,
    y_s_ptr,
    width,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < width
    tl.extra.cuda.gdc_wait()
    gate = tl.load(x_ptr + row * 2 * width + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + row * 2 * width + width + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    scale = tl.load(y_s_ptr).to(tl.float32)
    tl.extra.cuda.gdc_launch_dependents()
    # Match the BF16 activation boundary before the static FP8 quantization.
    y = (gate / (1.0 + tl.exp(-gate)) * up).to(tl.bfloat16).to(tl.float32)
    q = tl.clamp(y * (1.0 / scale), -448.0, 448.0).to(tl.float8e4nv)
    tl.store(
        y_q_ptr + row * width + cols,
        q.to(tl.uint8, bitcast=True),
        mask=mask,
    )


def build(device_index: int, gate_up_columns: int) -> None:
    """Compile the exact shared-expert tail before CUDA-graph capture."""
    if gate_up_columns <= 0 or gate_up_columns % 2:
        raise ValueError(
            f"shared gate/up columns must be positive and even, got {gate_up_columns}"
        )
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("compiling shared act+quant inside CUDA-graph capture")
    with torch.cuda.device(device_index):
        gate_up = torch.empty((1, gate_up_columns), dtype=torch.bfloat16, device="cuda")
        scale = torch.ones(1, dtype=torch.float32, device="cuda")
        silu_and_mul_static_quant_fp8(gate_up, scale)
        torch.cuda.synchronize(device_index)


def silu_and_mul_static_quant_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Return the shared expert activation as static per-tensor E4M3 rows."""
    if (
        not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.dim() != 2
        or not x.is_contiguous()
        or x.shape[1] <= 0
        or x.shape[1] % 2
    ):
        raise ValueError(
            "shared act+quant input must be contiguous CUDA BF16 [rows, 2 * width]"
        )
    if (
        scale.device != x.device
        or scale.dtype != torch.float32
        or tuple(scale.shape) != (1,)
        or not scale.is_contiguous()
    ):
        raise ValueError(
            "shared act+quant scale must be contiguous CUDA FP32 [1] "
            "on the input device"
        )

    rows, gate_up_columns = x.shape
    width = gate_up_columns // 2
    out = torch.empty((rows, width), device=x.device, dtype=torch.float8_e4m3fn)
    block = triton.next_power_of_2(width)
    _silu_and_mul_static_quant_fp8[(rows,)](
        x,
        out.view(torch.uint8),
        scale,
        width,
        BLOCK=block,
        num_warps=min(max(block // 256, 1), 8),
        num_stages=1,
        launch_pdl=True,
    )
    return out
