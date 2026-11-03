# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The router GEMM and the NVFP4 quantization of its input in one launch, for the
trtllm MoE runner, which routes from the logits itself (``src/router_front.cu``).

The logits equal SGLang's cuBLAS router GEMM bit for bit, and the quantization equals
the runner's own NVFP4 input quantizer (16-element blocks, e4m3 scales, linear layout).
``router_gemm_quant_shared`` also computes the shared expert's static-FP8 gate/up
projection of up to 8 rows in the same launch (SGLANG_Q35MF_ROUTE_SHARED on SGLang branch
nvidia/qwen35-agentx-gb300); it sums in another order than the FP8 linear's GEMM
(numeric).
"""

from __future__ import annotations

import ctypes
from functools import cache
from pathlib import Path

import torch

from sgl_nvidia_mega.common import jit

SOURCE = Path(__file__).parent / "src" / "router_front.cu"
HIDDEN, EXPERTS, MAX_TOKENS = 4096, 512, 16
THREADS, CTAS = 256, 64  # 8 experts a CTA
_ROW_BYTES = HIDDEN * 2 + 16
_PART_BYTES = 2 * 16 * 8 * 4
_SH_ROW_BYTES = (
    HIDDEN + 16
)  # an e4m3 row of the shared stage, padded like the bf16 ones
_SH_PART_BYTES = 4 * 16 * 8 * 4  # fp32 [kShSplits 4][16][8]
SHARED_MAX_M, SHARED_MAX_N = 8, CTAS * 8  # 8 shared rows a CTA
MODE_QUANT = 1


def max_m_bucket(m: int) -> int:
    """The compiled row capacity that serves m rows."""
    return 1 if m <= 1 else 8 if m <= 8 else 16


def shared_bytes(max_m: int, shared: bool = False) -> int:
    """GemvSmem<max_m, shared>::kBytes: 8 weight rows, max_m input rows, the fp32 partials
    and two mbarriers; with the shared stage also 8 FP8 weight rows, max_m FP8 input rows,
    its partials and a third mbarrier."""
    plain = 8 * _ROW_BYTES + max_m * _ROW_BYTES + _PART_BYTES
    if not shared:
        return plain + 16
    return plain + (8 + max_m) * _SH_ROW_BYTES + _SH_PART_BYTES + 24


class Params(ctypes.Structure):
    """qwen_router_front::Params."""

    _fields_ = [
        (name, ctypes.c_void_p)
        for name in (
            "x",
            "weight",
            "global_scale",
            "logits",
            "topk_weights",
            "topk_ids",
            "x_fp4",
            "x_sf",
            "tile_idx_to_expert_idx",
            "tile_idx_to_mn_limit",
            "expanded_idx_to_permuted_idx",
            "permuted_idx_to_expanded_idx",
            "total_num_padded_tokens",
            "num_non_exiting_tiles",
            "arrival",
            "trace",
        )
    ] + [
        ("num_tokens", ctypes.c_uint32),
        ("tile_log2", ctypes.c_uint32),
        ("mode", ctypes.c_uint32),
        ("sh_weight", ctypes.c_void_p),
        ("sh_in_scale", ctypes.c_void_p),
        ("sh_w_scale", ctypes.c_void_p),
        ("sh_out", ctypes.c_void_p),
        ("sh_n", ctypes.c_uint32),
    ]


@cache
def _cubin(device_index: int, max_m: int) -> bytes:
    major, minor = torch.cuda.get_device_capability(device_index)
    return jit.compile_source(
        SOURCE.read_text(),
        f"q35_router_front_m{max_m}",
        f"sm_{major}{minor}a",
        ("-std=c++17", "-default-device", f"-DQ35_ROUTER_MAX_M={max_m}"),
    )


@cache
def _kernel(device_index: int, max_m: int) -> jit.Kernel:
    return jit.load(
        _cubin(device_index, max_m),
        "q35_router_gemm_quant",
        (THREADS, 1, 1),
        shared_bytes(max_m),
    )


@cache
def _shared_kernel(device_index: int, max_m: int) -> jit.Kernel:
    return jit.load(
        _cubin(device_index, max_m),
        "q35_router_gemm_quant_shared",
        (THREADS, 1, 1),
        shared_bytes(max_m, shared=True),
    )


def build(device_index: int, max_rows: int, shared: bool = False) -> None:
    """Compile every row capacity up to max_rows, with ``shared`` the shared gate/up
    ones too (no compile inside a step)."""
    for m in sorted({max_m_bucket(r) for r in range(1, max_rows + 1)}):
        _kernel(device_index, m)
        if shared and m <= SHARED_MAX_M:
            _shared_kernel(device_index, m)


def router_gemm_quant(
    x: torch.Tensor, weight: torch.Tensor, global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x [M, 4096] bf16 (M <= 16), weight [512, 4096] bf16, global_scale [1] fp32:
    (logits bf16 [M, 512], x_fp4 uint8 [M, 2048], x_sf uint8 [M, 256])."""
    m = x.shape[0]
    # the kernel reads x as packed rows (bulk copies) and serves at most 16
    if x.dtype != torch.bfloat16 or x.shape[1:] != (HIDDEN,) or not x.is_contiguous():
        raise ValueError(
            f"router front: x {x.dtype} {tuple(x.stride())} {tuple(x.shape)}"
        )
    if not 1 <= m <= MAX_TOKENS:
        raise ValueError(f"router front: {m} rows (1-{MAX_TOKENS})")
    logits = torch.empty((m, EXPERTS), dtype=torch.bfloat16, device=x.device)
    x_fp4 = torch.empty((m, HIDDEN // 2), dtype=torch.uint8, device=x.device)
    x_sf = torch.empty((m, HIDDEN // 16), dtype=torch.uint8, device=x.device)
    p = Params(
        x=x.data_ptr(),
        weight=weight.data_ptr(),
        global_scale=global_scale.data_ptr(),
        logits=logits.data_ptr(),
        x_fp4=x_fp4.data_ptr(),
        x_sf=x_sf.data_ptr(),
        num_tokens=m,
        mode=MODE_QUANT,
    )
    kernel = _kernel(x.device.index, max_m_bucket(m))
    jit.launch(kernel, (CTAS, 1, 1), [p], torch.cuda.current_stream().cuda_stream)
    return logits, x_fp4, x_sf


def router_gemm_quant_shared(
    x: torch.Tensor,
    weight: torch.Tensor,
    global_scale: torch.Tensor,
    sh_weight: torch.Tensor,
    sh_in_scale: torch.Tensor,
    sh_w_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``router_gemm_quant`` of x [M, 4096] (M <= 8) plus the shared expert's gate/up
    projection in the same launch: sh_weight [N, 4096] e4m3 row-major (N % 8 == 0,
    N <= 512), static scales sh_in_scale / sh_w_scale [1] fp32. Returns (logits, x_fp4,
    x_sf, gate_up bf16 [M, N]); gate_up = (q(x) @ sh_weight.T) * sh_in_scale * sh_w_scale
    with q the static e4m3 quantization of x, summed in fp32."""
    m, n = x.shape[0], sh_weight.shape[0]
    if x.dtype != torch.bfloat16 or x.shape[1:] != (HIDDEN,) or not x.is_contiguous():
        raise ValueError(
            f"router front: x {x.dtype} {tuple(x.stride())} {tuple(x.shape)}"
        )
    if not 1 <= m <= SHARED_MAX_M:
        raise ValueError(f"router front with the shared stage: {m} rows (1-8)")
    if (
        sh_weight.shape[1:] != (HIDDEN,)
        or n % 8
        or n > SHARED_MAX_N
        or not sh_weight.is_contiguous()
    ):
        raise ValueError(f"router front: shared weight {tuple(sh_weight.shape)}")
    logits = torch.empty((m, EXPERTS), dtype=torch.bfloat16, device=x.device)
    x_fp4 = torch.empty((m, HIDDEN // 2), dtype=torch.uint8, device=x.device)
    x_sf = torch.empty((m, HIDDEN // 16), dtype=torch.uint8, device=x.device)
    gate_up = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    p = Params(
        x=x.data_ptr(),
        weight=weight.data_ptr(),
        global_scale=global_scale.data_ptr(),
        logits=logits.data_ptr(),
        x_fp4=x_fp4.data_ptr(),
        x_sf=x_sf.data_ptr(),
        num_tokens=m,
        mode=MODE_QUANT,
        sh_weight=sh_weight.data_ptr(),
        sh_in_scale=sh_in_scale.data_ptr(),
        sh_w_scale=sh_w_scale.data_ptr(),
        sh_out=gate_up.data_ptr(),
        sh_n=n,
    )
    kernel = _shared_kernel(x.device.index, max_m_bucket(m))
    jit.launch(kernel, (CTAS, 1, 1), [p], torch.cuda.current_stream().cuda_stream)
    return logits, x_fp4, x_sf, gate_up
