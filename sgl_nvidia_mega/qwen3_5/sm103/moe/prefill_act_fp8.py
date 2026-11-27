# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/kernels/ops/activation/activation.py (activation_module,
# _run_activation_inplace).

from __future__ import annotations

import functools
from pathlib import Path

import torch

SGLANG_READS = {
    "kernels/jit/utils/__init__.py": (
        "is_arch_support_pdl",
        "load_jit",
        "make_cpp_args",
    ),
}

SGLANG_MIRRORS = {
    "kernels/ops/activation/activation.py:activation_module": "bb41283e56cb3972",
    "kernels/ops/activation/activation.py:_run_activation_inplace": "3e89b11c44fc9aa8",
}
SGLANG_CUDA_MIRRORS = {
    "kernels/jit/csrc/elementwise/activation.cuh:act_and_mul_kernel": "99dc818ded47607c",
}
FAST_MATH = True
VEC_ELEMENTS = 16


@functools.cache
def _module(dtype: torch.dtype):
    import sglang
    from sglang.kernels.jit.utils import is_arch_support_pdl, load_jit, make_cpp_args

    args = make_cpp_args(dtype, is_arch_support_pdl(), FAST_MATH)
    return load_jit(
        "q35_prefill_act_fp8",
        *args,
        cuda_files=[str(Path(__file__).with_suffix(".cuh"))],
        extra_include_paths=[str(Path(sglang.__file__).parent / "kernels/jit/csrc")],
        cuda_wrappers=[("run", f"PrefillActFp8Kernel<{args}>::run")],
    )


def build(device_index: int) -> None:
    """Compile the kernel and run it once on a small tensor, outside any capture."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("compiling prefill_act_fp8 inside CUDA-graph capture")
    with torch.cuda.device(device_index):
        gate_up = torch.zeros(1, 2 * VEC_ELEMENTS, dtype=torch.bfloat16, device="cuda")
        silu_and_mul_static_fp8(gate_up, torch.ones(1, device="cuda"))
        torch.cuda.synchronize(device_index)


def supported(gate_up: torch.Tensor) -> bool:
    return (
        gate_up.is_cuda
        and gate_up.dtype == torch.bfloat16
        and gate_up.dim() == 2
        and gate_up.is_contiguous()
        and gate_up.shape[1] % (2 * VEC_ELEMENTS) == 0
    )


def silu_and_mul_static_fp8(gate_up: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if not supported(gate_up):
        raise ValueError("silu_and_mul_static_fp8: unsupported layout")
    if scale.numel() != 1 or scale.dtype != torch.float32:
        raise ValueError("silu_and_mul_static_fp8: a 1-element float32 static scale")
    width = gate_up.shape[1] // 2
    out_q = torch.empty(
        (gate_up.shape[0], width), dtype=torch.float8_e4m3fn, device=gate_up.device
    )
    _module(gate_up.dtype).run(gate_up, out_q.view(torch.uint8), scale.reshape(1))
    return out_q
