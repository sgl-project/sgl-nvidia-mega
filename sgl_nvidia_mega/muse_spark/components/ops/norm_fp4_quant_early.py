# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""SGLang's fused pre-expert RMS norm + per-token NVFP4 quant (``norm_fp4_quant``), built from a copy whose
programmatic-launch trigger is issued at kernel entry (csrc/norm_fp4_quant_early.cuh; the arithmetic is SGLang's).

``fused_norm_fp4_quant_early`` has the signature and the return value of SGLang's ``fused_norm_fp4_quant`` for the
quantize-only form (``quantize=True, want_normed=False``): ``(fp4 [M, K/2] u8, sf [M, K/16] u8, per_token_scale [M]
fp32, None)``. A programmatic dependent of this kernel may launch while it runs, and must read its outputs only after
its own dependency wait.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": (
        "is_arch_support_pdl",
        "load_jit",
        "make_cpp_args",
    ),
}


_CSRC = Path(__file__).resolve().parents[1] / "csrc"
_CUH = _CSRC / "norm_fp4_quant_early.cuh"
_BLOCK = 16
_MAX_THREADS = 1024


def _num_threads(hidden_size: int) -> int:
    """SGLang's thread-count rule for this kernel (the launch shape, and so the reduction order, is SGLang's)."""
    blocks = hidden_size // _BLOCK
    if hidden_size % _BLOCK or blocks % 32:
        raise RuntimeError(
            f"norm_fp4_quant_early: hidden_size={hidden_size} must be a multiple of 512"
        )
    for threads in (256, 512, _MAX_THREADS, 128, 64, 32):
        if blocks % threads == 0:
            return threads
    raise RuntimeError(f"norm_fp4_quant_early: unsupported hidden_size={hidden_size}")


@functools.lru_cache(maxsize=None)
def _module(hidden_size: int, num_threads: int):
    from sglang.kernels.jit.utils import is_arch_support_pdl, load_jit, make_cpp_args

    args = make_cpp_args(hidden_size, num_threads, is_arch_support_pdl())
    return load_jit(
        "muse_vr_norm_fp4_quant_early",
        *args,
        cuda_files=[str(_CUH)],
        cuda_wrappers=[
            ("run_quant", f"MuseVrNormFp4QuantEarly<{args}, true, false>::run")
        ],
        extra_include_paths=[str(_CSRC)],
    )


def preload(hidden_size: int) -> None:
    _module(hidden_size, _num_threads(hidden_size))


def fused_norm_fp4_quant_early(
    x: torch.Tensor, weight, eps: float, global_scale_inv: float
):
    assert x.dtype == torch.bfloat16 and x.dim() == 2 and x.is_contiguous()
    m, hidden_size = x.shape
    if weight is not None:
        assert weight.dtype == torch.bfloat16 and weight.shape == (hidden_size,)
        weight = weight.contiguous()
    fp4 = torch.empty(m, hidden_size // 2, dtype=torch.uint8, device=x.device)
    sf = torch.empty(m, hidden_size // _BLOCK, dtype=torch.uint8, device=x.device)
    pts = torch.empty(m, dtype=torch.float32, device=x.device)
    if m == 0:
        return fp4, sf, pts, None
    _module(hidden_size, _num_threads(hidden_size)).run_quant(
        x, weight, None, fp4, sf, pts, float(eps), float(global_scale_inv)
    )
    return fp4, sf, pts, None
