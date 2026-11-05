# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Sequence-parallel MoE front (JIT CUDA): the pre-expert RMSNorm of [T, 4096] bf16 rows and the per-token NVFP4
quantization of the normalized rows in one kernel, bitwise to the stock pair (torch RMSNorm -> bf16 -> FlashInfer
per-token NVFP4, linear scale layout, fast-math scale path). See ``csrc/pfront.cuh`` and ``tests/test_pfront.py``.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("load_jit",),
}


_CSRC = Path(__file__).resolve().parent.parent / "csrc" / "pfront.cuh"
HIDDEN = 4096


@functools.lru_cache(maxsize=None)
def _module():
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "muse_vr_pfront",
        cuda_files=[str(_CSRC)],
        cuda_wrappers=[("pfront", "muse_vr::PFront::run")],
    )


def pfront(x: torch.Tensor, weight: torch.Tensor, eps: float, global_scale_inv: float):
    """(codes ``[T, 2048]`` uint8, block scales ``[T, 256]`` uint8 (e4m3 bits), per-token scales ``[T]`` fp32) of
    ``quantize_per_token(bf16(rmsnorm(x) * weight))``.

    :param x: ``[T, 4096]`` bf16, contiguous, 16-byte aligned. :param weight: ``[4096]`` bf16 (effective weight).
    :param global_scale_inv: the quantizer's global scale inverse (``1 / (e4m3 max * 6)``).
    """
    t = x.shape[0]
    codes = torch.empty(t, HIDDEN // 2, dtype=torch.uint8, device=x.device)
    sf = torch.empty(t, HIDDEN // 16, dtype=torch.uint8, device=x.device)
    pts = torch.empty(t, dtype=torch.float32, device=x.device)
    _module().pfront(x, weight, codes, sf, pts, float(eps), float(global_scale_inv))
    return codes, sf, pts
