# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Sequence-parallel MoE finalize (JIT CUDA): unpermute + post-expert RMSNorm + router-weighted top-k sum, bitwise to
SGLang's single-CTA CuTe-DSL finalize kernel (cluster size 1, hidden 4096, top-8), on a persistent schedule that keeps
the GEMM2 rows in flight through a shared-memory ring of bulk copies. See ``csrc/pfin.cuh`` and ``tests/test_pfin.py``.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("load_jit", "make_cpp_args"),
}


_CSRC = Path(__file__).resolve().parent.parent / "csrc" / "pfin.cuh"
HIDDEN = 4096
TOPK = 8
# (row-slot ring depth, consumer groups per CTA) built; tunable together with the CTAs-per-SM setting
CONFIGS = ((8, 1), (12, 1), (16, 2), (16, 3), (24, 3), (24, 4), (24, 5))


# test-only builds (tests/test_pfin.py): randomized hand-off delays; a slow consumer (clock64 spin before each release)
_TEST_BUILDS = {
    "": None,
    "fuzz": ["-DMUSE_VR_PFIN_FUZZ=1"],
    "slow": ["-DMUSE_VR_PFIN_SLOW_CONSUMER=20000"],
}


@functools.lru_cache(maxsize=None)
def _module(
    slots: int, groups: int, weight_first: bool, bf16_weights: bool, build: str = ""
):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(slots, groups, weight_first, bf16_weights)
    return load_jit(
        "muse_vr_pfin" + (f"_{build}" if build else ""),
        *args,
        cuda_files=[str(_CSRC)],
        cuda_wrappers=[("pfin", f"muse_vr::PFin<{args}>::run")],
        extra_cuda_cflags=_TEST_BUILDS[build],
    )


@functools.lru_cache(maxsize=None)
def _sm_count(index: int) -> int:
    return int(torch.cuda.get_device_properties(index).multi_processor_count)


def grid_for(device: torch.device, ctas_per_sm: int) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return _sm_count(idx) * int(ctas_per_sm)


def pfin(
    gemm2: torch.Tensor,
    expert_weights: torch.Tensor,
    idx: torch.Tensor,
    rms_w: torch.Tensor,
    out: torch.Tensor,
    eps: float,
    weight_first: bool,
    slots: int = 24,
    groups: int = 4,
    ctas_per_sm: int = 1,
    _build: str = "",
) -> torch.Tensor:
    """out[t] = bf16(sum over local experts e of token t, ascending slot: w[t, e] * rmsnorm(gemm2[idx[t, e]]) ...), with
    the stock kernel's arithmetic (module doc of ``csrc/pfin.cuh``).

    :param gemm2: ``[rows, >= 4096]`` bf16, contiguous (rows addressed by the tensor width, as the stock op does).
    :param expert_weights: ``[T, 8]`` or ``[T * 8]`` fp32 or bf16, contiguous.
    :param idx: ``[T * 8]`` int32, contiguous; -1 = expert not on this rank.
    :param rms_w: ``[4096]`` bf16 (effective weight, 1.0 pre-added). :param out: ``[T, 4096]`` bf16, contiguous.
    """
    if (slots, groups) not in CONFIGS:
        raise ValueError(f"(slots, groups) must be one of {CONFIGS}")
    bf16_w = expert_weights.dtype == torch.bfloat16
    _module(int(slots), int(groups), bool(weight_first), bool(bf16_w), _build).pfin(
        gemm2,
        expert_weights,
        idx,
        rms_w,
        out,
        float(eps),
        grid_for(out.device, ctas_per_sm),
    )
    return out
