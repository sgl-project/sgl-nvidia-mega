# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MXFP8 quantizer with several loads in flight per thread (helper module for ``mxfp8_ops``; no component of its own).

``csrc/mxfp8_fq.cuh`` reproduces, operation by operation, the quantizer ``mxfp8_ops.quantize`` otherwise calls
(FlashInfer's ``mxfp8_quantize`` with swizzled scales): same block maximum, same E8M0 scale rule (round toward +inf),
same reciprocal and E4M3 conversion, same 128 x 4 swizzled scale layout with zero scales on the padding rows. The bytes
are identical (tests/test_mxfp8_fq.py); only the memory access differs: each thread issues several independent 32-byte
loads before converting, and ``quantize_group`` quantizes up to four tensors (e.g. one layer's weights) in one launch.

Used when ``MUSE_VR_FP8_PREFILL_FQ`` is set (see ``fp8_prefill``). Operands it does not take (K not a multiple of 128 or above 8192,
non-contiguous, not BF16) go to the stock quantizer.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("load_jit", "make_cpp_args"),
}


_CUH = Path(__file__).resolve().parent / "csrc" / "mxfp8_fq.cuh"
FP8 = torch.float8_e4m3fn
MAX_GROUP = 4
DEFAULT_UNROLL = 1  # rows per pipeline stage (two stages in registers)
DEFAULT_CTAS_PER_SM = 4


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "")
    return int(v) if v.strip() else default


@functools.lru_cache(maxsize=None)
def _module(unroll: int, use_pdl: bool, ftz: bool, lean: bool):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(int(unroll), bool(use_pdl), bool(ftz), bool(lean))
    return load_jit(
        "muse_vr_mxfp8_fq",
        *args,
        cuda_files=[str(_CUH)],
        cuda_wrappers=[("quantize", f"muse_vr::fq::Mxfp8FqKernel<{args}>::run")],
    )


@functools.lru_cache(maxsize=None)
def _sms(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


@functools.lru_cache(maxsize=None)
def _pdl(index: int) -> bool:
    return torch.cuda.get_device_capability(index)[0] >= 9


@functools.lru_cache(maxsize=None)
def config():
    """(unroll, CTAs per SM, flush-to-zero multiplies, lean form) from the environment (read once per process)."""
    return (
        _env_int("MUSE_VR_FQ_UNROLL", DEFAULT_UNROLL),
        _env_int("MUSE_VR_FQ_CTAS_PER_SM", DEFAULT_CTAS_PER_SM),
        _env_int("MUSE_VR_FQ_FTZ", 1) != 0,
        _env_int("MUSE_VR_FQ_LEAN", 1) != 0,
    )


MAX_K = 8192


def takes(x: torch.Tensor) -> bool:
    """True if the kernel takes this operand (BF16, 2-D contiguous, K a multiple of 128 up to 8192, 16-byte aligned)."""
    return (
        x.dtype == torch.bfloat16
        and x.dim() == 2
        and x.is_contiguous()
        and x.shape[1] % 128 == 0
        and 0 < x.shape[1] <= MAX_K
        and x.data_ptr() % 16 == 0
        and x.is_cuda
    )


def sf_numel(rows: int, k: int) -> int:
    return -(-rows // 128) * 128 * (k // 32)


def alloc(x: torch.Tensor):
    rows, k = x.shape
    return (
        torch.empty((rows, k), dtype=FP8, device=x.device),
        torch.empty((sf_numel(rows, k),), dtype=torch.uint8, device=x.device),
    )


def quantize_group_into(
    xs,
    outs,
    unroll: int | None = None,
    ctas_per_sm: int | None = None,
    ftz: bool | None = None,
    lean: bool | None = None,
):
    """Quantize up to four BF16 [rows, K] tensors in one launch into caller-owned (q [rows, K] E4M3, s 1-D uint8)
    buffers. Every tensor must pass ``takes``."""
    n = len(xs)
    if not 1 <= n <= MAX_GROUP or len(outs) != n:
        raise ValueError("mxfp8_fq: 1 to 4 tensors, one (q, s) pair each")
    du, dc, df, dl = config()
    u = du if unroll is None else unroll
    c = dc if ctas_per_sm is None else ctas_per_sm
    f = df if ftz is None else ftz
    lf = dl if lean is None else lean
    dev = xs[0].device
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    trip = []
    for x, (q, s) in zip(xs, outs):
        trip += [x, q.view(torch.uint8), s]
    while len(trip) < 3 * MAX_GROUP:
        trip += trip[:3]
    _module(u, _pdl(idx), f, lf).quantize(*trip, n, _sms(idx) * c)


def quantize(x: torch.Tensor):
    """[rows, K] BF16 -> (E4M3 [rows, K], swizzled E8M0 scales 1-D uint8), bytes identical to the stock quantizer."""
    q, s = alloc(x)
    quantize_group_into([x], [(q, s)])
    return q, s


def quantize_group(xs):
    """Quantize up to four tensors in one launch; returns [(q, s)] in order."""
    outs = [alloc(x) for x in xs]
    quantize_group_into(xs, outs)
    return outs
