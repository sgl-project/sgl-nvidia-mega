# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Row-wise exact top-k (sorted, at most 32) of a 2-D bf16 / fp16 / fp32 tensor in one launch, JIT CUDA.

See ``csrc/draft_topk.cuh`` for the order (value descending, then ascending column) and the work split.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("load_jit", "make_cpp_args"),
}


_CSRC = Path(__file__).resolve().parent.parent / "csrc" / "draft_topk.cuh"

KS = (8, 16, 32)  # compiled list lengths; a call uses the smallest one >= k
MAX_COLS = 65536
MAX_ROWS = 4096  # rows of the per-device counter buffer
DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_counters: dict = {}


@functools.lru_cache(maxsize=None)
def module(kk: int):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(kk)
    return load_jit(
        "muse_vr_draft_topk",
        *args,
        cuda_files=[str(_CSRC)],
        cuda_wrappers=[("run", f"muse_vr::draft_topk::DraftTopK<{args}>::run")],
    )


def splits_for(cols: int) -> int:
    """CTAs per row: slices of at most 2048 columns."""
    return max(1, min(32, -(-cols // 2048)))


def counters(device: torch.device, create: bool = True):
    """The per-device row counters (zero between launches); created on first use outside graph capture."""
    key = device.index if device.index is not None else torch.cuda.current_device()
    buf = _counters.get(key)
    if buf is None and create:
        buf = torch.zeros(MAX_ROWS, dtype=torch.int32, device=device)
        _counters[key] = buf
    return buf


def supported(x: torch.Tensor, k: int) -> bool:
    return (
        x.is_cuda
        and x.dim() == 2
        and x.is_contiguous()
        and x.dtype in DTYPES
        and 1 <= k <= KS[-1]
        and k <= x.shape[1] <= MAX_COLS
        and x.shape[0] <= MAX_ROWS
    )


def topk(x: torch.Tensor, k: int):
    """(values [rows, k] in x's dtype, int64 columns [rows, k]), sorted by value descending, ties by ascending column."""
    rows, cols = x.shape
    kk = next(v for v in KS if v >= k)
    s = splits_for(cols)
    vals = torch.empty((rows, k), dtype=x.dtype, device=x.device)
    ids = torch.empty((rows, k), dtype=torch.int64, device=x.device)
    scratch = torch.empty((max(1, rows * s * kk),), dtype=torch.int64, device=x.device)
    module(kk).run(x, vals, ids, scratch, counters(x.device), int(k), int(s))
    return vals, ids
