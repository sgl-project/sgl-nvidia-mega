# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Row-wise exact top-16 (sorted) of a 2-D bf16 tensor: one CTA per row, a lower bound from a sample of the row, then one streaming
filter pass and an exact select among the values that pass, JIT CUDA.

See ``csrc/draft_topk_sb.cuh`` for the order (the composite key of ``csrc/draft_topk.cuh``: value descending, then ascending
column), the bound and the overflow path. No per-device state (no counters, no scratch).
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("load_jit",),
}


_CSRC = Path(__file__).resolve().parent.parent / "csrc" / "draft_topk_sb.cuh"

K = 16
MAX_COLS = 65528


@functools.lru_cache(maxsize=None)
def module():
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "muse_vr_draft_topk_sb",
        cuda_files=[str(_CSRC)],
        cuda_wrappers=[
            ("run", "muse_vr::draft_topk_sb::DraftTopKSb::run"),
            ("run_flags", "muse_vr::draft_topk_sb::DraftTopKSb::run_flags"),
        ],
    )


def loaded() -> bool:
    """Whether the kernel was built and loaded in this process (the first eager call does it, before graph capture)."""
    return module.cache_info().currsize > 0


def supported(x: torch.Tensor, k: int) -> bool:
    return (
        x.is_cuda
        and x.dim() == 2
        and x.is_contiguous()
        and x.dtype == torch.bfloat16
        and k == K
        and K <= x.shape[1] <= MAX_COLS
        and x.shape[1] % 8 == 0
        and x.data_ptr() % 16 == 0
    )


def topk(x: torch.Tensor, k: int = K):
    """(values [rows, 16] bf16, int64 columns [rows, 16]), sorted by value descending, ties by ascending column."""
    rows = x.shape[0]
    vals = torch.empty((rows, K), dtype=x.dtype, device=x.device)
    ids = torch.empty((rows, K), dtype=torch.int64, device=x.device)
    module().run(x, vals, ids)
    return vals, ids


CAP = 2048  # candidate buffer entries (csrc kCap)


def topk_with_flags(x: torch.Tensor, k: int = K):
    """As ``topk``, plus int32 counts [rows]: the values that passed the filter; above ``CAP`` the row took the overflow path."""
    rows = x.shape[0]
    vals = torch.empty((rows, K), dtype=x.dtype, device=x.device)
    ids = torch.empty((rows, K), dtype=torch.int64, device=x.device)
    flags = torch.empty((rows,), dtype=torch.int32, device=x.device)
    module().run_flags(x, vals, ids, flags)
    return vals, ids, flags
