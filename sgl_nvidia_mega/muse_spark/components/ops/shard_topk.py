# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Row-wise top-16 of a bf16 matrix, sorted by value (descending), equal values by ascending column (JIT CUDA).

See ``csrc/shard_topk.cuh``.
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


_CSRC = Path(__file__).resolve().parent.parent / "csrc" / "shard_topk.cuh"

K = 16
CHUNK = 512
MAX_CHUNKS = 128


@functools.lru_cache(maxsize=None)
def use_pdl() -> bool:
    from sglang.kernels.jit.utils import is_arch_support_pdl

    return bool(is_arch_support_pdl())


@functools.lru_cache(maxsize=None)
def module(pdl: bool):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(pdl)
    return load_jit(
        "muse_vr_shard_topk",
        *args,
        cuda_files=[str(_CSRC)],
        cuda_wrappers=[("run", f"muse_vr::shard_topk::ShardTopK<{args}>::run")],
    )


def supported(x: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and x.stride(1) == 1
        and x.stride(0) % 8 == 0
        and x.data_ptr() % 16 == 0
        and K <= x.shape[1]
        and -(-x.shape[1] // CHUNK) <= MAX_CHUNKS
    )


def topk16(x: torch.Tensor, pdl: bool | None = None):
    """(values bf16 [rows, 16], columns int64 [rows, 16])."""
    rows, cols = x.shape
    chunks = -(-cols // CHUNK)
    scratch = torch.empty((rows, chunks * K), dtype=torch.int64, device=x.device)
    vals = torch.empty((rows, K), dtype=torch.bfloat16, device=x.device)
    ids = torch.empty((rows, K), dtype=torch.int64, device=x.device)
    module(use_pdl() if pdl is None else pdl).run(x, scratch, vals, ids)
    return vals, ids
