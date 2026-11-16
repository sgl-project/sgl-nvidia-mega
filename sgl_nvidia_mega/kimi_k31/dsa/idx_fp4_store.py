# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MXFP4 index-K store as a CUDA kernel in the FP8 fused_store_index_k_cache's launch form (idx_fp4_store.cuh: one warp per row, PDL), lane
prec-l2mk. idx_fp4.store_index_k dispatches here for BF16 contiguous keys and int64 slots (K31MK_DSA_IDX_FP4_STORE=cuda, the default); the
Triton writer (idx_fp4._store_kernel) stays the reference and the fallback. Bytes are equal (test_k31_idx_fp4_store.py)."""

from functools import cache
from pathlib import Path

# SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {"kernels/jit/utils/compile/loader.py": ("load_jit",), "kernels/jit/utils/arch.py": ("is_arch_support_pdl",)}


@cache
def _load():
    import sglang
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "k31_idx_fp4_store",
        cuda_files=[str(Path(__file__).with_suffix(".cuh"))],
        cuda_wrappers=[("run", "k31_idx_fp4_store::Binding::run")],
        extra_include_paths=[str(Path(sglang.__file__).parent / "kernels/jit/csrc")],
    )


@cache
def _pdl() -> bool:
    from sglang.kernels.jit.utils import is_arch_support_pdl

    return bool(is_arch_support_pdl())


def store(key, cache_pages, slots) -> None:
    """key BF16 [rows, 128] contiguous, cache_pages uint8 [pages, 64 * 68], slots int64 [>= rows] contiguous (negative = skip)."""
    _load().run(key, cache_pages, slots, _pdl())
