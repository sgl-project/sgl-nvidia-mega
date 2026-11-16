# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 DSA index-K LayerNorm + store as one warp-per-row kernel for the QFP profiles (index_kst.cuh; K31MK_DSA_QFP_KST):
replaces QFP's FlashInfer LayerNorm launch + index-K store launch with the query front's store row math (index_q.cuh store_index_k)."""

from functools import cache
from pathlib import Path

# SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {"kernels/jit/utils/compile/loader.py": ("load_jit",), "kernels/jit/utils/arch.py": ("is_arch_support_pdl",)}


@cache
def _load():
    import sglang
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "k31_index_kst",
        cuda_files=[str(Path(__file__).with_suffix(".cuh"))],
        cuda_wrappers=[("run", "k31_index_kst::Binding::run")],
        extra_include_paths=[str(Path(sglang.__file__).parent / "kernels/jit/csrc")],
    )


@cache
def _pdl() -> bool:
    from sglang.kernels.jit.utils import is_arch_support_pdl

    return bool(is_arch_support_pdl())


def store(projected, k_col, slots, cache_pages, key_norm) -> None:
    """``projected`` BF16 [rows, pitch] (index-K columns [k_col, k_col + 128)), ``slots`` int64 [>= rows] contiguous (negative = skip),
    ``cache_pages`` uint8 FP8 [pages, 64 * 132] or MXFP4 [pages, 64 * 68], ``key_norm`` = (FP32 weight [128], FP32 bias [128], eps)."""
    gamma, beta, eps = key_norm
    _load().run(projected, int(k_col), slots, cache_pages, gamma, beta, float(eps), _pdl())
