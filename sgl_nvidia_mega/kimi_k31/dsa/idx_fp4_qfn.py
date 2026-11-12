# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MXFP4 QFN index query as one CUDA kernel (idx_fp4_qfn.cuh: one CTA per row, PDL), . idx_fp4_q.qfn_quantize_query
dispatches here under K31MK_DSA_IDX_FP4_QFN=cuda (the default); the Triton kernel (idx_fp4_q._qfn_q4_kernel) stays the reference and the
fallback. quantize(): the same kernel without the rstd step for the stock chain (idx_fp4_q.quantize_query_served, K31MK_DSA_IDX_FP4_QUANT=cuda),
bit-identical to idx_fp4.quantize_query (the Triton reference). QFN form: the quantizer bytes equal idx_fp4.quantize_query of the
kernel's own BF16 query, and that query is within one BF16 ulp of the FP64 value (its rstd sums in another FP32 order than Triton's).
Tests: test_k31_idx_fp4_qfn_cuda.py."""

from functools import cache
from pathlib import Path

# SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {"kernels/jit/utils/compile/loader.py": ("load_jit",), "kernels/jit/utils/arch.py": ("is_arch_support_pdl",)}


@cache
def _load():
    import sglang
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "k31_idx_fp4_qfn",
        cuda_files=[str(Path(__file__).with_suffix(".cuh"))],
        cuda_wrappers=[("run", "k31_idx_fp4_qfn::Binding::run")],
        extra_include_paths=[str(Path(sglang.__file__).parent / "kernels/jit/csrc")],
    )


@cache
def _pdl() -> bool:
    from sglang.kernels.jit.utils import is_arch_support_pdl

    return bool(is_arch_support_pdl())


def build() -> None:
    """JIT-build the module (prepare time, never inside a graph capture)."""
    _load()


def qfn(acc, projected, gate, q4, sf, w, eps: float, debug_query=None) -> None:
    """acc BF16 [rows, 32, 128] contiguous, projected BF16 [rows, >= 1536] (unit column stride), gate BF16 [rows, 32] (unit column
    stride) -> q4 int8 [rows, 32, 64], sf int32 [rows, 32], w FP32 [rows, 32(, 1)] (contiguous). ``debug_query`` (tests): BF16
    [rows, 32, 128] that receives the query the kernel quantized."""
    debug = debug_query is not None
    _load().run(acc, projected, gate, q4, sf, w, debug_query if debug else acc, float(eps), _pdl(), debug, True)


def quantize(query, gate, q4, sf, w) -> None:
    """The stock chain's MXFP4 query (idx_fp4.quantize_query's math, bit-identical): query BF16 [rows, 32, 128] contiguous, gate BF16
    [rows, 32] (unit column stride) -> q4 int8 [rows, 32, 64], sf int32 [rows, 32], w FP32 [rows, 32(, 1)]."""
    _load().run(query, query, gate, q4, sf, w, query, 0.0, _pdl(), False, False)
