# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""FP8 QFN index query as one CUDA kernel (idx_qfn8.cuh: one CTA per row, PDL), . index_qfn.qfn_quant_head_gate dispatches
here under K31MK_DSA_QFN_CUDA=1 (default 0 = the Triton kernel, which stays the reference and the fallback). Outputs are bitwise those of
index_qfn._qfn_quant_head_gate_kernel (test_k31_idx_qfn8_cuda.py); SA-neutral (the same FP8 query bytes and head weights).
Build: JIT at the first EAGER call (load_jit); a call inside a CUDA-graph capture before that keeps the Triton kernel for that capture and
logs 'K31MK_DSA_QFN_CUDA DECLINED' once (never compiles inside a capture). Engagement: 'K31MK_DSA_QFN_CUDA ENGAGED rows=N' at the first
call per row count; STATS counts calls."""

import logging
from functools import cache
from pathlib import Path

import torch

# SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {"kernels/jit/utils/compile/loader.py": ("load_jit",), "kernels/jit/utils/arch.py": ("is_arch_support_pdl",)}

logger = logging.getLogger(__name__)
STATS = {"calls": 0, "declined": 0}
_LOGGED = set()
_STATE = {"built": False}


@cache
def _load(variants: bool = False):
    import sglang
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "k31_idx_qfn8_variants" if variants else "k31_idx_qfn8",
        cuda_files=[str(Path(__file__).with_suffix(".cuh"))],
        cuda_wrappers=[("run", "k31_idx_qfn8::Binding::run")],
        extra_include_paths=[str(Path(sglang.__file__).parent / "kernels/jit/csrc")],
        extra_cuda_cflags=["-DK31_QFN8_VARIANTS=1"] if variants else None,
    )


@cache
def _pdl() -> bool:
    from sglang.kernels.jit.utils import is_arch_support_pdl

    return bool(is_arch_support_pdl())


def build() -> None:
    """JIT-build the module (eager only, never inside a graph capture)."""
    _load()
    _pdl()
    _STATE["built"] = True


def qfn8(acc, projected, gate, q8, weights, eps: float, round_scale: bool) -> bool:
    """The served FP8 QFN query into ``q8`` (FP8 e4m3 [rows, 32, 128] contiguous) and ``weights`` (FP32 [rows, 32, 1] contiguous) from
    acc BF16 [rows, 32, 128] contiguous, projected BF16 [rows, >= 1536] (unit column stride), gate BF16 [rows, 32] (unit column stride).
    Returns False (outputs untouched) only when the module is not built and the current stream is capturing."""
    if not _STATE["built"]:
        if torch.cuda.is_current_stream_capturing():
            STATS["declined"] += 1
            if "declined" not in _LOGGED:
                _LOGGED.add("declined")
                logger.warning("K31MK_DSA_QFN_CUDA DECLINED rows=%d: module not built before this graph capture; the Triton kernel serves "
                               "this capture (bitwise-identical outputs)", acc.shape[0])
            return False
        build()
    _load().run(acc, projected, gate, q8.view(torch.uint8), weights, acc, acc, float(eps), _pdl(), False, bool(round_scale), 0, 0, 0, 0)
    STATS["calls"] += 1
    rows = int(acc.shape[0])
    if rows not in _LOGGED:
        _LOGGED.add(rows)
        logger.info("K31MK_DSA_QFN_CUDA ENGAGED rows=%d (FP8 QFN index query by idx_qfn8.cuh, bitwise == the Triton kernel)", rows)
    return True


def qfn8_test(acc, projected, gate, q8, weights, eps: float, round_scale: bool, debug_query=None, debug_rstd=None, variant=(0, 0, 0, 0),
              pdl=None) -> None:
    """Tests: the same kernel with optional debug outputs (BF16 query [rows, 32, 128], FP32 rstd [rows]) and, for variant != (0, 0, 0, 0),
    the K31_QFN8_VARIANTS build (inner summation order, cross-warp order, rstd tail, scale rounding)."""
    debug = debug_query is not None
    mod = _load(True) if tuple(variant) != (0, 0, 0, 0) else _load()
    mod.run(acc, projected, gate, q8.view(torch.uint8), weights, debug_query if debug else acc, debug_rstd if debug else acc, float(eps),
            _pdl() if pdl is None else bool(pdl), debug, bool(round_scale), int(variant[0]), int(variant[1]), int(variant[2]), int(variant[3]))
