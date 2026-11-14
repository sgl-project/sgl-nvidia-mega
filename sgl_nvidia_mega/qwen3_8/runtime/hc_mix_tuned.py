# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team): python/sglang/kernels/ops/gemm/hc_mix_splitk.py
# (hc_mix), and from the source's copy of it (its _TUNED_TACTICS / _tuned_tactic, read with
# SGLANG_HC_MACRO, default on, at TP1). Changed: the tactic comes from the source's row-aware
# rule; the kernels, scratch and layout are SGLang's.
"""The HC mix pair (down GEMV with the SiLU epilogue, gated up GEMV) at the source's row-aware
split-K tactics: from 9 rows the down GEMV halves split-K (s16 -> s8, 8 stages), and the up GEMV
uses 64-wide MMA tiles with 4 stages and an N tile of 8 / 16 / 32 by rows. SGLang main keeps
s16 for the down GEMV at every row count and its default tactic for the up GEMV."""

from __future__ import annotations

import os

import torch

SGLANG_READS = {
    "kernels/ops/gemm/hc_mix_splitk.py": ("_MAX_ROWS", "_get_scratch", "_get_tactic"),
    "kernels/ops/gemm/dense_bf16_gemm_sm100_splitk_epilogue.py": (
        "SplitKTactic",
        "validate_tactic",
        "run_splitk_dense_silu",
        "run_splitk_dense_gate",
    ),
    "srt/runtime_context.py": ("get_parallel",),
}

ENV = "SGLANG_HC_MACRO"
# The source's split-K mix rows with the switch on at TP1 (SGLang main: 24)
MAX_ROWS = 32
_cache: dict = {}


def enabled() -> bool:
    """The source's switch (its default on) and TP1, as the source gates its tuned tactics."""
    from sglang.srt.runtime_context import get_parallel

    on = os.environ.get(ENV, "1").strip().lower() not in ("0", "false", "no", "off")
    return on and get_parallel().tp_size == 1


def tactic(m: int, n: int, k: int):
    """The source's _tuned_tactic, else SGLang's choice for the key."""
    key = (m, n, k)
    found = _cache.get(key)
    if found is None:
        import sglang.kernels.ops.gemm.hc_mix_splitk as hc_mix_splitk
        from sglang.kernels.ops.gemm.dense_bf16_gemm_sm100_splitk_epilogue import (
            SplitKTactic,
            validate_tactic,
        )

        if n <= 512 and k % (128 * 16) == 0:
            found = (
                SplitKTactic(mma_m=64, mma_n=8, split_k=16, ab_stages=6)
                if m <= 8
                else SplitKTactic(mma_m=64, mma_n=8, split_k=8, ab_stages=8)
            )
        else:
            mma_n = 8 if m <= 8 else 16 if m <= 16 else 32
            found = SplitKTactic(mma_m=64, mma_n=mma_n, split_k=1, ab_stages=4)
        try:
            validate_tactic(found, m, n, k)
        except ValueError:
            found = hc_mix_splitk._get_tactic(m, n, k)
        _cache[key] = found
    return found


def hc_mix(
    hyper_input_normed: torch.Tensor,
    w_down: torch.Tensor,
    w_up_permuted_padded: torch.Tensor,
    hc_count: int,
    hidden_size: int,
) -> torch.Tensor:
    """SGLang's hc_mix at the source's tactics."""
    import sglang.kernels.ops.gemm.hc_mix_splitk as hc_mix_splitk
    from sglang.kernels.ops.gemm.dense_bf16_gemm_sm100_splitk_epilogue import (
        run_splitk_dense_gate,
        run_splitk_dense_silu,
    )

    rows, k = hyper_input_normed.shape
    lowrank = w_down.shape[0]
    kpad = w_up_permuted_padded.shape[1]
    out = torch.empty(
        (rows, hidden_size),
        dtype=hyper_input_normed.dtype,
        device=hyper_input_normed.device,
    )
    if rows == 0:
        return out
    inv_hc = 1.0 / hc_count
    t_pad = hc_mix_splitk._get_scratch(
        lowrank, hyper_input_normed.dtype, hyper_input_normed.device
    )
    step = hc_mix_splitk._MAX_ROWS
    for row_start in range(0, rows, step):
        row_end = min(row_start + step, rows)
        m = row_end - row_start
        chunk = hyper_input_normed[row_start:row_end]
        run_splitk_dense_silu(
            chunk, w_down.T, t_pad[:m, :lowrank], True, tactic(m, lowrank, k), inv_hc
        )
        run_splitk_dense_gate(
            t_pad[:m],
            w_up_permuted_padded.T,
            chunk,
            out[row_start:row_end],
            True,
            tactic(m, k, kpad),
            inv_hc,
            hc_count,
        )
    return out
