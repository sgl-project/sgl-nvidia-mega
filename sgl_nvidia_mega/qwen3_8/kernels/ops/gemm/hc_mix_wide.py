# SPDX-License-Identifier: Apache-2.0
# From SGLang python/sglang/kernels/ops/gemm/hc_mix_splitk.py: its wide path (hc_mix_wide), unchanged but for the
# imports: hc_mix_up_gate is the plugin's copy, and the split-K epilogue GEMM is SGLang's
# with its ``max_m`` (sgl-project/sglang#50194). The narrow path stays SGLang's hc_mix.
"""The HC mix of 33..512 rows: the CuTe split-K silu down GEMM over all rows, then
the Triton up GEMM with the gated mean fused (hc_mix_up_gate). The up weight is
``permute_pad_up_weight``'s layout (SGLang's hc_mix_splitk)."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

SGLANG_READS = {
    "kernels/ops/gemm/dense_bf16_gemm_sm100_splitk_epilogue.py": (
        "SplitKTactic",
        "validate_tactic",
        "run_splitk_dense_silu",
    ),
}

# hc_mix_splitk._MAX_ROWS: the narrow path's widest call.
_MAX_ROWS = 32


# Above 32 rows (TP1, hc 4): the CuTe split-K silu down over all rows, then a Triton up GEMM with the
# gated mean fused (hc_mix_up_gate, PDL). Best per row range on VR200; in isolation
# the gain fades above 320 rows, so SGLANG_HC_MIX_SPLITK_MAX_ROWS picks the cap.
# (rows up to, down (mma_m, mma_n, split_k, ab_stages), up+gate (BLOCK_M, BLOCK_J, BLOCK_K, warps, stages))
_WIDE_TACTICS = (
    (64, (64, 16, 8, 8), (64, 32, 64, 8, 3)),
    (144, (64, 16, 4, 8), (64, 32, 64, 8, 3)),
    (288, (64, 32, 4, 8), (64, 32, 64, 8, 3)),
    (320, (64, 16, 2, 8), (64, 32, 64, 8, 3)),
    (448, (64, 32, 2, 8), (64, 32, 64, 8, 3)),
    (512, (64, 32, 2, 8), (128, 32, 64, 8, 3)),
)
WIDE_MAX_ROWS = _WIDE_TACTICS[-1][0]
_wide_scratch_cache = {}
_wide_tactic_cache = {}
_wide_logged = False


def _get_wide_scratch(
    lowrank: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    key = (device, lowrank, dtype)
    buf = _wide_scratch_cache.get(key)
    if buf is None:
        buf = torch.zeros((WIDE_MAX_ROWS, lowrank), dtype=dtype, device=device)
        _wide_scratch_cache[key] = buf
    return buf


def _get_wide_tactics(m: int, lowrank: int, k: int):
    key = (m, lowrank, k)
    tactics = _wide_tactic_cache.get(key)
    if tactics is None:
        from sglang.kernels.ops.gemm.dense_bf16_gemm_sm100_splitk_epilogue import (
            SplitKTactic,
            validate_tactic,
        )

        down, up = next((d, u) for rows, d, u in _WIDE_TACTICS if m <= rows)
        tactics = (SplitKTactic(*down), up)
        validate_tactic(tactics[0], m, lowrank, k, max_m=WIDE_MAX_ROWS)
        _wide_tactic_cache[key] = tactics
    return tactics


def hc_mix_wide(
    hyper_input_normed: torch.Tensor,
    w_down: torch.Tensor,
    w_up_permuted_padded: torch.Tensor,
    hc_count: int,
    hidden_size: int,
) -> torch.Tensor:
    """The HC mix for 32 < rows <= WIDE_MAX_ROWS, hc_count 4: bf16(silu(x @ w_down.T / 4)) from the cluster
    split-K down (hc_mix's arithmetic), then the fused up GEMM and gated mean."""
    from sglang.kernels.ops.gemm.dense_bf16_gemm_sm100_splitk_epilogue import (
        run_splitk_dense_silu,
    )

    from sgl_nvidia_mega.qwen3_8.kernels.ops.gemm.hc_mix_up_gate import hc_up_gate

    global _wide_logged
    rows, k = hyper_input_normed.shape
    lowrank = w_down.shape[0]
    if hc_count != 4 or not _MAX_ROWS < rows <= WIDE_MAX_ROWS:
        raise ValueError(
            f"hc_mix_wide serves hc 4 and {_MAX_ROWS} < rows <= {WIDE_MAX_ROWS}, got hc {hc_count} rows {rows}"
        )
    t = _get_wide_scratch(lowrank, hyper_input_normed.dtype, hyper_input_normed.device)[
        :rows
    ]
    down, up = _get_wide_tactics(rows, lowrank, k)
    if not _wide_logged:
        _wide_logged = True
        logger.info(
            "HC mix wide on: rows %d..%d (CuTe split-K silu down + Triton up/gate, PDL); first rows=%d down %s up %s (SGLANG_HC_MIX_SPLITK_MAX_ROWS)",
            _MAX_ROWS + 1,
            WIDE_MAX_ROWS,
            rows,
            down,
            up,
        )
    run_splitk_dense_silu(
        hyper_input_normed, w_down.T, t, True, down, 1.0 / hc_count, max_m=WIDE_MAX_ROWS
    )
    return hc_up_gate(
        t,
        w_up_permuted_padded,
        hyper_input_normed,
        hidden_size,
        lowrank,
        config=up,
        pdl=True,
    )
