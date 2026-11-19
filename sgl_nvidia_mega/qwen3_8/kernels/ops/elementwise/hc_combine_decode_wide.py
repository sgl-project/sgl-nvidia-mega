# SPDX-License-Identifier: Apache-2.0
# From SGLang python/sglang/kernels/ops/elementwise/hc_combine_decode.py: the gate, the
# apply and the wide combine-norm, compiled from a copy of its hc_combine_decode.cuh
# (csrc/elementwise/hc_combine_decode_wide.cuh, unchanged).
"""HC gate partials, combine and wide combine + next norm for 33..128 rows.

Four branches of 2560 BF16 values. ``hc_combine_gate`` gives a read's [rows, 8, 4]
FP32 gate partials; ``hc_combine_apply`` and ``hc_combine_apply_norm_wide`` apply
them at the write-back, the latter with the next read's per-branch norm.
"""

import torch

from sgl_nvidia_mega.qwen3_8._jit import cache_once, load_jit

# HcCombineNormDecodeWide's row cap (its CHECK_HOST).
WIDE_MAX_ROWS = 128


@cache_once
def _jit_module():
    return load_jit(
        "hc_combine_decode_wide",
        cuda_files=["elementwise/hc_combine_decode_wide.cuh"],
        cuda_wrappers=[
            ("gate", "HcCombineDecode::gate"),
            ("apply", "HcCombineDecode::apply"),
            ("apply_norm_wide", "HcCombineNormDecodeWide::run"),
        ],
    )


def hc_combine_gate(normed: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    partials = torch.empty(
        (normed.shape[0], 8, 4), device=normed.device, dtype=torch.float32
    )
    _jit_module().gate(normed, weight, partials)
    return partials


def hc_combine_apply(
    block: torch.Tensor, residual: torch.Tensor, partials: torch.Tensor
) -> torch.Tensor:
    combined = torch.empty_like(residual)
    _jit_module().apply(block, residual, partials, combined)
    return combined


def hc_combine_apply_norm_wide(
    block: torch.Tensor,
    residual: torch.Tensor,
    partials: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """hc_combine_apply_norm at 33..128 rows: a two-CTA cluster per row-branch."""
    combined = torch.empty_like(residual)
    normalized = torch.empty_like(residual)
    _jit_module().apply_norm_wide(
        block, residual, partials, weight, combined, normalized, eps
    )
    return combined, normalized
