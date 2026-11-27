# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

SGLANG_READS = {
    "srt/layers/quantization/unquant.py": (
        "_use_cutedsl_bf16_gemm",
        "_enable_bf16_splitk_gemm",
        "use_bf16_splitk_gemm",
        "_use_hopper_bf16_gemv",
        "_use_hopper_bf16_tiny_gemv",
    ),
    "srt/models/qwen3_5.py": ("qkv_proj",),
}

OPTION = "draft_early_hops"
# the draft step widths the releases serve (the low-M races' widths)
MAX_ROWS = 16
_NOTED: set = set()


def on() -> bool:
    from sgl_nvidia_mega.qwen3_5 import runtime

    return runtime.measured_option(OPTION)


def note(key, line: str) -> None:
    """One engagement line per key, at graph capture."""
    if torch.cuda.is_current_stream_capturing() and key not in _NOTED:
        _NOTED.add(key)
        logger.info("NVIDIA megakernel: %s: %s", OPTION, line)


def qkv_takes_pdl(layer, rows: int) -> bool:
    mod = getattr(layer, "qkv_proj", None)
    w = getattr(mod, "weight", None)
    if (
        w is None
        or w.dtype != torch.bfloat16
        or w.dim() != 2
        or getattr(mod, "bias", None) is not None
        or type(getattr(mod, "quant_method", None)).__name__
        != "UnquantizedLinearMethod"
    ):
        return False
    return _tgv_takes(rows, int(w.shape[0]), int(w.shape[1]))


def _tgv_takes(rows: int, n: int, k: int) -> bool:
    try:
        import sglang.srt.layers.quantization.unquant as unquant
    except ImportError:
        return False
    if getattr(unquant, "_use_cutedsl_bf16_gemm", None) is None:
        return False
    if getattr(unquant, "_enable_bf16_splitk_gemm", False) and (
        unquant.use_bf16_splitk_gemm(rows, n, k)
    ):
        return False
    if getattr(unquant, "_use_hopper_bf16_gemv", None) is not None:
        return False
    if getattr(unquant, "_use_hopper_bf16_tiny_gemv", None) is not None:
        return False
    return bool(unquant._use_cutedsl_bf16_gemm(rows, n, k))


def releases_router(parts, rows: int) -> bool:
    if not on() or rows < 2 or rows > MAX_ROWS:
        return False
    block = parts.moe_block
    router = block.router if block is not None else None
    if router is not None and router.takes(rows):
        note(
            ("ffn_input", rows),
            f"the FFN-input consume of {rows} rows releases the low-M router",
        )
        return True
    note(("ffn_input", rows), f"the FFN-input consume of {rows} rows keeps its release")
    return False
