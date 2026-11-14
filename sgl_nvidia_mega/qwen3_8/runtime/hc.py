# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The hyper-connection write-back of a covered step (slot ``hc``): for 1..32 rows of
4 x 2560 BF16 streams, one launch applies the gate, combines the block output into
the streams, normalizes them for the next read and, when the next read is a gated
mix, computes that read's gate partials (``hc_combine_norm_gate``). The read's own
mix stays SGLang's ``GatedResidual.mix`` on the normalized rows it is handed.

Stock: ``GatedResidual.combine`` (or ``combine_and_normalize``) and the next read's
``hc_norm``. Numerics: the combined and normalized streams are SGLang's for the same
gate partials; the partials are a different 8-way split of the same dot, so the gate
coefficient can differ in its last bit (a tolerance slot)."""

from __future__ import annotations

import torch

MAX_ROWS = 32
HC_COUNT, HIDDEN = 4, 2560

SGLANG_READS = {
    "srt/layers/hyperconnection.py": (
        "hc_norm",
        "block_inject_weight",
        "variance_epsilon",
        "group_size",
        "hc_count",
        "hidden_size",
    ),
}


def layer_supported(layer) -> str:
    """'' when this layer's HC write-backs can run the kernel, else the reason."""
    for hc in (layer.attn_hyper_connection, layer.mlp_hyper_connection):
        if hc.hc_count != HC_COUNT or hc.hidden_size != HIDDEN:
            return f"HC {hc.hc_count} x {hc.hidden_size}, the kernel takes 4 x 2560"
        if hc.block_inject_weight.weight.dtype != torch.bfloat16:
            return "HC inject weight is not BF16"
        if hc.hc_norm.group_size != HIDDEN or hc.hc_norm.weight.dtype != torch.bfloat16:
            return "HC norm is not a BF16 per-branch norm"
    return ""


def build() -> str:
    """Compile the kernels on every rank before capture; '' or the failure."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return "the HC kernels need an SM10x GPU"
    try:
        from sgl_nvidia_mega.qwen3_8.kernels.ops.elementwise import hc_combine_decode

        hc_combine_decode._jit_module()
        hc_combine_decode._jit_norm_gate_module()
    except Exception as e:  # a compile failure keeps the stock write-back
        return f"HC kernels do not build: {type(e).__name__}: {e}"
    return ""


def join_gate_stream(module, partials: torch.Tensor) -> None:
    """Join the side stream SGLang's one-row mix computed its gate partials on (as
    its combine does), so a capture has no unjoined work."""
    stream = getattr(module, "_gate_stream", None)
    if stream is not None:
        current = torch.cuda.current_stream()
        current.wait_stream(stream)
        partials.record_stream(current)


def gate(normed: torch.Tensor, inject_weight: torch.Tensor) -> torch.Tensor:
    """The gate partials [rows, 8, 4] FP32 of a read's normalized streams."""
    from sgl_nvidia_mega.qwen3_8.kernels.ops.elementwise.hc_combine_decode import (
        hc_combine_gate,
    )

    return hc_combine_gate(normed, inject_weight)


def write(block, residual, partials, next_norm, next_inject=None):
    """(combined streams, normalized for the next read or None, the next read's
    partials or None)."""
    from sgl_nvidia_mega.qwen3_8.kernels.ops.elementwise.hc_combine_decode import (
        hc_combine_apply,
        hc_combine_norm_gate,
    )

    if next_norm is None:
        return hc_combine_apply(block, residual, partials), None, None
    return hc_combine_norm_gate(
        block,
        residual,
        partials,
        next_norm.weight,
        next_norm.variance_epsilon,
        inject_weight=next_inject,
    )
