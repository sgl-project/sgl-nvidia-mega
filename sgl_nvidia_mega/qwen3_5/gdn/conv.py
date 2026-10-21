# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2024, Tri Dao.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# This file has been modified from its original version (the channel tile and a token-major output).
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/kernels/ops/mamba/causal_conv1d_triton.py (causal_conv1d_update).
"""The causal conv update of a GDN target-verify step on SGLang's Triton kernel, with
the two launch choices ``causal_conv1d_update`` fixes: the channel tile (``block_n``)
and a token-major output, which the token-major projection views need (their rows
have gaps, so ``empty_like`` would make the output channel-major).

Only the verify case: no circular cache_seqlens, no accept counts, no tree mask.
"""

from __future__ import annotations

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/ops/mamba/causal_conv1d_triton.py": (
        "_causal_conv1d_update_kernel",
        "PAD_SLOT_ID",
    ),
    "kernels/jit/utils/__init__.py": ("is_arch_support_pdl",),
}

# The SGLang functions whose body this module mirrors, by the digest of their code
# (test/unit/test_contract.py recomputes them on the installed SGLang)
SGLANG_MIRRORS = {
    "kernels/ops/mamba/causal_conv1d_triton.py:causal_conv1d_update": "3394efa20c88d580",
}


def conv_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias,
    activation,
    conv_state_indices: torch.Tensor,
    intermediate_conv_window: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    *,
    block_n: int = 256,
    token_major_out: bool = False,
) -> torch.Tensor:
    """``causal_conv1d_update`` of x (batch, dim, seqlen) for a verify step, with the
    channel tile ``block_n`` and, with ``token_major_out``, an output whose
    transpose(1, 2) is contiguous."""
    import triton
    from sglang.kernels.jit.utils import is_arch_support_pdl
    from sglang.kernels.ops.mamba.causal_conv1d_triton import (
        PAD_SLOT_ID,
        _causal_conv1d_update_kernel,
    )

    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    num_cache_lines, _, state_len = conv_state.size()
    out = (
        torch.empty((batch, seqlen, dim), dtype=x.dtype, device=x.device).transpose(
            1, 2
        )
        if token_major_out
        else torch.empty_like(x)
    )
    state_len = width - 1
    pdl_kwargs = {"USE_GDC": True, "launch_pdl": True} if is_arch_support_pdl() else {}

    def grid(meta):
        return (batch, triton.cdiv(dim, meta["BLOCK_N"]))

    _causal_conv1d_update_kernel[grid](
        x,
        weight,
        bias,
        conv_state,
        None,
        conv_state_indices,
        None,
        intermediate_conv_window,
        intermediate_state_indices,
        None,
        None,
        None,
        out,
        batch,
        dim,
        seqlen,
        state_len,
        num_cache_lines,
        *x.stride(),
        *weight.stride(),
        *conv_state.stride(),
        conv_state_indices.stride(0),
        *intermediate_conv_window.stride(),
        intermediate_state_indices.stride(0),
        0,
        0,
        0,
        0,
        0,
        0,
        *out.stride(),
        PAD_SLOT_ID,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_CONTINUOUS_BATCHING=True,
        IS_SPEC_DECODING=False,
        NP2_STATELEN=triton.next_power_of_2(state_len),
        NP2_SEQLEN=triton.next_power_of_2(seqlen),
        USE_PAD_SLOT=True,
        BLOCK_N=block_n,
        SAVE_INTERMEDIATE=True,
        HAS_EAGLE_TREE_CUSTOM_ATTN_MASK=False,
        **pdl_kwargs,
    )
    return out
