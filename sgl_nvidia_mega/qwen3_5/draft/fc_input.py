# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from TensorRT-LLM (Apache-2.0): tensorrt_llm/_torch/modules/q35_mtp_glue.py.
# This file has been modified from its original version (strided token ids).
"""Slot ``draft_fc_input``: the MTP layer's fc input in one launch."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _q35_mtp_input_kernel(
    ids_ptr,
    ids_stride,
    emb_ptr,
    hid_ptr,
    hid_stride,
    w_emb_ptr,
    w_hid_ptr,
    out_ptr,
    out_stride,
    vocab,
    eps_emb,
    eps_hid,
    chunk_lo,
    chunk_w,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    TRIG: tl.constexpr = False,
):
    if TRIG:
        # the dependent loads only its weight before its own wait
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0)
    half = tl.program_id(1)
    lo = tl.maximum(chunk_lo, half * H)
    hi = tl.minimum(chunk_lo + chunk_w, (half + 1) * H)
    if lo < hi:  # this half intersects the rank's chunk
        offs = tl.arange(0, BLOCK_H)
        m = offs < H
        if half == 0:
            tok = tl.load(ids_ptr + row * ids_stride).to(tl.int64)
            valid = (tok >= 0) & (tok < vocab)
            tok = tl.where(valid, tok, 0)
            x = tl.load(emb_ptr + tok * H + offs, mask=m, other=0.0).to(tl.float32)
            x = tl.where(valid, x, 0.0)
            w = tl.load(w_emb_ptr + offs, mask=m, other=0.0).to(tl.float32)
            eps = eps_emb
        else:
            x = tl.load(
                hid_ptr + row.to(tl.int64) * hid_stride + offs, mask=m, other=0.0
            ).to(tl.float32)
            w = tl.load(w_hid_ptr + offs, mask=m, other=0.0).to(tl.float32)
            eps = eps_hid
        var = tl.sum(x * x, axis=0) / H
        y = x * tl.rsqrt(var + eps) * (1.0 + w)
        col = offs + half * H
        keep = m & (col >= lo) & (col < hi)
        tl.store(
            out_ptr + row.to(tl.int64) * out_stride + (col - chunk_lo),
            y.to(out_ptr.dtype.element_ty),
            mask=keep,
        )


def mtp_fused_input(
    input_ids: torch.Tensor,
    embed_weight: torch.Tensor,
    hidden_states: torch.Tensor,
    norm_emb_weight: torch.Tensor,
    norm_hid_weight: torch.Tensor,
    eps_emb: float,
    eps_hid: float,
    tp_size: int = 1,
    tp_rank: int = 0,
    release_next: bool = False,
) -> torch.Tensor:
    if input_ids.dim() != 1:
        input_ids = input_ids.reshape(-1)
    rows = input_ids.numel()
    h = hidden_states.shape[-1]
    if (
        embed_weight.shape[1] != h
        or hidden_states.shape[0] != rows
        or (2 * h) % tp_size
        or hidden_states.stride(-1) != 1
        or not embed_weight.is_contiguous()
    ):
        raise ValueError(
            f"mtp_fused_input: table {tuple(embed_weight.shape)}, hidden "
            f"{tuple(hidden_states.shape)} {tuple(hidden_states.stride())}, "
            f"{rows} ids, tp {tp_size}"
        )
    chunk_w = 2 * h // tp_size
    out = torch.empty(
        (rows, chunk_w), dtype=hidden_states.dtype, device=hidden_states.device
    )
    if rows == 0:
        return out
    block_h = triton.next_power_of_2(h)
    _q35_mtp_input_kernel[(rows, 2)](
        input_ids,
        input_ids.stride(0),
        embed_weight,
        hidden_states,
        hidden_states.stride(0),
        norm_emb_weight,
        norm_hid_weight,
        out,
        out.stride(0),
        embed_weight.shape[0],
        float(eps_emb),
        float(eps_hid),
        tp_rank * chunk_w,
        chunk_w,
        H=h,
        BLOCK_H=block_h,
        num_warps=8 if block_h >= 4096 else 4,
        **({"TRIG": True} if release_next else {}),
    )
    return out


__all__ = ["mtp_fused_input"]
