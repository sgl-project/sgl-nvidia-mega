# SPDX-License-Identifier: Apache-2.0
# From SGLang python/sglang/kernels/ops/mamba/qwen4_short_conv.py: the one-launch target-verify short conv that commit added
# (``_note_once`` through ``_launch_verify_conv_one_kernel``, unchanged) and its branch
# in ``fused_qwen4_verify_conv``, whose SGLANG_OPT_PLE_FUSED_VERIFY_CONV read became the
# ``one_kernel`` argument. The rest of that function (the stock prepare + F.conv1d +
# finish chain) is SGLang main's ``fused_qwen4_verify_conv``, which is the same code.
"""The Qwen4 PLE target-verify short conv in one launch.

``fused_qwen4_verify_conv`` is SGLang main's function of that name (prepare, F.conv1d,
finish) with the source's one-kernel branch first: below ATen's 32-bit index limit
(where stock F.conv1d runs its native depthwise kernel) one launch reads each tap from the
selected state copy or the request's rows, sums them as that kernel does (FP32 from zero,
one fused multiply-add per tap in tap order, one rounding) and applies the stock finish
(SiLU with its BF16 rounding, the residual add, the valid mask), writing the same
intermediate windows; the same bits as main's chain (the source's claim)."""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl
from sglang.kernels.numerics import round_bf16_to_fp32

logger = logging.getLogger(__name__)

SGLANG_READS = {
    "kernels/numerics.py": ("round_bf16_to_fp32",),
    "kernels/ops/mamba/qwen4_short_conv.py": (
        "can_fuse_qwen4_verify_conv",
        "fused_qwen4_verify_conv",
    ),
}

# ATen canUse32BitIndexMath on the contiguous [requests, channels, state + width] conv
# input: below it stock F.conv1d runs the native depthwise kernel (DepthwiseConv2d.cu).
_INT32_MAX = 2**31 - 1
_noted = set()


def fused_qwen4_verify_conv(
    x,
    residual,
    weight,
    state,
    state_indices,
    valid,
    width,
    dilation,
    intermediate,
    *,
    one_kernel=True,
):
    from sglang.kernels.ops.mamba.qwen4_short_conv import (
        can_fuse_qwen4_verify_conv,
    )

    if not can_fuse_qwen4_verify_conv(
        x, residual, weight, state, state_indices, valid, width, dilation, intermediate
    ):
        raise ValueError("unsupported input for Qwen4 PLE verify convolution")
    output = torch.empty_like(x)
    if x.shape[0] == 0:
        return output
    requests = state_indices.numel()
    channels, state_len = state.shape[1:]
    if one_kernel:
        # Below ATen's 32-bit index limit stock F.conv1d runs its native depthwise
        # kernel, whose arithmetic the one kernel reproduces; above it stock keeps its own.
        if requests * channels * (state_len + width) < _INT32_MAX:
            _note_once(
                "on",
                "PLE verify conv in one kernel (SGLANG_OPT_PLE_FUSED_VERIFY_CONV), first "
                "call requests=%d channels=%d width=%d state=%d intermediate=%s",
                requests,
                channels,
                width,
                state_len,
                None if intermediate is None else intermediate.dtype,
            )
            _launch_verify_conv_one_kernel(
                x,
                residual,
                weight,
                state,
                state_indices,
                valid,
                width,
                dilation,
                intermediate,
                output,
            )
            return output
        _note_once(
            "stock",
            "SGLANG_OPT_PLE_FUSED_VERIFY_CONV: kept the stock chain for a call stock "
            "F.conv1d does not run on its native depthwise kernel (requests=%d "
            "channels=%d columns=%d)",
            requests,
            channels,
            state_len + width,
        )
    return _stock_verify_conv(
        x, residual, weight, state, state_indices, valid, width, dilation, intermediate
    )


def _stock_verify_conv(*args):
    """SGLang main's prepare + F.conv1d + finish chain."""
    from sglang.kernels.ops.mamba.qwen4_short_conv import fused_qwen4_verify_conv

    return fused_qwen4_verify_conv(*args)


def _note_once(key, fmt, *args):
    if key not in _noted:
        _noted.add(key)
        logger.info(fmt, *args)


@triton.jit
def _fma_rn_f32(a, b, c):
    return tl.inline_asm_elementwise(
        asm="fma.rn.f32 $0, $1, $2, $3;",
        constraints="=f,f,f,f",
        args=[a, b, c],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _verify_conv_rows(
    x_ptr,
    residual_ptr,
    weight_ptr,
    state_ptr,
    valid_ptr,
    output_ptr,
    req,
    slot,
    part,
    state_stride0,
    state_stride1,
    state_stride2,
    weight_stride0,
    weight_stride2,
    CHANNELS: tl.constexpr,
    WIDTH: tl.constexpr,
    STATE_LEN: tl.constexpr,
    KSIZE: tl.constexpr,
    DILATION: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Output rows req * WIDTH + step for BLOCK_C channels. Tap k of step s is column
    # s + k * DILATION of stock's conv input [state (cast to x's dtype), the request's x rows],
    # read in place; the sum is the native depthwise kernel's: FP32 from zero, one fused
    # multiply-add per tap in tap order, rounded once to x's dtype. Then stock finish's arithmetic.
    step = tl.arange(0, BLOCK_W)
    channel = part * BLOCK_C + tl.arange(0, BLOCK_C)
    step_mask = step < WIDTH
    channel_mask = channel < CHANNELS
    mask = step_mask[:, None] & channel_mask[None, :]
    acc = tl.zeros((BLOCK_W, BLOCK_C), dtype=tl.float32)
    for k in tl.static_range(KSIZE):
        col = (step + k * DILATION)[:, None]
        from_state = col < STATE_LEN
        state = tl.load(
            state_ptr
            + slot * state_stride0
            + channel[None, :] * state_stride1
            + col * state_stride2,
            mask & from_state,
            other=0,
        ).to(x_ptr.dtype.element_ty)
        xv = tl.load(
            x_ptr + (req * WIDTH + col - STATE_LEN) * CHANNELS + channel[None, :],
            mask & (col >= STATE_LEN),
            other=0,
        )
        tap = tl.where(from_state, state, xv).to(tl.float32)
        w = tl.load(
            weight_ptr + channel * weight_stride0 + k * weight_stride2,
            channel_mask,
            other=0,
        ).to(tl.float32)
        acc = _fma_rn_f32(tl.broadcast_to(w[None, :], (BLOCK_W, BLOCK_C)), tap, acc)
    conv = acc.to(x_ptr.dtype.element_ty).to(tl.float32)
    activated = round_bf16_to_fp32(conv * tl.sigmoid(conv))
    token = req * WIDTH + step
    offsets = token[:, None] * CHANNELS + channel[None, :]
    residual = tl.load(residual_ptr + offsets, mask, other=0).to(tl.float32)
    valid = tl.load(valid_ptr + token, step_mask, other=0)
    tl.store(
        output_ptr + offsets,
        tl.where(valid[:, None], activated + residual, 0),
        mask,
    )


@triton.jit
def _verify_conv_windows_contiguous(
    x_ptr,
    state_ptr,
    valid_ptr,
    intermediate_ptr,
    req,
    slot,
    part,
    offsets,
    state_stride0,
    cache_stride0,
    CHANNELS: tl.constexpr,
    WIDTH: tl.constexpr,
    STATE_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # offsets = step * C * S + channel * S + state_col; the state column read is
    # col = step + 1 + state_col, at channel * S + col = offsets - step * C * S + step + 1.
    mask = offsets < WIDTH * CHANNELS * STATE_LEN
    if (CHANNELS * STATE_LEN) % BLOCK == 0:
        # A block lies in one step: one valid flag, and an invalid step only writes zeros.
        step = (part * BLOCK) // (CHANNELS * STATE_LEN)
        within = offsets - step * (CHANNELS * STATE_LEN)
        valid = tl.load(valid_ptr + req * WIDTH + step)
        if valid:
            channel = within // STATE_LEN
            col = step + 1 + within - channel * STATE_LEN
            state = tl.load(
                state_ptr + slot * state_stride0 + within + step + 1,
                mask & (col < STATE_LEN),
                other=0,
            ).to(x_ptr.dtype.element_ty)
            x = tl.load(
                x_ptr + (req * WIDTH + col - STATE_LEN) * CHANNELS + channel,
                mask & (col >= STATE_LEN),
                other=0,
            )
            value = tl.where(col < STATE_LEN, state, x)
        else:
            value = tl.zeros([BLOCK], dtype=x_ptr.dtype.element_ty)
        tl.store(intermediate_ptr + req * cache_stride0 + offsets, value, mask)
    else:
        step = offsets // (CHANNELS * STATE_LEN)
        within = offsets - step * (CHANNELS * STATE_LEN)
        channel = within // STATE_LEN
        state_col = within - channel * STATE_LEN
        col = step + 1 + state_col
        valid = tl.load(valid_ptr + req * WIDTH + step, mask, other=0)
        state = tl.load(
            state_ptr + slot * state_stride0 + within + step + 1,
            mask & (col < STATE_LEN),
            other=0,
        ).to(x_ptr.dtype.element_ty)
        x = tl.load(
            x_ptr + (req * WIDTH + col - STATE_LEN) * CHANNELS + channel,
            mask & (col >= STATE_LEN),
            other=0,
        )
        value = tl.where(valid, tl.where(col < STATE_LEN, state, x), 0)
        tl.store(intermediate_ptr + req * cache_stride0 + offsets, value, mask)


@triton.jit
def _verify_conv_windows(
    x_ptr,
    state_ptr,
    valid_ptr,
    intermediate_ptr,
    req,
    slot,
    part,
    state_stride0,
    state_stride1,
    state_stride2,
    cache_stride0,
    cache_stride1,
    cache_stride2,
    cache_stride3,
    CHANNELS: tl.constexpr,
    WIDTH: tl.constexpr,
    STATE_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
    CONTIGUOUS: tl.constexpr,
):
    # The intermediate windows, as _qwen4_verify_conv_prepare_kernel writes them.
    offsets = part * BLOCK + tl.arange(0, BLOCK)
    if CONTIGUOUS:
        # Contiguous [C, S] state rows and [W, C, S] windows: the same entries and values,
        # addressed as one span per request so the stores vectorize.
        _verify_conv_windows_contiguous(
            x_ptr,
            state_ptr,
            valid_ptr,
            intermediate_ptr,
            req,
            slot,
            part,
            offsets,
            state_stride0,
            cache_stride0,
            CHANNELS,
            WIDTH,
            STATE_LEN,
            BLOCK,
        )
    else:
        step = offsets // (CHANNELS * STATE_LEN)
        channel = (offsets // STATE_LEN) % CHANNELS
        state_col = offsets % STATE_LEN
        col = step + 1 + state_col
        mask = step < WIDTH
        valid = tl.load(valid_ptr + req * WIDTH + step, mask, other=0)
        state = tl.load(
            state_ptr
            + slot * state_stride0
            + channel * state_stride1
            + col * state_stride2,
            mask & (col < STATE_LEN),
            other=0,
        ).to(x_ptr.dtype.element_ty)
        x = tl.load(
            x_ptr + (req * WIDTH + col - STATE_LEN) * CHANNELS + channel,
            mask & (col >= STATE_LEN),
            other=0,
        )
        value = tl.where(valid, tl.where(col < STATE_LEN, state, x), 0)
        tl.store(
            intermediate_ptr
            + req * cache_stride0
            + step * cache_stride1
            + channel * cache_stride2
            + state_col * cache_stride3,
            value,
            mask,
        )


@triton.jit
def _qwen4_verify_conv_one_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    state_ptr,
    indices_ptr,
    valid_ptr,
    output_ptr,
    intermediate_ptr,
    state_stride0,
    state_stride1,
    state_stride2,
    index_stride,
    weight_stride0,
    weight_stride2,
    cache_stride0,
    cache_stride1,
    cache_stride2,
    cache_stride3,
    CHANNELS: tl.constexpr,
    WIDTH: tl.constexpr,
    STATE_LEN: tl.constexpr,
    KSIZE: tl.constexpr,
    DILATION: tl.constexpr,
    CONV_BLOCKS: tl.constexpr,
    HAS_INTERMEDIATE: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK: tl.constexpr,
    CONTIGUOUS: tl.constexpr,
    PDL_SMALL: tl.constexpr = False,
):
    # Stock prepare + F.conv1d + finish of fused_qwen4_verify_conv in one launch: programs
    # [0, CONV_BLOCKS) of a request write output rows, the rest its intermediate windows.
    if PDL_SMALL:
        tl.extra.cuda.gdc_wait()
    req = tl.program_id(0)
    part = tl.program_id(1)
    slot = tl.load(indices_ptr + req * index_stride)
    if part < CONV_BLOCKS:
        _verify_conv_rows(
            x_ptr,
            residual_ptr,
            weight_ptr,
            state_ptr,
            valid_ptr,
            output_ptr,
            req,
            slot,
            part,
            state_stride0,
            state_stride1,
            state_stride2,
            weight_stride0,
            weight_stride2,
            CHANNELS,
            WIDTH,
            STATE_LEN,
            KSIZE,
            DILATION,
            BLOCK_W,
            BLOCK_C,
        )
    elif HAS_INTERMEDIATE:
        _verify_conv_windows(
            x_ptr,
            state_ptr,
            valid_ptr,
            intermediate_ptr,
            req,
            slot,
            part - CONV_BLOCKS,
            state_stride0,
            state_stride1,
            state_stride2,
            cache_stride0,
            cache_stride1,
            cache_stride2,
            cache_stride3,
            CHANNELS,
            WIDTH,
            STATE_LEN,
            BLOCK,
            CONTIGUOUS,
        )
    if PDL_SMALL:
        tl.extra.cuda.gdc_launch_dependents()


# Launch shape of the one kernel: BLOCK_C channels x the verify width per conv program,
# BLOCK flattened (step, channel, state column) entries per intermediate program.
# Picked by a unit sweep on VR200 at 48 requests (all shapes bit-identical).
_ONE_KERNEL_BLOCK_C = 256
_ONE_KERNEL_BLOCK = 2048
_ONE_KERNEL_WARPS = 4


def _launch_verify_conv_one_kernel(
    x,
    residual,
    weight,
    state,
    state_indices,
    valid,
    width,
    dilation,
    intermediate,
    output,
):
    requests = state_indices.numel()
    channels, state_len = state.shape[1:]
    has_intermediate = intermediate is not None and state_len > 0
    contiguous = has_intermediate and (
        state.stride(1) == state_len
        and state.stride(2) == 1
        and intermediate.stride(1) == channels * state_len
        and intermediate.stride(2) == state_len
        and intermediate.stride(3) == 1
    )
    conv_blocks = triton.cdiv(channels, _ONE_KERNEL_BLOCK_C)
    intermediate_blocks = (
        triton.cdiv(width * channels * state_len, _ONE_KERNEL_BLOCK)
        if has_intermediate
        else 0
    )
    _qwen4_verify_conv_one_kernel[(requests, conv_blocks + intermediate_blocks)](
        x,
        residual,
        weight,
        state,
        state_indices,
        valid,
        output,
        intermediate if has_intermediate else x,
        *state.stride(),
        state_indices.stride(0),
        weight.stride(0),
        weight.stride(2),
        *(intermediate.stride() if has_intermediate else (0, 0, 0, 0)),
        CHANNELS=channels,
        WIDTH=width,
        STATE_LEN=state_len,
        KSIZE=weight.shape[2],
        DILATION=dilation,
        CONV_BLOCKS=conv_blocks,
        HAS_INTERMEDIATE=has_intermediate,
        BLOCK_W=max(2, triton.next_power_of_2(width)),
        BLOCK_C=_ONE_KERNEL_BLOCK_C,
        BLOCK=_ONE_KERNEL_BLOCK,
        CONTIGUOUS=contiguous,
        num_warps=_ONE_KERNEL_WARPS,
        enable_fp_fusion=False,
    )
    return output
