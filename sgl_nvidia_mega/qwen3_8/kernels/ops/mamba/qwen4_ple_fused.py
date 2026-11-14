# SPDX-License-Identifier: Apache-2.0
# From SGLang python/sglang/srt/models/qwen4_exp_ple_fused.py. Changed: ``conv_engages`` takes the
# SGLANG_OPT_PLE_FUSED_EXTEND_CONV_CUDNN_SHAPES switch as its ``cudnn_shapes`` argument
# instead of reading it (the executor's runtime/ple.py passes Options.extend_conv_cudnn).
# The kernels, their launches and every other decision are unchanged.
"""Qwen4 PLE extend short conv in one launch over the pass's token rows.

Under ``SGLANG_OPT_PLE_FUSED_EXTEND_CONV`` the extend short conv of ``Qwen4ExpPLELayer``
skips the stock padded layout (``new_zeros`` [B, W, C], ``index_put``, ``cat`` into
[B, C, S + W], the depthwise ``F.conv1d`` over B x W columns and the ``[req, offset]``
gather back) and reads each output's taps straight from ``x`` and the selected state
copy, so the pass needs [T, C] rows instead of three [B, ~W, C] temporaries.

The bytes are stock's: for a contiguous, dilated depthwise input that is 32-bit
indexable, ``F.conv1d`` runs ``conv_depthwise2d_forward_kernel_generic`` (torch
``DepthwiseConv2d.cu``), one thread per output summing ``w * x`` in FP32 from zero over
the taps in order and rounding once to BF16. BF16 x BF16 products are exact in FP32, so
the kernel's four ``fma.rn.f32`` in tap order give the same sum. The state snapshots are
copies of the same columns, and SiLU stays the caller's ``F.silu``. Passes stock hands to
cuDNN (not 32-bit indexable), non-EXTEND/MIXED passes, stream capture, non-CUDA tensors
and host lengths that disagree with the pass keep the stock code.

``SGLANG_OPT_PLE_FUSED_EXTEND_CONV_CUDNN_SHAPES`` (opt-in, only with the switch above) also
fuses the cuDNN-shape passes holding at most 16 sequences with the longest at most 32768
tokens. Stock there materializes the padded input three times (22.8 GiB for 15 sequences
with a 26,606-token longest at C 10240). Over that envelope the original branch measured
cuDNN's conv (``conv2d_grouped_direct_kernel``, torch 2.15.0+cu134 / cuDNN 9.26) byte-equal
to the native kernel's tap order on crafted tap-order, random, signed-zero and subnormal
data; other torch / cuDNN builds are not covered by that measurement.

One INFO line is logged the first time it engages and one per decline reason.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# ATen canUse32BitIndexMath on a contiguous tensor: numel < INT32_MAX.
_INT32_MAX = 2**31 - 1
_MAX_STATE_LEN = 16
# BLOCK_T, BLOCK_C, warps of the conv kernel (measured on VR200 by the original branch).
_CONV_TILE = (16, 256, 4)
# The cuDNN-shape envelope whose stock bytes were measured equal to the forward tap order.
_CUDNN_SHAPES_MAX_SEQS = 16
_CUDNN_SHAPES_MAX_ROW = 32768

STATS = {
    "conv_fused": 0,
    "conv_declined": 0,
    "conv_stock_cudnn_shape": 0,
    "conv_fused_cudnn_shape": 0,
}
_noted = set()


def _note_once(key: str, fmt: str, *args) -> None:
    if key not in _noted:
        _noted.add(key)
        logger.info(fmt, *args)


def native_depthwise(num_seqs: int, channels: int, columns: int) -> bool:
    """Whether stock F.conv1d runs a contiguous ``[num_seqs, channels, columns]``
    dilated depthwise input on the native kernel rather than cuDNN."""
    return num_seqs * channels * columns < _INT32_MAX


def _conv_decline_reason(
    x, batch, extend_seq_lens_cpu, weight, conv_state, channels, state_len
):
    if batch.use_decode_fast_path:
        return "decode fast path"
    if not 0 < state_len <= _MAX_STATE_LEN:
        return f"state length {state_len}"
    if extend_seq_lens_cpu is None:
        return "no host extend lengths"
    if not x.is_cuda or torch.version.hip is not None:
        return "not a CUDA tensor"
    if torch.cuda.is_current_stream_capturing():
        return "stream capture"
    if x.dtype != torch.bfloat16 or x.dim() != 2 or not x.is_contiguous():
        return f"x {x.dtype} {tuple(x.shape)} contiguous={x.is_contiguous()}"
    if x.shape[1] != channels:
        return "channel count"
    if conv_state.dim() != 3 or tuple(conv_state.shape[1:]) != (channels, state_len):
        return f"conv state shape {tuple(conv_state.shape)}"
    if (
        weight.dtype != torch.bfloat16
        or tuple(weight.shape) != (channels, 1, weight.shape[2])
        or not weight.is_contiguous()
    ):
        return f"weight {weight.dtype} {tuple(weight.shape)}"
    num_seqs, tokens = batch.lengths.shape[0], x.shape[0]
    lens = [int(n) for n in extend_seq_lens_cpu]
    if len(lens) != num_seqs or sum(lens) != tokens or min(lens, default=0) < 0:
        return "host extend lengths disagree with the pass"
    if batch.row_width != max(lens, default=0):
        return "row width is not the longest host length"
    if tokens * channels >= _INT32_MAX * 4:
        return "pass too large"
    return None


def conv_engages(
    x: torch.Tensor,
    batch,
    extend_seq_lens_cpu: Optional[Sequence[int]],
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    channels: int,
    state_len: int,
    cudnn_shapes: bool = False,
) -> bool:
    """Host-only decision for one pass of the stock general path (no device sync);
    the caller checks SGLANG_OPT_PLE_FUSED_EXTEND_CONV."""
    mode_name = batch.mode.name
    if mode_name not in ("EXTEND", "MIXED"):
        return False
    num_seqs, tokens = batch.lengths.shape[0], x.shape[0]
    reason = _conv_decline_reason(
        x, batch, extend_seq_lens_cpu, weight, conv_state, channels, state_len
    )
    if reason is not None:
        STATS["conv_declined"] += 1
        _note_once(
            "conv-decline:" + reason,
            "PLE fused extend conv declined a pass (mode %s), stock code ran (%s): "
            "seqs=%d tokens=%d; later passes declined for this reason are counted only",
            mode_name,
            reason,
            num_seqs,
            tokens,
        )
        return False
    if not native_depthwise(num_seqs, channels, state_len + batch.row_width):
        if not cudnn_shapes:
            STATS["conv_stock_cudnn_shape"] += 1
            _note_once(
                "conv-cudnn",
                "PLE fused extend conv kept stock for a pass stock F.conv1d hands to "
                "cuDNN (seqs=%d tokens=%d longest=%d); later such passes are counted only",
                num_seqs,
                tokens,
                batch.row_width,
            )
            return False
        if num_seqs > _CUDNN_SHAPES_MAX_SEQS or batch.row_width > _CUDNN_SHAPES_MAX_ROW:
            STATS["conv_stock_cudnn_shape"] += 1
            _note_once(
                "conv-cudnn-outside",
                "PLE fused extend conv kept stock for a pass stock F.conv1d hands to "
                "cuDNN outside the envelope (seqs=%d tokens=%d longest=%d; envelope "
                "seqs <= %d, longest <= %d); later such passes are counted only",
                num_seqs,
                tokens,
                batch.row_width,
                _CUDNN_SHAPES_MAX_SEQS,
                _CUDNN_SHAPES_MAX_ROW,
            )
            return False
        STATS["conv_fused_cudnn_shape"] += 1
        _note_once(
            "conv-cudnn-fused",
            "PLE fused extend conv fused a pass stock F.conv1d hands to cuDNN "
            "(SGLANG_OPT_PLE_FUSED_EXTEND_CONV_CUDNN_SHAPES; seqs=%d tokens=%d "
            "longest=%d; stock's padded temporaries about %.2f GiB); later such passes "
            "are counted only",
            num_seqs,
            tokens,
            batch.row_width,
            num_seqs * (3 * batch.row_width + state_len) * channels * 2 / 2**30,
        )
    STATS["conv_fused"] += 1
    _note_once(
        "conv-on",
        "PLE fused extend conv on (SGLANG_OPT_PLE_FUSED_EXTEND_CONV), first pass "
        "seqs=%d tokens=%d longest=%d (stock would run %d padded columns)",
        num_seqs,
        tokens,
        batch.row_width,
        num_seqs * (state_len + batch.row_width),
    )
    return True


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
def _ple_extend_conv_kernel(
    x_ptr,
    state_ptr,
    weight_ptr,
    req_ptr,
    offset_ptr,
    out_ptr,
    num_tokens,
    CHANNELS: tl.constexpr,
    STATE_LEN: tl.constexpr,
    KSIZE: tl.constexpr,
    DILATION: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # out[t, c] = sum_k w[c, k] * in[req(t), c, offset(t) + k * DILATION], where the
    # stock conv input column j is state[req, c, j] for j < STATE_LEN, else token row
    # (sequence start + j - STATE_LEN) = t + j - offset(t) - STATE_LEN.
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    t_mask = t < num_tokens
    c_mask = c < CHANNELS
    mask = t_mask[:, None] & c_mask[None, :]
    req = tl.load(req_ptr + t, mask=t_mask, other=0).to(tl.int64)
    offset = tl.load(offset_ptr + t, mask=t_mask, other=0).to(tl.int64)
    t64 = t.to(tl.int64)
    c64 = c.to(tl.int64)
    acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
    for k in tl.static_range(KSIZE):
        col = offset + k * DILATION
        from_state = (col < STATE_LEN)[:, None]
        state_off = (req * (CHANNELS * STATE_LEN) + col)[:, None] + (c64 * STATE_LEN)[
            None, :
        ]
        token_off = ((t64 + (k * DILATION - STATE_LEN)) * CHANNELS)[:, None] + c64[
            None, :
        ]
        v_state = tl.load(state_ptr + state_off, mask=mask & from_state, other=0.0)
        v_token = tl.load(x_ptr + token_off, mask=mask & (~from_state), other=0.0)
        v = tl.where(from_state, v_state, v_token).to(tl.float32)
        w = tl.load(weight_ptr + c64 * KSIZE + k, mask=c_mask, other=0.0).to(tl.float32)
        acc = _fma_rn_f32(tl.broadcast_to(w[None, :], (BLOCK_T, BLOCK_C)), v, acc)
    tl.store(
        out_ptr + (t64 * CHANNELS)[:, None] + c64[None, :],
        acc.to(tl.bfloat16),
        mask=mask,
    )


@triton.jit
def _ple_extend_state_kernel(
    x_ptr,
    state_ptr,
    start_ptr,
    offset_ptr,
    out_ptr,
    CHANNELS: tl.constexpr,
    STATE_LEN: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # out[b, c, j] = stock conv_input[b, c, offset[b] + j] for j < STATE_LEN.
    b = tl.program_id(0).to(tl.int64)
    c = (tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)).to(tl.int64)
    j = tl.arange(0, BLOCK_S).to(tl.int64)
    c_mask = (c < CHANNELS)[:, None]
    j_mask = (j < STATE_LEN)[None, :]
    start = tl.load(start_ptr + b).to(tl.int64)
    offset = tl.load(offset_ptr + b).to(tl.int64)
    col = offset + j
    from_state = (col < STATE_LEN)[None, :]
    v_state = tl.load(
        state_ptr
        + b * (CHANNELS * STATE_LEN)
        + (c * STATE_LEN)[:, None]
        + col[None, :],
        mask=c_mask & j_mask & from_state,
        other=0.0,
    )
    v_token = tl.load(
        x_ptr + ((start + col - STATE_LEN) * CHANNELS)[None, :] + c[:, None],
        mask=c_mask & j_mask & (~from_state),
        other=0.0,
    )
    tl.store(
        out_ptr + b * (CHANNELS * STATE_LEN) + (c * STATE_LEN)[:, None] + j[None, :],
        tl.where(from_state, v_state, v_token),
        mask=c_mask & j_mask,
    )


def fused_extend_conv(
    x: torch.Tensor,
    state: torch.Tensor,
    weight: torch.Tensor,
    req_indices: torch.Tensor,
    token_offsets: torch.Tensor,
    dilation: int,
) -> torch.Tensor:
    """Stock's pre-SiLU ``conv_output[req_indices, token_offsets]``, [T, C] BF16.

    ``x`` [T, C] BF16 (rows of a sequence contiguous and in order), ``state`` [B, C, S]
    BF16 (the selected state copy), ``weight`` [C, 1, K] BF16."""
    tokens, channels = x.shape
    state_len = state.shape[2]
    ksize = weight.shape[2]
    assert state_len == (ksize - 1) * dilation
    out = torch.empty((tokens, channels), dtype=x.dtype, device=x.device)
    if tokens == 0:
        return out
    block_t, block_c, warps = _CONV_TILE
    _ple_extend_conv_kernel[
        (triton.cdiv(tokens, block_t), triton.cdiv(channels, block_c))
    ](
        x,
        state.contiguous(),
        weight,
        req_indices,
        token_offsets,
        out,
        tokens,
        CHANNELS=channels,
        STATE_LEN=state_len,
        KSIZE=ksize,
        DILATION=dilation,
        BLOCK_T=block_t,
        BLOCK_C=block_c,
        num_warps=warps,
    )
    return out


def extend_state_at(
    x: torch.Tensor,
    state: torch.Tensor,
    starts: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Stock ``_gather_at(offsets)``: ``conv_input[b, :, offsets[b] + j]`` for
    ``j < S``, [B, C, S] in x's dtype. ``starts[b]`` is sequence b's first token row."""
    num_seqs, channels, state_len = state.shape
    out = torch.empty((num_seqs, channels, state_len), dtype=x.dtype, device=x.device)
    if num_seqs == 0:
        return out
    block_c = 128
    _ple_extend_state_kernel[(num_seqs, triton.cdiv(channels, block_c))](
        x,
        state.contiguous(),
        starts.contiguous(),
        offsets.contiguous(),
        out,
        CHANNELS=channels,
        STATE_LEN=state_len,
        BLOCK_S=triton.next_power_of_2(state_len),
        BLOCK_C=block_c,
        num_warps=4,
    )
    return out


def fused_short_conv_extend(x, state, weight, batch, *, dilation, state_offsets):
    """(pre-SiLU conv rows [T, C], one [B, C, S] state snapshot per ``state_offsets``)."""
    lengths = batch.lengths
    starts = torch.cumsum(lengths, dim=0) - lengths
    conv = fused_extend_conv(
        x, state, weight, batch.req_indices, batch.token_offsets, dilation
    )
    snaps = tuple(extend_state_at(x, state, starts, o) for o in state_offsets)
    return conv, snaps
