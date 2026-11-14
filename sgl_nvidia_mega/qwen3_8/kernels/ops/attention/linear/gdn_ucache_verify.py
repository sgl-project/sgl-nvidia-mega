# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/layers/attention/linear/gdn_replayssm_ucache.py (its verify half).
"""The verify-side kernels and helpers of the GDN u-cache (deferred-fold ReplaySSM)
target verify; the rings, the cursor commit, the resets and the track materializer are
``gdn_ucache_state.py``.

Byte-identical to the source: the Triton kernels (cursor gather by state
slot or by request row, padded-row zeroing, the flush-first row order, the q/k/v
split, the fp32 gate-parameter refresh) and their host launchers. Changed: the import
path of the vendored SM107 flush kernel (``gdn_ucache_rubin/``, here in the plugin);
SM100 / SM103 still import FlashInfer's ``gated_delta_rule_mtp_ucache_flush_umma``.
The environment-driven option readers stay in SGLang's branch; the executor passes
``GdnUcacheFlushOptions`` (``runtime/gdn_ucache.py``)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

import msgspec
import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.linear.gdn_ucache_state import (
        GdnUcacheBuffers,
    )

SGLANG_READS = {
    "kernels/ops/attention/fla/layernorm_gated.py": ("calc_rows_per_block",),
}

logger = logging.getLogger(__name__)

# FlashInfer u-cache ring geometry (gdn_replay_mtp_umma.RING_SLOTS / W_RING).
RING_SLOTS = 32
_RING_MASK = RING_SLOTS - 1
_MAX_WINDOW = 16
_CURSOR_BLOCK = 256


class GdnUcacheRows(msgspec.Struct, frozen=True, kw_only=True):
    """One verify's per-row view of the cursors; padded rows have slot -1."""

    slots: torch.Tensor
    hist_len: torch.Tensor
    cache_base: torch.Tensor
    # int32 ring row of each batch row (by_request_row only; 0 on padded rows).
    ring_rows: Optional[torch.Tensor] = None


class GdnUcacheFlushOptions(msgspec.Struct, frozen=True, kw_only=True):
    """Launch options of the SM107 flush kernel; all off by default, none changes a result."""

    # Read the packed post-conv q / k / v and the strided b / a views in place.
    strided_qkv: bool = False
    # Programmatic dependent launch behind the conv.
    pdl: bool = False
    # Rows that fold their window first in the launch order.
    flush_first: bool = False

    def kernel_kwargs(self) -> dict:
        kw = {}
        if self.strided_qkv:
            kw["strided_inputs"] = True
        if self.pdl:
            kw["pdl"] = True
        return kw


# Mamba slot 0 is the reserved dummy target of padded rows, so it is treated as
# padding (-1): the kernel then skips the row and leaves its output zero.
@triton.jit
def _gather_rows_kernel(
    slots_ptr,
    hist_pool_ptr,
    base_pool_ptr,
    slots_out_ptr,
    hist_out_ptr,
    base_out_ptr,
    num_rows,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = offs < num_rows
    slot = tl.load(slots_ptr + offs, mask=in_range, other=-1).to(tl.int64)
    live = in_range & (slot > 0)
    hist = tl.load(hist_pool_ptr + slot, mask=live, other=0)
    base = tl.load(base_pool_ptr + slot, mask=live, other=0)
    tl.store(slots_out_ptr + offs, tl.where(live, slot, -1).to(tl.int32), in_range)
    tl.store(hist_out_ptr + offs, hist, in_range)
    tl.store(base_out_ptr + offs, base, in_range)


# by_request_row: liveness still comes from the state slot (padded rows: slot <= 0), the
# cursors and ring row from the request row.
@triton.jit
def _gather_request_rows_kernel(
    slots_ptr,
    rows_ptr,
    hist_pool_ptr,
    base_pool_ptr,
    slots_out_ptr,
    hist_out_ptr,
    base_out_ptr,
    ring_out_ptr,
    num_rows,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = offs < num_rows
    slot = tl.load(slots_ptr + offs, mask=in_range, other=-1).to(tl.int64)
    live = in_range & (slot > 0)
    row = tl.load(rows_ptr + offs, mask=live, other=0).to(tl.int64)
    hist = tl.load(hist_pool_ptr + row, mask=live, other=0)
    base = tl.load(base_pool_ptr + row, mask=live, other=0)
    tl.store(slots_out_ptr + offs, tl.where(live, slot, -1).to(tl.int32), in_range)
    tl.store(hist_out_ptr + offs, hist, in_range)
    tl.store(base_out_ptr + offs, base, in_range)
    tl.store(ring_out_ptr + offs, tl.where(live, row, 0).to(tl.int32), in_range)


@triton.jit
def _zero_padded_rows_kernel(
    slots_ptr,
    out_ptr,
    row_numel,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    if tl.load(slots_ptr + row) < 0:
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        tl.store(
            out_ptr + row.to(tl.int64) * row_numel + offs,
            tl.zeros([BLOCK], dtype=out_ptr.dtype.element_ty),
            offs < row_numel,
        )


# Stable partition of the verify's rows, folding rows (hist_len >= flush_min) first: the
# flush kernel's row_order (flush_first_row_order's permutation, in one launch).
@triton.jit
def _flush_first_order_kernel(
    hist_ptr,
    order_ptr,
    num_rows,
    flush_min,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    valid = offs < num_rows
    hist = tl.load(hist_ptr + offs, mask=valid, other=0)
    folds = (valid & (hist >= flush_min)).to(tl.int32)
    rest = (valid & (hist < flush_min)).to(tl.int32)
    num_folds = tl.sum(folds, axis=0)
    pos = tl.where(
        folds == 1,
        tl.cumsum(folds, axis=0) - 1,
        num_folds + tl.cumsum(rest, axis=0) - 1,
    )
    tl.store(order_ptr + pos, offs.to(tl.int32), mask=valid)


# Copies the packed [q | k | v] token rows of one strided view into three
# contiguous buffers in a single launch (the FlashInfer verify reads contiguous q/k/v).
@triton.jit
def _split_qkv_kernel(
    src_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    src_row_stride,
    Q_COLS: tl.constexpr,
    K_COLS: tl.constexpr,
    V_COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    total = Q_COLS + K_COLS + V_COLS
    x = tl.load(src_ptr + token * src_row_stride + cols, mask=cols < total)
    tl.store(q_ptr + token * Q_COLS + cols, x, mask=cols < Q_COLS)
    k_cols = cols - Q_COLS
    tl.store(k_ptr + token * K_COLS + k_cols, x, mask=(k_cols >= 0) & (k_cols < K_COLS))
    v_cols = cols - Q_COLS - K_COLS
    tl.store(v_ptr + token * V_COLS + v_cols, x, mask=(v_cols >= 0) & (v_cols < V_COLS))


# Refreshes every layer's fp32 gate-parameter copy from its bf16 source in one
# launch; the int64 tables hold the source and destination addresses.
@triton.jit
def _refresh_gate_params_kernel(
    src_ptrs,
    dst_ptrs,
    numel,
    BLOCK: tl.constexpr,
):
    entry = tl.program_id(0)
    src = tl.load(src_ptrs + entry).to(tl.pointer_type(tl.bfloat16))
    dst = tl.load(dst_ptrs + entry).to(tl.pointer_type(tl.float32))
    offs = tl.arange(0, BLOCK)
    mask = offs < numel
    tl.store(dst + offs, tl.load(src + offs, mask=mask).to(tl.float32), mask=mask)


def gather_gdn_ucache_rows(
    *,
    slots: torch.Tensor,
    buffers: GdnUcacheBuffers,
    request_rows: Optional[torch.Tensor] = None,
) -> GdnUcacheRows:
    if buffers.by_request_row:
        return _gather_gdn_ucache_request_rows(
            slots=slots, request_rows=request_rows, buffers=buffers
        )
    num_rows = slots.shape[0]
    rows = GdnUcacheRows(
        slots=torch.empty(num_rows, dtype=torch.int32, device=slots.device),
        hist_len=torch.empty(num_rows, dtype=torch.int32, device=slots.device),
        cache_base=torch.empty(num_rows, dtype=torch.int32, device=slots.device),
    )
    if num_rows > 0:
        _gather_rows_kernel[(triton.cdiv(num_rows, _CURSOR_BLOCK),)](
            slots,
            buffers.hist_len,
            buffers.cache_base,
            rows.slots,
            rows.hist_len,
            rows.cache_base,
            num_rows,
            BLOCK=_CURSOR_BLOCK,
        )
    return rows


def _gather_gdn_ucache_request_rows(
    *,
    slots: torch.Tensor,
    request_rows: Optional[torch.Tensor],
    buffers: GdnUcacheBuffers,
) -> GdnUcacheRows:
    num_rows = slots.shape[0]
    if request_rows is None or request_rows.shape[0] < num_rows:
        raise RuntimeError(
            "GDN u-cache rings by request row: the verify needs its request rows."
        )
    device = slots.device
    rows = GdnUcacheRows(
        slots=torch.empty(num_rows, dtype=torch.int32, device=device),
        hist_len=torch.empty(num_rows, dtype=torch.int32, device=device),
        cache_base=torch.empty(num_rows, dtype=torch.int32, device=device),
        ring_rows=torch.empty(num_rows, dtype=torch.int32, device=device),
    )
    if num_rows > 0:
        _gather_request_rows_kernel[(triton.cdiv(num_rows, _CURSOR_BLOCK),)](
            slots,
            request_rows,
            buffers.hist_len,
            buffers.cache_base,
            rows.slots,
            rows.hist_len,
            rows.cache_base,
            rows.ring_rows,
            num_rows,
            BLOCK=_CURSOR_BLOCK,
        )
    return rows


def _load_flush_kernel(device: torch.device) -> Callable:
    capability = torch.cuda.get_device_capability(device)
    if capability == (10, 7):
        # The vendored copy of flashinfer#6791 (gdn_ucache_rubin/__init__.py), not the
        # image's FlashInfer: 0.7.3 has no Rubin build, and a later one may differ.
        from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.linear.gdn_ucache_rubin.gdn_replay_mtp_umma_rubin import (
            gated_delta_rule_mtp_ucache_flush_umma_rubin,
        )

        return gated_delta_rule_mtp_ucache_flush_umma_rubin
    # SM103 shares SM100's tcgen05 path and runs the SM100 build.
    if capability not in ((10, 0), (10, 3)):
        raise RuntimeError(
            f"GDN u-cache verify needs SM100, SM103 or SM107, got SM{capability[0]}{capability[1]}"
        )
    try:
        from flashinfer.gdn_kernels.gdn_replay_mtp_umma import (
            gated_delta_rule_mtp_ucache_flush_umma,
        )
    except ImportError as e:
        raise RuntimeError(
            "the GDN u-cache verify needs FlashInfer >= 0.7.3 "
            "(flashinfer.gdn_kernels.gdn_replay_mtp_umma)."
        ) from e
    return gated_delta_rule_mtp_ucache_flush_umma


def gdn_ucache_fused_norm_applies(rows: int, device) -> bool:
    """SGLANG_OPT_GDN_UCACHE_FUSED_NORM: the flush epilogue reduces a 128-wide row as 16
    threads x 8 elements, which is the stock gated-norm launch's order once it holds >= 2 rows per
    program (verify B >= 3); smaller verifies keep the separate norm."""
    from sglang.kernels.ops.attention.fla.layernorm_gated import calc_rows_per_block

    return calc_rows_per_block(rows, device) >= 2


def _flush_first_order(*, hist_len: torch.Tensor, flush_min: int) -> torch.Tensor:
    num_rows = hist_len.shape[0]
    order = torch.empty(num_rows, dtype=torch.int32, device=hist_len.device)
    if num_rows > 0:
        _flush_first_order_kernel[(1,)](
            hist_len,
            order,
            num_rows,
            flush_min,
            BLOCK=max(16, triton.next_power_of_2(num_rows)),
        )
    return order


def _log_strided_inputs(*, q, k, v, a, b) -> None:
    from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.linear.gdn_ucache_rubin.gdn_replay_mtp_umma_rubin import (
        strided_inputs_layout,
    )

    strides = dict(q=q.stride(), v=v.stride(), a=a.stride())
    if strided_inputs_layout(q, k, v, a, b) is None:
        logger.warning(
            "SGLANG_OPT_GDN_UCACHE_STRIDED_QKV: the flush kernel copies q / k / v / a / b "
            "(views it cannot address in place: strides %s)",
            strides,
        )
    else:
        logger.info(
            "SGLANG_OPT_GDN_UCACHE_STRIDED_QKV: the flush kernel reads q / k / v / a / b "
            "in place (strides %s)",
            strides,
        )


def _assert_not_capturing(what: str) -> None:
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"GDN u-cache verify: {what} must be prepared before CUDA graph capture."
        )


def _fp32_view_or_buffer(param: torch.Tensor) -> torch.Tensor:
    if param.dtype == torch.float32 and param.is_contiguous():
        return param.detach()
    return torch.empty(param.shape, dtype=torch.float32, device=param.device)
