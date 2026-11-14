# SPDX-License-Identifier: Apache-2.0
# From SGLang python/sglang/kernels/ops/mamba/gdn_prefill_state_glue.py; unchanged. The log lines name the
# source's switches, which the plugin's Options.fused_state_copy and
# Options.fused_track replace.
"""Eager GDN prefill state copies in one launch per family.

* ``gdn_gather_state_rows`` / ``gdn_scatter_state_rows``: the chunked kernel's
  gather and write-back of its pool rows, with ``state_pool_io``'s padding rule
  (a slot <= 0 reads zeros and writes a zero row to the sink slot 0).
* ``gdn_track_conv_snapshot``: ``conv_pool[dst[r]] = x[tok[r]].transpose(0, 1)``,
  the prefix-cache track window taken before the conv.
* ``gdn_track_state_rows``: ``pool[h_dst] = h[h_src]`` and
  ``pool[f_dst] = pool[f_src]``, the track copies after the chunked kernel.
  Destinations are track slots, never a live source slot.

Each moves the same bytes as the torch indexing it replaces.
"""

import logging
from typing import Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_BLOCK_CONV = 1024
# 16 KB of fp16 state per program; a 48 x 128 x 128 row is 96 programs.
_BLOCK_ROW = 8192
_logged: set = set()


def _log_once(key: str, msg: str, *args) -> None:
    if key not in _logged:
        _logged.add(key)
        logger.info(msg, *args)


@triton.jit
def _gdn_pool_rows_kernel(
    src,
    dst,
    slots,  # [rows] pool slots; <= 0 is padding
    row_elems,
    stride_src,
    stride_dst,
    GATHER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < row_elems
    slot = tl.load(slots + r).to(tl.int64)
    # A padded slot reads nothing (zeros) and writes to the sink slot 0.
    live = offs < tl.where(slot > 0, row_elems, 0)
    slot = tl.maximum(slot, 0)
    if GATHER:
        val = tl.load(src + slot * stride_src + offs, mask=live, other=0)
        tl.store(dst + r.to(tl.int64) * stride_dst + offs, val, mask=mask)
    else:
        val = tl.load(src + r.to(tl.int64) * stride_src + offs, mask=live, other=0)
        tl.store(
            dst + slot * stride_dst + offs, val.to(dst.dtype.element_ty), mask=mask
        )


@triton.jit
def _gdn_track_conv_snapshot_kernel(
    conv_pool,  # [slots, dim, STATE_LEN]
    x,  # [tokens, dim] raw pre-conv input, unit column stride
    tok,  # [rows, STATE_LEN] token rows
    dst,  # [rows] track slots
    stride_pool_slot,
    stride_pool_dim,
    stride_pool_pos,
    stride_x_token,
    stride_tok_row,
    stride_tok_pos,
    dim,
    STATE_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < dim
    slot = tl.load(dst + r).to(tl.int64)
    out = conv_pool + slot * stride_pool_slot + offs.to(tl.int64) * stride_pool_dim
    for j in tl.static_range(STATE_LEN):
        t = tl.load(tok + r * stride_tok_row + j * stride_tok_pos).to(tl.int64)
        val = tl.load(x + t * stride_x_token + offs, mask=mask)
        tl.store(out + j * stride_pool_pos, val, mask=mask)


@triton.jit
def _gdn_track_state_rows_kernel(
    pool,  # [slots, row_elems] (row view)
    h,  # [checkpoints, row_elems] (row view); pool when there are no h rows
    h_src,
    h_dst,
    f_src,
    f_dst,
    n_h,
    row_elems,
    stride_pool_slot,
    stride_h_row,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < row_elems
    if r < n_h:
        src = h + tl.load(h_src + r).to(tl.int64) * stride_h_row
        slot = tl.load(h_dst + r).to(tl.int64)
    else:
        src = pool + tl.load(f_src + (r - n_h)).to(tl.int64) * stride_pool_slot
        slot = tl.load(f_dst + (r - n_h)).to(tl.int64)
    val = tl.load(src + offs, mask=mask)
    tl.store(pool + slot * stride_pool_slot + offs, val, mask=mask)


def _row_major_tail(t: torch.Tensor) -> bool:
    """Every dim after the first is dense (one row = one contiguous span)."""
    expected = 1
    for size, stride in zip(reversed(t.shape[1:]), reversed(t.stride()[1:])):
        if size != 1 and stride != expected:
            return False
        expected *= size
    return True


def _pool_rows(src, dst, slots, gather: bool) -> None:
    rows, row_elems = slots.shape[0], (src if gather else dst)[0].numel()
    if rows:
        _gdn_pool_rows_kernel[(rows, triton.cdiv(row_elems, _BLOCK_ROW))](
            src,
            dst,
            slots,
            row_elems,
            src.stride(0),
            dst.stride(0),
            GATHER=gather,
            BLOCK=_BLOCK_ROW,
        )


def gdn_state_rows_supported(pool: torch.Tensor, slots: torch.Tensor) -> bool:
    """Whether the one-launch gather / write-back serves this pool and slot list."""
    if _row_major_tail(pool) and slots.dim() == 1 and slots.is_contiguous():
        return True
    _log_once(
        "rows-miss",
        "SGLANG_OPT_GDN_PREFILL_FUSED_STATE_COPY: stock copies (pool rows not dense "
        "or slots not a contiguous vector)",
    )
    return False


def gdn_gather_state_rows(pool: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """``gather_pool_states(pool, slots)[2]`` (zeros for slots <= 0) in one launch."""
    out = torch.empty(
        (slots.shape[0], *pool.shape[1:]), dtype=pool.dtype, device=pool.device
    )
    _pool_rows(pool, out, slots, gather=True)
    _log_once(
        "gather",
        "SGLANG_OPT_GDN_PREFILL_FUSED_STATE_COPY: chunked-GDN state rows gathered and "
        "written back by one launch each (first: %d rows of %d elements)",
        slots.shape[0],
        pool[0].numel(),
    )
    return out


def gdn_scatter_state_rows(
    pool: torch.Tensor, slots: torch.Tensor, rows: torch.Tensor
) -> None:
    """``scatter_pool_states`` in one launch: ``pool[slot] = rows[i]`` in the pool's
    dtype, a zero row to slot 0 for slots <= 0 (rows: contiguous)."""
    _pool_rows(rows, pool, slots, gather=False)


def gdn_track_conv_snapshot(
    *,
    conv_pool: torch.Tensor,
    x: torch.Tensor,
    token_rows: torch.Tensor,
    dst_slots: torch.Tensor,
) -> bool:
    """One launch for ``conv_pool[dst_slots] = x[token_rows].transpose(1, 2)``.
    Returns False (nothing written) when the layout is not the one handled."""
    rows = dst_slots.shape[0]
    reason = None
    if conv_pool.dim() != 3 or x.dim() != 2 or token_rows.dim() != 2:
        reason = "ranks"
    elif conv_pool.dtype != x.dtype:
        reason = f"dtypes {conv_pool.dtype} / {x.dtype}"
    elif x.stride(1) != 1 or token_rows.shape != (rows, conv_pool.shape[2]):
        reason = "layout"
    elif x.shape[1] != conv_pool.shape[1]:
        reason = "dims"
    if reason is not None:
        _log_once(
            "conv-miss",
            "SGLANG_OPT_GDN_PREFILL_FUSED_TRACK: conv window kept on torch (%s)",
            reason,
        )
        return False
    if rows == 0:
        return True
    dim = x.shape[1]
    _gdn_track_conv_snapshot_kernel[(rows, triton.cdiv(dim, _BLOCK_CONV))](
        conv_pool,
        x,
        token_rows,
        dst_slots,
        conv_pool.stride(0),
        conv_pool.stride(1),
        conv_pool.stride(2),
        x.stride(0),
        token_rows.stride(0),
        token_rows.stride(1),
        dim,
        STATE_LEN=conv_pool.shape[2],
        BLOCK=_BLOCK_CONV,
    )
    _log_once(
        "conv",
        "SGLANG_OPT_GDN_PREFILL_FUSED_TRACK: conv windows of the track slots in one "
        "launch (first: %d rows x %d channels)",
        rows,
        dim,
    )
    return True


def gdn_track_state_rows(
    *,
    pool: torch.Tensor,
    h: Optional[torch.Tensor],
    h_src: torch.Tensor,
    h_dst: torch.Tensor,
    f_src: torch.Tensor,
    f_dst: torch.Tensor,
) -> bool:
    """One launch for ``pool[h_dst] = h[h_src]`` and ``pool[f_dst] = pool[f_src]``
    (h: [checkpoints, *pool.shape[1:]], the pool's dtype). Returns False (nothing
    written) when the layout is not the one handled."""
    n_h, n_f = h_src.shape[0], f_src.shape[0]
    reason = None
    if not _row_major_tail(pool):
        reason = "pool rows not dense"
    elif n_h and h is None:
        reason = "no checkpoints"
    elif n_h and (
        h.dtype != pool.dtype
        or tuple(h.shape[1:]) != tuple(pool.shape[1:])
        or not _row_major_tail(h)
    ):
        reason = "checkpoint layout"
    if reason is not None:
        _log_once(
            "state-miss",
            "SGLANG_OPT_GDN_PREFILL_FUSED_TRACK: state rows kept on torch (%s)",
            reason,
        )
        return False
    if n_h + n_f == 0:
        return True
    row_elems = pool[0].numel()
    src_h = h if n_h else pool
    _gdn_track_state_rows_kernel[(n_h + n_f, triton.cdiv(row_elems, _BLOCK_ROW))](
        pool,
        src_h,
        h_src if n_h else f_src,
        h_dst if n_h else f_dst,
        f_src if n_f else h_src,
        f_dst if n_f else h_dst,
        n_h,
        row_elems,
        pool.stride(0),
        src_h.stride(0),
        BLOCK=_BLOCK_ROW,
    )
    _log_once(
        "state",
        "SGLANG_OPT_GDN_PREFILL_FUSED_TRACK: track states in one launch "
        "(first: %d checkpoint rows + %d final rows of %d elements)",
        n_h,
        n_f,
        row_elems,
    )
    return True
