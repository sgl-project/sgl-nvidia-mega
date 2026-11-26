# SPDX-License-Identifier: Apache-2.0
"""QSA prefill index scorer on tcgen05 (PREFILL_SELECT variant bit 4).

The caller's ``SGLANG_QSA_PREFILL_SCORER_TC=1`` runs ``prefill_select``'s
request-relative scorer as a Triton kernel whose K-tile x Q product is a ``tl.dot`` (tcgen05 on
SM100-family GPUs) instead of TileLang's ``mma.sync``. The epilogue is the
stock one, written out: ReLU, the 4-head sum as stock's reduce forms it
((0 + h0) + h1) + ((0 + h2) + h3) with flush-to-zero adds (the stock kernel is
built with fast math), and ``mul.rn`` by fp32(1/sqrt(128)). Rows are walked in
stock's 32-row blocks from the block's first key, and only stock's cells are
written (cut at its 64-key tile end).

Whether the tensor-core sum of each 128-wide dot product matches ``mma.sync``
bit for bit is the hardware's, so the variant is opt-in.
"""

import functools
import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Keys per tile, pipeline stages, warps, CTAs per SM: the source defaults (VR200
# microbench).
_BLOCK_N = 128
_STAGES = 2
_WARPS = 4
_CTAS_PER_SM = 2
_BLOCK_Q = 32  # stock's rows per block (128 // 4 index heads)
_STOCK_TILE = 64  # stock's key tile: the written coverage unit


@triton.jit
def _add_ftz(a, b):
    return tl.inline_asm_elementwise(
        "add.ftz.f32 $0, $1, $2;",
        "=r,r,r",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _relu_ftz(a):
    return tl.inline_asm_elementwise(
        "max.ftz.f32 $0, $1, 0f00000000;",
        "=r,r",
        [a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _scale(a):
    # mqa.py's qsa_mul_inv_sqrt128: mul.rn by fp32(1/sqrt(128)).
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, 0f3DB504F3;",
        "=r,r",
        [a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit(do_not_specialize=["keys", "width"])
def _qsa_prefill_scorer_tc(
    q,
    k,
    logits,
    starts,
    ends,
    keys,
    width,
    stride_l,
    BLOCK_N: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    STOCK_TILE: tl.constexpr,
    STAGES: tl.constexpr,
):
    row_base = tl.program_id(0) * BLOCK_Q
    splits = tl.num_programs(1)
    qi = tl.arange(0, BLOCK_Q)
    row_start = tl.load(starts + row_base + qi)
    row_end = tl.load(ends + row_base + qi)
    start_min = tl.min(tl.minimum(row_start, keys), axis=0)
    end_max = tl.max(tl.minimum(row_end, keys), axis=0)
    limit = start_min + tl.cdiv(end_max - start_min, STOCK_TILE) * STOCK_TILE
    num_tiles = tl.cdiv(end_max - start_min, BLOCK_N)
    tiles_per_split = tl.cdiv(num_tiles, splits)
    tile_begin = tl.program_id(1) * tiles_per_split
    tile_count = tl.maximum(tl.minimum(num_tiles - tile_begin, tiles_per_split), 0)

    qc = tl.arange(0, BLOCK_Q * HEADS)
    d = tl.arange(0, DIM)
    q_tile = tl.load(
        q + (row_base * HEADS + qc).to(tl.int64)[:, None] * DIM + d[None, :]
    )
    n = tl.arange(0, BLOCK_N)
    out_rows = logits + (row_base + qi).to(tl.int64)[None, :] * stride_l
    zero = tl.zeros((BLOCK_N, BLOCK_Q, 2), tl.float32)
    for ti in tl.range(0, tile_count, num_stages=STAGES):
        key = start_min + (tile_begin + ti) * BLOCK_N + n
        k_tile = tl.load(
            k + key.to(tl.int64)[:, None] * DIM + d[None, :],
            mask=(key < keys)[:, None],
            other=0.0,
        )
        s = _relu_ftz(tl.dot(k_tile, tl.trans(q_tile)))
        # Columns are row * HEADS + head; heads (0, 1) and (2, 3) pair first.
        a, b = tl.split(tl.reshape(s, (BLOCK_N, BLOCK_Q, 2, 2)))
        lo, hi = tl.split(_add_ftz(_add_ftz(zero, a), b))
        val = _scale(_add_ftz(lo, hi))
        col = key[:, None] - row_start[None, :]
        mask = (key < limit)[:, None] & (col >= 0) & (col < width)
        tl.store(out_rows + col, val, mask=mask)


def applies(q_rows: torch.Tensor, keys_2d: torch.Tensor) -> bool:
    return (
        q_rows.is_cuda
        and torch.cuda.get_device_capability(q_rows.device)[0] == 10
        and q_rows.dtype == torch.bfloat16
        and keys_2d.dtype == torch.bfloat16
        and q_rows.dim() == 3
        and tuple(q_rows.shape[1:]) == (4, 128)
        and q_rows.shape[0] % _BLOCK_Q == 0
        and q_rows.is_contiguous()
        and keys_2d.is_contiguous()
    )


@functools.lru_cache(maxsize=None)
def _log_once() -> None:
    logger.info(
        "QSA prefill scorer tc: on (block_n=%d stages=%d warps=%d ctas_per_sm=%d) "
        "(PREFILL_SELECT variant, SGLANG_QSA_PREFILL_SCORER_TC)",
        _BLOCK_N,
        _STAGES,
        _WARPS,
        _CTAS_PER_SM,
    )


def score(q_rows, keys_2d, logits, starts, ends, width, num_sms):
    """``prefill_select.score_chunk``'s output into ``logits``; returns the key split."""
    _log_once()
    row_blocks = q_rows.shape[0] // _BLOCK_Q
    max_useful = max(1, -(-width // _BLOCK_N))
    splits = max(
        1, min(max_useful, _CTAS_PER_SM * num_sms // max(row_blocks, 1), 65535)
    )
    _qsa_prefill_scorer_tc[(row_blocks, splits)](
        q_rows,
        keys_2d,
        logits,
        starts,
        ends,
        keys_2d.shape[0],
        width,
        logits.stride(0),
        BLOCK_N=_BLOCK_N,
        BLOCK_Q=_BLOCK_Q,
        HEADS=4,
        DIM=128,
        STOCK_TILE=_STOCK_TILE,
        STAGES=_STAGES,
        num_warps=_WARPS,
    )
    return splits
