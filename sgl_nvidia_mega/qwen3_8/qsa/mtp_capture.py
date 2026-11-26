# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/kernels/ops/speculative/mtp_capture_metadata.py
# (_mtp_capture_metadata_batched, try_capture_metadata_batched). Changed: the
# one-request variant and its PDL switch are dropped; the kernel is unchanged.
"""An MTP draft extend's seed capture of one QSA layer in one launch, at any batch.

SGLang's ``_capture_mtp_sparse_indices_from_extend_lens`` stores each request's last
accepted row of the draft extend's selection, and the sequence length it was selected
at, as the seed its draft decode steps look up: about a dozen torch launches a layer
(cumsum, the anchor arithmetic, where, index_select, two index_copy). This writes the
same rows and lengths from the same inputs with integer loads and stores only, so the
seed is bit-identical.
"""

import torch
import triton
import triton.language as tl

MAX_REQUESTS = 256


@triton.jit
def _mtp_capture_metadata_batched(
    TOPK,
    EXTEND,
    ACCEPT,
    REQ,
    SEQLENS,
    T2B,
    SEED,
    CAPTURED,
    BS,
    ROWS,
    BS_P2: tl.constexpr,
    COLS: tl.constexpr,
    TOPK_STRIDE: tl.constexpr,
    SEED_STRIDE: tl.constexpr,
    FRONT: tl.constexpr,
    TRASH: tl.constexpr,
    HAS_ACCEPT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program per (request, column block). Integer-only, the torch expression of
    # _capture_mtp_sparse_indices_from_extend_lens: block_end = cumsum(extend) - 1 (int64),
    # anchor = block_end - (extend - front - accept) when accept counts exist, clamped to
    # [0, ROWS - 1]; row = req if extend > 0 and req > 0 else the trash row; the seed row
    # takes topk[anchor, :COLS] and captured_len takes int32(seq_lens)[t2b[anchor]].
    b = tl.program_id(0)
    offs = tl.arange(0, BS_P2)
    extend_all = tl.load(EXTEND + offs, offs < BS, other=0).to(tl.int64)
    block_end = tl.sum(tl.where(offs <= b, extend_all, 0), 0) - 1
    extend = tl.load(EXTEND + b).to(tl.int64)
    if HAS_ACCEPT:
        accept = tl.load(ACCEPT + b).to(tl.int64)
        anchor = block_end - (extend - FRONT - accept)
    else:
        anchor = block_end
    anchor = tl.minimum(tl.maximum(anchor, 0), ROWS - 1)
    req = tl.load(REQ + b).to(tl.int64)
    row = tl.where((extend > 0) & (req > 0), req, TRASH)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(TOPK + anchor * TOPK_STRIDE + col, col < COLS, 0)
    tl.store(SEED + row * SEED_STRIDE + col, values, col < COLS)
    if tl.program_id(1) == 0:
        seq = tl.load(T2B + anchor).to(tl.int64)
        length = tl.load(SEQLENS + seq).to(tl.int32)
        tl.store(CAPTURED + row, length)


def reason(
    topk, extend, accept, req, seq_lens, token_to_batch, seed, captured, trash
) -> str:
    """'' when ``capture`` takes these inputs (host checks only), else why not."""
    bs = req.numel()
    if topk.ndim != 2 or topk.shape[0] == 0:
        return f"top-k {tuple(topk.shape)}"
    if topk.dtype != torch.int32 or topk.stride(1) != 1:
        return f"top-k {topk.dtype} with column stride {topk.stride(1)}"
    if not 0 < bs <= MAX_REQUESTS:
        return f"{bs} requests, the kernel takes 1..{MAX_REQUESTS}"
    checked = [
        ("extend lengths", extend, bs),
        ("request slots", req, None),
        ("sequence lengths", seq_lens, None),
        ("row requests", token_to_batch, None),
    ]
    if accept is not None:
        checked.append(("accept counts", accept, bs))
    for name, t, n in checked:
        if (
            t.dtype not in (torch.int32, torch.int64)
            or not t.is_contiguous()
            or (n is not None and t.numel() != n)
        ):
            return f"{name}: {t.dtype}, {tuple(t.shape)}"
    if token_to_batch.numel() < topk.shape[0]:
        return "fewer row requests than top-k rows"
    if not (
        seed.dtype == captured.dtype == torch.int32
        and seed.ndim == 2
        and seed.stride(1) == 1
        and captured.is_contiguous()
        and topk.shape[1] <= seed.shape[1]
        and int(trash) < min(seed.shape[0], captured.numel())
    ):
        return "the seed state's layout"
    return ""


def capture(
    topk, extend, accept, front, req, seq_lens, token_to_batch, seed, captured, trash
) -> None:
    """The seed capture of one layer; ``accept`` None for an extend without accept
    counts (its rows end at the seed). ``reason`` must be '' for these inputs."""
    bs = req.numel()
    block = 256
    _mtp_capture_metadata_batched[(bs, triton.cdiv(topk.shape[1], block))](
        topk,
        extend,
        accept if accept is not None else extend,
        req,
        seq_lens,
        token_to_batch,
        seed,
        captured,
        bs,
        topk.shape[0],
        BS_P2=triton.next_power_of_2(bs),
        COLS=topk.shape[1],
        TOPK_STRIDE=topk.stride(0),
        SEED_STRIDE=seed.stride(0),
        FRONT=int(front),
        TRASH=int(trash),
        HAS_ACCEPT=accept is not None,
        BLOCK=block,
        num_warps=4,
    )
