# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/layers/attention/qsa/fused_kv.py (_qsa_fused_kv_prepare) with its MTP
# lookup (_mtp_lookup_positions) and python/sglang/srt/layers/attention/qsa/mtp_lookup.py
# (DeferredMtpSparseIndices). Changed: the launcher takes only a draft decode's lookup
# (token indices, compress ratio 1); the kernel is unchanged.
"""An MTP draft decode step's K/V pack with the draft extend's selection looked up in
the kernel.

SGLang's draft decode reuses the selection its draft extend captured per request
(``QSAMTPSharedSparseIndices``): ``lookup`` gathers the request's row, appends the
positions drafted since the capture (-1 past the query) and the pack reads that row,
about nine torch launches a layer and step. Here the pack reads the captured row and
computes the tail itself, so the K/V written and packed and the valid counts are
bit-identical to the gather-then-pack chain.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class MtpLookup:
    """One QSA layer's captured selection, read per request by the pack."""

    indices: torch.Tensor  # [request slots + 1, T] int32: the top-k, then the tail
    captured_len: torch.Tensor  # [request slots + 1] int32: the length at capture
    requests: torch.Tensor  # [rows] the rows' request slots
    positions: torch.Tensor  # [rows] the rows' logical query positions
    tail_width: int

    @property
    def shape(self):
        return (self.requests.numel(), self.indices.shape[1])

    def materialize(self) -> torch.Tensor:
        """``QSAMTPSharedSparseIndices.lookup``: the rows the kernel reads (tests)."""
        rows = self.requests.to(torch.long)
        out = self.indices[rows]
        base = self.captured_len[rows].to(torch.int64)
        tail = base.unsqueeze(1) + torch.arange(self.tail_width, device=out.device)
        valid = tail <= self.positions.to(torch.int64).unsqueeze(1)
        out[:, -self.tail_width :] = torch.where(valid, tail, -1).to(out.dtype)
        return out


@triton.jit
def _mtp_lookup_positions(
    IDX, REQ, CAPTURED_LEN, QPOS, row, c, T: tl.constexpr, TAIL: tl.constexpr
):
    req = tl.load(REQ + row).to(tl.int64)
    pos = tl.load(IDX + req * T + c, c < T - TAIL, -1)
    base = tl.load(CAPTURED_LEN + req).to(tl.int64)
    tail = base + c - (T - TAIL)
    qpos = tl.load(QPOS + row).to(tl.int64)
    return tl.where(c < T - TAIL, pos, tl.where((c < T) & (tail <= qpos), tail, -1)).to(
        tl.int32
    )


@triton.jit
def _qsa_fused_kv_prepare(
    K,
    V,
    NK,
    NV,
    LOC,
    MAP,
    REQ,
    IDX,
    LENGTH,
    QPOS,
    COUNTS,
    OK,
    OV,
    LOOKUP_REQ,
    CAPTURED_LEN,
    LOOKUP_QPOS,
    H: tl.constexpr,
    D: tl.constexpr,
    MAP_STRIDE: tl.constexpr,
    NK_STRIDE: tl.constexpr,
    NV_STRIDE: tl.constexpr,
    T: tl.constexpr,
    S: tl.constexpr,
    W: tl.constexpr,
    B: tl.constexpr,
    BT: tl.constexpr,
    R: tl.constexpr,
    G: tl.constexpr,
    EXPAND: tl.constexpr,
    RATIO: tl.constexpr,
    IB: tl.constexpr,
    CHAIN: tl.constexpr,
    PAGE: tl.constexpr,
    LOOKUP: tl.constexpr,
    TAIL: tl.constexpr,
):
    tiles = tl.cdiv(S, B) + 2
    pid = tl.program_id(0)
    row = (pid // (G * tiles * H)) * G + pid % G
    tile = (pid // G) % tiles
    head = (pid // (G * tiles)) % H
    if row >= R:
        return
    req = tl.load(REQ + row)
    row_active = tl.load(LOC + row) != 0
    d = tl.arange(0, D)
    if tile < tl.cdiv(S, B):
        c = tile * B + tl.arange(0, B)
        if EXPAND:
            n = tl.minimum((tl.load(LENGTH + row) // RATIO), IB) * RATIO
            qpos = tl.load(QPOS + row)
            tail_start = (qpos + 1) // RATIO * RATIO
            if PAGE > 0:
                # Same count as the COUNTS tile below; the paged consumer never
                # reads past the page that holds it.
                tail_count = tl.maximum(
                    0, tl.minimum(qpos + 1, tl.load(LENGTH + row)) - tail_start
                )
                if tile * B >= (n + tail_count + PAGE - 1) // PAGE * PAGE:
                    return
            blocks = tl.load(IDX + row * IB + c // RATIO, c < IB * RATIO, -1)
            expanded = blocks * RATIO + c % RATIO
            tail_off = c - n
            tail = tail_start + tail_off
            pos = tl.where(
                (c < n) & (blocks >= 0),
                expanded,
                tl.where(
                    (tail_off >= 0) & (tail_off < RATIO - 1) & (tail < qpos + 1),
                    tail,
                    -1,
                ),
            )
        else:
            if LOOKUP:
                pos = _mtp_lookup_positions(
                    IDX, LOOKUP_REQ, CAPTURED_LEN, LOOKUP_QPOS, row, c, T, TAIL
                )
            else:
                pos = tl.load(IDX + row * T + c, c < T, -1)
        length = tl.load(LENGTH + row)
        valid = (pos >= 0) & (pos < length) & (c < T)
        valid = valid & row_active
        slot = tl.load(MAP + req.to(tl.int64) * MAP_STRIDE + pos, valid, 0).to(tl.int64)
        if EXPAND and CHAIN and W > 1:
            group_row = row // W * W
            first_pos = length - 1 - row % W
            relative_pos = pos - first_pos
            fresh = valid & (relative_pos >= 0)
            input_row = tl.where(fresh, group_row + relative_pos, 0)
        else:
            input_row = tl.full((B,), 0, tl.int32)
            fresh = tl.full((B,), False, tl.int1)
            for j in tl.static_range(W):
                candidate_row = row // W * W + j
                candidate_slot = tl.load(LOC + candidate_row)
                matches = valid & (candidate_slot != 0) & (slot == candidate_slot)
                input_row = tl.where(matches, candidate_row, input_row)
                fresh = fresh | matches
        oldoff = slot[:, None] * H * D + head * D + d[None, :]
        nkoff = input_row.to(tl.int64)[:, None] * NK_STRIDE + head * D + d[None, :]
        nvoff = input_row.to(tl.int64)[:, None] * NV_STRIDE + head * D + d[None, :]
        kp = tl.where(fresh[:, None], NK + nkoff, K + oldoff)
        vp = tl.where(fresh[:, None], NV + nvoff, V + oldoff)
        k = tl.load(kp, valid[:, None], 0.0)
        v = tl.load(vp, valid[:, None], 0.0)
        outoff = (row.to(tl.int64) * S + c[:, None]) * H * D + head * D + d[None, :]
        tl.store(OK + outoff, k, (c < S)[:, None])
        tl.store(OV + outoff, v, (c < S)[:, None])
    elif tile == tl.cdiv(S, B):
        slot = tl.load(LOC + row).to(tl.int64)
        k = tl.load(NK + row.to(tl.int64) * NK_STRIDE + head * D + d).to(
            K.dtype.element_ty
        )
        v = tl.load(NV + row.to(tl.int64) * NV_STRIDE + head * D + d).to(
            V.dtype.element_ty
        )
        tl.store(K + slot * H * D + head * D + d, k, slot != 0)
        tl.store(V + slot * H * D + head * D + d, v, slot != 0)
    elif head == 0:
        if EXPAND:
            n = tl.minimum((tl.load(LENGTH + row) // RATIO), IB) * RATIO
            qpos = tl.load(QPOS + row)
            tail_start = (qpos + 1) // RATIO * RATIO
            length = tl.load(LENGTH + row)
            tail_count = tl.maximum(0, tl.minimum(qpos + 1, length) - tail_start)
            tl.store(COUNTS + row, n + tail_count)
        else:
            c = tl.arange(0, BT)
            if LOOKUP:
                pos = _mtp_lookup_positions(
                    IDX, LOOKUP_REQ, CAPTURED_LEN, LOOKUP_QPOS, row, c, T, TAIL
                )
            else:
                pos = tl.load(IDX + row * T + c, c < T, -1)
            length = tl.load(LENGTH + row)
            tl.store(COUNTS + row, tl.sum(((pos >= 0) & (pos < length)).to(tl.int32)))


def fused_kv_lookup(
    k_cache,
    v_cache,
    k,
    v,
    loc,
    req_to_token,
    row_requests,
    lookup: MtpLookup,
    sequence_lengths,
    counts,
    packed_k,
    packed_v,
):
    """SGLang's ``fused_kv_prepare(..., width=1, compress_ratio=1)`` of a draft decode
    step on ``lookup``'s rows: writes the new K/V at ``loc`` (slot 0 skipped), packs
    each row's selected K/V at a row stride of ``packed_k.shape[0] // rows`` and its
    valid count into ``counts``."""
    rows, topk = lookup.shape
    if rows == 0:
        return
    heads, dim = k_cache.shape[1:]
    stride = packed_k.shape[0] // rows
    block = 16
    _qsa_fused_kv_prepare[(rows * heads * (triton.cdiv(stride, block) + 2),)](
        k_cache,
        v_cache,
        k,
        v,
        loc,
        req_to_token,
        row_requests,
        lookup.indices,
        sequence_lengths,
        None,
        counts,
        packed_k,
        packed_v,
        lookup.requests,
        lookup.captured_len,
        lookup.positions,
        heads,
        dim,
        req_to_token.stride(0),
        k.stride(0),
        v.stride(0),
        topk,
        stride,
        1,
        block,
        triton.next_power_of_2(topk),
        rows,
        1,
        False,
        1,
        topk,
        True,
        0,
        True,
        lookup.tail_width,
        num_warps=4,
    )
