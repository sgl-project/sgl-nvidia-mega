# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Host-side plan of the own extend-attention op: the engage window and the balanced work split.

Pure functions of a request's lengths, the head layout and the device's SM count (a runtime query by the caller); no
torch, no device access. The split is ops.prefill_attn.plan_sk's (equal contiguous key-tile ranges of the request's
work line, one per block); the window declines requests whose stock kernel already fills whole waves of blocks.
"""

from __future__ import annotations

from dataclasses import dataclass

PAGE = 128
TOKENS_PER_BLOCK = 32
STOCK_Q_TILE = 256
# Window (tunable; bench/bench_xs0.py): decline when the stock kernel's own block fill is at least FILL_HI (nothing left
# to recover); serve new-token counts in [T_LO, T_HI] over cached prefixes of at least C_LO keys. T_LO set from sweeps of
# 1..1,025 new tokens over 65,536..200,000 cached keys (bench/bench_xs0.py).
FILL_HI = 0.97
T_LO, T_HI = 1, 16384
C_LO = 65536
MIN_TILES = 64


@dataclass(frozen=True)
class ExtendCall:
    """What the op needs from the engine, passed by the caller (host values only)."""

    new_lens: tuple  # new tokens per request, in call order
    prefix_lens: tuple  # cached keys per request
    layer_id: int
    mode: str  # "extend" for a plain prefill extend; anything else declines
    capturing: bool  # CUDA-graph capture in progress


def stock_fill(new_tokens: int, num_q_heads: int, num_sms: int) -> float:
    """Fraction of block slots the stock context kernel keeps busy: its grid is ceil(T / 256) query tiles x heads."""
    blocks = -(-new_tokens // STOCK_Q_TILE) * num_q_heads
    waves = -(-blocks // num_sms)
    return blocks / (waves * num_sms)


def decline_reason(call: ExtendCall, num_q_heads: int, num_sms: int) -> str | None:
    """None when every request of the call is inside the window, else the first reason (counted by the caller)."""
    if call.capturing:
        return "capturing"
    if call.mode != "extend":
        return "mode"
    if not call.new_lens or len(call.new_lens) != len(call.prefix_lens):
        return "lens"
    for t, c in zip(call.new_lens, call.prefix_lens):
        if t < T_LO or t > T_HI:
            return "new_tokens"
        if c < C_LO:
            return "prefix"
        if stock_fill(t, num_q_heads, num_sms) >= FILL_HI:
            return "stock_fills_waves"
    return None


def split(
    new_lens, prefix_lens, num_kv_heads: int, num_sms: int, min_tiles: int = MIN_TILES
):
    """Balanced split (plan_sk), vectorized: returns (units [n, 8] int32 array, cta_offsets list, merges [m, 8] int32 array,
    slots). The tables are exactly split_reference's (tests/test_extend_attn_b_plan.py checks equality on a sweep)."""
    import numpy as np

    ncta = max(1, num_sms)
    parts, mparts, slot = [], [], 0
    for r, (q_len, prefix) in enumerate(zip(new_lens, prefix_lens)):
        if q_len <= 0:
            continue
        nt = -(-q_len // TOKENS_PER_BLOCK)
        tok0 = np.arange(nt, dtype=np.int64) * TOKENS_PER_BLOCK
        per_tok = (
            prefix + np.minimum(tok0 + TOKENS_PER_BLOCK, q_len) + PAGE - 1
        ) // PAGE
        tiles = np.tile(per_tok, num_kv_heads)  # blocks in (kv head, first token) order
        heads = np.repeat(np.arange(num_kv_heads, dtype=np.int64), nt)
        toks = np.tile(tok0, num_kv_heads)
        total = int(tiles.sum())
        nblk = min(ncta, max(1, total // max(1, min_tiles)))
        bounds = (np.arange(nblk + 1, dtype=np.int64) * total) // nblk
        end = np.cumsum(tiles)
        start = end - tiles
        k0 = np.searchsorted(bounds, start, side="right") - 1
        k1 = np.searchsorted(bounds, end, side="left") - 1
        npc = k1 - k0 + 1
        blk = np.repeat(np.arange(len(tiles)), npc)
        first = np.repeat(np.cumsum(npc) - npc, npc)
        pn = np.arange(int(npc.sum()), dtype=np.int64) - first
        kk = k0[blk] + pn
        a = np.maximum(start[blk], bounds[kk]) - start[blk]
        e = np.minimum(end[blk], bounds[kk + 1]) - start[blk]
        cut = npc > 1
        mcount = np.where(cut, npc, 0)
        base = slot + np.cumsum(mcount) - mcount
        sl = np.where(cut[blk], base[blk] + pn, -1)
        z = np.zeros_like(kk)
        parts.append(
            (
                kk,
                np.stack(
                    [np.full_like(kk, r), heads[blk], toks[blk], a, e, sl, z, z], 1
                ),
            )
        )
        if cut.any():
            mz = np.zeros(int(cut.sum()), dtype=np.int64)
            mparts.append(
                np.stack(
                    [
                        np.full_like(mz, r),
                        heads[cut],
                        toks[cut],
                        base[cut],
                        npc[cut],
                        mz,
                        mz,
                        mz,
                    ],
                    1,
                )
            )
        slot += int(mcount.sum())
    if not parts:
        return np.zeros((0, 8), np.int32), [], np.zeros((0, 8), np.int32), 0
    kk = np.concatenate([p[0] for p in parts])
    rows = np.concatenate([p[1] for p in parts])
    order = np.argsort(
        kk, kind="stable"
    )  # block-major; request / block / piece order kept inside each block
    counts = np.bincount(kk, minlength=ncta)
    counts = counts[counts > 0]
    offs = [0] + np.cumsum(counts).tolist()
    merges = np.concatenate(mparts) if mparts else np.zeros((0, 8), np.int64)
    return rows[order].astype(np.int32), offs, merges.astype(np.int32), slot


def split_reference(
    new_lens, prefix_lens, num_kv_heads: int, num_sms: int, min_tiles: int = MIN_TILES
):
    """Balanced split (plan_sk), the plain Python form: returns (units, cta_offsets, merges, slots) as Python lists.

    units rows: [request, kv head, first query token, tile begin, tile end, slot (-1: writes the output)];
    cta_offsets: unit offsets per block (len = blocks + 1); merges rows: [request, kv head, first token, first slot, count].
    """
    per_cta = [[] for _ in range(max(1, num_sms))]
    merges, slot = [], 0
    for r, (q_len, prefix) in enumerate(zip(new_lens, prefix_lens)):
        if q_len <= 0:
            continue
        blocks = []
        for h in range(num_kv_heads):
            for tok0 in range(0, q_len, TOKENS_PER_BLOCK):
                blocks.append(
                    (
                        h,
                        tok0,
                        (prefix + min(tok0 + TOKENS_PER_BLOCK, q_len) + PAGE - 1)
                        // PAGE,
                    )
                )
        total = sum(b[2] for b in blocks)
        nblk = min(len(per_cta), max(1, total // max(1, min_tiles)))
        bounds = [(i * total) // nblk for i in range(nblk + 1)]
        pos, i = 0, 0
        for h, tok0, tiles in blocks:
            start, end = pos, pos + tiles
            while bounds[i + 1] <= start:
                i += 1
            pieces = []
            k = i
            while k < nblk and bounds[k] < end:
                a, b = max(start, bounds[k]), min(end, bounds[k + 1])
                if b > a:
                    pieces.append((k, a - start, b - start))
                k += 1
            if len(pieces) == 1:
                per_cta[pieces[0][0]].append([r, h, tok0, 0, tiles, -1, 0, 0])
            else:
                merges.append([r, h, tok0, slot, len(pieces), 0, 0, 0])
                for n, (k, a, b) in enumerate(pieces):
                    per_cta[k].append([r, h, tok0, a, b, slot + n, 0, 0])
                slot += len(pieces)
            pos = end
    units, offs = [], [0]
    for lst in per_cta:
        if lst:
            units.extend(lst)
            offs.append(len(units))
    return units, (offs if units else []), merges, slot
