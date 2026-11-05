# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Dense multicast push of one rank's MXFP8 shard into every rank's gathered operand (csrc/mc_push_dense.cuh): rank p's rows
land at global rows [p x rows, (p + 1) x rows) of a DENSE [world x rows, K] operand, and its scales are written straight into that
operand's 128 x 4 swizzled layout (one 4-byte word per row and column tile; no two ranks write the same word), so one GEMM over
world x rows rows gives the final output with no compaction. A variant of ``ops.mc_push`` (which places each block at whole
128-row tiles); the ordering contract is the same (stream order before the launch, ``ops.a2a_pull.barrier`` after it, two slots by
layer parity). The caller zeroes the gathered buffer once at allocation (the scale words of the global layout's padding rows are
never stored; the GEMM reads them for its rows >= world x rows only).

``push(src_q, src_sf, dst_q_mc, dst_sf_mc, rows, rank, world, k, q_cap, sf_cap)``: one launch; ``dst_q_mc`` / ``dst_sf_mc`` = the
multicast addresses of the gathered slot's value byte 0 (global row 0) and scale byte 0; ``q_cap`` / ``sf_cap`` = the bytes of
those two regions. The kernel places this rank's rows at rank x rows; every rank must pass the same ``rows``.

``word_map`` / ``check_words``: the host mirror of the kernel's scale-word map, for units (the per-rank write sets are disjoint
and cover every word of the dense operand's rows).

Launch shape: ``MUSE_VR_FP8_PREFILL_AGF_UNROLL`` / ``MUSE_VR_FP8_PREFILL_AGF_CTAS_PER_SM`` (``ops.mc_push``'s knobs and defaults).
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

from . import mc_push

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("load_jit", "make_cpp_args"),
}


_CUH = Path(__file__).resolve().parents[1] / "csrc" / "mc_push_dense.cuh"
TILE_ROWS = 128
TILE_BYTES = 512


@functools.lru_cache(maxsize=None)
def module(unroll: int):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(int(unroll))
    return load_jit(
        "muse_vr_mc_push_dense",
        *args,
        cuda_files=[str(_CUH)],
        cuda_wrappers=[("push", f"muse_vr::mcpd::McPushDenseKernel<{args}>::run")],
    )


@functools.lru_cache(maxsize=None)
def _sms(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def word_byte(m: int, t: int, k: int) -> int:
    """Byte offset of the scale word of row ``m`` and column tile ``t`` (scale columns 4t .. 4t + 3) in the 128 x 4 swizzled
    layout of an operand with rows of ``k`` elements (the kernel's formula)."""
    kt = k // 128
    return ((m >> 7) * kt + t) * TILE_BYTES + ((m & 31) * 4 + ((m & 127) >> 5)) * 4


def word_map(rows: int, rank: int, k: int) -> list[tuple[int, int]]:
    """[(local byte, global byte)] for every scale word the kernel stores for ``rank`` with ``rows`` rows per rank, in the
    kernel's order (local memory order, padding rows skipped)."""
    kt = k // 128
    rows_pad = -(-rows // TILE_ROWS) * TILE_ROWS
    out = []
    for w in range(rows_pad * kt):
        lt, wi = w >> 7, w & 127
        rt, t = divmod(lt, kt)
        r = rt * TILE_ROWS + (wi & 3) * 32 + (wi >> 2)
        if r < rows:
            out.append((w * 4, word_byte(rank * rows + r, t, k)))
    return out


def check_words(rows: int, world: int, k: int) -> int:
    """Raise unless (a) every local word stored is the word of a local row < ``rows`` at the kernel's local byte, (b) the
    global words stored by the ``world`` ranks are pairwise disjoint and (c) they cover every word of the rows < world x rows
    exactly once. Returns the number of words checked."""
    kt = k // 128
    seen: dict[int, int] = {}
    for p in range(world):
        wm = word_map(rows, p, k)
        if len(wm) != rows * kt:
            raise AssertionError(
                f"rank {p}: {len(wm)} words stored, expected {rows * kt}"
            )
        for lb, gb in wm:
            if lb % 4 or gb % 4:
                raise AssertionError(f"rank {p}: unaligned word ({lb}, {gb})")
            if gb in seen:
                raise AssertionError(
                    f"global word at byte {gb} stored by ranks {seen[gb]} and {p}"
                )
            seen[gb] = p
    expected = {word_byte(m, t, k) for m in range(world * rows) for t in range(kt)}
    if set(seen) != expected:
        raise AssertionError(
            f"coverage: {len(expected - set(seen))} words of the dense rows not stored, "
            f"{len(set(seen) - expected)} stored outside them"
        )
    for p in range(world):
        for lb, gb in word_map(rows, p, k):
            m_t = _inverse(gb, k)
            if (
                m_t is None
                or m_t[0] // rows != p
                or word_byte(m_t[0] - p * rows, m_t[1], k) != lb
            ):
                raise AssertionError(
                    f"rank {p}: global byte {gb} does not hold the word read at local byte {lb}"
                )
    return len(seen)


def _inverse(byte: int, k: int):
    """(row, column tile) of the scale word at ``byte`` (the inverse of ``word_byte``)."""
    kt = k // 128
    tile, off = divmod(byte, TILE_BYTES)
    if off % 4:
        return None
    rt, t = divmod(tile, kt)
    wi = off // 4
    return rt * TILE_ROWS + (wi & 3) * 32 + (wi >> 2), t


def push(
    src_q: torch.Tensor,
    src_sf: torch.Tensor,
    dst_q_mc: int,
    dst_sf_mc: int,
    rows: int,
    rank: int,
    world: int,
    k: int,
    q_cap: int,
    sf_cap: int,
    unroll: int | None = None,
    ctas_per_sm: int | None = None,
) -> None:
    """Store this rank's shard densely through the multicast addresses (see the module doc)."""
    du, dc = mc_push.config()
    u = du if unroll is None else int(unroll)
    c = dc if ctas_per_sm is None else int(ctas_per_sm)
    dev = src_q.device
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    module(u).push(
        src_q,
        src_sf,
        int(dst_q_mc),
        int(dst_sf_mc),
        int(rows),
        int(rank),
        int(world),
        int(k),
        int(q_cap),
        int(sf_cap),
        _sms(idx) * c,
    )
