# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Weight forms of the small-M NVFP4 MoE engine (torch only, no kernels).

The engine reads the routed FC1 weights in place, from the engine's own processed
TRT-LLM NVFP4 tensors (``w13`` uint8 [E, 512, 3072], ``w13`` scales [E, 512 x 384]
bytes), so it relies on how that processing permutes rows:

* FC1: processed row ``128 j + 32 q + i`` (j, q in 0..3, i in 0..31) holds the gate row
  (if ``i & 8``) or the up row (otherwise) of intermediate index ``h = 64 j + 16 q + 2
  (i & 7) + (i >> 4)``;
* FC2: processed row ``32 b + p`` of ``w2`` [E, 6144, 128] holds source row ``32 b + 4
  (p & 7) + (p >> 3)``;
* block scales follow their rows and are stored in the 128 x 4 interleave: the byte of
  (row, column) sits at ``((row // 128) * (C // 4) + column // 4) * 512 + (row % 32) *
  16 + ((row % 128) // 32) * 4 + column % 4`` for C scale columns.

``layout_self_test`` runs the engine's own processing function on a synthetic layer and
refuses the single-copy form unless both maps hold byte for byte. FC2 is read from
compact per-quarter records (``fc2_records``), rebuilt from the processed tensors with
the inverse maps; the shared expert from ``shared_pack``."""

from __future__ import annotations

import torch

from .constants import HIDDEN, INTER

FC1_ROWS = 2 * INTER  # 512 (gate + up)
FC1_BYTES = HIDDEN // 2  # 3072 packed e2m1 bytes per FC1 row
FC1_SF = HIDDEN // 16  # 384 scales per FC1 row
FC2_BYTES = INTER // 2  # 128 packed bytes per FC2 row
FC2_SF = INTER // 16  # 16 scales per FC2 row
SLOT = 18432  # one record slot: 16 KiB of weights + 2 KiB of scales
FC2_SLOTS = 3
FC2_RECORD = FC2_SLOTS * SLOT  # 55,296 bytes per (expert, quarter, CTA)


class LayoutMismatch(RuntimeError):
    """The processed weights do not have the layout the engine reads."""


def fc1_sources(device=None) -> torch.Tensor:
    """[512] source row of each processed FC1 row, indexing cat([gate, up]) (gate
    0..255)."""
    r = torch.arange(FC1_ROWS)
    i = r % 32
    h = 64 * (r // 128) + 16 * ((r // 32) % 4) + 2 * (i & 7) + (i >> 4)
    return torch.where((i & 8) != 0, h, INTER + h).to(device)


def fc2_sources(device=None) -> torch.Tensor:
    """[6144] source row of each processed FC2 row."""
    r = torch.arange(HIDDEN)
    p = r % 32
    return ((r // 32) * 32 + 4 * (p & 7) + (p >> 3)).to(device)


def interleave_offsets(rows: int, cols: int, device=None) -> torch.Tensor:
    """[rows, cols] byte offset of each scale in the 128 x 4 interleave."""
    r = torch.arange(rows).view(rows, 1)
    c = torch.arange(cols).view(1, cols)
    off = (
        ((r // 128) * (cols // 4) + c // 4) * 512
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + c % 4
    )
    return off.to(device)


def interleave(sf: torch.Tensor) -> torch.Tensor:
    """[..., rows, cols] uint8 scales -> [..., rows * cols] interleaved bytes."""
    rows, cols = sf.shape[-2:]
    off = interleave_offsets(rows, cols, sf.device).flatten()
    out = torch.empty(*sf.shape[:-2], rows * cols, dtype=torch.uint8, device=sf.device)
    out[..., off] = sf.reshape(*sf.shape[:-2], rows * cols)
    return out


def deinterleave(flat: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Inverse of ``interleave``: [..., rows * cols] bytes -> [..., rows, cols]."""
    off = interleave_offsets(rows, cols, flat.device).flatten()
    return flat[..., off].reshape(*flat.shape[:-1], rows, cols)


def _u8(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.uint8 else t.view(torch.uint8)


def expected_processed(gate, up, gate_sf, up_sf, w2, w2_sf):
    """The processed tensors the engine reads, from source-layout inputs (uint8):
    gate / up [E, 256, 3072], their scales [E, 256, 384], w2 [E, 6144, 128], w2_sf
    [E, 6144, 16]. Returns (w13 [E, 512, 3072], w13_sf [E, 512 * 384], w2, w2_sf
    [E, 6144 * 16])."""
    src1 = fc1_sources(gate.device)
    src2 = fc2_sources(w2.device)
    w13 = torch.cat([gate, up], dim=1)[:, src1]
    s13 = interleave(torch.cat([gate_sf, up_sf], dim=1)[:, src1])
    return w13, s13, w2[:, src2], interleave(w2_sf[:, src2])


def layout_self_test(process, device, experts: int = 3, seed: int = 1000) -> int:
    """Run ``process(gate, up, gate_sf, up_sf, w2, w2_sf) -> (w13, w13_sf, w2, w2_sf)``
    (the engine's own stock processing, from its loader layout) on a synthetic layer of
    random bytes and compare every byte with the layout the engine reads. Returns the
    number of bytes compared; raises ``LayoutMismatch`` on any difference."""
    g = torch.Generator(device="cpu").manual_seed(seed)

    def rand(*shape):
        return torch.randint(0, 256, shape, generator=g, dtype=torch.uint8).to(device)

    src = (
        rand(experts, INTER, FC1_BYTES),
        rand(experts, INTER, FC1_BYTES),
        rand(experts, INTER, FC1_SF),
        rand(experts, INTER, FC1_SF),
        rand(experts, HIDDEN, FC2_BYTES),
        rand(experts, HIDDEN, FC2_SF),
    )
    want = expected_processed(*src)
    got = process(*[t.clone() for t in src])
    names = ("w13", "w13 scales", "w2", "w2 scales")
    count = 0
    for name, w, t in zip(names, want, got):
        t = _u8(t).reshape(experts, -1)
        w = w.reshape(experts, -1)
        if t.shape != w.shape or not torch.equal(t, w):
            raise LayoutMismatch(
                f"processed {name} does not have the layout the fused MoE reads "
                f"(shape {tuple(t.shape)} vs {tuple(w.shape)})"
            )
        count += w.numel()
    if count == 0:
        raise LayoutMismatch("layout self-test compared no bytes")
    return count


# ---------------------------------------------------------------- FC2 records
def fc2_records(w2: torch.Tensor, w2_sf: torch.Tensor, chunk: int = 32) -> torch.Tensor:
    """Compact FC2 records [E][4 j][4 c][3 x 18,432 B] from one layer's processed w2
    (uint8 [E, 6144, 128]) and its processed scales ([E, 6144 * 16] bytes), built
    ``chunk`` experts at a time. Record (e, j, c), slot s, tile t (row block
    ``1536 c + 512 s + 128 t``): 4 KiB of weights in the engine's K-major image
    [16 row groups][2 K halves][8 rows][16 B] for K bytes ``32 j .. 32 j + 31``, then
    512 B of scales [32 l][4 q][4 kb]."""
    experts = w2.shape[0]
    w2 = _u8(w2).reshape(experts, HIDDEN, FC2_BYTES)
    sf = _u8(w2_sf).reshape(experts, HIDDEN * FC2_SF)
    inv2 = torch.argsort(fc2_sources(w2.device))
    out = torch.empty(
        experts, 4, 4, FC2_SLOTS * SLOT, dtype=torch.uint8, device=w2.device
    )
    for e0 in range(0, experts, chunk):
        ne = min(chunk, experts - e0)
        raw = w2[e0 : e0 + ne][
            :, inv2
        ]  # [ne, 6144 = c*1536 + t*128 + rg*8 + rr, 128 = j*32 + kc*16 + b]
        a2 = (
            raw.view(ne, 4, 12, 16, 8, 4, 2, 16)
            .permute(0, 5, 1, 2, 3, 6, 4, 7)
            .reshape(ne, 4, 4, 3, 4 * 4096)
        )
        rs = deinterleave(sf[e0 : e0 + ne], HIDDEN, FC2_SF)[
            :, inv2
        ]  # [ne, 6144 = c*1536 + t*128 + q*32 + l, 16]
        s2 = (
            rs.view(ne, 4, 12, 4, 32, 4, 4)
            .permute(0, 5, 1, 2, 4, 3, 6)
            .reshape(ne, 4, 4, 3, 4 * 512)
        )
        out[e0 : e0 + ne] = torch.cat([a2, s2], dim=-1).reshape(
            ne, 4, 4, FC2_SLOTS * SLOT
        )
    return out


# ---------------------------------------------------------------- shared expert pack
SHARED_CHUNK = 32768
SHARED_RECORD = 18 * SHARED_CHUNK  # 589,824 bytes per (quarter, CTA)


def shared_rows(device=None) -> torch.Tensor:
    """[4 quarters x 128] row of gate_up [512, 6144] (gate 0..255, up 256..511) for each
    local row: quarter sq, local row 32 q + i is gate row 64 sq + 16 q + i (i < 16) or
    up row 64 sq + 16 q + i - 16."""
    sq = torch.arange(4).view(4, 1, 1)
    q = torch.arange(4).view(1, 4, 1)
    i = torch.arange(32).view(1, 1, 32)
    h = 64 * sq + 16 * q + (i % 16)
    return torch.where(i < 16, h, INTER + h).reshape(-1).to(device)


def shared_pack(gate_up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    """BF16 shared expert (gate_up [512, 6144], gate rows first; down [6144, 256]) -> [4
    quarters][4 CTAs][589,824 B]: FC1 as 24 boxes of 128 rows x 64 K (no-swizzle K-major
    images [16][8 K groups][8 rows][16 B]) over K range ``1536 c .. +1536``, then FC2 as
    12 tiles of 128 output rows x 64 K (``1536 c + 128 t + r``, K = h in the quarter).
    """
    if gate_up.dtype != torch.bfloat16 or tuple(gate_up.shape) != (FC1_ROWS, HIDDEN):
        raise ValueError(f"gate_up must be bf16 [{FC1_ROWS}, {HIDDEN}]")
    if down.dtype != torch.bfloat16 or tuple(down.shape) != (HIDDEN, INTER):
        raise ValueError(f"down must be bf16 [{HIDDEN}, {INTER}]")
    w = (
        gate_up[shared_rows(gate_up.device)].contiguous().view(torch.uint8)
    )  # [512, 12288]
    a = (
        w.view(4, 16, 8, 4, 24, 8, 16)
        .permute(0, 3, 4, 1, 5, 2, 6)
        .reshape(4, 4, 24 * 16384)
    )
    d = down.contiguous().view(
        torch.uint8
    )  # [6144 = c*1536 + t*128 + rg*8 + rr, 512 = sq*128 + kc*16 + b]
    f2 = (
        d.view(4, 12, 16, 8, 4, 8, 16)
        .permute(4, 0, 1, 2, 5, 3, 6)
        .reshape(4, 4, 12 * 16384)
    )
    return torch.cat([a, f2], dim=-1).contiguous()
