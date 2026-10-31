# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Weight packers for the attention kernels: pure functions of the checkpoint tensors.

Each packer returns a new contiguous tensor (a derived copy); the input is not
modified. Shapes are those of one tensor-parallel rank (8 attention heads).
"""

import torch

HEADS = 8
LATENT = 512  # kv_lora_rank
NOPE = 192  # qk_nope_head_dim
VALUE_DIM = 256  # v_head_dim
KV_B_ROWS_PER_HEAD = NOPE + VALUE_DIM


def pack_uv_proj(kv_b_weight: torch.Tensor, values_per_cta: int = 16) -> torch.Tensor:
    """W_UV tiles for ``attn.mla_uv_proj``.

    ``kv_b_weight``: ``kv_b_proj.weight`` of one rank, ``[8 * 448, 512]`` BF16 (rows
    ``[448 h + 192, 448 h + 448)`` are head ``h``'s value rows, each a contiguous
    512-element latent row). Result: flat BF16, ordered
    ``[head][value tile j][latent step s][value row r][32 latent]``, so the tile of
    CTA ``(h, j)`` is one contiguous ``values_per_cta x 1 KiB`` block.
    """
    if kv_b_weight.dtype != torch.bfloat16:
        raise ValueError(f"kv_b_proj weight must be bf16, got {kv_b_weight.dtype}")
    if tuple(kv_b_weight.shape) != (HEADS * KV_B_ROWS_PER_HEAD, LATENT):
        raise ValueError(
            f"kv_b_proj weight must be [{HEADS * KV_B_ROWS_PER_HEAD}, {LATENT}], "
            f"got {tuple(kv_b_weight.shape)}"
        )
    if values_per_cta != 16:
        raise ValueError("values_per_cta must be 16")
    w = kv_b_weight.view(HEADS, KV_B_ROWS_PER_HEAD, LATENT)[:, NOPE:, :]
    w = w.reshape(HEADS, VALUE_DIM // values_per_cta, values_per_cta, LATENT // 32, 32)
    return w.permute(0, 1, 3, 2, 4).contiguous().view(-1)


INDEXER_HEADS = 32
INDEXER_HEAD_DIM = 128
Q_LORA = 2048


def pack_indexer_q(wq_b: torch.Tensor) -> torch.Tensor:
    """Indexer query weight tiles for ``attn.indexer_q_proj``.

    ``wq_b``: the indexer ``wq_b.weight``, ``[32 * 128, 2048]`` BF16. Result: flat BF16
    ordered ``[head][part 4][latent step 64][row 32][32 latent]``: CTA ``(head, part)``
    reads one contiguous 128 KiB block.
    """
    if wq_b.dtype != torch.bfloat16 or tuple(wq_b.shape) != (
        INDEXER_HEADS * INDEXER_HEAD_DIM,
        Q_LORA,
    ):
        raise ValueError(
            "indexer wq_b must be bf16 "
            f"[{INDEXER_HEADS * INDEXER_HEAD_DIM}, {Q_LORA}], "
            f"got {wq_b.dtype} {tuple(wq_b.shape)}"
        )
    v = wq_b.contiguous().view(INDEXER_HEADS, 4, 32, Q_LORA // 32, 32)
    return v.permute(0, 1, 3, 2, 4).contiguous().view(-1)


QKV_A_K = 6144


def pack_qkv_a(w: torch.Tensor, grid: int) -> torch.Tensor:
    """Row-owned blocks for ``attn.qkv_a_proj_small_m`` launched on ``grid`` CTAs.

    ``w``: ``[N, 6144]`` BF16 (the projections' rows stacked: main rows first, then the
    auxiliary rows), N even. CTA ``c`` owns rows ``[2 floor(N/2 c / grid), 2 floor(N/2
    (c + 1) / grid))``; its block is ``[k-step 192][row][32]`` and the blocks are
    concatenated in CTA order. The result depends on ``grid``: pack with the grid the
    kernel is launched with.
    """
    n, k = w.shape
    if k != QKV_A_K or n % 2 or w.dtype != torch.bfloat16:
        raise ValueError(
            f"weight must be bf16 [even N, {QKV_A_K}], got {w.dtype} {tuple(w.shape)}"
        )
    groups = n // 2
    blocks = []
    for c in range(grid):
        r0, r1 = 2 * (groups * c // grid), 2 * (groups * (c + 1) // grid)
        blocks.append(
            w[r0:r1]
            .reshape(r1 - r0, k // 32, 32)
            .permute(1, 0, 2)
            .contiguous()
            .view(-1)
        )
    return torch.cat(blocks)


Q_HEAD = NOPE + 64  # q_b rows per head: nope 192 | rope 64


def _pack_rows_k(w: torch.Tensor) -> torch.Tensor:
    """``[N, K]`` BF16 -> 1 KiB blocks per (16-row tile, 32-wide k-step) in ``mma.sync``
    lane order: rows
    ``16 tile + 8 half + g``, elements ``32 step + 8 t + e`` at
    ``[tile][step][half][g][t][e]``."""
    n, k = w.shape
    if n % 16 or k % 32 or w.dtype != torch.bfloat16:
        raise ValueError(f"expected bf16 [16 i, 32 j], got {w.dtype} {tuple(w.shape)}")
    v = w.contiguous().view(n // 16, 2, 8, k // 32, 4, 8)
    return v.permute(0, 3, 1, 2, 4, 5).contiguous().view(-1)


def pack_q_b(w_qb: torch.Tensor) -> torch.Tensor:
    """q_b tiles for ``attn.mla_prologue``: ``q_b_proj.weight`` of one rank, ``[8 * 256,
    2048]`` BF16."""
    if tuple(w_qb.shape) != (HEADS * Q_HEAD, 2048):
        raise ValueError(
            f"q_b_proj weight must be [{HEADS * Q_HEAD}, 2048], got {tuple(w_qb.shape)}"
        )
    return _pack_rows_k(w_qb)


def pack_w_uk_t(kv_b_weight: torch.Tensor) -> torch.Tensor:
    """W_UK^T tiles for ``attn.mla_prologue``: from ``kv_b_proj.weight`` of one rank
    (``[8 * 448, 512]`` BF16, the
    first 192 rows of each head are W_UK), per head ``[512 (latent), 192 (nope)]``
    packed, heads concatenated."""
    if kv_b_weight.dtype != torch.bfloat16 or tuple(kv_b_weight.shape) != (
        HEADS * KV_B_ROWS_PER_HEAD,
        LATENT,
    ):
        raise ValueError(
            f"kv_b_proj weight must be bf16 [{HEADS * KV_B_ROWS_PER_HEAD}, {LATENT}]"
        )
    wt = (
        kv_b_weight.view(HEADS, KV_B_ROWS_PER_HEAD, LATENT)[:, :NOPE, :]
        .transpose(1, 2)
        .contiguous()
    )
    return torch.cat([_pack_rows_k(wt[h]) for h in range(HEADS)])
