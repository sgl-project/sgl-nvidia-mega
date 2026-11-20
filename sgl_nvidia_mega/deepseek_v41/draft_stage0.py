# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Derived from TensorRT-LLM's DSpark stage-0 token table (NVIDIA Corporation, Apache-2.0):
# the table, its fill by the live kernels, the one-launch gather and the bitwise
# self-check, adapted to SGLang's DSpark draft and this plugin's draft attention. This file
# has been modified from that design; NOTICE names the origin.
"""The DSpark draft's stage-0 token table (slot draft_stage0_table).

A draft block's rows are its token ids (the bonus token, then the mask token), the first
stage's residual streams are each row's embedding replicated over the hc streams, and the
first sublayer has no carried pre-mix, so its input is the input RMSNorm of stream 0
(``deepseek_v4_mhc._combine``). Everything from there to the attention -- that RMSNorm, the
``[wq_a; wkv]`` GEMM, the q-LoRA RMSNorm (+ MXFP8 quant) and ``wq_b`` -- is a function of the
row's token id alone. In this draft the query's RoPE runs inside the swap-AB attention as it
loads q, and the window K/V's RMSNorm + RoPE + FP8 store in SGLang's window-store kernel,
so (unlike TensorRT-LLM's table, which tabulates the values right before an in-gather RoPE) the
table holds exactly what those consumers read: per token id, this rank's query heads as
``wq_b`` writes them (BF16, unroped) and the K/V row as the fused GEMM writes it (BF16,
before the store's norm). One launch gathers a step's rows; the store and the attention then
run on them unchanged. Row ``vocab`` holds the rows of a zero embedding (ids outside the
vocabulary).

The table is filled by running the live chain (``draft_attention.query_and_kv``) over every
token id, one request's rows a launch (the decode row count, so every kernel runs the
tactic it runs at decode), and before any step takes it a sample of tokens runs through the
live chain at each decode width and is compared bit for bit with the gather; a width with a
mismatch on any rank keeps the live chain.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Tokens the check runs through the live chain at each width (TensorRT-LLM's 64)
CHECK_TOKENS = 64


class Stage0Table:
    """Per-token stage-0 rows of this rank's draft attention (row ``vocab``: zero embedding).

    ``q`` ``[V + 1, H, D]`` BF16 (wq_b's rows, unroped) and ``kv`` ``[V + 1, D]`` BF16 (the
    window K/V before the store's norm + RoPE)."""

    def __init__(self, vocab: int, heads: int, head_dim: int, device):
        self.vocab = int(vocab)
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        if self.head_dim & (self.head_dim - 1):
            raise ValueError(f"head_dim {head_dim} is not a power of two")
        rows = self.vocab + 1
        self.q = torch.empty(
            (rows, self.heads, self.head_dim), dtype=torch.bfloat16, device=device
        )
        self.kv = torch.empty(
            (rows, self.head_dim), dtype=torch.bfloat16, device=device
        )
        self.filled = False
        # the decode widths (rows a step) the check proved; a step takes the table only there
        self.checked: tuple = ()

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.q, self.kv))


# ----------------------------------------------------------------------------------------------
# The decode-time gather.
# ----------------------------------------------------------------------------------------------


@triton.jit
def _stage0_gather_kernel(
    ids_ptr,
    q_tab_ptr,
    kv_tab_ptr,
    q_out_ptr,
    kv_out_ptr,
    vocab,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    WAIT: tl.constexpr,
):
    """One ``(row, head)``: the row's query head from the table; head index ``HEADS``: the
    row's K/V. Ids outside the vocabulary read row ``vocab`` (a zero embedding's rows)."""
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    if WAIT:
        # the ids are a predecessor's output (the draft block's token ids)
        tl.extra.cuda.gdc_wait()
        # the consumers wait for this grid's completion before any read: let them set up
        tl.extra.cuda.gdc_launch_dependents()
    raw = tl.load(ids_ptr + row).to(tl.int64)
    in_vocab = (raw >= 0) & (raw < vocab)
    token = tl.where(in_vocab, raw, vocab)
    cols = tl.arange(0, DIM)
    if head < HEADS:
        src = (token * HEADS + head) * DIM
        dst = (row * HEADS + head) * DIM
        tl.store(q_out_ptr + dst + cols, tl.load(q_tab_ptr + src + cols))
    else:
        tl.store(
            kv_out_ptr + row * DIM + cols, tl.load(kv_tab_ptr + token * DIM + cols)
        )


def gather(
    table: Stage0Table, ids: torch.Tensor, *, use_pdl: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """The stage-0 attention's ``(q [rows, H, D], kv [rows, D])`` of the step's token ids
    ``ids`` ``[rows]`` (any integer type), in one launch."""
    rows = ids.shape[0]
    q = torch.empty(
        (rows, table.heads, table.head_dim), dtype=torch.bfloat16, device=ids.device
    )
    kv = torch.empty((rows, table.head_dim), dtype=torch.bfloat16, device=ids.device)
    _stage0_gather_kernel[(rows, table.heads + 1)](
        ids.contiguous(),
        table.q,
        table.kv,
        q,
        kv,
        table.vocab,
        HEADS=table.heads,
        DIM=table.head_dim,
        WAIT=use_pdl,
        num_warps=2,
        **({"launch_pdl": True} if use_pdl else {}),
    )
    return q, kv


# ----------------------------------------------------------------------------------------------
# The fill and the check. ``rows_fn(ids [n]) -> (q [n, H, D], kv [n, D])`` is the live chain
# from the token ids (embedding, the stage's input norm, its query and K/V kernels).
# ----------------------------------------------------------------------------------------------

RowsFn = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def fill(table: Stage0Table, rows_fn: RowsFn, width: int) -> float:
    """Fill ``table`` from the live chain over every token id (eager, outside graph
    capture), ``width`` rows a launch (the rows of one request's draft block). Ids past the
    vocabulary run as zero embeddings; rows past ``vocab`` are dropped. Returns seconds."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "the DSpark stage-0 table is filled outside CUDA-graph capture"
        )
    device = table.q.device
    total = table.vocab + 1
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for start in range(0, total, width):
        n = min(width, total - start)
        ids = torch.arange(start, start + width, dtype=torch.int64, device=device)
        q, kv = rows_fn(ids)
        table.q[start : start + n].copy_(q[:n].view(n, table.heads, table.head_dim))
        table.kv[start : start + n].copy_(kv[:n])
    torch.cuda.synchronize()
    table.filled = True
    return time.perf_counter() - t0


def _sample(table: Stage0Table, extra: Optional[torch.Tensor], num: int, seed: int):
    """The check's tokens: the zero row, the first and last ids, the step's own ids
    (bonus and mask tokens), then random ids."""
    fixed = [table.vocab, 0, table.vocab - 1]
    if extra is not None:
        fixed += [int(t) for t in extra.flatten().tolist() if 0 <= int(t) < table.vocab]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    rand = torch.randint(0, table.vocab, (max(0, num - len(fixed)),), generator=gen)
    return (fixed + rand.tolist())[: max(num, len(fixed))]


def compare(table: Stage0Table, rows_fn: RowsFn, ids: torch.Tensor) -> dict:
    """Mismatching elements of the gather vs the live chain for one step's ids."""
    live_q, live_kv = rows_fn(ids)
    got_q, got_kv = gather(table, ids)
    live_q = live_q.reshape(got_q.shape).view(torch.int16)
    return {
        "q": int((live_q != got_q.view(torch.int16)).sum()),
        "kv": int((live_kv.view(torch.int16) != got_kv.view(torch.int16)).sum()),
    }


def check(
    table: Stage0Table,
    rows_fn: RowsFn,
    widths,
    extra: Optional[torch.Tensor] = None,
    num_tokens: int = CHECK_TOKENS,
    seed: int = 0,
) -> dict:
    """Run the sample through the live chain and the gather at each width in ``widths``;
    returns ``{width: mismatching elements}`` (this rank only)."""
    tokens = _sample(table, extra, num_tokens, seed)
    device = table.q.device
    result = {}
    for w in widths:
        bad = 0
        for i in range(0, len(tokens), w):
            part = tokens[i : i + w]
            part = part + tokens[: w - len(part)]
            ids = torch.tensor(part, dtype=torch.int64, device=device)
            bad += sum(compare(table, rows_fn, ids).values())
        result[w] = bad
    return result
