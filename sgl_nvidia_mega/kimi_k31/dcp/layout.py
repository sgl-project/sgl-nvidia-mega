# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DCP index math and mailbox geometry for the K3.1 MK (pure Python; CPU-testable).

The layout is stock SGLang's (``layers/dcp/layout.py``,
``layers/attention/dsa/dsa_dcp_sharded.py``, ``layers/attention/dsa_backend.py``):

- DCP size ``N`` equals the TP size in the MK (the DCP group is the TP group). A request's MLA KV and DSA index-K
  are sharded by token: global position ``p`` lives on rank ``p % N`` at local position ``p // N``.
- ``req_to_token`` holds *widened* slots: one allocator page spans 64 rows on each of the N ranks
  (``span = 64 * N``). Global position ``p`` maps to ``w = page_table[p // span] * span + p % span``. Owner rule:
  ``w % N == rank``; the local row is ``w // N`` (same physical page id on every rank).
- The local length of a sequence of global length ``L`` on rank ``r`` is ``(L + N - 1 - r) // N``.
- Rank ``r`` owns query heads ``[r * H, (r + 1) * H)`` (H = 96 / TP). Under DCP decode every rank attends all
  ``N * H`` heads (padded to 128 for trtllm-gen, which rejects 64 < heads < 128) over its local KV; the owner of a
  head merges the N partials by LSE (base 2 for trtllm-gen).

Mailboxes (one receive region per rank in torch symmetric memory; every record is a single-copy-atomic 64-bit
``{payload32 (low), tag32 (high)}``; no sentinel, no re-arm, no fence (``uc_epoch``):

- ``q``:    [2 parity][N src][cap rows][H * 576 / 4 words]           the FP8 query bytes of the source's H heads
- ``cand``: [2 parity][N src][cap rows][topk][2]                       (score bits, global position) per candidate
- ``part``: [2 parity][N src][cap rows][H heads][258]                  512 BF16 (256 words) + LSE + pad per head

Parity is the MLA-layer ordinal's parity; the tag is unique per (forward epoch, MLA ordinal, exchange kind), so a
record of another layer or forward never equals the expected tag (equality compare; tag 0 is never valid, the
regions are zero-initialized).
"""

from __future__ import annotations

from dataclasses import dataclass

PAGE = 64
TOPK = 2048
HEAD_BYTES = 576  # FP8 latent 512 + rope 64
LATENT = 512
PADDED_HEADS = 128
TOTAL_HEADS = 96
EXCHANGES = 3  # q, cand, part
KIND_Q, KIND_CAND, KIND_PART = 0, 1, 2


def heads_per_rank(tp: int) -> int:
    if TOTAL_HEADS % tp:
        raise ValueError(f"96 MLA heads do not split over TP{tp}")
    return TOTAL_HEADS // tp


def padded_heads(dcp: int, heads: int) -> int:
    """trtllm-gen sparse MLA head count for ``dcp * heads`` gathered heads (stock ``_trtllm_padded_heads``)."""
    total = dcp * heads
    return PADDED_HEADS if 64 < total < 128 else total


def local_length(length: int, dcp: int, rank: int) -> int:
    """Keys of a global length on one rank (stock ``get_dcp_lens``, interleave 1)."""
    if length <= 0:
        return 0
    return (length + dcp - 1 - rank) // dcp


def owner(slot: int, dcp: int) -> int:
    return slot % dcp


def local_write_slot(slot: int, dcp: int, rank: int) -> int:
    """Widened write slot -> this rank's physical row, or -1 when another rank owns it (or ``slot < 0``)."""
    if slot < 0 or slot % dcp != rank:
        return -1
    return slot // dcp


def widened_slot(page_row, position: int, dcp: int) -> int:
    """Global position of a request -> its widened slot (stock ``dcp_global_topk`` docstring)."""
    span = PAGE * dcp
    return int(page_row[position // span]) * span + position % span


def global_position(local: int, dcp: int, rank: int) -> int:
    """Local key index -> global position (stock candidate epilogue: ``world``/``rank``)."""
    return local * dcp + rank


def localize_compact(slots, dcp: int, rank: int):
    """Global top-k slots of one row -> (this rank's local rows in order then -1, count). Stock
    ``_localize_compact_kernel``: owned = x >= 0 and x % N == rank; local = x // N."""
    owned = [int(x) // dcp for x in slots if int(x) >= 0 and int(x) % dcp == rank]
    return owned + [-1] * (len(slots) - len(owned)), len(owned)


def tag(epoch: int, ordinal: int, kind: int, layers: int = 24) -> int:
    """The 32-bit tag a record of (forward ``epoch``, MLA ``ordinal``, exchange ``kind``) carries; never 0. Must
    equal ``k31_dcp::tag`` in comm.cuh."""
    value = ((epoch * layers + ordinal) * EXCHANGES + kind + 1) & 0xFFFFFFFF
    return value or 1


@dataclass(frozen=True)
class Geometry:
    """Receive-region geometry of one rank (records are 8 bytes)."""

    dcp: int
    heads: int  # query heads per rank
    cap: int  # row capacity (decode requests, or verify tokens)
    topk: int = TOPK

    @property
    def q_words(self) -> int:  # 32-bit payload words of one source row
        return self.heads * HEAD_BYTES // 4

    @property
    def part_words(self) -> int:  # per (row, head): 256 BF16-pair words + LSE + pad (16-byte aligned blocks)
        return LATENT // 2 + 2

    def q_records(self) -> int:
        return 2 * self.dcp * self.cap * self.q_words

    def cand_records(self) -> int:
        return 2 * self.dcp * self.cap * self.topk * 2

    def part_records(self) -> int:
        return 2 * self.dcp * self.cap * self.heads * self.part_words

    def q_index(self, parity: int, src: int, row: int, word: int) -> int:
        return ((parity * self.dcp + src) * self.cap + row) * self.q_words + word

    def cand_index(self, parity: int, src: int, row: int, i: int, half: int) -> int:
        return (((parity * self.dcp + src) * self.cap + row) * self.topk + i) * 2 + half

    def part_index(self, parity: int, src: int, row: int, head: int, word: int) -> int:
        return (((parity * self.dcp + src) * self.cap + row) * self.heads + head) * self.part_words + word

    def bytes(self) -> int:
        return 8 * (self.q_records() + self.cand_records() + self.part_records())
