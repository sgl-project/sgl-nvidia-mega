# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fixed sizes and the mailbox layout of the tensor-parallel hop kernels.

The hops all-reduce rows of ``HIDDEN`` bf16 values across ``WORLD`` ranks. A row is sent
as ``NREC`` 64-bit records ``{3 x bf16, tag16}``; the tag is the row's epoch modulo
2^16, so a consumer accepts a record only if it was written in the epoch it is reading.
Buffers rotate through ``NBUF`` slots; a producer reuses a slot only after acquiring an
acknowledgement released by every consumer rank (every ``KACK`` epochs). Every value
here is the served value; none is a tuning knob.
"""

HIDDEN = 6144  # model hidden size (all-reduced row width)
O_PROJ_IN = 2048  # o_proj input width per rank
WORLD = 8  # tensor-parallel ranks
TOPK = 8  # routed experts per token
NREC = HIDDEN // 3  # 64-bit records per row
CLUSTER = 8  # consumer CTAs per row (one cluster)
REC_PER_CTA = NREC // CLUSTER  # 256 records per consumer CTA
PAIRS = REC_PER_CTA // 2  # 128 record pairs (16 bytes) per consumer CTA
ELEMS_PER_CTA = HIDDEN // CLUSTER  # 768
SOURCE_GROUPS = 4  # consumer thread groups, two source ranks each
POLL_THREADS = PAIRS * SOURCE_GROUPS  # 512
EPOCH_THREAD = POLL_THREADS  # lane 0 of the extra epoch warp
CONSUMER_THREADS = POLL_THREADS + 32  # 544

NBUF = 16  # buffer slots in rotation (a power of two: the u32 epoch wrap is seamless)
KACK = 4  # acknowledgement period in epochs (a power of two)
TAG_BITS = 16
TAG_MASK = (1 << TAG_BITS) - 1
assert NBUF >= 2 * KACK + 1 and NBUF & (NBUF - 1) == 0 and KACK & (KACK - 1) == 0

CTRL_BYTES = 4096
# control block, u32 word index (128-byte lines): ack[src] at 32 * src (written by rank
# src), E at 256, the per-row epochs from 264 (same line as E), the sticky error word at
# 512 and the arrival counter at 768 (local only).
W_ACK = 0
W_E = 256
W_EROW = 264
W_ERR = 512
W_ARRIVE = 768

POLL_TIMER_STRIDE = 16  # %globaltimer is read once every 16 polls
# bound of every cross-GPU wait; then the error record and trap
TIMEOUT_NS = 20_000_000_000
POISONED_TIMEOUT_NS = 2_000_000  # bound once this rank's error word is set
BF16_NAN = 0x7FC0

# error record kinds (host-mapped record, word 0)
ERR_RECORD_POLL = 1
ERR_ACK_WAIT = 2
ERR_ROUTER_POLL = 3
ERR_TOP1_POLL = 4  # the draft family's distributed top-1 (same record, its own kind)
ERR_KIND_NAMES = {
    ERR_RECORD_POLL: "record poll",
    ERR_ACK_WAIT: "ack wait",
    ERR_ROUTER_POLL: "router poll",
    ERR_TOP1_POLL: "top-1 poll",
}


def record_bytes(max_rows: int) -> int:
    """Bytes of the record area of a mailbox instance for ``max_rows`` rows."""
    return NBUF * WORLD * max_rows * NREC * 8


def mailbox_bytes(max_rows: int) -> int:
    return record_bytes(max_rows) + CTRL_BYTES
