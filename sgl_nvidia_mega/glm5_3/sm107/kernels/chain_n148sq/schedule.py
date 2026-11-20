# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The routed-MoE work split shared by the device code (moe.py) and the host, which derives the FC2 lists from it."""
import struct

CTAS = 212
CUT_TOTAL = 1112
CUT_RATE = 6
# (first CTA, end CTA, weight discount): CTAs that also carry another per-step role take fewer FC1 pieces
CUT_DISCOUNTS = ((65, 97, 1), (100, 148, 2), (180, 212, 1))
FC1_SPLIT_EXPERTS = 27
MAX_EXPERTS = 64
LIST_ROWS = MAX_EXPERTS * (MAX_EXPERTS + 1)


def _clipped(c, start, end):
    return 0 if c <= start else min(c, end) - start


def _cut_weight(c):
    return CUT_RATE * c - sum(k * _clipped(c, start, end) for start, end, k in CUT_DISCOUNTS)


def _aligned_f2(c, experts):
    if c >= CTAS:
        return 8 * experts
    slot = min(((c + 1) * experts - 1) // CTAS, experts - 1)
    c0 = slot * CTAS // experts
    width = (slot + 1) * CTAS // experts - c0
    reciprocal = (65536 + width - 1) // width if 3 <= width <= 7 else 65536 // 8
    return 8 * slot + ((8 * (c - c0) * reciprocal) >> 16)


def _fc1_cut(c, experts):
    if experts >= FC1_SPLIT_EXPERTS:
        return _cut_weight(c) * (24 * experts) // CUT_TOTAL - _aligned_f2(c, experts)
    return _cut_weight(c) * (16 * experts) // CUT_TOTAL


def fc2_lists_payload():
    """FC2 reduction lists, 4160 rows x 3 little-endian uint32.

    FC1 piece g of an E-expert step is K chunk g & 7 of block g >> 3 (one expert's gate or up half). CTA c owns
    pieces [cut(c), cut(c + 1)) and publishes one partial per block it touches as producer 2c + run. Row
    E(E - 1) + block lists, in CTA order (the order FC2 adds them), the producers of that block: the count in bits
    0-3, then 9-bit producer codes from bit 4.
    """
    words = []
    for experts in range(1, MAX_EXPERTS + 1):
        blocks = [[] for _ in range(2 * experts)]
        for c in range(CTAS):
            first, end = _fc1_cut(c, experts), _fc1_cut(c + 1, experts)
            run = 0
            for g in range(first, end):
                if g != first and g & 7 == 0:
                    run += 1
                if g == first or g & 7 == 0:
                    blocks[g >> 3].append(2 * c + run)
        for codes in blocks:
            assert len(codes) <= 8, (experts, codes)
            bits = len(codes)
            for q, code in enumerate(codes):
                bits |= code << (4 + 9 * q)
            words += [bits & 0xFFFFFFFF, (bits >> 32) & 0xFFFFFFFF, bits >> 64]
    return struct.pack(f"<{3 * LIST_ROWS}I", *words)
