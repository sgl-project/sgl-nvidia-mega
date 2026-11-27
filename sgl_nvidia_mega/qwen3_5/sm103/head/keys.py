# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import struct

__all__ = ["KEY_FLOOR", "NAN_NEG_INF", "NAN_SPLIT", "NAN_TORCH", "ordered_hi"]

KEY_FLOOR = -(2**63)


def ordered_hi(value: float) -> int:
    bits = struct.unpack("<i", struct.pack("<f", value))[0]
    if bits == -(2**31):
        bits = 0
    return bits ^ 0x7FFFFFFF if bits < 0 else bits


NAN_TORCH = 0x7FFFFFFF
NAN_NEG_INF = ordered_hi(float("-inf"))
NAN_SPLIT = ordered_hi(-1e30)
