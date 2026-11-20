# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Which kernel fills each slot of a covered step, per width (rows), fixed at build.

Rows are target-verify rows: the batch size times the DFLASH draft block, so a
server at --max-running-requests 112 with 8 draft tokens reaches 896.

Unlike deepseek_v41's, every slot here is width-independent so far: these
kernels replace a norm, an attention call or a boundary reduction and none of
them has a measured row bound yet. The per-width table is kept because the
contract is per width and because the bounds will arrive once each slot has an
A/B at several widths; until then a slot is on for every built width or off
everywhere, and saying so explicitly is better than implying a measurement that
was never made.
"""

from __future__ import annotations

import msgspec

# The widest covered step: --max-running-requests 112 x 8 draft tokens.
MAX_ROWS = 896


class Slots(msgspec.Struct, frozen=True):
    """Which kernels serve a step of this width."""

    # attention over the NVFP4 full-attention KV pool, and the FP8 sliding-window one
    nvfp4_kv_verify: bool = False
    nvfp4_kv_verify_wide: bool = False  # the wide form, for many requests
    full_verify_attn: bool = False
    swa_verify_attn: bool = False
    full_attn_wide: bool = False
    xattn_b: bool = False  # the own extend-attention op over the NVFP4 pool
    # norms and the layer boundary
    input_norm: bool = False
    final_norm: bool = False
    ar_attn_boundary: bool = False
    pull_bounded: bool = False  # bounded waits for the NVLink pull all-gather


class Facts(msgspec.Struct, frozen=True):
    """What build found, which every rank must agree on. All false: every slot
    runs its stock code and the step is not worth covering."""

    nvfp4_kv_verify: bool = False
    nvfp4_kv_verify_wide: bool = False
    full_verify_attn: bool = False
    swa_verify_attn: bool = False
    full_attn_wide: bool = False
    xattn_b: bool = False
    input_norm: bool = False
    final_norm: bool = False
    ar_attn_boundary: bool = False
    pull_bounded: bool = False


def slots(rows: int, facts: Facts) -> Slots:
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"{rows} rows are outside the covered widths 1-{MAX_ROWS}")
    return Slots(**msgspec.structs.asdict(facts))


def plan(widths: tuple[int, ...], facts: Facts) -> dict[int, Slots]:
    return {w: slots(w, facts) for w in widths}
