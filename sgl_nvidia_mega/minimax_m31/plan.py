# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Which kernels fill each width's verify steps (rows), fixed at build: a slot serves a
width when its build serves it there (``Facts``), within its kernels' row bound. With
every slot off the step functions run SGLang's own modules in SGLang's order.

The covered steps are DSpark verify steps of 1-16 requests, 8 rows a request (8-128
rows). Slot ``mega`` is one request's step only: MegaAttention and the token-parallel
MoE take every MoE layer, fixed at 8 rows by their kernels. ``moe_nofill`` and
``moe_swap`` serve every width their build serves; at a width ``mega`` serves they
have no MoE layer left to run, so the plan leaves them off there. ``lm_head_mxfp8``
(opt-in) closes the step ``mega`` serves: its LM-head stream takes at most 8 rows.
"""

from __future__ import annotations

import msgspec

from sgl_nvidia_mega.minimax_m31 import MAX_REQUESTS, MAX_ROWS, ROWS

# slot mega: MegaAttention's and the token-parallel MoE's rows, one request's verify step
MEGA_ROWS = ROWS


class Slots(msgspec.Struct, frozen=True):
    """The kernels that fill one width's steps (``runtime``); with every field False the
    step functions run SGLang's own modules."""

    # every MoE layer on MegaAttention + the token-parallel MoE (not bitwise)
    mega: bool = False
    # the routed experts' NVFP4 input quantized into buffers built once (bitwise)
    moe_nofill: bool = False
    # moe_nofill's MoE with the routed chain on the main stream (bitwise)
    moe_swap: bool = False
    # mega's step projects through an MXFP8 copy of the LM-head shard (not bitwise)
    lm_head_mxfp8: bool = False


class Facts(msgspec.Struct, frozen=True):
    """What build found, which every rank agrees on (``runtime.build`` checks it); all
    empty: stock. Each field is the widths its slot's build serves: mega at its one
    width or none, moe_nofill at every width or none, moe_swap at moe_nofill's widths
    or none, lm_head_mxfp8 at mega's width or none."""

    mega: frozenset = frozenset()
    moe_nofill: frozenset = frozenset()
    moe_swap: frozenset = frozenset()
    lm_head_mxfp8: frozenset = frozenset()


def served_widths(widths: tuple[int, ...]) -> tuple[int, ...]:
    """The widths SGLang asks for that are whole requests' verify steps of at most
    MAX_REQUESTS requests."""
    return tuple(w for w in widths if 0 < w <= MAX_ROWS and w % ROWS == 0)


def slots(rows: int, facts: Facts) -> Slots:
    if not (0 < rows <= MAX_ROWS and rows % ROWS == 0):
        raise ValueError(
            f"{rows} rows are not a verify step of 1-{MAX_REQUESTS} requests "
            f"({ROWS} rows a request)"
        )
    mega = rows == MEGA_ROWS and rows in facts.mega
    # mega takes every MoE layer of its steps
    moe_nofill = not mega and rows in facts.moe_nofill
    return Slots(
        mega=mega,
        moe_nofill=moe_nofill,
        # it swaps the streams of moe_nofill's MoE
        moe_swap=moe_nofill and rows in facts.moe_swap,
        # it projects the hidden rows of mega's step
        lm_head_mxfp8=mega and rows in facts.lm_head_mxfp8,
    )


def plan(widths: tuple[int, ...], facts: Facts) -> dict[int, Slots]:
    return {w: slots(w, facts) for w in widths}


def describe(plan: dict[int, Slots]) -> str:
    """The plan as runs of widths with the same slots, e.g. "8: mega; 16-128:
    moe_nofill, moe_swap" ("stock" where no slot serves)."""
    runs: list[list] = []
    for w in sorted(plan):
        names = [f for f in Slots.__struct_fields__ if getattr(plan[w], f)]
        if runs and runs[-1][2] == names and runs[-1][1] + ROWS == w:
            runs[-1][1] = w
        else:
            runs.append([w, w, names])
    return "; ".join(
        f"{lo if lo == hi else f'{lo}-{hi}'}: {', '.join(names) or 'stock'}"
        for lo, hi, names in runs
    )
