# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Where the MLA head gate sits in the packed input projection's output, for every reader (dense, sparse and DCP MLA).

``K31MK_DSA_QF_G18=1`` (default 0 = off; K3.1 TARGET_VERIFY build only, with ``K31MK_DSA_QF=1`` and
``K31MK_PROJ_TMEM8=1``): at load the plugin copies each QF-packed MLA projection weight [q_a | kv_a | gate | wk |
weights_proj] into a new tensor ordered [q_a | kv_a | wk | weights_proj | gate] (the host's packed storage is left as it is:
its stock Parameters are views of it). At 8 rows the TM8 in-projection then covers rows 0:2272 only (18 M tiles = 18
clusters of 7, instead of 24, so that the in-projection still fits one wave on the target GPU while another
kernel still occupies an SM) and the head-gate rows 2272:3040 run as one SplitKGemm with the same split K (7), on a side
stream forked right after AttnRes (the projection's call slot). Same MMAs and the same per-element split-K sum order as the
3040-row TM8 (TM8 is bitwise equal to SplitKGemm at the same split), so every projected value is bitwise equal to the
QF path's, at another column. Every reader takes the gate through ``head_gate``, which first joins that side stream into
the reader's stream. Other row counts run one projection over all rows of the new order. Extra device memory: one copy of
every MLA projection weight (TP16: 24 x 3040 x 7168 x 2 B = 1.05 GB per GPU).
"""

import os
from functools import cache

import torch

G18 = os.environ.get("K31MK_DSA_QF_G18", "0") == "1"
QUERY_ROWS = 2112  # q_a 1536 + kv_a 576
INDEX_ROWS = 128 + 32  # indexer wk + weights_proj
GATE_TACTIC = (128, 8, 7, 5)  # SplitKGemm at TM8's split K; 198,792 B SMEM by splitk_smem_estimate (normal allocation)
G18_ROWS = 8  # the row count whose projection splits (Breq 1, Q8)

_pending = {}  # device index -> the side stream holding this layer's unjoined gate GEMM


def gate_col():
    """First column of the head gate in the packed projection output: 2112, or 2272 under K31MK_DSA_QF_G18."""
    return QUERY_ROWS + INDEX_ROWS if G18 else QUERY_ROWS


def head_gate(projected, heads):
    """``projected``'s head-gate columns [rows, 128 * heads]. Under K31MK_DSA_QF_G18, a gate GEMM still running on the side
    stream is joined into the current stream first, so call this right before launching the reader on that stream."""
    pending = _pending.pop(projected.device.index, None)
    if pending is not None:
        torch.cuda.current_stream(projected.device).wait_stream(pending)
    col = gate_col()
    return projected[:, col : col + 128 * heads]


def relayout(weight, heads):
    """New tensor [q_a | kv_a | wk | weights_proj | gate] from the QF packing [q_a | kv_a | gate | wk | weights_proj]."""
    gate = 128 * heads
    if weight.dim() != 2 or weight.shape[0] != QUERY_ROWS + gate + INDEX_ROWS:
        raise ValueError(
            f"K31MK_DSA_QF_G18: want the QF-packed MLA projection ({QUERY_ROWS + gate + INDEX_ROWS} rows at {heads} heads), "
            f"got {tuple(weight.shape)}"
        )
    q, g = QUERY_ROWS, QUERY_ROWS + gate
    return torch.cat((weight[:q], weight[g:], weight[q:g]), dim=0).contiguous()


def restore(weight, heads):
    """Inverse of ``relayout`` (tests: the round trip)."""
    q, i = QUERY_ROWS, QUERY_ROWS + INDEX_ROWS
    return torch.cat((weight[:q], weight[i:], weight[q:i]), dim=0).contiguous()


@cache
def _side_stream(device):
    return torch.cuda.Stream(device=device)


class G18Projection:
    """The MLA input projection over a ``relayout`` weight. At 8 rows: TM8 over rows 0:2272 on the caller's stream and the
    head-gate rows as SplitKGemm ``GATE_TACTIC`` on a side stream forked at the call; ``head_gate`` joins it. Other row
    counts: one PreparedProjection over all rows with the caller's tactics."""

    def __init__(self, weight, policies, heads, projection_module):
        split = gate_col()
        if not G18 or weight.shape[0] != split + 128 * heads:
            raise ValueError(f"G18Projection needs K31MK_DSA_QF_G18=1 and {split + 128 * heads} rows, got {weight.shape[0]}")
        policies = dict(policies)
        tm8 = policies.pop(G18_ROWS, None)
        if tm8 is not None and len(tm8) != 7:
            raise ValueError(f"K31MK_DSA_QF_G18 needs the TM8 in-projection at 8 rows (K31MK_PROJ_TMEM8=1), got tactic {tm8}")
        PreparedProjection = projection_module.PreparedProjection
        self.weight, self.device, self.split = weight, weight.device, split
        self.full = PreparedProjection(weight, policies) if policies else None
        self.inproj = self.gate = None
        if tm8 is not None:
            self.inproj = PreparedProjection(weight[:split], {G18_ROWS: tm8})
            self.gate = PreparedProjection(weight[split:], {G18_ROWS: GATE_TACTIC})

    def prepare(self):
        for p in (self.full, self.inproj, self.gate):
            if p is not None:
                p.prepare()
        return self

    def run(self, x, out):
        if x.shape[0] != G18_ROWS or self.gate is None:
            if self.full is None:
                raise ValueError(f"G18Projection: no tactic for {x.shape[0]} rows")
            return self.full.run(x, out)
        device = x.device
        if device.index in _pending:
            raise RuntimeError("K31MK_DSA_QF_G18: the previous gate GEMM was never joined (no head_gate read)")
        main, side = torch.cuda.current_stream(device), _side_stream(device)
        side.wait_stream(main)  # the gate GEMM reads only this layer's input (AttnRes output)
        with torch.cuda.stream(side):
            self.gate.run(x, out[:, self.split :])
        _pending[device.index] = side
        self.inproj.run(x, out[:, : self.split])
        return out
