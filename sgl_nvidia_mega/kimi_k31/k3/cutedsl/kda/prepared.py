# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Prepared KDA core. One packed projection supplies Q/K/V, the output gate, F_A and
beta; per-batch policies select the gate schedule and the projection row pitch."""

import os
from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from ..common.host import fake, layout, stream, to_cute
from ..common.policy import batch_table
from .kernel import KdaSplit

# KdaSplit schedule "early.lanes.occupancy.cluster.evict" per batch by heads (TP8 12,
# TP16 6), from GB300 sweeps: early predecessor-independent loads (MMA gate modes only;
# 2 also stages the state rows in shared memory, 3 also the F_B operands), lanes per
# state row, minimum CTAs per SM, CTAs per head (0: defaults) and L2 evict-first state
# traffic (staged only).
SCHEDULE = {12: "16-32:3.16.3.2.1,64:3.16.4.2.1", 6: "16:1.0.0,32-64:3.16.3.2"}
# VR200 (SM107) B2-8 decode: GB300's B16+ gate modes and staged schedules measured faster
# there than GB300's B2-8 mode 2 (cold L2, same gates as the shipped modes).
# Rows -> (gate mode, schedule).
# ``K31MK_KDA_SM107_RETUNE=0`` keeps GB300's B2-8 policy on SM107.
SM107_DECODE = {
    2: (0, "1.0.0.4.0"),
    3: (0, "1.0.0.4.0"),
    **{b: (0, "3.16.3.2.1") for b in (4, 5, 6, 7)},
    8: (1, "3.16.3.2.1"),
    # B16/B32: GB300's B64 occupancy 4 (measured faster at B16 and B32).
    16: (0, "3.16.4.2.1"),
    32: (0, "3.16.4.2.1"),
}


def _sm107_retune(device):
    if os.environ.get("K31MK_KDA_SM107_RETUNE", "1") == "0":
        return False
    return torch.cuda.get_device_capability(device) == (10, 7)


def decode_mode(batch, device):
    """Gate mode of TP8 decode rows (GB300: B1 1, B2-8 2, B16/32/64 0)."""
    if batch in SM107_DECODE and _sm107_retune(device):
        return SM107_DECODE[batch][0]
    return 0 if batch in (16, 32, 64) else 1 if batch == 1 else 2


# Verify (q > 1) schedules by request count B, from GB300
# sweeps over q 4-8: no staging; eight-CTA clusters, then fewer as B grows.
SPEC_SCHEDULE = {
    12: "1:1.8.2.8,2-4:1.8.3.8,5-12:1.8.3.4,13-16:1.8.3.2",
    6: "1:1.8.2.8,2:1.8.0.8,3-12:1.8.3.8,13-16:1.8.3.4",
}


@cache
def executable(specs, heads, gate_mode, has_bias, device, tokens=1):
    args = tuple(fake(shape, stride, dtype) for shape, stride, dtype in specs)
    batch = specs[0][0][0]
    table = SCHEDULE if tokens == 1 else SPEC_SCHEDULE
    schedule = batch_table(table.get(heads, ""))
    if tokens == 1 and heads == 12 and batch in SM107_DECODE and _sm107_retune(device):
        retuned_mode, retuned = SM107_DECODE[batch]
        if gate_mode == retuned_mode:
            schedule[batch] = tuple(map(int, retuned.split(".")))
    schedule = schedule.get(batch // tokens, (0, 0, 0)) if gate_mode != 2 else (0,) * 3
    bf16_state = specs[9][2] == torch.bfloat16  # BF16 temporal pool
    # staging needs whole FP32 rows (BF16 state: early 1, no staging)
    if schedule[0] >= 2 and (specs[9][1][-2:] != (128, 1) or bf16_state):
        schedule = (1, *schedule[1:4])  # no evict hint without staging
    requests = batch // tokens
    kernel = KdaSplit(heads, requests, gate_mode, has_bias, schedule, tokens, bf16_state=bf16_state)
    scalars = (cutlass.Float32(v) for v in (-5, 128**-0.5, 1e-6))
    with torch.accelerator.device_index(device):
        return cute.compile(
            kernel, *args[:13], *scalars, *args[13:], make_fake_stream()
        )


def decode_conv_weight(weight):
    """The [3, width, D] (Q, K, V) decode layout of the [3 * D, 1, width] conv
    weight."""
    width = weight.shape[-1]
    return weight.detach().view(3, -1, width).transpose(1, 2).contiguous()


class PreparedKda:
    def __init__(self, attention, policies, conv_dim_first):
        bias, dt = attention.conv1d.bias, attention.dt_bias
        self.heads = attention.A_log.numel()
        self.dim, self.has_bias = self.heads * 128, bias is not None
        conv_w = decode_conv_weight(attention.conv1d.weight)
        bias, fb = dt if bias is None else bias, attention.f_b_proj.weight
        norm = attention.o_norm.weight.detach().float()  # BF16 slows the fused kernel
        self.weights = (conv_w, bias, fb, attention.A_log, dt, norm)
        self.device, self.policies = dt.device, policies
        scalars = (attention.gate_lower_bound, 128**-0.5, attention.o_norm.eps)
        self.scalars = tuple(map(cutlass.Float32, scalars))
        self.conv_dim_first, self.kernels = conv_dim_first, {}

    def _operands(self, caches, projected, slots, out, accepted=None):
        """Kernel tensors in launch order: 13 before the scalars, then F_B (and the
        verify step's accepted counts)."""
        conv, state = caches[:2]
        conv = conv if self.conv_dim_first else conv.transpose(-1, -2)
        w, bias, fb, a_log, dt, norm = self.weights
        d, h = self.dim, self.heads
        p = lambda start, width: projected[:, start : start + width]
        ops = (p(0, 3 * d), w, bias, conv, p(4 * d, 128), p(4 * d + 128, h), a_log, dt)
        tail = (fb,) if accepted is None else (fb, accepted)
        return ops + (slots, state, out, p(3 * d, d), norm) + tail

    def prepare_caches(self, caches):
        """Policy keys: a batch, or (requests, tokens) for verify steps."""
        for key, (pitch, mode) in self.policies.items():
            requests, tokens = key if isinstance(key, tuple) else (key, 1)
            batch, meta = requests * tokens, dict(device="meta")
            projected = torch.empty((batch, pitch), dtype=torch.bfloat16, **meta)
            out = torch.empty(batch, self.dim, dtype=torch.bfloat16, **meta)
            accepted = torch.empty(requests, dtype=torch.int32, **meta)
            shape = (requests, tokens) if tokens > 1 else (batch,)
            slots = torch.empty(shape, dtype=torch.int32, **meta)
            accepted = accepted if tokens > 1 else None
            operands = self._operands(caches, projected, slots, out, accepted)
            self.kernels[key] = executable(
                tuple(map(layout, operands)),
                *(self.heads, mode, self.has_bias, self.device.index, tokens),
            )
        return self

    def run(self, projected, out, *, caches, slots, accepted=None):
        """``accepted`` [>= B] (verify steps): ``slots`` is [>= B, q] contiguous."""
        batch = projected.shape[0]
        key, rows = batch, batch
        if accepted is not None:
            rows = batch // slots.shape[1]
            key, accepted = (rows, slots.shape[1]), accepted[:rows]
            if slots.stride() != (slots.shape[1], 1):  # e.g. a block-table slice
                slots = slots.contiguous()
        tensors = self._operands(caches, projected, slots[:rows], out, accepted)
        views = tuple(to_cute(t, 16) for t in tensors)
        self.kernels[key](*views[:13], *self.scalars, *views[13:], stream(self.device))
        return out
