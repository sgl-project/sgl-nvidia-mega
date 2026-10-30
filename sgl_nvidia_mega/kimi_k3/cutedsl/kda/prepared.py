# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Prepared KDA core. One packed projection supplies Q/K/V, the output gate, F_A and
beta; per-batch policies select the gate schedule and the projection row pitch."""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from ...shape_policy import VERIFY_QUERIES, token_widths
from ..common.host import fake, layout, stream, to_cute
from ..common.policy import batch_table
from .kernel import KdaSplit

# KdaSplit schedule "early.lanes.occupancy.cluster.evict" per batch by heads (TP8 12,
# TP16 6), from GB300 sweeps: early predecessor-independent loads (MMA gate modes only;
# 2 also stages the state rows in shared memory, 3 also the F_B operands), lanes per
# state row, minimum CTAs per SM, CTAs per head (0: defaults) and L2 evict-first state
# traffic (staged only).
SCHEDULE = {12: "16-32:3.16.3.2.1,64:3.16.4.2.1", 6: "16:1.0.0,32-64:3.16.3.2"}
# Verify (q > 1) schedules by request count B, from GB300
# sweeps over q 4-8: no staging; eight-CTA clusters, then fewer as B grows.
SPEC_SCHEDULE = {
    12: "1:1.8.2.8,2-4:1.8.3.8,5-12:1.8.3.4,13-16:1.8.3.2",
    6: "1:1.8.2.8,2:1.8.0.8,3-12:1.8.3.8,13-16:1.8.3.4",
}


@cache
def executable(specs, heads, gate_mode, has_bias, device, tokens=1, split_cache=False):
    args = tuple(fake(shape, stride, dtype) for shape, stride, dtype in specs)
    batch = specs[0][0][0]
    table = SCHEDULE if tokens == 1 else SPEC_SCHEDULE
    schedule = batch_table(table.get(heads, ""))
    schedule = schedule.get(batch // tokens, (0, 0, 0)) if gate_mode != 2 else (0,) * 3
    if schedule[0] >= 2 and specs[9][1][-2:] != (128, 1):  # staging needs whole rows
        schedule = (1, *schedule[1:4])  # no evict hint without staging
    requests = batch // tokens
    kernel = KdaSplit(
        heads, requests, gate_mode, has_bias, schedule, tokens, split_cache
    )
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
        self.verify_kernels, self.verify_layouts = {}, {}

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

    def _verify_operands(
        self, caches, projected, slots, out, intermediates, scratch_indices
    ):
        conv_windows, checkpoints = intermediates
        return self._operands(caches, projected, slots, out) + (
            scratch_indices,
            conv_windows.transpose(-1, -2),
            checkpoints,
        )

    def _check_verify_caches(self, caches, intermediates):
        conv, state = caches
        conv_windows, checkpoints = intermediates
        if self.heads != 12 or self.conv_dim_first:
            raise ValueError(
                "split-cache verification requires TP8 and window-first conv"
            )
        expected = (
            (conv, (3, 3 * self.dim), torch.bfloat16, 1),
            (state, (self.heads, 128, 128), torch.float32, 1),
            (conv_windows, (3, 3 * self.dim), torch.bfloat16, 2),
            (checkpoints, (self.heads, 128, 128), torch.float32, 2),
        )
        for tensor, tail, dtype, outer in expected:
            if (
                tensor.ndim != len(tail) + outer
                or tuple(tensor.shape[-len(tail) :]) != tail
                or tensor.dtype != dtype
                or tensor.device != self.device
            ):
                raise ValueError(
                    "invalid split-cache verification shape, dtype or device"
                )
            stride = 1
            for size, actual in zip(
                reversed(tensor.shape[1:]), reversed(tensor.stride()[1:])
            ):
                if actual != stride:
                    raise ValueError(
                        "split-cache verification requires contiguous cache rows"
                    )
                stride *= size
            if (
                tensor.stride(0) < stride
                or tensor.stride(0) * tensor.element_size() % 16
            ):
                raise ValueError(
                    "split-cache verification requires aligned, disjoint cache rows"
                )
            if tensor.storage_offset() * tensor.element_size() % 16:
                raise ValueError(
                    "split-cache verification requires 16-byte aligned caches"
                )
        if (
            conv.shape[0] != state.shape[0]
            or conv_windows.shape[:2] != checkpoints.shape[:2]
        ):
            raise ValueError("split-cache verification cache capacities do not match")
        if conv_windows.shape[1] < 4:
            raise ValueError(
                "split-cache verification requires at least four checkpoints"
            )

    def prepare_verify_caches(self, caches, intermediates):
        """Compile TP8/q4-8 policies for separate committed pools and checkpoints.

        ``caches``: BF16 conv [P, 3, 4608], FP32 SSM [P, 12, 128, 128].
        ``intermediates``: BF16 conv [S, >=q, 3, 4608], FP32 SSM
        [S, >=q, 12, 128, 128]. Only the leading pool stride may be padded.
        """
        self._check_verify_caches(caches, intermediates)
        policies = {
            key: policy
            for key, policy in self.policies.items()
            if isinstance(key, tuple)
        }
        if not policies:
            raise ValueError("split-cache verification requires (requests, q) policies")
        for key, (pitch, mode) in policies.items():
            requests, tokens = key
            if (
                tokens not in VERIFY_QUERIES
                or requests * tokens not in token_widths(tokens)
                or (pitch, mode) != (6288, 1)
            ):
                raise ValueError(
                    f"unsupported split-cache verification policy: {key}, {(pitch, mode)}"
                )
            if intermediates[0].shape[1] < tokens:
                raise ValueError(
                    f"split-cache verification requires at least {tokens} checkpoints"
                )
            meta = {"device": "meta"}
            projected = torch.empty(
                (requests * tokens, pitch), dtype=torch.bfloat16, **meta
            )
            out = torch.empty(
                (requests * tokens, self.dim), dtype=torch.bfloat16, **meta
            )
            slots = torch.empty(requests, dtype=torch.int32, **meta)
            scratch_indices = torch.empty_like(slots)
            operands = self._verify_operands(
                caches, projected, slots, out, intermediates, scratch_indices
            )
            specs = tuple(map(layout, operands))
            self.verify_kernels[key] = executable(
                specs, self.heads, mode, self.has_bias, self.device.index, tokens, True
            )
            self.verify_layouts[key] = specs
        return self

    def run_verify(
        self, projected, out, *, caches, slots, intermediates, scratch_indices
    ):
        """Read persistent state and publish q checkpoints without committing.

        ``slots`` and ``scratch_indices`` are independent contiguous int32 [B].
        Physical slots <= 0 pad a request; scratch row zero is valid. Live scratch
        rows must be distinct and in range. Graph replay may change index values,
        not the prepared layouts. The caller owns the subsequent accept commit.
        """
        if (
            slots.ndim != 1
            or projected.ndim != 2
            or slots.shape[0] <= 0
            or projected.shape[1] != 6288
            or projected.shape[0] % slots.shape[0]
        ):
            raise ValueError(
                "split-cache verification requires uniform rows per request"
            )
        key = (slots.shape[0], projected.shape[0] // slots.shape[0])
        if key not in self.verify_kernels:
            raise ValueError(f"unprepared split-cache verification policy: {key}")
        tensors = self._verify_operands(
            caches, projected, slots, out, intermediates, scratch_indices
        )
        if tuple(map(layout, tensors)) != self.verify_layouts[key]:
            raise ValueError("split-cache verification layouts differ from preparation")
        if any(tensor.device != self.device for tensor in tensors):
            raise ValueError(
                "split-cache verification operands must share the prepared device"
            )
        if any(tensor.data_ptr() % 16 for tensor in tensors):
            raise ValueError(
                "split-cache verification requires 16-byte aligned operands"
            )
        views = tuple(to_cute(tensor, 16) for tensor in tensors)
        self.verify_kernels[key](
            *views[:13], *self.scalars, *views[13:], stream(self.device)
        )
        return out
