# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Dense, linear-chain KDA TARGET_VERIFY with host-owned ReplaySSM commit.

Unlike PreparedKda's snapshot ABI, state_slots contains one committed checkpoint
per request. Verify never modifies that checkpoint or persistent convolution.
The host must complete commit/restore before launching the next verify.
"""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from ..common.host import fake, layout, stream, to_cute
from .kernel import KdaSplit
from .prepared import decode_conv_weight

# GB300 (SM103) TP8 (12 heads per rank) verify schedules (early, lanes per state row, minimum CTAs per SM, CTAs per
# head) by (requests, queries), from a TP8 sweep (cold L2, the KDA kernel gates): from Breq 2, 4 CTAs per head beat
# the default 8 (Breq 4: 9.4 vs 14.7 us). Q5 Breq 3 (15 rows): the KDA layer 34.278 -> 31.189 us on GB300 TP8.
# Not bitwise vs the default schedule.
SM103_TP8_HEADS = 12
SM103_TP8_VERIFY: dict = {
    (3, 5): (0, 8, 2, 4),
    (2, 8): (0, 8, 2, 4),
    (3, 8): (0, 8, 2, 4),
    (4, 8): (0, 8, 2, 4),
    (5, 8): (0, 8, 2, 4),
    (6, 8): (0, 8, 4, 4),
    (7, 8): (0, 8, 3, 4),
    (8, 8): (0, 8, 3, 4),
}


def verify_schedule(requests, queries, device, heads=None):
    """GB300: (0, 8, 1 if Q > 8 else 2, 8), and the TP8 table above."""
    capability = torch.cuda.get_device_capability(device)
    if capability == (10, 3) and heads == SM103_TP8_HEADS:
        tactic = SM103_TP8_VERIFY.get((requests, queries))
        if tactic is not None:
            return tactic
    return (0, 8, 1 if queries > 8 else 2, 8)


@cache
def executable(specs, heads, requests, queries, mode, has_bias, device, schedule=None):
    args = tuple(fake(*spec) for spec in specs)
    kernel = KdaSplit(
        heads,
        requests,
        mode,
        has_bias,
        schedule or (0, 8, 1 if queries > 8 else 2, 8),
        queries,
        replayssm=True,
    )
    scalars = tuple(cutlass.Float32(v) for v in (-5, 128**-0.5, 1e-6))
    with torch.accelerator.device_index(device):
        return cute.compile(
            kernel, *args[:13], *scalars, *args[13:], make_fake_stream()
        )


class PreparedKdaVerify:
    """Policies map (Breq, Q) to (packed projection pitch, gate MMA mode).

    Pool views, including nonstandard slot/conv pitches, are fixed at prepare.
    caches = (conv [S,3,3D] or [S,3D,3], temporal FP32[S,H,128,128]);
    conv_window = BF16[spec_rows,Qcap,3,3D]; rings = (rawv,rawk,g,beta),
    shaped [S,H,L,128] for the first three and [S,H,L] for beta.
    Raw V/K are BF16, g/beta FP32; L >= Q. Metadata is int32[Breq],
    state slot 0 is valid and a negative state/scratch row is padding.
    """

    def __init__(
        self,
        *,
        conv_weight,
        f_b_weight,
        a_log,
        dt_bias,
        norm_weight,
        lower_bound,
        eps,
        policies,
        conv_dim_first=False,
    ):
        if any(not isinstance(key, tuple) or len(key) != 2 for key in policies):
            raise ValueError("KDA verify policies require explicit (Breq, Q) keys")
        if any(mode not in (0, 1) for _, mode in policies.values()):
            raise ValueError("KDA verify requires gate MMA mode 0 or 1")
        self.heads = a_log.numel()
        self.dim, self.device = self.heads * 128, dt_bias.device
        self.has_bias = False  # K31 conv has no bias.
        self.weights = (
            decode_conv_weight(conv_weight),
            dt_bias,
            f_b_weight.detach().contiguous(),
            a_log.detach().reshape(-1),
            dt_bias.detach().reshape(-1),
            norm_weight.detach().float(),
        )
        self.scalars = tuple(map(cutlass.Float32, (lower_bound, 128**-0.5, eps)))
        self.policies, self.conv_dim_first, self.kernels = policies, conv_dim_first, {}

    def _verify_operands(
        self, caches, projected, state_slots, out, scratch_rows, conv_window, rings
    ):
        conv, state = caches
        conv = conv if self.conv_dim_first else conv.transpose(-1, -2)
        w, bias, fb, a_log, dt, norm = self.weights
        d, h = self.dim, self.heads
        p = lambda start, width: projected[:, start : start + width]
        return (
            p(0, 3 * d),
            w,
            bias,
            conv,
            p(4 * d, 128),
            p(4 * d + 128, h),
            a_log,
            dt,
            state_slots,
            state,
            out,
            p(3 * d, d),
            norm,
            fb,
            scratch_rows,
            conv_window.transpose(-1, -2),
            *rings,
        )

    def prepare_caches(self, caches, *, conv_window, rings):
        if caches[0].dtype != torch.bfloat16 or caches[1].dtype != torch.float32:
            raise ValueError("KDA verify requires BF16 conv and FP32 temporal pools")
        h, d = self.heads, self.dim
        if self.weights[0].dtype != torch.float32:
            raise ValueError("K31 verify requires FP32 convolution weights")
        if tuple(caches[1].shape[1:]) != (h, 128, 128):
            raise ValueError("expected temporal [S,H,128,128]")
        if caches[1].stride()[1:] != (128 * 128, 128, 1):
            raise ValueError("stock commit requires packed temporal H/V/K axes")
        if tuple(conv_window.shape[-2:]) != (3, 3 * d):
            raise ValueError("expected conv_window [spec_rows,Qcap,3,3D]")
        if conv_window.dtype != torch.bfloat16:
            raise ValueError("conv_window must preserve BF16 input taps")
        if len(rings) != 4:
            raise ValueError("ReplaySSM needs rawv, rawk, g and beta")
        length = rings[0].shape[2]
        for tensor, tail, dtype in zip(
            rings,
            ((h, length, 128),) * 3 + ((h, length),),
            (torch.bfloat16, torch.bfloat16, torch.float32, torch.float32),
        ):
            if tuple(tensor.shape[1:]) != tail or tensor.dtype != dtype:
                raise ValueError("ReplaySSM ring shape/dtype mismatch")
            packed = (length * 128, 128, 1) if len(tail) == 3 else (length, 1)
            if tensor.stride()[1:] != packed:
                raise ValueError("stock commit requires packed ring inner axes")
        for (requests, queries), (pitch, mode) in self.policies.items():
            if requests < 1 or queries < 2 or queries > 16:
                raise ValueError("KDA ReplaySSM verify covers Q=2..16")
            if queries > min(length, conv_window.shape[1]):
                raise ValueError("verify Q exceeds ring/conv-window capacity")
            if pitch < 4 * d + 128 + h or pitch % 8:
                raise ValueError(
                    "packed projection needs its full payload and 16-byte row alignment"
                )
            projected = torch.empty(
                requests * queries, pitch, device="meta", dtype=torch.bfloat16
            )
            out = torch.empty(
                requests * queries, d, device="meta", dtype=torch.bfloat16
            )
            indices = torch.empty(requests, device="meta", dtype=torch.int32)
            ops = self._verify_operands(
                caches, projected, indices, out, indices, conv_window, rings
            )
            self.kernels[requests, queries] = executable(
                tuple(map(layout, ops)),
                h,
                requests,
                queries,
                mode,
                self.has_bias,
                self.device.index,
                verify_schedule(requests, queries, self.device, heads=h),
            )
        return self

    def run(
        self,
        projected,
        out,
        *,
        caches,
        state_slots,
        scratch_rows,
        conv_window,
        rings,
        query_width,
    ):
        requests = projected.shape[0] // query_width
        ops = self._verify_operands(
            caches,
            projected,
            state_slots[:requests],
            out,
            scratch_rows[:requests],
            conv_window,
            rings,
        )
        views = tuple(to_cute(t, 16) for t in ops)
        self.kernels[requests, query_width](
            *views[:13], *self.scalars, *views[13:], stream(self.device)
        )
        return out
