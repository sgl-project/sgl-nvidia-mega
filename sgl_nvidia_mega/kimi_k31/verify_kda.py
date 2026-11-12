# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Adapt front's tensor-only KDA provider to the fixed K31 verify ABI."""

from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda.verify import (
    PreparedKdaVerify as FrontKdaVerify,
)


class PreparedKdaVerify:
    def __init__(self, weights, policies, device_index):
        if weights.has_bias or weights.scale != 128**-0.5:
            raise ValueError("K31 verify requires bias-free KDA and scale=128**-0.5")
        # Preserve FP32 coefficients while undoing the existing prepared decode
        # layout [QKV,tap,D] into the front provider's raw [QKV*D,1,tap].
        conv = weights.conv_weight.transpose(1, 2).reshape(-1, 1, 4).contiguous()
        self.provider = FrontKdaVerify(
            conv_weight=conv,
            f_b_weight=weights.f_b,
            a_log=weights.a_log,
            dt_bias=weights.dt_bias,
            norm_weight=weights.norm_weight,
            lower_bound=weights.lower_bound,
            eps=weights.norm_eps,
            policies={(b, q): (pitch, mode) for b, q, pitch, mode in policies},
            conv_dim_first=True,
        )
        self.pools = None

    def prepare_caches(self, pools):
        self.pools = pools
        self.provider.prepare_caches(
            (pools.conv, pools.temporal),
            conv_window=pools.conv_windows.transpose(-1, -2),
            rings=(pools.rawv, pools.rawk, pools.log_g, pools.beta),
        )
        return self

    def run(self, projected, out, *, pools, step):
        # Host metadata guarantees valid_requests == (state_slots >= 0);
        # padding has a negative physical state slot and is skipped by front.
        return self.provider.run(
            projected,
            out,
            caches=(pools.conv, pools.temporal),
            state_slots=step.state_slots,
            scratch_rows=step.scratch_rows,
            conv_window=pools.conv_windows.transpose(-1, -2),
            rings=(pools.rawv, pools.rawk, pools.log_g, pools.beta),
            query_width=step.query_width,
        )
