# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MoE FP32 latent publication, without modifying other projection publishers."""
import cutlass as c
import cutlass.cute as cute
from ..gemm.ws_gemm import WsGemm, positive_zero32
from ..common.primitives import multicast_words


class Fp32FrontGemm(WsGemm):
    @cute.jit
    def epilogue(self, args, token, col, values):
        out, mc, slot, base = args[2:6]
        if (col < self.valid) | (col >= self.split):
            first = c.Int32(0)
            if col < self.valid:
                first = 7168 + 2 * (slot + col)
            else:
                first = 2 * (base + col - self.split)
            for v in c.range_constexpr(2):
                words = tuple(positive_zero32(values[4 * v + i]) for i in range(4))
                offset = token * out.shape[2] + first + 8 * v
                multicast_words(mc + c.Int64(offset) * 2, words, kind="f32")
