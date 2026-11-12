# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Portions adapted from vLLM (https://github.com/vllm-project/vllm, Apache-2.0), commit
# 642c18d64c4cb6db3ded8fbe1786718cd4005fcb:
# vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/moe/experts.py (ExpertsPlanKernel).
# Modified by NVIDIA: plans above 64 rows (token t into row t % 64 of plan t // 64), quantized from the FP32 latent.
"""Plan writer above 64 rows for the vendored wide experts (``K31MK_MOE_WIDE``): token t into row t % 64 of the 64-row
MXFP8 plan t // 64 (the vendored ``WidePlanKernel`` layout), quantized from the FP32 latent as our 64-row ``ExpertsPlanKernel``
does (the vendored writer takes BF16 x, whose pre-rounding moves the result off the FP32-input reference)."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Uint8

from ..common.primitives import load_words, thread, view
from .experts import PLAN_BYTES, ExpertsPlanKernel, plan_chunk


class WidePlanKernel(ExpertsPlanKernel):
    """``ExpertsPlanKernel`` over ``batch`` (> 64) rows in consecutive 64-row plans, one CTA per token."""

    @cute.kernel
    def kernel(self, args):
        x, plan = args
        tid, _, _, token = thread()
        block = view(plan.iterator, token // 64 * PLAN_BYTES, Uint8, PLAN_BYTES)
        cute.arch.griddepcontrol_wait()
        for i in cutlass.range_constexpr(2):
            chunk = tid + 256 * i
            if chunk < 448:  # warp-uniform: chunks 256-447 are warps 0-5
                source = x.iterator + token * 3584 + chunk * 8
                if cutlass.const_expr(x.element_type == Float32):
                    words = (*load_words(source, count=4, mutable=True),
                             *load_words(source + 4, count=4, mutable=True))
                    plan_chunk(block, token % 64, chunk, words, fp32=True)
                else:
                    words = load_words(source, count=4, mutable=True)
                    plan_chunk(block, token % 64, chunk, words)
        cute.arch.griddepcontrol_launch_dependents()
