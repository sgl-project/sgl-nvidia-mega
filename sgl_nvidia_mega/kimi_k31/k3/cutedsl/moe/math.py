# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MoE-local stock arithmetic boundaries (independent of attention helpers)."""

import cutlass as c
import cutlass.cute as cute
from ..common.primitives import asm
from .shared import situ_gate, situ_up, situ_product


@asm(c.Float32, pure=True)
def reciprocal(value):
    return "rcp.approx.ftz.f32 $0,$1;", "=f,f", (value,)


@asm(c.Float32, pure=True)
def clamp_input(value):
    # The stock clmp_siTuGlu kernel defaults to FLT_MAX and preserves NaNs.
    code = "{ .reg .pred p; .reg .f32 v; setp.gt.ftz.f32 p,$1,0f7f7fffff; "
    code += "selp.f32 v,0f7f7fffff,$1,p; setp.lt.ftz.f32 p,v,0fff7fffff; "
    code += "selp.f32 $0,0fff7fffff,v,p; }"
    return code, "=f,f", (value,)


@cute.jit
def routed_situ(gate, up, beta, linear_beta, unused):
    """TensorRT-LLM FC1 SiTU: runtime FP32 parameter reciprocals, two branches.

    Keep beta*tanh(g/beta)*sigmoid(g) separate from lb*tanh(u/lb);
    folding beta*lb changes the FP32 operation graph before hidden quantization.
    The third scalar preserves the prepared three-scalar launch ABI.
    """
    gate, up = clamp_input(gate), clamp_input(up)
    g = situ_gate(gate, beta, reciprocal(beta))
    u = situ_up(up, linear_beta, reciprocal(linear_beta))
    return situ_product(u, g)


@cute.jit
def routed_situ_hoisted(gate, up, beta, linear_beta, inv_beta, inv_lb):
    """``routed_situ`` with the parameter reciprocals computed once per thread by the caller (the stock trtllm-gen
    FC1 epilogue computes them once too) and without the FLT_MAX input clamp, which is the identity on every finite
    input (FC1 outputs are finite): bitwise equal to ``routed_situ`` on every finite (gate, up)."""
    g = situ_gate(gate, beta, inv_beta)
    u = situ_up(up, linear_beta, inv_lb)
    return situ_product(u, g)


def situ_lean_enabled():
    """``K31MK_EXP_SITU_LEAN=1`` (default 0): the routed SiTU of the
    persistent experts and the shared SiTU of RouteSitu take their two parameter reciprocals once per thread (and the
    routed one drops the identity FLT_MAX clamp). Bitwise equal outputs; fewer MUFU.RCP / div.rn per element."""
    import os

    return os.environ.get("K31MK_EXP_SITU_LEAN", "0") == "1"


@asm(c.Float32, pure=True)
def latent_inv_rms(value, eps, *, m8=False):
    """Stock ar_fusion: RN mean, FP32 epsilon add, non-FTZ approximate rsqrt."""
    code = "{ .reg .f32 mean; div.rn.f32 mean,$1,0f45600000; "
    code += "add.f32 mean,mean,$2; rsqrt.approx.f32 $0,mean; }"
    return code, "=f,f,f", (c.Float32(value), c.Float32(eps))


@asm(c.Float32, pure=True)
def square_add(total, value):
    """Stock FP32 sum_eq += value*value (nvcc contracts the multiply/add)."""
    code = "fma.rn.f32 $0,$2,$2,$1;"
    return code, "=f,f,f", (total, value)
