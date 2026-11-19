# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""MoE-local stock arithmetic boundaries (independent of attention helpers)."""

import cutlass as c
import cutlass.cute as cute

from ..common.primitives import asm
from .shared import situ_gate, situ_product, situ_up


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
