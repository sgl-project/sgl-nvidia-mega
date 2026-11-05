# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Modified by NVIDIA: derived from FlashInfer 0.7.1's CuTe-DSL contiguous grouped GEMM + finalize kernel. Changes:
# the weight operand is read in the row order and block-scale layout of the TensorRT-LLM fused-MoE weight preparation
# (output columns are restored to natural order in the epilogue), a per-row operand-A scale replaces the routing
# weight, every valid row is written unweighted to its own row of the output (no scatter-add finalize), bounded
# barrier waits, architecture-specific MMA builders resolved at run time, test harness removed.

"""
Contiguous grouped block-scaled GEMM (second GEMM of a mixture-of-experts layer) writing per-row outputs.

  C[row, :] = alpha[expert(row)] * row_scale[row] * (SFA * A[row]) * (SFB * B[expert(row)])^T     (BF16)

A rows are in permuted (expert-grouped, tile-padded) order; every valid row r (r < the tile's mn_limit) is written to
row r of C, padding rows are not written. B is in the TensorRT-LLM fused-MoE preparation row order: within every block
of 32 output columns, accumulator column 8c + a holds output column 4a + c; the epilogue restores natural order.

With ``swiglu`` (first GEMM of the layer, B = the gate/up weight in the same preparation's gated row order) the epilogue
instead pairs, within every block of 32 accumulator columns, [up(i0 + 2a) | gate(i0 + 2a) | up(i0 + 2a + 1) |
gate(i0 + 2a + 1)] (a = 0..7, i0 = 16 * block) and writes C[row, i0 .. i0 + 15] = up * silu(gate) with
gate = acc * alpha[e] * row_scale[row] and up = acc * alpha_up[e] * row_scale[row]; C has N / 2 columns. With
``row_amax`` as well, the epilogue also reduces, per valid row, the largest |BF16 output value| of the row into
row_amax[row] (atomic max of the FP32 bit pattern; non-negative floats order as unsigned integers; the entries must be
zero before the launch).

With ``fuse_quant`` as well (pair-of-CTAs MMA only), the kernel also produces the per-token NVFP4 form of C (the input
of the second GEMM). The epilogue publishes the completion of each of its tiles on a per-row-block counter (after its
row-maximum updates) and hands each tile whose BF16 rows are written to two extra conversion warps through a queue in
shared memory. A conversion warp converts a region (128 rows x 128 columns) once its row block is complete (all N tiles
published, so the row maxima are final). The epilogue never waits for the conversion warps (a full queue drops the
tile), and once the epilogue has finished the conversion warps no longer wait for other CTAs: anything not converted
in the kernel keeps its done flag at 0 for a separate pass. The conversion is the per-row and per-16-value arithmetic
of the default recipe of FlashInfer's per-token NVFP4 quantizer (fast math, no 4over6 search), written out in PTX
here.
"""

from typing import NamedTuple, Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.tcgen05.mma import CollectorOp
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils.gemm.sm100 import (
    epilogue_smem_copy_and_partition,
    epilogue_tmem_copy_and_partition,
    transform_partitioned_tensor_layout,
)

from .arch import arch_helpers
from .bounded_wait import TEST_WAIT_SCALE, scaled_bound

# ============================================================================
# Inline utility functions
# ============================================================================


@dsl_user_op
def blk_copy_s2g(dst_gemm, src_smem, size, loc=None, ip=None):
    """Bulk copy of ``size`` bytes from shared to global memory (bulk-group completion)."""
    llvm.inline_asm(
        None,
        [
            dst_gemm.iterator.llvm_ptr,
            src_smem.iterator.llvm_ptr,
            size.ir_value(),
        ],
        "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2;",
        "l,l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def abs_max_f32(acc, x, loc=None, ip=None):
    """max(acc, |x|) with PTX max.f32 (a NaN operand yields the other operand; subnormals kept)."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [
                cutlass.Float32(acc).ir_value(loc=loc, ip=ip),
                cutlass.Float32(x).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t.reg .f32 t;\n\tabs.f32 t, $2;\n\tmax.f32 $0, $1, t;\n\t}",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def red_max_bf16_bits(dst, x, loc=None, ip=None):
    """Atomic max of the FP32 bit pattern of BF16(x) (round to nearest even, x >= 0) into the 32-bit word at dst."""
    llvm.inline_asm(
        None,
        [dst.iterator.llvm_ptr, cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b16 h, z;\n\t.reg .b32 v;\n\tcvt.rn.bf16.f32 h, $1;\n\tmov.b16 z, 0;\n\t"
        "mov.b32 v, {z, h};\n\tred.relaxed.gpu.global.max.u32 [$0], v;\n\t}",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _asm(ret, args, text, cons, loc, ip):
    return llvm.inline_asm(
        ret,
        args,
        text,
        cons,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def gaddr(t, loc=None, ip=None):
    """Address of a global tensor's first element as Int64."""
    return cutlass.Int64(llvm.ptrtoint(T.i64(), t.iterator.llvm_ptr, loc=loc, ip=ip))


@dsl_user_op
def ld_v4(addr, loc=None, ip=None):
    """Four 32-bit words from global memory (16-byte aligned address)."""
    r = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc,
        ip,
    )
    return [
        cutlass.Int32(llvm.extractvalue(T.i32(), r, [i], loc=loc, ip=ip))
        for i in range(4)
    ]


@dsl_user_op
def st_v4(addr, a, b, c, d, loc=None, ip=None):
    """Store four 32-bit words to global memory (16-byte aligned address)."""
    _asm(
        None,
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)]
        + [cutlass.Int32(v).ir_value(loc=loc, ip=ip) for v in (a, b, c, d)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc,
        ip,
    )


@dsl_user_op
def st_b32(addr, v, loc=None, ip=None):
    """Store one 32-bit word to global memory."""
    _asm(
        None,
        [
            cutlass.Int64(addr).ir_value(loc=loc, ip=ip),
            cutlass.Int32(v).ir_value(loc=loc, ip=ip),
        ],
        "st.global.b32 [$0], $1;",
        "l,r",
        loc,
        ip,
    )


@dsl_user_op
def ld_acquire_u32(addr, loc=None, ip=None):
    """32-bit load with acquire semantics at GPU scope."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.acquire.gpu.global.u32 $0, [$1];",
            "=r,l",
            loc,
            ip,
        )
    )


@dsl_user_op
def ld_relaxed_u32(addr, loc=None, ip=None):
    """32-bit load, relaxed at GPU scope."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.relaxed.gpu.global.u32 $0, [$1];",
            "=r,l",
            loc,
            ip,
        )
    )


@dsl_user_op
def smem_addr(t, loc=None, ip=None):
    """Shared-memory address of a shared tensor's first element as Int32."""
    return cutlass.Int32(
        llvm.ptrtoint(T.i32(), t.iterator.to_llvm_ptr(loc=loc, ip=ip), loc=loc, ip=ip)
    )


@dsl_user_op
def cp_async_16(saddr, gaddr_, loc=None, ip=None):
    """16-byte asynchronous copy global -> shared (L2 only)."""
    _asm(
        None,
        [
            cutlass.Int32(saddr).ir_value(loc=loc, ip=ip),
            cutlass.Int64(gaddr_).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.cg.shared.global [$0], [$1], 16;",
        "r,l",
        loc,
        ip,
    )


@dsl_user_op
def cp_async_commit(loc=None, ip=None):
    _asm(None, [], "cp.async.commit_group;", "", loc, ip)


@dsl_user_op
def cp_async_wait_all(loc=None, ip=None):
    _asm(None, [], "cp.async.wait_group 0;", "", loc, ip)


@dsl_user_op
def lds_v4(saddr, loc=None, ip=None):
    """Four 32-bit words from shared memory (16-byte aligned address)."""
    r = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [cutlass.Int32(saddr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc,
        ip,
    )
    return [
        cutlass.Int32(llvm.extractvalue(T.i32(), r, [i], loc=loc, ip=ip))
        for i in range(4)
    ]


@dsl_user_op
def ld_acquire_cta_s32(saddr, loc=None, ip=None):
    """Shared 32-bit load, acquire at CTA scope."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [cutlass.Int32(saddr).ir_value(loc=loc, ip=ip)],
            "ld.acquire.cta.shared::cta.u32 $0, [$1];",
            "=r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def st_release_cta_s32(saddr, v, loc=None, ip=None):
    """Shared 32-bit store, release at CTA scope."""
    _asm(
        None,
        [
            cutlass.Int32(saddr).ir_value(loc=loc, ip=ip),
            cutlass.Int32(v).ir_value(loc=loc, ip=ip),
        ],
        "st.release.cta.shared::cta.u32 [$0], $1;",
        "r,r",
        loc,
        ip,
    )


@dsl_user_op
def ld_cg_v4(addr, loc=None, ip=None):
    """Four 32-bit words from global memory through L2 only (16-byte aligned address)."""
    r = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.cg.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc,
        ip,
    )
    return [
        cutlass.Int32(llvm.extractvalue(T.i32(), r, [i], loc=loc, ip=ip))
        for i in range(4)
    ]


@dsl_user_op
def nanosleep_hint(loc=None, ip=None):
    """Suspend the warp briefly (PTX nanosleep, a hint)."""
    _asm(None, [], "nanosleep.u32 500;", "", loc, ip)


@dsl_user_op
def blk_wait_older(loc=None, ip=None):
    """Wait until every committed bulk copy of this thread but the most recent group has completed, then order
    those writes for the generic proxy."""
    _asm(
        None,
        [],
        "cp.async.bulk.wait_group 1;\n\tfence.proxy.async.global;",
        "",
        loc,
        ip,
    )


@dsl_user_op
def blk_wait_all_fenced(loc=None, ip=None):
    """Wait until every committed bulk copy of this thread has completed, then order them for the generic proxy."""
    _asm(
        None,
        [],
        "cp.async.bulk.wait_group 0;\n\tfence.proxy.async.global;",
        "",
        loc,
        ip,
    )


@dsl_user_op
def publish_add(addr, loc=None, ip=None):
    """Add 1 to the counter at addr with release semantics at GPU scope."""
    _asm(
        None,
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
        "red.release.gpu.global.add.u32 [$0], 1;",
        "l",
        loc,
        ip,
    )


@dsl_user_op
def pack_bf16x2(lo, hi, loc=None, ip=None):
    """BF16 pair (round to nearest even) of two FP32 values, lo in the low half."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [
                cutlass.Float32(lo).ir_value(loc=loc, ip=ip),
                cutlass.Float32(hi).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            loc,
            ip,
        )
    )


# 1 / (e4m3 max * e2m1 max) as FP32 bits (the activations' per-token global-scale recipe).
_PTS_INV = "0f39C30C31"


@dsl_user_op
def token_scales(amax_bits, loc=None, ip=None):
    """(per-token scale = row max * inverse global scale, encode scale = its approximate reciprocal or 0)."""
    r = _asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [cutlass.Int32(amax_bits).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .f32 a, p, r;\n\t.reg .pred q;\n\tmov.b32 a, $2;\n\t"
        f"mul.ftz.f32 p, a, {_PTS_INV};\n\t"
        "setp.neu.ftz.f32 q, p, 0f00000000;\n\trcp.approx.ftz.f32 r, p;\n\t"
        "selp.f32 $1, r, 0f00000000, q;\n\tmov.f32 $0, p;\n\t}",
        "=f,=f,r",
        loc,
        ip,
    )
    return (
        cutlass.Float32(llvm.extractvalue(T.f32(), r, [0], loc=loc, ip=ip)),
        cutlass.Float32(llvm.extractvalue(T.f32(), r, [1], loc=loc, ip=ip)),
    )


def _qblock_text():
    lines = [
        ".reg .b32 a<8>, m, u, v;",
        ".reg .f32 fl, fh, vmax, sf, sfd, rc, t, os, z, f0, f1;",
        ".reg .b16 e8, h0, h1;",
        ".reg .b32 e16;",
        ".reg .pred gt, nz;",
        ".reg .b8 b<8>;",
    ]
    lines += [f"and.b32 a{i}, ${3 + i}, 0x7FFF7FFF;" for i in range(8)]
    lines += ["max.bf16x2 m, a0, a1;"] + [
        f"max.bf16x2 m, m, a{i};" for i in range(2, 8)
    ]
    lines += [
        "shl.b32 u, m, 16;",
        "mov.b32 fl, u;",
        "and.b32 v, m, 0xFFFF0000;",
        "mov.b32 fh, v;",
        "setp.gt.f32 gt, fl, fh;",
        "selp.f32 vmax, fl, fh, gt;",
        "mul.ftz.f32 sf, vmax, 0f3E2AAAAB;",
        "mul.ftz.f32 sf, sf, $11;",
        "mov.f32 z, 0f00000000;",
        "cvt.rn.satfinite.e4m3x2.f32 e8, z, sf;",
        "cvt.rn.f16x2.e4m3x2 e16, e8;",
        "mov.b32 {h0, h1}, e16;",
        "cvt.f32.f16 sfd, h0;",
        "setp.neu.ftz.f32 nz, vmax, 0f00000000;",
        "rcp.approx.ftz.f32 rc, $11;",
        "mul.ftz.f32 t, sfd, rc;",
        "rcp.approx.ftz.f32 os, t;",
        "selp.f32 os, os, 0f00000000, nz;",
    ]
    for i in range(8):
        lines += [
            f"shl.b32 u, ${3 + i}, 16;",
            "mov.b32 f0, u;",
            f"and.b32 v, ${3 + i}, 0xFFFF0000;",
            "mov.b32 f1, v;",
            "mul.ftz.f32 f0, f0, os;",
            "mul.ftz.f32 f1, f1, os;",
            f"cvt.rn.satfinite.e2m1x2.f32 b{i}, f1, f0;",
        ]
    lines += [
        "mov.b32 $0, {b0, b1, b2, b3};",
        "mov.b32 $1, {b4, b5, b6, b7};",
        "cvt.u32.u16 u, e8;",
        "and.b32 $2, u, 0xFF;",
    ]
    return "{\n\t" + "\n\t".join(lines) + "\n\t}"


_QBLOCK_ASM = _qblock_text()


@dsl_user_op
def f32_bits(x, loc=None, ip=None):
    """FP32 bit pattern of x as Int32."""
    return cutlass.Int32(
        llvm.bitcast(
            T.i32(), cutlass.Float32(x).ir_value(loc=loc, ip=ip), loc=loc, ip=ip
        )
    )


@dsl_user_op
def quant_block16(words, enc, loc=None, ip=None):
    """NVFP4 conversion of 16 BF16 values (8 words, element 2i in the low half of word i) with the encode scale:
    (payload word of elements 0-7, payload word of elements 8-15, E4M3 scale byte)."""
    r = _asm(
        llvm.StructType.get_literal([T.i32()] * 3),
        [cutlass.Int32(w).ir_value(loc=loc, ip=ip) for w in words]
        + [cutlass.Float32(enc).ir_value(loc=loc, ip=ip)],
        _QBLOCK_ASM,
        "=r,=r,=r," + ",".join(["r"] * 8) + ",f",
        loc,
        ip,
    )
    return [
        cutlass.Int32(llvm.extractvalue(T.i32(), r, [i], loc=loc, ip=ip))
        for i in range(3)
    ]


def _hmax8_text():
    lines = [".reg .b32 a<4>, m, u, v;", ".reg .f32 fl, fh;", ".reg .pred gt;"]
    lines += [f"and.b32 a{i}, ${1 + i}, 0x7FFF7FFF;" for i in range(4)]
    lines += [
        "max.bf16x2 m, a0, a1;",
        "max.bf16x2 m, m, a2;",
        "max.bf16x2 m, m, a3;",
        "shl.b32 u, m, 16;",
        "mov.b32 fl, u;",
        "and.b32 v, m, 0xFFFF0000;",
        "mov.b32 fh, v;",
        "setp.gt.f32 gt, fl, fh;",
        "selp.f32 $0, fl, fh, gt;",
    ]
    return "{\n\t" + "\n\t".join(lines) + "\n\t}"


def _qhalf_text():
    lines = [
        ".reg .b32 u, v;",
        ".reg .f32 sf, sfd, rc, t, os, z, f0, f1;",
        ".reg .b16 e8, h0, h1;",
        ".reg .b32 e16;",
        ".reg .pred nz;",
        ".reg .b8 b<4>;",
        "mul.ftz.f32 sf, $6, 0f3E2AAAAB;",
        "mul.ftz.f32 sf, sf, $7;",
        "mov.f32 z, 0f00000000;",
        "cvt.rn.satfinite.e4m3x2.f32 e8, z, sf;",
        "cvt.rn.f16x2.e4m3x2 e16, e8;",
        "mov.b32 {h0, h1}, e16;",
        "cvt.f32.f16 sfd, h0;",
        "setp.neu.ftz.f32 nz, $6, 0f00000000;",
        "rcp.approx.ftz.f32 rc, $7;",
        "mul.ftz.f32 t, sfd, rc;",
        "rcp.approx.ftz.f32 os, t;",
        "selp.f32 os, os, 0f00000000, nz;",
    ]
    for i in range(4):
        lines += [
            f"shl.b32 u, ${2 + i}, 16;",
            "mov.b32 f0, u;",
            f"and.b32 v, ${2 + i}, 0xFFFF0000;",
            "mov.b32 f1, v;",
            "mul.ftz.f32 f0, f0, os;",
            "mul.ftz.f32 f1, f1, os;",
            f"cvt.rn.satfinite.e2m1x2.f32 b{i}, f1, f0;",
        ]
    lines += [
        "mov.b32 $0, {b0, b1, b2, b3};",
        "cvt.u32.u16 u, e8;",
        "and.b32 $1, u, 0xFF;",
    ]
    return "{\n\t" + "\n\t".join(lines) + "\n\t}"


_HMAX8_ASM = _hmax8_text()
_QHALF_ASM = _qhalf_text()


@dsl_user_op
def absmax8(words, loc=None, ip=None):
    """max |x| of 8 BF16 values (4 words) as FP32 (BF16 max with PTX max.bf16x2, NaN operands dropped)."""
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Int32(w).ir_value(loc=loc, ip=ip) for w in words],
            _HMAX8_ASM,
            "=f,r,r,r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def quant_half8(words, vmax, enc, loc=None, ip=None):
    """NVFP4 conversion of 8 of a block's 16 BF16 values (4 words) given the block max and the encode scale:
    (payload word, E4M3 scale byte)."""
    r = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [cutlass.Int32(w).ir_value(loc=loc, ip=ip) for w in words]
        + [
            cutlass.Float32(vmax).ir_value(loc=loc, ip=ip),
            cutlass.Float32(enc).ir_value(loc=loc, ip=ip),
        ],
        _QHALF_ASM,
        "=r,=r,r,r,r,r,f,f",
        loc,
        ip,
    )
    return [
        cutlass.Int32(llvm.extractvalue(T.i32(), r, [i], loc=loc, ip=ip))
        for i in range(2)
    ]


@dsl_user_op
def fmax_f32(a, b, loc=None, ip=None):
    """a > b ? a : b."""
    return cutlass.Float32(
        _asm(
            T.f32(),
            [
                cutlass.Float32(a).ir_value(loc=loc, ip=ip),
                cutlass.Float32(b).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t.reg .pred p;\n\tsetp.gt.f32 p, $1, $2;\n\tselp.f32 $0, $1, $2, p;\n\t}",
            "=f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def blk_commit_and_wait_read(loc=None, ip=None):
    """Commit this thread's bulk copies and wait until their shared-memory sources have been read."""
    llvm.inline_asm(
        None,
        [],
        "cp.async.bulk.commit_group;\n\tcp.async.bulk.wait_group.read 0;",
        "",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def blk_wait_all(loc=None, ip=None):
    """Wait until every committed bulk copy of this thread has completed."""
    llvm.inline_asm(
        None,
        [],
        "cp.async.bulk.wait_group 0;",
        "",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


# ----------------------------------------------------------------------------------------------------------------------
# Claimed work units (``sched`` "dyn" / "local", see GroupedGemmRowsKernel). Each helper is executed by all 32 lanes
# of the scheduler warp; lane 0 does the memory operation and the value is broadcast to the warp.
# ----------------------------------------------------------------------------------------------------------------------

# Mailbox word: [tag : 12 | m tile : 14 | n tile : 6]; tag = (sequence number + 1) mod 4096 in slot sequence % MBOX_RING.
# A slot is first waited for with a tag in 1..MBOX_RING (a zeroed slot never matches) and afterwards holds the tag of
# the sequence MBOX_RING earlier (never equal). The first CTA of a pair is at most a few units ahead of the second
# (it can only run ahead of the units its own MMA has consumed by the tile-info and accumulator stages, and that MMA
# needs the second CTA's operand loads of the same unit), far below MBOX_RING. The m / n fields of DONE are all ones.
MBOX_RING = 16
MBOX_DONE = 0xFFFFF
# Words before the first mailbox in a control block: [0] / [1] claim counters of queues 0 / 1; [2 + d] units that
# clusters of domain d took from their own queue, [4 + d] units they took from the other queue (record only).
SCHED_HEAD_WORDS = 32
# Bound of one mailbox wait (polls), then trap (x the test-only wait scale of bounded_wait).
MBOX_MAX_POLLS = scaled_bound(1 << 22, TEST_WAIT_SCALE)


@dsl_user_op
def claim_unit(addr, loc=None, ip=None):
    """Lane 0: atom.add.relaxed.gpu of 1 on the u32 counter at addr; the previous value to every lane."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .pred p;\n\t.reg .u32 l, v;\n\tmov.u32 v, 0;\n\tmov.u32 l, %laneid;\n\tsetp.eq.u32 p, l, 0;\n\t"
            "@p atom.relaxed.gpu.global.add.u32 v, [$1], 1;\n\t"
            "shfl.sync.idx.b32 $0, v, 0, 31, 0xffffffff;\n\t}",
            "=r,l",
            loc,
            ip,
        )
    )


@dsl_user_op
def sm_domain(table_addr, loc=None, ip=None):
    """Locality domain of the SM this thread runs on: table[%smid] (u32 entries)."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [cutlass.Int64(table_addr).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .u32 s;\n\t.reg .u64 a;\n\tmov.u32 s, %smid;\n\tmul.wide.u32 a, s, 4;\n\tadd.u64 a, a, $1;\n\t"
            "ld.global.u32 $0, [a];\n\t}",
            "=r,l",
            loc,
            ip,
        )
    )


@dsl_user_op
def lane0_red_add(addr, v, loc=None, ip=None):
    """Lane 0: red.add.relaxed.gpu of v to the u32 at addr."""
    _asm(
        None,
        [
            cutlass.Int64(addr).ir_value(loc=loc, ip=ip),
            cutlass.Int32(v).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .pred p;\n\t.reg .u32 l;\n\tmov.u32 l, %laneid;\n\tsetp.eq.u32 p, l, 0;\n\t"
        "@p red.relaxed.gpu.global.add.u32 [$0], $1;\n\t}",
        "l,r",
        loc,
        ip,
    )


@dsl_user_op
def named_barrier_sync_unaligned(barrier_id, num_threads, loc=None, ip=None):
    """barrier.sync without .aligned: a named barrier whose participating warps reach it from different instructions."""
    _asm(
        None,
        [
            cutlass.Int32(barrier_id).ir_value(loc=loc, ip=ip),
            cutlass.Int32(num_threads).ir_value(loc=loc, ip=ip),
        ],
        "barrier.sync $0, $1;",
        "r,r",
        loc,
        ip,
    )


@dsl_user_op
def mbox_post(addr, word, loc=None, ip=None):
    """Lane 0: st.release.gpu of one mailbox word (the word carries all of its data)."""
    _asm(
        None,
        [
            cutlass.Int64(addr).ir_value(loc=loc, ip=ip),
            cutlass.Int32(word).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .pred p;\n\t.reg .u32 l;\n\tmov.u32 l, %laneid;\n\tsetp.eq.u32 p, l, 0;\n\t"
        "@p st.release.gpu.global.u32 [$0], $1;\n\t}",
        "l,r",
        loc,
        ip,
    )


@dsl_user_op
def mbox_wait(addr, tag, loc=None, ip=None):
    """Poll the mailbox word at addr (lane 0, ld.acquire.gpu; broadcast) until its tag field equals tag; the word to
    every lane. At most MBOX_MAX_POLLS polls, then trap."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [
                cutlass.Int64(addr).ir_value(loc=loc, ip=ip),
                cutlass.Int32(tag).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t.reg .pred p, q;\n\t.reg .u32 l, v, t, n;\n\tmov.u32 n, 0;\n\tmov.u32 v, 0;\n\t"
            "mov.u32 l, %laneid;\n\tsetp.eq.u32 q, l, 0;\n\t"
            "MBOX_POLL:\n\t"
            "@q ld.acquire.gpu.global.u32 v, [$1];\n\t"
            "shfl.sync.idx.b32 v, v, 0, 31, 0xffffffff;\n\t"
            "shr.u32 t, v, 20;\n\tsetp.eq.u32 p, t, $2;\n\t@p bra MBOX_DONE;\n\t"
            "nanosleep.u32 64;\n\tadd.u32 n, n, 1;\n\t"
            f"setp.lt.u32 p, n, {MBOX_MAX_POLLS};\n\t@p bra MBOX_POLL;\n\ttrap;\n\t"
            "MBOX_DONE:\n\tmov.u32 $0, v;\n\t}",
            "=r,l,r",
            loc,
            ip,
        )
    )


class S2TCopyBundle(NamedTuple):
    """Bundle of tiled copy and partitioned tensors for smem-to-tmem copies."""

    tiled_copy: cute.TiledCopy
    sSF_compact: cute.Tensor  # Partitioned source (smem)
    tSF_compact: cute.Tensor  # Partitioned destination (tmem)


class GroupedGemmRowsKernel:
    """Contiguous grouped block-scaled GEMM writing scaled per-row outputs (see module docstring).

    Persistent tile scheduling with a dedicated scheduler warp, warp specialization (scheduler, TMA, MMA, epilogue),
    B-reuse (Bkeep-Breuse), per-expert alpha times per-row scale.

    :param sf_vec_size: Scale factor vector size (16)
    :param mma_inst_shape: Shape of MMA instruction (M, N, K); M must be 128 (1-CTA)
    :param mma_tiler: Shape of MMA tiler (M, N, K)
    :param cluster_shape_mn: Cluster dimensions (M, N)
    :param arch: target architecture string (from the device's compute capability at run time)
    :param raster_along_m: If True, raster tiles along M dimension first
    :param sched: how clusters get their work units (one unit = one MMA tile of 256 or 128 rows x tile N):
        "static" (default): the persistent static schedule (cluster c takes units c, c + clusters, ...);
        "dyn": every cluster claims its next unit from one counter (one atomic add per unit), in the static order
        (M tile major, N tile minor), so a cluster that starts late or runs slowly takes fewer units;
        "local": two such queues, queue d holding the units whose N tile lies in the d-th half of the weight rows
        (the halves an expert-weight placement may back on locality domain d); a cluster claims from the queue of
        the domain of the SM it runs on (a per-SM table) and, once that queue is empty, from the other one.
        Every unit is computed exactly as in the static schedule (same tile, same K order, same epilogue), so the
        outputs are bitwise identical for every schedule. With the pair-of-CTAs MMA the first CTA of the pair claims
        and hands each unit to the second CTA through a mailbox word in global memory (tagged with the sequence
        number). Only the "blocks" / "rows" quantizer forms (not fuse_quant) and clusters along M only.
    """

    SCHEDS = ("static", "dyn", "local")

    def __init__(
        self,
        sf_vec_size: int,
        mma_inst_shape: Tuple[int, int, int],
        mma_tiler: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        arch: str,
        raster_along_m: bool = False,
        swiglu: bool = False,
        row_amax: bool = False,
        fuse_quant: bool = False,
        tma_store: bool = True,
        sched: str = "static",
        sched_twin: bool = False,
        tmem_bar_unaligned: bool = False,
    ):
        if row_amax and not swiglu:
            raise ValueError("row_amax needs the SwiGLU epilogue")
        if fuse_quant and not (
            row_amax and mma_inst_shape[0] == 256 and mma_tiler[0] == 256
        ):
            raise ValueError(
                "fuse_quant needs row_amax and the pair-of-CTAs MMA with tile M 256"
            )
        if sched not in self.SCHEDS:
            raise ValueError(f"sched must be one of {self.SCHEDS}")
        if sched != "static" and (
            fuse_quant or raster_along_m or cluster_shape_mn[1] != 1
        ):
            raise ValueError(
                "claimed schedules need fuse_quant off, N rastering and clusters along M only"
            )
        if sched_twin and sched == "static":
            raise ValueError("sched_twin needs a claimed schedule")
        self.sched = sched
        # test only: units of N tile 0 use the next expert's weights (a schedule test must detect it)
        self.sched_twin = sched_twin
        # The library's tensor-memory handshake has the MMA warp and the epilogue warps reach the same named barrier from
        # two different instructions in its aligned form. With this set the handshake runs at ONE shared site for those
        # five warps, through one non-aligned named-barrier instruction. No value changes.
        if tmem_bar_unaligned and fuse_quant:
            raise ValueError("tmem_bar_unaligned is not available with fuse_quant")
        self.tmem_bar_unaligned = tmem_bar_unaligned
        self.tmem_cols_trace: dict = {}  # trace-time record: site ("alloc" / "free") -> column counts passed
        self.swiglu = swiglu
        self.row_amax_out = row_amax
        self.fuse_quant = fuse_quant
        # output tiles leave through tensor-memory-access stores of (128-row x epilogue-tile) staging buffers;
        # otherwise (and with fuse_quant) one bulk copy per valid row
        self.tma_store = tma_store and not fuse_quant
        self.sf_vec_size = sf_vec_size
        self.acc_dtype = cutlass.Float32
        self.mma_inst_shape = mma_inst_shape
        self.mma_tiler = mma_tiler
        self.cluster_shape_mn = cluster_shape_mn
        self.raster_along_m = raster_along_m

        self.use_2cta_instrs = mma_inst_shape[0] == 256
        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )
        self.arch = arch
        self.smem_capacity = utils.get_smem_capacity_in_bytes(self.arch)
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols(self.arch)

        self.occupancy = 1

        # Warp IDs for warp specialization
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.sched_warp_id = 6
        # fused conversion: two more warps
        self.conv_warp_ids = (7, 8) if fuse_quant else ()

        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.epilog_warp_id,
                self.mma_warp_id,
                self.tma_warp_id,
                self.sched_warp_id,
                *self.conv_warp_ids,
            )
        )
        self.threads_wo_sched = self.threads_per_warp * len(
            (
                *self.epilog_warp_id,
                self.mma_warp_id,
                self.tma_warp_id,
            )
        )

        # Set barriers for synchronization
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_warp * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.threads_per_warp
            * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        self.sched_sync_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=self.threads_per_warp,
        )

        # For epilogue compatibility
        self.epilogue_warp_id = self.epilog_warp_id
        self.epilog_sync_bar_id = self.epilog_sync_barrier.barrier_id

        # B-reuse pattern control
        self.enable_breuse = True if mma_tiler[0] // mma_inst_shape[0] == 2 else False

    def _get_mma_permutation_mnk(self):
        if cutlass.const_expr(self.use_2cta_instrs and self.enable_breuse):
            m_layout = cute.make_layout(
                shape=(self.mma_inst_shape[0] // 2, 2, 2),
                stride=(1, self.mma_inst_shape[0], self.mma_inst_shape[0] // 2),
            )
            return (m_layout, self.mma_inst_shape[1], self.mma_inst_shape[2])
        else:
            return (1, 1, 1)

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        cta_tile: Tuple[int, int, int],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        smem_capacity: int,
        occupancy: int,
        with_breuse: bool,
        tma_store_epi_n: int = 0,
    ) -> Tuple[int, int, int, int]:
        """Compute the number of stages for A/B/C/tile_info operands."""
        # ACC stages
        # Assumes the kernel has the whole tensor memory of the SM (one CTA per SM).
        num_acc_stage = 1 if (with_breuse and mma_tiler_mnk[1] in {192, 256}) else 2

        # Default C stages and tile info stages (5 elements now: bidx, bidy, bidz, valid, mn_limit)
        num_c_stage = 1  # Thinking about it
        num_tile_stage = 2

        # Calculate smem layout and size for one stage
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma, mma_tiler_mnk, a_dtype, 1
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma, mma_tiler_mnk, b_dtype, 1
        )
        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, mma_tiler_mnk, sf_vec_size, 1
        )
        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, mma_tiler_mnk, sf_vec_size, 1
        )

        # Shared memory for epilogue block reduce (if enabled)
        swizzled_pad = 16 // (c_dtype.width // 8)
        c_smem_layout_staged_one = cute.make_layout(
            (cta_tile[0], cta_tile[1]), stride=(cta_tile[1] + swizzled_pad, 1)
        )

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        )
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage
        if tma_store_epi_n:
            num_c_stage = 2
            c_bytes = (
                num_c_stage * cta_tile[0] * tma_store_epi_n * (c_dtype.width // 8)
                + 1024
            )

        # Calculate A/B/SFA/SFB stages
        num_ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        return num_acc_stage, num_ab_stage, num_c_stage, num_tile_stage

    def _setup_attributes(self):
        """Set up configurations dependent on GEMM inputs."""
        # Compute mma instruction shapes
        self.mma_inst_shape_sfb = (
            self.mma_inst_shape[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape[1], 128),
            self.mma_inst_shape[2],
        )

        tiled_mma = arch_helpers().make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
            atom_layout_mnk=(1, 1, 1),
            permutation_mnk=self._get_mma_permutation_mnk(),
        )

        tiled_mma_sfb = arch_helpers().make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_sfb,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
        )

        # Compute mma/cluster/tile shapes
        self.mma_tiler_sfb = (
            self.mma_inst_shape_sfb[0],
            self.mma_inst_shape_sfb[1],
            self.mma_tiler[2],
        )

        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        # columns of the C staging buffer: the output tile width (half the MMA tile in SwiGLU mode); the rest of the
        # shared memory goes to operand stages
        self.c_smem_cols = (
            self.cta_tile_shape_mnk[1] // 2
            if self.swiglu
            else self.cta_tile_shape_mnk[1]
        )
        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # Compute number of multicast CTAs
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        # Compute epilogue subtile
        self.epi_tile = arch_helpers().compute_epilogue_tile_shape(
            tiled_mma.op,
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )
        if self.swiglu:
            # one output subtile of 64 columns reads two 64-column accumulator subtiles
            self.epi_tile = (128, 64)
        self.epi_tile_n = cute.size(self.epi_tile[1])
        if self.epi_tile_n % 32 != 0:
            raise ValueError(
                f"epilogue tile N {self.epi_tile_n} is not a multiple of 32 (column restore)"
            )

        # Setup stage counts
        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_c_stage,
            self.num_tile_stage,
        ) = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.c_dtype,
            (self.cta_tile_shape_mnk[0], self.c_smem_cols, self.cta_tile_shape_mnk[2]),
            self.sf_dtype,
            self.sf_vec_size,
            self.smem_capacity,
            self.occupancy,
            self.enable_breuse,
            self.epi_tile_n if self.tma_store else 0,
        )

        # Compute A/B/SFA/SFB/C shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage
        )

        # C smem layout for block reduce (if enabled)
        swizzled_pad = 16 // (self.c_dtype.width // 8)
        if self.tma_store:
            self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
                self.c_dtype,
                self.c_layout,
                (self.cta_tile_shape_mnk[0], self.epi_tile_n),
                self.num_c_stage,
            )
        else:
            self.c_smem_layout_staged = cute.make_layout(
                (self.cta_tile_shape_mnk[0], self.c_smem_cols, self.num_c_stage),
                stride=(
                    self.c_smem_cols + swizzled_pad,
                    1,
                    self.cta_tile_shape_mnk[0] * (self.c_smem_cols + 8),
                ),
            )

        # Compute TMEM layouts for SFA/SFB
        self.tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0)),
        )
        self.tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0)),
        )

        # Compute TMEM column counts
        # Each column entry in TMEM is 32-bit wide, and so we recast the TMEM layout
        # from its original data type to a 32-bit wide data type. Moreover, TMEM
        # addresses are expressed as (row << 16) | col, which in CUTE are expressed
        # as an affine transformation row * (1<<16) + col, which can be seen as a CUTE
        # layout of (row, col):(1<<16, 1). As a result, by masking out the upper 16 bits
        # (keeping only the lower 16 bits), we extract the cosize corresponding
        # to only the columns.
        self.num_sfa_tmem_cols = (
            cute.cosize(cute.recast_layout(32, self.sf_dtype.width, self.tCtSFA_layout))
            & 0x0000FFFF
        )
        self.num_sfb_tmem_cols = (
            cute.cosize(cute.recast_layout(32, self.sf_dtype.width, self.tCtSFB_layout))
            & 0x0000FFFF
        )
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.num_accumulator_tmem_cols = (
            self.cta_tile_shape_mnk[1]
            * self.num_acc_stage
            * (2 if self.enable_breuse else 1)
        )

        # Epilogue vectorization config
        if cutlass.const_expr(self.c_dtype == cutlass.BFloat16):
            self.element_offset = 8
            self.epi_loop_size = self.epi_tile_n // 8
        elif cutlass.const_expr(self.c_dtype == cutlass.Float32):
            self.element_offset = 2
            self.epi_loop_size = self.epi_tile_n // 2
        else:
            self.element_offset = 1
            self.epi_loop_size = self.epi_tile_n

        # copy_size is in bytes for cp.reduce.async.bulk instruction
        self.out_tile_n = (
            self.cta_tile_shape_mnk[1] // 2
            if self.swiglu
            else self.cta_tile_shape_mnk[1]
        )
        # bytes between rows of sC (its padded row stride)
        self.c_smem_row_bytes = (self.c_smem_cols + 16 // (self.c_dtype.width // 8)) * (
            self.c_dtype.width // 8
        )
        self.copy_size = self.out_tile_n * (self.c_dtype.width // 8)

    def _tmem_cols(self, site: str, ncols: int) -> int:
        """Records (at trace time) the column count a tensor-memory allocate or free site passes and returns it
        unchanged; the unit checks that the free passes the allocated count."""
        self.tmem_cols_trace.setdefault(site, []).append(int(ncols))
        return ncols

    def _is_interleaved_utccp(self) -> bool:
        """Enable interleaving tcgen05.cp for Bkeep-Breuse case for 4xFP4 kernel."""
        return (
            self.a_dtype.width == 4 and self.b_dtype.width == 4 and self.enable_breuse
        )

    def _mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> S2TCopyBundle:
        """Make tiledCopy for smem to tmem load for scale factor tensor."""
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)

        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        def appendMNBroadcastMode(smem_layout: cute.Layout):
            mn_dim = cute.get(smem_layout, mode=[0, 0])
            mn_dim = cute.append(mn_dim, cute.make_layout((4), stride=(0)))
            layout = cute.append(
                cute.group_modes(mn_dim, 0), cute.get(smem_layout, mode=[0, 1])
            )
            layout = cute.append(
                cute.group_modes(layout, 0), cute.get(smem_layout, mode=[1])
            )
            layout = cute.append(layout, cute.get(smem_layout, mode=[2]))
            layout = cute.append(layout, cute.get(smem_layout, mode=[3]))
            return layout

        tCsSF_compact_bcast = cute.make_tensor(
            tCsSF_compact.iterator, appendMNBroadcastMode(tCsSF_compact.layout)
        )

        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact_bcast)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_
        )
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return S2TCopyBundle(tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t)

    def _mainloop_s2t_copies(
        self,
        stage_idx: int,
        sfa_s2t_bundle: S2TCopyBundle,
        sfb_s2t_bundle: S2TCopyBundle,
    ):
        """Copy SFA/SFB from smem to tmem."""
        s2t_stage_coord = (None, None, None, None, stage_idx)

        cute.copy(
            sfa_s2t_bundle.tiled_copy,
            sfa_s2t_bundle.sSF_compact[s2t_stage_coord],
            sfa_s2t_bundle.tSF_compact,
        )
        cute.copy(
            sfb_s2t_bundle.tiled_copy,
            sfb_s2t_bundle.sSF_compact[s2t_stage_coord],
            sfb_s2t_bundle.tSF_compact,
        )

    def _mainloop_s2t_interleaved_copies(
        self,
        k_block: int,
        stage_idx: int,
        sfa_s2t_bundle: S2TCopyBundle,
        sfb_s2t_bundle: S2TCopyBundle,
    ):
        """Interleaved tcgen05.cp for Bkeep-Breuse pattern."""
        s_sfa_crd_keep = (None, 0, None, k_block, stage_idx)
        s_sfa_crd_reuse = (None, 1, None, k_block, stage_idx)
        s_sfb_crd = (None, None, None, k_block, stage_idx)

        t_sfa_crd_keep = (None, 0, None, k_block)
        t_sfa_crd_reuse = (None, 1, None, k_block)
        t_sfb_crd = (None, None, None, k_block)

        cute.copy(
            sfa_s2t_bundle.tiled_copy,
            sfa_s2t_bundle.sSF_compact[s_sfa_crd_keep],
            sfa_s2t_bundle.tSF_compact[t_sfa_crd_keep],
        )
        cute.copy(
            sfb_s2t_bundle.tiled_copy,
            sfb_s2t_bundle.sSF_compact[s_sfb_crd],
            sfb_s2t_bundle.tSF_compact[t_sfb_crd],
        )
        cute.copy(
            sfa_s2t_bundle.tiled_copy,
            sfa_s2t_bundle.sSF_compact[s_sfa_crd_reuse],
            sfa_s2t_bundle.tSF_compact[t_sfa_crd_reuse],
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        tile_idx_to_group_idx: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        alpha: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        row_scale: cute.Tensor,
        alpha_up: cute.Tensor,
        row_amax: cute.Tensor,
        q_out: cute.Tensor,
        q_sf: cute.Tensor,
        q_scale: cute.Tensor,
        q_ctl: cute.Tensor,
        c_out: cute.Tensor,
        sched_ctl: Optional[cute.Tensor] = None,
        sm_dom: Optional[cute.Tensor] = None,
    ):
        """Execute the contiguous grouped GEMM with finalize fusion.

        :param a: Input tensor A (permuted_m, k, 1)
        :param b: Input tensor B (n, k, l)
        :param c: Output tensor (seq_len, n, 1)
        :param sfa: Scale factor tensor A
        :param sfb: Scale factor tensor B
        :param tile_idx_to_group_idx: Mapping from tile index to group ID
        :param num_non_exiting_tiles: Number of valid tiles
        :param tile_idx_to_mn_limit: M limit for each tile
        :param alpha: Alpha tensor for each group
        :param max_active_clusters: Maximum number of active clusters
        :param stream: CUDA stream
        :param row_scale: operand-A scale per permuted row, shape (permuted_m,)
        :param sched_ctl: claimed schedules: control block (int32, zeroed before the launch: SCHED_HEAD_WORDS + MBOX_RING
            words per cluster)
        :param sm_dom: claimed schedule "local": locality domain per SM (int32, multiprocessor count entries)
        """
        # Setup static attributes
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa.element_type
        self.final_scale_dtype = cutlass.Float32
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.ROW_MAJOR  # Always N-major for GEMM output

        # Check data types
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        # Setup attributes
        self._setup_attributes()

        # Setup sfa/sfb tensor
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a.shape, self.sf_vec_size)
        sfa = cute.make_tensor(sfa.iterator, sfa_layout)

        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b.shape, self.sf_vec_size)
        sfb = cute.make_tensor(sfb.iterator, sfb_layout)

        atom_layout_mnk = (1, 1, 1)
        permutation_mnk = self._get_mma_permutation_mnk()

        tiled_mma = arch_helpers().make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
            atom_layout_mnk=atom_layout_mnk,
            permutation_mnk=permutation_mnk,
        )

        tiled_mma.set(tcgen05.Field.NEGATE_A, False)
        tiled_mma.set(tcgen05.Field.NEGATE_B, False)

        tiled_mma_sfb = arch_helpers().make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_sfb,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
        )
        tiled_mma_sfb.set(tcgen05.Field.NEGATE_A, False)
        tiled_mma_sfb.set(tcgen05.Field.NEGATE_B, False)

        tiled_mma_bkeep = None
        tiled_mma_breuse = None
        if cutlass.const_expr(self.enable_breuse):
            tiled_mma_bkeep = arch_helpers().make_blockscaled_trivial_tiled_mma(
                self.a_dtype,
                self.b_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                self.mma_inst_shape,
                a_collector_op=CollectorOp.DISCARD,
                b_collector_op=CollectorOp.FILL,
                atom_layout_mnk=atom_layout_mnk,
                permutation_mnk=permutation_mnk,
            )
            tiled_mma_bkeep.set(tcgen05.Field.NEGATE_A, False)
            tiled_mma_bkeep.set(tcgen05.Field.NEGATE_B, False)

            tiled_mma_breuse = arch_helpers().make_blockscaled_trivial_tiled_mma(
                self.a_dtype,
                self.b_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                self.mma_inst_shape,
                a_collector_op=CollectorOp.DISCARD,
                b_collector_op=CollectorOp.LASTUSE,
                atom_layout_mnk=atom_layout_mnk,
                permutation_mnk=permutation_mnk,
            )
            tiled_mma_breuse.set(tcgen05.Field.NEGATE_A, False)
            tiled_mma_breuse.set(tcgen05.Field.NEGATE_B, False)

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for B
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for SFA
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # Setup TMA load for SFB
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # Setup TMA store for the output (logical output columns; epilogue-tile boxes)
        tma_atom_c, tma_tensor_c = (
            cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                c_out,
                cute.slice_(self.c_smem_layout_staged, (None, None, 0)),
                (self.cta_tile_shape_mnk[0], self.epi_tile_n),
            )
            if self.tma_store
            else (None, None)
        )

        # Handle cta_tile_shape_n=192 case
        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
            x = tma_tensor_sfb.stride[0][1]
            y = cute.ceil_div(tma_tensor_sfb.shape[0][1], 4)

            new_shape = (
                (tma_tensor_sfb.shape[0][0], ((2, 2), y)),
                tma_tensor_sfb.shape[1],
                tma_tensor_sfb.shape[2],
            )
            x_times_3 = 3 * x
            new_stride = (
                (tma_tensor_sfb.stride[0][0], ((x, x), x_times_3)),
                tma_tensor_sfb.stride[1],
                tma_tensor_sfb.stride[2],
            )
            tma_tensor_sfb_new_layout = cute.make_layout(new_shape, stride=new_stride)
            tma_tensor_sfb = cute.make_tensor(
                tma_tensor_sfb.iterator, tma_tensor_sfb_new_layout
            )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

        # Compute grid size based on GEMM shape
        gemm_m = a.shape[0]
        gemm_n = b.shape[0]
        gemm_l = a.shape[2]
        gemm_shape = (gemm_m, gemm_n, gemm_l)

        self.tile_sched_params, grid = self._compute_grid(
            gemm_shape,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            max_active_clusters,
            self.raster_along_m,
        )

        self.buffer_align_bytes = 1024

        @cute.struct
        class SharedStorage:
            # (bidx, bidy, expert_idx, valid, mn_limit)
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 5 * self.num_tile_stage],
                1,
            ]
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_tile_stage * 2
            ]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            qflag: cute.struct.MemRange[cutlass.Int32, 8 + 3 * self.QFIFO]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(
                        self.c_smem_layout_staged.outer
                        if self.tma_store
                        else self.c_smem_layout_staged
                    ),
                ],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel
        self.kernel(
            tiled_mma,
            tiled_mma_bkeep,
            tiled_mma_breuse,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            c,
            tile_idx_to_group_idx,
            num_non_exiting_tiles,
            tile_idx_to_mn_limit,
            alpha,
            row_scale,
            alpha_up,
            row_amax,
            q_out,
            q_sf,
            q_scale,
            q_ctl,
            tma_atom_c,
            tma_tensor_c,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.tCtSFA_layout,
            self.tCtSFB_layout,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            sched_ctl,
            sm_dom,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_bkeep: Optional[cute.TiledMma],
        tiled_mma_breuse: Optional[cute.TiledMma],
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        c: cute.Tensor,
        tile_idx_to_group_idx: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        alpha: cute.Tensor,
        row_scale: cute.Tensor,
        alpha_up: cute.Tensor,
        row_amax: cute.Tensor,
        q_out: cute.Tensor,
        q_sf: cute.Tensor,
        q_scale: cute.Tensor,
        q_ctl: cute.Tensor,
        tma_atom_c: Optional[cute.CopyAtom],
        mC_out: Optional[cute.Tensor],
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        tCtSFA_layout: cute.Layout,
        tCtSFB_layout: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        sched_ctl: Optional[cute.Tensor],
        sm_dom: Optional[cute.Tensor],
    ):
        """GPU device kernel for contiguous grouped GEMM with finalize fusion."""
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Prefetch TMA descriptors
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        # Setup coordinates
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()

        # Allocate shared memory
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Initialize pipelines
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (
            2 if use_2cta_instrs else 1
        )
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Initialize tile info pipeline
        tile_info_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * 1,
        )
        tile_info_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_wo_sched,
        )
        tile_info_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.tile_info_mbar_ptr.data_ptr(),
            num_stages=self.num_tile_stage,
            producer_group=tile_info_pipeline_producer_group,
            consumer_group=tile_info_pipeline_consumer_group,
        )

        # Initialize tensor memory allocator
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
            arch=self.arch,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # Setup smem tensors
        if cutlass.const_expr(self.tma_store):
            sC = storage.sC.get_tensor(
                c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
            )
        else:
            sC = storage.sC.get_tensor(c_smem_layout_staged)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        info_layout = cute.make_layout((5, self.num_tile_stage), stride=(1, 5))
        sInfo = storage.sInfo.get_tensor(info_layout)
        sQflag = storage.qflag.get_tensor(cute.make_layout((8 + 3 * self.QFIFO,)))
        if cutlass.const_expr(self.fuse_quant):
            if tidx < 8:
                sQflag[tidx] = cutlass.Int32(0)
            cute.arch.barrier()  # every thread of the CTA, before any role starts

        # Compute multicast masks
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )

        # Local_tile partition global tensors
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )
        gC_mnl = cute.local_tile(
            c, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cutlass.Int32(cute.size(gA_mkl, mode=[3]))

        # Partition global tensors for TiledMMA
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)

        # Partition for TMA load
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        sfa_cta_layout = a_cta_layout
        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            sfa_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        # Partition for MMA
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        # Cluster wait before tensor memory alloc
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        if cutlass.const_expr(self.tmem_bar_unaligned):
            # One shared handshake site for the MMA warp and the epilogue warps: the allocator warp allocates, then all
            # five warps meet at ONE non-aligned named barrier instruction (the role blocks below skip their own).
            if warp_idx <= self.mma_warp_id:
                if warp_idx < self.mma_warp_id:
                    tmem.allocate(self._tmem_cols("alloc", self.num_tmem_alloc_cols))
                named_barrier_sync_unaligned(
                    self.tmem_alloc_barrier.barrier_id,
                    self.tmem_alloc_barrier.num_threads,
                )

        #
        # Specialized Scheduler warp
        #
        if warp_idx == self.sched_warp_id:
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            tile_info_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_tile_stage
            )

            num_valid_tiles = num_non_exiting_tiles[0]

            if cutlass.const_expr(self.sched != "static"):
                # Claimed units (class docstring). Unit = (M tile, N tile). "local": queue q holds the N tiles of the
                # q-th half of the weight rows; "dyn": one queue (q = 0) over all N tiles. Claim index k of queue q is
                # unit (k // n_queue, q * n_queue + k % n_queue): M tile major, as in the static order.
                pair = cute.size(tiled_mma.thr_id.shape)
                n_tiles = cutlass.Int32(
                    cute.size(c, mode=[1]) // self.cta_tile_shape_mnk[1]
                )  # (n is Int64 at the wrapper)
                n_queue = n_tiles
                if cutlass.const_expr(self.sched == "local"):
                    n_queue = n_tiles // 2
                ctl_base = gaddr(sched_ctl)
                mbox = ctl_base + cutlass.Int64(SCHED_HEAD_WORDS + bidz * MBOX_RING) * 4
                seq = cutlass.Int32(0)
                running = cutlass.Boolean(1)
                if is_leader_cta:
                    own = cutlass.Int32(0)
                    if cutlass.const_expr(self.sched == "local"):
                        own = sm_domain(gaddr(sm_dom)) & 1
                    cur = own
                    n_own = cutlass.Int32(0)
                    while running:
                        k = claim_unit(ctl_base + cutlass.Int64(cur) * 4)
                        m_tile = k // n_queue
                        if m_tile < num_valid_tiles:
                            n_tile = cur * n_queue + k % n_queue
                            # the second CTA of the pair gets the unit before this CTA's own pipeline slot is free
                            # (it never waits on this CTA's pipeline, so the pair cannot deadlock here)
                            if cutlass.const_expr(pair == 2):
                                mbox_post(
                                    mbox + cutlass.Int64(seq % MBOX_RING) * 4,
                                    (((seq + 1) & 0xFFF) << 20)
                                    | (m_tile << 6)
                                    | n_tile,
                                )
                            seq = seq + 1
                            if cur == own:
                                n_own = n_own + 1
                            expert_idx = tile_idx_to_group_idx[m_tile]
                            if cutlass.const_expr(self.sched_twin):
                                if n_tile == 0:
                                    expert_idx = expert_idx ^ 1
                            mn_limit = tile_idx_to_mn_limit[m_tile]
                            tile_info_pipeline.producer_acquire(
                                tile_info_producer_state
                            )
                            with cute.arch.elect_one():
                                sInfo[(0, tile_info_producer_state.index)] = (
                                    m_tile * pair + mma_tile_coord_v
                                )
                                sInfo[(1, tile_info_producer_state.index)] = n_tile
                                sInfo[(2, tile_info_producer_state.index)] = expert_idx
                                sInfo[(3, tile_info_producer_state.index)] = (
                                    cutlass.Int32(1)
                                )
                                sInfo[(4, tile_info_producer_state.index)] = mn_limit
                            cute.arch.fence_proxy("async.shared", space="cta")

                            self.sched_sync_barrier.arrive_and_wait()
                            tile_info_pipeline.producer_commit(tile_info_producer_state)
                            tile_info_producer_state.advance()
                        else:
                            if cutlass.const_expr(self.sched == "local"):
                                if cur == own:
                                    cur = 1 - own
                                else:
                                    running = cutlass.Boolean(0)
                            else:
                                running = cutlass.Boolean(0)
                    if cutlass.const_expr(pair == 2):
                        mbox_post(
                            mbox + cutlass.Int64(seq % MBOX_RING) * 4,
                            (((seq + 1) & 0xFFF) << 20) | MBOX_DONE,
                        )
                    # record: units this cluster took from its own / the other queue
                    lane0_red_add(ctl_base + cutlass.Int64(2 + own) * 4, n_own)
                    lane0_red_add(ctl_base + cutlass.Int64(4 + own) * 4, seq - n_own)
                else:
                    # second CTA of the pair: the units in the first CTA's claim order
                    while running:
                        word = mbox_wait(
                            mbox + cutlass.Int64(seq % MBOX_RING) * 4, (seq + 1) & 0xFFF
                        )
                        code = word & 0xFFFFF
                        if code == MBOX_DONE:
                            running = cutlass.Boolean(0)
                        else:
                            m_tile = code >> 6
                            n_tile = code & 0x3F
                            seq = seq + 1
                            expert_idx = tile_idx_to_group_idx[m_tile]
                            if cutlass.const_expr(self.sched_twin):
                                if n_tile == 0:
                                    expert_idx = expert_idx ^ 1
                            mn_limit = tile_idx_to_mn_limit[m_tile]
                            tile_info_pipeline.producer_acquire(
                                tile_info_producer_state
                            )
                            with cute.arch.elect_one():
                                sInfo[(0, tile_info_producer_state.index)] = (
                                    m_tile * pair + mma_tile_coord_v
                                )
                                sInfo[(1, tile_info_producer_state.index)] = n_tile
                                sInfo[(2, tile_info_producer_state.index)] = expert_idx
                                sInfo[(3, tile_info_producer_state.index)] = (
                                    cutlass.Int32(1)
                                )
                                sInfo[(4, tile_info_producer_state.index)] = mn_limit
                            cute.arch.fence_proxy("async.shared", space="cta")

                            self.sched_sync_barrier.arrive_and_wait()
                            tile_info_pipeline.producer_commit(tile_info_producer_state)
                            tile_info_producer_state.advance()
            elif cutlass.const_expr(self.raster_along_m):
                while work_tile.is_valid_tile:
                    cur_tile_coord = work_tile.tile_idx
                    mma_tile_coord_m = cur_tile_coord[0] // cute.size(
                        tiled_mma.thr_id.shape
                    )

                    tile_idx = mma_tile_coord_m

                    if tile_idx < num_valid_tiles:
                        expert_idx = tile_idx_to_group_idx[tile_idx]
                        mn_limit = tile_idx_to_mn_limit[tile_idx]

                        tile_info_pipeline.producer_acquire(tile_info_producer_state)
                        with cute.arch.elect_one():
                            sInfo[(0, tile_info_producer_state.index)] = cur_tile_coord[
                                0
                            ]
                            sInfo[(1, tile_info_producer_state.index)] = cur_tile_coord[
                                1
                            ]
                            sInfo[(2, tile_info_producer_state.index)] = expert_idx
                            sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(
                                work_tile.is_valid_tile
                            )
                            sInfo[(4, tile_info_producer_state.index)] = mn_limit
                        cute.arch.fence_proxy("async.shared", space="cta")

                        self.sched_sync_barrier.arrive_and_wait()
                        tile_info_pipeline.producer_commit(tile_info_producer_state)
                        tile_info_producer_state.advance()

                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
            else:
                is_continue = cutlass.Boolean(1)
                while work_tile.is_valid_tile and is_continue:
                    cur_tile_coord = work_tile.tile_idx
                    mma_tile_coord_m = cur_tile_coord[0] // cute.size(
                        tiled_mma.thr_id.shape
                    )

                    tile_idx = mma_tile_coord_m

                    if tile_idx < num_valid_tiles:
                        expert_idx = tile_idx_to_group_idx[tile_idx]
                        mn_limit = tile_idx_to_mn_limit[tile_idx]
                        tile_info_pipeline.producer_acquire(tile_info_producer_state)
                        with cute.arch.elect_one():
                            sInfo[(0, tile_info_producer_state.index)] = cur_tile_coord[
                                0
                            ]
                            sInfo[(1, tile_info_producer_state.index)] = cur_tile_coord[
                                1
                            ]
                            sInfo[(2, tile_info_producer_state.index)] = expert_idx
                            sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(
                                work_tile.is_valid_tile
                            )
                            sInfo[(4, tile_info_producer_state.index)] = mn_limit
                        cute.arch.fence_proxy("async.shared", space="cta")

                        self.sched_sync_barrier.arrive_and_wait()
                        tile_info_pipeline.producer_commit(tile_info_producer_state)
                        tile_info_producer_state.advance()
                    else:
                        is_continue = cutlass.Boolean(0)

                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

            # Signal end of work
            tile_info_pipeline.producer_acquire(tile_info_producer_state)
            with cute.arch.elect_one():
                sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            self.sched_sync_barrier.arrive_and_wait()
            tile_info_pipeline.producer_commit(tile_info_producer_state)
            tile_info_producer_state.advance()
            tile_info_pipeline.producer_tail(tile_info_producer_state)

        #
        # Specialized TMA load warp
        #
        if warp_idx == self.tma_warp_id:
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get first tile info (5 elements)
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )

                tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, 0)]
                tBgB_slice = tBgB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]
                tAgSFA_slice = tAgSFA[(None, mma_tile_coord_mnl[0], None, 0)]

                slice_n = mma_tile_coord_mnl[1]
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = mma_tile_coord_mnl[1] // 2

                tBgSFB_slice = tBgSFB[(None, slice_n, None, mma_tile_coord_mnl[2])]

                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                        ab_producer_state
                    )

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    ab_pipeline.producer_acquire(
                        ab_producer_state, peek_ab_empty_status
                    )

                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, ab_producer_state.count)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, ab_producer_state.count)],
                        tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=b_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_slice[(None, ab_producer_state.count)],
                        tAsSFA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfa_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_slice[(None, ab_producer_state.count)],
                        tBsSFB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfb_full_mcast_mask,
                    )

                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                            ab_producer_state
                        )

                # Get next tile
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            ab_pipeline.producer_tail(ab_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            if cutlass.const_expr(not self.tmem_bar_unaligned):
                tmem.wait_for_alloc()

            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            sfa_s2t_bundle = self._mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            sfb_s2t_bundle = self._mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get first tile info (5 elements)
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )

                acc_stage_index = acc_producer_state.index
                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(
                        ab_consumer_state
                    )

                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                # TMEM pointer offset for cta_tile_shape_n=192 or 64
                tCtSFB_mma = tCtSFB
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] in {64, 192}):
                    offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + self.num_accumulator_tmem_cols
                        + self.num_sfa_tmem_cols
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)

                # MMA mainloop
                for k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        ab_pipeline.consumer_wait(
                            ab_consumer_state, peek_ab_full_status
                        )

                        if cutlass.const_expr(not self._is_interleaved_utccp()):
                            self._mainloop_s2t_copies(
                                ab_consumer_state.index, sfa_s2t_bundle, sfb_s2t_bundle
                            )

                        num_kblocks = cute.size(tCrA, mode=[2])
                        for k_block in cutlass.range(num_kblocks, unroll_full=True):
                            if cutlass.const_expr(
                                self.enable_breuse
                                and cute.size(tCtAcc.layout, mode=[1]) == 2
                                and cute.size(tCtAcc.layout, mode=[2]) == 1
                            ):
                                tCtAcc_bkeep = tCtAcc[(None, 0, 0)]
                                tCtAcc_breuse = tCtAcc[(None, 1, 0)]

                                a_kblk_crd_keep = (
                                    None,
                                    0,
                                    k_block,
                                    ab_consumer_state.index,
                                )
                                a_kblk_crd_reuse = (
                                    None,
                                    1,
                                    k_block,
                                    ab_consumer_state.index,
                                )
                                b_kblk_crd = (None, 0, k_block, ab_consumer_state.index)

                                sfa_kblk_crd_keep = (None, 0, k_block)
                                sfa_kblk_crd_reuse = (None, 1, k_block)
                                sfb_kblk_crd = (None, 0, k_block)

                                if cutlass.const_expr(self._is_interleaved_utccp()):
                                    self._mainloop_s2t_interleaved_copies(
                                        k_block,
                                        ab_consumer_state.index,
                                        sfa_s2t_bundle,
                                        sfb_s2t_bundle,
                                    )

                                # Bkeep
                                tiled_mma_bkeep.set(
                                    tcgen05.Field.ACCUMULATE,
                                    k_tile != 0 or k_block != 0,
                                )
                                cute.gemm(
                                    tiled_mma_bkeep,
                                    tCtAcc_bkeep,
                                    [tCrA[a_kblk_crd_keep], tCtSFA[sfa_kblk_crd_keep]],
                                    [tCrB[b_kblk_crd], tCtSFB_mma[sfb_kblk_crd]],
                                    tCtAcc_bkeep,
                                )
                                # Breuse
                                tiled_mma_breuse.set(
                                    tcgen05.Field.ACCUMULATE,
                                    k_tile != 0 or k_block != 0,
                                )
                                cute.gemm(
                                    tiled_mma_breuse,
                                    tCtAcc_breuse,
                                    [
                                        tCrA[a_kblk_crd_reuse],
                                        tCtSFA[sfa_kblk_crd_reuse],
                                    ],
                                    [tCrB[b_kblk_crd], tCtSFB_mma[sfb_kblk_crd]],
                                    tCtAcc_breuse,
                                )
                            else:
                                kblk_crd = (
                                    None,
                                    None,
                                    k_block,
                                    ab_consumer_state.index,
                                )
                                sf_kblk_crd = (None, None, k_block)

                                tiled_mma.set(
                                    tcgen05.Field.ACCUMULATE,
                                    k_tile != 0 or k_block != 0,
                                )
                                cute.gemm(
                                    tiled_mma,
                                    tCtAcc,
                                    [tCrA[kblk_crd], tCtSFA[sf_kblk_crd]],
                                    [tCrB[kblk_crd], tCtSFB_mma[sf_kblk_crd]],
                                    tCtAcc,
                                )

                        ab_pipeline.consumer_release(ab_consumer_state)

                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                ab_consumer_state
                            )

                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                # Get next tile (4 elements)
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            acc_pipeline.producer_tail(acc_producer_state)

        #
        # Fused conversion warps
        #
        if cutlass.const_expr(self.fuse_quant):
            if warp_idx >= self.conv_warp_ids[0]:
                self._conv_warps(
                    warp_idx - self.conv_warp_ids[0],
                    tidx % 32,
                    sQflag,
                    c,
                    row_amax,
                    q_out,
                    q_sf,
                    q_scale,
                    q_ctl,
                )

        #
        # Specialized epilogue warps with finalize fusion
        #
        if warp_idx < self.mma_warp_id:
            if cutlass.const_expr(not self.tmem_bar_unaligned):
                tmem.allocate(self.num_tmem_alloc_cols)
                tmem.wait_for_alloc()

            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # Layout transformation for tCtAcc_base and tCgC
            # ((MMA_ATOM_M, MMA_ATOM_N), MMA_M, MMA_N, ...rest)
            # -> ((MMA_ATOM_M, MMA_M), (MMA_ATOM_N, MMA_N), ...rest)
            tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)
            tCgC_transformed = transform_partitioned_tensor_layout(tCgC)

            # Partition for epilogue
            epi_tidx = tidx % 128
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc,
            ) = epilogue_tmem_copy_and_partition(
                self, epi_tidx, tCtAcc, tCgC_transformed, epi_tile, use_2cta_instrs
            )

            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tTR_rAcc_hi = cute.make_rmem_tensor(tTR_rAcc.shape, self.acc_dtype)

            # Setup smem copy for block reduce
            atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.c_dtype,
            )
            tiled_copy_r2s = cute.make_tiled_copy_D(atom, tiled_copy_t2r)
            thr_copy_r2s = tiled_copy_r2s.get_slice(epi_tidx)
            tRS_sC = thr_copy_r2s.partition_D(sC)
            tRS_rC = tiled_copy_r2s.retile(tTR_rC)
            bSG_sC = None
            bSG_gC = None
            c_pipeline = None
            c_count = None
            if cutlass.const_expr(self.tma_store):
                # staging buffers of (128 rows x epilogue tile) for tensor-memory-access stores of the output
                tiled_copy_r2s, tRS_rC, tRS_sC = epilogue_smem_copy_and_partition(
                    self, tiled_copy_t2r, tTR_rC, epi_tidx, sC
                )
                gC_epi = cute.flat_divide(
                    mC_out, (self.cta_tile_shape_mnk[0], self.epi_tile_n)
                )
                bSG_sC, bSG_gC = cpasync.tma_partition(
                    tma_atom_c,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sC, 0, 2),
                    cute.group_modes(gC_epi, 0, 2),
                )
                c_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.num_c_stage,
                    producer_group=pipeline.CooperativeGroup(
                        pipeline.Agent.Thread, 32 * len(self.epilog_warp_id)
                    ),
                )
                c_count = cute.make_rmem_tensor((1,), cutlass.Int32)
                c_count[0] = cutlass.Int32(0)

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Fused conversion: the previous tile (row block, N tile, mn_limit, valid) is handed to the conversion
            # warps once this thread's bulk copies of it have completed.
            qprev = cute.make_rmem_tensor((4,), cutlass.Int32)
            for qi in cutlass.range_constexpr(4):
                qprev[qi] = cutlass.Int32(0)
            q_sbase = smem_addr(sQflag)
            c_base = gaddr(c)
            c_cols = cutlass.Int64(cute.size(c, mode=[1]) // 2)
            n_tiles_total = cute.size(c, mode=[1]) // self.cta_tile_shape_mnk[1]
            n_row_blocks = cute.size(c, mode=[0]) // self.cta_tile_shape_mnk[0]
            q_base = gaddr(q_out)
            qsf_base = gaddr(q_sf)
            qscale_base = gaddr(q_scale)
            amax_base = gaddr(row_amax)
            ctl_base = gaddr(q_ctl)

            # Get first tile info (5 elements)
            tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(5, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )

                expert_idx = mma_tile_coord_mnl[2]
                alpha_val = alpha[expert_idx]

                # Compute base row index for this tile
                tile_m_start = tile_info[0] * self.cta_tile_shape_mnk[0]

                # Get accumulator stage index
                acc_stage_index = acc_consumer_state.index

                # Set tensor memory buffer for current tile
                tTR_tAcc = tTR_tAcc_base[
                    (None, None, None, None, None, acc_stage_index)
                ]

                # Wait for accumulator buffer full
                acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                # Group tRS_sC modes 1 and 2 (m_iter, n_subtile) into a single 2D mode
                # Before: ((1,32), 2, 4, 1) -> After: ((1,32), (2,4), 1)
                # This allows using tuple indexing like tTR_tAcc
                tRS_sC_grouped = cute.group_modes(tRS_sC, 1, 3)

                # Get m-iteration count and n-subtile count from tTR_tAcc shape
                # In B-reuse (cta_tile_m=256): mode-3 is (2,4), m_iter_cnt=2, n_subtile_cnt=4
                # In non-B-reuse (cta_tile_m=128): mode-3 is (1,4), m_iter_cnt=1, n_subtile_cnt=4
                m_iter_cnt = cute.size(tTR_tAcc.shape[3], mode=[0])
                n_subtile_cnt = cute.size(tTR_tAcc.shape[3], mode=[1])

                # Process all m-iterations and n-subtiles: scale, restore natural column order (or SwiGLU), stage in sC.
                if cutlass.const_expr(self.swiglu):
                    alpha_up_val = alpha_up[expert_idx]
                    LOG2_E = cutlass.Float32(1.4426950408889634)
                    for m_iter_idx in cutlass.range(m_iter_cnt):
                        permuted_row = tile_m_start + m_iter_idx * 128 + epi_tidx
                        is_valid_row = permuted_row < tile_info[4]
                        gate_scale = cutlass.Float32(0.0)
                        up_scale = cutlass.Float32(0.0)
                        if is_valid_row:
                            rs = row_scale[permuted_row]
                            gate_scale = cutlass.Float32(alpha_val) * rs
                            up_scale = cutlass.Float32(alpha_up_val) * rs
                        row_max = cutlass.Float32(0.0)
                        for j in cutlass.range(n_subtile_cnt // 2):
                            cute.copy(
                                tiled_copy_t2r,
                                tTR_tAcc[(None, None, None, (m_iter_idx, 2 * j))],
                                tTR_rAcc,
                            )
                            cute.copy(
                                tiled_copy_t2r,
                                tTR_tAcc[(None, None, None, (m_iter_idx, 2 * j + 1))],
                                tTR_rAcc_hi,
                            )
                            acc_lo = tTR_rAcc.load()
                            acc_hi = tTR_rAcc_hi.load()
                            tOut = cute.make_rmem_tensor(acc_lo.shape, self.acc_dtype)
                            for q in cutlass.range_constexpr(self.epi_tile_n // 16):
                                base = (q % 2) * 32
                                for a_ in cutlass.range_constexpr(8):
                                    if cutlass.const_expr(q < 2):
                                        up0 = acc_lo[base + a_]
                                        gate0 = acc_lo[base + 8 + a_]
                                        up1 = acc_lo[base + 16 + a_]
                                        gate1 = acc_lo[base + 24 + a_]
                                    else:
                                        up0 = acc_hi[base + a_]
                                        gate0 = acc_hi[base + 8 + a_]
                                        up1 = acc_hi[base + 16 + a_]
                                        gate1 = acc_hi[base + 24 + a_]
                                    o = 16 * q + 2 * a_
                                    g = cute.arch.mul_packed_f32x2(
                                        (gate0, gate1), (gate_scale, gate_scale)
                                    )
                                    u = cute.arch.mul_packed_f32x2(
                                        (up0, up1), (up_scale, up_scale)
                                    )
                                    t = cute.arch.mul_packed_f32x2(
                                        (g[0], g[1]), (-LOG2_E, -LOG2_E)
                                    )
                                    e = cute.arch.add_packed_f32x2(
                                        (
                                            cute.math.exp2(t[0], fastmath=True),
                                            cute.math.exp2(t[1], fastmath=True),
                                        ),
                                        (1.0, 1.0),
                                    )
                                    r0 = cute.arch.rcp_approx(e[0])
                                    r1 = cute.arch.rcp_approx(e[1])
                                    sg = cute.arch.mul_packed_f32x2(
                                        (r0, r1), (g[0], g[1])
                                    )
                                    tOut[o], tOut[o + 1] = cute.arch.mul_packed_f32x2(
                                        (sg[0], sg[1]), (u[0], u[1])
                                    )
                                    if cutlass.const_expr(self.row_amax_out):
                                        # BF16 rounding is monotone in |x|, so BF16(max |x|) = max |BF16(x)|.
                                        row_max = abs_max_f32(row_max, tOut[o])
                                        row_max = abs_max_f32(row_max, tOut[o + 1])
                            if cutlass.const_expr(self.tma_store):
                                self._tma_store_subtile(
                                    tiled_copy_r2s,
                                    tRS_rC,
                                    tRS_sC,
                                    tOut,
                                    bSG_sC,
                                    bSG_gC,
                                    tma_atom_c,
                                    c_pipeline,
                                    c_count,
                                    warp_idx,
                                    tile_info[0],
                                    tile_info[1] * (self.out_tile_n // self.epi_tile_n)
                                    + j,
                                )
                            else:
                                tRS_rC.store(
                                    tiled_copy_r2s.retile(tOut).load().to(self.c_dtype)
                                )
                                if is_valid_row:
                                    cute.copy(
                                        tiled_copy_r2s,
                                        tRS_rC[None, 0, 0],
                                        tRS_sC_grouped[(None, (m_iter_idx, j), 0)],
                                    )
                                cute.arch.fence_proxy("async.shared", space="cta")
                        if cutlass.const_expr(self.row_amax_out):
                            if is_valid_row:
                                red_max_bf16_bits(
                                    cute.domain_offset((permuted_row,), row_amax),
                                    row_max,
                                )
                else:
                    for m_iter_idx in cutlass.range(m_iter_cnt):
                        # Each thread handles row: tile_m_start + m_iter_idx * 128 + epi_tidx
                        permuted_row = tile_m_start + m_iter_idx * 128 + epi_tidx
                        is_valid_row = permuted_row < tile_info[4]
                        alpha_val_iter = cutlass.Float32(0.0)
                        if is_valid_row:
                            alpha_val_iter = (
                                cutlass.Float32(alpha_val) * row_scale[permuted_row]
                            )

                        for n_iter_idx in cutlass.range(n_subtile_cnt):
                            # Load accumulator from tensor memory buffer to register
                            tTR_tAcc_mn = tTR_tAcc[
                                (None, None, None, (m_iter_idx, n_iter_idx))
                            ]
                            cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                            acc_vec = tTR_rAcc.load()
                            # Within each block of 32 columns, accumulator column 8c + a holds output column 4a + c.
                            tPerm = cute.make_rmem_tensor(acc_vec.shape, self.acc_dtype)
                            for o in cutlass.range_constexpr(self.epi_tile_n):
                                f = (o // 32) * 32 + (o % 4) * 8 + (o % 32) // 4
                                tPerm[o] = acc_vec[f] * alpha_val_iter
                            if cutlass.const_expr(self.tma_store):
                                self._tma_store_subtile(
                                    tiled_copy_r2s,
                                    tRS_rC,
                                    tRS_sC,
                                    tPerm,
                                    bSG_sC,
                                    bSG_gC,
                                    tma_atom_c,
                                    c_pipeline,
                                    c_count,
                                    warp_idx,
                                    tile_info[0],
                                    tile_info[1] * (self.out_tile_n // self.epi_tile_n)
                                    + n_iter_idx,
                                )
                            else:
                                acc_vec_final = (
                                    tiled_copy_r2s.retile(tPerm).load().to(self.c_dtype)
                                )
                                tRS_rC.store(acc_vec_final)
                                if is_valid_row:
                                    cute.copy(
                                        tiled_copy_r2s,
                                        tRS_rC[None, 0, 0],
                                        tRS_sC_grouped[
                                            (None, (m_iter_idx, n_iter_idx), 0)
                                        ],
                                    )

                                cute.arch.fence_proxy("async.shared", space="cta")

                # Async arrive accumulator buffer empty
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

                if cutlass.const_expr(not self.tma_store):
                    # Write every valid row of the tile to its own output row (bulk copies from sC).
                    for m_iter_idx in cutlass.range(m_iter_cnt):
                        permuted_row = tile_m_start + m_iter_idx * 128 + epi_tidx
                        is_valid_row = permuted_row < tile_info[4]

                        if is_valid_row:
                            coord_n = mma_tile_coord_mnl[1] * self.out_tile_n
                            row_out_offset = cute.domain_offset(
                                (permuted_row, coord_n, 0), c
                            )
                            sC_row = m_iter_idx * 128 + epi_tidx
                            blk_copy_s2g(
                                row_out_offset,
                                sC[sC_row, None, 0],
                                cutlass.Int32(self.copy_size),
                            )
                    # sC is reused by the next tile: wait until this thread's copies have read it.
                    blk_commit_and_wait_read()
                if cutlass.const_expr(self.fuse_quant):
                    # The previous tile's rows are complete in memory (all groups but this tile's).
                    blk_wait_older()
                self.epilog_sync_barrier.arrive_and_wait()

                if cutlass.const_expr(self.fuse_quant):
                    # Every thread's row-maximum updates of this tile precede the barrier above: publish the tile, and
                    # hand the previous tile (its rows complete before the barrier) to the conversion warps.
                    if epi_tidx == 0:
                        publish_add(ctl_base + cutlass.Int64(tile_info[0]) * 4)
                        if qprev[3] == 1:
                            self._qpush(sQflag, q_sbase, qprev[0], qprev[1], qprev[2])
                    qprev[0] = tile_info[0]
                    qprev[1] = tile_info[1]
                    qprev[2] = tile_info[4]
                    qprev[3] = cutlass.Int32(1)

                # Get next tile (5 elements)
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(5, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            if cutlass.const_expr(self.fuse_quant):
                # Hand over the last tile once its rows are complete, then tell the conversion warps the epilogue is
                # finished.
                blk_wait_all_fenced()
                self.epilog_sync_barrier.arrive_and_wait()
                if epi_tidx == 0:
                    if qprev[3] == 1:
                        self._qpush(sQflag, q_sbase, qprev[0], qprev[1], qprev[2])
                    st_release_cta_s32(q_sbase + 3 * 4, cutlass.Int32(1))

            # Every output row written before the CTA exits.
            blk_wait_all()
            if cutlass.const_expr(self.tma_store):
                c_pipeline.producer_tail()
            # Dealloc tensor memory
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            if cutlass.const_expr(self.tmem_bar_unaligned):
                tmem.free(
                    acc_tmem_ptr, self._tmem_cols("free", self.num_tmem_alloc_cols)
                )  # allocated in the shared block
            else:
                tmem.free(acc_tmem_ptr)

    QFIFO = 8  # tiles queued for the conversion warps (tunable)
    QSTEPS = 16  # two-row steps whose loads a conversion warp issues together (tunable)
    QDRAIN = 1  # entries a conversion warp still converts after the epilogue finished (tunable)
    QPOLL_BOUND = 1 << 22  # polls of one wait before it gives up (error word set)

    @cute.jit
    def _qpush(self, sQ, q_sbase, rb, n, mn):
        """Epilogue thread 0: queue a tile for the conversion warps, or drop it if the queue is full (never waits)."""
        tail = sQ[0]
        h0 = ld_acquire_cta_s32(q_sbase + 1 * 4)
        h1 = ld_acquire_cta_s32(q_sbase + 2 * 4)
        hmin = h0
        if h1 < hmin:
            hmin = h1
        if tail - hmin + 1 < self.QFIFO:
            slot = tail % self.QFIFO
            sQ[8 + 3 * slot] = rb
            sQ[8 + 3 * slot + 1] = n
            sQ[8 + 3 * slot + 2] = mn
            st_release_cta_s32(q_sbase, tail + 1)

    @cute.jit
    def _conv_warps(self, w, lane, sQ, c, row_amax, q_out, q_sf, q_scale, q_ctl):
        """Conversion warp w (0 or 1) takes queue entries w, w + 2, ...: waits (bounded) for its row block to complete
        while the epilogue runs, converts the region, sets its done flag. After the epilogue finished it no longer
        waits: an entry whose row block is incomplete is left for the separate pass."""
        q_sbase = smem_addr(sQ)
        c_base = gaddr(c)
        c_cols = cutlass.Int64(cute.size(c, mode=[1]) // 2)
        n_tiles_total = cute.size(c, mode=[1]) // self.cta_tile_shape_mnk[1]
        n_row_blocks = cute.size(c, mode=[0]) // self.cta_tile_shape_mnk[0]
        amax_base = gaddr(row_amax)
        ctl_base = gaddr(q_ctl)
        q_base = gaddr(q_out)
        qsf_base = gaddr(q_sf)
        qscale_base = gaddr(q_scale)
        err_addr = ctl_base + (cutlass.Int64(n_row_blocks) * (1 + n_tiles_total)) * 4
        k = cutlass.Int32(w)
        running = cutlass.Int32(1)
        polls = cutlass.Int32(0)
        after_fin = cutlass.Int32(0)
        while running == 1:
            tail = cutlass.Int32(0)
            fin = cutlass.Int32(0)
            if lane == 0:
                fin = ld_acquire_cta_s32(q_sbase + 3 * 4)
                tail = ld_acquire_cta_s32(q_sbase)
            tail = cute.arch.shuffle_sync(tail, 0)
            fin = cute.arch.shuffle_sync(fin, 0)
            if fin == 1:
                after_fin = after_fin + 1
            if after_fin > self.QDRAIN:
                running = cutlass.Int32(0)
            elif k < tail:
                slot = k % self.QFIFO
                rb = sQ[8 + 3 * slot]
                n = sQ[8 + 3 * slot + 1]
                mn = sQ[8 + 3 * slot + 2]
                # wait (bounded) for the row block while the epilogue runs; one look after it finished
                ready = cutlass.Int32(0)
                looking = cutlass.Int32(1)
                while looking == 1:
                    cnt = cutlass.Int32(0)
                    fin2 = cutlass.Int32(0)
                    if lane == 0:
                        fin2 = ld_acquire_cta_s32(q_sbase + 3 * 4)
                        cnt = ld_acquire_u32(ctl_base + cutlass.Int64(rb) * 4)
                    cnt = cute.arch.shuffle_sync(cnt, 0)
                    fin2 = cute.arch.shuffle_sync(fin2, 0)
                    if cnt == n_tiles_total:
                        ready = cutlass.Int32(1)
                        looking = cutlass.Int32(0)
                    elif fin2 == 1:
                        looking = cutlass.Int32(0)
                    else:
                        polls = polls + 1
                        if polls >= self.QPOLL_BOUND:
                            if lane == 0:
                                st_b32(err_addr, cutlass.Int32(1))
                            looking = cutlass.Int32(0)
                        else:
                            nanosleep_hint()
                if ready == 1:
                    self._conv_region(
                        lane,
                        rb,
                        n,
                        mn,
                        c_base,
                        c_cols,
                        n_tiles_total,
                        n_row_blocks,
                        amax_base,
                        ctl_base,
                        q_base,
                        qsf_base,
                        qscale_base,
                    )
                k = k + 2
                if lane == 0:
                    st_release_cta_s32(q_sbase + (1 + w) * 4, k)
            else:
                if fin == 1:
                    running = cutlass.Int32(0)
                else:
                    polls = polls + 1
                    if polls >= self.QPOLL_BOUND:
                        if lane == 0:
                            st_b32(err_addr, cutlass.Int32(1))
                        running = cutlass.Int32(0)
                    else:
                        nanosleep_hint()

    @cute.jit
    def _conv_region(
        self,
        lane,
        rb,
        n,
        mn,
        c_base,
        c_cols,
        n_tiles_total,
        n_row_blocks,
        amax_base,
        ctl_base,
        q_base,
        qsf_base,
        qscale_base,
    ):
        """One warp converts one region (128 rows of a row block x the N tile's 128 output columns). Each step covers
        two rows: lanes 16h .. 16h + 15 read row 2i + h as 16 consecutive 16-byte pieces (8 values each, half a
        16-value block; the two halves of a block exchange their maxima), so every load and store is contiguous."""
        ro = lane // 16
        cc = lane % 16
        col = cutlass.Int64(n * self.out_tile_n + cc * 8)
        k_tiles = c_cols // 64
        for it4 in cutlass.range(self.cta_tile_shape_mnk[0] // (2 * self.QSTEPS)):
            words = []
            amaxs = []
            rows = []
            for u in cutlass.range_constexpr(self.QSTEPS):
                r = rb * self.cta_tile_shape_mnk[0] + 2 * self.QSTEPS * it4 + 2 * u + ro
                rows.append(r)
                words.append(ld_cg_v4(c_base + (cutlass.Int64(r) * c_cols + col) * 2))
                amaxs.append(ld_relaxed_u32(amax_base + cutlass.Int64(r) * 4))
            for u in cutlass.range_constexpr(self.QSTEPS):
                r = rows[u]
                own = absmax8(words[u])
                vmax = fmax_f32(own, cute.arch.shuffle_sync_bfly(own, 1))
                pts, enc = token_scales(amaxs[u])
                pay, sf = quant_half8(words[u], vmax, enc)
                # scale bytes of blocks 4j .. 4j + 3 (even lanes cc = 8j, 8j + 2, 8j + 4, 8j + 6) as one word
                word = (
                    sf
                    | (cute.arch.shuffle_sync_down(sf, 2) << 8)
                    | (cute.arch.shuffle_sync_down(sf, 4) << 16)
                    | (cute.arch.shuffle_sync_down(sf, 6) << 24)
                )
                if r < mn:
                    st_b32(
                        q_base
                        + cutlass.Int64(r) * (c_cols // 2)
                        + cutlass.Int64(n * (self.out_tile_n // 2) + cc * 4),
                        pay,
                    )
                    if cc % 8 == 0:
                        sf_row = (
                            cutlass.Int64(r // 128) * k_tiles * 512
                            + cutlass.Int64((r % 32) * 16)
                            + cutlass.Int64(((r % 128) // 32) * 4)
                        )
                        kt = cutlass.Int64(n * (self.out_tile_n // 64) + cc // 8)
                        st_b32(qsf_base + sf_row + kt * 512, word)
                    if n == 0:
                        if cc == 0:
                            st_b32(qscale_base + cutlass.Int64(r) * 4, f32_bits(pts))
        if lane == 0:
            st_b32(
                ctl_base
                + (
                    cutlass.Int64(n_row_blocks)
                    + cutlass.Int64(rb) * n_tiles_total
                    + cutlass.Int64(n)
                )
                * 4,
                cutlass.Int32(1),
            )

    @cute.jit
    def _tma_store_subtile(
        self,
        tiled_copy_r2s,
        tRS_rC,
        tRS_sC,
        vals,
        bSG_sC,
        bSG_gC,
        tma_atom_c,
        c_pipeline,
        c_count,
        warp_idx,
        m_tile,
        n_sub,
    ):
        """Stage one output subtile (128 rows x epilogue tile) in shared memory and store it with one
        tensor-memory-access copy (whole box: rows past mn_limit carry zeros, rows past the tensor are clipped)."""
        tRS_rC.store(tiled_copy_r2s.retile(vals).load().to(self.c_dtype))
        c_buffer = c_count[0] % self.num_c_stage
        c_count[0] = c_count[0] + 1
        cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
        cute.arch.fence_proxy("async.shared", space="cta")
        self.epilog_sync_barrier.arrive_and_wait()
        if warp_idx == self.epilog_warp_id[0]:
            cute.copy(
                tma_atom_c, bSG_sC[(None, c_buffer)], bSG_gC[(None, m_tile, n_sub, 0)]
            )
            c_pipeline.producer_commit()
            c_pipeline.producer_acquire()
        self.epilog_sync_barrier.arrive_and_wait()

    @staticmethod
    def _compute_grid(
        gemm_shape: Tuple[int, int, int],
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
        raster_along_m: bool,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        """Compute grid size based on GEMM shape."""
        (m, n, l) = gemm_shape  # noqa: E741

        num_ctas_m = cute.ceil_div(m, cta_tile_shape_mnk[0])
        num_ctas_n = cute.ceil_div(n, cta_tile_shape_mnk[1])
        num_ctas_l = l

        num_ctas_mnl = (num_ctas_m, num_ctas_n, num_ctas_l)
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl, raster_along_m=raster_along_m
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    @staticmethod
    def is_valid_dtypes_and_scale_factor_vec_size(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """Check if the dtypes and sf_vec_size are valid combinations."""
        valid_combinations = {
            # 4xFP4
            (cutlass.Float4E2M1FN, cutlass.Float4E2M1FN, cutlass.Float8E8M0FNU, 16),
            (cutlass.Float4E2M1FN, cutlass.Float4E2M1FN, cutlass.Float8E8M0FNU, 32),
            (cutlass.Float4E2M1FN, cutlass.Float4E2M1FN, cutlass.Float8E4M3FN, 16),
            (cutlass.Float4E2M1FN, cutlass.Float4E2M1FN, cutlass.Float8E4M3FN, 32),
            # 2xFP8
            (cutlass.Float8E5M2, cutlass.Float8E5M2, cutlass.Float8E8M0FNU, 32),
            (cutlass.Float8E5M2, cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, 32),
            (cutlass.Float8E4M3FN, cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, 32),
            (cutlass.Float8E4M3FN, cutlass.Float8E5M2, cutlass.Float8E8M0FNU, 32),
        }

        current_combination = (a_dtype, b_dtype, sf_dtype, sf_vec_size)
        if current_combination not in valid_combinations:
            return False

        if c_dtype not in {cutlass.Float32, cutlass.Float16, cutlass.BFloat16}:
            return False

        return True

    @staticmethod
    def is_valid_layouts(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """Check if layouts and dtypes are valid combinations."""
        if (
            a_dtype is cutlass.Float4E2M1FN
            and b_dtype is cutlass.Float4E2M1FN
            and not (a_major == "k" and b_major == "k")
        ):
            return False

        if c_dtype is cutlass.Float4E2M1FN and c_major == "m":
            return False

        return True

    @staticmethod
    def is_valid_mma_tiler_and_cluster_shape(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        mma_inst_shape: Tuple[int, int, int],
        mma_tiler: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
    ) -> bool:
        """Check if the mma tiler and cluster shape are valid."""
        # Check valid mma_inst_shape
        if mma_inst_shape[0] not in [128, 256]:
            return False
        if mma_inst_shape[1] not in [64, 128, 192, 256]:
            return False

        # Check valid mma_tiler
        if mma_tiler[0] not in [128, 256, 512]:
            return False
        if mma_tiler[1] not in [64, 128, 192, 256]:
            return False

        # Check MMA tiler vs MMA instruction relationship
        b_reuse = mma_tiler[0] // mma_inst_shape[0] == 2
        if mma_tiler[0] != mma_inst_shape[0] and not b_reuse:
            return False
        if mma_tiler[1] != mma_inst_shape[1]:
            return False

        # Check K-dimension constraints
        if a_dtype in {cutlass.Float8E4M3FN, cutlass.Float8E5M2} and b_dtype in {
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }:
            if mma_tiler[2] != 128 or mma_inst_shape[2] != 64:
                return False
        else:
            if mma_tiler[2] != 256 or mma_inst_shape[2] != 128:
                return False

        # Check cluster shape
        if cluster_shape_mn[0] % (2 if mma_inst_shape[0] == 256 else 1) != 0:
            return False

        # Check cluster shape validity
        def is_power_of_2(x):
            return x > 0 and (x & (x - 1)) == 0

        if (
            cluster_shape_mn[0] * cluster_shape_mn[1] > 16
            or cluster_shape_mn[0] <= 0
            or cluster_shape_mn[1] <= 0
            or cluster_shape_mn[0] > 4
            or cluster_shape_mn[1] > 4
            or not is_power_of_2(cluster_shape_mn[0])
            or not is_power_of_2(cluster_shape_mn[1])
        ):
            return False

        return True

    @staticmethod
    def is_valid_tensor_alignment(
        m: int,
        n: int,
        k: int,
        l: int,  # noqa: E741
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """Check if the tensor alignment is valid (16B alignment)."""

        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        if (
            not check_contigous_16B_alignment(a_dtype, a_major == "m", (m, k, l))
            or not check_contigous_16B_alignment(b_dtype, b_major == "n", (n, k, l))
            or not check_contigous_16B_alignment(c_dtype, c_major == "m", (m, n, l))
        ):
            return False

        return True

    @cute.jit
    def wrapper(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        row_scale_ptr: cute.Pointer,
        alpha_up_ptr: cute.Pointer,
        row_amax_ptr: cute.Pointer,
        q_out_ptr: cute.Pointer,
        q_sf_ptr: cute.Pointer,
        q_scale_ptr: cute.Pointer,
        q_ctl_ptr: cute.Pointer,
        tile_idx_to_group_idx_ptr: cute.Pointer,
        tile_idx_to_mn_limit_ptr: cute.Pointer,
        num_non_exiting_tiles_ptr: cute.Pointer,
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,  # noqa: E741
        tile_size: cutlass.Constexpr,
        scaling_vector_size: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        scale_k = k // scaling_vector_size
        num_tiles = m // tile_size
        a = cute.make_tensor(
            a_ptr, layout=cute.make_ordered_layout((m, k, 1), order=(1, 0, 2))
        )
        b = cute.make_tensor(
            b_ptr, layout=cute.make_ordered_layout((n, k, l), order=(1, 0, 2))
        )
        a_sf = cute.make_tensor(
            a_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, m // 128, 4, scale_k // 4, 1), order=(2, 1, 4, 0, 3, 5)
            ),
        )
        b_sf = cute.make_tensor(
            b_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, n // 128, 4, scale_k // 4, l), order=(2, 1, 4, 0, 3, 5)
            ),
        )
        # SwiGLU mode: C keeps the GEMM's (m, n) logical shape for the partitioning, rows are n / 2 wide in memory and
        # every write goes to an explicit (row, column) offset inside the first n / 2 columns.
        c_row = n // 2 if self.swiglu else n
        c = cute.make_tensor(
            c_ptr,
            layout=cute.make_layout((m, n, 1), stride=(c_row, 1, m * c_row)),
        )
        alpha = cute.make_tensor(alpha_ptr, layout=cute.make_layout((l,)))
        row_scale = cute.make_tensor(row_scale_ptr, layout=cute.make_layout((m,)))
        alpha_up = cute.make_tensor(alpha_up_ptr, layout=cute.make_layout((l,)))
        row_amax = cute.make_tensor(row_amax_ptr, layout=cute.make_layout((m,)))
        # fused conversion outputs (only their base addresses are used): payload, swizzled scales, per-row scales, ctl
        q_out = cute.make_tensor(q_out_ptr, layout=cute.make_layout((m,)))
        q_sf = cute.make_tensor(q_sf_ptr, layout=cute.make_layout((m,)))
        q_scale = cute.make_tensor(q_scale_ptr, layout=cute.make_layout((m,)))
        q_ctl = cute.make_tensor(q_ctl_ptr, layout=cute.make_layout((m,)))
        # logical output (rows, output columns) for the tensor-memory-access store
        c_out = cute.make_tensor(
            c_ptr, layout=cute.make_layout((m, c_row, 1), stride=(c_row, 1, m * c_row))
        )

        tile_idx_to_group_idx = cute.make_tensor(
            tile_idx_to_group_idx_ptr, layout=cute.make_layout((num_tiles,))
        )
        tile_idx_to_mn_limit = cute.make_tensor(
            tile_idx_to_mn_limit_ptr, layout=cute.make_layout((num_tiles,))
        )
        num_non_exiting_tiles = cute.make_tensor(
            num_non_exiting_tiles_ptr, layout=cute.make_layout((1,))
        )

        return self(
            a,
            b,
            c,
            a_sf,
            b_sf,
            tile_idx_to_group_idx,
            num_non_exiting_tiles,
            tile_idx_to_mn_limit,
            alpha,
            max_active_clusters=max_active_clusters,
            stream=stream,
            row_scale=row_scale,
            alpha_up=alpha_up,
            row_amax=row_amax,
            q_out=q_out,
            q_sf=q_sf,
            q_scale=q_scale,
            q_ctl=q_ctl,
            c_out=c_out,
        )

    @cute.jit
    def wrapper_sched(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        row_scale_ptr: cute.Pointer,
        alpha_up_ptr: cute.Pointer,
        row_amax_ptr: cute.Pointer,
        q_out_ptr: cute.Pointer,
        q_sf_ptr: cute.Pointer,
        q_scale_ptr: cute.Pointer,
        q_ctl_ptr: cute.Pointer,
        tile_idx_to_group_idx_ptr: cute.Pointer,
        tile_idx_to_mn_limit_ptr: cute.Pointer,
        num_non_exiting_tiles_ptr: cute.Pointer,
        sched_ctl_ptr: cute.Pointer,
        sm_dom_ptr: cute.Pointer,
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,  # noqa: E741
        tile_size: cutlass.Constexpr,
        scaling_vector_size: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """``wrapper`` for the claimed schedules: the same tensors plus the schedule's control block and the per-SM
        locality-domain table (only their base addresses are used)."""
        scale_k = k // scaling_vector_size
        num_tiles = m // tile_size
        a = cute.make_tensor(
            a_ptr, layout=cute.make_ordered_layout((m, k, 1), order=(1, 0, 2))
        )
        b = cute.make_tensor(
            b_ptr, layout=cute.make_ordered_layout((n, k, l), order=(1, 0, 2))
        )
        a_sf = cute.make_tensor(
            a_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, m // 128, 4, scale_k // 4, 1), order=(2, 1, 4, 0, 3, 5)
            ),
        )
        b_sf = cute.make_tensor(
            b_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, n // 128, 4, scale_k // 4, l), order=(2, 1, 4, 0, 3, 5)
            ),
        )
        c_row = n // 2 if self.swiglu else n
        c = cute.make_tensor(
            c_ptr,
            layout=cute.make_layout((m, n, 1), stride=(c_row, 1, m * c_row)),
        )
        alpha = cute.make_tensor(alpha_ptr, layout=cute.make_layout((l,)))
        row_scale = cute.make_tensor(row_scale_ptr, layout=cute.make_layout((m,)))
        alpha_up = cute.make_tensor(alpha_up_ptr, layout=cute.make_layout((l,)))
        row_amax = cute.make_tensor(row_amax_ptr, layout=cute.make_layout((m,)))
        q_out = cute.make_tensor(q_out_ptr, layout=cute.make_layout((m,)))
        q_sf = cute.make_tensor(q_sf_ptr, layout=cute.make_layout((m,)))
        q_scale = cute.make_tensor(q_scale_ptr, layout=cute.make_layout((m,)))
        q_ctl = cute.make_tensor(q_ctl_ptr, layout=cute.make_layout((m,)))
        c_out = cute.make_tensor(
            c_ptr, layout=cute.make_layout((m, c_row, 1), stride=(c_row, 1, m * c_row))
        )
        tile_idx_to_group_idx = cute.make_tensor(
            tile_idx_to_group_idx_ptr, layout=cute.make_layout((num_tiles,))
        )
        tile_idx_to_mn_limit = cute.make_tensor(
            tile_idx_to_mn_limit_ptr, layout=cute.make_layout((num_tiles,))
        )
        num_non_exiting_tiles = cute.make_tensor(
            num_non_exiting_tiles_ptr, layout=cute.make_layout((1,))
        )
        sched_ctl = cute.make_tensor(sched_ctl_ptr, layout=cute.make_layout((1,)))
        sm_dom = cute.make_tensor(sm_dom_ptr, layout=cute.make_layout((1,)))

        return self(
            a,
            b,
            c,
            a_sf,
            b_sf,
            tile_idx_to_group_idx,
            num_non_exiting_tiles,
            tile_idx_to_mn_limit,
            alpha,
            max_active_clusters=max_active_clusters,
            stream=stream,
            row_scale=row_scale,
            alpha_up=alpha_up,
            row_amax=row_amax,
            q_out=q_out,
            q_sf=q_sf,
            q_scale=q_scale,
            q_ctl=q_ctl,
            c_out=c_out,
            sched_ctl=sched_ctl,
            sm_dom=sm_dom,
        )

    @staticmethod
    def can_implement(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
        mma_inst_shape: Tuple[int, int, int],
        mma_tiler: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        m: int,
        n: int,
        k: int,
        l: int,  # noqa: E741
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """Check if the GEMM can be implemented."""
        # Check data types
        if not GroupedGemmRowsKernel.is_valid_dtypes_and_scale_factor_vec_size(
            a_dtype, b_dtype, sf_dtype, sf_vec_size, c_dtype
        ):
            return False

        # Check layouts
        if not GroupedGemmRowsKernel.is_valid_layouts(
            a_dtype, b_dtype, c_dtype, a_major, b_major, c_major
        ):
            return False

        # Check MMA tiler and cluster shape
        if not GroupedGemmRowsKernel.is_valid_mma_tiler_and_cluster_shape(
            a_dtype, b_dtype, mma_inst_shape, mma_tiler, cluster_shape_mn
        ):
            return False

        # Check tensor alignment
        if not GroupedGemmRowsKernel.is_valid_tensor_alignment(
            m, n, k, l, a_dtype, b_dtype, c_dtype, a_major, b_major, c_major
        ):
            return False

        return True


# ============================================================================
# Run utilities
# ============================================================================
