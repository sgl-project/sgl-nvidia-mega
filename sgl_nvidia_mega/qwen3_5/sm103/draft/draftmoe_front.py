# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from TensorRT-LLM (Apache-2.0): the helper functions of
# tensorrt_llm/_torch/kernels/blackwell/moe_decode_front.py.

from __future__ import annotations

from cutlass import Float32, Int32
from cutlass.cutlass_dsl import T, dsl_user_op

from sgl_nvidia_mega.qwen3_5.sm103.moe.router_ptx import _asm, _f32, _i32


@dsl_user_op
def _redux_max_u32(x: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_i32(x, loc, ip)],
            "redux.sync.max.u32 $0, $1, 0xffffffff;",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _max_u32(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_i32(a, loc, ip), _i32(b, loc, ip)],
            "max.u32 $0, $1, $2;",
            "=r,r,r",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sel_eq_f(
    x: Int32, y: Int32, a: Float32, b: Float32, *, loc=None, ip=None
) -> Float32:
    """``a if x == y else b`` (FP32 operands)."""
    return Float32(
        _asm(
            T.f32(),
            [_i32(x, loc, ip), _i32(y, loc, ip), _f32(a, loc, ip), _f32(b, loc, ip)],
            "{\n\t.reg .pred p;\n\tsetp.eq.b32 p, $1, $2;\n\tselp.f32 $0, $3, $4, p;\n\t}",
            "=f,r,r,f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _below(t: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_i32(t, loc, ip)],
            "{\n\t.reg .b32 a;\n\tmov.b32 a, 0xffffffff;\n\tshl.b32 a, a, $1;\n\t"
            "not.b32 $0, a;\n\t}",
            "=r,r",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _min_u32(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_i32(a, loc, ip), _i32(b, loc, ip)],
            "min.u32 $0, $1, $2;",
            "=r,r,r",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _key1(word: Int32, idx: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_i32(word, loc, ip), _i32(idx, loc, ip)],
            "{\n\t.reg .b32 bv, t, m;\n\t.reg .pred p;\n\t"
            "and.b32 bv, $1, 65535;\n\t"
            "and.b32 t, bv, 32768;\n\tsetp.ne.u32 p, t, 0;\n\tselp.b32 m, 65535, 32768, p;\n\t"
            "xor.b32 bv, bv, m;\n\tshl.b32 bv, bv, 16;\n\t"
            "mov.b32 t, 65535;\n\tsub.u32 t, t, $2;\n\tor.b32 $0, bv, t;\n\t}",
            "=r,r,r",
            False,
            loc=loc,
            ip=ip,
        )
    )


def _bitonic_pairs(n: int):
    pairs = []
    k = 2
    while k <= n:
        j = k // 2
        while j > 0:
            for i in range(n):
                partner = i ^ j
                if partner > i:
                    pairs.append((i, partner, (i & k) == 0))
            j //= 2
        k *= 2
    return pairs


_SORT16 = _bitonic_pairs(16)
