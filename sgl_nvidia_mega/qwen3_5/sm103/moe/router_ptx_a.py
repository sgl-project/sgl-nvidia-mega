# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from TensorRT-LLM (Apache-2.0): the inline-PTX helpers of
# tensorrt_llm/_torch/kernels/blackwell/moe_decode_front.py.
# This file has been modified from its original version (_nvfp4_block's zero-scale select).

from __future__ import annotations

from cutlass import Float32, Int32, Int64
from cutlass.cutlass_dsl import T, dsl_user_op

from .router_ptx import _asm, _f32, _i32, _i64, _struct, _unpack  # noqa: F401


@dsl_user_op
def _ld_v4(addr: Int64, *, loc=None, ip=None):
    out = _asm(
        _struct(4),
        [_i64(addr, loc, ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l,~{memory}",
        loc=loc,
        ip=ip,
    )
    return _unpack(out, 4, loc, ip)


@dsl_user_op
def _ld_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_i64(addr, loc, ip)],
            "ld.global.f32 $0, [$1];",
            "=f,l,~{memory}",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_b32(addr: Int64, value: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [_i64(addr, loc, ip), _i32(value, loc, ip)],
        "st.global.b32 [$0], $1;",
        "l,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_b16(addr: Int64, value: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [_i64(addr, loc, ip), _i32(value, loc, ip)],
        "{\n\t.reg .b16 h;\n\tcvt.u16.u32 h, $1;\n\tst.global.b16 [$0], h;\n\t}",
        "l,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_u8(addr: Int64, value: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [_i64(addr, loc, ip), _i32(value, loc, ip)],
        "{\n\t.reg .b16 h;\n\tcvt.u16.u32 h, $1;\n\tst.global.u8 [$0], h;\n\t}",
        "l,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_shared(addr: Int32, value: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [_i32(addr, loc, ip), _i32(value, loc, ip)],
        "st.shared.b32 [$0], $1;",
        "r,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _red_or_shared(addr: Int32, value: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [_i32(addr, loc, ip), _i32(value, loc, ip)],
        "red.shared.or.b32 [$0], $1;",
        "r,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sel_eq(x: Int32, y: Int32, a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    """``a if x == y else b``."""
    return Int32(
        _asm(
            T.i32(),
            [_i32(x, loc, ip), _i32(y, loc, ip), _i32(a, loc, ip), _i32(b, loc, ip)],
            "{\n\t.reg .pred p;\n\tsetp.eq.b32 p, $1, $2;\n\tselp.b32 $0, $3, $4, p;\n\t}",
            "=r,r,r,r,r",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sel_lt(x: Int32, y: Int32, a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    """``a if x < y else b`` (unsigned compare: a negative x is never below y)."""
    return Int32(
        _asm(
            T.i32(),
            [_i32(x, loc, ip), _i32(y, loc, ip), _i32(a, loc, ip), _i32(b, loc, ip)],
            "{\n\t.reg .pred p;\n\tsetp.lt.u32 p, $1, $2;\n\tselp.b32 $0, $3, $4, p;\n\t}",
            "=r,r,r,r,r",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bit(t: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_i32(t, loc, ip)],
            "{\n\t.reg .b32 one;\n\tmov.b32 one, 1;\n\tshl.b32 $0, one, $1;\n\t}",
            "=r,r",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _key_val(key: Int32, *, loc=None, ip=None) -> Float32:
    """The FP32 value of a key's BF16 logit (inverse twiddle)."""
    return Float32(
        _asm(
            T.f32(),
            [_i32(key, loc, ip)],
            "{\n\t.reg .b32 tw, m;\n\t.reg .pred p;\n\tshr.u32 tw, $1, 16;\n\t"
            "and.b32 m, tw, 32768;\n\tsetp.ne.u32 p, m, 0;\n\tselp.b32 m, 32768, 65535, p;\n\t"
            "xor.b32 tw, tw, m;\n\tshl.b32 tw, tw, 16;\n\tmov.b32 $0, tw;\n\t}",
            "=f,r",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sub(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_f32(a, loc, ip), _f32(b, loc, ip)],
            "sub.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _nvfp4_block(
    w0: Int32,
    w1: Int32,
    w2: Int32,
    w3: Int32,
    w4: Int32,
    w5: Int32,
    w6: Int32,
    w7: Int32,
    gs: Float32,
    *,
    loc=None,
    ip=None,
):
    unpack = []
    for i in range(8):
        unpack.append(f"shl.b32 f{2 * i}, ${3 + i}, 16;")
        unpack.append(f"and.b32 f{2 * i + 1}, ${3 + i}, 0xffff0000;")
    amax = ["abs.f32 m, f0;"]
    for j in range(1, 16):
        amax.append(f"abs.f32 a, f{j};")
        amax.append("max.f32 m, m, a;")
    pack = []
    for j in range(8):
        word = "u" if j < 4 else "v"
        pack.append(f"mul.f32 y0, osc, f{2 * j};")
        pack.append(f"mul.f32 y1, osc, f{2 * j + 1};")
        pack.append("cvt.rn.satfinite.e2m1x2.f32 q, y1, y0;")
        pack.append("cvt.u16.u8 hq, q;")
        pack.append("cvt.u32.u16 c, hq;")
        if j % 4 == 0:
            pack.append(f"mov.b32 {word}, c;")
        else:
            pack.append(f"shl.b32 c, c, {8 * (j % 4)};")
            pack.append(f"or.b32 {word}, {word}, c;")
    text = "\n\t".join(
        [
            "{",
            ".reg .b32 " + ", ".join(f"f{j}" for j in range(16)) + ";",
            ".reg .b32 a, m, t, s, r6, rg, sff, osc, y0, y1, c, u, v, sfw;",
            ".reg .b16 sfh, h0, h1, hq;",
            ".reg .b8 q;",
            ".reg .pred nz;",
        ]
        + unpack
        + amax
        + [
            "mov.b32 r6, 0f40C00000;",
            "rcp.approx.ftz.f32 r6, r6;",
            "mul.f32 t, r6, m;",
            "mul.f32 s, t, $11;",
            "cvt.rn.satfinite.e4m3x2.f32 sfh, 0f00000000, s;",
            "cvt.rn.f16x2.e4m3x2 sfw, sfh;",
            "mov.b32 {h0, h1}, sfw;",
            "cvt.f32.f16 sff, h0;",
            "rcp.approx.ftz.f32 rg, $11;",
            "mul.f32 t, rg, sff;",
            "rcp.approx.ftz.f32 osc, t;",
            "setp.neu.f32 nz, sff, 0f00000000;",
            "selp.f32 osc, osc, 0f00000000, nz;",
        ]
        + pack
        + [
            "mov.b32 $0, u;",
            "mov.b32 $1, v;",
            "cvt.u32.u16 c, sfh;",
            "and.b32 $2, c, 255;",
            "}",
        ]
    )
    out = _asm(
        _struct(3),
        [_i32(w, loc, ip) for w in (w0, w1, w2, w3, w4, w5, w6, w7)]
        + [_f32(gs, loc, ip)],
        text,
        "=r,=r,=r,r,r,r,r,r,r,r,r,f",
        False,
        loc=loc,
        ip=ip,
    )
    return _unpack(out, 3, loc, ip)


@dsl_user_op
def _exp_fast(x: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_f32(x, loc, ip)],
            "{\n\t.reg .f32 t;\n\tmul.ftz.f32 t, $1, 0f3FB8AA3B;\n\tex2.approx.ftz.f32 $0, t;\n\t}",
            "=f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _div_full_bf16(a: Float32, b: Float32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_f32(a, loc, ip), _f32(b, loc, ip)],
            "{\n\t.reg .b32 q;\n\t.reg .b16 h;\n\tdiv.full.ftz.f32 q, $1, $2;\n\t"
            "cvt.rn.bf16.f32 h, q;\n\tcvt.u32.u16 $0, h;\n\t}",
            "=r,f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )
