# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from TensorRT-LLM (Apache-2.0): the inline-PTX helpers of
# tensorrt_llm/_torch/kernels/blackwell/moe_decode_front.py.
"""Inline-PTX helpers of the router GEMM kernels (CuTe DSL)."""

from __future__ import annotations

from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


def _asm(
    res,
    operands,
    text: str,
    constraints: str,
    side_effects: bool = True,
    *,
    loc=None,
    ip=None,
):
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _v(x, loc=None, ip=None):
    return x.ir_value(loc=loc, ip=ip)


def _i32(x, loc=None, ip=None):
    return Int32(x).ir_value(loc=loc, ip=ip)


def _i64(x, loc=None, ip=None):
    return Int64(x).ir_value(loc=loc, ip=ip)


def _f32(x, loc=None, ip=None):
    return Float32(x).ir_value(loc=loc, ip=ip)


def _struct(n: int):
    return llvm.StructType.get_literal([T.i32()] * n)


def _unpack(out, n: int, loc=None, ip=None):
    return tuple(
        Int32(llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip)) for i in range(n)
    )


# PDL ops clobber memory so no memory access moves across them.
@dsl_user_op
def _gdc_wait(*, loc=None, ip=None) -> None:
    _asm(None, [], "griddepcontrol.wait;", "~{memory}", loc=loc, ip=ip)


@dsl_user_op
def _gdc_launch(*, loc=None, ip=None) -> None:
    _asm(None, [], "griddepcontrol.launch_dependents;", "~{memory}", loc=loc, ip=ip)


@dsl_user_op
def _st_v2(addr: Int64, w0: Int32, w1: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [_i64(addr, loc, ip), _i32(w0, loc, ip), _i32(w1, loc, ip)],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _smem_addr(ptr, *, loc=None, ip=None) -> Int32:
    return Int32(ptr.toint(loc=loc, ip=ip))


@dsl_user_op
def _ld_shared(addr: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_i32(addr, loc, ip)],
            "ld.shared.b32 $0, [$1];",
            "=r,r,~{memory}",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _add(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_f32(a, loc, ip), _f32(b, loc, ip)],
            "add.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _cluster_arrive_release(*, loc=None, ip=None) -> None:
    _asm(
        None, [], "barrier.cluster.arrive.release.aligned;", "~{memory}", loc=loc, ip=ip
    )


@dsl_user_op
def _cluster_wait_acquire(*, loc=None, ip=None) -> None:
    _asm(None, [], "barrier.cluster.wait.acquire.aligned;", "~{memory}", loc=loc, ip=ip)
