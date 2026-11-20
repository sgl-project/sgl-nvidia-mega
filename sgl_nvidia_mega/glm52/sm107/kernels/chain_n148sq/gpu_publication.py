# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Lamport publish / poll accesses. Polls are volatile so every iteration reloads; publishes are weak stores of the
exact width, kept in order by the release that follows them."""
import cutlass
from cutlass._mlir.dialects import llvm, nvvm, vector
from cutlass._mlir import ir
import cutlass.cute as cute
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def acquire_word(pointer: cute.Pointer, *, loc=None, ip=None):
    return cutlass.Uint32(cute.arch.load(pointer, cutlass.Uint32, sem="acquire", scope="gpu", loc=loc, ip=ip))


@dsl_user_op
def fence_sys(*, loc=None, ip=None):
    """Orders this thread's multicast stores before its later system-visible work."""
    cute.arch.fence_acq_rel_sys(loc=loc, ip=ip)


@dsl_user_op
def store_vector(pointer: cute.Pointer, a: cutlass.Uint32, b: cutlass.Uint32,
                 c: cutlass.Uint32, d: cutlass.Uint32, *, loc=None, ip=None):
    nvvm.store_ext(word_vector((a, b, c, d,), loc, ip), as_ptr(1, pointer, loc, ip), loc=loc, ip=ip)


@dsl_user_op
def store_pair(pointer: cute.Pointer, a: cutlass.Uint32, b: cutlass.Uint32, *, loc=None, ip=None):
    nvvm.store_ext(word_vector((a, b,), loc, ip), as_ptr(1, pointer, loc, ip), loc=loc, ip=ip)


@dsl_user_op
def store_word(pointer: cute.Pointer, value: cutlass.Uint32, *, loc=None, ip=None):
    nvvm.store_ext(value.ir_value(loc=loc, ip=ip), as_ptr(1, pointer, loc, ip), loc=loc, ip=ip)


@dsl_user_op
def global_vector(pointer: cute.Pointer, *, loc=None, ip=None):
    v = cute.arch.load(pointer, ir.VectorType.get([4], cutlass.Uint32.mlir_type), loc=loc, ip=ip)
    return tuple(cutlass.Uint32(vector.extract(v, [], [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def global_pair(pointer: cute.Pointer, *, loc=None, ip=None):
    v = cute.arch.load(pointer, ir.VectorType.get([2], cutlass.Uint32.mlir_type), loc=loc, ip=ip)
    return tuple(cutlass.Uint32(vector.extract(v, [], [i], loc=loc, ip=ip)) for i in range(2))


@dsl_user_op
def readonly_vector(pointer: cute.Pointer, *, loc=None, ip=None):
    v = llvm.load(ir.VectorType.get([4], T.i32()), as_ptr(1, pointer, loc, ip), invariant=True, alignment=16, loc=loc,
                  ip=ip)
    return tuple(cutlass.Uint32(vector.extract(v, [], [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def cached_vector(pointer: cute.Pointer, *, loc=None, ip=None):
    v = cute.arch.load(pointer, ir.VectorType.get([4], cutlass.Uint32.mlir_type), cop="cg", loc=loc, ip=ip)
    return tuple(cutlass.Uint32(vector.extract(v, [], [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def lds128(address: cutlass.Uint32, *, loc=None, ip=None):
    v = llvm.load(ir.VectorType.get([4], T.i32()), as_ptr(3, address, loc, ip), alignment=16, loc=loc, ip=ip)
    return tuple(cutlass.Uint32(vector.extract(v, [], [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def shared_vector(pointer: cute.Pointer, *, loc=None, ip=None):
    v = llvm.load(ir.VectorType.get([4], T.i32()), as_ptr(3, pointer, loc, ip), alignment=16, loc=loc, ip=ip)
    return tuple(cutlass.Uint32(vector.extract(v, [], [i], loc=loc, ip=ip)) for i in range(4))

from .common import as_ptr, word_vector
