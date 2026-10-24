# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DSL-level helpers of the draft kernels: static shared memory, bit casts and the
bf16 / NaN bit tests. Every PTX instruction the kernels emit comes from
``primitives.ptx_core`` / ``primitives.ptx_sys``.
"""

import cutlass.cute as cute
from cutlass import Float32, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

# ---- shared memory


def static_smem(dtype, shape, stride=None):
    """A statically allocated shared-memory tensor (16-byte aligned) of ``shape``
    (row-major unless ``stride``).
    """
    size = 1
    for d in shape if isinstance(shape, tuple) else (shape,):
        size *= d
    ptr = cute.arch.alloc_smem(dtype, size, 16)
    return cute.make_tensor(ptr, cute.make_layout(shape, stride=stride))


# ---- bit casts (pure)


@dsl_user_op
def f32_from_bits(bits, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.bitcast(
            Float32.mlir_type, Uint32(bits).ir_value(loc=loc, ip=ip), loc=loc, ip=ip
        )
    )


@dsl_user_op
def bits_from_f32(value, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.bitcast(
            Uint32.mlir_type, Float32(value).ir_value(loc=loc, ip=ip), loc=loc, ip=ip
        )
    )


def bf16_low(word):
    """FP32 value of the low BF16 half of a 32-bit word (exact)."""
    return f32_from_bits(Uint32(word) << Uint32(16))


def bf16_high(word):
    """FP32 value of the high BF16 half of a 32-bit word (exact)."""
    return f32_from_bits(Uint32(word) & Uint32(0xFFFF0000))


def is_nan(value):
    """True iff ``value`` is a NaN, from its bits."""
    return (bits_from_f32(value) & Uint32(0x7FFFFFFF)) > Uint32(0x7F800000)
