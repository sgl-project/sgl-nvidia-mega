# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Inline-PTX helpers the MoE kernels need beyond ``primitives/ptx_core``.

Each is a candidate for ``primitives/ptx_core``; moved there unchanged, the MoE bodies
then import it from there."""

from __future__ import annotations

from cutlass import Float32, Int32, Int64, Uint16, Uint32

from ..primitives.ptx_core import ptx_arith, ptx_op

# --------------------------------------------------------- plain-form arithmetic (pure)


@ptx_arith(Float32)
def add_f32(a, b):
    """Plain ``add.f32``."""
    return "add.f32 $0, $1, $2;", "=f,f,f", (Float32(a), Float32(b))


@ptx_arith(Float32)
def mul_f32(a, b):
    """Plain ``mul.f32``."""
    return "mul.f32 $0, $1, $2;", "=f,f,f", (Float32(a), Float32(b))


@ptx_arith(Float32)
def round_bf16(x):
    """``float(bfloat16_rn(x))`` as the CUDA headers compute it (cvt pair)."""
    code = "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; cvt.f32.bf16 $0, h; }"
    return code, "=f,f", (Float32(x),)


@ptx_arith(Float32)
def bf16_to_f32(bits):
    """FP32 of a BF16 bit pattern (``cvt.f32.bf16``)."""
    return "cvt.f32.bf16 $0, $1;", "=f,h", (Uint16(bits),)


@ptx_arith(Int32)
def popc(x):
    return "popc.b32 $0, $1;", "=r,r", (Uint32(x),)


@ptx_arith(Int32)
def clz(x):
    return "clz.b32 $0, $1;", "=r,r", (Uint32(x),)


@ptx_arith(Int32)
def ctz(x):
    """Trailing zero count (32 for 0)."""
    return "{ .reg .b32 t; brev.b32 t, $1; clz.b32 $0, t; }", "=r,r", (Uint32(x),)


# ----------------------------------------------- global loads / stores (side-effecting)


@ptx_op(Float32)
def ld_cg_f32(address):
    """Coherent-at-L2 load of data written during the step."""
    return "ld.global.cg.f32 $0, [$1];", "=f,l", (Int64(address),)


@ptx_op(Int32)
def ld_cg_s32(address):
    return "ld.global.cg.s32 $0, [$1];", "=r,l", (Int64(address),)


@ptx_op((Float32, Float32, Float32, Float32))
def ld_cg_v4_f32(address):
    return (
        "ld.global.cg.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        (Int64(address),),
    )


@ptx_op((Uint32, Uint32, Uint32, Uint32))
def ld_cg_v4_u32(address):
    return (
        "ld.global.cg.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        (Int64(address),),
    )


@ptx_op((Float32, Float32, Float32, Float32))
def ld_nc_v4_f32(address):
    """Non-coherent load of a static input (weights, biases)."""
    return (
        "ld.global.nc.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        (Int64(address),),
    )


@ptx_op()
def st_global_b32(address, word):
    return "st.global.b32 [$0], $1;", "l,r", (Int64(address), Uint32(word))


@ptx_op()
def st_global_b32f(address, value):
    """The 32 bits of an FP32 value."""
    return "st.global.b32 [$0], $1;", "l,f", (Int64(address), Float32(value))


@ptx_op()
def atom_or_global(address, bits):
    """Relaxed GPU-scope atomic ``or`` into a global word (the old value is
    discarded)."""
    return (
        "{ .reg .b32 o; atom.global.or.b32 o, [$0], $1; }",
        "l,r",
        (Int64(address), Uint32(bits)),
    )


@ptx_op()
def atom_or_shared(addr, bits):
    """Atomic ``or`` into a shared word through its generic address (the old value is
    discarded)."""
    code = (
        "{ .reg .b64 a, g; .reg .b32 o; cvt.u64.u32 a, $0; cvta.shared.u64 g, a; "
        "atom.or.b32 o, [g], $1; }"
    )
    return code, "r,r", (Int32(addr), Uint32(bits))


# ------------------------------------------------------- shared memory (side-effecting)


@ptx_op((Float32, Float32, Float32, Float32))
def ld_shared_v4_f32(addr):
    return "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];", "=f,=f,=f,=f,r", (Int32(addr),)


@ptx_op(Int32)
def ld_shared_s32(addr):
    return "ld.shared.s32 $0, [$1];", "=r,r", (Int32(addr),)


@ptx_op(Int32)
def ld_shared_s16(addr):
    return "ld.shared.s16 $0, [$1];", "=r,r", (Int32(addr),)


@ptx_op(Int32)
def ld_shared_s8(addr):
    return "ld.shared.s8 $0, [$1];", "=r,r", (Int32(addr),)


@ptx_op(Int32)
def ld_shared_u8(addr):
    return "ld.shared.u8 $0, [$1];", "=r,r", (Int32(addr),)


@ptx_op()
def st_shared_b8(addr, value):
    return "st.shared.b8 [$0], $1;", "r,r", (Int32(addr), Uint32(value))


@ptx_op()
def st_shared_v2_u32(addr, a, b):
    return (
        "st.shared.v2.u32 [$0], {$1, $2};",
        "r,r,r",
        (Int32(addr), Uint32(a), Uint32(b)),
    )


# ------------------------------------------------------------- control (side-effecting)


@ptx_op()
def trap_if(flag):
    """``trap`` when ``flag`` is non-zero (a launch-contract violation: fail closed)."""
    return "{ .reg .pred p; setp.ne.b32 p, $0, 0; @p trap; }", "r", (Int32(flag),)


# ---------------------------------------------------- mbarrier / DSMEM (side-effecting)


@ptx_op()
def mbarrier_arrive_remote(remote_bar):
    """Release-arrive (cluster scope) on a peer CTA's mbarrier (``mapa``-ed address)."""
    return (
        "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [$0];",
        "r",
        (Uint32(remote_bar),),
    )


@ptx_op()
def st_async_v4_b32(remote, remote_bar, a, b, c, d):
    """16-byte store to a peer CTA's shared memory completing 16 tx-bytes on its
    mbarrier."""
    code = (
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 "
        "[$0], {$2, $3, $4, $5}, [$1];"
    )
    return (
        code,
        "r,r,r,r,r,r",
        (
            Uint32(remote),
            Uint32(remote_bar),
            Uint32(a),
            Uint32(b),
            Uint32(c),
            Uint32(d),
        ),
    )


@ptx_arith((Float32, Float32))
def bf16x2_unpack(word):
    """The two BF16 halves of ``word`` as FP32 via ``cvt.f32.bf16``: (bits 0-15, bits
    16-31)."""
    code = (
        "{ .reg .b16 l, h; mov.b32 {l, h}, $2; "
        "cvt.f32.bf16 $0, l; cvt.f32.bf16 $1, h; }"
    )
    return code, "=f,=f,r", (Uint32(word),)


@ptx_op()
def st_global_f32(address, value):
    return "st.global.f32 [$0], $1;", "l,f", (Int64(address), Float32(value))


@ptx_op()
def st_async_v4_f32(remote, remote_bar, a, b, c, d):
    """16 bytes of FP32 to a peer CTA's shared memory, completing 16 tx-bytes on its
    mbarrier."""
    code = (
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 "
        "[$0], {$2, $3, $4, $5}, [$1];"
    )
    return (
        code,
        "r,r,f,f,f,f",
        (
            Uint32(remote),
            Uint32(remote_bar),
            Float32(a),
            Float32(b),
            Float32(c),
            Float32(d),
        ),
    )
