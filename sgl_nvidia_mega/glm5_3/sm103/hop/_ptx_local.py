# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bit views, shared-memory, cluster and routing helpers used only by the hop kernels.

The rounded arithmetic (``add.rn`` / ``mul.rn`` / ``fma.rn`` / ``rcp.rn`` /
``rsqrt.approx`` / ``cvt.rn.bf16`` / ``cvt.rn.bf16x2``) comes from
``primitives/ptx_core.py``. The helpers kept here either have no equivalent there or
differ from it in what they emit (bit views by ``llvm.bitcast``
instead of ``mov.b32``, shared-memory loads without a ``memory`` clobber, the cluster
mbarrier forms), so moving them is a separate, code-changing step.

Arithmetic helpers are pure (no side effects, no clobber). Shared-memory
synchronization helpers are side-effecting with a ``~{memory}`` clobber. Barriers and
warp collectives use the DSL's NVVM operations (convergent), never raw asm.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cutlass_dsl import dsl_user_op

_MEM = "~{memory}"


def _emit(code, constraints, operands, results, side_effects, loc, ip):
    if not results:
        ir_type = None
    elif len(results) == 1:
        ir_type = results[0].mlir_type
    else:
        ir_type = llvm.StructType.get_literal([r.mlir_type for r in results])
    value = llvm.inline_asm(
        ir_type,
        [v.ir_value(loc=loc, ip=ip) for v in operands],
        code,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    if not results:
        return None
    if len(results) == 1:
        return results[0](value)
    return tuple(
        r(llvm.extractvalue(r.mlir_type, value, [i], loc=loc, ip=ip))
        for i, r in enumerate(results)
    )


# ---------------------------------------------------------------- bit views
def bits_f32(word) -> Float32:
    return Float32(llvm.bitcast(Float32.mlir_type, Uint32(word).ir_value()))


def f32_bits(value) -> Uint32:
    return Uint32(llvm.bitcast(Uint32.mlir_type, Float32(value).ir_value()))


def bf16_lo(word) -> Float32:
    """fp32 value of the low bf16 half of a 32-bit word."""
    return bits_f32(Uint32(word) << 16)


def bf16_hi(word) -> Float32:
    return bits_f32(Uint32(word) & Uint32(0xFFFF0000))


@cute.jit
def no_neg_zero(word: Uint32) -> Uint32:
    """``word`` with each negative-zero bf16 half made +0."""
    w = Uint32(word)
    if (w & Uint32(0xFFFF)) == Uint32(0x8000):
        w = w & Uint32(0xFFFF0000)
    if (w >> 16) == Uint32(0x8000):
        w = w & Uint32(0x0000FFFF)
    return w


# ------------------------------------------------------------ mma (sm80+)
@dsl_user_op
def mma_bf16_m16n8k16(c, a, b, *, loc=None, ip=None):
    """``mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32``; returns 4 fp32."""
    code = (
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0,$1,$2,$3}, "
        "{$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};"
    )
    ops = tuple(Uint32(v) for v in (*a, *b)) + tuple(Float32(v) for v in c)
    cons = "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
    return _emit(code, cons, ops, (Float32,) * 4, False, loc, ip)


# ------------------------------------------------------- mbarrier / DSMEM
@dsl_user_op
def mbar_init(addr, count, *, loc=None, ip=None):
    code = "mbarrier.init.shared::cta.b64 [$0], $1;"
    _emit(code, "r,r," + _MEM, (Uint32(addr), Uint32(count)), (), True, loc, ip)


@dsl_user_op
def fence_mbarrier_init_release_cluster(*, loc=None, ip=None):
    _emit("fence.mbarrier_init.release.cluster;", _MEM, (), (), True, loc, ip)


@dsl_user_op
def mbar_arrive_expect_tx_release_cluster(addr, tx, *, loc=None, ip=None):
    """Cluster-scope release arrive (the peers' ``st.async`` complete_tx on this barrier
    are cluster-scope operations)."""
    code = "mbarrier.arrive.expect_tx.release.cluster.shared::cta.b64 _, [$0], $1;"
    _emit(code, "r,r," + _MEM, (Uint32(addr), Uint32(tx)), (), True, loc, ip)


@dsl_user_op
def mbar_try_wait_parity_acquire_cluster(addr, parity, *, loc=None, ip=None) -> Uint32:
    code = (
        "{ .reg .pred p; mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 "
        "p, [$1], $2; selp.u32 $0, 1, 0, p; }"
    )
    ops = (Uint32(addr), Uint32(parity))
    return _emit(code, "=r,r,r," + _MEM, ops, (Uint32,), True, loc, ip)


@cute.jit
def mbar_wait_cluster(addr: Uint32, parity: Uint32):
    ok = mbar_try_wait_parity_acquire_cluster(addr, parity)
    while ok == Uint32(0):
        ok = mbar_try_wait_parity_acquire_cluster(addr, parity)


@dsl_user_op
def mapa_u32(addr, rank, *, loc=None, ip=None) -> Uint32:
    code = "mapa.shared::cluster.u32 $0, $1, $2;"
    ops = (Uint32(addr), Uint32(rank))
    return _emit(code, "=r,r,r", ops, (Uint32,), True, loc, ip)


@dsl_user_op
def st_async_b32(raddr, value, rmbar, *, loc=None, ip=None):
    code = "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [$0], $1, [$2];"
    ops = (Uint32(raddr), Uint32(value), Uint32(rmbar))
    _emit(code, "r,r,r," + _MEM, ops, (), True, loc, ip)


# ------------------------------------------------------------------ barriers
def bar_sync(barrier_id: int, count: int):
    """Named CTA barrier (``bar.sync id, count``) through the convergent NVVM op."""
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=count)


@dsl_user_op
def cluster_arrive_relaxed(*, loc=None, ip=None):
    """``barrier.cluster.arrive.relaxed.aligned`` (convergent NVVM op)."""
    nvvm.cluster_arrive_relaxed(aligned=True, loc=loc, ip=ip)


@dsl_user_op
def cluster_wait(*, loc=None, ip=None):
    """``barrier.cluster.wait.aligned`` (convergent NVVM op)."""
    nvvm.cluster_wait(aligned=True, loc=loc, ip=ip)


def shfl_xor(value, offset: int):
    return cute.arch.shuffle_sync_bfly(value, offset=offset)


def shfl_up(value, offset: int):
    return cute.arch.shuffle_sync_up(value, offset=offset)


def thread_index() -> Int32:
    tid, _, _ = cute.arch.thread_idx()
    return Int32(tid)


def block_index() -> Int32:
    bid, _, _ = cute.arch.block_idx()
    return Int32(bid)


def grid_size() -> Int32:
    g, _, _ = cute.arch.grid_dim()
    return Int32(g)


def const(x):
    return cutlass.const_expr(x)


# ------------------------------------------------------------- shared memory
@dsl_user_op
def st_shared_v4_u32(addr, w0, w1, w2, w3, *, loc=None, ip=None):
    """16-byte shared store (side-effecting)."""
    code = "st.shared.v4.u32 [$0], {$1,$2,$3,$4};"
    ops = (Uint32(addr), Uint32(w0), Uint32(w1), Uint32(w2), Uint32(w3))
    _emit(code, "r,r,r,r,r", ops, (), True, loc, ip)


@dsl_user_op
def ld_shared_f32(addr, *, loc=None, ip=None) -> Float32:
    """One 32-bit shared-memory load."""
    code = "ld.shared.f32 $0, [$1];"
    return _emit(code, "=f,r", (Uint32(addr),), (Float32,), True, loc, ip)


@dsl_user_op
def ld_shared_v4_u32(addr, *, loc=None, ip=None):
    code = "ld.shared.v4.u32 {$0,$1,$2,$3}, [$4];"
    return _emit(code, "=r,=r,=r,=r,r", (Uint32(addr),), (Uint32,) * 4, True, loc, ip)


# ------------------------------------------------- routing scalars (stock order)
# The stock top-k routing (sigmoid scores + bias, top-8, scaled normalized weights):
# these reproduce its exact instruction forms.
@dsl_user_op
def routing_sigmoid(x, *, loc=None, ip=None) -> Float32:
    """``fma.rn.ftz(tanh.approx(mul.ftz(x, 0.5)), 0.5, 0.5)``."""
    code = (
        "{ .reg .f32 h, t; mul.ftz.f32 h, $1, 0f3F000000; tanh.approx.f32 t, h; "
        "fma.rn.ftz.f32 $0, t, 0f3F000000, 0f3F000000; }"
    )
    return _emit(code, "=f,f", (Float32(x),), (Float32,), False, loc, ip)


@dsl_user_op
def add_ftz(a, b, *, loc=None, ip=None) -> Float32:
    ops = (Float32(a), Float32(b))
    return _emit("add.ftz.f32 $0, $1, $2;", "=f,f,f", ops, (Float32,), False, loc, ip)


@dsl_user_op
def routing_weight(scale, s, total, *, loc=None, ip=None) -> Float32:
    """``bf16(div.approx.ftz(mul.ftz(scale, s), add.ftz(total, 1e-20)))`` as fp32."""
    code = (
        "{ .reg .f32 n, d, q; .reg .b16 h; mul.ftz.f32 n, $1, $2; "
        "add.ftz.f32 d, $3, 0f1E3CE508; "
        "div.approx.ftz.f32 q, n, d; cvt.rn.bf16.f32 h, q; mov.b32 $0, {0, h}; }"
    )
    ops = (Float32(scale), Float32(s), Float32(total))
    return _emit(code, "=f,f,f,f", ops, (Float32,), False, loc, ip)


def umax(a, b):
    return cutlass.max(Uint32(a), Uint32(b))


def umin(a, b):
    return cutlass.min(Uint32(a), Uint32(b))


def redux_max_u32(v):
    return cute.arch.warp_redux_sync(Uint32(v), "umax")


def redux_min_u32(v):
    return cute.arch.warp_redux_sync(Uint32(v), "umin")
