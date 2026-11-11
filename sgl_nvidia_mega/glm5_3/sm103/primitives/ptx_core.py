# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Inline-PTX helpers shared by the kernels of this package.

Every helper names one PTX instruction (or a short fixed sequence) and emits exactly
that instruction sequence.
Two decorators build them:

* ``ptx_op``: memory, synchronization and protocol instructions. They are
  emitted with side effects and a ``memory`` clobber.
* ``ptx_arith``: arithmetic with an explicit rounding mode. They are pure (no
  side effects, no clobber).
  Floating-point arithmetic on values that reach an output goes through these
  helpers, never through Python operators or ``cute.math``: ``add.rn`` /
  ``mul.rn`` each round their own result, and ``fma.rn`` rounds once.

Barrier and warp-collective instructions (``bar.sync``, ``barrier.cluster``,
shuffles, votes) use the ``cute.arch`` intrinsics; this module only re-exports
them under one name per instruction.

Operands are CuTe DSL numerics, Python ints / floats, or CuTe pointers (whose
address is used: 32-bit for shared memory, 64-bit for global memory).
"""

from functools import wraps

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Float64, Int32, Int64, Uint16, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

__all__ = [
    "ptx_op",
    "ptx_arith",
    "shared_address",
    # programmatic dependent launch
    "pdl_launch_dependents",
    "pdl_wait",
    # fences
    "fence_proxy_async_global",
    "fence_proxy_async_shared_cta",
    "fence_mbarrier_init_release_cluster",
    # mbarrier
    "mbarrier_init",
    "mbarrier_arrive",
    "mbarrier_arrive_expect_tx",
    "mbarrier_expect_tx",
    "mbarrier_try_wait_parity",
    "mbarrier_wait_parity",
    # bulk copies and L2 hints
    "bulk_copy_g2s",
    "bulk_prefetch_l2",
    "l2_policy",
    # loads and stores
    "ld_global_v4",
    "ld_shared_v4",
    "ld_shared_f32",
    "ld_shared_u32",
    "st_shared_u32",
    "st_shared_f32",
    "st_shared_v2_f32",
    "st_shared_v4",
    "st_global_b16",
    "st_global_u32",
    "st_global_v4",
    # cluster and distributed shared memory
    "sync_threads",
    "cluster_arrive_release",
    "cluster_arrive_relaxed",
    "cluster_wait_acquire",
    "mapa_shared_cluster",
    "ld_shared_cluster_u32",
    "ld_shared_cluster_f32",
    "ld_shared_cluster_v4",
    "st_async_f32",
    "st_async_v2_f32",
    # warp-level matrix instructions
    "mma_m16n8k16_bf16",
    "mma_m16n8k32_e4m3",
    "ldmatrix_x4",
    "ldmatrix_x2",
    # arithmetic with an explicit rounding mode
    "add_rn",
    "sub_rn",
    "mul_rn",
    "fma_rn",
    "div_rn",
    "div_full",
    "rsqrt_approx",
    "max_nan",
    "cvt_rn_bf16_f32",
    "cvt_rn_bf16x2_f32",
    "cvt_rn_satfinite_e4m3x2_f32",
    "cvt_rn_f16x2_e4m3x2",
    "cvt_rn_f16_f32",
    "bf16x2_to_f32x2",
    "ex2_approx",
    "prmt",
    "e4m3x2_to_f32x2",
    "st_shared_u8",
    "ld_shared_v2_f32",
    "ld_global_s32",
    "ld_global_s64",
    "ld_global_f32",
    "ld_global_u16",
    "st_global_f32",
    "max_f32",
    "abs_f32",
    "f32_bits",
    "bits_f32",
    "bf16_to_f32",
    "neg_f32",
    "mbarrier_wait_parity_spin",
    "st_async_v4_b32",
    "ld_global_u32_if",
    "ld_global_v4_if",
    "ld_global_v4_no_allocate",
    "st_global_u64",
]

_BY_CONSTRAINT = {
    "r": (Int32, Uint32),
    "l": (Int64, Uint64),
    "h": (Uint16,),
    "f": (Float32,),
    "d": (Float64,),
}


def _operand(kind, value, loc, ip):
    """IR value of one asm input for constraint letter ``kind``."""
    if hasattr(value, "toint"):  # a CuTe pointer: its address
        value = value.toint(loc=loc, ip=ip)
    accepted = _BY_CONSTRAINT[kind]
    if not isinstance(value, accepted):
        value = accepted[0](value)
    return value.ir_value(loc=loc, ip=ip)


def _emit(code, constraints, operands, result, side_effects, loc, ip):
    kinds = [c for c in constraints.split(",") if c and c[0] not in "=~"]
    if len(kinds) != len(operands):
        raise ValueError(f"asm operand count {len(operands)} != constraints {kinds}")
    values = [_operand(k, v, loc, ip) for k, v in zip(kinds, operands)]
    if result is None:
        ir_type = None
    elif isinstance(result, tuple):
        ir_type = llvm.StructType.get_literal([t.mlir_type for t in result])
    else:
        ir_type = result.mlir_type
    out = llvm.inline_asm(
        ir_type,
        values,
        code,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    if result is None:
        return None
    if isinstance(result, tuple):
        return tuple(
            t(llvm.extractvalue(t.mlir_type, out, [i], loc=loc, ip=ip))
            for i, t in enumerate(result)
        )
    return result(out)


def _decorator(result, side_effects, clobber):
    def wrap(fn):
        @dsl_user_op
        @wraps(fn)
        def op(*args, loc=None, ip=None, **kwargs):
            code, constraints, operands, *rest = fn(*args, **kwargs)
            res = rest[0] if rest else result
            if clobber:
                constraints = (
                    f"{constraints},~{{memory}}" if constraints else "~{memory}"
                )
            return _emit(code, constraints, tuple(operands), res, side_effects, loc, ip)

        return op

    return wrap


def ptx_op(result=None):
    """Decorator for memory / synchronization instructions.

    The wrapped function returns ``(code, constraints, operands[, result])``;
    the instruction is emitted with side effects and a ``memory`` clobber.
    """
    return _decorator(result, True, True)


def ptx_arith(result):
    """Decorator for pure arithmetic: no side effects, no clobber."""
    return _decorator(result, False, False)


def shared_address(pointer):
    """32-bit shared-memory address of a CuTe shared-memory pointer."""
    return Int32(pointer.toint())


def _imm_or_reg(value, index):
    """An immediate for a Python int, else operand ``$index``."""
    return (str(value), ()) if isinstance(value, int) else (f"${index}", (value,))


# ----------------------------------------------------------------------------
# Programmatic dependent launch


@ptx_op()
def pdl_launch_dependents():
    return "griddepcontrol.launch_dependents;", "", ()


@ptx_op()
def pdl_wait():
    return "griddepcontrol.wait;", "", ()


# ----------------------------------------------------------------------------
# Fences


@ptx_op()
def fence_proxy_async_global():
    return "fence.proxy.async.global;", "", ()


@ptx_op()
def fence_proxy_async_shared_cta():
    return "fence.proxy.async.shared::cta;", "", ()


@ptx_op()
def fence_mbarrier_init_release_cluster():
    return "fence.mbarrier_init.release.cluster;", "", ()


# ----------------------------------------------------------------------------
# mbarrier (``bar`` is a 32-bit shared-memory address). ``sem_scope`` is the
# PTX qualifier string, e.g. "" (the instruction's default), ".release.cta",
# ".release.cluster", ".acquire.cluster", ".relaxed.cta".


@ptx_op()
def mbarrier_init(bar, count):
    return "mbarrier.init.shared::cta.b64 [$0], $1;", "r,r", (bar, Int32(count))


@ptx_op()
def mbarrier_arrive(bar, sem_scope=""):
    return f"mbarrier.arrive{sem_scope}.shared::cta.b64 _, [$0];", "r", (bar,)


@ptx_op()
def mbarrier_arrive_expect_tx(bar, tx_bytes, sem_scope=""):
    code = f"mbarrier.arrive.expect_tx{sem_scope}.shared::cta.b64 _, [$0], $1;"
    return code, "r,r", (bar, Int32(tx_bytes))


@ptx_op()
def mbarrier_expect_tx(bar, tx_bytes, sem_scope=""):
    code = f"mbarrier.expect_tx{sem_scope}.shared::cta.b64 [$0], $1;"
    return code, "r,r", (bar, Int32(tx_bytes))


@ptx_op(Uint32)
def mbarrier_try_wait_parity(bar, parity, sem_scope=""):
    """1 if the phase with ``parity`` has completed, else 0 (one probe)."""
    p, extra = _imm_or_reg(parity, 2)
    code = (
        "{ .reg .pred P; "
        f"mbarrier.try_wait.parity{sem_scope}.shared::cta.b64 P, [$1], {p}; "
        "selp.u32 $0, 1, 0, P; }"
    )
    constraints = "=r,r" + (",r" if extra else "")
    return code, constraints, (bar, *(Int32(v) for v in extra))


@cute.jit
def mbarrier_wait_parity(bar, parity, sem_scope: cutlass.Constexpr[str] = ""):
    """Probe until the phase with ``parity`` has completed."""
    done = mbarrier_try_wait_parity(bar, parity, sem_scope)
    while done == Uint32(0):
        done = mbarrier_try_wait_parity(bar, parity, sem_scope)


@ptx_op()
def mbarrier_wait_parity_spin(bar, parity, sem_scope=""):
    """Probe until the phase with ``parity`` has completed, as one asm loop (the
    form whose op sequence matches a kernel that spins on ``try_wait`` in asm);
    ``sem_scope`` as for ``mbarrier_try_wait_parity``."""
    code = (
        "{ .reg .pred p; MBW_${:uid}: "
        f"mbarrier.try_wait.parity{sem_scope}.shared::cta.b64 p, [$0], $1; "
        "@!p bra MBW_${:uid}; }"
    )
    return code, "r,r", (Uint32(bar), Uint32(parity))


# ----------------------------------------------------------------------------
# Bulk copies and L2 hints (``src`` / ``addr`` are 64-bit global addresses)


@ptx_op()
def bulk_copy_g2s(dst, src, size, bar, policy=None):
    """Bulk copy of ``size`` bytes global -> shared, completing on ``bar``."""
    base = "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
    if policy is None:
        return f"{base} [$0], [$1], $2, [$3];", "r,l,r,r", (dst, src, Int32(size), bar)
    code = f"{base}.L2::cache_hint [$0], [$1], $2, [$3], $4;"
    return code, "r,l,r,r,l", (dst, src, Int32(size), bar, policy)


@ptx_op()
def bulk_prefetch_l2(addr, size, policy=None):
    """Bulk prefetch of ``size`` bytes into L2 (a hint; no output depends on it)."""
    if policy is None:
        return "cp.async.bulk.prefetch.L2.global [$0], $1;", "l,r", (addr, Int32(size))
    code = "cp.async.bulk.prefetch.L2.global.L2::cache_hint [$0], $1, $2;"
    return code, "l,r,l", (addr, Int32(size), policy)


def l2_policy(priority: str):
    """L2 cache policy (fraction 1.0) with eviction ``priority``: "evict_last",
    "evict_normal", "evict_first" or "evict_unchanged"."""
    return _createpolicy(priority)


@ptx_arith(Uint64)
def _createpolicy(priority):
    return f"createpolicy.fractional.L2::{priority}.b64 $0, 1.0;", "=l", ()


# ----------------------------------------------------------------------------
# Loads and stores. Global loads here are coherent (never ``.nc``): they read
# data that may be written in the same step, after the PDL wait.


@ptx_op((Uint32, Uint32, Uint32, Uint32))
def ld_global_v4(addr):
    return "ld.global.v4.u32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l", (addr,)


@ptx_op((Uint32, Uint32, Uint32, Uint32))
def ld_shared_v4(addr):
    return "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,r", (addr,)


@ptx_op(Float32)
def ld_shared_f32(addr):
    return "ld.shared.f32 $0, [$1];", "=f,r", (addr,)


@ptx_op(Uint32)
def ld_shared_u32(addr):
    return "ld.shared.u32 $0, [$1];", "=r,r", (addr,)


@ptx_op()
def st_shared_u32(addr, value):
    return "st.shared.u32 [$0], $1;", "r,r", (addr, Uint32(value))


@ptx_op()
def st_shared_f32(addr, value):
    return "st.shared.f32 [$0], $1;", "r,f", (addr, value)


@ptx_op()
def st_shared_v2_f32(addr, a, b):
    return "st.shared.v2.f32 [$0], {$1, $2};", "r,f,f", (addr, a, b)


@ptx_op()
def st_shared_v4(addr, words):
    return "st.shared.v4.u32 [$0], {$1, $2, $3, $4};", "r,r,r,r,r", (addr, *words)


@ptx_op()
def st_global_b16(addr, value):
    return "st.global.b16 [$0], $1;", "l,h", (addr, value)


@ptx_op()
def st_global_u32(addr, value):
    return "st.global.u32 [$0], $1;", "l,r", (addr, Uint32(value))


@ptx_op()
def st_global_v4(addr, words):
    return "st.global.v4.u32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", (addr, *words)


# ----------------------------------------------------------------------------
# CTA / cluster barriers (cute.arch intrinsics) and distributed shared memory


def sync_threads():
    """``bar.sync 0``: all threads of the CTA."""
    cute.arch.sync_threads()


def cluster_arrive_release():
    """``barrier.cluster.arrive`` with its default release semantics."""
    cute.arch.cluster_arrive()


def cluster_arrive_relaxed():
    """``barrier.cluster.arrive.relaxed``."""
    cute.arch.cluster_arrive_relaxed()


def cluster_wait_acquire():
    """``barrier.cluster.wait`` with its default acquire semantics."""
    cute.arch.cluster_wait()


@ptx_arith(Uint32)
def mapa_shared_cluster(addr, rank):
    """Shared-memory address of ``addr`` in the CTA of cluster rank ``rank``."""
    return "mapa.shared::cluster.u32 $0, $1, $2;", "=r,r,r", (addr, Int32(rank))


@ptx_op(Uint32)
def ld_shared_cluster_u32(addr):
    return "ld.shared::cluster.u32 $0, [$1];", "=r,r", (addr,)


@ptx_op(Float32)
def ld_shared_cluster_f32(addr):
    return "ld.shared::cluster.f32 $0, [$1];", "=f,r", (addr,)


@ptx_op((Uint32, Uint32, Uint32, Uint32))
def ld_shared_cluster_v4(addr):
    code = "ld.shared::cluster.v4.u32 {$0, $1, $2, $3}, [$4];"
    return code, "=r,=r,=r,=r,r", (addr,)


@ptx_op()
def st_async_f32(remote, value, remote_bar):
    """Asynchronous store to a peer CTA's shared memory, completing on its mbarrier
    (both addresses already mapped with ``mapa_shared_cluster``)."""
    code = "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [$0], $1, [$2];"
    return code, "r,f,r", (remote, value, remote_bar)


@ptx_op()
def st_async_v2_f32(remote, a, b, remote_bar):
    code = (
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 "
        "[$0], {$1, $2}, [$3];"
    )
    return code, "r,f,f,r", (remote, a, b, remote_bar)


@ptx_op()
def st_async_v4_b32(remote, w0, w1, w2, w3, remote_bar):
    """16 bytes to a peer CTA's shared memory, completing on its mbarrier."""
    code = (
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 "
        "[$0], {$1, $2, $3, $4}, [$5];"
    )
    return code, "r,r,r,r,r,r", (remote, w0, w1, w2, w3, remote_bar)


# ----------------------------------------------------------------------------
# Warp-level matrix instructions


@ptx_arith((Float32, Float32, Float32, Float32))
def mma_m16n8k16_bf16(a, b, c):
    """``mma.sync`` m16n8k16, BF16 inputs, FP32 accumulate: a 4-tuple of A
    words, a 2-tuple of B words, a 4-tuple of FP32 accumulators."""
    code = (
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};"
    )
    return code, "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f", (*a, *b, *c)


@ptx_arith((Float32, Float32, Float32, Float32))
def mma_m16n8k32_e4m3(a, b, c):
    """``mma.sync`` m16n8k32, E4M3 inputs, FP32 accumulate."""
    code = (
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};"
    )
    return code, "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f", (*a, *b, *c)


@ptx_op((Uint32, Uint32, Uint32, Uint32))
def ldmatrix_x4(addr, trans=False):
    t = ".trans" if trans else ""
    code = f"ldmatrix.sync.aligned.m8n8.x4{t}.shared.b16 {{$0, $1, $2, $3}}, [$4];"
    return code, "=r,=r,=r,=r,r", (addr,)


@ptx_op((Uint32, Uint32))
def ldmatrix_x2(addr, trans=False):
    t = ".trans" if trans else ""
    code = f"ldmatrix.sync.aligned.m8n8.x2{t}.shared.b16 {{$0, $1}}, [$2];"
    return code, "=r,=r,r", (addr,)


# ----------------------------------------------------------------------------
# Arithmetic with an explicit rounding mode (pure)


@ptx_arith(Float32)
def add_rn(a, b):
    return "add.rn.f32 $0, $1, $2;", "=f,f,f", (a, b)


@ptx_arith(Float32)
def sub_rn(a, b):
    return "sub.rn.f32 $0, $1, $2;", "=f,f,f", (a, b)


@ptx_arith(Float32)
def mul_rn(a, b):
    return "mul.rn.f32 $0, $1, $2;", "=f,f,f", (a, b)


@ptx_arith(Float32)
def fma_rn(a, b, c):
    return "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", (a, b, c)


@ptx_arith(Float32)
def div_rn(a, b):
    return "div.rn.f32 $0, $1, $2;", "=f,f,f", (a, b)


@ptx_arith(Float32)
def div_full(a, b):
    return "div.full.f32 $0, $1, $2;", "=f,f,f", (a, b)


@ptx_arith(Float32)
def rsqrt_approx(a, ftz=False):
    """``rsqrt.approx.f32`` (``ftz``: subnormal inputs and results flush to zero)."""
    op = "rsqrt.approx.ftz.f32" if ftz else "rsqrt.approx.f32"
    return f"{op} $0, $1;", "=f,f", (a,)


@ptx_arith(Float32)
def max_nan(a, b):
    return "max.NaN.f32 $0, $1, $2;", "=f,f,f", (a, b)


@ptx_arith(Uint16)
def cvt_rn_bf16_f32(a):
    return "cvt.rn.bf16.f32 $0, $1;", "=h,f", (a,)


@ptx_arith(Uint32)
def cvt_rn_bf16x2_f32(lo, hi):
    """BF16x2 word: ``lo`` in bits 0-15, ``hi`` in bits 16-31."""
    return "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f", (lo, hi)


@ptx_arith(Uint16)
def cvt_rn_satfinite_e4m3x2_f32(lo, hi):
    """E4M3x2 (satfinite): ``lo`` in bits 0-7, ``hi`` in bits 8-15."""
    return "cvt.rn.satfinite.e4m3x2.f32 $0, $2, $1;", "=h,f,f", (lo, hi)


@ptx_arith(Uint32)
def cvt_rn_f16x2_e4m3x2(pair):
    return "cvt.rn.f16x2.e4m3x2 $0, $1;", "=r,h", (pair,)


@ptx_arith(Uint16)
def cvt_rn_f16_f32(a):
    return "cvt.rn.f16.f32 $0, $1;", "=h,f", (a,)


@ptx_arith((Float32, Float32))
def bf16x2_to_f32x2(word):
    """The two BF16 halves of ``word`` as FP32 (exact): (bits 0-15, bits 16-31)."""
    return "shl.b32 $0, $2, 16; and.b32 $1, $2, 0xffff0000;", "=f,=f,r", (word,)


@ptx_arith(Float32)
def ex2_approx(a):
    """``ex2.approx.f32`` (not flushed to zero)."""
    return "ex2.approx.f32 $0, $1;", "=f,f", (a,)


@ptx_arith(Uint32)
def prmt(a, b, selector: int):
    """``prmt.b32``: byte permute of the pair (``b``:``a``) by an immediate selector."""
    return f"prmt.b32 $0, $1, $2, {selector:#x};", "=r,r,r", (Uint32(a), Uint32(b))


@ptx_arith((Float32, Float32))
def e4m3x2_to_f32x2(pair):
    """Two E4M3 values (bits 0-7, 8-15) as FP32, exact: (low byte, high byte)."""
    code = (
        "{ .reg .b32 h; .reg .b16 l, u; cvt.rn.f16x2.e4m3x2 h, $2; mov.b32 {l, u}, h; "
        "cvt.f32.f16 $0, l; cvt.f32.f16 $1, u; }"
    )
    return code, "=f,=f,h", (pair,)


@ptx_op()
def st_shared_u8(addr, value):
    return "st.shared.u8 [$0], $1;", "r,h", (addr, Uint16(value))


@ptx_op((Float32, Float32))
def ld_shared_v2_f32(addr):
    return "ld.shared.v2.f32 {$0, $1}, [$2];", "=f,=f,r", (addr,)


@ptx_op(Int32)
def ld_global_s32(addr):
    """Coherent 32-bit global load (data written in the same step, read after the PDL
    wait)."""
    return "ld.global.s32 $0, [$1];", "=r,l", (addr,)


@ptx_op(Int64)
def ld_global_s64(addr):
    return "ld.global.s64 $0, [$1];", "=l,l", (addr,)


@ptx_op(Float32)
def ld_global_f32(addr):
    return "ld.global.f32 $0, [$1];", "=f,l", (addr,)


@ptx_op(Uint16)
def ld_global_u16(addr):
    return "ld.global.u16 $0, [$1];", "=h,l", (addr,)


@ptx_op()
def st_global_f32(addr, value):
    return "st.global.f32 [$0], $1;", "l,f", (addr, value)


@ptx_arith(Float32)
def max_f32(a, b):
    """``max.f32``: a NaN operand yields the other operand."""
    return "max.f32 $0, $1, $2;", "=f,f,f", (a, b)


@ptx_arith(Float32)
def abs_f32(a):
    return "abs.f32 $0, $1;", "=f,f", (a,)


@ptx_arith(Uint32)
def f32_bits(a):
    return "mov.b32 $0, $1;", "=r,f", (a,)


@ptx_arith(Float32)
def bits_f32(a):
    return "mov.b32 $0, $1;", "=f,r", (Uint32(a),)


@ptx_arith(Float32)
def bf16_to_f32(h):
    """Exact BF16 -> FP32 (``cvt.f32.bf16``)."""
    return "cvt.f32.bf16 $0, $1;", "=f,h", (Uint16(h),)


@ptx_arith(Float32)
def neg_f32(a):
    return "neg.f32 $0, $1;", "=f,f", (a,)


@ptx_op((Uint32, Uint32, Uint32, Uint32))
def ld_global_nc_v4(addr):
    """Read-only (``.nc``) 16-byte load: constant data only (weights, norms)."""
    return "ld.global.nc.v4.u32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l", (addr,)


@ptx_op(Float32)
def ld_global_nc_f32(addr):
    """Read-only (``.nc``) load: constant data only."""
    return "ld.global.nc.f32 $0, [$1];", "=f,l", (addr,)


@ptx_op(Uint32)
def ld_global_u32(addr):
    return "ld.global.u32 $0, [$1];", "=r,l", (addr,)


@ptx_op((Uint32, Uint32))
def ld_global_v2(addr):
    return "ld.global.v2.u32 {$0, $1}, [$2];", "=r,=r,l", (addr,)


@ptx_op()
def st_global_v2(addr, a, b):
    return "st.global.v2.u32 [$0], {$1, $2};", "l,r,r", (addr, Uint32(a), Uint32(b))


@ptx_op()
def st_global_u8(addr, value):
    return "st.global.u8 [$0], $1;", "l,h", (addr, Uint16(value))


@ptx_op()
def st_shared_b16(addr, value):
    return "st.shared.b16 [$0], $1;", "r,h", (addr, Uint16(value))


@ptx_arith(Float32)
def min_f32(a, b):
    """``min.f32``: a NaN operand yields the other operand."""
    return "min.f32 $0, $1, $2;", "=f,f,f", (a, b)


@ptx_arith(Float32)
def rcp_rn(a):
    """``rcp.rn.f32``: rounded reciprocal."""
    return "rcp.rn.f32 $0, $1;", "=f,f", (a,)


# ----------------------------------------------------------------------------
# Shared-memory atomics, CTA reductions and further loads / stores


@ptx_op(Uint32)
def atom_shared_add_u32(addr, value):
    return "atom.shared.add.u32 $0, [$1], $2;", "=r,r,r", (addr, Uint32(value))


@ptx_op(Uint32)
def atom_shared_and_b32(addr, value):
    return "atom.shared.and.b32 $0, [$1], $2;", "=r,r,r", (addr, Uint32(value))


@ptx_op(Uint32)
def atom_shared_or_b32(addr, value):
    return "atom.shared.or.b32 $0, [$1], $2;", "=r,r,r", (addr, Uint32(value))


@ptx_op(Int32)
def sync_threads_or(pred):
    """``bar.red.or.pred`` over the CTA (barrier 0): 1 if ``pred`` is non-zero in any
    thread, else 0."""
    code = (
        "{ .reg .pred P, Q; setp.ne.u32 P, $1, 0; bar.red.or.pred Q, 0, P; "
        "selp.u32 $0, 1, 0, Q; }"
    )
    return code, "=r,r", (Uint32(pred),)


@ptx_op(Uint16)
def ld_shared_u16(addr):
    return "ld.shared.u16 $0, [$1];", "=h,r", (addr,)


@ptx_op(Uint64)
def ld_shared_u64(addr):
    return "ld.shared.u64 $0, [$1];", "=l,r", (addr,)


@ptx_op()
def st_shared_u64(addr, value):
    return "st.shared.u64 [$0], $1;", "r,l", (addr, Uint64(value))


@ptx_op((Uint64, Uint64))
def ld_global_v2_u64(addr):
    return "ld.global.v2.u64 {$0, $1}, [$2];", "=l,=l,l", (addr,)


@ptx_op()
def st_shared_v2_u64(addr, a, b):
    return "st.shared.v2.u64 [$0], {$1, $2};", "r,l,l", (addr, Uint64(a), Uint64(b))


@ptx_arith(Uint32)
def lowest_set_bit(x):
    """Index of the lowest set bit of a non-zero word (``__ffs(x) - 1``)."""
    return "{ .reg .b32 t; brev.b32 t, $1; bfind.shiftamt.u32 $0, t; }", "=r,r", (x,)


# Generic-address forms (the address may be in any state space; here always global
# memory reached through a plain pointer or an integer address)


@ptx_op(Float32)
def ld_f32(addr):
    return "ld.f32 $0, [$1];", "=f,l", (addr,)


@ptx_op((Uint64, Uint64))
def ld_v2_u64(addr):
    return "ld.v2.u64 {$0, $1}, [$2];", "=l,=l,l", (addr,)


# Predicated and streaming global accesses (coherent)


@ptx_op(Uint32)
def ld_global_u32_if(addr, pred):
    """Coherent 32-bit load executed only where ``pred`` holds (0 otherwise)."""
    code = (
        "{ .reg .pred p; setp.ne.b32 p, $2, 0; mov.b32 $0, 0; "
        "@p ld.global.b32 $0, [$1]; }"
    )
    return code, "=r,l,r", (addr, Int32(Boolean(pred)))


@ptx_op((Uint32, Uint32, Uint32, Uint32))
def ld_global_v4_if(addr, pred):
    """``ld_global_v4`` executed only where ``pred`` holds (zeros otherwise)."""
    code = (
        "{ .reg .pred p; setp.ne.b32 p, $5, 0; "
        "mov.b32 $0, 0; mov.b32 $1, 0; mov.b32 $2, 0; mov.b32 $3, 0; "
        "@p ld.global.v4.u32 {$0,$1,$2,$3}, [$4]; }"
    )
    return code, "=r,=r,=r,=r,l,r", (addr, Int32(Boolean(pred)))


@ptx_op((Uint32, Uint32, Uint32, Uint32))
def ld_global_v4_no_allocate(addr):
    """Four words of a static weight through the coherent path, no L1 allocation."""
    code = "ld.global.L1::no_allocate.v4.u32 {$0,$1,$2,$3}, [$4];"
    return code, "=r,=r,=r,=r,l", (addr,)


@ptx_op()
def st_global_u64(addr, value):
    return "st.global.u64 [$0], $1;", "l,l", (addr, Uint64(value))
