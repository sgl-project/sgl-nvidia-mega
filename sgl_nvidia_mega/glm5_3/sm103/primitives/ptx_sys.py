# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Inline-PTX helpers for global-memory ordering, cross-GPU signalling and control.

Every helper emits exactly one PTX instruction (or one fixed sequence).

Two emission classes:

* memory-model operations (scoped loads / stores, atomics, fences, proxy fences) are
  side-effecting and carry a ``~{memory}`` clobber;
* plain data loads (``ld.global.cg``, the non-coherent weight load) are side-effecting
  without a clobber.

Pointers are 64-bit generic addresses (``Int64``) unless stated otherwise.
"""

from cutlass import Boolean, Float32, Int32, Int64, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

__all__ = [
    "atom_add_acq_rel_gpu_u32",
    "atom_add_u32",
    "atom_or_b32",
    "atom_or_sys_u32",
    "atom_or_sys_u32_generic",
    "cp_async_cg_16",
    "cp_async_wait_all",
    "create_policy_evict_first",
    "exit_thread",
    "fence_acq_rel_gpu",
    "fence_acq_rel_sys",
    "fence_proxy_async_global",
    "fence_sc_sys",
    "globaltimer",
    "ld_acquire_gpu_u32",
    "ld_acquire_sys_u32",
    "ld_cg_s32",
    "ld_cg_u32",
    "ld_u32",
    "ld_u64",
    "ld_cg_v2_f32",
    "ld_cg_v4_f32",
    "ld_cg_v4_u32",
    "ld_nc_hint_v4_u32",
    "ld_nc_v4_f32",
    "ld_nc_u32",
    "ld_nc_v4_u32_if",
    "ld_relaxed_gpu_u32",
    "ld_relaxed_sys_u32",
    "ld_relaxed_sys_u64",
    "ld_relaxed_sys_v2_b64",
    "ld_volatile_u32",
    "red_add_release_gpu_u32",
    "st_global_u32",
    "st_relaxed_gpu_u32",
    "st_relaxed_sys_u32",
    "st_relaxed_sys_u64",
    "st_relaxed_sys_v2_b64",
    "st_relaxed_sys_v4_u32",
    "st_release_gpu_u32",
    "trap",
    "trap_if",
]

_MEM = "~{memory}"


def _emit(code, constraints, operands, results, side_effects, loc, ip):
    """One ``llvm.inline_asm`` op; ``results`` is a tuple of DSL numeric types."""
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


# --------------------------------------------------------------------- loads
@dsl_user_op
def ld_relaxed_gpu_u32(addr, *, loc=None, ip=None) -> Uint32:
    code = "ld.relaxed.gpu.global.u32 $0, [$1];"
    return _emit(code, "=r,l," + _MEM, (Int64(addr),), (Uint32,), True, loc, ip)


@dsl_user_op
def ld_acquire_gpu_u32(addr, *, loc=None, ip=None) -> Uint32:
    code = "ld.acquire.gpu.global.u32 $0, [$1];"
    return _emit(code, "=r,l," + _MEM, (Int64(addr),), (Uint32,), True, loc, ip)


@dsl_user_op
def ld_acquire_sys_u32(addr, *, loc=None, ip=None) -> Uint32:
    code = "ld.acquire.sys.global.u32 $0, [$1];"
    return _emit(code, "=r,l," + _MEM, (Int64(addr),), (Uint32,), True, loc, ip)


@dsl_user_op
def ld_relaxed_sys_u32(addr, *, loc=None, ip=None) -> Uint32:
    code = "ld.relaxed.sys.global.u32 $0, [$1];"
    return _emit(code, "=r,l," + _MEM, (Int64(addr),), (Uint32,), True, loc, ip)


@dsl_user_op
def ld_relaxed_sys_u64(addr, *, loc=None, ip=None) -> Uint64:
    code = "ld.relaxed.sys.global.u64 $0, [$1];"
    return _emit(code, "=l,l," + _MEM, (Int64(addr),), (Uint64,), True, loc, ip)


@dsl_user_op
def ld_relaxed_sys_v2_b64(addr, *, loc=None, ip=None):
    """Two single-copy-atomic 64-bit records (one 16-byte vector access; PTX ISA "vector
    data types": each element is its own scalar access)."""
    code = "ld.relaxed.sys.global.v2.b64 {$0,$1}, [$2];"
    return _emit(
        code, "=l,=l,l," + _MEM, (Int64(addr),), (Uint64, Uint64), True, loc, ip
    )


@dsl_user_op
def ld_volatile_u32(addr, *, loc=None, ip=None) -> Uint32:
    """``ld.volatile``: for reading a word once (never as a cross-GPU poll)."""
    code = "ld.volatile.global.u32 $0, [$1];"
    return _emit(code, "=r,l," + _MEM, (Int64(addr),), (Uint32,), True, loc, ip)


@dsl_user_op
def ld_u32(addr, *, loc=None, ip=None) -> Uint32:
    """Plain (weak) global load of one 32-bit word."""
    code = "ld.global.u32 $0, [$1];"
    return _emit(code, "=r,l", (Int64(addr),), (Uint32,), True, loc, ip)


@dsl_user_op
def ld_u64(addr, *, loc=None, ip=None) -> Uint64:
    """Plain (weak) global load of one 64-bit word, ordered as a memory operation."""
    code = "ld.global.u64 $0, [$1];"
    return _emit(code, "=l,l," + _MEM, (Int64(addr),), (Uint64,), True, loc, ip)


@dsl_user_op
def ld_u64_generic(addr, *, loc=None, ip=None) -> Uint64:
    """``ld_u64`` through a generic address (e.g. a host-mapped word)."""
    code = "ld.u64 $0, [$1];"
    return _emit(code, "=l,l," + _MEM, (Int64(addr),), (Uint64,), True, loc, ip)


@dsl_user_op
def ld_cg_u32(addr, *, loc=None, ip=None) -> Uint32:
    code = "ld.global.cg.u32 $0, [$1];"
    return _emit(code, "=r,l", (Int64(addr),), (Uint32,), True, loc, ip)


@dsl_user_op
def ld_cg_s32(addr, *, loc=None, ip=None) -> Int32:
    code = "ld.global.cg.s32 $0, [$1];"
    return _emit(code, "=r,l", (Int64(addr),), (Int32,), True, loc, ip)


@dsl_user_op
def ld_cg_v4_u32(addr, *, loc=None, ip=None):
    code = "ld.global.cg.v4.u32 {$0,$1,$2,$3}, [$4];"
    return _emit(code, "=r,=r,=r,=r,l", (Int64(addr),), (Uint32,) * 4, True, loc, ip)


@dsl_user_op
def ld_cg_v4_f32(addr, *, loc=None, ip=None):
    code = "ld.global.cg.v4.f32 {$0,$1,$2,$3}, [$4];"
    return _emit(code, "=f,=f,=f,=f,l", (Int64(addr),), (Float32,) * 4, True, loc, ip)


@dsl_user_op
def ld_cg_v2_f32(addr, *, loc=None, ip=None):
    code = "ld.global.cg.v2.f32 {$0,$1}, [$2];"
    return _emit(code, "=f,=f,l", (Int64(addr),), (Float32,) * 2, True, loc, ip)


@dsl_user_op
def create_policy_evict_first(*, loc=None, ip=None) -> Uint64:
    code = "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;"
    return _emit(code, "=l", (), (Uint64,), True, loc, ip)


@dsl_user_op
def ld_nc_v4_f32(addr, *, loc=None, ip=None):
    """Read-only (non-coherent) load of four static fp32 words."""
    code = "ld.global.nc.v4.f32 {$0,$1,$2,$3}, [$4];"
    return _emit(code, "=f,=f,=f,=f,l", (Int64(addr),), (Float32,) * 4, True, loc, ip)


@dsl_user_op
def ld_nc_hint_v4_u32(addr, policy, *, loc=None, ip=None):
    """Static-weight load (non-coherent, no L1 allocation, L2 cache-policy hint)."""
    code = "ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {$0,$1,$2,$3}, [$4], $5;"
    return _emit(
        code,
        "=r,=r,=r,=r,l,l",
        (Int64(addr), Uint64(policy)),
        (Uint32,) * 4,
        True,
        loc,
        ip,
    )


@dsl_user_op
def ld_nc_u32(addr, *, loc=None, ip=None) -> Uint32:
    """Read-only (non-coherent) load of one static 32-bit word: only for data no grid
    writes while the kernel is resident (weights, tables written before launch)."""
    code = "ld.global.nc.u32 $0, [$1];"
    return _emit(code, "=r,l", (Int64(addr),), (Uint32,), True, loc, ip)


@dsl_user_op
def ld_nc_v4_u32_if(addr, pred, *, loc=None, ip=None):
    """Static-weight load (non-coherent, no L1 allocation) of four 32-bit words,
    executed only where ``pred`` is non-zero; zeros otherwise."""
    code = (
        "{ .reg .pred p; setp.ne.b32 p, $5, 0; "
        "mov.b32 $0, 0; mov.b32 $1, 0; mov.b32 $2, 0; mov.b32 $3, 0; "
        "@p ld.global.nc.L1::no_allocate.v4.u32 {$0,$1,$2,$3}, [$4]; }"
    )
    ops = (Int64(addr), Int32(Boolean(pred)))
    return _emit(code, "=r,=r,=r,=r,l,r", ops, (Uint32,) * 4, True, loc, ip)


# -------------------------------------------------------------------- stores
@dsl_user_op
def st_global_u32(addr, value, *, loc=None, ip=None):
    code = "st.global.u32 [$0], $1;"
    _emit(code, "l,r," + _MEM, (Int64(addr), Uint32(value)), (), True, loc, ip)


@dsl_user_op
def st_relaxed_gpu_u32(addr, value, *, loc=None, ip=None):
    code = "st.relaxed.gpu.global.u32 [$0], $1;"
    _emit(code, "l,r," + _MEM, (Int64(addr), Uint32(value)), (), True, loc, ip)


@dsl_user_op
def st_release_gpu_u32(addr, value, *, loc=None, ip=None):
    code = "st.release.gpu.global.u32 [$0], $1;"
    _emit(code, "l,r," + _MEM, (Int64(addr), Uint32(value)), (), True, loc, ip)


@dsl_user_op
def st_relaxed_sys_u32(addr, value, *, loc=None, ip=None):
    code = "st.relaxed.sys.global.u32 [$0], $1;"
    _emit(code, "l,r," + _MEM, (Int64(addr), Uint32(value)), (), True, loc, ip)


@dsl_user_op
def st_relaxed_sys_u64(addr, value, *, loc=None, ip=None):
    code = "st.relaxed.sys.global.u64 [$0], $1;"
    _emit(code, "l,l," + _MEM, (Int64(addr), Uint64(value)), (), True, loc, ip)


@dsl_user_op
def st_relaxed_sys_v2_b64(addr, r0, r1, *, loc=None, ip=None):
    """Two single-copy-atomic 64-bit records in one 16-byte vector store."""
    code = "st.relaxed.sys.global.v2.b64 [$0], {$1,$2};"
    ops = (Int64(addr), Uint64(r0), Uint64(r1))
    _emit(code, "l,l,l," + _MEM, ops, (), True, loc, ip)


@dsl_user_op
def st_relaxed_sys_v4_u32(addr, w0, w1, w2, w3, *, loc=None, ip=None):
    code = "st.relaxed.sys.global.v4.u32 [$0], {$1,$2,$3,$4};"
    ops = (Int64(addr), Uint32(w0), Uint32(w1), Uint32(w2), Uint32(w3))
    _emit(code, "l,r,r,r,r," + _MEM, ops, (), True, loc, ip)


# ------------------------------------------------------------------ atomics
@dsl_user_op
def atom_add_acq_rel_gpu_u32(addr, value, *, loc=None, ip=None) -> Uint32:
    code = "atom.acq_rel.gpu.global.add.u32 $0, [$1], $2;"
    ops = (Int64(addr), Uint32(value))
    return _emit(code, "=r,l,r," + _MEM, ops, (Uint32,), True, loc, ip)


@dsl_user_op
def atom_add_u32(addr, value, *, loc=None, ip=None) -> Uint32:
    """Relaxed gpu-scope ``atom.add`` on a global word; returns the old value."""
    code = "atom.global.add.u32 $0, [$1], $2;"
    ops = (Int64(addr), Uint32(value))
    return _emit(code, "=r,l,r," + _MEM, ops, (Uint32,), True, loc, ip)


@dsl_user_op
def red_add_release_gpu_u32(addr, value, *, loc=None, ip=None):
    code = "red.release.gpu.global.add.u32 [$0], $1;"
    _emit(code, "l,r," + _MEM, (Int64(addr), Uint32(value)), (), True, loc, ip)


@dsl_user_op
def atom_or_b32(addr, value, *, loc=None, ip=None) -> Uint32:
    """Relaxed gpu-scope ``atom.or`` (the sticky error word)."""
    code = "atom.global.or.b32 $0, [$1], $2;"
    ops = (Int64(addr), Uint32(value))
    return _emit(code, "=r,l,r," + _MEM, ops, (Uint32,), True, loc, ip)


@dsl_user_op
def atom_or_sys_u32(addr, value, *, loc=None, ip=None) -> Uint32:
    """System-scope relaxed ``atom.or`` on a global word; returns the old value."""
    code = "atom.global.sys.or.b32 $0, [$1], $2;"
    ops = (Int64(addr), Uint32(value))
    return _emit(code, "=r,l,r," + _MEM, ops, (Uint32,), True, loc, ip)


@dsl_user_op
def atom_or_sys_u32_generic(addr, value, *, loc=None, ip=None) -> Uint32:
    """``atom_or_sys_u32`` through a generic address (a peer's mailbox word)."""
    code = "atom.sys.or.b32 $0, [$1], $2;"
    ops = (Int64(addr), Uint32(value))
    return _emit(code, "=r,l,r," + _MEM, ops, (Uint32,), True, loc, ip)


# ------------------------------------------------------------------- fences
@dsl_user_op
def fence_acq_rel_gpu(*, loc=None, ip=None):
    _emit("fence.acq_rel.gpu;", _MEM, (), (), True, loc, ip)


@dsl_user_op
def fence_acq_rel_sys(*, loc=None, ip=None):
    _emit("fence.acq_rel.sys;", _MEM, (), (), True, loc, ip)


@dsl_user_op
def fence_sc_sys(*, loc=None, ip=None):
    _emit("fence.sc.sys;", _MEM, (), (), True, loc, ip)


@dsl_user_op
def fence_proxy_async_global(*, loc=None, ip=None):
    """Generic <-> async proxy hand-off for global memory (PTX ISA 9.7.15.4)."""
    _emit("fence.proxy.async.global;", _MEM, (), (), True, loc, ip)


# -------------------------------------------------------- async copies (sm80+)
@dsl_user_op
def cp_async_cg_16(smem_addr, gmem_addr, *, loc=None, ip=None):
    """16-byte ``cp.async.cg`` (shared address is 32-bit)."""
    code = "cp.async.cg.shared.global [$0], [$1], 16;"
    ops = (Uint32(smem_addr), Int64(gmem_addr))
    _emit(code, "r,l," + _MEM, ops, (), True, loc, ip)


@dsl_user_op
def cp_async_wait_all(*, loc=None, ip=None):
    _emit("cp.async.wait_all;", _MEM, (), (), True, loc, ip)


# ------------------------------------------------------------------ control
@dsl_user_op
def globaltimer(*, loc=None, ip=None) -> Uint64:
    return _emit("mov.u64 $0, %globaltimer;", "=l", (), (Uint64,), True, loc, ip)


@dsl_user_op
def trap(*, loc=None, ip=None):
    _emit("trap;", _MEM, (), (), True, loc, ip)


@dsl_user_op
def trap_if(cond, *, loc=None, ip=None):
    """``trap`` when ``cond`` (an integer) is non-zero, without a branch."""
    code = "{ .reg .pred p; setp.ne.u32 p, $0, 0; @p trap; }"
    _emit(code, "r," + _MEM, (Uint32(cond),), (), True, loc, ip)


@dsl_user_op
def exit_thread(*, loc=None, ip=None):
    """Terminate the executing thread (PTX ``exit``). Used for whole warps or CTAs that
    leave a kernel early: an exited thread no longer counts for later
    ``barrier.cluster`` phases (PTX ISA 9.7.15.3), unlike a predicated-off one."""
    _emit("exit;", _MEM, (), (), True, loc, ip)
