# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from TensorRT-LLM (Apache-2.0): tensorrt_llm/_torch/kernels/blackwell/gdn_verify_tcgen05.py.

from __future__ import annotations

import functools
from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda_exp
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T as MT
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims

from sgl_nvidia_mega.qwen3_5.sm103.gdn.core import gate_f32

D = 128
RING = 16
NTHR = 128
ROWB = 128
REG_ROWS = 160
REG_E = REG_ROWS * 64  # elements per key-dim half region
REG_B = REG_ROWS * ROWB
KH_ROW, H_ROW = 16, 32  # history keys, checkpoint rows
UH_E = 2 * RING * 64
TCOLS = 64
TCOLS_FOLD = 128
KL_E = 2 * RING * 64
CTAS_PER_SM = 4
QSCALE = 0.08838834764831845
# FP32 shared block (floats)
FS_BETA, FS_EGK, FS_EGQ, FS_A, FS_C, FS_NORM, FS_INV, FS_N = (
    0,
    8,
    16,
    48,
    112,
    176,
    208,
    224,
)


def _asm(ret, args, ptx, cons, side, loc, ip):
    return llvm.inline_asm(
        ret,
        [a.ir_value(loc=loc, ip=ip) for a in args],
        ptx,
        cons,
        has_side_effects=side,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _f32op(ptx, nargs):
    cons = "=f" + ",f" * nargs

    @dsl_user_op
    def op(*args, loc=None, ip=None):
        return Float32(
            _asm(MT.f32(), [Float32(a) for a in args], ptx, cons, False, loc, ip)
        )

    return op


_fma = _f32op("fma.rn.f32 $0, $1, $2, $3;", 3)
_fms = _f32op(
    "{\n\t.reg .f32 t;\n\tneg.f32 t, $1;\n\tfma.rn.f32 $0, t, $2, $3;\n\t}", 3
)  # c - a b
_mul = _f32op("mul.rn.f32 $0, $1, $2;", 2)
_add = _f32op("add.rn.f32 $0, $1, $2;", 2)
_sub = _f32op("sub.rn.f32 $0, $1, $2;", 2)
_expf = _f32op(
    "{\n\t.reg .f32 t;\n\tmul.rn.f32 t, $1, 0f3FB8AA3B;\n\tex2.approx.f32 $0, t;\n\t}",
    1,
)
_lg2 = _f32op("lg2.approx.f32 $0, $1;", 1)
_fdiv = _f32op("div.approx.f32 $0, $1, $2;", 2)
_div_rn = _f32op("div.rn.f32 $0, $1, $2;", 2)
_rcp_ftz = _f32op("rcp.approx.ftz.f32 $0, $1;", 1)
_sqrt_ftz = _f32op("sqrt.approx.ftz.f32 $0, $1;", 1)
_rsqrt_ftz = _f32op("rsqrt.approx.ftz.f32 $0, $1;", 1)
_bf16r = _f32op(
    "{\n\t.reg .b16 b, z;\n\tmov.b16 z, 0;\n\tcvt.rn.bf16.f32 b, $1;\n\tmov.b32 $0, {z, b};\n\t}",
    1,
)


@dsl_user_op
def _ld_s32(addr, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            MT.i32(),
            [Int64(addr)],
            "ld.global.s32 $0, [$1];",
            "=r,l,~{memory}",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def _ld_s64lo(addr, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            MT.i32(),
            [Int64(addr)],
            "{\n\t.reg .b64 t;\n\tld.global.b64 t, [$1];\n\tcvt.u32.u64 $0, t;\n\t}",
            "=r,l,~{memory}",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def _ld_s8(addr, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            MT.i32(),
            [Int64(addr)],
            "ld.global.s8 $0, [$1];",
            "=r,l,~{memory}",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def _ld_f32(addr, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            MT.f32(),
            [Int64(addr)],
            "ld.global.f32 $0, [$1];",
            "=f,l,~{memory}",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def _ld_bf16(addr, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            MT.f32(),
            [Int64(addr)],
            "{\n\t.reg .b16 h, z;\n\tmov.b16 z, 0;\n\tld.global.b16 h, [$1];\n\tmov.b32 $0, {z, h};\n\t}",
            "=f,l,~{memory}",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def _st_f32(addr, v, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [Int64(addr), Float32(v)],
        "st.global.f32 [$0], $1;",
        "l,f,~{memory}",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _st_bf16(addr, v, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [Int64(addr), Float32(v)],
        "{\n\t.reg .b16 h;\n\tcvt.rn.bf16.f32 h, $1;\n\tst.global.b16 [$0], h;\n\t}",
        "l,f,~{memory}",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _st_e4m3(addr, v, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [Int64(addr), Float32(v)],
        "{\n\t.reg .b16 h;\n\tcvt.rn.satfinite.e4m3x2.f32 h, 0f00000000, $1;\n\tst.global.b8 [$0], h;\n\t}",
        "l,f,~{memory}",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _st_zero8(addr, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [Int64(addr)],
        "{\n\t.reg .b16 z;\n\tmov.b16 z, 0;\n\tst.global.b8 [$0], z;\n\t}",
        "l,~{memory}",
        True,
        loc,
        ip,
    )


def _ldg128(addr):
    return cute.arch.inline_ptx(
        "ld.global.v4.b32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[Int32] * 4,
        read_only_args=[Int64(addr)],
    )


def _sts128(saddr, w):
    cute.arch.inline_ptx(
        "st.shared.v4.b32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
        read_only_args=[
            Int32(saddr),
            Int32(w[0]),
            Int32(w[1]),
            Int32(w[2]),
            Int32(w[3]),
        ],
    )


def _lds128f(saddr):
    return cute.arch.inline_ptx(
        "ld.shared.v4.f32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[Float32] * 4,
        read_only_args=[Int32(saddr)],
    )


def _sts128f(saddr, v):
    cute.arch.inline_ptx(
        "st.shared.v4.f32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
        read_only_args=[
            Int32(saddr),
            Float32(v[0]),
            Float32(v[1]),
            Float32(v[2]),
            Float32(v[3]),
        ],
    )


def _lds_f32(saddr):
    return cute.arch.inline_ptx(
        "ld.shared.f32 {$w0}, [{$r0}];",
        write_only_types=[Float32],
        read_only_args=[Int32(saddr)],
    )


def _sts_f32(saddr, v):
    cute.arch.inline_ptx(
        "st.shared.f32 [{$r0}], {$r1};", read_only_args=[Int32(saddr), Float32(v)]
    )


def _sts_bf16(saddr, v):
    cute.arch.inline_ptx(
        "{\n\t.reg .b16 h;\n\tcvt.rn.bf16.f32 h, {$r1};\n\tst.shared.b16 [{$r0}], h;\n\t}",
        read_only_args=[Int32(saddr), Float32(v)],
    )


def _lds_bf16(saddr):
    return cute.arch.inline_ptx(
        "{\n\t.reg .b16 h, z;\n\tmov.b16 z, 0;\n\tld.shared.b16 h, [{$r0}];\n\tmov.b32 {$w0}, {z, h};\n\t}",
        write_only_types=[Float32],
        read_only_args=[Int32(saddr)],
    )


def _lds128(saddr):
    return cute.arch.inline_ptx(
        "ld.shared.v4.b32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[Int32] * 4,
        read_only_args=[Int32(saddr)],
    )


@dsl_user_op
def _bf_lo(w, *, loc=None, ip=None) -> Float32:
    """The FP32 value of the low BF16 of a packed pair."""
    return Float32(
        _asm(
            MT.f32(),
            [Int32(w)],
            "{\n\t.reg .b32 t;\n\tshl.b32 t, $1, 16;\n\tmov.b32 $0, t;\n\t}",
            "=f,r",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _bf_hi(w, *, loc=None, ip=None) -> Float32:
    """The FP32 value of the high BF16 of a packed pair."""
    return Float32(
        _asm(
            MT.f32(),
            [Int32(w)],
            "{\n\t.reg .b32 t;\n\tand.b32 t, $1, 0xffff0000;\n\tmov.b32 $0, t;\n\t}",
            "=f,r",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _bf_pack(lo, hi, *, loc=None, ip=None) -> Int32:
    """cvt.rn of (lo, hi) to a packed BF16 pair (lo in the low half)."""
    return Int32(
        _asm(
            MT.i32(),
            [Float32(lo), Float32(hi)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            False,
            loc,
            ip,
        )
    )


def _sf(c, a, b):
    return Float32(cutlass.select_(c, Float32(a), Float32(b)))


def _si(c, a, b):
    return Int32(cutlass.select_(c, Int32(a), Int32(b)))


def _saddr(arr):
    return Int32(llvm.ptrtoint(Int32.mlir_type, arr.ir_value()))


def _shfl(v, src):
    return cute.arch.shuffle_sync(v, src)


def _bfly(v, m):
    return cute.arch.shuffle_sync_bfly(v, m)


@cute.kernel
def _verify_tc_kernel(
    tm_h: cutlass.GridConstant[cuda_exp.TensorMap],
    tm_u: cutlass.GridConstant[cuda_exp.TensorMap],
    tm_k: cutlass.GridConstant[cuda_exp.TensorMap],
    tm_kl: cutlass.GridConstant[cuda_exp.TensorMap],
    tm_dl: cutlass.GridConstant[cuda_exp.TensorMap],
    pX: Int64,
    pZ: Int64,
    pA: Int64,
    pB: Int64,
    pAl: Int64,
    pDt: Int64,
    pNw: Int64,
    pScale: Int64,
    pSi: Int64,
    pRi: Int64,
    pWp: Int64,
    pBase: Int64,
    pFl: Int64,
    pRd: Int64,
    pRk: Int64,
    pRg: Int64,
    pRdl: Int64,
    pRkl: Int64,
    pY: Int64,
    pO: Int64,
    pSt: Int64,
    n_pairs: Int32,
    n_ctas: Int32,
    xs: Int32,
    zs: Int32,
    abst: Int32,
    d_slot: Int64,
    k_slot: Int64,
    g_slot: Int64,
    dl_slot: Int64,
    kl_slot: Int64,
    st_slot: Int64,
    eps: Float32,
    T_: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    HK: cutlass.Constexpr[int],
    NORM_F32: cutlass.Constexpr[bool],
    SI64: cutlass.Constexpr[bool],
    RI64: cutlass.Constexpr[bool],
    HAS_LO: cutlass.Constexpr[bool],
    EARLY: cutlass.Constexpr[bool],
    STORE_O: cutlass.Constexpr[bool],
    FOLD: cutlass.Constexpr[bool],
):
    GROUP: cutlass.Constexpr[int] = HV // HK
    BF: cutlass.Constexpr = cutlass.BFloat16
    SW: cutlass.Constexpr = prims.Tcgen05SmemSwizzle.SWIZZLE_128B
    TC: cutlass.Constexpr[int] = TCOLS_FOLD if FOLD else TCOLS

    reg = cutlass.Array(BF, 2 * REG_E, space=cutlass.AddressSpace.smem, alignment=1024)
    uhs = cutlass.Array(BF, UH_E, space=cutlass.AddressSpace.smem, alignment=1024)
    ulo = cutlass.Array(
        BF, UH_E if FOLD else 64, space=cutlass.AddressSpace.smem, alignment=1024
    )
    klo = cutlass.Array(
        BF, KL_E if FOLD else 64, space=cutlass.AddressSpace.smem, alignment=1024
    )
    cfs = cutlass.Array(BF, 256, space=cutlass.AddressSpace.smem, alignment=128)
    fs = cutlass.Array(Float32, FS_N, space=cutlass.AddressSpace.smem, alignment=16)
    bars = cutlass.Array(Int64, 4, space=cutlass.AddressSpace.smem, alignment=8)
    tmem_i32 = cutlass.Array(Int32, 1, space=cutlass.AddressSpace.smem, alignment=4)

    tid, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    warp = cute.arch.warp_idx()
    lane = tid % Int32(32)
    l16 = lane & Int32(15)
    sR = _saddr(reg)
    sU = _saddr(uhs)
    sUl = _saddr(ulo)
    sKl = _saddr(klo)
    sCf = _saddr(cfs)
    sF = _saddr(fs)
    if cutlass.const_expr(NORM_F32):
        nw = _ld_f32(pNw + Int64(tid) * Int64(4))
    else:
        nw = _ld_bf16(pNw + Int64(tid) * Int64(2))
    rscale = _div_rn(Float32(1.0), _ld_f32(pScale))

    if warp == Int32(0):
        if prims.elect_sync():
            for i in cutlass.range_constexpr(4):
                prims.mbarrier_init(bars.subview(i), 1)
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)
    if warp == Int32(0):
        prims.tcgen05_alloc(tmem_i32, TC, group="cta_1")
        prims.tcgen05_relinquish_alloc_permit(group="cta_1")
    prims.barrier_cta_sync(0)
    tmem_raw = tmem_i32.load()
    base_t = prims.TmemAddr(tmem_raw)
    npc = (
        n_pairs - bidx + n_ctas - Int32(1)
    ) // n_ctas  # pairs of this CTA: bidx, bidx + n_ctas, ...

    pnx = bidx
    nx = pnx // Int32(HV)
    hvn = pnx - nx * Int32(HV)
    if cutlass.const_expr(SI64):
        slot_n = _ld_s64lo(pSi + Int64(nx) * Int64(8))
    else:
        slot_n = _ld_s32(pSi + Int64(nx) * Int64(4))
    if cutlass.const_expr(RI64):
        rid_n = _ld_s64lo(pRi + Int64(nx) * Int64(8))
    else:
        rid_n = _ld_s32(pRi + Int64(nx) * Int64(4))
    rid_n = _si(slot_n >= Int32(0), cutlass.max(rid_n, Int32(0)), Int32(0))
    wp_n = _ld_s32(pWp + Int64(rid_n) * Int64(4))
    bs_n = _ld_s32(pBase + Int64(rid_n) * Int64(4))
    fl_n = _ld_s8(pFl + Int64(rid_n))
    if cutlass.const_expr(FOLD):
        live_n = _si(slot_n >= Int32(0), 1, 0)
        fo_n = _si((slot_n >= Int32(0)) & (fl_n != Int32(0)), 1, 0)
    else:
        live_n = _si((slot_n >= Int32(0)) & (fl_n == Int32(0)), 1, 0)
        fo_n = Int32(0)
    gj_n = _ld_f32(
        pRg
        + (
            Int64(rid_n) * g_slot
            + Int64(hvn * Int32(RING) + ((bs_n + l16) & Int32(RING - 1)))
        )
        * Int64(4)
    )
    al_n = _ld_f32(pAl + Int64(hvn) * Int64(4))
    dt_n = _ld_f32(pDt + Int64(hvn) * Int64(4))
    has_nx = npc > Int32(0)
    if has_nx & (live_n != Int32(0)):
        hkx0 = hvn // Int32(GROUP)
        # the ring halves (8 positions) holding committed entries [base, base + wp)
        mk0 = (Int32(1) << wp_n) - Int32(1)
        rot0 = ((mk0 << bs_n) | (mk0 >> (Int32(RING) - bs_n))) & Int32(0xFFFF)
        nd0 = _si((rot0 & Int32(0xFF)) != Int32(0), 1, 0)
        nd1 = _si((rot0 & Int32(0xFF00)) != Int32(0), 1, 0)
        if warp == Int32(0):
            if prims.elect_sync():
                prims.mbarrier_arrive_expect_tx(
                    bars.subview(0),
                    Int32(2 * 16384) + (nd0 + nd1) * (Int32(4096) + fo_n * Int32(4096)),
                )
                for x in cutlass.range_constexpr(2):
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        reg.subview(x * REG_E + H_ROW * 64),
                        tm_h.get_ptr(),
                        (Int32(64 * x), Int32(0), hvn, slot_n),
                        bars.subview(0),
                    )
                for jb in cutlass.range_constexpr(2):
                    if (nd0 if jb == 0 else nd1) != Int32(0):
                        for x in cutlass.range_constexpr(2):
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                uhs.subview(x * RING * 64 + jb * 8 * 64),
                                tm_u.get_ptr(),
                                (Int32(64 * x), Int32(8 * jb), hvn, rid_n),
                                bars.subview(0),
                            )
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                reg.subview(x * REG_E + (KH_ROW + 8 * jb) * 64),
                                tm_k.get_ptr(),
                                (Int32(64 * x), Int32(8 * jb), hkx0, rid_n),
                                bars.subview(0),
                            )
                            if cutlass.const_expr(FOLD):
                                if fo_n != Int32(0):
                                    prims.cp_async_bulk_tensor_shared_cta_global(
                                        klo.subview(x * RING * 64 + jb * 8 * 64),
                                        tm_kl.get_ptr(),
                                        (Int32(64 * x), Int32(8 * jb), hkx0, rid_n),
                                        bars.subview(0),
                                    )
                                    prims.cp_async_bulk_tensor_shared_cta_global(
                                        ulo.subview(x * RING * 64 + jb * 8 * 64),
                                        tm_dl.get_ptr(),
                                        (Int32(64 * x), Int32(8 * jb), hvn, rid_n),
                                        bars.subview(0),
                                    )
    cute.arch.griddepcontrol_wait()

    j = Int32(0)  # TMA'd pairs processed (barrier phase)
    for kk in cutlass.range(npc):
        if cutlass.const_expr(EARLY):
            if kk == npc - Int32(1):
                # the next launch may start once every CTA has started its last pair
                cute.arch.griddepcontrol_launch_dependents()
        pid = bidx + kk * n_ctas
        slot = slot_n
        rid = rid_n
        wp = wp_n
        bs = bs_n
        live = live_n
        fo = fo_n
        gj = gj_n
        a_log = al_n
        dtb = dt_n
        has_nx = (kk + Int32(1)) < npc
        # the next pair's scalars
        pnx = _si(has_nx, pid + n_ctas, pid)
        nx = pnx // Int32(HV)
        hvn = pnx - nx * Int32(HV)
        if cutlass.const_expr(SI64):
            slot_n = _ld_s64lo(pSi + Int64(nx) * Int64(8))
        else:
            slot_n = _ld_s32(pSi + Int64(nx) * Int64(4))
        if cutlass.const_expr(RI64):
            rid_n = _ld_s64lo(pRi + Int64(nx) * Int64(8))
        else:
            rid_n = _ld_s32(pRi + Int64(nx) * Int64(4))
        rid_n = _si(slot_n >= Int32(0), cutlass.max(rid_n, Int32(0)), Int32(0))
        wp_n = _ld_s32(pWp + Int64(rid_n) * Int64(4))
        bs_n = _ld_s32(pBase + Int64(rid_n) * Int64(4))
        fl_n = _ld_s8(pFl + Int64(rid_n))
        if cutlass.const_expr(FOLD):
            live_n = _si(slot_n >= Int32(0), 1, 0)
            fo_n = _si((slot_n >= Int32(0)) & (fl_n != Int32(0)), 1, 0)
        else:
            live_n = _si((slot_n >= Int32(0)) & (fl_n == Int32(0)), 1, 0)
        gj_n = _ld_f32(
            pRg
            + (
                Int64(rid_n) * g_slot
                + Int64(hvn * Int32(RING) + ((bs_n + l16) & Int32(RING - 1)))
            )
            * Int64(4)
        )
        al_n = _ld_f32(pAl + Int64(hvn) * Int64(4))
        dt_n = _ld_f32(pDt + Int64(hvn) * Int64(4))
        hkn = hvn // Int32(GROUP)
        mkn = (Int32(1) << wp_n) - Int32(1)
        rotn = ((mkn << bs_n) | (mkn >> (Int32(RING) - bs_n))) & Int32(0xFFFF)
        ndn0 = _si((rotn & Int32(0xFF)) != Int32(0), 1, 0)
        ndn1 = _si((rotn & Int32(0xFF00)) != Int32(0), 1, 0)
        nx_go = has_nx & (live_n != Int32(0))
        txn = Int32(2 * 16384) + (ndn0 + ndn1) * (Int32(4096) + fo_n * Int32(4096))

        n = pid // Int32(HV)
        hv = pid - n * Int32(HV)
        hk = hv // Int32(GROUP)
        row0 = n * Int32(T_)
        ph = j & Int32(1)
        if live == Int32(0):
            for t in cutlass.range_constexpr(T_):
                _st_zero8(
                    pY
                    + Int64(row0 + Int32(t)) * Int64(HV * D)
                    + Int64(hv * Int32(D) + tid)
                )
                if cutlass.const_expr(STORE_O):
                    _st_bf16(
                        pO
                        + (
                            Int64(row0 + Int32(t)) * Int64(HV * D)
                            + Int64(hv * Int32(D) + tid)
                        )
                        * Int64(2),
                        Float32(0.0),
                    )
            if nx_go:
                if warp == Int32(0):
                    if cutlass.const_expr(FOLD):
                        # an earlier pair's folded checkpoint store must have read its rows
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                        cute.arch.sync_warp()
                    if prims.elect_sync():
                        prims.mbarrier_arrive_expect_tx(bars.subview(0), txn)
                        for x in cutlass.range_constexpr(2):
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                reg.subview(x * REG_E + H_ROW * 64),
                                tm_h.get_ptr(),
                                (Int32(64 * x), Int32(0), hvn, slot_n),
                                bars.subview(0),
                            )
                        for jb in cutlass.range_constexpr(2):
                            if (ndn0 if jb == 0 else ndn1) != Int32(0):
                                for x in cutlass.range_constexpr(2):
                                    prims.cp_async_bulk_tensor_shared_cta_global(
                                        uhs.subview(x * RING * 64 + jb * 8 * 64),
                                        tm_u.get_ptr(),
                                        (Int32(64 * x), Int32(8 * jb), hvn, rid_n),
                                        bars.subview(0),
                                    )
                                    prims.cp_async_bulk_tensor_shared_cta_global(
                                        reg.subview(x * REG_E + (KH_ROW + 8 * jb) * 64),
                                        tm_k.get_ptr(),
                                        (Int32(64 * x), Int32(8 * jb), hkn, rid_n),
                                        bars.subview(0),
                                    )
                                    if cutlass.const_expr(FOLD):
                                        if fo_n != Int32(0):
                                            prims.cp_async_bulk_tensor_shared_cta_global(
                                                klo.subview(
                                                    x * RING * 64 + jb * 8 * 64
                                                ),
                                                tm_kl.get_ptr(),
                                                (
                                                    Int32(64 * x),
                                                    Int32(8 * jb),
                                                    hkn,
                                                    rid_n,
                                                ),
                                                bars.subview(0),
                                            )
                                            prims.cp_async_bulk_tensor_shared_cta_global(
                                                ulo.subview(
                                                    x * RING * 64 + jb * 8 * 64
                                                ),
                                                tm_dl.get_ptr(),
                                                (
                                                    Int32(64 * x),
                                                    Int32(8 * jb),
                                                    hvn,
                                                    rid_n,
                                                ),
                                                bars.subview(0),
                                            )
        else:
            gj = _sf(l16 < wp, gj, 0.0)
            for d in cutlass.range_constexpr(4):
                o = cute.arch.shuffle_sync_up(gj, 1 << d)
                gj = _sf(l16 >= Int32(1 << d), _add(gj, o), gj)
            g_start = _sf(
                wp > Int32(0), _shfl(gj, cutlass.max(wp - Int32(1), Int32(0))), 0.0
            )
            gw = Float32(0.0)  # G_w: the folded entries' total log-decay
            wf = Int32(
                0
            )  # the committed entries this pair folds (a flush row's, in a fold build)
            wpo = wp  # the committed entries the output replays (none once folded)
            if cutlass.const_expr(FOLD):
                gw = _sf(fo != Int32(0), g_start, 0.0)
                wf = _si(fo != Int32(0), wp, 0)
                g_start = _sf(fo != Int32(0), 0.0, g_start)
                wpo = _si(fo != Int32(0), 0, wp)
            neg_a = Float32(0.0) - _expf(a_log)
            r = tid >> Int32(3)
            c8 = tid & Int32(7)
            is_k = r < Int32(T_)
            is_q = (r >= Int32(8)) & (r < Int32(8 + T_))
            tok = _si(is_k, r, _si(is_q, r - Int32(8), 0))
            col = _si(is_k, Int32(HK * D) + hk * Int32(D), hk * Int32(D))
            for x in cutlass.range_constexpr(2):
                w = _ldg128(
                    pX
                    + (
                        Int64(row0 + tok) * Int64(xs)
                        + Int64(col + Int32(64 * x) + c8 * Int32(8))
                    )
                    * Int64(2)
                )
                keep = is_k | is_q
                wz = [_si(keep, w[i], 0) for i in range(4)]
                _sts128(
                    sR
                    + Int32(x * REG_B)
                    + r * Int32(ROWB)
                    + ((c8 ^ (r & Int32(7))) * Int32(16)),
                    wz,
                )
            vv = []
            zz = []
            for t in cutlass.range_constexpr(T_):
                vv.append(
                    _ld_bf16(
                        pX
                        + (
                            Int64(row0 + Int32(t)) * Int64(xs)
                            + Int64(Int32(2 * HK * D) + hv * Int32(D) + tid)
                        )
                        * Int64(2)
                    )
                )
                zz.append(
                    _ld_bf16(
                        pZ
                        + (
                            Int64(row0 + Int32(t)) * Int64(zs)
                            + Int64(hv * Int32(D) + tid)
                        )
                        * Int64(2)
                    )
                )
            lt = lane < Int32(T_)
            lc = _si(lt, lane, 0)
            a_t = _ld_bf16(pA + (Int64(row0 + lc) * Int64(abst) + Int64(hv)) * Int64(2))
            b_t = _ld_bf16(pB + (Int64(row0 + lc) * Int64(abst) + Int64(hv)) * Int64(2))
            xg = _add(a_t, dtb)
            spl = _mul(Float32(0.6931471805599453), _lg2(_add(Float32(1.0), _expf(xg))))
            spl = _sf(xg <= Float32(20.0), spl, xg)
            gstep = _sf(lt, _mul(neg_a, spl), 0.0)
            beta = _sf(
                lt,
                _fdiv(Float32(1.0), _add(Float32(1.0), _expf(Float32(0.0) - b_t))),
                0.0,
            )
            gl = gstep
            for d in cutlass.range_constexpr(3):
                o = cute.arch.shuffle_sync_up(gl, 1 << d)
                gl = _sf(lane >= Int32(1 << d), _add(gl, o), gl)
            gv = _add(g_start, gl)
            tcf = tid >> Int32(4)
            jcf = tid & Int32(15)
            jent = (jcf - bs + Int32(RING)) & Int32(RING - 1)
            ec = _expf(_sub(_shfl(gv, tcf), _shfl(gj, jent)))
            ia = tid & Int32(63)
            ta = ia >> Int32(3)
            sa = ia & Int32(7)
            is_a = tid >= Int32(64)
            exa = _expf(_sub(_shfl(gv, ta), _shfl(gv, sa)))
            bta = _shfl(beta, ta)

            while not prims.mbarrier_try_wait_parity(
                bars.subview(0), ph, time_limit=10_000_000
            ):
                pass
            jr = tid >> Int32(3)
            if ((jr - bs + Int32(RING)) & Int32(RING - 1)) >= wp:
                zero4 = [Int32(0)] * 4
                for x in cutlass.range_constexpr(2):
                    _sts128(
                        sU + Int32(x * RING * ROWB) + jr * Int32(ROWB) + c8 * Int32(16),
                        zero4,
                    )
                    _sts128(
                        sR
                        + Int32(x * REG_B + KH_ROW * ROWB)
                        + jr * Int32(ROWB)
                        + c8 * Int32(16),
                        zero4,
                    )
            if cutlass.const_expr(FOLD):
                if ((jr - bs + Int32(RING)) & Int32(RING - 1)) >= wf:
                    zk4 = [Int32(0)] * 4
                    for fx in cutlass.range_constexpr(2):
                        _sts128(
                            sKl
                            + Int32(fx * RING * ROWB)
                            + jr * Int32(ROWB)
                            + c8 * Int32(16),
                            zk4,
                        )
                fvx = tid >> Int32(6)
                fvc = tid & Int32(63)
                for fp in cutlass.range_constexpr(RING):
                    fje = (Int32(fp) - bs + Int32(RING)) & Int32(RING - 1)
                    fdec = _expf(_sub(gw, _shfl(gj, fje)))
                    fuo = (
                        fvx * Int32(RING * ROWB)
                        + Int32(fp * ROWB)
                        + (((fvc >> Int32(3)) ^ Int32(fp & 7)) << Int32(4))
                        + (fvc & Int32(7)) * Int32(2)
                    )
                    fdh = _lds_bf16(sU + fuo)
                    fdl = _lds_bf16(
                        sUl + fuo
                    )  # the update low parts, TMA'd with the high parts
                    fu = _mul(_add(fdh, fdl), fdec)
                    fuh = _bf16r(fu)
                    if fje < wf:
                        _sts_bf16(sU + fuo, fuh)
                    _sts_bf16(sUl + fuo, _sf(fje < wf, _sub(fu, fuh), 0.0))
            prims.fence_proxy(
                prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta
            )
            prims.barrier_cta_sync(0)

            if cutlass.const_expr(FOLD):
                if warp == Int32(0):
                    idf = prims.Tcgen05InstrDesc.build(
                        c_dtype=Float32,
                        a_dtype=BF,
                        b_dtype=BF,
                        n_dim=64,
                        m_dim=128,
                        a_major=1,
                        b_major=1,
                    )
                    if prims.elect_sync():
                        if wf > Int32(0):
                            for fx in cutlass.range_constexpr(2):
                                for kb in cutlass.range_constexpr(2):
                                    for ua in cutlass.range_constexpr(2):
                                        fda = prims.Tcgen05SmemDesc.build(
                                            start_address=uhs if ua == 0 else ulo,
                                            leading_byte_offset=2048,
                                            stride_byte_offset=1024,
                                            layout=SW,
                                        )
                                        fdb = prims.Tcgen05SmemDesc.build(
                                            start_address=reg.subview(
                                                fx * REG_E + KH_ROW * 64
                                            )
                                            if kb == 0
                                            else klo.subview(fx * RING * 64),
                                            leading_byte_offset=2048,
                                            stride_byte_offset=1024,
                                            layout=SW,
                                        )
                                        prims.tcgen05_mma(
                                            prims.Tcgen05MMAKind.F16,
                                            prims.CTAGroup.CTA_1,
                                            prims.TmemAddr.from_row_col(
                                                base_t.row_id,
                                                base_t.col_id + Int32(64 * fx),
                                            ).as_ptr(Float32),
                                            fda,
                                            fdb,
                                            idf,
                                            (kb > 0) or (ua > 0),
                                        )
                        prims.tcgen05_commit(
                            bars.subview(3), group=prims.CTAGroup.CTA_1
                        )
                while not prims.mbarrier_try_wait_parity(
                    bars.subview(3), ph, time_limit=10_000_000
                ):
                    pass
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                fegw = _expf(gw)
                frow = base_t.row_id + warp * Int32(32)
                for fx in cutlass.range_constexpr(2):
                    for fc in cutlass.range_constexpr(4):
                        facc = cutlass.Array(Float32, 16, alignment=32)
                        facc.store(
                            prims.tcgen05_ld(
                                "32x32b",
                                prims.TmemAddr.from_row_col(
                                    frow, base_t.col_id + Int32(64 * fx + 16 * fc)
                                ).as_ptr(Float32),
                                num=16,
                            ),
                            0,
                        )
                        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                        for fh in cutlass.range_constexpr(2):
                            fsa = (
                                sR
                                + Int32(fx * REG_B)
                                + (Int32(H_ROW) + tid) * Int32(ROWB)
                                + ((Int32(2 * fc + fh) ^ (tid & Int32(7))) << Int32(4))
                            )
                            fw = _lds128(fsa)
                            fn4 = [
                                _bf_pack(
                                    _fma(
                                        _bf_lo(fw[fe]),
                                        fegw,
                                        _sf(wf > Int32(0), facc[8 * fh + 2 * fe], 0.0),
                                    ),
                                    _fma(
                                        _bf_hi(fw[fe]),
                                        fegw,
                                        _sf(
                                            wf > Int32(0),
                                            facc[8 * fh + 2 * fe + 1],
                                            0.0,
                                        ),
                                    ),
                                )
                                for fe in range(4)
                            ]
                            _sts128(fsa, fn4)
                prims.fence_proxy(
                    prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta
                )
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                prims.barrier_cta_sync(0)
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                if wf > Int32(0):
                    if warp == Int32(0):
                        if prims.elect_sync():
                            for fx in cutlass.range_constexpr(2):
                                prims.cp_async_bulk_tensor_global_shared_cta(
                                    tm_h.get_ptr(),
                                    reg.subview(fx * REG_E + H_ROW * 64),
                                    [Int32(64 * fx), Int32(0), hv, slot],
                                )
                            cute.arch.cp_async_bulk_commit_group()

            if warp == Int32(0):
                idesc16 = prims.Tcgen05InstrDesc.build(
                    c_dtype=Float32, a_dtype=BF, b_dtype=BF, n_dim=16, m_dim=128
                )
                if prims.elect_sync():
                    for x in cutlass.range_constexpr(2):
                        dq = prims.Tcgen05SmemDesc.build(
                            start_address=reg.subview(x * REG_E),
                            leading_byte_offset=16,
                            stride_byte_offset=1024,
                            layout=SW,
                        )
                        dh = prims.Tcgen05SmemDesc.build(
                            start_address=reg.subview(x * REG_E + H_ROW * 64),
                            leading_byte_offset=16,
                            stride_byte_offset=1024,
                            layout=SW,
                        )
                        for kq in cutlass.range_constexpr(4):
                            prims.tcgen05_mma(
                                prims.Tcgen05MMAKind.F16,
                                prims.CTAGroup.CTA_1,
                                prims.TmemAddr.from_row_col(
                                    base_t.row_id, base_t.col_id + Int32(16)
                                ).as_ptr(Float32),
                                dq.advance_start_address(32 * kq),
                                dq.advance_start_address(32 * kq),
                                idesc16,
                                (x > 0) or (kq > 0),
                            )
                            prims.tcgen05_mma(
                                prims.Tcgen05MMAKind.F16,
                                prims.CTAGroup.CTA_1,
                                prims.TmemAddr.from_row_col(
                                    base_t.row_id, base_t.col_id
                                ).as_ptr(Float32),
                                dh.advance_start_address(32 * kq),
                                dq.advance_start_address(32 * kq),
                                idesc16,
                                (x > 0) or (kq > 0),
                            )
                    prims.tcgen05_commit(bars.subview(1), group=prims.CTAGroup.CTA_1)

            gram0 = sR + Int32(8 * ROWB)
            gram1 = sR + Int32(REG_B + 8 * ROWB)
            if warp == Int32(0):
                while not prims.mbarrier_try_wait_parity(
                    bars.subview(1), ph, time_limit=10_000_000
                ):
                    pass
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                if nx_go:
                    if cutlass.const_expr(FOLD):
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                        cute.arch.sync_warp()
                    if prims.elect_sync():
                        prims.mbarrier_arrive_expect_tx(bars.subview(0), txn)
                        for x in cutlass.range_constexpr(2):
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                reg.subview(x * REG_E + H_ROW * 64),
                                tm_h.get_ptr(),
                                (Int32(64 * x), Int32(0), hvn, slot_n),
                                bars.subview(0),
                            )
                        for jb in cutlass.range_constexpr(2):
                            if (ndn0 if jb == 0 else ndn1) != Int32(0):
                                for x in cutlass.range_constexpr(2):
                                    prims.cp_async_bulk_tensor_shared_cta_global(
                                        reg.subview(x * REG_E + (KH_ROW + 8 * jb) * 64),
                                        tm_k.get_ptr(),
                                        (Int32(64 * x), Int32(8 * jb), hkn, rid_n),
                                        bars.subview(0),
                                    )
                                    if cutlass.const_expr(FOLD):
                                        if fo_n != Int32(0):
                                            prims.cp_async_bulk_tensor_shared_cta_global(
                                                klo.subview(
                                                    x * RING * 64 + jb * 8 * 64
                                                ),
                                                tm_kl.get_ptr(),
                                                (
                                                    Int32(64 * x),
                                                    Int32(8 * jb),
                                                    hkn,
                                                    rid_n,
                                                ),
                                                bars.subview(0),
                                            )
                gacc = cutlass.Array(Float32, 16, alignment=32)
                gacc.store(
                    prims.tcgen05_ld(
                        "32x32b",
                        prims.TmemAddr.from_row_col(
                            base_t.row_id, base_t.col_id + Int32(16)
                        ).as_ptr(Float32),
                        num=16,
                    ),
                    0,
                )
                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                grow = _si(
                    lane < Int32(16),
                    gram0 + lane * Int32(64),
                    gram1 + (lane - Int32(16)) * Int32(64),
                )
                for q4 in cutlass.range_constexpr(4):
                    _sts128f(
                        grow + Int32(16 * q4), [gacc[4 * q4 + e] for e in range(4)]
                    )
                dg = gacc[0]
                for i in cutlass.range_constexpr(1, 16):
                    dg = _sf(lane == Int32(i), gacc[i], dg)
                inv = _rcp_ftz(_sqrt_ftz(_add(dg, Float32(1e-6))))
                inv = _sf(
                    lane >= Int32(8), _mul(Float32(QSCALE), inv), inv
                )  # rows 8..: scale / |q_t|
                if lane < Int32(16):
                    _sts_f32(sF + (Int32(FS_INV) + lane) * Int32(4), inv)
                iql = _shfl(inv, (lane + Int32(8)) & Int32(31))
                if lane < Int32(8):
                    eg = _expf(gv)
                    _sts_f32(
                        sF + (Int32(FS_BETA) + lane) * Int32(4), _sf(lt, beta, 0.0)
                    )
                    _sts_f32(
                        sF + (Int32(FS_EGK) + lane) * Int32(4),
                        _sf(lt, _mul(eg, inv), 0.0),
                    )
                    _sts_f32(
                        sF + (Int32(FS_EGQ) + lane) * Int32(4),
                        _sf(lt, _mul(eg, iql), 0.0),
                    )
                if lt & ((wp + lane) < Int32(RING)):
                    # this step's per-token log-decays into the ring
                    _st_f32(
                        pRg
                        + (
                            Int64(rid) * g_slot
                            + Int64(
                                hv * Int32(RING) + ((bs + wp + lane) & Int32(RING - 1))
                            )
                        )
                        * Int64(4),
                        gstep,
                    )
            prims.barrier_cta_sync(0)

            hjc = (jent < wpo) & (tcf < Int32(T_))
            ck = _sf(
                hjc,
                _mul(
                    _mul(_lds_f32(gram1 + jcf * Int32(64) + tcf * Int32(4)), ec),
                    _lds_f32(sF + (Int32(FS_INV) + tcf) * Int32(4)),
                ),
                0.0,
            )
            cq = _sf(
                hjc,
                _mul(
                    _mul(
                        _lds_f32(gram1 + jcf * Int32(64) + (tcf + Int32(8)) * Int32(4)),
                        ec,
                    ),
                    _lds_f32(sF + (Int32(FS_INV + 8) + tcf) * Int32(4)),
                ),
                0.0,
            )
            cbc = (
                sCf
                + (jcf >> Int32(3)) * Int32(128)
                + (jcf & Int32(7)) * Int32(2)
                + tcf * Int32(16)
            )
            _sts_bf16(cbc, ck)
            _sts_bf16(cbc + Int32(256), cq)
            iks = _lds_f32(sF + (Int32(FS_INV) + sa) * Int32(4))
            aval = _mul(
                _mul(
                    bta,
                    _mul(
                        _mul(
                            _lds_f32(gram0 + ta * Int32(64) + sa * Int32(4)),
                            _lds_f32(sF + (Int32(FS_INV) + ta) * Int32(4)),
                        ),
                        iks,
                    ),
                ),
                exa,
            )
            cval = _mul(
                _mul(
                    _mul(
                        _lds_f32(gram0 + (ta + Int32(8)) * Int32(64) + sa * Int32(4)),
                        exa,
                    ),
                    iks,
                ),
                _lds_f32(sF + (Int32(FS_INV + 8) + ta) * Int32(4)),
            )
            acv = _sf(
                is_a,
                _sf((sa < ta) & (ta < Int32(T_)), aval, 0.0),
                _sf((sa <= ta) & (ta < Int32(T_)), cval, 0.0),
            )
            _sts_f32(sF + (_si(is_a, FS_A, FS_C) + ta * Int32(8) + sa) * Int32(4), acv)
            prims.fence_proxy(
                prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta
            )
            prims.barrier_cta_sync(0)

            if warp == Int32(0):
                idesc_h = prims.Tcgen05InstrDesc.build(
                    c_dtype=Float32,
                    a_dtype=BF,
                    b_dtype=BF,
                    n_dim=16,
                    m_dim=128,
                    a_major=1,
                )
                if prims.elect_sync():
                    du = prims.Tcgen05SmemDesc.build(
                        start_address=uhs,
                        leading_byte_offset=2048,
                        stride_byte_offset=1024,
                        layout=SW,
                    )
                    dc = prims.Tcgen05SmemDesc.build(
                        start_address=cfs,
                        leading_byte_offset=128,
                        stride_byte_offset=256,
                        layout=prims.Tcgen05SmemSwizzle.NONE,
                    )
                    prims.tcgen05_mma(
                        prims.Tcgen05MMAKind.F16,
                        prims.CTAGroup.CTA_1,
                        prims.TmemAddr.from_row_col(
                            base_t.row_id, base_t.col_id + Int32(32)
                        ).as_ptr(Float32),
                        du,
                        dc,
                        idesc_h,
                        False,
                    )
                    prims.tcgen05_commit(bars.subview(2), group=prims.CTAGroup.CTA_1)

            # ---- output path: thread v = tid ----
            bt = _lds128f(sF + Int32(FS_BETA * 4)) + _lds128f(
                sF + Int32((FS_BETA + 4) * 4)
            )
            egk = _lds128f(sF + Int32(FS_EGK * 4)) + _lds128f(
                sF + Int32((FS_EGK + 4) * 4)
            )
            egq = _lds128f(sF + Int32(FS_EGQ * 4)) + _lds128f(
                sF + Int32((FS_EGQ + 4) * 4)
            )
            while not prims.mbarrier_try_wait_parity(
                bars.subview(2), ph, time_limit=10_000_000
            ):
                pass
            if nx_go:
                if warp == Int32(0):
                    if prims.elect_sync():
                        for jb in cutlass.range_constexpr(2):
                            if (ndn0 if jb == 0 else ndn1) != Int32(0):
                                for x in cutlass.range_constexpr(2):
                                    prims.cp_async_bulk_tensor_shared_cta_global(
                                        uhs.subview(x * RING * 64 + jb * 8 * 64),
                                        tm_u.get_ptr(),
                                        (Int32(64 * x), Int32(8 * jb), hvn, rid_n),
                                        bars.subview(0),
                                    )
                                    if cutlass.const_expr(FOLD):
                                        if fo_n != Int32(0):
                                            prims.cp_async_bulk_tensor_shared_cta_global(
                                                ulo.subview(
                                                    x * RING * 64 + jb * 8 * 64
                                                ),
                                                tm_dl.get_ptr(),
                                                (
                                                    Int32(64 * x),
                                                    Int32(8 * jb),
                                                    hvn,
                                                    rid_n,
                                                ),
                                                bars.subview(0),
                                            )
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            trow = base_t.row_id + warp * Int32(32)
            sacc = cutlass.Array(Float32, 16, alignment=32)
            hacc = cutlass.Array(Float32, 16, alignment=32)
            sacc.store(
                prims.tcgen05_ld(
                    "32x32b",
                    prims.TmemAddr.from_row_col(trow, base_t.col_id).as_ptr(Float32),
                    num=16,
                ),
                0,
            )
            hacc.store(
                prims.tcgen05_ld(
                    "32x32b",
                    prims.TmemAddr.from_row_col(trow, base_t.col_id + Int32(32)).as_ptr(
                        Float32
                    ),
                    num=16,
                ),
                0,
            )
            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)

            uu = []
            for i in cutlass.range_constexpr(T_):
                u = _mul(bt[i], _sub(_fms(egk[i], sacc[i], vv[i]), hacc[i]))
                if cutlass.const_expr(i > 0):
                    arow = _lds128f(sF + Int32((FS_A + 8 * i) * 4)) + _lds128f(
                        sF + Int32((FS_A + 8 * i + 4) * 4)
                    )
                    for s in cutlass.range_constexpr(i):
                        u = _fms(arow[s], uu[s], u)
                uu.append(u)
            xb = []
            for t in cutlass.range_constexpr(T_):
                qn = Float32(0.0)
                crow = _lds128f(sF + Int32((FS_C + 8 * t) * 4)) + _lds128f(
                    sF + Int32((FS_C + 8 * t + 4) * 4)
                )
                for s in cutlass.range_constexpr(t + 1):
                    qn = _fma(crow[s], uu[s], qn)
                ov = _add(_fma(egq[t], sacc[8 + t], hacc[8 + t]), qn)
                xb.append(_bf16r(ov))
                if cutlass.const_expr(STORE_O):
                    _st_bf16(
                        pO
                        + (
                            Int64(row0 + Int32(t)) * Int64(HV * D)
                            + Int64(hv * Int32(D) + tid)
                        )
                        * Int64(2),
                        xb[t],
                    )
            for t in cutlass.range_constexpr(T_):
                if (wp + Int32(t)) < Int32(RING):
                    pos = (bs + wp + Int32(t)) & Int32(RING - 1)
                    hi = _bf16r(uu[t])
                    _st_bf16(
                        pRd
                        + (
                            Int64(rid) * d_slot
                            + Int64((hv * Int32(RING) + pos) * Int32(D) + tid)
                        )
                        * Int64(2),
                        hi,
                    )
                    if cutlass.const_expr(HAS_LO):
                        _st_bf16(
                            pRdl
                            + (
                                Int64(rid) * dl_slot
                                + Int64((hv * Int32(RING) + pos) * Int32(D) + tid)
                            )
                            * Int64(2),
                            _sub(uu[t], hi),
                        )
            p = [_mul(xb[t], xb[t]) for t in range(T_)] + [Float32(0.0)] * (8 - T_)
            up16 = (lane & Int32(16)) != Int32(0)
            q4 = [
                _add(_sf(up16, p[i + 4], p[i]), _bfly(_sf(up16, p[i], p[i + 4]), 16))
                for i in range(4)
            ]
            up8 = (lane & Int32(8)) != Int32(0)
            q2 = [
                _add(_sf(up8, q4[i + 2], q4[i]), _bfly(_sf(up8, q4[i], q4[i + 2]), 8))
                for i in range(2)
            ]
            up4 = (lane & Int32(4)) != Int32(0)
            q1 = _add(_sf(up4, q2[1], q2[0]), _bfly(_sf(up4, q2[0], q2[1]), 4))
            q1 = _add(q1, _bfly(q1, 2))
            q1 = _add(q1, _bfly(q1, 1))
            ntok = (
                ((lane >> Int32(4)) & Int32(1)) * Int32(4)
                + ((lane >> Int32(3)) & Int32(1)) * Int32(2)
                + ((lane >> Int32(2)) & Int32(1))
            )
            if (lane & Int32(3)) == Int32(0):
                _sts_f32(sF + (Int32(FS_NORM) + ntok * Int32(4) + warp) * Int32(4), q1)
            prims.barrier_cta_sync(0)
            for t in cutlass.range_constexpr(T_):
                nv = _lds128f(sF + Int32((FS_NORM + 4 * t) * 4))
                ss = _add(_add(_add(nv[0], nv[1]), nv[2]), nv[3])
                rstd = _rsqrt_ftz(_fma(ss, Float32(1.0 / D), eps))
                yv = _mul(_mul(xb[t], rstd), nw)
                sg = _fdiv(
                    Float32(1.0), _add(Float32(1.0), _expf(Float32(0.0) - zz[t]))
                )
                yv = _mul(yv, _mul(zz[t], sg))
                _st_e4m3(
                    pY
                    + Int64(row0 + Int32(t)) * Int64(HV * D)
                    + Int64(hv * Int32(D) + tid),
                    _mul(_bf16r(yv), rscale),
                )
            if (hv % Int32(GROUP)) == Int32(0):
                kx = tid >> Int32(6)
                ke = tid & Int32(63)
                for t in cutlass.range_constexpr(T_):
                    if (wp + Int32(t)) < Int32(RING):
                        pos = (bs + wp + Int32(t)) & Int32(RING - 1)
                        kr = _lds_bf16(
                            sR
                            + kx * Int32(REG_B)
                            + Int32(t * ROWB)
                            + (((ke >> Int32(3)) ^ Int32(t & 7)) * Int32(16))
                            + (ke & Int32(7)) * Int32(2)
                        )
                        kn = _mul(kr, _lds_f32(sF + Int32((FS_INV + t) * 4)))
                        khi = _bf16r(kn)
                        koff = Int64((hk * Int32(RING) + pos) * Int32(D) + tid)
                        _st_bf16(pRk + (Int64(rid) * k_slot + koff) * Int64(2), khi)
                        if cutlass.const_expr(HAS_LO):
                            _st_bf16(
                                pRkl + (Int64(rid) * kl_slot + koff) * Int64(2),
                                _sub(kn, khi),
                            )
            j = j + Int32(1)
        cute.arch.sync_warp()
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        prims.barrier_cta_sync(0)
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)

    if cutlass.const_expr(EARLY):
        if npc == Int32(0):
            cute.arch.griddepcontrol_launch_dependents()
    else:
        cute.arch.griddepcontrol_launch_dependents()
    if cutlass.const_expr(FOLD):
        if warp == Int32(0):
            cute.arch.cp_async_bulk_wait_group(0, read=False)
    if warp == Int32(0):
        prims.tcgen05_dealloc(cutlass.inttoptr(tmem_raw, 6, Float32), TC, group="cta_1")


@cute.jit
def _launch(
    pX: Int64,
    pZ: Int64,
    pA: Int64,
    pB: Int64,
    pAl: Int64,
    pDt: Int64,
    pNw: Int64,
    pScale: Int64,
    pState: Int64,
    pSi: Int64,
    pRi: Int64,
    pWp: Int64,
    pBase: Int64,
    pFl: Int64,
    pRd: Int64,
    pRk: Int64,
    pRg: Int64,
    pRdl: Int64,
    pRkl: Int64,
    gY: cute.Tensor,
    pO: Int64,
    stream,
    n_pairs: Int32,
    n_ctas: Int32,
    xs: Int32,
    zs: Int32,
    abst: Int32,
    s_slots: Int32,
    sss16: Int64,
    r_slots: Int32,
    d16: Int64,
    k16: Int64,
    d_slot: Int64,
    k_slot: Int64,
    g_slot: Int64,
    dl_slot: Int64,
    kl_slot: Int64,
    kl16: Int64,
    st_slot: Int64,
    dl16: Int64,
    eps: Float32,
    T_: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    HK: cutlass.Constexpr[int],
    NORM_F32: cutlass.Constexpr[bool],
    SI64: cutlass.Constexpr[bool],
    RI64: cutlass.Constexpr[bool],
    HAS_LO: cutlass.Constexpr[bool],
    EARLY: cutlass.Constexpr[bool],
    STORE_O: cutlass.Constexpr[bool],
    FOLD: cutlass.Constexpr[bool],
):
    BF = cutlass.BFloat16
    sw = cuda_exp.TensorMapSwizzle.s128b
    pY = Int64(
        gY.iterator.toint()
    )  # the output as a tensor: the launch's stream is the caller's (its device)
    tm_h = cuda_exp.create_tensor_map_tiled(
        global_address=pState,
        dtype=BF,
        global_dims=[D, D, HV, s_slots],
        global_strides=[D * 2 // 16, D * D * 2 // 16, sss16],
        box_dims=[64, D, 1, 1],
        swizzle=sw,
    )
    tm_u = cuda_exp.create_tensor_map_tiled(
        global_address=pRd,
        dtype=BF,
        global_dims=[D, RING, HV, r_slots],
        global_strides=[D * 2 // 16, RING * D * 2 // 16, d16],
        box_dims=[64, 8, 1, 1],
        swizzle=sw,
    )
    tm_k = cuda_exp.create_tensor_map_tiled(
        global_address=pRk,
        dtype=BF,
        global_dims=[D, RING, HK, r_slots],
        global_strides=[D * 2 // 16, RING * D * 2 // 16, k16],
        box_dims=[64, 8, 1, 1],
        swizzle=sw,
    )
    tm_kl = cuda_exp.create_tensor_map_tiled(
        global_address=pRkl,
        dtype=BF,
        global_dims=[D, RING, HK, r_slots],
        global_strides=[D * 2 // 16, RING * D * 2 // 16, kl16],
        box_dims=[64, 8, 1, 1],
        swizzle=sw,
    )
    tm_dl = cuda_exp.create_tensor_map_tiled(
        global_address=pRdl,
        dtype=BF,
        global_dims=[D, RING, HV, r_slots],
        global_strides=[D * 2 // 16, RING * D * 2 // 16, dl16],
        box_dims=[64, 8, 1, 1],
        swizzle=sw,
    )
    _verify_tc_kernel(
        tm_h,
        tm_u,
        tm_k,
        tm_kl,
        tm_dl,
        pX,
        pZ,
        pA,
        pB,
        pAl,
        pDt,
        pNw,
        pScale,
        pSi,
        pRi,
        pWp,
        pBase,
        pFl,
        pRd,
        pRk,
        pRg,
        pRdl,
        pRkl,
        pY,
        pO,
        pState,
        n_pairs,
        n_ctas,
        xs,
        zs,
        abst,
        d_slot,
        k_slot,
        g_slot,
        dl_slot,
        kl_slot,
        st_slot,
        eps,
        T_,
        HV,
        HK,
        NORM_F32,
        SI64,
        RI64,
        HAS_LO,
        EARLY,
        STORE_O,
        FOLD,
    ).launch(
        grid=(n_ctas, 1, 1),
        block=(NTHR, 1, 1),
        stream=stream,
        use_pdl=True,
        min_blocks_per_mp=CTAS_PER_SM,
        max_number_threads=[NTHR, 1, 1],
    )


@functools.lru_cache(maxsize=None)
def _compiled(
    T_: int,
    HV: int,
    HK: int,
    norm_f32: bool,
    si64: bool,
    ri64: bool,
    has_lo: bool,
    early: bool,
    store_o: bool = False,
    fold: bool = False,
):
    i64 = [Int64(1)] * 19
    return cute.compile(
        _launch,
        *i64,
        cute.runtime.make_fake_compact_tensor(cutlass.Uint8, (1,), assumed_align=1),
        Int64(1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        Int64(1),
        Int32(1),
        Int64(1),
        Int64(1),
        Int64(1),
        Int64(1),
        Int64(1),
        Int64(1),
        Int64(1),
        Int64(1),
        Int64(1),
        Int64(1),
        Float32(1e-6),
        T_=T_,
        HV=HV,
        HK=HK,
        NORM_F32=norm_f32,
        SI64=si64,
        RI64=ri64,
        HAS_LO=has_lo,
        EARLY=early,
        STORE_O=store_o,
        FOLD=fold,
        options="--enable-tvm-ffi",
    )


_NSM: dict = {}
_BUILT: set = set()


def _key(
    qkv,
    checkpoint,
    ring_dlo,
    ring_klo,
    state_indices,
    replay_indices,
    norm_weight,
    rows_per_request,
    num_k_heads,
    early_dependents,
    store_o=False,
    fold=False,
):
    return (
        rows_per_request,
        checkpoint.shape[1],
        num_k_heads,
        norm_weight.dtype == torch.float32,
        state_indices.dtype == torch.int64,
        replay_indices.dtype == torch.int64,
        ring_dlo is not None and ring_klo is not None,
        bool(early_dependents),
        bool(store_o),
        bool(fold),
    )


def _get(key):
    fn = _compiled(*key)
    _BUILT.add(key)
    return fn


def launchable(
    qkv,
    z,
    a,
    b,
    A_log,
    dt_bias,
    checkpoint,
    ring_d,
    ring_k,
    ring_g,
    ring_dlo,
    ring_klo,
    state_indices,
    replay_indices,
    write_pos,
    cache_base,
    is_flush,
    norm_weight,
    input_scale,
    eps,
    out,
    *,
    rows_per_request,
    num_k_heads,
    early_dependents=False,
    fold=False,
) -> bool:
    key = _key(
        qkv,
        checkpoint,
        ring_dlo,
        ring_klo,
        state_indices,
        replay_indices,
        norm_weight,
        rows_per_request,
        num_k_heads,
        early_dependents,
        fold=fold,
    )
    return key in _BUILT or not torch.cuda.is_current_stream_capturing()


def build(
    rows_per_request: int,
    hk: int,
    hv: int,
    norm_f32: bool,
    state_index64=(False,),
    replay_index64=(False, True),
    has_lo=(True,),
    early_dependents=(False,),
    fold=(False,),
) -> None:
    for fo in fold:
        for early in early_dependents:
            for si64 in state_index64:
                for ri64 in replay_index64:
                    for lo in has_lo:
                        if fo and not lo:
                            continue
                        _get(
                            (
                                rows_per_request,
                                hv,
                                hk,
                                bool(norm_f32),
                                si64,
                                ri64,
                                lo,
                                early,
                                False,
                                bool(fo),
                            )
                        )


def core_tc(
    qkv: torch.Tensor,
    z: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    checkpoint: torch.Tensor,
    ring_d: torch.Tensor,
    ring_k: torch.Tensor,
    ring_g: torch.Tensor,
    ring_dlo,
    ring_klo,
    state_indices: torch.Tensor,
    replay_indices: torch.Tensor,
    write_pos: torch.Tensor,
    cache_base: torch.Tensor,
    is_flush: torch.Tensor,
    norm_weight: torch.Tensor,
    input_scale: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    *,
    rows_per_request: int,
    num_k_heads: int,
    early_dependents: bool = False,
    o_out: Optional[torch.Tensor] = None,
    fold: bool = False,
) -> torch.Tensor:
    rows, T = qkv.shape[0], rows_per_request
    n = rows // T
    HV = checkpoint.shape[1]
    H = num_k_heads
    A_log, dt_bias = gate_f32(A_log), gate_f32(dt_bias)
    if (
        not 2 <= T <= 8
        or rows % T
        or state_indices.numel() < n
        or replay_indices.numel() < n
    ):
        raise ValueError(f"{rows} rows of {T} a request")
    if (
        HV % H
        or qkv.dtype != torch.bfloat16
        or qkv.stride(1) != 1
        or qkv.data_ptr() % 16
        or qkv.stride(0) % 8
    ):
        raise ValueError(
            "q | k | v rows must be BF16, unit column stride, 16-byte aligned"
        )
    if (
        tuple(checkpoint.shape[2:]) != (D, D)
        or checkpoint.stride()[1:] != (D * D, D, 1)
        or checkpoint.stride(0) % 8
    ):
        raise ValueError(
            f"checkpoint {tuple(checkpoint.shape)} {tuple(checkpoint.stride())}"
        )
    if fold and checkpoint.data_ptr() % 16:
        raise ValueError("the fold writes 16-byte checkpoint chunks")
    for t, heads in ((ring_d, HV), (ring_k, H), (ring_dlo, HV), (ring_klo, H)):
        if t is not None and (
            tuple(t.shape[1:]) != (heads, RING, D)
            or t.stride()[1:] != (RING * D, D, 1)
            or t.stride(0) % 8
            or t.data_ptr() % 16
        ):
            raise ValueError(f"ring {tuple(t.shape)} {tuple(t.stride())}")
    if tuple(ring_g.shape[1:]) != (HV, RING) or ring_g.stride()[1:] != (RING, 1):
        raise ValueError(f"ring g {tuple(ring_g.shape)}")
    if (
        z.stride(2) != 1
        or z.stride(1) != D
        or a.stride(1) != 1
        or b.stride(1) != 1
        or a.stride(0) != b.stride(0)
    ):
        raise ValueError("z / a / b layouts")
    if (
        out.dtype != torch.float8_e4m3fn
        or out.shape != (rows, HV * D)
        or out.stride(0) != HV * D
    ):
        raise ValueError(f"out {out.dtype} {tuple(out.shape)}")
    if (
        write_pos.dtype != torch.int32
        or cache_base.dtype != torch.int32
        or is_flush.dtype != torch.int8
    ):
        raise ValueError("ring cursors int32, is_flush int8")
    has_lo = ring_dlo is not None and ring_klo is not None
    if fold and not has_lo:
        raise ValueError(
            "the fold needs the low-part rings (the compact commit's operands)"
        )
    dev = (
        qkv.device.index
        if qkv.device.index is not None
        else torch.cuda.current_device()
    )
    if dev not in _NSM:
        _NSM[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    n_pairs = n * HV
    n_ctas = min(n_pairs, CTAS_PER_SM * _NSM[dev])
    fn = _get(
        _key(
            qkv,
            checkpoint,
            ring_dlo,
            ring_klo,
            state_indices,
            replay_indices,
            norm_weight,
            T,
            H,
            early_dependents,
            o_out is not None,
            fold,
        )
    )
    if o_out is not None and (
        o_out.dtype != torch.bfloat16
        or not o_out.is_contiguous()
        or o_out.numel() != rows * HV * D
    ):
        raise ValueError("o_out must be [rows, HV, V] BF16, contiguous")
    rdl = ring_dlo if has_lo else ring_d
    rkl = ring_klo if has_lo else ring_k
    fn(
        qkv.data_ptr(),
        z.data_ptr(),
        a.data_ptr(),
        b.data_ptr(),
        A_log.data_ptr(),
        dt_bias.data_ptr(),
        norm_weight.data_ptr(),
        input_scale.data_ptr(),
        checkpoint.data_ptr(),
        state_indices.data_ptr(),
        replay_indices.data_ptr(),
        write_pos.data_ptr(),
        cache_base.data_ptr(),
        is_flush.data_ptr(),
        ring_d.data_ptr(),
        ring_k.data_ptr(),
        ring_g.data_ptr(),
        rdl.data_ptr(),
        rkl.data_ptr(),
        torch.as_strided(out.view(torch.uint8), (1,), (1,)),
        0 if o_out is None else o_out.data_ptr(),
        n_pairs,
        n_ctas,
        qkv.stride(0),
        z.stride(0),
        a.stride(0),
        checkpoint.shape[0],
        checkpoint.stride(0) * 2 // 16,
        ring_d.shape[0],
        ring_d.stride(0) * 2 // 16,
        ring_k.stride(0) * 2 // 16,
        ring_d.stride(0),
        ring_k.stride(0),
        ring_g.stride(0),
        rdl.stride(0),
        rkl.stride(0),
        rkl.stride(0) * 2 // 16,
        checkpoint.stride(0),
        rdl.stride(0) * 2 // 16,
        float(eps),
    )
    return out
