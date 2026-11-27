# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from TensorRT-LLM (Apache-2.0): tensorrt_llm/_torch/kernels/blackwell/gemm_twoshot_allreduce_norm.py
# (the fused o_proj / out_proj tcgen05 GEMM + two-shot all-reduce + residual RMSNorm).
# This file has been modified from its original version (the rounding points of the owner add and the norm).

from __future__ import annotations

import functools
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda_exp
import torch
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.experimental import primitives as prims

from sgl_nvidia_mega.qwen3_5.sm103.hop.slot_ar_kernel import (  # the shared PTX helpers
    _NEG_ZERO,
    _add,
    _bf16r,
    _e4m3x2,
    _hi,
    _ld_s64,
    _ld_v4,
    _ld_volatile_v4,
    _lo,
    _mul,
    _pack_bf16,
    _rsqrt,
    _sanitize,
    _shfl_bfly,
    _st_s32,
    _st_v2,
    _st_v4,
)

MMA_K = 128
K_GRANULE = 32  # FP8 K of one MMA
MMA_N = 32  # output columns per CTA
SW_ALIGN = 1024
TMA_WARP, MMA_WARP, EPI_WARP0 = 0, 1, 2
THREADS = 6 * 32
EPI_THREADS = 4 * 32
MIN_ROWS, MAX_ROWS = 1, 128
NORM_BAR = 1  # named barrier of the four epilogue warps
TAIL_WARPS, LOG_TAIL_WARPS = 16, 4
OPT_SPREAD = 8
OPT_TIGHT = 64
OPT_UNIFORM, OPT_PFDESC = 128, 256
OPT_PAD = 512
OPT_ONESM = 16384
ONESM_BYTES = 118 * 1024
OPT_EARLYALPHA = 16777216
OPT_ROWGATE = 33554432
OPT_LIBNUM = 67108864
GATE_BAR = 3  # named barrier of OPT_ROWGATE over all THREADS (0 = CTA, 1 = NORM_BAR)
OWN_WARPS = 2  # warps 0-1 (TMA, MMA) take the spread owner items
V5_OPTS = OPT_SPREAD | OPT_TIGHT | OPT_UNIFORM | OPT_PFDESC | OPT_PAD
DEFAULT_OPTS = V5_OPTS | OPT_ONESM | OPT_EARLYALPHA | OPT_ROWGATE


def per_cta(m: int, hidden: int, nranks: int, opts: int) -> int:
    mp = (m + 7) // 8 * 8 if opts & OPT_PAD else m
    nctas = hidden // MMA_N
    total = (nctas // nranks) * (MMA_N // 8) * mp
    return -(-total // nctas)


def _a(x, loc, ip):
    return x.ir_value(loc=loc, ip=ip)


@dsl_user_op
def _vote_any(p: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [_a(Int32(p), loc, ip)],
            "{\n\t.reg .pred pa, pb;\n\tsetp.ne.s32 pa, $1, 0;\n\tvote.sync.any.pred pb, pa, 0xffffffff;\n\t"
            "selp.s32 $0, 1, 0, pb;\n\t}",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shfl_idx(x: Float32, src: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [_a(Float32(x), loc, ip), _a(Int32(src), loc, ip)],
            "shfl.sync.idx.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=f,f,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _div_full(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [_a(Float32(a), loc, ip), _a(Float32(b), loc, ip)],
            "div.full.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _clamp448(x: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [_a(Float32(x), loc, ip)],
            "{\n\t.reg .f32 t;\n\tmax.f32 t, $1, 0fC3E00000;\n\tmin.f32 $0, t, 0f43E00000;\n\t}",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _clear_dirty(
    own: Int64,
    fl,
    buf_bytes: Int64,
    dirty: Int32,
    dirty_stages: Int32,
    gtid: Int64,
    gthreads: Int64,
):
    neg0 = Int32(_NEG_ZERO)
    for s_i in cutlass.range_constexpr(4):
        if Int32(s_i) < dirty_stages:
            stage_bytes = buf_bytes // Int64(dirty_stages)
            stage = (
                own + (Int64(dirty) * Int64(dirty_stages) + Int64(s_i)) * stage_bytes
            )
            sb = fl.subview(4 + s_i).load()
            clear_vecs = (Int64(sb) + Int64(15)) // Int64(16)
            c = gtid
            while c < clear_vecs:
                _st_v4(stage + c * Int64(16), neg0, neg0, neg0, neg0)
                c = c + gthreads


@cute.kernel
def _kernel(
    tma_a_desc: cutlass.GridConstant[cuda_exp.TensorMap],
    tma_w_desc: cutlass.GridConstant[cuda_exp.TensorMap],
    gSin: cute.Tensor,
    gSw: cute.Tensor,
    gRes: cute.Tensor,
    gNw: cute.Tensor,
    gScale: cute.Tensor,
    gResOut: cute.Tensor,
    gOut: cute.Tensor,
    gQuant: cute.Tensor,
    gWs: cute.Tensor,
    rank: Int32,
    active_m: cutlass.Constexpr[int],
    hidden: cutlass.Constexpr[int],
    stages: cutlass.Constexpr[int],
    nranks: cutlass.Constexpr[int],
    norm_eps: cutlass.Constexpr[float],
    quant: cutlass.Constexpr[bool],
    opts: cutlass.Constexpr[int],
    mma_m: cutlass.Constexpr[int],
    box_m: cutlass.Constexpr[int],
):
    AB: cutlass.Constexpr = tma_a_desc.dtype
    SWZ: cutlass.Constexpr = tma_a_desc.swizzle.to(prims.Tcgen05SmemSwizzle)
    UNIFORM: cutlass.Constexpr[bool] = (opts & OPT_UNIFORM) != 0
    PFDESC: cutlass.Constexpr[bool] = (opts & OPT_PFDESC) != 0
    PAD: cutlass.Constexpr[bool] = (opts & OPT_PAD) != 0
    EARLYALPHA: cutlass.Constexpr[bool] = (opts & OPT_EARLYALPHA) != 0
    ROWGATE: cutlass.Constexpr[bool] = (opts & OPT_ROWGATE) != 0
    LIBNUM: cutlass.Constexpr[bool] = (opts & OPT_LIBNUM) != 0
    MP: cutlass.Constexpr[int] = (
        (active_m + 7) // 8 * 8 if (opts & OPT_PAD) else active_m
    )  # Lamport row stride
    DESC_LEAD: cutlass.Constexpr[int] = MMA_K // 16
    DESC_STRIDE: cutlass.Constexpr[int] = 8 * MMA_K // 16
    A_STAGE: cutlass.Constexpr[int] = (
        box_m * MMA_K
    )  # activation rows TMA writes per stage (= mma_m unless tight)
    A_TAIL: cutlass.Constexpr[int] = (mma_m - box_m) * MMA_K
    W_STAGE: cutlass.Constexpr[int] = MMA_N * MMA_K
    ONESM_PAD: cutlass.Constexpr[int] = (
        max(0, ONESM_BYTES - stages * (A_STAGE + W_STAGE) - A_TAIL)
        if ((opts & OPT_ONESM) and mma_m == 128)
        else 0
    )
    NKB: cutlass.Constexpr[int] = MMA_K // K_GRANULE
    ROWS_PER_SUB: cutlass.Constexpr[int] = mma_m // 4
    VECS: cutlass.Constexpr[int] = hidden // 8
    VPT: cutlass.Constexpr[int] = (
        VECS // EPI_THREADS
    )  # row-phase vectors per epilogue thread
    CVECS: cutlass.Constexpr[int] = MMA_N // 8  # vectors of a CTA's column slice
    ROW_BYTES: cutlass.Constexpr[int] = hidden * 2
    NCTAS: cutlass.Constexpr[int] = hidden // MMA_N
    CPR: cutlass.Constexpr[int] = NCTAS // nranks  # slices each rank owns
    SLICE_ITEMS: cutlass.Constexpr[int] = (
        CVECS * MP
    )  # (vector, row) slots of one slice in the Lamport layout
    OWN_TOTAL: cutlass.Constexpr[int] = (
        CPR * SLICE_ITEMS
    )  # slots a rank owns (= items without padding)
    PER_CTA: cutlass.Constexpr[int] = (OWN_TOTAL + NCTAS - 1) // NCTAS

    smem_a = cutlass.Array(
        AB,
        stages * A_STAGE + A_TAIL,
        space=cutlass.AddressSpace.smem,
        alignment=SW_ALIGN,
    )
    smem_w = cutlass.Array(
        AB, stages * W_STAGE, space=cutlass.AddressSpace.smem, alignment=SW_ALIGN
    )
    full_bar = cutlass.Array(
        cutlass.Int64, stages, space=cutlass.AddressSpace.smem, alignment=8
    )
    acc_bar = cutlass.Array(
        cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8
    )
    tmem_i32 = cutlass.Array(Int32, 1, space=cutlass.AddressSpace.smem, alignment=4)
    fl = cutlass.Array(Int32, 8, space=cutlass.AddressSpace.smem, alignment=16)
    nred = cutlass.Array(
        Float32, 16, space=cutlass.AddressSpace.smem, alignment=16
    )  # the tail's warp sums
    sinkw = cutlass.Array(Int32, 4, space=cutlass.AddressSpace.smem, alignment=16)
    if cutlass.const_expr(ONESM_PAD > 0):
        onesm = cutlass.Array(
            Int32, ONESM_PAD // 4, space=cutlass.AddressSpace.smem, alignment=16
        )
        if rank < Int32(0):
            onesm.subview(ONESM_PAD // 4 - 1).store(rank)

    warp_idx = cute.arch.warp_idx()
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    lane = tidx % Int32(32)
    coord_n = bidx * Int32(MMA_N)
    is_epi = warp_idx >= Int32(EPI_WARP0)
    epi_tid = tidx - Int32(EPI_WARP0 * 32)
    vbase = Int32(32) * (
        Int32(4) * (warp_idx - Int32(EPI_WARP0)) + (lane >> Int32(3))
    ) + (lane & Int32(7))
    my_row = (warp_idx % Int32(4)) * Int32(ROWS_PER_SUB) + lane
    row_valid = is_epi & (lane < Int32(ROWS_PER_SUB)) & (my_row < Int32(active_m))
    row_cta = bidx < Int32(active_m)  # this CTA normalizes row bidx
    owner = bidx % Int32(nranks)  # the rank that reduces column slice bidx
    lslice = bidx // Int32(nranks)  # its index among that rank's slices
    g = bidx * Int32(PER_CTA) + tidx
    own_live = (
        (warp_idx < Int32(OWN_WARPS)) & (tidx < Int32(PER_CTA)) & (g < Int32(OWN_TOTAL))
    )
    lsl = g // Int32(SLICE_ITEMS)
    it_ = g - lsl * Int32(SLICE_ITEMS)
    o_vec = it_ // Int32(MP)
    o_row = it_ - o_vec * Int32(MP)
    if cutlass.const_expr(PAD):
        own_live = own_live & (
            o_row < Int32(active_m)
        )  # padding rows are never written
    idesc = prims.Tcgen05InstrDesc.build(
        c_dtype=Float32, a_dtype=AB, b_dtype=AB, n_dim=MMA_N, m_dim=mma_m
    )

    if warp_idx == TMA_WARP:
        if prims.elect_sync():
            for s in cutlass.range_constexpr(stages):
                prims.mbarrier_init(full_bar.subview(s), 1)
            prims.mbarrier_init(acc_bar, 1)
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)
    if warp_idx == TMA_WARP:
        prims.tcgen05_alloc(tmem_i32, 32, group="cta_1")
    prims.barrier_cta_sync(0)
    tmem_raw = tmem_i32.load()
    tmem_ptr = cutlass.inttoptr(tmem_raw, 6, Float32)
    prims.tcgen05_relinquish_alloc_permit(group="cta_1")

    tbl = Int64(gWs.iterator.toint())
    own = _ld_s64(tbl)
    mcast = _ld_s64(tbl + Int64(8))
    flags = _ld_s64(tbl + Int64(16))
    peer = _ld_s64(tbl + Int64(24) + Int64(owner) * Int64(8))
    nwr = cutlass.Array(Int32, 4 * VPT, alignment=16)
    if is_epi:
        for j in cutlass.range_constexpr(VPT):
            a0, a1, a2, a3 = _ld_v4(
                Int64(gNw.iterator.toint()) + (Int64(vbase) + Int64(8 * j)) * Int64(16)
            )
            nwr[4 * j + 0] = a0
            nwr[4 * j + 1] = a1
            nwr[4 * j + 2] = a2
            nwr[4 * j + 3] = a3

    a_ptr = tma_a_desc.get_ptr()
    w_ptr = tma_w_desc.get_ptr()
    tma_bytes = tma_a_desc.global_tx_bytes() + tma_w_desc.global_tx_bytes()
    desc_a = prims.Tcgen05SmemDesc.build(
        start_address=smem_a,
        leading_byte_offset=DESC_LEAD * 16,
        stride_byte_offset=DESC_STRIDE * 16,
        layout=SWZ,
    )
    desc_w = prims.Tcgen05SmemDesc.build(
        start_address=smem_w,
        leading_byte_offset=DESC_LEAD * 16,
        stride_byte_offset=DESC_STRIDE * 16,
        layout=SWZ,
    )

    # ---- TMA: weight tiles, dependency wait, activation tiles ----
    if warp_idx == TMA_WARP:
        if prims.elect_sync():
            if cutlass.const_expr(PFDESC):
                prims.prefetch_tensormap(a_ptr)
                prims.prefetch_tensormap(w_ptr)
            for s in cutlass.range_constexpr(stages):
                prims.mbarrier_arrive_expect_tx(full_bar.subview(s), tma_bytes)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    smem_w.subview(s * W_STAGE),
                    w_ptr,
                    (Int32(s * MMA_K), coord_n),
                    full_bar.subview(s),
                )
            cute.arch.griddepcontrol_wait()
            for s in cutlass.range_constexpr(stages):
                prims.cp_async_bulk_tensor_shared_cta_global(
                    smem_a.subview(s * A_STAGE),
                    a_ptr,
                    (Int32(s * MMA_K), Int32(0)),
                    full_bar.subview(s),
                )

    # ---- MMA ----
    if warp_idx == MMA_WARP:
        for s in cutlass.range_constexpr(stages):
            while not prims.mbarrier_try_wait_parity(
                full_bar.subview(s), 0, time_limit=10_000_000
            ):
                pass
            for kb in cutlass.range_constexpr(NKB):
                off_a = Int32(s * A_STAGE + kb * K_GRANULE)
                off_w = Int32(s * W_STAGE + kb * K_GRANULE)
                if prims.elect_sync():
                    prims.tcgen05_mma(
                        prims.Tcgen05MMAKind.F8F6F4,
                        prims.CTAGroup.CTA_1,
                        tmem_ptr,
                        desc_a.advance_start_address(off_a),
                        desc_w.advance_start_address(off_w),
                        idesc,
                        (s > 0) or (kb > 0),
                    )
        if prims.elect_sync():
            prims.tcgen05_commit(acc_bar, group=prims.CTAGroup.CTA_1)

    ores = cutlass.Array(Int32, 4, alignment=16)
    if warp_idx < Int32(OWN_WARPS):
        cute.arch.griddepcontrol_wait()
        if own_live:
            q0, q1, q2, q3 = _ld_v4(
                Int64(gRes.iterator.toint())
                + (
                    (
                        Int64(o_row) * Int64(NCTAS)
                        + Int64(lsl) * Int64(nranks)
                        + Int64(rank)
                    )
                    * Int64(CVECS)
                    + Int64(o_vec)
                )
                * Int64(16)
            )
            ores[0] = q0
            ores[1] = q1
            ores[2] = q2
            ores[3] = q3

    # ---- epilogue warps: dependency wait, the Lamport flags, the accumulator ----
    acc = cutlass.Array(Float32, MMA_N, alignment=32)
    alr = cutlass.Array(
        Float32, 1
    )  # OPT_EARLYALPHA: alpha, loaded right after the dependency wait
    row_off = Int64(bidx) * Int64(VECS) + Int64(
        vbase
    )  # vector index of this thread's j = 0 vector of row bidx
    if is_epi:
        cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(EARLYALPHA):
            alr[0] = _mul(gSin[0], gSw[0])
        if epi_tid == Int32(0):
            f0, f1, f2, f3 = _ld_volatile_v4(flags)
            f4, f5, f6, f7 = _ld_volatile_v4(flags + Int64(16))
            fl.subview(0).store(f0)
            fl.subview(1).store(f1)
            fl.subview(2).store(f2)
            fl.subview(3).store(f3)
            fl.subview(4).store(f4)
            fl.subview(5).store(f5)
            fl.subview(6).store(f6)
            fl.subview(7).store(f7)
        while not prims.mbarrier_try_wait_parity(acc_bar, 0, time_limit=10_000_000):
            pass
        base = prims.TmemAddr(tmem_raw)
        tsub = prims.TmemAddr.from_row_col(base.row_id, base.col_id).as_ptr(Float32)
        acc.store(prims.tcgen05_ld("32x32b", tsub, num=MMA_N), 0)
        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
    cute.arch.sync_warp()
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    prims.barrier_cta_sync(0)
    if warp_idx == TMA_WARP:
        prims.tcgen05_dealloc(tmem_ptr, 32, group="cta_1")

    cur = fl.subview(0).load()
    dirty = fl.subview(1).load()
    buf_bytes = Int64(fl.subview(2).load())
    dirty_stages = fl.subview(3).load()
    data_off = Int64(cur) * buf_bytes
    half = buf_bytes // Int64(2)
    neg0 = Int32(_NEG_ZERO)

    if row_valid:
        if cutlass.const_expr(EARLYALPHA):
            alpha = alr[0]
        else:
            alpha = _mul(gSin[0], gSw[0])
        dst = (
            peer
            + data_off
            + (
                (Int64(rank) * Int64(CPR) + Int64(lslice)) * Int64(CVECS) * Int64(MP)
                + Int64(my_row)
            )
            * Int64(16)
        )
        for v in cutlass.range_constexpr(CVECS):
            w0 = _sanitize(
                _pack_bf16(_mul(acc[8 * v + 0], alpha), _mul(acc[8 * v + 1], alpha))
            )
            w1 = _sanitize(
                _pack_bf16(_mul(acc[8 * v + 2], alpha), _mul(acc[8 * v + 3], alpha))
            )
            w2 = _sanitize(
                _pack_bf16(_mul(acc[8 * v + 4], alpha), _mul(acc[8 * v + 5], alpha))
            )
            w3 = _sanitize(
                _pack_bf16(_mul(acc[8 * v + 6], alpha), _mul(acc[8 * v + 7], alpha))
            )
            _st_v4(dst + Int64(16 * v * MP), w0, w1, w2, w3)

    if own_live:
        sinkw.subview(0).store(ores[0] ^ ores[1] ^ ores[2] ^ ores[3])
    cute.arch.sync_warp()
    prims.barrier_cta_sync(0)
    if tidx == Int32(0):
        cute.arch.griddepcontrol_launch_dependents()

    gtid = Int64(bidx) * Int64(THREADS) + Int64(tidx)
    gthreads = Int64(NCTAS * THREADS)
    _clear_dirty(own, fl, buf_bytes, dirty, dirty_stages, gtid, gthreads)

    if warp_idx < Int32(OWN_WARPS):
        if own_live:
            sgot = cutlass.Array(Int32, 4 * nranks, alignment=16)
            spend = Int32(1)
            while spend != Int32(0):
                spend = Int32(0)
                for rk in cutlass.range_constexpr(nranks):
                    # SCATTER [src rank][local slice][vector][row] = rk * OWN_TOTAL + g
                    h0, h1, h2, h3 = _ld_volatile_v4(
                        own + data_off + (Int64(rk * OWN_TOTAL) + Int64(g)) * Int64(16)
                    )
                    sgot[4 * rk + 0] = h0
                    sgot[4 * rk + 1] = h1
                    sgot[4 * rk + 2] = h2
                    sgot[4 * rk + 3] = h3
                    if h0 == neg0:
                        spend = Int32(1)
                    if h1 == neg0:
                        spend = Int32(1)
                    if h2 == neg0:
                        spend = Int32(1)
                    if h3 == neg0:
                        spend = Int32(1)
            ssum = cutlass.Array(Int32, 4, alignment=16)
            for i in cutlass.range_constexpr(4):
                tlo = Float32(0.0)
                thi = Float32(0.0)
                for rk in cutlass.range_constexpr(nranks):
                    tlo = _add(tlo, _lo(sgot[4 * rk + i]))
                    thi = _add(thi, _hi(sgot[4 * rk + i]))
                if cutlass.const_expr(LIBNUM):
                    tlo = _bf16r(tlo)
                    thi = _bf16r(thi)
                ssum[i] = _sanitize(
                    _pack_bf16(_add(tlo, _lo(ores[i])), _add(thi, _hi(ores[i])))
                )
            # BROADCAST [slice][vector][row], slice = local slice * W + rank
            _st_v4(
                mcast
                + data_off
                + half
                + (
                    (Int64(lsl) * Int64(nranks) + Int64(rank)) * Int64(SLICE_ITEMS)
                    + Int64(it_)
                )
                * Int64(16),
                ssum[0],
                ssum[1],
                ssum[2],
                ssum[3],
            )
        if cutlass.const_expr(ROWGATE):
            if row_cta:
                # this CTA's owner items are stored: release the epilogue warps' row polls
                cute.arch.barrier(barrier_id=GATE_BAR, number_of_threads=THREADS)

    if cutlass.const_expr(ROWGATE):
        if is_epi & row_cta:
            cute.arch.barrier(barrier_id=GATE_BAR, number_of_threads=THREADS)
    if is_epi & row_cta:
        res = cutlass.Array(Float32, 8 * VPT, alignment=32)
        bwa = cutlass.Array(Int32, 4 * VPT, alignment=16)
        pend = Int32(1)
        while pend != Int32(0):
            pend = Int32(0)
            for j in cutlass.range_constexpr(VPT):
                c0, c1, c2, c3 = _ld_volatile_v4(
                    own
                    + data_off
                    + half
                    + ((Int64(vbase) + Int64(8 * j)) * Int64(MP) + Int64(bidx))
                    * Int64(16)
                )
                bwa[4 * j + 0] = c0
                bwa[4 * j + 1] = c1
                bwa[4 * j + 2] = c2
                bwa[4 * j + 3] = c3
                if c0 == neg0:
                    pend = Int32(1)
                if c1 == neg0:
                    pend = Int32(1)
                if c2 == neg0:
                    pend = Int32(1)
                if c3 == neg0:
                    pend = Int32(1)
            if cutlass.const_expr(UNIFORM):
                pend = _vote_any(pend)
        for j in cutlass.range_constexpr(VPT):
            _st_v4(
                Int64(gResOut.iterator.toint()) + (row_off + Int64(8 * j)) * Int64(16),
                bwa[4 * j + 0],
                bwa[4 * j + 1],
                bwa[4 * j + 2],
                bwa[4 * j + 3],
            )
            for i in cutlass.range_constexpr(4):
                res[8 * j + 2 * i] = _lo(bwa[4 * j + i])
                res[8 * j + 2 * i + 1] = _hi(bwa[4 * j + i])
        qs = []
        for j in cutlass.range_constexpr(VPT):
            q = Float32(0.0)
            for e in cutlass.range_constexpr(8):
                q = _add(q, _mul(res[8 * j + e], res[8 * j + e]))
            qs.append(q)
        y = _add(_add(qs[0], qs[2]), _add(qs[1], qs[3]))
        y = _add(y, _shfl_bfly(y, Int32(4)))
        y = _add(y, _shfl_bfly(y, Int32(2)))
        y = _add(y, _shfl_bfly(y, Int32(1)))
        if (lane & Int32(7)) == Int32(0):
            nred.subview(
                Int32(4) * (warp_idx - Int32(EPI_WARP0)) + (lane >> Int32(3))
            ).store(y)
        cute.arch.barrier(barrier_id=NORM_BAR, number_of_threads=EPI_THREADS)
        u = nred.subview(lane & Int32(15)).load()
        for st in cutlass.range_constexpr(LOG_TAIL_WARPS):
            u = _add(u, _shfl_bfly(u, Int32(TAIL_WARPS >> (st + 1))))
        total = _shfl_idx(u, Int32(0))
        rstd = _rsqrt(
            _add(_mul(total, Float32(1.0 / float(hidden))), Float32(norm_eps))
        )
        inv = Float32(1.0)
        if cutlass.const_expr(quant):
            inv = _div_full(Float32(1.0), gScale[0])
        for j in cutlass.range_constexpr(VPT):
            y = cutlass.Array(Float32, 8, alignment=32)
            for i in cutlass.range_constexpr(4):
                y[2 * i] = _bf16r(
                    _mul(_mul(res[8 * j + 2 * i], rstd), _lo(nwr[4 * j + i]))
                )
                y[2 * i + 1] = _bf16r(
                    _mul(_mul(res[8 * j + 2 * i + 1], rstd), _hi(nwr[4 * j + i]))
                )
            vrow = row_off + Int64(8 * j)
            _st_v4(
                Int64(gOut.iterator.toint()) + vrow * Int64(16),
                _pack_bf16(y[0], y[1]),
                _pack_bf16(y[2], y[3]),
                _pack_bf16(y[4], y[5]),
                _pack_bf16(y[6], y[7]),
            )
            if cutlass.const_expr(quant):
                q01 = _e4m3x2(_clamp448(_mul(y[0], inv)), _clamp448(_mul(y[1], inv)))
                q23 = _e4m3x2(_clamp448(_mul(y[2], inv)), _clamp448(_mul(y[3], inv)))
                q45 = _e4m3x2(_clamp448(_mul(y[4], inv)), _clamp448(_mul(y[5], inv)))
                q67 = _e4m3x2(_clamp448(_mul(y[6], inv)), _clamp448(_mul(y[7], inv)))
                _st_v2(
                    Int64(gQuant.iterator.toint()) + vrow * Int64(8),
                    q01 | (q23 << 16),
                    q45 | (q67 << 16),
                )
        if bidx == Int32(0):
            if epi_tid == Int32(0):
                _st_s32(flags, (cur + Int32(1)) % Int32(3))
                _st_s32(flags + Int64(4), cur)
                _st_s32(flags + Int64(12), Int32(2))
                _st_s32(flags + Int64(16), Int32(MP * ROW_BYTES))
                _st_s32(flags + Int64(20), Int32(MP * ROW_BYTES))
                _st_s32(flags + Int64(24), Int32(0))
                _st_s32(flags + Int64(28), Int32(0))
                _st_s32(flags + Int64(32), Int32(0))


@cute.jit
def _launch(
    a: cute.Tensor,
    w: cute.Tensor,
    s_in: cute.Tensor,
    s_w: cute.Tensor,
    res: cute.Tensor,
    nw: cute.Tensor,
    scale: cute.Tensor,
    res_out: cute.Tensor,
    out: cute.Tensor,
    qout: cute.Tensor,
    table: cute.Tensor,
    rank: Int32,
    stream: cuda.CUstream,
    active_m: cutlass.Constexpr[int],
    hidden: cutlass.Constexpr[int],
    k_total: cutlass.Constexpr[int],
    nranks: cutlass.Constexpr[int],
    norm_eps: cutlass.Constexpr[float],
    quant: cutlass.Constexpr[bool],
    opts: cutlass.Constexpr[int],
):
    mma_m = 64 if active_m <= 64 else 128
    box_m = min(mma_m, (active_m + 7) // 8 * 8) if (opts & OPT_TIGHT) else mma_m
    tma_a = cuda_exp.create_tensor_map_tiled(
        global_address=a.iterator.toint(),
        dtype=cutlass.Float8E4M3FN,
        global_dims=[k_total, active_m],
        global_strides=[k_total // 16],
        box_dims=[MMA_K, box_m],
        swizzle=cuda_exp.TensorMapSwizzle.s128b,
    )
    tma_w = cuda_exp.create_tensor_map_tiled(
        global_address=w.iterator.toint(),
        dtype=cutlass.Float8E4M3FN,
        global_dims=[k_total, hidden],
        global_strides=[k_total // 16],
        box_dims=[MMA_K, MMA_N],
        swizzle=cuda_exp.TensorMapSwizzle.s128b,
    )
    k = _kernel(
        tma_a,
        tma_w,
        s_in,
        s_w,
        res,
        nw,
        scale,
        res_out,
        out,
        qout,
        table,
        rank,
        active_m,
        hidden,
        k_total // MMA_K,
        nranks,
        norm_eps,
        quant,
        opts,
        mma_m,
        box_m,
    )
    k.launch(
        grid=(hidden // MMA_N, 1, 1),
        block=(THREADS, 1, 1),
        stream=stream,
        use_pdl=True,
    )


@functools.lru_cache(maxsize=None)
def _compiled(
    m: int, k: int, hidden: int, nranks: int, eps: float, quant: bool, opts: int
):
    assert opts & OPT_SPREAD, "the owner phase carries the residual load: OPT_SPREAD"
    assert per_cta(m, hidden, nranks, opts) <= OWN_WARPS * 32, (m, hidden, nranks, opts)
    mk = cute.runtime.make_fake_compact_tensor
    u8 = cutlass.Uint8
    return cute.compile(
        _launch,
        mk(u8, (m, k), stride_order=(1, 0), assumed_align=128),
        mk(u8, (hidden, k), stride_order=(1, 0), assumed_align=128),
        mk(cutlass.Float32, (1,), assumed_align=4),
        mk(cutlass.Float32, (1,), assumed_align=4),
        mk(cutlass.BFloat16, (m, hidden), stride_order=(1, 0), assumed_align=16),
        mk(cutlass.BFloat16, (hidden,), assumed_align=16),
        mk(cutlass.Float32, (1,), assumed_align=4),
        mk(cutlass.BFloat16, (m, hidden), stride_order=(1, 0), assumed_align=16),
        mk(cutlass.BFloat16, (m, hidden), stride_order=(1, 0), assumed_align=16),
        mk(u8, (m, hidden), stride_order=(1, 0), assumed_align=16),
        mk(cutlass.Int64, (3 + nranks,), assumed_align=8),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        active_m=m,
        hidden=hidden,
        k_total=k,
        nranks=nranks,
        norm_eps=eps,
        quant=quant,
        opts=opts,
        options="--enable-tvm-ffi",
    )


_DUMMY = {}


def _dummy(kind: str, shape, dtype, device) -> torch.Tensor:
    key = (kind, tuple(shape), str(device))
    t = _DUMMY.get(key)
    if t is None:
        t = _DUMMY[key] = torch.zeros(shape, dtype=dtype, device=device)
    return t


def unsupported(
    m: int, k: int, hidden: int, nranks: int, sms: int, opts: int = DEFAULT_OPTS
) -> str:
    if not MIN_ROWS <= m <= MAX_ROWS:
        return f"{m} rows (serves {MIN_ROWS}-{MAX_ROWS})"
    if k % MMA_K or not 1 <= k // MMA_K <= 8:
        return f"K {k} (needs a multiple of {MMA_K}, at most {8 * MMA_K})"
    if (
        hidden % MMA_N
        or hidden // MMA_N > sms
        or (hidden // MMA_N) % nranks
        or hidden // 8 != 4 * EPI_THREADS
    ):
        return f"hidden {hidden} at world {nranks} on {sms} SMs"
    if nranks not in (2, 4, 8, 16):
        return f"world {nranks}"
    if per_cta(m, hidden, nranks, opts) > OWN_WARPS * 32:
        return f"{m} rows at world {nranks}: more owner items per CTA than its owner warps' 64 threads"
    return ""


def max_rows(hidden: int, nranks: int, opts: int = DEFAULT_OPTS) -> int:
    return max(
        (
            m
            for m in range(MIN_ROWS, MAX_ROWS + 1)
            if per_cta(m, hidden, nranks, opts) <= OWN_WARPS * 32
        ),
        default=0,
    )


def workspace_rows(m: int) -> int:
    return (m + 7) // 8 * 8


def gemm_ar_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    table: torch.Tensor,
    rank: int,
    nranks: int,
    scale: Optional[torch.Tensor] = None,
    opts: int = DEFAULT_OPTS,
) -> Tuple[torch.Tensor, ...]:
    m, k = x.shape
    hidden = weight.shape[0]
    dev = residual.device
    fn = _compiled(m, k, hidden, nranks, float(eps), scale is not None, opts)
    res_out = torch.empty_like(residual)
    norm = torch.empty_like(residual)
    quant = (
        torch.empty(residual.shape, dtype=torch.uint8, device=dev)
        if scale is not None
        else None
    )
    sc = scale if scale is not None else input_scale
    fn(
        x.view(torch.uint8),
        weight.view(torch.uint8),
        input_scale.reshape(1).float(),
        weight_scale.reshape(1).float(),
        residual,
        norm_weight.reshape(-1),
        sc.reshape(1).float(),
        res_out,
        norm,
        quant if quant is not None else _dummy("q", (m, hidden), torch.uint8, dev),
        table,
        rank,
    )
    if scale is None:
        return norm, res_out
    return norm, res_out, quant.view(torch.float8_e4m3fn)
