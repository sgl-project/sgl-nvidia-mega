# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from TensorRT-LLM (Apache-2.0): tensorrt_llm/_torch/kernels/blackwell/moe_bf16_engine.py.

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Int32, Int64
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import (
    Agent,
    CooperativeGroup,
    PipelineAsync,
    PipelineUmmaAsync,
    PipelineUserType,
    make_pipeline_state,
    pipeline_init_arrive,
    pipeline_init_wait,
)

from sgl_nvidia_mega.qwen3_5.sm103.draft.draftmoe_front import (
    _SORT16,
    _below,
    _key1,
    _max_u32,
    _min_u32,
    _redux_max_u32,
    _sel_eq_f,
)
from sgl_nvidia_mega.qwen3_5.sm103.moe.q35_moe_engine import (
    _add_rn,
    _atom_add_acq_rel,
    _atom_add_relaxed,
    _bar_sync,
    _cp_async_16,
    _cp_async_arrive_noinc,
    _cp_async_commit,
    _ex2,
    _fence_acq_rel_gpu,
    _ld_acquire_u32,
    _ld_relaxed_u32,
    _mul_rn,
    _policy_evict_first,
    _policy_evict_last,
    _popc,
    _ptr,
    _rcp,
    _red_release_add,
    _st_global_b32,
    _st_global_b32_hint,
    _st_relaxed_u32,
)
from sgl_nvidia_mega.qwen3_5.sm103.moe.router_ptx import _add
from sgl_nvidia_mega.qwen3_5.sm103.moe.router_ptx_a import (
    _bit,
    _key_val,
    _ld_v4,
    _red_or_shared,
    _sel_eq,
    _sel_lt,
    _st_b16,
    _st_b32,
)

TILE_M = 128
TILE_K = 128
KB = 64
NUM_ACC = 2
EPI_WARP0 = 4
E_ALL = 512  # router experts
TOPK = 10
M_MAX = 32  # rows routed in the prologue (one token bit per row)
M_MAX_WIDE = 128
NLOC = 128  # local experts
SEG_CAP = M_MAX * TOPK  # segments per call: every segment holds >= 1 local pick
CNT_CAP = 512  # counters per parity (>= seg_cap of every build)
WIDE_OFF = 16 + 2 * CNT_CAP
WIDE_PAR = 4 + NLOC * (M_MAX_WIDE // 32) + M_MAX_WIDE * TOPK
QD = 8  # unit queue slots
QL = 2  # claimed entries the scheduler may hold ahead of the weight loader
EPI_BAR = 1
LOG2E = 1.4426950216293334961


def _red_or_global(addr: Int64, v) -> None:
    cute.arch.inline_ptx(
        "red.relaxed.gpu.global.or.b32 [{$r0}], {$r1};", read_only_args=[addr, v]
    )


def _sub_rn_ftz(a, b):
    return cute.arch.inline_ptx(
        "sub.rn.ftz.f32 {$w0}, {$r0}, {$r1};",
        write_only_types=[cutlass.Float32],
        read_only_args=[a, b],
    )


def _mul_rn_ftz(a, b):
    return cute.arch.inline_ptx(
        "mul.rn.ftz.f32 {$w0}, {$r0}, {$r1};",
        write_only_types=[cutlass.Float32],
        read_only_args=[a, b],
    )


def _pack_bf16x2(lo, hi):
    """{bf16(hi), bf16(lo)} by cvt.rn.satfinite.bf16x2.f32 (lo in the low half)."""
    return cute.arch.inline_ptx(
        "cvt.rn.satfinite.bf16x2.f32 {$w0}, {$r1}, {$r0};",
        write_only_types=[Int32],
        read_only_args=[lo, hi],
    )


def _pack_bf16x2_rn(lo, hi):
    """{bf16(hi), bf16(lo)} by cvt.rn.bf16x2.f32 (no saturation)."""
    return cute.arch.inline_ptx(
        "cvt.rn.bf16x2.f32 {$w0}, {$r1}, {$r0};",
        write_only_types=[Int32],
        read_only_args=[lo, hi],
    )


def seg_cap_for(m_max: int, n_max: int) -> int:
    if m_max == M_MAX:
        return SEG_CAP
    return min(m_max * TOPK, NLOC + (m_max * TOPK) // n_max)


def smem_views(storage, seg_cap: int = SEG_CAP, mw: int = 1, m_max: int = M_MAX):

    def view(mr, n):
        return mr.get_tensor(cute.make_layout(n))

    return (
        view(storage.nseg, 4),
        view(storage.seg_e, seg_cap),
        view(storage.seg_r0, seg_cap),
        view(storage.seg_n, seg_cap),
        view(storage.mask, NLOC * mw),
        view(storage.eoff, NLOC + 1),
        view(storage.pick, m_max * TOPK),
        view(storage.p2t, m_max * TOPK),
    )


class Q35DraftMoeEngine:
    def __init__(
        self,
        hidden: int,
        inter: int,
        n_max: int,
        num_stages: int,
        use_pdl: bool = True,
        tile_k: int = TILE_K,
        epi_groups: int = 2,
        num_acc: int = NUM_ACC,
        fc2_chunk: int = 1,
        m_max: int = M_MAX,
    ):
        assert tile_k in (64, 128, 256)
        assert m_max in (M_MAX, M_MAX_WIDE)
        self.m_max = m_max
        self.mw = m_max // 32
        self.tile_k = tile_k
        assert (
            epi_groups in (1, 2)
            and num_acc >= 2
            and (epi_groups == 1 or num_acc % 2 == 0)
        )
        self.epi_groups = epi_groups
        self.num_acc = num_acc
        self.threads = 128 + 128 * epi_groups
        assert fc2_chunk in (1, 2, 4)
        self.fc2_chunk = fc2_chunk
        assert hidden % KB == 0
        assert (
            hidden % tile_k == 0
            and inter % tile_k == 0
            and (2 * inter) % TILE_M == 0
            and hidden % TILE_M == 0
        )
        assert n_max in (16, 32, 64, 128)
        self.hidden, self.inter = hidden, inter
        self.k1_tiles, self.k2_tiles = hidden // tile_k, inter // tile_k
        self.m1_tiles, self.m2_tiles = (2 * inter) // TILE_M, hidden // TILE_M
        self.n_max, self.num_stages = n_max, num_stages
        self.seg_cap = seg_cap_for(m_max, n_max)
        assert self.seg_cap <= CNT_CAP
        self.use_pdl = use_pdl
        self.cta_group = tcgen05.CtaGroup.ONE
        self.acc_dtype = cutlass.Float32
        self.b_stage_bytes = n_max * tile_k * 2
        self.kchunks = tile_k * 2 // 16

    def _setup(self):
        self.tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            (TILE_M, self.n_max),
        )
        total = self.num_acc * self.n_max
        cols = 32
        while cols < total:
            cols *= 2
        self.num_tmem_cols = cols
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (self.tiled_mma.thr_id.shape,)
        )
        tiler = (TILE_M, self.n_max, self.tile_k)
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            self.tiled_mma, tiler, self.a_dtype, self.num_stages
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            self.tiled_mma, tiler, self.b_dtype, self.num_stages
        )

    def _a_atoms(self, w_ptr, rows, k, e):
        w = cute.make_tensor(
            w_ptr,
            cute.make_layout(
                (rows, (KB, k // KB), e), stride=(KB, (1, rows * KB), rows * k)
            ),
        )
        a_op = sm100_utils.cluster_shape_to_tma_atom_A((1, 1), self.tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        atom, tens = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            w,
            a_smem_layout,
            (TILE_M, self.n_max, self.tile_k),
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        self.tma_bytes_a = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        return atom, tens

    @cute.jit
    def __call__(
        self,
        w1_ptr: cute.Pointer,
        w2_ptr: cute.Pointer,
        b_dummy_ptr: cute.Pointer,
        x_addr: Int64,
        x_row_bytes: Int64,
        lg_addr: Int64,
        m: Int32,
        local_start: Int32,
        wts_addr: Int64,
        e2p_addr: Int64,
        h_addr: Int64,
        out_addr: Int64,
        state_addr: Int64,
        num_experts: Int32,
        grid_ctas: Int32,
        stream: cuda.CUstream,
    ):
        e = num_experts
        self.a_dtype = w1_ptr.value_type
        self.b_dtype = b_dummy_ptr.value_type
        w_probe = cute.make_tensor(
            w1_ptr,
            cute.make_ordered_layout((2 * self.inter, self.hidden, e), order=(1, 0, 2)),
        )
        b_probe = cute.make_tensor(
            b_dummy_ptr,
            cute.make_ordered_layout((self.n_max, self.tile_k, 1), order=(1, 0, 2)),
        )
        self.a_major_mode = cutlass.utils.LayoutEnum.from_tensor(
            w_probe
        ).mma_major_mode()
        self.b_major_mode = cutlass.utils.LayoutEnum.from_tensor(
            b_probe
        ).mma_major_mode()
        self._setup()
        a1, t1 = self._a_atoms(w1_ptr, 2 * self.inter, self.hidden, e)
        a2, t2 = self._a_atoms(w2_ptr, self.hidden, self.inter, e)
        self.kernel(
            self.tiled_mma,
            a1,
            t1,
            a2,
            t2,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            x_addr,
            x_row_bytes,
            lg_addr,
            m,
            local_start,
            wts_addr,
            e2p_addr,
            h_addr,
            out_addr,
            state_addr,
            num_experts,
        ).launch(
            grid=[grid_ctas, 1, 1],
            block=[self.threads, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
            use_pdl=self.use_pdl,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_a1: cute.CopyAtom,
        mA1: cute.Tensor,
        tma_a2: cute.CopyAtom,
        mA2: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        x_addr: Int64,
        x_row_bytes: Int64,
        lg_addr: Int64,
        m: Int32,
        local_start: Int32,
        wts_addr: Int64,
        e2p_addr: Int64,
        h_addr: Int64,
        out_addr: Int64,
        state_addr: Int64,
        n_exp: Int32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdx, _, _ = cute.arch.grid_dim()
        lane = cute.arch.lane_idx()
        Stages = self.num_stages
        NA = self.num_acc
        NT = self.threads
        SegCap = self.seg_cap
        MW = self.mw
        MMax = self.m_max

        @cute.struct
        class SharedStorage:
            ab_bars: cute.struct.MemRange[cutlass.Int64, Stages * 2]
            acc_bars: cute.struct.MemRange[cutlass.Int64, NA * 2]
            tmem_bar: cutlass.Int64
            tmem_base: cutlass.Int32
            nseg: cute.struct.MemRange[cutlass.Int32, 4]
            seg_e: cute.struct.MemRange[cutlass.Int32, SegCap]
            seg_r0: cute.struct.MemRange[cutlass.Int32, SegCap]
            seg_n: cute.struct.MemRange[cutlass.Int32, SegCap]
            mask: cute.struct.MemRange[cutlass.Int32, NLOC * MW]
            eoff: cute.struct.MemRange[cutlass.Int32, NLOC + 1]
            pick: cute.struct.MemRange[cutlass.Int32, MMax * TOPK]
            p2t: cute.struct.MemRange[cutlass.Int32, MMax * TOPK]
            uq: cute.struct.MemRange[cutlass.Int32, QD]
            q_bars: cute.struct.MemRange[cutlass.Int64, 3 * QD]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        s_nseg, s_seg_e, s_seg_r0, s_seg_n, s_mask, s_eoff, s_pick, s_p2t = smem_views(
            storage, SegCap, MW, MMax
        )
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_a1)
            cpasync.prefetch_descriptor(tma_a2)
        ab_pipeline = PipelineAsync.create(
            barrier_storage=storage.ab_bars.data_ptr(),
            num_stages=Stages,
            producer_group=CooperativeGroup(Agent.Thread, 1 + 32),
            consumer_group=CooperativeGroup(Agent.Thread),
            defer_sync=True,
        )
        ab_full = ab_pipeline.sync_object_full.barrier_storage
        ab_empty = ab_pipeline.sync_object_empty.barrier_storage
        acc_pipeline = PipelineUmmaAsync.create(
            barrier_storage=storage.acc_bars.data_ptr(),
            num_stages=NA,
            producer_group=CooperativeGroup(Agent.Thread),
            consumer_group=CooperativeGroup(Agent.Thread, 128),
            defer_sync=True,
        )
        acc_full = acc_pipeline.sync_object_full.barrier_storage
        acc_empty = acc_pipeline.sync_object_empty.barrier_storage
        tmem_bar = storage.tmem_bar.ptr
        tmem_base_smem_ptr = storage.tmem_base.ptr
        s_q = storage.uq.get_tensor(cute.make_layout(QD))
        q_full = storage.q_bars.data_ptr()
        q_empty = q_full + QD
        q_taken = q_full + 2 * QD
        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(tmem_bar, 32 + 128 * self.epi_groups)
                for i in cutlass.range_constexpr(QD):
                    cute.arch.mbarrier_init(q_full + i, 1)
                    cute.arch.mbarrier_init(q_empty + i, 1)
                    cute.arch.mbarrier_init(q_taken + i, 1)
            cute.arch.mbarrier_init_fence()
        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=1024,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=1024,
            swizzle=b_smem_layout_staged.inner,
        )
        sb_base = sB.iterator.toint()
        pipeline_init_arrive(cluster_shape_mn=None, is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=None)

        for i in cutlass.range(tidx, NLOC * MW, NT, unroll=1):
            s_mask[i] = Int32(0)
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
        gen = _ld_relaxed_u32(state_addr)
        cute.arch.barrier()  # the masks are zero before any warp sets bits in them
        if cutlass.const_expr(self.mw == 1):
            self.route(
                lg_addr, m, local_start, wts_addr, s_mask, s_pick, warp_idx, lane, bidx
            )
        else:
            self.route_wide(
                lg_addr,
                m,
                local_start,
                wts_addr,
                s_mask,
                s_pick,
                warp_idx,
                lane,
                bidx,
                gdx,
                tidx,
                state_addr,
            )
        cute.arch.barrier()
        if warp_idx == 0:
            self.segments(s_mask, s_eoff, s_seg_e, s_seg_r0, s_seg_n, s_nseg, lane)
        cute.arch.barrier()
        self.rows(m, s_mask, s_eoff, s_pick, s_p2t, e2p_addr, tidx, bidx)
        cute.arch.barrier()
        nseg = s_nseg[0]
        par = gen & Int32(1)
        cnt_base = state_addr + Int64(64) + Int64(par) * Int64(CNT_CAP * 4)
        nxt_base = state_addr + Int64(64) + Int64(1 - par) * Int64(CNT_CAP * 4)
        for i in cutlass.range(bidx * NT + tidx, CNT_CAP, gdx * NT, unroll=1):
            _st_relaxed_u32(nxt_base + Int64(i) * 4, Int32(0))
        pool_addr = (
            state_addr + Int64(8) + Int64(par) * 4
        )  # FC2 ticket counter of this call
        if bidx == 0:
            if tidx == 0:
                _st_relaxed_u32(state_addr + Int64(8) + Int64(1 - par) * 4, Int32(0))
                _st_relaxed_u32(state_addr + Int64(16) + Int64(1 - par) * 4, Int32(0))

        m1 = self.m1_tiles
        m2 = self.m2_tiles
        u1 = nseg * m1
        u2 = nseg * m2
        n1 = Int32(0)
        if bidx < u1:
            n1 = (u1 - 1 - bidx) // gdx + 1

        if warp_idx == 0:
            self.load_a(
                ab_full,
                ab_empty,
                tma_a1,
                mA1,
                tma_a2,
                mA2,
                sA,
                tiled_mma,
                cluster_layout_vmnk,
                s_seg_e,
                s_q,
                q_full,
                q_taken,
                u1,
                bidx,
            )
        elif warp_idx == 1:
            self.load_b(
                ab_full,
                ab_empty,
                s_seg_r0,
                s_seg_n,
                s_p2t,
                sb_base,
                x_addr,
                x_row_bytes,
                h_addr,
                s_q,
                q_full,
                u1,
                lane,
                bidx,
                cnt_base,
            )
        elif warp_idx == 2:
            self.mma(
                ab_full,
                ab_empty,
                acc_full,
                acc_empty,
                tmem_bar,
                tmem_base_smem_ptr,
                tiled_mma,
                sA,
                sB,
                s_q,
                q_full,
                u1,
            )
        elif warp_idx == 3:
            self.scheduler(
                s_q,
                q_full,
                q_empty,
                q_taken,
                n1,
                u1,
                u2,
                bidx,
                gdx,
                pool_addr,
                cnt_base,
                lane,
            )
        elif warp_idx >= EPI_WARP0:
            self.epilogue(
                acc_full,
                acc_empty,
                tmem_bar,
                tmem_base_smem_ptr,
                tiled_mma,
                s_seg_e,
                s_seg_r0,
                s_seg_n,
                h_addr,
                out_addr,
                cnt_base,
                (tidx - EPI_WARP0 * 32) % 128,
                (tidx - EPI_WARP0 * 32) // 128,
                s_q,
                q_full,
                q_empty,
                u1,
                bidx,
                gdx,
            )
        cute.arch.barrier()
        if warp_idx == 2:
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype, 16, tmem_base_smem_ptr
            )
            cute.arch.dealloc_tmem(tmem_ptr, self.num_tmem_cols)
        if tidx == 0:
            old = _atom_add_acq_rel(state_addr + Int64(4), Int32(1))
            if old == gdx - 1:
                _st_relaxed_u32(state_addr + Int64(4), Int32(0))
                _st_relaxed_u32(state_addr, gen + Int32(1))
                if cutlass.const_expr(self.mw > 1):
                    wg = _ld_relaxed_u32(state_addr + Int64(WIDE_OFF * 4))
                    _st_relaxed_u32(state_addr + Int64(WIDE_OFF * 4), wg + Int32(1))
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()
        return

    @cute.jit
    def route(
        self, lg_addr, m, local_start, wts_addr, s_mask, s_pick, warp_idx, lane, bidx
    ):
        mask_addr = s_mask.iterator.toint()
        for row in cutlass.range(warp_idx, m, self.threads // 32, unroll=1):
            src = lg_addr + Int64(row) * Int64(E_ALL * 2) + Int64(lane) * Int64(32)
            a0, a1, a2, a3 = _ld_v4(src)
            b0, b1, b2, b3 = _ld_v4(src + Int64(16))
            ws = (a0, a1, a2, a3, b0, b1, b2, b3)
            e0 = lane * 16
            keys = cute.make_rmem_tensor(cute.make_layout((16,)), Int32)
            for j in cutlass.range_constexpr(8):
                keys[2 * j] = _key1(ws[j], e0 + 2 * j)
                keys[2 * j + 1] = _key1(ws[j] >> 16, e0 + 2 * j + 1)
            for pi in cutlass.range_constexpr(len(_SORT16)):
                i, jj, imax = _SORT16[pi]
                x = keys[i]
                y = keys[jj]
                hi = _max_u32(x, y)
                lo = _min_u32(x, y)
                if cutlass.const_expr(imax):
                    keys[i] = hi
                    keys[jj] = lo
                else:
                    keys[i] = lo
                    keys[jj] = hi
            sel = cute.make_rmem_tensor(cute.make_layout((TOPK,)), Int32)
            val = cute.make_rmem_tensor(cute.make_layout((TOPK,)), cutlass.Float32)
            for k in cutlass.range_constexpr(TOPK):
                head = keys[0]
                g = _redux_max_u32(head)
                sel[k] = Int32(65535) - (g & Int32(65535))
                val[k] = _key_val(g)
                if cutlass.const_expr(k < TOPK - 1):
                    for j in cutlass.range_constexpr(TOPK - 1):
                        keys[j] = _sel_eq(head, g, keys[j + 1], keys[j])
            vmax = val[0]
            ex = cute.make_rmem_tensor(cute.make_layout((TOPK,)), cutlass.Float32)
            for k in cutlass.range_constexpr(TOPK):
                ex[k] = _ex2(
                    _mul_rn_ftz(_sub_rn_ftz(val[k], vmax), cutlass.Float32(LOG2E))
                )
            ssum = _add(
                _add(_add(_add(ex[0], ex[8]), ex[4]), _add(ex[2], ex[6])),
                _add(_add(_add(ex[1], ex[9]), ex[5]), _add(ex[3], ex[7])),
            )
            my_sel = sel[0]
            my_e = ex[0]
            for k in cutlass.range_constexpr(TOPK - 1):
                my_sel = _sel_eq(lane, Int32(k + 1), sel[k + 1], my_sel)
                my_e = _sel_eq_f(lane, Int32(k + 1), ex[k + 1], my_e)
            if lane < TOPK:
                o = Int64(row) * Int64(TOPK) + Int64(lane)
                if bidx == 0:
                    q = _mul_rn_ftz(my_e, _rcp(ssum))
                    _st_b16(
                        wts_addr + o * Int64(2),
                        _pack_bf16x2_rn(q, q) & Int32(65535),
                    )
                lid = my_sel - local_start
                pick = _sel_lt(lid, Int32(NLOC), lid, Int32(-1))
                if pick >= Int32(0):
                    _red_or_shared(mask_addr + pick * 4, _bit(row))
                s_pick[row * TOPK + lane] = pick

    @cute.jit
    def _route_row_wide(
        self,
        lg_addr,
        row,
        local_start,
        wts_addr,
        lane,
        gmask,
        gpick,
    ):
        src = lg_addr + Int64(row) * Int64(E_ALL * 2) + Int64(lane) * Int64(32)
        a0, a1, a2, a3 = _ld_v4(src)
        b0, b1, b2, b3 = _ld_v4(src + Int64(16))
        ws = (a0, a1, a2, a3, b0, b1, b2, b3)
        e0 = lane * 16
        keys = cute.make_rmem_tensor(cute.make_layout((16,)), Int32)
        for j in cutlass.range_constexpr(8):
            keys[2 * j] = _key1(ws[j], e0 + 2 * j)
            keys[2 * j + 1] = _key1(ws[j] >> 16, e0 + 2 * j + 1)
        for pi in cutlass.range_constexpr(len(_SORT16)):
            i, jj, imax = _SORT16[pi]
            x = keys[i]
            y = keys[jj]
            hi = _max_u32(x, y)
            lo = _min_u32(x, y)
            if cutlass.const_expr(imax):
                keys[i] = hi
                keys[jj] = lo
            else:
                keys[i] = lo
                keys[jj] = hi
        sel = cute.make_rmem_tensor(cute.make_layout((TOPK,)), Int32)
        val = cute.make_rmem_tensor(cute.make_layout((TOPK,)), cutlass.Float32)
        for k in cutlass.range_constexpr(TOPK):
            head = keys[0]
            g = _redux_max_u32(head)
            sel[k] = Int32(65535) - (g & Int32(65535))
            val[k] = _key_val(g)
            if cutlass.const_expr(k < TOPK - 1):
                for j in cutlass.range_constexpr(TOPK - 1):
                    keys[j] = _sel_eq(head, g, keys[j + 1], keys[j])
        vmax = val[0]
        ex = cute.make_rmem_tensor(cute.make_layout((TOPK,)), cutlass.Float32)
        for k in cutlass.range_constexpr(TOPK):
            ex[k] = _ex2(_mul_rn_ftz(_sub_rn_ftz(val[k], vmax), cutlass.Float32(LOG2E)))
        ssum = _add(
            _add(_add(_add(ex[0], ex[8]), ex[4]), _add(ex[2], ex[6])),
            _add(_add(_add(ex[1], ex[9]), ex[5]), _add(ex[3], ex[7])),
        )
        my_sel = sel[0]
        my_e = ex[0]
        for k in cutlass.range_constexpr(TOPK - 1):
            my_sel = _sel_eq(lane, Int32(k + 1), sel[k + 1], my_sel)
            my_e = _sel_eq_f(lane, Int32(k + 1), ex[k + 1], my_e)
        if lane < TOPK:
            o = Int64(row) * Int64(TOPK) + Int64(lane)
            qw = _mul_rn_ftz(my_e, _rcp(ssum))
            _st_b16(wts_addr + o * Int64(2), _pack_bf16x2_rn(qw, qw) & Int32(65535))
            lid = my_sel - local_start
            pick = _sel_lt(lid, Int32(NLOC), lid, Int32(-1))
            word = pick * self.mw + (row >> 5)
            if pick >= Int32(0):
                _red_or_global(gmask + Int64(word) * 4, _bit(row & Int32(31)))
            _st_global_b32(gpick + o * Int64(4), pick)

    @cute.jit
    def route_wide(
        self,
        lg_addr,
        m,
        local_start,
        wts_addr,
        s_mask,
        s_pick,
        warp_idx,
        lane,
        bidx,
        gdx,
        tidx,
        state_addr,
    ):
        NW: cutlass.Constexpr = self.threads // 32
        wgen = _ld_relaxed_u32(state_addr + Int64(WIDE_OFF * 4))
        wpar = wgen & Int32(1)
        reg = state_addr + Int64((WIDE_OFF + 4) * 4) + Int64(wpar) * Int64(WIDE_PAR * 4)
        nxt = (
            state_addr
            + Int64((WIDE_OFF + 4) * 4)
            + Int64(1 - wpar) * Int64(WIDE_PAR * 4)
        )
        gmask = reg + Int64(16)
        gpick = gmask + Int64(NLOC * self.mw * 4)
        for row in cutlass.range(bidx + gdx * warp_idx, m, gdx * NW, unroll=1):
            self._route_row_wide(
                lg_addr,
                row,
                local_start,
                wts_addr,
                lane,
                gmask,
                gpick,
            )
        for i in cutlass.range(
            bidx * self.threads + tidx,
            4 + NLOC * self.mw,
            gdx * self.threads,
            unroll=1,
        ):
            _st_relaxed_u32(nxt + Int64(i) * 4, Int32(0))
        cute.arch.barrier()
        if tidx == 0:
            _fence_acq_rel_gpu()
            _red_release_add(reg, Int32(1))
            ready = Int32(0)
            while ready == 0:
                if _ld_acquire_u32(reg) >= gdx:
                    ready = Int32(1)
            _fence_acq_rel_gpu()
        cute.arch.barrier()
        for i in cutlass.range(tidx, NLOC * self.mw, self.threads, unroll=1):
            s_mask[i] = _ld_relaxed_u32(gmask + Int64(i) * 4)
        for i in cutlass.range(tidx, m * TOPK, self.threads, unroll=1):
            s_pick[i] = _ld_relaxed_u32(gpick + Int64(i) * 4)

    @cute.jit
    def segments(self, s_mask, s_eoff, s_seg_e, s_seg_r0, s_seg_n, s_nseg, lane):
        cnt = cute.make_rmem_tensor(cute.make_layout((NLOC // 32,)), Int32)
        nsg = cute.make_rmem_tensor(cute.make_layout((NLOC // 32,)), Int32)
        rows_l = Int32(0)
        segs_l = Int32(0)
        for q in cutlass.range_constexpr(NLOC // 32):
            if cutlass.const_expr(self.mw == 1):
                c = _popc(s_mask[lane * (NLOC // 32) + q])
            else:
                c = Int32(0)
                for w in cutlass.range_constexpr(self.mw):
                    c = c + _popc(s_mask[(lane * (NLOC // 32) + q) * self.mw + w])
            ns = (c + (self.n_max - 1)) // self.n_max
            cnt[q] = c
            nsg[q] = ns
            rows_l = rows_l + c
            segs_l = segs_l + ns
        ri = rows_l
        si = segs_l
        for off in (1, 2, 4, 8, 16):
            t1 = cute.arch.shuffle_sync_up(ri, off)
            t2 = cute.arch.shuffle_sync_up(si, off)
            if lane >= off:
                ri = ri + t1
                si = si + t2
        roff = ri - rows_l
        soff = si - segs_l
        for q in cutlass.range_constexpr(NLOC // 32):
            e = lane * (NLOC // 32) + q
            s_eoff[e] = roff
            for j in cutlass.range(nsg[q], unroll=1):
                s_seg_e[soff] = e
                s_seg_r0[soff] = roff + j * self.n_max
                s_seg_n[soff] = cutlass.min(cnt[q] - j * self.n_max, self.n_max)
                soff = soff + 1
            roff = roff + cnt[q]
        if lane == 31:
            s_nseg[0] = si
            s_eoff[NLOC] = ri

    @cute.jit
    def rows(self, m, s_mask, s_eoff, s_pick, s_p2t, e2p_addr, tidx, bidx):
        if cutlass.const_expr(self.mw == 1):
            for e in cutlass.range(tidx, NLOC, self.threads, unroll=1):
                mk = s_mask[e]
                r = s_eoff[e]
                while mk != Int32(0):
                    low = mk & (Int32(0) - mk)
                    s_p2t[r] = _popc(low - Int32(1))
                    r = r + 1
                    mk = mk ^ low
            if bidx == 0:
                for i in cutlass.range(tidx, m * TOPK, self.threads, unroll=1):
                    lid = s_pick[i]
                    v = Int32(-1)
                    if lid >= Int32(0):
                        v = s_eoff[lid] + _popc(s_mask[lid] & _below(i // TOPK))
                    _st_b32(e2p_addr + Int64(i) * Int64(4), v)
        else:
            for e in cutlass.range(tidx, NLOC, self.threads, unroll=1):
                r = s_eoff[e]
                for w in cutlass.range_constexpr(self.mw):
                    mk = s_mask[e * self.mw + w]
                    while mk != Int32(0):
                        low = mk & (Int32(0) - mk)
                        s_p2t[r] = _popc(low - Int32(1)) + Int32(32 * w)
                        r = r + 1
                        mk = mk ^ low
            if bidx == 0:
                for i in cutlass.range(tidx, m * TOPK, self.threads, unroll=1):
                    lid = s_pick[i]
                    v = Int32(-1)
                    if lid >= Int32(0):
                        t = i // TOPK
                        tw = t >> 5
                        v = s_eoff[lid]
                        bl = _below(t & Int32(31))
                        for w in cutlass.range_constexpr(self.mw):
                            mk = s_mask[lid * self.mw + w]
                            mk = _sel_lt(
                                Int32(w),
                                tw,
                                mk,
                                _sel_eq(Int32(w), tw, mk & bl, Int32(0)),
                            )
                            v = v + _popc(mk)
                    _st_b32(e2p_addr + Int64(i) * Int64(4), v)

    @cute.jit
    def decode(self, g, u1):
        is1 = Int32(0)
        seg = Int32(0)
        mt = Int32(0)
        if g < u1:
            is1 = Int32(1)
            seg = g // self.m1_tiles
            mt = g % self.m1_tiles
        else:
            seg = (g - u1) // self.m2_tiles
            mt = (g - u1) % self.m2_tiles
        return is1, seg, mt

    @cute.jit
    def _push(self, s_q, q_full, q_empty, q_taken, k, g):
        slot = k % QD
        if k >= QL:
            kk = k - QL
            cute.arch.mbarrier_wait(q_taken + (kk % QD), (kk // QD) & 1)
        if k >= QD:
            cute.arch.mbarrier_wait(q_empty + slot, ((k // QD) - 1) & 1)
        s_q[slot] = g
        cute.arch.mbarrier_arrive(q_full + slot)

    @cute.jit
    def scheduler(
        self,
        s_q,
        q_full,
        q_empty,
        q_taken,
        n1,
        u1,
        u2,
        bidx,
        gdx,
        pool_addr,
        cnt_base,
        lane,
    ):
        if lane == 0:
            k = Int32(0)
            for j in cutlass.range(n1, unroll=1):
                self._push(s_q, q_full, q_empty, q_taken, k, bidx + j * gdx)
                k = k + 1
            t = u2
            if u2 > 0:
                t = _atom_add_relaxed(pool_addr, Int32(self.fc2_chunk))
            while t < u2:
                tn = _atom_add_relaxed(pool_addr, Int32(self.fc2_chunk))
                for c in cutlass.range_constexpr(self.fc2_chunk):
                    self._push(s_q, q_full, q_empty, q_taken, k, u1 + t + c)
                    k = k + 1
                t = tn
            self._push(s_q, q_full, q_empty, q_taken, k, Int32(-1))
            if cutlass.const_expr(self.epi_groups == 2):
                self._push(s_q, q_full, q_empty, q_taken, k + 1, Int32(-1))

    @cute.jit
    def load_a(
        self,
        ab_full,
        ab_empty,
        tma_a1: cute.CopyAtom,
        mA1: cute.Tensor,
        tma_a2: cute.CopyAtom,
        mA2: cute.Tensor,
        sA: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cluster_layout_vmnk: cute.Layout,
        s_seg_e: cute.Tensor,
        s_q: cute.Tensor,
        q_full,
        q_taken,
        u1,
        bidx,
    ):
        """Weights of every unit, back to back through the ring (constant data)."""
        thr_mma = tiled_mma.get_slice(0)
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        g1 = cute.local_tile(mA1, (TILE_M, self.tile_k), (None, None, None))
        t1s, t1g = cpasync.tma_partition(
            tma_a1,
            0,
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(thr_mma.partition_A(g1), 0, 3),
        )
        g2 = cute.local_tile(mA2, (TILE_M, self.tile_k), (None, None, None))
        t2s, t2g = cpasync.tma_partition(
            tma_a2,
            0,
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(thr_mma.partition_A(g2), 0, 3),
        )
        st = make_pipeline_state(PipelineUserType.Producer, self.num_stages)
        wp = _policy_evict_first()
        k = Int32(0)
        g = Int32(-1)
        if bidx < u1:
            g = bidx  # the static first FC1 unit (the scheduler pushes the same entry)
        else:
            cute.arch.mbarrier_wait(q_full, 0)
            g = s_q[0]
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(q_taken)
        while g >= 0:
            is1, seg, mt = self.decode(g, u1)
            ex = s_seg_e[seg]
            if is1 != 0:
                for kt in cutlass.range(self.k1_tiles, unroll=1):
                    cute.arch.mbarrier_wait(ab_empty + st.index, st.phase)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            ab_full + st.index, self.tma_bytes_a
                        )
                    cute.copy(
                        tma_a1,
                        t1g[(None, mt, kt, ex)],
                        t1s[(None, st.index)],
                        tma_bar_ptr=ab_full + st.index,
                        cache_policy=wp,
                    )
                    st.advance()
            else:
                for kt in cutlass.range(self.k2_tiles, unroll=1):
                    cute.arch.mbarrier_wait(ab_empty + st.index, st.phase)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            ab_full + st.index, self.tma_bytes_a
                        )
                    cute.copy(
                        tma_a2,
                        t2g[(None, mt, kt, ex)],
                        t2s[(None, st.index)],
                        tma_bar_ptr=ab_full + st.index,
                        cache_policy=wp,
                    )
                    st.advance()
            k = k + 1
            slot = k % QD
            cute.arch.mbarrier_wait(q_full + slot, (k // QD) & 1)
            g = s_q[slot]
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(q_taken + slot)

    @cute.jit
    def _b_stage(
        self, is1, kt, rows, r0, sb_stage, x_addr, x_row_bytes, h_addr, s_p2t, lane
    ):
        kc = self.kchunks
        rb = (self.n_max // 8) * 1024
        if is1 != 0:
            for n in cutlass.range(lane, rows, 32, unroll=1):
                t = s_p2t[r0 + n]
                src = x_addr + Int64(t) * x_row_bytes + Int64(kt * (kc * 16))
                dbase = sb_stage + (n >> 3) * 1024 + (n & 7) * 128
                n7 = n & 7
                for c in cutlass.range_constexpr(kc):
                    _cp_async_16(
                        dbase + ((c >> 3) * rb) + (((c & 7) ^ n7) << 4),
                        src + Int64(c * 16),
                    )
        else:
            for n in cutlass.range(lane, rows, 32, unroll=1):
                r = r0 + n
                src = h_addr + Int64(r) * Int64(self.inter * 2) + Int64(kt * (kc * 16))
                dbase = sb_stage + (n >> 3) * 1024 + (n & 7) * 128
                n7 = n & 7
                for c in cutlass.range_constexpr(kc):
                    _cp_async_16(
                        dbase + ((c >> 3) * rb) + (((c & 7) ^ n7) << 4),
                        src + Int64(c * 16),
                    )
        _cp_async_commit()

    @cute.jit
    def load_b(
        self,
        ab_full,
        ab_empty,
        s_seg_r0,
        s_seg_n,
        s_p2t,
        sb_base,
        x_addr,
        x_row_bytes,
        h_addr,
        s_q,
        q_full,
        u1,
        lane,
        bidx,
        cnt_base,
    ):
        st = make_pipeline_state(PipelineUserType.Producer, self.num_stages)
        k = Int32(0)
        g = Int32(-1)
        if bidx < u1:
            g = bidx
        else:
            cute.arch.mbarrier_wait(q_full, 0)
            g = s_q[0]
        while g >= 0:
            is1, seg, mt = self.decode(g, u1)
            rows = s_seg_n[seg]
            r0 = s_seg_r0[seg]
            kts = self.k2_tiles
            if is1 != 0:
                kts = self.k1_tiles
            else:
                ready = Int32(0)
                while ready == 0:
                    if _ld_acquire_u32(cnt_base + Int64(seg) * 4) >= self.m1_tiles:
                        ready = Int32(1)
            for kt in cutlass.range(kts, unroll=1):
                cute.arch.mbarrier_wait(ab_empty + st.index, st.phase)
                self._b_stage(
                    is1,
                    kt,
                    rows,
                    r0,
                    sb_base + st.index * self.b_stage_bytes,
                    x_addr,
                    x_row_bytes,
                    h_addr,
                    s_p2t,
                    lane,
                )
                _cp_async_arrive_noinc((ab_full + st.index).toint())
                st.advance()
            k = k + 1
            slot = k % QD
            cute.arch.mbarrier_wait(q_full + slot, (k // QD) & 1)
            g = s_q[slot]

    @cute.jit
    def mma(
        self,
        ab_full,
        ab_empty,
        acc_full,
        acc_empty,
        tmem_bar,
        tmem_base_smem_ptr,
        tiled_mma: cute.TiledMma,
        sA: cute.Tensor,
        sB: cute.Tensor,
        s_q: cute.Tensor,
        q_full,
        u1,
    ):
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C((TILE_M, self.n_max))
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        cute.arch.alloc_tmem(self.num_tmem_cols, tmem_base_smem_ptr)
        cute.arch.mbarrier_arrive(tmem_bar)
        cute.arch.relinquish_tmem_alloc_permit()
        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_smem_ptr)
        num_kphases = cute.size(tCrA, mode=[2])
        ab_st = make_pipeline_state(PipelineUserType.Consumer, self.num_stages)
        acc_st = make_pipeline_state(PipelineUserType.Producer, self.num_acc)
        k = Int32(0)
        cute.arch.mbarrier_wait(q_full, 0)
        g = s_q[0]
        while g >= 0:
            kts = self.k2_tiles
            if g < u1:
                kts = self.k1_tiles
            cute.arch.mbarrier_wait(acc_empty + acc_st.index, acc_st.phase)
            tCtAcc = cute.make_tensor(
                tmem_ptr + acc_st.index * self.n_max, tCtAcc_fake.layout
            )
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for _kt in cutlass.range(kts, unroll=1):
                cute.arch.mbarrier_wait(ab_full + ab_st.index, ab_st.phase)
                for kp in cutlass.range(num_kphases, unroll_full=True):
                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        tCrA[(None, None, kp, ab_st.index)],
                        tCrB[(None, None, kp, ab_st.index)],
                        tCtAcc,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(ab_empty + ab_st.index, None, self.cta_group)
                ab_st.advance()
            with cute.arch.elect_one():
                tcgen05.commit(acc_full + acc_st.index, None, self.cta_group)
            acc_st.advance()
            k = k + 1
            slot = k % QD
            cute.arch.mbarrier_wait(q_full + slot, (k // QD) & 1)
            g = s_q[slot]

    @cute.jit
    def _swiglu(self, up, gate):
        sgm = _rcp(
            _add_rn(_ex2(-_mul_rn(gate, cutlass.Float32(LOG2E))), cutlass.Float32(1.0))
        )
        return _mul_rn(up, _mul_rn(gate, sgm))

    @cute.jit
    def epilogue(
        self,
        acc_full,
        acc_empty,
        tmem_bar,
        tmem_base_smem_ptr,
        tiled_mma: cute.TiledMma,
        s_seg_e,
        s_seg_r0,
        s_seg_n,
        h_addr,
        out_addr,
        cnt_base,
        epi_tid,
        grp,
        s_q: cute.Tensor,
        q_full,
        q_empty,
        u1,
        bidx,
        gdx,
    ):
        thr_mma = tiled_mma.get_slice(0)
        acc_shape = tiled_mma.partition_shape_C((TILE_M, self.n_max))
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        tCcL = thr_mma.partition_C(cute.make_identity_tensor((TILE_M, self.n_max)))
        cute.arch.mbarrier_arrive(tmem_bar)
        cute.arch.mbarrier_wait(tmem_bar, 0)
        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_smem_ptr)
        epi_tile = (TILE_M, 8)
        copy_atom_t2r = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition(8), tcgen05.Pack.NONE), self.acc_dtype
        )
        tAcc0 = cute.flat_divide(
            cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)[((None, None), 0, 0)],
            epi_tile,
        )
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc0[(None, None, 0, 0)]
        )
        thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tid)
        tTR_cL = thr_copy_t2r.partition_D(
            cute.flat_divide(tCcL[((None, None), 0, 0)], epi_tile)
        )
        rAcc = cute.make_rmem_tensor(
            tTR_cL[(None, None, None, 0, 0)].shape, self.acc_dtype
        )
        crd = tTR_cL[(None, None, None, 0, 0)]
        row0, _c0 = crd[0]
        lane = cute.arch.lane_idx()
        wq = row0 // 32
        q = row0 % 32
        hcol = wq * 32 + (q % 8) * 4 + q // 8
        lead = (q // 8) % 2 == 0
        bar_id = Int32(EPI_BAR) + grp
        k = grp
        slot = k % QD
        cute.arch.mbarrier_wait(q_full + slot, (k // QD) & 1)
        g = s_q[slot]
        while g >= 0:
            is1, seg, mt = self.decode(g, u1)
            rows = s_seg_n[seg]
            r0 = s_seg_r0[seg]
            buf = k % self.num_acc
            cute.arch.mbarrier_wait(acc_full + buf, (k // self.num_acc) & 1)
            nch = (rows + 7) // 8
            if is1 != 0:
                hc = Int64(mt * 64 + wq * 16 + lane * 2)
                for ch in cutlass.range(nch, unroll=1):
                    tAcc = cute.flat_divide(
                        cute.make_tensor(
                            tmem_ptr + buf * self.n_max, tCtAcc_fake.layout
                        )[((None, None), 0, 0)],
                        epi_tile,
                    )
                    cute.copy(
                        tiled_copy_t2r,
                        thr_copy_t2r.partition_S(tAcc)[(None, None, None, 0, ch)],
                        rAcc,
                    )
                    cute.arch.fence_view_async_tmem_load()
                    for t in cutlass.range_constexpr(8):
                        nrow = ch * 8 + t
                        xv = rAcc[t]
                        gate = cute.arch.shuffle_sync_bfly(xv, 8)
                        hv = self._swiglu(xv, gate)
                        hi = cute.arch.shuffle_sync_down(hv, 16)
                        if lane < 8:
                            if nrow < rows:
                                r = r0 + nrow
                                _st_global_b32(
                                    h_addr + (Int64(r) * Int64(self.inter) + hc) * 2,
                                    _pack_bf16x2_rn(hv, hi),
                                )
                cute.arch.fence_view_async_tmem_load()
                cute.arch.mbarrier_arrive(acc_empty + buf)
                # release the segment's counter once all 4 warps' rows are written
                _bar_sync(bar_id, 128)
                if epi_tid == 0:
                    _red_release_add(cnt_base + Int64(seg) * 4, Int32(1))
                    cute.arch.mbarrier_arrive(q_empty + slot)
            else:
                col = Int64(mt * TILE_M + hcol)
                op = _policy_evict_last()
                for ch in cutlass.range(nch, unroll=1):
                    tAcc = cute.flat_divide(
                        cute.make_tensor(
                            tmem_ptr + buf * self.n_max, tCtAcc_fake.layout
                        )[((None, None), 0, 0)],
                        epi_tile,
                    )
                    cute.copy(
                        tiled_copy_t2r,
                        thr_copy_t2r.partition_S(tAcc)[(None, None, None, 0, ch)],
                        rAcc,
                    )
                    cute.arch.fence_view_async_tmem_load()
                    for t in cutlass.range_constexpr(8):
                        partner = cute.arch.shuffle_sync_down(rAcc[t], 8)
                        nrow = ch * 8 + t
                        if lead:
                            if nrow < rows:
                                r = r0 + nrow
                                word = _pack_bf16x2(rAcc[t], partner)
                                _st_global_b32_hint(
                                    out_addr
                                    + (Int64(r) * Int64(self.hidden) + col) * 2,
                                    word,
                                    op,
                                )
                cute.arch.fence_view_async_tmem_load()
                cute.arch.mbarrier_arrive(acc_empty + buf)
                _bar_sync(
                    bar_id, 128
                )  # every thread of the group read the entry (and stored its rows)
                if epi_tid == 0:
                    cute.arch.mbarrier_arrive(q_empty + slot)
            k = k + self.epi_groups
            slot = k % QD
            cute.arch.mbarrier_wait(q_full + slot, (k // QD) & 1)
            g = s_q[slot]


_COMPILED = {}


def new_state(device):
    import torch

    return torch.zeros(WIDE_OFF + 4 + 2 * WIDE_PAR, dtype=torch.int32, device=device)


def stages_for(n_max: int, tile_k: int = TILE_K, budget: int = 200 * 1024) -> int:
    return min(10, budget // ((128 + n_max) * tile_k * 2))


def q35_draft_moe_engine(
    x,
    logits,
    w31,
    w2,
    h,
    out,
    wts,
    e2p,
    state,
    *,
    local_start: int,
    n_max: int = 0,
    num_stages: int = 0,
    grid_ctas: int = 0,
    use_pdl: bool = True,
    tile_k: int = TILE_K,
    epi_groups: int = 2,
    num_acc: int = NUM_ACC,
    fc2_chunk: int = 1,
    m_max: int = 0,
):
    import torch

    e, hb, two_i, kb = w31.shape
    hidden, inter = hb * kb, two_i // 2
    m = x.shape[0]
    assert kb == KB and tuple(w2.shape) == (e, inter // KB, hidden, KB), (
        "BlockMajorK weights"
    )
    assert (
        w31.dtype == torch.bfloat16
        and w2.dtype == torch.bfloat16
        and x.dtype == torch.bfloat16
    )
    if m_max <= 0:
        m_max = M_MAX if m <= M_MAX else M_MAX_WIDE
    assert m_max in (M_MAX, M_MAX_WIDE)
    assert 1 <= m <= m_max and e == NLOC, (
        "m <= m_max (32 or 128) rows routed in the prologue, 128 local experts"
    )
    assert (
        x.dim() == 2
        and x.shape[1] == hidden
        and x.stride(1) == 1
        and (x.stride(0) * 2) % 16 == 0
    )
    assert (
        logits.dtype == torch.bfloat16
        and tuple(logits.shape) == (m, E_ALL)
        and logits.is_contiguous()
    )
    assert (
        h.dtype == torch.bfloat16
        and h.shape[0] >= m * TOPK
        and h.shape[1] == inter
        and h.is_contiguous()
    )
    assert (
        out.dtype == torch.bfloat16
        and out.shape[0] >= m * TOPK
        and out.shape[1] == hidden
        and out.is_contiguous()
    )
    assert (
        wts.dtype == torch.bfloat16 and wts.numel() == m * TOPK and wts.is_contiguous()
    )
    assert e2p.dtype == torch.int32 and e2p.numel() == m * TOPK and e2p.is_contiguous()
    if n_max <= 0:
        n_max = 16 if m <= 16 else 32
    assert m_max == M_MAX_WIDE or n_max >= m, (
        "one segment per expert: an expert holds at most m rows"
    )
    if num_stages <= 0:
        num_stages = stages_for(n_max, tile_k)
    if grid_ctas <= 0:
        grid_ctas = torch.cuda.get_device_properties(x.device).multi_processor_count
    ptrs = (
        _ptr(cutlass.BFloat16, w31, 128),
        _ptr(cutlass.BFloat16, w2, 128),
        _ptr(cutlass.BFloat16, w2, 128),
    )
    addrs = (
        Int64(x.data_ptr()),
        Int64(x.stride(0) * 2),
        Int64(logits.data_ptr()),
        Int32(m),
        Int32(local_start),
        Int64(wts.data_ptr()),
        Int64(e2p.data_ptr()),
        Int64(h.data_ptr()),
        Int64(out.data_ptr()),
        Int64(state.data_ptr()),
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    key = (
        hidden,
        inter,
        e,
        n_max,
        num_stages,
        use_pdl,
        tile_k,
        epi_groups,
        num_acc,
        fc2_chunk,
        m_max,
    )
    fn = _COMPILED.get(key)
    if fn is None:
        kern = Q35DraftMoeEngine(
            hidden,
            inter,
            n_max,
            num_stages,
            use_pdl,
            tile_k,
            epi_groups,
            num_acc,
            fc2_chunk,
            m_max,
        )
        fn = cute.compile(kern, *ptrs, *addrs, Int32(e), Int32(grid_ctas), stream)
        _COMPILED[key] = fn
    fn(*ptrs, *addrs, Int32(e), Int32(grid_ctas), stream)
    return out, wts, e2p
