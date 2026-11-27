# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from TensorRT-LLM (Apache-2.0): tensorrt_llm/_torch/kernels/blackwell/moe_nvfp4_engine.py.
# This file has been modified from its original version (``rec``, ``early_unit``, SGLang's routing
# arithmetic and this package's routing tables).

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
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

from .router_ptx import _add as _addf
from .router_ptx import _ld_shared, _smem_addr
from .router_ptx_a import (
    _bit,
    _div_full_bf16,
    _exp_fast,
    _key_val,
    _ld_v4,
    _red_or_shared,
    _sel_eq,
    _sel_lt,
    _st_b16,
    _st_shared,
    _sub,
)

TILE_M = 128
TILE_K = 256
SF_VEC = 16
NUM_ACC = 2
EPI_WARP0 = 4
THREADS = 256
SEG_CAP = 512
CNT_CAP = 512  # counters per parity
B_INFLIGHT = (
    4  # activation stages the loader keeps in flight before signalling the oldest
)
QD = 8  # unit queue slots
QL = 2  # claimed entries the scheduler may hold ahead of the weight loader

EPI_BAR = 1
PRED_MAX = 32
R_PF, R_HIT, R_GHIT, R_RANK, R_NR, R_FIRST, R_H = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
)
R_GEN = 7  # ring record: the generation this call read
REC_HDR = 8
REC_WORDS = REC_HDR + 2 * PRED_MAX
LOG2E = 1.4426950216293334961
RCP6 = 0.16666667163372039795  # fp32(1 / 6)


def _ld_s32(addr: Int64):
    return cute.arch.inline_ptx(
        "ld.global.s32 {$w0}, [{$r0}];", write_only_types=[Int32], read_only_args=[addr]
    )


def _ld_nc_f32(addr: Int64):
    return cute.arch.inline_ptx(
        "ld.global.nc.f32 {$w0}, [{$r0}];",
        write_only_types=[cutlass.Float32],
        read_only_args=[addr],
    )


def _ld_relaxed_u32(addr: Int64):
    return cute.arch.inline_ptx(
        "ld.relaxed.gpu.global.u32 {$w0}, [{$r0}];",
        write_only_types=[Int32],
        read_only_args=[addr],
    )


def _ld_acquire_u32(addr: Int64):
    return cute.arch.inline_ptx(
        "ld.acquire.gpu.global.u32 {$w0}, [{$r0}];",
        write_only_types=[Int32],
        read_only_args=[addr],
    )


def _st_relaxed_u32(addr: Int64, v) -> None:
    cute.arch.inline_ptx(
        "st.relaxed.gpu.global.u32 [{$r0}], {$r1};", read_only_args=[addr, v]
    )


def _red_release_add(addr: Int64, v) -> None:
    cute.arch.inline_ptx(
        "red.release.gpu.global.add.u32 [{$r0}], {$r1};", read_only_args=[addr, v]
    )


def _atom_add_acq_rel(addr: Int64, v):
    return cute.arch.inline_ptx(
        "atom.acq_rel.gpu.global.add.u32 {$w0}, [{$r0}], {$r1};",
        write_only_types=[Int32],
        read_only_args=[addr, v],
    )


def _atom_add_relaxed(addr: Int64, v):
    return cute.arch.inline_ptx(
        "atom.relaxed.gpu.global.add.u32 {$w0}, [{$r0}], {$r1};",
        write_only_types=[Int32],
        read_only_args=[addr, v],
    )


def _fence_acq_rel_gpu() -> None:
    cute.arch.inline_ptx("fence.acq_rel.gpu;")


def _policy_evict_first():
    return cute.arch.inline_ptx(
        "createpolicy.fractional.L2::evict_first.b64 {$w0}, 1.0;",
        write_only_types=[Int64],
    )


def _policy_evict_last():
    return cute.arch.inline_ptx(
        "createpolicy.fractional.L2::evict_last.b64 {$w0}, 1.0;",
        write_only_types=[Int64],
    )


def _st_global_b32_hint(addr: Int64, v, pol) -> None:
    cute.arch.inline_ptx(
        "st.global.L2::cache_hint.b32 [{$r0}], {$r1}, {$r2};",
        read_only_args=[addr, v, pol],
    )


def _st_global_b32(addr: Int64, v) -> None:
    cute.arch.inline_ptx("st.global.b32 [{$r0}], {$r1};", read_only_args=[addr, v])


def _st_global_u8(addr: Int64, v) -> None:
    cute.arch.inline_ptx(
        "{ .reg .b16 t; cvt.u16.u32 t, {$r1}; st.global.u8 [{$r0}], t; }",
        read_only_args=[addr, v],
    )


def _cp_async_16(dst: Int32, src: Int64) -> None:
    cute.arch.inline_ptx(
        "cp.async.cg.shared.global [{$r0}], [{$r1}], 16;", read_only_args=[dst, src]
    )


def _cp_async_4(dst: Int32, src: Int64) -> None:
    cute.arch.inline_ptx(
        "cp.async.ca.shared.global [{$r0}], [{$r1}], 4;", read_only_args=[dst, src]
    )


def _cp_async_arrive_noinc(bar: Int32) -> None:
    cute.arch.inline_ptx(
        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [{$r0}];", read_only_args=[bar]
    )


def _redux_max_abs_f32(x):
    return cute.arch.inline_ptx(
        "redux.sync.max.abs.f32 {$w0}, {$r0}, 0xffffffff;",
        write_only_types=[cutlass.Float32],
        read_only_args=[x],
    )


def _bar_sync(bar_id, n: int) -> None:
    cute.arch.inline_ptx(f"bar.sync {{$r0}}, {n};", read_only_args=[bar_id])


def _cp_async_commit() -> None:
    cute.arch.inline_ptx("cp.async.commit_group;")


def _umax_u32(a, b):
    return cute.arch.inline_ptx(
        "max.u32 {$w0}, {$r0}, {$r1};", write_only_types=[Int32], read_only_args=[a, b]
    )


def _st_shared_v4(addr, a, b, c, d) -> None:
    cute.arch.inline_ptx(
        "st.shared.v4.b32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
        read_only_args=[addr, a, b, c, d],
    )


PASS_BAR = 9
ROUTED_BAR = 10


def _bar_arrive(bar_id, n: int) -> None:
    cute.arch.inline_ptx(f"bar.arrive {{$r0}}, {n};", read_only_args=[bar_id])


def _fence_cta(dep) -> None:
    cute.arch.inline_ptx(
        "{ .reg .b32 d; mov.b32 d, {$r0}; fence.acq_rel.cta; }", read_only_args=[dep]
    )


def _ballot(p):
    return cute.arch.inline_ptx(
        "{ .reg .pred q; setp.ne.s32 q, {$r0}, 0; vote.sync.ballot.b32 {$w0}, q, 0xffffffff; }",
        write_only_types=[Int32],
        read_only_args=[p],
    )


def _popc(v):
    return cute.arch.inline_ptx(
        "popc.b32 {$w0}, {$r0};", write_only_types=[Int32], read_only_args=[v]
    )


def _globaltimer():
    return Int64(cute.arch.globaltimer())


def _mul_rn(a, b):
    return cute.arch.inline_ptx(
        "mul.rn.f32 {$w0}, {$r0}, {$r1};",
        write_only_types=[cutlass.Float32],
        read_only_args=[a, b],
    )


def _add_rn(a, b):
    return cute.arch.inline_ptx(
        "add.rn.f32 {$w0}, {$r0}, {$r1};",
        write_only_types=[cutlass.Float32],
        read_only_args=[a, b],
    )


def _fma_rn(a, b, c):
    return cute.arch.inline_ptx(
        "fma.rn.f32 {$w0}, {$r0}, {$r1}, {$r2};",
        write_only_types=[cutlass.Float32],
        read_only_args=[a, b, c],
    )


def _rcp(a):
    return cute.arch.inline_ptx(
        "rcp.approx.ftz.f32 {$w0}, {$r0};",
        write_only_types=[cutlass.Float32],
        read_only_args=[a],
    )


def _ex2(a):
    return cute.arch.inline_ptx(
        "ex2.approx.ftz.f32 {$w0}, {$r0};",
        write_only_types=[cutlass.Float32],
        read_only_args=[a],
    )


def _e4m3_bits(x):
    return cute.arch.inline_ptx(
        "{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f32 t, 0f00000000, {$r0}; cvt.u32.u16 {$w0}, t; }",
        write_only_types=[Int32],
        read_only_args=[x],
    ) & Int32(0xFF)


def _e4m3_to_f32(b):
    return cute.arch.inline_ptx(
        "{ .reg .b16 t, l, u; .reg .b32 h; cvt.u16.u32 t, {$r0}; cvt.rn.f16x2.e4m3x2 h, t; mov.b32 {l, u}, h;"
        " cvt.f32.f16 {$w0}, l; }",
        write_only_types=[cutlass.Float32],
        read_only_args=[b],
    )


def _e2m1x2(hi, lo):
    return cute.arch.inline_ptx(
        "{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, {$r0}, {$r1}; cvt.u32.u8 {$w0}, t; }",
        write_only_types=[Int32],
        read_only_args=[hi, lo],
    )


def _scale_pack_bf16x2(lo, hi, s):
    return cute.arch.inline_ptx(
        "{ .reg .f32 a, b; mul.rn.f32 a, {$r0}, {$r2}; mul.rn.f32 b, {$r1}, {$r2};"
        " cvt.rn.satfinite.bf16x2.f32 {$w0}, b, a; }",
        write_only_types=[Int32],
        read_only_args=[lo, hi, s],
    )


def _sel_f32(p, a, b):
    """p != 0 ? a : b."""
    return cute.arch.inline_ptx(
        "{ .reg .pred q; setp.ne.s32 q, {$r0}, 0; selp.f32 {$w0}, {$r1}, {$r2}, q; }",
        write_only_types=[cutlass.Float32],
        read_only_args=[p, a, b],
    )


def smem_views(storage):

    def view(mr, n):
        return mr.get_tensor(cute.make_layout(n))

    return (
        view(storage.nseg, 4),
        view(storage.wsum, 12),
        view(storage.exp_e, 128),
        view(storage.exp_y0, 129),
        view(storage.exp_lim, 128),
        view(storage.seg_e, SEG_CAP),
        view(storage.seg_r0, SEG_CAP),
        view(storage.seg_n, SEG_CAP),
        view(storage.tok, 128),
    )


class Q35MoeEngine:
    def __init__(
        self,
        hidden: int,
        inter: int,
        n_max: int,
        num_stages: int,
        tile: int,
        use_pdl: bool = True,
        tile_k: int = TILE_K,
        early_trigger: int = 0,
        q0: int = 0,
        ring: int = 0,
        prerouted: int = 0,
        test_route_delay_ns: int = 0,
        early_unit: int = 0,
    ):
        assert tile_k in (256, 512)
        assert early_unit in (0, 1)
        self.early_unit = early_unit
        # 0: the route front's tables; 1: route from the router GEMM's per-tile candidates
        assert prerouted in (0, 1)
        self.prerouted = prerouted
        self.tile_k = tile_k
        self.epi_groups = 2  # groups of 4 epilogue warps taking alternate units
        self.num_acc = NUM_ACC
        self.threads = 128 + 128 * self.epi_groups
        assert early_trigger in (0, 2, 3)
        assert early_trigger != 3 or prerouted == 1, (
            "early_trigger 3 follows the engine's own routing stores"
        )
        self.early_trigger = early_trigger
        assert test_route_delay_ns >= 0 and (test_route_delay_ns == 0 or prerouted == 1)
        self.test_route_delay_ns = test_route_delay_ns
        self.q0 = q0
        assert ring == 0 or (
            q0 == 1
            and prerouted == 0
            and 0 < ring <= num_stages
            and ring <= hidden // tile_k
        )
        self.ring = ring
        assert (
            hidden % tile_k == 0
            and inter % tile_k == 0
            and (2 * inter) % TILE_M == 0
            and hidden % TILE_M == 0
        )
        assert n_max in (16, 32, 64, 128) and n_max % 8 == 0
        self.hidden, self.inter = hidden, inter
        self.k1_tiles, self.k2_tiles = hidden // tile_k, inter // tile_k
        self.m1_tiles, self.m2_tiles = (2 * inter) // TILE_M, hidden // TILE_M
        self.groups = inter // 64
        self.n_max, self.num_stages, self.tile = n_max, num_stages, tile
        self.b_inflight = max(1, min(B_INFLIGHT, num_stages - 1))
        self.use_pdl = use_pdl

        self.cta_group = tcgen05.CtaGroup.ONE
        self.acc_dtype = cutlass.Float32
        self.sfb_n = 128
        self.b_stage_bytes = n_max * tile_k // 2
        self.sfb_stage_bytes = self.sfb_n * (tile_k // SF_VEC)
        self.kchunks = tile_k // 32
        self.kgroups = tile_k // 64  # 4-scale groups per row per stage
        self.pre_stages = min(num_stages, self.k1_tiles)

    def _setup(self):
        self.tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            SF_VEC,
            self.cta_group,
            (TILE_M, self.n_max),
        )
        mma_inst_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
        self.mma_inst_tile_k = self.tile_k // mma_inst_k
        self.num_sfa_tmem_cols = (TILE_M // 32) * self.mma_inst_tile_k
        self.num_sfb_tmem_cols = self.mma_inst_tile_k * (self.sfb_n // 32)
        total = (
            self.num_acc * self.n_max + self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        )
        assert total <= 512, (
            f"TMEM: {self.num_acc} x {self.n_max} accumulator columns + scales > 512"
        )
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
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            self.tiled_mma, tiler, SF_VEC, self.num_stages
        )
        self.tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            SF_VEC,
            tcgen05.CtaGroup.ONE,
            (TILE_M, self.sfb_n),
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            self.tiled_mma_sfb,
            (TILE_M, self.sfb_n, self.tile_k),
            SF_VEC,
            self.num_stages,
        )

    def _a_atoms(self, w_ptr, wsf_ptr, rows, k, e):
        w = cute.make_tensor(
            w_ptr, cute.make_ordered_layout((rows, k, e), order=(1, 0, 2))
        )
        wsf = cute.make_tensor(
            wsf_ptr, blockscaled_utils.tile_atom_to_shape_SF(w.shape, SF_VEC)
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
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        atom_sf, tens_sf = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            wsf,
            sfa_smem_layout,
            (TILE_M, self.n_max, self.tile_k),
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        self.tma_bytes_a = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        return atom, tens, atom_sf, tens_sf

    @cute.jit
    def __call__(
        self,
        w1_ptr: cute.Pointer,
        w1sf_ptr: cute.Pointer,
        w2_ptr: cute.Pointer,
        w2sf_ptr: cute.Pointer,
        b_dummy_ptr: cute.Pointer,
        xq_addr: Int64,
        xsf_addr: Int64,
        p2t_addr: Int64,
        batch_addr: Int64,
        limit_addr: Int64,
        nctas_addr: Int64,
        max_ctas: Int32,
        scale_c_addr: Int64,
        alpha1_addr: Int64,
        alpha2_addr: Int64,
        h_addr: Int64,
        hsf_addr: Int64,
        out_addr: Int64,
        state_addr: Int64,
        pred_addr: Int64 | None,
        npred: Int32 | None,
        rec_addr: Int64 | None,
        num_experts: Int32,
        grid_ctas: Int32,
        cand_addr: Int64 | None,
        ids_addr: Int64 | None,
        wts_addr: Int64 | None,
        e2p_addr: Int64 | None,
        tp_addr: Int64 | None,
        m_rows: Int32 | None,
        local_start: Int32 | None,
        p2tp_addr: Int64 | None,
        stream: cuda.CUstream,
    ):
        e = num_experts
        self.a_dtype = w1_ptr.value_type
        self.b_dtype = b_dummy_ptr.value_type
        self.sf_dtype = w1sf_ptr.value_type
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
        a1, t1, a1s, t1s = self._a_atoms(
            w1_ptr, w1sf_ptr, 2 * self.inter, self.hidden, e
        )
        a2, t2, a2s, t2s = self._a_atoms(w2_ptr, w2sf_ptr, self.hidden, self.inter, e)
        self.kernel(
            self.tiled_mma,
            a1,
            t1,
            a1s,
            t1s,
            a2,
            t2,
            a2s,
            t2s,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            xq_addr,
            xsf_addr,
            p2t_addr,
            batch_addr,
            limit_addr,
            nctas_addr,
            max_ctas,
            scale_c_addr,
            alpha1_addr,
            alpha2_addr,
            h_addr,
            hsf_addr,
            out_addr,
            state_addr,
            pred_addr,
            npred,
            rec_addr,
            num_experts,
            cand_addr,
            ids_addr,
            wts_addr,
            e2p_addr,
            tp_addr,
            m_rows,
            local_start,
            p2tp_addr,
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
        tma_a1s: cute.CopyAtom,
        mA1s: cute.Tensor,
        tma_a2: cute.CopyAtom,
        mA2: cute.Tensor,
        tma_a2s: cute.CopyAtom,
        mA2s: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        xq_addr: Int64,
        xsf_addr: Int64,
        p2t_addr: Int64,
        batch_addr: Int64,
        limit_addr: Int64,
        nctas_addr: Int64,
        max_ctas: Int32,
        scale_c_addr: Int64,
        alpha1_addr: Int64,
        alpha2_addr: Int64,
        h_addr: Int64,
        hsf_addr: Int64,
        out_addr: Int64,
        state_addr: Int64,
        pred_addr: Int64 | None,
        npred: Int32 | None,
        rec_addr: Int64 | None,
        n_exp: Int32,
        cand_addr: Int64 | None,
        ids_addr: Int64 | None,
        wts_addr: Int64 | None,
        e2p_addr: Int64 | None,
        tp_addr: Int64 | None,
        m_rows: Int32 | None,
        local_start: Int32 | None,
        p2tp_addr: Int64 | None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdx, _, _ = cute.arch.grid_dim()
        lane = cute.arch.lane_idx()
        Stages = self.num_stages
        NA = self.num_acc
        NT = self.threads
        if cutlass.const_expr(self.prerouted == 1):
            APICK = 640  # the merged keys' picks (routing prologue)
        else:
            RP = PRED_MAX if self.ring > 0 else 1
            RI = 8 if self.ring > 0 else 1

        @cute.struct
        class SharedStorage:
            ab_bars: cute.struct.MemRange[cutlass.Int64, Stages * 2]
            acc_bars: cute.struct.MemRange[cutlass.Int64, NA * 2]
            tmem_bar: cutlass.Int64
            tmem_base: cutlass.Int32
            nseg: cute.struct.MemRange[cutlass.Int32, 4]
            wsum: cute.struct.MemRange[cutlass.Int32, 12]
            wsplit: cute.struct.MemRange[cutlass.Int32, 12]
            exp_e: cute.struct.MemRange[cutlass.Int32, 128]
            exp_y0: cute.struct.MemRange[cutlass.Int32, 129]
            exp_lim: cute.struct.MemRange[cutlass.Int32, 128]
            seg_e: cute.struct.MemRange[cutlass.Int32, SEG_CAP]
            seg_r0: cute.struct.MemRange[cutlass.Int32, SEG_CAP]
            seg_n: cute.struct.MemRange[cutlass.Int32, SEG_CAP]
            tok: cute.struct.MemRange[cutlass.Int32, 128]
            uq: cute.struct.MemRange[cutlass.Int32, QD]
            q_bars: cute.struct.MemRange[cutlass.Int64, 3 * QD]
            ex_a1: cute.struct.MemRange[cutlass.Float32, 128]
            ex_c: cute.struct.MemRange[cutlass.Float32, 128]
            ex_a2: cute.struct.MemRange[cutlass.Float32, 128]
            if cutlass.const_expr(self.prerouted == 1):
                apick: cute.struct.MemRange[cutlass.Int32, APICK]
            else:
                rpred: cute.struct.MemRange[cutlass.Int32, RP]
                rclaim: cute.struct.MemRange[cutlass.Int32, RP]
                rinfo: cute.struct.MemRange[cutlass.Int32, RI]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        (
            s_nseg,
            s_wsum,
            s_exp_e,
            s_exp_y0,
            s_exp_lim,
            s_seg_e,
            s_seg_r0,
            s_seg_n,
            s_tok,
        ) = smem_views(storage)
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_a1)
            cpasync.prefetch_descriptor(tma_a1s)
            cpasync.prefetch_descriptor(tma_a2)
            cpasync.prefetch_descriptor(tma_a2s)
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
        sSFA = smem.allocate_tensor(
            element_type=self.sf_dtype,
            layout=sfa_smem_layout_staged,
            byte_alignment=128,
        )
        sSFB = smem.allocate_tensor(
            element_type=self.sf_dtype,
            layout=sfb_smem_layout_staged,
            byte_alignment=128,
        )
        sb_base = sB.iterator.toint()
        sfb_base = sSFB.iterator.toint()
        pipeline_init_arrive(cluster_shape_mn=None, is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=None)

        s_a1 = storage.ex_a1.get_tensor(cute.make_layout(128))
        s_c = storage.ex_c.get_tensor(cute.make_layout(128))
        s_a2 = storage.ex_a2.get_tensor(cute.make_layout(128))
        if tidx >= 128:
            if tidx < 256:
                ei = tidx - 128
                if ei < n_exp:
                    s_a1[ei] = _ld_nc_f32(alpha1_addr + Int64(ei) * 4)
                    s_c[ei] = _ld_nc_f32(scale_c_addr + Int64(ei) * 4)
                    s_a2[ei] = _ld_nc_f32(alpha2_addr + Int64(ei) * 4)
        if cutlass.const_expr(self.prerouted == 1):
            s_rpred = s_rclaim = s_rinfo = None
        else:
            s_rpred = storage.rpred.get_tensor(cute.make_layout(RP))
            s_rclaim = storage.rclaim.get_tensor(cute.make_layout(RP))
            s_rinfo = storage.rinfo.get_tensor(cute.make_layout(RI))
        if cutlass.const_expr(self.ring > 0):
            if warp_idx == 0:
                pa = pred_addr
                rg = Int32(0)
                if rec_addr != Int64(0):
                    rg = _ld_relaxed_u32(rec_addr)
                    pa = (
                        rec_addr
                        + Int64(REC_HDR * 4)
                        + Int64(rg & Int32(1)) * Int64(PRED_MAX * 4)
                    )
                if lane == 0:
                    s_rinfo[R_GEN] = rg
                p = Int32(-1)
                if lane < npred:
                    p = _ld_s32(pa + Int64(lane) * 4)
                    if p >= n_exp:
                        p = Int32(-1)
                    if p < 0:
                        p = Int32(-1)
                dup = Int32(0)
                for j in cutlass.range_constexpr(PRED_MAX):
                    pj = cute.arch.shuffle_sync(p, j)
                    if j < lane:
                        if pj == p:
                            dup = Int32(1)
                if dup != 0:
                    p = Int32(-1)
                s_rpred[lane] = p
                i0 = bidx // self.m1_tiles
                pe = cute.arch.shuffle_sync(p, i0 % PRED_MAX)
                pf = Int32(0)
                if i0 < npred:
                    if pe >= 0:
                        pf = Int32(1)
                if lane == 0:
                    s_rinfo[R_PF] = pf
                if pf != 0:
                    self.ring_fill(
                        ab_full,
                        ab_empty,
                        tma_a1,
                        mA1,
                        tma_a1s,
                        mA1s,
                        sA,
                        sSFA,
                        tiled_mma,
                        cluster_layout_vmnk,
                        pe,
                        bidx % self.m1_tiles,
                    )
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(self.early_trigger == 2):
                cute.arch.griddepcontrol_launch_dependents()
        gen = _ld_relaxed_u32(state_addr)

        if cutlass.const_expr(self.prerouted == 0):
            n_tiles = _ld_s32(nctas_addr)
        warp_in = tidx // 32
        if cutlass.const_expr(self.prerouted == 0):
            lane_lt = (Int32(1) << lane) - Int32(1)
        nexp = Int32(0)
        if cutlass.const_expr(self.prerouted == 0):
            base = Int32(0)
            more = Int32(1)
        if cutlass.const_expr(self.prerouted == 1):
            s_apick = storage.apick.get_tensor(cute.make_layout(APICK))
            if cutlass.const_expr(self.early_unit == 1):
                nexp, anysplit = self._route_prologue(
                    cand_addr,
                    ids_addr,
                    wts_addr,
                    e2p_addr,
                    tp_addr,
                    p2t_addr,
                    m_rows,
                    local_start,
                    n_exp,
                    s_exp_e,
                    s_exp_y0,
                    s_exp_lim,
                    s_wsum,
                    s_apick,
                    tidx,
                    lane,
                    warp_in,
                    bidx,
                    max_ctas,
                    sb_base,
                    gdx,
                    p2tp_addr,
                    (
                        ab_full,
                        ab_empty,
                        tma_a1,
                        mA1,
                        tma_a1s,
                        mA1s,
                        sA,
                        sSFA,
                        tiled_mma,
                        cluster_layout_vmnk,
                    ),
                )
            else:
                nexp, anysplit = self._route_prologue(
                    cand_addr,
                    ids_addr,
                    wts_addr,
                    e2p_addr,
                    tp_addr,
                    p2t_addr,
                    m_rows,
                    local_start,
                    n_exp,
                    s_exp_e,
                    s_exp_y0,
                    s_exp_lim,
                    s_wsum,
                    s_apick,
                    tidx,
                    lane,
                    warp_in,
                    bidx,
                    max_ctas,
                    sb_base,
                    gdx,
                    p2tp_addr,
                )
        else:
            while more != 0:
                y = base + tidx
                bc = Int32(-1)
                bp = Int32(-1)
                bn = Int32(-1)
                lim = Int32(0)
                if y < max_ctas:
                    bc = _ld_s32(batch_addr + Int64(y) * 4)
                    lim = _ld_s32(limit_addr + Int64(y) * 4)
                    if y > 0:
                        bp = _ld_s32(batch_addr + Int64(y - 1) * 4)
                    if y + 1 < max_ctas:
                        bn = _ld_s32(batch_addr + Int64(y + 1) * 4)
                nt = cutlass.min(n_tiles, max_ctas)
                start = Int32(0)
                end = Int32(0)
                if y < nt:
                    if bc != bp:
                        start = Int32(1)
                    if bn != bc:
                        end = Int32(1)
                    if y + 1 == nt:
                        end = Int32(1)
                bal = _ballot(start)
                pre = _popc(bal & lane_lt)
                if lane == 0:
                    s_wsum[warp_in] = _popc(bal)
                cute.arch.barrier()
                off = Int32(0)
                tot = Int32(0)
                for w in cutlass.range_constexpr(NT // 32):
                    v = s_wsum[w]
                    if w < warp_in:
                        off = off + v
                    tot = tot + v
                incl = nexp + off + pre + start  # experts started at or before tile y
                if start != 0:
                    s_exp_e[incl - 1] = bc
                    s_exp_y0[incl - 1] = y
                if end != 0:
                    s_exp_lim[incl - 1] = lim
                nexp = nexp + tot
                base = base + NT
                more = Int32(0)
                if base < nt:
                    more = Int32(1)
                cute.arch.barrier()
            if cutlass.const_expr(self.early_unit == 1 and self.ring == 0):
                if warp_idx == 3:
                    epre, eex = self._early_unit_of(
                        s_exp_e, s_exp_y0, s_exp_lim, nexp, bidx, lane
                    )
                    if epre != Int32(0):
                        self._w_early(
                            ab_full,
                            ab_empty,
                            tma_a1,
                            mA1,
                            tma_a1s,
                            mA1s,
                            sA,
                            sSFA,
                            tiled_mma,
                            cluster_layout_vmnk,
                            eex,
                            bidx % self.m1_tiles,
                        )
        if cutlass.const_expr(self.prerouted == 1):
            # this CTA's permuted -> token copy (the activation loader reads it)
            p2t_ld = p2tp_addr + Int64(bidx) * Int64(
                (max_ctas + Int32(1)) * Int32(self.tile)
            ) * Int64(4)
        else:
            p2t_ld = p2t_addr
        if warp_idx == 0:
            cnts = cute.make_rmem_tensor((4,), Int32)
            mysum = Int32(0)
            for qi in cutlass.range_constexpr(4):
                xi = lane * 4 + qi
                c = Int32(0)
                if xi < nexp:
                    c = (
                        s_exp_lim[xi] - s_exp_y0[xi] * self.tile + (self.n_max - 1)
                    ) // self.n_max
                cnts[qi] = c
                mysum = mysum + c
            incl = mysum
            for off2 in (1, 2, 4, 8, 16):
                t = cute.arch.shuffle_sync_up(incl, off2)
                if lane >= off2:
                    incl = incl + t
            pos = incl - mysum
            for qi in cutlass.range_constexpr(4):
                xi = lane * 4 + qi
                if xi < nexp:
                    r0 = s_exp_y0[xi] * self.tile
                    rows = s_exp_lim[xi] - r0
                    for ch in cutlass.range(cnts[qi], unroll=1):
                        s_seg_e[pos] = s_exp_e[xi]
                        s_seg_r0[pos] = r0 + ch * self.n_max
                        s_seg_n[pos] = cutlass.min(rows - ch * self.n_max, self.n_max)
                        pos = pos + 1
            if lane == 31:
                s_nseg[0] = incl
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
        if cutlass.const_expr(self.ring > 0):
            if warp_idx == 0:
                p = s_rpred[lane]
                c = Int32(-1)
                if lane < npred:
                    if p >= 0:
                        lo = Int32(0)
                        hi = nseg
                        while lo < hi:
                            mid = (lo + hi) // 2
                            if s_seg_e[mid] < p:
                                lo = mid + 1
                            else:
                                hi = mid
                        if lo < nseg:
                            if s_seg_e[lo] == p:
                                c = lo
                hit = Int32(0)
                if c >= 0:
                    hit = Int32(1)
                hb = _ballot(hit)
                nh = _popc(hb)
                rk = Int32(0)
                for j in cutlass.range_constexpr(PRED_MAX):
                    cj = cute.arch.shuffle_sync(c, j)
                    if cj >= 0:
                        if cj < c:
                            rk = rk + 1
                if c >= 0:
                    s_rclaim[rk] = c
                i0 = bidx // m1
                my_c = cute.arch.shuffle_sync(c, i0 % PRED_MAX)
                my_hit = Int32(0)
                if i0 < npred:
                    if my_c >= 0:
                        my_hit = Int32(1)
                below = nh
                if i0 < PRED_MAX:
                    below = _popc(hb & ((Int32(1) << i0) - Int32(1)))
                rank = bidx - m1 * below
                if my_hit != 0:
                    rank = gdx - m1 * nh + m1 * below + bidx % m1
                nr = (nseg - nh) * m1
                first = Int32(-1)
                if my_hit != 0:
                    first = my_c * m1 + bidx % m1
                if lane == 0:
                    s_rinfo[R_HIT] = my_hit
                    s_rinfo[R_GHIT] = my_c * m1 + bidx % m1
                    s_rinfo[R_RANK] = rank
                    s_rinfo[R_NR] = nr
                    s_rinfo[R_H] = nh
                    if my_hit == 0:
                        if rank < nr:
                            first = Int32(-2)
                    s_rinfo[R_FIRST] = first
            cute.arch.barrier()
            hit0 = s_rinfo[R_HIT]
            rank0 = s_rinfo[R_RANK]
            nr0 = s_rinfo[R_NR]
            if rank0 < nr0:
                n1 = (nr0 - 1 - rank0) // gdx + 1
            n1 = n1 + hit0
            if warp_idx == 0:
                if lane == 0:
                    if s_rinfo[R_FIRST] == -2:
                        s_rinfo[R_FIRST] = self.runit(rank0, s_rclaim, s_rinfo[R_H])
            cute.arch.barrier()
        else:
            if bidx < u1:
                n1 = (u1 - 1 - bidx) // gdx + 1

        if warp_idx == 0:
            if cutlass.const_expr(self.early_unit == 1 and self.ring == 0):
                if cutlass.const_expr(self.prerouted == 1):
                    pre1 = Int32(0)
                    if anysplit == Int32(0):
                        if bidx < nexp * Int32(self.m1_tiles):
                            pre1 = Int32(self.pre_stages)
                else:
                    # the stages warp 3 streamed: the same rule on the same shared tables
                    pre1, _ex1 = self._early_unit_of(
                        s_exp_e, s_exp_y0, s_exp_lim, nexp, bidx, lane
                    )
                self.load_a(
                    ab_full,
                    ab_empty,
                    tma_a1,
                    mA1,
                    tma_a1s,
                    mA1s,
                    tma_a2,
                    mA2,
                    tma_a2s,
                    mA2s,
                    sA,
                    sSFA,
                    tiled_mma,
                    cluster_layout_vmnk,
                    s_seg_e,
                    s_q,
                    q_full,
                    q_taken,
                    u1,
                    bidx,
                    s_rinfo,
                    pre1,
                )
            else:
                self.load_a(
                    ab_full,
                    ab_empty,
                    tma_a1,
                    mA1,
                    tma_a1s,
                    mA1s,
                    tma_a2,
                    mA2,
                    tma_a2s,
                    mA2s,
                    sA,
                    sSFA,
                    tiled_mma,
                    cluster_layout_vmnk,
                    s_seg_e,
                    s_q,
                    q_full,
                    q_taken,
                    u1,
                    bidx,
                    s_rinfo,
                )
        elif warp_idx == 1:
            if cutlass.const_expr(self.prerouted == 1):
                _bar_sync(Int32(PASS_BAR), 32 + 128 * self.epi_groups)
            self.load_b(
                ab_full,
                ab_empty,
                s_seg_r0,
                s_seg_n,
                s_tok,
                sb_base,
                sfb_base,
                xq_addr,
                xsf_addr,
                p2t_ld,
                h_addr,
                hsf_addr,
                s_q,
                q_full,
                u1,
                lane,
                bidx,
                s_rinfo,
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
                sSFA,
                sSFB,
                sfa_smem_layout_staged,
                sfb_smem_layout_staged,
                s_q,
                q_full,
                u1,
                s_rinfo,
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
                s_rinfo,
                s_rclaim,
            )
        elif warp_idx >= EPI_WARP0:
            if cutlass.const_expr(self.ring > 0):
                if warp_idx == EPI_WARP0:
                    if bidx == gdx - 1:
                        if rec_addr != Int64(0):
                            self.ring_record(
                                rec_addr,
                                s_rinfo[R_GEN],
                                s_exp_e,
                                s_exp_y0,
                                s_exp_lim,
                                nexp,
                                lane,
                            )
            if cutlass.const_expr(self.prerouted == 1):
                self._route_deferred(
                    s_apick,
                    s_wsum,
                    sb_base,
                    ids_addr,
                    wts_addr,
                    e2p_addr,
                    tp_addr,
                    batch_addr,
                    limit_addr,
                    nctas_addr,
                    p2t_addr,
                    p2tp_addr,
                    m_rows,
                    local_start,
                    n_exp,
                    max_ctas,
                    tidx - Int32(EPI_WARP0 * 32),
                    bidx,
                    gdx,
                )
            self.epilogue(
                acc_full,
                acc_empty,
                tmem_bar,
                tmem_base_smem_ptr,
                tiled_mma,
                s_seg_e,
                s_seg_r0,
                s_seg_n,
                s_a1,
                s_c,
                s_a2,
                h_addr,
                hsf_addr,
                out_addr,
                cnt_base,
                (tidx - EPI_WARP0 * 32) % 128,
                (tidx - EPI_WARP0 * 32) // 128,
                s_q,
                q_full,
                q_empty,
                u1,
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
                if cutlass.const_expr(self.ring > 0):
                    if rec_addr != Int64(0):
                        # every CTA of this call read the record's generation before its wait
                        _st_relaxed_u32(rec_addr, s_rinfo[R_GEN] + Int32(1))
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()
        return

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
            seg, mt = self.fc2_unit(g - u1)
        return is1, seg, mt

    @cute.jit
    def runit(self, r, s_rclaim, nh):
        q = r // self.m1_tiles
        sg = q
        for j in cutlass.range(nh, unroll=1):
            if s_rclaim[j] <= sg:
                sg = sg + 1
        return sg * self.m1_tiles + r % self.m1_tiles

    @cute.jit
    def ring_record(
        self,
        rec_addr: Int64,
        rg,
        s_exp_e: cute.Tensor,
        s_exp_y0: cute.Tensor,
        s_exp_lim: cute.Tensor,
        nexp,
        lane,
    ):
        dst = (
            rec_addr
            + Int64(REC_HDR * 4)
            + Int64((rg + Int32(1)) & Int32(1)) * Int64(PRED_MAX * 4)
        )
        lane_lt = (Int32(1) << lane) - Int32(1)
        base = Int32(0)
        for t in cutlass.range_constexpr(3):
            for q in cutlass.range_constexpr(4):
                xi = lane + 32 * q
                inn = Int32(0)
                ex = Int32(0)
                if xi < nexp:
                    ex = s_exp_e[xi]
                    r = s_exp_lim[xi] - s_exp_y0[xi] * self.tile
                    if cutlass.const_expr(t == 0):
                        if r >= 4:
                            inn = Int32(1)
                    elif cutlass.const_expr(t == 1):
                        if r >= 2:
                            if r < 4:
                                inn = Int32(1)
                    else:
                        if r < 2:
                            inn = Int32(1)
                b = _ballot(inn)
                pos = base + _popc(b & lane_lt)
                if inn != 0:
                    if pos < PRED_MAX:
                        _st_global_b32(dst + Int64(pos) * 4, ex)
                base = base + _popc(b)
        if lane >= base:
            _st_global_b32(dst + Int64(lane) * 4, Int32(-1))

    @cute.jit
    def ring_fill(
        self,
        ab_full,
        ab_empty,
        tma_a1: cute.CopyAtom,
        mA1: cute.Tensor,
        tma_a1s: cute.CopyAtom,
        mA1s: cute.Tensor,
        sA: cute.Tensor,
        sSFA: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cluster_layout_vmnk: cute.Layout,
        ex,
        mt,
    ):
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
        g1s = cute.local_tile(mA1s, (TILE_M, self.tile_k), (None, None, None))
        t1ss, t1sg = cpasync.tma_partition(
            tma_a1s,
            0,
            a_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(thr_mma.partition_A(g1s), 0, 3),
        )
        t1ss = cute.filter_zeros(t1ss)
        t1sg = cute.filter_zeros(t1sg)
        st = make_pipeline_state(PipelineUserType.Producer, self.num_stages)
        wp = _policy_evict_first()
        for kt in cutlass.range_constexpr(self.ring):
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
            cute.copy(
                tma_a1s,
                t1sg[(None, mt, kt, ex)],
                t1ss[(None, st.index)],
                tma_bar_ptr=ab_full + st.index,
                cache_policy=wp,
            )
            st.advance()

    @cute.jit
    def _early_unit_of(
        self,
        s_exp_e: cute.Tensor,
        s_exp_y0: cute.Tensor,
        s_exp_lim: cute.Tensor,
        nexp,
        bidx,
        lane,
    ):
        i = bidx // self.m1_tiles
        big = Int32(0)
        if lane < i:
            if lane < nexp:
                if s_exp_lim[lane] - s_exp_y0[lane] * self.tile > self.n_max:
                    big = Int32(1)
        bal = _ballot(big)
        pre = Int32(0)
        ex = Int32(0)
        if i < 32:
            if i < nexp:
                if bal == Int32(0):
                    pre = Int32(self.pre_stages)
                    ex = s_exp_e[i]
        return pre, ex

    @cute.jit
    def _w_early(
        self,
        ab_full,
        ab_empty,
        tma_a1: cute.CopyAtom,
        mA1: cute.Tensor,
        tma_a1s: cute.CopyAtom,
        mA1s: cute.Tensor,
        sA: cute.Tensor,
        sSFA: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cluster_layout_vmnk: cute.Layout,
        ex,
        mt,
    ):
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
        g1s = cute.local_tile(mA1s, (TILE_M, self.tile_k), (None, None, None))
        t1ss, t1sg = cpasync.tma_partition(
            tma_a1s,
            0,
            a_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(thr_mma.partition_A(g1s), 0, 3),
        )
        t1ss = cute.filter_zeros(t1ss)
        t1sg = cute.filter_zeros(t1sg)
        st = make_pipeline_state(PipelineUserType.Producer, self.num_stages)
        wp = _policy_evict_first()
        for kt in cutlass.range_constexpr(self.pre_stages):
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
            cute.copy(
                tma_a1s,
                t1sg[(None, mt, kt, ex)],
                t1ss[(None, st.index)],
                tma_bar_ptr=ab_full + st.index,
                cache_policy=wp,
            )
            st.advance()

    @cute.jit
    def fc2_unit(self, t):
        """(segment, weight tile) of FC2 ticket t, segment-major."""
        seg = t // self.m2_tiles
        mt = t % self.m2_tiles
        return seg, mt

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
        s_rinfo,
        s_rclaim,
    ):
        if lane == 0:
            k = Int32(0)
            if cutlass.const_expr(self.ring > 0):
                if s_rinfo[R_HIT] != 0:
                    self._push(s_q, q_full, q_empty, q_taken, k, s_rinfo[R_GHIT])
                    k = k + 1
                r = s_rinfo[R_RANK]
                nr = s_rinfo[R_NR]
                nh = s_rinfo[R_H]
                while r < nr:
                    self._push(
                        s_q, q_full, q_empty, q_taken, k, self.runit(r, s_rclaim, nh)
                    )
                    k = k + 1
                    r = r + gdx
            else:
                for j in cutlass.range(n1, unroll=1):
                    self._push(s_q, q_full, q_empty, q_taken, k, bidx + j * gdx)
                    k = k + 1
            t = u2
            if u2 > 0:
                t = _atom_add_relaxed(pool_addr, Int32(1))
            while t < u2:
                tn = _atom_add_relaxed(pool_addr, Int32(1))
                seg, _mt = self.fc2_unit(t)
                ready = Int32(0)
                while ready == 0:
                    if _ld_acquire_u32(cnt_base + Int64(seg) * 4) >= self.m1_tiles:
                        ready = Int32(1)
                self._push(s_q, q_full, q_empty, q_taken, k, u1 + t)
                k = k + 1
                t = tn
            self._push(s_q, q_full, q_empty, q_taken, k, Int32(-1))
            self._push(s_q, q_full, q_empty, q_taken, k + 1, Int32(-1))

    @cute.jit
    def load_a(
        self,
        ab_full,
        ab_empty,
        tma_a1: cute.CopyAtom,
        mA1: cute.Tensor,
        tma_a1s: cute.CopyAtom,
        mA1s: cute.Tensor,
        tma_a2: cute.CopyAtom,
        mA2: cute.Tensor,
        tma_a2s: cute.CopyAtom,
        mA2s: cute.Tensor,
        sA: cute.Tensor,
        sSFA: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cluster_layout_vmnk: cute.Layout,
        s_seg_e: cute.Tensor,
        s_q: cute.Tensor,
        q_full,
        q_taken,
        u1,
        bidx,
        s_rinfo: cute.Tensor,
        pre=None,
    ):
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
        g1s = cute.local_tile(mA1s, (TILE_M, self.tile_k), (None, None, None))
        t1ss, t1sg = cpasync.tma_partition(
            tma_a1s,
            0,
            a_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(thr_mma.partition_A(g1s), 0, 3),
        )
        t1ss = cute.filter_zeros(t1ss)
        t1sg = cute.filter_zeros(t1sg)
        g2 = cute.local_tile(mA2, (TILE_M, self.tile_k), (None, None, None))
        t2s, t2g = cpasync.tma_partition(
            tma_a2,
            0,
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(thr_mma.partition_A(g2), 0, 3),
        )
        g2s = cute.local_tile(mA2s, (TILE_M, self.tile_k), (None, None, None))
        t2ss, t2sg = cpasync.tma_partition(
            tma_a2s,
            0,
            a_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(thr_mma.partition_A(g2s), 0, 3),
        )
        t2ss = cute.filter_zeros(t2ss)
        t2sg = cute.filter_zeros(t2sg)
        st = make_pipeline_state(PipelineUserType.Producer, self.num_stages)
        wp = _policy_evict_first()
        if cutlass.const_expr(self.prerouted == 1):
            if cutlass.const_expr(self.early_unit == 1):
                kt0 = Int32(0)
                if pre != Int32(0):
                    for _i in cutlass.range_constexpr(self.pre_stages):
                        st.advance()
                    kt0 = Int32(self.pre_stages)
        else:
            skip = Int32(0)  # ring: K-stages of the first unit issued before the wait
            if cutlass.const_expr(self.ring > 0):
                if s_rinfo[R_PF] != 0:
                    for _i in cutlass.range_constexpr(self.ring):
                        st.advance()
                    if s_rinfo[R_HIT] != 0:
                        skip = Int32(self.ring)
            if cutlass.const_expr(self.early_unit == 1 and self.ring == 0):
                if pre != Int32(0):
                    for _i in cutlass.range_constexpr(self.pre_stages):
                        st.advance()
                    skip = Int32(self.pre_stages)
        k = Int32(0)
        g = Int32(-1)
        if cutlass.const_expr(self.ring > 0):
            g = s_rinfo[R_FIRST]
            if g < 0:
                cute.arch.mbarrier_wait(q_full, 0)
                g = s_q[0]
        elif cutlass.const_expr(self.q0 == 1):
            if bidx < u1:
                g = bidx  # the static first FC1 unit (the scheduler pushes the same entry)
            else:
                cute.arch.mbarrier_wait(q_full, 0)
                g = s_q[0]
        else:
            cute.arch.mbarrier_wait(q_full, 0)
            g = s_q[0]
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(q_taken)
        while g >= 0:
            is1, seg, mt = self.decode(g, u1)
            ex = s_seg_e[seg]
            if is1 != 0:
                if cutlass.const_expr(self.prerouted == 0):
                    for kt in cutlass.range(skip, self.k1_tiles, unroll=1):
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
                        cute.copy(
                            tma_a1s,
                            t1sg[(None, mt, kt, ex)],
                            t1ss[(None, st.index)],
                            tma_bar_ptr=ab_full + st.index,
                            cache_policy=wp,
                        )
                        st.advance()
                elif cutlass.const_expr(self.early_unit == 1):
                    for kt in cutlass.range(kt0, self.k1_tiles, unroll=1):
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
                        cute.copy(
                            tma_a1s,
                            t1sg[(None, mt, kt, ex)],
                            t1ss[(None, st.index)],
                            tma_bar_ptr=ab_full + st.index,
                            cache_policy=wp,
                        )
                        st.advance()
                    kt0 = Int32(0)
                else:
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
                        cute.copy(
                            tma_a1s,
                            t1sg[(None, mt, kt, ex)],
                            t1ss[(None, st.index)],
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
                    cute.copy(
                        tma_a2s,
                        t2sg[(None, mt, kt, ex)],
                        t2ss[(None, st.index)],
                        tma_bar_ptr=ab_full + st.index,
                        cache_policy=wp,
                    )
                    st.advance()
            if cutlass.const_expr(self.prerouted == 0):
                skip = Int32(0)
            k = k + 1
            slot = k % QD
            cute.arch.mbarrier_wait(q_full + slot, (k // QD) & 1)
            g = s_q[slot]
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(q_taken + slot)

    @cute.jit
    def _b_stage(
        self,
        is1,
        kt,
        rows,
        r0,
        sb_stage,
        sfb_stage,
        xq_addr,
        xsf_addr,
        h_addr,
        hsf_addr,
        s_tok,
        lane,
    ):
        kc = self.kchunks
        kg = self.kgroups
        rb = (self.n_max // 8) * 1024
        if is1 != 0:
            for n in cutlass.range(lane, rows, 32, unroll=1):
                t = s_tok[n]
                src = (
                    xq_addr + Int64(t) * Int64(self.hidden // 2) + Int64(kt * (kc * 16))
                )
                ssrc = (
                    xsf_addr
                    + Int64(t) * Int64(self.hidden // SF_VEC)
                    + Int64(kt * (kg * 4))
                )
                dbase = sb_stage + (n >> 3) * 1024 + (n & 7) * 128
                n7 = n & 7
                for c in cutlass.range_constexpr(kc):
                    _cp_async_16(
                        dbase + ((c >> 3) * rb) + (((c & 7) ^ n7) << 4),
                        src + Int64(c * 16),
                    )
                sbase = sfb_stage + (n & 31) * 16 + (n >> 5) * 4
                for g in cutlass.range_constexpr(kg):
                    _cp_async_4(sbase + g * 512, ssrc + Int64(g * 4))
        else:
            for n in cutlass.range(lane, rows, 32, unroll=1):
                r = r0 + n
                src = h_addr + Int64(r) * Int64(self.inter // 2) + Int64(kt * (kc * 16))
                ssrc = (
                    hsf_addr
                    + (Int64(r >> 3) * Int64(self.groups) + Int64(kt * kg)) * 32
                    + Int64(r & 7) * 4
                )
                dbase = sb_stage + (n >> 3) * 1024 + (n & 7) * 128
                n7 = n & 7
                for c in cutlass.range_constexpr(kc):
                    _cp_async_16(
                        dbase + ((c >> 3) * rb) + (((c & 7) ^ n7) << 4),
                        src + Int64(c * 16),
                    )
                sbase = sfb_stage + (n & 31) * 16 + (n >> 5) * 4
                for g in cutlass.range_constexpr(kg):
                    _cp_async_4(sbase + g * 512, ssrc + Int64(g * 32))
        _cp_async_commit()

    @cute.jit
    def _route_prologue(
        self,
        cand_addr: Int64,
        ids_addr: Int64,
        wts_addr: Int64,
        e2p_addr: Int64,
        tp_addr: Int64,
        p2t_addr: Int64,
        m_rows: Int32,
        local_start: Int32,
        n_exp: Int32,
        s_exp_e,
        s_exp_y0,
        s_exp_lim,
        s_wsum,
        s_pick,
        tidx,
        lane,
        warp_in,
        bidx,
        max_ctas: Int32,
        sb_base,
        gdx,
        p2tp_addr: Int64,
        wtma=None,
    ):
        NT: cutlass.Constexpr = self.threads
        TK: cutlass.Constexpr = 10
        NC: cutlass.Constexpr = 40
        tile: cutlass.Constexpr = self.tile
        cbase = Int32(sb_base)
        mask_base = cbase + Int32(64 * NC * 4)
        start_base = mask_base + Int32(1024)  # each local expert's first tile
        sink = start_base + Int32(
            512
        )  # the mask OR of a pick on another rank lands here (never read)
        key_base = _smem_addr(s_pick.iterator)
        assert 2 * NT >= 64 * TK, (
            "two loads per thread must cover 64 tokens x 10 (384 threads, epi_groups 2)"
        )
        nv = m_rows * Int32(NC // 4)
        i0 = tidx
        i1 = tidx + Int32(NT)
        a0 = Int32(0)
        b0 = Int32(0)
        c0 = Int32(0)
        d0 = Int32(0)
        a1 = Int32(0)
        b1 = Int32(0)
        c1 = Int32(0)
        d1 = Int32(0)
        if i0 < nv:
            a0, b0, c0, d0 = _ld_v4(cand_addr + Int64(i0) * Int64(16))
        if i1 < nv:
            a1, b1, c1, d1 = _ld_v4(cand_addr + Int64(i1) * Int64(16))
        for i in cutlass.range(tidx, 256, NT, unroll=1):
            _st_shared(mask_base + i * Int32(4), Int32(0))
        if i0 < nv:
            _st_shared_v4(cbase + i0 * Int32(16), a0, b0, c0, d0)
        if i1 < nv:
            _st_shared_v4(cbase + i1 * Int32(16), a1, b1, c1, d1)
        cute.arch.barrier()
        t = tidx
        if t < m_rows:
            base = cbase + t * Int32(NC * 4)
            p0 = base
            p1 = base + Int32(40)
            p2 = base + Int32(80)
            p3 = base + Int32(120)
            h0 = _ld_shared(p0)
            h1 = _ld_shared(p1)
            h2 = _ld_shared(p2)
            h3 = _ld_shared(p3)
            n0 = _ld_shared(p0 + Int32(4))
            n1 = _ld_shared(p1 + Int32(4))
            n2 = _ld_shared(p2 + Int32(4))
            n3 = _ld_shared(p3 + Int32(4))
            gk = []
            for k in cutlass.range_constexpr(TK):
                g = _umax_u32(_umax_u32(h0, h1), _umax_u32(h2, h3))
                gk.append(g)
                if cutlass.const_expr(k < TK - 1):
                    p0 = _sel_eq(h0, g, p0 + Int32(4), p0)
                    p1 = _sel_eq(h1, g, p1 + Int32(4), p1)
                    p2 = _sel_eq(h2, g, p2 + Int32(4), p2)
                    p3 = _sel_eq(h3, g, p3 + Int32(4), p3)
                    h0 = _sel_eq(h0, g, n0, h0)
                    h1 = _sel_eq(h1, g, n1, h1)
                    h2 = _sel_eq(h2, g, n2, h2)
                    h3 = _sel_eq(h3, g, n3, h3)
                    n0 = _ld_shared(p0 + Int32(4))
                    n1 = _ld_shared(p1 + Int32(4))
                    n2 = _ld_shared(p2 + Int32(4))
                    n3 = _ld_shared(p3 + Int32(4))
            kb = key_base + t * Int32(TK * 4)
            mrow = mask_base + (t // Int32(32)) * Int32(4)
            mbit = _bit(t % Int32(32))
            for k in cutlass.range_constexpr(TK):
                _st_shared(kb + Int32(4 * k), gk[k])
            for k in cutlass.range_constexpr(TK):
                lid = Int32(65535) - (gk[k] & Int32(65535)) - local_start
                _red_or_shared(_sel_lt(lid, n_exp, mrow + lid * Int32(8), sink), mbit)
        cute.arch.barrier()
        if warp_in == Int32(0):
            cn = cute.make_rmem_tensor(cute.make_layout((4,)), Int32)
            pk = Int32(0)
            bigl = Int32(0)
            for qi in cutlass.range_constexpr(4):
                e = lane * Int32(4) + Int32(qi)
                c = Int32(0)
                if e < n_exp:
                    c = _popc(_ld_shared(mask_base + e * Int32(8))) + _popc(
                        _ld_shared(mask_base + e * Int32(8) + Int32(4))
                    )
                cn[qi] = c
                pk = (
                    pk
                    + (((c + Int32(tile - 1)) // Int32(tile)) << Int32(16))
                    + _sel_lt(Int32(0), c, Int32(1), Int32(0))
                )
                bigl = bigl | _sel_lt(Int32(self.n_max), c, Int32(1), Int32(0))
            incl = pk
            for off2 in (1, 2, 4, 8, 16):
                u = cute.arch.shuffle_sync_up(incl, off2)
                if lane >= off2:
                    incl = incl + u
            exl = incl - pk
            toff = exl >> Int32(16)
            aoff = exl & Int32(65535)
            for qi in cutlass.range_constexpr(4):
                e = lane * Int32(4) + Int32(qi)
                c = cn[qi]
                if e < n_exp:
                    _st_shared(start_base + e * Int32(4), toff)
                if c > Int32(0):
                    s_exp_e[aoff] = e
                    s_exp_y0[aoff] = toff
                    s_exp_lim[aoff] = toff * Int32(tile) + c
                aoff = aoff + _sel_lt(Int32(0), c, Int32(1), Int32(0))
                toff = toff + (c + Int32(tile - 1)) // Int32(tile)
            bbal = _ballot(bigl)
            if lane == Int32(31):
                s_wsum[0] = incl >> Int32(16)  # tiles
                s_wsum[1] = incl & Int32(65535)  # experts with rows
                s_wsum[2] = bbal
            if cutlass.const_expr(self.early_unit == 1):
                nx = cute.arch.shuffle_sync(incl, 31) & Int32(65535)
                if bbal == Int32(0):
                    if bidx < nx * Int32(self.m1_tiles):
                        cute.arch.sync_warp()
                        self._w_early(
                            *wtma,
                            s_exp_e[bidx // Int32(self.m1_tiles)],
                            bidx % Int32(self.m1_tiles),
                        )
        cute.arch.barrier()
        return s_wsum[1], s_wsum[2]

    @cute.jit
    def _route_deferred(
        self,
        s_pick,
        s_wsum,
        sb_base,
        ids_addr: Int64,
        wts_addr: Int64,
        e2p_addr: Int64,
        tp_addr: Int64,
        batch_addr: Int64,
        limit_addr: Int64,
        nctas_addr: Int64,
        p2t_addr: Int64,
        p2tp_addr: Int64,
        m_rows: Int32,
        local_start: Int32,
        n_exp: Int32,
        max_ctas: Int32,
        q,
        bidx,
        gdx,
    ):
        TK: cutlass.Constexpr = 10
        NC: cutlass.Constexpr = 40
        NE: cutlass.Constexpr = 128 * self.epi_groups  # epilogue threads: q in [0, NE)
        if cutlass.const_expr(self.test_route_delay_ns > 0):
            t_hold = _globaltimer()
            t_now = t_hold
            while t_now - t_hold < Int64(self.test_route_delay_ns):
                t_now = _globaltimer()
        tile: cutlass.Constexpr = self.tile
        key_base = _smem_addr(s_pick.iterator)
        mask_base = Int32(sb_base) + Int32(64 * NC * 4)
        start_base = mask_base + Int32(1024)
        p2tp_cta = p2tp_addr + Int64(bidx) * Int64(
            (max_ctas + Int32(1)) * Int32(tile)
        ) * Int64(4)
        for x in cutlass.range(q, m_rows * Int32(TK), NE, unroll=1):
            tt = x // Int32(TK)
            key = _ld_shared(key_base + x * Int32(4))
            lid = Int32(65535) - (key & Int32(65535)) - local_start
            own = Int32(0)
            if tt % gdx == bidx:
                own = Int32(1)
            row = Int32(-1)
            if _sel_lt(lid, n_exp, Int32(1), Int32(0)) != Int32(0):
                lo = _ld_shared(mask_base + lid * Int32(8))
                rank = Int32(0)
                if tt < Int32(32):
                    rank = _popc(lo & ((Int32(1) << tt) - Int32(1)))
                else:
                    hi = _ld_shared(mask_base + lid * Int32(8) + Int32(4))
                    rank = _popc(lo) + _popc(
                        hi & ((Int32(1) << (tt - Int32(32))) - Int32(1))
                    )
                row = _ld_shared(start_base + lid * Int32(4)) * Int32(tile) + rank
                _st_global_b32(p2tp_cta + Int64(row) * Int64(4), tt)
                if own != Int32(0):
                    _st_global_b32(p2t_addr + Int64(row) * Int64(4), tt)
            if own != Int32(0):
                _st_global_b32(e2p_addr + Int64(x) * Int64(4), row)
        e = q
        cnt = Int32(0)
        st0 = Int32(0)
        if e < n_exp:
            cnt = _popc(_ld_shared(mask_base + e * Int32(8))) + _popc(
                _ld_shared(mask_base + e * Int32(8) + Int32(4))
            )
            st0 = _ld_shared(start_base + e * Int32(4))
        _fence_cta(cnt)
        _bar_arrive(Int32(PASS_BAR), 32 + NE)
        if cnt > Int32(0):
            if e % gdx == bidx:
                # the tile tables of this expert's tiles (RouteTables.cta_expert / cta_row_limit)
                t_batch = batch_addr
                t_limit = limit_addr
                ntl = (cnt + Int32(tile - 1)) // Int32(tile)
                for c in cutlass.range(ntl, unroll=1):
                    _st_global_b32(t_batch + Int64(st0 + c) * Int64(4), e)
                    lim1 = (st0 + c + Int32(1)) * Int32(tile)
                    lim2 = st0 * Int32(tile) + cnt
                    _st_global_b32(
                        t_limit + Int64(st0 + c) * Int64(4), cutlass.min(lim1, lim2)
                    )
        if bidx == Int32(0):
            if q == 0:
                ntot = s_wsum[0]
                # RouteTables.live_ctas and total_padded
                _st_global_b32(nctas_addr, ntot)
                _st_global_b32(tp_addr, ntot * Int32(tile))
        t = bidx + q * gdx
        if t < m_rows:
            kb = key_base + t * Int32(TK * 4)
            val = []
            for k in cutlass.range_constexpr(TK):
                g = _ld_shared(kb + Int32(4 * k))
                o = Int64(t) * Int64(TK) + Int64(k)
                _st_global_b32(
                    ids_addr + o * Int64(4), Int32(65535) - (g & Int32(65535))
                )
                val.append(_key_val(g))
            ex = []
            for k in cutlass.range_constexpr(TK):
                ex.append(_exp_fast(_sub(val[k], val[0])))
            ssum = _addf(
                _addf(_addf(_addf(ex[0], ex[8]), ex[4]), _addf(ex[2], ex[6])),
                _addf(_addf(_addf(ex[1], ex[9]), ex[5]), _addf(ex[3], ex[7])),
            )
            for k in cutlass.range_constexpr(TK):
                o = Int64(t) * Int64(TK) + Int64(k)
                _st_b16(wts_addr + o * Int64(2), _div_full_bf16(ex[k], ssum))
        _fence_acq_rel_gpu()
        if cutlass.const_expr(self.use_pdl and self.early_trigger == 3):
            _bar_sync(Int32(ROUTED_BAR), NE)
            if q == 0:
                _fence_acq_rel_gpu()
                cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def load_b(
        self,
        ab_full,
        ab_empty,
        s_seg_r0,
        s_seg_n,
        s_tok,
        sb_base,
        sfb_base,
        xq_addr,
        xsf_addr,
        p2t_addr,
        h_addr,
        hsf_addr,
        s_q,
        q_full,
        u1,
        lane,
        bidx,
        s_rinfo: cute.Tensor,
    ):
        st = make_pipeline_state(PipelineUserType.Producer, self.num_stages)
        if cutlass.const_expr(self.ring > 0):
            if s_rinfo[R_PF] != 0:
                if s_rinfo[R_HIT] == 0:
                    # a miss: the pre-filled stages carry no unit; arrive on them without copies
                    for _i in cutlass.range_constexpr(self.ring):
                        cute.arch.mbarrier_wait(ab_empty + st.index, st.phase)
                        _cp_async_arrive_noinc((ab_full + st.index).toint())
                        st.advance()
        k = Int32(0)
        g = Int32(-1)
        if cutlass.const_expr(self.ring > 0):
            g = s_rinfo[R_FIRST]
            if g < 0:
                cute.arch.mbarrier_wait(q_full, 0)
                g = s_q[0]
        elif cutlass.const_expr(self.q0 == 1):
            if bidx < u1:
                g = bidx  # the static first FC1 unit (the scheduler pushes the same entry)
            else:
                cute.arch.mbarrier_wait(q_full, 0)
                g = s_q[0]
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
                cute.arch.sync_warp()
                for n in cutlass.range(lane, rows, 32, unroll=1):
                    s_tok[n] = _ld_s32(p2t_addr + Int64(r0 + n) * 4)
                cute.arch.sync_warp()
            for kt in cutlass.range(kts, unroll=1):
                cute.arch.mbarrier_wait(ab_empty + st.index, st.phase)
                self._b_stage(
                    is1,
                    kt,
                    rows,
                    r0,
                    sb_base + st.index * self.b_stage_bytes,
                    sfb_base + st.index * self.sfb_stage_bytes,
                    xq_addr,
                    xsf_addr,
                    h_addr,
                    hsf_addr,
                    s_tok,
                    lane,
                )
                _cp_async_arrive_noinc((ab_full + st.index).toint())
                st.advance()
            k = k + 1
            slot = k % QD
            cute.arch.mbarrier_wait(q_full + slot, (k // QD) & 1)
            g = s_q[slot]

    def _s2t(self, sSF: cute.Tensor, tSF: cute.Tensor):
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group), self.sf_dtype
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_
        )
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

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
        sSFA: cute.Tensor,
        sSFB: cute.Tensor,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        s_q: cute.Tensor,
        q_full,
        u1,
        s_rinfo: cute.Tensor,
    ):
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C((TILE_M, self.n_max))
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        cute.arch.alloc_tmem(self.num_tmem_cols, tmem_base_smem_ptr)
        cute.arch.mbarrier_arrive(tmem_bar)
        cute.arch.relinquish_tmem_alloc_permit()
        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_smem_ptr)
        sfa_col = self.num_acc * self.n_max
        sfb_col = sfa_col + self.num_sfa_tmem_cols
        tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma,
            tiled_mma.shape_mnk,
            SF_VEC,
            cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
        )
        tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma,
            tiled_mma.shape_mnk,
            SF_VEC,
            cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
        )
        tCtSFA = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + sfa_col, dtype=self.sf_dtype), tCtSFA_layout
        )
        tCtSFB = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + sfb_col, dtype=self.sf_dtype), tCtSFB_layout
        )
        copy_sfa, tCsSFA_s2t, tCtSFA_s2t = self._s2t(sSFA, tCtSFA)
        copy_sfb, tCsSFB_s2t, tCtSFB_s2t = self._s2t(sSFB, tCtSFB)
        num_kphases = cute.size(tCrA, mode=[2])
        ab_st = make_pipeline_state(PipelineUserType.Consumer, self.num_stages)
        acc_st = make_pipeline_state(PipelineUserType.Producer, self.num_acc)
        if cutlass.const_expr(self.ring > 0):
            if s_rinfo[R_PF] != 0:
                if s_rinfo[R_HIT] == 0:
                    for _i in cutlass.range_constexpr(self.ring):
                        cute.arch.mbarrier_wait(ab_full + ab_st.index, ab_st.phase)
                        with cute.arch.elect_one():
                            tcgen05.commit(ab_empty + ab_st.index, None, self.cta_group)
                        ab_st.advance()
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
                cute.copy(
                    copy_sfa,
                    tCsSFA_s2t[(None, None, None, None, ab_st.index)],
                    tCtSFA_s2t,
                )
                cute.copy(
                    copy_sfb,
                    tCsSFB_s2t[(None, None, None, None, ab_st.index)],
                    tCtSFB_s2t,
                )
                for kp in cutlass.range(num_kphases, unroll_full=True):
                    tiled_mma.set(tcgen05.Field.SFA, tCtSFA[(None, None, kp)].iterator)
                    tiled_mma.set(tcgen05.Field.SFB, tCtSFB[(None, None, kp)].iterator)
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
        s_a1: cute.Tensor,
        s_c: cute.Tensor,
        s_a2: cute.Tensor,
        h_addr,
        hsf_addr,
        out_addr,
        cnt_base,
        epi_tid,
        grp,
        s_q: cute.Tensor,
        q_full,
        q_empty,
        u1,
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
        is_up = Int32(0)
        if ((q >> 3) & 1) == 0:
            is_up = Int32(1)
        bar_id = Int32(EPI_BAR) + grp
        k = grp
        slot = k % QD
        cute.arch.mbarrier_wait(q_full + slot, (k // QD) & 1)
        g = s_q[slot]
        while g >= 0:
            is1, seg, mt = self.decode(g, u1)
            ex = s_seg_e[seg]
            rows = s_seg_n[seg]
            r0 = s_seg_r0[seg]
            buf = k % self.num_acc
            cute.arch.mbarrier_wait(acc_full + buf, (k // self.num_acc) & 1)
            nch = (rows + 7) // 8
            if is1 != 0:
                a = s_a1[ex]
                c = s_c[ex]
                c1 = _mul_rn(a, cutlass.Float32(LOG2E))
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
                        sgm = _rcp(
                            _add_rn(_ex2(-_mul_rn(gate, c1)), cutlass.Float32(1.0))
                        )
                        gs = _mul_rn(gate, sgm)
                        hv = _mul_rn(
                            _fma_rn(xv, a, _mul_rn(a, cutlass.Float32(0.0))), gs
                        )
                        hv = _sel_f32(is_up, hv, cutlass.Float32(0.0))
                        amax = _redux_max_abs_f32(_mul_rn(hv, c))
                        sfb = _e4m3_bits(_mul_rn(amax, cutlass.Float32(RCP6)))
                        sff = _e4m3_to_f32(sfb)
                        osc = _sel_f32(
                            sfb, _rcp(sff), cutlass.Float32(0.0)
                        )  # sfb != 0 <=> scale != 0
                        qv = _mul_rn(_mul_rn(hv, osc), c)
                        byte = _e2m1x2(cute.arch.shuffle_sync_down(qv, 16), qv)
                        if lane < 8:
                            if nrow < rows:
                                r = r0 + nrow
                                _st_global_u8(
                                    h_addr
                                    + Int64(r) * Int64(self.inter // 2)
                                    + Int64(mt * 32 + wq * 8 + lane),
                                    byte,
                                )
                                if lane == 0:
                                    saddr = (
                                        hsf_addr
                                        + (
                                            Int64(r >> 3) * Int64(self.groups)
                                            + Int64(mt)
                                        )
                                        * 32
                                        + Int64(r & 7) * 4
                                        + Int64(wq)
                                    )
                                    _st_global_u8(saddr, sfb)
                cute.arch.fence_view_async_tmem_load()
                cute.arch.mbarrier_arrive(acc_empty + buf)
                # release the segment's counter once all 4 warps' rows are written
                _bar_sync(bar_id, 128)
                if epi_tid == 0:
                    _red_release_add(cnt_base + Int64(seg) * 4, Int32(1))
                    cute.arch.mbarrier_arrive(q_empty + slot)
            else:
                s = s_a2[ex]
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
                                _st_global_b32_hint(
                                    out_addr
                                    + (Int64(r) * Int64(self.hidden) + col) * 2,
                                    _scale_pack_bf16x2(rAcc[t], partner, s),
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


def _ptr(dtype, t, align):
    from cutlass.cute.runtime import make_ptr

    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)


def new_state(device):
    """Per-device call state (int32): gen, done, pad, counters[2][CNT_CAP]; zero-initialized."""
    import torch

    return torch.zeros(16 + 2 * CNT_CAP, dtype=torch.int32, device=device)


def new_record(device, layers: int):
    import torch

    rec = torch.full((layers, REC_WORDS), -1, dtype=torch.int32, device=device)
    rec[:, :REC_HDR] = 0
    return rec


def q35_moe_engine(
    xq,
    xsf,
    p2t,
    cta_batch,
    cta_limit,
    num_ctas,
    w31,
    w31_sf,
    w2,
    w2_sf,
    scale_c,
    alpha1,
    alpha2,
    h,
    hsf,
    out,
    state,
    *,
    tile: int,
    n_max: int,
    num_stages: int = 0,
    grid_ctas: int = 0,
    use_pdl: bool = True,
    tile_k: int = TILE_K,
    early_trigger: int = 0,
    q0: int = 0,
    ring: int = 0,
    pred=None,
    rec=None,
    cand=None,
    ids=None,
    wts=None,
    e2p=None,
    total_padded=None,
    m: int = 0,
    local_start: int = 0,
    p2tp=None,
    test_route_delay_ns: int = 0,
    early_unit: int = 0,
):
    import torch

    e, two_i, half_h = w31.shape
    hidden, inter = 2 * half_h, two_i // 2
    assert e <= 128, "per-expert scalars are staged for <= 128 local experts"
    if num_stages <= 0:
        per_stage = (128 + n_max) * tile_k // 2 + 2 * 128 * tile_k // SF_VEC
        num_stages = min(10, 200 * 1024 // per_stage)
    if grid_ctas <= 0:
        grid_ctas = torch.cuda.get_device_properties(xq.device).multi_processor_count
    dummy = torch.empty(0, dtype=torch.uint8, device=xq.device)
    ptrs = (
        _ptr(cutlass.Float4E2M1FN, w31, 32),
        _ptr(cutlass.Float8E4M3FN, w31_sf, 16),
        _ptr(cutlass.Float4E2M1FN, w2, 32),
        _ptr(cutlass.Float8E4M3FN, w2_sf, 16),
        _ptr(cutlass.Float4E2M1FN, w2, 32),
    )
    addrs = (
        Int64(xq.data_ptr()),
        Int64(xsf.data_ptr()),
        Int64(p2t.data_ptr()),
        Int64(cta_batch.data_ptr()),
        Int64(cta_limit.data_ptr()),
        Int64(num_ctas.data_ptr()),
        Int32(cta_batch.numel()),
        Int64(scale_c.data_ptr()),
        Int64(alpha1.data_ptr()),
        Int64(alpha2.data_ptr()),
        Int64(h.data_ptr()),
        Int64(hsf.data_ptr()),
        Int64(out.data_ptr()),
        Int64(state.data_ptr()),
    )
    if cand is None:
        ring_words = (
            Int64(pred.data_ptr() if (ring > 0 and pred is not None) else 0),
            Int32(
                min(
                    pred.numel() if pred is not None else PRED_MAX,
                    PRED_MAX,
                    grid_ctas // ((2 * inter) // TILE_M),
                )
                if (ring > 0 and (pred is not None or rec is not None))
                else 0
            ),
            Int64(rec.data_ptr() if (ring > 0 and rec is not None) else 0),
        )
    else:
        assert ring == 0 and pred is None and rec is None, (
            "the ring is the route front's (prerouted 0)"
        )
        ring_words = (None, None, None)

    assert pred is None or (pred.dtype == torch.int32 and pred.is_contiguous()), (
        "pred: contiguous int32"
    )
    assert pred is None or rec is None, "pred or rec, not both"
    assert rec is None or (
        rec.dtype == torch.int32 and rec.is_contiguous() and rec.numel() >= REC_WORDS
    ), "rec: one contiguous int32 row of new_record"
    del dummy
    if cand is not None:
        assert (
            m <= 64
            and all(t is not None for t in (ids, wts, e2p, total_padded))
            and e2p.numel() == m * 10
        )
        assert ids.numel() == m * 10 and wts.numel() == m * 10 and e <= 128
    # four key lists per token: the router GEMM's per-tile candidates
    assert cand is None or (cand.numel() == 4 * 10 * m and cand.is_contiguous())
    p64 = lambda t: Int64(t.data_ptr() if t is not None else 0)  # noqa: E731
    p2tp_need = grid_ctas * (cta_batch.numel() + 1) * tile
    if cand is not None and p2tp is None:
        p2tp = torch.empty((p2tp_need,), dtype=torch.int32, device=xq.device)
    assert cand is None or (p2tp.dtype == torch.int32 and p2tp.numel() >= p2tp_need), (
        "p2tp too small"
    )
    extra = (
        (None,) * 8
        if cand is None
        else (
            p64(cand),
            p64(ids),
            p64(wts),
            p64(e2p),
            p64(total_padded),
            Int32(m),
            Int32(local_start),
            p64(p2tp),
        )
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    key = (
        hidden,
        inter,
        e,
        n_max,
        num_stages,
        tile,
        use_pdl,
        tile_k,
        early_trigger,
        q0,
        ring,
        cand is not None,
        test_route_delay_ns,
    ) + ((("early_unit", early_unit),) if early_unit else ())
    fn = _COMPILED.get(key)
    if fn is None:
        kern = Q35MoeEngine(
            hidden,
            inter,
            n_max,
            num_stages,
            tile,
            use_pdl,
            tile_k,
            early_trigger,
            q0,
            ring,
            prerouted=1 if cand is not None else 0,
            test_route_delay_ns=test_route_delay_ns,
            **({"early_unit": early_unit} if early_unit else {}),
        )
        fn = cute.compile(
            kern, *ptrs, *addrs, *ring_words, Int32(e), Int32(grid_ctas), *extra, stream
        )
        _COMPILED[key] = fn
    fn(*ptrs, *addrs, *ring_words, Int32(e), Int32(grid_ctas), *extra, stream)
    return out
