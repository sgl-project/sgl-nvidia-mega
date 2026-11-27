# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from TensorRT-LLM (Apache-2.0): tensorrt_llm/_torch/kernels/blackwell/moe_router_gemm.py.

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import BFloat16, Float32, Int32, Int64, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op

from .router_ptx import (
    _add,
    _asm,
    _cluster_arrive_release,
    _cluster_wait_acquire,
    _f32,
    _gdc_launch,
    _gdc_wait,
    _i32,
    _ld_shared,
    _smem_addr,
    _st_v2,
    _struct,
    _unpack,
)
from .router_ptx_a import _ld_f32, _nvfp4_block, _st_b32, _st_u8

EXPERT_TILE = 128
K_TILE = 64
MAX_TOKEN_TILE = 128


@dsl_user_op
def _tmem_ld16(taddr: Int32, *, loc=None, ip=None):
    regs = ", ".join(f"${i}" for i in range(16))
    out = _asm(
        _struct(16),
        [_i32(taddr, loc, ip)],
        f"tcgen05.ld.sync.aligned.32x32b.x16.b32 {{{regs}}}, [$16];\n\ttcgen05.wait::ld.sync.aligned;",
        ",".join(["=r"] * 16) + ",r,~{memory}",
        loc=loc,
        ip=ip,
    )
    return _unpack(out, 16, loc, ip)


@dsl_user_op
def _ld_shared_v4_f32(addr: Int32, *, loc=None, ip=None):
    out = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [_i32(addr, loc, ip)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r,~{memory}",
        loc=loc,
        ip=ip,
    )
    return tuple(
        Float32(llvm.extractvalue(T.f32(), out, [i], loc=loc, ip=ip)) for i in range(4)
    )


@dsl_user_op
def _pack_bf16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Int32:
    """BF16 (round to nearest even) of lo in the low half, hi in the high half."""
    return Int32(
        _asm(
            T.i32(),
            [_f32(lo, loc, ip), _f32(hi, loc, ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _redux_max_u32(v: Int32, *, loc=None, ip=None) -> Int32:
    """Unsigned max over the full warp."""
    return Int32(
        _asm(
            T.i32(),
            [_i32(v, loc, ip)],
            "redux.sync.max.u32 $0, $1, 0xffffffff;",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _umax(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_i32(a, loc, ip), _i32(b, loc, ip)],
            "max.u32 $0, $1, $2;",
            "=r,r,r",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _clear_eq(k: Int32, m: Int32, *, loc=None, ip=None) -> Int32:
    """k, or 0 where k == m (a taken key; valid keys are never 0)."""
    return Int32(
        _asm(
            T.i32(),
            [_i32(k, loc, ip), _i32(m, loc, ip)],
            "{\n\t.reg .pred ce_p;\n\tsetp.eq.u32 ce_p, $1, $2;\n\tselp.b32 $0, 0, $1, ce_p;\n\t}",
            "=r,r,r",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _topk_key(packed: Int32, half: int, expert: Int32, *, loc=None, ip=None) -> Int32:
    sh = "shr.b32 tk_b, $1, 16;" if half else "and.b32 tk_b, $1, 0xffff;"
    return Int32(
        _asm(
            T.i32(),
            [_i32(packed, loc, ip), _i32(expert, loc, ip)],
            "{\n\t.reg .b32 tk_b, tk_t, tk_x, tk_e;\n\t.reg .pred tk_n;\n\t"
            + sh
            + "\n\t"
            "and.b32 tk_t, tk_b, 0x8000;\n\tsetp.ne.u32 tk_n, tk_t, 0;\n\txor.b32 tk_x, tk_b, 0xffff;\n\t"
            "or.b32 tk_t, tk_b, 0x8000;\n\tselp.b32 tk_t, tk_x, tk_t, tk_n;\n\tshl.b32 tk_t, tk_t, 16;\n\t"
            "sub.u32 tk_e, 65535, $2;\n\tor.b32 $0, tk_t, tk_e;\n\t}",
            "=r,r,r",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_shared_v4_b32(addr: Int32, *, loc=None, ip=None):
    out = _asm(
        _struct(4),
        [_i32(addr, loc, ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r,~{memory}",
        loc=loc,
        ip=ip,
    )
    return _unpack(out, 4, loc, ip)


@dsl_user_op
def _tcgen05_after_sync(*, loc=None, ip=None) -> None:
    _asm(None, [], "tcgen05.fence::after_thread_sync;", "~{memory}", loc=loc, ip=ip)


@dsl_user_op
def _bulk_push(
    dst: Int32,
    mbar: Int32,
    src: Int32,
    rank: Int32,
    nbytes: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    _asm(
        None,
        [
            _i32(dst, loc, ip),
            _i32(mbar, loc, ip),
            _i32(src, loc, ip),
            _i32(rank, loc, ip),
            _i32(nbytes, loc, ip),
        ],
        "{\n\t.reg .b32 rd, rb;\n\tmapa.shared::cluster.u32 rd, $0, $3;\n\tmapa.shared::cluster.u32 rb, $1, $3;\n\t"
        "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [rd], [$2], $4, [rb];\n\t}",
        "r,r,r,r,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bulk_commit(*, loc=None, ip=None) -> None:
    _asm(None, [], "cp.async.bulk.commit_group;", "~{memory}", loc=loc, ip=ip)


@dsl_user_op
def _bulk_wait_read(*, loc=None, ip=None) -> None:
    _asm(None, [], "cp.async.bulk.wait_group.read 0;", "~{memory}", loc=loc, ip=ip)


@dsl_user_op
def _st_shared_b32(addr: Int32, value: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [_i32(addr, loc, ip), _i32(value, loc, ip)],
        "st.shared.b32 [$0], $1;",
        "r,r,~{memory}",
        loc=loc,
        ip=ip,
    )


class MoeRouterGemmKernel:
    threads = 128

    def __init__(
        self,
        *,
        num_rows: int,
        k_extent: int,
        num_experts: int,
        a_stride: int,
        splits: int = 16,
        token_tile: int = 128,
        use_pdl: bool = True,
        early_trigger: bool = False,
        emit_topk: bool = False,
        quant: bool = False,
    ) -> None:
        if num_experts % EXPERT_TILE:
            raise ValueError(
                f"num_experts={num_experts} must be a multiple of {EXPERT_TILE}"
            )
        if splits not in (8, 16):
            raise ValueError("splits must be 8 or 16 (cluster size)")
        if token_tile not in (32, 48, 64, 128):
            raise ValueError("token_tile must be 32, 48, 64 or 128")
        if k_extent % (splits * K_TILE):
            raise ValueError(
                f"k_extent={k_extent} must split into {splits} ranges of whole {K_TILE} tiles"
            )
        if a_stride % 8 or a_stride < k_extent:
            raise ValueError("x rows must be 16-byte aligned")
        self.num_rows = num_rows
        self.k_extent = k_extent
        self.num_experts = num_experts
        self.a_stride = a_stride
        self.splits = splits
        self.tn = token_tile
        self.token_tiles = -(-num_rows // token_tile)
        self.use_pdl = use_pdl
        self.early_trigger = early_trigger
        self.emit_topk = emit_topk
        self.quant = quant
        self.n_etiles = num_experts // EXPERT_TILE
        self.k_tiles = k_extent // splits // K_TILE
        if self.k_tiles < 4:
            raise ValueError(
                "the staging tile reuses the x buffers: at least 4 K tiles per split"
            )
        self.tps = token_tile // splits  # tokens of a CTA's output slice
        self.slice_bytes = self.tps * EXPERT_TILE * 4
        self.ctas = self.n_etiles * self.token_tiles * splits
        self.tmem_cols = 1 << (max(32, token_tile) - 1).bit_length()
        self.smem_bytes = (
            self.k_tiles * (EXPERT_TILE + token_tile) * K_TILE * 2
            + token_tile * EXPERT_TILE * 4
        )
        if quant and (token_tile != 32 or self.n_etiles * 8 != token_tile):
            raise ValueError(
                "quant takes 32-token tiles with 4 expert tiles (8 rows per CTA)"
            )
        if self.smem_bytes + 2048 > 232448:
            raise ValueError(f"shared memory {self.smem_bytes} B over budget")

    @cute.jit
    def __call__(
        self,
        x_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        out_ptr: Int64,
        cand_ptr: Int64 | None,
        gs_ptr: Int64 | None,
        xq_ptr: Int64 | None,
        xsf_ptr: Int64 | None,
        stream,
    ) -> None:
        mA = cute.make_tensor(
            w_ptr,
            cute.make_layout(
                (self.num_experts, self.k_extent, 1),
                stride=(self.k_extent, 1, self.num_experts * self.k_extent),
            ),
        )
        mB = cute.make_tensor(
            x_ptr,
            cute.make_layout(
                (self.num_rows, self.k_extent, 1),
                stride=(self.a_stride, 1, self.num_rows * self.a_stride),
            ),
        )
        a_major = utils.LayoutEnum.from_tensor(mA).mma_major_mode()
        b_major = utils.LayoutEnum.from_tensor(mB).mma_major_mode()
        tiled_mma = utils.sm100.make_trivial_tiled_mma(
            BFloat16,
            BFloat16,
            a_major,
            b_major,
            Float32,
            tcgen05.CtaGroup.ONE,
            (EXPERT_TILE, self.tn),
        )
        mma_inst_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_tiler = (EXPERT_TILE, self.tn, K_TILE)
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,)
        )
        a_smem_staged = utils.sm100.make_smem_layout_a(
            tiled_mma, mma_tiler, BFloat16, self.k_tiles
        )
        b_smem_staged = utils.sm100.make_smem_layout_b(
            tiled_mma, mma_tiler, BFloat16, self.k_tiles
        )
        a_smem_one = cute.slice_(a_smem_staged, (None, None, None, 0))
        b_smem_one = cute.slice_(b_smem_staged, (None, None, None, 0))
        a_op = utils.sm100.cluster_shape_to_tma_atom_A((1, 1), tiled_mma.thr_id)
        b_op = utils.sm100.cluster_shape_to_tma_atom_B((1, 1), tiled_mma.thr_id)
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, mA, a_smem_one, mma_tiler, tiled_mma, cluster_layout_vmnk.shape
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, mB, b_smem_one, mma_tiler, tiled_mma, cluster_layout_vmnk.shape
        )
        a_bytes = cute.size_in_bytes(BFloat16, a_smem_one)
        b_bytes = cute.size_in_bytes(BFloat16, b_smem_one)
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            out_ptr,
            cand_ptr,
            gs_ptr,
            xq_ptr,
            xsf_ptr,
            a_smem_staged,
            b_smem_staged,
            a_bytes + b_bytes,
            mma_inst_k,
        ).launch(
            grid=[self.ctas, 1, 1],
            block=[self.threads, 1, 1],
            cluster=[self.splits, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def _quant_phase(
        self,
        sB,
        full_mbar,
        gs_ptr: Int64,
        xq_ptr: Int64,
        xsf_ptr: Int64,
        s,
        e_tile,
        t_tile,
        tidx,
        phase: cutlass.Constexpr,
    ) -> None:
        tn: cutlass.Constexpr = self.tn
        r_loc = e_tile * Int32(8) + tidx // Int32(16)
        cbl = Int32(16 * phase) + tidx % Int32(16)
        kt = cbl // Int32(4)
        row = t_tile * Int32(tn) + r_loc
        if row < Int32(self.num_rows):
            gs = _ld_f32(gs_ptr)
            cute.arch.mbarrier_wait(full_mbar + kt, 0)
            c0 = (cbl % Int32(4)) * Int32(2)
            line = (
                _smem_addr(sB.iterator)
                + kt * Int32(tn * K_TILE * 2)
                + r_loc * Int32(128)
            )
            sw = r_loc % Int32(8)
            w0, w1, w2, w3 = _ld_shared_v4_b32(line + ((c0 ^ sw) * Int32(16)))
            w4, w5, w6, w7 = _ld_shared_v4_b32(
                line + (((c0 + Int32(1)) ^ sw) * Int32(16))
            )
            lo, hi, sf = _nvfp4_block(w0, w1, w2, w3, w4, w5, w6, w7, gs)
            cb = s * Int32(self.k_tiles * 4) + cbl
            _st_v2(
                xq_ptr + Int64(row) * Int64(self.k_extent // 2) + Int64(cb) * Int64(8),
                lo,
                hi,
            )
            _st_u8(xsf_ptr + Int64(row) * Int64(self.k_extent // 16) + Int64(cb), sf)

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB: cute.Tensor,
        out_ptr: Int64,
        cand_ptr: Int64 | None,
        gs_ptr: Int64 | None,
        xq_ptr: Int64 | None,
        xsf_ptr: Int64 | None,
        a_smem_staged: cute.ComposedLayout,
        b_smem_staged: cute.ComposedLayout,
        tx_bytes: cutlass.Constexpr,
        mma_inst_k: cutlass.Constexpr,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        splits: cutlass.Constexpr = self.splits
        k_tiles: cutlass.Constexpr = self.k_tiles
        tps: cutlass.Constexpr = self.tps
        tn: cutlass.Constexpr = self.tn
        slice_bytes: cutlass.Constexpr = self.slice_bytes
        s = bidx % Int32(splits)  # rank in the cluster (1-D clusters along x)
        tile = bidx // Int32(splits)
        e_tile = tile % Int32(self.n_etiles)
        t_tile = tile // Int32(self.n_etiles)

        if warp == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        @cute.struct
        class SharedStorage:
            full_mbar: cute.struct.MemRange[cutlass.Int64, k_tiles]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, 1]
            recv_mbar: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sA = smem.allocate_tensor(
            element_type=BFloat16,
            layout=a_smem_staged.outer,
            byte_alignment=1024,
            swizzle=a_smem_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=BFloat16,
            layout=b_smem_staged.outer,
            byte_alignment=1024,
            swizzle=b_smem_staged.inner,
        )
        recv = smem.allocate_tensor(
            Float32, cute.make_layout((tn * EXPERT_TILE,)), byte_alignment=128
        )
        full_mbar = storage.full_mbar.data_ptr()
        acc_mbar = storage.acc_mbar.data_ptr()
        recv_mbar = storage.recv_mbar.data_ptr()

        if warp == 0:
            with cute.arch.elect_one():
                for kt in cutlass.range_constexpr(k_tiles):
                    cute.arch.mbarrier_init(full_mbar + kt, 1)
                cute.arch.mbarrier_init(acc_mbar, 1)
                cute.arch.mbarrier_init(recv_mbar, 1)
                # This CTA's own arrive: one slice from every split of the cluster.
                cute.arch.mbarrier_arrive_and_expect_tx(recv_mbar, splits * slice_bytes)
            cute.arch.mbarrier_init_fence()
        if warp == 1:
            cute.arch.alloc_tmem(self.tmem_cols, storage.tmem_holding_buf)
        _cluster_arrive_release()
        cute.arch.sync_threads()

        gA = cute.local_tile(
            mA,
            cute.slice_((EXPERT_TILE, tn, K_TILE), (None, 0, None)),
            (None, None, None),
        )
        gB = cute.local_tile(
            mB,
            cute.slice_((EXPERT_TILE, tn, K_TILE), (0, None, None)),
            (None, None, None),
        )
        thr_mma = tiled_mma.get_slice(0)
        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tAgA_t = tAgA[(None, e_tile, None, 0)]
        tBgB_t = tBgB[(None, t_tile, None, 0)]
        k0 = s * Int32(k_tiles)

        # Producer: static weights before the dependency wait, x after it.
        if warp == 0:
            for kt in cutlass.range_constexpr(k_tiles):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(full_mbar + kt, tx_bytes)
                cute.copy(
                    tma_atom_a,
                    tAgA_t[(None, k0 + kt)],
                    tAsA[(None, kt)],
                    tma_bar_ptr=full_mbar + kt,
                )
        if const_expr(self.use_pdl):
            _gdc_wait()
            if const_expr(self.early_trigger):
                _gdc_launch()
        if warp == 0:
            for kt in cutlass.range_constexpr(k_tiles):
                cute.copy(
                    tma_atom_b,
                    tBgB_t[(None, k0 + kt)],
                    tBsB[(None, kt)],
                    tma_bar_ptr=full_mbar + kt,
                )

        # MMA warp: K tiles in order, 4 instructions each, one accumulator.
        tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, 16, storage.tmem_holding_buf)
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C((EXPERT_TILE, tn))
        tCtAcc = cute.make_tensor(tmem_ptr, tiled_mma.make_fragment_C(acc_shape).layout)
        if warp == 1:
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for kt in cutlass.range_constexpr(k_tiles):
                cute.arch.mbarrier_wait(full_mbar + kt, 0)
                for kb in cutlass.range_constexpr(K_TILE // mma_inst_k):
                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        tCrA[(None, None, kb, kt)],
                        tCrB[(None, None, kb, kt)],
                        tCtAcc,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            with cute.arch.elect_one():
                tcgen05.commit(acc_mbar)

        if const_expr(self.quant):
            self._quant_phase(
                sB, full_mbar, gs_ptr, xq_ptr, xsf_ptr, s, e_tile, t_tile, tidx, 0
            )
            cute.arch.sync_threads()
        cute.arch.mbarrier_wait(acc_mbar, 0)
        _tcgen05_after_sync()
        tmem_base = _ld_shared(_smem_addr(storage.tmem_holding_buf.ptr))
        taddr = tmem_base + ((warp * Int32(32)) << Int32(16))
        stage_addr = _smem_addr(sB.iterator)
        recv_addr = _smem_addr(recv.iterator)
        for c in cutlass.range_constexpr(tn // 16):
            words = _tmem_ld16(taddr + Int32(c * 16))
            for j in cutlass.range_constexpr(16):
                _st_shared_b32(
                    stage_addr + (Int32((c * 16 + j) * EXPERT_TILE) + tidx) * Int32(4),
                    words[j],
                )
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()
        _cluster_wait_acquire()  # every peer's receive mbarrier is initialized
        # Token slice o of the staging tile goes to slot s of CTA o's receive buffer.
        if warp == 0:
            with cute.arch.elect_one():
                for o in cutlass.range_constexpr(splits):
                    _bulk_push(
                        recv_addr + s * Int32(slice_bytes),
                        _smem_addr(recv_mbar),
                        stage_addr + Int32(o * slice_bytes),
                        Int32(o),
                        Int32(slice_bytes),
                    )
                _bulk_commit()
        if const_expr(self.quant):
            self._quant_phase(
                sB, full_mbar, gs_ptr, xq_ptr, xsf_ptr, s, e_tile, t_tile, tidx, 1
            )
        cute.arch.mbarrier_wait(recv_mbar, 0)

        lane = tidx % Int32(32)
        e4 = lane * Int32(4)
        for i in cutlass.range_constexpr(-(-tps // 4)):
            lt = warp + Int32(4 * i)
            if lt < Int32(tps):
                base = recv_addr + (lt * Int32(EXPERT_TILE) + e4) * Int32(4)
                a0, a1, a2, a3 = _ld_shared_v4_f32(base)
                for p in cutlass.range_constexpr(1, splits):
                    b0, b1, b2, b3 = _ld_shared_v4_f32(base + Int32(p * slice_bytes))
                    a0 = _add(a0, b0)
                    a1 = _add(a1, b1)
                    a2 = _add(a2, b2)
                    a3 = _add(a3, b3)
                row = t_tile * Int32(tn) + s * Int32(tps) + lt
                if row < Int32(self.num_rows):
                    dst = out_ptr + (
                        Int64(row) * Int64(self.num_experts)
                        + Int64(e_tile * Int32(EXPERT_TILE) + e4)
                    ) * Int64(2)
                    if const_expr(self.emit_topk):
                        p01 = _pack_bf16x2(a0, a1)
                        p23 = _pack_bf16x2(a2, a3)
                        _st_v2(dst, p01, p23)
                        eg = e_tile * Int32(EXPERT_TILE) + e4
                        key0 = _topk_key(p01, 0, eg)
                        key1 = _topk_key(p01, 1, eg + Int32(1))
                        key2 = _topk_key(p23, 0, eg + Int32(2))
                        key3 = _topk_key(p23, 1, eg + Int32(3))
                        mine = Int32(0)
                        for r in cutlass.range_constexpr(10):
                            m = _redux_max_u32(
                                _umax(_umax(key0, key1), _umax(key2, key3))
                            )
                            key0 = _clear_eq(key0, m)
                            key1 = _clear_eq(key1, m)
                            key2 = _clear_eq(key2, m)
                            key3 = _clear_eq(key3, m)
                            if lane == Int32(r):
                                mine = m
                        if lane < Int32(10):
                            _st_b32(
                                cand_ptr
                                + (Int64(row) * Int64(self.n_etiles) + Int64(e_tile))
                                * Int64(40)
                                + Int64(lane) * Int64(4),
                                mine,
                            )
                    else:
                        _st_v2(dst, _pack_bf16x2(a0, a1), _pack_bf16x2(a2, a3))
        if warp == 0:
            with cute.arch.elect_one():
                _bulk_wait_read()  # the staging tile may not go away under an outgoing copy
        cute.arch.sync_threads()
        if warp == 1:
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(tmem_ptr, self.tmem_cols)
        if const_expr(self.use_pdl and not self.early_trigger):
            _gdc_launch()


WIDE_MAX_ROWS, WIDE_CONFIG = 192, (8, 64)
MID_MAX_ROWS, MID_CONFIG = 144, (8, 48)


def router_gemm_config(m: int, mid_tiles: bool = True):
    if m <= 3 * 32:
        return 8, 32
    if mid_tiles and m <= MID_MAX_ROWS:
        return MID_CONFIG
    if m <= 2 * 64:
        return 8, 64
    if m <= WIDE_MAX_ROWS:
        return WIDE_CONFIG
    return None
