# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# Adapted from FlashInfer v0.7.0.post1 (Apache-2.0):
# flashinfer/gemm/kernels/dense_bf16_gemm_sm100_splitk.py
"""FlashInfer 0.7.0.post1 ``SplitKDenseGemmKernel`` (``dense_bf16_gemm_sm100_splitk``)
reduced to our use (PDL, no bias) with the weight (kernel-A) TMA issued through a
device tensor map with an L2 evict_first policy, and its synchronization fixed: the
owner waits for the peers' ``st.async`` partials at ``.acquire.cluster``; the MMA warp
waits for TMEM-pointer phase 0 before its release arrival (else its phase-1 wait can
pass before the epilogue has read TMEM); ``tcgen05.fence::*_thread_sync`` around
every mbarrier hand-off of tcgen05 work.

Each cluster rank accumulates an exact K slice in FP32; peers publish partials to
rank 0 through DSMEM, which reduces, casts and stores once. Kernel-M carries public N
and kernel-N public M.

``fenced_trigger`` ("fix A", default 0 = shipped: the weight warp, which never
waits, triggers after issuing its weight stages): the CTA's only PDL trigger is issued
by a thread that has executed ``griddepcontrol.wait``, behind ``fence.acq_rel.gpu``.
PTX 9.7.15.14: a release fence before ``launch_dependents`` in program order
synchronizes with the dependent grid's start (same memory sync domain, gpu scope), and
only a CTA's first trigger has an effect. So a dependent's pre-wait reads (the early KDA
verify's slot table, checkpoint and conv taps) are ordered after everything this grid's
wait orders. Where the trigger sits: 1 = the activation warp right after its wait (the
fence delays its first activation TMA); 2 = the idle warp 3 (wait, fence, trigger; same
time as 1, off the activation path); 3 = the weight warp after its weight stages, where
the shipped trigger sits (its wait then returns at once whenever K tiles > stages).

``act_fuse`` (default False = the shipped kernel, unchanged): a merged
gate_up weight [2I, K] (gate rows 0..I-1, up rows I..2I-1) and C = SiLU(gate) * up [rows, I].
CTA i loads gate rows [i*h, i*h+h) and up rows [I+i*h, I+i*h+h) (h = mma_m / 2) as two
half-height boxes into one A tile (no weight relayout); the owner rounds the reduced tile to
BF16 (the stock gate_up output), stages it in SMEM, and each epilogue thread computes
bf16(silu(gate) * up) with the sglang act_and_mul (--use_fast_math) instruction sequence."""


import functools

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Int32
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import tcgen05
from cutlass.cutlass_dsl import dsl_user_op
from ..common.primitives import asm
from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import (
    OWNER_RANK,
    SplitKTactic,
    _store_shared_remote_v4,
)

CTA_K, ATOM_K = 128, 64  # K per tile; K per 128-byte swizzle atom (one A box)
LARGE_SMEM_ATTRIBUTE = "CU_DEVICE_ATTRIBUTE_MAX_OVERSIZED_SHARED_MEMORY_PER_BLOCK"


@functools.lru_cache(maxsize=None)
def _large_smem_alloc(ordinal):
    attr = getattr(_cuda.CUdevice_attribute, LARGE_SMEM_ATTRIBUTE, None)
    if attr is None:
        raise RuntimeError(f"large-SMEM mode: this cuda-python has no {LARGE_SMEM_ATTRIBUTE}")
    status, value = _cuda.cuDeviceGetAttribute(attr, ordinal)
    if status != _cuda.CUresult.CUDA_SUCCESS or int(value) <= 0:
        raise RuntimeError(f"large-SMEM mode: {LARGE_SMEM_ATTRIBUTE} unavailable on device {ordinal} ({status}, {value})")
    return int(value)


def large_smem_alloc(device=None):
    """Per-block shared memory of the large-SMEM mode (the driver's MAX_OVERSIZED_SHARED_MEMORY_PER_BLOCK of ``device``,
    default the current CUDA device); usable only with the function's SHARED_MEMORY_MODE = ALLOW_OVERSIZED."""
    if device is None:
        import torch

        device = torch.cuda.current_device()
    return _large_smem_alloc(int(getattr(device, "index", device)))

__all__ = ["SplitKTactic", "SplitKGemm"]


def _asm(code, constraints, operands, loc, ip):
    llvm.inline_asm(
        None, operands, code, constraints, has_side_effects=True,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _acquire_tensor_map(tensor_map, *, loc=None, ip=None):
    code = "fence.proxy.tensormap::generic.acquire.sys [$0], 128; "
    _asm(code + "prefetch.tensormap [$0];", "l", [tensor_map.ir_value()], loc, ip)


@dsl_user_op
def _tma_load_evict_first(smem, mbar, tensor_map, k0, row0, *, loc=None, ip=None):
    """One {64, rows} box of A at (k0, row0), L2 evict_first."""
    code = "{ .reg .b64 pol; createpolicy.fractional.L2::evict_first.b64 pol, 1.0; "
    code += "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx"
    code += "::bytes.L2::cache_hint [$0], [$1, {$2, $3}], [$4], pol; }"
    operands = [smem.toint().ir_value(), tensor_map.ir_value(), k0.ir_value()]
    operands += [row0.ir_value(), mbar.toint().ir_value()]
    _asm(code, "r,l,r,r,r", operands, loc, ip)


@dsl_user_op
def _wait_cluster(mbar, phase, *, loc=None, ip=None):
    """``cute.arch.mbarrier_wait`` at ``.acquire.cluster`` (remote ``st.async``)."""
    code = "{ .reg .pred P1; LAB_WAIT: "
    code += "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [$0], $1, "
    code += "10000000; @P1 bra.uni DONE; bra.uni LAB_WAIT; DONE: }"
    _asm(code, "r,r", [mbar.toint().ir_value(), phase.ir_value()], loc, ip)


@dsl_user_op
def _arm_cluster(mbar, tx_bytes, *, loc=None, ip=None):
    """K-23 fix (always on): the owner arms
    bar_reduce after the cluster barrier, so peers' remote ``st.async`` complete_tx can land first; armed at
    CLUSTER scope the arm and those complete_tx are morally strong (PTX 8.7) and the owner's
    ``try_wait.acquire.cluster`` orders the peers' partials whichever order they take (litmus-tested: forbidden)."""
    code = "{ .reg .b64 st; mbarrier.arrive.expect_tx.relaxed.cluster.shared::cta.b64 st, [$0], $1; }"
    _asm(code, "r,r,~{memory}", [mbar.toint().ir_value(), Int32(tx_bytes).ir_value()], loc, ip)


@dsl_user_op
def _fenced_launch_dependents(*, loc=None, ip=None):
    """Acquire-release fence then trigger, one asm block so nothing separates them.
    ``fence.acq_rel.gpu`` (not ``.release``): the dependent's pre-wait
    reads then stay ordered even if ``griddepcontrol.wait`` is read as a relaxed
    observation of the predecessors' completion (litmus-tested)."""
    llvm.inline_asm(
        None, [], "fence.acq_rel.gpu;\n\tgriddepcontrol.launch_dependents;", "~{memory}",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )  # fmt: skip


# K-42 (fix A modes 1 / 2): the warp that triggers after its own PDL wait (the activation warp in mode 1, warp 3 in
# mode 2) first waits on this named barrier (64 threads), which the MMA warp arrives on once its tensor memory is
# allocated, so a dependent CTA on this SM cannot take tensor memory first. Modes 0 / 3 trigger from the weight warp,
# which is already behind the MMA warp's first commit (its bar_empty wait on tile ``stages``), and compile unchanged.
TRIGGER_BARRIER = 4


@dsl_user_op
def _trigger_barrier(arrive, *, loc=None, ip=None):
    op = "arrive" if arrive else "sync"
    llvm.inline_asm(
        None, [], f"bar.{op} {TRIGGER_BARRIER}, 64;", "~{memory}",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _fence_proxy_async_global(*, loc=None, ip=None):
    """K-16 (site KDA-1): the activation TMAs read B, which the PDL predecessor
    wrote with generic stores; the issuing warp orders the generic and async proxies for global memory
    after ``griddepcontrol.wait`` and before its first TMA (PTX 8.6 proxies, 9.7.15.4 fence.proxy)."""
    llvm.inline_asm(
        None, [], "fence.proxy.async.global;", "~{memory}",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _fence_tcgen05(*, before=False, loc=None, ip=None):
    side = "before" if before else "after"
    _asm(f"tcgen05.fence::{side}_thread_sync;", "", [], loc, ip)


@asm(cutlass.Float32, pure=True)
def _silu_mul_fast(gate, up):
    """sglang act_and_mul_kernel<bf16, kSiLU> (JIT, --use_fast_math): silu(g) = g / (1 + expf(-g)) -> nvcc's
    mul.ftz (g * -log2e), ex2.approx.ftz, add.ftz 1, div.approx.ftz; then mul.ftz by up (PTX of nvcc 12.6 -O3
    --use_fast_math; bitwise equality to the served kernel is a GPU test, not this comment)."""
    code = "{ .reg .f32 t,e,d,s; mul.ftz.f32 t,$1,0fbfb8aa3b; ex2.approx.ftz.f32 e,t; "
    code += "add.ftz.f32 d,e,0f3f800000; div.approx.ftz.f32 s,$1,d; mul.ftz.f32 $0,$2,s; }"
    return code, "=f,f,f", (gate, up)


def _smem(dtype, size, alignment, layout=None):
    layout = cute.make_layout(size) if layout is None else layout
    return cute_ext.allocate(dtype, cute.AddressSpace.smem, layout, alignment=alignment)


class SplitKGemm:
    """BF16 GEMM with a cluster-local split-K reduction; ``a_map`` is the device
    address of A's tensor map (box {64, mma_m}, 128-byte swizzle)."""

    def __init__(
        self, *, tactic: SplitKTactic, fenced_trigger: int = 0, proxy_fences: bool = False, act_fuse: bool = False,
        oversized: bool = False,
    ):  # fmt: skip
        assert int(fenced_trigger) in (0, 1, 2, 3), fenced_trigger
        self.fenced_trigger = int(fenced_trigger)
        # SiLU-mul epilogue over a merged gate_up weight (see the module doc); default off.
        self.act_fuse = bool(act_fuse)
        if self.act_fuse:
            assert tactic.mma_m in (64, 128) and tactic.mma_m * tactic.mma_n % (2 * 128) == 0, tactic
        # OV (default off): launch with the large-SMEM allocation so a tactic can hold more
        # weight stages before its PDL wait. Only the host launch's SMEM size changes; the caller sets the function's
        # SHARED_MEMORY_MODE to ALLOW_OVERSIZED before the first launch (projection.py).
        self.smem_alloc = large_smem_alloc() if oversized else utils.get_smem_capacity_in_bytes("sm_100")
        # K-16 fence on the B load (always on).
        self.k16_fence = True
        # K-23 (on since KDA-6): the owner's late bar_reduce arm at cluster scope.
        self.k23_arm = True
        self.acc_dtype = cutlass.Float32
        # griddepcontrol is not a proxy-preserving boundary (PTX 8.9.4): with proxy_fences the kernel fences
        # generic <-> async itself around its TMA reads of B and after its generic stores of C,
        # for producers / consumers that do not fence (stock kernels).
        self.proxy_fences = proxy_fences
        self.cta_m, self.cta_n, self.cta_k = tactic.mma_m, tactic.mma_n, CTA_K
        self.num_ab_stage, self.split_k = tactic.ab_stages, tactic.split_k
        self.threads_per_cta, self.epilog_threads = 256, 128
        self.mma_tiler_mn = (tactic.mma_m, tactic.mma_n)
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape = (1, tactic.split_k, 1)
        values_per_thread = tactic.mma_m * tactic.mma_n // self.epilog_threads
        assert values_per_thread % 4 == 0, "remote stores move 4 values"
        self.mailbox_elements = (
            (tactic.split_k - 1) * self.epilog_threads * values_per_thread
        )
        self.expected_transaction_bytes = self.mailbox_elements * 4

    @cute.experimental.jit
    def __call__(
        self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, bias: cute.Tensor,
        a_map: cutlass.Int64, stream: _cuda.CUstream,
    ):  # fmt: skip
        # Grid-y packs output-N tile and cluster rank. act_fuse: one CTA row per mma_m weight rows (C has half).
        m_rows = a.layout.shape[0] if cutlass.const_expr(self.act_fuse) else c.layout.shape[0]
        self.kernel(a, b, c, bias, a_map).launch(
            grid=(
                cute.ceil_div(m_rows, self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(self.smem_alloc),
            stream=stream,
            use_pdl=True,
        )

    @cute.experimental.kernel
    def kernel(
        self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, mBias: cute.Tensor,
        a_map: cutlass.Int64,
    ):  # fmt: skip
        """A (M, K, L) and B (N, K, L) K-major, C (M, N, L) M-major; bias unused."""
        stages = self.num_ab_stage
        ab_dtype = mA.element_type
        majors = (utils.LayoutEnum.from_tensor(t).mma_major_mode() for t in (mA, mB))
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype, ab_dtype, *majors, self.acc_dtype, self.cta_group,
            self.mma_tiler_mn,
        )  # fmt: skip

        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        block_idx = cute.arch.block_idx()
        bidx = block_idx[0]
        split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        n_idx = block_idx[1] // self.split_k
        l_idx = block_idx[2]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        shape = (tiled_mma, mnk_tiler, ab_dtype, stages)
        sA = _smem(ab_dtype, None, 1024, sm100_utils.make_smem_layout_a(*shape))
        sB = _smem(ab_dtype, None, 1024, sm100_utils.make_smem_layout_b(*shape))
        acc_layout = cute_ext.make_tmem_layout_acc(
            tiled_mma, self.mma_tiler_mn, acc_stage=1
        )
        bar_full = _smem(cutlass.Int64, stages, 8).iterator
        bar_empty = _smem(cutlass.Int64, stages, 8).iterator
        bar_tma_epilog = _smem(cutlass.Int64, 1, 8).iterator  # FlashInfer's bias
        bar_mma_epilog = _smem(cutlass.Int64, 1, 8).iterator
        bar_tmem_alloc = _smem(cutlass.Int64, 1, 8).iterator
        tmem_base_ptr = _smem(cutlass.Int32, 1, 4).iterator
        mailbox, bar_reduce = sA, bar_mma_epilog  # split 1: compile-time dead
        if cutlass.const_expr(self.split_k > 1):
            mailbox = _smem(cutlass.Float32, self.mailbox_elements, 128)
            bar_reduce = _smem(cutlass.Int64, 1, 8).iterator
        stage_d = sA  # act_fuse only: the owner's BF16 tile, (mma_m, mma_n) M-major
        if cutlass.const_expr(self.act_fuse):
            stage_d = _smem(
                cutlass.BFloat16, None, 16,
                cute.make_layout((self.cta_m, self.cta_n), stride=(1, self.cta_m)),
            )  # fmt: skip

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, 2)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_tma_epilog, 32)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)
                if cutlass.const_expr(self.split_k > 1):
                    # Owner arrival plus peer transaction-byte credits.
                    cute.arch.mbarrier_init(bar_reduce, 1)

        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.split_k > 1):
            cute.arch.cluster_arrive_relaxed()  # peers' barriers before their stores
        else:
            cute.arch.barrier()

        # K splits evenly over the cluster (tactics are validated for the shapes).
        k_tile_count = cute.size(mA, mode=[1]) // self.cta_k // self.split_k
        k_tile_start = split_rank * k_tile_count

        if cutlass.const_expr(self.split_k > 1):
            cute.arch.cluster_wait()

        # Warp 3 is idle; warps 4-7 run the epilogue.
        tiles = (k_tile_start, k_tile_count)
        if warp_idx == 0:
            if cutlass.const_expr(self.act_fuse):
                half = self.cta_m // 2
                up0 = cute.size(mA, mode=[0]) // 2 + bidx * half
                self.weight_warp_pair(bar_full, bar_empty, sA, a_map, bidx * half, up0, *tiles)
            else:
                self.weight_warp(bar_full, bar_empty, sA, a_map, bidx * self.cta_m, *tiles)
        elif warp_idx == 1:
            b_tile = cute.local_tile(mB, (self.cta_n, self.cta_k), (n_idx, None, l_idx))
            b_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")
            self.activation_warp(bar_full, bar_empty, b_tile, sB, b_map, *tiles)
        elif warp_idx == 2:
            k_blocks = self.cta_k // cute.size(tiled_mma.shape_mnk, mode=[2])
            self.mma_warp(
                bar_full, bar_empty, bar_mma_epilog, bar_tmem_alloc, tiled_mma, sA, sB,
                tmem_base_ptr, acc_layout, k_blocks, k_tile_count,
            )  # fmt: skip
        elif warp_idx >= 4:
            if cutlass.const_expr(self.act_fuse):
                tile = cute.local_tile(mC, (self.cta_m // 2, self.cta_n), (bidx, n_idx, l_idx))
                rows_left = cute.size(mC, mode=[1]) - n_idx * self.cta_n
                self.epilog_warp_act(
                    bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr, acc_layout, tile,
                    cute.arch.thread_idx()[0] - 128, mC.element_type,
                    utils.LayoutEnum.from_tensor(mC), mailbox, bar_reduce, split_rank, stage_d, rows_left,
                )  # fmt: skip
            else:
                tile = cute.local_tile(mC, (self.cta_m, self.cta_n), (bidx, n_idx, l_idx))
                self.epilog_warp(
                    bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr, acc_layout, tile,
                    cute.arch.thread_idx()[0] - 128, mC.element_type,
                    utils.LayoutEnum.from_tensor(mC), mailbox, bar_reduce, split_rank,
                )  # fmt: skip
        # Fix A mode 2: the idle warp 3 waits, fences and triggers. Compiled in only for
        # mode 2, outside the dispatch chain, so every other mode keeps the shipped chain
        # (default off: PTX identity, FX-3).
        if cutlass.const_expr(self.fenced_trigger == 2):
            if warp_idx == 3:
                cute.arch.griddepcontrol_wait()
                _trigger_barrier(False)  # K-42: after the MMA warp's TMEM alloc
                _fenced_launch_dependents()

    @cute.experimental.jit
    def weight_warp(
        self, bar_full, bar_empty, s_tile: cute.Tensor, a_map: cutlass.Int64,
        row0: cutlass.Int32, k_tile_start: cutlass.Int32, k_tile_count: cutlass.Int32,
    ):  # fmt: skip
        """A: two evict_first boxes per K tile, before the PDL wait (weights)."""
        stages = self.num_ab_stage
        stage_bytes = cute.size_in_bytes(
            s_tile.element_type, cute.slice_(s_tile.layout, (None, None, None, 0))
        )
        _acquire_tensor_map(a_map)
        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(bar_full + stage, stage_bytes)
            with cute.arch.elect_one():
                k0 = (k_tile_start + k_tile) * self.cta_k
                tile = s_tile[None, None, None, stage].iterator
                for atom in cutlass.range_constexpr(self.cta_k // ATOM_K):
                    box, mbar = tile + atom * self.cta_m * ATOM_K, bar_full + stage
                    k = k0 + atom * ATOM_K
                    _tma_load_evict_first(box, mbar, a_map, k, row0)
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1

        if cutlass.const_expr(self.fenced_trigger == 3):
            cute.arch.griddepcontrol_wait()
            _fenced_launch_dependents()
        elif cutlass.const_expr(self.fenced_trigger == 0):
            cute.arch.griddepcontrol_launch_dependents()
        self.drain(bar_empty, empty_phase, k_tile_count)

    @cute.experimental.jit
    def weight_warp_pair(
        self, bar_full, bar_empty, s_tile: cute.Tensor, a_map: cutlass.Int64,
        gate0: cutlass.Int32, up0: cutlass.Int32, k_tile_start: cutlass.Int32, k_tile_count: cutlass.Int32,
    ):  # fmt: skip
        """act_fuse A: per 64-K atom, a {64, mma_m/2} gate box into tile rows [0, h) and an up box into [h, 2h)
        (a_map's box is {64, mma_m/2}; h rows x 128 B = a whole number of 1024 B swizzle atoms, so each half keeps
        the full box's SW128 phase). Same stage bytes, same evict_first policy, same PDL trigger as weight_warp."""
        stages = self.num_ab_stage
        half = self.cta_m // 2
        stage_bytes = cute.size_in_bytes(
            s_tile.element_type, cute.slice_(s_tile.layout, (None, None, None, 0))
        )
        _acquire_tensor_map(a_map)
        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(bar_full + stage, stage_bytes)
            with cute.arch.elect_one():
                k0 = (k_tile_start + k_tile) * self.cta_k
                tile = s_tile[None, None, None, stage].iterator
                for atom in cutlass.range_constexpr(self.cta_k // ATOM_K):
                    box, mbar = tile + atom * self.cta_m * ATOM_K, bar_full + stage
                    k = k0 + atom * ATOM_K
                    _tma_load_evict_first(box, mbar, a_map, k, gate0)
                    _tma_load_evict_first(box + half * ATOM_K, mbar, a_map, k, up0)
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1

        if cutlass.const_expr(self.fenced_trigger == 3):
            cute.arch.griddepcontrol_wait()
            _fenced_launch_dependents()
        elif cutlass.const_expr(self.fenced_trigger == 0):
            cute.arch.griddepcontrol_launch_dependents()
        self.drain(bar_empty, empty_phase, k_tile_count)

    @cute.experimental.jit
    def activation_warp(
        self, bar_full, bar_empty, g_tile: cute.Tensor, s_tile: cute.Tensor,
        cta_v_map: cute.Layout, k_tile_start: cutlass.Int32,
        k_tile_count: cutlass.Int32,
    ):  # fmt: skip
        """B (the activations): after the PDL wait."""
        stages = self.num_ab_stage
        stage_bytes = cute.size_in_bytes(
            s_tile.element_type, cute.slice_(s_tile.layout, (None, None, None, 0))
        )
        cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.k16_fence):  # K-16: B was written generically upstream; TMA reads it
            _fence_proxy_async_global()
        if cutlass.const_expr(self.fenced_trigger == 1):
            _trigger_barrier(False)  # K-42: after the MMA warp's TMEM alloc
            _fenced_launch_dependents()
        if cutlass.const_expr(self.proxy_fences):
            _fence_proxy_async_global()  # D1: predecessor grids' generic stores -> these TMA reads
        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(bar_full + stage, stage_bytes)
            cute_ext.tma_load(
                g_tile[None, None, k_tile_start + k_tile],
                s_tile[None, None, None, stage],
                (bar_full + stage).value,
                cta_v_map=cta_v_map,
                tma_operation_type=cute_ext.OperationTypeEnum.SM90_TMA_LOAD,
                update_expect_tx=False,
            )
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1
        self.drain(bar_empty, empty_phase, k_tile_count)
        if cutlass.const_expr(self.proxy_fences):
            _fence_proxy_async_global()  # D2: these consumed TMA reads -> later grids' generic overwrites

    @cute.experimental.jit
    def drain(self, bar_empty, empty_phase: cutlass.Int32, k_tile_count: cutlass.Int32):
        stages = self.num_ab_stage
        for tail in cutlass.range(stages, unroll=1):
            stage = (tail + k_tile_count) % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1

    @cute.experimental.jit
    def mma_warp(
        self, bar_full, bar_empty, bar_mma_epilog, bar_tmem_alloc,
        tiled_mma: cute.TiledMma, sA: cute.Tensor, sB: cute.Tensor, tmem_base_ptr,
        acc_layout: cutlass.Constexpr, mma_inst_tile_k: cutlass.Constexpr,
        k_tile_count: cutlass.Int32,
    ):  # fmt: skip
        num_tmem_cols = 256
        cute.arch.alloc_tmem(num_tmem_cols, tmem_base_ptr, is_two_cta=False)
        _fence_tcgen05(before=True)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        if cutlass.const_expr(self.fenced_trigger in (1, 2)):  # K-42: the triggering warp waits for this
            _trigger_barrier(True)
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)

        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        accumulator = cute.make_tensor(tmem_ptr, acc_layout)[None, None, None, 0]
        mma_atom = cute.make_mma_atom(tiled_mma.op)
        full_phase = cutlass.Int32(0)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % self.num_ab_stage
            cute.arch.mbarrier_wait(bar_full + stage, full_phase)
            _fence_tcgen05()
            for k_block in range(mma_inst_tile_k):
                accumulate = k_tile != 0 if k_block == 0 else True
                mma_atom.set(tcgen05.Field.ACCUMULATE, accumulate)
                cute_ext.dot(
                    mma_atom,
                    cute.append_ones(sA[None, None, k_block, stage], up_to_rank=3),
                    cute.append_ones(sB[None, None, k_block, stage], up_to_rank=3),
                    accumulator,
                )
            with cute.arch.elect_one():
                tcgen05.commit(bar_empty + stage, None, self.cta_group)
            if stage == self.num_ab_stage - 1:
                full_phase = full_phase ^ 1

        with cute.arch.elect_one():
            tcgen05.commit(bar_mma_epilog, None, self.cta_group)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)  # the release lands in phase 1
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 1)
        _fence_tcgen05()
        cute.arch.dealloc_tmem(tmem_ptr, num_tmem_cols, is_two_cta=False)

    @cute.experimental.jit
    def epilog_warp(
        self, bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr,
        acc_layout: cutlass.Constexpr, gD_tile: cute.Tensor, epi_tid: cutlass.Int32,
        c_dtype: cutlass.Constexpr, d_layout: cutlass.Constexpr, mailbox, bar_reduce,
        split_rank: cutlass.Int32,
    ):  # fmt: skip
        # Wait until MMA publishes the TMEM base pointer.
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        _fence_tcgen05()

        acc_view = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr),
            acc_layout,
        )[((None, None), 0, 0, 0)]
        epi_tile = (self.cta_m, self.cta_n)
        load_op = sm100_utils.get_tmem_load_op(
            (self.cta_m, self.cta_n, self.cta_k), d_layout, c_dtype, self.acc_dtype,
            epi_tile, False,
        )  # fmt: skip
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(load_op, acc_view)
        gD_epi = cute.flat_divide(gD_tile, epi_tile)

        # Match each epilogue thread's TMEM partition in RMEM.
        rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, gD_epi, epi_tid)
        rmem = cute.AddressSpace.rmem
        rAcc = cute_ext.allocate(self.acc_dtype, rmem, rmem_layout, alignment=32)
        rD = cute_ext.allocate(c_dtype, rmem, rmem_layout, alignment=32)
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)

        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        _fence_tcgen05()
        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
        # Make tcgen05.ld visible before TMEM release and RMEM use.
        cute.arch.fence_view_async_tmem_load()
        _fence_tcgen05(before=True)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)

        # Peers publish FP32 partials; only rank 0 reduces and stores.
        if cutlass.const_expr(self.split_k > 1):
            vpt = cutlass.const_expr(cute.size(rmem_layout))  # values per thread
            vpp = cutlass.const_expr(self.epilog_threads * vpt)  # values per peer
            if split_rank != OWNER_RANK:
                for i in cutlass.range_constexpr(0, vpt, 4):
                    words = (rAcc[i], rAcc[i + 1], rAcc[i + 2], rAcc[i + 3])
                    slot = mailbox.iterator + (split_rank - Int32(1)) * vpp
                    slot = slot + epi_tid * vpt + i
                    _store_shared_remote_v4(*words, slot, bar_reduce, Int32(OWNER_RANK))
            else:
                if epi_tid == 0:
                    expected = self.expected_transaction_bytes
                    if cutlass.const_expr(self.k23_arm):
                        _arm_cluster(bar_reduce, expected)
                    else:
                        cute.arch.mbarrier_arrive_and_expect_tx(bar_reduce, expected)
                _wait_cluster(bar_reduce, Int32(0))
                for peer in cutlass.range_constexpr(self.split_k - 1):
                    for i in cutlass.range_constexpr(vpt):
                        rAcc[i] = rAcc[i] + mailbox[peer * vpp + epi_tid * vpt + i]

        if split_rank == OWNER_RANK:
            rD.store(rAcc.load().to(c_dtype))
            # Preserve TMEM coordinates; the copy predicates output tails.
            cute_ext.partition_and_copy(thr_t2r, rD, gD_epi[None, None, 0, 0])
            if cutlass.const_expr(self.proxy_fences):
                _fence_proxy_async_global()  # D3: these generic stores -> successors' TMA reads

    @cute.experimental.jit
    def epilog_warp_act(
        self, bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr,
        acc_layout: cutlass.Constexpr, gO_tile: cute.Tensor, epi_tid: cutlass.Int32,
        c_dtype: cutlass.Constexpr, d_layout: cutlass.Constexpr, mailbox, bar_reduce,
        split_rank: cutlass.Int32, stage_d: cute.Tensor, rows_left: cutlass.Int32,
    ):  # fmt: skip
        """epilog_warp up to the owner's BF16 cast (same TMEM load, same rank-order reduction, same cast); then the
        tile goes to SMEM, the 4 epilogue warps sync (named barrier 1), and thread t computes outputs t, t + 128, ...
        of the (mma_m / 2, mma_n) act tile: j = idx % h (feature), n = idx // h (row); gate = tile[j, n],
        up = tile[h + j, n]. Rows past the tensor's end are not stored."""
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        _fence_tcgen05()

        acc_view = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr),
            acc_layout,
        )[((None, None), 0, 0, 0)]
        epi_tile = (self.cta_m, self.cta_n)
        load_op = sm100_utils.get_tmem_load_op(
            (self.cta_m, self.cta_n, self.cta_k), d_layout, c_dtype, self.acc_dtype,
            epi_tile, False,
        )  # fmt: skip
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(load_op, acc_view)
        sD_epi = cute.flat_divide(stage_d, epi_tile)

        rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, sD_epi, epi_tid)
        rmem = cute.AddressSpace.rmem
        rAcc = cute_ext.allocate(self.acc_dtype, rmem, rmem_layout, alignment=32)
        rD = cute_ext.allocate(c_dtype, rmem, rmem_layout, alignment=32)
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)

        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        _fence_tcgen05()
        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
        cute.arch.fence_view_async_tmem_load()
        _fence_tcgen05(before=True)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)

        if cutlass.const_expr(self.split_k > 1):
            vpt = cutlass.const_expr(cute.size(rmem_layout))
            vpp = cutlass.const_expr(self.epilog_threads * vpt)
            if split_rank != OWNER_RANK:
                for i in cutlass.range_constexpr(0, vpt, 4):
                    words = (rAcc[i], rAcc[i + 1], rAcc[i + 2], rAcc[i + 3])
                    slot = mailbox.iterator + (split_rank - Int32(1)) * vpp
                    slot = slot + epi_tid * vpt + i
                    _store_shared_remote_v4(*words, slot, bar_reduce, Int32(OWNER_RANK))
            else:
                if epi_tid == 0:
                    expected = self.expected_transaction_bytes
                    if cutlass.const_expr(self.k23_arm):
                        _arm_cluster(bar_reduce, expected)
                    else:
                        cute.arch.mbarrier_arrive_and_expect_tx(bar_reduce, expected)
                _wait_cluster(bar_reduce, Int32(0))
                for peer in cutlass.range_constexpr(self.split_k - 1):
                    for i in cutlass.range_constexpr(vpt):
                        rAcc[i] = rAcc[i] + mailbox[peer * vpp + epi_tid * vpt + i]

        if split_rank == OWNER_RANK:
            rD.store(rAcc.load().to(c_dtype))
            cute_ext.partition_and_copy(thr_t2r, rD, sD_epi[None, None, 0, 0])
            # The 128 epilogue threads (warps 4-7) only: every staged value before any read of it.
            cute.arch.barrier(barrier_id=1, number_of_threads=self.epilog_threads)
            half = self.cta_m // 2
            for i in cutlass.range_constexpr(half * self.cta_n // self.epilog_threads):
                idx = epi_tid + i * self.epilog_threads
                j = idx % half
                n = idx // half
                if n < rows_left:
                    gate = stage_d[j, n].to(cutlass.Float32)
                    up = stage_d[half + j, n].to(cutlass.Float32)
                    gO_tile[j, n] = _silu_mul_fast(gate, up).to(c_dtype)
            if cutlass.const_expr(self.proxy_fences):
                _fence_proxy_async_global()  # D3: these generic stores -> successors' TMA reads
