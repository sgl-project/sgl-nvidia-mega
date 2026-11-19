# SPDX-License-Identifier: Apache-2.0
# The copied methods: FlashInfer v0.7.3 gdn_kernels/blackwell with FlashInfer #8854's
# hunks (Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES, BSD-3-Clause).
# From SGLang python/sglang/kernels/ops/mamba/gdn_prefill_tmem_dealloc.py. Changed:
# install() leaves the CP launcher alone (the executor declines CP sharding).
"""FlashInfer #8854 (56d946ed1, "order TMEM dealloc after the compute warps' tcgen05.ld in the Blackwell GDN
kernels"; after v0.7.3 and v0.7.4rc1, in no release yet) for FlashInfer v0.7.3's Blackwell GDN prefill kernels.
In each kernel one warp frees TMEM while the other warps of its group may still have tcgen05.ld in flight: the
fix has every reading warp drain its TMEM loads (fence_view_async_tmem_load) and meet at a named barrier before
relinquish_alloc_permit / free. Each class below subclasses v0.7.3's and overrides only ``__init__`` (the barrier)
and ``kernel`` (the fence and the barrier), both copied from the v0.7.3 release with #8854's hunks applied as is
(they apply to 0.7.3 without offsets); every other method, the launchers and the dispatch stay FlashInfer's.
Outputs are bitwise the stock kernels' (the change adds a fence and a barrier only). Regenerate from the release
and #8854's diff rather than edit.

This module: the non-CP chunked kernel (``GatedDeltaNetChunkedKernel``, used for every multi-sequence
prefill on SM10x, and its SM107 subclass ``RubinGatedDeltaNetChunkedKernel``, which 0.7.3 builds on Rubin and
which inherits ``kernel``) and ``install()``, which rebinds the kernel classes in FlashInfer's launcher
module (``gdn_kernels/blackwell/gdn_prefill.py``) so that FlashInfer's own launcher builds the fixed classes,
under their own CuTe-DSL disk-cache name.
"""

# ruff: noqa: F821  -- the copied methods read FlashInfer's module globals, bound below

import hashlib as _hashlib
import logging as _logging

from flashinfer.gdn_kernels.blackwell import gated_delta_net_chunked as _fi_chunked
from flashinfer.gdn_kernels.blackwell import (
    gated_delta_net_chunked_sm107 as _fi_chunked_sm107,
)
from flashinfer.gdn_kernels.blackwell import gdn_prefill as _fi_prefill

FI_FILES_SHA256 = {
    "flashinfer/gdn_kernels/blackwell/gated_delta_net_chunked.py": "906173728b8134af20808e64cefe4a200839831d21d03dc8ab04565d5af96016",
    "flashinfer/gdn_kernels/blackwell/gated_delta_net_chunked_sm107.py": "bb62ffeb6d1f428ee5df03dabc0da609d74d0597e5ed214081b0e015d5955e1e",
    "flashinfer/gdn_kernels/blackwell/gdn_prefill.py": "0068d71f55cc5028f91fe1537e9d7b42143f4ab1cb089ded2d1307ee73438ae6",
}
_CUTE_DSL_MODULE = "sglang_gdn_blackwell_prefill_tmem_dealloc"
_logger = _logging.getLogger(__name__)


def _check_file(module, path: str) -> None:
    """The copied methods run against the exact FlashInfer file they were copied from."""
    with open(module.__file__, "rb") as fh:
        got = _hashlib.sha256(fh.read()).hexdigest()
    if got != FI_FILES_SHA256[path]:
        raise ImportError(
            f"{module.__file__} is not FlashInfer v0.7.3's {path} (sha256 {got[:16]})"
        )


_check_file(_fi_chunked, "flashinfer/gdn_kernels/blackwell/gated_delta_net_chunked.py")
_check_file(
    _fi_chunked_sm107,
    "flashinfer/gdn_kernels/blackwell/gated_delta_net_chunked_sm107.py",
)
_check_file(_fi_prefill, "flashinfer/gdn_kernels/blackwell/gdn_prefill.py")

# The copied methods resolve the source module's globals (cute, cutlass, pipeline, tcgen05, GDNTileScheduler, ...).
globals().update(
    {
        k: v
        for k, v in vars(_fi_chunked).items()
        if not (k.startswith("__") and k.endswith("__"))
    }
)


class GatedDeltaNetChunkedKernelTmemDealloc(_fi_chunked.GatedDeltaNetChunkedKernel):
    """``GatedDeltaNetChunkedKernel`` (v0.7.3) with FlashInfer #8854's TMEM dealloc ordering."""

    def __init__(
        self,
        io_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        state_dtype: Type[cutlass.Numeric],
        mma_tiler_qk: Tuple[int, int, int],
        mma_tiler_qs: Tuple[int, int, int],
        mma_tiler_qkv: Tuple[int, int, int],
        mma_tiler_kv: Tuple[int, int, int],
        max_active_clusters: int,
        num_sm: int,
        is_GQA: bool,
        use_initial_state: bool,
        inverse_dtype: Type[cutlass.Numeric] = cutlass.Float16,
        store_final_state: bool = True,
        enable_checkpoints: bool = False,
        is_persistent: bool = True,
        state_stride_divisibility: int = 1,
        use_output_state_indices: bool = False,
        use_checkpoint_state_indices: bool = False,
        use_prefix_lens: bool = False,
    ):
        if state_stride_divisibility < 1 or (
            state_stride_divisibility & (state_stride_divisibility - 1)
        ):
            raise ValueError(
                "state_stride_divisibility must be a positive power of two, got "
                f"{state_stride_divisibility}"
            )
        if (use_checkpoint_state_indices or use_prefix_lens) and not (
            enable_checkpoints
        ):
            raise ValueError(
                "checkpoint_state_indices and prefix_lens require checkpoints"
            )
        self.io_dtype = io_dtype
        self.acc_dtype = acc_dtype
        self.state_dtype = state_dtype
        self.mma_tiler_qk = mma_tiler_qk
        self.mma_tiler_qs = mma_tiler_qs
        self.mma_tiler_qkv = mma_tiler_qkv
        self.mma_tiler_kv = mma_tiler_kv
        self.max_active_clusters = max_active_clusters
        self.num_sm = num_sm
        self.is_GQA = is_GQA
        self.inverse_dtype = inverse_dtype
        self.use_initial_state = use_initial_state
        self.store_final_state = store_final_state
        self.enable_checkpoints = enable_checkpoints
        self.is_persistent = is_persistent
        # Divisibility of the state slot/head/V strides; 1 = unknown.
        self.state_stride_divisibility = state_stride_divisibility
        # Slot addressing for a paged state cache (all off for packed calls):
        # separate final-state rows, checkpoint pool rows, and absolute
        # checkpoint positions. With any of them, a negative read row starts
        # from the zero state and a negative write row skips that store.
        self.use_output_state_indices = use_output_state_indices
        self.use_checkpoint_state_indices = use_checkpoint_state_indices
        self.use_prefix_lens = use_prefix_lens
        self.slot_mode = (
            use_output_state_indices or use_checkpoint_state_indices or use_prefix_lens
        )

        # ------------------------------------------------------------------
        # Warp assignments  (12 warps total)
        # ------------------------------------------------------------------
        # T-pairwise / kk_epi / qk_epi / inverse
        self.compute_group_0_warp_ids = [0, 1, 2, 3]
        # kv_decay_v / v-k*state / epi ops
        self.compute_group_1_warp_ids = [4, 5, 6, 7]
        self.mma_warp_id = 8
        self.tma_qkv_warp_id = 9
        # The second issuer owns the five state/output GEMMs.
        self.mma_cg1_warp_id = 10
        # store O
        self.epilogue_warp_id = 11
        # The lightly loaded O epilogue warp also prefetches gate/beta.
        self.load_gate_beta_warp_id = self.epilogue_warp_id

        # Give the MMA/TMA/gate/epilogue warps enough registers to keep their
        # pipeline handles resident. Fund the increase from CG1 first while
        # preserving the existing 64,512-register CTA budget.
        self.num_regs_compute_group_0 = 224
        self.num_regs_compute_group_1 = 256
        self.num_regs_other = 24
        if not self.use_initial_state:
            # The peeled zero-state MMA carries more pipeline cursors.  Transfer
            # twenty-four registers from each CG1 warp to each lightweight warp while
            # retaining the same 64,512-register CTA allocation.
            self.num_regs_compute_group_1 = 232
            self.num_regs_other = 48

        self.threads_per_cta = 32 * (
            len(
                (
                    self.mma_warp_id,
                    self.tma_qkv_warp_id,
                    self.mma_cg1_warp_id,
                    self.epilogue_warp_id,
                )
            )
            + len(self.compute_group_0_warp_ids)
            + len(self.compute_group_1_warp_ids)
        )

        self.use_2cta_instrs = False
        self.cluster_shape_mnk = (1, 1, 1)
        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )
        self.occupancy = 1
        self.threads_per_warp = 32
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols(self.arch)

        # ------------------------------------------------------------------
        # Named barriers - only TmemAllocator requires a NamedBarrier;
        # all other inter-warp synchronization uses mbarrier-based pipelines
        # created inside kernel().
        # ------------------------------------------------------------------
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_warp
            * len(
                (
                    self.mma_warp_id,
                    self.mma_cg1_warp_id,
                    *self.compute_group_0_warp_ids,
                    *self.compute_group_1_warp_ids,
                )
            ),
        )
        self.inverse_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_warp * len(self.compute_group_0_warp_ids),
        )
        # All CG1 warps must have drained their tcgen05.ld before warp 4
        # deallocates TMEM.
        self.tmem_dealloc_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.threads_per_warp * len(self.compute_group_1_warp_ids),
        )
        self.init_state_store_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=self.threads_per_warp * len(self.compute_group_1_warp_ids),
        )

    @cute.kernel
    def kernel(
        self,
        # Tiled MMAs (one per logical GEMM group)
        # GEMM 1 (kk) + GEMM 2 (qk)
        tiled_mma_qk: cute.TiledMma,
        # GEMM 3 (k*state) + GEMM 4 (q*state)
        tiled_mma_qs: cute.TiledMma,
        # GEMM 5 (new_v) + GEMM 6 (qkv)
        tiled_mma_qkv: cute.TiledMma,
        # GEMM 5 (new_v: A_inv@V) first tile
        tiled_mma_qkv_ss: cute.TiledMma,
        # GEMM 7 (kv_update)
        tiled_mma_kv: cute.TiledMma,
        # TMA descriptors and cute tensors
        tma_q: TmaInfo,
        tma_k: TmaInfo,
        tma_v: TmaInfo,
        mGate: cute.Tensor,
        mBeta: cute.Tensor,
        tma_o: TmaInfo,
        # (B+1,)  int32  cumulative seq lengths
        cu_seqlens: cute.Tensor,
        # initial state (fp32) from GMEM; None if not used
        mS_init: Optional[cute.Tensor],
        # final state output (fp32) to GMEM; None if not stored
        mS_out: Optional[cute.Tensor],
        mS_indices: Optional[cute.Tensor],
        mS_checkpoints: Optional[cute.Tensor],
        cu_checkpoints: Optional[cute.Tensor],
        # (B,) final-state rows; None = mS_indices rows (or packed)
        mS_out_indices: Optional[cute.Tensor],
        # checkpoint pool rows, indexed by the running checkpoint entry
        mCheckpoint_indices: Optional[cute.Tensor],
        # (B,) tokens processed before this call (absolute checkpoint grid)
        mPrefix_lens: Optional[cute.Tensor],
        checkpoint_every_n_tokens: cutlass.Int32,
        scale: cutlass.Float32,
        # SMEM staged layouts (needed to view shared_storage tensor buffers)
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        k_trans_smem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        cumsumlog_smem_layout_staged: cute.Layout,
        beta_smem_layout_staged: cute.Layout,
        ainv_smem_layout_staged: cute.ComposedLayout,
        ainv_cal_smem_layout_staged: cute.ComposedLayout,
        qk_smem_layout_staged: cute.ComposedLayout,
        o_smem_layout_staged: cute.ComposedLayout,
        # Scheduler
        scheduler_params: GDNTileSchedulerParams,
        # TMA descriptor workspace in GMEM (one q/k/v/o descriptor set per CTA)
        # Slots: Q=0, K=1, V=2, O=3
        mQ,
        mK,
        mV,
        # used for TMA descriptor update
        mO,
        # (num_ctas, num_tensormaps, bytes_per_tensormap) Int8 workspace
        tensormap_workspace: cute.Tensor,
    ):
        """
        Main GDN chunked kernel.

        Warp specialization is the outermost control flow: each warp role owns
        its own persistent tile-scheduler loop, iterating over (batch, head)
        tiles and then over chunks within each tile.
        """
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, bidy, bidz = cute.arch.block_idx()
        grid_dim = cute.arch.grid_dim()

        if cutlass.const_expr(self.use_initial_state):
            assert mS_init is not None, (
                "mS_init must be provided if use_initial_state is True"
            )
        else:
            assert mS_init is None, "mS_init must be None if use_initial_state is False"
        if cutlass.const_expr(self.store_final_state):
            assert mS_out is not None, (
                "mS_out must be provided if store_final_state is True"
            )
        else:
            assert mS_out is None, "mS_out must be None if store_final_state is False"

        # ------------------------------------------------------------------
        # TMA descriptor GMEM workspace - one q/k/v/o descriptor set per CTA.
        # Slots: Q=0, K=1, V=2, O=3.
        # ------------------------------------------------------------------
        cta_linear_idx = bidz * grid_dim[1] * grid_dim[0] + bidy * grid_dim[0] + bidx

        tensormap_manager = TensorMapManager(
            TensorMapUpdateMode.GMEM, self.bytes_per_tensormap
        )

        tensormap_workspace = self.initialize_workspace(tensormap_workspace, grid_dim)
        tensormap_q_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_workspace[(cta_linear_idx, 0, None)].iterator
        )
        tensormap_k_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_workspace[(cta_linear_idx, 1, None)].iterator
        )
        tensormap_v_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_workspace[(cta_linear_idx, 2, None)].iterator
        )
        tensormap_o_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_workspace[(cta_linear_idx, 3, None)].iterator
        )

        # ------------------------------------------------------------------
        # 1. Allocate SMEM / TMEM, prefetch TMA descriptors
        # ------------------------------------------------------------------
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        sQ = storage.sQ.get_tensor(
            q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner
        )
        sK = storage.sK.get_tensor(
            k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner
        )
        sK_trans = storage.sK.get_tensor(
            k_trans_smem_layout_staged.outer, swizzle=k_trans_smem_layout_staged.inner
        )
        sV = storage.sV.get_tensor(
            v_smem_layout_staged.outer, swizzle=v_smem_layout_staged.inner
        )
        # A_inverse MMA input.
        sAinv = storage.sAinv.get_tensor(
            ainv_smem_layout_staged.outer, swizzle=ainv_smem_layout_staged.inner
        )
        if cutlass.const_expr(self.inverse_dtype == self.io_dtype):
            sAinvCal = sAinv
        else:
            sAinvCal = storage.sAinvCal.get_tensor(
                ainv_cal_smem_layout_staged.outer,
                swizzle=ainv_cal_smem_layout_staged.inner,
            )
        # QK output / O store  (W_qk first, then O epilogue staging)
        sQk = storage.sQk.get_tensor(
            qk_smem_layout_staged.outer, swizzle=qk_smem_layout_staged.inner
        )
        sO = storage.sO.get_tensor(
            o_smem_layout_staged.outer, swizzle=o_smem_layout_staged.inner
        )
        # Gate scalar arrays (1D Float32, flat - no swizzle)
        sCumsumlog = storage.cumsumlog.get_tensor(cumsumlog_smem_layout_staged)
        sCumprod = storage.cumprod.get_tensor(cumsumlog_smem_layout_staged)
        sBeta = storage.beta.get_tensor(beta_smem_layout_staged)

        if warp_idx == self.mma_warp_id:
            cpasync.prefetch_descriptor(tma_q.atom)
            cpasync.prefetch_descriptor(tma_k.atom)
            cpasync.prefetch_descriptor(tma_v.atom)
            cpasync.prefetch_descriptor(tma_o.atom)

        # TMEM allocator object - CG1 will issue the actual allocation
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            # CG1 owns allocation and is the last group to release TMEM state.
            allocator_warp_id=self.compute_group_1_warp_ids[0],
        )

        # ------------------------------------------------------------------
        # mbarrier-based pipelines
        # Each pipeline is created by all threads; barrier_storage points into SMEM.
        # defer_sync=True means pipeline_init_arrive() flushes all at once below.
        # ------------------------------------------------------------------
        def _cg(num_threads):
            return pipeline.CooperativeGroup(pipeline.Agent.Thread, num_threads)

        # 1 thread (TMA issuer)
        cg_tma = _cg(len([self.tma_qkv_warp_id]))
        # 1 warp (gate/beta scalar load warp)
        cg_gate = _cg(self.threads_per_warp * len([self.load_gate_beta_warp_id]))
        # One producer per result pipeline; K/Q are consumed by both issuers.
        cg_mma = _cg(len([self.mma_warp_id]))
        cg_mma_both = _cg(len([self.mma_warp_id, self.mma_cg1_warp_id]))
        # 128 threads (CG0)
        cg_cg0 = _cg(self.threads_per_warp * len(self.compute_group_0_warp_ids))
        # 128 threads (CG1)
        cg_cg1 = _cg(self.threads_per_warp * len(self.compute_group_1_warp_ids))
        # 4 threads (one per CG1 warp, used for V load signaling)
        cg_cg1_v = _cg(len(self.compute_group_1_warp_ids))
        # 256 threads (CG0 + CG1)
        cg_both = _cg(
            self.threads_per_warp * len(self.compute_group_0_warp_ids)
            + self.threads_per_warp * len(self.compute_group_1_warp_ids)
        )
        # 32 threads (epilogue warp)
        cg_epi = _cg(self.threads_per_warp * len([self.epilogue_warp_id]))

        # TMA load pipelines: K/Q feed MMA; V is signaled to CG1 for ALU work.
        load_k_producer, load_k_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.smem_k_stages,
            producer_group=cg_tma,
            consumer_group=cg_mma_both,
            tx_count=self.tma_k_bytes,
            barrier_storage=storage.load_k_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        load_q_producer, load_q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.smem_q_stages,
            producer_group=cg_tma,
            consumer_group=cg_mma_both,
            tx_count=self.tma_q_bytes,
            barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        load_v_producer, load_v_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.smem_v_stages,
            producer_group=cg_tma,
            consumer_group=cg_cg1_v,
            tx_count=self.tma_v_bytes,
            barrier_storage=storage.load_v_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        # Unified loader warp 9 -> CG0 / CG1: gate/beta (software-signaled)
        # Scalar-load paths do not use TMA barriers; producer calls commit() after writes.
        load_gate_producer, load_gate_consumer = pipeline.PipelineAsync.create(
            num_stages=self.smem_gate_stages,
            producer_group=cg_gate,
            consumer_group=cg_both,
            barrier_storage=storage.load_gate_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        load_beta_producer, load_beta_consumer = pipeline.PipelineCpAsync.create(
            num_stages=self.smem_beta_stages,
            producer_group=cg_gate,
            consumer_group=cg_cg0,
            barrier_storage=storage.load_beta_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # MMA warp -> CG1:  kv_acc
        kv_acc_producer, kv_acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.tmem_kv_acc_stages,
            producer_group=cg_mma,
            consumer_group=cg_cg1,
            barrier_storage=storage.kv_acc_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # MMA warp -> CG1:  q_state_acc
        q_state_acc_producer, q_state_acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.tmem_q_state_acc_stages,
            producer_group=cg_mma,
            consumer_group=cg_cg1,
            barrier_storage=storage.q_state_acc_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # MMA warp -> CG0: KK/QK accumulator ring.
        cg0_shared_acc_producer, cg0_shared_acc_consumer = (
            pipeline.PipelineUmmaAsync.create(
                num_stages=self.tmem_cg0_shared_acc_stages,
                producer_group=cg_mma,
                consumer_group=cg_cg0,
                barrier_storage=storage.cg0_shared_acc_mbar_ptr.data_ptr(),
                defer_sync=True,
            ).make_participants()
        )

        # MMA warp -> CG1: KS/NV accumulator ring.
        cg1_shared_acc_producer, cg1_shared_acc_consumer = (
            pipeline.PipelineUmmaAsync.create(
                num_stages=self.tmem_cg1_shared_acc_stages,
                producer_group=cg_mma,
                consumer_group=cg_cg1,
                barrier_storage=storage.cg1_shared_acc_mbar_ptr.data_ptr(),
                defer_sync=True,
            ).make_participants()
        )

        # CG0 -> MMA warp:  a_inv_done
        a_inv_ready_producer, a_inv_ready_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.smem_ainv_stages,
            producer_group=cg_cg0,
            consumer_group=cg_mma,
            barrier_storage=storage.ainv_ready_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # CG0 -> MMA warp:  qk_done
        qk_ready_producer, qk_ready_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.smem_qk_stages,
            producer_group=cg_cg0,
            consumer_group=cg_mma,
            barrier_storage=storage.qk_ready_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # CG1 -> MMA warp: state input.
        state_inp_ready_producer, state_inp_ready_consumer = (
            pipeline.PipelineAsyncUmma.create(
                num_stages=self.tmem_state_inp_stages,
                producer_group=cg_cg1,
                consumer_group=cg_mma,
                barrier_storage=storage.state_inp_ready_mbar_ptr.data_ptr(),
                defer_sync=True,
            ).make_participants()
        )

        # CG1 -> MMA warp: fixed-slot, ready-only input notifications.
        vks_ready_producer, vks_ready_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=cg_cg1,
            consumer_group=cg_mma,
            barrier_storage=storage.vks_ready_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        nv_ready_producer, nv_ready_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=cg_cg1,
            consumer_group=cg_mma,
            barrier_storage=storage.nv_ready_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        decay_v_ready_producer, decay_v_ready_consumer = (
            pipeline.PipelineAsyncUmma.create(
                num_stages=1,
                producer_group=cg_cg1,
                consumer_group=cg_mma,
                barrier_storage=storage.decay_v_ready_mbar_ptr.data_ptr(),
                defer_sync=True,
            ).make_participants()
        )

        # CG1 -> epilogue warp:  output_ready
        o_store_producer, o_store_consumer = pipeline.PipelineAsync.create(
            num_stages=self.smem_o_stages,
            producer_group=cg_cg1,
            consumer_group=cg_epi,
            barrier_storage=storage.o_store_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        pipeline_init_arrive(is_relaxed=True)

        pipeline_init_wait()

        # ------------------------------------------------------------------
        # 2. Warp specialization - each warp role owns its own scheduler loop
        # ------------------------------------------------------------------

        # ==============================================================
        # COMPUTE WARP GROUP 0 (warps 0-3)
        # ==============================================================
        if (
            warp_idx >= self.compute_group_0_warp_ids[0]
            and warp_idx <= self.compute_group_0_warp_ids[-1]
        ):
            cute.arch.setmaxregister_increase(self.num_regs_compute_group_0)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            scheduler = GDNTileScheduler.create(
                scheduler_params, (bidx, bidy, bidz), grid_dim
            )
            work = scheduler.initial_work_tile_info()
            while work.is_valid_tile:
                batch_idx, head_idx, _ = work.tile_idx
                batch_start = cu_seqlens[batch_idx]
                seqlen_b = cutlass.Int32(cu_seqlens[batch_idx + 1] - batch_start)
                if cutlass.const_expr(self.use_prefix_lens):
                    seqlen_b += self._lead_rows(mPrefix_lens, batch_idx, seqlen_b)
                num_pairs = 2
                num_pairs_b = cute.ceil_div(seqlen_b, self.b_t * num_pairs)

                sQk_pisl = self._transform_to_position_independent_layout(
                    sQk, qk_smem_layout_staged.inner
                )
                sAinv_pisl = self._transform_to_position_independent_layout(
                    sAinv, ainv_smem_layout_staged.inner
                )
                cg0_smem_args = (
                    sCumsumlog,
                    sBeta,
                    sAinv_pisl,
                    sAinvCal,
                    sQk_pisl,
                )
                # Runtime first/last predicates keep one pair body in SASS instead
                # of peeling a complete first-pair copy before the steady loop.
                for pair_idx in cutlass.range(num_pairs_b):
                    (
                        load_gate_consumer,
                        load_beta_consumer,
                        cg0_shared_acc_consumer,
                        a_inv_ready_producer,
                        qk_ready_producer,
                    ) = self.compute_group_0_pair(
                        tidx,
                        tmem_ptr,
                        scale,
                        (tiled_mma_qk,),
                        cg0_smem_args,
                        (
                            load_gate_consumer,
                            load_beta_consumer,
                            cg0_shared_acc_consumer,
                            a_inv_ready_producer,
                            qk_ready_producer,
                        ),
                        (pair_idx == 0, pair_idx < num_pairs_b - 1),
                    )

                scheduler.advance_to_next_work()
                work = scheduler.get_current_work()
            a_inv_ready_producer.tail()
            qk_ready_producer.tail()

        # ==============================================================
        # COMPUTE WARP GROUP 1 (warps 4-7)
        # ==============================================================
        if (
            warp_idx >= self.compute_group_1_warp_ids[0]
            and warp_idx <= self.compute_group_1_warp_ids[-1]
        ):
            cute.arch.setmaxregister_increase(self.num_regs_compute_group_1)
            # Total TMEM columns: state + q_state_acc + shared_acc + input staging.
            tmem.allocate(self.tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            scheduler = GDNTileScheduler.create(
                scheduler_params, (bidx, bidy, bidz), grid_dim
            )
            work = scheduler.initial_work_tile_info()
            while work.is_valid_tile:
                batch_idx, head_idx, _ = work.tile_idx
                batch_start = cu_seqlens[batch_idx]
                seqlen_b = cutlass.Int32(cu_seqlens[batch_idx + 1] - batch_start)
                if cutlass.const_expr(self.use_prefix_lens):
                    seqlen_b += self._lead_rows(mPrefix_lens, batch_idx, seqlen_b)
                num_chunks_b = cute.ceil_div(seqlen_b, self.b_t)
                if num_chunks_b > 0:
                    checkpoint_offset = 0
                    if cutlass.const_expr(self.enable_checkpoints):
                        checkpoint_offset = cu_checkpoints[batch_idx]
                    # Slot mode: the absolute position (mod N) of the first chunk
                    # start, the checkpoint pool rows and the final-state rows.
                    slot_checkpoint_args = ()
                    slot_final_args = None
                    if cutlass.const_expr(self.slot_mode):
                        checkpoint_phase = None
                        if cutlass.const_expr(self.use_prefix_lens):
                            checkpoint_phase = self._checkpoint_phase(
                                mPrefix_lens, batch_idx, checkpoint_every_n_tokens
                            )
                        checkpoint_cursor = None
                        if cutlass.const_expr(self.enable_checkpoints):
                            checkpoint_cursor = self._init_checkpoint_cursor(
                                mCheckpoint_indices,
                                checkpoint_offset,
                                checkpoint_phase,
                                checkpoint_every_n_tokens,
                                seqlen_b,
                            )
                        slot_checkpoint_args = (  # type: ignore[assignment]
                            checkpoint_cursor,
                            mCheckpoint_indices,
                        )
                        slot_final_args = (
                            mS_out_indices,
                            checkpoint_cursor,
                            mCheckpoint_indices,
                        )
                    if cutlass.const_expr(self.use_initial_state):
                        kv_acc_producer = self._load_initial_state(
                            tidx,
                            mS_init,
                            mS_indices,
                            head_idx,
                            batch_idx,
                            tmem_ptr,
                            tiled_mma_kv,
                            kv_acc_producer,
                        )
                    sV_pisl = self._transform_to_position_independent_layout(
                        sV, v_smem_layout_staged.inner
                    )
                    sO_pisl = self._transform_to_position_independent_layout(
                        sO, o_smem_layout_staged.inner
                    )
                    num_pairs_b = cute.ceil_div(seqlen_b, self.b_t * 2)
                    num_chunks_padded = num_pairs_b * 2
                    is_first_chunk = True
                    for chunk_iter in cutlass.range(num_chunks_padded):
                        (
                            load_v_consumer,
                            load_gate_consumer,
                            cg1_shared_acc_consumer,
                            kv_acc_consumer,
                            q_state_acc_consumer,
                            kv_acc_producer,
                            state_inp_ready_producer,
                            vks_ready_producer,
                            nv_ready_producer,
                            decay_v_ready_producer,
                            o_store_producer,
                            checkpoint_offset,
                        ) = self.compute_group_1_chunk(
                            tidx,
                            tmem_ptr,
                            scale,
                            (tiled_mma_kv, tiled_mma_qs, tiled_mma_qkv),
                            (sV_pisl, sCumsumlog, sCumprod, sBeta, sO_pisl),
                            (
                                mS_checkpoints,
                                checkpoint_offset,
                                checkpoint_every_n_tokens,
                            )
                            + slot_checkpoint_args,
                            (
                                load_v_consumer,
                                load_gate_consumer,
                                cg1_shared_acc_consumer,
                                kv_acc_consumer,
                                q_state_acc_consumer,
                                kv_acc_producer,
                                state_inp_ready_producer,
                                vks_ready_producer,
                                nv_ready_producer,
                                decay_v_ready_producer,
                                o_store_producer,
                            ),
                            (
                                chunk_iter,
                                num_pairs_b,
                                head_idx,
                                seqlen_b,
                                is_first_chunk,
                            ),
                        )
                        is_first_chunk = False
                    if cutlass.const_expr(
                        self.store_final_state or self.enable_checkpoints
                    ):
                        kv_acc_consumer = self._store_final_state(
                            tidx,
                            mS_out,
                            mS_indices,
                            head_idx,
                            batch_idx,
                            tmem_ptr,
                            tiled_mma_kv,
                            kv_acc_consumer,
                            seqlen_b,
                            mS_checkpoints,
                            checkpoint_offset,
                            checkpoint_every_n_tokens,
                            slot_final_args,
                        )
                    else:
                        kv_acc_handle = kv_acc_consumer.wait_and_advance()
                        kv_acc_handle.release()
                elif cutlass.const_expr(
                    self.store_final_state
                    and self.use_initial_state
                    and not self.use_output_state_indices
                ):
                    # With output_state_indices an empty sequence writes
                    # nothing: row output_state_indices[b] keeps its contents.
                    self._store_empty_final_state(
                        tidx,
                        mS_init,
                        mS_out,
                        mS_indices,
                        head_idx,
                        batch_idx,
                    )

                scheduler.advance_to_next_work()
                work = scheduler.get_current_work()

            # Drain this warp's TMEM loads, then wait for all CG1 warps before
            # warp 4 deallocates the TMEM they were reading from.
            cute.arch.fence_view_async_tmem_load()
            self.tmem_dealloc_barrier.arrive_and_wait()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

            o_store_producer.tail()
            state_inp_ready_producer.tail()

        # ==============================================================
        # CG0 MMA ISSUER (warp 8: KK/QK)
        # ==============================================================
        elif warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            scheduler = GDNTileScheduler.create(
                scheduler_params, (bidx, bidy, bidz), grid_dim
            )
            work = scheduler.initial_work_tile_info()
            while work.is_valid_tile:
                batch_idx, _, _ = work.tile_idx
                batch_start = cu_seqlens[batch_idx]
                seqlen_b = cutlass.Int32(cu_seqlens[batch_idx + 1] - batch_start)
                if cutlass.const_expr(self.use_prefix_lens):
                    seqlen_b += self._lead_rows(mPrefix_lens, batch_idx, seqlen_b)
                num_pairs_b = cute.ceil_div(seqlen_b, self.b_t * 2)
                for _pair_idx in cutlass.range(num_pairs_b):
                    (
                        cg0_shared_acc_producer,
                        load_k_consumer,
                        load_q_consumer,
                    ) = self.mma_cg0_pair(
                        tmem_ptr,
                        (tiled_mma_qk, tiled_mma_qkv),
                        (sQ, sK),
                        (
                            cg0_shared_acc_producer,
                            load_k_consumer,
                            load_q_consumer,
                        ),
                    )
                scheduler.advance_to_next_work()
                work = scheduler.get_current_work()

            cg0_shared_acc_producer.tail()

        # ==============================================================
        # CG1 MMA ISSUER (warp 10: KS/QS/NV/QKV/KV)
        # ==============================================================
        elif warp_idx == self.mma_cg1_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            scheduler = GDNTileScheduler.create(
                scheduler_params, (bidx, bidy, bidz), grid_dim
            )
            work = scheduler.initial_work_tile_info()
            while work.is_valid_tile:
                batch_idx, _, _ = work.tile_idx
                batch_start = cu_seqlens[batch_idx]
                seqlen_b = cutlass.Int32(cu_seqlens[batch_idx + 1] - batch_start)
                if cutlass.const_expr(self.use_prefix_lens):
                    seqlen_b += self._lead_rows(mPrefix_lens, batch_idx, seqlen_b)
                num_chunks = cute.ceil_div(seqlen_b, self.b_t * 2) * 2

                run_cg1_mma = num_chunks > 0
                if cutlass.const_expr(self.use_initial_state):
                    run_cg1_mma = True
                if run_cg1_mma:
                    first_loop_chunk = 0
                    if cutlass.const_expr(not self.use_initial_state):
                        (
                            cg1_shared_acc_producer,
                            q_state_acc_producer,
                            kv_acc_producer,
                            load_k_consumer,
                            load_q_consumer,
                            a_inv_ready_consumer,
                            qk_ready_consumer,
                            state_inp_ready_consumer,
                            vks_ready_consumer,
                            nv_ready_consumer,
                            decay_v_ready_consumer,
                        ) = self.mma_cg1_chunk(
                            tmem_ptr,
                            (
                                tiled_mma_qs,
                                tiled_mma_qkv,
                                tiled_mma_qkv_ss,
                                tiled_mma_kv,
                            ),
                            (sQ, sK, sK_trans, sV, sAinv, sQk),
                            (
                                cg1_shared_acc_producer,
                                q_state_acc_producer,
                                kv_acc_producer,
                                load_k_consumer,
                                load_q_consumer,
                                a_inv_ready_consumer,
                                qk_ready_consumer,
                                state_inp_ready_consumer,
                                vks_ready_consumer,
                                nv_ready_consumer,
                                decay_v_ready_consumer,
                            ),
                            True,
                        )
                        first_loop_chunk = 1

                    for chunk_idx in cutlass.range(first_loop_chunk, num_chunks):
                        is_first_chunk = False
                        if cutlass.const_expr(self.use_initial_state):
                            is_first_chunk = chunk_idx == 0
                        (
                            cg1_shared_acc_producer,
                            q_state_acc_producer,
                            kv_acc_producer,
                            load_k_consumer,
                            load_q_consumer,
                            a_inv_ready_consumer,
                            qk_ready_consumer,
                            state_inp_ready_consumer,
                            vks_ready_consumer,
                            nv_ready_consumer,
                            decay_v_ready_consumer,
                        ) = self.mma_cg1_chunk(
                            tmem_ptr,
                            (
                                tiled_mma_qs,
                                tiled_mma_qkv,
                                tiled_mma_qkv_ss,
                                tiled_mma_kv,
                            ),
                            (sQ, sK, sK_trans, sV, sAinv, sQk),
                            (
                                cg1_shared_acc_producer,
                                q_state_acc_producer,
                                kv_acc_producer,
                                load_k_consumer,
                                load_q_consumer,
                                a_inv_ready_consumer,
                                qk_ready_consumer,
                                state_inp_ready_consumer,
                                vks_ready_consumer,
                                nv_ready_consumer,
                                decay_v_ready_consumer,
                            ),
                            is_first_chunk,
                        )

                scheduler.advance_to_next_work()
                work = scheduler.get_current_work()

            cg1_shared_acc_producer.tail()
            q_state_acc_producer.tail()
            kv_acc_producer.tail()

        # ==============================================================
        # TMA LOAD WARP (warp 9)
        # ==============================================================
        elif warp_idx == self.tma_qkv_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            scheduler = GDNTileScheduler.create(
                scheduler_params, (bidx, bidy, bidz), grid_dim
            )
            work = scheduler.initial_work_tile_info()

            # Init base descriptors once into GMEM (copies embedded atom descriptor)
            if work.is_valid_tile:
                tensormap_manager.init_tensormap_from_atom(
                    tma_q.atom, tensormap_q_ptr, self.tma_qkv_warp_id
                )
                tensormap_manager.init_tensormap_from_atom(
                    tma_k.atom, tensormap_k_ptr, self.tma_qkv_warp_id
                )
                tensormap_manager.init_tensormap_from_atom(
                    tma_v.atom, tensormap_v_ptr, self.tma_qkv_warp_id
                )
                tensormap_manager.fence_tensormap_initialization()

            while work.is_valid_tile:
                batch_idx, head_idx, _ = work.tile_idx
                batch_start = cu_seqlens[batch_idx]
                batch_end = cu_seqlens[batch_idx + 1]
                seqlen_b = cutlass.Int32(batch_end - batch_start)
                if cutlass.const_expr(self.use_prefix_lens):
                    lead = self._lead_rows(mPrefix_lens, batch_idx, seqlen_b)
                    real_seqlen_b = seqlen_b
                    seqlen_b += lead
                num_pairs = 2
                num_pairs_b = cute.ceil_div(seqlen_b, self.b_t * num_pairs)
                num_chunks_b = num_pairs_b * num_pairs

                # All warp roles skip empty workloads at the same granularity.
                if num_chunks_b > 0:
                    if cutlass.const_expr(self.use_prefix_lens):
                        # Descriptors span exactly the sequence's own rows, so
                        # the leading identity rows (negative coordinates) and
                        # the padded tail both read zero. The row strides are
                        # static ints, so the start is widened before the
                        # multiply: an Int32 product wraps for a sequence that
                        # starts at 2**31 / (H * D) tokens or later.
                        seq_start = cutlass.Int64(batch_start)
                        bounded_q = cute.make_tensor(
                            mQ.iterator + seq_start * mQ.stride[0],
                            cute.make_layout(
                                (real_seqlen_b, mQ.shape[1], mQ.shape[2]),
                                stride=(mQ.stride[0], mQ.stride[1], mQ.stride[2]),
                            ),
                        )
                        bounded_k = cute.make_tensor(
                            mK.iterator + seq_start * mK.stride[0],
                            cute.make_layout(
                                (real_seqlen_b, mK.shape[1], mK.shape[2]),
                                stride=(mK.stride[0], mK.stride[1], mK.stride[2]),
                            ),
                        )
                        bounded_v = cute.make_tensor(
                            mV.iterator + seq_start * mV.stride[1],
                            cute.make_layout(
                                (mV.shape[0], real_seqlen_b, mV.shape[2]),
                                stride=(mV.stride[0], mV.stride[1], mV.stride[2]),
                            ),
                        )
                        first_row = -lead
                    else:
                        # Bounded descriptors zero-fill partial and padded chunks.
                        bounded_q = cute.make_tensor(
                            mQ.iterator,
                            cute.make_layout(
                                (batch_end, mQ.shape[1], mQ.shape[2]),
                                stride=(mQ.stride[0], mQ.stride[1], mQ.stride[2]),
                            ),
                        )
                        bounded_k = cute.make_tensor(
                            mK.iterator,
                            cute.make_layout(
                                (batch_end, mK.shape[1], mK.shape[2]),
                                stride=(mK.stride[0], mK.stride[1], mK.stride[2]),
                            ),
                        )
                        bounded_v = cute.make_tensor(
                            mV.iterator,
                            cute.make_layout(
                                (mV.shape[0], batch_end, mV.shape[2]),
                                stride=(mV.stride[0], mV.stride[1], mV.stride[2]),
                            ),
                        )
                        first_row = batch_start
                    tensormap_manager.update_tensormap(
                        (bounded_q, bounded_k, bounded_v),
                        (tma_q.atom, tma_k.atom, tma_v.atom),
                        (tensormap_q_ptr, tensormap_k_ptr, tensormap_v_ptr),
                        self.tma_qkv_warp_id,
                        (None, None, None),
                    )

                    num_valid_chunks_b = cute.ceil_div(seqlen_b, self.b_t)
                    for chunk_idx in cutlass.range(num_valid_chunks_b - 1):
                        chunk_offset = first_row + chunk_idx * self.b_t
                        load_q_producer, load_k_producer, load_v_producer = (
                            self.tma_qkv_warp(
                                (tiled_mma_qk, tiled_mma_qkv_ss, tiled_mma_kv),
                                (tma_q, tma_k, tma_v),
                                (sQ, sK, sV),
                                (load_q_producer, load_k_producer, load_v_producer),
                                (chunk_offset, chunk_idx, batch_idx, head_idx),
                                (
                                    tensormap_manager,
                                    tensormap_q_ptr,
                                    tensormap_k_ptr,
                                    tensormap_v_ptr,
                                ),
                            )
                        )
                    chunk_idx = num_valid_chunks_b - 1
                    chunk_offset = first_row + chunk_idx * self.b_t
                    load_q_producer, load_k_producer, load_v_producer = (
                        self.tma_qkv_warp(
                            (tiled_mma_qk, tiled_mma_qkv_ss, tiled_mma_kv),
                            (tma_q, tma_k, tma_v),
                            (sQ, sK, sV),
                            (load_q_producer, load_k_producer, load_v_producer),
                            (chunk_offset, chunk_idx, batch_idx, head_idx),
                            (
                                tensormap_manager,
                                tensormap_q_ptr,
                                tensormap_k_ptr,
                                tensormap_v_ptr,
                            ),
                        )
                    )
                    for chunk_idx in cutlass.range(num_valid_chunks_b, num_chunks_b):
                        chunk_offset = first_row + chunk_idx * self.b_t
                        load_q_producer, load_k_producer, load_v_producer = (
                            self.tma_qkv_warp(
                                (tiled_mma_qk, tiled_mma_qkv_ss, tiled_mma_kv),
                                (tma_q, tma_k, tma_v),
                                (sQ, sK, sV),
                                (load_q_producer, load_k_producer, load_v_producer),
                                (chunk_offset, chunk_idx, batch_idx, head_idx),
                                (
                                    tensormap_manager,
                                    tensormap_q_ptr,
                                    tensormap_k_ptr,
                                    tensormap_v_ptr,
                                ),
                            )
                        )
                scheduler.advance_to_next_work()
                work = scheduler.get_current_work()

            load_q_producer.tail()
            load_k_producer.tail()
            load_v_producer.tail()

        # ==============================================================
        # EPILOGUE WARP (warp 11)
        # ==============================================================
        if warp_idx == self.epilogue_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            scheduler = GDNTileScheduler.create(
                scheduler_params, (bidx, bidy, bidz), grid_dim
            )
            work = scheduler.initial_work_tile_info()

            # Init O descriptor once into GMEM
            if work.is_valid_tile:
                tensormap_manager.init_tensormap_from_atom(
                    tma_o.atom, tensormap_o_ptr, self.epilogue_warp_id
                )
                tensormap_manager.fence_tensormap_initialization()

            while work.is_valid_tile:
                batch_idx, head_idx, _ = work.tile_idx
                batch_start = cu_seqlens[batch_idx]
                batch_end = cu_seqlens[batch_idx + 1]
                seqlen_b = cutlass.Int32(batch_end - batch_start)
                if cutlass.const_expr(self.use_prefix_lens):
                    lead = self._lead_rows(mPrefix_lens, batch_idx, seqlen_b)
                    real_seqlen_b = seqlen_b
                    seqlen_b += lead
                num_pairs = 2
                num_pairs_b = cute.ceil_div(seqlen_b, self.b_t * num_pairs)
                num_chunks_b = num_pairs_b * num_pairs

                # Keep the fixed prefetch and zero-trip store loop under one
                # guard. The wider scope lets the compiler share the dynamic
                # chunk predicate without introducing divergent loop state.
                if num_chunks_b > 0:
                    num_valid_chunks_b = cute.ceil_div(seqlen_b, self.b_t)
                    if cutlass.const_expr(self.use_prefix_lens):
                        # The descriptor spans exactly the sequence's own rows:
                        # stores of the leading identity rows (negative
                        # coordinates) and of the padded tail are dropped. The
                        # start is widened before the multiply, as for Q/K/V.
                        bounded_o = cute.make_tensor(
                            mO.iterator + cutlass.Int64(batch_start) * mO.stride[1],
                            cute.make_layout(
                                (mO.shape[0], real_seqlen_b, mO.shape[2]),
                                stride=(mO.stride[0], mO.stride[1], mO.stride[2]),
                            ),
                        )
                        o_first_row = -lead
                        # Gate/beta rows are absolute; the first tile predicates
                        # the leading rows to the neutral gate 1 / beta 0.
                        gate_first_row = batch_start - lead
                        gate_row_lo = (batch_start,)
                    else:
                        bounded_o = cute.make_tensor(
                            mO.iterator,
                            cute.make_layout(
                                (mO.shape[0], batch_end, mO.shape[2]),
                                stride=(mO.stride[0], mO.stride[1], mO.stride[2]),
                            ),
                        )
                        o_first_row = batch_start
                        gate_first_row = batch_start
                        gate_row_lo = ()  # type: ignore[assignment]
                    tensormap_manager.update_tensormap(
                        (bounded_o,),
                        (tma_o.atom,),
                        (tensormap_o_ptr,),
                        self.epilogue_warp_id,
                        (None,),
                    )
                    tensormap_manager.fence_tensormap_update(tensormap_o_ptr)

                    for prefetch_idx in range(2):
                        prefetch_offset = gate_first_row + prefetch_idx * self.b_t
                        is_last_tile = prefetch_idx >= num_valid_chunks_b - 1
                        if cutlass.const_expr(self.use_prefix_lens):
                            is_last_tile = is_last_tile or (
                                prefetch_idx == 0 and lead > 0
                            )
                        load_gate_producer, load_beta_producer = (
                            self.load_gate_beta_warp(
                                tidx,
                                (mGate, mBeta),
                                (sCumsumlog, sCumprod, sBeta),
                                (load_gate_producer, load_beta_producer),
                                (
                                    prefetch_offset,
                                    head_idx,
                                    is_last_tile,
                                    batch_end,
                                )
                                + gate_row_lo,
                            )
                        )

                    # Fill the remaining two lookahead stages together. Since the
                    # chunk count is even, chunks 2 and 3 are either both present
                    # or both absent.
                    if num_chunks_b > 2:
                        for prefetch_idx in range(2, 4):
                            prefetch_offset = gate_first_row + prefetch_idx * self.b_t
                            is_last_tile = prefetch_idx >= num_valid_chunks_b - 1
                            load_gate_producer, load_beta_producer = (
                                self.load_gate_beta_warp(
                                    tidx,
                                    (mGate, mBeta),
                                    (sCumsumlog, sCumprod, sBeta),
                                    (load_gate_producer, load_beta_producer),
                                    (
                                        prefetch_offset,
                                        head_idx,
                                        is_last_tile,
                                        batch_end,
                                    )
                                    + gate_row_lo,
                                )
                            )

                    for chunk_idx in cutlass.range(num_chunks_b):
                        prefetch_idx = chunk_idx + 4
                        if prefetch_idx < num_chunks_b:
                            prefetch_offset = gate_first_row + prefetch_idx * self.b_t
                            is_last_tile = prefetch_idx >= num_valid_chunks_b - 1
                            load_gate_producer, load_beta_producer = (
                                self.load_gate_beta_warp(
                                    tidx,
                                    (mGate, mBeta),
                                    (sCumsumlog, sCumprod, sBeta),
                                    (load_gate_producer, load_beta_producer),
                                    (
                                        prefetch_offset,
                                        head_idx,
                                        is_last_tile,
                                        batch_end,
                                    )
                                    + gate_row_lo,
                                )
                            )
                        chunk_offset = o_first_row + chunk_idx * self.b_t
                        if cutlass.const_expr(self.use_prefix_lens):
                            # A TMA store needs a non-negative box start: the
                            # chunk holding leading identity rows stores its
                            # valid rows with SIMT copies instead.
                            if chunk_offset < 0:
                                o_store_consumer = self.epilogue_store_leading_chunk(
                                    tidx,
                                    sO,
                                    bounded_o[None, None, head_idx],
                                    o_store_consumer,
                                    chunk_offset,
                                    real_seqlen_b,
                                )
                            else:
                                o_store_consumer = self.epilogue_warp(
                                    (sO,),
                                    (tma_o,),
                                    (o_store_consumer,),
                                    (head_idx, chunk_offset),
                                    (tensormap_manager, tensormap_o_ptr),
                                )
                        else:
                            o_store_consumer = self.epilogue_warp(
                                (sO,),
                                (tma_o,),
                                (o_store_consumer,),
                                (head_idx, chunk_offset),
                                (tensormap_manager, tensormap_o_ptr),
                            )
                scheduler.advance_to_next_work()
                work = scheduler.get_current_work()

            load_gate_producer.tail()
            load_beta_producer.tail()


class RubinGatedDeltaNetChunkedKernelTmemDealloc(
    _fi_chunked_sm107.RubinGatedDeltaNetChunkedKernel,
    GatedDeltaNetChunkedKernelTmemDealloc,
):
    """``RubinGatedDeltaNetChunkedKernel`` (v0.7.3): its ``__init__`` and hooks over the fixed base above
    (its ``super().__init__`` resolves to the fixed ``__init__``, and it inherits the fixed ``kernel``)."""


_installed = None


def install() -> bool:
    """Make FlashInfer's GDN prefill launchers build the #8854 classes (idempotent; once per process).

    Raises ImportError (and changes nothing) unless every FlashInfer file involved is v0.7.3's."""
    global _installed
    if _installed is not None:
        return _installed
    stock_source_files = _fi_prefill._kernel_source_files
    # Non-CP: the launcher looks these up at call time; its on-disk cache gets its own name and key files.
    _fi_prefill.GatedDeltaNetChunkedKernel = GatedDeltaNetChunkedKernelTmemDealloc
    _fi_prefill._PREFILL_KERNEL_CLASSES["sm100"] = GatedDeltaNetChunkedKernelTmemDealloc
    _fi_prefill._PREFILL_KERNEL_CLASSES["sm107"] = (
        RubinGatedDeltaNetChunkedKernelTmemDealloc
    )
    _fi_prefill._CUTE_DSL_MODULE = _CUTE_DSL_MODULE
    _fi_prefill._kernel_source_files = lambda: (*stock_source_files(), __file__)
    _fi_prefill._get_compiled_cache.cache_clear()
    _installed = True
    _logger.info(
        "GDN prefill TMEM dealloc fix (FlashInfer #8854 on FlashInfer 0.7.3's files): "
        "GatedDeltaNetChunkedKernel and RubinGatedDeltaNetChunkedKernel free TMEM after every "
        "reading warp's last tcgen05.ld (%s)",
        __file__,
    )
    return True
