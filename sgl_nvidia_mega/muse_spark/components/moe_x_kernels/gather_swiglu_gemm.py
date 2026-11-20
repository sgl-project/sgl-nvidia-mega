# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Modified by NVIDIA: derived from FlashInfer 0.7.1's CuTe-DSL gather grouped GEMM + SwiGLU kernel. Changes: the
# weight operand is read in the row order and block-scale layout of the TensorRT-LLM fused-MoE weight preparation
# (SwiGLU pairs columns in that order and writes the activation in natural order), separate gate / up global scales
# per expert, a per-token input scale, BF16 output only, bounded barrier waits, architecture-specific MMA builders
# resolved at run time, test harness removed.

from typing import NamedTuple, Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass._mlir.dialects import nvvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.tcgen05.mma import CollectorOp
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils.gemm.sm100 import (
    epilogue_smem_copy_and_partition,
    transform_partitioned_tensor_layout,
)

from .arch import arch_helpers
from .custom_pipeline import PipelineCpAsyncUmma
from .inline_ptx import sm100_tcgen05_st_32x32b_x4, sm100_tma_gather4_load

# ============================================================================
# Inline utility functions
# ============================================================================


@dsl_user_op
def fmin(
    a: Union[float, cutlass.Float32],
    b: Union[float, cutlass.Float32],
    *,
    nan=False,
    loc=None,
    ip=None,
) -> cutlass.Float32:
    return cutlass.Float32(
        nvvm.fmin(
            cutlass.Float32(a).ir_value(loc=loc, ip=ip),
            cutlass.Float32(b).ir_value(loc=loc, ip=ip),
            nan=nan,
            loc=loc,
            ip=ip,
        )
    )


def sigmoid_f32(
    a: Union[float, cutlass.Float32], fastmath: bool = False
) -> Union[float, cutlass.Float32]:
    """Compute the sigmoid of the input tensor."""
    return cute.arch.rcp_approx(1.0 + cute.math.exp(-a, fastmath=fastmath))


def silu_f32(
    a: Union[float, cutlass.Float32], fastmath: bool = False
) -> Union[float, cutlass.Float32]:
    """Compute the silu of the input tensor."""
    return a * sigmoid_f32(a, fastmath=fastmath)


class S2TCopyBundle(NamedTuple):
    """Bundle of tiled copy and partitioned tensors for smem-to-tmem copies."""

    tiled_copy: cute.TiledCopy
    sSF_compact: cute.Tensor  # Partitioned source (smem)
    tSF_compact: cute.Tensor  # Partitioned destination (tmem)


"""
Persistent block-scaled contiguous grouped GEMM with token gather and fused SwiGLU activation (first GEMM of a
mixture-of-experts layer), reading the expert weights in the layout produced by the TensorRT-LLM fused-MoE weight
preparation (``prepare_static_weights_for_trtllm_fp4_moe``).

Compute:
  acc  = (SFA * A[token_ids]) * (SFB * B)                    # GEMM, fp32 accumulation
  gate = acc_gate * alpha_gate[expert] * token_scale[token]
  up   = acc_up   * alpha_up[expert]   * token_scale[token]
  C    = up * silu(gate)                                       # BF16

Weight row order: the preparation interleaves the up and gate rows of each intermediate index and then shuffles every
block of 32 rows. Within each block of 32 output columns of the accumulator the columns are therefore
[up(i0 .. i0+14 step 2) | gate(same) | up(i0+1 .. i0+15 step 2) | gate(same)] with i0 = 16 * block, eight columns
each. The epilogue pairs these columns in registers and writes the 16 activations of the block in natural order, so
C is in natural intermediate order. The block scales use the same row order and the standard 128x4 swizzled layout.

Shapes: A is M x K x 1 (gathered by token_id_mapping; token_id == -1 marks padding rows); B is N x K x L (L = number of
experts); C is M x (N/2) x 1. token_scale is per token (orig_m), alpha_gate / alpha_up per expert.

Within a tile, valid_m varies per group; padding rows are handled at load: TMA gather4 passes -1 to zero-fill; CpAsync
predicates on `abs_row < mn_limit`.

For CUDA graph, A/C/SFA/token_id_mapping/tile_idx_to_expert_idx can be padded to permuted_m; padded tiles are filtered
by the scheduler.
"""


class GatherSwigluGroupedGemmKernel:
    """Contiguous grouped block-scaled GEMM with token gather on A/SFA and SwiGLU activation fusion in the epilogue.

    Persistent tile scheduling, warp specialization, B-reuse, block-scaled tcgen05 MMA, TMA B/SFB.

      - Token gather: A/SFA rows are gathered by token_id_mapping (token_id == -1 marks padding rows).
      - A load path (knob `a_path`):
          * cpasync - per-thread cp.async (default); separate a_pipeline.
          * tma     - TMA gather4; A and B share a single merged ab_pipeline.
        SFA is always loaded via cp.async, then reorganized into SFA tensor memory by transform warps
        (shared-memory loads + tcgen05.st, sfa_transform_pipeline).
      - SwiGLU epilogue in the weight row order described in the module docstring; BF16 output.

    Warp roles (20 warps):
      - 0-3   epilogue (tcgen05.ld -> SwiGLU -> TMA store)
      - 4-7   gather A         (cp.async or TMA gather4)
      - 8     MMA
      - 9     TMA B / SFB
      - 10    scheduler
      - 11    idle (1-CTA)
      - 12-15 gather SFA       (cp.async)
      - 16-19 SFA transform    (shared-memory loads + tcgen05.st into SFA tensor memory)

    :param sf_vec_size: Scale factor vector size (16).
    :param mma_inst_shape: MMA instruction shape (M, N, K); M must be 128 (1-CTA).
    :param mma_tiler: MMA tiler shape (M, N, K).
    :param cluster_shape_mn: Cluster dimensions (M, N).
    :param vectorized_f32: Use vectorized f32x2 ops in epilogue.
    :param topk: Experts selected per token.
    :param arch: target architecture string (from the device's compute capability at run time).
    :param raster_along_m: If True, raster tiles along M first.
    :param a_path: "cpasync" or "tma" - A load implementation.
    """

    def __init__(
        self,
        sf_vec_size: int,
        mma_inst_shape: Tuple[int, int, int],
        mma_tiler: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        vectorized_f32: bool,
        topk: cutlass.Int64,
        arch: str,
        raster_along_m: bool = False,
        a_path: str = "cpasync",
        use_pdl: bool = False,
    ):
        self.a_path = a_path
        self.sf_vec_size = sf_vec_size
        self.topk = topk
        self.acc_dtype = cutlass.Float32
        self.mma_inst_shape = mma_inst_shape
        self.mma_tiler = mma_tiler
        self.cluster_shape_mn = cluster_shape_mn
        self.raster_along_m = raster_along_m
        self.use_pdl = use_pdl

        self.use_2cta_instrs = mma_inst_shape[0] == 256
        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )
        self.arch = arch
        self.smem_capacity = utils.get_smem_capacity_in_bytes(self.arch)
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols(self.arch)

        self.occupancy = 1
        self.epilog_warp_id = (0, 1, 2, 3)
        self.gather_a_warp_id = (
            4,
            5,
            6,
            7,
        )
        self.mma_warp_id = 8
        self.tma_b_warp_id = 9
        self.sched_warp_id = 10
        # Warp 11: cpasync 2CTA A peer-sync relay (sync_transform_warp_id) /
        # idle on tma-A (dummy_warp_id). Slot reserved for SM occupancy.
        # sync_transform_warp_id is always defined so downstream traces
        # resolve in tma mode (body is const-gated out).
        self.sync_transform_warp_id = 11
        if self.a_path == "tma":
            self.dummy_warp_id = 11
        self.gather_sfa_warp_id = (
            12,
            13,
            14,
            15,
        )
        # 4 SFA transform warps (shared-memory load source + tcgen05.st destination into SFA TMEM).
        self.sfa_transform_warp_id = (
            16,
            17,
            18,
            19,
        )
        # Register reconfig (setmaxnreg) per warpgroup. Default 128/thread.
        self.num_regs_epilogue_warps = 168
        self.num_regs_gather_a_warps = 80
        self.num_regs_gather_sfa_warps = 80
        self.num_regs_sfa_transform_warps = 48
        self.num_regs_mma_group_warps = 128
        self.threads_per_warp = 32
        # warp 11 slot is always counted in threads_per_cta (SM occupancy)
        # regardless of cpasync-A peer-sync role vs TMA-A idle role.
        _warp11_id = (
            self.sync_transform_warp_id
            if self.a_path == "cpasync"
            else self.dummy_warp_id
        )
        self.threads_per_cta = self.threads_per_warp * len(
            (
                self.mma_warp_id,
                *self.gather_a_warp_id,
                self.tma_b_warp_id,
                *self.epilog_warp_id,
                self.sched_warp_id,
                _warp11_id,
                *self.gather_sfa_warp_id,
                *self.sfa_transform_warp_id,
            )
        )
        # warps_wo_sched = tile_info_pipeline consumers (all warps except
        # scheduler). Warp 11 counts only on cpasync 2CTA (relay needs tiles);
        # excluded on 1CTA or tma-A (warp 11 idle).
        if self.use_2cta_instrs and self.a_path == "cpasync":
            _wo_sched_warps = (
                *self.epilog_warp_id,
                self.mma_warp_id,
                self.tma_b_warp_id,
                self.sync_transform_warp_id,
                *self.gather_a_warp_id,
                *self.gather_sfa_warp_id,
                *self.sfa_transform_warp_id,
            )
        else:
            _wo_sched_warps = (
                *self.epilog_warp_id,
                self.mma_warp_id,
                self.tma_b_warp_id,
                *self.gather_a_warp_id,
                *self.gather_sfa_warp_id,
                *self.sfa_transform_warp_id,
            )
        self.warps_wo_sched = len(_wo_sched_warps)
        self.threads_wo_sched = self.threads_per_warp * self.warps_wo_sched

        # Set barrier for cta sync, epilogue sync and tmem ptr sync
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=32 * len(self.epilog_warp_id),
        )
        # tmem_alloc_barrier participants: epi (allocator) + mma (consumer)
        # + transform warps (tcgen05.st producers, need TMEM ptr to write SFA).
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=32
            * len(
                (
                    self.mma_warp_id,
                    *self.epilog_warp_id,
                    *self.sfa_transform_warp_id,
                )
            ),
        )
        self.sched_sync_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=self.threads_per_warp,
        )

        self.num_smem_capacity = self.smem_capacity
        # num_tmem_alloc_cols already set in __init__

        self.vectorized_f32 = vectorized_f32

        # For epilogue compatibility
        self.epilogue_warp_id = self.epilog_warp_id

        # B-reuse pattern control
        self.enable_breuse = True if mma_tiler[0] // mma_inst_shape[0] == 2 else False

        # Overlapping ACC TMEM: acc[0]/acc[1] share 64 cols. Epilogue
        # iterates the overlap region first (reverse for acc[0]) and
        # early-releases so MMA can write the next stage. Frees enough TMEM
        # for 4 SFA stages. Auto-on for non-breuse cta_tile_N=256.
        self.use_overlap_accum = (not self.enable_breuse) and (mma_tiler[1] == 256)

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B
        - Computing epilogue subtile
        - Setting up A/B/C stage counts in shared memory
        - Computing A/B/C shared memory layout
        - Computing tensor memory allocation columns
        """

        self.mma_inst_shape_sfb = (
            self.mma_inst_shape[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape[1], 128),
            self.mma_inst_shape[2],
        )

        # Configure tiled mma (architecture-specific builder)
        tiled_mma = arch_helpers().make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
            atom_layout_mnk=(1, 1, 1),
            permutation_mnk=self._get_mma_permutation_mnk(),
        )

        tiled_mma_sfb = arch_helpers().make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_sfb,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
        )

        # Compute mma/cluster/tile shapes
        self.mma_tiler_sfb = (
            self.mma_inst_shape_sfb[0],
            self.mma_inst_shape_sfb[1],
            self.mma_tiler[2],
        )

        self.mma_tiler_c = (
            self.mma_tiler[0],
            self.mma_tiler[1] // 2,
            self.mma_tiler[2],
        )

        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        # Number of CpAsync128.CG loads per thread for A matrix (each loads 16 M-rows)
        self.a_num_loads = self.cta_tile_shape_mnk[0] // 16

        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )

        self.cta_tile_shape_mnk_c = (
            self.mma_tiler_c[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_c[1],
            self.mma_tiler_c[2],
        )

        # Compute SFA tiler for CpAsync gather (use mma_inst_shape for M/N, scaled K for SF)
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = self.mma_tiler[2] // mma_inst_shape_k
        self.mma_tiler_sfa = (
            self.mma_inst_shape[0],
            self.mma_inst_shape[1],
            mma_inst_shape_k * mma_inst_tile_k // 16,
        )
        self.cta_tile_shape_mnk_sfa = (
            self.mma_tiler_sfa[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfa[1],
            self.mma_tiler_sfa[2],
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        # A multicast: cluster_N CTAs share A along N dim. Only meaningful when
        # cluster_N > 1. SFA multicast intentionally NOT enabled — was buggy
        # on cta_tile_N >= 256 and not worth the complexity.
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.is_a_mcast = self.num_mcast_ctas_a > 1

        # Fixed epilogue tile (128, 64) in output columns (SwiGLU halves N); every output subtile reads two
        # 64-column accumulator subtiles.
        self.epi_tile = (128, 64)
        self.epi_tile_n = cute.size(self.epi_tile[1])
        self.epi_tile_cnt = (
            self.cta_tile_shape_mnk_c[0] // cute.size(self.epi_tile[0]),
            self.cta_tile_shape_mnk_c[1] // cute.size(self.epi_tile[1]),
        )

        # Setup A/B/C/Scale stage count in shared memory and ACC stage count in tensor memory
        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_c_stage,
            self.num_tile_stage,
        ) = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.cta_tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.sf_dtype,
            self.sf_vec_size,
            self.smem_capacity,
            self.occupancy,
            self.enable_breuse,
        )

        # Compute A/B/C/Scale shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        # Canonical SFA SMEM layout (from blockscaled_utils), used only to
        # derive tCtSFA_layout below. The actual SFA SMEM uses the linear
        # layout built next; this canonical one isn't allocated.
        sfa_canon_smem_layout = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )

        # SFA SMEM is plain linear (M_per_cta, tile_K_sf, stage), no pad.
        # Each thread does one CpAsync128.CG (16B = tile_K_sf=16 × FP8) per row.
        # Layout exposes (row, k_sf_byte, stage) with byte strides.
        sfa_tile_k_sf = self.cta_tile_shape_mnk[2] // self.sf_vec_size
        sf_bytes_per_row = sfa_tile_k_sf * self.sf_dtype.width // 8
        sfa_bytes_per_stage = self.cta_tile_shape_mnk[0] * sf_bytes_per_row
        self.sfa_smem_layout_staged = cute.make_layout(
            (self.cta_tile_shape_mnk[0], sfa_tile_k_sf, self.num_ab_stage),
            stride=(sf_bytes_per_row, 1, sfa_bytes_per_stage),
        )
        self.sfa_smem_alloc_bytes = self.num_ab_stage * sfa_bytes_per_stage

        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_c_stage,
        )

        # Compute TMEM layouts for SFA/SFB
        self.tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            cute.slice_(sfa_canon_smem_layout, (None, None, None, 0)),
        )
        self.tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0)),
        )

        # Compute TMEM column counts.
        # SFA TMEM holds num_sfa_tmem_stage stages, each tCtSFA_layout wide.
        # TMEM layout: [acc | sfa (N stages) | sfb].
        self.num_sfa_tmem_cols_per_stage = (
            cute.cosize(cute.recast_layout(32, self.sf_dtype.width, self.tCtSFA_layout))
            & 0x0000FFFF
        )
        # SFA TMEM stages: 4 for non-breuse, 1 for breuse (B-reuse fills tensor memory with one SFA stage). At
        # non-breuse N=256, use_overlap_accum is auto-on to free tensor memory for 4 stages.
        self.num_sfa_tmem_stage = 1 if self.enable_breuse else 4
        self.num_sfa_tmem_cols = (
            self.num_sfa_tmem_cols_per_stage * self.num_sfa_tmem_stage
        )
        self.num_sfb_tmem_cols = (
            cute.cosize(cute.recast_layout(32, self.sf_dtype.width, self.tCtSFB_layout))
            & 0x0000FFFF
        )
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        # use_overlap_accum: pipeline tracks 1 acc stage; physically 2 stages
        # share TMEM with one epi_tile_n of overlap. Otherwise:
        # tile_N × num_acc_stage × (2 if breuse).
        if self.use_overlap_accum:
            self.num_acc_stage = 1  # logical 2 via overlap
            self.num_accumulator_tmem_cols = (
                self.cta_tile_shape_mnk[1] * 2 - self.epi_tile_n
            )
        else:
            self.num_accumulator_tmem_cols = (
                self.cta_tile_shape_mnk[1]
                * self.num_acc_stage
                * (2 if self.enable_breuse else 1)
            )
        # SFA TMEM offset (cols, 32-bit each): right after acc.
        self.sfa_tmem_offset = self.num_accumulator_tmem_cols
        # Validation against the queried tensor-memory column budget
        _total_used = (
            self.num_accumulator_tmem_cols
            + self.num_sfa_tmem_cols
            + self.num_sfb_tmem_cols
        )
        if _total_used > self.num_tmem_alloc_cols:
            raise ValueError(
                f"TMEM overflow: acc({self.num_accumulator_tmem_cols}) + "
                f"sfa({self.num_sfa_tmem_cols}) + "
                f"sfb({self.num_sfb_tmem_cols}) = {_total_used} > "
                f"max {self.num_tmem_alloc_cols}"
            )

    def _get_mma_permutation_mnk(self):
        if cutlass.const_expr(self.use_2cta_instrs and self.enable_breuse):
            m_layout = cute.make_layout(
                shape=(self.mma_inst_shape[0] // 2, 2, 2),
                stride=(1, self.mma_inst_shape[0], self.mma_inst_shape[0] // 2),
            )
            return (m_layout, self.mma_inst_shape[1], self.mma_inst_shape[2])
        else:
            return (1, 1, 1)

    def _is_interleaved_utccp(self) -> bool:
        """Enable interleaving tcgen05.cp for Bkeep-Breuse case for 4xFP4 kernel."""
        return (
            self.a_dtype.width == 4 and self.b_dtype.width == 4 and self.enable_breuse
        )

    def _mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> S2TCopyBundle:
        """Make tiledCopy for smem to tmem load for scale factor tensor."""
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)

        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        def appendMNBroadcastMode(smem_layout: cute.Layout):
            mn_dim = cute.get(smem_layout, mode=[0, 0])
            mn_dim = cute.append(mn_dim, cute.make_layout((4), stride=(0)))
            layout = cute.append(
                cute.group_modes(mn_dim, 0), cute.get(smem_layout, mode=[0, 1])
            )
            layout = cute.append(
                cute.group_modes(layout, 0), cute.get(smem_layout, mode=[1])
            )
            layout = cute.append(layout, cute.get(smem_layout, mode=[2]))
            layout = cute.append(layout, cute.get(smem_layout, mode=[3]))
            return layout

        tCsSF_compact_bcast = cute.make_tensor(
            tCsSF_compact.iterator, appendMNBroadcastMode(tCsSF_compact.layout)
        )

        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact_bcast)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_
        )
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return S2TCopyBundle(tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t)

    def _mainloop_s2t_copies(
        self,
        stage_idx: int,
        sfb_s2t_bundle: S2TCopyBundle,
    ):
        """Copy SFB from smem to tmem (tcgen05.cp). The SFA path uses shared-memory loads + tcgen05.st
        in the transform warps, no tcgen05.cp needed here."""
        s2t_stage_coord = (None, None, None, None, stage_idx)

        cute.copy(
            sfb_s2t_bundle.tiled_copy,
            sfb_s2t_bundle.sSF_compact[s2t_stage_coord],
            sfb_s2t_bundle.tSF_compact,
        )

    def _mainloop_s2t_interleaved_copies(
        self,
        k_block: int,
        stage_idx: int,
        sfa_s2t_bundle: S2TCopyBundle,
        sfb_s2t_bundle: S2TCopyBundle,
    ):
        """Interleaved tcgen05.cp for Bkeep-Breuse pattern."""
        s_sfa_crd_keep = (None, 0, None, k_block, stage_idx)
        s_sfa_crd_reuse = (None, 1, None, k_block, stage_idx)
        s_sfb_crd = (None, None, None, k_block, stage_idx)

        t_sfa_crd_keep = (None, 0, None, k_block)
        t_sfa_crd_reuse = (None, 1, None, k_block)
        t_sfb_crd = (None, None, None, k_block)

        cute.copy(
            sfa_s2t_bundle.tiled_copy,
            sfa_s2t_bundle.sSF_compact[s_sfa_crd_keep],
            sfa_s2t_bundle.tSF_compact[t_sfa_crd_keep],
        )
        cute.copy(
            sfb_s2t_bundle.tiled_copy,
            sfb_s2t_bundle.sSF_compact[s_sfb_crd],
            sfb_s2t_bundle.tSF_compact[t_sfb_crd],
        )
        cute.copy(
            sfa_s2t_bundle.tiled_copy,
            sfa_s2t_bundle.sSF_compact[s_sfa_crd_reuse],
            sfa_s2t_bundle.tSF_compact[t_sfa_crd_reuse],
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        token_id_mapping_tensor: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        alpha_gate: cute.Tensor,
        alpha_up: cute.Tensor,
        token_scale: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Execute the contiguous grouped GEMM with gather operation and SwiGLU fusion.

        :param a: Input tensor A (orig_m x K x 1), gathered using token_id_mapping
        :param b: Weight tensor B (N x K x L) in the fused-MoE preparation row order (see module docstring)
        :param c: Output tensor C (M x (N/2) x 1), BF16, natural intermediate order
        :param sfa: Scale factor tensor A (orig_m x K/16, linear), gathered using token_id_mapping
        :param sfb: Scale factor tensor B (128x4 swizzled per expert)
        :param tile_idx_to_expert_idx: tile index -> expert index, shape (permuted_m / cta_tile_m,)
        :param tile_idx_to_mn_limit: tile index -> end of the tile's valid rows, shape (permuted_m / cta_tile_m,)
        :param token_id_mapping_tensor: permuted row -> expanded index (token * topk + slot), -1 for padding rows
        :param num_non_exiting_tiles: number of valid tiles, shape (1,)
        :param alpha_gate: gate global scale per expert, shape (L,)
        :param alpha_up: up global scale per expert, shape (L,)
        :param token_scale: input scale per token, shape (orig_m,)
        :param max_active_clusters: Maximum number of active clusters
        :param stream: CUDA stream for asynchronous execution
        """
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        # Setup sfb tensor by filling B tensor to scale factor atom layout
        # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b.shape, self.sf_vec_size)
        sfb = cute.make_tensor(sfb.iterator, sfb_layout)

        atom_layout_mnk = (1, 1, 1)
        permutation_mnk = self._get_mma_permutation_mnk()

        tiled_mma = arch_helpers().make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
            atom_layout_mnk=atom_layout_mnk,
            permutation_mnk=permutation_mnk,
        )
        tiled_mma.set(tcgen05.Field.NEGATE_A, False)
        tiled_mma.set(tcgen05.Field.NEGATE_B, False)

        # For 2CTA blockscaled kernels, SFB needs to be replicated across peer CTAs.
        tiled_mma_sfb = arch_helpers().make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_sfb,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
        )
        tiled_mma_sfb.set(tcgen05.Field.NEGATE_A, False)
        tiled_mma_sfb.set(tcgen05.Field.NEGATE_B, False)

        tiled_mma_bkeep = None
        tiled_mma_breuse = None
        if cutlass.const_expr(self.enable_breuse):
            tiled_mma_bkeep = arch_helpers().make_blockscaled_trivial_tiled_mma(
                self.a_dtype,
                self.b_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                self.mma_inst_shape,
                a_collector_op=CollectorOp.DISCARD,
                b_collector_op=CollectorOp.FILL,
                atom_layout_mnk=atom_layout_mnk,
                permutation_mnk=permutation_mnk,
            )
            tiled_mma_bkeep.set(tcgen05.Field.NEGATE_A, False)
            tiled_mma_bkeep.set(tcgen05.Field.NEGATE_B, False)

            tiled_mma_breuse = arch_helpers().make_blockscaled_trivial_tiled_mma(
                self.a_dtype,
                self.b_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                self.mma_inst_shape,
                a_collector_op=CollectorOp.DISCARD,
                b_collector_op=CollectorOp.LASTUSE,
                atom_layout_mnk=atom_layout_mnk,
                permutation_mnk=permutation_mnk,
            )
            tiled_mma_breuse.set(tcgen05.Field.NEGATE_A, False)
            tiled_mma_breuse.set(tcgen05.Field.NEGATE_B, False)
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # cpasync-A: CpAsync128.CG gmem → sA SMEM, no TMA. 4 gather_a warps × 32
        # threads issue cp.async.cg.16B per (token row, k chunk). a_num_loads
        # (in _setup_attributes) controls CpAsync iterations per thread per k_tile.
        tma_atom_a = None
        tma_tensor_a = None
        # tma-A mode: build a 2D gather4 TMA atom for A (box_rows = 1,
        # SW128-permuted base) so gather4 writes sA at the same SW128 offsets
        # UMMA's K_SW128 reads expect.
        if cutlass.const_expr(self.a_path == "tma"):
            a_2d = a[(None, None, 0)]
            a_gather_base = cute.make_layout(
                (1, self.mma_tiler[2]),
                stride=(self.mma_tiler[2], 1),
            )
            sw128 = cute.make_swizzle(3, 4, 3)  # K_SW128
            a_gather_smem_layout = cute.make_composed_layout(sw128, 0, a_gather_base)
            tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                a_2d,
                a_gather_smem_layout,
                (1, self.mma_tiler[2]),  # cta_tiler, box_rows = 1
            )
            # tx_count split evenly across the 4 gather warps. Total = full A
            # tile bytes (× atom_thr_size for 2CTA's leader-mbar collapse).
            self.tma_gather_num_warps = len(self.gather_a_warp_id)
            self.a_num_tma_load_bytes_total = (
                self.cta_tile_shape_mnk[0]
                * self.cta_tile_shape_mnk[2]
                * self.a_dtype.width
                // 8
                * atom_thr_size
            )
            self.a_num_tma_load_bytes = (
                self.a_num_tma_load_bytes_total // self.tma_gather_num_warps
            )

        # cpasync-A SFA path is also CpAsync128.CG (no TMA). gather_sfa warps
        # issue cp.async per row; one row per thread (128 rows / 128 threads
        # in 4 warps). No tma_atom_sfa needed in either A mode.

        # Setup TMA load for B
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for SFB
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # This modifies the layout to handle overlapping 256x(# of scale factors for a single column of B (nNSF))
        # logical blocks for SFB when cta_tile_shape_n=192.
        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
            x = tma_tensor_sfb.stride[0][1]
            y = cute.ceil_div(tma_tensor_sfb.shape[0][1], 4)

            new_shape = (
                (tma_tensor_sfb.shape[0][0], ((2, 2), y)),
                tma_tensor_sfb.shape[1],
                tma_tensor_sfb.shape[2],
            )
            # Use right multiplication for ScaledBasis (3 * x instead of x * 3)
            x_times_3 = 3 * x
            new_stride = (
                (tma_tensor_sfb.stride[0][0], ((x, x), x_times_3)),
                tma_tensor_sfb.stride[1],
                tma_tensor_sfb.stride[2],
            )
            tma_tensor_sfb_new_layout = cute.make_layout(new_shape, stride=new_stride)
            tma_tensor_sfb = cute.make_tensor(
                tma_tensor_sfb.iterator, tma_tensor_sfb_new_layout
            )

        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (b_copy_size + sfb_copy_size) * atom_thr_size

        # Setup TMA store for C
        tma_atom_c = None
        tma_tensor_c = None
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c,
            epi_smem_layout,
            self.epi_tile,
        )

        # Compute grid size
        self.tile_sched_params, grid = self._compute_grid(
            c,
            self.cta_tile_shape_mnk_c,
            self.cluster_shape_mn,
            max_active_clusters,
            self.raster_along_m,
        )

        self.buffer_align_bytes = 1024

        # Define shared storage for kernel.
        @cute.struct
        class SharedStorageCpasync1cta:
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 5 * self.num_tile_stage],
                1,
            ]
            # cpasync mode: A and B use separate pipelines (CpAsync A is
            # CpAsync type, B is TmaUmma type — they can't share one mbar).
            # Each mbar set holds num_ab_stage * 2 (full + empty per stage).
            a_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            b_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            sfa_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            sfa_transform_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_sfa_tmem_stage * 2
            ]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_tile_stage * 2
            ]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # sSFA placed BEFORE sA (low shared-memory offset for the SFA cp.async destination; upstream layout).
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, self.sfa_smem_alloc_bytes],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        # 2CTA variant: adds a_sync_transform_mbar_ptr for the warp-11 relay
        # pipeline (PipelineAsyncUmma) that bridges per-CTA `a_pipeline` to
        # the cluster-wide MMA consumer (tcgen05.mma.cta_group::2).
        @cute.struct
        class SharedStorageCpasync2cta:
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 5 * self.num_tile_stage],
                1,
            ]
            a_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            b_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            sfa_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            sfa_transform_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_sfa_tmem_stage * 2
            ]
            a_sync_transform_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_ab_stage * 2
            ]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_tile_stage * 2
            ]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, self.sfa_smem_alloc_bytes],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        # TMA-A variant: A and B share one ab_pipeline mbar (4 gather_a + 1
        # tma_b producers arrive on the same mbar). No a_sync_transform mbar
        # needed (TMA gather4 has HW `.multicast::cluster`, no peer-sync relay).
        # Storage layout is independent of 1CTA/2CTA mode in TMA-A mode.
        @cute.struct
        class SharedStorageTma:
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 5 * self.num_tile_stage],
                1,
            ]
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            sfa_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            sfa_transform_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_sfa_tmem_stage * 2
            ]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_tile_stage * 2
            ]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # sSFA placed BEFORE sA (low shared-memory offset for the SFA cp.async destination; upstream layout).
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, self.sfa_smem_alloc_bytes],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        if cutlass.const_expr(self.a_path == "cpasync"):
            self.shared_storage = (
                SharedStorageCpasync2cta
                if self.use_2cta_instrs
                else SharedStorageCpasync1cta
            )
        else:  # "tma"
            self.shared_storage = SharedStorageTma

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tiled_mma_bkeep,
            tiled_mma_breuse,
            tiled_mma_sfb,
            a,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            token_id_mapping_tensor,
            num_non_exiting_tiles,
            alpha_gate,
            alpha_up,
            token_scale,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.tCtSFA_layout,
            self.tCtSFB_layout,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.use_pdl,
        )
        return

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for smem to tmem load for scale factor tensor, then use it to
        partition smem memory (source) and tensor memory (destination).

        :param sSF: The scale factor tensor in smem
        :type sSF: cute.Tensor
        :param tSF: The scale factor tensor in tmem
        :type tSF: cute.Tensor

        :return: A tuple containing (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t) where:
            - tiled_copy_s2t: The tiled copy operation for smem to tmem load for scale factor tensor(s2t)
            - tCsSF_compact_s2t: The partitioned scale factor tensor in smem
            - tSF_compact_s2t: The partitioned scale factor tensor in tmem
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSF_compact = cute.filter_zeros(sSF)
        # (MMA, MMA_MN, MMA_K)
        tCtSF_compact = cute.filter_zeros(tSF)

        # Make S2T CopyAtom and tiledCopy
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_
        )
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_bkeep: Optional[cute.TiledMma],
        tiled_mma_breuse: Optional[cute.TiledMma],
        tiled_mma_sfb: cute.TiledMma,
        mA_mkl: cute.Tensor,
        tma_atom_a: Optional[cute.CopyAtom],
        tma_tensor_a: Optional[cute.Tensor],
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        token_id_mapping_tensor: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        alpha_gate: cute.Tensor,
        alpha_up: cute.Tensor,
        token_scale: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        tCtSFA_layout: cute.Layout,
        tCtSFB_layout: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_b_warp_id:
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_c)
            if cutlass.const_expr(self.a_path == "tma"):
                cpasync.prefetch_descriptor(tma_atom_a)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )

        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )

        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # (a_pipeline created below alongside b_pipeline.)

        # SFA pipeline (PipelineCpAsync): gather_sfa warps → transform warps.
        # Producer: 4 gather_sfa warps × 32 threads (one CpAsync128.CG per row).
        # Consumer: 4 transform warps × 32 threads.
        # MMA waits on sfa_transform_pipeline downstream (after shared-memory load+tcgen05.st).
        sfa_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len(self.gather_sfa_warp_id) * self.threads_per_warp,
        )
        sfa_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len(self.sfa_transform_warp_id) * self.threads_per_warp,
        )
        sfa_pipeline = pipeline.PipelineCpAsync.create(
            barrier_storage=storage.sfa_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=sfa_pipeline_producer_group,
            consumer_group=sfa_pipeline_consumer_group,
            defer_sync=True,
        )

        # SFA transform pipeline: transform warps (tcgen05.st) → MMA. PipelineAsyncUmma
        # with cta_layout_vmnk so 2CTA peer-CTA arrives route to leader's mbar.
        # Producer: 4 transform warps × 32 threads × cta_v_size. Consumer: MMA.
        # num_stages = num_sfa_tmem_stage SFA TMEM slots, rotated.
        cta_v_size = cute.size(cluster_layout_vmnk, mode=[0])
        sfa_transform_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len(self.sfa_transform_warp_id) * self.threads_per_warp * cta_v_size,
        )
        sfa_transform_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        sfa_transform_pipeline = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.sfa_transform_mbar_ptr.data_ptr(),
            num_stages=self.num_sfa_tmem_stage,
            producer_group=sfa_transform_pipeline_producer_group,
            consumer_group=sfa_transform_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # A/B pipeline topology — branch on a_path:
        #   cpasync-A: separate a_pipeline (CpAsyncUmma) + b_pipeline (TmaUmma)
        #     + a_sync_transform_pipeline (AsyncUmma, 2CTA peer-sync relay).
        #   tma-A: single ab_pipeline (TmaUmma) shared by A and B producers.
        if cutlass.const_expr(self.a_path == "cpasync"):
            # cpasync-A: A producer = 128 threads (4 gather_a warps) issuing
            # cp.async.cg.16B; each thread arrives once per stage. Consumer =
            # MMA (UMMA), 1 thread.
            a_pipeline_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp * len(self.gather_a_warp_id),
            )
            a_pipeline = PipelineCpAsyncUmma.create(
                barrier_storage=storage.a_mbar_ptr.data_ptr(),
                num_stages=self.num_ab_stage,
                producer_group=a_pipeline_producer_group,
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )

            # cpasync 2CTA A sync-transform relay. Producer thread count
            # = 1 warp × cta_v_size so the cluster-wide arrive_count matches
            # warp 11 across CTAs.
            if cutlass.const_expr(self.use_2cta_instrs):
                a_sync_transform_pipeline_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    self.threads_per_warp * cta_v_size,
                )
                a_sync_transform_pipeline = pipeline.PipelineAsyncUmma.create(
                    barrier_storage=storage.a_sync_transform_mbar_ptr.data_ptr(),
                    num_stages=self.num_ab_stage,
                    producer_group=a_sync_transform_pipeline_producer_group,
                    consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                    cta_layout_vmnk=cluster_layout_vmnk,
                    defer_sync=True,
                )

            # B/SFB pipeline (TMA → UMMA), 1 producer thread (tma_b warp).
            # mcast_mode_mn=(0, 1): A is per-CTA cpasync (no N-multicast); B
            # is TMA per-CTA or M-multicast. Default (1,1) would release
            # across N peers — wrong since N-peers hold different B tiles.
            b_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            num_tma_producer = self.num_mcast_ctas_b
            b_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_tma_producer
            )
            b_pipeline = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.b_mbar_ptr.data_ptr(),
                num_stages=self.num_ab_stage,
                producer_group=b_pipeline_producer_group,
                consumer_group=b_pipeline_consumer_group,
                tx_count=self.num_tma_load_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                mcast_mode_mn=(0, 1),
                defer_sync=True,
            )
        else:  # tma — merged ab_pipeline (TmaUmma): 4 gather_a + 1 tma_b
            # on one mbar. Per-call expected_tx accumulates 4 × a_num_tma_load
            # + num_tma_load per stage. Consumer release routes A's N peers
            # + B's M peers → consumer_group = num_mcast_ctas_a + num_mcast_ctas_b - 1.
            ab_pipeline_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                len(self.gather_a_warp_id) + 1,  # 4 gather_a + 1 tma_b
            )
            num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
            ab_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_tma_producer
            )
            ab_pipeline = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.ab_mbar_ptr.data_ptr(),
                num_stages=self.num_ab_stage,
                producer_group=ab_pipeline_producer_group,
                consumer_group=ab_pipeline_consumer_group,
                tx_count=0,  # per-producer expected_tx overrides drive accumulation
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )

        # Pipeline Init: Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (
            2 if use_2cta_instrs else 1
        )
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Pipeline Init:Initialize tile info pipeline (barrier) and states
        tile_info_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * 1,
        )
        # All 4 gather A warps consume tile_info in both CpAsync and TMA paths.
        tile_info_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_wo_sched,
        )
        tile_info_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.tile_info_mbar_ptr.data_ptr(),
            num_stages=self.num_tile_stage,
            producer_group=tile_info_pipeline_producer_group,
            consumer_group=tile_info_pipeline_consumer_group,
        )

        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
            arch=self.arch,
        )

        # Cluster arrive after barrier init (pipeline_init_arrive)
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        #
        # Setup smem tensor A/B/C/Scale
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sC = storage.sC.get_tensor(
            c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        # SFA SMEM (linear+pad layout for TMA gather4).
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        # (granularity_n, repeat_n), (granularity_k, repeat_k), num_scale_stage)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        # (bidx, bidy, bidz, valid, mn_limit)
        info_layout = cute.make_layout((5, self.num_tile_stage), stride=(1, 5))
        sInfo = storage.sInfo.get_tensor(info_layout)

        #
        # Compute multicast mask for A/B buffer full
        #
        b_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_b_mcast or use_2cta_instrs):
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )
        # A multicast mask (tma-A only). cpasync-A path has no TMA multicast
        # (each thread issues its own cp.async.cg.16B). tma-A broadcasts A
        # along cluster N (mcast_mode=2); sm100_tma_gather4_load picks the
        # `.multicast::cluster` PTX variant when this mask is non-None.
        a_full_mcast_mask = None
        if cutlass.const_expr(self.a_path == "tma" and self.is_a_mcast):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, loopM, loopK, loopL)
        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_(self.cta_tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        # (bN, bK, loopN, loopK, loopL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )

        # (bM, bK, RestM, RestK, RestL)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl,
            cute.slice_(self.cta_tile_shape_mnk_sfa, (None, 0, None)),
            (None, None, None),
        )

        # (bN, bK, RestN, RestK, RestL)
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )

        gToken_ml = cute.local_tile(
            token_id_mapping_tensor,
            cute.slice_(self.cta_tile_shape_mnk, (None, 0, 0)),
            (None,),
        )

        # (bM, bN, loopM, loopN, loopL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler_c, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cutlass.Int32(cute.size(gA_mkl, mode=[3]))

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        # (MMA, MMA_N, MMA_K, loopN, loopK, loopL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        # (MMA, MMA_M, MMA_N, loopM, loopN, loopL)
        tCgC = thr_mma.partition_C(gC_mnl)

        #
        # Partition global/shared tensor for TMA load B
        #
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), loopM, loopK, loopL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        # TMA load SFB partition_S/D
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        if cutlass.const_expr(self.use_overlap_accum):
            # Pipeline tracks 1 stage but TMEM has 2 physical regions overlap-
            # ping by 64 cols. Build fragment with 2 stages and stride-hack
            # the stage dim to (256 - 64) = 192 cols (= cta_tile_N - overlap).
            num_acc_stage_overlapped = 2
            tCtAcc_fake = tiled_mma.make_fragment_C(
                cute.append(acc_shape, num_acc_stage_overlapped)
            )
            tCtAcc_fake = cute.make_tensor(
                tCtAcc_fake.iterator,
                cute.make_layout(
                    tCtAcc_fake.shape,
                    stride=(
                        tCtAcc_fake.stride[0],
                        tCtAcc_fake.stride[1],
                        tCtAcc_fake.stride[2],
                        (256 - 64) * tCtAcc_fake.stride[0][1],
                    ),
                ),
            )
        else:
            tCtAcc_fake = tiled_mma.make_fragment_C(
                cute.append(acc_shape, self.num_acc_stage)
            )

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        cute.arch.griddepcontrol_wait()

        #
        # Specialized Schedule Warp
        #
        if warp_idx == self.sched_warp_id:
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            # First tile
            work_tile = tile_sched.initial_work_tile_info()

            tile_info_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_tile_stage
            )

            num_non_exiting_tiles_value = num_non_exiting_tiles[0]

            if cutlass.const_expr(self.raster_along_m):
                while work_tile.is_valid_tile:
                    cur_tile_coord = work_tile.tile_idx
                    mma_tile_coord_m = cur_tile_coord[0] // cute.size(
                        tiled_mma.thr_id.shape
                    )
                    if mma_tile_coord_m < num_non_exiting_tiles_value:
                        tile_info_pipeline.producer_acquire(tile_info_producer_state)
                        cur_tile_coord = work_tile.tile_idx
                        expert_idx = tile_idx_to_expert_idx[mma_tile_coord_m]
                        mn_limit = tile_idx_to_mn_limit[mma_tile_coord_m]
                        with cute.arch.elect_one():
                            sInfo[(0, tile_info_producer_state.index)] = cur_tile_coord[
                                0
                            ]
                            sInfo[(1, tile_info_producer_state.index)] = cur_tile_coord[
                                1
                            ]
                            sInfo[(2, tile_info_producer_state.index)] = expert_idx
                            sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(
                                work_tile.is_valid_tile
                            )
                            sInfo[(4, tile_info_producer_state.index)] = mn_limit
                            # fence view async shared
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )

                        self.sched_sync_barrier.arrive_and_wait()
                        tile_info_pipeline.producer_commit(tile_info_producer_state)
                        tile_info_producer_state.advance()

                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
            else:
                is_continue = cutlass.Boolean(1)
                while work_tile.is_valid_tile and is_continue:
                    cur_tile_coord = work_tile.tile_idx
                    mma_tile_coord_m = cur_tile_coord[0] // cute.size(
                        tiled_mma.thr_id.shape
                    )
                    if mma_tile_coord_m < num_non_exiting_tiles_value:
                        tile_info_pipeline.producer_acquire(tile_info_producer_state)
                        cur_tile_coord = work_tile.tile_idx
                        expert_idx = tile_idx_to_expert_idx[mma_tile_coord_m]
                        mn_limit = tile_idx_to_mn_limit[mma_tile_coord_m]
                        with cute.arch.elect_one():
                            sInfo[(0, tile_info_producer_state.index)] = cur_tile_coord[
                                0
                            ]
                            sInfo[(1, tile_info_producer_state.index)] = cur_tile_coord[
                                1
                            ]
                            sInfo[(2, tile_info_producer_state.index)] = expert_idx
                            sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(
                                work_tile.is_valid_tile
                            )
                            sInfo[(4, tile_info_producer_state.index)] = mn_limit
                            # fence view async shared
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )

                        self.sched_sync_barrier.arrive_and_wait()
                        tile_info_pipeline.producer_commit(tile_info_producer_state)
                        tile_info_producer_state.advance()
                    else:
                        is_continue = cutlass.Boolean(0)

                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

            tile_info_pipeline.producer_acquire(tile_info_producer_state)
            with cute.arch.elect_one():
                sInfo[(0, tile_info_producer_state.index)] = work_tile.tile_idx[0]
                sInfo[(1, tile_info_producer_state.index)] = work_tile.tile_idx[1]
                sInfo[(2, tile_info_producer_state.index)] = -1
                sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(0)
                sInfo[(4, tile_info_producer_state.index)] = -1
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            self.sched_sync_barrier.arrive_and_wait()
            tile_info_pipeline.producer_commit(tile_info_producer_state)
            tile_info_producer_state.advance()
            tile_info_pipeline.producer_tail(tile_info_producer_state)

        # Gather A warps (warps 4-7). cpasync / tma bodies static-gated by
        # cutlass.const_expr — only one is traced. cpasync: 4 warps × 32
        # threads issue a_num_loads CpAsync128.CG per k_tile (thread layout
        # (16, 8) covers 16 M-rows × 8 K-chunks); padded rows predicate off.
        if (
            warp_idx <= self.gather_a_warp_id[-1]
            and warp_idx >= self.gather_a_warp_id[0]
        ):
            cute.arch.setmaxregister_decrease(self.num_regs_gather_a_warps)
            if cutlass.const_expr(self.a_path == "cpasync"):
                a_atom_copy = cute.make_copy_atom(
                    cute.nvgpu.cpasync.CopyG2SOp(
                        cache_mode=cpasync.LoadCacheMode.GLOBAL
                    ),
                    mA_mkl.element_type,
                    num_bits_per_copy=128,
                )
                a_thread_layout = cute.make_layout((16, 8), stride=(8, 1))
                a_value_layout = cute.make_layout((1, 32), stride=(32, 1))
                a_tiled_copy = cute.make_tiled_copy_tv(
                    a_atom_copy,
                    a_thread_layout,
                    a_value_layout,
                )
                tidx_in_warpgroup = tidx % 128

                sA_tiled = cute.make_tensor(
                    sA.iterator,
                    layout=cute.make_layout(
                        (
                            self.cta_tile_shape_mnk[0],
                            self.cta_tile_shape_mnk[2],
                            self.num_ab_stage,
                        ),
                        stride=(
                            self.cta_tile_shape_mnk[2],
                            1,
                            self.cta_tile_shape_mnk[0] * self.cta_tile_shape_mnk[2],
                        ),
                    ),
                )
                a_thr_copy = a_tiled_copy.get_slice(tidx_in_warpgroup)
                tAsA_tiled = a_thr_copy.partition_D(sA_tiled)

                a_token_offset_tensor = cute.make_rmem_tensor(
                    cute.make_layout((self.a_num_loads,)),
                    cutlass.Int32,
                )
                a_predicate_tensor = cute.make_rmem_tensor(
                    cute.make_layout((self.a_num_loads,)),
                    cutlass.Boolean,
                )

                tile_sched = utils.StaticPersistentTileScheduler.create(
                    tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
                )
                work_tile = tile_sched.initial_work_tile_info()

                a_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_ab_stage
                )
                tile_info_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_tile_stage
                )

                tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(5, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

                while is_valid_tile:
                    gToken_ml_tile = gToken_ml[(None, tile_info[0])]
                    for i in range(self.a_num_loads):
                        token_ml_tile_offset = (tidx_in_warpgroup // 8) + i * 16
                        a_token_offset_tensor[i] = gToken_ml_tile[token_ml_tile_offset]
                        a_predicate_tensor[i] = (
                            cutlass.Boolean(1)
                            if tile_info[0] * self.cta_tile_shape_mnk[0]
                            + token_ml_tile_offset
                            < tile_info[4]
                            else cutlass.Boolean(0)
                        )
                        a_token_offset_tensor[i] = (
                            a_token_offset_tensor[i] // self.topk
                            if tile_info[0] * self.cta_tile_shape_mnk[0]
                            + token_ml_tile_offset
                            < tile_info[4]
                            else 0
                        )

                    tAgA = gA_mkl[(None, None, 0, None, 0)]
                    A_gmem_thread_offset = cute.assume(
                        (tidx_in_warpgroup % 8) * 32, divby=32
                    )

                    a_producer_state.reset_count()
                    peek_a_empty_status = cutlass.Boolean(1)
                    if a_producer_state.count < k_tile_cnt:
                        peek_a_empty_status = a_pipeline.producer_try_acquire(
                            a_producer_state
                        )

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        a_pipeline.producer_acquire(
                            a_producer_state, peek_a_empty_status
                        )

                        tAgA_ktile = tAgA[(None, None, a_producer_state.count)]
                        tAsA_ktile = tAsA_tiled[
                            (None, None, None, a_producer_state.index)
                        ]

                        for i in range(self.a_num_loads):
                            A_gmem_slice_offset = A_gmem_thread_offset + cute.assume(
                                a_token_offset_tensor[i] * tAgA_ktile.layout[0].stride,
                                divby=32,
                            )
                            A_gmem_slice_offset = cute.assume(
                                A_gmem_slice_offset, divby=32
                            )
                            tAgA_slice_ptr = tAgA_ktile.iterator + A_gmem_slice_offset
                            tAgA_slice = cute.make_tensor(
                                tAgA_slice_ptr, layout=cute.make_layout((32,))
                            )
                            tAsA_slice = cute.make_tensor(
                                tAsA_ktile[(None, i, None)].iterator,
                                layout=cute.make_layout((32,)),
                            )
                            a_predicate_slice = cute.make_rmem_tensor(
                                cute.make_layout((1,)), cutlass.Boolean
                            )
                            a_predicate_slice[0] = a_predicate_tensor[i]
                            cute.copy_atom_call(
                                a_atom_copy,
                                tAgA_slice,
                                tAsA_slice,
                                pred=a_predicate_slice,
                            )

                        a_pipeline.producer_commit(a_producer_state)

                        a_producer_state.advance()
                        peek_a_empty_status = cutlass.Boolean(1)
                        if a_producer_state.count < k_tile_cnt:
                            peek_a_empty_status = a_pipeline.producer_try_acquire(
                                a_producer_state
                            )

                    tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                    for idx in cutlass.range(5, unroll_full=True):
                        tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                    is_valid_tile = tile_info[3] == 1
                    cute.arch.fence_proxy("async.shared", space="cta")
                    tile_info_pipeline.consumer_release(tile_info_consumer_state)
                    tile_info_consumer_state.advance()

                a_pipeline.producer_tail(a_producer_state)

            # tma-A: 4 warps × elect-one issuing TMA gather4. Each warp owns
            # 1/4 of the M rows and issues n_gather_per_warp gather4 calls
            # (each pulling 4 rows × cta_tile_K). token_id == -1 → TMA
            # zero-fills the row. Signals merged ab_pipeline mbar (shared w/ B).
            elif cutlass.const_expr(self.a_path == "tma"):
                warp_rel = warp_idx - self.gather_a_warp_id[0]
                rows_per_warp = self.cta_tile_shape_mnk[0] // self.tma_gather_num_warps
                n_gather_per_warp = rows_per_warp // 4

                a_row_ids = cute.make_rmem_tensor(
                    cute.make_layout((rows_per_warp,)),
                    cutlass.Int32,
                )

                tile_sched = utils.StaticPersistentTileScheduler.create(
                    tile_sched_params,
                    cute.arch.block_idx(),
                    cute.arch.grid_dim(),
                )
                work_tile = tile_sched.initial_work_tile_info()

                ab_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_ab_stage
                )
                tile_info_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_tile_stage
                )

                tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(5, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

                while is_valid_tile:
                    gToken_ml_tile = gToken_ml[(None, tile_info[0])]

                    # Each warp computes its own row range (1/4 of the tile).
                    # The routing helper initializes padded mapping entries to
                    # zero, so use mn_limit as the authoritative valid-row
                    # predicate before passing -1 to gather4 for zero-fill.
                    for i in range(rows_per_warp):
                        row_global = warp_rel * rows_per_warp + i
                        token_id = gToken_ml_tile[row_global]
                        valid_row = (
                            tile_info[0] * self.cta_tile_shape_mnk[0] + row_global
                            < tile_info[4]
                        )
                        row_id = (
                            token_id // self.topk if valid_row else cutlass.Int32(-1)
                        )
                        a_row_ids[i] = cutlass.Int32(-1) if token_id == -1 else row_id

                    # A multicast leader gate: when A is N-multicast, only the N=0
                    # CTA issues the mcast PTX; HW broadcasts data + mbar tx_count
                    # to the N-peer. The peer's producer_acquire still sets its
                    # mbar's expect_tx; that mbar fires via HW routing.
                    is_a_mcast_leader = (not self.is_a_mcast) or (
                        block_in_cluster_coord_vmnk[2] == 0
                    )
                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        # gather_a contributes a_num_tma_load_bytes per warp to
                        # the merged ab_pipeline mbar (4 such arrivals per stage).
                        ab_pipeline.producer_acquire(
                            ab_producer_state,
                            expected_tx=self.a_num_tma_load_bytes,
                        )

                        if is_a_mcast_leader:
                            with cute.arch.elect_one():
                                col_k = k_tile * self.cta_tile_shape_mnk[2]
                                mbar_ptr = ab_pipeline.producer_get_barrier(
                                    ab_producer_state
                                )
                                stage_base_elements = (
                                    ab_producer_state.index
                                    * self.cta_tile_shape_mnk[0]
                                    * self.cta_tile_shape_mnk[2]
                                )
                                warp_base_elements = (
                                    warp_rel
                                    * rows_per_warp
                                    * self.cta_tile_shape_mnk[2]
                                )

                                for g in range(n_gather_per_warp):
                                    row_start = g * 4
                                    dst_offset = (
                                        stage_base_elements
                                        + warp_base_elements
                                        + row_start * self.cta_tile_shape_mnk[2]
                                    )
                                    dst_ptr = sA.iterator + dst_offset
                                    sm100_tma_gather4_load(
                                        tma_atom_a,
                                        dst_ptr,
                                        mbar_ptr,
                                        col_k,
                                        a_row_ids[row_start],
                                        a_row_ids[row_start + 1],
                                        a_row_ids[row_start + 2],
                                        a_row_ids[row_start + 3],
                                        use_cta_group_2=self.use_2cta_instrs,
                                        mcast_mask=a_full_mcast_mask,
                                    )

                        ab_pipeline.producer_commit(ab_producer_state)
                        ab_producer_state.advance()

                    tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                    for idx in cutlass.range(5, unroll_full=True):
                        tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                    is_valid_tile = tile_info[3] == 1
                    cute.arch.fence_proxy("async.shared", space="cta")
                    tile_info_pipeline.consumer_release(tile_info_consumer_state)
                    tile_info_consumer_state.advance()

                ab_pipeline.producer_tail(ab_producer_state)

        # Gather SFA warps (warps 12-15): CpAsync128.CG per row. 4 warps × 32
        # threads, each loads one row (16B = 16 FP8 SFs). sSFA is plain linear
        # (M, tile_K_sf, stage). tiled_copy_tv for SMEM dest + manual per-thread
        # GMEM source tensor.
        if (
            warp_idx <= self.gather_sfa_warp_id[-1]
            and warp_idx >= self.gather_sfa_warp_id[0]
        ):
            cute.arch.setmaxregister_decrease(self.num_regs_gather_sfa_warps)

            sfa_tile_k_sf = self.cta_tile_shape_mnk[2] // self.sf_vec_size  # 16
            sfa_gather_threads = len(self.gather_sfa_warp_id) * self.threads_per_warp
            # Rows per thread = cta_tile_M / gather_threads.
            # - non-breuse cta_tile_M=128: 1 row per thread
            # - breuse cta_tile_M=256: 2 rows per thread (outer r-loop in inner k loop)
            sfa_rows_per_thread = self.cta_tile_shape_mnk[0] // sfa_gather_threads

            # One CpAsync128.CG per thread per row per k_tile.
            sfa_atom_copy = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                mSFA_mkl.element_type,
                num_bits_per_copy=128,
            )

            sSFA_tiled = cute.make_tensor(
                sSFA.iterator,
                layout=cute.make_layout(
                    (
                        self.cta_tile_shape_mnk[0],
                        sfa_tile_k_sf,
                        self.num_ab_stage,
                    ),
                    stride=(
                        sfa_tile_k_sf,
                        1,
                        self.cta_tile_shape_mnk[0] * sfa_tile_k_sf,
                    ),
                ),
            )

            tidx_in_warpgroup = (
                tidx - self.gather_sfa_warp_id[0] * self.threads_per_warp
            )

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            sfa_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(5, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                gToken_ml_tile = gToken_ml[(None, tile_info[0])]

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    sfa_pipeline.producer_acquire(sfa_producer_state)

                    # For each row this thread is responsible for
                    # (1 row at non-breuse cta_tile_M=128; 2 rows at breuse
                    # cta_tile_M=256). Inner r-loop fully unrolls.
                    for r in cutlass.range_constexpr(sfa_rows_per_thread):
                        row_in_cta = tidx_in_warpgroup + r * sfa_gather_threads

                        # Per-row token id → SFA gmem row index.
                        tok = gToken_ml_tile[row_in_cta]
                        sfa_row_id = (
                            cutlass.Int32(-1) if tok == -1 else tok // self.topk
                        )

                        # OOB predicate: rows past mn_limit do not store.
                        sfa_pred = cute.make_rmem_tensor(
                            cute.make_layout((1,)), cutlass.Boolean
                        )
                        sfa_pred[0] = (
                            cutlass.Boolean(1)
                            if (
                                tile_info[0] * self.cta_tile_shape_mnk[0] + row_in_cta
                                < tile_info[4]
                            )
                            else cutlass.Boolean(0)
                        )

                        # GMEM src for this row + k_tile.
                        tAgSFA_row = gSFA_mkl[(sfa_row_id, None, 0, None, 0)]
                        tAgSFA_ktile = tAgSFA_row[(None, k_tile)]
                        tAgSFA_slice = cute.make_tensor(
                            tAgSFA_ktile.iterator,
                            layout=cute.make_layout((sfa_tile_k_sf,)),
                        )

                        # SMEM dst: direct row-indexed slice (bypass partition_D
                        # so we naturally handle multiple rows per thread).
                        tAsSFA_slice = cute.make_tensor(
                            sSFA_tiled[
                                (row_in_cta, None, sfa_producer_state.index)
                            ].iterator,
                            cute.make_layout((sfa_tile_k_sf,)),
                        )

                        cute.copy_atom_call(
                            sfa_atom_copy,
                            tAgSFA_slice,
                            tAsSFA_slice,
                            pred=sfa_pred,
                        )

                    sfa_pipeline.producer_commit(sfa_producer_state)
                    sfa_producer_state.advance()

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(5, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            sfa_pipeline.producer_tail(sfa_producer_state)

        #
        # SFA Transform warps (warps 16-19) — shared-memory load + tcgen05.st consumer
        #
        if (
            warp_idx >= self.sfa_transform_warp_id[0]
            and warp_idx <= self.sfa_transform_warp_id[-1]
        ):
            cute.arch.setmaxregister_decrease(self.num_regs_sfa_transform_warps)

            if cutlass.const_expr(True):
                # Transform warps wait for TMEM alloc and compute
                # sfa_tmem_ptr = acc_tmem_ptr + offset for tcgen05.st destination.
                tmem.wait_for_alloc()
                acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                sfa_tmem_ptr = cute.recast_ptr(
                    acc_tmem_ptr + self.sfa_tmem_offset,
                    dtype=self.sf_dtype,
                )

                tile_sched = utils.StaticPersistentTileScheduler.create(
                    tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
                )
                _ = tile_sched.initial_work_tile_info()

                tile_info_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_tile_stage
                )
                sfa_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_ab_stage
                )
                # Producer state on sfa_transform_pipeline: tracks which TMEM
                # stage (of num_sfa_tmem_stage) is being filled this iter.
                sfa_transform_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer,
                    self.num_sfa_tmem_stage,
                )
                tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(5, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

                # All 4 transform warps run identical address logic: each does
                # the full cta_tile_M shared-memory load+tcgen05.st into its own per-warp TMEM lane;
                # UMMA reads all 4 to cover M. M-blocks = cta_tile_M / 32 (4
                # non-breuse, 8 breuse). Per warp: num_m_blocks 128-bit shared-memory loads + 4 tcgen05.st.
                num_m_blocks = self.cta_tile_shape_mnk[0] // 32
                lane_in_warp = tidx % 32

                while is_valid_tile:
                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        sfa_pipeline.consumer_wait(sfa_consumer_state)

                        # 128-bit shared-memory load -> registers (Uint32). Per ci read 16 SFs = 4 u32.
                        # Issue shared-memory load before producer_acquire so SMEM→RMEM moves
                        # overlap with prior MMA still holding the next TMEM
                        # stage; tcgen05.st still gates on producer_acquire below.
                        sfa_rmem_u32 = cute.make_rmem_tensor(
                            cute.make_layout((num_m_blocks, 4)),  # (ci, K-group)
                            cutlass.Uint32,
                        )
                        # Linear sSFA (M, K_sf, stage). Each shared-memory load reads 16
                        # contiguous FP8 SFs (4 u32) at row_global of the
                        # current stage.
                        for ci in range(num_m_blocks):
                            row_global = 32 * ci + lane_in_warp
                            smem_slice = sSFA[
                                (
                                    row_global,
                                    None,
                                    sfa_consumer_state.index,
                                )
                            ]
                            smem_slice_u32 = cute.make_tensor(
                                cute.recast_ptr(
                                    smem_slice.iterator, dtype=cutlass.Uint32
                                ),
                                cute.make_layout((4,)),
                            )
                            cute.autovec_copy(smem_slice_u32, sfa_rmem_u32[ci, None])

                        # Acquire SFA TMEM slot (MMA's UMMA consumer release
                        # frees it after consuming). Done after shared-memory load so the
                        # SMEM→RMEM stage isn't blocked on TMEM availability.
                        sfa_transform_pipeline.producer_acquire(
                            sfa_transform_producer_state
                        )

                        # tcgen05.st via inline PTX. tCtSFA_layout cols:
                        #   non-breuse (M=128): 1 half, cols 0..15 = 4 K-groups
                        #     × 4 M-blocks (gi*4 stride). 4 tcgen05.st x4.
                        #   breuse (M=256): 2 halves at +16 col offset (keep
                        #     ci 0..3 / reuse ci 4..7), each 4 tcgen05.st x4. 8 tcgen05.st x4 total.
                        stage_idx_in_tmem = sfa_transform_producer_state.index
                        sfa_tmem_addr_base = (
                            acc_tmem_ptr
                            + self.sfa_tmem_offset
                            + stage_idx_in_tmem * self.num_sfa_tmem_cols_per_stage
                        ).toint()
                        # Number of "halves" (keep/reuse splits) and the stride
                        # between them. non-breuse: 1 half, no second offset.
                        # breuse: 2 halves, second at +16 cols.
                        num_halves = num_m_blocks // 4  # 1 or 2
                        half_col_stride = 16  # cols between keep and reuse
                        for half in range(num_halves):
                            half_addr_base = sfa_tmem_addr_base + half * half_col_stride
                            ci_base = half * 4
                            for gi in range(4):
                                sm100_tcgen05_st_32x32b_x4(
                                    half_addr_base + gi * 4,
                                    sfa_rmem_u32[ci_base + 0, gi],
                                    sfa_rmem_u32[ci_base + 1, gi],
                                    sfa_rmem_u32[ci_base + 2, gi],
                                    sfa_rmem_u32[ci_base + 3, gi],
                                )
                        # Make TMEM stores visible to UMMA, then commit the
                        # transform pipeline producer slot.
                        cute.arch.fence_view_async_tmem_store()
                        sfa_transform_pipeline.producer_commit(
                            sfa_transform_producer_state
                        )
                        sfa_transform_producer_state.advance()

                        sfa_pipeline.consumer_release(sfa_consumer_state)
                        sfa_consumer_state.advance()

                    tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                    for idx in cutlass.range(5, unroll_full=True):
                        tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                    is_valid_tile = tile_info[3] == 1
                    cute.arch.fence_proxy("async.shared", space="cta")
                    tile_info_pipeline.consumer_release(tile_info_consumer_state)
                    tile_info_consumer_state.advance()

                # Drain transform pipeline before exit.
                sfa_transform_pipeline.producer_tail(sfa_transform_producer_state)

        # A Sync-Transform Warp (warp 11). Active only on cpasync 2CTA.
        # Consumes per-CTA `a_pipeline` and re-produces cluster-wide
        # `a_sync_transform_pipeline` so MMA's cta_group::2 sees both CTAs' A.
        # SFA needs no relay — its transform warps + sfa_transform_pipeline
        # handle cluster arrives. Idle on 1CTA / tma-A.
        if warp_idx == self.sync_transform_warp_id:
            if cutlass.const_expr(self.a_path == "cpasync" and self.use_2cta_instrs):
                tile_sched = utils.StaticPersistentTileScheduler.create(
                    tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
                )
                work_tile = tile_sched.initial_work_tile_info()

                a_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_ab_stage
                )
                a_sync_transform_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_ab_stage
                )
                tile_info_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_tile_stage
                )

                # First tile info (only need the validity flag).
                valid_tile_info = cute.make_rmem_tensor((1,), cutlass.Int32)
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                valid_tile_info[0] = sInfo[(3, tile_info_consumer_state.index)]
                is_valid_tile = valid_tile_info[0] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

                while is_valid_tile:
                    a_consumer_state.reset_count()
                    peek_a_full_status = cutlass.Boolean(1)
                    if a_consumer_state.count < k_tile_cnt:
                        peek_a_full_status = a_pipeline.consumer_try_wait(
                            a_consumer_state
                        )
                    a_sync_transform_producer_state.reset_count()

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        # Wait per-CTA A full → commit cluster-wide A sync-transform
                        # full. We do NOT release a_pipeline here; MMA owns its
                        # consumer_release so each CTA's producer sees the empty arrive.
                        a_pipeline.consumer_wait(a_consumer_state, peek_a_full_status)
                        a_sync_transform_pipeline.producer_commit(
                            a_sync_transform_producer_state
                        )
                        a_sync_transform_producer_state.advance()
                        a_consumer_state.advance()
                        peek_a_full_status = cutlass.Boolean(1)
                        if a_consumer_state.count < k_tile_cnt:
                            peek_a_full_status = a_pipeline.consumer_try_wait(
                                a_consumer_state
                            )

                    # Advance to next tile
                    tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                    valid_tile_info[0] = sInfo[(3, tile_info_consumer_state.index)]
                    is_valid_tile = valid_tile_info[0] == 1
                    cute.arch.fence_proxy("async.shared", space="cta")
                    tile_info_pipeline.consumer_release(tile_info_consumer_state)
                    tile_info_consumer_state.advance()

                # Drain A sync-transform pipeline before exit.
                a_sync_transform_pipeline.producer_tail(a_sync_transform_producer_state)

        # TMA B/SFB load warp (warp 9). Loads B/SFB GMEM → SMEM with multicast.
        if warp_idx == self.tma_b_warp_id:
            # B producer signals the pipeline owning A+B (tma → ab_pipeline)
            # or B alone (cpasync → b_pipeline). Body is the same; only the
            # pipeline alias and producer_acquire's expected_tx differ.
            if cutlass.const_expr(self.a_path == "cpasync"):
                _b_pipe = b_pipeline
            else:
                _b_pipe = ab_pipeline
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            # First tile
            work_tile = tile_sched.initial_work_tile_info()

            b_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get the first tile info
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )
                #
                # Slice to per mma tile index
                #
                # ((atom_v, rest_v), loopK)
                tBgB_slice = tBgB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                # Apply SFB slicing hack when cta_tile_shape_n=64
                slice_n = mma_tile_coord_mnl[1]
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = mma_tile_coord_mnl[1] // 2

                # ((atom_v, rest_v), RestK)
                tBgSFB_slice = tBgSFB[(None, slice_n, None, mma_tile_coord_mnl[2])]

                # Peek (try_wait) B buffer empty for k_tile = prefetch_k_tile_cnt
                b_producer_state.reset_count()
                peek_b_empty_status = cutlass.Boolean(1)
                if b_producer_state.count < k_tile_cnt:
                    peek_b_empty_status = _b_pipe.producer_try_acquire(b_producer_state)
                #
                # Tma load loop
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # Conditionally wait for B buffer empty.
                    # tma-A mode passes expected_tx (ab_pipeline has tx_count=0
                    # at create and accumulates per producer call); cpasync-A
                    # mode's b_pipeline has tx_count fixed at create.
                    if cutlass.const_expr(self.a_path == "cpasync"):
                        _b_pipe.producer_acquire(b_producer_state, peek_b_empty_status)
                    else:
                        _b_pipe.producer_acquire(
                            b_producer_state,
                            peek_b_empty_status,
                            expected_tx=self.num_tma_load_bytes,
                        )

                    tBgB_k = tBgB_slice[(None, b_producer_state.count)]
                    tBgSFB_k = tBgSFB_slice[(None, b_producer_state.count)]
                    tBsB_pipe = tBsB[(None, b_producer_state.index)]
                    tBsSFB_pipe = tBsSFB[(None, b_producer_state.index)]

                    tma_bar = _b_pipe.producer_get_barrier(b_producer_state)

                    # TMA load B
                    cute.copy(
                        tma_atom_b,
                        tBgB_k,
                        tBsB_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=b_full_mcast_mask,
                    )

                    # TMA load SFB
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_k,
                        tBsSFB_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=sfb_full_mcast_mask,
                    )

                    # Peek (try_wait) B buffer empty for k_tile + 1
                    b_producer_state.advance()
                    peek_b_empty_status = cutlass.Boolean(1)
                    if b_producer_state.count < k_tile_cnt:
                        peek_b_empty_status = _b_pipe.producer_try_acquire(
                            b_producer_state
                        )

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            #
            # Wait B buffer empty
            #
            _b_pipe.producer_tail(b_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # Bar sync for retrieve tensor memory ptr from shared mem
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # SFA TMEM base ptr (sf_dtype). Stage-indexed tCtSFA is rebuilt
            # per k_tile inside the loop using sfa_transform_consumer_state.index.
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )

            # Make SFB tmem tensor (using precomputed layout)
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            # SFA TMEM is filled by transform warps via shared-memory load+tcgen05.st (no tcgen05.cp).
            # Only SFB uses tcgen05.cp.
            sfb_s2t_bundle = self._mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            # MMA consumer states. cpasync: separate a / b states
            # (+ a_sync_transform in 2CTA). tma: single ab_consumer_state.
            if cutlass.const_expr(self.a_path == "cpasync"):
                a_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_ab_stage
                )
                b_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_ab_stage
                )
                if cutlass.const_expr(self.use_2cta_instrs):
                    a_sync_transform_consumer_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, self.num_ab_stage
                    )
            else:  # "tma"
                ab_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_ab_stage
                )
                # a_consumer_state and b_consumer_state both aliased to
                # ab_consumer_state so existing `.index` reads in shared
                # SMEM-slice code (sA[stage] / sB[stage] in the MMA mainloop)
                # still resolve (A and B share stages in tma-A mode).
                a_consumer_state = ab_consumer_state
                b_consumer_state = ab_consumer_state
            # sfa_transform_pipeline has num_sfa_tmem_stage slots, not num_ab_stage.
            sfa_transform_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_sfa_tmem_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get the first tile info from pipeline (scheduler has filtered out tiles >= num_non_exiting_tiles)
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                # Peek (try_wait) A / B / SFA buffer full for k_tile = 0.
                # cpasync-A 1CTA: A peek on a_pipeline.
                # cpasync-A 2CTA: A peek on a_sync_transform_pipeline (cluster-wide).
                # tma-A: A+B peek on ab_pipeline (single shared mbar).
                if cutlass.const_expr(self.a_path == "cpasync"):
                    if cutlass.const_expr(self.use_2cta_instrs):
                        a_sync_transform_consumer_state.reset_count()
                        peek_a_sync_transform_full_status = cutlass.Boolean(1)
                        if (
                            a_sync_transform_consumer_state.count < k_tile_cnt
                            and is_leader_cta
                        ):
                            peek_a_sync_transform_full_status = (
                                a_sync_transform_pipeline.consumer_try_wait(
                                    a_sync_transform_consumer_state
                                )
                            )
                        a_consumer_state.reset_count()
                    else:
                        a_consumer_state.reset_count()
                        peek_a_full_status = cutlass.Boolean(1)
                        if a_consumer_state.count < k_tile_cnt:
                            peek_a_full_status = a_pipeline.consumer_try_wait(
                                a_consumer_state
                            )
                    b_consumer_state.reset_count()
                    peek_b_full_status = cutlass.Boolean(1)
                    if b_consumer_state.count < k_tile_cnt and is_leader_cta:
                        peek_b_full_status = b_pipeline.consumer_try_wait(
                            b_consumer_state
                        )
                else:  # "tma"
                    ab_consumer_state.reset_count()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                        peek_ab_full_status = ab_pipeline.consumer_try_wait(
                            ab_consumer_state
                        )
                sfa_transform_consumer_state.reset_count()
                peek_sfa_full_status = cutlass.Boolean(1)
                if sfa_transform_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_sfa_full_status = sfa_transform_pipeline.consumer_try_wait(
                        sfa_transform_consumer_state
                    )

                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )

                # Accumulator stage. Overlap mode: pipeline tracks 1 stage
                # but TMEM has 2 regions (acc[0] cols 0..255, acc[1] 192..447);
                # use phase to pick. producer_state.phase starts at 1, so XOR
                # with 1 to align with TMEM slot 0 on tile 0.
                if cutlass.const_expr(self.use_overlap_accum):
                    acc_stage_index = acc_producer_state.phase ^ 1
                else:
                    acc_stage_index = acc_producer_state.index

                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                # Apply TMEM pointer offset hack when cta_tile_shape_n=192 or
                # cta_tile_shape_n=64
                tCtSFB_mma = tCtSFB
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
                    # If this is an ODD tile, shift the TMEM start address for
                    # cta_tile_shape_n=192 case by two words
                    # (ignores first 64 columns of SFB)
                    offset = (
                        cutlass.Int32(2)
                        if mma_tile_coord_mnl[1] % 2 == 1
                        else cutlass.Int32(0)
                    )
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + self.num_accumulator_tmem_cols
                        + self.num_sfa_tmem_cols
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
                elif cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    # Move in increments of 64 columns of SFB
                    offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + self.num_accumulator_tmem_cols
                        + self.num_sfa_tmem_cols
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
                    #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)
                #
                # Mma mainloop
                #

                for k_tile in cutlass.range(k_tile_cnt):
                    # Set tensor memory buffer for current tile
                    # (MMA, MMA_M, MMA_N)

                    if is_leader_cta:
                        # Wait for A / B / SFA buffer full.
                        # cpasync-A: 2 separate waits (A side + B side).
                        #   2CTA A wait uses a_sync_transform_pipeline (relay).
                        # tma-A: single ab_pipeline wait covers A and B.
                        if cutlass.const_expr(self.a_path == "cpasync"):
                            if cutlass.const_expr(self.use_2cta_instrs):
                                a_sync_transform_pipeline.consumer_wait(
                                    a_sync_transform_consumer_state,
                                    peek_a_sync_transform_full_status,
                                )
                            else:
                                a_pipeline.consumer_wait(
                                    a_consumer_state, peek_a_full_status
                                )
                            b_pipeline.consumer_wait(
                                b_consumer_state, peek_b_full_status
                            )
                            a_stage_idx = a_consumer_state.index
                            b_stage_idx = b_consumer_state.index
                        else:  # "tma"
                            ab_pipeline.consumer_wait(
                                ab_consumer_state, peek_ab_full_status
                            )
                            # Read ab_consumer_state.index directly (NOT via
                            # the aliased a_/b_consumer_state names): CuteDSL
                            # caches .index SSA values per variable *name*,
                            # so aliased reads go stale after ab.advance().
                            a_stage_idx = ab_consumer_state.index
                            b_stage_idx = ab_consumer_state.index
                        sfa_transform_pipeline.consumer_wait(
                            sfa_transform_consumer_state, peek_sfa_full_status
                        )

                        # Rebuild tCtSFA pointing at the current SFA TMEM
                        # stage (transform warps rotate through stages;
                        # sfa_transform_consumer_state.index tracks which to read).
                        tCtSFA = cute.make_tensor(
                            sfa_tmem_ptr
                            + sfa_transform_consumer_state.index
                            * self.num_sfa_tmem_cols_per_stage
                            * 4,
                            tCtSFA_layout,
                        )

                        # SFB tcgen05.cp (SFA is already in TMEM from transform warps).
                        self._mainloop_s2t_copies(b_stage_idx, sfb_s2t_bundle)

                        num_kblocks = cute.size(tCrA, mode=[2])

                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            if cutlass.const_expr(
                                self.enable_breuse
                                and cute.size(tCtAcc.layout, mode=[1]) == 2
                                and cute.size(tCtAcc.layout, mode=[2]) == 1
                            ):
                                tCtAcc_bkeep = tCtAcc[(None, 0, 0)]
                                tCtAcc_breuse = tCtAcc[(None, 1, 0)]

                                a_kblk_crd_keep = (None, 0, kblock_idx, a_stage_idx)
                                a_kblk_crd_reuse = (None, 1, kblock_idx, a_stage_idx)
                                b_kblk_crd = (None, 0, kblock_idx, b_stage_idx)

                                sfa_kblk_crd_keep = (None, 0, kblock_idx)
                                sfa_kblk_crd_reuse = (None, 1, kblock_idx)
                                sfb_kblk_crd = (None, 0, kblock_idx)

                                # Bkeep
                                tiled_mma_bkeep.set(
                                    tcgen05.Field.ACCUMULATE,
                                    k_tile != 0 or kblock_idx != 0,
                                )
                                cute.gemm(
                                    tiled_mma_bkeep,
                                    tCtAcc_bkeep,
                                    [tCrA[a_kblk_crd_keep], tCtSFA[sfa_kblk_crd_keep]],
                                    [tCrB[b_kblk_crd], tCtSFB_mma[sfb_kblk_crd]],
                                    tCtAcc_bkeep,
                                )
                                # Breuse
                                tiled_mma_breuse.set(
                                    tcgen05.Field.ACCUMULATE,
                                    k_tile != 0 or kblock_idx != 0,
                                )
                                cute.gemm(
                                    tiled_mma_breuse,
                                    tCtAcc_breuse,
                                    [
                                        tCrA[a_kblk_crd_reuse],
                                        tCtSFA[sfa_kblk_crd_reuse],
                                    ],
                                    [tCrB[b_kblk_crd], tCtSFB_mma[sfb_kblk_crd]],
                                    tCtAcc_breuse,
                                )
                            else:
                                a_kblock_coord = (None, None, kblock_idx, a_stage_idx)
                                b_kblock_coord = (None, None, kblock_idx, b_stage_idx)
                                sf_kblock_coord = (None, None, kblock_idx)

                                tiled_mma.set(
                                    tcgen05.Field.ACCUMULATE,
                                    k_tile != 0 or kblock_idx != 0,
                                )
                                cute.gemm(
                                    tiled_mma,
                                    tCtAcc,
                                    [tCrA[a_kblock_coord], tCtSFA[sf_kblock_coord]],
                                    [tCrB[b_kblock_coord], tCtSFB_mma[sf_kblock_coord]],
                                    tCtAcc,
                                )

                        # Release A/B/SFA buffer-empty (async arrive).
                        # cpasync: a_pipeline + a_sync_transform_pipeline
                        # (2CTA only) + b_pipeline. tma: single ab_pipeline.
                        if cutlass.const_expr(self.a_path == "cpasync"):
                            a_pipeline.consumer_release(a_consumer_state)
                            if cutlass.const_expr(self.use_2cta_instrs):
                                a_sync_transform_pipeline.consumer_release(
                                    a_sync_transform_consumer_state
                                )
                            b_pipeline.consumer_release(b_consumer_state)
                        else:  # "tma"
                            ab_pipeline.consumer_release(ab_consumer_state)
                        sfa_transform_pipeline.consumer_release(
                            sfa_transform_consumer_state
                        )

                    # Peek (try_wait) A / B / SFA buffer full for k_tile + 1.
                    if cutlass.const_expr(self.a_path == "cpasync"):
                        if cutlass.const_expr(self.use_2cta_instrs):
                            a_sync_transform_consumer_state.advance()
                            peek_a_sync_transform_full_status = cutlass.Boolean(1)
                            if (
                                a_sync_transform_consumer_state.count < k_tile_cnt
                                and is_leader_cta
                            ):
                                peek_a_sync_transform_full_status = (
                                    a_sync_transform_pipeline.consumer_try_wait(
                                        a_sync_transform_consumer_state
                                    )
                                )
                            a_consumer_state.advance()
                        else:
                            a_consumer_state.advance()
                            peek_a_full_status = cutlass.Boolean(1)
                            if a_consumer_state.count < k_tile_cnt:
                                peek_a_full_status = a_pipeline.consumer_try_wait(
                                    a_consumer_state
                                )
                        b_consumer_state.advance()
                        peek_b_full_status = cutlass.Boolean(1)
                        if b_consumer_state.count < k_tile_cnt and is_leader_cta:
                            peek_b_full_status = b_pipeline.consumer_try_wait(
                                b_consumer_state
                            )
                    else:  # "tma"
                        ab_consumer_state.advance()
                        peek_ab_full_status = cutlass.Boolean(1)
                        if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                ab_consumer_state
                            )
                    sfa_transform_consumer_state.advance()
                    peek_sfa_full_status = cutlass.Boolean(1)
                    if (
                        sfa_transform_consumer_state.count < k_tile_cnt
                        and is_leader_cta
                    ):
                        peek_sfa_full_status = sfa_transform_pipeline.consumer_try_wait(
                            sfa_transform_consumer_state
                        )

                #
                # Async arrive accumulator buffer full(each kblock)
                #
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)

                # Peek (try_wait) Acc buffer empty for k_tile = k_tile + 1
                acc_producer_state.advance()

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)

        #
        # Specialized epilogue warps
        #
        if warp_idx <= self.epilog_warp_id[-1]:
            # Register reconfig: epilogue needs many regs for SwiGLU
            cute.arch.setmaxregister_increase(self.num_regs_epilogue_warps)
            #
            # Alloc tensor memory buffer
            #
            tmem.allocate(self.num_tmem_alloc_cols)

            #
            # Bar sync for retrieve tensor memory ptr from shared memory
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            # Epilogue partition: transform both accumulator and C layout.
            # transform_partitioned_tensor_layout merges (MMA_ATOM, MMA_M) → flat M.
            tCtAcc_transformed = transform_partitioned_tensor_layout(tCtAcc_base)
            tCgC_for_epi = transform_partitioned_tensor_layout(tCgC)

            epi_tidx = tidx % 128
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc_up,
                tTR_rAcc_gate,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_transformed, tCgC_for_epi, epi_tile, use_2cta_instrs
            )

            tTR_rC = None
            tiled_copy_r2s = None
            tRS_rC = None
            tRS_sC = None
            bSG_sC = None
            bSG_gC_partitioned = None
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc_up.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = epilogue_smem_copy_and_partition(
                self, tiled_copy_t2r, tTR_rC, epi_tidx, sC
            )
            (
                tma_atom_c,
                bSG_sC,
                bSG_gC_partitioned,
            ) = self.epilog_gmem_copy_and_partition(
                epi_tidx, tma_atom_c, tCgC_for_epi, epi_tile, sC
            )

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            c_pipeline = None
            # Threads/warps participating in tma store pipeline
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=c_producer_group,
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get the first tile info (m tile, n tile, expert, valid, mn_limit)
            tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)

            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(5, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            num_prev_subtiles = cutlass.Int32(0)
            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )
                #
                # Per-expert gate / up global scales
                #
                expert_idx = mma_tile_coord_mnl[2]
                alpha_gate_val = alpha_gate[expert_idx]
                alpha_up_val = alpha_up[expert_idx]
                tile_m_start = tile_info[0] * self.cta_tile_shape_mnk[0]

                #
                # Slice to per mma tile index
                #
                bSG_gC = None
                # ((ATOM_V, REST_V), EPI_M, EPI_N)
                bSG_gC = bSG_gC_partitioned[
                    (
                        None,
                        None,
                        None,
                        mma_tile_coord_mnl[0],
                        mma_tile_coord_mnl[1],
                        0,
                    )
                ]

                # Get accumulator stage index. Overlap mode uses phase
                # (alternates 0/1) and reverse-iterates acc[0]'s subtiles so
                # the overlap region (high cols of acc[0]) is consumed first
                # → early-release lets MMA write acc[1] in those cols.
                if cutlass.const_expr(self.use_overlap_accum):
                    # Consumer state starts at phase=0 (per make_pipeline_state)
                    # so phase directly maps to TMEM slot index.
                    acc_stage_index = acc_consumer_state.phase
                    reverse_subtile = (
                        cutlass.Boolean(True)
                        if acc_stage_index == 0
                        else cutlass.Boolean(False)
                    )
                else:
                    acc_stage_index = acc_consumer_state.index

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[
                    (None, None, None, None, None, acc_stage_index)
                ]

                #
                acc_pipeline.consumer_wait(acc_consumer_state)

                # SwiGLU epilogue. Output subtile j (epi_tile_n output columns) reads accumulator subtiles 2j and
                # 2j + 1. In the weight row order every 32 accumulator columns hold
                # [up(i0 + 2a) | gate(i0 + 2a) | up(i0 + 2a + 1) | gate(i0 + 2a + 1)], a = 0..7, i0 = 16 * block,
                # which become the 16 consecutive output columns i0 .. i0 + 15.
                #   tTR_tAcc: (T2R, T2R_M, T2R_N, EPI_M, EPI_N, STAGE), sliced on STAGE.
                #   bSG_gC:   ((ATOM_V, REST_V), EPI_M, EPI_N, loopM, loopN, loopL).
                epi_m_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                acc_n_subtile_cnt = cute.size(tTR_tAcc.shape, mode=[4])
                out_n_subtile_cnt = (
                    acc_n_subtile_cnt // 2
                )  # N/2 output subtiles per M subtile

                for epi_m_idx in cutlass.range(epi_m_cnt):
                    # Input scale of this thread's row (one accumulator row per epilogue thread).
                    permuted_row = (
                        tile_m_start + epi_m_idx * self.epi_tile[0] + epi_tidx
                    )
                    row_scale = cutlass.Float32(0.0)
                    if permuted_row < tile_info[4]:
                        expanded_idx = token_id_mapping_tensor[permuted_row]
                        if expanded_idx >= 0:
                            row_scale = token_scale[expanded_idx // self.topk]
                    gate_scale = cutlass.Float32(alpha_gate_val) * row_scale
                    up_scale = cutlass.Float32(alpha_up_val) * row_scale
                    for out_n_idx in cutlass.range(out_n_subtile_cnt):
                        # acc[0] in overlap mode iterates in reverse (consume high-col overlap region first).
                        if cutlass.const_expr(self.use_overlap_accum):
                            real_out_n_idx = (
                                (out_n_subtile_cnt - 1 - out_n_idx)
                                if reverse_subtile
                                else out_n_idx
                            )
                        else:
                            real_out_n_idx = out_n_idx
                        lo_n_subtile = real_out_n_idx * 2
                        hi_n_subtile = real_out_n_idx * 2 + 1
                        #
                        # Load accumulator from tensor memory buffer to register
                        #
                        tTR_tAcc_mn_lo = tTR_tAcc[
                            (None, None, None, epi_m_idx, lo_n_subtile)
                        ]
                        tTR_tAcc_mn_hi = tTR_tAcc[
                            (None, None, None, epi_m_idx, hi_n_subtile)
                        ]

                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn_lo, tTR_rAcc_up)
                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn_hi, tTR_rAcc_gate)

                        # Overlap mode: after iter 0 the tcgen05.ld has covered the overlap columns (reverse for
                        # acc[0], forward for acc[1]). Fence + early-release so MMA can write the next stage into the
                        # overlap region without racing.
                        if cutlass.const_expr(self.use_overlap_accum):
                            if out_n_idx == 0:
                                cute.arch.fence_view_async_tmem_load()
                                with cute.arch.elect_one():
                                    acc_pipeline.consumer_release(acc_consumer_state)
                                acc_consumer_state.advance()

                        acc_vec_lo = tTR_rAcc_up.load()
                        acc_vec_hi = tTR_rAcc_gate.load()

                        tCompute = cute.make_rmem_tensor(
                            acc_vec_hi.shape, self.acc_dtype
                        )
                        LOG2_E = cutlass.Float32(1.4426950408889634)
                        # 64 accumulator columns per subtile register array = two 32-column blocks.
                        for q in cutlass.range_constexpr(self.epi_tile_n // 16):
                            base = (q % 2) * 32
                            for a in cutlass.range_constexpr(8):
                                # (h = 0, h = 1) pair -> output columns 16q + 2a, 16q + 2a + 1
                                if cutlass.const_expr(q < 2):
                                    up0 = acc_vec_lo[base + a]
                                    gate0 = acc_vec_lo[base + 8 + a]
                                    up1 = acc_vec_lo[base + 16 + a]
                                    gate1 = acc_vec_lo[base + 24 + a]
                                else:
                                    up0 = acc_vec_hi[base + a]
                                    gate0 = acc_vec_hi[base + 8 + a]
                                    up1 = acc_vec_hi[base + 16 + a]
                                    gate1 = acc_vec_hi[base + 24 + a]
                                o = 16 * q + 2 * a
                                if cutlass.const_expr(self.vectorized_f32):
                                    g = cute.arch.mul_packed_f32x2(
                                        (gate0, gate1), (gate_scale, gate_scale)
                                    )
                                    u = cute.arch.mul_packed_f32x2(
                                        (up0, up1), (up_scale, up_scale)
                                    )
                                    t = cute.arch.mul_packed_f32x2(
                                        (g[0], g[1]), (-LOG2_E, -LOG2_E)
                                    )
                                    e = cute.arch.add_packed_f32x2(
                                        (
                                            cute.math.exp2(t[0], fastmath=True),
                                            cute.math.exp2(t[1], fastmath=True),
                                        ),
                                        (1.0, 1.0),
                                    )
                                    r0 = cute.arch.rcp_approx(e[0])
                                    r1 = cute.arch.rcp_approx(e[1])
                                    sg = cute.arch.mul_packed_f32x2(
                                        (r0, r1), (g[0], g[1])
                                    )
                                    tCompute[o], tCompute[o + 1] = (
                                        cute.arch.mul_packed_f32x2(
                                            (sg[0], sg[1]), (u[0], u[1])
                                        )
                                    )
                                else:
                                    g0 = gate0 * gate_scale
                                    g1 = gate1 * gate_scale
                                    tCompute[o] = (up0 * up_scale) * silu_f32(
                                        g0, fastmath=True
                                    )
                                    tCompute[o + 1] = (up1 * up_scale) * silu_f32(
                                        g1, fastmath=True
                                    )

                        #
                        # Convert to C type
                        #
                        acc_vec = tiled_copy_r2s.retile(tCompute).load()
                        tRS_rC.store(acc_vec.to(self.c_dtype))

                        #
                        # Store C to shared memory
                        #
                        num_prev_subtiles = num_prev_subtiles + 1
                        c_buffer = num_prev_subtiles % self.num_c_stage

                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rC,
                            tRS_sC[(None, None, None, c_buffer)],
                        )
                        # Fence and barrier to make sure shared memory store is visible to TMA store
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )
                        self.epilog_sync_barrier.arrive_and_wait()
                        #
                        # TMA store C to global memory
                        #
                        if warp_idx == self.epilog_warp_id[0]:
                            cute.copy(
                                tma_atom_c,
                                bSG_sC[(None, c_buffer)],
                                bSG_gC[(None, epi_m_idx, real_out_n_idx)],
                            )
                            # Fence and barrier to make sure shared memory store is visible to TMA store
                            c_pipeline.producer_commit()
                            c_pipeline.producer_acquire()
                        self.epilog_sync_barrier.arrive_and_wait()

                #
                # Async arrive accumulator buffer empty. Overlap mode already
                # released early inside the subtile loop; skip the final one.
                #
                if cutlass.const_expr(not self.use_overlap_accum):
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(5, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
            #
            # Wait for C store complete
            #
            c_pipeline.producer_tail()

        cute.arch.griddepcontrol_launch_dependents()

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory
        (source) and register array (destination).

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param tAcc: The accumulator tensor to be copied and partitioned
        :type tAcc: cute.Tensor
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param use_2cta_instrs: Whether use_2cta_instrs is enabled
        :type use_2cta_instrs: bool

        :return: A tuple containing (tiled_copy_t2r, tTR_tAcc, tTR_rAcc_up, tTR_rAcc_gate) where:
            - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
            - tTR_tAcc: The partitioned accumulator tensor
            - tTR_rAcc_up: The partitioned accumulator tensor for acc up
            - tTR_rAcc_gate: The partitioned accumulator tensor for acc gate
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor, cute.Tensor]
        """
        # Make tiledCopy for tensor memory load (transformed layout)
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )

        # tAcc is already transformed: (M, N, STAGE) layout
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(
            tAcc,
            epi_tile,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # gC_mnl is already transformed: (M, N_half, loopM, loopN, loopL)
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        gC_mnl_epi = cute.flat_divide(gC_mnl, epi_tile)

        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, loopM, loopN, loopL)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)

        # (T2R, T2R_M, T2R_N)
        tTR_rAcc_up = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc_gate = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc_up, tTR_rAcc_gate

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory store, then use it to partition register
        array (source) and shared memory (destination).

        :param tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rC: The partitioned accumulator tensor
        :type tTR_rC: cute.Tensor
        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor
        :type sepi: cute.Tensor

        :return: A tuple containing (tiled_copy_r2s, tRS_rC, tRS_sC) where:
            - tiled_copy_r2s: The tiled copy operation for register to smem copy(r2s)
            - tRS_rC: The partitioned tensor C (register source)
            - tRS_sC: The partitioned tensor C (smem destination)
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        # (R2S, R2S_M, R2S_N)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def epilog_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
    ) -> Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        """Make tiledCopy for global memory store, then use it to:
        - partition register array (source) and global memory (destination) for none TMA store version;
        - partition shared memory (source) and global memory (destination) for TMA store version.

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param atom: The copy_atom_c to be used for TMA store version, or tiled_copy_t2r for none TMA store version
        :type atom: cute.CopyAtom or cute.TiledCopy
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor

        :return: A tuple containing :
            - For TMA store: (tma_atom_c, bSG_sC, bSG_gC) where:
                - tma_atom_c: The TMA copy atom
                - bSG_sC: The partitioned shared memory tensor C
                - bSG_gC: The partitioned global tensor C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]
        """
        # gC_mnl is already transformed: (M, N_half, loopM, loopN, loopL)
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        gC_epi = cute.flat_divide(gC_mnl, epi_tile)
        tma_atom_c = atom
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, loopM, loopN, loopL)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )
        return tma_atom_c, bSG_sC, bSG_gC

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        cta_tile_shape_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        epi_tile: cute.Tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        num_smem_capacity: int,
        occupancy: int,
        with_breuse: bool = False,
    ) -> Tuple[int, int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: The shape (M, N, K) of the MMA tiler.
        :type mma_tiler_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param epi_tile: The epilogue tile shape.
        :type epi_tile: cute.Tile
        :param c_dtype: Data type of operand C (output).
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: Layout of operand C.
        :type c_layout: utils.LayoutEnum
        :param sf_dtype: Data type of scale factor.
        :type sf_dtype: type[cutlass.Numeric]
        :param sf_vec_size: Vector size of scale factor.
        :type sf_vec_size: int
        :param num_smem_capacity: Total available shared memory capacity in bytes.
        :type num_smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (ACC stages, A/B operand stages, C stages)
        :rtype: tuple[int, int, int]
        """
        # Default ACC stages
        num_acc_stage = 1 if (with_breuse and mma_tiler_mnk[1] in {192, 256}) else 2

        # Default C stages
        num_c_stage = 2

        # Default Tile info stages
        num_tile_stage = 2

        # Calculate smem layout and size for one stage of A, B, and C
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,  # a tmp 1 stage is provided
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,  # a tmp 1 stage is provided
        )

        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )

        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,
        )

        # SFA SMEM is plain linear (M, tile_K_sf), no pad.
        # Per stage = cta_tile_M × tile_K_sf bytes (FP8 = 1 byte/element).
        sfa_tile_k_sf = cta_tile_shape_mnk[2] // sf_vec_size
        sf_bytes_per_row = sfa_tile_k_sf * sf_dtype.width // 8
        sfa_bytes_per_stage_one = cta_tile_shape_mnk[0] * sf_bytes_per_row

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + sfa_bytes_per_stage_one
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        )
        # 1024B alignment
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage

        # Calculate A/B stages:
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes and initial C stages bytes
        # Divide remaining by bytes needed per A/B stage
        num_ab_stage = (
            num_smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B stages and reserved bytes
        # Add remaining unused smem to epilogue
        num_c_stage += (
            num_smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)
        return num_acc_stage, num_ab_stage, num_c_stage, num_tile_stage

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
        raster_along_m: bool = False,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        """Use persistent tile scheduler to compute the grid size for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr

        :return: A tuple containing:
            - tile_sched_params: Parameters for the persistent tile scheduler.
            - grid: Grid shape for kernel launch.
        :rtype: Tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]
        """
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl, raster_along_m=raster_along_m
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    @staticmethod
    def _get_tma_atom_kind(
        atom_sm_cnt: cutlass.Int32, mcast: cutlass.Boolean
    ) -> Union[
        cpasync.CopyBulkTensorTileG2SMulticastOp, cpasync.CopyBulkTensorTileG2SOp
    ]:
        """
        Select the appropriate TMA copy atom based on the number of SMs and the multicast flag.

        :param atom_sm_cnt: The number of SMs
        :type atom_sm_cnt: cutlass.Int32
        :param mcast: The multicast flag
        :type mcast: cutlass.Boolean

        :return: The appropriate TMA copy atom kind
        :rtype: cpasync.CopyBulkTensorTileG2SMulticastOp or cpasync.CopyBulkTensorTileG2SOp

        :raise ValueError: If the atom_sm_cnt is invalid
        """
        if atom_sm_cnt == 2 and mcast:
            return cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.TWO)
        elif atom_sm_cnt == 2 and not mcast:
            return cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO)
        elif atom_sm_cnt == 1 and mcast:
            return cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.ONE)
        elif atom_sm_cnt == 1 and not mcast:
            return cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)

        raise ValueError(f"Invalid atom_sm_cnt: {atom_sm_cnt} and {mcast}")

    @staticmethod
    def get_dtype_rcp_limits(dtype: Type[cutlass.Numeric]) -> float:
        """
        Calculates the reciprocal of the maximum absolute value for a given data type.

        :param dtype: Data type
        :type dtype: Type[cutlass.Numeric]

        :return: An float representing the reciprocal of the maximum absolute value
        :rtype: float
        """
        if dtype == cutlass.Float4E2M1FN:
            return 1 / 6.0
        if dtype == cutlass.Float8E4M3FN:
            return 1 / 448.0
        if dtype == cutlass.Float8E5M2:
            return 1 / 128.0
        return 1.0

    @staticmethod
    def is_valid_dtypes_and_scale_factor_vec_size(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """
        Check if the dtypes are valid

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size of the scale factor
        :type sf_vec_size: int
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]

        :return: True if the dtypes are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        if ab_dtype not in {
            cutlass.Float4E2M1FN,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            is_valid = False

        # Check valid sf_vec_size
        if sf_vec_size not in {16, 32}:
            is_valid = False

        # Check valid sf_dtype
        if sf_dtype not in {cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN}:
            is_valid = False

        # Check valid sf_dtype and sf_vec_size combinations
        if sf_dtype == cutlass.Float8E4M3FN and sf_vec_size == 32:
            is_valid = False
        if ab_dtype in {cutlass.Float8E5M2, cutlass.Float8E4M3FN} and sf_vec_size == 16:
            is_valid = False

        # Check valid c_dtype
        if c_dtype not in {
            cutlass.Float32,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
            cutlass.Float4E2M1FN,
        }:
            is_valid = False

        return is_valid

    @staticmethod
    def is_valid_layouts(
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if layouts and dtypes are valid combinations

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major dimension of the A tensor
        :type a_major: str
        :param b_major: The major dimension of the B tensor
        :type b_major: str
        :param c_major: The major dimension of the C tensor
        :type c_major: str

        :return: True if the layouts are valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        if ab_dtype is cutlass.Float4E2M1FN and not (a_major == "k" and b_major == "k"):
            is_valid = False
        if c_dtype is cutlass.Float4E2M1FN and c_major == "m":
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_mma_tiler_and_cluster_shape(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        mma_inst_shape: Tuple[int, int, int],
        mma_tiler: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
    ) -> bool:
        """Check if the mma tiler and cluster shape are valid."""
        # Check valid mma_inst_shape
        if mma_inst_shape[0] not in [128, 256]:
            return False
        # SwiGLU Fusion requires even epi_tile counts
        if mma_inst_shape[1] not in [128, 256]:
            return False

        # Check valid mma_tiler
        if mma_tiler[0] not in [128, 256, 512]:
            return False
        if mma_tiler[1] not in [128, 256]:
            return False

        # Check MMA tiler vs MMA instruction relationship
        # mma_tiler[0] == mma_inst_shape[0] (no B-reuse) or 2 * mma_inst_shape[0] (B-reuse)
        if mma_tiler[0] not in (mma_inst_shape[0], 2 * mma_inst_shape[0]):
            return False
        if mma_tiler[1] != mma_inst_shape[1]:
            return False

        # Check K-dimension constraints based on data type
        if a_dtype in {cutlass.Float8E4M3FN, cutlass.Float8E5M2} and b_dtype in {
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }:
            if mma_tiler[2] != 128 or mma_inst_shape[2] != 64:
                return False
        else:
            if mma_tiler[2] != 256 or mma_inst_shape[2] != 128:
                return False

        # Cluster-M constraint: cluster_M must EQUAL the MMA CTA-group size along M
        # (atom_cta_m): 1 for 1-CTA MMA, 2 for 2-CTA MMA (mma_inst_shape[0] == 256).
        # Splitting the gathered-token (M) dimension across MORE cluster CTAs than the
        # MMA group (cluster_M > atom_cta_m) is not correctly handled by the gather /
        # tile-scheduler row mapping and produces wrong output rows (verified: ~1-2%
        # of rows mismatch on the nvf4 accuracy sweep), so reject it.
        # Cluster-N multicast of A is unaffected and remains supported.
        atom_cta_m = 2 if mma_inst_shape[0] == 256 else 1
        if cluster_shape_mn[0] != atom_cta_m:
            return False

        # Check cluster shape validity
        def _is_power_of_2(x):
            return x > 0 and (x & (x - 1)) == 0

        if (
            cluster_shape_mn[0] * cluster_shape_mn[1] > 16
            or cluster_shape_mn[0] <= 0
            or cluster_shape_mn[1] <= 0
            or cluster_shape_mn[0] > 4
            or cluster_shape_mn[1] > 4
            or not _is_power_of_2(cluster_shape_mn[0])
            or not _is_power_of_2(cluster_shape_mn[1])
        ):
            return False

        return True

    @staticmethod
    def is_valid_tensor_alignment(
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,  # noqa: E741
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if the tensor alignment is valid

        :param m: The number of rows in the A tensor
        :type m: cutlass.Int64
        :param n: The number of columns in the B tensor
        :type n: cutlass.Int64
        :param k: The number of columns in the A tensor
        :type k: cutlass.Int64
        :param l: The number of columns in the C tensor
        :type l: cutlass.Int64
        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str

        :return: True if the problem shape is valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        if (
            not check_contigous_16B_alignment(ab_dtype, a_major == "m", (m, k, l))
            or not check_contigous_16B_alignment(ab_dtype, b_major == "n", (n, k, l))
            or not check_contigous_16B_alignment(c_dtype, c_major == "m", (m, n, l))
        ):
            is_valid = False
        return is_valid

    @classmethod
    def can_implement(
        cls,
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
        mma_inst_shape: Tuple[int, int, int],
        mma_tiler: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,  # noqa: E741
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if the gemm can be implemented

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size of the scale factor
        :type sf_vec_size: int
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]
        :param m: The number of rows in the A tensor
        :type m: cutlass.Int64
        :param n: The number of columns in the B tensor
        :type n: cutlass.Int64
        :param k: The number of columns in the A tensor
        :type k: cutlass.Int64
        :param l: The number of columns in the C tensor
        :type l: cutlass.Int64
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str

        :return: True if the gemm can be implemented, False otherwise
        :rtype: bool
        """
        # Check data types
        if not cls.is_valid_dtypes_and_scale_factor_vec_size(
            a_dtype, sf_dtype, sf_vec_size, c_dtype
        ):
            return False

        # Check layouts
        if not cls.is_valid_layouts(a_dtype, c_dtype, a_major, b_major, c_major):
            return False

        # Check MMA tiler and cluster shape
        if not cls.is_valid_mma_tiler_and_cluster_shape(
            a_dtype, b_dtype, mma_inst_shape, mma_tiler, cluster_shape_mn
        ):
            return False

        # Check tensor alignment
        if not cls.is_valid_tensor_alignment(
            m, n, k, l, a_dtype, c_dtype, a_major, b_major, c_major
        ):
            return False

        # Check A/B layout
        if not (a_major == "k" and b_major == "k"):
            return False
        return True

    @cute.jit
    def wrapper(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        alpha_gate_ptr: cute.Pointer,
        alpha_up_ptr: cute.Pointer,
        token_scale_ptr: cute.Pointer,
        tile_idx_to_group_idx_ptr: cute.Pointer,
        tile_idx_to_mn_limit_ptr: cute.Pointer,
        token_id_mapping_ptr: cute.Pointer,
        num_non_exiting_tiles_ptr: cute.Pointer,
        orig_m: cutlass.Int64,
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,  # noqa: E741
        tile_size: cutlass.Constexpr,
        scaling_vector_size: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        scale_k = k // scaling_vector_size
        interm_size = n // 2
        num_tiles = m // tile_size
        a = cute.make_tensor(
            a_ptr, layout=cute.make_ordered_layout((orig_m, k, 1), order=(1, 0, 2))
        )
        b = cute.make_tensor(
            b_ptr, layout=cute.make_ordered_layout((n, k, l), order=(1, 0, 2))
        )
        a_sf = cute.make_tensor(
            a_sf_ptr,
            layout=cute.make_ordered_layout((orig_m, scale_k, 1), order=(1, 0, 2)),
        )
        b_sf = cute.make_tensor(
            b_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, n // 128, 4, scale_k // 4, l), order=(2, 1, 4, 0, 3, 5)
            ),
        )
        # c: runtime Int64 row stride (a runtime stride also avoids an MLIR alignment issue seen with
        # make_layout(..., stride=ordered_layout.stride)).
        c = cute.make_tensor(
            c_ptr,
            layout=cute.make_layout(
                (m, interm_size, 1),
                stride=(interm_size, 1, m * interm_size),
            ),
        )
        alpha_gate = cute.make_tensor(alpha_gate_ptr, layout=cute.make_layout((l,)))
        alpha_up = cute.make_tensor(alpha_up_ptr, layout=cute.make_layout((l,)))
        token_scale = cute.make_tensor(
            token_scale_ptr, layout=cute.make_layout((orig_m,))
        )

        tile_idx_to_group_idx = cute.make_tensor(
            tile_idx_to_group_idx_ptr, layout=cute.make_layout((num_tiles,))
        )
        tile_idx_to_mn_limit = cute.make_tensor(
            tile_idx_to_mn_limit_ptr, layout=cute.make_layout((num_tiles,))
        )
        token_id_mapping = cute.make_tensor(
            token_id_mapping_ptr, layout=cute.make_layout((m,))
        )
        num_non_exiting_tiles = cute.make_tensor(
            num_non_exiting_tiles_ptr, layout=cute.make_layout((1,))
        )

        return self(
            a,
            b,
            c,
            a_sf,
            b_sf,
            tile_idx_to_group_idx,
            tile_idx_to_mn_limit,
            token_id_mapping,
            num_non_exiting_tiles,
            alpha_gate,
            alpha_up,
            token_scale,
            max_active_clusters=max_active_clusters,
            stream=stream,
        )
