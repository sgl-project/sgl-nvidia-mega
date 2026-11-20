# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 by FlashInfer team.
# Copyright 2026 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Modified by NVIDIA: copied from SGLang's python/sglang/kernels/ops/gemm/cutedsl_bf16_gemm.py; adds an optional epilogue
# mode (constructor argument ``push_mode``, default 0 = unchanged) in which each output tile is also restaged through
# shared memory and stored, as 16-byte vectors with the push all-reduce's empty-marker remap, into the peer slots of a
# push workspace (one-shot: every peer; reduce-scatter: the row owner; mode 3: every slot including this rank's own,
# plus per-CTA ready flags set by the otherwise idle warp); the B-load warp reads the workspace phase after its
# dependency wait; this copy's torch custom-op registrations and public wrappers are removed (SGLang's own module keeps
# them). The main loop is unchanged.
# Modified by NVIDIA: modes 5 and 6 also copy the consumer's residual rows and the workspace phase before each CTA's
# fenced dependents trigger (mode 5: warp 3, one vector in flight per lane; mode 6: warp 3 and the four epilogue warps,
# eight vectors in flight per thread).
# Modified by NVIDIA: mode 7 (publish) stores each 16-byte vector once through the multicast address of a gather slot
# instead (multimem.st), with the all-gather's per-half empty-marker remap, and no plain output.
"""CuTe DSL TGV BF16 GEMM (low-latency SM10x GEMM).

Computes ``out[M, N] = x[M, K] @ weight[N, K].T (+ bias[N])`` for bf16 inputs,
fp32 accumulation, bf16 output. The kernel writes M-contiguous output, so the
runner swaps A and B (``gemm_fn(b.t(), a.t(), ...)``) to land row-major (M, N).

Toggled by ``use_2cta`` in the constructor:
  use_2cta=False -> 1x1 cluster, 1-CTA tcgen05.mma, cta_n in [8, 256] step 8
  use_2cta=True  -> 2x1 cluster, 2-CTA tcgen05.mma, cta_n in [16, 256] step 16

Warp specialization (8 warps, 256 threads/CTA; warp 3 idle):
  Warp 0    DMA_A   TMA-loads A tiles
  Warp 1    DMA_B   TMA-loads B tiles; PDL griddepcontrol.wait
  Warp 2    MMA     tcgen05.mma into TMEM; owns alloc/dealloc
  Warps 4-7 EPILOG  TMEM -> RMEM -> bf16 cast -> st.global
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack, make_fake_stream
from cutlass.cutlass_dsl import T
from sglang.srt.utils import get_device_core_count, is_sm100_supported

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/utils/*.py": ("get_device_core_count", "is_sm100_supported"),
}


# Tuple format: (cta_m, cta_n, num_ab_stage, use_2cta); cta_k is fixed at
# 128. Sorted within each use_2cta block by (cta_m, cta_n, num_ab_stage).
# Tactic 1 (1-CTA 64×8 stages=8) is the default.
_TGV_CUTE_EXT_CTA_K: int = 128
_TGV_CUTE_EXT_DEFAULT_TACTIC: int = 1
_TGV_CUTE_EXT_TACTIC_CONFIGS: List[Tuple[int, int, int, bool]] = [
    # 1-CTA configs
    (64, 8, 6, False),  # 0
    (64, 8, 8, False),  # 1 (default)
    (64, 8, 10, False),  # 2
    (64, 8, 12, False),  # 3
    (64, 16, 6, False),  # 4
    (64, 16, 8, False),  # 5
    (64, 16, 11, False),  # 6
    (64, 32, 6, False),  # 7
    (64, 32, 9, False),  # 8
    (64, 64, 7, False),  # 9
    (64, 128, 4, False),  # 10
    (128, 8, 6, False),  # 11
    (128, 16, 6, False),  # 12
    (128, 32, 5, False),  # 13
    (128, 64, 4, False),  # 14
    (128, 128, 3, False),  # 15
    # 2-CTA configs
    (64, 16, 6, True),  # 16
    (64, 16, 8, True),  # 17
    (64, 16, 12, True),  # 18
    (64, 32, 6, True),  # 19
    (64, 32, 8, True),  # 20
    (64, 32, 11, True),  # 21
    (64, 64, 6, True),  # 22
    (64, 64, 9, True),  # 23
    (64, 128, 7, True),  # 24
    (128, 16, 6, True),  # 25
    (128, 32, 6, True),  # 26
    (128, 64, 5, True),  # 27
    (128, 128, 4, True),  # 28
]


# ---- push epilogue helper: one inline-PTX sequence per 16-byte row vector ----
# push_mode 6 copy batch: up to eight 16-byte vectors ($3 = how many remain, predicates k < $3) at stride $2 from
# $0 to $1; all loads are issued before the first store (plain weak accesses, byte-exact).
_COPY8_ASM = "{\n.reg .pred p<8>;\n.reg .b32 c<32>;\n.reg .b64 s, d;\nmov.b64 s, $0;\nmov.b64 d, $1;\nsetp.gt.s32 p0, $3, 0;\nsetp.gt.s32 p1, $3, 1;\nsetp.gt.s32 p2, $3, 2;\nsetp.gt.s32 p3, $3, 3;\nsetp.gt.s32 p4, $3, 4;\nsetp.gt.s32 p5, $3, 5;\nsetp.gt.s32 p6, $3, 6;\nsetp.gt.s32 p7, $3, 7;\n@p0 ld.global.v4.b32 {c0, c1, c2, c3}, [s];\nadd.s64 s, s, $2;\n@p1 ld.global.v4.b32 {c4, c5, c6, c7}, [s];\nadd.s64 s, s, $2;\n@p2 ld.global.v4.b32 {c8, c9, c10, c11}, [s];\nadd.s64 s, s, $2;\n@p3 ld.global.v4.b32 {c12, c13, c14, c15}, [s];\nadd.s64 s, s, $2;\n@p4 ld.global.v4.b32 {c16, c17, c18, c19}, [s];\nadd.s64 s, s, $2;\n@p5 ld.global.v4.b32 {c20, c21, c22, c23}, [s];\nadd.s64 s, s, $2;\n@p6 ld.global.v4.b32 {c24, c25, c26, c27}, [s];\nadd.s64 s, s, $2;\n@p7 ld.global.v4.b32 {c28, c29, c30, c31}, [s];\n@p0 st.global.v4.b32 [d], {c0, c1, c2, c3};\nadd.s64 d, d, $2;\n@p1 st.global.v4.b32 [d], {c4, c5, c6, c7};\nadd.s64 d, d, $2;\n@p2 st.global.v4.b32 [d], {c8, c9, c10, c11};\nadd.s64 d, d, $2;\n@p3 st.global.v4.b32 [d], {c12, c13, c14, c15};\nadd.s64 d, d, $2;\n@p4 st.global.v4.b32 [d], {c16, c17, c18, c19};\nadd.s64 d, d, $2;\n@p5 st.global.v4.b32 [d], {c20, c21, c22, c23};\nadd.s64 d, d, $2;\n@p6 st.global.v4.b32 [d], {c24, c25, c26, c27};\nadd.s64 d, d, $2;\n@p7 st.global.v4.b32 [d], {c28, c29, c30, c31};\n}"


def publish_vector_ptx(smem_operand: str, mc_operand: str) -> str:
    """PTX of the push_mode 7 store of one restaged 16-byte vector (also built into the bit-preservation unit):
    ld.shared.v4 the vector; remap every 16-bit half that is +0.0 to -0.0 (+0.0 in either half marks a gather slot word
    that has not arrived, as in the all-gather's own push); one multimem.st.relaxed.sys.global.v4.f32 through the
    multicast address, which stores the vector into every rank's copy of the slot (the all-gather's own push stores
    .v4.f32 lanes too)."""
    lines = [
        "{",
        ".reg .b32 a<4>, b<4>, t;",
        ".reg .pred p;",
        f"ld.shared.v4.b32 {{a0, a1, a2, a3}}, [{smem_operand}];",
    ]
    for k in range(4):
        lines.append(f"and.b32 t, a{k}, 0x0000FFFF;")
        lines.append("setp.eq.b32 p, t, 0;")
        lines.append(f"selp.b32 b{k}, 0x00008000, 0, p;")
        lines.append(f"and.b32 t, a{k}, 0xFFFF0000;")
        lines.append("setp.eq.b32 p, t, 0;")
        lines.append(f"@p or.b32 b{k}, b{k}, 0x80000000;")
        lines.append(f"or.b32 b{k}, a{k}, b{k};")
    lines.append(
        f"multimem.st.relaxed.sys.global.v4.f32 [{mc_operand}], {{b0, b1, b2, b3}};"
    )
    lines.append("}")
    return "\n".join(lines)


def _publish_vector(s_addr, mc_addr):
    """push_mode 7: one restaged 16-byte vector (publish_vector_ptx)."""
    llvm.inline_asm(
        None,
        [cutlass.Int32(s_addr).ir_value(), cutlass.Int64(mc_addr).ir_value()],
        publish_vector_ptx("$0", "$1"),
        "r,l",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def _push_vector(s_addr, local_addr, peer_addrs, peer_flags):
    """ld.shared.v4 the restaged vector; st.global.v4 to the local output; remap the push all-reduce's empty marker
    (an all-zero 32-bit word is sent as 0x00008000, i.e. the pair (-0.0, +0.0), as the consumer's own push path does);
    then one st.relaxed.sys.global.v4.b32 per destination whose flag is non-zero."""
    n = len(peer_addrs)
    lines = [
        "{",
        ".reg .b32 a<4>, b<4>;",
        ".reg .pred p<4>, q;",
        "ld.shared.v4.b32 {a0, a1, a2, a3}, [$0];",
        "st.global.v4.b32 [$1], {a0, a1, a2, a3};",
    ]
    for k in range(4):
        lines.append(f"setp.eq.b32 p{k}, a{k}, 0;")
        lines.append(f"selp.b32 b{k}, 32768, a{k}, p{k};")
    for d in range(n):
        lines.append(f"setp.ne.b32 q, ${3 + 2 * d}, 0;")
        lines.append(
            f"@q st.relaxed.sys.global.v4.b32 [${2 + 2 * d}], {{b0, b1, b2, b3}};"
        )
    lines.append("}")
    operands = [cutlass.Int32(s_addr).ir_value(), cutlass.Int64(local_addr).ir_value()]
    cons = ["r", "l"]
    for a, f in zip(peer_addrs, peer_flags):
        operands += [cutlass.Int64(a).ir_value(), cutlass.Int32(f).ir_value()]
        cons += ["l", "r"]
    llvm.inline_asm(
        None,
        operands,
        "\n".join(lines),
        ",".join(cons),
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class WorkTileInfo(NamedTuple):
    """Which output tile this CTA processes. Mirrors the original WorkTileInfo."""

    M_idx: cutlass.Int32
    N_idx: cutlass.Int32
    L_idx: cutlass.Int32
    K_idx_start: cutlass.Int32
    K_idx_end: cutlass.Int32


class TgvGemmCuteExtKernel:
    """
    Low-latency Blackwell GEMM kernel rewritten with cute_ext primitives,
    keeping the raw-mbarrier 7-warp specialization from the C++ kernel
    in ``include/flashinfer/gemm/tgv_gemm.cuh``.
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        cta_m: int = 64,
        cta_n: int = 8,
        cta_k: int = _TGV_CUTE_EXT_CTA_K,
        num_ab_stage: int = 8,
        use_2cta: bool = False,
        use_pdl: bool = False,
        pdl_launch: Optional[bool] = None,
        pdl_count: int = -1,
        has_bias: bool = False,
        out_dtype: Type[cutlass.Numeric] = cutlass.BFloat16,
        push_mode: int = 0,
        push_world: int = 1,
        test_drop_restage_barrier: bool = False,
        test_trigger_before_copy: bool = False,
    ):
        # push_mode: 0 = plain epilogue (as SGLang); 1 = one-shot push (every peer slot); 2 = reduce-scatter push (the row
        # owner's slot, owner chunk = rows / push_world). Push modes need bf16 output, no bias and one batch.
        # push_mode 3 = one-shot to every slot INCLUDING this rank's own, plus per-CTA ready flags for a poll-first
        # consumer (that consumer's entry ordering is NOT guaranteed by the PTX memory model; no served caller, and any
        # served caller must gate it behind a default-off _UNSAFE knob).
        # push_mode 5 = one-shot to every slot INCLUDING this rank's own, plus a producer-owned copy of the consumer's
        # cross-grid inputs (the residual rows and the workspace phase), written by the otherwise idle warp 3 after its
        # dependency wait and ordered before each CTA's dependents trigger by a CTA barrier + release fence; a
        # poll-first consumer then reads only this grid's writes, static weights and data-as-flag words before its
        # own dependency wait (PTX griddepcontrol: a release fence before launch_dependents synchronizes with the start
        # of the dependent grid).
        # push_mode 6 = push_mode 5 with the residual copy spread over warp 3 and the four epilogue warps (idle until the
        # accumulator is ready), each thread keeping several 16-byte vectors in flight; warp 3 and the epilogue warps
        # arrive on named barrier 2, the DMA_A warp waits on it, then fences (release) and triggers, as in mode 5.
        # push_mode 7 = publish: no plain output; each 16-byte vector (8 features of one row) goes once, after the
        # per-half empty-marker remap, through the multicast address meta[8] + phase * meta[1] + (row * meta[6] +
        # feature) * 2, i.e. into this rank's block of a gather slot laid out [source rank][rows][meta[6]] in every
        # rank's copy of the workspace half of the current phase (the all-gather's slot layout and marker rule).
        if push_mode not in (0, 1, 2, 3, 5, 6, 7):
            raise ValueError(f"push_mode={push_mode} not in (0, 1, 2, 3, 5, 6, 7)")
        if push_mode and (
            has_bias
            or out_dtype is not cutlass.BFloat16
            or push_world < 2
            or push_world > 8
        ):
            raise ValueError(
                "push epilogue needs bf16 output, no bias and 2 <= push_world <= 8"
            )
        if push_mode in (5, 6) and (
            not use_pdl
            or not (pdl_launch if pdl_launch is not None else use_pdl)
            or pdl_count >= 0
        ):
            # The consumer's pre-wait reads rely on each CTA's explicit fence + trigger after its copy: an implicit
            # trigger at block exit (no PDL) gives no visibility, and an in-loop trigger would fire before the copy.
            raise ValueError(
                "push_mode 5 / 6 needs a PDL launch with the trigger after the copy (pdl_count < 0)"
            )
        self.push_mode = push_mode
        self.push_world = push_world
        # Test builds only (sanitizer non-vacuity): drop the named barrier between the restage writes and the vector
        # reads, which racecheck must report. Never set by any served path.
        self.test_drop_restage_barrier = test_drop_restage_barrier
        # Test builds only (non-vacuity of the copy-input ordering): trigger the dependents BEFORE the copy and delay
        # the copy, so a consumer reads a stale copy; the bitwise checks must then fail. Never set by any served path.
        self.test_trigger_before_copy = test_trigger_before_copy
        self.acc_dtype = acc_dtype
        self.out_dtype = out_dtype
        self.cta_m = cta_m
        self.cta_n = cta_n
        self.cta_k = cta_k
        self.num_ab_stage = num_ab_stage
        self.use_2cta = use_2cta
        self.use_pdl = use_pdl
        self.pdl_launch = pdl_launch if pdl_launch is not None else use_pdl
        self.pdl_count = pdl_count
        # has_bias: when True, kernel reads a (Gemm_M, Gemm_N, Gemm_L):(1,0,0)
        # bias tensor (M-broadcast over N,L), converts it to fp32 in RMEM, and
        # adds it to the accumulator before the bf16 cast. When False, all
        # bias-related code is elided via cutlass.const_expr.
        self.has_bias = has_bias

        # 1-CTA: cta_n ∈ [8, 256] step 8 (bf16 tcgen05.mma atom limit).
        # 2-CTA: cta_n ∈ [16, 256] step 16 (bf16 K-major cluster mma).
        min_n, step_n = (16, 16) if use_2cta else (8, 8)
        if cta_n < min_n or cta_n > 256 or cta_n % step_n != 0:
            raise ValueError(
                f"cta_n={cta_n} invalid for use_2cta={use_2cta}: "
                f"bf16 K-major mma requires N ∈ [{min_n}, 256] step {step_n}"
            )

        # Fixed configuration matching the C++ / DSL kernels.
        self.threads_per_cta = 256  # 8 warps (warp 3 unused)
        if use_2cta:
            # 2-CTA cluster along M; joint MMA tile = (cta_m*2, cta_n).
            self.cluster_shape = (2, 1, 1)
            self.mma_tiler_mn = (cta_m * 2, cta_n)
            self.cta_group = tcgen05.CtaGroup.TWO
            self.tma_op = cute_ext.OperationTypeEnum.SM100_TMA_LOAD_2SM
        else:
            # 1 SM mode, 1x1 cluster, no multicast.
            self.cluster_shape = (1, 1, 1)
            self.mma_tiler_mn = (cta_m, cta_n)
            self.cta_group = tcgen05.CtaGroup.ONE
            self.tma_op = cute_ext.OperationTypeEnum.SM90_TMA_LOAD

    def __repr__(self) -> str:
        return (
            f"TgvGemmCuteExtKernel_cta{self.cta_m}x{self.cta_n}x{self.cta_k}"
            f"_2cta{int(self.use_2cta)}_pdl{int(self.use_pdl)}"
            f"_bias{int(self.has_bias)}"
            f"_out{self.out_dtype.__name__.lower()}"
            f"_push{self.push_mode}w{self.push_world}"
        )

    @cute.experimental.jit
    def __call__(
        self,
        a: cute.Tensor,  # (Gemm_M, Gemm_K, Gemm_L), K-major
        b: cute.Tensor,  # (Gemm_N, Gemm_K, Gemm_L), K-major
        c: cute.Tensor,  # (Gemm_M, Gemm_N, Gemm_L), M-major
        bias: cute.Tensor,  # (Gemm_M, Gemm_N, Gemm_L):(1,0,0) — unused when has_bias=False
        meta: cute.Tensor,  # (PUSH_META_LEN,) int64 push arguments (push_mode != 0); any tensor otherwise (never read)
        stream: cuda.CUstream,
    ):
        # Each CTA processes one (CTA_M, CTA_N) output tile, no persistence.
        # Round the (M, N) tile counts up to a full cluster so 2-CTA launches
        # don't leave an odd M-CTA without its peer (e.g. M=2880, cta_m=64 →
        # 45 M-tiles, invalid for cluster=(2,1,1)). cute_ext OOB-protects the
        # extra CTA's GMEM stores.
        grid = cute.round_up(
            (
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n),
                c.layout.shape[2],
            ),
            self.cluster_shape,
        )
        self.kernel(a, b, c, bias, meta).launch(
            grid=grid,
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(utils.get_smem_capacity_in_bytes("sm_100")),
            stream=stream,
            use_pdl=self.pdl_launch,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        mA: cute.Tensor,  # (Gemm_M, Gemm_K, Gemm_L), K-major
        mB: cute.Tensor,  # (Gemm_N, Gemm_K, Gemm_L), K-major
        mC: cute.Tensor,  # (Gemm_M, Gemm_N, Gemm_L), M-major
        mBias: cute.Tensor,  # (Gemm_M, Gemm_N, Gemm_L):(1,0,0) — unused when has_bias=False
        mMeta: cute.Tensor,  # (PUSH_META_LEN,) int64 push arguments; never read when push_mode == 0
    ):
        """
        Device-side dispatcher: build MMA descriptor, allocate SMEM/TMEM/
        mbarriers, init barriers, dispatch into dma_a/dma_b/mma/epilog warps.
        """
        DMA_Stage = self.num_ab_stage

        # ---- Tiled MMA ----
        # make_trivial_tiled_mma picks the largest tcgen05.mma atom that fits
        # mma_tiler_mn. 1-CTA bf16 (64,8): Mma_M=(16,4)=64, Mma_N=8, Mma_K=16.
        # 2-CTA bf16 K-major (128,N): Mma_M=128 split across cluster, Mma_K=16.
        a_major = utils.LayoutEnum.from_tensor(mA).mma_major_mode()
        b_major = utils.LayoutEnum.from_tensor(mB).mma_major_mode()
        ab_dtype = mA.element_type  # bf16
        c_dtype = mC.element_type  # bf16
        d_layout = utils.LayoutEnum.from_tensor(mC)

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype,
            ab_dtype,
            a_major,
            b_major,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_mn,
        )
        num_mma_ctas = cute.size(tiled_mma.thr_id.shape)  # 1 (1-CTA) or 2 (2-CTA)

        # NumMma_K = CTA_K/Mma_K — inner-K trip count for the MMA warp.
        # bf16 default: Mma_K=16, CTA_K=128 → mma_inst_tile_k = 8.
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = self.cta_k // mma_inst_shape_k

        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        a_tiler_mk = (self.cta_m, self.cta_k)  # CTA tile of A
        # 2-CTA splits B across cluster M (each CTA loads Mma_N/2 cols).
        b_tiler_nk = (self.cta_n // num_mma_ctas, self.cta_k)
        c_tiler_mn = (self.cta_m, self.cta_n)  # CTA tile of C

        # ---- WorkTileInfo (static 1-tile-per-CTA scheduler) ----
        # blockIdx maps directly to (M_tile, N_tile, batch). K_idx_start/end
        # carry over from C++ for future split-K; here every CTA reduces the
        # full K range (= ceil(K/CTA_K) tiles, e.g. 12 for K=1536).
        bidx, bidy, bidz = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        # In 2-CTA mode: identify which CTA within the (2,1,1) cluster owns
        # the leader role (issues MMAs / multicasts commits). In 1-CTA mode
        # every CTA is its own leader.
        leader_rank = cutlass.Int32(0)
        if cutlass.const_expr(self.use_2cta):
            cta_rank_in_cluster = cute.arch.make_warp_uniform(
                cute.arch.block_idx_in_cluster()
            )
            is_leader = cta_rank_in_cluster == 0
        else:
            cta_rank_in_cluster = leader_rank
            is_leader = cutlass.Boolean(True)

        # ---- SMEM A/B (DMA_Stage-staged, swizzled ComposedLayout) ----
        # alignment=1024 covers TMA's natural alignment requirements. 2-CTA
        # halves the M extent of sA and the N extent of sB internally (the SS
        # atom splits A across cluster M and B across cluster M).
        a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mnk_tiler,
            ab_dtype,
            DMA_Stage,
        )  # ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage) — Sw<3,4,3>
        b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mnk_tiler,
            ab_dtype,
            DMA_Stage,
        )  # ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage) — Sw<3,4,3>

        # sA per-CTA M extent = cta_m for both modes; sB per-CTA N extent =
        # cta_n (1-CTA) or cta_n/2 (2-CTA, SS atom splits B across cluster M).
        sA = cute_ext.allocate(  # ((Mma_M_per_cta, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
            ab_dtype,
            cute.AddressSpace.smem,
            a_smem_layout_staged,
            alignment=1024,
        )
        sB = cute_ext.allocate(  # ((Mma_N_per_cta, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
            ab_dtype,
            cute.AddressSpace.smem,
            b_smem_layout_staged,
            alignment=1024,
        )

        # ---- TMEM accumulator layout (manual alloc; see MMA warp) ----
        # cute_ext.make_tmem_layout_acc accounts for cta_group when handed
        # the cluster MMA tile, so the resulting layout is already per-CTA.
        # acc_layout per-CTA: ((Mma_M_per_cta, Mma_N), NumMma_M, NumMma_N, AccStage)
        # = (((16,4), cta_n), 1, 1, 1) for both 1-CTA and 2-CTA modes.
        # We still allocate TMEM manually (via cute.arch.alloc_tmem below) for better performance.
        acc_layout = cute_ext.make_tmem_layout_acc(
            tiled_mma,
            self.mma_tiler_mn,
            acc_stage=1,
        )

        # ---- Raw mbarriers (Int64 SMEM arrays) + tmem_base_ptr Int32 slot ----
        # Two flavors of barriers exist on Hopper/Blackwell:
        #   1. Named barriers (bar.arv/bar.sync) — Ampere-era, 16 hardware
        #      barriers per SM, only handle intra-CTA thread sync.
        #   2. Mbarriers (mbarrier.* PTX) — 64-bit in SMEM per barrier,
        #      software-programmable. Used here for (a) thread sync within a
        #      CTA, (b) TMA transaction-count tracking (TMA→SM signaling),
        #      and (c) 1-arrival "wake the consumer warp" patterns.
        # Allocated as 1D tensors; .iterator gives a Pointer[Int64] supporting
        # `bar + stage` arithmetic. Arrival counts are in the module docstring.
        bar_full_arr = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(DMA_Stage),
            alignment=8,
        )
        bar_empty_arr = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(DMA_Stage),
            alignment=8,
        )
        bar_tma_epilog_arr = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
        )
        bar_mma_epilog_arr = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
        )
        bar_tmem_alloc_arr = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
        )
        # alloc_tmem writes the TMEM base address to this slot; both MMA and
        # EPILOG read it back via retrieve_tmem_ptr.
        tmem_base_arr = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        )
        # Push epilogue: the workspace phase, read by DMA_B after its dependency wait and handed to the epilogue
        # through bar_tma_epilog (mbarrier arrive / wait order the shared-memory word).
        push_phase_arr = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        )

        bar_full = bar_full_arr.iterator  # Pointer[Int64], DMA_Stage
        bar_empty = bar_empty_arr.iterator  # Pointer[Int64], DMA_Stage
        bar_tma_epilog = bar_tma_epilog_arr.iterator  # Pointer[Int64]
        bar_mma_epilog = bar_mma_epilog_arr.iterator  # Pointer[Int64]
        bar_tmem_alloc = bar_tmem_alloc_arr.iterator  # Pointer[Int64]
        tmem_base_ptr = tmem_base_arr.iterator  # Pointer[Int32]

        # ---- Barrier init (1 thread, PTX mbarrier.init is single-thread) ----
        # bar_full arrival = 2 * num_mma_ctas:
        #   1-CTA: 2 (own DMA_A + DMA_B)
        #   2-CTA: 4 (own + peer DMA_A/DMA_B, routed via SM100_TMA_LOAD_2SM
        #          onto the leader's bar). In 2-CTA the peer's copy is unused.
        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(DMA_Stage):
                    cute.arch.mbarrier_init(bar_full + i, 2 * num_mma_ctas)
                for i in range(DMA_Stage):
                    cute.arch.mbarrier_init(bar_empty + i, 1)  # MMA tcgen05.commit
                cute.arch.mbarrier_init(bar_tma_epilog, 32)  # whole DMA_B warp
                cute.arch.mbarrier_init(bar_mma_epilog, 1)  # MMA tcgen05.commit
                cute.arch.mbarrier_init(
                    bar_tmem_alloc, 32 + 128
                )  # MMA + 4 EPILOG warps

        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.use_2cta):
            # Cluster-wide sync: ensure peer's barriers are visible before any
            # cross-CTA arrives. Splitting arrive/wait lets the per-CTA tile
            # setup below overlap with the cluster barrier. (Implicit
            # intra-CTA sync — cluster.arrive requires all threads to reach.)
            cute.arch.cluster_arrive_relaxed()
        else:
            cute.arch.barrier()

        work_tile_info = WorkTileInfo(
            M_idx=bidx,
            N_idx=bidy,
            L_idx=bidz,
            K_idx_start=cutlass.Int32(0),
            K_idx_end=cute.ceil_div(cute.size(mA, mode=[1]), self.cta_k),
        )
        k_tile_count = work_tile_info.K_idx_end - work_tile_info.K_idx_start

        # ---- CTA-to-Value maps for TMA (constexpr) ----
        # Encodes "which cluster CTA is responsible for which slice", i.e.
        # the CTA-coord → logical-multicast-id layout. For our 1×1 cluster
        # (no multicast) it's an identity layout; cute_ext.tma_load still
        # needs this explicit layout to construct the TMA descriptor.
        a_cta_v_map = cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "A")
        b_cta_v_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")

        # ---- Per-CTA tile views ----
        # local_tile(t, tiler, coord) = zipped_divide(t, tiler)[coord]; modes
        # passed `None` stay free. Per-CTA M/N extents:
        #   gA M extent = cta_m   (both modes; SS atom splits cluster M)
        #   gB N extent = cta_n   (1-CTA) or cta_n//2 (2-CTA)
        #   gD M×N      = cta_m × cta_n (cluster N is identical for both CTAs)
        gA_tile = cute.local_tile(  # (cta_m, cta_k, Tiles_K)
            mA,
            a_tiler_mk,
            (work_tile_info.M_idx, None, work_tile_info.L_idx),
        )
        # gB n-index: 1-CTA just bidy; 2-CTA each cluster N_idx covers
        # num_mma_ctas tiles of b_tiler, each CTA loads one half.
        if cutlass.const_expr(self.use_2cta):
            gB_n_idx = work_tile_info.N_idx * num_mma_ctas + cta_rank_in_cluster
        else:
            gB_n_idx = work_tile_info.N_idx
        gB_tile = cute.local_tile(  # (cta_n//num_mma_ctas, cta_k, Tiles_K)
            mB,
            b_tiler_nk,
            (gB_n_idx, None, work_tile_info.L_idx),
        )
        gD_tile = cute.local_tile(  # (cta_m, cta_n)
            mC,
            c_tiler_mn,
            (work_tile_info.M_idx, work_tile_info.N_idx, work_tile_info.L_idx),
        )
        # gBias_tile: same (cta_m, cta_n) shape as gD_tile but stride (1, 0)
        # — the same M element is repeated across N. local_tile preserves the
        # (1, 0, 0) stride from mBias, so this works automatically.
        gBias_tile = cute.local_tile(  # (cta_m, cta_n) stride (1, 0)
            mBias,
            c_tiler_mn,
            (work_tile_info.M_idx, work_tile_info.N_idx, work_tile_info.L_idx),
        )

        if cutlass.const_expr(self.use_2cta):
            # Cluster sync deferred to here — tile-view setup above runs in
            # parallel with the cluster barrier latency.
            cute.arch.cluster_wait()

        # ---- Warp dispatch (warp 3 idle, kept for 256-thread parity) ----
        if warp_idx == 0:
            self.dma_a_warp(
                bar_full,
                bar_empty,
                leader_rank,
                gA_tile,
                sA,
                a_cta_v_map,
                k_tile_count,
            )
        elif warp_idx == 1:
            self.dma_b_warp(
                bar_full,
                bar_empty,
                bar_tma_epilog,
                leader_rank,
                gB_tile,
                sB,
                b_cta_v_map,
                k_tile_count,
                push_phase_arr,
                mMeta,
            )
        elif warp_idx == 2:
            self.mma_warp(
                is_leader,
                bar_full,
                bar_empty,
                bar_mma_epilog,
                bar_tmem_alloc,
                tiled_mma,
                sA,
                sB,
                tmem_base_ptr,
                acc_layout,
                mma_inst_tile_k,
                k_tile_count,
            )
        elif warp_idx == 3:
            # Otherwise idle. push_mode 3 only: CTA (0, 0) sets the consumer's per-CTA ready flags after its own
            # dependency wait, each lane behind its own fence (release pattern).
            if cutlass.const_expr(self.push_mode == 3):
                self.ready_warp(mMeta, bidx, bidy)
            if cutlass.const_expr(self.push_mode == 5):
                self.copy_warp(mMeta, bidx, bidy)
            if cutlass.const_expr(self.push_mode == 6):
                self.copy_batched(mMeta, bidx, bidy, cute.arch.lane_idx(), True)
        elif warp_idx >= 4:
            # Epilog tid is 128..255 in the CTA; offset to 0..127 for partition.
            epi_tid = tidx - 128
            self.epilog_warp(
                bar_tma_epilog,
                bar_mma_epilog,
                bar_tmem_alloc,
                tmem_base_ptr,
                acc_layout,
                gD_tile,
                gBias_tile,
                epi_tid,
                c_dtype,
                d_layout,
                sA,
                push_phase_arr,
                mMeta,
                work_tile_info.M_idx,
                work_tile_info.N_idx,
            )

        if cutlass.const_expr(self.use_2cta):
            # Cluster-wide exit barrier (sgl-project/sglang#32907): cluster
            # sync previously existed only at kernel entry, so one CTA could
            # retire while its peer still had in-flight peer-redirected
            # mbarrier arrives / multicast tcgen05.commit / TMEM dealloc
            # targeting it -> CUDBG_EXCEPTION_CLUSTER_BLOCK_NOT_PRESENT
            # (Xid 13 "CTA Not Present"). All 8 warps (256 threads) of both
            # cluster CTAs arrive here after their dispatch branch returns,
            # so no CTA can exit until the peer's tail work has landed.
            # relaxed arrive (matches the entry barrier at L378): this fence is
            # only about CTA lifetime, not inter-CTA smem visibility -- every
            # cross-CTA datum above already flows through mbarrier phases.
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

    # ====================================================================
    # DMA_A WARP — TMA-loads A tiles into sA[..., stage], one per K-iter.
    # 1-CTA: SM90_TMA_LOAD into local sA, arrives on local bar_full.
    # 2-CTA: SM100_TMA_LOAD_2SM (cluster-aware); peer's complete_tx routes
    #        onto the LEADER's bar_full. arrive_and_expect_tx is also
    #        peer-redirected to leader_rank.
    # ====================================================================
    @cute.experimental.jit
    def dma_a_warp(
        self,
        bar_full,  # Pointer[Int64], DMA_Stage entries
        bar_empty,  # Pointer[Int64], DMA_Stage entries
        leader_rank: cutlass.Int32,  # used for the 2-CTA arrive-peer redirect
        gA_tile: cute.Tensor,  # (CTA_M, CTA_K, Tiles_K) — this CTA's A strip
        sA: cute.Tensor,  # ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
        a_cta_v_map: cute.Layout,
        k_tile_count: cutlass.Int32,
    ):
        DMA_Stage = self.num_ab_stage

        # Phase-bit walkthrough for `bar_empty` (DMA_Stage=2 example).
        # Init phase = 0; phase flips on each arrival-count round. We wait
        # on the *old* phase about to flip away.
        #   k_tile  stage  old empty_phase  → wait for flip
        #     0       0          1                0→1 (init=0≠1, passes)
        #     1       1          1                0→1
        #     2       0          0                1→0 (slot reused)
        #     3       1          0                1→0
        #     4       0          1                0→1
        # Flip empty_phase once per DMA_Stage iters.
        empty_phase = cutlass.Int32(1)
        pdl_count = self.pdl_count

        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % DMA_Stage
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)

            # cute_ext.tma_load only manages TX bytes, NEVER the arrival
            # count (in the pipeline path producer_commit handles that).
            # bar_full[stage] needs 2 (1-CTA) or 4 (2-CTA) arrivals; we
            # arrive ourselves via the fused mbarrier.arrive.expect_tx PTX,
            # paired with update_expect_tx=False so TX bytes aren't
            # double-counted. peer_cta_rank_in_cluster=leader_rank in 2-CTA
            # routes the arrive onto the leader's bar; None (1-CTA) lands
            # on the local CTA's bar.
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bar_full + stage,
                    # 1 stage of A SMEM = sizeof(bf16) × CTA_M × CTA_K
                    cute.size_in_bytes(
                        sA.element_type,
                        cute.slice_(sA.layout, (None, None, None, 0)),
                    ),
                    peer_cta_rank_in_cluster=leader_rank if self.use_2cta else None,
                )
            cute_ext.tma_load(
                gA_tile[None, None, k_tile],  # (CTA_M, CTA_K) GMEM slice
                sA[None, None, None, stage],  # ((Mma_M,Mma_K),NumMma_M,NumMma_K)
                (bar_full + stage).value,  # Pointer→ir.Value bridge
                cta_v_map=a_cta_v_map,
                tma_operation_type=self.tma_op,
                update_expect_tx=False,
            )

            if stage == (DMA_Stage - 1):
                empty_phase = empty_phase ^ 1

            # PDL: launch dependent grid at pdl_count (default -1 = end).
            if cutlass.const_expr(self.use_pdl):
                if k_tile == pdl_count:
                    cute.arch.griddepcontrol_launch_dependents()

        if cutlass.const_expr(
            (self.push_mode == 5 or self.push_mode == 6)
            and self.test_trigger_before_copy
        ):
            cute.arch.griddepcontrol_launch_dependents()  # test mutant only: trigger before the copy is ordered
        if cutlass.const_expr(self.push_mode == 5):
            # Wait for warp 3's copy (named barrier 2: warp 3 + this whole warp), then a release fence, then trigger.
            cute.arch.sync_warp()
            # Non-aligned form: warp 3 reaches this barrier at a different instruction (PTX: .aligned requires every
            # participating thread to execute the same barrier instruction).
            llvm.inline_asm(
                None,
                [],
                "barrier.sync 2, 64;",
                "~{memory}",
                has_side_effects=True,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
            llvm.inline_asm(
                None,
                [],
                "fence.acq_rel.gpu;",
                "~{memory}",
                has_side_effects=True,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        if cutlass.const_expr(self.push_mode == 6):
            # Wait for every copy thread (named barrier 2: warp 3 + the four epilogue warps ARRIVE, this warp SYNCs;
            # 192 threads), then a release fence, then trigger.
            cute.arch.sync_warp()
            llvm.inline_asm(
                None,
                [],
                "barrier.sync 2, 192;",
                "~{memory}",
                has_side_effects=True,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
            llvm.inline_asm(
                None,
                [],
                "fence.acq_rel.gpu;",
                "~{memory}",
                has_side_effects=True,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        # producer tail: drain the remaining bar_empty arrivals from MMA so the
        # CTA stays alive until MMA's last tcgen05.commit lands — otherwise the
        # arrive can hit freed SMEM (illegal memory access).
        for k_tile in cutlass.range(DMA_Stage, unroll=1):
            stage = (k_tile + k_tile_count) % DMA_Stage
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            if stage == (DMA_Stage - 1):
                empty_phase = empty_phase ^ 1

    # ====================================================================
    # DMA_B WARP — same as DMA_A but for B; also drives the PDL
    # griddepcontrol.wait and signals the epilog when all activation loads
    # are issued.
    # 2-CTA: SM100_TMA_LOAD_2SM + arrive redirected to leader_rank.
    # ====================================================================
    @cute.experimental.jit
    def dma_b_warp(
        self,
        bar_full,  # Pointer[Int64], DMA_Stage entries
        bar_empty,  # Pointer[Int64], DMA_Stage entries
        bar_tma_epilog,  # Pointer[Int64], 1 entry (32-arrival)
        leader_rank: cutlass.Int32,
        gB_tile: cute.Tensor,  # (CTA_N, CTA_K, Tiles_K) — this CTA's B strip
        sB: cute.Tensor,  # ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
        b_cta_v_map: cute.Layout,
        k_tile_count: cutlass.Int32,
        push_phase: cute.Tensor,  # (1,) Int32 in shared memory (push epilogue only)
        mMeta: cute.Tensor,  # (PUSH_META_LEN,) int64 push arguments (push epilogue only)
    ):
        DMA_Stage = self.num_ab_stage

        # PDL griddepcontrol.wait: B is the activation, produced by an
        # upstream kernel. Block until that kernel signals completion. A
        # is weights (already resident), so DMA_A doesn't need this.
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        # Push epilogue: read the workspace phase AFTER the dependency wait (the counter was last written by the
        # previous user of the plane), publish it in shared memory, then arrive on bar_tma_epilog (32 arrivals); the
        # epilogue waits on that barrier before reading the word.
        if cutlass.const_expr(self.push_mode != 0):
            with cute.arch.elect_one():
                counter_addr = cutlass.Int64(mMeta[0])
                cnt = cutlass.Int32(
                    llvm.inline_asm(
                        T.i32(),
                        [counter_addr.ir_value()],
                        "ld.relaxed.gpu.global.b32 $0, [$1];",
                        "=r,l",
                        has_side_effects=True,
                        asm_dialect=llvm.AsmDialect.AD_ATT,
                    )
                )
                push_phase[0] = cnt & cutlass.Int32(1)
            cute.arch.mbarrier_arrive(bar_tma_epilog)

        # See dma_a_warp for empty_phase + fused arrive-and-expect-tx logic.
        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % DMA_Stage
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)

            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bar_full + stage,
                    # 1 stage of B SMEM = sizeof(bf16) × CTA_N × CTA_K
                    cute.size_in_bytes(
                        sB.element_type,
                        cute.slice_(sB.layout, (None, None, None, 0)),
                    ),
                    peer_cta_rank_in_cluster=leader_rank if self.use_2cta else None,
                )
            cute_ext.tma_load(
                gB_tile[None, None, k_tile],  # (CTA_N, CTA_K) GMEM slice
                sB[None, None, None, stage],  # ((Mma_N,Mma_K),NumMma_N,NumMma_K)
                (bar_full + stage).value,
                cta_v_map=b_cta_v_map,
                tma_operation_type=self.tma_op,
                update_expect_tx=False,
            )

            if stage == (DMA_Stage - 1):
                empty_phase = empty_phase ^ 1

        # 32-thread arrive on bar_tma_epilog: signals epilog "activations
        # issued" so the bias-load can fire after all B tiles have landed.
        # Only needed when has_bias=True; elided otherwise.
        if cutlass.const_expr(self.has_bias):
            cute.arch.mbarrier_arrive(bar_tma_epilog)

        # producer tail: drain the remaining bar_empty arrivals from MMA so the
        # CTA stays alive until MMA's last tcgen05.commit lands — otherwise the
        # arrive can hit freed SMEM (illegal memory access).
        for k_tile in cutlass.range(DMA_Stage, unroll=1):
            stage = (k_tile + k_tile_count) % DMA_Stage
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            if stage == (DMA_Stage - 1):
                empty_phase = empty_phase ^ 1

    # ====================================================================
    # MMA WARP — owns the TMEM accumulator and issues every tcgen05.mma.
    # 1-CTA: every CTA's MMA warp runs the K-loop on its own TMEM.
    # 2-CTA: BOTH CTAs alloc/dealloc TMEM (cluster-coherent via
    #        is_two_cta=True); only the leader runs the K-loop; commits
    #        multicast to both CTAs (mask=0b11) so each one's bar_empty /
    #        bar_mma_epilog still gets signaled.
    # ====================================================================
    @cute.experimental.jit
    def mma_warp(
        self,
        is_leader: cutlass.Boolean,
        bar_full,  # Pointer[Int64], DMA_Stage entries
        bar_empty,  # Pointer[Int64], DMA_Stage entries
        bar_mma_epilog,  # Pointer[Int64], 1 entry
        bar_tmem_alloc,  # Pointer[Int64], 1 entry, 160-arrival
        tiled_mma: cute.TiledMma,
        sA: cute.Tensor,  # ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
        sB: cute.Tensor,  # ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
        tmem_base_ptr,  # Pointer[Int32] — SMEM slot for TMEM addr
        acc_layout: cutlass.Constexpr,  # TMEM accumulator layout
        mma_inst_tile_k: cutlass.Constexpr,  # NumMma_K — inner loop count
        k_tile_count: cutlass.Int32,  # Tiles_K — outer loop count
    ):
        DMA_Stage = self.num_ab_stage

        # ---- TMEM allocation (manual) ----
        # We allocate TMEM explicitly here rather than via the convenience
        # helper ``cute_ext.allocate(tmem)`` so that we can *hide the TMEM
        # allocation latency*: the alloc must happen here in MMA's prolog
        # (we need the TMEM ptr before the first tcgen05.mma), but it
        # serializes against the next CTA's own alloc on the back-to-back
        # PDL launch path. By manually emitting
        # ``relinquish_tmem_alloc_permit`` right after our alloc completes,
        # the next CTA can start ITS allocation in parallel with this
        # CTA's MMA work rather than stalling at its own prolog waiting
        # for us. cute_ext.allocate(tmem) does not expose this early-
        # relinquish control — worth ~3% on the PDL hot path.
        #
        # TMEM on SM100 has 128 lanes × 512 columns × 4B = 256KB total. We
        # allocate half (256 of 512 cols) so the next CTA can prefetch its
        # alloc on the other half. For CTA_M=64 the accumulator only uses
        # 64 lanes (16 lanes × 4 subpartitions), well within half-of-TMEM.
        # 2-CTA: alloc/relinquish/dealloc are cluster-coherent (is_two_cta=True).
        num_tmem_cols = 256
        cute.arch.alloc_tmem(num_tmem_cols, tmem_base_ptr, is_two_cta=self.use_2cta)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)  # phase 0: 32 of 160
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=self.use_2cta)

        # Bind acc_layout to the just-allocated TMEM ptr. Drop the trailing
        # AccStage=1 mode so the MMA atom sees (MMA, Mma_M, Mma_N).
        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        tAcc = cute.make_tensor(
            tmem_ptr, acc_layout
        )  # ((Mma_M_per_cta, Mma_N), NumMma_M, NumMma_N, AccStage)
        acc_view = tAcc[
            None, None, None, 0
        ]  # ((Mma_M_per_cta, Mma_N), NumMma_M, NumMma_N)

        if is_leader:
            # MMA atom: each cute_ext.dot call = 1 tcgen05.mma. ACCUMULATE:
            #   False → C  = A*B  (first inst of first k-tile only)
            #   True  → C += A*B
            mma_atom = cute.make_mma_atom(tiled_mma.op)
            # 2-CTA: mask=0b11 multicasts the commit to both CTAs' bar_empty.
            commit_mask = 0b11 if self.use_2cta else None

            full_phase = cutlass.Int32(0)
            for k_tile in cutlass.range(k_tile_count, unroll=1):
                stage = k_tile % DMA_Stage
                cute.arch.mbarrier_wait(bar_full + stage, full_phase)

                # Inner-K loop: NumMma_K straight-line tcgen05.mma instructions.
                # cute_ext.dot expects rank-3 operands in the MMA fragment
                # profile (MMA_atom, REST_M, REST_K). Slicing
                # `sA[None,None,k_block,stage]` leaves rank 2
                # ((Mma_M, Mma_K), NumMma_M), so pad explicitly via
                # cute.append_ones.
                for k_block in range(mma_inst_tile_k):
                    if k_block == 0:
                        # First inst of the very first k_tile clears TMEM.
                        mma_atom.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                    else:
                        mma_atom.set(tcgen05.Field.ACCUMULATE, True)
                    a_frag = cute.append_ones(
                        sA[None, None, k_block, stage],
                        up_to_rank=3,
                    )
                    b_frag = cute.append_ones(
                        sB[None, None, k_block, stage],
                        up_to_rank=3,
                    )
                    cute_ext.dot(mma_atom, a_frag, b_frag, acc_view)

                # CRITICAL: tcgen05.commit MUST be inside elect_one(). Unlike
                # the C++ ``cutlass::arch::umma_arrive`` helper (which has an
                # internal elect_one_sync), the DSL's tcgen05.commit has no
                # guard — without this, all 32 lanes commit (32× redundant
                # arrivals on bar_empty). This was a major perf bug encountered
                # during development.
                with cute.arch.elect_one():
                    tcgen05.commit(bar_empty + stage, commit_mask, self.cta_group)

                if (k_tile % DMA_Stage) == (DMA_Stage - 1):
                    full_phase = full_phase ^ 1

            # Wake the epilog (1-thread arrive on the 1-arrival barrier).
            with cute.arch.elect_one():
                tcgen05.commit(bar_mma_epilog, commit_mask, self.cta_group)

        # ---- TMEM dealloc: wait for own EPILOG's tcgen05.ld to retire, free.
        # bar_tmem_alloc phase 1 fires after EPILOG's tcgen05.ld is observable
        # (post fence_view_async_tmem_load). 2-CTA dealloc is per-CTA (no
        # cross-CTA handshake) because each CTA owns its own physical TMEM
        # half; the cluster-shared accumulator is just a logical view.
        cute.arch.mbarrier_arrive(bar_tmem_alloc)  # phase 1: 32 of 160
        cute.arch.mbarrier_wait(bar_tmem_alloc, 1)
        cute.arch.dealloc_tmem(tmem_ptr, num_tmem_cols, is_two_cta=self.use_2cta)

    @cute.experimental.jit
    def copy_warp(self, mMeta: cute.Tensor, bidx: cutlass.Int32, bidy: cutlass.Int32):
        # After this warp's own dependency wait: copy this CTA's share of the residual rows (plain 16-byte loads /
        # stores, byte-exact) and, in CTA (0, 0), the workspace phase; then named barrier 2 with the DMA_A warp, which
        # fences (release) and triggers the dependents. Every CTA runs this before its trigger.
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.test_trigger_before_copy):
            for _ in cutlass.range(64, unroll=1):  # test mutant only: delay the copy
                llvm.inline_asm(
                    None,
                    [],
                    "nanosleep.u32 1000;",
                    "~{memory}",
                    has_side_effects=True,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                )
        lane = cute.arch.lane_idx()
        gx, gy, _ = cute.arch.grid_dim()
        cta = bidy * gx + bidx
        ncta = gx * gy
        src = cutlass.Int64(mMeta[17])
        dst = cutlass.Int64(mMeta[18])
        # Vector counts fit 32 bits (rows x hidden x 4 bytes / 16); addresses stay 64-bit. No value is reassigned inside
        # a branch (the DSL needs both branches of a conditional to yield one type).
        nvec = cutlass.Int32(mMeta[19] // 16)
        per = (nvec + ncta - 1) // ncta
        v0 = cta * per
        for i in cutlass.range(lane, per, 32, unroll=1):
            v = v0 + i
            if v < nvec:
                off = cutlass.Int64(v) * cutlass.Int64(16)
                llvm.inline_asm(
                    None,
                    [(src + off).ir_value(), (dst + off).ir_value()],
                    "{\n.reg .b32 c<4>;\nld.global.v4.b32 {c0, c1, c2, c3}, [$0];\nst.global.v4.b32 [$1], {c0, c1, c2, c3};\n}",
                    "l,l,~{memory}",
                    has_side_effects=True,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                )
        if cta == cutlass.Int32(0):
            if lane == cutlass.Int32(0):
                counter_addr = cutlass.Int64(mMeta[0])
                phase_out = cutlass.Int64(mMeta[20])
                llvm.inline_asm(
                    None,
                    [counter_addr.ir_value(), phase_out.ir_value()],
                    "{\n.reg .b32 c;\nld.relaxed.gpu.global.b32 c, [$0];\nand.b32 c, c, 1;\nst.global.b32 [$1], c;\n}",
                    "l,l,~{memory}",
                    has_side_effects=True,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                )
        cute.arch.sync_warp()  # lanes leave the copy loop / phase branch at different points: reconverge first
        # Non-aligned named barrier 2 (64 threads: this warp + the DMA_A warp, which reaches it at another instruction).
        llvm.inline_asm(
            None,
            [],
            "barrier.sync 2, 64;",
            "~{memory}",
            has_side_effects=True,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

    @cute.experimental.jit
    def copy_batched(
        self,
        mMeta: cute.Tensor,
        bidx: cutlass.Int32,
        bidy: cutlass.Int32,
        ct: cutlass.Int32,
        with_phase: cutlass.Constexpr,
    ):
        # push_mode 6: this thread's share (index ct of 160 copy threads: warp 3 = 0..31, epilogue warps = 32..159) of
        # the CTA's residual rows, after this thread's own dependency wait (warp 3 waits here; the epilogue warps wait
        # before calling), eight 16-byte vectors in flight per batch (predicated loads, then the stores; byte-exact);
        # warp 3 lane 0 of CTA 0 also copies the workspace phase. Then ARRIVE on named barrier 2 (192 threads with
        # the DMA_A warp, which waits there, fences and triggers).
        if cutlass.const_expr(with_phase):
            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.test_trigger_before_copy):
            for _ in cutlass.range(64, unroll=1):  # test mutant only: delay the copy
                llvm.inline_asm(
                    None,
                    [],
                    "nanosleep.u32 1000;",
                    "~{memory}",
                    has_side_effects=True,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                )
        gx, gy, _ = cute.arch.grid_dim()
        cta = bidy * gx + bidx
        ncta = gx * gy
        src = cutlass.Int64(mMeta[17])
        dst = cutlass.Int64(mMeta[18])
        nvec = cutlass.Int32(mMeta[19] // 16)
        per = (nvec + ncta - 1) // ncta
        v0 = cta * per
        # end = min(v0 + per, nvec) and avail = max(end - (v0 + ct), 0) without reassigning a value inside a branch.
        over = v0 + per - nvec
        end = v0 + per - cutlass.Int32(over > cutlass.Int32(0)) * over
        rem = end - (v0 + ct)
        avail = rem * cutlass.Int32(rem > cutlass.Int32(0))
        nj = (avail + cutlass.Int32(159)) // cutlass.Int32(160)
        stride = cutlass.Int64(160 * 16)
        for b in cutlass.range(0, nj, 8, unroll=1):
            off = cutlass.Int64(v0 + ct + b * cutlass.Int32(160)) * cutlass.Int64(16)
            llvm.inline_asm(
                None,
                [
                    (src + off).ir_value(),
                    (dst + off).ir_value(),
                    stride.ir_value(),
                    (nj - b).ir_value(),
                ],
                _COPY8_ASM,
                "l,l,l,r,~{memory}",
                has_side_effects=True,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        if cutlass.const_expr(with_phase):
            if cta == cutlass.Int32(0):
                if ct == cutlass.Int32(0):
                    counter_addr = cutlass.Int64(mMeta[0])
                    phase_out = cutlass.Int64(mMeta[20])
                    llvm.inline_asm(
                        None,
                        [counter_addr.ir_value(), phase_out.ir_value()],
                        "{\n.reg .b32 c;\nld.relaxed.gpu.global.b32 c, [$0];\nand.b32 c, c, 1;\nst.global.b32 [$1], c;\n}",
                        "l,l,~{memory}",
                        has_side_effects=True,
                        asm_dialect=llvm.AsmDialect.AD_ATT,
                    )
        cute.arch.sync_warp()  # lanes leave the batch loop / phase branch at different points: reconverge first
        # Non-aligned ARRIVE on named barrier 2 (PTX: barrier.arrive orders this thread's prior accesses before the
        # barrier completes and synchronizes with the DMA_A warp's barrier.sync on the same barrier).
        llvm.inline_asm(
            None,
            [],
            "barrier.arrive 2, 192;",
            "~{memory}",
            has_side_effects=True,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

    @cute.experimental.jit
    def ready_warp(self, mMeta: cute.Tensor, bidx: cutlass.Int32, bidy: cutlass.Int32):
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
        if bidx == cutlass.Int32(0):
            if bidy == cutlass.Int32(0):
                lane = cute.arch.lane_idx()
                ready = cutlass.Int64(mMeta[7])
                grid = cutlass.Int32(mMeta[16])
                llvm.inline_asm(
                    None,
                    [],
                    "fence.acq_rel.gpu;",
                    "~{memory}",
                    has_side_effects=True,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                )
                for b in cutlass.range(lane, grid, 32, unroll=1):
                    llvm.inline_asm(
                        None,
                        [
                            (ready + cutlass.Int64(b) * cutlass.Int64(4)).ir_value(),
                            cutlass.Int32(1).ir_value(),
                        ],
                        "st.relaxed.gpu.global.b32 [$0], $1;",
                        "l,r,~{memory}",
                        has_side_effects=True,
                        asm_dialect=llvm.AsmDialect.AD_ATT,
                    )

    # ====================================================================
    # EPILOG WARPS — TMEM → RMEM → bf16 cast → direct st.global to GMEM.
    # ====================================================================
    @cute.experimental.jit
    def epilog_warp(
        self,
        bar_tma_epilog,  # Pointer[Int64], 1 entry (32-arrival)
        bar_mma_epilog,  # Pointer[Int64], 1 entry
        bar_tmem_alloc,  # Pointer[Int64], 1 entry, 160-arrival
        tmem_base_ptr,  # Pointer[Int32] — SMEM slot from MMA
        acc_layout: cutlass.Constexpr,
        gD_tile: cute.Tensor,  # (CTA_M, CTA_N) — this CTA's output tile
        gBias_tile: cute.Tensor,  # (CTA_M, CTA_N) with stride (1,0) — bias broadcast
        epi_tid: cutlass.Int32,  # 0..127 within the 4 EPILOG warps
        c_dtype: cutlass.Constexpr,
        d_layout: cutlass.Constexpr,
        sA: cute.Tensor,  # A stage buffers: reused as the restage tile by the push epilogue (after the last MMA)
        push_phase: cute.Tensor,  # (1,) Int32 in shared memory (push epilogue only)
        mMeta: cute.Tensor,  # (PUSH_META_LEN,) int64 push arguments (push epilogue only)
        m_idx: cutlass.Int32,  # this CTA's tile index along kernel M (output features)
        n_idx: cutlass.Int32,  # this CTA's tile index along kernel N (rows)
    ):
        # Sync with MMA's alloc_tmem (phase 0 of bar_tmem_alloc): 128 arrivals
        # from this warp + 32 from MMA = 160. tmem_base_ptr is uninitialized
        # before the wait clears.
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)

        # Push epilogue: everything the push needs except the tile itself is loaded here, while the main loop runs.
        # The phase word is published by DMA_B after its dependency wait (bar_tma_epilog, 32 arrivals).
        if cutlass.const_expr(self.push_mode != 0):
            cute.arch.mbarrier_wait(bar_tma_epilog, 0)
            if cutlass.const_expr(self.push_mode == 6):
                # This thread's share of the residual copy, after its own dependency wait, while the main loop runs.
                if cutlass.const_expr(self.use_pdl):
                    cute.arch.griddepcontrol_wait()
                ebx, eby, _ = cute.arch.block_idx()
                self.copy_batched(mMeta, ebx, eby, epi_tid + cutlass.Int32(32), False)
            push_plane_off = cutlass.Int64(push_phase[0]) * cutlass.Int64(mMeta[1])
            push_rank = cutlass.Int32(mMeta[2])
            push_rows = cutlass.Int64(mMeta[3])
            push_chunk = cutlass.Int64(mMeta[4])
            push_out = cutlass.Int64(mMeta[5])
            push_ldc = cutlass.Int64(mMeta[6])
            push_peers = []
            for d in cutlass.range_constexpr(self.push_world):
                push_peers.append(cutlass.Int64(mMeta[8 + d]) + push_plane_off)

        # Bind acc_layout to the TMEM ptr written by alloc_tmem.
        # acc_layout shape: ((Mma_M, Mma_N), NumMma_M, NumMma_N, AccStage)
        #                 = (((16,4), Mma_N), 1, 1, 1).
        # acc_view drops the 1×1×1 outer modes → (Mma_M, Mma_N).
        #
        # Concrete TMEM layout for 1-CTA bf16 M64 N8:
        #   tmem_[32b](0x0...) o (((16,4),8),1,1,1):(((65536,2097152),1),0,0,0)
        #   TMEM addr = [31:16=dp_lane, 15:0=column]:
        #     stride between dp lanes = 1<<16 = 65536           (Mma_M_per_subp)
        #     stride between subparts = 65536 × 32 = 2097152    (NumSubp=4)
        #     stride between cols     = 1                       (N is contiguous)
        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        tCtAcc = cute.make_tensor(
            tmem_ptr, acc_layout
        )  # ((Mma_M_per_cta, Mma_N), NumMma_M, NumMma_N, AccStage)
        acc_view = tCtAcc[((None, None), 0, 0, 0)]  # (Mma_M_per_cta, Mma_N)

        # ---- t2r tiled-copy + per-thread RMEM layout ----
        # ``sm100_utils.get_tmem_load_op`` picks the best tcgen05.ld atom
        # for the (M, N, K) tile and output dtype. For CTA_M=64, M/N-major
        # output, the chosen atom is SM100_TMEM_LOAD_16dp256b1x — the 16dp
        # variant is required because each MMA subpartition only uses 16
        # of the 32 lanes per subpartition (CTA_M=64 split across 4 subps =
        # 16 lanes each). Other tcgen05.ld atoms would also be functional;
        # 16dp is just optimal here.
        # epi_tile = full CTA tile: N=8 fits in one tcgen05.ld instance
        # (16dp × 8col → 4 regs per thread × 128 threads). Larger N would
        # need a sub-tile loop here.
        epi_tile = (self.cta_m, self.cta_n)
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            (self.cta_m, self.cta_n, self.cta_k),
            d_layout,
            c_dtype,
            self.acc_dtype,
            epi_tile,
            self.use_2cta,
        )
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(copy_atom_t2r, acc_view)

        # flat_divide → (CTA_M, CTA_N, Rest_M=1, Rest_N=1); the rank-4 shape
        # is required because make_t2r_rmem_layout internally does [...,0,0].
        gD_epi = cute.flat_divide(gD_tile, epi_tile)

        # rmem_layout matches one t2r copy's destination. The TMEM-side
        # partition (CpyS) per-subpartition view is shape ((CpyS_N=8,
        # CpyS_M=16), NumCpy_M=1, NumCpy_N=Mma_N/CpyS_N) — i.e. each
        # tcgen05.ld.16dp256bit.x1 instance moves 16dp×8col elements per
        # subpartition. Per the PTX page for tcgen05.ld.16dp256bit, thread
        # 0 of an SM100_TMEM_LOAD_16dp256b1x copy receives 4 registers at
        # output coordinates (0,0), (0,1), (8,0), (8,1) — i.e. a (2,2)
        # per-thread value-tile, hence CpyD = (2,2). The dst rmem layout
        # mirrors this CpyD shape but with stride (1, 2) so the registers
        # are stored contiguously.
        rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, gD_epi, epi_tid)
        rAcc = cute_ext.allocate(  # fp32, per-thread
            self.acc_dtype,
            cute.AddressSpace.rmem,
            rmem_layout,
            alignment=32,
        )
        rD = cute_ext.allocate(  # bf16, per-thread
            c_dtype,
            cute.AddressSpace.rmem,
            rmem_layout,
            alignment=32,
        )
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)

        # ---- Bias setup (GMEM→RMEM via partition_and_copy, predicate auto) ----
        # gBias_tile has the same (CTA_M, CTA_N) shape as gD_tile but with
        # stride (1, 0) — the bias value depends only on the M coord. Each
        # lane needs to read the same (m, n) coord it writes back to gD so
        # bias and accumulator align register-for-register.
        #
        # partition_and_copy slices the GMEM src using
        # `tiled_copy.layout_src_tv_tiled`. With t2r that's the TMEM (CpyS)
        # side, which is 16dp×8col=128 elements/group — doesn't match rBias's
        # CpyD=4 elements/thread. `make_tiled_copy_D` builds a new TiledCopy
        # whose src/dst TV layouts both equal t2r's *dst* (CpyD) layout, so
        # the GMEM src gets sliced to CpyD per thread — matching rBias. OOB
        # is handled by the cute_ext lowering's auto-predBounds (same story
        # as the st.global to mC).
        if cutlass.const_expr(self.has_bias):
            bias_dtype = gBias_tile.element_type
            gBias_epi = cute.flat_divide(gBias_tile, epi_tile)
            tiled_copy_g2r = cute.make_tiled_copy_D(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), bias_dtype),
                tiled_copy_t2r,
            )
            thr_g2r = tiled_copy_g2r.get_slice(epi_tid)
            rBias = cute_ext.allocate(  # bias dtype (e.g. bf16)
                bias_dtype,
                cute.AddressSpace.rmem,
                rmem_layout,
                alignment=32,
            )
            rBiasAcc = cute_ext.allocate(  # converted to fp32
                self.acc_dtype,
                cute.AddressSpace.rmem,
                rmem_layout,
                alignment=32,
            )

        # ---- Wait for data: B loads done + MMA done ----
        # Only needed when has_bias=True (DMA_B signals bar_tma_epilog, epilog
        # waits before issuing the bias load so it hides behind MMA). Without
        # bias the bar is unused — neither arrive nor wait fires.
        if cutlass.const_expr(self.has_bias):
            cute.arch.mbarrier_wait(bar_tma_epilog, 0)  # all activations issued

        # ---- Bias load: GMEM → RMEM (predicated ld.global) and convert to fp32 ----
        # Done before the MMA wait so the load latency hides behind MMA.
        if cutlass.const_expr(self.has_bias):
            cute_ext.partition_and_copy(thr_g2r, gBias_epi[None, None, 0, 0], rBias)
            rBiasAcc.store(rBias.load().to(self.acc_dtype))

        cute.arch.mbarrier_wait(bar_mma_epilog, 0)  # accumulator ready

        # TMEM → RMEM (one tcgen05.ld for the 64×8 tile).
        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)

        # tcgen05.ld is async — fence makes the result visible to (a) the
        # rAcc.load() below and (b) MMA's dealloc_tmem after we arrive
        # on bar_tmem_alloc phase 1.
        cute.arch.fence_view_async_tmem_load()

        # Phase 1 of bar_tmem_alloc: 128 from this warp + 32 from MMA = 160.
        # MMA's mbarrier_wait(bar_tmem_alloc, 1) clears and it dealloc's.
        cute.arch.mbarrier_arrive(bar_tmem_alloc)

        # Add bias in fp32 before the dtype cast.
        if cutlass.const_expr(self.has_bias):
            rAcc.store(rAcc.load() + rBiasAcc.load())

        # fp32 → bf16 cast in RMEM.
        rD.store(rAcc.load().to(c_dtype))

        # RMEM → GMEM: cute_ext.partition_and_copy does two things —
        #   (a) PARTITION: splits non-RMEM operands by thr_copy's thread-value
        #       layout (RMEM operands skipped — already per-thread). Here only
        #       gD_epi (the dst) gets partitioned; rD stays as-is.
        #   (b) INSTRUCTION: picked from (src, dst) memspace pair —
        #         TMEM→RMEM = tcgen05.ld     (atom-driven)
        #         RMEM→GMEM = st.global     (simt_auto_vec_copy)
        #         GMEM→SMEM = cp.async       (async_op=True)
        # We REUSE thr_t2r so each lane writes to the same (m, n) GMEM coords
        # where it read from TMEM. A different copy atom would scatter
        # registers to the wrong addresses.
        #
        # OOB STORES are handled automatically by the cute_ext lowering —
        # simt_auto_vec_copy infers predBounds from the destination tensor's
        # MemRef shape (propagated from mC through local_tile / flat_divide).
        # Lanes whose (m, n) coord is past mC's shape simply don't issue an
        # st.global. Verified with a non-aligned M=63 test: writes to in-bounds rows
        # 0..62 land correctly, the M=63 row + the rest of the (64,8) tile
        # stay untouched. So the C++ / vanilla-DSL kernels' explicit predicate
        # dance (make_identity_tensor + elem_less + basic_copy_if) isn't
        # needed; the explicit `predicated_tensor_origin` marker isn't either.
        if cutlass.const_expr(self.push_mode == 0):
            cute_ext.partition_and_copy(thr_t2r, rD, gD_epi[None, None, 0, 0])
        else:
            # Restage the tile in the A stage buffers (row-major per kernel-N row, padded pitch). They were last
            # accessed by the async proxy (TMA writes, MMA reads, all complete once bar_mma_epilog completed); order
            # those accesses before the generic-proxy writes.
            s_base = cutlass.Int32(sA.iterator.toint())
            sD = cute.make_tensor(
                cute.make_ptr(
                    c_dtype, s_base, cute.AddressSpace.smem, assumed_align=128
                ),
                cute.make_layout((self.cta_m, self.cta_n), stride=(1, self.cta_m + 8)),
            )
            sD_epi = cute.flat_divide(sD, epi_tile)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute_ext.partition_and_copy(thr_t2r, rD, sD_epi[None, None, 0, 0])
            if cutlass.const_expr(not self.test_drop_restage_barrier):
                cute.arch.barrier(barrier_id=1, number_of_threads=128)
            pitch = self.cta_m + 8
            vecs_per_row = self.cta_m // 8
            n_vec = self.cta_n * vecs_per_row
            for i in cutlass.range_constexpr((n_vec + 127) // 128):
                v = epi_tid + cutlass.Int32(128 * i)
                t = v // cutlass.Int32(vecs_per_row)
                j = v - t * cutlass.Int32(vecs_per_row)
                tok = cutlass.Int64(n_idx) * cutlass.Int64(self.cta_n) + cutlass.Int64(
                    t
                )
                feat = cutlass.Int64(m_idx) * cutlass.Int64(self.cta_m) + cutlass.Int64(
                    j
                ) * cutlass.Int64(8)
                if v < cutlass.Int32(n_vec):
                    if tok < push_rows:
                        s_addr = s_base + (
                            t * cutlass.Int32(pitch) + j * cutlass.Int32(8)
                        ) * cutlass.Int32(2)
                        goff = (tok * push_ldc + feat) * cutlass.Int64(2)
                        addrs, flags = [], []
                        if cutlass.const_expr(self.push_mode == 7):
                            _publish_vector(s_addr, push_peers[0] + goff)
                        elif cutlass.const_expr(self.push_mode == 1):
                            for d in cutlass.range_constexpr(self.push_world):
                                addrs.append(push_peers[d] + goff)
                                flags.append(
                                    cutlass.Int32(push_rank != cutlass.Int32(d))
                                )
                        elif cutlass.const_expr(
                            self.push_mode == 3
                            or self.push_mode == 5
                            or self.push_mode == 6
                        ):
                            for d in cutlass.range_constexpr(self.push_world):
                                addrs.append(push_peers[d] + goff)
                                flags.append(cutlass.Int32(1))
                        else:
                            owner = cutlass.Int32(0)
                            for d in cutlass.range_constexpr(1, self.push_world):
                                owner = owner + cutlass.Int32(
                                    tok >= push_chunk * cutlass.Int64(d)
                                )
                            loff = (
                                (tok - cutlass.Int64(owner) * push_chunk) * push_ldc
                                + feat
                            ) * cutlass.Int64(2)
                            for d in cutlass.range_constexpr(self.push_world):
                                addrs.append(push_peers[d] + loff)
                                flags.append(
                                    cutlass.Int32(owner == cutlass.Int32(d))
                                    * cutlass.Int32(push_rank != cutlass.Int32(d))
                                )
                        if cutlass.const_expr(self.push_mode != 7):
                            _push_vector(s_addr, push_out + goff, addrs, flags)

    # ====================================================================
    # PUSH EPILOGUE (push_mode != 0). Restage the bf16 tile through shared memory (the A stage buffers, last read by
    # the MMA before bar_mma_epilog completed), then let each thread move whole 16-byte row vectors: a plain store to
    # this rank's output (the partial the consumer reads for its own rank) and, after the empty-marker remap
    # (an all-zero 32-bit word becomes 0x00008000), relaxed system-scope stores into the peer slots of the current
    # workspace phase. Consecutive threads take consecutive vectors of a row, so a warp store covers whole 128-byte
    # row segments.
    # ====================================================================


# =====================================================================
# BMM wrappers — feed (L, M, K) / (L, K, N) / (L, M, N) PyTorch-shaped
# cute tensors and select-permute them into the kernel's (M, K, L) /
# (N, K, L) / (M, N, L) view.
# =====================================================================
@cute.experimental.jit
def _bmm_no_bias(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,  # (L, M, K) from PyTorch
    b: cute.Tensor,  # (L, K, N) from PyTorch (permuted)
    c: cute.Tensor,  # (L, M, N) from PyTorch (permuted)
    stream: cuda.CUstream,
):
    a = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0]))
    b = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0]))
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    # Dummy bias aliasing C — never dereferenced when has_bias=False.
    bias = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[0, 1, 2]))
    gemm_op(a, b, c, bias, bias, stream)


@cute.experimental.jit
def _bmm_push(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,  # (L, M, K) from PyTorch
    b: cute.Tensor,  # (L, K, N) from PyTorch (permuted)
    c: cute.Tensor,  # (L, M, N) from PyTorch (permuted)
    meta: cute.Tensor,  # (PUSH_META_LEN,) int64 push arguments
    stream: cuda.CUstream,
):
    a = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0]))
    b = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0]))
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    bias = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[0, 1, 2]))
    gemm_op(a, b, c, bias, meta, stream)


@cute.experimental.jit
def _bmm_bias(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,  # (L, M, K) from PyTorch
    b: cute.Tensor,  # (L, K, N) from PyTorch (permuted)
    c: cute.Tensor,  # (L, M, N) from PyTorch (permuted)
    bias: cute.Tensor,  # (L, M, N):(0,1,0) — M-broadcast via as_strided
    stream: cuda.CUstream,
):
    a = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0]))
    b = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0]))
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    # After [1,2,0] permute: (M, N, L):(1, 0, 0) — matches C++ make_layout_Bias.
    bias = cute.make_tensor(bias.iterator, cute.select(bias.layout, mode=[1, 2, 0]))
    gemm_op(a, b, c, bias, bias, stream)


# =====================================================================
# Compile + cache helpers (FlashInfer-side wrappers around the kernel).
# Caches by *config* only — the kernel uses ``mark_layout_dynamic`` so
# the same compiled binary handles any shape. Representative tensors used
# at compile time pin the dtype, rank, and leading-dim pattern; their
# concrete extents are erased.
# =====================================================================
_TORCH_TO_CUTLASS_DTYPE = {
    torch.bfloat16: cutlass.BFloat16,
}

_TORCH_TO_CUTLASS_OUT_DTYPE = {
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}

# Per-process cache mapping (dtype, config…) → compiled cute_ext callable.
# We need to construct concrete cute.Tensors once and reuse the resulting
# compiled function across all live calls; a fresh build per call would
# defeat the cache.
_TGV_CUTE_EXT_COMPILE_CACHE: dict = {}


def _detect_leading_dim(t: torch.Tensor) -> int:
    """Return the index of the stride-1 dim. Used by ``_to_cute_swap`` so the
    integration layer auto-adapts to whichever of K-major / MN-major the
    caller hands us. Falls back to the innermost dim if no stride-1 dim
    exists (shouldn't happen for well-formed tensors)."""
    for i, s in enumerate(t.stride()):
        if s == 1:
            return i
    return t.dim() - 1


def _make_layout_tensor(
    shape: Tuple[int, ...], dtype: torch.dtype, leading_dim: int
) -> torch.Tensor:
    """Allocate a torch tensor of ``shape`` where ``leading_dim`` is the
    contiguous (stride-1) dim. Used to synthesize representative tensors at
    compile time for arbitrary layout patterns."""
    ndim = len(shape)
    # Permute leading_dim to last, allocate contig, then permute back.
    perm = [i for i in range(ndim) if i != leading_dim] + [leading_dim]
    permuted_shape = tuple(shape[p] for p in perm)
    t = torch.empty(permuted_shape, dtype=dtype, device="cuda")
    inv_perm = [perm.index(i) for i in range(ndim)]
    return t.permute(inv_perm)


def _make_compile_repr_tensors(
    dtype: torch.dtype,
    c_dtype: torch.dtype,
    has_bias: bool,
    a_leading: int,
    b_leading: int,
    c_leading: int,
):
    """Build representative tensors with strides matching the requested
    leading-dim pattern. After the A↔B swap, the cute_ext kernel sees:
        A_ce: shape (L, N_pt, K)   <- (L=1, dim 1, dim 2)
        B_ce: shape (L, K, M_pt)   <- (L=1, dim 1, dim 2)
        C_ce: shape (L, N_pt, M_pt) <- (L=1, dim 1, dim 2)
    Each of {a, b, c}_leading ∈ {1, 2} selects which dim is the stride-1
    one (K vs kernel-M for A, K vs kernel-N for B, kernel-M vs kernel-N
    for C). The compiled binary is specialized per ``(a, b, c)_leading``
    tuple.
    """
    # Pick aligned-but-small extents that satisfy the kernel's tile shape
    # while leaving room for either dim to be the contig one.
    M, N, K, L = 64, 8, 128, 1  # M_pt=M, N_pt=N
    A_t = _make_layout_tensor(
        (L, N, K), dtype, a_leading
    )  # kernel A shape (L, N_pt, K)
    B_t = _make_layout_tensor(
        (L, K, M), dtype, b_leading
    )  # kernel B shape (L, K, M_pt)
    C_t = _make_layout_tensor(
        (L, N, M), c_dtype, c_leading
    )  # kernel C shape (L, N_pt, M_pt)

    a_ = from_dlpack(A_t, assumed_align=32).mark_layout_dynamic(leading_dim=a_leading)
    b_ = from_dlpack(B_t, assumed_align=32).mark_layout_dynamic(leading_dim=b_leading)
    c_ = from_dlpack(C_t, assumed_align=32).mark_layout_dynamic(leading_dim=c_leading)

    if not has_bias:
        return a_, b_, c_, None

    # bias is per-output-feature in PyTorch terms; after the A↔B swap that
    # maps to broadcast over the kernel's N axis and L (stride (0, 1, 0)).
    bias_t = torch.empty((N,), dtype=dtype, device="cuda")
    bias_3d = bias_t.as_strided(size=(L, N, M), stride=(0, 1, 0))
    bias_ = from_dlpack(bias_3d, assumed_align=2).mark_layout_dynamic(leading_dim=1)
    return a_, b_, c_, bias_


def _get_compiled_cute_ext_kernel(
    dtype: torch.dtype,
    c_dtype: torch.dtype,
    cta_m: int,
    cta_n: int,
    cta_k: int,
    num_ab_stage: int,
    use_2cta: bool,
    use_pdl: bool,
    has_bias: bool,
    a_leading: int,
    b_leading: int,
    c_leading: int,
    push_mode: int = 0,
    push_world: int = 1,
    test_drop_restage_barrier: bool = False,
    test_trigger_before_copy: bool = False,
):
    """Compile (or fetch from cache) a TgvGemmCuteExtKernel for the given
    config + layout pattern. Returns a callable with signature
        compiled(a_, b_, c_, [bias_,] stream)
    accepting cute.Tensors. The same compiled binary handles any
    (M, N, K, L) consistent with the (a, b, c)_leading layout pattern
    thanks to ``mark_layout_dynamic``.
    """
    key = (
        dtype,
        c_dtype,
        cta_m,
        cta_n,
        cta_k,
        num_ab_stage,
        bool(use_2cta),
        bool(use_pdl),
        bool(has_bias),
        a_leading,
        b_leading,
        c_leading,
        int(push_mode),
        int(push_world),
        bool(test_drop_restage_barrier),
        bool(test_trigger_before_copy),
    )
    cached = _TGV_CUTE_EXT_COMPILE_CACHE.get(key)
    if cached is not None:
        return cached

    if dtype not in _TORCH_TO_CUTLASS_DTYPE:
        raise ValueError(
            f"TGV cute_ext backend supports {list(_TORCH_TO_CUTLASS_DTYPE)}; got {dtype}."
        )
    if c_dtype not in _TORCH_TO_CUTLASS_OUT_DTYPE:
        raise ValueError(
            f"TGV cute_ext output supports {list(_TORCH_TO_CUTLASS_OUT_DTYPE)}; "
            f"got {c_dtype}."
        )

    gemm = TgvGemmCuteExtKernel(
        acc_dtype=cutlass.Float32,
        cta_m=cta_m,
        cta_n=cta_n,
        cta_k=cta_k,
        num_ab_stage=num_ab_stage,
        use_2cta=use_2cta,
        use_pdl=use_pdl,
        has_bias=has_bias,
        out_dtype=_TORCH_TO_CUTLASS_OUT_DTYPE[c_dtype],
        push_mode=push_mode,
        push_world=push_world,
        test_drop_restage_barrier=test_drop_restage_barrier,
        test_trigger_before_copy=test_trigger_before_copy,
    )

    a_, b_, c_, bias_ = _make_compile_repr_tensors(
        dtype,
        c_dtype,
        has_bias,
        a_leading,
        b_leading,
        c_leading,
    )
    fake_stream = make_fake_stream()

    if push_mode:
        meta_ = from_dlpack(
            torch.zeros((PUSH_META_LEN,), dtype=torch.int64, device="cuda"),
            assumed_align=16,
        )
        compiled = cute_ext.compile(_bmm_push, gemm, a_, b_, c_, meta_, fake_stream)
    elif has_bias:
        compiled = cute_ext.compile(_bmm_bias, gemm, a_, b_, c_, bias_, fake_stream)
    else:
        compiled = cute_ext.compile(_bmm_no_bias, gemm, a_, b_, c_, fake_stream)

    _TGV_CUTE_EXT_COMPILE_CACHE[key] = compiled
    return compiled


# =====================================================================
# Tensor adaptation: PyTorch (M, N) -> cute view via the A↔B swap. Now
# auto-detects each tensor's stride-1 dim so callers can pass either
# K-major or MN-major a/b (and either M-contig or N-contig out).
# =====================================================================
def _to_cute_swap(
    a_pt: torch.Tensor,  # (..., M, K)
    b_pt: torch.Tensor,  # (..., K, N)
    out_pt: torch.Tensor,  # (..., M, N)
    bias_pt: Optional[torch.Tensor],  # (N,) or None — bias is per-output-feature
):
    """Build the cute.Tensors fed to ``_bmm_no_bias`` / ``_bmm_bias`` AND
    report the detected (a_leading, b_leading, c_leading) tuple so the
    caller can fetch the matching compiled binary.

    Internally swaps A↔B so the kernel writes its (kernel-M)-contiguous
    output into the N-axis of a row-major PyTorch tensor — same trick as
    the C++ runner's ``gemm_fn(b.t(), a.t(), …)``. After swap, the
    kernel's M = PyTorch N and the kernel's N = PyTorch M; bias broadcast
    over the kernel's M axis (= PyTorch N) correctly mirrors the
    per-output-feature semantics.

    The kernel itself (`make_trivial_tiled_mma` + `mma_major_mode()`)
    adapts to whichever axis is the stride-1 one in each operand, so we
    just need to make sure ``mark_layout_dynamic`` pins the right dim.
    """
    # Lift to 3D batched shape with a leading L=1 axis when called from mm.
    if a_pt.dim() == 2:
        a_pt = a_pt.unsqueeze(0)
        b_pt = b_pt.unsqueeze(0)
        out_pt = out_pt.unsqueeze(0)

    # Swap: feed b.transpose(-2,-1) as A_ce (shape (L, N_pt, K)),
    # a.transpose(-2,-1) as B_ce (shape (L, K, M_pt)), and
    # out.transpose(-2,-1) as C_ce (shape (L, N_pt, M_pt)).
    a_swap = b_pt.transpose(-2, -1)
    b_swap = a_pt.transpose(-2, -1)
    c_swap = out_pt.transpose(-2, -1)

    a_leading = _detect_leading_dim(a_swap)
    b_leading = _detect_leading_dim(b_swap)
    c_leading = _detect_leading_dim(c_swap)

    a_ = from_dlpack(a_swap, assumed_align=32).mark_layout_dynamic(
        leading_dim=a_leading
    )
    b_ = from_dlpack(b_swap, assumed_align=32).mark_layout_dynamic(
        leading_dim=b_leading
    )
    c_ = from_dlpack(c_swap, assumed_align=32).mark_layout_dynamic(
        leading_dim=c_leading
    )

    layout = (a_leading, b_leading, c_leading)

    if bias_pt is None:
        return a_, b_, c_, None, layout

    # bias is per-output-feature in PyTorch terms. After the A↔B swap the
    # kernel's M-axis maps to PyTorch's N-axis, which is exactly what the
    # bias indexes — so this is a direct (0, 1, 0) broadcast.
    L = c_swap.shape[0]
    M_ce = c_swap.shape[1]  # == PyTorch N
    N_ce = c_swap.shape[2]  # == PyTorch M
    bias_3d = bias_pt.as_strided(size=(L, M_ce, N_ce), stride=(0, 1, 0))
    bias_ = from_dlpack(bias_3d.detach(), assumed_align=2).mark_layout_dynamic(
        leading_dim=1
    )
    return a_, b_, c_, bias_, layout


def _resolve_tactic(tactic: int) -> Tuple[int, int, int, bool]:
    """Return (cta_m, cta_n, num_ab_stage, use_2cta) for the given tactic id."""
    if tactic < 0:
        tactic = _TGV_CUTE_EXT_DEFAULT_TACTIC
    if tactic >= len(_TGV_CUTE_EXT_TACTIC_CONFIGS):
        raise ValueError(
            f"TGV cute_ext tactic {tactic} out of range [0, {len(_TGV_CUTE_EXT_TACTIC_CONFIGS)})."
        )
    return _TGV_CUTE_EXT_TACTIC_CONFIGS[tactic]


def _run_tgv(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
    pdl: bool,
    tactic: int,
) -> torch.Tensor:
    """Dispatch one TGV GEMM: a (M, K) @ b (K, N) -> out (M, N), bias (N,)."""
    cta_m, cta_n, num_ab_stage, use_2cta = _resolve_tactic(tactic)
    has_bias = bias is not None

    a_, b_, c_, bias_, (a_leading, b_leading, c_leading) = _to_cute_swap(
        a,
        b,
        out,
        bias,
    )

    compiled = _get_compiled_cute_ext_kernel(
        dtype=a.dtype,
        c_dtype=out.dtype,
        cta_m=cta_m,
        cta_n=cta_n,
        cta_k=_TGV_CUTE_EXT_CTA_K,
        num_ab_stage=num_ab_stage,
        use_2cta=use_2cta,
        use_pdl=bool(pdl),
        has_bias=has_bias,
        a_leading=a_leading,
        b_leading=b_leading,
        c_leading=c_leading,
    )

    stream = cuda.CUstream(torch.cuda.current_stream(a.device).cuda_stream)

    if has_bias:
        compiled(a_, b_, c_, bias_, stream)
    else:
        compiled(a_, b_, c_, stream)
    return out


# (tactic, cta_m, cta_n, num_mma_ctas), smallest cta_n first.
_TGV_TACTIC_LADDER = (
    (18, 64, 16, 2),
    (21, 64, 32, 2),
    (23, 64, 64, 2),
    (24, 64, 128, 2),
    (12, 128, 16, 1),
    (26, 128, 32, 2),
    (27, 128, 64, 2),
)


def _grid_ctas(m: int, n: int, cta_m: int, cta_n: int, num_mma_ctas: int) -> int:
    # Kernel axes are swapped: kernel-M = pytorch N, kernel-N = pytorch M.
    cluster_m = cta_m * num_mma_ctas
    return num_mma_ctas * -(n // -cluster_m) * -(m // -cta_n)


def _pick_tactic(m: int, n: int, k: int) -> int:
    """Pick the first ladder tactic whose grid fits one wave of the SMs."""
    num_sms = get_device_core_count(torch.cuda.current_device())
    best, best_ctas = None, None
    for tactic, cta_m, cta_n, num_mma_ctas in _TGV_TACTIC_LADDER:
        ctas = _grid_ctas(m, n, cta_m, cta_n, num_mma_ctas)
        if ctas <= num_sms:
            return tactic
        if best_ctas is None or ctas < best_ctas:
            best, best_ctas = tactic, ctas
    return best


# Kimi-K3 per-rank dense-GEMM shapes (TP8). They sit in this heuristic's
# unmeasured region (hidden=7168 inputs fail k > 6144; o_proj-style k=1536
# fails k < 2048), but TGV wins 1.04-2.43x on every one of them on GB300
# (L2-defeating weight rotation + CUDA-graph timing; serving A/B confirmed
# e2e with GSM8K parity). kv_a (n=576) loses (0.84x) and stays out. Gated to
# small decode batches; larger m stays on the measured heuristic below.
_K3_TGV_WIN_SHAPES = frozenset(
    {
        (6144, 7168),  # KDA fused qkvg
        (6016, 7168),  # merged MoE front (gate_up | router | latent down)
        (7168, 1536),  # KDA / MLA o_proj
        (1536, 7168),  # MLA q_a / shared gate_up
        (2304, 1536),  # MLA q_b
        (3584, 7168),  # MoE latent down (unfused fallback)
        (7168, 3584),  # MoE latent up
        (7168, 768),  # shared down
    }
)


# MuseSpark TP4/TP8 projections using TGV on GB300; other M use cuBLAS.
_MUSE_SPARK_TGV_M_RANGES = {
    (4096, 8192): (1, 16),  # replicated pooler
    (8192, 4096): (1, 128),  # replicated unpooler
    (4608, 8192): (1, 256),  # TP4 qkvg
    (8192, 2048): (1, 128),  # TP4 o_proj
    (1024, 8192): (1, 256),  # TP4 pooler slice
    (2048, 4096): (1, 256),  # TP4 unpooler slice
    (2304, 8192): (40, 512),  # TP8 qkvg
    (8192, 1024): (1, 64),  # TP8 o_proj
    (512, 8192): (256, 512),  # TP8 pooler slice
    (1024, 4096): (1, 512),  # TP8 unpooler slice
}


def use_cutedsl_bf16_gemm(m: int, n: int, k: int) -> bool:
    """TGV-vs-cuBLAS (``F.linear``) decision, CUPTI-measured on B300 under CUDA
    graph capture (cold L2). Conservative: ties and unmeasured regions fall
    back to cuBLAS."""
    if m <= 0:
        # Empty batch: DP-attention idle groups run a 0-token dummy forward to
        # keep the mlp-sync lockstep. A 0-CTA TGV grid is a driver-level
        # CUDA_ERROR_INVALID_VALUE, so leave empty inputs to cuBLAS.
        return False
    if k % 8 != 0:  # TMA requires 16B-aligned rows
        return False
    m_range = _MUSE_SPARK_TGV_M_RANGES.get((n, k))
    if m_range is not None:
        return m_range[0] <= m <= m_range[1]
    if m <= 8 and (n, k) in _K3_TGV_WIN_SHAPES:
        return True
    if n < 1024 or k < 2048 or k > 6144:
        return False
    ragged = m % 16 != 0
    if n <= 1024:
        return m <= 512 and (m >= 48 or m % 16 >= 9)
    if n <= 2624:
        return m <= 128 and k >= 4096
    if n <= 4096:
        return m <= 64
    if k < 4096:
        return k >= 3072 and 25 <= m <= 48 and ragged
    if n <= 6144:
        if m <= 64:
            return m <= 32 or k >= 6144 or ragged
        return m <= 72 and k >= 6144 and ragged
    if n <= 8192:
        return m <= 48 or (m <= 63 and ragged)
    return n <= 12288 and k >= 6144 and (m <= 32 or (m <= 48 and ragged))


def _tgv_bf16_gemm_run(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    if not is_sm100_supported():
        raise RuntimeError("cutedsl_bf16_gemm requires an SM10x GPU")
    assert x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
    assert x.stride(-1) == 1, "x must be K-major [M, K]"
    assert weight.stride(-1) == 1, "weight must be K-major [N, K]"
    out = torch.empty(
        (x.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x.device
    )
    if x.shape[0] == 0:
        # Match cuBLAS/F.linear semantics for empty batches; a 0-CTA launch
        # would fail with CUDA_ERROR_INVALID_VALUE.
        return out
    return _run_tgv(
        x,
        weight.t(),
        bias,
        out,
        pdl=True,
        tactic=_pick_tactic(x.shape[0], weight.shape[0], weight.shape[1]),
    )


def _tgv_bf16_gemm_out_run(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> None:
    if not is_sm100_supported():
        raise RuntimeError("cutedsl_bf16_gemm requires an SM10x GPU")
    assert x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
    assert out.dtype in (torch.bfloat16, torch.float32) and out.device == x.device
    assert x.ndim == 2 and weight.ndim == 2 and out.ndim == 2
    assert x.stride(-1) == 1, "x must be K-major [M, K]"
    assert weight.stride(-1) == 1, "weight must be K-major [N, K]"
    assert out.is_contiguous() and out.shape == (x.shape[0], weight.shape[0])
    if x.shape[0] == 0:
        return None
    _run_tgv(
        x,
        weight.t(),
        bias,
        out,
        pdl=True,
        tactic=_pick_tactic(x.shape[0], weight.shape[0], weight.shape[1]),
    )
    return None


def _tgv_bf16_gemm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]))


def _tgv_bf16_gemm_out_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> None:
    return None


# ---- push epilogue entry points (this module registers no torch ops; SGLang's served module keeps its own) ----
PUSH_META_LEN = 24  # [0] counter [1] phase stride [2] rank [3] rows [4] owner chunk [5] out [6] ldc [7] ready flags
#                     [8, 8 + W) peer slot bases [16] consumer grid (ready flags to set)
#                     [17] copy source (residual) [18] copy destination [19] copy bytes [20] phase word (mode 5)
PUSH_META_PEERS = (
    8  # meta[8 + d] = destination d's slot base for this rank as source (phase 0)
)


def make_push_meta(
    counter_addr: int,
    phase_stride: int,
    rank: int,
    rows: int,
    chunk_rows: int,
    out: torch.Tensor,
    peer_slot_bases,
    ready_addr: int = 0,
    consumer_grid: int = 0,
    copy_src: int = 0,
    copy_dst: int = 0,
    copy_bytes: int = 0,
    phase_out: int = 0,
) -> torch.Tensor:
    """int64 argument block for the push epilogue (host-built once per buffer set; graph replays reuse it)."""
    vals = [0] * PUSH_META_LEN
    vals[0] = int(counter_addr)
    vals[1] = int(phase_stride)
    vals[2] = int(rank)
    vals[3] = int(rows)
    vals[4] = int(chunk_rows)
    vals[5] = int(out.data_ptr())
    vals[6] = int(out.stride(0))
    vals[7] = int(ready_addr)
    vals[16] = int(consumer_grid)
    vals[17] = int(copy_src)
    vals[18] = int(copy_dst)
    vals[19] = int(copy_bytes)
    vals[20] = int(phase_out)
    for d, base in enumerate(peer_slot_bases):
        vals[PUSH_META_PEERS + d] = int(base)
    t = torch.tensor(vals, dtype=torch.int64, device=out.device)
    _META_HOST[t.data_ptr()] = vals
    return t


_META_HOST: dict = {}  # meta device address -> its host values (graph replays reuse the device block unchanged)


def set_copy_source(meta: torch.Tensor, src: torch.Tensor) -> None:
    """Point a push_mode 5 / 6 block at another residual tensor (same size); keeps the host mirror in step."""
    host = _META_HOST[meta.data_ptr()]
    assert (
        src.dtype == torch.float32
        and src.is_contiguous()
        and src.numel() * 4 == host[19]
    )
    host[17] = int(src.data_ptr())
    meta[17] = int(src.data_ptr())


def check_copy_pair(
    meta: torch.Tensor, stream: torch.Tensor, phase_in: torch.Tensor
) -> None:
    """The copy-input consumer may only read what its producer copied: its `stream` must be the copy destination and
    its phase word the producer's phase output."""
    host = _META_HOST.get(meta.data_ptr())
    if host is None or stream.data_ptr() != host[18] or phase_in.data_ptr() != host[20]:
        raise ValueError(
            "copy-input consumer: stream / phase_in are not the producer's copy and phase word"
        )


def run_tgv(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    tactic: Optional[int] = None,
    pdl: bool = True,
    push_mode: int = 0,
    push_world: int = 1,
    meta: Optional[torch.Tensor] = None,
    _test_drop_restage_barrier: bool = False,
    _test_trigger_before_copy: bool = False,
) -> torch.Tensor:
    """out[M, N] = x[M, K] @ weight[N, K].T with this module's kernel; push_mode 1 / 2 also stores the tile into the
    peer slots described by ``meta`` (make_push_meta). tactic None = the served pick (_pick_tactic)."""
    assert (
        x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and out.dtype == torch.bfloat16
    )
    assert x.stride(-1) == 1 and weight.stride(-1) == 1 and out.is_contiguous()
    m, n = x.shape[0], weight.shape[0]
    if tactic is None:
        tactic = _pick_tactic(m, n, weight.shape[1])
    if m == 0:
        return out
    if not push_mode:
        return _run_tgv(x, weight.t(), None, out, pdl=pdl, tactic=tactic)
    cta_m, cta_n, num_ab_stage, use_2cta = _resolve_tactic(tactic)
    assert (
        meta is not None and meta.dtype == torch.int64 and meta.numel() == PUSH_META_LEN
    )
    if push_mode in (5, 6):
        if not pdl:
            raise ValueError("push_mode 5 / 6 needs a PDL launch")
        host = _META_HOST.get(meta.data_ptr())
        if (
            host is None
            or host[19] % 16
            or host[19] != m * n * 4
            or not host[17]
            or not host[18]
            or not host[20]
        ):
            raise ValueError(
                "push_mode 5 / 6: copy block must be M x N fp32 (16-byte multiple) with source, copy and phase"
            )
    assert n % cta_m == 0, "push epilogue needs whole feature tiles"
    a_, b_, c_, _, (a_leading, b_leading, c_leading) = _to_cute_swap(
        x, weight.t(), out, None
    )
    compiled = _get_compiled_cute_ext_kernel(
        dtype=x.dtype,
        c_dtype=out.dtype,
        cta_m=cta_m,
        cta_n=cta_n,
        cta_k=_TGV_CUTE_EXT_CTA_K,
        num_ab_stage=num_ab_stage,
        use_2cta=use_2cta,
        use_pdl=bool(pdl),
        has_bias=False,
        a_leading=a_leading,
        b_leading=b_leading,
        c_leading=c_leading,
        push_mode=push_mode,
        push_world=push_world,
        test_drop_restage_barrier=_test_drop_restage_barrier,
        test_trigger_before_copy=_test_trigger_before_copy,
    )
    meta_ = from_dlpack(meta, assumed_align=16)
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    compiled(a_, b_, c_, meta_, stream)
    return out
