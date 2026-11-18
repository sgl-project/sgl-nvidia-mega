# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Draft conv projection with the DFlash grouped conv's prepare side folded into the
split-K epilogue: ``ConvSplitKGemm`` subclasses the vendored
``SplitKGemm`` (whose module, class and compile cache stay untouched, so the target's
projections are byte-identical) and differs in three places:

- The projection weight rows are permuted (``conv_row_order``): δ column 4g + 2·side + tap
  holds group g's coefficient, so a 64-row M tile owns 16 whole groups (256 channels).
- After the owner rank stores its δ tile, its epilogue warps apply side 0 to the tile's
  channels for the block's rows: x'[t] = (base0 + δ00) x[t] + (base1 + δ01) x[t - 1]
  (row t - 1 of the same 8-row block), and store x' (``x_out``). The ConvPrepare kernel
  and its dependent phase disappear.
- The PDL trigger moves from the weight warp to after the activation warp's wait.
  ConvPrepare used to be the chain's residual fence (it triggered only after its wait,
  and the next hop's consumer prefetches the residual before its own wait); triggering
  after the wait keeps that fence here.

``__call__`` / ``kernel`` / ``weight_warp`` are SplitKGemm's with those edits."""

import os
from functools import cache

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Float32
from cutlass.cute import experimental as cute_ext
from cutlass.cute.runtime import make_fake_tensor

from ..k3.common.host import architecture
from ..k3.common.host import stream as current_stream
from ..k3.common.primitives import (
    at,
    load_f32x4,
    load_vector,
    load_words,
    named_barrier,
    pack_pair,
    store_words,
    unpack_bf16,
)
from ..k3.gemm.projection import matrix_view, weight_map
from ..k3.gemm.splitk import (
    ATOM_K,
    OWNER_RANK,
    SplitKGemm,
    SplitKTactic,
    _acquire_tensor_map,
    _smem,
    _tma_load_evict_first,
)

GROUP, GROUPS, TAPS, BLOCK = 16, 448, 2, 8
CONV_BARRIER = 3  # named barrier of the 128 epilogue threads (SplitKGemm uses none)


def conv_row_order(device=None):
    """Old δ row of each new row: new 4g + 2s + t <- old (2s + t) * 448 + g."""
    g = torch.arange(GROUPS).repeat_interleave(4)
    k = torch.arange(4).repeat(GROUPS)
    return ((k * GROUPS) + g).to(device)


class ConvSplitKGemm(SplitKGemm):
    def __init__(self, *, tactic: SplitKTactic, early_trigger: bool = False):
        """``early_trigger``: SplitKGemm's weight-warp trigger, which drops the residual
        fence. Negative control of the race test only."""
        super().__init__(tactic=tactic)
        self.early_trigger = early_trigger
        if tactic.mma_n != BLOCK or tactic.mma_m % 4:
            raise ValueError(
                "the folded conv wants N tile = one 8-row block, M % 4 == 0"
            )

    @cute.experimental.jit
    def __call__(
        self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, bias: cute.Tensor,
        a_map: cutlass.Int64, base: cute.Tensor, x_out: cute.Tensor,
        stream: _cuda.CUstream,
    ):  # fmt: skip
        # Grid-y packs output-N tile and cluster rank.
        self.kernel(a, b, c, bias, a_map, base, x_out).launch(
            grid=(
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(utils.get_smem_capacity_in_bytes("sm_100")),
            stream=stream,
            use_pdl=True,
        )

    @cute.experimental.kernel
    def kernel(
        self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, mBias: cute.Tensor,
        a_map: cutlass.Int64, mBase: cute.Tensor, mOut: cute.Tensor,
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
            tile = cute.local_tile(mC, (self.cta_m, self.cta_n), (bidx, n_idx, l_idx))
            self.epilog_warp(
                bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr, acc_layout, tile,
                cute.arch.thread_idx()[0] - 128, mC.element_type,
                utils.LayoutEnum.from_tensor(mC), mailbox, bar_reduce, split_rank,
            )  # fmt: skip
            if split_rank == OWNER_RANK:
                self.conv_epilogue(mB, mC, mBase, mOut, bidx, n_idx, l_idx)

    @cute.experimental.jit
    def weight_warp(
        self, bar_full, bar_empty, s_tile: cute.Tensor, a_map: cutlass.Int64,
        row0: cutlass.Int32, k_tile_start: cutlass.Int32, k_tile_count: cutlass.Int32,
    ):  # fmt: skip
        """A: two evict_first boxes per K tile, before the PDL wait (weights). Unlike
        SplitKGemm, no PDL trigger here: see ``activation_warp``."""
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

        if cutlass.const_expr(self.early_trigger):
            cute.arch.griddepcontrol_launch_dependents()
        self.drain(bar_empty, empty_phase, k_tile_count)

    @cute.experimental.jit
    def activation_warp(
        self, bar_full, bar_empty, g_tile: cute.Tensor, s_tile: cute.Tensor,
        cta_v_map: cute.Layout, k_tile_start: cutlass.Int32,
        k_tile_count: cutlass.Int32,
    ):  # fmt: skip
        """SplitKGemm's, after this grid's PDL wait and the trigger (the residual fence)."""
        cute.arch.griddepcontrol_wait()
        # INVARIANT: the trigger must stay after griddepcontrol.wait. It is the residual
        # fence for the next DraftFused's pre-wait prefetch;
        # only the negative control (early_trigger) moves it.
        if cutlass.const_expr(not self.early_trigger):
            cute.arch.griddepcontrol_launch_dependents()
        SplitKGemm.activation_warp(
            self, bar_full, bar_empty, g_tile, s_tile, cta_v_map, k_tile_start,
            k_tile_count,
        )  # fmt: skip

    @cute.experimental.jit
    def conv_epilogue(
        self, mB: cute.Tensor, mC: cute.Tensor, mBase: cute.Tensor, mOut: cute.Tensor,
        bidx: cutlass.Int32, n_idx: cutlass.Int32, l_idx: cutlass.Int32,
    ):  # fmt: skip
        """Owner epilogue warps: side 0 of the tile's groups for the block's rows."""
        epi_tid = cute.arch.thread_idx()[0] - 128
        named_barrier(CONV_BARRIER, 128)  # the δ tile stores are visible
        groups = cutlass.const_expr(self.cta_m // 4)
        chunks = cutlass.const_expr(groups * 2)  # 8-channel halves of a group
        for e in cutlass.range(epi_tid, chunks * self.cta_n, 128, unroll=1):
            chunk, r = e % chunks, e // chunks
            g = bidx * groups + chunk // 2
            ch = g * GROUP + (chunk % 2) * 8
            row = n_idx * self.cta_n + r
            # Pointer loads (the experimental DSL has no scalar gmem element loads):
            # δ[4g], δ[4g + 1] are side 0's taps (M-major δ: adjacent); .cg sees the
            # tile the epilogue warps just stored (after the named barrier).
            delta = load_words(at(mC, 4 * g, row, l_idx), count=2, mutable=True)
            c0, c1 = unpack_bf16(delta[0])
            base0 = load_f32x4(at(mBase, 0, ch, 0)) + load_f32x4(
                at(mBase, 0, ch + 4, 0)
            )
            base1 = load_f32x4(at(mBase, 1, ch, 0)) + load_f32x4(
                at(mBase, 1, ch + 4, 0)
            )
            # Row 0 of a block has no previous row: weight 0 on a re-read of itself
            # (scalars only; no register tensors or branch-local writes).
            prev_row, keep = row - 1, Float32(1.0)
            if r % BLOCK == 0:
                prev_row, keep = row, Float32(0.0)
            current = load_vector(at(mB, row, ch, l_idx))
            previous = load_vector(at(mB, prev_row, ch, l_idx))
            out = []
            for j in cutlass.range_constexpr(4):
                x = unpack_bf16(current[j])
                p = unpack_bf16(previous[j])
                for h in cutlass.range_constexpr(2):
                    i = 2 * j + h
                    tap1 = (base1[i] + c1) * keep
                    out.append((base0[i] + c0) * x[h] + tap1 * p[h])
            packed = tuple(
                pack_pair(out[2 * j], out[2 * j + 1], policy=None) for j in range(4)
            )
            store_words(at(mOut, row, ch, 0), packed)


@cache
def compile_conv_projection(tactic, arch, early_trigger=False):
    def fake(leading, dtype=cutlass.BFloat16):
        shape = tuple(cute.sym_int() for _ in range(3))
        stride = (1, cute.sym_int(64), 0) if leading == 0 else (cute.sym_int(64), 1, 0)
        return make_fake_tensor(dtype, shape, stride, assumed_align=16)

    kernel = ConvSplitKGemm(tactic=SplitKTactic(*tactic), early_trigger=early_trigger)
    y = fake(0)
    args = (fake(1), fake(1), y, y, cutlass.Int64(0), fake(1, cutlass.Float32), fake(1))
    return cute.experimental.compile(
        kernel, *args, _fake_stream(), options=f"--gpu-arch={arch}"
    )


def _fake_stream():
    from cutlass.cute.runtime import make_fake_stream

    return make_fake_stream()


class PreparedConvProjection:
    """The replicated 7168 -> 1792 conv projection with the prepare side folded in:
    ``run(x, delta, base, x_out)`` writes δ (permuted columns) and x' = conv_side0(x)."""

    def __init__(self, weight, rows, tactic, *, early_trigger=False):
        if early_trigger and os.environ.get("K31DRAFT_FOLD_EARLY_TRIGGER") != "1":
            raise RuntimeError(
                "the folded conv projection's early trigger drops the residual fence; "
                "it is a negative control (K31DRAFT_FOLD_EARLY_TRIGGER=1) only"
            )
        order = conv_row_order(weight.device)
        self.weight = weight.detach().to(torch.bfloat16)[order].contiguous()
        self.device, self.rows, self.tactic = self.weight.device, rows, tuple(tactic)
        self.weight_view = matrix_view(self.weight)
        with torch.accelerator.device_index(self.device.index):
            self.map = weight_map(self.weight, self.tactic[0])
            arch = architecture(self.device)
            self.code = compile_conv_projection(self.tactic, arch, early_trigger)

    def run(self, x, delta, base, x_out):
        if x.shape[0] != self.rows:
            raise ValueError(f"conv projection rows {x.shape[0]} != {self.rows}")
        y = matrix_view(delta, output=True)
        args = (
            self.weight_view,
            matrix_view(x),
            y,
            y,
            cutlass.Int64(self.map.data_ptr()),
        )
        args += (matrix_view(base), matrix_view(x_out))
        self.code(*args, current_stream(self.device))
        return x_out
