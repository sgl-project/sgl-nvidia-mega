# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""SplitKRsGemm (default off): ``SplitKGemm``'s operand pipeline with a
reduce-scatter epilogue, for the MID KDA in-projection (TP16 rank: 3216 x 7168 BF16 weights,
65-128 token rows). Selected only by a 6-field ``K31MK_PROJ_TACTICS`` entry
``m.n.split.stages.owners.cta_k`` (projection.py); every served tactic keeps its kernel.

Why (Nsight Compute profiles): the served (64, 64, 2, 6) at 104-128
rows reads each weight byte twice (two token tiles) and the activations 51 times (187 MB through TMA);
one 128 x 128 tile per weight slice reads each weight byte once and the activations 26 times (~95 MB).
That tile at split 2 is 26 clusters of 2 (52 CTAs: too few to stream), and the all-to-one reduction of
``SplitKGemm`` cannot hold more ranks (split 6 at 128 x 128 needs a 327,680 B mailbox). Here the split
is 6 (26 clusters of 6) and the partials are
reduce-scattered: each epilogue thread holds one accumulator row (its TMEM lane, ``n`` FP32 token
columns) and sends it to the owner of its ``128 / owners``-row share; owners sum the ``split`` slots in
rank order, ((p0 + p1) + p2) + ..., round once to BF16 and store their rows.

Pipeline (unchanged ``SplitKGemm`` warps): warp 0 issues the weight boxes (evict_first) before the
PDL wait, warp 1 the activation tiles after it, warp 2 the MMAs from shared memory, one ring of
``stages`` (A + B) stages; ``fenced_trigger`` as ``SplitKGemm``. K splits UNEVENLY over the ranks when
``K / cta_k`` is not a multiple of ``split`` (rank r < (K / cta_k) % split takes one more tile, as
TM64): K 7168 at cta_k 128 = 56 tiles -> 10, 10, 9, 9, 9, 9.

Synchronization (sync forms of the served kernels, no new protocol): one cluster barrier after the
mbarrier inits (``SplitKGemm``), so every peer's ``bar_reduce`` exists before any DSMEM store; peers'
16-byte ``st.async`` partials complete_tx on the owner's ``bar_reduce``; the owner arms it at CLUSTER
scope (K-23, as ``SplitKGemm`` and TM64 MBOX: no barrier orders the arm before the peers' stores) and
waits ``try_wait.acquire.cluster`` (``SplitKGemm._wait_cluster``); its own slot is written with plain
``st.shared`` and ordered for the other epilogue threads by named barrier 3 (TM64). The mailbox and the
output tile have their own shared memory (written once per launch), so no peer store can land in a
buffer this CTA's MMAs or TMA loads still use (TM64 MBOX form).

Numerics: split 6 partials summed in rank order vs the served split 2: another FP32 order of the same
products (KL class; gate = FP64 reference, as TM64's KDA17.1)."""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Int32
from cutlass.cute import experimental as cute_ext

from ..common.primitives import at, bits_float, float_bits, named_barrier, push_float, store_words
from .splitk import (
    large_smem_alloc,
    SplitKGemm,
    SplitKTactic,
    _arm_cluster,
    _fence_proxy_async_global,
    _fence_tcgen05,
    _fenced_launch_dependents,
    _smem,
    _trigger_barrier,
    _wait_cluster,
)
from .splitk_tmem64 import _ld_shared_v4, _load_row, _st_shared_bf16, _st_shared_v4

__all__ = ["RS_N_TILES", "SplitKRsGemm", "rs_check"]

RS_N_TILES = (16, 32, 64, 96, 112, 128)  # token columns per tile (the accumulator's TMEM columns); 16 / 32: c=8 rows (v3); 112 (v4)
EPI_BARRIER = 3  # named barrier of the 128 epilogue threads (TM64's id; SplitKGemm's trigger barrier is 4)


def rs_layout(n, split, stages, owners, cta_k):
    """(shared memory of one CTA in the kernel's allocation order, output tile inside the mailbox): the operand ring (1024-aligned
    A then B stages), the barriers, the owners' mailbox [split][n / 4 quads][share rows][4] FP32 (1024-aligned) and the owner's
    BF16 output tile [n tokens][share rows] (128-aligned) after it - or, when that last tile alone would pass the oversized
    allocation (v4, deep rings), inside the mailbox, written only after every owner thread has summed it (one more named barrier).
    The launch's FX-4 check (projection.smem_overflow) stays the authority on the DSL's own count."""
    share = 128 // owners
    up = lambda x, a: -(-x // a) * a
    ring = stages * (128 + n) * cta_k * 2
    mailbox_end = up(ring + (2 * stages + 3) * 8 + 4, 1024) + split * share * n * 4
    separate = up(mailbox_end, 128) + n * share * 2
    if separate <= large_smem_alloc():
        return separate, False
    return mailbox_end, True


def rs_smem_bytes(n, split, stages, owners, cta_k):
    return rs_layout(n, split, stages, owners, cta_k)[0]


def rs_check(tactic, k):
    """Raise unless ``tactic`` = (128, n, split, stages, owners, cta_k) is a valid RS tactic for reduction length
    ``k``: n in RS_N_TILES, 2 <= split <= 8, owners <= split with 128 / owners rows a multiple of 8 (the store's
    8-row words), whole quads per owner thread, whole cta_k tiles with at least one per rank and more than
    ``stages`` per rank (K-42 for the weight-warp trigger modes), and the oversized shared-memory cap."""
    m, n, split, stages, owners, cta_k = tactic
    ok = m == 128 and n in RS_N_TILES and 2 <= split <= 8 and cta_k in (64, 128) and k % cta_k == 0
    ok = ok and 1 <= owners <= split and 128 % owners == 0 and (128 // owners) % 8 == 0
    ok = ok and (n * (128 // owners) // 128) % 4 == 0 and stages >= 2
    ok = ok and (k // cta_k) // split > stages
    ok = ok and rs_layout(n, split, stages, owners, cta_k)[0] <= large_smem_alloc()
    if not ok:
        raise ValueError(f"RS tactic {tuple(tactic)} is invalid for K {k}")


class SplitKRsGemm(SplitKGemm):
    """``tactic`` (128, n, split, stages) plus ``owners`` and ``cta_k``; ``k``: the reduction length. Arguments as
    ``SplitKGemm`` (``bias`` unused). Always launches with the oversized shared-memory allocation (the caller sets
    the function's oversized mode, projection.set_oversized)."""

    def __init__(
        self, *, tactic: SplitKTactic, k: int, owners: int = 4, cta_k: int = 128, fenced_trigger: int = 0,
        proxy_fences: bool = False,
    ):  # fmt: skip
        super().__init__(tactic=tactic, fenced_trigger=fenced_trigger, proxy_fences=proxy_fences, oversized=True)
        rs_check((tactic.mma_m, tactic.mma_n, tactic.split_k, tactic.ab_stages, owners, cta_k), k)
        self.cta_k = cta_k
        self.n = tactic.mma_n
        self.owners, self.share = owners, 128 // owners
        self.k_total = k // cta_k
        self.base_tiles, self.extra = divmod(self.k_total, self.split_k)
        self.quads = self.n // 4
        self.mailbox_bytes = self.split_k * self.share * self.n * 4
        self.tile_bytes = self.n * self.share * 2
        self.slot_bytes = self.quads * self.share * 16  # one rank's partial of one owner's share
        self.tile_alias = rs_layout(self.n, self.split_k, tactic.ab_stages, owners, cta_k)[1]  # v4: output tile inside the mailbox

    @cute.experimental.kernel
    def kernel(
        self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, mBias: cute.Tensor,
        a_map: cutlass.Int64,
    ):  # fmt: skip
        """A (M, K, L) and B (N, K, L) K-major, C (M, N, L) M-major (BF16); bias unused."""
        stages = self.num_ab_stage
        ab_dtype = mA.element_type
        majors = (utils.LayoutEnum.from_tensor(t).mma_major_mode() for t in (mA, mB))
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype, ab_dtype, *majors, self.acc_dtype, self.cta_group,
            self.mma_tiler_mn,
        )  # fmt: skip

        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        bidx, by, l_idx = cute.arch.block_idx()
        split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        n_idx = by // self.split_k
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        shape = (tiled_mma, mnk_tiler, ab_dtype, stages)
        sA = _smem(ab_dtype, None, 1024, sm100_utils.make_smem_layout_a(*shape))
        sB = _smem(ab_dtype, None, 1024, sm100_utils.make_smem_layout_b(*shape))
        acc_layout = cute_ext.make_tmem_layout_acc(
            tiled_mma, self.mma_tiler_mn, acc_stage=1
        )
        bar_full = _smem(cutlass.Int64, stages, 8).iterator
        bar_empty = _smem(cutlass.Int64, stages, 8).iterator
        bar_mma_epilog = _smem(cutlass.Int64, 1, 8).iterator
        bar_tmem_alloc = _smem(cutlass.Int64, 1, 8).iterator
        bar_reduce = _smem(cutlass.Int64, 1, 8).iterator
        tmem_base_ptr = _smem(cutlass.Int32, 1, 4).iterator
        # Own regions (written once per launch): the owners' mailbox and the owner's BF16 output tile.
        mbox = Int32(_smem(cutlass.Uint8, self.mailbox_bytes, 1024).iterator.toint())
        if cutlass.const_expr(self.tile_alias):
            tile = mbox  # v4 deep rings: the owner's tile reuses its own mailbox after the sums (reduce_scatter)
        else:
            tile = Int32(_smem(cutlass.Uint8, self.tile_bytes, 128).iterator.toint())

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, 2)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)
                # Owner arrival plus the peers' transaction-byte credits.
                cute.arch.mbarrier_init(bar_reduce, 1)

        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()  # peers' barriers before their stores

        # This rank's K tiles: [first, first + tiles); ranks below ``extra`` take one more (TM64's split).
        base_tiles = self.base_tiles
        tiles = base_tiles + Int32(split_rank < self.extra)
        over = Int32(split_rank > self.extra) * (split_rank - self.extra)
        first = split_rank * (base_tiles + 1) - over

        cute.arch.cluster_wait()

        # Warp 3 is idle (or the fix A mode 2 trigger); warps 4-7 run the epilogue.
        if warp_idx == 0:
            self.weight_warp(bar_full, bar_empty, sA, a_map, bidx * self.cta_m, first, tiles)
        elif warp_idx == 1:
            b_tile = cute.local_tile(mB, (self.cta_n, self.cta_k), (n_idx, None, l_idx))
            b_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")
            self.activation_warp(bar_full, bar_empty, b_tile, sB, b_map, first, tiles)
        elif warp_idx == 2:
            k_blocks = self.cta_k // cute.size(tiled_mma.shape_mnk, mode=[2])
            self.mma_warp(
                bar_full, bar_empty, bar_mma_epilog, bar_tmem_alloc, tiled_mma, sA, sB,
                tmem_base_ptr, acc_layout, k_blocks, tiles,
            )  # fmt: skip
        elif warp_idx >= 4:
            self.reduce_scatter(
                cute.arch.thread_idx()[0] - 128, bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr,
                mbox, tile, bar_reduce, split_rank, mC, bidx * self.cta_m, n_idx * self.cta_n, l_idx,
            )  # fmt: skip
        if cutlass.const_expr(self.fenced_trigger == 2):
            if warp_idx == 3:
                cute.arch.griddepcontrol_wait()
                _trigger_barrier(False)  # K-42: after the MMA warp's TMEM alloc
                _fenced_launch_dependents()

    @cute.experimental.jit
    def reduce_scatter(
        self, epi_tid: cutlass.Int32, bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr,
        mb: Int32, tile: Int32, bar_reduce, split_rank: cutlass.Int32, mC: cute.Tensor,
        row0: cutlass.Int32, col0: cutlass.Int32, l_idx: cutlass.Int32,
    ):  # fmt: skip
        """Accumulator row ``epi_tid`` (TMEM lane epi_tid, n FP32 token columns) -> slot ``split_rank`` of its
        owner's mailbox [split][quads][share rows][4]; owners sum the slots in rank order and ``finish``."""
        # Wait until the MMA warp publishes the TMEM base (served protocol of bar_tmem_alloc, 160 arrivals).
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        _fence_tcgen05()
        base = Int32(cute.make_tensor(tmem_base_ptr, cute.make_layout(1))[0])
        quadrant = epi_tid // 32  # warps 4-7 read TMEM lanes 32 (warp % 4) ..
        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        _fence_tcgen05()
        values = _load_row(base + (quadrant * 32 << 16), self.n)
        _fence_tcgen05(before=True)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)  # TMEM release (the MMA warp deallocates after phase 1)

        split, share, owners = self.split_k, self.share, self.owners
        quads: cutlass.Constexpr = self.quads
        if (epi_tid == 0) & (split_rank < owners):
            # K-23: no barrier orders this arm before the peers' st.async complete_tx -> arm at cluster scope.
            _arm_cluster(bar_reduce, (split - 1) * self.slot_bytes)
        owner, r = epi_tid // share, epi_tid % share
        slot = mb + (split_rank * quads * share + r) * 16  # at the owner's address (same layout in every CTA)
        mailbox = cute.make_ptr(Float32, slot, cute.AddressSpace.smem, assumed_align=16)
        if owner == split_rank:
            for i in cutlass.range_constexpr(quads):
                bits = tuple(float_bits(values[4 * i + j]) for j in range(4))
                _st_shared_v4(slot + i * share * 16, bits)
        else:
            for i in cutlass.range_constexpr(quads):
                quad = tuple(values[4 * i + j] for j in range(4))
                push_float(mailbox + i * share * 4, quad, bar_reduce, owner)
        if split_rank < owners:
            named_barrier(EPI_BARRIER, 128)  # this CTA's own slot
            _wait_cluster(bar_reduce, Int32(0))
            # thread -> (row rr of the share, tokens width * g .. width * (g + 1) - 1)
            width: cutlass.Constexpr = self.n * share // 128
            rr, g = epi_tid % share, epi_tid // share
            totals = cute.make_rmem_tensor(width, Float32)
            ranks: cutlass.Constexpr = split
            for s in cutlass.range_constexpr(ranks):
                for v in cutlass.range_constexpr(width // 4):
                    offset = ((s * quads + width // 4 * g + v) * share + rr) * 16
                    words = _ld_shared_v4(mb + offset)
                    for j in cutlass.range_constexpr(4):
                        part = bits_float(words[j])
                        if cutlass.const_expr(s == 0):
                            totals[4 * v + j] = part
                        else:
                            totals[4 * v + j] = totals[4 * v + j] + part
            if cutlass.const_expr(self.tile_alias):
                # v4: every epilogue thread has read its mailbox slots (the phase completed, so no peer writes it again)
                # before any thread writes the tile into the same bytes.
                named_barrier(EPI_BARRIER, 128)
            self.finish(epi_tid, totals, tile, mC, row0 + split_rank * share, col0, l_idx)
            if cutlass.const_expr(self.proxy_fences):
                _fence_proxy_async_global()  # D3: these generic stores -> successors' TMA reads

    @cute.experimental.jit
    def finish(
        self, epi_tid: cutlass.Int32, totals: cute.Tensor, tile: Int32, mC: cute.Tensor,
        row0: cutlass.Int32, col0: cutlass.Int32, l_idx: cutlass.Int32,
    ):  # fmt: skip
        """An owner's rows [row0, row0 + share) of tokens col0 ..: thread (row rr, tokens width g ..) holds
        ``totals``; rounded to BF16, transposed through the shared-memory ``tile`` [n tokens][share rows] and
        stored as 16-byte (8-row) words of C (M-major); rows / tokens past C's end are not stored (C's rows are
        a multiple of 8, projection.py checks)."""
        share = self.share
        rr, g = epi_tid % share, epi_tid // share
        width: cutlass.Constexpr = self.n * share // 128  # tokens per thread
        for t in cutlass.range_constexpr(width):
            _st_shared_bf16(tile + ((width * g + t) * share + rr) * 2, totals[t])
        named_barrier(EPI_BARRIER, 128)
        rows, columns = mC.layout.shape[0], mC.layout.shape[1]
        words_per_token: cutlass.Constexpr = share // 8
        words: cutlass.Constexpr = self.n * words_per_token  # 16-byte words of the owner's tile (< 128 at n 16: some threads idle)
        for i in cutlass.range_constexpr((words + 127) // 128):
            idx = epi_tid + i * 128
            token, c = idx // words_per_token, idx % words_per_token
            inside = (col0 + token < columns) & (row0 + 8 * c < rows)
            if cutlass.const_expr(words % 128 != 0):
                inside = inside & (idx < words)
            if inside:
                packet = _ld_shared_v4(tile + (token * share + 8 * c) * 2)
                store_words(at(mC, row0 + 8 * c, col0 + token, l_idx), packet)
