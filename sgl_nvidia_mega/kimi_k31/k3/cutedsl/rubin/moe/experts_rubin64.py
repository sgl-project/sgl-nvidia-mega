# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Portions adapted from vLLM vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/moe/experts.py
# (https://github.com/vllm-project/vllm, commit 642c18d64c4cb6db3ded8fbe1786718cd4005fcb); the rest is NVIDIA-authored.
"""Rubin (cc 10.7) routed experts for TP16 verify steps of 64 tokens: the grouped
``ExpertsKernel`` engine (units of <= 8 tokens per expert, per-slot gathered MXFP8
activation slices, FC1 K-split partials, polled hidden images, FC2(B) stealing) with
``RubinExpertsKernel``'s weight path: the FP4 weights land packed (plain bytes, one
128 B x 128-row box per FC1 stage, the 96 data bytes of each w2 row per FC2 tile) in
16 KiB slots and feed Rubin's K64 block-scaled MMAs, and the w13 / w2 descriptors are
grid-constant parameters (no tensormap proxy fence in the prologue). Real verify
routes reach ~300-400 distinct experts per layer at 64 tokens, so the kernel is a
weight stream: both phases run at the die-oblivious HBM rate (the w2
rows' 32 padding bytes still move through the memory system although only their 96
data bytes land in SMEM). The FC2 tail is a pool (``fc2_pool``): the last FC2(B) jobs
of every CTA go into one pool for each die that free CTAs claim in grains, because the
CTAs' FC1 streaming rates differ measurably per call and the CTAs that finish FC1
late would otherwise set the end. Published modes stage
each lane's own BF16 half of the FC2 rows (no shuffles or branches), so the epilogue
keeps up with the MMAs."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int16, Int32, Int64, Uint8, Uint32, Uint64

from ..common.primitives import (
    alloc,
    asm,
    bf16,
    bf16x2,
    bits_float,
    commit,
    copy_evict,
    copy_g2s,
    fence_tmem,
    fma,
    load_tma3,
    load_tmem,
    load_volatile,
    load_words,
    multicast_words,
    named_barrier,
    pack_pair,
    partial_bits,
    release_tmem,
    shared_address,
    sleep_ns,
    smem_desc,
    store_relaxed,
    store_words,
    thread,
    view,
)
from ..common.tensormap import GridMap, encode_map, map_address, prefetch_map



@asm()
def stage16_sib(*args, **kwargs):
    """W2SIB (``K31MK_MOE_RUBIN64_W2SIB``): ``stage16`` for an FC2 job whose weight scales sit in the slot's w2 box
    (rows 0-63, bytes 96-111, 128B swizzle; descriptor = the slot's A descriptor + 6): the second 512 B half (rows 32-63)
    starts 4096 B on (descriptor + 256), not 512 B on (+ 32). The same bytes reach the same TMEM columns."""
    import inspect

    fn = inspect.unwrap(stage16)
    assert fn is not stage16, 'stage16 has no __wrapped__ PTX builder'
    code, constraints, values = fn(*args, **kwargs)
    assert code.count('add.s64 as1,$4,32;') == 1, 'stage16 second-half scale offset not found'
    return code.replace('add.s64 as1,$4,32;', 'add.s64 as1,$4,256;'), constraints, values
from .experts import PLAN_XSF, ExpertsKernel, claim, swizzled
from .experts_rubin import (
    DESCRIPTOR_LIMIT,
    RubinExpertsKernel,
    die_words,
    nth_set,
    sm_id,
    stage16,
    store_shared_if,
)

shuffle = cute.arch.shuffle_sync


@asm(Uint32, pure=True)
def stage_bits(value):
    """``value`` as BF16 with ``store_stage_half``'s NaN canonicalization, in the low
    half of the word."""
    code = "{ .reg .b32 w,bits; .reg .pred p; cvt.rn.bf16x2.f32 w,0f00000000,$1; "
    code += "and.b32 bits,w,0x7fff; setp.gt.u32 p,bits,0x7f80; @p mov.b32 w,0x7fc0; "
    code += "and.b32 $0,w,0xffff; }"
    return code, "=r,f", (Float32(value),)


shuffle_bfly, shuffle_up = (cute.arch.shuffle_sync_bfly, cute.arch.shuffle_sync_up)


@asm(Uint64)
def census_add(pointer, increment):
    """Relaxed GPU-scope ``atom.add.u64`` (the die census): the value before."""
    return (
        "atom.relaxed.gpu.global.add.u64 $0,[$1],$2;",
        "=l,l,l,~{memory}",
        (pointer, Uint64(increment)),
    )


@asm(Uint64)
def census_add_if(pointer, increment, active):
    """``census_add`` predicated on ``active`` != 0 (0 otherwise), in one asm block: no branch around the atomic
    (``RubinGroupedExpertsKernel.gcut``)."""
    return (
        "{ .reg .pred p; setp.ne.u32 p,$3,0; mov.u64 $0,0; @p atom.relaxed.gpu.global.add.u64 $0,[$1],$2; }",
        "=l,l,l,r,~{memory}",
        (pointer, Uint64(increment), Uint32(active)),
    )


@asm()
def prefetch_box_l2(descriptor, row, chunk):  # K31MK_MOE_RUBIN64_PRESTREAM
    """One packed box of ``descriptor`` (the one ``load_tma3`` reads at ``row`` / ``chunk``) into L2: a cache hint."""
    return "cp.async.bulk.prefetch.tensor.3d.L2.global.tile [$0, {0, $1, $2}];", "l,r,r", (descriptor, Int32(row), Int32(chunk))


@asm()
def prefetch_bytes_l2(src, size):  # K31MK_MOE_RUBIN64_PRESTREAM
    """``size`` contiguous bytes at ``src`` into L2: a cache hint."""
    return "cp.async.bulk.prefetch.L2.global [$0], $1;", "l,r", (src, Int32(size))


@asm()
def census_rearm(pointer):
    """The census word back to all-ones: one ``st.relaxed.gpu`` of 64 bits, the width of the adds, so the store and the
    adds overlap completely and are morally strong (PTX memory model) instead of a mixed-size pair of 32-bit halves."""
    return "st.relaxed.gpu.global.u64 [$0],$1;", "l,l,~{memory}", (pointer, Uint64(18446744073709551615))


@asm()
def require_below(value, bound):
    """Trap unless ``value`` < ``bound`` (unsigned): a census word that was not
    re-armed (see ``arrive``)."""
    return "{ .reg .pred p; setp.ge.u32 p,$0,$1; @p trap; }", "r,r", (value, bound)


@cute.jit
def match_arrival(bits, grid: cutlass.Constexpr[int], die, k):
    """The partition rank of the k-th CTA (arrival order) on ``die``: ``bits`` (one
    word per 32 ranks, bit set = the rank's middle weights sit on die 1); the k-th rank
    homed on its die, or past them (a spare) the other die's ranks from the end. The
    other die's CTAs take its ranks from the front, and a die's spares number exactly
    the other die's ranks without a CTA there: every rank runs once."""
    words: cutlass.Constexpr = (grid + 31) // 32
    h1 = Int32(0)
    for w in cutlass.range_constexpr(words):
        valid = grid - 32 * w
        mask = Uint32(4294967295) if valid >= 32 else Uint32((1 << valid) - 1)
        h1 += Int32(cute.arch.popc(bits[w] & mask))
    own, other = (grid - h1, h1)
    if die == 1:
        own, other = (h1, grid - h1)
    target, index = (die, k)
    if k >= own:
        target, index = (1 - die, other - 1 - (k - own))
    rank = Int32(-1)
    for w in cutlass.range_constexpr(words):
        valid = grid - 32 * w
        mask = Uint32(4294967295) if valid >= 32 else Uint32((1 << valid) - 1)
        word = bits[w]
        if target == 0:
            word = ~word
        word &= mask
        count = Int32(cute.arch.popc(word))
        if (rank < 0) & (index < count):
            rank = 32 * w + Int32(nth_set(word, index))
        elif rank < 0:
            index -= count
    return rank


# The die census (``dies``): one 64-bit word (in 16 B) past ``ExpertsGeometry``'s
# workspace (``experts_runtime.scratch``), armed all-ones like the workspace.
CENSUS_BYTES = 16
# FC2 tail pool: the producer lets the claim warp claim its first grain this many static
# jobs before its last (a claim and its hand-over take longer than a job; tuned on the target GPU).
POOL_AHEAD = 3
# ``K31MK_MOE_RUBIN64_POOL_FIX`` (default off): the first claim's lead with the fix. A claim takes longer, from
# issue to the posted entry at 24-32 rows, than 3 jobs take; 6 cover it (measured on one GPU).
POOL_AHEAD_FIX = 6


class RubinGroupedExpertsKernel(ExpertsKernel):
    """TP16 capacity-64 grouped ``ExpertsKernel`` on packed weights and K64 MMAs.
    ``stages``: 16 KiB weight slots (each with its 2 KiB activation slice; 7 beat
    6 and 8-10: more bytes in flight only queue); ``publish``: as ``ExpertsKernel``.
    Arguments as ``ExpertsKernel`` plus the w13 / w2 addresses before the stream.
    Numerics: the MMAs sum K64 instead of K32 groups (same E2M1 / MXFP8 / UE8M0 inputs
    and FP32 accumulators, a different grouping inside the MMA).
    ``static_groups``: ``ExpertsKernel.group_sync``'s constant barrier ids."""

    # K-19 trigger fence (``K31MK_K19_FENCE`` site rubin64, default off): fence.acq_rel.gpu right before
    # the trigger that follows griddepcontrol.wait. Set per instance by ``experts_runtime.compile_rubin64``.
    k19_fence = False
    # FC2 tail pool fix (``K31MK_MOE_RUBIN64_POOL_FIX``, default off): the pool's jobs take the static path's accumulator
    # acquire-ahead (two waits per eight jobs instead of one per job) and the first claim goes out ``POOL_AHEAD_FIX`` static jobs before the
    # end. Same MMAs, accumulators and job order, so outputs are bitwise the knob-off engine's. Set per instance by ``compile_rubin64``.
    pool_fix = False
    # Grouping cuts (``K31MK_MOE_RUBIN64_GCUT``, default off; die-matched engines only): the die-census ticket as one
    # predicated atomic (``census_add_if``; no branch around it, so warp 2 waits for the value at its use in ``arrive``, not at the branch
    # end) and each list's matched rank's own thread publishes its partition ranges (``home_bits_gcut`` -> misc 2928; ``published_ranges``
    # reads them instead of four ``partition`` calls). Same atomic, ranks, ranges, owners and partial-sum order: outputs are bitwise the
    # knob-off engine's. Set per instance by ``compile_rubin64``.
    gcut = False
    # Pre-stream L2 warm-up (``K31MK_MOE_RUBIN64_PRESTREAM``, default 0 = off; die-matched engines only): FC1 boxes per
    # routed id that the grouping CTAs L2-prefetch right after the ids land (``group``). Set per instance by ``compile_rubin64``.
    pst_nb = 0
    # FC2 pair loads (``K31MK_MOE_FC2_PAIRLOAD``, wide engines only, default off): two consecutive static
    # FC2 jobs of one unit in two adjacent ring slots load as ONE 256-row w2 box and ONE 2 KiB scale copy, both completing
    # on the first slot's full barrier; the second slot's full barrier gets a plain 33-count arrive (its data is covered by
    # the first barrier, which the MMA warp waits on first: it consumes the jobs in order). Set per instance by the wide
    # engines' constructors; ``encode_maps`` then adds the 256-row w2 map.
    fc2_pairload = False
    # Transposed local FC2 rows (``K31MK_MOE_RUBIN64_LEAN_LOCAL``, default off): the non-published (local rows)
    # FC2 epilogue writes each item through its warp's transpose buffer (``off_fc2s``) and one 16-byte store per lane, as the
    # published modes' lean path does, instead of eight 4-byte stores per lane. Set per instance by ``compile_rubin64``.
    lean_local = False

    def __init__(
        self, experts, grid, slots, publish=False, stages=7, pool=(12, 6),
        dies=(), static_groups=False, w2_row_bytes=128,
    ):  # fmt: skip
        super().__init__(16, 64, experts, grid, publish, True, False, True, slots)
        self.static_groups = static_groups
        # w2_row_bytes (K31MK_MOE_RUBIN64_W2_DEPAD): the global row stride of the w2 bytes. 128 = the
        # host layout (96 data bytes + 32 zero padding bytes per row); 96 = a de-padded copy (``depad_w2``). The
        # FC2 box is the same 96 B x 128 rows either way, so the same bytes land in SMEM; only the stride of the
        # descriptor and the die census's row address change.
        # 112 (K31MK_MOE_RUBIN64_W2SIB): FC2 records, the live 96 bytes + 16 B of the tile's scales per row
        # (rows 0-63): one 112 B x 128-row box per FC2 job carries the weights and their scales.
        assert w2_row_bytes in (96, 112, 128), w2_row_bytes
        self.w2_row_bytes = w2_row_bytes
        self.w2sib = w2_row_bytes == 112
        # FC2 tail pool (``fc2_pool``, ``claims``): (zone, grain) = the jobs at the end
        # of every FC2(B) range that go into the pool, and the jobs per claim. A claim
        # and a new unit's hidden image must hide under one grain,
        # so grain >= (claim + image) / job time (job time follows the w2 tile's K:
        # tuned on the target GPU); the zone covers the late CTAs' excess (a fraction of a CTA's FC1
        # time), in whole grains. One GPU, real calls: (12, 6) and
        # (8, 4) measured fastest, (12, 3) and (16, 4) less (per-claim and per-unit
        # costs). ``ExpertsKernel.fc2_steal`` (a claim per victim) lost.
        self.pool_zone, self.pool_grain = pool
        self.grain, self.zone, self.victims = (pool[1], 0, 0)
        # Published modes: every lane stages its own row's BF16 half of each pair's
        # FC2 row (``fc2``) instead of ``pack_pair`` and four gathering shuffles per
        # token (the same words at the same addresses): the staging kept a measurable backlog
        # behind the MMAs at 64 tokens.
        self.lean = bool(publish)
        # dies: the die of every smid (``common.dies.sm_domains``). The routed w13 and
        # w2 sit in allocation-granularity chunks alternating die 0 / 1 (``experts_runtime.place``):
        # each CTA counts itself into the die census after its dependency wait, and the
        # k-th CTA of die d (in census order) runs the k-th FC1(A) / FC1(B) / FC2(A) /
        # FC2(B) partition rank whose middle stage sits on die d (spares take the other
        # die's ranks from the end). Ranks, ranges, owners and partial-sum order are
        # those of the die-oblivious partition: the outputs are the same.
        self.dies = tuple(dies)
        if dies:
            self.die_bits = die_words(dies)
            self.ws_census = (self.workspace_bytes + 15) // 16 * 16
            self.workspace_bytes = self.ws_census + CENSUS_BYTES
            assert grid <= 256
        # RubinExpertsKernel.encode_maps' packed maps.
        self.packed, self.parts, self.grid_maps = True, 1, True
        self.stages = stages
        self.slot_bytes = 16384
        self.off_sfa = stages * self.slot_bytes
        self.off_x = self.off_sfa + stages * 1024
        self.off_h = self.off_x + stages * 2048
        self.off_xsf = self.off_h + 2 * self.h_image_bytes
        self.off_hsf = self.off_xsf + stages * 256 + 384
        self.off_misc = self.off_hsf + 2 * self.h_scale_bytes
        assert self.off_misc <= DESCRIPTOR_LIMIT
        # The unit tables past misc, where they outlive the grouping (the producer
        # streams into the ring while the other warps still rank the pairs, and every
        # warp reads the tables by global unit index; no per-CTA windows): experts
        # [1024] Int32, rows [1024, 8] Int16, then the grouping's masks [896, 2],
        # bases [896], warp totals [12] and a copy of the routes [1024].
        self.off_tables = (self.off_misc + 5120 + 127) // 128 * 128
        self.off_masks = self.off_tables + 4096 + 16384
        self.off_bases = self.off_masks + 8 * experts
        self.off_totals = self.off_bases + 4 * experts
        self.off_pairs = self.off_totals + 64
        end = self.off_pairs + 4096
        # ``fc2``'s per-warp transpose buffers ([8 tokens, 64 B] per epilogue warp).
        self.off_fc2s = (end + 127) // 128 * 128
        end = self.off_fc2s + 8 * 512
        assert end <= 227 * 1024
        # One CTA per SM (two co-resident CTAs would serialize in tcgen05.alloc while
        # waiting on each other's partials).
        self.smem_bytes = max(end, 212096)
        self.bar_full, self.bar_empty, self.bar_accfull = (0, 8 * stages, 16 * stages)
        acc = self.bar_accfull
        self.bar_accempty, self.bar_hfull = (acc + 128, acc + 256)
        self.bar_hempty, self.bar_xready = (acc + 272, acc + 288)
        assert acc + 288 <= 600  # below the accumulator metadata
        # Stage metadata (misc 2560-).
        self.off_meta = self.off_misc + 2560
        assert 2560 + 16 * stages <= 4608
        self.off_queue = self.off_misc + 4608
        # Pool jobs' accumulator metadata (misc 600-, as ``ExpertsKernel``) is released
        # by its own barriers [16] past the unit queue's (misc 4768-4896), then the
        # claim hand-over (``claim_queue``: full, empty, start; the entry).
        self.bar_pmeta = self.bar_queue + 128
        self.bar_claims = self.bar_pmeta + 128
        assert self.bar_claims + 28 <= 5120


    @cute.jit
    def __call__(self, *args):
        options = dict(grid=(self.grid, 1, 1), block=(384, 1, 1), cluster=(1, 1, 1))
        maps = self.encode_maps(args[-3], args[-2])
        if cutlass.const_expr(self.fc2_pairload):
            kernel = self.kernel_maps3(args[:-3], *maps)
        else:
            kernel = self.kernel_maps(args[:-3], *maps)
        kernel.launch(stream=args[-1], use_pdl=True, **options)

    def encode_maps(self, w13, w2):
        maps = self.encode_maps2(w13, w2)
        if cutlass.const_expr(self.fc2_pairload):
            # ``fc2_pairload``: the w2 map again with a 256-row box (two adjacent 16 KiB slots: SMEM rows at 128 B)
            u8 = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8
            swizzle = cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
            rows, pitch = (self.experts * 3584, self.w2_row_bytes)
            maps = (*maps, encode_map(
                w2, u8, (pitch, rows, 1), (pitch, pitch * rows), (96, 256, 1), swizzle,
                cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
            ))  # fmt: skip
        return maps

    def encode_maps2(self, w13, w2):
        # RubinExpertsKernel's packed descriptors: they read only experts, w13_rows,
        # parts and packed, which this kernel sets the same way.
        if cutlass.const_expr(self.w2_row_bytes == 128):
            return RubinExpertsKernel.encode_maps(self, w13, w2)  # type: ignore[arg-type]
        # De-padded w2 (96-byte rows): RubinExpertsKernel's packed w13 map unchanged, the packed w2 map with the
        # 96-byte row stride and the same 96 B x 128-row box (the same bytes land in SMEM).
        u8 = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8
        swizzle = cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
        promotion = cuda.CUtensorMapL2promotion
        rows = (self.experts * self.w13_rows, self.experts * 3584)
        return (
            encode_map(
                w13, u8, (128, rows[0], 14), (1792, 128), (128, 128, 1), swizzle,
                promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            ),
            encode_map(
                w2, u8, (self.w2_row_bytes, rows[1], 1), (self.w2_row_bytes, self.w2_row_bytes * rows[1]), (self.w2_row_bytes, 128, 1), swizzle,  # 96: de-padded rows; 112: W2SIB records
                promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
            ),
        )  # fmt: skip

    @cute.jit
    def initialize(self, sm, maps, batch):
        """``ExpertsKernel.initialize`` (grouped barrier counts) with the grid-constant
        descriptors prefetched instead of acquired."""
        tid, _, _ = cute.arch.thread_idx()
        barriers = view(sm.iterator, self.off_misc, Uint64, 65)
        tmem_base = view(sm.iterator, self.off_misc + 856, Uint32, 1)
        if tid == 0:
            acc = 2 * self.stages
            for i in cutlass.range_constexpr(acc + 36):
                # accfull: the MMAs' commit only (``epilogue`` walks the sequence)
                count = 4 if acc + 16 <= i < acc + 32 else 1
                if i < self.stages:
                    count = 33  # the producer and the gather warp's 32 lanes
                if self.stages <= i < 2 * self.stages:
                    count = 2  # empty (K-10): the MMAs' commit + lane 0's arrive after the warp's join
                cute.arch.mbarrier_init(barriers.iterator + i, count)
            queue = view(sm.iterator, self.off_misc + self.bar_queue, Uint64, 16)
            pmeta = view(sm.iterator, self.off_misc + self.bar_pmeta, Uint64, 16)
            claims = view(sm.iterator, self.off_misc + self.bar_claims, Uint64, 3)
            for i in cutlass.range_constexpr(16):
                cute.arch.mbarrier_init(queue.iterator + i, 1)
                cute.arch.mbarrier_init(pmeta.iterator + i, 1)
            for i in cutlass.range_constexpr(3):
                cute.arch.mbarrier_init(claims.iterator + i, 1)
            cute.arch.mbarrier_init_fence()
            for descriptor in maps:
                prefetch_map(descriptor)  # parameter space: no proxy fence
        if tid // 32 == 3:
            cute.arch.alloc_tmem(512, tmem_base.iterator)
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.fence_proxy("async.shared", space="cta")
        return tmem_base

    @cute.jit
    def fetch(self, sm, desc, scales, full, slot, row, chunk, offset, size, more=None):
        """One packed weight box and its 1 KiB of scales into ``slot``."""
        dst = sm.iterator + slot * self.slot_bytes
        load_tma3(dst, desc, full.iterator + slot, row, chunk)
        dst = sm.iterator + self.off_sfa + slot * 1024
        copy_evict(dst, scales.iterator + offset, size, full.iterator + slot)

    @cute.jit
    def issue(
        self, ring, slot, phase, issued, meta, tma, fc2: cutlass.Constexpr, more=None
    ):
        """``ExpertsKernel.issue`` for packed boxes: 16 KiB (FC1) or 12 KiB (FC2: the
        96 data bytes of 128 w2 rows) plus 1 KiB of scales. ``meta`` None: no stage
        metadata (the MMA warp walks FC1 and the static FC2 jobs itself)."""
        sm, barriers, metadata = ring[:3]
        desc, scales, row, chunk, offset, size = tma
        if issued >= self.stages:
            self.sync(barriers.iterator + self.stages + slot, phase ^ 1)
        if cutlass.const_expr(meta is not None):
            for i in cutlass.range_constexpr(4):
                metadata[4 * slot + i] = meta[i]
        full = barriers.iterator + slot
        box = 12288 if fc2 else 16384
        if cutlass.const_expr(fc2 and self.w2sib):  # W2SIB: one 112 B x 128-row record box (weights + their scales)
            cute.arch.mbarrier_arrive_and_expect_tx(full, 14336)
            load_tma3(sm.iterator + slot * self.slot_bytes, desc, barriers.iterator + slot, row, chunk)
        else:
            cute.arch.mbarrier_arrive_and_expect_tx(full, box + size)
            self.fetch(sm, desc, scales, barriers, slot, row, chunk, offset, size)
        if cutlass.const_expr(fc2):
            cute.arch.mbarrier_arrive(full, arrive_count=32)
        slot += 1
        if slot == self.stages:
            slot, phase = (Int32(0), phase ^ 1)
        return slot, phase, issued + 1

    @cute.jit
    def fc2_job(self, ring, state, maps, experts, ranges, j, pool: cutlass.Constexpr):
        """``ExpertsKernel.fc2_job`` (TP16: one tile per job) on the w2 map; stage
        metadata only for the ``pool``'s jobs."""
        slot, phase, issued, pushed, current = state
        unit, job = (j // self.jobs, j % self.jobs)
        if unit != current:
            pushed = self.push_unit(ring[0], pushed, unit)
            current = unit
        expert = experts[unit]
        row = expert * 3584 + job * 128
        offset = Int64(expert) * self.w2_scale_bytes + job * self.w2_tile_scale_bytes
        tma = (maps[1], ring[3], row, Int32(0), offset, Int32(1024))
        meta = (Int32(1), unit, job, Int32(0)) if pool else None
        slot, phase, issued = self.issue(ring, slot, phase, issued, meta, tma, True)
        return slot, phase, issued, pushed, current

    @cute.jit
    def zone_jobs(self, ranges):
        """The jobs at the end of every FC2(B) range that go into the pool: the zone in
        whole grains, at most the shortest FC2(B) range (the same for every CTA)."""
        units = ranges[1, 1] // (self.tiles * 14)
        zone = Int32(self.pool_zone)
        shortest = self.jobs * units // self.grid
        if shortest < zone:
            zone = shortest
        return zone // self.pool_grain * self.pool_grain

    @cute.jit
    def rank_ranges(self, sm, units, rank):
        """``partition`` of ``rank`` (a subclass may take its FC2 ranges from shared
        tables: ``home_bits``, ``matched_ranges`` and ``claims`` come here)."""
        return self.partition(units, rank)

    @cute.jit
    def fc2b_end(self, sm, group, rank):
        """The end of ``rank``'s FC2(B) job range; ``group``: FC2(B)'s (first unit,
        units), as ``partition``."""
        start, units = group
        return self.jobs * start + (rank + 1) * self.jobs * units // self.grid

    @cute.jit
    def pool_homed(self, sm, die):
        """The FC2(B) ranks in ``die``'s pool, one word per 32 ranks: with ``dies``, the
        ranks whose middle w2 tile sits on ``die`` (``home_bits``); else every rank."""
        words: cutlass.Constexpr = (self.grid + 31) // 32
        masks = [(1 << min(32, self.grid - 32 * w)) - 1 for w in range(words)]
        homed = [Uint32(m) for m in masks]
        if cutlass.const_expr(len(self.dies) > 0):
            bits = view(sm.iterator, self.off_misc + 2704 + 48 * 3, Uint32, 12)
            flip = Uint32(0)
            if die == 0:
                flip = Uint32(4294967295)
            homed = [(bits[w] ^ flip) & Uint32(masks[w]) for w in range(words)]
        return homed

    @cute.jit
    def pool_rank(self, homed, index):
        """The ``index``-th (0-based) rank set in ``homed``."""
        rank, before = (Int32(0), Int32(0))
        for w in cutlass.range_constexpr(len(homed)):
            count = Int32(cute.arch.popc(homed[w]))
            if (index >= before) & (index < before + count):
                rank = 32 * w + Int32(nth_set(homed[w], index - before))
            before += count
        return rank

    @cute.jit
    def claim_queue(self, sm):
        """The claim warp's hand-over to the producer: one entry (a grain's first job,
        -1 ends) with its full / empty barriers, and the producer's start."""
        bars = view(sm.iterator, self.off_misc + self.bar_claims, Uint64, 3)
        entry = view(sm.iterator, self.off_misc + self.bar_claims + 24, Int32, 1)
        return entry, bars

    @cute.jit
    def fc2_pool(self, ring, state, maps, experts, ranges, workspace, parity):
        """FC2: this CTA's static jobs (FC2(A), then FC2(B) less its last ``zone_jobs``
        jobs; no stage metadata), then the pool's grains as the claim warp hands them
        over (``claims``: the grain's first job; -1 ends), with metadata. The producer
        issues no atomics itself: its slot barriers' arrives are releases, which would
        wait for a claim in flight; and it does no more per pool job than per static
        job (FC2 here is bound by the producer's issue rate as much as by the MMAs)."""
        sm = ring[0]
        lo, hi = (ranges[1, 4], ranges[1, 5] - self.zone_jobs(ranges))
        base, bars = self.claim_queue(sm)
        if cutlass.const_expr(self.fc2_pairload):
            state = self.fc2_pairs(ring, state, maps, experts, ranges, ranges[0, 4], ranges[0, 5], Int32(-1), bars)
        else:
            for j in cutlass.range(ranges[0, 4], ranges[0, 5], unroll=1):
                state = self.fc2_job(ring, state, maps, experts, ranges, j, False)
        # The first claim goes out ``POOL_AHEAD`` static jobs before the end, so its
        # grain is there when the static jobs are issued.
        kick = hi - POOL_AHEAD
        if cutlass.const_expr(self.pool_fix):
            kick = hi - POOL_AHEAD_FIX
        if kick < lo:
            kick = lo
        if cutlass.const_expr(self.fc2_pairload):
            state = self.fc2_pairs(ring, state, maps, experts, ranges, lo, hi, kick, bars)
        else:
            for j in cutlass.range(lo, hi, unroll=1):
                if j == kick:
                    cute.arch.mbarrier_arrive(bars.iterator + 2)
                state = self.fc2_job(ring, state, maps, experts, ranges, j, False)
        if hi <= kick:
            cute.arch.mbarrier_arrive(bars.iterator + 2)
        taken, ended = (Int32(0), False)
        while not ended:
            self.sync(bars.iterator, taken & 1)
            first = base[0]
            cute.arch.mbarrier_arrive(bars.iterator + 1)
            taken += 1
            if first < 0:
                ended = True
            else:
                for j in cutlass.range(first, first + self.pool_grain, unroll=1):
                    state = self.fc2_job(ring, state, maps, experts, ranges, j, True)
        slot, phase, issued, pushed, current = state
        self.push_unit(sm, pushed, Int32(-1))
        return slot, phase, issued

    @cute.jit
    def fc2_pairs(self, ring, state, maps, experts, ranges, lo, hi, kick, bars):
        """``fc2_pairload``: static FC2 jobs [lo, hi) in order; two consecutive jobs of one unit whose slots do not
        wrap the ring go out as one pair (``fc2_pair``), the rest as single jobs (``fc2_job``). ``kick`` (>= 0): the
        claim warp's start signal before job ``kick``, as ``fc2_pool``."""
        j = lo
        while j < hi:
            slot = state[0]
            pair = (j + 1 < hi) & (j // self.jobs == (j + 1) // self.jobs) & (slot + 1 < self.stages)
            if (j == kick) | (pair & (j + 1 == kick)):
                cute.arch.mbarrier_arrive(bars.iterator + 2)
            if pair:
                state = self.fc2_pair(ring, state, maps, experts, j)
                j += 2
            else:
                state = self.fc2_job(ring, state, maps, experts, ranges, j, False)
                j += 1
        return state

    @cute.jit
    def fc2_pair(self, ring, state, maps, experts, j):
        """``fc2_pairload``: static FC2 jobs j and j + 1 (one unit) into slots s and s + 1 (s + 1 < stages): both slots'
        empty waits, then one expect-tx arrive for both tiles and their scales on full[s], the 256-row box and the 2 KiB
        scale copy on full[s], full[s]'s 32 gather-lane arrives, and full[s + 1]'s 33 arrives (no bytes)."""
        slot, phase, issued, pushed, current = state
        sm, barriers = ring[:2]
        unit, job = (j // self.jobs, j % self.jobs)
        if unit != current:
            pushed = self.push_unit(sm, pushed, unit)
            current = unit
        expert = experts[unit]
        row = expert * 3584 + job * 128
        offset = Int64(expert) * self.w2_scale_bytes + job * self.w2_tile_scale_bytes
        if issued >= self.stages:
            self.sync(barriers.iterator + self.stages + slot, phase ^ 1)
        if issued + 1 >= self.stages:  # slot + 1 < stages: the same lap, so the same phase
            self.sync(barriers.iterator + self.stages + slot + 1, phase ^ 1)
        full = barriers.iterator + slot
        cute.arch.mbarrier_arrive_and_expect_tx(full, 2 * (12288 + 1024))
        load_tma3(sm.iterator + slot * self.slot_bytes, maps[2], full, row, Int32(0))
        copy_evict(sm.iterator + self.off_sfa + slot * 1024, ring[3].iterator + offset, Int32(2048), full)
        cute.arch.mbarrier_arrive(full, arrive_count=32)
        cute.arch.mbarrier_arrive(full + 1, arrive_count=33)
        slot += 2
        if slot == self.stages:
            slot, phase = (Int32(0), phase ^ 1)
        return (slot, phase, issued + 2, pushed, current)

    @cute.jit
    def claims(self, sm, workspace, ranges, parity):
        """Warp 3 lane 0, once the producer reached its last static job: the pool
        claims. Pool d holds the zones of the FC2(B) ranks homed on die d
        (``pool_homed``) in rank order; this CTA's die's pool first, then the other.
        One counter per pool and call parity (``ExpertsKernel.ws_steal``, re-armed to
        -1 a call ahead by ``rearm``). A position's job is its zone rank's job,
        whoever runs it, so each job still runs exactly once. A grain is claimed once
        the producer took the previous one (a late CTA holds at most one grain it has
        not started); its first job is handed over, -1 ends. A pool the last grain
        emptied is not claimed again (the end costs one claim, not two)."""
        g = self.pool_grain
        zone = self.zone_jobs(ranges)
        units, start = (ranges[1, 1] // (self.tiles * 14), ranges[1, 0])
        base, bars = self.claim_queue(sm)
        counters = workspace.iterator + self.ws_steal + 4 * self.slots * parity
        own = Int32(0)
        if cutlass.const_expr(len(self.dies) > 0):
            own = view(sm.iterator, self.off_misc + 2688, Int32, 4)[0]
        self.sync(bars.iterator + 2, 0)
        posted = Int32(0)
        for p in cutlass.range_constexpr(2 if len(self.dies) > 0 else 1):
            homed = self.pool_homed(sm, own ^ p)
            ranks = Int32(0)
            for w in cutlass.range_constexpr(len(homed)):
                ranks += Int32(cute.arch.popc(homed[w]))
            size = zone * ranks
            active = size > 0
            while active:
                if posted > 0:
                    self.sync(bars.iterator + 1, posted - 1 & 1)
                position = Int32(claim(counters + 4 * (own ^ p), g)) + 1
                active = position < size
                if active:
                    rank = self.pool_rank(homed, position // zone)
                    end = self.fc2b_end(sm, (start, units), rank)
                    base[0] = end - zone + position % zone
                    cute.arch.mbarrier_arrive(bars.iterator)
                    posted += 1
                    # Counters only grow: the next claim here would come back empty.
                    active = position + g < size
        if posted > 0:
            self.sync(bars.iterator + 1, posted - 1 & 1)
        base[0] = Int32(-1)
        cute.arch.mbarrier_arrive(bars.iterator)

    @cute.jit
    def stream(self, sm, maps, w13_scales, w2_scales, ranges, experts, steal=None):
        """``ExpertsKernel.stream`` (grouped, TP16) on the packed maps: FC1(A), FC1(B),
        then the FC2 jobs (``fc2_pool``); stage metadata only for the pool's jobs."""
        w13, w2 = (maps[0], maps[1])
        barriers = view(sm.iterator, self.off_misc, Uint64, 2 * self.stages)
        metadata = view(sm.iterator, self.off_meta, Int32, 4 * self.stages)
        ring = (sm, barriers, metadata, w2_scales)
        slot, phase, issued = (Int32(0), Int32(0), Int32(0))
        for group in cutlass.range_constexpr(2):
            lo, hi = (ranges[group, 2], ranges[group, 3])
            ks, tile = (lo % 14, 3 * ranges[group, 0] + lo // 14)
            r, index = (tile % 3, self.first_unit(ranges, 0, group))
            expert = experts[index]
            for _ in cutlass.range(lo, hi, unroll=1):
                row = expert * self.w13_rows + r * 128
                offset = Int64(expert) * self.w13_scale_bytes + r * 14336 + ks * 1024
                tma = (w13, w13_scales, row, ks, offset, Int32(1024))
                slot, phase, issued = self.issue(
                    ring, slot, phase, issued, None, tma, False
                )
                ks += 1
                if ks == 14:
                    ks, tile, r = (Int32(0), tile + 1, r + 1)
                    if r == 3:
                        r, index = (Int32(0), index + 1)
                        expert = experts[index]
        state = (slot, phase, issued, Int32(0), Int32(-1))
        work = (ring, state, maps, experts, ranges, *steal)
        slot, phase, issued = self.fc2_pool(*work)
        if issued >= self.stages:
            self.sync(barriers.iterator + self.stages + slot, phase ^ 1)
        for i in cutlass.range_constexpr(4):
            metadata[4 * slot + i] = Int32(2 if i == 0 else 0)
        cute.arch.mbarrier_arrive(barriers.iterator + slot, arrive_count=33)

    @cute.jit
    def consume(self, sm, tmem, ranges):
        """``ExpertsKernel.consume`` (grouped, TP16) with ``stage16``'s packed K64 MMAs:
        FC1 stages read the slot's weights and its own activation slice and scales,
        FC2 stages one tile against the unit's hidden image. The FC1 stages come with
        the producer's metadata; the static FC2 jobs are walked from this CTA's matched
        FC2 ranks in the producer's order (no metadata loads or kind dispatch on FC2's
        per-stage path), and the pool's jobs (``fc2_pool``) follow in their own loop
        with the producer's metadata."""
        lane, stages = (thread()[2], self.stages)
        bars: tuple = (self.bar_full, self.bar_empty, self.bar_accfull)
        bars += (self.bar_accempty, self.bar_hfull, self.bar_hempty)
        full, empty, accfull, accempty, hfull, hempty = (
            view(sm.iterator, self.off_misc + o, Uint64, n)
            for o, n in zip(bars, (stages, stages, 16, 16, 2, 2))
        )
        meta = view(sm.iterator, self.off_meta, Int32, 4 * stages)
        offsets = (0, self.off_sfa, self.off_x, self.off_xsf, self.off_h, self.off_hsf)
        a0, asf0, x0, xsf0, h0, hsf0 = (
            smem_desc(sm.iterator + o, i % 2 == 0) for i, o in enumerate(offsets)
        )
        slot16 = self.slot_bytes // 16
        uses, acc, count = (Int32(0), Int32(0), Int32(0))
        current, ordinal, hbuffer = (Int32(-1), Int32(-1), Int32(0))
        for g in cutlass.range_constexpr(2):  # FC1(A), FC1(B): the producer's order
            lo, hi = (ranges[g, 2], ranges[g, 3])
            ks = lo % 14
            for s in cutlass.range(lo, hi, unroll=1):
                slot = count % stages
                self.sync(full.iterator + slot, count // stages & 1, True)
                weights = (tmem + 128 + slot * 8, asf0 + slot * 64)
                released = shared_address(empty.iterator + slot)
                first, last = ((s == lo) | (ks == 0), (s == hi - 1) | (ks == 13))
                if first:
                    self.acquire(accempty, acc, uses)
                # The slot's own activation slice (its cp.async completed full[slot]).
                cute.arch.fence_proxy("async.shared", space="cta")
                completion = Uint32(0)
                if last:
                    completion = shared_address(accfull.iterator + acc)
                stage16(
                    (tmem + acc * 8, a0 + slot * slot16, x0 + slot * 128),
                    weights,
                    (tmem + 256 + slot * 8, xsf0 + slot * 16, Uint32(1)),
                    (Uint32(not first), released),
                    completion,
                    kind="fc1",
                    early=False,
                    packed=True,
                )
                named_barrier(12, 32)  # K-10: the warp's reads of the slot happen before lane 0's release (PTX 8.9.4)
                if thread()[2] == 0:
                    cute.arch.mbarrier_arrive(empty.iterator + slot)  # empty counts 2: the MMAs' commit + this arrive
                if last:  # the commit signals the epilogue
                    uses += 1
                    acc = uses % 16
                count += 1
                ks += 1
                if ks == 14:
                    ks = Int32(0)
        # FC2: the static jobs (FC2(A), then FC2(B) less its pooled zone), walked;
        # then the pool's jobs as the producer claimed them (stage metadata).
        bars = (full, empty, accempty, accfull, hfull, hempty)
        mma = (tmem, a0, asf0, h0, hsf0, bars)
        slot, phase = (count % stages, count // stages & 1)
        state = (slot, phase, uses, acc, current, ordinal, hbuffer, Int32(0))
        zone = self.zone_jobs(ranges)
        for g in cutlass.range_constexpr(2):
            end = ranges[g, 5] - (zone if g == 1 else 0)
            for j in cutlass.range(ranges[g, 4], end, unroll=1):
                self.sync(full.iterator + state[0], state[1], True)
                state = self.fc2_mma(state, mma, j // self.jobs)
        accmeta = view(sm.iterator, self.off_misc + 600, Int32, 64)
        pmeta = view(sm.iterator, self.off_misc + self.bar_pmeta, Uint64, 16)
        ended = False
        while not ended:
            self.sync(full.iterator + state[0], state[1], True)
            kind, a, b = (meta[4 * state[0] + i] for i in range(3))
            if kind != 1:  # the end marker
                ended = True
            else:
                state = self.fc2_mma(state, mma, a, (accmeta, pmeta, b))
        # the end: one marker per epilogue group, then release the last hidden image
        slot, phase, uses, acc, current, ordinal, hbuffer, _ = state
        for _ in cutlass.range(2, unroll=1):
            self.acquire(accempty, acc, uses)
            self.signal(accmeta, pmeta, acc, (Int32(2), Int32(0), Int32(0), Int32(0)))
            uses += 1
            acc = uses % 16
        if (lane == 0) & (current >= 0) & (hbuffer < 2):
            commit(hempty.iterator + hbuffer)

    @cute.jit
    def fc2_mma(self, state, mma, unit, pool=None):
        """One FC2 job on the MMA warp, its slot's ``full`` waited: the unit's hidden
        image (a new unit releases the previous buffer through the MMAs' commit and
        waits for its own), an accumulator, the MMAs and their commit (the epilogue's
        signal). Static jobs acquire eight accumulators at a time: each epilogue group
        frees its (every other) accumulator in order, so uses + 6 and uses + 7 being
        free means uses .. uses + 5 are too. ``pool`` (accumulator metadata, its
        barriers, the job's tile): one at a time, the job's metadata released to the
        epilogue."""
        slot, phase, uses, acc, current, ordinal, hbuffer, ahead = state
        tmem, a0, asf0, h0, hsf0, bars = mma
        full, empty, accempty, accfull, hfull, hempty = bars
        lane, stages = (thread()[2], self.stages)
        weights = (tmem + 128 + slot * 8, asf0 + slot * 64)
        stage = stage16
        if cutlass.const_expr(self.w2sib):  # W2SIB: the scales in the slot's record box (column 96 of rows 0-63)
            weights = (tmem + 128 + slot * 8, a0 + slot * (self.slot_bytes // 16) + 6)
            stage = stage16_sib
        released = shared_address(empty.iterator + slot)
        copy_scales, release = (Uint32(0), Uint32(0))
        if unit != current:
            if (current >= 0) & (hbuffer < 2):
                release = shared_address(hempty.iterator + hbuffer)
            current = unit
            ordinal += 1
            hbuffer = ordinal & 1
            self.sync(hfull.iterator + hbuffer, ordinal >> 1 & 1, True)
            copy_scales = Uint32(1)
        hd = h0 + hbuffer * (self.h_image_bytes // 16)
        hs = hsf0 + hbuffer * (self.h_scale_bytes // 16)
        sfbc = tmem + 128 + 8 * stages + hbuffer * self.w2_scale_columns
        # ``pool_fix``: the pool's jobs acquire ahead as well; the epilogue groups release their (every other) accumulators in order
        # through the pool jobs too (``epilogue``: m += 2 per group), so the same argument holds.
        if cutlass.const_expr(pool is None or self.pool_fix):
            if ahead == 0:
                for i in cutlass.range_constexpr(6, 8):
                    self.acquire(accempty, (uses + i) % 16, uses + i)
                ahead = Int32(8)
            ahead -= 1
        else:
            self.acquire(accempty, acc, uses)
        slot16 = self.slot_bytes // 16
        stage(
            (tmem + acc * 8, a0 + slot * slot16, hd),
            weights,
            (sfbc, hs, copy_scales),
            (release, released),
            shared_address(accfull.iterator + acc),
            kind="fc2",
            early=False,
            packed=True,
        )
        named_barrier(12, 32)  # K-10: the warp's reads of the slot (and its metadata) happen before lane 0's release (PTX 8.9.4)
        if thread()[2] == 0:
            cute.arch.mbarrier_arrive(empty.iterator + slot)  # empty counts 2: the MMAs' commit + this arrive
        if cutlass.const_expr(pool is not None):
            accmeta, pmeta, tile = pool
            self.signal(accmeta, pmeta, acc, (Int32(1), unit, tile, Int32(0)))
        uses += 1
        acc = uses % 16
        slot += 1
        if slot == stages:
            slot, phase = (Int32(0), phase ^ 1)
        return (slot, phase, uses, acc, current, ordinal, hbuffer, ahead)

    @cute.jit
    def epilogue(self, sm, workspace, output, tmem, ranges, batch, parity, scalars):
        """``ExpertsKernel.epilogue`` without accumulator metadata for the static work:
        the accumulators come in ``consume``'s order (one per FC1(A) and FC1(B) tile
        this CTA's stage ranges touch, then its static FC2(A) and FC2(B) jobs, all from
        its matched ranks), and the two epilogue groups take every other one. Then the
        pool's jobs with their metadata (released by ``pmeta``) up to one end marker
        per group."""
        eg = (thread()[1] - 4) // 4
        full = view(sm.iterator, self.off_misc + self.bar_accfull, Uint64, 16)
        empty = view(sm.iterator, self.off_misc + self.bar_accempty, Uint64, 16)
        zone = self.zone_jobs(ranges)
        acc, phase, offset = (eg, Int32(0), Int32(0))
        for k in cutlass.range_constexpr(4):
            g = k % 2
            if cutlass.const_expr(k < 2):  # FC1 tiles
                lo, hi = (ranges[g, 2], ranges[g, 3])
                start, count = (self.tiles * ranges[g, 0] + lo // 14, Int32(0))
                if lo < hi:
                    count = (hi - 1) // 14 - lo // 14 + 1
            else:  # static FC2 jobs
                start = ranges[g, 4]
                count = ranges[g, 5] - start - (zone if g == 1 else 0)
            for i in cutlass.range((eg ^ offset) & 1, count, 2, unroll=1):
                values = self.accumulator(tmem, (full, empty), acc, phase)
                acc += 2
                if acc >= 16:
                    acc -= 16
                    phase ^= 1
                item = start + i
                if cutlass.const_expr(k < 2):
                    if self.partials(workspace, values, item, ranges, batch):
                        store = (sm, workspace, values, item, parity, scalars[:3])
                        self.store_hidden(*store)
                else:
                    unit, tile = (item // self.jobs, item % self.jobs)
                    self.fc2(sm, workspace, output, values, unit, tile, ranges)
            offset += count
        # The pool: this group's uses offset + m (m of its parity); pmeta[acc]'s phase
        # counts the earlier pool uses of the slot (m // 16).
        accmeta = view(sm.iterator, self.off_misc + 600, Int32, 64)
        pmeta = view(sm.iterator, self.off_misc + self.bar_pmeta, Uint64, 16)
        m, ended = ((eg ^ offset) & 1, False)
        while not ended:
            self.sync(pmeta.iterator + acc, m // 16 & 1)
            kind, unit, tile = (accmeta[4 * acc + i] for i in range(3))
            if kind != 1:  # this group's end marker
                ended = True
            else:
                values = self.accumulator(tmem, (full, empty), acc, phase)
                acc += 2
                if acc >= 16:
                    acc -= 16
                    phase ^= 1
                m += 2
                self.fc2(sm, workspace, output, values, unit, tile, ranges)
        if cutlass.const_expr(self.publish):
            self.finalize(workspace, output, batch, scalars[3:], sm=sm)

    @cute.jit
    def accumulator(self, tmem, bars, acc, phase):
        """Epilogue warp: accumulator ``acc``'s 8 FP32 columns of this warp's 32 rows
        once its MMAs completed, then the accumulator released."""
        _, warp, lane, _ = thread()
        full, empty = bars
        self.sync(full.iterator + acc, phase, True)
        raw = load_tmem(tmem + (Uint32(32 * (warp & 3)) << 16) + acc * 8, 8)
        values = cute.make_rmem_tensor(self.width, Float32)
        for t in cutlass.range_constexpr(self.width):
            values[t] = raw[t]
        fence_tmem(before=True)
        cute.arch.sync_warp()
        if lane == 0:
            cute.arch.mbarrier_arrive(empty.iterator + acc)
        return values

    def tables(self, sm, window=True):
        """The unit tables (experts, eight Int16 pair rows per unit, the rows as four
        words per unit) past misc, indexed by global unit (no windows)."""
        base, units = (self.off_tables, self.units)
        experts = view(sm.iterator, base, Int32, units)
        rows = view(sm.iterator, base + 4 * units, Int16, 8 * units)
        return experts, rows, view(sm.iterator, base + 4 * units, Uint32, 4 * units)

    @cute.jit
    def clear(self, sm):
        """Every thread: ``group``'s token masks to zero and the row words to -1 (the
        caller's barrier follows)."""
        tid = thread()[0]
        masks = view(sm.iterator, self.off_masks, Uint32, 2 * self.experts)
        row_words = self.tables(sm)[2]
        for i in range(tid, 2 * self.experts, self.threads):
            masks[i] = Uint32(0)
        for i in range(tid, 4 * self.units, self.threads):
            row_words[i] = Uint32(4294967295)

    @cute.jit
    def group(self, sm, ids, workspace, batch: Int32, scalars=None, arrival=None, pst=None):  # K31MK_MOE_RUBIN64_PRESTREAM: pst = (w13 map, w13 scales)
        """``ExpertsKernel.group`` into the tables past misc: per-expert token masks;
        expert e's tokens, in token order, fill ceil(n / 8) units numbered in expert
        order; misc scalars (units, call, grouped call). Warp 0 returns once the
        experts table and the unit count are written (its producer lane streams while
        warps 1-11 rank the pairs into the rows; they meet at named barrier 6)."""
        tid, warp, lane, cta = thread()
        masks = view(sm.iterator, self.off_masks, Uint32, 2 * self.experts)
        base = view(sm.iterator, self.off_bases, Int32, self.experts)
        totals = view(sm.iterator, self.off_totals, Int32, self.threads // 32)
        pairs = view(sm.iterator, self.off_pairs, Int32, 16 * self.capacity)
        experts, rows, row_words = self.tables(sm)
        scalars = view(sm.iterator, self.off_misc + 1140, Int32, 3)
        threads: cutlass.Constexpr = self.threads
        rounds: cutlass.Constexpr = (16 * self.capacity + threads - 1) // threads
        mine = cute.make_rmem_tensor(rounds, Int32)
        # The call counters and the ids in one round trip; the masks and row words
        # are cleared while it is in flight.
        ws, slots = (workspace.iterator, self.slots)
        # unconditional (the same two words for every thread): a value produced under a
        # branch would be waited for at the branch's end, before the ids are loaded
        calls = view(ws, self.ws_counter, Int32, slots)[cta]
        grouped = view(ws, self.ws_calls, Int32, slots)[cta]
        for k in cutlass.range_constexpr(rounds):
            mine[k] = Int32(0)
            if tid + threads * k < batch * 16:
                mine[k] = ids[tid + threads * k]
        self.clear(sm)  # while the ids are in flight
        for k in cutlass.range_constexpr(rounds):
            if tid + threads * k < batch * 16:
                pairs[tid + threads * k] = mine[k]
        if cutlass.const_expr(self.pst_nb > 0 and pst is not None):  # K31MK_MOE_RUBIN64_PRESTREAM
            # Each routed id's first ``pst_nb`` FC1 boxes (row tile 0) and their scales into L2 while the grouping runs: item
            # (id p, piece j) by CTA (p x (pst_nb + 1) + j) % grid, issued by the thread that holds id p. Read-only weights,
            # cache hints only: no ordering, outputs unchanged.
            pst_n: cutlass.Constexpr = self.pst_nb
            pst_items: cutlass.Constexpr = self.pst_nb + 1
            for k in cutlass.range_constexpr(rounds):
                p = tid + threads * k
                if p < batch * 16:
                    j = (cta + self.grid - (p * pst_items) % self.grid) % self.grid
                    if j < pst_items:
                        e = mine[k]
                        if j < pst_n:
                            prefetch_box_l2(pst[0], e * self.w13_rows, j)
                        else:
                            prefetch_bytes_l2(pst[1].iterator + Int64(e) * self.w13_scale_bytes, pst_n * 1024)
        if tid == 0:
            scalars[1], scalars[2] = (calls, grouped)
        cute.arch.sync_threads()
        for k in cutlass.range_constexpr(rounds):
            pair = tid + threads * k
            if pair < batch * 16:
                token = pair >> 4
                bit = Uint32(1) << (token & 31)
                word = masks.iterator + 2 * mine[k] + (token >> 5)
                cute.arch.atomic_or(word, bit, sem="relaxed", scope="cta")
        cute.arch.sync_threads()
        span: cutlass.Constexpr = (self.experts + threads - 1) // threads
        units = cute.make_rmem_tensor(span, Int32)
        count = Int32(0)
        for j in cutlass.range_constexpr(span):
            units[j], e = (Int32(0), span * tid + j)
            if e < self.experts:
                n = cute.arch.popc(masks[2 * e]) + cute.arch.popc(masks[2 * e + 1])
                units[j] = Int32(n + 7) >> 3
            count += units[j]
        total = count
        for shift in cutlass.range_constexpr(5):
            other = cute.arch.shuffle_sync_up(total, 1 << shift)
            if lane >= 1 << shift:
                total += other
        if lane == 31:
            totals[warp] = total
        cute.arch.sync_threads()
        start = total - count
        for w in cutlass.range_constexpr(threads // 32 - 1):
            if warp > w:
                start += totals[w]
        if tid == threads - 1:
            scalars[0] = start + count
        most: cutlass.Constexpr = (self.capacity + 7) // 8  # units of one expert
        for j in cutlass.range_constexpr(span):
            e = span * tid + j
            if e < self.experts:
                base[e] = start
                for v in cutlass.range_constexpr(most):  # predicated, no loop
                    if v < units[j]:
                        experts[start + v] = e
            start += units[j]
        if cutlass.const_expr(len(self.dies) > 0):  # the ticket has long returned
            self.arrive(sm, workspace, arrival)
        cute.arch.sync_threads()
        if cutlass.const_expr(len(self.dies) > 0):  # before warp 0 starts streaming
            if cutlass.const_expr(self.gcut):
                gcut_ranges = self.home_bits_gcut(sm, experts, scalars[0])
            else:
                self.home_bits(sm, experts, scalars[0])
            cute.arch.sync_threads()
            if warp == 0:
                self.matched(sm)
            cute.arch.sync_threads()
            if cutlass.const_expr(self.gcut):  # each list's matched rank's own thread publishes its ranges (misc 2928, 14 words)
                if warp < (self.grid + 31) // 32:
                    gcut_ranks = view(sm.iterator, self.off_misc + 2904, Int32, 4)
                    gcut_table = view(sm.iterator, self.off_misc + 2928, Int32, 14)
                    for g in cutlass.range_constexpr(2):
                        if tid == gcut_ranks[g]:
                            for i in cutlass.range_constexpr(4):
                                gcut_table[7 * g + i] = gcut_ranges[g, i]
                            gcut_table[7 * g + 6] = tid
                        if tid == gcut_ranks[2 + g]:
                            gcut_table[7 * g + 4] = gcut_ranges[g, 4]
                            gcut_table[7 * g + 5] = gcut_ranges[g, 5]
                cute.arch.sync_threads()
        if warp > 0:
            for pair in cutlass.range(tid - 32, batch * 16, threads - 32, unroll=1):
                e, token = (pairs[pair], pair >> 4)
                low, word = (masks[2 * e], masks[2 * e + 1])
                rank = Int32(0)
                if token < 32:
                    word = low
                else:
                    rank = Int32(cute.arch.popc(low))
                below = (Uint32(1) << (token & 31)) - 1
                rank += Int32(cute.arch.popc(word & below))
                rows[(base[e] + (rank >> 3)) * 8 + (rank & 7)] = Int16(pair)
            cute.arch.barrier(barrier_id=6, number_of_threads=threads - 32)

    @cute.jit
    def gather_x(self, sm, plan, ranges, batch):
        """``ExpertsKernel.gather_x`` on the global unit tables."""
        lane, ws = (thread()[2], plan.iterator)
        stages, rows = (self.stages, self.tables(sm)[1])
        full, empty = (
            view(sm.iterator, self.off_misc + o, Uint64, stages)
            for o in (self.bar_full, self.bar_empty)
        )
        chunk, lanes = (lane & 15, lane // 16)
        image = [self.off_x + swizzled(chunk, 2 * j + lanes) for j in range(4)]
        scale = self.off_xsf + lane // 8 * 128 + (lane & 7) * 16
        sources = cute.make_rmem_tensor(5, Int32)
        slot, phase, issued = (Int32(0), Int32(0), Int32(0))
        for group in cutlass.range_constexpr(2):
            lo, hi = (ranges[group, 2], ranges[group, 3])
            ks, r = (lo % 14, lo // 14 % 3)
            index = self.first_unit(ranges, 0, group)
            if lo < hi:
                self.row_sources(rows, index, sources, batch)
            for s in cutlass.range(lo, hi, unroll=1):
                if issued >= stages:
                    self.sync(empty.iterator + slot, phase ^ 1)
                for j in cutlass.range_constexpr(4):
                    src = sources[j] + ks * 256 + chunk * 16
                    copy_g2s(sm.iterator + image[j] + slot * 2048, ws + src)
                if lane < 16:
                    src = PLAN_XSF + sources[4] + (2 * ks + lane // 8) * 16
                    copy_g2s(sm.iterator + scale + slot * 256, ws + src)
                cute.arch.cp_async_mbarrier_arrive_noinc(full.iterator + slot)
                slot, issued = (slot + 1, issued + 1)
                if slot == stages:
                    slot, phase = (Int32(0), phase ^ 1)
                ks += 1
                if ks == 14:
                    ks, r = (Int32(0), r + 1)
                    if r == 3:
                        r, index = (Int32(0), index + 1)
                        if s + 1 < hi:
                            self.row_sources(rows, index, sources, batch)

    @cute.jit
    def die(self):
        """This SM's die (``dies``)."""
        smid = sm_id()
        word = Uint32(0)
        for w in cutlass.range_constexpr(len(self.die_bits)):
            if smid >> 5 == w:
                word = Uint32(self.die_bits[w])
        return Int32(word >> (smid & 31) & 1)

    @cute.jit
    def ticket(self, workspace):
        """Warp 2 lane 0 right after the dependency wait (``dies``): this CTA's census
        add, 1 + (die << 32) on the word of arrivals (low half) and die-1 arrivals (high
        half); the value before it is consumed in ``group`` (``arrive``), so the
        atomic's latency overlaps the grouping."""
        before = Uint64(0)
        if cutlass.const_expr(self.gcut):  # every thread runs the asm; only warp 2 lane 0's predicate is set
            active = Int32(thread()[1] == 2) * Int32(thread()[2] == 0)
            increment = Uint64(1) + (Uint64(self.die()) << Uint64(32))
            before = census_add_if(workspace.iterator + self.ws_census, increment, Uint32(active))
        else:
            if (thread()[1] == 2) & (thread()[2] == 0):
                increment = Uint64(1) + (Uint64(self.die()) << Uint64(32))
                before = census_add(workspace.iterator + self.ws_census, increment)
        return before

    @cute.jit
    def arrive(self, sm, workspace, before):
        """Warp 2 lane 0, before the experts table's barrier: (die, arrivals before this
        CTA, call, its rank k among the call's CTAs on its die) into misc 2688. The
        word counts from all-ones (the armed workspace), so before + 1 is (earlier
        die-1 arrivals) << 32 | (earlier arrivals) modulo 2^64. The call's last arrival
        re-arms the word: every CTA of the call has added by then, and the next call's
        CTAs add only after their dependency wait, which follows this call's outputs
        through the step's kernel chain (its routes, attention and front); a word that
        was not re-armed traps (an arrival count of grid or more)."""
        info = view(sm.iterator, self.off_misc + 2688, Int32, 4)
        if (thread()[1] == 2) & (thread()[2] == 0):
            counts = before + Uint64(1)
            arrivals = Uint32(counts & Uint64(4294967295))
            ones = Uint32(counts >> Uint64(32))
            require_below(arrivals, Uint32(self.grid))
            die = self.die()
            k = Int32(arrivals) - Int32(ones)
            if die == 1:
                k = Int32(ones)
            if arrivals == Uint32(self.grid - 1):
                census_rearm(workspace.iterator + self.ws_census)
            info[0], info[1], info[3] = (die, Int32(arrivals), k)

    @cute.jit
    def home_bits(self, sm, experts, units):
        """Threads 0 .. grid - 1 (rank = thread): the die of the middle w13 stage of
        rank's FC1(A) / FC1(B) range and of the middle w2 tile of its FC2(A) / FC2(B)
        range; one ballot word per warp and list (misc 2704 + list x 48 B). The four
        dies first (independent loads and arithmetic), then the four ballots."""
        tid, warp, lane, _ = thread()
        bits = view(sm.iterator, self.off_misc + 2704, Uint32, 48)
        if warp < (self.grid + 31) // 32:
            ranges = self.rank_ranges(sm, units, tid)
            dies = []
            for kind in cutlass.range_constexpr(2):
                for group in cutlass.range_constexpr(2):
                    lo, hi = (ranges[group, 2 + 2 * kind], ranges[group, 3 + 2 * kind])
                    mid = (lo + hi) // 2
                    if cutlass.const_expr(kind == 0):
                        tile = 3 * ranges[group, 0] + mid // 14
                        row = experts[tile // 3] * self.w13_rows + tile % 3 * 128 + 64
                        address = Int64(row) * 1792 + mid % 14 * 128
                    else:
                        row = experts[mid // 28] * 3584 + mid % 28 * 128 + 64
                        address = Int64(row) * self.w2_row_bytes
                    live = (tid < self.grid) & (lo < hi)
                    dies.append(live & ((address >> 21 & 1) != 0))
            for p in cutlass.range_constexpr(4):
                ballot = cute.arch.vote_ballot_sync(dies[p])
                if lane == 0:
                    bits[p * 12 + warp] = Uint32(ballot)

    @cute.jit
    def home_bits_gcut(self, sm, experts, units):
        """``home_bits`` for ``gcut``: every thread also computes and returns its own rank's ``rank_ranges``
        (the owner publish in ``group``). Threads 0 .. grid - 1 (rank = thread): the die of the middle w13 stage of
        rank's FC1(A) / FC1(B) range and of the middle w2 tile of its FC2(A) / FC2(B)
        range; one ballot word per warp and list (misc 2704 + list x 48 B). The four
        dies first (independent loads and arithmetic), then the four ballots."""
        tid, warp, lane, _ = thread()
        bits = view(sm.iterator, self.off_misc + 2704, Uint32, 48)
        ranges = self.rank_ranges(sm, units, tid)  # every thread: returned for the owner publish (``group``)
        if warp < (self.grid + 31) // 32:
            dies = []
            for kind in cutlass.range_constexpr(2):
                for group in cutlass.range_constexpr(2):
                    lo, hi = (ranges[group, 2 + 2 * kind], ranges[group, 3 + 2 * kind])
                    mid = (lo + hi) // 2
                    if cutlass.const_expr(kind == 0):
                        tile = 3 * ranges[group, 0] + mid // 14
                        row = experts[tile // 3] * self.w13_rows + tile % 3 * 128 + 64
                        address = Int64(row) * 1792 + mid % 14 * 128
                    else:
                        row = experts[mid // 28] * 3584 + mid % 28 * 128 + 64
                        address = Int64(row) * self.w2_row_bytes
                    live = (tid < self.grid) & (lo < hi)
                    dies.append(live & ((address >> 21 & 1) != 0))
            for p in cutlass.range_constexpr(4):
                ballot = cute.arch.vote_ballot_sync(dies[p])
                if lane == 0:
                    bits[p * 12 + warp] = Uint32(ballot)
        return ranges

    @cute.jit
    def matched(self, sm):
        """Warp 0: ``match_arrival`` of the four lists (FC1(A), FC1(B), FC2(A),
        FC2(B); ``home_bits`` and the arrival rank) into misc 2904 + 4 p, lanes 8 p of
        list p, each alone: the list's words (independent loads), the die-1 total, the
        homed words and their running counts, the word holding the index and its
        index-th set bit by halving popcounts (no shuffles, no ``fns``). The same
        ranks as the word-serial walk."""
        lane = thread()[2]
        info = view(sm.iterator, self.off_misc + 2688, Int32, 4)
        die, k = (info[0], info[3])
        grid: cutlass.Constexpr = self.grid
        words: cutlass.Constexpr = (grid + 31) // 32
        p = lane >> 3
        bits = view(sm.iterator, self.off_misc + 2704 + 48 * p, Uint32, 12)
        masks = [(1 << min(32, grid - 32 * w)) - 1 for w in range(words)]
        word = [bits[w] & Uint32(masks[w]) for w in range(words)]
        h1 = Int32(0)
        for w in cutlass.range_constexpr(words):
            h1 += Int32(cute.arch.popc(word[w]))
        own, other = (grid - h1, h1)
        if die == 1:
            own, other = (h1, grid - h1)
        target, index = (die, k)
        if k >= own:
            target, index = (1 - die, other - 1 - (k - own))
        before, hit, base = (Int32(0), Uint32(0), Int32(-1))
        for w in cutlass.range_constexpr(words):
            homed = word[w]
            if target == 0:
                homed = ~word[w] & Uint32(masks[w])
            count = Int32(cute.arch.popc(homed))
            if (index >= before) & (index < before + count):
                hit, base = (homed, 32 * w + index - before)
            before += count
        rank = Int32(-1)
        if base >= 0:
            # the (base % 32)-th set bit of hit, by halves
            n, position = (base % 32, Int32(0))
            for width in cutlass.range_constexpr(5):
                half: cutlass.Constexpr = 16 >> width
                low = Int32(cute.arch.popc(hit & Uint32((1 << half) - 1)))
                if n >= low:
                    n, position = (n - low, position + half)
                    hit = hit >> Uint32(half)
            rank = base - base % 32 + position
        ranks = view(sm.iterator, self.off_misc + 2904, Int32, 4)
        if (lane & 7 == 0) & (p < 4):
            ranks[p] = rank

    @cute.jit
    def published_ranges(self, sm):
        """``matched_ranges`` (``gcut``) from the 14 words the matched ranks' threads published (misc 2928: FC1(A) columns 0-3 + rank
        in 6, FC2(A) columns 4-5; FC1(B) / FC2(B) at +7)."""
        table = view(sm.iterator, self.off_misc + 2928, Int32, 14)
        ranges = cute.make_rmem_tensor((2, 7), Int32)
        for g in cutlass.range_constexpr(2):
            for i in cutlass.range_constexpr(7):
                ranges[g, i] = table[7 * g + i]
        return ranges

    @cute.jit
    def matched_ranges(self, sm, units):
        """``partition`` with each list's ranges from this CTA's matched rank of that
        list (column 6: the FC1 rank, for ``partials``). Ranges, owners and partial-sum
        order are those of the die-oblivious partition."""
        ranks = view(sm.iterator, self.off_misc + 2904, Int32, 4)
        ranges = cute.make_rmem_tensor((2, 7), Int32)
        for g in cutlass.range_constexpr(2):
            fc1 = self.partition(units, ranks[g])
            fc2 = self.rank_ranges(sm, units, ranks[2 + g])
            for i in cutlass.range_constexpr(4):
                ranges[g, i] = fc1[g, i]
            ranges[g, 4], ranges[g, 5] = (fc2[g, 4], fc2[g, 5])
            ranges[g, 6] = ranks[g]
        return ranges

    @cute.jit
    def partials(self, workspace, values, tile, ranges, batch):
        """``ExpertsKernel.partials`` over FC1 partition ranks (``dies``: this CTA's
        matched rank of the tile's group, ``matched_ranges`` column 6, in place of the
        CTA index): the same slots, owners, peers and summation order."""
        if cutlass.const_expr(not self.dies):
            return ExpertsKernel.partials(self, workspace, values, tile, ranges, batch)
        ws = workspace.iterator
        _, warp, lane, _ = thread()
        row, slots = (32 * (warp & 3) + lane, self.slots)
        group = Int32(tile >= self.tiles * ranges[1, 0])
        start = (tile - self.tiles * ranges[group, 0]) * 14
        mine = ranges[group, 6]
        owner = start >= ranges[group, 2]
        if not owner:
            offset = self.ws_partial + ((Int64(group) * slots + mine) * 1024 + row) * 4
            for t in cutlass.range_constexpr(self.width):
                store_words(ws + offset + t * 512, (partial_bits(values[t]),), scope="gpu")
        else:
            peer, size = (mine + 1, ranges[group, 1])
            while (peer < self.grid) & (peer * size // self.grid < start + 14):
                if (peer + 1) * size // self.grid != peer * size // self.grid:
                    source = ws + self.ws_partial
                    source += ((Int64(group) * slots + peer) * 1024 + row) * 4
                    words = cute.make_rmem_tensor(self.width, Uint32)
                    ready = False
                    while not ready:
                        bad = False
                        for t in cutlass.range_constexpr(self.width):
                            words[t] = load_volatile(source + t * 512)[0]
                            bad |= words[t] == Uint32(4294967295)
                        ready = cute.arch.vote_all_sync(not bad)
                    for t in cutlass.range_constexpr(self.width):
                        values[t] += bits_float(words[t])
                        store_words(source + t * 512, (Uint32(4294967295),), scope="gpu")
                peer += 1
        return owner

    @cute.jit
    def finalize(self, workspace, weights, batch, box, sm=None):
        """``ExpertsKernel.finalize`` with the lines over ten warps per CTA (the hidden
        loader, the gather warp and the eight epilogue warps): 64 tokens x 56 lines
        over 8 warps gave the first CTAs a third line (a measurably later kernel end);
        a poll that finds a row unwritten sleeps before it retries. The line's routing
        weights and (two-shot) its owner's slot address are loaded before the poll:
        after the line's last row lands only the sum and the push remain (two
        dependent global loads there measurably delay the kernel's end). ``sm``: unused
        here (the wide engines' same-die finalize reads the CTA's census rank)."""
        _, warp, lane, cta = thread()
        for line in range(cta * 10 + warp - 2, batch * 56, self.grid * 10):
            token, column = line // 56, line % 56 * 64
            source = workspace.iterator + self.ws_stage
            source += (Int64(token) * 16 * 1792 + column // 2 + lane) * 4
            words = cute.make_rmem_tensor(16, Uint32)
            scale = cute.make_rmem_tensor(16, Float32)
            for slot in cutlass.range_constexpr(16):
                scale[slot] = Float32(weights[token * 16 + slot])
            target = Int64(0)
            if cutlass.const_expr(self.publish == "twoshot"):
                recv, peers, rslot = box[0:3]
                width: cutlass.Constexpr = recv.shape[2]
                chunk = column + 8 * (lane & 7)
                owner = chunk // width
                place = cute.crd2idx((rslot, token, chunk - owner * width), recv.layout)
                target = Int64(peers[owner]) + Int64(place) * 2
            ready = False
            while not ready:
                bad = False
                for slot in cutlass.range_constexpr(16):
                    words[slot] = load_volatile(source + slot * 7168)[0]
                    bad |= words[slot] == Uint32(4294967295)
                ready = cute.arch.vote_all_sync(not bad)
                if not ready:
                    sleep_ns(64)
            low, high = Float32(0), Float32(0)
            for slot in cutlass.range_constexpr(16):
                store_words(source + slot * 7168, (Uint32(4294967295),), scope="gpu")
                low = fma(bf16(words[slot]), scale[slot], low)
                high = fma(bf16(words[slot], True), scale[slot], high)
            word = pack_pair(low, high, policy="latent")
            packed = [shuffle(word, 4 * (lane & 7) + i) for i in range(4)]
            if lane < 8:
                if cutlass.const_expr(self.publish == "twoshot"):
                    store_relaxed(target, packed)  # ``twoshot.push``, address ahead
                else:
                    mc, parity, rank = box[0:3]
                    offset = ((parity * self.tp + rank) * self.box_rows + token) * 3584 + column
                    offset += 8 * lane
                    multicast_words(mc + Int64(offset) * 2, packed)

    def staged_word(self, pair, tile, warp):
        """Word offset in the staged rows of the 16 words (columns 128 tile +
        32 (warp & 3) + [0, 32)) that epilogue warp ``warp`` stages of ``pair``'s FC2
        row: [pair][3584] BF16 here."""
        return Int64(pair) * 1792 + tile * 64 + (warp & 3) * 16

    @cute.jit
    def fc2(self, sm, workspace, output, values, unit, tile, ranges):
        """``ExpertsKernel.fc2``; published, through a shared transpose: each lane puts
        its row's 8 BF16 values (``store_stage_half``'s bits) at their ``pack_pair``
        positions in its warp's [8 tokens, 64 B] buffer, then stores 16 B of one
        token's 64 B row segment (``staged_word``): the same bytes at the same
        addresses as one 2-byte store per value, in 1 vector store per lane instead of
        8 (measured faster per call at M16 / 32 / 64)."""
        if cutlass.const_expr(not self.lean):
            if cutlass.const_expr(self.lean_local):
                self.fc2_transposed(sm, output, values, unit, tile)
            else:
                _, warp, lane, _ = thread()
                rows = self.tables(sm)[1]
                for t in cutlass.range_constexpr(8):
                    pair = Int32(rows[unit * 8 + t])
                    if pair >= 0:
                        other = cute.arch.shuffle_sync_down(values[t], 8)
                        if lane & 8 == 0:
                            column = tile * 128 + (warp & 3) * 32
                            column += 4 * (lane % 8) + lane // 8
                            dst = output.iterator + Int64(pair) * 3584 + column
                            store_words(dst, (bf16x2(values[t], other),))
        else:
            _, warp, lane, _ = thread()
            low = lane % 16
            pos = (lane - 8 * (low // 8)) % 16
            half = 4 * (4 * (pos // 2) + 2 * (pos % 2) + lane // 16) + 2 * (low // 8)
            buffer = sm.iterator + self.off_fc2s + (warp - 4) * 512
            cute.arch.sync_warp()  # the previous tile's reads of the buffer
            for t in cutlass.range_constexpr(8):
                cell = shared_address(buffer + t * 64 + half)
                store_shared_if(cell, stage_bits(values[t]), Uint32(1), kind="u16")
            cute.arch.sync_warp()
            t = lane // 4
            word = self.tables(sm)[2][4 * unit + (t >> 1)]
            if t & 1 == 0:
                word = word << 16
            pair = Int32(word) >> 16  # the Int16 pair index, sign-extended
            part = lane % 4 * 16
            chunk = load_words(
                shared_address(buffer + t * 64 + part), count=4, shared=True
            )
            if pair >= 0:
                row = self.staged_word(pair, tile, warp)
                store_words(workspace.iterator + self.ws_stage + row * 4 + part, chunk, scope="gpu")

    @cute.jit
    def fc2_transposed(self, sm, output, values, unit, tile):
        """``lean_local``: ``fc2``'s local rows through the published path's transpose. Each lane puts its row's 8 BF16 values
        (``bf16x2``'s low half: the bits of the eight-store form) at their positions in its warp's [8 tokens, 64 B] buffer; those
        positions are the local row's column order (lane L's value is column 4 (L % 8) + L // 8 of the warp's 32), so each
        lane then stores 16 B of one token's 64 B row segment: the same bytes at the same addresses as the eight 4-byte
        stores per lane of the default form (faster at 16-48 rows, measured on one GPU). The buffer is the
        warp's own; bar.warp.sync orders its writes before the other lanes' reads and the previous item's reads before
        them (PTX ISA, bar.warp.sync memory ordering among the participating threads)."""
        _, warp, lane, _ = thread()
        low = lane % 16
        pos = (lane - 8 * (low // 8)) % 16
        half = 4 * (4 * (pos // 2) + 2 * (pos % 2) + lane // 16) + 2 * (low // 8)
        buffer = sm.iterator + self.off_fc2s + (warp - 4) * 512
        cute.arch.sync_warp()  # the previous item's reads of the buffer
        for t in cutlass.range_constexpr(8):
            cell = shared_address(buffer + t * 64 + half)
            store_shared_if(cell, bf16x2(values[t], Float32(0.0)), Uint32(1), kind="u16")
        cute.arch.sync_warp()
        t = lane // 4
        word = self.tables(sm)[2][4 * unit + (t >> 1)]
        if t & 1 == 0:
            word = word << 16
        pair = Int32(word) >> 16  # the Int16 pair index, sign-extended
        part = lane % 4 * 16
        chunk = load_words(shared_address(buffer + t * 64 + part), count=4, shared=True)
        if pair >= 0:
            dst = output.iterator + Int64(pair) * 3584 + tile * 128 + (warp & 3) * 32 + lane % 4 * 8
            store_words(dst, chunk)

    @cute.kernel
    def kernel_maps(self, args, w13_map: GridMap, w2_map: GridMap):
        self.body(args, (map_address(w13_map), map_address(w2_map)))

    @cute.jit
    def body(self, args, maps):
        """``ExpertsKernel.kernel`` (grouped, not pregrouped); ``maps``: the (w13, w2)
        descriptor addresses."""
        x, ids, _, w13_scales, w2_scales, workspace, output, batch, *scalars = args
        tid, warp, lane, cta = thread()
        sm = alloc(cutlass.utils.SmemAllocator(), Uint8, self.smem_bytes, 1024)
        tmem_base = self.initialize(sm, maps, batch)
        cute.arch.griddepcontrol_wait()
        counters = view(workspace.iterator, self.ws_counter, Int32, self.slots)
        # Images and rearm alternate over grouped calls; ``x`` is the plan.
        routed = view(sm.iterator, self.off_misc + 1140, Int32, 3)
        arrival = None
        if cutlass.const_expr(len(self.dies) > 0):
            arrival = self.ticket(workspace)
        if cutlass.const_expr(self.pst_nb > 0):  # K31MK_MOE_RUBIN64_PRESTREAM
            self.group(sm, ids, workspace, batch, arrival=arrival, pst=(maps[0], w13_scales))
        else:
            self.group(sm, ids, workspace, batch, arrival=arrival)
        distinct, call_count, generation = (routed[0], routed[1], routed[2])
        experts = self.tables(sm)[0]
        fence_tmem()
        tmem = tmem_base[0]
        if cutlass.const_expr(self.k19_fence):  # K-19 (``K31MK_K19_FENCE`` rubin64; off)
            cute.arch.fence_acq_rel_gpu()
        cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(len(self.dies) > 0):
            if cutlass.const_expr(self.gcut):
                ranges = self.published_ranges(sm)
            else:
                ranges = self.matched_ranges(sm, distinct)
        else:
            ranges = self.partition(distinct, cta)
        if warp == 0:
            if lane == 0:
                weights = (sm, maps, w13_scales, w2_scales, ranges, experts)
                self.stream(*weights, (workspace, generation & 1))
        else:
            if warp == 1:
                self.consume(sm, tmem, ranges)
            elif warp == 2:
                self.load_hidden(sm, workspace, ranges, generation & 1)
                if cutlass.const_expr(self.publish):
                    self.finalize(workspace, output, batch, scalars[3:])
            elif warp == 3:
                self.gather_x(sm, x, ranges, batch)
                self.rearm(workspace, generation)
                if lane == 0:
                    self.claims(sm, workspace, ranges, generation & 1)
                cute.arch.sync_warp()
                if cutlass.const_expr(self.publish):
                    self.finalize(workspace, output, batch, scalars[3:])
            else:
                if cutlass.const_expr(self.publish == "push"):  # before any accumulator
                    self.push_shared(scalars[3:9], batch, tid - 128, cta, 256)
                self.epilogue(
                    sm, workspace, output, tmem, ranges, batch, generation & 1, scalars
                )
        release_tmem(tmem, 512, warp, 3)
        if tid == 0:
            counters[cta] = call_count + 1
            if cta < self.slots - self.grid:
                counters[self.grid + cta] = call_count + 1
            grouped = view(workspace.iterator, self.ws_calls, Int32, self.slots + 2)
            grouped[cta] = generation + 1
            if cta < self.slots - self.grid:
                grouped[self.grid + cta] = generation + 1
            if cta == 0:
                grouped[self.slots + (generation & 1)] = distinct
