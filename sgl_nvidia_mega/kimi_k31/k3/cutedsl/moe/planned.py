# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Re-planned routed experts for B <= 8 at TP8 (opt-in ``K31MK_EXPERTS_PLANNED``, default off).

Same arithmetic as ``experts.ExpertsKernel`` (the same MXFP8 x, FC1 / FC2 MMA stages,
SiTU and hidden quantization, output epilogue), a different schedule:

- FC1 works on whole tiles: tile t = (distinct expert t // 6, 128-row N tile t % 6), all
  14 K stages on one CTA, so no split-K partials go through global memory. CTA c takes
  tile c first (block index: unique even if CTAs co-reside), then claims tiles G, G + 1,
  ... (G = grid) from a second counter, so a CTA slowed by work sharing its SM (the shared
  expert's side stream, NVLink CTAs) takes fewer tiles instead of straggling.
- FC2 works on tile pairs (expert, pair p of 14; three 16 KB stages) claimed through one
  relaxed GPU-scope counter in expert order, i.e. in the order their hidden images become
  ready (expert e's tiles finish in wave (6e + 5) // G). A CTA claims pairs only after it
  has issued its own FC1 tiles, so early finishers take the early-ready experts and the
  load evens out to about one pair; the next claim is in flight while a pair issues.
- The producer queues each new FC2 unit for the hidden loader (``push_unit``), so the
  loader polls exactly the hidden images this CTA's FC2 stages need, in order.

Protocol state (workspace): hidden images and per-CTA call counters as the shipped
non-grouped engine (so the two engines can share one workspace). Appended at ``ws_claim``:
the claim counters (two parities, baseline -1 = the 0xFF scratch fill) and this engine's
own per-CTA call counters, so shipped calls in between do not shift its parity. Planned
call k uses claim parity k & 1; at its start (after the dependency wait) CTA 0's producer
re-arms parity (k + 1) & 1, which call k - 1 used (complete before call k's wait returned)
and call k + 1 reads only after its own wait. ``scratch(..., planned=True)`` sizes the
workspace."""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32, Uint64

from ..common.primitives import acquire_tensormap, asm, global_timer_ns, store_words, thread, view
from ..gemm.splitk import large_smem_alloc
from .experts import MAX_CTAS, ExpertsKernel, claim


def _pad_2m(nbytes):
    # Integrated tree: w3moe's opt-in die-split experts (which define this helper and
    # PersistentExperts.dielocal) are not in it; the die-split path stays unreachable.
    return -(-nbytes // (2 << 20)) * (2 << 20)

@asm()
def load_tma3_keep(dst, descriptor, barrier, row, chunk):
    """3-D TMA tile load without the evict-first hint (x is read by every CTA)."""
    code = "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier"
    code += "::complete_tx::bytes [$0],[$1,{0,$3,$4}],[$2];"
    return code, "r,l,r,r,r", (dst, descriptor, barrier, Int32(row), Int32(chunk))


# claim counters (int32 [16]: FC2 [2 parities], FC1 [2]; die-split: index 4 + parity * 4 +
# kind * 2 + die), call counters [MAX_CTAS] at +64, then the SM -> slot table (1040 B)
CLAIM_BYTES = 64 + 4 * MAX_CTAS + 1040


def planned_workspace_bytes(base_bytes):
    """Workspace size with the claim counters appended."""
    return base_bytes + CLAIM_BYTES


class PlannedExpertsKernel(ExpertsKernel):
    """TP8, capacity <= 8, non-grouped. ``stages``: weight ring depth in 32 KB SMEM slots
    (5 = the shipped layout; more needs the large-SMEM mode, see ``large_smem_alloc``)."""

    TPS = (8,)  # subclasses that also handle TP16 (the opt-in W4A4 engine) widen it

    def __init__(
        self, tp, capacity, experts, grid, publish=False, activation_floor=None, stages=5,
        xplan=False, wide=False, chain2=False, x_delay_ns=0, die_split=False,
    ):
        super().__init__(
            tp, capacity, experts, grid, publish, True, activation_floor=activation_floor
        )
        # ``xplan``: x arrives as the MXFP8 plan (``ExpertsPlanKernel``) and each CTA fetches
        # its SMEM image with one x TMA and one scale TMA per K stage (maps 6 and 7) onto
        # xready[stage], instead of loading FP32 x and quantizing it in every CTA.
        self.xplan = xplan
        self.wide = wide  # experimental K=64 MMAs (experts.stage), gated by the FP64 check
        # two FC1 accumulate chains (odd K=32 steps into a shadow accumulator at +384 TMEM
        # columns, free above the x scales); the epilogue adds the two FP32 partials
        self.chain2 = chain2
        # Benchmark-only (E3 X8POST, default 0): with xplan, warp 2 issues the x TMAs
        # ``x_delay_ns`` after this CTA's entry instead of the producer at its start, so the
        # prologue, routing and ring fill run before "the latent lands"; outputs unchanged.
        self.x_delay_ns = x_delay_ns
        assert not x_delay_ns or xplan
        # ``die_split`` (opt-in): weights in w3moe's die-major layout (``die_major``: FC1 tiles
        # 3d..3d+2 and FC2 tiles 14d..14d+13 of every expert in die d's localized half); a CTA
        # claims FC1 tiles and FC2 pairs from its own die's counters first, then steals from
        # the other die's, so dies with unequal CTA counts finish together.
        self.die_split = die_split
        if die_split:
            self.s13_half = _pad_2m(experts * self.w13_scale_bytes // 2)
            self.s2_half = _pad_2m(experts * self.w2_scale_bytes // 2)
        assert tp in self.TPS and capacity <= 8 and not self.grouped, (tp, capacity)
        # FC2 always on 16 KB tile-pair stages; the unit queue (producer -> hidden loader)
        # of the shipped stealing path carries the claimed units.
        self.paired, self.jobs, self.steal = True, 14, True
        self.ws_claim = self.workspace_bytes
        self.workspace_bytes = planned_workspace_bytes(self.workspace_bytes)
        self.ws_dietable = self.ws_claim + 64 + 4 * MAX_CTAS  # read by ExpertsKernel.die_info
        if stages != self.stages:
            self.relayout(stages)

    def relayout(self, stages):
        """The shipped SMEM layout with ``stages`` ring slots; the ring metadata moves to
        misc + 4224 (free; misc + 520.. would reach the accumulator metadata at 600)."""
        assert 5 <= stages <= 6, stages
        self.stages = stages
        self.off_sfa = stages * 32768
        self.off_x = self.off_sfa + stages * 1024
        self.off_h = self.off_x + 28672
        self.off_xsf = self.off_h + 2 * self.h_image_bytes
        self.off_hsf = self.off_xsf + 28 * 128 + 384
        self.off_misc = self.off_hsf + 2 * self.h_scale_bytes
        self.smem_bytes = self.off_misc + 5120
        self.bar_full, self.bar_empty, self.bar_accfull = (0, 8 * stages, 16 * stages)
        acc = self.bar_accfull
        self.bar_accempty, self.bar_hfull = (acc + 128, acc + 256)
        self.bar_hempty, self.bar_xready = (acc + 272, acc + 288)
        self.off_meta = self.off_misc + 4224
        self.off_queue = self.off_misc + 4608
        # tcgen05 SMEM descriptors encode the start address in 14 bits of addr >> 4: every
        # MMA / tcgen05.cp operand (ring, scales, x, hidden) must lie below 256 KB, so the
        # oversized mode buys one slot (7 slots would put x past 256 KB).
        assert self.off_misc <= 1 << 18, self.off_misc
        assert acc + 288 + 14 * 8 <= 600 and self.smem_bytes <= large_smem_alloc()

    @cute.jit
    def initialize(self, sm, maps, batch):
        """The shipped ``initialize`` with every barrier of a ``stages``-slot layout
        (full, empty, accumulator full/empty, hidden full/empty, x ready)."""
        tid, _, _ = cute.arch.thread_idx()
        count_all: cutlass.Constexpr = 2 * self.stages + 50
        barriers = view(sm.iterator, self.off_misc, Uint64, count_all)
        tmem_base = view(sm.iterator, self.off_misc + 856, Uint32, 1)
        if cutlass.const_expr(self.x_delay_ns > 0):
            if tid == 0:
                stash = view(sm.iterator, self.off_misc + 4400, Int64, 2)
                stash[0] = Int64(maps.iterator.toint())
                stash[1] = global_timer_ns()
        if tid == 0:
            for i in range(count_all):
                acc = 2 * self.stages
                count = 1 if not acc <= i < acc + 32 else 2 if i < acc + 16 else 4
                xcount = 1 if self.xplan else batch * 8
                count = xcount if acc + 36 <= i < acc + 50 else count
                cute.arch.mbarrier_init(barriers.iterator + i, count)
            queue = view(sm.iterator, self.off_misc + self.bar_queue, Uint64, 16)
            for i in cutlass.range_constexpr(16):
                cute.arch.mbarrier_init(queue.iterator + i, 1)
            cute.arch.mbarrier_init_fence()
            acquire_tensormap(maps.iterator)
            acquire_tensormap(maps.iterator + 128)
            acquire_tensormap(maps.iterator + 640)
            if cutlass.const_expr(self.xplan):
                acquire_tensormap(maps.iterator + 768)
                acquire_tensormap(maps.iterator + 896)
        if tid // 32 == 3:
            cute.arch.alloc_tmem(512, tmem_base.iterator)
            cute.arch.relinquish_tmem_alloc_permit()
        if cutlass.const_expr(not self.xplan):  # x TMAs zero-fill the padded token rows
            self.clear_x(sm, tid, batch)
        cute.arch.fence_proxy("async.shared", space="cta")
        return tmem_base

    @cute.jit
    def load_x(self, x, raw, batch):
        if cutlass.const_expr(not self.xplan):
            ExpertsKernel.load_x(self, x, raw, batch)

    @cute.jit
    def quantize_x(self, sm, raw, barriers, batch):
        if cutlass.const_expr(not self.xplan):  # xplan: the producer fetches x (fetch_x)
            ExpertsKernel.quantize_x(self, sm, raw, barriers, batch)
        elif cutlass.const_expr(self.x_delay_ns > 0):
            if thread()[0] == 64:
                stash = view(sm.iterator, self.off_misc + 4400, Int64, 2)
                start = stash[1]
                while global_timer_ns() - start < Int64(self.x_delay_ns):
                    pass
                self.fetch_x(sm, stash[0])

    @cute.jit
    def fetch_x(self, sm, maps):
        """``xplan``: stage b's 8-token MXFP8 slice (2 KB of 128B-swizzled atoms, map 6) and
        its scale words (256 B, map 7) onto xready[b]; rows >= batch are zero-filled."""
        ready = view(sm.iterator, self.off_misc + self.bar_xready, Uint64, 14)
        for b in cutlass.range(14, unroll=1):
            bar = ready.iterator + b
            cute.arch.mbarrier_arrive_and_expect_tx(bar, 2048 + 256)
            load_tma3_keep(sm.iterator + self.off_x + b * 2048, maps + 768, bar, 0, 2 * b)
            load_tma3_keep(sm.iterator + self.off_xsf + b * 256, maps + 896, bar, 0, 2 * b)

    @cute.jit
    def partition(self, distinct: Int32, cta: Int32, die=None):
        """Only the distinct count is needed: [0, 0] = D."""
        ranges = cute.make_rmem_tensor((2, 6), Int32)
        ranges.fill(0)
        ranges[0, 0] = distinct
        return ranges

    @cute.jit
    def partials(self, workspace, values, tile, ranges, batch):
        """Whole FC1 tiles: this CTA owns every tile it accumulates."""
        return True

    @cute.jit
    def claim_counter(self, workspace, parity):
        return workspace.iterator + self.ws_claim + 4 * parity

    def calls(self, workspace):
        return view(workspace.iterator, self.ws_claim + 64, Int32, MAX_CTAS)

    @cute.jit
    def advance_calls(self, calls, cta, call):
        """This CTA's planned call counter, and the mirror entry for larger grids."""
        calls[cta] = call + 1
        if cta < MAX_CTAS - self.grid:
            calls[self.grid + cta] = call + 1

    @cute.jit
    def fc1_tile(self, ring, state, maps, w13_scales, experts, t, lo, hi):
        """K stages [lo, hi) of FC1 tile ``t`` (opens at 0, closes at 13)."""
        slot, phase, issued = state
        unit, r = (t // self.tiles, t % self.tiles)
        expert = experts[unit]
        row = expert * self.w13_rows + r * 128
        base = Int64(expert) * self.w13_scale_bytes + r * 14336
        for ks in cutlass.range(lo, hi, unroll=1):
            c = Int32(ks == 0) | Int32(ks == 13) << 1
            tma = (maps.iterator, w13_scales, row, 2 * ks, base + ks * 1024, Int32(1024))
            meta = (Int32(0), t, ks, c)
            slot, phase, issued = self.issue(ring, slot, phase, issued, meta, tma, False)
        return slot, phase, issued

    @cute.jit
    def stream(self, sm, maps, w13_scales, w2_scales, ranges, experts, steal=None):
        """Producer lane: this CTA's FC1 tiles (wave order), then claimed FC2 pairs."""
        workspace = steal[0]
        cta = thread()[3]
        if cutlass.const_expr(self.xplan and self.x_delay_ns == 0):
            self.fetch_x(sm, maps.iterator)
        if cutlass.const_expr(self.die_split):
            self.stream_split(sm, maps, w13_scales, w2_scales, ranges, experts, workspace)
        else:
            self.stream_mixed(sm, maps, w13_scales, w2_scales, ranges, experts, workspace)

    @cute.jit
    def stream_mixed(self, sm, maps, w13_scales, w2_scales, ranges, experts, workspace):
        """Interleaved weights: first FC1 tile by block index, then claims."""
        cta = thread()[3]
        barriers = view(sm.iterator, self.off_misc, Uint64, 2 * self.stages)
        metadata = view(sm.iterator, self.off_meta, Int32, 4 * self.stages)
        ring = (sm, barriers, metadata, w2_scales)
        state = (Int32(0), Int32(0), Int32(0))  # slot, phase, issued
        distinct = ranges[0, 0]
        tiles = self.tiles * distinct
        # First tile by CTA index (unique, no claim latency at start): its first ring's
        # worth of stages issue before the counters are read; later tiles are claimed in
        # tile order from G, the next claim in flight while a tile issues.
        t = cta
        head: cutlass.Constexpr = min(self.stages, 14)
        if t < tiles:
            state = self.fc1_tile(ring, state, maps, w13_scales, experts, t, 0, head)
        calls = self.calls(workspace)
        call = calls[cta]
        parity = call & 1
        if cta == 0:
            store_words(self.claim_counter(workspace, parity ^ 1), (Uint32(4294967295),))
            store_words(self.claim_counter(workspace, 2 + (parity ^ 1)), (Uint32(4294967295),))
        tile_counter = self.claim_counter(workspace, 2 + parity)
        pending = claim(tile_counter, 1)
        if t < tiles:
            state = self.fc1_tile(ring, state, maps, w13_scales, experts, t, head, 14)
            t = self.grid + Int32(pending + Uint32(1))
            if t < tiles:
                pending = claim(tile_counter, 1)
        while t < tiles:
            state = self.fc1_tile(ring, state, maps, w13_scales, experts, t, 0, 14)
            t = self.grid + Int32(pending + Uint32(1))
            if t < tiles:
                pending = claim(tile_counter, 1)
        slot, phase, issued = state
        counter = self.claim_counter(workspace, parity)
        pairs = self.jobs * distinct
        pending = claim(counter, 1)
        pushed, current, active = (Int32(0), Int32(-1), True)
        while active:
            q = Int32(pending + Uint32(1))
            active = q < pairs
            if active:
                ahead = claim(counter, 1)
                unit, pair = (q // self.jobs, q % self.jobs)
                if unit != current:
                    pushed = self.push_unit(sm, pushed, unit)
                    current = unit
                expert = experts[unit]
                work = (maps.iterator, expert, unit, 2 * pair)
                slot, phase, issued = self.fc2_pair(ring, slot, phase, issued, *work)
                pending = ahead
        self.push_unit(sm, pushed, Int32(-1))
        if issued >= self.stages:
            self.sync(barriers.iterator + self.stages + slot, phase ^ 1)
        for i in cutlass.range_constexpr(4):
            metadata[4 * slot + i] = Int32(2 if i == 0 else 0)
        cute.arch.mbarrier_arrive(barriers.iterator + slot, arrive_count=1)
        self.advance_calls(calls, cta, call)

    @cute.jit
    def stream_split(self, sm, maps, w13_scales, w2_scales, ranges, experts, workspace):
        """Die-major weights: FC1 tiles, then FC2 pairs, each claimed from this CTA's die's
        counter and then the other die's. Metadata keeps the global tile ids, so consumer,
        epilogue and hidden images are unchanged. A CTA's FC2 units may repeat across the two
        passes; producer and loader switch hidden buffers on every change, so that is fine."""
        cta = thread()[3]
        die = self.die_info(workspace)[0]
        barriers = view(sm.iterator, self.off_misc, Uint64, 2 * self.stages)
        metadata = view(sm.iterator, self.off_meta, Int32, 4 * self.stages)
        ring = (sm, barriers, metadata, w2_scales)
        slot, phase, issued = (Int32(0), Int32(0), Int32(0))
        distinct = ranges[0, 0]
        calls = self.calls(workspace)
        call = calls[cta]
        parity = call & 1
        if cta == 0:
            for i in cutlass.range_constexpr(4):
                store_words(self.claim_counter(workspace, 4 + (parity ^ 1) * 4 + i), (Uint32(4294967295),))
        for side in cutlass.range_constexpr(2):  # own die, then the other die
            d = die if side == 0 else 1 - die
            counter = self.claim_counter(workspace, 4 + parity * 4 + d)
            pending = claim(counter, 1)
            t = Int32(pending + Uint32(1))
            while t < 3 * distinct:
                ahead = claim(counter, 1)
                unit, local = (t // 3, t % 3)
                expert = experts[unit]
                row = (d * self.experts + expert) * 384 + local * 128
                base = Int64(d) * self.s13_half + Int64(expert) * 43008 + local * 14336
                tile = unit * self.tiles + 3 * d + local
                for ks in cutlass.range(14, unroll=1):
                    c = Int32(ks == 0) | Int32(ks == 13) << 1
                    tma = (maps.iterator, w13_scales, row, 2 * ks, base + ks * 1024, Int32(1024))
                    slot, phase, issued = self.issue(
                        ring, slot, phase, issued, (Int32(0), tile, ks, c), tma, False
                    )
                t = Int32(ahead + Uint32(1))
        pushed, current = (Int32(0), Int32(-1))
        for side in cutlass.range_constexpr(2):
            d = die if side == 0 else 1 - die
            counter = self.claim_counter(workspace, 4 + parity * 4 + 2 + d)
            pending = claim(counter, 1)
            q = Int32(pending + Uint32(1))
            while q < 7 * distinct:
                ahead = claim(counter, 1)
                unit, local = (q // 7, q % 7)
                if unit != current:
                    pushed = self.push_unit(sm, pushed, unit)
                    current = unit
                expert = experts[unit]
                rows = (d * self.experts + expert) * 1792
                at = (rows + 256 * local, Int64(d) * self.s2_half + Int64(expert) * 21504 + 2 * local * 1536)
                work = (maps.iterator, expert, unit, 14 * d + 2 * local)
                slot, phase, issued = self.fc2_pair(ring, slot, phase, issued, *work, at=at)
                q = Int32(ahead + Uint32(1))
        self.push_unit(sm, pushed, Int32(-1))
        if issued >= self.stages:
            self.sync(barriers.iterator + self.stages + slot, phase ^ 1)
        for i in cutlass.range_constexpr(4):
            metadata[4 * slot + i] = Int32(2 if i == 0 else 0)
        cute.arch.mbarrier_arrive(barriers.iterator + slot, arrive_count=1)
        self.advance_calls(calls, cta, call)

    @cute.jit
    def rearm(self, workspace, call_count):
        """The other parity's 128 hidden images (as the shipped non-grouped engine; the
        claim counter is re-armed by the producer)."""
        _, _, lane, cta = thread()
        items: cutlass.Constexpr = 128 * self.h_compact_bytes // 16
        lo, hi = (cta * items // self.grid, (cta + 1) * items // self.grid)
        other = self.image(workspace, call_count & 1 ^ 1, 0)
        for i in range(lo + lane, hi, 32):
            store_words(other + i * 16, (Uint32(4294967295),) * 4)


# --------------------------------------------------------------------------------- runtime
import functools  # noqa: E402

import torch  # noqa: E402
from cutlass.cute.runtime import make_fake_stream  # noqa: E402

from ..common.host import check_residency, fake  # noqa: E402
from . import experts_runtime as ER  # noqa: E402
from ..common.primitives import load_words  # noqa: E402
from .experts import PLAN_BYTES, PLAN_XSF, ExpertsGeometry, ExpertsPlanKernel, plan_chunk  # noqa: E402


@asm()
def fence_proxy_async_global():
    """Bi-directional generic <-> async proxy fence for global memory (PTX 9.7.15.4)."""
    return "fence.proxy.async.global;", "~{memory}", ()


class EarlyPlanKernel(ExpertsPlanKernel):
    """``ExpertsPlanKernel`` (FP32 x -> MXFP8 plan, one CTA per token) that releases its
    dependents at entry, so the experts kernel launches and runs its pre-wait prologue
    (barriers, TMEM, tensormap prefetch) beside it; the experts still read the plan only
    after their ``griddepcontrol.wait`` (completion of this grid). The shipped
    ``ExpertsGroupKernel(early=True)`` pattern."""

    @cute.kernel
    def kernel(self, args):
        x, plan = args
        tid, _, _, token = thread()
        cute.arch.griddepcontrol_launch_dependents()
        cute.arch.griddepcontrol_wait()
        # The previous experts call read this plan through TMA (async proxy): order those
        # reads before these generic overwrites (WAR across proxies).
        fence_proxy_async_global()
        for i in cutlass.range_constexpr(2):
            chunk = tid + 256 * i
            if chunk < 448:  # warp-uniform: chunks 256-447 are warps 0-5
                source = x.iterator + token * 3584 + chunk * 8
                words = (*load_words(source, count=4, mutable=True),
                         *load_words(source + 4, count=4, mutable=True))
                plan_chunk(plan, token, chunk, words, fp32=True)
        # The experts kernel reads these generic stores with TMA after its
        # griddepcontrol.wait; griddepcontrol is not a proxy-preserving boundary in PTX 8.9.4,
        # so every writer fences generic -> async before completing.
        fence_proxy_async_global()


@functools.cache
def compile_early_plan(batch, device):
    args = (fake((batch, 3584), None, cutlass.Float32), fake((PLAN_BYTES,), (1,), cutlass.Uint8),
            make_fake_stream())
    with torch.accelerator.device_index(device):
        return cute.compile(EarlyPlanKernel(batch), *args)


def plan_maps(plan, batch, device):
    """Maps 6 and 7 over the MXFP8 plan: x rows [batch, 3584] as (128 B, rows, 28 atoms),
    128B swizzle, box (128, 8, 2); scale words [batch, 28, 16 B] as (16 B, rows, 28), box
    (16, 8, 2). Rows >= batch are out of bounds (zero-filled)."""
    import cuda.bindings.driver as cuda

    from ..common.host import tensor_maps

    u8 = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8
    sw = cuda.CUtensorMapSwizzle
    none = cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE
    base = plan.data_ptr()
    specs = [
        (base, u8, (128, batch, 28), (3584, 128), (128, 8, 2), sw.CU_TENSOR_MAP_SWIZZLE_128B, none),
        (base + PLAN_XSF, u8, (16, batch, 28), (448, 16), (16, 8, 2), sw.CU_TENSOR_MAP_SWIZZLE_NONE, none),
    ]
    return tensor_maps(specs, device)


def scratch(tp, device):
    """``experts_runtime.scratch`` with the planned engine's claim and call counters
    appended (0xFF-filled: every counter starts at its -1 baseline)."""
    buffers = ER.scratch(tp, device)
    # at least ER.scratch's size
    size = max(planned_workspace_bytes(ExpertsGeometry(tp).workspace_bytes), buffers["workspace"].numel())
    buffers["workspace"] = torch.full((size,), 255, device=device, dtype=torch.uint8)
    return buffers


def set_oversized(code, device):
    """CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE (17) = ALLOW_OVERSIZED (3): with the mode set the
    driver ignores MAX_DYNAMIC_SHARED_SIZE_BYTES and allows a larger per-CTA allocation (cuda.h 13.4).
    Verified to work on a CuTe-compiled kernel."""
    import cuda.bindings.driver as cuda

    from ..common.host import resolve_kernel

    _, function = resolve_kernel(code, device)
    attr = getattr(cuda.CUfunction_attribute, "CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE", None)
    attr = attr if attr is not None else cuda.CUfunction_attribute(17)
    (status,) = cuda.cuFuncSetAttribute(function, attr, 3)
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"large-SMEM mode: {status}")


@functools.cache
def compile_planned(
    tp, experts, batch, grid, device, publish=False, activation_floor=False, stages=5,
    xplan=False, wide=False, chain2=False, x_delay_ns=0, die_split=False, box_rows=64,
):
    capacity = 1 << (batch - 1).bit_length()
    moe = PlannedExpertsKernel(
        tp, capacity, experts, grid, publish, activation_floor=activation_floor,
        stages=stages, xplan=xplan, wide=wide, chain2=chain2, x_delay_ns=x_delay_ns,
        die_split=die_split,
    )
    moe.box_rows = box_rows  # the tail box's rows (``ExpertsKernel.box_rows``)
    flat = lambda n, dtype=cutlass.Uint8, align=16: fake((n,), (1,), dtype, align)  # noqa: E731
    box = (Int64(0), Int32(0), Int32(0)) if publish else ()
    if publish == "twoshot":
        box = (fake((3 * tp, box_rows, 3584 // tp)), fake((tp,), None, Int64), Int32(0))
    x, ids = fake((batch, 3584), (3584, 1), cutlass.Float32), flat(batch * 16, Int32)
    if xplan:
        x = flat(PLAN_BYTES)
    scales = (flat(experts * n) for n in (moe.w13_scale_bytes, moe.w2_scale_bytes))
    out = flat(batch * 16 * (1 if publish else 3584), cutlass.BFloat16)
    maps = flat(1024 if xplan else 768, align=64)
    tensors = (x, ids, maps, *scales, flat(moe.workspace_bytes), out)
    scalars = (Int32(batch), *map(cutlass.Float32, (4, 25, 0)), *box)
    with torch.accelerator.device_index(device):
        code = cute.compile(moe, *tensors, *scalars, make_fake_stream())
        if moe.smem_bytes > 227 * 1024:
            set_oversized(code, device)
        return code


# Batches the planned engine takes (others keep the shipped engine). Measured (3-run
# ABBA): the planned engine is slower at B1-B2 and faster at B4 and B8
# (stages 6 + xplan). Whole-tile FC1 needs about one tile per CTA, so
# small batches keep the shipped split-K engine. ``K31MK_EXPERTS_PLANNED_BATCHES`` overrides.
PLANNED_BATCHES = "4-8:1"


def enabled():
    """Switch for the MK providers: on by default in the integrated tree (bit-identical
    rows to the shipped engine, measured faster at B4 / B8 on the target GPU); ``K31MK_EXPERTS_PLANNED=0`` restores the shipped
    engine everywhere."""
    import os

    return os.environ.get("K31MK_EXPERTS_PLANNED", "1") == "1"


def options():
    """PlannedExperts options from the environment: ``K31MK_EXPERTS_PLANNED_STAGES`` (default 6,
    the large-SMEM mode; 5 = shipped layout), ``K31MK_EXPERTS_PLANNED_XPLAN`` (default 1:
    x as the MXFP8 plan + TMA; stages 6 without it is measurably slower at B8, the small
    L1 of the oversized mode starves the per-CTA FP32 x loads), ``K31MK_EXPERTS_PLANNED_BATCHES``
    (batch_table spec, default 4-8)."""
    import os

    return {
        "stages": int(os.environ.get("K31MK_EXPERTS_PLANNED_STAGES", "6")),
        "xplan": os.environ.get("K31MK_EXPERTS_PLANNED_XPLAN", "1") == "1",
        "batches": os.environ.get("K31MK_EXPERTS_PLANNED_BATCHES"),
    }


def planned_batches(spec=None):
    import os

    from ..common.policy import batch_table

    spec = spec or os.environ.get("K31MK_EXPERTS_PLANNED_BATCHES", PLANNED_BATCHES)
    return {b for b, on in batch_table(spec).items() if on}


class PlannedExperts(ER.PersistentExperts):
    """``PersistentExperts`` with the planned engine for TP8 B <= 8 (other batches keep
    the shipped engine). Needs ``scratch`` (above) for its workspace. ``stages``: ring
    slots (5 = shipped layout; 6-8 run in the large-SMEM mode, 1 CTA per SM with no
    room for a co-resident PDL neighbour, as the shipped engine at 209 KB)."""

    def __init__(
        self, *args, stages=5, xplan=False, batches=None, wide=False, chain2=False, x_delay_ns=0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.stages, self.xplan, self.wide, self.chain2 = stages, xplan, wide, chain2
        self.x_delay_ns = x_delay_ns
        self.batch_maps = {}
        self.batches = planned_batches(batches)

    def prepare(
        self, batch, buffers, out, publish=(), planned=False, twoshot=False, early=False
    ):
        if self.tp != 8 or batch > 8 or batch not in self.batches:
            return super().prepare(batch, buffers, out, publish, planned, twoshot, early)
        need = planned_workspace_bytes(ExpertsGeometry(self.tp).workspace_bytes)
        if buffers["workspace"].numel() < need:
            raise ValueError("planned experts need planned.scratch() workspace")
        mode = "twoshot" if twoshot else bool(publish)
        grid = self.grid_for(batch)
        code = compile_planned(
            self.tp, self.experts, batch, grid, self.device.index, mode,
            activation_floor=self.activation_floor, stages=self.stages, xplan=self.xplan,
            wide=self.wide, chain2=self.chain2, x_delay_ns=self.x_delay_ns,
            die_split=bool(getattr(self, "dielocal", False)),
            **self.box_kw,
        )
        if self.stages == 5:  # the occupancy query does not see the oversized mode
            check_residency(code, self.device.index, grid, 384, "Kimi K3 planned experts")
        elif grid > torch.cuda.get_device_properties(self.device).multi_processor_count:
            raise RuntimeError(f"planned experts: grid {grid} > SMs")
        if getattr(self, "dielocal", False):  # SM -> slot table at ws_dietable (die_info), counts after it
            base = need - 1040
            table = buffers["workspace"][base : base + 1032].view(torch.int32)
            table[:256].copy_(self.die_table[0])
            table[256:258].copy_(self.die_table[1])
        ws = ER.tensor(buffers["workspace"])
        maps, plan = self.all_maps, None
        if self.xplan:
            # x -> MXFP8 plan by ExpertsPlanKernel (one CTA per token) on the same stream;
            # in the MK the front kernel could write it instead (as the B>8 routing does).
            both = torch.cat((self.maps[:768], plan_maps(buffers["plan"], batch, self.device)))
            self.batch_maps[batch] = both
            maps = ER.to_cute(both, 64)
            plan = (ER.tensor(buffers["plan"]), compile_early_plan(batch, self.device.index))
            if planned:  # the caller (e.g. the MK front) writes the plan before ``run``
                plan = plan[:1]
        self.prepared[batch] = (out, ER.tensor(out), ws, code, publish, plan, maps, None)
