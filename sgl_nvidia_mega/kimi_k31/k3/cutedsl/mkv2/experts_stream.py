# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MKv2 stream-rate routed experts, TP8 B <= 8 (opt-in, default off).

The planned engine (``moe.planned.PlannedExpertsKernel`` with the x plan: same x TMAs, FC1 / FC2
MMA stages, accumulators, SiTU, hidden images, epilogue and arithmetic) with a new producer:

- One ticket queue per call: FC1 tiles (expert, 128-row N tile; 14 K stages) and FC2 groups
  (expert, ``grain`` tile pairs; 3 stages each). ``order="planned"`` lists every FC1 tile, then
  every FC2 group in expert order (the planned engine's order); ``order="lag"`` interleaves the
  FC2 groups of expert e - ``lag`` after the FC1 tiles of expert e. A CTA takes ticket ``cta``
  first, then claims ``grid + k`` from one relaxed GPU-scope counter (parity per call, re-armed
  by CTA 0 as the planned engine's counters), the next claim in flight.
- ``lanes`` warp-converged producer lanes: per pass, lane i issues stage ``base + i`` of the
  current ticket into ring slot ``slot + i`` (its own empty wait, metadata, expect_tx, TMA and
  scale copy), so one pass costs about one lane's instruction path. The cache policy is created
  once, not per copy.
- ``hbufs`` hidden-image buffers (2 = shipped; 4 lets the loader run further ahead).
- ``prefetch``: on claiming its next ticket the producer issues one L2 bulk prefetch of that
  ticket's weight bytes and one of its scales.

Every MMA reads the same operands into the same accumulator in the same order as the planned
engine, so outputs are bitwise equal to it (test_mkv2_experts.py::test_engine).

``ablate`` (benchmark-only, timing diagnosis; outputs WRONG unless empty): "hid" (hidden loader
does not poll the images), "epi" (epilogue releases accumulators, no work; needs "hid"), "mma"
(no MMAs, commits kept), "drain" (MMA warp frees each slot at landing; implies "hid", "epi"),
"scale" (no weight-scale copies).

Sync (memory-model audit; edges L1-1..L1-6): the planned engine's edges
Q1-Q11 / X1 unchanged except: the per-slot ring protocol is now
run by several lanes of one warp, each on its own slot (same full / empty barriers, same
expect_tx, same parity rule); one ticket counter replaces the FC1 / FC2 counters (Q1 / Q2
pattern); hidden buffers 3-4 use barriers at misc + 4800 with the Q8 / Q9 pattern; the prefetch
is a hint with no ordering role."""

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint8, Uint16, Uint32, Uint64
from cutlass.cute.runtime import make_fake_stream

from ..common.host import check_residency, fake
from ..common.primitives import (
    acquire_tensormap,
    alloc,
    asm,
    commit,
    prefetch_l2,
    fence_tmem,
    fp8_bits,
    group_barrier,
    load_volatile,
    mx_rescale,
    release_tmem,
    shared_address,
    sleep_ns,
    smem_desc,
    smem_words,
    store_shared_words,
    store_words,
    thread,
    view,
)
from ..moe import experts_runtime as ER
from ..moe import planned as PLN
from ..moe.experts import MAX_CTAS, ExpertsGeometry, claim, has_ff_byte, hidden_exponent, swizzled
from ..moe.math import routed_situ as situ

shuffle = cute.arch.shuffle_sync
MAP_BYTES = 1024 + 64  # maps 0-7 (planned + x plan), then the w13 / w2 base addresses (prefetch)
HBARS = 4800  # misc offset of the hidden full [hbufs] / empty [hbufs] barriers
ABLATIONS = ("hid", "epi", "mma", "drain", "scale")


@asm(Uint64)
def evict_first_policy():
    return "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;", "=l", ()


@asm()
def tma3(dst, descriptor, barrier, row, chunk, policy):
    """``primitives.load_tma3`` with the evict-first policy as an operand."""
    code = "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
    code += ".L2::cache_hint [$0],[$1,{0,$3,$4}],[$2],$5;"
    return code, "r,l,r,r,r,l,~{memory}", (dst, descriptor, barrier, Int32(row), Int32(chunk), Uint64(policy))


@asm()
def bulk(dst, src, barrier, size, policy):
    """``primitives.copy_evict`` (dynamic size) with the policy as an operand."""
    code = "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
    code += " [$0],[$1],$3,[$2],$4;"
    return code, "r,l,r,r,l,~{memory}", (dst, src, barrier, Int32(size), Uint64(policy))


@asm()
def prefetch(address, size):
    return "cp.async.bulk.prefetch.L2.global [$0], $1;", "l,r,~{memory}", (Int64(address), Int32(size))


@asm()
def fence_acq_rel_gpu():
    """fence.acq_rel.gpu before the publishing W8's trigger (a memory-model audit fix)."""
    return "fence.acq_rel.gpu;", "~{memory}", ()


@asm()
def store_u16_relaxed(pointer, value):
    return "st.relaxed.gpu.global.u16 [$0], $1;", "l,h,~{memory}", (pointer, Uint16(value))


@asm(Uint16)
def load_u16_volatile(pointer):
    return "ld.volatile.global.u16 $0, [$1];", "=h,l,~{memory}", (pointer,)


@asm()
def stage(mma, weight_scales, activation_scales, release, done, *, kind, nomma=False):
    """``moe.experts.stage`` (TP8 kinds fc1 / fc2a / fc2b / fc2p; no wide / chain2), plus
    ``nomma`` (benchmark-only ablation: the MMAs are not issued, copies and commits are)."""
    half = kind in ("fc2b", "fc2p")
    steps = 4 if half else 8
    cp = "tcgen05.cp.cta_group::1.32x128b.warpx4"
    arrive = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
    code = [
        "{ .reg .pred e,p0,pt,pc,pa,ph; .reg .b32 i<4>; .reg .b64 as1,bs1,bs2;",
        f".reg .b32 af1,bf1,bf2; .reg .b64 xa<{steps}>,xb<{steps}>;",
        "elect.sync _|e,0xffffffff; setp.eq.b32 pt,0,0; setp.ne.b32 pc,$7,0;",
        "and.pred pc,pc,e; setp.ne.b32 pa,$10,0; and.pred pa,pa,e;",
    ]
    if kind == "fc1":
        code.append("setp.ne.b32 p0,$8,0;")
    elif half:
        code.append("setp.eq.b32 p0,0,0;")
    else:
        code.append("setp.eq.b32 p0,0,1; setp.ne.b32 ph,$8,0; and.pred ph,ph,e;")
        code.append(f"@ph {arrive} [$8];")
    code += [f"mov.b32 i{k},{142738048 | k << 29 | k << 4};" for k in range(4)]
    if not half:
        code.append("add.s64 as1,$4,32; add.s64 bs1,$6,8;")
        code.append("add.u32 af1,$3,4; add.u32 bf1,$5,4;")
        if kind == "fc2a":
            code.append("add.s64 bs2,$6,16; add.u32 bf2,$5,8;")
        code += [f"@pc {cp} [$5],$6;", f"@pc {cp} [bf1],bs1;"]
        if kind == "fc2a":
            code.append(f"@pc {cp} [bf2],bs2;")
    code.append(f"@e {cp} [$3],$4;")
    if not half:
        code.append(f"@e {cp} [af1],as1;")
    for k in range(steps):
        a = k // 4 * 1024 + k % 4 * 2
        b = k // 4 * 64 + k % 4 * 2
        code.append(f"add.s64 xa{k},$1,{a}; add.s64 xb{k},$2,{b};")
    for k in range(0 if nomma else steps):
        af, bf = ("$3", "$5") if k < 4 else ("af1", "bf1")
        code.append(
            "@e tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale "
            f"[$0],xa{k},xb{k},i{k % 4},[{af}],[{bf}],{'p0' if k == 0 else 'pt'};"
        )
    if kind != "fc2p":
        code.append(f"@e {arrive} [$9];")
    if kind != "fc2a":
        code.append(f"@pa {arrive} [$10];")
    code.append("}")
    kinds = "rllrlrlrrrr"
    operands = (*mma, *weight_scales, *activation_scales, *release, done)
    values = [(Uint64 if k == "l" else Uint32)(v) for k, v in zip(kinds, operands)]
    return "\n".join(code), ",".join(kinds) + ",~{memory}", values


class StreamExpertsKernel(PLN.PlannedExpertsKernel):
    """The planned engine (xplan, TP8, capacity <= 8) with the ticket producer above.

    SMEM (``stages`` 32 KB slots, ``hbufs`` hidden buffers): ring, weight scales [stages x 1 KB],
    x [28 KB], hidden images [hbufs x 3 KB], x scales, hidden scales [hbufs x 768 B], misc
    (5120 B). Everything a tcgen05 descriptor reads lies inside the descriptor's address range (asserted). TMEM as the
    planned engine; hidden scales at 128 + 8 x stages + 12 x buffer (< 256 for hbufs <= 4)."""

    def __init__(
        self, tp, capacity, experts, grid, publish=False, activation_floor=None, stages=6, lanes=1,
        order="planned", lag=20, grain=1, hbufs=2, prefetch=False, ablate=(), strict=False, producer="ticket",
        xone=False,
    ):
        super().__init__(tp, capacity, experts, grid, publish, activation_floor, stages=5, xplan=True)
        # ``strict``: the FC1 -> FC2 hidden hand-off with strong accesses (Q5 fix): relaxed.gpu data
        # stores, one 16-bit relaxed store of both block scales per (tile, token), 16-bit scale polls.
        self.strict = strict
        # ``producer``: "ticket" (this module's) or "planned" (``PlannedExpertsKernel.stream``, lane 0 only: the
        # control arm that isolates the producer code; requires lanes 1, planned order, grain 1)
        assert producer in ("ticket", "planned") and (producer == "ticket" or (lanes, order, grain) == (1, "planned", 1))
        self.producer = producer
        # ``xone``: the x plan arrives in two TMA ops (all 14 K stages of x and of its scales, onto xready[0])
        # instead of 28 ops issued by the producer lane before its first weight op (same SMEM image).
        self.xone = xone
        self.entry_prefetch = False  # W8 kwarg: see kernel()
        assert lanes in (1, 2, 4) and order in ("planned", "lag") and grain in (1, 2, 7), (lanes, order, grain)
        assert hbufs in (2, 4) and set(ablate) <= set(ABLATIONS), (hbufs, ablate)
        ablate = set(ablate)
        if "drain" in ablate:
            ablate |= {"hid", "epi"}
        assert "epi" not in ablate or "hid" in ablate, "epilogue off needs hidden polls off (images unwritten)"
        self.lanes, self.order, self.lag, self.grain = lanes, order, lag, grain
        self.hbufs, self.prefetch, self.ablate = hbufs, prefetch, frozenset(ablate)
        self.x_bytes = 28672
        self.layout(stages)

    def layout(self, stages):
        """``PlannedExpertsKernel.relayout`` with ``hbufs`` hidden buffers."""
        assert 4 <= stages <= 6, stages
        self.stages = stages
        self.off_sfa = stages * 32768
        self.off_x = self.off_sfa + stages * 1024
        self.off_h = self.off_x + 28672
        self.off_xsf = self.off_h + self.hbufs * self.h_image_bytes
        self.off_hsf = self.off_xsf + 28 * 128 + 384
        self.off_misc = self.off_hsf + self.hbufs * self.h_scale_bytes
        self.smem_bytes = self.off_misc + 5120
        self.bar_full, self.bar_empty, self.bar_accfull = (0, 8 * stages, 16 * stages)
        acc = self.bar_accfull
        self.bar_accempty, self.bar_hfull = (acc + 128, acc + 256)
        self.bar_hempty, self.bar_xready = (acc + 272, acc + 288)
        self.off_meta = self.off_misc + 4224
        self.off_queue = self.off_misc + 4608
        assert self.off_misc <= 1 << 18, self.off_misc  # tcgen05 descriptor window
        assert acc + 288 + 14 * 8 <= 600 and 4224 + 16 * stages <= 4400
        assert HBARS + 16 * self.hbufs <= 5120 and 128 + 8 * stages + 12 * self.hbufs <= 256
        assert self.smem_bytes <= PLN.large_smem_alloc(), self.smem_bytes

    @cute.jit
    def fetch_x(self, sm, maps):
        if cutlass.const_expr(self.xone):
            bar = view(sm.iterator, self.off_misc + self.bar_xready, Uint64, 14).iterator
            cute.arch.mbarrier_arrive_and_expect_tx(bar, self.x_bytes + 28 * 128)
            PLN.load_tma3_keep(sm.iterator + self.off_x, maps + 768, bar, 0, 0)
            PLN.load_tma3_keep(sm.iterator + self.off_xsf, maps + 896, bar, 0, 0)
        else:
            PLN.PlannedExpertsKernel.fetch_x(self, sm, maps)

    # ------------------------------------------------------------------------------ tickets
    def fc2_groups(self):
        return 14 // self.grain

    @cute.jit
    def ticket_count(self, distinct):
        return distinct * (self.tiles + self.fc2_groups())

    @cute.jit
    def decode(self, t, distinct):
        """Ticket ``t`` -> (kind 0 FC1 / 1 FC2, distinct-expert index, tile or first pair)."""
        g2: cutlass.Constexpr = self.fc2_groups()
        kind, unit, sub = (Int32(0), Int32(0), Int32(0))
        if cutlass.const_expr(self.order == "planned"):
            if t < self.tiles * distinct:
                unit, sub = (t // self.tiles, t % self.tiles)
            else:
                q = t - self.tiles * distinct
                kind, unit, sub = (Int32(1), q // g2, q % g2 * self.grain)
        else:
            lag = Int32(self.lag)
            head, mid = (distinct, Int32(0))  # experts with FC1 only; with FC1 of e + FC2 of e - lag
            if distinct > lag:
                head, mid = (lag, distinct - lag)
            width: cutlass.Constexpr = self.tiles + g2
            if t < self.tiles * head:
                unit, sub = (t // self.tiles, t % self.tiles)
            else:
                q = t - self.tiles * head
                if q < width * mid:
                    block, r = (q // width, q % width)
                    if r < self.tiles:
                        unit, sub = (lag + block, r)
                    else:
                        kind, unit, sub = (Int32(1), block, (r - self.tiles) * self.grain)
                else:
                    q2 = q - width * mid
                    kind, unit, sub = (Int32(1), mid + q2 // g2, q2 % g2 * self.grain)
        return kind, unit, sub

    @cute.jit
    def prefetch_ticket(self, w13_base, w2_base, w13_scales, w2_scales, work):
        """L2 bulk prefetch of a decoded ticket's weight bytes and scales (one op each)."""
        kind, unit, sub, expert32, count = work
        expert = Int64(expert32)
        if kind == 0:
            row = expert * self.w13_rows + sub * 128
            prefetch(w13_base + row * 1792, 128 * 1792)
            prefetch(w13_scales.iterator.toint() + expert * self.w13_scale_bytes + sub * 14336, 14336)
        else:
            row = expert * 3584 + 2 * sub * 128
            prefetch(w2_base + row * 192, 256 * 192 * self.grain)
            step: cutlass.Constexpr = self.w2_tile_scale_bytes
            at = w2_scales.iterator.toint() + expert * self.w2_scale_bytes + 2 * sub * step
            prefetch(at, 2 * step * self.grain)

    # ------------------------------------------------------------------------------ producer
    @cute.jit
    def ticket(self, t, distinct, experts):
        """(kind, unit, sub, expert, stage count) of ticket ``t``."""
        kind, unit, sub = self.decode(t, distinct)
        expert = experts[unit]
        count = Int32(14)
        if kind == 1:
            count = Int32(3 * self.grain)
        return kind, unit, sub, expert, count

    @cute.jit
    def issue_range(self, ring, maps, w13_scales, w2_scales, policy, work, state, lo, hi):
        """Stages [lo, min(hi, count)) of a decoded ticket, ``lanes`` per pass (lane i: slot + i)."""
        sm, barriers, metadata = ring
        kind, unit, sub, expert, count = work
        slot, phase, issued = state
        lane = thread()[2]
        S: cutlass.Constexpr = self.stages
        P: cutlass.Constexpr = self.lanes
        end = count
        if end > hi:
            end = Int32(hi)
        base = Int32(lo)
        while base < end:
            active = end - base
            if active > P:
                active = Int32(P)
            if lane < active:
                my, ph = (slot + lane, phase)
                if my >= S:
                    my, ph = (my - S, phase ^ 1)
                if issued + lane >= S:
                    self.sync(barriers.iterator + S + my, ph ^ 1)
                self.issue_stage(sm, maps, w13_scales, w2_scales, barriers, metadata, my, kind, unit, expert,
                                 sub, base + lane, policy, base + lane)
            cute.arch.sync_warp()
            slot, issued = (slot + active, issued + active)
            if slot >= S:
                slot, phase = (slot - S, phase ^ 1)
            base += P
        return slot, phase, issued

    @cute.jit
    def stream_warp(self, sm, maps, w13_scales, w2_scales, distinct, experts, workspace):
        """Warp 0: the x plan TMAs (lane 0), then tickets in claim order, ``lanes`` stages per pass. As the
        planned producer, the first ticket (by block index) issues its first ring's worth of stages before
        the call counter and the first claim are read, and each claim stays in flight while the current
        ticket issues (its result is read only after)."""
        _, _, lane, cta = thread()
        S: cutlass.Constexpr = self.stages
        barriers = view(sm.iterator, self.off_misc, Uint64, 2 * S)
        metadata = view(sm.iterator, self.off_meta, Int32, 4 * S)
        ring = (sm, barriers, metadata)
        policy = evict_first_policy()
        w13_base, w2_base = (Int64(0), Int64(0))
        if cutlass.const_expr(self.prefetch):
            bases = view(maps.iterator, 1024, Int64, 2)
            w13_base, w2_base = (bases[0], bases[1])
        if lane == 0:
            self.fetch_x(sm, maps.iterator)
        total = self.ticket_count(distinct)
        state = (Int32(0), Int32(0), Int32(0))
        pushed, current = (Int32(0), Int32(-1))
        t = Int32(cta)
        first = t
        if first >= total:
            first = Int32(0)
        work = self.ticket(first, distinct, experts)
        if t < total:
            if work[0] == 1:
                if lane == 0:
                    self.push_unit(sm, pushed, work[1])
                pushed, current = (pushed + 1, work[1])
            state = self.issue_range(ring, maps, w13_scales, w2_scales, policy, work, state, 0, S)
        calls = self.calls(workspace)
        call = calls[cta]
        parity = call & 1
        if (cta == 0) & (lane == 0):
            store_words(self.claim_counter(workspace, parity ^ 1), (Uint32(4294967295),))
        counter = self.claim_counter(workspace, parity)
        got = Uint32(0)
        if lane == 0:
            got = claim(counter, 1)
        if t < total:
            state = self.issue_range(ring, maps, w13_scales, w2_scales, policy, work, state, S, 14)
        while t < total:
            pending = shuffle(got, 0)
            t = Int32(self.grid) + Int32(pending + Uint32(1))
            if t < total:
                got = Uint32(0)
                if lane == 0:
                    got = claim(counter, 1)
                work = self.ticket(t, distinct, experts)
                if work[0] == 1:
                    if work[1] != current:
                        if lane == 0:
                            self.push_unit(sm, pushed, work[1])
                        pushed, current = (pushed + 1, work[1])
                if cutlass.const_expr(self.prefetch):
                    if lane == 0:
                        self.prefetch_ticket(w13_base, w2_base, w13_scales, w2_scales, work)
                state = self.issue_range(ring, maps, w13_scales, w2_scales, policy, work, state, 0, 14)
        slot, phase, issued = state
        if lane == 0:
            self.push_unit(sm, pushed, Int32(-1))
            if issued >= S:
                self.sync(barriers.iterator + S + slot, phase ^ 1)
            for i in cutlass.range_constexpr(4):
                metadata[4 * slot + i] = Int32(2 if i == 0 else 0)
            cute.arch.mbarrier_arrive(barriers.iterator + slot, arrive_count=1)
            self.advance_calls(calls, cta, call)

    @cute.jit
    def issue_stage(self, sm, maps, w13_scales, w2_scales, barriers, metadata, my, kind, unit, expert, sub, s,
                    policy, src):
        """Stage ``s`` of ticket (kind, unit, sub) into slot ``my``: metadata, expect_tx, weight
        TMA and scale copies (FC1: K step s of tile ``sub``; FC2: stage s % 3 of pair sub + s // 3).
        ``src`` (= ``s``; a mutation test changes it) selects the bytes fetched."""
        full = barriers.iterator + my
        dst = sm.iterator + my * 32768
        sdst = sm.iterator + self.off_sfa + my * 1024
        noscale: cutlass.Constexpr = "scale" in self.ablate
        if kind == 0:
            c = Int32(s == 0) | Int32(s == 13) << 1
            for i, v in enumerate((Int32(0), unit * self.tiles + sub, s, c)):
                metadata[4 * my + i] = v
            ks = src
            row = expert * self.w13_rows + sub * 128
            off = Int64(expert) * self.w13_scale_bytes + sub * 14336 + ks * 1024
            cute.arch.mbarrier_arrive_and_expect_tx(full, 16384 if noscale else 16384 + 1024)
            tma3(dst, maps.iterator, full, row, 2 * ks, policy)
            if cutlass.const_expr(not noscale):
                bulk(sdst, w13_scales.iterator + off, full, 1024, policy)
        else:
            step: cutlass.Constexpr = self.w2_tile_scale_bytes
            j = s % 3
            tile = 2 * (sub + s // 3)
            second = Int32(j == 1)
            for i, v in enumerate((Int32(1) + 2 * Int32(j == 2), unit, tile + second, Int32(0))):
                metadata[4 * my + i] = v
            j = src % 3
            tile = 2 * (sub + src // 3)
            second = Int32(j == 1)
            row = expert * 3584 + (tile + second) * 128
            off = Int64(expert) * self.w2_scale_bytes + (tile + second) * step
            cute.arch.mbarrier_arrive_and_expect_tx(full, 16384 if noscale else 16384 + 1024)
            if j < 2:
                tma3(dst, maps.iterator + 128, full, row, 0, policy)
                if cutlass.const_expr(not noscale):
                    bulk(sdst, w2_scales.iterator + off, full, 1024, policy)
            else:
                tma3(dst, maps.iterator + 640, full, row, 2, policy)
                if cutlass.const_expr(not noscale):
                    bulk(sdst, w2_scales.iterator + off + 1024, full, 512, policy)
                    bulk(sdst + 512, w2_scales.iterator + off + step + 1024, full, 512, policy)

    # ------------------------------------------------------------------------------ hidden loader
    def hidden_barriers(self, sm):
        bars = self.off_misc + HBARS
        return view(sm.iterator, bars, Uint64, self.hbufs), view(sm.iterator, bars + 8 * self.hbufs, Uint64, self.hbufs)

    @cute.jit
    def load_hidden(self, sm, workspace, ranges, parity):
        """``ExpertsKernel.load_hidden`` (stealing path: units from the producer's queue) over
        ``hbufs`` buffers; ablation "hid": no poll (the barrier protocol is kept), "drain": pop only."""
        lane = thread()[2]
        full, empty = self.hidden_barriers(sm)
        H: cutlass.Constexpr = self.hbufs
        ordinal = Int32(-1)
        unit = self.pop_unit(sm, Int32(0))
        while unit >= 0:
            ordinal += 1
            if cutlass.const_expr("drain" not in self.ablate):
                buffer = ordinal % H
                if ordinal >= H:
                    self.sync(empty.iterator + buffer, (ordinal // H - 1) & 1)
                    cute.arch.fence_proxy("async.shared", space="cta")  # MMA read it
                if cutlass.const_expr("hid" not in self.ablate):
                    self.poll_hidden(sm, self.image(workspace, parity, unit), buffer)
                cute.arch.sync_warp()
                if lane == 0:
                    cute.arch.mbarrier_arrive(full.iterator + buffer)
            unit = self.pop_unit(sm, ordinal + 1)

    # ------------------------------------------------------------------------------ MMA warp
    @cute.jit
    def consume(self, sm, tmem, workspace=None):
        """``ExpertsKernel.consume`` (streamed TP8 paired path) over ``hbufs`` hidden buffers."""
        lane = thread()[2]
        stages, H = (self.stages, self.hbufs)
        nomma: cutlass.Constexpr = "mma" in self.ablate
        bars: tuple = (self.bar_full, self.bar_empty, self.bar_accfull, self.bar_accempty, self.bar_xready)
        full, empty, accfull, accempty, xready = (
            view(sm.iterator, self.off_misc + o, Uint64, n) for o, n in zip(bars, (stages, stages, 16, 16, 14))
        )
        hfull, hempty = self.hidden_barriers(sm)
        meta = view(sm.iterator, self.off_meta, Int32, 4 * stages)
        accmeta = view(sm.iterator, self.off_misc + 600, Int32, 64)
        offsets = (0, self.off_sfa, self.off_x, self.off_xsf, self.off_h, self.off_hsf)
        a0, asf0, x0, xsf0, h0, hsf0 = (smem_desc(sm.iterator + o, i % 2 == 0) for i, o in enumerate(offsets))
        xmask, ended = (Uint32(0), False)
        slot, uses, acc, count = (Int32(0), Int32(0), Int32(0), Int32(0))
        current, ordinal, hbuffer = (Int32(-1), Int32(-1), Int32(0))
        while not ended:
            slot = count % stages
            self.sync(full.iterator + slot, count // stages & 1, True)
            kind, a, b, c = (meta[4 * slot + i] for i in range(4))
            if kind == 2:
                if (lane == 0) & (current >= 0):
                    commit(hempty.iterator + hbuffer)
                for _ in cutlass.range(2, unroll=1):
                    self.acquire(accempty, acc, uses)
                    self.signal(accmeta, accfull, acc, (kind, a, b, c), 2)
                    uses += 1
                    acc = uses % 16
                ended = True
            else:
                publish = False
                released = shared_address(empty.iterator + slot)
                weights = (tmem + 128 + slot * 8, asf0 + slot * 64)
                if kind == 0:
                    first, last = (c & 1 != 0, c & 2 != 0)
                    if first:
                        self.acquire(accempty, acc, uses)
                    copy_scales = Uint32(0)
                    if xmask >> b & 1 == 0:
                        self.sync(xready.iterator + (0 if self.xone else b), 0, True)
                        xmask |= Uint32(1) << b
                        copy_scales = Uint32(1)
                    completion = Uint32(0)
                    if last:
                        completion = shared_address(accfull.iterator + acc)
                    stage(
                        (tmem + acc * 8, a0 + slot * 2048, x0 + b * 128),
                        weights,
                        (tmem + 256 + b * 8, xsf0 + b * 16, copy_scales),
                        (Uint32(not first), released),
                        completion,
                        kind="fc1",
                        nomma=nomma,
                    )
                    cute.arch.sync_warp()
                    publish = last
                else:
                    copy_scales, release = (Uint32(0), Uint32(0))
                    if a != current:
                        if current >= 0:
                            release = shared_address(hempty.iterator + hbuffer)
                        current = a
                        ordinal += 1
                        hbuffer = ordinal % H
                        self.sync(hfull.iterator + hbuffer, ordinal // H & 1, True)
                        copy_scales = Uint32(1)
                    hd = h0 + hbuffer * (self.h_image_bytes // 16)
                    hs = hsf0 + hbuffer * (self.h_scale_bytes // 16)
                    sfbc = tmem + 128 + 8 * stages + hbuffer * self.w2_scale_columns
                    if kind == 1:  # chunks 0-1 of the pair's tile b
                        target, owner = (acc, uses)
                        if b % 2 == 1:
                            target, owner = ((uses + 1) % 16, uses + 1)
                        self.acquire(accempty, target, owner)
                        stage(
                            (tmem + target * 8, a0 + slot * 2048, hd),
                            weights,
                            (sfbc, hs, copy_scales),
                            (release, released),
                            0,
                            kind="fc2a",
                            nomma=nomma,
                        )
                    else:  # chunk 2 of tiles b (rows 0-127) and b + 1 (128-255)
                        for t in cutlass.range_constexpr(2):
                            rows = a0 + slot * 2048 + 1024 * t
                            stage(
                                (tmem + acc * 8, rows, hd + 128),
                                (weights[0] + 4 * t, weights[1] + 32 * t),
                                (sfbc + 8, 0, 0),
                                (0, released),
                                shared_address(accfull.iterator + acc),
                                kind="fc2b" if t else "fc2p",
                                nomma=nomma,
                            )
                            if cutlass.const_expr(t == 0):
                                cute.arch.sync_warp()
                                self.signal(accmeta, accfull, acc, (Int32(1), a, b, c))
                                uses += 1
                                acc = uses % 16
                                cute.arch.sync_warp()
                        publish, kind, b = (True, Int32(1), b + 1)
                    cute.arch.sync_warp()
                if publish:
                    self.signal(accmeta, accfull, acc, (kind, a, b, c))
                    uses += 1
                    acc = uses % 16
                count += 1

    @cute.jit
    def drain(self, sm, workspace=None):
        """Ablation "drain": free each slot once its TMA lands; at the end, release the epilogue."""
        lane = thread()[2]
        stages = self.stages
        full = view(sm.iterator, self.off_misc + self.bar_full, Uint64, stages)
        empty = view(sm.iterator, self.off_misc + self.bar_empty, Uint64, stages)
        accfull = view(sm.iterator, self.off_misc + self.bar_accfull, Uint64, 16)
        meta = view(sm.iterator, self.off_meta, Int32, 4 * stages)
        accmeta = view(sm.iterator, self.off_misc + 600, Int32, 64)
        count, ended = (Int32(0), False)
        while not ended:
            slot = count % stages
            self.sync(full.iterator + slot, count // stages & 1)
            if meta[4 * slot] == 2:
                for acc in cutlass.range_constexpr(2):
                    self.signal(accmeta, accfull, Int32(acc), (Int32(2), Int32(0), Int32(0), Int32(0)), 2)
                ended = True
            else:
                cute.arch.sync_warp()
                if lane == 0:
                    cute.arch.mbarrier_arrive(empty.iterator + slot)
                count += 1

    # ------------------------------------------------------------------------------ strict hidden hand-off
    @cute.jit
    def store_hidden(self, sm, workspace, values, tile, parity, scalars):
        """``ExpertsKernel.store_hidden``; ``strict``: the image words by ``st.relaxed.gpu`` and the
        tile's two block scales of each token by one 16-bit ``st.relaxed.gpu`` (warp ew 0 also
        computes block 1's scale from ``shared_max``: the same fmax of the same two warp maxima)."""
        if cutlass.const_expr(not self.strict):
            PLN.PlannedExpertsKernel.store_hidden(self, sm, workspace, values, tile, parity, scalars)
        else:
            _, warp, lane, _ = thread()
            ew, eg = (warp & 3, (warp - 4) // 4)
            expert, chunk = (tile // self.tiles, tile % self.tiles)
            up = lane & 8 == 0
            h, maxima = (cute.make_rmem_tensor(self.width, Float32) for _ in range(2))
            shared_max = view(sm.iterator, self.off_misc + 884, Float32, 64)
            for t in cutlass.range_constexpr(self.width):
                gate = cute.arch.shuffle_sync_bfly(values[t], 8)
                h[t] = situ(gate, values[t], *scalars) if up else Float32(0)
                maxima[t] = abs(h[t])
            for t in cutlass.range_constexpr(self.width):
                for off in cutlass.range_constexpr(5):
                    other = cute.arch.shuffle_sync_bfly(maxima[t], 1 << off)
                    maxima[t] = cute.arch.fmax(maxima[t], other)
                if lane == 0:
                    shared_max[(eg * 4 + ew) * 8 + t] = maxima[t]
            group_barrier(eg)
            stage_base, scale_token, scale_high = (self.off_misc + 3200 + (eg * 4 + ew) * 128, Int32(0), Int32(0))
            for t in cutlass.range_constexpr(self.width):
                peer = shared_max[(eg * 4 + (ew ^ 1)) * 8 + t]
                scale = hidden_exponent(cute.arch.fmax(maxima[t], peer), floor=self.activation_floor)
                if lane == t:
                    scale_token = scale
                    if ew == 0:
                        high = cute.arch.fmax(shared_max[(eg * 4 + 2) * 8 + t], shared_max[(eg * 4 + 3) * 8 + t])
                        scale_high = hidden_exponent(high, floor=self.activation_floor)
                if up:
                    bits = fp8_bits(h[t] * mx_rescale(scale))
                    sm[stage_base + t * 16 + 2 * (lane % 8) + lane // 16] = bits
            cute.arch.sync_warp()
            if lane < 8:
                kc, data = (4 * chunk + ew, (Uint32(0), Uint32(0), Uint32(0), Uint32(0)))
                if lane < self.width:
                    data = smem_words(sm.iterator + stage_base + lane * 16)
                image = self.image(workspace, parity, expert)
                store_words(image + swizzled(kc, lane), data, scope="gpu")
                if ew == 0:
                    pair = (Uint32(scale_token) & 255) | (Uint32(scale_high) & 255) << 8
                    store_u16_relaxed(image + self.h_image_bytes + lane * 16 + 2 * chunk, Uint16(pair))
            group_barrier(eg)

    @cute.jit
    def poll_hidden(self, sm, source, buffer: Int32):
        """``ExpertsKernel.poll_hidden``; ``strict``: the scale bytes are polled as two 16-bit volatile loads
        (the width they are stored at) and combined into the same 32-bit word."""
        if cutlass.const_expr(not self.strict):
            PLN.PlannedExpertsKernel.poll_hidden(self, sm, source, buffer)
        else:
            lane = thread()[2]
            chunks: cutlass.Constexpr = self.h_image_bytes // 512
            values, scale = (cute.make_rmem_tensor((chunks, 4), Uint32), Uint32(0))
            valid_bytes = min(4, max(0, self.tiles * 2 - 4 * (lane & 3)))
            mask = Uint32(4294967295)
            if valid_bytes < 4:
                mask = (Uint32(1) << 8 * valid_bytes) - 1
            ready = False
            while not ready:
                bad = False
                for v in cutlass.range_constexpr(chunks):
                    index = lane + 32 * v
                    words = load_volatile(source + index * 16, count=4)
                    for w in cutlass.range_constexpr(4):
                        values[v, w] = words[w]
                    atom, token, position = (index >> 6, index >> 3 & 7, index & 7)
                    if 8 * atom + (position ^ token) < self.tiles * 4:
                        for w in cutlass.range_constexpr(4):
                            bad |= has_ff_byte(words[w])
                at = source + self.h_image_bytes + lane * 4
                scale = Uint32(load_u16_volatile(at)) | Uint32(load_u16_volatile(at + 2)) << 16
                if valid_bytes > 0:
                    bad |= has_ff_byte(scale & mask)
                ready = cute.arch.vote_all_sync(not bad)
            image = sm.iterator + self.off_h + buffer * self.h_image_bytes
            for v in cutlass.range_constexpr(chunks):
                words = tuple(values[v, w] for w in range(4))
                store_shared_words(image + (lane + 32 * v) * 16, words)
            word = cute.arch.shuffle_sync(scale, 4 * (lane & 7) + (lane >> 3))
            if lane < 8 * self.h_chunks:
                sf = sm.iterator + self.off_hsf + buffer * self.h_scale_bytes
                store_shared_words(sf + (lane >> 3) * 128 + (lane & 7) * 16, (word,) * 4)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_warp()

    # ------------------------------------------------------------------------------ epilogue
    @cute.jit
    def epilogue(self, sm, workspace, output, tmem, ranges, batch, parity, scalars):
        if cutlass.const_expr("epi" in self.ablate):
            _, warp, lane, _ = thread()
            full = view(sm.iterator, self.off_misc + self.bar_accfull, Uint64, 16)
            empty = view(sm.iterator, self.off_misc + self.bar_accempty, Uint64, 16)
            meta = view(sm.iterator, self.off_misc + 600, Int32, 64)
            acc, phase, ended = ((warp - 4) // 4, Int32(0), False)
            while not ended:
                self.sync(full.iterator + acc, phase, True)
                if meta[4 * acc] == 2:
                    ended = True
                else:
                    fence_tmem(before=True)
                    cute.arch.sync_warp()
                    if lane == 0:
                        cute.arch.mbarrier_arrive(empty.iterator + acc)
                    acc += 2
                    if acc >= 16:
                        acc -= 16
                        phase ^= 1
        else:
            PLN.PlannedExpertsKernel.epilogue(self, sm, workspace, output, tmem, ranges, batch, parity, scalars)

    # ------------------------------------------------------------------------------ kernel
    @cute.jit
    def initialize(self, sm, maps, batch):
        """``PlannedExpertsKernel.initialize`` (x plan) plus the ``hbufs`` hidden barriers."""
        tid = thread()[0]
        count_all: cutlass.Constexpr = 2 * self.stages + 50
        barriers = view(sm.iterator, self.off_misc, Uint64, count_all)
        tmem_base = view(sm.iterator, self.off_misc + 856, Uint32, 1)
        if tid == 0:
            for i in cutlass.range_constexpr(count_all):
                acc = 2 * self.stages
                count = 1 if not acc <= i < acc + 32 else 2 if i < acc + 16 else 4
                cute.arch.mbarrier_init(barriers.iterator + i, count)
            queue = view(sm.iterator, self.off_misc + self.bar_queue, Uint64, 16)
            for i in cutlass.range_constexpr(16):
                cute.arch.mbarrier_init(queue.iterator + i, 1)
            hbars = view(sm.iterator, self.off_misc + HBARS, Uint64, 2 * self.hbufs)
            for i in cutlass.range_constexpr(2 * self.hbufs):
                cute.arch.mbarrier_init(hbars.iterator + i, 1)
            cute.arch.mbarrier_init_fence()
            for offset in (0, 128, 640, 768, 896):
                acquire_tensormap(maps.iterator + offset)
        if tid // 32 == 3:
            cute.arch.alloc_tmem(512, tmem_base.iterator)
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.fence_proxy("async.shared", space="cta")
        return tmem_base

    @cute.kernel
    def kernel(self, args):
        """``ExpertsKernel.kernel`` (non-grouped, x plan): warp 0 produces, 1 MMA (or drain),
        2 hidden loader, 3 re-arm, 4-11 epilogue."""
        self.kernel_body(args)

    @cute.jit
    def kernel_body(self, args):
        """The kernel's body (a separate ``cute.jit`` so a subclass kernel with another signature, the
        W8 ``__grid_constant__`` tensormap form, can run it; ``maps`` is anything with ``.iterator + offset``)."""
        x, ids, maps, w13_scales, w2_scales, workspace, output, batch, *scalars = args
        tid, warp, lane, cta = thread()
        sm = alloc(cutlass.utils.SmemAllocator(), Uint8, self.smem_bytes, 1024)
        if cutlass.const_expr(self.entry_prefetch):  # E2: overlap the prologue's first-touch latencies with init + the wait
            if warp == 1:
                if lane < 4:
                    prefetch_l2(ids.iterator + 32 * lane)
                elif lane == 4:
                    prefetch_l2(workspace.iterator + self.ws_counter + 4 * cta)
                elif lane == 5:
                    prefetch_l2(workspace.iterator + self.ws_claim + 64 + 4 * cta)
                elif lane == 6:
                    prefetch_l2(workspace.iterator + self.ws_claim)
                elif lane == 7:
                    prefetch_l2(self.image(workspace, 0, 0))
                elif lane == 8:
                    prefetch_l2(self.image(workspace, 1, 0))
                elif lane == 9:
                    prefetch_l2(x.iterator)
                elif lane == 10:
                    prefetch_l2(x.iterator + PLN.PLAN_XSF)
                elif lane == 11:
                    prefetch_l2(output.iterator)
                elif lane < 20:
                    prefetch_l2(maps.iterator + 128 * (lane - 12))
        tmem_base = self.initialize(sm, maps, batch)
        cute.arch.griddepcontrol_wait()
        counters = view(workspace.iterator, self.ws_counter, Int32, MAX_CTAS)
        distinct, call_count, experts = self.route(sm, ids, counters, batch)
        generation = call_count
        fence_tmem()
        tmem = tmem_base[0]
        if cutlass.const_expr(getattr(self, "pub_fence", False)):  # memory-model audit fix (W8 publish forms finf / f2spf only):
            if tid == 128:  # this thread's griddepcontrol.wait (above) -> fence.acq_rel.gpu -> the CTA's one trigger (PTX 9.7.15.14)
                fence_acq_rel_gpu()
                cute.arch.griddepcontrol_launch_dependents()
        else:
            cute.arch.griddepcontrol_launch_dependents()
        ranges = self.partition(distinct, cta)
        if warp == 0:
            if cutlass.const_expr(self.producer == "planned"):
                if lane == 0:
                    PLN.PlannedExpertsKernel.stream(self, sm, maps, w13_scales, w2_scales, ranges, experts, (workspace,))
            else:
                self.stream_warp(sm, maps, w13_scales, w2_scales, distinct, experts, workspace)
        elif warp == 1:
            if cutlass.const_expr("drain" in self.ablate):
                self.drain(sm, workspace)
            else:
                self.consume(sm, tmem, workspace)
        elif warp == 2:
            self.load_hidden(sm, workspace, ranges, generation & 1)
        elif warp == 3:
            self.rearm(workspace, generation)
        else:
            self.epilogue(sm, workspace, output, tmem, ranges, batch, generation & 1, scalars)
        release_tmem(tmem, 512, warp, 3)
        if tid == 0:
            counters[cta] = call_count + 1
            if cta < MAX_CTAS - self.grid:
                counters[self.grid + cta] = call_count + 1


# --------------------------------------------------------------------------------- runtime
def plan_maps(plan, batch, device, atoms=2):
    """``planned.plan_maps`` with ``atoms`` 128-K atoms per box (2: one K stage; 28: all of x in one op)."""
    import cuda.bindings.driver as cuda

    from ..common.host import tensor_maps

    u8 = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8
    sw = cuda.CUtensorMapSwizzle
    none = cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE
    base = plan.data_ptr()
    specs = [
        (base, u8, (128, batch, 28), (3584, 128), (128, 8, atoms), sw.CU_TENSOR_MAP_SWIZZLE_128B, none),
        (base + PLN.PLAN_XSF, u8, (16, batch, 28), (448, 16), (16, 8, atoms), sw.CU_TENSOR_MAP_SWIZZLE_NONE, none),
    ]
    return tensor_maps(specs, device)


@functools.cache
def compile_stream(tp, experts, batch, grid, device, publish=False, activation_floor=False, knobs=(), box_rows=64):
    """``knobs``: sorted (name, value) pairs of ``StreamExpertsKernel`` options."""
    capacity = 1 << (batch - 1).bit_length()
    moe = StreamExpertsKernel(tp, capacity, experts, grid, publish, activation_floor, **dict(knobs))
    moe.box_rows = box_rows  # the tail box's rows (``ExpertsKernel.box_rows``)
    flat = lambda n, dtype=cutlass.Uint8, align=16: fake((n,), (1,), dtype, align)  # noqa: E731
    box = (cutlass.Int64(0), Int32(0), Int32(0)) if publish else ()
    if publish == "twoshot":
        box = (fake((3 * tp, box_rows, 3584 // tp)), fake((tp,), None, cutlass.Int64), Int32(0))
    ids = flat(batch * 16, Int32)
    scales = (flat(experts * n) for n in (moe.w13_scale_bytes, moe.w2_scale_bytes))
    out = flat(batch * 16 * (1 if publish else 3584), cutlass.BFloat16)
    tensors = (flat(PLN.PLAN_BYTES), ids, flat(MAP_BYTES, align=64), *scales, flat(moe.workspace_bytes), out)
    scalars = (Int32(batch), *map(cutlass.Float32, (4, 25, 0)), *box)
    with torch.accelerator.device_index(device):
        code = cute.compile(moe, *tensors, *scalars, make_fake_stream())
        if moe.smem_bytes > 227 * 1024:
            PLN.set_oversized(code, device)
        return code, moe.smem_bytes


class StreamExperts(PLN.PlannedExperts):
    """``PlannedExperts`` (x plan) with ``StreamExpertsKernel`` for TP8 B <= 8 in ``batches``.
    ``knobs``: ``StreamExpertsKernel`` options (stages, lanes, order, lag, grain, hbufs,
    prefetch, ablate)."""

    def __init__(self, *args, batches=None, **knobs):
        engine = {k: knobs.pop(k) for k in list(knobs) if k in (
            "stages", "lanes", "order", "lag", "grain", "hbufs", "prefetch", "ablate", "strict", "producer",
            "xone")}
        engine["ablate"] = tuple(sorted(engine.get("ablate", ())))
        super().__init__(*args, stages=engine.get("stages", 6), xplan=True, batches=batches, **knobs)
        self.knobs = tuple(sorted(engine.items()))

    def prepare(self, batch, buffers, out, publish=(), planned=False, twoshot=False, early=False):
        if self.tp != 8 or batch > 8 or batch not in self.batches:
            return ER.PersistentExperts.prepare(self, batch, buffers, out, publish, planned, twoshot, early)
        need = PLN.planned_workspace_bytes(ExpertsGeometry(self.tp).workspace_bytes)
        if buffers["workspace"].numel() < need:
            raise ValueError("stream experts need planned.scratch() workspace")
        mode = "twoshot" if twoshot else bool(publish)
        grid = self.grid_for(batch)
        code, smem = compile_stream(
            self.tp, self.experts, batch, grid, self.device.index, mode,
            activation_floor=self.activation_floor, knobs=self.knobs, **self.box_kw,
        )
        if smem <= 227 * 1024:
            check_residency(code, self.device.index, grid, 384, "Kimi K3 MKv2 stream experts")
        elif grid > torch.cuda.get_device_properties(self.device).multi_processor_count:
            raise RuntimeError(f"stream experts: grid {grid} > SMs")
        w13, _, w2, _ = self.weights
        maps = torch.zeros(MAP_BYTES, dtype=torch.uint8, device=self.device)
        maps[:768] = self.maps[:768]
        maps[768:1024] = plan_maps(buffers["plan"], batch, self.device, 28 if dict(self.knobs).get("xone") else 2)
        bases = torch.tensor([w13.data_ptr(), w2.data_ptr()], dtype=torch.int64, device=self.device)
        maps[1024:1040] = bases.view(torch.uint8)
        self.batch_maps[batch] = maps
        plan = (ER.tensor(buffers["plan"]), PLN.compile_early_plan(batch, self.device.index))
        if planned:
            plan = plan[:1]
        ws = ER.tensor(buffers["workspace"])
        self.prepared[batch] = (out, ER.tensor(out), ws, code, publish, plan, ER.to_cute(maps, 64), None)
