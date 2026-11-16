# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MKv2 W4A4 routed experts, TP8 and TP16, B <= 8 (opt-in, default off;
TP16 needs the MK flag ``K31MK_EXPERTS_W4A4``). A PRECISION CHANGE: activations (FC1 input x
and the FC1 -> FC2 hidden) are MXFP4, not MXFP8. A lossy option: default off; the
W4A8 engines stay the normal path, and it ships only after the real-kernel accuracy evals pass.

Method (the w4a4eval fake-quant recipe = DeepGEMM MegaMoE's MXFP4 rule = the MK's MXFP8 rule applied to E2M1):
per 32 elements, scale = 2^ceil(log2(amax / 6)) (x: amax clamped to 1e-10, as the MK's MXFP8 input path; hidden:
amax 0 -> 2^0, as stock FC1), elements rounded to nearest-even E2M1 with saturation (``cvt.rn.satfinite.e2m1x2``).
FC1 / FC2 are ``tcgen05.mma.kind::mxf4.block_scale.scale_vec::2X`` (E2M1 x E2M1, UE8M0 per 32, K = 64, FP32
accumulation); SiTU, the BF16 output rows and the routing are the planned engine's.

Data path (packed FP4 everywhere, no 2x padding):
- Ring of ``slots`` 64 KB slots. One TMA op fills a slot: an FC1 op is 4 K stages of a 128-row tile (8 chunks of 128
  K, ``[chunk][128 rows][64 B]``, 64B swizzle; the tile's last op is 2 stages), an FC2 op is a tile pair
  (``[3 chunks][256 rows][64 B]``, 48 KB). One bulk copy per op brings the op's weight scales (4 KB / 2 KB / 3 KB).
- The producer is the planned engine's (one lane; FC1 tiles by block index then claims, FC2 pairs claimed in
  expert order; each claim in flight while the current unit issues), with 4x fewer, 4x larger ops.
- x: MXFP4 plan (``W4A4PlanKernel``: [64][1792 B] + the MXFP8 plan's scale layout) fetched per K stage (1 KB +
  256 B scales) by TMA; hidden image [3 chunks][8 tokens][64 B] (64B swizzle) + 128 B scales per expert.
- TP16 (I = 192 per rank, host-padded to 256): FC1 = the first 3 tiles of the 512-row W13 (the host's layout puts the
  384 real rows first); W2 rows are 128 B with real K bytes 0-95, so an FC2 op is a tile pair of 256 rows x two 64 B
  chunks at byte offsets 0 and 32 (32 KB; the shipped TP16 engine's overlap read), 3 K = 64 MMAs per tile (K 0-127 from
  chunk 0, K 128-191 from the upper half of chunk 1); hidden image [2 chunks][8 tokens][64 B] (chunk 1 half used) + 6
  block scales per token.
- FC1 -> FC2 hand-off: every epilogue warp stores its 8 bytes per token of the image, ``bar.sync`` (the 4 warps of
  the group), then warp 0 publishes both block scales of each token with one ``st.release.gpu.u16``; the FC2 loader
  polls the scale words with ``ld.acquire.gpu.u16`` (0xFF = re-armed; a UE8M0 scale is never 0xFF), ``bar.warp.sync``,
  then reads the image. (Packed E2M1 data cannot carry a sentinel: 0xFF is a valid pair, -6 / -6.)

Sync (memory-model audit; edges W1-W6): the ring / unit-queue / accumulator / x-ready
protocols are the planned engine's (one arrive.expect_tx per op), the hidden hand-off is release / acquire."""

import functools
import os

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint8, Uint16, Uint32, Uint64
from cutlass.cute.runtime import make_fake_stream

from ..common.host import fake
from ..common.primitives import (
    acquire_tensormap,
    asm,
    bits_float,
    commit,
    float_bits,
    group_barrier,
    load_words,
    mx_rescale,
    shared_address,
    sleep_ns,
    smem_desc,
    wait_deadline_check,
    wait_deadline_start,
    store_shared_words,
    store_words,
    thread,
    view,
)
from ..moe import experts_runtime as ER
from ..moe import planned as PLN
from ..moe.experts import ExpertsPlanKernel, claim, has_ff_byte
from ..moe.math import routed_situ as situ
from . import experts_stream as MV

SLOT = 65536  # one TMA op: 4 K stages of 16 KB packed
SLOT_SF = 4096  # weight-scale bytes per slot
PLAN4_XSF = 64 * 1792  # MXFP4 plan: data [64 tokens][1792 B], then scale words [64][28][16 B] (bytes 0-3 used)
H_IMAGE4 = 1536  # TP8 hidden image [3 chunks][8 tokens][64 B] (TP16: 2 chunks, ``W4A4ExpertsKernel.h4``)
MAPS4 = 1024  # map buffer: 0 W13 (8-chunk box), 128 W13 (4-chunk box), 256 W2 pairs, 768 x plan, 896 x scales


@cute.jit
def mx4_exponent(amax: Float32):
    """Biased UE8M0 exponent of 2^ceil(log2(amax / 6)), clamped to [0, 254]; exact (6 = 1.5 x 2^2)."""
    bits = Uint32(float_bits(amax))
    e = Int32(bits >> 23)
    result = Int32(0)
    if e != 0:
        result = min(max(e - 2 + Int32(bits & 8388607 > 4194304), 0), 254)
    return result


@cute.jit
def hidden4_exponent(amax: Float32):
    return Int32(127) if amax == 0 else mx4_exponent(amax)


@asm(Uint32, pure=True)
def e2m1x8(values):
    """Eight FP32 values (already scaled) -> one word of packed E2M1: byte i = (v[2i+1] << 4) | v[2i]."""
    code = "{ .reg .b8 q0,q1,q2,q3; "
    code += " ".join(f"cvt.rn.satfinite.e2m1x2.f32 q{i},${2 * i + 2},${2 * i + 1};" for i in range(4))
    code += " mov.b32 $0,{q0,q1,q2,q3}; }"
    return code, "=r," + ",".join(["f"] * 8), values


@asm(Uint32, pure=True)
def e2m1x2(odd, even):
    """(odd << 4) | even as the low byte of a word (upper bytes copies)."""
    code = "{ .reg .b8 q; cvt.rn.satfinite.e2m1x2.f32 q,$1,$2; mov.b32 $0,{q,q,q,q}; }"
    return code, "=r,f,f", (Float32(odd), Float32(even))


@asm()
def store_release_u16(pointer, value):
    return "st.release.gpu.global.u16 [$0], $1;", "l,h,~{memory}", (pointer, Uint16(value))


@asm(Uint16)
def load_acquire_u16(pointer):
    return "ld.acquire.gpu.global.u16 $0, [$1];", "=h,l,~{memory}", (pointer,)


@cute.jit
def desc64(pointer: cute.Pointer):
    """tcgen05 SMEM matrix descriptor, K-major, 64-byte swizzle (layout 4), SBO 512 B (8 rows x 64 B)."""
    address = shared_address(pointer)
    return Uint64((address & 262143) >> 4) | Uint64(1) << 16 | Uint64(32) << 32 | Uint64(1) << 46 | Uint64(4) << 61


def idesc4(a_id, b_id, sv):
    """kind::mxf4 instruction descriptor (PTX 9.4 Table 53): E2M1 x E2M1, UE8M0, M128, N8, K64, scale IDs."""
    return (b_id << 4) | (1 << 7) | (1 << 10) | (sv << 12) | (1 << 17) | (1 << 23) | (1 << 27) | (a_id << 29)


def _group(steps, a_cps, b_cps, sv, nomma, commit_done, commit_release, pre_release):
    """PTX of one MMA group. Operands: $0 accumulator, $1 A desc, $2 B desc, $3 A scale column, $4 A scale desc,
    $5 B scale column, $6 B scale desc, $7 B-copy flag, $8 accumulate flag (first MMA), $9 done barrier,
    $10 release barrier, $11 pre-release barrier. ``steps``: (A offset, B offset, A column offset, B column offset,
    scale ID) in 16-byte units / TMEM columns; ``a_cps`` / ``b_cps``: (column offset, desc offset) copies."""
    cp = "tcgen05.cp.cta_group::1.32x128b.warpx4"
    arrive = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
    code = ["{ .reg .pred e,p0,pt,pc,pr; .reg .b32 c<8>,sa,sb,id; .reg .b64 d<8>,xa,xb;",
            "elect.sync _|e,0xffffffff; setp.eq.b32 pt,0,0; setp.ne.b32 pc,$7,0; and.pred pc,pc,e;",
            "setp.ne.b32 p0,$8,0;"]
    if pre_release:
        code.append(f"setp.ne.b32 pr,$11,0; and.pred pr,pr,e; @pr {arrive} [$11];")
    for i, (col, off) in enumerate(b_cps):
        code.append(f"add.u32 c{i},$5,{col}; add.s64 d{i},$6,{off}; @pc {cp} [c{i}],d{i};")
    for i, (col, off) in enumerate(a_cps):
        code.append(f"add.u32 c{i + 4},$3,{col}; add.s64 d{i + 4},$4,{off}; @e {cp} [c{i + 4}],d{i + 4};")
    for k, (aoff, boff, acol, bcol, sid) in enumerate(steps):
        code.append(f"add.s64 xa,$1,{aoff}; add.s64 xb,$2,{boff}; add.u32 sa,$3,{acol}; add.u32 sb,$5,{bcol};")
        code.append(f"mov.b32 id,{idesc4(sid, sid, sv)};")
        if not nomma:
            code.append("@e tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X "
                        f"[$0],xa,xb,id,[sa],[sb],{'p0' if k == 0 else 'pt'};")
    if commit_done:
        code.append(f"setp.ne.b32 pr,$9,0; and.pred pr,pr,e; @pr {arrive} [$9];")
    if commit_release:
        code.append(f"setp.ne.b32 pr,$10,0; and.pred pr,pr,e; @pr {arrive} [$10];")
    code.append("}")
    return "\n".join(code)


@asm()
def group4(d, a, b, a_scales, b_scales, accumulate, done, release, pre, *, part, sv, nomma=False):
    """One W4A4 MMA group. ``part``: "fc1" (one 256-K stage: 4 MMAs, A [2 chunks][128 rows] at chunk stride 8 KB,
    B = x stage [2 chunks][8 rows], weight scales 1 KB -> 2 cps, x scales 2 cps if copied), "fc2" (TP8: one tile of
    an FC2 pair: K 0-255 then 256-383, A chunk stride 16 KB, B = hidden image, weight scales 1 KB + 512 B -> 3 cps,
    hidden scales 3 cps if copied), "fc2t16" (TP16: K 0-127 from A chunk 0, K 128-191 from the upper 32 B of A chunk 1
    (W2 elements 64-191), B hidden chunk 1's first 32 B; weight scales 2 x 512 B -> 2 cps (K blocks 0-3, 4-7), hidden
    scales 2 cps if copied)."""
    if part == "fc1":
        steps = [((j // 2) * 512 + (j % 2) * 2, (j // 2) * 32 + (j % 2) * 2, 4 * (j // 2), 4 * (j // 2), (j % 2) * 2)
                 for j in range(4)]
        a_cps, b_cps = [(0, 0), (4, 32)], [(0, 0), (4, 8)]
    elif part == "fc2t16":
        steps = [(j * 2, j * 2, 0, 0, j * 2) for j in range(2)]
        steps += [(1024 + 2, 32, 4, 4, 0)]
        a_cps, b_cps = [(0, 0), (4, 32)], [(0, 0), (4, 8)]
    else:
        steps = [((j // 2) * 1024 + (j % 2) * 2, (j // 2) * 32 + (j % 2) * 2, 4 * (j // 2), 4 * (j // 2), (j % 2) * 2)
                 for j in range(4)]
        steps += [(2048 + j * 2, 64 + j * 2, 8, 8, j * 2) for j in range(2)]
        a_cps, b_cps = [(0, 0), (4, 32), (8, 64)], [(0, 0), (4, 8), (8, 16)]
    code = _group(steps, a_cps, b_cps, sv, nomma, True, True, True)
    kinds = "rllrlrlrrrrr"
    operands = (d, a, b, *a_scales, *b_scales, accumulate, done, release, pre)
    values = [(Uint64 if k == "l" else Uint32)(v) for k, v in zip(kinds, operands)]
    return code, ",".join(kinds) + ",~{memory}", values


@asm()
def tm_acquire_gpu(pointer):
    """Diagnosis (benchmark-only): ``acquire_tensormap`` at .gpu scope."""
    return "fence.proxy.tensormap::generic.acquire.gpu [$0],128; prefetch.tensormap [$0];", "l", (pointer,)


@asm()
def tm_prefetch(pointer):
    """Diagnosis (benchmark-only): the tensormap prefetch without the proxy fence."""
    return "prefetch.tensormap [$0];", "l", (pointer,)


class W4A4ExpertsKernel(MV.StreamExpertsKernel):
    """The W4A4 engine (see the module doc). ``slots``: 64 KB ring slots (3 in the large-SMEM mode, 2 in
    the normal one). TMEM: accumulators 0-127, weight scales 128 + 32 x slot (<= 3 slots), hidden scales 224 + 12 x buffer,
    x scales 256 + 8 x stage."""

    TPS = (8, 16)
    # ``meta_bar`` (x3 meta R4 form, K-10d; set only by the served W8 engine, ``W4A8ExpertsKernel(served=True)``):
    # the MMA warp's 32 lanes read meta[slot], join with ``bar.sync 12, 32`` (PTX 8.9.4) and lane 0 alone arrives on a
    # per-slot count-1 meta barrier (misc ``bar_meta``); the producer waits on it (acquire) before it rewrites meta[slot].
    # empty[slot] then counts 1 (the stage's tcgen05.commit only) and guards the slot's weight bytes. Off: as before.
    meta_bar = False
    queue_bar = False  # W8 served only: queue-empty[] counts the 32 loader lanes (``W4A8ExpertsKernel.pop_unit``)
    # ``fc2_ahead`` (W8 only, default 1 = unchanged): FC2 pair claims in flight per producer. Claims by one
    # thread to one counter return increasing values (same-location coherence follows program order), so the first claim
    # >= pairs ends the loop exactly as today; surplus claims only advance the counter (re-armed per parity).
    fc2_ahead = 1
    # ``fc2_quad`` (W8 only, default False = unchanged): FC2 ops of four W2 tiles (two tile-pair boxes, one
    # 64 KB slot), claimed as quads (``jobs // 2`` per unit) in expert order; see ``fc2_quad_op`` / W8 ``consume``.
    fc2_quad = False

    def __init__(self, tp, capacity, experts, grid, publish=False, activation_floor=None, slots=3, hbufs=2,
                 ablate=(), sv=1, xone=False, deadline_ns=0):
        self.slots_req = slots
        # ``deadline_ns`` (> 0, tests): the FC2 loader's poll of another CTA's hidden scale words fail-stops (trap)
        # after that long instead of spinning forever on a protocol bug (hole H3); 0 = no check (production).
        self.deadline_ns = deadline_ns
        # Hidden image bytes / 128-K chunks; W2 64 B chunks per FC2 op row; FC2 TMEM / SMEM scale stride per tile.
        self.h4, self.hk = (H_IMAGE4, 3) if tp == 8 else (1024, 2)
        self.w2_chunks = 3 if tp == 8 else 2
        self.fc2_scale_cols, self.fc2_scale_units = (12, 96) if tp == 8 else (8, 64)
        super().__init__(tp, capacity, experts, grid, publish, activation_floor, stages=slots, hbufs=hbufs,
                         ablate=ablate, xone=xone)
        self.x_bytes = 14336
        assert hbufs == 2 and set(self.ablate) <= {"hid", "epi", "mma"}, (hbufs, self.ablate)
        self.sv = sv  # instruction-descriptor sparsity version bit: 1 on sm_107a, 0 on sm_100a / sm_103a (Table 53)

    def layout(self, stages):
        self.h_image_bytes, self.h_compact_bytes = self.h4, self.h4 + 128
        slots = stages
        assert 2 <= slots <= 3, slots
        self.stages = slots
        self.slot_bytes = SLOT
        self.off_sfa = slots * SLOT
        self.off_x = self.off_sfa + slots * SLOT_SF
        self.off_h = self.off_x + 14 * 1024
        self.off_xsf = self.off_h + self.hbufs * self.h4
        self.off_hsf = self.off_xsf + 28 * 128 + 384
        self.off_misc = self.off_hsf + self.hbufs * self.h_scale_bytes
        self.smem_bytes = self.off_misc + 5120
        self.bar_full, self.bar_empty, self.bar_accfull = (0, 8 * slots, 16 * slots)
        acc = self.bar_accfull
        self.bar_accempty, self.bar_hfull = (acc + 128, acc + 256)
        self.bar_hempty, self.bar_xready = (acc + 272, acc + 288)
        self.off_meta = self.off_misc + 4224
        self.off_queue = self.off_misc + 4608
        assert self.off_misc <= 1 << 18 and self.smem_bytes <= PLN.large_smem_alloc(), (self.off_misc, self.smem_bytes)
        assert self.off_h % 512 == 0 and self.off_x % 1024 == 0

    # ------------------------------------------------------------------------------ producer
    @cute.jit
    def fetch_x(self, sm, maps):
        """K stage b's 8-token MXFP4 slice (1 KB, [2 chunks][8 rows][64 B], map at +768) and its scale words
        (256 B, map at +896) onto xready[b]; rows >= batch are zero-filled."""
        ready = view(sm.iterator, self.off_misc + self.bar_xready, Uint64, 14)
        if cutlass.const_expr(self.xone):
            bar = ready.iterator
            cute.arch.mbarrier_arrive_and_expect_tx(bar, 14336 + 28 * 128)
            PLN.load_tma3_keep(sm.iterator + self.off_x, maps + 768, bar, 0, 0)
            PLN.load_tma3_keep(sm.iterator + self.off_xsf, maps + 896, bar, 0, 0)
        else:
            for b in cutlass.range(14, unroll=1):
                bar = ready.iterator + b
                cute.arch.mbarrier_arrive_and_expect_tx(bar, 1024 + 256)
                PLN.load_tma3_keep(sm.iterator + self.off_x + b * 1024, maps + 768, bar, 0, 2 * b)
                PLN.load_tma3_keep(sm.iterator + self.off_xsf + b * 256, maps + 896, bar, 0, 2 * b)

    @cute.jit
    def op(self, ring, state, meta, desc, row, chunk, wbytes, scales, sbytes, policy):
        """One TMA op into the next slot: metadata, expect_tx, weights box, scale bytes."""
        sm, barriers, metadata = ring
        slot, phase, issued = state
        if issued >= self.stages:
            self.sync(barriers.iterator + self.stages + slot, phase ^ 1)
            if cutlass.const_expr(self.meta_bar):  # R4: every MMA-warp lane's read of meta[slot] (use - 3) is done
                metabar = view(sm.iterator, self.off_misc + self.bar_meta, Uint64, self.stages)
                self.sync(metabar.iterator + slot, phase ^ 1)
        for i in cutlass.range_constexpr(4):
            metadata[4 * slot + i] = meta[i]
        full = barriers.iterator + slot
        cute.arch.mbarrier_arrive_and_expect_tx(full, wbytes + sbytes)
        MV.tma3(sm.iterator + slot * SLOT, desc, full, row, chunk, policy)
        MV.bulk(sm.iterator + self.off_sfa + slot * SLOT_SF, scales, full, sbytes, policy)
        slot += 1
        if slot == self.stages:
            slot, phase = (Int32(0), phase ^ 1)
        return slot, phase, issued + 1

    @cute.jit
    def fc1_ops(self, ring, state, maps, w13_scales, experts, t, lo, hi, policy):
        """Ops [lo, hi) of FC1 tile t: op i = K stages 4i..4i+3 (op 3: stages 12-13)."""
        unit, r = (t // self.tiles, t % self.tiles)
        expert = experts[unit]
        row = expert * self.w13_rows + r * 128
        base = Int64(expert) * self.w13_scale_bytes + r * 14336
        for i in cutlass.range(lo, hi, unroll=1):
            tail = Int32(i == 3)
            stages = 4 - 2 * tail
            meta = (Int32(0), t, 4 * i, stages)
            work = (maps.iterator + 128 * tail, row, 8 * i, stages * 16384, w13_scales.iterator + base + 4096 * i,
                    stages * 1024)
            state = self.op(ring, state, meta, *work, policy)
        return state

    @cute.jit
    def fc2_op(self, ring, state, maps, w2_scales, expert, unit, tile, policy):
        """FC2 tiles ``tile`` and ``tile + 1``: 256 rows x 384 K (48 KB) and their 3 KB of scales (TP16: 256 rows x two
        64 B chunks, 32 KB, and 2 KB of scales)."""
        row = expert * 3584 + tile * 128
        offset = Int64(expert) * self.w2_scale_bytes + tile * self.w2_tile_scale_bytes
        meta = (Int32(1), unit, tile, Int32(0))
        wbytes: cutlass.Constexpr = 256 * 64 * self.w2_chunks
        sbytes: cutlass.Constexpr = 2 * self.w2_tile_scale_bytes
        work = (maps.iterator + 256, row, Int32(0), Int32(wbytes), w2_scales.iterator + offset, Int32(sbytes))
        return self.op(ring, state, meta, *work, policy)

    @cute.jit
    def fc2_quad_op(self, ring, state, maps, w2_scales, expert, unit, tile, policy):
        """C4: FC2 tiles ``tile`` .. ``tile + 3`` of ``unit``: two tile-pair boxes (``fc2_op``'s box, rows + 0 / + 256) into one
        slot (bytes 0 / 32 KB) and their 4 KB of scales (contiguous per expert), under one arrive.expect_tx."""
        sm, barriers, metadata = ring
        slot, phase, issued = state
        if issued >= self.stages:
            self.sync(barriers.iterator + self.stages + slot, phase ^ 1)
            if cutlass.const_expr(self.meta_bar):  # R4: every MMA-warp lane's read of meta[slot] (use - 3) is done
                metabar = view(sm.iterator, self.off_misc + self.bar_meta, Uint64, self.stages)
                self.sync(metabar.iterator + slot, phase ^ 1)
        meta = (Int32(1), unit, tile, Int32(4))
        for i in cutlass.range_constexpr(4):
            metadata[4 * slot + i] = meta[i]
        row = expert * 3584 + tile * 128
        offset = Int64(expert) * self.w2_scale_bytes + tile * self.w2_tile_scale_bytes
        box: cutlass.Constexpr = 256 * 64 * self.w2_chunks
        sbytes: cutlass.Constexpr = 4 * self.w2_tile_scale_bytes
        full = barriers.iterator + slot
        cute.arch.mbarrier_arrive_and_expect_tx(full, 2 * box + sbytes)
        MV.tma3(sm.iterator + slot * SLOT, maps.iterator + 256, full, row, Int32(0), policy)
        MV.tma3(sm.iterator + slot * SLOT + box, maps.iterator + 256, full, row + 256, Int32(0), policy)
        MV.bulk(sm.iterator + self.off_sfa + slot * SLOT_SF, w2_scales.iterator + offset, full, Int32(sbytes), policy)
        slot += 1
        if slot == self.stages:
            slot, phase = (Int32(0), phase ^ 1)
        return slot, phase, issued + 1

    @cute.jit
    def stream_warp(self, sm, maps, w13_scales, w2_scales, distinct, experts, workspace):
        """Lane 0 of warp 0: the planned producer's order (``PlannedExpertsKernel.stream_mixed``) on W4A4 ops."""
        _, _, lane, cta = thread()
        if lane == 0:
            barriers = view(sm.iterator, self.off_misc, Uint64, 2 * self.stages)
            metadata = view(sm.iterator, self.off_meta, Int32, 4 * self.stages)
            ring = (sm, barriers, metadata)
            policy = MV.evict_first_policy()
            if cutlass.const_expr(getattr(self, "tm_mode", "") == "late"):  # tid 0's acquires, moved from init
                for offset in (0, 128, 256, 768, 896):  # to just before its first tensormap use (fetch_x / fc1 ops)
                    acquire_tensormap(maps.iterator + offset)
            self.fetch_x(sm, maps.iterator)
            state = (Int32(0), Int32(0), Int32(0))
            tiles = self.tiles * distinct
            t = Int32(cta)
            head: cutlass.Constexpr = min(self.stages, 4)
            if t < tiles:
                state = self.fc1_ops(ring, state, maps, w13_scales, experts, t, 0, head, policy)
            calls = self.calls(workspace)
            call = calls[cta]
            parity = call & 1
            if cta == 0:
                store_words(self.claim_counter(workspace, parity ^ 1), (Uint32(4294967295),))
                store_words(self.claim_counter(workspace, 2 + (parity ^ 1)), (Uint32(4294967295),))
            tile_counter = self.claim_counter(workspace, 2 + parity)
            pending = claim(tile_counter, 1)
            if t < tiles:
                state = self.fc1_ops(ring, state, maps, w13_scales, experts, t, head, 4, policy)
                t = self.grid + Int32(pending + Uint32(1))
                if t < tiles:
                    pending = claim(tile_counter, 1)
            while t < tiles:
                state = self.fc1_ops(ring, state, maps, w13_scales, experts, t, 0, 4, policy)
                t = self.grid + Int32(pending + Uint32(1))
                if t < tiles:
                    pending = claim(tile_counter, 1)
            counter = self.claim_counter(workspace, parity)
            pairs = self.jobs * distinct
            quads = Int32(0)
            if cutlass.const_expr(self.fc2_quad):  # C4b: quads (jobs // 2 per unit) for the units whose FC1 tiles are all
                # in the first wave, tile pairs for the last ``tail`` units (2nd-wave FC1 tiles: their hidden lands last, and
                # fine ops keep the end balanced). Claim index: quads of units 0 .. distinct - tail - 1, then pairs.
                tail = Int32(0)
                if self.tiles * distinct > self.grid:
                    tail = (self.tiles * distinct - self.grid + self.tiles - 1) // self.tiles
                quads = self.jobs // 2 * (distinct - tail)
                pairs = quads + self.jobs * tail
            pending = claim(counter, 1)
            pending2 = pending
            if cutlass.const_expr(self.fc2_ahead == 2):  # C1: the second claim in flight before the first is used
                pending2 = claim(counter, 1)
            pushed, current, active = (Int32(0), Int32(-1), True)
            while active:
                q = Int32(pending + Uint32(1))
                active = q < pairs
                if active:
                    ahead = claim(counter, 1)
                    unit, pair = (q // self.jobs, q % self.jobs)
                    if cutlass.const_expr(self.fc2_quad):
                        if q < quads:
                            unit, pair = (q // (self.jobs // 2), q % (self.jobs // 2))
                        else:
                            unit = quads // (self.jobs // 2) + (q - quads) // self.jobs
                            pair = (q - quads) % self.jobs
                    if unit != current:
                        pushed = self.push_unit(sm, pushed, unit)
                        current = unit
                    if cutlass.const_expr(self.fc2_quad):
                        if q < quads:
                            state = self.fc2_quad_op(ring, state, maps, w2_scales, experts[unit], unit, 4 * pair, policy)
                        else:
                            state = self.fc2_op(ring, state, maps, w2_scales, experts[unit], unit, 2 * pair, policy)
                    else:
                        state = self.fc2_op(ring, state, maps, w2_scales, experts[unit], unit, 2 * pair, policy)
                    if cutlass.const_expr(self.fc2_ahead == 2):
                        pending = pending2
                        pending2 = ahead
                    else:
                        pending = ahead
            self.push_unit(sm, pushed, Int32(-1))
            slot, phase, issued = state
            if issued >= self.stages:
                self.sync(barriers.iterator + self.stages + slot, phase ^ 1)
                if cutlass.const_expr(self.meta_bar):  # R4: the end marker rewrites meta[slot] too
                    metabar = view(sm.iterator, self.off_misc + self.bar_meta, Uint64, self.stages)
                    self.sync(metabar.iterator + slot, phase ^ 1)
            for i in cutlass.range_constexpr(4):
                metadata[4 * slot + i] = Int32(2 if i == 0 else 0)
            cute.arch.mbarrier_arrive(barriers.iterator + slot, arrive_count=1)
            self.advance_calls(calls, cta, call)

    # ------------------------------------------------------------------------------ MMA warp
    @cute.jit
    def consume(self, sm, tmem, workspace=None):
        lane = thread()[2]
        slots, H = (self.stages, self.hbufs)
        nomma: cutlass.Constexpr = "mma" in self.ablate
        sv: cutlass.Constexpr = self.sv
        bars: tuple = (self.bar_full, self.bar_empty, self.bar_accfull, self.bar_accempty, self.bar_xready)
        full, empty, accfull, accempty, xready = (
            view(sm.iterator, self.off_misc + o, Uint64, n) for o, n in zip(bars, (slots, slots, 16, 16, 14))
        )
        hfull, hempty = self.hidden_barriers(sm)
        meta = view(sm.iterator, self.off_meta, Int32, 4 * slots)
        accmeta = view(sm.iterator, self.off_misc + 600, Int32, 64)
        a0, x0, h0 = (desc64(sm.iterator + o) for o in (0, self.off_x, self.off_h))
        asf0, xsf0, hsf0 = (smem_desc(sm.iterator + o, False) for o in (self.off_sfa, self.off_xsf, self.off_hsf))
        xmask, ended = (Uint32(0), False)
        uses, acc, count = (Int32(0), Int32(0), Int32(0))
        current, ordinal, hbuffer = (Int32(-1), Int32(-1), Int32(0))
        while not ended:
            slot = count % slots
            self.sync(full.iterator + slot, count // slots & 1, True)
            kind, a, b, c = (meta[4 * slot + i] for i in range(4))
            # U1-W3 (= U1-9) fix, L7 ``fx_u1w_meta_arrive``: the producer's next metadata write to this slot waits
            # for a generic release arrive after every lane's metadata read (empty count 2: this + the commit).
            cute.arch.sync_warp()
            if lane == 0:
                cute.arch.mbarrier_arrive(empty.iterator + slot)
            released = shared_address(empty.iterator + slot)
            wcol = tmem + 128 + slot * 32
            if kind == 2:
                if (lane == 0) & (current >= 0):
                    commit(hempty.iterator + hbuffer)
                for _ in cutlass.range(2, unroll=1):
                    self.acquire(accempty, acc, uses)
                    self.signal(accmeta, accfull, acc, (kind, a, b, c), 2)
                    uses += 1
                    acc = uses % 16
                ended = True
            elif kind == 0:  # FC1 op of tile a: K stages b .. b + c - 1
                for s in cutlass.range_constexpr(4):
                    if s < c:
                        ks = b + s
                        first, last = (ks == 0, ks == 13)
                        if first:
                            self.acquire(accempty, acc, uses)
                        copy_x = Uint32(0)
                        if xmask >> ks & 1 == 0:
                            self.sync(xready.iterator + (0 if self.xone else ks), 0, True)
                            xmask |= Uint32(1) << ks
                            copy_x = Uint32(1)
                        done = Uint32(0)
                        if last:
                            done = shared_address(accfull.iterator + acc)
                        release = Uint32(0)
                        if s == c - 1:
                            release = released
                        group4(
                            tmem + acc * 8, a0 + slot * (SLOT // 16) + s * 1024, x0 + ks * 64,
                            (wcol + s * 8, asf0 + slot * (SLOT_SF // 16) + s * 64),
                            (tmem + 256 + ks * 8, xsf0 + ks * 16, copy_x),
                            Uint32(not first), done, release, Uint32(0), part="fc1", sv=sv, nomma=nomma,
                        )
                        cute.arch.sync_warp()
                        if last:
                            self.signal(accmeta, accfull, acc, (Int32(0), a, ks, Int32(2)))
                            uses += 1
                            acc = uses % 16
            else:  # FC2 pair op: unit a, tiles b and b + 1
                copy_h, pre = (Uint32(0), Uint32(0))
                if a != current:
                    if current >= 0:
                        pre = shared_address(hempty.iterator + hbuffer)
                    current = a
                    ordinal += 1
                    hbuffer = ordinal % H
                    self.sync(hfull.iterator + hbuffer, ordinal // H & 1, True)
                    copy_h = Uint32(1)
                hd = h0 + hbuffer * (self.h4 // 16)
                hs = hsf0 + hbuffer * (self.h_scale_bytes // 16)
                sfbc = tmem + 224 + hbuffer * 12
                self.acquire(accempty, acc, uses)
                self.acquire(accempty, (uses + 1) % 16, uses + 1)
                for t in cutlass.range_constexpr(2):
                    target = acc if t == 0 else (uses + 1) % 16
                    group4(
                        tmem + target * 8, a0 + slot * (SLOT // 16) + t * 512, hd,
                        (wcol + self.fc2_scale_cols * t, asf0 + slot * (SLOT_SF // 16) + t * self.fc2_scale_units),
                        (sfbc, hs, copy_h if t == 0 else Uint32(0)),
                        Uint32(0), shared_address(accfull.iterator + target), released if t == 1 else Uint32(0),
                        pre if t == 0 else Uint32(0), part="fc2" if self.tp == 8 else "fc2t16", sv=sv, nomma=nomma,
                    )
                    cute.arch.sync_warp()
                for t in cutlass.range_constexpr(2):
                    self.signal(accmeta, accfull, acc, (Int32(1), a, b + t, Int32(0)))
                    uses += 1
                    acc = uses % 16
            count += 1

    # ------------------------------------------------------------------------------ hidden hand-off
    @cute.jit
    def store_hidden(self, sm, workspace, values, tile, parity, scalars):
        """FC1 epilogue of one tile (64 hidden columns): SiTU, MXFP4 per 32 columns, the 32 B per token at their
        64B-swizzled place, then (after the group barrier) warp 0 releases each token's two block scales."""
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
            scale = hidden4_exponent(cute.arch.fmax(maxima[t], peer))
            if lane == t:
                scale_token = scale
                if ew == 0:
                    high = cute.arch.fmax(shared_max[(eg * 4 + 2) * 8 + t], shared_max[(eg * 4 + 3) * 8 + t])
                    scale_high = hidden4_exponent(high)
            v = h[t] * mx_rescale(scale)
            odd = cute.arch.shuffle_sync_down(v, 16)
            if lane < 8:
                sm[stage_base + t * 8 + lane] = Uint8(e2m1x2(odd, v))
        cute.arch.sync_warp()
        image = self.image(workspace, parity, expert)
        if lane < 8:
            data = (Uint32(0), Uint32(0))
            if lane < self.width:
                data = load_words(sm.iterator + stage_base + lane * 8, count=2, shared=True, mutable=True)
            piece = (2 * (chunk % 2) + ew // 2) ^ (lane >> 1 & 3)
            store_words(image + chunk // 2 * 512 + lane * 64 + piece * 16 + 8 * (ew % 2), data)
        group_barrier(eg)
        if (ew == 0) & (lane < 8):
            pair = (Uint32(scale_token) & 255) | (Uint32(scale_high) & 255) << 8
            store_release_u16(image + self.h4 + lane * 16 + 2 * chunk, Uint16(pair))
        group_barrier(eg)

    @cute.jit
    def poll_hidden(self, sm, source, buffer: Int32):
        """FC2 loader: acquire each token's tile scale words (never 0xFF once published), then copy the image.
        TP8: 8 tokens x 6 tiles of 2 scale bytes, lanes 4t..4t+2 hold token t's 6 pairs. TP16: 3 tiles, lane 4t holds
        bytes 0-3 and lane 4t + 1 bytes 4-5 (bytes 6-7 are never published: masked)."""
        lane = thread()[2]
        mask = Uint32(4294967295)
        if cutlass.const_expr(self.tp == 8):
            valid = (lane & 3) < 3
        else:
            valid = (lane & 3) < 2
            if lane & 3 == 1:
                mask = Uint32(65535)
        scale, ready = (Uint32(0), False)
        at = source + self.h4 + lane * 4
        started, polls = (wait_deadline_start(self.deadline_ns), Int32(0))
        while not ready:
            scale = Uint32(load_acquire_u16(at)) | Uint32(load_acquire_u16(at + 2)) << 16
            bad = False
            if valid:
                bad = has_ff_byte(scale & mask)
            ready = cute.arch.vote_all_sync(not bad)
            if not ready:
                polls = wait_deadline_check(started, polls, self.deadline_ns)
        cute.arch.sync_warp()
        image = sm.iterator + self.off_h + buffer * self.h4
        for v in cutlass.range_constexpr(self.hk):
            words = load_words(source + (lane + 32 * v) * 16, count=4, mutable=True)
            store_shared_words(image + (lane + 32 * v) * 16, words)
        word = cute.arch.shuffle_sync(scale, 4 * (lane & 7) + (lane >> 3))
        if lane < 8 * self.hk:
            sf = sm.iterator + self.off_hsf + buffer * self.h_scale_bytes
            store_shared_words(sf + (lane >> 3) * 128 + (lane & 7) * 16, (word,) * 4)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_warp()

    # ------------------------------------------------------------------------------ setup
    @cute.jit
    def initialize(self, sm, maps, batch):
        tid = thread()[0]
        count_all: cutlass.Constexpr = 2 * self.stages + 50 + (self.stages if self.meta_bar else 0)
        barriers = view(sm.iterator, self.off_misc, Uint64, count_all)
        tmem_base = view(sm.iterator, self.off_misc + 856, Uint32, 1)
        if tid == 0:
            for i in cutlass.range_constexpr(count_all):
                acc = 2 * self.stages
                count = 1 if not acc <= i < acc + 32 else 2 if i < acc + 16 else 4
                if self.stages <= i < 2 * self.stages:  # empty[slot]: the commit + the U1-W3 generic arrive
                    count = 1 if self.meta_bar else 2  # R4: the commit only (meta[slot] has its own barrier)
                cute.arch.mbarrier_init(barriers.iterator + i, count)
            queue = view(sm.iterator, self.off_misc + self.bar_queue, Uint64, 16)
            for i in cutlass.range_constexpr(16):  # queue full [8] count 1; empty [8] count 1, or 32 (every loader lane) under queue_bar
                cute.arch.mbarrier_init(queue.iterator + i, 32 if (i >= 8 and self.queue_bar) else 1)
            hbars = view(sm.iterator, self.off_misc + MV.HBARS, Uint64, 2 * self.hbufs)
            for i in cutlass.range_constexpr(2 * self.hbufs):
                cute.arch.mbarrier_init(hbars.iterator + i, 1)
            cute.arch.mbarrier_init_fence()
            tm: cutlass.Constexpr = getattr(self, "tm_mode", "")  # W8 kwarg, benchmark-only ("" = unchanged)
            if cutlass.const_expr(tm == ""):
                for offset in (0, 128, 256, 768, 896):
                    acquire_tensormap(maps.iterator + offset)
            elif cutlass.const_expr(tm == "gpu"):  # .gpu-scope proxy acquire (CUTLASS's scope)
                for offset in (0, 128, 256, 768, 896):
                    tm_acquire_gpu(maps.iterator + offset)
            elif cutlass.const_expr(tm in ("pf", "late", "gc")):  # prefetch only here (pf: no fence at all, diagnosis only;
                for offset in (0, 128, 256, 768, 896):  # late: the producer issues the same .sys acquires before its first TMA;
                    tm_prefetch(maps.iterator + offset)  # gc: __grid_constant__ maps, no acquire needed
            elif cutlass.const_expr(tm == "one"):  # one .sys acquire (map 0) + prefetch of the rest (per-fence cost; diagnosis)
                acquire_tensormap(maps.iterator)
                for offset in (128, 256, 768, 896):
                    tm_prefetch(maps.iterator + offset)
        if cutlass.const_expr(getattr(self, "tm_mode", "") == "par"):  # one .sys acquire per thread, lane 0
            for i in cutlass.range_constexpr(5):  # of warps 4-8 (not in routing); tid 0 uses them after route()'s CTA barrier
                if tid == 128 + 32 * i:  # (CUDA Programming Guide: acquire by one thread, use by others after __syncthreads)
                    acquire_tensormap(maps.iterator + (0, 128, 256, 768, 896)[i])
        if cutlass.const_expr(getattr(self, "tm_mode", "") == "t128"):  # the .sys acquires by tid 128
            if tid == 128:
                for offset in (0, 128, 256, 768, 896):
                    acquire_tensormap(maps.iterator + offset)
        if tid // 32 == 3:
            cute.arch.alloc_tmem(512, tmem_base.iterator)
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.fence_proxy("async.shared", space="cta")
        return tmem_base


def enabled():
    """MK switch, default off (a lossy option): ``K31MK_EXPERTS_W4A4=1`` runs the TP16 routed experts
    of every non-publish batch <= 8 on this engine; TP8, publish / two-shot batches (TP16 B >= 9, the grouped engine
    with its own image region) and everything else keep their W4A8 engines. There is no per-batch subset: the W4A4
    hidden images (1152 B per unit at TP16) and the shipped non-grouped engine's (2176 B) share one workspace region
    and parity scheme, so a call of one engine could poll the other's stale images (the re-arm covers only its own
    layout)."""
    return os.environ.get("K31MK_EXPERTS_W4A4", "0") == "1"


def options():
    """The configuration the Rubin gate times (arm b16): the x plan in one TMA per tensor (``xone``, L1 r10)."""
    return {"xone": True}


# --------------------------------------------------------------------------------- x plan (MXFP4)
class W4A4PlanKernel(ExpertsPlanKernel):
    """FP32 x -> the MXFP4 plan, one CTA per token (``planned.EarlyPlanKernel``'s release at entry and its
    generic -> async proxy fences)."""

    @cute.kernel
    def kernel(self, args):
        x, plan = args
        tid, _, lane, token = thread()
        cute.arch.griddepcontrol_launch_dependents()
        cute.arch.griddepcontrol_wait()
        PLN.fence_proxy_async_global()
        for i in cutlass.range_constexpr(2):
            chunk = tid + 256 * i
            if chunk < 448:  # warp-uniform: chunks 256-447 are warps 0-5; lanes 4k..4k+3 = one 32-block
                source = x.iterator + token * 3584 + chunk * 8
                words = (*load_words(source, count=4, mutable=True), *load_words(source + 4, count=4, mutable=True))
                values = cute.make_rmem_tensor(8, Float32)
                for j in cutlass.range_constexpr(8):
                    values[j] = bits_float(words[j])
                top = abs(values[0])
                for j in cutlass.range_constexpr(1, 8):
                    top = cute.arch.fmax(top, abs(values[j]))
                for width in (1, 2):
                    top = cute.arch.fmax(top, cute.arch.shuffle_sync_bfly(top, width))
                scale = mx4_exponent(max(top, Float32(1e-10)))
                reciprocal = mx_rescale(scale)
                packed = e2m1x8(tuple(values[j] * reciprocal for j in range(8)))
                store_words(plan.iterator + token * 1792 + chunk * 4, (packed,))
                word = Uint32(scale)
                for s in cutlass.range_constexpr(1, 4):
                    word |= cute.arch.shuffle_sync(Uint32(scale), lane + 4 * s) << 8 * s
                if chunk % 16 == 0:
                    dst = plan.iterator + PLAN4_XSF + token * 448 + chunk // 16 * 16
                    store_words(dst, (word, Uint32(0), Uint32(0), Uint32(0)))
        PLN.fence_proxy_async_global()


@functools.cache
def compile_plan4(batch, device):
    args = (fake((batch, 3584), None, cutlass.Float32), fake((PLN.PLAN_BYTES,), (1,), cutlass.Uint8),
            make_fake_stream())
    with torch.accelerator.device_index(device):
        return cute.compile(W4A4PlanKernel(batch), *args)


def maps4(w13, w2, plan, batch, device, atoms=2):
    """Packed-U8 TMA maps (64B swizzle): W13 boxes of 8 / 4 chunks of 128 K x 128 rows, W2 tile pairs (TP8: 256 rows x
    3 chunks of 64 B; TP16: 256 rows x 2 chunks of 64 B at byte offsets 0 and 32 of the 128 B row, i.e. W2 elements
    0-127 and 64-191), the MXFP4 plan's x (8 rows x 2 chunks per K stage) and its scale words."""
    import cuda.bindings.driver as cuda

    from ..common.host import tensor_maps

    e, rows13 = w13.shape[0], w13.shape[1]
    w2_row, w2_chunks, w2_stride = (192, 3, 64) if rows13 == 768 else (128, 2, 32)
    u8 = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8
    sw = cuda.CUtensorMapSwizzle
    l2 = cuda.CUtensorMapL2promotion
    promo, none = l2.CU_TENSOR_MAP_L2_PROMOTION_L2_256B, l2.CU_TENSOR_MAP_L2_PROMOTION_NONE
    s64, s0 = sw.CU_TENSOR_MAP_SWIZZLE_64B, sw.CU_TENSOR_MAP_SWIZZLE_NONE
    base = plan.data_ptr()
    specs = [
        (w13.data_ptr(), u8, (64, e * rows13, 28), (1792, 64), (64, 128, 8), s64, promo),
        (w13.data_ptr(), u8, (64, e * rows13, 28), (1792, 64), (64, 128, 4), s64, promo),
        (w2.data_ptr(), u8, (64, e * 3584, w2_chunks), (w2_row, w2_stride), (64, 256, w2_chunks), s64, none),
        (base, u8, (64, batch, 28), (1792, 64), (64, 8, atoms), s64, none),
        (base + PLAN4_XSF, u8, (16, batch, 28), (448, 16), (16, 8, atoms), s0, none),
    ]
    encoded = tensor_maps(specs, device)
    full = torch.zeros(MAPS4, dtype=torch.uint8, device=device)
    full[:384] = encoded[:384]
    full[768:1024] = encoded[384:640]
    return full


@functools.cache
def compile_w4a4(tp, experts, batch, grid, device, activation_floor=False, knobs=()):
    capacity = 1 << (batch - 1).bit_length()
    moe = W4A4ExpertsKernel(tp, capacity, experts, grid, False, activation_floor, **dict(knobs))
    flat = lambda n, dtype=cutlass.Uint8, align=16: fake((n,), (1,), dtype, align)  # noqa: E731
    ids = flat(batch * 16, Int32)
    scales = (flat(experts * n) for n in (moe.w13_scale_bytes, moe.w2_scale_bytes))
    out = flat(batch * 16 * 3584, cutlass.BFloat16)
    tensors = (flat(PLN.PLAN_BYTES), ids, flat(MAPS4, align=64), *scales, flat(moe.workspace_bytes), out)
    scalars = (Int32(batch), *map(cutlass.Float32, (4, 25, 0)))
    with torch.accelerator.device_index(device):
        code = cute.compile(moe, *tensors, *scalars, make_fake_stream())
        if moe.smem_bytes > 227 * 1024:
            PLN.set_oversized(code, device)
        return code, moe.smem_bytes


# TP16 W4A4 grid by architecture. Whole-tile FC1 has 3 x E_d tiles: when one CTA per SM leaves some CTAs with an extra
# tile, the late experts' FC2 waits on them. Measured (B8, the c=1 verify's E_d): a reduced grid is faster on the
# architecture in the table below.
# ``K31MK_EXPERTS_W4A4_GRID`` overrides (TP16 only).
W4A4_FREE = {(10, 7): 52}  # TP16: SMs left free beside the grid (grid = SM count - free), by compute capability


class W4A4Experts(PLN.PlannedExperts):
    """W4A4 routed experts for every TP8 / TP16 non-publish batch <= 8 (all of them: see ``enabled``); publish,
    two-shot and larger batches keep ``PlannedExperts`` / ``PersistentExperts``. ``slots``: 64 KB ring slots."""

    def __init__(self, *args, batches=None, slots=None, ablate=(), xone=False,
                 deadline_ns=0, **kwargs):
        super().__init__(*args, stages=6, xplan=True, batches=batches, **kwargs)
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        grid = os.environ.get("K31MK_EXPERTS_W4A4_GRID")
        free = W4A4_FREE.get(torch.cuda.get_device_capability(self.device))
        self.w4_grid = (int(grid) if grid else None if free is None else sms - free) if self.tp == 16 else None
        major, minor = torch.cuda.get_device_capability(self.device)
        self.knobs = (("ablate", tuple(sorted(ablate))), ("slots", slots or (3 if torch.cuda.get_device_capability(self.device) == (10, 7) else 2)),
                      ("sv", 1 if (major, minor) == (10, 7) else 0),
                      ("xone", xone), ("deadline_ns", deadline_ns))

    def grid_for(self, batch):
        """TP16 batches <= 8 (all W4A4: none publishes) use the W4A4 grid; the MK front reads it too (a grid below the SM
        count lets the shared expert run beside the experts)."""
        if self.w4_grid and batch <= 8:
            return self.w4_grid
        return super().grid_for(batch)

    def prepare(self, batch, buffers, out, publish=(), planned=False, twoshot=False, early=False):
        """``planned``: the caller writes the MXFP4 plan (``compile_plan4``) before ``run`` (never an MXFP8 one)."""
        if publish or twoshot or batch > 8:
            return super().prepare(batch, buffers, out, publish, planned, twoshot, early)
        need = PLN.planned_workspace_bytes(ER.ExpertsGeometry(self.tp).workspace_bytes)
        if buffers["workspace"].numel() < need:
            raise ValueError("W4A4 experts need planned.scratch() workspace")
        grid = self.grid_for(batch)
        code, smem = compile_w4a4(self.tp, self.experts, batch, grid, self.device.index, self.activation_floor,
                                  self.knobs)
        if grid > torch.cuda.get_device_properties(self.device).multi_processor_count:
            raise RuntimeError(f"W4A4 experts: grid {grid} > SMs")
        w13, _, w2, _ = self.weights
        maps = maps4(w13, w2, buffers["plan"], batch, self.device, 28 if dict(self.knobs)["xone"] else 2)
        self.batch_maps[batch] = maps
        plan = (ER.tensor(buffers["plan"]), compile_plan4(batch, self.device.index))
        if planned:  # the caller writes the plan before ``run``
            plan = plan[:1]
        ws = ER.tensor(buffers["workspace"])
        self.prepared[batch] = (out, ER.tensor(out), ws, code, (), plan, ER.to_cute(maps, 64), None)
