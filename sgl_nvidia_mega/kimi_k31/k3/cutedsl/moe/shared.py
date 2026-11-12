# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fused shared expert for B>8: gate/up, SiTU, down and the routed tail's reduce-scatter
push in one persistent tcgen05 kernel, synchronized by DSMEM and one global-memory
barrier instead of kernel bounds.

120 CTAs of 160 threads in clusters of ks = 120 / tiles (TP8 5, TP16 10). Loader warps
0-3 feed a 128B-swizzled ring of W [64, 64] (TMA) and x [N, 64] (cp.async) K blocks to
warp 4, which issues each block's four M=64 kind::f16 MMAs warp-uniformly
(``mma_bf16_block``). Gate/up M-tile m (32 gate rows and the matching 32 up rows)
belongs to cluster m, whose CTAs split its K. Each CTA pushes its FP32 partial rows
over DSMEM to the owner of their h columns (``width`` columns per owner); an owner
thread per (column, token) sums the parts in K (rank) order, rounds gate and up to BF16
for SiTU, and the owner stores h rows as vectors. After the grid's h barrier, CTAs
below 112 run 64 down rows (the first ring's worth of weights already staged) and
peer-store them, -0 made +0, into the tail's RS mailboxes. The last CTA through the
barrier re-arms the counters."""

import functools

import cuda.bindings.driver as cuda
import cutlass as c
import cutlass.cute as cute
import torch
from cutlass import Constexpr, Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream

from ..common.host import fake, tensor_maps
from ..common.primitives import (
    acquire_tensormap,
    alloc,
    asm,
    at,
    bits_float,
    commit_asm,
    copy_g2s,
    copy_s2cluster,
    div_rn,
    fence_tmem,
    float_bits,
    launch,
    load_tma2,
    load_tmem,
    mma_bf16_block,
    multicast_words,
    named_barrier,
    pack_pair,
    push_float,
    release_tmem,
    smem_desc,
    smem_words,
    split,
    store_relaxed,
    store_words,
    thread,
    view,
    wait,
)
from ..gemm.ws_gemm import positive_zero32

GRID, ROWS = 120, 112  # CTAs; CTAs with 64 down rows
# With the MoE front rows: the K split by TP and a deeper ring that still fits beside
# three o-proj consumer CTAs (<= 30 KB each; the kernel launches at the consumer's
# entry and loads its first ring meanwhile).
FRONT_KS, FRONT_RING = {8: 4, 16: 6}, {8: 112 * 1024, 16: 96 * 1024}
COUNTERS = 4  # h ready, exit


@asm(Float32, pure=True)
def situ_gate(gate, beta, inv_beta):
    code = "{ .reg .f32 x,t,e,d,r; mul.ftz.f32 x,$1,$3; "
    code += "tanh.approx.f32 t,x; mul.ftz.f32 t,$2,t; "
    code += "mul.ftz.f32 x,$1,0fbfb8aa3b; ex2.approx.ftz.f32 e,x; "
    code += "add.ftz.f32 d,e,0f3f800000; rcp.approx.ftz.f32 r,d; "
    code += "mul.ftz.f32 $0,t,r; }"
    return code, "=f,f,f,f", (gate, beta, inv_beta)


@asm(Float32, pure=True)
def situ_up(up, linear_beta, inverse):
    code = "{ .reg .f32 x; mul.ftz.f32 x,$1,$3; "
    code += "tanh.approx.f32 x,x; mul.ftz.f32 $0,$2,x; }"
    return code, "=f,f,f,f", (up, linear_beta, inverse)


@asm(Float32, pure=True)
def situ_product(gate, up):
    return "mul.ftz.f32 $0,$1,$2;", "=f,f,f", (gate, up)


@cute.jit
def situ(gate, up, beta, linear_beta):
    """Stock CUDA SiTU: FP32 fast tanh/exp/rcp, then one output conversion."""
    g = situ_gate(gate, beta, div_rn(c.Float32(1), beta))
    if linear_beta > 0:
        up = situ_up(up, linear_beta, div_rn(c.Float32(1), linear_beta))
    return situ_product(g, up)


@cute.jit
def situ_hoisted(gate, up, beta, linear_beta, inv_beta, inv_lb):
    """``situ`` with ``inv_beta`` = div_rn(1, beta) and ``inv_lb`` = div_rn(1, linear_beta) computed once by the
    caller (the same IEEE division of the same operands: bitwise equal to ``situ``)."""
    g = situ_gate(gate, beta, inv_beta)
    if linear_beta > 0:
        up = situ_up(up, linear_beta, inv_lb)
    return situ_product(g, up)


@asm()
def prefetch_tma2(descriptor, column, row):
    """L2 prefetch of the 2D TMA box at (``column``, ``row``)."""
    code = "cp.async.bulk.prefetch.tensor.2d.L2.global.tile [$0,{$1,$2}];"
    return code, "l,r,r", (descriptor, Int32(column), Int32(row))


@asm()
def release_count(address):
    return "red.release.gpu.global.add.u32 [$0],1;", "l,~{memory}", (address,)


@asm(Uint32)
def acquire_count(address):
    return "ld.acquire.gpu.global.u32 $0,[$1];", "=r,l,~{memory}", (address,)


@asm(Uint32)
def fetch_count(address):
    return "atom.relaxed.gpu.global.add.u32 $0,[$1],1;", "=r,l,~{memory}", (address,)


@cute.jit
def wait_count(address, target):
    seen = Int32(acquire_count(address))
    while seen < target:
        seen = Int32(acquire_count(address))


def weight_maps(gate_up, down, front=None, down_rows=32):
    """TMA maps (W [64, 32] boxes, 128B swizzle) of gate/up, down, then (optional) the
    MoE front GEMM's packed router / latent rows. ``down_rows=64``: the down map takes
    [64, 64] boxes (``SharedDown`` with ``one64`` only)."""
    dtype = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
    swizzle = cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
    l2 = cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B
    weights = (gate_up, down) if front is None else (gate_up, down, front)
    specs = [
        (w.data_ptr(), dtype, (w.shape[1], w.shape[0]), (2 * w.stride(0),))
        + ((64, down_rows if i == 1 else 32), swizzle, l2)
        for i, w in enumerate(weights)
    ]
    return tensor_maps(specs, gate_up.device)


class SharedBase:
    """What the fused kernel and the split down kernel share: geometry, TMA weight and
    cp.async row tiles, the M=64 accumulator readout and the RS push of 64 rows."""

    def __init__(self, tp, batch):
        self.tp, self.batch, self.hidden = tp, batch, 6144 // tp
        self.down = self.hidden // 64  # down K blocks
        self.n = (batch + 7) // 8 * 8
        self.columns = max(32, 1 << (self.n - 1).bit_length())  # one accumulator

    @cute.jit
    def load_weight(self, tile, descriptor, barrier, row0, row1, column):
        """W rows [row0, + 32) and [row1, + 32), 64 columns from ``column``, as a
        K-major 128B-swizzled [64, 64] tile (two TMA boxes)."""
        cute.arch.mbarrier_arrive_and_expect_tx(barrier, 8192)
        load_tma2(tile, descriptor, barrier, column, row0)
        load_tma2(tile + 4096, descriptor, barrier, column, row1)

    @cute.jit
    def load_rows(self, tile, source, column):
        """Rows [0, batch) of ``source``, 64 columns from ``column``, as the B tile
        (threads [0, 128))."""
        tid, _, _, _ = thread()
        count: Constexpr = self.batch * 8
        for i in c.range_constexpr((count + 127) // 128):
            item = tid + 128 * i
            row, chunk = item // 8, item % 8
            dst = tile + row // 8 * 1024 + row % 8 * 128 + (chunk ^ row % 8) * 16
            if c.const_expr((i + 1) * 128 <= count):
                copy_g2s(dst, at(source, row, column + chunk * 8))
            else:
                if item < count:  # noqa: SIM102
                    copy_g2s(dst, at(source, row, column + chunk * 8))

    @cute.jit
    def accumulated(self, tmem):
        """This lane's accumulator row (M=64: rows 16 warp + lane, lane < 16); later
        MMAs may then reuse the columns."""
        _, warp, _, _ = thread()
        values = load_tmem(tmem + Uint32(32 * warp << 16), self.columns)
        fence_tmem(before=True)
        return values

    @cute.jit
    def elected(self, barrier):
        with cute.arch.elect_one():
            commit_asm(barrier)

    @cute.jit
    def push(self, mailbox, staging, values, row0):
        """Down rows [row0, + 64) (``values`` from ``accumulated``) -> FP32 [token,
        row] tile at ``staging`` -> BF16 (-0 made +0) 16-byte peer stores into the RS
        mailbox slot; ``mailbox`` is (peers, parity, rank, capacity). Warps 0-3."""
        peers, parity, rank, capacity = mailbox
        tid, warp, lane, _ = thread()
        tile = view(staging, 0, Float32, (self.n, 68), (68, 1))
        if lane < 16:
            for t in c.range_constexpr(self.batch):
                tile[t, 16 * warp + lane] = values[t]
        named_barrier(1, 128)
        shard: Constexpr = 7168 // self.tp
        peer, column = row0 // shard, row0 % shard
        count: Constexpr = self.batch * 8
        for i in c.range_constexpr((count + 127) // 128):
            item = tid + 128 * i
            if item < count:
                t, part = item // 8, item % 8
                reduced = []
                for half in c.range_constexpr(2):
                    words = smem_words(at(tile, t, part * 8 + half * 4))
                    reduced += [bits_float(w) for w in words]
                packed = tuple(
                    pack_pair(reduced[2 * v], reduced[2 * v + 1], policy="latent")
                    for v in range(4)
                )
                row = (parity * self.tp + rank) * capacity + t
                offset = Int64(row * shard + column + part * 8)
                store_relaxed(peers[peer] + offset * 2, packed)
        named_barrier(1, 128)


class SharedUmma(SharedBase):
    def __init__(self, tp, batch, budget=80 * 1024, gate=False, front=None):
        super().__init__(tp, batch)
        # gate: the split variant's front-window half (gate/up and SiTU into h; the
        # down rows run in ``SharedDown`` beside the routed experts). front: (rows,
        # split, valid, mailbox row) of the MoE front GEMM, whose packed router /
        # latent rows join as extra M-tiles (gate mode, on the main stream after the
        # o-proj consumer: the first ring's weights load before the dependency wait).
        self.gate, self.front = (gate or front is not None, front)
        self.tiles = 2 * self.hidden // 64  # gate/up M-tiles, one per cluster
        self.grid = GRID
        if front is not None:
            self.gtiles = self.tiles
            self.tiles += -(-front[0] // 64)
            self.grid = FRONT_KS[tp] * self.tiles
        self.ks = self.grid // self.tiles  # cluster size: the K split
        self.blocks = -(-112 // self.ks)  # gate/up K blocks per CTA, at most
        self.stage = 8192 + self.n * 128
        later = 0 if self.gate else self.down
        self.stages = min(self.blocks + later, budget // self.stage)
        # Owners 0 .. 32 / width - 1 take a tile's h columns ``width`` at a time.
        self.width = 8 if self.ks < 8 else 4
        self.owners = 32 // self.width
        self.received = self.ks * 2 * self.width * self.n * 4  # an owner's bytes

    @cute.jit
    def __call__(self, *args):
        options = dict(pdl=self.front is not None, min_blocks_per_mp=1)
        launch(self.kernel, args, self.grid, 160, self.ks, **options)

    @cute.kernel
    def kernel(self, args):
        maps = args[1]
        tid, warp, lane, cta = thread()
        rank = cta % self.ks
        stages, stage = self.stages, self.stage
        smem = c.utils.SmemAllocator()
        ring = alloc(smem, c.Uint8, stages * stage, 1024).iterator
        received = alloc(smem, Float32, self.received // 4, 16).iterator
        staged = alloc(smem, c.BFloat16, self.n * self.width, 16)  # owned h rows
        if c.const_expr(self.front is not None):  # owned front rows [token, half]
            staged = (staged, alloc(smem, Float32, self.n * 2 * (self.width + 1), 16))
        bars = alloc(smem, c.Uint64, 2 * stages + 3, 8).iterator
        slot = alloc(smem, Uint32, 1)
        if tid == 0:
            for s in c.range_constexpr(stages):
                cute.arch.mbarrier_init(bars + s, 129)  # TMA + 128 cp.async arrivals
                cute.arch.mbarrier_init(bars + stages + s, 1)
            for i in c.range_constexpr(3):
                cute.arch.mbarrier_init(bars + 2 * stages + i, 1)
            cute.arch.mbarrier_init_fence()
            # Every rank's gate and up partials of the owned columns.
            nbytes = self.received if rank < self.owners else 0
            cute.arch.mbarrier_arrive_and_expect_tx(bars + 2 * stages + 2, nbytes)
            acquire_tensormap(maps.iterator)
            acquire_tensormap(maps.iterator + 128)
            if c.const_expr(self.front is not None):
                acquire_tensormap(maps.iterator + 256)
        if warp == 4:
            cute.arch.alloc_tmem(self.columns, slot.iterator)
            cute.arch.relinquish_tmem_alloc_permit()
        fence_tmem(before=True)
        cute.arch.sync_threads()
        fence_tmem()
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()  # every owner's receive barrier is armed
        tmem = slot[0]
        if c.const_expr(self.front is not None):
            # Weights do not depend on the predecessor: the first ring's worth.
            if tid == 0:
                m, k0, q = self.span(cta)
                for kb in c.range_constexpr(stages):
                    if kb < q:
                        self.load_tile(ring + kb * stage, maps, bars + kb, m, k0 + kb)
                # The rest of the front tiles' boxes (on the routing path) into L2.
                if m >= self.gtiles:
                    for kb in c.range(stages, q, unroll=1):
                        self.prefetch_tile(maps, m, k0 + kb)
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        if warp < 4:
            self.stream(args, ring, (received, staged), bars, tmem, cta)
        else:
            self.multiply(ring, bars, tmem, cta)
        release_tmem(tmem, self.columns, warp, 4)
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()

    @cute.jit
    def load_tile(self, tile, maps, barrier, m, k):
        """K block ``k`` of M-tile ``m``: 32 gate and the matching 32 up rows, or (front
        tiles past the gate/up ones) 64 consecutive front rows."""
        if m < self.gtiles:
            rows = (32 * m, self.hidden + 32 * m)
            self.load_weight(tile, maps.iterator, barrier, *rows, 64 * k)
        else:
            row = 64 * (m - self.gtiles)
            self.load_weight(tile, maps.iterator + 256, barrier, row, row + 32, 64 * k)

    @cute.jit
    def prefetch_tile(self, maps, m, k):
        """L2 prefetch of ``load_tile``'s two boxes."""
        if m < self.gtiles:
            prefetch_tma2(maps.iterator, 64 * k, 32 * m)
            prefetch_tma2(maps.iterator, 64 * k, self.hidden + 32 * m)
        else:
            row = 64 * (m - self.gtiles)
            prefetch_tma2(maps.iterator + 256, 64 * k, row)
            prefetch_tma2(maps.iterator + 256, 64 * k, row + 32)

    @cute.jit
    def span(self, cta):
        """(tile, first K block, K blocks) of the CTA's gate/up work."""
        k0, count = split(112, cta % self.ks, self.ks)
        return cta // self.ks, k0, count

    @cute.jit
    def multiply(self, ring, bars, tmem, cta):
        """Warp 4: every K block's four K16 MMAs and its stage release; gate/up
        completes barrier 0 of three, down (the same columns) barrier 1."""
        stages, q = self.stages, self.span(cta)[2]
        total = q
        if c.const_expr(not self.gate):
            total = q + (self.down if cta < ROWS else 0)
        instruction: Constexpr = 1 << 4 | 1 << 7 | 1 << 10 | self.n >> 3 << 17 | 4 << 24
        done = bars + 2 * stages
        for kb in c.range(total, unroll=1):
            s = kb % stages
            wait(bars + s, kb // stages % 2, scope="cta")
            cute.arch.fence_proxy("async.shared", space="cta")
            fence_tmem()
            tile = ring + s * self.stage
            a, b = smem_desc(tile, True), smem_desc(tile + 8192, True)
            index = kb - q  # block index within the product
            if kb < q:
                index = kb
            release = Int32(kb + stages < total)
            mma_bf16_block(tmem, a, b, instruction, index, bars + stages + s, release)
            if kb == q - 1:
                self.elected(done)
        if c.const_expr(not self.gate):  # noqa: SIM102
            if cta < ROWS:
                self.elected(done + 1)

    @cute.jit
    def stream(self, args, ring, buffers, bars, tmem, cta):
        """Loader warps: gate/up blocks, the first down weights, the partial push,
        owned h columns, the h barrier, then the down blocks' h (and remaining
        weights) and the push."""
        x, maps, h, counters = args[0:4]
        stages, stage, hidden = self.stages, self.stage, self.hidden
        tid, _, _, _ = thread()
        m, k0, q = self.span(cta)
        gate_up, down = maps.iterator, maps.iterator + 128
        for kb in c.range(q, unroll=1):
            s = kb % stages
            tile = ring + s * stage
            if kb >= stages:
                wait(bars + stages + s, (kb // stages - 1) % 2, scope="cta")
            k = k0 + kb
            if c.const_expr(self.front is not None):
                if (tid == 0) & (kb >= stages):  # the first ring went before the wait
                    self.load_tile(tile, maps, bars + s, m, k)
            elif tid == 0:
                rows = (32 * m, hidden + 32 * m)
                self.load_weight(tile, gate_up, bars + s, *rows, 64 * k)
            self.load_rows(tile + 8192, x, 64 * k)
            cute.arch.cp_async_mbarrier_arrive_noinc(bars + s)
        received, staged = buffers
        if c.const_expr(self.gate):
            self.partial(received, bars, tmem, cta, ring)
            self.reduce(args, received, staged, bars, cta)
            return
        early = min(self.down, stages)
        row0 = 64 * cta
        if cta < ROWS:
            for j in c.range(early, unroll=1):
                s = (q + j) % stages
                if q + j >= stages:
                    wait(bars + stages + s, ((q + j) // stages - 1) % 2, scope="cta")
                if tid == 0:
                    tile = ring + s * stage
                    self.load_weight(tile, down, bars + s, row0, row0 + 32, 64 * j)
        self.partial(received, bars, tmem, cta)
        self.reduce(args, received, staged, bars, cta)
        if tid == 0:
            wait_count(counters.iterator, GRID)
            if fetch_count(counters.iterator + 1) == GRID - 1:
                counters[0], counters[1] = Int32(0), Int32(0)
        named_barrier(1, 128)
        if cta < ROWS:
            for j in c.range(self.down, unroll=1):
                s = (q + j) % stages
                tile = ring + s * stage
                if j >= early:
                    wait(bars + stages + s, ((q + j) // stages - 1) % 2, scope="cta")
                    if tid == 0:
                        self.load_weight(tile, down, bars + s, row0, row0 + 32, 64 * j)
                self.load_rows(tile + 8192, h, 64 * j)
                cute.arch.cp_async_mbarrier_arrive_noinc(bars + s)
            wait(bars + 2 * self.stages + 1, 0, scope="cta")
            fence_tmem()
            self.push(args[4:8], ring, self.accumulated(tmem), row0)

    @cute.jit
    def partial(self, received, bars, tmem, cta, ring=None):
        """Push the CTA's FP32 gate/up rows to their h columns' owners: row r (gate
        r, or up r - 32) of column g = r % 32 lands in owner g // width's slot
        [rank, r // 32, g % width]. With the front rows, the slots are staged in the
        drained ring and each owner's leaves in one DSMEM bulk copy."""
        tid, warp, lane, _ = thread()
        rank = cta % self.ks
        wait(bars + 2 * self.stages, 0, scope="cta")
        fence_tmem()
        values = self.accumulated(tmem)
        if c.const_expr(self.front is not None):
            share: Constexpr = 2 * self.width * self.n  # one owner's floats
            if lane < 16:
                row = 16 * warp + lane
                g = row % 32
                owner = g // self.width
                cell = ((owner * 2 + row // 32) * self.width + g % self.width) * self.n
                for v in c.range_constexpr(self.n // 4):
                    words = tuple(float_bits(values[4 * v + i]) for i in range(4))
                    at_v = cell + self.slot(4 * v, g % self.width)
                    store_words(ring + at_v * 4, words, shared=True)
            cute.arch.fence_proxy("async.shared", space="cta")
            named_barrier(1, 128)
            if tid < self.owners:
                arrived = bars + 2 * self.stages + 2
                source = ring + tid * share * 4
                dst = received + rank * share
                copy_s2cluster(dst, source, arrived, tid, size=share * 4)
            return
        if lane < 16:
            row = 16 * warp + lane
            g = row % 32
            cell = ((rank * 2 + row // 32) * self.width + g % self.width) * self.n
            arrived = bars + 2 * self.stages + 2  # the owner's receive barrier
            for v in c.range_constexpr(self.n // 4):
                chunk = tuple(values[4 * v + i] for i in range(4))
                push_float(received + cell + 4 * v, chunk, arrived, g // self.width)

    @cute.jit
    def slot(self, t, j):
        """Token ``t``'s place in a partial row of column ``j``: with the front rows
        the row's 16-byte chunks rotate by ``j`` (conflict-free staging stores)."""
        if c.const_expr(self.front is None):
            return t
        chunks: Constexpr = self.n // 4
        return (t // 4 + j) % chunks * 4 + t % 4

    @cute.jit
    def reduce(self, args, received, staged, bars, cta):
        """Owner: a thread per (column, token) sums the parts in rank order and
        applies SiTU; the rows leave in ``width``-column vectors. Then the CTA counts
        into the h barrier (fused only)."""
        h, counters = args[2:4]
        beta, linear_beta = args[8:10]
        tid, _, _, _ = thread()
        m, rank = cta // self.ks, cta % self.ks
        wait(bars + 2 * self.stages + 2, 0)  # peers' DSMEM stores
        parts = view(received, 0, Float32, self.received // 4)
        if c.const_expr(self.front is None):
            if rank < self.owners:
                self.owned_h(h, parts, staged, m, rank, beta, linear_beta)
        else:
            staged, rows = staged
            if rank < self.owners:
                if m < self.gtiles:
                    self.owned_h(h, parts, staged, m, rank, beta, linear_beta)
                else:
                    self.publish(args, parts, rows, m - self.gtiles, rank)
        named_barrier(1, 128)
        if c.const_expr(not self.gate):  # noqa: SIM102
            if tid == 0:
                cute.arch.fence_acq_rel_gpu()
                release_count(counters.iterator)

    @cute.jit
    def owned_h(self, h, parts, staged, m, rank, beta, linear_beta):
        """Owner of gate/up tile ``m``: SiTU of the summed parts into ``width``-column
        h vectors."""
        tid, _, _, _ = thread()
        batch, width = self.batch, self.width
        for i in c.range_constexpr((width * batch + 127) // 128):
            item = tid + 128 * i
            if item < width * batch:
                j, t = item // batch, item % batch
                sums = [Float32(0), Float32(0)]
                for source in c.range_constexpr(self.ks):
                    for half in c.range_constexpr(2):
                        cell = (source * 2 + half) * width + j
                        sums[half] = sums[half] + parts[cell * self.n + self.slot(t, j)]
                gate = Float32(c.BFloat16(sums[0]))
                up = Float32(c.BFloat16(sums[1]))
                value = c.BFloat16(situ(gate, up, beta, linear_beta))
                staged[t * width + j] = value
        named_barrier(1, 128)
        if tid < batch:
            row = staged.iterator + tid * width
            words = smem_words(row) if width == 8 else smem_words(row, count=2)
            store_words(at(h, tid, 32 * m + width * rank), words)

    @cute.jit
    def publish(self, args, parts, rows, f, rank):
        """Owner of front tile ``f``: sums each (row, token) in rank order, then
        publishes ``width``-row vectors like the front GEMM: FP32 router logits (rows
        < valid) and FP32 latent columns (rows >= split; every TP, as stock's FP32
        front, A4), -0 made +0, as multicast stores into the front mailbox (``mc``;
        this rank's ``slot`` / ``base``)."""
        mc, slot, base = args[10:13]
        _, split, valid, pitch = self.front
        tid, _, _, _ = thread()
        batch, width = self.batch, self.width
        shard: Constexpr = 3584 // self.tp
        for i in c.range_constexpr((width * batch + 127) // 128):
            item = tid + 128 * i
            if item < width * batch:
                j, t = item // batch, item % batch
                for half in c.range_constexpr(2):
                    total = Float32(0)
                    for source in c.range_constexpr(self.ks):
                        cell = (source * 2 + half) * width + j
                        total = total + parts[cell * self.n + self.slot(t, j)]
                    rows[(t * 2 + half) * (width + 1) + j] = total
        named_barrier(1, 128)
        if tid < 2 * batch:
            t, half = tid // 2, tid % 2
            row = 64 * f + 32 * half + width * rank
            values = [rows[tid * (width + 1) + v] for v in range(width)]
            if row < valid:
                for v in c.range_constexpr(width // 4):
                    words = tuple(positive_zero32(values[4 * v + i]) for i in range(4))
                    offset = t * pitch + 7168 + 2 * (slot + row + 4 * v)
                    multicast_words(mc + Int64(offset) * 2, words, kind="f32")
            elif (row >= split) & (row - split < shard):
                for v in c.range_constexpr(width // 4):
                    words = tuple(positive_zero32(values[4 * v + i]) for i in range(4))
                    offset = t * pitch + 2 * (base + row - split + 4 * v)
                    multicast_words(mc + Int64(offset) * 2, words, kind="f32")


class SharedDown(SharedBase):
    """The split variant's down projection beside the routed experts: ``ctas``
    CTAs stage h [N, hidden] once, take every ``ctas``-th 64-row tile,
    stream its W blocks through a TMA ring (warp 5)
    into alternating TMEM accumulators (warp 4) and peer-store each tile like the
    fused kernel (warps 0-3). Exactly the fused kernel's down numerics.

    ``one64`` (``K31MK_MOE_SHARED_DOWN_ONE64``, default off): each 8 KB W stage is ONE
    [64, 64] TMA box (``weight_maps(..., down_rows=64)``) instead of two [64, 32] boxes:
    the same 128B-swizzled SMEM bytes with half the requests (bitwise identical)."""

    def __init__(self, tp, batch, ctas, budget=200 * 1024, one64=False):
        super().__init__(tp, batch)
        self.one64 = one64
        self.ctas, self.image = ctas, self.down * self.n * 128  # h as B tiles
        ring = budget - self.image - self.n * 68 * 4
        self.stages = min(24, ring // 8192)  # W [64, 64] K blocks (8 KB)

    @cute.jit
    def __call__(self, *args):
        options = dict(pdl=False, min_blocks_per_mp=1)
        launch(self.kernel, args, self.ctas, 192, **options)

    @cute.kernel
    def kernel(self, args):
        h, maps = args[0:2]
        tid, warp, lane, cta = thread()
        stages, columns = self.stages, self.columns
        smem = c.utils.SmemAllocator()
        image = alloc(smem, c.Uint8, self.image, 1024).iterator
        ring = alloc(smem, c.Uint8, stages * 8192, 1024).iterator
        staging = alloc(smem, c.Uint8, self.n * 68 * 4, 16).iterator
        # full [stages], empty [stages], accumulator full [2] / empty [2], h ready
        bars = alloc(smem, c.Uint64, 2 * stages + 5, 8).iterator
        slot = alloc(smem, Uint32, 1)
        acc_full, acc_empty, ready = (bars + 2 * stages + i for i in (0, 2, 4))
        if tid == 0:
            for s in c.range_constexpr(stages):
                cute.arch.mbarrier_init(bars + s, 1)
                cute.arch.mbarrier_init(bars + stages + s, 1)
            for i in c.range_constexpr(2):
                cute.arch.mbarrier_init(acc_full + i, 1)
                cute.arch.mbarrier_init(acc_empty + i, 128)
            cute.arch.mbarrier_init(ready, 128)
            cute.arch.mbarrier_init_fence()
        if warp == 4:
            cute.arch.alloc_tmem(2 * columns, slot.iterator)
            cute.arch.relinquish_tmem_alloc_permit()
        fence_tmem(before=True)
        cute.arch.sync_threads()
        fence_tmem()
        tmem = slot[0]
        tiles = (ROWS - cta + self.ctas - 1) // self.ctas
        if warp < 4:
            for j in c.range_constexpr(self.down):
                self.load_rows(image + j * self.n * 128, h, 64 * j)
            cute.arch.cp_async_mbarrier_arrive_noinc(ready)
            for i in c.range(tiles, unroll=1):
                wait(acc_full + i % 2, i // 2 % 2, scope="cta")
                fence_tmem()
                values = self.accumulated(tmem + Uint32(i % 2 * columns))
                cute.arch.mbarrier_arrive(acc_empty + i % 2)
                self.push(args[2:6], staging, values, 64 * (cta + i * self.ctas))
        elif warp == 4:
            self.multiply(image, ring, bars, tmem, tiles)
        elif lane == 0:
            down = maps.iterator + 128
            acquire_tensormap(down)  # in the thread that issues its TMAs
            for i in c.range(tiles, unroll=1):
                row0 = 64 * (cta + i * self.ctas)
                for j in c.range(self.down, unroll=1):
                    g = i * self.down + j
                    s = g % stages
                    if g >= stages:
                        wait(bars + stages + s, (g // stages - 1) % 2, scope="cta")
                    if c.const_expr(self.one64):  # one [64, 64] box, same SMEM image
                        cute.arch.mbarrier_arrive_and_expect_tx(bars + s, 8192)
                        load_tma2(ring + s * 8192, down, bars + s, 64 * j, row0)
                    else:
                        self.load_weight(
                            ring + s * 8192, down, bars + s, row0, row0 + 32, 64 * j
                        )
        release_tmem(tmem, 2 * columns, warp, 4)

    @cute.jit
    def multiply(self, image, ring, bars, tmem, tiles):
        """Warp 4: each tile's K blocks into accumulator half i % 2."""
        stages, columns = self.stages, self.columns
        instruction: Constexpr = 1 << 4 | 1 << 7 | 1 << 10 | self.n >> 3 << 17 | 4 << 24
        acc_full, acc_empty, ready = (bars + 2 * stages + i for i in (0, 2, 4))
        total = tiles * self.down
        wait(ready, 0, scope="cta")
        cute.arch.fence_proxy("async.shared", space="cta")
        for i in c.range(tiles, unroll=1):
            if i >= 2:
                wait(acc_empty + i % 2, (i // 2 - 1) % 2, scope="cta")
            fence_tmem()
            d = tmem + Uint32(i % 2 * columns)
            for j in c.range(self.down, unroll=1):
                g = i * self.down + j
                s = g % stages
                wait(bars + s, g // stages % 2, scope="cta")
                fence_tmem()
                a = smem_desc(ring + s * 8192, True)
                b = smem_desc(image + j * self.n * 128, True)
                release = Int32(g + stages < total)
                mma_bf16_block(d, a, b, instruction, j, bars + stages + s, release)
            self.elected(acc_full + i % 2)


@functools.cache
def compile_shared(tp, batch, device, gate=False, front=None):
    """Arguments: x, ``weight_maps``, h, counters, RS peers, parity, rank, capacity,
    beta, linear beta. ``gate``: the split variant's gate/up + SiTU half; ``front``
    (rows, split, valid, mailbox row): with the MoE front GEMM's rows, whose results
    go to the front mailbox (more arguments: multicast address, logit slot, latent
    base; ``weight_maps`` with the front's)."""
    maps = fake((256 if front is None else 384,), None, c.Uint8, 64)
    args: tuple = (fake((batch, 7168)), maps, fake((batch, 6144 // tp)))
    args += (fake((COUNTERS,), None, Int32), fake((tp,), None, Int64))
    args += (Int32(0), Int32(0), Int32(0), Float32(4), Float32(25))
    if front is not None:
        args += (Int64(0), Int32(0), Int32(0))
    ring = 80 * 1024 if front is None else FRONT_RING[tp]
    kernel = SharedUmma(tp, batch, ring, gate, front)
    with torch.accelerator.device_index(device):
        return cute.compile(kernel, *args, make_fake_stream())


@functools.cache
def compile_down(tp, batch, device, sms, *, ctas=None, budget=None, one64=False):
    """Arguments: h, ``weight_maps``, RS peers, parity, rank, capacity.

    ``sms`` is the caller's reservation beside the routed experts; ``ctas`` and
    ``budget`` independently select the down grid and per-CTA shared storage.
    ``one64``: one [64, 64] box per W stage (needs ``weight_maps(..., down_rows=64)``).
    """
    args: tuple = (fake((batch, 6144 // tp)), fake((256,), None, c.Uint8, 64))
    args += (fake((tp,), None, Int64), Int32(0), Int32(0), Int32(0))
    # Each output tile can leave the queue as soon as an expert SM becomes free.
    # A few persistent CTAs cannot expand after the routed grid has drained.
    wide = tp == 8 and 13 <= batch <= 24
    if ctas is None:
        ctas = ROWS if wide else sms
    if budget is None:
        budget = ((64 if batch <= 16 else 96) if wide else 200) * 1024
    kernel = SharedDown(tp, batch, ctas, budget, one64)
    with torch.accelerator.device_index(device):
        return cute.compile(kernel, *args, make_fake_stream())


def shared_down_one64_enabled():
    """``K31MK_MOE_SHARED_DOWN_ONE64=1`` (default off): the split ``SharedDown`` loads each
    8 KB W stage as one [64, 64] TMA box instead of two [64, 32] boxes (same SMEM bytes, bitwise identical output;
    measured faster than the served kernel at 16-64 rows, isolated)."""
    import os

    return os.environ.get("K31MK_MOE_SHARED_DOWN_ONE64", "0") == "1"


def scratch(device):
    """Model-owned fused shared-expert counters (zeroed; re-armed by each launch)."""
    return torch.zeros(COUNTERS, dtype=torch.int32, device=device)


# ------------------------------------------------------------------------------ SiTU in the gate/up split-K
def shared_sks_enabled():
    """``K31MK_EXP_SHARED_SKS=1`` (default off): B<=8 verify batches run the shared expert's gate/up as
    ``SplitKSituGemm`` (the shipped split-K GEMM and tactic, SiTU in the owner rank's epilogue, h BF16 out) instead of
    the gate/up GEMM (FP32 out) followed by RouteSitu's situ role. Same FP32 sums in the same order, same SiTU."""
    import os

    return os.environ.get("K31MK_EXP_SHARED_SKS", "0") == "1"


def shared_sks_k23_arm():
    """K-23h (always on): SKS's owner arms bar_reduce at .relaxed.cluster."""
    return True


def interleave_gate_up(gate_up, tile=64):
    """Load-time row relayout (lossless; ``deinterleave_gate_up`` inverts it): [gate (H rows); up (H rows)] ->
    tile-interleaved [gate rows 32m..32m+31; up rows 32m..32m+31] per 64-row tile m, so one split-K M tile holds the
    gate and up rows of the same 32 intermediate columns."""
    rows = gate_up.shape[0]
    half, pair = rows // 2, tile // 2
    if rows % tile or half % pair:
        raise ValueError(f"interleave_gate_up: {rows} rows do not split into {tile}-row tiles")
    order = torch.arange(rows, device=gate_up.device).view(2, half // pair, pair).transpose(0, 1).reshape(-1)
    return gate_up.index_select(0, order).contiguous(), order


def deinterleave_gate_up(interleaved, order):
    out = torch.empty_like(interleaved)
    out[order] = interleaved
    return out


def _sks_classes():
    """Built on first use: the vendored split-K engine imports FlashInfer names."""
    import cutlass
    import cutlass.utils as utils
    import cutlass.utils.blackwell_helpers as sm100_utils
    from cutlass.cute import experimental as cute_ext

    from cutlass.cutlass_dsl import dsl_user_op

    from ..gemm import splitk as SK

    @dsl_user_op
    def arm_cluster(mbar, tx_bytes, *, loc=None, ip=None):
        """K-23h fix (K-23h / K-23b): the owner arms bar_reduce after the cluster barrier, so
        peers' remote ``st.async`` complete_tx can land first; armed at CLUSTER scope the arm and those complete_tx are
        morally strong (PTX 8.7) and the owner's ``try_wait.acquire.cluster`` orders the peers' partials whichever order
        they take (litmus-tested: forbidden with the .relaxed.cluster arm). Same instruction as SplitKGemm's K-23b arm."""
        code = "{ .reg .b64 st; mbarrier.arrive.expect_tx.relaxed.cluster.shared::cta.b64 st, [$0], $1; }"
        SK._asm(code, "r,r,~{memory}", [mbar.toint().ir_value(), Int32(tx_bytes).ir_value()], loc, ip)

    class SplitKSituGemm(SK.SplitKGemm):
        """``SplitKGemm`` whose owner rank, after the in-order peer reduction, stages its FP32 sums in SMEM, pairs
        tile rows j (gate) and 32 + j (up) of the interleaved weight, applies ``situ`` and stores h BF16 [B, H] at
        column 32 * (M tile) + j. Requires mma_m == 64 and the ``interleave_gate_up`` weight."""

        def __init__(self, *, tactic, epiwait=True, k23_arm=True):
            super().__init__(tactic=tactic)
            assert self.cta_m == 64, self.cta_m
            # Every epilogue thread executes griddepcontrol.wait before it
            # stores h, so its h writes are ordered after this grid's prerequisite grids (PTX 9.7.15.14 .wait applies to
            # the executing thread); before, only the activation warp waited and the h writers were linked to it only
            # through tcgen05.commit. epiwait=False is the old form (not selectable).
            self.epiwait = epiwait
            self.k23_arm = k23_arm  # K-23h: the owner's late bar_reduce arm at cluster scope (shared_sks_k23_arm)

        @cute.experimental.jit
        def __call__(self, a, b, c, h, a_map, beta, linear_beta, stream):
            self.kernel(a, b, c, h, a_map, beta, linear_beta).launch(
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
        def kernel(self, mA, mB, mC, mH, a_map, beta, linear_beta):
            """As ``SplitKGemm.kernel``; ``mC`` only shapes the grid and tiles (never written), ``mH`` (H, N, L)
            receives h."""
            stages = self.num_ab_stage
            ab_dtype = mA.element_type
            majors = (utils.LayoutEnum.from_tensor(t).mma_major_mode() for t in (mA, mB))
            tiled_mma = sm100_utils.make_trivial_tiled_mma(
                ab_dtype, ab_dtype, *majors, self.acc_dtype, self.cta_group, self.mma_tiler_mn,
            )
            mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
            block_idx = cute.arch.block_idx()
            bidx = block_idx[0]
            split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            n_idx = block_idx[1] // self.split_k
            l_idx = block_idx[2]
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            shape = (tiled_mma, mnk_tiler, ab_dtype, stages)
            sA = SK._smem(ab_dtype, None, 1024, sm100_utils.make_smem_layout_a(*shape))
            sB = SK._smem(ab_dtype, None, 1024, sm100_utils.make_smem_layout_b(*shape))
            acc_layout = cute_ext.make_tmem_layout_acc(tiled_mma, self.mma_tiler_mn, acc_stage=1)
            bar_full = SK._smem(cutlass.Int64, stages, 8).iterator
            bar_empty = SK._smem(cutlass.Int64, stages, 8).iterator
            bar_tma_epilog = SK._smem(cutlass.Int64, 1, 8).iterator
            bar_mma_epilog = SK._smem(cutlass.Int64, 1, 8).iterator
            bar_tmem_alloc = SK._smem(cutlass.Int64, 1, 8).iterator
            tmem_base_ptr = SK._smem(cutlass.Int32, 1, 4).iterator
            staging = SK._smem(cutlass.Float32, self.cta_m * self.cta_n, 16)
            mailbox, bar_reduce = sA, bar_mma_epilog
            if cutlass.const_expr(self.split_k > 1):
                mailbox = SK._smem(cutlass.Float32, self.mailbox_elements, 128)
                bar_reduce = SK._smem(cutlass.Int64, 1, 8).iterator
            if warp_idx == 0:
                with cute.arch.elect_one():
                    for i in range(stages):
                        cute.arch.mbarrier_init(bar_full + i, 2)
                        cute.arch.mbarrier_init(bar_empty + i, 1)
                    cute.arch.mbarrier_init(bar_tma_epilog, 32)
                    cute.arch.mbarrier_init(bar_mma_epilog, 1)
                    cute.arch.mbarrier_init(bar_tmem_alloc, 160)
                    if cutlass.const_expr(self.split_k > 1):
                        cute.arch.mbarrier_init(bar_reduce, 1)
            cute.arch.mbarrier_init_fence()
            if cutlass.const_expr(self.split_k > 1):
                cute.arch.cluster_arrive_relaxed()
            else:
                cute.arch.barrier()
            k_tile_count = cute.size(mA, mode=[1]) // self.cta_k // self.split_k
            k_tile_start = split_rank * k_tile_count
            if cutlass.const_expr(self.split_k > 1):
                cute.arch.cluster_wait()
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
                )
            elif warp_idx >= 4:
                tile = cute.local_tile(mC, (self.cta_m, self.cta_n), (bidx, n_idx, l_idx))
                self.situ_epilog(
                    bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr, acc_layout, tile,
                    cute.arch.thread_idx()[0] - 128, mC.element_type, utils.LayoutEnum.from_tensor(mC),
                    mailbox, bar_reduce, split_rank, staging, mH, bidx, n_idx, l_idx, beta, linear_beta,
                )

        @cute.experimental.jit
        def situ_epilog(
            self, bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr, acc_layout, gD_tile, epi_tid, c_dtype, d_layout,
            mailbox, bar_reduce, split_rank, staging, mH, bidx, n_idx, l_idx, beta, linear_beta,
        ):
            if cutlass.const_expr(self.epiwait):
                cute.arch.griddepcontrol_wait()
            cute.arch.mbarrier_arrive(bar_tmem_alloc)
            cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
            SK._fence_tcgen05()
            acc_view = cute.make_tensor(
                cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr), acc_layout,
            )[((None, None), 0, 0, 0)]
            epi_tile = (self.cta_m, self.cta_n)
            load_op = sm100_utils.get_tmem_load_op(
                (self.cta_m, self.cta_n, self.cta_k), d_layout, c_dtype, self.acc_dtype, epi_tile, False,
            )
            tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(load_op, acc_view)
            gD_epi = cute.flat_divide(gD_tile, epi_tile)
            rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, gD_epi, epi_tid)
            rAcc = cute_ext.allocate(self.acc_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32)
            thr_t2r = tiled_copy_t2r.get_slice(epi_tid)
            cute.arch.mbarrier_wait(bar_mma_epilog, 0)
            SK._fence_tcgen05()
            cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
            cute.arch.fence_view_async_tmem_load()
            SK._fence_tcgen05(before=True)
            cute.arch.mbarrier_arrive(bar_tmem_alloc)
            if cutlass.const_expr(self.split_k > 1):
                vpt = cutlass.const_expr(cute.size(rmem_layout))
                vpp = cutlass.const_expr(self.epilog_threads * vpt)
                if split_rank != SK.OWNER_RANK:
                    for i in cutlass.range_constexpr(0, vpt, 4):
                        words = (rAcc[i], rAcc[i + 1], rAcc[i + 2], rAcc[i + 3])
                        slot = mailbox.iterator + (split_rank - Int32(1)) * vpp
                        slot = slot + epi_tid * vpt + i
                        SK._store_shared_remote_v4(*words, slot, bar_reduce, Int32(SK.OWNER_RANK))
                else:
                    if epi_tid == 0:
                        if cutlass.const_expr(self.k23_arm):
                            arm_cluster(bar_reduce, self.expected_transaction_bytes)
                        else:
                            cute.arch.mbarrier_arrive_and_expect_tx(bar_reduce, self.expected_transaction_bytes)
                    SK._wait_cluster(bar_reduce, Int32(0))
                    for peer in cutlass.range_constexpr(self.split_k - 1):
                        for i in cutlass.range_constexpr(vpt):
                            rAcc[i] = rAcc[i] + mailbox[peer * vpp + epi_tid * vpt + i]
            if split_rank == SK.OWNER_RANK:
                # FP32 sums -> SMEM [tile row, token] (column-major tile, as gD's (M, N) coordinates).
                sD = cute.make_tensor(staging.iterator, cute.make_layout((self.cta_m, self.cta_n), stride=(1, self.cta_m)))
                cute_ext.partition_and_copy(thr_t2r, rAcc, sD)
                cute.arch.barrier(barrier_id=1, number_of_threads=self.epilog_threads)
                tokens = mH.layout.shape[1]
                items: cutlass.Constexpr = 32 * self.cta_n
                for r in cutlass.range_constexpr((items + self.epilog_threads - 1) // self.epilog_threads):
                    item = epi_tid + r * self.epilog_threads
                    j, t = item % 32, item // 32
                    token = n_idx * self.cta_n + t
                    if item < items and token < tokens:
                        value = situ(sD[j, t], sD[32 + j, t], beta, linear_beta)
                        mH[(bidx * 32 + j, token, l_idx)] = c.BFloat16(value)

    return SplitKSituGemm


@functools.cache
def compile_sks(tactic, arch, epiwait=True, k23_arm=True):
    """Code-only cache of ``SplitKSituGemm`` for a split-K tactic (mma_m 64)."""
    import cutlass
    from cutlass.cute.runtime import make_fake_tensor

    from ..gemm.splitk import SplitKTactic

    def fake3(leading, dtype=cutlass.BFloat16):
        shape = tuple(cute.sym_int() for _ in range(3))
        stride = (1, cute.sym_int(64), 0) if leading == 0 else (cute.sym_int(64), 1, 0)
        return make_fake_tensor(dtype, shape, stride, assumed_align=16)

    kernel = _sks_classes()(tactic=SplitKTactic(*tactic), epiwait=epiwait, k23_arm=k23_arm)
    args = (fake3(1), fake3(1), fake3(0, cutlass.Float32), fake3(0), cutlass.Int64(0), Float32(4), Float32(25))
    return cute.experimental.compile(kernel, *args, make_fake_stream(), options=f"--gpu-arch={arch}")


class SharedSitu:
    """The B<=8 shared expert's gate/up + SiTU as one split-K kernel (``K31MK_EXP_SHARED_SKS``). Owns the interleaved
    copy of the gate/up weight (``interleave_gate_up``; checked to invert bitwise at construction)."""

    def __init__(self, gate_up, policies, beta, linear_beta):
        from ..common.host import architecture
        from ..gemm import projection as PJ

        self.weight, self.order = interleave_gate_up(gate_up)
        if not torch.equal(deinterleave_gate_up(self.weight, self.order), gate_up):
            raise RuntimeError("interleave_gate_up: the relayout does not invert bitwise")
        self.hidden = gate_up.shape[0] // 2
        # INT-F3: no module object on the instance. providers._owned_resource_inventory walks every attribute of the
        # resource set; a module here led it through sys.modules into transformers' lazy auto mappings (Run F boot).
        self.device = gate_up.device
        self.scalars = (Float32(beta), Float32(linear_beta))
        arch = architecture(self.device)
        self.kernels, self.maps, self.extra = {}, {}, {}
        self.epiwait = True  # SKS's epilogue threads run griddepcontrol.wait (engagement record)
        self.k23_arm = shared_sks_k23_arm()  # engagement record (K-23h)
        with torch.accelerator.device_index(self.device.index):
            for batch, tactic in dict(policies).items():
                if tactic[0] != 64:
                    raise ValueError(f"SharedSitu needs 64-row M tiles, got tactic {tactic} at B{batch}")
                self.kernels[batch] = compile_sks(tuple(tactic), arch, self.epiwait, self.k23_arm)
                if 64 not in self.maps:
                    self.maps[64] = PJ.weight_map(self.weight, 64)
        self.weight_view = PJ.matrix_view(self.weight)
        self.batches = frozenset(self.kernels)

    def run(self, x, hg, h):
        """x [B, 7168] BF16 -> h [B, H] BF16; ``hg`` [B, 2H] FP32 only shapes the grid (never written)."""
        from ..common.host import stream as current_stream
        from ..gemm import projection as PJ

        code = self.kernels[x.shape[0]]
        args = (self.weight_view, PJ.matrix_view(x), PJ.matrix_view(hg, output=True))
        args += (PJ.matrix_view(h, output=True), cutlass_int64(self.maps[64].data_ptr()))
        code(*args, *self.scalars, current_stream(self.device))
        return h


def cutlass_int64(value):
    import cutlass

    return cutlass.Int64(value)
