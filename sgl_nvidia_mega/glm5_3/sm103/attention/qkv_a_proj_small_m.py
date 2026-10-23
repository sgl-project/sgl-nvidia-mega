# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Row-owned skinny GEMM for the attention input projections at decode / verify row
counts (rows <= 8).

    out[m, n] = bf16( sum_k x[m, k] * W[n, k] ),   x: [rows, 6144] BF16, W: [N, 6144]
    BF16, FP32 accumulate

Rows ``n < n_main`` go to ``out_main[:, n]``, the rest to ``out_aux[:, n - n_main]``
(one launch serves the fused q_a / kv_a projection and the indexer key / weights
projection of a layer).

Work split: CTA ``c`` of ``grid`` CTAs owns the even-aligned output rows ``[2 floor(G2 c
/ grid), 2 floor(G2 (c + 1) / grid))`` with ``G2 = N / 2`` and the full K: no cross-CTA
reduction. Its packed weight block (``packing.pack_qkv_a``, which depends on ``grid``)
is contiguous: ``[k-step 192][row R][32]``. Warp ``w`` owns k-steps ``[24 w, 24 w +
24)`` of every row and streams them through a ring of ``slots`` stages (4 k-steps each,
one mbarrier per slot); the first ``slots`` stages are issued before the PDL wait
(constant weights only). x stays in registers (24 x 16 B per lane, loaded right after
the wait). ``mma.sync`` m16n8k16 with the rows of x on A (rows ``g < rows``; rows ``g +
8`` zero) and 8 weight rows per n8 tile on B (the tile count of the largest CTA; rows a
CTA does not own are neither loaded nor stored); one 16-byte shared load per tile and
k-step feeds two MMAs. The 8 warp partials are
added in warp order from 0.0, rounded to BF16 once, and stored.

Option fields (compile time): ``n_rows`` (N), ``n_main``, ``grid`` (the launch's CTA
count, normally the SM count; the packing must use the same value), ``slots`` (ring
depth, ``plan_slots``).

Ordering: dependents are triggered at entry; only the constant weights are read before
``griddepcontrol.wait``; ``fence.proxy.async.global`` follows the wait and each storing
thread's output store. A ring slot is re-filled only after the warp's reads of it
(``__syncwarp``) and ``fence.proxy.async.shared::cta``.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32

from ..primitives import ptx_core as ptx

K = 6144
STEPS = K // 32  # 192 k-steps of 32
WARPS = 8
THREADS = WARPS * 32
STEPS_PER_WARP = STEPS // WARPS  # 24
STEPS_PER_STAGE = 4
STAGES = STEPS_PER_WARP // STEPS_PER_STAGE  # 6
MAX_ROWS = 8
MAX_TILES = 4  # n8 tiles per CTA (<= 32 output rows)
SMEM_BUDGET = 227 * 1024


def rows_of(n_rows: int, grid: int, c: int) -> int:
    groups = n_rows // 2
    return 2 * (groups * (c + 1) // grid - groups * c // grid)


def plan_slots(n_rows: int, grid: int, smem_budget: int = SMEM_BUDGET) -> int:
    """Ring depth for the shared-memory budget (0 if even one stage per warp does not
    fit)."""
    rmax = max(rows_of(n_rows, grid, c) for c in range(grid))
    stride = STEPS_PER_STAGE * rmax * 64
    budget = min(smem_budget, 227 * 1024 - 1024)
    slots = min(STAGES, budget // (WARPS * stride))
    if slots * stride < MAX_TILES * 8 * 8 * 4:
        slots = 0
    return slots


class QkvAProjSmallM:
    """See the module docstring. Call arguments (device addresses; strides in elements):
    ``weights`` (packed), ``x`` and its row stride, ``rows`` (1..8), ``out_main`` and
    its row stride, ``out_aux`` and its row stride (unused
    when ``n_main == n_rows``), then the stream."""

    def __init__(self, n_rows: int, n_main: int, grid: int, slots: int = 0):
        if n_rows < 2 or n_rows % 2 or n_main < 2 or n_main % 2 or n_main > n_rows:
            raise ValueError("n_rows and n_main must be even, 2 <= n_main <= n_rows")
        if not 1 <= grid <= 512:
            raise ValueError("grid must be in [1, 512]")
        rs = [rows_of(n_rows, grid, c) for c in range(grid)]
        if min(rs) < 2 or max(rs) > 8 * MAX_TILES:
            raise ValueError(
                f"rows per CTA must be in [2, 32], got {min(rs)}..{max(rs)}: "
                "choose grid >= N / 32"
            )
        self.n_rows, self.n_main, self.grid = n_rows, n_main, grid
        self.slots = slots or plan_slots(n_rows, grid)
        if self.slots < 1:
            raise ValueError(
                "shared memory budget too small for one ring stage per warp"
            )
        self.stage_stride = STEPS_PER_STAGE * max(rs) * 64
        # n8 tiles per CTA: the most any CTA owns; a CTA with fewer rows skips the
        # shared loads of its missing rows (their MMAs see zeros, their outputs are not
        # stored)
        self.tiles = max((r + 7) // 8 for r in rs)

    @cute.jit
    def __call__(
        self,
        weights: Int64,
        x: Int64,
        x_stride: Int64,
        rows: Int32,
        out_main: Int64,
        out_main_stride: Int64,
        out_aux: Int64,
        out_aux_stride: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(
            weights,
            x,
            x_stride,
            rows,
            out_main,
            out_main_stride,
            out_aux,
            out_aux_stride,
        ).launch(
            grid=[self.grid, 1, 1],
            block=[THREADS, 1, 1],
            stream=stream,
            use_pdl=True,
            max_number_threads=[THREADS, 1, 1],
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        weights: Int64,
        x: Int64,
        x_stride: Int64,
        rows: Int32,
        out_main: Int64,
        out_main_stride: Int64,
        out_aux: Int64,
        out_aux_stride: Int64,
    ):
        self.body(
            weights,
            x,
            x_stride,
            rows,
            out_main,
            out_main_stride,
            out_aux,
            out_aux_stride,
        )

    @cute.jit
    def body(
        self,
        weights,
        x,
        x_stride,
        rows,
        out_main,
        out_main_stride,
        out_aux,
        out_aux_stride,
        dn_bsum=None,
        dn_eps=None,
    ):
        """The kernel's body; ``dn_bsum`` selects the epilogue-scaled form
        (``QkvAProjSmallMDn``)."""
        slots = self.slots
        stride = self.stage_stride
        tid_, _, _ = cute.arch.thread_idx()
        cta_, _, _ = cute.arch.block_idx()
        tid = Int32(tid_)
        cta = Int64(cta_)
        smem = cutlass.utils.SmemAllocator()
        rings = smem.allocate_tensor(
            cutlass.Uint8, cute.make_layout(WARPS * slots * stride), byte_alignment=1024
        )
        bars = smem.allocate_array(Int64, WARPS * STAGES)
        s_ring0 = ptx.shared_address(rings.iterator)
        s_bar0 = ptx.shared_address(bars)

        ptx.pdl_launch_dependents()
        lane = tid % 32
        warp = tid // 32
        g = lane // 4
        t = lane % 4
        groups = Int64(self.n_rows // 2)
        g0 = groups * cta // Int64(self.grid)
        g1 = groups * (cta + Int64(1)) // Int64(self.grid)
        r0 = Int32(g0 * 2)
        nr = Int32((g1 - g0) * 2)  # this CTA's output rows R
        stage_bytes = STEPS_PER_STAGE * nr * 64
        wsrc = (
            weights
            + Int64(r0) * (K * 2)
            + Int64(warp * STEPS_PER_WARP) * Int64(nr) * 64
        )
        ring = s_ring0 + warp * (slots * stride)
        bar0 = s_bar0 + warp * (STAGES * 8)
        policy = ptx.l2_policy("evict_first")

        # constant weights: the first `slots` stages of this warp, issued before the PDL
        # wait
        if lane == 0:
            for k in cutlass.range_constexpr(slots):
                ptx.mbarrier_init(bar0 + k * 8, 1)
            ptx.fence_mbarrier_init_release_cluster()
            for q in cutlass.range_constexpr(min(slots, STAGES)):
                self.issue(ring, bar0, wsrc, stage_bytes, q, policy)
        cute.arch.sync_warp()

        self.wait_inputs()

        # epilogue-scaled form: this thread's output row's scale, from the block sums
        # the predecessor wrote (loaded after the wait, used after the GEMM)
        inv = Float32(1.0)
        if cutlass.const_expr(dn_bsum is not None):
            mi = (tid // 8) % 8
            tot = Float32(0.0)
            if mi < rows:
                for cc in cutlass.range_constexpr(8):
                    tot = ptx.add_rn(
                        tot,
                        ptx.ld_global_f32(dn_bsum + (Int64(mi) * 8 + Int64(cc)) * 4),
                    )
            inv = ptx.rsqrt_approx(
                ptx.fma_rn(tot, ptx.rcp_rn(Float32(float(K))), dn_eps)
            )

        # x fragments for this warp's 24 k-steps (rows g < rows): coherent loads back to
        # back
        xrow = x + (Int64(g) * x_stride + Int64(warp * STEPS_PER_WARP * 32 + 8 * t)) * 2
        xr = []
        for i in cutlass.range_constexpr(STEPS_PER_WARP):
            xr.append(self.load_row(xrow + i * 64, g < rows))

        zero = Float32(0.0)
        acc = [(zero, zero, zero, zero) for _ in range(self.tiles)]
        z = Uint32(0)
        for q in cutlass.range_constexpr(STAGES):
            slot = q % slots
            self.wait_stage(bar0 + slot * 8, (q // slots) & 1)
            st = ring + slot * stride
            for ii in cutlass.range_constexpr(STEPS_PER_STAGE):
                a = xr[q * STEPS_PER_STAGE + ii]
                blk = st + ii * nr * 64
                for j in cutlass.range_constexpr(self.tiles):
                    b0 = Uint32(0)
                    b1 = Uint32(0)
                    b2 = Uint32(0)
                    b3 = Uint32(0)
                    if 8 * j + g < nr:
                        b0, b1, b2, b3 = ptx.ld_shared_v4(
                            blk + (8 * j + g) * 64 + t * 16
                        )
                    c = ptx.mma_m16n8k16_bf16((a[0], z, a[1], z), (b0, b1), acc[j])
                    acc[j] = ptx.mma_m16n8k16_bf16((a[2], z, a[3], z), (b2, b3), c)
            if cutlass.const_expr(q + slots < STAGES):
                self.refill_visible()
                if lane == 0:
                    ptx.fence_proxy_async_shared_cta()
                    self.issue(ring, bar0, wsrc, stage_bytes, q + slots, policy)

        # cross-warp reduction in warp order: each warp's partials go to its own (fully
        # consumed) ring
        if g < rows:
            for j in cutlass.range_constexpr(self.tiles):
                ptx.st_shared_v2_f32(
                    ring + ((j * 8 + g) * 8 + 2 * t) * 4, acc[j][0], acc[j][1]
                )
        self.sync_partials()
        j = tid // 64
        m = (tid // 8) % 8
        n = tid % 8
        row = 8 * j + n
        if (row < nr) & (m < rows):
            v = Float32(0.0)
            for w in cutlass.range_constexpr(WARPS):
                v = ptx.add_rn(
                    v,
                    ptx.ld_shared_f32(
                        s_ring0 + w * (slots * stride) + ((j * 8 + m) * 8 + n) * 4
                    ),
                )
            if cutlass.const_expr(dn_bsum is not None):
                v = ptx.mul_rn(v, inv)
            o = ptx.cvt_rn_bf16_f32(v)
            gr = r0 + row
            if gr < self.n_main:
                ptx.st_global_b16(
                    out_main + (Int64(m) * out_main_stride + Int64(gr)) * 2, o
                )
            else:
                ptx.st_global_b16(
                    out_aux + (Int64(m) * out_aux_stride + Int64(gr - self.n_main)) * 2,
                    o,
                )
            self.release_outputs()

    # ------------------------------------------------------------------ steps
    @cute.jit
    def issue(
        self,
        ring: Int32,
        bar0: Int32,
        wsrc: Int64,
        stage_bytes: Int32,
        q: cutlass.Constexpr[int],
        policy,
    ):
        """Lane 0: stage q of this warp into slot q % slots."""
        bar = bar0 + (q % self.slots) * 8
        ptx.mbarrier_arrive_expect_tx(bar, stage_bytes)
        ptx.bulk_copy_g2s(
            ring + (q % self.slots) * self.stage_stride,
            wsrc + Int64(q) * Int64(stage_bytes),
            stage_bytes,
            bar,
            policy,
        )

    @cute.jit
    def load_row(self, addr: Int64, valid):
        w0 = Uint32(0)
        w1 = Uint32(0)
        w2 = Uint32(0)
        w3 = Uint32(0)
        if valid:
            w0, w1, w2, w3 = ptx.ld_global_v4(addr)
        return w0, w1, w2, w3

    @cute.jit
    def wait_inputs(self):
        ptx.pdl_wait()
        ptx.fence_proxy_async_global()

    @cute.jit
    def wait_stage(self, bar: Int32, parity: cutlass.Constexpr[int]):
        ptx.mbarrier_wait_parity(bar, parity)

    @cute.jit
    def refill_visible(self):
        """The warp's generic reads of a slot -> the async-proxy re-fill of that slot
        (with the proxy fence)."""
        cute.arch.sync_warp()

    @cute.jit
    def sync_partials(self):
        ptx.sync_threads()

    @cute.jit
    def release_outputs(self):
        ptx.fence_proxy_async_global()


class QkvAProjSmallMDn(QkvAProjSmallM):
    """``QkvAProjSmallM`` with the input's RMSNorm scale applied in the epilogue: the
    input is ``u = bf16(x * gamma)`` (un-normalized) and every output row is

        out[m, n] = bf16( inv_m * sum_k u[m, k] * W[n, k] ),
        inv_m = rsqrt(fma(sum_c bsum[m][c], rcp(6144), eps)),

    the fp32 accumulator scaled before its single bf16 rounding, so the output is
    rounded at the scale of the normed input. ``bsum`` ``[rows, 8]`` fp32 are the
    block sums of squares of ``x`` that the producer hop wrote (summed from 0 in
    ``c`` order, the norm's own arithmetic); they are read after ``griddepcontrol.wait``.
    Call arguments: those of ``QkvAProjSmallM``, then ``bsum`` and ``eps``, then the
    stream."""

    @cute.jit
    def __call__(
        self,
        weights: Int64,
        x: Int64,
        x_stride: Int64,
        rows: Int32,
        out_main: Int64,
        out_main_stride: Int64,
        out_aux: Int64,
        out_aux_stride: Int64,
        bsum: Int64,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        self.kernel_dn(
            weights,
            x,
            x_stride,
            rows,
            out_main,
            out_main_stride,
            out_aux,
            out_aux_stride,
            bsum,
            eps,
        ).launch(
            grid=[self.grid, 1, 1],
            block=[THREADS, 1, 1],
            stream=stream,
            use_pdl=True,
            max_number_threads=[THREADS, 1, 1],
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel_dn(
        self,
        weights: Int64,
        x: Int64,
        x_stride: Int64,
        rows: Int32,
        out_main: Int64,
        out_main_stride: Int64,
        out_aux: Int64,
        out_aux_stride: Int64,
        bsum: Int64,
        eps: Float32,
    ):
        self.body(
            weights,
            x,
            x_stride,
            rows,
            out_main,
            out_main_stride,
            out_aux,
            out_aux_stride,
            bsum,
            eps,
        )
