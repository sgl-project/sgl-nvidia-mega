# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MLA value up-projection for decode / verify rows, with L2 staging of static tensors.

For the 8 attention heads of one tensor-parallel rank:

    out[m, 256 h + v] = sum_l attn[m, h, l] * W_UV[h][v, l]        (m < rows)

with BF16 inputs, FP32 accumulation and one BF16 rounding of each output. ``W_UV[h]``
is rows ``[448 h + 192, 448 h + 448)`` of ``kv_b_proj.weight`` (see
``packing.pack_uv_proj``). It replaces the batched GEMM ``bmm(attn[8, M, 512],
W_UV[8, 512, 256])`` at up to ``max_rows`` rows (8 or 16).

Work split: 128 CTAs of 256 threads. CTA ``(h, j)`` owns value dims ``[16 j, 16 j +
16)`` of head ``h`` over the whole latent (K = 512); warp ``w`` owns latent columns
``[64 w, 64 w + 64)`` and sums them with ``mma.sync`` m16n8k16 (A = attention rows, B =
weights). The 8 warp partials are added in warp order 0..7 from 0.0, then rounded to
BF16 once, so the result of a row does not depend on the other rows or on ``max_rows``.

Ordering (PTX ISA memory consistency model):
  * The kernel triggers its dependents at entry; nothing it writes is read by a
    dependent before that dependent's own ``griddepcontrol.wait``.
  * Before ``griddepcontrol.wait`` it reads only constant data: its weight tile (one
    bulk copy completing on an mbarrier) and the L2 prefetches below.
  * After the wait, ``fence.proxy.async.global`` orders the attention rows (which a
    predecessor may have written through the async proxy) before the generic loads.
  * Partials go through shared memory and ``bar.sync``.
  * Every thread runs ``fence.proxy.async.global`` after its last output store, for a
    consumer that reads the output through the async proxy.

L2 staging: before the wait, the CTAs issue bulk L2 prefetches of up to two static
tensors (for example weights read later in the same step). Each slot stages either
its first ``bytes`` bytes (``stride == 0``) or, for every ``stride``-byte slab, the
first ``prefix`` bytes of that slab, with eviction priority ``policy`` (0 last,
1 normal, 2 first). The work items (4 KiB chunks) of slot 0 then slot 1 form one
list; item ``i`` is issued by CTA ``i % 128``. Prefetches are hints: no output
depends on them.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint8

from ..primitives import ptx_core as ptx

HEADS = 8
LATENT = 512
VALUE_DIM = 256
VALUES_PER_CTA = 16
CTAS_PER_HEAD = VALUE_DIM // VALUES_PER_CTA
GRID = HEADS * CTAS_PER_HEAD
THREADS = 256
WARPS = THREADS // 32
STEPS_PER_WARP = LATENT // 32 // WARPS
N_TILES = VALUES_PER_CTA // 8
WEIGHT_TILE_BYTES = (LATENT // 32) * VALUES_PER_CTA * 64
STAGE_CHUNK = 4096
MAX_ROWS = (8, 16)
POLICIES = ("evict_last", "evict_normal", "evict_first")
assert max(MAX_ROWS) * VALUES_PER_CTA <= THREADS


class MlaUvProj:
    """``out[rows, 2048] = attn[rows, 8, 512] x W_UV`` for ``rows <= max_rows``.

    Arguments of a call (all addresses are device addresses; strides in elements):
    ``weights`` (packed by ``packing.pack_uv_proj``), ``attn`` (BF16, unit stride
    in the latent, rows and heads 16-byte aligned) with its row and head strides,
    ``out`` (BF16, unit stride in the value dim) with its row stride, ``rows``, then
    two staging slots ``(address or 0, bytes, stride, prefix, policy)``, then the
    stream.
    """

    def __init__(self, max_rows: int):
        if max_rows not in MAX_ROWS:
            raise ValueError(f"max_rows must be one of {MAX_ROWS}, got {max_rows}")
        self.max_rows = max_rows

    @cute.jit
    def __call__(
        self,
        weights: Int64,
        attn: Int64,
        attn_row_stride: Int64,
        attn_head_stride: Int64,
        out: Int64,
        out_row_stride: Int64,
        rows: Int32,
        stage0: Int64,
        stage0_bytes: Int64,
        stage0_stride: Int64,
        stage0_prefix: Int64,
        stage0_policy: Int32,
        stage1: Int64,
        stage1_bytes: Int64,
        stage1_stride: Int64,
        stage1_prefix: Int64,
        stage1_policy: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            weights,
            attn,
            attn_row_stride,
            attn_head_stride,
            out,
            out_row_stride,
            rows,
            stage0,
            stage0_bytes,
            stage0_stride,
            stage0_prefix,
            stage0_policy,
            stage1,
            stage1_bytes,
            stage1_stride,
            stage1_prefix,
            stage1_policy,
        ).launch(
            grid=[GRID, 1, 1],
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
        attn: Int64,
        attn_row_stride: Int64,
        attn_head_stride: Int64,
        out: Int64,
        out_row_stride: Int64,
        rows: Int32,
        stage0: Int64,
        stage0_bytes: Int64,
        stage0_stride: Int64,
        stage0_prefix: Int64,
        stage0_policy: Int32,
        stage1: Int64,
        stage1_bytes: Int64,
        stage1_stride: Int64,
        stage1_prefix: Int64,
        stage1_policy: Int32,
    ):
        max_rows = self.max_rows
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        tile = smem.allocate_tensor(
            Uint8, cute.make_layout(WEIGHT_TILE_BYTES), byte_alignment=1024
        )
        partial = smem.allocate_tensor(
            Float32,
            cute.make_layout(WARPS * max_rows * VALUES_PER_CTA),
            byte_alignment=16,
        )
        mbar = smem.allocate_array(Int64, 1)
        tile_s = ptx.shared_address(tile.iterator)
        partial_s = ptx.shared_address(partial.iterator)
        bar = ptx.shared_address(mbar)

        ptx.pdl_launch_dependents()
        tid = Int32(tid)
        cta = Int32(cta)
        lane = tid % 32
        warp = tid // 32
        g = lane // 4
        t = lane % 4
        head = cta // CTAS_PER_HEAD
        part = cta % CTAS_PER_HEAD

        # Constant weight tile: one bulk copy, issued before the PDL wait.
        if tid == 0:
            ptx.mbarrier_init(bar, 1)
            ptx.fence_mbarrier_init_release_cluster()
            ptx.mbarrier_arrive_expect_tx(bar, WEIGHT_TILE_BYTES)
            src = weights + Int64(cta) * WEIGHT_TILE_BYTES
            ptx.bulk_copy_g2s(tile_s, src, WEIGHT_TILE_BYTES, bar)
        ptx.sync_threads()

        self.stage_l2(
            cta,
            tid,
            stage0,
            stage0_bytes,
            stage0_stride,
            stage0_prefix,
            stage0_policy,
            stage1,
            stage1_bytes,
            stage1_stride,
            stage1_prefix,
            stage1_policy,
        )

        self.wait_inputs()

        # A fragments: lane (g, t) of warp w holds latent [64 w + 32 i + 8 t, +8) of
        # rows g and (16-row form) g + 8.
        lo = cute.make_rmem_tensor((STEPS_PER_WARP, 4), cutlass.Uint32)
        lo.fill(0)
        hi = cute.make_rmem_tensor((STEPS_PER_WARP, 4), cutlass.Uint32)
        hi.fill(0)
        row_lo = (
            attn
            + (
                Int64(g) * attn_row_stride
                + Int64(head) * attn_head_stride
                + Int64(warp * (STEPS_PER_WARP * 32) + 8 * t)
            )
            * 2
        )
        row_hi = row_lo + Int64(8) * attn_row_stride * 2
        for i in cutlass.range_constexpr(STEPS_PER_WARP):
            if g < rows:
                v = ptx.ld_global_v4(row_lo + i * 64)
                for k in cutlass.range_constexpr(4):
                    lo[i, k] = v[k]
            if cutlass.const_expr(max_rows == 16):  # noqa: SIM102
                if g + 8 < rows:
                    v = ptx.ld_global_v4(row_hi + i * 64)
                    for k in cutlass.range_constexpr(4):
                        hi[i, k] = v[k]

        self.wait_weights(bar)

        acc = cute.make_rmem_tensor((N_TILES, 4), Float32)
        acc.fill(0.0)
        for i in cutlass.range_constexpr(STEPS_PER_WARP):
            block = tile_s + (warp * STEPS_PER_WARP + i) * (VALUES_PER_CTA * 64)
            for n in cutlass.range_constexpr(N_TILES):
                b = ptx.ld_shared_v4(block + (8 * n + g) * 64 + t * 16)
                c = (acc[n, 0], acc[n, 1], acc[n, 2], acc[n, 3])
                c = ptx.mma_m16n8k16_bf16(
                    (lo[i, 0], hi[i, 0], lo[i, 1], hi[i, 1]), (b[0], b[1]), c
                )
                c = ptx.mma_m16n8k16_bf16(
                    (lo[i, 2], hi[i, 2], lo[i, 3], hi[i, 3]), (b[2], b[3]), c
                )
                for k in cutlass.range_constexpr(4):
                    acc[n, k] = c[k]

        # Warp partials -> shared memory -> fixed-order sum -> BF16.
        if g < rows:
            for n in cutlass.range_constexpr(N_TILES):
                at = (
                    partial_s
                    + ((warp * max_rows + g) * VALUES_PER_CTA + 8 * n + 2 * t) * 4
                )
                ptx.st_shared_v2_f32(at, acc[n, 0], acc[n, 1])
        if cutlass.const_expr(max_rows == 16):  # noqa: SIM102
            if g + 8 < rows:
                for n in cutlass.range_constexpr(N_TILES):
                    at = (
                        partial_s
                        + ((warp * max_rows + g + 8) * VALUES_PER_CTA + 8 * n + 2 * t)
                        * 4
                    )
                    ptx.st_shared_v2_f32(at, acc[n, 2], acc[n, 3])
        self.sync_partials()

        # One output element per thread (rows * 16 <= 256 threads): thread tid owns row
        # tid / 16, value tid % 16 of this CTA's slice.
        if tid < rows * VALUES_PER_CTA:
            r = tid // VALUES_PER_CTA
            d = tid % VALUES_PER_CTA
            s = Float32(0.0)
            for w in cutlass.range_constexpr(WARPS):
                at = partial_s + ((w * max_rows + r) * VALUES_PER_CTA + d) * 4
                s = ptx.add_rn(s, ptx.ld_shared_f32(at))
            col = head * VALUE_DIM + part * VALUES_PER_CTA + d
            dst = out + (Int64(r) * out_row_stride + Int64(col)) * 2
            ptx.st_global_b16(dst, ptx.cvt_rn_bf16_f32(s))
        self.release_outputs()

    # Synchronization points, one method each.

    @cute.jit
    def wait_inputs(self):
        """Wait for the predecessor grid; its async-proxy writes before our loads."""
        ptx.pdl_wait()
        ptx.fence_proxy_async_global()

    @cute.jit
    def wait_weights(self, bar: Int32):
        """Wait for the weight tile's bulk copy (phase 0 of ``bar``)."""
        ptx.mbarrier_wait_parity(bar, 0)

    @cute.jit
    def sync_partials(self):
        """Warp partials in shared memory are visible to every thread."""
        ptx.sync_threads()

    @cute.jit
    def release_outputs(self):
        """Every thread, after its last output store: order the generic stores before
        a consumer's async-proxy reads."""
        ptx.fence_proxy_async_global()

    @cute.jit
    def issue_prefetch(self, addr: Int64, nbytes: Int32, policy: cutlass.Uint64):
        ptx.bulk_prefetch_l2(addr, nbytes, policy)

    @cute.jit
    def stage_l2(
        self,
        cta: Int32,
        tid: Int32,
        stage0: Int64,
        stage0_bytes: Int64,
        stage0_stride: Int64,
        stage0_prefix: Int64,
        stage0_policy: Int32,
        stage1: Int64,
        stage1_bytes: Int64,
        stage1_stride: Int64,
        stage1_prefix: Int64,
        stage1_policy: Int32,
    ):
        """Issue this thread's share of the L2 prefetches (see the module docstring)."""
        count0, per0, prefix0 = self.slot_items(
            stage0, stage0_bytes, stage0_stride, stage0_prefix
        )
        count1, per1, prefix1 = self.slot_items(
            stage1, stage1_bytes, stage1_stride, stage1_prefix
        )
        total = count0 + count1
        if total > Int64(0):
            policy0 = self.policy(stage0_policy)
            policy1 = self.policy(stage1_policy)
            item = Int64(tid) * GRID + Int64(cta)
            while item < total:
                base = stage0
                size = stage0_bytes
                stride = stage0_stride
                per = per0
                prefix = prefix0
                policy = policy0
                i = item
                if item >= count0:
                    base = stage1
                    size = stage1_bytes
                    stride = stage1_stride
                    per = per1
                    prefix = prefix1
                    policy = policy1
                    i = item - count0
                slab = i // per
                chunk = (i % per) * STAGE_CHUNK
                offset = slab * stride + chunk
                remaining = prefix - chunk
                if size - offset < remaining:
                    remaining = size - offset
                if remaining > Int64(0):
                    nbytes = Int64(STAGE_CHUNK)
                    if remaining < nbytes:
                        nbytes = remaining
                    nbytes32 = Int32(nbytes) & Int32(-16)
                    if nbytes32 != Int32(0):
                        self.issue_prefetch(base + offset, nbytes32, policy)
                item += Int64(GRID * THREADS)

    @cute.jit
    def slot_items(self, base: Int64, size: Int64, stride: Int64, prefix: Int64):
        """(items, items per slab, bytes staged per slab) of one staging slot."""
        count = Int64(0)
        per = Int64(1)
        staged = Int64(0)
        if base != Int64(0):
            staged = size
            slabs = Int64(1)
            if stride > Int64(0):
                staged = stride
                if prefix < stride:
                    staged = prefix
                slabs = (size + stride - 1) // stride
            per = (staged + STAGE_CHUNK - 1) // STAGE_CHUNK
            count = slabs * per
        return count, per, staged

    @cute.jit
    def policy(self, code: Int32):
        result = ptx.l2_policy(POLICIES[0])
        if code == Int32(1):
            result = ptx.l2_policy(POLICIES[1])
        elif code == Int32(2):
            result = ptx.l2_policy(POLICIES[2])
        return result
