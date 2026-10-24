# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""All-reduce of a bf16 ``[rows, HIDDEN]`` partial over the tensor-parallel group, fused
with the residual add and RMSNorm (``residual`` / ``norm`` variants).

Grid: ``rows`` clusters of 8 consumer CTAs (one per row) plus one acknowledgement
cluster; 544 threads (512 polling + one epoch warp); cooperative (every CTA spins on
records written by CTAs of the same grid on other ranks, so all CTAs must be
co-resident); programmatic dependent launch: ``griddepcontrol.wait`` then
``launch_dependents`` at entry. Each CTA publishes the 256 records of its slice of the
row to every rank and consumes the same slice.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass import Float32, Int32, Int64, Uint32

from ..primitives import ptx_sys as sys_ops
from . import _ptx_local as P
from .constants import BF16_NAN, CONSUMER_THREADS, HIDDEN, POLL_THREADS, TAG_MASK
from .transport import ConsSmem, Consumer, Mbox


class AllreduceNormHop(Consumer):
    """``out_res = residual + allreduce(partial)`` (or the plain all-reduce without a
    residual), ``out_norm = rmsnorm(out_res) * gamma`` when ``norm``."""

    def __init__(self, *, residual: bool = True, norm: bool = True, max_rows: int = 8):
        if residual and not norm:
            raise ValueError("residual without RMSNorm is not provided")
        if max_rows != 8:
            raise ValueError("this hop is provided for up to 8 rows")
        self.residual, self.norm, self.max_rows = residual, norm, max_rows

    @cute.jit
    def __call__(
        self,
        partial: Int64,
        residual: Int64,
        gamma: Int64,
        out_res: Int64,
        out_norm: Int64,
        eps: Float32,
        rows: Int32,
        p0: Int64,
        p1: Int64,
        p2: Int64,
        p3: Int64,
        p4: Int64,
        p5: Int64,
        p6: Int64,
        p7: Int64,
        rank: Int32,
        err_host: Int64,
        site: Int32,
        stream: cuda.CUstream,
    ):
        """``p0 .. p7``: mailbox bases rotated so that ``p0`` is this rank's own."""
        self.kernel(
            partial,
            residual,
            gamma,
            out_res,
            out_norm,
            eps,
            rows,
            p0,
            p1,
            p2,
            p3,
            p4,
            p5,
            p6,
            p7,
            rank,
            err_host,
            site,
        ).launch(
            grid=[rows * 8 + 8, 1, 1],
            block=[CONSUMER_THREADS, 1, 1],
            max_number_threads=[CONSUMER_THREADS, 1, 1],
            cluster=[8, 1, 1],
            cooperative=True,
            use_pdl=True,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        partial: Int64,
        residual: Int64,
        gamma: Int64,
        out_res: Int64,
        out_norm: Int64,
        eps: Float32,
        rows: Int32,
        p0: Int64,
        p1: Int64,
        p2: Int64,
        p3: Int64,
        p4: Int64,
        p5: Int64,
        p6: Int64,
        p7: Int64,
        rank: Int32,
        err_host: Int64,
        site: Int32,
    ):
        mb = Mbox((p0, p1, p2, p3, p4, p5, p6, p7), rank, self.max_rows, err_host, site)
        t = P.thread_index()
        bx = P.block_index()
        smem = cutlass.utils.SmemAllocator()
        cs = ConsSmem.allocate(smem, self.max_rows)
        nmain = rows * 8
        if cutlass.const_expr(self.norm):  # noqa: SIM102
            if bx < nmain:
                self.cons_prologue(cs)
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        sys_ops.fence_proxy_async_global()  # inputs may come from an async-proxy writer
        if bx >= nmain:
            e = self.epoch_begin(mb, cs.es, False)
            if t == 0:
                self.epoch_arrive(mb, cs.es, rows, P.grid_size())
                if bx == nmain:
                    self.ack_cta(mb, e)
            sys_ops.exit_thread()
        m = bx // 8
        c = bx % 8
        h0 = c * 768 + 6 * t
        w = cute.make_rmem_tensor((3,), Uint32)
        ru = cute.make_rmem_tensor((3,), Uint32)
        gu = cute.make_rmem_tensor((3,), Uint32)
        w.fill(0)
        ru.fill(0)
        gu.fill(0)
        # every input load is issued before the epoch / acknowledgement check
        if t < 128:
            off = Int64(2) * (Int64(m) * HIDDEN + Int64(h0))
            if cutlass.const_expr(self.residual):
                for k in cutlass.range_constexpr(3):
                    ru[k] = sys_ops.ld_cg_u32(residual + off + Int64(4 * k))
            if cutlass.const_expr(self.norm):
                for k in cutlass.range_constexpr(3):
                    gu[k] = sys_ops.ld_cg_u32(
                        gamma + Int64(2) * Int64(h0) + Int64(4 * k)
                    )
            for k in cutlass.range_constexpr(3):
                w[k] = sys_ops.ld_cg_u32(partial + off + Int64(4 * k))
        self.epoch_begin(mb, cs.es, True)
        pz = cs.es.pz[0]
        em = cs.es.erow[m]  # row epoch: buffer slot and tag of row m
        buf = self.buffer_of(em)
        tag = em & Uint32(TAG_MASK)
        if t < 128:
            for k in cutlass.range_constexpr(3):
                if pz != 0:
                    w[k] = Uint32(BF16_NAN | (BF16_NAN << 16))
                else:
                    w[k] = P.no_neg_zero(w[k])
            r0, r1 = self.pack_records(w[0], w[1], w[2], tag)
            self.delay(mb, 1)
            self.publish_records(mb, buf, m, c * 256 + 2 * t, r0, r1)
        if t >= POLL_THREADS:  # the epoch warp arrives, then leaves the CTA
            if t == POLL_THREADS:
                self.epoch_arrive(mb, cs.es, rows, P.grid_size())
            sys_ops.exit_thread()
        self.consume_cta(
            mb,
            cs,
            m,
            c,
            em,
            pz,
            ru,
            gu,
            out_res,
            out_norm,
            eps,
            self.residual,
            self.norm,
        )
