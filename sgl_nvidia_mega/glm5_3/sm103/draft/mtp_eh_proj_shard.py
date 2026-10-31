# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Column-sharded GEMV for the draft layer's input projection (``eh_proj``).

``out[m, col_off + n] = bf16(sum_k x[m, k] * w[n, k])`` for this rank's ``n_local`` rows
of the projection weight, written straight into the rank's column slice of a ``[M,
out_ld]`` bf16 buffer (no copy kernel), so a stock all-gather or all-reduce can combine
the slices.

* One CTA computes ``cols`` output columns over the full K; warp ``w`` owns column ``w
  // ksplit`` and K-part ``w % ksplit``; every lane streams ``vpl`` 16-byte weight
  vectors and keeps them in registers.
* Per row: an fp32 FMA chain in k order inside each lane, a fixed butterfly across the
  warp, then the K-parts summed in part order from +0, and one round to bf16. The
  summation order is fixed by the configuration, so the result is deterministic and each
  row is independent of the other rows.
* The weight is static (never written by a preceding grid), so its stream is issued
  before the grid-dependency wait (coherent loads with no L1 allocation); the
  activations are read only after it. Dependents may launch at entry: they wait for this
  grid's completion before reading the output.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32

from ..primitives import ptx_core as core
from . import _draft_ptx as ptx

# (cols, ksplit, vectors per lane) for K = 12288: rows 1-2 use 4 x 2 x 24, rows >= 3 use
# 1 x 8 x 6.
_CONFIGS = {"wide": (4, 2, 24), "deep": (1, 8, 6)}


def plan_config(rows: int) -> str:
    return "wide" if rows <= 2 else "deep"


class EhProjShard:
    """One compiled instance per row count ``rows`` (1..16)."""

    def __init__(self, rows: int, k: int = 12288, config: str | None = None):
        if not 1 <= rows <= 16:
            raise ValueError(f"rows must be in [1, 16], got {rows}")
        self.rows = rows
        self.k = k
        self.config = plan_config(rows) if config is None else config
        self.cols, self.ksplit, self.vpl = _CONFIGS[self.config]
        if k != self.ksplit * 32 * self.vpl * 8:
            raise ValueError(
                f"k must be {self.ksplit * 32 * self.vpl * 8} for config {self.config}"
            )
        self.threads = self.cols * self.ksplit * 32

    def grid(self, n_local: int) -> int:
        # every CTA owns `cols` existing rows, so no load needs a guard
        if n_local % self.cols:
            raise ValueError(f"n_local {n_local} must be a multiple of {self.cols}")
        return n_local // self.cols

    @cute.jit
    def __call__(
        self,
        weight: Int64,
        x: Int64,
        out: Int64,
        n_local: Int32,
        out_ld: Int32,
        col_off: Int32,
        grid: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(weight, x, out, n_local, out_ld, col_off).launch(
            grid=[grid, 1, 1], block=[self.threads, 1, 1], stream=stream, use_pdl=True
        )

    def weight_stream(self, w_row, lane):
        """This lane's ``vpl`` 16-byte weight vectors (a trace-time loop: fully
        unrolled). Coherent loads with no L1 allocation.
        """
        return [
            core.ld_global_v4_no_allocate(
                w_row + (Int64(i * 32) + Int64(lane)) * Int64(16)
            )
            for i in range(self.vpl)
        ]

    @cute.jit
    def wait_for_inputs(self):
        core.pdl_wait()

    @cute.kernel
    def kernel(
        self,
        weight: Int64,
        x: Int64,
        out: Int64,
        n_local: Int32,
        out_ld: Int32,
        col_off: Int32,
    ):
        t, _, _ = cute.arch.thread_idx()
        b, _, _ = cute.arch.block_idx()
        tid = Int32(t)
        core.pdl_launch_dependents()
        warp = tid // 32
        lane = tid % 32
        c = warp // self.ksplit
        kp = warp % self.ksplit
        n = Int32(b) * self.cols + c
        kbase = kp * (self.k // self.ksplit)

        # static weight stream: issued before the wait, held in registers
        w_row = weight + (Int64(n) * Int64(self.k) + Int64(kbase)) * Int64(2)
        wv = self.weight_stream(w_row, lane)

        self.wait_for_inputs()

        acc = [Float32(0.0)] * self.rows
        for i in cutlass.range_constexpr(self.vpl):
            wf = []
            for j in cutlass.range_constexpr(4):
                wf.append(ptx.bf16_low(wv[i][j]))
                wf.append(ptx.bf16_high(wv[i][j]))
            for m in cutlass.range_constexpr(self.rows):
                xa = x + (
                    Int64(m * self.k)
                    + Int64(kbase)
                    + (Int64(i * 32) + Int64(lane)) * Int64(8)
                ) * Int64(2)
                xv = core.ld_global_v4(xa)
                a = acc[m]
                for j in cutlass.range_constexpr(4):
                    a = core.fma_rn(wf[2 * j], ptx.bf16_low(xv[j]), a)
                    a = core.fma_rn(wf[2 * j + 1], ptx.bf16_high(xv[j]), a)
                acc[m] = a

        for m in cutlass.range_constexpr(self.rows):
            a = acc[m]
            for s in cutlass.range_constexpr(5):
                a = core.add_rn(a, cute.arch.shuffle_sync_bfly(a, 16 >> s))
            acc[m] = a

        part = ptx.static_smem(
            Float32,
            (self.cols, self.ksplit, self.rows),
            stride=(self.ksplit * self.rows, self.rows, 1),
        )
        if lane == 0:
            # one 4-byte store per row partial
            pbase = Uint32(part.iterator.toint()) + Uint32(
                (c * self.ksplit + kp) * self.rows * 4
            )
            for m in cutlass.range_constexpr(self.rows):
                core.st_shared_f32(pbase + Uint32(m * 4), acc[m])
        cute.arch.sync_threads()

        if tid < self.cols * self.rows:
            cc = tid // self.rows
            mm = tid % self.rows
            nn = Int32(b) * self.cols + cc
            # always true (n_local % cols == 0); a defensive bound
            if nn < n_local:
                s = Float32(0.0)
                for k in cutlass.range_constexpr(self.ksplit):
                    s = core.add_rn(s, part[cc, k, mm])
                dst = out + (
                    Int64(mm) * Int64(out_ld) + Int64(col_off) + Int64(nn)
                ) * Int64(2)
                core.st_global_b16(dst, core.cvt_rn_bf16_f32(s))
