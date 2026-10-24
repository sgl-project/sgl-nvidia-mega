# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Draft-layer input front: embedding gather + two RMSNorms + concat, one CTA per token.

``out[i, :H] = rmsnorm(table[ids[i]]) * enorm_w`` and ``out[i, H:] = rmsnorm(prev[i]) *
hnorm_w`` in bf16, for a replicated ``[vocab, H]`` embedding table. The arithmetic of
each norm is described by option fields (no named configurations live here; each caller
passes the values that reproduce its own reference):

* ``sum_rule``: how the squares are summed. ``"lane_fma_cta_tree"``: each thread holds 8
  consecutive elements and accumulates ``fma(x, x, s)`` from +0 in element order, then a
  warp butterfly (xor 16, 8, 4, 2, 1; ``s + other``), then warp 0 reduces the per-warp
  sums (lanes past the warp count contribute +0) with the same butterfly.
* ``mean_rule``: ``"div_rn"``: ``mean = div.rn(sum, H)``, then ``mean + eps``.
* ``rsqrt_ftz``: whether the reciprocal square root flushes subnormals
  (``rsqrt.approx[.ftz].f32``).
* ``apply_rule``: ``"x_r_w"``: ``bf16_rn((x * r) * w)`` with two round-to-nearest
  products.

An id outside ``[0, vocab)`` traps. The ids, the rows and the previous hidden states are
read after the grid-dependency wait; dependents are released after the last store.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64

from ..primitives import ptx_core as core
from ..primitives import ptx_sys as sys_ops
from . import _draft_ptx as ptx

SUM_RULES = ("lane_fma_cta_tree",)
MEAN_RULES = ("div_rn",)
APPLY_RULES = ("x_r_w",)


class EmbedEhNorm:
    """One compiled instance per hidden size and option set; any number of tokens (one
    CTA each).
    """

    def __init__(
        self,
        hidden: int = 6144,
        *,
        sum_rule: str,
        mean_rule: str,
        rsqrt_ftz: bool,
        apply_rule: str,
    ):
        if (
            sum_rule not in SUM_RULES
            or mean_rule not in MEAN_RULES
            or apply_rule not in APPLY_RULES
        ):
            raise ValueError(
                "unsupported norm options "
                f"{sum_rule!r} / {mean_rule!r} / {apply_rule!r}"
            )
        if hidden % 256 or not 256 < hidden <= 8192:
            raise ValueError(f"hidden size {hidden}: a multiple of 256 in (256, 8192]")
        self.hidden = hidden
        self.threads = hidden // 8
        self.warps = self.threads // 32
        self.rsqrt_ftz = rsqrt_ftz

    @cute.jit
    def __call__(
        self,
        ids: Int64,
        table: Int64,
        table_stride: Int64,
        vocab: Int64,
        prev: Int64,
        prev_stride: Int64,
        enorm_w: Int64,
        hnorm_w: Int64,
        out: Int64,
        out_stride: Int64,
        eps: Float32,
        tokens: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            ids,
            table,
            table_stride,
            vocab,
            prev,
            prev_stride,
            enorm_w,
            hnorm_w,
            out,
            out_stride,
            eps,
        ).launch(
            grid=[tokens, 1, 1], block=[self.threads, 1, 1], stream=stream, use_pdl=True
        )

    @cute.jit
    def wait_for_inputs(self):
        core.pdl_wait()

    @cute.jit
    def inv_rms(self, words, smem, tid: Int32, eps: Float32):
        # per-lane chain over the 8 elements, in element order, from +0
        s = Float32(0.0)
        for j in cutlass.range_constexpr(4):
            lo = ptx.bf16_low(words[j])
            hi = ptx.bf16_high(words[j])
            s = core.fma_rn(lo, lo, s)
            s = core.fma_rn(hi, hi, s)
        for k in cutlass.range_constexpr(5):
            s = core.add_rn(s, cute.arch.shuffle_sync_bfly(s, 16 >> k))
        warp = tid // 32
        lane = tid % 32
        # every lane writes its warp's (identical) sum, as the stock CTA norm does
        smem[warp] = s
        cute.arch.sync_threads()
        if warp == 0:
            t = Float32(0.0)
            if lane < self.warps:
                t = smem[lane]
            for k in cutlass.range_constexpr(5):
                t = core.add_rn(t, cute.arch.shuffle_sync_bfly(t, 16 >> k))
            mean = core.div_rn(t, Float32(float(self.hidden)))
            smem[32] = core.rsqrt_approx(core.add_rn(mean, eps), self.rsqrt_ftz)
        cute.arch.sync_threads()
        return smem[32]

    @cute.jit
    def apply(self, words, wwords, r):
        res = []
        for j in cutlass.range_constexpr(4):
            a = core.mul_rn(
                core.mul_rn(ptx.bf16_low(words[j]), r), ptx.bf16_low(wwords[j])
            )
            b = core.mul_rn(
                core.mul_rn(ptx.bf16_high(words[j]), r), ptx.bf16_high(wwords[j])
            )
            res.append(core.cvt_rn_bf16x2_f32(a, b))
        return res

    @cute.kernel
    def kernel(
        self,
        ids: Int64,
        table: Int64,
        table_stride: Int64,
        vocab: Int64,
        prev: Int64,
        prev_stride: Int64,
        enorm_w: Int64,
        hnorm_w: Int64,
        out: Int64,
        out_stride: Int64,
        eps: Float32,
    ):
        t, _, _ = cute.arch.thread_idx()
        b, _, _ = cute.arch.block_idx()
        tid = Int32(t)
        tok = Int64(b)
        smem = ptx.static_smem(Float32, 33)
        col = Int64(tid) * Int64(16)

        self.wait_for_inputs()
        tid_ = sys_ops.ld_u64_generic(ids + tok * Int64(8))
        idv = Int64(tid_)
        sys_ops.trap_if((idv < Int64(0)) | (idv >= vocab))
        e = core.ld_global_v4(table + idv * table_stride * Int64(2) + col)
        ew = core.ld_global_v4(enorm_w + col)
        r = self.inv_rms(e, smem, tid, eps)
        o = self.apply(e, ew, r)
        orow = out + tok * out_stride * Int64(2)
        core.st_global_v4(orow + col, o)

        h = core.ld_global_v4(prev + tok * prev_stride * Int64(2) + col)
        hw = core.ld_global_v4(hnorm_w + col)
        r2 = self.inv_rms(h, smem, tid, eps)
        o2 = self.apply(h, hw, r2)
        core.st_global_v4(orow + Int64(self.hidden * 2) + col, o2)
        core.pdl_launch_dependents()
