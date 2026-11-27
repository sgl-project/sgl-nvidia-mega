# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MLA attention prologue for decode / verify rows (rows <= 8): everything between the
fused q_a / kv_a projection and the sparse attention core, in one cluster launch, with
an engine's own rounding points.

For each row ``m < rows`` (8 local heads, q_lora 2048, kv_lora 512, nope 192, rope 64):

    q_a = bf16(RMSNorm(q_c[m]) * q_norm_w)          (optionally written to qc_out)
    q = bf16(q_a @ W_qb^T)                          [8, 256] = nope 192 | rope 64
    mqa_q[m, h, :512] = e4m3(bf16(q_nope[h] @ W_UK[h]^T) / q_scale)
    mqa_q[m, h, 512:] = e4m3(RoPE(q_rope[h]) / q_scale)
    MLA row of slot[m] = e4m3(RMSNorm(kv_c[m]) * kv_norm_w / k_scale)
                         | e4m3(RoPE(k_pe[m]) / k_scale)
    full (indexer) layers: index-K row = e4m3(RoPE(LayerNorm(idx_k[m])) / s) with
    s from max(amax, 1e-4) / 448 (stored with the row); top-k buffer rows set to -1.

Work split: 9 clusters of 8 CTAs (256 threads). Clusters 0..7 = heads: CTA ``r`` owns
q_b rows ``[32 r, 32 r + 32)`` of its head and absorbed columns ``[64 r, 64 r + 64)``;
its packed q_b (128 KiB) and W_UK^T (24 KiB) slices are bulk-copied before the PDL wait.
Ranks 0..5 publish their q_nope rows in shared memory; after a cluster barrier every
rank gathers the head's 192 nope values over distributed shared memory. Cluster 8: CTA 0
writes the MLA rows, CTA 1 the index-K rows, CTAs 2..7 the top-k fill.

Option fields (compile time): ``max_rows`` (8 or 16), ``overlap16`` (16 rows only: the
head CTA issues every load of a phase before the stores of that phase, so the rows'
latencies overlap instead of adding up; per row the same operations in the same order,
so the same outputs), ``overlap8`` (8 rows only: the
head CTA issues each phase's loads before that phase's stores and keeps its global
q_pe stores out of the hand-off; same outputs), ``kv_split`` (16 rows only: the
kv cluster's rows 8..15 run on CTAs 2 (MLA rows) and 3 (index-K rows) beside rows 0..7 on
CTAs 0 and 1, instead of a second pass of the same warps; the top-k fill moves to CTAs
4..7; per row the same operations, so the same outputs), with ``overlap16`` and the
strided sums: ``row_norm`` (warp w loads q rows w / w + 8 in the strided-sum lane
layout, row w first, sums and normalizes each from its registers and stores only the
normalized rows: no shared-memory round trip for the sums, no CTA-wide normalize phase;
each head CTA stores 1 KiB of ``qc_out`` from its XS slots after the q_b products; per
row the same operations, so the same outputs), with ``push``: ``early_arm`` (the QN
barrier is armed by a CTA-scope arrive before the weight bulk copies are issued, instead
of a cluster-scope release arrive after them, which waited ~1.5 us for the copies
before the PDL wait; same outputs), with ``overlap8`` and ``push``:
``early_arm8`` (the 8-row head CTA's QN barrier is armed by a CTA-scope arrive before the
weight bulk copies are issued, instead of a cluster-scope release arrive after them,
which waits for the copies before the PDL wait; same outputs), ``numerics``
(``PrologueNumerics``: the sum orders, rounding points and scale rule, each named by
the arithmetic it selects), ``rope_cache_fp32`` and ``index_rope_cache_fp32`` (cos /
sin caches FP32, else BF16; layout [positions, 64] = cos | sin),
``index_scale_ue8m0x4`` (with the power-of-two index-K scale: store it as four equal
UE8M0 bytes instead of an FP32). The layer kind (indexer or not) and ``qc_out`` (null
= not written) are runtime.

Ordering: dependents are triggered at entry (``trigger_at_end``: after each CTA's last
output store and fence instead, so no prologue CTA is still resident when the dependents'
CTAs are placed; outputs are the same); only constant weights / norm parameters /
scales are read before ``griddepcontrol.wait``, and every CTA executes the wait on every
path; ``fence.proxy.async.global`` follows the wait and each thread's last output store.
The q_nope hand-off publishes with ``barrier.cluster.arrive`` (release) / ``wait``
(acquire); a second arrive after the remote reads and the final wait keep every CTA's
shared memory alive while its peers read it.
"""

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32, Int64, Uint32, Uint64

from ..primitives import ptx_core as ptx
from ..primitives import ptx_sys

@ptx.ptx_op()
def _sts_v4_if(addr, words, valid):
    """``st.shared.v4.u32`` where ``valid`` holds (predicated, no branch)."""
    code = "{ .reg .pred p; setp.ne.b32 p, $5, 0; @p st.shared.v4.u32 [$0], {$1, $2, $3, $4}; }"
    return code, "r,r,r,r,r,r", (addr, *words, Int32(Boolean(valid)))


@ptx.ptx_op()
def _stg_v4_if(addr, words, valid):
    """``st.global.v4.u32`` where ``valid`` holds (predicated, no branch)."""
    code = "{ .reg .pred p; setp.ne.b32 p, $5, 0; @p st.global.v4.u32 [$0], {$1, $2, $3, $4}; }"
    return code, "l,r,r,r,r,r", (addr, *words, Int32(Boolean(valid)))


@ptx.ptx_arith((Float32, Float32))
def _mul_f32x2(a0, a1, b0, b1):
    """``mul.rn.f32x2``: (a0 b0, a1 b1), each rounded to nearest as ``mul.rn.f32``."""
    code = (
        "{ .reg .b64 a, b, d; mov.b64 a, {$2, $3}; mov.b64 b, {$4, $5}; "
        "mul.rn.f32x2 d, a, b; mov.b64 {$0, $1}, d; }"
    )
    return code, "=f,=f,f,f,f,f", (a0, a1, b0, b1)


# probe builds (benchmarks only): per CTA, PROBE_MARKS x {clock64, globaltimer} u64 slots
PROBE_MARKS = 16


@ptx.ptx_op(Uint64)
def _clock64():
    return "mov.u64 $0, %clock64;", "=l", ()


@ptx.ptx_op(Float32)
def _lds_f32_if(addr, valid):
    """``ld.shared.f32`` where ``valid`` holds, else 0 (predicated, no branch)."""
    code = (
        "{ .reg .pred p; setp.ne.b32 p, $2, 0; mov.f32 $0, 0f00000000; "
        "@p ld.shared.f32 $0, [$1]; }"
    )
    return code, "=f,r,r", (addr, Int32(Boolean(valid)))


@ptx.ptx_op((Uint32, Uint32, Uint32, Uint32))
def _lds_v4_if(addr, valid):
    """``ld.shared.v4.u32`` where ``valid`` holds, else zeros (predicated, no branch)."""
    code = (
        "{ .reg .pred p; setp.ne.b32 p, $5, 0; "
        "mov.b32 $0, 0; mov.b32 $1, 0; mov.b32 $2, 0; mov.b32 $3, 0; "
        "@p ld.shared.v4.u32 {$0, $1, $2, $3}, [$4]; }"
    )
    return code, "=r,=r,=r,=r,r,r", (addr, Int32(Boolean(valid)))


@ptx.ptx_op(Float32)
def _lds_f32_pred(addr, valid):
    """``ld.shared.f32`` where ``valid`` holds, else 0 (predicated, no branch)."""
    code = (
        "{ .reg .pred p; setp.ne.b32 p, $2, 0; mov.f32 $0, 0f00000000; "
        "@p ld.shared.f32 $0, [$1]; }"
    )
    return code, "=f,r,r", (addr, Int32(Boolean(valid)))


@ptx.ptx_op((Uint32, Uint32, Uint32, Uint32))
def _lds_v4_pred(addr, valid):
    """``ld.shared.v4.u32`` where ``valid`` holds, else zeros (predicated)."""
    code = (
        "{ .reg .pred p; setp.ne.b32 p, $5, 0; "
        "mov.b32 $0, 0; mov.b32 $1, 0; mov.b32 $2, 0; mov.b32 $3, 0; "
        "@p ld.shared.v4.u32 {$0, $1, $2, $3}, [$4]; }"
    )
    return code, "=r,=r,=r,=r,r,r", (addr, Int32(Boolean(valid)))


HEADS = 8
Q_LORA = 2048
KV_LORA = 512
ROPE = 64
NOPE = 192
Q_HEAD = NOPE + ROPE  # 256
CLUSTER = 8
THREADS = 256
MQA_DIM = KV_LORA + ROPE  # 576 bytes per head per row
INDEX_DIM = 128
GRID = (HEADS + 1) * CLUSTER
NORM_ORDERS = ("strided_8x256", "lane_contiguous")
INDEX_MEAN_SUMS = ("sequential", "pairwise")
INDEX_ROPE_ODD = ("round_x1_cos", "round_x0_sin")
INDEX_SCALES = ("fp32_amax", "ue8m0")
Q_ROPES = ("one_product_rounded", "both_products_rounded")


@dataclass(frozen=True)
class PrologueNumerics:
    """The prologue's numeric options, each named by the arithmetic it selects.

    ``norm_order`` (q_a and kv_a RMSNorm sums of squares): ``"strided_8x256"``: one
    warp per row; lane l adds the separately rounded squares of elements 8 l + 256 b +
    v (q_a: b, v = 0..7 in (b, v) order; kv_a: b = 0, 1) to 0, then a butterfly xor 1,
    2, 4, 8, 16. ``"lane_contiguous"``: each thread accumulates the squares of its
    contiguous elements with FMA from 0 (kv_a: lane l owns 16 l .. 16 l + 15), then a
    butterfly xor 16, 8, 4, 2, 1; the 8 warp sums of a q_a row are added in warp order.
    ``norm_weight_plus_zero``: the norm weights enter as ``w + 0``.
    ``kv_norm_bf16``: the kv_a norm output is rounded to BF16 before the FP8
    quantization. ``skip_slot0``: cache slot 0 (a padding slot) is never written; else
    only negative slots are skipped.
    ``index_mean_sum`` (index-K LayerNorm, a lane's 4 elements): ``"sequential"``:
    (((0 + x0) + x1) + x2) + x3; ``"pairwise"``: (x0 + x1) + (x2 + x3).
    ``index_rope_odd`` (index-K RoPE, odd dims e0 sin + e1 cos): which product is
    rounded before the fused other one. ``index_k_bf16``: the RoPE output is rounded
    to BF16 before the scale. ``index_scale``: ``"fp32_amax"``: s = max(amax, 1e-4) /
    448, values x (1 / s) clipped to +-448; ``"ue8m0"``: s rounded up to a power of
    two, values x / s.
    ``q_rope`` (q_pe RoPE): ``"one_product_rounded"``: x0 cos - x1 sin with x1 sin
    rounded and x1 cos + x0 sin with x0 sin rounded, the other product fused;
    ``"both_products_rounded"``: both products rounded, then the difference / sum.
    """

    norm_order: str
    norm_weight_plus_zero: bool
    kv_norm_bf16: bool
    skip_slot0: bool
    index_mean_sum: str
    index_rope_odd: str
    index_k_bf16: bool
    index_scale: str
    q_rope: str

    def __post_init__(self):
        for value, allowed in (
            (self.norm_order, NORM_ORDERS),
            (self.index_mean_sum, INDEX_MEAN_SUMS),
            (self.index_rope_odd, INDEX_ROPE_ODD),
            (self.index_scale, INDEX_SCALES),
            (self.q_rope, Q_ROPES),
        ):
            if value not in allowed:
                raise ValueError(f"{value!r} is not one of {allowed}")


# shared memory of a head CTA (bytes)
WQB_BYTES = 2 * (Q_LORA // 32) * 1024  # 2 row tiles x 64 k-steps x 1 KiB
WUK_BYTES = 4 * (NOPE // 32) * 1024  # 4 row tiles x 6 k-steps x 1 KiB
XS_STRIDE = Q_LORA * 2 + 64
QN_STRIDE = NOPE * 2 + 64


# overlap8 partials, one FP32 row per (warp or absorb half, token): q_b 32 output rows + 4
# pad, absorb 64 columns + 4 pad, so a reader warp's 32 lanes read consecutive words and
# a writer warp's lanes (8 t + g) hit distinct banks (the C-fragment layout of head_cta
# costs 8 wavefronts per reduction load)
P8_RED_QB_ROW = 32 + 4
P8_RED_UK_ROW = 64 + 4
P8_RED_BYTES = 8 * 8 * P8_RED_QB_ROW * 4  # 9,216 (the absorb partials: 4,352)


def layout_overlap8(off):
    """``layout(8)`` with the larger overlap8 partials region (``red``); the regions
    after it move up."""
    off = dict(off)
    grow = P8_RED_BYTES - (off["ssq"] - off["red"])
    for k in ("ssq", "mbar", "total"):
        off[k] += grow
    return off


# overlap16 partials, one FP32 row per (warp or half, token tile, token): q_b 32 output
# rows + 4 pad, absorb 64 columns + 4 pad, so a reader warp's 32 lanes read consecutive
# words and a writer warp's lanes (8 t + g) hit distinct banks
RED_QB_ROW = 32 + 4
RED_UK_ROW = 64 + 4
RED16O_BYTES = 8 * 2 * 8 * RED_QB_ROW * 4  # 18,432 (the absorb partials: 8,704)


def layout(max_rows, overlap16=False, push=False):
    off = {"wqb": 0, "wuk": WQB_BYTES}
    off["xs"] = off["wuk"] + WUK_BYTES
    if max_rows == 8:
        off["pub"] = off["xs"] + max_rows * XS_STRIDE
        off["qn"] = off["pub"] + max_rows * 32 * 2
        off["red"] = off["qn"] + max_rows * QN_STRIDE
        off["ssq"] = off["red"] + 8 * 2 * 32 * 16
    elif push:
        # 16 rows, pushed hand-off: peers write the q_nope rows (QN) as soon as their
        # own partials are summed, so QN cannot alias this CTA's token rows; the
        # partials still do (no published rows: no PUB region)
        off["red"] = off["xs"]
        off["qn"] = off["xs"] + max_rows * XS_STRIDE
        off["pub"] = off["qn"]
        off["ssq"] = off["qn"] + max_rows * QN_STRIDE
    else:
        # 16 rows: the partials and the gathered q_nope rows alias the token rows,
        # which are dead once the q_b products are done (a CTA barrier separates them)
        off["red"] = off["xs"]
        off["qn"] = off["red"] + (RED16O_BYTES if overlap16 else 8 * 2 * 2 * 32 * 16)
        off["pub"] = off["xs"] + max_rows * XS_STRIDE
        off["ssq"] = off["pub"] + max_rows * 32 * 2
    off["mbar"] = off["ssq"] + 8 * max_rows * 4
    off["total"] = off["mbar"] + 16
    return off


class MlaPrologue:
    """See the module docstring. Call arguments (device addresses; row strides in
    elements; cache strides in bytes): ``q_c``, ``kv_c``, ``k_pe``, ``idx_k`` with their
    row strides (``idx_k`` unused on shared layers), ``positions``, ``slot_mapping``,
    ``idx_slot_mapping`` (int64), ``rows``, ``has_indexer``, packed ``w_qb`` / ``w_uk``
    (``packing.pack_q_b`` / ``pack_w_uk_t``), ``q_norm_w`` / ``kv_norm_w`` (BF16),
    ``q_eps`` / ``kv_eps``, ``rope_cache`` and its row stride, ``q_scale`` / ``k_scale``
    (FP32 device scalars), ``ik_w`` / ``ik_b`` (FP32 [128]), ``ik_eps``,
    ``index_rope_cache`` and its row stride, ``mqa_q`` and its row stride, the MLA cache
    (base, block stride, entry stride, block size), the index-K cache (base, block
    stride, block size), ``topk`` (int32) with its row stride and
    width, ``qc_out`` (or 0) and its row stride, then the stream."""

    def __init__(
        self,
        max_rows: int,
        numerics: PrologueNumerics,
        rope_cache_fp32: bool = True,
        index_rope_cache_fp32: bool = True,
        index_scale_ue8m0x4: bool = False,
        trigger_at_end: bool = False,
        overlap16: bool = False,
        overlap8: bool = False,
        kv_split: bool = False,
        probe: bool = False,
        push: bool = False,
        f32x2: bool = False,
        row_norm: bool = False,
        early_arm: bool = False,
        early_arm8: bool = False,
        heads: int = HEADS,
    ):
        if max_rows not in (8, 16, 32):
            raise ValueError(f"max_rows must be 8, 16 or 32, got {max_rows}")
        if heads not in (8, 16):
            raise ValueError(f"heads must be 8 or 16, got {heads}")
        self.heads = heads
        self.grid = (heads + 1) * CLUSTER
        if overlap8 and max_rows != 8:
            raise ValueError("overlap8 applies to max_rows 8 only")
        if overlap16 and max_rows not in (16, 32):
            raise ValueError("overlap16 applies to max_rows 16 or 32 only")
        if kv_split and max_rows not in (16, 32):
            raise ValueError("kv_split applies to max_rows 16 or 32 only")
        if max_rows == 32 and not (overlap16 and push and row_norm and early_arm):
            raise ValueError(
                "max_rows 32 requires overlap16, push, row_norm and early_arm"
            )
        self.max_rows = max_rows
        self.overlap16 = overlap16
        self.overlap8 = overlap8
        self.kv_split = kv_split
        # benchmark builds: phase timestamps to the top-k pointer (shared layers only)
        self.probe = probe
        if (push or f32x2) and not (overlap8 or overlap16):
            raise ValueError("push / f32x2 apply to the overlap8 / overlap16 forms")
        # push: the q_nope hand-off as st.async pushes into every rank's QN rows,
        # completing on its mbarrier (no release barrier, no remote gather)
        self.push = push
        # f32x2: the q_a normalize branch-free with packed f32x2 multiplies
        self.f32x2 = f32x2
        if early_arm8 and not (overlap8 and push):
            raise ValueError("early_arm8 applies to overlap8 with push")
        # early_arm8: the 8-row QN barrier armed (CTA scope) before the weight bulk copies
        self.early_arm8 = early_arm8
        self.num = numerics
        self.rope_cache_fp32 = rope_cache_fp32
        self.index_rope_cache_fp32 = index_rope_cache_fp32
        if index_scale_ue8m0x4 and numerics.index_scale != "ue8m0":
            raise ValueError(
                "index_scale_ue8m0x4 applies to the power-of-two scale only"
            )
        self.index_scale_ue8m0x4 = index_scale_ue8m0x4
        self.trigger_at_end = trigger_at_end
        if row_norm and not (overlap16 and numerics.norm_order == "strided_8x256"):
            raise ValueError("row_norm applies to overlap16 with the strided q_a sums")
        if early_arm and not (overlap16 and push):
            raise ValueError("early_arm applies to overlap16 with push")
        # row_norm: the q rows summed and normalized by the warp that loads them
        self.row_norm = row_norm
        # early_arm: the QN barrier armed (CTA scope) before the weight bulk copies
        self.early_arm = early_arm
        self.off = layout(min(max_rows, 16), overlap16, push)
        if overlap8:
            self.off = layout_overlap8(self.off)

    @cute.jit
    def __call__(
        self,
        q_c: Int64,
        q_c_stride: Int64,
        kv_c: Int64,
        kv_c_stride: Int64,
        k_pe: Int64,
        k_pe_stride: Int64,
        idx_k: Int64,
        idx_k_stride: Int64,
        positions: Int64,
        slot_mapping: Int64,
        idx_slot_mapping: Int64,
        rows: Int32,
        has_indexer: Int32,
        w_qb: Int64,
        w_uk: Int64,
        q_norm_w: Int64,
        kv_norm_w: Int64,
        q_eps: Float32,
        kv_eps: Float32,
        rope_cache: Int64,
        rope_stride: Int64,
        q_scale: Int64,
        k_scale: Int64,
        ik_w: Int64,
        ik_b: Int64,
        ik_eps: Float32,
        index_rope_cache: Int64,
        index_rope_stride: Int64,
        mqa_q: Int64,
        mqa_stride: Int64,
        mla_cache: Int64,
        mla_block_stride: Int64,
        mla_entry_stride: Int64,
        mla_block_size: Int64,
        idx_cache: Int64,
        idx_block_stride: Int64,
        idx_block_size: Int64,
        topk: Int64,
        topk_stride: Int64,
        topk_width: Int32,
        qc_out: Int64,
        qc_out_stride: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(
            q_c,
            q_c_stride,
            kv_c,
            kv_c_stride,
            k_pe,
            k_pe_stride,
            idx_k,
            idx_k_stride,
            positions,
            slot_mapping,
            idx_slot_mapping,
            rows,
            has_indexer,
            w_qb,
            w_uk,
            q_norm_w,
            kv_norm_w,
            q_eps,
            kv_eps,
            rope_cache,
            rope_stride,
            q_scale,
            k_scale,
            ik_w,
            ik_b,
            ik_eps,
            index_rope_cache,
            index_rope_stride,
            mqa_q,
            mqa_stride,
            mla_cache,
            mla_block_stride,
            mla_entry_stride,
            mla_block_size,
            idx_cache,
            idx_block_stride,
            idx_block_size,
            topk,
            topk_stride,
            topk_width,
            qc_out,
            qc_out_stride,
        ).launch(
            grid=[self.grid, 1, 1],
            block=[THREADS, 1, 1],
            cluster=[CLUSTER, 1, 1],
            stream=stream,
            use_pdl=True,
            max_number_threads=[THREADS, 1, 1],
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        q_c: Int64,
        q_c_stride: Int64,
        kv_c: Int64,
        kv_c_stride: Int64,
        k_pe: Int64,
        k_pe_stride: Int64,
        idx_k: Int64,
        idx_k_stride: Int64,
        positions: Int64,
        slot_mapping: Int64,
        idx_slot_mapping: Int64,
        rows: Int32,
        has_indexer: Int32,
        w_qb: Int64,
        w_uk: Int64,
        q_norm_w: Int64,
        kv_norm_w: Int64,
        q_eps: Float32,
        kv_eps: Float32,
        rope_cache: Int64,
        rope_stride: Int64,
        q_scale: Int64,
        k_scale: Int64,
        ik_w: Int64,
        ik_b: Int64,
        ik_eps: Float32,
        index_rope_cache: Int64,
        index_rope_stride: Int64,
        mqa_q: Int64,
        mqa_stride: Int64,
        mla_cache: Int64,
        mla_block_stride: Int64,
        mla_entry_stride: Int64,
        mla_block_size: Int64,
        idx_cache: Int64,
        idx_block_stride: Int64,
        idx_block_size: Int64,
        topk: Int64,
        topk_stride: Int64,
        topk_width: Int32,
        qc_out: Int64,
        qc_out_stride: Int64,
    ):
        tid_, _, _ = cute.arch.thread_idx()
        cta_, _, _ = cute.arch.block_idx()
        tid = Int32(tid_)
        cta = Int32(cta_)
        smem = cutlass.utils.SmemAllocator()
        buf = smem.allocate_tensor(
            cutlass.Uint8, cute.make_layout(self.off["total"]), byte_alignment=1024
        )
        base = ptx.shared_address(buf.iterator)

        if cutlass.const_expr(not self.trigger_at_end):
            ptx.pdl_launch_dependents()
        cluster_id = cta // CLUSTER
        r = cta % CLUSTER
        if cutlass.const_expr(self.probe):
            self._probe_base = topk
        if cluster_id < self.heads:
            if cutlass.const_expr(self.max_rows == 32):
                self.head_cta32o(
                    base,
                    cluster_id,
                    r,
                    tid,
                    q_c,
                    q_c_stride,
                    positions,
                    rows,
                    w_qb,
                    w_uk,
                    q_norm_w,
                    q_eps,
                    rope_cache,
                    rope_stride,
                    q_scale,
                    mqa_q,
                    mqa_stride,
                    qc_out,
                    qc_out_stride,
                )
            elif cutlass.const_expr(self.overlap16):
                self.head_cta16o(
                    base,
                    cluster_id,
                    r,
                    tid,
                    q_c,
                    q_c_stride,
                    positions,
                    rows,
                    w_qb,
                    w_uk,
                    q_norm_w,
                    q_eps,
                    rope_cache,
                    rope_stride,
                    q_scale,
                    mqa_q,
                    mqa_stride,
                    qc_out,
                    qc_out_stride,
                )
            elif cutlass.const_expr(self.max_rows == 16):
                self.head_cta16(
                    base,
                    cluster_id,
                    r,
                    tid,
                    q_c,
                    q_c_stride,
                    positions,
                    rows,
                    w_qb,
                    w_uk,
                    q_norm_w,
                    q_eps,
                    rope_cache,
                    rope_stride,
                    q_scale,
                    mqa_q,
                    mqa_stride,
                    qc_out,
                    qc_out_stride,
                )
            elif cutlass.const_expr(self.overlap8):
                self.head_cta8o(
                    base,
                    cluster_id,
                    r,
                    tid,
                    q_c,
                    q_c_stride,
                    positions,
                    rows,
                    w_qb,
                    w_uk,
                    q_norm_w,
                    q_eps,
                    rope_cache,
                    rope_stride,
                    q_scale,
                    mqa_q,
                    mqa_stride,
                    qc_out,
                    qc_out_stride,
                )
            else:
                self.head_cta(
                    base,
                    cluster_id,
                    r,
                    tid,
                    q_c,
                    q_c_stride,
                    positions,
                    rows,
                    w_qb,
                    w_uk,
                    q_norm_w,
                    q_eps,
                    rope_cache,
                    rope_stride,
                    q_scale,
                    mqa_q,
                    mqa_stride,
                    qc_out,
                    qc_out_stride,
                )
        else:
            self.mark(0, tid)
            self.kv_cta(
                r,
                tid,
                kv_c,
                kv_c_stride,
                k_pe,
                k_pe_stride,
                idx_k,
                idx_k_stride,
                positions,
                slot_mapping,
                idx_slot_mapping,
                rows,
                has_indexer,
                kv_norm_w,
                kv_eps,
                rope_cache,
                rope_stride,
                k_scale,
                ik_w,
                ik_b,
                ik_eps,
                index_rope_cache,
                index_rope_stride,
                mla_cache,
                mla_block_stride,
                mla_entry_stride,
                mla_block_size,
                idx_cache,
                idx_block_stride,
                idx_block_size,
                topk,
                topk_stride,
                topk_width,
            )
            self.mark(13, tid)
            self.release_outputs()
        if cutlass.const_expr(self.trigger_at_end):
            ptx.pdl_launch_dependents()

    # ------------------------------------------------------------------ head clusters
    @cute.jit
    def head_cta(
        self,
        base: Int32,
        h: Int32,
        r: Int32,
        tid: Int32,
        q_c: Int64,
        q_c_stride: Int64,
        positions: Int64,
        rows: Int32,
        w_qb: Int64,
        w_uk: Int64,
        q_norm_w: Int64,
        q_eps: Float32,
        rope_cache: Int64,
        rope_stride: Int64,
        q_scale: Int64,
        mqa_q: Int64,
        mqa_stride: Int64,
        qc_out: Int64,
        qc_out_stride: Int64,
    ):
        off = self.off
        max_rows = self.max_rows
        lane = tid % 32
        warp = tid // 32
        bar = base + off["mbar"]
        s_xs = base + off["xs"]
        s_red = base + off["red"]
        s_ssq = base + off["ssq"]
        s_pub = base + off["pub"]
        s_qn = base + off["qn"]
        # constant weights: q_b slice (two 64 KiB halves) + W_UK^T slice (24 KiB),
        # before the PDL wait
        if tid == 0:
            ptx.mbarrier_init(bar, 1)
            ptx.fence_mbarrier_init_release_cluster()
            ptx.fence_proxy_async_shared_cta()
            ptx.mbarrier_arrive_expect_tx(bar, WQB_BYTES + WUK_BYTES)
            src_qb = w_qb + Int64(h * 16 + 2 * r) * ((Q_LORA // 32) * 1024)
            src_uk = w_uk + Int64(h * 32 + 4 * r) * ((NOPE // 32) * 1024)
            for part in cutlass.range_constexpr(2):
                ptx.bulk_copy_g2s(
                    base + off["wqb"] + part * (WQB_BYTES // 2),
                    src_qb + part * (WQB_BYTES // 2),
                    WQB_BYTES // 2,
                    bar,
                )
            ptx.bulk_copy_g2s(base + off["wuk"], src_uk, WUK_BYTES, bar)
        qw = ptx.ld_global_nc_v4(q_norm_w + Int64(tid) * 16)
        inv_qs = ptx.rcp_rn(ptx.ld_global_nc_f32(q_scale))
        ptx.sync_threads()

        self.wait_q_inputs()

        # every predecessor-data load first (their latencies overlap)
        qv = []
        for n in cutlass.range_constexpr(max_rows):
            qv.append(
                self.load16(
                    q_c + (Int64(n) * q_c_stride + Int64(tid * 8)) * 2, n < rows
                )
            )
        rope_c = Float32(0.0)
        rope_s = Float32(0.0)
        if (r >= NOPE // 32) & (tid < 32 * rows):
            pos = ptx.ld_global_s64(positions + Int64(tid // 32) * 8)
            pr = ((r - NOPE // 32) * 32 + lane) // 2
            rope_c = self.rope_value(
                rope_cache, rope_stride, pos, pr, self.rope_cache_fp32
            )
            rope_s = self.rope_value(
                rope_cache, rope_stride, pos, 32 + pr, self.rope_cache_fp32
            )
        if cutlass.const_expr(self.num.norm_order == "strided_8x256"):
            self.q_norm_sums_strided(s_xs, s_ssq, qv, warp, lane, tid, rows)
        else:
            self.q_norm_sums_contiguous(s_ssq, qv, warp, lane, rows)
        ptx.sync_threads()
        wv = []
        for e in cutlass.range_constexpr(4):
            lo, hi = ptx.bf16x2_to_f32x2(qw[e])
            if cutlass.const_expr(self.num.norm_weight_plus_zero):
                lo = ptx.add_rn(lo, Float32(0.0))
                hi = ptx.add_rn(hi, Float32(0.0))
            wv.append(lo)
            wv.append(hi)
        for n in cutlass.range_constexpr(max_rows):
            if n < rows:
                if cutlass.const_expr(self.num.norm_order == "strided_8x256"):
                    s = ptx.ld_shared_f32(s_ssq + n * 4)
                else:
                    s = Float32(0.0)
                    for w in cutlass.range_constexpr(8):
                        s = ptx.add_rn(
                            s, ptx.ld_shared_f32(s_ssq + (w * max_rows + n) * 4)
                        )
                rr = ptx.rsqrt_approx(ptx.fma_rn(s, Float32(1.0 / Q_LORA), q_eps))
                words = []
                for e in cutlass.range_constexpr(4):
                    x0, x1 = ptx.bf16x2_to_f32x2(qv[n][e])
                    words.append(
                        ptx.cvt_rn_bf16x2_f32(
                            ptx.mul_rn(ptx.mul_rn(x0, rr), wv[2 * e]),
                            ptx.mul_rn(ptx.mul_rn(x1, rr), wv[2 * e + 1]),
                        )
                    )
                packed = (words[0], words[1], words[2], words[3])
                ptx.st_shared_v4(s_xs + n * XS_STRIDE + tid * 16, packed)
                if (qc_out != Int64(0)) & (h == 0) & (r == 0):
                    ptx.st_global_v4(
                        qc_out + (Int64(n) * qc_out_stride + Int64(tid * 8)) * 2, packed
                    )
        ptx.sync_threads()

        self.wait_weights(bar)

        g = lane // 4
        t = lane % 4
        # q_b MMA (tokens on N = 8): warp w covers k-steps [8 w, 8 w + 8) for both
        # 16-row tiles
        zero = Float32(0.0)
        c = [(zero, zero, zero, zero), (zero, zero, zero, zero)]
        for i in cutlass.range_constexpr(8):
            s_ = warp * 8 + i
            bq = self.lds16(s_xs + g * XS_STRIDE + (s_ * 32 + 8 * t) * 2, g < rows)
            for j in cutlass.range_constexpr(2):
                blk = base + off["wqb"] + (j * (Q_LORA // 32) + s_) * 1024
                lo = ptx.ld_shared_v4(blk + lane * 16)
                hi = ptx.ld_shared_v4(blk + 512 + lane * 16)
                c1 = ptx.mma_m16n8k16_bf16(
                    (lo[0], hi[0], lo[1], hi[1]), (bq[0], bq[1]), c[j]
                )
                c[j] = ptx.mma_m16n8k16_bf16(
                    (lo[2], hi[2], lo[3], hi[3]), (bq[2], bq[3]), c1
                )
        for j in cutlass.range_constexpr(2):
            ptx.st_shared_v4(
                s_red + ((warp * 2 + j) * 32 + lane) * 16,
                (
                    ptx.f32_bits(c[j][0]),
                    ptx.f32_bits(c[j][1]),
                    ptx.f32_bits(c[j][2]),
                    ptx.f32_bits(c[j][3]),
                ),
            )
        self.sync_partials()
        if tid < 32 * rows:
            rho = tid % 32
            n = tid // 32
            jt = rho // 16
            rr_ = rho % 16
            ln = (rr_ % 8) * 4 + n // 2
            reg = (rr_ // 8) * 2 + n % 2
            v = Float32(0.0)
            for w in cutlass.range_constexpr(8):
                v = ptx.add_rn(
                    v,
                    ptx.ld_shared_f32(s_red + ((w * 2 + jt) * 32 + ln) * 16 + reg * 4),
                )
            qb = ptx.cvt_rn_bf16_f32(v)
            if r < NOPE // 32:
                ptx.st_shared_b16(s_pub + (n * 32 + rho) * 2, qb)
            else:
                d = (r - NOPE // 32) * 32 + rho
                own = ptx.bf16_to_f32(qb)
                other = cute.arch.shuffle_sync_bfly(own, 1)
                o = Float32(0.0)
                if d % 2 == 0:
                    o = self.q_rope_even(own, other, rope_c, rope_s)
                else:
                    o = self.q_rope_odd(other, own, rope_c, rope_s)
                o = ptx.mul_rn(o, inv_qs)
                dst = mqa_q + Int64(n) * mqa_stride + Int64(h * MQA_DIM + KV_LORA + d)
                ptx.st_global_u8(
                    dst,
                    cutlass.Uint16(
                        Uint32(ptx.cvt_rn_satfinite_e4m3x2_f32(o, Float32(0.0)))
                        & Uint32(0xFF)
                    ),
                )
        # q_nope hand-off: ranks 0..5 published 32 rows each
        self.handoff(base, tid, rows)
        self.done_reading()
        ptx.sync_threads()
        # W_UK absorb: rows l in [64 r, 64 r + 64) (4 tiles), K = 192 (6 k-steps); warp
        # -> (tile w % 4, half w // 4)
        tile = warp % 4
        half = warp // 4
        cw = (zero, zero, zero, zero)
        for i in cutlass.range_constexpr(3):
            s_ = half * 3 + i
            bq = self.lds16(s_qn + g * QN_STRIDE + (s_ * 32 + 8 * t) * 2, g < rows)
            blk = base + off["wuk"] + (tile * (NOPE // 32) + s_) * 1024
            lo = ptx.ld_shared_v4(blk + lane * 16)
            hi = ptx.ld_shared_v4(blk + 512 + lane * 16)
            c1 = ptx.mma_m16n8k16_bf16((lo[0], hi[0], lo[1], hi[1]), (bq[0], bq[1]), cw)
            cw = ptx.mma_m16n8k16_bf16((lo[2], hi[2], lo[3], hi[3]), (bq[2], bq[3]), c1)
        ptx.st_shared_v4(
            s_red + (warp * 32 + lane) * 16,
            (
                ptx.f32_bits(cw[0]),
                ptx.f32_bits(cw[1]),
                ptx.f32_bits(cw[2]),
                ptx.f32_bits(cw[3]),
            ),
        )
        self.sync_partials()
        for k in cutlass.range_constexpr((64 * max_rows + THREADS - 1) // THREADS):
            idx = tid + k * THREADS
            if idx < 64 * rows:
                rho = idx % 64
                n = idx // 64
                tl = rho // 16
                rr_ = rho % 16
                ln = (rr_ % 8) * 4 + n // 2
                reg = (rr_ // 8) * 2 + n % 2
                v = ptx.add_rn(
                    ptx.ld_shared_f32(s_red + (tl * 32 + ln) * 16 + reg * 4),
                    ptx.ld_shared_f32(s_red + ((tl + 4) * 32 + ln) * 16 + reg * 4),
                )
                qb = ptx.bf16_to_f32(ptx.cvt_rn_bf16_f32(v))
                dst = mqa_q + Int64(n) * mqa_stride + Int64(h * MQA_DIM + r * 64 + rho)
                ptx.st_global_u8(
                    dst,
                    cutlass.Uint16(
                        Uint32(
                            ptx.cvt_rn_satfinite_e4m3x2_f32(
                                ptx.mul_rn(qb, inv_qs), Float32(0.0)
                            )
                        )
                        & Uint32(0xFF)
                    ),
                )
        self.release_outputs()
        self.exit_wait()

    @cute.jit
    def head_cta8o(
        self,
        base: Int32,
        h: Int32,
        r: Int32,
        tid: Int32,
        q_c: Int64,
        q_c_stride: Int64,
        positions: Int64,
        rows: Int32,
        w_qb: Int64,
        w_uk: Int64,
        q_norm_w: Int64,
        q_eps: Float32,
        rope_cache: Int64,
        rope_stride: Int64,
        q_scale: Int64,
        mqa_q: Int64,
        mqa_stride: Int64,
        qc_out: Int64,
        qc_out_stride: Int64,
    ):
        """``head_cta`` (same operations per row, same order, same rounding) in an issue
        order where a phase's loads precede its stores (memory operations keep their
        program order, and a load after a store to the same space waits for it): the
        RoPE position before the q rows, its cos / sin after them; every row's sum of
        squares before any normalized row is stored; the q_b reduction's 8 partials
        before its sum; both absorb slots' partials before their stores; token-tile
        operands loaded predicated (no lane-divergent branch); the q_b and absorb
        partials as one padded FP32 row per (warp or half, token) (``P8_RED_*``), free
        of the fragment layout's 8-way bank conflicts; the RoPE ranks' FP8 q_pe bytes
        kept in registers and stored with the absorbed columns at the end, so no global
        store is outstanding at the hand-off's cluster barrier release."""
        off = self.off
        max_rows = self.max_rows
        lane = tid % 32
        warp = tid // 32
        bar = base + off["mbar"]
        s_xs = base + off["xs"]
        s_red = base + off["red"]
        s_ssq = base + off["ssq"]
        s_pub = base + off["pub"]
        s_qn = base + off["qn"]
        strided = self.num.norm_order == "strided_8x256"
        # constant weights: q_b slice (two 64 KiB halves) + W_UK^T slice (24 KiB),
        # before the PDL wait
        qbar = bar + 8  # push: this CTA's QN barrier
        if tid == 0:
            ptx.mbarrier_init(bar, 1)
            if cutlass.const_expr(self.push):
                ptx.mbarrier_init(qbar, 1)
            ptx.fence_mbarrier_init_release_cluster()
            ptx.fence_proxy_async_shared_cta()
            if cutlass.const_expr(self.early_arm8):
                # every q_nope rank's rows of this CTA: 6 x rows x 64 bytes. A CTA-scope
                # arrive before the copies: the waiters acquire the peers' pushes (their
                # complete_tx), not this thread's prior writes, so no cluster-scope release
                # (which waits for the copies in flight); the co-start barrier still
                # orders the init before any push
                ptx.mbarrier_arrive_expect_tx(qbar, (NOPE // 32) * 64 * rows)
            ptx.mbarrier_arrive_expect_tx(bar, WQB_BYTES + WUK_BYTES)
            src_qb = w_qb + Int64(h * 16 + 2 * r) * ((Q_LORA // 32) * 1024)
            src_uk = w_uk + Int64(h * 32 + 4 * r) * ((NOPE // 32) * 1024)
            for part in cutlass.range_constexpr(2):
                ptx.bulk_copy_g2s(
                    base + off["wqb"] + part * (WQB_BYTES // 2),
                    src_qb + part * (WQB_BYTES // 2),
                    WQB_BYTES // 2,
                    bar,
                )
            ptx.bulk_copy_g2s(base + off["wuk"], src_uk, WUK_BYTES, bar)
            if cutlass.const_expr(self.push and not self.early_arm8):
                # every q_nope rank's rows of this CTA: 6 x rows x 64 bytes
                ptx.mbarrier_arrive_expect_tx(
                    qbar, (NOPE // 32) * 64 * rows, ".release.cluster"
                )
        qw = ptx.ld_global_nc_v4(q_norm_w + Int64(tid) * 16)
        inv_qs = ptx.rcp_rn(ptx.ld_global_nc_f32(q_scale))
        ptx.sync_threads()
        if cutlass.const_expr(self.push):
            # co-start: every peer's QN barrier exists before this CTA pushes
            ptx.cluster_arrive_relaxed()

        self.wait_q_inputs()

        # RoPE CTAs: the position first, then the q rows, then the cos / sin loads that
        # need the position
        is_rope = r >= NOPE // 32
        rope_ok = is_rope & (tid < 32 * rows)
        pos = self.p8_ldg_s64_if(positions + Int64(tid // 32) * 8, rope_ok)
        qv = []
        for n in cutlass.range_constexpr(max_rows):
            if cutlass.const_expr(self.f32x2):
                # rows predicated, not branched
                qv.append(
                    ptx.ld_global_v4_if(
                        q_c + (Int64(n) * q_c_stride + Int64(tid * 8)) * 2, n < rows
                    )
                )
            else:
                qv.append(
                    self.load16(
                        q_c + (Int64(n) * q_c_stride + Int64(tid * 8)) * 2, n < rows
                    )
                )
        pr = ((r - NOPE // 32) * 32 + lane) // 2
        rope_c, rope_s = self.p8_rope_at(rope_ok, pos, pr, rope_cache, rope_stride)
        if cutlass.const_expr(strided):
            self.q_norm_sums_strided(s_xs, s_ssq, qv, warp, lane, tid, rows)
        else:
            self.q_norm_sums_contiguous(s_ssq, qv, warp, lane, rows)
        ptx.sync_threads()
        wv = []
        for e in cutlass.range_constexpr(4):
            lo, hi = ptx.bf16x2_to_f32x2(qw[e])
            if cutlass.const_expr(self.num.norm_weight_plus_zero):
                lo = ptx.add_rn(lo, Float32(0.0))
                hi = ptx.add_rn(hi, Float32(0.0))
            wv.append(lo)
            wv.append(hi)
        # every row's sum of squares, then the normalized rows (raw rows in registers)
        ss = []
        for n in cutlass.range_constexpr(max_rows):
            if cutlass.const_expr(strided):
                ss.append(_lds_f32_pred(s_ssq + n * 4, n < rows))
            else:
                part = []
                for w in cutlass.range_constexpr(8):
                    part.append(
                        _lds_f32_pred(s_ssq + (w * max_rows + n) * 4, n < rows)
                    )
                ss.append(part)
        qc_on = (qc_out != Int64(0)) & (h == 0) & (r == 0)
        for n in cutlass.range_constexpr(max_rows if self.f32x2 else 0):
            # f32x2: every row computed (an unused row is never stored), stores
            # predicated: no per-row branch; the multiplies packed
            # (own names: the DSL types a name over the whole function, not per trace)
            snf = Float32(0.0)
            if cutlass.const_expr(strided):
                snf = ss[n]
            else:
                for w in cutlass.range_constexpr(8):
                    snf = ptx.add_rn(snf, ss[n][w])
            rrf = ptx.rsqrt_approx(ptx.fma_rn(snf, Float32(1.0 / Q_LORA), q_eps))
            pkf = self.norm_words(qv[n], rrf, wv)
            _sts_v4_if(s_xs + n * XS_STRIDE + tid * 16, pkf, n < rows)
            _stg_v4_if(
                qc_out + (Int64(n) * qc_out_stride + Int64(tid * 8)) * 2,
                pkf,
                qc_on & (n < rows),
            )
        for n in cutlass.range_constexpr(0 if self.f32x2 else max_rows):
            if n < rows:
                sn = Float32(0.0)
                if cutlass.const_expr(strided):
                    sn = ss[n]
                else:
                    for w in cutlass.range_constexpr(8):
                        sn = ptx.add_rn(sn, ss[n][w])
                rr = ptx.rsqrt_approx(ptx.fma_rn(sn, Float32(1.0 / Q_LORA), q_eps))
                words = []
                for e in cutlass.range_constexpr(4):
                    x0, x1 = ptx.bf16x2_to_f32x2(qv[n][e])
                    words.append(
                        ptx.cvt_rn_bf16x2_f32(
                            ptx.mul_rn(ptx.mul_rn(x0, rr), wv[2 * e]),
                            ptx.mul_rn(ptx.mul_rn(x1, rr), wv[2 * e + 1]),
                        )
                    )
                packed = (words[0], words[1], words[2], words[3])
                ptx.st_shared_v4(s_xs + n * XS_STRIDE + tid * 16, packed)
                if (qc_out != Int64(0)) & (h == 0) & (r == 0):
                    ptx.st_global_v4(
                        qc_out + (Int64(n) * qc_out_stride + Int64(tid * 8)) * 2, packed
                    )
        ptx.sync_threads()

        self.wait_weights(bar)

        g = lane // 4
        t = lane % 4
        # q_b MMA (tokens on N = 8): warp w covers k-steps [8 w, 8 w + 8) for both
        # 16-row tiles
        zero = Float32(0.0)
        c = [(zero, zero, zero, zero), (zero, zero, zero, zero)]
        for i in cutlass.range_constexpr(8):
            s_ = warp * 8 + i
            bq = _lds_v4_pred(s_xs + g * XS_STRIDE + (s_ * 32 + 8 * t) * 2, g < rows)
            for j in cutlass.range_constexpr(2):
                blk = base + off["wqb"] + (j * (Q_LORA // 32) + s_) * 1024
                lo = ptx.ld_shared_v4(blk + lane * 16)
                hi = ptx.ld_shared_v4(blk + 512 + lane * 16)
                c1 = ptx.mma_m16n8k16_bf16(
                    (lo[0], hi[0], lo[1], hi[1]), (bq[0], bq[1]), c[j]
                )
                c[j] = ptx.mma_m16n8k16_bf16(
                    (lo[2], hi[2], lo[3], hi[3]), (bq[2], bq[3]), c1
                )
        # each accumulator to its (warp, token, q_b row) word: the C fragment holds
        # rows g / g + 8 of tile j, tokens 2 t / 2 t + 1
        for j in cutlass.range_constexpr(2):
            for e in cutlass.range_constexpr(4):
                ptx.st_shared_f32(
                    s_red
                    + (
                        (warp * 8 + 2 * t + e % 2) * P8_RED_QB_ROW
                        + j * 16
                        + g
                        + 8 * (e // 2)
                    )
                    * 4,
                    c[j][e],
                )
        self.sync_partials()
        # the 8 partials of this thread's (token, q_b row), then their sum in warp order
        rho = tid % 32
        n = tid // 32
        red_ok = tid < 32 * rows
        parts = []
        for w in cutlass.range_constexpr(8):
            parts.append(
                _lds_f32_pred(s_red + ((w * 8 + n) * P8_RED_QB_ROW + rho) * 4, red_ok)
            )
        v = Float32(0.0)
        for w in cutlass.range_constexpr(8):
            v = ptx.add_rn(v, parts[w])
        qb = ptx.cvt_rn_bf16_f32(v)
        if cutlass.const_expr(self.push):
            ptx.cluster_wait_acquire()  # co-start
            self.push_qnope(base, qbar, r, n, lane, qb, (r < NOPE // 32) & red_ok)
        else:
            if r < NOPE // 32:
                if red_ok:
                    ptx.st_shared_b16(s_pub + (n * 32 + rho) * 2, qb)
        # the RoPE ranks keep their FP8 q_pe byte (stored at the end)
        rope_b = self.p8_q_rope_byte(
            is_rope, (r - NOPE // 32) * 32 + rho, qb, rope_c, rope_s, inv_qs
        )
        # q_nope hand-off: ranks 0..5 published 32 rows each
        if cutlass.const_expr(self.push):
            ptx.mbarrier_wait_parity(qbar, 0, ".acquire.cluster")
            ptx.cluster_arrive_relaxed()  # exit: waited before this CTA leaves
        else:
            self.handoff(base, tid, rows)
            self.done_reading()
            ptx.sync_threads()
        # W_UK absorb: rows l in [64 r, 64 r + 64) (4 tiles), K = 192 (6 k-steps); warp
        # -> (tile w % 4, half w // 4)
        tile = warp % 4
        half = warp // 4
        cw = (zero, zero, zero, zero)
        for i in cutlass.range_constexpr(3):
            s_ = half * 3 + i
            bq = _lds_v4_pred(s_qn + g * QN_STRIDE + (s_ * 32 + 8 * t) * 2, g < rows)
            blk = base + off["wuk"] + (tile * (NOPE // 32) + s_) * 1024
            lo = ptx.ld_shared_v4(blk + lane * 16)
            hi = ptx.ld_shared_v4(blk + 512 + lane * 16)
            c1 = ptx.mma_m16n8k16_bf16((lo[0], hi[0], lo[1], hi[1]), (bq[0], bq[1]), cw)
            cw = ptx.mma_m16n8k16_bf16((lo[2], hi[2], lo[3], hi[3]), (bq[2], bq[3]), c1)
        # each accumulator to its (half, token, absorbed column) word
        for e in cutlass.range_constexpr(4):
            ptx.st_shared_f32(
                s_red
                + (
                    (half * 8 + 2 * t + e % 2) * P8_RED_UK_ROW
                    + tile * 16
                    + g
                    + 8 * (e // 2)
                )
                * 4,
                cw[e],
            )
        self.sync_partials()
        nk = (64 * max_rows + THREADS - 1) // THREADS
        pa = []
        pb = []
        for k in cutlass.range_constexpr(nk):
            idx = tid + k * THREADS
            col = idx % 64
            row = idx // 64
            # half 0 (k-steps 0..2) + half 1 (k-steps 3..5), as head_cta
            pa.append(
                _lds_f32_pred(
                    s_red + ((0 * 8 + row) * P8_RED_UK_ROW + col) * 4, idx < 64 * rows
                )
            )
            pb.append(
                _lds_f32_pred(
                    s_red + ((1 * 8 + row) * P8_RED_UK_ROW + col) * 4, idx < 64 * rows
                )
            )
        for k in cutlass.range_constexpr(nk):
            idx = tid + k * THREADS
            if idx < 64 * rows:
                col = idx % 64
                row = idx // 64
                qf = ptx.bf16_to_f32(ptx.cvt_rn_bf16_f32(ptx.add_rn(pa[k], pb[k])))
                dst = mqa_q + Int64(row) * mqa_stride + Int64(h * MQA_DIM + r * 64 + col)
                ptx.st_global_u8(
                    dst,
                    cutlass.Uint16(
                        Uint32(
                            ptx.cvt_rn_satfinite_e4m3x2_f32(
                                ptx.mul_rn(qf, inv_qs), Float32(0.0)
                            )
                        )
                        & Uint32(0xFF)
                    ),
                )
        # the RoPE ranks' q_pe bytes (computed before the hand-off)
        if is_rope:
            if red_ok:
                d = (r - NOPE // 32) * 32 + rho
                ptx.st_global_u8(
                    mqa_q + Int64(n) * mqa_stride + Int64(h * MQA_DIM + KV_LORA + d),
                    rope_b,
                )
        self.release_outputs()
        self.exit_wait()

    @cute.jit
    def p8_q_rope_byte(
        self,
        valid,
        d: Int32,
        qb,
        c: Float32,
        s: Float32,
        inv_qs: Float32,
    ):
        """The FP8 q_pe byte of dim d (``head_cta``'s RoPE arithmetic on the BF16 q_b
        output ``qb``; the pair partner is the neighbouring lane), 0 unless ``valid``
        (CTA-uniform: the shuffle sees the whole warp)."""
        byte = cutlass.Uint16(0)
        if valid:
            own = ptx.bf16_to_f32(qb)
            other = cute.arch.shuffle_sync_bfly(own, 1)
            o = Float32(0.0)
            if d % 2 == 0:
                o = self.q_rope_even(own, other, c, s)
            else:
                o = self.q_rope_odd(other, own, c, s)
            o = ptx.mul_rn(o, inv_qs)
            byte = cutlass.Uint16(
                Uint32(ptx.cvt_rn_satfinite_e4m3x2_f32(o, Float32(0.0))) & Uint32(0xFF)
            )
        return byte

    @cute.jit
    def p8_ldg_s64_if(self, addr: Int64, valid):
        v = Int64(0)
        if valid:
            v = ptx.ld_global_s64(addr)
        return v

    @cute.jit
    def p8_rope_at(self, valid, pos: Int64, pair: Int32, cache: Int64, stride: Int64):
        """The cos / sin of one q_pe RoPE pair with the position already loaded."""
        c = Float32(0.0)
        s = Float32(0.0)
        if valid:
            c = self.rope_value(cache, stride, pos, pair, self.rope_cache_fp32)
            s = self.rope_value(cache, stride, pos, 32 + pair, self.rope_cache_fp32)
        return c, s

    @cute.jit
    def head_cta16(
        self,
        base: Int32,
        h: Int32,
        r: Int32,
        tid: Int32,
        q_c: Int64,
        q_c_stride: Int64,
        positions: Int64,
        rows: Int32,
        w_qb: Int64,
        w_uk: Int64,
        q_norm_w: Int64,
        q_eps: Float32,
        rope_cache: Int64,
        rope_stride: Int64,
        q_scale: Int64,
        mqa_q: Int64,
        mqa_stride: Int64,
        qc_out: Int64,
        qc_out_stride: Int64,
    ):
        """The head CTA at up to 16 rows. Per row, every operation and its order are
        the 8-row form's (rows are independent columns of the tensor-core products),
        so each row equals that row of an 8-row call. Tokens use two 8-wide N tiles;
        the raw and normalized rows live in the XS slots (re-read, not registers); the
        partials and the gathered q_nope rows reuse the XS region after the q_b
        products (a CTA barrier separates them)."""
        off = self.off
        max_rows = self.max_rows
        lane = tid % 32
        warp = tid // 32
        bar = base + off["mbar"]
        s_xs = base + off["xs"]
        s_red = base + off["red"]
        s_ssq = base + off["ssq"]
        s_pub = base + off["pub"]
        s_qn = base + off["qn"]
        strided = self.num.norm_order == "strided_8x256"
        if tid == 0:
            ptx.mbarrier_init(bar, 1)
            ptx.fence_mbarrier_init_release_cluster()
            ptx.fence_proxy_async_shared_cta()
            ptx.mbarrier_arrive_expect_tx(bar, WQB_BYTES + WUK_BYTES)
            src_qb = w_qb + Int64(h * 16 + 2 * r) * ((Q_LORA // 32) * 1024)
            src_uk = w_uk + Int64(h * 32 + 4 * r) * ((NOPE // 32) * 1024)
            for part in cutlass.range_constexpr(2):
                ptx.bulk_copy_g2s(
                    base + off["wqb"] + part * (WQB_BYTES // 2),
                    src_qb + part * (WQB_BYTES // 2),
                    WQB_BYTES // 2,
                    bar,
                )
            ptx.bulk_copy_g2s(base + off["wuk"], src_uk, WUK_BYTES, bar)
        qw = ptx.ld_global_nc_v4(q_norm_w + Int64(tid) * 16)
        inv_qs = ptx.rcp_rn(ptx.ld_global_nc_f32(q_scale))
        ptx.sync_threads()

        self.wait_q_inputs()

        qv = []
        for n in cutlass.range_constexpr(max_rows):
            qv.append(
                self.load16(
                    q_c + (Int64(n) * q_c_stride + Int64(tid * 8)) * 2, n < rows
                )
            )
        # RoPE CTAs: two (token, pair) slots per thread (token tid / 32 and + 8)
        rope_c = [Float32(0.0), Float32(0.0)]
        rope_s = [Float32(0.0), Float32(0.0)]
        for it in cutlass.range_constexpr(2):
            rope_c[it], rope_s[it] = self.rope_pair(
                (r >= NOPE // 32) & (tid + THREADS * it < 32 * rows),
                positions,
                (tid + THREADS * it) // 32,
                ((r - NOPE // 32) * 32 + lane) // 2,
                rope_cache,
                rope_stride,
            )
        for n in cutlass.range_constexpr(max_rows):
            if n < rows:
                ptx.st_shared_v4(s_xs + n * XS_STRIDE + tid * 16, qv[n])
        if cutlass.const_expr(strided):
            ptx.sync_threads()
            for grp in cutlass.range_constexpr(2):
                n = warp + 8 * grp
                if n < rows:
                    s = Float32(0.0)
                    row = s_xs + n * XS_STRIDE
                    for b in cutlass.range_constexpr(8):
                        u = ptx.ld_shared_v4(row + (lane + 32 * b) * 16)
                        for e in cutlass.range_constexpr(4):
                            lo, hi = ptx.bf16x2_to_f32x2(u[e])
                            s = ptx.add_rn(s, ptx.mul_rn(lo, lo))
                            s = ptx.add_rn(s, ptx.mul_rn(hi, hi))
                    for k in cutlass.range_constexpr(5):
                        s = ptx.add_rn(s, cute.arch.shuffle_sync_bfly(s, 1 << k))
                    if lane == 0:
                        ptx.st_shared_f32(s_ssq + n * 4, s)
        else:
            self.q_norm_sums_contiguous(s_ssq, qv, warp, lane, rows)
        ptx.sync_threads()
        wv = []
        for e in cutlass.range_constexpr(4):
            lo, hi = ptx.bf16x2_to_f32x2(qw[e])
            if cutlass.const_expr(self.num.norm_weight_plus_zero):
                lo = ptx.add_rn(lo, Float32(0.0))
                hi = ptx.add_rn(hi, Float32(0.0))
            wv.append(lo)
            wv.append(hi)
        for n in cutlass.range_constexpr(max_rows):
            if n < rows:
                if cutlass.const_expr(strided):
                    s = ptx.ld_shared_f32(s_ssq + n * 4)
                else:
                    s = Float32(0.0)
                    for w in cutlass.range_constexpr(8):
                        s = ptx.add_rn(
                            s, ptx.ld_shared_f32(s_ssq + (w * max_rows + n) * 4)
                        )
                rr = ptx.rsqrt_approx(ptx.fma_rn(s, Float32(1.0 / Q_LORA), q_eps))
                raw = ptx.ld_shared_v4(s_xs + n * XS_STRIDE + tid * 16)
                words = []
                for e in cutlass.range_constexpr(4):
                    x0, x1 = ptx.bf16x2_to_f32x2(raw[e])
                    words.append(
                        ptx.cvt_rn_bf16x2_f32(
                            ptx.mul_rn(ptx.mul_rn(x0, rr), wv[2 * e]),
                            ptx.mul_rn(ptx.mul_rn(x1, rr), wv[2 * e + 1]),
                        )
                    )
                packed = (words[0], words[1], words[2], words[3])
                ptx.st_shared_v4(s_xs + n * XS_STRIDE + tid * 16, packed)
                if (qc_out != Int64(0)) & (h == 0) & (r == 0):
                    ptx.st_global_v4(
                        qc_out + (Int64(n) * qc_out_stride + Int64(tid * 8)) * 2, packed
                    )
        ptx.sync_threads()

        self.wait_weights(bar)

        g = lane // 4
        t = lane % 4
        zero = Float32(0.0)
        z4 = (zero, zero, zero, zero)
        c = [z4, z4, z4, z4]  # [j * 2 + token tile]
        for i in cutlass.range_constexpr(8):
            s_ = warp * 8 + i
            b0 = self.lds16(s_xs + g * XS_STRIDE + (s_ * 32 + 8 * t) * 2, g < rows)
            b1 = self.lds16(
                s_xs + (g + 8) * XS_STRIDE + (s_ * 32 + 8 * t) * 2, g + 8 < rows
            )
            for j in cutlass.range_constexpr(2):
                blk = base + off["wqb"] + (j * (Q_LORA // 32) + s_) * 1024
                lo = ptx.ld_shared_v4(blk + lane * 16)
                hi = ptx.ld_shared_v4(blk + 512 + lane * 16)
                for tt in cutlass.range_constexpr(2):
                    bq = b0 if tt == 0 else b1
                    c1 = ptx.mma_m16n8k16_bf16(
                        (lo[0], hi[0], lo[1], hi[1]), (bq[0], bq[1]), c[j * 2 + tt]
                    )
                    c[j * 2 + tt] = ptx.mma_m16n8k16_bf16(
                        (lo[2], hi[2], lo[3], hi[3]), (bq[2], bq[3]), c1
                    )
        # the partials reuse the token rows: every warp must be done reading them
        self.rows_dead()
        for j in cutlass.range_constexpr(2):
            for tt in cutlass.range_constexpr(2):
                cc = c[j * 2 + tt]
                ptx.st_shared_v4(
                    s_red + (((warp * 2 + j) * 2 + tt) * 32 + lane) * 16,
                    (
                        ptx.f32_bits(cc[0]),
                        ptx.f32_bits(cc[1]),
                        ptx.f32_bits(cc[2]),
                        ptx.f32_bits(cc[3]),
                    ),
                )
        self.sync_partials()
        for it in cutlass.range_constexpr(2):
            idx = tid + THREADS * it
            if idx < 32 * rows:
                rho = idx % 32
                n = idx // 32
                tt = n // 8
                nn = n % 8
                jt = rho // 16
                rr_ = rho % 16
                ln = (rr_ % 8) * 4 + nn // 2
                reg = (rr_ // 8) * 2 + nn % 2
                v = Float32(0.0)
                for w in cutlass.range_constexpr(8):
                    v = ptx.add_rn(
                        v,
                        ptx.ld_shared_f32(
                            s_red + (((w * 2 + jt) * 2 + tt) * 32 + ln) * 16 + reg * 4
                        ),
                    )
                qb = ptx.cvt_rn_bf16_f32(v)
                if r < NOPE // 32:
                    ptx.st_shared_b16(s_pub + (n * 32 + rho) * 2, qb)
                else:
                    d = (r - NOPE // 32) * 32 + rho
                    own = ptx.bf16_to_f32(qb)
                    other = cute.arch.shuffle_sync_bfly(own, 1)
                    o = Float32(0.0)
                    if d % 2 == 0:
                        o = self.q_rope_even(own, other, rope_c[it], rope_s[it])
                    else:
                        o = self.q_rope_odd(other, own, rope_c[it], rope_s[it])
                    o = ptx.mul_rn(o, inv_qs)
                    dst = (
                        mqa_q + Int64(n) * mqa_stride + Int64(h * MQA_DIM + KV_LORA + d)
                    )
                    ptx.st_global_u8(
                        dst,
                        cutlass.Uint16(
                            Uint32(ptx.cvt_rn_satfinite_e4m3x2_f32(o, Float32(0.0)))
                            & Uint32(0xFF)
                        ),
                    )
        self.publish()
        for it in cutlass.range_constexpr(2):
            self.gather(base, tid + THREADS * it, rows)
        self.done_reading()
        ptx.sync_threads()
        tile = warp % 4
        half = warp // 4
        cw = [z4, z4]
        for i in cutlass.range_constexpr(3):
            s_ = half * 3 + i
            b0 = self.lds16(s_qn + g * QN_STRIDE + (s_ * 32 + 8 * t) * 2, g < rows)
            b1 = self.lds16(
                s_qn + (g + 8) * QN_STRIDE + (s_ * 32 + 8 * t) * 2, g + 8 < rows
            )
            blk = base + off["wuk"] + (tile * (NOPE // 32) + s_) * 1024
            lo = ptx.ld_shared_v4(blk + lane * 16)
            hi = ptx.ld_shared_v4(blk + 512 + lane * 16)
            for tt in cutlass.range_constexpr(2):
                bq = b0 if tt == 0 else b1
                c1 = ptx.mma_m16n8k16_bf16(
                    (lo[0], hi[0], lo[1], hi[1]), (bq[0], bq[1]), cw[tt]
                )
                cw[tt] = ptx.mma_m16n8k16_bf16(
                    (lo[2], hi[2], lo[3], hi[3]), (bq[2], bq[3]), c1
                )
        for tt in cutlass.range_constexpr(2):
            ptx.st_shared_v4(
                s_red + ((warp * 2 + tt) * 32 + lane) * 16,
                (
                    ptx.f32_bits(cw[tt][0]),
                    ptx.f32_bits(cw[tt][1]),
                    ptx.f32_bits(cw[tt][2]),
                    ptx.f32_bits(cw[tt][3]),
                ),
            )
        self.sync_partials()
        for k in cutlass.range_constexpr((64 * max_rows + THREADS - 1) // THREADS):
            idx = tid + k * THREADS
            if idx < 64 * rows:
                rho = idx % 64
                n = idx // 64
                tt = n // 8
                nn = n % 8
                tl = rho // 16
                rr_ = rho % 16
                ln = (rr_ % 8) * 4 + nn // 2
                reg = (rr_ // 8) * 2 + nn % 2
                v = ptx.add_rn(
                    ptx.ld_shared_f32(s_red + ((tl * 2 + tt) * 32 + ln) * 16 + reg * 4),
                    ptx.ld_shared_f32(
                        s_red + (((tl + 4) * 2 + tt) * 32 + ln) * 16 + reg * 4
                    ),
                )
                qb = ptx.bf16_to_f32(ptx.cvt_rn_bf16_f32(v))
                dst = mqa_q + Int64(n) * mqa_stride + Int64(h * MQA_DIM + r * 64 + rho)
                ptx.st_global_u8(
                    dst,
                    cutlass.Uint16(
                        Uint32(
                            ptx.cvt_rn_satfinite_e4m3x2_f32(
                                ptx.mul_rn(qb, inv_qs), Float32(0.0)
                            )
                        )
                        & Uint32(0xFF)
                    ),
                )
        self.release_outputs()
        self.exit_wait()

    @cute.jit
    def head_cta16o(
        self,
        base: Int32,
        h: Int32,
        r: Int32,
        tid: Int32,
        q_c: Int64,
        q_c_stride: Int64,
        positions: Int64,
        rows: Int32,
        w_qb: Int64,
        w_uk: Int64,
        q_norm_w: Int64,
        q_eps: Float32,
        rope_cache: Int64,
        rope_stride: Int64,
        q_scale: Int64,
        mqa_q: Int64,
        mqa_stride: Int64,
        qc_out: Int64,
        qc_out_stride: Int64,
    ):
        """``head_cta16`` (same operations per row, same order, same rounding) in an
        issue order where a phase's loads all precede its stores: the memory operations
        are ordered as written, and a load after a store to the same space waits for it,
        so ``head_cta16`` paid the second row slot's latencies after the first's. Here:
        the RoPE positions before the q rows (one L2 round trip for both cos / sin
        slots); each warp's two strided sum rows interleaved; every row's sum and raw row
        before any normalized row is stored; both reduction slots' partials, both gather
        slots' remote loads and all four absorb slots' partials before their stores;
        token-tile operands loaded predicated (no lane-divergent branch); the q_b and
        absorb partials stored as one padded FP32 row per (warp or half, token tile,
        token) (``RED_QB_ROW`` / ``RED_UK_ROW``), so the reduction's reads are free of
        the 8-way bank conflicts of head_cta16's fragment layout."""
        off = self.off
        max_rows = self.max_rows
        lane = tid % 32
        warp = tid // 32
        bar = base + off["mbar"]
        s_xs = base + off["xs"]
        s_red = base + off["red"]
        s_ssq = base + off["ssq"]
        s_pub = base + off["pub"]
        s_qn = base + off["qn"]
        strided = self.num.norm_order == "strided_8x256"
        rn = self.row_norm
        self.mark(0, tid)
        qbar = bar + 8  # push: this CTA's QN barrier
        if tid == 0:
            ptx.mbarrier_init(bar, 1)
            if cutlass.const_expr(self.push):
                ptx.mbarrier_init(qbar, 1)
            ptx.fence_mbarrier_init_release_cluster()
            ptx.fence_proxy_async_shared_cta()
            if cutlass.const_expr(self.early_arm):
                # every q_nope rank's rows of this CTA: 6 x rows x 64 bytes. A CTA-scope
                # arrive before the copies: the waiters acquire the peers' pushes, not this
                # thread's prior writes, so no cluster-scope release (which waited for the
                # copies in flight); the co-start barrier still orders the init
                ptx.mbarrier_arrive_expect_tx(qbar, (NOPE // 32) * 64 * rows)
            ptx.mbarrier_arrive_expect_tx(bar, WQB_BYTES + WUK_BYTES)
            src_qb = w_qb + Int64(h * 16 + 2 * r) * ((Q_LORA // 32) * 1024)
            src_uk = w_uk + Int64(h * 32 + 4 * r) * ((NOPE // 32) * 1024)
            for part in cutlass.range_constexpr(2):
                ptx.bulk_copy_g2s(
                    base + off["wqb"] + part * (WQB_BYTES // 2),
                    src_qb + part * (WQB_BYTES // 2),
                    WQB_BYTES // 2,
                    bar,
                )
            ptx.bulk_copy_g2s(base + off["wuk"], src_uk, WUK_BYTES, bar)
            if cutlass.const_expr(self.push and not self.early_arm):
                # every q_nope rank's rows of this CTA: 6 x rows x 64 bytes
                ptx.mbarrier_arrive_expect_tx(
                    qbar, (NOPE // 32) * 64 * rows, ".release.cluster"
                )
        self.mark(15, tid)  # probe: thread 0's barrier / copy block done
        qw = ptx.ld_global_nc_v4(q_norm_w + Int64(tid) * 16)
        nw = []
        if cutlass.const_expr(rn):
            # row_norm: the norm weights of this lane's chunks l + 32 b (b = 0..7) of a
            # row, "+ 0" applied in FP32 and repacked (exact: BF16 values)
            for b in cutlass.range_constexpr(8):
                q_w = ptx.ld_global_nc_v4(q_norm_w + Int64(lane + 32 * b) * 16)
                words = []
                for e in cutlass.range_constexpr(4):
                    lo, hi = ptx.bf16x2_to_f32x2(q_w[e])
                    if cutlass.const_expr(self.num.norm_weight_plus_zero):
                        lo = ptx.add_rn(lo, Float32(0.0))
                        hi = ptx.add_rn(hi, Float32(0.0))
                    words.append(ptx.cvt_rn_bf16x2_f32(lo, hi))
                nw.append(words)
        inv_qs = ptx.rcp_rn(ptx.ld_global_nc_f32(q_scale))
        ptx.sync_threads()
        if cutlass.const_expr(self.push):
            # co-start: every peer's QN barrier exists before this CTA pushes
            ptx.cluster_arrive_relaxed()
        self.mark(1, tid)

        self.wait_q_inputs()
        self.mark(2, tid)

        # RoPE CTAs: the two (token, pair) slots' positions first, then the q rows, then
        # the cos / sin loads that need the positions
        is_rope = r >= NOPE // 32
        pair = ((r - NOPE // 32) * 32 + lane) // 2
        pv = []
        for it in cutlass.range_constexpr(2):
            pv.append(
                self.ldg_s64_if(
                    positions + Int64((tid + THREADS * it) // 32) * 8,
                    is_rope & (tid + THREADS * it < 32 * rows),
                )
            )
        # rows are predicated, not branched, so the scheduler can interleave them
        qv = []
        if cutlass.const_expr(rn):
            # row_norm: warp w loads rows w and w + 8, lane l chunks l + 32 b (the lane
            # layout of the strided sums) into qv[8 half + b], row w first (it lands
            # first, so its sums and normalize overlap row w + 8's loads)
            for hf in cutlass.range_constexpr(2):
                for b in cutlass.range_constexpr(8):
                    qv.append(
                        ptx.ld_global_v4_if(
                            q_c
                            + (
                                Int64(warp + 8 * hf) * q_c_stride
                                + Int64((lane + 32 * b) * 8)
                            )
                            * 2,
                            warp + 8 * hf < rows,
                        )
                    )
        else:
            for n in cutlass.range_constexpr(max_rows):
                qv.append(
                    ptx.ld_global_v4_if(
                        q_c + (Int64(n) * q_c_stride + Int64(tid * 8)) * 2, n < rows
                    )
                )
        rope_c = [Float32(0.0), Float32(0.0)]
        rope_s = [Float32(0.0), Float32(0.0)]
        for it in cutlass.range_constexpr(2):
            rope_c[it], rope_s[it] = self.rope_at(
                is_rope & (tid + THREADS * it < 32 * rows),
                pv[it],
                pair,
                rope_cache,
                rope_stride,
            )
        if cutlass.const_expr(rn):
            self.row_norm16(s_xs, qv, nw, tid, warp, lane, rows, q_eps)
        else:
            for n in cutlass.range_constexpr(max_rows):
                _sts_v4_if(s_xs + n * XS_STRIDE + tid * 16, qv[n], n < rows)
            self.mark(3, tid)
            if cutlass.const_expr(strided):
                ptx.sync_threads()
                if warp + 8 < rows:
                    self.q_norm_pair_strided(s_xs, s_ssq, warp, lane)
                else:
                    if warp < rows:
                        self.q_norm_row_strided(s_xs, s_ssq, warp, lane)
            else:
                self.q_norm_sums_contiguous(s_ssq, qv, warp, lane, rows)
        if cutlass.const_expr(not rn):
            ptx.sync_threads()
        self.mark(4, tid)
        wv = []
        for e in cutlass.range_constexpr(4):
            lo, hi = ptx.bf16x2_to_f32x2(qw[e])
            if cutlass.const_expr(self.num.norm_weight_plus_zero):
                lo = ptx.add_rn(lo, Float32(0.0))
                hi = ptx.add_rn(hi, Float32(0.0))
            wv.append(lo)
            wv.append(hi)
        if cutlass.const_expr(strided and not rn):
            # every row's sum of squares and raw row, then the normalized rows
            ss = []
            raw = []
            for n in cutlass.range_constexpr(max_rows):
                ss.append(_lds_f32_if(s_ssq + n * 4, n < rows))
            for n in cutlass.range_constexpr(max_rows):
                raw.append(_lds_v4_if(s_xs + n * XS_STRIDE + tid * 16, n < rows))
            # every row computed (an unused row's arithmetic is never stored), the
            # stores predicated: no per-row branch, so the rows' latencies overlap
            qc_on = (qc_out != Int64(0)) & (h == 0) & (r == 0)
            for n in cutlass.range_constexpr(max_rows):
                rr = ptx.rsqrt_approx(ptx.fma_rn(ss[n], Float32(1.0 / Q_LORA), q_eps))
                packed = self.norm_words(raw[n], rr, wv)
                _sts_v4_if(s_xs + n * XS_STRIDE + tid * 16, packed, n < rows)
                _stg_v4_if(
                    qc_out + (Int64(n) * qc_out_stride + Int64(tid * 8)) * 2,
                    packed,
                    qc_on & (n < rows),
                )
        elif cutlass.const_expr(not rn):
            for n in cutlass.range_constexpr(max_rows):
                if n < rows:
                    s = Float32(0.0)
                    for w in cutlass.range_constexpr(8):
                        s = ptx.add_rn(
                            s, ptx.ld_shared_f32(s_ssq + (w * max_rows + n) * 4)
                        )
                    rr = ptx.rsqrt_approx(ptx.fma_rn(s, Float32(1.0 / Q_LORA), q_eps))
                    raw_n = ptx.ld_shared_v4(s_xs + n * XS_STRIDE + tid * 16)
                    self.store_normed(
                        s_xs, raw_n, rr, wv, n, tid, h, r, qc_out, qc_out_stride
                    )
        ptx.sync_threads()
        self.mark(5, tid)

        self.wait_weights(bar)
        self.mark(6, tid)

        g = lane // 4
        t = lane % 4
        zero = Float32(0.0)
        z4 = (zero, zero, zero, zero)
        c = [z4, z4, z4, z4]  # [j * 2 + token tile]
        for i in cutlass.range_constexpr(8):
            s_ = warp * 8 + i
            b0 = _lds_v4_if(s_xs + g * XS_STRIDE + (s_ * 32 + 8 * t) * 2, g < rows)
            b1 = _lds_v4_if(
                s_xs + (g + 8) * XS_STRIDE + (s_ * 32 + 8 * t) * 2, g + 8 < rows
            )
            for j in cutlass.range_constexpr(2):
                blk = base + off["wqb"] + (j * (Q_LORA // 32) + s_) * 1024
                lo = ptx.ld_shared_v4(blk + lane * 16)
                hi = ptx.ld_shared_v4(blk + 512 + lane * 16)
                for tt in cutlass.range_constexpr(2):
                    bq = b0 if tt == 0 else b1
                    c1 = ptx.mma_m16n8k16_bf16(
                        (lo[0], hi[0], lo[1], hi[1]), (bq[0], bq[1]), c[j * 2 + tt]
                    )
                    c[j * 2 + tt] = ptx.mma_m16n8k16_bf16(
                        (lo[2], hi[2], lo[3], hi[3]), (bq[2], bq[3]), c1
                    )
        if cutlass.const_expr(rn):
            # the q_a output (indexer layers): every head CTA holds all normalized rows,
            # so head CTA q = 8 h + r stores row q / 4, chunks 64 (q % 4) .. + 63 (1 KiB)
            # from its XS slot before the partials reuse it
            if qc_out != Int64(0):
                n_q = (h * 8 + r) // 4
                c_q = ((h * 8 + r) % 4) * 64 + tid
                if (tid < 64) & (n_q < rows):
                    ptx.st_global_v4(
                        qc_out + (Int64(n_q) * qc_out_stride + Int64(c_q * 8)) * 2,
                        ptx.ld_shared_v4(s_xs + n_q * XS_STRIDE + c_q * 16),
                    )
        # the partials reuse the token rows: every warp must be done reading them
        self.rows_dead()
        self.mark(7, tid)
        # each accumulator to its (warp, token tile, token, q_b row) word: the C fragment
        # holds rows g / g + 8 of tile j, tokens 2 t / 2 t + 1 (no bank conflict)
        for j in cutlass.range_constexpr(2):
            for tt in cutlass.range_constexpr(2):
                cc = c[j * 2 + tt]
                for e in cutlass.range_constexpr(4):
                    ptx.st_shared_f32(
                        s_red
                        + (
                            ((warp * 2 + tt) * 8 + 2 * t + e % 2) * RED_QB_ROW
                            + j * 16
                            + g
                            + 8 * (e // 2)
                        )
                        * 4,
                        cc[e],
                    )
        self.sync_partials()
        self.mark(8, tid)
        # both (row, column) slots' partials first (a warp reads 32 consecutive words),
        # then the sums in warp order
        parts = []
        for it in cutlass.range_constexpr(2):
            idx = tid + THREADS * it
            rho = idx % 32
            n = idx // 32
            p = []
            for w in cutlass.range_constexpr(8):
                p.append(
                    _lds_f32_if(
                        s_red
                        + (((w * 2 + n // 8) * 8 + n % 8) * RED_QB_ROW + rho) * 4,
                        idx < 32 * rows,
                    )
                )
            parts.append(p)
        # q_nope ranks publish their BF16 rows; the RoPE ranks keep their FP8 q_pe bytes
        # in registers and store them with the absorbed columns at the end, so no global
        # store is outstanding at the cluster barriers' release
        if cutlass.const_expr(self.push):
            ptx.cluster_wait_acquire()  # co-start
        qbs = []
        for it in cutlass.range_constexpr(2):
            idx = tid + THREADS * it
            rho = idx % 32
            n = idx // 32
            v = Float32(0.0)
            for w in cutlass.range_constexpr(8):
                v = ptx.add_rn(v, parts[it][w])
            qb = ptx.cvt_rn_bf16_f32(v)
            if cutlass.const_expr(self.push):
                self.push_qnope(
                    base, qbar, r, n, lane, qb, (r < NOPE // 32) & (idx < 32 * rows)
                )
            else:
                if r < NOPE // 32:
                    if idx < 32 * rows:
                        ptx.st_shared_b16(s_pub + (n * 32 + rho) * 2, qb)
            qbs.append(qb)
        self.mark(9, tid)
        if cutlass.const_expr(self.push):
            rope_b = []
            for it in cutlass.range_constexpr(2):
                rope_b.append(
                    self.q_rope_byte(
                        r >= NOPE // 32,
                        (r - NOPE // 32) * 32 + (tid + THREADS * it) % 32,
                        qbs[it],
                        rope_c[it],
                        rope_s[it],
                        inv_qs,
                    )
                )
            self.mark(10, tid)
            ptx.mbarrier_wait_parity(qbar, 0, ".acquire.cluster")
            ptx.cluster_arrive_relaxed()  # exit: waited before this CTA leaves
            self.mark(11, tid)
        else:
            rb0, rb1 = self.handoff16o(
                base, tid, rows, r, qbs, rope_c, rope_s, inv_qs
            )
            rope_b = [rb0, rb1]
        tile = warp % 4
        half = warp // 4
        cw = [z4, z4]
        for i in cutlass.range_constexpr(3):
            s_ = half * 3 + i
            b0 = _lds_v4_if(s_qn + g * QN_STRIDE + (s_ * 32 + 8 * t) * 2, g < rows)
            b1 = _lds_v4_if(
                s_qn + (g + 8) * QN_STRIDE + (s_ * 32 + 8 * t) * 2, g + 8 < rows
            )
            blk = base + off["wuk"] + (tile * (NOPE // 32) + s_) * 1024
            lo = ptx.ld_shared_v4(blk + lane * 16)
            hi = ptx.ld_shared_v4(blk + 512 + lane * 16)
            for tt in cutlass.range_constexpr(2):
                bq = b0 if tt == 0 else b1
                c1 = ptx.mma_m16n8k16_bf16(
                    (lo[0], hi[0], lo[1], hi[1]), (bq[0], bq[1]), cw[tt]
                )
                cw[tt] = ptx.mma_m16n8k16_bf16(
                    (lo[2], hi[2], lo[3], hi[3]), (bq[2], bq[3]), c1
                )
        # each accumulator to its (half, token tile, token, absorbed column) word
        for tt in cutlass.range_constexpr(2):
            for e in cutlass.range_constexpr(4):
                ptx.st_shared_f32(
                    s_red
                    + (
                        ((half * 2 + tt) * 8 + 2 * t + e % 2) * RED_UK_ROW
                        + tile * 16
                        + g
                        + 8 * (e // 2)
                    )
                    * 4,
                    cw[tt][e],
                )
        self.sync_partials()
        self.mark(12, tid)
        nk = (64 * max_rows + THREADS - 1) // THREADS
        pa = []
        pb = []
        for k in cutlass.range_constexpr(nk):
            idx = tid + k * THREADS
            col = idx % 64
            n = idx // 64
            # half 0 (K steps 0..2) + half 1 (K steps 3..5), as head_cta16
            pa.append(
                _lds_f32_if(
                    s_red + ((((0 * 2 + n // 8) * 8 + n % 8) * RED_UK_ROW) + col) * 4,
                    idx < 64 * rows,
                )
            )
            pb.append(
                _lds_f32_if(
                    s_red + ((((1 * 2 + n // 8) * 8 + n % 8) * RED_UK_ROW) + col) * 4,
                    idx < 64 * rows,
                )
            )
        for k in cutlass.range_constexpr(nk):
            idx = tid + k * THREADS
            if idx < 64 * rows:
                # (own names: values assigned under a branch must keep one type per name)
                col = idx % 64
                row = idx // 64
                qf = ptx.bf16_to_f32(ptx.cvt_rn_bf16_f32(ptx.add_rn(pa[k], pb[k])))
                dst = mqa_q + Int64(row) * mqa_stride + Int64(h * MQA_DIM + r * 64 + col)
                ptx.st_global_u8(
                    dst,
                    cutlass.Uint16(
                        Uint32(
                            ptx.cvt_rn_satfinite_e4m3x2_f32(
                                ptx.mul_rn(qf, inv_qs), Float32(0.0)
                            )
                        )
                        & Uint32(0xFF)
                    ),
                )
        # the RoPE ranks' q_pe bytes (computed before the hand-off)
        if r >= NOPE // 32:
            for it in cutlass.range_constexpr(2):
                idx = tid + THREADS * it
                if idx < 32 * rows:
                    d = (r - NOPE // 32) * 32 + idx % 32
                    ptx.st_global_u8(
                        mqa_q
                        + Int64(idx // 32) * mqa_stride
                        + Int64(h * MQA_DIM + KV_LORA + d),
                        rope_b[it],
                    )
        self.mark(13, tid)
        self.release_outputs()
        self.exit_wait()
        self.mark(14, tid)

    @cute.jit
    def head_cta32o(
        self,
        base: Int32,
        h: Int32,
        r: Int32,
        tid: Int32,
        q_c: Int64,
        q_c_stride: Int64,
        positions: Int64,
        rows: Int32,
        w_qb: Int64,
        w_uk: Int64,
        q_norm_w: Int64,
        q_eps: Float32,
        rope_cache: Int64,
        rope_stride: Int64,
        q_scale: Int64,
        mqa_q: Int64,
        mqa_stride: Int64,
        qc_out: Int64,
        qc_out_stride: Int64,
    ):
        """``head_cta16o`` (overlap16 + push + row_norm + early_arm) extended to up to
        32 rows by running 1-2 chunks of 16 rows while keeping the 128 KiB W_qb and
        24 KiB W_UK resident in shared memory."""
        off = self.off
        lane = tid % 32
        warp = tid // 32
        bar = base + off["mbar"]
        qbar = bar + 8

        c0_rows = rows
        if c0_rows > 16:
            c0_rows = Int32(16)

        if tid == 0:
            ptx.mbarrier_init(bar, 1)
            ptx.mbarrier_init(qbar, 1)
            ptx.fence_mbarrier_init_release_cluster()
            ptx.fence_proxy_async_shared_cta()
            ptx.mbarrier_arrive_expect_tx(qbar, (NOPE // 32) * 64 * c0_rows)
            ptx.mbarrier_arrive_expect_tx(bar, WQB_BYTES + WUK_BYTES)
            src_qb = w_qb + Int64(h * 16 + 2 * r) * ((Q_LORA // 32) * 1024)
            src_uk = w_uk + Int64(h * 32 + 4 * r) * ((NOPE // 32) * 1024)
            for part in cutlass.range_constexpr(2):
                ptx.bulk_copy_g2s(
                    base + off["wqb"] + part * (WQB_BYTES // 2),
                    src_qb + part * (WQB_BYTES // 2),
                    WQB_BYTES // 2,
                    bar,
                )
            ptx.bulk_copy_g2s(base + off["wuk"], src_uk, WUK_BYTES, bar)

        nw = []
        for b in cutlass.range_constexpr(8):
            q_w = ptx.ld_global_nc_v4(q_norm_w + Int64(lane + 32 * b) * 16)
            words = []
            for e in cutlass.range_constexpr(4):
                lo, hi = ptx.bf16x2_to_f32x2(q_w[e])
                if cutlass.const_expr(self.num.norm_weight_plus_zero):
                    lo = ptx.add_rn(lo, Float32(0.0))
                    hi = ptx.add_rn(hi, Float32(0.0))
                words.append(ptx.cvt_rn_bf16x2_f32(lo, hi))
            nw.append(words)
        inv_qs = ptx.rcp_rn(ptx.ld_global_nc_f32(q_scale))
        ptx.sync_threads()
        ptx.cluster_arrive_relaxed()

        self.wait_q_inputs()

        self.head_chunk16o(
            base,
            h,
            r,
            tid,
            warp,
            lane,
            bar,
            qbar,
            nw,
            inv_qs,
            q_c,
            q_c_stride,
            positions,
            c0_rows,
            Int32(0),
            q_eps,
            rope_cache,
            rope_stride,
            mqa_q,
            mqa_stride,
            qc_out,
            qc_out_stride,
            0,
            True,
        )

        if rows > 16:
            ptx.sync_threads()
            c1_rows = rows - Int32(16)
            if tid == 0:
                ptx.mbarrier_arrive_expect_tx(qbar, (NOPE // 32) * 64 * c1_rows)
            ptx.sync_threads()
            ptx.cluster_arrive_release()
            self.head_chunk16o(
                base,
                h,
                r,
                tid,
                warp,
                lane,
                bar,
                qbar,
                nw,
                inv_qs,
                q_c,
                q_c_stride,
                positions,
                c1_rows,
                Int32(16),
                q_eps,
                rope_cache,
                rope_stride,
                mqa_q,
                mqa_stride,
                qc_out,
                qc_out_stride,
                1,
                False,
            )

        ptx.cluster_arrive_relaxed()
        self.release_outputs()
        self.exit_wait()

    @cute.jit
    def head_chunk16o(
        self,
        base: Int32,
        h: Int32,
        r: Int32,
        tid: Int32,
        warp: Int32,
        lane: Int32,
        bar: Int32,
        qbar: Int32,
        nw,
        inv_qs: Float32,
        q_c: Int64,
        q_c_stride: Int64,
        positions: Int64,
        c_rows: Int32,
        row_off: Int32,
        q_eps: Float32,
        rope_cache: Int64,
        rope_stride: Int64,
        mqa_q: Int64,
        mqa_stride: Int64,
        qc_out: Int64,
        qc_out_stride: Int64,
        phase: cutlass.Constexpr[int],
        wait_w: cutlass.Constexpr[bool],
    ):
        off = self.off
        s_xs = base + off["xs"]
        s_red = base + off["red"]
        s_qn = base + off["qn"]

        is_rope = r >= NOPE // 32
        pair = ((r - NOPE // 32) * 32 + lane) // 2
        pv = []
        for it in cutlass.range_constexpr(2):
            pv.append(
                self.ldg_s64_if(
                    positions + Int64(row_off + (tid + THREADS * it) // 32) * 8,
                    is_rope & (tid + THREADS * it < 32 * c_rows),
                )
            )
        qv = []
        for hf in cutlass.range_constexpr(2):
            for b in cutlass.range_constexpr(8):
                qv.append(
                    ptx.ld_global_v4_if(
                        q_c
                        + (
                            Int64(row_off + warp + 8 * hf) * q_c_stride
                            + Int64((lane + 32 * b) * 8)
                        )
                        * 2,
                        warp + 8 * hf < c_rows,
                    )
                )
        rope_c = [Float32(0.0), Float32(0.0)]
        rope_s = [Float32(0.0), Float32(0.0)]
        for it in cutlass.range_constexpr(2):
            rope_c[it], rope_s[it] = self.rope_at(
                is_rope & (tid + THREADS * it < 32 * c_rows),
                pv[it],
                pair,
                rope_cache,
                rope_stride,
            )
        self.row_norm16(s_xs, qv, nw, tid, warp, lane, c_rows, q_eps)
        ptx.sync_threads()

        if cutlass.const_expr(wait_w):
            self.wait_weights(bar)

        g = lane // 4
        t = lane % 4
        zero = Float32(0.0)
        z4 = (zero, zero, zero, zero)
        c = [z4, z4, z4, z4]
        for i in cutlass.range_constexpr(8):
            s_ = warp * 8 + i
            b0 = _lds_v4_if(s_xs + g * XS_STRIDE + (s_ * 32 + 8 * t) * 2, g < c_rows)
            b1 = _lds_v4_if(
                s_xs + (g + 8) * XS_STRIDE + (s_ * 32 + 8 * t) * 2, g + 8 < c_rows
            )
            for j in cutlass.range_constexpr(2):
                blk = base + off["wqb"] + (j * (Q_LORA // 32) + s_) * 1024
                lo = ptx.ld_shared_v4(blk + lane * 16)
                hi = ptx.ld_shared_v4(blk + 512 + lane * 16)
                for tt in cutlass.range_constexpr(2):
                    bq = b0 if tt == 0 else b1
                    c1 = ptx.mma_m16n8k16_bf16(
                        (lo[0], hi[0], lo[1], hi[1]), (bq[0], bq[1]), c[j * 2 + tt]
                    )
                    c[j * 2 + tt] = ptx.mma_m16n8k16_bf16(
                        (lo[2], hi[2], lo[3], hi[3]), (bq[2], bq[3]), c1
                    )
        if qc_out != Int64(0):
            n_q = (h * 8 + r) // 4
            c_q = ((h * 8 + r) % 4) * 64 + tid
            if (h < 8) & (tid < 64) & (n_q < c_rows):
                ptx.st_global_v4(
                    qc_out
                    + (Int64(row_off + n_q) * qc_out_stride + Int64(c_q * 8)) * 2,
                    ptx.ld_shared_v4(s_xs + n_q * XS_STRIDE + c_q * 16),
                )
        self.rows_dead()
        for j in cutlass.range_constexpr(2):
            for tt in cutlass.range_constexpr(2):
                cc = c[j * 2 + tt]
                for e in cutlass.range_constexpr(4):
                    ptx.st_shared_f32(
                        s_red
                        + (
                            ((warp * 2 + tt) * 8 + 2 * t + e % 2) * RED_QB_ROW
                            + j * 16
                            + g
                            + 8 * (e // 2)
                        )
                        * 4,
                        cc[e],
                    )
        self.sync_partials()

        parts = []
        for it in cutlass.range_constexpr(2):
            idx = tid + THREADS * it
            rho = idx % 32
            n = idx // 32
            p = []
            for w in cutlass.range_constexpr(8):
                p.append(
                    _lds_f32_if(
                        s_red
                        + (((w * 2 + n // 8) * 8 + n % 8) * RED_QB_ROW + rho) * 4,
                        idx < 32 * c_rows,
                    )
                )
            parts.append(p)

        ptx.cluster_wait_acquire()
        qbs = []
        for it in cutlass.range_constexpr(2):
            idx = tid + THREADS * it
            n = idx // 32
            v = Float32(0.0)
            for w in cutlass.range_constexpr(8):
                v = ptx.add_rn(v, parts[it][w])
            qb = ptx.cvt_rn_bf16_f32(v)
            self.push_qnope(
                base, qbar, r, n, lane, qb, (r < NOPE // 32) & (idx < 32 * c_rows)
            )
            qbs.append(qb)

        rope_b = []
        for it in cutlass.range_constexpr(2):
            rope_b.append(
                self.q_rope_byte(
                    r >= NOPE // 32,
                    (r - NOPE // 32) * 32 + (tid + THREADS * it) % 32,
                    qbs[it],
                    rope_c[it],
                    rope_s[it],
                    inv_qs,
                )
            )
        ptx.mbarrier_wait_parity(qbar, phase, ".acquire.cluster")

        tile = warp % 4
        half = warp // 4
        cw = [z4, z4]
        for i in cutlass.range_constexpr(3):
            s_ = half * 3 + i
            b0 = _lds_v4_if(s_qn + g * QN_STRIDE + (s_ * 32 + 8 * t) * 2, g < c_rows)
            b1 = _lds_v4_if(
                s_qn + (g + 8) * QN_STRIDE + (s_ * 32 + 8 * t) * 2, g + 8 < c_rows
            )
            blk = base + off["wuk"] + (tile * (NOPE // 32) + s_) * 1024
            lo = ptx.ld_shared_v4(blk + lane * 16)
            hi = ptx.ld_shared_v4(blk + 512 + lane * 16)
            for tt in cutlass.range_constexpr(2):
                bq = b0 if tt == 0 else b1
                c1 = ptx.mma_m16n8k16_bf16(
                    (lo[0], hi[0], lo[1], hi[1]), (bq[0], bq[1]), cw[tt]
                )
                cw[tt] = ptx.mma_m16n8k16_bf16(
                    (lo[2], hi[2], lo[3], hi[3]), (bq[2], bq[3]), c1
                )
        for tt in cutlass.range_constexpr(2):
            for e in cutlass.range_constexpr(4):
                ptx.st_shared_f32(
                    s_red
                    + (
                        ((half * 2 + tt) * 8 + 2 * t + e % 2) * RED_UK_ROW
                        + tile * 16
                        + g
                        + 8 * (e // 2)
                    )
                    * 4,
                    cw[tt][e],
                )
        self.sync_partials()
        nk = (64 * 16 + THREADS - 1) // THREADS
        pa = []
        pb = []
        for k in cutlass.range_constexpr(nk):
            idx = tid + k * THREADS
            col = idx % 64
            n = idx // 64
            pa.append(
                _lds_f32_if(
                    s_red + ((((0 * 2 + n // 8) * 8 + n % 8) * RED_UK_ROW) + col) * 4,
                    idx < 64 * c_rows,
                )
            )
            pb.append(
                _lds_f32_if(
                    s_red + ((((1 * 2 + n // 8) * 8 + n % 8) * RED_UK_ROW) + col) * 4,
                    idx < 64 * c_rows,
                )
            )
        for k in cutlass.range_constexpr(nk):
            idx = tid + k * THREADS
            if idx < 64 * c_rows:
                col = idx % 64
                row = idx // 64
                qf = ptx.bf16_to_f32(ptx.cvt_rn_bf16_f32(ptx.add_rn(pa[k], pb[k])))
                dst = (
                    mqa_q
                    + Int64(row_off + row) * mqa_stride
                    + Int64(h * MQA_DIM + r * 64 + col)
                )
                ptx.st_global_u8(
                    dst,
                    cutlass.Uint16(
                        Uint32(
                            ptx.cvt_rn_satfinite_e4m3x2_f32(
                                ptx.mul_rn(qf, inv_qs), Float32(0.0)
                            )
                        )
                        & Uint32(0xFF)
                    ),
                )
        if r >= NOPE // 32:
            for it in cutlass.range_constexpr(2):
                idx = tid + THREADS * it
                if idx < 32 * c_rows:
                    d = (r - NOPE // 32) * 32 + idx % 32
                    ptx.st_global_u8(
                        mqa_q
                        + Int64(row_off + idx // 32) * mqa_stride
                        + Int64(h * MQA_DIM + KV_LORA + d),
                        rope_b[it],
                    )

    @cute.jit
    def row_norm16(
        self,
        s_xs: Int32,
        qv,
        nw,
        tid: Int32,
        warp: Int32,
        lane: Int32,
        rows: Int32,
        q_eps: Float32,
    ):
        """row_norm: warp w's rows w / w + 8 (``qv[8 half + b]`` = chunk l + 32 b of
        lane l), one after the other: the row summed in q_norm_row_strided's order from
        registers; after the butterfly every lane holds the row's sum (each step adds
        the same two values in either order), so each lane normalizes its chunks with
        head_cta16's arithmetic and stores them to the row's XS slot. No branch: an
        unused row is computed and not stored."""
        wf = []
        for b in cutlass.range_constexpr(8):
            wb = []
            for e in cutlass.range_constexpr(4):
                lo, hi = ptx.bf16x2_to_f32x2(nw[b][e])
                wb.append(lo)
                wb.append(hi)
            wf.append(wb)
        for hf in cutlass.range_constexpr(2):
            sm = Float32(0.0)
            for b in cutlass.range_constexpr(8):
                for e in cutlass.range_constexpr(4):
                    lo, hi = ptx.bf16x2_to_f32x2(qv[8 * hf + b][e])
                    sm = ptx.add_rn(sm, ptx.mul_rn(lo, lo))
                    sm = ptx.add_rn(sm, ptx.mul_rn(hi, hi))
            for k in cutlass.range_constexpr(5):
                sm = ptx.add_rn(sm, cute.arch.shuffle_sync_bfly(sm, 1 << k))
            rr = ptx.rsqrt_approx(ptx.fma_rn(sm, Float32(1.0 / Q_LORA), q_eps))
            for b in cutlass.range_constexpr(8):
                _sts_v4_if(
                    s_xs + (warp + 8 * hf) * XS_STRIDE + (lane + 32 * b) * 16,
                    self.norm_words(qv[8 * hf + b], rr, wf[b]),
                    warp + 8 * hf < rows,
                )
            if cutlass.const_expr(hf == 0):
                self.mark(3, tid)

    @cute.jit
    def q_rope_byte(
        self,
        valid,
        d: Int32,
        qb,
        c: Float32,
        s: Float32,
        inv_qs: Float32,
    ):
        """The FP8 q_pe byte of dim d (``head_cta16``'s RoPE arithmetic on the BF16
        q_b output ``qb``; the pair partner is the neighbouring lane), 0 unless
        ``valid`` (CTA-uniform: the shuffle sees the whole warp)."""
        byte = cutlass.Uint16(0)
        if valid:
            own = ptx.bf16_to_f32(qb)
            other = cute.arch.shuffle_sync_bfly(own, 1)
            o = Float32(0.0)
            if d % 2 == 0:
                o = self.q_rope_even(own, other, c, s)
            else:
                o = self.q_rope_odd(other, own, c, s)
            o = ptx.mul_rn(o, inv_qs)
            byte = cutlass.Uint16(
                Uint32(ptx.cvt_rn_satfinite_e4m3x2_f32(o, Float32(0.0))) & Uint32(0xFF)
            )
        return byte

    @cute.jit
    def mark(self, k: cutlass.Constexpr[int], tid: Int32):
        """Probe builds: thread 0 stores {clock64, globaltimer} to slot k of its CTA."""
        if cutlass.const_expr(self.probe):
            if tid == 0:
                cta_, _, _ = cute.arch.block_idx()
                a = self._probe_base + Int64((Int32(cta_) * PROBE_MARKS + k) * 16)
                ptx.st_global_u64(a, _clock64())
                ptx.st_global_u64(a + Int64(8), ptx_sys.globaltimer())

    @cute.jit
    def handoff16o(self, base: Int32, tid: Int32, rows: Int32, r: Int32, qbs, rope_c, rope_s, inv_qs: Float32):
        """overlap16 without push: the cluster-barrier hand-off (publish, both gather
        slots' remote loads, the RoPE ranks' q_pe bytes while they are in flight, the
        local stores, done reading); returns the q_pe bytes."""
        self.publish()
        self.mark(10, tid)
        gv = []
        for it in cutlass.range_constexpr(2):
            gv.append(self.gather_load(base, tid + THREADS * it, rows))
        rope_b = []
        for it in cutlass.range_constexpr(2):
            rope_b.append(
                self.q_rope_byte(
                    r >= NOPE // 32,
                    (r - NOPE // 32) * 32 + (tid + THREADS * it) % 32,
                    qbs[it],
                    rope_c[it],
                    rope_s[it],
                    inv_qs,
                )
            )
        for it in cutlass.range_constexpr(2):
            self.gather_store(base, tid + THREADS * it, rows, gv[it])
        self.done_reading()
        ptx.sync_threads()
        self.mark(11, tid)
        return rope_b[0], rope_b[1]

    @cute.jit
    def push_qnope(
        self, base: Int32, qbar: Int32, r: Int32, n: Int32, lane: Int32, qb, valid
    ):
        """push: a warp's 32 q_nope values of row n (lane = column) to every rank's QN
        row n, columns 32 r ..: lane pairs packed by a shuffle, 8-value chunks by four,
        then lane l stores chunk l / 8 to rank l % 8 (one st.async.v4 each),
        completing on that rank's QN barrier. ``valid`` is warp-uniform."""
        q32 = Uint32(qb)
        pk = q32 | (Uint32(cute.arch.shuffle_sync_bfly(q32, 1)) << Uint32(16))
        c = lane // 8
        w0 = Uint32(cute.arch.shuffle_sync(pk, 8 * c))
        w1 = Uint32(cute.arch.shuffle_sync(pk, 8 * c + 2))
        w2 = Uint32(cute.arch.shuffle_sync(pk, 8 * c + 4))
        w3 = Uint32(cute.arch.shuffle_sync(pk, 8 * c + 6))
        if valid:
            d = lane % 8
            raddr = ptx.mapa_shared_cluster(
                base + self.off["qn"] + n * QN_STRIDE + r * 64 + c * 16, d
            )
            ptx.st_async_v4_b32(
                raddr, w0, w1, w2, w3, ptx.mapa_shared_cluster(qbar, d)
            )

    @cute.jit
    def norm_words(self, raw, rr: Float32, wv):
        """A row's 8 normalized q_a values (bf16(x rr w), the products rounded as
        head_cta's), as 4 BF16x2 words; f32x2: the multiplies packed."""
        words = []
        for e in cutlass.range_constexpr(4):
            x0, x1 = ptx.bf16x2_to_f32x2(raw[e])
            if cutlass.const_expr(self.f32x2):
                a0, a1 = _mul_f32x2(x0, x1, rr, rr)
                y0, y1 = _mul_f32x2(a0, a1, wv[2 * e], wv[2 * e + 1])
                words.append(ptx.cvt_rn_bf16x2_f32(y0, y1))
            else:
                words.append(
                    ptx.cvt_rn_bf16x2_f32(
                        ptx.mul_rn(ptx.mul_rn(x0, rr), wv[2 * e]),
                        ptx.mul_rn(ptx.mul_rn(x1, rr), wv[2 * e + 1]),
                    )
                )
        return words[0], words[1], words[2], words[3]

    @cute.jit
    def store_normed(
        self,
        s_xs: Int32,
        raw,
        rr: Float32,
        wv,
        n: cutlass.Constexpr[int],
        tid: Int32,
        h: Int32,
        r: Int32,
        qc_out: Int64,
        qc_out_stride: Int64,
    ):
        """Row n's normalized q_a words (``head_cta16``'s arithmetic) into its XS slot
        and, from head 0 rank 0, to ``qc_out``."""
        words = []
        for e in cutlass.range_constexpr(4):
            x0, x1 = ptx.bf16x2_to_f32x2(raw[e])
            words.append(
                ptx.cvt_rn_bf16x2_f32(
                    ptx.mul_rn(ptx.mul_rn(x0, rr), wv[2 * e]),
                    ptx.mul_rn(ptx.mul_rn(x1, rr), wv[2 * e + 1]),
                )
            )
        packed = (words[0], words[1], words[2], words[3])
        ptx.st_shared_v4(s_xs + n * XS_STRIDE + tid * 16, packed)
        if (qc_out != Int64(0)) & (h == 0) & (r == 0):
            ptx.st_global_v4(
                qc_out + (Int64(n) * qc_out_stride + Int64(tid * 8)) * 2, packed
            )

    @cute.jit
    def q_norm_row_strided(self, s_xs: Int32, s_ssq: Int32, n: Int32, lane: Int32):
        """Row n's strided sum of squares (``head_cta16``'s order) into SSQ."""
        s = Float32(0.0)
        row = s_xs + n * XS_STRIDE
        for b in cutlass.range_constexpr(8):
            u = ptx.ld_shared_v4(row + (lane + 32 * b) * 16)
            for e in cutlass.range_constexpr(4):
                lo, hi = ptx.bf16x2_to_f32x2(u[e])
                s = ptx.add_rn(s, ptx.mul_rn(lo, lo))
                s = ptx.add_rn(s, ptx.mul_rn(hi, hi))
        for k in cutlass.range_constexpr(5):
            s = ptx.add_rn(s, cute.arch.shuffle_sync_bfly(s, 1 << k))
        if lane == 0:
            ptx.st_shared_f32(s_ssq + n * 4, s)

    @cute.jit
    def q_norm_pair_strided(self, s_xs: Int32, s_ssq: Int32, n: Int32, lane: Int32):
        """Rows n and n + 8's strided sums of squares, the two chains interleaved
        (each in ``q_norm_row_strided``'s order): both rows' loads first."""
        row0 = s_xs + n * XS_STRIDE
        row1 = row0 + 8 * XS_STRIDE
        u0 = []
        u1 = []
        for b in cutlass.range_constexpr(8):
            u0.append(ptx.ld_shared_v4(row0 + (lane + 32 * b) * 16))
        for b in cutlass.range_constexpr(8):
            u1.append(ptx.ld_shared_v4(row1 + (lane + 32 * b) * 16))
        s0 = Float32(0.0)
        s1 = Float32(0.0)
        for b in cutlass.range_constexpr(8):
            for e in cutlass.range_constexpr(4):
                lo0, hi0 = ptx.bf16x2_to_f32x2(u0[b][e])
                lo1, hi1 = ptx.bf16x2_to_f32x2(u1[b][e])
                s0 = ptx.add_rn(s0, ptx.mul_rn(lo0, lo0))
                s1 = ptx.add_rn(s1, ptx.mul_rn(lo1, lo1))
                s0 = ptx.add_rn(s0, ptx.mul_rn(hi0, hi0))
                s1 = ptx.add_rn(s1, ptx.mul_rn(hi1, hi1))
        for k in cutlass.range_constexpr(5):
            s0 = ptx.add_rn(s0, cute.arch.shuffle_sync_bfly(s0, 1 << k))
            s1 = ptx.add_rn(s1, cute.arch.shuffle_sync_bfly(s1, 1 << k))
        if lane == 0:
            ptx.st_shared_f32(s_ssq + n * 4, s0)
            ptx.st_shared_f32(s_ssq + (n + 8) * 4, s1)

    @cute.jit
    def ldg_s64_if(self, addr: Int64, valid):
        v = Int64(0)
        if valid:
            v = ptx.ld_global_s64(addr)
        return v

    @cute.jit
    def rope_at(self, valid, pos: Int64, pair: Int32, cache: Int64, stride: Int64):
        """``rope_pair`` with the position already loaded."""
        c = Float32(0.0)
        s = Float32(0.0)
        if valid:
            c = self.rope_value(cache, stride, pos, pair, self.rope_cache_fp32)
            s = self.rope_value(cache, stride, pos, 32 + pair, self.rope_cache_fp32)
        return c, s

    @cute.jit
    def gather_load(self, base: Int32, tid: Int32, rows: Int32):
        """``gather``'s remote load (zeros for an unused slot)."""
        w0 = Uint32(0)
        w1 = Uint32(0)
        w2 = Uint32(0)
        w3 = Uint32(0)
        if tid < (NOPE // 32) * rows * 4:
            src = tid // (rows * 4)
            rem = tid % (rows * 4)
            n = rem // 4
            ch = rem % 4
            raddr = ptx.mapa_shared_cluster(
                base + self.off["pub"] + n * 64 + ch * 16, src
            )
            w0, w1, w2, w3 = ptx.ld_shared_cluster_v4(raddr)
        return w0, w1, w2, w3

    @cute.jit
    def gather_store(self, base: Int32, tid: Int32, rows: Int32, v):
        """``gather``'s local store of a ``gather_load`` result."""
        if tid < (NOPE // 32) * rows * 4:
            src = tid // (rows * 4)
            rem = tid % (rows * 4)
            n = rem // 4
            ch = rem % 4
            ptx.st_shared_v4(
                base + self.off["qn"] + n * QN_STRIDE + src * 64 + ch * 16, v
            )

    # ------------------------------------------------------------------ kv cluster
    @cute.jit
    def kv_cta(
        self,
        r: Int32,
        tid: Int32,
        kv_c: Int64,
        kv_c_stride: Int64,
        k_pe: Int64,
        k_pe_stride: Int64,
        idx_k: Int64,
        idx_k_stride: Int64,
        positions: Int64,
        slot_mapping: Int64,
        idx_slot_mapping: Int64,
        rows: Int32,
        has_indexer: Int32,
        kv_norm_w: Int64,
        kv_eps: Float32,
        rope_cache: Int64,
        rope_stride: Int64,
        k_scale: Int64,
        ik_w: Int64,
        ik_b: Int64,
        ik_eps: Float32,
        index_rope_cache: Int64,
        index_rope_stride: Int64,
        mla_cache: Int64,
        mla_block_stride: Int64,
        mla_entry_stride: Int64,
        mla_block_size: Int64,
        idx_cache: Int64,
        idx_block_stride: Int64,
        idx_block_size: Int64,
        topk: Int64,
        topk_stride: Int64,
        topk_width: Int32,
    ):
        lane = tid % 32
        warp = tid // 32
        # kv_split: at 16 rows, CTA 0/2 (MLA) and 1/3 (index-K) split rows 0..7 / 8..15;
        # at 32 rows, even CTAs 0,2,4,6 (MLA) and odd CTAs 1,3,5,7 (index-K) run all 4
        # 8-row groups in parallel across the 8-CTA cluster.
        nsplit = 6 if (self.kv_split and self.max_rows == 32) else (2 if self.kv_split else 0)
        mla_grp = Int32(-1)
        idx_grp = Int32(-1)
        is_mla = r == 0
        is_idx = r == 1
        if cutlass.const_expr(self.kv_split and self.max_rows == 32):
            is_mla = (r % 2) == 0
            is_idx = (r % 2) == 1
            mla_grp = r // 2
            idx_grp = r // 2
        elif cutlass.const_expr(self.kv_split):
            is_mla = (r == 0) | (r == 2)
            is_idx = (r == 1) | (r == 3)
            mla_grp = Int32(0)
            idx_grp = Int32(0)
            if r == 2:
                mla_grp = Int32(1)
            if r == 3:
                idx_grp = Int32(1)
        if is_mla:
            self.mla_rows(
                lane,
                warp,
                kv_c,
                kv_c_stride,
                k_pe,
                k_pe_stride,
                positions,
                slot_mapping,
                rows,
                kv_norm_w,
                kv_eps,
                rope_cache,
                rope_stride,
                k_scale,
                mla_cache,
                mla_block_stride,
                mla_entry_stride,
                mla_block_size,
                mla_grp,
            )
        else:
            if has_indexer == 0:
                # shared layers: no work, but every path of the grid executes the PDL
                # wait
                ptx.pdl_wait()
            else:
                if is_idx:
                    self.index_k_rows(
                        lane,
                        warp,
                        idx_k,
                        idx_k_stride,
                        positions,
                        slot_mapping,
                        idx_slot_mapping,
                        rows,
                        ik_w,
                        ik_b,
                        ik_eps,
                        index_rope_cache,
                        index_rope_stride,
                        idx_cache,
                        idx_block_stride,
                        idx_block_size,
                        idx_grp,
                    )
                else:
                    if cutlass.const_expr(2 + nsplit < CLUSTER):
                        self.wait_inputs()
                        per_row = topk_width // 4
                        end = rows * per_row
                        neg = Uint32(0xFFFFFFFF)
                        first = 2 + nsplit
                        for idx in cutlass.range(
                            (r - first) * THREADS + tid,
                            end,
                            (CLUSTER - first) * THREADS,
                            unroll=1,
                        ):
                            n = idx // per_row
                            cc = idx % per_row
                            ptx.st_global_v4(
                                topk + (Int64(n) * topk_stride + Int64(cc * 4)) * 4,
                                (neg, neg, neg, neg),
                            )

    @cute.jit
    def mla_rows(
        self,
        lane: Int32,
        warp: Int32,
        kv_c: Int64,
        kv_c_stride: Int64,
        k_pe: Int64,
        k_pe_stride: Int64,
        positions: Int64,
        slot_mapping: Int64,
        rows: Int32,
        kv_norm_w: Int64,
        kv_eps: Float32,
        rope_cache: Int64,
        rope_stride: Int64,
        k_scale: Int64,
        mla_cache: Int64,
        mla_block_stride: Int64,
        mla_entry_stride: Int64,
        mla_block_size: Int64,
        only_grp: Int32,
    ):
        """The kv RMSNorm + BF16 + per-tensor FP8 of the latent and the k_pe RoPE: one
        MLA row per token (warp n = row n). Lane l owns elements 8 l + [0, 8) and
        256 + 8 l + [0, 8). Slot <= 0 is not written (slot 0 is the padding slot)."""
        strided = self.num.norm_order == "strided_8x256"
        # strided: lane l owns elements 8 l + [0, 8) and 256 + 8 l + [0, 8);
        # contiguous: 16 l + [0, 16)
        e0 = lane * 8 if strided else lane * 16
        e1 = KV_LORA // 2 + lane * 8 if strided else lane * 16 + 8
        w0 = ptx.ld_global_nc_v4(kv_norm_w + Int64(e0) * 2)
        w1 = ptx.ld_global_nc_v4(kv_norm_w + Int64(e1) * 2)
        ksc = ptx.ld_global_nc_f32(k_scale)
        self.wait_kv_inputs()
        for grp in cutlass.range_constexpr(self.max_rows // 8):
            if (only_grp < 0) | (only_grp == grp):
                self.mla_rows_row(
                    warp + grp * 8,
                    lane,
                    kv_c,
                    kv_c_stride,
                    k_pe,
                    k_pe_stride,
                    positions,
                    slot_mapping,
                    rows,
                    kv_eps,
                    rope_cache,
                    rope_stride,
                    mla_cache,
                    mla_block_stride,
                    mla_entry_stride,
                    mla_block_size,
                    e0,
                    e1,
                    w0,
                    w1,
                    ksc,
                )

    @cute.jit
    def mla_rows_row(
        self,
        n: Int32,
        lane: Int32,
        kv_c: Int64,
        kv_c_stride: Int64,
        k_pe: Int64,
        k_pe_stride: Int64,
        positions: Int64,
        slot_mapping: Int64,
        rows: Int32,
        kv_eps: Float32,
        rope_cache: Int64,
        rope_stride: Int64,
        mla_cache: Int64,
        mla_block_stride: Int64,
        mla_entry_stride: Int64,
        mla_block_size: Int64,
        e0: Int32,
        e1: Int32,
        w0,
        w1,
        ksc: Float32,
    ):
        strided = self.num.norm_order == "strided_8x256"
        if n < rows:
            slot = ptx.ld_global_s64(slot_mapping + Int64(n) * 8)
            row = kv_c + Int64(n) * kv_c_stride * 2
            a = ptx.ld_global_v4(row + Int64(e0) * 2)
            b = ptx.ld_global_v4(row + Int64(e1) * 2)
            pos = ptx.ld_global_s64(positions + Int64(n) * 8)
            kp = ptx.ld_global_u32(
                k_pe + (Int64(n) * k_pe_stride + Int64(2 * lane)) * 2
            )
            c = self.rope_value(
                rope_cache, rope_stride, pos, lane, self.rope_cache_fp32
            )
            sn = self.rope_value(
                rope_cache, rope_stride, pos, 32 + lane, self.rope_cache_fp32
            )
            # negative slots are never written, slot 0 (padding) not under skip_slot0
            if slot > (Int64(0) if self.num.skip_slot0 else Int64(-1)):
                xs = (a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3])
                ws = (w0[0], w0[1], w0[2], w0[3], w1[0], w1[1], w1[2], w1[3])
                x = []
                for e in cutlass.range_constexpr(8):
                    lo, hi = ptx.bf16x2_to_f32x2(xs[e])
                    x.append(lo)
                    x.append(hi)
                s = Float32(0.0)
                for i in cutlass.range_constexpr(16):
                    if cutlass.const_expr(strided):
                        s = ptx.add_rn(s, ptx.mul_rn(x[i], x[i]))
                    else:
                        s = ptx.fma_rn(x[i], x[i], s)
                for k in cutlass.range_constexpr(5):
                    o = (1 << k) if strided else (16 >> k)
                    s = ptx.add_rn(s, cute.arch.shuffle_sync_bfly(s, o))
                rr = ptx.rsqrt_approx(ptx.fma_rn(s, Float32(1.0 / KV_LORA), kv_eps))
                entry = (
                    mla_cache
                    + (slot // mla_block_size) * mla_block_stride
                    + (slot % mla_block_size) * mla_entry_stride
                )
                q = []
                for e in cutlass.range_constexpr(4):
                    word = Uint32(0)
                    for k in cutlass.range_constexpr(4):
                        i = e * 4 + k
                        wlo, whi = ptx.bf16x2_to_f32x2(ws[i // 2])
                        wi = whi if i % 2 == 1 else wlo
                        xr = ptx.mul_rn(x[i], rr)
                        if cutlass.const_expr(self.num.norm_weight_plus_zero):
                            wi = ptx.add_rn(wi, Float32(0.0))
                        y = ptx.mul_rn(xr, wi)
                        if cutlass.const_expr(self.num.kv_norm_bf16):
                            # the norm output materialized in BF16
                            y = ptx.bf16_to_f32(ptx.cvt_rn_bf16_f32(y))
                        byte = Uint32(
                            ptx.cvt_rn_satfinite_e4m3x2_f32(
                                ptx.div_full(y, ksc), Float32(0.0)
                            )
                        ) & Uint32(0xFF)
                        word = word | (byte << Uint32(8 * k))
                    q.append(word)
                if cutlass.const_expr(strided):
                    ptx.st_global_v2(entry + Int64(e0), q[0], q[1])
                    ptx.st_global_v2(entry + Int64(e1), q[2], q[3])
                else:
                    ptx.st_global_v4(entry + Int64(e0), (q[0], q[1], q[2], q[3]))
                x1, x2 = ptx.bf16x2_to_f32x2(kp)
                r1 = ptx.fma_rn(x1, c, ptx.neg_f32(ptx.mul_rn(x2, sn)))
                r2 = ptx.fma_rn(x2, c, ptx.mul_rn(x1, sn))
                b1 = Uint32(
                    ptx.cvt_rn_satfinite_e4m3x2_f32(ptx.div_full(r1, ksc), Float32(0.0))
                ) & Uint32(0xFF)
                b2 = Uint32(
                    ptx.cvt_rn_satfinite_e4m3x2_f32(ptx.div_full(r2, ksc), Float32(0.0))
                ) & Uint32(0xFF)
                ptx.st_global_b16(
                    entry + Int64(KV_LORA + 2 * lane),
                    cutlass.Uint16(b1 | (b2 << Uint32(8))),
                )

    @cute.jit
    def index_k_rows(
        self,
        lane: Int32,
        warp: Int32,
        idx_k: Int64,
        idx_k_stride: Int64,
        positions: Int64,
        slot_mapping: Int64,
        idx_slot_mapping: Int64,
        rows: Int32,
        ik_w: Int64,
        ik_b: Int64,
        ik_eps: Float32,
        rope_cache: Int64,
        rope_stride: Int64,
        idx_cache: Int64,
        idx_block_stride: Int64,
        idx_block_size: Int64,
        only_grp: Int32,
    ):
        """Index-K: LayerNorm(128) + interleaved RoPE (dims < 64) + BF16 + FP8 with an
        FP32 scale, one row per token
        (warp n = row n, lane l = dims 4 l .. 4 l + 3). Rows whose MLA or index slot is
        negative are not written."""
        lw = ptx.ld_global_nc_v4(ik_w + Int64(lane) * 16)
        lb = ptx.ld_global_nc_v4(ik_b + Int64(lane) * 16)
        self.wait_inputs()
        for grp in cutlass.range_constexpr(self.max_rows // 8):
            if (only_grp < 0) | (only_grp == grp):
                self.index_k_rows_row(
                    warp + grp * 8,
                    lane,
                    idx_k,
                    idx_k_stride,
                    positions,
                    slot_mapping,
                    idx_slot_mapping,
                    rows,
                    ik_eps,
                    rope_cache,
                    rope_stride,
                    idx_cache,
                    idx_block_stride,
                    idx_block_size,
                    lw,
                    lb,
                )

    @cute.jit
    def index_k_rows_row(
        self,
        n: Int32,
        lane: Int32,
        idx_k: Int64,
        idx_k_stride: Int64,
        positions: Int64,
        slot_mapping: Int64,
        idx_slot_mapping: Int64,
        rows: Int32,
        ik_eps: Float32,
        rope_cache: Int64,
        rope_stride: Int64,
        idx_cache: Int64,
        idx_block_stride: Int64,
        idx_block_size: Int64,
        lw,
        lb,
    ):
        if n < rows:
            mla_slot = ptx.ld_global_s64(slot_mapping + Int64(n) * 8)
            slot = ptx.ld_global_s64(idx_slot_mapping + Int64(n) * 8)
            ipos = ptx.ld_global_s64(positions + Int64(n) * 8)
            kv = ptx.ld_global_v2(
                idx_k + (Int64(n) * idx_k_stride + Int64(4 * lane)) * 2
            )
            x0, x1 = ptx.bf16x2_to_f32x2(kv[0])
            x2, x3 = ptx.bf16x2_to_f32x2(kv[1])
            x = (x0, x1, x2, x3)
            num = self.num
            if cutlass.const_expr(num.index_mean_sum == "sequential"):
                sm = ptx.add_rn(
                    ptx.add_rn(ptx.add_rn(ptx.add_rn(Float32(0.0), x0), x1), x2), x3
                )
            else:
                sm = ptx.add_rn(ptx.add_rn(x0, x1), ptx.add_rn(x2, x3))
            for k in cutlass.range_constexpr(5):
                sm = ptx.add_rn(sm, cute.arch.shuffle_sync_bfly(sm, 16 >> k))
            d = []
            vs = Float32(0.0)
            for e in cutlass.range_constexpr(4):
                # x - mean with the mean's multiply folded in (one rounding)
                d.append(ptx.fma_rn(sm, Float32(-1.0 / INDEX_DIM), x[e]))
                vs = ptx.fma_rn(d[e], d[e], vs)
            for k in cutlass.range_constexpr(5):
                vs = ptx.add_rn(vs, cute.arch.shuffle_sync_bfly(vs, 16 >> k))
            rstd = ptx.rsqrt_approx(ptx.fma_rn(vs, Float32(1.0 / INDEX_DIM), ik_eps))
            nm = []
            for e in cutlass.range_constexpr(4):
                nm.append(
                    ptx.fma_rn(
                        ptx.mul_rn(d[e], rstd), ptx.bits_f32(lw[e]), ptx.bits_f32(lb[e])
                    )
                )
            res = self.index_rope(
                lane < 16,
                lane,
                nm[0],
                nm[1],
                nm[2],
                nm[3],
                rope_cache,
                rope_stride,
                ipos,
            )
            rb = []
            for e in cutlass.range_constexpr(4):
                if cutlass.const_expr(num.index_k_bf16):
                    rb.append(ptx.bf16_to_f32(ptx.cvt_rn_bf16_f32(res[e])))
                else:
                    rb.append(res[e])
            am = ptx.max_f32(
                ptx.max_f32(ptx.abs_f32(rb[0]), ptx.abs_f32(rb[1])),
                ptx.max_f32(ptx.abs_f32(rb[2]), ptx.abs_f32(rb[3])),
            )
            for k in cutlass.range_constexpr(5):
                am = ptx.max_f32(am, cute.arch.shuffle_sync_bfly(am, 16 >> k))
            sc = ptx.div_rn(ptx.max_f32(Float32(1e-4), am), Float32(448.0))
            inv_sc = Float32(1.0)
            if cutlass.const_expr(num.index_scale == "fp32_amax"):
                inv_sc = ptx.div_rn(Float32(1.0), sc)
            else:
                # power-of-two scale 2^ceil(log2(s)), values divided by it
                sc = cute.math.exp2(cute.math.ceil(cute.math.log2(sc)))
            if (mla_slot >= Int64(0)) & (slot >= Int64(0)):
                blk = slot // idx_block_size
                ofs = slot % idx_block_size
                row_base = idx_cache + blk * idx_block_stride
                word = Uint32(0)
                for e in cutlass.range_constexpr(4):
                    if cutlass.const_expr(num.index_scale == "fp32_amax"):
                        # x (1 / s), clipped to +-448
                        v = ptx.max_f32(
                            ptx.min_f32(ptx.mul_rn(rb[e], inv_sc), Float32(448.0)),
                            Float32(-448.0),
                        )
                    else:
                        v = ptx.div_rn(rb[e], sc)
                    byte = Uint32(
                        ptx.cvt_rn_satfinite_e4m3x2_f32(v, Float32(0.0))
                    ) & Uint32(0xFF)
                    word = word | (byte << Uint32(8 * e))
                ptx.st_global_u32(row_base + ofs * INDEX_DIM + Int64(4 * lane), word)
                if lane == 0:
                    at = row_base + idx_block_size * INDEX_DIM + ofs * 4
                    if cutlass.const_expr(self.index_scale_ue8m0x4):
                        # the power-of-two scale as four equal UE8M0 bytes
                        e8 = (ptx.f32_bits(sc) >> Uint32(23)) & Uint32(0xFF)
                        ptx.st_global_u32(at, e8 * Uint32(0x01010101))
                    else:
                        ptx.st_global_f32(at, sc)

    # ------------------------------------------------------------------ steps
    @cute.jit
    def q_norm_sums_strided(
        self,
        s_xs: Int32,
        s_ssq: Int32,
        qv,
        warp: Int32,
        lane: Int32,
        tid: Int32,
        rows: Int32,
    ):
        """q_a sums of squares, strided order: the raw rows go to the XS slots; warp n
        sums row n, lane l adding the rounded squares of elements 8 l + 256 b + v
        (b, v = 0..7) to 0 in (b, v) order, then a butterfly xor 1, 2, 4, 8, 16."""
        for n in cutlass.range_constexpr(self.max_rows):
            if n < rows:
                ptx.st_shared_v4(s_xs + n * XS_STRIDE + tid * 16, qv[n])
        ptx.sync_threads()
        if warp < rows:
            s = Float32(0.0)
            row = s_xs + warp * XS_STRIDE
            for b in cutlass.range_constexpr(8):
                u = ptx.ld_shared_v4(row + (lane + 32 * b) * 16)
                for e in cutlass.range_constexpr(4):
                    lo, hi = ptx.bf16x2_to_f32x2(u[e])
                    s = ptx.add_rn(s, ptx.mul_rn(lo, lo))
                    s = ptx.add_rn(s, ptx.mul_rn(hi, hi))
            for k in cutlass.range_constexpr(5):
                s = ptx.add_rn(s, cute.arch.shuffle_sync_bfly(s, 1 << k))
            if lane == 0:
                ptx.st_shared_f32(s_ssq + warp * 4, s)

    @cute.jit
    def q_norm_sums_contiguous(
        self, s_ssq: Int32, qv, warp: Int32, lane: Int32, rows: Int32
    ):
        """q_a sums of squares, contiguous order: each thread fuses the squares of its 8
        elements into 0 in order, a butterfly xor 16 .. 1 sums the warp, and the 8 warp
        sums of row n are added in warp order when the row is normalized."""
        for n in cutlass.range_constexpr(self.max_rows):
            if n < rows:
                s = Float32(0.0)
                for e in cutlass.range_constexpr(4):
                    lo, hi = ptx.bf16x2_to_f32x2(qv[n][e])
                    s = ptx.fma_rn(lo, lo, s)
                    s = ptx.fma_rn(hi, hi, s)
                for k in cutlass.range_constexpr(5):
                    s = ptx.add_rn(s, cute.arch.shuffle_sync_bfly(s, 16 >> k))
                if lane == 0:
                    ptx.st_shared_f32(s_ssq + (warp * self.max_rows + n) * 4, s)

    @cute.jit
    def load16(self, addr: Int64, valid):
        w0 = Uint32(0)
        w1 = Uint32(0)
        w2 = Uint32(0)
        w3 = Uint32(0)
        if valid:
            w0, w1, w2, w3 = ptx.ld_global_v4(addr)
        return w0, w1, w2, w3

    @cute.jit
    def lds16(self, addr: Int32, valid):
        w0 = Uint32(0)
        w1 = Uint32(0)
        w2 = Uint32(0)
        w3 = Uint32(0)
        if valid:
            w0, w1, w2, w3 = ptx.ld_shared_v4(addr)
        return w0, w1, w2, w3

    @cute.jit
    def rope_value(
        self,
        cache: Int64,
        stride: Int64,
        pos: Int64,
        col: Int32,
        fp32: cutlass.Constexpr[bool],
    ):
        idx = pos * stride + Int64(col)
        if cutlass.const_expr(fp32):
            v = ptx.ld_global_f32(cache + idx * 4)
        else:
            v = ptx.bf16_to_f32(ptx.ld_global_u16(cache + idx * 2))
        return v

    @cute.jit
    def index_rope(
        self,
        valid,
        lane: Int32,
        n0: Float32,
        n1: Float32,
        n2: Float32,
        n3: Float32,
        cache: Int64,
        stride: Int64,
        pos: Int64,
    ):
        """Interleaved RoPE of this lane's dims 4 l .. 4 l + 3 when they are < 64 (pairs
        (4 l, 4 l + 1), (4 l + 2, 4 l + 3); cos / sin index = dim / 2). Even: e0 cos -
        e1 sin with e1 sin rounded. Odd: e0 sin + e1 cos with e1 cos rounded
        (``round_x1_cos``) or e1 cos + e0 sin with e0 sin rounded (``round_x0_sin``)."""
        o0 = n0
        o1 = n1
        o2 = n2
        o3 = n3
        if valid:
            c0 = self.rope_value(
                cache, stride, pos, 2 * lane, self.index_rope_cache_fp32
            )
            s0 = self.rope_value(
                cache, stride, pos, 32 + 2 * lane, self.index_rope_cache_fp32
            )
            c1 = self.rope_value(
                cache, stride, pos, 2 * lane + 1, self.index_rope_cache_fp32
            )
            s1 = self.rope_value(
                cache, stride, pos, 33 + 2 * lane, self.index_rope_cache_fp32
            )
            o0 = ptx.fma_rn(n0, c0, ptx.neg_f32(ptx.mul_rn(n1, s0)))
            o2 = ptx.fma_rn(n2, c1, ptx.neg_f32(ptx.mul_rn(n3, s1)))
            if cutlass.const_expr(self.num.index_rope_odd == "round_x1_cos"):
                o1 = ptx.fma_rn(n0, s0, ptx.mul_rn(n1, c0))
                o3 = ptx.fma_rn(n2, s1, ptx.mul_rn(n3, c1))
            else:
                o1 = ptx.fma_rn(n1, c0, ptx.mul_rn(n0, s0))
                o3 = ptx.fma_rn(n3, c1, ptx.mul_rn(n2, s1))
        return o0, o1, o2, o3

    @cute.jit
    def q_rope_even(self, x0: Float32, x1: Float32, c: Float32, s: Float32) -> Float32:
        """q_pe RoPE, even dim: x0 cos - x1 sin (one product rounded: x1 sin rounded,
        the rest fused; else both products rounded, then the difference)."""
        o = Float32(0.0)
        if cutlass.const_expr(self.num.q_rope == "one_product_rounded"):
            o = self.rope_even(x0, x1, c, s)
        else:
            o = ptx.sub_rn(ptx.mul_rn(x0, c), ptx.mul_rn(x1, s))
        return o

    @cute.jit
    def q_rope_odd(self, x0: Float32, x1: Float32, c: Float32, s: Float32) -> Float32:
        """q_pe RoPE, odd dim: x1 cos + x0 sin (one product rounded: x0 sin rounded, the
        rest fused; else both products rounded, then the sum)."""
        o = Float32(0.0)
        if cutlass.const_expr(self.num.q_rope == "one_product_rounded"):
            o = self.rope_odd(x0, x1, c, s)
        else:
            o = ptx.add_rn(ptx.mul_rn(x1, c), ptx.mul_rn(x0, s))
        return o

    @cute.jit
    def rope_even(self, x0: Float32, x1: Float32, c: Float32, s: Float32) -> Float32:
        """x0 cos - x1 sin, the product x1 sin rounded, the other fused."""
        return ptx.fma_rn(x0, c, ptx.neg_f32(ptx.mul_rn(x1, s)))

    @cute.jit
    def rope_odd(self, x0: Float32, x1: Float32, c: Float32, s: Float32) -> Float32:
        """x1 cos + x0 sin, the product x0 sin rounded, the other fused."""
        return ptx.fma_rn(x1, c, ptx.mul_rn(x0, s))

    @cute.jit
    def gather(self, base: Int32, tid: Int32, rows: Int32):
        """The head's 192 nope values per row from ranks 0..5 (distributed shared
        memory) into QN."""
        if tid < (NOPE // 32) * rows * 4:
            src = tid // (rows * 4)
            rem = tid % (rows * 4)
            n = rem // 4
            ch = rem % 4
            raddr = ptx.mapa_shared_cluster(
                base + self.off["pub"] + n * 64 + ch * 16, src
            )
            v = ptx.ld_shared_cluster_v4(raddr)
            ptx.st_shared_v4(
                base + self.off["qn"] + n * QN_STRIDE + src * 64 + ch * 16, v
            )

    @cute.jit
    def rope_pair(
        self,
        valid,
        positions: Int64,
        token: Int32,
        pair: Int32,
        cache: Int64,
        stride: Int64,
    ):
        """The cos / sin of one (token, pair) q_pe RoPE slot, zero when unused."""
        c = Float32(0.0)
        s = Float32(0.0)
        if valid:
            pos = ptx.ld_global_s64(positions + Int64(token) * 8)
            c = self.rope_value(cache, stride, pos, pair, self.rope_cache_fp32)
            s = self.rope_value(cache, stride, pos, 32 + pair, self.rope_cache_fp32)
        return c, s

    @cute.jit
    def rows_dead(self):
        """Every warp's reads of the token rows -> the partials that reuse them."""
        ptx.sync_threads()

    @cute.jit
    def handoff(self, base: Int32, tid: Int32, rows: Int32):
        self.publish()
        self.gather(base, tid, rows)

    @cute.jit
    def wait_inputs(self):
        ptx.pdl_wait()
        ptx.fence_proxy_async_global()

    @cute.jit
    def wait_q_inputs(self):
        ptx.pdl_wait()
        ptx.fence_proxy_async_global()

    @cute.jit
    def wait_kv_inputs(self):
        ptx.pdl_wait()
        ptx.fence_proxy_async_global()

    @cute.jit
    def wait_weights(self, bar: Int32):
        ptx.mbarrier_wait_parity(bar, 0)

    @cute.jit
    def sync_partials(self):
        ptx.sync_threads()

    @cute.jit
    def publish(self):
        ptx.cluster_arrive_release()
        ptx.cluster_wait_acquire()

    @cute.jit
    def done_reading(self):
        ptx.cluster_arrive_release()

    @cute.jit
    def release_outputs(self):
        ptx.fence_proxy_async_global()

    @cute.jit
    def exit_wait(self):
        ptx.cluster_wait_acquire()
