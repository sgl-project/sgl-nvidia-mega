# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Portions Copyright 2023-2024 SGLang Team (Apache-2.0)
# Derived from SGLang (https://github.com/sgl-project/sglang), Apache License 2.0; see ../include/LICENSE.sglang.
# Source: python/sglang/kernels/ops/communication/muse_spark_moe_finalize_all_reduce_cutedsl.py at revision 306ccf6.
# Modified by NVIDIA: the one-shot push exchange of the fused finalize is replaced by a two-shot exchange through a
# plugin-owned arena (each rank reduces one column slice of every token in source-rank order and pushes the reduced
# slice to every peer), with credit-ordered slot reuse, per-block transfer counters and bounded waits. The PTX wrappers,
# reduction helpers, configuration and the finalize arithmetic of the kernel are unchanged.
"""CuTeDSL kernel: Muse Spark MoE finalize (unpermute + post-expert RMSNorm + weighted top-k sum) fused with a two-shot
all-reduce of the finalized rows across the tensor-parallel group.

Per token, every rank computes its finalized partial exactly as SGLang's fused finalize all-reduce kernel does (same
thread map, same cluster / thread configuration chosen by the caller, one bf16 rounding of the fp32 accumulator), then:

1. Column slice ``s`` (``H / world`` columns) of the bf16 partial goes to rank ``s`` (phase 1). Words are 32-bit pairs
   of bf16 values; a word equal to 0 is replaced by 0x00008000 before it is sent, exactly as the one-shot exchange
   does, so the summands are the same bits.
2. Rank ``s`` sums its slice over source ranks 0..world-1 in fp32 (the one-shot exchange's order), rounds to bf16 once
   and writes it to its output and to every peer (phase 2). A phase-2 word equal to 0xFFFFFFFF (two NaNs with that
   payload) is sent as 0x7FC07FC0 (two quiet NaNs): 0xFFFFFFFF is the empty marker of phase-2 slots.
3. Every rank copies the other slices from its phase-2 slots into its output.

The result is therefore bitwise the one-shot exchange's result, except the NaN payload of a word whose two values are
both the 0xFFFF NaN.

Exchange memory (one symmetric arena per tensor-parallel group, ``fin2.Fin2Arena``):
- phase-1 slots on rank ``o``: ``[kSlots][world src][rows_max][slice words]`` (empty word = 0);
- phase-2 slots on rank ``d``: ``[kSlots][world owner][rows_max][slice words]`` (empty word = 0xFFFFFFFF);
- credit words on every rank: ``[world]``, entry ``c`` written by consumer rank ``c``;
- per-block transfer counters (device memory, this rank).

Words are published with ``st.relaxed.sys`` 16-byte vector stores of 32-bit elements and polled with ``ld.relaxed.sys``
loads of the same vectors; a word is accepted when it differs from the empty marker. Acceptance relies on single-copy
atomicity of aligned 32-bit accesses, with vector accesses treated as one access per element (PTX ISA, memory
consistency model); every element of a vector is checked. The consumer restores the empty marker in every word it read,
with the same vector width.

Slot reuse (write after read). Transfer ``seq`` uses slot ``seq % kSlots``. Before any thread of a block stores into a
slot for ``seq >= kSlots``, one thread acquires (``ld.acquire.sys``) credit ``>= seq - kSlots + 1`` from every consumer
and the block synchronizes. A rank releases credit ``seq`` (transfers ``<= seq - 1`` consumed and re-armed) from the
NEXT launch, in a dedicated block, after that launch's ``griddepcontrol.wait``: one ``fence.acq_rel.sys`` followed by
relaxed stores to every producer (a release pattern per store).

Transfer numbers. Every block reads its own counter after the dependency wait and writes it + 1 when it ends; blocks of
a launch also advance the counters of block indices the launch does not use. All counters (and all ranks, which issue
the same launches) therefore agree on ``seq``.

Waits are bounded: after ``kSpinPolls`` polls a waiting thread sleeps between polls; after ``max_polls`` polls it sets
the sticky error word and traps.

Cross-rank co-scheduling: a block waits only on the blocks that handle the same tokens on the peer ranks (the same block
index), never on a block of its own grid; all blocks of every rank must make progress (the same requirement as the
one-shot exchange). The grid is capped at the caller's block budget (persistent token loop above it).
"""

import torch

try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass import Float32, Int32, Int64
    from cutlass._mlir.dialects import llvm
    from cutlass.cute.runtime import make_ptr
    from cutlass.cutlass_dsl import T, dsl_user_op

    HAS_CUTEDSL = True
except ImportError:
    HAS_CUTEDSL = False


K_SLOTS = 3
K_SPIN_POLLS = 4096
K_BACKOFF_NS = 256
P1_EMPTY = 0x00000000
P2_EMPTY = 0xFFFFFFFF
P2_NAN_PAIR = 0x7FC07FC0
ERR_TIMEOUT_PAYLOAD = 1
ERR_TIMEOUT_CREDIT = 2


if HAS_CUTEDSL:
    AR_COPY_BITS = 128
    AR_VEC_SIZE = AR_COPY_BITS // cutlass.BFloat16.width
    AR_VEC_WORDS = AR_COPY_BITS // cutlass.Uint32.width

    def make_fin2_compile_args(world: int):
        """Compilation placeholders matching Fin2Arena.arguments(world)."""

        def uint32_ptr():
            return make_ptr(
                cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16
            )

        return (
            tuple(
                uint32_ptr() for _ in range(world)
            ),  # phase-1 slot regions of every rank
            tuple(
                uint32_ptr() for _ in range(world)
            ),  # phase-2 slot regions of every rank
            tuple(uint32_ptr() for _ in range(world)),  # credit words of every rank
            uint32_ptr(),  # this rank's phase-1 slot region
            uint32_ptr(),  # this rank's phase-2 slot region
            uint32_ptr(),  # this rank's credit words
            uint32_ptr(),  # this rank's per-block transfer counters
            uint32_ptr(),  # sticky error word
            Int32(0),  # rank
            Int32(1),  # rows_max (slot row stride)
            Int32(1),  # max polls per wait
        )

    # =================================================================
    # Exchange PTX wrappers
    # =================================================================

    @dsl_user_op
    def _ld_acquire_sys(ptr, *, loc=None, ip=None):
        return cutlass.Uint32(
            llvm.inline_asm(
                T.i32(),
                [ptr.toint(loc=loc, ip=ip).ir_value()],
                "ld.acquire.sys.global.u32 $0, [$1];",
                "=r,l,~{memory}",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @dsl_user_op
    def _st_relaxed_sys_u32(ptr, value, *, loc=None, ip=None):
        llvm.inline_asm(
            None,
            [
                ptr.toint(loc=loc, ip=ip).ir_value(),
                cutlass.Uint32(value).ir_value(loc=loc, ip=ip),
            ],
            "st.relaxed.sys.global.u32 [$0], $1;",
            "l,r,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

    @dsl_user_op
    def _fence_acq_rel_sys(*, loc=None, ip=None):
        llvm.inline_asm(
            None,
            [],
            "fence.acq_rel.sys;",
            "~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

    @dsl_user_op
    def _nanosleep(*, loc=None, ip=None):
        llvm.inline_asm(
            None,
            [],
            f"nanosleep.u32 {K_BACKOFF_NS};",
            "~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

    @dsl_user_op
    def _fail(err_ptr, code, *, loc=None, ip=None):
        # Sticky error word, then trap: the CUDA error surfaces on the host.
        llvm.inline_asm(
            None,
            [
                err_ptr.toint(loc=loc, ip=ip).ir_value(),
                cutlass.Uint32(code).ir_value(loc=loc, ip=ip),
            ],
            "red.relaxed.sys.global.or.b32 [$0], $1; fence.sc.sys; trap;",
            "l,r,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

    _torch_to_cutlass_dtype = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }

    def _vec_at(addr, word):
        """A 16-byte vector of 32-bit words at byte address `addr` + 4 * `word`."""
        return cute.make_tensor(
            cute.make_ptr(
                cutlass.Uint32,
                addr + Int64(word) * 4,
                cute.AddressSpace.gmem,
                assumed_align=AR_COPY_BITS // 8,
            ),
            cute.make_layout((AR_VEC_WORDS,)),
        )

    def _relaxed_sys_atoms():
        load_atom = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            cutlass.Uint32,
            num_bits_per_copy=AR_COPY_BITS,
            memory_order=cute.nvgpu.MemoryOrder.RELAXED,
            memory_scope=cute.nvgpu.MemoryScope.SYS,
        )
        store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyR2GOp(),
            cutlass.Uint32,
            num_bits_per_copy=AR_COPY_BITS,
            memory_order=cute.nvgpu.MemoryOrder.RELAXED,
            memory_scope=cute.nvgpu.MemoryScope.SYS,
        )
        return load_atom, store_atom

    def _lt_wrap(a, b):
        """a < b for 32-bit transfer numbers that may wrap."""
        return ((a - b) & cutlass.Uint32(0x80000000)) != cutlass.Uint32(0)

    def _err_code(code, site, rank):
        return cutlass.Uint32(0x80000000 | (code << 16) | site) | (
            cutlass.Uint32(rank) << cutlass.Uint32(8)
        )

    @cute.jit
    def acquire_credits(ex, seq, c, world: cutlass.Constexpr):
        """Thread c (< world): wait until consumer c released the transfer that used this slot before (seq - kSlots).
        The caller follows it with a block barrier before any thread of the block stores into the slot."""
        credit_local, err, rank, max_polls = ex[5], ex[7], ex[8], ex[10]
        if seq >= cutlass.Uint32(K_SLOTS):
            if rank != c:
                need = seq - cutlass.Uint32(K_SLOTS - 1)
                polls = cutlass.Uint32(0)
                got = _ld_acquire_sys(credit_local + c)
                while _lt_wrap(got, need):
                    polls += cutlass.Uint32(1)
                    if polls > cutlass.Uint32(K_SPIN_POLLS):
                        _nanosleep()
                    if polls > cutlass.Uint32(max_polls):
                        _fail(err, _err_code(ERR_TIMEOUT_CREDIT, 0, rank))
                    got = _ld_acquire_sys(credit_local + c)

    @cute.jit
    def release_credits(ex, seq, world: cutlass.Constexpr):
        """One thread of the release block, after griddepcontrol.wait: transfers <= seq - 1 are consumed and re-armed."""
        credits, rank = ex[2], ex[8]
        if seq >= cutlass.Uint32(1):
            _fence_acq_rel_sys()
            for p in cutlass.range_constexpr(world):
                if rank != p:
                    _st_relaxed_sys_u32(credits[p] + rank, seq)

    @cute.jit
    def push_slice(
        acc,
        ex,
        token,
        offset,
        threads: cutlass.Constexpr,
        hidden: cutlass.Constexpr,
        cluster: cutlass.Constexpr,
        world: cutlass.Constexpr,
        seq,
    ):
        """Phase 1: this thread's bf16 partial (one bf16 rounding) to the owner of its columns' phase-1 slot (the owner may
        be this rank: its own summand travels through its own slot like every other source's)."""
        p1, rank, rows_max = ex[0], ex[8], ex[9]
        vec, iters = acc.shape
        assert cutlass.const_expr(vec == AR_VEC_SIZE), "Finalize AR requires vec=8"
        slice_cols = hidden // world
        slice_words = slice_cols // 2
        slot = Int32(seq % cutlass.Uint32(K_SLOTS))
        expert_values = cute.make_rmem_tensor(
            cute.make_layout((vec,)), cutlass.BFloat16
        )
        packed = cute.recast_tensor(expert_values, cutlass.Uint32)
        _, store_atom = _relaxed_sys_atoms()
        for p in cutlass.range_constexpr(iters):
            # Round once at the finalize boundary, then view pairs as Lamport words.
            expert_values.store(acc[None, p].load().to(cutlass.BFloat16))
            for v in cutlass.range_constexpr(AR_VEC_WORDS):
                value = packed[v]
                if value == 0:
                    value = cutlass.Uint32(0x8000)
                packed[v] = value
            col = offset + p * threads * cluster * vec
            owner = col // slice_cols
            word = (col - owner * slice_cols) // 2
            for d in cutlass.range_constexpr(world):
                if owner == d:
                    row = (
                        (slot * world + rank) * rows_max + token
                    ) * slice_words + word
                    cute.copy(store_atom, packed, _vec_at(p1[d].toint(), row))

    @cute.jit
    def reduce_slice(
        output,
        ex,
        token,
        offset,
        threads: cutlass.Constexpr,
        hidden: cutlass.Constexpr,
        cluster: cutlass.Constexpr,
        world: cutlass.Constexpr,
        iters: cutlass.Constexpr,
        seq,
    ):
        """Owner: phase-1 words of every source -> fp32 sum in source-rank order -> bf16 -> output and every peer's
        phase-2 slot; the phase-1 slots read are re-armed. Non-owner: phase-2 words of the owner -> output; re-armed."""
        p2 = ex[1]
        p1_local, p2_local, err, rank, rows_max, max_polls = (
            ex[3],
            ex[4],
            ex[7],
            ex[8],
            ex[9],
            ex[10],
        )
        p1_addr = p1_local.toint()
        p2_addr = p2_local.toint()
        slice_cols = hidden // world
        slice_words = slice_cols // 2
        slot = Int32(seq % cutlass.Uint32(K_SLOTS))
        load_atom, store_atom = _relaxed_sys_atoms()
        out_atom = cute.make_copy_atom(
            cute.nvgpu.CopyR2GOp(),
            cutlass.Uint32,
            num_bits_per_copy=AR_COPY_BITS,
            memory_order=cute.nvgpu.MemoryOrder.WEAK,
            memory_scope=cute.nvgpu.MemoryScope.CTA,
        )
        rank_values = cute.make_rmem_tensor(
            cute.make_layout((AR_VEC_WORDS, world)), cutlass.Uint32
        )
        rank_bf16 = cute.recast_tensor(rank_values, cutlass.BFloat16)
        result = cute.make_rmem_tensor(
            cute.make_layout((AR_VEC_SIZE,)), cutlass.BFloat16
        )
        result_words = cute.recast_tensor(result, cutlass.Uint32)
        empty1 = cute.make_rmem_tensor(
            cute.make_layout((AR_VEC_WORDS,)), cutlass.Uint32
        )
        empty1.fill(P1_EMPTY)
        empty2 = cute.make_rmem_tensor(
            cute.make_layout((AR_VEC_WORDS,)), cutlass.Uint32
        )
        empty2.fill(P2_EMPTY)
        got = cute.make_rmem_tensor(cute.make_layout((AR_VEC_WORDS,)), cutlass.Uint32)
        for p in cutlass.range_constexpr(iters):
            col = offset + p * threads * cluster * AR_VEC_SIZE
            owner = col // slice_cols
            word = (col - owner * slice_cols) // 2
            out_vec = cute.make_tensor(
                cute.make_ptr(
                    cutlass.Uint32,
                    (output.iterator + token * hidden + col).toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=AR_COPY_BITS // 8,
                ),
                cute.make_layout((AR_VEC_WORDS,)),
            )
            if owner == rank:
                # Every source's summand (this rank's included) from this rank's phase-1 slot; all probes are issued before
                # any load-dependent readiness test.
                polls = cutlass.Uint32(0)
                ready = False
                while not ready:
                    for src in cutlass.range_constexpr(world):
                        row = (
                            (slot * world + src) * rows_max + token
                        ) * slice_words + word
                        cute.copy(
                            load_atom, _vec_at(p1_addr, row), rank_values[None, src]
                        )
                    ready = True
                    for src in cutlass.range_constexpr(world):
                        for a in cutlass.range_constexpr(AR_VEC_WORDS):
                            ready = ready & (
                                rank_values[a, src] != cutlass.Uint32(P1_EMPTY)
                            )
                    if not ready:
                        polls += cutlass.Uint32(1)
                        if polls > cutlass.Uint32(K_SPIN_POLLS):
                            _nanosleep()
                        if polls > cutlass.Uint32(max_polls):
                            _fail(err, _err_code(ERR_TIMEOUT_PAYLOAD, 1, rank))
                # Reinterpret BF16 payloads and accumulate ranks in the original order.
                total = rank_bf16[None, 0].load().to(Float32)
                for src in cutlass.range_constexpr(1, world):
                    total = total + rank_bf16[None, src].load().to(Float32)
                result.store(total.to(cutlass.BFloat16))
                cute.copy(out_atom, result_words, out_vec)
                for v in cutlass.range_constexpr(AR_VEC_WORDS):
                    value = result_words[v]
                    if value == cutlass.Uint32(P2_EMPTY):
                        value = cutlass.Uint32(P2_NAN_PAIR)
                    result_words[v] = value
                for d in cutlass.range_constexpr(world):
                    if rank != d:
                        row = (
                            (slot * world + rank) * rows_max + token
                        ) * slice_words + word
                        cute.copy(store_atom, result_words, _vec_at(p2[d].toint(), row))
                for src in cutlass.range_constexpr(world):
                    row = ((slot * world + src) * rows_max + token) * slice_words + word
                    cute.copy(store_atom, empty1, _vec_at(p1_addr, row))
            else:
                row = ((slot * world + owner) * rows_max + token) * slice_words + word
                polls = cutlass.Uint32(0)
                ready = False
                while not ready:
                    cute.copy(load_atom, _vec_at(p2_addr, row), got)
                    ready = True
                    for a in cutlass.range_constexpr(AR_VEC_WORDS):
                        ready = ready & (got[a] != cutlass.Uint32(P2_EMPTY))
                    if not ready:
                        polls += cutlass.Uint32(1)
                        if polls > cutlass.Uint32(K_SPIN_POLLS):
                            _nanosleep()
                        if polls > cutlass.Uint32(max_polls):
                            _fail(err, _err_code(ERR_TIMEOUT_PAYLOAD, 2, rank))
                cute.copy(out_atom, got, out_vec)
                cute.copy(store_atom, empty2, _vec_at(p2_addr, row))

    def _get_sm_version(device=None) -> int:
        if not torch.cuda.is_available():
            return 80
        props = torch.cuda.get_device_properties(device)
        return props.major * 10 + props.minor

    # =================================================================
    # PTX Wrappers
    # =================================================================

    @dsl_user_op
    def _first_lane(mask, *, loc=None, ip=None):
        # Find the lowest set-bit lane in a nonzero mask.
        return Int32(
            llvm.inline_asm(
                T.i32(),
                [Int32(mask).ir_value(loc=loc, ip=ip)],
                "{ .reg .b32 rev; brev.b32 rev, $1; clz.b32 $0, rev; }",
                "=r,r",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @dsl_user_op
    def _store_dsmem(dst, bar, peer, a, b, c, d, *, loc=None, ip=None):
        # Store four FP32 sums to a peer CTA and complete 16 barrier bytes.
        # The caller includes self in the same DSMEM completion protocol.
        llvm.inline_asm(
            None,
            [
                dst.toint(loc=loc, ip=ip).ir_value(),
                bar.toint(loc=loc, ip=ip).ir_value(),
                peer.ir_value(),
                a.bitcast(Int32).ir_value(),
                b.bitcast(Int32).ir_value(),
                c.bitcast(Int32).ir_value(),
                d.bitcast(Int32).ir_value(),
            ],
            "{ .reg .b32 rd, rb; "
            "mapa.shared::cluster.u32 rd, $0, $2; "
            "mapa.shared::cluster.u32 rb, $1, $2; "
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 "
            "[rd], {$3, $4, $5, $6}, [rb]; }",
            "r,r,r,r,r,r,r,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

    @dsl_user_op
    def _wait_cluster_tx(bar, *, loc=None, ip=None):
        # Wait for phase-0 transactions and acquire visibility of DSMEM writes.
        llvm.inline_asm(
            None,
            [bar.toint(loc=loc, ip=ip).ir_value()],
            "{ .reg .pred ready; WAIT: "
            "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 ready, [$0], 0; "
            "@!ready bra WAIT; }",
            "r,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

    # =================================================================
    # Reduction helpers
    # =================================================================

    @cute.jit
    def _warp_sum(x: Float32, width: cutlass.Constexpr = 32):
        for i in cutlass.range_constexpr(5):
            if cutlass.const_expr((16 >> i) < width):
                x = x + cute.arch.shuffle_sync_bfly(x, offset=16 >> i)
        return x

    @cute.jit
    def _joint(s: cute.Tensor, num_valids: Int32):
        # Reduce 1-4 independent expert sums across a full warp, in place.
        # Input: each lane holds its partial sum for each expert in s[e].
        # Output: every lane holds sum_over_lanes(s[e]) for each valid expert.
        # Assign experts to lane residues (lane % 2 or lane % 4), reduce them
        # together, then broadcast from lanes 0..num_valids-1.
        lane = cute.arch.lane_idx()
        if num_valids == 1:
            s[0] = _warp_sum(s[0])
        else:
            a = Float32(0)
            b = Float32(0)
            if num_valids == 2:
                send = s[0] if lane & 1 else s[1]
                recv = cute.arch.shuffle_sync_bfly(send, offset=1)
                a = (s[1] if lane & 1 else s[0]) + recv
                for i in cutlass.range_constexpr(1, 5):
                    a = a + cute.arch.shuffle_sync_bfly(a, offset=1 << i)
            else:
                if num_valids == 3:
                    s[3] = Float32(0)
                send0 = s[0] if lane & 2 else s[2]
                send1 = s[1] if lane & 2 else s[3]
                a = (s[2] if lane & 2 else s[0]) + cute.arch.shuffle_sync_bfly(
                    send0, offset=2
                )
                b = (s[3] if lane & 2 else s[1]) + cute.arch.shuffle_sync_bfly(
                    send1, offset=2
                )
                send = a if lane & 1 else b
                recv = cute.arch.shuffle_sync_bfly(send, offset=1)
                a = (b if lane & 1 else a) + recv
                for i in cutlass.range_constexpr(2, 5):
                    a = a + cute.arch.shuffle_sync_bfly(a, offset=1 << i)
                s[2] = cute.arch.shuffle_sync(a, 2)
                s[3] = cute.arch.shuffle_sync(a, 3)
            s[0] = cute.arch.shuffle_sync(a, 0)
            s[1] = cute.arch.shuffle_sync(a, 1)

    # =================================================================
    # Configuration
    # =================================================================

    class MOEFinalizeRMSNormConfig:
        """Configuration for the MOE finalize + RMSNorm kernel.

        One token per block or cluster, all threads process the same row.
        Uses tiled copies vectorized up to 128 bits for global memory access.
        """

        COPY_BITS = 128

        def __init__(
            self,
            dtype: type,
            hidden_dim: int,
            hidden_dim_padded: int,
            threads_per_row=None,
            cluster_size=1,
        ):
            if cluster_size not in (1, 2, 4, 8, 16):
                raise ValueError("cluster_size must be 1, 2, 4, 8, or 16")
            self.dtype = dtype
            self.N = hidden_dim
            self.cluster_size = cluster_size
            self.threads_per_row, self.vec_size = self._compute_threads_per_row(
                hidden_dim, hidden_dim_padded, threads_per_row
            )
            self.num_threads = self.threads_per_row
            self.num_iters = hidden_dim // (
                self.num_threads * cluster_size * self.vec_size
            )
            self.warps_per_row = max(self.threads_per_row // 32, 1)

        def _compute_threads_per_row(
            self, N: int, hidden_dim_padded: int, threads_per_row=None
        ) -> tuple[int, int]:
            """Select compatible threads per row and vector width together."""
            # Initial preference based on dimension size
            if N <= 64:
                tpr = 8
            elif N <= 128:
                tpr = 16
            elif N <= 2048:
                tpr = 32
            elif N <= 2560:
                tpr = 64
            elif N <= 16384:
                tpr = 128
            else:
                tpr = 256
            if threads_per_row is not None:
                tpr = threads_per_row
            if tpr < 1 or tpr > 1024 or tpr & (tpr - 1):
                raise ValueError("threads_per_row must be a power of two up to 1024")
            # Select an aligned copy width; scalar copies handle odd row strides.
            while tpr >= (32 if self.cluster_size > 1 else 1):
                for vec_size in (8, 4, 2, 1):
                    if (
                        vec_size * self.dtype.width <= self.COPY_BITS
                        and N > 0
                        and N % (tpr * self.cluster_size * vec_size) == 0
                        and hidden_dim_padded % vec_size == 0
                    ):
                        return tpr, vec_size
                if threads_per_row is not None:
                    break
                tpr //= 2
            raise ValueError("No compatible thread/vector layout")

        def smem_size_in_bytes(self, topk_groups: int) -> int:
            """Shared memory for expert sums, transaction barriers, and warp sums."""
            return (
                topk_groups * self.cluster_size * self.warps_per_row * 16
                + topk_groups * 8
                + self.warps_per_row * 4
            )

    # =================================================================

    # =================================================================

    class Fin2FinalizeKernel:
        """CuTeDSL kernel: unpermute + RMSNorm + weighted accumulate, then the two-shot exchange (module docstring).

        Supports both weight-first (``rms_norm(weight*x)``) and standard
        (``rms_norm(x)*weight``) forms, parametrised by ``weight_first`` and
        ``eps`` so the kernel reproduces the same math as the Python
        ``MuseSpark1xRMSNorm`` modules.

        Uses tiled copies vectorized up to 128 bits for global memory access
        and FP32 register-level computation.
        """

        def __init__(
            self,
            dtype: type,
            num_tokens: int,
            hidden_dim: int,
            topk: int,
            hidden_dim_padded: int,
            threads_per_row=None,
            use_pdl=False,
            eps: float = 1e-8,
            weight_first: bool = True,
            cluster_size: int = 1,
            world: int = 2,
            max_blocks: int = 1,
            num_counters: int = 1,
        ):
            self.cfg = MOEFinalizeRMSNormConfig(
                dtype, hidden_dim, hidden_dim_padded, threads_per_row, cluster_size
            )
            self.max_blocks = max_blocks
            self.num_counters = num_counters
            self.cluster_size = cluster_size
            self.topk = topk
            self.hidden_dim_padded = hidden_dim_padded
            self.group_width = 4 if cluster_size > 1 else 1
            self.topk_groups = (topk + self.group_width - 1) // self.group_width
            self.use_pdl = use_pdl
            self.eps = float(eps)
            self.weight_first = bool(weight_first)
            self.world = world
            self.num_tokens = num_tokens
            self.is_persistent = num_tokens * cluster_size > max_blocks
            if self.is_persistent and cluster_size > 1:
                raise ValueError("Persistent finalize exchange requires cluster_size=1")
            if self.cfg.vec_size != AR_VEC_SIZE:
                raise ValueError(
                    "Finalize exchange requires eight BF16 values per lane"
                )
            if hidden_dim % (world * 2 * AR_VEC_SIZE) or world < 2:
                raise ValueError(
                    "Finalize exchange requires H divisible by 16 * world, world >= 2"
                )
            self.num_token_blocks = num_tokens * cluster_size
            if self.is_persistent:
                self.num_token_blocks = max_blocks
            if self.num_token_blocks + cluster_size > num_counters:
                raise ValueError("Not enough transfer counters for the grid")

        @cute.jit
        def __call__(
            self,
            gemm2_ptr: cute.Pointer,
            expert_weights_ptr: cute.Pointer,
            idx_map_ptr: cute.Pointer,
            rms_weight_ptr: cute.Pointer,
            out_ptr: cute.Pointer,
            num_tokens: Int32,
            stream: cuda.CUstream,
            ex,
        ):
            """Host function to launch the kernel.

            ar is None for standalone finalize, otherwise a tuple from
            FinalizePushWorkspace.arguments(): (peer Uint32 pointers, controller
            Uint32 pointer, rank, slot_words, local inbox Uint32 pointer).
            """
            cfg = self.cfg
            hidden_dim = cfg.N
            topk = self.topk
            hidden_dim_padded = self.hidden_dim_padded
            # Both paths specialize M; the runtime argument is retained for the ABI.
            num_tokens = self.num_tokens

            mGemm2 = cute.make_tensor(
                gemm2_ptr,
                cute.make_layout(
                    (num_tokens * topk, hidden_dim),
                    stride=(hidden_dim_padded, 1),
                ),
            )

            mOut = cute.make_tensor(
                out_ptr,
                cute.make_layout((num_tokens, hidden_dim), stride=(hidden_dim, 1)),
            )

            mW = cute.make_tensor(
                rms_weight_ptr,
                cute.make_layout((1, hidden_dim), stride=(hidden_dim, 1)),
            )

            mExpW = cute.make_tensor(
                expert_weights_ptr,
                cute.make_layout((num_tokens * topk,), stride=(1,)),
            )

            mIdx = cute.make_tensor(
                idx_map_ptr,
                cute.make_layout((num_tokens * topk,), stride=(1,)),
            )

            # Token blocks, then one cluster whose first thread releases the previous transfer's credits.
            num_blocks = self.num_token_blocks + self.cluster_size
            self.kernel(mGemm2, mOut, mW, mExpW, mIdx, ex).launch(
                grid=[num_blocks, 1, 1],
                block=[cfg.num_threads, 1, 1],
                cluster=[self.cluster_size, 1, 1],
                smem=cfg.smem_size_in_bytes(self.topk_groups),
                stream=stream,
                use_pdl=self.use_pdl,
            )

        @cute.kernel
        def kernel(
            self,
            mGemm2: cute.Tensor,
            mOut: cute.Tensor,
            mW: cute.Tensor,
            mExpW: cute.Tensor,
            mIdx: cute.Tensor,
            ex,
        ):
            """Device kernel using tiled copies vectorized up to 128 bits."""
            cfg = self.cfg
            topk = self.topk
            hidden_dim = cfg.N
            threads_per_row = cfg.threads_per_row
            warps_per_row = cfg.warps_per_row

            tidx = cute.arch.thread_idx()[0]
            lane = tidx % 32
            warp = tidx // 32
            cta_rank = cute.arch.block_idx()[0] % self.cluster_size
            token_start = cute.arch.block_idx()[0] // self.cluster_size
            smem = utils.SmemAllocator()
            # DSMEM inbox: [expert group, source CTA, source warp, 4 expert sums].
            partials = smem.allocate_tensor(
                Float32,
                cute.make_layout(
                    (self.topk_groups * self.cluster_size * warps_per_row * 4,)
                ),
                byte_alignment=16,
            )
            # One transaction-completion barrier per expert group; no phase reuse.
            bars = smem.allocate_tensor(
                cutlass.Uint64, cute.make_layout((self.topk_groups,)), byte_alignment=8
            )
            # C=1 scratch: one sum per warp for the current expert.
            warp_sums = smem.allocate_tensor(
                Float32, cute.make_layout((warps_per_row,)), byte_alignment=4
            )

            # ---- Initialize mbarriers when cluster size > 1 ----
            if cutlass.const_expr(self.cluster_size > 1):
                # Each source warp sends four FP32 sums (16 bytes) to every CTA.
                for g in range(tidx, self.topk_groups, threads_per_row):
                    cute.arch.mbarrier_init(bars.iterator + g, 1)
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        bars.iterator + g, self.cluster_size * warps_per_row * 16
                    )
                cute.arch.mbarrier_init_fence()
                cute.arch.barrier()
                # All peer barriers must be initialized before any DSMEM stores.
                cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()

            token_end = token_start + 1
            token_stride = 1
            token_iterations = 1
            if cutlass.const_expr(self.is_persistent):
                token_end = self.num_tokens
                token_stride = self.max_blocks
                token_iterations = (
                    token_end - token_start + token_stride - 1
                ) // token_stride
            # Preserve the host-validated alignment for vectorized copies.
            offset = cute.assume(
                (cta_rank * threads_per_row + tidx) * cfg.vec_size, divby=cfg.vec_size
            )
            layout = cute.make_layout(
                (cfg.vec_size, cfg.num_iters),
                stride=(1, threads_per_row * self.cluster_size * cfg.vec_size),
            )
            reg_layout = cute.make_layout((cfg.vec_size, cfg.num_iters))
            atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                cfg.dtype,
                num_bits_per_copy=cfg.vec_size * cfg.dtype.width,
            )
            norm = cute.make_rmem_tensor(reg_layout, cfg.dtype)
            # Read weights before PDL wait since weights are already fixed.
            cute.copy(atom, cute.make_tensor(mW.iterator + offset, layout), norm)
            # Read this block's transfer counter after the PDL dependency completes.
            block = cute.arch.block_idx()[0]
            counters = cute.make_tensor(ex[6], cute.make_layout((self.num_counters,)))
            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()
            seq = counters[block]
            n_iter = Int32(token_iterations)
            if block >= self.num_token_blocks:
                # Release block: transfers <= seq - 1 were consumed (and their slots re-armed) by earlier launches.
                if tidx == 0 and cta_rank == 0:
                    release_credits(ex, seq, self.world)
                n_iter = Int32(0)
            else:
                # Credits of this transfer's slot: one consumer per thread, then the block barrier.
                if tidx < self.world:
                    acquire_credits(ex, seq, tidx, self.world)
                cute.arch.barrier()
            for token_iteration in cutlass.range(n_iter):
                token_idx = token_start + token_iteration * token_stride

                acc = cute.make_rmem_tensor(reg_layout, Float32)
                acc.fill(0)
                input_frag = cute.make_rmem_tensor(
                    cute.make_layout((cfg.vec_size, cfg.num_iters, self.group_width)),
                    cfg.dtype,
                )
                scaled_frag = cute.make_rmem_tensor(
                    cute.make_layout((cfg.vec_size, cfg.num_iters, self.group_width)),
                    Float32,
                )
                sum_sq = cute.make_rmem_tensor(cute.make_layout((4,)), Float32)
                expert_weights = cute.make_rmem_tensor(
                    cute.make_layout((self.group_width,)), Float32
                )
                permuted_idxs = cute.make_rmem_tensor(
                    cute.make_layout((self.group_width,)), Int32
                )
                lane_row = Int32(-1)
                lane_weight = Float32(0)
                remaining = cutlass.Uint32(0)

                # Extract metadata parallelly for the current token.
                if cutlass.const_expr(topk <= min(32, threads_per_row)):
                    if lane < topk:
                        lane_row = mIdx[token_idx * topk + lane]
                        lane_weight = Float32(mExpW[token_idx * topk + lane])
                    remaining = cute.arch.vote_ballot_sync(lane_row >= 0)
                next_expert = Int32(0)
                # Iterate over all group of topK
                for group in cutlass.range_constexpr(self.topk_groups):
                    sum_sq.fill(0)
                    num_valids = Int32(0)
                    # Find valid experts in the current group.
                    for e in cutlass.range_constexpr(self.group_width):
                        row = Int32(-1)
                        route = Float32(0)
                        if cutlass.const_expr(topk <= min(32, threads_per_row)):
                            source = (
                                _first_lane(remaining) if remaining != 0 else Int32(0)
                            )
                            permuted_idx = cute.arch.shuffle_sync(lane_row, source)
                            expert_weight = cute.arch.shuffle_sync(lane_weight, source)
                            if remaining != 0:
                                row = permuted_idx
                                route = expert_weight
                            remaining = remaining & (remaining - cutlass.Uint32(1))
                        else:
                            searching = True
                            while next_expert < topk and searching:
                                candidate = mIdx[token_idx * topk + next_expert]
                                if candidate >= 0:
                                    row = candidate
                                    route = Float32(
                                        mExpW[token_idx * topk + next_expert]
                                    )
                                    searching = False
                                next_expert += 1
                        permuted_idxs[e] = row
                        expert_weights[e] = route
                        if row >= 0:
                            num_valids += 1
                            input_offset = cute.assume(
                                Int64(row) * self.hidden_dim_padded + Int64(offset),
                                divby=cfg.vec_size,
                            )
                            cute.copy(
                                atom,
                                cute.make_tensor(
                                    mGemm2.iterator + input_offset, layout
                                ),
                                input_frag[None, None, e],
                            )
                    # Load the whole group before consuming any expert's values.
                    for e in cutlass.range_constexpr(self.group_width):
                        if permuted_idxs[e] >= 0:
                            sq = Float32(0)
                            for p in cutlass.range_constexpr(cfg.num_iters):
                                for v in cutlass.range_constexpr(cfg.vec_size):
                                    value = Float32(input_frag[v, p, e])
                                    if cutlass.const_expr(self.weight_first):
                                        value = value * Float32(norm[v, p])
                                    scaled_frag[v, p, e] = value
                                    sq = cute.fma(value, value, sq)
                            sum_sq[e] = sq
                    if num_valids > 0:
                        if cutlass.const_expr(self.cluster_size > 1):
                            # ---- Reduce within each warp, then publish to all CTAs ----
                            _joint(sum_sq, num_valids)
                            if lane < self.cluster_size:
                                dst = (
                                    partials.iterator
                                    + (
                                        group * self.cluster_size * warps_per_row
                                        + cta_rank * warps_per_row
                                        + warp
                                    )
                                    * 4
                                )
                                _store_dsmem(
                                    dst,
                                    bars.iterator + group,
                                    lane,
                                    sum_sq[0],
                                    sum_sq[1],
                                    sum_sq[2],
                                    sum_sq[3],
                                )
                            # ---- Wait for the inbox, then reduce all CTA/warp sums ----
                            _wait_cluster_tx(bars.iterator + group)
                            sum_sq.fill(0)
                            for slot in range(
                                lane, self.cluster_size * warps_per_row, 32
                            ):
                                for e in cutlass.range_constexpr(self.group_width):
                                    if e < num_valids:
                                        sum_sq[e] += partials[
                                            (
                                                group
                                                * self.cluster_size
                                                * warps_per_row
                                                + slot
                                            )
                                            * 4
                                            + e
                                        ]
                            _joint(sum_sq, num_valids)
                        else:
                            # ---- Single-CTA reduction through warp_sums ----
                            s = _warp_sum(sum_sq[0], min(threads_per_row, 32))
                            if lane == 0:
                                warp_sums[warp] = s
                            cute.arch.barrier()
                            s = Float32(0)
                            if lane < warps_per_row:
                                s = warp_sums[lane]
                            sum_sq[0] = _warp_sum(s, min(threads_per_row, 32))
                            cute.arch.barrier()
                        # ---- Normalize each expert and accumulate router-weighted output ----
                        for e in cutlass.range_constexpr(self.group_width):
                            if permuted_idxs[e] >= 0:
                                inv = cute.math.rsqrt(
                                    sum_sq[e] / hidden_dim + Float32(self.eps),
                                    fastmath=True,
                                )
                                for p in cutlass.range_constexpr(cfg.num_iters):
                                    for v in cutlass.range_constexpr(cfg.vec_size):
                                        value = scaled_frag[v, p, e] * inv
                                        if cutlass.const_expr(not self.weight_first):
                                            value = value * Float32(norm[v, p])
                                        acc[v, p] = (
                                            acc[v, p] + expert_weights[e] * value
                                        )
                # Phase 1: every column slice to its owner's phase-1 slot (this rank's own slice included).
                push_slice(
                    acc,
                    ex,
                    token_idx,
                    offset,
                    threads_per_row,
                    hidden_dim,
                    self.cluster_size,
                    self.world,
                    seq,
                )

            # ---- PDL: Trigger next kernel here to increase overlap
            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_launch_dependents()
            # Phase 2, after every token of this block was pushed: reduce the own slice and share it; collect the others.
            for token_iteration in cutlass.range(n_iter):
                token_idx = token_start + token_iteration * token_stride
                reduce_slice(
                    mOut,
                    ex,
                    token_idx,
                    offset,
                    threads_per_row,
                    hidden_dim,
                    self.cluster_size,
                    self.world,
                    cfg.num_iters,
                    seq,
                )

            # Keep shared memory alive until peer DSMEM accesses finish.
            if cutlass.const_expr(self.cluster_size > 1):
                cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()
            # This block's counter, then the counters of block indices this launch does not use.
            next_seq = seq + cutlass.Uint32(1)
            if tidx == 0:
                counters[block] = next_seq
            global_thread = block * threads_per_row + tidx
            for counter in range(
                cute.arch.grid_dim()[0] + global_thread,
                self.num_counters,
                cute.arch.grid_dim()[0] * threads_per_row,
            ):
                counters[counter] = next_seq

    # =================================================================
    # Compilation cache
    # =================================================================

    _compile_cache: dict = {}

    def get_compiled_fin2(
        dtype: type,
        num_tokens: int,
        hidden_dim: int,
        topk: int,
        hidden_dim_padded: int,
        stream: cuda.CUstream,
        threads_per_row=None,
        use_pdl=False,
        eps: float = 1e-8,
        weight_first: bool = True,
        expert_weights_dtype: type = cutlass.BFloat16,
        cluster_size: int = 1,
        world: int = 2,
        max_blocks: int = 1,
        num_counters: int = 1,
    ):
        if num_tokens <= 0:
            raise ValueError("Finalize requires a positive token count")
        key = (
            dtype,
            hidden_dim,
            topk,
            hidden_dim_padded,
            threads_per_row,
            use_pdl,
            eps,
            weight_first,
            expert_weights_dtype,
            cluster_size,
            world,
            max_blocks,
            num_counters,
            num_tokens,
        )
        if key not in _compile_cache:
            kernel_obj = Fin2FinalizeKernel(
                dtype,
                num_tokens,
                hidden_dim,
                topk,
                hidden_dim_padded,
                threads_per_row,
                use_pdl=use_pdl,
                eps=eps,
                weight_first=weight_first,
                cluster_size=cluster_size,
                world=world,
                max_blocks=max_blocks,
                num_counters=num_counters,
            )
            compiled = cute.compile(
                kernel_obj,
                make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(
                    expert_weights_dtype,
                    16,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                make_ptr(
                    cutlass.Int32,
                    16,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
                Int32(1),  # num_tokens
                stream,
                make_fin2_compile_args(world),
                options="--enable-tvm-ffi",
            )
            _compile_cache[key] = compiled
        return _compile_cache[key]
