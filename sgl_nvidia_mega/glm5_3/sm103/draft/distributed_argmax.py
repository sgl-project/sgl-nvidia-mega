# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Distributed first-maximum argmax over a vocab-parallel logits shard.

Every rank of a tensor-parallel group holds ``logits[M, V_local]`` (bf16, the rank's
contiguous slice of the vocab). One launch per rank returns, on every rank, the global
token id and the fp32 value of the first maximum of each row of the gathered ``[M, W *
V_local]`` logits, i.e. ``torch.argmax`` over the gathered rows:

* NaN ranks above every number and the first NaN wins (``nan_to_neg = 0``), or every NaN
  is first replaced by a given finite value (``nan_to_neg = 1``, the rule of
  ``torch.nan_to_num(nan=-1e30)`` on a bf16 tensor);
* ties (including -0 vs +0) go to the lowest global index; -0 is reported as +0;
* columns at or beyond ``n_valid`` (vocab padding) are treated as -inf.

Protocol (one grid of ``max_rows`` CTAs per rank, CTA ``m`` serves row ``m``):

* each CTA reduces its row slice to one candidate and publishes it as a 64-bit packet
  ``value_bits << 32 | (epoch & 0x3FFF) << 18 | global_index`` with one relaxed
  system-scope store into its slot of every peer's mailbox (unicast); idle rows (``m >=
  M``) still publish, so no slot holds a packet older than two epochs and the 14-bit tag
  cannot alias;
* the mailbox is ``[2][W][max_rows]`` u64 in symmetric memory, double-buffered by epoch
  parity; one thread per CTA executes ``fence.sc.sys`` after the grid-dependency wait
  and before the CTA's packet stores, which orders a writer's reuse of a parity slot
  after the reader's previous read of it;
* each active CTA polls its W slots with relaxed system-scope loads until every slot
  carries the current epoch's tag, bounded by a timeout: on expiry the sticky error word
  is set, the host-mapped error record (if given) names this rank, the late peer, the
  row count and the epoch, and the launch traps (fail-stop);
* the per-CTA epoch lives in a control word on the local GPU, read after the wait and
  advanced at exit; there is no early ``launch_dependents``, so dependents start after
  this grid completes;
* the grid is launched cooperatively, so all CTAs of a rank are co-resident; no CTA
  waits for another CTA of its own grid.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32, Int64, Uint32, Uint64

from ..primitives import ptx_core as core
from ..primitives import ptx_sys as sys_ops
from . import _draft_ptx as ptx

_NEG_INF_BITS = 0xFF800000
_CANONICAL_NAN_BITS = 0x7FC00000
_INT32_MAX = 0x7FFFFFFF


class DistributedArgmax:
    """One instance per row bucket (``max_rows`` in {4, 8, 16}); each instance has its
    own mailbox and epochs.
    """

    THREADS = 1024
    LOADER_THREADS = 992  # warps 0..30 load and reduce; warp 31 issues the fence
    FENCE_THREAD = 992
    VECTORS_PER_THREAD = 3  # 16-byte vectors per loader thread: V_local <= 3 * 8 * 992
    MAX_RANKS = 8
    # [0] sticky error word, [1] timeouts, [2..4] reserved, [16 + m] per-CTA epochs
    CTRL_WORDS = 64
    EPOCH_WORD = 16
    TAG_MASK = 0x3FFF
    INDEX_MASK = 0x3FFFF
    TIMEOUT_NS = 10_000_000_000
    ERR_TIMEOUT = 1
    ERR_BAD_INDEX = 4
    # host-mapped error record (the hop transport's layout): {magic, kind, site,
    # rank | peer << 8 | rows << 16, epoch}
    ERR_RECORD_MAGIC = 0x48455252
    ERR_RECORD_KIND = 4  # top-1 poll

    def __init__(self, max_rows: int):
        if max_rows not in (4, 8, 16):
            raise ValueError(f"max_rows must be 4, 8 or 16, got {max_rows}")
        self.max_rows = max_rows

    # ---- host contract

    @classmethod
    def max_local_vocab(cls) -> int:
        return cls.VECTORS_PER_THREAD * 8 * cls.LOADER_THREADS

    def mailbox_words(self, world: int) -> int:
        return 2 * world * self.max_rows

    def check_shapes(
        self, rows: int, local_vocab: int, n_valid: int, world: int, rank: int
    ) -> None:
        if not 1 <= world <= self.MAX_RANKS or not 0 <= rank < world:
            raise ValueError(f"world {world} / rank {rank} out of range")
        if not 1 <= rows <= self.max_rows:
            raise ValueError(f"rows {rows} not in [1, {self.max_rows}]")
        if local_vocab < 8 or local_vocab % 8 or local_vocab > self.max_local_vocab():
            raise ValueError(
                f"local vocab {local_vocab}: a multiple of 8 in "
                f"[8, {self.max_local_vocab()}]"
            )
        if world * local_vocab >= 1 << 18:
            raise ValueError("world * local vocab must be < 2^18 (18-bit global index)")
        if not 1 <= n_valid <= local_vocab:
            raise ValueError(f"n_valid {n_valid} not in [1, {local_vocab}]")

    # ---- launch

    @cute.jit
    def __call__(
        self,
        logits: Int64,
        mailbox: Int64,
        peers: Int64,
        ctrl: Int64,
        out_ids: Int64,
        out_vals: Int64,
        rows: Int32,
        local_vocab: Int32,
        n_valid: Int32,
        world: Int32,
        rank: Int32,
        timeout_ns: Uint64,
        nan_to_neg: Int32,
        nan_replacement_bits: Uint32,
        err_host: Int64,
        site: Int32,
        stream: cuda.CUstream,
    ):
        """``err_host``: device address of a host-mapped error record (0: none);
        ``site``: the caller's id for this call site, written into the record."""
        self.kernel(
            logits,
            mailbox,
            peers,
            ctrl,
            out_ids,
            out_vals,
            rows,
            local_vocab,
            n_valid,
            world,
            rank,
            timeout_ns,
            nan_to_neg,
            nan_replacement_bits,
            err_host,
            site,
        ).launch(
            grid=[self.max_rows, 1, 1],
            block=[self.THREADS, 1, 1],
            stream=stream,
            use_pdl=True,
            cooperative=True,
        )

    # ---- decision points

    def better(self, v, i, best, best_i):
        """True iff (v, i) precedes (best, best_i) in argmax order: NaN first, then
        larger, then lower index.
        """
        v_nan = ptx.is_nan(v)
        v_num = ptx.bits_from_f32(v) & Uint32(0x7FFFFFFF) <= Uint32(0x7F800000)
        b_num = ptx.bits_from_f32(best) & Uint32(0x7FFFFFFF) <= Uint32(0x7F800000)
        lower = i < best_i
        numeric = (v > best) | ((v == best) & lower)
        return (v_nan & (b_num | lower)) | (v_num & b_num & numeric)

    @cute.jit
    def order_publish(self):
        """Orders this rank's earlier reads of the parity slots before this CTA's packet
        stores.
        """
        sys_ops.fence_sc_sys()

    def accept(self, tag, epoch):
        """True iff a polled packet's tag belongs to this call."""
        return tag == (epoch & Uint32(self.TAG_MASK))

    @cute.jit
    def on_timeout(self, ctrl: Int64, err_host: Int64, site, rank, peer, rows, epoch):
        """Sticky error word and count, the host-mapped record, then ``trap``."""
        sys_ops.atom_or_b32(ctrl, Uint32(self.ERR_TIMEOUT))
        sys_ops.atom_add_u32(ctrl + Int64(4), Uint32(1))
        if err_host != Int64(0):
            meta = (
                Uint32(rank)
                | (Uint32(peer) << Uint32(8))
                | (Uint32(rows) << Uint32(16))
            )
            sys_ops.st_relaxed_sys_v4_u32(
                err_host,
                Uint32(self.ERR_RECORD_MAGIC),
                Uint32(self.ERR_RECORD_KIND),
                Uint32(site),
                meta,
            )
            sys_ops.st_relaxed_sys_u32(err_host + Int64(16), epoch)
        sys_ops.fence_sc_sys()
        sys_ops.trap()

    # ---- helpers

    def canonical_bits(self, v):
        bits = ptx.bits_from_f32(v)
        bits = Uint32(
            cutlass.select_((bits & Uint32(0x7FFFFFFF)) == Uint32(0), Uint32(0), bits)
        )
        return Uint32(cutlass.select_(ptx.is_nan(v), Uint32(_CANONICAL_NAN_BITS), bits))

    # ---- kernel

    @cute.kernel
    def kernel(
        self,
        logits: Int64,
        mailbox: Int64,
        peers: Int64,
        ctrl: Int64,
        out_ids: Int64,
        out_vals: Int64,
        rows: Int32,
        local_vocab: Int32,
        n_valid: Int32,
        world: Int32,
        rank: Int32,
        timeout_ns: Uint64,
        nan_to_neg: Int32,
        nan_replacement_bits: Uint32,
        err_host: Int64,
        site: Int32,
    ):
        t, _, _ = cute.arch.thread_idx()
        m, _, _ = cute.arch.block_idx()
        tid = Int32(t)
        row = Int32(m)

        core.pdl_wait()
        if tid == self.FENCE_THREAD:
            self.order_publish()
        epoch_addr = ctrl + Int64(self.EPOCH_WORD + row) * Int64(4)
        epoch = sys_ops.ld_volatile_u32(epoch_addr)
        poisoned = sys_ops.ld_volatile_u32(ctrl) != Uint32(0)

        warp_vals = ptx.static_smem(Float32, 32)
        warp_ids = ptx.static_smem(Int32, 32)
        rank_vals = ptx.static_smem(Float32, self.MAX_RANKS)
        rank_ids = ptx.static_smem(Int32, self.MAX_RANKS)
        rank_bad = ptx.static_smem(Int32, self.MAX_RANKS)

        active = row < rows
        best = ptx.f32_from_bits(Uint32(_NEG_INF_BITS))
        best_i = Int32(_INT32_MAX)
        replacement = ptx.f32_from_bits(nan_replacement_bits)
        neg_inf = ptx.f32_from_bits(Uint32(_NEG_INF_BITS))

        # ---- local first maximum over this thread's 16-byte vectors
        if active & (tid < self.LOADER_THREADS):
            row_addr = logits + Int64(row) * Int64(local_vocab) * Int64(2)
            n_vec = local_vocab // 8
            words = []
            for k in cutlass.range_constexpr(self.VECTORS_PER_THREAD):
                v = tid + k * self.LOADER_THREADS
                words.append(
                    core.ld_global_v4_if(row_addr + Int64(v) * Int64(16), v < n_vec)
                )
            for k in cutlass.range_constexpr(self.VECTORS_PER_THREAD):
                v = tid + k * self.LOADER_THREADS
                in_row = v < n_vec
                for e in cutlass.range_constexpr(8):
                    w = words[k][e // 2]
                    f = ptx.bf16_low(w) if e % 2 == 0 else ptx.bf16_high(w)
                    i = v * 8 + e
                    f = Float32(
                        cutlass.select_(
                            (nan_to_neg != Int32(0)) & ptx.is_nan(f), replacement, f
                        )
                    )
                    f = Float32(cutlass.select_(i >= n_valid, neg_inf, f))
                    take = in_row & self.better(f, i, best, best_i)
                    best = Float32(cutlass.select_(take, f, best))
                    best_i = Int32(cutlass.select_(take, i, best_i))

        # ---- warp reduction (fixed butterfly order)
        for s in cutlass.range_constexpr(5):
            off = 16 >> s
            other = cute.arch.shuffle_sync_bfly(best, off)
            other_i = cute.arch.shuffle_sync_bfly(best_i, off)
            take = self.better(other, other_i, best, best_i)
            best = Float32(cutlass.select_(take, other, best))
            best_i = Int32(cutlass.select_(take, other_i, best_i))
        if tid % 32 == 0:
            warp_vals[tid // 32] = best
            warp_ids[tid // 32] = best_i
        peer_base = Uint64(0)
        if tid < world:
            peer_base = sys_ops.ld_u64(peers + Int64(tid) * Int64(8))
        cute.arch.sync_threads()

        buf_off = Int64(epoch & Uint32(1)) * Int64(world) * Int64(self.max_rows)
        tag = Uint64(epoch & Uint32(self.TAG_MASK)) << Uint64(18)

        # ---- publish: lane 0 of warp 0 builds the packet, lanes 0..W-1 store it to
        # their peer
        if tid < 32:
            packet = Uint64(0)
            if tid == 0:
                if active:
                    for w in cutlass.range_constexpr(1, 32):
                        wv = warp_vals[w]
                        wi = warp_ids[w]
                        take = self.better(wv, wi, best, best_i)
                        best = Float32(cutlass.select_(take, wv, best))
                        best_i = Int32(cutlass.select_(take, wi, best_i))
                    gidx = Uint32(rank) * Uint32(local_vocab) + Uint32(best_i)
                    if (best_i < Int32(0)) | (best_i >= local_vocab):
                        sys_ops.atom_or_b32(ctrl, Uint32(self.ERR_BAD_INDEX))
                        gidx = Uint32(0)
                    packet = (
                        (Uint64(self.canonical_bits(best)) << Uint64(32))
                        | tag
                        | Uint64(gidx & Uint32(self.INDEX_MASK))
                    )
                else:
                    packet = (Uint64(_NEG_INF_BITS) << Uint64(32)) | tag
            lo = cute.arch.shuffle_sync(Uint32(packet & Uint64(0xFFFFFFFF)), 0)
            hi = cute.arch.shuffle_sync(Uint32(packet >> Uint64(32)), 0)
            packet = (Uint64(hi) << Uint64(32)) | Uint64(lo)
            if tid < world:
                slot = buf_off + Int64(rank) * Int64(self.max_rows) + Int64(row)
                sys_ops.st_relaxed_sys_u64(Int64(peer_base) + slot * Int64(8), packet)

        # ---- gather: lane s polls the slot rank s wrote into this rank's mailbox
        if active & (tid < world):
            slot_addr = mailbox + (
                buf_off + Int64(tid) * Int64(self.max_rows) + Int64(row)
            ) * Int64(8)
            got = Uint64(0)
            timed_out = Int32(0)
            if poisoned:
                timed_out = Int32(1)
            else:
                got = sys_ops.ld_relaxed_sys_u64(slot_addr)
                if not self.accept(
                    Uint32((got >> Uint64(18)) & Uint64(self.TAG_MASK)), epoch
                ):
                    start = sys_ops.globaltimer()
                    done = Boolean(False)
                    while not done:
                        got = sys_ops.ld_relaxed_sys_u64(slot_addr)
                        if self.accept(
                            Uint32((got >> Uint64(18)) & Uint64(self.TAG_MASK)), epoch
                        ):
                            done = Boolean(True)
                        else:
                            if sys_ops.globaltimer() - start > timeout_ns:
                                self.on_timeout(
                                    ctrl, err_host, site, rank, tid, rows, epoch
                                )
                                timed_out = Int32(1)
                                done = Boolean(True)
            rank_vals[tid] = ptx.f32_from_bits(Uint32(got >> Uint64(32)))
            rank_ids[tid] = Int32(got & Uint64(self.INDEX_MASK))
            rank_bad[tid] = timed_out
        cute.arch.sync_threads()

        # ---- combine the W candidates in rank order
        if active & (tid == 0):
            val = rank_vals[0]
            idx = rank_ids[0]
            bad = rank_bad[0]
            for s in cutlass.range(1, world, unroll=1):
                bad = bad | rank_bad[s]
                sv = rank_vals[s]
                si = rank_ids[s]
                take = self.better(sv, si, val, idx)
                val = Float32(cutlass.select_(take, sv, val))
                idx = Int32(cutlass.select_(take, si, idx))
            if bad != Int32(0):
                idx = Int32(0)
                val = ptx.f32_from_bits(Uint32(_CANONICAL_NAN_BITS))
            core.st_global_u64(out_ids + Int64(row) * Int64(8), Uint64(Int64(idx)))
            core.st_global_f32(out_vals + Int64(row) * Int64(4), val)
        if tid == 0:
            core.st_global_u32(epoch_addr, epoch + Uint32(1))
