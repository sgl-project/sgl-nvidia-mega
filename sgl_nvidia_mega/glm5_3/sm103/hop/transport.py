# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Epoch-tagged mailbox transport of the tensor-parallel hop kernels.

Protocol (one-shot push all-reduce, unicast only, no re-arm):

* Records. A row of ``HIDDEN`` bf16 values is ``NREC`` 64-bit records ``{bf16[3j],
  bf16[3j + 1], bf16[3j + 2], tag}``. A producer writes each record with one
  ``st.relaxed.sys`` (element of a ``.v2.b64`` vector) into every rank's copy; a
  consumer reads its local copy with ``ld.relaxed.sys`` and accepts a record only if its
  tag equals the current row epoch modulo 2^16. Each 64-bit element of a vector access
  is a single-copy-atomic scalar access (PTX ISA, vector data types), so an accepted
  record is exactly the one written in that epoch.
* Epochs. A per-rank global epoch ``E`` advances once per hop; per-row epochs
  ``E_row[m]`` advance only in hops with more than ``m`` rows and select the buffer slot
  (``E_row[m] mod NBUF``) and the tag, so tags never alias whatever the row count does.
  Every hop kernel reads them after ``griddepcontrol.wait``; the last CTA to arrive
  (``atom.acq_rel.gpu`` counter) stores the new row epochs (``st.relaxed.gpu``) and then
  ``E + 1`` (``st.release.gpu``).
* Buffer reuse. A producer may write slot ``e mod NBUF`` only after acquiring
  (``ld.acquire.sys``, then a CTA barrier) an acknowledgement ``ack[d] >= e - NBUF``
  from every rank ``d``. Every ``KACK``-th epoch one thread of an extra CTA releases
  ``fence.acq_rel.sys`` + ``st.relaxed.sys ack[rank] = e - 1`` to every rank.
* Bounded waits. Every cross-GPU wait reads ``%globaltimer`` once per 16 polls; past its
  bound it sets the local sticky error word, writes an error record to host-mapped
  memory and executes ``trap``. If the error word is already set when a hop starts, the
  hop publishes NaN and uses a short bound (poisoned lock step), so a failure stops
  every rank instead of hanging.

The device functions take an ``Mbox`` view: the eight peer bases rotated so that index 0
is this rank's own mailbox (stores are staggered starting from it), this rank, and the
record-area size of the instance.
"""

from dataclasses import dataclass
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
from cutlass import Float32, Int32, Int64, Uint32, Uint64

from ..primitives import ptx_core as C
from ..primitives import ptx_sys as sys_ops
from . import _ptx_local as P
from .constants import (
    CTRL_BYTES,
    ERR_ACK_WAIT,
    ERR_KIND_NAMES,
    ERR_RECORD_POLL,
    HIDDEN,
    KACK,
    NBUF,
    NREC,
    POISONED_TIMEOUT_NS,
    POLL_TIMER_STRIDE,
    TAG_MASK,
    TIMEOUT_NS,
    W_ACK,
    W_ARRIVE,
    W_E,
    W_EROW,
    W_ERR,
    WORLD,
    mailbox_bytes,
    record_bytes,
)

# host-mapped error record: [magic, kind, site, rank | peer << 8 | row << 16, epoch,
# ...]
ERR_RECORD_WORDS = 8
ERR_MAGIC = 0x48455252


@dataclass(frozen=True)
class Mbox:
    """Device-side view of one mailbox instance (built inside a kernel)."""

    peers: tuple  # 8 x Int64, rotated: peers[i] = base of rank (rank + i) mod 8
    rank: Int32
    max_rows: int  # rows of the instance (Constexpr)
    err_host: Int64  # host-mapped error record (device address)
    site: Int32  # caller's site id, copied into the error record

    @property
    def rec_bytes(self) -> int:
        return record_bytes(self.max_rows)

    def ctrl(self, word):
        """Address of control word ``word`` of this rank's own mailbox."""
        return self.peers[0] + Int64(self.rec_bytes + 4 * word)

    def rec_off(self, buf, src, m, j):
        return (
            ((Int64(buf) * WORLD + Int64(src)) * self.max_rows + Int64(m)) * NREC
            + Int64(j)
        ) * 8


@dataclass(frozen=True)
class EpochSmem:
    """Shared-memory epoch snapshot of one CTA: ``e``, the poison flag and the row
    epochs (``max_rows`` words)."""

    e: Any
    pz: Any
    erow: Any

    @staticmethod
    def allocate(smem, max_rows):
        return EpochSmem(
            smem.allocate_tensor(Uint32, cute.make_layout(1), 4),
            smem.allocate_tensor(Int32, cute.make_layout(1), 4),
            smem.allocate_tensor(Uint32, cute.make_layout(max_rows), 4),
        )


class Transport:
    """Protocol device functions. Every protocol decision point is its own method so
    that a test subclass can override it (fault injection); ``delay`` is empty in the
    served classes."""

    timeout_ns = TIMEOUT_NS
    poisoned_timeout_ns = POISONED_TIMEOUT_NS

    # ------------------------------------------------------- decision points
    @cute.jit
    def delay(self, mb: Mbox, site: cutlass.Constexpr[int]):
        pass

    def tag_ok(self, rec, tag):
        return (Uint64(rec) >> 48) == Uint64(tag)

    def buffer_of(self, e):
        return Uint32(e) % Uint32(NBUF)

    def ack_value(self, e):
        return Uint32(e) - Uint32(1)

    def check_ack_enabled(self) -> bool:
        return True

    def release_ack_enabled(self) -> bool:
        return True

    @cute.jit
    def release_fence(self):
        sys_ops.fence_acq_rel_sys()

    # ------------------------------------------------------------- failure
    @cute.jit
    def fail(
        self,
        mb: Mbox,
        kind: cutlass.Constexpr[int],
        code: cutlass.Constexpr[int],
        peer,
        row,
        epoch,
    ):
        """Sticky error word, the host-mapped error record, then ``trap``. ``row``: the
        row the timed-out poll was reading (0 for an acknowledgement wait); ``epoch``:
        the awaited epoch (a record poll stores its 16-bit tag)."""
        sys_ops.atom_or_b32(mb.ctrl(W_ERR), Uint32(code))
        meta = Uint32(mb.rank) | (Uint32(peer) << 8) | (Uint32(row) << 16)
        sys_ops.st_relaxed_sys_v4_u32(
            mb.err_host, Uint32(ERR_MAGIC), Uint32(kind), Uint32(mb.site), meta
        )
        sys_ops.st_relaxed_sys_u32(mb.err_host + Int64(16), Uint32(epoch))
        sys_ops.fence_sc_sys()
        sys_ops.trap()

    # --------------------------------------------------------------- epochs
    @cute.jit
    def epoch_begin(
        self,
        mb: Mbox,
        es: EpochSmem,
        check: cutlass.Constexpr[bool],
        rt0: cutlass.Constexpr[int] = 0,
    ) -> Uint32:
        """Every thread, after ``griddepcontrol.wait``. Threads ``rt0 .. rt0 + 7`` read
        ``E`` and one row epoch each (rows ``max_rows`` > 8 use threads ``rt0 .. rt0 +
        max_rows - 1``), thread ``rt0`` the error word, and (``check``) acquire the
        acknowledgement of rank ``t - rt0``; the CTA barrier publishes the snapshot."""
        t = P.thread_index() - rt0
        nread: cutlass.Constexpr = max(WORLD, mb.max_rows)
        if t >= 0 and t < nread:
            e = sys_ops.ld_relaxed_gpu_u32(mb.ctrl(W_E))
            er = Uint32(0)
            if t < mb.max_rows:
                er = sys_ops.ld_relaxed_gpu_u32(mb.ctrl(W_EROW) + Int64(4) * Int64(t))
            pz = Uint32(0)
            if t == 0:
                pz = sys_ops.ld_relaxed_gpu_u32(mb.ctrl(W_ERR))
            a = Uint32(0)
            if cutlass.const_expr(check and self.check_ack_enabled()):  # noqa: SIM102
                if t < WORLD:
                    a = sys_ops.ld_acquire_sys_u32(
                        mb.ctrl(W_ACK) + Int64(4 * 32) * Int64(t)
                    )
            if t < mb.max_rows:
                es.erow[t] = er
            if t == 0:
                es.e[0] = e
                es.pz[0] = Int32(pz)
            if cutlass.const_expr(check and self.check_ack_enabled()):  # noqa: SIM102
                if t < WORLD:
                    need = e - Uint32(NBUF)
                    if Int32(a - need) < 0:
                        self.delay(mb, 5)
                        t0 = Uint64(0)
                        it = Uint32(0)
                        while Int32(a - need) < 0:
                            a = sys_ops.ld_acquire_sys_u32(
                                mb.ctrl(W_ACK) + Int64(4 * 32) * Int64(t)
                            )
                            it = it + Uint32(1)
                            if (it & Uint32(POLL_TIMER_STRIDE - 1)) == Uint32(0):
                                now = sys_ops.globaltimer()
                                if t0 == Uint64(0):
                                    t0 = now
                                elif now - t0 > Uint64(self.timeout_ns):
                                    self.fail(mb, ERR_ACK_WAIT, 2, t, 0, e)
        cute.arch.sync_threads()
        return es.e[0]

    @cute.jit
    def epoch_arrive(
        self,
        mb: Mbox,
        es: EpochSmem,
        rows,
        total,
        wait_prod: cutlass.Constexpr[bool] = False,
    ):
        """One thread per CTA of an epoch-advancing hop, after the barrier that ordered
        this CTA's reads of ``E`` / ``E_row``. The last arriver (of ``total`` CTAs)
        stores the next row epochs of the ``rows`` rows of this hop, then ``E + 1``.
        ``wait_prod``: the consumer of a fenced-trigger o_proj hop may start before a
        slow producer CTA has read ``E``, so the last arriver first waits for the
        producer grid (``griddepcontrol.wait``)."""
        old = sys_ops.atom_add_acq_rel_gpu_u32(mb.ctrl(W_ARRIVE), Uint32(1))
        if old == Uint32(total) - Uint32(1):
            if cutlass.const_expr(wait_prod):
                self.wait_producer()
            sys_ops.st_relaxed_gpu_u32(mb.ctrl(W_ARRIVE), Uint32(0))
            for m in range(rows):
                sys_ops.st_relaxed_gpu_u32(
                    mb.ctrl(W_EROW) + Int64(4) * Int64(m), es.erow[m] + Uint32(1)
                )
            sys_ops.st_release_gpu_u32(mb.ctrl(W_E), es.e[0] + Uint32(1))

    @cute.jit
    def wait_producer(self):
        cute.arch.griddepcontrol_wait()

    @cute.jit
    def ack_cta(self, mb: Mbox, e: Uint32):
        """Thread 0 of the acknowledgement CTA: every ``KACK``-th epoch, release "every
        read of epochs <= e - 1 on this rank is done" to every rank."""
        if cutlass.const_expr(self.release_ack_enabled()):  # noqa: SIM102
            if P.thread_index() == 0:  # noqa: SIM102
                if (e % Uint32(KACK)) == Uint32(0):
                    self.delay(mb, 4)
                    self.release_fence()
                    a = self.ack_value(e)
                    for i in cutlass.range_constexpr(WORLD):
                        addr = (
                            mb.peers[i]
                            + Int64(mb.rec_bytes)
                            + Int64(4) * (Int64(W_ACK) + Int64(32) * Int64(mb.rank))
                        )
                        sys_ops.st_relaxed_sys_u32(addr, a)

    # ------------------------------------------------------------ records
    @cute.jit
    def pack_records(self, w0: Uint32, w1: Uint32, w2: Uint32, tag: Uint32):
        """Six bf16 (three little-endian pairs) -> two records carrying ``tag``."""
        t48 = Uint64(tag) << 48
        r0 = Uint64(w0) | (Uint64(w1 & Uint32(0xFFFF)) << 32) | t48
        r1 = Uint64(w1 >> 16) | (Uint64(w2) << 16) | t48
        return r0, r1

    @cute.jit
    def publish_records(self, mb: Mbox, buf, m, j, r0: Uint64, r1: Uint64):
        """Record pair ``j`` of row ``m`` into slot ``buf`` of every rank, starting with
        this rank's own copy (destinations staggered across ranks)."""
        off = mb.rec_off(buf, mb.rank, m, j)
        for i in cutlass.range_constexpr(WORLD):
            sys_ops.st_relaxed_sys_v2_b64(mb.peers[i] + off, r0, r1)

    @cute.jit
    def poll_records(self, mb: Mbox, buf, m, j, s0, tag: Uint32, poisoned, x0, x1):
        """Poll record pair ``j`` of row ``m`` from sources ``s0`` and ``s0 + 1`` in the
        local slot ``buf`` until all four records carry ``tag``; unpack the six values
        of each source into ``x0`` / ``x1`` (fp32 register tensors). Poisoned: NaN."""
        base = mb.peers[0]
        a0 = base + mb.rec_off(buf, s0, m, j)
        a1 = base + mb.rec_off(buf, s0 + 1, m, j)
        r00, r01 = sys_ops.ld_relaxed_sys_v2_b64(a0)
        r10, r11 = sys_ops.ld_relaxed_sys_v2_b64(a1)
        pending = Uint32(0)
        if not (self.tag_ok(r00, tag) and self.tag_ok(r01, tag)):
            pending = pending | Uint32(1)
        if not (self.tag_ok(r10, tag) and self.tag_ok(r11, tag)):
            pending = pending | Uint32(2)
        t0 = Uint64(0)
        it = Uint32(0)
        lim = Uint64(self.timeout_ns)
        if poisoned != 0 and lim > Uint64(self.poisoned_timeout_ns):
            lim = Uint64(self.poisoned_timeout_ns)
        while pending != Uint32(0):
            if (pending & Uint32(1)) != Uint32(0):
                r00, r01 = sys_ops.ld_relaxed_sys_v2_b64(a0)
                if self.tag_ok(r00, tag) and self.tag_ok(r01, tag):
                    pending = pending & Uint32(2)
            if (pending & Uint32(2)) != Uint32(0):
                r10, r11 = sys_ops.ld_relaxed_sys_v2_b64(a1)
                if self.tag_ok(r10, tag) and self.tag_ok(r11, tag):
                    pending = pending & Uint32(1)
            if pending != Uint32(0):
                it = it + Uint32(1)
                if (it & Uint32(POLL_TIMER_STRIDE - 1)) == Uint32(0):
                    now = sys_ops.globaltimer()
                    if t0 == Uint64(0):
                        t0 = now
                    elif now - t0 > lim:
                        peer = Int32(s0) + Int32((pending & Uint32(1)) == Uint32(0))
                        self.fail(mb, ERR_RECORD_POLL, 1, peer, m, tag)
        if poisoned != 0:
            for k in cutlass.range_constexpr(6):
                x0[k] = P.bits_f32(Uint32(0x7FC00000))
                x1[k] = P.bits_f32(Uint32(0x7FC00000))
        else:
            self.unpack_records(r00, r01, x0)
            self.unpack_records(r10, r11, x1)

    @cute.jit
    def unpack_records(self, r0: Uint64, r1: Uint64, x):
        x[0] = P.bits_f32(Uint32(r0 & Uint64(0xFFFF)) << 16)
        x[1] = P.bits_f32(Uint32((r0 >> 16) & Uint64(0xFFFF)) << 16)
        x[2] = P.bits_f32(Uint32((r0 >> 32) & Uint64(0xFFFF)) << 16)
        x[3] = P.bits_f32(Uint32(r1 & Uint64(0xFFFF)) << 16)
        x[4] = P.bits_f32(Uint32((r1 >> 16) & Uint64(0xFFFF)) << 16)
        x[5] = P.bits_f32(Uint32((r1 >> 32) & Uint64(0xFFFF)) << 16)


# ====================================================================== host side
class ErrorRecord:
    """Host-mapped pinned words written by a timed-out wait just before its ``trap``, so
    the host can report which wait failed after the CUDA context is lost."""

    def __init__(self, device):
        from cuda.bindings import driver as cu

        self._cu = cu
        flags = cu.CU_MEMHOSTALLOC_DEVICEMAP | cu.CU_MEMHOSTALLOC_PORTABLE
        err, host = cu.cuMemHostAlloc(4 * ERR_RECORD_WORDS, flags)
        _check(cu, err, "cuMemHostAlloc")
        self.host_ptr = int(host)
        err, dptr = cu.cuMemHostGetDevicePointer(self.host_ptr, 0)
        _check(cu, err, "cuMemHostGetDevicePointer")
        self.device_ptr = int(dptr)
        self.clear()

    def _view(self):
        import ctypes

        return (ctypes.c_uint32 * ERR_RECORD_WORDS).from_address(self.host_ptr)

    def clear(self):
        v = self._view()
        for i in range(ERR_RECORD_WORDS):
            v[i] = 0

    def read(self):
        """None if no wait timed out, else the decoded record: ``kind``, ``site``, the
        waiting ``rank``, the ``peer`` it waited for, the ``row`` it was polling and the
        awaited ``epoch`` (see ``Transport.fail``)."""
        v = list(self._view())
        if v[0] != ERR_MAGIC:
            return None
        return {
            "kind": ERR_KIND_NAMES.get(v[1], v[1]),
            "site": v[2],
            "rank": v[3] & 0xFF,
            "peer": (v[3] >> 8) & 0xFF,
            "row": v[3] >> 16,
            "epoch": v[4],
        }


def _check(cu, err, what):
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{what} failed: {err}")


class HopMailbox:
    """One symmetric mailbox instance (records for ``max_rows`` rows + control block) on
    every rank of a tensor-parallel group. Collective: every rank constructs it.
    ``barrier_group``: the process group the init's barriers run on (default: the group
    ``group_name`` names); a CPU group keeps them from creating a device communicator."""

    def __init__(
        self,
        group_name: str,
        max_rows: int,
        device,
        start: int = -1,
        barrier_group=None,
    ):
        import torch.distributed._symmetric_memory as symm_mem

        self.max_rows = max_rows
        self.device = torch.device(device)
        self.nbytes = mailbox_bytes(max_rows)
        # symmetric allocations need not be page aligned: over-allocate and use the 4
        # KiB
        # aligned window of every rank's buffer (each rank's offset follows from its own
        # base)
        self.storage = symm_mem.empty(
            self.nbytes + ALIGN, dtype=torch.uint8, device=self.device
        )
        handle = symm_mem.rendezvous(self.storage, group_name)
        self.rank = int(handle.rank)
        if int(handle.world_size) != WORLD:
            raise ValueError(
                f"hop mailbox needs {WORLD} ranks, got {handle.world_size}"
            )
        self.peers = [_aligned(int(p)) for p in handle.buffer_ptrs]
        off = self.peers[self.rank] - self.storage.data_ptr()
        self.tensor = self.storage[off : off + self.nbytes]
        self.group_name = group_name
        self.barrier_group = barrier_group
        get_backend = getattr(symm_mem, "get_backend", None)
        self.backend = get_backend(self.device) if get_backend is not None else None
        self.init(start)

    @classmethod
    def create(
        cls,
        group_name: str,
        max_rows: int,
        device=None,
        start: int = -1,
        barrier_group=None,
    ):
        return cls(
            group_name,
            max_rows,
            device or torch.accelerator.current_accelerator(),
            start,
            barrier_group,
        )

    def rotated_peers(self):
        return [self.peers[(self.rank + i) % WORLD] for i in range(WORLD)]

    def init(self, start: int = -1, before_last_barrier=None):
        """Records zero (tag 0), acknowledgements ``start - 1``, epochs ``start`` (NBUF:
        the first use of every slot has a non-zero tag), error / arrival words 0.
        Collective (quiesce, init, quiesce); ``before_last_barrier`` re-initializes
        state that must restart with this mailbox (the router-partial words)."""
        start = NBUF if start < 0 else start
        group = self.barrier_group
        if group is None:
            group = _group(self.group_name)
        torch.accelerator.synchronize()
        dist.barrier(group=group)
        _init_mailbox(self.tensor, self.max_rows, start)
        if before_last_barrier is not None:
            before_last_barrier()
        torch.accelerator.synchronize()
        dist.barrier(group=group)


ALIGN = 4096


def _aligned(ptr: int) -> int:
    return ptr + (-ptr) % ALIGN


def _group(group_name):
    from torch.distributed.distributed_c10d import _resolve_process_group

    return _resolve_process_group(group_name)


def _init_mailbox(tensor: torch.Tensor, max_rows: int, start: int):
    """Mailbox image, written with torch ops (stream-ordered): records 0; control
    block 0 except each source's ack word (``start - 1``) and the E and per-row
    E_row words (``start``), as 32-bit values."""
    rb = record_bytes(max_rows)
    tensor.zero_()
    ctrl = tensor[rb : rb + CTRL_BYTES].view(torch.int32)
    s = start & 0xFFFFFFFF
    sm1 = (start - 1) & 0xFFFFFFFF
    as_i32 = lambda v: v - (1 << 32) if v >= (1 << 31) else v  # noqa: E731
    for src in range(WORLD):
        ctrl[W_ACK + 32 * src] = as_i32(sm1)
    ctrl[W_E] = as_i32(s)
    for m in range(max_rows):
        ctrl[W_EROW + m] = as_i32(s)


# ================================================================== consumer side
@dataclass(frozen=True)
class ConsSmem:
    """Shared memory of one consumer CTA (row ``m``, cluster rank ``c``)."""

    # groups 1..3: both sources' raw values (group 0 sums every rank in order)
    raw: Any
    # 768 words: the 96 per-chunk sums of squares, or the CTA's 768 bf16-rounded
    # values when the consumer squares them from shared memory
    rs: Any
    wsum: Any
    csum: Any
    mbar: Any
    es: EpochSmem

    @staticmethod
    def allocate(smem, max_rows):
        f32 = cutlass.Float32
        return ConsSmem(
            smem.allocate_tensor(
                f32, cute.make_layout((3, 128, 12), stride=(128 * 12, 12, 1)), 16
            ),
            smem.allocate_tensor(f32, cute.make_layout(768), 16),
            smem.allocate_tensor(f32, cute.make_layout(17), 4),
            smem.allocate_tensor(f32, cute.make_layout(8), 4),
            smem.allocate_tensor(cutlass.Int64, cute.make_layout(1), 8),
            EpochSmem.allocate(smem, max_rows),
        )

    def mbar_addr(self):
        return Uint32(self.mbar.iterator.toint())


class Consumer(Transport):
    """Consumer epilogue of a hop: all-reduce in rank order (+ residual) (+ RMSNorm),
    numerically the one-shot fused all-reduce + residual + RMSNorm order: fp32 sum from
    +0 over ranks 0..7, bf16 with -0 made +0, bf16 residual add, per-8-element chunk
    sums of bf16-rounded squares, xor butterflies, block reduce ``(w0 + w2) + w1``,
    cluster-rank-order total, ``rsqrt(fma(total, rcp(HIDDEN), eps))``, ``y = bf16((x *
    inv) * (0 + g))``."""

    @cute.jit
    def cons_prologue(self, cs: ConsSmem):
        """Initialize the mbarrier, then cluster arrive (before cluster accesses)."""
        t = P.thread_index()
        if t == 0:
            P.mbar_init(cs.mbar_addr(), 1)
            P.fence_mbarrier_init_release_cluster()
        cute.arch.sync_threads()
        # the cluster-scope release arrive is issued by the idle epoch-warp thread
        if t == 512:
            P.mbar_arrive_expect_tx_release_cluster(cs.mbar_addr(), 8 * 4)
        P.cluster_arrive_relaxed()

    def block_reduce(self, v0, v1, v2):
        """Block sum of the three warp sums: ``(w0 + w2) + w1``."""
        return C.add_rn(C.add_rn(v0, v2), v1)

    def chunk_carry(self, run):
        return run

    @cute.jit
    def consume_cta(
        self,
        mb: Mbox,
        cs: ConsSmem,
        m,
        c,
        e,
        poisoned,
        ru,
        gu,
        out_res: Int64,
        out_norm: Int64,
        eps: Float32,
        res: cutlass.Constexpr[bool],
        norm: cutlass.Constexpr[bool],
        dn_bsum=None,
    ):
        """The 512 poll threads of consumer CTA (row ``m``, cluster rank ``c``); thread
        ``f + 128 g`` polls record pair ``f`` from sources ``2g`` and ``2g + 1``; group
        0 finalizes six elements per thread.

        With ``dn_bsum`` (the deferred-norm form, ``norm`` set): the CTA writes
        ``bf16(x * (0 + g))`` to ``out_norm`` and its block sum of squares (the norm's
        own value and order) to ``dn_bsum[m][c]`` instead of normalizing; the block sums
        are not exchanged in the cluster, and the row's scale is applied downstream."""
        t = P.thread_index()
        f = t % 128
        grp = t // 128
        buf = self.buffer_of(e)
        tag = Uint32(e) & Uint32(TAG_MASK)
        j = c * 256 + 2 * f
        acc = cute.make_rmem_tensor((6,), Float32)
        x0 = cute.make_rmem_tensor((6,), Float32)
        x1 = cute.make_rmem_tensor((6,), Float32)
        acc.fill(0.0)
        self.delay(mb, 2)
        self.poll_records(mb, buf, m, j, 2 * grp, tag, poisoned, x0, x1)
        if grp > 0:
            for k in cutlass.range_constexpr(6):
                cs.raw[grp - 1, f, k] = x0[k]
                cs.raw[grp - 1, f, 6 + k] = x1[k]
        else:
            for k in cutlass.range_constexpr(6):
                acc[k] = C.add_rn(C.add_rn(Float32(0.0), x0[k]), x1[k])
        P.bar_sync(1, 512)
        r = cute.make_rmem_tensor((6,), Float32)
        r.fill(0.0)
        h0 = c * 768 + 6 * f
        if grp == 0:
            for q in cutlass.range_constexpr(1, 4):
                for k in cutlass.range_constexpr(6):
                    acc[k] = C.add_rn(
                        C.add_rn(acc[k], cs.raw[q - 1, f, k]), cs.raw[q - 1, f, 6 + k]
                    )
            rb = cute.make_rmem_tensor((3,), Uint32)
            for k in cutlass.range_constexpr(3):
                ab = P.no_neg_zero(C.cvt_rn_bf16x2_f32(acc[2 * k], acc[2 * k + 1]))
                if cutlass.const_expr(res):
                    rb[k] = C.cvt_rn_bf16x2_f32(
                        C.add_rn(P.bf16_lo(ab), P.bf16_lo(ru[k])),
                        C.add_rn(P.bf16_hi(ab), P.bf16_hi(ru[k])),
                    )
                else:
                    rb[k] = ab
            row = out_res + Int64(2) * (Int64(m) * HIDDEN + Int64(h0))
            for k in cutlass.range_constexpr(3):
                sys_ops.st_global_u32(row + Int64(4 * k), rb[k])
            for k in cutlass.range_constexpr(3):
                r[2 * k] = P.bf16_lo(rb[k])
                r[2 * k + 1] = P.bf16_hi(rb[k])
            if cutlass.const_expr(norm):
                self.chunk_squares(cs, f, r)
        if cutlass.const_expr(not norm):
            if grp == 0:
                sys_ops.fence_proxy_async_global()
        else:
            lane = t % 32
            warp = t // 32
            bs = Float32(0.0)
            if grp == 0:
                P.bar_sync(2, 128)
                if warp == 0:
                    v0 = cs.rs[lane]
                    v1 = cs.rs[32 + lane]
                    v2 = cs.rs[64 + lane]
                    for i in cutlass.range_constexpr(5):
                        o: cutlass.Constexpr = 16 >> i
                        v0 = C.add_rn(v0, P.shfl_xor(v0, o))
                        v1 = C.add_rn(v1, P.shfl_xor(v1, o))
                        v2 = C.add_rn(v2, P.shfl_xor(v2, o))
                    bs = self.block_reduce(v0, v1, v2)
            if cutlass.const_expr(dn_bsum is not None):
                if t == 0:
                    sys_ops.st_global_u32(
                        dn_bsum + Int64(4) * (Int64(m) * 8 + Int64(c)), P.f32_bits(bs)
                    )
                if grp == 0:
                    row = out_norm + Int64(2) * (Int64(m) * HIDDEN + Int64(h0))
                    for k in cutlass.range_constexpr(3):
                        g0 = C.add_rn(Float32(0.0), P.bf16_lo(gu[k]))
                        g1 = C.add_rn(Float32(0.0), P.bf16_hi(gu[k]))
                        u = C.cvt_rn_bf16x2_f32(
                            C.mul_rn(r[2 * k], g0), C.mul_rn(r[2 * k + 1], g1)
                        )
                        sys_ops.st_global_u32(row + Int64(4 * k), u)
                    sys_ops.fence_proxy_async_global()  # thread 0 is in group 0
                # completes the cluster barrier phase the consumer prologue opened
                P.cluster_wait()
            else:
                # cluster exchange of the 8 block sums (st.async + mbarrier complete_tx)
                crank = Int32(cute.arch.block_idx_in_cluster())
                P.cluster_wait()  # every CTA of the cluster started (mbarrier initialized)
                if t == 0:
                    la = Uint32((cs.csum.iterator + crank).toint())
                    lm = cs.mbar_addr()
                    for cc in cutlass.range_constexpr(8):
                        P.st_async_b32(
                            P.mapa_u32(la, cc), P.f32_bits(bs), P.mapa_u32(lm, cc)
                        )
                P.mbar_wait_cluster(cs.mbar_addr(), Uint32(0))
                P.cluster_arrive_relaxed()  # exit barrier, arrive: inbound st.async landed
                if grp == 0:  # the output threads read the cluster's sums of squares
                    tot = Float32(0.0)
                    for cc in cutlass.range_constexpr(8):
                        tot = C.add_rn(tot, cs.csum[cc])
                    inv = C.rsqrt_approx(C.fma_rn(tot, C.rcp_rn(Float32(6144.0)), eps))
                    row = out_norm + Int64(2) * (Int64(m) * HIDDEN + Int64(h0))
                    for k in cutlass.range_constexpr(3):
                        g0 = C.add_rn(Float32(0.0), P.bf16_lo(gu[k]))
                        g1 = C.add_rn(Float32(0.0), P.bf16_hi(gu[k]))
                        y = C.cvt_rn_bf16x2_f32(
                            C.mul_rn(C.mul_rn(r[2 * k], inv), g0),
                            C.mul_rn(C.mul_rn(r[2 * k + 1], inv), g1),
                        )
                        sys_ops.st_global_u32(row + Int64(4 * k), y)
                    sys_ops.fence_proxy_async_global()
                # exit barrier, wait: no CTA exits with its st.async in flight
                P.cluster_wait()

    @cute.jit
    def chunk_squares(self, cs: ConsSmem, f, r):
        """Chunk ``p`` (elements ``[8p, 8p + 8)``) of the CTA's 768 elements is summed
        from 0 in element order. Thread ``f = 4q + ph`` owns ``[24q + 6ph, +6)``: it
        adds its squares into the tail of the previous thread's chunk, carried by one
        ``shfl.up``, and writes the chunk it completes."""
        q = cute.make_rmem_tensor((6,), Float32)
        for k in cutlass.range_constexpr(3):
            w = C.cvt_rn_bf16x2_f32(
                C.mul_rn(r[2 * k], r[2 * k]), C.mul_rn(r[2 * k + 1], r[2 * k + 1])
            )
            q[2 * k] = P.bf16_lo(w)
            q[2 * k + 1] = P.bf16_hi(w)
        ph = f % 4
        tail = Float32(0.0)
        if ph == 0:
            for k in cutlass.range_constexpr(6):
                tail = C.add_rn(tail, q[k])
        elif ph == 1:
            for k in cutlass.range_constexpr(2, 6):
                tail = C.add_rn(tail, q[k])
        elif ph == 2:
            tail = C.add_rn(C.add_rn(tail, q[4]), q[5])
        run = self.chunk_carry(P.shfl_up(tail, 1))
        if ph == 1:
            for k in cutlass.range_constexpr(2):
                run = C.add_rn(run, q[k])
            cs.rs[3 * (f // 4)] = run
        elif ph == 2:
            for k in cutlass.range_constexpr(4):
                run = C.add_rn(run, q[k])
            cs.rs[3 * (f // 4) + 1] = run
        elif ph == 3:
            for k in cutlass.range_constexpr(6):
                run = C.add_rn(run, q[k])
            cs.rs[3 * (f // 4) + 2] = run

    @cute.jit
    def consume_cta_bsum(
        self, mb: Mbox, cs: ConsSmem, m, c, e, poisoned, ru, out_res: Int64, bsum: Int64
    ):
        """Consumer epilogue without the norm (deferred): all-reduce + residual, then
        this CTA's block sum of bf16-rounded squares in the same order as the norm's (96
        chunks of 8 from 0, xor butterflies, ``(w0 + w2) + w1``) to ``bsum[m][c]``; the
        norm is applied by a later kernel."""
        t = P.thread_index()
        f = t % 128
        grp = t // 128
        buf = self.buffer_of(e)
        tag = Uint32(e) & Uint32(TAG_MASK)
        acc = cute.make_rmem_tensor((6,), Float32)
        x0 = cute.make_rmem_tensor((6,), Float32)
        x1 = cute.make_rmem_tensor((6,), Float32)
        acc.fill(0.0)
        self.delay(mb, 2)
        self.poll_records(mb, buf, m, c * 256 + 2 * f, 2 * grp, tag, poisoned, x0, x1)
        if grp > 0:
            for k in cutlass.range_constexpr(6):
                cs.raw[grp - 1, f, k] = x0[k]
                cs.raw[grp - 1, f, 6 + k] = x1[k]
        else:
            for k in cutlass.range_constexpr(6):
                acc[k] = C.add_rn(C.add_rn(Float32(0.0), x0[k]), x1[k])
        P.bar_sync(1, 512)
        h0 = c * 768 + 6 * f
        if grp == 0:
            for q in cutlass.range_constexpr(1, 4):
                for k in cutlass.range_constexpr(6):
                    acc[k] = C.add_rn(
                        C.add_rn(acc[k], cs.raw[q - 1, f, k]), cs.raw[q - 1, f, 6 + k]
                    )
            rb = cute.make_rmem_tensor((3,), Uint32)
            for k in cutlass.range_constexpr(3):
                ab = P.no_neg_zero(C.cvt_rn_bf16x2_f32(acc[2 * k], acc[2 * k + 1]))
                rb[k] = C.cvt_rn_bf16x2_f32(
                    C.add_rn(P.bf16_lo(ab), P.bf16_lo(ru[k])),
                    C.add_rn(P.bf16_hi(ab), P.bf16_hi(ru[k])),
                )
            row = out_res + Int64(2) * (Int64(m) * HIDDEN + Int64(h0))
            for k in cutlass.range_constexpr(3):
                sys_ops.st_global_u32(row + Int64(4 * k), rb[k])
            for k in cutlass.range_constexpr(3):
                cs.rs[6 * f + 2 * k] = P.bf16_lo(rb[k])
                cs.rs[6 * f + 2 * k + 1] = P.bf16_hi(rb[k])
        P.bar_sync(1, 512)
        lane = t % 32
        warp = t // 32
        if t < 96:
            ts = Float32(0.0)
            for i in cutlass.range_constexpr(4):
                a0 = cs.rs[8 * t + 2 * i]
                a1 = cs.rs[8 * t + 2 * i + 1]
                wq = C.cvt_rn_bf16x2_f32(C.mul_rn(a0, a0), C.mul_rn(a1, a1))
                ts = C.add_rn(C.add_rn(ts, P.bf16_lo(wq)), P.bf16_hi(wq))
            for i in cutlass.range_constexpr(5):
                o: cutlass.Constexpr = 16 >> i
                ts = C.add_rn(ts, P.shfl_xor(ts, o))
            if lane == 0:
                cs.wsum[warp] = ts
        P.bar_sync(1, 512)
        if t == 0:
            # the three warp sums, read one word each (no vector read of a fourth word)
            wa = Uint32(cs.wsum.iterator.toint())
            v = self.block_reduce(
                P.ld_shared_f32(wa), P.ld_shared_f32(wa + 4), P.ld_shared_f32(wa + 8)
            )
            sys_ops.st_global_u32(
                bsum + Int64(4) * (Int64(m) * 8 + Int64(c)), P.f32_bits(v)
            )
        if grp == 0:
            sys_ops.fence_proxy_async_global()
