# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Barrier-free all-gather with epoch flags, for the request-sharded DSA selection (``K31MK_DSA_REQ_SHARD_AG=epoch``).

Each rank multicasts its block of rows into copy ``c`` of a symmetric buffer on every rank, then releases one epoch flag per
(rank, block) into every peer's flag table through that peer's unicast pointer. A reader waits until every peer's flag holds
the expected epoch and then reads copy ``c`` in place. There is no entry or exit barrier, no copy-out and no re-arm.

The sync shapes are the multimem gatherer's (host triton_symm_mem_ag.py, acq_rel form). The writer stores the rows with
multimem.st, then bar.sync, then threads tid < world release with st.release.sys. The reader acquires with ld.acquire.sys,
then fence.proxy.alias, then bar.sync. Two things differ:

* Epoch flags replace the self-resetting signals. A writer stores e = (uses of copy c so far) + 1. A reader waits for == e
  and then counts the use. Values only grow (u32 wrap is harmless: the stale value e - 1 is never e), so flags are never
  re-armed.
* Buffer reuse (WAR) is ordered by the previous call's flags. A push into copy c first acquires every peer's flag of the
  previous call, whose copy is another one. Each peer released that flag after its reads of copy c, which precede its next
  push in stream order. The reader has already seen those flags, so this acquire finds them set and does not spin. Copies
  alternate (``copy_for``), so consecutive calls never share one.

Every spin traps after ``deadline_ns``, so a lost or late peer stops the step instead of letting it read stale rows.
"""

import torch
import triton
import triton.language as tl

BLOCK = 1024  # 16-byte chunks per push block (one flag per rank and block)
SELF_CHECK_NS = 60_000_000_000  # the build-time self-check's spin limit


def copy_for(ordinal, layers):
    """Mailbox copy of DSA layer ``ordinal`` (0-based) in a forward that runs ``layers`` layers: consecutive calls, including
    the last layer and the next forward's first layer, always use different copies (needs layers >= 2)."""
    if layers < 2:
        raise ValueError(f"epoch gather needs >= 2 calls per forward to alternate copies, got {layers}")
    if layers % 2 and ordinal == layers - 1:
        return 2
    return ordinal % 2


def copies_for(layers):
    return 3 if layers % 2 else 2


@triton.jit
def _tid3():
    return tl.inline_asm_elementwise("mov.u32 $0, %tid.x;\n\tmov.u32 $1, %tid.y;\n\tmov.u32 $2, %tid.z;", "=r,=r,=r", [],
                                     dtype=(tl.uint32, tl.uint32, tl.uint32), is_pure=True, pack=1)


@triton.jit
def _ntid3():
    return tl.inline_asm_elementwise("mov.u32 $0, %ntid.x;\n\tmov.u32 $1, %ntid.y;\n\tmov.u32 $2, %ntid.z;", "=r,=r,=r", [],
                                     dtype=(tl.uint32, tl.uint32, tl.uint32), is_pure=True, pack=1)


@triton.jit
def _flat_tid():
    tx, ty, tz = _tid3()
    nx, ny, _ = _ntid3()
    return tz * ny * nx + ty * nx + tx


@triton.jit
def _bar():
    tl.inline_asm_elementwise("bar.sync 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _acquire_eq(addrs, want, deadline_ns):
    # Per element: spin ld.acquire.sys until the flag == want, then fence.proxy.alias (the rows were written through the
    # multicast alias). Trap after deadline_ns: nothing is read, released or counted.
    return tl.inline_asm_elementwise(
        """
        {
            .reg .u32 %v;
            .reg .pred %p;
            .reg .u64 %t0, %t1;
            mov.u64 %t0, %globaltimer;
            spin:
            ld.acquire.sys.global.u32 %v, [$1];
            setp.eq.u32 %p, %v, $2;
            @%p bra ok;
            mov.u64 %t1, %globaltimer;
            sub.u64 %t1, %t1, %t0;
            setp.lt.u64 %p, %t1, $3;
            @%p bra spin;
            trap;
            ok:
            fence.proxy.alias;
            mov.u32 $0, %v;
        }
        """,
        "=r,l,r,l",
        args=[addrs, want, deadline_ns.to(tl.int64)],  # an int arg below 2**31 arrives as i32; the asm reads 64 bits
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _st_release_sys(addrs, v):
    return tl.inline_asm_elementwise("st.release.sys.global.u32 [$1], $2;", "=r,l,r", args=[addrs, v], dtype=tl.uint32,
                                     is_pure=False, pack=1)


@triton.jit
def _ld_v4(ptr, mask):  # ld.relaxed.sys.global.v4.b32
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.eq.s32 %p0, $5, 1;
            mov.u32 $0, 0;
            mov.u32 $1, 0;
            mov.u32 $2, 0;
            mov.u32 $3, 0;
            @!%p0 bra end;
            ld.relaxed.sys.global.v4.b32 {$0, $1, $2, $3}, [$4];
            end:
        }
        """,
        "=r,=r,=r,=r,l,r",
        args=[ptr, mask.to(tl.int32)],
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
        is_pure=False,
        pack=1,
    )


@triton.jit
def _st_mc_v4(ptr, x, y, z, w, mask):  # multimem.st.relaxed.sys.global.v4.f32 (raw bits; the gatherer's store)
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.eq.s32 %p0, $6, 1;
            @!%p0 bra end;
            multimem.st.relaxed.sys.global.v4.f32 [$1], {$2, $3, $4, $5};
            end:
        }
        """,
        "=r,l,r,r,r,r,r",
        args=[ptr, x, y, z, w, mask.to(tl.int32)],
        dtype=(tl.uint32),
        is_pure=False,
        pack=1,
    )


# No value specialization on the per-call ints: every call of one EpochGather then runs the one variant compiled in
# __init__ (a rank that JIT-compiles a new variant inside a step while its peers spin would trip their deadline).
@triton.jit(do_not_specialize=["src_ptr", "mc_ptr", "c", "p", "n_own_chunks", "deadline_ns"])
def _push_kernel(src_ptr, mc_ptr, fl_ptr, fl_peers, ep_ptr, c, p, n_own_chunks, deadline_ns,
                 RANK: tl.constexpr, W: tl.constexpr, NB: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    tid = _flat_tid()
    rr = tl.arange(0, W)
    # (1) WAR: every peer's flag of the previous call (copy p) for block b. Our previous wait already saw it, so no spin.
    prev_want = tl.load(ep_ptr + (p * W + rr) * NB + b).to(tl.uint32)
    if tid < W:
        _acquire_eq(fl_ptr + (p * W + rr) * NB + b, prev_want, deadline_ns)
    _bar()
    # (2) own chunk block b into every rank's copy c (multicast address of copy c)
    offs = b * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_own_chunks
    x, y, z, w = _ld_v4(src_ptr.to(tl.pointer_type(tl.uint64)) + offs * 2, mask)
    _st_mc_v4(mc_ptr.to(tl.int64).to(tl.pointer_type(tl.uint64)) + (RANK * n_own_chunks + offs) * 2, x, y, z, w, mask)
    _bar()
    # (3) release flag[c][RANK][b] = ep[c][RANK][b] + 1 into every peer's table (unicast pointer per peer)
    e = tl.load(ep_ptr + (c * W + RANK) * NB + b).to(tl.uint32) + 1
    peers = tl.load(fl_peers.to(tl.pointer_type(tl.uint64)) + rr).to(tl.pointer_type(tl.uint32))
    if tid < W:
        _st_release_sys(peers + (c * W + RANK) * NB + b, tl.zeros([W], dtype=tl.uint32) + e)


@triton.jit(do_not_specialize=["c", "deadline_ns"])
def _wait_kernel(fl_ptr, ep_ptr, c, deadline_ns, W: tl.constexpr, NB: tl.constexpr):
    b = tl.program_id(0)
    tid = _flat_tid()
    rr = tl.arange(0, W)
    want = tl.load(ep_ptr + (c * W + rr) * NB + b).to(tl.uint32) + 1
    if tid < W:
        _acquire_eq(fl_ptr + (c * W + rr) * NB + b, want, deadline_ns)
        tl.store(ep_ptr + (c * W + rr) * NB + b, want.to(tl.int32))


class EpochGather:
    """``copies`` symmetric buffers of [world x n_own] int32 over ``group`` (a torch process group), their epoch flag tables and
    this rank's epoch counters. Collective: every rank constructs it with the same arguments."""

    def __init__(self, group, rank, world, n_own, copies, device, deadline_ns):
        import torch.distributed._symmetric_memory as symm_mem

        if n_own % 4:
            raise ValueError(f"epoch gather moves 16-byte chunks; n_own {n_own} is not a multiple of 4 int32")
        if world & (world - 1):
            raise ValueError(f"epoch gather: world {world} must be a power of two (tl.arange)")
        self.rank, self.world, self.n_own, self.copies = int(rank), int(world), int(n_own), int(copies)
        self.deadline_ns = int(deadline_ns)
        self.nb = triton.cdiv(self.n_own // 4, BLOCK)
        self.buf = symm_mem.empty((self.copies, self.world * self.n_own), dtype=torch.int32, device=device)
        self.buf_hdl = symm_mem.rendezvous(self.buf, group=group)
        if self.buf_hdl.multicast_ptr == 0:
            raise ValueError(f"K31MK_DSA_REQ_SHARD_AG=epoch: no multicast for world {self.world}")
        self.flags = symm_mem.empty((self.copies, self.world, self.nb), dtype=torch.int32, device=device)
        self.flags_hdl = symm_mem.rendezvous(self.flags, group=group)
        self.flags.zero_()
        self.epochs = torch.zeros((self.copies, self.world, self.nb), dtype=torch.int32, device=device)
        # Compile both kernels for the served deadline and the self-check's (an int deadline below 2**31 is passed as i32, above
        # as i64: two signatures) BEFORE the barrier, so no rank compiles while a peer spins on its flags.
        own = torch.zeros(self.n_own, dtype=torch.int32, device=device)
        for d in sorted({self.deadline_ns, max(self.deadline_ns, SELF_CHECK_NS)}):
            _push_kernel.warmup(own, self.buf_hdl.multicast_ptr, self.flags, self.flags_hdl.buffer_ptrs_dev, self.epochs, 0, 1,
                                self.n_own // 4, d, RANK=self.rank, W=self.world, NB=self.nb, BLOCK=BLOCK, num_warps=8, grid=(1,))
            _wait_kernel.warmup(self.flags, self.epochs, 0, d, W=self.world, NB=self.nb, num_warps=1, grid=(1,))
        torch.cuda.synchronize(device)
        torch.distributed.barrier(group=group)  # every peer's flags are zero and every rank has compiled before anyone releases

    def gather(self, c, p, own):
        """Push ``own`` ([n_own] int32, contiguous) into copy ``c`` everywhere and wait for every peer's block; ``p`` = the copy of
        the previous call. Returns copy ``c`` as [world x n_own] int32 (read it in place until the next call of copy ``c``)."""
        nc_own = self.n_own // 4
        mc = self.buf_hdl.multicast_ptr + c * self.world * self.n_own * 4
        _push_kernel[(self.nb,)](own, mc, self.flags, self.flags_hdl.buffer_ptrs_dev, self.epochs, c, p, nc_own, self.deadline_ns,
                                 RANK=self.rank, W=self.world, NB=self.nb, BLOCK=BLOCK, num_warps=8)
        _wait_kernel[(self.nb,)](self.flags, self.epochs, c, self.deadline_ns, W=self.world, NB=self.nb, num_warps=1)
        return self.buf[c]

    def self_check(self, layers):
        """Build-time check (collective): one gather of a known pattern (including -1 words, an f32 NaN bit pattern on the
        multimem store) on the copy of the last layer's call, so the first forward's chain continues from it; a 60 s limit
        covers host skew between ranks at build time. Raises unless every rank's block arrived intact."""
        c, p = copy_for(layers - 1, layers), copy_for(layers - 2, layers)
        i = torch.arange(self.n_own, dtype=torch.int64, device=self.buf.device)

        def pattern(r):
            v = (r * 7919 + i * 2654435761) % (1 << 22)
            return torch.where(i % 97 == 5, torch.full_like(v, -1), v).to(torch.int32)

        saved, self.deadline_ns = self.deadline_ns, max(self.deadline_ns, SELF_CHECK_NS)
        try:
            got = self.gather(c, p, pattern(self.rank).contiguous())
        finally:
            self.deadline_ns = saved
        want = torch.cat([pattern(r) for r in range(self.world)])
        bad = int((got != want).sum())
        if bad:
            raise RuntimeError(f"epoch gather self-check: rank {self.rank}: {bad} of {want.numel()} words differ")
