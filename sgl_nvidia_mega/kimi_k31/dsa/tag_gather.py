# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Flag-free all-gather with epoch-tagged words, for the request-sharded DSA selection (``K31MK_DSA_REQ_SHARD_AG=tag``).

Every int32 slot id v (-1 <= v < 2**25 - 1) travels as (tag << 25) | (v + 1), where tag = epoch & 0x7F. Each rank stores
its block of tagged words into copy ``c`` of every rank's mailbox (st.relaxed.sys through the peer's symmetric-memory
mapping, generic proxy; no multicast). A rank's poll reads its own mailbox (ld.relaxed.sys) until every word carries the
tag of the epoch it expects, then decodes the words into a local buffer. A word is its own flag, so there is no flag,
no sentinel and no re-arm; each 32-bit word is checked on its own.

Buffer reuse: rank j stores into rank i's copy c at call k only after j's poll of call k - 1 saw rank i's words of that
call, and i stored those after its poll of call k - 2, the last read of copy c (copies alternate by DSA layer:
``epoch_gather.copy_for``). The push starts with fence.acq_rel.sys before its stores and the poll ends with
fence.acq_rel.sys after its loads, so that chain is a release / acquire synchronization. Tags are 7 bits: when a reader
waits for epoch e, a word of copy c holds epoch e - 1 or e (the chain above keeps every writer at most one use of a copy
ahead), and every word of a copy is rewritten on every use.

An expired poll traps (nothing decoded, no epoch advanced). Epoch counters live in device memory (per CTA), so CUDA graph
replays advance them. Both kernels are compiled at construction, before the barrier, without value specialization on the
per-call ints, so no rank compiles inside a step while its peers poll."""

import torch
import triton
import triton.language as tl

from .epoch_gather import SELF_CHECK_NS, copies_for, copy_for  # noqa: F401  (copy rule shared with the epoch route)

BLOCK = 1024  # 16-byte chunks per CTA
VALUE_LIMIT = (1 << 25) - 1  # slot ids must be below this (and >= -1)


@triton.jit
def _ld4(ptr, mask):  # ld.relaxed.sys.global.v4.b32 (strong; not pure: re-issued every poll iteration)
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
def _st4(ptr, x, y, z, w, mask):  # st.relaxed.sys.global.v4.b32 through a peer's unicast mapping
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.eq.s32 %p0, $6, 1;
            @!%p0 bra end;
            st.relaxed.sys.global.v4.b32 [$1], {$2, $3, $4, $5};
            end:
        }
        """,
        "=r,l,r,r,r,r,r",
        args=[ptr, x, y, z, w, mask.to(tl.int32)],
        dtype=(tl.uint32),
        is_pure=False,
        pack=1,
    )


@triton.jit
def _fence_sys():
    tl.inline_asm_elementwise("fence.acq_rel.sys;", "=r", [], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _trap():
    tl.inline_asm_elementwise("trap;", "=r", [], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _bar():
    tl.inline_asm_elementwise("bar.sync 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _globaltimer():
    return tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1)


@triton.jit
def _enc(v, tagv):
    return tagv | ((v + 1) & 0x1FFFFFF)


@triton.jit
def _stale(x, y, z, w, tg):
    return ((x >> 25) != tg) | ((y >> 25) != tg) | ((z >> 25) != tg) | ((w >> 25) != tg)


@triton.jit(do_not_specialize=["src_ptr", "c", "n_own_chunks"])
def _push_kernel(src_ptr, mb_peers, ep_ptr, c, n_own_chunks, RANK: tl.constexpr, W: tl.constexpr, NBO: tl.constexpr,
                 BLOCK: tl.constexpr, FENCE: tl.constexpr):
    # grid (NBO, W): CTA (b, j) stores own chunk block b, tagged, into peer j's copy c at chunk RANK x n_own_chunks
    b = tl.program_id(0)
    j = tl.program_id(1)
    slot = (c * W + j) * NBO + b
    e = tl.load(ep_ptr + slot).to(tl.uint32) + 1
    if FENCE:  # 0 only under K31MK_DSA_REQ_SHARD_TAG_NOFENCE (TAGNF: copy reuse rests on the served layer chain, fails closed)
        _fence_sys()  # release: the stores below follow this rank's earlier (stream-ordered) reads of its mailboxes
    offs = b * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_own_chunks
    x, y, z, w = _ld4(src_ptr.to(tl.pointer_type(tl.uint64)) + offs * 2, mask)
    tagv = (e & 0x7F) << 25
    base = tl.load(mb_peers.to(tl.pointer_type(tl.uint64)) + j).to(tl.pointer_type(tl.uint64))
    dst = base + (c * (W * n_own_chunks) + RANK * n_own_chunks + offs) * 2
    _st4(dst, _enc(x, tagv), _enc(y, tagv), _enc(z, tagv), _enc(w, tagv), mask)
    _bar()  # every thread has read the counter before it advances
    tl.store(ep_ptr + slot, e.to(tl.int32))


@triton.jit(do_not_specialize=["src_ptr", "ep_src", "c", "n_own_chunks"])
def _push_ext_kernel(src_ptr, mb_peers, ep_src, c, n_own_chunks, RANK: tl.constexpr, W: tl.constexpr, BLOCK: tl.constexpr,
                     FENCE: tl.constexpr):
    # K31MK_DSA_REQ_SHARD_TKPUSH fallback (a top-k launch path without the in-kernel push): _push_kernel with the epoch read
    # from the consumer's use counter of copy c (ep_src, as dsa/tag_push.cuh) instead of a push counter; advances nothing.
    b = tl.program_id(0)
    j = tl.program_id(1)
    e = tl.load(ep_src).to(tl.uint32) + 1
    if FENCE:
        _fence_sys()  # release, as _push_kernel
    offs = b * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_own_chunks
    x, y, z, w = _ld4(src_ptr.to(tl.pointer_type(tl.uint64)) + offs * 2, mask)
    tagv = (e & 0x7F) << 25
    base = tl.load(mb_peers.to(tl.pointer_type(tl.uint64)) + j).to(tl.pointer_type(tl.uint64))
    dst = base + (c * (W * n_own_chunks) + RANK * n_own_chunks + offs) * 2
    _st4(dst, _enc(x, tagv), _enc(y, tagv), _enc(z, tagv), _enc(w, tagv), mask)


@triton.jit(do_not_specialize=["c", "n_chunks", "deadline_ns"])
def _poll_kernel(mb_ptr, out_ptr, ep_ptr, c, n_chunks, deadline_ns, NBP: tl.constexpr, BLOCK: tl.constexpr, FENCE: tl.constexpr):
    # grid NBP: CTA b waits until its chunks of copy c carry this epoch's tag, decodes them into out, then advances
    b = tl.program_id(0)
    e = tl.load(ep_ptr + c * NBP + b).to(tl.uint32) + 1
    tg = e & 0x7F
    offs = b * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_chunks
    p = mb_ptr.to(tl.pointer_type(tl.uint64)) + (c * n_chunks + offs) * 2
    x, y, z, w = _ld4(p, mask)
    dirty = mask & _stale(x, y, z, w, tg)
    t0 = _globaltimer()
    live = tl.max(dirty.to(tl.int32), 0)
    expired = live * 0
    while (live > 0) & (expired == 0):
        x2, y2, z2, w2 = _ld4(p, dirty)
        x = tl.where(dirty, x2, x)
        y = tl.where(dirty, y2, y)
        z = tl.where(dirty, z2, z)
        w = tl.where(dirty, w2, w)
        dirty = dirty & _stale(x, y, z, w, tg)
        live = tl.max(dirty.to(tl.int32), 0)
        expired = (_globaltimer() - t0 > deadline_ns).to(tl.int32)
    if live > 0:
        _trap()  # fail closed: nothing decoded, the epoch is not advanced
    # acquire: the words observed above precede this rank's next push. Placed right after the observing loads (the acquire
    # pattern needs only ld.relaxed; fence.acq_rel), not after the decode stores, so it does not wait for them.
    if FENCE:  # 0 only under TAGNF (see _push_kernel)
        _fence_sys()
    o = out_ptr + offs * 4
    tl.store(o + 0, (x & 0x1FFFFFF).to(tl.int32) - 1, mask=mask)
    tl.store(o + 1, (y & 0x1FFFFFF).to(tl.int32) - 1, mask=mask)
    tl.store(o + 2, (z & 0x1FFFFFF).to(tl.int32) - 1, mask=mask)
    tl.store(o + 3, (w & 0x1FFFFFF).to(tl.int32) - 1, mask=mask)
    _bar()
    tl.store(ep_ptr + c * NBP + b, e.to(tl.int32))


def encode(v, epoch):
    """Host form of the word format (what the push stores): slot ids (tensor) and an epoch -> int32 tagged words."""
    w = ((epoch & 0x7F) << 25) | ((v.to(torch.int64) + 1) & 0x1FFFFFF)
    return torch.where(w >= (1 << 31), w - (1 << 32), w).to(torch.int32)


def decode(w):
    """Host form of the poll's decode: int32 tagged words -> slot ids."""
    return ((w.to(torch.int64) & 0x1FFFFFF) - 1).to(torch.int32)


class TagGather:
    """``copies`` mailboxes of [world x n_own] int32 over ``group`` (a torch process group), per-CTA epoch counters and
    one decoded output per copy. Collective: every rank constructs it with the same arguments."""

    def __init__(self, group, rank, world, n_own, copies, device, deadline_ns, ext=False, fence=True):
        import torch.distributed._symmetric_memory as symm_mem

        if n_own % 4:
            raise ValueError(f"tag gather moves 16-byte chunks; n_own {n_own} is not a multiple of 4 int32")
        self.rank, self.world, self.n_own, self.copies = int(rank), int(world), int(n_own), int(copies)
        self.deadline_ns = int(deadline_ns)
        # fence=False: K31MK_DSA_REQ_SHARD_TAG_NOFENCE (TAGNF; fails closed) - the push / poll skip
        # fence.acq_rel.sys; the exact-tag acceptance and the deadline trap (armed, > 0) are what keep it fail-closed.
        self.fc = 1 if fence else 0
        if not fence and self.deadline_ns <= 0:
            raise ValueError("tag gather without fences needs an armed deadline (deadline_ns > 0)")
        self.nbo = triton.cdiv(self.n_own // 4, BLOCK)
        self.nbp = triton.cdiv(self.world * self.n_own // 4, BLOCK)
        self.mailbox = symm_mem.empty((self.copies, self.world * self.n_own), dtype=torch.int32, device=device)
        self.hdl = symm_mem.rendezvous(self.mailbox, group=group)
        self.mailbox.zero_()  # tag 0: never awaited before every word of the copy is rewritten (epochs start at 1)
        self.push_epochs = torch.zeros((self.copies, self.world, self.nbo), dtype=torch.int32, device=device)
        self.poll_epochs = torch.zeros((self.copies, self.nbp), dtype=torch.int32, device=device)
        self.out = torch.full((self.copies, self.world * self.n_own), -1, dtype=torch.int32, device=device)
        own = torch.zeros(self.n_own, dtype=torch.int32, device=device)
        _push_kernel.warmup(own, self.hdl.buffer_ptrs_dev, self.push_epochs, 0, self.n_own // 4, RANK=self.rank, W=self.world,
                            NBO=self.nbo, BLOCK=BLOCK, FENCE=self.fc, num_warps=8, grid=(1, 1))
        for d in sorted({self.deadline_ns, max(self.deadline_ns, SELF_CHECK_NS)}):
            _poll_kernel.warmup(self.mailbox, self.out[0], self.poll_epochs, 0, self.world * self.n_own // 4, d, NBP=self.nbp,
                                BLOCK=BLOCK, FENCE=self.fc, num_warps=8, grid=(1,))
        if ext:  # K31MK_DSA_REQ_SHARD_TKPUSH: the fallback push, compiled here too (never inside a step)
            _push_ext_kernel.warmup(own, self.hdl.buffer_ptrs_dev, self.poll_epochs[0, :1], 0, self.n_own // 4, RANK=self.rank,
                                    W=self.world, BLOCK=BLOCK, FENCE=self.fc, num_warps=8, grid=(1, 1))
        torch.cuda.synchronize(device)
        torch.distributed.barrier(group=group)  # every mailbox is zero and every rank has compiled before anyone stores

    def gather(self, c, own):
        """Store ``own`` ([n_own] int32 slot ids, contiguous) into copy ``c`` of every rank and wait for every peer's block.
        Returns [world x n_own] int32 (copy ``c``'s decoded output; valid until the next call of copy ``c``)."""
        _push_kernel[(self.nbo, self.world)](own, self.hdl.buffer_ptrs_dev, self.push_epochs, c, self.n_own // 4,
                                             RANK=self.rank, W=self.world, NBO=self.nbo, BLOCK=BLOCK, FENCE=self.fc, num_warps=8)
        _poll_kernel[(self.nbp,)](self.mailbox, self.out[c], self.poll_epochs, c, self.world * self.n_own // 4, self.deadline_ns,
                                  NBP=self.nbp, BLOCK=BLOCK, FENCE=self.fc, num_warps=8)
        return self.out[c]

    def push(self, c, own):
        """The store half of ``gather`` only (K31MK_DSA_REQ_SHARD_AG=fused: the sparse core polls copy ``c`` itself)."""
        _push_kernel[(self.nbo, self.world)](own, self.hdl.buffer_ptrs_dev, self.push_epochs, c, self.n_own // 4,
                                             RANK=self.rank, W=self.world, NBO=self.nbo, BLOCK=BLOCK, FENCE=self.fc, num_warps=8)

    # K31MK_DSA_REQ_SHARD_TKPUSH (dsa/tkpush.py): the owner top-k stores this rank's block itself, so the route needs the
    # mailbox addresses, the epoch source (the poll's own use counter of the copy: every counter of a copy advances once per
    # use, push and poll alike, so after k uses the poll counter holds k on every rank) and the poll / fallback push alone.
    def peers(self):
        """Every rank's mailbox base address (rank order; this process's mappings), for the top-k kernel's parameters."""
        return [int(p) for p in self.hdl.buffer_ptrs]

    def tag_epoch(self, c):
        """1-element view of the poll's use counter of copy ``c`` (poll CTA 0): the tag of copy c's next use is it + 1."""
        return self.poll_epochs[c, :1]

    def poll(self, c):
        """The wait half of ``gather`` only: copy ``c``'s decoded [world x n_own] rows once every block carries this use's tag."""
        _poll_kernel[(self.nbp,)](self.mailbox, self.out[c], self.poll_epochs, c, self.world * self.n_own // 4, self.deadline_ns,
                                  NBP=self.nbp, BLOCK=BLOCK, FENCE=self.fc, num_warps=8)
        return self.out[c]

    def push_ext(self, c, own, epoch):
        """The store half with the epoch read from ``epoch`` (a consumer's use counter of copy ``c``; see _push_ext_kernel)."""
        _push_ext_kernel[(self.nbo, self.world)](own, self.hdl.buffer_ptrs_dev, epoch, c, self.n_own // 4, RANK=self.rank,
                                                 W=self.world, BLOCK=BLOCK, FENCE=self.fc, num_warps=8)

    def words(self, c):
        """Copy ``c`` of this rank's mailbox: [world x n_own] int32 tagged words (rank-major blocks)."""
        return self.mailbox[c]

    def self_check(self, layers):
        """Build-time check (collective): one gather of a known pattern (including -1 and VALUE_LIMIT - 1) on the copy of the
        last layer's call, so the first forward's chain continues from it. Raises unless every rank's block arrived intact."""
        c = copy_for(layers - 1, layers)
        i = torch.arange(self.n_own, dtype=torch.int64, device=self.mailbox.device)

        def pattern(r):
            v = (r * 7919 + i * 2654435761) % (1 << 22)
            v = torch.where(i % 97 == 5, torch.full_like(v, -1), v)
            return torch.where(i % 89 == 7, torch.full_like(v, VALUE_LIMIT - 1), v).to(torch.int32)

        saved, self.deadline_ns = self.deadline_ns, max(self.deadline_ns, SELF_CHECK_NS)
        try:
            got = self.gather(c, pattern(self.rank).contiguous())
        finally:
            self.deadline_ns = saved
        want = torch.cat([pattern(r) for r in range(self.world)])
        bad = int((got != want).sum())
        if bad:
            raise RuntimeError(f"tag gather self-check: rank {self.rank}: {bad} of {want.numel()} words differ")
