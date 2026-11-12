# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""H-HOME for the two-shot o-projection (default off; the ``K31MK_OPROJ_HHOME*`` knobs turn it on).

A Rubin GPU has two locality domains (0 / 1); ordinary device memory interleaves over both, so some of the lines a CTA
polls are homed on the other die. The two-shot o-projection has two polled hops: the owner's receive slots (hop 1, polled by
``OprojUmma.reduce_owned``) and the gather rows (hop 2, polled by ``OprojConsumer``). The lever homes each polled line
on its poller's die.
"""

import os

import cutlass
import cutlass.cute as cute
from cutlass import Uint32

from ..common.primitives import asm

# ---------------------------------------------------------------------------------------------------------------------------
# HH-G static follow form (DRAFT, default off): the two-shot gather rows homed on the die of the consumer cluster that polls them.
# ``K31MK_OPROJ_HHOME_G`` = unset (default: ``common.twoshot.TwoShot``, nothing here runs) | "b<k>" | "b<k>r": token t's gather row
# lives in segment s = bit k of t; segment 0 on locality domain 0 and segment 1 on locality domain 1 ("b<k>"), or swapped ("b<k>r", the
# mismatched control). "t2" = "b0" (token t on locality domain t % 2). Every two-shot batch of the
# buffer uses the homed layout (the parity views of one allocation must not alias), so the rule is per buffer, not per batch.
# Only addresses change: the kernels re-layout the gather tensor at entry (``homed``) and index it as before; owners multicast
# into, and consumers poll and re-arm, the same (token, column) words as with the flat layout.
GATHER_ENV = "K31MK_OPROJ_HHOME_G"
SEG_ELEMS = 1 << 20  # one localized segment of BF16 (the localized allocation granularity; see the device query)


def gather_rule(value=None):
    """(bit k, swapped, dynamic) of ``K31MK_OPROJ_HHOME_G``, or None (default). b<k> / b<k>r: static (segment s on locality domain s, or
    swapped); d<k>: dynamic (segment s on locality domain s; clusters take die-matched tokens by ticket, ``cluster_token``)."""
    value = os.environ.get(GATHER_ENV, "") if value is None else value
    if value == "":
        return None
    v = {"t2": "b0", "t2r": "b0r"}.get(value, value)
    swapped = v.endswith("r") and v[0] == "b"
    k = v[1:-1] if swapped else v[1:]
    if v[:1] not in ("b", "d") or not k.isdigit() or int(k) > 6:
        raise ValueError(f"{GATHER_ENV} must be unset, t2, t2r, b<k>, b<k>r or d<k> (k <= 6), got {value!r}")
    return int(k), swapped, v[0] == "d"


GATHER_ROWS_ENV = "K31MK_OPROJ_HHOME_G_ROWS"


def gather_rows(value=None):
    """The two-shot batches HH-G applies to (``K31MK_OPROJ_HHOME_G_ROWS``, a batch table, rows 17-128, values 0 / 1), or None (default:
    every two-shot batch of the buffer). With a table, ``HomedTwoShot`` also allocates a plain symm_mem gather that the other two-shot
    batches use with the served code (d0 faster than served at 96 / 104 rows, slower at 112 / 128)."""
    from ..common.policy import batch_table

    value = os.environ.get(GATHER_ROWS_ENV, "") if value is None else value
    if value == "":
        return None
    table = batch_table(value)
    bad = {b: v for b, v in table.items() if not 17 <= b <= 128 or v not in (0, 1)}
    if bad or not table:
        raise ValueError(f"{GATHER_ROWS_ENV}: rows 17-128 with value 0 or 1, got {value!r}")
    return frozenset(b for b, v in table.items() if v)


def gather_layout(cap, width, k):
    """(1, (2^k, 2, cap / 2^(k+1)), width): token t = a + 2^k s + 2^(k+1) c -> row (a + 2^k c) of segment s."""
    lo = 1 << k
    if cap % (2 * lo):
        raise ValueError(f"{GATHER_ENV}: cap {cap} is not a multiple of {2 * lo}")
    return cute.make_layout((1, (lo, 2, cap // (2 * lo)), width), stride=(cap * width, (width, SEG_ELEMS, lo * width), 1))


@cute.jit
def homed(tensor, k: cutlass.Constexpr[int]):
    """``tensor`` ([1, cap, width] gather parity view) re-laid out for rule bit ``k``."""
    return cute.make_tensor(tensor.iterator, gather_layout(tensor.shape[1], tensor.shape[2], k))


class HomedTwoShot:
    """``common.twoshot.TwoShot`` whose gather sits in two localized segments (one per locality domain, ``mfx_localized.LocalizedSymm``):
    parity p, token t -> segment s(t), element offset s * SEG_ELEMS + p * (cap / 2) * width + row(t) * width. ``recv`` stays
    ``symm_mem`` (unchanged). ``views`` returns the parity base pointers as flat [1, cap, width] views (the kernels re-layout
    them); ``gather`` is [parities, 1, cap / 2, width] so that ``gather[0].numel()`` is the parity stride (CDR's offset).
    ``lds``: the locality domain of segments 0 / 1 on THIS rank (default from the rule; a rank may differ: the multicast addresses by
    offset and each rank backs each offset with its own memory)."""

    def __init__(self, group, device, tp, rank, width=7168, cap=64, parities=2, rule=None, lds=None, recv_dyn=None):
        import torch
        import torch.distributed._symmetric_memory as symm_mem

        from ..common.twoshot import SENTINEL

        self.tp, self.rank, self.cap = tp, rank, cap
        self.recv_dynamic = recv_rule() if recv_dyn is None else bool(recv_dyn)
        self.hr_rows = None  # HH-R row gate (K31MK_OPROJ_HHOME_R_ROWS; set by alloc_recv; None: every two-shot batch homed)
        if self.recv_dynamic:
            self.alloc_recv(group, device, tp, width // tp, cap, parities)
        else:
            self.recv = symm_mem.empty((parities, tp, cap, width // tp), dtype=torch.bfloat16, device=device)
            handle = symm_mem.rendezvous(self.recv, group)
            size = self.recv[0].numel() * 2
            self.peers = torch.tensor([[int(b) + p * size for b in handle.buffer_ptrs] for p in range(parities)],
                                      dtype=torch.int64, device=device)
        self.alloc_gather(group, device, width, cap, parities, rule, lds)
        (self.rflat if self.recv_dynamic else self.recv).view(torch.int32).fill_(SENTINEL)
        self.flat.view(torch.int32).fill_(SENTINEL)
        torch.accelerator.synchronize(device)
        torch.distributed.barrier(group=group)

    def ensure_smtab(self, device):
        import torch

        if getattr(self, "smtab", None) is None:
            idx = device if isinstance(device, int) else torch.device(device).index
            self.smtab, self.sm_margin = sm_ld_table(torch.cuda.current_device() if idx is None else idx)

    def alloc_recv(self, group, device, tp, own, cap, parities, lds=(0, 1)):
        """HH-R receive slots (collective): segment s (tokens t % 2 == s) on locality domain lds[s]; sets rbox, rflat, rpstride, recv
        ([parities, tp, cap / 2, own] for shape checks), peers, tickets_r, smtab. lds (1, 0) = the all-mismatched control (tests)."""
        import torch

        from .mfx_localized import LocalizedSymm

        if parities * tp * (cap // 2) * own > SEG_ELEMS:
            raise ValueError(f"{RECV_ENV}: cap {cap} x {parities} parities does not fit two {SEG_ELEMS}-element segments")
        seg = SEG_ELEMS * 2
        self.rlds = tuple(lds)
        self.rbox = LocalizedSymm(group, device, 2 * seg, self.rlds[0], segments=[(self.rlds[0], seg), (self.rlds[1], seg)])
        self.rflat = self.rbox.tensor((2 * SEG_ELEMS,), torch.bfloat16)
        self.rpstride, self.own = tp * (cap // 2) * own, own
        self.recv = self.rflat[: parities * self.rpstride].view(parities, tp, cap // 2, own)
        self.peers = torch.tensor([[int(b) + p * self.rpstride * 2 for b in self.rbox.buffer_ptrs] for p in range(parities)],
                                  dtype=torch.int64, device=device)
        self.tickets_r = torch.zeros((parities, 2), dtype=torch.int64, device=device)  # 16 B per parity (alignment)
        self.ensure_smtab(device)
        self.hr_rows = recv_rows()  # HH-R row gate (default None: every two-shot batch of this buffer homed)
        if self.hr_rows is not None:  # + the served symm_mem receive slots for the two-shot batches outside the gate (collective)
            self.alloc_recv_plain(group, device, tp, own, cap, parities)

    def alloc_recv_plain(self, group, device, tp, own, cap, parities):
        """HH-R row gate (``K31MK_OPROJ_HHOME_R_ROWS``): the served symm_mem receive slots (collective) of the two-shot batches outside
        the table; sets precv ([parities, tp, cap, own], served layout) and ppeers (per parity, every owner's base), SENTINEL-filled."""
        import torch
        import torch.distributed._symmetric_memory as symm_mem

        from ..common.twoshot import SENTINEL

        self.precv = symm_mem.empty((parities, tp, cap, own), dtype=torch.bfloat16, device=device)
        handle = symm_mem.rendezvous(self.precv, group)
        size = self.precv[0].numel() * 2
        self.ppeers = torch.tensor([[int(b) + p * size for b in handle.buffer_ptrs] for p in range(parities)],
                                   dtype=torch.int64, device=device)
        self.precv.view(torch.int32).fill_(SENTINEL)

    def recv_offset(self, parity, slot, token, local):
        """Element offset of (parity, slot, token, column ``local``) in ``rflat`` (the layout ``recv_layout`` encodes)."""
        own = self.own
        return (token & 1) * SEG_ELEMS + parity * self.rpstride + slot * (self.cap // 2) * own + (token >> 1) * own + local

    def alloc_gather(self, group, device, width, cap, parities, rule=None, lds=None):
        """The homed gather (collective over ``group``); sets rule, lds, box, flat, pstride, gather, gather_mc."""
        import torch

        from .mfx_localized import LocalizedSymm

        self.rule = gather_rule() if rule is None else (rule or None)
        if self.rule is None:  # HH-R only: a plain symm_mem gather (served layout)
            if not getattr(self, "recv_dynamic", False):
                raise ValueError(f"HomedTwoShot needs {GATHER_ENV} or {RECV_ENV}")
            import torch.distributed._symmetric_memory as symm_mem

            self.width, self.cap, self.parities, self.dynamic = width, cap, parities, False
            self.gather = symm_mem.empty((parities, 1, cap, width), dtype=torch.bfloat16, device=device)
            self.gather_mc = int(symm_mem.rendezvous(self.gather, group).multicast_ptr or 0)
            if not self.gather_mc:
                raise RuntimeError("Kimi K3 two-shot all-reduce needs multicast memory")
            self.flat, self.pstride = self.gather.view(-1), cap * width
            self.hh_rows = None  # no homed gather: no row gate
            return
        k, swapped = self.rule[:2]
        if cap % (2 << k) or parities * (cap // 2) * width > SEG_ELEMS:
            raise ValueError(f"{GATHER_ENV}: cap {cap} x {parities} parities does not fit two {SEG_ELEMS}-element segments")
        self.width, self.cap, self.parities = width, cap, parities
        self.dynamic = len(self.rule) > 2 and bool(self.rule[2])
        self.lds = tuple(lds) if lds is not None else ((1, 0) if swapped else (0, 1))
        # dynamic: clusters take tokens of segment = their die, so lds (0, 1) follows; lds (1, 0) is the all-mismatched control (tests)
        seg = SEG_ELEMS * 2  # bytes
        self.box = LocalizedSymm(group, device, 2 * seg, self.lds[0], segments=[(self.lds[0], seg), (self.lds[1], seg)])
        self.flat = self.box.tensor((2 * SEG_ELEMS,), torch.bfloat16)
        self.pstride = (cap // 2) * width
        self.gather = self.flat[: parities * self.pstride].view(parities, 1, cap // 2, width)
        self.gather_mc = int(self.box.multicast_ptr)
        if self.dynamic:  # one ticket word per parity (zero = no cluster yet) and this GPU's SM -> locality domain table
            self.tickets = torch.zeros((parities, 2), dtype=torch.int64, device=device)  # 16 B per parity: the kernels assume 16 B alignment
            self.ensure_smtab(device)
        self.hh_rows = gather_rows()  # row gate (default None: every two-shot batch of this buffer homed)
        if self.hh_rows is not None:  # + the served symm_mem gather for the two-shot batches outside the gate (collective)
            import torch.distributed._symmetric_memory as symm_mem

            from ..common.twoshot import SENTINEL

            self.pgather = symm_mem.empty((parities, 1, cap, width), dtype=torch.bfloat16, device=device)
            self.pgather_mc = int(symm_mem.rendezvous(self.pgather, group).multicast_ptr or 0)
            if not self.pgather_mc:
                raise RuntimeError("Kimi K3 two-shot all-reduce needs multicast memory")
            self.pgather.view(torch.int32).fill_(SENTINEL)

    def offset(self, parity, token):
        """Element offset of (parity, token)'s gather row in ``flat`` (the layout ``gather_layout`` encodes)."""
        k = self.rule[0]
        s, row = (token >> k) & 1, (token & ((1 << k) - 1)) + ((token >> (k + 1)) << k)
        return s * SEG_ELEMS + parity * self.pstride + row * self.width

    def views(self, parity, plain=False, plain_recv=False):
        """(recv, peers, gather base [1, cap, width] flat view, gather multicast address) of layer parity; with HH-R the recv
        view is the flat [tp, cap, own] base of the parity (the kernels re-layout it). ``plain`` (HH-G row gate only): the plain
        symm_mem gather of this parity instead (served layout, ``pgather``). ``plain_recv`` (HH-R row gate only): the plain
        symm_mem receive slots and their peers instead (served layout, ``precv`` / ``ppeers``)."""
        base = parity * self.pstride
        gather = self.flat[base : base + self.cap * self.width].view(1, self.cap, self.width)
        recv, peers = self.recv[parity], self.peers[parity]
        if plain_recv:
            recv, peers = self.precv[parity], self.ppeers[parity]
        elif getattr(self, "recv_dynamic", False):
            rb = parity * self.rpstride
            own = self.own
            recv = self.rflat[rb : rb + self.tp * self.cap * own].view(self.tp, self.cap, own)
        if plain:
            return recv, peers, self.pgather[parity], self.pgather_mc + parity * self.pgather[0].numel() * 2
        return recv, peers, gather, self.gather_mc + base * 2


# ---------------------------------------------------------------------------------------------------------------------------
# HH-G DYNAMIC form (default off; ``K31MK_OPROJ_HHOME_G`` = "d<k>"): the gather layout of rule b<k>, segment s on locality domain s
# on every GPU, and each consumer cluster TAKES a token whose row is homed on its own die, so no placement assumption is needed.
# Per launch and layer parity one 64-bit ticket word (low half: clusters on die 0 so far, high half: die 1): CTA rank 0 of a cluster
# reads its die d from an SM -> domain table (``sm_ld_table``, built once per process), adds 1 << 32 d and gets (r = its rank on die d,
# n = clusters before it). Tokens of segment d: T_d (|T_d| = Q_d). r < Q_d: token T_d[r]; else T_(1-d)[B - 1 - r] (the T_(1-d)
# members that die-(1-d) clusters cannot reach: with n_d > Q_d, n_(1-d) = B - n_d < Q_(1-d) and B - 1 - r covers
# [n_(1-d), Q_(1-d) - 1] exactly). Every token is taken once per launch for any placement. The cluster with n = B - 1 (the last
# ticket) stores 0 into the word; the next launch of this parity is two o-proj layers later (DRA-class cross-grid order).
# The token reaches the other 6 CTAs through rank 0's shared memory after one cluster barrier (release / acquire).
from cutlass import Int32, Uint64  # noqa: E402


@asm(Uint32)
def _smid():
    return "mov.u32 $0, %smid;", "=r", ()


@asm(Uint64)
def _ticket(word, inc):
    return "atom.relaxed.gpu.global.add.u64 $0, [$1], $2;", "=l,l,l,~{memory}", (word, Uint64(inc))


@asm()
def _store_u64(word, value):
    return "st.relaxed.gpu.global.u64 [$0], $1;", "l,l,~{memory}", (word, Uint64(value))


@asm(Uint32)
def _load_rank0(local):
    code = "{ .reg .b32 a; mapa.shared::cluster.u32 a, $1, 0; ld.shared::cluster.u32 $0, [a]; }"
    return code, "=r,r,~{memory}", (local,)


@cute.jit
def _member(s, idx, k: cutlass.Constexpr[int]):
    """The idx-th token (ascending) whose bit k is s."""
    lo: cutlass.Constexpr = 1 << k
    return (idx % lo) + (idx // lo) * (2 * lo) + s * lo


@cute.jit
def _count(B, s, k: cutlass.Constexpr[int]):
    """Tokens t < B with bit k of t == s."""
    lo: cutlass.Constexpr = 1 << k
    full, rem = B // (2 * lo), B % (2 * lo)
    extra = rem - s * lo
    if extra < 0:
        extra = Int32(0)
    if extra > lo:
        extra = Int32(lo)
    return full * lo + extra


@cute.jit
def cluster_token(tickets, smtab, tok, rank, k: cutlass.Constexpr[int]):
    """This cluster's token (see above). ``tickets``: Int64 [1] word of this parity; ``smtab``: Uint8 [256] SM -> domain (> 1 = unknown,
    taken as 0); ``tok``: an Int32 shared word of this CTA; ``rank``: the CTA's rank in the cluster (cluster dims (7, 1, 1))."""
    tid, _, _ = cute.arch.thread_idx()
    _, gy, _ = cute.arch.grid_dim()
    B = Int32(gy)
    if rank == 0:
        if tid == 0:
            d = Int32(smtab[Int32(_smid())])
            if d > 1:
                d = Int32(0)
            word = tickets.iterator.toint()
            old = _ticket(word, Uint64(1) << Uint64(32 * d))
            r = Int32((old >> Uint64(32 * d)) & Uint64(0xFFFFFFFF))
            n = Int32((old & Uint64(0xFFFFFFFF)) + (old >> Uint64(32)))
            token = Int32(0)
            if r < _count(B, d, k):
                token = _member(d, r, k)
            else:
                token = _member(1 - d, B - 1 - r, k)
            if n == B - 1:
                _store_u64(word, Uint64(0))
            tok[0] = token
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()
    return Int32(_load_rank0(tok.iterator))


SM_TABLE_SOURCE = r"""
typedef unsigned int u32;
typedef unsigned long long u64;
__device__ __forceinline__ u32 hh_smid() { u32 r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r; }
__device__ __forceinline__ u64 hh_now() { u64 r; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(r)); return r; }
extern "C" __global__ void hh_chain_init(char* base, const u32* perm, u32 n, u32 stride) {
  for (u32 i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    u64* node = (u64*)(base + (u64)perm[i] * stride);
    *node = (u64)(base + (u64)perm[(i + 1) % n] * stride);
  }
}
extern "C" __global__ void hh_chase(const char* base, const u32* perm, u32 n, u32 stride, int hops, u64* out) {
  extern __shared__ char hh_smem[];
  if (threadIdx.x) return;
  u64 p = (u64)(base + (u64)perm[(u64)blockIdx.x * n / gridDim.x] * stride);
  for (int i = 0; i < 4; i++) asm volatile("ld.global.cg.u64 %0, [%0];" : "+l"(p)::"memory");
  u64 t0 = hh_now();
  for (int i = 0; i < hops; i++) asm volatile("ld.global.cg.u64 %0, [%0];" : "+l"(p)::"memory");
  u64 t1 = hh_now();
  out[3 * blockIdx.x] = hh_smid();
  out[3 * blockIdx.x + 1] = t1 - t0;
  out[3 * blockIdx.x + 2] = p;
}
"""


def sm_ld_table(device, nbytes=64 << 20, hops=400, reps=3):
    """Uint8 [256] SM -> domain of ``device`` (index), 255 where unmeasured: per SM the domain whose localized memory a dependent DRAM chase
    reads faster (a dependent-load latency probe). Also returns the minimum margin."""
    import ctypes

    import torch
    from cuda.bindings import driver as cu

    from sgl_nvidia_mega.common import jit

    from .mfx_localized import _ok, localized_prop

    major, minor = torch.cuda.get_device_capability(device)
    cubin = jit.compile_source(SM_TABLE_SOURCE, "k31_hhome_smtab", f"sm_{major}{minor}a", ("--std=c++17",))
    with torch.cuda.device(device):
        init = jit.load(cubin, "hh_chain_init", (256, 1, 1), 0)
        chase = jit.load(cubin, "hh_chase", (32, 1, 1), 200 * 1024)
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        stream = torch.cuda.current_stream(device).cuda_stream
        lat = []
        for ld in (0, 1):
            prop = localized_prop(device, ld)
            prop.requestedHandleTypes = cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
            flag = cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
            gran = int(_ok(cu.cuMemGetAllocationGranularity(prop, flag), "granularity"))
            size = -(-nbytes // gran) * gran
            handle = _ok(cu.cuMemCreate(size, prop, 0), "cuMemCreate")
            va = int(_ok(cu.cuMemAddressReserve(size, gran, 0, 0), "reserve"))
            _ok(cu.cuMemMap(va, size, 0, handle, 0), "map")
            access = cu.CUmemAccessDesc()
            access.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            access.location.id = device
            access.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            _ok(cu.cuMemSetAccess(va, size, [access], 1), "access")
            try:
                n = size // 256
                gen = torch.Generator(device=f"cuda:{device}")
                gen.manual_seed(19)
                perm = torch.randperm(n, device=f"cuda:{device}", generator=gen).to(torch.int32)
                flush = torch.empty(512 << 20, dtype=torch.uint8, device=f"cuda:{device}")
                out = torch.zeros(3 * sms, dtype=torch.int64, device=f"cuda:{device}")
                c_ = ctypes
                jit.launch(init, (1024, 1, 1), [c_.c_uint64(va), c_.c_uint64(perm.data_ptr()), c_.c_uint32(n), c_.c_uint32(256)], stream)
                samples = {}
                for _ in range(reps):
                    flush.fill_(7)
                    out.zero_()
                    jit.launch(chase, (sms, 1, 1), [c_.c_uint64(va), c_.c_uint64(perm.data_ptr()), c_.c_uint32(n), c_.c_uint32(256),
                                                     c_.c_int32(hops), c_.c_uint64(out.data_ptr())], stream)
                    for smid, ns, _ in out.view(sms, 3).tolist():
                        samples.setdefault(int(smid), []).append(ns / hops)
                torch.cuda.synchronize(device)
                lat.append({s: sorted(v)[len(v) // 2] for s, v in samples.items()})
            finally:
                torch.cuda.synchronize(device)
                cu.cuMemUnmap(va, size)
                cu.cuMemAddressFree(va, size)
                cu.cuMemRelease(handle)
        table = torch.full((256,), 255, dtype=torch.uint8)
        margin = float("inf")
        for s in lat[0]:
            if s in lat[1] and s < 256:
                table[s] = 0 if lat[0][s] < lat[1][s] else 1
                margin = min(margin, abs(lat[0][s] - lat[1][s]))
        return table.to(f"cuda:{device}"), margin


# ---------------------------------------------------------------------------------------------------------------------------
# HH-R DYNAMIC form (DRAFT, default off; ``K31MK_OPROJ_HHOME_R`` = "d0"): the owner receive slots of token t live in localized segment
# t % 2 (one localized segment per domain, s on domain s; parity stride tp * cap / 2 * own), and each producer CTA owner-reduces item
# blocks of its own die's segment. Per launch and parity one 64-bit ticket word (low / high half: CTAs on die 0 / 1 so far): thread 0
# reads its die before griddepcontrol.wait (``smtab`` is static), adds 1 << 32 d after its loads (the result is used only at the owner
# reduce, after the GEMM), and gets r = its rank on die d and n = CTAs before it. Segment s holds Q_s tokens = I_s items = Bk_s blocks
# of 32 = U_s units of R blocks (R = the served owner rounds). P = 112 CTAs >= U_0 + U_1 (checked at build), so: r < U_d -> own unit r;
# r >= P - U_(1-d) -> other-segment unit P - 1 - r (exactly the units die-(1-d) CTAs cannot reach); else idle. n == P - 1 resets the
# word. Pushers address the slots through the same nested layout (``homed_recv``); the poll, sum, multicast and re-arm are
# ``common.twoshot.reduce`` unchanged. Producer ``hhome_r``: 0 off, 1 homed layout for the pushes only (consumer-reduced batches),
# 2 pushes + die-matched owner reduce.
RECV_ENV = "K31MK_OPROJ_HHOME_R"
PRODUCER_CTAS = 112


def recv_rule(value=None):
    """True when ``K31MK_OPROJ_HHOME_R`` = "d0" (default unset: False)."""
    value = os.environ.get(RECV_ENV, "") if value is None else value
    if value not in ("", "d0"):
        raise ValueError(f"{RECV_ENV} must be unset or d0, got {value!r}")
    return value == "d0"


RECV_ROWS_ENV = "K31MK_OPROJ_HHOME_R_ROWS"


def recv_rows(value=None):
    """The two-shot batches HH-R applies to (``K31MK_OPROJ_HHOME_R_ROWS``, a batch table, rows 17-128, values 0 / 1), or None (default:
    every two-shot batch of the buffer). With a table, ``HomedTwoShot`` also allocates the served symm_mem receive slots (``precv`` /
    ``ppeers``) that the other two-shot batches use with the served producer. HH-R is refused on PRW producers, so a MID word with PRW
    at its high rows can carry HH-R only through this table."""
    from ..common.policy import batch_table

    value = os.environ.get(RECV_ROWS_ENV, "") if value is None else value
    if value == "":
        return None
    table = batch_table(value)
    bad = {b: v for b, v in table.items() if not 17 <= b <= 128 or v not in (0, 1)}
    if bad or not table:
        raise ValueError(f"{RECV_ROWS_ENV}: rows 17-128 with value 0 or 1, got {value!r}")
    return frozenset(b for b, v in table.items() if v)


def recv_layout(tp, cap, own):
    """(tp, (2, cap / 2), own): slot r, token t -> segment t % 2, row t // 2 of that segment's [tp, cap / 2, own] plane."""
    if cap % 2:
        raise ValueError(f"{RECV_ENV}: odd cap {cap}")
    return cute.make_layout((tp, (2, cap // 2), own), stride=((cap // 2) * own, (SEG_ELEMS, own), 1))


@cute.jit
def homed_recv(tensor):
    """``tensor`` ([tp, cap, own] receive parity view) re-laid out for HH-R."""
    return cute.make_tensor(tensor.iterator, recv_layout(tensor.shape[0], tensor.shape[1], tensor.shape[2]))


def owner_geometry(batch, per, rounds):
    """(Q, I, Bk, U) per segment for ``batch`` tokens, ``per`` chunks per token, ``rounds`` blocks per unit; raises if P < U_0 + U_1."""
    Q = ((batch + 1) // 2, batch // 2)
    I = tuple(q * per for q in Q)  # noqa: E741
    Bk = tuple(-(-i // 32) for i in I)
    U = tuple(-(-b // rounds) for b in Bk)
    if U[0] + U[1] > PRODUCER_CTAS:
        raise ValueError(f"{RECV_ENV}: batch {batch} needs {U[0] + U[1]} owner units > {PRODUCER_CTAS} CTAs")
    return Q, I, Bk, U


@cute.jit
def owner_token(i, per: cutlass.Constexpr[int], s):
    """Token of item i of segment s's list (tokens t % 2 == s, ascending; ``per`` chunks each)."""
    return (i // per) * 2 + s


@cute.jit
def die_of_sm(smtab):
    """This SM's locality domain from the table (> 1 = unknown -> 0)."""
    d = Int32(smtab[Int32(_smid())])
    if d > 1:
        d = Int32(0)
    return d


@cute.jit
def owner_code(tickets, old, d, U0: cutlass.Constexpr[int], U1: cutlass.Constexpr[int]):
    """Lane 0 of the owner warp: s * 65536 + unit, or -1 (idle), from this CTA's ticket result ``old`` on die ``d``; the last ticket
    resets the word."""
    r = Int32((old >> Uint64(32 * d)) & Uint64(0xFFFFFFFF))
    n = Int32((old & Uint64(0xFFFFFFFF)) + (old >> Uint64(32)))
    ud = Int32(U0)
    uo = Int32(U1)
    if d == 1:
        ud = Int32(U1)
        uo = Int32(U0)
    code = Int32(-1)
    if r < ud:
        code = d * 65536 + r
    elif r >= PRODUCER_CTAS - uo:
        code = (1 - d) * 65536 + (PRODUCER_CTAS - 1 - r)
    if n == PRODUCER_CTAS - 1:
        _store_u64(tickets.iterator.toint(), Uint64(0))
    return code


# ---------------------------------------------------------------------------------------------------------------------------
# HH-M (DRAFT, default off; ``K31MK_OPROJ_HHOME_M`` = "d0"): the ONE-SHOT mailbox of the split batches with 9-16 rows (TP16:
# ``OprojUmma`` producer + groups-4 ``OprojConsumer``; the c=8 B2 verify rounds) moves to a separate homed buffer: token t's rows
# of every rank slot live in localized segment t % 2 (segment s on locality domain s; per segment [parity][TP][cap / 2][width], so
# the parity stride tp * cap / 2 * width is one constant for every word: CDR's offset), and each consumer cluster takes a token of
# its own die's segment by ticket (``cluster_token`` with k = 0: one word per parity, reset by the last cluster). The producer
# multicasts into the same (slot, token, column) words through ``homed_mailbox``; polls, sums, re-arms and fences are unchanged.
# The served mailbox (``prepared.OprojMailbox``) keeps every other batch (fused B<=8, MLA split B<5, the two-shot rows). Measured:
# with the whole mailbox on one locality domain, consumer clusters on the other die finish measurably later at 16 rows.
# Form "s0" (STATIC): no ticket; cluster by of a batch in ``STATIC_PERM`` takes token STATIC_PERM[b][by] (``static_token``), a fixed
# bijection chosen so that each cluster's token sits in the segment of the die that cluster ran on in the rig (hhm1 / hhm3: consumer
# placement at 16 rows static and identical on 8 GPUs of 2 nodes, cluster dies 0101011010101010). Correctness never depends on the
# placement (any bijection is correct); a different served placement only costs speed. Measured (test rig, 16 rows): the
# ticket form d0 costs more for the ticket than it gains from homing; the static form keeps the gain without the ticket.
# The rig placement is a property of the kernel before the pair (the rigs' 1-CTA PDL spin kernel -> the s0 pattern; no
# kernel before -> cluster j on die j % 2), so the static forms are a family that the served placement (trace rider) chooses from:
# "s0" = the pattern 0101011010101010, "s1" = 0101010101010101 (token = cluster), "p<16 digits>" = any pattern with eight 0s and eight
# 1s (cluster j takes the next unused token of parity digit j). Every static form compiles the same kernel kind "s0" with its
# permutation in ``STATIC_PERM`` (set by ``HomedMailbox``; one form per process).
MAILBOX_ENV = "K31MK_OPROJ_HHOME_M"
MBOX_ROWS = 16  # the homed mailbox's rows (split one-shot batches 9-16)
MBOX_SEG_ELEMS = 1 << 21  # one 4 MiB localized segment of BF16 (two parities of [16, 8, 7168] = 3.5 MiB)
STATIC_PERM = {16: (0, 1, 2, 3, 4, 5, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14)}  # form s0: batch -> token of cluster by


STATIC_PATTERNS = {"s0": "0101011010101010", "s1": "0101010101010101"}  # static form -> cluster die pattern at 16 rows


def static_perm(form):
    """The 16-row permutation of static form ``form`` ("s0" / "s1" / "p<16 digits>"): cluster j takes the next unused token whose
    parity (= its mailbox segment) is the j-th digit; a bijection because the pattern has eight 0s and eight 1s."""
    pattern = STATIC_PATTERNS.get(form, form[1:] if form.startswith("p") else "")
    if len(pattern) != MBOX_ROWS or set(pattern) - {"0", "1"} or pattern.count("0") != MBOX_ROWS // 2:
        raise ValueError(f"{MAILBOX_ENV}: static form {form!r} needs s0, s1 or p + 16 digits 0 / 1 with eight 0s")
    nxt = {"0": iter(range(0, MBOX_ROWS, 2)), "1": iter(range(1, MBOX_ROWS, 2))}
    return tuple(next(nxt[c]) for c in pattern)


def mailbox_rule(value=None):
    """The form of ``K31MK_OPROJ_HHOME_M``: "" (default unset: off), "d0" (ticket) or a static form (``static_perm``)."""
    value = os.environ.get(MAILBOX_ENV, "") if value is None else value
    if value not in ("", "d0"):
        static_perm(value)  # raises unless a valid static form
    return value


@cute.jit
def static_token(by, b: cutlass.Constexpr[int]):
    """Form s0: the token of cluster ``by`` (``STATIC_PERM[b]``; batches not listed keep token = by)."""
    token = Int32(by)
    for i in cutlass.range_constexpr(len(STATIC_PERM.get(b, ()))):
        if by == i:
            token = Int32(STATIC_PERM[b][i])
    return token


def mailbox_layout(tp, cap, width):
    """(tp, (2, cap / 2), width): slot r, token t -> segment t % 2, row t // 2 of that segment's [tp, cap / 2, width] plane."""
    if cap % 2:
        raise ValueError(f"{MAILBOX_ENV}: odd cap {cap}")
    return cute.make_layout((tp, (2, cap // 2), width), stride=((cap // 2) * width, (MBOX_SEG_ELEMS, width), 1))


@cute.jit
def homed_mailbox(tensor):
    """``tensor`` ([tp, cap, width] one-shot mailbox parity view) re-laid out for HH-M."""
    return cute.make_tensor(tensor.iterator, mailbox_layout(tensor.shape[0], tensor.shape[1], tensor.shape[2]))


class HomedMailbox:
    """The HH-M one-shot mailbox (collective over ``group``): two localized segments (segment s on domain ``lds[s]``; lds (1, 0) = the
    all-mismatched control, tests), sentinel-armed. ``slot(parity)`` = (flat [tp, cap, width] view at the parity base, which the
    kernels re-layout; the multicast address of that base), as ``prepared.OprojMailbox.slot``; ``pstride`` = the parity stride;
    ``tickets`` = one 64-bit word per parity (16 B apart); ``smtab`` = this GPU's SM -> locality domain table (``sm_ld_table``)."""

    def __init__(self, group, device, tp, rank, width=7168, cap=MBOX_ROWS, parities=2, lds=(0, 1), form="d0"):
        import torch

        from .mfx_localized import LocalizedSymm

        if cap % 2 or parities * tp * (cap // 2) * width > MBOX_SEG_ELEMS:
            raise ValueError(f"{MAILBOX_ENV}: {parities} x [{tp}, {cap} / 2, {width}] does not fit one {MBOX_SEG_ELEMS}-element segment")
        if form != "d0":
            STATIC_PERM[MBOX_ROWS] = static_perm(form)  # before any compile of this process (one form per process)
        self.spec = form  # the env form; ``form`` = the kernel kind: "d0" (ticket) or "s0" (any static permutation)
        form = "d0" if form == "d0" else "s0"
        self.tp, self.rank, self.cap, self.width, self.parities = tp, rank, cap, width, parities
        self.lds, self.form = tuple(lds), form
        # the batches this form homes: d0 every split one-shot batch 9..cap; s0 only those with a static permutation
        self.batches = frozenset(range(9, cap + 1)) if form == "d0" else frozenset(b for b in STATIC_PERM if 9 <= b <= cap)
        seg = MBOX_SEG_ELEMS * 2  # bytes
        self.box = LocalizedSymm(group, device, 2 * seg, self.lds[0], segments=[(self.lds[0], seg), (self.lds[1], seg)])
        self.flat = self.box.tensor((2 * MBOX_SEG_ELEMS,), torch.bfloat16)
        self.mc = int(self.box.multicast_ptr or 0)
        if not self.mc:
            raise RuntimeError(f"{MAILBOX_ENV}: the homed mailbox needs multicast memory")
        self.pstride = tp * (cap // 2) * width
        self.tickets = torch.zeros((parities, 2), dtype=torch.int64, device=device)  # 16 B per parity: the kernels assume 16 B alignment
        idx = device if isinstance(device, int) else torch.device(device).index
        self.smtab, self.sm_margin = sm_ld_table(torch.cuda.current_device() if idx is None else idx)
        self.flat.view(torch.int32).fill_(-0x80000000)  # the sentinel of prepared.OprojMailbox
        torch.accelerator.synchronize(device)
        torch.distributed.barrier(group=group)

    def offset(self, parity, slot, token, column):
        """Element offset of (parity, slot, token, column) in ``flat`` (the layout ``mailbox_layout`` encodes)."""
        row = slot * (self.cap // 2) * self.width + (token >> 1) * self.width + column
        return (token & 1) * MBOX_SEG_ELEMS + parity * self.pstride + row

    def slot(self, parity):
        base = parity * self.pstride
        view = self.flat[base : base + self.tp * self.cap * self.width].view(self.tp, self.cap, self.width)
        return view, self.mc + base * 2

    def segments_ld(self):
        """The driver's locality domain attribute of segments 0 / 1 on this rank."""
        from .mfx_localized import pointer_ld

        return tuple(pointer_ld(self.box.local_ptr + s * MBOX_SEG_ELEMS * 2) for s in (0, 1))
