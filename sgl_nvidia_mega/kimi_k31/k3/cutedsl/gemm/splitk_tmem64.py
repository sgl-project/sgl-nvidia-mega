# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Portions adapted from vLLM (https://github.com/vllm-project/vllm, Apache-2.0), commit
# 642c18d64c4cb6db3ded8fbe1786718cd4005fcb:
# vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/gemm/splitk.py (SplitKGemm), which vLLM
# adapted from FlashInfer v0.7.0.post1 (Apache-2.0): flashinfer/gemm/kernels/dense_bf16_gemm_sm100_splitk.py.
# Modified by NVIDIA: 64 activation rows with the weights in tensor memory, a one-hop DSMEM reduce-scatter and the options listed below.
# Our changes: (1) ``fenced_trigger`` 2 = the idle warp 3 waits, fences (fence.acq_rel.gpu) and
# releases the dependents (our served projection's fix A mode); 0 = the source's weight-warp trigger before any wait (kept as
# a measurement control); (2) K-16: the x warp orders the generic and async proxies for global memory after its PDL
# wait and before its first x TMA (``_fence_proxy_async_global``, as SplitKGemm); (3) ``_copy_to_tmem`` and the
# AttnRes-beside SMEM bound are local (the originals live in kda.head / attn_res.kernel); (4) the 3-D x-map path is dropped;
# (5) a short rank (one K tile fewer) does not load its unused last A slot; (6) the dependents' trigger waits until the
# MMA warp holds its tensor memory (named barrier ``TRIGGER_BARRIER``); (7) (K31MK_MOE_FRONT_N32): an N32
# tile for subclasses that set ``n_allowed`` (every K tile of an 8-way split in tensor memory beside two 32-column
# accumulators, 4 KB x tiles inside an unchanged 112 KB ring); N64 / N80 / N96 / N128 compile as before; (8)
# (65-112 rows): N40 / N48 / N56 tiles for subclasses that set ``any_width`` (x's ring rounded up to whole 16 KB stages; an
# owner sums by (row, quad) items with 16-byte loads when N / 8 tokens per thread is not a whole number of quads); (9)
# (``l2pf``, default 0): an L2 prefetch of the A tiles past the ring's first round before the first ring load; (10)
# (``lean`` / ``tx576``, default 0): a 116 KB shared-memory floor; one exclusive 576-column tensor-memory allocation (all K tiles in TMEM);
# (11)  (``xsep``, default 0): x's tiles do not wait for the whole weight drain (see ``_xsep_plan``);
# (12)  (``rfirst`` class attribute, default 0): a subclass maps each cluster to its logical (tile, N tile)
# (``cluster_tile``), so the clusters the GPU launches first take the tiles it needs first.
"""Rubin split-K projection of 64 activation rows (spec verify B8 x q8) with the
weights on chip before x lands.

Each CTA of a ``split`` cluster owns one K slice of one 128-row weight tile, one CTA
per SM. Each projection takes the largest split whose clusters still fit one wave on the target GPU,
so the KDA rows (26 tiles) take split 6 and the MLA rows
(23 tiles) split 7; the 112 K tiles (64 wide) are spread over the ranks as evenly as
they go (rank r < 112 % split gets one more). A slice's first 14 K tiles are staged
in tensor memory before the PDL wait (the weight warp streams them through a ring of
16 KB stages and the MMA warp copies each with ``tcgen05.cp`` as it lands); the rest
stay in their own shared-memory slots. After the PDL wait, once the ring has
drained, x's K tiles (64 rows x 64, 8 KB each) are loaded into ``slots`` 8 KB slots
of the ring's memory, each on its own barrier (a tile beyond the slots waits for
the MMAs of the tile before it in its slot). The M128 N64 K16 MMAs read A from
tensor memory (measured faster than from shared memory).

Reduce-scatter over the cluster in one DSMEM hop: once every rank's MMAs are done
(a cluster barrier: the owners' mailboxes reuse the A slots and x's slots), each
epilogue thread pushes its accumulator row (one TMEM lane, 64 token columns) to the
owner of its 128 / ``owners`` row share (16-byte ``st.async``); owners sum the
partials in rank order, ((p0 + p1) + p2) + ..., round once to BF16 and store their
rows. Versus ``SplitKGemm`` at (64, 64, 2, stages) this reorders the FP32
accumulation (``split`` partials over K / ``split`` instead of two over K / 2)."""

from typing import Any

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import tcgen05

from ..common.primitives import (
    asm,
    at,
    bits_float,
    float_bits,
    load_tmem,
    mailbox_load,
    mailbox_store,
    named_barrier,
    push_float,
    store_words,
)
from .splitk import (
    SplitKGemm,
    SplitKTactic,
    _acquire_tensor_map,
    _arm_cluster,
    _fence_proxy_async_global,
    _fence_tcgen05,
    _fenced_launch_dependents,
    _smem,
    _wait_cluster,
)

# Shared memory of one ClusterAttnRes CTA as the original attn_res.kernel sizes it (not checked against ours; a sizing guard
# only): BESIDE_SMEM is the largest request (a KB multiple) that fits beside one such CTA in an SM's shared memory.
_ATTNRES_SMEM = 64 * 4 + 3 * 112 * 4 + 3 * 16 * 4 + 3 * 8
BESIDE_SMEM = (228 * 1024 - 2 * 1024 - _ATTNRES_SMEM) // 1024 * 1024

__all__ = ["N_TILES", "SplitKTmemGemm64", "acc_columns"]

# Token columns per N tile: 64 / 128 (the shipped bins), 80 / 96 (65-96 rows: one
# tile of 16-row granularity instead of the generic split-K above 64 rows).
N_TILES = (64, 80, 96, 128)


def acc_columns(n):
    """Tensor-memory columns of an N-column accumulator: whole 32-column A tiles."""
    return (n + 31) // 32 * 32


def _load_row(address, n):
    """``n`` FP32 TMEM columns of this lane from ``address``, as ``tcgen05.ld``
    32x32b loads of descending power-of-two counts (a plain trace-time helper)."""
    values, column, rest = (), 0, n
    while rest:
        count = 1 << (rest.bit_length() - 1)
        part = load_tmem(address + column, count)
        values += part if count > 1 else (part,)
        column, rest = column + count, rest - count
    return values


A_STAGE, B_TILE = 128 * 64 * 2, 64 * 64 * 2  # bytes: one A box, one x tile
TMEM_TILES = 14  # A tiles beside the 64-column accumulator in 512 TMEM columns
OWNERS = 4  # ranks that reduce and store a 32-row share (default)
SHARE = 128 // OWNERS
# Epilogue shared memory (bytes into the A slots + x's slots): the owners' mailboxes
# [split][16 quads][share rows][4] FP32 at 0, an owner's [64 tokens][share]
# output tile at TILE_OFFSET.
TILE_OFFSET = 64 * 1024
# kind::f16 instruction descriptor: D FP32, A and B BF16, both K-major, N 64, M 128
IDESC_M128_N64 = 0x08000490 | (64 >> 3) << 17
# Shared-memory descriptor high word: SBO 1024 B, version 1, 128-byte swizzle
DESC_HIGH = 0x40004040
# Shared memory requested per CTA: above half of the SM's shared memory, so
# no two CTAs of this kernel share an SM (the second waits for the first's tensor
# memory: in a split-8 prototype behind a 112 x 200 KB predecessor, half the K
# splits landed their first weight tile late).
ALONE_SMEM = 170 * 1024
# LEAN (K31MK_MOE_FRONT_LEAN): the per-SM shared memory the device reports in the normal mode leaves no room for a second
# 116 KB CTA (+ 1 KB reserved each), so two such CTAs already cannot share an SM;
# 170 KB also keeps the front off any SM holding four 15.7 KB o-proj consumer CTAs (its PDL predecessor) until they exit.
LEAN_SMEM = 116 * 1024
# Named barrier (64 threads): the MMA warp arrives once its tensor memory is allocated;
# the warp that triggers the dependents (3 in ``fenced_trigger`` 2, the weight warp in 0)
# waits on it first. Barrier 3 is the epilogue's.
TRIGGER_BARRIER = 4
# F-L (``lamx``): the box sentinel word (BF16 0xFFFF pairs; the predecessor packs x with NaN halves canonical, so no x word is
# it), the box's rows per parity (the largest FTM batch), and the polling warps (1, 3, 4-7: poller p takes K tiles p, p + 6, ..).
LAMX_SENTINEL = 0xFFFFFFFF
LAMX_ROWS = 32
LAMX_WARPS = (1, 3, 4, 5, 6, 7)


@asm()
def _tma_box(dst, barrier, tensor_map, k0, row0):
    """One {64, 128} box of A at (k0, row0) to shared address ``dst``, L2
    evict_first, completing on ``barrier``."""
    code = "{ .reg .b64 pol; createpolicy.fractional.L2::evict_first.b64 pol, 1.0; "
    code += "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx"
    code += "::bytes.L2::cache_hint [$0], [$1, {$2, $3}], [$4], pol; }"
    operands = (Int32(dst), tensor_map, Int32(k0), Int32(row0), barrier)
    return code, "r,l,r,r,r,~{memory}", operands


@asm()
def _tmem_alloc_x576(dst):
    """TX576: the SM's whole tensor memory (576 columns, sm_107f) as one exclusive allocation; address to shared ``dst``."""
    return "tcgen05.alloc.exclusive.cta_group::1.sync.aligned.shared::cta.b32 [$0], 576;", "r,~{memory}", (Int32(dst),)


@asm()
def _tmem_dealloc_x576(taddr):
    """TX576: release the exclusive 576-column allocation (same column count, .exclusive)."""
    return "tcgen05.dealloc.exclusive.cta_group::1.sync.aligned.b32 $0, 576;", "r,~{memory}", (Int32(taddr),)


@asm()
def _tma_prefetch_box(tensor_map, k0, row0):
    """L2 prefetch of the box at (k0, row0) of ``tensor_map`` (its box size; no shared memory, no barrier)."""
    code = "cp.async.bulk.prefetch.tensor.2d.L2.global.tile [$0, {$1, $2}];"
    return code, "l,r,r,~{memory}", (tensor_map, Int32(k0), Int32(row0))


@asm()
def _mma_tmem64(d, a, b_desc, accumulate, idesc=IDESC_M128_N64):
    """D[tmem] (+)= A[tmem] B[smem]^T, M128 N64 K16 (``idesc``: N128)."""
    code = "{ .reg .pred p; .reg .b64 bd; .reg .b32 bh; setp.ne.b32 p, $3, 0; "
    code += f"mov.b32 bh, {DESC_HIGH}; mov.b64 bd, {{$2, bh}}; "
    code += f"tcgen05.mma.cta_group::1.kind::f16 [$0], [$1], bd, {idesc}, p; }}"
    operands = (Int32(d), Int32(a), Int32(b_desc), Int32(accumulate))
    return code, "r,r,r,r,~{memory}", operands


@asm()
def _mma_tmem_rn(d, a, b_desc, accumulate, idesc):
    """``_mma_tmem64`` with the instruction descriptor in a register (``runtime_n``: the
    N field is the call's live token count rounded up to 8, at most the compiled N)."""
    code = "{ .reg .pred p; .reg .b64 bd; .reg .b32 bh; setp.ne.b32 p, $3, 0; "
    code += f"mov.b32 bh, {DESC_HIGH}; mov.b64 bd, {{$2, bh}}; "
    code += "tcgen05.mma.cta_group::1.kind::f16 [$0], [$1], bd, $4, p; }"
    operands = (Int32(d), Int32(a), Int32(b_desc), Int32(accumulate), Int32(idesc))
    return code, "r,r,r,r,r,~{memory}", operands


@asm()
def _mma_smem64(d, a_desc, b_desc, accumulate, idesc=IDESC_M128_N64):
    """D[tmem] (+)= A[smem] B[smem]^T, M128 N64 K16 (low descriptor words;
    ``idesc``: N128)."""
    code = "{ .reg .pred p; .reg .b64 ad, bd; .reg .b32 h; setp.ne.b32 p, $3, 0; "
    code += f"mov.b32 h, {DESC_HIGH}; mov.b64 ad, {{$1, h}}; mov.b64 bd, {{$2, h}}; "
    code += f"tcgen05.mma.cta_group::1.kind::f16 [$0], ad, bd, {idesc}, p; }}"
    operands = (Int32(d), Int32(a_desc), Int32(b_desc), Int32(accumulate))
    return code, "r,r,r,r,~{memory}", operands


@asm()
def _copy_to_tmem(t, a_desc):
    """128 rows x 32 bytes of A (one K16 slice) from shared to tensor memory."""
    code = f"{{ .reg .b64 sd; .reg .b32 sh; mov.b32 sh, {DESC_HIGH}; "
    code += "mov.b64 sd, {$1, sh}; tcgen05.cp.cta_group::1.128x256b [$0], sd; }"
    return code, "r,r,~{memory}", (Int32(t), Int32(a_desc))


@asm()
def _st_shared_v4(address, words):
    code = "st.shared.v4.b32 [$0], {$1, $2, $3, $4};"
    return code, "r,r,r,r,r,~{memory}", (Int32(address), *(Int32(w) for w in words))


@asm((Uint32,) * 4)
def _ld_shared_v4(address):
    code = "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];"
    return code, "=r,=r,=r,=r,r,~{memory}", (Int32(address),)




@asm()
def _st_shared_bf16(address, value):
    code = "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; st.shared.b16 [$0], h; }"
    return code, "r,f,~{memory}", (Int32(address), Float32(value))


@asm()
def _cluster_arrive():
    return "barrier.cluster.arrive.release;", "~{memory}", ()


@asm()
def _cluster_wait():
    return "barrier.cluster.wait.acquire;", "~{memory}", ()


@asm()
def _cluster_phases():
    """Arrive on and pass a cluster barrier phase (threads without epilogue work)."""
    code = "barrier.cluster.arrive.release; barrier.cluster.wait.acquire;"
    return code, "~{memory}", ()


class SplitKTmemGemm64(SplitKGemm):
    """``tactic`` (128, N, split, slots), N 64 or 128 token columns (N128: 12 A tiles
    in tensor memory beside a 128-column accumulator, 16 KB x tiles, mailboxes and
    the owner's tile sized by N): ``slots`` x's tiles (also the pre-wait ring, of
    16 KB stages); ``k``: the reduction length. Arguments
    as ``SplitKGemm`` (``bias`` unused)."""

    # N tiles this class accepts. N32 is only for a subclass with its own ``finish`` (``MoeFrontTmem64`` under
    # K31MK_MOE_FRONT_N32): the default ``finish`` stores 64- or 128-token tiles; projection.py's TM64 tables never reach it.
    n_allowed = N_TILES
    # ``any_width`` (N16 / N24 / N40 / N48 / N56; a subclass with ``stage_quad`` and a ``finish`` that takes None): the
    # owners' sums accept N x share / 128 that is not a multiple of 4, as (row, quad) items with 16-byte loads (``owner_quad``).
    any_width = False
    # ``runtime_n`` (K31MK_MOE_FRONT_N32=2; only with every K tile in tensor memory): the MMAs carry
    # N = the tile's live tokens rounded up to 8 (<= the compiled N) in a register descriptor; the epilogue still moves the
    # compiled N columns (columns past N hold stale tensor memory and are never stored: ``finish`` stores live tokens only).
    runtime_n = False
    # ``extra_a_maps``: a subclass whose ``issue_a`` reads another global tensor map for some rows acquires, in
    # ``acquire_a_maps(a_map, row0)``, the one map its ``issue_a`` / ``prefetch_a`` read for ``row0``
    # (``fence.proxy.tensormap::generic.acquire``) instead of ``a_map``; default False.
    extra_a_maps = False
    # ``lamx`` (F-L, K31MK_MOE_FRONT_LAMX; default False = served code; only a subclass that sets ``xbox``
    # (Int64 byte address of a [2][LAMX_ROWS][K] BF16 box, Int32 parity) at trace time before ``body``): x is not loaded by TMA
    # after the PDL wait. The predecessor stores x as Lamport words (LAMX_SENTINEL never occurs: its NaN halves are canonical)
    # into box[parity]; warps 1, 3-7 poll this rank's K tiles there (relaxed gpu loads, all in flight) until no word is the
    # sentinel, then, once the weight ring has drained, write them into x's slots in the UMMA SW128 K-major layout,
    # fence.proxy.async.shared::cta and arrive on b_full[k] (32 arrivals) (``lamx_canary``: a canary spin first). Warp 3 first re-arms its share of box[1 - parity] (the previous ordinal's x) + one gpu fence at entry
    # (the c=1 small-batch chain's front form), and keeps its PDL wait + fenced trigger. The epilogue executes griddepcontrol.wait
    # before ``finish``, so every output store stays after the predecessor's completion (as served). Needs every K tile in its own
    # slot (slots == k_tiles) and no short rank.
    lamx = False
    # ``lamx_canary`` (F-L only): False (served F-L) = full passes from the start; True (measurement form, knob value 2) = each poller first
    # spins on one canary chunk per lane, then full passes: one more L2 round trip after x lands (measured: the canary form gains less than the
    # no-canary form at 16 rows, with or without an L2-bound consumer).
    lamx_canary = False
    # ``lamx_prog`` (F-L only): 0 (served F-L) = a poller writes its tiles once ALL of them are published;
    # 1 = each tile is tested and voted on its own and written + arrived as soon as it is published (every pass still reloads all
    # of the poller's tiles, clamped as in 0); 2 = 1 + a pass reloads only the tiles still pending (one branch per tile group;
    # tiles past k_tiles are never loaded).
    lamx_prog = 0
    # ``rfirst`` (K31MK_MOE_FRONT_RFIRST, read by MoeFrontTmem64): the kernel computes (tile, N tile) from
    # ``cluster_tile`` instead of (block x, block y // split).
    rfirst = 0

    def __init__(
        self, *, tactic: SplitKTactic, k: int, alone: bool = True,
        owners: int = OWNERS, fenced_trigger: int = 2, promote: int = 0, mbox: int = 0, l2pf: int = 0,
        lean: int = 0, tx576: int = 0, xsep: int = 0,
    ):  # fmt: skip
        super().__init__(tactic=tactic)
        # XSEP (K31MK_MOE_FRONT_XSEP; 0 = off): a CTA that enters after its PDL predecessor has ended
        # (the MID front's last clusters, after the o-proj consumer in situ) no longer drains all its weight tiles into
        # tensor memory before x loads and the MMAs start. 1 = each x tile goes to the ring slot that frees earliest and loads once
        # the ring stages under that slot have had their last tensor-memory copy (per-stage ``stage_free`` barriers, committed by the
        # copying warp after a stage's final tcgen05.cp) instead of after the whole drain; warp 2 allocates and copies (committing
        # ``a_ready[k]`` per tile), a ninth warp (warp 8, 288 threads) issues the MMAs in K order (waiting ``a_ready[k]`` and x[k]);
        # warp 3 keeps the served trigger (issuing the MMAs from warp 3 after its fenced trigger measured slower).
        # 2 = 1 plus two x slots of their own (+2 x tiles of shared memory) for the last two K tiles. Same MMAs, K order,
        # accumulators and sums: bitwise expected (the GPU rig decides).
        assert int(xsep) in (0, 1, 2), xsep
        self.xsep = int(xsep)
        self.x_extra = 2 if self.xsep == 2 else 0
        # LEAN (K31MK_MOE_FRONT_LEAN; 0 = off): 1 = the per-CTA shared-memory floor is LEAN_SMEM instead of ALONE_SMEM.
        # TX576 (K31MK_MOE_FRONT_TX576; 0 = off): 1 = one exclusive 576-column tensor-memory allocation (sm_107f) instead
        # of 512, so every K tile of a split sits in tensor memory beside the accumulators up to N64 under PROMOTE 2 (no A slots).
        assert int(lean) in (0, 1) and int(tx576) in (0, 1), (lean, tx576)
        self.lean, self.tx576 = int(lean), int(tx576)
        self.tmem_cols = 576 if self.tx576 else 512
        # L2PF (K31MK_MOE_FRONT_L2PF; 0 = off): 1 = the weight warp first prefetches to L2 the A tiles the
        # ring cannot take in its first round (tiles ``num_ab_stage`` .. ``tmem_tiles`` - 1), so their HBM reads overlap the
        # first round's instead of starting one ring round trip later. A hint only: same loads, same data, same order.
        assert int(l2pf) in (0, 1), l2pf
        self.l2pf = int(l2pf)
        # MBOX (K31MK_MOE_FRONT_MBOX): 1 = the reduce mailboxes get their own shared-memory region after every
        # served buffer, so no peer write can land in bytes this CTA's MMAs still read (the WAR the cluster barrier phase
        # ordered) and that barrier phase is dropped in every warp. Payload edge and FP32 sum order unchanged: bitwise.
        assert int(mbox) in (0, 1), mbox
        self.mbox = int(mbox)
        assert int(fenced_trigger) in (0, 2), "0 = weight warp before any wait (original); 2 = warp 3 after its wait, fenced"
        self.fenced_trigger = int(fenced_trigger)
        # ``promote`` (0 = off, today's single accumulator): every ``promote`` K tiles (4 MMAs each) the
        # MMAs switch to the other of two TMEM accumulators and the epilogue warps add the finished one into FP32
        # registers (CUDA-core round-to-nearest adds). One long TMEM accumulation chain loses accuracy with its length
        # (measured), so promotion bounds the chain.
        # Two accumulators leave 12 A tiles in TMEM at N64 (the rest in their shared-memory slots).
        self.promote = int(promote)
        self.acc_slots = 2 if self.promote else 1
        assert tactic.mma_m == 128 and tactic.mma_n in self.n_allowed, "M128 N64-N128 (N32: a subclass that sets n_allowed)"
        # the owners' sums read whole quads: N x share / 128 tokens per thread, a
        # multiple of 4 (N80 needs share 32, i.e. 4 owners)
        assert tactic.mma_n * (128 // owners) // 128 % 4 == 0 or (self.any_width and tactic.mma_n % 8 == 0), "N x share / 128 % 4"
        # N token columns: the accumulator's TMEM columns (``acc_columns``: N rounded
        # up to whole 32-column A tiles), the TMEM-staged A tiles beside it (14 at
        # N64, 13 at N80 / N96, 12 at N128), x's tile bytes per K tile.
        self.n = tactic.mma_n
        self.acc_columns = acc_columns(self.n)
        self.acc_base = self.acc_slots * self.acc_columns  # TMEM column of the first staged A tile
        self.tmem_tiles = (self.tmem_cols - self.acc_base) // 32
        self.b_tile = self.n * 64 * 2
        self.idesc = 0x08000490 | (self.n >> 3) << 17
        self.cta_k = 64
        self.k_total = k // self.cta_k
        assert self.k_total * self.cta_k == k
        split = self.split_k
        self.k_tiles = -(-self.k_total // split)  # the longest rank's tiles
        self.extra = self.k_total % split  # ranks below it hold k_tiles
        self.owners, self.share = owners, 128 // owners
        if self.n < 64 or self.tx576:  # N32 / TX576: every K tile fits beside the accumulator(s); never more than a rank holds
            self.tmem_tiles = min(self.tmem_tiles, self.k_total // split)
        tmem_tiles = self.tmem_tiles
        assert self.k_tiles >= tmem_tiles and split >= owners and self.share % 4 == 0
        # every rank, short ones included, fills all TMEM_TILES ring stages; a short rank
        # (rank >= extra) skips only its last A slot (``weight_warp64``)
        assert self.k_total // split >= tmem_tiles
        if self.promote:  # every rank holds k_tiles (no short rank): the groups and their phases are the same everywhere
            assert self.extra == 0 and self.promote >= 1, "promote needs k_total % split == 0"
            self.groups = -(-self.k_tiles // self.promote)
        self.smem_tiles = self.k_tiles - tmem_tiles  # A tiles kept in shared memory
        self.slots = tactic.ab_stages  # x's tiles in flight (8 KB each)
        # N32 (K31MK_MOE_FRONT_N32): x's tiles are 4 KB, so the region the pre-wait weight ring shares with
        # them keeps the N64 size (``ring_slots`` 4 KB tiles = the same 16 KB stages); x uses its first ``slots`` tiles.
        # N40 / N48 / N56 (``any_width``): x's 5 / 6 / 7 KB tiles rounded up to at least the N64 size (23 / 19 / 16 slots, 7 stages).
        self.ring_slots = -(-self.slots * B_TILE // self.b_tile) if self.n < 64 else self.slots
        assert self.ring_slots * self.b_tile >= self.slots * B_TILE and self.slots <= self.k_tiles
        assert self.ring_slots * self.b_tile % A_STAGE == 0 or self.any_width
        self.num_ab_stage = self.ring_slots * self.b_tile // A_STAGE  # 16 KB ring stages
        if self.lamx:  # F-L: every K tile in its own x slot (no b_empty reuse), every rank the same tiles
            assert self.slots == self.k_tiles and self.extra == 0 and self.n % 8 == 0 and self.n * 8 % 32 == 0, "lamx"
        self.xbox: Any = None
        assert not (self.lamx and self.xsep), "lamx (F-L) and xsep share x's slots and the MMA warp roles: never built together"
        if self.xsep:
            assert self.fenced_trigger == 2, "xsep: warp 3 issues the MMAs after its PDL wait and the trigger (fenced_trigger 2)"
            assert self.slots >= self.k_tiles, "xsep: every x tile has its own slot (no slot reuse)"
            self.x_slot, self.x_wait = self._xsep_plan()
            self.threads_per_cta = 256 + 32  # warp 8: the MMA warp
        # Shared memory: A slots, x's slots (the ring before), barriers; the owners'
        # mailboxes [split][16 quads][share rows][4] FP32 reuse the first two.
        self.mailbox_bytes = split * self.share * self.n * 4
        self.tile_offset = TILE_OFFSET  # the owner's output tile past the mailboxes
        if self.n != 64:
            self.tile_offset = (self.mailbox_bytes + 1023) // 1024 * 1024
        data = self.smem_tiles * A_STAGE + self.ring_slots * self.b_tile
        assert self.mailbox_bytes <= self.tile_offset
        assert data >= self.tile_offset + self.n * self.share * 4
        used = data + self.mailbox_bytes * int(self.mbox == 1) + 1024 * int(self.mbox == 1) + (2 * self.num_ab_stage + 2 * self.k_tiles + 6 + 4 * (self.acc_slots - 1)) * 8 + 4
        if self.xsep:  # XSEP: the extra x slots (value 2) and the a_ready / stage_free barriers
            used += self.x_extra * self.b_tile + (self.tmem_tiles + self.num_ab_stage) * 8
        self.smem_bytes = (used + 2048 + 1023) // 1024 * 1024
        if alone:  # ``alone`` False: few CTAs, beside small-smem predecessors
            self.smem_bytes = max(self.smem_bytes, LEAN_SMEM if self.lean else ALONE_SMEM)
        assert self.smem_bytes <= BESIDE_SMEM  # the in-projections beside an AttnRes

    def _xsep_plan(self):
        """XSEP: (x_slot, x_wait) per K tile k: the shared-memory x slot of x tile k, and the ring stages whose final weight copy must
        complete (``stage_free``) before x tile k may load there. A stage's final copy is its last K tile below ``tmem_tiles``; a slot is
        free once every stage its bytes overlap is (slots past the stages, or over stages no weight tile uses, at once). The x tiles take
        the earliest-free slots in K order (value 2: the last two K tiles take the two extra slots past the ring)."""
        stages, a_tiles = self.num_ab_stage, self.tmem_tiles
        final = {s: max(k for k in range(a_tiles) if k % stages == s) for s in range(min(stages, a_tiles))}
        ring = stages * A_STAGE
        slots = []
        for slot in range(self.ring_slots):
            lo, hi = slot * self.b_tile, (slot + 1) * self.b_tile
            over = tuple(s for s in range(lo // A_STAGE, min(hi - 1, ring - 1) // A_STAGE + 1) if s in final) if lo < ring else ()
            slots.append((max((final[s] for s in over), default=-1), slot, over))
        slots.sort()
        n_ring = self.k_tiles - self.x_extra
        x_slot = [slot for _, slot, _ in slots[:n_ring]] + [self.ring_slots + e for e in range(self.x_extra)]
        x_wait = [over for _, _, over in slots[:n_ring]] + [() for _ in range(self.x_extra)]
        assert len(set(x_slot)) == self.k_tiles
        return x_slot, x_wait

    @cute.experimental.jit
    def __call__(
        self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, bias: cute.Tensor,
        a_map: cutlass.Int64, stream: _cuda.CUstream,
    ):  # fmt: skip
        self.kernel(a, b, c, bias, a_map).launch(
            grid=(
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(self.smem_bytes),
            stream=stream,
            use_pdl=True,
        )

    @cute.experimental.kernel
    def kernel(
        self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, mS: cute.Tensor,
        a_map: cutlass.Int64,
    ):  # fmt: skip
        """A (M, K, L) and B (N, K, L) K-major, C (M, N, L) M-major."""
        self.body(mA, mB, mS, a_map, (mC,))

    @cute.experimental.jit
    def body(
        self, mA: cute.Tensor, mB: cute.Tensor, mS: cute.Tensor, a_map: cutlass.Int64,
        out,
    ):  # fmt: skip
        """The kernel; ``out``: the epilogue's arguments (``finish``)."""
        stages, k_tiles = self.num_ab_stage, self.k_tiles
        ab_dtype = mA.element_type
        majors = (utils.LayoutEnum.from_tensor(t).mma_major_mode() for t in (mA, mB))
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype, ab_dtype, *majors, self.acc_dtype, self.cta_group,
            self.mma_tiler_mn,
        )  # fmt: skip
        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        bidx, by, l_idx = cute.arch.block_idx()
        split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        n_idx = by // self.split_k
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if cutlass.const_expr(self.rfirst > 0):  # RFIRST (subclass): this cluster's logical (tile, N tile)
            bidx, n_idx = self.cluster_tile(bidx, n_idx)
        # This rank's K tiles: [first, first + tiles)
        base_tiles = self.k_total // self.split_k
        tiles = base_tiles + Int32(split_rank < self.extra)
        over = Int32(split_rank > self.extra) * (split_rank - self.extra)
        first = split_rank * (base_tiles + 1) - over

        # Shared memory: [A slots][x's slots] (x's slots are the pre-wait ring).
        b_shape = (tiled_mma, mnk_tiler, ab_dtype, self.ring_slots + self.x_extra)  # XSEP 2: two x slots past the ring
        b_layout = sm100_utils.make_smem_layout_b(*b_shape)
        if cutlass.const_expr(self.smem_tiles > 0):
            a_slots = _smem(cutlass.Uint8, self.smem_tiles * A_STAGE, 1024)
            data = Int32(a_slots.iterator.toint())
        sB = _smem(ab_dtype, None, 1024, b_layout)
        ring = Int32(sB.iterator.toint())
        if cutlass.const_expr(self.smem_tiles == 0):
            data = ring
        bar_full = _smem(cutlass.Int64, stages, 8).iterator
        bar_empty = _smem(cutlass.Int64, stages, 8).iterator
        a_full = _smem(cutlass.Int64, max(self.smem_tiles, 1), 8).iterator
        b_full = _smem(cutlass.Int64, k_tiles, 8).iterator
        b_empty = _smem(cutlass.Int64, k_tiles, 8).iterator
        bar_drained = _smem(cutlass.Int64, 1, 8).iterator
        bar_mma_epilog = _smem(cutlass.Int64, 1, 8).iterator
        bar_tmem_alloc = _smem(cutlass.Int64, 1, 8).iterator
        bar_reduce = _smem(cutlass.Int64, 1, 8).iterator
        tmem_base_ptr = _smem(cutlass.Int32, 1, 4).iterator
        self.mbox_base = None  # MBOX: the mailboxes' own region, after every served buffer
        if cutlass.const_expr(self.mbox == 1):
            self.mbox_base = Int32(_smem(cutlass.Uint8, self.mailbox_bytes, 1024).iterator.toint())
        accs = None
        if cutlass.const_expr(self.promote > 0):
            # acc_full[j]: accumulator j's group of MMAs done (tcgen05.commit); acc_empty[j]: the epilogue's 128
            # threads have loaded it (CUTLASS's sm100 accumulator pipeline form, PipelineUmmaAsync full / empty)
            acc_full = _smem(cutlass.Int64, 2, 8).iterator
            acc_empty = _smem(cutlass.Int64, 2, 8).iterator
            accs = (acc_full, acc_empty)
        self.accs = accs  # trace time: the MMA warp and the epilogue read it under ``promote``
        a_ready, stage_free = None, None
        if cutlass.const_expr(self.xsep > 0):
            # XSEP: a_ready[k] = A tile k is in tensor memory (tcgen05.commit after its copy); stage_free[s] = ring stage s had its
            # final copy (x may overwrite it). Each completes exactly one phase.
            a_ready = _smem(cutlass.Int64, self.tmem_tiles, 8).iterator
            stage_free = _smem(cutlass.Int64, stages, 8).iterator

        if warp_idx == 0:
            with cute.arch.elect_one():
                if cutlass.const_expr(self.xsep > 0):
                    for i in range(self.tmem_tiles):
                        cute.arch.mbarrier_init(a_ready + i, 1)
                    for i in range(stages):
                        cute.arch.mbarrier_init(stage_free + i, 1)
                if cutlass.const_expr(self.promote > 0):
                    for i in range(2):
                        cute.arch.mbarrier_init(acc_full + i, 1)
                        cute.arch.mbarrier_init(acc_empty + i, 128)
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, 1)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                for i in range(max(self.smem_tiles, 1)):
                    cute.arch.mbarrier_init(a_full + i, 1)
                for i in range(k_tiles):
                    cute.arch.mbarrier_init(b_full + i, 32 if self.lamx else 1)  # F-L: a polling warp's 32 lanes arrive
                    cute.arch.mbarrier_init(b_empty + i, 1)
                cute.arch.mbarrier_init(bar_drained, 1)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)
                cute.arch.mbarrier_init(bar_reduce, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()  # every barrier before any DSMEM traffic
        cute.arch.cluster_wait()
        idesc_rt = None  # ``runtime_n``: the live tokens' N field in a register (else the compiled immediate)
        if cutlass.const_expr(self.runtime_n):
            assert self.smem_tiles == 0, "runtime_n needs every K tile in tensor memory"
            live = cutlass.max(cutlass.min(Int32(mB.layout.shape[0]) - n_idx * self.cta_n, Int32(self.n)), Int32(8))
            idesc_rt = Int32(0x08000490) | (((live + 7) // 8) << 17)

        # Warp 3 is idle; warps 4-7 run the epilogue. One more cluster barrier phase:
        # the mailboxes are free (the epilogue's arrival follows its CTA's MMAs);
        # warps 0-3 pass it after their roles.
        if warp_idx == 0:
            self.weight_warp64(bar_full, bar_empty, a_full, ring, data, a_map,
                               bidx * self.cta_m, first, tiles)  # fmt: skip
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(bar_drained)
            if cutlass.const_expr(self.mbox == 0):
                _cluster_phases()
        elif warp_idx == 1:
            if cutlass.const_expr(self.lamx):  # F-L: poll x's Lamport words instead of the PDL wait + TMA
                self.lamx_poll(Int32(0), b_full, bar_drained, ring, first, Int32(mB.layout.shape[0]), n_idx)
            else:
                b_tile = cute.local_tile(mB, (self.cta_n, self.cta_k), (n_idx, None, l_idx))
                b_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")
                if cutlass.const_expr(self.xsep > 0):  # XSEP: per-stage waits instead of the whole drain
                    self.x_warp_xsep(b_full, stage_free, b_tile, sB, b_map, first, tiles)
                else:
                    cute.arch.mbarrier_wait(bar_drained, 0)
                    self.x_warp64(b_full, b_empty, b_tile, sB, b_map, first, tiles)
            if cutlass.const_expr(self.mbox == 0):
                _cluster_phases()
        elif warp_idx == 2:
            if cutlass.const_expr(self.xsep > 0):  # XSEP: tensor memory and the weight copies only (warp 3 issues the MMAs)
                self.copy_warp_xsep(bar_full, bar_empty, a_ready, stage_free, bar_tmem_alloc, ring, tmem_base_ptr)
            else:
                self.staged_mma_warp64(
                    bar_full, bar_empty, a_full, b_full, b_empty, bar_mma_epilog,
                    bar_tmem_alloc, ring, data, tmem_base_ptr, tiles, idesc_rt,
                )  # fmt: skip
            if cutlass.const_expr(self.mbox == 0):
                _cluster_phases()
        elif warp_idx == 3:
            if cutlass.const_expr(self.lamx):
                # F-L: the previous ordinal's box parity at entry (the c=1 small-batch chain's front form), then this warp's tiles.
                # (Measured: the re-arm + its gpu fence after the trigger delayed warp 3's cluster-barrier arrival, so every front CTA's
                # reduce-scatter waited.)
                self.lamx_rearm(bidx, by)
                self.lamx_poll(Int32(1), b_full, bar_drained, ring, first, Int32(mB.layout.shape[0]), n_idx)
            if cutlass.const_expr(self.fenced_trigger == 2):  # fix A mode 2 (SplitKGemm): wait, fence, trigger
                cute.arch.griddepcontrol_wait()
                named_barrier(TRIGGER_BARRIER, 64)  # the MMA warp holds its 512 TMEM columns
                _fenced_launch_dependents()
            if cutlass.const_expr(self.mbox == 0):
                _cluster_phases()
        else:
            if cutlass.const_expr(self.lamx):
                self.lamx_poll(Int32(warp_idx - 2), b_full, bar_drained, ring, first, Int32(mB.layout.shape[0]), n_idx)
            if cutlass.const_expr(self.xsep > 0):  # XSEP: warp 8 issues the MMAs, in K order, as each A tile and x tile is ready
                if warp_idx == 8:
                    self.mma_warp_xsep(
                        a_ready, a_full, b_full, b_empty, bar_mma_epilog, bar_tmem_alloc, ring, data, tmem_base_ptr, tiles, idesc_rt,
                    )  # fmt: skip
                    if cutlass.const_expr(self.mbox == 0):
                        _cluster_phases()
                else:
                    self.reduce_scatter64(
                        cute.arch.thread_idx()[0] - 128, bar_mma_epilog, bar_tmem_alloc,
                        tmem_base_ptr, data, bar_reduce, split_rank, out,
                        bidx * self.cta_m, n_idx * self.cta_n, l_idx,
                    )  # fmt: skip
            else:
                self.reduce_scatter64(
                    cute.arch.thread_idx()[0] - 128, bar_mma_epilog, bar_tmem_alloc,
                    tmem_base_ptr, data, bar_reduce, split_rank, out,
                    bidx * self.cta_m, n_idx * self.cta_n, l_idx,
                )  # fmt: skip

    def cluster_tile(self, bidx, n_idx):
        """``rfirst`` subclasses: the logical (tile, N tile) of the cluster at grid (``bidx``, ``n_idx``)."""
        raise NotImplementedError("rfirst needs a subclass cluster_tile")

    def issue_a(self, dst, barrier, a_map, k0, row0):
        """One A K tile (128 rows from ``row0``, 64 columns from ``k0``) to ``dst``."""
        _tma_box(dst, barrier, a_map, k0, row0)

    @cute.experimental.jit
    def prefetch_a(self, a_map, k0, row0):
        """L2PF: ``issue_a``'s box at (k0, row0), to L2 only."""
        _tma_prefetch_box(a_map, k0, row0)

    def owner_of(self, row, row0):
        """(owner rank, row in its share) of accumulator row ``row``."""
        return row // self.share, row % self.share

    @cute.experimental.jit
    def weight_warp64(
        self, bar_full, bar_empty, a_full, ring: Int32, data: Int32,
        a_map: cutlass.Int64, row0: cutlass.Int32, first: cutlass.Int32,
        tiles: cutlass.Int32,
    ):  # fmt: skip
        """A, before the PDL wait: one evict_first {64, 128} box per K tile, the
        first TMEM_TILES into ring stage k % stages, the rest into their A slots (a
        short rank's last slot stays empty: nothing waits on it, so a copy there
        could still be in flight when the CTA exits); the dependents' trigger
        (``fenced_trigger`` 0) once every tile is requested and the MMA warp holds
        its tensor memory; then the ring's drain (every ring tile copied to tensor
        memory)."""
        stages = self.num_ab_stage
        if cutlass.const_expr(self.extra_a_maps):
            self.acquire_a_maps(a_map, row0)  # the one map this CTA's issue_a / prefetch_a read (subclass)
        else:
            _acquire_tensor_map(a_map)
        if cutlass.const_expr(self.l2pf == 1):  # L2PF: the tiles past the ring's first round, to L2 now
            with cute.arch.elect_one():
                for k_pf in cutlass.range_constexpr(stages, self.tmem_tiles):
                    if k_pf < tiles:
                        self.prefetch_a(a_map, (first + k_pf) * self.cta_k, row0)
        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(self.tmem_tiles, unroll=1):
            stage = k_tile % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(bar_full + stage, A_STAGE)
                k0 = (first + k_tile) * self.cta_k
                self.issue_a(ring + stage * A_STAGE, bar_full + stage, a_map, k0, row0)
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1
            if cutlass.const_expr(self.smem_tiles > 0):  # noqa: SIM102
                if k_tile == 0:  # the A slots' tiles right behind the first stage
                    with cute.arch.elect_one():
                        for s in cutlass.range_constexpr(self.smem_tiles):
                            if self.tmem_tiles + s < tiles:
                                k_a = (first + self.tmem_tiles + s) * self.cta_k
                                dst = data + s * A_STAGE
                                cute.arch.mbarrier_arrive_and_expect_tx(a_full + s, A_STAGE)
                                self.issue_a(dst, a_full + s, a_map, k_a, row0)
        if cutlass.const_expr(self.fenced_trigger == 0):
            named_barrier(TRIGGER_BARRIER, 64)  # the MMA warp holds its 512 TMEM columns
            cute.arch.griddepcontrol_launch_dependents()
        self.drain(bar_empty, empty_phase, self.tmem_tiles)

    @cute.experimental.jit
    def x_warp64(
        self, b_full, b_empty, g_tile: cute.Tensor, s_tile: cute.Tensor,
        cta_v_map: cute.Layout, first: cutlass.Int32, tiles: cutlass.Int32,
    ):  # fmt: skip
        """Loads x's K tiles after the PDL wait, each on its own barrier; tile
        k >= slots waits for the MMAs of tile k - slots (its slot's previous tile)."""
        tile_bytes = cute.size_in_bytes(
            s_tile.element_type, cute.slice_(s_tile.layout, (None, None, None, 0))
        )
        cute.arch.griddepcontrol_wait()
        _fence_proxy_async_global()  # K-16: the predecessor wrote x with generic stores; TMA reads it
        for k_tile in cutlass.range(tiles, unroll=1):
            slot = k_tile % self.slots
            if k_tile >= self.slots:
                cute.arch.mbarrier_wait(b_empty + k_tile - self.slots, 0)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(b_full + k_tile, tile_bytes)
            cute_ext.tma_load(
                g_tile[None, None, first + k_tile],
                s_tile[None, None, None, slot],
                (b_full + k_tile).value,
                cta_v_map=cta_v_map,
                tma_operation_type=cute_ext.OperationTypeEnum.SM90_TMA_LOAD,
                update_expect_tx=False,
            )

    @cute.experimental.jit
    def lamx_poll(
        self, p: cutlass.Int32, b_full, bar_drained, ring: Int32, first: cutlass.Int32,
        tokens: cutlass.Int32, n_idx: cutlass.Int32,
    ):  # fmt: skip
        """F-L poller ``p`` (warp LAMX_WARPS[p]): this rank's K tiles p, p + 6, .. (< k_tiles) of x from box[parity] (every
        16-byte chunk in flight each pass, until no word of a row below ``tokens`` is LAMX_SENTINEL), then, after the ring
        drain, into x's slots (SW128 K-major: 8-row atoms of 1024 B, chunk c of row r at r % 8 * 128 + (c ^ r % 8) * 16),
        fence.proxy.async.shared::cta, b_full[k] arrive (all 32 lanes). Rows at or past ``tokens`` (served TMA: zero fill) hold a
        copy of row tokens - 1 instead: finite, and an MMA column depends only on its own B row, so they reach only accumulator
        columns that are never stored (live columns stay bitwise)."""
        nw: cutlass.Constexpr = len(LAMX_WARPS)
        nt: cutlass.Constexpr = -(-self.k_tiles // nw)  # tiles per poller (the last ones may hold fewer)
        per: cutlass.Constexpr = self.n * 8 // 32  # 16-byte chunks of one tile per lane
        width: cutlass.Constexpr = self.k_total * self.cta_k
        xaddr, par = self.xbox
        lane = cute.arch.lane_idx()
        col0 = n_idx * self.cta_n
        words = cute.make_rmem_tensor((nt, per, 4), Uint32)
        # canary: one 16-byte chunk per lane (its first tile's last row chunk) until every lane's is published, so the full passes
        # below (all of this CTA's slice, 9 CTAs per slice) do not spin over L2 while x is still being produced
        if cutlass.const_expr(self.lamx_canary):
            k0 = cutlass.min(p, Int32(self.k_tiles - 1))
            qc = lane + 32 * (per - 1)
            rowc = cutlass.min(col0 + (qc >> 3), tokens - 1)
            offc = ((par * LAMX_ROWS + rowc) * width + (first + k0) * self.cta_k + (qc & 7) * 8) * 2
            deadc = p >= self.k_tiles
            seen = False
            while not seen:
                canary = mailbox_load(xaddr + Int64(offc), count=4)
                okc = deadc | ((canary[0] != Uint32(LAMX_SENTINEL)) & (canary[1] != Uint32(LAMX_SENTINEL))
                               & (canary[2] != Uint32(LAMX_SENTINEL)) & (canary[3] != Uint32(LAMX_SENTINEL)))
                seen = cute.arch.vote_all_sync(okc)
        if cutlass.const_expr(self.lamx_prog == 0):
            ready = False
            while not ready:
                # every load of the pass issued before any word is tested (no branch between loads: a load inside a per-chunk
                # branch whose value is tested in the same branch serialized the pass); a chunk past ``tokens`` or a
                # tile past k_tiles re-reads a live address (clamped) and counts as published
                for t in cutlass.range_constexpr(nt):
                    k = cutlass.min(p + nw * t, Int32(self.k_tiles - 1))
                    for i in cutlass.range_constexpr(per):
                        q = lane + 32 * i
                        row = cutlass.min(col0 + (q >> 3), tokens - 1)
                        off = ((par * LAMX_ROWS + row) * width + (first + k) * self.cta_k + (q & 7) * 8) * 2
                        got = mailbox_load(xaddr + Int64(off), count=4)
                        for j in cutlass.range_constexpr(4):
                            words[t, i, j] = got[j]
                valid = True
                for t in cutlass.range_constexpr(nt):
                    for i in cutlass.range_constexpr(per):
                        dead = (col0 + ((lane + 32 * i) >> 3) >= tokens) | (p + nw * t >= self.k_tiles)
                        for j in cutlass.range_constexpr(4):
                            valid = valid & ((words[t, i, j] != Uint32(LAMX_SENTINEL)) | dead)
                ready = cute.arch.vote_all_sync(valid)
            cute.arch.mbarrier_wait(bar_drained, 0)  # x's slots are the weight ring's memory
            for t in cutlass.range_constexpr(nt):
                k = p + nw * t
                if k < self.k_tiles:
                    slot = ring + k * self.b_tile  # slots == k_tiles: tile k in slot k
                    for i in cutlass.range_constexpr(per):
                        q = lane + 32 * i
                        n = q >> 3
                        r8 = n & 7
                        dst = slot + ((n >> 3) << 10) + (r8 << 7) + (((q & 7) ^ r8) << 4)
                        _st_shared_v4(dst, tuple(words[t, i, j] for j in range(4)))
                    cute.arch.fence_proxy("async.shared", space="cta")  # these generic writes -> the MMA's async-proxy reads
                    cute.arch.mbarrier_arrive(b_full + k)
        else:
            # lamx_prog 1 / 2: tile t of this poller is written + arrived as soon as its own words are published (bit t of
            # ``pend``, warp-uniform: it changes only on a vote), so a late sibling tile no longer holds an early one back.
            # (branch bodies call plain helpers: names bound inside a dynamic branch must not be rebound in a sibling branch)
            nlive = (Int32(self.k_tiles) - p + (nw - 1)) // nw  # tiles p, p + nw, .. below k_tiles (0 if p >= k_tiles)
            pend = (Int32(1) << nlive) - 1
            while pend != 0:
                for t in cutlass.range_constexpr(nt):  # every load of the pass before any test
                    k = cutlass.min(p + nw * t, Int32(self.k_tiles - 1))
                    if cutlass.const_expr(self.lamx_prog == 2):
                        if ((pend >> t) & 1) != 0:  # one branch per tile group: only pending tiles are reloaded
                            self._lamx_load(words, t, k, lane, col0, tokens, xaddr, par, first, per, width)
                    else:
                        self._lamx_load(words, t, k, lane, col0, tokens, xaddr, par, first, per, width)
                for t in cutlass.range_constexpr(nt):
                    if ((pend >> t) & 1) != 0:
                        if self._lamx_ready(words, t, lane, col0, tokens, per):
                            cute.arch.mbarrier_wait(bar_drained, 0)  # x's slots are the weight ring's memory
                            self._lamx_write(words, t, p + nw * t, ring, lane, per, b_full)
                            pend = pend & Int32(~(1 << t))

    def _lamx_load(self, words, t, k, lane, col0, tokens, xaddr, par, first, per, width):
        """lamx_prog 1 / 2: every 16-byte chunk of x tile ``k`` for this lane into words[t] (all loads issued, none tested)."""
        for i in range(per):
            q = lane + 32 * i
            row = cutlass.min(col0 + (q >> 3), tokens - 1)
            off = ((par * LAMX_ROWS + row) * width + (first + k) * self.cta_k + (q & 7) * 8) * 2
            got = mailbox_load(xaddr + Int64(off), count=4)
            for j in range(4):
                words[t, i, j] = got[j]

    def _lamx_ready(self, words, t, lane, col0, tokens, per):
        """lamx_prog 1 / 2: the warp vote 'no word of words[t] in a live row is LAMX_SENTINEL'."""
        valid = True
        for i in range(per):
            dead = col0 + ((lane + 32 * i) >> 3) >= tokens
            for j in range(4):
                valid = valid & ((words[t, i, j] != Uint32(LAMX_SENTINEL)) | dead)
        return cute.arch.vote_all_sync(valid)

    def _lamx_write(self, words, t, k, ring, lane, per, b_full):
        """lamx_prog 1 / 2: words[t] into x slot ``k`` (SW128 K-major, as lamx_prog 0), proxy fence, b_full[k] arrive."""
        slot = ring + k * self.b_tile
        for i in range(per):
            q = lane + 32 * i
            n = q >> 3
            r8 = n & 7
            dst = slot + ((n >> 3) << 10) + (r8 << 7) + (((q & 7) ^ r8) << 4)
            _st_shared_v4(dst, tuple(words[t, i, j] for j in range(4)))
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.mbarrier_arrive(b_full + k)

    @cute.experimental.jit
    def lamx_rearm(self, bidx: cutlass.Int32, by: cutlass.Int32):
        """F-L: this CTA's share (16-byte chunks cta, cta + CTAs, ..; 32 lanes) of box[1 - parity], every row, to the
        sentinel, then one gpu-scope acq_rel fence (the c=1 small-batch chain's front re-arm form)."""
        width: cutlass.Constexpr = self.k_total * self.cta_k
        chunks: cutlass.Constexpr = LAMX_ROWS * width // 8
        xaddr, par = self.xbox
        grid_x, grid_y, _ = cute.arch.grid_dim()
        other = xaddr + Int64((1 - par) * LAMX_ROWS * width * 2)
        q = (bidx + grid_x * by) * 32 + cute.arch.lane_idx()
        step = grid_x * grid_y * 32
        sentinel = Uint32(LAMX_SENTINEL)
        while q < chunks:
            mailbox_store(other + Int64(q) * 16, (sentinel, sentinel, sentinel, sentinel))
            q = q + step
        cute.arch.fence_acq_rel_gpu()

    @cute.experimental.jit
    def staged_mma_warp64(
        self, bar_full, bar_empty, a_full, b_full, b_empty, bar_mma_epilog,
        bar_tmem_alloc, ring: Int32, data: Int32, tmem_base_ptr,
        tiles: cutlass.Int32, idesc_rt=None,
    ):  # fmt: skip
        """The ring's tiles to tensor memory as they land (each stage released for
        the next tile), then every tile's four MMAs in K order once x's tile has
        landed: A from tensor memory for the first TMEM_TILES, then from the A
        slots."""
        stages, k_blocks = self.num_ab_stage, self.cta_k // 16
        a_step, b_step = A_STAGE // 16, self.b_tile // 16  # descriptor units
        tmem_tiles, acc_columns = self.tmem_tiles, self.acc_columns
        columns = self.cta_k // 2  # tensor-memory columns of one A tile
        if cutlass.const_expr(self.tx576 == 1):
            _tmem_alloc_x576(Int32(tmem_base_ptr.toint()))
        else:
            cute.arch.alloc_tmem(512, tmem_base_ptr, is_two_cta=False)
        _fence_tcgen05(before=True)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        # The triggering warp waits for this before releasing the dependents: a
        # dependent CTA on this SM then cannot take tensor memory ahead of us.
        named_barrier(TRIGGER_BARRIER, 64, arrive=True)
        if cutlass.const_expr(self.tx576 == 0):  # an exclusive allocation holds the permit until its dealloc
            cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)
        _fence_tcgen05()
        base = Int32(cute.make_tensor(tmem_base_ptr, cute.make_layout(1))[0])
        ring_desc = ((ring >> 4) & 0x3FFF) | 0x10000
        data_desc = ((data >> 4) & 0x3FFF) | 0x10000

        for k in cutlass.range_constexpr(tmem_tiles):
            stage, phase = k % stages, (k // stages) % 2
            cute.arch.mbarrier_wait(bar_full + stage, phase)
            _fence_tcgen05()
            with cute.arch.elect_one():
                for kb in cutlass.range_constexpr(k_blocks):
                    source = ring_desc + stage * a_step + 2 * kb
                    _copy_to_tmem(base + self.acc_base + columns * k + 8 * kb, source)
                tcgen05.commit(bar_empty + stage, None, self.cta_group)
        for k in cutlass.range_constexpr(self.k_tiles):
            if k < tiles:
                cute.arch.mbarrier_wait(b_full + k, 0)
                if cutlass.const_expr(k >= tmem_tiles):
                    cute.arch.mbarrier_wait(a_full + k - tmem_tiles, 0)
                first_k, acc = k == 0, base
                if cutlass.const_expr(self.promote > 0):  # FTM-P: group k // promote into accumulator (k // promote) % 2
                    group = k // self.promote
                    first_k, acc = k % self.promote == 0, base + (group % 2) * acc_columns
                    if cutlass.const_expr(first_k and group >= 2):  # the epilogue has loaded this accumulator's last group
                        cute.arch.mbarrier_wait(self.accs[1] + group % 2, (group // 2 - 1) % 2)
                _fence_tcgen05()
                with cute.arch.elect_one():
                    for kb in cutlass.range_constexpr(k_blocks):
                        b = ring_desc + (k % self.slots) * b_step + 2 * kb
                        accumulate = Int32(0 if first_k and kb == 0 else 1)
                        if cutlass.const_expr(k < tmem_tiles and self.runtime_n):
                            a = base + self.acc_base + columns * k + 8 * kb
                            _mma_tmem_rn(acc, a, b, accumulate, idesc_rt)
                        elif cutlass.const_expr(k < tmem_tiles):
                            a = base + self.acc_base + columns * k + 8 * kb
                            _mma_tmem64(acc, a, b, accumulate, idesc=self.idesc)
                        else:
                            a = data_desc + (k - tmem_tiles) * a_step + 2 * kb
                            _mma_smem64(acc, a, b, accumulate, idesc=self.idesc)
                    if cutlass.const_expr(k + self.slots < self.k_tiles):
                        tcgen05.commit(b_empty + k, None, self.cta_group)
                    if cutlass.const_expr(self.promote > 0):  # noqa: SIM102
                        if cutlass.const_expr((k + 1) % self.promote == 0 or k == self.k_tiles - 1):
                            tcgen05.commit(self.accs[0] + (k // self.promote) % 2, None, self.cta_group)
        with cute.arch.elect_one():
            tcgen05.commit(bar_mma_epilog, None, self.cta_group)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)  # the release lands in phase 1
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 1)
        _fence_tcgen05()
        if cutlass.const_expr(self.tx576 == 1):
            _tmem_dealloc_x576(base)
        else:
            tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
            cute.arch.dealloc_tmem(tmem_ptr, 512, is_two_cta=False)

    @cute.experimental.jit
    def x_warp_xsep(
        self, b_full, stage_free, g_tile: cute.Tensor, s_tile: cute.Tensor,
        cta_v_map: cute.Layout, first: cutlass.Int32, tiles: cutlass.Int32,
    ):  # fmt: skip
        """XSEP: x's K tiles after the PDL wait, tile k into slot ``x_slot[k]`` once the ring stages under it had their final
        tensor-memory copy (``stage_free``, one phase each; the same consumer-release -> TMA-overwrite edge as the weight ring's
        ``bar_empty`` refill), each on its own barrier."""
        tile_bytes = cute.size_in_bytes(
            s_tile.element_type, cute.slice_(s_tile.layout, (None, None, None, 0))
        )
        cute.arch.griddepcontrol_wait()
        _fence_proxy_async_global()  # K-16: the predecessor wrote x with generic stores; TMA reads it
        for k in cutlass.range_constexpr(self.k_tiles):
            if k < tiles:
                for s in cutlass.range_constexpr(len(self.x_wait[k])):
                    cute.arch.mbarrier_wait(stage_free + self.x_wait[k][s], 0)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(b_full + k, tile_bytes)
                cute_ext.tma_load(
                    g_tile[None, None, first + k],
                    s_tile[None, None, None, self.x_slot[k]],
                    (b_full + k).value,
                    cta_v_map=cta_v_map,
                    tma_operation_type=cute_ext.OperationTypeEnum.SM90_TMA_LOAD,
                    update_expect_tx=False,
                )

    @cute.experimental.jit
    def copy_warp_xsep(
        self, bar_full, bar_empty, a_ready, stage_free, bar_tmem_alloc, ring: Int32, tmem_base_ptr,
    ):  # fmt: skip
        """XSEP warp 2: allocates tensor memory, copies each ring tile into it as it lands (committing the stage's release
        ``bar_empty``, the tile's ``a_ready`` and, at a stage's final copy, ``stage_free``), then deallocates once the epilogue
        has loaded the accumulators (the same ``bar_tmem_alloc`` handshake as ``staged_mma_warp64``)."""
        stages, k_blocks = self.num_ab_stage, self.cta_k // 16
        a_step = A_STAGE // 16
        tmem_tiles = self.tmem_tiles
        columns = self.cta_k // 2
        if cutlass.const_expr(self.tx576 == 1):
            _tmem_alloc_x576(Int32(tmem_base_ptr.toint()))
        else:
            cute.arch.alloc_tmem(512, tmem_base_ptr, is_two_cta=False)
        _fence_tcgen05(before=True)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        named_barrier(TRIGGER_BARRIER, 64, arrive=True)  # warp 3's trigger waits for the allocation (as in staged_mma_warp64)
        if cutlass.const_expr(self.tx576 == 0):
            cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)
        _fence_tcgen05()
        base = Int32(cute.make_tensor(tmem_base_ptr, cute.make_layout(1))[0])
        ring_desc = ((ring >> 4) & 0x3FFF) | 0x10000
        for k in cutlass.range_constexpr(tmem_tiles):
            stage, phase = k % stages, (k // stages) % 2
            cute.arch.mbarrier_wait(bar_full + stage, phase)
            _fence_tcgen05()
            with cute.arch.elect_one():
                for kb in cutlass.range_constexpr(k_blocks):
                    source = ring_desc + stage * a_step + 2 * kb
                    _copy_to_tmem(base + self.acc_base + columns * k + 8 * kb, source)
                tcgen05.commit(bar_empty + stage, None, self.cta_group)
                tcgen05.commit(a_ready + k, None, self.cta_group)
                if cutlass.const_expr(k + stages >= tmem_tiles):  # the stage's final copy: x may take its bytes
                    tcgen05.commit(stage_free + stage, None, self.cta_group)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)  # the release lands in phase 1
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 1)
        _fence_tcgen05()
        if cutlass.const_expr(self.tx576 == 1):
            _tmem_dealloc_x576(base)
        else:
            tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
            cute.arch.dealloc_tmem(tmem_ptr, 512, is_two_cta=False)

    @cute.experimental.jit
    def mma_warp_xsep(
        self, a_ready, a_full, b_full, b_empty, bar_mma_epilog, bar_tmem_alloc, ring: Int32, data: Int32, tmem_base_ptr,
        tiles: cutlass.Int32, idesc_rt=None,
    ):  # fmt: skip
        """XSEP warp 8: once the tensor memory is allocated (``bar_tmem_alloc`` phase 0, as the epilogue waits for it), every
        tile's four MMAs in K order once its A tile is in tensor memory (``a_ready``) or its A slot landed, and x's tile landed in
        ``x_slot[k]``; the same MMAs, accumulators, promote groups and commits as ``staged_mma_warp64``. It does not arrive on
        ``bar_tmem_alloc`` (phase 1 = warp 2 + the epilogue, after the accumulators are loaded)."""
        k_blocks = self.cta_k // 16
        a_step, b_step = A_STAGE // 16, self.b_tile // 16  # descriptor units
        tmem_tiles, acc_columns = self.tmem_tiles, self.acc_columns
        columns = self.cta_k // 2
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        _fence_tcgen05()
        base = Int32(cute.make_tensor(tmem_base_ptr, cute.make_layout(1))[0])
        ring_desc = ((ring >> 4) & 0x3FFF) | 0x10000
        data_desc = ((data >> 4) & 0x3FFF) | 0x10000
        for k in cutlass.range_constexpr(self.k_tiles):
            if k < tiles:
                if cutlass.const_expr(k < tmem_tiles):
                    cute.arch.mbarrier_wait(a_ready + k, 0)
                cute.arch.mbarrier_wait(b_full + k, 0)
                if cutlass.const_expr(k >= tmem_tiles):
                    cute.arch.mbarrier_wait(a_full + k - tmem_tiles, 0)
                first_k, acc = k == 0, base
                if cutlass.const_expr(self.promote > 0):  # FTM-P: group k // promote into accumulator (k // promote) % 2
                    group = k // self.promote
                    first_k, acc = k % self.promote == 0, base + (group % 2) * acc_columns
                    if cutlass.const_expr(first_k and group >= 2):  # the epilogue has loaded this accumulator's last group
                        cute.arch.mbarrier_wait(self.accs[1] + group % 2, (group // 2 - 1) % 2)
                _fence_tcgen05()
                with cute.arch.elect_one():
                    for kb in cutlass.range_constexpr(k_blocks):
                        b = ring_desc + self.x_slot[k] * b_step + 2 * kb
                        accumulate = Int32(0 if first_k and kb == 0 else 1)
                        if cutlass.const_expr(k < tmem_tiles and self.runtime_n):
                            a = base + self.acc_base + columns * k + 8 * kb
                            _mma_tmem_rn(acc, a, b, accumulate, idesc_rt)
                        elif cutlass.const_expr(k < tmem_tiles):
                            a = base + self.acc_base + columns * k + 8 * kb
                            _mma_tmem64(acc, a, b, accumulate, idesc=self.idesc)
                        else:
                            a = data_desc + (k - tmem_tiles) * a_step + 2 * kb
                            _mma_smem64(acc, a, b, accumulate, idesc=self.idesc)
                    if cutlass.const_expr(self.promote > 0):  # noqa: SIM102
                        if cutlass.const_expr((k + 1) % self.promote == 0 or k == self.k_tiles - 1):
                            tcgen05.commit(self.accs[0] + (k // self.promote) % 2, None, self.cta_group)
        with cute.arch.elect_one():
            tcgen05.commit(bar_mma_epilog, None, self.cta_group)

    @cute.experimental.jit
    def reduce_scatter64(
        self, epi_tid: cutlass.Int32, bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr,
        data: Int32, bar_reduce, split_rank: cutlass.Int32, out,
        row0: cutlass.Int32, col0: cutlass.Int32, l_idx: cutlass.Int32,
    ):  # fmt: skip
        """Accumulator row ``epi_tid`` (64 FP32 token columns) -> slot
        ``split_rank`` of its owner's mailbox [split][16 quads][SHARE rows][4]
        (``st.async`` after the cluster barrier); owners sum the slots in rank order
        and ``finish``."""
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        _fence_tcgen05()
        base = Int32(cute.make_tensor(tmem_base_ptr, cute.make_layout(1))[0])
        quadrant = epi_tid // 32  # warps 4-7 read TMEM lanes 32 (warp % 4) ..
        if cutlass.const_expr(self.promote > 0):
            # FTM-P: each group's accumulator into FP32 registers as it completes, summed in group order (RN adds);
            # an accumulator the MMA warp reuses is released once loaded
            for group in cutlass.range_constexpr(self.groups):
                cute.arch.mbarrier_wait(self.accs[0] + group % 2, (group // 2) % 2)
                _fence_tcgen05()
                loaded = _load_row(base + (quadrant * 32 << 16) + (group % 2) * self.acc_columns, self.n)
                if cutlass.const_expr(group + 2 < self.groups):
                    _fence_tcgen05(before=True)
                    cute.arch.mbarrier_arrive(self.accs[1] + group % 2)
                if cutlass.const_expr(group == 0):
                    values = loaded
                else:
                    values = tuple(values[i] + loaded[i] for i in range(self.n))
        else:
            cute.arch.mbarrier_wait(bar_mma_epilog, 0)
            _fence_tcgen05()
            values = _load_row(base + (quadrant * 32 << 16), self.n)
        _fence_tcgen05(before=True)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)

        split, share = self.split_k, self.share
        mb = data  # the mailboxes' base (served: the A slots + x's slots; MBOX: its own region)
        if cutlass.const_expr(self.mbox == 1):
            mb = self.mbox_base
        quads: cutlass.Constexpr = self.n // 4  # a row's token quads
        slot_bytes = quads * share * 16
        if (epi_tid == 0) & (split_rank < self.owners):
            expected = (split - 1) * slot_bytes
            if cutlass.const_expr(self.mbox == 1 and self.k23_arm):
                # K-23 (MBOX read): no cluster barrier precedes this arm, so peers' st.async complete_tx may
                # land first; arm at cluster scope.
                _arm_cluster(bar_reduce, expected)
            else:
                cute.arch.mbarrier_arrive_and_expect_tx(bar_reduce, expected)
        if cutlass.const_expr(self.mbox == 0):
            _cluster_arrive()  # this CTA's MMAs are done: its slots may take partials
            _cluster_wait()
        # The epilogue's own inputs, loaded under the reduce-scatter (None: none;
        # after the cluster barrier, whose release would wait for the loads).
        early = self.finish_inputs(epi_tid, out, row0 + split_rank * self.share, col0)
        finish_out = out if early is None else (*out, early)
        owner, r = self.owner_of(epi_tid, row0)
        slot = mb + (split_rank * quads * share + r) * 16  # at the owner's address
        mailbox = cute.make_ptr(Float32, slot, cute.AddressSpace.smem, assumed_align=16)
        if owner == split_rank:
            for i in cutlass.range_constexpr(quads):
                bits = tuple(float_bits(values[4 * i + j]) for j in range(4))
                _st_shared_v4(slot + i * share * 16, bits)
        else:
            for i in cutlass.range_constexpr(quads):
                quad = tuple(values[4 * i + j] for j in range(4))
                push_float(mailbox + i * share * 4, quad, bar_reduce, owner)
        if split_rank < self.owners:
            named_barrier(3, 128)  # this CTA's own slot
            _wait_cluster(bar_reduce, Int32(0))
            # thread -> (row rr, tokens width g .. width (g + 1) - 1)
            width: cutlass.Constexpr = self.n * share // 128
            if cutlass.const_expr(width % 4 == 0):
                rr, g = epi_tid % share, epi_tid // share
                totals = cute.make_rmem_tensor(width, Float32)
                for s in cutlass.range_constexpr(split):
                    for v in cutlass.range_constexpr(width // 4):
                        offset = ((s * quads + width // 4 * g + v) * share + rr) * 16
                        words = _ld_shared_v4(mb + offset)
                        for j in cutlass.range_constexpr(4):
                            part = bits_float(words[j])
                            if cutlass.const_expr(s == 0):
                                totals[4 * v + j] = part
                            else:
                                totals[4 * v + j] = totals[4 * v + j] + part
                row = row0 + split_rank * share
                tile = data + self.tile_offset
                if cutlass.const_expr(self.lamx):  # F-L: every output store after the predecessor's completion (as served)
                    cute.arch.griddepcontrol_wait()
                self.finish(epi_tid, totals, tile, finish_out, row, col0, l_idx)
            else:  # ``any_width``: (row, quad) items with 16-byte loads, staged straight into the subclass's tile; ``finish`` gets None
                tile = data + self.tile_offset
                items: cutlass.Constexpr = share * quads
                for rnd in cutlass.range_constexpr(-(-items // 128)):
                    item = epi_tid + 128 * rnd
                    if cutlass.const_expr((rnd + 1) * 128 <= items):
                        self.owner_quad(item, mb, tile)
                    else:
                        if item < items:
                            self.owner_quad(item, mb, tile)
                row = row0 + split_rank * share
                if cutlass.const_expr(self.lamx):  # F-L: every output store after the predecessor's completion (as served)
                    cute.arch.griddepcontrol_wait()
                self.finish(epi_tid, None, tile, finish_out, row, col0, l_idx)

    @cute.experimental.jit
    def owner_quad(self, item: cutlass.Int32, mb: Int32, tile: Int32):
        """``any_width`` owner sum of one (row, quad) item of this owner's [share rows] x [N / 4 quads]: the split slots' 16-byte
        partials added in rank order (the same FP32 sums as the per-thread form), then ``stage_quad`` into the transposed tile.
        A warp's items are 16 rows of 2 quads: 512 contiguous mailbox bytes per slot."""
        split, share, quads = self.split_k, self.share, self.n // 4
        rr, q = item % share, item // share
        totals = cute.make_rmem_tensor(4, Float32)
        for s in cutlass.range_constexpr(split):
            words = _ld_shared_v4(mb + ((s * quads + q) * share + rr) * 16)
            for j in cutlass.range_constexpr(4):
                part = bits_float(words[j])
                if cutlass.const_expr(s == 0):
                    totals[j] = part
                else:
                    totals[j] = totals[j] + part
        self.stage_quad(tile, q, rr, totals)

    def stage_quad(self, tile, q, rr, totals):
        """``any_width`` subclasses: stage one owner (row ``rr``, tokens 4 q .. 4 q + 3) quad of FP32 sums into ``tile``."""
        raise NotImplementedError("any_width needs a subclass stage_quad")

    def finish_inputs(self, epi_tid, out, row0, col0):
        """Epilogue inputs an owner of rows [row0, row0 + share) loads before the
        reduce-scatter, passed to ``finish`` as the last item of ``out``; None:
        nothing."""
        return None

    @cute.experimental.jit
    def finish(
        self, epi_tid: cutlass.Int32, totals: cute.Tensor, tile: Int32, out,
        row0: cutlass.Int32, col0: cutlass.Int32, l_idx: cutlass.Int32,
    ):  # fmt: skip
        """An owner's rows [row0, row0 + SHARE) of tokens col0 ..: thread (row rr,
        tokens width g .. width g + width - 1, width = N / 4) holds ``totals``; rounded
        to BF16, transposed through the shared-memory ``tile`` [N tokens][SHARE] and
        stored as 16-byte rows of C (``out`` = (C,)). Default owners only (share =
        SHARE)."""
        mC = out[0]
        rr, g = epi_tid % SHARE, epi_tid // SHARE
        width: cutlass.Constexpr = self.n * SHARE // 128  # tokens per thread
        for t in cutlass.range_constexpr(width):
            _st_shared_bf16(tile + ((width * g + t) * SHARE + rr) * 2, totals[t])
        named_barrier(3, 128)
        rows, columns = mC.layout.shape[0], mC.layout.shape[1]
        if cutlass.const_expr(self.n == 64):
            token, half = epi_tid // 2, epi_tid % 2
            row = row0 + 16 * half
            if (row < rows) & (col0 + token < columns):
                for c in cutlass.range_constexpr(2):
                    source = tile + (token * SHARE + 16 * half + 8 * c) * 2
                    words = _ld_shared_v4(source)
                    store_words(at(mC, row + 8 * c, col0 + token, l_idx), words)
        else:  # N128: a token per thread, its SHARE rows in 8-row stores
            token = epi_tid
            if col0 + token < columns:
                for c in cutlass.range_constexpr(SHARE // 8):
                    if row0 + 8 * c < rows:  # a last tile's share may hold fewer rows
                        words = _ld_shared_v4(tile + (token * SHARE + 8 * c) * 2)
                        store_words(at(mC, row0 + 8 * c, col0 + token, l_idx), words)
