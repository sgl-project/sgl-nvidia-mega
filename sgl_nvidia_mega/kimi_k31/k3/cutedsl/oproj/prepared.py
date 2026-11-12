# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Options described as "ported" follow NVIDIA's Rubin (sm_107) specialized-decode kernels (the rubin package).
"""Output projection fused with its TP all-reduce and post-attention AttnRes: rank
partials go into Lamport slots of a multicast [TP, 64, 7168] mailbox (parity alternates
by layer) and the consumer sums and re-arms them. TP16 B<=8 fuses producer and consumer
(MLA layers below B5 keep them split); above B8 the producer runs 16-token groups.
At ``twoshot_batches``, given ``TwoShot`` buffers, the all-reduce is ``common.twoshot``
(chunks pushed to their column owner, owner sums + multicasts, all Lamport-polled), so
each GPU receives ~2 x 14 KB per token instead of TP x 14 KB."""

import os
import sys
from functools import cache
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed._symmetric_memory as symm_mem
from cutlass.cute.runtime import make_fake_stream

from ..common import k19
from ..common.host import _checked, dynamic_fake, dynamic_view, fake, prepare_shared_carveout, resolve_kernel, stream, to_cute
from ..common.policy import batch_table
from .consumer import SINGLE_ROUND, OprojConsumer, OprojFused
from .consumer_cta import OprojConsumerCta
from .producer import OprojProducer, OprojUmma


CAP = 64  # default mailbox / two-shot rows (every batch of a boot up to B64)


def capacity(batches):
    """Mailbox and two-shot rows of a boot: CAP, or its largest batch when that is more
    (verify at Breq 16 = 128 rows). Boots up to 64 rows keep the 64-row buffers."""
    if not batches:
        raise ValueError("o_proj capacity needs the boot's batch sizes (got none)")
    return max(CAP, *batches)


class OprojMailbox:
    """Sentinel-armed multicast mailbox; defaults to [parity, TP, CAP, 7168] BF16."""

    def __init__(self, group, device, tp, rank, shape=None):
        self.tp, self.rank = tp, rank
        shape = (2, tp, CAP, 7168) if shape is None else shape
        self.buffer = symm_mem.empty(shape, dtype=torch.bfloat16, device=device)
        self.handle = handle = symm_mem.rendezvous(self.buffer, group)
        self.mc = int(handle.multicast_ptr or 0)
        if not self.mc:  # slot() offsets this address
            raise RuntimeError(
                "Kimi K3 specialized o-projection needs multicast memory"
            )
        self.buffer.view(torch.int32).fill_(-0x80000000)
        torch.accelerator.synchronize(device)
        torch.distributed.barrier(group=group)
        # HH-M (K31MK_OPROJ_HHOME_M, default unset: no companion and nothing imported): the o-proj mailbox
        # [2, TP16, cap, 7168] gets a homed one-shot companion for its split batches with 9-16 rows (``mfx_hhome.HomedMailbox``;
        # ``PreparedOproj`` uses it at those batches). Collective (every rank builds the same mailboxes in the same order).
        self.hhm = None
        if os.environ.get("K31MK_OPROJ_HHOME_M", "") and len(shape) == 4 and shape[0] == 2 and shape[3] == 7168:
            from . import mfx_hhome as HH

            form = HH.mailbox_rule()  # "d0" (ticket) / "s0" (static permutation)
            if form:
                if tp != 16:
                    raise ValueError(f"K31MK_OPROJ_HHOME_M: TP16 only (tp {tp})")
                self.hhm = HH.HomedMailbox(group, device, tp, rank, form=form)
                print(f"K31MK_OPROJ_HHOME_M engaged form={form} mailbox rows={self.hhm.cap} segments_ld={self.hhm.segments_ld()} "
                      f"pid={os.getpid()}", file=sys.stderr, flush=True)

    def slot(self, parity):
        return self.buffer[parity], self.mc + parity * self.buffer[0].numel() * 2


# Two-shot all-reduce batches per TP (2-/4-node GB300, 32 MLA layers, us/layer, max over
# ranks, multicast vs two-shot: TP8 B16 8.85/11.40, B32 10.63/11.32, B64 18.06/13.18;
# TP16 B16-B64 measured likewise). TP16 up to 128 rows: on VR200 at
# world 16 (7168 BF16 columns) the two-shot class beats one-shot at 128 rows
# (measured per all-reduce). TP8 above 64 rows is unmeasured and stays one-shot.
TWOSHOT = {8: "33-64:1", 16: "17-128:1"}


def twoshot_batches(tp):
    """``TWOSHOT[tp]``, then at TP16 ``K31MK_OPROJ_TWOSHOT`` (default unset: no change; a
    batch table such as "32:0", rows 9-128, values 0 / 1: the VR200 re-tune of this
    GB300 table). Read when ``PreparedOproj`` is built."""
    extra = os.environ.get("K31MK_OPROJ_TWOSHOT", "") if tp == 16 else ""
    bad = {b: v for b, v in batch_table(extra).items() if not 9 <= b <= 128 or v not in (0, 1)}
    if bad:
        raise ValueError(f"K31MK_OPROJ_TWOSHOT: rows 9-128 with value 0 or 1 only, got {bad}")
    table = batch_table(f"{TWOSHOT.get(tp, '')},{extra}")
    return {batch for batch, on in table.items() if on}


# CC (K31MK_OPROJ_CONS_CARVEOUT; default unset = no change): the ported CONSUMER_CARVEOUT {16: "32:100,
# 64:100}: the two-shot consumer of these TP16 batches gets a preferred shared-memory carve-out (percent) as a driver
# function attribute. The reason: the consumer CTAs need ~22 KB, and at the driver's default carve-out the MoE
# front's ~170 KB MoeFrontTmem64 CTAs launched behind it (K31MK_MOE_FRONT_TMEM) cannot share an SM with them; at 100
# the front preloads beside the consumer (reported faster per decode step in the source's 16-rank serving). A launch attribute only: the
# kernel and its outputs are unchanged. A batch table, rows 9-128, values 0-100; SM107 only; applied where the batch runs
# the two-shot consumer (as the source's form; other batches unchanged). The compiled consumer is shared by every layer at its batch (one
# function), so the attribute holds for all of them. Read when ``PreparedOproj`` is built.
def consumer_carveout(tp, device):
    """{batch: carve-out percent} of K31MK_OPROJ_CONS_CARVEOUT at ``tp`` on ``device``."""
    spec = os.environ.get("K31MK_OPROJ_CONS_CARVEOUT", "")
    table = batch_table(spec)
    bad = {b: v for b, v in table.items() if not 9 <= b <= 128 or type(v) is not int or not 0 <= v <= 100}
    if bad:
        raise ValueError(f"K31MK_OPROJ_CONS_CARVEOUT: rows 9-128 with value 0-100 only, got {bad}")
    if table and (tp != 16 or torch.cuda.get_device_capability(device) != (10, 7)):
        raise ValueError(f"K31MK_OPROJ_CONS_CARVEOUT={spec!r}: TP16 on SM107 (VR200) only")
    return table


# Fused o-projection options of the source's Rubin decode (
# ``oproj/prepared.py`` SHARED_X / DRAIN; a 1-GPU KdaSplit -> o-proj -> MoE front proxy,
# B8: x_shared faster per layer, drain an earlier PDL release), both bitwise
# identical: batch tables such as "8:1" of the TP16 B<=8 fused batches that take them
# (``K31MK_OPROJ_X_SHARED``, ``K31MK_OPROJ_DRAIN``; default none, read once per
# ``PreparedOproj``).
def fused_options(tp):
    """(x_shared batches, drain batches) of the fused o-projection at ``tp``."""
    names = ("K31MK_OPROJ_X_SHARED", "K31MK_OPROJ_DRAIN")
    tables = (batch_table(os.environ.get(n, "") if tp == 16 else "") for n in names)
    return tuple(frozenset(b for b, on in t.items() if on and b <= 8) for t in tables)


# Package (b) (the source's ``oproj/prepared.py`` Rubin options, ported onto our
# producer / consumer; every one bitwise identical to the path it replaces, default off). Batch tables, TP16 on
# SM107 (VR200) only, read when ``PreparedOproj`` is built:
# - K31MK_OPROJ_BATCHED (the ported TWOSHOT_BATCHED {16: "17-64:1"}; rows 9-128, 0 / 1): two-shot owners poll their TP
#   receive slots with every load in flight (``common.twoshot.poll_slots``);
# - K31MK_OPROJ_PRODUCER_REDUCE (PRODUCER_REDUCE {16: "17-64:1"}; rows 9-128, 0 / 1; tcgen05 producer only): the
#   two-shot producer's 112 resident CTAs owner-reduce and multicast the gather (with the prefix under
#   K31MK_OPROJ_SINGLE_ROUND, as the consumer's owner adds it); the consumer only polls the gather. The source reports, with 16 ranks:
#   faster per call at B64 and B32;
# - K31MK_OPROJ_RESIDENT (RESIDENT_RING {16: "17-64:1"}; rows 9-64, 0 / 1; tcgen05 producer): one ring stage per
#   K block, all weights before the PDL wait (reported faster at B64 with PRODUCER_REDUCE; B32 unchanged). The source's
#   X_MULTICAST / ONESHOT_RING x multicast needs a fixed x buffer for its TMA map: not ported (no x buffer here);
# - K31MK_OPROJ_PROD_CARVEOUT (PRODUCER_CARVEOUT {16: "9-40:80"}; rows 9-128, 0-100): the split producer's
#   preferred shared-memory carve-out (a launch attribute; at M32 behind the q4 fold verify the source reports the producer faster,
#   one CTA per SM instead of two);
# - K31MK_OPROJ_VOTE (OPROJ_VOTE {16: "1-16:1"}; rows 1-128, 0 / 1): the one-shot consumer (fused B<=8 and the
#   split one-shot batches) reloads every rank slot each pass and leaves the poll on a warp vote (reported faster for the M16 one-shot
#   pair). Not for two-shot batches (their consumer polls one gather slot).
OPR_KNOBS = {
    "K31MK_OPROJ_BATCHED": ((9, 128), (0, 1)),
    "K31MK_OPROJ_PRODUCER_REDUCE": ((9, 128), (0, 1)),
    "K31MK_OPROJ_RESIDENT": ((9, 64), (0, 1)),
    "K31MK_OPROJ_PROD_CARVEOUT": ((9, 128), tuple(range(101))),
    "K31MK_OPROJ_VOTE": ((1, 128), (0, 1)),
    "K31MK_OPROJ_CONS1": ((17, 128), (0, 1)),
}
# CONS1 (K31MK_OPROJ_CONS1; a batch table, rows 17-128, 0 / 1; default unset = no change): at a batch whose
# o-projection is two-shot AND producer-reduced (K31MK_OPROJ_PRODUCER_REDUCE), the consumer runs as ONE 896-thread CTA per
# token (``consumer_cta.OprojConsumerCta``) instead of a cluster of 7 CTAs: its pre / post / norm exchanges go through
# shared memory and CTA barriers instead of DSMEM pushes + mbarrier waits; same values, same order (bitwise identical). It
# raises at a batch that is not two-shot producer-reduced, and with an x_moe publish (front batches; untested there).
# Rows 65-128 (verify rounds; the two-shot consumer there is the same groups-1 cluster form): one 896-thread CTA per
# token holds a whole SM (1024 threads, 64 K registers), so B tokens take B SMs beside the producer's resident CTAs.
# Engagement: one stderr line ``K31MK_OPROJ_CONS1 engaged rows=<rows> pid=<pid>`` per process and row set.
_CONS1_ENGAGED: set = set()


def opr_options(tp, device):
    """{knob: {batch: value}} of the package (b) knobs at ``tp`` on ``device`` (empty when unset)."""
    tables = {}
    for name, ((lo, hi), values) in OPR_KNOBS.items():
        spec = os.environ.get(name, "")
        table = batch_table(spec)
        bad = {b: v for b, v in table.items() if not lo <= b <= hi or v not in values}
        if bad:
            raise ValueError(f"{name}: rows {lo}-{hi} with values {values[0]}..{values[-1]} only, got {bad}")
        if table and (tp != 16 or torch.cuda.get_device_capability(device) != (10, 7)):
            raise ValueError(f"{name}={spec!r}: TP16 on SM107 (VR200) only")
        tables[name] = {b: v for b, v in table.items() if v} if values == (0, 1) else table
    return tables


def engine(batch, tp):
    """Producer engine above B8 (GB300): grouped mma.sync for TP8 up to B16, else
    tcgen05. At TP16 ``K31MK_OPROJ_ENGINE`` (default unset: no change; a batch table
    such as "64:mma", rows 9-128, "mma" or "umma") overrides it: the VR200 re-tune."""
    default = "mma" if tp == 8 and batch <= 16 else "umma"
    if tp != 16 or batch <= 8:
        return default
    table = batch_table(os.environ.get("K31MK_OPROJ_ENGINE", ""))
    bad = {b: v for b, v in table.items() if not 9 <= b <= 128 or v not in ("mma", "umma")}
    if bad:
        raise ValueError(f"K31MK_OPROJ_ENGINE: rows 9-128 with mma or umma only, got {bad}")
    return table.get(batch, default)


def staged_blocks(batch, tp, mla):
    """K blocks (of 12 at TP8) the B<=8 producer stages in SMEM; the rest stay in
    registers. The partial rows are bit-identical for every count (same MMA and K-part
    order); only speed changes. GB300: 4 at B1/MLA, else 8. SM107 (VR200, TP4 in-tray,
    producer per call over cold weights): 2 is the fastest at B2-8 (measured ahead of 8),
    and 8 at B8 is much slower (the 64-register build); at B1, 2 and
    4 tie, so B1 keeps GB300's 4. ``K31MK_OPROJ_STAGED`` overrides the
    count (A/B only)."""
    if tp == 16:
        return 0
    override = os.environ.get("K31MK_OPROJ_STAGED")
    if override:
        return int(override)
    if torch.cuda.get_device_capability() == (10, 7):
        return 4 if batch == 1 else 2
    return 4 if mla or batch == 1 else 8


def producer(k, batch, tp, mla, twoshot=False, kind=None, opr=(False, False, False, 0)):
    """B<=8: one mma.sync pass staging ``staged_blocks`` of the K blocks in SMEM. Above
    B8 (``kind`` overrides ``engine``): the tcgen05 engine, or 2-tile warps over 8 K
    parts in 16-token groups, TP8 staging 2 of 6 K blocks per warp. ``opr``: the tcgen05
    producer's package (b) options (reduce, residual, resident, deadline_ns).
    K-19 (``K31MK_K19_FENCE`` oproj_producer / oproj_umma, default off): the producer
    fences before its trigger; set only when on, so default objects stay as they were."""
    kernel = _producer(k, batch, tp, mla, twoshot, kind, opr)
    if k19.fence("oproj_umma" if isinstance(kernel, OprojUmma) else "oproj_producer"):
        kernel.k19_fence = True
    if isinstance(kernel, OprojUmma) and kernel.reduce and batch in prw_batches(tp):  # PRW (default none)
        kernel.prw = True
    return kernel


# PRW (K31MK_OPROJ_PRW; a batch table, rows 65-128, 0 / 1; default unset = no change): at a producer-reduced
# two-shot batch above 64 rows, ``OprojUmma.reduce_owned`` runs its owner rounds on loader warps 0 .. rounds - 1 at once
# instead of serially on warp 0 (``producer.OprojUmma.prw``; bitwise identical). TP16 only; PreparedOproj raises at a listed
# batch that is not producer-reduced. Engagement: one stderr line ``K31MK_OPROJ_PRW engaged rows=<rows> pid=<pid>``.
_PRW_ENGAGED: set = set()


def prw_batches(tp):
    """Batches of ``K31MK_OPROJ_PRW`` at ``tp`` (rows 65-128, value 0 / 1; TP16 only)."""
    spec = os.environ.get("K31MK_OPROJ_PRW", "")
    table = batch_table(spec)
    bad = {b: v for b, v in table.items() if not 65 <= b <= 128 or v not in (0, 1)}
    if bad:
        raise ValueError(f"K31MK_OPROJ_PRW: rows 65-128 with value 0 or 1 only, got {bad}")
    if table and tp != 16:
        raise ValueError(f"K31MK_OPROJ_PRW={spec!r}: TP16 only")
    return {b for b, v in table.items() if v}


# RING (K31MK_OPROJ_RING; a batch table, rows 65-128, value = cp.async ring stages 1-12; default unset = no
# change): the tcgen05 producer's ring holds ``value`` K blocks (a stage = W [64, 64] + x [rows, 64] BF16) instead of what the
# 184 KB budget gives (96 rows: 9 of 12, 128: 7), so fewer K blocks wait after the PDL wait for an MMA to free their stage
# (measured: those waits are significant at 96 / 128 rows). The same MMAs in the same order: bitwise
# identical. Above its budget the producer CTA leaves no room for a co-resident consumer CTA on its SM (timing decides). TP16 only;
# raises when the ring does not fit the per-CTA SMEM limit or the batch is RESIDENT. Read in ``_producer`` (not in the
# ``compile_oproj`` cache key, like PRW: one env per process).
_RING_SMEM_LIMIT = 232448 - 2048  # sharedMemPerBlockOptin (CC 10.x, CUDA C++ Programming Guide) minus the barriers, TMEM slot and 1 KB alignment slack
_RING_ENGAGED: set = set()


def ring_batches(tp):
    """{batch: ring stages} of ``K31MK_OPROJ_RING`` at ``tp`` (rows 65-128, value 0-12; TP16 only)."""
    spec = os.environ.get("K31MK_OPROJ_RING", "")
    table = batch_table(spec)
    bad = {b: v for b, v in table.items() if not 65 <= b <= 128 or not 0 <= v <= 12}
    if bad:
        raise ValueError(f"K31MK_OPROJ_RING: rows 65-128 with value 0-12 only, got {bad}")
    if table and tp != 16:
        raise ValueError(f"K31MK_OPROJ_RING={spec!r}: TP16 only")
    return {b: v for b, v in table.items() if v}


def _ring_budget(batch, tp, resident):
    """RING: the budget giving ``K31MK_OPROJ_RING`` stages at ``batch``, or None (served budget)."""
    stages = ring_batches(tp).get(batch)
    if not stages:
        return None
    if resident:
        raise ValueError(f"K31MK_OPROJ_RING at batch {batch}: RESIDENT batch")
    stage = 8192 + (batch + 7) // 8 * 8 * 128
    if stages * stage > _RING_SMEM_LIMIT:
        raise ValueError(f"K31MK_OPROJ_RING at batch {batch}: {stages} x {stage} B exceeds {_RING_SMEM_LIMIT} B")
    return stages * stage


# LEANP (K31MK_OPROJ_LEANP; a batch table, rows 17-128, 0 / 1; default unset = no change): the groups-1 (two-shot,
# OprojConsumer, not CONS1) consumer of a listed batch allocates no ``parts`` buffer (consumer.py LEANP; 4 KB less per CTA, so 8 CTAs
# fit an SM at nb 8 instead of 7). Bitwise identical. TP16 only; PreparedOproj raises at a listed batch whose consumer is not that
# kernel. Engagement: one stderr line ``K31MK_OPROJ_LEANP engaged rows=<rows> pid=<pid>``.
_LEANP_ENGAGED: set = set()


def leanp_batches(tp):
    """Batches of ``K31MK_OPROJ_LEANP`` at ``tp`` (rows 17-128, value 0 / 1; TP16 only)."""
    spec = os.environ.get("K31MK_OPROJ_LEANP", "")
    table = batch_table(spec)
    bad = {b: v for b, v in table.items() if not 17 <= b <= 128 or v not in (0, 1)}
    if bad:
        raise ValueError(f"K31MK_OPROJ_LEANP: rows 17-128 with value 0 or 1 only, got {bad}")
    if table and tp != 16:
        raise ValueError(f"K31MK_OPROJ_LEANP={spec!r}: TP16 only")
    return {b for b, v in table.items() if v}


# Carve-out engagement (for COLOC = LEANP + K31MK_OPROJ_CONS_CARVEOUT / _PROD_CARVEOUT): one stderr line per process and
# setting, ``K31MK_OPROJ_CARVEOUT engaged prod=[(batch, percent), ...] cons=[...] pid=<pid>``, each percent read back from the driver
# (CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT of the prepared kernel) after it was set. No line when neither knob applies.
_CARVE_ENGAGED: set = set()


def carveout_now(prepared, device_index):
    """The preferred shared-memory carve-out the driver holds for a prepared kernel (-1 = none set)."""
    import cuda.bindings.driver as cuda

    function = resolve_kernel(getattr(prepared, "compiled", prepared), device_index)[1]
    attribute = cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
    return _checked(cuda.cuFuncGetAttribute(attribute, function))[0]


def _producer(k, batch, tp, mla, twoshot, kind, opr):
    if batch <= 8:
        staged = staged_blocks(batch, tp, mla)
        return OprojProducer(k, staged_blocks=staged, twoshot=twoshot)
    if (kind or engine(batch, tp)) == "umma":
        ring = _ring_budget(batch, tp, bool(opr[2]))  # RING (default none)
        ring = {} if ring is None else dict(budget=ring)
        if any(opr[:3]):
            reduce, residual, resident, deadline_ns = opr
            options = dict(reduce=reduce, residual=residual, resident=resident, deadline_ns=deadline_ns)
            return OprojUmma(k, twoshot=twoshot, **options, **ring)
        return OprojUmma(k, twoshot=twoshot, **ring)
    if any(opr[:3]):
        raise ValueError("package (b) producer options need the tcgen05 (umma) engine")
    grouped = dict(rows=2, group_tiles=2, twoshot=twoshot)
    return OprojProducer(k, staged_blocks=0 if tp == 16 else 2, **grouped)


@cache
def compile_oproj(
    kind,
    k,
    batch,
    tp,
    nb,
    has_delta,
    mla,
    front=False,
    cap=64,
    twoshot=False,
    deadline_ns=0,
    x_shared=False,
    drain=False,
    opr=(),
    cdr=(0, True),
    nmerge=False,
    hhome=-1,
    hhdyn=False,
    hhome_r=0,
    hhome_m=False,
    lamx=False,
    leanp=False,
):
    """Producer, consumer or fused kernel over a [TP, cap, 7168] parity view;
    ``front``: the consumer also publishes x_moe [1, 4, 7168] and the epoch hint.
    ``twoshot``: the two-shot pair (arguments as ``TwoShot.views``). ``x_shared`` /
    ``drain``: the fused kernel's options (``fused_options``). ``opr``: package (b)
    options (``opr_options``) of this kind: producer (reduce, residual, resident), consumer
    (batched, reduced, vote), fused (vote,); empty = none (the unchanged kernels)."""
    box = fake((1 if twoshot else tp, cap, 7168), (cap * 7168, 7168, 1))
    x, w = fake((batch, k), (k, 1)), fake((7168, k), (k, 1))
    mc, slot, eps = cutlass.Int64(0), cutlass.Int32(0), cutlass.Float32(1e-6)
    recv, peers = fake((tp, cap, 7168 // tp)), fake((tp,), None, torch.int64)
    if kind.startswith("producer"):  # producer_mma / producer_umma: engine A/B
        if opr:  # package (b): (reduce, residual, resident)
            kernel = producer(k, batch, tp, mla, twoshot, kind[9:] or None, (*opr, deadline_ns))
        else:
            kernel = producer(k, batch, tp, mla, twoshot, kind[9:] or None)
        tail = (recv, peers, slot) if twoshot else (box, mc, slot)
        if opr and opr[0]:  # reduce: the gather and its multicast address (+ the prefix)
            tail += (box, mc) + ((dynamic_fake(2),) if opr[1] else ())
            if hhome >= 0:  # HH-G: the owner multicasts into the homed gather layout
                kernel.hhome = hhome
        elif hhome >= 0:
            raise ValueError("K31MK_OPROJ_HHOME_G: producer-reduced two-shot producers only")
        if hhdyn:  # HH-G dynamic: the consumer takes its ticket before its PDL wait (the K-19 edge); the producer's
            kernel.k19_fence = True  # fence before its trigger orders that ticket after the last same-parity reset
        if hhome_r:  # HH-R: homed receive pushes (1), + die-matched owner reduce (2)
            if not twoshot or not isinstance(kernel, OprojUmma) or (hhome_r == 2 and not (opr and opr[0])) or kernel.prw:
                raise ValueError("K31MK_OPROJ_HHOME_R: tcgen05 two-shot producers only (owner reduce: producer-reduced, not PRW)")
            kernel.hhome_r = hhome_r
            if hhome_r == 2:
                tail += (fake((1,), None, torch.int64), fake((256,), None, torch.uint8))
        if hhome_m:  # HH-M: multicast into the homed one-shot mailbox layout
            if twoshot or not isinstance(kernel, OprojUmma):
                raise ValueError("K31MK_OPROJ_HHOME_M: tcgen05 one-shot producers only")
            kernel.hhome_m = True
        return cute.compile(kernel, x, w, *tail, make_fake_stream())
    row, weight = dynamic_fake(2), fake((7168,), (1,))
    xmoe = fake((1, 4, 7168), (4 * 7168, 7168, 1)) if front else None
    hints = fake((224,), (1,), torch.int32) if front else None
    if lamx:  # F-L (K31MK_MOE_FRONT_LAMX): ``hints`` carries this layer's [32, 7168] BF16 Lamport x box view
        if front or kind != "consumer":
            raise ValueError("K31MK_MOE_FRONT_LAMX: split OprojConsumer without an x_moe publish only")
        hints = fake((32, 7168), (7168, 1))
    rows = (row, dynamic_fake(3), weight, weight, weight, row, xmoe, hints, eps, eps)
    if kind == "consumer_cta":  # CONS1 (K31MK_OPROJ_CONS1): the producer-reduced two-shot consumer on one CTA per token
        if nmerge:
            raise ValueError("K31MK_OPROJ_NMERGE is not ported to CONS1 (consumer_cta)")
        if not twoshot or front:
            raise ValueError("consumer_cta: two-shot producer-reduced consumer without an x_moe publish only")
        kernel = OprojConsumerCta(nb, has_delta=has_delta, deadline_ns=deadline_ns, defer=cdr[0], keep_own=cdr[1])  # CDR as consumer
        return cute.compile(kernel, box, *rows, recv, mc, slot, make_fake_stream())
    if kind == "consumer":  # B64: one 128-thread group per CTA polls every rank
        options = dict(has_delta=has_delta)
        if opr:  # package (b): (batched, reduced, vote)
            options.update(batched=opr[0], reduced=opr[1], vote=opr[2])
        if cdr[0]:  # CDR (K31MK_OPROJ_DEFER_REARM): (defer, keep_own)
            if front:
                raise ValueError("CDR: consumers that publish x_moe keep their own re-arm")
            options.update(defer=cdr[0], keep_own=cdr[1])
        if nmerge:  # NMERGE (K31MK_OPROJ_NMERGE): numerics-changing, groups-1 consumers only; the value is the form (1 / 2)
            options.update(nmerge=int(nmerge))
        groups = 4 if batch <= 32 and not twoshot else 1
        kernel = OprojConsumer(
            nb,
            groups=groups,
            twoshot=twoshot,
            deadline_ns=deadline_ns,
            **options,
        )
        if hhome >= 0:  # HH-G: poll / reduce / re-arm the homed gather layout
            if not twoshot or front:
                raise ValueError("K31MK_OPROJ_HHOME_G: two-shot consumers without an x_moe publish only")
            kernel.hhome = hhome
        extra = (recv, mc, slot) if twoshot else ()
        if hhome_r:  # HH-R: a consumer-reduced two-shot consumer polls the homed receive layout
            if not twoshot or (opr and opr[1]):
                raise ValueError("K31MK_OPROJ_HHOME_R: consumer side only for consumer-reduced two-shot batches")
            kernel.hhome_r = 1
        if hhdyn:  # HH-G dynamic: + this parity's ticket word and the SM -> locality domain table
            if hhome < 0:
                raise ValueError("K31MK_OPROJ_HHOME_G dynamic needs hhome")
            kernel.hhdyn = True
            extra += (fake((1,), None, torch.int64), fake((256,), None, torch.uint8))
        if hhome_m:  # HH-M: poll / re-arm the homed one-shot mailbox; d0: + this parity's ticket word and the SM -> locality domain table
            if twoshot or front or groups != 4 or hhdyn or hhome >= 0 or hhome_m not in ("d0", "s0"):
                raise ValueError("K31MK_OPROJ_HHOME_M: groups-4 one-shot consumers without an x_moe publish only (form d0 / s0)")
            kernel.hhm, kernel.hhm_b = hhome_m, batch
            if hhome_m == "d0":
                extra += (fake((1,), None, torch.int64), fake((256,), None, torch.uint8))
        if lamx:  # F-L: publish also stores x as Lamport words into ``hints`` (the front's x box parity view)
            kernel.lamx = True
        if leanp:  # LEANP: no parts buffer in the groups-1 consumer
            if groups != 1:
                raise ValueError("K31MK_OPROJ_LEANP: groups-1 (two-shot) consumers only")
            kernel.leanp = True
        return cute.compile(kernel, box, *rows, *extra, make_fake_stream())
    options = dict(x_shared=x_shared, drain=drain)
    if opr:  # package (b): (vote,)
        options.update(vote=opr[0])
    kernel = OprojFused(nb, has_delta=has_delta, deadline_ns=deadline_ns, **options)
    return cute.compile(kernel, x, w, box, mc, slot, *rows, make_fake_stream())


_cdr_engaged: set = set()
_HHM_ENGAGED: set = set()  # HH-M (K31MK_OPROJ_HHOME_M): one stderr line per distinct batch set
_HHG_ROWS_ENGAGED: set = set()  # HH-G row gate (K31MK_OPROJ_HHOME_G_ROWS): one stderr line per distinct homed / plain split
_HHR_ROWS_ENGAGED: set = set()  # HH-R row gate (K31MK_OPROJ_HHOME_R_ROWS): one stderr line per distinct homed / plain split


# NMERGE (K31MK_OPROJ_NMERGE; a batch table, rows 17-128 (form 2: 9-128), 0 / 1 / 2; default unset = no change; NUMERICS-CHANGING,
# not bitwise): at a two-shot split batch (groups-1 consumer) the consumer computes the output RMS norm by the quadratic identity
# from its post exchange (consumer.py NMERGE) and drops its third cluster exchange. TP16 on SM107 only; PreparedOproj raises at
# a listed batch that is fused, CONS1 or publishes x_moe, or one-shot under form 1 (untested there). Engagement: one stderr line
# ``K31MK_OPROJ_NMERGE engaged rows=<rows> pid=<pid>``. Value 2 = form 2 (NM4: the four post values in one reduce_scatter;
# output bitwise equal to form 1's).
_NMERGE_ENGAGED: set = set()


def nmerge_batches(tp, device):
    """{batch: form} of ``K31MK_OPROJ_NMERGE`` at ``tp`` on ``device`` (rows 17-128 for form 1, 9-128 for form 2; TP16 on SM107 only)."""
    spec = os.environ.get("K31MK_OPROJ_NMERGE", "")
    table = batch_table(spec)
    bad = {b: v for b, v in table.items() if v not in (0, 1, 2) or not (17 if v == 1 else 9) <= b <= 128}
    if bad:
        raise ValueError(f"K31MK_OPROJ_NMERGE: value 0 / 2 at rows 9-128 or 1 at rows 17-128 only, got {bad}")
    if table and (tp != 16 or torch.cuda.get_device_capability(device) != (10, 7)):
        raise ValueError(f"K31MK_OPROJ_NMERGE={spec!r}: TP16 on SM107 (VR200) only")
    return {b: v for b, v in table.items() if v}


def oproj_defer_batches(tp):
    """CDR (K31MK_OPROJ_DEFER_REARM, default unset: none): batches (rows 9-128, value 0 / 1) whose split
    consumer re-arms the previous layer's slots at its entry instead of its own after its poll (consumer.py CDR). Read when
    ``PreparedOproj`` is built with ``cdr_last`` given (the provider passes it for every decoder layer)."""
    table = batch_table(os.environ.get("K31MK_OPROJ_DEFER_REARM", ""))
    bad = {b: v for b, v in table.items() if not 9 <= b <= 128 or v not in (0, 1)}
    if bad:
        raise ValueError(f"K31MK_OPROJ_DEFER_REARM: rows 9-128 with value 0 or 1 only, got {bad}")
    return {b for b, v in table.items() if v}


class PreparedOproj:
    """One decoder layer's o-projection + all-reduce + post-attention AttnRes."""

    def __init__(
        self,
        decoder,
        mailbox,
        batches,
        mla,
        front=None,
        twoshot=None,
        *,
        deadline_ns,
        cdr_last=None,
        lamx=None,
    ):
        # ``cdr_last`` (CDR, K31MK_OPROJ_DEFER_REARM): None = CDR off for this object; else whether this is the last o-proj
        # layer of the forward (it also re-arms its own slots). The provider passes it for every decoder layer.
        weight = decoder.self_attn.o_proj.weight
        norm, out_norm = decoder.mlp_res_norm, decoder.post_attention_layernorm
        weights = (norm.weight, decoder.mlp_res_proj.weight.squeeze(0), out_norm.weight)
        self.views = tuple(to_cute(w.reshape(-1)) for w in weights)
        self.eps = tuple(cutlass.Float32(n.variance_epsilon) for n in (norm, out_norm))
        parity, rank = decoder.layer_idx % 2, cutlass.Int32(mailbox.rank)
        cap = mailbox.buffer.shape[2]  # rows of every parity view (capacity(batches) at build)
        if max(batches) > cap:
            raise ValueError(f"o-projection batch {max(batches)} exceeds the {cap}-row mailbox")
        if twoshot is not None and getattr(twoshot, "cap", twoshot.recv.shape[2]) != cap:
            raise ValueError(f"two-shot rows {twoshot.recv.shape[2]} != mailbox rows {cap}")
        local_box, mc = mailbox.slot(parity)
        self.device, self.box = weight.device, to_cute(local_box)
        self.head = (self.box, cutlass.Int64(mc), rank)
        self.twoshot: Any = None
        self.twoshot_plain: Any = None  # HH-G row gate (K31MK_OPROJ_HHOME_G_ROWS): the plain gather of two-shot batches outside it
        self.twoshot_rplain: Any = None  # HH-R row gate (K31MK_OPROJ_HHOME_R_ROWS): {plain gather?: views} with the plain receive slots
        if twoshot is not None:
            recv, peers, gather, gather_mc = twoshot.views(parity)
            recv = to_cute(recv)
            self.twoshot = (
                (recv, to_cute(peers), rank),
                to_cute(gather),
                (recv, cutlass.Int64(gather_mc), rank),
            )
            if getattr(twoshot, "hh_rows", None) is not None:
                _, _, pgather, pgather_mc = twoshot.views(parity, plain=True)
                self.twoshot_plain = (self.twoshot[0], to_cute(pgather), (recv, cutlass.Int64(pgather_mc), rank))
            if getattr(twoshot, "hr_rows", None) is not None:  # HH-R row gate: the plain receive slots, by gather choice (homed / plain)
                precv, ppeers, _, _ = twoshot.views(parity, plain_recv=True)
                precv = to_cute(precv)
                head_p = (precv, to_cute(ppeers), rank)
                self.twoshot_rplain = {False: (head_p, self.twoshot[1], (precv, self.twoshot[2][1], rank))}
                if self.twoshot_plain is not None:
                    self.twoshot_rplain[True] = (head_p, self.twoshot_plain[1], (precv, self.twoshot_plain[2][1], rank))
        self.wview, tp, k = to_cute(weight), mailbox.tp, weight.shape[1]
        # HH-M: the mailbox's homed one-shot companion (``OprojMailbox.hhm``, default None: no change); its split
        # one-shot batches (``hhm.batches``: d0 9..cap, s0 the static-permutation batches) use it (views of this parity; d0: + ticket, SM table).
        hhm = getattr(mailbox, "hhm", None)
        self.hhm = set()  # HH-M batches (tests read it)
        self.hhm_head, self.hhm_box, self.hhm_args = None, None, ()
        if hhm is not None:
            hbox, hmc = hhm.slot(parity)
            self.hhm_box = to_cute(hbox)
            self.hhm_head = (self.hhm_box, cutlass.Int64(hmc), rank)
            self.hhm_args = (to_cute(hhm.tickets[parity, :1]), to_cute(hhm.smtab)) if hhm.form == "d0" else ()
        # CDR: element offset from this layer's consumer box (mailbox[parity] one-shot, gather[parity] two-shot) to the other
        # parity of the same allocation (the previous layer's).
        sign = 1 - 2 * parity
        cdr_n = {False: sign * local_box.numel(), True: sign * (twoshot.gather[0].numel() if twoshot is not None else 0)}
        cdr_plain = sign * twoshot.pgather[0].numel() if self.twoshot_plain is not None else 0  # HH-G row gate: plain-gather batches
        cdr_on = oproj_defer_batches(tp) if cdr_last is not None else set()
        self.cdr: dict[int, dict] = {}  # batch -> {defer, keep_own} the consumer was compiled with (tests read it)
        nb = decoder.prev_valid_blocks + decoder.is_block_write_layer
        delta = not decoder.is_block_write_layer
        self.front = None if front is None else (*map(to_cute, front[:2]), front[2])
        # F-L (K31MK_MOE_FRONT_LAMX; default None = served code): (this layer's [32, 7168] BF16 parity view of the
        # MoE front's Lamport x box, its batches); those batches' consumer stores x there too (``OprojConsumer.lamx``).
        self.lamx = None if lamx is None else (to_cute(lamx[0]), frozenset(lamx[1]))
        self.codes, self.twoshots = {}, twoshot_batches(tp) if self.twoshot else set()
        x_shared, drain = fused_options(tp)
        carve = consumer_carveout(tp, self.device)  # K31MK_OPROJ_CONS_CARVEOUT (default none)
        opr = opr_options(tp, self.device)  # package (b) knobs (default none)
        self.reduced, self.residual = set(), delta and SINGLE_ROUND  # producer-reduced batches; prefix to it
        self.opr = {}  # batch -> {kind: package (b) options it was compiled with} (tests read it)
        self.cons1 = set()  # CONS1 (K31MK_OPROJ_CONS1) batches: the consumer is OprojConsumerCta (tests read it)
        nm_on = nmerge_batches(tp, self.device)  # NMERGE (default none)
        # HH-G: a ``mfx_hhome.HomedTwoShot`` buffer carries its rule (bit k, swapped); every two-shot
        # batch of this object then uses the homed gather layout (default: plain TwoShot, hh_k -1, no change).
        hh_k = twoshot.rule[0] if twoshot is not None and getattr(twoshot, "rule", None) else -1
        hh_dyn = hh_k >= 0 and bool(getattr(twoshot, "dynamic", False))
        # HH-G row gate (K31MK_OPROJ_HHOME_G_ROWS, default unset = every two-shot batch): two-shot batches outside it use the buffer's
        # plain symm_mem gather (``HomedTwoShot.pgather``) with the served producer / consumer code, i.e. HH-G off at those rows.
        hh_rows = getattr(twoshot, "hh_rows", None) if twoshot is not None else None
        self.hhome = set()  # HH-G batches (tests read it)
        self.hh_args = (to_cute(twoshot.tickets[parity, :1]), to_cute(twoshot.smtab)) if hh_dyn else ()
        # HH-R: the receive slots homed by token parity (HomedTwoShot.recv_dynamic)
        hh_r = twoshot is not None and bool(getattr(twoshot, "recv_dynamic", False))
        # HH-R row gate (K31MK_OPROJ_HHOME_R_ROWS, default unset = every two-shot batch): two-shot batches outside it use the buffer's
        # plain symm_mem receive slots (``HomedTwoShot.precv``) with the served producer, i.e. HH-R off at those rows (e.g. PRW rows).
        hr_rows = getattr(twoshot, "hr_rows", None) if twoshot is not None else None
        self.hhr = {}  # batch -> producer hhome_r (2 = die-matched owner reduce, 1 = homed pushes only) (tests read it)
        self.hhr_args = (to_cute(twoshot.tickets_r[parity, :1]), to_cute(twoshot.smtab)) if hh_r else ()
        self.nmerge = set()  # NMERGE batches (tests read it)
        self.nmform = {}  # NMERGE batch -> form 1 / 2 (tests read it)
        lp_on = leanp_batches(tp)  # LEANP (default none)
        self.leanp = set()  # LEANP batches (tests read it)
        self.carved = {}  # (kind, batch) -> carve-out percent read back after K31MK_OPROJ_PROD_CARVEOUT / _CONS_CARVEOUT (tests read it)
        for b in batches:
            publish = front is not None and b in front[2]
            fused = tp == 16 and (not mla or b >= 5) and b <= 8
            kinds = ("fused",) if fused else ("producer", "consumer")
            twoshot = b in self.twoshots and not fused
            shape = (k, b, tp, nb, delta, mla)
            options = dict(x_shared=b in x_shared, drain=b in drain) if fused else {}
            umma = not fused and engine(b, tp) == "umma"
            reduced = twoshot and b in opr["K31MK_OPROJ_PRODUCER_REDUCE"]
            if (reduced or b in opr["K31MK_OPROJ_RESIDENT"]) and not umma:
                raise ValueError(f"K31MK_OPROJ_PRODUCER_REDUCE / _RESIDENT at batch {b}: tcgen05 producer only")
            if reduced:
                self.reduced.add(b)
            if b in nm_on:  # NMERGE
                if fused or publish or b in opr["K31MK_OPROJ_CONS1"] or (not twoshot and nm_on[b] != 2):
                    raise ValueError(f"K31MK_OPROJ_NMERGE at batch {b}: split consumer without x_moe publish or CONS1 (form 1: two-shot) only")
                self.nmerge.add(b)
                self.nmform[b] = nm_on[b]
            if b in prw_batches(tp) and not reduced:  # PRW
                raise ValueError(f"K31MK_OPROJ_PRW at batch {b}: needs a two-shot producer-reduced batch")
            if b in opr["K31MK_OPROJ_CONS1"]:
                if not reduced or publish:
                    raise ValueError(f"K31MK_OPROJ_CONS1 at batch {b}: needs a two-shot producer-reduced batch without x_moe publish")
                self.cons1.add(b)
            if b in lp_on:  # LEANP
                if not twoshot or fused or b in self.cons1:
                    raise ValueError(f"K31MK_OPROJ_LEANP at batch {b}: two-shot split OprojConsumer (not CONS1) only")
                self.leanp.add(b)
            if twoshot and hh_r and (hr_rows is None or b in hr_rows):  # HH-R (row gate: only the batches of K31MK_OPROJ_HHOME_R_ROWS)
                if not umma or b in self.cons1:
                    raise ValueError(f"K31MK_OPROJ_HHOME_R at batch {b}: tcgen05 two-shot without CONS1 only")
                self.hhr[b] = 2 if reduced else 1
            if twoshot and hh_k >= 0 and (hh_rows is None or b in hh_rows):  # HH-G (row gate: only the batches of K31MK_OPROJ_HHOME_G_ROWS)
                if b in self.cons1 or publish:
                    raise ValueError(f"K31MK_OPROJ_HHOME_G at batch {b}: not with CONS1 or an x_moe publish")
                self.hhome.add(b)
            if hhm is not None and not fused and not twoshot and b in hhm.batches:  # HH-M (d0: 9..cap; s0: STATIC_PERM batches)
                if not umma or publish:
                    raise ValueError(f"K31MK_OPROJ_HHOME_M at batch {b}: tcgen05 one-shot split batches without an x_moe publish only")
                self.hhm.add(b)
            if self.lamx is not None and b in self.lamx[1] and (fused or publish or b in self.cons1):  # F-L
                raise ValueError(f"K31MK_MOE_FRONT_LAMX at batch {b}: split OprojConsumer without x_moe publish or CONS1 only")
            vote = not twoshot and b in opr["K31MK_OPROJ_VOTE"]
            by_kind = {  # package (b) options per kind; () = the unchanged kernel
                "producer": (reduced, reduced and self.residual, umma and b in opr["K31MK_OPROJ_RESIDENT"]),
                "consumer": (twoshot and b in opr["K31MK_OPROJ_BATCHED"], reduced, vote),
                "fused": (vote,),
            }
            by_kind = {kind: v if any(v) else () for kind, v in by_kind.items()}
            self.opr[b] = {kind: by_kind[kind] for kind in kinds}  # what each kernel of this batch was built with
            cdr_b = (0, True)
            cdr_tw = cdr_plain if twoshot and self.twoshot_plain is not None and b not in self.hhome else cdr_n[twoshot]  # row gate
            if b in cdr_on and not fused:
                if publish or (twoshot and not reduced) or not cdr_tw:
                    raise ValueError(f"K31MK_OPROJ_DEFER_REARM at batch {b}: split consumer without x_moe publish, one-shot or "
                                     "producer-reduced two-shot only")
                cdr_b = (sign * hhm.pstride if b in self.hhm else cdr_tw, bool(cdr_last))  # HH-M: its parity stride
                self.cdr[b] = {"defer": cdr_b[0], "keep_own": cdr_b[1]}
                if b not in _cdr_engaged:
                    print(f"K31MK_OPROJ_DEFER_REARM engaged batch={b} pid={os.getpid()}", file=sys.stderr, flush=True)
                    _cdr_engaged.add(b)
            self.codes[b] = tuple(
                compile_oproj(
                    "consumer_cta" if kind == "consumer" and b in self.cons1 else kind,
                    *shape,
                    publish and kind != "producer",
                    hhm.cap if b in self.hhm else cap,  # HH-M: the homed mailbox's rows
                    twoshot,
                    deadline_ns,
                    **options,
                    **(dict(opr=by_kind[kind]) if by_kind[kind] else {}),
                    **(dict(cdr=cdr_b) if kind == "consumer" and cdr_b[0] else {}),  # CDR
                    **(dict(nmerge=nm_on[b]) if kind == "consumer" and b in self.nmerge else {}),  # NMERGE (form)
                    **(dict(hhome=hh_k) if b in self.hhome and (kind == "consumer" or by_kind[kind][:1] == (True,)) else {}),  # HH-G
                    **(dict(hhdyn=True) if b in self.hhome and hh_dyn and kind in ("consumer", "producer") else {}),  # HH-G dynamic
                    **(dict(hhome_r=self.hhr[b]) if b in self.hhr and kind == "producer" else {}),  # HH-R
                    **(dict(hhome_r=1) if self.hhr.get(b) == 1 and kind == "consumer" else {}),  # HH-R consumer-reduced
                    **(dict(hhome_m=hhm.form) if b in self.hhm else {}),  # HH-M (producer and consumer; the form)
                    **(dict(lamx=True) if kind == "consumer" and self.lamx is not None and b in self.lamx[1] else {}),  # F-L
                    **(dict(leanp=True) if kind == "consumer" and b in self.leanp else {}),  # LEANP
                )
                for kind in kinds
            )
            pcarve = opr["K31MK_OPROJ_PROD_CARVEOUT"].get(b)
            if pcarve is not None and not fused:  # K31MK_OPROJ_PROD_CARVEOUT: the split producer's carve-out
                self.codes[b] = (prepare_shared_carveout(self.codes[b][0], self.device.index, pcarve), *self.codes[b][1:])
                self.carved[("prod", b)] = carveout_now(self.codes[b][0], self.device.index)
            if b in carve and twoshot:  # K31MK_OPROJ_CONS_CARVEOUT: the two-shot consumer's preferred carve-out
                self.codes[b] = (self.codes[b][0], prepare_shared_carveout(self.codes[b][1], self.device.index, carve[b]))
                self.carved[("cons", b)] = carveout_now(self.codes[b][1], self.device.index)
        if self.nmerge and frozenset(self.nmerge) not in _NMERGE_ENGAGED:
            _NMERGE_ENGAGED.add(frozenset(self.nmerge))
            print(f"K31MK_OPROJ_NMERGE engaged rows={sorted(self.nmerge)} forms={sorted(set(self.nmform.values()))} pid={os.getpid()}",
                  file=sys.stderr, flush=True)
        prw = frozenset(prw_batches(tp) & self.reduced) if self.twoshot else frozenset()
        if prw and prw not in _PRW_ENGAGED:
            _PRW_ENGAGED.add(prw)
            print(f"K31MK_OPROJ_PRW engaged rows={sorted(prw)} pid={os.getpid()}", file=sys.stderr, flush=True)
        ring = {b: v for b, v in ring_batches(tp).items() if b in self.codes and len(self.codes[b]) > 1}  # RING (default none)
        if ring and frozenset(ring.items()) not in _RING_ENGAGED:
            _RING_ENGAGED.add(frozenset(ring.items()))
            print(f"K31MK_OPROJ_RING engaged rows={sorted(ring.items())} pid={os.getpid()}", file=sys.stderr, flush=True)
        if self.cons1 and frozenset(self.cons1) not in _CONS1_ENGAGED:
            _CONS1_ENGAGED.add(frozenset(self.cons1))
            print(f"K31MK_OPROJ_CONS1 engaged rows={sorted(self.cons1)} pid={os.getpid()}", file=sys.stderr, flush=True)
        if self.leanp and frozenset(self.leanp) not in _LEANP_ENGAGED:
            _LEANP_ENGAGED.add(frozenset(self.leanp))
            print(f"K31MK_OPROJ_LEANP engaged rows={sorted(self.leanp)} pid={os.getpid()}", file=sys.stderr, flush=True)
        if self.carved and frozenset(self.carved.items()) not in _CARVE_ENGAGED:
            _CARVE_ENGAGED.add(frozenset(self.carved.items()))
            prod = sorted((b, v) for (kind, b), v in self.carved.items() if kind == "prod")
            cons = sorted((b, v) for (kind, b), v in self.carved.items() if kind == "cons")
            print(f"K31MK_OPROJ_CARVEOUT engaged prod={prod} cons={cons} pid={os.getpid()}", file=sys.stderr, flush=True)
        if self.hhm and frozenset(self.hhm) not in _HHM_ENGAGED:
            _HHM_ENGAGED.add(frozenset(self.hhm))
            print(f"K31MK_OPROJ_HHOME_M engaged form={getattr(hhm, 'spec', hhm.form)} rows={sorted(self.hhm)} pid={os.getpid()}",
                  file=sys.stderr, flush=True)
        if self.twoshot_plain is not None:  # HH-G row gate: which two-shot batches are homed / plain in this object
            key = (frozenset(self.hhome), frozenset(self.twoshots & set(self.codes)))
            if key not in _HHG_ROWS_ENGAGED:
                _HHG_ROWS_ENGAGED.add(key)
                print(f"K31MK_OPROJ_HHOME_G_ROWS engaged homed={sorted(self.hhome)} plain={sorted(key[1] - key[0])} pid={os.getpid()}",
                      file=sys.stderr, flush=True)
        if self.twoshot_rplain is not None:  # HH-R row gate: which two-shot batches use the homed / plain receive slots
            key = (frozenset(self.hhr), frozenset(self.twoshots & set(self.codes)))
            if key not in _HHR_ROWS_ENGAGED:
                _HHR_ROWS_ENGAGED.add(key)
                print(f"K31MK_OPROJ_HHOME_R_ROWS engaged homed={sorted(self.hhr)} plain={sorted(key[1] - key[0])} pid={os.getpid()}",
                      file=sys.stderr, flush=True)

    def run(self, local, prefix, blocks, out):
        batch, s = local.shape[0], stream(self.device)
        codes, x = self.codes[batch], to_cute(local)
        front = (None, None)
        if self.front is not None and batch in self.front[2]:
            front = self.front[:2]
        if self.lamx is not None and batch in self.lamx[1]:  # F-L: the Lamport x box view rides in the hints slot
            front = (None, self.lamx[0])
        rows = (dynamic_view(prefix), dynamic_view(blocks), *self.views)
        rows += (dynamic_view(out), *front, *self.eps, s)
        if len(codes) == 1:
            codes[0](x, self.wview, *self.head, *rows)
        elif batch in self.twoshots:
            gplain = self.twoshot_plain is not None and batch not in self.hhome  # HH-G row gate: plain gather
            if self.twoshot_rplain is not None and batch not in self.hhr:  # HH-R row gate: plain receive slots
                head, gather, tail = self.twoshot_rplain[gplain]
            else:
                head, gather, tail = self.twoshot_plain if gplain else self.twoshot
            if batch in self.reduced:  # package (b): the producer reduces into the gather
                extra = (gather, tail[1]) + ((rows[0],) if self.residual else ())
                codes[0](x, self.wview, *head, *extra, *(self.hhr_args if self.hhr.get(batch) == 2 else ()), s)
            else:
                codes[0](x, self.wview, *head, s)
            codes[1](gather, *rows[:-1], *tail, *(self.hh_args if batch in self.hhome else ()), s)
        elif batch in self.hhm:  # HH-M: the homed one-shot mailbox, + this parity's ticket word and the SM -> locality domain table
            codes[0](x, self.wview, *self.hhm_head, s)
            codes[1](self.hhm_box, *rows[:-1], *self.hhm_args, s)
        else:
            codes[0](x, self.wview, *self.head, s)
            codes[1](self.box, *rows)
        return out, prefix, blocks
