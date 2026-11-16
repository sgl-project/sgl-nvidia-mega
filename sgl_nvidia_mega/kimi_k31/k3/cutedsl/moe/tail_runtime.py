# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Options described as "ported" follow NVIDIA's Rubin (sm_107) specialized-decode kernels (the rubin package).
"""Routed MoE tail: one fused kernel finalizes this rank's top-16 rows, multicasts the
latent partial and peer-stores the shared shard (three rotating parities), all-reduces
and RMS-normalizes per K-slice, runs the K-split up projection and multicasts the output
row into the next AttnRes mailbox. At TP8 B5+ and TP16 B9+ the experts publish the
latent partial, WaveMoeTail.push the shared shard and WaveMoeTail does the rest. At
``twoshot_batches`` the latent all-reduce is ``common.twoshot``: the experts push each
partial chunk to its column owner, whose WaveMoeTail CTAs sum and multicast it."""

import os
import sys
from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8
from cutlass.cute.runtime import make_fake_stream

from ..common import k19
from ..common.host import fake, prepare_shared_carveout, stream, tensor_maps, to_cute
from ..common.policy import batch_table
from ..common.twoshot import TwoShot
from ..oproj.prepared import OprojMailbox
from .tail import MoeTail, WaveMoeTail

# MoE tail mailbox tokens (padded decode batches) of a boot up to 64 rows (``MoeTailBox``
# takes the boot's capacity, as ``oproj.prepared.capacity``); small-batch front_poll
# G/W/S rows per parity.
CAP, POLL_CAP = 64, 8
# Two-shot latent all-reduce batches per TP (B9+): 2-/4-node GB300, us/layer of RS push
# + experts + WaveMoeTail, max over ranks, multicast vs two-shot: TP8 B16 41.74/42.80,
# B32 52.20/53.18, B64 79.27/81.50; TP16 B16 33.98/34.49, B32 42.19/42.52, B64
# 65.05/60.51. TP16 65-128 (Breq 16) keeps B64's choices in this table, EARLY and UMMA
# (not measured above 64 rows).
TWOSHOT = {8: "", 16: "33-128:1"}
# WaveMoeTail batches that release the next AttnRes at entry.
EARLY = {8: "9-64:1", 16: "9-128:1"}
# WaveMoeTail batches whose up projection runs on tcgen05 (B9+; 1-GPU WaveMoeTail us
# mma.sync vs tcgen05, B16/32/64: TP8 4.53/5.80/9.05 vs 4.45/5.26/7.80, TP16 two-shot
# 4.50/5.39/7.75 vs 4.52/5.46/7.21).
UMMA = {8: "9-64:1", 16: "33-128:1"}
# VR200 re-tune of the three GB300 tables above (default unset: no change). TP16 batch
# tables appended to TWOSHOT / EARLY / UMMA (later entries win), rows 9-128, values
# 0 / 1, e.g. K31MK_MOE_TAIL_TWOSHOT="64:0" (one-shot latent all-reduce at 64 rows).
# Read when ``PreparedTail`` / ``MoeTailWeights`` are built.
TABLE_ENV = {"twoshot": "K31MK_MOE_TAIL_TWOSHOT", "early": "K31MK_MOE_TAIL_EARLY", "umma": "K31MK_MOE_TAIL_UMMA"}


def tail_table(name, tp):
    """``TWOSHOT`` / ``EARLY`` / ``UMMA`` (``name`` "twoshot" / "early" / "umma") of
    ``tp``, then that table's ``TABLE_ENV`` entries at TP16."""
    default = {"twoshot": TWOSHOT, "early": EARLY, "umma": UMMA}[name].get(tp, "")
    extra = os.environ.get(TABLE_ENV[name], "") if tp == 16 else ""
    bad = {b: v for b, v in batch_table(extra).items() if not 9 <= b <= 128 or v not in (0, 1)}
    if bad:
        raise ValueError(f"{TABLE_ENV[name]}: rows 9-128 with value 0 or 1 only, got {bad}")
    return batch_table(f"{default},{extra}")


def flat(t):
    return to_cute(t.reshape(-1), 16)


def front_options(tp, batch):
    """Front-mode MoeTail poll options, ported from the source's Rubin tables
    (RUBIN_POLL_NS, RUBIN_GATE, RUBIN_TAIL_RELOAD), default off:
    ``K31MK_MOE_TAIL_POLL_NS`` (ns), ``K31MK_MOE_TAIL_GATE`` (TP16) and
    ``K31MK_MOE_TAIL_RELOAD`` (batch tables such as "3-8:1"). Returns (poll_ns or None,
    gate, reload); read once per ``MoeTailWeights``."""
    ns = os.environ.get("K31MK_MOE_TAIL_POLL_NS", "")
    gate = batch_table(os.environ.get("K31MK_MOE_TAIL_GATE", "")).get(batch, 0)
    reload = batch_table(os.environ.get("K31MK_MOE_TAIL_RELOAD", "")).get(batch, 0)
    return (int(ns) if ns else None), bool(gate) and tp == 16, bool(reload)


def late_rms_mode(tp, batch):
    """Default off: ``K31MK_MOE_TAIL_LATE_RMS``, a batch table such as
    "8:1,16:1" (TP16 only): 1 = late RMS (the up projection on BF16(x * gamma), 1/rms
    after it; ``MoeTail`` / ``WaveMoeTail`` ``late_rms``), 3 = late RMS with the squares
    sent to the cluster leader only (front-mode ``MoeTail``). Read once per
    ``MoeTailWeights``."""
    if tp != 16:
        return 0
    return batch_table(os.environ.get("K31MK_MOE_TAIL_LATE_RMS", "")).get(batch, 0)


def fp32_rms_mode(tp, batch):
    """K-G3-03 (default off): ``K31MK_MOE_TAIL_FP32_RMS``, a batch table
    such as "16:1" (TP16 one-shot WaveMoeTail batches, B9-32): the shared reduced latent
    rows are FP32 and the RMS takes its squares from the FP32 sums (``WaveMoeTail``
    ``fp32_rms``), as stock and MoeTail do. Read once per ``MoeTailWeights``."""
    if tp != 16 or batch <= 8:
        return False
    return bool(batch_table(os.environ.get("K31MK_MOE_TAIL_FP32_RMS", "")).get(batch, 0))


def defer_rearm_mode():
    """K31MK_ATTNRES_DEFER_REARM=1 (DRA, default off; bitwise): the tail output mailbox gets two parities and
    MoE ordinal k publishes into parity k % 2, so the next ordinal's AttnRes re-arms these rows before its own poll
    (attn_res ``rearm_prev``) instead of this ordinal's AttnRes after its output. Read when the box is built."""
    return os.environ.get("K31MK_ATTNRES_DEFER_REARM", "0") == "1"


def published(tp, batch):
    """The experts publish the latent and WaveMoeTail runs the tail (else MoeTail
    front); also the TP16 B<=8 batches of ``extra_published`` (TAE)."""
    return batch > (4 if tp == 8 else 8) or batch in extra_published(tp)


def extra_published(tp):
    """TAE (default none; ``K31MK_MOE_TP16_PUBLISH``): TP16 B<=8
    batches that publish too, a batch table such as "8:1". Read at construction."""
    spec = os.environ.get("K31MK_MOE_TP16_PUBLISH", "") if tp == 16 else ""
    return frozenset(b for b, on in batch_table(spec).items() if on and b <= 8)


WAVE8 = ("direct", "lean_owner", "early_rearm", "early", "late1", "owner_squares", "poll0", "reload", "store16",
         "tfence", "sxbar", "vote")  # tfence: fence before the Wave trigger; sxbar: SX barrier;
# vote: the ported warp-vote direct poll (needs direct)


def sxbar_mode(tp, batch):
    """Default off: ``K31MK_MOE_TAIL_SXBAR``, a batch
    table such as "8:1": that batch's MoeTail (front modes) ends with a trailing cluster barrier
    (``MoeTail`` ``sxbar``). The published WaveMoeTail takes WAVE8 "sxbar" instead. Read by
    ``compile_tail``, so every caller of it sees the knob."""
    return bool(batch_table(os.environ.get("K31MK_MOE_TAIL_SXBAR", "")).get(batch, 0))


def wave8_options(tp, batch):
    """TAE (default off): ``K31MK_MOE_TAIL_WAVE8``, a "+"-joined subset of
    ``WAVE8`` for the published TP16 B<=8 WaveMoeTail (the source's 8-row tail
    engine): returns the sorted option names (``early`` is the
    entry release, the others are ``WaveMoeTail`` keywords)."""
    spec = os.environ.get("K31MK_MOE_TAIL_WAVE8", "")
    if tp != 16 or batch > 8 or not spec or batch not in extra_published(tp):
        return ()
    names = tuple(sorted(set(spec.split("+"))))
    bad = [n for n in names if n not in WAVE8]
    if bad:
        raise ValueError(f"K31MK_MOE_TAIL_WAVE8: unknown options {bad}")
    return names


def k19_tail(mode, batch):
    """K-19 (``K31MK_K19_FENCE`` site tail9, default off): the 9-128-row published / two-shot WaveMoeTail
    fences before every trigger (``tfence``). Rows <= 8 keep ``K31MK_MOE_TAIL_WAVE8`` "tfence". Asked only for
    such batches, so the engagement line appears only when a fenced tail is built. Read when ``PreparedTail``
    is built."""
    return mode in ("publish", "twoshot") and 9 <= batch <= 128 and k19.fence("tail9")


SPREAD_ENV, _spread_engaged = "K31MK_MOE_TAIL_SPREAD", set()


def spread_batches(tp):
    """``K31MK_MOE_TAIL_SPREAD`` (default unset: off): TP16 two-shot batches whose WaveMoeTail
    reduces this rank's latent slice in all 16 CTAs of each cluster instead of only CTA q == rank (``WaveMoeTail``
    ``spread``; bitwise the default's outputs). A batch table such as "33-128:1" (rows 9-128, 0 / 1); a listed batch
    that is not two-shot fails at ``PreparedTail`` build. Read when ``PreparedTail`` is built."""
    extra = os.environ.get(SPREAD_ENV, "") if tp == 16 else ""
    table = batch_table(extra)
    bad = {b: v for b, v in table.items() if not 9 <= b <= 128 or v not in (0, 1)}
    if bad:
        raise ValueError(f"{SPREAD_ENV}: rows 9-128 with value 0 or 1 only, got {bad}")
    return {batch for batch, on in table.items() if on}


def poll0_batches(tp):
    """Default unset (none): ``K31MK_MOE_TAIL_POLL0``, a TP16 batch table such as "16:1,96:1,128:1" (rows 9-128,
    values 0 / 1): that published / two-shot WaveMoeTail re-polls its Lamport sources and reduced / gathered rows without the 64 ns
    backoff (``WaveMoeTail`` ``poll0``; the B<=8 TAE form is ``K31MK_MOE_TAIL_WAVE8`` "poll0"). Read when ``MoeTailWeights`` is built."""
    extra = os.environ.get("K31MK_MOE_TAIL_POLL0", "") if tp == 16 else ""
    table = batch_table(extra)
    bad = {b: v for b, v in table.items() if not 9 <= b <= 128 or v not in (0, 1)}
    if bad:
        raise ValueError(f"K31MK_MOE_TAIL_POLL0: rows 9-128 with value 0 or 1 only, got {bad}")
    return {batch for batch, on in table.items() if on}


def sq_owner_batches(tp):
    """Default unset (none): ``K31MK_MOE_TAIL_SQ_OWNER``, a TP16 batch table such as "64:1,128:1" (rows 9-128,
    values 0 / 1): that published / two-shot WaveMoeTail, which must run late RMS (``K31MK_MOE_TAIL_LATE_RMS`` value 1), sends each
    warp's squares row only to its tokens' line owners (``WaveMoeTail`` ``sq_owner``). Read when ``MoeTailWeights`` is built."""
    extra = os.environ.get("K31MK_MOE_TAIL_SQ_OWNER", "") if tp == 16 else ""
    table = batch_table(extra)
    bad = {b: v for b, v in table.items() if not 9 <= b <= 128 or v not in (0, 1)}
    if bad:
        raise ValueError(f"K31MK_MOE_TAIL_SQ_OWNER: rows 9-128 with value 0 or 1 only, got {bad}")
    return {batch for batch, on in table.items() if on}


def tswap_batches(tp):
    """Default unset (none): ``K31MK_MOE_TAIL_TSWAP``, a TP16 batch table such as "96-128:2" (rows 33-128, values
    0 / 1 / 2): that two-shot tcgen05 WaveMoeTail runs its up projection with the tokens as M = 128 and pushes each token's partials
    from registers (1: no SMEM stage) or through a swizzled [token, row] SMEM stage and the served push loop (2: TSWAP-S)
    (``WaveMoeTail`` ``tswap``). Returns {batch: value} of the nonzero entries. Read when ``MoeTailWeights`` is built."""
    extra = os.environ.get("K31MK_MOE_TAIL_TSWAP", "") if tp == 16 else ""
    table = batch_table(extra)
    bad = {b: v for b, v in table.items() if not 33 <= b <= 128 or v not in (0, 1, 2)}
    if bad:
        raise ValueError(f"K31MK_MOE_TAIL_TSWAP: rows 33-128 with value 0, 1 or 2 only, got {bad}")
    return {batch: v for batch, v in table.items() if v}


def k224_batches(tp):
    """Default unset (none): ``K31MK_MOE_TAIL_K224``, a TP16 batch table such as "33-128:1" (rows 33-128, values
    0 / 1): that tcgen05 WaveMoeTail skips the two all-padding K16 MMAs of its slice's last 64-K block (``WaveMoeTail`` ``k224``).
    Read when ``MoeTailWeights`` is built."""
    extra = os.environ.get("K31MK_MOE_TAIL_K224", "") if tp == 16 else ""
    table = batch_table(extra)
    bad = {b: v for b, v in table.items() if not 33 <= b <= 128 or v not in (0, 1)}
    if bad:
        raise ValueError(f"K31MK_MOE_TAIL_K224: rows 33-128 with value 0 or 1 only, got {bad}")
    return {batch for batch, on in table.items() if on}


def rearm_bal_batches(tp):
    """Default unset (none): ``K31MK_MOE_TAIL_REARM_BAL``, a TP16 batch table such as "16:1,24:1,32:1" or "33-128:1" (rows 9-128,
    values 0 / 1 / 2): that published one-shot or two-shot WaveMoeTail spreads its trailing re-arm stores over all CTAs (1), or issues
    those balanced stores before the landing wait instead of after the output (2) (``WaveMoeTail`` ``rearm_bal``; the same stores).
    Returns {batch: value} of the nonzero entries. Read when ``MoeTailWeights`` is built."""
    extra = os.environ.get("K31MK_MOE_TAIL_REARM_BAL", "") if tp == 16 else ""
    table = batch_table(extra)
    bad = {b: v for b, v in table.items() if not 9 <= b <= 128 or v not in (0, 1, 2)}
    if bad:
        raise ValueError(f"K31MK_MOE_TAIL_REARM_BAL: rows 9-128 with value 0, 1 or 2 only, got {bad}")
    return {batch: v for batch, v in table.items() if v}


def wlate_batches(tp):
    """WLATE (default unset: none): ``K31MK_MOE_TAIL_WLATE``, a TP16 batch table such as "16:1,24:1,32:1"
    (rows 9-32, values 0-6): that published one-shot mma.sync WaveMoeTail loads its weight fragments after phase 1 (1), the same plus a
    bulk L2 prefetch of its weight rows at entry (2), right behind the phase-1 source loads (3), 2 with the fragments after phase 2 (4),
    2 with the prefetch from warps 6-7 (5), or 5's prefetch with 4's fragment point (6), instead of at entry before the
    Lamport poll (``WaveMoeTail`` ``wlate``; same loads, bitwise). Returns {batch: value} of the nonzero entries. Read when
    ``MoeTailWeights`` is built."""
    extra = os.environ.get("K31MK_MOE_TAIL_WLATE", "") if tp == 16 else ""
    table = batch_table(extra)
    bad = {b: v for b, v in table.items() if not 9 <= b <= 32 or v not in (0, 1, 2, 3, 4, 5, 6)}
    if bad:
        raise ValueError(f"K31MK_MOE_TAIL_WLATE: rows 9-32 with value 0-6 only, got {bad}")
    return {batch: v for batch, v in table.items() if v}


_rbal_seen: set = set()


def _rbal_line(text, tp, batch):
    """One stderr line '<text> tp=<tp> batch=<b> pid=<p>' per (text, tp, batch) per process (REARM_BAL engagement / decline)."""
    if (text, tp, batch) not in _rbal_seen:
        _rbal_seen.add((text, tp, batch))
        print(f"{text} tp={tp} batch={batch} pid={os.getpid()}", file=sys.stderr, flush=True)


_mfx_seen: set = set()


def _mfx_engaged(name, tp, batch):
    """One stderr line '<name> engaged tp=<tp> batch=<b> pid=<p>' per (knob, tp, batch) per process (tail knobs)."""
    if (name, tp, batch) not in _mfx_seen:
        _mfx_seen.add((name, tp, batch))
        print(f"{name} engaged tp={tp} batch={batch} pid={os.getpid()}", file=sys.stderr, flush=True)


def twoshot_batches(tp):
    table = tail_table("twoshot", tp)
    return {batch for batch, on in table.items() if on and batch > 8}


def early_batches(tp):
    table = tail_table("early", tp)
    return {batch for batch, on in table.items() if on and published(tp, batch)}


def umma_batches(tp):
    table = tail_table("umma", tp)
    return {batch for batch, on in table.items() if on and batch > 8}


def shares(tp, batch, twoshot=False):
    """WaveMoeTail clusters share reduced latent rows (all 112 CTAs must be
    co-resident)."""
    return published(tp, batch) and WaveMoeTail(batch, tp, twoshot).reduce


@cache
def resident_clusters(device):
    """Co-resident 16-CTA clusters at one CTA per SM."""
    with torch.accelerator.device_index(device):
        return cutlass.utils.HardwareInfo(device).get_max_active_clusters(16)


class MoeTailBox:
    """Model-owned MoE tail mailboxes: latent [3, TP, cap, 3584] (multicast), shared
    reduce-scatter [3, TP, cap, 7168/TP] (peer stores), output [1, cap, 7168], the
    local all-reduced latent rows [3, cap, 3584] that WaveMoeTail clusters share and the
    two-shot latent buffers (``twoshot_args``/``twoshot_tail`` replace args/tail).
    ``cap``: the boot's rows (``oproj.prepared.capacity``: CAP up to 64-row boots, else
    the largest batch); the tail kernels take it at run time (``MoeTailWeights``)."""

    def __init__(self, group, device, tp, rank, cap=CAP):
        if cap < CAP:
            raise ValueError(f"MoE tail capacity {cap} < {CAP}")
        self.rank, self.cap = rank, cap
        self.latent = OprojMailbox(group, device, tp, rank, (3, tp, cap, 3584))
        self.rs = OprojMailbox(group, device, tp, rank, (3, tp, cap, 7168 // tp))
        # DRA (``defer_rearm_mode``): [2, cap, 7168], parity = MoE ordinal % 2 (``out_parity``); else [1, cap, 7168].
        self.dra = defer_rearm_mode()
        self.out = OprojMailbox(group, device, tp, rank, (2 if self.dra else 1, cap, 7168))
        ptrs = self.rs.handle.buffer_ptrs
        self.peers = torch.tensor(ptrs, dtype=torch.int64, device=device)
        self.args = (flat(self.latent.buffer), flat(self.rs.buffer))
        self.tail: tuple = (to_cute(self.peers, 16), flat(self.out.buffer))
        self.tail += (Int64(self.latent.mc), Int64(self.out.mc))
        self.reduced = torch.empty((3, cap, 3584), dtype=torch.bfloat16, device=device)
        self.reduced.view(torch.int32).fill_(-0x80000000)
        # ``fp32_rms`` batches (K31MK_MOE_TAIL_FP32_RMS): FP32 reduced rows, as BF16 pairs.
        self.reduced32 = None
        if os.environ.get("K31MK_MOE_TAIL_FP32_RMS", ""):
            self.reduced32 = torch.empty((3, cap, 7168), dtype=torch.bfloat16, device=device)
            self.reduced32.view(torch.int32).fill_(-0x80000000)
        self.twoshot = box = TwoShot(group, device, tp, rank, 3584, cap, parities=3)
        self.twoshot_args = (flat(box.recv), self.args[1])
        self.twoshot_tail = (*self.tail[:2], Int64(box.gather_mc), self.tail[3])
        # Experts' push: every parity's receive slots [3 x TP, cap, 3584/TP] and the
        # owners' buffer bases (slot parity x TP + rank).
        self.push_box = (
            to_cute(box.recv.view(3 * tp, cap, -1)),
            to_cute(box.peers[0], 16),
        )

    def out_parity(self, parity):
        """DRA: (the [1, cap, 7168] view of output parity ``parity``, its multicast address)."""
        view = self.out.buffer[parity : parity + 1]
        return view, Int64(self.out.mc + parity * self.out.buffer[0].numel() * 2)


@cache
def compile_tail(
    tp,
    batch,
    device,
    mode="front",
    trigger=2,
    early=False,
    umma=False,
    deadline_ns=0,
    poll_ns=None,
    gate=False,
    reload=False,
    late_rms=0,
    fp32_rms=False,
    wave8=(),
    cap=CAP,
    k19_tfence=False,
    spread=False,
    poll0=False,
    sq_owner=False,
    tswap=False,
    k224=False,
    rearm_bal=0,
    wlate=0,
):
    """front: rows complete at the PDL wait; front_poll: three 0xffff-armed G/W/S parity
    slots polled per row and re-armed after use; publish (``published``): WaveMoeTail
    over the published mailboxes, reading the weight shard directly, not by TMA;
    twoshot: publish over the two-shot receive slots and gather. ``cap``: the
    ``MoeTailBox`` rows (callers pass it only when it is not CAP, so 64-row boots keep
    their cache keys). ``k19_tfence`` (``K31MK_K19_FENCE`` tail9, ``k19_tail``; callers pass it
    only when set): the 9-128-row WaveMoeTail's ``tfence``."""
    m8, twoshot = mode in ("publish", "twoshot"), mode == "twoshot"
    if poll0 and not (m8 and 9 <= batch <= 128):
        raise ValueError(f"K31MK_MOE_TAIL_POLL0: published / two-shot batches 9-128 only, got {mode} B{batch}")
    if sq_owner and not (m8 and 9 <= batch <= 128 and late_rms == 1):
        raise ValueError(f"K31MK_MOE_TAIL_SQ_OWNER: late-RMS published / two-shot batches 9-128 only, got {mode} B{batch} late {late_rms}")
    if tswap and not (twoshot and umma and tp == 16 and 33 <= batch <= 128):
        raise ValueError(f"K31MK_MOE_TAIL_TSWAP: TP16 two-shot umma batches 33-128 only, got {mode} B{batch} umma {umma}")
    if k224 and not (m8 and umma and tp == 16 and 33 <= batch <= 128):
        raise ValueError(f"K31MK_MOE_TAIL_K224: TP16 tcgen05 batches 33-128 only, got {mode} B{batch} umma {umma}")
    if rearm_bal and not (tp == 16 and rearm_bal in (1, 2) and ((m8 and 9 <= batch <= 64) or (twoshot and 33 <= batch <= 128))):
        raise ValueError(f"K31MK_MOE_TAIL_REARM_BAL: TP16 published / two-shot batches 9-64 or two-shot 33-128, value 1 or 2 only, got {mode} B{batch} {rearm_bal}")
    if wlate and not (mode == "publish" and 9 <= batch <= 32 and not umma and wlate in (1, 2, 3, 4, 5, 6)):
        raise ValueError(f"K31MK_MOE_TAIL_WLATE: one-shot mma.sync batches 9-32, value 1-6 only, got {mode} B{batch} umma {umma} {wlate}")
    if k19_tfence and not (m8 and 9 <= batch <= 128):
        raise ValueError(f"K31MK_K19_FENCE tail9: published / two-shot batches 9-128 only, got {mode} B{batch}")
    vec = lambda n, dtype=BFloat16, align=16: fake((n,), None, dtype, align)
    slots = 3 * POLL_CAP if mode == "front_poll" else batch
    latent = 3 * tp * cap * (3584 // tp if twoshot else 3584)
    tensors: tuple = (vec(latent), vec(3 * tp * cap * (7168 // tp)))
    weights = vec(7168 // tp * 3584) if m8 and not umma else vec(128, Uint8, 64)
    tensors += (weights, vec(3584))
    tensors += (vec(slots * 16 * 3584), vec(slots * 16), vec(slots * 7168))
    tensors += (vec(tp, Int64), vec(cap * 7168))
    scalars: tuple = (Int64(0), Int64(0), Int32(0), Int32(0), Int32(cap), Float32(1e-6))
    scalars += (Int32(2), Int32(cap))
    kernel: MoeTail
    if m8:
        late = dict(late_rms=late_rms) if late_rms else {}
        if fp32_rms:
            late["fp32_rms"] = True
        late.update({o: True for o in wave8 if o != "early"})  # TAE (``wave8_options``)
        if k19_tfence:  # K-19 tail9: fence.acq_rel.gpu before every trigger (WaveMoeTail ``tfence``)
            late["tfence"] = True
        if spread:  # K31MK_MOE_TAIL_SPREAD (b3): every CTA of a cluster reduces (two-shot only)
            if not twoshot:
                raise ValueError(f"{SPREAD_ENV}: two-shot batches only, got {mode} B{batch}")
            late["spread"] = True
        if poll0:  # _MOE_TAIL_POLL0 (9-128 rows): no re-poll backoff
            late["poll0"] = True
        if sq_owner:  # _MOE_TAIL_SQ_OWNER (9-128 rows, late RMS): squares rows to the line owners only
            late["sq_owner"] = True
        if tswap:  # K31MK_MOE_TAIL_TSWAP (33-128 rows, two-shot umma): tokens as tcgen05 M (1: register pushes, 2: TSWAP-S)
            late["tswap"] = int(tswap)
        if k224:  # K31MK_MOE_TAIL_K224 (33-128 rows, tcgen05): the slice's last K block stops at K 224
            late["k224"] = True
        if rearm_bal:  # K31MK_MOE_TAIL_REARM_BAL (one-shot 9-64 / two-shot 33-128 rows): trailing re-arm stores over all CTAs (1) / before the landing wait (2)
            late["rearm_bal"] = int(rearm_bal)
        if wlate:  # K31MK_MOE_TAIL_WLATE (9-32 rows, one-shot): weight fragments after phase 1 (1, 2 + L2 prefetch) / behind its loads (3)
            late["wlate"] = int(wlate)
        early = early or "early" in wave8
        kernel = wave = WaveMoeTail(batch, tp, twoshot, early, umma, deadline_ns, **late)
        width = 7168 if fp32_rms else 3584  # FP32 reduced rows as BF16 pairs
        scalars += (vec(3 * cap * width),) if wave.reduce else ()
    else:
        options = dict(poll_ns=poll_ns, gate=gate, reload=reload)
        if late_rms:
            options["late_rms"] = late_rms
        if sxbar_mode(tp, batch):  # passed only when set: default cache keys stay
            options["sxbar"] = True
        kernel = MoeTail(
            tp, batch, mode=mode, trigger=trigger, deadline_ns=deadline_ns, **options
        )
    options = "" if m8 else "--ptxas-options=--maxrregcount=120"
    with torch.accelerator.device_index(device):
        args = (*tensors, *scalars, make_fake_stream())
        code = cute.compile(kernel, *args, options=options)
        # MoE tail requests 100% shared carveout so the next large PDL kernel can
        # coexist.
        return prepare_shared_carveout(code, device, 100)


def push_pdl_mode(tp, batch):
    """Default off (placement knob): ``K31MK_MOE_TAIL_PUSH_PDL``,
    a TP16 batch table such as "8:1": the published batch's shared-row push launches with
    PDL behind the shared down GEMM (``WaveMoeTail`` ``push_pdl``). Read at construction."""
    if tp != 16:
        return False
    return bool(batch_table(os.environ.get("K31MK_MOE_TAIL_PUSH_PDL", "")).get(batch, 0))


@cache
def compile_push(batch, device, tp=8, **pdl):
    args = (fake((batch, 7168)), fake((tp,), None, Int64), Int32(0), Int32(0))
    with torch.accelerator.device_index(device):
        return cute.compile(
            WaveMoeTail(batch, tp, **pdl).push, *args, Int32(CAP), make_fake_stream()
        )


class MoeTailWeights:
    """One layer's up-projection shard [7168/TP, 3584], RMS weight and TMA map."""

    def __init__(
        self,
        tp,
        weight,
        gamma,
        eps,
        ordinal,
        count,
        batches,
        twoshots=(),
        deadline_ns=0,
        cap=CAP,
    ):
        """``cap``: the rows of the ``MoeTailBox`` this layer runs over (its kernels'
        run-time capacity)."""
        self.weight, self.gamma = weight, gamma
        # Three parities rotate over the model's MoE layers; each launch re-arms the
        # previous MoE layer's slots of its batch (wrapping across decode steps, where
        # the first layer re-arms the largest batch the last layer may have published).
        parity, rearm = ordinal % 3, (ordinal - 1) % count % 3
        if parity == rearm:  # a launch must never re-arm the slots it publishes to
            raise ValueError(f"MoE tail parities collide for {count} MoE layers")
        if max(batches) > cap:
            raise ValueError(f"MoE tail: batch {max(batches)} exceeds the tail box's {cap} rows")
        self.parity, rearm = Int32(parity), Int32(rearm)
        self.cap = cap
        self.scalars = (Int32(cap), Float32(eps), rearm)
        grown = dict(cap=cap) if cap != CAP else {}  # passed only when set: default cache keys stay
        self.wrap = 0 if ordinal else max(batches)
        self.out_parity = ordinal % 2  # DRA only (``MoeTailBox.dra``): the output parity this layer publishes into
        cols = 512 if tp == 8 else 448
        u16 = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT16
        dims = (cols // 2, 3584 // (cols // 2), 7168 // tp)
        spec: tuple = (weight.data_ptr(), u16, dims, (cols, 7168), (cols // 2, 2, 32))
        spec += (cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,)
        spec += (cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,)
        self.descriptor = tensor_maps([spec], weight.device)
        self.views = (to_cute(self.descriptor, 64), to_cute(gamma, 16))
        self.codes, device, early = {}, weight.device.index, early_batches(tp)
        wlates = wlate_batches(tp)  # K31MK_MOE_TAIL_WLATE
        self.fp32 = {}  # batches whose WaveMoeTail shares FP32 reduced rows
        umma, spread = umma_batches(tp), spread_batches(tp)
        polled0, owned_sq = poll0_batches(tp), sq_owner_batches(tp)  # _MOE_TAIL_POLL0 / _SQ_OWNER
        swapped = tswap_batches(tp)  # _MOE_TAIL_TSWAP
        k224s = k224_batches(tp)  # _MOE_TAIL_K224
        rbals = rearm_bal_batches(tp)  # _MOE_TAIL_REARM_BAL
        if any(published(tp, b) and b in umma for b in batches):
            # WaveMoeTail tcgen05: [64 x lines, 64] boxes of the shard, 128B swizzle
            spec = (weight.data_ptr(), u16, (3584, 7168 // tp), (7168,))
            spec += ((64, 128 if tp == 8 else 64),)
            spec += (cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,)
            spec += (cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,)
            self.umma_map = tensor_maps([spec], weight.device)
            self.umma_views = (to_cute(self.umma_map, 64), self.views[1])
        for b in batches:  # WaveMoeTail reads the shard directly, not by TMA
            twoshot = b in twoshots
            mode = "twoshot" if twoshot else "publish" if published(tp, b) else "front"
            views = self.views if mode == "front" else (flat(weight), self.views[1])
            on = mode != "front" and b in umma
            views = self.umma_views if on else views
            front = mode == "front"
            ns, gate, reload = front_options(tp, b) if front else (None, False, False)
            late = late_rms_mode(tp, b)  # passed only when set: default cache keys stay
            if mode == "publish" and b <= 8:
                late = 0  # TAE: the one-group Wave's late RMS is wave8 "late1"
            fp32 = fp32_rms_mode(tp, b) and mode == "publish" and not on
            extra = dict(late_rms=late) if late else {}
            if fp32:
                extra["fp32_rms"] = True
            wave8 = wave8_options(tp, b) if mode == "publish" else ()
            if wave8:
                extra["wave8"] = wave8
            if b in wlates:  # passed only when set: default cache keys stay
                if mode != "publish" or on or "direct" in wave8:
                    _rbal_line(f"K31MK_MOE_TAIL_WLATE declined ({mode} tail, umma {on}, not the one-shot mma.sync tail)", tp, b)
                else:
                    extra["wlate"] = wlates[b]
                    _rbal_line(f"K31MK_MOE_TAIL_WLATE mode={wlates[b]} engaged ({mode})", tp, b)
            if k19_tail(mode, b):
                extra["k19_tfence"] = True
            if b in spread:  # passed only when set: default cache keys stay
                if not twoshot:
                    raise ValueError(f"{SPREAD_ENV}: batch {b} is not a two-shot batch ({sorted(twoshots)})")
                extra["spread"] = True
                if (tp, b) not in _spread_engaged:
                    _spread_engaged.add((tp, b))
                    print(f"{SPREAD_ENV} engaged tp={tp} batch={b} pid={os.getpid()}", file=sys.stderr, flush=True)
            if b in polled0 and mode != "front":  # passed only when set: default cache keys stay
                extra["poll0"] = True
                _mfx_engaged("K31MK_MOE_TAIL_POLL0", tp, b)
            if b in owned_sq:  # passed only when set: default cache keys stay
                if mode == "front" or late != 1:
                    raise ValueError(f"K31MK_MOE_TAIL_SQ_OWNER: batch {b} needs the late-RMS WaveMoeTail ({mode}, late {late})")
                extra["sq_owner"] = True
                _mfx_engaged("K31MK_MOE_TAIL_SQ_OWNER", tp, b)
            if b in swapped:  # passed only when set: default cache keys stay
                if not (twoshot and on):
                    raise ValueError(f"K31MK_MOE_TAIL_TSWAP: batch {b} needs the two-shot tcgen05 WaveMoeTail ({mode}, umma {on})")
                extra["tswap"] = swapped[b]
                _mfx_engaged(f"K31MK_MOE_TAIL_TSWAP mode={swapped[b]}", tp, b)
            if b in k224s:  # passed only when set: default cache keys stay
                if not on:
                    raise ValueError(f"K31MK_MOE_TAIL_K224: batch {b} needs the tcgen05 WaveMoeTail ({mode}, umma {on})")
                extra["k224"] = True
                _mfx_engaged("K31MK_MOE_TAIL_K224", tp, b)
            if b in rbals:  # passed only when set: default cache keys stay
                if mode == "front":
                    _rbal_line(f"K31MK_MOE_TAIL_REARM_BAL declined ({mode} tail, no rearm_rows)", tp, b)
                else:
                    extra["rearm_bal"] = rbals[b]
                    _mfx_engaged(f"K31MK_MOE_TAIL_REARM_BAL mode={rbals[b]}", tp, b)
            code = compile_tail(
                tp,
                b,
                device,
                mode,
                early=b in early,
                umma=on,
                deadline_ns=deadline_ns,
                poll_ns=ns,
                gate=gate,
                reload=reload,
                **extra,
                **grown,
            )
            share = shares(tp, b, twoshot) and "direct" not in wave8  # TAE direct: no rows
            self.codes[b] = code, views, share, twoshot
            self.fp32[b] = fp32
        shared = any(shares(tp, b, b in twoshots) for b in batches)
        if shared and resident_clusters(device) < 7:
            raise RuntimeError("WaveMoeTail needs seven co-resident 16-CTA clusters")

    def run(self, box, expanded, weights, shared, code=None, batch=None):
        batch = batch or weights.shape[0]
        if code is None:
            code, views, share, twoshot = self.codes[batch]
        else:
            views, share, twoshot = self.views, 0, False
        args, tail = box.args, box.tail
        if twoshot:
            args, tail = box.twoshot_args, box.twoshot_tail
        if box.dra:  # DRA: this layer's output parity (multicast address = tail[3])
            tail = (*tail[:3], box.out_parity(self.out_parity)[1])
        rows = (flat(expanded), flat(weights), flat(shared), *tail, self.parity)
        rows += (Int32(box.rank), *self.scalars, Int32(self.wrap or batch))
        if share and self.fp32.get(batch):
            rows += (flat(box.reduced32),)
        elif share:
            rows += (flat(box.twoshot.gather if twoshot else box.reduced),)
        code(*args, *views, *rows, stream(expanded.device))


class PreparedTail:
    """The layer's MoE tail; placement = (model MoeTailBox, MoE ordinal, MoE layer
    count). At WaveMoeTail batches the experts publish into latent slot ``publish[b]``
    (multicast, parity, rank; at ``twoshots`` the two-shot push box) and the side
    stream pushes the shared-expert output (``push``)."""

    def __init__(self, op, norm, up, batches, placement, *, deadline_ns):
        tp, rank = op.contract.tp_size, op.rank
        local_up = up.narrow(0, rank * (7168 // tp), 7168 // tp)
        self.tail_box, ordinal, count = placement
        eps = op.contract.rms_eps
        m8 = [b for b in batches if published(tp, b)]
        self.twoshots = twoshot_batches(tp) & set(m8)
        args = (
            tp,
            local_up,
            norm,
            eps,
            ordinal,
            count,
            batches,
            self.twoshots,
            deadline_ns,
        )
        self.tail_weights = MoeTailWeights(*args, cap=self.tail_box.cap)
        self.peers, _, mc, _ = self.tail_box.tail
        self.ring = (self.tail_weights.parity, Int32(rank))
        # push_pdl: passed only when set, so default cache keys stay.
        pdl = lambda b: dict(push_pdl=True) if push_pdl_mode(tp, b) else {}
        self.pushers = {b: compile_push(b, norm.device.index, tp, **pdl(b)) for b in m8}
        self.publish = {b: (mc, *self.ring) for b in m8}
        slot = Int32(ordinal % 3 * tp + rank)
        self.publish.update({b: (*self.tail_box.push_box, slot) for b in self.twoshots})

    def push(self, shared):
        if shared.shape[0] in self.pushers:
            args = (to_cute(shared), self.peers, *self.ring, Int32(self.tail_box.cap))
            self.pushers[shared.shape[0]](*args, stream(shared.device))

    def run(self, expanded, weights, shared, out, *, consume):
        self.tail_weights.run(self.tail_box, expanded, weights, shared)
        if consume(out) is not out:
            raise RuntimeError("The tail output must be consumed from its mailbox ring")
        return out
