# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Options described as "ported" follow NVIDIA's Rubin (sm_107) specialized-decode kernels (the rubin package).
"""MoE front: the weight-stationary [router shard; down shard] GEMM publishes logits
and the latent shard into a multicast mailbox row; the routing kernel polls all 896
logits, ranks the top16 (sigmoid + bias, renormalized) and Lamport-copies the latent."""

import functools
import os
import sys

import cutlass as c
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32, Uint64
from cutlass.cute.runtime import make_fake_stream

from ..common import k19
from ..common.host import fake, stream, to_cute
from ..common.mailbox import poll, rearm, rearm_fence
from ..common.policy import batch_table
from ..common.primitives import (
    alloc,
    asm,
    bits_float,
    float_bits,
    launch,
    redux,
    store_words,
    view,
)
from ..gemm.projection import PreparedProjection
from ..gemm.projection import matrix_view, weight_map  # K31MK_MOE_FRONT_TMEM
from .experts import PLAN_BYTES, plan_chunk
from .experts import plan_image_chunk_f32  # K31MK_MOE_RUBIN8
from .projection import Fp32FrontGemm
from .routing import router_add, router_weight
from .routing import router_sigmoid as sigmoid
from .shared import compile_down, compile_shared, situ, weight_maps
from .shared import situ_hoisted  # hook: K31MK_EXP_SITU_LEAN
from .math import situ_lean_enabled  # hook: K31MK_EXP_SITU_LEAN
from ..common.primitives import div_rn  # hook: K31MK_EXP_SITU_LEAN
from .shared import SharedSitu, shared_sks_enabled  # hook: K31MK_EXP_SHARED_SKS
from .shared import shared_down_one64_enabled  # K31MK_MOE_SHARED_DOWN_ONE64
from .tail_runtime import CAP
from .experts_runtime import xplan_route_enabled  # hook: K31MK_EXP_XPLAN_ROUTE
from .experts_runtime import rubin8_enabled  # K31MK_MOE_RUBIN8
from .experts_runtime import route_poll8_enabled  # K31MK_MOE_ROUTE_POLL
from .experts_runtime import wide_fold_rows  # : K31MK_MOE_WIDE_FOLD
from .planned import fence_proxy_async_global  # hook (c-exp L22): K31MK_EXP_XPLAN_ROUTE

# EF2 (benchmark-only form, default off): at B <= 8
# with polled logits the route role issues its re-arm stores + re-arm fence after its last ids / weights store instead of between
# the H2 poll and the routing compute. "1": the same threads (tid < 224) re-arm their own cells; "w": warps 1-6 re-arm their own
# cells and warp 7 re-arms warp 0's (after the block barriers that follow every poll), so warp 0, which ranks and stores ids /
# weights, runs no fence. Same stores, scope and slots; still before exit. Default off; K31MK_MOE_EF2. Read when a kernel is
# compiled (a test may set ``EF2`` before it compiles, like ``mailbox.REARM_FENCE``).
EF2 = os.environ.get("K31MK_MOE_EF2", "")
assert EF2 in ("", "1", "w"), f"K31MK_MOE_EF2={EF2!r}"
# LN8 (benchmark-only form, default off): at B 3-8 with polled logits the route role runs the lean path that
# B > 8 runs (the ported ROUTE_LEAN_WIDE "3-8:1": candidates ranked by 8 threads each; same packed key, same ties, same
# publish). "1": no PDL wait in the route role (as at B > 8); "w": the route role executes griddepcontrol.wait after its
# ranking and before its ids / weights stores (K-25 order: polls and compute first, stores after the wait). With EF2 set, the
# lean route role's re-arm + fence move after its stores exactly as EF2 moves them on the non-lean path. Default off;
# K31MK_MOE_ROUTE_LEAN8. Read when a kernel is compiled (a test may set ``LN8`` before it compiles, like ``EF2``).
LN8 = os.environ.get("K31MK_MOE_ROUTE_LEAN8", "")
assert LN8 in ("", "1", "w"), f"K31MK_MOE_ROUTE_LEAN8={LN8!r}"
# CH (benchmark-only form, default off): batch table, e.g. "8:1". At those batches the polled copy role runs
# on two CTAs per token (the ported COPY_HALVES): CTA half h copies chunks [224 h, 224 h + 224) of the token's latent row, one
# 8-element chunk per thread (threads 0-223, whole warps for the plan's block shuffles), instead of one CTA whose threads
# 0-191 copy two chunks (tid, tid + 256) and 192-255 one. Same polls, stores, plan words and re-arms per chunk; each half
# runs the copy role's own waits / fences (XR's griddepcontrol.wait + fence.proxy.async, the re-arm fence). The grid grows
# from 2 B to 3 B CTAs. Default off; K31MK_MOE_COPY_HALVES. Read when a kernel is compiled (a test may set ``CH``, like ``LN8``).
CH = os.environ.get("K31MK_MOE_COPY_HALVES", "")
# P3D (benchmark-only forms, default off): cost controls and one real form for the
# routing kernel's post-front tail at TP16 B8 (pricing of the front-step gap). Each is read when a kernel is compiled (a
# test sets the attribute before it compiles, like ``LN8``); default off = the code above byte for byte.
#   P3NF / P3XW: the two unsafe cost controls (drop the re-arm fence / the copy role's griddepcontrol.wait) are not
#        selectable; both stay "" (fenced, waited).
#   P3SC (env K31MK_P3_COPY_SYNC) "1": a CTA barrier between the copy role's polls and its first store (the ported COPY_SYNC: plan shuffles not queued
#        behind sibling warps' volatile polls). Bitwise (same loads, stores, values); one copy CTA per token only.
#        P3E: with CH (copy halves) the barrier is named barrier 1 over the 224 copying threads (warps 0-6) of each half.
P3NF = ""
P3XW = ""
P3SC = os.environ.get("K31MK_P3_COPY_SYNC", "")
# DRR (K31MK_MOE_ROUTE_DEFER_REARM, default unset): batch table, rows 9-128 (polled lean path; 65-128 = the wide
# boots' 128-row box,  for MID), e.g. "9-64:1". The
# front box has two parities (MoE ordinal % 2); ordinal k's routing kernel re-arms ordinal k-1's cells + one re-arm fence at
# entry (under the hop wait) instead of its own cells after its poll; the last ordinal also re-arms its own.
# One stderr line 'K31MK_MOE_ROUTE_DEFER_REARM engaged batch=<b> pid=<p>' per deferred batch per process (boot checks).
_drr_engaged: set = set()


def route_defer_batches(tp):
    """K31MK_MOE_ROUTE_DEFER_REARM (default unset: none): batches (rows 9-128, value 0 / 1) whose routing kernel defers its
    re-arm to the next MoE ordinal's entry. Read when ``SharedFront`` is built."""
    table = batch_table(os.environ.get("K31MK_MOE_ROUTE_DEFER_REARM", ""))
    bad = {b: v for b, v in table.items() if not 9 <= b <= 128 or v not in (0, 1)}
    if bad:
        raise ValueError(f"K31MK_MOE_ROUTE_DEFER_REARM: rows 9-128 with value 0 or 1 only, got {bad}")
    return {batch for batch, on in table.items() if on}


class FrontParityView:
    """DRR (K31MK_MOE_ROUTE_DEFER_REARM): parity ``parity`` of a [2, rows, W] front box, with the attributes ``SharedFront`` reads
    (``buffer`` [1, rows, W], ``mc``, ``rank``) and ``full`` = the whole allocation (same form as the DRR test rig)."""

    def __init__(self, box, parity):
        self.full, self.rank = box.buffer, box.rank
        self.buffer = box.buffer[parity : parity + 1]
        self.mc = box.mc + parity * box.buffer[0].numel() * 2


@asm(Int32)
def shared_add(address, value):
    code = "atom.shared.add.u32 $0,[$1],$2;"
    return code, "=r,r,r,~{memory}", (address, Int32(value))


@cute.jit
def ordered(score):
    """Stock monotone key: NaNs become -1e30 and signed zeros share a key."""
    score = score if score == score else c.Float32(-1e30)
    score = c.Float32(0) if score == 0 else score
    bits = Uint32(float_bits(score))
    key = ~bits if (bits & Uint32(0x80000000)) != 0 else bits | Uint32(0x80000000)
    return key


class RouteSitu:
    def __init__(
        self, batch, shared, renorm, early, roles, polled, plan=False, deadline_ns=0,
        bounded=False, fp32=False, situ_lean=False, plan_tma=False, image=False, blocks=1, defer=0, keep_own=True,
        signal=False, early_ids=False, early_front=False,
    ):  # fmt: skip
        # Hook (K31MK_EXP_XPLAN_ROUTE, default off): the plan is read by the experts' TMAs (async proxy), so
        # the copy role fences generic <-> async around its plan stores (as EarlyPlanKernel).
        self.plan_tma = plan_tma and plan
        # ``image`` (K31MK_MOE_RUBIN8): the copy role writes the vendored RubinExpertsKernel's MXFP8 SMEM image of the FP32
        # latent (our quantization, the vendored layout: ``experts.plan_image_chunk_f32``) instead of our plan.
        assert not image or fp32, "image: FP32 latent words only"
        self.image = image and self.plan_tma
        # ``blocks`` (K31MK_MOE_WIDE_FOLD; default 1 = one 64-row plan): above 64 rows the copy role writes token t's
        # MXFP8 chunks at row t % 64 of the 64-row plan block t // 64, the wide engine's layout (``wide_plan.WidePlanKernel``,
        # ``rubin.moe.experts_wide`` row_sources), so the wide experts skip their plan kernel. FP32 latent words, no TMA / image.
        assert blocks == 1 or (plan and fp32 and not plan_tma and not image and 64 < batch <= 64 * blocks), "blocks: FP32 plan above 64 rows"
        self.blocks = blocks
        # Hook (K31MK_EXP_SITU_LEAN, default off): the situ role takes shared.situ's two parameter reciprocals
        # once per thread instead of once per element (the same div.rn of the same operands: bitwise equal).
        self.situ_lean = situ_lean
        # B>8: few registers, so routing CTAs co-reside with the front and the fused
        # shared expert, and polled logits skip the front's completion wait.
        self.fp32 = fp32
        self.lean = batch > 8
        self.lean8 = bool(LN8) and polled and 3 <= batch <= 8  # LN8: the lean path at B 3-8
        self.lean = self.lean or self.lean8
        self.batch, self.shared, self.experts, self.renorm = batch, shared, 896, renorm
        # early: release the persistent experts at entry (safe once down drains first);
        # polled: logits arrive as FP32 words in mailbox columns [3584, 5376) (Lamport;
        # [7168, 8960) after the FP32 latent, which every TP publishes since A4);
        # plan: the copy role also writes the latent as MXFP8 into the experts' plan.
        self.early, self.polled, self.plan = early, polled, plan
        self.deadline_ns = deadline_ns
        self.bounded = bounded and polled
        self.roles = tuple(r for r, on in zip(("route", "copy", "situ"), roles) if on)
        # CH: a second copy role ("copy2") = the token's second copy CTA; roles stay in CTA-range order.
        self.halves = bool(batch_table(CH).get(batch, 0)) and polled and "copy" in self.roles
        if self.halves:
            at = self.roles.index("copy") + 1
            self.roles = self.roles[:at] + ("copy2",) + self.roles[at:]
        self.chunk0 = tuple(224 if r == "copy2" else 0 for r in self.roles)  # CH: first chunk of each role's half
        # DRR: ``defer`` = element offset from this ordinal's box parity to the previous ordinal's (0 = served code);
        # ``keep_own``: the last MoE ordinal also re-arms its own cells (forward boundary: every cell armed).
        self.defer, self.keep_own = int(defer), bool(keep_own)
        assert not self.defer or (polled and batch > 8 and not self.lean8 and "situ" not in self.roles and not P3NF), "DRR: polled B 9-128"
        # ROUTE_POLL (K31MK_MOE_ROUTE_POLL, default off; the ported ``signal`` / ``early_ids`` / RELEASE_AFTER_FRONT forms):
        # the route role also writes the token's 16 ids into the vendored RubinExpertsKernel's route mailbox
        # (``signal``: ROUTE_WORDS int32 = 128 ids + the image count; each id word valid on its own and written exactly once per layer), and
        # each copy half, after a CTA barrier that follows its image stores, adds 1 to the count with a release (2 B halves: count 2 B - 1),
        # so the polled experts dedupe and stream their first weights before this kernel completes. ``early_ids`` (lean): the ranking thread
        # stores each chosen id (the bounded radix fallback stores in ``publish`` instead). ``early_front``: every CTA waits for the front GEMM
        # (griddepcontrol.wait), then releases the experts at entry (the ported "front" release; an entry release without the wait is unsafe with
        # the tail's reloading polls). Same ids, weights, latent and image bytes as without it.
        self.signal = signal
        assert not signal or (self.halves and self.image and self.lean), "signal: copy halves + RUBIN8 image + lean route only"
        self.early_ids = early_ids and signal and self.lean
        self.early_front = early_front
        if signal:
            from ..rubin.moe.experts_rubin import count_image  # lazy: only under K31MK_MOE_ROUTE_POLL

            self.count_image = count_image

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, len(self.roles) * self.batch, 256)

    @cute.kernel
    def kernel(self, args):
        logits, bias, ids, weights, mailbox, latent, gate_up, hidden = args[0:8]
        scale, beta, linear_beta = args[8:11]
        plan = args[11] if self.plan else None
        signal = args[11 + bool(self.plan)] if self.signal else None  # ROUTE_POLL: the experts' route mailbox
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        token, role = cta % self.batch, cta // self.batch
        if c.const_expr(self.early_front):  # ROUTE_POLL: release the experts once the front GEMM completed
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        elif c.const_expr(self.early):
            cute.arch.griddepcontrol_launch_dependents()
        for r in c.range_constexpr(len(self.roles)):
            if role == r:
                if c.const_expr(self.roles[r] == "route"):
                    self.route(logits, bias, ids, weights, scale, token, tid, mailbox, signal=signal)
                elif c.const_expr(self.roles[r] in ("copy", "copy2") and self.halves):
                    # CH: half h = 0 ("copy") / 1 ("copy2"): chunks 224 h + tid, one per thread of warps 0-6.
                    base = mailbox.iterator + token * mailbox.shape[2] + (self.chunk0[r] + tid) * (16 if self.fp32 else 8)
                    if tid < 224:
                        self.copy(base, latent, token, tid, 1, plan, self.chunk0[r])
                    if c.const_expr(self.signal):
                        # ROUTE_POLL: the half's image part is written (each copying thread's stores, then its proxy and re-arm fences),
                        # then its count: one release after the CTA barrier (cumulative over it), as the ported copy halves.
                        cute.arch.sync_threads()
                        if tid == 0:
                            self.count_image(signal.iterator + 128)
                elif c.const_expr(self.roles[r] == "copy"):
                    # Lamport: no dependency wait; the sentinel poll is the barrier.
                    base = mailbox.iterator + token * mailbox.shape[2] + tid * (16 if self.fp32 else 8)
                    if tid < 192:
                        self.copy(base, latent, token, tid, 2, plan, 0)
                    else:
                        self.copy(base, latent, token, tid, 1, plan, 0)
                else:
                    cute.arch.griddepcontrol_wait()
                    rounds = (self.shared + 255) // 256  # issue every load first
                    pair = cute.make_rmem_tensor((rounds, 2), c.Float32)
                    for i in c.range_constexpr(rounds):
                        if tid + 256 * i < self.shared:
                            pair[i, 0] = gate_up[token, tid + 256 * i]
                            pair[i, 1] = gate_up[token, self.shared + tid + 256 * i]
                    if c.const_expr(self.situ_lean):
                        inv_beta = div_rn(c.Float32(1), beta)
                        inv_lb = div_rn(c.Float32(1), linear_beta)
                        for i in c.range_constexpr(rounds):
                            if tid + 256 * i < self.shared:
                                gate, up = pair[i, 0], pair[i, 1]
                                value = situ_hoisted(gate, up, beta, linear_beta, inv_beta, inv_lb)
                                hidden[token, tid + 256 * i] = c.BFloat16(value)
                    else:
                        for i in c.range_constexpr(rounds):
                            if tid + 256 * i < self.shared:
                                gate, up = pair[i, 0], pair[i, 1]
                                value = situ(gate, up, beta, linear_beta)
                                hidden[token, tid + 256 * i] = c.BFloat16(value)
        # Late release: the persistent experts must not take SMs from the down GEMM.
        if c.const_expr(not self.early and not self.early_front):
            cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def select(self, key, threshold, buffers, signal=None, token=None):
        """B>8 candidates (keys at or above ``threshold``) append to ``dense`` at a
        per-warp offset from the shared count, then rank themselves: 8 threads per
        candidate (4 past 32 candidates) each count their share of the candidates above
        it (key, then lower expert). Ranks below 16 store the expert and its sigmoid."""
        dense, sig, chosen, picked, found, counts = buffers
        tid, _, _ = cute.arch.thread_idx()
        lane = tid % 32
        packed = cute.make_rmem_tensor(4, Uint64)
        ballots = cute.make_rmem_tensor(4, Uint32)
        total = Int32(0)
        for i in c.range_constexpr(4):  # packed 0: not a candidate
            expert = self.expert(tid, i)
            hit = (expert < self.experts) & (key[i] >= threshold)
            ballots[i] = cute.arch.vote_ballot_sync(hit)
            packed[i] = Uint64(0)
            if hit:
                packed[i] = (Uint64(key[i]) << 32) | Uint64(~Uint32(expert))
            total += cute.arch.popc(ballots[i])
        offset = Int32(0)
        if (lane == 0) & (total > 0):
            offset = shared_add(found.iterator, total)
        offset = cute.arch.shuffle_sync(offset, 0)
        below = (Uint32(1) << lane) - Uint32(1)
        for i in c.range_constexpr(4):
            if packed[i] != Uint64(0):
                dense[offset + cute.arch.popc(ballots[i] & below)] = packed[i]
            offset += cute.arch.popc(ballots[i])
        cute.arch.sync_threads()
        n = found[0]
        large = Int32(0)
        if c.const_expr(self.bounded):
            large = Int32(n > 64)
            if large != 0:
                self.radix(key, sig, chosen, picked, counts)
        if large == 0:
            if n <= 32:  # usually: 8 threads per candidate, 4 slots each
                part, index = tid % 8, tid // 8
                mine, rank = Uint64(0), Int32(0)
                if index < n:
                    mine = Uint64(dense[index])
                for k in c.range_constexpr(4):  # slots past n do not count
                    j = part + 8 * k
                    rank += Int32((j < n) & (Uint64(dense[j]) > mine))
                for shift in c.range_constexpr(3):
                    rank += cute.arch.shuffle_sync_bfly(rank, 1 << shift)
                if (part == 0) & (index < n) & (rank < 16):
                    expert = Int32(~Uint32(mine))
                    chosen[rank], picked[rank] = expert, sig[expert]
                    if c.const_expr(self.early_ids):  # ROUTE_POLL
                        store_words(signal.iterator + (token * 16 + rank), (expert,), scope="gpu")
            else:
                part = tid % 4
                for first in c.range(0, n, 64, unroll=1):
                    index = first + tid // 4
                    mine, rank = Uint64(0), Int32(0)
                    if index < n:
                        mine = Uint64(dense[index])
                    if n <= 64:  # every load issues at once; slots past n do not count
                        for k in c.range_constexpr(16):
                            j = part + 4 * k
                            rank += Int32((j < n) & (Uint64(dense[j]) > mine))
                    else:
                        for j in c.range(part, n, 4, unroll=2):
                            rank += Int32(Uint64(dense[j]) > mine)
                    rank += cute.arch.shuffle_sync_bfly(rank, 1)
                    rank += cute.arch.shuffle_sync_bfly(rank, 2)
                    if (part == 0) & (index < n) & (rank < 16):
                        expert = Int32(~Uint32(mine))
                        chosen[rank], picked[rank] = expert, sig[expert]
                        if c.const_expr(self.early_ids):  # ROUTE_POLL
                            store_words(signal.iterator + (token * 16 + rank), (expert,), scope="gpu")
        cute.arch.sync_threads()

    @cute.jit
    def radix(self, key, sig, chosen, picked, counts):
        """Bound the large-candidate fallback: exact 32-bit radix selection.

        Polled rows give each thread four consecutive expert ids. Equal scores
        select lower ids; selected rows publish in expert-id order. The normal
        <=64-candidate path retains its existing ranking and rounding.
        """
        tid, _, _ = cute.arch.thread_idx()
        warp, lane = tid // 32, tid % 32
        prefix, mask, left = Uint32(0), Uint32(0), Int32(16)
        for j in c.range(32, unroll=1):
            bit, total = Uint32(1) << (31 - j), Int32(0)
            for i in c.range_constexpr(4):
                active = (4 * tid + i < 896) & ((key[i] & mask) == prefix)
                votes = cute.arch.vote_ballot_sync(active & ((key[i] & bit) != 0))
                total += Int32(cute.arch.popc(votes))
            if lane == 0:
                counts[warp] = total
            cute.arch.sync_threads()
            total = Int32(0)
            for w in c.range_constexpr(8):
                total += counts[w]
            if total >= left:
                prefix |= bit
            else:
                left -= total
            mask |= bit
            cute.arch.sync_threads()

        equal, selected = (cute.make_rmem_tensor(4, Int32) for _ in range(2))
        local = Int32(0)
        for i in c.range_constexpr(4):
            equal[i] = Int32((4 * tid + i < 896) & (key[i] == prefix))
            local += equal[i]
        scan = local
        for shift in c.range_constexpr(5):
            other = cute.arch.shuffle_sync_up(scan, 1 << shift)
            if lane >= 1 << shift:
                scan += other
        if lane == 31:
            counts[warp] = scan
        cute.arch.sync_threads()
        before = scan - local
        for w in c.range_constexpr(7):
            if warp > w:
                before += counts[w]
        total = Int32(0)
        for i in c.range_constexpr(4):
            keep = (4 * tid + i < 896) & (
                (key[i] > prefix) | ((equal[i] != 0) & (before < left))
            )
            selected[i] = Int32(keep)
            before += equal[i]
            total += selected[i]
        # All readers retire before the same warp-count storage is reused.
        cute.arch.sync_threads()
        scan = total
        for shift in c.range_constexpr(5):
            other = cute.arch.shuffle_sync_up(scan, 1 << shift)
            if lane >= 1 << shift:
                scan += other
        if lane == 31:
            counts[warp] = scan
        cute.arch.sync_threads()
        slot = scan - total
        for w in c.range_constexpr(7):
            if warp > w:
                slot += counts[w]
        for i in c.range_constexpr(4):
            if selected[i] != 0:
                expert = 4 * tid + i
                chosen[slot], picked[slot] = expert, sig[expert]
            slot += selected[i]
        cute.arch.sync_threads()

    @cute.jit
    def publish(self, chosen, picked, ids, weights, scale, token, signal=None, found=None):
        """Warp 0: the 16 ids by rank and their sigmoids renormalized (lane r holds
        rank r; the butterfly sums in the same order for every lane)."""
        tid, _, _ = cute.arch.thread_idx()
        lane = tid % 32
        if tid < 32:
            value = c.Float32(0)
            if lane < 16:
                ids[token, lane] = chosen[lane]
                if c.const_expr(signal is not None and not self.early_ids):  # ROUTE_POLL: ids into the experts' route mailbox
                    store_words(signal.iterator + (token * 16 + lane), (chosen[lane],), scope="gpu")
                elif c.const_expr(signal is not None and self.bounded):  # ROUTE_POLL: the radix fallback (n > 64) stored no early ids
                    if found[0] > 64:
                        store_words(signal.iterator + (token * 16 + lane), (chosen[lane],), scope="gpu")
                value = picked[lane]
            total = value
            for shift in c.range_constexpr(4):
                total += cute.arch.shuffle_sync_bfly(total, 1 << shift)
            if lane < 16:
                value = router_weight(value, total, scale, renorm=self.renorm)
                weights[token, lane] = c.BFloat16(value)

    def expert(self, tid, i):
        return tid * 4 + i if self.polled else tid + i * 256

    @cute.jit
    def copy(
        self, base, latent, token, tid, count: c.Constexpr[int], plan=None, chunk0: c.Constexpr[int] = 0,
    ):  # fmt: skip
        stride: c.Constexpr = 4096 if self.fp32 else 2048
        if c.const_expr(self.defer != 0):  # DRR: the previous ordinal's cells, under the hop wait
            for i in c.range_constexpr(count):
                rearm(base + self.defer + i * stride)
                if c.const_expr(self.fp32):
                    rearm(base + self.defer + i * stride + 8)
            rearm_fence()
        got = poll(base, stride, count, self.deadline_ns)
        hi = poll(base + 8, stride, count, self.deadline_ns) if self.fp32 else got
        if c.const_expr(P3SC == "1" and not self.halves):  # P3D SC: every thread's polls land before any store
            cute.arch.sync_threads()
        if c.const_expr(P3SC == "1" and self.halves):  # P3E CHSC: the half's 224 copying threads (warps 0-6) only
            cute.arch.barrier(barrier_id=1, number_of_threads=224)
        if c.const_expr(plan is not None and self.plan_tma):  # hook: the last call's TMA reads first
            # The copy role otherwise never waits: without this its plan stores are ordered after the previous layer's
            # experts TMA reads only by the Lamport data chain. With it the chain is W2's.
            if c.const_expr(P3XW != "1"):  # P3D XW: an unsafe cost control drops this wait
                cute.arch.griddepcontrol_wait()
            fence_proxy_async_global()
        for i in c.range_constexpr(count):
            dst = latent.iterator + token * latent.stride[0] + (chunk0 + tid + 256 * i) * 8  # chunk0: CH half
            words = [got[i, k] for k in range(4)]
            if c.const_expr(self.fp32):
                words += [hi[i, k] for k in range(4)]
                store_words(dst + 4, words[4:])
            store_words(dst, words[:4])
            if c.const_expr(self.defer == 0 or self.keep_own):  # DRR: else the next ordinal re-arms these at its entry
                rearm(base + i * stride)
                if c.const_expr(self.fp32):
                    rearm(base + i * stride + 8)
            if c.const_expr(plan is not None and self.image):  # chunk0: a multiple of 16, so the scale words stay whole
                plan_image_chunk_f32(plan, token, chunk0 + tid + 256 * i, words)
            elif c.const_expr(plan is not None and self.blocks > 1):  # : row token % 64 of plan block token // 64
                block = view(plan.iterator, token // 64 * PLAN_BYTES, c.Uint8, PLAN_BYTES)
                plan_chunk(block, token % 64, chunk0 + tid + 256 * i, words, fp32=self.fp32)
            elif c.const_expr(plan is not None):
                plan_chunk(plan, token, chunk0 + tid + 256 * i, words, fp32=self.fp32)
        if c.const_expr(plan is not None and self.plan_tma):  # hook: plan stores before TMA reads
            fence_proxy_async_global()
        if c.const_expr("c" not in P3NF and (self.defer == 0 or self.keep_own)):  # P3D NF "c"; DRR: fence moved to entry
            rearm_fence()

    @cute.jit
    def route(self, logits, bias, ids, weights, scale, token, tid, mailbox, signal=None):
        """Exact top16 by threads [0, 256): candidates at or above the 16th largest
        lane-group maximum are compacted and ranked by an unrolled 64-way compare."""
        warp, lane = tid // 32, tid % 32
        smem = c.utils.SmemAllocator()
        km = alloc(smem, Uint32, (8, 32))
        cand, dense = (alloc(smem, Uint64, shape) for shape in ((8, 128), 64))
        sig = alloc(smem, c.Float32, 1024)
        counts = alloc(smem, Int32, 8)
        cutoff = alloc(smem, Uint32, 1)
        chosen = alloc(smem, Int32, 16)
        if c.const_expr(self.lean or self.bounded):
            found, picked = alloc(smem, Int32, 1), alloc(smem, c.Float32, 16)
        # Thread t owns experts t + 256 i, or 4 t + i when polling 16-byte chunks.
        offsets = cute.make_rmem_tensor(4, c.Float32)
        for i in c.range_constexpr(4):  # bias is a weight: load before the wait
            offsets[i] = c.Float32(0)
            if self.expert(tid, i) < self.experts:
                offsets[i] = bias[self.expert(tid, i)]
        raw = cute.make_rmem_tensor(4, c.Float32)
        key, top = cute.make_rmem_tensor(4, Uint32), Uint32(0)
        for i in c.range_constexpr(4):
            key[i] = Uint32(0)
        if c.const_expr(self.polled):
            if tid < 224:
                cell = mailbox.iterator + token * mailbox.shape[2] + (7168 if self.fp32 else 3584) + tid * 8
                if c.const_expr(self.defer != 0):  # DRR: the previous ordinal's cell, under the hop wait
                    rearm(cell + self.defer)
                    rearm_fence()
                got = poll(cell, 0, 1, self.deadline_ns)
                if c.const_expr(not (bool(EF2) and (not self.lean or self.lean8)) and (self.defer == 0 or self.keep_own)):
                    rearm(cell)
                    if c.const_expr("r" not in P3NF):  # P3D NF "r": an unsafe cost control drops the route role's re-arm fence
                        rearm_fence()
                for i in c.range_constexpr(4):
                    raw[i] = bits_float(got[0, i])
                if c.const_expr(self.lean):  # all four experts: one block, no branches
                    scores = [sigmoid(raw[i]) for i in range(4)]
                    for i in c.range_constexpr(4):
                        sig[4 * tid + i] = scores[i]
                        key[i] = ordered(router_add(scores[i], offsets[i]))
            if c.const_expr(not self.lean):
                cute.arch.griddepcontrol_wait()
        else:
            cute.arch.griddepcontrol_wait()
            for i in c.range_constexpr(4):
                if self.expert(tid, i) < self.experts:
                    raw[i] = logits[token, self.expert(tid, i)]
        for i in c.range_constexpr(4):
            expert = self.expert(tid, i)
            if c.const_expr(not (self.lean and self.polled)):  # noqa: SIM102
                if expert < self.experts:
                    score = sigmoid(raw[i])
                    sig[expert] = score
                    key[i] = ordered(router_add(score, offsets[i]))
            top = max(top, key[i])
        km[warp, lane] = top
        if c.const_expr(self.lean):  # noqa: SIM102
            if tid == 0:
                found[0] = Int32(0)
        cute.arch.sync_threads()
        if warp == 0:
            group = km[0, lane]
            for w in c.range_constexpr(1, 8):
                group = max(group, km[w, lane])
            km[0, lane] = group
            cute.arch.sync_warp()
            rank = Int32(0)
            for j in c.range_constexpr(32):
                rank += Int32(km[0, j] > group)
            low = redux(group if rank < 16 else Uint32(0xFFFFFFFF))
            if lane == 0:
                cutoff[0] = low
        cute.arch.sync_threads()
        if c.const_expr(self.lean):
            dense = view(cand.iterator, 0, Uint64, (1024,))
            self.select(key, cutoff[0], (dense, sig, chosen, picked, found, counts), signal, token)
            if c.const_expr(self.lean8 and LN8 == "w"):  # LN8 "w": stores after the PDL wait (K-25 order)
                cute.arch.griddepcontrol_wait()
            self.publish(chosen, picked, ids, weights, scale, token, signal, found)
            if c.const_expr(self.lean8 and bool(EF2)):  # EF2 on the LN8 path: the route role's re-arm, after its stores
                if (tid >= (32 if EF2 == "w" else 0)) & (tid < (256 if EF2 == "w" else 224)):
                    src = tid - 224 * Int32(tid >= 224)  # "w": warp 7 takes warp 0's cells
                    rearm(mailbox.iterator + token * mailbox.shape[2] + (7168 if self.fp32 else 3584) + src * 8)
                    if c.const_expr("r" not in P3NF):  # P3D NF "r": an unsafe cost control drops the route role's re-arm fence
                        rearm_fence()
            return
        threshold, offset = cutoff[0], Int32(0)
        for i in c.range_constexpr(4):
            expert = self.expert(tid, i)
            hit = (expert < self.experts) & (key[i] >= threshold)
            ballot = cute.arch.vote_ballot_sync(hit)
            if hit:
                below = cute.arch.popc(ballot & ((Uint32(1) << lane) - Uint32(1)))
                packed = (Uint64(key[i]) << 32) | Uint64(~Uint32(expert))
                cand[warp, offset + below] = packed
            offset += cute.arch.popc(ballot)
        if lane == 0:
            counts[warp] = offset
        cute.arch.sync_threads()
        large = Int32(0)
        if c.const_expr(self.bounded):
            candidates = Int32(0)
            for w in c.range_constexpr(8):
                candidates += counts[w]
            large = Int32(candidates > 64)
            if large != 0:
                self.radix(key, sig, chosen, picked, counts)
                self.publish(chosen, picked, ids, weights, scale, token)
        if large == 0:
            if warp == 0:
                n = Int32(0)
                for v in c.range_constexpr(8):
                    n += counts[v]
                for half in c.range_constexpr(2):  # compact the regions densely
                    index = lane + 32 * half
                    rest, region = Int32(index), Int32(0)
                    for v in c.range_constexpr(8):
                        if (region == v) & (rest >= counts[v]):
                            rest -= counts[v]
                            region += 1
                    value = Uint64(0)
                    if index < n:
                        value = cand[region, rest]
                    dense[index] = value
                cute.arch.sync_warp()
                if n <= 64:
                    for half in c.range_constexpr(2):
                        mine, rank = dense[lane + 32 * half], Int32(0)
                        for j in c.range_constexpr(64):
                            rank += Int32(dense[j] > mine)
                        if (lane + 32 * half < n) & (rank < 16):
                            chosen[rank] = Int32(~Uint32(mine))
                else:  # rare: rank every candidate against all regions
                    for q in range(lane, n, 32):
                        rest, region = Int32(q), Int32(0)
                        for v in c.range_constexpr(8):
                            if (region == v) & (rest >= counts[v]):
                                rest -= counts[v]
                                region += 1
                        mine, rank = cand[region, rest], Int32(0)
                        for v in c.range_constexpr(8):
                            for j in range(counts[v]):
                                rank += Int32(cand[v, j] > mine)
                        if rank < 16:
                            chosen[rank] = Int32(~Uint32(mine))
                cute.arch.sync_warp()
                expert, value = chosen[lane % 16], c.Float32(0)
                if lane < 16:
                    ids[token, lane] = expert
                    value = sig[expert]
                total = value
                for shift in c.range_constexpr(4):
                    total += cute.arch.shuffle_sync_bfly(total, 1 << shift)
                if lane < 16:
                    value = router_weight(value, total, scale, renorm=self.renorm)
                    weights[token, lane] = c.BFloat16(value)
        if c.const_expr(bool(EF2) and self.polled and not self.lean):  # EF2: the route role's re-arm, after its stores
            if (tid >= (32 if EF2 == "w" else 0)) & (tid < (256 if EF2 == "w" else 224)):
                src = tid - 224 * Int32(tid >= 224)  # "w": warp 7 takes warp 0's cells
                rearm(mailbox.iterator + token * mailbox.shape[2] + (7168 if self.fp32 else 3584) + src * 8)
                if c.const_expr("r" not in P3NF):  # P3D NF "r": an unsafe cost control drops the route role's re-arm fence
                    rearm_fence()


# Front GEMM geometry "ks.kb.tg.rt.ch" per batch, by packed rows (TP8 560, TP16 288),
# from GB300 sweeps. ks: cluster K-split; kb: 32-wide
# K blocks per warp, so a row tile takes 224 / (ks * kb) warps (at most 16); tg: tokens
# per group (0: the single-pass path); rt: 16-row tiles per CTA; ch: accumulator chains
# per 8-token tile (0: 4 / tiles per group).
# 65-128 (boots whose front box holds more than 64 rows): not swept. TP8 keeps its 33-64 geometry; TP16 takes rt 2 instead
# of 3, the largest rt whose 128-row x slice still fits front_geometry's SMEM budget (rt 3 would drop to the single pass).
FRONT = {
    560: "1-4:4.14.0.1.0,5-32:16.7.16.5.2,33-64:16.14.16.5.2,65-128:16.14.16.5.2",
    288: "1-4:4.14.0.1.0,5-64:16.7.16.3.2,65-128:16.7.16.2.2",
}
# Batches whose shared expert is the fused kernel (``SharedUmma``), not the split-K
# chain; of those, split batches run gate/up + SiTU beside the front and the down rows
# (``SharedDown``) on SPLIT SMs beside the routed experts (0: fused).
FUSED, SPLIT = "9-64:1", "9-12:6,13-24:4,25-64:2"
#  (wide decode tier): K31MK_MOE_FRONT_TMEM rows 65-128 run the shared expert's down rows as ``SharedDown``
# beside the wide experts on this many SMs (the source's FrontRows form: 4 SMs at 65-96 rows; 97-128 not swept). Read
# only for those rows; K31MK_MOE_SPLIT_SMS rows 65-128 override it.
WIDE_DOWN = "65-128:4"
# Per-TP overrides of SPLIT entries, (q = 1, verify steps): on real prompts (GB300
# SPEED-Bench 8k mixed) SharedDown on 2 SMs (~123 us at M32) outlasts the experts (TP8
# q=1 B32: 3 SMs, -5.4% step). Verify steps (q > 1) share experts among a request's q
# tokens: TP8 DL7 M16/32/64 -5.9/-12.9/-1.1%, TP16 M32 -5.9%.
SPLITS = {8: ("25-32:3", "13-16:6,25-32:4,33-64:3"), 16: ("", "25-32:3")}


def split_table(tp, verify):
    """``SharedDown`` SMs per batch: SPLIT, then the per-TP entries of SPLITS, then
    ``K31MK_MOE_SPLIT_SMS`` (default unset: no change; a batch table such as "33-64:6",
    rows 9-128 (65-128: K31MK_MOE_FRONT_TMEM wide rows only, ``WIDE_DOWN``), 0-32 SMs, 0 = fused). A faster routed engine at these rows (e.g.
    ``K31MK_MOE_RUBIN64``) can finish well before a 2-SM ``SharedDown``, so the tail waits
    for the down rows; more down CTAs take their SMs from the routed grid, which
    ``PreparedMoE`` caps at the device's SMs minus this table's entry."""
    extra = os.environ.get("K31MK_MOE_SPLIT_SMS", "")
    bad = {b: v for b, v in batch_table(extra).items() if not 9 <= b <= 128 or type(v) is not int or not 0 <= v <= 32}
    if bad:
        raise ValueError(f"K31MK_MOE_SPLIT_SMS: rows 9-128 with 0-32 SMs only, got {bad}")
    return batch_table(f"{SPLIT},{SPLITS[tp][verify]},{WIDE_DOWN},{extra}")


# Split batches whose front GEMM rows join the gate/up kernel as extra M-tiles (one
# kernel instead of two that slow each other ~2x side by side).
MERGED = {8: "41-64:1", 16: "33-64:1"}


def merged_table(tp):
    """``MERGED[tp]``, then ``K31MK_MOE_FRONT_MERGED`` (default unset: no change; a batch table
    such as "33-64:0", rows 9-64, value 0 only). A 0 entry runs that batch's front GEMM as its
    own ``Fp32FrontGemm`` (K-split ``front_ks``) and the shared gate/up as the separate gate
    kernel, as at the split batches outside ``MERGED``. The two forms sum the router logits
    and latent columns in different FP32 orders, so a 0 entry is not bitwise equal to the
    merged form."""
    extra = os.environ.get("K31MK_MOE_FRONT_MERGED", "")
    bad = {b: v for b, v in batch_table(extra).items() if not 9 <= b <= 64 or v != 0}
    if bad:
        raise ValueError(f"K31MK_MOE_FRONT_MERGED: rows 9-64 with value 0 only, got {bad}")
    return batch_table(f"{MERGED[tp]},{extra}")


# FTM (K31MK_MOE_FRONT_TMEM; default unset = no change): the ported MERGED_WIDE {16: "16-32:1"} +
# FRONT_TMEM_WIDE {288: "16-32:1,64:1"}: at the listed split batches the front rows and the shared expert's gate/up run as
# one ``MoeFrontTmem64`` grid (front_tmem.py: 128-row weight tiles staged in tensor memory before the PDL wait, split K
# 8 in clusters of 8, DSMEM reduce-scatter; h for SharedDown as the merged form). Rows 65-128 (wide
# boots: a front box of >= that many rows): the same grid over ceil(rows / 64) token tiles (the source's ``MoeFrontRows``
# role), the down rows as ``SharedDown`` on ``WIDE_DOWN`` SMs beside the wide experts. Judge its speed at TP16 only (a
# 1-GPU proxy gave the wrong sign). Numerics: the 8
# partials sum in rank order, an FP32 reorder of the split / merged forms (not bitwise; KL / AL arm). A batch table,
# rows 9-64, value 0 or 1; TP16 (288 packed rows) on SM107 only; a listed batch must be a split batch (``split_table``).
# Read when SharedFront is built.
def front_tmem_table(tp, n, device):
    """K31MK_MOE_FRONT_TMEM batches (value 1) for this front, or an empty set."""
    spec = os.environ.get("K31MK_MOE_FRONT_TMEM", "")
    table = batch_table(spec)
    bad = {b: v for b, v in table.items() if not 9 <= b <= 128 or v not in (0, 1)}
    if bad:
        raise ValueError(f"K31MK_MOE_FRONT_TMEM: rows 9-128 with value 0 or 1 only, got {bad}")
    on = {b for b, v in table.items() if v == 1}
    if not on:
        return set()
    if tp != 16 or n != 288:
        raise ValueError(f"K31MK_MOE_FRONT_TMEM={spec!r}: TP16 (288 packed front rows) only, got TP{tp} / {n} rows")
    if torch.cuda.get_device_capability(device) != (10, 7):
        raise ValueError(f"K31MK_MOE_FRONT_TMEM={spec!r}: SM107 (VR200) only")
    return on


def front_tmem_promote():
    """``K31MK_MOE_FRONT_TMEM_PROMOTE`` (default unset / 0 = the ported single TMEM accumulator): with
    ``K31MK_MOE_FRONT_TMEM``, every this many K tiles (4 MMAs each, 1-14) the merged front's MMAs switch between two TMEM
    accumulators and the epilogue warps add each finished one into FP32 registers. Long tensor-core accumulation chains lose
    accuracy (measured), so the 56-MMA chains of the 8-way split sit measurably further from FP64 than the served
    ``Fp32FrontGemm``; 2 bounds a chain at 8 MMAs (CPU model: close to the served error). An FP32 reorder: not bitwise."""
    value = os.environ.get("K31MK_MOE_FRONT_TMEM_PROMOTE", "") or "0"
    promote = int(value)
    if not 0 <= promote <= 14:
        raise ValueError(f"K31MK_MOE_FRONT_TMEM_PROMOTE={value!r}: 0 (off) or 1-14 K tiles")
    return promote


# MBOX (K31MK_MOE_FRONT_MBOX; default unset = served code): at the listed FTM batches (a batch table, rows 9-64,
# value 0 or 1, each also a K31MK_MOE_FRONT_TMEM batch) the front's split-K reduce mailboxes get their own 32 KB of shared
# memory and the cluster barrier phase that guarded the aliased A / x slots is dropped (front_tmem / splitk_tmem64 mbox=1).
# Same partials, same rank-order FP32 sum: bitwise to the served FTM. Measured faster per front call at
# 16 / 24 / 32 rows. One stderr line 'K31MK_MOE_FRONT_MBOX engaged batch=<b> pid=<p>' per engaged batch per process.
_mbox_engaged: set = set()


def front_tmem_mbox(ftm, used):
    """K31MK_MOE_FRONT_MBOX batches (value 1) among ``used`` (this front's FTM batches); every listed batch must be in
    ``ftm`` (the K31MK_MOE_FRONT_TMEM table's batches)."""
    spec = os.environ.get("K31MK_MOE_FRONT_MBOX", "")
    table = batch_table(spec)
    bad = {b: v for b, v in table.items() if not 9 <= b <= 128 or v not in (0, 1)}  # 65-128:  nxmb r1 (bitwise, faster)
    if bad:
        raise ValueError(f"K31MK_MOE_FRONT_MBOX: rows 9-128 with value 0 or 1 only, got {bad}")
    on = {b for b, v in table.items() if v == 1}
    if on - set(ftm):
        raise ValueError(f"K31MK_MOE_FRONT_MBOX={spec!r}: batches {sorted(on - set(ftm))} are not K31MK_MOE_FRONT_TMEM batches")
    on &= set(used)
    for b in sorted(on - _mbox_engaged):
        print(f"K31MK_MOE_FRONT_MBOX engaged batch={b} pid={os.getpid()}", file=sys.stderr, flush=True)
        _mbox_engaged.add(b)
    return on


def b_mbox_conflict(mbox, batches):
    """F-L: MBOX's own mailbox region plus the 2-parity x box is untested together; refuse the overlap."""
    return bool(set(mbox) & set(batches))


@functools.cache
def compile_front_tmem(split, valid, shard, hidden, device, fenced_trigger=2, promote=0, rows=64, mbox=0, n32=None, lamx=0):
    """``MoeFrontTmem64`` (K31MK_MOE_FRONT_TMEM): the packed front rows (router ``valid``
    padded to ``split``, then the latent shard) and ``hidden`` gate/up pairs; ``promote``:
    ``front_tmem_promote``. ``rows``: the front box's rows (passed only above 64, as for
    ``compile_publish``: the kernel addresses a token's box row by token, so rows only size the box).
    ``mbox``: ``front_tmem_mbox``; ``n32``: ``front_tmem.front_n32_for``
    (None = the scalar K31MK_MOE_FRONT_N32, read in ``MoeFrontTmem64``; passed on only when set, so a tree whose
    front_tmem.py has no ``n32`` keyword still builds the scalar / unset forms)."""
    from cutlass.cute.runtime import make_fake_tensor

    from .front_tmem import MoeFrontTmem64

    if lamx == 1:  # F-L (K31MK_MOE_FRONT_LAMX): x polled from the consumer's Lamport box; + (xbox, parity)
        from .front_tmem import MoeFrontTmem64Lamx as MoeFrontTmem64  # noqa: N814
    elif lamx == 2:  # F-L measurement form with the canary spin
        from .front_tmem import MoeFrontTmem64LamxC as MoeFrontTmem64  # noqa: N814
    elif lamx == 3:  # F-L per-tile form (each K tile written as soon as it is published)
        from .front_tmem import MoeFrontTmem64LamxD as MoeFrontTmem64  # noqa: N814
    kernel = MoeFrontTmem64(
        valid=valid, split_rows=split, shard=shard, hidden=hidden, fenced_trigger=fenced_trigger, promote=promote, mbox=mbox,
        **({"n32": n32} if n32 is not None else {}),
    )

    def matrix():
        shape = tuple(cute.sym_int() for _ in range(3))
        stride = (cute.sym_int(64), 1, 0)
        return make_fake_tensor(c.BFloat16, shape, stride, assumed_align=16)

    width = 7168 + 2 * 896  # FP32 latent words + FP32 logit words (``compile_publish``)
    args: tuple = (matrix(), matrix(), fake((64, hidden)), fake((1, rows, width)))
    args += (c.Int64(0), Int32(0), Int32(0), c.Float32(1), c.Float32(1))
    args += (c.Int64(0), c.Int64(0)) + ((c.Int64(0), Int32(0)) if lamx else ()) + (make_fake_stream(),)
    with torch.accelerator.device_index(device):
        arch = "sm_{}{}a".format(*torch.cuda.get_device_capability(device))
        return cute.experimental.compile(kernel, *args, options=f"--gpu-arch={arch}")


# Shared-expert split-K tactics (M tile, N tile, split, stages) by width (TP8 768, TP16
# 384): gate_up, then down. 65-128 (front boxes above 64 rows; split-K chain, FUSED stops at 64): the 33-64 tactics, not
# swept (more N tiles of the same kernel).
SHARED_TACTICS = {
    768: (
        "1-8:128.8.4.6,9-16:64.16.4.10,17-32:64.32.4.8,33-64:64.64.4.5,65-128:64.64.4.5",
        "1-8:128.8.1.6,9-16:64.16.1.6,17-32:64.32.1.6,33-64:64.64.1.6,65-128:64.64.1.6",
    ),
    384: (
        "1-16:64.8.4.8,17-32:64.16.4.10,33-64:64.32.4.8,65-128:64.32.4.8",
        "1-8:128.8.1.3,9-16:64.16.1.3,17-32:64.32.1.3,33-64:64.64.1.3,65-128:64.64.1.3",
    ),
}
# The routing kernel writes the experts' MXFP8 plan only up to this many rows: the plan buffer is one 64-row block
# (experts.PLAN_BYTES). Above it the experts engine writes its own plans, except at K31MK_MOE_WIDE_FOLD rows (
# M1(a)), where the routing kernel writes the wide engine's 64-row plan blocks into its plan buffer (``wide_plan``).
PLAN_ROWS = 64


@functools.cache
def front_table(n):
    table = batch_table(FRONT.get(n, ""))
    # FW (benchmark-only form, default off): K31MK_FRONT_WIDE overrides TP16 (288-row) entries, e.g. the ported
    # FRONT_WIDE "3-8:8.7.16.3.2" (an 8-CTA K-split over 48 CTAs beside the o-projection; a stated REORDER: the K-split
    # sum order changes). Default off. Read when the table is first built (a test clears ``front_table``'s cache per form).
    if n == 288:
        table.update(batch_table(os.environ.get("K31MK_FRONT_WIDE", "")))
    return table


def front_ks(batch, n):
    return front_table(n).get(batch, (4,))[0]


def front_geometry(batch, n, ks, kb=None, tg=None, rt=None, ch=None):
    """(rt, kq, staged, tg, ch) of the front GEMM over ``n`` rows. Unset arguments come
    from the table when ``ks`` is its K-split, else from the single pass; token groups
    whose x slice does not fit in SMEM fall back to the single pass."""
    entry = front_table(n).get(batch, ())
    table = entry[1:] if entry[:1] == (ks,) else (14, 0, 1, 0)
    kb, tg, rt, ch = (t if v is None else v for v, t in zip((kb, tg, rt, ch), table))
    kq = min(16, 224 // (ks * kb))
    kb, nt = 224 // (ks * kq), -(-batch // 8)
    staged = max(kb - 8, 0) if kq == 16 else 0
    used = (
        rt * kq * (nt * 544 + max(staged, 1) * 1024) + -(-nt * rt // ks) * ks * kq * 512
    )
    if nt * 8 * (7168 // ks * 2 + 64) > 220 * 1024 - used:
        tg = 0
    return rt, kq, staged, tg, ch


@functools.cache
def compile_route(
    batch,
    shared,
    renorm,
    roles,
    device,
    polled=False,
    plan=False,
    deadline_ns=0,
    bounded=False,
    fp32=False,
    situ_lean=False,
    release_early=False,
    plan_tma=False,
    image=False,
    rows=64,
    blocks=1,
    defer=0,
    keep_own=True,
    signal=False,
):
    """``release_early``: release the dependents at entry at every batch (SE:
    the SiTU kernel's dependent is the shared down GEMM, which waits for it anyway).
    Optional kernel arguments after ``plan``, in this fixed order (``RouteSitu.kernel`` reads them at these slots):
    ``signal`` (K31MK_MOE_ROUTE_POLL): ``RouteSitu`` signal + early_ids + early_front, one more argument (the
    experts' route mailbox, ROUTE_WORDS int32), at args[11 + plan].
    ``rows``: the front box's rows (callers pass it only above 64, so 64-row cache keys stay).
    ``blocks`` (K31MK_MOE_WIDE_FOLD): 64-row plan blocks the copy role writes (``RouteSitu`` blocks;
    callers pass it only above 64 rows)."""
    kernel = RouteSitu(
        batch,
        shared,
        renorm,
        batch <= 2 or release_early,
        roles,
        polled,
        plan,
        deadline_ns,
        bounded,
        fp32,
        situ_lean,
        **({"plan_tma": True} if plan_tma else {}),  # hook
        **({"image": True} if image else {}),  # K31MK_MOE_RUBIN8
        **({"blocks": blocks} if blocks > 1 else {}),  # : K31MK_MOE_WIDE_FOLD
        **({"defer": defer, "keep_own": keep_own} if defer else {}),  # DRR (default objects unchanged)
        **({"signal": True, "early_ids": True, "early_front": True} if signal else {}),  # ROUTE_POLL
    )
    f32 = c.Float32
    args: tuple = (fake((batch, 896), None, f32), fake((896,), None, f32))
    width = (7168 if fp32 else 3584) + 2 * 896
    dtype = f32 if fp32 else c.BFloat16
    args += (fake((batch, 16), None, c.Int32), fake((batch, 16)), fake((1, rows, width)))
    args += (fake((batch, 3584), None, dtype), fake((batch, 2 * shared), None, f32))
    args += (fake((batch, shared)), *(f32(v) for v in (1, 4, 25)))
    args += (fake((blocks * PLAN_BYTES,), None, c.Uint8),) * plan
    if signal:  # ROUTE_POLL: the route mailbox (lazy import: only under the knob)
        from ..rubin.moe.experts_rubin import ROUTE_WORDS

        args += (fake((ROUTE_WORDS,), None, c.Int32),)
    args += (make_fake_stream(),)
    with torch.accelerator.device_index(device):
        return cute.compile(kernel, *args)


@functools.cache
def compile_publish(n, split, valid, batch, device, ks, geometry=None, early_weights=0, tma_fence=0, rows=64):
    """Front GEMM that only publishes; routing then runs in the polled kernel. The latent
    shard leaves as FP32 words at every TP (stock's FP32 front).
    ``early_weights``: WsGemm.early_weights (F1; 0 = shipped). ``tma_fence``:
    WsGemm.tma_fence (K-16 consumer fence; 0 = shipped). ``rows``: the front box's rows
    (passed only above 64; the publish addresses token * width, so rows only size the box)."""
    rt, kq, staged, tg, ch = geometry or front_geometry(batch, n, ks)
    kernel = Fp32FrontGemm(n, 7168, batch, rt, kq, staged, split, valid, ks, tg, ch)
    kernel.early_weights = early_weights
    kernel.tma_fence = tma_fence
    if k19.fence("front_gemm"):  # K-19 trigger fence (default off; set only when on: default objects unchanged)
        kernel.k19_fence = True
    width = 7168 + 2 * 896
    args: tuple = (fake((batch, 7168)), fake((n, 7168)), fake((1, rows, width)))
    args += (c.Int64(0), c.Int32(0), c.Int32(0), fake((batch, 896), None, c.Float32))
    with torch.accelerator.device_index(device):
        return cute.compile(kernel, *args, make_fake_stream())


class SharedFront:
    def __init__(
        self, moe, packed, box, storage, batches, beta, linear_beta, plan=None,
        counters=None, q=1, deadline_ns=0, wide_plan=None, drr=None, lamx=None, route_box=None,
    ):  # fmt: skip
        """``route_box`` (ROUTE_POLL): the B8 RUBIN8 engine's workspace, whose route mailbox the routing kernel signals.
        ``plan``: the experts' plan buffer, written by the routing kernel at 8 < B <= PLAN_ROWS;
        ``wide_plan``: the wide engine's plan buffer (``experts_runtime.scratch``'s "rubin_wide" plan), written by the
        routing kernel as 64-row blocks at K31MK_MOE_WIDE_FOLD rows (needed only under that knob);
        ``counters``: the fused shared expert's model-owned barrier counters; ``q``: the
        deployment's verify length (its SPLITS entries). ``box``: its rows bound the batches."""
        rows = box.buffer.shape[-2]  # the publish writes one box row per token
        if max(batches) > rows:
            raise ValueError(f"SharedFront: batch {max(batches)} exceeds the front box's {rows} rows")
        experts = moe.shared_experts
        gate_up, down = experts.gate_up_proj.weight, experts.down_proj.weight
        self.shared = shared = gate_up.shape[0] // 2
        tp = 6144 // shared
        shard, valid = 3584 // tp, 896 // tp
        self.device, self.packed, self.storage = packed.device, to_cute(packed), storage
        gu, dn = map(batch_table, SHARED_TACTICS[shared])
        self.gate_up = PreparedProjection(
            gate_up, {b: gu[b] for b in batches}, output_dtype=torch.float32
        ).prepare()
        self.down = PreparedProjection(down, {b: dn[b] for b in batches}).prepare()
        # Hook (K31MK_EXP_SHARED_SKS, default off): B<=8 gate/up with SiTU in the split-K epilogue (h out).
        self.sks = None
        if shared_sks_enabled():
            small = {b: gu[b] for b in batches if b <= 8 and gu[b][0] == 64}
            self.sks = SharedSitu(gate_up, small, beta, linear_beta) if small else None
        scale, renorm = moe.routed_scaling_factor, moe.moe_renormalize
        self.scalars = tuple(map(c.Float32, (scale, beta, linear_beta)))
        self.box: tuple = (to_cute(box.buffer), c.Int64(box.mc))
        self.box += (c.Int32(box.rank * valid), c.Int32(box.rank * shard))
        bias = to_cute(moe.gate.e_score_correction_bias)
        dev, n = packed.device.index, packed.shape[0]
        # Default off, read once here: F1 front weight loads above the PDL wait
        # (batch table of WsGemm.early_weights modes) and SE, the SiTU kernel releasing the
        # shared down GEMM at entry (split-K shared expert batches only).
        early_weights = batch_table(os.environ.get("K31MK_FRONT_EARLY_WEIGHTS", ""))
        situ_early = batch_table(os.environ.get("K31MK_MOE_SITU_EARLY", ""))
        # K-16 (default off; batch table, e.g. "1-64:1"): consumer-side fence.proxy.async.global
        # in the front GEMM's x bulk-copy warp after the PDL wait.
        tma_fence = batch_table(os.environ.get("K31MK_FRONT_TMA_FENCE", ""))
        tmem = front_tmem_table(tp, n, packed.device) & set(batches)
        # : K31MK_MOE_FRONT_TMEM rows above 64 (wide boots) take the split shared expert (FTM gate/up + SiTU,
        # SharedDown beside the experts on WIDE_DOWN SMs), as the 9-64-row FTM batches do; FUSED itself stops at 64.
        wide_tmem = {b for b in tmem if b > 64}
        if wide_tmem and counters is None:
            raise ValueError(f"K31MK_MOE_FRONT_TMEM rows {sorted(wide_tmem)}: the split shared expert needs counters")
        fused = batch_table(FUSED)
        fused = {b for b in batches if b > 8 and counters is not None and fused.get(b)} | wide_tmem
        verify = q > 1
        split = split_table(tp, verify)
        self.split = {b: split[b] for b in fused if split.get(b)}  # down SMs
        if fused:  # TMA maps and counters of the fused kernel
            self.maps = weight_maps(gate_up, down)
            self.fused = to_cute(self.maps, 64), to_cute(counters)
        # Default off, read once here: K31MK_MOE_SHARED_DOWN_ONE64 -- the split SharedDown
        # reads its own maps blob whose down map has [64, 64] boxes (one TMA per 8 KB W stage).
        self.down_maps = None
        if self.split and shared_down_one64_enabled():
            self.down_maps_t = weight_maps(gate_up, down, down_rows=64)
            self.down_maps = to_cute(self.down_maps_t, 64)
        merged = merged_table(tp)
        self.merged = {b for b in self.split if merged.get(b)}
        # K31MK_MOE_FRONT_TMEM (default unset: none): these split batches run the front + gate/up as one MoeFrontTmem64
        # grid (``front_tmem``, built above); like MERGED batches, h then comes from ``route`` and the down rows run
        # beside the experts.
        if tmem - set(self.split):
            raise ValueError(f"K31MK_MOE_FRONT_TMEM: batches {sorted(tmem - set(self.split))} are not split batches")
        self.merged |= tmem
        self.front_tmem: dict[int, tuple] = {}
        self.lamx_batches, self.lamx_args = set(), ()  # F-L (K31MK_MOE_FRONT_LAMX): set below at its FTM batches
        if tmem:
            # Kept alive here: the maps are device tensors the kernel reads by address.
            self.tmem_maps = (weight_map(packed, 128), weight_map(gate_up, 64))
            self.tmem_views = matrix_view(packed)
            maps = tuple(c.Int64(m.data_ptr()) for m in self.tmem_maps)
            mbox = front_tmem_mbox(front_tmem_table(tp, n, packed.device), tmem)
            # Per-batch N32 (batch-table K31MK_MOE_FRONT_N32, e.g. the c=8 table 16-32:3; None = scalar). front_n32_for
            # lives in the N32 front_tmem.py; imported only for the table form.
            n32_for = None
            if ":" in os.environ.get("K31MK_MOE_FRONT_N32", ""):
                from .front_tmem import front_n32_for as n32_for
            # F-L (K31MK_MOE_FRONT_LAMX; default unset = served code): ``lamx`` = (the [2, 32, 7168] Lamport x box
            # tensor, this MoE ordinal's parity); its listed FTM batches compile MoeFrontTmem64Lamx and pass (box address, parity).
            if os.environ.get("K31MK_MOE_FRONT_LAMX", ""):
                from .front_tmem import front_lamx_for

                self.lamx_modes = {b: front_lamx_for(b) for b in tmem}
                self.lamx_batches = {b for b, mode in self.lamx_modes.items() if mode}
                if self.lamx_batches and (lamx is None or b_mbox_conflict(mbox, self.lamx_batches)):
                    raise ValueError("K31MK_MOE_FRONT_LAMX: SharedFront needs lamx=(box, parity) and no MBOX at its batches")
                if self.lamx_batches:
                    xbox, parity = lamx
                    if tuple(xbox.shape) != (2, 32, 7168) or xbox.element_size() != 2 or parity not in (0, 1):
                        raise ValueError(f"K31MK_MOE_FRONT_LAMX: lamx box must be [2, 32, 7168] 16-bit, parity 0 / 1, got {tuple(xbox.shape)} / {parity}")
                    self.lamx_box = xbox  # kept alive: the kernel reads it by address
                    self.lamx_args = (c.Int64(xbox.data_ptr()), Int32(parity))
            for b in sorted(tmem):
                n32 = n32_for(b) if n32_for is not None else None
                code = compile_front_tmem(
                    n - shard, valid, shard, gate_up.shape[0] // 2, dev, promote=front_tmem_promote(), mbox=int(b in mbox),
                    **({"rows": rows} if rows != 64 else {}),  # wide boots: the box is longer; 64-row boots keep their keys
                    **({"n32": n32} if n32 is not None else {}),  # batch-table N32 only: scalar / unset boots keep their keys
                    **({"lamx": self.lamx_modes[b]} if b in self.lamx_batches else {}),  # F-L only: other batches keep their keys
                )
                self.front_tmem[b] = code, maps
        if self.merged:  # gate/up, down and front maps; unused RS arguments
            self.merged_maps = weight_maps(gate_up, down, front=packed)
            # Kept alive here: a CuTe view does not hold its tensor.
            self.unused = torch.zeros(tp, dtype=torch.int64, device=packed.device)
            self.merged_args = to_cute(self.merged_maps, 64), to_cute(self.unused)
            geometry = (n, n - shard, valid, box.buffer.shape[-1])
        #  (K31MK_MOE_WIDE_FOLD, default unset): at these wide rows the routing kernel's copy role writes the
        # wide engine's 64-row plan blocks into ``wide_plan`` (``RouteSitu`` blocks), so the wide experts run no plan kernel.
        fold = {b for b, v in wide_fold_rows().items() if v and b in batches}
        if fold and wide_plan is None:
            raise ValueError(f"K31MK_MOE_WIDE_FOLD rows {sorted(fold)}: SharedFront needs wide_plan (the wide engine's plan)")
        # DRR (K31MK_MOE_ROUTE_DEFER_REARM): ``drr`` = (parity, last) of this MoE ordinal; ``box`` is that parity's view with
        # ``full`` = the [2, 64, W] allocation (ordinal k uses parity k % 2).
        self.drr: dict[int, dict] = {}
        deferred = route_defer_batches(tp) & {b for b in batches if b > 8}
        if deferred:
            full = getattr(box, "full", None)
            if drr is None or full is None or full.shape[0] != 2:
                raise ValueError("K31MK_MOE_ROUTE_DEFER_REARM: SharedFront needs drr=(parity, last) and a 2-parity front box view")
            parity, last = drr
            if box.buffer.data_ptr() != full[parity].data_ptr() or box.buffer.shape[0] != 1:
                raise ValueError(f"K31MK_MOE_ROUTE_DEFER_REARM: box is not parity {parity} of its allocation")
            delta = (1 - 2 * int(parity)) * full[0].numel()
            self.drr = {b: {"defer": delta, "keep_own": bool(last)} for b in sorted(deferred)}
            for b in sorted(deferred - _drr_engaged):
                print(f"K31MK_MOE_ROUTE_DEFER_REARM engaged batch={b} pid={os.getpid()}", file=sys.stderr, flush=True)
                _drr_engaged.add(b)
        self.plans: dict[int, tuple] = {}
        self.route_words = {}  # ROUTE_POLL: the route mailbox views (kept alive: a CuTe view does not hold its tensor)
        grown = {"rows": rows} if rows != 64 else {}  # a box above 64 rows; 64-row boots keep their cache keys
        for b in batches:
            views = tuple(t[:b] for t in storage)
            logits, gate, ids, weights, h, _, latent = map(to_cute, views)
            front = compile_publish(
                n, n - shard, valid, b, dev, front_ks(b, n), None, early_weights.get(b, 0),
                tma_fence.get(b, 0), **grown,
            )
            planned = plan is not None and 8 < b <= PLAN_ROWS
            blocks = -(-b // 64) if b in fold else 1  # : the wide engine's 64-row plan blocks
            planned = planned or b in fold
            # Hook (K31MK_EXP_XPLAN_ROUTE with K31MK_EXP_XPLAN, default off): the copy role also writes the
            # experts' MXFP8 plan at B<=8, so the TMA-fed experts skip their plan kernel (``planned`` in prepare).
            xroute = plan is not None and b == 8 and tp == 16 and xplan_route_enabled()  # B8 only, as the experts
            planned = planned or xroute
            roles = (True, True, False)
            poll8 = xroute and rubin8_enabled() and route_poll8_enabled()  # ROUTE_POLL (B8 only, as xroute)
            if poll8:
                missing = [k for k, ok in (("K31MK_MOE_COPY_HALVES 8", bool(batch_table(CH).get(b, 0))), ("K31MK_MOE_ROUTE_LEAN8", bool(LN8)),
                                           ("route_box", route_box is not None)) if not ok]  # fmt: skip
                if missing:
                    raise ValueError(f"K31MK_MOE_ROUTE_POLL needs {missing}")
                from ..rubin.moe.experts_runtime import route_mailbox

                self.route_words[b] = route_mailbox(route_box)
            route = compile_route(
                b,
                shared,
                renorm,
                roles,
                dev,
                True,
                planned,
                deadline_ns,
                bounded=q > 1 and b in (8, 16),
                fp32=True,  # FP32 latent words at every TP (A4)
                **({"plan_tma": True} if xroute else {}),  # hook
                **({"image": True} if xroute and rubin8_enabled() else {}),  # K31MK_MOE_RUBIN8
                **grown,
                **({"blocks": blocks} if blocks > 1 else {}),  # : K31MK_MOE_WIDE_FOLD
                **self.drr.get(b, {}),  # DRR (K31MK_MOE_ROUTE_DEFER_REARM)
                **({"signal": True} if poll8 else {}),  # ROUTE_POLL
            )
            situ = compile_route(
                b,
                shared,
                renorm,
                (False, False, True),
                dev,
                deadline_ns=deadline_ns,
                fp32=True,
                **({"situ_lean": True} if situ_lean_enabled() else {}),  # hook
                release_early=bool(situ_early.get(b, 0)),
                **grown,
            )
            args = (logits, bias, ids, weights, self.box[0], latent, gate, h)
            target = plan
            if b in fold:  # exactly the blocks the copy role writes (its compile took blocks * PLAN_BYTES)
                if wide_plan.numel() < blocks * PLAN_BYTES:
                    raise ValueError(f"B{b}: wide_plan holds {wide_plan.numel()} bytes < {blocks} plan blocks")
                target = wide_plan[: blocks * PLAN_BYTES]
            extra = (to_cute(target),) if planned else ()
            extra += (to_cute(self.route_words[b]),) if poll8 else ()  # ROUTE_POLL: after the plan (RouteSitu's args[12])
            code = None
            one64 = {"one64": True} if self.down_maps is not None else {}  # K31MK_MOE_SHARED_DOWN_ONE64
            if b in self.front_tmem:  # K31MK_MOE_FRONT_TMEM: no gate kernel (``route`` leaves h)
                code = None, compile_down(tp, b, dev, self.split[b], **one64)
            elif b in self.merged:
                gate = compile_shared(tp, b, dev, True, geometry)
                code = gate, compile_down(tp, b, dev, self.split[b], **one64)
            elif b in self.split:
                gate = compile_shared(tp, b, dev, gate=True)
                code = gate, compile_down(tp, b, dev, self.split[b], **one64)
            elif b in fused:
                code = compile_shared(tp, b, dev)
            self.plans[b] = views, front, route, situ, args + self.scalars, extra, code

    def route(self, x, between=None):
        """Router logits + published down shard, then top16 and the latent copy (and
        at B>8 the experts' MXFP8 plan). Merged batches publish them from the gate/up
        kernel, which also leaves h for ``shared_expert``'s down rows. ``between``: work
        launched on this stream after the publish and before the routing kernel polls
        the other ranks' shards (it runs while the latent hop is in flight)."""
        views, front, route, _, args, extra, code = self.plans[x.shape[0]]
        s = stream(self.device)
        if x.shape[0] in self.front_tmem:  # K31MK_MOE_FRONT_TMEM
            tmem, maps = self.front_tmem[x.shape[0]]
            work = (self.tmem_views, matrix_view(x), args[7], *self.box)
            lamx = self.lamx_args if x.shape[0] in self.lamx_batches else ()  # F-L: (box address, parity)
            tmem(*work, *self.scalars[1:], *maps, *lamx, s)
        elif x.shape[0] in self.merged:
            maps, unused = self.merged_args
            work = (to_cute(x), maps, args[7], self.fused[1], unused)
            work += (Int32(0), Int32(0), Int32(CAP), *self.scalars[1:], *self.box[1:])  # unused RS args (gate=True)
            code[0](*work, s)
        else:
            front(to_cute(x), self.packed, *self.box, args[0], s)
        if between is not None:
            between()
        route(*args, *extra, s)
        return views[2], views[3], views[6]

    def merged_batch(self, batch):
        return batch in self.merged

    def planned(self, batch):
        return bool(self.plans[batch][5])

    def shared_expert(self, x, tail):
        """The shared expert's rows into the tail's RS mailboxes (``tail.push``); the
        fused kernel peer-stores them itself and leaves the returned rows unwritten.
        Returns (rows, the split's down launch to run beside the experts, or None)."""
        views, _, _, situ, args, _, code = self.plans[x.shape[0]]
        s = stream(self.device)
        if code is not None:
            maps, counters = self.fused
            mailbox = (tail.peers, *tail.ring, Int32(tail.tail_box.cap))  # RS row stride
            front = (to_cute(x), maps, args[7], counters, *mailbox, *self.scalars[1:])
            if x.shape[0] not in self.split:
                code(*front, s)
                return views[5], None
            gate, down = code
            if x.shape[0] not in self.merged:  # merged: h comes from ``route``
                gate(*front, s)
            dmaps = maps if self.down_maps is None else self.down_maps  # K31MK_MOE_SHARED_DOWN_ONE64
            return views[5], lambda: down(args[7], dmaps, *mailbox, stream(self.device))
        if self.sks is not None and x.shape[0] in self.sks.batches:  # hook: h straight from the GEMM
            self.sks.run(x, views[1], views[4])
        else:
            self.gate_up.run(x, views[1])
            situ(*args, s)
        shared = self.down.run(views[4], views[5])
        tail.push(shared)
        return shared, None
