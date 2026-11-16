# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Options described as "ported" follow NVIDIA's Rubin (sm_107) specialized-decode kernels (the rubin package).
"""Cluster-split KDA recurrence; an optional fifth warp projects F_B and multicasts the
decays while the main warps run the convolution and query/key normalization."""

import os
from functools import partial

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass.cute.arch import store_async_dsmem

from ..common.primitives import (
    alloc,
    asm,
    at,
    bits_float,
    copy_g2s,
    dot,
    float_bits,
    fma,
    launch,
    load_f32x4,
    prefetch_bytes,
    prefetch_l2,
    load_words,
    mma_into,
    named_barrier,
    push_float,
    sigmoid,
    smem_words,
    store_f32x4,
    try_wait,
    store_shared_words,
    unpack_bf16,
    scatter_reduce,
    shared_address,
    wait,
    warp_sum,
    zeros,
)

load_bf16x4 = partial(load_words, count=2, cache_policy="nc", word_type="u32")
load_weight_bf16x8 = partial(load_words, count=4, cache_policy="nc", word_type="u32")
load_activation_bf16x4 = partial(load_words, count=2, word_type="u32", mutable=True)
load_activation_bf16x8 = partial(load_words, count=4, word_type="u32", mutable=True)
load_f32x2 = partial(load_words, count=2, result_type=Float32, word_type="f32")
# BF16 recurrent state (``bf16_state``): four state values = two words, the same weak ld.global.cg as
# load_f32x4 of the FP32 state; each BF16 half widens to FP32 exactly (unpack_bf16).
load_state_bf16x4 = partial(load_words, count=2, word_type="u32")
load_staged = partial(
    load_words, count=4, shared=True, mutable=True, result_type=Float32, word_type="f32"
)

@asm((Float32, Int32), pure=True)
def reciprocal_guess(denominator):
    """``sigmoid_reciprocal``'s refined fast path (IEEE mode) and whether it holds."""
    code = "{ .reg .f32 r,error,neg_error; .reg .b32 bits; .reg .pred regular; "
    code += "mov.b32 bits,$2; add.u32 bits,bits,0x01800000; "
    code += "and.b32 bits,bits,0x7f800000; setp.gt.u32 regular,bits,0x01ffffff; "
    code += "rcp.approx.ftz.f32 r,$2; fma.rn.f32 error,$2,r,0fbf800000; "
    code += "neg.f32 neg_error,error; fma.rn.f32 $0,r,neg_error,r; "
    return code + "selp.u32 $1,1,0,regular; }", "=f,=r,f", (denominator,)


@asm(Float32, pure=True)
def reciprocal_rn(value):
    return "div.rn.f32 $0,0f3f800000,$1;", "=f,f", (value,)


@cute.jit
def scatter_reduce_lanes(
    values, lane, count: cutlass.Constexpr[int], lanes: cutlass.Constexpr[int]
):
    """Reduce ``count`` values over aligned groups of ``lanes`` lanes (butterflies of
    offset lanes / 2 .. 1). Each lane ends with ``count // lanes`` totals in
    ``values[0 ..]``, the totals of value indices ``selected + i``."""
    selected = Int32(0)
    for stage in cutlass.range_constexpr(lanes.bit_length() - 1):
        offset = (lanes // 2) >> stage
        half = count >> (stage + 1)
        upper = (lane & offset) != 0
        for i in cutlass.range_constexpr(half):
            send = values[i] if upper else values[i + half]
            keep = values[i + half] if upper else values[i]
            values[i] = keep + cute.arch.shuffle_sync_bfly(send, offset)
        if upper:
            selected += half
    return selected


@cute.jit
def sigmoids(values):
    """``sigmoid(v, False)`` of every value, bit for bit, with all fast paths ahead
    of the division fallbacks' branches (verify tokens get ILP)."""
    count: cutlass.Constexpr = len(values)
    den = [Float32(1) + cute.math.exp(-Float32(v), fastmath=True) for v in values]
    result, regular = zeros(count, Float32), zeros(count, Int32)
    irregular = Int32(0)
    for i in cutlass.range_constexpr(count):
        result[i], regular[i] = reciprocal_guess(den[i])
        irregular |= regular[i] ^ 1
    if irregular != 0:  # rare: one branch for the whole batch
        for i in cutlass.range_constexpr(count):
            if regular[i] == 0:
                result[i] = reciprocal_rn(den[i])
    return result


def advance(base, n):
    """``base + n``, or ``base`` itself for a static zero ``n`` (keeps q = 1 code)."""
    return base + n if n else base


@asm((Float32, Float32), pure=True)
def mul2_ftz(a, b, *, ftz=True):
    """Lanewise a * b of FP32 pairs (packed; ``ftz`` as the scalar path)."""
    code = "{ .reg .b64 a,b,d; mov.b64 a,{$2,$3}; mov.b64 b,{$4,$5}; "
    code += f"mul.rn{'.ftz' if ftz else ''}.f32x2 d,a,b; mov.b64 {{$0,$1}},d; }}"
    return code, "=f,=f,f,f,f,f", (*a, *b)


@asm((Float32, Float32), pure=True)
def fma2_ftz(a, b, c, *, ftz=True):
    """Lanewise a * b + c of FP32 pairs (packed fma.rn; ``ftz`` as the scalar path)."""
    code = "{ .reg .b64 a,b,c,d; mov.b64 a,{$2,$3}; mov.b64 b,{$4,$5}; "
    code += f"mov.b64 c,{{$6,$7}}; fma.rn{'.ftz' if ftz else ''}.f32x2 d,a,b,c; "
    code += "mov.b64 {$0,$1},d; }"
    return code, "=f,=f,f,f,f,f,f,f", (*a, *b, *c)


@asm()
def store_bf16_when(pointer, value, when):
    """Opt "dl": one BF16 global store of an FP32 value rounded RN (as ``BFloat16()``), predicated (no branch)."""
    code = "{ .reg .pred w; .reg .b16 h; setp.ne.b32 w,$2,0; cvt.rn.bf16.f32 h,$1; @w st.global.b16 [$0],h; }"
    return code, "l,f,r,~{memory}", (pointer, Float32(value), Int32(when))


@asm()
def store_f32_when(pointer, value, when):
    """Opt "dl": one FP32 global store, predicated (no branch)."""
    code = "{ .reg .pred w; setp.ne.b32 w,$2,0; @w st.global.f32 [$0],$1; }"
    return code, "l,f,r,~{memory}", (pointer, Float32(value), Int32(when))


@asm()
def fenced_trigger():
    """K-19 pricing knob (K31MK_KDA_TRIGGER_FENCE=1, default off, read when the
    kernel object is built): ``fence.acq_rel.gpu`` then the PDL trigger in one asm block, at every KdaSplit
    trigger that follows its ``griddepcontrol.wait`` (the B1 entry trigger precedes the wait: unchanged)."""
    return "fence.acq_rel.gpu;\n\tgriddepcontrol.launch_dependents;", "~{memory}", ()


@asm()
def arm_cluster(barrier, size):
    """K-23 fix (always on): arm a barrier that peer CTAs complete with remote
    ``st.async`` at CLUSTER scope. The DSL's default arm is ``.release.cta``, which is not morally strong with a peer's
    complete_tx (PTX 8.7), so
    when the peer's bytes land before the arm the waiter's ``try_wait.acquire.cluster`` reads the arm and the peer's
    data is not ordered for it (8.9.2 observation order, 8.9.4). ``.relaxed.cluster`` restores the chain (litmus-tested:
    forbidden) and publishes nothing itself (no fence). PTX ISA 8.6+, sm_90+."""
    code = "{ .reg .b64 st; mbarrier.arrive.expect_tx.relaxed.cluster.shared::cta.b64 st, [$0], $1; }"
    return code, "r,r,~{memory}", (barrier, Int32(size))


# One-CTA clusters (cluster == 1, the "wlf" one-wave form): a weak st.async requires %cluster_nctarank > 1 (PTX ISA,
# st.async: otherwise undefined), so there every sender stores into its own CTA's shared memory (``put_float``) and
# then arrives once with release semantics (``arrive_release_cta``); ``setup`` inits each barrier with its number of
# senders instead of arming it with bytes. The waiters keep ``try_wait.acquire`` (same CTA: morally strong).
@asm()
def put_float(pointer, value, when=None):
    """Own-CTA shared store of one FP32 ``value`` (``when``: predicated on it)."""
    guard = "" if when is None else ".reg .pred w; setp.ne.b32 w,$2,0; @w "
    code = "{ " + guard + "st.shared::cta.b32 [$0],$1; }"
    constraints = "r,f" + (",r" if when is not None else "")
    operands = (pointer, Float32(value)) + (() if when is None else (Int32(when),))
    return code, constraints + ",~{memory}", operands


@asm()
def arrive_release_cta(barrier):
    """One arrival on an own-CTA barrier with .release.cta: this thread's earlier shared stores are ordered before
    the phase completion a same-CTA ``try_wait.acquire`` observes."""
    code = "{ .reg .b64 st; mbarrier.arrive.release.cta.shared::cta.b64 st, [$0]; }"
    return code, "r,~{memory}", (barrier,)


@asm()
def subset_barrier(barrier_id, threads):
    """Opt "tsp": named barrier for a subset of the CTA's warps in the non-.aligned form
    (``barrier.sync``), which PTX allows when not every thread of the CTA reaches it."""
    return "barrier.sync $0, $1;", "r,r,~{memory}", (Int32(barrier_id), Int32(threads))


# Opt "tsp" engages up to this many requests: in-chain it measured faster per verify at
# Breq 1 / 2 (with "dmc") but slower at Breq 8 (bitwise either way); Breq 3-7 unmeasured, so they keep the base.
# Opt "cg"'s gate-warp rendezvous (MAIN + 32 threads) has its own named barrier; tsp's phase 1 uses
# barriers 2 / 3 (128 main threads each), barrier 1 is the main warps', 0 is __syncthreads.
CG_BARRIER = 4
TSP_MAX_REQUESTS = 2
# "dmc" and the "chunk" family (chunk, cg, cs, csp, gb) engage only up to
# the same request count. dmc measured slower at Breq 8, and chunk slower at Breq 4,
# and at Breq 4 (4-CTA clusters on SM107) cg / cs cannot build (they asserted). Larger batches run the base PTX.
DMC_MAX_REQUESTS = CHUNK_MAX_REQUESTS = TSP_MAX_REQUESTS
# Opts "b3w" / "b3c" (default off): Breq 3 Q8 runs 8-CTA clusters (verify.verify_schedule; 18 clusters, one
# wave) with "tsp" + "dmc" ("b3w", bitwise vs the served 4-CTA base) or also the "chunk" family ("b3c", the Breq 2 served form; not
# bitwise). 1 GPU, TP16 heads 6, cold graph, same-GPU ABBA x 2 GPUs, FP32-reference gates std / wd / md:
# b3w and b3c measured faster than served at Breq 3. Breq 4 keeps its caps: its 8-CTA clusters exceed the
# one-wave cover and measured slower there.
B3_MAX_REQUESTS = 3
# Opt "tsp4" (default off, bitwise by construction): tsp's token split on the 4-CTA clusters of 32 value
# rows that SM107 runs at Breq 4 (and at Breq 5 / 6 under K31MK_KDA_VERIFY_B57=1): CTA p forms tokens 2p, 2p + 1 once for the
# cluster (``prologue_tsp4``) instead of every CTA forming all eight tokens' q / k. Engages up to this many requests.
TSP4_MAX_REQUESTS = 6


@asm()
def store_bf16x4(pointer, values, *, evict=False, when=None):
    """Four FP32 state values rounded to nearest-even BF16 (the state's one rounding, at its
    store) in one 8-byte store; ``evict``: L2 evict-first; ``when``: predicated, no branch."""
    words = "{ .reg .b32 w0, w1; .reg .b64 pol; "
    words += "cvt.rn.bf16x2.f32 w0,$2,$1; cvt.rn.bf16x2.f32 w1,$4,$3; "
    if evict:
        words += "createpolicy.fractional.L2::evict_first.b64 pol,1.0; "
        store = "st.global.L2::cache_hint.v2.b32 [$0],{w0,w1},pol;"
    else:
        store = "st.global.v2.b32 [$0],{w0,w1};"
    if when is None:
        return words + store + " }", "l,f,f,f,f", (pointer, *values)
    code = words + ".reg .pred p; setp.ne.b32 p,$5,0; @p " + store + " }"
    return code, "l,f,f,f,f,r", (pointer, *values, Int32(when))


@asm()
def store_evict_f32x4(pointer, values):
    code = "{ .reg .b64 pol; createpolicy.fractional.L2::evict_first.b64 pol,1.0; "
    code += "st.global.L2::cache_hint.v4.f32 [$0],{$1,$2,$3,$4},pol; }"
    return code, "l,f,f,f,f", (pointer, *values)


class KdaSplit:
    """128 main threads and C value partitions. Conv state [slot, 3 * heads * 128, 3]
    and recurrent state [slot, head, value, key] keep their strides; slots <= 0 are
    padding (never written) and active rows own distinct slots. Weights must be ready
    at launch; each run waits for prior cache restoration and slot publication.

    ``schedule`` = (early, lanes per state row, minimum CTAs per SM, CTAs per head);
    zeros keep the batch defaults. Two CTAs per head give each CTA 64 state rows and
    each gate lane two decays. Early issues every load the predecessor does not
    produce (slot, state and conv rows, conv / F_B / norm weights) before the PDL
    wait, converts operands at first use, and syncs the cluster only before each
    role's first DSMEM send. Safe because those loads read only weights and data a
    graph or non-PDL boundary orders before this launch (states and conv rows of this
    layer's previous call or a fresh prompt's non-PDL fill, the host's slot table),
    never same-step producer output, which is read after the wait; PDL-chained calls
    without such a boundary must not share state slots. Early 2 also bulk-copies
    the CTA's (contiguous) state rows to shared memory and updates four passes at a
    time, which frees the registers that held them across the input phase; early 3
    (two CTAs per head) also stages the gate warp's F_B rows.

    ``tokens`` q > 1 (speculative verify; ``batch`` counts requests, rows b * q + t)
    runs each request's q tokens in order with the speculative cache contract: slots
    [B, >= q] and accepted [B] (an extra last tensor, read early like the slots); the
    state comes from slot[b, accepted - 1] and token t's state goes to slot[b, t] if
    > 0; conv slot[b, 0] holds [dim, q + 2] with taps at accepted - 1 ..., rewritten to
    [c_accepted, c_accepted + 1, x_0, ..., x_q-1]. Token t's F_B row is MMA column t;
    CTA t % C normalizes token t.

    ``replayssm`` instead reads one committed slots[B] checkpoint (0 is valid,
    negative is padding), leaves persistent state/conv unchanged, and appends
    scratch_rows, conv_window [scratch,Q,3*dim,3], rawv, rawk, log-g and beta
    tensors after F_B. Raw K/V are rounded to BF16 only on ring write; recurrence
    retains FP32 conv values, matching stock nv_cutedsl verify. Commit remains
    with the host. The Q16 gate uses two eight-column MMAs and one rendezvous.

    ``bf16_state`` (default off; set from the temporal pool's dtype,
    ``--mamba-ssm-dtype bfloat16``; a lossy option): the recurrent state pages are BF16.
    Loads widen each value to FP32 exactly, so all recurrence math stays FP32 and, given the
    same state values, every output and ring is bit for bit the FP32-state kernel's. Stores
    (non-ReplaySSM) round the updated state to nearest-even BF16 once. Not with staging
    (early >= 2 copies FP32 rows to shared memory); callers fall back to early 1.
    """

    def __init__(
        self,
        heads,
        batch,
        gate_mode,
        has_bias,
        schedule=(0, 0, 0),
        tokens=1,
        replayssm=False,
        opt=frozenset(),
        bf16_state=False,
    ):
        # A second eight-column gate tile supplies the full Q16 barrier payload.
        limit = 16 if replayssm else 8
        if not 1 <= tokens <= limit:
            raise ValueError(
                f"KdaSplit supports 1-{limit} tokens per request; got {tokens}"
            )
        self.replayssm = replayssm
        self.heads, self.gate_mode, self.has_bias = heads, gate_mode, has_bias
        early, lanes, self.occupancy, cluster, evict = (*schedule, 0, 0)[:5]
        self.early, self.staged = bool(early), early >= 2
        self.bf16_state = bool(bf16_state)  # BF16 recurrent state pages (FP32 math)
        self.row_bytes = 256 if self.bf16_state else 512  # one state row of 128 values
        assert not (self.bf16_state and self.staged), "BF16 state: no staged schedule (early >= 2)"
        # ReplaySSM reads the host's just-committed checkpoint/conv; do not move
        # these reads ahead of the dependency wait without a restore boundary.
        # Opt-in "early": the host commit/restore and slot tables are
        # stream-ordered before the verify graph launch and no in-graph kernel
        # writes this layer's checkpoint or conv, the decode early contract.
        self.opt = frozenset(opt)
        # "pre" with "early": prefetch, wait at entry, then the early code path
        # (deferred cluster wait, weights first); its state/slot loads then
        # follow the wait, and later griddepcontrol.wait calls are no-ops.
        assert "pre" not in self.opt or replayssm
        # "pw" (prefetch weights): replayssm, no conv bias, and not with early
        # (early already loads the weights before the wait).
        assert "pw" not in self.opt or (replayssm and not has_bias and not early)
        assert not replayssm or (
            tokens > 1 and (not early or "early" in self.opt) and gate_mode != 2
        )
        self.gate_staged, self.tokens = early == 3, tokens
        assert gate_mode != 2 or not (early or cluster)
        assert tokens == 1 or not (self.staged or gate_mode == 2 or cluster == 16)
        # TP8 (12 heads) uses fast math; B1 releases dependents at entry.
        self.fast_math, self.trigger_at_entry = heads == 12, batch == 1
        self.trigger_fence = os.environ.get("K31MK_KDA_TRIGGER_FENCE", "0") == "1"  # K-19 pricing (off)
        # K-23 (always on): setup() arms at .cluster.
        self.k23_arm = True
        # Opt-in "lt": release dependents after the input loads (the
        # non-B1 point) instead of at entry. With "early" the entry trigger precedes
        # griddepcontrol.wait, so back-to-back launches can chain without bound.
        if "lt" in self.opt:
            self.trigger_at_entry = False
        self.cluster = cluster or (8 if batch == 1 else 4)
        self.lanes_per_row = lanes or (8 if batch <= (4 if heads == 6 else 2) else 16)
        # Opt-in "wide" (ReplaySSM Q <= 8): 256 main threads. Phase 1 splits
        # each channel's tokens over two thread halves (same per-token math), and
        # the recurrence spreads the CTA's rows over all 256 threads (lanes per row
        # = 256 / rows), so each warp issues about half the per-token work.
        # "quad": the same with 512 main threads (two tokens per phase-1 thread);
        # one CTA per SM (Breq <= 2 grids fit in one wave).
        self.halves = 4 if "quad" in self.opt else 2 if "wide" in self.opt else 1
        self.main = 128 * self.halves
        if self.halves > 1:
            assert replayssm and tokens % self.halves == 0 and tokens <= 8
            # 1 CTA/SM lifts the 2-CTA register cap (65536 / (2 x 288) = 112);
            # the Breq1 grid (96 CTAs) fits one wave either way.
            if self.halves == 4 or batch == 1:
                self.occupancy = 1
            if "wl8" not in self.opt:  # "wl8": keep the schedule's lanes (spare groups idle)
                # "wlf": the schedule's lanes are a floor, so 64-row CTAs (2-CTA
                # clusters) keep 8 lanes = 2 state rows per thread, half the per-token k / q / decay
                # shared-memory reads, while 16 / 32-row CTAs keep wide's 16 / 8 lanes.
                floor = self.lanes_per_row if "wlf" in self.opt else 1
                self.lanes_per_row = max(floor, min(32, self.main // (128 // self.cluster)))
        self.rows, self.groups = 128 // self.cluster, self.main // self.lanes_per_row
        # Opt-in "occ1" (default off; launch bound only, same code): 1 CTA / SM on the 32-row CTAs of the TP16 4-CTA clusters,
        # as Breq 1 already runs: lifts the 2-CTA register cap (ptxas 96 -> 168 registers at Breq 4; 96 / 120 / 144 CTAs at Breq 4 / 5 / 6 B57).
        if "occ1" in self.opt and self.cluster == 4 and self.rows == 32 and heads == 6:
            self.occupancy = 1
        assert "dmc" not in self.opt or (replayssm and tokens <= 8)
        b3w, b3c = "b3w" in self.opt, "b3c" in self.opt
        dmc_cap = B3_MAX_REQUESTS if b3w or b3c else DMC_MAX_REQUESTS
        self.dmc = "dmc" in self.opt and batch <= dmc_cap  # FX-5
        th = tokens // self.halves
        assert "rs" not in self.opt or (2 * th) & (2 * th - 1) == 0
        assert "la" not in self.opt or (replayssm and not self.staged)
        self.passes = (self.rows + self.groups - 1) // self.groups
        # Packed (f32x2) decay / update / row dots: opt "pk" everywhere, or with "wlf" wherever a thread holds
        # 2+ state rows (the 64-row CTAs it widens; Breq 1 / 2's chunk path forbids "pk" in the opt string).
        self.packed = "pk" in self.opt or ("wlf" in self.opt and self.passes > 1)
        # Opt-in "pkb" (default off, bitwise): "pk"'s f32x2 decay multiply and delta-rule update (``update``) on the
        # 32-row CTAs of the 4-CTA clusters (Breq 4; Breq 5 / 6 with B57), where the path is otherwise scalar, with the scalar row dots
        # kept: each lane of mul.rn.f32x2 / fma.rn.f32x2 rounds as the scalar mul / fma.rn does, and no summation order changes.
        self.pkb = "pkb" in self.opt and self.rows == 32 and not self.packed
        # Opt-in "sred" (default off, bitwise): the k-dot's 8-lane butterfly of ``update`` (xor 4, 2, 1) replaced, on the
        # 32-row CTAs (one pass, 8 lanes per row), by one shared store per lane, a warp barrier and two 16-byte loads of the row's 8
        # partials, summed in the butterfly's own order ((l0 + l4) + (l2 + l6)) + ((l1 + l5) + (l3 + l7)) (IEEE add is commutative, so
        # every lane of the butterfly holds exactly that value). Two token slots: a slot is rewritten only after the next token's barrier.
        self.sred = "sred" in self.opt and self.rows == 32 and self.lanes_per_row == 8 and self.passes == 1 and not self.staged
        self.sred = self.sred and "la" not in self.opt
        self.wave = min(4, self.passes) if self.staged else self.passes
        self.chunks = 32 // self.lanes_per_row
        assert not self.gate_staged or self.rows >= 64
        self.evict = bool(evict)  # staged state traffic marked L2 evict-first
        assert self.staged or not self.evict
        # Opt-in "chunk" (ReplaySSM Q8 with "wide"): the token loop becomes one
        # chunked (WY) step, see ``chunk_outputs``. Not bitwise: a reordering of the same
        # FP32 products (outcome-based accuracy rule).
        self.lean = "lean" in self.opt
        if self.lean:
            assert replayssm and tokens == 8 and self.halves == 2 and self.cluster == 8
        assert "bal" not in self.opt or replayssm
        self.chunk = "chunk" in self.opt and batch <= (B3_MAX_REQUESTS if b3c else CHUNK_MAX_REQUESTS)  # FX-8
        if self.chunk:
            assert replayssm and tokens == 8 and self.halves == 2 and not self.staged
            assert "la" not in self.opt and "pk" not in self.opt
            assert self.passes == 1 and self.cluster in (4, 8)
        # Opt-in "tsp" (ReplaySSM Q8 with "wide"): token-split phase 1,
        # see ``prologue_tsp``. Bitwise: same arithmetic, other threads / CTAs. Engages only where
        # one CTA per token exists (8-CTA clusters, Q8) and at the request counts where it measured
        # faster (TSP_MAX_REQUESTS); other shapes keep the base phase 1. prologue_tsp has no fast-math form, so
        # at 12 heads (TP8, fast math) tsp switches itself off and the base phase 1 runs.
        self.tsp = "tsp" in self.opt and tokens == 8 and self.cluster == 8
        tsp_cap = B3_MAX_REQUESTS if b3w or b3c else TSP_MAX_REQUESTS
        self.tsp = self.tsp and batch <= tsp_cap and not self.fast_math
        if self.tsp:
            assert replayssm and self.halves == 2 and not self.staged
            assert not (has_bias or self.fast_math or self.lean)
            assert "bal" not in self.opt and "la" not in self.opt
        # Opt-in "cg" / "cs" (with "chunk", Q8, 8-CTA clusters of 16
        # rows): "cg" = the idle gate warp of CTA p computes Gram row s = p once per head and
        # sends it to every CTA of the cluster (``gram_row``) instead of each CTA's eight main
        # warps recomputing all eight rows; "cs" = P / R with each warp on 16 key columns of all
        # 16 rows, one cross-warp reduction and one solver warp (``cs_products``), so a warp
        # reads 16 columns of decays / keys / queries per token instead of all 128.
        # FX-8: an opt string naming cg / cs / csp without its prerequisite still fails; the paths follow self.chunk.
        assert "chunk" in self.opt or not ({"cg", "cs"} & self.opt)
        assert "cs" in self.opt or "csp" not in self.opt
        self.cgram, self.csplit = self.chunk and "cg" in self.opt, self.chunk and "cs" in self.opt
        self.cs_packed = self.csplit and "csp" in self.opt  # "cs" token loop in f32x2 pairs (a reordering)
        # "gb" (with "cs", not "cg"): the Gram rows by the main warps with balanced work
        # (``gram_balanced``) instead of 1b's one row per warp.
        assert "gb" not in self.opt or ("cs" in self.opt and "cg" not in self.opt)
        self.gbal = self.csplit and "gb" in self.opt
        if self.cgram or self.csplit:
            assert self.chunk and tokens == 8 and self.cluster == 8 and self.rows == 16
            assert self.lanes_per_row == 16 and self.chunks == 2 and self.main == 256
        # One-CTA clusters deliver outputs and decays with own-CTA stores (``put_float``); the paths that
        # send other rows with st.async (decode, tsp, dmc, chunk, cg, cs, lean, two decay rounds) are not built there.
        self.local = self.cluster == 1
        if self.local:
            assert 1 < tokens <= 8 and self.gate_mode != 2
            assert not (self.chunk or self.tsp or self.dmc or self.cgram or self.csplit or self.lean)
        # Opt-in "wy" (ReplaySSM Q8 with "wide", where "chunk" does not engage: 2 / 4-CTA clusters
        # of 64 / 32 rows and one-CTA clusters of 128 rows): the token loop becomes one chunked (WY) step in the
        # dot-basis form, see ``wy_outputs``. Not bitwise: a reordering of the same FP32 products (as "chunk").
        self.wy = "wy" in self.opt and not self.chunk and self.cluster in (1, 2, 4)
        if self.wy:
            assert replayssm and tokens == 8 and self.halves == 2 and self.main == 256 and not self.staged
            assert "la" not in self.opt and not (self.tsp or self.dmc or self.lean)
            assert self.rows == self.passes * self.groups and self.lanes_per_row <= 2 * tokens
        self.chunked = self.chunk or self.wy
        # Opt-in "dq" (bitwise by construction): the serial token loop keeps each token's
        # q-dot partials in registers; their butterflies and the st.async output pushes run after the loop, so the
        # per-token divergent push (BSSY / BSYNC around STAS in SASS) no longer splits the loop into 8 scheduling
        # regions. Same per-value reduction order, same sends / barriers / byte counts, only later in program order
        # (every push still precedes this CTA's waits on its owned outputs). Unstaged multi-CTA token-loop path only.
        self.dq = (
            "dq" in self.opt and replayssm and tokens > 1 and not self.staged and not self.local
            and not self.chunked and "la" not in self.opt
        )
        # Opt-in "dql" (default off, bitwise; with "dq" on the 32-row CTAs): token t - 1's q-dot butterfly and output push run inside
        # the token loop right after token t's update instead of all after the loop (same levels in the same order per value); the loop is
        # latency-bound there (measured), so the reduce can hide in it. Only the last token's stays after the loop.
        self.dql = "dql" in self.opt and self.dq and self.rows == 32
        # Opt-in "vp" (bitwise by construction; with "dq" on 2-CTA clusters, Q8, two halves): each output sender pushes its
        # row's four outputs for one owner CTA (tokens q, q + 2, q + 4, q + 6 of owner q) as ONE 16-byte st.async into an owner-major layout
        # shared_o[4 (position + position // 16) + u] (u = t // 2; one 16-byte pad per 16 positions, so the norm's 8-lane reads hit 8 banks; the
        # unpadded layout measured slower), completing ONE barrier armed with 2048 bytes, instead of 16 scalar pushes on four 512-byte
        # barriers. The owners read the same values in the same order (``output_norm_vp``). Same protocol (st.async complete_tx, byte-counted
        # mbarrier, try_wait.acquire.cluster), larger packets, one barrier.
        self.vp = "vp" in self.opt and self.dq and tokens == 8 and self.cluster == 2 and self.halves == 2
        # merge (c=8 + MID lines): "dql" and "vp" both rewrite the post-loop output push; they were never built together.
        assert not (self.dql and self.vp), "KDA verify opts dql and vp are exclusive"
        # Opt-in "tsp2" (bitwise; ReplaySSM Q8 with "wide" on the 2-CTA clusters of 64 rows that
        # Breq 8 / 16 run on SM107): tsp's token split at cluster 2, CTA p forms tokens 4p .. 4p + 3 once for the cluster
        # (``prologue_tsp2``) instead of both CTAs forming all eight tokens' q / k. With "wy", ``wy_outputs`` reads the same
        # shared q / k / v / beta that ``prologue_tsp2`` leaves (all waited on before it returns), and skips its own
        # cluster wait under an early schedule because ``prologue_tsp2`` already waited before its sends.
        self.tsp2 = "tsp2" in self.opt and tokens == 8 and self.cluster == 2 and self.rows == 64
        self.tsp2 = self.tsp2 and not (self.fast_math or self.tsp)
        if self.tsp2:
            assert replayssm and self.halves == 2 and not self.staged
            assert not (has_bias or self.lean or self.dmc or self.chunk)
            assert "bal" not in self.opt and "la" not in self.opt
        # Opt-in "tsp4" (default off, bitwise; ReplaySSM Q8 with "wide" on the 4-CTA clusters of 32 rows that
        # Breq 4 runs on SM107): ``prologue_tsp4``. Switches itself off (base phase 1) where another phase-1 form or a path
        # that reads phase 1 differently is built (tsp / tsp2, dmc, chunk / wy, lean, fast math) and above TSP4_MAX_REQUESTS.
        self.tsp4 = "tsp4" in self.opt and tokens == 8 and self.cluster == 4 and self.rows == 32
        self.tsp4 = self.tsp4 and batch <= TSP4_MAX_REQUESTS and self.main == 256 and replayssm and not self.staged
        self.tsp4 = self.tsp4 and not (self.fast_math or self.tsp or self.tsp2 or self.dmc or self.chunked or self.lean)
        if self.tsp4:
            assert self.halves == 2 and self.passes == 1 and not has_bias
            assert "bal" not in self.opt and "la" not in self.opt
        # Opt-in "g4" (bitwise; Q8 verify on 64-row CTAs): the F_B projection and decay multicast
        # run on four gate warps of 16 rows each instead of one warp of 64 rows (4 instead of 16 decays per lane). Each row's
        # MMA tile, K order and gate math are the one-warp path's (``gate_warp`` keeps the 64-row CTA's K layout), so bitwise.
        self.gate_warps = 1
        if "g4" in self.opt and tokens == 8 and self.rows == 64 and not self.local:
            assert replayssm and self.gate_mode != 2 and not (self.dmc or self.gate_staged or self.cgram)
            self.gate_warps = 4
        # Opt-in "g2" (default off, bitwise): "g4"'s split on the 32-row CTAs of the 4-CTA clusters (Breq 4; Breq 5 / 6
        # with B57): two gate warps of 16 rows each instead of one of 32 (2 instead of 4 features x 2 tokens of gate math and DSMEM
        # decays per lane). Same MMA tile, K order (``gate_warp``: the CTA's 32 rows are not "wide" either way) and gate math per row.
        # Switches itself off where g4's prerequisites fail (gate mode 2, dmc, staged gate rows, cg).
        if "g2" in self.opt and tokens == 8 and self.rows == 32 and not self.local and self.gate_warps == 1:
            if replayssm and self.gate_mode != 2 and not (self.dmc or self.gate_staged or self.cgram):
                self.gate_warps = 2
        # Opt-in "ds" (bitwise by construction; with "dq", base phase 1): phase 1's side stores (raw-K / raw-V / beta rings and
        # the dense q / k / v conv windows, consumed only by later kernels) are issued after this CTA's output pushes instead of inside phase 1, so the
        # writing CTA (partition 0 writes every q / k ring and window) reaches the token loop as early as its peer. Same values, same addresses.
        self.ds = "ds" in self.opt and self.dq and not (self.tsp or self.tsp2 or self.lean or self.fast_math)
        # Opt-in "dl" (bitwise by construction; with "dq", base phase 1, two halves, Q8; exclusive with "ds"): the same side stores as
        # "ds", issued inside the token loop instead (iteration t < 4 stores phase token t of the thread's half), as predicated stores (no branch in the
        # loop), so they fill the loop's idle load / store slots (under "ds" they cost time in phase 1 and after the pushes).
        self.dl = ("dl" in self.opt and self.dq and not (self.tsp or self.tsp2 or self.lean or self.fast_math or self.ds)
                   and tokens == 8 and self.halves == 2)
        # Opt-in "pq" (default off, KL class: a reordering of the same FP32 products; with "dq" on the packed f32x2 loop):
        # each row's deferred q-dot as one f32x2 accumulator chain over all chunks, then lo + hi (at 4 chunks 1 FMUL2 + 7 FFMA2 + 1 FADD instead
        # of 4 FMUL2 + 4 FFMA2 + 7 FADD). The k-dot (on the per-token chain) keeps its tree. Measured: out within two BF16 ulps of the
        # base in every cell, 2-7 cells changed of 49K-98K, FP64 rms no worse.
        self.pq = "pq" in self.opt and self.dq and self.packed
    @cute.jit
    def __call__(self, *args):
        rows = args[0].shape[0]
        grid = (self.cluster, self.heads, rows // self.tokens)
        bound = dict(min_blocks_per_mp=self.occupancy) if self.occupancy else {}
        launch(self.kernel, args, grid, self.main + 32 * self.gate_warps, self.cluster, **bound)

    @cute.kernel
    def kernel(self, args):
        x, weights, bias, conv_state, raw_gate, beta, a_log, dt_bias, slots = args[:9]
        state, output, gate, norm_weight, lower_bound, scale, eps, f_b_weight = args[
            9:17
        ]
        T: cutlass.Constexpr = self.tokens
        defer_norm_gate: cutlass.Constexpr = self.replayssm and T > 8
        pre: cutlass.Constexpr = "pre" in self.opt
        pw: cutlass.Constexpr = "pw" in self.opt
        if cutlass.const_expr(not self.early and not (pre or pw)):
            cute.arch.griddepcontrol_wait()
        thread, _, _ = cute.arch.thread_idx()
        partition, head, token = cute.arch.block_idx()
        if cutlass.const_expr(pw):
            # Opt-in "pw": before griddepcontrol.wait, L2 prefetches of
            # this CTA's constant weights only (F_B rows, conv weights, dt, A_log,
            # norm weight), written once at load; no value is read before the wait.
            if thread == 0:
                rows_w, dim_w = partition * self.rows, self.heads * 128
                prefetch_bytes(at(f_b_weight, head * 128 + rows_w, 0), self.rows * 256)
                width: cutlass.Constexpr = weights.element_type.width // 8
                for s in cutlass.range_constexpr(3):
                    base_w = head * 128 + (rows_w if s == 2 else 0)
                    span: cutlass.Constexpr = (self.rows if s == 2 else 128) * width
                    for t in cutlass.range_constexpr(4):
                        prefetch_bytes(at(weights, s, t, base_w), span)
                prefetch_l2(at(dt_bias, head * 128 + rows_w))
                prefetch_l2(at(a_log, head))
                prefetch_bytes(at(norm_weight, 0), 128 * (norm_weight.element_type.width // 8))
            if cutlass.const_expr(not pre):
                cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(pre):
            # Opt-in "pre": before griddepcontrol.wait, only L2
            # prefetches of this CTA's checkpoint rows and conv taps, addressed
            # by a slot read that may be stale (clamped; its value only picks
            # the prefetch address). Every value used is loaded after the wait,
            # as in the shipped path, so the pre-wait code observes nothing.
            slots_n: cutlass.Constexpr = state.shape[0]
            s_pre = cutlass.max(cutlass.min(slots[token], slots_n - 1), 0)
            rows_base = partition * self.rows
            if thread == 0:
                rows_page = state[Int64(s_pre), None, None, None]
                prefetch_bytes(at(rows_page, head, rows_base, 0), self.rows * self.row_bytes)
            if thread < 128:
                taps_page = conv_state[Int64(s_pre), None, None]
                dim_pre: cutlass.Constexpr = self.heads * 128
                for s in cutlass.range_constexpr(3):
                    ch_pre = head * 128 + (rows_base if s == 2 else 0) + thread
                    if (s != 2) | (thread < self.rows):
                        for t in cutlass.range_constexpr(3):
                            prefetch_l2(at(taps_page, s * dim_pre + ch_pre, t))
            cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.replayssm):
            request, token = token, token * T
            slot = slots[request]
            conv_slot, offset = slot, 0
            scratch_row = args[17][request]
        elif cutlass.const_expr(T == 1):
            slot = slots[token]
            conv_slot, offset, writes = slot, 0, [slot]
        else:
            request, token = token, token * T
            offset = args[17][request] - 1
            slot, conv_slot = slots[request, offset], slots[request, 0]
            writes = [slots[request, t] for t in range(T)]
        padded = slot < 0 if self.replayssm else slot <= 0
        if cutlass.const_expr(self.replayssm):
            padded = padded | (scratch_row < 0)
        if padded:
            cute.arch.griddepcontrol_wait()
            if (partition == 0) & (thread < 128):
                for t in cutlass.range_constexpr(T):
                    output[advance(token, t), head * 128 + thread] = BFloat16(0)
            if cutlass.const_expr(self.trigger_fence):
                fenced_trigger()
            else:
                cute.arch.griddepcontrol_launch_dependents()
        else:
            lane, warp = thread % 32, thread // 32
            group = thread // self.lanes_per_row
            group_lane = thread % self.lanes_per_row
            dim: cutlass.Constexpr = self.heads * 128
            MAIN: cutlass.Constexpr = self.main
            TH: cutlass.Constexpr = T // self.halves  # tokens per phase-1 half
            # Phase-1 channel of this thread and its token half (wide: tokens
            # half * TH ... half * TH + TH - 1); plain: the thread, all tokens.
            ch = thread if self.halves == 1 else thread % 128
            half = 0 if self.halves == 1 else thread // 128
            value_base, qk_channel = partition * self.rows, head * 128 + ch
            value_channel = head * 128 + value_base + ch

            smem = cutlass.utils.SmemAllocator()
            shared_q, shared_k, shared_decay, shared_v, shared_o, reductions = (
                alloc(smem, Float32, size * T)
                for size in (128, 128, 128, self.rows, 128, 8)
            )
            if cutlass.const_expr(self.halves > 1):  # betas of every token
                shared_beta = alloc(smem, Float32, T)
            kred = None
            if cutlass.const_expr(self.sred):  # opt "sred": [token slot, main thread] k-dot partials
                kred = alloc(smem, Float32, 2 * MAIN)
            barrier, gate_barrier = (smem.allocate(cutlass.Uint64, 8) for _ in range(2))
            owned: cutlass.Constexpr = -(-T // self.cluster)  # tokens a CTA normalizes
            if cutlass.const_expr(owned > 1):  # one output barrier per owned token
                barrier = smem.allocate(8 * owned, 8)
                barrier = cute.recast_ptr(barrier, dtype=cutlass.Uint64)
            if cutlass.const_expr(self.staged):
                staged = alloc(smem, Float32, (self.rows, 128), align=128)
                state_barrier = smem.allocate(cutlass.Uint64, 8)
            gate_rows = None
            if cutlass.const_expr(self.gate_staged):  # [copy, lane] 16-byte slots
                gate_rows = alloc(smem, Uint32, (self.rows // 2, 32, 4), align=128)
            decay_stage = None
            if cutlass.const_expr(self.dmc):  # [decay | log-decay, token, row]
                decay_stage = alloc(smem, Float32, (2, T, self.rows))
            chunk_smem = None
            if cutlass.const_expr(self.chunk):  # P/R [row, 2T]; Gram A/B [s, 2T]
                sizes = (self.rows * 2 * T, 2 * T * T)
                chunk_smem = tuple(alloc(smem, Float32, size) for size in sizes)
            if cutlass.const_expr(self.wy):  # P/R [row, 2T]; Gram A/B [s, 2T]; bases (128 columns x 2T, 16-byte slots)
                sizes = (self.rows * 2 * T, 2 * T * T, 128 * 2 * T)
                chunk_smem = tuple(alloc(smem, Float32, size) for size in sizes)
            gram_barrier = None  # opt "cg": the peers' Gram rows (512 bytes)
            if cutlass.const_expr(self.csplit):  # state [16, 132]; partials [warp, 256]; P/R [16, 20]
                sizes = (16 * 132, 8 * 16 * 2 * T, 16 * 20)
                chunk_smem += tuple(alloc(smem, Float32, size) for size in sizes)
            if cutlass.const_expr(self.gbal):  # Gram parts [column half, s, 2T]
                chunk_smem += (alloc(smem, Float32, 2 * 2 * T * T),)
            qk_barrier, armed = None, ()  # opt "tsp": the normalized q / k rows' barrier
            if cutlass.const_expr(self.tsp or self.tsp2 or self.tsp4):  # tsp2 / tsp4: the same 1024 T bytes per CTA
                qk_barrier = smem.allocate(cutlass.Uint64, 8)
                armed = ((qk_barrier, True, 1024 * T),)
            if cutlass.const_expr(self.cgram):  # opt "cg": every CTA's Gram row, 2T FP32 from each CTA
                gram_barrier = smem.allocate(cutlass.Uint64, 8)
                armed = (*armed, (gram_barrier, True, 2 * T * 4 * self.cluster))
            if cutlass.const_expr(self.trigger_at_entry):
                cute.arch.griddepcontrol_launch_dependents()
            if cutlass.const_expr(not self.early):
                self.setup(barrier, gate_barrier, partition, thread, armed)
                cute.arch.cluster_wait()
            if thread >= MAIN:
                if cutlass.const_expr(self.early):
                    cute.arch.cluster_arrive_relaxed()
                coords = (head, token, value_base, lane)
                if cutlass.const_expr(self.gate_warps > 1):  # opt "g4" / "g2": gate warp w projects rows 16 w .. 16 w + 15
                    gate_rows_n: cutlass.Constexpr = self.rows // self.gate_warps
                    coords = (head, token, value_base + gate_rows_n * ((thread - MAIN) // 32), lane)
                self.gate_warp(
                    args, shared_decay, gate_barrier, coords, gate_rows, decay_stage
                )
                if cutlass.const_expr(self.cgram):  # then Gram row s = partition
                    rings = (shared_q, shared_k, shared_decay)
                    gram_targets = (chunk_smem[1], gram_barrier)
                    self.gram_row(rings, gate_barrier, gram_targets, partition, lane)

            if thread < MAIN:
                if cutlass.const_expr(self.early):  # thread 0 arms; barrier 1 orders it
                    self.setup(barrier, gate_barrier, partition, thread, armed)
                # 64-bit page bases (slot x page stride passes Int32), 32-bit offsets.
                page = lambda s: state[Int64(s), None, None, None]
                state_page = page(slot)
                conv_page = conv_state[Int64(conv_slot), None, None]
                state_row = lambda r, c, s=state_page: at(s, head, value_base + r, c)
                history = zeros((self.passes, self.chunks, 4), Float32)
                hraw = None  # BF16 state: the raw state words, widened at their first use (``widen``)
                if cutlass.const_expr(self.bf16_state):
                    hraw = zeros((self.passes, self.chunks, 2), Uint32)
                if cutlass.const_expr(self.staged):  # noqa: SIM102
                    if thread == 0:
                        cute.arch.mbarrier_init(state_barrier, 1)
                        cute.arch.mbarrier_init_fence()
                        size = self.rows * 512
                        cute.arch.mbarrier_arrive_and_expect_tx(state_barrier, size)
                        src, policy = state_row(0, 0), dict(evict_first=self.evict)
                        copy_g2s(staged.iterator, src, size, state_barrier, **policy)
                for p in cutlass.range_constexpr(0 if self.staged else self.passes):
                    row = p * self.groups + group
                    if row < self.rows:
                        for j in cutlass.range_constexpr(self.chunks):
                            column = 4 * (group_lane + self.lanes_per_row * j)
                            if cutlass.const_expr(self.bf16_state):  # no consumer before the PDL wait
                                words = load_state_bf16x4(state_row(row, column))
                                hraw[p, j, 0] = words[0]
                                hraw[p, j, 1] = words[1]
                            else:
                                values = load_f32x4(state_row(row, column))
                                for e in cutlass.range_constexpr(4):
                                    history[p, j, e] = values[e]
                if cutlass.const_expr(not (self.tsp or self.tsp2 or self.tsp4)):  # base phase-1 operands (tsp: prologue_tsp)
                    taps = zeros((3, 3), BFloat16)
                    coeff = zeros((3, 4), weights.element_type if self.early else Float32)
                    accum = zeros(3 * TH, Float32)
                    for s in cutlass.range_constexpr(3):
                        channel = value_channel if s == 2 else qk_channel
                        if (s != 2) | (ch < self.rows):
                            if cutlass.const_expr(self.halves == 1):
                                for t in cutlass.range_constexpr(3):
                                    taps[s, t] = conv_page[s * dim + channel, offset + t]
                            else:  # window index half*TH + t: conv taps below 3, inputs after
                                for t in cutlass.range_constexpr(3):
                                    if half * TH + t < 3:
                                        tap = offset + half * TH + t
                                        taps[s, t] = conv_page[s * dim + channel, tap]
                            for t in cutlass.range_constexpr(4):
                                coeff[s, t] = weights[s, t, channel]
                            if cutlass.const_expr(self.has_bias):
                                accum[s] = bias[s * dim + channel]
                                for t in cutlass.range_constexpr(1, TH):
                                    accum[3 * t + s] = accum[s]
                nw = norm_weight[ch]

                if cutlass.const_expr(self.tsp):  # opt "tsp": token-split phase 1
                    place = (thread, partition, head, token, slot, scratch_row, value_base)
                    buffers = (shared_q, shared_k, shared_v, shared_beta, reductions, qk_barrier)
                    gate_owned = self.prologue_tsp(args, place, buffers, conv_page)
                    gates, qks = [], []  # base-path names (unused under tsp)
                    taps, inputs = zeros((3, 3), BFloat16), zeros(3 * TH, BFloat16)
                    channel = value_channel  # (staged-flow joins below need these defined)
                elif cutlass.const_expr(self.tsp2):  # opt "tsp2" (D3): tsp's token split at 2-CTA clusters
                    place = (thread, partition, head, token, slot, scratch_row, value_base)
                    buffers = (shared_q, shared_k, shared_v, shared_beta, reductions, qk_barrier)
                    gate_owned = self.prologue_tsp2(args, place, buffers, conv_page)
                    gates, qks = [], []  # base-path names (unused under tsp2)
                    taps, inputs = zeros((3, 3), BFloat16), zeros(3 * TH, BFloat16)
                    channel = value_channel
                elif cutlass.const_expr(self.tsp4):  # opt "tsp4": tsp's token split at 4-CTA clusters
                    place = (thread, partition, head, token, slot, scratch_row, value_base)
                    buffers = (shared_q, shared_k, shared_v, shared_beta, reductions, qk_barrier)
                    gate_owned = self.prologue_tsp4(args, place, buffers, conv_page)
                    gates, qks = [], []  # base-path names (unused under tsp4)
                    taps, inputs = zeros((3, 3), BFloat16), zeros(3 * TH, BFloat16)
                    channel = value_channel
                else:
                    cute.arch.griddepcontrol_wait()
                    if cutlass.const_expr(self.halves > 1):
                        for s in cutlass.range_constexpr(3):
                            channel = value_channel if s == 2 else qk_channel
                            if (s != 2) | (ch < self.rows):
                                for t in cutlass.range_constexpr(3):
                                    if half * TH + t >= 3:  # an input token, after the wait
                                        row = token + half * TH - 3 + t
                                        taps[s, t] = x[row, s * dim + channel]
                    inputs = cute.make_rmem_tensor(3 * TH, BFloat16)
                    betas, gates = [], []
                    for t in cutlass.range_constexpr(TH):
                        row = advance(token, t) if self.halves == 1 else token + half * TH + t
                        inputs[3 * t], inputs[3 * t + 1] = (
                            x[row, qk_channel],
                            x[row, dim + qk_channel],
                        )
                        inputs[3 * t + 2] = BFloat16(0)
                        if cutlass.const_expr(t == TH - 1):  # noqa: SIM102 (all tokens)
                            if ch < self.rows:
                                for u in cutlass.range_constexpr(TH):
                                    channel = 2 * dim + value_channel
                                    u_row = (
                                        advance(token, u)
                                        if self.halves == 1
                                        else token + half * TH + u
                                    )
                                    inputs[3 * u + 2] = x[u_row, channel]
                        if cutlass.const_expr(self.lean):  # opt "lean": see below
                            betas.append(beta[row, head])
                        elif cutlass.const_expr(self.early):  # converted at first use
                            betas.append(beta[row, head])
                            if cutlass.const_expr(not defer_norm_gate):
                                gates.append(gate[row, qk_channel])
                        else:
                            betas.append(Float32(beta[row, head]))
                            if cutlass.const_expr(not defer_norm_gate):
                                gates.append(Float32(gate[row, qk_channel]))
                    if cutlass.const_expr(self.lean):
                        # Opt-in "lean" (after the ported LEAN_VERIFY): only
                        # the owned token's output gate is activated (CTA p owns token p:
                        # cluster 8, Q8), and only the beta stagers (ch 0) activate betas.
                        # Same sigmoid as the batch path (``sigmoids`` is bit for bit
                        # ``sigmoid``): bitwise.
                        gate_owned = gate[token + partition, qk_channel]
                    if cutlass.const_expr(self.trigger_fence):
                        fenced_trigger()
                    else:
                        cute.arch.griddepcontrol_launch_dependents()

                    vs, squares, qks = [], [], []
                    for t in cutlass.range_constexpr(TH):  # window: 3 taps, then the inputs
                        for s in cutlass.range_constexpr(3):
                            if cutlass.const_expr(self.replayssm):
                                # Stock nv_cutedsl contracts the FIRST product of
                                # the initial pair into FMA. Reversing that FMA
                                # changes BF16 tie rounding in the raw K/V ring.
                                values = [
                                    Float32(taps[s, t + i])
                                    if t + i < 3
                                    else Float32(inputs[3 * (t + i - 3) + s])
                                    for i in range(4)
                                ]
                                accum[3 * t + s] = fma(
                                    values[0],
                                    Float32(coeff[s, 0]),
                                    values[1] * Float32(coeff[s, 1]),
                                )
                                for i in cutlass.range_constexpr(2, 4):
                                    accum[3 * t + s] = fma(
                                        values[i], Float32(coeff[s, i]), accum[3 * t + s]
                                    )
                            else:
                                for i in cutlass.range_constexpr(4):
                                    w = t + i
                                    term = taps[s, w] if w < 3 else inputs[3 * (w - 3) + s]
                                    term = Float32(term)
                                    cf = Float32(coeff[s, i]) if self.early else coeff[s, i]
                                    acc = accum[3 * t + s]
                                    accum[3 * t + s] = fma(
                                        term, cf, acc, ftz=self.fast_math
                                    )
                    act = lambda i, value: self.activation(value)
                    if cutlass.const_expr(T > 1 and not self.fast_math):
                        raw = [accum[i] for i in range(3 * TH)]
                        if cutlass.const_expr(not self.lean):
                            raw += betas + gates
                        activated = sigmoids(raw)
                        act = partial(lambda batch, i, value: batch[i], activated)
                    for t in cutlass.range_constexpr(TH):
                        gt = t if self.halves == 1 else half * TH + t  # token in the window
                        q = accum[3 * t] * act(3 * t, accum[3 * t])
                        k = accum[3 * t + 1] * act(3 * t + 1, accum[3 * t + 1])
                        v = accum[3 * t + 2] * act(3 * t + 2, accum[3 * t + 2])
                        # Opt-in "bal": the raw-K ring, beta ring and q / k
                        # conv windows of token gt are written by CTA gt % cluster instead
                        # of all by partition 0 (which also owns token 0): same values.
                        qk_writer = partition == 0
                        if cutlass.const_expr("bal" in self.opt):
                            qk_writer = partition == gt % self.cluster
                        if cutlass.const_expr(self.lean):
                            if ch == 0:
                                beta_act = self.activation(Float32(betas[t]))
                                shared_beta[gt] = beta_act
                                if qk_writer:
                                    args[22][Int64(slot), head, gt] = beta_act
                        else:
                            betas[t] = act(3 * TH + t, Float32(betas[t]))
                            if cutlass.const_expr(not defer_norm_gate):
                                gates[t] = act(4 * TH + t, Float32(gates[t]))
                            if cutlass.const_expr(self.halves > 1):
                                if ch == 0:
                                    shared_beta[gt] = betas[t]
                        if cutlass.const_expr(self.replayssm and not (self.ds or self.dl)):
                            if qk_writer:
                                args[20][Int64(slot), head, gt, ch] = BFloat16(k)
                                if cutlass.const_expr(not self.lean):
                                    if ch == 0:
                                        args[22][Int64(slot), head, gt] = betas[t]
                            if ch < self.rows:
                                args[19][Int64(slot), head, gt, value_base + ch] = (
                                    BFloat16(v)
                                )
                            # Dense stock intermediate window: post-token raw taps,
                            # indexed by the scratch row, independently of state slot.
                            for section in cutlass.range_constexpr(3):
                                channel = value_channel if section == 2 else qk_channel
                                writer = ch < self.rows if section == 2 else qk_writer
                                if writer:
                                    for i in cutlass.range_constexpr(3):
                                        index = t + i + 1
                                        value = (
                                            taps[section, index]
                                            if index < 3
                                            else inputs[3 * (index - 3) + section]
                                        )
                                        args[18][
                                            Int64(scratch_row),
                                            gt,
                                            section * dim + channel,
                                            i,
                                        ] = value
                        if cutlass.const_expr(self.halves == 1):
                            shared_q[advance(thread, 128 * t)] = q
                            shared_k[advance(thread, 128 * t)] = k
                        else:
                            shared_q[ch + 128 * gt] = q
                            shared_k[ch + 128 * gt] = k
                        vs.append(v)
                        qks += [q, k]
                        if cutlass.const_expr(t == TH - 1):  # noqa: SIM102
                            if ch < self.rows:
                                for u in cutlass.range_constexpr(TH):
                                    if cutlass.const_expr(self.halves == 1):
                                        shared_v[advance(thread, self.rows * u)] = vs[u]
                                    else:
                                        gu = half * TH + u
                                        shared_v[ch + self.rows * gu] = vs[u]
                        if cutlass.const_expr("rs" in self.opt):
                            squares += [q * q, k * k]  # reduced together after the loop
                        else:
                            squares += [warp_sum(q * q), warp_sum(k * k)]
                        if cutlass.const_expr(t == TH - 1 and "rs" in self.opt):
                            # Opt-in "rs": one reduce-scatter of all 2 * TH sums
                            # of squares over the warp (2 * TH + log2(32 / 2TH) shuffles
                            # instead of 10 * TH); lane groups own one total each.
                            sq = cute.make_rmem_tensor(2 * TH, Float32)
                            for u in cutlass.range_constexpr(2 * TH):
                                sq[u] = squares[u]
                            total, index = scatter_reduce(sq, lane, 2 * TH)
                            if lane % (32 // (2 * TH)) == 0:
                                which, gu = index % 2, index // 2
                                if cutlass.const_expr(self.halves > 1):
                                    gu = gu + half * TH
                                reductions[warp % 4 + 4 * which + 8 * gu] = total
                        elif cutlass.const_expr(t == TH - 1):  # noqa: SIM102
                            if lane == 0:
                                for u in cutlass.range_constexpr(TH):
                                    if cutlass.const_expr(self.halves == 1):
                                        reductions[advance(warp, 8 * u)] = squares[2 * u]
                                        reductions[advance(warp + 4, 8 * u)] = squares[
                                            2 * u + 1
                                        ]
                                    else:
                                        gu = half * TH + u
                                        reductions[warp % 4 + 8 * gu] = squares[2 * u]
                                        reductions[warp % 4 + 4 + 8 * gu] = squares[
                                            2 * u + 1
                                        ]
                    cute.arch.barrier(barrier_id=1, number_of_threads=MAIN)

                def factors(t):
                    r = lambda i: reductions[8 * t + i]
                    qn = (r(0) + r(2)) + (r(1) + r(3))
                    kn = (r(4) + r(6)) + (r(5) + r(7))
                    fq = cute.math.rsqrt(qn + 1.0e-6, fastmath=True) * scale
                    return fq, cute.math.rsqrt(kn + 1.0e-6, fastmath=True)

                if cutlass.const_expr(T > 1 and not (self.tsp or self.tsp2 or self.tsp4)):  # normalize each token's q, k once
                    for t in cutlass.range_constexpr(TH):
                        if cutlass.const_expr(self.halves == 1):
                            fq, fk = factors(t)
                            shared_q[128 * t + thread] = qks[2 * t] * fq
                            shared_k[128 * t + thread] = qks[2 * t + 1] * fk
                        else:
                            gt = half * TH + t
                            fq, fk = factors(gt)
                            shared_q[128 * gt + ch] = qks[2 * t] * fq
                            shared_k[128 * gt + ch] = qks[2 * t + 1] * fk
                    cute.arch.barrier(barrier_id=1, number_of_threads=MAIN)

                def operands(t):
                    fq, fk = factors(t) if T == 1 else (None, None)
                    qr = cute.make_rmem_tensor((self.chunks, 4), Float32)
                    kr = cute.make_rmem_tensor((self.chunks, 4), Float32)
                    for j in cutlass.range_constexpr(self.chunks):
                        column = 4 * (group_lane + self.lanes_per_row * j)
                        q4 = load_f32x4(
                            advance(shared_q.iterator, 128 * t) + column, shared=True
                        )
                        k4 = load_f32x4(
                            advance(shared_k.iterator, 128 * t) + column, shared=True
                        )
                        for e in cutlass.range_constexpr(4):
                            qr[j, e] = q4[e] * fq if T == 1 else q4[e]
                            kr[j, e] = k4[e] * fk if T == 1 else k4[e]
                    if cutlass.const_expr(t == 0):
                        wait(gate_barrier)
                    decays = []
                    for j in cutlass.range_constexpr(self.chunks):
                        column = 4 * (group_lane + self.lanes_per_row * j)
                        decay_ptr = advance(shared_decay.iterator, 128 * t) + column
                        decays.append(load_f32x4(decay_ptr, shared=True))
                    values = shared_v
                    if cutlass.const_expr(T > 1):
                        pointer = shared_v.iterator + self.rows * t
                        values = cute.make_tensor(pointer, cute.make_layout(self.rows))
                    if cutlass.const_expr(self.halves > 1):
                        return kr, qr, decays, values, shared_beta[t]
                    return kr, qr, decays, values, betas[t]

                self.widen(history, hraw)  # BF16 state: first use of the state (no-op for FP32)
                if cutlass.const_expr(self.chunk):
                    self.chunk_outputs(
                        history,
                        (shared_q, shared_k, shared_decay, shared_v, shared_beta),
                        chunk_smem,
                        (gate_barrier, barrier, shared_o),
                        thread,
                        (group, group_lane),
                        value_base,
                        gram_barrier,
                    )
                if cutlass.const_expr(self.wy):
                    self.wy_outputs(
                        history,
                        (shared_q, shared_k, shared_decay, shared_v, shared_beta),
                        chunk_smem,
                        (gate_barrier, barrier, shared_o),
                        thread,
                        (group, group_lane),
                        value_base,
                    )
                ready = None if self.chunked else operands(0)
                look = None  # opt "la": (A_t per pass, vn_{t-1} per pass, c_{t-1})
                qd = None  # opt "dq": token t's q-dot partials per pass, reduced and pushed after the loop
                if cutlass.const_expr(self.dq):
                    qd = cute.make_rmem_tensor((T, self.passes), Float32)
                for t in cutlass.range_constexpr(0 if self.chunked else T):
                    current, ready = ready, operands(t + 1) if t + 1 < T else None
                    # Butterfly order: interleave rows per level, publish before state.
                    dots = cute.make_rmem_tensor(self.passes, Float32)
                    if cutlass.const_expr(self.staged):
                        wait(state_barrier)
                    source = staged if self.staged else None
                    owner, bar = t % self.cluster, advance(barrier, t // self.cluster)
                    for start in cutlass.range_constexpr(0, self.passes, self.wave):
                        stop = min(start + self.wave, self.passes)
                        lanes = (group, group_lane)
                        if cutlass.const_expr("la" in self.opt):
                            look = self.update_lookahead(
                                history, dots, current, ready, lanes, look
                            )
                        else:
                            red = None
                            if cutlass.const_expr(self.sred):  # this token's slot of the k-dot partials (two slots)
                                red = (kred, (t % 2) * MAIN, thread)
                            self.update(
                                history, dots, current, lanes, (start, stop), source, red
                            )
                        if cutlass.const_expr(self.early and start == 0 and t == 0):  # noqa: SIM102
                            if cutlass.const_expr(not (self.tsp or self.tsp2 or self.tsp4)):  # tsp / tsp2 / tsp4: waited before their sends
                                cute.arch.cluster_wait()
                        for p in cutlass.range_constexpr(start, stop):
                            row = p * self.groups + group
                            send = (row < self.rows) & (group_lane == 0)
                            if cutlass.const_expr(self.dq):
                                qd[t, p] = dots[p]
                            elif cutlass.const_expr(T > 1):  # predicated: no branches
                                position = self.output_position(value_base + row)
                                dst = advance(shared_o.iterator, 128 * t) + position
                                if cutlass.const_expr(self.local):
                                    put_float(dst, dots[p], when=send)
                                else:
                                    push_float(dst, dots[p], bar, owner, when=send)
                            elif send:
                                position = self.output_position(value_base + row)
                                dst = shared_o.iterator + position
                                store_async_dsmem(dst, float_bits(dots[p]), bar, owner)
                        if cutlass.const_expr(not self.replayssm):
                            rows = partial(state_row, s=page(writes[t]))
                            when = None if T == 1 else writes[t] > 0
                            self.store_rows(history, rows, (start, stop), lanes, when)
                    if cutlass.const_expr(self.local):  # token t's senders: setup() counts them
                        if (group_lane == 0) & (group < self.rows):
                            arrive_release_cta(bar)
                    if cutlass.const_expr(self.dql and t >= 1):  # opt "dql": token t - 1's q-dot reduce + push inside the loop
                        for shift in reversed(range(self.lanes_per_row.bit_length() - 1)):
                            for p in cutlass.range_constexpr(self.passes):
                                qd[t - 1, p] += cute.arch.shuffle_sync_bfly(qd[t - 1, p], 1 << shift)
                        for p in cutlass.range_constexpr(self.passes):
                            row = p * self.groups + group
                            send = (row < self.rows) & (group_lane == 0)
                            position = self.output_position(value_base + row)
                            dst = advance(shared_o.iterator, 128 * (t - 1)) + position
                            push_float(dst, qd[t - 1, p], advance(barrier, (t - 1) // self.cluster), (t - 1) % self.cluster, when=send)
                    if cutlass.const_expr(self.dl and t < TH):  # opt "dl": phase token t's side stores (this thread's half)
                        dl_gt = half * TH + t
                        dl_qk = partition == 0
                        if cutlass.const_expr("bal" in self.opt):
                            dl_qk = partition == dl_gt % self.cluster
                        store_bf16_when(at(args[20][Int64(slot), None, None, None], head, dl_gt, ch), qks[2 * t + 1], dl_qk)
                        store_f32_when(at(args[22][Int64(slot), None, None], head, dl_gt), betas[t], dl_qk & (ch == 0))
                        store_bf16_when(at(args[19][Int64(slot), None, None, None], head, dl_gt, value_base + ch), vs[t], ch < self.rows)
                        for dl_s in cutlass.range_constexpr(3):
                            dl_channel = value_channel if dl_s == 2 else qk_channel
                            dl_writer = (ch < self.rows) if dl_s == 2 else dl_qk
                            for dl_i in cutlass.range_constexpr(3):
                                dl_index = t + dl_i + 1
                                dl_value = taps[dl_s, dl_index] if dl_index < 3 else inputs[3 * (dl_index - 3) + dl_s]
                                store_bf16_when(at(args[18][Int64(scratch_row), None, None, None], dl_gt, dl_s * dim + dl_channel, dl_i),
                                                Float32(dl_value), dl_writer)
                if cutlass.const_expr(self.dq):
                    # The same butterfly levels per value as ``update`` (level order unchanged), interleaved over tokens.
                    levels = self.lanes_per_row.bit_length() - 1
                    tail = (T - 1,) if self.dql else tuple(range(T))  # opt "dql": the earlier tokens were reduced and pushed in the loop
                    for shift in reversed(range(levels)):
                        for t in tail:
                            for p in cutlass.range_constexpr(self.passes):
                                qd[t, p] += cute.arch.shuffle_sync_bfly(qd[t, p], 1 << shift)
                    for t in (() if self.vp else tail):
                        owner, bar = t % self.cluster, advance(barrier, t // self.cluster)
                        for p in cutlass.range_constexpr(self.passes):
                            row = p * self.groups + group
                            send = (row < self.rows) & (group_lane == 0)
                            position = self.output_position(value_base + row)
                            dst = advance(shared_o.iterator, 128 * t) + position
                            push_float(dst, qd[t, p], bar, owner, when=send)
                    if cutlass.const_expr(self.vp):  # opt "vp": one 16-byte packet per (row, owner CTA)
                        for q in cutlass.range_constexpr(self.cluster):
                            for p in cutlass.range_constexpr(self.passes):
                                row = p * self.groups + group
                                send = (row < self.rows) & (group_lane == 0)
                                position = self.output_position(value_base + row)
                                dst = shared_o.iterator + 4 * (position + (position >> 4))
                                words = tuple(qd[q + self.cluster * u, p] for u in range(4))
                                push_float(dst, words, barrier, q, when=send)
                if cutlass.const_expr(self.ds):  # opt "ds": phase 1's ring / window side stores, now (same values, same addresses)
                    for t in cutlass.range_constexpr(TH):
                        gt = t if self.halves == 1 else half * TH + t
                        qk_writer = partition == 0
                        if cutlass.const_expr("bal" in self.opt):
                            qk_writer = partition == gt % self.cluster
                        if qk_writer:
                            args[20][Int64(slot), head, gt, ch] = BFloat16(qks[2 * t + 1])
                            if ch == 0:
                                args[22][Int64(slot), head, gt] = betas[t]
                        if ch < self.rows:
                            args[19][Int64(slot), head, gt, value_base + ch] = BFloat16(vs[t])
                        for section in cutlass.range_constexpr(3):
                            channel = value_channel if section == 2 else qk_channel
                            writer = ch < self.rows if section == 2 else qk_writer
                            if writer:
                                for i in cutlass.range_constexpr(3):
                                    index = t + i + 1
                                    value = taps[section, index] if index < 3 else inputs[3 * (index - 3) + section]
                                    args[18][Int64(scratch_row), gt, section * dim + channel, i] = value
                if (thread < self.rows) & (not self.replayssm):
                    channel = 2 * dim + value_channel
                    window = (
                        taps[2, 1],
                        taps[2, 2],
                        *(inputs[3 * t + 2] for t in range(T)),
                    )
                    for i, value in enumerate(window):
                        conv_page[channel, i] = value

                if cutlass.const_expr(defer_norm_gate):
                    # A Q16 CTA owns only two outputs. Read their gates here,
                    # rather than retaining all sixteen gates in every CTA.
                    for u in cutlass.range_constexpr(owned):
                        owned_token = partition + u * self.cluster
                        if owned_token < T:
                            wait(barrier + u)
                            gate_raw = Float32(gate[token + owned_token, qk_channel])
                            gate_value = self.activation(gate_raw)
                            result = self.output_norm(
                                shared_o, owned_token, thread, lane, nw, gate_value, eps
                            )
                            output[token + owned_token, qk_channel] = BFloat16(result)
                else:
                    for t in cutlass.range_constexpr(T):
                        owner = partition == t % self.cluster
                        if cutlass.const_expr(self.halves > 1):
                            owner = owner & (half == t // TH)
                        if owner:
                            wait(barrier if self.vp else advance(barrier, t // self.cluster))
                            if cutlass.const_expr(self.tsp2):  # owned tokens of this half: 4 half + p, 4 half + p + 2
                                gate_t = self.activation(Float32(gate_owned[(t % TH) // 2]))
                            else:
                                gate_t = (
                                    self.activation(Float32(gate_owned))
                                    if self.lean or self.tsp or self.tsp4  # tsp4: the half's one owned token 4 half + p
                                    else gates[t % TH]
                                )
                            if cutlass.const_expr(self.vp):
                                result = self.output_norm_vp(shared_o, t // self.cluster, ch, lane, nw, gate_t, eps)
                            else:
                                result = self.output_norm(shared_o, t, ch, lane, nw, gate_t, eps)
                            output[advance(token, t), qk_channel] = BFloat16(result)
                            if cutlass.const_expr(t == 0 and not self.replayssm):
                                for section in cutlass.range_constexpr(2):
                                    channel = section * dim + qk_channel
                                    window = (taps[section, 1], taps[section, 2])
                                    window += tuple(
                                        inputs[3 * u + section] for u in range(T)
                                    )
                                    for i, value in enumerate(window):
                                        conv_page[channel, i] = value

    @cute.jit
    def prologue_tsp(self, args, place, buffers, conv_page):
        """Opt-in "tsp": token-split phase 1 of the ReplaySSM Q8 verify on an
        8-CTA cluster. CTA p alone forms token p's q (threads 0-127) and k (128-255) of channel
        thread % 128: conv over window indices p .. p + 3 (conv taps below 3, input rows
        after), SiLU, the warp's sum of squares, the token's factor from the four warp sums,
        and stores the normalized row into every CTA's shared q / k (st.async, 16-byte
        packets, completing ``qk_barrier``: 8 tokens x 2 x 512 B per CTA). It writes token
        p's raw-K ring row and q / k conv windows. Threads 0-127 also form v of value row
        thread % rows for token thread // rows (shared v, raw-V ring, v windows); threads
        128-135 the betas (CTA p writes token p's beta ring entry). Every value is the base
        path's arithmetic in the base order (FMA chain, sigmoid, xor-butterfly sums, factor
        formula), computed once per cluster instead of once per CTA: bitwise. Returns the
        raw output gate of the owned token p (activated at the output, as opt "lean")."""
        x, weights, beta, gate, scale = args[0], args[1], args[5], args[11], args[14]
        thread, partition, head, token, slot, scratch_row, value_base = place
        shared_q, shared_k, shared_v, shared_beta, reductions, qk_barrier = buffers
        T: cutlass.Constexpr = self.tokens
        MAIN: cutlass.Constexpr = self.main
        dim: cutlass.Constexpr = self.heads * 128
        rows: cutlass.Constexpr = self.rows
        lane, warp = thread % 32, thread // 32
        ch, section = thread % 128, thread // 128
        column = section * dim + head * 128 + ch  # token p's q / k channel
        v_token, v_row = thread // rows, thread % rows  # v items: threads < rows * T
        v_column = 2 * dim + head * 128 + value_base + v_row
        has_v = thread < rows * T
        is_beta = (thread >= 128) & (thread < 128 + T)
        # Window index w = p + i: conv tap w if w < 3, else input row w - 3 (base order).
        window, v_window = zeros(4, BFloat16), zeros(4, BFloat16)
        coeff, v_coeff = zeros(4, weights.element_type), zeros(4, weights.element_type)
        beta_raw = zeros(1, beta.element_type)
        for i in cutlass.range_constexpr(3):
            if partition + i < 3:
                window[i] = conv_page[column, partition + i]
        for i in cutlass.range_constexpr(4):
            coeff[i] = weights[section, i, head * 128 + ch]
        if has_v:
            for i in cutlass.range_constexpr(3):
                if v_token + i < 3:
                    v_window[i] = conv_page[v_column, v_token + i]
            for i in cutlass.range_constexpr(4):
                v_coeff[i] = weights[2, i, head * 128 + value_base + v_row]
        cute.arch.griddepcontrol_wait()
        for i in cutlass.range_constexpr(4):
            if partition + i >= 3:
                window[i] = x[token + partition + i - 3, column]
        if has_v:
            for i in cutlass.range_constexpr(4):
                if v_token + i >= 3:
                    v_window[i] = x[token + v_token + i - 3, v_column]
        gate_owned = gate[token + partition, head * 128 + ch]
        if is_beta:
            beta_raw[0] = beta[token + thread - 128, head]
        if cutlass.const_expr(self.trigger_fence):
            fenced_trigger()
        else:
            cute.arch.griddepcontrol_launch_dependents()

        def conv(taps, weights4):  # stock nv_cutedsl FMA order (first product contracted)
            values = [Float32(taps[i]) for i in range(4)]
            acc = fma(values[0], Float32(weights4[0]), values[1] * Float32(weights4[1]))
            for i in cutlass.range_constexpr(2, 4):
                acc = fma(values[i], Float32(weights4[i]), acc)
            return acc

        acc = conv(window, coeff)
        qk = acc * self.activation(acc)
        square = warp_sum(qk * qk)  # the base reduce-scatter's xor-butterfly order
        if lane == 0:
            reductions[warp] = square  # q sums 0-3, k sums 4-7 (base slots of token p)
        subset_barrier(2 + section, 128)  # the section's four warps
        r = lambda i: reductions[4 * section + i]
        factor = zeros(1, Float32)
        factor[0] = cute.math.rsqrt((r(0) + r(2)) + (r(1) + r(3)) + 1.0e-6, fastmath=True)
        if section == 0:
            factor[0] = factor[0] * scale
        normalized = qk * factor[0]
        # 16-byte packets: lane group 4j holds channels 4j .. 4j + 3; lane 4j + d sends the
        # packet to CTAs d and d + 4 (the cluster waits for every peer's armed barrier).
        packet = tuple(cute.arch.shuffle_sync(normalized, (lane & 28) + e) for e in range(4))
        base_q = Int32(shared_address(shared_q.iterator))
        base_k = Int32(shared_address(shared_k.iterator))
        address = base_q + section * (base_k - base_q) + 4 * (128 * partition + (ch & 124))
        if cutlass.const_expr(self.early):  # (otherwise the CTA waited at entry, after setup)
            cute.arch.cluster_wait()
        for d in cutlass.range_constexpr(2):
            push_float(address, packet, qk_barrier, (lane & 3) + 4 * d)
        # Side outputs of token p (stock values): raw K ring, q / k conv windows.
        if section == 1:
            args[20][Int64(slot), head, partition, ch] = BFloat16(qk)
        for i in cutlass.range_constexpr(3):
            args[18][Int64(scratch_row), partition, column, i] = window[i + 1]
        if has_v:
            v_acc = conv(v_window, v_coeff)
            v = v_acc * self.activation(v_acc)
            shared_v[v_row + rows * v_token] = v
            args[19][Int64(slot), head, v_token, value_base + v_row] = BFloat16(v)
            for i in cutlass.range_constexpr(3):
                args[18][Int64(scratch_row), v_token, v_column, i] = v_window[i + 1]
        if is_beta:
            beta_act = self.activation(Float32(beta_raw[0]))
            shared_beta[thread - 128] = beta_act
            if thread - 128 == partition:
                args[22][Int64(slot), head, partition] = beta_act
        subset_barrier(1, MAIN)  # main warps only (the gate warp never joins)
        wait(qk_barrier)  # every token's normalized q / k rows (acquire.cluster)
        return gate_owned

    @cute.jit
    def prologue_tsp2(self, args, place, buffers, conv_page):
        """Opt-in "tsp2": ``prologue_tsp`` at 2-CTA clusters (64 value rows per CTA, Q8).
        CTA p forms tokens 4p .. 4p + 3 once for the cluster: q (threads 0-127) / k (128-255) of channel thread % 128
        over window indices 4p .. 4p + 6 (conv taps below 3, input rows after), SiLU, per token the warp's sum of
        squares (one reduce-scatter of the four tokens' squares: the base "rs" xor-butterfly tree, so the same sums),
        the four warp sums -> the token's factor, and stores the normalized rows into both CTAs' shared q / k (16-byte
        st.async packets completing ``qk_barrier``: 2 CTAs x 256 lanes x 16 B = 1024 T bytes per CTA). It writes those
        tokens' raw-K ring rows, q / k conv windows and beta ring entries. Every main thread also forms v of value row
        thread % 64 for tokens 2 (thread // 64) and 2 (thread // 64) + 1 (shared v, raw-V ring, v windows); threads
        128-135 the betas of every token. The base path's arithmetic in the base order: bitwise. Returns the raw output
        gates of the thread's two owned tokens 4 half + p and 4 half + p + 2 (activated at the output)."""
        x, weights, beta, gate, scale = args[0], args[1], args[5], args[11], args[14]
        thread, partition, head, token, slot, scratch_row, value_base = place
        shared_q, shared_k, shared_v, shared_beta, reductions, qk_barrier = buffers
        T: cutlass.Constexpr = self.tokens
        MAIN: cutlass.Constexpr = self.main
        dim: cutlass.Constexpr = self.heads * 128
        rows: cutlass.Constexpr = self.rows
        TP: cutlass.Constexpr = T // self.cluster  # tokens formed per CTA (4)
        TH: cutlass.Constexpr = T // self.halves  # tokens per output half (4)
        lane, warp = thread % 32, thread // 32
        ch, section = thread % 128, thread // 128
        column = section * dim + head * 128 + ch  # this thread's q / k channel
        first = partition * TP  # first token formed by this CTA
        v_row, v_pair = thread % rows, thread // rows  # v of tokens 2 v_pair, 2 v_pair + 1
        v_column = 2 * dim + head * 128 + value_base + v_row
        is_beta = (thread >= 128) & (thread < 128 + T)
        window = zeros(TP + 3, BFloat16)  # window index first + i: conv tap if < 3, else input row - 3
        v_window = zeros(5, BFloat16)  # window index 2 v_pair + i
        coeff, v_coeff = zeros(4, weights.element_type), zeros(4, weights.element_type)
        beta_raw = zeros(1, beta.element_type)
        gate_owned = zeros(2, gate.element_type)
        for i in cutlass.range_constexpr(TP + 3):
            if first + i < 3:
                window[i] = conv_page[column, first + i]
        for i in cutlass.range_constexpr(4):
            coeff[i] = weights[section, i, head * 128 + ch]
            v_coeff[i] = weights[2, i, head * 128 + value_base + v_row]
        for i in cutlass.range_constexpr(5):
            if 2 * v_pair + i < 3:
                v_window[i] = conv_page[v_column, 2 * v_pair + i]
        cute.arch.griddepcontrol_wait()
        for i in cutlass.range_constexpr(TP + 3):
            if first + i >= 3:
                window[i] = x[token + first + i - 3, column]
        for i in cutlass.range_constexpr(5):
            if 2 * v_pair + i >= 3:
                v_window[i] = x[token + 2 * v_pair + i - 3, v_column]
        for u in cutlass.range_constexpr(2):
            gate_owned[u] = gate[token + TH * section + partition + 2 * u, head * 128 + ch]
        if is_beta:
            beta_raw[0] = beta[token + thread - 128, head]
        if cutlass.const_expr(self.trigger_fence):
            fenced_trigger()
        else:
            cute.arch.griddepcontrol_launch_dependents()

        def conv(taps, start, weights4):  # stock nv_cutedsl FMA order (first product contracted)
            values = [Float32(taps[start + i]) for i in range(4)]
            acc = fma(values[0], Float32(weights4[0]), values[1] * Float32(weights4[1]))
            for i in cutlass.range_constexpr(2, 4):
                acc = fma(values[i], Float32(weights4[i]), acc)
            return acc

        qk = cute.make_rmem_tensor(TP, Float32)
        squares = cute.make_rmem_tensor(TP, Float32)
        for u in cutlass.range_constexpr(TP):
            acc = conv(window, u, coeff)
            qk[u] = acc * self.activation(acc)
            squares[u] = qk[u] * qk[u]
        total, index = scatter_reduce(squares, lane, TP)  # lane group 8 j owns token first + j's warp total
        if lane % (32 // TP) == 0:
            reductions[warp % 4 + 4 * section + 8 * (first + index)] = total  # base slots (q 0-3, k 4-7)
        subset_barrier(2 + section, 128)  # the section's four warps
        normalized = cute.make_rmem_tensor(TP, Float32)
        factor = zeros(1, Float32)
        for u in cutlass.range_constexpr(TP):
            r = lambda i, u=u: reductions[8 * (first + u) + 4 * section + i]
            factor[0] = cute.math.rsqrt((r(0) + r(2)) + (r(1) + r(3)) + 1.0e-6, fastmath=True)
            if section == 0:
                factor[0] = factor[0] * scale
            normalized[u] = qk[u] * factor[0]
        # 16-byte packets: lane 4j + d gathers token first + d's channels 4j .. 4j + 3 (step k: every lane offers its
        # token (lane + k) % 4 value and reads lane 4j + (lane - k) % 4), then sends the packet to both CTAs.
        packet = cute.make_rmem_tensor(4, Float32)
        offer = zeros(1, Float32)
        for k in cutlass.range_constexpr(4):
            pick = (lane + k) & 3
            offer[0] = normalized[0]
            for d in cutlass.range_constexpr(1, TP):
                if pick == d:
                    offer[0] = normalized[d]
            got = cute.arch.shuffle_sync(offer[0], (lane & 28) | ((lane - k) & 3))
            place_e = (lane - k) & 3
            for e in cutlass.range_constexpr(4):
                if place_e == e:
                    packet[e] = got
        base_q = Int32(shared_address(shared_q.iterator))
        base_k = Int32(shared_address(shared_k.iterator))
        address = base_q + section * (base_k - base_q) + 4 * (128 * (first + (lane & 3)) + (ch & 124))
        words = tuple(packet[e] for e in range(4))
        if cutlass.const_expr(self.early):  # (otherwise the CTA waited at entry, after setup)
            cute.arch.cluster_wait()
        for d in cutlass.range_constexpr(self.cluster):
            push_float(address, words, qk_barrier, d)
        # Side outputs of the formed tokens (stock values): raw K ring, q / k conv windows.
        for u in cutlass.range_constexpr(TP):
            if section == 1:
                args[20][Int64(slot), head, first + u, ch] = BFloat16(qk[u])
            for i in cutlass.range_constexpr(3):
                args[18][Int64(scratch_row), first + u, column, i] = window[u + i + 1]
        for j in cutlass.range_constexpr(2):
            t_v = 2 * v_pair + j
            v_acc = conv(v_window, j, v_coeff)
            v = v_acc * self.activation(v_acc)
            shared_v[v_row + rows * t_v] = v
            args[19][Int64(slot), head, t_v, value_base + v_row] = BFloat16(v)
            for i in cutlass.range_constexpr(3):
                args[18][Int64(scratch_row), t_v, v_column, i] = v_window[j + i + 1]
        if is_beta:
            beta_act = self.activation(Float32(beta_raw[0]))
            shared_beta[thread - 128] = beta_act
            if (thread - 128) // TP == partition:
                args[22][Int64(slot), head, thread - 128] = beta_act
        subset_barrier(1, MAIN)  # main warps only (the gate warps never join)
        wait(qk_barrier)  # every token's normalized q / k rows (acquire.cluster)
        return gate_owned

    @cute.jit
    def prologue_tsp4(self, args, place, buffers, conv_page):
        """Opt-in "tsp4": ``prologue_tsp2`` at 4-CTA clusters (32 value rows per CTA, Q8). CTA p forms
        tokens 2p, 2p + 1 once for the cluster: q (threads 0-127) / k (128-255) of channel thread % 128 over window
        indices 2p .. 2p + 4 (conv taps below 3, input rows after), SiLU, per token the warp's sum of squares (one
        reduce-scatter of the two tokens' squares over the same xor-butterfly levels 16 .. 1 as the base "rs" tree and
        ``warp_sum``, so the same sums), the four warp sums -> the token's factor, and stores the normalized rows into all
        four CTAs' shared q / k: lane 4j + d of a warp holds the 16-byte packet of token 2p + (d % 2), channels 4j .. 4j + 3,
        and sends it to CTAs d // 2 and d // 2 + 2, so each packet reaches each CTA exactly once (4 CTAs x 8 warps x 32
        lanes x 2 sends x 16 B / 4 = 1024 T bytes per CTA, completing ``qk_barrier``). It writes those tokens' raw-K ring
        rows, q / k conv windows and beta ring entries. Every main thread forms v of value row thread % 32 for token
        thread // 32 (shared v, raw-V ring, v windows; ``prologue_tsp``'s v form); threads 128-135 the betas of every
        token. The base path's arithmetic in the base order: bitwise. Returns the raw output gate of the thread's one
        owned token 4 half + p (activated at the output, as tsp)."""
        x, weights, beta, gate, scale = args[0], args[1], args[5], args[11], args[14]
        thread, partition, head, token, slot, scratch_row, value_base = place
        shared_q, shared_k, shared_v, shared_beta, reductions, qk_barrier = buffers
        T: cutlass.Constexpr = self.tokens
        MAIN: cutlass.Constexpr = self.main
        dim: cutlass.Constexpr = self.heads * 128
        rows: cutlass.Constexpr = self.rows
        TP: cutlass.Constexpr = T // self.cluster  # tokens formed per CTA (2)
        TH: cutlass.Constexpr = T // self.halves  # tokens per output half (4)
        REP: cutlass.Constexpr = 4 // TP  # lanes of a 4-lane group holding the same packet (2)
        lane, warp = thread % 32, thread // 32
        ch, section = thread % 128, thread // 128
        column = section * dim + head * 128 + ch  # this thread's q / k channel
        first = partition * TP  # first token formed by this CTA
        v_token, v_row = thread // rows, thread % rows  # v of token thread // 32 (every main thread: rows * T == MAIN)
        v_column = 2 * dim + head * 128 + value_base + v_row
        is_beta = (thread >= 128) & (thread < 128 + T)
        window = zeros(TP + 3, BFloat16)  # window index first + i: conv tap if < 3, else input row - 3
        v_window = zeros(4, BFloat16)  # window index v_token + i
        coeff, v_coeff = zeros(4, weights.element_type), zeros(4, weights.element_type)
        beta_raw = zeros(1, beta.element_type)
        for i in cutlass.range_constexpr(TP + 3):
            if first + i < 3:
                window[i] = conv_page[column, first + i]
        for i in cutlass.range_constexpr(4):
            coeff[i] = weights[section, i, head * 128 + ch]
            v_coeff[i] = weights[2, i, head * 128 + value_base + v_row]
        for i in cutlass.range_constexpr(3):
            if v_token + i < 3:
                v_window[i] = conv_page[v_column, v_token + i]
        cute.arch.griddepcontrol_wait()
        for i in cutlass.range_constexpr(TP + 3):
            if first + i >= 3:
                window[i] = x[token + first + i - 3, column]
        for i in cutlass.range_constexpr(4):
            if v_token + i >= 3:
                v_window[i] = x[token + v_token + i - 3, v_column]
        gate_owned = gate[token + TH * section + partition, head * 128 + ch]
        if is_beta:
            beta_raw[0] = beta[token + thread - 128, head]
        if cutlass.const_expr(self.trigger_fence):
            fenced_trigger()
        else:
            cute.arch.griddepcontrol_launch_dependents()

        def conv(taps, start, weights4):  # stock nv_cutedsl FMA order (first product contracted)
            values = [Float32(taps[start + i]) for i in range(4)]
            acc = fma(values[0], Float32(weights4[0]), values[1] * Float32(weights4[1]))
            for i in cutlass.range_constexpr(2, 4):
                acc = fma(values[i], Float32(weights4[i]), acc)
            return acc

        qk = cute.make_rmem_tensor(TP, Float32)
        squares = cute.make_rmem_tensor(TP, Float32)
        for u in cutlass.range_constexpr(TP):
            acc = conv(window, u, coeff)
            qk[u] = acc * self.activation(acc)
            squares[u] = qk[u] * qk[u]
        total, index = scatter_reduce(squares, lane, TP)  # lane group 16 j owns token first + j's warp total
        if lane % (32 // TP) == 0:
            reductions[warp % 4 + 4 * section + 8 * (first + index)] = total  # base slots (q 0-3, k 4-7)
        subset_barrier(2 + section, 128)  # the section's four warps
        normalized = cute.make_rmem_tensor(TP, Float32)
        factor = zeros(1, Float32)
        for u in cutlass.range_constexpr(TP):
            r = lambda i, u=u: reductions[8 * (first + u) + 4 * section + i]
            factor[0] = cute.math.rsqrt((r(0) + r(2)) + (r(1) + r(3)) + 1.0e-6, fastmath=True)
            if section == 0:
                factor[0] = factor[0] * scale
            normalized[u] = qk[u] * factor[0]
        # 16-byte packets: lane 4j + d gathers token first + d % TP's channels 4j .. 4j + 3 (step k: every lane offers its
        # token ((lane + k) % 4) % TP value and reads lane 4j + (lane - k) % 4), then sends it to REP-strided CTAs.
        packet = cute.make_rmem_tensor(4, Float32)
        offer = zeros(1, Float32)
        for k in cutlass.range_constexpr(4):
            pick = ((lane + k) & 3) % TP
            offer[0] = normalized[0]
            for d in cutlass.range_constexpr(1, TP):
                if pick == d:
                    offer[0] = normalized[d]
            got = cute.arch.shuffle_sync(offer[0], (lane & 28) | ((lane - k) & 3))
            place_e = (lane - k) & 3
            for e in cutlass.range_constexpr(4):
                if place_e == e:
                    packet[e] = got
        base_q = Int32(shared_address(shared_q.iterator))
        base_k = Int32(shared_address(shared_k.iterator))
        address = base_q + section * (base_k - base_q) + 4 * (128 * (first + (lane & 3) % TP) + (ch & 124))
        words = tuple(packet[e] for e in range(4))
        replica = (lane & 3) // TP  # which copy of the packet this lane holds (0 .. REP - 1)
        if cutlass.const_expr(self.early):  # (otherwise the CTA waited at entry, after setup)
            cute.arch.cluster_wait()
        for i in cutlass.range_constexpr(self.cluster // REP):
            push_float(address, words, qk_barrier, replica + REP * i)
        # Side outputs of the formed tokens (stock values): raw K ring, q / k conv windows.
        for u in cutlass.range_constexpr(TP):
            if section == 1:
                args[20][Int64(slot), head, first + u, ch] = BFloat16(qk[u])
            for i in cutlass.range_constexpr(3):
                args[18][Int64(scratch_row), first + u, column, i] = window[u + i + 1]
        v_acc = conv(v_window, 0, v_coeff)
        v = v_acc * self.activation(v_acc)
        shared_v[v_row + rows * v_token] = v
        args[19][Int64(slot), head, v_token, value_base + v_row] = BFloat16(v)
        for i in cutlass.range_constexpr(3):
            args[18][Int64(scratch_row), v_token, v_column, i] = v_window[i + 1]
        if is_beta:
            beta_act = self.activation(Float32(beta_raw[0]))
            shared_beta[thread - 128] = beta_act
            if (thread - 128) // TP == partition:
                args[22][Int64(slot), head, thread - 128] = beta_act
        subset_barrier(1, MAIN)  # main warps only (the gate warp never joins)
        wait(qk_barrier)  # every token's normalized q / k rows (acquire.cluster; tsp4)
        return gate_owned

    @cute.jit
    def chunk_outputs(
        self, history, operands, chunk_smem, barriers, thread, lanes, base, gram_barrier
    ):
        """Opt-in "chunk": the ReplaySSM verify's T outputs from the checkpoint S0 in one
        chunked (WY) step instead of T serial token updates. Per key column c,
        D_t = d_0 ... d_t (decay prefix products, in token order) and, for s <= t,
        E_st = d_(s+1) ... d_t (empty product 1). Per state row, P_t = (S0 D_t) . k_t
        and R_t = (S0 D_t) . q_t; per head, A_ts = (k_s E_st) . k_t (s < t) and
        B_ts = (k_s E_st) . q_t (s <= t). Then per row
        u_t = (v_t - (P_t + sum_s<t A_ts u_s)) beta_t (``update``'s v_new) and
        o_t = R_t + sum_s<=t B_ts u_s (its query dot of the updated state): the token
        loop's quantities, from the same factors multiplied and summed in another order
        (the state is decayed before its dots, as the loop does; no division). Verify
        writes no state (ReplaySSM), so no S_t is materialized."""
        shared_q, shared_k, shared_decay, shared_v, shared_beta = operands
        products, gram = chunk_smem[:2]
        gate_barrier, barrier, shared_o = barriers
        group, group_lane = lanes
        T: cutlass.Constexpr = self.tokens
        MAIN: cutlass.Constexpr = self.main
        L: cutlass.Constexpr = self.lanes_per_row
        W: cutlass.Constexpr = 2 * T  # dots per row (P, R) and per Gram row (A, B)
        lane, warp = thread % 32, thread // 32
        if cutlass.const_expr(self.cgram):  # q / k normalized: the gate warp may read them
            named_barrier(CG_BARRIER, MAIN + 32, arrive=True)
        state = None
        if cutlass.const_expr(self.csplit):  # before the decays arrive
            state = self.cs_remap(history, chunk_smem[2], thread)
        wait(gate_barrier)
        if cutlass.const_expr(self.csplit):
            self.cs_products(state, operands, chunk_smem, thread)
        if cutlass.const_expr(not self.csplit):
            # 1a. P_t, R_t of this thread's state row (its 4 * chunks columns), then the row's
            # L lanes reduce-scatter the 2T sums.
            sums = cute.make_rmem_tensor(W, Float32)
            prefix = cute.make_rmem_tensor((self.chunks, 4), Float32)
            for t in cutlass.range_constexpr(T):
                pk = cute.make_rmem_tensor(self.chunks, Float32)
                pq = cute.make_rmem_tensor(self.chunks, Float32)
                for j in cutlass.range_constexpr(self.chunks):
                    col = 128 * t + 4 * (group_lane + L * j)
                    d4 = load_f32x4(shared_decay.iterator + col, shared=True)
                    k4 = load_f32x4(shared_k.iterator + col, shared=True)
                    q4 = load_f32x4(shared_q.iterator + col, shared=True)
                    for e in cutlass.range_constexpr(4):
                        prefix[j, e] = d4[e] if t == 0 else prefix[j, e] * d4[e]
                    decayed = [history[0, j, e] * prefix[j, e] for e in range(4)]
                    pk[j] = dot(decayed, (k4[e] for e in range(4)), ftz=self.fast_math)
                    pq[j] = dot(decayed, (q4[e] for e in range(4)), ftz=self.fast_math)
                for shift in cutlass.range_constexpr(self.chunks.bit_length() - 2, -1, -1):
                    for j in cutlass.range_constexpr(1 << shift):
                        pk[j] += pk[j + (1 << shift)]
                        pq[j] += pq[j + (1 << shift)]
                sums[t], sums[T + t] = pk[0], pq[0]
            selected = scatter_reduce_lanes(sums, group_lane, W, L)
            for i in cutlass.range_constexpr(W // L):
                products[W * group + selected + i] = sums[i]
        if cutlass.const_expr(not (self.cgram or self.gbal)):
            # 1b. Gram row s = warp (T warps): lane owns columns 4 lane ... 4 lane + 3.
            source = warp
            k_src = load_f32x4(shared_k.iterator + (128 * source + 4 * lane), shared=True)
            ratio = [Float32(1) for _ in range(4)]
            grams = cute.make_rmem_tensor(W, Float32)
            for t in cutlass.range_constexpr(T):
                col = 128 * t + 4 * lane
                d4 = load_f32x4(shared_decay.iterator + col, shared=True)
                k4 = load_f32x4(shared_k.iterator + col, shared=True)
                q4 = load_f32x4(shared_q.iterator + col, shared=True)
                later = t > source
                for e in cutlass.range_constexpr(4):
                    ratio[e] = ratio[e] * d4[e] if later else ratio[e]
                scaled = [k_src[e] * ratio[e] for e in range(4)]
                grams[t] = dot(scaled, (k4[e] for e in range(4)), ftz=self.fast_math)
                grams[T + t] = dot(scaled, (q4[e] for e in range(4)), ftz=self.fast_math)
            total, index = scatter_reduce(grams, lane, W)
            if lane % 2 == 0:
                gram[W * warp + index] = total
        cute.arch.barrier(barrier_id=1, number_of_threads=MAIN)
        if cutlass.const_expr(self.early and not self.tsp):  # before this CTA's first DSMEM send (tsp: waited)
            cute.arch.cluster_wait()
        if cutlass.const_expr(self.cgram or self.csplit):
            if cutlass.const_expr(self.csplit):  # one solver warp: lane r = row r
                solver, row, row_products = thread < 16, thread, chunk_smem[4].iterator
                row_products = row_products + 20 * thread
            else:
                solver, row = group_lane == 0, group
                row_products = products.iterator + W * group
            fed = (gram, gram_barrier, shared_v, shared_beta)
            pushed = (barrier, shared_o, base)
            self.solve_rows(solver, row, row_products, fed, pushed)
        if cutlass.const_expr(not (self.cgram or self.csplit)):
            # 3. Triangular solve and outputs, one thread per state row; DSMEM to token owners.
            if group_lane == 0:
                p = []
                for i in cutlass.range_constexpr(W // 4):
                    p += list(load_f32x4(products.iterator + (W * group + 4 * i), shared=True))
                acc, out = p[:T], p[T:]
                for s in cutlass.range_constexpr(T):
                    g = []
                    for i in cutlass.range_constexpr(W // 4):
                        g += list(load_f32x4(gram.iterator + (W * s + 4 * i), shared=True))
                    u = (shared_v[self.rows * s + group] - acc[s]) * shared_beta[s]
                    for t in cutlass.range_constexpr(s + 1, T):
                        acc[t] = fma(g[t], u, acc[t], ftz=self.fast_math)
                    for t in cutlass.range_constexpr(s, T):
                        out[t] = fma(g[T + t], u, out[t], ftz=self.fast_math)
                position = self.output_position(base + group)
                for t in cutlass.range_constexpr(T):
                    owner, bar = t % self.cluster, advance(barrier, t // self.cluster)
                    dst = advance(shared_o.iterator, 128 * t) + position
                    push_float(dst, out[t], bar, owner)

    @cute.jit
    def wy_outputs(self, history, operands, wy_smem, barriers, thread, lanes, base):
        """Opt-in "wy": ``chunk_outputs``' chunked (WY) step for 32 / 64 / 128-row CTAs, in the
        dot-basis form of a chunked KDA verify. Per key column c, D_t = d_0 ... d_t and the bases D_t k_t, D_t q_t are
        formed once per CTA (threads 0-127, one column each, decay prefix in token order) while warps 4-7 form the
        head's Gram rows A_st = (k_s E_st) . k_t, B_st = (k_s E_st) . q_t (``chunk_outputs`` 1b, two rows per warp).
        Per state row, P_t = S0 . (D_t k_t) and R_t = S0 . (D_t q_t): FP32 FMA chains over the thread's columns (chunk
        j, then element e, in order; packed f32x2, lanewise IEEE), then a reduce-scatter over the row's lanes into
        shared products. Then one solver thread per row: u_t = (v_t - (P_t + sum_s<t A_st u_s)) beta_t and
        o_t = R_t + sum_s<=t B_st u_s (``chunk_outputs`` 3), and o_t to token t's owner. The token loop's quantities
        from the same factors in another order (the decay prefix multiplies k_t / q_t, not the state row; no
        division). Verify writes no state (ReplaySSM), so no S_t is materialized."""
        shared_q, shared_k, shared_decay, shared_v, shared_beta = operands
        products, gram, bases = wy_smem
        gate_barrier, barrier, shared_o = barriers
        group, group_lane = lanes
        T: cutlass.Constexpr = self.tokens
        MAIN: cutlass.Constexpr = self.main
        L: cutlass.Constexpr = self.lanes_per_row
        J: cutlass.Constexpr = self.chunks
        P: cutlass.Constexpr = self.passes
        W: cutlass.Constexpr = 2 * T  # dots per row (P, R) and per Gram row (A, B)
        SLAB: cutlass.Constexpr = 4 * J * L  # 16-byte slots per four basis entries (128 columns)
        lane, warp = thread % 32, thread // 32
        wait(gate_barrier)
        if thread < 128:
            # Column c = thread: slot ((m J + j) 4 + e) L + l of the bases holds entries 4m .. 4m + 3 of column
            # 4 (l + L j) + e, so a row's L lanes read L consecutive 16-byte slots for each (j, e, m).
            unit = thread // 4
            slot = ((unit // L) * 4 + thread % 4) * L + unit % L
            vals = cute.make_rmem_tensor(W, Float32)
            cumulative = shared_decay[thread]
            for t in cutlass.range_constexpr(T):
                if cutlass.const_expr(t > 0):
                    cumulative = cumulative * shared_decay[128 * t + thread]
                vals[t] = cumulative * shared_k[128 * t + thread]
                vals[T + t] = cumulative * shared_q[128 * t + thread]
            for m in cutlass.range_constexpr(W // 4):
                pointer = bases.iterator + 4 * (SLAB * m + slot)
                store_shared_words(pointer, tuple(float_bits(vals[4 * m + e]) for e in range(4)))
        else:
            for r in cutlass.range_constexpr(2):  # Gram row s = 2 (warp - 4) + r: lane owns columns 4 lane .. + 3
                source = 2 * (warp - 4) + r
                k_src = load_f32x4(shared_k.iterator + (128 * source + 4 * lane), shared=True)
                ratio = [Float32(1) for _ in range(4)]
                grams = cute.make_rmem_tensor(W, Float32)
                for t in cutlass.range_constexpr(T):
                    col = 128 * t + 4 * lane
                    d4 = load_f32x4(shared_decay.iterator + col, shared=True)
                    k4 = load_f32x4(shared_k.iterator + col, shared=True)
                    q4 = load_f32x4(shared_q.iterator + col, shared=True)
                    later = t > source
                    for e in cutlass.range_constexpr(4):
                        ratio[e] = ratio[e] * d4[e] if later else ratio[e]
                    scaled = [k_src[e] * ratio[e] for e in range(4)]
                    grams[t] = dot(scaled, (k4[e] for e in range(4)), ftz=self.fast_math)
                    grams[T + t] = dot(scaled, (q4[e] for e in range(4)), ftz=self.fast_math)
                total, index = scatter_reduce(grams, lane, W)
                if lane % 2 == 0:
                    gram[W * source + index] = total
        cute.arch.barrier(barrier_id=1, number_of_threads=MAIN)
        if cutlass.const_expr(self.early and not self.tsp2):  # before this CTA's first DSMEM send (tsp2: waited)
            cute.arch.cluster_wait()
        # P_t (t < T) and R_t (T + t) of the thread's rows p groups + group over its columns.
        dots = cute.make_rmem_tensor((P, W), Float32)
        fz = self.fast_math
        for j in cutlass.range_constexpr(J):
            for e in cutlass.range_constexpr(4):
                b = []
                for m in cutlass.range_constexpr(W // 4):
                    pointer = bases.iterator + 4 * (SLAB * m + (j * 4 + e) * L + group_lane)
                    b += list(load_f32x4(pointer, shared=True))
                for p in cutlass.range_constexpr(P):
                    h0 = history[p, j, e]
                    for i in cutlass.range_constexpr(W // 2):
                        pair = (b[2 * i], b[2 * i + 1])
                        if cutlass.const_expr(j == 0 and e == 0):
                            dots[p, 2 * i], dots[p, 2 * i + 1] = mul2_ftz((h0, h0), pair, ftz=fz)
                        else:
                            prev = (dots[p, 2 * i], dots[p, 2 * i + 1])
                            dots[p, 2 * i], dots[p, 2 * i + 1] = fma2_ftz((h0, h0), pair, prev, ftz=fz)
        for p in cutlass.range_constexpr(P):
            sums = cute.make_rmem_tensor(W, Float32)
            for n in cutlass.range_constexpr(W):
                sums[n] = dots[p, n]
            selected = scatter_reduce_lanes(sums, group_lane, W, L)
            row = p * self.groups + group
            for i in cutlass.range_constexpr(W // L):
                products[W * row + selected + i] = sums[i]
        cute.arch.barrier(barrier_id=1, number_of_threads=MAIN)
        if thread < self.rows:  # the solve, one thread per state row; outputs to token t's owner
            row = thread
            pr = []
            for i in cutlass.range_constexpr(W // 4):
                pr += list(load_f32x4(products.iterator + (W * row + 4 * i), shared=True))
            acc, out = pr[:T], pr[T:]
            for s in cutlass.range_constexpr(T):
                g = []
                for i in cutlass.range_constexpr(W // 4):
                    g += list(load_f32x4(gram.iterator + (W * s + 4 * i), shared=True))
                u = (shared_v[self.rows * s + row] - acc[s]) * shared_beta[s]
                for t in cutlass.range_constexpr(s + 1, T):
                    acc[t] = fma(g[t], u, acc[t], ftz=fz)
                for t in cutlass.range_constexpr(s, T):
                    out[t] = fma(g[T + t], u, out[t], ftz=fz)
            position = self.output_position(base + row)
            for t in cutlass.range_constexpr(T):
                dst = advance(shared_o.iterator, 128 * t) + position
                if cutlass.const_expr(self.local):  # own CTA: store, then this sender's arrival (setup counts rows)
                    put_float(dst, out[t])
                    arrive_release_cta(advance(barrier, t))
                else:
                    owner, bar = t % self.cluster, advance(barrier, t // self.cluster)
                    push_float(dst, out[t], bar, owner)

    @cute.jit
    def gram_balanced(self, operands, gram_parts, thread):
        """Opt "gb" (with "cs", not "cg"): the head's Gram with balanced work per warp. Warp w takes
        rows a = w % 4 and b = T - 1 - a (together 2T + 1 token steps for every warp) over key
        columns 64 (w // 4) .. + 63 (2 per lane) and only the 16 entries the solve reads (A_st
        for t > s, B_st for t >= s), with E_st = d_(s+1) ... d_t built in token order as 1b. A
        warp-uniform switch on a makes the row indices static (no selects). One scatter_reduce
        of the 16 sums; lane pairs store their total into Gram part [half][s][2T]."""
        T: cutlass.Constexpr = self.tokens
        W: cutlass.Constexpr = 2 * T
        lane, warp = thread % 32, thread // 32
        column = 64 * (warp // 4) + 2 * lane
        entries = cute.make_rmem_tensor(W, Float32)
        quarter = warp % 4
        for a in cutlass.range_constexpr(4):
            if quarter == a:
                self.gram_pair(a, T - 1 - a, operands, column, entries)
        total, index = scatter_reduce(entries, lane, W)
        # Packed order: B_a (t = a ..), A_a (t = a + 1 ..), B_b (t = b ..), A_b (t = b + 1 ..).
        a_row = quarter
        b_row = T - 1 - quarter
        first_two = (T - a_row) + (T - 1 - a_row)
        in_b = index >= first_two
        row = b_row if in_b else a_row
        k = index - first_two if in_b else index
        count_b = T - row  # B entries of that row (t = row .. T - 1)
        slot = row + 1 + (k - count_b) if k >= count_b else T + row + k
        if lane % 2 == 0:
            gram_parts[W * T * (warp // 4) + W * row + slot] = total

    def gram_pair(self, a, b, operands, column, entries):
        # Plain trace-time helper of ``gram_balanced`` (static a < b; Python control flow only).
        shared_q, shared_k, shared_decay = operands[:3]
        T = self.tokens
        fz = self.fast_math
        pair = lambda buf, t: load_f32x2(buf.iterator + (128 * t + column), shared=True)
        order = [("B", a, t) for t in range(a, T)] + [("A", a, t) for t in range(a + 1, T)]
        order += [("B", b, t) for t in range(b, T)] + [("A", b, t) for t in range(b + 1, T)]
        slot = {key: i for i, key in enumerate(order)}
        own = {a: pair(shared_k, a), b: pair(shared_k, b)}
        span = {a: None, b: None}
        for t in range(a, T):
            k_t = own[t] if t in own else pair(shared_k, t)
            q_t = pair(shared_q, t)
            d_t = pair(shared_decay, t) if t > a else None
            for s in (a, b):
                if t < s:
                    continue
                if t > s:  # E_st = d_(s+1) ... d_t, in token order
                    span[s] = d_t if span[s] is None else tuple(span[s][e] * d_t[e] for e in range(2))
                weighted = own[s] if span[s] is None else tuple(own[s][e] * span[s][e] for e in range(2))
                entries[slot["B", s, t]] = dot(weighted, q_t, ftz=fz)
                if t > s:
                    entries[slot["A", s, t]] = dot(weighted, k_t, ftz=fz)

    @cute.jit
    def cs_remap(self, history, remap, thread):
        """Opt "cs": the CTA's 16 x 128 checkpoint rows from the row layout (16 lanes x 8
        columns per row) to the column-split one (warp w = key columns 16 w .. 16 w + 15 of
        all rows; lane = rows 2 (lane // 4) + {0, 1} x columns 16 w + 4 (lane % 4) + 0..3,
        the row it keeps in ``cs_products``' reduce-scatter first), through shared rows of 132
        words (16-byte accesses without bank conflicts)."""
        group, quad = thread // 16, thread % 16
        for j in cutlass.range_constexpr(2):
            pointer = remap.iterator + (132 * group + 64 * j + 4 * quad)
            store_shared_words(pointer, tuple(float_bits(history[0, j, e]) for e in range(4)))
        cute.arch.barrier(barrier_id=1, number_of_threads=self.main)
        lane, warp = thread % 32, thread // 32
        column = 16 * warp + 4 * (lane % 4)
        state = cute.make_rmem_tensor((2, 4), Float32)
        for h in cutlass.range_constexpr(2):  # lane-relative: the kept row first (cs_products)
            row = 2 * (lane // 4) + (h ^ ((lane >> 1) & 1))
            values = load_f32x4(remap.iterator + (132 * row + column), shared=True)
            for e in cutlass.range_constexpr(4):
                state[h, e] = values[e]
        return state

    @cute.jit
    def cs_products(self, state, operands, chunk_smem, thread):
        """Opt "cs": P_t / R_t of every row in the column-split layout (``cs_remap``). A lane
        dots its 4 key columns of two rows (state decayed first, as ``chunk_outputs`` 1a);
        the 4 lanes of a row pair reduce-scatter their 2 x 2T sums (2T / 2 totals each);
        each warp stores its partials [warp, row, 2T]; thread 2T r + i adds row r's eight
        warp partials of value i in a fixed pairwise order into P/R [row, 20]."""
        shared_q, shared_k, shared_decay = operands[:3]
        partials, row_sums = chunk_smem[3], chunk_smem[4]
        T: cutlass.Constexpr = self.tokens
        W: cutlass.Constexpr = 2 * T
        lane, warp = thread % 32, thread // 32
        column = 16 * warp + 4 * (lane % 4)
        # Lane-relative order, so the reduce-scatter needs no selects: lanes with lane bit 0
        # set dot q first (their first 2T / 2 sums are R), and ``cs_remap`` put the row a lane
        # keeps (lane bit 1) first. Shared addresses as integers (one select of a base).
        flip = Uint32(lane & 1)
        k_base, q_base = shared_address(shared_k.iterator), shared_address(shared_q.iterator)
        first_base = k_base + flip * (q_base - k_base)
        second_base = q_base + k_base - first_base
        sums = cute.make_rmem_tensor(2 * W, Float32)  # [kept row, other row][first 2T/2, second]
        running = cute.make_rmem_tensor(4, Float32)  # D_t of the lane's columns
        for t in cutlass.range_constexpr(T):
            col = 128 * t + column
            d4 = load_f32x4(shared_decay.iterator + col, shared=True)
            a4 = load_f32x4(first_base + Uint32(4 * col), shared=True)
            b4 = load_f32x4(second_base + Uint32(4 * col), shared=True)
            if cutlass.const_expr(self.cs_packed):
                # Opt "csp": the same products as f32x2 pairs of columns (0, 1) and (2, 3); each
                # dot sums its column pairs, then the two lanes ((c0 + c2) + (c1 + c3)).
                ftz = self.fast_math
                for e in cutlass.range_constexpr(0, 4, 2):
                    if cutlass.const_expr(t == 0):
                        running[e], running[e + 1] = d4[e], d4[e + 1]
                    else:
                        pair = (running[e], running[e + 1])
                        running[e], running[e + 1] = mul2_ftz(pair, (d4[e], d4[e + 1]), ftz=ftz)
                for h in cutlass.range_constexpr(2):
                    low = mul2_ftz((state[h, 0], state[h, 1]), (running[0], running[1]), ftz=ftz)
                    high = mul2_ftz((state[h, 2], state[h, 3]), (running[2], running[3]), ftz=ftz)
                    for slot, v4 in ((W * h + t, a4), (W * h + T + t, b4)):
                        acc2 = mul2_ftz(low, (v4[0], v4[1]), ftz=ftz)
                        acc2 = fma2_ftz(high, (v4[2], v4[3]), acc2, ftz=ftz)
                        sums[slot] = acc2[0] + acc2[1]
            else:
                for e in cutlass.range_constexpr(4):
                    running[e] = d4[e] if t == 0 else running[e] * d4[e]
                for h in cutlass.range_constexpr(2):
                    decayed = [state[h, e] * running[e] for e in range(4)]
                    sums[W * h + t] = dot(
                        decayed, (a4[e] for e in range(4)), ftz=self.fast_math
                    )
                    sums[W * h + T + t] = dot(
                        decayed, (b4[e] for e in range(4)), ftz=self.fast_math
                    )
        for i in cutlass.range_constexpr(W):  # lane bit 1: keep the first row, send the other
            sums[i] = sums[i] + cute.arch.shuffle_sync_bfly(sums[W + i], 2)
        for i in cutlass.range_constexpr(W // 2):  # lane bit 0: keep the first operand's sums
            sums[i] = sums[i] + cute.arch.shuffle_sync_bfly(sums[W // 2 + i], 1)
        picked = W * ((lane >> 1) & 1) + W // 2 * (lane & 1)  # (row of the pair, P or R)
        first = 16 * W * warp + 2 * W * (lane // 4) + picked
        for i in cutlass.range_constexpr(0, W // 2, 4):
            words = tuple(float_bits(sums[i + e]) for e in range(4))
            store_shared_words(partials.iterator + (first + i), words)
        if cutlass.const_expr(self.gbal):
            self.gram_balanced(operands, chunk_smem[5], thread)
        cute.arch.barrier(barrier_id=1, number_of_threads=self.main)
        if cutlass.const_expr(self.gbal):  # Gram [s, 2T] = half 0 + half 1 (the entries the solve reads)
            gram_parts, gram = chunk_smem[5], chunk_smem[1]
            if thread < W * T:
                s_row, j = thread // W, thread % W
                used = ((j < T) & (j > s_row)) | ((j >= T) & (j - T >= s_row))
                if used:
                    gram[thread] = gram_parts[thread] + gram_parts[W * T + thread]
        parts = [partials[16 * W * w + thread] for w in range(8)]
        total = ((parts[0] + parts[1]) + (parts[2] + parts[3])) + (
            (parts[4] + parts[5]) + (parts[6] + parts[7])
        )
        row_sums[20 * (thread // W) + thread % W] = total

    @cute.jit
    def gram_row(self, operands, gate_barrier, targets, partition, lane):
        """Opt "cg" (the gate warp, after its decay multicast): Gram row s = partition of
        the head, A_st = (k_s E_st) . k_t and B_st = (k_s E_st) . q_t with E_st = d_(s+1) ...
        d_t, in ``chunk_outputs`` 1b's order (same values), sent to every CTA of the cluster
        into its Gram [s, 2T], completing on its ``gram_barrier`` (8 senders x 2T x 4 bytes).
        q / k are read after named barrier 2 (the main warps arrive after normalizing them),
        the decays after the gate barrier."""
        shared_q, shared_k, shared_decay = operands
        gram, gram_barrier = targets
        T: cutlass.Constexpr = self.tokens
        W: cutlass.Constexpr = 2 * T
        named_barrier(CG_BARRIER, self.main + 32)
        wait(gate_barrier)
        own = partition
        k_own = load_f32x4(shared_k.iterator + (128 * own + 4 * lane), shared=True)
        span = [Float32(1) for _ in range(4)]
        entries = cute.make_rmem_tensor(W, Float32)
        for t in cutlass.range_constexpr(T):
            col = 128 * t + 4 * lane
            d4 = load_f32x4(shared_decay.iterator + col, shared=True)
            k4 = load_f32x4(shared_k.iterator + col, shared=True)
            q4 = load_f32x4(shared_q.iterator + col, shared=True)
            after = t > own
            for e in cutlass.range_constexpr(4):
                span[e] = span[e] * d4[e] if after else span[e]
            weighted = [k_own[e] * span[e] for e in range(4)]
            entries[t] = dot(weighted, (k4[e] for e in range(4)), ftz=self.fast_math)
            entries[T + t] = dot(weighted, (q4[e] for e in range(4)), ftz=self.fast_math)
        total, index = scatter_reduce(entries, lane, W)
        destination = gram.iterator + (W * own + index)
        half: cutlass.Constexpr = self.cluster // 2  # lanes 2i and 2i + 1 hold one total
        for i in cutlass.range_constexpr(half):
            push_float(destination, total, gram_barrier, half * (lane % 2) + i)

    @cute.jit
    def solve_rows(self, solver, row, row_products, fed, pushed):
        """Opts "cg" / "cs": ``chunk_outputs``' triangular solve and output pushes (same
        arithmetic) for state row ``row`` with its P/R at ``row_products``, after the peers'
        Gram rows arrived ("cg")."""
        gram, gram_barrier, shared_v, shared_beta = fed
        barrier, shared_o, base = pushed
        T: cutlass.Constexpr = self.tokens
        W: cutlass.Constexpr = 2 * T
        if cutlass.const_expr(gram_barrier is not None):
            # Every main thread waits (not only the solver lanes): a poll loop inside the
            # lane-divergent solver branch made ptxas emit divergent-path copies of the later
            # warp-synchronous code; waiting here keeps warps converged.
            wait(gram_barrier)
        if solver:
            known = []
            for i in cutlass.range_constexpr(W // 4):
                known += list(load_f32x4(row_products + 4 * i, shared=True))
            pending, result = known[:T], known[T:]
            for s in cutlass.range_constexpr(T):
                coef = []
                for i in cutlass.range_constexpr(W // 4):
                    coef += list(load_f32x4(gram.iterator + (W * s + 4 * i), shared=True))
                u = (shared_v[self.rows * s + row] - pending[s]) * shared_beta[s]
                for t in cutlass.range_constexpr(s + 1, T):
                    pending[t] = fma(coef[t], u, pending[t], ftz=self.fast_math)
                for t in cutlass.range_constexpr(s, self.tokens):
                    result[t] = fma(coef[T + t], u, result[t], ftz=self.fast_math)
            position = self.output_position(base + row)
            for t in cutlass.range_constexpr(T):
                owner, bar = t % self.cluster, advance(barrier, t // self.cluster)
                dst = advance(shared_o.iterator, 128 * t) + position
                push_float(dst, result[t], bar, owner)

    @cute.jit
    def activation(self, value):
        # Stock TP8 uses the hardware approximate reciprocal, not div.approx.
        if cutlass.const_expr(self.fast_math):
            return cute.arch.rcp_approx(
                Float32(1) + cute.math.exp(-value, fastmath=True)
            )
        return sigmoid(value, self.fast_math)

    @cute.jit
    def output_norm_vp(self, shared_o, u, thread, lane, weight, gate, eps):
        """Opt "vp": ``output_norm`` on the padded owner-major layout shared_o[4 (position + position // 16) + u] (same values, same order)."""
        own = self.output_position(thread)
        raw_o = shared_o[4 * (own + (own >> 4)) + u]
        sums = cute.make_rmem_tensor(4, Float32)
        for g in cutlass.range_constexpr(4):
            base = lane % 8 * 16 + g * 4  # positions base .. base + 3 share base // 16 = lane % 8
            o4 = [shared_o[4 * (base + i + lane % 8) + u] for i in range(4)]
            s0, s1, s2, s3 = (o * o for o in o4)
            sums[g] = warp_sum((s0 + s2) + (s1 + s3), 8)
        sumsq = (sums[0] + sums[2]) + (sums[1] + sums[3])
        rstd = cute.math.rsqrt(sumsq * (1.0 / 128) + eps, fastmath=True)
        return raw_o * rstd * weight * gate

    @cute.jit
    def output_norm(self, shared_o, token, thread, lane, weight, gate, eps):
        row = shared_o.iterator + 128 * token
        raw_o = shared_o[128 * token + self.output_position(thread)]
        sums = cute.make_rmem_tensor(4, Float32)
        for g in cutlass.range_constexpr(4):
            base = lane % 8 * 16 + g * 4
            o4 = load_f32x4(row + base, shared=True)
            s0, s1, s2, s3 = (o * o for o in o4)
            sums[g] = warp_sum((s0 + s2) + (s1 + s3), 8)
        sumsq = (sums[0] + sums[2]) + (sums[1] + sums[3])
        rstd = cute.math.rsqrt(sumsq * (1.0 / 128) + eps, fastmath=True)
        return raw_o * rstd * weight * gate

    def widen(self, history, hraw):
        """BF16 state: the raw words loaded early (``hraw``), widened exactly to FP32 into ``history`` here, at the
        state's first use after phase 1, so no instruction waits on the (cold) state load before the PDL wait (as with
        the FP32 state, whose loads land directly in ``history``). No-op for an FP32 state."""
        if self.bf16_state:
            for p in range(self.passes):
                for j in range(self.chunks):
                    values = (*unpack_bf16(hraw[p, j, 0]), *unpack_bf16(hraw[p, j, 1]))
                    for e in range(4):
                        history[p, j, e] = values[e]

    @cute.jit
    def store_rows(self, history, state_row, span, lanes, when=None):
        """Store the passes in ``span`` of the CTA's state rows (``when``: predicated
        on it, without branches)."""
        (start, stop), (group, group_lane) = span, lanes
        for p in cutlass.range_constexpr(start, stop):
            row = p * self.groups + group
            if cutlass.const_expr(when is not None):
                for j in cutlass.range_constexpr(self.chunks):
                    column = 4 * (group_lane + self.lanes_per_row * j)
                    values = [history[p, j, e] for e in range(4)]
                    pointer = state_row(row % self.rows, column)
                    if cutlass.const_expr(self.bf16_state):
                        store_bf16x4(pointer, values, when=when & (row < self.rows))
                    else:
                        store_f32x4(pointer, values, when=when & (row < self.rows))
            elif row < self.rows:
                for j in cutlass.range_constexpr(self.chunks):
                    column = 4 * (group_lane + self.lanes_per_row * j)
                    pointer = state_row(row, column)
                    row_values = [history[p, j, e] for e in range(4)]
                    if cutlass.const_expr(self.bf16_state):
                        store_bf16x4(pointer, row_values, evict=self.evict)
                    elif cutlass.const_expr(self.evict):
                        store_evict_f32x4(pointer, row_values)
                    else:
                        store_f32x4(pointer, row_values)

    @cute.jit
    def update(self, history, dots, operands, lanes, span, staged, red=None):
        """Decay, delta-rule update and query dot of the passes in ``span``, whose rows
        ``staged`` (if given) first supplies from shared memory. ``red`` (opt "sred"):
        (partials tensor, this token's slot offset, thread) for the k-dot's reduction."""
        kr, qr, decays, shared_v, beta = operands
        (group, lane), (start, stop) = lanes, span
        if cutlass.const_expr(staged is not None):
            for p in cutlass.range_constexpr(start, stop):
                row = (p * self.groups + group) % self.rows
                for j in cutlass.range_constexpr(self.chunks):
                    column = 4 * (lane + self.lanes_per_row * j)
                    values = load_staged(staged.iterator + (row * 128 + column))
                    for e in cutlass.range_constexpr(4):
                        history[p, j, e] = values[e]
        packed: cutlass.Constexpr = self.packed or self.pkb  # "pkb": decay / update only (``row_dot`` keeps self.packed)
        for p in cutlass.range_constexpr(start, stop):
            for j in cutlass.range_constexpr(self.chunks):
                if cutlass.const_expr(packed):  # same products, f32x2 instructions
                    for e in cutlass.range_constexpr(0, 4, 2):
                        history[p, j, e], history[p, j, e + 1] = mul2_ftz(
                            (history[p, j, e], history[p, j, e + 1]),
                            (decays[j][e], decays[j][e + 1]),
                            ftz=self.fast_math,
                        )
                else:
                    for e in cutlass.range_constexpr(4):
                        history[p, j, e] *= decays[j][e]
            dots[p] = self.row_dot(history, kr, p)
        if cutlass.const_expr(red is None):
            for shift in reversed(range(self.lanes_per_row.bit_length() - 1)):
                for p in cutlass.range_constexpr(start, stop):
                    dots[p] += cute.arch.shuffle_sync_bfly(dots[p], 1 << shift)
        else:  # opt "sred" (one pass, 8 lanes per row): the butterfly's sum order from shared memory
            partials, slot, thread = red
            put_float(partials.iterator + (slot + thread), dots[start])
            cute.arch.sync_warp()
            row8 = partials.iterator + (slot + thread - lane)  # this row's 8 lane partials (32-byte aligned)
            lo, hi = load_f32x4(row8, shared=True), load_f32x4(row8 + 4, shared=True)  # lanes 0-3, 4-7
            dots[start] = ((lo[0] + hi[0]) + (lo[2] + hi[2])) + ((lo[1] + hi[1]) + (lo[3] + hi[3]))
        for p in cutlass.range_constexpr(start, stop):
            safe_row = (p * self.groups + group) % self.rows
            v_new = (shared_v[safe_row] - dots[p]) * beta
            for j in cutlass.range_constexpr(self.chunks):
                if cutlass.const_expr(packed):
                    for e in cutlass.range_constexpr(0, 4, 2):
                        history[p, j, e], history[p, j, e + 1] = fma2_ftz(
                            (kr[j, e], kr[j, e + 1]),
                            (v_new, v_new),
                            (history[p, j, e], history[p, j, e + 1]),
                            ftz=self.fast_math,
                        )
                else:
                    for e in cutlass.range_constexpr(4):
                        history[p, j, e] = fma(
                            kr[j, e], v_new, history[p, j, e], ftz=self.fast_math
                        )
            if cutlass.const_expr(self.pq):  # opt "pq": one f32x2 accumulator chain (KL class)
                dots[p] = self.row_dot_chain(history, qr, p)
            else:
                dots[p] = self.row_dot(history, qr, p)
        if cutlass.const_expr(not self.dq):  # "dq": the caller reduces the q-dot partials after the token loop
            for shift in reversed(range(self.lanes_per_row.bit_length() - 1)):
                for p in cutlass.range_constexpr(start, stop):
                    dots[p] += cute.arch.shuffle_sync_bfly(dots[p], 1 << shift)

    def update_lookahead(self, history, dots, operands, following, lanes, look):
        # Plain trace-time helper (no dynamic control flow): carries tuples/None.
        """Opt-in "la" (NOT a reduction-order change, needs an owner decision
        under the accuracy rule): token t's k-dot as A_t + vn_{t-1} c_{t-1}, with
        A_t = sum_k (d_t P_{t-1})[k] k_t[k] reduced during token t-1 and
        c_{t-1} = sum_k d_t[k] k_{t-1}[k] k_t[k] state-free, so the serial
        token-to-token path loses the k-dot butterfly. State update, q-dot and
        every stored value are computed as in ``update``. Returns the look-ahead
        carry for the next token. Requires one pass window (not staged)."""
        kr, qr, decays, shared_v, beta = operands
        group, lane = lanes
        levels: cutlass.Constexpr = self.lanes_per_row.bit_length() - 1
        kd = cute.make_rmem_tensor(self.passes, Float32)
        for p in range(self.passes):
            for j in range(self.chunks):
                for e in range(4):
                    history[p, j, e] = history[p, j, e] * decays[j][e]
            if cutlass.const_expr(look is None):
                kd[p] = self.row_dot(history, kr, p)
        if cutlass.const_expr(look is None):
            for shift in reversed(range(levels)):
                for p in range(self.passes):
                    kd[p] += cute.arch.shuffle_sync_bfly(kd[p], 1 << shift)
        else:
            a_now, vn_prev, c_prev = look
            for p in range(self.passes):
                kd[p] = fma(vn_prev[p], c_prev, a_now[p], ftz=self.fast_math)
        carry = None
        if cutlass.const_expr(following is not None):
            k_next, _, d_next, _, _ = following
            a_next = cute.make_rmem_tensor(self.passes, Float32)
            for p in range(self.passes):
                partial = cute.make_rmem_tensor(self.chunks, Float32)
                for j in range(self.chunks):
                    scaled = [history[p, j, e] * d_next[j][e] for e in range(4)]
                    partial[j] = dot(
                        scaled, (k_next[j, e] for e in range(4)), ftz=self.fast_math
                    )
                for shift in range(self.chunks.bit_length() - 2, -1, -1):
                    for j in range(1 << shift):
                        partial[j] += partial[j + (1 << shift)]
                a_next[p] = partial[0]
            cpart = cute.make_rmem_tensor(self.chunks, Float32)
            for j in range(self.chunks):
                pair = [kr[j, e] * d_next[j][e] for e in range(4)]
                cpart[j] = dot(pair, (k_next[j, e] for e in range(4)), ftz=self.fast_math)
            for shift in range(self.chunks.bit_length() - 2, -1, -1):
                for j in range(1 << shift):
                    cpart[j] += cpart[j + (1 << shift)]
            c_now = cpart[0]
            for shift in reversed(range(levels)):
                for p in range(self.passes):
                    a_next[p] += cute.arch.shuffle_sync_bfly(a_next[p], 1 << shift)
                c_now += cute.arch.shuffle_sync_bfly(c_now, 1 << shift)
        vn = cute.make_rmem_tensor(self.passes, Float32)
        for p in range(self.passes):
            safe_row = (p * self.groups + group) % self.rows
            vn[p] = (shared_v[safe_row] - kd[p]) * beta
            for j in range(self.chunks):
                for e in range(4):
                    history[p, j, e] = fma(
                        kr[j, e], vn[p], history[p, j, e], ftz=self.fast_math
                    )
            dots[p] = self.row_dot(history, qr, p)
        for shift in reversed(range(levels)):
            for p in range(self.passes):
                dots[p] += cute.arch.shuffle_sync_bfly(dots[p], 1 << shift)
        if cutlass.const_expr(following is not None):
            carry = (a_next, vn, c_now)
        return carry

    def act_load(self, f_a, row, col, elements):
        """F_A activation words [col, col + elements) of ``row``: load_activation_bf16x4 / x8."""
        load = load_activation_bf16x4 if elements == 4 else load_activation_bf16x8
        return load(at(f_a, row, col))

    @cute.jit
    def gate_warp(self, args, shared_decay, barrier, coords, staged_rows, stage=None):
        """Project one value partition and multicast its decays to every CTA. Modes fix
        the F_B reduction order: 0 = one chain of eight MMAs, 1 = four chains of two,
        2 = sixteen 8-FMA partials summed in a butterfly. Projection math stays IEEE.
        ``staged_rows``: shared [copy, lane, 4] words; each lane cp.asyncs its own
        16-byte F_B operands there before the PDL wait and reads them back after it."""
        f_a, a_log, dt_bias, lower_bound, weight = (args[i] for i in (4, 6, 7, 13, 16))
        head, token, value_base, lane = coords
        # Opt "g4": this warp's rows (16 of the CTA's 64; ``value_base`` already points at them).
        rows, mode = self.rows // self.gate_warps, self.gate_mode
        per: cutlass.Constexpr = max(1, rows // 32)  # decays per lane
        a = a_log[head]
        if cutlass.const_expr(mode == 2):
            feature = lane % rows
            row = head * 128 + value_base + feature
            # uint4 weights before the wait, two uint2 activation loads per 8-FMA
            # partial after it; keep that granularity and the low-then-high FMA order.
            wr = [load_weight_bf16x8(at(weight, row, 8 * i)) for i in range(16)]
            dt = dt_bias[row]
            cute.arch.griddepcontrol_wait()
            partial = cute.make_rmem_tensor(16, Float32)
            for i in cutlass.range_constexpr(16):
                a0 = self.act_load(f_a, token, 8 * i, 4)
                a1 = self.act_load(f_a, token, 8 * i + 4, 4)
                av = (*a0, *a1)
                acc = Float32(0)
                for e in cutlass.range_constexpr(4):
                    x0, x1 = unpack_bf16(av[e])
                    w0, w1 = unpack_bf16(wr[i][e])
                    acc = fma(x0, w0, acc)
                    acc = fma(x1, w1, acc)
                partial[i] = acc
            for shift in cutlass.range_constexpr(3, -1, -1):
                for i in cutlass.range_constexpr(1 << shift):
                    partial[i] += partial[i | (1 << shift)]
            raw_g = [Float32(BFloat16(partial[0]))]
            dt = [dt]
        else:
            tiles: cutlass.Constexpr = rows // 16
            chains: cutlass.Constexpr = 4 if mode == 1 else 1
            group, quarter = lane >> 2, lane & 3
            feature = 16 * (quarter >> 1) + group + 8 * (quarter & 1)
            dt = [Float32(0) for _ in range(per)]
            if cutlass.const_expr(self.tokens > 1):  # token column, its features
                token += group % self.tokens
                features = [
                    16 * (i // 2) + group + 8 * (i % 2) for i in range(rows // 8)
                ]
                dt = [dt_bias[head * 128 + value_base + f] for f in features]
            for r in cutlass.range_constexpr(per if self.tokens == 1 else 0):
                if feature + 32 * r < rows:
                    dt[r] = dt_bias[head * 128 + value_base + 32 * r + feature]
            # Two decays per lane load 16 bytes per row: chunk c then covers columns
            # 32 (c // 2) + 8 quarter + 4 (c % 2) + [0, 4) instead of 16 c + 4 quarter.
            # Opt "g4": the K layout follows the CTA's rows (64: wide), not the warp's 16, so every MMA tile sums the same
            # K columns in the same order as the one-warp path (bitwise).
            wide: cutlass.Constexpr = max(1, self.rows // 32) >= 2
            wr = cute.make_rmem_tensor((tiles, 8, 4), Uint32)
            if cutlass.const_expr(staged_rows is not None):
                for tile in cutlass.range_constexpr(tiles):
                    row = head * 128 + value_base + 16 * tile + group
                    for chunk in cutlass.range_constexpr(4):
                        col = 32 * chunk + 8 * quarter
                        for half in cutlass.range_constexpr(2):
                            slot = staged_rows[
                                (tile * 4 + chunk) * 2 + half, lane, None
                            ]
                            source = at(weight, row + 8 * half, col)
                            copy_g2s(slot.iterator, source)
                cute.arch.cp_async_commit_group()
            else:
                for tile in cutlass.range_constexpr(tiles):
                    row = head * 128 + value_base + 16 * tile + group
                    for chunk in cutlass.range_constexpr(4 if wide else 8):
                        if cutlass.const_expr(wide):
                            col = 32 * chunk + 8 * quarter
                            low = load_weight_bf16x8(at(weight, row, col))
                            high = load_weight_bf16x8(at(weight, row + 8, col))
                            self.split_pairs(wr, tile, chunk, low, high)
                        else:
                            col = 16 * chunk + 4 * quarter
                            low = load_bf16x4(at(weight, row, col))
                            high = load_bf16x4(at(weight, row + 8, col))
                            wr[tile, chunk, 0], wr[tile, chunk, 2] = low
                            wr[tile, chunk, 1], wr[tile, chunk, 3] = high
            cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(wide):
                pairs = [
                    self.act_load(f_a, token, 32 * c + 8 * quarter, 8)
                    for c in range(4)
                ]
                ar = [pairs[c // 2][2 * (c % 2) : 2 * (c % 2) + 2] for c in range(8)]
            else:
                ar = [
                    self.act_load(f_a, token, 16 * c + 4 * quarter, 4)
                    for c in range(8)
                ]
            if cutlass.const_expr(staged_rows is not None):
                cute.arch.cp_async_wait_group(0)
                for tile in cutlass.range_constexpr(tiles):
                    for chunk in cutlass.range_constexpr(4):
                        copy = (tile * 4 + chunk) * 2
                        low = smem_words(staged_rows[copy, lane, None].iterator)
                        high = smem_words(staged_rows[copy + 1, lane, None].iterator)
                        self.split_pairs(wr, tile, chunk, low, high)
            acc = zeros((chains, tiles, 4), Float32)
            for chunk in cutlass.range_constexpr(8):
                chain: cutlass.Constexpr = chunk // 2 if mode == 1 else 0
                for tile in cutlass.range_constexpr(tiles):
                    wa = tuple(wr[tile, chunk, e] for e in range(4))
                    mma_into(acc[chain, tile, None], wa, ar[chunk], bf16=True)
            if cutlass.const_expr(self.tokens > 1):
                gates = (a, lower_bound, dt, features)
                targets = (shared_decay, barrier, value_base)
                request = token // self.tokens
                if cutlass.const_expr(self.early):
                    cute.arch.cluster_wait()
                if cutlass.const_expr(self.dmc):
                    self.multicast_packets(
                        acc, gates, targets, lane, args, head, request, stage
                    )
                else:
                    self.multicast(acc, gates, targets, lane, args, head, request, 0)
                if cutlass.const_expr(self.tokens > 8):
                    # Reuse F_B fragments. Only activation columns and the D
                    # accumulators change; the cluster rendezvous runs once.
                    second = request * self.tokens + 8 + group % (self.tokens - 8)
                    if cutlass.const_expr(wide):
                        pairs = [
                            self.act_load(f_a, second, 32 * c + 8 * quarter, 8)
                            for c in range(4)
                        ]
                        ar = [
                            pairs[c // 2][2 * (c % 2) : 2 * (c % 2) + 2]
                            for c in range(8)
                        ]
                    else:
                        ar = [
                            self.act_load(f_a, second, 16 * c + 4 * quarter, 4)
                            for c in range(8)
                        ]
                    tail = zeros((chains, tiles, 4), Float32)
                    for chunk in cutlass.range_constexpr(8):
                        chain: cutlass.Constexpr = chunk // 2 if mode == 1 else 0
                        for tile in cutlass.range_constexpr(tiles):
                            wa = tuple(wr[tile, chunk, e] for e in range(4))
                            mma_into(tail[chain, tile, None], wa, ar[chunk], bf16=True)
                    self.multicast(tail, gates, targets, lane, args, head, request, 8)
                return
            raw_g = []
            for r in cutlass.range_constexpr(per):
                partial = cute.make_rmem_tensor(chains, Float32)
                for chain in cutlass.range_constexpr(chains):
                    selected = acc[chain, 2 * r, 0]
                    if quarter & 1:
                        selected = acc[chain, 2 * r, 2]
                    if cutlass.const_expr(tiles > 2 * r + 1):  # noqa: SIM102
                        if quarter >> 1:
                            selected = acc[chain, 2 * r + 1, 0]
                            if quarter & 1:
                                selected = acc[chain, 2 * r + 1, 2]
                    partial[chain] = selected
                value = partial[0]
                if cutlass.const_expr(mode == 1):
                    value = Float32(0) + (
                        ((value + partial[1]) + partial[2]) + partial[3]
                    )
                raw_g.append(Float32(BFloat16(value)))

        decay = []
        for r in cutlass.range_constexpr(per):
            gate = cute.math.exp(a, fastmath=True) * (raw_g[r] + dt[r])
            gate = self.activation(gate)
            decay.append(cute.math.exp(lower_bound * gate, fastmath=True))
        packets: cutlass.Constexpr = rows // 4
        packet, destination = lane % packets, lane // packets
        values = cute.make_rmem_tensor(4, Float32)
        for e in cutlass.range_constexpr(4):
            source = 4 * packet + e
            if cutlass.const_expr(per == 2):
                feature = source & 31
                source = (feature & 7) << 2 | (feature >> 4) << 1 | feature >> 3 & 1
                low = cute.arch.shuffle_sync(decay[0], source)
                high = cute.arch.shuffle_sync(decay[1], source)
                values[e] = low
                if packet >= 8:
                    values[e] = high
            elif cutlass.const_expr(per > 2):  # one CTA per head: 128 features
                feature = source & 31
                source = (feature & 7) << 2 | (feature >> 4) << 1 | feature >> 3 & 1
                shuffled = [cute.arch.shuffle_sync(d, source) for d in decay]
                values[e] = shuffled[0]
                for r in cutlass.range_constexpr(1, per):
                    if packet >> 3 == r:
                        values[e] = shuffled[r]
            else:
                if cutlass.const_expr(mode != 2):
                    source = (source & 7) << 2 | (source >> 4) << 1 | source >> 3 & 1
                values[e] = cute.arch.shuffle_sync(decay[0], source)
        offset = shared_decay.iterator + value_base + 4 * packet
        pointer = cute.make_ptr(
            Float32, offset.llvm_ptr, cute.AddressSpace.smem, assumed_align=16
        )
        words = tuple(float_bits(values[e]) for e in range(4))
        if cutlass.const_expr(self.early):
            cute.arch.cluster_wait()
        store_async_dsmem(pointer, words, barrier, destination)

    @cute.jit
    def multicast(self, acc, gates, targets, lane, args, head, request, token_base):
        """Decays (q > 1) of the lane's D fragments (feature rows, token columns 2
        quarter + {0, 1}) to every CTA's shared [token, 128] decays."""
        (a, lower_bound, dt, features), (shared_decay, barrier, value_base) = (
            gates,
            targets,
        )
        quarter = lane & 3
        for i in cutlass.range_constexpr(len(features)):
            tile, e = i // 2, (i % 2) * 2
            for column in cutlass.range_constexpr(2):
                value = acc[0, tile, e + column]
                if cutlass.const_expr(self.gate_mode == 1):
                    parts = [acc[c, tile, e + column] for c in range(4)]
                    value = Float32(0) + (((value + parts[1]) + parts[2]) + parts[3])
                raw_g = Float32(BFloat16(value))
                gate = cute.math.exp(a, fastmath=True) * (raw_g + dt[i])
                gate = self.activation(gate)
                log_decay = lower_bound * gate
                decay = float_bits(cute.math.exp(log_decay, fastmath=True))
                token = token_base + 2 * quarter + column
                offset = token * 128 + value_base + features[i]
                pointer = shared_decay.iterator + offset
                if token < self.tokens:
                    if cutlass.const_expr(self.replayssm):
                        slot = args[8][request]
                        args[21][Int64(slot), head, token, value_base + features[i]] = (
                            log_decay
                        )
                    if cutlass.const_expr(self.local):
                        put_float(pointer, bits_float(decay))
                    else:
                        for rank in cutlass.range_constexpr(self.cluster):
                            store_async_dsmem(pointer, decay, barrier, rank)
        if cutlass.const_expr(self.local):  # one arrival per gate lane: setup() counts 32
            arrive_release_cta(barrier)

    @cute.jit
    def multicast_packets(self, acc, gates, targets, lane, args, head, request, stage):
        """Opt-in "dmc" (q <= 8): ``multicast``'s per-value math, staged in
        shared memory, then sent as 16-byte packets (one token, four rows) so each
        peer gets T * rows / 4 vector ``st.async`` per lane-set instead of T * rows
        scalar ones, and the log-decay ring gets 16-byte stores."""
        (a, lower_bound, dt, features), (shared_decay, barrier, value_base) = (
            gates,
            targets,
        )
        quarter = lane & 3
        for i in cutlass.range_constexpr(len(features)):
            tile, e = i // 2, (i % 2) * 2
            for column in cutlass.range_constexpr(2):
                value = acc[0, tile, e + column]
                if cutlass.const_expr(self.gate_mode == 1):
                    parts = [acc[c, tile, e + column] for c in range(4)]
                    value = Float32(0) + (((value + parts[1]) + parts[2]) + parts[3])
                raw_g = Float32(BFloat16(value))
                gate = cute.math.exp(a, fastmath=True) * (raw_g + dt[i])
                gate = self.activation(gate)
                log_decay = lower_bound * gate
                decay = cute.math.exp(log_decay, fastmath=True)
                token = 2 * quarter + column
                if token < self.tokens:
                    stage[0, token, features[i]] = decay
                    stage[1, token, features[i]] = log_decay
        cute.arch.sync_warp()
        per_token: cutlass.Constexpr = self.rows // 4
        packets: cutlass.Constexpr = self.tokens * per_token
        slot = args[8][request]
        for k in cutlass.range_constexpr(-(-packets // 32)):
            packet = lane + 32 * k
            if packet < packets:
                token, quad = packet // per_token, packet % per_token
                row = token * self.rows + 4 * quad
                decays = load_f32x4(stage.iterator + row, shared=True)
                logs = load_f32x4(stage.iterator + self.tokens * self.rows + row, shared=True)
                feature = value_base + 4 * quad
                ring_page = args[21][Int64(slot), None, None, None]
                store_f32x4(at(ring_page, head, token, feature), logs)
                offset = shared_decay.iterator + (token * 128 + feature)
                pointer = cute.make_ptr(
                    Float32, offset.llvm_ptr, cute.AddressSpace.smem, assumed_align=16
                )
                words = tuple(float_bits(decays[e]) for e in range(4))
                for rank in cutlass.range_constexpr(self.cluster):
                    store_async_dsmem(pointer, words, barrier, rank)

    @staticmethod
    def split_pairs(wr, tile, chunk, low, high):
        """16 bytes of rows r and r + 8 hold the A fragments of chunks 2c and 2c + 1."""
        for h in range(2):
            c = 2 * chunk + h
            wr[tile, c, 0], wr[tile, c, 2] = low[2 * h : 2 * h + 2]
            wr[tile, c, 1], wr[tile, c, 3] = high[2 * h : 2 * h + 2]

    @cute.jit
    def setup(self, barrier, gate_barrier, partition, thread, armed=()):
        """Thread 0 arms the output (leader) and decay barriers (and ``armed``'s (barrier,
        CTA, bytes) rows); the CTA arrives on the cluster barrier peers wait on before their
        sends."""
        tokens, cluster = self.tokens, self.cluster
        if cutlass.const_expr(self.local):
            # One CTA: token t's barrier completes after its min(groups, rows) output senders (the group_lane-0
            # threads) arrive, the decay barrier after the 32 gate lanes; nothing is armed with bytes.
            senders = self.rows if self.wy else min(self.groups, self.rows)  # "wy": one solver thread per row
            if thread == 0:
                for k in cutlass.range_constexpr(tokens):
                    cute.arch.mbarrier_init(barrier + k, senders)
                cute.arch.mbarrier_init(gate_barrier, 32)
                cute.arch.mbarrier_init_fence()
        else:
            owned = [(barrier, partition == 0, 512)]  # token t's outputs: CTA t % cluster
            if cutlass.const_expr(tokens > 1):
                owned = [
                    (barrier + k, partition + k * cluster < tokens, 512)
                    for k in range(-(-tokens // cluster))
                ]
            if cutlass.const_expr(self.vp):  # opt "vp": one barrier for all four owned tokens (2 CTAs x 64 rows x 16 B)
                owned = [(barrier, True, 2048)]
            for bar, cta, size in (*owned, (gate_barrier, True, 512 * tokens), *armed):
                if cta & (thread == 0):
                    cute.arch.mbarrier_init(bar, 1)
                    if cutlass.const_expr(self.k23_arm):  # K-23: peers st.async into every barrier armed here
                        arm_cluster(bar, size)
                    else:
                        cute.arch.mbarrier_arrive_and_expect_tx(bar, size)
                    cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()

    @cute.jit
    def row_dot_chain(self, history, vector, p):
        """Opt "pq": sum_c history[p, c] vector[c] as one f32x2 accumulator chain over all chunks, then lo + hi."""
        ftz = self.fast_math
        a0, a1 = mul2_ftz((history[p, 0, 0], history[p, 0, 1]), (vector[0, 0], vector[0, 1]), ftz=ftz)
        a0, a1 = fma2_ftz((history[p, 0, 2], history[p, 0, 3]), (vector[0, 2], vector[0, 3]), (a0, a1), ftz=ftz)
        for j in cutlass.range_constexpr(1, self.chunks):
            for e in cutlass.range_constexpr(0, 4, 2):
                a0, a1 = fma2_ftz((history[p, j, e], history[p, j, e + 1]), (vector[j, e], vector[j, e + 1]), (a0, a1), ftz=ftz)
        return a0 + a1

    @cute.jit
    def row_dot(self, history, vector, p):
        partial = cute.make_rmem_tensor(self.chunks, Float32)
        for j in cutlass.range_constexpr(self.chunks):
            if cutlass.const_expr(self.packed):
                # Two interleaved lanes per chunk (f32x2), then their sum: a
                # different FP32 summation order of the same products.
                h = [history[p, j, e] for e in range(4)]
                x = [vector[j, e] for e in range(4)]
                ftz = self.fast_math
                a0, a1 = mul2_ftz((h[0], h[1]), (x[0], x[1]), ftz=ftz)
                a0, a1 = fma2_ftz((h[2], h[3]), (x[2], x[3]), (a0, a1), ftz=ftz)
                partial[j] = a0 + a1
            else:
                row = (history[p, j, e] for e in range(4))
                partial[j] = dot(
                    row, (vector[j, e] for e in range(4)), ftz=self.fast_math
                )
        for shift in cutlass.range_constexpr(self.chunks.bit_length() - 2, -1, -1):
            for j in cutlass.range_constexpr(1 << shift):
                partial[j] += partial[j + (1 << shift)]
        return partial[0]

    @cute.jit
    def output_position(self, row: Int32):
        return ((row & 7) << 4) | ((row >> 5) << 2) | ((row & 31) >> 3)
