# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Dense, linear-chain KDA TARGET_VERIFY with host-owned ReplaySSM commit.

Unlike PreparedKda's snapshot ABI, state_slots contains one committed checkpoint
per request. Verify never modifies that checkpoint or persistent convolution.
The host must complete commit/restore before launching the next verify.
"""

import logging
import os
from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from ..common.host import fake, layout, stream, to_cute
from .kernel import KdaSplit
from .prepared import decode_conv_weight

logger = logging.getLogger(__name__)


# VR200 (SM107) verify schedules (early, lanes per state row, minimum CTAs per SM, CTAs
# per head) by (requests, queries), from an SM107 sweep (cold L2, same gates): Breq4 Q8
# on 4-CTA clusters (one wave) measured faster than GB300's schedule;
# GB300's 8-CTA clusters stay best at Breq 1 and 2.
# Breq 8 / 16 Q8 (TP16 heads 6, same-GPU ABBA x 2 on 4 GPUs, same gates): 2-CTA clusters at
# 4 CTAs per SM measured faster at Breq 8 and Breq 16; with "wide" the lanes per row follow
# the cluster, so 8.4.2 and 16.4.2 build the same kernel.
# Breq 32 Q8 (1 GPU, graph of 48 PDL-chained calls, forward + reverse passes, FP32-reference gates): the same
# 8.4.2 kernel, measured faster than GB300's 8-CTA fallback.
SM107_VERIFY: dict = {(4, 8): (0, 8, 2, 4), (8, 8): (0, 8, 4, 2), (16, 8): (0, 8, 4, 2), (32, 8): (0, 8, 4, 2)}
# With opt "wlf": where the 2-CTA schedule (0, 8, 4, 2) would need more than one wave (2 x heads x Breq CTAs > SMs; 152-168
# registers x 288 threads = 1 CTA per SM), run single-CTA clusters of 128 rows instead (8 lanes per row = 4 rows per
# thread, packed): half the CTAs. Cold graph, 1 GPU, us per call, same-GPU ABBA x 2, both "wide" + 8 lanes + packed,
# FP32-reference gates, 2-CTA -> 1-CTA: measured faster at each measured TP16 / TP8 shape where the 2-CTA grid needs more
# than one wave. Where it is one wave already, the 128-row CTA loses. Without "wlf" the 128-row CTA would get "wide"'s 2 lanes
# per row, a form never measured, so the rule needs the opt.
WLF_ONE_WAVE = (0, 8, 1, 1)
# With "wy" + "wlf" : the 2-CTA 64-row CTAs run 16 lanes per state row (4 rows x 8 columns per thread: half
# the basis loads of 8 lanes in the dot sweep). Cold graph, 1 GPU, TP16: 16 lanes measured faster than 8 at Breq 8
# and Breq 16. 128-row CTAs keep 8 lanes
# (16 lanes spill there and measured slower at Breq 32).
WY_TWO_CTA = (0, 16, 4, 2)
# Short-Q verify schedules (default off; K31MK_KDA_VERIFY_SHORTQ=1): at Q < 8 SM107_VERIFY has no entry, so every
# short-Q verify runs GB300's (0, 8, 2, 8). Keyed (heads per rank, Breq, Q) -> (early, lanes, CTAs/SM, CTAs per head); a key that is
# missing keeps GB300's form. Entries = the measured winners (1 GPU, TP16 6 heads, served wide-tier opts, every test_kda_verify gate
# passed, cold graph, same-process ABBA; a form enters only if both reads beat GB300's form by >= 3 %).
# Forms (as keyed in the table below): Q5 Breq 8-13 8.2.4, Breq 16 8.3.4; Q4 / Q6 Breq 8-13 16.2.4, Breq 16 8.4.2;
# Q7 Breq 4-13 8.2.4, Breq 16 8.3.4.
# Breq 1 / 2 (and 4 / 6 except Q6 / Q7 Breq 4) keep GB300's form: every other form was slower there. Breq 10 / 11 / 14 / 15 (unswept)
# take Breq 13's form; Breq 3 / 5 / 7 keep GB300's form until measured
SM107_SHORTQ: dict = {
    (6, 8, 4): (0, 16, 2, 4),
    (6, 9, 4): (0, 16, 2, 4),
    (6, 12, 4): (0, 16, 2, 4),
    (6, 13, 4): (0, 16, 2, 4),
    (6, 10, 4): (0, 16, 2, 4),  # Breq 13's form (unswept)
    (6, 11, 4): (0, 16, 2, 4),  # Breq 13's form (unswept)
    (6, 14, 4): (0, 16, 2, 4),  # Breq 13's form (unswept)
    (6, 15, 4): (0, 16, 2, 4),  # Breq 13's form (unswept)
    (6, 16, 4): (0, 8, 4, 2),
    (6, 8, 5): (0, 8, 2, 4),
    (6, 9, 5): (0, 8, 2, 4),
    (6, 12, 5): (0, 8, 2, 4),
    (6, 13, 5): (0, 8, 2, 4),
    (6, 10, 5): (0, 8, 2, 4),  # Breq 13's form (unswept)
    (6, 11, 5): (0, 8, 2, 4),  # Breq 13's form (unswept)
    (6, 14, 5): (0, 8, 2, 4),  # Breq 13's form (unswept)
    (6, 15, 5): (0, 8, 2, 4),  # Breq 13's form (unswept)
    (6, 16, 5): (0, 8, 3, 4),
    (6, 4, 6): (0, 16, 2, 4),
    (6, 6, 6): (0, 16, 2, 4),
    (6, 8, 6): (0, 16, 2, 4),
    (6, 9, 6): (0, 16, 2, 4),
    (6, 12, 6): (0, 16, 2, 4),
    (6, 13, 6): (0, 16, 2, 4),
    (6, 10, 6): (0, 16, 2, 4),  # Breq 13's form (unswept)
    (6, 11, 6): (0, 16, 2, 4),  # Breq 13's form (unswept)
    (6, 14, 6): (0, 16, 2, 4),  # Breq 13's form (unswept)
    (6, 15, 6): (0, 16, 2, 4),  # Breq 13's form (unswept)
    (6, 16, 6): (0, 8, 4, 2),
    (6, 4, 7): (0, 8, 2, 4),
    (6, 8, 7): (0, 8, 2, 4),
    (6, 9, 7): (0, 8, 2, 4),
    (6, 12, 7): (0, 8, 2, 4),
    (6, 13, 7): (0, 8, 2, 4),
    (6, 10, 7): (0, 8, 2, 4),  # Breq 13's form (unswept)
    (6, 11, 7): (0, 8, 2, 4),  # Breq 13's form (unswept)
    (6, 14, 7): (0, 8, 2, 4),  # Breq 13's form (unswept)
    (6, 15, 7): (0, 8, 2, 4),  # Breq 13's form (unswept)
    (6, 16, 7): (0, 8, 3, 4),
}


def shortq_enabled():
    return os.environ.get("K31MK_KDA_VERIFY_SHORTQ", "0") == "1"


# Breq 5 / 6 Q8 verify schedules (default off; K31MK_KDA_VERIFY_B57=1). Breq 5-7 Q8 otherwise take the
# Breq 8 graph's 2-CTA form (0, 8, 4, 2). Keyed (heads per rank, Breq) -> (early, lanes, CTAs/SM, CTAs per head); a missing key
# keeps the served form. 1 GPU, TP16 6 heads, the served wide-tier opt string (early+wide+rs+lt+tsp+dmc+chunk+cg+cs+wlf+b3c+dq),
# every test_kda_verify FP32-reference gate passed, cold graph, mirrored passes: 4-CTA clusters at 2 CTAs
# per SM measured faster than served at Breq 5 and 6 (both passes); Breq 4 already runs it, Breq 7 unmeasured.
# Another cluster split of the same FP32 products (not bitwise to the served form): a KL arm before serving.
SM107_B57_Q8: dict = {
    (6, 5): (0, 8, 2, 4),
    (6, 6): (0, 8, 2, 4),
}


def b57_enabled():
    return os.environ.get("K31MK_KDA_VERIFY_B57", "0") == "1"


def verify_schedule(requests, queries, device, opt=frozenset(), heads=None):
    """GB300: (0, 8, 1 if Q > 8 else 2, 8). ``K31MK_KDA_VERIFY_SCHEDULE="l.o.c"`` forces
    lane count, occupancy and cluster (tuning sweeps); ``K31MK_KDA_SM107_RETUNE=0`` keeps GB300's.
    ``opt``: the verify opts (``verify_opt()``); with "wlf" and ``heads`` (KDA heads per rank), WLF_ONE_WAVE replaces
    the 2-CTA schedule when that grid would exceed one wave; with "wy" too, WY_TWO_CTA replaces the remaining 2-CTA one."""
    forced = os.environ.get("K31MK_KDA_VERIFY_SCHEDULE")
    if forced:
        return (0, *map(int, forced.split(".")))
    retune = os.environ.get("K31MK_KDA_SM107_RETUNE", "1") != "0"
    if retune and torch.cuda.get_device_capability(device) == (10, 7):
        # Breq 3 / 5-7 Q8 (own graphs): the schedule of the Breq 4 / 8 graph they were padded to. The
        # one-wave rule below counts the CTAs of the request count actually launched.
        padded = {3: 4, 5: 8, 6: 8, 7: 8}.get(requests, requests) if queries == 8 else requests
        # Wide tier (Breq 9-15 Q8, own graphs, : the schedule of the Breq 16 graph they would pad to.
        padded = 16 if queries == 8 and 8 < requests < 16 else padded
        if requests == 3 and queries == 8 and ("b3w" in opt or "b3c" in opt):
            return B3_8CTA
        if queries == 8 and b57_enabled():
            # Measured as is (each form forced), so the wlf / wy rewrites below do not apply to it.
            b57 = SM107_B57_Q8.get((heads, requests))
            if b57 is not None:
                return b57
        schedule = SM107_VERIFY.get((padded, queries))
        if schedule is None and queries < 8 and shortq_enabled():
            # Measured as is (each form forced), so the wlf / wy rewrites below do not apply to it.
            short = SM107_SHORTQ.get((heads, requests, queries))
            if short is not None:
                return short
        if "wlf" in opt and heads is not None and schedule == (0, 8, 4, 2):
            if 2 * heads * requests > torch.cuda.get_device_properties(device).multi_processor_count:
                return WLF_ONE_WAVE
            if "wy" in opt:
                return WY_TWO_CTA
        if schedule is not None:
            return schedule
    return (0, 8, 1 if queries > 8 else 2, 8)


# Opt-in ReplaySSM verify variants (default off): '+'-joined names in
# K31MK_KDA_VERIFY_OPT, read at prepare. See KdaSplit's docstring for each; "gm1"
# forces gate MMA mode 1 (four chains of two MMAs, then their sum: the FP32 order of
# the F_B reduction, as GB300 decode uses at B1) instead of the policy's mode.
VERIFY_OPTS = ("early", "pre", "pw", "wide", "quad", "pk", "wl8", "dmc", "rs", "la", "lt", "gm1")
# Default off: "chunk" = the chunked (WY) verify step (KdaSplit.chunk_outputs;
# Q8 with "wide"; not bitwise, a reordering of the same FP32 products).
VERIFY_OPTS += ("chunk",)
# "lean" / "bal" (bitwise): owned-gate-only activation + betas by their stagers; the
# ReplaySSM q / k side writes of token t by CTA t % cluster instead of all by partition 0.
VERIFY_OPTS += ("lean", "bal")
# "tsp" (bitwise; with early+wide, engages at Q8 on 8-CTA clusters): token-split phase 1,
# CTA p forms token p's q / k once and stores them into every CTA of the cluster (KdaSplit.prologue_tsp).
VERIFY_OPTS += ("tsp",)
# "cg" / "cs" (with "chunk"): the Gram once per head on the gate
# warps (cluster-distributed rows, DSMEM) / P-R in a column-split layout with one solver warp.
VERIFY_OPTS += ("cg", "cs")
# "csp" (with "cs"): its token loop as packed f32x2 multiplies / FMAs (column pairs; a reordering).
VERIFY_OPTS += ("csp",)
# "vp" / "ds" (default off, bitwise; MID 2-CTA clusters with "dq", Q8): "vp" = one 16-byte output push per (row, owner CTA)
# into a padded owner-major layout and one owned-output barrier; "ds" = phase 1's ring / window side stores issued after the output pushes.
VERIFY_OPTS += ("vp", "ds")
# "dl" (default off, bitwise; exclusive with "ds"): the same side stores inside the token loop as predicated stores.
VERIFY_OPTS += ("dl",)
# "pq" (default off, KL class): each row's deferred q-dot as one f32x2 accumulator chain (KdaSplit.row_dot_chain).
VERIFY_OPTS += ("pq",)
# "gb" (with "cs", not "cg"): the Gram rows by the main warps with balanced work (KdaSplit.gram_balanced).
VERIFY_OPTS += ("gb",)
# "wlf" (with "wide"; default off): the schedule's lanes per state row are a floor under "wide", so the
# 64-row CTAs of Breq 8 / 16 / 32 on SM107 run 2 rows per thread, and those multi-row threads use the packed f32x2 decay /
# update / row dots ("pk"); Breq 1 / 4 build the same kernel as without it. Cold graph, 1 GPU, same-GPU ABBA:
# 2 rows per thread, then packed, each measured faster at Breq 8 / 16 / 32;
# 16 or 32 lanes per row are slower. Not bitwise: the row dots
# reduce over 8 lanes in f32x2 pairs instead of 4 (a reordering of the same FP32 products). With "wlf", a 2-CTA grid that
# would exceed one wave also switches to WLF_ONE_WAVE's 128-row CTAs (4 rows per thread; verify_schedule).
VERIFY_OPTS += ("wlf",)
# "wy" (default off): the chunked (WY) verify step in the dot-basis form for the 32 / 64 / 128-row
# CTAs where "chunk" does not engage (KdaSplit.wy_outputs; not bitwise, a reordering of the same FP32 products).
VERIFY_OPTS += ("wy",)
# "tsp2" / "g4" (default off, bitwise): tsp's token split at 2-CTA clusters (64-row CTAs, Breq 8 / 16
# on SM107; KdaSplit.prologue_tsp2) / four gate warps of 16 rows instead of one of 64.
VERIFY_OPTS += ("tsp2", "g4")
# "b3w" / "b3c" (default off): Breq 3 Q8 on SM107 runs B3_8CTA (8-CTA clusters, one wave) with tsp + dmc
# ("b3w", bitwise) or also the chunk family ("b3c"); see kernel.B3_MAX_REQUESTS.
VERIFY_OPTS += ("b3w", "b3c")
# "dq" (default off, bitwise by construction): q-dot butterflies and output pushes deferred
# after the serial token loop (KdaSplit.dq).
VERIFY_OPTS += ("dq",)
# "tsp4" (default off, bitwise by construction): tsp's token split on the 4-CTA clusters of 32 rows that SM107
# runs at Breq 4 Q8 (and Breq 5 / 6 under K31MK_KDA_VERIFY_B57=1), up to kernel.TSP4_MAX_REQUESTS; CTA p forms tokens 2p, 2p + 1
# once for the cluster (KdaSplit.prologue_tsp4). Elsewhere (other clusters, dmc / chunk / wy / lean builds) it switches itself off.
VERIFY_OPTS += ("tsp4",)
# "g2" (default off, bitwise): "g4"'s gate-warp split on the 32-row CTAs of those 4-CTA clusters (2 gate warps of 16 rows;
# KdaSplit.gate_warp). D3G_BUILT records the gate warps compiled.
VERIFY_OPTS += ("g2",)
# "pkb" (default off, bitwise): "pk"'s f32x2 decay multiply and update on those 32-row CTAs, scalar row dots kept
# (lanewise mul.rn / fma.rn round as the scalar ones; no summation order changes). Elsewhere it switches itself off.
VERIFY_OPTS += ("pkb",)
# "occ1" (default off, same code): 1 CTA / SM (launch bound) on the TP16 32-row CTAs of the 4-CTA clusters (Breq 4; 5 / 6 with B57).
VERIFY_OPTS += ("occ1",)
# "sred" (default off, bitwise): the k-dot's 8-lane butterfly on those 32-row CTAs as one shared store + warp barrier + two loads,
# summed in the butterfly's order. Elsewhere it switches itself off.
VERIFY_OPTS += ("sred",)
# "dql" (default off, bitwise; with "dq" on the 32-row CTAs): token t - 1's q-dot butterfly + output push inside the token loop.
VERIFY_OPTS += ("dql",)


B3_8CTA = (0, 8, 2, 8)


def verify_opt():
    names = frozenset(n for n in os.environ.get("K31MK_KDA_VERIFY_OPT", "").split("+") if n)
    unknown = names - set(VERIFY_OPTS)
    if unknown:
        raise ValueError(f"unknown K31MK_KDA_VERIFY_OPT names {sorted(unknown)}")
    return names


# Q-port stage 1 (kimi_k31 K31MK_VERIFY_QPORT, default off;, owner review c-kda): the verify riders below were
# built for Q8 and assert on other widths where they engage (kernel.py KdaSplit.__init__: "chunk" / "lean" / "wy" / the
# cg-cs family need tokens == 8; "wide" / "quad" need tokens % halves == 0; "rs" needs 2 * tokens / halves a power of two),
# or switch themselves off there (tsp / tsp2 / g4 / b3w / b3c). At Q 2-7 the provider builds the generic ReplaySSM
# KdaSplit without them; at Q8 (served) and Q 9-16 (legacy Q16) the opt set is returned unchanged.
Q8_ONLY_OPTS = frozenset({"chunk", "cg", "cs", "csp", "gb", "lean", "wy", "tsp", "tsp2", "g4", "b3w", "b3c"})


def verify_opt_for(queries, opt):
    """``opt`` (a ``verify_opt()`` set) as the KdaSplit verify build at Q ``queries`` uses it (identity unless 2 <= Q < 8)."""
    opt = frozenset(opt)
    if not 2 <= queries < 8:
        return opt
    opt -= Q8_ONLY_OPTS
    halves = 4 if "quad" in opt else 2 if "wide" in opt else 1
    if queries % halves:
        opt -= {"wide", "quad"}
        halves = 1
    half_tokens = queries // halves
    if "rs" in opt and (2 * half_tokens) & (2 * half_tokens - 1):
        opt -= {"rs"}
    return opt


# Fix A mode under "early" (SplitKGemm fenced_trigger; see gemm/splitk.py): 3 = the weight
# warp after its weight stages (the shipped trigger's place).
EARLY_FENCE_MODE = 3


def projection_fence():
    """The "fix A" mode for the KDA in-projection (the verify's and decode core's PDL
    predecessor): 0 = shipped trigger; 1-3 = the trigger follows a PDL wait behind a release
    fence. K31MK_KDA_PROJ_FENCE=1|2|3 forces a mode; otherwise EARLY_FENCE_MODE whenever the
    verify runs "early", whose pre-wait reads (slot table, scratch rows, checkpoint rows,
    conv taps) are ordered only through that fence (PTX 9.7.15.14)."""
    forced = os.environ.get("K31MK_KDA_PROJ_FENCE", "")
    if forced:
        if forced not in ("1", "2", "3"):
            raise ValueError("K31MK_KDA_PROJ_FENCE takes 1, 2 or 3 (see gemm/splitk.py)")
        return int(forced)
    return EARLY_FENCE_MODE if "early" in verify_opt() else 0



@cache
def executable(
    specs, heads, requests, queries, mode, has_bias, device, schedule=None, opt=frozenset()
):
    args = tuple(fake(*spec) for spec in specs)
    schedule = schedule or (0, 8, 1 if queries > 8 else 2, 8)
    if "early" in opt:
        schedule = (1, *schedule[1:])
    if "gm1" in opt:
        mode = 1
    kernel = KdaSplit(
        heads,
        requests,
        mode,
        has_bias,
        schedule,
        queries,
        replayssm=True,
        opt=opt,
        bf16_state=specs[9][2] == torch.bfloat16,  # temporal pool dtype
    )
    BUILT[requests, queries, frozenset(opt)] = kernel.chunk  # engagement record (tests, benches)
    TSP_BUILT[requests, queries, frozenset(opt)] = kernel.tsp
    DMC_BUILT[requests, queries, frozenset(opt)] = kernel.dmc
    PATHS[requests, queries, frozenset(opt)] = (kernel.chunk, kernel.cgram, kernel.csplit)
    CLUSTERS[requests, queries, frozenset(opt)] = (kernel.cluster, kernel.occupancy, kernel.lanes_per_row)
    WY_BUILT[requests, queries, frozenset(opt)] = (kernel.wy, kernel.cluster, kernel.passes, kernel.lanes_per_row)
    D3G_BUILT[requests, queries, frozenset(opt)] = (kernel.tsp2, kernel.gate_warps)
    TSP4_BUILT[requests, queries, frozenset(opt)] = kernel.tsp4
    PKB_BUILT[requests, queries, frozenset(opt)] = kernel.pkb
    DQL_BUILT[requests, queries, frozenset(opt)] = kernel.dql
    if kernel.tsp4:  # opt "tsp4" engagement line; the line below is unchanged for every other build
        logger.info("KdaSplit verify Breq %d Q %d heads %d: tsp4 ENGAGED (cluster %d)", requests, queries, heads, kernel.cluster)
    OPT_BUILT[requests, queries, frozenset(opt)] = (kernel.vp, kernel.ds, kernel.gate_warps)
    DL_BUILT[requests, queries, frozenset(opt)] = (kernel.dl, kernel.ds, kernel.gate_warps)
    PQ_BUILT[requests, queries, frozenset(opt)] = kernel.pq  # engagement record
    # Once per compiled schedule (this function is cached): what the verify runs at this Breq.
    logger.info(
        "KdaSplit verify Breq %d Q %d heads %d: cluster %d, CTAs per SM %d, lanes per row %d; tsp %s, dmc %s, chunk %s, cg %s, "
        "cs %s; opt %s", requests, queries, heads, kernel.cluster, kernel.occupancy, kernel.lanes_per_row, kernel.tsp,
        kernel.dmc, kernel.chunk, kernel.cgram, kernel.csplit, "+".join(sorted(opt)) or "-",
    )
    scalars =tuple(cutlass.Float32(v) for v in (-5, 128**-0.5, 1e-6))
    with torch.accelerator.device_index(device):
        return cute.compile(
            kernel, *args[:13], *scalars, *args[13:], make_fake_stream()
        )


# (requests, queries, opt) -> whether that KdaSplit compiled the "chunk" path.
BUILT: dict = {}
# (requests, queries, opt) -> whether that KdaSplit compiled the "tsp" phase 1.
TSP_BUILT: dict = {}
# (requests, queries, opt) -> whether that KdaSplit compiled "dmc" (FX-5: only up to kernel.DMC_MAX_REQUESTS).
DMC_BUILT: dict = {}
# (requests, queries, opt) -> (chunk, cg, cs) paths compiled.
PATHS: dict = {}
# (requests, queries, opt) -> (CTAs per cluster, min CTAs per SM, lanes per row) the KdaSplit was built with.
CLUSTERS: dict = {}
# (requests, queries, opt) -> (wy, CTAs per cluster, rows per thread, lanes per row) (engagement record).
WY_BUILT: dict = {}
# (requests, queries, opt) -> (tsp2 compiled, gate warps) (engagement record).
D3G_BUILT: dict = {}
# (requests, queries, opt) -> whether that KdaSplit compiled the "tsp4" phase 1 (engagement record).
TSP4_BUILT: dict = {}
# (requests, queries, opt) -> whether that KdaSplit compiled the "pkb" f32x2 decay / update (engagement record).
PKB_BUILT: dict = {}
# (requests, queries, opt) -> whether that KdaSplit compiled "dql" (q-dot reduce + push of token t - 1 inside the loop).
DQL_BUILT: dict = {}
# (requests, queries, opt) -> (vp compiled, ds compiled, gate warps) (engagement record).
OPT_BUILT: dict = {}
# (requests, queries, opt) -> (dl compiled, ds compiled, gate warps) (engagement record).
DL_BUILT: dict = {}
# (requests, queries, opt) -> pq compiled (engagement record).
PQ_BUILT: dict = {}


class PreparedKdaVerify:
    """Policies map (Breq, Q) to (packed projection pitch, gate MMA mode).

    Pool views, including nonstandard slot/conv pitches, are fixed at prepare.
    caches = (conv [S,3,3D] or [S,3D,3], temporal FP32[S,H,128,128]);
    conv_window = BF16[spec_rows,Qcap,3,3D]; rings = (rawv,rawk,g,beta),
    shaped [S,H,L,128] for the first three and [S,H,L] for beta.
    Raw V/K are BF16, g/beta FP32; L >= Q. Metadata is int32[Breq],
    state slot 0 is valid and a negative state/scratch row is padding.
    """

    def __init__(
        self,
        *,
        conv_weight,
        f_b_weight,
        a_log,
        dt_bias,
        norm_weight,
        lower_bound,
        eps,
        policies,
        conv_dim_first=False,
    ):
        if any(not isinstance(key, tuple) or len(key) != 2 for key in policies):
            raise ValueError("KDA verify policies require explicit (Breq, Q) keys")
        if any(mode not in (0, 1) for _, mode in policies.values()):
            raise ValueError("KDA verify requires gate MMA mode 0 or 1")
        self.heads = a_log.numel()
        self.dim, self.device = self.heads * 128, dt_bias.device
        self.has_bias = False  # K31 conv has no bias.
        self.weights = (
            decode_conv_weight(conv_weight),
            dt_bias,
            f_b_weight.detach().contiguous(),
            a_log.detach().reshape(-1),
            dt_bias.detach().reshape(-1),
            norm_weight.detach().float(),
        )
        self.scalars = tuple(map(cutlass.Float32, (lower_bound, 128**-0.5, eps)))
        self.policies, self.conv_dim_first, self.kernels = policies, conv_dim_first, {}

    def _verify_operands(
        self, caches, projected, state_slots, out, scratch_rows, conv_window, rings
    ):
        conv, state = caches
        conv = conv if self.conv_dim_first else conv.transpose(-1, -2)
        w, bias, fb, a_log, dt, norm = self.weights
        d, h = self.dim, self.heads
        # [rows, pitch] BF16: the column slices.
        p = lambda start, width: projected[..., start : start + width]
        return (
            p(0, 3 * d),
            w,
            bias,
            conv,
            p(4 * d, 128),
            p(4 * d + 128, h),
            a_log,
            dt,
            state_slots,
            state,
            out,
            p(3 * d, d),
            norm,
            fb,
            scratch_rows,
            conv_window.transpose(-1, -2),
            *rings,
        )

    def prepare_caches(self, caches, *, conv_window, rings):
        # FP32 temporal = the default and the published setting; BF16 = an optional lossy setting
        # (--mamba-ssm-dtype bfloat16): same FP32 math, BF16 pages, rounded once per host commit.
        if caches[0].dtype != torch.bfloat16 or caches[1].dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("KDA verify requires a BF16 conv and an FP32 (or BF16) temporal pool")
        h, d = self.heads, self.dim
        if self.weights[0].dtype != torch.float32:
            raise ValueError("K31 verify requires FP32 convolution weights")
        if tuple(caches[1].shape[1:]) != (h, 128, 128):
            raise ValueError("expected temporal [S,H,128,128]")
        if caches[1].stride()[1:] != (128 * 128, 128, 1):
            raise ValueError("stock commit requires packed temporal H/V/K axes")
        if tuple(conv_window.shape[-2:]) != (3, 3 * d):
            raise ValueError("expected conv_window [spec_rows,Qcap,3,3D]")
        if conv_window.dtype != torch.bfloat16:
            raise ValueError("conv_window must preserve BF16 input taps")
        if len(rings) != 4:
            raise ValueError("ReplaySSM needs rawv, rawk, g and beta")
        length = rings[0].shape[2]
        for tensor, tail, dtype in zip(
            rings,
            ((h, length, 128),) * 3 + ((h, length),),
            (torch.bfloat16, torch.bfloat16, torch.float32, torch.float32),
        ):
            if tuple(tensor.shape[1:]) != tail or tensor.dtype != dtype:
                raise ValueError("ReplaySSM ring shape/dtype mismatch")
            packed = (length * 128, 128, 1) if len(tail) == 3 else (length, 1)
            if tensor.stride()[1:] != packed:
                raise ValueError("stock commit requires packed ring inner axes")
        if "dmc" in verify_opt() and rings[2].stride(0) % 4:
            raise ValueError("opt dmc stores 16-byte log-g records: slot stride % 4")
        for (requests, queries), (pitch, mode) in self.policies.items():
            if requests < 1 or queries < 2 or queries > 16:
                raise ValueError("KDA ReplaySSM verify covers Q=2..16")
            if queries > min(length, conv_window.shape[1]):
                raise ValueError("verify Q exceeds ring/conv-window capacity")
            if pitch < 4 * d + 128 + h or pitch % 8:
                raise ValueError(
                    "packed projection needs its full payload and 16-byte row alignment"
                )
            projected = torch.empty(
                requests * queries, pitch, device="meta", dtype=torch.bfloat16
            )
            out = torch.empty(
                requests * queries, d, device="meta", dtype=torch.bfloat16
            )
            indices = torch.empty(requests, device="meta", dtype=torch.int32)
            ops = self._verify_operands(
                caches, projected, indices, out, indices, conv_window, rings
            )
            self.kernels[requests, queries] = executable(
                tuple(map(layout, ops)),
                h,
                requests,
                queries,
                mode,
                self.has_bias,
                self.device.index,
                verify_schedule(requests, queries, self.device, verify_opt_for(queries, verify_opt()), h),
                verify_opt_for(queries, verify_opt()),
            )
        return self

    def run(
        self,
        projected,
        out,
        *,
        caches,
        state_slots,
        scratch_rows,
        conv_window,
        rings,
        query_width,
    ):
        requests = projected.shape[0] // query_width
        ops = self._verify_operands(
            caches,
            projected,
            state_slots[:requests],
            out,
            scratch_rows[:requests],
            conv_window,
            rings,
        )
        views = tuple(to_cute(t, 16) for t in ops)
        self.kernels[requests, query_width](
            *views[:13], *self.scalars, *views[13:], stream(self.device)
        )
        return out
