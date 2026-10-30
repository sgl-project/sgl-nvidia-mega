# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Which kernel fills each slot of a covered step, per width (rows), fixed at build.

A slot runs only where the runtime's default or explicit slot policy requests it, it is
built and the rows are inside its bounds; every other op is SGLang's own. The covered steps are the
target verify of 1 to 128 requests (``REQUESTS``: every decode graph batch size SGLang
captures up to 128); a width of more than one request is covered only where a slot runs.
``route`` (the router GEMM and the routed experts' NVFP4 input in one launch) runs up to
16 rows; ``route_topk`` materializes the top-10 and NVFP4 input behind SGLang's router
GEMM at wider rows. Under TEP8, the internal routed-body plan uses route plus a
routing-only top-k launch through 16 rows, the low-M shared expert at 21/28 rows, and
the clustered route-big front from 33 through 896 rows. ``mkm`` (on route's outputs,
the top-10 and both expert GEMMs in one launch)
runs up to 7 rows, and with ``mkm_wide`` (its 16-token engine) up to 16. ``mkm_m32`` runs
the same routed-expert arithmetic on route_topk's 17–32 rows. ``gdn_verify`` (the
GatedDeltaNet verify chain with its launch choices changed) covers every width, with
``gdn_core`` (numeric: its ReplaySSM verify, gated norm and FP8 output in one launch)
up to 896 rows, and ``hop`` (the boundary's fused all-reduce points on the decode hop) the
widths where FlashInfer's low-latency kernels, which it equals bit for bit, run, or with
``hop_mid`` (numeric: its one-shot protocol past FlashInfer's) up to 28 rows.
``gdn_push`` (numeric: out_proj as the hop's push GEMM, on gdn_verify's FP8 norm) runs up
to 8 rows. ``gdn_front`` (numeric: the GDN projections and conv in one launch, then
gdn_pdl's verify) covers one request of 7 rows; ``gdn_trt_front`` replaces its projection
front through four requests. ``route_lin`` (numeric) starts the next router before the
current hidden all-reduce at exactly one request. ``route_shared`` (numeric: the shared
expert's gate/up in route's launch) up to 8 rows. ``skv2`` runs the full attention's core
of one request through the split-KV attention of SGLang branch
nvidia/qwen35-agentx-gb300, ``mka`` the fused attention prologue and epilogue around it
(up to 8 rows; with four query heads a rank, its wide entries up to 14 rows and past
them its front path, bit for bit SGLang's, at every width), and ``mka_push`` (with
``hop``) has the epilogue push its rows into the hop. ``lowm_in`` / ``lowm_out`` (numeric) run the static FP8 input / output projections on a
low-M warp-MMA launch with the weights loaded before the dependency wait, up to each one's
measured rows (``LOWM_*_MAX_ROWS``), and with ``lowm_push`` (with ``hop``) push their rows into the hop;
``lowm_se`` (numeric) the shared expert in two such launches. ``hop_fp8`` (with ``hop``) has the hop
also write the next projection's static FP8 input, which it then reads without a quantize kernel.
"""

from __future__ import annotations

import msgspec

MAX_ROWS = 896  # the widest covered step: 128 requests of 7 verify rows (TP8 bs128)
# Every decode graph size, so no batch pads to the next power of two to reach a covered
# one; each slot's own row bounds still determine whether the plugin serves the step.
REQUESTS = tuple(range(1, 129))
# the TP sizes whose steps of more than one request are covered when requested (checked at
# TP8); at the others a covered step is one request
MULTI_REQUEST_TP = (8,)
# the name that requests them in the slot list
MULTI_REQUEST = "multi_request"
GDN_FRONT_ROWS = 7  # the one-request custom front's fixed width
GDN_TRT_FRONT_MAX_ROWS = 28  # TensorRT-LLM's front, through four requests

# Each slot's widest step: its launcher's bound (sm103/moe/router.py MAX_TOKENS,
# sm103/moe/mkm.py MAX_TOKENS, sm103/hop OPROJ_MAX_M, sm103/attention/mka.py), and for
# hop the rows up to which FlashInfer's MNNVL CuTe DSL fusion runs the one-shot
# all-reduce the hop equals bit for bit (its finalize runs one-shot up to 23 rows)
ROUTE_MAX_ROWS = 16
ROUTE_TOPK_MAX_ROWS = MAX_ROWS
# The router front's 16-row tiles replace SGLang's router GEMM through two
# waves; wider route_topk rows keep the stock GEMM.
ROUTE_GEMM_MAX_ROWS = 32
GDN_CORE_MAX_ROWS = 896  # and gdn_core's launch (TRT-L4: checked at 64 and 128 requests)
MKM_MAX_ROWS = 7
MKM_WIDE_MAX_ROWS = 16
MKM_M32_MAX_ROWS = 32
MKM_M64_MAX_ROWS = 64
MKM_M128_MAX_ROWS = 128
MKM_M256_MAX_ROWS = 256
MKM_M512_MAX_ROWS = 512
HOP_MAX_ROWS = 15
# With hop_mid (numeric): the one-shot hop up to four requests in place of BT routes.
HOP_MID_MAX_ROWS = 28
GDN_PUSH_MAX_ROWS = 8
# the rows slot route_shared's shared stage fits shared memory for
ROUTE_SHARED_ROWS = 8
MKA_MAX_ROWS = 8
# with four query heads a rank: the wide entries, then the front path (attention/mka.py)
MKA_WIDE_MAX_ROWS = MAX_ROWS
# slots lowm / lowm_se: the rows up to which each low-M launch measured faster than SGLang's quantize +
# cuBLAS path on GB300 (results/trt-1022/l6): the input projections (in_proj_qkvz + in_proj_ba, qkv_proj),
# the output projections (out_proj, o_proj) and the shared expert. The shared expert's
# 21-row launch also measured faster; the 28/56-row launches stay on SGLang's stock path.
LOWM_IN_MAX_ROWS = 14
LOWM_OUTPROJ_MAX_ROWS = 14
LOWM_OPROJ_MAX_ROWS = 14
LOWM_SE_MAX_ROWS = 21
LOWM_SE_PRE_MAX_ROWS = 28
# TEP8's route-big front uses thread-block clusters to build the routed-body tables.
ROUTE_BIG_MIN_ROWS = 33
# The formal TEP8 route-big path covers every L-tier graph. Past 96 requests it uses
# TRT-LLM r3's 64-CTA launch instead of the 32-CTA launch that overlaps the shared expert.
ROUTE_BIG_MAX_ROWS = MAX_ROWS
ROUTE_BIG_WIDE_MIN_ROWS, ROUTE_BIG_WIDE_CTAS = 673, 64
# The TEP8 MoE front's row tiers (each with its TRT-LLM r3 path; PLAN.md section 4): name, widest step
MOE_FRONT_TIERS = (("S1", 8), ("S2", 16), ("M", 32), ("L1", 224), ("L2", MAX_ROWS))
# TEP8 tiers S1/S2: route's front followed by a routing-only top-k launch.
SFRONT_TOPK_MAX_ROWS = ROUTE_MAX_ROWS


def moe_front_tier(rows: int) -> str:
    """The TEP8 MoE front tier of a step of ``rows`` rows."""
    return next(
        (name for name, top in MOE_FRONT_TIERS if rows <= top), MOE_FRONT_TIERS[-1][0]
    )


def lowm_se_pre_rows(tp_size: int, tep8_m: bool = False) -> int:
    """The shared expert's widest tier-M step under the TEP8 front policy."""
    tep8 = tep8_m and tp_size == 8
    return LOWM_SE_PRE_MAX_ROWS if tep8 else 0


class Slots(msgspec.Struct, frozen=True):
    """The kernels that fill one width's step."""

    route: bool = False  # router GEMM + NVFP4 input quantization (sm103/moe/router.py)
    # behind SGLang's router GEMM: top-10 + NVFP4 input in one launch (moe/route.py)
    route_topk: bool = False
    route_gemm: bool = False  # with the router front's tiles in place of that GEMM
    # Internal TEP8 path: the MoE body on route_topk/route-big tables (moe/body.py)
    route_pre: bool = False
    # Internal TEP8 S1/S2 path: route followed by routing-only route_topk tables
    route_pre_topk: bool = False
    mkm: bool = (
        False  # top-10 + FC1 + SwiGLU + FC2 on route's outputs (sm103/moe/mkm.py)
    )
    # in route_topk's place: the 32-token engine on the router GEMM's logits
    mkm_m32: bool = False
    # in route_topk's runner: the 64-token engine on its materialized routing
    mkm_m64: bool = False
    mkm_m128: bool = False  # likewise, its 128-token engine
    mkm_m256: bool = False  # likewise, its 256-token engine
    mkm_m512: bool = False  # likewise, its 512-token engine
    gdn_verify: bool = False  # the GDN verify chain's launch choices (gdn/verify.py)
    gdn_pdl: bool = False  # its ReplaySSM verify launched dependent on the conv
    # its ReplaySSM verify, gated norm and FP8 output in one launch (sm103/gdn/core.py)
    gdn_core: bool = False
    # its conv with every x row loaded first and each window row in one store (gdn/conv.py; bitwise)
    gdn_conv: bool = False
    hop: bool = False  # the boundary's fused all-reduce points (boundary/hop.py)
    # out_proj pushes its partial into the hop from its GEMM (gdn/verify.py; numeric)
    gdn_push: bool = False
    # the projections and the conv in one launch (gdn/front.py; numeric)
    gdn_front: bool = False
    # TensorRT-LLM's front replaces the custom front through four requests (numeric)
    gdn_trt_front: bool = False
    # the shared expert's gate/up in the router launch (moe/route.py; numeric)
    route_shared: bool = False
    skv2: bool = False  # the full attention's core through SKV2 (attention/split_kv.py)
    mka: bool = False  # the fused attention prologue and epilogue (attention/mka.py)
    mka_push: bool = False  # the epilogue pushes its rows into the hop
    # the next router runs before the mixer's hidden all-reduce (numeric)
    route_lin: bool = False
    # the input projections on the low-M launch (lowm.py; numeric)
    lowm_in: bool = False
    lowm_outproj: bool = False  # GDN out_proj on the low-M launch (lowm.py; numeric)
    lowm_oproj: bool = False  # attention o_proj on the low-M launch (lowm.py; numeric)
    lowm_se: bool = False  # the shared expert in two low-M launches (lowm.py; numeric)
    # the low-M output projections push their rows into the hop (lowm.py; numeric)
    lowm_push: bool = False
    # the hop also writes the next projection's static FP8 input (lowm.py; bitwise)
    hop_fp8: bool = False


class Facts(msgspec.Struct, frozen=True):
    """What build found, which every rank must agree on. All off: every step
    function runs its stock code."""

    route: bool = False  # the router front built for every MoE layer
    route_topk: bool = (
        False  # the top-10 + NVFP4 input launch built for every MoE layer
    )
    route_gemm: bool = False  # and the router front's tiles
    tep8_moe: bool = False  # the internal S/M/L routing-table and routed-body plan
    mkm: bool = False  # the fused routed experts built for every MoE layer
    mkm_wide: bool = False  # with their 16-token engine
    mkm_m32: bool = False  # the 32-token engine built for every MoE layer
    mkm_m64: bool = False  # the 64-token engine built for every MoE layer
    mkm_m128: bool = False  # the 128-token engine built for every MoE layer
    mkm_m256: bool = False  # the 256-token engine built for every MoE layer
    mkm_m512: bool = False  # the 512-token engine built for every MoE layer
    gdn_verify: bool = False  # the GDN verify slot built for every GDN layer
    gdn_pdl: bool = False  # with its dependent verify launch
    gdn_core: bool = False  # with the one-launch verify core
    gdn_conv: bool = False  # with the hoisted conv writing each window row once
    hop: bool = False  # the decode hop, built for the widest step it covers
    hop_mid: bool = False  # with its one-shot protocol up to HOP_MID_MAX_ROWS
    gdn_push: bool = False  # out_proj packed for the push GEMM on every GDN layer
    gdn_front: bool = False  # in_proj_qkvz packed for the front on every GDN layer
    gdn_trt_front: bool = False  # TensorRT-LLM's front built for every GDN layer
    route_shared: bool = (
        False  # the router front with the shared stage, every MoE layer
    )
    skv2: bool = False  # SKV2 for every full-attention layer
    mka: bool = False  # the fused prologue and epilogue for every full-attention layer
    mka_wide: bool = False  # with its wide entries (four query heads a rank)
    mka_push: bool = False  # with the epilogue's push into the hop
    route_lin: bool = False  # the router-before-all-reduce path built for every layer
    lowm_in: bool = False  # the low-M input projections built for every layer
    lowm_out: bool = False  # the low-M output projections built for every layer
    lowm_se: bool = False  # the low-M shared expert built for every MoE layer
    lowm_se_pre: bool = False  # also at route_pre's tier-M rows
    lowm_push: bool = False  # with the output projections' push into the hop
    hop_fp8: bool = False  # the hop's FP8 copy of the norm for the next projection


def slots(rows: int, facts: Facts, tokens_per_request: int = GDN_FRONT_ROWS) -> Slots:
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"{rows} rows are outside the covered widths 1-{MAX_ROWS}")
    route = facts.route and rows <= ROUTE_MAX_ROWS
    route_topk = facts.route_topk and not route and rows <= ROUTE_TOPK_MAX_ROWS
    tep8 = facts.tep8_moe
    hop = facts.hop and rows <= (HOP_MID_MAX_ROWS if facts.hop_mid else HOP_MAX_ROWS)
    mka = facts.mka and rows <= (MKA_WIDE_MAX_ROWS if facts.mka_wide else MKA_MAX_ROWS)
    mkm = (
        route
        and facts.mkm
        and not tep8
        and rows <= (MKM_WIDE_MAX_ROWS if facts.mkm_wide else MKM_MAX_ROWS)
    )
    gdn_front_base = facts.gdn_verify and facts.gdn_pdl and facts.gdn_front
    gdn_trt_front = (
        gdn_front_base
        and facts.gdn_trt_front
        and rows % GDN_FRONT_ROWS == 0
        and rows <= GDN_TRT_FRONT_MAX_ROWS
    )
    pre_front_topk = route and tep8 and not mkm and rows <= SFRONT_TOPK_MAX_ROWS
    pre_low = route_topk and tep8 and facts.lowm_se_pre and rows <= LOWM_SE_PRE_MAX_ROWS
    lowm_se = facts.lowm_se and (rows <= LOWM_SE_MAX_ROWS or pre_low)
    # At one request lowm_se replaces both route_shared's shared stage and
    # SGLang's shared-expert tail.
    route_shared = (
        route
        and facts.route_shared
        and rows <= ROUTE_SHARED_ROWS
        and not lowm_se
        and not pre_front_topk
    )
    pre_topk = pre_low or (
        route_topk and tep8 and ROUTE_BIG_MIN_ROWS <= rows <= ROUTE_BIG_MAX_ROWS
    )
    return Slots(
        route=route,
        route_topk=route_topk,
        # TEP8's M/L bodies keep SGLang's router GEMM; route_topk or route-big
        # handles their routing and quantization.
        route_gemm=route_topk
        and facts.route_gemm
        and rows <= ROUTE_GEMM_MAX_ROWS
        and not pre_topk,
        route_pre=pre_topk,
        route_pre_topk=pre_front_topk,
        mkm=mkm,
        mkm_m32=route_topk
        and not tep8
        and facts.mkm_m32
        and MKM_WIDE_MAX_ROWS < rows <= MKM_M32_MAX_ROWS,
        mkm_m64=route_topk
        and not tep8
        and facts.mkm_m64
        and MKM_M32_MAX_ROWS < rows <= MKM_M64_MAX_ROWS,
        mkm_m128=route_topk
        and not tep8
        and facts.mkm_m128
        and MKM_M64_MAX_ROWS < rows <= MKM_M128_MAX_ROWS,
        mkm_m256=route_topk
        and not tep8
        and facts.mkm_m256
        and MKM_M128_MAX_ROWS < rows <= MKM_M256_MAX_ROWS,
        mkm_m512=route_topk
        and not tep8
        and facts.mkm_m512
        and MKM_M256_MAX_ROWS < rows <= MKM_M512_MAX_ROWS,
        gdn_verify=facts.gdn_verify,
        gdn_pdl=facts.gdn_verify and facts.gdn_pdl,
        gdn_core=facts.gdn_verify and facts.gdn_core and rows <= GDN_CORE_MAX_ROWS,
        gdn_conv=facts.gdn_verify and facts.gdn_conv,
        hop=hop,
        gdn_push=facts.gdn_verify
        and hop
        and facts.gdn_push
        and rows <= GDN_PUSH_MAX_ROWS,
        gdn_front=gdn_front_base and (rows == GDN_FRONT_ROWS or gdn_trt_front),
        gdn_trt_front=gdn_trt_front,
        route_shared=route_shared,
        # SKV2 attends one sequence
        skv2=facts.skv2 and rows <= tokens_per_request,
        mka=mka,
        mka_push=mka and facts.mka_push and hop,
        route_lin=facts.route_lin
        and rows == GDN_FRONT_ROWS
        and route
        and mkm
        and hop
        and not route_shared,
        lowm_in=facts.lowm_in and rows <= LOWM_IN_MAX_ROWS,
        lowm_outproj=facts.lowm_out and rows <= LOWM_OUTPROJ_MAX_ROWS,
        lowm_oproj=facts.lowm_out and rows <= LOWM_OPROJ_MAX_ROWS,
        lowm_se=lowm_se,
        lowm_push=facts.lowm_out
        and facts.lowm_push
        and hop
        and rows <= max(LOWM_OUTPROJ_MAX_ROWS, LOWM_OPROJ_MAX_ROWS),
        # Wherever the one-shot hop runs, including hop_mid's four-request width.
        hop_fp8=hop
        and facts.hop_fp8
        and rows <= (HOP_MID_MAX_ROWS if facts.hop_mid else HOP_MAX_ROWS),
    )


def plan(
    widths: tuple[int, ...], facts: Facts, tokens_per_request: int = GDN_FRONT_ROWS
) -> dict[int, Slots]:
    return {w: slots(w, facts, tokens_per_request) for w in widths}


def multi_request(tp_size: int, requested: frozenset) -> bool:
    """Whether steps of more than one request are covered: ``MULTI_REQUEST`` requested
    at a TP size in ``MULTI_REQUEST_TP``."""
    return MULTI_REQUEST in requested and tp_size in MULTI_REQUEST_TP


def candidates(
    offered: tuple[int, ...], tokens_per_request: int, multi: bool = True
) -> tuple[int, ...]:
    """The widths SGLang offers (``BuildContext.widths``) of ``REQUESTS`` requests, or of
    one request without ``multi``."""
    requests = REQUESTS if multi else (1,)
    return tuple(
        w
        for w in offered
        if w % tokens_per_request == 0 and w // tokens_per_request in requests
    )


def widest(widths: tuple[int, ...], max_rows: int) -> int:
    """The widest of ``widths`` a slot bounded by ``max_rows`` rows serves (0: none)."""
    return max((w for w in widths if w <= max_rows), default=0)


def covered(
    widths: tuple[int, ...], facts: Facts, tokens_per_request: int
) -> tuple[int, ...]:
    """The widths a build covers: one request always (with no slot it runs the stock
    step), more requests only where a slot runs."""
    return tuple(
        w
        for w in widths
        if w == tokens_per_request or slots(w, facts, tokens_per_request) != Slots()
    )
