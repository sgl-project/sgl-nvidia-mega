# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Which kernel fills each slot of a covered step, per width (rows), fixed at build.

A slot runs only where it is requested (``SGLANG_NVIDIA_MEGAKERNEL_SLOTS``), built and
inside its row bounds; every other op is SGLang's own. The covered steps are the
target verify of 1, 2 or 4 requests (``REQUESTS``: SGLang's decode graphs of batch
sizes 1, 2 and 4); a width of 2 or 4 requests is covered only where a slot runs there.
``route`` (the router GEMM and the routed experts' NVFP4 input in one launch) runs up to
16 rows, ``mkm`` (on its outputs, the top-10 and both expert GEMMs in one launch) up to
7, and with ``mkm_wide`` (its 16-token engine) up to 16. ``gdn_verify`` (the GatedDeltaNet verify chain with its launch choices changed) covers
every width, and ``hop`` (the boundary's fused all-reduce points on the decode hop) the
widths where FlashInfer's low-latency kernels, which it equals bit for bit, run.
``gdn_push`` (numeric: out_proj as the hop's push GEMM, on gdn_verify's FP8 norm) runs up
to 8 rows. ``gdn_front`` (numeric: the GDN projections and conv in one launch, then
gdn_pdl's verify) covers one request of 7 rows, ``route_shared`` (numeric: the shared
expert's gate/up in route's launch) up to 8 rows. ``skv2`` runs the full attention's core
of one request through the split-KV attention of SGLang branch
nvidia/qwen35-agentx-gb300, ``mka`` the fused attention prologue and epilogue around it
(up to 8 rows), and ``mka_push`` (with ``hop``) has the epilogue push its rows into the
hop.
"""

from __future__ import annotations

import msgspec

MAX_ROWS = 28  # the widest covered step: four requests of 7 verify rows (TP8 bs4)
REQUESTS = (1, 2, 4)  # the requests of a covered step (SGLang's decode graph sizes)
# the TP sizes whose steps of more than one request are covered when requested (checked at
# TP8); at the others a covered step is one request
MULTI_REQUEST_TP = (8,)
# the name that requests them in the slot list
MULTI_REQUEST = "multi_request"
GDN_FRONT_ROWS = 7  # the rows slot gdn_front's front is built for
# Each slot's widest step: its launcher's bound (sm103/moe/router.py MAX_TOKENS,
# sm103/moe/mkm.py MAX_TOKENS, sm103/hop OPROJ_MAX_M, sm103/attention/mka.py), and for
# hop the rows up to which FlashInfer's MNNVL CuTe DSL fusion runs the one-shot
# all-reduce the hop equals bit for bit (its finalize runs one-shot up to 23 rows)
ROUTE_MAX_ROWS = 16
MKM_MAX_ROWS = 7
MKM_WIDE_MAX_ROWS = 16
HOP_MAX_ROWS = 15
GDN_PUSH_MAX_ROWS = 8
# the rows slot route_shared's shared stage fits shared memory for
ROUTE_SHARED_ROWS = 8
MKA_MAX_ROWS = 8


class Slots(msgspec.Struct, frozen=True):
    """The kernels that fill one width's step."""

    route: bool = False  # router GEMM + NVFP4 input quantization (sm103/moe/router.py)
    mkm: bool = (
        False  # top-10 + FC1 + SwiGLU + FC2 on route's outputs (sm103/moe/mkm.py)
    )
    gdn_verify: bool = False  # the GDN verify chain's launch choices (gdn/verify.py)
    gdn_pdl: bool = False  # its ReplaySSM verify launched dependent on the conv
    hop: bool = False  # the boundary's fused all-reduce points (boundary/hop.py)
    # out_proj pushes its partial into the hop from its GEMM (gdn/verify.py; numeric)
    gdn_push: bool = False
    # the projections and the conv in one launch (gdn/front.py; numeric)
    gdn_front: bool = False
    # the shared expert's gate/up in the router launch (moe/route.py; numeric)
    route_shared: bool = False
    skv2: bool = False  # the full attention's core through SKV2 (attention/split_kv.py)
    mka: bool = False  # the fused attention prologue and epilogue (attention/mka.py)
    mka_push: bool = False  # the epilogue pushes its rows into the hop


class Facts(msgspec.Struct, frozen=True):
    """What build found, which every rank must agree on. All off: every step
    function runs its stock code."""

    route: bool = False  # the router front built for every MoE layer
    mkm: bool = False  # the fused routed experts built for every MoE layer
    mkm_wide: bool = False  # with their 16-token engine
    gdn_verify: bool = False  # the GDN verify slot built for every GDN layer
    gdn_pdl: bool = False  # with its dependent verify launch
    hop: bool = False  # the decode hop, built for the widest step it covers
    gdn_push: bool = False  # out_proj packed for the push GEMM on every GDN layer
    gdn_front: bool = False  # in_proj_qkvz packed for the front on every GDN layer
    route_shared: bool = (
        False  # the router front with the shared stage, every MoE layer
    )
    skv2: bool = False  # SKV2 for every full-attention layer
    mka: bool = False  # the fused prologue and epilogue for every full-attention layer
    mka_push: bool = False  # with the epilogue's push into the hop


def slots(rows: int, facts: Facts, tokens_per_request: int = GDN_FRONT_ROWS) -> Slots:
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"{rows} rows are outside the covered widths 1-{MAX_ROWS}")
    route = facts.route and rows <= ROUTE_MAX_ROWS
    hop = facts.hop and rows <= HOP_MAX_ROWS
    mka = facts.mka and rows <= MKA_MAX_ROWS
    return Slots(
        route=route,
        mkm=route
        and facts.mkm
        and rows <= (MKM_WIDE_MAX_ROWS if facts.mkm_wide else MKM_MAX_ROWS),
        gdn_verify=facts.gdn_verify,
        gdn_pdl=facts.gdn_verify and facts.gdn_pdl,
        hop=hop,
        gdn_push=facts.gdn_verify
        and hop
        and facts.gdn_push
        and rows <= GDN_PUSH_MAX_ROWS,
        gdn_front=facts.gdn_verify
        and facts.gdn_pdl
        and facts.gdn_front
        and rows == GDN_FRONT_ROWS,
        route_shared=route and facts.route_shared and rows <= ROUTE_SHARED_ROWS,
        # SKV2 attends one sequence
        skv2=facts.skv2 and rows <= tokens_per_request,
        mka=mka,
        mka_push=mka and facts.mka_push and hop,
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
