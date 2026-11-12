# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fixed request/query axes for the K31 tensor-only verify binding."""

import os

# Q8 request batches the verify MK serves. Breq 8 (64 rows) runs the existing 64-row kernels
# (kimi_k3.providers._BATCHES). One list for the contract and every per-key check.
# Breq 3 / 5 / 6 / 7 (24 / 40 / 48 / 56 rows) get their own keys, so a boot that names them in
# K31MK_VERIFY_BATCHES captures its own graph for them instead of padding to 4 / 8. Their kernels
# take the tactics of the next served width (gemm/projection.py, kda/verify.py).
# Breq 9-16 (72-128 rows, the MID tier of the wide decode plan; port of ad05010 onto the c=8 line) are keys too:
# every row-wise buffer sizes from the boot's largest batch (providers.build_resources), and 9-15 get exact keys
# (their own graphs, no padding to 16). A boot only builds them when K31MK_VERIFY_BATCHES names them; the
# default stays "1,2,4", so c=1 / c=8 boots do not change.
Q8_REQUEST_BATCHES = tuple(range(1, 17))

# Short verify widths (Q-port stage 1, default off): K31MK_VERIFY_QPORT=1 also admits Q 4-7 (3-6 drafted tokens,
# --speculative-num-draft-tokens 4-7) at the Q8 request batches, rows b * q. Kernels whose verify form is Q8-only take
# their generic path at these widths (kda/verify.py verify_opt_for; dsa/verify.py keeps DeepGEMM logits and the stock
# selection, TSI / request sharding stay at Q8), and DCP > 1 is refused at short widths (verify_spec.py). Off (the
# default), the contract is Q8 and legacy Q16 only, exactly as before.
SHORT_QUERY_WIDTHS = (4, 5, 6, 7)


def qport_enabled(environ=None):
    return (os.environ if environ is None else environ).get("K31MK_VERIFY_QPORT", "0") == "1"


def served_query_widths(environ=None):
    """Verify widths the contract admits: Q8 and legacy Q16, plus SHORT_QUERY_WIDTHS under K31MK_VERIFY_QPORT=1."""
    return (8, 16) + (SHORT_QUERY_WIDTHS if qport_enabled(environ) else ())


def query_width_reasons(query_width, dcp_size=1, environ=None):
    """The verify width check of verify_spec.config_disable_reasons: [] when this contract serves Q ``query_width`` at
    DCP ``dcp_size`` (the DCP = TP rule itself is kimi_k31/spec.py _dcp_reasons), else the refusal reasons."""
    q = int(query_width or 0)
    if q not in served_query_widths(environ):
        return ["K31 verify requires explicit Q8 or Q16 (Q4-Q7 with K31MK_VERIFY_QPORT=1)"]
    if q in SHORT_QUERY_WIDTHS and int(dcp_size or 1) != 1:
        # Stage 1: DCP verify (dcp/verify.py) and TSIMO8 (rows % 8, dcp/tsimo8.py check_rows) keep Q8.
        return ["K31 verify Q4-Q7 (K31MK_VERIFY_QPORT=1) serves DCP1 only"]
    return []


def verify_keys(query_width, request_batches, environ=None):
    q = int(query_width)
    batches = tuple(int(b) for b in request_batches)
    if q not in served_query_widths(environ):
        raise ValueError("K31 verify requires Q8 or legacy Q16 (Q4-Q7 only with K31MK_VERIFY_QPORT=1)")
    if not batches or batches != tuple(sorted(set(batches))) or batches[0] != 1:
        raise ValueError("verify request capacities must be sorted, unique and include 1")
    allowed = (1,) if q == 16 else Q8_REQUEST_BATCHES
    if any(b not in allowed for b in batches):
        raise ValueError("K31 verify covers Breq1-16 Q8 (and Q4-Q7 under K31MK_VERIFY_QPORT=1) and legacy Breq1 Q16")
    return tuple((b, q, b * q) for b in batches)


def served_request_batches(environ=None, query_width=8):
    """Request batches a boot builds (keys, pools, graph capacities), from K31MK_VERIFY_BATCHES (a comma list).
    The default "1,2,4" is what every c=1 build has built (the host passed (1, 2, 4)); MK batch boots set "1,2,4,8"
    with --max-running-requests 8 and --cuda-graph-bs-decode 1 2 4 8 (wide tier: "1,2,...,16" with 16 and 1 2 ... 16).
    A value the contract refuses fails here, at weights_view time, not at the first verify. ``query_width``: the boot's
    Q (a short Q needs K31MK_VERIFY_QPORT=1)."""
    raw = (os.environ if environ is None else environ).get("K31MK_VERIFY_BATCHES", "1,2,4")
    try:
        batches = tuple(int(x) for x in raw.split(","))
    except ValueError:
        raise ValueError(f"K31MK_VERIFY_BATCHES must be a comma list of request batches, got {raw!r}") from None
    verify_keys(query_width, batches, environ)
    return batches


def request_capacity(requests, request_batches):
    """Graph planning only; padding remains owned by SGLang's graph runner."""
    return next((b for b in request_batches if 0 < requests <= b), None)
