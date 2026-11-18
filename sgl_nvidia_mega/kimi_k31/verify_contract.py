# SPDX-License-Identifier: Apache-2.0
"""Fixed request/query axes for the K31 tensor-only verify binding."""

import os


def q5_enabled():
    """Experimental TP8/DCP8 Q5; Q8 defaults are unchanged."""
    return os.environ.get("K31MK_VERIFY_Q5", "0") == "1"


def supported_query_width(query_width):
    q = int(query_width)
    return q in (8, 16) or (q == 5 and q5_enabled())


# Requests per TARGET_VERIFY step the K31 verify serves at Q8 (Q16 is legacy Breq1 only). The
# sparse validate, the host weights view and the graph policy all read served_request_batches.
Q8_REQUEST_BATCHES = (1, 2, 3, 4, 5, 6, 7, 8)


def served_request_batches(query_width):
    q = int(query_width)
    if q == 5 and q5_enabled():
        return Q8_REQUEST_BATCHES
    return Q8_REQUEST_BATCHES if q == 8 else (1,)


def verify_keys(query_width, request_batches):
    q = int(query_width)
    batches = tuple(int(b) for b in request_batches)
    if not supported_query_width(q):
        raise ValueError(
            "K31 verify requires Q8, legacy Q16, or opt-in K31MK_VERIFY_Q5=1"
        )
    if not batches or batches != tuple(sorted(set(batches))) or batches[0] != 1:
        raise ValueError(
            "verify request capacities must be sorted, unique and include 1"
        )
    allowed = served_request_batches(q)
    if any(b not in allowed for b in batches):
        raise ValueError(f"K31 verify covers Breq {allowed} at Q{q}")
    return tuple((b, q, b * q) for b in batches)


def request_capacity(requests, request_batches):
    """Graph planning only; padding remains owned by SGLang's graph runner."""
    return next((b for b in request_batches if 0 < requests <= b), None)
