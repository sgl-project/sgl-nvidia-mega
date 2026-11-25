# SPDX-License-Identifier: Apache-2.0
"""Fixed request/query axes for the K31 tensor-only verify binding."""


def supported_query_width(query_width):
    """The served draft token counts: Q5 and Q8 (one DFlash block of 5 or 8), legacy Q16."""
    return int(query_width) in (5, 8, 16)


# Requests per TARGET_VERIFY step the K31 verify serves at Q5 and Q8 (Q16 is legacy Breq1 only). The
# sparse validate, the host weights view and the graph policy all read served_request_batches.
Q8_REQUEST_BATCHES = (1, 2, 3, 4, 5, 6, 7, 8)


def served_request_batches(query_width):
    return Q8_REQUEST_BATCHES if int(query_width) in (5, 8) else (1,)


def verify_keys(query_width, request_batches):
    q = int(query_width)
    batches = tuple(int(b) for b in request_batches)
    if not supported_query_width(q):
        raise ValueError("K31 verify requires Q5, Q8 or legacy Q16")
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
