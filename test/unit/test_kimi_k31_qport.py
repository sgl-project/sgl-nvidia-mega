# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Q-port stage 1 (K31MK_VERIFY_QPORT, default off): the verify contract admits Q 4-7 (3-6 drafted tokens) at the Q8
request batches, and the KDA verify build drops the Q8-only KdaSplit opts at those widths. Expected values are
hand-derived from the contract text (rows = b * q) and from KdaSplit.__init__'s own assertions (kda/kernel.py), which
the second test also runs as the oracle (constructor only: no compile, no GPU)."""
from unittest.mock import patch

import pytest

from sgl_nvidia_mega.kimi_k31.verify_contract import (
    query_width_reasons,
    served_query_widths,
    served_request_batches,
    verify_keys,
)

ON = {"K31MK_VERIFY_QPORT": "1"}


def test_qport_off_keeps_the_q8_q16_contract():
    assert served_query_widths({}) == (8, 16)
    assert served_query_widths({"K31MK_VERIFY_QPORT": "0"}) == (8, 16)
    assert verify_keys(8, (1, 2, 4), {}) == ((1, 8, 8), (2, 8, 16), (4, 8, 32))
    for q in (4, 5, 6, 7):
        with pytest.raises(ValueError):
            verify_keys(q, (1,), {})
        assert query_width_reasons(q, 1, {}) != []
        with pytest.raises(ValueError):
            served_request_batches({"K31MK_VERIFY_BATCHES": "1,2"}, query_width=q)
    assert query_width_reasons(8, 1, {}) == [] and query_width_reasons(16, 1, {}) == []


def test_qport_on_admits_q4_to_q7_at_the_q8_request_batches():
    assert served_query_widths(ON) == (8, 16, 4, 5, 6, 7)
    assert verify_keys(6, (1, 2, 3, 4), ON) == ((1, 6, 6), (2, 6, 12), (3, 6, 18), (4, 6, 24))
    assert [r for _, _, r in verify_keys(4, range(1, 9), ON)] == [4, 8, 12, 16, 20, 24, 28, 32]
    assert [r for _, _, r in verify_keys(5, (1, 8), ON)] == [5, 40]
    assert [r for _, _, r in verify_keys(7, (1, 8), ON)] == [7, 56]
    # Wide tier (Breq 9-16, rows b * q): the short widths take the same request batches as Q8 (hand-derived rows).
    assert [r for _, _, r in verify_keys(5, (1, 9, 12, 16), ON)] == [5, 45, 60, 80]
    assert [r for _, _, r in verify_keys(4, range(1, 17), ON)] == [4 * b for b in range(1, 17)]
    assert [r for _, _, r in verify_keys(6, (1, 16), ON)] == [6, 96]
    for q, batches in [(3, (1,)), (2, (1,)), (9, (1,)), (12, (1,)), (5, (1, 17)), (6, (2, 4)), (16, (1, 2)), (4, (1, 4, 2))]:
        with pytest.raises(ValueError):
            verify_keys(q, batches, ON)
    assert served_request_batches({**ON, "K31MK_VERIFY_BATCHES": "1,2,3,4"}, query_width=6) == (1, 2, 3, 4)
    assert served_request_batches({**ON, "K31MK_VERIFY_BATCHES": "1,2,4,8"}) == (1, 2, 4, 8)  # Q8 default unchanged
    for q in (4, 5, 6, 7):
        assert query_width_reasons(q, 1, ON) == []
        assert query_width_reasons(q, 8, ON) != []  # stage 1: DCP verify / TSIMO8 keep Q8
    assert query_width_reasons(8, 8, ON) == []  # DCP at Q8 is decided by spec._dcp_reasons, not here


SERVED = frozenset("early+wide+rs+lt+dmc+chunk+cg+cs+wlf".split("+"))  # XCH8R xball K31MK_KDA_VERIFY_OPT
RIDERS = frozenset("early+wide+rs+lt+dmc+wy+tsp2+g4+wlf".split("+"))  # D3G + wy riders (STATE b2), a second served form


def test_kda_verify_opt_for_hand_derived():
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda.verify import verify_opt_for

    for opt in (SERVED, RIDERS):
        assert verify_opt_for(8, opt) == opt and verify_opt_for(16, opt) == opt  # served Q8 / legacy Q16 untouched
    q8_only = {"chunk", "cg", "cs"}
    assert verify_opt_for(4, SERVED) == SERVED - q8_only  # wide: 4 % 2 == 0; rs: 2 * 2 = 4 is a power of two
    assert verify_opt_for(6, SERVED) == SERVED - q8_only - {"rs"}  # rs: 2 * 3 = 6 is not
    for q in (5, 7):  # odd: no thread halves; rs: 2 * 5 / 2 * 7 not powers of two
        assert verify_opt_for(q, SERVED) == SERVED - q8_only - {"rs", "wide"}
    assert verify_opt_for(4, RIDERS) == RIDERS - {"wy", "tsp2", "g4"}


@pytest.mark.parametrize("opt", [SERVED, RIDERS])
def test_kda_verify_builds_construct_at_short_widths(opt):
    """Oracle = KdaSplit.__init__'s own assertions with the schedule the provider would pass (SM107, TP16
    heads 6). Without verify_opt_for the served set fails at Q4 Breq 1 ("chunk" asserts tokens == 8)."""
    import torch

    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda import verify as kv
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda.kernel import KdaSplit

    class Props:
        multi_processor_count = 2 * 6 * 16  # fake SM count: every Breq <= 16 at 6 heads fits one wave

    def build(requests, queries, o):
        with patch.object(torch.cuda, "get_device_capability", return_value=(10, 7)), patch.object(
            torch.cuda, "get_device_properties", return_value=Props()
        ):
            schedule = kv.verify_schedule(requests, queries, "cpu", o, 6)
        if "early" in o:
            schedule = (1, *schedule[1:])
        return KdaSplit(6, requests, 0, False, schedule, queries, replayssm=True, opt=o)

    for queries in (4, 5, 6, 7):
        for requests in range(1, 9):
            k = build(requests, queries, kv.verify_opt_for(queries, opt))
            assert k.tokens == queries and not (k.chunk or k.wy or k.tsp or k.tsp2 or k.cgram or k.csplit)
    build(1, 8, SERVED)  # the served Q8 build itself constructs (the oracle is not vacuous)
    with pytest.raises(AssertionError):
        build(1, 4, SERVED)  # the unfiltered served set at Q4: "chunk" asserts tokens == 8
