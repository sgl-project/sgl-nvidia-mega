# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K31MK_KDA_VERIFY_B57 (default off): the schedule lookup only (CPU; the device queries are stubbed to VR200).

Expected values come from the measured forms in the verify.py comments, not from calling the function under test: served
Breq 5-7 Q8 = the Breq 8 graph's 2-CTA form (0, 8, 4, 2) (WY_TWO_CTA (0, 16, 4, 2) with "wy"); ckda131's 4-CTA form
(0, 8, 2, 4) at Breq 5 / 6 with the knob; every other (heads, Breq, Q) unchanged; a forced K31MK_KDA_VERIFY_SCHEDULE wins.
The kernel at that schedule is the one test_k31_kda.test_kda_verify gates on a GPU (FP32 reference) - not run here.
"""

import os

import pytest
import torch

from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda import verify as V

SERVED = frozenset("early+wide+rs+lt+tsp+dmc+chunk+cg+cs+wlf+b3c+dq".split("+"))
TWO_CTA, FOUR_CTA, WY2 = (0, 8, 4, 2), (0, 8, 2, 4), (0, 16, 4, 2)


@pytest.fixture
def vr200(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 7))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device=None: type("P", (), {"multi_processor_count": 2 * 6 * 16})())
    for k in ("K31MK_KDA_VERIFY_B57", "K31MK_KDA_VERIFY_SCHEDULE", "K31MK_KDA_SM107_RETUNE", "K31MK_KDA_VERIFY_SHORTQ"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def sched(requests, queries=8, opt=SERVED, heads=6):
    return V.verify_schedule(requests, queries, "cuda:0", opt, heads)


def test_off_is_served(vr200):
    assert sched(5) == TWO_CTA and sched(6) == TWO_CTA and sched(7) == TWO_CTA
    assert sched(5, opt=SERVED | {"wy"}) == WY2


def test_on_breq5_6(vr200):
    vr200.setenv("K31MK_KDA_VERIFY_B57", "1")
    assert sched(5) == FOUR_CTA and sched(6) == FOUR_CTA
    # measured as forced: the wlf / wy rewrites do not apply to it
    assert sched(5, opt=SERVED | {"wy"}) == FOUR_CTA


@pytest.mark.parametrize("requests", [1, 2, 3, 4, 7, 8, 9, 12, 13, 16, 32])
def test_on_leaves_other_breq(vr200, requests):
    off = sched(requests)
    vr200.setenv("K31MK_KDA_VERIFY_B57", "1")
    assert sched(requests) == off
    want = {1: (0, 8, 2, 8), 2: (0, 8, 2, 8), 3: V.B3_8CTA, 4: FOUR_CTA, 7: TWO_CTA, 8: TWO_CTA, 9: TWO_CTA, 12: TWO_CTA,
            13: TWO_CTA, 16: TWO_CTA, 32: V.WLF_ONE_WAVE}[requests]
    assert off == want, (requests, off, want)


def test_on_other_heads_and_q(vr200):
    vr200.setenv("K31MK_KDA_VERIFY_B57", "1")
    assert sched(5, heads=12) == TWO_CTA  # TP8: no measured entry
    assert sched(5, queries=5) == (0, 8, 2, 8)  # short Q without K31MK_KDA_VERIFY_SHORTQ: GB300's


def test_forced_wins(vr200):
    vr200.setenv("K31MK_KDA_VERIFY_B57", "1")
    vr200.setenv("K31MK_KDA_VERIFY_SCHEDULE", "8.4.2")
    assert sched(5) == TWO_CTA


def test_retune_off_keeps_gb300(vr200):
    vr200.setenv("K31MK_KDA_VERIFY_B57", "1")
    vr200.setenv("K31MK_KDA_SM107_RETUNE", "0")
    assert sched(5) == (0, 8, 2, 8)
