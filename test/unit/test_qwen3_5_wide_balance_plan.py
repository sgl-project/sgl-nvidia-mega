# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from sgl_nvidia_mega.qwen3_5 import topology

WIDE = frozenset({"route_topk", "multi_request", "mkm_m256", "mkm_m512"})


def test_tp8_plan_adds_balanced_wide_assignment():
    assert topology.plan(WIDE, topology.Topology(8)) == WIDE | {"mkm_wide_bal"}


def test_tep8_plan_keeps_the_existing_assignment():
    assert topology.plan(WIDE, topology.Topology(8, moe_ep_size=8)) == WIDE
