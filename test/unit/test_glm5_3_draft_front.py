# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""SGLANG_NVIDIA_MEGAKERNEL_GLM_DRAFT_FRONT (the TP8 draft's row-parallel eh_proj + the hop's
all-reduce with the input norm): the switch is off by default, its hop unit is compiled only on
request and is the residual-free norm form, and the K windows tile 2H with each inside one half."""

import os
from unittest import mock

import pytest

draft_step = pytest.importorskip("sgl_nvidia_mega.glm5_3.draft_step")
hop_api = pytest.importorskip("sgl_nvidia_mega.glm5_3.sm103.hop.api")


def test_front_mode_default_off():
    with mock.patch.dict(os.environ, {}, clear=True):
        assert draft_step.draft_front_mode() is None
    for v in ("", "0", "off"):
        with mock.patch.dict(os.environ, {draft_step.DRAFT_FRONT_ENV: v}):
            assert draft_step.draft_front_mode() is None
    with mock.patch.dict(os.environ, {draft_step.DRAFT_FRONT_ENV: "rowpar"}):
        assert draft_step.draft_front_mode() == "rowpar"
    with mock.patch.dict(os.environ, {draft_step.DRAFT_FRONT_ENV: "rowparallel"}):
        with pytest.raises(ValueError):
            draft_step.draft_front_mode()


def test_hop_unit_only_on_request():
    names = {s.name for s in hop_api.unit_specs(plain_allreduce=True)}
    assert "allreduce_rmsnorm" not in names
    units = {s.name: s for s in hop_api.unit_specs(allreduce_rmsnorm=True)}
    unit = units["allreduce_rmsnorm"]
    # the residual-free norm form: the sum is the written residual, the norm the attention input
    assert unit.obj.residual is False and unit.obj.norm is True
    assert units["allreduce_norm"].key != unit.key


def test_windows_tile_the_input_inside_one_half():
    h, w, world = draft_step.HIDDEN, draft_step.ROWPAR_WINDOW, draft_step.ROWPAR_WORLD
    assert w * world == 2 * h and w % 8 == 0
    for rank in range(world):
        lo, hi = rank * w, (rank + 1) * w
        assert (hi <= h) or (lo >= h), rank  # embedding half or hidden half, never both


def test_site_named():
    assert draft_step.SITE_EH_NORM in draft_step.SITE_NAMES
    assert len(set(draft_step.SITE_NAMES)) == len(draft_step.SITE_NAMES)
