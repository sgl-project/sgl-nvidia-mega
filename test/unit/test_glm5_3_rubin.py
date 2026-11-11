# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GLM-5.3 / GLM-5.2 on Rubin and NVL72 trays: both specs admit SM107 next to SM103
and TP8 across two nodes, and the no-multicast opt-in lets the covered steps run
where SGLang's FlashInfer all-reduce fusion does not apply."""

import pytest

from sgl_nvidia_mega.glm5_3 import draft, spec
from sgl_nvidia_mega.glm5_3.runtime import ALLOW_NO_AR_FUSION_ENV, ar_fusion_ok


@pytest.mark.parametrize("which", [spec, draft])
def test_the_specs_admit_sm103_and_sm107(which):
    assert which.SPEC.requires.isas == frozenset({"sm_103", "sm_107"})


@pytest.mark.parametrize("which", [spec, draft])
def test_the_specs_serve_tp8_across_two_trays(which):
    assert spec.Feature.MULTI_NODE_TP in which.SPEC.features


def test_the_ar_fusion_is_required_by_default(monkeypatch):
    monkeypatch.delenv(ALLOW_NO_AR_FUSION_ENV, raising=False)
    assert ar_fusion_ok(lambda rows: True, 16)
    assert not ar_fusion_ok(lambda rows: False, 16)


@pytest.mark.parametrize("value, ok", [("1", True), ("0", False)])
def test_the_no_multicast_opt_in(monkeypatch, value, ok):
    monkeypatch.setenv(ALLOW_NO_AR_FUSION_ENV, value)
    assert ar_fusion_ok(lambda rows: False, 16) is ok
