# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GLM-5.3 / GLM-5.2 in a PD-disaggregated deployment (``spec.pd_reasons``), on CPU:
the decode role is served at the TP4 profiles and at the TP8 envelope (4 or 6 tokens
per request); the TP8 prefill role and TP8 widths outside the envelope stay refused."""

from types import SimpleNamespace

import pytest

from sgl_nvidia_mega.glm5_3 import spec


def cfg(tp, tokens, mode):
    return SimpleNamespace(
        tp_size=tp,
        disaggregation_mode=mode,
        speculative_algorithm="EAGLE",
        speculative_num_draft_tokens=tokens,
    )


@pytest.mark.parametrize(
    "tp, tokens, mode",
    [
        (8, 4, "decode"),
        (8, 5, "decode"),
        (8, 6, "decode"),
        (8, 7, "decode"),
        (8, 8, "decode"),
        (4, 6, "decode"),
        (4, 3, "decode"),
        (8, 6, "null"),
        (8, 6, None),
    ],
)
def test_served(tp, tokens, mode):
    assert spec.pd_reasons(cfg(tp, tokens, mode)) == []


@pytest.mark.parametrize(
    "tp, tokens, mode",
    [
        (8, 9, "decode"),
        (8, 3, "decode"),
        (8, 6, "prefill"),
        (8, 4, "prefill"),
        (2, 6, "decode"),
    ],
)
def test_refused(tp, tokens, mode):
    (why,) = spec.pd_reasons(cfg(tp, tokens, mode))
    assert f"PD disaggregation ({mode} role)" in why
    assert f"not at TP{tp} with {tokens} tokens per request" in why
