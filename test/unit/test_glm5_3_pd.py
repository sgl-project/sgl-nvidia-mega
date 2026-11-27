# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GLM-5.3 / GLM-5.2 in a PD-disaggregated deployment (``spec.pd_reasons``), on CPU:
the decode role is served at the TP4 profiles and at the TP8 envelope (4 or 6 tokens
per request); the TP8 prefill role and TP8 widths outside the envelope stay refused.
The TP8 decode role resolves its profile as TP4 does (the target picked once per build,
on GB300 and Rubin) and runs the TP8 families, so no ``ll.*`` unit or per-GPU plan
applies to it."""

from types import SimpleNamespace

import pytest

from sgl_nvidia_mega.glm5_3 import profile as profile_mod
from sgl_nvidia_mega.glm5_3 import spec, targets
from sgl_nvidia_mega.glm5_3.ll.target import GB300
from sgl_nvidia_mega.glm5_3.profile import TP8, Profile, profile_reasons
from sgl_nvidia_mega.glm5_3.sm107.tp4 import RUBIN


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
        (8, 6, "decode"),
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
        (8, 5, "decode"),
        (8, 3, "decode"),
        (2, 6, "decode"),
    ],
)
def test_refused(tp, tokens, mode):
    (why,) = spec.pd_reasons(cfg(tp, tokens, mode))
    assert f"PD disaggregation ({mode} role)" in why
    assert f"not at TP{tp} with {tokens} tokens per request" in why


@pytest.mark.parametrize("tokens", sorted(t for _, t in TP8))
def test_the_tp8_prefill_role_is_refused_by_role(tokens):
    (why,) = spec.pd_reasons(cfg(8, tokens, "prefill"))
    assert "PD disaggregation (prefill role)" in why
    assert "only in the decode role" in why
    assert "without --enable-nvidia-megakernel" in why
    assert "tokens per request" not in why


@pytest.mark.parametrize(
    "sm, sms, want", [((10, 3), 152, GB300), ((10, 7), 212, RUBIN)]
)
@pytest.mark.parametrize("tokens", sorted(t for _, t in TP8))
def test_the_tp8_decode_role_resolves_once_and_runs_no_unit(
    monkeypatch, sm, sms, want, tokens
):
    calls = []
    select = targets.select
    monkeypatch.setattr(
        targets, "select", lambda arch: calls.append(arch) or select(arch)
    )
    monkeypatch.setattr(profile_mod, "_device", lambda capability: (sm, sms))
    args = cfg(8, tokens, "decode")
    args.speculative_num_steps = tokens - 1
    profile = Profile.from_args(args, SimpleNamespace(num_attention_heads=64))
    assert calls == [sm] and profile.target is want
    assert profile.pd_decode and not profile.units
    assert profile.max_rows == TP8[(8, tokens)]
    assert spec.pd_reasons(args) == [] and profile_reasons(args) == []
