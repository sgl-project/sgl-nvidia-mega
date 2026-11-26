# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The one-request serving step on CPU: at attention TP8 + MoE TP4 x EP2 with one request,
resolution sets the v41_new KV layout the user left unset, and without
--ep-num-redundant-experts logs that slot moe_balanced is idle; any other shape, a layout
the user set, or a slot list without attn_core_lean leaves the layout alone."""

import logging
import os
from types import SimpleNamespace

import pytest

pytest.importorskip("sglang.srt.arg_groups.resolution_hooks")

from sglang.srt.arg_groups import model_override_base  # noqa: E402

from sgl_nvidia_mega.deepseek_v41 import runtime, serving  # noqa: E402

LAYOUT = "SGLANG_DSV4_KV_LAYOUT"
IDLE = "slot moe_balanced is idle"


def _server(**over):
    fields = dict(
        enable_nvidia_megakernel=True,
        tp_size=8,
        ep_size=2,
        max_running_requests=1,
        disaggregation_mode="null",
        ep_num_redundant_experts=0,
        enable_eplb=False,
        init_expert_location="trivial",
    )
    fields.update(over)
    return SimpleNamespace(**fields)


@pytest.fixture
def resolve(monkeypatch, caplog):
    """choose() over a stand-in resolved view of a DeepSeek-V4.1 server, the layout unset;
    returns (what was set, the layout after, whether the idle note was logged)."""
    hf_config = SimpleNamespace(
        architectures=["DeepseekV4ForCausalLM"],
        model_type="deepseek_v41",
        n_routed_experts=384,
    )
    monkeypatch.setattr(model_override_base, "resolving_view", lambda sa: sa)
    monkeypatch.setattr(
        model_override_base,
        "model_config_of",
        lambda sa: SimpleNamespace(hf_config=hf_config),
    )
    monkeypatch.setattr(runtime, "SLOTS", frozenset(runtime.SHIPPED_SLOTS))
    monkeypatch.setenv(LAYOUT, "x")  # restored (unset) at teardown
    monkeypatch.delenv(LAYOUT)

    def run(server):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=serving.__name__):
            chosen = serving.choose(server)
        return chosen, os.environ.get(LAYOUT), IDLE in caplog.text

    return run


def test_one_request_at_tp8_takes_the_layout(resolve):
    chosen, layout, idle = resolve(_server())
    assert layout == "v41_new" and len(chosen) == 1
    assert idle  # the redundant experts stay the user's choice


def test_with_the_redundant_experts_no_idle_note(resolve):
    chosen, layout, idle = resolve(_server(ep_num_redundant_experts=384))
    assert layout == "v41_new" and not idle


@pytest.mark.parametrize(
    "over",
    [
        dict(max_running_requests=2),
        dict(max_running_requests=None),
        dict(tp_size=4, ep_size=1),
        dict(disaggregation_mode="decode"),
        dict(enable_nvidia_megakernel=False),
    ],
    ids=["two_requests", "request_limit_unset", "tp4", "pd_decode", "megakernel_off"],
)
def test_another_shape_keeps_sglang_s_defaults(resolve, over):
    assert resolve(_server(**over)) == ([], None, False)


def test_a_layout_the_user_set_is_kept(resolve, monkeypatch):
    monkeypatch.setenv(LAYOUT, "v41")
    assert resolve(_server())[:2] == ([], "v41")


def test_a_slot_list_without_the_slots_keeps_the_layout(resolve, monkeypatch):
    rest = set(runtime.SHIPPED_SLOTS) - {"attn_core_lean", "moe_balanced"}
    monkeypatch.setattr(runtime, "SLOTS", frozenset(rest))
    assert resolve(_server()) == ([], None, False)


def test_the_step_runs_ahead_of_megakernel_admission(monkeypatch):
    from sglang.srt.arg_groups import resolution_hooks

    monkeypatch.setattr(resolution_hooks, "_HOOKS", {})
    monkeypatch.setattr(serving, "_registered", False)
    serving.register()
    serving.register()
    assert resolution_hooks._HOOKS == {serving.STEP: [serving._step]}
