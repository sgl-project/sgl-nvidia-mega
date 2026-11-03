# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The Qwen3.5 megakernel's spec on CPU: the setups SGLang's startup checks accept
and refuse with it, the slot requests it refuses before the load, and the slot plan."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from sglang.kernels.plugin_kernels.nvidia import spec as registry
from sglang.srt.kernel_plugins.nvidia import binding as guards
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    default_cuda_graph_config,
    with_phase,
)
from sglang.srt.server_args import ServerArgs

from sgl_nvidia_mega import plugin
from sgl_nvidia_mega.qwen3_5 import (
    TEXT_CONFIG,
    plan,
    runtime,
    spec,
)


@pytest.fixture(autouse=True)
def registered(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.delenv(plugin.PARITY_SORT_ENV, raising=False)
    plugin.register()


# The measured TP8 server (the speculative algorithm as resolved: NEXTN is EAGLE)
DOCUMENTED = dict(
    model_path="dummy",
    enable_nvidia_megakernel=True,
    tp_size=8,
    nnodes=2,
    kv_cache_dtype="fp8_e4m3",
    speculative_algorithm="EAGLE",
    speculative_num_steps=6,
    speculative_eagle_topk=1,
    speculative_num_draft_tokens=7,
    moe_runner_backend="flashinfer_trtllm",
    flashinfer_allreduce_fusion_backend="cutedsl",
    attention_backend="trtllm_mha",
    linear_attn_decode_backend="triton",
    enable_linear_replayssm_spec=True,
    mamba_ssm_dtype="bfloat16",
    enable_hierarchical_cache=True,
    cuda_graph_config=default_cuda_graph_config(),
)


def _refusal(text=None, dtype="bfloat16", isa="sm_103", **args) -> str:
    """SGLang's resolution step on these arguments: "" when it accepts them, else its
    error message."""
    hf_config = SimpleNamespace(
        architectures=[spec.ARCHITECTURE],
        model_type="qwen3_5_moe",
        text_config=SimpleNamespace(**{**TEXT_CONFIG, **(text or {})}),
    )
    with (
        patch.object(guards, "plugin_reasons", return_value=[]),
        patch.object(guards, "device_isa", return_value=isa),
        patch.object(guards, "model_config_of") as model_config,
    ):
        model_config.return_value = SimpleNamespace(hf_config=hf_config, dtype=dtype)
        try:
            guards.handle_nvidia_megakernel(ServerArgs(**{**DOCUMENTED, **args}))
        except ValueError as e:
            return str(e)
    return ""


def test_the_documented_setup_is_accepted():
    assert _refusal() == ""


def test_tp4_on_one_tray_is_accepted():
    assert _refusal(tp_size=4, nnodes=1) == ""


BREAKABLE = with_phase(
    default_cuda_graph_config(), Phase.DECODE, backend=Backend.BREAKABLE
)


@pytest.mark.parametrize(
    "kw, want",
    [
        (dict(tp_size=2, nnodes=1), "tp_size 2"),
        (dict(pp_size=2), "pipeline parallelism is in use"),
        (dict(enable_dp_attention=True, dp_size=8), "DP attention is in use"),
        (dict(kv_cache_dtype="auto"), "kv_cache_dtype 'auto'"),
        (dict(speculative_algorithm=None), "speculative_algorithm None"),
        (dict(speculative_eagle_topk=4), "speculative_eagle_topk 4"),
        (
            dict(speculative_use_rejection_sampling=True),
            "speculative_use_rejection_sampling True",
        ),
        (dict(moe_runner_backend="flashinfer_cutedsl"), "moe_runner_backend"),
        (dict(flashinfer_allreduce_fusion_backend=None), "flashinfer_allreduce_fusion"),
        (dict(attention_backend="triton"), "attention_backend 'triton'"),
        (dict(linear_attn_decode_backend="flashinfer"), "linear_attn_decode_backend"),
        (dict(enable_linear_replayssm_spec=False), "enable_linear_replayssm_spec"),
        (dict(mamba_ssm_dtype="float32"), "mamba_ssm_dtype 'float32'"),
        (dict(startup_weight_load_mode="overlap"), "startup_weight_load_mode"),
        (dict(cuda_graph_config=BREAKABLE), "decode CUDA graphs"),
        (dict(text=dict(num_experts=256)), "config.text_config.num_experts 256"),
        (dict(text=dict(hidden_size=5120)), "config.text_config.hidden_size 5120"),
        (dict(dtype="float16"), "dtype 'float16'"),
        (dict(isa="sm_100"), "not sm_100"),
    ],
)
def test_each_refusal_names_its_reason(kw, want):
    assert want in _refusal(**kw)


def test_an_sglang_without_the_eagle_forward_view_is_refused(monkeypatch):
    # an import of this name now raises ImportError
    monkeypatch.setitem(
        sys.modules, "sglang.srt.kernel_plugins.nvidia.eagle_binding", None
    )
    assert "this SGLang has no EAGLE forward view" in _refusal()


@pytest.mark.parametrize(
    "slots, want",
    [
        ({"hop2"}, "unknown slot 'hop2'"),
        ({"verify_sample"}, "'verify_sample' is no longer an entry: drop it"),
        ({"mkm"}, "slot 'mkm' runs on slot 'route''s outputs"),
        ({"gdn_pdl"}, "slot 'gdn_pdl' launches slot 'gdn_verify''s verify"),
        (
            {"gdn_verify", "gdn_push"},
            "slot 'gdn_push' pushes slot 'gdn_verify''s out_proj into slot 'hop'",
        ),
        (
            {"gdn_verify", "gdn_front"},
            "slot 'gdn_front' runs slot 'gdn_pdl''s dependent verify",
        ),
        ({"route_shared"}, "slot 'route_shared' runs in slot 'route''s launch"),
        ({"route", "mkm_wide"}, "slot 'mkm_wide' is slot 'mkm''s 16-token engine"),
        (
            {"route", "mkm", "mkm_wide"},
            "slot 'mkm_wide' serves the steps of more than one request ('multi_request')",
        ),
    ],
)
def test_a_slot_set_that_cannot_run_is_refused_before_the_load(
    monkeypatch, slots, want
):
    monkeypatch.setattr(runtime, "SLOTS", frozenset(slots))
    assert want in _refusal()


def test_every_reason_is_listed_at_once():
    message = _refusal(tp_size=2, nnodes=1, kv_cache_dtype="auto", pp_size=2)
    assert message.count("\n  - ") == 3, message


def test_the_slot_plan_covers_one_request_of_7_rows():
    assert plan.plan((7,), plan.Facts()) == {7: plan.Slots()}
    assert plan.plan((7,), plan.Facts(route=True)) == {7: plan.Slots(route=True)}
    assert plan.plan((7,), plan.Facts(route=True, mkm=True)) == {
        7: plan.Slots(route=True, mkm=True)
    }
    assert plan.plan((7,), plan.Facts(hop=True)) == {7: plan.Slots(hop=True)}
    # gdn_push pushes gdn_verify's out_proj into the hop: it needs both built
    both = dict(gdn_verify=True, hop=True)
    assert plan.plan((7,), plan.Facts(**both, gdn_push=True)) == {
        7: plan.Slots(**both, gdn_push=True)
    }
    assert not plan.slots(7, plan.Facts(gdn_verify=True, gdn_push=True)).gdn_push
    # gdn_front runs gdn_pdl's verify, for one request of 7 rows only
    front = dict(gdn_verify=True, gdn_pdl=True, gdn_front=True)
    assert plan.plan((7,), plan.Facts(**front)) == {7: plan.Slots(**front)}
    assert not plan.slots(4, plan.Facts(**front)).gdn_front
    assert not plan.slots(7, plan.Facts(gdn_verify=True, gdn_front=True)).gdn_front
    # route_shared rides route's launch, up to 8 rows
    shared = dict(route=True, route_shared=True)
    assert plan.plan((7,), plan.Facts(**shared)) == {7: plan.Slots(**shared)}
    assert not plan.slots(7, plan.Facts(route_shared=True)).route_shared
    with pytest.raises(ValueError):
        plan.slots(plan.MAX_ROWS + 1, plan.Facts())


def _on(slots):
    return sorted(name for name in plan.Slots.__struct_fields__ if getattr(slots, name))


def test_the_slot_plan_gates_each_slot_by_the_rows_its_kernels_serve():
    # every slot built except mkm's 16-token engine (mkm_wide, below)
    every = plan.Facts(
        **{**dict.fromkeys(plan.Facts.__struct_fields__, True), "mkm_wide": False}
    )
    assert _on(plan.slots(7, every)) == sorted(plan.Slots.__struct_fields__)
    # two requests: the 16-row router front, the GDN chain and the hop's one-shot rows
    assert _on(plan.slots(14, every)) == ["gdn_pdl", "gdn_verify", "hop", "route"]
    # four requests: past the router front's and the one-shot all-reduce's rows
    assert _on(plan.slots(28, every)) == ["gdn_pdl", "gdn_verify"]
    # SGLang offers every multiple of 7 up to MAX_ROWS; three requests are not covered
    offered = tuple(range(7, plan.MAX_ROWS + 1, 7))
    assert plan.candidates(offered, 7) == (7, 14, 28)
    assert plan.candidates(offered, 7, multi=False) == (7,)
    # more than one request only when requested, and only at TP8; TP4 keeps the
    # one-request step it is checked at
    assert plan.multi_request(8, frozenset({"multi_request"}))
    assert not plan.multi_request(8, frozenset())
    assert not plan.multi_request(4, frozenset({"multi_request"}))
    assert plan.covered((7, 14, 28), every, 7) == (7, 14, 28)
    # one request is always covered, more only where a slot runs
    assert plan.covered((7, 14, 28), plan.Facts(), 7) == (7,)
    assert plan.covered((7, 14, 28), plan.Facts(route=True, mkm=True), 7) == (7, 14)
    attention = plan.Facts(skv2=True, mka=True, mka_push=True, hop=True)
    assert plan.covered((7, 14, 28), attention, 7) == (7, 14)
    assert _on(plan.slots(14, attention)) == ["hop"]
    # mkm_wide: mkm's 16-token engine at two requests, not past the router front's rows
    wide = plan.Facts(route=True, mkm=True, mkm_wide=True)
    assert _on(plan.slots(14, wide)) == ["mkm", "route"]
    assert _on(plan.slots(28, wide)) == []
    assert _on(plan.slots(14, plan.Facts(route=True, mkm=True))) == ["route"]
    # the router front and the hop's mailbox are built for the widest width they serve
    assert plan.widest((7, 14, 28), plan.ROUTE_MAX_ROWS) == 14
    assert plan.widest((7, 14, 28), plan.HOP_MAX_ROWS) == 14
    assert plan.widest((28,), plan.HOP_MAX_ROWS) == 0


def test_the_slot_plans_row_bounds_are_the_kernels():
    from sgl_nvidia_mega.qwen3_5.sm103 import hop
    from sgl_nvidia_mega.qwen3_5.sm103.attention import mka
    from sgl_nvidia_mega.qwen3_5.sm103.moe import mkm, router

    assert plan.ROUTE_MAX_ROWS == router.MAX_TOKENS
    assert plan.ROUTE_SHARED_ROWS == router.SHARED_MAX_M
    assert plan.MKM_MAX_ROWS == mkm.MAX_TOKENS
    assert plan.MKM_WIDE_MAX_ROWS == mkm.M16_MAX_TOKENS
    assert plan.GDN_PUSH_MAX_ROWS == hop.OPROJ_MAX_M
    assert plan.MKA_MAX_ROWS == mka.MAX_ROWS


def test_a_numeric_slot_turns_the_claim_into_its_tolerances():
    from sglang.kernels.plugin_kernels.nvidia.spec import NumericsKind

    assert spec.numerics(frozenset({"route", "hop"})).kind is NumericsKind.BITWISE
    claim = spec.numerics(frozenset({"gdn_verify", "hop", "gdn_push"}))
    assert claim.kind is NumericsKind.TOLERANCE
    assert dict(claim.tolerances) == spec.NUMERIC_TOLERANCES["gdn_push"]
    shared = spec.numerics(frozenset({"route", "route_shared"}))
    assert dict(shared.tolerances) == spec.NUMERIC_TOLERANCES["route_shared"]
    both = spec.numerics(frozenset({"gdn_verify", "gdn_pdl", "gdn_front", "gdn_push"}))
    assert dict(both.tolerances) == {
        **spec.NUMERIC_TOLERANCES["gdn_push"],
        **spec.NUMERIC_TOLERANCES["gdn_front"],
    }
    attention = spec.numerics(frozenset({"skv2", "mka", "mka_push", "hop"}))
    assert attention.kind is NumericsKind.TOLERANCE
    assert dict(attention.tolerances) == {
        **spec.NUMERIC_TOLERANCES["skv2"],
        **spec.NUMERIC_TOLERANCES["mka"],
    }
