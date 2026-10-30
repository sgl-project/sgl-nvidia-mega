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


def test_the_decode_side_of_pd_is_accepted():
    assert _refusal(disaggregation_mode="decode") == ""


def test_the_prefill_side_of_pd_is_refused():
    # SGLang refuses ReplaySSM on a PD prefill server itself
    message = _refusal(
        disaggregation_mode="prefill", enable_linear_replayssm_spec=False
    )
    assert "a PD prefill server runs no covered step" in message, message


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
        (
            {"gdn_trt_front"},
            "slot 'gdn_trt_front' replaces slot 'gdn_front' in front of slot "
            "'gdn_core'",
        ),
        (
            {"route_lin"},
            "slot 'route_lin' runs the router before the all-reduce",
        ),
        (
            {
                "route",
                "mkm",
                "hop",
                "gdn_verify",
                "gdn_push",
                "mka",
                "mka_push",
                "route_lin",
                "route_shared",
            },
            "slot 'route_lin' needs 'lowm_se' to replace 'route_shared'",
        ),
        ({"route_shared"}, "slot 'route_shared' runs in slot 'route''s launch"),
        ({"hop_mid"}, "slot 'hop_mid' widens slot 'hop'"),
        ({"route", "mkm_wide"}, "slot 'mkm_wide' is slot 'mkm''s 16-token engine"),
        (
            {"route", "mkm", "mkm_wide"},
            "slot 'mkm_wide' serves the steps of more than one request ('multi_request')",
        ),
        ({"mkm_m32"}, "slot 'mkm_m32' takes slot 'route_topk''s rows"),
        ({"route_pre"}, "unknown slot 'route_pre'"),
        ({"route_pre_front"}, "unknown slot 'route_pre_front'"),
        (
            {"route_topk", "mkm_m32"},
            "slot 'mkm_m32' serves the steps of more than two requests ('multi_request')",
        ),
        ({"mkm_m64"}, "slot 'mkm_m64' runs on slot 'route_topk''s routing"),
        (
            {"route_topk", "mkm_m64"},
            "slot 'mkm_m64' serves the steps of more than four requests ('multi_request')",
        ),
        ({"mkm_m128"}, "slot 'mkm_m128' runs on slot 'route_topk''s routing"),
        (
            {"route_topk", "mkm_m128"},
            "slot 'mkm_m128' serves the steps of more than nine requests ('multi_request')",
        ),
        ({"mkm_m256"}, "slot 'mkm_m256' runs on slot 'route_topk''s routing"),
        (
            {"route_topk", "mkm_m256"},
            "slot 'mkm_m256' serves the steps of more than 18 requests ('multi_request')",
        ),
        ({"mkm_m512"}, "slot 'mkm_m512' runs on slot 'route_topk''s routing"),
        (
            {"route_topk", "mkm_m512"},
            "slot 'mkm_m512' serves the steps of more than 36 requests ('multi_request')",
        ),
        (
            {"mkm_wide_bal"},
            "slot 'mkm_wide_bal' changes the work split of slot 'mkm_m256' or "
            "'mkm_m512'",
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
    # lowm_se replaces route_shared's shared stage where both could run.
    assert _on(plan.slots(7, plan.Facts(**shared, lowm_se=True))) == [
        "lowm_se",
        "route",
    ]
    with pytest.raises(ValueError):
        plan.slots(plan.MAX_ROWS + 1, plan.Facts())


def _on(slots):
    return sorted(name for name in plan.Slots.__struct_fields__ if getattr(slots, name))


def test_the_trt_gdn_front_widens_slot_gdn_front_to_four_requests():
    """The custom front serves one request; the explicit TRT front serves four."""
    front = plan.Facts(gdn_verify=True, gdn_pdl=True, gdn_front=True)
    assert plan.slots(7, front).gdn_front
    assert not plan.slots(14, front).gdn_front
    trt = plan.Facts(
        gdn_verify=True,
        gdn_pdl=True,
        gdn_front=True,
        gdn_trt_front=True,
    )
    for rows in (7, 14, 21, 28):
        slots = plan.slots(rows, trt)
        assert slots.gdn_front and slots.gdn_trt_front
    assert not plan.slots(35, front).gdn_front
    assert not plan.slots(35, trt).gdn_front


def test_the_slot_plan_gates_each_slot_by_the_rows_its_kernels_serve():
    # Every public slot is built except mkm's 16-token engine (mkm_wide, below).
    every = plan.Facts(
        **{
            **dict.fromkeys(plan.Facts.__struct_fields__, True),
            "mkm_wide": False,
            "tep8_moe": False,
        }
    )
    assert _on(plan.slots(7, every)) == sorted(
        set(plan.Slots.__struct_fields__)
        - {
            "mkm_m32",
            "mkm_m64",
            "mkm_m128",
            "mkm_m256",
            "mkm_m512",
            "route_gemm",
            "route_pre",
            "route_pre_topk",
            "route_shared",
            "route_topk",
        }
    )
    # Two requests also take every measured low-M launch and the hop's FP8 copy.
    assert _on(plan.slots(14, every)) == [
        "gdn_conv",
        "gdn_core",
        "gdn_front",
        "gdn_pdl",
        "gdn_trt_front",
        "gdn_verify",
        "hop",
        "hop_fp8",
        "lowm_in",
        "lowm_oproj",
        "lowm_outproj",
        "lowm_push",
        "lowm_se",
        "mka",
        "mka_push",
        "route",
    ]
    # Three requests keep only the shared expert's validated low-M launch.
    three = plan.slots(21, every)
    assert three.lowm_se
    assert three.gdn_front and three.gdn_trt_front
    assert three.hop and three.hop_fp8 and three.mka_push
    assert not (three.lowm_in or three.lowm_outproj or three.lowm_oproj)
    # The 28/56-row low-M projection paths stay stock; hop_mid's one-shot path and
    # FP8 copy also serve 28 rows, while GDN core and MKA serve through 896.
    for rows in (28, 56, 112, 224):
        expected = [
            "gdn_conv",
            "gdn_core",
            "gdn_pdl",
            "gdn_verify",
            "mka",
            "route_topk",
        ]
        if rows <= plan.GDN_TRT_FRONT_MAX_ROWS:
            expected.extend(("gdn_front", "gdn_trt_front"))
        if rows <= plan.MKM_M32_MAX_ROWS:
            expected.append("mkm_m32")
        if plan.MKM_M32_MAX_ROWS < rows <= plan.MKM_M64_MAX_ROWS:
            expected.append("mkm_m64")
        if plan.MKM_M64_MAX_ROWS < rows <= plan.MKM_M128_MAX_ROWS:
            expected.append("mkm_m128")
        if plan.MKM_M128_MAX_ROWS < rows <= plan.MKM_M256_MAX_ROWS:
            expected.append("mkm_m256")
        if rows <= plan.ROUTE_GEMM_MAX_ROWS:
            expected.append("route_gemm")
        if rows <= plan.HOP_MID_MAX_ROWS:
            expected.extend(("hop", "hop_fp8", "mka_push"))
        assert _on(plan.slots(rows, every)) == sorted(expected)
    for rows in (448, 896):
        expected = [
            "gdn_conv",
            "gdn_core",
            "gdn_pdl",
            "gdn_verify",
            "mka",
            "route_topk",
        ]
        if rows <= plan.MKM_M512_MAX_ROWS:
            expected.append("mkm_m512")
        assert _on(plan.slots(rows, every)) == sorted(expected)
    assert _on(plan.slots(28, plan.Facts(route_topk=True))) == ["route_topk"]
    assert _on(plan.slots(7, plan.Facts(route_topk=True))) == ["route_topk"]
    assert not plan.slots(28, plan.Facts(route_gemm=True)).route_gemm
    mid = plan.Facts(hop=True, hop_mid=True)
    assert _on(plan.slots(28, mid)) == ["hop"]
    assert _on(plan.slots(56, mid)) == []
    assert _on(plan.slots(28, plan.Facts(hop=True, hop_mid=True, hop_fp8=True))) == [
        "hop",
        "hop_fp8",
    ]
    # TEP8's routed-body plan is internal: S1/S2 use route plus a routing-only
    # top-k launch, M uses the low-M shared expert, and L uses route-big. It
    # supersedes the all-expert MKM engines and the numeric tiled router front.
    tep8 = dict(route=True, route_topk=True, tep8_moe=True)
    for rows in (7, 14):
        assert _on(plan.slots(rows, plan.Facts(**tep8))) == [
            "route",
            "route_pre_topk",
        ]
    for rows in (21, 28):
        assert _on(plan.slots(rows, plan.Facts(**tep8))) == ["route_topk"]
    tier_m = plan.Facts(
        **tep8,
        route_gemm=True,
        lowm_se=True,
        lowm_se_pre=True,
    )
    for rows in (21, 28):
        assert _on(plan.slots(rows, tier_m)) == [
            "lowm_se",
            "route_pre",
            "route_topk",
        ]
    tier_l = plan.Facts(
        **tep8,
        route_gemm=True,
        mkm=True,
        mkm_wide=True,
        mkm_m32=True,
        mkm_m64=True,
        mkm_m128=True,
        mkm_m256=True,
        mkm_m512=True,
    )
    for rows in (35, 112, 224, 448, 672, 896):
        assert _on(plan.slots(rows, tier_l)) == ["route_pre", "route_topk"]
    assert plan.lowm_se_pre_rows(8, tep8_m=True) == 28
    assert plan.lowm_se_pre_rows(4, tep8_m=True) == 0
    assert [plan.moe_front_tier(r) for r in (7, 14, 28, 35, 224, 231, 896)] == [
        "S1",
        "S2",
        "M",
        "L1",
        "L1",
        "L2",
        "L2",
    ]
    # The S tiers run beside the low-M shared expert and replace route_shared.
    s_tier = plan.Facts(
        **tep8,
        route_shared=True,
        lowm_se=True,
        mkm=True,
    )
    for rows in (7, 14):
        assert _on(plan.slots(rows, s_tier)) == ["lowm_se", "route", "route_pre_topk"]
    assert _on(plan.slots(21, s_tier)) == ["lowm_se", "route_topk"]
    # SGLang offers every multiple of 7 up to MAX_ROWS; every request count is a candidate
    offered = tuple(range(7, plan.MAX_ROWS + 1, 7))
    assert plan.candidates(offered, 7) == offered
    assert plan.candidates(offered + (903,), 7) == offered
    assert plan.candidates(offered, 7, multi=False) == (7,)
    # more than one request only when requested, and only at TP8; TP4 keeps the
    # one-request step it is checked at
    assert plan.multi_request(8, frozenset({"multi_request"}))
    assert not plan.multi_request(8, frozenset())
    assert not plan.multi_request(4, frozenset({"multi_request"}))
    assert plan.covered((7, 14, 28), every, 7) == (7, 14, 28)
    assert plan.covered((7, 448, 896), every, 7) == (7, 448, 896)
    # one request is always covered, more only where a slot runs
    assert plan.covered((7, 14, 28), plan.Facts(), 7) == (7,)
    assert plan.covered((7, 14, 28), plan.Facts(route=True, mkm=True), 7) == (7, 14)
    attention = plan.Facts(skv2=True, mka=True, mka_push=True, hop=True)
    assert plan.covered((7, 14, 28), attention, 7) == (7, 14)
    assert _on(plan.slots(14, attention)) == ["hop"]
    wide_attention = plan.Facts(mka=True, mka_wide=True)
    assert plan.covered((7, 14, 28, 56, 112, 224), wide_attention, 7) == (
        7,
        14,
        28,
        56,
        112,
        224,
    )
    # mkm_wide: mkm's 16-token engine at two requests, not past the router front's rows
    wide = plan.Facts(route=True, mkm=True, mkm_wide=True)
    assert _on(plan.slots(14, wide)) == ["mkm", "route"]
    assert _on(plan.slots(28, wide)) == []
    assert _on(plan.slots(14, plan.Facts(route=True, mkm=True))) == ["route"]
    m32 = plan.Facts(route_topk=True, mkm_m32=True)
    assert _on(plan.slots(28, m32)) == ["mkm_m32", "route_topk"]
    assert _on(plan.slots(14, m32)) == ["route_topk"]
    assert _on(plan.slots(56, m32)) == ["route_topk"]
    assert _on(plan.slots(28, plan.Facts(mkm_m32=True))) == []
    m64 = plan.Facts(route_topk=True, mkm_m64=True)
    assert _on(plan.slots(56, m64)) == ["mkm_m64", "route_topk"]
    assert _on(plan.slots(28, m64)) == ["route_topk"]
    assert _on(plan.slots(112, m64)) == ["route_topk"]
    assert _on(plan.slots(56, plan.Facts(mkm_m64=True))) == []
    m128 = plan.Facts(route_topk=True, mkm_m128=True)
    assert _on(plan.slots(112, m128)) == ["mkm_m128", "route_topk"]
    assert _on(plan.slots(56, m128)) == ["route_topk"]
    assert _on(plan.slots(224, m128)) == ["route_topk"]
    assert _on(plan.slots(112, plan.Facts(mkm_m128=True))) == []
    wide = plan.Facts(route_topk=True, mkm_m256=True, mkm_m512=True)
    assert _on(plan.slots(224, wide)) == ["mkm_m256", "route_topk"]
    assert _on(plan.slots(448, wide)) == ["mkm_m512", "route_topk"]
    assert _on(plan.slots(112, wide)) == ["route_topk"]
    assert _on(plan.slots(896, wide)) == ["route_topk"]
    # the router front and the hop's mailbox are built for the widest width they serve
    assert plan.widest((7, 14, 28), plan.ROUTE_MAX_ROWS) == 14
    assert plan.widest((7, 14, 28), plan.HOP_MAX_ROWS) == 14
    assert plan.widest((28,), plan.HOP_MAX_ROWS) == 0


def test_the_slot_plans_row_bounds_are_the_kernels():
    from sgl_nvidia_mega.qwen3_5.sm103 import hop
    from sgl_nvidia_mega.qwen3_5.sm103.attention import mka
    from sgl_nvidia_mega.qwen3_5.sm103.moe import mkm, router

    assert plan.ROUTE_MAX_ROWS == router.MAX_TOKENS
    assert plan.ROUTE_TOPK_MAX_ROWS == router.TOPK_MAX_ROWS
    assert plan.ROUTE_SHARED_ROWS == router.SHARED_MAX_M
    assert plan.MKM_MAX_ROWS == mkm.MAX_TOKENS
    assert plan.MKM_WIDE_MAX_ROWS == mkm.M16_MAX_TOKENS
    assert plan.MKM_M32_MAX_ROWS == mkm.M32_MAX_TOKENS
    assert plan.MKM_M64_MAX_ROWS == mkm.M64_MAX_TOKENS
    assert plan.MKM_M128_MAX_ROWS == mkm.M128_MAX_TOKENS
    assert plan.MKM_M256_MAX_ROWS == mkm.M256_MAX_TOKENS
    assert plan.MKM_M512_MAX_ROWS == mkm.M512_MAX_TOKENS
    assert plan.GDN_PUSH_MAX_ROWS == hop.OPROJ_MAX_M
    assert plan.MKA_MAX_ROWS == mka.MAX_ROWS
    from sgl_nvidia_mega.qwen3_5.attention import mka as mka_step

    assert plan.MKA_WIDE_MAX_ROWS == plan.MAX_ROWS
    assert mka_step.FUSED_MAX_ROWS <= mka.WIDE_MAX_ROWS


def test_the_validated_moe_schedules_are_shape_closed():
    from sgl_nvidia_mega.qwen3_5.sm103.moe import mkm

    assert mkm.MKM7_DEFINES == (
        "-DSPEC_STAGES=3",
        "-DPACE_D=4",
        "-DFC2SF_PRE=0",
    )
    assert mkm.engine_options(32) == (
        "-DTHR_D=2",
        "-DTOPK_NOINLINE=1",
        "-DONEPASS=1",
        "-DMMA_SFONLY=1",
    )
    assert mkm.engine_options(64) == ("-DMMA_SFONLY=1", "-DONEPASS=1")
    assert mkm.engine_options(64, plain=True) == ("-DMMA_SFONLY=1",)
    assert [mkm.reserved_sms(32, rows) for rows in (21, 22)] == [0, 32]
    assert [mkm.reserved_sms(64, rows) for rows in (35, 42, 43)] == [32, 16, 0]
    assert not mkm.plain_schedule(64, 42)
    assert mkm.plain_schedule(64, 43)


@pytest.mark.parametrize(
    "rows", [56, 112, 126, 128, 129, 133, 140, 168, 224, 256, 280, 512]
)
def test_bal_changes_only_the_wide_engine_schedule(rows):
    from sgl_nvidia_mega.qwen3_5.sm103.moe import mkm

    engine = mkm.engine_rows(rows)
    off = mkm.engine_options(engine)
    on = mkm.engine_options(engine, wide_bal=True)
    assert on == off + (("-DBAL_SPLIT=1",) if rows > 128 else ())
    assert mkm.plain_schedule(engine, rows, wide_bal=True) == mkm.plain_schedule(
        engine, rows
    )
    assert mkm.engine_options(engine, plain=True, wide_bal=True) == (
        mkm.engine_options(engine, plain=True)
    )


@pytest.mark.parametrize(
    "rows,plain", [(129, False), (210, False), (211, True), (280, True)]
)
def test_bal_internal_row_cap_selects_the_static_split(monkeypatch, rows, plain):
    from sgl_nvidia_mega.qwen3_5.sm103.moe import mkm

    monkeypatch.setattr(mkm, "WIDE_BAL_MAX_ROWS", 210)
    engine = mkm.engine_rows(rows)
    assert mkm.plain_schedule(engine, rows, wide_bal=True) is plain
    assert (
        "-DBAL_SPLIT=1" in mkm.engine_options(engine, plain, wide_bal=True)
    ) is not plain


def test_a_numeric_slot_turns_the_claim_into_its_tolerances():
    from sglang.kernels.plugin_kernels.nvidia.spec import NumericsKind

    route = spec.numerics(frozenset({"route", "hop"}))
    assert route.kind is NumericsKind.TOLERANCE
    assert dict(route.tolerances) == spec.NUMERIC_TOLERANCES["route"]
    tiled_route = spec.numerics(frozenset({"route_topk", "route_gemm"}))
    assert tiled_route.kind is NumericsKind.TOLERANCE
    assert dict(tiled_route.tolerances) == spec.NUMERIC_TOLERANCES["route_gemm"]
    claim = spec.numerics(frozenset({"gdn_verify", "hop", "gdn_push"}))
    assert claim.kind is NumericsKind.TOLERANCE
    assert dict(claim.tolerances) == spec.NUMERIC_TOLERANCES["gdn_push"]
    shared = spec.numerics(frozenset({"route", "route_shared"}))
    assert dict(shared.tolerances) == {
        **spec.NUMERIC_TOLERANCES["route"],
        **spec.NUMERIC_TOLERANCES["route_shared"],
    }
    both = spec.numerics(frozenset({"gdn_verify", "gdn_pdl", "gdn_front", "gdn_push"}))
    assert dict(both.tolerances) == {
        **spec.NUMERIC_TOLERANCES["gdn_push"],
        **spec.NUMERIC_TOLERANCES["gdn_front"],
    }
    core = spec.numerics(frozenset({"gdn_verify", "gdn_core"}))
    assert dict(core.tolerances) == spec.NUMERIC_TOLERANCES["gdn_core"]
    attention = spec.numerics(frozenset({"skv2", "mka", "mka_push", "hop"}))
    assert attention.kind is NumericsKind.TOLERANCE
    assert dict(attention.tolerances) == {
        **spec.NUMERIC_TOLERANCES["skv2"],
        **spec.NUMERIC_TOLERANCES["mka"],
    }
    small = spec.numerics(frozenset({"gdn_front", "gdn_trt_front", "route_lin"}))
    assert dict(small.tolerances) == {
        **spec.NUMERIC_TOLERANCES["gdn_front"],
        **spec.NUMERIC_TOLERANCES["gdn_trt_front"],
        **spec.NUMERIC_TOLERANCES["route_lin"],
    }


def test_plugin_runtime_slot_declares_its_actual_numerics():
    from sglang.kernels.plugin_kernels.nvidia.spec import NumericsKind

    from sgl_nvidia_mega.qwen3_5 import draft

    sampled = spec.numerics(frozenset({"sharded_gumbel"}))
    assert sampled.kind is NumericsKind.TOLERANCE
    assert dict(sampled.tolerances) == spec.NUMERIC_TOLERANCES["sharded_gumbel"]

    # The deferred producer changes the draft's finalize/reduction order.  It uses
    # the draft spec's existing final-logit envelope, not a target-model tolerance.
    finalize = draft.numerics(frozenset({"draft_fin"}))
    assert finalize.kind is NumericsKind.TOLERANCE
    assert dict(finalize.tolerances) == {"draft_logits.rel": 2**-7}


def test_the_shipped_policy_claims_bitwise_numerics():
    from sglang.kernels.plugin_kernels.nvidia.spec import NumericsKind

    from sgl_nvidia_mega.qwen3_5 import draft

    shipped = frozenset(runtime.SHIPPED_SLOTS)
    assert spec.numerics(shipped).kind is NumericsKind.BITWISE
    assert draft.numerics(shipped).kind is NumericsKind.BITWISE
