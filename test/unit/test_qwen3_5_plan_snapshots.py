# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest

from sgl_nvidia_mega.qwen3_5 import runtime, topology

# SGLang's switch of the megakernel runtime optimizations
RUNTIME_ENV = "SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION"
_B300 = topology.Sku(10, 3, 148)


def _names(text: str) -> frozenset:
    return frozenset(text.split())


_RECIPE = _names(
    """
    route route_topk route_gemm route_shared route_lin mkm mkm_wide mkm_wide_bal
    mkm_m32 mkm_m64 mkm_m128 mkm_m256 mkm_m512 gdn_verify gdn_pdl gdn_conv gdn_core
    gdn_push gdn_front gdn_trt_front hop hop_mid hop_fp8 skv2 skv2_cp mka mka_push
    lowm_in lowm_out lowm_se lowm_push multi_request sharded_gumbel draft_loop
    draft_extend draft_multi draft_embed draft_moe draft_q1 draft_xchg draft_hop
    draft_fin draft_front draft_skv2 draft_q1_cp draft_skv2_cp
    """
)
# (tp, attention tp, attention dp, MoE ep, pp) per key
_SHAPES = {
    "tp8": (8, 8, 1, 1, 1),
    "tep8": (8, 8, 1, 4, 1),
    "tp4": (4, 4, 1, 1, 1),
    "tp2": (2, 2, 1, 1, 1),
    "dep8": (8, 1, 8, 8, 1),
    "dep4": (4, 1, 4, 4, 1),
    "p_tp1_pp4": (1, 1, 1, 1, 4),
    "p_tp1_pp2": (1, 1, 1, 1, 2),
}
_UPSTREAM = {
    ("tp8", False): _names(
        "draft_embed draft_extend draft_loop draft_multi hop hop_fp8 mkm_m128 mkm_m256"
        " mkm_m32 mkm_m512 mkm_m64 mkm_wide_bal multi_request route_topk"
    ),
    ("tep8", False): _names(
        "draft_embed draft_extend draft_loop draft_multi hop hop_fp8 multi_request"
        " route_topk"
    ),
    ("tp4", False): _names(
        "draft_embed draft_extend draft_loop draft_multi hop hop_fp8 multi_request"
        " route_topk"
    ),
    ("tp2", False): _names("multi_request route_topk"),
    ("dep8", False): _names("multi_request"),
    ("p_tp1_pp4", False): frozenset(),
    ("tp8", True): _names(
        "draft_embed draft_extend draft_fin draft_front draft_hop draft_loop draft_moe"
        " draft_multi draft_q1 draft_q1_cp draft_skv2 draft_skv2_cp draft_xchg gdn_conv"
        " gdn_core gdn_front gdn_pdl gdn_push gdn_trt_front gdn_verify hop hop_fp8"
        " hop_mid lowm_in lowm_out lowm_push lowm_se mka mka_push mkm mkm_m128 mkm_m256"
        " mkm_m32 mkm_m512 mkm_m64 mkm_wide mkm_wide_bal multi_request route route_gemm"
        " route_lin route_shared route_topk sharded_gumbel skv2 skv2_cp"
    ),
    ("tep8", True): _names(
        "draft_embed draft_extend draft_fin draft_front draft_hop draft_loop"
        " draft_multi draft_q1 draft_q1_cp draft_skv2 draft_skv2_cp draft_xchg gdn_conv"
        " gdn_core gdn_front gdn_pdl gdn_push gdn_trt_front gdn_verify hop hop_fp8"
        " hop_mid lowm_in lowm_out lowm_push lowm_se mka mka_push multi_request route"
        " route_gemm route_lin route_shared route_topk sharded_gumbel skv2 skv2_cp"
    ),
    ("tp4", True): _names(
        "draft_embed draft_extend draft_fin draft_front draft_hop draft_loop draft_moe"
        " draft_multi draft_q1 draft_q1_cp draft_skv2 draft_skv2_cp draft_xchg gdn_conv"
        " gdn_core gdn_pdl gdn_push gdn_verify hop hop_fp8 hop_mid lowm_in lowm_out"
        " lowm_push lowm_se mka mka_push mkm multi_request route route_gemm route_shared"
        " route_topk sharded_gumbel skv2 skv2_cp"
    ),
    ("tp2", True): _names(
        "gdn_conv gdn_core gdn_pdl gdn_verify mka multi_request route route_gemm"
        " route_topk sharded_gumbel"
    ),
    ("dep8", True): _names(
        "gdn_conv gdn_pdl gdn_verify mka multi_request sharded_gumbel"
    ),
    ("p_tp1_pp4", True): frozenset(),
}
_MEASURED = {
    "tp8": _names(
        "bt_tail bt_tail_early bt_tail_early_fin cp_lam_q cp_merge_cg"
        " cp_merge_plain_wait cp_o_push cp_ptxas_o1 cp_q_gather cp_q_push cp_qw_entry"
        " cp_smv2 draft_cp draft_embed draft_embed_local draft_extend draft_fc_input"
        " draft_fin draft_front draft_gate_fold draft_head_lowm draft_hop"
        " draft_loop draft_moe draft_multi draft_o_push draft_q1 draft_q1_cp"
        " draft_q_gather draft_q_push"
        " draft_router_lowm draft_select"
        " draft_skv2 draft_skv2_cp gdn_conv gdn_conv_core gdn_core gdn_front gdn_pdl"
        " gdn_push gdn_trt_front gdn_verify gumbel_fast hop hop_fp8"
        " hop_mid lowm_ba lowm_in lowm_out lowm_push lowm_se mka mka_push mkm mkm_m128"
        " mkm_m256 mkm_m32 mkm_m512 mkm_m64 mkm_wide mkm_wide_bal multi_request route"
        " route_lin route_shared route_topk sharded_gumbel skv2 skv2_cp verify_head"
        " verify_head_keys"
    ),
    "tep8": _names(
        "bt_tail bt_tail_early bt_tail_early_fin cp_lam_q cp_merge_cg"
        " cp_merge_plain_wait cp_o_push cp_ptxas_o1 cp_q_gather cp_q_push cp_qw_entry"
        " cp_smv2 draft_cp draft_embed draft_embed_local draft_extend draft_fc_input"
        " draft_fc_tp draft_fin draft_front draft_gate_fold"
        " draft_head_lowm draft_hop draft_loop draft_multi draft_o_push draft_q1"
        " draft_q1_cp draft_q_gather draft_q_push"
        " draft_router_lowm draft_select"
        " draft_skv2 draft_skv2_cp fc2ar gdn_conv gdn_conv_core gdn_core gdn_front"
        " gdn_pdl gdn_push gdn_trt_front gdn_verify gemm_ar gumbel_fast"
        " hop hop_fp8 hop_mid lowm_ba lowm_in lowm_out lowm_push lowm_se"
        " lowm_se_coresident lowm_se_tep8 lowm_se_wide mka mka_push moe_engine"
        " moe_engine_route moe_ring multi_request route route_big_gemm route_big_mid"
        " route_lean route_shared route_topk sharded_gumbel skv2 skv2_cp verify_head"
        " verify_head_keys"
    ),
    "dep4": _names("gathered_cdf gdn_commit_fold gdn_core gdn_core_tc gdn_verify"),
    "p_tp1_pp2": _names(
        "attn_prefill_qk_rows gdn_prefill_front gdn_prefill_norm_fp8"
        " gdn_prefill_track_snapshot moe_prefill_add_norm moe_prefill_input_nvfp4"
        " moe_prefill_noclone prefill_act_fp8 prefill_gate_fp8 prefill_norm_fp8"
    ),
}
_OURS = {
    "gumbel_fast",
    "verify_head",
    "verify_head_keys",
    "draft_cp",
    "draft_gate_fold",
    "draft_o_push",
}


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    from sglang.srt.kernel_plugins.nvidia import binding as plugin_binding

    monkeypatch.setattr(
        plugin_binding, "megakernel_agree", lambda *args, **kwargs: None
    )
    monkeypatch.delenv(RUNTIME_ENV, raising=False)
    monkeypatch.delenv(runtime.SLOTS_ENV, raising=False)
    monkeypatch.setattr(runtime, "MEASURED_KEY", None)
    monkeypatch.setattr(topology, "device_sku", _no_gpu)


def _no_gpu():
    raise AssertionError("read the GPU")


def _resolve(monkeypatch, key, sku=topology.GB300):
    tp, attn_tp, dp, ep, pp = _SHAPES[key]
    parallel = SimpleNamespace(
        tp_size=tp,
        attn_tp_size=attn_tp,
        attn_dp_size=dp,
        moe_ep_size=ep,
        pp_size=pp,
        tp_group=SimpleNamespace(world_size=tp),
    )
    monkeypatch.setattr(runtime, "SLOTS", runtime._requested_slots())
    runtime.apply_measured_plan(parallel, sku)
    if pp > 1:
        return runtime.SLOTS & set(runtime.declared_prefill_slots(parallel))
    runtime.apply_expert_parallel(ep)
    runtime.apply_cp_lite_topology(attn_tp)
    runtime.apply_topology(parallel)
    return runtime.SLOTS


def _options_on() -> set:
    return {name for name in runtime.MEASURED_OPTIONS if runtime.measured_option(name)}


@pytest.mark.parametrize(
    "switch, slot_list",
    [(None, False), ("0", False), (None, True), ("1", True)],
    ids=["mk", "switch-off", "slot-list", "switch-and-slot-list"],
)
@pytest.mark.parametrize("key", ["tp8", "tep8", "tp4", "tp2", "dep8", "p_tp1_pp4"])
def test_without_the_measured_stack_each_key_resolves_as_upstream(
    monkeypatch, switch, slot_list, key
):
    if switch is not None:
        monkeypatch.setenv(RUNTIME_ENV, switch)
    if slot_list:
        monkeypatch.setenv(runtime.SLOTS_ENV, ",".join(sorted(_RECIPE)))
    assert _resolve(monkeypatch, key) == _UPSTREAM[key, slot_list]
    assert runtime.MEASURED_KEY is None
    # no GEMM tactic pin, value split or recurrent-commit provider
    assert _options_on() == set()
    assert not runtime.SLOTS & _OURS
    assert not runtime._cp_lite_constants_on()


@pytest.mark.parametrize("key", ["tp8", "tep8", "dep4", "p_tp1_pp2"])
def test_the_measured_stack_resolves_each_key_to_its_plan(monkeypatch, key):
    monkeypatch.setenv(RUNTIME_ENV, "1")
    assert _resolve(monkeypatch, key) == _MEASURED[key]
    assert runtime.MEASURED_KEY == (key, topology.GB300)
    assert not runtime.SLOTS & {"draft_gather_early", "draft_xchg"}
    decode = key in ("tp8", "tep8")
    assert _options_on() == (set(runtime.MEASURED_OPTIONS) if decode else set())
    assert runtime._cp_lite_constants_on() is decode
    assert (runtime.SLOTS >= _OURS) is decode


def test_the_measured_plans_differ_only_by_teps_own_rows():
    # TEP8's routed body, the all-reduces on the BT tail's points and the draft's fc
    assert _MEASURED["tep8"] - _MEASURED["tp8"] == set(runtime.MEASURED_TEP8) | set(
        runtime.MEASURED_BT_AR + runtime.MEASURED_DRAFT_TEP8
    )
    assert _MEASURED["tp8"] - _MEASURED["tep8"] == runtime.EP_STOCK_SLOTS | {
        "route_lin"
    }


def test_the_measured_stack_on_another_gpu_runs_the_shipped_slots(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV, "1")
    assert _resolve(monkeypatch, "tep8", _B300) == _UPSTREAM["tep8", False]
    assert runtime.MEASURED_KEY == ("tep8", _B300)
    assert _options_on() == set()
    assert not runtime._cp_lite_constants_on()
