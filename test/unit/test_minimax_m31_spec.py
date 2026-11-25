# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The MiniMax-M3.1 specs through SGLang's registry and startup checks on CPU, as the
plugin registers them: both load under megakernel API v2 and their built objects offer
SGLang no optional entry; the documented setup passes, and every other setup is
refused at argument resolution with each reason named."""

import contextlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia import spec as registry
from sglang.srt.environ import envs
from sglang.srt.kernel_plugins.nvidia import binding as guards
from sglang.srt.model_executor.cuda_graph_config import default_cuda_graph_config
from sglang.srt.server_args import ServerArgs

from sgl_nvidia_mega import plugin
from sgl_nvidia_mega.minimax_m31 import HF_CONFIG, SPARSE_CONFIG, draft, spec
from sgl_nvidia_mega.minimax_m31.sm103 import nvcc

# the documented launch (README.md)
DOCUMENTED = dict(
    model_path="dummy",
    enable_nvidia_megakernel=True,
    tp_size=4,
    kv_cache_dtype="fp8_e4m3",
    page_size=128,
    moe_runner_backend="flashinfer_trtllm_routed",
    moe_a2a_backend="none",
    speculative_algorithm="DSPARK",
    speculative_num_steps=1,
    speculative_eagle_topk=1,
    speculative_num_draft_tokens=8,
    speculative_dspark_block_size=7,
    disable_attn_tp_gather=True,
    cuda_graph_config=default_cuda_graph_config(),
)
DOCUMENTED_ENV = {"SGLANG_FORCE_COARSE_WAR_BARRIER": True}
TEXT = dict(
    HF_CONFIG,
    sparse_attention_config=dict(SPARSE_CONFIG, sparse_attention_freq=[1] * 6),
    moe_layer_freq=[0, 0, 0, 1, 1, 1],
    max_position_embeddings=1 << 20,
)


@pytest.fixture(autouse=True)
def registered(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.delenv(plugin.PARITY_SORT_ENV, raising=False)
    plugin.register()


def _refusal(
    text=None, dtype="bfloat16", isa="sm_103", env=None, toolchain=(), **args
) -> str:
    """SGLang's resolution step on these arguments: "" when it accepts them, else
    its error message."""
    hf_config = SimpleNamespace(
        architectures=["MiniMaxM3SparseForConditionalGeneration"],
        text_config=SimpleNamespace(**{**TEXT, **(text or {})}),
    )
    with (
        contextlib.ExitStack() as stack,
        patch.object(guards, "plugin_reasons", return_value=[]),
        patch.object(guards, "device_isa", return_value=isa),
        patch.object(guards, "model_config_of") as model_config,
        patch.object(spec, "toolchain_reasons", return_value=list(toolchain)),
    ):
        model_config.return_value = SimpleNamespace(hf_config=hf_config, dtype=dtype)
        for name, value in {**DOCUMENTED_ENV, **(env or {})}.items():
            stack.enter_context(getattr(envs, name).override(value))
        try:
            guards.handle_nvidia_megakernel(ServerArgs(**{**DOCUMENTED, **args}))
        except ValueError as e:
            return str(e)
    return ""


# Against an SGLang that predates the MiniMax-M3.1 support the package is built on, the
# spec refuses every configuration (the test below); the other checks of SGLang's
# resolution step then have nothing to accept or refuse
needs_host = pytest.mark.skipif(
    bool(spec.host_reasons(ServerArgs(**DOCUMENTED))), reason=spec.NO_M31_HOST
)


def test_both_specs_load_through_the_registry_and_offer_no_optional_entry():
    """Megakernel API v2: SGLang's registry resolves the target's and the draft's
    spec, and its binding reads the optional entries and graph variants of what they
    build as absent, which keeps SGLang's own draft loop and graph path. (Slot markov
    sets ``draft_sampler`` on the draft's built object at its build.)"""
    from sgl_nvidia_mega.minimax_m31.draft_step import DraftStep
    from sgl_nvidia_mega.minimax_m31.runtime.executor import MegakernelDecode

    assert spec.SPEC.api_version == registry.MEGAKERNEL_API_VERSION == 2
    assert spec.SPEC.max_rows == 128  # 8 verify rows a request, 16 requests
    assert spec.SPEC.requires.server_args["speculative_num_draft_tokens"] == {8}
    for architecture, want, cls in (
        ("MiniMaxM3SparseForConditionalGeneration", spec.SPEC, MegakernelDecode),
        ("DSparkMiniMaxDraftModel", draft.SPEC, DraftStep),
    ):
        found, why = registry.find_spec(architectures=[architecture], isa="sm_103")
        assert found is want, why
        # the draft shares the target's GPUs and contract version
        assert found.api_version == spec.API_VERSION
        for field in ("isas", "tp_sizes", "dtypes", "decode_graph_backends"):
            assert getattr(found.requires, field) == getattr(spec.SPEC.requires, field)
        built = object.__new__(cls)
        model = SimpleNamespace(
            _megakernel=guards._InstalledMegakernel(
                built=built, binding=None, prepare=None
            )
        )
        assert guards._offered_entries(built) == ()
        assert guards._offered_graph_variants(built) == {}
        assert built.speculative_runtime_hooks is None
        for name in registry.ENTRIES:
            assert guards.megakernel_entry(model, name) is None
        for runner in registry.GRAPH_VARIANT_RUNNERS:
            variants = guards.megakernel_graph_variants(
                model, runner, batch_sizes=(1, 2)
            )
            assert variants is None


def test_an_sglang_without_the_m31_support_is_refused_with_that_reason():
    with patch.object(spec, "host_reasons", return_value=[spec.NO_M31_HOST]):
        assert spec.NO_M31_HOST in _refusal()
    assert spec.host_reasons(SimpleNamespace()) == [spec.NO_M31_HOST]
    assert (
        spec.host_reasons(SimpleNamespace(speculative_dspark_lm_head_dtype=None)) == []
    )


@needs_host
def test_the_documented_setup_is_accepted():
    assert _refusal() == ""


@needs_host
@pytest.mark.parametrize(
    "kw, want",
    [
        (dict(tp_size=8), "tp_size 8"),
        (dict(moe_runner_backend="flashinfer_cutlass"), "moe_runner_backend"),
        (dict(speculative_num_draft_tokens=6), "speculative_num_draft_tokens 6"),
        (dict(fp8_gemm_runner_backend="flashinfer_trtllm"), "fp8_gemm_runner_backend"),
        (dict(context_length=1 << 21), "exceeds the kernels' 1048576 tokens"),
        (dict(text=dict(hidden_size=4096)), "config.hidden_size 4096"),
        (dict(text=dict(rope_scaling={"rope_type": "yarn"})), "plain RoPE"),
        (
            dict(env={"SGLANG_FORCE_COARSE_WAR_BARRIER": False}),
            "SGLANG_FORCE_COARSE_WAR_BARRIER=1",
        ),
        (dict(isa="sm_100"), "not sm_100"),
    ],
)
def test_each_refusal_names_its_reason(kw, want):
    assert want in _refusal(**kw)


@needs_host
def test_a_missing_toolchain_is_refused_before_the_weights_load():
    with (
        patch.object(spec, "_has_module", return_value=True),
        patch.object(nvcc, "find_nvcc", return_value=None),
    ):
        reasons = spec.toolchain_reasons()
    assert reasons and "nvcc" in reasons[0]
    assert "nvcc" in _refusal(toolchain=reasons)


def _toolchain(slots, missing=()):
    """``spec.toolchain_reasons`` for ``slots`` on a machine without the ``missing``
    modules, and the nvcc check it made."""
    with (
        patch.object(spec, "_has_module", lambda name: name not in missing),
        patch.object(nvcc, "why_not", return_value="") as why_not,
    ):
        return spec.toolchain_reasons(frozenset(slots)), why_not


@pytest.mark.parametrize(
    "slots, missing, want",
    [
        ({"mega"}, ("cuda.bindings",), ["needs the CUDA Python bindings"]),
        ({"markov"}, ("triton",), ["slot markov needs Triton"]),
        (
            {"moe_nofill"},
            ("flashinfer", "cutlass"),
            [
                "slot moe_nofill needs FlashInfer",
                "slot moe_nofill needs CuTe DSL (nvidia-cutlass-dsl)",
            ],
        ),
        # what a slot does not launch through is not asked of the machine
        ({"mega", "markov"}, ("flashinfer", "cutlass"), []),
        ({"moe_nofill", "moe_swap"}, ("cuda.bindings", "triton"), []),
    ],
)
def test_each_slot_names_the_toolchain_it_misses(slots, missing, want):
    assert _toolchain(slots, missing)[0] == want


def test_only_slot_mega_asks_for_the_nvcc_floor():
    """Slot markov builds with any nvcc; the QMUL4 floor is slot
    mega's; the MoE slots build nothing."""
    for slots, floor in (
        ({"mega", "markov"}, nvcc.QMUL4_VERSION),
        ({"markov"}, None),
    ):
        _toolchain(slots)[1].assert_called_once_with(floor)
    _toolchain({"moe_nofill", "moe_swap"})[1].assert_not_called()


@needs_host
def test_a_default_rope_type_and_a_serial_scheduler_are_accepted():
    assert _refusal(text=dict(rope_scaling={"rope_type": "default"})) == ""
    off = {"SGLANG_FORCE_COARSE_WAR_BARRIER": False}
    assert _refusal(env=off, disable_overlap_schedule=True) == ""


def test_the_attention_scales_are_tiled_128x4():
    """MegaAttention's scale layout: [R / 128][C / 4][32][4][4], element
    (rb, c4, r32, r4, c) holding row rb * 128 + r4 * 32 + r32, column c4 * 4 + c."""
    from sgl_nvidia_mega.minimax_m31.weights import tile_scales

    scale = torch.randint(0, 256, (256, 8), dtype=torch.uint8)
    got = tile_scales(scale).view(2, 2, 32, 4, 4)
    for rb, c4, r32, r4, c in torch.cartesian_prod(
        *map(torch.arange, got.shape)
    ).tolist():
        assert got[rb, c4, r32, r4, c] == scale[rb * 128 + r4 * 32 + r32, c4 * 4 + c]


@needs_host
@pytest.mark.parametrize(
    "slots, want",
    [
        ({"hop_mxfp8"}, "unknown slot 'hop_mxfp8'"),
        ({"moe_swap"}, "slot 'moe_swap' needs slot 'moe_nofill'"),
    ],
)
def test_a_slot_list_refusal_names_it(slots, want):
    with patch.object(spec, "SLOTS", frozenset(slots)):
        assert want in _refusal()


@needs_host
def test_the_draft_slots_checks_apply_only_with_them():
    """Slot markov ships: the documented setup runs it. The draft's slots replace
    SGLang's folded DSpark proposal, so a server without it is refused while one is
    requested; it runs SGLang's own quantized draft LM head too."""
    assert "markov" in spec.SLOTS
    unfolded = {"SGLANG_DSPARK_FOLDED_PROPOSAL": False}
    assert "SGLANG_DSPARK_FOLDED_PROPOSAL is off" in _refusal(env=unfolded)
    int8_walk = {"SGLANG_DSPARK_OPT_INT8_MARKOV_WALK": True}
    assert "asks for SGLang's int8 walk" in _refusal(env=int8_walk)
    assert _refusal(speculative_dspark_lm_head_dtype="mxfp8") == ""
    with patch.object(spec, "SLOTS", spec.SLOTS - {"markov"}):
        assert _refusal(env={**unfolded, **int8_walk}) == ""


@needs_host
def test_slot_megas_checks_apply_only_with_it():
    """MegaAttention's checks (the config it is built for, the persistent grid's
    residency) refuse a server only while slot mega is requested."""
    off = {"SGLANG_FORCE_COARSE_WAR_BARRIER": False}
    assert "SGLANG_FORCE_COARSE_WAR_BARRIER=1" in _refusal(env=off)
    assert "config.hidden_size 4096" in _refusal(text=dict(hidden_size=4096))
    with patch.object(spec, "SLOTS", frozenset({"moe_nofill", "moe_swap"})):
        assert _refusal(env=off) == ""
        assert _refusal(text=dict(hidden_size=4096)) == ""
        # the checks of every slot list still apply
        assert "no MoE layer" in _refusal(text=dict(moe_layer_freq=[0] * 6))
    with patch.object(spec, "SLOTS", frozenset()):
        assert _refusal(env=off) == ""
