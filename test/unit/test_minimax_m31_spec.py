# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The MiniMax-M3.1 spec through SGLang's startup checks on CPU, as the plugin
registers it: the documented setup passes, and every other setup is refused at
argument resolution with each reason named."""

import contextlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from sglang.srt.environ import envs

try:
    from sglang.kernels.plugin_kernels.nvidia import spec as registry
    from sglang.srt.kernel_plugins.nvidia import binding as guards
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.megakernels import guards
    from sglang.srt.megakernels import spec as registry

from sglang.srt.model_executor.cuda_graph_config import default_cuda_graph_config
from sglang.srt.server_args import ServerArgs

from sgl_nvidia_mega import plugin
from sgl_nvidia_mega.minimax_m31 import HF_CONFIG, SPARSE_CONFIG, draft, spec

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


def test_the_documented_setup_is_accepted():
    assert _refusal() == ""


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


def test_a_missing_toolchain_is_refused_before_the_weights_load():
    from sgl_nvidia_mega.minimax_m31.sm103 import nvcc

    with patch.object(nvcc, "find_nvcc", return_value=None):
        reasons = spec.toolchain_reasons()
    assert reasons and "nvcc" in reasons[0]
    assert "nvcc" in _refusal(toolchain=reasons)


def test_a_default_rope_type_and_a_serial_scheduler_are_accepted():
    assert _refusal(text=dict(rope_scaling={"rope_type": "default"})) == ""
    off = {"SGLANG_FORCE_COARSE_WAR_BARRIER": False}
    assert _refusal(env=off, disable_overlap_schedule=True) == ""


def test_the_draft_spec_follows_the_target_spec():
    """The draft shares the target's GPUs and contract version."""
    assert draft.SPEC.api_version == spec.API_VERSION
    for field in ("isas", "tp_sizes", "dtypes", "decode_graph_backends"):
        assert getattr(draft.SPEC.requires, field) == getattr(spec.SPEC.requires, field)


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
