# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GLM-5.2 on the GLM-5.3 specs: the published config of nvidia/GLM-5.2-NVFP4 has
the architecture and every config value the target and draft specs pin, so SGLang's
requirement check accepts it; a config that differs in one pinned value, or lacks
it, is still refused, naming it."""

import contextlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from sglang.srt.environ import envs

try:
    from sglang.srt.kernel_plugins.nvidia import binding as sglang_runtime
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.megakernels import runtime as sglang_runtime

from sgl_nvidia_mega import plugin
from sgl_nvidia_mega.glm5_3 import TP_SIZE, draft, spec

# The top-level fields of config.json in nvidia/GLM-5.2-NVFP4 at revision
# 53e0691e21895a3863a606dfd12910c69eba94ab (its git blob
# b63d4c44cdd013e928cbc37a755633e8c001626f), unchanged. Left out: the per-layer lists
# (indexer_types, layer_types, mlp_layer_types) and quantization_config, which no
# spec pins.
GLM5_2_NVFP4 = {
    "architectures": ["GlmMoeDsaForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 0,
    "dtype": "bfloat16",
    "eos_token_id": [154820, 154827, 154829],
    "ep_size": 1,
    "first_k_dense_replace": 3,
    "head_dim": 192,
    "hidden_act": "silu",
    "hidden_size": 6144,
    "index_head_dim": 128,
    "index_n_heads": 32,
    "index_share_for_mtp_iteration": True,
    "index_skip_topk_offset": 3,
    "index_topk": 2048,
    "index_topk_freq": 4,
    "index_topk_pattern": None,
    "indexer_rope_interleave": True,
    "initializer_range": 0.02,
    "intermediate_size": 12288,
    "kv_lora_rank": 512,
    "max_position_embeddings": 1048576,
    "mlp_bias": False,
    "model_type": "glm_moe_dsa",
    "moe_intermediate_size": 2048,
    "moe_layer_freq": 1,
    "n_group": 1,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "norm_topk_prob": True,
    "num_attention_heads": 64,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 78,
    "num_key_value_heads": 64,
    "num_nextn_predict_layers": 1,
    "pad_token_id": 154820,
    "pretraining_tp": 1,
    "q_lora_rank": 2048,
    "qk_head_dim": 256,
    "qk_nope_head_dim": 192,
    "qk_rope_head_dim": 64,
    "rms_norm_eps": 1e-05,
    "rope_interleave": True,
    "rope_parameters": {"rope_theta": 8000000, "rope_type": "default"},
    "routed_scaling_factor": 2.5,
    "scoring_func": "sigmoid",
    "tie_word_embeddings": False,
    "topk_group": 1,
    "topk_method": "noaux_tc",
    "transformers_version": "5.11.0",
    "use_cache": True,
    "v_head_dim": 256,
    "vocab_size": 154880,
}

SPECS = {"target": spec.SPEC, "draft": draft.SPEC}
# the fields each spec pins, so that a pin dropped from a spec fails here
DRAFT_PINS = {
    "hidden_size",
    "num_attention_heads",
    "q_lora_rank",
    "kv_lora_rank",
    "qk_rope_head_dim",
    "qk_nope_head_dim",
    "v_head_dim",
    "index_n_heads",
    "index_head_dim",
    "index_topk",
    "model_type",
}
TARGET_PINS = DRAFT_PINS | {
    "n_routed_experts",
    "num_experts_per_tok",
    "n_shared_experts",
    "n_group",
    "topk_group",
    "norm_topk_prob",
    "scoring_func",
    "topk_method",
    "moe_intermediate_size",
    "first_k_dense_replace",
    "num_hidden_layers",
}
PINS = [(which, key) for which, s in SPECS.items() for key in s.requires.hf_config]


def _reasons(megakernel_spec, config: dict) -> list[str]:
    """SGLang's requirement check of the spec at load on this config, read as SGLang
    reads it (an attribute per field, with no defaults filled in), on 8 ranks with
    full decode graphs and the environment the spec requires."""
    graphs = SimpleNamespace(decode=SimpleNamespace(backend="full"))
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch.object(
                sglang_runtime,
                "get_parallel",
                return_value=SimpleNamespace(tp_size=TP_SIZE),
            )
        )
        stack.enter_context(
            patch.object(
                sglang_runtime,
                "get_exec",
                return_value=SimpleNamespace(
                    graph=SimpleNamespace(cuda_graph_config=graphs)
                ),
            )
        )
        for name, value in megakernel_spec.requires.env.items():
            stack.enter_context(getattr(envs, name).override(value))
        return sglang_runtime._requirement_reasons(
            spec=megakernel_spec, hf_config=SimpleNamespace(**config)
        )


def _changed(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    return f"{value}-changed"


def test_the_published_config_names_the_glm5_3_architecture():
    rows = dict(plugin.SPECS)
    assert GLM5_2_NVFP4["architectures"] == [spec.ARCHITECTURE]
    assert rows[spec.ARCHITECTURE] == "sgl_nvidia_mega.glm5_3.spec:SPEC"
    # SGLang loads the NextN draft from the same config, renaming the architecture
    assert rows[draft.ARCHITECTURE] == "sgl_nvidia_mega.glm5_3.draft:SPEC"
    assert GLM5_2_NVFP4["dtype"] in spec.SPEC.requires.dtypes


def test_the_specs_pin_these_fields():
    assert set(spec.SPEC.requires.hf_config) == TARGET_PINS
    assert set(draft.SPEC.requires.hf_config) == DRAFT_PINS


@pytest.mark.parametrize("which", SPECS)
def test_the_published_config_passes_every_pin(which):
    assert _reasons(SPECS[which], GLM5_2_NVFP4) == []


@pytest.mark.parametrize("which, key", PINS)
def test_a_changed_pin_is_refused(which, key):
    want = SPECS[which].requires.hf_config[key]
    reasons = _reasons(SPECS[which], {**GLM5_2_NVFP4, key: _changed(want)})
    assert len(reasons) == 1 and reasons[0].startswith(f"config.{key} ")


@pytest.mark.parametrize("which", SPECS)
def test_a_config_without_a_pinned_field_is_refused(which):
    config = {k: v for k, v in GLM5_2_NVFP4.items() if k != "index_topk"}
    reasons = _reasons(SPECS[which], config)
    assert len(reasons) == 1 and reasons[0].startswith("config.index_topk None ")
