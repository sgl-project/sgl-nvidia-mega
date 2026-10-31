# SPDX-License-Identifier: Apache-2.0
"""GLM5's SGLang model contract and native-weight target-verify executor.

Producer indexers and uncovered steps keep SGLang's implementation. Validation
records distinguish operator composition from task accuracy and serving speed.
"""

try:
    from sglang.kernels.plugin_kernels.nvidia.spec import (
        Feature,
        MegakernelSpec,
        Numerics,
        NumericsKind,
        Requirements,
    )
except ModuleNotFoundError:
    from sglang.srt.megakernels.spec import (
        Feature,
        MegakernelSpec,
        Numerics,
        NumericsKind,
        Requirements,
    )

SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": (
        "Feature",
        "MegakernelSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
        "SPECULATIVE_DECODING",
        "TOLERANCE",
    ),
    "srt/distributed/parallel_state.py": ("get_attn_tp_group",),
}

API_VERSION = 1
HF_CONFIG = {
    "hidden_size": 6144,
    "num_attention_heads": 64,
    "q_lora_rank": 2048,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 192,
    "qk_rope_head_dim": 64,
    "v_head_dim": 256,
    "n_routed_experts": 256,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 2048,
    "rms_norm_eps": 1e-5,
    "routed_scaling_factor": 2.5,
    "n_shared_experts": 1,
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "scoring_func": "sigmoid",
    "hidden_act": "silu",
    "rope_interleave": True,
    "index_topk": 2048,
}


def _nondefault_rope(hf_config) -> bool:
    # Transformers now stores default RoPE in rope_parameters and exposes it
    # through the legacy rope_scaling alias as a nonempty dictionary.
    params = getattr(hf_config, "rope_parameters", None)
    if params is None:
        params = getattr(hf_config, "rope_scaling", None)
    return params is not None and (
        not isinstance(params, dict)
        or params.get("rope_type", params.get("type", "default")) != "default"
    )


def _config_disable_reasons(cfg, hf_config):
    reasons = []
    tokens = cfg.speculative_num_draft_tokens
    if tokens not in (4, 8) or cfg.speculative_num_steps != tokens - 1:
        reasons.append(
            "GLM5 needs 4/8 draft tokens and draft_tokens-1 speculative steps"
        )
    if cfg.enable_dsa_cache_layer_split:
        reasons.append("GLM5 does not yet support DSA layer-split cache ownership")
    if getattr(hf_config, "router_fp32", False):
        reasons.append("GLM5 needs the native BF16 router")
    if _nondefault_rope(hf_config):
        reasons.append("GLM5 currently requires the default RoPE attention scale")
    return reasons


def _model_disable_reasons(model, hf_config):
    from sglang.srt.distributed import get_attn_tp_group

    from sgl_nvidia_mega.glm_5.plan import Variant
    from sgl_nvidia_mega.glm_5.runtime.weights import (
        mla_weights_from_layer,
        moe_weights_from_layer,
        norm_dtype,
    )

    reasons, norms = [], {"mla": set(), "moe": set()}
    tp = get_attn_tp_group().world_size
    layers = [layer for layer in model.model.layers if layer.is_layer_sparse]
    if not layers or not any(layer.self_attn.skip_topk for layer in layers):
        return ["GLM5 needs sparse layers with native producer/shared indexers"]
    for layer in layers:
        try:
            weights = {"moe": moe_weights_from_layer(layer)}
            if layer.self_attn.skip_topk:
                weights["mla"] = mla_weights_from_layer(layer)
                if layer.self_attn.scaling != 1 / 16:
                    raise ValueError("GLM5 MLA needs attention scale 1/16")
                if getattr(layer.self_attn.attn_mqa, "k_scale_float", None) not in (
                    None,
                    1.0,
                ):
                    raise ValueError("GLM5 MLA needs raw E4M3 cache with unit scale")
            for op, value in weights.items():
                dtype = norm_dtype(value)
                norms[op].add(dtype)
                device = value.kernel_args()[0].device
                for rows in (4, 8):
                    value.validate(Variant(op, rows, tp), device, norm_dtype=dtype)
        except (AttributeError, TypeError, ValueError) as error:
            reasons.append(f"GLM5 layer {layer.layer_id}: {error}")
    for op, dtypes in norms.items():
        if len(dtypes) > 1:
            reasons.append(f"GLM5 {op} layers have mixed norm dtypes: {sorted(dtypes)}")
    return reasons


def _build(model, hf_config, context):
    from sgl_nvidia_mega.glm_5.runtime.executor import build

    return build(model, hf_config, context)


REQUIREMENTS = Requirements(
    isas=frozenset({"sm_103"}),
    tp_sizes=frozenset({4, 8}),
    resident_blocks=148,
    dtypes=frozenset({"bfloat16"}),
    server_args={
        "attention_backend": frozenset({"dsa"}),
        "dsa_decode_backend": frozenset({"trtllm"}),
        "kv_cache_dtype": frozenset({"fp8_e4m3"}),
        "speculative_algorithm": frozenset({"EAGLE"}),
        "speculative_eagle_topk": frozenset({1}),
        "moe_runner_backend": frozenset({"triton"}),
        "startup_weight_load_mode": frozenset({"serial"}),
        "enable_layernorm_sp": frozenset({False}),
        "enable_attn_tp_input_scattered": frozenset({False}),
    },
    hf_config=HF_CONFIG,
)

# Short target-verify compositions: TP8, rows 4/8, CUDA 13.0.
# Observed worst relative L2: logits 0.14988, auxiliary hidden 0.15686.
# These calibration bounds do not establish GSM8K accuracy; task results and
# wider configurations are qualified separately by the serving validation run.
NUMERICS = Numerics(
    kind=NumericsKind.TOLERANCE,
    tolerances={
        "tp8.short_verify.logits.relative_l2": 0.16,
        "tp8.short_verify.hidden.relative_l2": 0.16,
    },
)

SPEC = MegakernelSpec(
    name="GLM5",
    requires=REQUIREMENTS,
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    numerics=NUMERICS,
    max_rows=8,
    build=_build,
    api_version=API_VERSION,
    model_disable_reasons=_model_disable_reasons,
    config_disable_reasons=_config_disable_reasons,
)
