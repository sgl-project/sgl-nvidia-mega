# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The GLM-5 DSA megakernel's spec: the MTP target-verify steps of 1-4 requests
(4 / 8 / 12 / 16 rows) of ``GlmMoeDsaForCausalLM`` on SM103 / SM107 at TP8.

SGLang checks the requirements at argument resolution and the model at load
(``sglang.srt.megakernels``); an unsupported setup stops the server on every rank,
listing every reason. This module stays light: SGLang imports it in the launcher.
"""

from __future__ import annotations

import importlib.util

from sglang.srt.megakernels.spec import (
    Feature,
    MegakernelSpec,
    Numerics,
    NumericsKind,
    Requirements,
)

from sgl_nvidia_mega.glm_dsa import switches

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/megakernels/spec.py": (
        "Feature",
        "MULTI_NODE_TP",
        "SPECULATIVE_DECODING",
        "MegakernelSpec",
        "Numerics",
        "NumericsKind",
        "TOLERANCE",
        "Requirements",
        "config_disable_reasons",
        "model_disable_reasons",
    ),
    "srt/models/glm4_moe.py": ("GlmMoeDsaForCausalLM", "GlmMoeDsaForCausalLMNextN"),
    "srt/arg_groups/fields/*.py": (
        "kv_cache_dtype",
        "page_size",
        "attention_backend",
        "dsa_decode_backend",
        "moe_runner_backend",
        "speculative_algorithm",
        "speculative_eagle_topk",
        "speculative_num_draft_tokens",
        "speculative_num_steps",
        "cuda_graph_config",
    ),
}

ARCHITECTURE = "GlmMoeDsaForCausalLM"
# the sglang.srt.megakernels contract this spec is written against
API_VERSION = 1
# verify rows: 1-4 requests x 4 tokens (3 draft tokens + 1)
MAX_ROWS = 16

REQUIREMENTS = Requirements(
    isas=frozenset({"sm_103", "sm_107"}),
    tp_sizes=frozenset({8}),
    dtypes=frozenset({"bfloat16"}),
    # the fused step is recorded only into captured FULL decode graphs
    decode_graph_backends=frozenset({"full"}),
    server_args={
        "kv_cache_dtype": frozenset({"fp8_e4m3"}),
        "page_size": frozenset({64}),
        "attention_backend": frozenset({"dsa"}),
        "dsa_decode_backend": frozenset({"trtllm"}),
        "moe_runner_backend": frozenset({"flashinfer_trtllm"}),
        # NEXTN resolves to EAGLE; the MTP chain of 3 draft tokens
        "speculative_algorithm": frozenset({"EAGLE"}),
        "speculative_eagle_topk": frozenset({1}),
        "speculative_num_draft_tokens": frozenset({4}),
        "speculative_num_steps": frozenset({3}),
    },
    hf_config={"model_type": "glm_moe_dsa"},
    # the fused draft's embedding front reads the replicated table
    env={"SGLANG_ENABLE_EMBED_REPLICATION": True} if switches.fused_draft() else {},
)

# The validated claim: the step's layer loop with every step on its stock body is
# bitwise SGLang's forward (the in-graph replica check). The fused kernels' own
# per-op tolerances are not stated yet.
NUMERICS = Numerics(
    kind=NumericsKind.TOLERANCE,
    tolerances={"loop.stock_bodies.max_abs": 0.0},
)


def toolchain_reasons() -> list[str]:
    reasons = [
        f"needs {what}"
        for module, what in (
            ("cutlass", "CuTe DSL (nvidia-cutlass-dsl)"),
            ("cuda.bindings", "the CUDA Python bindings"),
        )
        if importlib.util.find_spec(module) is None
    ]
    if not reasons:
        from sgl_nvidia_mega.glm_dsa.kernels.cute import cutlass_supported

        ok, why = cutlass_supported()
        if not ok:
            reasons.append(why)
    return reasons


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    from sgl_nvidia_mega.glm_dsa import eligibility

    return toolchain_reasons() + eligibility.construction_reasons(hf_config, cfg)


def _model_disable_reasons(model, hf_config) -> list[str]:
    from sgl_nvidia_mega.glm_dsa import eligibility

    try:
        eligibility.check_construction(model)
        eligibility.check_loaded(model)
    except ValueError as e:
        return [str(e)]
    return []


class _Built:
    """The installed target forward: covered verify steps run the fused stack."""

    draft_sampler = None

    def __init__(self, model, fused):
        self.model, self.fused = model, fused
        self.widths = tuple(sorted(fused.providers))

    def summary(self) -> str:
        fused = {
            rows: sorted(
                k
                for k, v in p.describe().items()
                if not v.startswith(("AttentionDecode", "HopSteps", "MlpSteps"))
            )
            for rows, p in self.fused.providers.items()
        }
        return "GLM-5 DSA fused verify; fused steps by rows: " + "; ".join(
            f"{rows}: {', '.join(names) or 'stock'}" for rows, names in fused.items()
        )

    def plan(self, input_ids, forward_batch, *args, **kwargs):
        from sglang.srt.megakernels.spec import Step

        input_embeds = kwargs.get("input_embeds", args[0] if args else None)
        pp_proxy = kwargs.get("pp_proxy_tensors", args[1] if len(args) > 1 else None)
        rows = int(input_ids.shape[0])
        if not self.fused.engages(rows, forward_batch, input_embeds, pp_proxy):
            return None
        return Step(mode="target_verify", rows=rows)

    def forward(self, step, original, input_ids, positions, forward_batch, *a, **kw):
        hidden_states = self.fused.run(input_ids, positions, forward_batch)
        return self.model.logits_processor(
            input_ids, hidden_states, self.model.lm_head, forward_batch, None
        )


def _build(model, hf_config, ctx):
    from sgl_nvidia_mega.glm_dsa.runtime import GlmDsaFusedDecode

    fused = GlmDsaFusedDecode(model)
    fused.prepare()
    return _Built(model, fused)


SPEC = MegakernelSpec(
    name="GLM-5 DSA",
    requires=REQUIREMENTS,
    features=frozenset({Feature.SPECULATIVE_DECODING, Feature.MULTI_NODE_TP}),
    numerics=NUMERICS,
    max_rows=MAX_ROWS,
    build=_build,
    api_version=API_VERSION,
    model_disable_reasons=_model_disable_reasons,
    config_disable_reasons=_config_disable_reasons,
)
