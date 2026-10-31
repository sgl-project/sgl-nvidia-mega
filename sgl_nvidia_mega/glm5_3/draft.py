# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The GLM-5.3 NextN draft on SM103 (B300): the megakernel draft spec.

With slot "draft" the draft's forward runs on the plugin's draft step
(``draft_step.FusedDraft``) inside SGLang's draft-decode and draft-extend CUDA graphs;
otherwise it stays SGLang's. SGLang checks a draft spec at load only (its ISA, TP
size, decode graphs, config and environment, and ``model_disable_reasons``); the
target spec checks the draft's server arguments at argument resolution.
"""

from __future__ import annotations

try:
    from sglang.kernels.plugin_kernels.nvidia.spec import (
        Feature,
        MegakernelSpec,
        Numerics,
        NumericsKind,
        Requirements,
    )
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.megakernels.spec import (
        Feature,
        MegakernelSpec,
        Numerics,
        NumericsKind,
        Requirements,
    )

from sgl_nvidia_mega.glm5_3 import HF_CONFIG, MODEL_TYPE
from sgl_nvidia_mega.glm5_3 import spec as target
from sgl_nvidia_mega.glm5_3.plan import MAX_ROWS
from sgl_nvidia_mega.glm5_3.runtime import SLOTS

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": (
        "Feature",
        "SPECULATIVE_DECODING",
        "MegakernelSpec",
        "Numerics",
        "NumericsKind",
        "TOLERANCE",
        "Requirements",
        "model_disable_reasons",
    ),
    "srt/runtime_context.py": ("get_parallel", "get_spec"),
    "srt/arg_groups/fields/*.py": (
        "pp_size",
        "enable_dp_attention",
        "enable_dp_lm_head",
        "speculative_eagle_topk",
        "speculative_use_rejection_sampling",
    ),
}

ARCHITECTURE = "GlmMoeDsaForCausalLMNextN"

# The config fields the draft layer's kernels are built for (the fused embedding + eh
# norms and the eh_proj shard: the hidden size; the MLA prologue: the attention and
# indexer shapes). SGLang loads the draft's config from the target checkpoint's, with
# only the architecture renamed, so these are the target's values.
DRAFT_HF_FIELDS = (
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
)

# Per kernel output against SGLang's op on the same GPU; each key names the test
# (test/kernels) that measures it
NUMERICS = Numerics(
    NumericsKind.TOLERANCE,
    {
        # the fused embedding gather + eh norms: SGLang's fused_eh_norm(F.embedding(...))
        # bit for bit (test_glm5_3_draft.py)
        "draft.embed_eh_norm.max_abs": 0.0,
        # the sharded eh_proj GEMV: within 1.1x the stock GEMM's own error against fp64
        # (test_glm5_3_draft.py, which reads it from here)
        "draft.eh_proj.err_ratio": 1.1,
        # the distributed top-1: torch.argmax's ids, NaN and tie rules included
        # (test_glm5_3_draft.py)
        "draft.top1.max_abs": 0.0,
        # the eh gather on the target's mailbox: the all-gather of the column shards,
        # -0 coming out +0 (test_glm5_3_hops.py)
        "draft.eh_gather.max_abs": 0.0,
    },
)


class NoDraftStep:
    """The draft spec without slot "draft": SGLang's draft forward on every step."""

    widths: tuple[int, ...] = ()
    draft_sampler = None
    draft_loop = None
    draft_extend = None
    begin_draft_round = None
    graph_variants = None

    def summary(self) -> str:
        return "GLM-5.3 / GLM-5.2 NextN draft: SGLang's forward (slot draft off)"

    def plan(self, input_ids, forward_batch, *args, **kwargs):
        return None

    def forward(self, step, original, *args, **kwargs):
        raise AssertionError("the draft covers no step without slot draft")


def _model_disable_reasons(model, hf_config) -> list[str]:
    """What the draft step needs beyond the requirements, read at load."""
    if "draft" not in SLOTS:
        return []
    from sglang.srt.runtime_context import get_parallel, get_spec

    from sgl_nvidia_mega.glm5_3.sm103 import cutlass_supported

    problems = []

    def need(ok: bool, what: str) -> None:
        if not ok:
            problems.append(what)

    ok, why = cutlass_supported()
    need(ok, f"a validated CuTe DSL ({why})")
    for reason in target.draft_capture_reasons():
        problems.append(f"an SGLang that checks its draft graphs ({reason})")
    spec, par = get_spec(), get_parallel()
    need(getattr(par, "pp_size", 1) == 1, "pipeline parallel size 1")
    need(not getattr(par, "enable_dp_attention", False), "no DP attention")
    need(not getattr(par, "enable_dp_lm_head", False), "no DP lm_head")
    need(getattr(spec, "speculative_eagle_topk", 1) == 1, "MTP / EAGLE with topk 1")
    need(
        not getattr(spec, "speculative_use_rejection_sampling", False),
        "greedy draft acceptance (no draft probs)",
    )
    return [f"the GLM-5.3 / GLM-5.2 draft step needs {p}" for p in problems]


def _build(model, hf_config, ctx):
    if "draft" not in SLOTS:
        return NoDraftStep()
    from sgl_nvidia_mega.glm5_3.draft_step import FusedDraft

    return FusedDraft(
        model,
        widths=ctx.widths,
        deadline_s=ctx.deadline_s,
        # SGLang's runtime state for the draft forward (its megakernel binding)
        binding=getattr(ctx, "binding", None),
    )


def _max_rows() -> int:
    """The draft extend's widest step with slot "draft", else 0 (SGLang's forward)."""
    return MAX_ROWS if "draft" in SLOTS else 0


SPEC = MegakernelSpec(
    name="GLM-5.3 / GLM-5.2 NextN draft",
    # the target's device, TP size, dtype and decode graphs: the draft shares its GPUs
    requires=Requirements(
        isas=target.SPEC.requires.isas,
        tp_sizes=target.SPEC.requires.tp_sizes,
        dtypes=target.SPEC.requires.dtypes,
        decode_graph_backends=target.SPEC.requires.decode_graph_backends,
        hf_config={
            **{field: HF_CONFIG[field] for field in DRAFT_HF_FIELDS},
            "model_type": MODEL_TYPE,
        },
        env=target.SPEC.requires.env,
    ),
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    numerics=NUMERICS,
    max_rows=_max_rows(),
    build=_build,
    # the contract version this package is written against, as the target's
    api_version=target.API_VERSION,
    model_disable_reasons=_model_disable_reasons,
)
