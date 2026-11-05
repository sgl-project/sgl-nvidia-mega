# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team
"""The GLM-5 DSA megakernel's spec: the setups it serves and its build. It serves
GLM-5.2 (NVFP4) on SM103 (B300) and SM107 (Rubin / VR NVL72).

SGLang checks the requirements at argument resolution and the model at load;
an unsupported setup stops the server on every rank, listing every reason.
This module stays light: SGLang imports it in the launcher.

Stated limits of this spec:

* The hops' cross-GPU waits use a fixed bound compiled into the kernels
  (``kernels/cute/hop/constants.py`` ``TIMEOUT_NS``), not ``BuildContext.deadline_s``:
  the spec refuses a ``--watchdog-timeout`` whose deadline is shorter (below 40 s).
* The first capture of each width checks, on every rank, that SGLang's attention
  takes the branches the stock step copies; a reason on any rank stops every rank
  (``runtime/executor.py``).
* The covered step reads the DSA backend and its KV pool through SGLang's running
  attention backend (``get_attn_backend()``), not through a binding view, so it runs on
  SGLang's megakernel API version 2 as released.
* TP8 may span two 4-GPU trays of one NVL72 rack (multi-node TP); without NVLink
  multicast between them, see ``switches.py`` (``ALLOW_NO_AR_FUSION``).
"""

from __future__ import annotations

import importlib.util

try:
    from sglang.kernels.plugin_kernels.nvidia.spec import (
        Feature,
        MegakernelSpec,
        Numerics,
        NumericsKind,
        Requirements,
        Step,
    )
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.megakernels.spec import (
        Feature,
        MegakernelSpec,
        Numerics,
        NumericsKind,
        Requirements,
        Step,
    )

from sgl_nvidia_mega.glm_dsa import HF_CONFIG, MODEL_TYPE, TP_SIZE
from sgl_nvidia_mega.glm_dsa.eligibility import (
    MAX_ROWS,
    SERVED_TOKENS_PER_REQUEST,
    draft_length_reason,
)
from sgl_nvidia_mega.glm_dsa.kernels.cute.hop.constants import CLUSTER, TIMEOUT_NS
from sgl_nvidia_mega.glm_dsa.runtime import SLOTS, SLOTS_ENV, parity, slot_reasons

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": (
        "Feature",
        "SPECULATIVE_DECODING",
        "MULTI_NODE_TP",
        "KV_OFFLOAD",
        "MegakernelSpec",
        "Numerics",
        "NumericsKind",
        "TOLERANCE",
        "Requirements",
        "Step",
        "decode_graph_backends",
        "config_disable_reasons",
        "model_disable_reasons",
        "draft_loop",
        "draft_extend",
        "begin_draft_round",
        "graph_variants",
    ),
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/guards.py": (
        "deadline_s",
        "deadline_s(watchdog_timeout)",
    ),
    "srt/model_executor/cuda_graph_config.py": ("decode", "backend", '"disabled"'),
    "srt/environ.py": (
        "SGLANG_DSA_FUSE_TOPK",
        "SGLANG_DSA_TOPK_BROADCAST",
        "SGLANG_EXPERIMENTAL_LORA_OPTI",
        "SGLANG_ENABLE_DSA_Q8KV8_BORN_FP8_Q",
        "SGLANG_ENABLE_EMBED_REPLICATION",
    ),
    "srt/arg_groups/fields/*.py": (
        "cuda_graph_config",
        "watchdog_timeout",
        "kv_cache_dtype",
        "page_size",
        "attention_backend",
        "decode_attention_backend",
        "dsa_decode_backend",
        "moe_runner_backend",
        "moe_a2a_backend",
        "init_expert_location",
        "expert_distribution_recorder_mode",
        "moe_dp_size",
        "dp_size",
        "speculative_algorithm",
        "speculative_eagle_topk",
        "speculative_num_draft_tokens",
        "speculative_num_steps",
        "speculative_adaptive",
        "enable_multi_layer_eagle",
        "enable_cp_decode_attn_tp",
        "enable_kv_cache_sharding",
        "enable_layernorm_sp",
        "enable_attn_tp_input_scattered",
        "enable_dense_mlp_attn_tp",
        "enable_shared_experts_attn_tp",
        "enable_quant_communications",
        "enable_fused_moe_sum_all_reduce",
        "enable_dp_lm_head",
        "enable_dsa_cache_layer_split",
        "enforce_shared_experts_fusion",
        "ep_num_redundant_experts",
        "disable_cuda_graph",
        "disable_decode_cuda_graph",
        "speculative_use_rejection_sampling",
    ),
}

ARCHITECTURE = "GlmMoeDsaForCausalLM"
# the SGLang megakernel contract this spec is written against
API_VERSION = 2

# The co-resident cooperative grid the served kernels need at one CTA per SM
RESIDENT_BLOCKS = MAX_ROWS * CLUSTER + 1

# The hops' compiled wait bound: SGLang's wait deadline must not be shorter
HOP_WAIT_S = TIMEOUT_NS / 1e9

_OFF = (None, False)
SERVER_ARGS = {
    "kv_cache_dtype": frozenset({"fp8_e4m3"}),
    "page_size": frozenset({64}),
    "attention_backend": frozenset({"dsa"}),
    "decode_attention_backend": frozenset({None, "dsa"}),
    "dsa_decode_backend": frozenset({"trtllm"}),
    "moe_runner_backend": frozenset({"flashinfer_trtllm"}),
    "moe_a2a_backend": frozenset({None, "none"}),
    "init_expert_location": frozenset({None, "trivial"}),
    "expert_distribution_recorder_mode": frozenset({None}),
    "moe_dp_size": frozenset({None, 1}),
    "dp_size": frozenset({None, 1}),
    "speculative_algorithm": frozenset({"EAGLE", "NEXTN"}),
    "speculative_eagle_topk": frozenset({1}),
    # top-1 drafts of 3 or 5 steps: verify steps of 4 or 6 rows a request (the pair
    # is checked in _config_disable_reasons)
    "speculative_num_draft_tokens": frozenset(SERVED_TOKENS_PER_REQUEST),
    "speculative_num_steps": frozenset(t - 1 for t in SERVED_TOKENS_PER_REQUEST),
    "speculative_adaptive": frozenset(_OFF),
    "enable_multi_layer_eagle": frozenset(_OFF),
    **dict.fromkeys(
        (
            "enable_cp_decode_attn_tp",
            "enable_kv_cache_sharding",
            "enable_layernorm_sp",
            "enable_attn_tp_input_scattered",
            "enable_dense_mlp_attn_tp",
            "enable_shared_experts_attn_tp",
            "enable_quant_communications",
            "enable_fused_moe_sum_all_reduce",
            "enable_dp_lm_head",
            "enable_dsa_cache_layer_split",
            "enforce_shared_experts_fusion",
        ),
        frozenset(_OFF),
    ),
    **dict.fromkeys(
        ("disable_cuda_graph", "disable_decode_cuda_graph"),
        frozenset({*_OFF, True} if parity.eager_steps_allowed() else _OFF),
    ),
    "ep_num_redundant_experts": frozenset({None, 0}),
    **(
        {"speculative_use_rejection_sampling": frozenset(_OFF)}
        if "draft" in SLOTS
        else {}
    ),
}

ENV = {
    "SGLANG_DSA_FUSE_TOPK": True,
    "SGLANG_DSA_TOPK_BROADCAST": False,
    "SGLANG_EXPERIMENTAL_LORA_OPTI": False,
    "SGLANG_ENABLE_DSA_Q8KV8_BORN_FP8_Q": False,
    **({"SGLANG_ENABLE_EMBED_REPLICATION": True} if "draft" in SLOTS else {}),
}


def decode_graph_backends() -> frozenset:
    return frozenset({"full", "disabled"} if parity.eager_steps_allowed() else {"full"})


REQUIREMENTS = Requirements(
    isas=frozenset({"sm_103", "sm_107"}),
    tp_sizes=frozenset({TP_SIZE}),
    resident_blocks=RESIDENT_BLOCKS,
    dtypes=frozenset({"bfloat16"}),
    decode_graph_backends=decode_graph_backends(),
    server_args=SERVER_ARGS,
    hf_config={**HF_CONFIG, "model_type": MODEL_TYPE},
    env=ENV,
)

# Per-kernel output tolerances vs. SGLang's op (or an fp64 reference).
# Each key names the test (test/kernels) that measures it.
TOLERANCES = {
    # q_a / kv_a (+ indexer's key / head-gate) on the small-M GEMM
    "attn.qkv_a.gemm_bound": 2**-12,
    # the MLA prologue: MLA rows, index-K rows + scales and normed q_a equal SGLang's
    "attn.prologue.caches.max_abs": 0.0,
    # its FP8 query differs only through the q_b / W_UK GEMM order
    "attn.prologue.q_fp8.mean_err_ratio": 1.05,
    "attn.prologue.q_fp8.max_err_ratio": 2.0,
    # the indexer query: within half an E4M3 step plus one bf16 rounding of fp64
    "attn.index_q.e4m3_steps": 0.5,
    "attn.index_q.weights.max_abs": 0.0,
    # the sparse MLA against fp64, relative L2 per (row, head)
    "attn.sparse_mla.rel_l2": 5e-3,
    # the value up-projection: one bf16 ulp plus 2^-12 x |x| @ |W| of fp64
    "attn.uv_proj.gemm_bound": 2**-12,
    # the MoE routing (expert ids, bf16 weights) and routed experts equal FlashInfer's
    "moe.routing.max_abs": 0.0,
    "moe.routed.max_abs": 0.0,
    # routed + shared expert vs SGLang's MoE + shared expert
    "moe.with_shared.rel": 2**-7,
    # the hops: residual, normed, o_proj GEMV, fold residual
    "hop.residual.max_abs": 0.0,
    "hop.normed.ulp": 2,
    "hop.o_proj.gemv_bound": 2**-8,
    "hop.fold.residual.max_abs": 0.0,
    # the o_proj hop's router
    "hop.router.logit_bound": 2**-16,
    "hop.router.score_abs": 2**-11,
    "hop.router.stock_key_margin": 2**-6,
    "hop.router.stock_logit_abs": 2**-4,
    # slot dnorm: epilogue-scaled q_a / kv_a projection and fold block sums
    "dnorm.qkv_a.gemm_bound": 2**-12,
    "dnorm.block_sums.max_abs": 0.0,
}
DNORM_KEYS = ("dnorm.qkv_a.gemm_bound", "dnorm.block_sums.max_abs")
NUMERICS = Numerics(
    kind=NumericsKind.TOLERANCE,
    tolerances={
        key: value
        for key, value in TOLERANCES.items()
        if "dnorm" in SLOTS or key not in DNORM_KEYS
    },
)

def toolchain_reasons() -> list:
    reasons = []
    for module, what in (
        ("cutlass", "CuTe DSL (nvidia-cutlass-dsl)"),
        ("flashinfer", "FlashInfer"),
        ("cuda.bindings", "the CUDA Python bindings"),
    ):
        if importlib.util.find_spec(module) is None:
            reasons.append(f"needs {what}")
    if not reasons:
        from sgl_nvidia_mega.glm_dsa.kernels.cute import cutlass_supported

        ok, why = cutlass_supported()
        if not ok:
            reasons.append(why)
    return reasons


def deadline_reasons(cfg) -> list:
    try:
        from sglang.srt.kernel_plugins.nvidia.binding import deadline_s
    except ModuleNotFoundError:
        from sglang.srt.megakernels.guards import deadline_s

    dl = deadline_s(cfg.watchdog_timeout)
    if dl < HOP_WAIT_S:
        return [
            f"--watchdog-timeout {cfg.watchdog_timeout:g} s leaves a wait deadline of "
            f"{dl:g} s, under the hops' {HOP_WAIT_S:g} s; pass at least "
            f"{2 * HOP_WAIT_S:g}"
        ]
    return []


def _config_disable_reasons(cfg, hf_config) -> list:
    reasons = toolchain_reasons() + slot_reasons() + deadline_reasons(cfg)
    reasons += draft_length_reason(cfg)
    if (
        cfg.cuda_graph_config.decode.backend == "disabled"
        and not parity.eager_steps_allowed()
    ):
        reasons.append(
            "decode CUDA graphs are disabled: the covered step runs only inside "
            "captured decode graphs; enable decode CUDA graphs"
        )
    return reasons


def _model_disable_reasons(model, hf_config) -> list:
    from sgl_nvidia_mega.glm_dsa import eligibility

    return eligibility.model_construction_reasons(model, hf_config)


class _Built:
    """The installed target forward: covered verify steps run the fused stack."""

    # SGLang's own proposal, draft loop and draft extend; one graph per batch size
    draft_sampler = None
    draft_loop = None
    draft_extend = None
    begin_draft_round = None
    graph_variants = None

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
    from sgl_nvidia_mega.glm_dsa.runtime.executor import GlmDsaFusedDecode

    fused = GlmDsaFusedDecode(model)
    fused.prepare()
    return _Built(model, fused)


SPEC = MegakernelSpec(
    name="GLM-5.2 / GLM-5 DSA",
    requires=REQUIREMENTS,
    features=frozenset(
        {
            Feature.SPECULATIVE_DECODING,
            # TP8 spans the two 4-GPU trays of one NVL72 rack (Rubin, GB300 NVL72)
            Feature.MULTI_NODE_TP,
            Feature.KV_OFFLOAD,
        }
    ),
    numerics=NUMERICS,
    max_rows=MAX_ROWS,
    build=_build,
    api_version=API_VERSION,
    model_disable_reasons=_model_disable_reasons,
    config_disable_reasons=_config_disable_reasons,
)
