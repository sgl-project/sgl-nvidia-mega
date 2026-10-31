# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The GLM-5.3 megakernel's spec: the setups it serves and its build. It serves
GLM-5.2 as well, whose architecture and pinned config values (``HF_CONFIG``) are
GLM-5.3's.

SGLang checks the requirements at argument resolution and the model at load
(``sglang.kernels.plugin_kernels.nvidia``); an unsupported setup stops the server on every rank,
listing every reason. This module stays light: SGLang imports it in the launcher.

Stated limits of this spec:

* The hops' cross-GPU waits use a fixed bound compiled into the kernels (20 s,
  ``sm103/hop/constants.py`` ``TIMEOUT_NS``), not ``BuildContext.deadline_s``: the
  spec refuses a ``--watchdog-timeout`` whose deadline is shorter (below 40 s).
* The first capture of each width checks, on every rank, that SGLang's attention
  takes the branches the stock step copies; a reason on any rank stops every rank
  (``runtime/executor.py``).
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
    )
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.megakernels.spec import (
        Feature,
        MegakernelSpec,
        Numerics,
        NumericsKind,
        Requirements,
    )

from sgl_nvidia_mega.glm5_3 import HF_CONFIG, MODEL_TYPE, TP_SIZE
from sgl_nvidia_mega.glm5_3.plan import MAX_ROWS, SERVED_TOKENS_PER_REQUEST
from sgl_nvidia_mega.glm5_3.runtime import SLOTS, parity
from sgl_nvidia_mega.glm5_3.sm103.hop.constants import CLUSTER, TIMEOUT_NS

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": (
        "Feature",
        "SPECULATIVE_DECODING",
        "KV_OFFLOAD",
        "MegakernelSpec",
        "Numerics",
        "NumericsKind",
        "TOLERANCE",
        "Requirements",
        "decode_graph_backends",
        "config_disable_reasons",
        "model_disable_reasons",
        # slot draft: SGLang's EAGLE draft graph runners run a draft's capture check
        "EAGLE_DRAFT_CAPTURE_CHECK",
    ),
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/guards.py": (
        "deadline_s",
        "deadline_s(watchdog_timeout)",
    ),
    # the binding members the covered step reads (BINDING_MEMBERS)
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/binding.py": (
        "ForwardView",
        "KvPoolView",
        "DsaView",
        "MegakernelBinding",
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

# The co-resident cooperative grid the served kernels need at one CTA per SM: the
# 9-16-row MoE fold hop (sm103/hop/moe_fold.py, MoeFoldArBsum) launches rows x CLUSTER
# consumer CTAs plus one acknowledgement CTA as one cooperative grid, 16 x 8 + 1 = 129
# at the widest width. The hops' other cooperative forms are smaller (at most 8 rows:
# 8 x 8 + 8 = 72) or sized by the device's occupancy (the split fold), and the MoE
# engine sizes its grid from the device's co-resident 4-CTA clusters (moe/api.py,
# default_clusters; at least 8).
RESIDENT_BLOCKS = MAX_ROWS * CLUSTER + 1

# The hops' compiled wait bound (sm103/hop/constants.py TIMEOUT_NS, a kernel constant):
# SGLang's wait deadline (min(60 s, --watchdog-timeout / 2)) must not be shorter
HOP_WAIT_S = TIMEOUT_NS / 1e9

# Server arguments the step copies and the kernels are built for: argument -> the
# accepted values (None: unset). Pipeline, expert and context parallelism, DP
# attention, EPLB, HiSparse, LoRA, two-batch overlap, torch.compile, deterministic
# inference and returning routed experts / indexer top-k are SGLang's own refusals
# (the spec does not accept those features).
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
    # the expert distribution recorder is not supported
    "expert_distribution_recorder_mode": frozenset({None}),
    "moe_dp_size": frozenset({None, 1}),
    "dp_size": frozenset({None, 1}),
    # MTP / EAGLE with top-1 drafts of 3 or 5 steps: verify steps of 4 or 6 rows a
    # request (plan.SERVED_TOKENS_PER_REQUEST; NEXTN resolves to EAGLE)
    "speculative_algorithm": frozenset({"EAGLE", "NEXTN"}),
    "speculative_eagle_topk": frozenset({1}),
    "speculative_num_draft_tokens": frozenset(SERVED_TOKENS_PER_REQUEST),
    "speculative_num_steps": frozenset(t - 1 for t in SERVED_TOKENS_PER_REQUEST),
    "speculative_adaptive": frozenset(_OFF),
    "enable_multi_layer_eagle": frozenset(_OFF),
    # layouts and communication variants that change the verify step's op sequence
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
            # shared-experts fusion off (--disable-shared-experts-fusion)
            "enforce_shared_experts_fusion",
        ),
        frozenset(_OFF),
    ),
    # the step runs only inside captured decode graphs (eager steps only under the
    # test-only parity switch)
    **dict.fromkeys(
        ("disable_cuda_graph", "disable_decode_cuda_graph"),
        frozenset({*_OFF, True} if parity.eager_steps_allowed() else _OFF),
    ),
    "ep_num_redundant_experts": frozenset({None, 0}),
    # slot draft: the NextN draft's conditions, checked here before the weights load
    # (SGLang checks a draft spec only at load): greedy draft acceptance
    **(
        {"speculative_use_rejection_sampling": frozenset(_OFF)}
        if "draft" in SLOTS
        else {}
    ),
}

# SGLang environment variables of the copied attention step: the fused DSA top-k page
# table on; the indexer's rank-0 top-k broadcast, the LoRA and born-FP8-query
# variants off
ENV = {
    "SGLANG_DSA_FUSE_TOPK": True,
    "SGLANG_DSA_TOPK_BROADCAST": False,
    "SGLANG_EXPERIMENTAL_LORA_OPTI": False,
    "SGLANG_ENABLE_DSA_Q8KV8_BORN_FP8_Q": False,
    # slot draft: the draft's fused embedding reads the replicated table (with the
    # flag, SGLang sets the variable when it is unset: megakernels FLAG_SETS_ENV)
    **({"SGLANG_ENABLE_EMBED_REPLICATION": True} if "draft" in SLOTS else {}),
}


def decode_graph_backends() -> frozenset:
    """FULL decode CUDA graphs; eager steps too under the test-only parity switch."""
    return frozenset({"full", "disabled"} if parity.eager_steps_allowed() else {"full"})


REQUIREMENTS = Requirements(
    isas=frozenset({"sm_103"}),
    tp_sizes=frozenset({TP_SIZE}),
    resident_blocks=RESIDENT_BLOCKS,
    dtypes=frozenset({"bfloat16"}),
    decode_graph_backends=decode_graph_backends(),
    server_args=SERVER_ARGS,
    hf_config={**HF_CONFIG, "model_type": MODEL_TYPE},
    env=ENV,
)

# Per kernel output against SGLang's op (or an fp64 reference, where SGLang's own op
# carries the same error) on the same GPU; each key names the test (test/kernels) that
# measures it, and those tests read their bounds from here
TOLERANCES = {
    # q_a / kv_a (+ the indexer's key / head-gate) on the small-M GEMM: within one
    # bf16 ulp of the exact value plus 2^-12 x |x| @ |W| (the stock fused-A GEMM meets
    # the same bound; test_glm5_3_attention_qkv_a.py)
    "attn.qkv_a.gemm_bound": 2**-12,
    # the MLA prologue: the MLA rows, the index-K rows + scales and the normed q_a
    # equal SGLang's prologue byte for byte (test_glm5_3_attention_prologue.py)
    "attn.prologue.caches.max_abs": 0.0,
    # its FP8 query differs only through the q_b / W_UK GEMM order: its error vs
    # fp64 at most 1.05x (mean) / 2x (max) the reference's own
    # (test_glm5_3_attention_prologue.py)
    "attn.prologue.q_fp8.mean_err_ratio": 1.05,
    "attn.prologue.q_fp8.max_err_ratio": 2.0,
    # the indexer query: within half an E4M3 step plus one bf16 rounding of fp64;
    # its head gates exact (test_glm5_3_attention_index_q.py)
    "attn.index_q.e4m3_steps": 0.5,
    "attn.index_q.weights.max_abs": 0.0,
    # the sparse MLA against fp64, relative L2 per (row, head)
    # (test_glm5_3_attention_sparse_mla.py)
    "attn.sparse_mla.rel_l2": 5e-3,
    # the value up-projection: one bf16 ulp plus 2^-12 x |x| @ |W| of fp64
    # (test_glm5_3_attention_uv_proj.py)
    "attn.uv_proj.gemm_bound": 2**-12,
    # the MoE routing (expert ids, bf16 weights) and the routed experts (shared
    # expert off) equal FlashInfer's TRT-LLM NVFP4 MoE bit for bit
    # (test_glm5_3_moe.py)
    "moe.routing.max_abs": 0.0,
    "moe.routed.max_abs": 0.0,
    # the routed experts with the shared expert against SGLang's routed MoE + shared
    # expert: the largest difference relative to the output's peak magnitude,
    # measured 7.353e-3 (test_glm5_3_moe.py::test_with_shared_matches_stock, fixed
    # seeds); the bound is that value rounded up to a power of two
    "moe.with_shared.rel": 2**-7,
    # the hops: the residual equals the rank-order fp32 sum of the eight bf16
    # partials with one bf16 rounding; the normed row within two bf16 ulps of fp64
    # RMSNorm; the o_proj GEMV within one ulp plus 2^-8 x sum |x| |w|; the fold's
    # residual its documented rounding (test_glm5_3_hops.py)
    "hop.residual.max_abs": 0.0,
    "hop.normed.ulp": 2,
    "hop.o_proj.gemv_bound": 2**-8,
    "hop.fold.residual.max_abs": 0.0,
    # the o_proj hop's router (MoE layers): its logits within 2^-16 x (rstd x the sum
    # of the absolute terms + |logit|) of the documented arithmetic in fp64, its
    # scores within 2^-11 (the hardware tanh of the sigmoid, as in the stock routing
    # kernel), and its expert ids and bf16 weights the DeepSeek-V3 routing of those;
    # against the stock routing of SGLang's gate on the hop's normed output, the same
    # expert ids wherever the stock 8th and 9th keys differ by more than 2^-6
    # (test_glm5_3_hops.py)
    "hop.router.logit_bound": 2**-16,
    "hop.router.score_abs": 2**-11,
    "hop.router.stock_key_margin": 2**-6,
    # and its expert ids and weights the DeepSeek-V3 routing of the stock gate's own
    # logits (the hop's normed output against SGLang's gate, in fp64) within 2^-4:
    # the router reads its logits through the hoisted weights on the unrounded
    # partials, so they differ from the stock logits by design, but by less than this
    # (test_glm5_3_hops.py)
    "hop.router.stock_logit_abs": 2**-4,
    # slot dnorm: the epilogue-scaled q_a / kv_a projection (the qkv_a bound;
    # test_glm5_3_attention_qkv_a.py) and the fold's block sums, which give the
    # normed fold bit for bit (test_glm5_3_hops.py)
    "dnorm.qkv_a.gemm_bound": 2**-12,
    "dnorm.block_sums.max_abs": 0.0,
}
# the keys of slot dnorm, declared only when it runs (SLOTS_ENV can leave it out)
DNORM_KEYS = ("dnorm.qkv_a.gemm_bound", "dnorm.block_sums.max_abs")
NUMERICS = Numerics(
    kind=NumericsKind.TOLERANCE,
    tolerances={
        key: value
        for key, value in TOLERANCES.items()
        if "dnorm" in SLOTS or key not in DNORM_KEYS
    },
)


def toolchain_reasons() -> list[str]:
    """The kernel toolchains, and the CuTe DSL versions the kernels were validated on."""
    reasons = []
    for module, what in (
        ("cutlass", "CuTe DSL (nvidia-cutlass-dsl)"),
        ("flashinfer", "FlashInfer"),
        ("cuda.bindings", "the CUDA Python bindings"),
    ):
        if importlib.util.find_spec(module) is None:
            reasons.append(f"needs {what}")
    if not reasons:
        from sgl_nvidia_mega.glm5_3.sm103 import cutlass_supported

        ok, why = cutlass_supported()
        if not ok:
            reasons.append(why)
    return reasons


# The SGLang megakernel binding members the covered steps read: the
# DSA view and the MLA pool members come with SGLang's megakernel DSA binding view
BINDING_MEMBERS = {
    "MegakernelBinding": ("forward_view",),
    "ForwardView": ("dsa",),
    "KvPoolView": (
        "get_key_buffer",
        "get_index_k_with_scale_buffer",
        "index_page_size",
        "set_mla_kv_buffer",
    ),
    "DsaView": (
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "kv_cache_dim",
        "physical_page_size",
        "decode_impl",
        "use_mha",
        "kv_cache_dtype",
        "use_fused_topk",
        "topk_backend",
        "dcp_size",
        "index_kpool",
        "hisparse",
        "workspace_buffer",
        "multi_ctas_kv_counter_buffer",
        "indexer_metadata",
    ),
}


def binding_reasons() -> list[str]:
    """The SGLang binding members the covered steps read, each one present."""
    try:
        import sglang.srt.kernel_plugins.nvidia.binding as binding
    except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
        import sglang.srt.megakernels.binding as binding

    missing = [
        f"{owner}.{member}"
        for owner, members in BINDING_MEMBERS.items()
        for member in members
        if not hasattr(getattr(binding, owner, None), member)
    ]
    if missing:
        return [
            "this SGLang's megakernel binding lacks the DSA view members "
            f"{', '.join(missing)}; update SGLang"
        ]
    return []


def draft_capture_reasons() -> list[str]:
    """Slot draft: SGLang's EAGLE draft graph runners must run the draft spec's
    capture check (its liveness check, and the refusal when none of its draft graphs
    runs it), so that no draft graph runs the draft step unchecked."""
    from sgl_nvidia_mega.glm5_3 import runtime

    if "draft" not in runtime.SLOTS:
        return []
    try:
        import sglang.kernels.plugin_kernels.nvidia.spec as layer
    except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
        import sglang.srt.megakernels.spec as layer

    if getattr(layer, "EAGLE_DRAFT_CAPTURE_CHECK", False):
        return []
    return [
        "slot draft: this SGLang's EAGLE draft graph runners do not run a draft "
        "spec's capture check (spec.EAGLE_DRAFT_CAPTURE_CHECK); update SGLang, or "
        f"leave slot draft out of {runtime.SLOTS_ENV}"
    ]


def deadline_reasons(cfg) -> list[str]:
    """The hops' compiled wait bound must fit SGLang's wait deadline."""
    try:
        from sglang.srt.kernel_plugins.nvidia.binding import deadline_s
    except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
        from sglang.srt.megakernels.guards import deadline_s

    deadline = deadline_s(cfg.watchdog_timeout)
    if deadline < HOP_WAIT_S:
        return [
            f"--watchdog-timeout {cfg.watchdog_timeout:g} s leaves a wait deadline of "
            f"{deadline:g} s, under the hops' {HOP_WAIT_S:g} s; pass at least "
            f"{2 * HOP_WAIT_S:g}"
        ]
    return []


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    from sgl_nvidia_mega.glm5_3 import runtime

    reasons = toolchain_reasons() + binding_reasons() + runtime.slot_reasons()
    reasons += draft_capture_reasons() + deadline_reasons(cfg)
    if (
        cfg.cuda_graph_config.decode.backend == "disabled"
        and not parity.eager_steps_allowed()
    ):
        reasons.append(
            "decode CUDA graphs are disabled: the covered step runs only inside "
            "captured decode graphs; enable decode CUDA graphs"
        )
    return reasons


def _model_disable_reasons(model, hf_config) -> list[str]:
    from sgl_nvidia_mega.glm5_3 import runtime

    return runtime.model_disable_reasons(model)


def _build(model, hf_config, ctx):
    from sgl_nvidia_mega.glm5_3 import runtime

    return runtime.build(model, ctx)


SPEC = MegakernelSpec(
    name="GLM-5.3 / GLM-5.2",
    requires=REQUIREMENTS,
    # MTP verify steps, also when serving with HiCache (KV offload: the covered steps
    # read only the running requests' device KV pages)
    features=frozenset({Feature.SPECULATIVE_DECODING, Feature.KV_OFFLOAD}),
    numerics=NUMERICS,
    # whole requests up to 16 verify rows (plan.supported_rows)
    max_rows=MAX_ROWS,
    build=_build,
    api_version=API_VERSION,
    model_disable_reasons=_model_disable_reasons,
    config_disable_reasons=_config_disable_reasons,
)
