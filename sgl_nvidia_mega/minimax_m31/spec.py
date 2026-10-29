# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The MiniMax-M3.1 megakernel's spec: the setups it serves and its build.

SGLang checks the requirements at argument resolution and the model at load
(``sglang.kernels.plugin_kernels.nvidia``); an unsupported setup stops the server on
every rank, listing every reason. This module stays light: SGLang imports it in the
launcher.
"""

from __future__ import annotations

import importlib.util
import os

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

from sgl_nvidia_mega.minimax_m31 import (
    HF_CONFIG,
    MAX_CONTEXT,
    PAGE,
    ROWS,
    SPARSE_CONFIG,
    SPARSE_OFF,
    TP,
)

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
        "config_disable_reasons",
        "model_disable_reasons",
    ),
    "srt/model_executor/runner/__init__.py": ("get_is_capture_mode",),
    "srt/arg_groups/fields/*.py": (
        "speculative_algorithm",
        "speculative_num_draft_tokens",
        "speculative_eagle_topk",
        "page_size",
        "kv_cache_dtype",
        "moe_runner_backend",
        "moe_a2a_backend",
        "fp8_gemm_runner_backend",
        "quantization",
        "context_length",
        "disable_overlap_schedule",
    ),
    "srt/environ.py": (
        "envs",
        "SGLANG_FORCE_COARSE_WAR_BARRIER",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM",
    ),
}

# the SGLang megakernel contract this spec is written against
API_VERSION = 1
# MXFP8 GEMM backends that keep each linear's checkpoint-layout weight (flashinfer_trtllm
# shuffles it in place)
MXFP8_GEMM_BACKENDS = frozenset(
    {"auto", "deep_gemm", "flashinfer_cutlass", "flashinfer_cutedsl"}
)


# Test only: with SGLANG_NVIDIA_MEGAKERNEL_PARITY=1 every covered step also runs eagerly,
# after SGLang's own forward on the same step, logging how far apart they are
PARITY_ENV = "SGLANG_NVIDIA_MEGAKERNEL_PARITY"


def parity() -> bool:
    return os.environ.get(PARITY_ENV, "") not in ("", "0")


def serves_this_step() -> bool:
    """A covered step runs the megakernel only while SGLang captures its decode CUDA
    graphs (their warm-up runs included), or eagerly under the parity switch; any other
    eager step runs SGLang's forward."""
    from sglang.srt.model_executor.runner import get_is_capture_mode

    return get_is_capture_mode() or parity()


REQUIREMENTS = Requirements(
    isas=frozenset({"sm_103"}),
    tp_sizes=frozenset({TP}),
    # MegaAttention's persistent grid, all of it resident (the MoE takes 144)
    resident_blocks=148,
    dtypes=frozenset({"bfloat16"}),
    decode_graph_backends=frozenset({"full", "disabled"} if parity() else {"full"}),
    server_args={
        # one request's eight-row DSpark verify: the MoE kernel's fixed rows
        "speculative_algorithm": frozenset({"DSPARK"}),
        "speculative_num_draft_tokens": frozenset({ROWS}),
        "page_size": frozenset({PAGE}),
        # SGLang's NVFP4 MSA-v1 KV pool
        "kv_cache_dtype": frozenset({"fp8_e4m3"}),
        # TP4 routed experts (the kernels keep a copy in their own layout)
        "moe_runner_backend": frozenset({"flashinfer_trtllm_routed"}),
        "moe_a2a_backend": frozenset({"none"}),
        "fp8_gemm_runner_backend": MXFP8_GEMM_BACKENDS,
    },
)

# The covered step's logits against SGLang's forward on the same eager step
# (SGLANG_NVIDIA_MEGAKERNEL_PARITY=1; 835 verify steps of the first 20 GSM8K replies as
# test/e2e/test_minimax_m31_server.py asks them, on 4 x GB300), measured with about 15%
# headroom. Per step: the largest |difference| over the largest |logit|, its median and
# 99th percentile (measured 0.101, 0.262); the fraction of rows whose argmax agrees, a
# lower bound (0.957); the largest row KL(SGLang || this) at the median and 99th
# percentile (0.045, 1.00). The difference is FP8 / NVFP4
# rounding noise: fed SGLang's inputs, each layer's attention and MoE are as far from an
# FP32 oracle as SGLang's own (0.087 / 0.085 and 0.096 / 0.095 relative), and SGLang's
# forward moves as far from itself when only its embedding moves by one BF16 ulp (argmax
# agreement 95.6% either way). GSM8K (100, greedy) and the mean accept length matched
# SGLang's (0.97 / 0.97, 4.37 / 4.34)
NUMERICS = Numerics(
    kind=NumericsKind.TOLERANCE,
    tolerances={
        "verify.logits.rel.p50": 0.12,
        "verify.logits.rel.p99": 0.3,
        "verify.logits.top1": 0.94,
        "verify.logits.kl.p50": 0.052,
        "verify.logits.kl.p99": 1.15,
    },
)


def text_config(hf_config):
    return getattr(hf_config, "text_config", None) or hf_config


def scaled_rope(config) -> bool:
    """Whether the config scales RoPE (transformers may report the default type)."""
    scaling = getattr(config, "rope_scaling", None) or {}
    return scaling.get("rope_type", scaling.get("type", "default")) != "default"


def toolchain_reasons() -> list[str]:
    """The kernels' build toolchain: the CUDA Python bindings and nvcc 13.4 or newer."""
    if importlib.util.find_spec("cuda.bindings") is None:
        return ["needs the CUDA Python bindings"]
    from sgl_nvidia_mega.minimax_m31.sm103 import nvcc

    reason = nvcc.why_not(nvcc.QMUL4_VERSION)
    return [reason] if reason else []


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    """The checks ``REQUIREMENTS`` cannot state: nested HF fields, derived values and the
    toolchain."""
    from sglang.srt.environ import envs

    reasons = toolchain_reasons()
    text = text_config(hf_config)
    for key, want in HF_CONFIG.items():
        value = getattr(text, key, None)
        if value != want:
            reasons.append(f"config.{key} {value!r} is not {want!r}")
    if scaled_rope(text):
        reasons.append(
            f"config.rope_scaling {text.rope_scaling!r}: the kernels read plain RoPE"
        )
    sparse = getattr(text, "sparse_attention_config", None) or {}
    for key, want in SPARSE_CONFIG.items():
        if sparse.get(key) != want:
            reasons.append(
                f"sparse_attention_config.{key} {sparse.get(key)!r} is not {want!r}"
            )
    for key in SPARSE_OFF:
        if sparse.get(key):
            reasons.append(f"sparse_attention_config.{key} is not supported")
    if sparse.get("sparse_index_compress_type") not in (None, "none"):
        reasons.append("MSA-v1 index compression is not supported")
    moe = getattr(text, "moe_layer_freq", None) or []
    freq = sparse.get("sparse_attention_freq") or [0] * len(moe)
    if not any(moe):
        reasons.append("config.moe_layer_freq has no MoE layer")
    elif len(freq) != len(moe) or not all(f for f, m in zip(freq, moe) if m):
        reasons.append("every MoE layer must be an MSA sparse attention layer")
    if cfg.quantization not in (None, "minimax_m3_nvfp4"):
        reasons.append(
            f"--quantization {cfg.quantization!r}: the kernels read the minimax_m3_nvfp4 "
            "checkpoint (MXFP8 attention, NVFP4 experts)"
        )
    if (cfg.speculative_eagle_topk or 1) > 1:
        reasons.append("tree verify (--speculative-eagle-topk > 1) is not supported")
    context = cfg.context_length or getattr(text, "max_position_embeddings", 0)
    if context and context > MAX_CONTEXT:
        reasons.append(
            f"context length {context} exceeds the kernels' {MAX_CONTEXT} tokens; pass "
            f"--context-length <= {MAX_CONTEXT}"
        )
    # MegaAttention keeps all its CTAs resident and they wait for each other: no
    # kernel of the next batch may hold SMs while it runs
    if not (envs.SGLANG_FORCE_COARSE_WAR_BARRIER.get() or cfg.disable_overlap_schedule):
        reasons.append(
            "the next batch's preparation would overlap the persistent kernels; set "
            "SGLANG_FORCE_COARSE_WAR_BARRIER=1 (or --disable-overlap-schedule)"
        )
    if envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        reasons.append(
            "SGLANG_ENABLE_OVERLAP_PLAN_STREAM runs plan kernels beside the persistent "
            "kernels"
        )
    return reasons


def _model_disable_reasons(model, hf_config) -> list[str]:
    from sgl_nvidia_mega.minimax_m31 import runtime

    return runtime.model_disable_reasons(model, hf_config)


def _build(model, hf_config, ctx):
    from sgl_nvidia_mega.minimax_m31 import runtime

    return runtime.VerifyStep(model, hf_config, ctx)


SPEC = MegakernelSpec(
    name="MiniMax-M3.1",
    requires=REQUIREMENTS,
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    numerics=NUMERICS,
    max_rows=ROWS,
    build=_build,
    api_version=API_VERSION,
    model_disable_reasons=_model_disable_reasons,
    config_disable_reasons=_config_disable_reasons,
)
