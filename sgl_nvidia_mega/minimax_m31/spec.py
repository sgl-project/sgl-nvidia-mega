# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The MiniMax-M3.1 megakernel's spec: the setups it serves and its build.

One spec serves DSpark verify steps of 1-16 requests (8 rows a request, 8-128 rows) in
one server: slot mega at one request, SGLang's step code with the bitwise MoE slots
moe_nofill and moe_swap at 2-16 (``plan``, ``runtime``). SGLang checks the
requirements at argument resolution and the model at load
(``sglang.kernels.plugin_kernels.nvidia``); an unsupported setup stops the server on
every rank, listing every reason. The checks of slot mega's kernels (their toolchain,
the config fields they are built for, the context their page tables hold, the
persistent grid's residency) apply while it is requested; so do those of the DSpark
draft's slot markov (``draft_config_reasons``), which the draft
spec builds (``draft``). This module stays light: SGLang imports it in the launcher.
"""

from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path

from sglang.kernels.plugin_kernels.nvidia.spec import (
    Feature,
    MegakernelSpec,
    Numerics,
    NumericsKind,
    Requirements,
)

from sgl_nvidia_mega.minimax_m31 import (
    HF_CONFIG,
    MAX_CONTEXT,
    MAX_ROWS,
    PAGE,
    ROWS,
    SPARSE_CONFIG,
    SPARSE_OFF,
    TP,
)
from sgl_nvidia_mega.minimax_m31.runtime import (
    BITWISE_SLOTS,
    SLOTS,
    draft_slots,
    slot_reasons,
    target_slots,
)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py": (
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
        # the newest of the SGLang arguments the package mirrors (host_reasons)
        "speculative_dspark_lm_head_dtype",
    ),
    "srt/environ.py": (
        "envs",
        "SGLANG_FORCE_COARSE_WAR_BARRIER",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM",
        # the DSpark switches draft_config_reasons reads
        "SGLANG_DSPARK_FOLDED_PROPOSAL",
        "SGLANG_DSPARK_OPT_INT8_MARKOV_WALK",
    ),
}

# the SGLang megakernel contract this spec is written against
API_VERSION = 2
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
    # slot mega: MegaAttention's persistent grid, all of it resident (the MoE takes 144)
    resident_blocks=148 if "mega" in SLOTS else 0,
    dtypes=frozenset({"bfloat16"}),
    decode_graph_backends=frozenset({"full", "disabled"} if parity() else {"full"}),
    server_args={
        # DSpark verify steps of eight rows a request (slot mega: the MoE kernel's
        # fixed rows); 1-16 requests a step
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

# Slot mega's step's logits against SGLang's forward on the same eager step
# (SGLANG_NVIDIA_MEGAKERNEL_PARITY=1; the verify steps of the first 20 GSM8K replies as
# test/e2e/test_minimax_m31_parity.py asks them, on 4 x GB300), measured with about 15%
# headroom on SGLang 8cb94fcc3f and d89bc5b574 with FlashInfer 0.7.4, where the same
# kernels sit farther from SGLang's forward than on the SGLang with FlashInfer 0.7.2
# they were first measured on (0.101, 0.262; 0.957; 0.045, 1.00). Per step: the largest
# |difference| over the largest |logit|, its median and 99th percentile (measured
# 0.113, 0.265); the fraction of rows whose argmax agrees, a lower bound (0.941); the
# largest row KL(SGLang || this) at the median and 99th percentile (0.069, 1.27). The
# test has one of two outcomes (KL 0.0677 / 1.271 or 0.0693 / 1.181); the bounds cover
# both. The difference is FP8 / NVFP4
# rounding noise: fed SGLang's inputs, each layer's attention and MoE are as far from an
# FP32 oracle as SGLang's own (0.087 / 0.085 and 0.096 / 0.095 relative), and SGLang's
# forward moves as far from itself when only its embedding moves by one BF16 ulp (argmax
# agreement 95.6% either way). GSM8K (100, greedy) and the mean accept length matched
# SGLang's (0.97 / 0.97, 4.37 / 4.34)
MEGA_TOLERANCES = {
    "verify.logits.rel.p50": 0.13,
    "verify.logits.rel.p99": 0.31,
    "verify.logits.top1": 0.93,
    "verify.logits.kl.p50": 0.08,
    "verify.logits.kl.p99": 1.46,
}


def _numerics(slots: frozenset = SLOTS) -> Numerics:
    """TOLERANCE with slot mega (its steps' tolerances above), else BITWISE: the stock
    executor runs SGLang's modules in SGLang's order, and the other slots
    (runtime.BITWISE_SLOTS) run SGLang's kernels with SGLang's arguments. These are the
    target's verify steps against SGLang's on the same inputs; the DSpark draft's slots
    change the draft tokens those steps verify, and state their tolerances in the draft
    spec's numerics (draft.NUMERICS)."""
    if target_slots(slots) <= BITWISE_SLOTS:
        return Numerics(kind=NumericsKind.BITWISE)
    return Numerics(kind=NumericsKind.TOLERANCE, tolerances=MEGA_TOLERANCES)


NUMERICS = _numerics()


def text_config(hf_config):
    return getattr(hf_config, "text_config", None) or hf_config


def scaled_rope(config) -> bool:
    """Whether the config scales RoPE (transformers may report the default type)."""
    scaling = getattr(config, "rope_scaling", None) or {}
    return scaling.get("rope_type", scaling.get("type", "default")) != "default"


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:  # a dotted name whose parent package is absent
        return False


def toolchain_reasons(slots: frozenset = SLOTS) -> list[str]:
    """The requested slots' toolchains, by import spec only: slot mega's kernels build
    with the CUDA Python bindings and nvcc 13.4 or newer; slot markov's chain with the
    bindings and any nvcc, its noise kernel with Triton; slot moe_nofill launches
    FlashInfer's CuTe-DSL quantize (FlashInfer, CuTe DSL)."""
    reasons = []
    if slots & {"mega", "markov"}:
        if not _has_module("cuda.bindings"):
            reasons.append("needs the CUDA Python bindings")
        else:
            from sgl_nvidia_mega.minimax_m31.sm103 import nvcc

            # the QMUL4 floor is slot mega's
            reason = nvcc.why_not(nvcc.QMUL4_VERSION if "mega" in slots else None)
            reasons += [reason] if reason else []
    if "markov" in slots and not _has_module("triton"):
        reasons.append("slot markov needs Triton")
    if "moe_nofill" in slots:
        reasons += [
            f"slot moe_nofill needs {what}"
            for module, what in (
                ("flashinfer", "FlashInfer"),
                ("cutlass", "CuTe DSL (nvidia-cutlass-dsl)"),
            )
            if not _has_module(module)
        ]
    return reasons


# Slot moe_nofill launches FlashInfer's per-token NVFP4 kernel through these names of
# flashinfer/quantization/kernels/nvfp4_quantize.py (FlashInfer 0.7.4), read here from
# the installed source without importing it; moe_nofill.config_reasons checks them again
# at build, on the imported module, with the kernel getter's parameters
FLASHINFER_NAMES = (
    "_get_compiled_kernel_nvfp4_per_token",
    "SF_LAYOUT_LINEAR",
    "_env_flag_enabled",
    "resolve_nvfp4_4over6",
    "_per_token_cta_threads",
    "get_device_sm_count",
    "_MAX_THREADS_PER_BLOCK",
)
FLASHINFER_QUANTIZE = "quantization/kernels/nvfp4_quantize.py"


def _top_level_names(path: Path) -> frozenset:
    """The names a module defines, assigns or imports at its top level (ast)."""
    names = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(t.id for t in targets if isinstance(t, ast.Name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
    return frozenset(names)


def flashinfer_api_reasons(slots: frozenset = SLOTS) -> list[str]:
    """The FlashInfer names slot moe_nofill launches through ([] when it is not
    requested, or FlashInfer is absent, which ``toolchain_reasons`` names)."""
    found = (
        importlib.util.find_spec("flashinfer")
        if "moe_nofill" in slots and _has_module("flashinfer")
        else None
    )
    if found is None or found.origin is None:
        return []
    path = Path(found.origin).parent / FLASHINFER_QUANTIZE
    provided = _top_level_names(path) if path.is_file() else frozenset()
    return [
        f"FlashInfer's {FLASHINFER_QUANTIZE} has no {name} (slot moe_nofill)"
        for name in FLASHINFER_NAMES
        if name not in provided
    ]


def draft_config_reasons(cfg, slots: frozenset = SLOTS) -> list[str]:
    """Why the requested draft slots would not run on this server ([] when they would,
    or none is requested): they replace SGLang's folded DSpark proposal, the tail of the
    DSpark draft's CUDA graph (``REQUIREMENTS`` asks for DSpark and, outside the parity
    switch, for decode CUDA graphs), so a server without it would run SGLang's proposal
    under their names."""
    requested = sorted(draft_slots(slots))
    if not requested:
        return []
    from sglang.srt.environ import envs

    reasons = []
    if not envs.SGLANG_DSPARK_FOLDED_PROPOSAL.get():
        reasons.append(
            f"slots {requested} replace the folded DSpark proposal; "
            "SGLANG_DSPARK_FOLDED_PROPOSAL is off"
        )
    if envs.SGLANG_DSPARK_OPT_INT8_MARKOV_WALK.get():
        reasons.append(
            "slot markov replaces the draft's Markov walk, and "
            "SGLANG_DSPARK_OPT_INT8_MARKOV_WALK asks for SGLang's int8 walk: unset one"
        )
    return reasons


# The package mirrors SGLang's MiniMax-M3.1 step code and reads its DSpark switches as
# of sgl-project/sglang#46693 (the model's layer boundary and expert requantization),
# #42855 and #49595 (the DSpark switches the draft's slots refuse). On an older SGLang
# the spec still imports and refuses every configuration; the newest of the three, the
# --speculative-dspark-lm-head-dtype argument, is the marker.
NO_M31_HOST = (
    "this SGLang predates the MiniMax-M3.1 support the megakernel is built on "
    "(sgl-project/sglang#46693, #42855, #49595)"
)


def host_reasons(cfg) -> list[str]:
    """Why this SGLang cannot host the package ([] when it can)."""
    return [] if hasattr(cfg, "speculative_dspark_lm_head_dtype") else [NO_M31_HOST]


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    """The checks ``REQUIREMENTS`` cannot state: the hosting SGLang, the requested
    slots, nested HF fields, derived values and the toolchains."""
    host = host_reasons(cfg)
    if host:
        return host
    reasons = (
        slot_reasons(SLOTS)
        + toolchain_reasons(SLOTS)
        + flashinfer_api_reasons(SLOTS)
        + draft_config_reasons(cfg, SLOTS)
    )
    if cfg.quantization not in (None, "minimax_m3_nvfp4"):
        reasons.append(
            f"--quantization {cfg.quantization!r}: the kernels read the minimax_m3_nvfp4 "
            "checkpoint (MXFP8 attention, NVFP4 experts)"
        )
    if (cfg.speculative_eagle_topk or 1) > 1:
        reasons.append("tree verify (--speculative-eagle-topk > 1) is not supported")
    text = text_config(hf_config)
    moe = getattr(text, "moe_layer_freq", None) or []
    if not any(moe):
        reasons.append("config.moe_layer_freq has no MoE layer")
    if "mega" in SLOTS:
        reasons += mega_config_reasons(cfg, text)
    return reasons


def mega_config_reasons(cfg, text) -> list[str]:
    """Slot mega's checks: the config fields its kernels are built for, the context
    their page tables hold and the persistent grid's residency."""
    from sglang.srt.environ import envs

    reasons = []
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
    if any(moe) and (
        len(freq) != len(moe) or not all(f for f, m in zip(freq, moe) if m)
    ):
        reasons.append("every MoE layer must be an MSA sparse attention layer")
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

    return runtime.build(model, hf_config, ctx)


SPEC = MegakernelSpec(
    name="MiniMax-M3.1",
    requires=REQUIREMENTS,
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    numerics=NUMERICS,
    max_rows=MAX_ROWS,
    build=_build,
    api_version=API_VERSION,
    model_disable_reasons=_model_disable_reasons,
    config_disable_reasons=_config_disable_reasons,
)
