# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The Muse Spark Open megakernel's spec: the setups it serves and its build.

SGLang checks the requirements at argument resolution and the model at load
(``sglang.kernels.plugin_kernels.nvidia``); an unsupported setup stops the
server on every rank, listing every reason. This module stays light: SGLang
imports it in the launcher.

Scope today is the **target-verify** step. The plugin this is ported from also
carries nine prefill/extend kernels, which cannot be registered here until
SGLang's binding dispatches prefill steps; see ``PORT-NOTES.md``.
"""

from __future__ import annotations

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

from sgl_nvidia_mega.muse_spark import HF_CONFIG
from sgl_nvidia_mega.muse_spark.plan import MAX_ROWS

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": (
        "Feature",
        "MegakernelSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
    ),
    "srt/model_executor/runner/__init__.py": ("get_is_capture_mode",),
}


# the SGLang megakernel contract this spec is written against
API_VERSION = 2

EAGER_TEST_SWITCHES = ("SGLANG_NVIDIA_MEGAKERNEL_PARITY",)


def eager_steps_allowed() -> bool:
    return any(os.environ.get(k, "") not in ("", "0") for k in EAGER_TEST_SWITCHES)


def decode_graph_backends() -> frozenset:
    return frozenset({"full", "disabled"} if eager_steps_allowed() else {"full"})


def serves_this_step() -> bool:
    """A covered step runs the megakernel only while SGLang captures its decode
    CUDA graphs (their warm-ups included), or eagerly under a test-only switch."""
    from sglang.srt.model_executor.runner import get_is_capture_mode

    return get_is_capture_mode() or eager_steps_allowed()


# Every value below is the one the accepted rows ran with (P108 at c96, and
# P106 / P102 / P100 / P52 before it at c112 / c96 / c80). A setup that differs is refused rather than served by
# kernels that were never measured on it.
#
# This line serves GB300 (sm_103) only. The kernels come from the sm_107 (Rubin) line, where the rows above were
# measured; PORT-NOTES-sm103.md says what was changed and checked for sm_103 and which per-device tuning tables
# still have no GB300 entry.
ISAS = frozenset({"sm_103"})

REQUIREMENTS = Requirements(
    isas=ISAS,
    tp_sizes=frozenset({4}),
    dtypes=frozenset({"bfloat16"}),
    decode_graph_backends=decode_graph_backends(),
    server_args={
        # NVFP4 on the full-attention layers, FP8 on the sliding-window ones;
        # the verify kernels read the packed pool directly
        "kv_cache_dtype": frozenset({"nvfp4"}),
        "quantization": frozenset({"modelopt_fp4"}),
        # uniform verify steps of batch x draft block
        "speculative_algorithm": frozenset({"DFLASH"}),
        "moe_runner_backend": frozenset({"flashinfer_trtllm"}),
        "moe_a2a_backend": frozenset({"flashinfer"}),
        "attention_backend": frozenset({"trtllm_mha"}),
        # the boundary reduction the ar_attn_boundary slot replaces
        "disable_attn_tp_gather": frozenset({True}),
        # the kernels read weights in place, after SGLang's last conversion
        "startup_weight_load_mode": frozenset({"serial"}),
    },
    hf_config=HF_CONFIG,
)

# The slots replace a norm, an attention call or a boundary reduction with a
# fused form of the same arithmetic in the same order; none is bitwise with the
# stock path, so the declared agreement is a tolerance.
NUMERICS = Numerics(
    kind=NumericsKind.TOLERANCE,
    tolerances={
        # End to end, measured on a GB300 TP4 (SGLang 306ccf6, DFLASH, real weights): GSM8K 5-shot, 1319 questions,
        # greedy, 0.953 with the components against 0.951 stock. The bound is about one standard error of the
        # difference of two such runs at this accuracy (sqrt(2 * 0.95 * 0.05 / 1319) ~= 0.0085).
        # TODO: per-slot bounds from an A/B of each slot against the stock forward on the same GPU.
        "gsm8k_5shot.accuracy.abs": 0.01,
    },
)


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    from sgl_nvidia_mega.muse_spark import runtime

    reasons = runtime.slot_reasons()
    if cfg.cuda_graph_config.decode.backend == "disabled" and not eager_steps_allowed():
        reasons.append(
            "decode CUDA graphs are disabled: this spec covers the target-verify "
            "step as it is captured; enable decode CUDA graphs"
        )
    return reasons


def _model_disable_reasons(model, hf_config) -> list[str]:
    from sgl_nvidia_mega.muse_spark import runtime

    return runtime.model_disable_reasons(model)


def _build(model, hf_config, ctx):
    from sgl_nvidia_mega.muse_spark import runtime

    return runtime.build(model, ctx)


SPEC = MegakernelSpec(
    name="Muse Spark Open",
    requires=REQUIREMENTS,
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    numerics=NUMERICS,
    max_rows=MAX_ROWS,
    build=_build,
    api_version=API_VERSION,
    model_disable_reasons=_model_disable_reasons,
    config_disable_reasons=_config_disable_reasons,
)
