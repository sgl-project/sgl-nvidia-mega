# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax-M3.1's DSpark draft: the megakernel draft spec. Its covered step, one
request's seven-row draft block on the draft backbone, is ``draft_step``.

SGLang checks a draft's spec only at load, on the loaded draft model; this module stays
light."""

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

from sgl_nvidia_mega.minimax_m31 import DRAFT_ROWS
from sgl_nvidia_mega.minimax_m31 import spec as target

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": (
        "Feature",
        "MegakernelSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
    ),
}

# The block's final-normed rows against SGLang's draft forward on the same eager step
# (SGLANG_NVIDIA_MEGAKERNEL_PARITY=1, the target's workload: 866 blocks), measured with
# about 15% headroom: MXFP8 activations requantized before every GEMM, FP32 accumulation
# and rank-ordered TP4 sums. Per block: the largest |difference| over the largest |value|
# at the median and 99th percentile (measured 0.045, 0.115), and the block's smallest row
# cosine at the 1st percentile, a lower bound (0.9933)
NUMERICS = Numerics(
    kind=NumericsKind.TOLERANCE,
    tolerances={
        "draft.hidden.rel.p50": 0.052,
        "draft.hidden.rel.p99": 0.13,
        "draft.hidden.cosine.p1": 0.99,
    },
)


def _model_disable_reasons(model, hf_config) -> list[str]:
    from sgl_nvidia_mega.minimax_m31 import draft_step

    return draft_step.model_disable_reasons(model, hf_config)


def _build(model, hf_config, ctx):
    from sgl_nvidia_mega.minimax_m31 import draft_step

    return draft_step.DraftStep(model, hf_config, ctx)


SPEC = MegakernelSpec(
    name="MiniMax-M3.1 DSpark draft",
    # the target's device, TP size, dtype and decode graphs: the draft shares its GPUs
    requires=Requirements(
        isas=target.SPEC.requires.isas,
        tp_sizes=target.SPEC.requires.tp_sizes,
        dtypes=target.SPEC.requires.dtypes,
        decode_graph_backends=target.SPEC.requires.decode_graph_backends,
    ),
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    numerics=NUMERICS,
    max_rows=DRAFT_ROWS,
    build=_build,
    api_version=target.API_VERSION,
    model_disable_reasons=_model_disable_reasons,
)
