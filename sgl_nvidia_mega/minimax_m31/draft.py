# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax-M3.1's DSpark draft: the megakernel draft spec. Its covered step, one
request's seven-row draft block on the draft backbone, is ``draft_step``.

With slot ``markov`` the built object also offers SGLang's folded proposal a
``draft_sampler`` (``sampler``): the draft LM head and the whole Markov chain of each
draft block at 1-16 requests, the chain in one kernel per rank. Without ``markov`` SGLang's
proposal runs. The draft's slots are requested in the target's list
(``runtime.SLOTS``); ``EngagementCheck`` stops a server that requested them and whose
proposal never asked for the sampler.

SGLang checks a draft's spec only at load, on the loaded draft model; this module stays
light."""

from __future__ import annotations

from sglang.kernels.plugin_kernels.nvidia.spec import (
    Feature,
    MegakernelSpec,
    Numerics,
    NumericsKind,
    Requirements,
)

from sgl_nvidia_mega.minimax_m31 import DRAFT_ROWS
from sgl_nvidia_mega.minimax_m31 import spec as target
from sgl_nvidia_mega.minimax_m31.runtime import SLOTS, draft_tolerances

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py": (
        "Feature",
        "MegakernelSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
    ),
    # EngagementCheck: SGLang's graph-capture flag
    "srt/model_executor/runner/__init__.py": ("get_is_capture_mode",),
    # the decode graphs the folded proposal is the tail of
    "srt/model_executor/cuda_graph_config.py": (
        "Backend",
        "Phase",
        "check_cuda_graph_backend",
        "DECODE",
        "DISABLED",
    ),
}

# The block's final-normed rows against SGLang's draft forward on the same eager step
# (SGLANG_NVIDIA_MEGAKERNEL_PARITY=1, the target's workload: 866 blocks), measured with
# about 15% headroom: MXFP8 activations requantized before every GEMM, FP32 accumulation
# and rank-ordered TP4 sums. Per block: the largest |difference| over the largest |value|
# at the median and 99th percentile (measured 0.045, 0.115), and the block's smallest row
# cosine at the 1st percentile, a lower bound (0.9933)
BACKBONE_TOLERANCES = {
    "draft.hidden.rel.p50": 0.052,
    "draft.hidden.rel.p99": 0.13,
    "draft.hidden.cosine.p1": 0.99,
}


def _numerics(slots: frozenset = SLOTS) -> Numerics:
    """The backbone's tolerances and the requested draft slots' against SGLang's
    proposal (``runtime.DRAFT_TOLERANCES``)."""
    return Numerics(
        kind=NumericsKind.TOLERANCE,
        tolerances={**BACKBONE_TOLERANCES, **draft_tolerances(slots)},
    )


NUMERICS = _numerics()

# How often this process's draft specs were built and their samplers asked for
# (EngagementCheck reads it)
_COUNTS = {"built": 0, "sampler": 0}


def proposal_off_reason() -> str:
    """Why this server has no folded proposal for the draft slots to replace, or "".
    Only the eager test mode has none: the spec's requirements admit disabled decode
    CUDA graphs under the parity switch alone (``spec.REQUIREMENTS``), and SGLang folds
    the proposal into the draft's decode graph."""
    from sglang.srt.model_executor.cuda_graph_config import (
        Backend,
        Phase,
        check_cuda_graph_backend,
    )

    if check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED):
        return (
            "decode CUDA graphs are disabled (eager steps under "
            f"{target.PARITY_ENV}): SGLang's eager proposal runs"
        )
    return ""


class EngagementCheck:
    """The draft slots' engagement, checked once from the target's executor (its plan
    calls it on every forward until it returns True). SGLang captures the target's decode
    graphs, then the DSpark draft's, whose folded proposal asks the draft spec for its
    sampler; the first forward outside graph capture after the target's capture is past
    both. A server whose proposal never asked (the draft graph left out for lack of
    memory, or no draft spec built) would serve SGLang's proposal under the slots'
    names: it stops there instead."""

    def __init__(self, slots):
        self.slots = sorted(slots)
        self.captured = False

    def __call__(self) -> bool:
        from sglang.srt.model_executor.runner import get_is_capture_mode

        if get_is_capture_mode():
            self.captured = True
            return False
        if not self.captured:
            return False
        if not _COUNTS["sampler"]:
            why = (
                "no DSpark draft spec was built"
                if not _COUNTS["built"]
                else "SGLang's folded proposal never asked for the draft sampler (no "
                "DSpark draft CUDA graph?)"
            )
            raise RuntimeError(
                f"NVIDIA megakernel: draft slots {self.slots} are requested, but {why}; "
                "SGLang's proposal would serve under their names"
            )
        return True


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
