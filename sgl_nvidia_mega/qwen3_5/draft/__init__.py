# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Qwen3.5's MTP draft on SM103 (GB300): the megakernel draft spec and its built
object.

The draft's forward stays SGLang's (the built object covers no step width). Requested
in ``SGLANG_NVIDIA_MEGAKERNEL_SLOTS``, the built object offers SGLang's EAGLE draft
entries for one request (bs 1), which run inside SGLang's draft graphs:

- ``draft_loop`` (with ``begin_draft_round``): the whole draft loop of a served draft
  graph (``loop``);
- ``draft_extend``: the model step and top-1 of a served draft-extend graph
  (``extend``).

Inside them every op is SGLang's own unless a draft slot replaces it: ``draft_moe``
(the single-launch MoE), ``draft_q1`` (the q1 split-KV decode), ``draft_xchg`` (the
step boundary's fc split, exchange and head), ``draft_hop`` (the draft hops and the
o_proj GEMV), ``draft_front`` (the attention front writing the FP8 KV itself) and
``draft_skv2`` (the draft extend's split-KV decode, on the attention package's SKV2).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

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

from sgl_nvidia_mega.qwen3_5 import spec as target

logger = logging.getLogger(__name__)

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
        '"draft_loop"',
        '"draft_extend"',
        '"begin_draft_round"',
        "binding",
        "deadline_s",
    ),
    # where SGLang calls the entries (their keyword arguments are pinned by the GPU tests)
    "srt/speculative/eagle_megakernel.py": (
        "_draft_loop",
        "_draft_extend",
        "_begin_draft_round",
        "bind_draft_graphs",
        "bind_extend_graphs",
        "begin_draft_round",
    ),
}

ARCHITECTURE = "Qwen3_5ForCausalLMMTP"


def numerics(slots: frozenset) -> Numerics:
    """SGLang's draft bit for bit with no draft slot; with one, 2.2's draft kernels,
    which differ from SGLang's ops only in fp32 summation order (q1 split-KV, the
    o_proj / fc / MoE GEMVs, the draft head), so a draft token can differ where two
    logits are within a bf16 rounding."""
    if not slots & {
        "draft_moe",
        "draft_q1",
        "draft_xchg",
        "draft_hop",
        "draft_front",
        "draft_skv2",
    }:
        return target.NUMERICS
    return Numerics(NumericsKind.TOLERANCE, {"draft_logits.rel": 2**-7})


# the one request a draft entry serves: 2.2's draft kernels are single-row
SERVED_BS = 1
# the requests a draft_loop step serves (two only where plan.multi_request): two run the
# draft steps on 2.2's two-row kernels (the split fc, the hop's all-reduce points, the
# two-token MoE), the rest SGLang's ops
LOOP_BS = (1, 2)


class MtpDraft:
    """The draft spec's built object: SGLang's forward on every step, and the EAGLE
    draft entries requested."""

    widths: tuple[int, ...] = ()
    draft_sampler = None  # SGLang's draft proposal

    def __init__(
        self,
        lm,
        *,
        binding,
        deadline_s: float,
        slots: frozenset[str],
        context_len=None,
    ):
        self.lm = lm
        # the draft's ModelConfig.context_len (trtllm-gen's max_seq_len)
        self.context_len = context_len
        self.binding = binding  # SGLang's EagleBinding (BuildContext.binding)
        self.deadline_s = deadline_s
        self.slots = slots
        self.parts = None  # draft.parts.Parts, built by the first factory call
        self._loop = None
        # the entries offered (spec.ENTRIES); None leaves SGLang's own code
        self.draft_loop: Optional[Callable] = (
            self._draft_loop if "draft_loop" in slots else None
        )
        self.begin_draft_round: Optional[Callable] = (
            self._begin_draft_round if "draft_loop" in slots else None
        )
        self.draft_extend: Optional[Callable] = (
            self._draft_extend if "draft_extend" in slots else None
        )
        # the KV-length graph variants by runner kind, for an SGLang that captures them
        # (2.2 routes the draft steps' q1 decode by KV length)
        self.graph_variants = None
        if "draft_loop" in slots and "draft_q1" in slots:
            from sgl_nvidia_mega.qwen3_5.draft.attention import Q1Variants

            self.graph_variants = {"draft_loop": Q1Variants()}
        if "draft_extend" in slots and "draft_skv2" in slots:
            from sgl_nvidia_mega.qwen3_5.attention import route

            self.graph_variants = {
                **(self.graph_variants or {}),
                "draft_extend": route.EXTEND,
            }

    def summary(self) -> str:
        entries = [
            name
            for name in ("draft_loop", "draft_extend")
            if getattr(self, name) is not None
        ]
        return f"Qwen3.5 MTP draft: entries {entries or 'none'}"

    def plan(self, input_ids, forward_batch, *args, **kwargs):
        return None

    def forward(self, step, original, *args, **kwargs):
        raise AssertionError("the MTP draft covers no step width")

    def _parts(self):
        """What the entries share, built once the draft's weights are final: the
        factories run before SGLang captures, after the EAGLE worker gave the draft the
        target's embedding and LM head."""
        if self.parts is None:
            from sgl_nvidia_mega.qwen3_5.draft.parts import Parts

            self.parts = Parts.build(self)
        return self.parts

    def _draft_loop(self, *, num_steps: int, max_bs: int, draft_tokens):
        from sgl_nvidia_mega.qwen3_5.draft.loop import DraftLoop

        self._loop = DraftLoop(self._parts(), num_steps=num_steps)
        return self._loop.step_for

    def _begin_draft_round(self, io) -> None:
        if self._loop is not None:
            self._loop.begin_round(io)

    def _draft_extend(
        self, *, tokens_per_request: int, max_bs: int, topk_index, topk_p, hidden
    ):
        from sgl_nvidia_mega.qwen3_5.draft.extend import DraftExtend

        extend = DraftExtend(self._parts(), tokens_per_request=tokens_per_request)
        return extend.step_for


def _requested() -> frozenset:
    """The draft's entries and slots in SGLANG_NVIDIA_MEGAKERNEL_SLOTS."""
    from sgl_nvidia_mega.qwen3_5 import runtime

    return runtime.SLOTS & (set(runtime.DRAFT_ENTRIES) | set(runtime.DRAFT_SLOTS))


def _build(model, hf_config, ctx) -> MtpDraft:
    return MtpDraft(
        model,
        binding=ctx.binding,
        deadline_s=ctx.deadline_s,
        slots=frozenset(_requested()),
        context_len=getattr(hf_config, "context_len", None),
    )


SPEC = MegakernelSpec(
    name="Qwen3.5 MTP draft",
    # the target's devices, TP size, dtype and decode graphs: the draft shares its GPUs
    requires=Requirements(
        isas=target.REQUIREMENTS.isas,
        tp_sizes=target.REQUIREMENTS.tp_sizes,
        dtypes=target.REQUIREMENTS.dtypes,
        decode_graph_backends=target.REQUIREMENTS.decode_graph_backends,
    ),
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    numerics=numerics(_requested()),
    # the draft's forward stays SGLang's: no width is built
    max_rows=0,
    build=_build,
    api_version=target.API_VERSION,
)
