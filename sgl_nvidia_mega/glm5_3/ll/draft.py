# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""P6a (``ll.draft_eh``): the GLM-5 NextN draft's ``eh_proj``, row-parallel over the TP group (TP4).

``eh_proj`` maps ``[enorm(embed) | hnorm(hidden)]`` (2H) to H. Rank r keeps the weight
columns ``[r * 2H / tp, (r + 1) * 2H / tp)`` (a K shard, sliced once at build: 37.7 MB per
rank at TP4 instead of the replicated 151 MB), multiplies them with the same columns of
the draft front's output (a strided view: the fused front already wrote both halves, and
its columns are bitwise ``fused_eh_norm``'s), and the TP group sums the partials. At H =
6144 a rank's window lies inside one half (TP4: ranks 0-1 the embedding half, 2-3 the
hidden half).

The sum: when the target serves its TP4 hops (``ll.hops``) and a step group runs the draft
attention (``prepared``: it takes the normed input and sets the attention inputs itself),
the target's LL hop sums the partials and applies the draft layer's input RMSNorm in one
launch (its residual-free ``allreduce_norm``), and the draft layer skips its attention
prepare (``draft_attention.PreparedInput``); otherwise (no LL hops, or the stock draft
attention, whose inputs the prepare sets) SGLang's TP all-reduce of the BF16 partials,
then the stock input norm. Numerics differ from the replicated GEMM at the BF16
level, as every row-parallel projection's do: each partial rounds to BF16 before the sum
(``TOLERANCES``).

E-D1 (GB300, TP4, per draft forward, CUDA-graph replay, L2 flushed): stock replicated GEMM
about 40 us at 1-48 rows; the TP8 draft's column shard + SGLang's all-gather 34 us (1 row) to 67 us
(15 rows); this unit 21.5-25.7 us.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sgl_nvidia_mega.glm5_3.registry import Candidate

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/distributed/communication_op.py": ("tensor_model_parallel_all_reduce",),
    "srt/distributed/parallel_state.py": (
        "get_tp_group",
        "rank_in_group",
        "world_size",
    ),
    "srt/models/deepseek_nextn.py": ("eh_proj", "decoder", "embed_tokens"),
    "srt/models/deepseek_v2.py": ("input_layernorm",),
    "srt/layers/layernorm.py": ("weight", "variance_epsilon"),
}

logger = logging.getLogger(__name__)

HIDDEN = 6144
# the error-record site of the sum in the target's LL hop (>= ll.hops.EXTERNAL_SITES)
SITE_EH_SUM = 1004
# the TP sizes served: one tray, so the partial sum is a one-tray all-reduce (TP8 spans
# two trays and keeps the TP8 draft's column shard on the target's hop)
TP_SIZES = (4,)

# test/kernels/test_glm5_3_draft_eh_proj.py: the served sum is the FP32 sum of the
# ranks' BF16 partials up to the all-reduce's rounding, and so within this share of each
# row's RMS of FP64 (each partial rounds to BF16 before the sum)
TOLERANCES = {
    "ll.draft_eh.sum_max_ulp": 1.0,
    "ll.draft_eh.max_rel_rms": 2.0**-5,
}


class RowParallelEhProj:
    """One rank's K shard of eh_proj and its forward (any row count)."""

    def __init__(self, weight: torch.Tensor, rank: int, world: int) -> None:
        if weight.dtype != torch.bfloat16 or tuple(weight.shape) != (
            HIDDEN,
            2 * HIDDEN,
        ):
            raise ValueError(
                f"row-parallel eh_proj: weight {tuple(weight.shape)} {weight.dtype} is not GLM-5's"
            )
        width = window_width(world)
        if width is None:
            raise ValueError(f"row-parallel eh_proj: no window layout at TP{world}")
        self.rank, self.world, self.width = rank, world, width
        self.col = rank * width
        self.weight = weight[:, self.col : self.col + width].contiguous()

    def partial(self, eh_input: torch.Tensor) -> torch.Tensor:
        """This rank's ``[rows, H]`` contribution: its K window of ``eh_input`` (``[rows, 2H]``) times its shard."""
        return torch.nn.functional.linear(
            eh_input[:, self.col : self.col + self.width], self.weight
        )

    def __call__(self, eh_input: torch.Tensor) -> torch.Tensor:
        from sglang.srt.distributed import tensor_model_parallel_all_reduce

        return tensor_model_parallel_all_reduce(self.partial(eh_input))


def window_width(world: int) -> Optional[int]:
    """The K window per rank, or None when a rank's window would straddle the two halves
    (or not be 16-byte aligned)."""
    if world < 2 or (2 * HIDDEN) % world:
        return None
    width = 2 * HIDDEN // world
    if HIDDEN % width or width % 8:
        return None
    return width


class RowParallelEhFamily:
    """The ``ll.draft_eh`` family: the draft_eh step group of every width (one shard)."""

    def __init__(self, model, derived, profile) -> None:
        self.model, self.derived, self.profile = model, derived, profile
        self.rp: Optional[RowParallelEhProj] = None
        # the target's LL hop family when it sums the partials with the input norm
        self.hop = None
        self.hop_rows = frozenset()

    def compile(self, widths) -> None:
        from sglang.srt.distributed.parallel_state import get_tp_group

        tp = get_tp_group()
        self.rp = RowParallelEhProj(
            self.model.model.eh_proj.weight, tp.rank_in_group, tp.world_size
        )
        self.derived.add("draft.eh_proj_row_parallel", self.rp.weight)
        hop = target_hop_family(self.model)
        why = "the target's LL hops are not served" if hop is None else None
        if hop is not None:
            norm = self.model.model.decoder.input_layernorm
            if float(hop.eps) != float(norm.variance_epsilon):
                why = f"the hop's eps {hop.eps} is not the draft input norm's {norm.variance_epsilon}"
            else:
                self.hop, self.gamma = hop, norm.weight
                self.hop_rows = frozenset(r for r in widths if hop.reason(r) is None)
                hop.register_sites({SITE_EH_SUM: "draft eh_proj sum (P6a)"})
        if tp.rank_in_group == 0:
            logger.info(
                "ll.draft_eh: row-parallel eh_proj (TP%d, %d input columns per rank); the sum %s",
                tp.world_size,
                self.rp.width,
                f"with the input norm in the target's LL hop at rows {sorted(self.hop_rows)}"
                if self.hop is not None
                else f"by SGLang's all-reduce ({why})",
            )

    def provider(self, candidate, rows: int, mode: str) -> RowParallelEhFamily:
        return self

    def eh_proj(self, eh_input: torch.Tensor, *, prepared: bool = False):
        """The draft_eh step at any row count: the summed projection, or, when the caller's
        draft attention is a step group (``prepared``), in the target's LL hop the summed
        projection through the draft layer's input norm (``draft_attention.PreparedInput``)."""
        if not eh_input.is_contiguous():
            return None
        if not prepared or self.hop is None or eh_input.shape[0] not in self.hop_rows:
            return self.rp(eh_input)
        from sgl_nvidia_mega.glm5_3.draft_attention import PreparedInput

        norm, summed = self.hop.allreduce_norm(
            self.rp.partial(eh_input), None, self.gamma, site=SITE_EH_SUM
        )
        return PreparedInput(norm, summed)


def target_hop_family(nextn):
    """The built target's TP4 LL hop family (``ll.hops``) when its hops are served on
    every rank (they agreed on the target's composition), else None: registered by the
    target's build under the embedding table it shares with the draft
    (``runtime.register_target_hops``)."""
    from sgl_nvidia_mega.glm5_3 import runtime

    table = getattr(getattr(nextn.model, "embed_tokens", None), "weight", None)
    family = None if table is None else runtime.target_hops(table)
    if family is None or getattr(family, "buffer", None) is None:
        return None
    if not hasattr(family, "allreduce_norm"):
        return None
    return family


def _covers(profile, rows: int, mode: str) -> Optional[str]:
    if profile.tp not in TP_SIZES:
        return (
            f"row-parallel eh_proj serves TP {TP_SIZES} (one tray), not TP{profile.tp}"
        )
    if window_width(profile.tp) is None:
        return f"no K window layout at TP{profile.tp}"
    return None


CANDIDATES = (
    Candidate("ll.draft_eh", "draft_eh", ("eh_proj",), "ll.draft_eh", _covers),
)
FAMILIES = {"ll.draft_eh": RowParallelEhFamily}
