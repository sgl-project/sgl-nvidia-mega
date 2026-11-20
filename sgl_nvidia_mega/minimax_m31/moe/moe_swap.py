# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/minimax_m3.py (MiniMaxM3MoE.forward_normal).
"""Slot ``moe_swap`` (bitwise; needs moe_nofill): an MoE layer's routed chain on the
main stream.

Stock (and slot moe_nofill), a captured step forks every MoE layer
(``MiniMaxM3MoE.forward_normal``): after an event on the FFN-input boundary kernel the
routed chain (router GEMM, top-4, FlashInfer's routed MoE: routing, GEMM1, the FC2
quantize, GEMM2) runs on ``mlp.alt_stream`` and the shared expert on the current
stream, where its gate_up is the boundary kernel's PDL successor; the current stream
then waits for the alternate one, and the next layer's attention-input boundary kernel
takes both outputs. The routed chain is the critical one in every MoE layer (the shared
expert finishes 8-30 us earlier), yet its router follows the boundary kernel through the
fork's cross-stream edge: it starts about 0.8 us after that kernel ends, against about
0.3 us for the boundary's same-stream successor (GB300 nsys, jobs 1288415 / 1288416).

Slot moe_swap: the same calls with the same arguments, every kernel the same, on
swapped streams: after the same fork the routed chain runs on the current stream
(captured first, so its router GEMM is the boundary kernel's first PDL dependent) and
the shared expert on ``mlp.alt_stream``; then the same join. Eagerly (the parity
check), and wherever SGLang runs the MoE on one stream, the routed chain runs first,
then the shared expert. Every kernel reads what it reads in SGLang's order and
computes what it computes there, so the layer's handoff is SGLang's bit for bit
(``SGLANG_MIRRORS`` pins the SGLang code it follows).

Buffers and streams. Every kernel of the step waits for the kernels it depends on
before it reads what another kernel of the step writes or writes anything (one
launched with PDL in ``griddepcontrol.wait``, any other at its launch). A captured
graph keeps the fork and the join as dependencies of the next kernel on the waiting
stream. With the streams swapped:

- slot moe_nofill's quantize buffers are written and read on the current stream within
  the routed chain, and rewritten only by the next layer's chain, after its waits;
- the routed chain's outputs (router logits, top-k, FlashInfer's deferred triple) are
  allocated and read on the current stream, as SGLang's are on the alternate one; the
  shared expert's output is allocated on the alternate stream and read by the next
  attention-input boundary kernel after the join, as SGLang's routed triple is;
  PyTorch's allocator reuses a block only on the stream that allocated it (no
  record_stream), and the next alternate-stream kernel that could write it, after the
  next layer's fork, writes only after its wait, by which the attention-input boundary
  that read the output has completed;
- the FFN input's bf16 rows are freed when the MoE step returns and reused by
  current-stream kernels after the join, each of which writes only after its wait.

Decode graphs of different widths, which share every buffer above, never run at the
same time.
"""

from __future__ import annotations

import logging
from typing import Optional

import msgspec
import torch

from sgl_nvidia_mega.minimax_m31.moe import moe_nofill

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    # read through the model module's namespace, as MiniMaxM3MoE.forward_normal does
    "srt/models/minimax_m3.py": (
        "MoeDeferredFinalize",
        "get_is_capture_mode",
        "is_in_breakable_cuda_graph",
        "alt_stream",
        "shared_experts",
        "num_fused_shared_experts",
    ),
    "srt/layers/moe/utils.py": ("sum_post_experts_output",),
    "srt/layers/layer_boundary/fusions/*.py": (
        "from_flashinfer",
        "from_flashinfer(gated_shared_output)",
        "from_flashinfer(m)",
        "from_flashinfer(reduce)",
    ),
}

# The SGLang function whose body this module mirrors, by the digest of its code
# (test/unit/test_contract.py recomputes it on the installed SGLang)
SGLANG_MIRRORS = {
    "srt/models/minimax_m3.py:MiniMaxM3MoE.forward_normal": "1a64464f8e449287",
}

logger = logging.getLogger(__name__)

SLOT = "moe_swap"


def forward(mlp, x: torch.Tensor, quant):
    """``moe_nofill.forward`` with the routed chain on the current stream: while
    capturing (SGLang's condition) the routed experts on the current stream and the
    shared expert on ``mlp.alt_stream`` after the fork, otherwise the routed chain then
    the shared expert on one stream; the same deferred-finalize handoff."""
    from sglang.srt.layers.moe.utils import sum_post_experts_output
    from sglang.srt.models.minimax_m3 import (
        MoeDeferredFinalize,
        get_is_capture_mode,
        is_in_breakable_cuda_graph,
    )

    if (
        mlp.alt_stream is not None
        and mlp.shared_experts is not None
        and get_is_capture_mode()
        and not is_in_breakable_cuda_graph()
    ):
        current_stream = torch.cuda.current_stream()
        # the fork first: the alternate stream waits for the boundary kernel alone
        mlp.alt_stream.wait_stream(current_stream)
        # captured before the shared expert: the router GEMM is the
        # boundary kernel's first dependent, and its successor on this stream
        routed = moe_nofill._router_experts(mlp, x, quant)
        with torch.cuda.stream(mlp.alt_stream):
            shared_output = mlp._forward_shared_experts(x)
        current_stream.wait_stream(mlp.alt_stream)
    else:
        routed = moe_nofill._router_experts(mlp, x, quant)
        shared_output = mlp._forward_shared_experts(x)
    return MoeDeferredFinalize.from_flashinfer(
        routed,
        gated_shared_output=shared_output,
        m=x.shape[0],
        reduce=sum_post_experts_output,
    )


# --------------------------------------------------------------------- builds


class Bound(msgspec.Struct):
    """The build's result: the widths the slot serves (every width moe_nofill serves, or
    none) and why it serves none ("" when it serves)."""

    widths: tuple = ()
    reason: str = ""

    def serves(self, rows: int) -> bool:
        return rows in self.widths


def two_stream_reason(mlp) -> str:
    """Why SGLang's MoE ``mlp`` does not fork its routed chain onto an alternate stream
    while capturing ("" when it does), which leaves the slot nothing to swap."""
    if mlp.alt_stream is None:
        return "the MoE has no alternate stream"
    if mlp.shared_experts is None or mlp.num_fused_shared_experts:
        return "the MoE has no separate shared expert"
    return ""


def build(
    model, widths: tuple[int, ...], slots: frozenset, moe_bound
) -> Optional[Bound]:
    """Slot moe_swap's build (checks only: it launches moe_nofill's kernels), or None
    when it is not requested: every width moe_nofill serves when every MoE layer it
    took forks onto an alternate stream, else none. Host state, alike on every rank."""
    if SLOT not in slots:
        return None
    reasons = []
    if moe_bound is None or not moe_bound.widths:
        reasons.append("slot moe_nofill serves no width")
    else:
        for layer_id, mlp in sorted(moe_bound.mlps.items()):
            if why := two_stream_reason(mlp):
                reasons.append(f"layer {layer_id}: {why}")
    if reasons:
        reason = "; ".join(reasons)
        logger.warning("NVIDIA megakernel: slot %s serves no width: %s", SLOT, reason)
        return Bound(reason=reason)
    logger.info(
        "NVIDIA megakernel: slot %s on %d MoE layers at %d-%d rows: the routed chain "
        "on the main stream, the shared expert on the alternate stream",
        SLOT,
        len(moe_bound.mlps),
        min(moe_bound.widths),
        max(moe_bound.widths),
    )
    return Bound(widths=tuple(moe_bound.widths))
