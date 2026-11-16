# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The FFN sublayer of a covered step the stock executor runs: the dense MLP of the
leading dense layers, and the MoE of the others (the shared expert on the main
stream; the router, the top-4 and the NVFP4 routed experts on the alternate stream,
swapped with slot moe_swap; their finalize deferred into the next boundary).

Stock: SGLang's ``MiniMaxM3MLP`` / ``MiniMaxM3MoE`` modules.

Slot ``moe_nofill`` (``moe_nofill.py``, bitwise): an MoE layer runs SGLang's own calls
in SGLang's order, but quantizes the routed experts' NVFP4 input into buffers built
once, without the chain's fill launch and its allocations. A step that misses one of
the slot's preconditions runs SGLang's module for that layer.

Slot ``moe_swap`` (``moe_swap.py``, bitwise; needs moe_nofill): moe_nofill's MoE with
the streams swapped, the routed chain on the main stream (the boundary kernel's PDL
successor) and the shared expert on the alternate one.

Both were bitwise to SGLang's forward on the previous SGLang (the composition check,
GB300 jobs 1287884 / 1287885, and served token identity); their GPU check on this
SGLang is the parity switch at every width they serve.
"""

from __future__ import annotations

from typing import Any

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/minimax_m3.py": (
        "is_layer_sparse",
        "layer_id",
        "mlp",
        "forward(forward_batch)",
    ),
}


def step(layer, x, st) -> Any:
    """``layer``'s FFN output from its normed input ``x``: an unreduced partial, or the
    MoE's deferred-finalize handoff (``MiniMaxM3DecoderLayer.forward``'s
    ``self.mlp(hidden_states, forward_batch=forward_batch)``)."""
    if st.slots.moe_nofill and layer.is_layer_sparse:
        from sgl_nvidia_mega.minimax_m31.moe import moe_nofill, moe_swap

        bound = st.bound.get("moe")
        why = moe_nofill.fallback_reason(bound, layer, x)
        if not why:
            if st.slots.moe_swap:
                return moe_swap.forward(layer.mlp, x, bound.quant)
            return moe_nofill.forward(layer.mlp, x, bound.quant)
        moe_nofill.log_fallback(layer.layer_id, why)
    return layer.mlp(x, forward_batch=st.fb)


def build(model, widths: tuple[int, ...], slots: frozenset):
    """Slot moe_nofill's build (weights checked, buffers allocated, its kernel checked
    against FlashInfer's own call), or None when it is not requested; a collective."""
    if "moe_nofill" not in slots:
        return None
    from sgl_nvidia_mega.minimax_m31.moe import moe_nofill

    return moe_nofill.build(model, widths)


def build_swap(model, widths: tuple[int, ...], slots: frozenset, moe_bound):
    """Slot moe_swap's build (checks only, on moe_nofill's), or None when it is not
    requested."""
    from sgl_nvidia_mega.minimax_m31.moe import moe_swap

    return moe_swap.build(model, widths, slots, moe_bound)


def serves(bound, rows: int) -> bool:
    """Whether a slot's build serves a step of this many rows."""
    return bound is not None and bound.serves(rows)
