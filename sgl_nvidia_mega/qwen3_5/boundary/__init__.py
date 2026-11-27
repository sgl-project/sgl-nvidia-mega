# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The residual stream between sublayers: the stream's start (``open_stream``), each
layer's input norm before the mixer (``pre``), the mixer's all-reduce + residual add +
norm into the MoE input (``post_attn``), the MoE's exit (``ffn_exit``, whose ``finish``
completes or carries its output), and the final norm (``close_stream``).

Stock: SGLang's own stage boundaries of each layer, in the decoder layers' order, and
its ``residual_batch`` around the layers. With SGLang's CuTe DSL fusion they run
FlashInfer's fused all-reduce + residual add + RMSNorm and take the MoE's deferred
finalize. Slot ``hop`` (``hop.py``) runs those three fused points on the decode hop.
"""

from __future__ import annotations

from sgl_nvidia_mega.qwen3_5.boundary import hop
from sgl_nvidia_mega.qwen3_5.boundary.hop import build  # noqa: F401

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/qwen3_5.py": (
        "attn_boundary",
        "ffn_boundary",
        "norm",
        "flashinfer_mnnvl_cutedsl_fusion",
    ),
    "srt/layers/layer_boundary/stage.py": (
        "prepare",
        "_prepare_attention(capture_gathered)",
        "finish",
        "exit",
    ),
    "srt/layers/layer_boundary/exit.py": ("defer_moe_finalize", "finish"),
    "srt/layers/layer_boundary/residual/batch.py": (
        "start",
        "final_norm",
        "final_norm(finalize_norm)",
        "final_norm(skip_empty)",
    ),
}


def open_stream(st) -> None:
    """Start the residual stream for the first layer."""
    from sglang.srt.layers.layer_boundary.residual.batch import start

    start(st.fb)


def _engine(st):
    if st.slots.hop:
        return st.bound["boundary"].hop
    if st.slots.bt_tail:
        return st.bound["boundary"].bt
    return None


def pre(layer, hidden, st, capture=None):
    """``layer``'s mixer input from the stream (its input norm)."""
    engine = _engine(st) if capture is None else None
    if engine is not None:
        x = hop.layer_input(engine, layer, hidden, st)
        if x is not None:
            return x
    _refuse_lamport(st, layer)
    return layer.attn_boundary.prepare(hidden, st.fb, capture_gathered=capture)


def _refuse_lamport(st, layer=None) -> None:
    from sgl_nvidia_mega.qwen3_5.moe import fc2ar

    if fc2ar.active() is not None:
        point = "final norm" if layer is None else f"layer {layer.layer_id}'s input"
        fc2ar.refuse_unconsumed(st, point)


def post_attn(layer, part, st):
    """The mixer's partial output into the stream; returns the MoE input."""
    hidden = layer.attn_boundary.finish(part, st.fb)
    engine = _engine(st)
    if engine is not None:
        x = hop.moe_input(engine, layer, hidden, st)
        if x is not None:
            return x
    if st.slots.gemm_ar:
        # a mixer projection deferred into this all-reduce runs before SGLang's
        st.bound["boundary"].bt.gemm_ar.settle()
    return layer.ffn_boundary.prepare(hidden, st.fb)


def ffn_exit(layer, st):
    """The MoE's exit scope: its ``defer_moe_finalize`` tells the MoE whether to hand
    over its finalize, and ``finish(out)`` completes or carries the output."""
    return layer.ffn_boundary.exit(st.fb)


def close_stream(st, hidden):
    """The final norm after the last layer, with the CuTe DSL finalize when the last
    MoE handed one over."""
    from sglang.srt.layers.layer_boundary.residual.batch import final_norm

    if st.slots.hop:
        x = hop.final_norm(st.bound["boundary"].hop, hidden, st)
        if x is not None:
            return x
    from sgl_nvidia_mega.qwen3_5.moe import fc2ar

    lam = fc2ar.pending(st) if fc2ar.active() is not None else None
    if lam is not None:
        x = hop.final_norm(lam, hidden, st)
        if x is not None:
            return x
    _refuse_lamport(st)
    m = st.model
    return final_norm(
        hidden,
        st.fb,
        m.norm,
        finalize_norm=m.flashinfer_mnnvl_cutedsl_fusion,
        skip_empty=True,
    )
