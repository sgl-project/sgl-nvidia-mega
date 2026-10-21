# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/qwen3_5.py (Qwen3_5AttentionDecoderLayer.self_attention).
"""The gated full attention of a covered target-verify step, over the layer's loaded
weights, SGLang's attention backend and its KV cache.

Stock: the CUDA arm of ``Qwen3_5AttentionDecoderLayer.self_attention`` with SGLang's own
ops in its order: the qkv projection with the fused q / k Gemma RMSNorm + RoPE + gate
split, the attention through the backend (which writes this step's KV rows), the
sigmoid output gate and ``o_proj``. ``step`` returns the TP-partial output, whose
all-reduce the boundary runs. Slot ``skv2`` (``split_kv.py``) runs the attention's core
through the split-KV attention of SGLang branch nvidia/qwen35-agentx-gb300, and slot
``mka`` (``mka.py``) the rest of the layer through that branch's fused prologue and
epilogue.
"""

from __future__ import annotations

import msgspec

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/qwen3_vl.py": ("is_mrope_enabled",),
    "srt/models/qwen3_5.py": (
        "Qwen3_5AttentionDecoderLayer",
        "layers",
        "start_layer",
        "end_layer",
        "_prepare_qkv_gate",
        "_prepare_qkv_gate(positions)",
        "_prepare_qkv_gate(hidden_states)",
        "_prepare_qkv_gate(forward_batch)",
        "attn",
        "attn_output_gate",
        "o_proj",
    ),
    "kernels/ops/elementwise/elementwise.py": (
        "fused_sigmoid_mul",
        "fused_sigmoid_mul(inplace)",
    ),
}

# The SGLang functions whose body this module mirrors, by the digest of their code
# (test/unit/test_contract.py recomputes them on the installed SGLang)
SGLANG_MIRRORS = {
    "srt/models/qwen3_5.py:Qwen3_5AttentionDecoderLayer.self_attention": "db0f4d98e0fb1abf",
}

ATTENTION_SLOTS = ("skv2", "mka", "mka_push")


class Bound(msgspec.Struct, frozen=True):
    """The attention package's build result (``st.bound["attention"]``)."""

    skv2: bool = False  # slot skv2 serves every full-attention layer
    mka: bool = False
    mka_push: bool = False
    reason: str = ""  # why a requested slot is not built


def build(model, ctx) -> Bound:
    """The requested attention slots for every full-attention layer, or why a layer
    cannot take them."""
    from sglang.srt.models.qwen3_5 import Qwen3_5AttentionDecoderLayer

    from sgl_nvidia_mega.qwen3_5.attention import mka, split_kv
    from sgl_nvidia_mega.qwen3_5.runtime import SLOTS

    if not SLOTS & set(ATTENTION_SLOTS):
        return Bound(reason="stock (SGLANG_NVIDIA_MEGAKERNEL_SLOTS)")
    m = model.model
    layers = [
        (i, m.layers[i])
        for i in range(m.start_layer, m.end_layer)
        if isinstance(m.layers[i], Qwen3_5AttentionDecoderLayer)
    ]
    for i, layer in layers:
        reason = ""
        if "skv2" in SLOTS:
            reason = split_kv.unsupported(layer.attn, ctx.tokens_per_request)
        if not reason and "mka" in SLOTS:
            reason = (
                mka.unsupported(layer)
                if model.is_mrope_enabled
                else "no M-RoPE positions (MKA reads them)"
            )
        if reason:
            return Bound(reason=f"layer {i}: {reason}")
    if "mka" in SLOTS:
        for _, layer in layers:
            mka.pack(layer)
    return Bound(skv2="skv2" in SLOTS, mka="mka" in SLOTS, mka_push="mka_push" in SLOTS)


def pool_reasons(view, binding, skv2: bool = True, mka: bool = False) -> list[str]:
    """Why the attention slots cannot read this forward's KV pools ([] when they can)."""
    from sgl_nvidia_mega.qwen3_5.attention import mka as fused
    from sgl_nvidia_mega.qwen3_5.attention import split_kv

    reasons = split_kv.pool_reasons(view.kv, binding.page_size) if skv2 else []
    return reasons + (fused.pool_reasons(view.kv) if mka else [])


def step(layer, x, st):
    """``layer``'s attention on the normed input ``x`` [rows, hidden]: the TP-partial
    output [rows, hidden]."""
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

    if st.slots.mka:
        from sgl_nvidia_mega.qwen3_5.attention import mka

        return mka.step(layer, x, st)
    q, k, v, gate = layer._prepare_qkv_gate(
        positions=st.positions, hidden_states=x, forward_batch=st.fb
    )
    if st.slots.skv2:
        from sgl_nvidia_mega.qwen3_5.attention import split_kv

        o = split_kv.attend(layer.attn, q, k, v, st)
    else:
        o = layer.attn(q, k, v, st.fb)
    if layer.attn_output_gate:
        o = fused_sigmoid_mul(o, gate, inplace=True)
    out, _ = layer.o_proj(o)
    return out
