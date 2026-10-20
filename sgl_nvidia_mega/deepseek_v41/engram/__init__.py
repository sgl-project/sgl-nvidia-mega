# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/layers/engram.py (Engram.forward) and python/sglang/srt/models/deepseek_v4.py.
"""The Engram layers: table lookup, WKV GEMM and gate, before the layer's attention.

``hashes`` computes the step's token hashes once; ``apply`` runs one Engram layer
on the materialized streams and writes its output back as the residual. Stock:
``Engram.forward`` (lookup -> WKV -> ``engram_gate``) from its own parts, and the
DeepSeek-V4.1 image-token rows keep their residual, as SGLang's layer loop does.
With the ``engram`` slot (``prefetch``), ``hashes`` also issues every layer's own
rows on a side stream, ``before_moe`` / ``before_layer`` run the prefetch's two
main-stream all-reduces at its layers, and ``apply`` only gates.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from sgl_nvidia_mega.deepseek_v41 import boundary

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/layers/engram.py": (
        "engram_gate",
        "embed",
        "forward(cp_all_tokens)",
        "wkv",
        "q_weight",
        "k_weight",
        "eps",
        "clamp_value",
        "layer_hash_index",
    ),
    "srt/models/deepseek_v4.py": ("engram_hasher", "engram", "config"),
    "srt/configs/deepseek_v41.py": ("vision_n_layers",),
    "srt/model_executor/forward_batch_info.py": ("contains_mm_inputs",),
}


class Bound(NamedTuple):
    """``st.bound["engram"]``: the prefetch (or None) and why it is off."""

    prefetch: object
    reason: str


def build(model, ctx, group):
    from sgl_nvidia_mega.deepseek_v41 import runtime
    from sgl_nvidia_mega.deepseek_v41.engram import prefetch

    if "engram" not in runtime.SLOTS:
        return Bound(None, "stock (SGLANG_NVIDIA_MEGAKERNEL_SLOTS)")
    return Bound(*prefetch.build(model, ctx, group))


def hashes(st) -> None:
    """The step's token hashes [rows, Engram layers, hash columns], or None; with the
    prefetch, every Engram layer's lookup + WKV issued right after them."""
    hasher = st.model.engram_hasher
    st.hash_ids = hasher(st.input_ids, st.fb) if hasher is not None else None
    if st.slots.engram_prefetch and st.hash_ids is not None:
        st.bound["engram"].prefetch.issue(st.hash_ids)


def before_moe(i: int, st) -> None:
    """Layer ``i`` between its attention and its MoE: the prefetch's lookup reduce."""
    if st.slots.engram_prefetch and st.hash_ids is not None:
        prefetch = st.bound["engram"].prefetch
        if i == prefetch.reduce_at:
            prefetch.reduce()


def before_layer(i: int, st) -> None:
    """Layer ``i`` before its Engram: the prefetch's WKV gather."""
    if st.slots.engram_prefetch and st.hash_ids is not None:
        prefetch = st.bound["engram"].prefetch
        if i == prefetch.gather_at:
            prefetch.gather()


def apply(layer, st) -> None:
    """``layer.engram`` on ``st.state.residual`` (materialized by the runtime)."""
    from sglang.srt.layers.engram import engram_gate

    engram, config = layer.engram, st.model.config
    x = st.state.residual
    if st.slots.engram_prefetch:
        out = st.bound["engram"].prefetch.gate(st.layer_id, x)
    else:
        ids = st.hash_ids[:, engram.layer_hash_index]
        emb = engram.embed(ids, st.fb, cp_all_tokens=False)
        kv, _ = engram.wkv(emb.flatten(-2))
        out = engram_gate(
            x, kv, engram.q_weight, engram.k_weight, engram.eps, engram.clamp_value
        )
    # Only multimodal placeholders become image_token_id, so a text-only batch (every
    # decode / verify graph among them) skips the full-residual select.
    if (
        config.model_type == "deepseek_v41"
        and config.vision_n_layers > 0
        and st.fb.contains_mm_inputs()
    ):
        image = st.input_ids == config.image_token_id
        out = torch.where(image[:, None, None], x, out)
    boundary.set_residual(st, out)
