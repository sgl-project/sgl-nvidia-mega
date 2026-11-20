# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/minimax_m3.py (MiniMaxM3Model.forward, MiniMaxM3DecoderLayer.forward)
# and python/sglang/srt/managers/mm_utils.py (general_mm_embed_routine).
"""The residual stream between sublayers of a covered step the stock executor runs:
the embedding and the stream's start (``start``), each layer's attention input
(``attn_input``) and FFN input (``ffn_input``), and the final norm (``final``).

Stock: SGLang's own stage boundaries (``attn_boundary`` / ``ffn_boundary``, the
push-plane all-reduce + residual + norm fusions) and residual batch, in the order
``MiniMaxM3DecoderLayer.forward`` calls them.
"""

from __future__ import annotations

from typing import Any

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    # residual_batch: the module alias SGLang's MiniMax-M3 model imports too
    "srt/models/minimax_m3.py": (
        "residual_batch",
        "norm",
        "finalize_norm",
        "end_layer",
        "layers_to_capture",
        "attn_boundary",
        "ffn_boundary",
    ),
    "srt/layers/layer_boundary/residual/batch.py": (
        "start",
        "final_norm",
        "final_norm(finalize_norm)",
        "final_norm(capture)",
        "final_norm(skip_empty)",
    ),
    "srt/managers/mm_utils.py": ("prepare_mm_inputs",),
    "srt/layers/layer_boundary/stage.py": (
        "prepare",
        "_prepare_attention(capture_gathered)",
        "finish",
    ),
    "srt/layers/aux_hidden_states.py": ("capture",),
}

# The SGLang function whose body this module mirrors, by the digest of its code
# (test/unit/test_contract.py recomputes it on the installed SGLang)
SGLANG_MIRRORS = {
    "srt/managers/mm_utils.py:general_mm_embed_routine": "4230344c2697664e",
}


def start(st) -> Any:
    """The step's embeddings [rows, hidden]: ``general_mm_embed_routine``'s
    ``prepare_mm_inputs`` (its decode / verify branch: the embedding, copied into the
    batch's input buffer when it has one), then ``MiniMaxM3Model.forward``'s
    ``residual_batch.start``."""
    from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
    from sglang.srt.managers.mm_utils import prepare_mm_inputs

    x = prepare_mm_inputs(st.input_ids, st.fb, st.model, {}, st.lm)
    residual_batch.start(st.fb)
    return x


def attn_input(layer, x, st, captured) -> Any:
    """The attention input of ``layer``: the previous sublayer's output reduced into
    the stream, then the input norm; ``captured`` (the DSpark capture list) takes the
    updated stream at a capture layer."""
    return layer.attn_boundary.prepare(x, st.fb, capture_gathered=captured)


def ffn_input(layer, out, st) -> Any:
    """The FFN input of ``layer`` from its attention output ``out``."""
    out = layer.attn_boundary.finish(out, st.fb)
    return layer.ffn_boundary.prepare(out, st.fb)


def final(st, x, aux) -> Any:
    """The last layer's output reduced into the stream and normed; the DSpark capture
    of the last layer's stream when the draft reads it."""
    from sglang.srt.layers.layer_boundary.residual import batch as residual_batch

    m = st.model
    return residual_batch.final_norm(
        x,
        st.fb,
        m.norm,
        finalize_norm=m.finalize_norm,
        capture=aux.capture if m.end_layer in m.layers_to_capture else None,
        skip_empty=True,
    )
