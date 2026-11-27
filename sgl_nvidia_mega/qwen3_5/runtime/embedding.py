# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/managers/mm_utils.py (general_mm_embed_routine).
"""The token embedding of a covered step. Stock: SGLang's input preparation of the VL
model (the vocabulary-parallel lookup, copied into the batch's embedding buffer when
the batch has one), or the language model's own lookup for a language-only model.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/managers/mm_utils.py": (
        "prepare_mm_inputs",
        "prepare_mm_inputs(multimodal_model)",
        "prepare_mm_inputs(use_deepstack)",
    ),
    "srt/models/qwen3_vl.py": ("language_model_only", "use_deepstack"),
    "srt/models/qwen3_5.py": ("embed_tokens",),
    "srt/layers/logits_processor.py": ("vocab_size",),
    "srt/model_executor/forward_batch_info.py": ("input_embeds",),
}

# verify widths (rows) whose capture logged the engagement line
_NOTED: set = set()

# The SGLang functions whose body this module mirrors, by the digest of their code
# (test/unit/test_contract.py recomputes them on the installed SGLang)
SGLANG_MIRRORS = {
    "srt/managers/mm_utils.py:general_mm_embed_routine": "4230344c2697664e",
}


def build_local(model, binding):
    from sgl_nvidia_mega.qwen3_5 import runtime
    from sgl_nvidia_mega.qwen3_5.sm103.draft import embed_local

    if not runtime.measured_option("embed_local"):
        return None
    return embed_local.build(
        name="embed_local",
        embed=model.model.embed_tokens,
        vocab=int(model.logits_processor.vocab_size),
        binding=binding,
        what="the verify",
    )


def _note(rows: int) -> None:
    import torch

    if rows in _NOTED or not torch.cuda.is_current_stream_capturing():
        return
    _NOTED.add(rows)
    logger.info(
        "NVIDIA megakernel: embed_local at %d rows: the verify's embedding rows from "
        "the replicated table in one gather (no all-reduce)",
        rows,
    )


def embed(st):
    """[rows, hidden] for ``st.input_ids``."""
    from sglang.srt.managers.mm_utils import prepare_mm_inputs

    lm = st.lm
    local = st.bound.get("embed_local")
    if local is not None:
        _note(int(st.input_ids.shape[0]))
        rows = local(st.input_ids)
        buf = None if lm.language_model_only else getattr(st.fb, "input_embeds", None)
        if buf is not None:
            buf.copy_(rows)
            return buf
        return rows
    if lm.language_model_only:
        return st.model.embed_tokens(st.input_ids)
    # a verify step carries no multimodal input, so kwargs gains no deepstack rows
    return prepare_mm_inputs(
        st.input_ids,
        st.fb,
        st.model,
        {"positions": st.positions, "pp_proxy_tensors": None},
        multimodal_model=lm,
        use_deepstack=lm.use_deepstack,
    )
