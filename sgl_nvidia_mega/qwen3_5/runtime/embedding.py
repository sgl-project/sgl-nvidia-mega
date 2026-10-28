# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/managers/mm_utils.py (general_mm_embed_routine).
"""The token embedding of a covered step. Stock: SGLang's input preparation of the VL
model (the vocabulary-parallel lookup, copied into the batch's embedding buffer when
the batch has one), or the language model's own lookup for a language-only model."""

from __future__ import annotations

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/managers/mm_utils.py": (
        "prepare_mm_inputs",
        "prepare_mm_inputs(multimodal_model)",
        "prepare_mm_inputs(use_deepstack)",
    ),
    "srt/models/qwen3_vl.py": ("language_model_only", "use_deepstack"),
    "srt/models/qwen3_5.py": ("embed_tokens",),
}

# The SGLang functions whose body this module mirrors, by the digest of their code
# (test/unit/test_contract.py recomputes them on the installed SGLang)
SGLANG_MIRRORS = {
    "srt/managers/mm_utils.py:general_mm_embed_routine": "4230344c2697664e",
}


def embed(st):
    """[rows, hidden] for ``st.input_ids``."""
    from sglang.srt.managers.mm_utils import prepare_mm_inputs

    lm = st.lm
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
