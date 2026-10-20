# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/deepseek_v4.py (the tails of DeepseekV4Model.forward and DeepseekV4ForCausalLM.forward).
"""The head after the last layer: the mHC combine + norm (``head``), then the LM
head and logits (``logits``, SGLang's logits processor). Stock: the tail of
``DeepseekV4Model.forward`` and of ``DeepseekV4ForCausalLM.forward``."""

from __future__ import annotations

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/ops/layernorm/mhc.py": ("mhc_combine",),
    "srt/models/deepseek_v4.py": (
        "norm",
        "hc_mult",
        "logits_processor",
        "lm_head",
    ),
    "srt/layers/logits_processor.py": ("forward(hidden_states_before_norm)",),
}


def head(st, residual, pre):
    """(hidden [rows, dim], pre_hc_head [rows, hc * dim]) from the final streams."""
    from sglang.kernels.ops.layernorm.mhc import mhc_combine

    m = st.model
    pre_hc_head = residual.flatten(1)
    hidden = m.norm(mhc_combine(pre_hc_head.float(), pre, m.hc_mult, residual.dtype))
    return hidden, pre_hc_head


def logits(st, hidden, pre_hc_head, aux):
    """SGLang's ``LogitsProcessorOutput`` for the step."""
    lm = st.lm
    return lm.logits_processor(
        st.input_ids,
        hidden,
        lm.lm_head,
        st.fb,
        aux,
        hidden_states_before_norm=None if aux is not None else pre_hc_head,
    )
