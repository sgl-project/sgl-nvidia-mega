# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team
"""GLM-5 DSA (GlmMoeDsaForCausalLM, NVFP4, TP8) megakernel on SM103 / SM107.

Covers verify decode (``spec.SPEC``) and the NextN MTP draft decode
(``draft_spec.SPEC``). ``eligibility`` admits SM103 (B300) and SM107 (Rubin /
VR NVL72); the two GPU families run the same kernels through CuTe DSL dispatch.
"""

from __future__ import annotations

MODEL_TYPE = "glm_moe_dsa"
TP_SIZE = 8
HIDDEN = 6144

# Published HuggingFace config fields for the GLM-5.2-NVFP4 / GLM-5 DSA model.
# eligibility.py validates these against the loaded model at startup.
HF_CONFIG = dict(
    hidden_size=6144,
    num_attention_heads=64,
    q_lora_rank=2048,
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    qk_nope_head_dim=192,
    v_head_dim=256,
    index_n_heads=32,
    index_head_dim=128,
    index_topk=2048,
    n_routed_experts=256,
    num_experts_per_tok=8,
    n_shared_experts=1,
    n_group=1,
    topk_group=1,
    norm_topk_prob=True,
    scoring_func="sigmoid",
    topk_method="noaux_tc",
    moe_intermediate_size=2048,
    first_k_dense_replace=3,
    num_hidden_layers=78,
)
