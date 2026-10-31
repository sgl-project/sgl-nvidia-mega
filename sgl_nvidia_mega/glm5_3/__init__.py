# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GLM-5.3 (NVFP4) megakernel: its spec, the setups it serves (``spec``), the step
groups per width (``plan``), the build and the executor of a covered MTP verify step
(``runtime``), the step code per package (``attention``, ``boundary``, ``moe``,
``logits``), and the SM103 (B300) CuTe DSL kernels (``sm103``).

GLM-5.2 has the same architecture and the same value of every config field the specs
pin (``HF_CONFIG``), so they serve it as well."""

MODEL_TYPE = "glm_moe_dsa"
# One tensor-parallel group of 8 ranks; the hidden size the kernels are built for
TP_SIZE = 8
HIDDEN = 6144

# GLM-5.3's shapes and routing, which the kernels are built for (GLM-5.2's are the same)
HF_CONFIG = dict(
    hidden_size=HIDDEN,
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
    # the routing the o_proj hop's router and the MoE engine build in: sigmoid scores
    # with the correction bias, one expert group, the top-8 weights renormalized
    n_group=1,
    topk_group=1,
    norm_topk_prob=True,
    scoring_func="sigmoid",
    topk_method="noaux_tc",
    moe_intermediate_size=2048,
    first_k_dense_replace=3,
    num_hidden_layers=78,
)
