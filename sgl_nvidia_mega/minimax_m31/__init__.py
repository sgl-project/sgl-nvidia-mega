# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax-M3.1-Flash megakernel: the target's spec (``spec``) and its covered verify
step (``runtime``, operands from ``weights``), the DSpark draft's spec (``draft``) and
step (``draft_step``), and the SM103 kernels (``sm103``): MegaAttention, the
token-parallel MoE and the draft backbone."""

# Fixed by the kernels; per TP4 rank unless noted
TP = 4
ROWS = 8  # one request's eight verify rows
DRAFT_ROWS = 7  # one request's draft block: the bonus token and six mask tokens
DRAFT_LAYERS = 5
PAGE = 128
MAX_CONTEXT = 8192 * PAGE  # the attention's page tables: 1M tokens

# The text config fields the target kernels are built for
HF_CONFIG = dict(
    hidden_size=6144,
    head_dim=128,
    num_attention_heads=64,
    num_key_value_heads=4,
    rotary_dim=64,
    num_local_experts=128,
    num_experts_per_tok=4,
    intermediate_size=3072,
    n_shared_experts=1,
    scoring_func="sigmoid",
    hidden_act="swigluoai",
    swiglu_alpha=1.702,
    swiglu_limit=7.0,
    use_gemma_norm=True,
    qk_norm_type="per_head",
)
# sparse_attention_config, as SGLang's MSA-v1 backend reads it
SPARSE_CONFIG = dict(
    sparse_index_dim=128,
    sparse_num_index_heads=4,
    sparse_topk_blocks=16,
    sparse_block_size=128,
    sparse_local_block=1,
    sparse_init_block=0,
    sparse_score_type="max",
)
# sparse_attention_config switches the kernels do not implement
SPARSE_OFF = (
    "sparse_enable_kv_share",
    "sparse_enable_vo_rope",
    "sparse_kvshare_value_drop_rope_dim",
    "sparse_indexer_lookahead_num_layers",
)
# The draft config fields the draft backbone is built for (SGLang derives the draft's
# config from the checkpoint's dspark fields)
DRAFT_HF_CONFIG = dict(
    hidden_size=6144,
    head_dim=128,
    num_attention_heads=64,
    num_key_value_heads=4,
    rotary_dim=64,
    num_hidden_layers=DRAFT_LAYERS,
    dense_intermediate_size=12288,
    hidden_act="swigluoai",
    swiglu_alpha=1.702,
    swiglu_limit=7.0,
    use_gemma_norm=True,
    qk_norm_type="per_head",
    sliding_window=4096,
)
