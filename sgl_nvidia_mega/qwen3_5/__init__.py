# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Qwen3.5 (397B-A17B, NVFP4) megakernel: its spec and the setups it serves (``spec``),
the slot plan (``plan``), the executor of a covered target-verify step and its build
(``runtime``), the step code per package (``boundary``, ``gdn``, ``attention``, ``moe``,
``logits``), and the SM103 (GB300) kernels (``sm103``). The MTP draft runs SGLang's own
forward."""

# Qwen3.5-397B-A17B's text_config (the HF config nests it under text_config), which
# the kernels are built for
TEXT_CONFIG = dict(
    model_type="qwen3_5_moe_text",
    hidden_size=4096,
    num_hidden_layers=60,
    full_attention_interval=4,
    num_attention_heads=32,
    num_key_value_heads=2,
    head_dim=256,
    attn_output_gate=True,
    linear_num_key_heads=16,
    linear_num_value_heads=64,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    num_experts=512,
    num_experts_per_tok=10,
    moe_intermediate_size=1024,
    shared_expert_intermediate_size=1024,
    hidden_act="silu",
    rms_norm_eps=1e-6,
)
