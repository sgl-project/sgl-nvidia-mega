# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0): Qwen3_5AttentionDecoderLayer.forward,
# self_attention and forward_prepare_cuda_fused.

from __future__ import annotations

SLOT = "attn_prefill_qk_rows"

SGLANG_READS = {
    "srt/models/qwen3_5.py": (
        "Qwen3_5AttentionDecoderLayer",
        "Qwen2MoeSparseMoeBlock",
        "fused_qk_gemma_rmsnorm_rope_gate",
        "attn_output_gate",
        "qkv_proj",
        "q_size",
        "kv_size",
        "q_norm",
        "k_norm",
        "rotary_emb",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "attn",
        "o_proj",
        "attn_boundary",
        "ffn_boundary",
        "mlp",
    ),
    "kernels/ops/elementwise/elementwise.py": (
        "fused_sigmoid_mul",
        "fused_sigmoid_mul(inplace)",
    ),
    "srt/layers/layernorm.py": ("GemmaRMSNorm", "weight", "variance_epsilon"),
    "srt/layers/rotary_embedding/base.py": ("cos_sin_cache", "rotary_dim"),
    "srt/layers/rotary_embedding/mrope.py": ('"axis_map"',),
    "srt/layers/layer_boundary/stage.py": ("prepare", "finish", "exit"),
    "srt/layers/layer_boundary/exit.py": ("defer_moe_finalize", "finish"),
    "srt/model_executor/forward_mode.py|srt/model_executor/forward_batch_info.py": (
        "is_idle",
    ),
}

SGLANG_MIRRORS = {
    "srt/models/qwen3_5.py:Qwen3_5AttentionDecoderLayer.forward": "6473f673a1a0796b",
    "srt/models/qwen3_5.py:Qwen3_5AttentionDecoderLayer.self_attention": "db0f4d98e0fb1abf",
    "srt/models/qwen3_5.py:Qwen3_5AttentionDecoderLayer._prepare_qkv_gate": (
        "1af0524597d94935"
    ),
    "srt/models/qwen3_5.py:Qwen3_5AttentionDecoderLayer.forward_prepare_cuda_fused": (
        "d10e23f5651990d0"
    ),
    "kernels/ops/attention/fused_qk_rmsnorm_rope_gate.py:_fused_qk_rmsnorm_rope_gate_kernel": (
        "e5ed36f681b597da",
        "39a65fdc24872fe9",
    ),
    "kernels/ops/attention/fused_qk_rmsnorm_rope_gate.py:fused_qk_gemma_rmsnorm_rope_gate": (
        "7d985b797c7ae63c",
        "618c8b9510090454",
    ),
    "kernels/ops/elementwise/elementwise.py:fused_sigmoid_mul": (
        "bf3c570a79e580a8",
        "6d5feb8d305459a5",
    ),
    "kernels/ops/elementwise/elementwise.py:_fused_sigmoid_mul_kernel": (
        "a00af816bd364384",
        "8919dc07cad4d085",
    ),
    "srt/layers/layernorm.py:GemmaRMSNorm.__init__": "637c469e021fb1c9",
}


def layer_unsupported(layer, rows) -> str:
    import torch
    from sglang.srt.layers.layernorm import GemmaRMSNorm

    if not layer.attn_output_gate:
        return "no output gate"
    norms = (layer.q_norm, layer.k_norm)
    if not all(isinstance(n, GemmaRMSNorm) for n in norms):
        return "Q / K norm is not GemmaRMSNorm"
    if layer.q_norm.variance_epsilon != layer.k_norm.variance_epsilon:
        return "Q and K norm epsilons differ"
    if any(tuple(n.weight.shape) != (layer.head_dim,) for n in norms):
        return "Q / K norm weight is not one head wide"
    if layer.q_norm.weight.dtype != torch.bfloat16:
        return f"the model is {layer.q_norm.weight.dtype}, not BF16"
    rotary = layer.rotary_emb
    if getattr(rotary, "cos_sin_cache", None) is None:
        return "no cos / sin cache"
    return rows.shape_unsupported(layer.head_dim, rotary.rotary_dim)


def warm(layer, rows) -> None:
    rows.warm(
        layer.q_norm.weight.data,
        layer.k_norm.weight.data,
        layer.rotary_emb.cos_sin_cache,
        layer.q_norm.variance_epsilon,
        layer.num_heads,
        layer.num_kv_heads,
        layer.head_dim,
        layer.rotary_emb.rotary_dim,
        getattr(layer.rotary_emb, "axis_map", None),
    )


def prepare(layer, positions, hidden_states):
    from sgl_nvidia_mega.qwen3_5.sm103.attention import qk_norm_rope_rows as rows

    qkv, _ = layer.qkv_proj(hidden_states)
    q_gate, k, v = qkv.split([layer.q_size * 2, layer.kv_size, layer.kv_size], dim=-1)
    rotary = layer.rotary_emb
    args = (
        q_gate,
        k,
        layer.q_norm.weight.data,
        layer.k_norm.weight.data,
        rotary.cos_sin_cache,
        positions,
        layer.q_norm.variance_epsilon,
        layer.num_heads,
        layer.num_kv_heads,
        layer.head_dim,
        rotary.rotary_dim,
    )
    axis_map = rotary.axis_map if positions.dim() == 2 else None
    seq_len = qkv.shape[0]
    if rows.qk_norm_rope_rows_supported(q_gate, k, layer.head_dim, rotary.rotary_dim):
        q, k, gate = rows.qk_gemma_rmsnorm_rope_rows(
            *args, has_gate=True, mrope_axis_map=axis_map
        )
    else:
        from sglang.srt.models.qwen3_5 import fused_qk_gemma_rmsnorm_rope_gate

        q, k, gate = fused_qk_gemma_rmsnorm_rope_gate(
            *args, has_gate=True, mrope_axis_map=axis_map
        )
        gate = gate.view(seq_len, -1)
    return q.view(seq_len, -1), k.view(seq_len, -1), v, gate


def self_attention(layer, positions, hidden_states, forward_batch, gate=None):
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

    q, k, v, g = prepare(layer, positions, hidden_states)
    attn_output = layer.attn(q, k, v, forward_batch)
    if gate is None:
        attn_output = fused_sigmoid_mul(attn_output, g, inplace=True)
    else:
        attn_output = gate(attn_output, g)
    output, _ = layer.o_proj(attn_output)
    return output


def forward(
    layer,
    positions,
    hidden_states,
    forward_batch,
    captured_last_layer_outputs=None,
    gate=None,
):
    from sglang.srt.models.qwen3_5 import Qwen2MoeSparseMoeBlock

    hidden_states = layer.attn_boundary.prepare(
        hidden_states,
        forward_batch,
        capture_gathered=captured_last_layer_outputs,
    )
    hs = hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
    if not forward_batch.forward_mode.is_idle() and hs.shape[0] > 0:
        hidden_states = self_attention(
            layer, positions, hidden_states, forward_batch, gate
        )
    hidden_states = layer.attn_boundary.finish(hidden_states, forward_batch)
    hidden_states = layer.ffn_boundary.prepare(hidden_states, forward_batch)
    with layer.ffn_boundary.exit(forward_batch) as ffn_exit:
        if isinstance(layer.mlp, Qwen2MoeSparseMoeBlock):
            hidden_states = layer.mlp(
                hidden_states,
                forward_batch,
                defer_finalize=ffn_exit.defer_moe_finalize,
            )
        else:
            hidden_states = layer.mlp(hidden_states)
    return ffn_exit.finish(hidden_states)
