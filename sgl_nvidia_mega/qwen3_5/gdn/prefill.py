# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0): GDNAttnBackend.forward_extend and
# MambaAttnBackendBase._track_mamba_state_extend.
"""Slot gdn_prefill_split_conv: channel-sliced conv into contiguous Q/K/V.

The native Triton recurrence, state writes, gated norm and output projection
retain SGLang's order. Only the conv output layout changes, above the fused
Q/K/V split limit. The caller checks the view before any layer mutates state.
"""

from __future__ import annotations

SGLANG_READS = {
    "kernels/ops/mamba/causal_conv1d_triton.py": ("causal_conv1d_fn",),
    "kernels/ops/attention/triton_gdn_fused_proj.py": (
        "qwen3_5_gdn_prefill_projection_views",
    ),
    "kernels/ops/attention/linear/fused_gdn_gating.py": ("fused_gdn_gating",),
    "kernels/ops/attention/fla/chunk.py": ("chunk_gated_delta_rule",),
    "srt/models/qwen3_5.py": (
        "_forward_input_proj",
        "num_k_heads",
        "num_v_heads",
        "attn_tp_size",
        "head_k_dim",
        "head_v_dim",
        "attn",
        "norm",
        "out_proj",
    ),
    "srt/layers/radix_linear_attention.py": (
        "layer_id",
        "q_dim",
        "k_dim",
        "v_dim",
        "num_q_heads",
        "num_k_heads",
        "num_v_heads",
        "head_q_dim",
        "head_k_dim",
        "head_v_dim",
        "conv_weights",
        "bias",
        "activation",
        "A_log",
        "dt_bias",
    ),
    "srt/model_executor/forward_batch_info.py": (
        "extend_prefix_lens",
        "extend_seq_lens_cpu",
    ),
    "srt/kernel_plugins/nvidia/eagle_binding.py": (
        "LinearPrefillView",
        "LinearPrefillBuffers",
        "state",
        "layer_ids",
        "conv",
        "temporal",
        "query_start_loc",
        "state_indices",
        "conv_block_table",
        "kernel_id",
        "is_graph",
        "has_mis",
        "has_track",
        "track_conv_indices",
        "track_conv_dst",
        "track_ssm_h_src",
        "track_ssm_h_dst",
        "track_ssm_final_src",
        "track_ssm_final_dst",
        "track_ssm_recompute_dst",
    ),
}

SGLANG_MIRRORS = {
    # Upstream #48004 reserves staging row 0 for padding; this slot rejects strided
    # pools before writes and passes the native pool indices unchanged.
    "srt/layers/attention/linear/gdn_backend.py:GDNAttnBackend.forward_extend": (
        "12a835fc4f78af33",
        "6ed4eb6a818f7e20",
        "bfd997695f32d409",
    ),
    "srt/layers/attention/hybrid_linear_attn_backend.py:MambaAttnBackendBase._track_mamba_state_extend": "84c447ca707eba64",
}

FUSED_SPLIT_LIMIT = 8192


def unsupported(view) -> str:
    """The first implementation follows the eager native Triton prefill chain."""
    if view.kernel_id != 1:
        return "not the native Triton prefill kernel"
    if view.is_graph or view.has_mis:
        return "prefill graph or multi-item layout"
    if not view.state.temporal.is_contiguous() or any(
        not x.is_contiguous() for x in view.state.conv
    ):
        return "strided committed state pool"
    if (
        view.track_ssm_recompute_dst is not None
        and view.track_ssm_recompute_dst.numel()
    ):
        return "recomputed checkpoint layout"
    return ""


def split_conv(attn, mixed_qkv, view, forward_batch, conv_states):
    """Three native conv launches, writing dense [1, T, heads, dim] tensors."""
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn

    extra = (
        {"block_table": view.conv_block_table}
        if view.conv_block_table is not None
        else {}
    )
    has_initial_state = forward_batch.extend_prefix_lens > 0
    tokens = mixed_qkv.shape[0]
    outputs = []
    start = 0
    for heads, dim in (
        (attn.num_q_heads, attn.head_q_dim),
        (attn.num_k_heads, attn.head_k_dim),
        (attn.num_v_heads, attn.head_v_dim),
    ):
        channels = slice(start, start + heads * dim)
        start = channels.stop
        out = causal_conv1d_fn(
            mixed_qkv[:, channels].transpose(0, 1),
            attn.conv_weights[channels],
            None if attn.bias is None else attn.bias[channels],
            activation=attn.activation,
            conv_states=conv_states[:, channels],
            has_initial_state=has_initial_state,
            cache_indices=view.state_indices,
            query_start_loc=view.query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            **extra,
        )
        outputs.append(out.transpose(0, 1)[:tokens].view(1, tokens, heads, dim))
    return outputs


def step(gdn, hidden, view, forward_batch):
    from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
    from sglang.kernels.ops.attention.linear.fused_gdn_gating import fused_gdn_gating
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
        qwen3_5_gdn_prefill_projection_views,
    )

    attn = gdn.attn
    qkvz, ba = gdn._forward_input_proj(hidden)
    mixed, z, b, a = qwen3_5_gdn_prefill_projection_views(
        qkvz,
        ba,
        gdn.num_k_heads // gdn.attn_tp_size,
        gdn.num_v_heads // gdn.attn_tp_size,
        gdn.head_k_dim,
        gdn.head_v_dim,
    )
    li = view.state.layer_ids.index(attn.layer_id)
    conv_states = view.state.conv[0][li]
    ssm = view.state.temporal[li]
    if view.has_track:
        conv_states[view.track_conv_dst] = mixed[view.track_conv_indices].transpose(
            1, 2
        )
    query, key, value = split_conv(attn, mixed, view, forward_batch, conv_states)
    g, beta = fused_gdn_gating(attn.A_log, a, b, attn.dt_bias)
    core, _, h = chunk_gated_delta_rule(
        q=query,
        k=key,
        v=value,
        g=g,
        beta=beta,
        initial_state=ssm,
        initial_state_indices=view.state_indices,
        cu_seqlens=view.query_start_loc,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
        inplace_update=True,
    )
    if view.has_track:
        if view.track_ssm_h_src.numel():
            ssm[view.track_ssm_h_dst] = h.squeeze(0)[view.track_ssm_h_src].to(
                ssm.dtype, copy=False
            )
        if view.track_ssm_final_src.numel():
            ssm[view.track_ssm_final_dst] = ssm[view.track_ssm_final_src]
    core = gdn.norm(core.reshape(-1, core.shape[-1]), z)
    return gdn.out_proj(core.reshape(qkvz.shape[0], -1))[0]
