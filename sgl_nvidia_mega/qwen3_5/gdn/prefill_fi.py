# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0): GDNAttnBackend.forward_extend,
# FlashInferGDNKernel.extend, RadixLinearAttention.forward,
# _linear_attention_with_output_impl and Qwen3_5GatedDeltaNet.forward.

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

SGLANG_READS = {
    "kernels/ops/attention/triton_gdn_fused_proj.py": (
        "qwen3_5_gdn_prefill_projection_views",
    ),
    "kernels/ops/attention/linear/fused_gdn_gating.py": ("fused_gdn_gating",),
    "srt/layers/attention/linear/kernels/gdn_flashinfer.py": (
        "FlashInferGDNKernel",
        "_run_prefill",
        "_prefill_state_dtype",
    ),
    "srt/layers/attention/linear/kernels/state_pool_io.py": (
        "gather_pool_states",
        "scatter_pool_states",
        "state_pool_indices",
    ),
    "srt/layers/attention/linear/gdn_backend.py": (
        "kernel_dispatcher",
        "extend_kernel",
    ),
    "srt/layers/attention/hybrid_linear_attn_backend.py": (
        "linear_attn_backend",
        "state_layer_cache",
    ),
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "srt/model_executor/runner_utils/forward_batch.py": ("get_forward_batch",),
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
        "_linear_attention_with_output_impl",
        "get_attn_backend",
        "layer_id",
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
        "global_num_token_non_padded_cpu",
    ),
    # the track rows' names in SGLang's forward metadata, which its copy helpers read
    "srt/layers/attention/mamba/mamba2_metadata.py": (
        "track_conv_indices",
        "conv_states_mask_indices",
        "track_ssm_recompute_dst",
        "track_ssm_h_src",
        "track_ssm_h_dst",
        "track_ssm_final_src",
        "track_ssm_final_dst",
    ),
    "srt/kernel_plugins/nvidia/eagle_binding.py": (
        "prefill_view",
        "LinearPrefillView",
        "LinearPrefillBuffers",
        "state",
        "conv",
        "temporal",
        "query_start_loc",
        "state_indices",
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
        "state_checkpoint_cu_starts",
        "num_state_checkpoints",
        "state_checkpoint_every_n_tokens",
    ),
}
SGLANG_OPTIONAL_READS = {
    "kernels/ops/mamba/gdn_extend_state_copies.py": (
        "gather_pool_rows",
        "gather_pool_rows_ok",
        "scatter_pool_rows",
        "scatter_pool_rows_ok",
    ),
    "srt/layers/attention/linear/gdn_extend_track.py": (
        "track_conv_windows",
        "track_ssm_states_in_one_launch",
        "track_conv_windows(conv_states)",
        "track_conv_windows(mixed_qkv)",
        "track_conv_windows(metadata)",
        "track_ssm_states_in_one_launch(ssm_states)",
        "track_ssm_states_in_one_launch(h)",
        "track_ssm_states_in_one_launch(h_track_buf)",
        "track_ssm_states_in_one_launch(metadata)",
    ),
}

SGLANG_MIRRORS = {
    "srt/layers/attention/linear/gdn_backend.py:GDNAttnBackend.forward_extend": (
        "109b2f4d32797e43",
        "6d141f9de947654f",
        "590ab7ea97205370",
    ),
    "srt/layers/attention/linear/kernels/gdn_flashinfer.py:FlashInferGDNKernel.extend": (
        "c3844dc2ff568b4a",
        "42f47a126391000f",
    ),
    "srt/layers/attention/linear/kernels/gdn_flashinfer.py:FlashInferGDNKernel._run_prefill": "201bcf05dbed9433",
    "srt/layers/attention/hybrid_linear_attn_backend.py:MambaAttnBackendBase._track_mamba_state_extend": "84c447ca707eba64",
    # which conv and which gates SGLang's extend runs (the front reproduces those)
    "srt/layers/attention/linear/gdn_backend.py:GDNAttnBackend._convolve_prefill": (
        "461ff4301749be26",
        "42cdd1981b505c03",
    ),
    "srt/layers/attention/linear/gdn_backend.py:GDNAttnBackend._prefill_gates": "75a88d86adb8cbea",
    "srt/layers/radix_linear_attention.py:RadixLinearAttention.forward": "70ae36d7dff0686f",
    "srt/layers/radix_linear_attention.py:_linear_attention_with_output_impl": "433070c77e669c68",
    "srt/models/qwen3_5.py:Qwen3_5GatedDeltaNet.forward": "604e1c2afbed13db",
}


def sglang_unsupported(binding) -> str:
    """What this SGLang lacks for the slot, or "" when it has every read it needs."""
    from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
        FlashInferGDNKernel,
    )

    for name in ("_run_prefill", "_prefill_state_dtype"):
        if not hasattr(FlashInferGDNKernel, name):
            return f"no FlashInferGDNKernel.{name}"
    if not hasattr(binding, "prefill_view"):
        return "no linear prefill view"
    return ""


def _state_copies():
    from types import SimpleNamespace

    try:
        from sglang.kernels.ops.mamba.gdn_extend_state_copies import (
            gather_pool_rows,
            gather_pool_rows_ok,
            scatter_pool_rows,
            scatter_pool_rows_ok,
        )
        from sglang.srt.layers.attention.linear.gdn_extend_track import (
            track_conv_windows,
            track_ssm_states_in_one_launch,
        )
    except ImportError:
        return None
    return (
        SimpleNamespace(
            gather_pool_rows=gather_pool_rows,
            gather_pool_rows_ok=gather_pool_rows_ok,
            scatter_pool_rows=scatter_pool_rows,
            scatter_pool_rows_ok=scatter_pool_rows_ok,
        ),
        SimpleNamespace(
            track_conv_windows=track_conv_windows,
            track_ssm_states_in_one_launch=track_ssm_states_in_one_launch,
        ),
    )


def linear_backend():
    """The forward's linear-attention backend, as SGLang's GDN layer reaches it."""
    from sglang.srt.model_executor.forward_context import get_attn_backend

    backend = get_attn_backend()
    return getattr(backend, "linear_attn_backend", backend)


def layer_unsupported(attn, front) -> str:
    if not (
        attn.num_q_heads == attn.num_k_heads
        and attn.head_q_dim == attn.head_k_dim
        and front.supported(
            num_k_heads=attn.num_k_heads,
            num_v_heads=attn.num_v_heads,
            head_k_dim=attn.head_k_dim,
            head_v_dim=attn.head_v_dim,
            conv_width=attn.conv_weights.shape[-1],
            activation=attn.activation,
        )
    ):
        return "layer shape"
    w = attn.conv_weights.reshape(attn.conv_weights.shape[0], -1)
    if w.dtype != torch.bfloat16 or not w.is_contiguous() or w.data_ptr() % 16:
        return "conv weight layout"
    if attn.bias is not None and (
        attn.bias.dtype != torch.bfloat16 or not attn.bias.is_contiguous()
    ):
        return "conv bias"
    return ""


def unsupported(view, linear) -> str:
    from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
        FlashInferGDNKernel,
    )

    if not isinstance(linear.kernel_dispatcher.extend_kernel, FlashInferGDNKernel):
        return "not the FlashInfer prefill kernel"
    if view.is_graph or view.has_mis:
        return "linear extend inside the graph or multi-item layout"
    if not view.state.temporal.is_contiguous() or any(
        not x.is_contiguous() for x in view.state.conv
    ):
        return "strided committed state pool"
    if (
        view.track_ssm_recompute_dst is not None
        and view.track_ssm_recompute_dst.numel()
    ):
        return "recomputed checkpoint layout"
    for index in (view.query_start_loc, view.state_indices):
        if index.dtype != torch.int32 or not index.is_contiguous():
            return "index dtype / layout"
    if any(x.dtype != torch.bfloat16 for x in view.state.conv):
        return "conv state dtype"
    return ""


def _track_metadata(view, h_src=None):
    from types import SimpleNamespace

    return SimpleNamespace(
        track_conv_indices=view.track_conv_indices,
        conv_states_mask_indices=view.track_conv_dst,
        track_ssm_h_src=view.track_ssm_h_src if h_src is None else h_src,
        track_ssm_h_dst=view.track_ssm_h_dst,
        track_ssm_final_src=view.track_ssm_final_src,
        track_ssm_final_dst=view.track_ssm_final_dst,
        track_ssm_recompute_dst=view.track_ssm_recompute_dst,
    )


def _flashinfer_extend(
    kernel, q, k, v, g, beta, ssm_states, view, output, copies, snapshot=None
):
    from sglang.srt.layers.attention.linear.kernels.state_pool_io import (
        gather_pool_states,
        scatter_pool_states,
        state_pool_indices,
    )

    tokens, heads, head_dim = v.shape
    alpha = torch.exp(g[0].to(torch.float32))
    beta = beta[0].to(torch.float32)
    indices = view.state_indices
    state_dtype = kernel._prefill_state_dtype(ssm_states.dtype)
    pool_rows = copies is not None and copies.gather_pool_rows_ok(
        states=ssm_states, indices=indices, dtype=state_dtype
    )
    if pool_rows:
        initial = copies.gather_pool_rows(states=ssm_states, indices=indices)
    else:
        slots, live_rows, initial = gather_pool_states(ssm_states, indices)
        initial = initial.to(state_dtype)
    plan = (
        None
        if snapshot is None
        else snapshot.plan(view, kernel, max(q.shape[1], v.shape[1]))
    )
    if plan is not None:
        from sgl_nvidia_mega.qwen3_5.gdn import fi_track_snapshot

        # only the entries the track copy reads, one row each, in its order
        checkpoints = initial.new_empty((plan.slots.numel(), *initial.shape[1:]))
        output, final = fi_track_snapshot.run_prefill(
            kernel,
            q=q,
            k=k,
            v=v,
            alpha=alpha,
            beta=beta,
            initial_state=initial,
            cu_seqlens=view.query_start_loc.to(torch.int64),
            output=None if output is None else output[0],
            state_checkpoints=checkpoints,
            checkpoint_cu_starts=view.state_checkpoint_cu_starts,
            checkpoint_every_n_tokens=view.state_checkpoint_every_n_tokens,
            checkpoint_rows=plan.rows,
        )
        h_src = plan.slots
    else:
        checkpoints = (
            initial.new_empty((view.num_state_checkpoints, *initial.shape[1:]))
            if view.num_state_checkpoints > 0
            else None
        )
        output, final = kernel._run_prefill(
            q=q,
            k=k,
            v=v,
            alpha=alpha,
            beta=beta,
            initial_state=initial,
            cu_seqlens=view.query_start_loc.to(torch.int64),
            output=None if output is None else output[0],
            state_checkpoints=checkpoints,
            checkpoint_cu_starts=view.state_checkpoint_cu_starts,
            checkpoint_every_n_tokens=view.state_checkpoint_every_n_tokens,
        )
        h_src = None
    if pool_rows and copies.scatter_pool_rows_ok(
        states=ssm_states, indices=indices, updates=final
    ):
        copies.scatter_pool_rows(states=ssm_states, indices=indices, updates=final)
    else:
        if pool_rows:
            slots = state_pool_indices(indices)
            live_rows = (indices > 0).view(-1, *((1,) * (ssm_states.ndim - 1)))
        scatter_pool_states(ssm_states, slots, live_rows, final)
    h = checkpoints.unsqueeze(0) if checkpoints is not None else None
    return output.view(1, tokens, heads, head_dim), h, h_src


def extend(attn, mixed, a, b, view, forward_batch, linear, output=None, snapshot=None):
    from sglang.kernels.ops.attention.linear.fused_gdn_gating import fused_gdn_gating

    from sgl_nvidia_mega.qwen3_5.sm103.gdn import prefill_front as front

    cache = linear.state_layer_cache(attn.layer_id)
    conv_states, ssm = cache.conv[0], cache.temporal
    if front.eligible(
        mixed,
        attn.conv_weights,
        attn.bias,
        conv_states,
        view.query_start_loc,
        view.state_indices,
    ):
        return None
    helpers = _state_copies()
    copies, track = helpers if helpers is not None else (None, None)
    metadata = _track_metadata(view) if view.has_track else None
    if view.has_track:
        if track is not None:
            track.track_conv_windows(
                conv_states=conv_states, mixed_qkv=mixed, metadata=metadata
            )
        else:
            conv_states[view.track_conv_dst] = mixed[view.track_conv_indices].transpose(
                1, 2
            )
    query, key, value = front.gdn_prefill_front(
        mixed,
        attn.conv_weights,
        attn.bias,
        conv_states,
        view.query_start_loc,
        view.state_indices,
        forward_batch.extend_prefix_lens > 0,
        num_k_heads=attn.num_k_heads,
        num_v_heads=attn.num_v_heads,
        activation=attn.activation,
    )
    if front.gates_eligible(attn.A_log, a, b, attn.dt_bias):
        g, beta = front.gdn_prefill_gates(attn.A_log, a, b, attn.dt_bias)
    else:
        g, beta = fused_gdn_gating(attn.A_log, a, b, attn.dt_bias)
    core, h, h_src = _flashinfer_extend(
        linear.kernel_dispatcher.extend_kernel,
        query,
        key,
        value,
        g,
        beta,
        ssm,
        view,
        output,
        copies,
        snapshot,
    )
    if h_src is not None:
        metadata = _track_metadata(view, h_src)
    if view.has_track and not (
        track is not None
        and track.track_ssm_states_in_one_launch(
            ssm_states=ssm, h=h, h_track_buf=None, metadata=metadata
        )
    ):
        if metadata.track_ssm_h_src.numel():
            ssm[view.track_ssm_h_dst] = h.squeeze(0)[metadata.track_ssm_h_src].to(
                ssm.dtype, copy=False
            )
        if view.track_ssm_final_src.numel():
            ssm[view.track_ssm_final_dst] = ssm[view.track_ssm_final_src]
    return core


def core(attn, mixed, a, b, output, binding, declined, snapshot=None) -> None:
    from sglang.srt.layers.radix_linear_attention import (
        _linear_attention_with_output_impl,
        get_attn_backend,
    )
    from sglang.srt.model_executor.runner_utils.forward_batch import get_forward_batch

    forward_batch = get_forward_batch()
    real = forward_batch.global_num_token_non_padded_cpu
    real = mixed.shape[0] if real is None else min(real, mixed.shape[0])
    linear = linear_backend()
    view = binding.prefill_view() if real else None
    why = unsupported(view, linear) if real else ""
    if real and not why:
        logical = output[:, :real]
        got = extend(
            attn,
            mixed[:real],
            a[:real],
            b[:real],
            view,
            forward_batch,
            linear,
            logical,
            snapshot,
        )
        if got is not None:
            # FlashInfer writes into the logical rows; another kernel's output is copied
            if got.data_ptr() != logical.data_ptr():
                logical.copy_(got)
            output[:, real:].zero_()
            return
        why = f"layer {attn.layer_id}'s tensors"
    if why and why not in declined:
        declined.add(why)
        logger.info(
            "NVIDIA megakernel: Qwen3.5 slot gdn_prefill_front: SGLang's GDN core "
            "runs this step (%s)",
            why,
        )
    if forward_batch.global_num_token_non_padded_cpu is None:
        got = get_attn_backend().forward(
            layer=attn, forward_batch=forward_batch, mixed_qkv=mixed, a=a, b=b
        )
        output.copy_(got.reshape(output.shape))
        return
    _linear_attention_with_output_impl(mixed, a, b, output, attn, forward_batch)


def step(gdn, hidden, run_core, norm=None):
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
        qwen3_5_gdn_prefill_projection_views,
    )

    qkvz, ba = gdn._forward_input_proj(hidden)
    mixed, z, b, a = qwen3_5_gdn_prefill_projection_views(
        qkvz,
        ba,
        gdn.num_k_heads // gdn.attn_tp_size,
        gdn.num_v_heads // gdn.attn_tp_size,
        gdn.head_k_dim,
        gdn.head_v_dim,
    )
    attn = gdn.attn
    output = torch.empty(
        (1, mixed.shape[0], attn.num_v_heads, attn.head_v_dim),
        dtype=mixed.dtype,
        device=mixed.device,
    )
    run_core(attn, mixed, a, b, output)
    core = output.reshape(-1, output.shape[-1])
    core = gdn.norm(core, z) if norm is None else norm(gdn, core, z)
    return gdn.out_proj(core.reshape(qkvz.shape[0], -1))[0]
