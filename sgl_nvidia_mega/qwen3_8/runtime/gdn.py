# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/qwen3_5.py (Qwen3_5GatedDeltaNet.forward) and
# python/sglang/srt/layers/attention/linear/gdn_backend.py (GDNAttnBackend.forward_extend,
# its circular ReplaySSM target-verify branch, and _replayssm_target_verify).
"""The GDN layer step of a covered forward.

Target verify with the circular ReplaySSM spec rings (``--enable-linear-replayssm-spec``)
runs here, slot ``gdn_native_pdl``: the input projection, the q/k/v/z/b/a split, the
causal conv update into the verify's intermediate windows, the q/k/v split, the ReplaySSM
verify core (the plugin's native kernel: SGLang's circular verify kernel launched with
PDL), the gated RMSNorm and the output projection. With slot ``gdn_verify_glue``, a
verify of 2-8 requests runs everything between the input and the output projection as
one launch per layer (``kernels/ops/attention/linear/gdn_fused_verify_glue.py``). With
slot ``gdn_front``, a verify of one request (4 rows) runs two launches before the output
projection: the input projection with the causal conv update in its epilogue, then the
MMA ReplaySSM core with the gated RMSNorm (``kernels/ops/attention/gdn_front/mk_*.py``).
Every other mode, and verify without the slots, runs SGLang's ``GatedDeltaNet.forward``.

What the verify reads and writes (the pools are SGLang's): the conv states by state
slot (the conv update also writes them; SGLang's commit restores them), the
intermediate conv windows by verify row, the checkpoint (``temporal``) read-only, the
ring entries of this verify by request slot, and the cursors read-only (SGLang's commit
advances them)."""

from __future__ import annotations

import torch

from sgl_nvidia_mega.qwen3_8.runtime import bf16_gemm_pins

STOCK = "stock"
# slot gdn_verify_glue: 2-8 requests of 4 rows; one program per (request, value head)
GLUE_MAX_ROWS = 32
GLUE_NORM_ACTIVATIONS = ("swish", "silu", "sigmoid")
# slot gdn_front: one request of 4 rows at hidden 2560; the core's gate is sigmoid(z)
FRONT_ROWS = 4
FRONT_HIDDEN = 2560

SGLANG_READS = {
    "srt/models/qwen3_5.py": (
        "_forward_input_proj",
        "_fused_in_proj_weight",
        "num_k_heads",
        "num_v_heads",
        "head_k_dim",
        "head_v_dim",
        "attn_tp_size",
        "norm",
        "out_proj",
        "attn",
        "forward",
        "_gdn_decode_fused_proj_conv",
        "_is_xpu",
        "_is_cpu",
        "_is_cuda",
        "_GDN_FUSED_QKVZBA_RATIOS",
        "_fp8_tuple_input",
        "_layer_norm_fwd",
        "get_exec",
        "fix_query_key_value_ordering",
        "qwen3_5_gdn_prefill_projection_views",
        "fused_qkvzba_split_reshape_cat_contiguous",
    ),
    "kernels/ops/attention/fla/layernorm_gated.py": (
        "group_size",
        "norm_before_gate",
        "activation",
        "eps",
        "bias",
        "weight",
    ),
    "srt/layers/attention/linear/gdn_backend.py": (
        "forward_metadata",
        "retrieve_next_token",
        "retrieve_next_sibling",
        "retrieve_parent_token",
        "verify_intermediate_state_indices",
        "_use_strided_target_verify_qkv",
        "MAX_FUSED_QKV_SPLIT_DIM",
    ),
    "srt/layers/attention/hybrid_linear_attn_backend.py": (
        "mamba_cache_indices",
        "state_layer_cache",
        "state_pool",
        "req_to_token_pool",
    ),
    "srt/models/qwen4_exp.py": ("Qwen4ExpLinearDecoderLayer",),
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "srt/layers/radix_linear_attention.py": (
        "conv_weights",
        "bias",
        "activation",
        "layer_id",
        "A_log",
        "dt_bias",
        "q_dim",
        "k_dim",
        "v_dim",
        "num_q_heads",
        "head_q_dim",
        "num_v_heads",
        "head_v_dim",
    ),
    "kernels/ops/attention/triton_gdn_fused_proj.py": (
        "fused_qkvzba_split_reshape_cat_contiguous",
        "fused_qkv_split_gdn_prefill",
    ),
    "kernels/ops/mamba/causal_conv1d_triton.py": (
        "causal_conv1d_update",
        "PAD_SLOT_ID",
    ),
    "srt/mem_cache/memory_pool.py": (
        "conv",
        "temporal",
        "intermediate_conv_window",
        "replayssm_d",
        "replayssm_k",
        "replayssm_g",
        "replayssm_rawv",
        "replayssm_rawk",
        "replayssm_beta",
        "replayssm_spec_write_pos",
        "replayssm_cache_base",
        "replayssm_is_flush",
        "replayssm_spec_fold",
        "replayssm_is_kda",
    ),
    "srt/layers/attention/linear/utils.py": (
        "select_verify_intermediate_state_indices",
        "pp_spec_stable_rows_enabled",
    ),
}

# forward mirrors GatedDeltaNet.forward when BF16 pins are on
SGLANG_MIRRORS = {
    "srt/models/qwen3_5.py:Qwen3_5GatedDeltaNet.forward": "ff40a6653f83b8b6",
}

# Reads and mirrors that need an SGLang change newer than the unit job's pin: the
# contract test skips each, naming the change, only where the installed SGLang lacks it.
SGLANG_REQUIRES = {
    "srt/models/qwen3_5.py:Qwen3_5GatedDeltaNet.forward": (
        "sglang#44017",
        ("a5e9e3bd054ff18f",),
    ),
}


def linear_backend():
    from sglang.srt.model_executor.forward_context import get_attn_backend

    backend = get_attn_backend()
    return getattr(backend, "linear_attn_backend", backend)


def verify_reasons(model) -> str:
    """'' when every GDN layer's verify can run the native core, else why not."""
    from sglang.srt.models.qwen4_exp import Qwen4ExpLinearDecoderLayer

    m = model.model
    for i in range(m.start_layer, m.end_layer):
        layer = m.layers[i]
        if not isinstance(layer, Qwen4ExpLinearDecoderLayer):
            continue
        attn = layer.linear_attn
        h, hv = (
            attn.num_k_heads // attn.attn_tp_size,
            attn.num_v_heads // attn.attn_tp_size,
        )
        if (h, hv) not in ((4, 12), (16, 48)):
            return f"GDN heads per rank {(h, hv)}, the kernel takes (4, 12) or (16, 48)"
        if attn.head_k_dim != 128 or attn.head_v_dim != 128:
            return "GDN head dims are not 128"
        if attn.num_v_heads // attn.num_k_heads != 3:
            return "the fused q/k/v/z/b/a split needs 3 value heads per key head"
    return ""


def pool_reasons(backend, layer_id: int, tokens_per_request: int) -> str:
    """'' when the pool holds the circular ReplaySSM spec rings the core takes."""
    pool = backend.state_pool
    cache = backend.state_layer_cache(layer_id)
    if cache.replayssm_d is None or pool.replayssm_cache_base is None:
        return "the pool has no circular ReplaySSM spec rings (--enable-linear-replayssm-spec)"
    if pool.replayssm_spec_fold or pool.replayssm_is_kda:
        return "the pool's ReplaySSM rings fold every commit or are KDA's"
    if cache.replayssm_d.shape[-2] != 16:
        return (
            f"ReplaySSM ring length {cache.replayssm_d.shape[-2]}, the kernel takes 16"
        )
    if cache.temporal.dtype not in (torch.bfloat16, torch.float32):
        return f"GDN checkpoint dtype {cache.temporal.dtype}"
    if tokens_per_request != 4:
        return f"{tokens_per_request} verify rows a request, the kernel takes 4"
    return ""


def step(executor, layer, x, st):
    """The layer's mixer output ([rows, hidden], before the stage's reduction)."""
    fb = st.fb
    if executor.gdn_prefill is not None and st.step.mode == "prefill":
        from sgl_nvidia_mega.qwen3_8.runtime import gdn_prefill

        out = gdn_prefill.extend(executor, layer, x, st)
        if out is not None:
            return out
    if executor.gdn_ucache is not None and fb.forward_mode.is_target_verify():
        from sgl_nvidia_mega.qwen3_8.runtime import gdn_ucache

        return gdn_ucache.verify(executor, layer.linear_attn, x, st, st.view)
    if fb.forward_mode.is_target_verify() and executor.gdn_front_ready:
        out = _verify_front(executor, layer.linear_attn, x, st)
        if out is not None:
            return out
    if fb.forward_mode.is_target_verify() and executor.gdn_glue_ready:
        out = _verify_glue(executor, layer.linear_attn, x, st)
        if out is not None:
            return out
    if not (executor.gdn_pdl_ready and fb.forward_mode.is_target_verify()):
        st.ran("gdn", STOCK)
        if bf16_gemm_pins.routed():
            return forward(layer.linear_attn, x, fb)
        return layer.linear_attn(x, fb)
    return _verify(executor, layer.linear_attn, x, st)


def forward(gdn, hidden_states, forward_batch):
    """``Qwen3_5GatedDeltaNet.forward`` with its input and output projections through
    ``bf16_gemm_pins`` (the stock GDN step with BF16 pins on)."""
    import triton
    from sglang.srt.models.qwen3_5 import (
        _GDN_FUSED_QKVZBA_RATIOS,
        _fp8_tuple_input,
        _gdn_decode_fused_proj_conv,
        _is_cpu,
        _is_cuda,
        _is_xpu,
        _layer_norm_fwd,
        fused_qkvzba_split_reshape_cat_contiguous,
        get_exec,
        qwen3_5_gdn_prefill_projection_views,
    )

    projected_states_qkvz, projected_states_ba = bf16_gemm_pins.in_proj(
        gdn, hidden_states
    )

    use_strided_prefill_z = False
    packed_verify = False
    if forward_batch.forward_mode.is_target_verify():
        from sglang.srt.model_executor.forward_context import get_attn_backend

        backend = get_attn_backend()
        backend = getattr(backend, "linear_attn_backend", backend)
        packed_verify = getattr(backend, "supports_packed_verify", False)
    use_fused_proj_conv = (
        _gdn_decode_fused_proj_conv
        and not (_is_xpu and get_exec().mamba.linear_attn_backend == "intel_xpu")
        and (forward_batch.forward_mode.is_decode() or packed_verify)
        and isinstance(projected_states_qkvz, torch.Tensor)
        and isinstance(projected_states_ba, torch.Tensor)
    )
    use_fused_contiguous_unpack = (
        gdn.num_v_heads // gdn.num_k_heads in _GDN_FUSED_QKVZBA_RATIOS
    )
    if use_fused_proj_conv:
        mixed_qkv = (projected_states_qkvz, projected_states_ba)
        z = None
        b = projected_states_ba
        a = projected_states_ba
    elif use_fused_contiguous_unpack:
        if _is_cpu:
            num_k_heads_tp = gdn.num_k_heads // gdn.attn_tp_size
            num_v_heads_tp = gdn.num_v_heads // gdn.attn_tp_size
        else:
            num_k_heads_tp = triton.cdiv(gdn.num_k_heads, gdn.attn_tp_size)
            num_v_heads_tp = triton.cdiv(gdn.num_v_heads, gdn.attn_tp_size)
        use_strided_prefill_z = (
            _is_cuda
            and forward_batch.logical_forward_mode.is_extend_without_speculative()
        )
        split_fn = (
            qwen3_5_gdn_prefill_projection_views
            if use_strided_prefill_z
            else fused_qkvzba_split_reshape_cat_contiguous
        )
        mixed_qkv, z, b, a = split_fn(
            projected_states_qkvz,
            projected_states_ba,
            num_k_heads_tp,
            num_v_heads_tp,
            gdn.head_k_dim,
            gdn.head_v_dim,
        )
    else:
        query, key, value, z, b, a = gdn.fix_query_key_value_ordering(
            projected_states_qkvz, projected_states_ba
        )
        b = b.contiguous()
        a = a.contiguous()
        query, key, value = map(
            lambda x: x.reshape(x.shape[0], -1), (query, key, value)
        )
        mixed_qkv = torch.cat((query, key, value), dim=-1)

    attn_result = gdn.attn(forward_batch, mixed_qkv=mixed_qkv, a=a, b=b)
    if use_fused_proj_conv:
        if not isinstance(attn_result, tuple) or len(attn_result) != 2:
            raise RuntimeError(
                "Fused GDN projection/Conv1D backend must return (core_attn_out, z)"
            )
        core_attn_out, z = attn_result
    else:
        core_attn_out = attn_result
    assert z is not None

    z_shape_og = z.shape
    core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
    if use_strided_prefill_z:
        z_flat_shape = (z.numel() // z.shape[-1], z.shape[-1])
    else:
        z = z.reshape(-1, z.shape[-1])
        z_flat_shape = z.shape

    if core_attn_out.shape != z_flat_shape:  # DP-attention padding
        core_attn_out_pad = z.new_zeros(z_flat_shape)
        core_attn_out_pad[: core_attn_out.shape[0], :] = core_attn_out
        core_attn_out = core_attn_out_pad

    nv = gdn.num_v_heads // gdn.attn_tp_size
    if not use_strided_prefill_z and _fp8_tuple_input(gdn.out_proj, len(z) // nv):
        fp8, _, _ = _layer_norm_fwd(
            core_attn_out,
            gdn.norm.weight,
            None,
            gdn.norm.eps,
            z=z,
            is_rms_norm=True,
            activation=gdn.norm.activation,
            quant_heads=nv,
        )
        return gdn.out_proj(fp8)[0]

    core_attn_out = gdn.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(z_shape_og)
    core_attn_out = core_attn_out.reshape(
        *core_attn_out.shape[:-2],
        core_attn_out.shape[-2] * core_attn_out.shape[-1],
    )
    return bf16_gemm_pins.linear(gdn.out_proj, core_attn_out)


def glue_reasons(model) -> str:
    """'' when every GDN layer fits the one-launch verify glue, else why not."""
    from sglang.srt.models.qwen4_exp import Qwen4ExpLinearDecoderLayer

    reason = verify_reasons(model)
    if reason:
        return reason
    m = model.model
    for i in range(m.start_layer, m.end_layer):
        layer = m.layers[i]
        if not isinstance(layer, Qwen4ExpLinearDecoderLayer):
            continue
        gdn, attn = layer.linear_attn, layer.linear_attn.attn
        norm = gdn.norm
        if gdn.attn_tp_size != 1:
            return "the kernel takes TP1"
        if (
            norm.group_size is not None
            or not norm.norm_before_gate
            or norm.bias is not None
        ):
            return "the gated RMSNorm is grouped, gate-first or has a bias"
        if norm.activation not in GLUE_NORM_ACTIVATIONS:
            return f"gated RMSNorm activation {norm.activation!r}"
        if attn.bias is not None or attn.activation not in ("silu", "swish"):
            return "the causal conv has a bias or no SiLU"
        if (attn.num_q_heads, attn.head_q_dim) != (attn.num_k_heads, attn.head_k_dim):
            return "q and k heads differ"
    return ""


def glue_pool_reasons(backend, layer_id: int, tokens_per_request: int) -> str:
    """'' when the pool's states fit the glue (pool_reasons plus BF16 states)."""
    reason = pool_reasons(backend, layer_id, tokens_per_request)
    if reason:
        return reason
    cache = backend.state_layer_cache(layer_id)
    window = cache.intermediate_conv_window[0]
    if cache.temporal.dtype != torch.bfloat16:
        return f"GDN checkpoint dtype {cache.temporal.dtype}, the kernel takes BF16"
    if cache.conv[0].dtype != torch.bfloat16 or window.dtype != torch.bfloat16:
        return "the conv states are not BF16"
    if cache.conv[0].dim() != 3 or cache.conv[0].shape[-1] != 3 or window.dim() != 4:
        return "the conv state is not width 4"
    return ""


def front_reasons(model) -> str:
    """'' when every GDN layer fits the verify front and its fused-norm core, else why
    not (glue_reasons plus SGLang's packed BF16 in_proj weight and a sigmoid gate)."""
    from sglang.srt.models.qwen4_exp import Qwen4ExpLinearDecoderLayer

    reason = glue_reasons(model)
    if reason:
        return reason
    m = model.model
    for i in range(m.start_layer, m.end_layer):
        layer = m.layers[i]
        if not isinstance(layer, Qwen4ExpLinearDecoderLayer):
            continue
        gdn, attn = layer.linear_attn, layer.linear_attn.attn
        weight = gdn._fused_in_proj_weight
        rows = attn.q_dim + attn.k_dim + 2 * attn.v_dim + 2 * attn.num_v_heads
        shape = (rows, FRONT_HIDDEN)  # [q|k|v|z|b|a] rows
        if weight is None or weight.dtype != torch.bfloat16:
            return "no packed BF16 in_proj weight (SGLang's _fused_in_proj_weight)"
        if tuple(weight.shape) != shape:
            return (
                f"packed in_proj weight {tuple(weight.shape)}, the front takes {shape}"
            )
        if gdn.norm.activation != "sigmoid":
            return f"gated RMSNorm activation {gdn.norm.activation!r}, not sigmoid"
        if gdn.norm.weight.dtype not in (torch.bfloat16, torch.float32):
            return f"gated RMSNorm weight dtype {gdn.norm.weight.dtype}"
        if attn.A_log.dtype != torch.float32:
            return f"A_log dtype {attn.A_log.dtype}, the core takes FP32"
        if attn.dt_bias.dtype not in (torch.bfloat16, torch.float32):
            return f"dt_bias dtype {attn.dt_bias.dtype}"
    return ""


def front_pool_reasons(backend, layer_id: int, tokens_per_request: int) -> str:
    """'' when the pool's states fit the front and its core (glue_pool_reasons plus
    BF16 rings that keep low parts)."""
    reason = glue_pool_reasons(backend, layer_id, tokens_per_request)
    if reason:
        return reason
    cache = backend.state_layer_cache(layer_id)
    if (
        cache.replayssm_d.dtype != torch.bfloat16
        or cache.replayssm_rawv is None
        or cache.replayssm_rawk is None
        or cache.replayssm_beta is not None
    ):
        return "the ReplaySSM rings are not BF16 with low parts"
    return ""


def _verify_front(executor, gdn, x, st):
    """``Qwen3_5GatedDeltaNet.forward``'s fused-front verify of one request: the input
    projection with the causal conv update, the ReplaySSM core with the gated RMSNorm
    (``GDNAttnBackend.forward_inproj_conv``) and the output projection; None outside
    one request of 4 rows or for a tree verify."""
    fb = st.fb
    rows = x.shape[0]
    if rows != FRONT_ROWS or fb.spec_info.draft_token_num != FRONT_ROWS:
        return None
    from sglang.kernels.ops.mamba.causal_conv1d_triton import PAD_SLOT_ID
    from sglang.srt.layers.attention.linear.utils import (
        select_verify_intermediate_state_indices,
    )

    from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.gdn_front import (
        mk_core,
        mk_front,
    )
    from sgl_nvidia_mega.qwen3_8.runtime import smallm_gemm

    backend = linear_backend()
    md = backend.forward_metadata
    if (
        md.retrieve_next_token is not None
        or md.retrieve_next_sibling is not None
        or md.retrieve_parent_token is not None
    ):
        return None  # tree verify: the conv walks parent links
    layer = gdn.attn
    cache = backend.state_layer_cache(layer.layer_id)
    pool = backend.state_pool
    cache_indices = md.mamba_cache_indices
    window_rows = st.scratch.get("gdn.front_window_rows")
    if window_rows is None:
        # the same for every layer of the step: one launch, not one a layer
        window_rows = st.scratch["gdn.front_window_rows"] = (
            select_verify_intermediate_state_indices(
                backend.verify_intermediate_state_indices,
                fb.req_pool_indices,
                cache_indices[:1] >= 0,
                backend.req_to_token_pool.size,
            )[:1]
        )
    packed, qkv = mk_front.run(
        x,
        gdn._fused_in_proj_weight,
        layer.conv_weights,
        None,
        cache.conv[0],
        cache_indices[:1],
        cache.intermediate_conv_window[0],
        window_rows,
        pad_slot=PAD_SLOT_ID,
    )
    z_start = layer.q_dim + layer.k_dim + layer.v_dim
    b_start = z_start + layer.v_dim
    hv = layer.num_v_heads
    raw = x.new_empty(rows, hv, layer.head_v_dim)
    normalized = torch.empty_like(raw)
    mk_core.core(
        qkv,
        packed[:, b_start + hv :],
        packed[:, b_start : b_start + hv],
        layer.A_log,
        layer.dt_bias,
        cache.temporal,
        cache.replayssm_d,
        cache.replayssm_k,
        cache.replayssm_g,
        cache.replayssm_rawv,
        cache.replayssm_rawk,
        cache_indices,
        fb.req_pool_indices,
        pool.replayssm_spec_write_pos,
        pool.replayssm_cache_base,
        pool.replayssm_is_flush,
        raw,
        z=packed[:, z_start:b_start],
        norm_weight=gdn.norm.weight,
        normalized=normalized,
        eps=gdn.norm.eps,
    )
    st.ran("gdn", "front")
    core = normalized.view(rows, -1)
    out = smallm_gemm.out_proj(executor, gdn.out_proj, core, st, "gdn.out_proj")
    if out is None:
        out = bf16_gemm_pins.linear(gdn.out_proj, core)
    return out


def _verify_glue(executor, gdn, x, st):
    """``Qwen3_5GatedDeltaNet._forward_fused_verify_glue``: the input projection, one
    glue launch (split, causal conv update, ReplaySSM verify, gated RMSNorm) and the
    output projection; None outside 2-8 requests or for a tree verify."""
    from sglang.srt.layers.attention.linear.utils import (
        pp_spec_stable_rows_enabled,
        select_verify_intermediate_state_indices,
    )

    from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.linear.gdn_fused_verify_glue import (
        gdn_fused_verify_glue,
    )
    from sgl_nvidia_mega.qwen3_8.runtime import smallm_gemm

    fb = st.fb
    rows = x.shape[0]
    tokens = fb.spec_info.draft_token_num
    if not (2 * tokens <= rows <= GLUE_MAX_ROWS and rows % tokens == 0):
        return None
    backend = linear_backend()
    md = backend.forward_metadata
    if (
        md.retrieve_next_token is not None
        or md.retrieve_next_sibling is not None
        or md.retrieve_parent_token is not None
    ):
        return None  # tree verify: the conv walks parent links
    layer = gdn.attn
    qkvz, ba = bf16_gemm_pins.in_proj(gdn, x)
    cache = backend.state_layer_cache(layer.layer_id)
    pool = backend.state_pool
    cache_indices = md.mamba_cache_indices
    batch = rows // tokens
    if pp_spec_stable_rows_enabled():
        rows_state = select_verify_intermediate_state_indices(
            backend.verify_intermediate_state_indices,
            fb.req_pool_indices,
            cache_indices[:batch] >= 0,
            backend.req_to_token_pool.size,
        )
    else:
        # as the source: the stock call evaluates the >= 0 mask (a launch a layer) and discards it
        rows_state = backend.verify_intermediate_state_indices
    core = gdn_fused_verify_glue(
        mixed_qkvz=qkvz,
        mixed_ba=ba,
        conv_state=cache.conv[0],
        conv_weight=layer.conv_weights,
        conv_bias=None,
        conv_activation=layer.activation,
        intermediate_conv_window=cache.intermediate_conv_window[0],
        intermediate_state_indices=rows_state,
        A_log=layer.A_log,
        dt_bias=layer.dt_bias,
        norm_weight=gdn.norm.weight,
        norm_eps=gdn.norm.eps,
        norm_activation=gdn.norm.activation,
        checkpoint_state=cache.temporal,
        d_cache=cache.replayssm_d,
        k_cache=cache.replayssm_k,
        g_cache=cache.replayssm_g,
        rawv_cache=cache.replayssm_rawv,
        rawk_cache=cache.replayssm_rawk,
        beta_cache=cache.replayssm_beta,
        cache_indices=cache_indices,
        replay_indices=fb.req_pool_indices,
        write_pos=pool.replayssm_spec_write_pos,
        cache_base=pool.replayssm_cache_base,
        is_flush=pool.replayssm_is_flush,
        num_k_heads=layer.num_k_heads,
        head_k_dim=layer.head_k_dim,
        draft_token_num=tokens,
    )
    st.ran("gdn", "verify_glue")
    out = smallm_gemm.out_proj(executor, gdn.out_proj, core, st, "gdn.out_proj")
    if out is None:
        out = bf16_gemm_pins.linear(gdn.out_proj, core)
    return out


def _verify(executor, gdn, x, st):
    """``Qwen3_5GatedDeltaNet.forward`` for a target verify without packed verify."""
    import triton
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
        fused_qkvzba_split_reshape_cat_contiguous,
    )

    from sgl_nvidia_mega.qwen3_8.runtime import smallm_gemm

    fb = st.fb
    qkvz, ba = bf16_gemm_pins.in_proj(gdn, x)
    nk = triton.cdiv(gdn.num_k_heads, gdn.attn_tp_size)
    nv = triton.cdiv(gdn.num_v_heads, gdn.attn_tp_size)
    mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat_contiguous(
        qkvz, ba, nk, nv, gdn.head_k_dim, gdn.head_v_dim
    )
    core = _verify_core(executor, gdn.attn, fb, mixed_qkv, a, b)
    st.ran("gdn", "native_pdl")
    z_shape = z.shape
    core = core.reshape(-1, core.shape[-1])
    z = z.reshape(-1, z.shape[-1])
    if core.shape != z.shape:  # DP-attention padding
        padded = z.new_zeros(z.shape)
        padded[: core.shape[0], :] = core
        core = padded
    core = gdn.norm(core, z)
    core = core.reshape(z_shape)
    core = core.reshape(*core.shape[:-2], core.shape[-2] * core.shape[-1])
    out = smallm_gemm.out_proj(executor, gdn.out_proj, core, st, "gdn.out_proj")
    if out is None:
        out = bf16_gemm_pins.linear(gdn.out_proj, core)
    return out


def _verify_core(executor, layer, fb, mixed_qkv, a, b):
    """``GDNAttnBackend.forward_extend``'s circular-ReplaySSM target-verify branch
    with the native PDL core; returns [rows, HV, V]."""
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
        fused_qkv_split_gdn_prefill,
    )
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
    from sglang.srt.layers.attention.linear.utils import (
        select_verify_intermediate_state_indices,
    )

    backend = linear_backend()
    seq_len = mixed_qkv.shape[0]
    md = backend.forward_metadata
    query_start_loc = md.query_start_loc
    cache_indices = md.mamba_cache_indices
    cache = backend.state_layer_cache(layer.layer_id)
    window = cache.intermediate_conv_window[0]
    rows_state = select_verify_intermediate_state_indices(
        backend.verify_intermediate_state_indices,
        fb.req_pool_indices,
        cache_indices[: query_start_loc.shape[0] - 1] >= 0,
        backend.req_to_token_pool.size,
    )
    tokens = fb.spec_info.draft_token_num
    batch = seq_len // tokens
    conv_in = mixed_qkv.view(batch, tokens, -1).transpose(1, 2)
    conv_out = causal_conv1d_update(
        conv_in,
        cache.conv[0],
        layer.conv_weights,
        layer.bias,
        layer.activation,
        conv_state_indices=cache_indices[:batch],
        intermediate_conv_window=window,
        intermediate_state_indices=rows_state[:batch],
        retrieve_next_token=md.retrieve_next_token,
        retrieve_next_sibling=md.retrieve_next_sibling,
        retrieve_parent_token=md.retrieve_parent_token,
    )
    mixed_qkv = conv_out.transpose(1, 2).view(seq_len, -1)
    query, key, value = fused_qkv_split_gdn_prefill(
        mixed_qkv,
        layer.num_q_heads,
        layer.num_k_heads,
        layer.num_v_heads,
        layer.head_q_dim,
        layer.head_k_dim,
        layer.head_v_dim,
    )
    return _native_pdl_core(
        executor,
        layer=layer,
        backend=backend,
        cache=cache,
        query=query,
        key=key,
        value=value,
        a=a,
        b=b,
        cache_indices=cache_indices,
        replay_indices=fb.req_pool_indices,
        query_start_loc=query_start_loc,
    )


def _native_pdl_core(
    executor,
    *,
    layer,
    backend,
    cache,
    query,
    key,
    value,
    a,
    b,
    cache_indices,
    replay_indices,
    query_start_loc,
):
    """``_replayssm_target_verify`` with the plugin's PDL launch of the same circular
    verify kernel (``gdn_front/replayssm_pdl.py``)."""
    from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.gdn_front.replayssm_pdl import (
        gdn_replayssm_spec_circular_kernel,
    )

    pool = backend.state_pool
    h, k_dim = layer.num_k_heads, layer.head_k_dim
    hv, v_dim = layer.num_v_heads, layer.head_v_dim
    seq_len = query.numel() // (h * k_dim)
    q = query.reshape(seq_len, h, k_dim)
    k = key.reshape(seq_len, h, k_dim)
    v = value.reshape(seq_len, hv, v_dim)
    a = a.reshape(seq_len, hv)
    b = b.reshape(seq_len, hv)
    out = q.new_empty(seq_len, hv, v_dim)
    bsz = query_start_loc.shape[0] - 1
    d, kc, g = cache.replayssm_d, cache.replayssm_k, cache.replayssm_g
    rawv = cache.replayssm_rawv if cache.replayssm_rawv is not None else d
    rawk = cache.replayssm_rawk if cache.replayssm_rawk is not None else kc
    beta = cache.replayssm_beta if cache.replayssm_beta is not None else g
    storage = (
        2
        if cache.replayssm_beta is not None
        else (1 if cache.replayssm_rawv is not None else 0)
    )
    gdn_replayssm_spec_circular_kernel[(2, bsz, hv)](
        q,
        k,
        v,
        a,
        b,
        layer.A_log,
        layer.dt_bias,
        out,
        cache.temporal,
        d,
        kc,
        g,
        rawv,
        rawk,
        beta,
        query_start_loc,
        cache_indices,
        replay_indices,
        pool.replayssm_spec_write_pos,
        pool.replayssm_cache_base,
        pool.replayssm_is_flush,
        k_dim**-0.5,
        q.stride(0),
        k.stride(0),
        v.stride(0),
        a.stride(0),
        b.stride(0),
        out.stride(0),
        cache.temporal.stride(0),
        d.stride(0),
        kc.stride(0),
        g.stride(0),
        rawv.stride(0),
        rawk.stride(0),
        beta.stride(0),
        query_start_loc.stride(0),
        cache_indices.stride(0),
        replay_indices.stride(0),
        H=h,
        HV=hv,
        K=k_dim,
        V=v_dim,
        BK=128,
        BV=64,
        BS=4,
        BC=16,
        NK=2,
        BKT=64,
        MAX_CACHE_LEN=16,
        SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=True,
        STATE_FP32=cache.temporal.dtype == torch.float32,
        STORE_EXACT_RAW=storage == 2,
        STORE_D_RESIDUAL=storage == 1,
        STORE_K_RESIDUAL=storage == 1,
        IS_FLUSH=False,
        NULL_BLOCK_ID=-1,
        DOT_PRECISION="tf32",
        num_warps=1,
        num_stages=2,
        USE_GDC=True,
        launch_pdl=True,
        WAIT_FOR_GATE_PRODUCER=False,
        PREFETCH_CHECKPOINT=False,
    )
    return out.reshape(value.shape)
