# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang branch nvidia/qwen35-agentx-gb300:
# python/sglang/srt/models/qwen3_5.py (Qwen3_5AttentionDecoderLayer._mka_static_eligible,
# _mka_eligible and _mka_self_attention).
"""Slot ``mka``: the full attention of a covered target-verify step through the fused
prologue and epilogue of SGLang branch nvidia/qwen35-agentx-gb300
(``sm103/attention/mka.py``) around the attention.

The prologue takes the normed rows and writes the FP8 query rows, the bf16 gate and this
step's FP8 K / V rows into the forward view's pool; the attention reads them (slot
``skv2``'s core when it runs, else SGLang's attention with no K / V to write); the
epilogue gates, quantizes and runs ``o_proj`` into the TP-partial output, in SGLang's
symmetric pool as ``RowParallelLinear``'s output. The weights are packed at build. With
slot ``mka_push`` the epilogue also pushes its rows into the decode hop's mailbox where
the MoE input's all-reduce will run the hop for them, and that all-reduce only consumes
(the branch's MKA_PUSH).
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/qwen3_5.py": (
        "attn_output_gate",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "hidden_size",
        "rotary_emb",
        "qkv_proj",
        "o_proj",
        "q_norm",
        "k_norm",
        "attn",
    ),
    "srt/layers/rotary_embedding/mrope.py": (
        "MRotaryEmbedding",
        '"axis_map"',
        "cos_sin_cache",
    ),
    "srt/layers/rotary_embedding/base.py": ("rotary_dim", "cos_sin_cache"),
    "srt/layers/layernorm.py": ("variance_epsilon",),
    "srt/layers/quantization/modelopt_quant.py": (
        "ModelOptFp8LinearMethod",
        "use_marlin",
        '"input_scale"',
        '"weight_scale"',
    ),
    "srt/layers/radix_attention.py": ("layer_id",),
    "srt/distributed/device_communicators/pynccl_allocator.py": (
        "use_symmetric_memory",
    ),
    "srt/layers/dp_attention.py": ("is_allocation_symmetric",),
    "srt/runtime_context.py": ("get_parallel",),
    "srt/model_executor/forward_batch_info.py": ("out_cache_loc",),
    "srt/kernel_plugins/nvidia/eagle_binding.py|srt/megakernels/eagle_binding.py": (
        "kv",
        "layer_ids",
        "k",
        "v",
    ),
}

# the packed (qkv, o_proj) weights per full-attention layer id, filled at build
WEIGHTS: dict = {}
# the rows the slot has run at (each logged once, as engagement evidence)
_ENGAGED: set = set()


def _per_tensor_fp8(linear) -> bool:
    method = linear.quant_method
    return (
        type(method).__name__ == "ModelOptFp8LinearMethod"
        and not method.use_marlin
        and linear.weight.dtype == torch.float8_e4m3fn
        and linear.input_scale is not None
        and linear.input_scale.numel() == 1
        and linear.weight_scale.numel() == 1
    )


def unsupported(layer) -> str:
    """Why the slot cannot serve ``layer`` (a Qwen3_5AttentionDecoderLayer; "" when it
    can): the branch's ``_mka_static_eligible`` and its M-RoPE positions."""
    from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding

    rope = layer.rotary_emb
    if not layer.attn_output_gate:
        return "no attention output gate"
    if layer.num_heads not in (4, 8) or layer.num_kv_heads != 1:
        return (
            f"{layer.num_heads} query / {layer.num_kv_heads} KV heads a rank "
            "(MKA runs 4 or 8 / 1)"
        )
    if layer.head_dim != 256 or layer.hidden_size != 4096:
        return f"head_dim {layer.head_dim}, hidden {layer.hidden_size} (MKA runs 256, 4096)"
    if not isinstance(rope, MRotaryEmbedding) or rope.axis_map is None:
        return "no M-RoPE axis map"
    if rope.rotary_dim != 64 or rope.cos_sin_cache.dtype != torch.float32:
        return f"rotary_dim {rope.rotary_dim} with a {rope.cos_sin_cache.dtype} cos/sin cache"
    if not (_per_tensor_fp8(layer.qkv_proj) and _per_tensor_fp8(layer.o_proj)):
        return "qkv_proj or o_proj is not a static per-tensor FP8 linear"
    return ""


def pool_reasons(kv) -> list[str]:
    """Why the prologue cannot write these KV pools ([] when it can): one FP8 KV head
    of 256 per slot."""
    for buf in (*kv.k, *kv.v):
        if buf.dtype != torch.float8_e4m3fn or buf.stride(0) != 256:
            return [f"a {buf.dtype} KV pool buffer of strides {buf.stride()}"]
    return []


def pack(layer) -> None:
    """Pack ``layer``'s qkv (+ gate) and o_proj weights for the kernels (at build)."""
    from sgl_nvidia_mega.qwen3_5.sm103.attention import mka as kernels

    kernels.build(layer.qkv_proj.weight.device)
    WEIGHTS[layer.attn.layer_id] = (
        kernels.pack_qkv(layer.qkv_proj.weight.t(), layer.num_heads),
        kernels.pack_o(layer.o_proj.weight.t()),
    )


def step(layer, x, st):
    """``layer``'s attention on the normed rows ``x`` [rows, hidden]: the TP-partial
    output [rows, hidden]."""
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        use_symmetric_memory,
    )
    from sglang.srt.layers.dp_attention import is_allocation_symmetric
    from sglang.srt.runtime_context import get_parallel

    from sgl_nvidia_mega.qwen3_5.attention import split_kv
    from sgl_nvidia_mega.qwen3_5.sm103.attention import mka as kernels

    m = x.shape[0]
    if m not in _ENGAGED:
        _ENGAGED.add(m)
        logger.info("NVIDIA megakernel: Qwen3.5 slot mka runs at %d rows", m)
    attn = layer.attn
    width = layer.num_heads * layer.head_dim
    q = torch.empty((m, width), dtype=torch.float8_e4m3fn, device=x.device)
    gate = torch.empty((m, width), dtype=torch.bfloat16, device=x.device)
    i = st.view.kv.layer_ids.index(attn.layer_id)
    k_scale, v_scale = split_kv.kv_scales(attn, x.device)
    w_qkv, w_o = WEIGHTS[attn.layer_id]
    kernels.prologue(
        x=x.contiguous(),
        w_packed=w_qkv,
        sx=layer.qkv_proj.input_scale,
        sw=layer.qkv_proj.weight_scale,
        q_norm_w=layer.q_norm.weight.data,
        k_norm_w=layer.k_norm.weight.data,
        cos_sin=layer.rotary_emb.cos_sin_cache,
        positions=st.positions,
        axis_map=layer.rotary_emb.axis_map,
        cache_loc=st.fb.out_cache_loc,
        k_cache=st.view.kv.k[i],
        v_cache=st.view.kv.v[i],
        k_scale=k_scale,
        v_scale=v_scale,
        q_out=q,
        gate_out=gate,
        nq=layer.num_heads,
        eps=layer.q_norm.variance_epsilon,
    )
    if st.slots.skv2:
        o = split_kv.core(attn, q, st)
    else:
        o = attn(q, None, None, st.fb)
    o = o.reshape(m, width).contiguous()
    # the same allocation as RowParallelLinear's output, which the boundary all-reduces
    with use_symmetric_memory(
        get_parallel().tp_group, disabled=not is_allocation_symmetric()
    ):
        out = torch.empty((m, layer.hidden_size), dtype=torch.bfloat16, device=x.device)
    hop = _push_hop(layer, st, m) if st.slots.mka_push else None
    kernels.epilogue(
        attn=o,
        gate=gate,
        w_packed=w_o,
        sx=layer.o_proj.input_scale,
        sw=layer.o_proj.weight_scale,
        out=out,
        nq=layer.num_heads,
        site=None if hop is None else hop.site,
    )
    if hop is not None:
        hop.mark_pushed(out)
    return out


def _push_hop(layer, st, m: int):
    """Slot mka_push: the hop the epilogue pushes its rows into, where the MoE input's
    all-reduce runs the hop for them (it then only consumes); else None."""
    from sgl_nvidia_mega.qwen3_5.boundary import hop as boundary_hop

    hop = st.bound["boundary"].hop
    if not boundary_hop.takes_moe_input(hop, layer, st, m):
        return None
    if ("push", m) not in _ENGAGED:
        _ENGAGED.add(("push", m))
        logger.info("NVIDIA megakernel: Qwen3.5 slot mka_push runs at %d rows", m)
    return hop
