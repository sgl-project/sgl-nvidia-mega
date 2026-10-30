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

Steps of several requests (TP8, with ``multi_request``): up to ``FUSED_MAX_ROWS`` rows
the wide entries of the same kernels; past it the front path, which keeps SGLang's qkv
and o_proj GEMMs and fuses the rest bit for bit: the q / k norm, RoPE and gate split with
the FP8 q and K / V write in one launch (the attention front of slot ``draft_front``),
and the output gate with o_proj's static FP8 input quantization in one launch.
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
    "srt/layers/layer_boundary/fusions/norm.py": ("fused_norm_gamma",),
    "srt/model_executor/forward_batch_info.py": ("out_cache_loc",),
    "srt/kernel_plugins/nvidia/eagle_binding.py|srt/megakernels/eagle_binding.py": (
        "kv",
        "layer_ids",
        "k",
        "v",
    ),
    "srt/layers/quantization/fp8_utils.py": ("flashinfer_bmm_fp8",),
}

# the widest step the fused kernels serve; wider steps take the front path, which measured
# faster from four requests on (TP8, results/trt-1022/l2a micro e58d29c)
FUSED_MAX_ROWS = 14

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


def pack(layer, route_lin: bool = False) -> str:
    """Pack ``layer``'s qkv (+ gate) and o_proj weights at build.

    Returns why its router-before-all-reduce weights cannot be packed, when requested.
    """
    from sgl_nvidia_mega.qwen3_5.sm103.attention import mka as kernels

    kernels.build(layer.qkv_proj.weight.device)
    WEIGHTS[layer.attn.layer_id] = (
        kernels.pack_qkv(layer.qkv_proj.weight.t(), layer.num_heads),
        kernels.pack_o(layer.o_proj.weight.t()),
    )
    return _pack_route_lin(layer) if route_lin else ""


# Slot route_lin (numeric): layer id -> router-before-all-reduce weights on this TP8 rank.
RL_WEIGHTS: dict = {}


def _pack_route_lin(layer) -> str:
    import torch
    import torch.distributed as dist
    from sglang.srt.layers.layer_boundary.fusions.norm import fused_norm_gamma
    from sglang.srt.runtime_context import get_parallel

    from sgl_nvidia_mega.qwen3_5.sm103.hop import RL_WORLD, pack_router_lin

    group = get_parallel().tp_group
    mlp = getattr(layer, "mlp", None)
    if group.world_size != RL_WORLD:
        return f"TP {group.world_size} (serves TP {RL_WORLD})"
    if layer.num_heads != 4:
        return f"{layer.num_heads} query heads a rank (serves 4)"
    boundary = getattr(layer, "ffn_boundary", None)
    norm = getattr(getattr(boundary, "plan", None), "norm", None)
    gamma = fused_norm_gamma(norm) if norm is not None else None
    layer_id = getattr(layer, "layer_id", None)
    if (
        mlp is None
        or getattr(mlp, "gate", None) is None
        or gamma is None
        or layer_id != layer.attn.layer_id
        or getattr(mlp, "layer_id", layer_id) != layer_id
    ):
        return "no router or fusible FFN-input norm"
    try:
        with torch.no_grad():
            RL_WEIGHTS[layer.attn.layer_id] = pack_router_lin(
                mlp.gate.weight,
                gamma,
                layer.o_proj.weight.t(),
                layer.o_proj.input_scale,
                layer.o_proj.weight_scale,
                dist.get_rank(group.device_group),
            )
    except ValueError as error:
        return str(error)
    return ""


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
        logger.info(
            "NVIDIA megakernel: Qwen3.5 slot mka runs at %d rows (%s)",
            m,
            "fused" if m <= FUSED_MAX_ROWS else "front path",
        )
    if m > FUSED_MAX_ROWS:
        return _front_step(layer, x, st)
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
    if st.slots.skv2 or split_kv.cp_multi(m):
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
    h0 = _route_lin_h0(hop, layer, m, st)
    if h0 is not None:
        # Slot route_lin: the router partial rides the o_proj push.
        kernels.epilogue_rl(
            attn=o,
            gate=gate,
            w_packed=w_o,
            sx=layer.o_proj.input_scale,
            sw=layer.o_proj.weight_scale,
            out=out,
            site=hop.site,
            rl_args=hop.rl_push_args(RL_WEIGHTS[attn.layer_id], h0),
        )
        hop.mark_pushed_rl(out)
        # the router launch does not wait on the epilogue: keep its output's memory out of reuse for the whole step
        st.scratch.setdefault("rlin.keep", []).append(out)
        return out
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


def _route_lin_h0(hop, layer, rows: int, st):
    """The layer's residual input when its router partial can ride the o_proj push (as gdn/verify.py's); else None."""
    if (
        not st.slots.route_lin
        or hop is None
        or hop.rl is None
        or layer.attn.layer_id not in RL_WEIGHTS
    ):
        return None
    lid, h0 = st.scratch.get("rlin.h0", (None, None))
    if lid != layer.layer_id or h0 is None or tuple(h0.shape) != (rows, 4096):
        return None
    return h0


def _front_step(layer, x, st):
    """The front path of a wide step: SGLang's qkv GEMM, the front with the FP8 KV write,
    the attention, the gate with o_proj's input quantization, SGLang's o_proj GEMM."""
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        use_symmetric_memory,
    )
    from sglang.srt.layers.dp_attention import is_allocation_symmetric
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_bmm_fp8
    from sglang.srt.runtime_context import get_parallel

    from sgl_nvidia_mega.qwen3_5 import lowm
    from sgl_nvidia_mega.qwen3_5.attention import split_kv
    from sgl_nvidia_mega.qwen3_5.sm103.attention.gate_quant import gate_quant
    from sgl_nvidia_mega.qwen3_5.sm103.attention.q35_cp_lite import CpLite
    from sgl_nvidia_mega.qwen3_5.sm103.draft.front import (
        fused_qk_gemma_rmsnorm_rope_gate,
    )

    m = x.shape[0]
    attn = layer.attn
    width = layer.num_heads * layer.head_dim
    # slot hop_fp8: qkv_proj reads the hop's E4M3 copy of x (no quantize kernel)
    x8 = lowm.take_fp8(st, x) if st.slots.hop_fp8 else None
    if x8 is not None:
        lowm.engaged("hop_fp8", "qkv_proj bmm", m)
        qkv = flashinfer_bmm_fp8(
            x8,
            layer.qkv_proj.weight,
            layer.qkv_proj.input_scale,
            layer.qkv_proj.weight_scale,
            x.dtype,
        )
    else:
        qkv, _ = layer.qkv_proj(x)
    q_gate, k, v = qkv.split(
        [
            2 * width,
            layer.num_kv_heads * layer.head_dim,
            layer.num_kv_heads * layer.head_dim,
        ],
        dim=-1,
    )
    i = st.view.kv.layer_ids.index(attn.layer_id)
    k_scale, v_scale = split_kv.kv_scales(attn, x.device)
    rope = layer.rotary_emb
    q, _, gate = fused_qk_gemma_rmsnorm_rope_gate(
        q_gate,
        k,
        layer.q_norm.weight.data,
        layer.k_norm.weight.data,
        rope.cos_sin_cache,
        st.positions,
        layer.q_norm.variance_epsilon,
        layer.num_heads,
        layer.num_kv_heads,
        layer.head_dim,
        rope.rotary_dim,
        has_gate=True,
        mrope_axis_map=rope.axis_map,
        v=v,
        kv_cache=(st.view.kv.k[i], st.view.kv.v[i]),
        cache_loc=st.fb.out_cache_loc,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    gate = gate.reshape(m, width)
    cp_lite = CpLite._instance if split_kv.cp_multi(m) else None
    oq = None
    if cp_lite is not None:
        oq = torch.empty((m, width), dtype=torch.float8_e4m3fn, device=x.device)
        cp_lite.offer_gate_epilogue(gate, layer.o_proj.input_scale, oq)
    q = q.reshape(m, width)
    o = (
        split_kv.core(attn, q, st)
        if cp_lite is not None
        else attn(q, None, None, st.fb)
    ).reshape(m, width)
    if cp_lite is None or not cp_lite.take_gate_epilogue():
        oq = gate_quant(o.contiguous(), gate, layer.o_proj.input_scale)
    if st.slots.lowm_oproj:
        # slot lowm_out: o_proj on the low-M launch, pushing into the hop with lowm_push
        hop = None
        if st.slots.lowm_push:
            from sgl_nvidia_mega.qwen3_5.boundary import hop as boundary_hop

            if boundary_hop.takes_moe_input(st.bound["boundary"].hop, layer, st, m):
                hop = st.bound["boundary"].hop
        return lowm.row_parallel(layer.o_proj, oq, torch.bfloat16, "o_proj", hop)
    # the same allocation as RowParallelLinear's output, which the boundary all-reduces
    with use_symmetric_memory(
        get_parallel().tp_group, disabled=not is_allocation_symmetric()
    ):
        return flashinfer_bmm_fp8(
            oq,
            layer.o_proj.weight,
            layer.o_proj.input_scale,
            layer.o_proj.weight_scale,
            torch.bfloat16,
        )


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
