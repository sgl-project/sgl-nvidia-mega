# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/qwen3_5.py (Qwen3_5GatedDeltaNet.forward).
"""The GatedDeltaNet mixer of a covered target-verify step, over the layer's loaded
weights, SGLang's linear-attention backend and its mamba pool.

Stock: the CUDA target-verify arm of ``Qwen3_5GatedDeltaNet.forward`` with SGLang's own
ops in its order: the input projections, the fused split of q / k / v / z / b / a, the
causal conv + ReplaySSM verify through the backend (``RadixLinearAttention``), the
gated RMSNorm and ``out_proj``. ``step`` returns the TP-partial output, whose
all-reduce the boundary runs. Slot ``gdn_verify`` (``verify.py``) runs the step with
the backend's chain opened up and its launch choices changed; slot ``gdn_push`` pushes
``out_proj``'s partial into the decode hop from its GEMM (numeric), and slot ``gdn_front``
(``front.py``) runs the projections and the conv of one request of 7 rows in one launch
(numeric).
"""

from __future__ import annotations

import msgspec

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/qwen3_5.py": (
        "Qwen3_5LinearDecoderLayer",
        "layers",
        "start_layer",
        "end_layer",
        "linear_attn",
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
    "kernels/ops/attention/triton_gdn_fused_proj.py": (
        "fused_qkvzba_split_reshape_cat_contiguous",
    ),
    "srt/layers/radix_linear_attention.py": (
        "forward(mixed_qkv)",
        "forward(a)",
        "forward(b)",
    ),
    "srt/layers/attention/hybrid_linear_attn_backend.py": ("linear_attn_backend",),
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "srt/layers/layer_boundary/fusions/norm.py": ("fused_norm_gamma",),
    "srt/layers/quantization/modelopt_quant.py": ("orig_dtype",),
    "srt/runtime_context.py": ("get_parallel",),
}

# The SGLang functions whose body this module mirrors, by the digest of their code
# (test/unit/test_contract.py recomputes them on the installed SGLang)
SGLANG_MIRRORS = {
    "srt/models/qwen3_5.py:Qwen3_5GatedDeltaNet.forward": "a5e9e3bd054ff18f",
}


class Bound(msgspec.Struct, frozen=True):
    """The GDN package's build result (``st.bound["gdn"]``)."""

    gdn_verify: bool = False  # slot gdn_verify serves every GDN layer
    gdn_pdl: bool = False  # and launches its verify dependent on the conv
    gdn_core: bool = False  # and runs its verify, gated norm and FP8 in one launch
    gdn_conv: bool = (
        False  # and its conv with the x rows loaded first and one window store
    )
    # slot gdn_push: layer id -> out_proj's weights packed for the hop's push GEMM
    packed: dict = msgspec.field(default_factory=dict)
    # slot gdn_front: layer id -> in_proj_qkvz's weights packed for the front
    front: dict = msgspec.field(default_factory=dict)
    gdn_trt_front: bool = False  # TensorRT-LLM front built through four requests
    # slot route_lin (numeric): layer id -> router-before-all-reduce packed weights
    rlin: dict = msgspec.field(default_factory=dict)
    error: object = None  # the front's sticky error word, int32, zero
    reason: str = ""  # why a requested slot is not built


def build(model, ctx) -> Bound:
    """Slot ``gdn_verify`` for every GatedDeltaNet layer, or why a layer cannot take it."""
    from sglang.srt.models.qwen3_5 import Qwen3_5LinearDecoderLayer

    from sgl_nvidia_mega.qwen3_5.gdn import verify
    from sgl_nvidia_mega.qwen3_5.runtime import SLOTS

    if "gdn_verify" not in SLOTS:
        return Bound(reason="stock (SGLANG_NVIDIA_MEGAKERNEL_SLOTS)")
    m = model.model
    gdns = [
        (i, m.layers[i].linear_attn)
        for i in range(m.start_layer, m.end_layer)
        if isinstance(m.layers[i], Qwen3_5LinearDecoderLayer)
    ]
    for i, gdn in gdns:
        reason = verify.unsupported(gdn)
        if reason:
            return Bound(reason=f"layer {i}: {reason}")
    reasons = []
    packed = {}
    if "gdn_push" in SLOTS:
        import torch

        from sgl_nvidia_mega.qwen3_5.sm103.hop import pack_out_proj

        for i, gdn in gdns:
            # The hop's out-projection kernel writes BF16.  Keep FP16 layers on
            # the non-push path, whose FlashInfer BMM honors layer.orig_dtype.
            if getattr(gdn.out_proj, "orig_dtype", torch.bfloat16) != torch.bfloat16:
                packed = {}
                break
            # the method keeps the weight as the [K, N] transpose of a row-major [N, K]
            try:
                packed[gdn.attn.layer_id] = pack_out_proj(gdn.out_proj.weight.t())
            except ValueError as e:
                reasons.append(f"gdn_push: layer {i}: {e}")
                packed = {}
                break
    rlin = {}
    if "route_lin" in SLOTS:
        if packed:
            rlin, reason = _route_lin(m, gdns)
            if reason:
                reasons.append(f"route_lin: {reason}")
                rlin = {}
        else:
            reasons.append("route_lin: requires supported gdn_push")
    front_packed, error = {}, None
    if "gdn_front" in SLOTS:
        import torch

        from sgl_nvidia_mega.qwen3_5.gdn import front

        if ctx.tokens_per_request != front.ROWS:
            reasons.append(f"gdn_front: {ctx.tokens_per_request} rows a request")
        else:
            for i, gdn in gdns:
                reason = front.unsupported(gdn)
                if reason:
                    reasons.append(f"gdn_front: layer {i}: {reason}")
                    break
            else:
                front_packed = front.build(model, gdns)
                error = torch.zeros(1, dtype=torch.int32, device="cuda")
    core = "gdn_core" in SLOTS
    if core:
        import torch

        from sgl_nvidia_mega.qwen3_5.sm103.gdn import core as core_kernel

        attn = gdns[0][1].attn
        core_kernel.build(
            torch.cuda.current_device(),
            ctx.tokens_per_request,
            attn.num_k_heads,
            attn.num_v_heads,
            gdns[0][1].norm.weight.dtype == torch.float32,
        )
    trt_front = "gdn_trt_front" in SLOTS and bool(front_packed) and core
    if "gdn_trt_front" in SLOTS and not trt_front:
        reasons.append("gdn_trt_front: requires supported gdn_front and gdn_core")
    return Bound(
        gdn_verify=True,
        gdn_pdl="gdn_pdl" in SLOTS,
        gdn_core=core,
        gdn_conv="gdn_conv" in SLOTS,
        packed=packed,
        front=front_packed,
        gdn_trt_front=trt_front,
        rlin=rlin,
        error=error,
        reason="; ".join(reasons),
    )


def step(layer, x, st):
    """``layer``'s GatedDeltaNet on the normed input ``x`` [rows, hidden]: the
    TP-partial output [rows, hidden]."""
    if st.slots.gdn_verify:
        from sgl_nvidia_mega.qwen3_5.boundary import hop
        from sgl_nvidia_mega.qwen3_5.gdn import verify

        opts = verify.SLOT_PDL if st.slots.gdn_pdl else verify.SLOT
        if st.slots.gdn_core:
            opts = msgspec.structs.replace(opts, core=True)
            verify.engaged("gdn_core", x.shape[0])
        if st.slots.gdn_conv:
            opts = msgspec.structs.replace(opts, conv_rows=True)
            verify.engaged("gdn_conv", x.shape[0])
        # push only where the MoE input's all-reduce will consume on the hop
        if st.slots.gdn_push and hop.takes_moe_input(
            st.bound["boundary"].hop, layer, st, x.shape[0]
        ):
            opts = msgspec.structs.replace(opts, push=True)
            verify.engaged("gdn_push", x.shape[0])
        # slot lowm_push: out_proj's low-M launch pushes its rows into the hop instead
        elif st.slots.lowm_push and hop.takes_moe_input(
            st.bound["boundary"].hop, layer, st, x.shape[0]
        ):
            opts = msgspec.structs.replace(opts, lowm_push=True)
        if st.slots.gdn_front:
            from sgl_nvidia_mega.qwen3_5.gdn import front

            return front.step(layer.linear_attn, x, st, opts)
        return verify.step(layer.linear_attn, x, st, opts)
    import triton
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
        fused_qkvzba_split_reshape_cat_contiguous,
    )
    from sglang.srt.model_executor.forward_context import get_attn_backend

    gdn = layer.linear_attn
    qkvz, ba = gdn._forward_input_proj(x)
    backend = get_attn_backend()
    backend = getattr(backend, "linear_attn_backend", backend)
    if getattr(backend, "supports_packed_verify", False):
        # SGLang folds the split and conv into such a backend; this step does not
        raise NotImplementedError("a linear-attention backend with packed verify")
    mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat_contiguous(
        qkvz,
        ba,
        triton.cdiv(gdn.num_k_heads, gdn.attn_tp_size),
        triton.cdiv(gdn.num_v_heads, gdn.attn_tp_size),
        gdn.head_k_dim,
        gdn.head_v_dim,
    )
    core = gdn.attn(st.fb, mixed_qkv=mixed_qkv, a=a, b=b)
    z_shape = z.shape
    core = core.reshape(-1, core.shape[-1])
    z = z.reshape(-1, z.shape[-1])
    # DP attention may return only the real rows while the projection and z
    # retain the graph's padded rows. Match SGLang before the gated norm.
    if core.shape != z.shape:
        padded = z.new_zeros(z.shape)
        padded[: core.shape[0], :] = core
        core = padded
    core = gdn.norm(core, z)
    core = core.reshape(z_shape)
    core = core.reshape(*core.shape[:-2], core.shape[-2] * core.shape[-1])
    out, _ = gdn.out_proj(core)
    return out


def _route_lin(m, gdns) -> tuple[dict, str]:
    """Packed route-lin weights for every GDN layer, or why they cannot be built."""
    import torch
    import torch.distributed as dist
    from sglang.srt.layers.layer_boundary.fusions.norm import fused_norm_gamma
    from sglang.srt.runtime_context import get_parallel

    from sgl_nvidia_mega.qwen3_5.sm103.hop import RL_WORLD, pack_router_lin

    group = get_parallel().tp_group
    if group.world_size != RL_WORLD:
        return {}, f"TP {group.world_size} (serves TP {RL_WORLD})"
    rank = dist.get_rank(group.device_group)
    out = {}
    for i, gdn in gdns:
        layer = m.layers[i]
        mlp = getattr(layer, "mlp", None)
        boundary = getattr(layer, "ffn_boundary", None)
        norm = getattr(getattr(boundary, "plan", None), "norm", None)
        gamma = fused_norm_gamma(norm) if norm is not None else None
        if (
            mlp is None
            or getattr(mlp, "gate", None) is None
            or gamma is None
            or layer.layer_id != gdn.attn.layer_id
            or getattr(mlp, "layer_id", layer.layer_id) != layer.layer_id
        ):
            return {}, f"layer {i} has no router or fusible FFN-input norm"
        try:
            with torch.no_grad():
                out[gdn.attn.layer_id] = pack_router_lin(
                    mlp.gate.weight,
                    gamma,
                    gdn.out_proj.weight.t(),
                    gdn.out_proj.input_scale,
                    gdn.out_proj.weight_scale,
                    rank,
                )
        except ValueError as error:
            return {}, f"layer {i}: {error}"
    return out, ""
