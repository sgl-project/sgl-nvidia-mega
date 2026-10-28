# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from python/sglang/srt/models/qwen3_5.py (Qwen3_5GatedDeltaNet._try_fused_verify)
# of SGLang branch nvidia/qwen35-agentx-gb300 and SGLang's
# python/sglang/srt/layers/attention/linear/gdn_backend.py (GDNAttnBackend.forward_extend).
"""Slot ``gdn_front`` (numeric): slot gdn_verify's step for one request of 7 rows with
the fused GDN front of SGLang branch nvidia/qwen35-agentx-gb300 (SGLANG_Q35MF_MKG_FUSED +
MKG_BA_TC + MKM_GDN_FRONT):

1. ``sm103/gdn/front.py`` in one launch: the static-FP8 ``in_proj_qkvz`` projection of
   the BF16 rows, the conv update of its q / k / v channels, the BF16 ``in_proj_ba``
   projection, and an L2 prefetch of the verify's checkpoint and history ring;
2. the ReplaySSM verify on the conv output, launched dependent on the front and waiting
   for it before it reads the gates b / a (``replayssm.verify``);
3. the gated RMSNorm writing ``out_proj``'s static FP8 input (``front_norm.py``),
   released dependents at entry;
4. ``out_proj``: FlashInfer's BMM, or with slot gdn_push the hop's push GEMM.

The two projections sum in another order than SGLang's GEMMs (numeric); every launch
is the branch's kernel with the branch's launch, bit for bit on the same inputs.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/qwen3_5.py": (
        "in_proj_qkvz",
        "in_proj_ba",
        "attn_tp_size",
        "conv_kernel_size",
        "activation",
        "head_k_dim",
        "head_v_dim",
        "attn",
        "norm",
        "out_proj",
    ),
    "srt/layers/radix_linear_attention.py": (
        "layer_id",
        "conv_weights",
        "bias",
        "num_k_heads",
        "num_v_heads",
        "q_dim",
        "k_dim",
        "v_dim",
    ),
    "kernels/ops/attention/fla/layernorm_gated.py": (
        "eps",
        "weight",
        "bias",
        "activation",
        "group_size",
        "norm_before_gate",
    ),
    "srt/layers/quantization/modelopt_quant.py": (
        "ModelOptFp8LinearMethod",
        "use_marlin",
        "use_sm120_fp8",
        '"input_scale"',
        '"weight_scale"',
    ),
    "srt/layers/quantization/unquant.py": ("UnquantizedLinearMethod",),
    "srt/kernel_plugins/nvidia/eagle_binding.py|srt/megakernels/eagle_binding.py": (
        "gdn",
        "mamba",
        "state_indices",
        "intermediate_indices",
        "pad_slot_id",
        "layer_ids",
        "conv",
        "temporal",
        "intermediate_conv_window",
        "ring_d",
        "ring_k",
        "ring_g",
        '"ForwardBatch.req_pool_indices"',
    ),
    "srt/model_executor/forward_batch_info.py": ("req_pool_indices",),
}

ROWS = 7  # one request of 7 verify rows
CONV_CHANNELS, CONV_WIDTH = 1536, 4

# the step widths that have run the slot (each logged once, as engagement evidence)
_ENGAGED: set = set()


def _fp8_static(linear, weight_shape) -> bool:
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8LinearMethod

    method = linear.quant_method
    return (
        isinstance(method, ModelOptFp8LinearMethod)
        and not method.use_marlin
        and not method.use_sm120_fp8
        and linear.weight.dtype == torch.float8_e4m3fn
        and tuple(linear.weight.shape) == weight_shape
        and linear.input_scale.numel() == 1
        and linear.weight_scale.numel() == 1
        and linear.input_scale.dtype == torch.float32
        and linear.weight_scale.dtype == torch.float32
    )


def unsupported(gdn) -> str:
    """Why the front cannot serve ``gdn`` (a Qwen3_5GatedDeltaNet) ("" when it can):
    the TP8 shapes and static FP8 / BF16 projections the branch's kernel is built for."""
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    attn, norm, ba = gdn.attn, gdn.norm, gdn.in_proj_ba
    if gdn.attn_tp_size != 8 or (attn.num_k_heads, attn.num_v_heads) != (2, 8):
        return (
            f"TP {gdn.attn_tp_size} with {attn.num_k_heads} / {attn.num_v_heads} heads"
        )
    if (gdn.head_k_dim, gdn.head_v_dim) != (128, 128) or gdn.conv_kernel_size != 4:
        return "head dims or conv width"
    if gdn.activation not in ("silu", "swish") or norm.activation not in (
        "silu",
        "swish",
    ):
        return "an activation other than SiLU"
    if (
        not norm.norm_before_gate
        or norm.bias is not None
        or norm.group_size is not None
    ):
        return "a grouped, biased or gate-first norm"
    if attn.bias is not None:
        return "a conv bias"
    w = attn.conv_weights
    if (
        tuple(w.shape) != (CONV_CHANNELS, CONV_WIDTH)
        or w.dtype != torch.bfloat16
        or not w.is_contiguous()
    ):
        return f"conv weights {w.dtype} {tuple(w.shape)}"
    if not _fp8_static(gdn.in_proj_qkvz, (4096, 2560)):
        return "in_proj_qkvz is not a static per-tensor FP8 [4096, 2560] linear"
    if not _fp8_static(gdn.out_proj, (1024, 4096)):
        return "out_proj is not a static per-tensor FP8 [1024, 4096] linear"
    if (
        not isinstance(ba.quant_method, UnquantizedLinearMethod)
        or ba.bias is not None
        or ba.weight.dtype != torch.bfloat16
        or tuple(ba.weight.shape) != (16, 4096)
        or not ba.weight.is_contiguous()
    ):
        return "in_proj_ba is not an unbiased BF16 [16, 4096] linear"
    return ""


def pool_reasons(pools) -> list[str]:
    """Why the front cannot run on the mamba pools (``Step.view.mamba``) ([] when it
    can): BF16 conv states [slots, 1536, 3] and deduplicated BF16 windows."""
    reasons = []
    conv, windows = pools.conv[0], pools.intermediate_conv_window[0]
    if conv.dtype != torch.bfloat16 or tuple(conv.shape[2:]) != (CONV_CHANNELS, 3):
        reasons.append(f"conv states {conv.dtype} {tuple(conv.shape)}")
    if (
        windows.dtype != torch.bfloat16
        or tuple(windows.shape[2:]) != (ROWS, CONV_CHANNELS, 3)
        or tuple(windows.stride()[2:]) != (1, 9, 1)
        or windows.stride(1) < CONV_CHANNELS * 9
    ):
        reasons.append(
            f"conv windows {tuple(windows.shape)} {tuple(windows.stride())} are not "
            "the deduplicated BF16 windows of 7 rows"
        )
    return reasons


def build(model, gdns) -> dict:
    """Each GDN layer's in_proj_qkvz packed for the front, by layer id; compiles the
    front."""
    from sgl_nvidia_mega.qwen3_5.sm103.gdn import front

    front.build(torch.cuda.current_device())
    # the method keeps the weight as the [K, N] transpose of a row-major [N, K]
    return {
        gdn.attn.layer_id: front.pack(gdn.in_proj_qkvz.weight.t()) for _, gdn in gdns
    }


def step(gdn, x: torch.Tensor, st, opts) -> torch.Tensor:
    """``Qwen3_5GatedDeltaNet.forward`` of a target-verify step of 7 rows on the front:
    the TP-partial output [7, hidden] of the normed BF16 rows x; ``opts`` is slot
    gdn_verify's (the verify's tiles, and ``push`` for slot gdn_push)."""
    from sgl_nvidia_mega.qwen3_5.gdn import replayssm
    from sgl_nvidia_mega.qwen3_5.gdn.front_norm import norm
    from sgl_nvidia_mega.qwen3_5.gdn.verify import _out_proj
    from sgl_nvidia_mega.qwen3_5.sm103.gdn import front

    if x.shape[0] not in _ENGAGED:
        _ENGAGED.add(x.shape[0])
        logger.info(
            "NVIDIA megakernel: Qwen3.5 slot gdn_front runs at %d rows", x.shape[0]
        )
    attn, bound = gdn.attn, st.bound["gdn"]
    meta, pools = st.view.gdn, st.view.mamba
    li = pools.layer_ids.index(attn.layer_id)
    projected = x.new_empty((ROWS, 2560))
    conv = x.new_empty((ROWS, CONV_CHANNELS))
    ba = x.new_empty((ROWS, 16))
    front.front(
        x,
        bound.front[attn.layer_id],
        gdn.in_proj_qkvz.input_scale,
        gdn.in_proj_qkvz.weight_scale,
        attn.conv_weights,
        pools.conv[0][li],
        pools.intermediate_conv_window[0][li],
        meta.state_indices[:1],
        meta.intermediate_indices[:1],
        projected,
        conv,
        bound.error,
        ba_weight=gdn.in_proj_ba.weight,
        ba_out=ba,
        checkpoint=pools.temporal[li],
        history_d=pools.ring_d[li],
        history_k=pools.ring_k[li],
        history_g=pools.ring_g[li],
        ring_indices=st.fb.req_pool_indices[:1],
        pad_slot=meta.pad_slot_id,
    )
    query, key, value = torch.split(conv, [attn.q_dim, attn.k_dim, attn.v_dim], dim=-1)
    H, K = attn.num_k_heads, attn.head_k_dim
    HV, V = attn.num_v_heads, attn.head_v_dim
    d_cache = pools.ring_d[li]
    core = query.new_empty(ROWS, HV, V)

    def ring(t):
        return None if t is None else t[li]

    # b / a in the front's b8 / a8 order
    replayssm.verify(
        query.reshape(ROWS, H, K),
        key.reshape(ROWS, H, K),
        value.reshape(ROWS, HV, V),
        ba[:, 8:],
        ba[:, :8],
        attn.A_log,
        attn.dt_bias,
        core,
        pools.temporal[li],
        d_cache,
        pools.ring_k[li],
        pools.ring_g[li],
        ring(pools.ring_rawv),
        ring(pools.ring_rawk),
        ring(pools.ring_beta),
        meta.query_start_loc,
        meta.state_indices,
        st.fb.req_pool_indices,
        pools.ring_write_pos,
        pools.ring_cache_base,
        pools.ring_is_flush,
        max_cache_len=d_cache.shape[-2],
        max_spec_len=ROWS,
        scale=K**-0.5,
        null_block_id=-1,
        block_v=opts.block_v,
        num_warps=opts.num_warps,
        pdl=True,
        wait_for_gate_producer=True,
    )
    quant = torch.empty((ROWS * HV, V), dtype=torch.float8_e4m3fn, device=x.device)
    rstd = torch.empty((ROWS * HV,), dtype=torch.float32, device=x.device)
    norm(
        core.view(ROWS * HV, V),
        projected[:, CONV_CHANNELS:].view(ROWS, HV, V),
        gdn.norm.weight,
        gdn.out_proj.input_scale,
        quant,
        rstd,
        gdn.norm.eps,
    )
    x_fp8 = quant.view(ROWS, HV * V)
    if opts.push:
        return st.bound["boundary"].hop.out_proj_push(
            x_fp8=x_fp8,
            packed_w=bound.packed[attn.layer_id],
            input_scale=gdn.out_proj.input_scale,
            weight_scale=gdn.out_proj.weight_scale,
        )
    return _out_proj(gdn.out_proj, x_fp8, x.dtype)
