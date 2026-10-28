# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/qwen3_5.py (Qwen3_5GatedDeltaNet.forward),
# python/sglang/srt/layers/attention/linear/gdn_backend.py (GDNAttnBackend.forward_extend,
# GDNAttnBackend._replayssm_target_verify), python/sglang/srt/layers/linear.py
# (RowParallelLinear.forward) and python/sglang/srt/layers/quantization/fp8_utils.py
# (apply_fp8_linear_bmm_flashinfer).
"""Slot ``gdn_verify``: a GatedDeltaNet target-verify step with SGLang's chain opened up,
so its launches take the choices SGLang fixes. The chain is the backend's
target-verify arm for the circular ReplaySSM ring (the causal conv update, the strided
q / k / v split, the ReplaySSM verify), reading the step's GDN metadata and the mamba
pools from the forward's view (``Step.view.gdn`` and ``.mamba``) and its request slots
from the forward batch, then the gated norm and ``out_proj``. The choices, all bitwise
to SGLang's own:

- token-major projection views, with a token-major conv output (no split copy);
- a 128-channel conv tile;
- the ReplaySSM verify with 32-wide value tiles on 2 warps, on SGLang's kernel
  (``replayssm.py``), and with slot ``gdn_pdl`` launched dependent on the conv;
- the gated norm writing the static FP8 ``out_proj`` takes (``norm_fp8.py``), which
  ``out_proj``'s FlashInfer BMM then reads without quantizing it again.

Slot ``gdn_push`` (numeric, not bitwise) runs ``out_proj`` on the decode hop's GEMM
instead, whose epilogue pushes the partial into the hop's mailbox; the MoE input's
all-reduce then only consumes it.
"""

from __future__ import annotations

import logging

import msgspec
import torch

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/qwen3_5.py": (
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
        "qwen3_5_gdn_prefill_projection_views",
    ),
    "srt/layers/radix_linear_attention.py": (
        "layer_id",
        "conv_weights",
        "bias",
        "activation",
        "q_dim",
        "k_dim",
        "v_dim",
        "num_q_heads",
        "num_k_heads",
        "num_v_heads",
        "head_q_dim",
        "head_k_dim",
        "head_v_dim",
        "A_log",
        "dt_bias",
    ),
    "kernels/ops/attention/fla/layernorm_gated.py": (
        "eps",
        "weight",
        "activation",
        "group_size",
        "norm_before_gate",
    ),
    "srt/layers/quantization/modelopt_quant.py": (
        "ModelOptFp8LinearMethod",
        "use_marlin",
        "use_sm120_fp8",
        "use_flashinfer_bmm",
        '"input_scale"',
        '"weight_scale"',
    ),
    "srt/layers/linear.py": (
        "reduce_results",
        "use_dp_attention_reduce",
        "use_decode_attn_tp",
        "input_is_parallel",
        "bias",
    ),
    "srt/layers/quantization/fp8_utils.py": ("flashinfer_bmm_fp8",),
    "srt/distributed/device_communicators/pynccl_allocator.py": (
        "use_symmetric_memory",
    ),
    "srt/layers/dp_attention.py": ("is_allocation_symmetric",),
    "srt/runtime_context.py": ("get_parallel",),
    "srt/model_executor/forward_batch_info.py": ("req_pool_indices",),
    "srt/kernel_plugins/nvidia/eagle_binding.py|srt/megakernels/eagle_binding.py": (
        "mamba",
        "tokens_per_request",
        "gdn",
        "state_indices",
        "query_start_loc",
        "intermediate_indices",
        '"ForwardBatch.req_pool_indices"',
        "layer_ids",
        "conv",
        "temporal",
        "intermediate_conv_window",
        "ring_d",
        "ring_k",
        "ring_g",
        "ring_rawv",
        "ring_rawk",
        "ring_beta",
        "ring_write_pos",
        "ring_cache_base",
        "ring_is_flush",
    ),
}

# The SGLang functions whose body this module mirrors, by the digest of their code
# (test/unit/test_contract.py recomputes them on the installed SGLang)
SGLANG_MIRRORS = {
    # Without and with the early return of the GDN extend in breakable prefill graphs,
    # which target verify never takes; the first goes once all supported SGLangs have it
    "srt/layers/attention/linear/gdn_backend.py:GDNAttnBackend.forward_extend": (
        "ab7b048fafb07d0e",
        "12a835fc4f78af33",
    ),
    # Importing the ReplaySSM kernel from fla/ and, after sgl-project/sglang#45112, from
    # linear/; the first goes once all supported SGLangs have the move
    "srt/layers/attention/linear/gdn_backend.py:GDNAttnBackend._replayssm_target_verify": (
        "28b89fa4b48b47ad",
        "739e7115c370ac01",
    ),
    "srt/layers/linear.py:RowParallelLinear.forward": "b291a3c452e428a5",
    "srt/layers/quantization/fp8_utils.py:apply_fp8_linear_bmm_flashinfer": "c66562e1de0008db",
    "srt/layers/quantization/modelopt_quant.py:ModelOptFp8LinearMethod.apply": "91b563cf074bff42",
}

RING_LEN, MAX_DRAFT = 16, 8  # the verify tiles' bounds, as measured

# the step widths that have run the slot (each logged once, as engagement evidence)
_ENGAGED: set = set()


class Options(msgspec.Struct, frozen=True):
    """The launch choices; ``STOCK`` is SGLang's own."""

    views: bool  # token-major projection views and conv output
    block_n: int  # the conv update's channel tile
    block_v: int  # the ReplaySSM verify's value tile
    num_warps: int
    norm_fp8: bool  # the gated norm writes out_proj's static FP8 input
    pdl: bool = False  # the verify launches dependent on the conv (slot gdn_pdl)
    push: bool = False  # out_proj pushes its partial into the hop (slot gdn_push)


STOCK = Options(views=False, block_n=256, block_v=64, num_warps=1, norm_fp8=False)
SLOT = Options(views=True, block_n=128, block_v=32, num_warps=2, norm_fp8=True)
SLOT_PDL = msgspec.structs.replace(SLOT, pdl=True)


def engaged(slot: str, rows: int) -> None:
    """Log once per (slot, rows) that a GDN slot ran, as engagement evidence."""
    if (slot, rows) not in _ENGAGED:
        _ENGAGED.add((slot, rows))
        logger.info("NVIDIA megakernel: Qwen3.5 slot %s runs at %d rows", slot, rows)


def unsupported(gdn) -> str:
    """Why the slot cannot serve ``gdn`` (a Qwen3_5GatedDeltaNet) ("" when it can)."""
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8LinearMethod

    attn, norm, out = gdn.attn, gdn.norm, gdn.out_proj
    if attn.head_k_dim != 128 or attn.head_v_dim != 128:
        return f"head dims {attn.head_k_dim} / {attn.head_v_dim}"
    if norm.group_size is not None or not norm.norm_before_gate:
        return "a grouped or gate-first norm"
    method = out.quant_method
    if (
        not isinstance(method, ModelOptFp8LinearMethod)
        or method.use_marlin
        or method.use_sm120_fp8
        or not out.use_flashinfer_bmm
        or out.input_scale.numel() != 1
    ):
        return "out_proj is not a static per-tensor FP8 linear on FlashInfer's BMM"
    if (
        out.reduce_results
        or out.use_dp_attention_reduce
        or out.use_decode_attn_tp
        or not out.input_is_parallel
        or out.bias is not None
    ):
        return "out_proj reduces its output, splits its input or adds a bias"
    return ""


def pool_reasons(pools, tokens_per_request: int) -> list[str]:
    """Why the slot cannot run on the mamba pools (``Step.view.mamba``) at
    ``tokens_per_request`` verify rows ([] when it can): the circular ReplaySSM ring
    of the verify tiles' bounds, BF16 checkpoints."""
    reasons = []
    if pools.ring_d is None or pools.ring_cache_base is None:
        reasons.append("no circular ReplaySSM verify ring")
    elif pools.ring_d.shape[-2] != RING_LEN:
        reasons.append(f"ring length {pools.ring_d.shape[-2]}")
    if pools.temporal.dtype != torch.bfloat16:
        reasons.append(f"{pools.temporal.dtype} checkpoints")
    if tokens_per_request > MAX_DRAFT:
        reasons.append(f"{tokens_per_request} rows a request")
    return reasons


def step(gdn, x: torch.Tensor, st, opts: Options = SLOT) -> torch.Tensor:
    """``Qwen3_5GatedDeltaNet.forward`` of a target-verify step with the launch
    choices ``opts``: the TP-partial output [rows, hidden] of the normed input x."""
    import triton
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
        fused_qkvzba_split_reshape_cat_contiguous,
        qwen3_5_gdn_prefill_projection_views,
    )

    from sgl_nvidia_mega.qwen3_5.gdn.norm_fp8 import rms_norm_gated_fp8

    if x.shape[0] not in _ENGAGED:
        _ENGAGED.add(x.shape[0])
        logger.info(
            "NVIDIA megakernel: Qwen3.5 slot gdn_verify runs at %d rows", x.shape[0]
        )
    qkvz, ba = gdn._forward_input_proj(x)
    split = (
        qwen3_5_gdn_prefill_projection_views
        if opts.views
        else fused_qkvzba_split_reshape_cat_contiguous
    )
    mixed_qkv, z, b, a = split(
        qkvz,
        ba,
        triton.cdiv(gdn.num_k_heads, gdn.attn_tp_size),
        triton.cdiv(gdn.num_v_heads, gdn.attn_tp_size),
        gdn.head_k_dim,
        gdn.head_v_dim,
    )
    core = _core(gdn.attn, mixed_qkv, a, b, st, opts)
    core = core.reshape(-1, core.shape[-1])
    nv = gdn.num_v_heads // gdn.attn_tp_size
    if opts.norm_fp8:
        norm = gdn.norm
        fp8 = rms_norm_gated_fp8(
            core,
            norm.weight,
            norm.eps,
            z if opts.views else z.reshape(-1, z.shape[-1]),
            gdn.out_proj.input_scale,
            activation=norm.activation,
            early_dependents=opts.push,
        )
        x = fp8.reshape(-1, nv * gdn.head_v_dim)
        if opts.push:
            return st.bound["boundary"].hop.out_proj_push(
                x_fp8=x,
                packed_w=st.bound["gdn"].packed[gdn.attn.layer_id],
                input_scale=gdn.out_proj.input_scale,
                weight_scale=gdn.out_proj.weight_scale,
            )
        return _out_proj(gdn.out_proj, x, core.dtype)
    z_shape = z.shape
    core = gdn.norm(core, z if opts.views else z.reshape(-1, z.shape[-1]))
    core = core.reshape(z_shape)
    core = core.reshape(*core.shape[:-2], core.shape[-2] * core.shape[-1])
    return gdn.out_proj(core)[0]


def _out_proj(out_proj, fp8: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``RowParallelLinear.forward`` of the FP8 ``out_proj`` on its pre-quantized input:
    ModelOpt's FlashInfer BMM with the static input scale as the activation scale (what
    its own quantization returns), the TP-partial output in SGLang's symmetric pool."""
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        use_symmetric_memory,
    )
    from sglang.srt.layers.dp_attention import is_allocation_symmetric
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_bmm_fp8
    from sglang.srt.runtime_context import get_parallel

    with use_symmetric_memory(
        get_parallel().tp_group, disabled=not is_allocation_symmetric()
    ):
        return flashinfer_bmm_fp8(
            fp8, out_proj.weight, out_proj.input_scale, out_proj.weight_scale, dtype
        )


def _core(attn, mixed_qkv, a, b, st, opts):
    """The backend's target-verify arm for the circular ReplaySSM ring."""
    from sgl_nvidia_mega.qwen3_5.gdn import replayssm
    from sgl_nvidia_mega.qwen3_5.gdn.conv import conv_update

    meta, pools = st.view.gdn, st.view.mamba
    li = pools.layer_ids.index(attn.layer_id)
    seq_len = mixed_qkv.shape[0]
    draft = st.binding.tokens_per_request
    bs = seq_len // draft
    states = meta.state_indices
    y = conv_update(
        mixed_qkv.view(bs, draft, -1).transpose(1, 2),
        pools.conv[0][li],
        attn.conv_weights,
        attn.bias,
        attn.activation,
        states[:bs],
        pools.intermediate_conv_window[0][li],
        meta.intermediate_indices[:bs],
        block_n=opts.block_n,
        token_major_out=opts.views,
    )
    mixed_qkv = y.transpose(1, 2).view(seq_len, -1)
    query, key, value = torch.split(
        mixed_qkv, [attn.q_dim, attn.k_dim, attn.v_dim], dim=-1
    )
    H, K = attn.num_k_heads, attn.head_k_dim
    HV, V = attn.num_v_heads, attn.head_v_dim
    q = query.reshape(seq_len, H, K)
    k = key.reshape(seq_len, H, K)
    v = value.reshape(seq_len, HV, V)
    d_cache = pools.ring_d[li]
    out = q.new_empty(seq_len, HV, V)

    def ring(t):
        return None if t is None else t[li]

    replayssm.verify(
        q,
        k,
        v,
        a.reshape(seq_len, HV),
        b.reshape(seq_len, HV),
        attn.A_log,
        attn.dt_bias,
        out,
        pools.temporal[li],
        d_cache,
        pools.ring_k[li],
        pools.ring_g[li],
        ring(pools.ring_rawv),
        ring(pools.ring_rawk),
        ring(pools.ring_beta),
        meta.query_start_loc,
        states,
        st.fb.req_pool_indices,
        pools.ring_write_pos,
        pools.ring_cache_base,
        pools.ring_is_flush,
        max_cache_len=d_cache.shape[-2],
        max_spec_len=draft,
        scale=K**-0.5,
        null_block_id=-1,
        block_v=opts.block_v,
        num_warps=opts.num_warps,
        pdl=opts.pdl,
    )
    return out.reshape(1, seq_len, HV, V)
