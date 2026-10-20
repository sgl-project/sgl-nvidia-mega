# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/deepseek_v4.py (MQALayer.forward).
"""The single ops of SGLang's DeepSeek-V4.1 decode attention on CUDA, as the stock
step runs them: the query path, the window KV store, and wo_a / wo_b.

Each function is the CUDA branch of the ``MQALayer`` method it names, on the layer's
own weights and SGLang's kernels, so the step stays bitwise to SGLang's."""

from __future__ import annotations

import torch

# The SGLang names this module reads, by file under the sglang package;
# test/unit/test_contract.py pins them.
SGLANG_READS = {
    "srt/models/deepseek_v4.py": (
        "_FP8_WO_A_UE8M0",
        "_FUSED_WO_A_MAX_TOKENS",
        "_apply_wo_a_bf16_matmul",
        "_fp8_wo_a_einsum",
        "use_flashinfer_mxfp8_wo_b",
        "use_fused_wo_a",
        "wo_a_fp8",
        # the module alias SGLang's DSV4 model imports too
        "mhc",
    ),
    "srt/models/deepseek_v4_mhc.py": ("AttnOutput",),
    "kernels/ops/attention/dsv4/__init__.py": (
        "fused_norm_rope_inplace",
        "fused_q_norm_rope",
        "fused_rope_inplace",
    ),
    "kernels/ops/attention/dsv4/wo_a.py": ("fused_rope_wo_a_bf16",),
    "kernels/ops/quantization/fp8_kernel.py": ("sglang_per_token_group_quant_fp8",),
    "srt/layers/quantization/fp8_utils.py": (
        "Mxfp8DenseGemmBackend",
        "FLASHINFER_CUTEDSL",
    ),
    "srt/layers/quantization/mxfp8_input.py": ("Mxfp8SwizzledInput",),
    "srt/layers/moe/utils.py": ("should_skip_mlp_all_reduce",),
    "srt/megakernels/binding.py": (
        "ForwardView",
        "kv_pool",
        "store_cache",
        "swa_out_cache_loc",
        "set_swa_key_buffer_radix_fused_norm_rope",
    ),
    "srt/batch_invariant_ops/*.py": ("is_batch_invariant_mode_enabled",),
    "srt/runtime_context.py": ("get_exec", "get_forward", '"sp_active"'),
    "srt/environ.py": ("envs", "SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE"),
}


def normalize_q_lora(attn, q):
    """``MQALayer._normalize_q_lora``: the BF16 normed row (the indexer's) and wq_b's
    input, quantized with SGLang's fused norm + MXFP8 epilogue at 1-8 rows."""
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
    from sglang.srt.layers.quantization.fp8_utils import Mxfp8DenseGemmBackend
    from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput
    from sglang.srt.runtime_context import get_exec

    from sgl_nvidia_mega.deepseek_v41 import sglang_ops

    method = attn.wq_b.quant_method
    if (
        attn.is_dsv41
        and q.dtype == attn.q_norm.weight.dtype == torch.bfloat16
        and q.ndim == 2
        and 0 < q.shape[0] <= 8
        and q.shape[1] == 1280
        and q.stride(1) == 1
        and getattr(method, "mxfp8_dense_backend", None)
        == Mxfp8DenseGemmBackend.FLASHINFER_CUTEDSL
        and (
            getattr(method, "use_mxfp8", False)
            or getattr(attn.wq_b, "block_fp8_mxfp8_ready", False)
        )
        and not (
            is_batch_invariant_mode_enabled()
            or get_exec().deterministic.enable_deterministic_inference
        )
    ):
        y, quant, scale = sglang_ops.rmsnorm_mxfp8(
            q, attn.q_norm.weight, attn.q_norm.variance_epsilon
        )
        return y, Mxfp8SwizzledInput(quant, scale)
    q = attn.q_norm(q)
    return q, q


def index_q_input(attn, q_lora, q_for_wqb):
    """The indexer wq_b's input, as SGLang passes it: q_lora."""
    return q_lora


def qkv_a(attn, x):
    """``MQALayer``'s fused ``wqkv_a(x)``: [rows, q_lora_rank + head_dim]."""
    from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll

    if isinstance(x, mxfp8_ll.RowTileMxfp8):
        # row-tile scales are not a layout SGLang's wqkv_a reads: never a silent fallback
        raise RuntimeError(
            "NVIDIA megakernel: SGLang's wqkv_a got row-tile MXFP8 scales "
            f"({x.quant.shape[0]} rows)"
        )
    out, _ = attn.wqkv_a(x)
    return out


def compute_q_a(attn, x, qkv_a):
    """``MQALayer._compute_q_a``."""
    if qkv_a is not None:
        q = qkv_a[..., : attn.q_lora_rank]
    else:
        q, _ = attn.wq_a(x)
    return normalize_q_lora(attn, q)


def compute_q_b(attn, q, positions, q_out):
    """``MQALayer._compute_q_b`` on CUDA: wq_b, then RoPE (and the per-head norm with
    ``q_head_norm``) into ``q_out``."""
    from sglang.kernels.ops.attention.dsv4 import fused_q_norm_rope, fused_rope_inplace

    q, _ = attn.wq_b(q)
    q = q.view(-1, attn.n_local_heads, attn.head_dim)
    if not attn.q_head_norm:
        if q_out is not None:
            # the eps None form ropes without a norm, FP8 or BF16 q_out alike
            fused_q_norm_rope(q, q_out, None, attn.freqs_cis, positions)
            return q_out
        fused_rope_inplace(
            q[..., -attn.qk_rope_head_dim :], None, attn.freqs_cis, positions=positions
        )
        return q
    if q_out is None:
        q_out = torch.empty_like(q)
    fused_q_norm_rope(q, q_out, attn.eps, attn.freqs_cis, positions)
    return q_out


def kv_to_cache(attn, x, positions, fb, view, qkv_a):
    """``MQALayer._compute_kv_to_cache`` on CUDA: KV norm + RoPE + FP8 store into the
    paged window cache."""
    from sglang.kernels.ops.attention.dsv4 import fused_norm_rope_inplace
    from sglang.srt.environ import envs

    if envs.SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE.get():
        kv = qkv_a[..., attn.q_lora_rank :] if qkv_a is not None else attn.wkv(x)[0]
        kv = kv.contiguous()
        fused_norm_rope_inplace(
            kv, attn.kv_norm.weight.data, attn.eps, attn.freqs_cis, positions
        )
        view.store_cache(layer_id=attn.layer_id, swa_k=kv, forward_batch=fb)
        return
    kv = qkv_a[..., attn.q_lora_rank :] if qkv_a is not None else attn.wkv(x)[0]
    view.kv_pool.set_swa_key_buffer_radix_fused_norm_rope(
        layer_id=attn.layer_id,
        swa_loc=view.swa_out_cache_loc(fb),
        kv=kv,
        kv_weight=attn.kv_norm.weight.data,
        eps=attn.eps,
        freqs_cis=attn.freqs_cis,
        positions=positions,
    )


def wo_a(attn, o, positions, fb):
    """The CUDA arm of ``MQALayer.forward`` after attention: the inverse RoPE and
    wo_a, as wo_b's input (FlashInfer-swizzled MXFP8 when wo_b takes it)."""
    from sglang.kernels.ops.attention.dsv4 import fused_rope_inplace
    from sglang.kernels.ops.attention.dsv4.wo_a import fused_rope_wo_a_bf16
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )
    from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput
    from sglang.srt.models.deepseek_v4 import (
        _FP8_WO_A_UE8M0,
        _FUSED_WO_A_MAX_TOKENS,
        _apply_wo_a_bf16_matmul,
        _fp8_wo_a_einsum,
    )
    from sglang.srt.runtime_context import get_forward

    from sgl_nvidia_mega.deepseek_v41 import sglang_ops

    fuse_mxfp8_quant = attn.use_flashinfer_mxfp8_wo_b and not get_forward().sp_active
    fuse_rope_wo_a = attn.use_fused_wo_a and 0 < o.shape[0] <= _FUSED_WO_A_MAX_TOKENS
    if not fuse_rope_wo_a:
        fused_rope_inplace(
            o[..., -attn.qk_rope_head_dim :],
            None,
            attn.freqs_cis,
            positions=positions,
            inverse=True,
        )
    o = o.view(o.shape[0], attn.n_local_groups, -1)
    if fuse_rope_wo_a:
        weight = attn.wo_a.weight.view(attn.n_local_groups, attn.o_lora_rank, -1)
        out = fused_rope_wo_a_bf16(
            o,
            weight,
            torch.view_as_real(attn.freqs_cis).flatten(1),
            positions,
            out_mxfp8=fuse_mxfp8_quant,
        )
        return Mxfp8SwizzledInput(*out) if fuse_mxfp8_quant else out[0]
    if attn.wo_a_fp8:
        t, g, d = o.shape
        r = attn.o_lora_rank
        if _FP8_WO_A_UE8M0:
            o_fp8, o_s = sglang_ops.per_token_group_quant_fp8_wo_a(o)
            recipe = (1, 1, 128)
        else:
            o_fp8, o_s = sglang_per_token_group_quant_fp8(
                o.reshape(t * g, d).contiguous(), group_size=128, scale_ue8m0=False
            )
            o_fp8, o_s, recipe = o_fp8.view(t, g, d), o_s.view(t, g, -1), (1, 128, 128)
        return _fp8_wo_a_einsum(
            o_fp8,
            o_s,
            attn.wo_a.weight.view(g, r, d),
            attn.wo_a.weight_scale_inv.data,
            recipe=recipe,
        )
    mode = fb.forward_mode
    weight = attn.wo_a.weight.view(attn.n_local_groups, attn.o_lora_rank, -1)
    return _apply_wo_a_bf16_matmul(
        o,
        weight,
        is_decode=mode.is_decode(),
        is_target_verify=mode.is_target_verify(),
        is_prefill=mode.is_extend_without_speculative(),
        fast_path=attn.is_dsv41,
        fuse_mxfp8_quant=fuse_mxfp8_quant,
        fp8_grid=False,
        emit_fp8=False,
    )


def wo_b(attn, o, publish=None):
    """``MQALayer._project_wo_b(o, defer_all_reduce=True)`` on CUDA: the TP-partial
    output, whose all-reduce the attention post runs. ``publish`` (hop A) runs wo_b
    itself and stores the rows into the Lamport mailbox the post collects."""
    from sglang.srt.layers.moe.utils import should_skip_mlp_all_reduce
    from sglang.srt.models import deepseek_v4_mhc as mhc

    from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll

    if publish is not None:
        partial = publish(attn.wo_b, o.flatten(1) if isinstance(o, torch.Tensor) else o)
        if partial is not None:
            return mhc.AttnOutput(partial)
    if isinstance(o, mxfp8_ll.RowTileMxfp8):
        # row-tile scales are not a layout SGLang's wo_b reads: never a silent fallback
        raise RuntimeError(
            "NVIDIA megakernel: wo_b got row-tile MXFP8 scales but hop A did not "
            f"publish {o.quant.shape[0]} rows"
        )
    defer = attn.attn_tp_size > 1 and not should_skip_mlp_all_reduce()
    out, _ = attn.wo_b(
        o.flatten(1) if isinstance(o, torch.Tensor) else o, skip_all_reduce=defer
    )
    if defer:
        return mhc.AttnOutput(out)
    return out
