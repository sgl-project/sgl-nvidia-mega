# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/layers/attention/trtllm_mha_backend.py (TRTLLMHAAttnBackend.forward_extend,
# _fused_fp8_qkv_kv_cache, _get_bmm_scales, _reshape_paged_kv_cache), and from SGLang branch
# nvidia/qwen35-agentx-gb300: the same file's _run_fixed_q_len_decode and _run_skv2
# (use_split_kv with use_skv2).
"""Slot ``skv2``: the full attention of a covered target-verify step with its core
through SKV2 (``sm103/attention/skv2.py``), the split-KV attention of SGLang branch
nvidia/qwen35-agentx-gb300: a CuTe DSL producer over 148 page splits and the Triton
merge, bit for bit the branch's.

``attend`` stands in for ``layer.attn(q, k, v, forward_batch)``: as TRT-LLM MHA's
``forward_extend`` does for a target verify on an FP8 KV cache, it writes the step's
K / V rows and quantizes Q in one ``fused_fp8_qkv_kv_cache``, then runs ``core``.
``core`` takes an FP8 Q whose K / V rows are written (slot ``mka``'s prologue writes
them itself) and runs SKV2 where the step takes split-KV attention (``route``),
SGLang's attention otherwise. It reads the page table, the KV lengths and the pool from
the forward's view (``Step.view``).
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/layers/radix_attention.py": (
        "layer_id",
        "k_scale",
        "v_scale",
        "tp_q_head_num",
        "tp_k_head_num",
        "tp_v_head_num",
        "head_dim",
        "scaling",
        "sliding_window_size",
    ),
    "kernels/ops/kvcache/fused_fp8_qkv_kv_cache.py": (
        "fused_fp8_qkv_kv_cache",
        "fused_fp8_qkv_kv_cache(q)",
        "fused_fp8_qkv_kv_cache(k)",
        "fused_fp8_qkv_kv_cache(v)",
        "fused_fp8_qkv_kv_cache(k_cache)",
        "fused_fp8_qkv_kv_cache(v_cache)",
        "fused_fp8_qkv_kv_cache(cache_loc)",
        "fused_fp8_qkv_kv_cache(k_scale)",
        "fused_fp8_qkv_kv_cache(v_scale)",
    ),
    "kernels/ops/attention/utils.py": ("canonicalize_stride",),
    "srt/kernel_plugins/nvidia/eagle_binding.py|srt/megakernels/eagle_binding.py": (
        "attn",
        "kv",
        "page_table",
        "cache_seqlens",
        "layer_ids",
        "k_scale",
        "v_scale",
        "page_size",
        "tokens_per_request",
    ),
    "srt/model_executor/forward_batch_info.py": ("out_cache_loc",),
    "srt/environ.py": ("envs", "SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR"),
}

# The SGLang functions whose body this module mirrors, by the digest of their code
# (test/unit/test_contract.py recomputes them on the installed SGLang)
SGLANG_MIRRORS = {
    "srt/layers/attention/trtllm_mha_backend.py:TRTLLMHAAttnBackend.forward_extend": "8e36076dafabff66",
    "srt/layers/attention/trtllm_mha_backend.py:TRTLLMHAAttnBackend._should_use_fused_fp8_path": "e0b14791d523a019",
    "srt/layers/attention/trtllm_mha_backend.py:TRTLLMHAAttnBackend._fused_fp8_qkv_kv_cache": "363661c3585e6a0f",
    "srt/layers/attention/trtllm_mha_backend.py:TRTLLMHAAttnBackend._get_bmm_scales": "90a1387e8f561611",
    "srt/layers/attention/trtllm_mha_backend.py:TRTLLMHAAttnBackend._reshape_paged_kv_cache": "53ae20937769b99a",
}

PAGE_SIZE, HEAD_DIM, ROWS = 64, 256, 7
# the producer's compiled plans and storage, per input layout (as the branch keeps them)
_PLANS: dict = {}
# the FP8 KV scale of a layer without checkpoint scales (TRT-LLM MHA's default)
_ONES: dict = {}
# the rows the slot has run at, by route (each logged once, as engagement evidence)
_ENGAGED: set = set()


def unsupported(attn, tokens_per_request: int) -> str:
    """Why SKV2 cannot run this RadixAttention ("" when it can): the branch's
    ``_run_fixed_q_len_decode`` support check, on what build knows."""
    from sglang.srt.environ import envs

    if tokens_per_request != ROWS:
        return f"{tokens_per_request} rows a request (SKV2 runs 7)"
    if attn.tp_q_head_num not in (4, 8) or attn.tp_k_head_num != 1:
        return (
            f"{attn.tp_q_head_num} query / {attn.tp_k_head_num} KV heads a rank "
            "(SKV2 runs 4 or 8 / 1)"
        )
    if attn.head_dim != HEAD_DIM:
        return f"head_dim {attn.head_dim} (SKV2 runs 256)"
    if attn.sliding_window_size >= 0:
        return f"a sliding window of {attn.sliding_window_size}"
    if envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get() not in (None, 0.0):
        return "SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR is set"
    if torch.cuda.get_device_capability()[0] != 10:
        return "SKV2 runs on SM10x"
    return ""


def pool_reasons(kv, page_size: int) -> list[str]:
    """Why SKV2 cannot read these KV pools ([] when it can)."""
    reasons = []
    if page_size != PAGE_SIZE:
        reasons.append(f"page size {page_size} (SKV2 reads pages of 64)")
    for buf in (*kv.k, *kv.v):
        if buf.dtype != torch.float8_e4m3fn or not buf.is_contiguous():
            reasons.append(f"a {buf.dtype} KV pool buffer that is not contiguous FP8")
            break
        if buf.numel() % (PAGE_SIZE * HEAD_DIM):
            reasons.append(f"a KV pool buffer of shape {tuple(buf.shape)}")
            break
    return reasons


def kv_scales(attn, device):
    """The FP8 KV write's scale tensors TRT-LLM MHA passes: the layer's, else ones."""
    ones = _ONES.get(device)
    if ones is None:
        ones = _ONES[device] = torch.ones(1, dtype=torch.float32, device=device)
    return (
        attn.k_scale if attn.k_scale is not None else ones,
        attn.v_scale if attn.v_scale is not None else ones,
    )


def attend(attn, q, k, v, st):
    """``attn(q, k, v, st.fb)`` for a covered verify step: the attention output
    [rows, heads * head_dim]."""
    from sglang.kernels.ops.kvcache.fused_fp8_qkv_kv_cache import (
        fused_fp8_qkv_kv_cache,
    )

    i = st.view.kv.layer_ids.index(attn.layer_id)
    k_scale, v_scale = kv_scales(attn, q.device)
    q = fused_fp8_qkv_kv_cache(
        q=q,
        k=k,
        v=v,
        k_cache=st.view.kv.k[i],
        v_cache=st.view.kv.v[i],
        cache_loc=st.fb.out_cache_loc,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    return core(attn, q, st)


def core(attn, q, st):
    """The attention of the FP8 query rows ``q`` over the KV cache, whose rows for this
    step are written: [rows, heads * head_dim] in bf16."""
    from sgl_nvidia_mega.qwen3_5.attention import route

    split = route.split_kv(route.VERIFY, st.fb)
    if (split, q.shape[0]) not in _ENGAGED:
        _ENGAGED.add((split, q.shape[0]))
        logger.info(
            "NVIDIA megakernel: Qwen3.5 slot skv2 runs at %d rows (%s)",
            q.shape[0],
            "SKV2" if split else "SGLang's attention below the split-KV length",
        )
    if not split:
        return attn(q, None, None, st.fb)
    return skv2(attn, q, st.view.kv, st.view.attn)


def skv2(attn, q, kv, meta):
    """SKV2 on one request's 7 FP8 query rows ``q`` of ``attn`` over the pool ``kv``
    (KvPoolBuffers) at ``meta``'s pages and lengths (AttnStepMetadata), whose rows for
    this step are written: [rows, heads * head_dim] in bf16. The draft extend's
    attention (``EXTEND``'s route) runs it on the draft's pool too."""
    from sglang.kernels.ops.attention.utils import canonicalize_stride

    from sgl_nvidia_mega.qwen3_5.sm103.attention.skv2 import StockSplitPlan

    i = kv.layer_ids.index(attn.layer_id)
    # the pool's NHD rows as HND pages, as TRT-LLM MHA hands them to its decode kernel
    caches = tuple(
        canonicalize_stride(
            buf.view(-1, PAGE_SIZE, heads, attn.head_dim).permute(0, 2, 1, 3)
        )
        for buf, heads in ((kv.k[i], attn.tp_k_head_num), (kv.v[i], attn.tp_v_head_num))
    )
    q = q.reshape(-1, attn.tp_q_head_num, attn.head_dim)
    table, lengths = meta.page_table, meta.cache_seqlens
    key = (q.shape, caches[0].shape, table.shape, table.stride(), lengths.shape)
    plan = _PLANS.get(key)
    if plan is None:
        # compiles the producer: the first covered step runs eagerly, before capture
        plan = _PLANS[key] = StockSplitPlan(q, caches, table, lengths)
    out = torch.empty_like(q, dtype=torch.bfloat16)
    # TRT-LLM MHA's BMM scales with Q unscaled: q_scale * k_scale * scaling, v_scale
    plan.run(
        q,
        caches,
        table,
        lengths,
        qk_scale=float(1.0 * kv.k_scale[i] * attn.scaling),
        v_scale=float(kv.v_scale[i]),
        out=out,
    )
    return out.view(-1, attn.tp_q_head_num * attn.head_dim)
