# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Portions Copyright 2023-2024 SGLang Team (Apache-2.0)
# Derived from SGLang (https://github.com/sgl-project/sglang), Apache License 2.0; see include/LICENSE.sglang.
# build_hybrid_swa_kv_pool follows KVCacheConfigurator._build_hybrid_swa_kv_pool (python/sglang/srt/mem_cache/
# kv_cache_configurator.py at the pinned revision).
# Modified by NVIDIA: the NVFP4 hybrid sliding-window pool passes the FP4 KV quant method to both sub-pools (or, with
# the swa_fp8 part, builds the sliding-window sub-pool as a plain FP8 pool), tags the sub-pools with their global layer
# ids (per-layer KV scales) and never takes the post-capture (VA-backed) path, which SGLang implements for unquantized
# pools only; fix_hybrid_sizer gives the hybrid sizer the NVFP4 bytes per token of the NVFP4 layers.
"""NVFP4 hybrid sliding-window KV pool for the ``nvfp4_kv`` component (part ``swa_pool``)."""

from __future__ import annotations

import logging

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/mem_cache/memory_pool.py": ("MHATokenToKVPool",),
    "srt/mem_cache/swa_memory_pool.py": ("SWAKVPool",),
    "srt/runtime_context.py": ("get_exec", "get_parallel", "get_spec"),
    "srt/layers/quantization/fp4_kv_cache_quant_method.py": (
        "get_kv_cache_quant_method",
    ),
    "srt/configs/model_config.py": ("AttentionArch",),
}


logger = logging.getLogger("muse_vr_plugin")


def _quant_method(kvc):
    """The NVFP4 quant method shared by both sub-pools (scales are indexed by global layer id)."""
    qm = kvc._build_mha_quant_method(num_layers=kvc.layer_info.num_effective_layers)
    if qm is None or getattr(qm, "name", None) != "nvfp4":
        raise RuntimeError(
            f"nvfp4_kv: expected the nvfp4 KV quant method, got {getattr(qm, 'name', qm)!r}"
        )
    if not qm.needs_native_fp4_scales() or qm.needs_linear_scale_buffer():
        raise NotImplementedError(
            "nvfp4_kv: the hybrid sliding-window pool is built for the native trtllm-gen layout only "
            "(native NVFP4 prefill and decode on trtllm_mha)."
        )
    # Sub-pools receive writes with a sub-pool-local layer index; the pool's scale lookup then needs the local -> global
    # map that the kv_scales part installs (see nvfp4_kv_scales). All scales 1.0 (no calibration) is exact without it.
    from . import nvfp4_kv_scales

    scales = list(qm.k_scales_float) + list(qm.v_scales_float)
    if any(s != 1.0 for s in scales) and not nvfp4_kv_scales.engaged():
        raise NotImplementedError(
            "nvfp4_kv: per-layer KV scales with the hybrid sliding-window pool need the kv_scales part."
        )
    return qm


def build_hybrid_swa_kv_pool(
    kvc,
    *,
    full_max_total_num_tokens,
    swa_max_total_num_tokens,
    mha_pool_class,
):
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
    from sglang.srt.runtime_context import get_exec, get_parallel, get_spec

    if kvc.is_hybrid_swa_compress:
        raise NotImplementedError(
            "nvfp4_kv: compressed hybrid sliding-window geometry is not supported."
        )
    if mha_pool_class is not MHATokenToKVPool:
        raise NotImplementedError(
            f"nvfp4_kv: the NVFP4 hybrid pool needs MHATokenToKVPool sub-pools, got {mha_pool_class.__name__}."
        )
    if kvc.post_capture_kv_active:
        raise ValueError(
            "nvfp4_kv: post-capture KV sizing is not implemented for quantized KV pools; "
            "serve with SGLANG_ENABLE_POST_CAPTURE_KV_SIZING=0."
        )
    from . import nvfp4_kv_scales

    qm = _quant_method(kvc)
    size_swa = (
        full_max_total_num_tokens
        if kvc.draft_swa_full_capacity
        else swa_max_total_num_tokens
    )
    from . import nvfp4_kv_mixed

    mixed = nvfp4_kv_mixed.requested()
    pool = SWAKVPool(
        size=full_max_total_num_tokens,
        size_swa=size_swa,
        page_size=kvc.pool_page_size,
        dtype=kvc.kv_cache_dtype,
        post_capture_active=False,
        head_num=kvc.model_config.get_num_kv_heads(
            get_parallel().attn_tp_size, get_parallel().attn_dcp_size
        ),
        head_dim=kvc.model_config.head_dim,
        swa_attention_layer_ids=kvc.layer_info.swa_attention_layer_ids,
        full_attention_layer_ids=kvc.layer_info.full_attention_layer_ids,
        device=kvc.device,
        enable_kv_cache_copy=(get_spec().speculative_algorithm is not None),
        token_to_kv_pool_class=MHATokenToKVPool,
        swa_kv_pool_class=nvfp4_kv_mixed.fp8_sub_pool if mixed else None,
        enable_memory_saver=get_exec().features.enable_memory_saver,
        quant_method=qm,
    )
    for sub in (pool.full_kv_pool,) if mixed else (pool.full_kv_pool, pool.swa_kv_pool):
        if sub.quant_method is not qm or sub.native_k_scale_buffer is None:
            raise RuntimeError(
                "nvfp4_kv: sub-pool was not built with the native NVFP4 layout"
            )
    if mixed:
        import torch

        sw = pool.swa_kv_pool
        if (
            sw.dtype != torch.float8_e4m3fn
            or sw.is_quantized_kv_cache
            or getattr(sw, "native_k_scale_buffer", None)
        ):
            raise RuntimeError(
                "nvfp4_kv: the sliding-window sub-pool of the mixed pool is not a plain FP8 pool"
            )
        pool._muse_vr_swa_fp8 = True
    nvfp4_kv_scales.tag_sub_pools(pool)
    logger.warning(
        "muse_vr_plugin: nvfp4_kv hybrid pool: full %d tokens x %d layers (NVFP4), swa %d tokens x %d layers (%s), "
        "%.2f GB",
        pool.size,
        pool.full_layer_nums,
        pool.size_swa,
        pool.swa_layer_nums,
        "FP8" if mixed else "NVFP4",
        pool.mem_usage,
    )
    return pool


def nvfp4_bytes_per_token_layer(
    kvc, head_num: int, head_dim: int, v_head_dim: int
) -> int:
    """Device bytes one token costs in one layer of a native NVFP4 pool (K + V data and block scales)."""
    from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
        get_kv_cache_quant_method,
    )

    if head_dim != v_head_dim:
        raise NotImplementedError(
            "nvfp4_kv: asymmetric K/V head dims are not supported"
        )
    qm = get_kv_cache_quant_method(
        "nvfp4", num_layers=1, device=kvc.device, page_size=kvc.page_size
    )
    qm.configure_attention_backends_from_server_args(kvc.server_args)
    return int(qm.compute_cell_size(head_num, head_dim, 1, 1))


def fix_hybrid_sizer(sizer, kvc) -> None:
    """Replace the hybrid sizer's one-byte-per-element FP4 estimate with the NVFP4 pool's real bytes."""
    from sglang.srt.configs.model_config import AttentionArch
    from sglang.srt.runtime_context import get_parallel

    mc = kvc.model_config
    if mc.attention_arch == AttentionArch.MLA:
        raise NotImplementedError("nvfp4_kv: MLA hybrid pools are not supported")
    tp = get_parallel().attn_tp_size
    n_full, n_swa = mc.get_num_kv_heads(tp), mc.get_swa_num_kv_heads(tp)
    old_full = n_full * (mc.head_dim + mc.v_head_dim)
    old_swa = n_swa * (mc.swa_head_dim + mc.swa_v_head_dim)
    if sizer._full_per_token != old_full or sizer._swa_per_token != old_swa:
        raise RuntimeError(
            "nvfp4_kv: hybrid sizer per-token bytes are not the expected one-byte-per-element estimate "
            f"({sizer._full_per_token}/{sizer._swa_per_token} vs {old_full}/{old_swa})"
        )
    from . import nvfp4_kv_mixed

    sizer._full_per_token = nvfp4_bytes_per_token_layer(
        kvc, n_full, mc.head_dim, mc.v_head_dim
    )
    if (
        not nvfp4_kv_mixed.requested()
    ):  # mixed pool: the sliding-window layers are FP8, one byte per element (as is)
        sizer._swa_per_token = nvfp4_bytes_per_token_layer(
            kvc, n_swa, mc.swa_head_dim, mc.swa_v_head_dim
        )
    sizer._recompute_cell_size()
    logger.warning(
        "muse_vr_plugin: nvfp4_kv sizer: %d / %d B per token per full / swa layer (was %d / %d)",
        sizer._full_per_token,
        sizer._swa_per_token,
        old_full,
        old_swa,
    )
