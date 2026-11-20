# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Portions Copyright 2023-2024 SGLang Team (Apache-2.0)
# Derived from SGLang (https://github.com/sgl-project/sglang), Apache License 2.0; see include/LICENSE.sglang.
# handle_kv4_compatibility follows the step of the same name in python/sglang/srt/arg_groups/kv_cache_hook.py at the
# pinned revision.
# Modified by NVIDIA: (1) DFLASH is accepted with native NVFP4 target KV when the draft runs fa4; its draft KV pool
# stays in the model dtype (SGLang's fa4 override), so the draft-only prefix-valid writer never touches the native
# NVFP4 pool, and the draft keeps fa4 instead of being moved to trtllm_mha. (2) HiCache is accepted when the plugin's
# NVFP4 host pool is installed (page_first layout, kernel I/O, no storage backend). (3) Post-capture KV sizing, which
# SGLang does not implement for quantized pools, is refused at argument time. Every other check is unchanged.
"""``handle_kv4_compatibility`` resolution step for the ``nvfp4_kv`` component (part ``resolution``)."""

from __future__ import annotations

import logging
from typing import Any, Callable

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/arg_groups/overrides.py": (
        "attention_backends_of",
        "declare_resolution",
        "post_capture_kv_sizing_planned",
        "resolved_view",
        "resolving_view",
        "use_mla_backend",
    ),
    "srt/runtime_context.py": ("get_platform",),
}


logger = logging.getLogger("muse_vr_plugin")

_DFLASH_DRAFT_BACKENDS_WITH_OWN_POOL = ("fa4",)


def handle_kv4_compatibility(
    server_args: Any, previous: Callable[[Any], None], *, allow_hicache: bool
) -> None:
    from sglang.srt.arg_groups.overrides import (
        attention_backends_of,
        declare_resolution,
        post_capture_kv_sizing_planned,
        resolved_view,
        resolving_view,
        use_mla_backend,
    )
    from sglang.srt.runtime_context import get_platform

    cfg = resolving_view(server_args)
    if cfg.kv_cache_dtype != "nvfp4" or not get_platform().is_cuda:
        return previous(server_args)

    uses_mla = use_mla_backend(server_args)
    prefill_backend, decode_backend = attention_backends_of(resolved_view(server_args))
    attention_backend = resolved_view(server_args).attention_backend

    if not (get_platform().is_sm100 or get_platform().is_sm120):
        raise RuntimeError(
            "--kv-cache-dtype=nvfp4 requires Blackwell SM100 or SM120. "
            "Use --kv-cache-dtype=fp4_mx_block16 for the block-size-16 FP4 recipe."
        )
    if cfg.enable_unified_memory:
        raise ValueError(
            "FP4 KV cache does not yet support --enable-unified-memory: "
            "the unified MHA pool does not allocate FP4 block scales or "
            "the prefill dequant workspace."
        )

    uses_sm100_trtllm_nvfp4 = get_platform().is_sm100 and "trtllm_mha" in (
        prefill_backend,
        decode_backend,
    )
    uses_mixed_nvfp4 = (
        prefill_backend == "flashinfer" and decode_backend == "trtllm_mha"
    )
    uses_sm100_mixed_nvfp4 = uses_sm100_trtllm_nvfp4 and uses_mixed_nvfp4
    speculative_algorithm = (
        cfg.speculative_algorithm.upper()
        if cfg.speculative_algorithm is not None
        else None
    )
    supported_native_spec_algorithms = {"EAGLE", "EAGLE3", "NEXTN", "NGRAM"}
    draft_backend = cfg.speculative_draft_attention_backend
    dflash_own_draft_pool = (
        speculative_algorithm == "DFLASH"
        and draft_backend in _DFLASH_DRAFT_BACKENDS_WITH_OWN_POOL
    )
    if (
        uses_sm100_trtllm_nvfp4
        and speculative_algorithm in ("DFLASH", "DSPARK")
        and not dflash_own_draft_pool
    ):
        raise ValueError(
            "SM100 native NVFP4 speculative decoding does not yet support "
            f"{speculative_algorithm}; use EAGLE/NEXTN or NGRAM (muse_vr_plugin nvfp4_kv: DFLASH is accepted "
            "with --speculative-draft-attention-backend fa4, whose draft KV pool stays in the model dtype)."
        )
    if (
        uses_sm100_trtllm_nvfp4
        and speculative_algorithm is not None
        and speculative_algorithm not in supported_native_spec_algorithms
        and not dflash_own_draft_pool
    ):
        raise ValueError(
            "SM100 native NVFP4 speculative decoding supports EAGLE, "
            "EAGLE3, NEXTN, and breadth-1 NGRAM; got "
            f"{speculative_algorithm}."
        )
    if (
        uses_sm100_trtllm_nvfp4
        and speculative_algorithm == "NGRAM"
        and cfg.speculative_ngram_max_bfs_breadth != 1
    ):
        raise ValueError(
            "SM100 native NVFP4 NGRAM speculative decoding requires "
            "--speculative-ngram-max-bfs-breadth=1 because trtllm_mha "
            "supports linear target verification only; got "
            f"{cfg.speculative_ngram_max_bfs_breadth}."
        )
    uses_draft_model = speculative_algorithm not in (None, "NGRAM")
    if uses_sm100_trtllm_nvfp4 and uses_draft_model and not dflash_own_draft_pool:
        if draft_backend is None:
            logger.warning(
                "SM100 native NVFP4 speculative decoding uses trtllm_mha for the draft worker."
            )
            declare_resolution(
                server_args,
                "_handle_kv4_compatibility",
                speculative_draft_attention_backend="trtllm_mha",
            )
        elif draft_backend != "trtllm_mha":
            raise ValueError(
                "SM100 native NVFP4 speculative decoding requires "
                "--speculative-draft-attention-backend=trtllm_mha so the "
                "draft worker consumes its physical NVFP4 KV layout; got "
                f"{draft_backend!r}."
            )
    if dflash_own_draft_pool and uses_sm100_trtllm_nvfp4:
        if uses_mixed_nvfp4:
            raise ValueError(
                "muse_vr_plugin nvfp4_kv: DFLASH with NVFP4 target KV supports native NVFP4 prefill "
                "(--prefill-kv-cache-dequant-dtype auto/nvfp4) only."
            )
        logger.warning(
            "muse_vr_plugin: nvfp4_kv: DFLASH with native NVFP4 target KV; the %s draft keeps its own KV pool in the "
            "model dtype.",
            draft_backend,
        )
    if (
        uses_sm100_mixed_nvfp4
        and cfg.speculative_algorithm is not None
        and cfg.speculative_attention_mode == "prefill"
    ):
        logger.warning(
            "SM100 mixed NVFP4 speculative decoding routes target verify "
            "to trtllm_mha; overriding --speculative-attention-mode=prefill "
            "to decode."
        )
        declare_resolution(
            server_args,
            "_handle_kv4_compatibility",
            speculative_attention_mode="decode",
        )
    if uses_sm100_trtllm_nvfp4 and cfg.disaggregation_mode != "null":
        raise ValueError(
            "SM100 native NVFP4 with trtllm_mha does not yet support PD "
            "disaggregation because its physical block-scale layout is not "
            "implemented by the KV transfer path."
        )
    if uses_sm100_trtllm_nvfp4 and cfg.enable_lmcache:
        raise ValueError(
            "SM100 native NVFP4 with trtllm_mha does not yet support "
            "hierarchical KV cache or LMCache because their host pools do "
            "not preserve the physical block-scale layout."
        )
    if uses_sm100_trtllm_nvfp4 and cfg.enable_hierarchical_cache:
        problems = []
        if not allow_hicache:
            problems.append("the plugin's NVFP4 host pool is not installed")
        if uses_mixed_nvfp4:
            problems.append(
                "mixed FP8-prefill NVFP4 is not supported by the plugin host pool"
            )
        if cfg.hicache_mem_layout != "page_first":
            problems.append(
                f"--hicache-mem-layout must be page_first (got {cfg.hicache_mem_layout!r})"
            )
        if cfg.hicache_io_backend != "kernel":
            problems.append(
                f"--hicache-io-backend must be kernel (got {cfg.hicache_io_backend!r})"
            )
        if cfg.hicache_storage_backend is not None:
            problems.append(
                "no --hicache-storage-backend (L3 pages are not implemented for NVFP4)"
            )
        if problems:
            raise ValueError(
                "SM100 native NVFP4 with trtllm_mha and hierarchical KV cache: "
                + "; ".join(problems)
                + "."
            )
    if uses_sm100_trtllm_nvfp4 and post_capture_kv_sizing_planned(server_args):
        raise ValueError(
            "muse_vr_plugin nvfp4_kv: post-capture KV sizing is not implemented for quantized KV pools; "
            "serve with SGLANG_ENABLE_POST_CAPTURE_KV_SIZING=0."
        )

    if prefill_backend != decode_backend and prefill_backend != "fa4":
        if not uses_mixed_nvfp4:
            logger.warning(
                f"Attention: Using KV4 with PREFILL = {prefill_backend} "
                f"and DECODE = {decode_backend}. "
                "Compatibility issues are unlikely, but may occur in rare "
                "edge cases."
            )
    else:
        if prefill_backend == "fa4":
            if uses_mla:
                choices = ["flashinfer", "trtllm_mla"]
                assert decode_backend in choices, (
                    f"KV4 FA4 MLA expects decode_attention_backend to be one of {choices}, but got {decode_backend}"
                )
            else:
                choices = ["triton", "torch_native", "flex_attention"]
                assert decode_backend in choices, (
                    f"KV4 FA4 MHA expects decode_attention_backend to be one of {choices}, but got {decode_backend}"
                )
        else:
            if uses_mla:
                choices = ["flashinfer", "trtllm_mla"]
                assert attention_backend in choices, (
                    f"KV4 MLA expects attention_backend to be one of {choices}, but got {attention_backend}"
                )
            else:
                choices = ["triton", "torch_native", "flex_attention", "trtllm_mha"]
                assert attention_backend in choices, (
                    f"KV4 MHA expects attention_backend to be one of {choices}, but got {attention_backend}"
                )
