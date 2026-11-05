# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``kv4_dec`` (``MUSE_VR_KV4_DEC=1``, default off): the verify-attention components of ``verify_attn``
serve the sliding-window layers of a mixed NVFP4 / FP8 KV pool.

With ``nvfp4_kv`` and its ``swa_fp8`` part, the full-attention layers hold NVFP4 KV and the sliding-window layers FP8
KV, but the trtllm_mha backend describes its KV format per backend, not per layer: it reports the NVFP4 format for
every layer, and ``nvfp4_kv_mixed`` switches it to the FP8-KV settings only inside ``forward_extend`` /
``forward_decode`` for the FP8 layers. ``verify_attn`` decides at the model's ``gated_attention`` (before that switch)
and therefore steps aside on every layer of such a pool ("kv cache format").

This component wraps ``MuseSpark1xAttention.gated_attention`` OUTSIDE ``verify_attn``'s hook (it must be registered
after ``verify_attn``): for a target-verify call of a layer whose KV lives in the FP8 sliding-window sub-pool, the rest
of the chain (``verify_attn``'s hook, then SGLang's method) runs with the backend in that layer's FP8 view
(``nvfp4_kv_mixed``: KV data type FP8, native-FP4 flags off, the sub-pool's quant method) - the state in which the
FP8-KV recipe runs the layer. Every check, kernel and number of ``verify_attn`` then applies as in the FP8-KV recipe;
the layer's KV buffers, page table, write locations and scales are the sub-pool's, as ``nvfp4_kv_mixed`` already
arranges for the stock path. Full-attention (NVFP4) layers and every other call pass through unchanged (their verify
attention is ``nvfp4_kv``'s, with its ``verify_wide`` sub-part for many requests). The view is resolved in Python before
any kernel launch, so CUDA-graph capture records each layer's own kernels.
"""

from __future__ import annotations

import logging

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
}


logger = logging.getLogger("muse_vr_plugin")

NAME = "kv4_dec"
TARGET = "sglang.srt.models.muse_spark.MuseSpark1xAttention.gated_attention"
PINS = {
    "sglang/srt/models/muse_spark.py": (
        "7e5018b3cb2ef41ee6cf09407e91e28dde069e5a33719ea58beb0d7137087355",
        "8bafbb8802fb0d9f81c190e87b9012fca546182be08bef6c8b3db814d27bfe19",
    ),
    "sglang/srt/layers/attention/trtllm_mha_backend.py": (
        "566d1031d644f71a8adb84d1e2a1581d3a6d1c3cac1e507451ac4cf034c39da4",
        "07223cbf82ab5ac14fcbaa9a47a8351132991016bae66c122bf09762ed8ed646",
    ),
}


def register() -> bool:
    if not bed.enabled(NAME):
        return False
    from . import nvfp4_kv_mixed

    if not nvfp4_kv_mixed.requested():
        bed.report(
            NAME,
            False,
            reason="needs MUSE_VR_NVFP4_KV_SWA_FP8=1 (mixed NVFP4 / FP8 pool)",
        )
        return False
    pins_ok = all(bed.pinned(rel, sha) for rel, sha in PINS.items())
    if not pins_ok:
        bed.report(NAME, False, reason="pins")
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(TARGET, _around_gated_attention, HookType.AROUND)
    bed.report(NAME, True, part="swa_fp8_view")
    return True


def _around_gated_attention(original, attn, positions, qkv, forward_batch):
    if not forward_batch.forward_mode.is_target_verify():
        return original(attn, positions, qkv, forward_batch)
    from sglang.srt.model_executor.forward_context import get_attn_backend

    from . import nvfp4_kv_mixed

    backend = get_attn_backend()
    sub = nvfp4_kv_mixed._fp8_sub_pool_of(
        getattr(backend, "token_to_kv_pool", None), attn.attn.layer_id
    )
    if sub is None:
        return original(attn, positions, qkv, forward_batch)
    bed.count(NAME, "fp8_view")
    with nvfp4_kv_mixed._fp8_view(backend, sub):
        return original(attn, positions, qkv, forward_batch)
