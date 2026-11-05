# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Portions Copyright 2023-2024 SGLang Team (Apache-2.0)
# Derived from SGLang (https://github.com/sgl-project/sglang), Apache License 2.0; see include/LICENSE.sglang.
# _store_around follows MuseSpark1xAttention._fused_kv_store (python/sglang/srt/models/muse_spark.py at the pinned
# revision).
# Modified by NVIDIA: for a layer whose KV lives in the FP8 sliding-window sub-pool of a mixed NVFP4/FP8 hybrid pool, the
# fused QK-norm KV store is enabled with SGLang's FP8 conditions evaluated against that sub-pool instead of the hybrid
# pool's FP4 dtype.
"""Part ``swa_fp8`` of the ``nvfp4_kv`` component: NVFP4 KV on the full-attention layers only, FP8 KV on the
sliding-window layers (``MUSE_VR_NVFP4_KV_SWA_FP8=1``).

With a hybrid sliding-window model nearly all KV bytes of long requests sit in the full-attention layers (the window
caps every sliding-window layer), so storing only those in NVFP4 keeps most of the capacity gain while the
sliding-window layers keep the FP8-KV recipe's storage and kernels. Pieces:

* pool: the hybrid pool's sliding-window sub-pool is a plain FP8 (E4M3) ``MHATokenToKVPool`` (no FP4 quant method);
  the full-attention sub-pool stays native NVFP4. The hybrid sizer charges the sliding-window layers one byte per
  element (FP8) and the full layers the NVFP4 bytes (``nvfp4_kv_pool``).
* attention: SGLang's trtllm_mha backend decides the KV format per backend, not per layer. For a layer whose KV lives
  in the FP8 sub-pool, ``forward_extend`` / ``forward_decode`` run with the backend's FP8-KV settings (KV dtype FP8,
  native-FP4 flags off, the sub-pool's quant method), i.e. exactly the stock FP8-KV code path; full layers keep the
  NVFP4 path. The plugin verify kernel (``nvfp4_kv_verify``) steps aside for FP8 layers.
* model KV store: the model's fused QK-norm + KV store (decode and verify) is enabled by SGLang only when the pool and
  backend dtype are FP8 or BF16; for an FP8 layer the same conditions are evaluated against its FP8 sub-pool, so those
  layers write their KV exactly as the FP8-KV recipe does.

All switches are per layer and resolved in Python before any kernel launch, so CUDA-graph capture records each layer's
own kernels.
"""

from __future__ import annotations

import contextlib
import logging
import os

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/mem_cache/memory_pool.py": ("MHATokenToKVPool",),
    "srt/layers/attention/trtllm_mha_backend.py": ("TRTLLMHAAttnBackend",),
    "srt/layers/cp/utils.py": ("is_cp_active",),
    "srt/model_executor/forward_context.py": (
        "get_attn_backend",
        "get_token_to_kv_pool",
    ),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


logger = logging.getLogger("muse_vr_plugin")

TARGET_EXTEND = (
    "sglang.srt.layers.attention.trtllm_mha_backend.TRTLLMHAAttnBackend.forward_extend"
)
TARGET_DECODE = (
    "sglang.srt.layers.attention.trtllm_mha_backend.TRTLLMHAAttnBackend.forward_decode"
)
TARGET_STORE = "sglang.srt.models.muse_spark.MuseSpark1xAttention._fused_kv_store"
MODEL_PIN = (
    "sglang/srt/models/muse_spark.py",
    (
        "7e5018b3cb2ef41ee6cf09407e91e28dde069e5a33719ea58beb0d7137087355",
        "8bafbb8802fb0d9f81c190e87b9012fca546182be08bef6c8b3db814d27bfe19",
    ),
)

_FLAGS = (
    "prefill_uses_native_fp4",
    "decode_uses_native_fp4",
    "is_nvfp4_kvcache",
    "uses_trtllm_gen_native_fp4",
)
_state: dict = {"fp8_layers": {}, "calls": 0}


def requested() -> bool:
    return os.environ.get("MUSE_VR_NVFP4_KV_SWA_FP8", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def fp8_sub_pool(size, dtype, layer_num, **kwargs):
    """``SWAKVPool`` sliding-window sub-pool factory: a plain FP8 MHA pool (no FP4 quant method)."""
    import torch
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    kwargs.pop("quant_method", None)
    return MHATokenToKVPool(
        size=size, dtype=torch.float8_e4m3fn, layer_num=layer_num, **kwargs
    )


def _fp8_sub_pool_of(pool, layer_id: int):
    """The FP8 sub-pool holding ``layer_id`` of a mixed hybrid pool, else None (cached per pool and layer)."""
    key = (id(pool), int(layer_id))
    hit = _state["fp8_layers"].get(key)
    if hit is not None:
        return hit or None
    sub = None
    mapping = getattr(pool, "layers_mapping", None)
    if (
        mapping is not None
        and layer_id in mapping
        and getattr(pool, "_muse_vr_swa_fp8", False)
    ):
        _, is_swa = mapping[layer_id]
        if is_swa:
            sub = pool.swa_kv_pool
    _state["fp8_layers"][key] = sub if sub is not None else False
    return sub


def layer_is_fp8(backend, layer) -> bool:
    return (
        _fp8_sub_pool_of(getattr(backend, "token_to_kv_pool", None), layer.layer_id)
        is not None
    )


@contextlib.contextmanager
def _fp8_view(backend, sub):
    """Backend attributes of the stock FP8-KV path for the duration of one layer call."""
    import torch

    saved = {
        k: getattr(backend, k) for k in _FLAGS + ("data_type", "kv_cache_quant_method")
    }
    try:
        for k in _FLAGS:
            setattr(backend, k, False)
        backend.data_type = torch.float8_e4m3fn
        backend.kv_cache_quant_method = sub.quant_method
        yield
    finally:
        for k, v in saved.items():
            setattr(backend, k, v)


def _extend_around(original, self, q, k, v, layer, forward_batch, *args, **kwargs):
    sub = _fp8_sub_pool_of(self.token_to_kv_pool, layer.layer_id)
    if sub is None:
        return original(self, q, k, v, layer, forward_batch, *args, **kwargs)
    with _fp8_view(self, sub):
        return original(self, q, k, v, layer, forward_batch, *args, **kwargs)


def _decode_around(original, self, q, k, v, layer, forward_batch, *args, **kwargs):
    return _extend_around(
        original, self, q, k, v, layer, forward_batch, *args, **kwargs
    )


def _store_around(original, self, forward_batch, *args, **kwargs):
    """``MuseSpark1xAttention._fused_kv_store`` for an FP8 layer of a mixed pool: SGLang's conditions with the FP8
    sub-pool's dtype (the hybrid pool reports FP4), then the same (k_cache, v_cache, loc) triple; other layers stock."""
    import torch
    from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
    from sglang.srt.layers.cp.utils import is_cp_active
    from sglang.srt.model_executor.forward_context import (
        get_attn_backend,
        get_token_to_kv_pool,
    )

    pool = get_token_to_kv_pool()
    sub = _fp8_sub_pool_of(pool, self.attn.layer_id)
    if sub is None:
        return original(self, forward_batch, *args, **kwargs)
    backend = get_attn_backend()
    if not (
        isinstance(backend, TRTLLMHAAttnBackend)
        and sub.dtype == torch.float8_e4m3fn
        and not backend.is_xqa_impl
        and self.attn.k_scale is None
        and self.attn.v_scale is None
        and forward_batch.dcp_kv_mask is None
        and (
            forward_batch.forward_mode.is_decode()
            or forward_batch.forward_mode.is_target_verify()
        )
        and not is_cp_active(forward_batch)
    ):
        return None
    cache_loc = backend._get_layer_cache_loc(self.attn, forward_batch)
    k_cache = pool.get_key_buffer(self.layer_idx)
    v_cache = pool.get_value_buffer(self.layer_idx)
    return (
        k_cache.view(k_cache.shape[0], -1),
        v_cache.view(v_cache.shape[0], -1),
        cache_loc,
    )


def install() -> bool:
    if not requested():
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    if not bed.pinned(*MODEL_PIN):
        raise RuntimeError(f"{MODEL_PIN[0]} does not match its pin")
    HookRegistry.register(TARGET_EXTEND, _extend_around, HookType.AROUND)
    HookRegistry.register(TARGET_DECODE, _decode_around, HookType.AROUND)
    HookRegistry.register(TARGET_STORE, _store_around, HookType.AROUND)
    return True
