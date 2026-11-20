# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Part ``verify_kernel`` of the ``nvfp4_kv`` component: DFLASH target verify over the native NVFP4 KV pool runs the
plugin's tensor-core kernel (``ops/nvfp4_verify_attn.py``) instead of the trtllm-gen E2M1-KV generation kernels.

The stock E2M1-KV generation kernels exist only as one-token-per-CTA tiles (no token + head grouping for mixed Q/KV
dtypes), so a verify step streams the KV once per verify token. The plugin kernel serves all 8 tokens x 8 query heads
of a KV head in one 64-row MMA, as the FP8-KV kernels do.

Engaged per call (else the stock path runs unchanged) when: the backend is trtllm_mha with native NVFP4 decode, the
batch is TARGET_VERIFY with a fixed 8 rows per request (not ragged, no context parallelism), the layer is a decoder
layer without attention sinks, q is bf16 with 8 query heads per KV head, head_dim 64, an even KV-head count per rank and
page size 128. The KV write is SGLang's own (``set_kv_buffer`` with the batch's write locations), then attention.

Numerics vs the stock native path: K/V are the same stored E2M1 values and scales; the kernel rounds the dequantized
product to E4M3 (the FP8 MMA operand), q to E4M3 (as stock) and the probabilities to E4M3 (as the FP8 recipe), and writes
the attention output as BF16 (the stock E2M1 generation kernels write FP8).

Sub-part ``verify_wide`` (``MUSE_VR_NVFP4_KV_VERIFY_WIDE=1``, default off): full-attention layers (no sliding window)
with 2 or 4 KV heads per rank and between ``MUSE_VR_NVFP4_KV_VERIFY_WIDE_MIN_REQS`` (default: per KV-head count, see
``_WIDE_MIN``) and 256 requests run ``ops/kv4_attn_wide`` instead: the persistent, length-balanced kernel of
``ops/full_attn_wide`` with its K/V tiles dequantized in the kernel from the NVFP4 pools. Its operands are the same
(q rounded to E4M3 by the same conversion as the FP8-KV recipe's, K/V = one E4M3 rounding of the exact E2M1 x scale
product); it differs from the kernel above in the per-tile maxima (64-key tiles at 4 KV heads), the key split and the
merge order, as ``full_attn_wide`` does from the stock FP8 kernel. Every other call keeps the kernel above.
"""

from __future__ import annotations

import logging
import os

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/layers/attention/trtllm_mha_backend.py": ("_trtllm_native_nvfp4_kv_buffer",),
    "srt/runtime_context.py": ("get_parallel",),
    "srt/layers/cp/utils.py": ("is_cp_active",),
    "srt/layers/radix_attention.py": ("AttentionType",),
    "srt/mem_cache/memory_pool.py": ("KVWriteLoc",),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


logger = logging.getLogger("muse_vr_plugin")

TARGET_FWD = (
    "sglang.srt.layers.attention.trtllm_mha_backend.TRTLLMHAAttnBackend.forward_extend"
)
TARGET_INIT = (
    "sglang.srt.layers.attention.trtllm_mha_backend.TRTLLMHAAttnBackend.__init__"
)
_state: dict = {"preloaded": set()}


def enabled() -> bool:
    return os.environ.get(
        "MUSE_VR_NVFP4_KV_VERIFY_KERNEL", "1"
    ).strip().lower() not in ("0", "false", "no", "off")


# verify_wide: fewest requests per step that take ops/kv4_attn_wide, per KV heads per rank (below: the kernel above).
# From a pre-registered table (bench/bench_kv4_attn.py, --impl v3,kv4_faw, 16K-160K contexts): the smallest request
# count from which kv4_attn_wide measured <= 0.97x the kernel above at every context, and for every larger count.
# MUSE_VR_NVFP4_KV_VERIFY_WIDE_MIN_REQS overrides it for every head count.
_WIDE_MIN = {4: 16, 2: 32}
_WIDE_MAX = 256


def wide_enabled() -> bool:
    return os.environ.get("MUSE_VR_NVFP4_KV_VERIFY_WIDE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _wide_min(num_kv_heads: int) -> int:
    raw = os.environ.get("MUSE_VR_NVFP4_KV_VERIFY_WIDE_MIN_REQS", "").strip()
    if raw.isdigit():
        return max(1, int(raw))
    return _WIDE_MIN.get(num_kv_heads, _WIDE_MAX + 1)


def _use_pdl() -> bool:
    """Programmatic dependent launch for the kernel and its merge (the kernel reads q, the pool rows, the page table
    and the lengths only after griddepcontrol.wait). MUSE_VR_NVFP4_VERIFY_ATTN_PDL=0 turns it off."""
    return os.environ.get("MUSE_VR_NVFP4_VERIFY_ATTN_PDL", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _shape_ok(backend, layer) -> bool:
    from .ops import nvfp4_verify_attn as op

    return op.supported(
        layer.tp_q_head_num, layer.tp_k_head_num, layer.head_dim, backend.page_size
    ) and (layer.v_head_dim == layer.head_dim)


def _init_after(original, self, model_runner, *args, **kwargs):
    original(self, model_runner, *args, **kwargs)
    try:
        if not (self.uses_trtllm_gen_native_fp4 and self.decode_uses_native_fp4):
            return
        cfg = model_runner.model_config
        from sglang.srt.runtime_context import get_parallel

        tp = get_parallel().attn_tp_size
        hq, hkv = cfg.num_attention_heads // tp, cfg.get_num_kv_heads(tp)
        from .ops import nvfp4_verify_attn as op

        if not op.supported(hq, hkv, cfg.head_dim, self.page_size):
            bed.report(
                "nvfp4_kv_verify",
                False,
                reason=f"shape hq={hq} hkv={hkv} hd={cfg.head_dim} page={self.page_size}",
            )
            return
        import torch

        dev = torch.device(self.device)
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        pdl = _use_pdl()
        key = (str(dev), hq, hkv, pdl)
        if key not in _state["preloaded"]:
            # build/load the kernels and size the workspace before any CUDA-graph capture
            op.preload(
                int(model_runner.max_running_requests) + 1, hq, hkv, dev, use_pdl=pdl
            )
            _state["preloaded"].add(key)
        self._muse_vr_nvfp4_verify = True
        self._muse_vr_nvfp4_verify_pdl = pdl
        bed.report(
            "nvfp4_kv_verify",
            True,
            hq=hq,
            hkv=hkv,
            max_requests=int(model_runner.max_running_requests) + 1,
            pdl=pdl,
        )
        self._muse_vr_nvfp4_wide = False
        if wide_enabled():
            from .ops import kv4_attn_wide as kw

            if kw.supported(hq, hkv, cfg.head_dim, self.page_size):
                wkey = ("wide", str(dev), hq, hkv, pdl)
                if wkey not in _state["preloaded"]:
                    kw.preload(
                        hq, hkv, dev, use_pdl=pdl
                    )  # build, load and size before any CUDA-graph capture
                    _state["preloaded"].add(wkey)
                self._muse_vr_nvfp4_wide = True
            bed.report(
                "nvfp4_kv_verify_wide",
                self._muse_vr_nvfp4_wide,
                hq=hq,
                hkv=hkv,
                min_requests=_wide_min(hkv),
                max_requests=_WIDE_MAX,
                pdl=pdl,
                skel=kw._skel_for(hkv),
                dq_warps=kw.DQ_WARPS,
            )
    except Exception as e:  # noqa: BLE001 - the stock path stays in place
        self._muse_vr_nvfp4_verify = False
        bed.report("nvfp4_kv_verify", False, reason=f"{type(e).__name__}: {e}")


def _forward_extend_around(
    original, self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
):
    if not getattr(self, "_muse_vr_nvfp4_verify", False):
        return original(self, q, k, v, layer, forward_batch, save_kv_cache, **kwargs)
    import torch
    from sglang.srt.layers.cp.utils import is_cp_active
    from sglang.srt.layers.radix_attention import AttentionType
    from sglang.srt.mem_cache.memory_pool import KVWriteLoc

    md = self.forward_metadata
    if (
        not forward_batch.forward_mode.is_target_verify()
        or md is None
        or md.is_ragged_verify
        or md.max_seq_len_q != 8
        or layer.attn_type == AttentionType.ENCODER_ONLY
        or kwargs.get("sinks") is not None
        or q.dtype != torch.bfloat16
        or not self._forward_extend_uses_native_fp4(forward_batch)
        or is_cp_active(forward_batch)
        or not _shape_ok(self, layer)
        or _layer_is_fp8(self, layer)
    ):
        bed.count("nvfp4_kv_verify", "stock")
        return original(self, q, k, v, layer, forward_batch, save_kv_cache, **kwargs)
    import sglang.srt.layers.attention.trtllm_mha_backend as tb

    from .ops.nvfp4_verify_attn import nvfp4_verify_attn

    if save_kv_cache and k is not None:
        self.token_to_kv_pool.set_kv_buffer(
            layer,
            KVWriteLoc.for_batch(forward_batch, swa_loc=md.swa_out_cache_loc),
            k,
            v,
            *self._kv_write_scales(layer),
        )
    k_fp4, v_fp4, k_sf, v_sf = tb._trtllm_native_nvfp4_kv_buffer(
        self.token_to_kv_pool, layer.layer_id
    )
    hq = layer.tp_q_head_num
    qv = (
        q.reshape(-1, hq, layer.head_dim)
        if not q.is_contiguous()
        else q.view(-1, hq, layer.head_dim)
    )
    k_scale, v_scale = self._get_nvfp4_bmm_scales(layer)
    out = forward_batch._attn_output
    if out is not None and out.numel() == qv.numel() and out.is_contiguous():
        out = out.view(qv.shape)
    else:
        out = None
    reqs = qv.shape[0] // 8
    if (
        getattr(self, "_muse_vr_nvfp4_wide", False)
        and (layer.sliding_window_size is None or layer.sliding_window_size < 0)
        and _wide_min(layer.tp_k_head_num) <= reqs <= _WIDE_MAX
    ):
        from .ops.kv4_attn_wide import kv4_attn_wide

        q8 = qv.to(
            torch.float8_e4m3fn
        )  # the FP8-KV recipe's q conversion (forward_extend); contiguous
        o = kv4_attn_wide(
            q8,
            k_fp4,
            v_fp4,
            k_sf,
            v_sf,
            self._get_layer_page_table(layer, forward_batch),
            md.cache_seqlens_int32[:reqs],
            k_scale * layer.scaling,
            v_scale,
            out=out,
            use_pdl=getattr(self, "_muse_vr_nvfp4_verify_pdl", False),
        )
        bed.count("nvfp4_kv_verify_wide", f"reqs{reqs}")
        return o.view(-1, hq * layer.head_dim)
    o = nvfp4_verify_attn(
        qv,
        k_fp4,
        v_fp4,
        k_sf,
        v_sf,
        self._get_layer_page_table(layer, forward_batch),
        md.cache_seqlens_int32,
        layer.sliding_window_size,
        k_scale * layer.scaling,
        v_scale,
        out=out,
        use_pdl=getattr(self, "_muse_vr_nvfp4_verify_pdl", False),
    )
    return o.view(-1, hq * layer.head_dim)


def _layer_is_fp8(backend, layer) -> bool:
    """A layer of the FP8 sub-pool of a mixed pool (part ``swa_fp8``): not NVFP4, the stock FP8 path serves it."""
    from . import nvfp4_kv_mixed

    return nvfp4_kv_mixed.requested() and nvfp4_kv_mixed.layer_is_fp8(backend, layer)


def install() -> bool:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(TARGET_INIT, _init_after, HookType.AROUND)
    HookRegistry.register(TARGET_FWD, _forward_extend_around, HookType.AROUND)
    return True
