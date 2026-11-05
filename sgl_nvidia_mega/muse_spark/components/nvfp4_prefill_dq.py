# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Part ``prefill_dq`` of the ``nvfp4_kv`` component (``MUSE_VR_NVFP4_PREFILL_DQ=1``, default off): prefill (context)
attention over the native NVFP4 KV pool runs the trtllm-gen FP8-KV context kernel over an FP8 workspace.

The E2M1-KV context kernel converts each K/V tile once for every query tile that reads it, so a long cached prefix is
converted many times per layer. Here each layer's pages of the batch are converted once
(``ops/nvfp4_prefill_dq``: one kernel, memory-bound) into a workspace shared by all layers, and the FlashInfer context
call is made with that workspace as an FP8 KV cache.

Engaged per call (else the stock call runs unchanged) when: the backend serves prefill over native NVFP4 KV, the batch
is a plain EXTEND (not target verify or draft extend, no context parallelism, not inside CUDA-graph capture), the layer
is a full-attention decoder layer of the NVFP4 pool (no sliding window, no sinks; with ``swa_fp8`` the FP8 layers keep
the stock FP8 path), the FlashInfer call is the trtllm-gen paged HND form without LSE / FP4 output / skip-softmax /
special cubin variants and carries only the keywords in ``KNOWN_KEYS``, and the batch's pages fit the workspace (``MUSE_VR_NVFP4_PREFILL_DQ_PAGES`` pages, default
4096). The batch's page counts come from the extend lengths SGLang already holds on the host (no device sync).

Numerics: the workspace holds e4m3(e2m1 x block scale) (the exact product rounded once, round to nearest even,
satfinite: SGLang's FP8 dequant-workspace semantics for NVFP4 KV and the plugin verify kernel's dequant); the global
K/V scales stay in bmm1 / bmm2 as on the stock path; q is the same E4M3 q; the output dtype is whatever the call asks
for (BF16 with ``prefill_bf16_out``). The FP8-KV kernel over this workspace differs from the E2M1-KV kernel in a small
fraction of output elements, by one bf16 ulp (kernel-internal arithmetic: the differences remain when the dequantized
values are exactly representable); against an fp64 reference both have the same error.

In-place output (``MUSE_VR_NVFP4_PREFILL_DQ_INPLACE=1``, default off): inside SGLang's eager attention region of the
prefill graphs the layer's output rows are ``forward_batch._attn_output``; the region copies the backend's result into
them unless the result already lives there (it compares data pointers). With this knob the workspace call writes its BF16
output directly into those rows instead of the fresh tensor the BF16-output wrapper allocates, so the result is that
buffer and the copy is skipped. Same kernels, same inputs, same output bytes; only the output address changes. A call
whose destination does not fit (no buffer, another dtype, size or device, a non-contiguous buffer, a non-BF16 output)
keeps the fresh tensor. Counted as ``inplace`` / ``inplace:skip:<reason>``.
"""

from __future__ import annotations

import contextvars
import logging
import os

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/runtime_context.py": ("get_parallel",),
    "srt/layers/cp/utils.py": ("is_cp_active",),
    "srt/layers/radix_attention.py": ("AttentionType",),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


logger = logging.getLogger("muse_vr_plugin")

NAME = "nvfp4_prefill_dq"
TARGET_FWD = (
    "sglang.srt.layers.attention.trtllm_mha_backend.TRTLLMHAAttnBackend.forward_extend"
)
TARGET_INIT = (
    "sglang.srt.layers.attention.trtllm_mha_backend.TRTLLMHAAttnBackend.__init__"
)

_ACTIVE: contextvars.ContextVar = contextvars.ContextVar(
    "muse_vr_prefill_dq", default=None
)
# In-place mode only: (destination or None,) for the engaged forward_extend call; None when the mode is off.
_DEST: contextvars.ContextVar = contextvars.ContextVar(
    "muse_vr_prefill_dq_dest", default=None
)


def requested() -> bool:
    return os.environ.get("MUSE_VR_NVFP4_PREFILL_DQ", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def inplace_requested() -> bool:
    return os.environ.get("MUSE_VR_NVFP4_PREFILL_DQ_INPLACE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _workspace_pages() -> int:
    return max(1, int(os.environ.get("MUSE_VR_NVFP4_PREFILL_DQ_PAGES", "4096")))


def _use_pdl() -> bool:
    """Programmatic dependent launch for the dequant kernel (it reads nothing before griddepcontrol.wait).
    MUSE_VR_NVFP4_PREFILL_DQ_PDL=0 turns it off."""
    return os.environ.get("MUSE_VR_NVFP4_PREFILL_DQ_PDL", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _init_after(original, self, model_runner, *args, **kwargs):
    original(self, model_runner, *args, **kwargs)
    self._muse_vr_prefill_dq = None
    try:
        if not (
            getattr(self, "uses_trtllm_gen_native_fp4", False)
            and getattr(self, "prefill_uses_native_fp4", False)
        ):
            return
        if getattr(self, "skip_prefill", False) or getattr(self, "use_fmha_v2", False):
            return
        import torch
        from sglang.srt.runtime_context import get_parallel

        from .ops import nvfp4_prefill_dq as op

        cfg = model_runner.model_config
        hkv = cfg.get_num_kv_heads(get_parallel().attn_tp_size)
        if (
            cfg.head_dim != op.HEAD_DIM
            or self.page_size % 4 != 0
            or self.page_size * hkv * 2 > 1024
        ):
            bed.report(
                NAME,
                False,
                reason=f"shape hkv={hkv} hd={cfg.head_dim} page={self.page_size}",
            )
            return
        dev = torch.device(self.device)
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        max_req = int(model_runner.max_running_requests) + 1
        ws = op.Workspace(
            _workspace_pages(),
            hkv,
            self.page_size,
            max_req,
            int(self.max_num_pages),
            dev,
        )
        op._module(_use_pdl())  # build / load before serving
        self._muse_vr_prefill_dq = ws
        bed.report(
            NAME,
            True,
            pages=ws.pages,
            kv_heads=hkv,
            page_size=self.page_size,
            max_requests=max_req,
            workspace_bytes=ws.nbytes(),
            pdl=_use_pdl(),
        )
    except Exception as e:  # noqa: BLE001 - the stock path stays in place
        self._muse_vr_prefill_dq = None
        bed.report(NAME, False, reason=f"{type(e).__name__}: {e}")


def _plan(forward_batch, page_size: int):
    """(total pages, max pages per request) of an extend batch from its host-side lengths, cached on the batch."""
    cached = getattr(forward_batch, "_muse_vr_prefill_dq_plan", None)
    if cached is not None:
        return cached
    pre, ext = forward_batch.extend_prefix_lens_cpu, forward_batch.extend_seq_lens_cpu
    if (
        pre is None
        or ext is None
        or len(pre) != len(ext)
        or len(ext) != forward_batch.batch_size
    ):
        plan = (None, None)
    else:
        from .ops.nvfp4_prefill_dq import pages_of

        plan = pages_of((int(a) + int(b) for a, b in zip(pre, ext)), page_size)
    try:
        forward_batch._muse_vr_prefill_dq_plan = plan
    except Exception:  # noqa: BLE001 - a frozen batch object: recompute next time
        pass
    return plan


def _layer_is_fp8(backend, layer) -> bool:
    from . import nvfp4_kv_mixed

    return nvfp4_kv_mixed.requested() and nvfp4_kv_mixed.layer_is_fp8(backend, layer)


def _forward_extend_around(
    original, self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
):
    ws = getattr(self, "_muse_vr_prefill_dq", None)
    if ws is None:
        return original(self, q, k, v, layer, forward_batch, save_kv_cache, **kwargs)
    import torch
    from sglang.srt.layers.cp.utils import is_cp_active
    from sglang.srt.layers.radix_attention import AttentionType

    fm = forward_batch.forward_mode
    if (
        not fm.is_extend()
        or fm.is_target_verify()
        or fm.is_draft_extend_v2()
        or is_cp_active(forward_batch)
        or layer.attn_type == AttentionType.ENCODER_ONLY
        or layer.sliding_window_size is None
        or layer.sliding_window_size >= 0
        or kwargs.get("sinks") is not None
        or not self._forward_extend_uses_native_fp4(forward_batch)
        or _layer_is_fp8(self, layer)
        or torch.cuda.is_current_stream_capturing()
    ):
        return original(self, q, k, v, layer, forward_batch, save_kv_cache, **kwargs)
    total, max_pages = _plan(forward_batch, self.page_size)
    if total is None or not ws.fits(total, forward_batch.batch_size, max_pages):
        bed.count(NAME, "stock_capacity" if total is not None else "stock_no_host_lens")
        return original(self, q, k, v, layer, forward_batch, save_kv_cache, **kwargs)
    token = _ACTIVE.set((ws, total, max_pages, forward_batch.batch_size))
    dest = (
        _DEST.set((getattr(forward_batch, "_attn_output", None),))
        if inplace_requested()
        else None
    )
    try:
        return original(self, q, k, v, layer, forward_batch, save_kv_cache, **kwargs)
    finally:
        if dest is not None:
            _DEST.reset(dest)
        _ACTIVE.reset(token)


# Keyword arguments whose meaning the workspace call preserves: SGLang's served context call plus the ones _call_ok checks
# by value. A call carrying any other keyword keeps the stock kernel and is counted as "stock_call:key:<names>"; each new
# key set of an engaged call is recorded once as "keys:<names>".
KNOWN_KEYS = frozenset(
    {
        "query",
        "kv_cache",
        "workspace_buffer",
        "block_tables",
        "seq_lens",
        "max_q_len",
        "max_kv_len",
        "bmm1_scale",
        "bmm2_scale",
        "batch_size",
        "cum_seq_lens_q",
        "cum_seq_lens_kv",
        "window_left",
        "sinks",
        "skip_softmax_threshold_scale_factor",
        "out",
        "out_dtype",
        "kv_cache_sf",
        "causal",
        "enable_pdl",
        "kv_layout",
        "uses_shared_paged_kv_idx",
        "lse",
        "return_lse",
        "o_sf_scale",
        "use_fp16_softmax",
        "uses_spcompress",
        "backend",
    }
)
_seen_key_sets: set = set()


def _call_ok(kwargs: dict, batch: int) -> bool:
    kv, sf = kwargs.get("kv_cache"), kwargs.get("kv_cache_sf")
    if not (
        isinstance(kv, tuple)
        and len(kv) == 2
        and isinstance(sf, tuple)
        and len(sf) == 2
    ):
        return False
    if kv[0].dim() != 4 or kv[1].dim() != 4 or sf[0].dim() != 4 or sf[1].dim() != 4:
        return False
    window = kwargs.get("window_left", -1)
    return (
        kwargs.get("batch_size") == batch
        and kwargs.get("seq_lens") is not None
        and kwargs["seq_lens"].shape[0] == batch
        and (window is None or window < 0)
        and kwargs.get("sinks") is None
        and kwargs.get("kv_layout", "HND") == "HND"
        and kwargs.get("uses_shared_paged_kv_idx", True)
        and kwargs.get("lse") is None
        and not kwargs.get("return_lse", False)
        and kwargs.get("o_sf_scale") is None
        and kwargs.get("out_dtype") != "nvfp4"
        and not kwargs.get("skip_softmax_threshold_scale_factor")
        and kwargs.get("use_fp16_softmax") is None
        and kwargs.get("uses_spcompress") is None
        and kwargs.get("backend", "trtllm-gen") == "trtllm-gen"
    )


def _to_dest(kwargs: dict, dest) -> None:
    """In-place mode: point the workspace call's BF16 output at ``dest`` (the layer's output rows, see the module
    docstring), else leave the call as it is (counted with the reason).

    Inside an engaged ``forward_extend`` call the backend's native-NVFP4 context call passes its E4M3 staging buffer as
    ``out``; a BF16 ``out`` with the query's shape here is the tensor the BF16-output wrapper allocated for that call. The
    context functions below return ``out``, so the call's result becomes a view of ``dest``."""
    import torch

    query, out = kwargs.get("query"), kwargs.get("out")
    if dest is None:
        why = "no_dest"
    elif not (
        isinstance(query, torch.Tensor)
        and isinstance(out, torch.Tensor)
        and out.dtype == torch.bfloat16
        and out.shape == query.shape
        and out.is_contiguous()
    ):
        why = "out"
    elif not (
        isinstance(dest, torch.Tensor)
        and dest.is_cuda
        and dest.dtype == torch.bfloat16
        and dest.is_contiguous()
        and dest.device == query.device
    ):
        why = "dest"
    elif dest.numel() != query.numel():
        why = "numel"
    else:
        kwargs["out"] = dest.view(out.shape)
        bed.count(NAME, "inplace")
        return
    bed.count(NAME, "inplace:skip:" + why)


def wrap_context(context_fn):
    """Wrap FlashInfer's ``trtllm_batch_context_with_kv_cache``: inside an engaged ``forward_extend`` call (see the
    module docstring), a keyword call over the NVFP4 pool goes through the FP8 workspace; every other call is unchanged."""

    def ctx(*args, **kwargs):
        plan = _ACTIVE.get()
        if plan is None or args or kwargs.get("kv_cache_sf") is None:
            return context_fn(*args, **kwargs)
        ws, total, max_pages, batch = plan
        unknown = sorted(set(kwargs) - KNOWN_KEYS)
        if unknown:
            bed.count(NAME, "stock_call:key:" + ",".join(unknown))
            return context_fn(*args, **kwargs)
        keys = frozenset(kwargs)
        if keys not in _seen_key_sets:
            _seen_key_sets.add(keys)
            bed.count(NAME, "keys:" + ",".join(sorted(keys)))
        if not _call_ok(kwargs, batch):
            bed.count(NAME, "stock_call")
            return context_fn(*args, **kwargs)
        from .ops import nvfp4_prefill_dq as op

        bed.count(NAME, "dq")
        box = _DEST.get()
        if box is not None:
            _to_dest(kwargs, box[0])
        return op.context_over_workspace(
            context_fn, ws, total, max_pages, kwargs, use_pdl=_use_pdl()
        )

    ctx.__muse_vr_derived_from__ = getattr(
        context_fn, "__muse_vr_derived_from__", context_fn
    )
    return ctx


def install() -> bool:
    """Register the backend hooks (the context-function wrapper is installed by the ``fi_gate`` part)."""
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(TARGET_INIT, _init_after, HookType.AROUND)
    HookRegistry.register(TARGET_FWD, _forward_extend_around, HookType.AROUND)
    return True
