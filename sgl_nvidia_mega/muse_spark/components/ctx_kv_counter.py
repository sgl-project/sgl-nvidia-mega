# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``ctx_kv_counter`` (env ``MUSE_VR_CTX_KV_COUNTER=1``, default off): one persistent multi-CTA KV counter buffer per
device for FlashInfer's trtllm-gen context attention calls that pass none.

``flashinfer.prefill.trtllm_batch_context_with_kv_cache`` takes an optional ``multi_ctas_kv_counter_buffer``. When the caller
passes none (SGLang's context call does not), FlashInfer allocates and zero-fills a fresh buffer on every call: one allocation
and one memset per attention layer of every prefill pass. FlashInfer documents that a caller-held buffer is zero-filled once,
at allocation, and needs no re-zeroing between calls. This component wraps the function and passes such a buffer: allocated
outside CUDA-graph capture, zero-filled at allocation, grown (a new zero-filled buffer) when a call needs more bytes than it
holds. A call that already carries a counter buffer, passes positional arguments, or arrives during capture with no
large-enough buffer yet runs unchanged.

The wrapper must sit directly around FlashInfer's function, below the plugin's other wrappers of this function: ``nvfp4_kv``
derives its context-function chain from the installed function at its registration, so this component registers first and
patches the module attribute before that (the chain then wraps this wrapper), and the wrappers above it see the caller's own
keyword set.

Numerics: the attention output does not depend on where the counters live. Decisions are counted under ``ctx_kv_counter``
in ``bed`` ("persistent", "caller_buffer", "stock:capturing").
"""

from __future__ import annotations

import functools
import inspect
import logging
import threading

from . import bed

logger = logging.getLogger("muse_vr_plugin")

NAME = "ctx_kv_counter"
FLASHINFER_VERSIONS = ("0.7.2",)
KW = "multi_ctas_kv_counter_buffer"

_state: dict = {"engaged": None, "bufs": {}, "keep": [], "sig": None, "need": {}}
_lock = threading.Lock()


def _buffer(q, batch):
    """The device's buffer when it can serve this call (allocating or growing it outside capture), else None. Every buffer
    ever handed out stays referenced: a captured graph may hold its address."""
    import torch
    from flashinfer import utils as fu

    key = (q.device, int(batch), int(q.size(1)))
    need = _state["need"].get(key)
    if need is None:
        need = int(
            fu.get_trtllm_gen_multi_ctas_kv_counter_bytes(
                key[1], key[2], fu.get_device_sm_count(q.device)
            )
        )
        _state["need"][key] = need
    buf = _state["bufs"].get(q.device)
    if buf is not None and buf.numel() >= need:
        return buf
    if torch.cuda.is_current_stream_capturing():
        return None
    buf = torch.zeros(
        max(need, 2 * (buf.numel() if buf is not None else 0), 4096),
        dtype=torch.uint8,
        device=q.device,
    )
    _state["bufs"][q.device] = buf
    _state["keep"].append(buf)
    return buf


def _wrap(fn):
    @functools.wraps(fn)
    def ctx(*args, **kwargs):
        if args:  # keyword calls only (SGLang's call is one); a positional call runs unchanged
            bed.count(NAME, "stock:positional")
            return fn(*args, **kwargs)
        if kwargs.get(KW) is not None:
            bed.count(NAME, "caller_buffer")
            return fn(**kwargs)
        q, batch = kwargs.get("query"), kwargs.get("batch_size")
        buf = _buffer(q, batch) if q is not None and batch is not None else None
        if buf is None:
            bed.count(
                NAME,
                "stock:capturing"
                if q is not None and batch is not None
                else "stock:args",
            )
            return fn(**kwargs)
        bed.count(NAME, "persistent")
        return fn(**dict(kwargs, **{KW: buf}))

    ctx.__wrapped_by_ctx_kv_counter__ = True
    return ctx


def register() -> bool:
    with _lock:
        if _state["engaged"] is not None:
            return _state["engaged"]
        _state["engaged"] = False
        if not bed.enabled(NAME):
            return False
        try:
            import flashinfer
            import flashinfer.prefill as fp
            from flashinfer import utils as fu

            ver = getattr(flashinfer, "__version__", "")
            if ver not in FLASHINFER_VERSIONS:
                bed.report(
                    NAME, False, reason=f"flashinfer {ver} not in {FLASHINFER_VERSIONS}"
                )
                return False
            fn = fp.trtllm_batch_context_with_kv_cache
            if getattr(fn, "__wrapped_by_ctx_kv_counter__", False):
                _state["engaged"] = True
                return True
            sig = inspect.signature(fn)
            missing = [
                p for p in ("query", "batch_size", KW) if p not in sig.parameters
            ]
            if (
                missing
                or not hasattr(fu, "get_trtllm_gen_multi_ctas_kv_counter_bytes")
                or not hasattr(fu, "get_device_sm_count")
            ):
                bed.report(
                    NAME, False, reason=f"signature / helpers missing: {missing}"
                )
                return False
            _state["sig"] = sig
            fp.trtllm_batch_context_with_kv_cache = _wrap(fn)
        except Exception as e:  # noqa: BLE001 - a failed opt-in component leaves FlashInfer's function as it was
            bed.report(NAME, False, reason=f"install failed: {type(e).__name__}: {e}")
            return False
        _state["engaged"] = True
        bed.report(NAME, True, flashinfer=ver)
        return True
