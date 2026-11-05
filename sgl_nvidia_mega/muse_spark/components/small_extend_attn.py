# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``small_extend_attn``: run the attention of small prefill extends with the split-KV decode kernel.

SGLang's TRT-LLM MHA backend computes the attention of every EXTEND forward with FlashInfer's trtllm-gen context
kernel (``trtllm_batch_context_with_kv_cache``). That kernel parallelises over query tiles, so an extend with a few
hundred new tokens over a long cached prefix (a follow-up turn on a long agentic context) keeps only a handful of
CTAs busy streaming the whole prefix: its time is nearly that of a full query tile, independent of how few tokens
there are. The same backend already runs multi-token queries over long caches with the multi-CTA split-KV decode
kernel (``trtllm_batch_decode_with_kv_cache`` with ragged ``cum_seq_lens_q``; the speculative verify path), which
splits the prefix over CTAs and merges the partial softmax results. It is causal within each request's new tokens and
honours the sliding window, so it computes the same attention for an extend.

This component routes the attention call of an EXTEND forward to that kernel when its longest per-request extend is
at most ``MUSE_VR_SMALL_EXTEND_ATTN_MAX_Q`` tokens (default 768; above that the context kernel is as fast). Everything
else in the backend's extend path (the K/V store, query conversion, cache layout, scales, page table, output buffer)
is unchanged. Numerics: a different reduction order (split-KV partials merged in a fixed order); measured error vs an
fp64 reference is at or below the context kernel's for every shape tested (``tests/test_small_extend_attn.py``).

Opt in with ``MUSE_VR_SMALL_EXTEND_ATTN=1``. Engagement is guarded by sha256 pins on the backend and FlashInfer files.
"""

from __future__ import annotations

import contextvars
import functools
import hashlib
import importlib.util
import logging
import os
from pathlib import Path

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/layers/attention/trtllm_mha_backend.py": ("TRTLLMHAAttnBackend",),
    "srt/layers/cp/utils.py": ("is_cp_active",),
    "srt/layers/radix_attention.py": ("AttentionType",),
    "srt/model_executor/forward_batch_info.py": ("ForwardMode",),
}


logger = logging.getLogger("muse_vr_plugin")

NAME = "small_extend_attn"
DEFAULT_MAX_Q = 768

SGLANG_PINS = {
    "sglang/srt/layers/attention/trtllm_mha_backend.py": (
        "566d1031d644f71a8adb84d1e2a1581d3a6d1c3cac1e507451ac4cf034c39da4",
        "07223cbf82ab5ac14fcbaa9a47a8351132991016bae66c122bf09762ed8ed646",
    ),
}
FLASHINFER_PINS = {
    "flashinfer/prefill.py": (
        "0f1f9e03a7aa59ab9b95f270b85b240041717f3e08ee20c30d1332d5136d19a0",  # FlashInfer 0.7.1
        "531e220624ebc543fe865b1f6f078c5646233abc287b8407ac955e26d44f73b8",  # FlashInfer 0.7.2
    ),
    "flashinfer/decode.py": (
        "8917d7f1a80707d823455d69d105d90a57e8bb8a71b248348f02d62c547a6f91",  # FlashInfer 0.7.1
        "de1f242bf2f598bc973a9d2f9625f8db1e4ccfc24085f686b2d0304c9076b20d",  # FlashInfer 0.7.2
    ),
}

_state: dict = {
    "engaged": None,
    "max_q": DEFAULT_MAX_Q,
    "orig_ctx": None,
    "orig_fwd": None,
}
# set by the forward_extend wrapper for an eligible call: {"mcb": the backend's multi-CTA counter buffer}
_route: contextvars.ContextVar = contextvars.ContextVar(
    "muse_vr_small_extend_route", default=None
)

# Context-kernel keywords the redirect understands; any other keyword (LSE output, FP4 output scales, sage/sparse
# variants, ...) keeps the stock context kernel.
_HANDLED = {
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
    "out",
    "out_dtype",
    "sinks",
    "skip_softmax_threshold_scale_factor",
    "kv_cache_sf",
    "causal",
}


def _pins_ok() -> tuple[bool, str]:
    for rel, sha in SGLANG_PINS.items():
        if not bed.pinned(rel, sha):
            return False, rel
    spec = importlib.util.find_spec("flashinfer")
    if spec is None or not spec.submodule_search_locations:
        return False, "flashinfer"
    root = Path(list(spec.submodule_search_locations)[0]).resolve().parent
    for rel, sha in FLASHINFER_PINS.items():
        try:
            got = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        except OSError:
            got = None
        if not bed.sha_ok(got, sha):
            logger.warning(
                "muse_vr_plugin: %s sha256 %s does not match the pin %s",
                rel,
                (got or "<missing>")[:16],
                bed.sha_label(sha),
            )
            return False, rel
    return True, ""


def _context_attn(*args, **kwargs):
    """Stand-in for ``flashinfer.prefill.trtllm_batch_context_with_kv_cache``: diverts eligible calls only."""
    route = _route.get()
    if (
        route is None
        or args
        or not set(kwargs) <= _HANDLED
        or not kwargs.get("causal", True)
        or kwargs.get("kv_cache_sf") is not None
        or kwargs.get("skip_softmax_threshold_scale_factor") is not None
    ):
        return _state["orig_ctx"](*args, **kwargs)
    import flashinfer

    bed.count(NAME, "decode_kernel")
    return flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        query=kwargs["query"],
        kv_cache=kwargs["kv_cache"],
        workspace_buffer=kwargs["workspace_buffer"],
        block_tables=kwargs["block_tables"],
        seq_lens=kwargs["seq_lens"],
        max_seq_len=kwargs["max_kv_len"],
        bmm1_scale=kwargs["bmm1_scale"],
        bmm2_scale=kwargs["bmm2_scale"],
        window_left=kwargs.get("window_left", -1),
        sinks=kwargs.get("sinks"),
        out=kwargs.get("out"),
        out_dtype=kwargs.get("out_dtype"),
        q_len_per_req=None,
        max_q_len=kwargs["max_q_len"],
        cum_seq_lens_q=kwargs["cum_seq_lens_q"],
        multi_ctas_kv_counter_buffer=route["mcb"],
    )


def eligible(backend, layer, forward_batch) -> bool:
    from sglang.srt.layers.cp.utils import is_cp_active
    from sglang.srt.layers.radix_attention import AttentionType
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    md = getattr(backend, "forward_metadata", None)
    mcb = getattr(backend, "_multi_ctas_kv_counter_buffer", None)
    return bool(
        forward_batch.forward_mode == ForwardMode.EXTEND
        and md is not None
        and mcb is not None
        and isinstance(getattr(md, "max_seq_len_q", None), int)
        and 0 < md.max_seq_len_q <= _state["max_q"]
        and getattr(md, "cu_seqlens_q", None) is not None
        and layer.attn_type == AttentionType.DECODER
        and not getattr(backend, "use_fmha_v2", False)
        and not is_cp_active(forward_batch)
        and not backend._forward_extend_uses_native_fp4(forward_batch)
    )


def _wrap_forward_extend(fn):
    @functools.wraps(fn)
    def forward_extend(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
    ):
        if not eligible(self, layer, forward_batch):
            return fn(self, q, k, v, layer, forward_batch, save_kv_cache, **kwargs)
        token = _route.set({"mcb": self._multi_ctas_kv_counter_buffer})
        try:
            return fn(self, q, k, v, layer, forward_batch, save_kv_cache, **kwargs)
        finally:
            _route.reset(token)

    forward_extend.__wrapped_by_small_extend__ = True
    return forward_extend


def install(max_q: int) -> None:
    import flashinfer.prefill
    import sglang.srt.layers.attention.trtllm_mha_backend as tb

    _state["max_q"] = int(max_q)
    if _state["orig_ctx"] is None:
        _state["orig_ctx"] = flashinfer.prefill.trtllm_batch_context_with_kv_cache
        flashinfer.prefill.trtllm_batch_context_with_kv_cache = _context_attn
    cls = tb.TRTLLMHAAttnBackend
    if not getattr(cls.forward_extend, "__wrapped_by_small_extend__", False):
        cls.forward_extend = _wrap_forward_extend(cls.forward_extend)


def engage() -> bool:
    if _state["engaged"] is not None:
        return _state["engaged"]
    if not bed.enabled(NAME):
        _state["engaged"] = False
        return False
    ok, why = _pins_ok()
    if not ok:
        bed.report(NAME, False, reason=f"pin mismatch: {why}")
        _state["engaged"] = False
        return False
    try:
        max_q = int(
            os.environ.get("MUSE_VR_SMALL_EXTEND_ATTN_MAX_Q", str(DEFAULT_MAX_Q))
        )
        if max_q <= 0:
            raise ValueError("must be positive")
    except ValueError as e:
        bed.report(NAME, False, reason=f"bad MUSE_VR_SMALL_EXTEND_ATTN_MAX_Q ({e})")
        _state["engaged"] = False
        return False
    try:
        install(max_q)
    except Exception as e:  # noqa: BLE001
        bed.report(NAME, False, reason=f"install failed: {type(e).__name__}: {e}")
        _state["engaged"] = False
        return False
    bed.report(NAME, True, max_q=max_q)
    _state["engaged"] = True
    return True
