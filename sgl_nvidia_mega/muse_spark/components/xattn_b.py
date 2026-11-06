# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``xattn_b``: SGLang adapter of the own extend-attention op (``extend_attn_b``).

The only code that touches SGLang or FlashInfer for the op. An AROUND hook on the TRT-LLM MHA backend's
``forward_extend`` builds the op's ExtendCall from the batch's host-side lengths (plain EXTEND forwards of full-attention
decoder layers; FP8 KV; not under capture; no context parallelism); a stand-in for FlashInfer's context call then asks the
op to run the call. The op returns None for every call outside its window (counted, ``decline:<reason>``), and the
stand-in runs the previous (stock) function. The op's kernels are built and run once on every rank right after the
model is loaded (``extend_attn_b.prepare``), never inside a served forward.

Opt in with ``MUSE_VR_XATTN_B=1``. Refused at boot when ``extend_attn``, ``small_extend_attn`` or ``prefill_attn`` is
also enabled (they route the same calls). ``MUSE_VR_XATTN_B_PFA_OPT`` (default 0) selects the attention kernel's
option (extend_attn_b.launch: bits 1 / 2 / 8 synchronization only, bit 4 the tensor-core row sum); a value outside
launch's list refuses the component at boot, and a nonzero value is recorded as ``pfa_opt`` in the engagement record. Engagement is
guarded by the sha256 pins of the backend and FlashInfer files.

NVFP4 KV pool (``nvfp4_kv``): the op reads E4M3 pages only, so over an NVFP4 pool it serves the context calls that
``nvfp4_kv``'s ``prefill_dq`` part makes over its FP8 workspace, and nothing else. ``nvfp4_kv`` builds that chain itself
(``compose_context``): the stand-in sits innermost, directly around FlashInfer's function and inside ``prefill_dq``'s
workspace wrapper, so the calls it sees carry the workspace as an E4M3 cache; calls over the NVFP4 pages themselves
(``kv_cache_sf`` set, e.g. a batch that does not fit the workspace) are declined and run stock. In that form the
FlashInfer attribute is not patched here (``nvfp4_kv``'s replacement owns it), and the composition is refused, loudly,
unless the function it wraps is the pinned FlashInfer function object and the workspace calls carry a BF16 output
(``MUSE_VR_NVFP4_KV_PREFILL_BF16_OUT`` on). With ``nvfp4_kv`` enabled and no composition (``prefill_dq`` off), the
component is refused at boot. Per-request routing is refused in the composed form.

Load-time charge. ``prepare`` builds the op's kernels and grows its split scratch to the cap right after the target's
weights load, i.e. before SGLang sizes the KV pool, so the pool is smaller by that much. ``MUSE_VR_XATTN_B_DEFER_PREPARE=1``
(default 0) moves the same ``prepare`` call: the load hook only records the device (count ``prepare_deferred``) and a BEFORE
hook on ``ModelRunner.init_cuda_graphs`` (the target runner only; after the KV pool is sized, before any graph capture) runs
it. Until then the op declines every call as ``not_prepared`` (stock path; no allocation). A ``prepare`` that fails there is
loud and leaves the op off on that device (warning, count ``prepare_failed``, report ``xattn_b_defer`` engaged=False; the
device is left unprepared, so every call runs stock). ``MUSE_VR_XATTN_B_PREP_REQUESTS`` (1..4, default 4 = ``prepare``'s
default) trims the scratch cap (``prepare(max_requests=...)``); a call that needs more slots than the cap declines as
``scratch`` and runs stock. Other values of either knob refuse the component at boot.
"""

from __future__ import annotations

import contextvars
import functools
import logging

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/layers/attention/trtllm_mha_backend.py": ("TRTLLMHAAttnBackend",),
    "srt/model_executor/model_runner.py": ("ModelRunner",),
    "srt/layers/radix_attention.py": ("AttentionType",),
    "srt/model_executor/forward_batch_info.py": ("ForwardMode",),
    "srt/layers/cp/utils.py": ("is_cp_active",),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


logger = logging.getLogger("muse_vr_plugin")

NAME = "xattn_b"
CONFLICTS = ("extend_attn", "small_extend_attn", "prefill_attn")
PFA_OPTS = tuple(
    str(v) for v in range(16) if not (v & 1 and v & 8)
)  # = extend_attn_b.launch.OPTS
_state: dict = {
    "engaged": None,
    "prev_ctx": None,
    "route": False,
    "compose": False,
    "composed": False,
    "defer": False,
    "prep_requests": 4,
    "pending": {},
    "hook_ran": False,
}
T_GRAPHS = "sglang.srt.model_executor.model_runner.ModelRunner.init_cuda_graphs"
_call: contextvars.ContextVar = contextvars.ContextVar(
    "muse_vr_xattn_b_call", default=None
)


def _context_attn(*args, **kwargs):
    """Stand-in for ``flashinfer.prefill.trtllm_batch_context_with_kv_cache``: the op, else the previous function."""
    call = _call.get()
    if (
        call is None
        or args
        or kwargs.get("lse") is not None
        or kwargs.get("return_lse", False)
    ):
        return _state["prev_ctx"](*args, **kwargs)
    from . import extend_attn_b as xb

    try:
        out = xb.extend_attention(call, **kwargs)
        why = xb.last_decline() if out is None else None
    except Exception as e:  # noqa: BLE001 - the adapter keeps a catch-to-stock; the op itself fails loud
        bed.count(NAME, "error")
        logger.warning(
            "muse_vr_plugin: %s op error (%s: %s); stock call",
            NAME,
            type(e).__name__,
            e,
        )
        out, why = None, None
    if out is None:
        if why:
            bed.count(NAME, "decline:" + why)  # visible in the engagement record
            if len(call.new_lens) > 1:
                serve = _serve_mask(call, kwargs)
                lost = sum(serve)
                if lost and _state["route"] and lost < len(serve):
                    routed = _route(call, kwargs, serve)
                    if routed is not None:
                        bed.count(NAME, "routed_calls")
                        bed.count(NAME, "routed_requests", lost)
                        return routed
                if lost:  # requests the op would serve alone, run by stock because a batch-mate is out of window
                    bed.count(NAME, "decline_lost_requests", lost)
        return _state["prev_ctx"](*args, **kwargs)
    bed.count(NAME, "own")
    return out


_MASKS: dict = {}


def _serve_mask(call, kwargs) -> tuple:
    """Per request: would the op's plan admit it on its own (host only; cached per call shape, so the layers of one pass
    pay it once)."""
    from . import extend_attn_b as xb
    from .extend_attn_b import ExtendCall, plan

    q = kwargs.get("query")
    if q is None or q.dim() != 3:
        return tuple(False for _ in call.new_lens)
    key = (
        q.device.index,
        int(q.shape[1]),
        call.mode,
        call.capturing,
        call.new_lens,
        call.prefix_lens,
    )
    m = _MASKS.get(key)
    if m is None:
        sms = xb._num_sms(q.device.index)
        m = tuple(
            plan.decline_reason(
                ExtendCall((t,), (c,), call.layer_id, call.mode, call.capturing),
                int(q.shape[1]),
                sms,
            )
            is None
            for t, c in zip(call.new_lens, call.prefix_lens)
        )
        if len(_MASKS) >= 64:
            _MASKS.clear()
        _MASKS[key] = m
    return m


def _route(call, kwargs, serve):
    """Per-request routing (MUSE_VR_XATTN_B_ROUTE=1): the op runs the in-window requests on the full call tensors, and the
    stock function runs each maximal contiguous run of the other requests as a sub-call on row / slot views of the same
    tensors (rebased cumulative lengths computed on the device in stream order). Returns the output, or None (then the
    caller runs stock for the whole call; nothing has been written)."""
    import torch

    from . import extend_attn_b as xb

    q = kwargs["query"]
    out = kwargs.get("out")
    if out is None:
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    kw = dict(kwargs, out=out)
    if xb.extend_attention(call, serve=serve, **kw) is None:
        return None
    cu = [0]
    for t in call.new_lens:
        cu.append(cu[-1] + t)
    n, a = len(serve), 0
    while a < n:
        if serve[a]:
            a += 1
            continue
        b = a
        while b < n and not serve[b]:
            b += 1
        ra, rb = cu[a], cu[b]
        cq, ck = kwargs["cum_seq_lens_q"], kwargs["cum_seq_lens_kv"]
        sub = dict(kwargs)
        # rebased cumulative lengths computed on the device in stream order (no host copy, no host wait)
        sub.update(
            query=q[ra:rb],
            block_tables=kwargs["block_tables"][a:b],
            seq_lens=kwargs["seq_lens"][a:b],
            cum_seq_lens_q=cq[a : b + 1] - cq[a],
            cum_seq_lens_kv=ck[a : b + 1] - ck[a],
            batch_size=b - a,
            max_q_len=max(call.new_lens[a:b]),
            out=out[ra:rb],
        )
        _state["prev_ctx"](**sub)
        a = b
    return out


def _extend_call(backend, layer, forward_batch):
    """The op's ExtendCall for an eligible layer call, else None."""
    import torch
    from sglang.srt.layers.radix_attention import AttentionType
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    from .extend_attn_b import ExtendCall

    if (
        forward_batch.forward_mode != ForwardMode.EXTEND
        or layer.attn_type != AttentionType.DECODER
    ):
        return None
    w = getattr(layer, "sliding_window_size", None)
    if (w is not None and w >= 0) or getattr(backend, "use_fmha_v2", False):
        return None
    if getattr(backend, "data_type", None) != torch.float8_e4m3fn and not (
        _state["composed"] and getattr(backend, "_muse_vr_prefill_dq", None) is not None
    ):
        return None  # NVFP4 pool: only behind prefill_dq's FP8 workspace (the op declines the NVFP4 calls themselves)
    try:
        from sglang.srt.layers.cp.utils import is_cp_active

        if is_cp_active(forward_batch):
            return None
    except ImportError:
        pass
    pre = getattr(forward_batch, "extend_prefix_lens_cpu", None)
    ext = getattr(forward_batch, "extend_seq_lens_cpu", None)
    if pre is None or ext is None or len(pre) != len(ext) or not len(ext):
        return None
    return ExtendCall(
        tuple(int(x) for x in ext),
        tuple(int(x) for x in pre),
        int(getattr(layer, "layer_id", -1)),
        "extend",
        bool(torch.cuda.is_current_stream_capturing()),
    )


def _wrap_forward_extend(fn):
    @functools.wraps(fn)
    def forward_extend(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
    ):
        call = _extend_call(self, layer, forward_batch)
        if call is None:
            return fn(self, q, k, v, layer, forward_batch, save_kv_cache, **kwargs)
        token = _call.set(call)
        try:
            return fn(self, q, k, v, layer, forward_batch, save_kv_cache, **kwargs)
        finally:
            _call.reset(token)

    forward_extend.__wrapped_by_xattn_b__ = True
    return forward_extend


def _wrap_load_model(fn):
    @functools.wraps(fn)
    def load_model(self, *args, **kwargs):
        out = fn(self, *args, **kwargs)
        try:
            import torch

            from . import extend_attn_b as xb
            from .extend_attn_b import plan

            mc = self.model_config
            tp = int(getattr(self, "tp_size", 1) or 1)
            hq = mc.num_attention_heads // tp
            hkv = max(1, mc.get_num_kv_heads(tp))
            if hq == 8 * hkv:
                dix = torch.cuda.current_device()
                if (
                    _state["defer"] and not _state["hook_ran"]
                ):  # MUSE_VR_XATTN_B_DEFER_PREPARE=1: after the KV pool
                    _state["pending"][dix] = (hkv, hq)
                    bed.count(NAME, "prepare_deferred")
                else:
                    xb.prepare(
                        dix,
                        max_requests=_state["prep_requests"],
                        max_new_tokens=plan.T_HI,
                        num_kv_heads=hkv,
                        num_q_heads=hq,
                    )
                    bed.count(NAME, "prepared")
            else:
                bed.count(NAME, "prepare_skipped:head_layout")
        except Exception as e:  # noqa: BLE001 - without prepare the op declines every call ("not_prepared")
            bed.count(NAME, "prepare_failed")
            logger.warning(
                "muse_vr_plugin: %s prepare failed (%s: %s); the op stays off",
                NAME,
                type(e).__name__,
                e,
            )
        return out

    load_model.__wrapped_by_xattn_b__ = True
    return load_model


def _is_target_runner(runner) -> bool:
    """Whether a ModelRunner serves the target MuseSpark1xModel (not the draft model; the mid_rs form)."""
    m = getattr(runner, "model", None)
    for path in (("model", "language_model"), ("language_model",), ("model",)):
        x = m
        for p in path:
            x = getattr(x, p, None)
        if x is not None and type(x).__name__ == "MuseSpark1xModel":
            return True
    return False


def _before_graphs(runner, *a, **kw):
    """ModelRunner.init_cuda_graphs (BEFORE), MUSE_VR_XATTN_B_DEFER_PREPARE=1 only: the deferred ``prepare`` of every device
    the load hook recorded, for the target runner, after SGLang sized the KV pool and before any capture (``prepare``
    refuses under capture). A failure is loud and leaves that device unprepared (the op declines, stock path)."""
    if not _is_target_runner(runner):
        return None
    _state["hook_ran"] = True
    if not _state["pending"]:
        return None
    from . import extend_attn_b as xb
    from .extend_attn_b import plan

    done, failed = [], []
    for dix, (hkv, hq) in sorted(_state["pending"].items()):
        try:
            xb.prepare(
                dix,
                max_requests=_state["prep_requests"],
                max_new_tokens=plan.T_HI,
                num_kv_heads=hkv,
                num_q_heads=hq,
            )
            bed.count(NAME, "prepared")
            done.append(dix)
        except Exception as e:  # noqa: BLE001 - loud step-aside: the device stays unprepared, every call runs stock
            for k in ("kernels", "ws", "cap_slots"):
                xb._state[k].pop(dix, None)
            bed.count(NAME, "prepare_failed")
            failed.append(f"{dix}:{type(e).__name__}: {e}")
            logger.warning(
                "muse_vr_plugin: %s deferred prepare FAILED on device %d (%s: %s); the op stays off there "
                "(stock path)",
                NAME,
                dix,
                type(e).__name__,
                e,
            )
    _state["pending"].clear()
    bed.report(
        "xattn_b_defer",
        not failed,
        devices=",".join(str(d) for d in done),
        failed="; ".join(failed) or None,
        when="before init_cuda_graphs",
        prep_requests=_state["prep_requests"],
    )
    return None


def install() -> None:
    import flashinfer.prefill
    import sglang.srt.layers.attention.trtllm_mha_backend as tb
    import sglang.srt.model_executor.model_runner as mr

    if _state["prev_ctx"] is None:
        _state["prev_ctx"] = flashinfer.prefill.trtllm_batch_context_with_kv_cache
        if not _state["compose"]:
            # composed form: nvfp4_kv's replacement owns the attribute; patching it here would hand the hook registry's
            # stale-binding propagation this module's stand-in to rebind
            flashinfer.prefill.trtllm_batch_context_with_kv_cache = _context_attn
    cls = tb.TRTLLMHAAttnBackend
    if not getattr(cls.forward_extend, "__wrapped_by_xattn_b__", False):
        cls.forward_extend = _wrap_forward_extend(cls.forward_extend)
    runner = mr.ModelRunner
    if not getattr(runner.load_model, "__wrapped_by_xattn_b__", False):
        runner.load_model = _wrap_load_model(runner.load_model)


def engage() -> bool:
    if _state["engaged"] is not None:
        return _state["engaged"]
    if not bed.enabled(NAME):
        _state["engaged"] = False
        return False
    clash = [c for c in CONFLICTS if bed.enabled(c)]
    if clash:
        bed.report(NAME, False, reason=f"refused: also enabled {clash} (same calls)")
        _state["engaged"] = False
        return False
    if bed.enabled("nvfp4_kv") and not _state["compose"]:
        bed.report(
            NAME,
            False,
            reason="refused: NVFP4 KV pool without the prefill_dq composition (the op reads E4M3 "
            "pages only; nvfp4_kv composes it with MUSE_VR_NVFP4_PREFILL_DQ=1)",
        )
        _state["engaged"] = False
        return False
    import os

    _state["route"] = os.environ.get("MUSE_VR_XATTN_B_ROUTE", "0") == "1"
    defer = os.environ.get("MUSE_VR_XATTN_B_DEFER_PREPARE", "0").strip()
    preq = os.environ.get("MUSE_VR_XATTN_B_PREP_REQUESTS", "4").strip()
    if defer not in ("0", "1") or preq not in ("1", "2", "3", "4"):
        bed.report(
            NAME,
            False,
            reason=f"MUSE_VR_XATTN_B_DEFER_PREPARE {defer!r} not 0 / 1 or MUSE_VR_XATTN_B_PREP_REQUESTS "
            f"{preq!r} not 1..4",
        )
        _state["engaged"] = False
        return False
    _state["defer"], _state["prep_requests"] = defer == "1", int(preq)
    # the attention kernel's option (extend_attn_b.launch.OPT_ENV; launch validates it again at build). Checked here
    # without importing the op, so a refused value is reported before anything is installed.
    raw_opt = os.environ.get("MUSE_VR_XATTN_B_PFA_OPT", "0").strip() or "0"
    if raw_opt not in PFA_OPTS:
        bed.report(
            NAME,
            False,
            reason=f"MUSE_VR_XATTN_B_PFA_OPT {raw_opt!r} not one of {', '.join(PFA_OPTS)}",
        )
        _state["engaged"] = False
        return False
    pfa_opt = int(raw_opt)
    from .small_extend_attn import _pins_ok

    ok, why = _pins_ok()
    if not ok:
        bed.report(NAME, False, reason=f"pin mismatch: {why}")
        _state["engaged"] = False
        return False
    if _state[
        "defer"
    ]:  # registered before install(): a registration failure refuses with nothing patched
        try:
            from sglang.srt.plugins.hook_registry import HookRegistry, HookType

            HookRegistry.register(T_GRAPHS, _before_graphs, HookType.BEFORE)
        except Exception as e:  # noqa: BLE001
            bed.report(
                NAME,
                False,
                reason=f"deferred-prepare hook registration failed: {type(e).__name__}: {e}",
            )
            _state["engaged"] = False
            return False
        logger.warning(
            "muse_vr_plugin: %s prepare deferred to after the KV pool (MUSE_VR_XATTN_B_DEFER_PREPARE=1)",
            NAME,
        )
    try:
        install()
    except Exception as e:  # noqa: BLE001
        bed.report(NAME, False, reason=f"install failed: {type(e).__name__}: {e}")
        _state["engaged"] = False
        return False
    extra = {}  # the default record is unchanged (keys only when a knob is set)
    if _state["defer"]:
        extra["defer_prepare"] = True
    if _state["prep_requests"] != 4:
        extra["prep_requests"] = _state["prep_requests"]
    if pfa_opt:
        extra["pfa_opt"] = pfa_opt
    bed.report(NAME, True, composed=bool(_state["compose"]), **extra)
    _state["engaged"] = True
    return True


def compose_context(fn, *, bf16_out: bool):
    """For ``nvfp4_kv``'s ``fi_gate`` part with ``prefill_dq``: the context function to place directly under
    ``prefill_dq``'s workspace wrapper, i.e. this component's stand-in around ``fn``; or ``fn`` itself when the component
    is off or refuses (reported).

    ``fn`` must be FlashInfer's installed ``trtllm_batch_context_with_kv_cache`` object (what ``nvfp4_kv`` derives on a
    FlashInfer whose context function holds no FP4-KV device check), checked here by identity behind this component's
    FlashInfer file pins; any other function (e.g. a re-compiled copy) is refused, never wrapped. ``bf16_out`` says the
    workspace calls carry a BF16 output (``nvfp4_kv``'s BF16-output wrapper sits outside ``prefill_dq``'s); the op
    writes BF16 only, so without it the composition is refused instead of declining every call."""
    if not bed.enabled(NAME):
        return fn
    if _state["engaged"] is not None:
        return fn  # already decided outside the composition (refused at engage when nvfp4_kv is enabled)
    import os

    import flashinfer.prefill

    why = None
    if not bf16_out:
        why = "the workspace calls carry an E4M3 output (MUSE_VR_NVFP4_KV_PREFILL_BF16_OUT=0); the op writes BF16 only"
    elif fn is not flashinfer.prefill.trtllm_batch_context_with_kv_cache:
        why = "the function to wrap is not FlashInfer's installed trtllm_batch_context_with_kv_cache"
    elif os.environ.get("MUSE_VR_XATTN_B_ROUTE", "0") == "1":
        why = "per-request routing is not part of the composed form"
    if why:
        bed.report(NAME, False, reason="refused: " + why)
        _state["engaged"] = False
        return fn
    _state["compose"] = True
    if not engage():
        return fn
    if _state["prev_ctx"] is not fn:
        # the previous function was captured before this call (never in a server: nvfp4_kv registers first)
        bed.report(
            NAME,
            False,
            reason="refused: the stand-in's previous function is not the function to wrap",
        )
        _state["engaged"] = False
        return fn
    _state["composed"] = True

    def ctx(*args, **kwargs):
        return _context_attn(*args, **kwargs)

    ctx.__muse_vr_derived_from__ = getattr(fn, "__muse_vr_derived_from__", fn)
    ctx.__wrapped_by_xattn_b__ = True
    return ctx
