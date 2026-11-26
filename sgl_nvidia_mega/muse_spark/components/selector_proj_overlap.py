# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``selector_proj_overlap`` (env ``MUSE_VR_SELECTOR_PROJ_OVERLAP=1``, default off).

The DFLASH2 candidate selector, folded into the draft CUDA graph (``_SelectorDraftSampler``), first takes the candidate
top-k from the target head (draft head, top-k, cross-rank candidate exchange) and then builds its lattice
(``CandidateSelector.build_lattice``), which starts with ``hidden_projection`` of the draft block's predicted hidden
states. That projection depends only on the hidden states, so this component runs it on a side stream at the start of
the sampler call, concurrently with the candidate top-k, and hands the result to ``build_lattice``.

Mechanism: an AROUND hook on ``_SelectorDraftSampler.__call__`` forks a side stream, runs the selector's own
``hidden_projection`` on the same input view as the lattice will receive (into a buffer allocated on the calling stream,
one copy on the side stream), records an event and calls the original. An AROUND hook on
``CandidateSelector.build_lattice`` recognises that input (same storage, shape, strides and dtype): the calling stream
waits for the event, and the projection module is stood in for (for that call only) by one returning the precomputed
tensor. The side branch always joins the calling stream, also on an exception, so CUDA graph capture stays valid. Same
module call on the same input: the lattice is bit-identical (tests/test_selector_proj_overlap.py).
"""

from __future__ import annotations

import logging
import threading

import torch

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


logger = logging.getLogger(__name__)

NAME = "selector_proj_overlap"  # env MUSE_VR_SELECTOR_PROJ_OVERLAP
T_CALL = "sglang.srt.speculative.dflash_worker_v2._SelectorDraftSampler.__call__"
T_LATTICE = "sglang.srt.models.dflash.CandidateSelector.build_lattice"

PINS = {
    "sglang/srt/models/dflash.py": (
        "59ae9eab87fd3416828604018c36ad6a57be07c4ff6d17f699bb9c0be8c719ff",
        "b7b541368cbbb072a679d08ba5bf0f952f4a43f02b6c078562b2ea7a9ddfd01b",
    ),
    "sglang/srt/speculative/dflash_worker_v2.py": (
        "7d462f5f175828bcc3b4d47fcdfa33cb6af4b14669ef016eeb79b1a5191eec8c",
        "8f82de8d666f81f3c1aa43dc651006f4182ff58593fdfe4eea28c5ee2a14bfc0",
    ),
}

_tls = threading.local()
_streams: dict = {}


def enabled() -> bool:
    return bed.enabled(NAME)


def pins_match() -> tuple[bool, str]:
    for rel, want in PINS.items():
        if not bed.pinned(rel, want):
            return False, f"{rel} differs from the pinned revision"
    return True, "ok"


def _key(t: torch.Tensor):
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype)


def _side(device: torch.device):
    idx = device.index if device.index is not None else torch.cuda.current_device()
    s = _streams.get(idx)
    if s is None:
        s = torch.cuda.Stream(device=device)
        _streams[idx] = s
    return s


class _Precomputed(torch.nn.Module):
    def __init__(self, key, value):
        super().__init__()
        self._k, self._v = key, value

    def forward(self, x):
        if _key(x) != self._k:
            raise RuntimeError(
                "selector_proj_overlap: projection input differs from the precomputed one"
            )
        return self._v


def around_call(original, sampler, hidden_states, input_ids, *args, **kwargs):
    sel = getattr(sampler, "selector", None)
    proj_mod = getattr(sel, "hidden_projection", None)
    if args or kwargs or proj_mod is None or not hidden_states.is_cuda:
        return original(sampler, hidden_states, input_ids, *args, **kwargs)
    bs = hidden_states.shape[0] // sampler.block_size
    hs = hidden_states.view(bs, sampler.block_size, -1)[
        :, 1:, :
    ]  # as the stock call slices it
    main = torch.cuda.current_stream(hidden_states.device)
    side = _side(hidden_states.device)
    out = torch.empty(
        hs.shape[:-1] + (proj_mod.out_features,),
        dtype=proj_mod.weight.dtype,
        device=hs.device,
    )
    side.wait_stream(main)
    with torch.cuda.stream(side):
        out.copy_(proj_mod(hs))
    ev = torch.cuda.Event()
    ev.record(side)
    pending = {"sel": sel, "key": _key(hs), "out": out, "ev": ev, "used": False}
    prev = getattr(_tls, "pending", None)
    _tls.pending = pending
    try:
        return original(sampler, hidden_states, input_ids)
    finally:
        _tls.pending = prev
        if not pending["used"]:
            main.wait_event(
                ev
            )  # join the side branch (graph capture needs every forked stream joined)
            bed.count(NAME, "unused")


def around_build_lattice(original, sel, *args, **kwargs):
    p = getattr(_tls, "pending", None)
    hs = kwargs.get("hidden_states")
    if (
        p is None
        or p["sel"] is not sel
        or p["used"]
        or hs is None
        or _key(hs) != p["key"]
    ):
        return original(sel, *args, **kwargs)
    p["used"] = True
    torch.cuda.current_stream(hs.device).wait_event(p["ev"])
    saved = sel.hidden_projection
    sel.hidden_projection = _Precomputed(p["key"], p["out"])
    try:
        out = original(sel, *args, **kwargs)
    finally:
        sel.hidden_projection = saved
    if not torch.cuda.is_current_stream_capturing():
        bed.count(NAME, "overlapped")
    return out


def register() -> bool:
    if not enabled():
        return False
    ok, why = pins_match()
    if not ok:
        bed.report(NAME, False, reason=why)
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(T_CALL, around_call, HookType.AROUND)
    HookRegistry.register(T_LATTICE, around_build_lattice, HookType.AROUND)
    bed.report(NAME, True)
    return True
