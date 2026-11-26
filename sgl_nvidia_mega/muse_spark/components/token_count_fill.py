# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``token_count_fill`` (env ``MUSE_VR_TOKEN_COUNT_FILL=1``, default off).

When the MoE runs with expert parallelism, ``ForwardBatch.init_new`` gives every batch a one-element device tensor with
the number of non-padded tokens (``global_num_token_non_padded``), made with ``torch.tensor(n, pin_memory=True)
.to(device, non_blocking=True)``: a 4-byte host-to-device copy on the compute stream. Right before the verify graph that
copy reads host memory while the GPU has nothing else queued, and the stream idles around it. This component writes the
same tensor with a fill kernel instead (``torch.full``: no host memory read).

Mechanism: an AROUND hook on ``ForwardBatch.init_new`` runs the original with the module's
``enable_num_token_non_padded`` answering False for that call (its only use in ``forward_batch_info.py`` is the guarded
copy; calls are serialized with a lock and the original function is restored afterwards, also on an exception), then,
if the original function answers True, sets ``global_num_token_non_padded`` to ``torch.full((), n, int32)`` on the
batch's device, where ``n`` is ``global_num_token_non_padded_cpu`` (set by the same code). Nothing in ``init_new`` reads
the tensor after the guarded line. Same value, dtype, shape and device; the first call per process also compares the
value with the host count (one synchronizing read, counted as ``checked``).

``MUSE_VR_TOKEN_COUNT_FILL_REUSE=1`` (opt-in): a target-verify batch reuses the tensor made for the previous one with
the same count on the same device, so a steady decode launches no fill at all (counted as ``reuse``). The verify count
is a per-shape constant (requests times draft tokens), and the verify path only reads the tensor: the registry localizes
it into the graph runner's own buffer, and an eager verify derives a new LOCAL tensor from it. A reused tensor that was
written since it was made (its ``_version`` moved) is not reused again (``reuse: written``); none is made or reused
while a CUDA graph is being captured.
"""

from __future__ import annotations

import logging
import sys
import threading

import torch

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


logger = logging.getLogger(__name__)

NAME = "token_count_fill"  # env MUSE_VR_TOKEN_COUNT_FILL
MODULE = "sglang.srt.model_executor.forward_batch_info"
TARGET = MODULE + ".ForwardBatch.init_new"
FLAG = "enable_num_token_non_padded"

PINS = {
    "sglang/srt/model_executor/forward_batch_info.py": (
        "9fedcb5d0415ba719d1f996f99b831de29f903faeb32f3da180bdc5ac9b02a99",
        "d15f12fa976acae9ba0bd4950b9366d1873abd46a8de66fd8c6ce45e3f0894f9",
    ),
}

_lock = threading.RLock()
_state = {"flag": None, "checked": False}
_reuse: dict = {}  # (device, current CUDA device, n) -> (tensor, its _version when made): target-verify counts


def enabled() -> bool:
    return bed.enabled(NAME)


def pins_match() -> tuple[bool, str]:
    for rel, want in PINS.items():
        if not bed.pinned(rel, want):
            return False, f"{rel} differs from the pinned revision"
    return True, "ok"


def _off() -> bool:
    return False


def _is_target_verify(batch) -> bool:
    mode = getattr(batch, "forward_mode", None)
    f = getattr(mode, "is_target_verify", None)
    return bool(f()) if callable(f) else False


def around_init_new(original, *args, **kwargs):
    mod = sys.modules[MODULE]
    flag = _state["flag"]
    if flag is None or not flag():
        return original(*args, **kwargs)
    with _lock:
        setattr(mod, FLAG, _off)
        try:
            ret = original(*args, **kwargs)
        finally:
            setattr(mod, FLAG, flag)
    n = getattr(ret, "global_num_token_non_padded_cpu", None)
    model_runner = args[2] if len(args) > 2 else kwargs.get("model_runner")
    device = getattr(model_runner, "device", None)
    if (
        n is None
        or device is None
        or getattr(ret, "global_num_token_non_padded", None) is not None
    ):
        bed.count(NAME, "unexpected batch state")
        return ret
    key = None
    if (
        bed.flag(NAME + "_reuse")
        and _is_target_verify(ret)
        and not torch.cuda.is_current_stream_capturing()
    ):
        key = (
            str(device),
            torch.cuda.current_device() if torch.cuda.is_available() else -1,
            int(n),
        )
        hit = _reuse.get(key)
        if hit is not None:
            if hit[0]._version == hit[1]:
                ret.global_num_token_non_padded = hit[0]
                bed.count(NAME, "reuse")
                return ret
            bed.count(NAME, "reuse: written")
    ret.global_num_token_non_padded = torch.full(
        (), int(n), dtype=torch.int32, device=device
    )
    if key is not None:
        t = ret.global_num_token_non_padded
        _reuse[key] = (t, t._version)
    if not _state["checked"] and not torch.cuda.is_current_stream_capturing():
        _state["checked"] = True
        t = ret.global_num_token_non_padded
        ok = t.dtype == torch.int32 and t.dim() == 0 and int(t.item()) == int(n)
        bed.count(NAME, "checked" if ok else "check MISMATCH")
        if not ok:
            logger.error(
                "token_count_fill: filled tensor does not match the host count %d", n
            )
    bed.count(NAME, "fill")
    return ret


def register() -> bool:
    if not enabled():
        return False
    ok, why = pins_match()
    if not ok:
        bed.report(NAME, False, reason=why)
        return False
    import importlib

    mod = importlib.import_module(MODULE)
    _state["flag"] = getattr(mod, FLAG)
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(TARGET, around_init_new, HookType.AROUND)
    bed.report(NAME, True, reuse=bed.flag(NAME + "_reuse"))
    return True
