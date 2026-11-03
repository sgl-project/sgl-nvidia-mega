# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``kv_accounting`` (env ``MUSE_VR_KV_ACCOUNTING=1``, default off): boot-time KV capacity record. Logging only.

After the KV cache is built (post-capture sizing already applied), each rank logs one WARNING line and, when
``MUSE_VR_ENGAGE_DIR`` is set, writes ``kv_accounting_<pid>.json`` with, per device pool (target full / sliding-window
sub-pools, DFLASH draft pool or its sub-pools) and per HiCache host pool (target pools and draft sidecars): tokens,
layers and bytes (backed bytes for post-capture pools). It changes nothing the server does, so both arms of a capacity
comparison can run it.
"""

from __future__ import annotations

import json
import logging
import os
import time

from . import bed

logger = logging.getLogger(__name__)

NAME = "kv_accounting"  # env MUSE_VR_KV_ACCOUNTING
T_BUILD = "sglang.srt.mem_cache.kv_cache_builder.build_kv_cache"
T_DRAFT_ALLOC = (
    "sglang.srt.speculative.dflash_worker_v2.DFlashWorkerV2.alloc_memory_pool"
)

_state: dict = {"draft_pool": None, "record": None}


def enabled() -> bool:
    return bed.enabled(NAME)


def _bytes(pool) -> int:
    try:
        if getattr(pool, "post_capture_active", False):
            return int(pool.post_capture_backed_bytes)
        size = pool.get_kv_size_bytes()
        return int(sum(size) if isinstance(size, tuple) else size)
    except Exception:  # noqa: BLE001 - logging only
        return -1


def _sub(pool) -> dict:
    tokens = int(getattr(pool, "size", 0) or 0)
    b = _bytes(pool)
    return {
        "tokens": tokens,
        "layers": int(getattr(pool, "layer_num", 0) or 0),
        "bytes": b,
        "bytes_per_token": round(b / tokens, 1) if tokens and b >= 0 else None,
    }


def device_record(pool) -> dict:
    rec = {"type": type(pool).__name__}
    full, swa = getattr(pool, "full_kv_pool", None), getattr(pool, "swa_kv_pool", None)
    if full is not None and swa is not None:
        rec["full"] = _sub(full)
        rec["swa"] = _sub(swa)
    else:
        rec["all"] = _sub(pool)
    return rec


def host_record(tree_cache) -> list:
    out = []
    controller = getattr(tree_cache, "cache_controller", None)
    group = getattr(controller, "mem_pool_host", None)
    for entry in getattr(group, "entries", []) or []:
        hp = getattr(entry, "host_pool", None)
        if hp is None:
            continue
        size = int(getattr(hp, "size", 0) or 0)
        per = getattr(hp, "size_per_token", None)
        out.append(
            {
                "name": str(getattr(entry, "name", "?")),
                "label": getattr(hp, "pool_label", None),
                "tokens": size,
                "layers": int(getattr(hp, "layer_num", 0) or 0),
                "bytes": int(size * per) if per else None,
                "bytes_per_token": per,
            }
        )
    return out


def _rank() -> int:
    try:
        import torch.distributed as dist

        return dist.get_rank() if dist.is_initialized() else 0
    except Exception:  # noqa: BLE001
        return 0


def _draft_alloc(original, self, *args, **kwargs):
    out = original(self, *args, **kwargs)
    _state["draft_pool"] = self.draft_model_runner.token_to_kv_pool
    return out


def _build(original, *args, **kwargs):
    result = original(*args, **kwargs)
    try:
        tp_worker = kwargs.get("tp_worker")
        rec = {
            "rank": _rank(),
            "pid": os.getpid(),
            "t": time.time(),
            "target": device_record(tp_worker.model_runner.token_to_kv_pool)
            if tp_worker is not None
            else None,
            "draft": device_record(_state["draft_pool"])
            if _state["draft_pool"] is not None
            else None,
            "host": host_record(result.tree_cache),
        }
        _state["record"] = rec
        d = os.environ.get("MUSE_VR_ENGAGE_DIR")
        if d:
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"kv_accounting_{os.getpid()}.json"), "w") as f:
                json.dump(rec, f, indent=1)

        def fmt(r):
            if r is None:
                return "-"
            return " ".join(
                f"{k}={v['tokens']}tok/{v['layers']}L/{v['bytes'] / 1e9:.2f}GB"
                for k, v in r.items()
                if isinstance(v, dict)
            )

        hosts = " ".join(
            f"{h['name']}={h['tokens']}tok/{(h['bytes'] or 0) / 1e9:.2f}GB"
            for h in rec["host"]
        )
        logger.warning(
            "muse_vr_plugin: kv_accounting rank=%d device target[%s] draft[%s] host[%s]",
            rec["rank"],
            fmt(rec["target"]),
            fmt(rec["draft"]),
            hosts or "-",
        )
        bed.count(NAME, "recorded")
    except Exception as e:  # noqa: BLE001 - logging only, never break the boot
        logger.warning(
            "muse_vr_plugin: kv_accounting failed: %s: %s", type(e).__name__, e
        )
    return result


def register() -> bool:
    if not enabled():
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(T_DRAFT_ALLOC, _draft_alloc, HookType.AROUND)
    HookRegistry.register(T_BUILD, _build, HookType.AROUND)
    bed.report(NAME, True)
    return True
