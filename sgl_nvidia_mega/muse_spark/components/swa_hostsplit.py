# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``swa_hostsplit`` (env ``MUSE_VR_SWA_HOSTSPLIT=1`` + ``MUSE_VR_SWA_HOST_RATIO=<r_host>``, default off): size
the HiCache host tiers of a full + sliding-window model as if the device pool used ``--swa-full-tokens-ratio r_host``,
while the device pool keeps the served (smaller) ratio.

SGLang splits ``--hicache-size`` across a hybrid model's device pools in proportion to each pool's
``host_capacity_bytes``, falling back to the pool's device KV bytes when that attribute is unset
(``hybrid_pool_assembler._split_hicache_size``). On the plain ``SWAKVPool`` path nothing sets the attribute, so the
host sliding-window share always follows the device ratio. A prefix hit needs the trailing sliding window of the match
point on the device or on the host, so a small host sliding-window tier limits prefix hits while the device
sliding-window pool can sit mostly idle. This component sets ``host_capacity_bytes`` on the two pools for the duration
of the host-pool build only (``_SwaStrategy.build``): the full pool's per-token bytes times its token count, and the
sliding-window pool's per-token bytes times ``r_host`` times the full pool's token count. The host total stays
``--hicache-size``; per-token sizes, layouts, transfers and every device pool are unchanged. A DFLASH draft sidecar
sizes itself from the target's sliding-window host pool by ratio and follows.

Engaged only with hierarchical caching on, an explicit ``--hicache-size`` (> 0), the plain ``SWAKVPool`` path with
``host_capacity_bytes`` unset, a realized device ratio below ``r_host``, and the SGLang sources matching their pins;
otherwise it steps aside, loudly.
"""

from __future__ import annotations

import logging
import os

from . import bed

logger = logging.getLogger(__name__)

NAME = "swa_hostsplit"  # env MUSE_VR_SWA_HOSTSPLIT (+ MUSE_VR_SWA_HOST_RATIO)
ENV_RATIO = "MUSE_VR_SWA_HOST_RATIO"
TARGET = "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler._SwaStrategy.build"

PINS = {
    "sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py": (
        "35365ec237d193ac77b5adbba9fe239cde205e3747b0da635d3491126b594ac7",
        "a5e1156a9db13593641ea9d649769761ce745fe82cf17c087a59b300d67e17ae",
    ),
    "sglang/srt/mem_cache/swa_memory_pool.py": (
        "5d518102fad0bb444176815e1e829a63197b6939ff656bf282253ffa09c90fab",
        "14f2b1daebdc9d872b5786cfc0beed26a413bb41717b70a737a4cb4c31951dfc",
    ),
}


def ratio() -> float:
    try:
        return float(os.environ.get(ENV_RATIO, "0") or 0)
    except ValueError:
        return 0.0


def enabled() -> bool:
    return bed.enabled(NAME) and ratio() > 0


def pins_match() -> tuple[bool, str]:
    for rel, want in PINS.items():
        if not bed.pinned(rel, want):
            return False, f"{rel} differs from the pinned revision"
    return True, "ok"


def bytes_per_token(pool) -> float:
    """Device KV bytes per token slot of one pool (all its layers, K + V), from its allocated rows; independent of a
    post-capture resize that sets ``.size`` below the reserved rows."""
    return sum(pool.get_kv_size_bytes()) / pool.k_buffer[0].shape[0]


def plan(full_pool, swa_pool, r_host: float):
    """(full host_capacity_bytes, sliding-window host_capacity_bytes, realized device ratio, reason)."""
    r_dev = swa_pool.size / full_pool.size
    if not r_host > r_dev:
        return None, None, r_dev, f"device ratio {r_dev:.4f} >= host ratio {r_host:.4f}"
    fb = int(bytes_per_token(full_pool) * full_pool.size)
    sb = int(bytes_per_token(swa_pool) * full_pool.size * r_host)
    return fb, sb, r_dev, "ok"


def _build(original, self, *args, **kwargs):
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
    from sglang.srt.runtime_context import get_memory

    mem = get_memory()
    kvcache = kwargs.get("kvcache")
    hs = getattr(mem, "hicache_size", 0) or 0
    why = None
    if not getattr(mem, "enable_hierarchical_cache", False):
        why = "hierarchical caching is off"
    elif type(kvcache) is not SWAKVPool:
        why = f"pool class {type(kvcache).__name__} is not the plain SWAKVPool"
    elif not hs > 0:
        why = "no explicit --hicache-size (ratio sizing)"
    if why is None:
        full, swa = kvcache.full_kv_pool, kvcache.swa_kv_pool
        if (
            getattr(full, "host_capacity_bytes", None) is not None
            or getattr(swa, "host_capacity_bytes", None) is not None
        ):
            why = "host_capacity_bytes already set"
    if why is None:
        fb, sb, r_dev, reason = plan(full, swa, ratio())
        if fb is None:
            why = reason
    if why is not None:
        bed.report(NAME, False, reason=why)
        logger.warning(
            "muse_vr_plugin: swa_hostsplit: STEPPED ASIDE (%s): stock host split", why
        )
        return original(self, *args, **kwargs)
    full.host_capacity_bytes, swa.host_capacity_bytes = fb, sb
    try:
        result = original(self, *args, **kwargs)
    finally:
        full.host_capacity_bytes = swa.host_capacity_bytes = None
    hp = {}
    try:
        from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import PoolName

        hp = {
            n: result.host_pool_group.get_pool(p).size
            for n, p in (("full", PoolName.KV), ("swa", PoolName.SWA))
        }
    except Exception:  # noqa: BLE001 - the record is best-effort; the build itself already succeeded
        pass
    bed.report(
        NAME,
        True,
        r_device=round(r_dev, 4),
        r_host=ratio(),
        hicache_size_gb=hs,
        host_tokens=hp,
    )
    logger.warning(
        "muse_vr_plugin: swa_hostsplit: host split as ratio %.3f (device %.4f), host tokens %s",
        ratio(),
        r_dev,
        hp,
    )
    return result


def register() -> bool:
    if not enabled():
        return False
    ok, why = pins_match()
    if not ok:
        bed.report(NAME, False, reason=why)
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(TARGET, _build, HookType.AROUND)
    return True
