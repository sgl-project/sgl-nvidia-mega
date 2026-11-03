# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``kv_hostbook`` (env ``MUSE_VR_KV_HOSTBOOK=1``, default off): every HiCache host pool of a rank books
against one host-memory snapshot taken before any rank of the node pins its pools.

Without it each host pool, as it is built, re-samples the machine's free memory (cgroup-bounded) and divides it by the
local rank count (``pool_host.base.host_memory_budget_bytes``). Ranks build their pools concurrently, so a rank that
samples after its peers pinned theirs sees a fraction of the memory that is actually its share, and a large
``--hicache-size`` fails at boot on some ranks and not others ("Not enough host memory available"). SGLang already has
the snapshot form (``host_memory_budget_scope``, used by its HiCache auto-sizing): pools built inside the scope book
their bytes against one budget instead of re-sampling. This component takes that snapshot for the whole KV-cache build
(``kv_cache_builder.build_kv_cache``: target host pools and draft sidecars) after a barrier of the node's ranks, so every
rank sees the same free memory, divides it by the local rank count exactly as stock does, and keeps the stock 10 GiB
reserve. Host pool sizes are whatever the flags ask for; only the admission check changes.

Engaged only when hierarchical caching is on and the SGLang sources match their pins.
"""

from __future__ import annotations

import logging

from . import bed

logger = logging.getLogger(__name__)

NAME = "kv_hostbook"  # env MUSE_VR_KV_HOSTBOOK
TARGET = "sglang.srt.mem_cache.kv_cache_builder.build_kv_cache"

PINS = {
    "sglang/srt/mem_cache/kv_cache_builder.py": (
        "79808d4f054dc9edbae0c5c1e2d92818442d74b8a66381f6835eaa64f135853c",
        "578ce61e11904331f6a0bf7d85bf7f5679249b4d1a6718860a9c71b44ad45147",
    ),
    "sglang/srt/mem_cache/pool_host/base.py": (
        "982db6b57fe90ced7d9b052f8ce965c48ccfec7f4d71ec9cdab03cbce253ad4c",
        "e57a060e2fd54905a4f16c4e440bafb8356f1967713c8a50f2667b2057d65bf6",
    ),
    "sglang/srt/mem_cache/host_memory.py": (
        "96be32c23831dd5d6cd7d54ca85de10cdcb850316172008299448b3d9b81cab5",
        "98ad595e237cfb8e0c555fde8992eef590a709e1d8b250b315d858d115991640",
    ),
}


def enabled() -> bool:
    return bed.enabled(NAME)


def pins_match() -> tuple[bool, str]:
    for rel, want in PINS.items():
        if not bed.pinned(rel, want):
            return False, f"{rel} differs from the pinned revision"
    return True, "ok"


def snapshot_budget() -> tuple[int, int, int]:
    """(per-rank budget, node free bytes after the reserve, local ranks): the stock per-pool rule, taken once."""
    from sglang.srt.mem_cache.host_memory import available_host_memory_bytes
    from sglang.srt.mem_cache.pool_host.base import (
        HICACHE_HOST_MEMORY_RESERVE_BYTES,
        ranks_per_host,
    )

    free = (
        available_host_memory_bytes(allow_cgroup_fallback=True)
        - HICACHE_HOST_MEMORY_RESERVE_BYTES
    )
    ranks = ranks_per_host()
    return max(free, 0) // ranks, free, ranks


def _barrier() -> None:
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return
    from sglang.srt.runtime_context import get_parallel

    dist.barrier(group=get_parallel().world_group.cpu_group)


def _build(original, *args, **kwargs):
    from sglang.srt.runtime_context import get_memory

    if not get_memory().enable_hierarchical_cache:
        return original(*args, **kwargs)
    from sglang.srt.mem_cache.pool_host import base as host_base

    _barrier()  # nobody has pinned yet
    budget, free, ranks = snapshot_budget()
    _barrier()  # everybody has its snapshot
    with host_base.host_memory_budget_scope(budget):
        result = original(*args, **kwargs)
        left = host_base._host_memory_budget.get()
    claimed = budget - (left if left is not None else budget)
    bed.report(
        NAME,
        True,
        budget_gb=round(budget / 1e9, 2),
        node_free_gb=round(free / 1e9, 2),
        ranks=ranks,
        claimed_gb=round(claimed / 1e9, 2),
    )
    logger.warning(
        "muse_vr_plugin: kv_hostbook: budget %.2f GB/rank (node free %.2f GB after reserve, %d ranks), "
        "claimed %.2f GB",
        budget / 1e9,
        free / 1e9,
        ranks,
        claimed / 1e9,
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
