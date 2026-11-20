# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DFlash draft layer boundary: all-reduce + output-side block convolution + residual add + RMSNorm, JIT CUDA.

See ``csrc/draft_ar_norm.cuh`` for the arithmetic and the work split. ``MUSE_VR_DRAFT_ARN_POLL_ALL=1`` (default off) selects
the variant that reads every peer slot at once (same outputs, bit for bit). ``MUSE_VR_DRAFT_ARN_EARLY_TRIGGER=1`` (default off)
signals the dependent launch right after the kernel's own grid dependency wait instead of at its end (same outputs, bit for
bit; the following kernel can prefetch while this one runs).
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": (
        "is_arch_support_pdl",
        "load_jit",
        "make_cpp_args",
    ),
}


_CSRC = Path(__file__).resolve().parent.parent / "csrc" / "draft_ar_norm.cuh"

BLOCK = 8  # tokens per request block
CLUSTER = 16  # CTAs per request block (columns split this many ways); tunable (tests/test_draft_ar_norm.py --bench)
# above SMALL_ROWS rows, fewer CTAs per request block so that every request block gets a cluster of its own within the
# plane's block counters; tunable (tests/dist_draft_boundary.py --bench)
SMALL_ROWS = 64
LARGE_CLUSTER = 8


def cluster_for(rows: int) -> int:
    return CLUSTER if rows <= SMALL_ROWS else LARGE_CLUSTER


@functools.lru_cache(maxsize=None)
def use_pdl() -> bool:
    from sglang.kernels.jit.utils import is_arch_support_pdl

    return bool(is_arch_support_pdl())


@functools.lru_cache(maxsize=None)
def poll_all_from_env() -> bool:
    return os.environ.get("MUSE_VR_DRAFT_ARN_POLL_ALL", "0").strip() not in ("", "0")


@functools.lru_cache(maxsize=None)
def early_trigger_from_env() -> bool:
    return os.environ.get("MUSE_VR_DRAFT_ARN_EARLY_TRIGGER", "0").strip() not in (
        "",
        "0",
    )


@functools.lru_cache(maxsize=None)
def mc_push_from_env() -> bool:
    """``MUSE_VR_DRAFT_ARN_MC=1`` (default off): one store per vector through the push plane's multicast address
    instead of one per peer (``kMcPush``; bit-identical outputs). Needs a plane with a multicast address."""
    return os.environ.get("MUSE_VR_DRAFT_ARN_MC", "0").strip() not in ("", "0")


@functools.lru_cache(maxsize=None)
def module(
    hidden: int,
    world: int,
    pdl: bool,
    cluster: int = CLUSTER,
    poll_all: bool = False,
    early_trigger: bool = False,
    consumer_only: bool = False,
    mc_push: bool = False,
    mbarx: bool = False,
):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    if mbarx:
        args = make_cpp_args(
            hidden,
            world,
            cluster,
            pdl,
            poll_all,
            early_trigger,
            consumer_only,
            mc_push,
            True,
        )
    elif mc_push:
        args = make_cpp_args(
            hidden, world, cluster, pdl, poll_all, early_trigger, consumer_only, True
        )
    elif consumer_only:
        args = make_cpp_args(hidden, world, cluster, pdl, poll_all, early_trigger, True)
    elif early_trigger:
        args = make_cpp_args(hidden, world, cluster, pdl, poll_all, True)
    elif poll_all:
        args = make_cpp_args(hidden, world, cluster, pdl, True)
    else:
        args = make_cpp_args(hidden, world, cluster, pdl)
    return load_jit(
        "muse_vr_draft_ar_norm",
        *args,
        cuda_files=[str(_CSRC)],
        cuda_wrappers=[
            ("run", f"MuseVrDraftArNorm<{args}>::run"),
            ("loopback", f"MuseVrDraftArNorm<{args}>::loopback"),
        ],
    )


@functools.lru_cache(maxsize=None)
def split_from_env() -> int:
    """``MUSE_VR_DRAFT_ARN_SPLIT=k`` (k in 2, 4, 8; default off): a request block's 8 rows are shared by k clusters
    (``draft_ar_norm_split_kernel``), k times as many CTAs on the exchange; the row before each part travels twice.
    Bit-identical outputs. Applies to calls with one-by-one polling and their own pushes (not poll-all, not
    consumer-only)."""
    raw = os.environ.get("MUSE_VR_DRAFT_ARN_SPLIT", "").strip()
    return int(raw) if raw in ("2", "4", "8") else 0


@functools.lru_cache(maxsize=None)
def split_module(
    hidden: int,
    world: int,
    pdl: bool,
    cluster: int,
    split: int,
    early_trigger: bool = False,
    mc_push: bool = False,
):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(hidden, world, cluster, split, pdl, early_trigger, mc_push)
    return load_jit(
        "muse_vr_draft_ar_norm_split",
        *args,
        cuda_files=[str(_CSRC)],
        cuda_wrappers=[("run", f"MuseVrDraftArNormSplit<{args}>::run")],
    )


def run(
    comm,
    partial,
    coef,
    coef_offset,
    coef_stride,
    base,
    residual,
    weight,
    eps,
    out=None,
    cluster=None,
    poll_all=None,
    early_trigger=None,
    consumer_only=False,
    split=None,
    mbarx=None,
    producer_mc=False,
):
    """All-reduce ``partial`` over the communicator's push plane and apply the boundary; updates ``residual`` in place
    and returns the normalized rows. ``poll_all`` / ``early_trigger``: None = from the environment. ``consumer_only``: the
    partial's vectors were already pushed by the producing GEMV (``ops.draft_gemv.push``); this call does not push them.
    ``producer_mc`` (with consumer_only): that GEMV stored them through the multicast address, so this rank's own slot was
    written too and is polled and scrubbed (the kernel's kMcPush form, without its push)."""
    hidden = partial.shape[-1]
    if out is None:
        out = torch.empty_like(partial)
    if cluster is None:
        cluster = cluster_for(partial.shape[0])
    if poll_all is None:
        poll_all = poll_all_from_env()
    if early_trigger is None:
        early_trigger = early_trigger_from_env()
    # the multicast push replaces this call's own pushes; a consumer-only call pushes nothing, and polls its own slot when
    # the producing GEMV stored through the multicast address
    mc_push = (consumer_only and bool(producer_mc)) or (
        mc_push_from_env() and not consumer_only
    )
    if split is None:
        split = split_from_env()
    rows = partial.shape[0]
    if split and (rows + (rows // BLOCK) * (split - 1)) * hidden * 2 > int(
        comm.max_push_size
    ):
        split = (
            0  # the previous-row copies would not fit the push slot: the plain kernel
        )
    if split and not poll_all and not consumer_only:
        split_module(
            hidden,
            comm.world_size,
            use_pdl(),
            cluster,
            int(split),
            bool(early_trigger),
            bool(mc_push),
        ).run(
            comm.obj,
            partial,
            coef,
            int(coef_offset),
            int(coef_stride),
            base,
            residual,
            weight,
            out,
            float(eps),
        )
        return out
    if mbarx is None:
        mbarx = mbarx_from_env()
    module(
        hidden,
        comm.world_size,
        use_pdl(),
        cluster,
        bool(poll_all),
        bool(early_trigger),
        bool(consumer_only),
        bool(mc_push),
        bool(mbarx),
    ).run(
        comm.obj,
        partial,
        coef,
        int(coef_offset),
        int(coef_stride),
        base,
        residual,
        weight,
        out,
        float(eps),
    )
    return out


@functools.lru_cache(maxsize=None)
def mbarx_from_env() -> bool:
    """``MUSE_VR_DRAFT_ARN_MBARX=1`` (default off): the row sums of squares cross the cluster by st.async into each CTA's
    shared memory with an mbarrier transaction count instead of two full cluster barriers (``kMbarX``). Same partial sums,
    same CTA-order total: bit-identical outputs."""
    return os.environ.get("MUSE_VR_DRAFT_ARN_MBARX", "0").strip() not in ("", "0")


def loopback(
    partial_all,
    rank,
    coef,
    coef_offset,
    coef_stride,
    base,
    residual,
    weight,
    eps,
    cluster=CLUSTER,
    max_clusters=0,
):
    """Test entry: every rank's partial as ``[world, M, H]`` on one GPU; plays ``rank``."""
    world, _, hidden = partial_all.shape
    out = torch.empty_like(residual)
    module(hidden, world, False, cluster).loopback(
        partial_all,
        int(rank),
        coef,
        int(coef_offset),
        int(coef_stride),
        base,
        residual,
        weight,
        out,
        float(eps),
        int(max_clusters),
    )
    return out
