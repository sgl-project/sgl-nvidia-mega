# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DFlash draft layer boundary: all-reduce + output-side block convolution + residual add + RMSNorm, JIT CUDA.

See ``csrc/draft_ar_norm.cuh`` for the arithmetic and the work split.
"""

from __future__ import annotations

import functools
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
def module(hidden: int, world: int, pdl: bool, cluster: int = CLUSTER):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

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
):
    """All-reduce ``partial`` over the communicator's push plane and apply the boundary; updates ``residual`` in place
    and returns the normalized rows."""
    hidden = partial.shape[-1]
    if out is None:
        out = torch.empty_like(partial)
    if cluster is None:
        cluster = cluster_for(partial.shape[0])
    module(hidden, comm.world_size, use_pdl(), cluster).run(
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
