# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Dense unicast push of one rank's MXFP8 shard into every rank's gathered operand (csrc/uc_push_dense.cuh): the same destination
layout and per-rank write sets as ``ops.mc_push_dense`` (rank p's rows at global rows [p x rows, (p + 1) x rows) of a DENSE
[world x rows, K] operand, its scale words scattered into that operand's 128 x 4 swizzled layout), stored once per rank through
that rank's mapping of its gathered slot instead of once through a multicast address. Same ordering contract (stream order before
the launch, ``ops.a2a_pull.barrier`` after it, two slots by layer parity); the caller zeroes the gathered buffer once at
allocation.

``push(src_q, src_sf, dst_q, dst_sf, rows, rank, world, k, q_cap, sf_cap, fence=True)``: one launch; ``dst_q`` / ``dst_sf`` = per
rank (a list of ``world`` addresses) the address through which this rank reaches that rank's gathered slot value byte 0 / scale
byte 0 (its own address for itself). ``fence``: every storing thread ends with an alias fence (as ``ops.mc_push_dense``); without
it the barrier's alias fence is the only one.

The scale-word map is ``ops.mc_push_dense.word_map`` / ``check_words`` (the kernels share it).

Launch shape: ``MUSE_VR_FP8_PREFILL_AGF_UNROLL`` / ``MUSE_VR_FP8_PREFILL_AGF_CTAS_PER_SM`` (``ops.mc_push``'s knobs and defaults).
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

from . import mc_push

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("load_jit", "make_cpp_args"),
}


_CUH = Path(__file__).resolve().parents[1] / "csrc" / "uc_push_dense.cuh"
MAX_RANKS = 8


@functools.lru_cache(maxsize=None)
def module(unroll: int, fence: bool):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(int(unroll), int(bool(fence)))
    return load_jit(
        "muse_vr_uc_push_dense",
        *args,
        cuda_files=[str(_CUH)],
        cuda_wrappers=[("push", f"muse_vr::ucpd::UcPushDenseKernel<{args}>::run")],
    )


@functools.lru_cache(maxsize=None)
def _sms(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def push(
    src_q: torch.Tensor,
    src_sf: torch.Tensor,
    dst_q: list,
    dst_sf: list,
    rows: int,
    rank: int,
    world: int,
    k: int,
    q_cap: int,
    sf_cap: int,
    fence: bool = True,
    unroll: int | None = None,
    ctas_per_sm: int | None = None,
) -> None:
    """Store this rank's shard densely into every rank's gathered slot through the given per-rank addresses (module doc)."""
    if len(dst_q) != world or len(dst_sf) != world or not 1 <= world <= MAX_RANKS:
        raise ValueError(
            f"uc_push_dense: need one value and one scale address per rank (world {world})"
        )
    du, dc = mc_push.config()
    u = du if unroll is None else int(unroll)
    c = dc if ctas_per_sm is None else int(ctas_per_sm)
    q = [int(a) for a in dst_q] + [0] * (MAX_RANKS - world)
    s = [int(a) for a in dst_sf] + [0] * (MAX_RANKS - world)
    dev = src_q.device
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    module(u, bool(fence)).push(
        src_q,
        src_sf,
        *q,
        *s,
        int(rows),
        int(rank),
        int(world),
        int(k),
        int(q_cap),
        int(sf_cap),
        _sms(idx) * c,
    )
