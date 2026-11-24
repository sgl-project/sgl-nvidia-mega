# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Multicast push of one rank's MXFP8 shard into every rank's gathered operand (csrc/mc_push.cuh), the data movement of
``fp8_prefill``'s AGF form (MUSE_VR_FP8_PREFILL_AGF=1).

``push(src_q, src_sf, dst_q_mc, dst_sf_mc, q_bytes, sf_bytes)``: one launch stores ``q_bytes`` of ``src_q`` (this rank's
E4M3 values) and ``sf_bytes`` of ``src_sf`` (their swizzled scales) through the multicast addresses ``dst_q_mc`` /
``dst_sf_mc`` (this rank's place in the gathered operand of a symmetric buffer with a multicast mapping), so every rank's
copy receives the bytes unchanged. The caller orders it: the shard is complete before the launch (stream order), and a
bounded barrier (``ops.a2a_pull.barrier``) follows before any rank reads its gathered operand.

Launch shape: ``MUSE_VR_FP8_PREFILL_AGF_UNROLL`` (default 4) 16-byte vectors in flight per thread,
``MUSE_VR_FP8_PREFILL_AGF_CTAS_PER_SM`` (default 2) blocks of 512 threads per SM; both read once per process (tunable;
sweep with the AGF unit's timing form).
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("load_jit", "make_cpp_args"),
}


_CUH = Path(__file__).resolve().parents[1] / "csrc" / "mc_push.cuh"
DEFAULT_UNROLL = 4
DEFAULT_CTAS_PER_SM = 2
UNROLLS = (1, 2, 4, 8)


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "")
    return int(v) if v.strip() else default


@functools.lru_cache(maxsize=None)
def config() -> tuple[int, int]:
    """(unroll, CTAs per SM) from the environment (read once per process)."""
    u = _env_int("MUSE_VR_FP8_PREFILL_AGF_UNROLL", DEFAULT_UNROLL)
    c = _env_int("MUSE_VR_FP8_PREFILL_AGF_CTAS_PER_SM", DEFAULT_CTAS_PER_SM)
    if u not in UNROLLS or not 1 <= c <= 8:
        raise ValueError(
            f"MUSE_VR_FP8_PREFILL_AGF_UNROLL={u} (one of {UNROLLS}) / MUSE_VR_FP8_PREFILL_AGF_CTAS_PER_SM={c} (1..8)"
        )
    return u, c


@functools.lru_cache(maxsize=None)
def module(unroll: int):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(int(unroll))
    return load_jit(
        "muse_vr_mc_push",
        *args,
        cuda_files=[str(_CUH)],
        cuda_wrappers=[("push", f"muse_vr::mcp::McPushKernel<{args}>::run")],
    )


@functools.lru_cache(maxsize=None)
def _sms(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def push(
    src_q: torch.Tensor,
    src_sf: torch.Tensor,
    dst_q_mc: int,
    dst_sf_mc: int,
    q_bytes: int,
    sf_bytes: int,
    unroll: int | None = None,
    ctas_per_sm: int | None = None,
) -> None:
    """Store this rank's shard through the multicast addresses (see the module doc)."""
    du, dc = config()
    u = du if unroll is None else int(unroll)
    c = dc if ctas_per_sm is None else int(ctas_per_sm)
    dev = src_q.device
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    module(u).push(
        src_q,
        src_sf,
        int(dst_q_mc),
        int(dst_sf_mc),
        int(q_bytes),
        int(sf_bytes),
        _sms(idx) * c,
    )
