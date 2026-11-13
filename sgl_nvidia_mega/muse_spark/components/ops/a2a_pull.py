# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""All-to-all pull of MXFP8 row blocks and its bounded barrier (csrc/a2a_pull.cuh), the data movement of
``o_rowshard``'s pull form.

``pull(dst_q, dst_sf, srcs, width, rows)``: in one launch, for every source rank r, copies the block that rank staged
for this rank (E4M3 values [rows, width] followed by their 128 x 4 swizzled scale tiles) into columns
[r * width, (r + 1) * width) of ``dst_q`` (row-major [rows, world * width]) and into tile columns [r * T, (r + 1) * T) of
every row tile of ``dst_sf`` (T = width / 128), which is the swizzled scale layout of the [rows, world * width] operand.
Bytes are unchanged. ``srcs[r]`` is the device address of rank r's block (its symmetric buffer's mapped address plus
the block offset). The caller orders the launch between two barriers.

``barrier(like, pads, rank, max_polls)``: every rank of the group signals every other rank (release, system scope) and
waits for every other rank's signal (acquire, system scope) through ``pads`` (per rank, the address of its symmetric
signal pad of ``world`` zero-initialized int32 slots). Each wait is bounded by ``max_polls`` attempts, then the kernel
traps (a CUDA error on the host) after printing one line.

Launch shape of the pull: ``MUSE_VR_O_PULL_UNROLL`` (default 8) vectors in flight per thread, ``MUSE_VR_O_PULL_CTAS_PER_SM``
(default 2) blocks of 512 threads per SM; both are read once per process.
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


_CUH = Path(__file__).resolve().parents[1] / "csrc" / "a2a_pull.cuh"
MAX_RANKS = 8
DEFAULT_UNROLL = 8
DEFAULT_CTAS_PER_SM = 2
UNROLLS = (1, 2, 4, 8, 16)


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "")
    return int(v) if v.strip() else default


@functools.lru_cache(maxsize=None)
def config() -> tuple[int, int]:
    """(unroll, CTAs per SM) from the environment (read once per process)."""
    u = _env_int("MUSE_VR_O_PULL_UNROLL", DEFAULT_UNROLL)
    c = _env_int("MUSE_VR_O_PULL_CTAS_PER_SM", DEFAULT_CTAS_PER_SM)
    if u not in UNROLLS or not 1 <= c <= 8:
        raise ValueError(
            f"MUSE_VR_O_PULL_UNROLL={u} (one of {UNROLLS}) / MUSE_VR_O_PULL_CTAS_PER_SM={c} (1..8)"
        )
    return u, c


@functools.lru_cache(maxsize=None)
def module(unroll: int):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(int(unroll))
    return load_jit(
        "muse_vr_a2a_pull",
        *args,
        cuda_files=[str(_CUH)],
        cuda_wrappers=[
            ("pull", f"muse_vr::a2a::A2aPullKernel<{args}>::run"),
            ("barrier", "muse_vr::a2a::A2aBarrierKernel::run"),
        ],
    )


@functools.lru_cache(maxsize=None)
def _sms(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def _pad8(addrs) -> list[int]:
    a = [int(x) for x in addrs]
    if not 1 <= len(a) <= MAX_RANKS:
        raise ValueError(f"a2a_pull: 1 to {MAX_RANKS} ranks")
    return a + [0] * (MAX_RANKS - len(a))


def pull(
    dst_q: torch.Tensor,
    dst_sf: torch.Tensor,
    srcs,
    width: int,
    rows: int,
    unroll: int | None = None,
    ctas_per_sm: int | None = None,
) -> None:
    """Copy every rank's block for this rank into the full-width operand (see the module doc)."""
    du, dc = config()
    u = du if unroll is None else int(unroll)
    c = dc if ctas_per_sm is None else int(ctas_per_sm)
    dev = dst_q.device
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    module(u).pull(
        dst_q, dst_sf, *_pad8(srcs), len(srcs), int(width), int(rows), _sms(idx) * c
    )


def barrier(like: torch.Tensor, pads, rank: int, max_polls: int) -> None:
    """The bounded barrier over the ranks whose signal pads are ``pads`` (see the module doc)."""
    module(config()[0]).barrier(
        like, *_pad8(pads), len(pads), int(rank), int(max_polls)
    )
