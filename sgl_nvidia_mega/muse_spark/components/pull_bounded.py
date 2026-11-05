# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Option ``pull_bounded`` (env ``MUSE_VR_PULL_BOUNDED=1``, default off): bounded barrier waits for the NVLink pull
all-gather that ``verify_logits_gather`` and ``ctx_proj_overlap`` (shard mode) issue.

Both components gather with SGLang's pull kernel (``nvlink_comm.all_gather_pull``). Its entry and exit barriers poll a
semaphore until every rank of the group has arrived, with no limit, and the gather is issued from the eager part of
the step: if one rank never issues its half (for example it raised between two collectives), the other ranks poll
forever instead of failing. With this option on, those two call sites launch ``csrc/pull_gather_bounded.cuh``
instead: SGLang's kernel and launcher with the same grid, block, barrier windows, loads, stores and memory ordering,
except that each barrier wait ends the kernel with a trap (a CUDA error on the host, after one diagnostic line from
the device) after ``MUSE_VR_PULL_BOUNDED_MAX_POLLS`` unsuccessful polls (default ``DEFAULT_MAX_POLLS``; the wall time
that takes is the poll count times the latency of one semaphore load, measured by ``tests/dist_pull_bounded.py
--mode missing``). The gathered bytes are the same as the stock kernel's (``tests/dist_pull_bounded.py``).

The plugin copy is used only while the SGLang sources it was derived from match their sha256 pins and the copy itself
matches ``SOURCE_SHA256`` (the validated file); otherwise (and with the option off) the call goes to SGLang's own
``all_gather_pull``. Both kernels keep the same barrier windows on the plane's semaphores, so calls of the two may be
interleaved on one plane (the bitwise test does). ``prepare(dtype)`` builds the kernel module; the components call it
before the collective that creates their gather plane, so no rank can wait on a peer that is still compiling.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import torch

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/ops/communication/nvlink_comm.py": (
        "all_gather_pull",
        "get_multicast_ptr",
    ),
    "kernels/jit/utils/__init__.py": (
        "is_arch_support_pdl",
        "load_jit",
        "make_cpp_args",
    ),
}


logger = logging.getLogger(__name__)

NAME = "pull_bounded"  # env MUSE_VR_PULL_BOUNDED
ENV_MAX_POLLS = "MUSE_VR_PULL_BOUNDED_MAX_POLLS"
DEFAULT_MAX_POLLS = 1 << 27
MIN_MAX_POLLS, MAX_MAX_POLLS = 1 << 10, (1 << 32) - 1
NUM_UNROLL = 4  # SGLang's all_gather_pull default, which both call sites use

_CUH = Path(__file__).resolve().parent / "csrc" / "pull_gather_bounded.cuh"

# sha256 of the SGLang sources (relative to the directory holding the ``sglang`` package) the plugin copy reproduces:
# the pull kernel and its launcher, the barrier and semaphore operations, the PTX wrappers they use, and the Python
# wrapper whose defaults (unroll, multicast pointer, block hint) the copy is launched with.
PINS = {
    "sglang/kernels/jit/csrc/distributed/nvlink_comm.cuh": (
        "8fb785518279cf8944a6a438036cfe39a2c00ecc6b0b5b86a6a2a0b46ad01e37",
        "e3c8f3c93a61c8788d4b97b9fbfa9cc0b5b1c9ee1b2154c9e7e04ab642431367",
    ),
    "sglang/kernels/jit/include/sgl_kernel/distributed/communicator.cuh": "30d5c75e86e86da7ce99cd6eab85212fdbc71f83a8e3c715c4e9f2c6c1d8986f",
    "sglang/kernels/jit/include/sgl_kernel/distributed/ptx.cuh": "1ef631431ff74974a5d05758c4af9335cf8937e42ed9f0dc591a8341598be822",
    "sglang/kernels/ops/communication/nvlink_comm.py": (
        "6ba171bd054fd70b1248be37214419887f6731eea9c927c6e0aa9ee0997bc3fd",
        "0388df5a897cc9c437e7fb0a2e6e21231a7a8d0ce0033798846331c0698bdaaa",
    ),
}

# sha256 of csrc/pull_gather_bounded.cuh as validated (bitwise, fault and sanitizer tests); any other file is not used.
SOURCE_SHA256 = "b6aa914f1a704fd1bcdb02f89253b354364dda57350f33013d99d02c923749c0"

_state: dict = {"engaged": None, "max_polls": None}


def _max_polls() -> int:
    v = bed.env_int(ENV_MAX_POLLS, DEFAULT_MAX_POLLS)
    if not MIN_MAX_POLLS <= v <= MAX_MAX_POLLS:
        raise ValueError(
            f"{ENV_MAX_POLLS}={v} outside [{MIN_MAX_POLLS}, {MAX_MAX_POLLS}]"
        )
    return v


def engage() -> bool:
    """Decide once per process (pins, poll limit) and record the decision; True when the bounded copy is in use."""
    if _state["engaged"] is not None:
        return _state["engaged"]
    if not bed.enabled(NAME):
        _state["engaged"] = False
        return False
    for rel, want in PINS.items():
        if not bed.pinned(rel, want):
            bed.report(NAME, False, reason=f"{rel} differs from the pinned revision")
            _state["engaged"] = False
            return False
    got = hashlib.sha256(_CUH.read_bytes()).hexdigest()
    if got != SOURCE_SHA256:
        bed.report(
            NAME,
            False,
            reason=f"{_CUH.name} sha256 {got[:16]} is not the validated {SOURCE_SHA256[:16]}",
        )
        _state["engaged"] = False
        return False
    try:
        max_polls = _max_polls()
    except ValueError as e:
        bed.report(NAME, False, reason=str(e))
        _state["engaged"] = False
        return False
    _state["max_polls"] = max_polls
    _state["engaged"] = True
    bed.report(NAME, True, max_polls=max_polls)
    return True


def _jit_module(dtype: torch.dtype):
    key = ("module", dtype)
    mod = _state.get(key)
    if mod is None:
        from sglang.kernels.jit.utils import (
            is_arch_support_pdl,
            load_jit,
            make_cpp_args,
        )

        args = make_cpp_args(dtype, is_arch_support_pdl())
        mod = _state[key] = load_jit(
            "muse_vr_pull_bounded",
            *args,
            f"unroll{NUM_UNROLL}",
            cuda_files=[str(_CUH)],
            cuda_wrappers=[
                (
                    "all_gather",
                    f"sglang::muse_vr_pull::MuseVrPullBounded<{args}>::all_gather_pull<{NUM_UNROLL}>",
                )
            ],
        )
    return mod


def prepare(dtype: torch.dtype) -> None:
    """Build (or load) the bounded kernel for ``dtype`` when the option is engaged. Call it on every rank before the
    collective that creates the gather plane."""
    if engage():
        _jit_module(dtype)


def all_gather_pull(
    comm,
    input: torch.Tensor,
    output: torch.Tensor,
    *,
    out_mc_ptr: int = 0,
    site: str = "",
) -> bool:
    """``nvlink_comm.all_gather_pull(comm, input, output, out_mc_ptr=...)``, with bounded barrier waits when engaged
    (tallied per call site under this option's name). Returns whether the bounded copy ran."""
    import sglang.kernels.ops.communication.nvlink_comm as nvlink_comm

    if not engage():
        nvlink_comm.all_gather_pull(comm, input, output, out_mc_ptr=out_mc_ptr)
        return False
    _jit_module(input.dtype).all_gather(
        comm,
        input,
        output,
        None,
        out_mc_ptr or nvlink_comm.get_multicast_ptr(output),
        0,
        _state["max_polls"],
    )
    bed.count(NAME, site or "calls")
    return True
