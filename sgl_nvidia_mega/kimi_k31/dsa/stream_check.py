# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""KNOWN K-18 / TSI I5: one-time fail-closed check that tvm-ffi launches on the stream the caller's event joins and PDL
edges assume. The verify path calls ``ensure`` at the head of each stream context that issues tvm-ffi launches (the
selection stream: DCP1 topk_v2 / plan and TSI prep / scan / merge; the fused index query; the index-K store). It
resolves tvm-ffi's launch stream exactly as LaunchKernel does (stream_check.cuh) and raises before the context
launches anything when it differs from ``expected``.

Key: (device, graph capture id or "eager", torch's current stream, expected stream). One check per key; eager keys
carry the stream, so a new stream is checked again. Cost per call: one cudaStreamIsCapturing and a set lookup
(graph replays run no Python)."""

from functools import cache
from pathlib import Path

import torch

# SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {"kernels/jit/utils/compile/loader.py": ("load_jit",)}

_CHECKED = set()


@cache
def _load():
    import sglang
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "k31_stream_check",
        cuda_files=[str(Path(__file__).with_suffix(".cuh"))],
        cuda_wrappers=[("check", "k31_stream_check::Probe::check")],
        extra_include_paths=[str(Path(sglang.__file__).parent / "kernels/jit/csrc")],
    )


def key(device_index, expected):
    """This context's K-18 key: (device, capture id | "eager", torch's current stream, expected stream)."""
    current = torch.cuda.current_stream(device_index).cuda_stream
    capture = "eager"
    if torch.cuda.is_current_stream_capturing():
        import cuda.bindings.driver as drv

        info = drv.cuStreamGetCaptureInfo(drv.CUstream(current))
        if info[0] != drv.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"K-18 stream check: cuStreamGetCaptureInfo failed ({info[0]})")
        capture = int(info[2])
    return (device_index, capture, int(current), int(expected))


def ensure(like: torch.Tensor, expected) -> None:
    """Raise unless tvm-ffi resolves ``expected`` (a torch.cuda.Stream or a raw handle) for launches whose first
    CUDA tensor is on ``like``'s device in the current torch stream context."""
    expected = expected.cuda_stream if isinstance(expected, torch.cuda.Stream) else int(expected)
    k = key(like.device.index, expected)
    if k in _CHECKED:
        return
    _load().check(like, expected)
    _CHECKED.add(k)
