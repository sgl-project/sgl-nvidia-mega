# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Helpers shared by the DeepSeek-V4.1 slot kernel tests (not collected by pytest)."""

import ctypes

import torch

from sgl_nvidia_mega.common import jit

_LATE = r"""
extern "C" __global__ void late_copy(const int* src, int* dst, int n,
                                     unsigned long long delay_ns) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < delay_ns);
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
}
"""
_late_kernel = None


def late_copy(src: torch.Tensor, dst: torch.Tensor, delay_us: float = 50.0) -> None:
    """``dst`` <- ``src`` on the current stream by a kernel that releases its PDL
    dependents at entry and writes only after ``delay_us`` (a late writer)."""
    global _late_kernel
    if _late_kernel is None:
        major, minor = torch.cuda.get_device_capability()
        cubin = jit.compile_source(
            _LATE, "late_copy", f"sm_{major}{minor}a", ("-std=c++17",)
        )
        _late_kernel = jit.load(cubin, "late_copy", (256, 1, 1), 0)
    jit.launch(
        _late_kernel,
        (1, 1, 1),
        [
            jit.pointer(src),
            jit.pointer(dst),
            ctypes.c_int32(src.numel() * src.element_size() // 4),
            ctypes.c_uint64(int(delay_us * 1000)),
        ],
        torch.cuda.current_stream().cuda_stream,
    )


def poison() -> None:
    """Fill memory the caching allocator hands to the next allocations."""
    junk = torch.full((1 << 24,), 0x7F, dtype=torch.uint8, device="cuda")
    del junk


def bitwise(a: torch.Tensor, b: torch.Tensor) -> None:
    assert torch.equal(a.view(torch.int16), b.view(torch.int16)), (
        f"{(a.view(torch.int16) != b.view(torch.int16)).sum().item()} of {a.numel()} differ"
    )
