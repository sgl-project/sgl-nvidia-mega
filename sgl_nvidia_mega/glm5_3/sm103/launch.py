# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Launch-side helpers for the compiled kernels: the driver function of a compiled
kernel and the co-residency checks that cooperative and cluster launches rely on.

Kernels are launched by the DSL launcher (``.launch(..., cooperative=..., cluster=...,
use_pdl=...)``), which covers every combination these kernels use (cooperative + cluster
+ programmatic dependent launch in one launch). The launcher sets no
memory-synchronization domain, so every kernel runs in the stream's default domain; the
fenced start edge of the o_proj hop (PTX ISA 9.7.15.14, release fence at gpu scope)
needs producer and consumer in the same domain, which holds as long as no stream carries
a domain attribute.

A cooperative grid's CTAs must all be resident at once (they wait on each other), and a
cluster grid's clusters must fit the device. Both are checked once per compiled kernel
with the driver's occupancy queries on the kernel's own function and shared-memory size,
and the caller uses a launch form only where its check passes.
"""

from dataclasses import dataclass

from cuda.bindings import driver as cu

__all__ = ["KernelHandle", "kernel_handle", "fits_cooperative", "max_active_clusters"]


def _ok(res, what):
    err = res[0]
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{what} failed: {err}")
    return res[1:] if len(res) > 2 else (res[1] if len(res) == 2 else None)


@dataclass(frozen=True)
class KernelHandle:
    """The driver function of a compiled single-kernel unit and its dynamic shared
    memory."""

    function: object  # CUfunction (current context)
    dynamic_smem: int
    num_sms: int


def kernel_handle(compiled, device_index: int, dynamic_smem: int) -> KernelHandle:
    """Resolve the (single) kernel of a compiled unit on the current context.
    ``dynamic_smem`` is the unit's dynamic shared-memory size (the kernel class states
    it)."""
    library = compiled.library
    count = _ok(cu.cuLibraryGetKernelCount(library), "cuLibraryGetKernelCount")
    if count != 1:
        raise RuntimeError(f"expected one kernel in the compiled unit, found {count}")
    kernels = _ok(
        cu.cuLibraryEnumerateKernels(count, library), "cuLibraryEnumerateKernels"
    )
    function = _ok(cu.cuKernelGetFunction(kernels[0]), "cuKernelGetFunction")
    smem = dynamic_smem
    dev = _ok(cu.cuDeviceGet(device_index), "cuDeviceGet")
    sms = _ok(
        cu.cuDeviceGetAttribute(
            cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev
        ),
        "cuDeviceGetAttribute",
    )
    return KernelHandle(function, int(smem), int(sms))


def fits_cooperative(handle: KernelHandle, ctas: int, threads: int) -> bool:
    """True iff ``ctas`` CTAs of ``threads`` threads can all be resident at once."""
    per_sm = _ok(
        cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            handle.function, threads, handle.dynamic_smem
        ),
        "cuOccupancyMaxActiveBlocksPerMultiprocessor",
    )
    return int(per_sm) * handle.num_sms >= ctas


def max_active_clusters(
    handle: KernelHandle, grid: int, threads: int, cluster: int, stream=None
) -> int:
    """Clusters of ``cluster`` CTAs of this kernel that can be resident at once."""
    cfg = cu.CUlaunchConfig()
    cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid, 1, 1
    cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = threads, 1, 1
    cfg.sharedMemBytes = handle.dynamic_smem
    cfg.hStream = stream if stream is not None else cu.CUstream(0)
    attr = cu.CUlaunchAttribute()
    attr.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
    attr.value.clusterDim.x, attr.value.clusterDim.y, attr.value.clusterDim.z = (
        cluster,
        1,
        1,
    )
    cfg.attrs = [attr]
    cfg.numAttrs = 1
    return int(
        _ok(
            cu.cuOccupancyMaxActiveClusters(handle.function, cfg),
            "cuOccupancyMaxActiveClusters",
        )
    )
