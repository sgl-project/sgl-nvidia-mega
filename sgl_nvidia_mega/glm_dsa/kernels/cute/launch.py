# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team
# ---- glm_dsa_cute shared body below ----
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
and a failed check declines the fused path (the caller falls back to the existing
implementation).
"""

from dataclasses import dataclass

from cuda.bindings import driver as cu

__all__ = [
    "KernelHandle",
    "kernel_handle",
    "fits_cooperative",
    "max_active_clusters",
    "spread_smem",
    "set_shared_memory_attributes",
]


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


def _device_attr(dev, attr):
    return int(_ok(cu.cuDeviceGetAttribute(attr, dev), "cuDeviceGetAttribute"))


def spread_smem(device_index: int, ctas_per_sm: int) -> int:
    """Dynamic shared memory per CTA (a multiple of 128 B) at which at most
    ``ctas_per_sm`` CTAs of a kernel fit on one SM of this device. The kernel never
    touches the padding; it only keeps the hardware from packing more CTAs per SM."""
    A = cu.CUdevice_attribute
    dev = _ok(cu.cuDeviceGet(device_index), "cuDeviceGet")
    per_sm = _device_attr(dev, A.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)
    per_block = _device_attr(dev, A.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
    reserved = _device_attr(dev, A.CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK)
    need = per_sm // (ctas_per_sm + 1) - reserved + 1
    return min(-(-need // 128) * 128, per_block)


def set_shared_memory_attributes(
    compiled, device_index: int, *, max_carveout: bool, max_dynamic_smem: int = 0
) -> None:
    """Set shared-memory attributes on every kernel of a compiled unit:
    ``max_carveout`` prefers the maximum shared-memory carveout (a scheduling hint:
    an SM running these CTAs stays in the configuration a large-shared-memory kernel
    launched next can share); ``max_dynamic_smem`` raises the dynamic shared-memory
    limit to a padded launch size."""
    lib = getattr(compiled, "library", None)
    if lib is None and hasattr(compiled, "to"):
        compiled.to(device_index)
        lib = getattr(compiled, "library", None)
    if lib is None:
        raise RuntimeError("no driver library for the compiled unit")
    count = _ok(cu.cuLibraryGetKernelCount(lib), "cuLibraryGetKernelCount")
    kernels = _ok(cu.cuLibraryEnumerateKernels(count, lib), "cuLibraryEnumerateKernels")
    dev = _ok(cu.cuDeviceGet(device_index), "cuDeviceGet")
    F = cu.CUfunction_attribute
    attrs = []
    if max_carveout:
        attrs.append(
            (
                F.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                int(cu.CUshared_carveout.CU_SHAREDMEM_CARVEOUT_MAX_SHARED),
            )
        )
    if max_dynamic_smem:
        attrs.append(
            (F.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, int(max_dynamic_smem))
        )
    for k in kernels:
        for attr, value in attrs:
            _ok(cu.cuKernelSetAttribute(attr, value, k, dev), "cuKernelSetAttribute")
