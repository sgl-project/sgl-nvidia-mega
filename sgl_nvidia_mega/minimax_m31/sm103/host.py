# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Host pieces shared by the MiniMax-M3.1 kernels: the loaded cubin, tensor maps,
launches, the TP4 arenas in symmetric memory and the operand checks."""

from __future__ import annotations

import ctypes
from typing import Callable, TypeVar

import torch
import torch.distributed as dist

from sgl_nvidia_mega.common.jit import check, driver

TP = 4
T = TypeVar("T")


class Module:
    """A loaded cubin: its kernel and the constants the build appended to it."""

    def __init__(self, cubin: bytes, symbol: str):
        cu = driver()
        (self.module,) = check(cu.cuModuleLoadData(cubin), "cuModuleLoadData")
        (self.function,) = check(
            cu.cuModuleGetFunction(self.module, symbol.encode()), symbol
        )
        self.cubin = cubin

    def constant(self, name: str, ctype=ctypes.c_int64) -> int:
        """The value of the ``__device__`` constant ``name``."""
        cu = driver()
        address, size = check(
            cu.cuModuleGetGlobal(self.module, name.encode()),
            f"cuModuleGetGlobal({name})",
        )
        value = ctype()
        if size != ctypes.sizeof(value):
            raise RuntimeError(
                f"{name} is {size} bytes, expected {ctypes.sizeof(value)}"
            )
        check(cu.cuMemcpyDtoH(ctypes.addressof(value), address, size), "cuMemcpyDtoH")
        return value.value

    def array(self, name: str, count: int) -> tuple:
        """The ``__device__ long long name[count]`` constants."""
        cu = driver()
        address, size = check(
            cu.cuModuleGetGlobal(self.module, name.encode()),
            f"cuModuleGetGlobal({name})",
        )
        values = (ctypes.c_int64 * count)()
        if size != ctypes.sizeof(values):
            raise RuntimeError(
                f"{name} is {size} bytes, expected {ctypes.sizeof(values)}"
            )
        check(cu.cuMemcpyDtoH(ctypes.addressof(values), address, size), "cuMemcpyDtoH")
        return tuple(values)

    def set_shared_memory(self, size: int) -> None:
        cu = driver()
        check(
            cu.cuFuncSetAttribute(
                self.function,
                cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                size,
            ),
            "cuFuncSetAttribute",
        )

    def parameter_size(self, index: int) -> int:
        _, size = check(
            driver().cuFuncGetParamInfo(self.function, index), "cuFuncGetParamInfo"
        )
        return size

    def resident_ctas(self, device: torch.device, threads: int, shared: int) -> int:
        """How many CTAs of the kernel the device holds at once."""
        (per_sm,) = check(
            driver().cuOccupancyMaxActiveBlocksPerMultiprocessor(
                self.function, threads, shared
            ),
            "cuOccupancyMaxActiveBlocksPerMultiprocessor",
        )
        return per_sm * torch.cuda.get_device_properties(device).multi_processor_count


def encode(
    dtype: str,
    pointer: int,
    dims,
    strides,
    box,
    swizzle: str,
    promotion: str,
):
    """cuTensorMapEncodeTiled with unit element strides, no interleave and no
    out-of-bounds fill; ``dtype``, ``swizzle`` and ``promotion`` name the enums'
    suffixes (``UINT8``, ``128B``, ``L2_256B``)."""
    cu = driver()
    u64, u32 = cu.cuuint64_t, cu.cuuint32_t
    (tensor_map,) = check(
        cu.cuTensorMapEncodeTiled(
            getattr(cu.CUtensorMapDataType, f"CU_TENSOR_MAP_DATA_TYPE_{dtype}"),
            len(dims),
            pointer,
            [u64(d) for d in dims],
            [u64(s) for s in strides],
            [u32(b) for b in box],
            [u32(1)] * len(dims),
            cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            getattr(cu.CUtensorMapSwizzle, f"CU_TENSOR_MAP_SWIZZLE_{swizzle}"),
            getattr(
                cu.CUtensorMapL2promotion, f"CU_TENSOR_MAP_L2_PROMOTION_{promotion}"
            ),
            cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        ),
        "cuTensorMapEncodeTiled",
    )
    return tensor_map


def place_map(blob: ctypes.Array, offset: int, tensor_map) -> None:
    ctypes.memmove(ctypes.addressof(blob) + offset, tensor_map.getPtr(), 128)


def launch(
    function,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    shared: int,
    parameters: list[ctypes.Array],
    stream: int,
    *,
    cooperative: bool = False,
) -> None:
    """One launch without programmatic dependent launch (the kernels do not wait on a
    predecessor's grid), each parameter passed as its bytes; graph-capturable."""
    cu = driver()
    addresses = (ctypes.c_void_p * len(parameters))(
        *(ctypes.addressof(p) for p in parameters)
    )
    config = cu.CUlaunchConfig()
    config.gridDimX, config.gridDimY, config.gridDimZ = grid
    config.blockDimX, config.blockDimY, config.blockDimZ = block
    config.sharedMemBytes = shared
    config.hStream = cu.CUstream(stream)
    attributes = []
    if cooperative:
        attribute = cu.CUlaunchAttribute()
        attribute.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_COOPERATIVE
        attribute.value.cooperative = 1
        attributes.append(attribute)
    config.attrs = attributes
    config.numAttrs = len(attributes)
    check(
        cu.cuLaunchKernelEx(config, function, ctypes.addressof(addresses), 0),
        "cuLaunchKernelEx",
    )


def cuda_device(device) -> torch.device:
    """``device`` as ``cuda:N``; an index-less CUDA device is the current one."""
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"the MiniMax-M3.1 kernels run on CUDA devices, not {device}")
    return (
        device
        if device.index is not None
        else torch.device("cuda", torch.cuda.current_device())
    )


def agree(group, step: Callable[[], T], what: str) -> T:
    """Run ``step`` on every rank of ``group``; if it raised on any rank, raise on every
    rank with each rank's error, so no rank waits on a collective the others left."""
    result, error = None, None
    try:
        result = step()
    except Exception as e:  # gathered, then raised on every rank
        error = f"{type(e).__name__}: {e}"
    errors = [None] * dist.get_world_size(group)
    dist.all_gather_object(errors, error, group=group)
    failed = [f"rank {rank}: {e}" for rank, e in enumerate(errors) if e]
    if failed:
        raise RuntimeError(f"{what} failed: " + "; ".join(failed))
    return result


class Arena:
    """One zeroed arena per rank in symmetric memory, mapped by every TP4 peer;
    created collectively before CUDA-graph capture."""

    def __init__(self, group, device, nbytes: int, *, fill=None):
        from torch.distributed import _symmetric_memory as symm_mem

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("TP4 arenas must be created before CUDA-graph capture")
        if dist.get_world_size(group) != TP:
            raise ValueError("the MiniMax-M3.1 kernels run on TP4")
        self.rank = dist.get_rank(group)
        self.device = device = cuda_device(device)
        with torch.cuda.device(device):
            self.buffer = agree(
                group,
                lambda: symm_mem.empty(nbytes, dtype=torch.uint8, device=device),
                "allocating a TP4 arena",
            )
            self.handle = symm_mem.rendezvous(self.buffer, group)
            self.peers = [int(p) for p in self.handle.buffer_ptrs]
            self.multicast = int(self.handle.multicast_ptr or 0)
            if fill is None:
                self.buffer.zero_()
            else:
                fill(self.buffer)
            torch.cuda.synchronize(device)
        dist.barrier(group=group)

    @property
    def base(self) -> int:
        return self.peers[self.rank]


def check_tensor(
    tensor: torch.Tensor, name: str, dtype, shape, device: torch.device
) -> None:
    """A contiguous, 16-byte aligned ``dtype`` tensor of ``shape`` (-1: any extent) on
    ``device``."""
    ok = (
        isinstance(tensor, torch.Tensor)
        and tensor.device == device
        and tensor.dtype == dtype
        and tensor.dim() == len(shape)
        and all(s < 0 and n > 0 or s == n for s, n in zip(shape, tensor.shape))
        and tensor.is_contiguous()
        and tensor.data_ptr() % 16 == 0
    )
    if not ok:
        got = (
            f"{tensor.dtype} {list(tensor.shape)} on {tensor.device}"
            if isinstance(tensor, torch.Tensor)
            else type(tensor).__name__
        )
        raise ValueError(
            f"{name} must be a contiguous, 16-byte aligned {dtype} {list(shape)} tensor "
            f"on {device}, got {got}"
        )


def _extent(tensor: torch.Tensor) -> tuple[int, int]:
    """The byte range a (possibly strided) tensor spans."""
    span = sum((n - 1) * stride for n, stride in zip(tensor.shape, tensor.stride()))
    start = tensor.data_ptr()
    return start, start + (span + 1) * tensor.element_size()


def overlaps(a: torch.Tensor, b: torch.Tensor) -> bool:
    (a0, a1), (b0, b1) = _extent(a), _extent(b)
    return a0 < b1 and b0 < a1
