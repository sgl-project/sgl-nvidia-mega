# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""TMA descriptors as ``__grid_constant__`` kernel parameters.

A descriptor in global memory written by the host (``host.tensor_maps``) must be
acquired by every thread that issues TMAs with it
(``fence.proxy.tensormap::generic.acquire``) before its first use in each launch.
That fence has a measurable cost per descriptor, which is on the critical path
whenever a kernel's prologue is not hidden behind its predecessor (a launch gated
by a full dependency). A descriptor passed as a grid-constant parameter needs no
fence (CUDA programming guide, "Using TMA to transfer multi-dimensional arrays"):
the driver places it in parameter space with the launch.

Usage, in a ``@cute.jit`` host function and its ``@cute.kernel``:

    weights = encode_map(address, encoding, dims, strides, box, swizzle, l2)
    self.kernel(args, weights).launch(...)

    @cute.kernel
    def kernel(self, args, weights: GridMap):
        descriptor = map_address(weights)    # generic address for TMA asm
        prefetch_map(descriptor)             # optional, no fence needed

``encode_map`` runs ``cuTensorMapEncodeTiled`` in the launch's host code with the
same arguments ``host.tensor_maps`` takes, so the descriptor bytes are identical to
a host-encoded map of the same spec; under CUDA graph capture the encoded
descriptor is captured with the kernel's parameters.
"""

import cutlass
from cutlass import Int64

from .primitives import asm

try:  # CuTe DSL >= 4.8 (the Rubin image); older DSLs keep global descriptors
    import cuda.bindings.driver as cuda
    from cutlass.base_dsl.typing import GridConstant
    from cutlass.experimental.cuda import (
        TensorMap,
        TensorMapDataFormat,
        TensorMapL2Promotion,
        TensorMapSwizzle,
        create_tensor_map_tiled,
    )

    # Driver data type of each map this package encodes -> (logical dtype, TMA data format) of create_tensor_map_tiled.
    _FORMATS = {
        int(cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8): (cutlass.Uint8, None),
        int(cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B): (
            cutlass.Float4E2M1FNx2,
            TensorMapDataFormat.B4X16_P64,
        ),
    }

    # Kernel parameter annotation for a descriptor built by ``encode_map``.
    GridMap = GridConstant[TensorMap]
    AVAILABLE = True
except ImportError:
    GridMap, AVAILABLE = None, False


def encode_map(address, encoding, dims, strides, box, swizzle, l2):
    """Tiled TMA descriptor of ``address`` (host code of a ``@cute.jit`` function), built
    by CuTe DSL's ``create_tensor_map_tiled``. ``encoding``, ``swizzle`` and ``l2``: CUDA
    driver enum values (``CUtensorMapDataType`` / ``CUtensorMapSwizzle`` /
    ``CUtensorMapL2promotion``); ``dims`` in elements, innermost first; ``strides`` in
    bytes (one per outer dimension, multiples of 16); ``box`` in elements; no
    interleave, no OOB fill, unit element strides (``host.tensor_maps``)."""
    rank = len(dims)
    assert len(strides) == rank - 1 and len(box) == rank
    assert all(s % 16 == 0 for s in strides), strides
    dtype, tma_format = _FORMATS[int(encoding)]
    return create_tensor_map_tiled(
        Int64(address),
        dtype,
        list(dims),
        [s // 16 for s in strides],
        list(box),
        swizzle=TensorMapSwizzle(int(swizzle)),
        l2_promotion=TensorMapL2Promotion(int(l2)),
        tma_format=tma_format,
    )


def map_address(descriptor):
    """Generic address of a ``GridMap`` parameter, for TMA instructions."""
    return descriptor.get_ptr().toint()


@asm()
def prefetch_map(address):
    """``prefetch.tensormap`` of a grid-constant descriptor (no proxy fence)."""
    return "prefetch.tensormap [$0];", "l", (Int64(address),)
