# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Host-side helpers for the prepared CuTe providers."""

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_tensor

from ....k3.tensor_adapter import to_cute

_NAMES = ("BFloat16", "Float32", "Uint8", "Uint32", "Int32", "Int64")
_DTYPES = {getattr(torch, name.lower()): getattr(cutlass, name) for name in _NAMES}


def stream(device):
    return cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)


def architecture(device):
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}a"


def fake(shape, stride=None, dtype=torch.bfloat16, align=16):
    """Row-major unless ``stride`` is given."""
    stride = stride or tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))
    dtype = _DTYPES.get(dtype, dtype)
    return make_fake_tensor(dtype, shape, stride, assumed_align=align)


def dynamic_fake(rank, dtype=torch.bfloat16):
    shape = tuple(cute.sym_int() for _ in range(rank))
    return fake(shape, (*(cute.sym_int(64) for _ in range(rank - 1)), 1), dtype)


def dynamic_view(tensor, align=16):
    return to_cute(tensor, align).mark_layout_dynamic(leading_dim=tensor.ndim - 1)
