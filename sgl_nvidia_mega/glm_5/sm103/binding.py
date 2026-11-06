# SPDX-License-Identifier: Apache-2.0
"""Thin ownership adapter for the typed TVM-FFI binding."""

from __future__ import annotations


class CompiledModule:
    """Keep the FFI library alive; C++ owns Args, descriptors and CUDA launch."""

    def __init__(self, native, cubin, variant, device):
        self.native = native
        metadata, self._bind = native.load(cubin, variant.symbol, device)
        self.values = tuple(map(int, metadata))

    def metadata(self):
        return self.values[:9]

    def bind(self, *values):
        return self._bind(*values)
