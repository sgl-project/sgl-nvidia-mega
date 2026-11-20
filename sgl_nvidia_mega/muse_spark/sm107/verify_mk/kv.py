# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The megakernel's KV-cache fields on SGLang's paged FP8 pool (the ``*_sg_s4`` builds: ``MSO_MK_SG_KV=1`` in the vendored kernel).

The vendored stages are built with a small TRT-LLM-shaped placeholder pool and block table (their asserts check that
geometry); ``LayerKv.fields`` then overwrites the KV fields of the layer's parameter blocks with SGLang's buffers:

* K and V buffers of the layer, ``[slots][2 heads][64]`` FP8 (``get_key_buffer`` / ``get_value_buffer``);
* the layer's pool's page table, int32 ``[1, max_pages]`` of 128-slot pages (``swa_page_table`` on sliding-window
  layers, ``page_table`` on full layers), one row for K and V;
* two TMA maps per layer over those buffers (``msoMkSgKvTensorMap``): no swizzle for the sliding-window entry copies,
  SWIZZLE_64B for the full layers' ring.
"""

from __future__ import annotations

import ctypes
from typing import Dict, Tuple

import torch

#: Slots of a page of SGLang's pool (``--page-size``), the builds' ``MSO_MK_SG_PAGE``.
PAGE = 128
KV_HEADS = 2
HEAD_DIM = 64


def check_build(lib) -> None:
    """The library reads SGLang's pool with pages of ``PAGE`` slots (``msoMkSgKv``)."""
    fn = lib.lib.msoMkSgKv
    fn.restype = ctypes.c_int
    fn.argtypes = []
    got = int(fn())
    if got != PAGE:
        raise RuntimeError(
            f"library {lib.mode}: MSO_MK_SG_KV page {got}, the pool's pages hold {PAGE} slots"
        )


def check_buffer(buf: torch.Tensor) -> None:
    assert buf.element_size() == 1 and buf.is_contiguous(), (buf.dtype, buf.stride())
    assert tuple(buf.shape[1:]) == (KV_HEADS, HEAD_DIM), tuple(buf.shape)
    assert buf.shape[0] % PAGE == 0 and buf.data_ptr() % 16 == 0


def tensor_maps(
    lib, k_buf: torch.Tensor, v_buf: torch.Tensor, swizzle64: bool
) -> torch.Tensor:
    """The layer's K and V buffer maps (256 B on the buffers' device: K at 0, V at 128)."""
    fn = lib.lib.msoMkSgKvTensorMap
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_void_p, ctypes.c_int]
    host = (ctypes.c_uint8 * 256)()
    for i, buf in enumerate((k_buf, v_buf)):
        check_buffer(buf)
        rc = fn(
            buf.data_ptr(),
            buf.shape[0],
            ctypes.addressof(host) + 128 * i,
            int(swizzle64),
        )
        assert rc == 0, f"msoMkSgKvTensorMap: error {rc}"
    out = torch.frombuffer(bytearray(host), dtype=torch.uint8).to(k_buf.device)
    assert out.data_ptr() % 128 == 0
    return out


def placeholder(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """A one-block TRT-LLM pool view ``[1, 2, 32, 64]`` FP8 and block table ``[2, 1]`` for the stages' asserts; their
    fields are replaced by ``LayerKv.fields`` before any launch."""
    pool = torch.zeros(
        1, KV_HEADS, 32, HEAD_DIM, dtype=torch.float8_e4m3fn, device=device
    )
    table = torch.zeros(2, 1, dtype=torch.int32, device=device)
    return pool, table


class LayerKv:
    """One layer's SGLang KV binding: its buffers, its pool's page table and the two TMA maps."""

    def __init__(
        self,
        lib,
        k_buf: torch.Tensor,
        v_buf: torch.Tensor,
        table: torch.Tensor,
        sliding: bool,
    ):
        assert k_buf.shape == v_buf.shape and k_buf.device == v_buf.device
        assert table.dtype == torch.int32 and table.dim() == 2 and table.stride(1) == 1
        self.k, self.v, self.table, self.sliding = k_buf, v_buf, table, bool(sliding)
        self.maps = tensor_maps(lib, k_buf, v_buf, swizzle64=not self.sliding)

    def fields(self) -> Dict[str, int]:
        """The parameter fields that replace the placeholder's (row 0 of the table: the one sequence)."""
        if self.sliding:
            return dict(
                AhPool=self.k.data_ptr(),
                AhVPool=self.v.data_ptr(),
                AhBlockTable=self.table.data_ptr(),
                AhBtStride=0,
                AhTmap=self.maps.data_ptr(),
            )
        return dict(
            FaPool=self.k.data_ptr(),
            FaVPool=self.v.data_ptr(),
            FaTable=self.table.data_ptr(),
            FaTableStride=0,
            FaTmap=self.maps.data_ptr(),
        )
