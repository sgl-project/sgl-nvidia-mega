# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Portions Copyright 2023-2024 SGLang Team (Apache-2.0)
# Derived from SGLang (https://github.com/sgl-project/sglang), Apache License 2.0; see include/LICENSE.sglang.
# MHATokenToKVPoolNVFP4Host follows MHATokenToKVPoolMXFP8Host (python/sglang/srt/mem_cache/pool_host/mha_mxfp8.py at
# the pinned revision).
# Modified by NVIDIA: host pool for the native NVFP4 (trtllm-gen) device pool. Packed E2M1 rows ride the MHA host pool
# with a head_dim of head_dim / 2 bytes; the native E4M3 block-scale tiles (one contiguous [head, token, scale] block per
# page and layer, V four-token interleaved inside the page) move as whole-page rows.
"""HiCache host pool for native NVFP4 KV (``nvfp4_kv`` component, part ``host_pool``).

HiCache moves whole pages (host and device indices are page-aligned runs of ``page_size`` slots), so the in-page
order of a native scale tile never has to be decoded: a page's tile is copied byte for byte. Load-back and
write-back use the same staged JIT kernels as the stock MXFP8 host pool.
"""

from __future__ import annotations

from typing import Sequence

import sglang.kernels.ops.kvcache.hicache as _sgl_kernels_ops_kvcache_hicache
import torch
from sglang.kernels.ops.kvcache.hicache import can_use_write_back_jit_kernel

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/ops/kvcache/hicache.py": (
        "can_use_write_back_jit_kernel",
        "transfer_hicache_all_layer_mla_staged_lf_pf",
        "transfer_hicache_one_layer_mla",
    ),
    "srt/mem_cache/pool_host/common.py": (
        "ALLOC_MEMORY_FUNCS",
        "_cuda_host_unregister",
    ),
    "srt/mem_cache/pool_host/mha.py": ("MHATokenToKVPoolHost", "_is_cuda", "_is_hip"),
}

jit_transfer_hicache_all_layer_mla_staged_lf_pf = (
    _sgl_kernels_ops_kvcache_hicache.transfer_hicache_all_layer_mla_staged_lf_pf
)
import sglang.kernels.ops.kvcache.hicache as _sgl_kernels_ops_kvcache_hicache

jit_transfer_hicache_one_layer_mla = (
    _sgl_kernels_ops_kvcache_hicache.transfer_hicache_one_layer_mla
)
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    _cuda_host_unregister,
)
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost, _is_cuda, _is_hip


def is_native_nvfp4_pool(device_pool) -> bool:
    """A device MHA pool that stores packed E2M1 data with trtllm-gen's native scale tiles."""
    qm = getattr(device_pool, "quant_method", None)
    return (
        getattr(qm, "name", None) == "nvfp4"
        and getattr(device_pool, "native_k_scale_buffer", None) is not None
        and getattr(device_pool, "native_v_scale_buffer", None) is not None
    )


class PackedRowView:
    """The geometry ``prepare_mha_write_back_staging`` reads, with K/V rows in packed bytes (head_dim / 2)."""

    def __init__(self, pool):
        object.__setattr__(self, "_pool", pool)

    @property
    def head_num(self):
        return self._pool.head_num

    @property
    def head_dim(self):
        return self._pool.head_dim // 2

    @property
    def v_head_dim(self):
        return self._pool.v_head_dim // 2

    @property
    def row_dim(self):
        return self._pool.head_num * (self._pool.head_dim // 2)

    @property
    def store_dtype(self):
        return torch.uint8

    @property
    def device(self):
        return self._pool.device

    @property
    def hicache_write_back_staging(self):
        return self._pool.hicache_write_back_staging

    @hicache_write_back_staging.setter
    def hicache_write_back_staging(self, value):
        self._pool.hicache_write_back_staging = value

    def __setattr__(self, name, value):
        if name == "hicache_write_back_staging":
            self._pool.hicache_write_back_staging = value
        else:
            raise AttributeError(f"PackedRowView is read-only ({name})")


class MHATokenToKVPoolNVFP4Host(MHATokenToKVPoolHost):
    def __init__(
        self,
        device_pool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
        *,
        mtp_draft_device_pools: Sequence = (),
        pool_label: str = "kv",
    ):
        if layout != "page_first":
            raise NotImplementedError(
                f"NVFP4 KV host pool supports only the page_first layout, got {layout!r}."
            )
        if mtp_draft_device_pools:
            raise NotImplementedError(
                "NVFP4 KV host pool does not pack MTP draft KV layers."
            )
        if not is_native_nvfp4_pool(device_pool):
            raise NotImplementedError(
                "NVFP4 KV host pool needs a native NVFP4 device pool."
            )
        if getattr(device_pool, "k_scale_buffer", None) is not None:
            raise NotImplementedError(
                "NVFP4 KV host pool does not carry the linear (FP8-prefill) scale layout."
            )
        if device_pool.head_dim != device_pool.v_head_dim or device_pool.head_dim % 32:
            raise NotImplementedError(
                "NVFP4 KV host pool needs equal K/V head dims divisible by 32."
            )
        # Bytes of native scales one page holds per layer ([head, page_size, head_dim / 16]).
        self.k_sf_page_bytes = device_pool.native_k_scale_buffer[0][0].numel()
        self.v_sf_page_bytes = device_pool.native_v_scale_buffer[0][0].numel()
        self.k_scale_host: torch.Tensor | None = None
        self.v_scale_host: torch.Tensor | None = None
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
            allocator_type,
            pool_label=pool_label,
        )
        if self.page_size != device_pool.page_size:
            raise ValueError(
                "NVFP4 KV host pool moves scales per page, so the host page size "
                f"({self.page_size}) must equal the device page size ({device_pool.page_size})."
            )
        if self.element_dim != device_pool.k_buffer[0][0].numel():
            raise RuntimeError(
                f"NVFP4 KV host pool row of {self.element_dim} B does not match the device row of "
                f"{device_pool.k_buffer[0][0].numel()} B."
            )
        if not self.can_use_write_back_jit or not all(
            can_use_write_back_jit_kernel(element_size=size)
            for size in (self.k_sf_page_bytes, self.v_sf_page_bytes)
        ):
            raise NotImplementedError(
                "NVFP4 KV host pool needs the staged JIT write-back kernel (io_backend='kernel', page_first layout)."
            )
        self._init_scale_buffers()

    def get_size_per_token(self):
        # Payload rows in packed bytes: head_dim / 2 per head (two E2M1 values per byte, uint8 storage).
        self.head_num = self.device_pool.head_num
        self.head_dim = self.device_pool.head_dim // 2
        self.layer_num = self.target_layer_num
        payload = (
            self.head_dim * self.head_num * self.layer_num * self.dtype.itemsize * 2
        )
        scales = (self.k_sf_page_bytes + self.v_sf_page_bytes) // self.page_size
        return payload + scales * self.layer_num

    def _init_scale_buffers(self):
        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        # Host: page-first like the payload, one contiguous scale tile per (page, layer).
        self.k_scale_host = alloc_func(
            (self.page_num, self.layer_num, self.k_sf_page_bytes),
            dtype=torch.uint8,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
        )
        self.v_scale_host = alloc_func(
            (self.page_num, self.layer_num, self.v_sf_page_bytes),
            dtype=torch.uint8,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
        )
        self.k_scale_host_layers = list(self.k_scale_host.transpose(0, 1))
        self.v_scale_host_layers = list(self.v_scale_host.transpose(0, 1))
        # Device: each layer's native scale tensor [page, head, page_size, sf] as [page, bytes] rows.
        self.k_scale_device_rows = [
            buf.view(torch.uint8).reshape(buf.shape[0], -1)
            for buf in self.device_pool.native_k_scale_buffer
        ]
        self.v_scale_device_rows = [
            buf.view(torch.uint8).reshape(buf.shape[0], -1)
            for buf in self.device_pool.native_v_scale_buffer
        ]
        self.k_scale_device_ptrs = torch.tensor(
            [rows.data_ptr() for rows in self.k_scale_device_rows],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        self.v_scale_device_ptrs = torch.tensor(
            [rows.data_ptr() for rows in self.v_scale_device_rows],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        self.k_scale_staging = torch.empty(
            (self.staging_page_capacity, self.layer_num, self.k_sf_page_bytes),
            dtype=torch.uint8,
            device=self.device_pool.device,
        )
        self.v_scale_staging = torch.empty(
            (self.staging_page_capacity, self.layer_num, self.v_sf_page_bytes),
            dtype=torch.uint8,
            device=self.device_pool.device,
        )

    def _page_ids(self, indices: torch.Tensor) -> torch.Tensor:
        """Page-aligned token indices -> one page id per page, same device."""
        return indices[:: self.page_size] // self.page_size

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if io_backend != "kernel":
            raise NotImplementedError(
                f"NVFP4 KV host pool supports only io_backend='kernel', got {io_backend!r}."
            )
        super().backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend
        )
        device_pages = self._page_ids(device_indices)
        host_pages = self._page_ids(host_indices)
        if host_pages.is_cuda:
            host_pages = host_pages.cpu()
        for ptr_src, staging, dst in (
            (self.k_scale_device_ptrs, self.k_scale_staging, self.k_scale_host),
            (self.v_scale_device_ptrs, self.v_scale_staging, self.v_scale_host),
        ):
            jit_transfer_hicache_all_layer_mla_staged_lf_pf(
                ptr_src=ptr_src,
                src_indices=device_pages,
                dst_indices=host_pages,
                staging=staging,
                dst=dst,
                page_size=1,
            )

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        if is_draft:
            raise NotImplementedError("NVFP4 KV host pool has no draft layers.")
        if io_backend != "kernel":
            raise NotImplementedError(
                f"NVFP4 KV host pool supports only io_backend='kernel', got {io_backend!r}."
            )
        super().load_to_device_per_layer(
            device_pool,
            host_indices,
            device_indices,
            layer_id,
            io_backend,
            is_draft=is_draft,
        )
        if not self._is_device_layer_owned(device_pool, layer_id):
            return
        host_layer_id = self._host_layer_index(layer_id)
        device_pages = self._page_ids(device_indices)
        host_pages = self._page_ids(host_indices)
        for dst_rows, src_rows, sf_page_bytes in (
            (
                self.k_scale_device_rows[layer_id],
                self.k_scale_host_layers[host_layer_id],
                self.k_sf_page_bytes,
            ),
            (
                self.v_scale_device_rows[layer_id],
                self.v_scale_host_layers[host_layer_id],
                self.v_sf_page_bytes,
            ),
        ):
            jit_transfer_hicache_one_layer_mla(
                cache_dst=dst_rows,
                indices_dst=device_pages,
                cache_src=src_rows,
                indices_src=host_pages,
                element_dim=sf_page_bytes,
            )

    def destroy(self):
        for buf in (self.k_scale_host, self.v_scale_host):
            if buf is not None and self.pin_memory and (_is_cuda or _is_hip):
                _cuda_host_unregister(buf)
        self.k_scale_host = None
        self.v_scale_host = None
        super().destroy()

    def _storage_pages_unsupported(self) -> NotImplementedError:
        return NotImplementedError(
            "NVFP4 KV host pool does not expose flat storage (L3) pages: a page's block scales live outside kv_buffer."
        )

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        raise self._storage_pages_unsupported()

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        raise self._storage_pages_unsupported()

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        raise self._storage_pages_unsupported()

    def get_page_buffer_meta(self, indices):
        raise self._storage_pages_unsupported()

    def get_split_heads_page_buffer_meta(
        self, indices: torch.Tensor, split_factor: int
    ):
        raise self._storage_pages_unsupported()


# ------------------------------------------------------------------------------------- SGLang main's host-pool API
# On SGLang main, host pools are built from the device pool's buffer description (MHABufferInfo) instead of the
# device pool itself (pool_host/mha.py, pool_host/mha_mxfp8.py). The description of a native NVFP4 pool is made to
# describe packed rows (packed_buffer_info, installed on MHATokenToKVPool._mha_buffer_info), so the stock write-back
# staging and the payload transfer see head_dim / 2 bytes per head; the native scale tiles, which have no stock scale
# layout, reach the host pool as keyword arguments from the assembler hook (nvfp4_kv._host_inputs_around).


def packed_buffer_info(pool, info):
    """``info`` (stock description of ``pool``) with packed-row geometry: head_dim / 2 bytes per head, uint8."""
    import msgspec

    buffers = info.buffers
    if any(t.dtype != torch.uint8 for t in (*buffers.k, *buffers.v)):
        buffers = msgspec.structs.replace(
            buffers,
            k=tuple(t.view(torch.uint8) for t in buffers.k),
            v=tuple(t.view(torch.uint8) for t in buffers.v),
        )
    return msgspec.structs.replace(
        info,
        buffers=buffers,
        head_num=pool.head_num,
        head_dim=pool.head_dim // 2,
        v_head_dim=pool.v_head_dim // 2,
        dtype=torch.uint8,
        scales=None,
    )


class MHATokenToKVPoolNVFP4HostMain(MHATokenToKVPoolHost):
    """MHATokenToKVPoolNVFP4Host for SGLang main's constructor (buffer description + keyword geometry)."""

    def __init__(
        self,
        buffer_info,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
        *,
        device_capacity_tokens: int,
        start_layer: int,
        end_layer: int,
        draft_buffer_infos: Sequence = (),
        write_back_staging=None,
        pool_label: str = "kv",
        native_k_scales: Sequence[torch.Tensor] = (),
        native_v_scales: Sequence[torch.Tensor] = (),
    ):
        if layout != "page_first":
            raise NotImplementedError(
                f"NVFP4 KV host pool supports only the page_first layout, got {layout!r}."
            )
        if draft_buffer_infos:
            raise NotImplementedError(
                "NVFP4 KV host pool does not pack MTP draft KV layers."
            )
        if (
            not native_k_scales
            or len(native_k_scales) != len(native_v_scales)
            or (len(native_k_scales) != len(buffer_info.buffers.k))
        ):
            raise NotImplementedError(
                "NVFP4 KV host pool needs the native scale tiles of every device layer."
            )
        if (
            buffer_info.dtype != torch.uint8
            or buffer_info.head_dim != buffer_info.v_head_dim
            or ((2 * buffer_info.head_dim) % 32)
        ):
            raise NotImplementedError(
                "NVFP4 KV host pool needs packed uint8 rows with equal K/V head dims divisible by 32."
            )
        # Bytes of native scales one page holds per layer ([head, page_size, head_dim / 16]); read by
        # get_size_per_token during the base initialization.
        self.k_sf_page_bytes = native_k_scales[0][0].numel()
        self.v_sf_page_bytes = native_v_scales[0][0].numel()
        self.k_scale_host: torch.Tensor | None = None
        self.v_scale_host: torch.Tensor | None = None
        super().__init__(
            buffer_info,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
            allocator_type,
            device_capacity_tokens=device_capacity_tokens,
            start_layer=start_layer,
            end_layer=end_layer,
            write_back_staging=write_back_staging,
            pool_label=pool_label,
        )
        if self.page_size != buffer_info.page_size:
            raise ValueError(
                "NVFP4 KV host pool moves scales per page, so the host page size "
                f"({self.page_size}) must equal the device page size ({buffer_info.page_size})."
            )
        row = (
            buffer_info.buffers.k[0][0].numel()
            * buffer_info.buffers.k[0].element_size()
        )
        if self.element_dim != row:
            raise RuntimeError(
                f"NVFP4 KV host pool row of {self.element_dim} B does not match the device row of {row} B."
            )
        if not self.can_use_write_back_jit or not all(
            can_use_write_back_jit_kernel(element_size=size)
            for size in (self.k_sf_page_bytes, self.v_sf_page_bytes)
        ):
            raise NotImplementedError(
                "NVFP4 KV host pool needs the staged JIT write-back kernel (io_backend='kernel', page_first layout)."
            )
        self._init_scale_buffers(native_k_scales, native_v_scales)

    def get_size_per_token(self):
        payload = (
            super().get_size_per_token()
        )  # packed rows: head_dim / 2 bytes per head
        scales = (self.k_sf_page_bytes + self.v_sf_page_bytes) // self.page_size
        return payload + scales * self.layer_num

    def _init_scale_buffers(self, k_scales, v_scales):
        alloc_func = ALLOC_MEMORY_FUNCS[self._transfer_device.type]
        # Host: page-first like the payload, one contiguous scale tile per (page, layer).
        self.k_scale_host = alloc_func(
            (self.page_num, self.layer_num, self.k_sf_page_bytes),
            dtype=torch.uint8,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
        )
        self.v_scale_host = alloc_func(
            (self.page_num, self.layer_num, self.v_sf_page_bytes),
            dtype=torch.uint8,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
        )
        self.k_scale_host_layers = list(self.k_scale_host.transpose(0, 1))
        self.v_scale_host_layers = list(self.v_scale_host.transpose(0, 1))
        # Device: each layer's native scale tensor [page, head, page_size, sf] as [page, bytes] rows.
        self.k_scale_device_rows = [
            buf.view(torch.uint8).reshape(buf.shape[0], -1) for buf in k_scales
        ]
        self.v_scale_device_rows = [
            buf.view(torch.uint8).reshape(buf.shape[0], -1) for buf in v_scales
        ]
        self.k_scale_device_ptrs = torch.tensor(
            [rows.data_ptr() for rows in self.k_scale_device_rows],
            dtype=torch.uint64,
            device=self._transfer_device,
        )
        self.v_scale_device_ptrs = torch.tensor(
            [rows.data_ptr() for rows in self.v_scale_device_rows],
            dtype=torch.uint64,
            device=self._transfer_device,
        )
        self.k_scale_staging = torch.empty(
            (self.staging_page_capacity, self.layer_num, self.k_sf_page_bytes),
            dtype=torch.uint8,
            device=self._transfer_device,
        )
        self.v_scale_staging = torch.empty(
            (self.staging_page_capacity, self.layer_num, self.v_sf_page_bytes),
            dtype=torch.uint8,
            device=self._transfer_device,
        )

    def _page_ids(self, indices: torch.Tensor) -> torch.Tensor:
        """Page-aligned token indices -> one page id per page, same device."""
        return indices[:: self.page_size] // self.page_size

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if io_backend != "kernel":
            raise NotImplementedError(
                f"NVFP4 KV host pool supports only io_backend='kernel', got {io_backend!r}."
            )
        super().backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend
        )
        device_pages = self._page_ids(device_indices)
        host_pages = self._page_ids(host_indices)
        if host_pages.is_cuda:
            host_pages = host_pages.cpu()
        for ptr_src, staging, dst in (
            (self.k_scale_device_ptrs, self.k_scale_staging, self.k_scale_host),
            (self.v_scale_device_ptrs, self.v_scale_staging, self.v_scale_host),
        ):
            jit_transfer_hicache_all_layer_mla_staged_lf_pf(
                ptr_src=ptr_src,
                src_indices=device_pages,
                dst_indices=host_pages,
                staging=staging,
                dst=dst,
                page_size=1,
            )

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        if is_draft:
            raise NotImplementedError("NVFP4 KV host pool has no draft layers.")
        if io_backend != "kernel":
            raise NotImplementedError(
                f"NVFP4 KV host pool supports only io_backend='kernel', got {io_backend!r}."
            )
        super().load_to_device_per_layer(
            device_pool,
            host_indices,
            device_indices,
            layer_id,
            io_backend,
            is_draft=is_draft,
        )
        if not 0 <= layer_id < self.target_layer_num:
            return
        device_pages = self._page_ids(device_indices)
        host_pages = self._page_ids(host_indices)
        for dst_rows, src_rows, sf_page_bytes in (
            (
                self.k_scale_device_rows[layer_id],
                self.k_scale_host_layers[layer_id],
                self.k_sf_page_bytes,
            ),
            (
                self.v_scale_device_rows[layer_id],
                self.v_scale_host_layers[layer_id],
                self.v_sf_page_bytes,
            ),
        ):
            jit_transfer_hicache_one_layer_mla(
                cache_dst=dst_rows,
                indices_dst=device_pages,
                cache_src=src_rows,
                indices_src=host_pages,
                element_dim=sf_page_bytes,
            )

    def destroy(self):
        for buf in (self.k_scale_host, self.v_scale_host):
            if buf is not None and self.pin_memory and (_is_cuda or _is_hip):
                _cuda_host_unregister(buf)
        self.k_scale_host = None
        self.v_scale_host = None
        super().destroy()

    def _storage_pages_unsupported(self) -> NotImplementedError:
        return NotImplementedError(
            "NVFP4 KV host pool does not expose flat storage (L3) pages: a page's block scales live outside kv_buffer."
        )

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        raise self._storage_pages_unsupported()

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        raise self._storage_pages_unsupported()

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        raise self._storage_pages_unsupported()

    def get_page_buffer_meta(self, indices):
        raise self._storage_pages_unsupported()

    def get_split_heads_page_buffer_meta(
        self, indices: torch.Tensor, split_factor: int
    ):
        raise self._storage_pages_unsupported()
