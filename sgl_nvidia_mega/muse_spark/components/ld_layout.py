# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Locality-domain helpers: per-domain device memory through the CUDA virtual memory management (VMM) API, and the
SM -> locality domain table.

A GPU that reports more than one locality domain (``CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT``) can back device
memory on one domain (``CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN``) and run work on that domain's SMs (green
contexts from an SM resource split by domain), see the CUDA Programming Guide section on locality domains.
``cudaMalloc`` memory (and so every PyTorch allocation) is not placed on a domain.

``alloc_pattern(device, chunk_bytes, domains)``
    One contiguous virtual address range made of ``len(domains)`` chunks; chunk ``i`` is backed by its own physical
    allocation on domain ``domains[i]`` (``-1``: no domain). Returned as a 1-D uint8 tensor that owns the mapping: the
    range is unmapped when the last tensor viewing it is freed. ``cuMemMap`` maps whole physical allocations only, so
    every chunk is its own handle; ``chunk_bytes`` must be a multiple of the allocation granularity.
    Callers should place the same number of chunks on each domain.
``split_rows_copy(src, parts)``
    A copy of a contiguous ``[E, R, C]`` tensor whose leading-dimension block ``e`` is cut into ``parts`` equal row
    ranges, range ``p`` placed on domain ``p``. Same shape, dtype, strides and bytes as ``src``: only the physical
    placement differs, so every kernel reading it computes the same values.
``sm_domain_table(device)``
    int8 tensor ``[multiprocessor_count]``: the domain of each SM. The SM resource is split by domain (green contexts,
    one probe kernel per domain records ``%smid``); SMs outside both domains' splits are attributed by a second split
    that asks each domain for half of them with the backfill flag.
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache

import torch

logger = logging.getLogger("muse_vr_plugin")

_lock = threading.Lock()


def _cu():
    from cuda.bindings import driver as cu

    return cu


def _check(res, what: str):
    cu = _cu()
    err = res[0] if isinstance(res, tuple) else res
    if err != cu.CUresult.CUDA_SUCCESS:
        _, name = cu.cuGetErrorName(err)
        raise RuntimeError(
            f"{what} failed: {name.decode() if isinstance(name, bytes) else name}"
        )
    if isinstance(res, tuple):
        return res[1] if len(res) == 2 else res[1:]
    return None


def _device_ordinal(device) -> int:
    d = (
        torch.device(device)
        if not isinstance(device, int)
        else torch.device("cuda", device)
    )
    return d.index if d.index is not None else torch.cuda.current_device()


def domain_count(device=None) -> int:
    """Locality domains the device reports (1 when the attribute is unsupported)."""
    cu = _cu()
    try:
        _check(cu.cuInit(0), "cuInit")
        dev = _check(cu.cuDeviceGet(_device_ordinal(device)), "cuDeviceGet")
        return int(
            _check(
                cu.cuDeviceGetAttribute(
                    cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, dev
                ),
                "cuDeviceGetAttribute",
            )
        )
    except (RuntimeError, AttributeError):
        return 1


def _prop(ordinal: int, domain: int):
    cu = _cu()
    p = cu.CUmemAllocationProp()
    p.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    if domain < 0:
        p.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        p.location.id = ordinal
    else:
        p.location.type = (
            cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN
        )
        p.location.localized.deviceId = ordinal
        p.location.localized.localityDomainId = domain
    return p


def granularity(device=None, domain: int = 0) -> int:
    cu = _cu()
    g = _check(
        cu.cuMemGetAllocationGranularity(
            _prop(_device_ordinal(device), domain),
            cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
        ),
        "cuMemGetAllocationGranularity",
    )
    return int(g)


def pointer_domain(ptr: int) -> int:
    """Locality domain ordinal of the memory at ``ptr`` (-1: not placed on a domain)."""
    cu = _cu()
    v = _check(
        cu.cuPointerGetAttribute(
            cu.CUpointer_attribute.CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL, ptr
        ),
        "cuPointerGetAttribute",
    )
    return int(v)


PLACE_COUNTS = {
    "allocs": 0,
    "mapped": 0,
}  # cuMemCreate calls / mapped bytes made by _Mapping (diagnostics; MUSE_VR_MOE_LD_PLACEDIAG reads them)
PLACED: dict[
    int, int
] = {}  # base address -> size of every live _Mapping (is_placed reads it)
LAYOUT: dict[
    int, tuple
] = {}  # base address -> (chunk bytes, domain order) of every live _Mapping (units compare layouts)


class _Mapping:
    """A mapped VA range; exposes ``__cuda_array_interface__`` so PyTorch tensors keep it alive."""

    def __init__(self, ordinal: int, chunk: int, domains: list[int]):
        cu = _cu()
        self.size = chunk * len(domains)
        self.va = 0
        self._mapped = 0
        gran = max(granularity(ordinal, 0), granularity(ordinal, -1))
        if chunk % gran:
            raise ValueError(
                f"chunk {chunk} is not a multiple of the allocation granularity {gran}"
            )
        self.va = int(
            _check(cu.cuMemAddressReserve(self.size, gran, 0, 0), "cuMemAddressReserve")
        )
        try:
            for i, d in enumerate(domains):
                h = _check(cu.cuMemCreate(chunk, _prop(ordinal, d), 0), "cuMemCreate")
                try:
                    _check(cu.cuMemMap(self.va + i * chunk, chunk, 0, h, 0), "cuMemMap")
                finally:
                    _check(
                        cu.cuMemRelease(h), "cuMemRelease"
                    )  # the mapping keeps the allocation alive
                self._mapped = (i + 1) * chunk
            acc = cu.CUmemAccessDesc()
            acc.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            acc.location.id = ordinal
            acc.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            _check(cu.cuMemSetAccess(self.va, self.size, [acc], 1), "cuMemSetAccess")
        except Exception:
            self._free()
            raise
        self.__cuda_array_interface__ = {
            "shape": (self.size,),
            "typestr": "|u1",
            "data": (self.va, False),
            "version": 3,
            "strides": None,
        }
        PLACE_COUNTS["allocs"] += len(domains)
        PLACE_COUNTS["mapped"] += self.size
        PLACED[self.va] = self.size
        LAYOUT[self.va] = (int(chunk), tuple(int(d) for d in domains))

    def _free(self):
        if not self.va:
            return
        PLACED.pop(self.va, None)
        LAYOUT.pop(self.va, None)
        try:
            cu = _cu()
            if self._mapped:
                cu.cuMemUnmap(self.va, self._mapped)
            cu.cuMemAddressFree(self.va, self.size)
        except Exception:  # interpreter shutdown
            pass
        self.va = 0

    def __del__(self):
        self._free()


def alloc_pattern(device, chunk_bytes: int, domains: list[int]) -> torch.Tensor:
    ordinal = _device_ordinal(device)
    m = _Mapping(ordinal, int(chunk_bytes), [int(d) for d in domains])
    return torch.as_tensor(m, device=torch.device("cuda", ordinal))


def alloc_rows_split(device, shape, dtype, parts: int = 2) -> torch.Tensor:
    """Uninitialized contiguous [E, R, ...] tensor in split_rows_copy's layout: rows [p*R/parts, (p+1)*R/parts) of every
    block e on domain p (the same chunks, the same domain order)."""
    shape = tuple(int(x) for x in shape)
    if len(shape) < 2:
        raise ValueError("alloc_rows_split needs at least 2 dimensions")
    e, r = shape[0], shape[1]
    if r % parts:
        raise ValueError(f"rows {r} not divisible by {parts}")
    nbytes = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
    chunk = nbytes // (e * parts)
    raw = alloc_pattern(device, chunk, [p for _ in range(e) for p in range(parts)])
    return raw.view(dtype).view(shape)


def split_rows_copy(src: torch.Tensor, parts: int = 2) -> torch.Tensor:
    """Copy of a contiguous [E, R, ...] tensor; rows [p*R/parts, (p+1)*R/parts) of every block e on domain p."""
    if not src.is_cuda or not src.is_contiguous() or src.dim() < 2:
        raise ValueError(
            "split_rows_copy needs a contiguous CUDA tensor with at least 2 dimensions"
        )
    out = alloc_rows_split(src.device, src.shape, src.dtype, parts)
    out.copy_(src)
    return out


def is_placed(t: torch.Tensor) -> bool:
    """True when t is a whole live placement mapping (alloc_rows_split / split_rows_copy output, not a view of part of one)."""
    try:
        return (
            t.is_cuda
            and t.is_contiguous()
            and PLACED.get(int(t.data_ptr())) == t.numel() * t.element_size()
        )
    except Exception:  # noqa: BLE001 - a tensor without storage is not placed
        return False


@lru_cache(maxsize=None)
def _smid_probe_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _probe(out_ptr, naps: tl.constexpr):
        pid = tl.program_id(0)
        smid = tl.inline_asm_elementwise(
            "mov.u32 $0, %smid;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
        )
        # stay resident for a while so that the programs spread over all SMs of the context
        for _ in range(naps):
            tl.inline_asm_elementwise(
                "nanosleep.u32 1000; mov.u32 $0, 0;",
                "=r",
                [],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
        tl.store(out_ptr + pid, smid)

    return _probe


_tables: dict[int, torch.Tensor] = {}


def _probe_split(cu, dev, ordinal: int, nsm: int, extra: int) -> list[int]:
    """SM -> domain from one split of the SM resource by domain (``extra`` > 0: backfill to ``per-domain + extra``)."""
    table = [-1] * nsm
    res_all = _check(
        cu.cuDeviceGetDevResource(dev, cu.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM),
        "cuDeviceGetDevResource",
    )
    per_domain = int(
        _check(
            cu.cuDeviceGetAttribute(
                cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_MULTIPROCESSOR_COUNT,
                dev,
            ),
            "cuDeviceGetAttribute",
        )
    )
    groups = []
    for d in range(2):
        g = cu.CU_DEV_SM_RESOURCE_GROUP_PARAMS()
        g.flags = (
            cu.CUdevSmResourceGroup_flags.CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID
        )
        if extra > 0:
            g.flags = (
                g.flags
                | cu.CUdevSmResourceGroup_flags.CU_DEV_SM_RESOURCE_GROUP_BACKFILL
            )
            g.smCount = per_domain + extra
        g.localityDomainId = d
        groups.append(g)
    out = _check(cu.cuDevSmResourceSplit(2, res_all, 0, groups), "cuDevSmResourceSplit")
    parts = out[0] if isinstance(out, (tuple, list)) else out
    probe = _smid_probe_kernel()
    n_prog = 8 * nsm
    buf = torch.full(
        (n_prog,), -1, dtype=torch.int32, device=torch.device("cuda", ordinal)
    )
    for d in range(2):
        desc = _check(
            cu.cuDevResourceGenerateDesc([parts[d]], 1), "cuDevResourceGenerateDesc"
        )
        gctx = _check(
            cu.cuGreenCtxCreate(
                desc, dev, cu.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM
            ),
            "cuGreenCtxCreate",
        )
        try:
            stream = _check(
                cu.cuGreenCtxStreamCreate(
                    gctx, cu.CUstream_flags.CU_STREAM_NON_BLOCKING, 0
                ),
                "cuGreenCtxStreamCreate",
            )
            try:
                ext = torch.cuda.ExternalStream(
                    int(stream), device=torch.device("cuda", ordinal)
                )
                buf.fill_(-1)
                torch.cuda.synchronize(ordinal)
                with torch.cuda.stream(ext):
                    probe[(n_prog,)](buf, naps=20)
                ext.synchronize()
                for s in set(buf.tolist()):
                    if 0 <= s < nsm:
                        table[s] = d
            finally:
                cu.cuStreamDestroy(stream)
        finally:
            cu.cuGreenCtxDestroy(gctx)
    return table


def sm_domain_table(device=None) -> torch.Tensor:
    """int8 [multiprocessor_count] SM -> locality domain table (cached per device)."""
    ordinal = _device_ordinal(device)
    with _lock:
        if ordinal in _tables:
            return _tables[ordinal]
    cu = _cu()
    nsm = torch.cuda.get_device_properties(ordinal).multi_processor_count
    nld = domain_count(ordinal)
    table = [0] * nsm
    n_split = n_backfill = 0
    if nld >= 2:
        dev = _check(cu.cuDeviceGet(ordinal), "cuDeviceGet")
        torch.cuda.synchronize(ordinal)
        table = _probe_split(cu, dev, ordinal, nsm, 0)
        rest = [s for s in range(nsm) if table[s] < 0]
        n_split = nsm - len(rest)
        if rest:
            back = _probe_split(cu, dev, ordinal, nsm, (len(rest) + 1) // 2)
            for s in rest:
                if back[s] >= 0:
                    table[s] = back[s]
                    n_backfill += 1
        table = [max(t, 0) for t in table]
    t = torch.tensor(table, dtype=torch.int8, device=torch.device("cuda", ordinal))
    with _lock:
        _tables[ordinal] = t
    logger.warning(
        "muse_vr_plugin: sm_domain_table device %d domains %d: %d SMs from the domain split, %d by backfill, "
        "%d defaulted to 0",
        ordinal,
        nld,
        n_split,
        n_backfill,
        nsm - n_split - n_backfill,
    )
    return t
