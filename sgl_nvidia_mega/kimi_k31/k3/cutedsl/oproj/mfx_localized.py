# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Symmetric buffers whose local bytes sit on chosen locality domains (default off).

Port of an earlier ``oproj/localized.py`` (mechanism verified on the target GPU:
localized FABRIC allocations export, import, map and bind to a multicast object; stores land bitwise) without its
K31MK_MAILBOX_HOME / K31MK_OPROJ_HOME readers. ``LocalizedSymm`` does what a symmetric-memory rendezvous does, with the
location type changed (CUDA PG 4.17): every rank exports its allocation(s) as FABRIC handles (POSIX fds when every rank
shares a host and IMEX is missing), imports and maps every peer's (``buffer_ptrs``); rank 0 creates a multicast object that
every rank joins, binds its own segments to and maps (``multicast_ptr``)."""

import ctypes
import functools
import os
import socket

import torch
import torch.distributed as dist

ENV = "K31MK_OPROJ_HHOME"


def _driver():
    from cuda.bindings import driver

    return driver


def _ok(result, what):
    status, *values = result if isinstance(result, tuple) else (result,)
    if int(status) != 0:
        raise RuntimeError(f"{what} failed: {status!r}")
    if not values:
        return None
    return values[0] if len(values) == 1 else tuple(values)


@functools.cache
def _libcuda():
    return ctypes.CDLL("libcuda.so.1")


def pointer_ld(pointer):
    """CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL (22) of ``pointer``: the locality domain, or -1 when not localized."""
    value = ctypes.c_int(-99)
    status = _libcuda().cuPointerGetAttribute(
        ctypes.byref(value), ctypes.c_int(22), ctypes.c_uint64(int(pointer))
    )
    if status != 0:
        raise RuntimeError(f"cuPointerGetAttribute(LOCALITY_DOMAIN_ORDINAL) failed: {status}")
    return value.value


def localized_prop(device, ld):
    """Pinned device memory on locality domain ``ld`` of ``device``, exportable as a FABRIC handle."""
    cu = _driver()
    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN
    prop.location.localized.deviceId = device
    prop.location.localized.localityDomainId = ld
    # CUmemLocation = {int type; union {int id; struct {uchar deviceId, localityDomainId;} localized;}}
    raw = (ctypes.c_uint8 * 2).from_address(int(prop.location.getPtr()) + 4)
    if (raw[0], raw[1]) != (device, ld):
        raise RuntimeError(f"localized location {device}/{ld} not set: {tuple(raw)}")
    prop.requestedHandleTypes = cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
    return prop


def _round(value, multiple):
    return -(-value // multiple) * multiple


def _kind_type(kind):
    cu = _driver()
    kinds = cu.CUmemAllocationHandleType
    return kinds.CU_MEM_HANDLE_TYPE_FABRIC if kind == "fabric" else kinds.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR


def _fabric_ok(device, ld):
    """Whether a localized FABRIC allocation can be created and exported here (needs an IMEX channel)."""
    cu = _driver()
    prop = localized_prop(device, ld)
    flag = cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
    status, gran = cu.cuMemGetAllocationGranularity(prop, flag)
    if int(status) != 0:
        return False
    status, handle = cu.cuMemCreate(int(gran), prop, 0)
    if int(status) != 0:
        return False
    status, _ = cu.cuMemExportToShareableHandle(handle, _kind_type("fabric"), 0)
    cu.cuMemRelease(handle)
    return int(status) == 0


def _pidfd_getfd(pid, fd):
    libc = ctypes.CDLL(None, use_errno=True)
    pidfd = os.pidfd_open(pid)
    try:
        got = libc.syscall(438, pidfd, fd, 0)  # SYS_pidfd_getfd
        if got < 0:
            raise OSError(ctypes.get_errno(), f"pidfd_getfd({pid}, {fd})")
        return got
    finally:
        os.close(pidfd)


class _Bytes:
    """``__cuda_array_interface__`` of a byte range; the torch view keeps its owner alive."""

    def __init__(self, owner, pointer, nbytes):
        self.owner, self.pointer, self.nbytes = owner, pointer, nbytes

    @property
    def __cuda_array_interface__(self):
        return {
            "shape": (self.nbytes,),
            "typestr": "|u1",
            "data": (self.pointer, False),
            "version": 3,
            "strides": None,
            "stream": None,
        }


class LocalizedSymm:
    """A symmetric buffer of ``nbytes`` over ``group`` whose local memory is on locality domain ``ld``, or, with ``segments``
    [(ld, nbytes), ...], consecutive segments each on its own locality domain (sizes multiples of the granularity). Collective:
    every rank of ``group`` constructs it with the same sizes. ``buffer_ptrs[r]``: rank r's buffer mapped here;
    ``multicast_ptr``: the multicast mapping (a store there lands in every rank's buffer)."""

    def __init__(self, group, device, nbytes, ld, segments=None):
        cu = _driver()
        index = torch.device(device).index
        if index is None:
            index = torch.cuda.current_device()
        self.device, self.ld = index, ld
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)
        self._maps, self._imports, self._handle, self._mc, self._fds = [], [], None, None, []
        self._reserved, self._handles = [], []
        with torch.cuda.device(index):
            facts = [None] * self.world
            probe_ld = ld if segments is None else segments[0][0]
            dist.all_gather_object(facts, (_fabric_ok(index, probe_ld), socket.gethostname()), group=group)
            if all(ok for ok, _ in facts):
                self.kind = "fabric"
            elif len({host for _, host in facts}) == 1:
                self.kind = "fd"
            else:
                raise RuntimeError(f"{ENV}: FABRIC handles unavailable across hosts: {facts}")
            kind = _kind_type(self.kind)
            lds = sorted({ld} if segments is None else {s_ld for s_ld, _ in segments})
            props = {}
            for s_ld in lds:
                props[s_ld] = localized_prop(index, s_ld)
                props[s_ld].requestedHandleTypes = kind
            recommended = cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
            gran = max(
                int(_ok(cu.cuMemGetAllocationGranularity(props[s_ld], recommended), "allocation granularity"))
                for s_ld in lds
            )
            mc_prop = cu.CUmulticastObjectProp()
            mc_prop.numDevices = self.world
            mc_prop.handleTypes = int(kind)
            mc_prop.flags = 0
            mc_prop.size = _round(nbytes, gran)
            # MINIMUM multicast granularity: with FABRIC handles the RECOMMENDED one is
            # much larger (cuMulticastGetGranularity), which would inflate every segment; bind
            # offsets and sizes need only the minimum (CUDA driver API cuMulticastBindMem). Both are recorded on the object.
            grans = cu.CUmulticastGranularity_flags
            self.mc_gran_recommended = int(_ok(cu.cuMulticastGetGranularity(mc_prop, grans.CU_MULTICAST_GRANULARITY_RECOMMENDED),
                                               "multicast granularity"))
            mc_gran = int(_ok(cu.cuMulticastGetGranularity(mc_prop, grans.CU_MULTICAST_GRANULARITY_MINIMUM), "multicast granularity"))
            self.mc_gran_minimum, self.alloc_gran = mc_gran, int(gran)
            self.align = max(int(gran), int(mc_gran))  # both powers of two
            if segments is None:
                segments = [(ld, _round(nbytes, self.align))]
            if any(n <= 0 or n % self.align for _, n in segments):
                raise ValueError(f"segment sizes must be positive multiples of {self.align}: {segments}")
            self.segments = list(segments)
            self.size = size = sum(n for _, n in segments)
            mc_prop.size = size
            try:
                self._handles = []
                for s_ld, n in segments:
                    self._handles.append(_ok(cu.cuMemCreate(n, props[s_ld], 0), f"cuMemCreate locality domain {s_ld} {n} B"))
                self._handle = self._handles[0]
                blobs = [None] * self.world
                dist.all_gather_object(blobs, [self._export(h) for h in self._handles], group=group)
                ptrs = []
                for r, blob in enumerate(blobs):
                    handles = self._handles
                    if r != self.rank:
                        handles = [self._import(b, f"import {r}.{i}") for i, b in enumerate(blob)]
                        self._imports.extend(handles)
                    ptrs.append(self._map_segments(handles))
                self.buffer_ptrs = ptrs
                self.local_ptr = ptrs[self.rank]
                blob = [None]
                if self.rank == 0:
                    self._mc = _ok(cu.cuMulticastCreate(mc_prop), "cuMulticastCreate")
                    blob[0] = self._export(self._mc)
                dist.broadcast_object_list(blob, src=dist.get_global_rank(group, 0), group=group)
                if self.rank != 0:
                    self._mc = self._import(blob[0], "mc import")
                dist.barrier(group=group)  # every importer holds its fds before the exporters close theirs
                for fd in self._fds:
                    os.close(fd)
                self._fds = []
                self._cudev = _ok(cu.cuDeviceGet(index), "cuDeviceGet")
                _ok(cu.cuMulticastAddDevice(self._mc, self._cudev), "cuMulticastAddDevice")
                dist.barrier(group=group)  # every device joins before any bind (cuMulticastCreate)
                offset = 0
                for handle, (_, n) in zip(self._handles, segments):
                    _ok(cu.cuMulticastBindMem(self._mc, offset, handle, 0, n, 0), "cuMulticastBindMem")
                    offset += n
                self._bound = True
                dist.barrier(group=group)
                self.multicast_ptr = self._map(self._mc, size)
                dist.barrier(group=group)
            except BaseException:
                self.close()
                raise

    def _export(self, handle):
        """A picklable shareable handle: FABRIC bytes, or (pid, fd) kept open until every rank imported."""
        cu = _driver()
        got = _ok(cu.cuMemExportToShareableHandle(handle, _kind_type(self.kind), 0), f"export ({self.kind})")
        if self.kind == "fabric":
            return bytes(got.data)
        self._fds.append(int(got))
        return (os.getpid(), int(got))

    def _import(self, blob, what):
        cu = _driver()
        if self.kind == "fabric":  # osHandle points at the 64-byte CUmemFabricHandle (a buffer here)
            return _ok(cu.cuMemImportFromShareableHandle(bytearray(blob), _kind_type("fabric")), what)
        fd = _pidfd_getfd(*blob)  # osHandle is the descriptor value itself (an int here)
        try:
            return _ok(cu.cuMemImportFromShareableHandle(fd, _kind_type("fd")), what)
        finally:
            os.close(fd)

    def _map_segments(self, handles):
        """One reserved VA range holding ``handles`` (this buffer's segment sizes) back to back."""
        cu = _driver()
        va = int(_ok(cu.cuMemAddressReserve(self.size, self.align, 0, 0), "cuMemAddressReserve"))
        self._reserved.append((va, self.size))
        offset = 0
        for handle, (_, n) in zip(handles, self.segments):
            _ok(cu.cuMemMap(va + offset, n, 0, handle, 0), "cuMemMap")
            self._maps.append((va + offset, n, True))
            offset += n
        self._access(va, self.size)
        return va

    def _map(self, handle, size):
        cu = _driver()
        va = int(_ok(cu.cuMemAddressReserve(size, self.align, 0, 0), "cuMemAddressReserve"))
        self._reserved.append((va, size))
        _ok(cu.cuMemMap(va, size, 0, handle, 0), "cuMemMap")
        self._maps.append((va, size, True))
        self._access(va, size)
        return va

    def _access(self, va, size):
        cu = _driver()
        access = cu.CUmemAccessDesc()
        access.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = self.device
        access.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        _ok(cu.cuMemSetAccess(va, size, [access], 1), "cuMemSetAccess")

    def tensor(self, shape, dtype):
        """Zero-copy torch view of this rank's buffer (it keeps the allocation alive)."""
        numel = 1
        for s in shape:
            numel *= int(s)
        nbytes = numel * torch.empty((), dtype=dtype).element_size()
        if nbytes > self.size:
            raise ValueError(f"{tuple(shape)} {dtype} exceeds the {self.size} B buffer")
        raw = torch.as_tensor(_Bytes(self, self.local_ptr, nbytes), device=f"cuda:{self.device}")
        return raw.view(dtype).view(tuple(shape))

    def close(self):
        cu = _driver()
        try:
            torch.cuda.synchronize(self.device)
        except Exception:  # noqa: BLE001 (teardown never raises)
            pass
        for fd in getattr(self, "_fds", []):
            os.close(fd)
        self._fds = []
        for va, size, mapped in reversed(getattr(self, "_maps", [])):
            if mapped:
                cu.cuMemUnmap(va, size)
        for va, size in reversed(getattr(self, "_reserved", [])):
            cu.cuMemAddressFree(va, size)
        self._maps, self._reserved = [], []
        if getattr(self, "_mc", None) is not None:
            if getattr(self, "_bound", False):
                offset = 0
                for _, n in self.segments:
                    cu.cuMulticastUnbind(self._mc, self._cudev, offset, n)
                    offset += n
            cu.cuMemRelease(self._mc)
            self._mc = None
        for handle in getattr(self, "_imports", []) + getattr(self, "_handles", []):
            cu.cuMemRelease(handle)
        self._imports, self._handles, self._handle = [], [], None

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
