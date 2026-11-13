# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The megakernel's cross-GPU buffers on SGLang's TP group.

TRT-LLM allocates the two buffers that cross the GPUs, the MoE hop workspace and the rl workspace, as
``McastGPUBuffer`` over an MPI communicator (MNNVL fabric handles). Here they are torch symmetric memory on the TP
group's CPU (gloo) group with its multicast address: the allocation SGLang's custom all-reduce v2 makes across the two
trays (``custom_all_reduce_v2._allocate_symmetric_memory``). The kernel publishes with plain stores to the multicast
address and polls its own unicast view, so a buffer without a multicast address is refused on every rank.

The vendored workspace classes keep their fields (``uc``, ``mc``, ``peers``, ``arena``, ``total``, ``copies``); only the
allocation and the communicator change. Allocations are collective and matched by position: every rank makes the same
sequence of them.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.distributed as dist

from ..vendor.trtllm_ab93b78.megakernel import layer_mk, moe_hops, rl


class TpComm:
    """The calls the vendored host code makes on its mpi4py communicator (``Get_size``, ``Get_rank``, ``Barrier``,
    ``allgather``), over a torch process group: SGLang's TP ``cpu_group``."""

    def __init__(self, group: dist.ProcessGroup):
        self.group = group
        self._size = dist.get_world_size(group)
        self._rank = dist.get_rank(group)

    def Get_size(self) -> int:
        return self._size

    def Get_rank(self) -> int:
        return self._rank

    def Barrier(self) -> None:
        dist.barrier(group=self.group)

    def allgather(self, obj: object) -> List[object]:
        out: List[object] = [None] * self._size
        dist.all_gather_object(out, obj, group=self.group)
        return out


class McView:
    """The multicast address of a buffer range. The vendored code only reads its ``data_ptr()``."""

    def __init__(self, ptr: int):
        self._ptr = int(ptr)

    def data_ptr(self) -> int:
        return self._ptr


class SymmMcastBuffer:
    """``McastGPUBuffer``'s calls (``get_uc_buffer``, ``get_mc_buffer``) over one torch symmetric-memory allocation of
    ``nbytes`` per rank. Collective over ``comm``; raises on every rank when any rank fails to allocate or has no
    multicast address."""

    def __init__(self, nbytes: int, comm: TpComm, device: torch.device):
        from torch._C._distributed_c10d import _SymmetricMemory

        name = comm.group.group_name
        if (
            torch.__version__ < "2.11.0"
        ):  # as custom_all_reduce_v2._allocate_symmetric_memory
            import torch.distributed._symmetric_memory as torch_symm_mem

            torch_symm_mem.enable_symm_mem_for_group(name)
        self.nbytes = int(nbytes)
        tensor, error = None, ""
        try:
            tensor = _SymmetricMemory.empty_strided_p2p(
                (self.nbytes,), [1], torch.uint8, device, name
            )
        except Exception as e:  # agree first: a rank that raised alone would leave the others in the rendezvous
            error = f"{type(e).__name__}: {e}"
        _agree(comm, error, f"symmetric allocation of {self.nbytes} B")
        self.tensor = tensor
        self.handle = _SymmetricMemory.rendezvous(tensor)
        self.mc_ptr = int(self.handle.multicast_ptr)
        _agree(
            comm,
            "" if self.mc_ptr else "no multicast address",
            f"multicast mapping of {self.nbytes} B",
        )

    def get_uc_buffer(
        self, rank: int, shape: Sequence[int], dtype: torch.dtype, offset: int
    ) -> torch.Tensor:
        """Rank ``rank``'s unicast mapping of the buffer as a ``shape`` tensor (offset 0 only, all the callers use)."""
        assert offset == 0
        return self.handle.get_buffer(int(rank), list(shape), dtype, 0)

    def get_mc_buffer(
        self, shape: Sequence[int], dtype: torch.dtype, offset: int
    ) -> McView:
        """The multicast address of the buffer (offset 0 only)."""
        assert offset == 0
        n = dtype.itemsize
        for s in shape:
            n *= int(s)
        assert n <= self.nbytes
        return McView(self.mc_ptr)


def _agree(comm: TpComm, error: str, what: str) -> None:
    errors = comm.allgather(error)
    bad = {r: e for r, e in enumerate(errors) if e}
    if bad:
        raise RuntimeError(f"{what} failed on ranks {sorted(bad)}: {bad}")


class HopWorkspace(layer_mk.HopWorkspace):
    """``layer_mk.HopWorkspace`` (the served path's 4 copies) on a ``SymmMcastBuffer``: the same size, fields, -0.0
    fill and closing barrier. ``local_rank`` is kept for the vendored signature (``McastGPUBuffer`` used it)."""

    def __init__(self, lib, comm: TpComm, rank: int, local_rank: int, copies: int = 4):
        assert copies >= 3 and comm.Get_size() == moe_hops.RANKS
        self.arena_bytes = moe_hops.arena_bytes(
            lib
        )  # the arena copy of the build's ``nb`` level
        self.fz_bytes = moe_hops.fz_bytes(
            lib
        )  # the build's ``fz`` block per copy, after the copies
        self.copies = copies
        self.rank = int(rank)
        assert not self.fz_bytes or copies <= moe_hops.FZ_MAX_COPIES, (
            "the fz blocks follow FZ_MAX_COPIES copies"
        )
        if self.fz_bytes:  # the fz blocks follow the room of FZ_MAX_COPIES copies
            total = (
                moe_hops.FZ_MAX_COPIES * moe_hops.COPY_BYTES + copies * self.fz_bytes
            )
        else:
            total = copies * moe_hops.COPY_BYTES
        self.total = total
        dev = torch.device("cuda", torch.cuda.current_device())
        self.mcast = SymmMcastBuffer(total, comm, dev)
        self.uc = self.mcast.get_uc_buffer(self.rank, (total // 4,), torch.float32, 0)
        self.mc = self.mcast.get_mc_buffer((total // 4,), torch.float32, 0)
        self.uc.fill_(-0.0)
        self.arena = torch.full(
            (copies * self.arena_bytes // 4,), -0.0, dtype=torch.float32, device=dev
        )
        self.peers = torch.tensor(
            [
                self.mcast.get_uc_buffer(r, (4,), torch.float32, 0).data_ptr()
                for r in range(moe_hops.RANKS)
            ],
            dtype=torch.int64,
            device=dev,
        )
        torch.cuda.synchronize()
        comm.Barrier()


class RlWorkspace(rl.RlWorkspace):
    """``rl.RlWorkspace`` on a ``SymmMcastBuffer``: the same size, fields, -0.0 fill and closing barrier."""

    def __init__(self, comm: TpComm, rank: int, local_rank: int, copies: int):
        assert comm.Get_size() == rl.RANKS
        self.copies = int(copies)
        total = self.copies * rl.COPY_BYTES
        dev = torch.device("cuda", torch.cuda.current_device())
        self.mcast = SymmMcastBuffer(total, comm, dev)
        self.uc = self.mcast.get_uc_buffer(int(rank), (total // 4,), torch.float32, 0)
        self.mc = self.mcast.get_mc_buffer((total // 4,), torch.float32, 0)
        self.uc.fill_(-0.0)
        torch.cuda.synchronize()
        comm.Barrier()
