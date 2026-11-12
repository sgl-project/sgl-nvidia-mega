# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Host side of the K3.1 DCP exchanges (comm.cuh): epoch-tagged receive regions in torch symmetric memory, the
per-forward epoch, and the four PDL launches.

Serving: ``DcpComm.from_group`` rendezvouses one receive region per exchange kind on every rank of the DCP group (the
TP group). Tests: ``DcpComm(..., peers=...)`` takes explicit region tensors (all ranks emulated on one GPU, or virtual
ranks next to live ones), and ``phases`` splits each exchange into its push and poll halves.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

import torch

from .layout import KIND_CAND, KIND_PART, KIND_Q, Geometry

SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("load_jit",),
}

DEADLINE_NS = int(float(os.environ.get("K31MK_DCP_DEADLINE_S", "10")) * 1e9)
PUSH, POLL, BOTH = 1, 2, 3


@cache
def load():
    import sglang
    from sglang.kernels.jit.utils import load_jit

    tag = ("k31_dcp_comm",)
    return load_jit(
        *tag,
        cuda_files=[str(Path(__file__).with_name("comm.cuh"))],
        cuda_wrappers=[
            ("prologue", "k31_dcp::Binding::prologue"),
            ("qpush", "k31_dcp::Binding::qpush"),
            ("cand", "k31_dcp::Binding::cand"),
            ("merge", "k31_dcp::Binding::merge"),
        ],
        extra_include_paths=[str(Path(sglang.__file__).parent / "kernels/jit/csrc")],
    )


def _sms(device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


class DcpComm:
    """The DCP exchange state of one model: three receive regions (q, cand, part) per rank, int64 records."""

    def __init__(
        self,
        device,
        dcp: int,
        rank: int,
        heads: int,
        cap: int,
        *,
        layers: int = 24,
        peers: dict[int, list[torch.Tensor]] | None = None,
        epoch: torch.Tensor | None = None,
        deadline_ns: int = DEADLINE_NS,
        blocks: int | None = None,
    ):
        """``peers``: {kind: [region tensor of rank 0..dcp-1]} (tests). ``epoch``: an int32 device tensor whose
        element 0 is the epoch counter (shared by every exchange of this model)."""
        if not 2 <= dcp <= 16 or not 0 <= rank < dcp:
            raise ValueError(f"DCP size {dcp} / rank {rank} unsupported (2..16)")
        self.device = torch.device(device)
        self.dcp, self.rank, self.heads, self.cap, self.layers = dcp, rank, heads, cap, layers
        self.geometry = Geometry(dcp, heads, cap)
        self.deadline_ns = int(deadline_ns)
        self.blocks = blocks or min(_sms(self.device), 128)
        self.epoch = epoch if epoch is not None else torch.zeros(4, dtype=torch.int32, device=self.device)
        self.regions = peers or {}
        self._meta: dict[tuple[int, int, int], torch.Tensor] = {}
        self.module = None

    @classmethod
    def from_group(cls, group, device, dcp, rank, heads, cap, *, live=None, **kwargs):
        """Collective over ``group``: allocate, zero and rendezvous the three receive regions. ``live`` < dcp
        (tests): ``group`` holds ranks 0..live-1; stores to virtual ranks live..dcp-1 land
        in local scratch, and the test fills this rank's slots of virtual sources (``fill_virtual``)."""
        import torch.distributed._symmetric_memory as symm_mem

        live = dcp if live is None else live

        geometry = Geometry(dcp, heads, cap)
        owned, handles = {}, {}
        for kind, records in (
            (KIND_Q, geometry.q_records()),
            (KIND_CAND, geometry.cand_records()),
            (KIND_PART, geometry.part_records()),
        ):
            buffer = symm_mem.empty(records, dtype=torch.int64, device=device)
            buffer.zero_()  # tag 0 is never valid
            owned[kind] = buffer
        torch.cuda.synchronize(device)
        for kind, buffer in owned.items():
            handles[kind] = symm_mem.rendezvous(buffer, group)
        self = cls(device, dcp, rank, heads, cap, **kwargs)
        self.owned, self.handles = owned, handles
        self.pointers = {kind: list(handle.buffer_ptrs) for kind, handle in handles.items()}
        if any(len(p) != live for p in self.pointers.values()):
            raise RuntimeError("DCP rendezvous did not return one region per live rank")
        self.virtual = {kind: [torch.zeros_like(owned[kind]) for _ in range(dcp - live)] for kind in owned}
        for kind, scratch in self.virtual.items():
            self.pointers[kind] += [int(t.data_ptr()) for t in scratch]
        torch.distributed.barrier(group=group)
        return self

    def pointer_list(self, kind: int) -> list[int]:
        if hasattr(self, "pointers"):
            return self.pointers[kind]
        return [int(t.data_ptr()) for t in self.regions[kind]]

    def meta(self, kind: int, ordinal: int, phases: int = BOTH) -> torch.Tensor:
        """The CPU parameter vector of one exchange site (comm.cuh Binding::common)."""
        key = (kind, ordinal, phases)
        if key not in self._meta:
            values = [self.dcp, self.rank, ordinal % 2, ordinal, self.layers, self.cap]
            values += [self.deadline_ns, self.blocks, phases, *self.pointer_list(kind)]
            self._meta[key] = torch.tensor(values, dtype=torch.int64)
        return self._meta[key]

    def _module(self):
        if self.module is None:
            self.module = load()
        return self.module

    # ----------------------------------------------------------------------------------------- launches
    def prologue(self, write_slots: torch.Tensor, local_slots: torch.Tensor) -> None:
        """epoch += 1; widened write slots -> this rank's rows (-1 elsewhere). Once per forward, before layer 0."""
        self._module().prologue(self.epoch, write_slots, local_slots, self.dcp, self.rank)

    def qpush(self, ordinal, q, glens, *, dense=None, visible=None, table=None, rows_per_request=1, phases=BOTH):
        empty = torch.empty(0, dtype=torch.int32, device=self.device)
        flags = (1 if dense is not None else 0) | (2 if visible is not None else 0)
        vis, count = visible if visible is not None else (empty, empty)
        self._module().qpush(
            self.meta(KIND_Q, ordinal, phases),
            self.epoch,
            q,
            dense if dense is not None else empty.float(),
            glens,
            vis,
            count,
            table if table is not None else empty,
            rows_per_request,
            flags,
        )

    def cand(self, ordinal, candidates, glens, dense, q_local, q_out, *, phases=BOTH):
        """``candidates`` None: q poll only (dense graphs)."""
        push = candidates is not None
        empty = torch.empty((0, 1, 2), dtype=torch.int32, device=self.device)
        self._module().cand(
            self.meta(KIND_CAND, ordinal, phases),
            self.meta(KIND_Q, ordinal, phases),
            self.epoch,
            candidates if push else empty,
            glens,
            dense if push else empty.float().view(0, 1),
            q_local,
            q_out,
            int(push),
        )

    def merge(self, ordinal, part, lse, counts, totals, w_uv, gate, out, *, phases=BOTH):
        self._module().merge(
            self.meta(KIND_PART, ordinal, phases), self.epoch, part, lse, counts, totals, w_uv, gate, out
        )
