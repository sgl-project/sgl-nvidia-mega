# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``req_bcast`` (env ``MUSE_VR_REQ_BCAST=1``, default off): the scheduler's per-step request broadcast across
tensor-parallel ranks on several hosts carries large payloads over NVLink instead of the hosts' TCP network.

Every scheduler step, TP rank 0 broadcasts the requests it received (pickled, ``broadcast_pyobj`` over the TP group's
gloo CPU group: one size message, then the payload) and every other rank waits for them. When the TP group spans hosts
(e.g. tensor parallel 8 over two 4-GPU trays) that payload crosses the hosts' TCP network; a request carries its full
``input_ids`` list, so a long-context request costs tens of milliseconds during which every rank's scheduler is blocked.

This component is an AROUND hook on ``SchedulerRequestReceiver._broadcast_reqs_across_ranks``. For the
non-data-parallel-attention branch it issues the same sequence as SGLang (append local requests, broadcast from the TP
group's first rank) with one change: the size message still goes over the gloo group, and when the pickled payload is
larger than ``MUSE_VR_REQ_BCAST_MIN_KB`` (default 64) it is broadcast from a device buffer over a dedicated NCCL group of
the same ranks (created on first use, collectively, on a dedicated stream that is synchronized before unpickling).
Smaller payloads take SGLang's gloo path unchanged. Every rank takes the same branch (decided from the broadcast size).
The data-parallel-attention branch, single-host groups and TP size 1 run SGLang's method unchanged. The objects every
rank ends up with are the pickled objects themselves (byte-identical payload), so scheduling is unchanged.

Engagement requires the sha256 of the receiver module and of ``broadcast_pyobj``'s module to match the pinned revision.
"""

from __future__ import annotations

import os
import pickle

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/runtime_context.py": ("get_parallel",),
    "srt/utils/*.py": ("broadcast_pyobj",),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


NAME = "req_bcast"
SGLANG_PINS = {
    "sglang/srt/managers/scheduler_components/request_receiver.py": "101558ad106b28d70de7a72a3159dedb11b21af6d787d8d8b1e75993e64d1a5d",
    "sglang/srt/utils/common.py": (
        "61293a66e44b76cb4f4320a0942cc6b817387f10d472d8615373da01641b1063",
        "6bb2a7eee7885551916f4ef762649c74e2fbafd97c48ed273ad6d1afd96d8366",
    ),
}
TARGET = "sglang.srt.managers.scheduler_components.request_receiver.SchedulerRequestReceiver._broadcast_reqs_across_ranks"
_state: dict = {"engaged": None, "groups": {}}


def _min_bytes() -> int:
    return int(os.environ.get("MUSE_VR_REQ_BCAST_MIN_KB", "64")) * 1024


def _quiesce() -> bool:
    """``MUSE_VR_REQ_BCAST_QUIESCE=1`` (default off): before a large payload's NCCL broadcast, the host waits until the
    device has finished every queued kernel (``torch.cuda.synchronize``), so the broadcast kernel never runs beside a
    forward in flight. A cross-rank kernel that holds SMs while it waits on other ranks can starve a kernel that needs
    every SM resident at once (the one-launch MoE kernel); paying one device drain on the rare large-payload steps
    (new long requests) removes that overlap. Small payloads keep the gloo path and are unaffected."""
    return os.environ.get("MUSE_VR_REQ_BCAST_QUIESCE", "0").strip() not in ("", "0")


class _Plane:
    """Dedicated NCCL group + stream + growable device buffer for one CPU (gloo) group."""

    def __init__(self, cpu_group):
        import socket

        import torch
        import torch.distributed as dist

        self.ranks = dist.get_process_group_ranks(cpu_group)
        hosts = [None] * len(self.ranks)
        dist.all_gather_object(hosts, socket.gethostname(), group=cpu_group)
        self.multi_host = len(set(hosts)) > 1
        self.group = None
        if self.multi_host:
            # every member calls this at the same point (first large-capable broadcast); only members synchronize
            self.group = dist.new_group(
                ranks=self.ranks, backend="nccl", use_local_synchronization=True
            )
            self.device = torch.device("cuda", torch.cuda.current_device())
            self.stream = torch.cuda.Stream(device=self.device)
            # Every device tensor of the plane is allocated ON the plane's stream: the caching allocator then never
            # hands this side stream a block that work queued on another stream may still be using (a block is
            # reused only by allocations on the stream it was allocated on).
            with torch.cuda.stream(self.stream):
                self.buf = torch.empty(0, dtype=torch.uint8, device=self.device)
                # initialize the communicator now (first scheduler step, before the server reports ready), not on
                # the first large request
                warm = torch.zeros(1, dtype=torch.uint8, device=self.device)
                dist.broadcast(warm, src=self.ranks[0], group=self.group)
            self.stream.synchronize()
        bed.report(
            NAME + ":plane",
            self.multi_host,
            world=len(self.ranks),
            hosts=len(set(hosts)),
        )

    def ensure(self, n: int):
        import torch

        if self.buf.numel() < n:
            # on the plane's stream (see __init__); the old buffer's last use ended in a plane.stream.synchronize()
            with torch.cuda.stream(self.stream):
                self.buf = torch.empty(
                    max(n, 1 << 20) * 2, dtype=torch.uint8, device=self.device
                )
        return self.buf[:n]


def _plane(cpu_group):
    key = id(cpu_group)
    p = _state["groups"].get(key)
    if p is None:
        p = _Plane(cpu_group)
        _state["groups"][key] = p
    return p


def broadcast(data, rank: int, cpu_group, src: int):
    """Same contract as ``sglang.srt.utils.broadcast_pyobj(data, rank, cpu_group, src)`` (CPU path)."""
    import numpy as np
    import torch
    import torch.distributed as dist

    plane = _plane(cpu_group)
    if not plane.multi_host:
        from sglang.srt.utils import broadcast_pyobj

        return broadcast_pyobj(data, rank, cpu_group, src=src)
    thr = _min_bytes()
    if rank == src:
        if len(data) == 0:
            dist.broadcast(
                torch.tensor([0], dtype=torch.long), src=src, group=cpu_group
            )
            return data
        payload = pickle.dumps(data)
        n = len(payload)
        dist.broadcast(torch.tensor([n], dtype=torch.long), src=src, group=cpu_group)
        host = torch.from_numpy(np.frombuffer(payload, dtype=np.uint8))
        if n <= thr:
            dist.broadcast(host.clone(), src=src, group=cpu_group)
            return data
        dev = plane.ensure(n)
        if _quiesce():
            torch.cuda.synchronize(plane.device)
            bed.count(NAME, "quiesce")
        with torch.cuda.stream(plane.stream):
            dev.copy_(host)
            dist.broadcast(dev, src=src, group=plane.group)
        plane.stream.synchronize()
        _tally(n)
        return data
    size = torch.tensor([0], dtype=torch.long)
    dist.broadcast(size, src=src, group=cpu_group)
    n = int(size.item())
    if n == 0:
        return []
    if n <= thr:
        host = torch.empty(n, dtype=torch.uint8)
        dist.broadcast(host, src=src, group=cpu_group)
        return pickle.loads(host.numpy().tobytes())
    dev = plane.ensure(n)
    if _quiesce():
        torch.cuda.synchronize(plane.device)
        bed.count(NAME, "quiesce")
    with torch.cuda.stream(plane.stream):
        dist.broadcast(dev, src=src, group=plane.group)
        host = dev.to("cpu", non_blocking=False)
    plane.stream.synchronize()
    _tally(n)
    return pickle.loads(host.numpy().tobytes())


def _tally(n: int) -> None:
    b = 1 << max(16, (n - 1).bit_length())
    bed.count(NAME, f"nccl:<= {b // 1024} KB")


def _around(original, self, recv_reqs, local_reqs=None):
    from sglang.srt.runtime_context import get_parallel

    par = get_parallel()
    if par.attn_dp_enabled or par.tp_size == 1:
        return original(self, recv_reqs, local_reqs)
    local_reqs = local_reqs or []
    if recv_reqs is not None:
        recv_reqs = [*recv_reqs, *local_reqs]
    tp_group = par.tp_group
    return broadcast(
        recv_reqs, tp_group.rank, tp_group.cpu_group, src=tp_group.ranks[0]
    )


def register() -> bool:
    if _state["engaged"] is not None:
        return _state["engaged"]
    if not bed.enabled(NAME):
        _state["engaged"] = False
        return False
    for rel, sha in SGLANG_PINS.items():
        if not bed.pinned(rel, sha):
            bed.report(NAME, False, reason=f"pin mismatch: {rel}")
            _state["engaged"] = False
            return False
    try:
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType

        HookRegistry.register(TARGET, _around, HookType.AROUND)
    except Exception as e:  # noqa: BLE001
        bed.report(NAME, False, reason=f"install failed: {type(e).__name__}: {e}")
        _state["engaged"] = False
        return False
    bed.report(NAME, True, min_kb=_min_bytes() // 1024, quiesce=_quiesce())
    _state["engaged"] = True
    return True
