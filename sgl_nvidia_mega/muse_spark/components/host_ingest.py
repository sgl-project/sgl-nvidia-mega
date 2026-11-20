# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``host_ingest`` (env ``MUSE_VR_HOST_INGEST=1``, default off): the scheduler's per-step request broadcast
runs on a helper thread, overlapped with the GPU work of the step that was just launched.

In every scheduler loop SGLang polls the tokenizer socket on tensor-parallel rank 0 and broadcasts what it received
(``broadcast_pyobj`` over the tensor-parallel group's gloo CPU group; one size message even when nothing arrived). Every
other rank blocks in that broadcast until rank 0 reaches it, so each loop pays the poll, the broadcast latency and the
wait for rank 0 on the scheduler thread, before the next batch can be scheduled and launched.

With this component each rank owns a helper thread and a dedicated gloo group over the same ranks. Right after a batch
is launched (``Scheduler.run_batch`` returns), the scheduler thread starts one *round* on the helper: rank 0 polls the
sockets exactly as SGLang does and broadcasts the list over the dedicated group; the other ranks receive it. The next
``recv_requests`` call takes that round's list instead of polling. When no round is in flight (an idle loop, or the
first call), ``recv_requests`` runs a round and waits for it, which is SGLang's behaviour. Every rank therefore consumes
the same list in the same loop iteration (rounds are numbered, started from identical points of the lock-step loop and
consumed in order); a request is scheduled at most one loop later than SGLang would schedule it.

Payloads above ``MUSE_VR_HOST_INGEST_MAX_KB`` (default 64) are not sent by the helper: it broadcasts a marker and rank 0
keeps the list; the consuming ``recv_requests`` call then broadcasts it with SGLang's own method on the scheduler
thread (so components wrapping that method, e.g. ``req_bcast``, see it unchanged). Requests produced locally by the
scheduler (timeout aborts) join the next round. Everything after the broadcast (unpickling wrappers, multimodal
receivers, shared-memory features) runs on the scheduler thread as in SGLang.

``MUSE_VR_HOST_INGEST_ASYNC=1`` selects a round without a helper thread: right after ``run_batch`` rank 0 polls on the
scheduler thread, packs a size header and a small payload (at most ``MUSE_VR_HOST_INGEST_ASYNC_BYTES``, default 1024)
into one fixed-size buffer and starts an asynchronous gloo broadcast of it over the dedicated group (the other ranks
start the matching receive); ``recv_requests`` waits for that operation. The transfer runs on gloo's own threads, so it
needs no Python interpreter time while the scheduler thread works. Larger payloads take the marker path above.

``MUSE_VR_HOST_PIN=1`` (experiment, placement only) pins each scheduler thread to one physical core of the CPU set SGLang
bound the process to, a different core per tensor-parallel rank of the pod, and moves the process's other threads off
the reserved cores (see ``_pin``).

Steps aside to SGLang's method (reported once): tensor parallel size 1, data-parallel attention, pipeline parallelism,
a receive skipper, an input blocker, a scripted scheduler hook, the Rust server ingress, or an idle sleeper (they poll
the same sockets). Engagement requires the receiver and scheduler sources to match their sha256 pins. A helper error
stops the server loudly (ranks could otherwise disagree on the request sequence).
"""

from __future__ import annotations

import logging
import os
import pickle
import threading

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/environ.py": ("envs",),
    "srt/runtime_context.py": ("get_device", "get_parallel"),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


logger = logging.getLogger(__name__)

NAME = "host_ingest"
PINS = {
    "sglang/srt/managers/scheduler_components/request_receiver.py": "101558ad106b28d70de7a72a3159dedb11b21af6d787d8d8b1e75993e64d1a5d",
    "sglang/srt/managers/scheduler.py": (
        "c1edcfe6f85e49bf85faf1f79129f4452445585de17cf1cdcc821de5008b1491",
        "fc96ed4b680fdf55621a5397f844cfd4cbd30213f44eec8a888b7cce75cafe60",
    ),
}
TARGET_RECV = "sglang.srt.managers.scheduler_components.request_receiver.SchedulerRequestReceiver.recv_requests"
TARGET_RUN = "sglang.srt.managers.scheduler.Scheduler.run_batch"
ENV_MAX_KB = "MUSE_VR_HOST_INGEST_MAX_KB"
ENV_ASYNC = "MUSE_VR_HOST_INGEST_ASYNC"
ENV_ASYNC_BYTES = "MUSE_VR_HOST_INGEST_ASYNC_BYTES"
ENV_PIN = "MUSE_VR_HOST_PIN"  # debug / experiment: pin each scheduler's main thread to one physical core (see _pin)
PIN_AFTER = (
    64  # loops before the first pinning (every thread of the process exists by then)
)
PIN_EVERY = 4096  # loops between re-applications to threads created later
PIN_SLOTS = 4  # cores reserved per CPU set: one per scheduler that can share it (ranks on one pod)

_LARGE = -1  # header marker: payload kept on rank 0, broadcast by the consumer with SGLang's method
_state: dict = {"engaged": None, "pumps": {}, "declined": set()}


def max_bytes() -> int:
    return max(0, bed.env_int(ENV_MAX_KB, 64)) * 1024


def async_mode() -> bool:
    return bed.env_int(ENV_ASYNC, 0) == 1


def async_bytes() -> int:
    return max(0, bed.env_int(ENV_ASYNC_BYTES, 1024))


class Pump:
    """One helper thread + dedicated gloo group per receiver. ``start`` begins a round, ``take`` returns its result.

    ``pull`` (rank 0 only) returns the raw request list; ``group`` / ``src`` / ``is_src`` describe the broadcast.
    Results are ``(list, None)`` for a list delivered by the helper or ``(None, held)`` for a marker round, where
    ``held`` is rank 0's list (``None`` on other ranks) to be broadcast by the caller.
    """

    def __init__(self, pull, group, src: int, is_src: bool, limit: int):
        self.pull, self.group, self.src, self.is_src, self.limit = (
            pull,
            group,
            src,
            is_src,
            limit,
        )
        self._cv = threading.Condition()
        self._job = None  # (round index, local requests)
        self._result = None  # (round index, payload, held, error)
        self.started = 0
        self.taken = 0
        self._thread = threading.Thread(
            target=self._run, name="muse-vr-host-ingest", daemon=True
        )
        self._thread.start()

    def in_flight(self) -> bool:
        return self.started > self.taken

    def start(self, local_reqs) -> None:
        with self._cv:
            if self.started > self.taken:
                raise RuntimeError(
                    f"{NAME}: round {self.started} started while round {self.taken + 1} is pending"
                )
            self.started += 1
            self._job = (self.started, list(local_reqs or []))
            self._cv.notify_all()

    def take(self):
        with self._cv:
            if self.started == self.taken:
                raise RuntimeError(f"{NAME}: take() without a started round")
            want = self.taken + 1
            while self._result is None or self._result[0] != want:
                if not self._thread.is_alive():
                    raise RuntimeError(f"{NAME}: helper thread died")
                self._cv.wait(timeout=1.0)
            _, payload, held, err = self._result
            self._result = None
            self.taken = want
        if err is not None:
            raise RuntimeError(
                f"{NAME}: request round {want} failed on the helper: {err}"
            ) from err
        return payload, held

    def _run(self) -> None:
        import torch
        import torch.distributed as dist

        while True:
            with self._cv:
                while self._job is None:
                    self._cv.wait()
                idx, local = self._job
                self._job = None
            payload = held = err = None
            try:
                header = torch.zeros(1, dtype=torch.long)
                if self.is_src:
                    reqs = self.pull()
                    reqs = [*(reqs or []), *local]
                    blob = pickle.dumps(reqs) if reqs else b""
                    n = len(blob)
                    if n > self.limit:
                        header[0] = _LARGE
                        dist.broadcast(header, src=self.src, group=self.group)
                        held = reqs
                    else:
                        header[0] = n
                        dist.broadcast(header, src=self.src, group=self.group)
                        if n:
                            import numpy as np

                            data = torch.from_numpy(
                                np.frombuffer(blob, dtype=np.uint8).copy()
                            )
                            dist.broadcast(data, src=self.src, group=self.group)
                        payload = reqs
                else:
                    dist.broadcast(header, src=self.src, group=self.group)
                    n = int(header[0])
                    if n == _LARGE:
                        held = None
                    elif n == 0:
                        payload = []
                    else:
                        data = torch.empty(n, dtype=torch.uint8)
                        dist.broadcast(data, src=self.src, group=self.group)
                        payload = pickle.loads(data.numpy().tobytes())
            except Exception as e:  # noqa: BLE001 - surfaced by take()
                err = e
            with self._cv:
                self._result = (idx, payload, held, err)
                self._cv.notify_all()


class AsyncPump:
    """``Pump``'s interface without a helper thread: a round is one asynchronous gloo broadcast of a fixed-size buffer
    (8-byte size header + up to ``small`` payload bytes), started by ``start`` on the scheduler thread and completed by
    ``take``. One round in flight at a time, so the buffer is never rewritten while a transfer may still read it."""

    def __init__(self, pull, group, src: int, is_src: bool, small: int):
        import torch

        self.pull, self.group, self.src, self.is_src, self.small = (
            pull,
            group,
            src,
            is_src,
            small,
        )
        self.buf = torch.zeros(8 + small, dtype=torch.uint8)
        self.header = self.buf[:8].view(torch.int64)
        self._work = None
        self._own = None  # rank 0: the list it sent (or holds, on a marker round)
        self.started = 0
        self.taken = 0

    def in_flight(self) -> bool:
        return self.started > self.taken

    def start(self, local_reqs) -> None:
        import torch
        import torch.distributed as dist

        if self.started > self.taken:
            raise RuntimeError(
                f"{NAME}: round {self.started} started while round {self.taken + 1} is pending"
            )
        self.started += 1
        self._own = None
        if self.is_src:
            reqs = [*(self.pull() or []), *(local_reqs or [])]
            blob = pickle.dumps(reqs) if reqs else b""
            n = len(blob)
            if n > self.small:
                self.header[0] = _LARGE
            else:
                self.header[0] = n
                if n:
                    self.buf[8 : 8 + n].copy_(
                        torch.frombuffer(bytearray(blob), dtype=torch.uint8)
                    )
            self._own = reqs
        self._work = dist.broadcast(
            self.buf, src=self.src, group=self.group, async_op=True
        )

    def take(self):
        if self.started == self.taken:
            raise RuntimeError(f"{NAME}: take() without a started round")
        work, self._work = self._work, None
        want = self.taken + 1
        try:
            work.wait()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"{NAME}: request round {want} failed: {e}") from e
        self.taken = want
        n = int(self.header[0])
        if n == _LARGE:
            return None, (self._own if self.is_src else None)
        if self.is_src:
            return self._own, None
        if n == 0:
            return [], None
        return pickle.loads(self.buf[8 : 8 + n].numpy().tobytes()), None


def _eligible(receiver) -> str | None:
    """None when the component may serve this receiver, else the reason it steps aside."""
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_parallel

    par = get_parallel()
    if par.tp_size == 1:
        return "tensor parallel size 1"
    if par.attn_dp_enabled:
        return "data-parallel attention"
    if par.pp_size != 1:
        return "pipeline parallelism"
    if receiver.recv_skipper is not None:
        return "receive skipper set"
    if receiver.input_blocker is not None:
        return "input blocker set"
    if receiver.scripted_scheduler_hook is not None:
        return "scripted scheduler hook set"
    if envs.SGLANG_RUST_SERVER.get():
        return "Rust server ingress"
    from sglang.srt.runtime_context import get_device

    if (
        par.pp_rank == 0
        and par.attn_tp_rank == 0
        and getattr(get_device(), "sleep_on_idle", False)
    ):
        return "idle sleeper polls the same sockets"
    return None


def _ready_pull(receiver):
    """The receiver's pull, with a readiness check first: when neither of its two sockets has a message waiting, the
    pull would return an empty list after two non-blocking receives that each raise and catch an error; this returns
    that empty list directly. Any socket without the readiness query keeps the plain pull."""
    pull = receiver._pull_raw_reqs
    socks = [
        getattr(receiver, n, None) for n in ("recv_from_tokenizer", "recv_from_rpc")
    ]
    try:
        import zmq

        ev, pollin = zmq.EVENTS, zmq.POLLIN
    except Exception:  # noqa: BLE001
        return pull
    if not all(s is not None and hasattr(s, "getsockopt") for s in socks):
        return pull

    def ready_pull():
        try:
            if not (socks[0].getsockopt(ev) & pollin) and not (
                socks[1].getsockopt(ev) & pollin
            ):
                return []
        except Exception:  # noqa: BLE001 - fall back to the plain pull
            pass
        return pull()

    return ready_pull


def _pump(receiver):
    key = id(receiver)
    p = _state["pumps"].get(key)
    if p is not None or key in _state["declined"]:
        return p
    why = _eligible(receiver)
    if why is not None:
        _state["declined"].add(key)
        bed.report(NAME + ":receiver", False, reason=why)
        return None
    import torch.distributed as dist
    from sglang.srt.runtime_context import get_parallel

    tp = get_parallel().tp_group
    # every rank of the group reaches its first recv_requests call at the same point of the lock-step loop
    group = dist.new_group(
        ranks=list(tp.ranks), backend="gloo", use_local_synchronization=True
    )
    src, is_src = tp.ranks[0], tp.rank == tp.ranks[0]
    if async_mode():
        p = AsyncPump(
            _ready_pull(receiver) if is_src else receiver._pull_raw_reqs,
            group,
            src=src,
            is_src=is_src,
            small=async_bytes(),
        )
        mode = {"mode": "async", "async_bytes": async_bytes()}
    else:
        p = Pump(
            receiver._pull_raw_reqs, group, src=src, is_src=is_src, limit=max_bytes()
        )
        mode = {"mode": "thread", "max_kb": max_bytes() // 1024}
    _state["pumps"][key] = p
    _state["local"] = []
    bed.report(NAME + ":receiver", True, world=len(tp.ranks), **mode)
    return p


def _finish(receiver, recv_reqs):
    """SGLang's post-broadcast steps of ``recv_requests``, unchanged."""
    from sglang.srt.runtime_context import get_parallel

    if get_parallel().pp_rank == 0:
        receiver.unwrap_pickle_wrapper(recv_reqs)
    recv_reqs = receiver._apply_mm_receiver(recv_reqs)
    receiver._finalize_shm_features(recv_reqs)
    return recv_reqs


def _around_recv(original, receiver, local_reqs=None):
    p = _pump(receiver)
    if p is None:
        return original(receiver, local_reqs=local_reqs)
    if local_reqs:
        _state["local"].extend(local_reqs)
    if not p.in_flight():
        p.start(_state.pop("local", None) or [])
        _state["local"] = []
        bed.count(NAME, "round:sync")
    else:
        bed.count(NAME, "round:overlapped")
    payload, held = p.take()
    if (
        payload is None
    ):  # large payload: rank 0 holds it; broadcast with SGLang's method on this thread
        bed.count(NAME, "round:large")
        payload = receiver._broadcast_reqs_across_ranks(held, None)
    return _finish(receiver, payload)


def _cpu_list(text: str) -> set:
    out = set()
    for part in text.strip().split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out


def _cores(allowed: set) -> list:
    """The physical cores lying wholly inside ``allowed``, each as the sorted tuple of its logical CPUs."""
    seen, cores = set(), []
    for c in sorted(allowed):
        if c in seen:
            continue
        try:
            with open(
                f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list"
            ) as f:
                ids = _cpu_list(f.read())
        except OSError:
            ids = {c}
        seen |= ids
        if ids <= allowed:
            cores.append(tuple(sorted(ids)))
    return cores


def _pin(scheduler) -> None:
    """``MUSE_VR_HOST_PIN=1``: the scheduler thread gets one physical core of the CPU set SGLang bound the process to
    (both of its logical CPUs, the last cores of the set, a different core per tensor-parallel rank of the pod), and the
    process's other threads get the rest of the set minus every such reserved core; re-applied now and then to threads
    created later. Placement only: no computation, ordering or value changes."""
    st = _state.setdefault("pin", {"n": 0})
    st["n"] += 1
    n = st["n"]
    if (
        n < PIN_AFTER
        or (n > PIN_AFTER and (n - PIN_AFTER) % PIN_EVERY)
        or st.get("off")
    ):
        return
    try:
        if "core" not in st:
            from sglang.srt.runtime_context import get_parallel

            allowed = os.sched_getaffinity(0)
            cores = _cores(allowed)
            if len(cores) < 2 * PIN_SLOTS:
                st["off"] = True
                bed.report(
                    NAME + ":pin",
                    False,
                    reason=f"{len(cores)} whole cores in the CPU set",
                )
                return
            reserved = cores[-PIN_SLOTS:]
            core = reserved[-1 - (get_parallel().tp_rank % PIN_SLOTS)]
            st["core"], st["main"] = set(core), threading.get_native_id()
            st["rest"] = allowed - {c for r in reserved for c in r}
            os.sched_setaffinity(0, st["core"])
            bed.report(NAME + ":pin", True, core=sorted(core), rest=len(st["rest"]))
        moved = 0
        for tid in os.listdir("/proc/self/task"):
            t = int(tid)
            if t != st["main"]:
                try:
                    os.sched_setaffinity(t, st["rest"])
                    moved += 1
                except OSError:
                    pass
        bed.count(NAME, "pin:threads moved", moved)
    except Exception as e:  # noqa: BLE001 - placement is optional: never break the loop
        st["off"] = True
        bed.report(NAME + ":pin", False, reason=f"{type(e).__name__}: {e}")


def _after_run(result, scheduler, *args, **kwargs):
    if _state.get("pin_on"):
        _pin(scheduler)
    receiver = getattr(scheduler, "request_receiver", None)
    if receiver is None:
        return result
    p = _state["pumps"].get(id(receiver))
    if p is not None and not p.in_flight():
        p.start(_state.pop("local", None) or [])
        _state["local"] = []
    return result


def register() -> bool:
    if _state["engaged"] is not None:
        return _state["engaged"]
    if not bed.enabled(NAME):
        _state["engaged"] = False
        return False
    for rel, sha in PINS.items():
        if not bed.pinned(rel, sha):
            bed.report(NAME, False, reason=f"{rel} differs from the pinned revision")
            _state["engaged"] = False
            return False
    try:
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType

        def recv_hook(original, *args, **kwargs):
            if len(args) != 1 or set(kwargs) - {"local_reqs"}:
                return original(*args, **kwargs)
            return _around_recv(original, args[0], kwargs.get("local_reqs"))

        HookRegistry.register(TARGET_RECV, recv_hook, HookType.AROUND)
        HookRegistry.register(TARGET_RUN, _after_run, HookType.AFTER)
    except Exception as e:  # noqa: BLE001
        bed.report(NAME, False, reason=f"install failed: {type(e).__name__}: {e}")
        _state["engaged"] = False
        return False
    _state["pin_on"] = os.environ.get(ENV_PIN) == "1"
    bed.report(
        NAME, True, max_kb=max_bytes() // 1024, pid=os.getpid(), pin=_state["pin_on"]
    )
    _state["engaged"] = True
    return True
