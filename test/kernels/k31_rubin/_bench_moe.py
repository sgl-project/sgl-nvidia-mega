# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Benchmark-only stand-in kernels for the MoE experiments (never shipped).

``FrontStandIn`` emulates ``MoeFrontTP8`` for the routed experts that follow it (E5 of
the evolve design, E-EXPERTS a): after its dependency wait it re-arms the latent
mailbox row (sentinel 0x80000000), releases its dependents, publishes the 16 expert IDs
``t_ids`` ns after its start and the FP32 latent ``t_lat`` ns after its start (the
latent gather, hop A, lands shortly after x_moe on TP8). A release fence orders
the re-arm before ``launch_dependents`` (PTX griddepcontrol: a dependent's pre-wait
reads are ordered only by a release before the trigger)."""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream


def _primitives():
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common import primitives as P

    return P


def build_standin():
    P = _primitives()

    @P.asm()
    def fence_acq_rel_gpu():
        return "fence.acq_rel.gpu;", "~{memory}", ()

    class FrontStandIn:
        def __init__(self, t_ids, t_lat, wait_first=True):
            self.t_ids, self.t_lat, self.wait_first = t_ids, t_lat, wait_first

        @cute.jit
        def __call__(self, *args):
            P.launch(self.kernel, args, 1, 128)

        @cute.kernel
        def kernel(self, args):
            ids_src, ids, lat_src, lat = args
            tid, _, _, _ = P.thread()
            if cutlass.const_expr(self.wait_first):
                cute.arch.griddepcontrol_wait()
            start = P.global_timer_ns()
            sentinel = (Uint32(0x80000000),) * 4
            for i in range(tid, 896, 128):
                P.store_words(lat.iterator + 4 * i, sentinel, scope="gpu")
            fence_acq_rel_gpu()
            cute.arch.sync_threads()
            cute.arch.griddepcontrol_launch_dependents()
            now = P.global_timer_ns()
            while now - start < Int64(self.t_ids):
                now = P.global_timer_ns()
            if tid < 16:
                P.store_words(ids.iterator + tid, (Uint32(ids_src[tid]),), scope="gpu")
            now = P.global_timer_ns()
            while now - start < Int64(self.t_lat):
                now = P.global_timer_ns()
            for i in range(tid, 896, 128):
                words = P.load_words(lat_src.iterator + 4 * i, count=4, mutable=True)
                P.store_words(lat.iterator + 4 * i, words, scope="sys")

    return FrontStandIn


_CACHE = {}


def standin(t_ids_ns, t_lat_ns, device_index, wait_first=True):
    """Compiled stand-in: call(ids_src i32[16], ids i32[16], lat_src f32[3584], lat f32[3584], stream)."""
    key = (t_ids_ns, t_lat_ns, device_index, wait_first)
    if key not in _CACHE:
        import torch
        from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import fake

        cls = build_standin()
        args = (fake((16,), None, torch.int32), fake((16,), None, torch.int32),
                fake((3584,), None, torch.float32), fake((3584,), None, torch.float32))
        with torch.accelerator.device_index(device_index):
            _CACHE[key] = cute.compile(cls(t_ids_ns, t_lat_ns, wait_first), *args, make_fake_stream())
    return _CACHE[key]


def build_ring_consumer():
    """Stand-in for the next AttnRes's ring consumer (``run_ring``): polls the first
    ``words`` 32-bit words of the MoE output ring row until none is the 0x80000000
    sentinel (relaxed sys-scope loads, as the MK polls), copies them to ``save`` and
    re-arms them. It never releases its dependents early, so the next layer's kernels
    start after it completes (the real chain orders this through the o-proj hop)."""
    P = _primitives()

    class RingConsumer:
        def __init__(self, words, deadline_ns=2_000_000_000):
            self.words, self.deadline_ns = words, deadline_ns

        @cute.jit
        def __call__(self, *args):
            P.launch(self.kernel, args, (self.words // 4 + 255) // 256, 256)

        @cute.kernel
        def kernel(self, args):
            ring, save = args
            tid, _, _, cta = P.thread()
            cute.arch.griddepcontrol_wait()
            v = cta * 256 + tid
            if v < self.words // 4:
                src = ring.iterator + 4 * v
                words = P.load_words(src, count=4, scope="sys")
                started, polls = Int64(0), Int32(0)
                if not P.published(words):
                    started = P.wait_deadline_start(self.deadline_ns)
                while not P.published(words):
                    polls = P.wait_deadline_check(started, polls, self.deadline_ns)
                    words = P.load_words(src, count=4, scope="sys")
                P.store_words(save.iterator + 4 * v, words)
                P.store_words(src, (Uint32(0x80000000),) * 4, scope="gpu")

    return RingConsumer


def ring_consumer(words, device_index):
    """Compiled consumer: call(ring i32[words], save i32[words], stream)."""
    key = ("ring", words, device_index)
    if key not in _CACHE:
        import torch
        from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import fake

        cls = build_ring_consumer()
        args = (fake((words,), None, torch.int32), fake((words,), None, torch.int32))
        with torch.accelerator.device_index(device_index):
            _CACHE[key] = cute.compile(cls(words), *args, make_fake_stream())
    return _CACHE[key]


KNOWN = ("MoeFrontTP8", "ClusterExpertsTP8Coarse", "ClusterExpertsTP8", "MoeSharedTP8", "MoeTailRep",
         "WaveMoeTail", "MoeTail", "RingConsumer", "SharedPushFull", "push_kernel", "Memcpy", "elementwise")


def kname(name):
    """Short kernel key for profiler summaries (CuTe names embed the class)."""
    for k in KNOWN:
        if k in name:
            return k
    return name[:60]


_PUSH_FULL = {}


def compile_push_full(batch, device, capacity=64, width=7168):
    """Test stand-in for a rank's publish of whole rows into its slot of every rank's
    [3, 8, capacity, width] multicast mailbox (-0 made +0): the harness uses it for the
    experts' latent publish (width 3584). Same kernel as an earlier opt-in
    ``tail_rep.SharedPushFull`` (not part of the shipped plugin), kept here as test code."""
    key = (batch, device, capacity, width)
    if key in _PUSH_FULL:
        return _PUSH_FULL[key]
    import torch
    from cutlass import Int32 as I32, Int64 as I64

    P = _primitives()
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import fake

    assert width % 896 == 0

    class PushFull:
        def __init__(self):
            self.batch, self.tp, self.width = batch, 8, width

        @cute.jit
        def __call__(self, *args):
            P.launch(self.kernel, args, self.batch, 112, pdl=False)

        @cute.kernel
        def kernel(self, args):
            shared, shared_mc, parity, rank, cap = args
            tid, _, _, token = P.thread()
            for i in cutlass.range_constexpr(self.width // 896):
                column = 8 * (tid + 112 * i)
                words = P.load_words(shared.iterator + token * self.width + column, count=4)
                packed = tuple(P.no_neg_zero(v) for v in words)
                offset = ((parity * self.tp + rank) * cap + token) * self.width + column
                P.multicast_words(shared_mc + I64(offset) * 2, packed)

    args = (fake((batch, width)), I64(0), I32(0), I32(0), I32(capacity))
    with torch.accelerator.device_index(device):
        code = cute.compile(PushFull(), *args, make_fake_stream())
    _PUSH_FULL[key] = code
    return code
