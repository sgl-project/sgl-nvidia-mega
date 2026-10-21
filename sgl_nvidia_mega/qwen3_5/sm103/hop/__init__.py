# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The decode hop (``src/hop.cu`` on the device protocol ``src/q35_hop.cuh``): the TP sum
of a verify step's bf16 partials with the residual add and RMSNorm (+ static FP8) for
hidden 4096, through one NVLink multicast mailbox per process. ``publish`` (or
``publish_fin``, which finalizes the routed MoE rows first) pushes this rank's rows;
``consume`` pulls every rank's, sums them in rank order, writes the residual and the norm
and re-arms the mailbox. The outputs are those of FlashInfer's MNNVL CuTe DSL low-latency
all-reduce + residual + RMSNorm, bit for bit.

A producer may push its partial itself, from its own epilogue: ``out_proj_push`` (the
GatedDeltaNet's static-FP8 ``out_proj``) does, and so may another package's kernel built
with ``headers()``, given ``Hop.site``, which then calls ``Hop.mark_pushed``. The next
all-reduce of that partial only consumes.

A consumer that waits past the deadline faults at address 0x0 (fail-stop) instead of
returning: the deadline is compiled in, as SGLang's ``BuildContext.deadline_s``.
"""

from __future__ import annotations

import ctypes
from functools import cache
from pathlib import Path

import torch

from sgl_nvidia_mega.common import jit

SOURCE = Path(__file__).parent / "src" / "hop.cu"
HEADER = Path(__file__).parent / "src" / "q35_hop.cuh"
HIDDEN, STAGES, ROW_CTAS, ROW_THREADS, QUAD_CTAS, MAX_TOP_K = 4096, 3, 4, 128, 8, 16
OPROJ_THREADS, OPROJ_MAX_M, OPROJ_STEPS = 256, 8, (4, 8)  # K = 256 * steps
SENTINEL = -32768  # bf16 -0.0 as int16: an empty mailbox slot
WORLDS = (2, 4, 8)


class Site(ctypes.Structure):
    _fields_ = [
        ("mc", ctypes.c_void_p),
        ("uc", ctypes.c_void_p),
        ("ctrl", ctypes.c_void_p),
        ("rank", ctypes.c_int),
        ("world", ctypes.c_int),
        ("m_cap", ctypes.c_int),
    ]


class Args(ctypes.Structure):
    _fields_ = (
        [("site", Site)]
        + [
            (name, ctypes.c_void_p)
            for name in (
                "partial",
                "routed",
                "expert_w",
                "permuted",
                "shared",
                "residual_in",
                "gamma",
                "fp8_scale",
                "residual_out",
                "norm_out",
                "fp8_out",
            )
        ]
        + [("m", ctypes.c_int), ("top_k", ctypes.c_int), ("eps", ctypes.c_float)]
    )


class OprojArgs(ctypes.Structure):
    _fields_ = (
        [("site", Site)]
        + [(name, ctypes.c_void_p) for name in ("w", "x", "sx", "sw", "out")]
        + [("m", ctypes.c_int)]
    )


def pack_out_proj(weight_nk: torch.Tensor) -> torch.Tensor:
    """[4096, K] e4m3 row-major -> the push kernel's mma.m16n8k32 A fragments:
    [cta 128][warp 8][k-step S][block 2][lane 32][16 B], K = 8 * S * 32."""
    w = weight_nk.contiguous().view(torch.uint8)
    n, k = w.shape
    steps = k // 256
    if n != HIDDEN or steps not in OPROJ_STEPS or k != 256 * steps:
        raise ValueError(
            f"out_proj packing needs [4096, 256 * S], got {tuple(w.shape)}"
        )
    v = w.view(n // 32, 2, 2, 8, 8, steps, 2, 4, 4)
    return v.permute(0, 4, 5, 1, 3, 7, 6, 2, 8).contiguous().view(-1)


def headers() -> dict[str, str]:
    """The NVRTC header of the hop's device protocol, for a kernel that pushes into the
    hop (``#include "q35_hop.cuh"``)."""
    return {HEADER.name: HEADER.read_text()}


@cache
def _kernels(device_index: int, deadline_ns: int) -> dict[str, jit.Kernel]:
    major, minor = torch.cuda.get_device_capability(device_index)
    cubin = jit.compile_source(
        SOURCE.read_text(),
        "q35_hop",
        f"sm_{major}{minor}a",
        ("-std=c++17", "-default-device", f"-DQ35_HOP_DEADLINE_NS={deadline_ns}ull"),
        headers(),
    )
    names = ["q35hop_publish", "q35hop_publish_fin_top10", "q35hop_publish_fin"]
    names += [f"q35hop_consume_w{w}" for w in WORLDS]
    kernels = {n: jit.load(cubin, n, (ROW_THREADS, 1, 1), 0) for n in names}
    for s in OPROJ_STEPS:
        name = f"q35hop_oproj_push_s{s}"
        kernels[name] = jit.load(cubin, name, (OPROJ_THREADS, 1, 1), 0)
    return kernels


class Hop:
    """One process's mailbox and control block, shared by every hop of every step;
    construct on every rank of ``group`` (a collective) before any capture."""

    def __init__(self, *, max_m: int, eps: float, group, device, deadline_s: float):
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symm_mem

        self.world = dist.get_world_size(group)
        if self.world not in WORLDS:
            raise ValueError(f"the hop serves TP {WORLDS}, not {self.world}")
        self.rank = dist.get_rank(group)
        self.max_m, self.eps, self.device = int(max_m), float(eps), torch.device(device)
        self.kernels = _kernels(self.device.index, int(deadline_s * 1e9))
        if symm_mem.get_backend(self.device) is None:
            # the backend FlashInfer's fusion workspace would select later in this process
            symm_mem.set_backend("NCCL")
        self._mailbox = symm_mem.empty(
            (STAGES, self.world, self.max_m, HIDDEN),
            dtype=torch.int16,
            device=self.device,
        )
        handle = symm_mem.rendezvous(self._mailbox, group)
        multicast = int(handle.multicast_ptr or 0)
        if not multicast or multicast % 16:
            raise RuntimeError("the hop needs an NVLink multicast mapping")
        self._mailbox.fill_(SENTINEL)
        self._ctrl = torch.zeros(4, dtype=torch.int32, device=self.device)
        self._handle = handle  # keeps the mapping alive
        # (address, shape) of a partial its producer already pushed; its all-reduce
        # only consumes
        self._pushed = None
        self.site = Site(
            multicast,
            self._mailbox.data_ptr(),
            self._ctrl.data_ptr(),
            self.rank,
            self.world,
            self.max_m,
        )
        torch.cuda.synchronize(self.device)
        dist.barrier(group=group)

    def supports(self, m: int) -> bool:
        return 1 <= m <= self.max_m

    def all_reduce_residual_rms_norm(
        self, *, local_contribution, residual, gamma, fp8_scale=None
    ):
        """(norm, residual_out[, fp8]) [M, 4096] of sum_ranks(partial) + residual; a
        partial out_proj_push returned is already in the mailbox."""
        m = int(local_contribution.shape[0])
        args, outs = self._args(m, residual, gamma, fp8_scale)
        if self.has_pushed(local_contribution):
            self._pushed = None
        else:
            self._check_consumed()
            args.partial = local_contribution.data_ptr()
            self._launch("q35hop_publish", args, ROW_CTAS)
        self._consume(args)
        return outs

    def out_proj_push(self, *, x_fp8, packed_w, input_scale, weight_scale):
        """The bf16 partial [M, 4096] of a static-FP8 out_proj on x_fp8 [M, K], also
        pushed into the mailbox: its all-reduce must come next."""
        m = int(x_fp8.shape[0])
        steps = packed_w.numel() // (HIDDEN * 256)
        if not 1 <= m <= min(OPROJ_MAX_M, self.max_m) or steps not in OPROJ_STEPS:
            raise ValueError(f"the out_proj push serves 1-{OPROJ_MAX_M} rows, not {m}")
        self._check_consumed()
        out = torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=self.device)
        args = OprojArgs(
            site=self.site,
            w=packed_w.data_ptr(),
            x=x_fp8.data_ptr(),
            sx=input_scale.data_ptr(),
            sw=weight_scale.data_ptr(),
            out=out.data_ptr(),
            m=m,
        )
        jit.launch(
            self.kernels[f"q35hop_oproj_push_s{steps}"],
            (HIDDEN // 32, 1, 1),
            [args],
            torch.cuda.current_stream(self.device).cuda_stream,
        )
        self.mark_pushed(out)
        return out

    def mark_pushed(self, partial: torch.Tensor) -> None:
        """Record that a producer kernel, launched on this stream after the last
        all-reduce, pushed the bf16 ``partial`` [M, 4096] into the mailbox (through
        ``site``): its all-reduce, which must come next, only consumes."""
        self._check_consumed()
        self._pushed = (partial.data_ptr(), partial.shape)

    def has_pushed(self, partial: torch.Tensor) -> bool:
        return self._pushed == (partial.data_ptr(), partial.shape)

    def _check_consumed(self):
        # a pushed partial left unconsumed would leave its mailbox stage dirty
        if self._pushed is not None:
            raise RuntimeError("the hop pushed a partial that was never consumed")

    def moe_finalize_all_reduce_rms_norm(
        self,
        *,
        routed_output,
        expert_weights,
        permuted_indices,
        gated_shared_output,
        residual,
        gamma,
        fp8_scale=None,
    ):
        """The same with this rank's partial the finalized MoE rows: per token the top-k
        routed rows weighted in k order, plus the gated shared output."""
        m, top_k = (int(n) for n in permuted_indices.shape)
        if top_k > MAX_TOP_K:
            raise ValueError(f"top-{top_k} routing (at most {MAX_TOP_K})")
        self._check_consumed()
        args, outs = self._args(m, residual, gamma, fp8_scale)
        args.routed, args.expert_w = routed_output.data_ptr(), expert_weights.data_ptr()
        args.permuted, args.top_k = permuted_indices.data_ptr(), top_k
        args.shared = jit.null_or_pointer(gated_shared_output).value
        name = "q35hop_publish_fin_top10" if top_k == 10 else "q35hop_publish_fin"
        self._launch(name, args, QUAD_CTAS)
        self._consume(args)
        return outs

    def _args(self, m: int, residual, gamma, fp8_scale):
        if not self.supports(m):
            raise ValueError(f"the hop serves 1-{self.max_m} rows, not {m}")
        shape = (m, HIDDEN)
        outs = [
            torch.empty(shape, dtype=torch.bfloat16, device=self.device),
            torch.empty(shape, dtype=torch.bfloat16, device=self.device),
        ]
        if fp8_scale is not None:
            outs.append(
                torch.empty(shape, dtype=torch.float8_e4m3fn, device=self.device)
            )
        args = Args(
            site=self.site,
            residual_in=residual.data_ptr(),
            gamma=gamma.data_ptr(),
            fp8_scale=jit.null_or_pointer(fp8_scale).value,
            residual_out=outs[1].data_ptr(),
            norm_out=outs[0].data_ptr(),
            fp8_out=outs[2].data_ptr() if fp8_scale is not None else None,
            m=m,
            eps=self.eps,
        )
        return args, tuple(outs)

    def _launch(self, name, args, ctas, cluster=None):
        jit.launch(
            self.kernels[name],
            (args.m, ctas, 1),
            [args],
            torch.cuda.current_stream(self.device).cuda_stream,
            cluster,
        )

    def _consume(self, args):
        self._launch(
            f"q35hop_consume_w{self.world}", args, ROW_CTAS, cluster=(1, ROW_CTAS, 1)
        )
