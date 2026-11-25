# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# The exchange's kernels (exchange.cu) are derived from muLLM v0.4.1's DeepSeek-V4.1
# Engram kernels (NVIDIA Corporation, Apache-2.0); exchange.cu names what it carries over.
"""Slot engram_publish: the Engram prefetch's WKV columns exchanged through tagged
mailboxes instead of the main-stream all-reduce.

Problem. Without the slot every rank computes its 1/world of each Engram layer's WKV
output columns on the Engram stream, zero-pads them, and the main stream sums all
layers' padded rows in one two-shot all-reduce at the first Engram layer. That puts
every Engram layer's WKV (layer 14's too) and the all-reduce on layer 1's path.

Responsibility. :class:`EngramExchange` keeps one mailbox region per Engram layer in
symmetric memory. ``publish`` (Engram stream, never waits) stores the rank's columns
as tagged words into every rank's mailbox; ``collect`` (main stream, at the layer's
gate) waits for every word of this step and writes the layer's [rows, n] BF16 rows.
Each column is computed by exactly one rank and the exchange copies BF16 bits, so the
rows equal SGLang's whole WKV GEMM bit for bit (the all-reduce's sum with zeros turns
a -0.0 into +0.0; the exchange keeps it). Every wait is bounded by the build's deadline.
"""

from __future__ import annotations

import ctypes
import functools
from pathlib import Path

import torch
import torch.distributed as dist

from sgl_nvidia_mega.common.jit import check as _check
from sgl_nvidia_mega.common.jit import compile_source as _compile_source
from sgl_nvidia_mega.common.jit import driver as _driver

SOURCE = Path(__file__).resolve().parent / "exchange.cu"
PUBLISH_SYMBOL = "dsv41_engram_publish"
COLLECT_SYMBOL = "dsv41_engram_collect"
NVRTC_OPTIONS = ("-std=c++17", "-default-device")
THREADS = 256  # exchange.cu kThreads
CONTROL_BYTES = 256  # epoch and arrivals, padded to an aligned slice
# collect: at most this many CTAs (one 4-column group a thread up to this grid)
MAX_COLLECT_CTAS = 256


@functools.cache
def _kernels(device_index: int) -> dict:
    cu = _driver()
    device = torch.device("cuda", device_index)
    major, minor = torch.cuda.get_device_capability(device)
    cubin = _compile_source(
        SOURCE.read_text(), "engram_exchange", f"sm_{major}{minor}a", NVRTC_OPTIONS
    )
    with torch.cuda.device(device):
        torch.zeros(1, device=device)  # make the primary context current
        (module,) = _check(cu.cuModuleLoadData(cubin), "cuModuleLoadData")
        return {
            name: _check(cu.cuModuleGetFunction(module, name.encode()), name)[0]
            for name in (PUBLISH_SYMBOL, COLLECT_SYMBOL)
        }


def _launch(function, grid: int, arguments: list, stream: int) -> None:
    """A plain launch (no programmatic serialization), graph-capturable."""
    cu = _driver()
    addresses = [ctypes.addressof(argument) for argument in arguments]
    params = (ctypes.c_void_p * len(addresses))(*addresses)
    _check(
        cu.cuLaunchKernel(
            function,
            grid,
            1,
            1,
            THREADS,
            1,
            1,
            0,
            cu.CUstream(stream),
            ctypes.addressof(params),
            0,
        ),
        "cuLaunchKernel",
    )


class EngramExchange:
    """Mailboxes of the Engram layers ``layer_ids``: [2 stages, max_rows, n] tagged u32
    words and two control words per layer, in one symmetric buffer per rank."""

    def __init__(
        self, group, layer_ids, *, n: int, max_rows: int, deadline_s: float, device
    ):
        if deadline_s <= 0:
            raise ValueError(f"the Engram exchange needs a deadline, got {deadline_s}")
        self.rank, self.world = group.rank_in_group, group.world_size
        if n % (4 * self.world):
            raise ValueError(f"{n} WKV columns do not split into 4-column groups")
        self.n, self.max_rows = n, max_rows
        self.timeout_ns = int(deadline_s * 1e9)
        payload = 2 * max_rows * n * 4
        region = payload + CONTROL_BYTES
        self.offsets = {
            layer: (j * region, j * region + payload)
            for j, layer in enumerate(layer_ids)
        }
        size = region * len(self.offsets)
        self.device = torch.device(device)
        self.buffer, pointers = self._allocate(group, size)
        self.peers = torch.tensor(pointers, dtype=torch.int64, device=self.device)
        self.kernels = _kernels(self.device.index)

    def _allocate(self, group, size: int):
        import torch.distributed._symmetric_memory as symm_mem

        buffer = symm_mem.empty(size, dtype=torch.uint8, device=self.device)
        handle = symm_mem.rendezvous(buffer, group.device_group)
        pointers = [
            handle.get_buffer(peer, (size,), torch.uint8, storage_offset=0).data_ptr()
            for peer in range(self.world)
        ]
        buffer.zero_()  # no tag matches zero; epoch 0 everywhere
        self._handle = handle
        torch.cuda.synchronize(self.device)
        dist.barrier(group=group.cpu_group)
        return buffer, pointers

    def publish(self, layer: int, columns: torch.Tensor) -> None:
        """Store this rank's ``columns`` [rows, n / world] (BF16, contiguous) of
        ``layer`` into every rank's mailbox, on the current stream."""
        rows, cols = columns.shape
        if cols * self.world != self.n or not 0 < rows <= self.max_rows:
            raise ValueError(
                f"publish takes [1-{self.max_rows}, {self.n // self.world}]"
            )
        if columns.dtype != torch.bfloat16 or not columns.is_contiguous():
            raise ValueError("publish takes contiguous BF16 columns")
        payload, control = self.offsets[layer]
        groups = rows * cols // 4
        grid = max(1, min((groups + THREADS - 1) // THREADS, MAX_COLLECT_CTAS))
        _launch(
            self.kernels[PUBLISH_SYMBOL],
            grid,
            [
                ctypes.c_void_p(columns.data_ptr()),
                ctypes.c_void_p(self.peers.data_ptr()),
                ctypes.c_int64(payload),
                ctypes.c_int64(control),
                ctypes.c_int(self.rank),
                ctypes.c_int(self.world),
                ctypes.c_int(rows),
                ctypes.c_int(self.n),
                ctypes.c_int(self.max_rows),
            ],
            torch.cuda.current_stream().cuda_stream,
        )

    def collect(self, layer: int, out: torch.Tensor) -> None:
        """Wait for every rank's columns of ``layer`` and write them into ``out``
        [rows, n] (BF16, contiguous), on the current stream; advances the epoch."""
        rows, n = out.shape
        if n != self.n or not 0 < rows <= self.max_rows:
            raise ValueError(f"collect writes [1-{self.max_rows}, {self.n}]")
        if out.dtype != torch.bfloat16 or not out.is_contiguous():
            raise ValueError("collect writes contiguous BF16 rows")
        payload, control = self.offsets[layer]
        groups = rows * n // 4
        grid = max(1, min((groups + THREADS - 1) // THREADS, MAX_COLLECT_CTAS))
        _launch(
            self.kernels[COLLECT_SYMBOL],
            grid,
            [
                ctypes.c_void_p(self.peers.data_ptr()),
                ctypes.c_int64(payload),
                ctypes.c_int64(control),
                ctypes.c_int(self.rank),
                ctypes.c_int(rows),
                ctypes.c_int(self.n),
                ctypes.c_int(self.max_rows),
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_int64(self.timeout_ns),
            ],
            torch.cuda.current_stream().cuda_stream,
        )
