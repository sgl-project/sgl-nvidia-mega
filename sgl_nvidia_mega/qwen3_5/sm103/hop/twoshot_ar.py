# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from TensorRT-LLM (Apache-2.0): the two-shot all-reduce's geometry and Lamport table of
# tensorrt_llm/_torch/kernels/blackwell/gemm_twoshot_allreduce_norm.py.

from __future__ import annotations

import torch

_FLAGS_BYTES = 256


def ctas_per_token(rows: int, hidden: int, sms: int) -> int:
    """CTAs per token as the library one-shot's ``adjustGridConfig`` picks them."""
    threads = hidden // 8
    ctas = 8
    while ctas > 1 and (
        threads // ctas < 64 or (threads // ctas) % 32 or rows * ctas > sms
    ):
        ctas //= 2
    return ctas


def tokens_per_cta(rows: int, ctas: int, sms: int) -> int:
    if ctas != 1:
        return 1
    wave = 2 * sms
    return 1 if rows <= wave else -(-rows // wave)


class TwoShotWorkspace:
    def __init__(self, *, hidden: int, eps: float, group, device, max_rows: int):
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symm_mem

        from sgl_nvidia_mega.qwen3_5.runtime import SlotUnavailable, require_multicast

        self.hidden, self.eps, self.max_rows = int(hidden), float(eps), int(max_rows)
        self.device = torch.device(device)
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        if (
            self.hidden % 256
            or self.hidden // 8 > 1024
            or self.world not in (2, 4, 8, 16)
        ):
            raise ValueError(
                f"two-shot workspace: hidden {hidden} / world {self.world}"
            )
        self.sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        row_bytes = self.hidden * 2
        self.buf_bytes = 2 * (-(-self.max_rows // self.world)) * self.world * row_bytes
        self.flags_off = 3 * self.buf_bytes
        if symm_mem.get_backend(self.device) is None:
            symm_mem.set_backend("NCCL")
        self._buf = symm_mem.empty(
            ((self.flags_off + _FLAGS_BYTES) // 4,),
            dtype=torch.float32,
            device=self.device,
        )
        self._handle = symm_mem.rendezvous(self._buf, group)
        mcast = int(self._handle.multicast_ptr or 0)
        require_multicast("gemm_ar", mcast, group, self.device)
        peers = [int(p) for p in (self._handle.buffer_ptrs or [])]
        ok = torch.tensor(
            [len(peers) == self.world and all(p and p % 16 == 0 for p in peers)],
            dtype=torch.int32,
            device=self.device,
        )
        dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
        if not int(ok.item()):
            raise SlotUnavailable(
                "gemm_ar", "needs every rank's unicast mapping (buffer_ptrs)"
            )
        self._buf.fill_(-0.0)
        flags = self._buf.view(torch.int32)[self.flags_off // 4 :]
        flags.zero_()  # current buffer 0, nothing dirty, arrival counter 0
        flags[2] = self.buf_bytes
        own = self._buf.data_ptr()
        self.table = torch.tensor(
            [own, mcast, own + self.flags_off] + peers,
            dtype=torch.int64,
            device=self.device,
        )
        torch.cuda.synchronize(self.device)
        # every rank has cleared its buffers before any rank can write into them
        dist.barrier(group=group)

    def flags(self) -> list[int]:
        return self._buf.view(torch.int32)[self.flags_off // 4 :][:9].tolist()
