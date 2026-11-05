# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unpooler projection pushed straight to every rank, fused into the FFN boundary (``csrc/moe_tail.cuh``).

Replaces, for the MoE tail of a Muse Spark decoder layer at decode/verify row counts, the SGLang pair
``gemm_local`` (column slice of the unpooler) + ``all_gather_norm_residual`` (and, above SGLang's small-projection row
limit, the replicated unpooler GEMM + ``fused_norm_residual``) with two kernels:

* ``push``: each rank multiplies the all-reduced expert output ``y`` [M, K] by its row block of the unpooler weight
  (staged in shared memory before the kernel waits on its predecessor), rounds to bf16 and stores the columns into
  every rank's receive buffer.
* ``norm``: SGLang's cluster boundary arithmetic (post-norm, gated residual, pre-norm) on the gathered row, after
  every rank's columns arrived.

Hand-off between them (``TailWorkspace(tagged=...)``, see ``csrc/moe_tail.cuh``): arrival counters with
release / acquire at system scope (default), or tagged 32-bit words without fences (``tagged=True``; relies on
ordering the PTX memory model does not guarantee for buffer reuse across calls).

The projection accumulates in fp32 with a different summation order than SGLang's GEMM, so its bf16 output can
differ in the last bit; the boundary arithmetic is identical. See ``tests/dist_moe_tail.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch

from . import sgl_compat

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": (
        "is_arch_support_pdl",
        "load_jit",
        "make_cpp_args",
    ),
    "kernels/jit/utils/compile/paths.py": ("KERNEL_PATH",),
    "kernels/ops/layernorm/muse_spark_norm_residual_jit.py|kernels/ops/muse_spark_v12/norm_residual.py": (
        "norm_residual_launch_config",
    ),
    "srt/distributed/device_communicators/custom_all_reduce_v2.py": (
        "_allocate_symmetric_memory",
    ),
}


_CUH = Path(__file__).resolve().parent / "csrc" / "moe_tail.cuh"
ROWS_PER_CTA = 16
MAX_WORLD = 8
MAX_ROWS = 64
_NT_CHOICES = (1, 2, 4, 8)
_COUNTER_BYTES = 256


def _cache(fn):
    from functools import lru_cache

    return lru_cache(maxsize=None)(fn)


@_cache
def _push_module(hidden: int, k: int, world: int, nt: int, tagged: bool, fuzz: bool):
    from sglang.kernels.jit.utils import is_arch_support_pdl, load_jit, make_cpp_args
    from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH

    args = make_cpp_args(hidden, k, world, nt, is_arch_support_pdl(), tagged, fuzz)
    return load_jit(
        "muse_vr_moe_tail_push",
        *args,
        cuda_files=[str(_CUH)],
        cuda_wrappers=[("run", f"MuseVrMoeTailPush<{args}>::run")],
        extra_include_paths=sgl_compat.csrc_include_paths(KERNEL_PATH / "csrc"),
    )


@_cache
def _norm_module(
    hidden: int,
    world: int,
    threads: int,
    cluster: int,
    scale: bool,
    post_w: bool,
    gate: bool,
    tagged: bool,
    fuzz: bool,
):
    from sglang.kernels.jit.utils import is_arch_support_pdl, load_jit, make_cpp_args
    from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH

    args = make_cpp_args(
        hidden,
        world,
        threads,
        cluster,
        is_arch_support_pdl(),
        scale,
        post_w,
        gate,
        tagged,
        fuzz,
    )
    return load_jit(
        "muse_vr_moe_tail_norm",
        *args,
        cuda_files=[str(_CUH)],
        cuda_wrappers=[("run", f"MuseVrMoeTailNorm<{args}>::run")],
        extra_include_paths=sgl_compat.csrc_include_paths(KERNEL_PATH / "csrc"),
    )


def norm_launch_config(hidden: int, num_rows: int) -> tuple[int, int]:
    """(cluster size, threads) exactly as SGLang's boundary kernels pick them for this shape."""
    from sglang.kernels.ops.muse_spark_v12.norm_residual import (
        norm_residual_launch_config,
    )

    cluster, threads, _ = norm_residual_launch_config(
        hidden_size=hidden, num_tokens=num_rows
    )
    return cluster, threads


def _fuzz() -> bool:
    """Test-only: random delays in both kernels (tests/stress_moe_tail.py). Off in production."""
    import os

    return os.environ.get("MUSE_VR_MOE_TAIL_TEST_FUZZ", "0") == "1"


def _nt_for(num_rows: int) -> int:
    for nt in _NT_CHOICES:
        if num_rows <= nt * 8:
            return nt
    raise ValueError(f"row count {num_rows} above {MAX_ROWS}")


class TailWorkspace:
    """Symmetric receive buffers + arrival counters on one tensor-parallel group, allocated before graph capture.

    Every rank of ``group`` must construct it at the same point (collective rendezvous). The receive region holds
    two halves of ``max_rows * hidden`` bf16; the counter block holds one uint32 per source rank.
    """

    def __init__(
        self,
        group,
        device: torch.device,
        rank: int,
        world: int,
        hidden: int,
        k: int,
        max_rows: int = MAX_ROWS,
        tagged: bool = False,
    ):
        import torch.distributed as dist
        from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
            _allocate_symmetric_memory,
        )

        if not 2 <= world <= MAX_WORLD:
            raise ValueError("world size outside 2..8")
        if hidden % (world * ROWS_PER_CTA) or k % 256:
            raise ValueError("unsupported projection shape")
        self.rank, self.world, self.hidden, self.k = rank, world, hidden, k
        self.max_rows = max_rows
        self.tagged = bool(tagged)
        self.n_local = hidden // world
        self.push_grid = self.n_local // ROWS_PER_CTA
        self.parity_bytes = (
            max_rows * hidden * 4
        )  # room for the tagged layout (one 32-bit word per value)
        total = 2 * self.parity_bytes + _COUNTER_BYTES
        self._tensor, symm = _allocate_symmetric_memory(
            total, device=device, group=group
        )
        slabs = [symm.get_buffer(i, [total], torch.uint8) for i in range(world)]
        slabs[rank].zero_()
        self.seq = torch.zeros(
            self.push_grid, dtype=torch.int32, device=device
        )  # used as uint32 counters
        self.done = torch.zeros(1, dtype=torch.int32, device=device)
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        recv = [s.data_ptr() for s in slabs]
        arrive = [s.data_ptr() + 2 * self.parity_bytes for s in slabs]
        self.ptrs = torch.tensor(
            recv + arrive, dtype=torch.int64, device="cpu"
        )  # host table, read at launch
        self._slabs = slabs  # keep the peer views alive

    def counters(self) -> torch.Tensor:
        """This rank's arrival counters (debug)."""
        off = 2 * self.parity_bytes
        return self._slabs[self.rank][off : off + 4 * self.world].view(torch.int32)


def supported(ws: Optional[TailWorkspace], num_rows: int, hidden: int, k: int) -> bool:
    if ws is None or not 1 <= num_rows <= min(ws.max_rows, MAX_ROWS):
        return False
    if hidden != ws.hidden or k != ws.k:
        return False
    try:
        cluster, _ = norm_launch_config(hidden, num_rows)
    except (RuntimeError, ValueError):
        return False
    return cluster >= 2


def push(ws: TailWorkspace, y: torch.Tensor, weight: torch.Tensor) -> None:
    """Kernel 1: this rank's columns of ``y @ weight.T`` into every rank's receive buffer. ``weight`` is the full
    replicated [hidden, k] unpooler weight; only rows ``rank * n_local .. +n_local`` are read."""
    m = y.shape[0]
    w_local = weight[ws.rank * ws.n_local : (ws.rank + 1) * ws.n_local]
    _push_module(ws.hidden, ws.k, ws.world, _nt_for(m), ws.tagged, _fuzz()).run(
        y, w_local, ws.ptrs, ws.seq, ws.done, ws.rank, ws.parity_bytes
    )


def norm(
    ws: TailWorkspace,
    stream: torch.Tensor,
    post_eps: float,
    pre_eps: float,
    *,
    post_norm_is_input_scale: bool,
    post_weight: Optional[torch.Tensor],
    pre_weight: Optional[torch.Tensor],
    alpha: Optional[torch.Tensor],
    beta: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Kernel 2: boundary on the gathered rows. Returns (bf16 next-layer input, fp32 new stream)."""
    m, hidden = stream.shape
    cluster, threads = norm_launch_config(hidden, m)
    output = torch.empty(m, hidden, dtype=torch.bfloat16, device=stream.device)
    new_stream = torch.empty_like(stream)
    mod = _norm_module(
        hidden,
        ws.world,
        threads,
        cluster,
        bool(post_norm_is_input_scale),
        post_weight is not None,
        alpha is not None,
        ws.tagged,
        _fuzz(),
    )
    mod.run(
        ws.ptrs,
        ws.seq,
        ws.rank,
        ws.parity_bytes,
        stream,
        output,
        new_stream,
        post_weight,
        pre_weight,
        alpha,
        beta,
        float(post_eps),
        float(pre_eps),
    )
    return output, new_stream


def moe_tail(
    ws: TailWorkspace,
    y: torch.Tensor,
    weight: torch.Tensor,
    stream: torch.Tensor,
    post_eps: float,
    pre_eps: float,
    *,
    post_norm_is_input_scale: bool = True,
    post_weight: Optional[torch.Tensor] = None,
    pre_weight: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert (
        y.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and stream.dtype == torch.float32
    )
    assert y.is_contiguous() and weight.is_contiguous() and stream.is_contiguous()
    push(ws, y, weight)
    return norm(
        ws,
        stream,
        post_eps,
        pre_eps,
        post_norm_is_input_scale=post_norm_is_input_scale,
        post_weight=post_weight,
        pre_weight=pre_weight,
        alpha=alpha,
        beta=beta,
    )
