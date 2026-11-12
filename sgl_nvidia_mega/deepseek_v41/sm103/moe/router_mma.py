# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot gate_route_early: the router GEMM that SGLang's ``MoEGate`` runs at decode widths
(``tiny_gemm_bf16``'s tensor-core kernel, FP32 logits), rebuilt from ``src/router_mma.cu``
with its programmatic dependents released once its weight loads are issued (TensorRT-LLM's
low-M router GEMM's release point), so the router's top-k launches while the GEMM runs
instead of after it.

The schedule (warps, cluster, token tiles) and the row range are SGLang's own
(``_mma_schedule``, ``_mma_min_m``, ``MoEGate.tiny_router_gemm_max_tokens``), read for
the layer's gate; any call SGLang would serve on another kernel (fewer rows than the MMA
kernel takes, more rows than the tiny router GEMM takes, FP32 weights, deterministic
inference) runs ``MoEGate.forward``. ``EarlyRouter.build`` checks the logits against
``tiny_gemm_bf16`` bit for bit on the layer's weight at every served width and declines
otherwise.
"""

from __future__ import annotations

import ctypes
import os
from functools import cache
from pathlib import Path
from typing import Optional

import torch

from sgl_nvidia_mega.common.jit import compile_source as _compile_source
from sgl_nvidia_mega.common.jit import launch as _launch
from sgl_nvidia_mega.common.jit import load as _load
from sgl_nvidia_mega.common.jit import pointer as _pointer

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/ops/gemm/tiny_gemm.py": (
        "_mma_schedule",
        "_mma_min_m",
        "tiny_gemm_bf16",
        "num_warps",
        "cluster_size",
        "max_m_tiles",
    ),
    "srt/models/deepseek_v2.py": ("tiny_router_gemm_max_tokens", "MoEGate"),
    "srt/runtime_context.py": ("get_exec",),
}

SOURCE = Path(__file__).resolve().parent / "src" / "router_mma.cu"
# Where the one-launch top-k behind the early router releases GEMM1 (gate_route.cuh
# GATE_ROUTE_RELEASE): 1 = after its grid-dependency wait (default), 2 = at its completion
# (TensorRT-LLM's one-cluster route), 0 = at entry (gate_route's own release, which also
# launches GEMM1 early and lets its CTAs hold SMs while the router still runs)
RELEASE_ENV = "SGL_NVIDIA_MEGA_GATE_ROUTE_EARLY_RELEASE"


def gate_route_release() -> int:
    value = os.environ.get(RELEASE_ENV, "1")
    if value not in ("0", "1", "2"):
        raise ValueError(f"{RELEASE_ENV} must be 0, 1 or 2, got {value!r}")
    return int(value)


SYMBOL = "dsv41_router_mma"
# rows of one token tile: the plugin's decode widths (1-8 rows) take one tile per CTA
TILE_M = 8
MAX_ROWS = 8
TILE_N = 16


@cache
def load_router_mma(
    device_index: int,
    n: int,
    k: int,
    warps: int,
    cluster: int,
    early_release: bool = True,
):
    """The router GEMM for an [n, k] BF16 weight on SGLang's (warps, cluster) schedule, one
    token tile per CTA; ``early_release`` False keeps SGLang's release point (after the
    cluster reduction), for the kernel tests and the chain bench."""
    device = torch.device("cuda", device_index)
    with torch.cuda.device(device):
        torch.zeros(1, device=device)
        major, minor = torch.cuda.get_device_capability(device)
        options = (
            "-std=c++17",
            f"-DROUTER_N={n}",
            f"-DROUTER_K={k}",
            f"-DROUTER_WARPS={warps}",
            f"-DROUTER_CLUSTER={cluster}",
            "-DROUTER_MTILES=1",
            f"-DROUTER_EARLY_RELEASE={int(early_release)}",
        )
        name = f"router_mma_n{n}_k{k}_w{warps}_c{cluster}" + (
            "" if early_release else "_late"
        )
        cubin = _compile_source(
            SOURCE.read_text(), name, f"sm_{major}{minor}a", options
        )
        return _load(cubin, SYMBOL, (32 * warps, 1, 1), 0)


def sglang_schedule(n: int, k: int) -> Optional[tuple[int, int, int]]:
    """SGLang's tiny_mma schedule for an [n, k] BF16 weight as (warps, cluster, smallest
    MMA row count), or None when SGLang would never take its MMA kernel there."""
    from sglang.kernels.ops.gemm.tiny_gemm import _mma_min_m, _mma_schedule

    min_m = _mma_min_m(n, k, torch.bfloat16)
    if min_m > MAX_ROWS:
        return None
    schedule = _mma_schedule(n, k)
    return int(schedule.num_warps), int(schedule.cluster_size), int(min_m)


class EarlyRouter:
    """A layer's router GEMM on ``router_mma.cu`` at the widths SGLang's ``MoEGate`` would
    serve on its MMA kernel (``serves``), else None from ``build``."""

    def __init__(self, gate: torch.nn.Module, warps: int, cluster: int, min_rows: int):
        self.gate = gate
        self.weight = gate.weight
        self.n, self.k = (int(d) for d in gate.weight.shape)
        self.min_rows = min_rows
        self.max_rows = min(MAX_ROWS, int(gate.tiny_router_gemm_max_tokens))
        self.ctas = self.n // TILE_N * cluster
        self.cluster = cluster
        self.kernel = load_router_mma(
            gate.weight.device.index, self.n, self.k, warps, cluster
        )

    @staticmethod
    def why_not(gate: torch.nn.Module) -> str:
        """Why the layer's gate cannot take the early router ("" when it can)."""
        from sglang.srt.runtime_context import get_exec

        w = gate.weight
        if w.dtype != torch.bfloat16:
            return f"router weight {w.dtype}"
        if (
            w.dim() != 2
            or w.size(0) % TILE_N
            or w.size(1) % 64
            or not w.is_contiguous()
        ):
            return f"router weight {tuple(w.shape)}"
        if get_exec().deterministic.enable_deterministic_inference:
            return "deterministic inference routes on torch.mm"
        if sglang_schedule(int(w.size(0)), int(w.size(1))) is None:
            return "SGLang serves this router shape off its MMA kernel"
        return ""

    @classmethod
    def build(cls, gate: torch.nn.Module) -> tuple[Optional[EarlyRouter], str]:
        """The early router of ``gate`` and "" when it serves at least one width bit for
        bit with ``tiny_gemm_bf16``, else None and why."""
        why = cls.why_not(gate)
        if why:
            return None, why
        warps, cluster, min_rows = sglang_schedule(*gate.weight.shape)
        router = cls(gate, warps, cluster, min_rows)
        if router.min_rows > router.max_rows:
            return None, "no decode width takes SGLang's MMA router"
        why = router.check()
        return (None, why) if why else (router, "")

    def serves(self, x: torch.Tensor) -> bool:
        return (
            self.min_rows <= x.size(0) <= self.max_rows
            and x.dtype == torch.bfloat16
            and x.dim() == 2
            and x.size(1) == self.k
            and x.stride(1) == 1
            and x.stride(0) % 16 == 0
            and x.data_ptr() % 32 == 0
        )

    def __call__(
        self, x: torch.Tensor, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        rows = x.size(0)
        if out is None:
            out = torch.empty((rows, self.n), dtype=torch.float32, device=x.device)
        _launch(
            self.kernel,
            (self.ctas, 1, 1),
            [
                _pointer(x),
                _pointer(self.weight),
                _pointer(out),
                ctypes.c_int64(x.stride(0)),
                ctypes.c_uint32(rows),
            ],
            torch.cuda.current_stream(x.device).cuda_stream,
            cluster=(self.cluster, 1, 1) if self.cluster > 1 else None,
        )
        return out

    def check(self) -> str:
        """ "" when the logits equal ``tiny_gemm_bf16``'s bit for bit at every served width
        (the layer's weight, random rows with a wide exponent range), else why not."""
        from sglang.kernels.ops.gemm.tiny_gemm import tiny_gemm_bf16

        generator = torch.Generator(device=self.weight.device).manual_seed(41)
        for rows in range(self.min_rows, self.max_rows + 1):
            x = torch.randn(
                rows, self.k, device=self.weight.device, generator=generator
            ) * torch.exp2(
                torch.randint(
                    -8, 8, (rows, 1), device=self.weight.device, generator=generator
                ).float()
            )
            x = x.bfloat16()
            want = tiny_gemm_bf16(x, self.weight, out_dtype=torch.float32)
            got = self(x)
            torch.cuda.synchronize(self.weight.device)
            if not torch.equal(want.view(torch.int32), got.view(torch.int32)):
                return f"logits differ from tiny_gemm_bf16 at {rows} rows"
        return ""
