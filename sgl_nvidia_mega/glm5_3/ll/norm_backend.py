# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Whether ``ll.attention`` may fold the kv_a RMSNorm into its KV-half kernel.

The KV-half kernel reproduces the arithmetic of FlashInfer's CuTe DSL RMSNorm at
hidden 512 (the kernel SGLang's ``RMSNorm.forward_cuda`` reaches through
``sgl_kernel.rmsnorm`` on SM100-family GPUs). Folding is bitwise only where that is the
norm that runs, so the fold engages when both hold:

- FlashInfer's norm backend, rebuilt from its public surface (``fi_norm_backend``):
  the documented ``FLASHINFER_USE_CUDA_NORM`` and ``flashinfer.cute_dsl``'s public
  ``is_cute_dsl_available`` / ``is_cute_dsl_arch_supported``, as
  ``flashinfer.norm._use_cuda_norm`` decides (unchanged 0.7.1 -> 0.7.2), is CuTe DSL;
- a probe, run once at build before any capture (``probe_bits``): the stock norm
  module and the kernel's norm stage give the same bits on the step's own row counts,
  with rows laid out as the fused-A output holds them (strided) and order-sensitive
  magnitudes. It covers what the public rebuild cannot see (the CuTe DSL norm module
  failing to import, a reduction change in a FlashInfer or CuTe DSL upgrade).

Otherwise the stock norm keeps running; ``decide`` says which way and why, and
``ll.attention`` logs it once.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Iterable, Tuple

import torch

# the documented FlashInfer switch to its CUDA norm kernels (read at import of
# flashinfer.norm)
FLASHINFER_CUDA_NORM_ENV = "FLASHINFER_USE_CUDA_NORM"
CUTE_DSL = "cute-dsl"
CUDA = "cuda"

# A norm under test: (x [rows, hidden] bf16, possibly row-strided) -> [rows, hidden] bf16
NormFn = Callable[[torch.Tensor], torch.Tensor]


def fi_norm_backend(device) -> str:
    """The RMSNorm backend FlashInfer's ``rmsnorm`` dispatches to on ``device``:
    ``"cuda"`` or ``"cute-dsl"``."""
    import flashinfer.norm  # noqa: F401  (the env var is read at its import)
    from flashinfer.cute_dsl import is_cute_dsl_arch_supported, is_cute_dsl_available

    if os.environ.get(FLASHINFER_CUDA_NORM_ENV, "0") == "1":
        return CUDA
    if not is_cute_dsl_available():
        return CUDA
    try:
        supported = is_cute_dsl_arch_supported(
            *torch.cuda.get_device_capability(device)
        )
    except Exception:  # FlashInfer counts a failing probe as supported
        supported = True
    return CUTE_DSL if supported else CUDA


def probe_inputs(
    rows: Iterable[int],
    hidden: int,
    *,
    row_stride: int,
    col_offset: int,
    device,
    seed: int = 0,
) -> Iterable[torch.Tensor]:
    """Per row count, a bf16 ``[rows, hidden]`` view at column ``col_offset`` of a
    ``[rows, row_stride]`` buffer (how the kv_a latent sits in the fused-A output).
    Half of the rows have magnitudes spread over 2^-10..2^10 across columns, so the sum
    of squares depends on its association order; the rest are plain normals; one row is
    all zeros and one tiny (squares below the FP32 normal range)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    spread = torch.exp2(torch.linspace(-10.0, 10.0, hidden, device=device))
    spread = spread[torch.randperm(hidden, generator=gen, device=device)]
    for m in rows:
        buf = torch.randn(m, row_stride, generator=gen, device=device)
        x = buf[:, col_offset : col_offset + hidden]
        x[: (m + 1) // 2] *= spread
        if m >= 3:
            x[1] = 0.0
            x[2] *= 1e-21
        yield buf.bfloat16()[:, col_offset : col_offset + hidden]


def probe_bits(
    stock: NormFn,
    ours: NormFn,
    rows: Iterable[int],
    hidden: int,
    *,
    row_stride: int,
    col_offset: int,
    device,
) -> Tuple[bool, str]:
    """(equal, detail): the two norms on ``probe_inputs``, compared bit for bit."""
    rows = tuple(rows)
    for x in probe_inputs(
        rows, hidden, row_stride=row_stride, col_offset=col_offset, device=device
    ):
        a = stock(x).contiguous().view(torch.int16)
        b = ours(x).contiguous().view(torch.int16)
        if a.shape != b.shape:
            return False, f"shapes differ at {x.shape[0]} rows: {a.shape} / {b.shape}"
        diff = int((a != b).sum())
        if diff:
            return (
                False,
                f"{diff} of {a.numel()} bf16 values differ at {x.shape[0]} rows",
            )
    return True, f"bitwise at rows {list(rows)}"


@dataclass(frozen=True)
class NormFold:
    """The fold decision: ``fold`` and the reason, for the log."""

    fold: bool
    reason: str


def decide(
    stock: NormFn,
    ours: NormFn,
    rows: Iterable[int],
    hidden: int,
    *,
    row_stride: int,
    col_offset: int,
    device,
    backend: Callable[[object], str] = fi_norm_backend,
) -> NormFold:
    """Fold iff FlashInfer's norm backend is CuTe DSL and the probe is bitwise.
    Errors raise: a probe that cannot run is a build failure, not a decline."""
    which = backend(device)
    if which != CUTE_DSL:
        return NormFold(False, f"FlashInfer's norm backend is {which}")
    ok, detail = probe_bits(
        stock,
        ours,
        rows,
        hidden,
        row_stride=row_stride,
        col_offset=col_offset,
        device=device,
    )
    if not ok:
        return NormFold(False, f"probe against the stock norm: {detail}")
    return NormFold(True, f"FlashInfer norm backend {which}, probe {detail}")
