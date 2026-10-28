# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0): python/sglang/srt/layers/q35_hop.py
# (Q35Hop.draft_all_reduce_rms_norm) and python/sglang/kernels/ops/speculative/mkd_draft_gemv.py,
# on branch nvidia/qwen35-agentx-gb300.
"""The MTP draft's hops on the decode hop's mailbox (``sm103/hop``), and its o_proj GEMV.

``draft_all_reduce_rms_norm`` pushes this rank's partial with the decode hop's
``q35hop_publish`` (skipped when a producer already pushed it) and consumes with
``src/draft_hop.cu``, which keeps the rounding of the kernel the draft step replaces:
``add_norm=False`` FlashInfer's legacy MNNVL one-shot all-reduce + residual + RMSNorm
(``gamma`` the applied multiplier), ``add_norm=True`` SGLang's one-shot push all-reduce,
then FlashInfer's CuTe DSL FusedAddRMSNorm (multiplier ``weight_bias + gamma``).
``draft_gemv`` (``src/draft_gemv.cu``) is the one-row bf16 GEMV with fp32 accumulation;
with ``hop_site`` it also pushes its output into the hop, whose all-reduce then only
consumes; with ``gate`` it takes ``x * sigmoid(gate)`` as its input, in the arithmetic of
SGLang's attention output gate kernel.

Kernels compile in ``build`` (never under CUDA-graph capture); the consume's wait
deadline is compiled in, as the decode hop's.
"""

from __future__ import annotations

import ctypes
from functools import cache
from pathlib import Path
from typing import Optional

import torch

from sgl_nvidia_mega.common import jit
from sgl_nvidia_mega.qwen3_5.sm103 import hop as decode_hop

SOURCE = Path(__file__).parent / "src" / "draft_hop.cu"
GEMV_SOURCE = Path(__file__).parent / "src" / "draft_gemv.cu"
HIDDEN = decode_hop.HIDDEN
GEMV_ROWS, GEMV_PUSH_ROWS = 4, 8  # warps (rows in flight) per CTA


class Args(ctypes.Structure):
    """q35hop_draft::Args: the decode hop's Args with the draft consume's fields."""

    _fields_ = list(decode_hop.Args._fields_) + [
        ("weight_bias", ctypes.c_float),
        ("delay_ns", ctypes.c_int),
    ]


class GemvParams(ctypes.Structure):
    """q35draft_gemv::Params."""

    _fields_ = [
        ("x", ctypes.c_void_p),
        ("gate", ctypes.c_void_p),
        ("w", ctypes.c_void_p),
        ("y", ctypes.c_void_p),
        ("n", ctypes.c_int),
        ("hop", decode_hop.Site),
        ("hop_token", ctypes.c_int),
    ]


def consume_name(world: int, add_norm: bool, early: bool) -> str:
    mode = "add_norm" if add_norm else "mnnvl"
    return f"q35hop_draft_{mode}{'_early' if early else ''}_w{world}"


@cache
def _consume_kernels(device_index: int, deadline_ns: int) -> dict[str, jit.Kernel]:
    major, minor = torch.cuda.get_device_capability(device_index)
    cubin = jit.compile_source(
        SOURCE.read_text(),
        "q35_draft_hop",
        f"sm_{major}{minor}a",
        ("-std=c++17", "-default-device", f"-DQ35_HOP_DEADLINE_NS={deadline_ns}ull"),
    )
    names = [
        consume_name(w, add_norm, early)
        for w in decode_hop.WORLDS
        for add_norm in (False, True)
        for early in (False, True)
    ]
    return {n: jit.load(cubin, n, (decode_hop.ROW_THREADS, 1, 1), 0) for n in names}


@cache
def _gemv_kernels(
    device_index: int, k: int, gate: bool = False
) -> tuple[jit.Kernel, jit.Kernel, int]:
    """(plain, pushing, the plain kernel's grid cap: resident CTAs on the device), the
    gated twins with ``gate``."""
    if k % 256:
        raise ValueError(f"draft GEMV: K {k} is not a multiple of 256")
    major, minor = torch.cuda.get_device_capability(device_index)
    cubin = jit.compile_source(
        GEMV_SOURCE.read_text(),
        f"q35_draft_gemv_k{k}",
        f"sm_{major}{minor}a",
        ("-std=c++17", "-default-device", f"-DQ35_DRAFT_GEMV_K={k}"),
    )
    suffix = "_gate" if gate else ""
    plain = jit.load(cubin, f"q35_draft_gemv{suffix}", (GEMV_ROWS * 32, 1, 1), 0)
    push = jit.load(
        cubin, f"q35_draft_gemv_push{suffix}", (GEMV_PUSH_ROWS * 32, 1, 1), 0
    )
    cu = jit.driver()
    (per_sm,) = jit.check(
        cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            plain.function, GEMV_ROWS * 32, 0
        ),
        "cuOccupancyMaxActiveBlocksPerMultiprocessor",
    )
    sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    return plain, push, int(per_sm) * sms


# device index -> the consume kernels built for it
_CONSUME: dict[int, dict[str, jit.Kernel]] = {}


def build(device_index: int, deadline_ns: int, gemv_ks: tuple[int, ...] = ()) -> None:
    """Compile the consume kernels (with the decode hop's wait deadline) and the GEMV
    for each reduction length in ``gemv_ks``."""
    _CONSUME[device_index] = _consume_kernels(device_index, int(deadline_ns))
    for k in gemv_ks:
        for gate in (False, True):
            _gemv_kernels(device_index, int(k), gate)


def draft_all_reduce_rms_norm(
    hop,
    *,
    local_contribution: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
    weight_bias: float,
    add_norm: bool,
    early_release: bool = False,
    delay_ns: int = 0,
    pushed: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(norm, residual_out) [M, 4096] bf16 of sum_ranks(partial) + residual on ``hop``
    (an ``sm103.hop.Hop``). ``pushed``: a producer already pushed this partial, so only
    the consume runs. ``early_release``: the consume releases its dependents at entry
    (they must read only static data before their PDL wait)."""
    m = int(local_contribution.shape[0])
    shape = (m, HIDDEN)
    if (
        not hop.supports(m)
        or tuple(residual.shape) != shape
        or tuple(gamma.shape) != (HIDDEN,)
        or residual.dtype != torch.bfloat16
        or gamma.dtype != torch.bfloat16
        or not (residual.is_contiguous() and gamma.is_contiguous())
    ):
        raise ValueError(
            f"draft hop: rows {m} (1-{hop.max_m}), residual {residual.dtype} "
            f"{tuple(residual.shape)}, gamma {gamma.dtype} {tuple(gamma.shape)}"
        )
    stream = torch.cuda.current_stream(hop.device).cuda_stream
    norm = torch.empty(shape, dtype=torch.bfloat16, device=hop.device)
    residual_out = torch.empty(shape, dtype=torch.bfloat16, device=hop.device)
    if not pushed:
        if (
            tuple(local_contribution.shape) != shape
            or local_contribution.dtype != torch.bfloat16
            or not local_contribution.is_contiguous()
        ):
            raise ValueError(
                f"draft hop: partial {local_contribution.dtype} "
                f"{tuple(local_contribution.shape)}"
            )
        publish = decode_hop.Args(
            site=hop.site, partial=local_contribution.data_ptr(), m=m, eps=eps
        )
        jit.launch(
            hop.kernels["q35hop_publish"],
            (m, decode_hop.ROW_CTAS, 1),
            [publish],
            stream,
        )
    args = Args(
        site=hop.site,
        residual_in=residual.data_ptr(),
        gamma=gamma.data_ptr(),
        residual_out=residual_out.data_ptr(),
        norm_out=norm.data_ptr(),
        m=m,
        eps=eps,
        weight_bias=weight_bias,
        delay_ns=delay_ns,
    )
    kernel = _CONSUME[hop.device.index][
        consume_name(hop.world, add_norm, early_release)
    ]
    jit.launch(
        kernel,
        (m, decode_hop.ROW_CTAS, 1),
        [args],
        stream,
        cluster=(1, decode_hop.ROW_CTAS, 1),
    )
    return norm, residual_out


def draft_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    hop_site: Optional[decode_hop.Site] = None,
    hop_token: int = 0,
    gate: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``out`` [1, N] bf16 = ``x`` [1, K] @ ``weight`` [N, K]^T with fp32 accumulation,
    one warp per row. ``hop_site``: the decode hop's ``Site``; the output (N = 4096) is
    then also pushed into that hop as row ``hop_token``, and its all-reduce only
    consumes. ``gate`` [1, K] bf16: the input is then ``x`` * sigmoid(``gate``),
    rounded to bf16."""
    k, n = int(x.shape[-1]), int(weight.shape[0])
    if (
        x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or out.dtype != torch.bfloat16
        or tuple(x.shape) != (1, k)
        or tuple(weight.shape) != (n, k)
        or tuple(out.shape) != (1, n)
        or not (x.is_contiguous() and weight.is_contiguous() and out.is_contiguous())
        or x.data_ptr() % 16
        or weight.data_ptr() % 16
    ):
        raise ValueError(
            f"draft GEMV: x {x.dtype} {tuple(x.shape)}, weight {weight.dtype} "
            f"{tuple(weight.shape)}, out {out.dtype} {tuple(out.shape)}"
        )
    if gate is not None and (
        gate.dtype != torch.bfloat16
        or tuple(gate.shape) != (1, k)
        or not gate.is_contiguous()
        or gate.data_ptr() % 16
    ):
        raise ValueError(f"draft GEMV: gate {gate.dtype} {tuple(gate.shape)}")
    plain, push, resident = _gemv_kernels(x.device.index, k, gate is not None)
    params = GemvParams(
        x=x.data_ptr(),
        gate=gate.data_ptr() if gate is not None else x.data_ptr(),
        w=weight.data_ptr(),
        y=out.data_ptr(),
        n=n,
        hop=hop_site if hop_site is not None else decode_hop.Site(),
        hop_token=hop_token,
    )
    if hop_site is not None:
        if n != HIDDEN or not 0 <= hop_token < hop_site.m_cap:
            raise ValueError(
                f"draft GEMV: a pushed partial is one hidden row of {HIDDEN} at a "
                f"hop row below {hop_site.m_cap}, not {n} at row {hop_token}"
            )
        kernel, grid = push, (n + GEMV_PUSH_ROWS - 1) // GEMV_PUSH_ROWS
    else:
        # rows beyond one resident wave loop inside the warps
        kernel, grid = plain, min((n + GEMV_ROWS - 1) // GEMV_ROWS, resident)
    jit.launch(
        kernel, (grid, 1, 1), [params], torch.cuda.current_stream(x.device).cuda_stream
    )
    return out
