# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import ctypes
from functools import cache
from pathlib import Path
from typing import Optional

import torch

from sgl_nvidia_mega.common import jit
from sgl_nvidia_mega.qwen3_5.sm103.hop import HIDDEN, headers

SOURCE = Path(__file__).parent / "src" / "bt_tail.cu"
THREADS = HIDDEN // 8
STAGES = 3  # the protocol's prenorm generations
ENTRIES = {False: "q35bt_tail", True: "q35bt_tail_early"}


class TailArgs(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_void_p)
        for name in (
            "prenorm",
            "stage_state",
            "gamma",
            "fp8_scale",
            "residual_out",
            "norm_out",
            "fp8_out",
        )
    ] + [("m_cap", ctypes.c_int), ("eps", ctypes.c_float)]


@cache
def kernels(device_index: int) -> dict[bool, jit.Kernel]:
    """The tail's two trigger variants, compiled for ``device_index`` (at build)."""
    major, minor = torch.cuda.get_device_capability(device_index)
    cubin = jit.compile_source(
        SOURCE.read_text(),
        "q35_bt_tail",
        f"sm_{major}{minor}a",
        ("-std=c++17", "-default-device"),
        headers(),
    )
    return {
        early: jit.load(cubin, name, (THREADS, 1, 1), 0)
        for early, name in ENTRIES.items()
    }


def tail(
    compiled: dict[bool, jit.Kernel],
    *,
    prenorm: torch.Tensor,
    stage_state: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
    m: int,
    residual_out: torch.Tensor,
    norm_out: torch.Tensor,
    fp8_scale: Optional[torch.Tensor] = None,
    fp8_out: Optional[torch.Tensor] = None,
    early: bool = False,
) -> None:
    if (fp8_scale is None) != (fp8_out is None):
        raise ValueError("the FP8 output needs both its scale and its tensor")
    stages, m_cap, hidden = prenorm.shape
    if stages != STAGES or hidden != HIDDEN or not 1 <= m <= m_cap:
        raise ValueError(f"prenorm mailbox {tuple(prenorm.shape)} for {m} rows")
    args = TailArgs(
        prenorm=prenorm.data_ptr(),
        stage_state=stage_state.data_ptr(),
        gamma=gamma.data_ptr(),
        fp8_scale=jit.null_or_pointer(fp8_scale).value,
        residual_out=residual_out.data_ptr(),
        norm_out=norm_out.data_ptr(),
        fp8_out=jit.null_or_pointer(fp8_out).value,
        m_cap=m_cap,
        eps=eps,
    )
    jit.launch(
        compiled[early],
        (m, 1, 1),
        [args],
        torch.cuda.current_stream(prenorm.device).cuda_stream,
    )
