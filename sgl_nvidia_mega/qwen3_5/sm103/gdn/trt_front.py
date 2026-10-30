# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from TensorRT-LLM r3 (tekit 7df45abe5c), tensorrt_llm/_torch/modules/mamba/gdn_mixer.py
# (Qwen3NextGatedDeltaNet._lowm_in_proj and its fused-front caller).
"""The GatedDeltaNet verify front of one to four requests on TensorRT-LLM r3's
``lowm_gemm`` (``sm103/trt_lowm``): in one launch the static-FP8 ``in_proj_qkvz`` on the hop's E4M3 rows
(``fp8``, TensorRT-LLM's mixed8 tactic) or on the BF16 rows quantized in registers (``quant``, its mixed tactic,
where no E4M3 copy exists), the BF16 ``in_proj_ba`` on the BF16 rows, and the causal conv update of the q / k / v
channels as the epilogue. This wrapper admits the validated TP8 shapes.

Outputs: ``out`` [7, QKVZ] BF16 holds the convolved q | k | v then the raw z (the front's ``conv_out`` and
``projected[:, CONV:]`` in one buffer), ``ba`` [7, 2 HV] BF16 b then a; the conv state is updated at the request's
state index (none for a pad index, -1) and the windows after every token at intermediate row 0 (SGLang's
``verify_intermediate_state_indices`` gives batch row r row r). The sums run in TensorRT-LLM's order: numeric, not
SGLang's GEMMs bit for bit; the conv is the arithmetic of Triton's ``causal_conv1d_update``.

The launch is a programmatic dependent that reads only weights, scales and conv weights before its
``griddepcontrol.wait`` (the rows, the state index and the conv state after it) and releases its dependents once
those weight loads are issued; the next launch waits for it before reading any output. ``prepare`` compiles both
activation forms where the pools first exist (before graph capture); ``front`` refuses an unprepared one.
"""

from __future__ import annotations

import torch

MAX_REQUESTS = 4
ROWS, HIDDEN = 7, 4096
# TensorRT-LLM's _TACTICS mode of each activation form (lowm_linear.default_tactic)
MODES = {"fp8": "mixed8", "quant": "mixed"}

_SPECS: dict = {}  # (activation form, qkvz rows, requests) -> the prepared trt_lowm.Spec


def takes(requests: int) -> bool:
    """Whether ``requests`` is inside the front's validated request range."""
    return 1 <= requests <= MAX_REQUESTS


def spec(
    act: str, qkvz: int, ba: int, conv_weight, conv_state, windows, requests: int = 1
):
    """The ``trt_lowm`` launch of the front of ``requests`` requests of 7 rows with the ``act`` activations (``fp8`` /
    ``quant``) over a [qkvz, 4096] FP8 and a [ba, 4096] BF16 weight, on these conv tensors' layouts: TensorRT-LLM's
    tactic of the shape at M = 7 requests (BF16 segment: its k_warps and ksplit, one tile a CTA), as its GDN mixer
    passes it; request r's windows at row r."""
    from sgl_nvidia_mega.qwen3_5.sm103 import trt_lowm
    from sgl_nvidia_mega.qwen3_5.sm103.trt_lowm.lowm_linear import (
        default_tactic,
        lowm_conv_supported,
    )

    m = ROWS * requests
    tactic = default_tactic(m, qkvz, HIDDEN, mode=MODES[act])
    if not lowm_conv_supported(m, tactic, conv_weight.shape[0], ROWS):
        raise ValueError(
            f"TensorRT-LLM's {MODES[act]} tactic {tactic} has no conv epilogue for {m} x {qkvz}"
        )
    return trt_lowm.spec(
        m=m,
        k=HIDDEN,
        n_fp8=qkvz,
        n_bf16=ba,
        fp8_act=act,
        tactic=tactic,
        tactic_bf16=(tactic[0], 1, tactic[2] if len(tactic) > 2 else 1),
        split_out=qkvz,
        out_dtype=torch.bfloat16,
        conv=trt_lowm.conv_spec(
            conv_weight, conv_state, windows, requests=requests, tokens=ROWS
        ),
    )


def prepare(qkvz: int, ba: int, conv_weight, conv_state, windows) -> None:
    """Compile both activation forms through the request-window pool's capacity,
    capped at ``MAX_REQUESTS`` (not inside a graph capture)."""
    from sgl_nvidia_mega.qwen3_5.sm103 import trt_lowm

    max_requests = min(MAX_REQUESTS, int(windows.shape[0]))
    for requests in range(1, max_requests + 1):
        for act in MODES:
            s = spec(act, qkvz, ba, conv_weight, conv_state, windows, requests)
            trt_lowm.prepare(s)
            _SPECS[act, qkvz, requests] = s


def front(
    x: torch.Tensor,
    x_fp8,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    ba_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    windows: torch.Tensor,
    state_indices: torch.Tensor,
    out: torch.Tensor,
    ba: torch.Tensor,
) -> None:
    """The front of R requests' BF16 rows ``x`` [7 R, 4096] (and their E4M3 copy ``x_fp8``, quantized with
    ``input_scale``, or None) over ``weight`` [QKVZ, 4096] E4M3 (row-major [N, K]) and ``ba_weight`` [2 HV, 4096]
    BF16 into ``out`` [7 R, QKVZ] and ``ba`` [7 R, 2 HV]; ``conv_state`` [slots, C, 3] / ``windows`` [rows, 7, C, 3]
    updated at ``state_indices`` [R] int32 and intermediate rows 0..R-1."""
    from sgl_nvidia_mega.qwen3_5.sm103 import trt_lowm

    act = "fp8" if x_fp8 is not None else "quant"
    requests = x.shape[0] // ROWS
    s = _SPECS.get((act, weight.shape[0], requests))
    if s is None:
        raise RuntimeError(
            f"the TensorRT-LLM GDN front ({act}, {weight.shape[0]} rows, {requests} requests) was not prepared"
        )
    trt_lowm.run(
        s,
        out=out,
        out2=ba,
        x16=x,
        x8=x_fp8,
        w_fp8=weight,
        weight_scale=weight_scale,
        input_scale=input_scale,
        w_bf16=ba_weight,
        conv=(conv_weight, conv_state, state_indices, windows),
    )
