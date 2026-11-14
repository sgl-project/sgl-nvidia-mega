# SPDX-License-Identifier: Apache-2.0
# From SGLang python/sglang/kernels/ops/elementwise/qwen4_gate.py: the section that commit added (the prefill gate after the
# reduction), unchanged; the shape constants it shares with the rest of that file are
# imported from SGLang main's qwen4_gate.py, which defines them identically.
from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import triton
import triton.language as tl
from sglang.kernels.ops.elementwise.qwen4_gate import (
    _QWEN4_HC_COUNT,
    _QWEN4_HIDDEN_SIZE,
)
from triton.language.extra.cuda import libdevice

logger = logging.getLogger(__name__)

SGLANG_READS = {
    "kernels/ops/elementwise/qwen4_gate.py": ("_QWEN4_HC_COUNT", "_QWEN4_HIDDEN_SIZE"),
}


# ---- Prefill gate after the reduction (SGLANG_OPT_PLE_PREFILL_FUSED_GATE) ----
# Qwen4ExpPLELayer's eager gate, from the BF16 reduction on:
#     gate = gate / math.sqrt(2560)                              # x * f32(1 / f32(sqrt 2560))
#     gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
#     gate = torch.sigmoid(gate)                                 # 1 / (1 + expf(-x)), IEEE divide
#     gated_value = gate * value.unsqueeze(-2)
# is eight launches, the last a broadcast multiply writing [T, 4, 2560]. One kernel replays every
# step in FP32 with each eager BF16 rounding (cvt.rn), IEEE sqrt and divide, libdevice expf and no
# FMA contraction. The reduction stays torch's. The first use runs a self-test over all 65,536
# BF16 gate inputs against the eager chain and keeps eager on any mismatch.


@triton.jit
def _bf16(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _qwen4_gate_post_kernel(
    gate_ptr,
    value_ptr,
    output_ptr,
    scale,
    lower,
    HC_COUNT: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    DIVIDE: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(value_ptr + token * HIDDEN_SIZE + cols).to(tl.float32)
    for h in tl.static_range(HC_COUNT):
        g = tl.load(gate_ptr + token * HC_COUNT + h).to(tl.float32)
        if DIVIDE:  # x / f32(sqrt 2560), IEEE
            g = _bf16(tl.div_rn(g, scale))
        else:  # x * f32(1 / f32(sqrt 2560))
            g = _bf16(g * scale)
        a = tl.abs(g)
        m = _bf16(tl.where(a != a, a, tl.maximum(a, lower)))
        r = _bf16(tl.sqrt_rn(m))
        s = (g > 0.0).to(tl.float32) - (g < 0.0).to(tl.float32)
        t = _bf16(r * s)
        sg = _bf16(tl.div_rn(1.0, 1.0 + libdevice.exp(-t)))
        tl.store(
            output_ptr + (token * HC_COUNT + h) * HIDDEN_SIZE + cols,
            (sg * value).to(tl.bfloat16),
        )


_POST_BLOCK = 512
_POST_STATE = {"ok": None, "reason": "", "divide": False, "consts": None}


def _post_consts(divide: bool) -> tuple[float, float]:
    """(scale, lower) as FP32 values. torch's true divide by a CPU scalar multiplies by
    opmath(1) / opmath(scalar) (divide=False); divide=True divides by opmath(scalar).
    The self-test picks the one this torch build matches."""
    if _POST_STATE["consts"] is None:
        root = torch.tensor(math.sqrt(_QWEN4_HIDDEN_SIZE), dtype=torch.float32)
        inv = torch.tensor(1.0, dtype=torch.float32) / root
        lower = float(torch.tensor(1e-6, dtype=torch.float32))
        _POST_STATE["consts"] = {False: (float(inv), lower), True: (float(root), lower)}
    return _POST_STATE["consts"][divide]


def _qwen4_gate_post_eager(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """The eager chain the kernel replaces, verbatim (models/qwen4_exp.py)."""
    gate = gate / math.sqrt(_QWEN4_HIDDEN_SIZE)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gate = torch.sigmoid(gate)
    return gate * value.unsqueeze(-2)


def _launch_qwen4_gate_post(
    gate: torch.Tensor, value: torch.Tensor, divide: Optional[bool] = None
) -> torch.Tensor:
    divide = _POST_STATE["divide"] if divide is None else divide
    output = torch.empty(
        (gate.shape[0], _QWEN4_HC_COUNT, _QWEN4_HIDDEN_SIZE),
        dtype=value.dtype,
        device=value.device,
    )
    if gate.shape[0]:
        _qwen4_gate_post_kernel[(gate.shape[0], _QWEN4_HIDDEN_SIZE // _POST_BLOCK)](
            gate,
            value,
            output,
            *_post_consts(divide),
            HC_COUNT=_QWEN4_HC_COUNT,
            HIDDEN_SIZE=_QWEN4_HIDDEN_SIZE,
            BLOCK=_POST_BLOCK,
            DIVIDE=divide,
            num_warps=2,
            enable_fp_fusion=False,
        )
    return output


def _qwen4_gate_post_selftest(device, divide: bool = False) -> tuple[bool, str]:
    """Every BF16 gate bit pattern (65,536, NaN, Inf and subnormals included) times random
    and special BF16 values, kernel against the eager chain, compared as raw bits."""
    gen = torch.Generator(device=device).manual_seed(4031)
    patterns = torch.arange(-32768, 32768, dtype=torch.int32, device=device).to(
        torch.int16
    )
    gates = patterns.view(torch.bfloat16).reshape(-1, _QWEN4_HC_COUNT, 1)
    special = torch.tensor(
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            65504.0,
            -65504.0,
            1e-30,
            -1e-30,
            3.0e38,
            float("inf"),
            float("-inf"),
        ],
        dtype=torch.float32,
        device=device,
    ).to(torch.bfloat16)
    rows = 2048
    mismatched = 0
    for start in range(0, gates.shape[0], rows):
        g = gates[start : start + rows].contiguous()
        scale = torch.exp2(
            torch.randint(
                -20, 21, (g.shape[0], 1), generator=gen, device=device
            ).float()
        )
        value = (
            torch.randn(g.shape[0], _QWEN4_HIDDEN_SIZE, generator=gen, device=device)
            * scale
        ).to(torch.bfloat16)
        value[:, : special.numel()] = special
        ref = _qwen4_gate_post_eager(g, value)
        new = _launch_qwen4_gate_post(g, value, divide)
        mismatched += int((ref.view(torch.int16) != new.view(torch.int16)).sum())
    torch.cuda.synchronize(device)
    return (
        mismatched == 0,
        f"{mismatched} of {gates.shape[0] * _QWEN4_HIDDEN_SIZE} outputs differ",
    )


def can_fuse_qwen4_gate_post(gate: torch.Tensor, value: torch.Tensor) -> bool:
    return (
        gate.is_cuda
        and gate.dtype == torch.bfloat16
        and gate.dim() == 3
        and gate.shape[1:] == (_QWEN4_HC_COUNT, 1)
        and gate.is_contiguous()
        and value.is_cuda
        and value.dtype == gate.dtype
        and value.device == gate.device
        and value.shape == (gate.shape[0], _QWEN4_HIDDEN_SIZE)
        and value.is_contiguous()
    )


def is_qwen4_gate_post_ready(device) -> bool:
    """Run the self-test once per process (never under graph capture); True once it passed."""
    if _POST_STATE["ok"] is None:
        if torch.cuda.is_current_stream_capturing():
            return False
        ok, detail = False, ""
        for divide in (False, True):
            try:
                ok, result = _qwen4_gate_post_selftest(device, divide)
            except Exception as exc:  # noqa: BLE001 - any failure keeps the eager chain
                ok, result = False, f"self-test raised {type(exc).__name__}: {exc}"
            detail += f"{'divide' if divide else 'reciprocal'}: {result}; "
            if ok:
                _POST_STATE["divide"] = divide
                break
        _POST_STATE["ok"], _POST_STATE["reason"] = ok, detail
        if ok:
            logger.info(
                "PLE prefill gate fused: on (SGLANG_OPT_PLE_PREFILL_FUSED_GATE; startup "
                "self-test bitwise equal to the eager chain over all 65536 BF16 gate "
                "inputs: %s)",
                detail,
            )
        else:
            logger.warning(
                "PLE prefill gate fused: kept the eager chain "
                "(SGLANG_OPT_PLE_PREFILL_FUSED_GATE; self-test: %s)",
                detail,
            )
    return bool(_POST_STATE["ok"])


def fused_qwen4_gate_post(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """gate: the BF16 sum of key * query, [T, 4, 1]; value [T, 2560] -> gated value
    [T, 4, 2560]."""
    if not can_fuse_qwen4_gate_post(gate, value):
        raise ValueError("unsupported input for the fused Qwen4 PLE prefill gate")
    return _launch_qwen4_gate_post(gate, value)
