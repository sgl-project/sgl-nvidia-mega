# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from TensorRT-LLM r3 (tekit 7df45abe5c),
# tensorrt_llm/_torch/modules/lowm_linear.py.
"""The TensorRT-LLM low-M compiler helpers used by the Qwen3.5 GDN front.

Only the validated TP8 GDN projection shapes and TensorRT-LLM's graph-safe CuTe
compiler adapter are kept here. Slot selection, row bounds and dependent-launch
policy belong to the plugin plan and its caller, rather than environment switches.
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import torch

_NUM_SMS_DEFAULT = 148


@functools.lru_cache(maxsize=None)
def _num_sms() -> int:
    if not torch.cuda.is_available():
        return _NUM_SMS_DEFAULT
    return torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count


Tactic = Tuple[int, int, int, int, int]


def _tactic3(t) -> Tactic:
    """Normalize a two- to five-entry tactic to five entries."""
    return (
        t[0],
        t[1],
        t[2] if len(t) > 2 else 1,
        t[3] if len(t) > 3 else 0,
        t[4] if len(t) > 4 else 1,
    )


# TensorRT-LLM r3 tactics used by the TP8 GDN front. The final entry is also
# the fixed fallback beyond its measured crossover, through four requests.
_TACTICS = {
    (2560, 4096, "mixed"): ((8, (16, 4, 1, 1)), (16, (16, 4))),
    (2560, 4096, "mixed8"): ((8, (16, 2, 1, 1, 2)), (16, (8, 4))),
}


def lowm_conv_supported(m: int, tactic, conv_channels: int, tokens: int) -> bool:
    """Whether the tactic supports the GDN causal-conv epilogue."""
    k_warps, n_tiles = tactic[0], tactic[1]
    ksplit = tactic[2] if len(tactic) > 2 else 1
    cols = 8 * n_tiles
    return (
        ksplit == 1
        and tokens > 0
        and m % tokens == 0
        and conv_channels % cols == 0
        and (m // tokens) * cols <= 32 * k_warps
    )


def default_tactic(
    m: int,
    n: int,
    k: int,
    fp8: bool = True,
    even_tiles: bool = False,
    mode: Optional[str] = None,
) -> Tactic:
    """Return TensorRT-LLM's fixed tactic for an ``m x n x k`` GEMM."""
    entries = _TACTICS.get((n, k, mode or ("quant" if fp8 else "bf16")))
    if entries:
        for max_rows, tactic in entries:
            if m <= max_rows or max_rows == entries[-1][0]:
                normalized = _tactic3(tactic)
                k_warps, n_tiles, ksplit, swap, min_blocks = normalized
                if even_tiles and n_tiles % 2:
                    n_tiles *= 2
                if n % (8 * n_tiles) == 0 and (k // 64) % (k_warps * ksplit) == 0:
                    return k_warps, n_tiles, ksplit, swap, min_blocks
                break

    chunks = k // 64
    per_chunk = 16 if fp8 else 32
    sms = _num_sms()
    best = None
    for n_tiles in (1, 2, 4):
        if n % (8 * n_tiles) or (even_tiles and n_tiles % 2):
            continue
        ctas = n // (8 * n_tiles)
        for k_warps in (1, 2, 4, 8, 16):
            if chunks % k_warps:
                continue
            per_warp = chunks // k_warps
            if per_warp * n_tiles * per_chunk > 256:
                continue
            waves = -(-ctas // (2 * sms))
            key = (waves, per_warp > 8, -min(ctas, 2 * sms), per_warp, -k_warps)
            if best is None or key < best[0]:
                best = (key, (k_warps, n_tiles, 1, 0, 1))
    if best is None:
        raise ValueError(f"no low-M tactic for N={n}, K={k}")
    return best[1]


class _GraphSafeDlpack:
    """DLPack export that does not synchronize streams (CUDA-graph capture safe)."""

    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor

    def __dlpack__(self, stream=None):
        return self._tensor.__dlpack__(stream=-1)

    def __dlpack_device__(self):
        return self._tensor.__dlpack_device__()


def _cute(t: torch.Tensor):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(_GraphSafeDlpack(t), assumed_align=16)


_DUMMIES: dict = {}


def _dummy(kind: str, device) -> torch.Tensor:
    """Placeholder operands with fixed layouts (unused by the compiled variant)."""
    key = (kind, device)
    if key not in _DUMMIES:
        if kind == "a16":
            t = torch.zeros((1, 64), dtype=torch.bfloat16, device=device)
        elif kind == "a8":
            t = torch.zeros((1, 64), dtype=torch.uint8, device=device)
        elif kind == "b8":
            t = torch.zeros((8, 64), dtype=torch.uint8, device=device)
        elif kind == "b16":
            t = torch.zeros((8, 64), dtype=torch.bfloat16, device=device)
        elif kind == "s":
            t = torch.ones((1,), dtype=torch.float32, device=device)
        elif kind == "c":
            t = torch.zeros((1, 8), dtype=torch.bfloat16, device=device)
        elif kind == "cw":
            t = torch.zeros((8, 4), dtype=torch.bfloat16, device=device)
        elif kind == "cs":
            t = torch.zeros((1, 8, 3), dtype=torch.bfloat16, device=device)
        elif kind == "slot":
            t = torch.zeros((1,), dtype=torch.int32, device=device)
        elif kind == "win":
            t = torch.zeros((1, 1, 8, 3), dtype=torch.bfloat16, device=device)
        else:
            raise KeyError(kind)
        _DUMMIES[key] = t
    return _DUMMIES[key]


def _row_view(rows: int, cols: int, stride: int, dtype, device) -> torch.Tensor:
    return torch.empty((rows, stride), dtype=dtype, device=device)[:, :cols]


@functools.lru_cache(maxsize=None)
def _compiled(
    m: int,
    k: int,
    k_warps: int,
    n_fp8: int,
    n_tiles: int,
    fp8_act: str,
    swiglu_fp8: bool,
    n_bf16: int,
    n_tiles_bf16: int,
    swiglu_bf16: bool,
    split_out: int,
    n_valid: int,
    quant_out: bool,
    n_out: int,
    n_out2: int,
    out_dtype: torch.dtype,
    a16_stride: int,
    a8_stride: int,
    use_pdl: bool,
    early_trigger: bool,
    release_after_wait: bool,
    unsafe_skip_wait: bool = False,
    ksplit: int = 1,
    swap_ab: bool = False,
    min_blocks: int = 1,
    conv_channels: int = 0,
    conv_tokens: int = 1,
    conv_write_state: bool = True,
    conv_shapes: tuple = (),
):
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute

    from .lowm_mma_gemm import LowMMmaGemmKernel

    kernel = LowMMmaGemmKernel(
        num_rows=m,
        k_extent=k,
        k_warps=k_warps,
        use_pdl=use_pdl,
        n_fp8=n_fp8,
        n_tiles=n_tiles,
        fp8_act=fp8_act,
        swiglu_fp8=swiglu_fp8,
        n_bf16=n_bf16,
        n_tiles_bf16=n_tiles_bf16,
        swiglu_bf16=swiglu_bf16,
        split_out=split_out,
        n_valid=n_valid,
        quant_out=quant_out,
        early_trigger=early_trigger,
        release_after_wait=release_after_wait,
        unsafe_skip_wait=unsafe_skip_wait,
        ksplit=ksplit,
        swap_ab=swap_ab,
        min_blocks=min_blocks,
        conv_channels=conv_channels,
        conv_tokens=conv_tokens,
        conv_write_state=conv_write_state,
    )
    dev = torch.device("cuda")
    if conv_channels:
        (cw_shape, cw_stride), (cs_shape, cs_stride), n_req, (win_shape, win_stride) = (
            conv_shapes
        )
        cw = torch.empty_strided(cw_shape, cw_stride, dtype=torch.bfloat16, device=dev)
        cs = torch.empty_strided(cs_shape, cs_stride, dtype=torch.bfloat16, device=dev)
        slots = torch.zeros((n_req,), dtype=torch.int32, device=dev)
        win = torch.empty_strided(
            win_shape, win_stride, dtype=torch.bfloat16, device=dev
        )
    else:
        cw, cs, slots, win = (
            _dummy("cw", dev),
            _dummy("cs", dev),
            _dummy("slot", dev),
            _dummy("win", dev),
        )
    a16 = (
        _row_view(m, k, a16_stride, torch.bfloat16, dev)
        if a16_stride
        else _dummy("a16", dev)
    )
    a8 = (
        _row_view(m, k, a8_stride, torch.uint8, dev) if a8_stride else _dummy("a8", dev)
    )
    b = (
        torch.empty((n_fp8, k), dtype=torch.uint8, device=dev)
        if n_fp8
        else _dummy("b8", dev)
    )
    b2 = (
        torch.empty((n_bf16, k), dtype=torch.bfloat16, device=dev)
        if n_bf16
        else _dummy("b16", dev)
    )
    s = _dummy("s", dev)
    c = torch.empty(
        (m, n_out), dtype=torch.uint8 if quant_out else out_dtype, device=dev
    )
    c2 = (
        torch.empty((m, n_out2), dtype=out_dtype, device=dev)
        if n_out2
        else _dummy("c", dev)
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute.compile(
        kernel,
        _cute(a16),
        _cute(a8),
        _cute(b),
        _cute(s),
        _cute(s),
        _cute(s),
        _cute(b2),
        _cute(c),
        _cute(c2),
        _cute(cw),
        _cute(cs),
        _cute(slots),
        _cute(win),
        stream,
    )
