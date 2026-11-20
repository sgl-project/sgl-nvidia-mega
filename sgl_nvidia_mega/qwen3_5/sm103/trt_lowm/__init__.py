# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""TensorRT-LLM r3's low-M GEMM (``LowMMmaGemmKernel``, CuTe DSL: ``lowm_mma_gemm.py`` with ``utils.py`` and
``lowm_linear.py``, vendored from tekit 7df45abe5c) for callers that give their own shapes and tactics.

``spec(...)`` names one launch by TensorRT-LLM's compile key (``lowm_linear._compiled``): rows, K, the FP8 and BF16
weight segments with their tactics, the activation operand, the output split, the activations' row strides and the
GDN conv epilogue's tensor layouts (``conv_spec``); the kernel is specialized on all of them. ``prepare(spec)``
compiles it and refuses inside a graph capture: prepare every spec a step can take before the first capture (in a
package's build, or where the pools first exist for a spec that names them). ``run(spec, ...)`` launches a prepared
spec into the caller's buffers and refuses a spec that was not prepared, so a step never compiles.

Numerics are TensorRT-LLM's (``lowm_mma_gemm.py`` module doc): FP32 sums in its order, not SGLang's GEMMs bit for
bit. Every launch is a programmatic dependent that reads only weights, scales and conv weights ahead of its
``griddepcontrol.wait`` and, with ``early_trigger``, releases its dependents once those loads are issued (they wait
for its completion before reading its outputs).
"""

from __future__ import annotations

import logging
import time
from typing import NamedTuple, Optional

import torch

logger = logging.getLogger(__name__)


class Spec(NamedTuple):
    """One compiled launch: the arguments of ``lowm_linear._compiled``, in its order."""

    m: int
    k: int
    k_warps: int
    n_fp8: int
    n_tiles: int
    fp8_act: str
    swiglu_fp8: bool
    n_bf16: int
    n_tiles_bf16: int
    swiglu_bf16: bool
    split_out: int
    n_valid: int
    quant_out: bool
    n_out: int
    n_out2: int
    out_dtype: torch.dtype
    a16_stride: int
    a8_stride: int
    use_pdl: bool
    early_trigger: bool
    release_after_wait: bool
    unsafe_skip_wait: bool = False
    ksplit: int = 1
    swap_ab: bool = False
    min_blocks: int = 1
    conv_channels: int = 0
    conv_tokens: int = 1
    conv_write_state: bool = True
    conv_shapes: tuple = ()


def conv_spec(
    weight, state, windows, requests: int, tokens: int, write_state: bool = True
) -> tuple:
    """The GDN conv epilogue of a spec, from the tensors ``run`` will get (the kernel is specialized on their shapes
    and strides): ``weight`` [C, 4] BF16, ``state`` [slots, C, 3] BF16, ``windows`` [>= requests, tokens, C, 3] BF16;
    ``requests`` of ``tokens`` rows (``lowm_linear.lowm_gemm``'s ``conv``)."""
    if weight.dim() != 2 or weight.shape[1] != 4 or weight.dtype != torch.bfloat16:
        raise ValueError(f"conv weight {weight.dtype} {tuple(weight.shape)}")
    if (
        state.dim() != 3
        or tuple(state.shape[1:]) != (weight.shape[0], 3)
        or state.dtype != torch.bfloat16
    ):
        raise ValueError(f"conv state {state.dtype} {tuple(state.shape)}")
    if (
        windows.dim() != 4
        or tuple(windows.shape[1:]) != (tokens, weight.shape[0], 3)
        or windows.dtype != torch.bfloat16
        or windows.shape[0] < requests
    ):
        raise ValueError(f"conv windows {windows.dtype} {tuple(windows.shape)}")
    return (
        weight.shape[0],
        tokens,
        write_state,
        (
            (tuple(weight.shape), tuple(weight.stride())),
            (tuple(state.shape), tuple(state.stride())),
            requests,
            (tuple(windows.shape), tuple(windows.stride())),
        ),
    )


def spec(
    *,
    m: int,
    k: int,
    n_fp8: int = 0,
    n_bf16: int = 0,
    fp8_act: str = "fp8",
    tactic: Optional[tuple] = None,
    tactic_bf16: Optional[tuple] = None,
    swiglu_fp8: bool = False,
    swiglu_bf16: bool = False,
    split_out: int = 0,
    n_valid: int = 0,
    quant_out: bool = False,
    out_dtype: torch.dtype = torch.bfloat16,
    a16_stride: Optional[int] = None,
    a8_stride: Optional[int] = None,
    pdl: bool = True,
    early_trigger: bool = True,
    release_after_wait: bool = False,
    conv: Optional[tuple] = None,
) -> Spec:
    """A launch of ``m`` rows of K ``k`` over ``n_fp8`` FP8 and ``n_bf16`` BF16 weight rows, resolved as
    ``lowm_linear.lowm_gemm`` does. ``tactic`` / ``tactic_bf16``: (k_warps, n_tiles[, ksplit[, swap_ab[,
    min_blocks]]]) of the FP8 / BF16 segment, required for a non-empty segment (TensorRT-LLM's choice per shape:
    ``lowm_linear.default_tactic``); one k_warps and ksplit for both (the BF16 segment takes one tile a CTA
    otherwise). ``fp8_act``: ``fp8`` (E4M3 rows ``x8``), ``quant`` (BF16 rows quantized in registers) or ``bf16``.
    ``a16_stride`` / ``a8_stride``: the row strides (elements) of the activations ``run`` will get (default ``k``).
    ``pdl`` / ``early_trigger`` / ``release_after_wait`` select the dependent-launch
    behavior explicitly. ``conv``: ``conv_spec(...)``."""
    from sgl_nvidia_mega.qwen3_5.sm103.trt_lowm.lowm_linear import _tactic3

    if (n_fp8 and tactic is None) or (n_bf16 and tactic_bf16 is None):
        raise ValueError("every non-empty segment needs its tactic")
    k_warps, n_tiles, ksplit, swap, min_blocks = (
        _tactic3(tactic) if n_fp8 else (0, 1, 1, 0, 1)
    )
    kw16, nt16, ks16, swap16, mb16 = (
        _tactic3(tactic_bf16) if n_bf16 else (0, 1, 1, 0, 1)
    )
    if not n_fp8:
        k_warps, ksplit, swap, min_blocks = kw16, ks16, swap16, mb16
    elif n_bf16 and (kw16, ks16) != (k_warps, ksplit):
        nt16 = 1
    n_logical = (n_fp8 // 2 if swiglu_fp8 else n_fp8) + (
        n_bf16 // 2 if swiglu_bf16 else n_bf16
    )
    if n_valid:
        if not split_out < n_valid <= n_logical:
            raise ValueError(
                f"n_valid {n_valid} with split_out {split_out} of {n_logical} columns"
            )
        n_logical = n_valid
    uses_a16 = n_bf16 > 0 or (n_fp8 > 0 and fp8_act != "fp8")
    uses_a8 = n_fp8 > 0 and fp8_act == "fp8"
    conv_args = conv if conv is not None else (0, 1, True, ())
    return Spec(
        m,
        k,
        k_warps,
        n_fp8,
        n_tiles,
        fp8_act,
        swiglu_fp8,
        n_bf16,
        nt16,
        swiglu_bf16,
        split_out,
        n_valid,
        quant_out,
        split_out if split_out else n_logical,
        n_logical - split_out if split_out else 0,
        out_dtype,
        (a16_stride or k) if uses_a16 else 0,
        (a8_stride or k) if uses_a8 else 0,
        pdl,
        early_trigger and not release_after_wait,
        release_after_wait,
        False,
        ksplit,
        bool(swap) and m <= 8,
        min_blocks,
        *conv_args,
    )


_PREPARED: dict = {}


def prepare(s: Spec) -> None:
    """Compile ``s`` once (CuTe DSL), outside graph capture."""
    if s in _PREPARED:
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"TensorRT-LLM low-M GEMM {s}: prepare it before graph capture"
        )
    from sgl_nvidia_mega.qwen3_5.sm103.trt_lowm.lowm_linear import _compiled, _dummy

    dev = torch.device("cuda", torch.cuda.current_device())
    # the placeholder operands of unused segments, made here and not inside a capture
    for kind in ("a16", "a8", "b8", "b16", "s", "c", "cw", "cs", "slot", "win"):
        _dummy(kind, dev)
    start = time.perf_counter()
    _PREPARED[s] = _compiled(*s)
    logger.info(
        "NVIDIA megakernel: TensorRT-LLM low-M GEMM compiled in %.1f s (m=%d k=%d fp8=%d/%s bf16=%d, "
        "k_warps=%d n_tiles=%d/%d ksplit=%d swap_ab=%d min_blocks=%d conv=%d)",
        time.perf_counter() - start,
        s.m,
        s.k,
        s.n_fp8,
        s.fp8_act,
        s.n_bf16,
        s.k_warps,
        s.n_tiles,
        s.n_tiles_bf16,
        s.ksplit,
        s.swap_ab,
        s.min_blocks,
        s.conv_channels,
    )


def prepared(s: Spec) -> bool:
    return s in _PREPARED


def _check(
    t: torch.Tensor, name: str, shape: tuple, stride: tuple, dtypes: tuple
) -> None:
    if t is None:
        raise ValueError(f"{name} is required")
    if (
        tuple(t.shape) != shape
        or tuple(t.stride()) != stride
        or t.dtype not in dtypes
        or t.data_ptr() % 16
    ):
        raise ValueError(
            f"{name} {t.dtype} {tuple(t.shape)} {tuple(t.stride())} is not the prepared "
            f"{dtypes[0]} {shape} {stride} (16-byte aligned)"
        )


def run(
    s: Spec,
    *,
    out: torch.Tensor,
    out2: Optional[torch.Tensor] = None,
    x16: Optional[torch.Tensor] = None,
    x8: Optional[torch.Tensor] = None,
    w_fp8: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
    input_scale: Optional[torch.Tensor] = None,
    w_bf16: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
    conv: Optional[tuple] = None,
) -> None:
    """Launch the prepared ``s`` on the current stream: ``out`` [m, n_out] (``out_dtype``, E4M3 with ``quant_out``)
    and ``out2`` [m, n_out2] contiguous; ``x16`` [m, k] BF16 / ``x8`` [m, k] E4M3 rows at the spec's row strides;
    ``w_fp8`` [n_fp8, k] E4M3 with ``weight_scale`` and ``input_scale`` (FP32 [1]); ``w_bf16`` [n_bf16, k];
    ``out_scale`` with ``quant_out``; ``conv`` = (weight, state, slots [requests] int32, windows) at the layouts of
    the spec's ``conv_spec`` (TensorRT-LLM ``lowm_gemm``'s operands)."""
    import cuda.bindings.driver as cuda

    from sgl_nvidia_mega.qwen3_5.sm103.trt_lowm.lowm_linear import _cute, _dummy

    fn = _PREPARED.get(s)
    if fn is None:
        raise RuntimeError(
            f"TensorRT-LLM low-M GEMM {s} was not prepared (no compile inside a step)"
        )
    dev = out.device
    e4m3 = (torch.float8_e4m3fn, torch.uint8)
    _check(
        out,
        "out",
        (s.m, s.n_out),
        (s.n_out, 1),
        e4m3 if s.quant_out else (s.out_dtype,),
    )
    if s.n_out2:
        _check(out2, "out2", (s.m, s.n_out2), (s.n_out2, 1), (s.out_dtype,))
    if s.a16_stride:
        _check(x16, "x16", (s.m, s.k), (s.a16_stride, 1), (torch.bfloat16,))
    if s.a8_stride:
        _check(x8, "x8", (s.m, s.k), (s.a8_stride, 1), e4m3)
    if s.n_fp8:
        _check(w_fp8, "w_fp8", (s.n_fp8, s.k), (s.k, 1), e4m3)
    if s.n_bf16:
        _check(w_bf16, "w_bf16", (s.n_bf16, s.k), (s.k, 1), (torch.bfloat16,))
    if s.conv_channels:
        (cw, cws), (cs, css), n_req, (win, wins) = s.conv_shapes
        weight, state, slots, windows = conv
        _check(weight, "conv weight", cw, cws, (torch.bfloat16,))
        _check(state, "conv state", cs, css, (torch.bfloat16,))
        _check(slots, "conv slots", (n_req,), (1,), (torch.int32,))
        _check(windows, "conv windows", win, wins, (torch.bfloat16,))
        conv_tensors = (weight, state, slots, windows)
    else:
        conv_tensors = (
            _dummy("cw", dev),
            _dummy("cs", dev),
            _dummy("slot", dev),
            _dummy("win", dev),
        )
    ones = _dummy("s", dev)
    fn(
        _cute(x16 if s.a16_stride else _dummy("a16", dev)),
        _cute(x8.view(torch.uint8) if s.a8_stride else _dummy("a8", dev)),
        _cute(w_fp8.view(torch.uint8) if s.n_fp8 else _dummy("b8", dev)),
        _cute(weight_scale.reshape(1) if s.n_fp8 else ones),
        _cute(input_scale.reshape(1) if (s.n_fp8 and s.fp8_act != "bf16") else ones),
        _cute(out_scale.reshape(1) if s.quant_out else ones),
        _cute(w_bf16 if s.n_bf16 else _dummy("b16", dev)),
        _cute(out.view(torch.uint8) if s.quant_out else out),
        _cute(out2 if s.n_out2 else _dummy("c", dev)),
        *(_cute(t) for t in conv_tensors),
        cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream),
    )
