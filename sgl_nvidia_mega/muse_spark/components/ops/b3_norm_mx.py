# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Boundary norm with an MXFP8 epilogue (csrc/b3_norm_mx.cuh), the producer form of ``fp8_prefill``'s activation quantize.

SGLang's fused post-norm + gated residual + pre-norm (its one-CTA-per-row path, used above 64 rows), with the same
per-row arithmetic and launch shape (thread count and weight prefetch from SGLang's own launch policy for the row count),
that also writes the BF16 pre-norm output as MXFP8: E4M3 values and one E8M0 scale per 32 elements, the scales in the
128 x 4 swizzled layout. The MXFP8 bytes are what ``mxfp8_ops.quantize`` gives for the BF16 output (the arithmetic of
``csrc/mxfp8_fq.cuh``); the BF16 output, the new residual stream and the router copy are byte-equal to SGLang's kernel.

``pad_quant``: the scale layout covers the rows rounded up to 128. With ``pad_quant=True`` the padding rows are written as
quantized zero rows (zero values in ``q_out`` rows ``[M, rows_pad)`` and the scale byte of a zero block), which is what
quantizing the zero-padded input gives; ``q_out`` then needs at least ``rows_pad`` rows. With ``pad_quant=False`` the
padding rows get scale bytes 0 and no values (what quantizing the unpadded input gives); ``q_out`` has ``M`` rows.
Rows <= 64 (SGLang's cluster path) are not handled: ``launch_config`` returns None and the caller keeps the unfused form.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Optional, Tuple

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": (
        "is_arch_support_pdl",
        "load_jit",
        "make_cpp_args",
    ),
    "kernels/jit/utils/compile/paths.py": ("KERNEL_PATH",),
}


_CSRC = Path(__file__).resolve().parents[1] / "csrc"
_CUH = _CSRC / "b3_norm_mx.cuh"
H = 8192
SF_TILE_ROWS = 128


@functools.lru_cache(maxsize=None)
def _use_pdl() -> bool:
    from sglang.kernels.jit.utils import is_arch_support_pdl

    return bool(is_arch_support_pdl())


@functools.lru_cache(maxsize=None)
def module(hidden: int, threads: int, prefetch: bool, pdl: bool):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args
    from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH

    from sgl_nvidia_mega.muse_spark.components.sgl_compat import csrc_include_paths

    args = make_cpp_args(hidden, threads, pdl, prefetch)
    t = "sglang::muse_vr_b3::B3NormMx"
    wrappers = []
    for scale in (False, True):
        for router in (False, True):
            for padq in (False, True):
                name = (
                    ("run_scale" if scale else "run_std")
                    + ("_router" if router else "")
                    + ("_padq" if padq else "")
                )
                wrappers.append(
                    (
                        name,
                        f"{t}<{args}, {str(scale).lower()}, {str(router).lower()}, {str(padq).lower()}>::run",
                    )
                )
    return load_jit(
        "muse_vr_b3_norm_mx",
        *args,
        cuda_files=[str(_CUH)],
        cuda_wrappers=wrappers,
        extra_include_paths=csrc_include_paths(KERNEL_PATH / "csrc") + [str(_CSRC)],
    )


def launch_config(hidden: int, num_tokens: int) -> Optional[Tuple[int, bool]]:
    """SGLang's (threads, prefetch) for this row count on its one-CTA-per-row path, or None (cluster path, other H)."""
    if hidden != H or num_tokens <= 0:
        return None
    from sgl_nvidia_mega.muse_spark.components import sgl_compat

    nr = sgl_compat.module("sglang.kernels.ops.muse_spark_v12.norm_residual")
    cluster, threads, prefetch = nr.norm_residual_launch_config(hidden, num_tokens)
    if cluster != 1:
        return None
    return int(threads), bool(prefetch)


def rows_pad(num_tokens: int) -> int:
    return -(-num_tokens // SF_TILE_ROWS) * SF_TILE_ROWS


def sf_bytes(num_tokens: int, hidden: int = H) -> int:
    return rows_pad(num_tokens) * (hidden // 32)


def norm_residual_mx(
    branch: torch.Tensor,
    stream: torch.Tensor,
    post_eps: float,
    pre_eps: float,
    *,
    q_out: torch.Tensor,
    sf_out: torch.Tensor,
    pad_quant: bool,
    post_norm_is_input_scale: bool = False,
    post_weight: Optional[torch.Tensor] = None,
    pre_weight: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    want_router: bool = False,
    pre_norm_bf16: bool = False,
    output: Optional[torch.Tensor] = None,
    new_stream_out: Optional[torch.Tensor] = None,
    router_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """``(output bf16, new_stream fp32, router fp32 or None)`` as SGLang's fused_norm_residual, plus the MXFP8 of
    ``output`` in ``q_out`` (uint8 [M or >= rows_pad, H]) and ``sf_out`` (uint8, >= rows_pad * H / 32)."""
    assert branch.dtype == torch.bfloat16 and stream.dtype == torch.float32
    assert branch.dim() == 2 and stream.shape == branch.shape
    assert branch.is_contiguous() and stream.is_contiguous()
    m, hidden = branch.shape
    cfg = launch_config(hidden, m)
    if cfg is None:
        raise ValueError(f"b3_norm_mx: unsupported shape (M={m}, H={hidden})")
    if output is None:
        output = torch.empty_like(branch)
    if new_stream_out is None:
        new_stream_out = torch.empty_like(stream)
    if want_router and router_out is None:
        router_out = torch.empty_like(stream)
    if not want_router:
        router_out = None
    threads, prefetch = cfg
    mod = module(hidden, threads, prefetch, _use_pdl())
    name = (
        ("run_scale" if post_norm_is_input_scale else "run_std")
        + ("_router" if want_router else "")
        + ("_padq" if pad_quant else "")
    )
    getattr(mod, name)(
        branch,
        stream,
        output,
        new_stream_out,
        router_out,
        post_weight,
        pre_weight,
        alpha,
        beta,
        float(post_eps),
        float(pre_eps),
        bool(pre_norm_bf16),
        q_out,
        sf_out,
    )
    return output, new_stream_out, router_out
