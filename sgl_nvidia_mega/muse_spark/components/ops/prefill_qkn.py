# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Prefill QK path (JIT CUDA): per-head RMS norm of the q and k heads of a QKV projection output and RoPE.

``qk_norm``: bf16 q and k exactly as SGLang's fused QK-norm Triton kernel writes them in its no-store launch, for row
counts where that kernel reduces 8 columns per thread (one warp per token; graph-capture safe: no host sync, no
dynamic cache locations). ``prefill_qkn``: the same plus q in e4m3 and the e4m3 K / V cache store, bitwise to the
Triton launch followed by SGLang's fused FP8 q / KV-cache store with unit scales. See ``csrc/prefill_qkn.cuh`` and
``tests/test_prefill_qkn.py``. ``qk_norm_q8``: ``qk_norm`` with q written as e4m3, byte for byte
``qk_norm(...)[0].to(torch.float8_e4m3fn)`` (k unchanged); its own module (``csrc/prefill_qkn_q8.cuh``,
``tests/test_prefill_qkn_q8.py``).
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("load_jit", "make_cpp_args"),
}


_CSRC = Path(__file__).resolve().parent.parent / "csrc" / "prefill_qkn.cuh"
_CSRC_Q8 = Path(__file__).resolve().parent.parent / "csrc" / "prefill_qkn_q8.cuh"

HEAD_DIM = 64
MAX_HEADS = 64  # q + k + v head rows per token
# The Triton kernel reduces 8 columns per thread from this many (token, head) rows of its no-store launch up
# (``fused_qk_norm._triton_config``: 16 or 32 rows per 4-warp block).
MIN_TRITON_ROWS = 8192


def covers(num_tokens: int, num_q_heads: int, num_kv_heads: int) -> bool:
    """Whether the stock Triton launch for this call uses the column split whose arithmetic this kernel reproduces."""
    return (
        num_tokens * (num_q_heads + num_kv_heads) >= MIN_TRITON_ROWS
        and num_q_heads + 2 * num_kv_heads <= MAX_HEADS
    )


def _unfused() -> bool:
    """The unfused arithmetic form (ops/swa_verify_attn.py, qk_arith)."""
    from .swa_verify_attn import qk_arith

    return qk_arith() == "unfused"


@functools.lru_cache(maxsize=None)
def _norm_module(rope: bool, unfused: bool = False):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(rope)
    return load_jit(
        "muse_vr_prefill_qk_norm" + ("_qku" if unfused else ""),
        *args,
        cuda_files=[str(_CSRC)],
        cuda_wrappers=[("qk_norm", f"muse_vr::PrefillQkNorm<{args}>::run")],
        extra_cuda_cflags=["-DMUSE_VR_QK_UNFUSED=1"] if unfused else None,
    )


def qk_norm(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor | None,
    num_q_heads: int,
    num_kv_heads: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(q, k): bf16 ``[T, Hq * 64]`` and ``[T, Hkv * 64]``, normed per head (and rotated when ``cos_sin`` is given)."""
    t = qkv.shape[0]
    q = torch.empty(t, num_q_heads * HEAD_DIM, dtype=torch.bfloat16, device=qkv.device)
    k = torch.empty(t, num_kv_heads * HEAD_DIM, dtype=torch.bfloat16, device=qkv.device)
    rope = cos_sin is not None
    _norm_module(rope, _unfused()).qk_norm(
        qkv,
        positions,
        cos_sin if rope else positions,
        q,
        k,
        int(num_q_heads),
        int(num_kv_heads),
        float(eps),
    )
    return q, k


@functools.lru_cache(maxsize=None)
def _norm_q8_module(rope: bool, unfused: bool = False):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(rope)
    return load_jit(
        "muse_vr_prefill_qk_norm_q8" + ("_qku" if unfused else ""),
        *args,
        cuda_files=[str(_CSRC_Q8)],
        cuda_wrappers=[
            ("qk_norm_q8", f"muse_vr::PrefillQkNormQ8<{args}>::run"),
            ("e4m3_of_bf16", "muse_vr::E4m3OfBf16::run"),
        ],
        extra_cuda_cflags=["-DMUSE_VR_QK_UNFUSED=1"] if unfused else None,
    )


def qk_norm_q8(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor | None,
    num_q_heads: int,
    num_kv_heads: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(q, k): e4m3 ``[T, Hq * 64]`` (the bytes of ``qk_norm``'s bf16 q converted with ``.to(torch.float8_e4m3fn)``) and
    bf16 ``[T, Hkv * 64]`` (``qk_norm``'s k). Same inputs and requirements as ``qk_norm``."""
    t = qkv.shape[0]
    q = torch.empty(
        t, num_q_heads * HEAD_DIM, dtype=torch.float8_e4m3fn, device=qkv.device
    )
    k = torch.empty(t, num_kv_heads * HEAD_DIM, dtype=torch.bfloat16, device=qkv.device)
    rope = cos_sin is not None
    _norm_q8_module(rope, _unfused()).qk_norm_q8(
        qkv,
        positions,
        cos_sin if rope else positions,
        q,
        k,
        int(num_q_heads),
        int(num_kv_heads),
        float(eps),
    )
    return q, k


def e4m3_of_bf16(x: torch.Tensor) -> torch.Tensor:
    """``qk_norm_q8``'s e4m3 conversion alone, element-wise: uint8 bytes of a 1-D contiguous bf16 tensor (tests)."""
    out = torch.empty(x.shape[0], dtype=torch.uint8, device=x.device)
    _norm_q8_module(False, _unfused()).e4m3_of_bf16(x, out)
    return out


@functools.lru_cache(maxsize=None)
def _module(rope: bool, unfused: bool = False):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(rope)
    return load_jit(
        "muse_vr_prefill_qkn" + ("_qku" if unfused else ""),
        *args,
        cuda_files=[str(_CSRC)],
        cuda_wrappers=[("prefill_qkn", f"muse_vr::PrefillQkn<{args}>::run")],
        extra_cuda_cflags=["-DMUSE_VR_QK_UNFUSED=1"] if unfused else None,
    )


def prefill_qkn(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor | None,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    cache_loc: torch.Tensor,
    q_out: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    eps: float,
) -> torch.Tensor:
    """q_out[t] = e4m3(rope(rmsnorm(q heads of qkv[t]))); K / V pools at row cache_loc[t] get e4m3(rope(rmsnorm(k)))
    and e4m3(v) (every location, 0 included, as the stock FP8 store writes them). ``cos_sin`` None: no RoPE.

    :param qkv: ``[T, >= (Hq + 2 Hkv) * 64]`` bf16, row stride a multiple of 8, 16-byte aligned (q, k, v heads first).
    :param positions: ``[T]`` int64. :param cos_sin: ``[max position, 64]`` fp32, cos then sin halves, or None.
    :param k_pool: / :param v_pool: ``[slots, >= Hkv * 64]`` e4m3 views (row stride in elements = bytes).
    :param cache_loc: ``[T]`` int64. :param q_out: ``[T, Hq * 64]`` e4m3, contiguous.
    """
    rope = cos_sin is not None
    _module(rope, _unfused()).prefill_qkn(
        qkv,
        positions,
        cos_sin if rope else positions,
        k_pool,
        v_pool,
        cache_loc,
        q_out,
        int(num_q_heads),
        int(num_kv_heads),
        float(eps),
    )
    return q_out


_loaded: dict = {}


def is_loaded(rope: bool) -> bool:
    return bool(_loaded.get(rope, False))


def preload(device, rope: bool) -> None:
    """Build and load the kernel (no build or module load inside a later call)."""
    if _loaded.get(rope):
        return
    qkv = torch.zeros(1, 3 * HEAD_DIM * 8, dtype=torch.bfloat16, device=device)
    pos = torch.zeros(1, dtype=torch.int64, device=device)
    cs = torch.zeros(1, HEAD_DIM, dtype=torch.float32, device=device) if rope else None
    pool = torch.zeros(2, HEAD_DIM, dtype=torch.float8_e4m3fn, device=device)
    loc = torch.zeros(1, dtype=torch.int64, device=device)
    q = torch.empty(1, 8 * HEAD_DIM, dtype=torch.float8_e4m3fn, device=device)
    prefill_qkn(qkv, pos, cs, pool, pool, loc, q, 8, 1, 1e-6)
    torch.cuda.current_stream(device).synchronize()
    _loaded[rope] = True
