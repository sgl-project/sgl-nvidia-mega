# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Paged NVFP4-KV verify attention on tensor-core MMA (tcgen05) with a length-proportional key split, JIT CUDA.

See ``csrc/nvfp4_verify_attn.cuh`` for the arithmetic, the work split and the synchronization. The op covers the verify
step of attention layers with 8 query heads per KV head, head_dim 64, an even number of KV heads per rank, bf16 Q, the
native NVFP4 KV layout (packed E2M1 rows + E4M3 block scales, V scales four-token interleaved), page size 128 and bf16
output, for full attention (``window_left < 0``) or a sliding window.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("load_jit", "make_cpp_args"),
}


_CSRC = Path(__file__).resolve().parent.parent / "csrc" / "nvfp4_verify_attn.cuh"

TOKENS_PER_REQUEST = 8
HEAD_DIM = 64
Q_PER_KV = 8
PAGE = 128
REC_FLOATS = 68
ROWS = 64

# Key splits per (request, KV-head pair) unit (bench sweeps, 4 KV heads per rank, one block per SM):
#   full attention: one wave (SMs // units splits) while the units fill at most half the SMs, else the split count in
#   2..8 whose blocks fill their last wave best (measured faster than a fixed wave count in A/B inside a
#   GEMM -> attention -> GEMM CUDA graph; tunable, sweep with bench/bench_nvfp4_kv_attn.py);
#   sliding window (at most 17 pages per unit): about 0.8 x SMs / units splits, capped at 12. With one split the
#   kernel writes the output itself (no merge launch).
# MUSE_VR_NVFP4_VERIFY_ATTN_BLOCKS overrides with a fixed block count (blocks // units splits).


@functools.lru_cache(maxsize=None)
def _module(use_pdl: bool):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(use_pdl)
    return load_jit(
        "muse_vr_nvfp4_verify_attn",
        *args,
        cuda_files=[str(_CSRC)],
        cuda_wrappers=[
            ("nvfp4_verify_attn", f"muse_vr::Nvfp4VerifyAttnKernel<{args}>::run")
        ],
    )


@functools.lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def num_splits(
    num_requests: int, num_kv_heads: int, device, window_left: int = -1
) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    sms = _sm_count(idx)
    units = max(1, num_requests * (num_kv_heads // 2))
    raw = os.environ.get("MUSE_VR_NVFP4_VERIFY_ATTN_BLOCKS", "")
    if raw:
        return max(1, int(raw) // units)
    if window_left >= 0:
        return max(1, min(12, round(0.8 * sms / units)))
    if 2 * units <= sms:
        return max(1, sms // units)
    return _fill_waves(units, sms)


def _fill_waves(units: int, sms: int) -> int:
    """Splits in 2..8 whose block count fills its last wave best (blocks / (waves * SMs)); ties -> fewer splits."""
    best, best_eff = 2, -1.0
    for s in range(2, 9):
        blocks = units * s
        eff = blocks / (-(-blocks // sms) * sms)
        if eff > best_eff + 1e-9:
            best, best_eff = s, eff
    return best


_workspace: dict = {}


def workspace(device, num_floats: int) -> torch.Tensor:
    """Per-device fp32 scratch for the partial results; grows, never shrinks (allocate before graph capture)."""
    key = (device.type, device.index)
    buf = _workspace.get(key)
    if buf is None or buf.numel() < num_floats:
        buf = torch.empty(num_floats, dtype=torch.float32, device=device)
        _workspace[key] = buf
    return buf


def workspace_floats(num_requests: int, num_kv_heads: int, splits: int) -> int:
    return num_requests * num_kv_heads * splits * ROWS * REC_FLOATS


def supported(
    num_q_heads: int, num_kv_heads: int, head_dim: int, page_size: int
) -> bool:
    return (
        head_dim == HEAD_DIM
        and num_kv_heads * Q_PER_KV == num_q_heads
        and num_kv_heads % 2 == 0
        and page_size == PAGE
    )


def nvfp4_verify_attn(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    k_scales: torch.Tensor,
    v_scales: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    window_left: int,
    bmm1_scale: float,
    bmm2_scale: float,
    out: torch.Tensor | None = None,
    splits: int | None = None,
    use_pdl: bool = False,
) -> torch.Tensor:
    """Attention of 8 consecutive query tokens per request over a paged native-NVFP4 KV pool.

    :param q: ``[R * 8, Hq, 64]`` bf16, heads and dims contiguous; rows of request r are ``8r .. 8r + 7``.
    :param k_pool: ``[slots, Hkv, 32]`` uint8 (or ``[slots, Hkv * 32]``), contiguous: packed E2M1 rows.
    :param k_scales: ``[pages, Hkv, 128, 4]`` E4M3 (any 1-byte dtype), contiguous; ``pages * 128 == slots``.
    :param page_table: ``[>= R, pages]`` int32, inner dimension contiguous.
    :param seq_lens: ``[R]`` int32, KV length per request including the 8 query tokens.
    """
    m, hq = q.shape[0], q.shape[1]
    reqs = m // TOKENS_PER_REQUEST
    kp = k_pool.view(torch.uint8).reshape(k_pool.shape[0], -1)
    vp = v_pool.view(torch.uint8).reshape(v_pool.shape[0], -1)
    hkv = kp.shape[1] // (HEAD_DIM // 2)
    if out is None:
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    s = splits or num_splits(reqs, hkv, q.device, int(window_left))
    ws = workspace(q.device, workspace_floats(reqs, hkv, s))
    _module(bool(use_pdl)).nvfp4_verify_attn(
        q,
        kp,
        vp,
        k_scales.view(torch.uint8),
        v_scales.view(torch.uint8),
        page_table,
        seq_lens,
        out,
        ws,
        int(window_left),
        PAGE,
        int(s),
        float(bmm1_scale),
        float(bmm2_scale),
    )
    return out


def preload(
    max_requests: int,
    num_q_heads: int,
    num_kv_heads: int,
    device,
    use_pdl: bool = False,
) -> None:
    """Build and load the kernels and size the workspace for up to ``max_requests`` requests, so that no build,
    module load, function attribute call or allocation happens inside a CUDA-graph capture."""
    key = (device.type, device.index, max_requests, num_q_heads, num_kv_heads, use_pdl)
    if key in _preloaded:
        return
    workspace(
        device,
        max(
            workspace_floats(n, num_kv_heads, num_splits(n, num_kv_heads, device, w))
            for n in range(1, max_requests + 1)
            for w in (-1, 0)
        ),
    )
    q = torch.zeros(
        TOKENS_PER_REQUEST, num_q_heads, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    pool = torch.zeros(
        2 * PAGE, num_kv_heads, HEAD_DIM // 2, dtype=torch.uint8, device=device
    )
    scales = torch.zeros(
        2, num_kv_heads, PAGE, HEAD_DIM // 16, dtype=torch.uint8, device=device
    )
    table = torch.zeros(1, 2, dtype=torch.int32, device=device)
    lens = torch.full((1,), TOKENS_PER_REQUEST, dtype=torch.int32, device=device)
    nvfp4_verify_attn(
        q, pool, pool, scales, scales, table, lens, -1, 1.0, 1.0, use_pdl=use_pdl
    )
    torch.cuda.current_stream(device).synchronize()
    _preloaded.add(key)


_preloaded: set = set()
