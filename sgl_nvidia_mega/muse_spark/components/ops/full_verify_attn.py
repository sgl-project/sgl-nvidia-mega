# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Paged FP8 verify attention on tensor-core MMA (tcgen05) with a length-proportional key split, JIT CUDA.

See ``csrc/full_verify_attn.cuh`` for the arithmetic, the work split and the synchronization. The op covers the
verify step of attention layers with 8 query heads per KV head, head_dim 64, an even number of KV heads per rank,
e4m3 Q/K/V, page size 128 and bf16 output, for full attention (``window_left < 0``) or a sliding window.
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


_CSRC = Path(__file__).resolve().parent.parent / "csrc" / "full_verify_attn.cuh"

TOKENS_PER_REQUEST = 8
HEAD_DIM = 64
Q_PER_KV = 8
PAGE = 128
REC_FLOATS = 68
ROWS = 64

# Thread blocks per call, as a multiple of the SM count: blocks = round(factor * SMs), split evenly over the
# (request, KV-head pair) units. Tunable (bench/bench_full_verify_attn.py); MUSE_VR_FULL_VERIFY_ATTN_BLOCKS overrides.
_BLOCKS_PER_SM = 1.0


@functools.lru_cache(maxsize=None)
def _module(
    use_pdl: bool,
    cluster: bool = False,
    norm_gate: bool = False,
    qk: int = 0,
    unfused: bool = False,
):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(use_pdl, cluster, norm_gate, qk, unfused)
    return load_jit(
        "muse_vr_full_verify_attn",
        *args,
        cuda_files=[str(_CSRC)],
        cuda_wrappers=[
            ("full_verify_attn", f"muse_vr::FullVerifyAttnKernel<{args}>::run")
        ],
    )


@functools.lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def num_splits(num_requests: int, num_kv_heads: int, device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    raw = os.environ.get("MUSE_VR_FULL_VERIFY_ATTN_BLOCKS", "")
    blocks = int(raw) if raw else round(_BLOCKS_PER_SM * _sm_count(idx))
    units = max(1, num_requests * (num_kv_heads // 2))
    return max(1, blocks // units)


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


def full_verify_attn(
    q: torch.Tensor | None,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    window_left: int,
    page_size: int,
    bmm1_scale: float,
    bmm2_scale: float,
    out: torch.Tensor | None = None,
    splits: int | None = None,
    use_pdl: bool = False,
    gate: torch.Tensor | None = None,
    norm_eps: float = 0.0,
    cluster: bool | None = None,
    qkv: torch.Tensor | None = None,
    cache_loc: torch.Tensor | None = None,
    qk_eps: float = 0.0,
    num_q_heads: int | None = None,
) -> torch.Tensor:
    """Attention of 8 consecutive query tokens per request over a paged FP8 KV pool (full attention when
    ``window_left < 0``, else keys [p - window_left, p] for the token at position p).

    :param q: ``[R * 8, Hq, 64]`` e4m3, heads and dims contiguous; rows of request r are ``8r .. 8r + 7``. None with
        ``qkv``.
    :param k_pool: ``[slots, Hkv * 64]`` e4m3, contiguous (slot = page * 128 + offset).
    :param v_pool: same layout as ``k_pool``.
    :param page_table: ``[>= R, pages]`` int32, inner dimension contiguous.
    :param seq_lens: ``[R]`` int32, KV length per request including the 8 query tokens.
    :param gate: optional ``[R * 8, >= Hq * 64]`` bf16 (inner dimension contiguous): when given, the output is
        ``rms_norm(attention, eps=norm_eps) * sigmoid(gate)`` per (token, head) row, computed as SGLang's norm_gate
        kernel computes it for up to 4 requests (see the kernel source).
    :param cluster: merge the splits of a (request, KV-head pair) inside a thread-block cluster (distributed shared
        memory, no second kernel; needs 2 to 16 splits, else the merge kernel is used). Default: when the split
        count is at most
        ``MUSE_VR_FULL_VERIFY_ATTN_CLUSTER_MAX`` (default 0, i.e. off: measured slower than the merge kernel with
        16 clusters of 13 blocks, bench/bench_full_verify_attn.py --cluster 0,1 at 8 requests).
    :param qkv: query/key prologue (layers without RoPE): ``[R * 8, >= (2 Hq + 2 Hkv) * 64]`` bf16 QKV projection
        output (q, k, v, gate columns), inner dimension contiguous. The kernel then does the work of SGLang's fused
        QK-norm call with an FP8 KV-cache store (``FusedQKNorm._forward_triton`` without RoPE) itself: q from the QK
        norm of the q columns, and the K/V rows of the 8 new tokens of every request, stored into the pools at
        ``cache_loc`` (``[R * 8]`` int64; 0: not stored). Needs ``num_q_heads`` and a row count that
        ``swa_verify_attn.qk_class`` covers.
    """
    hkv = k_pool.shape[1] // HEAD_DIM
    if qkv is not None:
        from .swa_verify_attn import qk_class

        m, hq = qkv.shape[0], int(num_q_heads)
        qk = qk_class(m, hq, hkv)
        if qk is None:
            raise ValueError("query/key prologue: row count not covered")
        dev = qkv.device
    else:
        m, hq, qk, dev = q.shape[0], q.shape[1], 0, q.device
    reqs = m // TOKENS_PER_REQUEST
    if out is None:
        out = torch.empty(m, hq, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    s = splits or num_splits(reqs, hkv, dev)
    if cluster is None:
        cluster = s <= int(os.environ.get("MUSE_VR_FULL_VERIFY_ATTN_CLUSTER_MAX", "0"))
    cluster = cluster and 2 <= s <= 16
    ws = workspace(dev, workspace_floats(reqs, hkv, s) if not cluster else 1)
    flat = out.view(out.shape[0], -1)
    from .swa_verify_attn import qk_unfused

    _module(
        bool(use_pdl), bool(cluster), gate is not None, qk, qk_unfused(qk)
    ).full_verify_attn(
        q if q is not None else out,
        k_pool,
        v_pool,
        page_table,
        seq_lens,
        out,
        ws,
        int(window_left),
        int(page_size),
        int(s),
        float(bmm1_scale),
        float(bmm2_scale),
        gate if gate is not None else flat,
        float(norm_eps),
        qkv if qkv is not None else flat,
        cache_loc if cache_loc is not None else seq_lens,
        float(qk_eps),
    )
    return out


def preload(
    max_requests: int,
    num_q_heads: int,
    num_kv_heads: int,
    device,
    norm_gate: bool = False,
    use_pdl: bool = False,
    qk_max_requests: int = 0,
) -> None:
    """Build and load the kernels and size the workspace for up to ``max_requests`` requests (and the query/key
    prologue variants for up to ``qk_max_requests``), so that no build, module load, function attribute call or
    allocation happens inside a CUDA-graph capture."""
    key = (
        device.type,
        device.index,
        max_requests,
        num_q_heads,
        num_kv_heads,
        norm_gate,
        use_pdl,
        qk_max_requests,
    )
    if key in _preloaded:
        return
    workspace(
        device,
        max(
            workspace_floats(n, num_kv_heads, num_splits(n, num_kv_heads, device))
            for n in range(1, max_requests + 1)
        ),
    )
    q = torch.zeros(
        TOKENS_PER_REQUEST,
        num_q_heads,
        HEAD_DIM,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    pool = torch.zeros(
        2 * PAGE, num_kv_heads * HEAD_DIM, dtype=torch.float8_e4m3fn, device=device
    )
    table = torch.zeros(1, 2, dtype=torch.int32, device=device)
    lens = torch.full((1,), TOKENS_PER_REQUEST, dtype=torch.int32, device=device)
    gate = torch.zeros(
        TOKENS_PER_REQUEST, num_q_heads * HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    full_verify_attn(
        q,
        pool,
        pool,
        table,
        lens,
        -1,
        PAGE,
        1.0,
        1.0,
        use_pdl=use_pdl,
        gate=gate if norm_gate else None,
        norm_eps=1e-6,
    )
    from .swa_verify_attn import qk_class

    width = (2 * num_q_heads + 2 * num_kv_heads) * HEAD_DIM
    first = {}  # prologue column count -> the smallest request count that uses it
    for n in range(1, qk_max_requests + 1):
        first.setdefault(qk_class(n * TOKENS_PER_REQUEST, num_q_heads, num_kv_heads), n)
    for qk, n in first.items():
        if qk is None:
            continue
        rows = n * TOKENS_PER_REQUEST
        g = torch.zeros(
            rows, num_q_heads * HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        full_verify_attn(
            None,
            pool,
            pool,
            table.expand(n, -1).contiguous(),
            lens.expand(n).contiguous(),
            -1,
            PAGE,
            1.0,
            1.0,
            use_pdl=use_pdl,
            gate=g if norm_gate else None,
            norm_eps=1e-6,
            qkv=torch.zeros(rows, width, dtype=torch.bfloat16, device=device),
            cache_loc=torch.zeros(rows, dtype=torch.int64, device=device),
            qk_eps=1e-6,
            num_q_heads=num_q_heads,
        )
    torch.cuda.current_stream(device).synchronize()
    _preloaded.add(key)


_preloaded: set = set()
