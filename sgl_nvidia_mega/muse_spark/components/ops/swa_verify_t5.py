# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Sliding-window FP8 paged attention for speculative verify on the 5th-generation tensor core, JIT CUDA.

See ``csrc/swa_verify_t5.cuh`` for the arithmetic (the rounding points of ``ops.swa_verify_attn`` and of the stock
trtllm-gen kernel) and the work split. Shape class: 8 query heads per KV head, head_dim 64, e4m3 Q/K/V, bf16 output,
128-token pages, 8 query tokens per request. Not wired into any hook yet (opt-in by direct call).

The key range of every (request, KV head) is split into S runs of whole pages, one thread block each; the blocks
write fp32 partial records into a workspace and the last block of each (request, KV head) merges them. S defaults
to about two blocks per SM over all (request, KV head) pairs, clamped to 3..16 (at most 6 pages per block);
MUSE_VR_SWA_T5_SPLITS forces it.

MUSE_VR_SWA_T5_IMPL=6 selects the t6 kernel instead (same arithmetic class): one block per (request, KV head, split)
with three page pipelines sharing a 12-page shared-memory ring that one warp keeps full with TMA page loads; default
one block per (request, KV head). With MUSE_VR_SWA_T6_CLUSTER=1, 2..3 blocks per pair run as one cluster and merge
through shared memory (no workspace round trip); the default split is then the largest of 3, 2 whose grid fits in one
wave of resident clusters (runtime occupancy query), else 1. The workspace (partials + self-resetting arrival counters) is
allocated per device on first use, outside any CUDA-graph capture (``prepare``). For t6, ``prefetch`` with ``use_pdl``
signals the next kernel at the start and issues L2 prefetches of the block's K / V boxes before the PDL wait.
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


_CSRC = Path(__file__).resolve().parent.parent / "csrc" / "swa_verify_t5.cuh"

TOKENS_PER_REQUEST = 8
HEAD_DIM = 64
Q_PER_KV = 8
PAGE = 128
MAX_SPLITS = 16
MAX_PAGES_PER_BLOCK = 6
REC_FLOATS = 68


# t6 shape dispatch: a call of M = requests x 8 query tokens falls in the smallest bucket of T6_BUCKETS that holds M
# (none above the largest). The t6 kernel is meant to be engaged only for the buckets listed in MUSE_VR_SWA_T6_BUCKETS
# (comma list of bucket sizes; unset or empty = none); callers ask t6_engaged(M) and use the served kernel otherwise.
T6_BUCKETS = (64, 128, 192, 256)
_t6_step_aside_seen: set = set()


def t6_bucket(num_tokens: int) -> int | None:
    """The t6 dispatch bucket of a call with ``num_tokens`` query tokens, or None above the largest bucket."""
    for b in T6_BUCKETS:
        if num_tokens <= b:
            return b
    return None


def t6_enabled_buckets() -> frozenset:
    raw = os.environ.get("MUSE_VR_SWA_T6_BUCKETS", "")
    return frozenset(int(x) for x in raw.split(",") if x.strip())


def t6_engaged(num_tokens: int) -> bool:
    """True iff a call with ``num_tokens`` query tokens should use the t6 kernel; otherwise the caller steps aside to
    the served kernel (announced once per bucket on stdout)."""
    b = t6_bucket(int(num_tokens))
    if b is not None and b in t6_enabled_buckets():
        return True
    if b not in _t6_step_aside_seen:
        _t6_step_aside_seen.add(b)
        print(
            f"muse_vr_plugin: swa_verify t6 steps aside to the served kernel for {num_tokens} query tokens "
            f"(bucket {b if b is not None else 'none'} not enabled in MUSE_VR_SWA_T6_BUCKETS)",
            flush=True,
        )
    return False


def max_pages(window_left: int) -> int:
    """Pages a window of window_left + 8 keys can touch."""
    return (int(window_left) + TOKENS_PER_REQUEST + PAGE - 1) // PAGE + 1


def pages_per_block(window_left: int, splits: int) -> int:
    return -(-max_pages(window_left) // int(splits))


@functools.lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def splits_for(num_requests: int, num_kv_heads: int, window_left: int, device) -> int:
    raw = os.environ.get("MUSE_VR_SWA_T5_SPLITS")
    if raw:
        s = int(raw)
    else:
        dev = torch.device(device)
        sms = _sm_count(
            dev.index if dev.index is not None else torch.cuda.current_device()
        )
        s = (2 * sms) // max(1, num_requests * num_kv_heads)
    s = max(1, min(MAX_SPLITS, s))
    while pages_per_block(window_left, s) > MAX_PAGES_PER_BLOCK and s < MAX_SPLITS:
        s += 1
    return s


_ws: dict = {}


def workspace(device, num_requests: int, num_kv_heads: int, splits: int):
    """(partials fp32, counters int32) large enough for this call; grows only (call ``prepare`` before a capture)."""
    dev = torch.device(device)
    need_p = num_requests * num_kv_heads * splits * 64 * REC_FLOATS
    need_c = 2 * num_requests * num_kv_heads
    cur = _ws.get(dev)
    if cur is None or cur[0].numel() < need_p or cur[1].numel() < need_c:
        np_ = max(need_p, cur[0].numel() if cur else 0)
        nc = max(need_c, cur[1].numel() if cur else 0)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "swa_verify_t5 workspace must be prepared before CUDA-graph capture"
            )
        cur = (
            torch.empty(np_, dtype=torch.float32, device=dev),
            torch.zeros(nc, dtype=torch.int32, device=dev),
        )
        _ws[dev] = cur
    return cur


def prepare(device, max_requests: int, num_kv_heads: int, window_left: int) -> None:
    """Allocate the workspace for every request count up to max_requests (no allocation inside a capture)."""
    worst = max(
        n * splits_for(n, num_kv_heads, window_left, device)
        for n in range(1, max_requests + 1)
    )
    workspace(device, worst, num_kv_heads, 1)
    workspace(device, max_requests, num_kv_heads, 1)


def supported(
    num_q_heads: int, num_kv_heads: int, head_dim: int, window_left: int, page_size: int
) -> bool:
    return (
        head_dim == HEAD_DIM
        and num_kv_heads * Q_PER_KV == num_q_heads
        and window_left >= 0
        and page_size == PAGE
        and pages_per_block(window_left, MAX_SPLITS) <= MAX_PAGES_PER_BLOCK
    )


@functools.lru_cache(maxsize=None)
def _module6(use_pdl: bool, cluster: bool = False, early: bool = False):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(use_pdl, cluster, early)
    return load_jit(
        "muse_vr_swa_verify_t6",
        *args,
        cuda_files=[str(_CSRC)],
        cuda_wrappers=[
            ("swa_verify_t6", f"muse_vr::SwaVerifyT6Kernel<{args}>::run"),
            ("swa_verify_t6_capacity", f"muse_vr::SwaVerifyT6Kernel<{args}>::capacity"),
        ],
    )


@functools.lru_cache(maxsize=None)
def _capacity6(device_index: int, splits: int, use_pdl: bool, early: bool) -> int:
    """Blocks of the t6 instantiation used at `splits` that can be resident on the device at once (runtime query)."""
    out = torch.zeros(1, dtype=torch.int32)
    with torch.cuda.device(device_index):
        _module6(use_pdl, splits > 1, early).swa_verify_t6_capacity(out, int(splits))
    return int(out.item())


def splits_for6(
    num_requests: int, num_kv_heads: int, device, use_pdl: bool = False
) -> int:
    """Blocks per (request, KV head) for t6: with the cluster merge enabled, the largest of 3, 2 whose grid fits in one
    wave of resident clusters (runtime capacity query), else 1; without it, 1 (the split-K merge through global memory
    cost more than the shorter page run saved, DEV bench). MUSE_VR_SWA_T5_SPLITS forces a count."""
    raw = os.environ.get("MUSE_VR_SWA_T5_SPLITS")
    if raw:
        return max(1, min(MAX_SPLITS, int(raw)))
    if os.environ.get("MUSE_VR_SWA_T6_CLUSTER", "0") != "1":
        return 1
    dev = torch.device(device)
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    pairs = int(num_requests) * int(num_kv_heads)
    for s in (3, 2):
        if pairs * s <= _capacity6(idx, s, bool(use_pdl), bool(use_pdl)):
            return s
    return 1


@functools.lru_cache(maxsize=None)
def _module(kpages: int, use_pdl: bool, prefetch: bool = False):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(kpages, use_pdl, prefetch)
    return load_jit(
        "muse_vr_swa_verify_t5",
        *args,
        cuda_files=[str(_CSRC)],
        cuda_wrappers=[("swa_verify_t5", f"muse_vr::SwaVerifyT5Kernel<{args}>::run")],
    )


def swa_verify_attn_t5(
    q: torch.Tensor,
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
    prefetch: bool | None = False,
) -> torch.Tensor:
    """Attention of 8 consecutive query tokens per request over a sliding window of a paged FP8 KV pool.

    Arguments as ``ops.swa_verify_attn.swa_verify_attn`` (q ``[R * 8, Hq, 64]`` e4m3; pools ``[slots, Hkv * 64]``
    e4m3; page_table ``[>= R, pages]`` int32; seq_lens ``[R]`` int32 including the 8 query tokens); page_size must be
    128. ``splits``: blocks per (request, KV head), 1..16 (default ``splits_for`` / ``splits_for6``). ``prefetch``
    None selects the t6 policy (t5: off).
    """
    m, hq = q.shape[0], q.shape[1]
    if out is None:
        out = torch.empty(m, hq, HEAD_DIM, dtype=torch.bfloat16, device=k_pool.device)
    reqs = m // TOKENS_PER_REQUEST
    hkv = k_pool.shape[1] // HEAD_DIM
    if os.environ.get("MUSE_VR_SWA_T5_IMPL", "5") == "6":
        s6 = int(splits) if splits else splits_for6(reqs, hkv, k_pool.device, use_pdl)
        part, count = workspace(k_pool.device, reqs, hkv, s6)
        cl = 1 < s6 <= 3 and os.environ.get("MUSE_VR_SWA_T6_CLUSTER", "0") == "1"
        # prefetch=None: the policy (pre-wait L2 prefetch only with 3 blocks per pair; at 1-2 blocks per pair it
        # measured slower in the DEV bench)
        pf = (cl and s6 == 3) if prefetch is None else bool(prefetch)
        _module6(bool(use_pdl), cl, bool(use_pdl) and pf).swa_verify_t6(
            q,
            k_pool,
            v_pool,
            page_table,
            seq_lens,
            out,
            part,
            count,
            int(window_left),
            int(page_size),
            float(bmm1_scale),
            float(bmm2_scale),
            s6,
        )
        return out
    s = int(splits) if splits else splits_for(reqs, hkv, window_left, k_pool.device)
    kp = pages_per_block(window_left, s)
    if kp > MAX_PAGES_PER_BLOCK:
        raise ValueError(
            f"window_left {window_left} needs more than {MAX_PAGES_PER_BLOCK} pages per block at {s} splits"
        )
    part, count = workspace(k_pool.device, reqs, hkv, s)
    _module(kp, bool(use_pdl), bool(prefetch or False)).swa_verify_t5(
        q,
        k_pool,
        v_pool,
        page_table,
        seq_lens,
        out,
        part,
        count,
        int(window_left),
        int(page_size),
        float(bmm1_scale),
        float(bmm2_scale),
        s,
    )
    return out
