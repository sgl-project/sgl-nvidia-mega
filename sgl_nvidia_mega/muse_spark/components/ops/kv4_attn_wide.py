# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Paged full-attention verify for many requests per step over a native NVFP4 KV cache, JIT CUDA
(``csrc/kv4_attn_wide.cuh``).

The persistent kernel of ``ops.full_attn_wide`` (one thread block per SM, grid fixed at the SM count so the launch is
CUDA-graph safe; the summed key range of all requests split evenly over the blocks; tensor-core MMA with per-tile-maximum
probability quantization; a second kernel merges the pieces of requests that several blocks share), with its K/V tiles
built inside the kernel from the packed E2M1 pools and their E4M3 block scales. The tile handed to the MMA equals the one
``ops.full_attn_wide`` loads from an FP8 pool holding the dequantized values (one e4m3 rounding of the exact
E2M1 x scale product, the arithmetic of ``ops.nvfp4_verify_attn`` and ``ops.nvfp4_prefill_dq``), so the outputs equal
``ops.full_attn_wide`` over that FP8 pool. Covers 8 query tokens per request, 8 query heads per KV head, 2 or 4 KV heads
per rank, head_dim 64, e4m3 Q, page size 128, bf16 output, full attention, at most 256 requests.
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


_CSRC = Path(__file__).resolve().parent.parent / "csrc" / "kv4_attn_wide.cuh"

TOKENS_PER_REQUEST = 8
HEAD_DIM = 64
Q_PER_KV = 8
PAGE = 128
REC_FLOATS = 68
TRACE_FLOATS = (
    3 * 24 * 256 * 2
)  # development timeline (DQ_MODE bit 16) after the partial records
ROWS = 64
MAX_REQUESTS = 256
# FP8 stage ring and raw (packed) stage ring depths; tunable, sweep with bench/bench_kv4_attn.py
F8_STAGES = int(os.environ.get("MUSE_VR_KV4_ATTN_WIDE_F8_STAGES", "3"))
RAW_STAGES = int(os.environ.get("MUSE_VR_KV4_ATTN_WIDE_RAW_STAGES", "3"))
# softmax threads per query row at 4 KV heads per rank (full_attn_wide's served choice: 1); 2 at 2 KV heads
TPR = int(os.environ.get("MUSE_VR_KV4_ATTN_WIDE_TPR", "1"))
# development probe of the dequantization stage (0 = the kernel; others give wrong outputs, timing only)
DQ_MODE = int(os.environ.get("MUSE_VR_KV4_ATTN_WIDE_DQ_MODE", "0"))
# sanitizer runs: a small bound on every mbarrier wait (the kernel traps instead of polling for minutes when slowed)
SAN_POLLS = int(os.environ.get("MUSE_VR_KV4_ATTN_WIDE_SAN_POLLS", "0"))
# development build with 8 dequantization warps (20-warp blocks; default 4)
DQ_WARPS = int(os.environ.get("MUSE_VR_KV4_ATTN_WIDE_DQ_WARPS", "4"))
# skeleton options (bitmask, default 0 = off; every option keeps every output bit): 1 = tile maximum of interior tiles from
# the score loads, 2 = e4m3 probability packing joined in one statement, 4 = dequantization warps with a larger register
# share and a tile's shared-memory loads issued before its stores (4 or 8 dequantization warps); 8 = (with 4) the table
# dequantization's shift as a funnel shift; 16 = a suspend-time hint on the load warp's raw-stage refill wait only;
# 32 = (with 4, 4 KV heads, 4 dequantization warps) a rescale warpgroup keeps the output accumulator in tensor memory
# (20-warp blocks);
# 64 = (with 4) the table dequantization's high selector as a high-word multiply; 128 = (with 4) the table entry address
# by an integer multiply-add; 256 = (probabilities in tensor memory, no 32) the previous tile's output combine before the
# store of the probabilities; 512 = every output column of a tile loaded before one wait in the combine; 1024 = the load
# warp's page-list loads issued eight per lane before their stores
SKEL = int(os.environ.get("MUSE_VR_KV4_ATTN_WIDE_SKEL", "0"))


def _skel_for(hkv: int, skel: int | None = None, dq: int | None = None) -> int:
    """The skeleton options a build at ``hkv`` KV heads takes: option 32 only with option 4 at 4 KV heads, one softmax
    thread per row, probabilities in tensor memory and 4 dequantization warps (other builds drop it)."""
    s = SKEL if skel is None else int(skel)
    dqm = DQ_MODE if dq is None else int(dq)
    if not (
        hkv == 4 and TPR == 1 and (s & 4) and DQ_WARPS == 4 and not (dqm >> 17) & 1
    ):
        s &= ~32
    return s


@functools.lru_cache(maxsize=None)
def _module(
    pairs: int, f8: int, raw: int, use_pdl: bool, tpr: int, dq: int = 0, skel: int = 0
):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(pairs, f8, raw, use_pdl, tpr, dq, skel)
    extra = [f"-DK4W_MAX_POLLS={SAN_POLLS}u"] if SAN_POLLS > 0 else []
    if DQ_WARPS != 4:
        extra.append(f"-DK4W_DQ_WARPS={DQ_WARPS}")
    return load_jit(
        "muse_vr_kv4_attn_wide"
        + (f"_san{SAN_POLLS}" if SAN_POLLS > 0 else "")
        + (f"_dqw{DQ_WARPS}" if DQ_WARPS != 4 else ""),
        *args,
        cuda_files=[str(_CSRC)],
        cuda_wrappers=[("kv4_attn_wide", f"muse_vr::Kv4AttnWideKernel<{args}>::run")],
        extra_cuda_cflags=extra,
    )


@functools.lru_cache(maxsize=None)
def _probe_module():
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "muse_vr_kv4_lut_probe",
        cuda_files=[str(_CSRC)],
        cuda_wrappers=[("kv4_lut_probe", "muse_vr::Kv4LutProbe::run")],
    )


def lut_probe(words: torch.Tensor, scales: torch.Tensor):
    """Test entry: dequantize ``words`` (int32 [n], 8 packed E2M1 values each) with the per-word scale bytes ``scales``
    (uint8 [n]) through the kernel's table path and through the conversion chain; returns (table, chain) uint8 [n, 8]."""
    n = words.numel()
    a = torch.empty(n, 8, dtype=torch.uint8, device=words.device)
    c = torch.empty(n, 8, dtype=torch.uint8, device=words.device)
    _probe_module().kv4_lut_probe(words, scales, a, c)
    return a, c


@functools.lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def num_blocks(device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    raw = os.environ.get("MUSE_VR_KV4_ATTN_WIDE_BLOCKS", "")
    return int(raw) if raw else _sm_count(idx)


_workspace: dict = {}


def workspace(device, num_kv_heads: int, blocks: int):
    """Per-device fp32 partial records and the per-request counters (zero; the kernel leaves them zero). Grows, never
    shrinks; allocate before graph capture."""
    key = (
        device.type,
        device.index if device.index is not None else torch.cuda.current_device(),
    )
    need = blocks * 2 * num_kv_heads * ROWS * REC_FLOATS + TRACE_FLOATS
    ws = _workspace.get(key)
    if ws is None or ws[0].numel() < need:
        counters = (
            ws[1]
            if ws is not None
            else torch.zeros(MAX_REQUESTS, dtype=torch.int32, device=device)
        )
        ws = (torch.empty(need, dtype=torch.float32, device=device), counters)
        _workspace[key] = ws
    return ws


def supported(
    num_q_heads: int, num_kv_heads: int, head_dim: int, page_size: int
) -> bool:
    return (
        head_dim == HEAD_DIM
        and num_kv_heads * Q_PER_KV == num_q_heads
        and num_kv_heads in (2, 4)
        and page_size == PAGE
    )


def kv4_attn_wide(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    k_scales: torch.Tensor,
    v_scales: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    bmm1_scale: float,
    bmm2_scale: float,
    out: torch.Tensor | None = None,
    use_pdl: bool = False,
    blocks: int | None = None,
    f8_stages: int | None = None,
    raw_stages: int | None = None,
    skel: int | None = None,
    dq_mode: int | None = None,
) -> torch.Tensor:
    """Full attention of 8 consecutive query tokens per request over a paged native-NVFP4 KV pool.

    :param q: ``[R * 8, Hq, 64]`` e4m3, heads and dims contiguous, token stride a multiple of 16 bytes.
    :param k_pool: ``[slots, Hkv, 32]`` or ``[slots, Hkv * 32]`` uint8, contiguous: packed E2M1 rows (slot = page * 128
        + offset); Hkv 2 or 4.
    :param k_scales: ``[pages, Hkv, 128, 4]`` E4M3 (any 1-byte dtype), contiguous.
    :param v_pool: / ``v_scales``: same layouts (V scales four-token interleaved inside a page).
    :param page_table: ``[>= R, pages]`` int32, inner dimension contiguous.
    :param seq_lens: ``[R]`` int32, KV length per request including the 8 query tokens.
    :param blocks: thread blocks (default: the SM count; ``MUSE_VR_KV4_ATTN_WIDE_BLOCKS`` overrides).
    :param skel: / ``dq_mode``: per-call overrides of ``MUSE_VR_KV4_ATTN_WIDE_SKEL`` / ``MUSE_VR_KV4_ATTN_WIDE_DQ_MODE``
        (benchmarks compare builds in one process; a nonzero development mode gives wrong outputs).
    """
    kp = k_pool.view(torch.uint8).reshape(k_pool.shape[0], -1)
    vp = v_pool.view(torch.uint8).reshape(v_pool.shape[0], -1)
    hkv = kp.shape[1] // (HEAD_DIM // 2)
    m, hq = q.shape[0], q.shape[1]
    if out is None:
        out = torch.empty(m, hq, HEAD_DIM, dtype=torch.bfloat16, device=q.device)
    g = blocks or num_blocks(q.device)
    ws, counters = workspace(q.device, hkv, g)
    _module(
        hkv // 2,
        f8_stages or F8_STAGES,
        raw_stages or RAW_STAGES,
        bool(use_pdl),
        TPR if hkv == 4 else 2,
        DQ_MODE if dq_mode is None else int(dq_mode),
        _skel_for(hkv, skel, dq_mode),
    ).kv4_attn_wide(
        q,
        kp,
        vp,
        k_scales.view(torch.uint8),
        v_scales.view(torch.uint8),
        page_table,
        seq_lens,
        out,
        ws,
        counters,
        int(g),
        float(bmm1_scale),
        float(bmm2_scale),
    )
    return out


def preload(num_q_heads: int, num_kv_heads: int, device, use_pdl: bool = False) -> None:
    """Build and load the kernel and size the workspace, so that no build, module load, function attribute call or
    allocation happens inside a CUDA-graph capture."""
    key = (
        device.type,
        device.index if device.index is not None else torch.cuda.current_device(),
        num_q_heads,
        num_kv_heads,
        use_pdl,
    )
    if key in _preloaded:
        return
    q = torch.zeros(
        TOKENS_PER_REQUEST,
        num_q_heads,
        HEAD_DIM,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    pool = torch.zeros(
        2 * PAGE, num_kv_heads, HEAD_DIM // 2, dtype=torch.uint8, device=device
    )
    scales = torch.zeros(
        2, num_kv_heads, PAGE, HEAD_DIM // 16, dtype=torch.uint8, device=device
    )
    table = torch.zeros(1, 2, dtype=torch.int32, device=device)
    lens = torch.full((1,), TOKENS_PER_REQUEST, dtype=torch.int32, device=device)
    kv4_attn_wide(q, pool, pool, scales, scales, table, lens, 1.0, 1.0, use_pdl=use_pdl)
    torch.cuda.current_stream(device).synchronize()
    _preloaded.add(key)


_preloaded: set = set()
