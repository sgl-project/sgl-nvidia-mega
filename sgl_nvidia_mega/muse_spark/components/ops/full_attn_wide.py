# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Paged FP8 full-attention verify for many requests per step, JIT CUDA (``csrc/full_attn_wide.cuh``).

A persistent kernel (one thread block per SM, grid fixed at the SM count so the launch is CUDA-graph safe) splits the
summed key range of all requests evenly over its blocks, reads whole token rows of the K/V pools (every KV head of the
rank) and runs the attention on tensor-core MMA with per-tile-maximum probability quantization (as
``ops.full_verify_attn``); a second kernel merges the pieces of requests that several blocks share. Covers 8 query tokens per request, 8 query heads per KV head,
2 or 4 KV heads per rank, head_dim 64, e4m3 Q/K/V, page size 128, bf16 output, full attention, at most 256 requests.

With ``gate`` the main kernel is unchanged; the merge kernel writes every request (one that lies inside one block: the
main kernel's output row read back) and also applies the output norm (no weight) and the sigmoid gate with SGLang's
norm_gate arithmetic in its 8- or 16-byte configuration (bitwise to running this kernel and then that norm_gate
configuration: ``tests/test_full_attn_wide_norm_gate.py``).
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


_CSRC = Path(__file__).resolve().parent.parent / "csrc" / "full_attn_wide.cuh"

TOKENS_PER_REQUEST = 8
HEAD_DIM = 64
Q_PER_KV = 8
PAGE = 128
REC_FLOATS = 68
ROWS = 64
MAX_REQUESTS = 256
# fewest requests per step at which the kernel was faster than the stock decode kernel in the microbench, per KV heads
# per rank (below it the stock path runs); MUSE_VR_FULL_ATTN_WIDE_MIN_REQS overrides it
BREAK_EVEN_REQUESTS = {4: 10, 2: 9}
TRACE_FLOATS = (
    3 * 16 * 256 * 2
)  # development mode 4: clock stamps after the partial records
# K/V ring depth (5: one Q buffer, the fastest ring in the microbench at both KV head counts)
STAGES = int(os.environ.get("MUSE_VR_FULL_ATTN_WIDE_STAGES", "5"))
# keys of every 8 whose 2^x is computed on the FMA pipes instead of the transcendental unit (even, 0..8)
POLY = int(os.environ.get("MUSE_VR_FULL_ATTN_WIDE_POLY", "0"))
# tiles ahead whose K/V boxes are prefetched into L2 (0: off); development probes: MODE 1 = load-only (no attention)
PF = int(os.environ.get("MUSE_VR_FULL_ATTN_WIDE_PF", "0"))
MODE = int(os.environ.get("MUSE_VR_FULL_ATTN_WIDE_MODE", "0"))
# softmax threads per query row at 4 KV heads per rank (1: faster in the microbench; 2: the two-thread layout used at 2 KV
# heads per rank, where a 128-key tile per thread would not fit the registers)
TPR = int(os.environ.get("MUSE_VR_FULL_ATTN_WIDE_TPR", "1"))
# Skeleton options of the main kernel (bitmask, default 0 = the default code). Every option keeps every output bit
# (tests/test_full_attn_wide_skel.py): 1 = interior-tile score maximum from the score loads, 2 = e4m3 probability packing
# joined in one statement, 4 = one wait for a tile's output loads, 8 = page-list loads in flight together, 16 = suspend-time
# hint on the load warp's ring-refill wait, 32 = probabilities in tensor memory (needs 64 keys per softmax thread: not with
# MUSE_VR_FULL_ATTN_WIDE_TPR=2 at 4 KV heads, where that bit is dropped with a warning).
SKEL_BITS = 63


def _skel_from_env() -> int:
    raw = os.environ.get("MUSE_VR_FULL_ATTN_WIDE_SKEL", "0").strip() or "0"
    try:
        v = int(raw, 0)
    except ValueError:
        v = -1
    if v < 0 or v & ~SKEL_BITS:
        import logging

        logging.getLogger(__name__).warning(
            "muse_vr_plugin: full_attn_wide: MUSE_VR_FULL_ATTN_WIDE_SKEL=%r is not a mask of bits 0-5; using 0 "
            "(the default code)",
            raw,
        )
        return 0
    return v


SKEL = _skel_from_env()
_skel_warned: set = set()


def _skel_for(skel: int, pairs: int, tpr: int) -> int:
    """The skeleton options a call builds with: bit 5 needs 64 keys per softmax thread (dropped, with a warning, at
    4 KV heads with two softmax threads per row)."""
    if skel & 32 and pairs == 2 and tpr == 2:
        if skel not in _skel_warned:
            _skel_warned.add(skel)
            import logging

            logging.getLogger(__name__).warning(
                "muse_vr_plugin: full_attn_wide: skeleton option 32 needs one softmax thread per row at 4 KV heads; "
                "building SKEL=%d without it",
                skel,
            )
        skel &= ~32
    return skel


def effective_skel(num_kv_heads: int) -> int:
    """The skeleton options the served calls build with at this KV-head count (MUSE_VR_FULL_ATTN_WIDE_SKEL; 0 after a
    failed load-time self-check)."""
    if _skel_failed:
        return 0
    return _skel_for(SKEL, num_kv_heads // 2, TPR if num_kv_heads == 4 else 2)


# load-time bitwise self-check of the skeleton options (preload): key -> passed
_skel_checks: dict = {}
_skel_failed = False
_SELFCHECK_LENS = (129, 3001, 4097, 700, 2500, 64, 8, 1999, 3500, 1200, 257, 4000)


def skel_selfchecks() -> dict:
    """Results of the load-time self-checks so far: (device index, KV heads, PDL, gate) -> passed."""
    return dict(_skel_checks)


def _skel_selfcheck(
    device, num_q_heads: int, num_kv_heads: int, use_pdl: bool, gate: int
) -> None:
    """Run a seeded multi-request case (edge tiles, requests split over blocks) with the default build and with the
    configured skeleton options; any output difference -> loud warning and SKEL 0 for every later call of this process."""
    global _skel_failed
    sk = effective_skel(num_kv_heads)
    key = (device.index, num_kv_heads, bool(use_pdl), int(gate))
    if sk == 0 or key in _skel_checks:
        return
    g = torch.Generator(device=device).manual_seed(20261023)
    lens = list(_SELFCHECK_LENS)
    need = [(n + PAGE - 1) // PAGE for n in lens]
    rows = sum(need) * PAGE
    k_pool = torch.randn(rows, num_kv_heads * HEAD_DIM, generator=g, device=device).to(
        torch.float8_e4m3fn
    )
    v_pool = torch.randn(rows, num_kv_heads * HEAD_DIM, generator=g, device=device).to(
        torch.float8_e4m3fn
    )
    table = torch.zeros(len(lens), max(need), dtype=torch.int32, device=device)
    nxt = 0
    for r, n in enumerate(need):
        table[r, :n] = torch.arange(nxt, nxt + n, dtype=torch.int32, device=device)
        nxt += n
    seq = torch.tensor(lens, dtype=torch.int32, device=device)
    q = torch.randn(
        len(lens) * TOKENS_PER_REQUEST,
        num_q_heads,
        HEAD_DIM,
        generator=g,
        device=device,
    ).to(torch.float8_e4m3fn)
    gt = None
    if gate:
        gt = torch.randn(
            len(lens) * TOKENS_PER_REQUEST,
            num_q_heads * HEAD_DIM,
            generator=g,
            device=device,
        ).to(torch.bfloat16)
    outs = []
    for s in (0, sk):
        outs.append(
            full_attn_wide(
                q,
                k_pool,
                v_pool,
                table,
                seq,
                0.125,
                1.0,
                use_pdl=use_pdl,
                gate=gt,
                norm_eps=1e-6,
                ng_vec=int(gate) if gate else 8,
                skel=s,
            )
        )
    ok = bool(torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16)))
    _skel_checks[key] = ok
    import logging

    log = logging.getLogger(__name__)
    if ok:
        log.info(
            "muse_vr_plugin: full_attn_wide: skeleton options %d self-check passed (%s)",
            sk,
            key,
        )
    else:
        n = int((outs[0].view(torch.int16) != outs[1].view(torch.int16)).sum().item())
        _skel_failed = True
        log.warning(
            "muse_vr_plugin: full_attn_wide: skeleton options %d self-check FAILED (%s): %d output elements "
            "differ from the default build; every later call uses the default build (SKEL 0)",
            sk,
            key,
            n,
        )


@functools.lru_cache(maxsize=None)
def _module(
    pairs: int,
    stages: int,
    use_pdl: bool,
    poly: int,
    pf: int = 0,
    mode: int = 0,
    tpr: int = 2,
    ng: int = 0,
    skel: int = 0,
):
    """ng: 0 = no output norm + gate; 8 / 16 = with it, in SGLang norm_gate's 8- / 16-byte-vector arithmetic. skel: the
    main kernel's skeleton options (0: the template arguments, and so the build, of the default code)."""
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    if ng not in (0, 8, 16):
        raise ValueError(f"ng {ng}: 0, 8 or 16")
    if skel < 0 or skel & ~SKEL_BITS:
        raise ValueError(f"skel {skel}: a mask of bits 0-5")
    if skel:
        args = make_cpp_args(
            pairs, stages, use_pdl, poly, pf, mode, tpr, ng != 0, ng or 8, skel
        )
    else:
        args = make_cpp_args(
            pairs, stages, use_pdl, poly, pf, mode, tpr, ng != 0, ng or 8
        )
    return load_jit(
        "muse_vr_full_attn_wide",
        *args,
        cuda_files=[str(_CSRC)],
        cuda_wrappers=[("full_attn_wide", f"muse_vr::FullAttnWideKernel<{args}>::run")],
    )


@functools.lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def num_blocks(device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    raw = os.environ.get("MUSE_VR_FULL_ATTN_WIDE_BLOCKS", "")
    return int(raw) if raw else _sm_count(idx)


_workspace: dict = {}


def workspace(device, num_kv_heads: int, blocks: int, gate: bool = False):
    """Per-device fp32 partial records and the per-request counters (zero; the kernel leaves them zero). Grows, never
    shrinks; allocate before graph capture. ``gate`` is accepted for callers; the norm + gate variant needs no extra
    records."""
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


def full_attn_wide(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    bmm1_scale: float,
    bmm2_scale: float,
    out: torch.Tensor | None = None,
    use_pdl: bool = False,
    blocks: int | None = None,
    stages: int | None = None,
    poly: int | None = None,
    gate: torch.Tensor | None = None,
    norm_eps: float = 0.0,
    ng_vec: int = 8,
    skel: int | None = None,
) -> torch.Tensor:
    """Full attention of 8 consecutive query tokens per request over a paged FP8 KV pool.

    :param q: ``[R * 8, Hq, 64]`` e4m3, heads and dims contiguous, token stride a multiple of 16 bytes.
    :param k_pool: ``[slots, Hkv * 64]`` e4m3, contiguous (slot = page * 128 + offset); Hkv 2 or 4.
    :param v_pool: same layout as ``k_pool``.
    :param page_table: ``[>= R, pages]`` int32, inner dimension contiguous.
    :param seq_lens: ``[R]`` int32, KV length per request including the 8 query tokens.
    :param blocks: thread blocks (default: the SM count; ``MUSE_VR_FULL_ATTN_WIDE_BLOCKS`` overrides).
    :param gate: ``[R * 8, >= Hq * 64]`` bf16 (inner contiguous): also apply the output norm (eps ``norm_eps``, no weight)
        and the sigmoid gate, SGLang's norm_gate arithmetic in its ``ng_vec``-byte configuration (8: 16 threads per row,
        16: 8 threads per row) - the configuration SGLang would pick for this call.
    :param skel: skeleton options of the main kernel (default: ``MUSE_VR_FULL_ATTN_WIDE_SKEL``, 0 = the default code; the
        same output bits for every value).
    """
    hkv = k_pool.shape[1] // HEAD_DIM
    m, hq = q.shape[0], q.shape[1]
    if out is None:
        out = torch.empty(m, hq, HEAD_DIM, dtype=torch.bfloat16, device=q.device)
    g = blocks or num_blocks(q.device)
    use_gate = gate is not None
    ws, counters = workspace(q.device, hkv, g, use_gate)
    gt = gate if use_gate else _no_gate(q.device)
    tpr = TPR if hkv == 4 else 2
    sk = (
        _skel_for(SKEL if skel is None else int(skel), hkv // 2, tpr)
        if (skel is not None or not _skel_failed)
        else 0
    )
    _module(
        hkv // 2,
        stages or STAGES,
        bool(use_pdl),
        POLY if poly is None else int(poly),
        PF,
        MODE,
        tpr,
        int(ng_vec) if use_gate else 0,
        sk,
    ).full_attn_wide(
        q,
        k_pool,
        v_pool,
        page_table,
        seq_lens,
        out,
        ws,
        counters,
        int(g),
        float(bmm1_scale),
        float(bmm2_scale),
        gt,
        float(norm_eps),
    )
    return out


_NO_GATE: dict = {}


def _no_gate(device) -> torch.Tensor:
    key = (device.type, device.index)
    t = _NO_GATE.get(key)
    if t is None:
        t = torch.zeros(1, dtype=torch.bfloat16, device=device)
        _NO_GATE[key] = t
    return t


def preload(
    num_q_heads: int, num_kv_heads: int, device, use_pdl: bool = False, gate: int = 0
) -> None:
    """Build and load the kernel and size the workspace, so that no build, module load, function attribute call or
    allocation happens inside a CUDA-graph capture. gate: 0, or the norm_gate vector width 8 / 16 of the gated variant."""
    key = (
        device.type,
        device.index if device.index is not None else torch.cuda.current_device(),
        num_q_heads,
        num_kv_heads,
        use_pdl,
    ) + ((int(gate),) if gate else ())
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
        2 * PAGE, num_kv_heads * HEAD_DIM, dtype=torch.float8_e4m3fn, device=device
    )
    table = torch.zeros(1, 2, dtype=torch.int32, device=device)
    lens = torch.full((1,), TOKENS_PER_REQUEST, dtype=torch.int32, device=device)
    g = (
        torch.zeros(
            TOKENS_PER_REQUEST,
            num_q_heads * HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        if gate
        else None
    )
    full_attn_wide(
        q,
        pool,
        pool,
        table,
        lens,
        1.0,
        1.0,
        use_pdl=use_pdl,
        gate=g,
        norm_eps=1e-6,
        ng_vec=int(gate) if gate else 8,
    )
    torch.cuda.current_stream(device).synchronize()
    if SKEL:
        # skeleton options: the load-time bitwise self-check against the default build (outside any graph capture)
        _skel_selfcheck(
            torch.device(device.type, key[1]), num_q_heads, num_kv_heads, use_pdl, gate
        )
        torch.cuda.current_stream(device).synchronize()
    _preloaded.add(key)


_preloaded: set = set()
