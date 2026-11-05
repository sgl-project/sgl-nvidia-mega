# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Paged FP8 causal prefill attention on tensor-core MMA (tcgen05), JIT CUDA.

See ``csrc/prefill_attn.cuh`` for the arithmetic, the work split and the synchronization. The op covers prefill
(context) forwards of attention layers with 8 query heads per KV head, head_dim 64, e4m3 Q/K/V, page size 128,
bf16 output and full causal attention. It is not wired into any serving path (no component uses it).
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("load_jit", "make_cpp_args"),
}


_CSRC = Path(__file__).resolve().parent.parent / "csrc" / "prefill_attn.cuh"

HEAD_DIM = 64
Q_PER_KV = 8
PAGE = 128
TOKENS_PER_BLOCK = 32


# polynomial exp2 (FMA pipe) for this many of every 8 key pairs; the rest use the hardware ex2 (both fp32-accurate)
POLY_PAIRS = 0
# exp modes beyond the polynomial counts (see csrc/prefill_attn.cuh): half exp with half partial row sums; half exp with
# the row sum on the tensor core
HALF_EXP = 16
HALF_EXP_TC = 17


@functools.lru_cache(maxsize=None)
def _module(use_pdl: bool, poly_pairs: int):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(use_pdl, poly_pairs)
    return load_jit(
        "muse_vr_prefill_attn",
        *args,
        cuda_files=[str(_CSRC)],
        cuda_wrappers=[("prefill_attn", f"muse_vr::PrefillAttnKernel<{args}>::run")],
    )


def supported(
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    *,
    window_left: int = -1,
    causal: bool = True,
    has_sinks: bool = False,
    q_dtype=None,
    kv_dtype=None,
    out_dtype=None,
) -> bool:
    """Shapes and options the kernel computes exactly: full causal attention (no sliding window, no attention sinks),
    8 query heads per KV head, head_dim 64, page size 128, e4m3 Q / K / V, bf16 output."""
    e4m3 = torch.float8_e4m3fn
    return (
        head_dim == HEAD_DIM
        and num_kv_heads > 0
        and num_kv_heads * Q_PER_KV == num_q_heads
        and page_size == PAGE
        and window_left == -1
        and causal
        and not has_sinks
        and (q_dtype is None or q_dtype == e4m3)
        and (kv_dtype is None or kv_dtype == e4m3)
        and (out_dtype is None or out_dtype == torch.bfloat16)
    )


def plan_units(
    seq_lens_cpu,
    cu_q_cpu,
    num_kv_heads: int,
    num_sms: int,
    chunk_tiles: int | None = None,
    per_request: bool = False,
    include=None,
):
    """Work units for one call (host side, from CPU lengths).

    Each 32-token query block of a request and KV head needs the key tiles [0, ceil(key_end / 128)). Blocks longer than
    the chunk size C = max(64, ceil(total tiles / (4 * num_sms))) (or ``chunk_tiles``) are split into near-equal chunks
    of whole tiles; their partial results go to workspace slots and are merged. Units are ordered by KV head, largest
    first within a head.

    ``per_request``: each request's chunk size comes from its own tiles only (the plan it would get alone in a call),
    so its split units, and with them its output bits, do not depend on the other requests of the call.
    ``include``: per-request flags; requests flagged False get no units (their output rows are not written).

    :returns: (units [n, 8] int32, merge units [m, 8] int32, number of slots) as CPU tensors.
    """
    blocks = []
    for r in range(len(seq_lens_cpu)):
        if include is not None and not include[r]:
            continue
        q_len = int(cu_q_cpu[r + 1]) - int(cu_q_cpu[r])
        prefix = int(seq_lens_cpu[r]) - q_len
        for tok0 in range(0, q_len, TOKENS_PER_BLOCK):
            key_end = prefix + min(tok0 + TOKENS_PER_BLOCK, q_len)
            tiles = (key_end + PAGE - 1) // PAGE
            for h in range(num_kv_heads):
                blocks.append((r, h, tok0, tiles))
    total = sum(blk[3] for blk in blocks)
    c = chunk_tiles if chunk_tiles is not None else max(64, -(-total // (4 * num_sms)))
    c_req = {}
    if per_request and chunk_tiles is None:
        for r, _, _, tiles in blocks:
            c_req[r] = c_req.get(r, 0) + tiles
        c_req = {r: max(64, -(-t // (4 * num_sms))) for r, t in c_req.items()}
    units, merges, slot = [], [], 0
    for r, h, tok0, tiles in blocks:
        n = -(-tiles // c_req.get(r, c))
        if n <= 1:
            units.append((tiles, [r, h, tok0, 0, tiles, -1, 0, 0]))
            continue
        bounds = [(i * tiles) // n for i in range(n + 1)]
        merges.append([r, h, tok0, slot, n, 0, 0, 0])
        for i in range(n):
            units.append(
                (
                    bounds[i + 1] - bounds[i],
                    [r, h, tok0, bounds[i], bounds[i + 1], slot + i, 0, 0],
                )
            )
        slot += n
    # KV-head major (concurrent blocks read the same head's pages), largest first within a head
    units.sort(key=lambda u: (u[1][1], -u[0]))
    ut = torch.tensor([u[1] for u in units], dtype=torch.int32).reshape(-1, 8)
    mt = torch.tensor(merges, dtype=torch.int32).reshape(-1, 8)
    return ut, mt, slot


# Balanced plan (tunable; bench/bench_xs0.py): a request whose work is shorter than SK_MIN_TILES tiles per block uses
# fewer blocks.
SK_MIN_TILES = 64


def plan_sk(
    seq_lens_cpu,
    cu_q_cpu,
    num_kv_heads: int,
    num_sms: int,
    include=None,
    min_tiles: int = SK_MIN_TILES,
):
    """Balanced work split: per request, its work line (KV heads in order; per head its 32-token query blocks in order;
    per block its 128-key tiles in key order) is cut into P contiguous ranges of equal length (to one tile), P =
    min(num_sms, max(1, total tiles // min_tiles)). Block i of the launch runs range i of every request, one range after
    another in request order. A query block cut by range boundaries becomes several units (key-tile pieces) that write
    partial results, merged in key order by prefill_attn_merge. The cut points depend only on the request's own lengths
    and num_sms, so a request's output bits do not depend on the other requests of the call.

    :returns: (units [n, 8], cta_units [blocks + 1], merge units [m, 8], number of slots) as CPU int32 tensors.
    """
    per_cta = [[] for _ in range(max(1, num_sms))]
    merges, slot = [], 0
    for r in range(len(seq_lens_cpu)):
        if include is not None and not include[r]:
            continue
        q_len = int(cu_q_cpu[r + 1]) - int(cu_q_cpu[r])
        if q_len <= 0:
            continue
        prefix = int(seq_lens_cpu[r]) - q_len
        blocks = []
        for h in range(num_kv_heads):
            for tok0 in range(0, q_len, TOKENS_PER_BLOCK):
                blocks.append(
                    (
                        h,
                        tok0,
                        (prefix + min(tok0 + TOKENS_PER_BLOCK, q_len) + PAGE - 1)
                        // PAGE,
                    )
                )
        total = sum(blk[2] for blk in blocks)
        nblk = min(len(per_cta), max(1, total // max(1, min_tiles)))
        bounds = [(i * total) // nblk for i in range(nblk + 1)]
        pos, i = 0, 0
        for h, tok0, tiles in blocks:
            start, end = pos, pos + tiles
            while bounds[i + 1] <= start:
                i += 1
            pieces = []
            k = i
            while k < nblk and bounds[k] < end:
                a, b = max(start, bounds[k]), min(end, bounds[k + 1])
                if b > a:
                    pieces.append((k, a - start, b - start))
                k += 1
            if len(pieces) == 1:
                per_cta[pieces[0][0]].append([r, h, tok0, 0, tiles, -1, 0, 0])
            else:
                merges.append([r, h, tok0, slot, len(pieces), 0, 0, 0])
                for n, (k, a, b) in enumerate(pieces):
                    per_cta[k].append([r, h, tok0, a, b, slot + n, 0, 0])
                slot += len(pieces)
            pos = end
    rows, offs = [], [0]
    for lst in per_cta:
        if lst:
            rows.extend(lst)
            offs.append(len(rows))
    ut = torch.tensor(rows, dtype=torch.int32).reshape(-1, 8)
    ct = torch.tensor(offs if rows else [], dtype=torch.int32)
    mt = torch.tensor(merges, dtype=torch.int32).reshape(-1, 8)
    return ut, ct, mt, slot


# Both caches are per (device, current stream): device tensors of a plan and the split workspace are only ever used in
# the order of the stream that created them (the caching allocator's stream ordering covers their reuse after eviction).
_PLAN_CACHE: dict = {}
_WS_CACHE: dict = {}


def _stream_key(device) -> tuple:
    return str(device), torch.cuda.current_stream(device).cuda_stream


def _plan(
    seq_lens_cpu,
    cu_q_cpu,
    num_kv_heads,
    device,
    chunk_tiles,
    per_request=False,
    include=None,
    plan="units",
    sk_min_tiles=None,
):
    inc = None if include is None else tuple(bool(v) for v in include)
    key = (
        tuple(int(v) for v in seq_lens_cpu),
        tuple(int(v) for v in cu_q_cpu),
        num_kv_heads,
        _stream_key(device),
        chunk_tiles,
        bool(per_request),
        inc,
        plan,
        sk_min_tiles,
    )
    hit = _PLAN_CACHE.get(key)
    if hit is None:
        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        if plan == "sk":
            ut, ct, mt, slots = plan_sk(
                seq_lens_cpu,
                cu_q_cpu,
                num_kv_heads,
                num_sms,
                inc,
                SK_MIN_TILES if sk_min_tiles is None else int(sk_min_tiles),
            )
        elif plan == "units":
            ut, mt, slots = plan_units(
                seq_lens_cpu,
                cu_q_cpu,
                num_kv_heads,
                num_sms,
                chunk_tiles,
                bool(per_request),
                inc,
            )
            ct = torch.empty(0, dtype=torch.int32)
        else:
            raise ValueError(f"plan {plan!r} not units / sk")
        hit = (ut.to(device), mt.to(device), slots, ct.to(device))
        if len(_PLAN_CACHE) > 64:
            _PLAN_CACHE.clear()
        _PLAN_CACHE[key] = hit
    return hit


def _workspace(device, slots):
    rows = max(slots, 1) * 2 * 128
    key = _stream_key(device)
    ws = _WS_CACHE.get(key)
    if ws is None or ws[0].numel() < rows * HEAD_DIM:
        ws = (
            torch.empty(rows * HEAD_DIM, dtype=torch.float32, device=device),
            torch.empty(rows * 2, dtype=torch.float32, device=device),
        )
        _WS_CACHE[key] = ws
    return ws


def prefill_attn(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_q_len: int,
    bmm1_scale: float,
    bmm2_scale: float,
    out: torch.Tensor | None = None,
    use_pdl: bool = False,
    trace: torch.Tensor | None = None,
    poly_pairs: int | None = None,
    seq_lens_cpu=None,
    cu_seqlens_q_cpu=None,
    chunk_tiles: int | None = None,
    per_request: bool = False,
    include=None,
    plan: str = "units",
    sk_min_tiles: int | None = None,
) -> torch.Tensor:
    """Causal attention of each request's new tokens over its paged FP8 KV range.

    :param q: ``[tokens, Hq, 64]`` e4m3, heads and dims contiguous; tokens of request r are
        ``cu_seqlens_q[r] .. cu_seqlens_q[r + 1] - 1`` (the last ``q_len`` positions of its KV range).
    :param k_cache: ``[pages, Hkv, 128, 64]`` e4m3 view, dims contiguous (any page / head / token strides).
    :param v_cache: same layout as ``k_cache``.
    :param page_table: ``[>= R, pages]`` int32, inner dimension contiguous.
    :param seq_lens: ``[R]`` int32 KV length per request including its new tokens.
    :param cu_seqlens_q: ``[R + 1]`` int32.
    :param max_q_len: upper bound of the per-request new-token counts.
    :param seq_lens_cpu: / :param cu_seqlens_q_cpu: the same lengths on the host (plans the work units; read from the
        device tensors, with a synchronization, when absent).
    :param chunk_tiles: override of the chunk size in 128-key tiles (tests).
    :param per_request: plan each request's split units from its own lengths only (see ``plan_units``).
    :param include: per-request flags; only flagged requests are computed (other output rows are left as they are).
    :param plan: ``"units"`` (``plan_units``: one block per work unit) or ``"sk"`` (``plan_sk``: balanced ranges, each
        block runs its list of units in order; ``chunk_tiles`` / ``per_request`` do not apply).
    :param sk_min_tiles: override of ``SK_MIN_TILES`` for ``plan="sk"`` (tests / bench).
    """
    if out is None:
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    if seq_lens_cpu is None:
        seq_lens_cpu = seq_lens.tolist()
    if cu_seqlens_q_cpu is None:
        cu_seqlens_q_cpu = cu_seqlens_q.tolist()
    units, munits, slots, cta_units = _plan(
        seq_lens_cpu,
        cu_seqlens_q_cpu,
        k_cache.shape[1],
        q.device,
        chunk_tiles,
        per_request,
        include,
        plan,
        sk_min_tiles,
    )
    if include is not None and units.shape[0] == 0:
        return out
    ws_o, ws_ml = _workspace(q.device, slots)
    pp = POLY_PAIRS if poly_pairs is None else int(poly_pairs)
    _module(bool(use_pdl), pp).prefill_attn(
        units,
        cta_units,
        munits,
        ws_o,
        ws_ml,
        q,
        k_cache,
        v_cache,
        page_table,
        seq_lens,
        cu_seqlens_q,
        out,
        int(max_q_len),
        float(bmm1_scale),
        float(bmm2_scale),
        trace.data_ptr() if trace is not None else 0,
    )
    return out


def preload(device, use_pdl: bool = False, poly_pairs: int | None = None) -> None:
    """Build and load the kernel (no build or module load inside a later call)."""
    q = torch.zeros(1, Q_PER_KV, HEAD_DIM, dtype=torch.float8_e4m3fn, device=device)
    cache = torch.zeros(1, 1, PAGE, HEAD_DIM, dtype=torch.float8_e4m3fn, device=device)
    table = torch.zeros(1, 1, dtype=torch.int32, device=device)
    lens = torch.ones(1, dtype=torch.int32, device=device)
    cu = torch.tensor([0, 1], dtype=torch.int32, device=device)
    prefill_attn(
        q,
        cache,
        cache,
        table,
        lens,
        cu,
        1,
        0.125,
        1.0,
        use_pdl=use_pdl,
        poly_pairs=poly_pairs,
    )
    torch.cuda.current_stream(device).synchronize()
