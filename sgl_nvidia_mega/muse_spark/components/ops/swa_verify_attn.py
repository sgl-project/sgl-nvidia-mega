# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Sliding-window FP8 paged attention for speculative verify (8 query tokens per request), JIT CUDA.

See ``csrc/swa_verify_attn.cuh`` for the arithmetic and the work split. The op covers the shape class of the
verify step of sliding-window layers: 8 query heads per KV head, head_dim 64, e4m3 Q/K/V, bf16 output, a window
of at most 16 * chunk keys.

With ``qkv=`` the kernel also does the per-head query/key RMS norm, RoPE and the FP8 KV-cache store of the new
tokens (the work of SGLang's fused QK-norm kernel, same arithmetic for the same row count; see ``qk_class``).
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": (
        "load_jit",
        "make_cpp_args",
        "is_arch_support_pdl",
    ),
}


_CSRC = Path(__file__).resolve().parent.parent / "csrc" / "swa_verify_attn.cuh"

TOKENS_PER_REQUEST = 8
HEAD_DIM = 64
Q_PER_KV = 8
MAX_CLUSTER = 16

# (max requests, (keys per block, warps per row group, row groups per block)) and the request counts launched
# without programmatic dependent launch, for the kernel on e4m3 queries and for the query/key prologue variant.
# In a served model's verify step (torch profiler, qkvg GEMM start to o_proj start per sliding-window layer, 4 ranks,
# 64K context): the kernel on e4m3 queries measured faster with programmatic dependent launch at 1-4 requests (larger
# counts follow the same choice); the query/key prologue variant measured faster without it at 1-3 requests.
# bench/bench_attn_slice.py, a chain of the same kernels outside the server, disagreed for the first and is not used
# for these choices.
# Tunable (bench/bench_swa_verify_attn.py --chain, bench/bench_attn_slice.py);
# MUSE_VR_SWA_VERIFY_ATTN_CFG="chunk,split,rg" forces one config for every size;
# MUSE_VR_SWA_VERIFY_ATTN_NO_PDL / MUSE_VR_SWA_VERIFY_ATTN_QK_NO_PDL (lists of request counts separated by "," or "/")
# override the sets.
_POLICY = ((1 << 30, (192, 4, 1)),)
_NO_PDL_REQUESTS = frozenset()
_NO_PDL_REQUESTS_QK = frozenset({1, 2, 3, 4})


def _env_set(name: str, default: frozenset) -> frozenset:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return frozenset(int(v) for v in raw.replace("/", ",").split(",") if v.strip())


def cfg(num_requests: int) -> tuple[int, int, int]:
    raw = os.environ.get("MUSE_VR_SWA_VERIFY_ATTN_CFG", "")
    if raw:
        c, k, r = (int(v) for v in raw.split(","))
        return c, k, r
    for bound, choice in _POLICY:
        if num_requests <= bound:
            return choice
    return _POLICY[-1][1]


def cluster_size(window_left: int, kchunk: int) -> int:
    return (window_left + TOKENS_PER_REQUEST + kchunk - 1) // kchunk


# Arithmetic form of SGLang's fused QK-norm Triton kernel per (Triton release, FP-fusion default) the prologues were
# validated against bit for bit (tests/test_swa_qk_prologue.py, tests/test_full_qk_prologue.py): "fused" = with FP
# fusion (fma contractions, the mean as div.full), "unfused" = without (every product and sum rounded separately, the
# mean as a product with div.approx(1, 64)). The kernel is launched without an explicit enable_fp_fusion, so the
# release's default (triton.knobs.language) decides; any other combination keeps the stock call.
QK_ARITH = {("3.8.0", True): "fused", ("3.9.0", False): "unfused"}


@functools.lru_cache(maxsize=None)
def qk_arith() -> str | None:
    """The prologue arithmetic form for the running Triton, or None when it is not validated."""
    try:
        import triton
        from triton import knobs

        lang = knobs.language
        fusion = bool(getattr(lang, "default_fp_fusion", True)) and not bool(
            getattr(lang, "force_disable_fp_fusion", False)
        )
        return QK_ARITH.get((str(triton.__version__), fusion))
    except Exception:  # noqa: BLE001
        return None


def qk_unfused(qk: int) -> bool:
    return bool(qk) and qk_arith() == "unfused"


def qk_class(num_tokens: int, num_q_heads: int, num_kv_heads: int) -> int | None:
    """Columns per thread of the query/key prologue for this many tokens: the column count per thread of SGLang's
    fused QK-norm Triton kernel, whose block size follows the row count (``fused_qk_norm._triton_config``:
    4 / 8 / 16 rows per block below 1024 / 8192 / 65536 rows, 4 warps). None: not covered."""
    n_rows = num_tokens * (num_q_heads + 2 * num_kv_heads)
    if n_rows < 1024:
        return 2
    if n_rows < 8192:
        return 4
    if n_rows < 65536:
        return 8
    return None


@functools.lru_cache(maxsize=None)
def _module(
    kchunk: int,
    ksplit: int,
    krg: int,
    kc: int,
    use_pdl: bool,
    prefetch: bool,
    norm_gate: bool,
    qk: int = 0,
    rope: bool = False,
    unfused: bool = False,
):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(
        kchunk, ksplit, krg, kc, use_pdl, prefetch, norm_gate, qk, rope, unfused
    )
    return load_jit(
        "muse_vr_swa_verify_attn",
        *args,
        cuda_files=[str(_CSRC)],
        cuda_wrappers=[
            ("swa_verify_attn", f"muse_vr::SwaVerifyAttnKernel<{args}>::run")
        ],
    )


def pdl_for(num_requests: int, qk: int = 0) -> bool:
    if qk:
        no_pdl = _env_set("MUSE_VR_SWA_VERIFY_ATTN_QK_NO_PDL", _NO_PDL_REQUESTS_QK)
    else:
        no_pdl = _env_set("MUSE_VR_SWA_VERIFY_ATTN_NO_PDL", _NO_PDL_REQUESTS)
    return _use_pdl() and num_requests not in no_pdl


def module_key(
    num_requests: int,
    window_left: int,
    norm_gate: bool = False,
    use_pdl: bool | None = None,
    prefetch: bool = False,
    qk: int = 0,
    rope: bool = False,
) -> tuple:
    c, k, r = cfg(num_requests)
    pdl = pdl_for(num_requests, qk) if use_pdl is None else use_pdl
    return (
        c,
        k,
        r,
        cluster_size(int(window_left), c),
        pdl,
        bool(prefetch),
        bool(norm_gate),
        int(qk),
        bool(rope) if qk else False,
        qk_unfused(qk),
    )


def is_loaded(key: tuple) -> bool:
    """Whether the JIT module for ``key`` (from ``module_key``) is built and loaded in this process."""
    return key in _loaded


def preload(
    window_left: int,
    max_requests: int,
    num_q_heads: int,
    num_kv_heads: int,
    device,
    norm_gate: bool = False,
    fuse_qk: bool = False,
    rope: bool = False,
) -> None:
    """Build, load and launch once (on zeros) the kernels every request count up to ``max_requests`` uses, so that
    no build, module load or function attribute call happens inside a CUDA-graph capture."""
    keys = set()
    for n in range(1, max_requests + 1):
        qk = (
            qk_class(n * TOKENS_PER_REQUEST, num_q_heads, num_kv_heads)
            if fuse_qk
            else 0
        )
        keys.add(module_key(n, window_left, norm_gate, qk=qk or 0, rope=rope))
    if keys <= _loaded:
        return
    page = 64
    q = torch.zeros(
        TOKENS_PER_REQUEST,
        num_q_heads,
        HEAD_DIM,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    pool = torch.zeros(
        2 * page, num_kv_heads * HEAD_DIM, dtype=torch.float8_e4m3fn, device=device
    )
    table = torch.zeros(1, 2, dtype=torch.int32, device=device)
    lens = torch.full((1,), TOKENS_PER_REQUEST, dtype=torch.int32, device=device)
    out = torch.empty(q.shape, dtype=torch.bfloat16, device=device)
    gate = torch.zeros(
        TOKENS_PER_REQUEST, num_q_heads * HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    qkv = torch.zeros(
        TOKENS_PER_REQUEST,
        (2 * num_q_heads + 2 * num_kv_heads) * HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    pos = torch.zeros(TOKENS_PER_REQUEST, dtype=torch.int64, device=device)
    cos_sin = torch.zeros(1, HEAD_DIM, dtype=torch.float32, device=device)
    loc = torch.zeros(
        TOKENS_PER_REQUEST, dtype=torch.int64, device=device
    )  # 0: no store
    for key in keys - _loaded:
        _module(*key).swa_verify_attn(
            q,
            pool,
            pool,
            table,
            lens,
            out,
            int(window_left),
            page,
            1.0,
            1.0,
            gate,
            1e-6,
            qkv,
            pos,
            cos_sin,
            loc,
            1e-6,
        )
        _loaded.add(key)
    torch.cuda.current_stream(device).synchronize()


_loaded: set = set()


@functools.lru_cache(maxsize=None)
def _use_pdl() -> bool:
    from sglang.kernels.jit.utils import is_arch_support_pdl

    return bool(is_arch_support_pdl())


def supported(
    num_q_heads: int, num_kv_heads: int, head_dim: int, window_left: int
) -> bool:
    if not (
        head_dim == HEAD_DIM
        and num_kv_heads * Q_PER_KV == num_q_heads
        and window_left >= 0
    ):
        return False
    # at least 2 blocks per cluster: the kernel exchanges partial results with st.async, which the PTX ISA defines
    # only for clusters of more than one block
    return all(
        2 <= cluster_size(window_left, c[0]) <= MAX_CLUSTER for _, c in _POLICY
    ) and (
        not os.environ.get("MUSE_VR_SWA_VERIFY_ATTN_CFG")
        or 2 <= cluster_size(window_left, cfg(1)[0]) <= MAX_CLUSTER
    )


def swa_verify_attn(
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
    config: tuple[int, int, int] | None = None,
    use_pdl: bool | None = None,
    prefetch: bool = False,
    gate: torch.Tensor | None = None,
    norm_eps: float = 0.0,
    qkv: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    cos_sin: torch.Tensor | None = None,
    cache_loc: torch.Tensor | None = None,
    qk_eps: float = 0.0,
    num_q_heads: int | None = None,
) -> torch.Tensor:
    """Attention of 8 consecutive query tokens per request over a sliding window of a paged FP8 KV pool.

    :param q: ``[R * 8, Hq, 64]`` e4m3, heads and dims contiguous; rows of request r are ``8r .. 8r + 7`` in
        position order.
    :param k_pool: ``[slots, Hkv * 64]`` e4m3 (token-major pool; slot = page * page_size + offset).
    :param v_pool: same layout as ``k_pool``.
    :param page_table: ``[>= R, pages]`` int32, inner dimension contiguous.
    :param seq_lens: ``[R]`` int32, KV length per request including the 8 query tokens.
    :param bmm1_scale: softmax scale applied to q.k (descales included).
    :param bmm2_scale: output scale (V descale).
    :param gate: optional ``[R * 8, >= Hq * 64]`` bf16 (inner dimension contiguous): when given, the output is
        ``rms_norm(attention, eps=norm_eps) * sigmoid(gate)`` per (token, head) row, computed as SGLang's
        norm_gate kernel computes it (see the kernel source).
    :param qkv: optional ``[R * 8, >= (Hq + 2 Hkv) * 64]`` bf16 QKV projection output (q heads, k heads, v heads;
        inner dimension contiguous). When given, ``q`` is ignored (may be None; pass ``num_q_heads``): the kernel
        computes the e4m3 queries and the e4m3 K/V rows of the 8 new tokens per request (RMS norm with ``qk_eps``,
        RoPE when ``positions`` / ``cos_sin`` are given, as SGLang's fused QK-norm kernel does), uses them, and stores
        the K/V rows into ``k_pool`` / ``v_pool`` at ``cache_loc`` (int64; rows with location 0 are not stored).
    """
    if qkv is not None:
        m = qkv.shape[0]
        hq = int(num_q_heads if num_q_heads is not None else q.shape[1])
        hkv = k_pool.shape[1] // HEAD_DIM
        qk = qk_class(m, hq, hkv)
        if qk is None:
            raise ValueError("query/key prologue: row count not covered")
        rope = positions is not None
    else:
        m, hq, qk, rope = q.shape[0], q.shape[1], 0, False
    if out is None:
        out = torch.empty(m, hq, HEAD_DIM, dtype=torch.bfloat16, device=k_pool.device)
    reqs = m // TOKENS_PER_REQUEST
    c, k, r = config or cfg(reqs)
    kc = cluster_size(int(window_left), c)
    pdl = pdl_for(reqs, qk) if use_pdl is None else use_pdl
    key = (c, k, r, kc, pdl, bool(prefetch), gate is not None, qk, rope, qk_unfused(qk))
    mod = _module(*key)
    _loaded.add(key)
    dummy = out.view(out.shape[0], -1)
    mod.swa_verify_attn(
        q if q is not None else out,
        k_pool,
        v_pool,
        page_table,
        seq_lens,
        out,
        int(window_left),
        int(page_size),
        float(bmm1_scale),
        float(bmm2_scale),
        gate if gate is not None else dummy,
        float(norm_eps),
        qkv if qkv is not None else dummy,
        positions if positions is not None else seq_lens,
        cos_sin if cos_sin is not None else dummy,
        cache_loc if cache_loc is not None else seq_lens,
        float(qk_eps),
    )
    return out
