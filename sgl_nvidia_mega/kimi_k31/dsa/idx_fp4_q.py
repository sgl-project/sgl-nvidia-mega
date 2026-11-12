# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The fp4 setting of the indexer-query toggle (``K31MK_DSA_IDX_Q=fp4`` with ``K31MK_DSA_IDX_FP4=1``): the MXFP4 index
query FUSED into the kernel that produces it, on every verify path that produces one, against the MXFP4 index-K pages.
Label: "MXFP4 indexer q" (an MXFP4 query against the MXFP4 key cache, the benchmark precision list's "indexer cache FP4").

Producers (each replaces the FP8 quantizer inside the same kernel; no re-quantize pass):
- 1-8 rows (fused index query) and the query front (K31MK_DSA_QF, 1-16 rows): index_q.cuh's kQ4 instances (run_q4 / run_copy_q4 /
  run_qf_q4), the FP8 kernel's BF16 query with the MXFP4 quantizer in its epilogue (Fp4FusedIndexQuery);
- QFN rows (K31MK_DSA_QFN: the norm-folded wq_b GEMM) and QFP profiles: qfn_quantize_query, index_qfn.py's rstd step with the MXFP4
  quantizer in place of the FP8 one: idx_fp4_qfn.cuh (one CTA per row, PDL) under K31MK_DSA_IDX_FP4_QFN=cuda (default), else the Triton
  program per row below (the reference);
- the stock chain (rmsnorm, wq_b GEMM): quantize_query_served = idx_fp4.quantize_query's bytes (it replaces indexer_q_quant_head_gate);
  the CUDA kernel idx_fp4_qfn.cuh<kRstd=false> under K31MK_DSA_IDX_FP4_QUANT=cuda (default), else the Triton idx_fp4.quantize_query.
Quantizer = idx_fp4.quantize_query's (sglang fp4_indexer_quant row math, RNE off): per 32-group ue8m0 exponent ceil(max(amax / 6, 1e-4)),
e2m1 threshold codes of q / 2^(e - 127), low nibble = even element; head weights float(bf16(gate * 32^-0.5)) * 128^-0.5 (no q scale).
Engagement: 'K31MK_DSA_IDX_FP4 ENGAGED site=query_fused rows=N' / 'site=query_qf rows=N' / 'site=query_qfn rows=N' at the first call
per site (eager, before capture); STATS counts calls.
"""

from __future__ import annotations

import logging
import os
from functools import cache
from pathlib import Path

import torch
import triton
import triton.language as tl

# SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/ops/quantization/fp4_indexer_quant.py": ("_ceil_ue8m0_exp", "_fp4_e2m1_code"),
    "kernels/jit/utils/compile/loader.py": ("load_jit",),
}

from sglang.kernels.ops.quantization.fp4_indexer_quant import _ceil_ue8m0_exp, _fp4_e2m1_code  # noqa: E402

from . import idx_fp4  # noqa: E402
from .idx_fp4 import HEAD_DIM, HEADS, Fp4Query  # noqa: E402

logger = logging.getLogger(__name__)
STATS = {"query_fused": 0, "query_qf": 0, "query_qfn": 0}
# ``K31MK_DSA_IDX_FP4_QFN`` (default cuda): the QFN / QFP rows' MXFP4 query kernel - cuda = idx_fp4_qfn.cuh, triton = _qfn_q4_kernel below.
QFN_IMPL = os.environ.get("K31MK_DSA_IDX_FP4_QFN", "cuda")
if QFN_IMPL not in ("cuda", "triton"):
    raise ValueError(f"K31MK_DSA_IDX_FP4_QFN must be cuda or triton, got {QFN_IMPL!r}")
# ``K31MK_DSA_IDX_FP4_QUANT`` (default cuda): the stock chain's MXFP4 query kernel - cuda = idx_fp4_qfn.cuh without its rstd step
# (bit-identical to idx_fp4.quantize_query), triton = idx_fp4.quantize_query (the reference; tests use it as such).
QUANT_IMPL = os.environ.get("K31MK_DSA_IDX_FP4_QUANT", "cuda")
if QUANT_IMPL not in ("cuda", "triton"):
    raise ValueError(f"K31MK_DSA_IDX_FP4_QUANT must be cuda or triton, got {QUANT_IMPL!r}")
_LOGGED = set()


def _engaged(site: str, rows: int) -> None:
    STATS[site] += 1
    if site not in _LOGGED:
        _LOGGED.add(site)
        logger.info("K31MK_DSA_IDX_FP4 ENGAGED site=%s rows=%d (MXFP4 index query fused into its producer; label 'MXFP4 indexer q')",
                    site, rows)


@cache
def _load():
    import sglang
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "k31_index_q_fp4",
        cuda_files=[str(Path(__file__).with_name("index_q.cuh"))],
        cuda_wrappers=[("run_q4", "k31_index_q::Binding::run_q4"), ("run_copy_q4", "k31_index_q::Binding::run_copy_q4"),
                       ("run_qf_q4", "k31_index_q::Binding::run_qf_q4")],
        extra_include_paths=[str(Path(sglang.__file__).parent / "kernels/jit/csrc")],
    )


class Fp4FusedIndexQuery:
    """index_q.FusedIndexQuery with the MXFP4 query (index_q.cuh kQ4): run / run_qf return (Fp4Query [rows, 32, 64] + [rows, 32] scales,
    head weights [rows, 32, 1] FP32), owned buffers. ``qf_only=True`` admits 9-16 rows for the query-front form (run_qf only)."""

    def __init__(self, weights, rows=1, qf_only=False):
        max_rows = 16 if qf_only else 8
        if weights.query.shape != (4096, 1536) or weights.gate.shape != (32, 7168) or not 1 <= rows <= max_rows:
            raise ValueError(f"fused MXFP4 index query: replicated 32x128 heads, 1-{max_rows} rows")
        self.weights, self.rows = weights, rows
        device = weights.query.device
        self.q4 = torch.empty((rows, HEADS, HEAD_DIM // 2), dtype=torch.int8, device=device)
        self.sf = torch.empty((rows, HEADS), dtype=torch.int32, device=device)
        self.gates = torch.empty((rows, HEADS, 1), dtype=torch.float32, device=device)
        self.query = Fp4Query(self.q4, self.sf)
        self.module = _load()

    def run(self, hidden, projected, mla_query=None, padded_query=None):
        w = self.weights
        args = (projected, hidden, w.query_norm, w.query, w.gate, self.q4, self.sf, self.gates.view(-1, HEADS), float(w.epsilon))
        if mla_query is None:
            self.module.run_q4(*args)
        else:
            self.module.run_copy_q4(*args, mla_query.view(torch.uint8).reshape(mla_query.shape[0], -1),
                                    padded_query.view(torch.uint8).reshape(padded_query.shape[0], -1))
        _engaged("query_fused", self.rows)
        return self.query, self.gates

    def run_qf(self, projected, key_col, gate_col, slots, cache, key_norm):
        """Query-front form: as FusedIndexQuery.run_qf, into the MXFP4 index-K pages [pages, 64 * 68] (the binding refuses FP8 pages)."""
        w = self.weights
        idx_fp4.check_cache(cache)
        gamma, beta, eps = key_norm
        self.module.run_qf_q4(projected, w.query_norm, w.query, self.q4, self.sf, self.gates.view(-1, HEADS), float(w.epsilon),
                              int(key_col), int(gate_col), slots, cache, gamma, beta, float(eps))
        _engaged("query_qf", self.rows)
        return self.query, self.gates


# ------------------------------------------------------------------------------------------------------------------------------
# Logits over MXFP4 index-K pages for either toggle setting (TsiState.scan's local pages; VerifySelection.run makes the same choice).
def _q8_vendor() -> bool:
    import os

    return os.environ.get("K31MK_DSA_IDX_FP4_VENDOR", "0") == "1" or os.environ.get("K31MK_DSA_DG_VENDOR", "0") == "1"


def paged_logits(query, gates, cache, contexts, table, schedule, max_context_len, batch_requests, query_width):
    """FP32 logits [B * N, max_context_len] (DeepGEMM's padded row stride; callers restore it) over the MXFP4 pages [pages, 64 * 68]:
    K31MK_DSA_IDX_Q=fp8 -> dg_logits_fp4k (FP8 query x MXFP4 index-K); fp4 -> dg_logits_fp4 (vendored PDL) at Q8 when K31MK_DSA_DG_VENDOR=1
    or K31MK_DSA_IDX_FP4_VENDOR=1, else idx_fp4.paged_logits (DeepGEMM fp8_fp4). ``query`` is the setting's (FP8 tensor / Fp4Query)."""
    from .idx_fp4_knob import FP8Q

    if FP8Q:
        from .dg_logits_fp4k import verify_logits as fn
    elif _q8_vendor() and int(query_width) == 8:
        from .dg_logits_fp4 import verify_logits as fn
    else:
        fn = idx_fp4.paged_logits
    return fn(query, gates, cache, contexts, table, schedule, max_context_len, batch_requests, query_width)


def build_logits(device_index: int, query_width: int) -> None:
    """JIT-build the kernel paged_logits launches at ``query_width`` (prepare time, never inside a graph capture)."""
    from .idx_fp4_knob import FP8Q

    if FP8Q and query_width != 16:
        from .dg_logits_fp4k import build

        build(device_index, query_width)
    elif not FP8Q and _q8_vendor() and query_width == 8:
        from .dg_logits_fp4 import build

        build(device_index, 8)


# ------------------------------------------------------------------------------------------------------------------------------
# QFN rows: index_qfn._qfn_quant_head_gate_kernel's rstd step (same row load, block and warps), then idx_fp4's MXFP4 quantizer.
_QFN_BLOCK_X, _QFN_WARPS = 2048, 8  # index_qfn's launch (BLOCK_X 2048, num_warps 8): the same rstd reduction


@triton.jit
def _qfn_q4_kernel(ACC, X, G, Q4, QSF, W, QDBG, X_STRIDE, GATE_STRIDE, EPS, LORA: tl.constexpr, HEADS_: tl.constexpr,
                   GATE_SCALE: tl.constexpr, SOFTMAX_SCALE: tl.constexpr, BLOCK_X: tl.constexpr, STORE_Q: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    xo = tl.arange(0, BLOCK_X)
    x = tl.load(X + row * X_STRIDE + xo, xo < LORA, other=0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, 0) / LORA + EPS)
    grp = tl.arange(0, HEADS_ * 4)  # (head, 32-group) rows of the [HEADS * 128] query row
    offs = row * (HEADS_ * 128) + grp[:, None] * 32 + tl.arange(0, 32)[None, :]
    q = (tl.load(ACC + offs).to(tl.float32) * rstd).to(tl.bfloat16).to(tl.float32)
    if STORE_Q:  # test hook (constexpr; the served launch passes False): the BF16 query this program quantizes
        tl.store(QDBG + offs, q.to(tl.bfloat16))
    exp = _ceil_ue8m0_exp(tl.maximum(tl.max(tl.abs(q), axis=1) / 6.0, 1.0e-4))
    scale = (exp << 23).to(tl.float32, bitcast=True)
    v0, v1 = tl.split(tl.reshape(q / scale[:, None], (HEADS_ * 4, 16, 2)))
    packed = (_fp4_e2m1_code(v0) & 0x0F) | ((_fp4_e2m1_code(v1) & 0x0F) << 4)
    tl.store(Q4 + row * (HEADS_ * 64) + grp[:, None] * 16 + tl.arange(0, 16)[None, :], packed)
    shifts = tl.arange(0, 4) * 8  # the four exponents occupy disjoint bytes: an integer sum packs them
    packed_sf = tl.sum(tl.reshape(exp.to(tl.uint32), (HEADS_, 4)) << shifts[None, :], axis=1)
    heads = tl.arange(0, HEADS_)
    tl.store(QSF + row * HEADS_ + heads, packed_sf.to(tl.int32))
    gate = tl.load(G + row * GATE_STRIDE + heads).to(tl.float32)
    gate = (gate * GATE_SCALE).to(tl.bfloat16).to(tl.float32)
    tl.store(W + row * HEADS_ + heads, gate * SOFTMAX_SCALE)


def build_qfn() -> None:
    """JIT-build the CUDA query module when K31MK_DSA_IDX_FP4_QFN or _QUANT selects it (prepare time, never inside a graph capture)."""
    if QFN_IMPL == "cuda" or QUANT_IMPL == "cuda":
        from .idx_fp4_qfn import build

        build()


def quantize_query_served(query, gate, out, impl: str = ""):
    """The fp4 setting's stock-chain query (verify.py Fp4IndexProjection without a folded query): idx_fp4.quantize_query's contract and
    bytes; the CUDA kernel under K31MK_DSA_IDX_FP4_QUANT=cuda (default), else idx_fp4.quantize_query. ``impl`` (tests) overrides."""
    if (impl or QUANT_IMPL) != "cuda":
        return idx_fp4.quantize_query(query, gate, out)
    if query.dtype != torch.bfloat16 or query.ndim != 3 or tuple(query.shape[1:]) != (HEADS, HEAD_DIM) or not query.is_contiguous():
        raise ValueError(f"{idx_fp4.ENV} query: BF16 contiguous [T, 32, 128], got {query.dtype} {tuple(query.shape)}")
    t = query.shape[0]
    if tuple(gate.shape) != (t, HEADS) or gate.stride(1) != 1 or gate.dtype != torch.bfloat16:
        raise ValueError(f"{idx_fp4.ENV} query: gate BF16 [T, 32] with unit column stride, got {gate.dtype} {tuple(gate.shape)}")
    if out.q4.shape[0] < t:
        raise ValueError(f"{idx_fp4.ENV} query: buffers hold {out.q4.shape[0]} rows, {t} given")
    q4, sf, w = out.q4[:t], out.sf[:t], out.weights[:t]
    if t:
        from .idx_fp4_qfn import quantize

        quantize(query, gate, q4, sf, w)
        idx_fp4._engaged("query")
    return Fp4Query(q4, sf), w


def qfn_quantize_query(acc, projected, gate, eps, out, softmax_scale: float = HEAD_DIM ** -0.5, debug_query=None, impl: str = ""):
    """(Fp4Query, head weights [rows, 32, 1] FP32) of the QFN rows: ``acc`` = bf16(x @ wq_bg^T) [rows, 32, 128] contiguous, the raw q_a in
    ``projected[:, :1536]`` (row stride free), the BF16 head gate [rows, 32] (row stride free), ``out`` an idx_fp4.QueryBuffers.
    ``debug_query`` (tests only): a BF16 [rows, 32, 128] buffer that receives the query the kernel quantized. ``impl`` (tests): 'cuda' /
    'triton' overrides K31MK_DSA_IDX_FP4_QFN (the CUDA kernel serves the default softmax scale only)."""
    if acc.dtype != torch.bfloat16 or not acc.is_contiguous() or tuple(acc.shape[1:]) != (HEADS, HEAD_DIM):
        raise ValueError(f"{idx_fp4.ENV} QFN query: acc BF16 contiguous [rows, 32, 128], got {acc.dtype} {tuple(acc.shape)}")
    rows = acc.shape[0]
    if projected.dtype != torch.bfloat16 or projected.shape[0] != rows or projected.shape[1] < 1536 or projected.stride(1) != 1:
        raise ValueError(f"{idx_fp4.ENV} QFN query: projected BF16 [rows, >= 1536] with unit column stride")
    if tuple(gate.shape) != (rows, HEADS) or gate.stride(1) != 1 or gate.dtype != torch.bfloat16:
        raise ValueError(f"{idx_fp4.ENV} QFN query: gate BF16 [rows, 32] with unit column stride, got {gate.dtype} {tuple(gate.shape)}")
    if out.q4.shape[0] < rows:
        raise ValueError(f"{idx_fp4.ENV} QFN query: buffers hold {out.q4.shape[0]} rows, {rows} given")
    q4, sf, w = out.q4[:rows], out.sf[:rows], out.weights[:rows]
    if rows and (impl or QFN_IMPL) == "cuda" and softmax_scale == HEAD_DIM ** -0.5:
        from .idx_fp4_qfn import qfn

        qfn(acc, projected, gate, q4, sf, w, eps, debug_query)
        _engaged("query_qfn", rows)
    elif rows:
        store = debug_query is not None
        _qfn_q4_kernel[(rows,)](acc, projected, gate, q4, sf, w, debug_query if store else acc, projected.stride(0), gate.stride(0),
                                float(eps), 1536, HEADS, HEADS ** -0.5, softmax_scale, _QFN_BLOCK_X, store, num_warps=_QFN_WARPS)
        _engaged("query_qfn", rows)
    return Fp4Query(q4, sf), w
