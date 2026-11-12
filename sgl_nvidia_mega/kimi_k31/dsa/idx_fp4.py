# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MXFP4 index-K + MXFP4 index query on the K3.1 verify path (``K31MK_DSA_IDX_FP4=1``; default 0 = off = the FP8 index-K pages of
64 x 132 bytes and every kernel of today's path, unchanged).

Label: "MXFP4 indexer (k + q)" (the FP4 key cache is the benchmark precision list's "indexer cache FP4"; an MXFP4 query is allowed
against MXFP4 keys). ON:
- index-K pages hold 64 tokens as [64 x 64 B e2m1 (low nibble = even element)][64 x 4 B ue8m0 (byte g = 32-group g)] = 4352 bytes,
  DeepGEMM's fused MXFP4 paged layout and the layout the host writes under K31HT_PREC_IDX_FP4=1 (index_key_cache 68 B rows);
  the verify writer stores MXFP4 keys with the host prefill writer's row math (fp4_indexer_quant.quantize_fp4_indexer_row, threshold
  rounding), so decode-written and prefill-written rows are the same bytes for the same key;
- the index query is quantized per 32-element block to MXFP4 (same row math) and the BF16 head gate is folded without a q scale:
  weights = float(bf16(gate * 32**-0.5)) * 128**-0.5 (the MX block scales feed the UMMA);
- the verify logits run deep_gemm.fp8_fp4_paged_mqa_logits (FP32 gates, FP32 logits; the stock DeepGEMM launch, stream ordered).
Paths that read or write FP8 index-K bytes themselves are refused (verify.py / indexer.py checks; fail closed at import or build):
the TSI scan, L37, the query front (K31MK_DSA_QF), the fused index query (K31MK_FUSED_INDEX_Q_VERIFY), DCP, and the non-verify
decode builds. Request sharding (K31MK_DSA_REQ_SHARD) composes: its owner rows run VerifySelection's MXFP4 branch (Fp4Query.index_select).
Engagement (one log line per site at its first call, which is eager, before graph capture): 'K31MK_DSA_IDX_FP4 ENGAGED site=writer',
'site=query', 'site=logits'; STATS counts calls (graph capture counts once per captured call).
"""

from __future__ import annotations

import logging
import os

import torch
import triton
import triton.language as tl

# SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/ops/quantization/fp4_indexer_quant.py": ("_ceil_ue8m0_exp", "_fp4_e2m1_code", "quantize_fp4_indexer_row"),
}

from sglang.kernels.ops.quantization.fp4_indexer_quant import (  # noqa: E402
    _ceil_ue8m0_exp,
    _fp4_e2m1_code,
    quantize_fp4_indexer_row,
)

from .idx_fp4_knob import ENABLED, ENV, refuse  # noqa: E402,F401  (the switch; refuse() fails closed when it is on)

PAGE = 64  # tokens per index-K page
VAL_B = 64  # e2m1 bytes per token (128 dims)
SF_B = 4  # ue8m0 bytes per token (4 groups of 32)
ROW_B = VAL_B + SF_B  # 68
PAGE_B = PAGE * ROW_B  # 4352
HEADS, HEAD_DIM = 32, 128

logger = logging.getLogger(__name__)
STATS = {"writer": 0, "query": 0, "logits": 0}


def _engaged(site: str) -> None:
    STATS[site] += 1
    if STATS[site] == 1:
        from .idx_fp4_knob import FP8Q

        form = ("FP8 index query; label 'MXFP4 indexer cache (FP8 query)' = SA" if FP8Q else
                "MXFP4 index query; label 'MXFP4 indexer q'")
        logger.info("K31MK_DSA_IDX_FP4 ENGAGED site=%s (MXFP4 index-K 68 B/token, %s)", site, form)


def check_cache(cache: torch.Tensor) -> None:
    """The host's MXFP4 index-K pool: uint8 [pages, 64 * 68], contiguous (K31HT_PREC_IDX_FP4=1 on the host)."""
    if cache.dtype != torch.uint8 or cache.ndim != 2 or cache.shape[1] != PAGE_B or not cache.is_contiguous():
        raise ValueError(f"{ENV}=1 needs the host's MXFP4 index-K pool uint8 [pages, {PAGE_B}] (K31HT_PREC_IDX_FP4=1 on every server); "
                         f"got {cache.dtype} {tuple(cache.shape)}")


# ------------------------------------------------------------------------------------------------------------------------------
# Writer: one program per token row; the host prefill writer's row math (bitwise equal bytes for equal keys).
@triton.jit
def _store_kernel(K, K_STRIDE, LOC, BUF_U8, BUF_I32, PAGE_BYTES: tl.constexpr, PAGE_N: tl.constexpr):
    t = tl.program_id(0)
    loc = tl.load(LOC + t).to(tl.int64)
    if loc >= 0:
        offs = tl.arange(0, 128)
        pairs = tl.arange(0, 64)
        row = K + t.to(tl.int64) * K_STRIDE
        values = tl.load(row + offs).to(tl.float32)
        v0 = tl.load(row + pairs * 2).to(tl.float32)
        v1 = tl.load(row + pairs * 2 + 1).to(tl.float32)
        packed, packed_sf = quantize_fp4_indexer_row(values, v0, v1, BLOCK_N=128, GROUP_N=32, RNE=False)
        page = loc // PAGE_N
        off = loc % PAGE_N
        tl.store(BUF_U8 + page * PAGE_BYTES + off * 64 + pairs, packed.to(tl.uint8))
        tl.store(BUF_I32 + page * (PAGE_BYTES // 4) + (PAGE_N * 64) // 4 + off, packed_sf)


# ``K31MK_DSA_IDX_FP4_STORE`` (default cuda): cuda = BF16 contiguous keys with int64 slots go to idx_fp4_store.cuh (the FP8 fused store's
# warp-per-row PDL launch form; the same bytes); every other input, and triton, use the Triton writer below (the reference).
STORE_IMPL = os.environ.get("K31MK_DSA_IDX_FP4_STORE", "cuda")
if STORE_IMPL not in ("cuda", "triton"):
    raise ValueError(f"K31MK_DSA_IDX_FP4_STORE must be cuda or triton, got {STORE_IMPL!r}")


def store_index_k(key: torch.Tensor, cache: torch.Tensor, slots: torch.Tensor, impl: str = "") -> None:
    """MXFP4 counterpart of the host's fused_store_index_k_cache(key, cache, slots, 64): key BF16 / FP32 [T, 128] (row stride free),
    slots int [T] token slots (negative = skip), cache uint8 [pages, 4352]. Graph-safe (one launch, no host sync). ``impl`` (tests):
    'triton' / 'cuda' overrides K31MK_DSA_IDX_FP4_STORE."""
    if key.shape[0] == 0:
        return
    if key.ndim != 2 or key.shape[1] != HEAD_DIM or key.stride(1) != 1 or key.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"{ENV} writer: key [T, 128] BF16 / FP32 with unit column stride, got {key.dtype} {tuple(key.shape)}")
    check_cache(cache)
    loc = slots.reshape(-1)
    if loc.numel() < key.shape[0]:
        raise ValueError(f"{ENV} writer: {loc.numel()} slots for {key.shape[0]} rows")
    if (impl or STORE_IMPL) == "cuda" and key.dtype == torch.bfloat16 and key.stride(0) == HEAD_DIM and loc.dtype == torch.int64 \
            and loc.stride(0) == 1:
        from .idx_fp4_store import store

        store(key, cache, loc)
        _engaged("writer")
        return
    _store_kernel[(key.shape[0],)](key, key.stride(0), loc, cache, cache.view(torch.int32), PAGE_BYTES=PAGE_B, PAGE_N=PAGE, num_warps=1)
    _engaged("writer")


# ------------------------------------------------------------------------------------------------------------------------------
# Index query: BF16 query [T, 32, 128] -> MXFP4 (per 32-group ue8m0) + the BF16 head gate folded without a q scale.
class Fp4Query:
    """MXFP4 index query handed from the q/gate kernel to the logits: q4 int8 [T, 32, 64], sf int32 [T, 32]. ``shape`` mirrors the FP8
    query's [T, 32, 128]."""

    __slots__ = ("q4", "sf", "shape")

    def __init__(self, q4: torch.Tensor, sf: torch.Tensor):
        self.q4, self.sf = q4, sf
        self.shape = (q4.shape[0], q4.shape[1], 2 * q4.shape[2])

    def index_select(self, rows: torch.Tensor) -> "Fp4Query":
        """These query rows (request sharding's owner rows, dsa/req_shard.py), as new contiguous tensors."""
        return Fp4Query(self.q4.index_select(0, rows), self.sf.index_select(0, rows))

    def slice_rows(self, lo: int, n: int):
        """Rows lo .. lo + n - 1 as views (K31MK_DSA_REQ_SHARD_SLICE: the owner block without the gathers), or None when the
        buffers are not row-contiguous (the caller then gathers with index_select)."""
        if not (self.q4.is_contiguous() and self.sf.is_contiguous()):
            return None
        return Fp4Query(self.q4[lo : lo + n], self.sf[lo : lo + n])


@triton.jit(do_not_specialize=["ROWS"])
def _query_kernel(Q, G, Q4, QSF, W, ROWS, HEADS_: tl.constexpr, GATE_STRIDE, GATE_SCALE: tl.constexpr, SOFTMAX_SCALE: tl.constexpr,
                  BLOCK_M: tl.constexpr):
    # BLOCK_M (token, head) rows per program; each 32-value group is one reduction row (fp4_indexer_quant._quantize_fp4_indexer_rows'
    # arithmetic: same exponent and codes as quantize_fp4_indexer_row).
    grp = tl.program_id(0) * (BLOCK_M * 4) + tl.arange(0, BLOCK_M * 4)
    gmask = grp[:, None] < ROWS * 4
    values = tl.load(Q + grp[:, None].to(tl.int64) * 32 + tl.arange(0, 32)[None, :], gmask, other=0).to(tl.float32)
    exp = _ceil_ue8m0_exp(tl.maximum(tl.max(tl.abs(values), axis=1) / 6.0, 1.0e-4))
    scale = (exp << 23).to(tl.float32, bitcast=True)
    v0, v1 = tl.split(tl.reshape(values / scale[:, None], (BLOCK_M * 4, 16, 2)))
    code0 = _fp4_e2m1_code(v0)
    code1 = _fp4_e2m1_code(v1)
    packed = (code0 & 0x0F) | ((code1 & 0x0F) << 4)
    tl.store(Q4 + grp[:, None].to(tl.int64) * 16 + tl.arange(0, 16)[None, :], packed, gmask)
    shifts = tl.arange(0, 4) * 8  # the four exponents occupy disjoint bytes: an integer sum packs them
    packed_sf = tl.sum(tl.reshape(exp.to(tl.uint32), (BLOCK_M, 4)) << shifts[None, :], axis=1)
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < ROWS
    tl.store(QSF + rows, packed_sf.to(tl.int32), rmask)
    gate = tl.load(G + (rows // HEADS_) * GATE_STRIDE + rows % HEADS_, rmask, other=0).to(tl.float32)
    gate = (gate * GATE_SCALE).to(tl.bfloat16).to(tl.float32)
    tl.store(W + rows, gate * SOFTMAX_SCALE, rmask)


class QueryBuffers:
    """Owned outputs of the q/gate kernel for one row capacity (no step-time allocation inside a captured graph)."""

    def __init__(self, rows: int, device):
        self.q4 = torch.empty((rows, HEADS, HEAD_DIM // 2), dtype=torch.int8, device=device)
        self.sf = torch.empty((rows, HEADS), dtype=torch.int32, device=device)
        self.weights = torch.empty((rows, HEADS, 1), dtype=torch.float32, device=device)


def quantize_query(query: torch.Tensor, gate: torch.Tensor, out: QueryBuffers, softmax_scale: float = HEAD_DIM ** -0.5,
                   block_m: int = 16):
    """query BF16 contiguous [T, 32, 128], gate BF16 [T, 32] (row stride free) -> (Fp4Query, weights FP32 [T, 32, 1])."""
    if query.dtype != torch.bfloat16 or query.ndim != 3 or tuple(query.shape[1:]) != (HEADS, HEAD_DIM) or not query.is_contiguous():
        raise ValueError(f"{ENV} query: BF16 contiguous [T, 32, 128], got {query.dtype} {tuple(query.shape)}")
    t = query.shape[0]
    if tuple(gate.shape) != (t, HEADS) or gate.stride(1) != 1 or gate.dtype != torch.bfloat16:
        raise ValueError(f"{ENV} query: gate BF16 [T, 32] with unit column stride, got {gate.dtype} {tuple(gate.shape)}")
    if out.q4.shape[0] < t:
        raise ValueError(f"{ENV} query: buffers hold {out.q4.shape[0]} rows, {t} given")
    q4, sf, w = out.q4[:t], out.sf[:t], out.weights[:t]
    if t:
        _query_kernel[(triton.cdiv(t * HEADS, block_m),)](query, gate, q4, sf, w, t * HEADS, HEADS, gate.stride(0), HEADS ** -0.5,
                                                           softmax_scale, block_m, num_warps=4)
        _engaged("query")
    return Fp4Query(q4, sf), w


# ------------------------------------------------------------------------------------------------------------------------------
def paged_logits(query: Fp4Query, gates: torch.Tensor, cache: torch.Tensor, context_lens: torch.Tensor, block_table: torch.Tensor,
                 schedule: torch.Tensor, max_context_len: int, batch_requests: int, query_width: int) -> torch.Tensor:
    """deep_gemm.fp8_fp4_paged_mqa_logits over the MXFP4 pages: FP32 logits [B * N, max_context_len] (DG row stride; the caller
    restores it as for the FP8 call). gates FP32 [B * N, 32] (or [B * N, 32, 1]); context_lens [B, N] int32; block_table [B, W]."""
    import deep_gemm

    if not isinstance(query, Fp4Query):
        raise TypeError(f"{ENV} logits: expected the MXFP4 query (Fp4Query), got {type(query).__name__}")
    check_cache(cache)
    b, n = int(batch_requests), int(query_width)
    rows = b * n
    if query.q4.shape[0] != rows:
        raise ValueError(f"{ENV} logits: query rows {query.q4.shape[0]} != {b} x {n}")
    _engaged("logits")
    return deep_gemm.fp8_fp4_paged_mqa_logits(
        (query.q4.view(b, n, HEADS, HEAD_DIM // 2), query.sf.view(b, n, HEADS)),
        cache.view(-1, PAGE, 1, ROW_B),
        gates.reshape(rows, HEADS),
        context_lens,
        block_table,
        schedule,
        max_context_len,
        False,
        logits_dtype=torch.float32,
    )
