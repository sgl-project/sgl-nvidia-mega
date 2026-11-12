# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tensor-only K3.1 index query and the stock-compatible paged selector.

The cache is planar within each page: 64*128 FP8 bytes, then 64 FP32 scales.
The [pages,64,1,132] shape is DeepGEMM's ABI, not an interleaved token layout.
"""

import logging
import os
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


# SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/ops/attention/dsa/paged_mqa_logits.py": ("_restore_row_stride",),
    "kernels/ops/quantization/act_quant_triton.py": ("act_quant",),
    "kernels/ops/attention/dsv4/topk.py": ("topk_transform_paged_v2",),
}
# The QF head gate (K31MK_DSA_QF) reads a kernel of the Kimi K3.1 SGLang host changes; without it that path fails at import.
SGLANG_OPTIONAL_READS = {"kernels/ops/attention/dsa/head_gate.py": ("indexer_q_quant_head_gate",)}

# Index-q FP8 scale format: stock builds the K3.1 Indexer with scale_fmt="ue8m0" (models/deepseek_v2.py)
SCALE_FMT = "ue8m0"

# ``K31MK_DSA_QFN=1`` (default 0 = off, the stock chain): IndexProjection.project_query on >= K31MK_DSA_QFN_MIN_ROWS rows (default 2)
# drops the q_a RMSNorm launch - the wq_b GEMM reads the raw q_a against wq_bg = bf16(wq_b * w_qa) (index_qfw.folded_query, one
# copy per DSA layer, shared with QF-wide) and index_qfn's quant kernel applies rstd before the stock quantization and head gate.
# Numerics-changing (index_qfn.py header). Rows served by a fused index query (QF / QF-wide) never reach project_query.
DSA_QFN = os.environ.get("K31MK_DSA_QFN", "0") == "1"
DSA_QFN_MIN_ROWS = int(os.environ.get("K31MK_DSA_QFN_MIN_ROWS", "2"))
if DSA_QFN_MIN_ROWS < 2:
    raise ValueError(f"K31MK_DSA_QFN_MIN_ROWS must be >= 2 (the 1-row decode chain stays stock), got {DSA_QFN_MIN_ROWS}")
_qfn_logged = set()


@dataclass(frozen=True)
class IndexWeights:
    query: torch.Tensor
    gate: torch.Tensor
    query_norm: torch.Tensor
    epsilon: float


class IndexProjection:
    def __init__(self, weights: IndexWeights, rows=1):
        if weights.query.shape != (4096, 1536) or weights.gate.shape != (32, 7168):
            raise ValueError("K3.1 indexer requires replicated 32x128 heads")
        if any(w.dtype != torch.bfloat16 for w in (weights.query, weights.gate)):
            raise ValueError("K3.1 sparse decode requires BF16 index projections")
        self.rows = rows
        self.weights = weights
        options = dict(device=weights.query.device, dtype=torch.bfloat16)
        self.query = torch.empty((rows, 32, 128), **options)
        self.gate = torch.empty((rows, 32), **options)
        self.folded = None  # K31MK_DSA_QFN: bf16(wq_b * w_qa), built here (before any graph capture), shared per layer
        if DSA_QFN and rows >= DSA_QFN_MIN_ROWS:
            from .index_qfw import folded_query

            self.folded = folded_query(weights)
            if rows not in _qfn_logged:
                _qfn_logged.add(rows)
                logger.info("DSA QFN engaged: rows %d (index query without the q_a RMSNorm launch)", rows)

    def prepare_gate(self, hidden):
        torch.mm(hidden, self.weights.gate.T, out=self.gate)

    def project_query(self, projected):
        """Preserve the stock BF16 GEMM and head-gate rounding points."""
        if self.folded is not None:  # K31MK_DSA_QFN
            from .index_qfn import qfn_quant_head_gate

            torch.mm(projected[:, :1536], self.folded.T, out=self.query.view(-1, 4096))
            return qfn_quant_head_gate(self.query, projected, self.gate, self.weights.epsilon, 128**-0.5, SCALE_FMT)
        from sgl_kernel import rmsnorm
        from sglang.kernels.ops.attention.dsa.head_gate import indexer_q_quant_head_gate

        w = self.weights
        normalized = rmsnorm(projected[:, :1536], w.query_norm, w.epsilon)
        torch.mm(normalized, w.query.T, out=self.query.view(-1, 4096))
        return indexer_q_quant_head_gate(self.query, self.gate, 128**-0.5, SCALE_FMT)

    def run(self, hidden, projected):
        self.prepare_gate(hidden)
        return self.project_query(projected)


class PagedSelection:
    """Stock DeepGEMM scores and exact same top-k implementation/tie mode.

    Metadata belongs to the current decode step. It is refreshed on-device
    before this class runs and is shared by all indexer layers.
    """

    def __init__(self, device, rows=1):
        from .idx_fp4_knob import refuse

        refuse("the non-verify paged selection (dsa/indexer.PagedSelection)")
        self.selected = torch.empty((rows, 2048), device=device, dtype=torch.int32)
        self.raw = torch.empty_like(self.selected)

    def run(self, query, gates, cache, table, lengths, schedule, plan):
        import deep_gemm
        from sglang.kernels.ops.attention.dsa.paged_mqa_logits import (
            _restore_row_stride,
        )
        from sglang.kernels.ops.attention.dsv4.topk import topk_transform_paged_v2

        scores = deep_gemm.fp8_paged_mqa_logits(
            query.unsqueeze(1),
            cache.view(-1, 64, 1, 132),
            gates.squeeze(-1),
            lengths.view(-1, 1),
            table,
            schedule,
            table.shape[1] * 64,
            clean_logits=False,
        )
        scores = _restore_row_stride(scores)
        topk_transform_paged_v2(
            scores,
            lengths,
            table,
            self.selected,
            64,
            plan,
            self.raw,
            deterministic=False,
        )
        return self.selected
