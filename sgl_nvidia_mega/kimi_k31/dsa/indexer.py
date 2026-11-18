# SPDX-License-Identifier: Apache-2.0
"""Tensor-only K3.1 index query and the stock-compatible paged selector.

The cache is planar within each page: 64*128 FP8 bytes, then 64 FP32 scales.
The [pages,64,1,132] shape is DeepGEMM's ABI, not an interleaved token layout.
"""

from dataclasses import dataclass

import torch

from .scratch import row_buffer

# Names read from SGLang. A contract test checks these before GPU use.
SGLANG_READS = {
    "kernels/ops/attention/dsa/head_gate.py": ("indexer_q_quant_head_gate",),
    "kernels/ops/attention/dsa/paged_mqa_logits.py": ("_restore_row_stride",),
    "kernels/ops/attention/dsv4/topk.py": ("topk_transform_paged_v2",),
}


@dataclass(frozen=True)
class IndexWeights:
    query: torch.Tensor
    gate: torch.Tensor
    query_norm: torch.Tensor
    epsilon: float


class IndexProjection:
    def __init__(self, weights: IndexWeights, rows=1, *, scratch=None):
        if weights.query.shape != (4096, 1536) or weights.gate.shape != (32, 7168):
            raise ValueError("K3.1 indexer requires replicated 32x128 heads")
        if any(w.dtype != torch.bfloat16 for w in (weights.query, weights.gate)):
            raise ValueError("K3.1 sparse decode requires BF16 index projections")
        self.rows = rows
        self.weights = weights
        device = weights.query.device
        self.query = row_buffer(
            scratch, "index_projection.query", rows, (32, 128), torch.bfloat16, device
        )
        self.gate = row_buffer(
            scratch, "index_projection.gate", rows, (32,), torch.bfloat16, device
        )

    def prepare_gate(self, hidden):
        torch.mm(hidden, self.weights.gate.T, out=self.gate)

    def project_query(self, projected):
        """Preserve the stock BF16 GEMM and head-gate rounding points: the stock indexer's fused Q quant and head
        gate at every row count (act_quant + the BF16 gate fold in one kernel, as SGLang's decode since #174)."""
        from sgl_kernel import rmsnorm
        from sglang.kernels.ops.attention.dsa.head_gate import indexer_q_quant_head_gate

        w = self.weights
        normalized = rmsnorm(projected[:, :1536], w.query_norm, w.epsilon)
        torch.mm(normalized, w.query.T, out=self.query.view(-1, 4096))
        return indexer_q_quant_head_gate(self.query, self.gate, 128**-0.5)

    def run(self, hidden, projected):
        self.prepare_gate(hidden)
        return self.project_query(projected)


class PagedSelection:
    """Stock DeepGEMM scores and exact same top-k implementation/tie mode.

    Metadata belongs to the current decode step. It is refreshed on-device
    before this class runs and is shared by all indexer layers.
    """

    def __init__(self, device, rows=1):
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
