# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 DSA index query + head gate as one PDL kernel (index_q.cuh): replaces IndexProjection's stock decode chain
(rmsnorm, wq_b mm, act_quant, bf16 head gate) and prepare_gate's weights_proj mm with the same rounding points."""

from functools import cache
from pathlib import Path

import torch


# SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {"kernels/jit/utils/compile/loader.py": ("load_jit",)}


@cache
def _load():
    import sglang
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "k31_index_q",
        cuda_files=[str(Path(__file__).with_suffix(".cuh"))],
        cuda_wrappers=[("run", "k31_index_q::Binding::run"), ("run_copy", "k31_index_q::Binding::run_copy"),
                       ("run_qf", "k31_index_q::Binding::run_qf")],
        extra_include_paths=[str(Path(sglang.__file__).parent / "kernels/jit/csrc")],
    )


class FusedIndexQuery:
    """Drop-in for IndexProjection.run(hidden, projected) on 1-8 rows: returns (q8 [rows,32,128] FP8,
    weights [rows,32,1] FP32), owned buffers. ``qf_only=True`` also admits 9-16 rows for the query-front form
    (``run_qf`` only; K31MK_DSA_QF_MAX_ROWS=16)."""

    def __init__(self, weights, rows=1, qf_only=False):
        max_rows = 16 if qf_only else 8
        if weights.query.shape != (4096, 1536) or weights.gate.shape != (32, 7168) or not 1 <= rows <= max_rows:
            raise ValueError(f"fused index query: replicated 32x128 heads, 1-{max_rows} rows")
        self.weights, self.rows = weights, rows
        device = weights.query.device
        self.query = torch.empty((rows, 32, 128), dtype=torch.uint8, device=device)
        self.gates = torch.empty((rows, 32, 1), dtype=torch.float32, device=device)
        self.module = _load()

    def run(self, hidden, projected, mla_query=None, padded_query=None):
        """``mla_query`` [rows, H, 576] (H = 12 at TP8, 6 at TP16) / ``padded_query`` [rows, 1, 16, 576]: also copy
        the MLA query into the sparse core's head-padded buffer (one launch fewer than SparseMla.prepare_query)."""
        w = self.weights
        args = (projected, hidden, w.query_norm, w.query, w.gate, self.query, self.gates.view(-1, 32), float(w.epsilon))
        if mla_query is None:
            self.module.run(*args)
        else:
            self.module.run_copy(*args, mla_query.view(torch.uint8).reshape(mla_query.shape[0], -1),
                                 padded_query.view(torch.uint8).reshape(padded_query.shape[0], -1))
        return self.query.view(torch.float8_e4m3fn), self.gates

    def run_qf(self, projected, key_col, gate_col, slots, cache, key_norm):
        """Query-front form (K31MK_DSA_QF): ``projected`` [rows, pitch] also holds the index-K rows at
        [key_col, key_col + 128) and the head-gate logits at [gate_col, gate_col + 32), both BF16 outputs of the input
        projection. Also stores the rows' index-K into ``cache`` [pages, 64 * 132] at ``slots`` (int64; negative skips),
        with ``key_norm`` = (FP32 weight [128], FP32 bias [128], eps), as LayerNorm + fused_store_index_k_cache do."""
        w = self.weights
        gamma, beta, eps = key_norm
        self.module.run_qf(projected, w.query_norm, w.query, self.query, self.gates.view(-1, 32), float(w.epsilon),
                           int(key_col), int(gate_col), slots, cache, gamma, beta, float(eps))
        return self.query.view(torch.float8_e4m3fn), self.gates
