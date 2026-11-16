# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 DSA index query, query-front WIDE form (K31MK_DSA_QFW with K31MK_DSA_QF; index_qfw.cuh): the sparse verify steps of
9-32 rows (Breq 2-4 x Q8) run the index query as one PDL kernel behind the q in-proj, read the head gate from the packed
projection and store the new tokens' index-K rows, as the 8-row query front (index_q.py run_qf) does.

Numerics-changing vs stock (header of index_qfw.cuh): the q_a RMSNorm weight is folded into wq_b at load time
(wq_bg = bf16(wq_b * w_qa)) and rstd is applied to the FP32 product (q = bf16(rstd * (x @ wq_bg^T))). Memory: one folded
[4096, 1536] BF16 copy per DSA layer (12.6 MB), shared by every row profile of the layer."""

from functools import cache
from pathlib import Path

import torch


# SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {"kernels/jit/utils/compile/loader.py": ("load_jit",)}

# (wq_b data_ptr, w_qa data_ptr) -> (wq_b, w_qa, wq_b._version, w_qa._version, wq_bg). The entry holds the source tensors, so their
# memory cannot be freed and handed to new weights at the same address while the entry exists; the version counters (shared by
# views) catch an in-place update of the same memory. Either case refolds instead of returning another tensor's fold.
_FOLDED = {}


@cache
def _load():
    import sglang
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "k31_index_qfw",
        cuda_files=[str(Path(__file__).with_suffix(".cuh"))],
        cuda_wrappers=[("run_qfw", "k31_index_qfw::Binding::run_qfw")],
        extra_include_paths=[str(Path(sglang.__file__).parent / "kernels/jit/csrc")],
    )


def folded_query(weights):
    """bf16(wq_b * w_qa) of an IndexWeights (one copy per layer, built once; refolded if the weights changed in place since). A
    FusedIndexQueryWide keeps the fold it was built with: weights changed in place after that need a new instance."""
    query, norm = weights.query, weights.query_norm
    key = (query.data_ptr(), norm.data_ptr())
    entry = _FOLDED.get(key)
    if entry is not None and entry[2] == query._version and entry[3] == norm._version:
        return entry[4]
    folded = (query.float() * norm.float()[None, :]).to(torch.bfloat16).contiguous()
    _FOLDED[key] = (query, norm, query._version, norm._version, folded)
    return folded


class FusedIndexQueryWide:
    """Drop-in for index_q.FusedIndexQuery's query-front form (run_qf) on 9-32 rows: returns (q8 [rows, 32, 128] FP8,
    weights [rows, 32, 1] FP32), owned buffers, and stores the rows' index-K."""

    def __init__(self, weights, rows):
        if weights.query.shape != (4096, 1536) or weights.query_norm.shape != (1536,) or not 9 <= rows <= 32:
            raise ValueError("QF-wide index query: replicated 32x128 heads, 9-32 rows")
        self.weights, self.rows = weights, rows
        device = weights.query.device
        self.folded = folded_query(weights)
        self.query = torch.empty((rows, 32, 128), dtype=torch.uint8, device=device)
        self.gates = torch.empty((rows, 32, 1), dtype=torch.float32, device=device)
        self.module = _load()

    def run_qf(self, projected, key_col, gate_col, slots, cache, key_norm, kr=0):
        """``projected`` [rows, pitch] holds q_a at [0, 1536), the index-K rows at [key_col, key_col + 128) and the head-gate
        logits at [gate_col, gate_col + 32); the rows' index-K go to ``cache`` [pages, 64 * 132] at ``slots`` (int64; negative
        skips) with ``key_norm`` = (FP32 weight [128], FP32 bias [128], eps). ``kr`` forces the 16 / 24 / 32-row instance
        (tests; 0 = the smallest that holds the rows)."""
        gamma, beta, eps = key_norm
        self.module.run_qfw(projected, self.folded, self.query, self.gates.view(-1, 32), float(self.weights.epsilon),
                            int(key_col), int(gate_col), slots, cache, gamma, beta, float(eps), int(kr))
        return self.query.view(torch.float8_e4m3fn), self.gates
