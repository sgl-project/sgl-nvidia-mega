# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 DSA index query, query-front CROSS form (K31MK_DSA_QFX with K31MK_DSA_QF; index_qfx.cuh): the sparse verify steps of
33-128 rows (Breq 5-16 x Q8) run the index query as one PDL kernel behind the q in-proj, read the head gate from the packed
projection and store the new tokens' index-K rows, as QF-wide (index_qfw.py, 9-32 rows) does.

Same method and rounding points as QF-wide, and each row's outputs are bitwise QF-wide's on that row (index_qfx.cuh header):
numerics-changing vs stock in the same way (q_a RMSNorm weight folded into wq_b at load, rstd applied to the FP32 product). The
fold is QF-wide's (index_qfw.folded_query): one [4096, 1536] BF16 copy per DSA layer, shared with the QF-wide profiles."""

import os
from functools import cache
from pathlib import Path

import torch

from .index_qfw import folded_query

# SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {"kernels/jit/utils/compile/loader.py": ("load_jit",)}

MIN_ROWS, MAX_ROWS = 9, 128  # 9-32 only under K31MK_DSA_QFX_LOW (verify.py), in place of QF-wide
# ``K31MK_DSA_QFX_TRIGGER`` (default 0): where the kernel lets its PDL dependents (the index logits) launch - 0 right after its own
# griddepcontrol.wait (QF-wide's placement), 1 after its DSMEM exchange, 2 at its end. Outputs are identical in every mode (index_qfx.cuh).
TRIGGER = int(os.environ.get("K31MK_DSA_QFX_TRIGGER", "0"))
if TRIGGER not in (0, 1, 2):
    raise ValueError(f"K31MK_DSA_QFX_TRIGGER must be 0, 1 or 2, got {TRIGGER}")


@cache
def _load():
    import sglang
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "k31_index_qfx",
        cuda_files=[str(Path(__file__).with_suffix(".cuh"))],
        cuda_wrappers=[("run_qfx", "k31_index_qfx::Binding::run_qfx")],
        extra_include_paths=[str(Path(sglang.__file__).parent / "kernels/jit/csrc")],
    )


class FusedIndexQueryCross:
    """Drop-in for index_q.FusedIndexQuery's query-front form (run_qf) on 33-128 rows (9-128 under K31MK_DSA_QFX_LOW): returns (q8 [rows, 32, 128] FP8,
    weights [rows, 32, 1] FP32), owned buffers, and stores the rows' index-K."""

    def __init__(self, weights, rows):
        if weights.query.shape != (4096, 1536) or weights.query_norm.shape != (1536,) or not MIN_ROWS <= rows <= MAX_ROWS:
            raise ValueError(f"QF-cross index query: replicated 32x128 heads, {MIN_ROWS}-{MAX_ROWS} rows")
        self.weights, self.rows = weights, rows
        device = weights.query.device
        self.folded = folded_query(weights)
        self.query = torch.empty((rows, 32, 128), dtype=torch.uint8, device=device)
        self.gates = torch.empty((rows, 32, 1), dtype=torch.float32, device=device)
        self.module = _load()
        self.trig = TRIGGER  # tests / rigs may set it per instance

    def run_qf(self, projected, key_col, gate_col, slots, cache, key_norm, kt=0):
        """``projected`` [rows, pitch] holds q_a at [0, 1536), the index-K rows at [key_col, key_col + 128) and the head-gate
        logits at [gate_col, gate_col + 32); the rows' index-K go to ``cache`` [pages, 64 * 132] at ``slots`` (int64; negative
        skips) with ``key_norm`` = (FP32 weight [128], FP32 bias [128], eps). ``kt`` forces an instance of kt n8 row tiles
        (tests; 0 = ceil(rows / 8))."""
        gamma, beta, eps = key_norm
        self.module.run_qfx(projected, self.folded, self.query, self.gates.view(-1, 32), float(self.weights.epsilon),
                            int(key_col), int(gate_col), slots, cache, gamma, beta, float(eps), int(kt), int(self.trig))
        return self.query.view(torch.float8_e4m3fn), self.gates
