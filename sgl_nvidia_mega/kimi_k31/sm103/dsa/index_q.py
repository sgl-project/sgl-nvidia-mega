# SPDX-License-Identifier: Apache-2.0
"""K3.1 DSA index query + head gate as one PDL kernel (index_q.cuh): replaces IndexProjection's stock decode chain
(rmsnorm, wq_b mm, act_quant, bf16 head gate) and prepare_gate's weights_proj mm with the same rounding points."""

from functools import cache
from pathlib import Path

import torch

# Names read from SGLang. A contract test checks these before GPU use.
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("load_jit",),
}


@cache
def _load():
    import sglang
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "k31_index_q",
        cuda_files=[str(Path(__file__).with_suffix(".cuh"))],
        cuda_wrappers=[
            ("run", "k31_index_q::Binding::run"),
            ("run_copy", "k31_index_q::Binding::run_copy"),
        ],
        extra_include_paths=[str(Path(sglang.__file__).parent / "kernels/jit/csrc")],
    )


class FusedIndexQuery:
    """Drop-in for IndexProjection.run(hidden, projected) on 1-8 rows: returns (q8 [rows,32,128] FP8,
    weights [rows,32,1] FP32), owned buffers."""

    def __init__(self, weights, rows=1):
        if (
            weights.query.shape != (4096, 1536)
            or weights.gate.shape != (32, 7168)
            or not 1 <= rows <= 8
        ):
            raise ValueError("fused index query: replicated 32x128 heads, 1-8 rows")
        self.weights, self.rows = weights, rows
        device = weights.query.device
        self.query = torch.empty((rows, 32, 128), dtype=torch.uint8, device=device)
        self.gates = torch.empty((rows, 32, 1), dtype=torch.float32, device=device)
        self.module = _load()

    def run(self, hidden, projected, mla_query=None, padded_query=None):
        """``mla_query`` [rows, heads, 576] / ``padded_query`` [rows, 1, 16, 576]: also copy the MLA query into the
        sparse core's head-padded buffer (one launch fewer than SparseMla.prepare_query)."""
        w = self.weights
        args = (
            projected,
            hidden,
            w.query_norm,
            w.query,
            w.gate,
            self.query,
            self.gates.view(-1, 32),
            float(w.epsilon),
        )
        if mla_query is None:
            self.module.run(*args)
        else:
            self.module.run_copy(
                *args,
                mla_query.view(torch.uint8).reshape(mla_query.shape[0], -1),
                padded_query.view(torch.uint8).reshape(padded_query.shape[0], -1),
                mla_query.shape[1],
            )
        return self.query.view(torch.float8_e4m3fn), self.gates
