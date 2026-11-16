# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The folded page64 writer, with independent short and long decode graphs.

Decode graphs of 1-8 rows run the stock index chain's method as one PDL kernel (index_q.FusedIndexQuery: rmsnorm,
wq_b, act_quant, head gate and the weights_proj gate in one launch, which also writes the sparse core's 16-head padded
query) and read the sparse core's 16-head output directly in MlaValue (no latent copy). Selection stays stock
(DeepGEMM logits + topk_transform_paged_v2). ``K31MK_FUSED_INDEX_Q=0`` restores the stock chain at every width.
"""

import os

import torch

from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import stream, to_cute
from sgl_nvidia_mega.kimi_k31.folded_index import PreparedIndexMla

from .indexer import IndexProjection, IndexWeights, PagedSelection
from .mla import SparseMla
from sgl_nvidia_mega.kimi_k31.k3.cutedsl.mla.head_gate import head_gate

FUSED_INDEX_Q = os.environ.get("K31MK_FUSED_INDEX_Q", "1") != "0"
FUSED_INDEX_Q_MAX_ROWS = 8


class PreparedSparseIndexMla(PreparedIndexMla):
    def __init__(self, attention_and_indexer, *args, **kwargs):
        attention, indexer = attention_and_indexer
        if kwargs.get("dcp") is not None or kwargs.get("queries", (1,)) != (1,):
            raise ValueError(
                "K3.1 sparse provider currently serves ordinary DCP1 decode"
            )
        if indexer.num_init_tokens or indexer.num_local_tokens:
            raise ValueError("K3.1 sparse provider requires pure score-based top2048")
        super().__init__(attention_and_indexer, *args, **kwargs)
        weights = IndexWeights(
            indexer.wq_b.weight,
            indexer.weights_proj.weight,
            attention.q_a_layernorm.weight,
            float(attention.rms_norm_eps),
        )
        self.fused_rows = {r for r in self.pitches if FUSED_INDEX_Q and r <= FUSED_INDEX_Q_MAX_ROWS}
        if self.fused_rows:
            from .index_q import FusedIndexQuery
        self.sparse_rows = {
            rows: (
                FusedIndexQuery(weights, rows) if rows in self.fused_rows else IndexProjection(weights, rows),
                PagedSelection(self.device, rows),
                SparseMla(self.device, self.max_seq, rows, self.heads),
            )
            for rows in self.pitches
        }
        self.hidden = None
        self.padded_values = {}
        self.index_projection, self.selection, self.sparse_mla = next(iter(self.sparse_rows.values()))
        self.sparse = False
        self.index_step = None

    def set_step(self, sparse, index_step):
        # This number selects captured work; replay selection stays in SGLang.
        self.sparse, self.index_step = bool(sparse), index_step

    def prepare_gate(self, hidden):
        if self.sparse:
            rows = hidden.shape[0]
            self.index_projection, self.selection, self.sparse_mla = self.sparse_rows[rows]
            if rows in self.fused_rows:
                self.hidden = hidden  # the fused index-query kernel computes the head gate itself
            else:
                self.index_projection.prepare_gate(hidden)

    def prepare_cache(self, cache):
        out = super().prepare_cache(cache)
        if self.fused_rows:
            from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import layout

            from .verify import _padded_value

            for rows, q in self.kernels:
                if q == 1 and rows in self.fused_rows:
                    self.padded_values[rows] = _padded_value(
                        rows, self.pitches[rows], layout(self.weights[4]), self.device.index, 1, self.heads
                    )
        return out

    def run(
        self, projected, out, *, cache, slots, block_table, lengths, totals=None, q=1
    ):
        if not self.sparse:
            return super().run(
                projected,
                out,
                cache=cache,
                slots=slots,
                block_table=block_table,
                lengths=lengths,
                totals=totals,
                q=q,
            )
        rows = projected.shape[0]
        if q != 1 or self.index_step is None:
            raise ValueError("K3.1 sparse MLA requires an ordinary decode index step")
        mq, value, query, latent = self.kernels[rows, 1]
        qnorm, kvnorm, qb, uk, uv, qinv, kinv = self.views
        args = (to_cute(projected[:, :2112]), qnorm, kvnorm, qb, uk)
        args += (to_cute(cache.view(torch.uint8)), to_cute(slots, 8))
        args += (qinv, kinv, self.eps, query)
        s = stream(self.device)
        # Includes the current token's main KV and planar index-K writes.
        mq(*args, *self.query_views, s)
        if rows in self.fused_rows:
            # Also copies the MLA query into the sparse core's 16-head buffer (pad heads stay zero).
            iq, gates = self.index_projection.run(
                self.hidden, projected, self.query_scratch[:rows], self.sparse_mla.query
            )
            selected = self.selection.run(
                iq,
                gates,
                self.index_cache,
                block_table,
                lengths,
                self.index_step.schedule,
                self.index_step.topk_plan,
            )
            self.sparse_mla.run_core(cache, selected, lengths, self.stock_scales)
            self.padded_values[rows](
                to_cute(self.sparse_mla.output[:, 0, : self.heads]),
                uv,
                to_cute(head_gate(projected, self.heads)),
                to_cute(lengths, 4),
                to_cute(out),
                s,
            )
            return out
        iq, gates = self.index_projection.project_query(projected)
        selected = self.selection.run(
            iq,
            gates,
            self.index_cache,
            block_table,
            lengths,
            self.index_step.schedule,
            self.index_step.topk_plan,
        )
        self.sparse_mla.run(
            self.query_scratch[:rows],
            cache,
            selected,
            lengths,
            self.stock_scales,
            self.latent[:rows],
        )
        value(
            latent,
            uv,
            to_cute(head_gate(projected, self.heads)),
            to_cute(lengths, 4),
            to_cute(out),
            s,
        )
        return out
