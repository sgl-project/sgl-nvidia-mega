# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 TARGET_VERIFY (DFLASH Q8, Breq 1/2/4) under DCP = TP (``K31MK_DCP=1``).

Stock runs target verify on the DSA decode implementation under DCP (``speculative_attention_mode=decode``): every
draft token is its own row with its causal global length P + j + 1, local length on this rank, its own global top-k
and its own LSE merge. The DCP layer core (dcp/attention.py) already serves such rows; this module adapts the
tensor-ABI verify build:

- MLA cores: PreparedDcpSparseMla over the verify build's PreparedMla (H local heads; KV inserted only at this
  rank's slots), with the stock 8-row index chain (``IndexProjection``: rmsnorm, wq_b mm, act_quant + head gate).
- Index-K writes: the stock writer with this rank's slots (``fused_store_index_k_cache`` skips negative slots).
- Executor: one prologue per verify forward (epoch, widened slots -> local rows).

Host step fields under DCP (SparseVerifyStep): ``score_contexts`` [Breq, Q] LOCAL causal lengths,
``score_schedule`` their DeepGEMM schedule, ``causal_lengths`` [R] GLOBAL causal lengths, ``topk_plan`` the topk_v2
plan of the global causal lengths, ``block_table`` widened pages, ``write_slots`` widened slots.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch

from ..dsa.indexer import IndexProjection, IndexWeights
from ..dsa.verify import BuiltSparseVerify, SparseVerifyStep
from ..verify_executor import K31VerifyExecutor

SGLANG_READS = {
    "srt/runtime_context.py": ("get_parallel",),
}


@dataclass(frozen=True)
class _DcpRows:
    """The DCP layer's view of one verify step."""

    schedule: torch.Tensor
    topk_plan: torch.Tensor
    local_lengths: torch.Tensor
    global_lengths: torch.Tensor


class DcpVerifyMla:
    def __init__(self, base, weights: IndexWeights, comm, ordinal, widths, max_context, dense):
        from .attention import PreparedDcpSparseMla

        self.base = base
        self.projections = {rows: IndexProjection(weights, rows) for rows in widths}
        self.core = PreparedDcpSparseMla(
            base, comm, ordinal, max_context=max_context, dense=dense, index_query=self._index_query
        )
        self._step = None

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, view: SparseVerifyStep):
        self._step = view
        rows = _DcpRows(view.score_schedule, view.topk_plan, view.score_contexts, view.causal_lengths)
        self.core.set_step(bool(view.sparse), rows)

    def set_step(self, sparse, step):  # PreparedDcpSparseMla's base hook: PreparedMla has no indexer state
        pass

    @property
    def latent(self):
        return self.base.latent

    def _index_query(self, projected, rows):
        return self.projections[rows].project_query(projected)

    def prepare_gate(self, hidden):
        if self._step is not None and self._step.sparse:
            self.projections[hidden.shape[0]].prepare_gate(hidden)

    def prepare_cache(self, cache):
        self.core.prepare_cache(cache)
        return self

    def run(self, projected, out, *, cache, slots, block_table, lengths, totals=None, q=1):
        return self.core.run(projected, out, cache=cache, slots=slots, block_table=block_table, lengths=lengths, q=q)


class DcpVerifyIndexKey:
    """The stock index-K writer at this rank's slots, plus the index gate of this layer's verify rows."""

    def __init__(self, writer, core: DcpVerifyMla):
        self.writer, self.core = writer, core

    def bind(self, cache, page_size, rows=(8,)):
        self.core.base.index_cache = cache
        self.core.core.base.index_cache = cache
        return self.writer.bind(cache, page_size, rows)

    def run(self, hidden, slots):
        self.writer.run(hidden, slots)
        self.core.prepare_gate(hidden)


def local_slot_rows(cap: int) -> int:
    """Rows of the prologue's local-slot buffer: the DCP mailboxes' capacity (``DcpComm.cap`` = the boot's largest verify
    width), never fewer than the 64 every boot up to Breq 8 held (their buffer is unchanged). The prologue refuses a buffer
    with fewer slots than rows (comm.cuh RuntimeCheck "local slots"), so a fixed 64 failed every Breq 16 (128-row) step."""
    return max(64, int(cap))


class DcpVerifyExecutor(K31VerifyExecutor):
    def __init__(self, *args, comm, **kwargs):
        super().__init__(*args, **kwargs)
        self.comm = comm
        self.local_slots = torch.full((local_slot_rows(comm.cap),), -1, dtype=torch.int64, device=comm.device)

    def run(self, embeddings, view):
        rows = int(view.row_count)
        local = self.local_slots[:rows]
        self.comm.prologue(view.write_slots, local)
        return super().run(embeddings, dataclasses.replace(view, write_slots=local))


class BuiltDcpVerify(BuiltSparseVerify):
    def __init__(self, weights, deadline_s):
        from sglang.srt.runtime_context import get_parallel

        from .attention import dense_scores
        from .comm import DcpComm

        super().__init__(weights, deadline_s)
        parallel = get_parallel()
        tp, rank = int(parallel.tp_size), int(parallel.tp_rank)
        dcp = int(getattr(parallel, "attn_dcp_size", 1))
        if dcp != tp:
            raise ValueError(f"the K3.1 MK DCP verify needs DCP = TP, got DCP{dcp} / TP{tp}")
        device = self.resources.scratch["hidden.0"].device
        heads = self.resources.heads
        self.dcp_comm = comm = DcpComm.from_group(
            parallel.tp_group.device_group, device, dcp, rank, heads, max(self.widths)
        )
        dense = dense_scores(max(self.widths), weights.context_length, dcp, device)
        ordinals = list(weights.full_attention_ids)
        for record in weights.layers:
            if record.attention_kind != 0:
                continue
            a, sc = record.attention, record.attention_scalars
            attention = self.resources.attention[record.layer_id]
            base = self.sparse_cores[record.layer_id].base  # the verify build's PreparedMla
            core = DcpVerifyMla(
                base, IndexWeights(a[10], a[11], a[1], sc[0]), comm, ordinals.index(record.layer_id),
                self.widths, weights.context_length, dense,
            )
            attention.core = core
            self.sparse_cores[record.layer_id] = core
            writer = self.indexers[record.layer_id].base  # PreparedVerifyIndexKey under the DCP1 wrapper
            self.indexers[record.layer_id] = DcpVerifyIndexKey(writer, core)
        self._replace_executor(())

    def _replace_executor(self, capture_layers):
        old = self.executor
        self.executor = DcpVerifyExecutor(
            self.resources, self.indexers, capture_layers, query_width=self.query_width,
            request_batches=self.request_batches, comm=self.dcp_comm,
        )
        self.executor.kda_cores = old.kda_cores
        self.executor.kda_pools = old.kda_pools

    def configure_aux(self, capture_layers):
        self._replace_executor(capture_layers)

    def summary(self):
        tp = 96 // self.resources.heads
        return f"TARGET_VERIFY Breq={self.request_batches} Q={self.query_width}; TP{tp}/DCP{tp}; causal sparse DSA"

    def run_verify(self, embeddings, view):
        if not isinstance(view, SparseVerifyStep):
            raise ValueError("DCP verify requires the host causal-metadata adapter")
        view.validate()
        for core in self.sparse_cores.values():
            core.step = view
        return super(BuiltSparseVerify, self).run_verify(embeddings, view)


def build_from_tensors(weights, deadline_s):
    return BuiltDcpVerify(weights, deadline_s)
