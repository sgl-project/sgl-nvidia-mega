# SPDX-License-Identifier: Apache-2.0
"""Independent request/query executor; commit and logits remain on the host."""

from __future__ import annotations

from functools import partial

import torch

from .sm103.k3.attn_res.prepared import join_aux
from .sm103.k3.mla.verify_overlap import VerifyMlaIndexOverlap
from .verify_binding import VerifyStepView
from .verify_contract import verify_keys

# The process's live executors: the buffer owners the draft chain's build checks its mailboxes against
# (draft/serving.py, chain.DraftComm.assert_disjoint).
INSTANCES: list = []


class K31VerifyExecutor:
    def __init__(
        self,
        resources,
        indexers,
        capture_layers,
        *,
        query_width=8,
        request_batches=(1,),
    ):
        self.compile_keys = tuple(
            (b, q) for b, q, _ in verify_keys(query_width, request_batches)
        )
        self.resources, self.indexers = resources, indexers
        self.mla_index_overlap = (
            VerifyMlaIndexOverlap(resources.scratch["hidden.0"].device)
            if query_width == 8
            else None
        )
        self.capture_layers = tuple(capture_layers)
        self.layers = tuple(
            (int(l.layer_idx), int(l._is_moe_layer)) for l in resources.layers
        )
        self.capture_consumers = {}
        self.kda_cores, self.kda_pools = {}, {}
        for layer_id in self.capture_layers:
            consumer = (
                resources.final if layer_id == 92 else resources.before[layer_id + 1]
            )
            consumer.bind_aux()
            self.capture_consumers[layer_id] = consumer
        # Q rows use generic routing/dispatch even at c1. SmallBatchChain consumes
        # a B1-specific producer mailbox and cannot represent one Q-token request.
        for _, feed in resources.feed_forward.items():
            if hasattr(feed, "small_batch"):
                feed.small_batch = None
        INSTANCES.append(self)

    def _aux(self, view, layer_id):
        if layer_id not in self.capture_layers:
            return None
        index = self.capture_layers.index(layer_id)
        return view.aux_hidden[:, index * 7168 : (index + 1) * 7168]

    def run(self, embeddings: torch.Tensor, view: VerifyStepView):
        view.validate(auxiliary_width=7168 * len(self.capture_layers))
        r, q = view.row_count, view.query_width
        res = self.resources
        if embeddings.dtype != torch.bfloat16 or embeddings.shape != (r, 7168):
            raise ValueError("K31 verify requires TP-reduced BF16 embeddings [R,7168]")
        if (view.batch_requests, q) not in self.compile_keys:
            raise ValueError("K31 verify requires a prepared request/query key")
        prefix, blocks = res.scratch["prefix"][:r], res.scratch["blocks"][:r]
        first, second = res.scratch["hidden.0"][:r], res.scratch["hidden.1"][:r]
        output = res.scratch["hidden.out"][:r]
        mla_p, kda_p = (
            res.scratch["mla.projected"][:r],
            res.scratch["kda.projected"][:r],
        )
        local = res.scratch["attention.local"][:r]
        # Sparse writers own their streams; only short Q8 uses this fork/join.
        overlap = self.mla_index_overlap if not getattr(view, "sparse", 0) else None
        hidden = embeddings
        ready = False
        for layer_id, _ in self.layers:
            if not ready:
                res.before[layer_id].run(
                    hidden, prefix, blocks, first, aux=self._aux(view, layer_id - 1)
                )
            attn = res.attention[layer_id]
            projected = mla_p if attn.kind == "mla" else kda_p
            if attn.kind == "mla":
                if overlap is None:
                    self.indexers[layer_id].run(first, view.write_slots)
                else:
                    overlap.run(self.indexers[layer_id], first, view.write_slots)
            attn.projection.run(first, projected)
            if attn.kind == "mla":
                attn.core.run(
                    projected,
                    local,
                    cache=attn.cache,
                    slots=view.write_slots,
                    block_table=view.block_table,
                    lengths=view.sequence_lengths,
                    q=q,
                )
            else:
                core = self.kda_cores.get(layer_id)
                if core is None:
                    raise RuntimeError("front KDA verify provider is not integrated")
                core.run(
                    projected, local, pools=self.kda_pools[layer_id], step=view.kda
                )
            join_aux(torch.cuda.current_stream())  # the o-proj rewrites the prefix row
            attn.output.run(local, prefix, blocks, second)
            if attn.kind == "mla" and overlap is not None:
                # feed.run may overwrite first, including through fused consumers.
                overlap.join()
            consumer = res.fused_before.get(layer_id)
            ready = consumer is not None
            feed = res.feed_forward[layer_id]
            if ready:
                target = output if consumer is res.final else first
                consume = partial(
                    consumer.run_ring, prefix, blocks, aux=self._aux(view, layer_id)
                )
                hidden = feed.run(second, target, consume=consume)
            else:
                hidden = feed.run(second, first)
                first, second = second, first
        if not ready:
            res.final.run(hidden, prefix, blocks, output, aux=self._aux(view, 92))
        # The draft reads aux_hidden after the step.
        join_aux(torch.cuda.current_stream())
        return output, view.aux_hidden
