# SPDX-License-Identifier: Apache-2.0
"""Context-split DSA verify indexer, local side.

Each of the ``world`` TP ranks scores the logical pages ``16 j + rank`` of every request (page-strided
over the replicated index-K cache) and takes its exact local top-2048 as merge candidates: (FP32 score
bits, global logical position), ``(0, -1)`` padded, under the merge's order (score desc, position asc).
The 16-rank merge turns them into the KV slots the sparse core reads.

A row with causal length ``c`` owns a prefix of its local positions: every owned page below ``c // 64``
is full, so ``L(c) = 64 * full + (c % 64 if page 16 * full + rank == c // 64 else 0)``.
"""

from functools import cache

import torch

# Names read from SGLang. A contract test checks these before GPU use.
SGLANG_READS = {
    "kernels/ops/attention/dsa/paged_mqa_logits.py": ("_restore_row_stride",),
    "kernels/ops/attention/dsv4/topk.py": ("topk_transform_paged_candidates_v2",),
    "srt/layers/attention/dsa/dsa_dcp_sharded.py": ("DcpTopKMerge",),
}

PAGE, PAGE_BITS, TOPK = 64, 6, 2048
# Context guard: split a forward only when its longest causal length reaches this; below it the replicated
# path is faster (the split's local logits + top-k + merge are a fixed cost). Measured crossover: 16-rank merge
# unit B1 on GB300: +5.8 us/layer at 134K, ~0 at 338K, -13 us/layer at 600K; prepare-early moved it to 320000.
SPLIT_MIN = 320000
# Rows longer than this run the clustered candidate top-k (16384 beats 8192 at 8.4K local).
CLUSTER_FLOOR = 16384
# The clustered top-k's coarse bin width. At 10 bits ~5 % of real AgentX local rows overflow the 2048-entry tie
# buffer and are redone exact on one block; 12 bits narrow the bin 4x (exact either way; 62.5K local rows that
# overflowed ran 8 us instead of 23, non-overflowing rows +0.3 us; AgentX overflow 5.0 -> 0.9 %).
COARSE_BITS = 12


def overflow_rows(device):
    """int32 [1]: local rows whose clustered top-k overflowed its tie buffer and was redone exact, this run."""
    index = (
        torch.device("cuda", device).index if isinstance(device, int) else device.index
    )
    return _overflow_rows(torch.cuda.current_device() if index is None else index)


@cache
def _overflow_rows(index):
    return torch.zeros(1, dtype=torch.int32, device=torch.device("cuda", index))


def local_length(lengths, rank, world):
    """Owned local prefix of causal ``lengths`` (int32) on ``rank``: 64 per full stripe of ``world`` pages plus the
    rank's share of the last stripe, the page formula above in 6 elementwise launches."""
    stripe = PAGE * world
    tail = torch.clamp(lengths % stripe - PAGE * rank, min=0, max=PAGE)
    return (lengths // stripe * PAGE + tail).to(torch.int32)


class LocalIndexer:
    """One per device, shared by every DSA layer of the verify graph. ``prepare`` runs once per forward (on the
    index stream at forward start, else the first layer's call); ``run`` per layer returns the [rows, 2048, 2]
    candidates."""

    def __init__(self, device, *, requests, query_width, max_pages, world, rank):
        self.world, self.rank, self.q = world, rank, query_width
        rows = requests * query_width
        self.local_pages = -(-max_pages // world)
        self.global_page = torch.arange(self.local_pages, device=device) * world + rank
        self.local_table = torch.zeros(
            requests, self.local_pages, dtype=torch.int32, device=device
        )
        # [local contexts; local causal lengths], computed in one pass
        self.local_both = torch.zeros(2, rows, dtype=torch.int32, device=device)
        self.local_contexts = self.local_both[0].view(requests, query_width)
        self.local_lengths = self.local_both[1]
        self.candidates = torch.empty(rows, TOPK, 2, dtype=torch.int32, device=device)
        self.overflow_rows = overflow_rows(device)
        self.schedule = None
        self.scores = None  # the last layer's local scores (tests)
        self.prepared_for = None

    def prepare(self, table, step):
        """The forward's local subtable, local context / causal lengths and the DeepGEMM schedule over them.
        Pages past the table repeat its last column; they lie beyond every local length, never read."""
        import deep_gemm

        torch.index_select(
            table,
            1,
            self.global_page.clamp(max=table.shape[1] - 1),
            out=self.local_table,
        )
        both = torch.stack((step.score_contexts.reshape(-1), step.causal_lengths))
        self.local_both.copy_(local_length(both, self.rank, self.world))
        # the SM count fp8_paged_mqa_logits runs on, as stock's resolve_num_sms
        self.schedule = deep_gemm.get_paged_mqa_logits_metadata(
            self.local_contexts, PAGE, deep_gemm.get_num_sms()
        )
        self.prepared_for = step

    def run(self, query, gates, cache, table, step):
        import deep_gemm
        from sglang.kernels.ops.attention.dsa.paged_mqa_logits import (
            _restore_row_stride,
        )
        from sglang.kernels.ops.attention.dsv4.topk import (
            topk_transform_paged_candidates_v2,
        )

        if self.prepared_for is not step:
            self.prepare(table, step)
        scores = deep_gemm.fp8_paged_mqa_logits(
            query.view(step.batch_requests, step.query_width, 32, 128),
            cache.view(-1, PAGE, 1, 132),
            gates.squeeze(-1),
            self.local_contexts,
            self.local_table,
            self.schedule,
            self.local_pages * PAGE,
            clean_logits=False,
        )
        self.scores = _restore_row_stride(scores)
        topk_transform_paged_candidates_v2(
            self.scores,
            self.local_lengths,
            out_candidates=self.candidates,
            world=self.world,
            rank=self.rank,
            deterministic=True,
            cluster_floor=CLUSTER_FLOOR,
            page_bits=PAGE_BITS,
            overflow_rows=self.overflow_rows,
            coarse_bits=COARSE_BITS,
        )
        return self.candidates


class SplitMerge:
    """The 16-rank DCP top-k merge in split mode, shared by the layers: from every rank's candidates, each rank
    writes the row's whole global top-2048 as KV slots (stable ties: score desc, position asc)."""

    def __init__(self, group, max_rows):
        from sglang.srt.layers.attention.dsa.dsa_dcp_sharded import DcpTopKMerge

        self.merge = DcpTopKMerge.create(group, max_rows, TOPK, group.rank_in_group)
        # Match the allocation factory without importing its private cand_k.
        self.cand_k = (
            (self.merge.buffer.numel() // (2 * max_rows) - TOPK) // group.world_size
            if self.merge is not None
            else 0
        )

    def usable(self, rows):
        return self.merge is not None and self.merge.usable(rows)

    def run(self, candidates, causal_lengths, row_table, out):
        """``row_table``: [rows, pages] logical page -> KV page of each row (row stride 0 allowed)."""
        from ..sm103.dsa.split_merge import dcp_topk_merge

        if self.merge is None:
            raise RuntimeError("Context-split merge is unavailable")
        merge = self.merge
        dcp_topk_merge(
            candidates,
            causal_lengths,
            row_table,
            buffer=merge.buffer,
            multicast_ptr=merge.multicast_ptr,
            peer_ptrs=merge.peer_ptrs,
            counters=merge.counters,
            union_scratch=merge.union_scratch,
            out_slots=out,
            out_fallback=merge.fallback[: out.shape[0]],
            rank=merge.dcp_rank,
            span=PAGE,
            cand_k=self.cand_k,
            split=True,
        )
        return out
