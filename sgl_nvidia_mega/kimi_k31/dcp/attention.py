# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 MLA/DSA layer under decode context parallelism (DCP = TP; TP16/DCP16 or TP8/DCP8), ``K31MK_DCP=1``.

Stock layout and method (stock SGLang ``dsa_dcp_sharded.py`` / ``dsa_backend.py`` /
``forward_mla.py``; see dcp/layout.py): MLA KV and index-K sharded by token (p on rank p % N at p // N), widened
page tables, per-rank local top-k candidates (``topk_transform_candidates_v2``, stable ties) exchanged and scattered
at their global positions, one global ``topk_transform_paged_v2`` (stable ties) over the widened table, the owned
slots compacted (``_localize_compact_kernel``), trtllm-gen sparse MLA of all N*H heads (padded to 128) over the local
slots with an LSE, then per head: W_UV of each source partial (stock ``dcp_v_preproject``), LSE combine (stock
``dcp_push_combine``), output gate.

One layer, ``rows`` query rows (decode: one per request; verify: one per draft token, causal lengths):

  MlaQuery (H local heads; KV + folded index-K written only at this rank's own slots) -> qpush (q records to every
  peer; -inf fill of the global score rows, or the all-visible local list in dense graphs) -> [sparse graphs]
  FusedIndexQuery (replicated index q) -> DeepGEMM logits over the local shard -> local candidates -> cand (exchange
  + scatter; polls the q records into the 128-head query) -> [sparse] global top-k + localize -> trtllm-gen sparse
  MLA (128 heads, LSE) -> merge (partials to head owners, W_UV, LSE combine, gate) -> o_proj input.

Differences from stock (all within the accuracy rule: same method and precision, order of FP32 sums only):
- q: each rank sends its own heads' FP8 query (stock default under fi_a2a replicates q_b instead); the bytes are
  bitwise those of the replicated projection (same MlaQuery per head).
- W_UV: the owner projects all N source partials of its heads (M = 16 MMA rows) instead of every rank projecting
  all heads before the exchange; one BF16 rounding of each projected partial either way.
"""

from __future__ import annotations

import os

import torch

from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import stream, to_cute

from .comm import BOTH, POLL, PUSH, DcpComm
from .layout import PAGE, TOPK, padded_heads
from sgl_nvidia_mega.kimi_k31.k3.cutedsl.mla.head_gate import head_gate

DCP_ENABLED = os.environ.get("K31MK_DCP", "0") == "1"


def _workspace(device, heads):
    from flashinfer.utils import get_trtllm_gen_multi_ctas_kv_counter_bytes

    from sgl_nvidia_mega.kimi_k31.k3.compat import TRTLLM_KV_COUNTER_ROWS

    sms = torch.cuda.get_device_properties(device).multi_processor_count
    size = get_trtllm_gen_multi_ctas_kv_counter_bytes(TRTLLM_KV_COUNTER_ROWS, heads, sms)
    return (
        torch.zeros(128 << 20, dtype=torch.int8, device=device),
        torch.zeros(size, dtype=torch.uint8, device=device),
    )


class _Profile:
    """Fixed buffers of one row capacity (graph width); no step-time allocation."""

    def __init__(self, device, rows, heads_padded, dense):
        options = dict(device=device)
        self.rows = rows
        # Pad heads (>= N*H) are never written: their FP8 zeros give zero partials nobody reads.
        self.query = torch.zeros((rows, 1, heads_padded, 576), dtype=torch.float8_e4m3fn, **options)
        self.part = torch.empty((rows, 1, heads_padded, 512), dtype=torch.bfloat16, **options)
        self.lse = torch.empty((rows, heads_padded), dtype=torch.float32, **options)
        self.candidates = torch.empty((rows, TOPK, 2), dtype=torch.int32, **options)
        self.slots = torch.empty((rows, TOPK), dtype=torch.int32, **options)
        self.compact = torch.empty((rows, TOPK), dtype=torch.int32, **options)
        self.counts = torch.empty((rows,), dtype=torch.int32, **options)
        self.dense = dense[:rows]
        self.row_tables = {}

    def row_table(self, table, q):
        """Stock top-k's per-row page table: row r reads request r // q (decode q = 1: the table itself)."""
        if q == 1:
            return table
        b, pages = table.shape
        if b == 1:
            return table.expand(q, pages)  # row stride 0: no copy
        rows = self.row_tables.get(pages)
        if rows is None:
            rows = self.row_tables[pages] = table.new_empty((b * q, pages))
        rows.view(b, q, pages).copy_(table.unsqueeze(1).expand(b, q, pages))
        return rows


class PreparedDcpSparseMla:
    """Wraps a prepared K3.1 index MLA (``base``: PreparedSparseIndexMla built with dcp=None at H local heads, which
    owns the MlaQuery kernels, the folded index-K writer and the index projection) and replaces its run() under
    DCP. ``ordinal``: this layer's MLA ordinal (0..23): mailbox parity and tag."""

    def __init__(self, base, comm: DcpComm, ordinal: int, *, max_context: int, dense: torch.Tensor, index_query=None):
        """``index_query(projected, rows) -> (q8 [rows, 32, 128], gates [rows, 32, 1])``: defaults to the base's
        decode index chain (FusedIndexQuery up to 8 rows); the verify build passes the stock 8-row chain."""
        self.base, self.comm, self.ordinal = base, comm, ordinal
        self.index_query = index_query or self._index_query
        self.device, self.heads = base.device, base.heads
        if comm.heads != self.heads:
            raise ValueError(f"DCP comm serves {comm.heads} heads per rank, the layer has {self.heads}")
        self.dcp, self.rank = comm.dcp, comm.rank
        self.span = PAGE * self.dcp
        self.heads_padded = padded_heads(self.dcp, self.heads)
        self.max_context = int(max_context)
        self.workspace, self.counter = _workspace(self.device, self.heads_padded)
        self.dense, self.profiles = dense, {}
        self.sparse, self.step = False, None

    def __getattr__(self, name):
        return getattr(self.base, name)

    # The executor's per-step hooks (SparseDecodeExecutor.before_attention).
    def set_step(self, sparse, step):
        self.sparse, self.step = bool(sparse), step
        hook = getattr(self.base, "set_step", None)  # PreparedSparseIndexMla (decode); PreparedMla (verify) has none
        if hook is not None:
            hook(sparse, step)

    def prepare_gate(self, hidden):
        hook = getattr(self.base, "prepare_gate", None)
        if hook is not None:
            hook(hidden)

    def prepare_cache(self, cache):
        """Compile the H-head query kernels (no dense-core warm call: DCP layers never run the dense MLA core), then
        one buffer profile per prepared (rows, q)."""
        base = self.base
        stock_core = base.stock_core
        base.stock_core = lambda *args, **kwargs: None  # the dense core is not used under DCP
        try:
            type(base).prepare_cache(base, cache)
        finally:
            base.stock_core = stock_core
        for rows, q in base.kernels:
            if rows > self.comm.cap:
                raise ValueError(f"DCP mailboxes hold {self.comm.cap} rows, the layer prepares {rows}")
            self.profiles[rows, q] = _Profile(self.device, rows, self.heads_padded, self.dense)
        return self

    # ------------------------------------------------------------------------------------------- the layer
    def run(self, projected, out, *, cache, slots, block_table, lengths, totals=None, q=1):
        """``slots``: this rank's KV rows of the written tokens (-1: another rank owns them; DcpComm.prologue);
        ``block_table``: [requests, pages] widened page ids; the step (``set_step``) carries the DCP lengths,
        schedule and plan. ``lengths`` / ``totals`` are unused (the step's global lengths are per row)."""
        for _ in self.stages(projected, out, cache=cache, slots=slots, block_table=block_table, q=q):
            pass
        return out

    def stages(self, projected, out, *, cache, slots, block_table, q=1, split=False):
        """The layer as a generator. ``split`` (single-GPU multi-rank tests): every exchange launches its push half,
        yields so the test can run the other ranks' pushes, then launches its poll half. Serving (``split`` False)
        launches each exchange once (push and poll in one kernel) and never yields."""
        step, base, comm = self.step, self.base, self.comm
        if step is None:
            raise RuntimeError("DCP MLA layer needs its step (local/global lengths, schedule, top-k plan)")
        rows = projected.shape[0]
        prof = self.profiles[rows, q]
        glens = step.global_lengths[:rows]
        table = block_table
        push, poll = ((PUSH,), (POLL,)) if split else ((BOTH,), ())
        # 1. Local query rows [rows, H, 576] + this rank's KV / index-K writes (MlaQuery, IndexMlaQuery.finish).
        mq, _value, query, _latent = base.kernels[rows, q]
        qnorm, kvnorm, qb, uk, uv, qinv, kinv = base.views
        args = (to_cute(projected[:, :2112]), qnorm, kvnorm, qb, uk)
        args += (to_cute(cache.view(torch.uint8)), to_cute(slots[:rows], 8))
        args += (qinv, kinv, base.eps, query)
        mq(*args, *base.query_views, stream(self.device))
        q_local = base.query_scratch[:rows]
        # 2. Push this rank's query heads; prepare the global score rows (sparse) or the all-visible list (dense).
        if self.sparse:
            comm.qpush(self.ordinal, q_local, glens, dense=prof.dense, phases=push[0])
        else:
            comm.qpush(
                self.ordinal, q_local, glens, visible=(prof.compact, prof.counts), table=table,
                rows_per_request=q, phases=push[0],
            )
        candidates = None
        if self.sparse:
            # 3. Replicated index query, logits over the local index-K shard, local candidates (stock method).
            iq, gates = self.index_query(projected, rows)
            candidates = self._local_candidates(iq, gates, table, step, rows, q, prof)
        # 4. Exchange: candidates scattered at their global positions; every source's query heads polled in.
        q_out = prof.query.view(rows, self.heads_padded, 576)
        for phases in push + poll:
            comm.cand(self.ordinal, candidates, glens, prof.dense, q_local, q_out, phases=phases)
            if split and phases == PUSH:
                yield "cand"
        if self.sparse:
            # 5. Global top-k over the widened table, then this rank's slots compacted (stock kernels).
            from sglang.kernels.ops.attention.dsv4.topk import topk_transform_paged_v2

            topk_transform_paged_v2(
                prof.dense, glens, prof.row_table(table, q), prof.slots, self.span, step.topk_plan,
                deterministic=True,
            )
            self._localize(prof)
        # 6. trtllm-gen sparse MLA of all heads over this rank's selected keys, with the LSE (log2).
        self._sparse_core(cache, prof)
        # 7. Partials to head owners; W_UV, LSE combine and gate of this rank's heads.
        gate = head_gate(projected, self.heads)
        for phases in push + poll:
            comm.merge(
                self.ordinal, prof.part.view(rows, self.heads_padded, 512), prof.lse, prof.counts, glens,
                base.weights[4], gate, out, phases=phases,
            )
            if split and phases == PUSH:
                yield "merge"

    def _index_query(self, projected, rows):
        base = self.base
        index_projection = base.sparse_rows[rows][0]
        if rows in base.fused_rows:
            return index_projection.run(base.hidden, projected)  # FusedIndexQuery: no MLA-query side copy
        return index_projection.project_query(projected)

    def _local_candidates(self, iq, gates, table, step, rows, q, prof):
        import deep_gemm
        from sglang.kernels.ops.attention.dsa.paged_mqa_logits import _restore_row_stride
        from sglang.kernels.ops.attention.dsv4.topk import topk_transform_candidates_v2

        local = step.local_lengths  # int32 [requests, q]: keys of each row on this rank
        requests = rows // q
        scores = deep_gemm.fp8_paged_mqa_logits(
            iq.view(requests, q, 32, 128),
            self.base.index_cache.view(-1, 64, 1, 132),
            gates.squeeze(-1),
            local,
            table,
            step.schedule,
            table.shape[1] * 64,
            clean_logits=False,
        )
        scores = _restore_row_stride(scores)
        topk_transform_candidates_v2(
            scores,
            local.reshape(-1),
            out_candidates=prof.candidates,
            dcp_size=self.dcp,
            dcp_rank=self.rank,
            deterministic=True,
        )
        return prof.candidates

    def _localize(self, prof):
        from sglang.srt.layers.attention.dsa.dsa_dcp_sharded import _localize_compact_kernel

        _localize_compact_kernel[(prof.rows,)](
            prof.slots, prof.slots.stride(0), prof.compact, prof.compact.stride(0), prof.counts,
            self.dcp, self.rank, TOPK=TOPK,
        )

    def _sparse_core(self, cache, prof):
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

        softmax, output_scale = self.base.stock_scales
        trtllm_batch_decode_with_kv_cache_mla(
            query=prof.query,
            kv_cache=cache.view(torch.float8_e4m3fn).unsqueeze(1),
            workspace_buffer=self.workspace,
            qk_nope_head_dim=128,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            block_tables=prof.compact.unsqueeze(1),
            seq_lens=prof.counts,
            max_seq_len=self.max_context,
            sparse_mla_top_k=TOPK,
            out=prof.part,
            return_lse=True,
            lse=prof.lse,
            bmm1_scale=softmax,
            bmm2_scale=output_scale,
            enable_pdl=True,
            backend="trtllm-gen",
            multi_ctas_kv_counter_buffer=self.counter,
        )


def dense_scores(rows: int, max_context: int, dcp: int, device) -> torch.Tensor:
    """The global score rows shared by every DCP layer (consumed before the next layer writes them)."""
    span = PAGE * dcp
    width = -(-max(max_context, 1) // span) * span
    return torch.empty((rows, width), dtype=torch.float32, device=device)


def phases_default() -> int:
    return BOTH
