# SPDX-License-Identifier: Apache-2.0
"""Causal sparse DSA extension of the tensor-only TARGET_VERIFY entry.

The existing executor, KDA rings, MoE, projection and aux consumers are reused.
Index-K is written for all query rows before scoring. Native paged MQA matches
stock's [1,Q,H,D] score layout; per-row causal lengths mask top-k selection.
"""

import logging
from dataclasses import dataclass
from functools import cache

import cutlass.cute as cute
import sglang.srt.layers.attention.dsa.dsa_dcp_sharded as DCP_SHARDED
import torch
from cutlass.cute.runtime import make_fake_stream
from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
    is_one_nvlink_clique,
)
from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import (
    all_gather_inner,
    create_state,
)
from sglang.srt.distributed.parallel_state import in_the_same_node_as
from sglang.srt.runtime_context import get_parallel

from sgl_nvidia_mega.kimi_k31.sm103.k3.common.host import fake, layout, stream, to_cute
from sgl_nvidia_mega.kimi_k31.sm103.k3.mla.value import MlaValue

from ..verify_binding import VerifyStepView
from ..verify_contract import served_request_batches, verify_keys
from ..verify_spec import BuiltK31Verify
from .indexer import IndexProjection, IndexWeights
from .mla import _CP_MAX_ROWS, CpSparseMla, SparseMla
from .scratch import RowScratch, row_buffer

# Names read from SGLang. A contract test checks these before GPU use.
SGLANG_READS = {
    "srt/distributed/device_communicators/custom_all_reduce_utils.py": (
        "is_one_nvlink_clique",
    ),
    "srt/distributed/device_communicators/triton_symm_mem_ag.py": (
        "all_gather_inner",
        "create_state",
    ),
    "srt/distributed/parallel_state.py": (
        "in_the_same_node_as",
        "get_tp_group",
    ),
    "srt/runtime_context.py": ("get_parallel",),
    "kernels/ops/attention/dsa/paged_mqa_logits.py": ("_restore_row_stride",),
    "kernels/ops/attention/dsv4/topk.py": ("topk_transform_paged_v2",),
    "srt/layers/attention/dsa/dsa_dcp_sharded.py": (
        "dcp_global_topk",
        "dcp_localize_compact",
        "_TOPK_MERGE",
    ),
    "srt/layers/dcp/comm.py": ("dcp_a2a_lse_reduce",),
    "kernels/ops/kvcache/dsa_fused_store_index_cache.py": (
        "fused_store_index_k_cache",
    ),
}

logger = logging.getLogger(__name__)


@cache
def _index_stream(device):
    return torch.cuda.Stream(device=device)


@cache
def _gate_stream(device):
    return torch.cuda.Stream(device=device)


@cache
def _direct_query_state(group, device):
    """One stable buffer of the DCP core's query rows (CpSparseMla: up to _CP_MAX_ROWS rows of
    128 heads) for serial DCP sparse layers and graph profiles, or None (the query is
    all-gathered on the group) where the group is not one NVLink clique with multicast.

    The entry barrier stays enabled on every gather: a rank cannot overwrite the
    previous layer's query until all ranks have finished consuming it on main.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "the direct DCP query gather must initialize before graph capture"
        )
    if not all(in_the_same_node_as(group.cpu_group)) and not is_one_nvlink_clique(
        group.cpu_group, group.device
    ):
        logger.info(
            "NVIDIA megakernel: the DCP group is not one NVLink clique; the query is "
            "all-gathered on the group"
        )
        return None
    state = create_state(
        group.device_group,
        group.rank_in_group,
        _CP_MAX_ROWS,
        128 * 576 // 2,
        device=device,
    )
    # Raw FP8 bytes are transported as bf16-sized words, without conversion. The
    # padding heads past DCP x heads must stay zero, including on the first replay.
    state.comm_buff.zero_()
    torch.cuda.synchronize(device)
    # Also the barrier: no rank writes into a peer's buffer before the peer zeroed it. The
    # ranks agree on multicast, or they would gather on different paths.
    multicast = torch.tensor([int(state.symm_mem_hdl.multicast_ptr != 0)])
    torch.distributed.all_reduce(
        multicast, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group
    )
    if not multicast.item():
        logger.info(
            "NVIDIA megakernel: the DCP group has no multicast; the query is all-gathered "
            "on the group"
        )
        return None
    return state


@cache
def _padded_value(rows, heads, pitch, weight_layout, device, q, latent_heads=16):
    """Read the real heads directly from the sparse core's ``latent_heads``-head output
    (16: the padded DCP1 core; ``heads``: the merged DCP rows). MlaValue releases its PDL
    dependent (the o-proj) at entry, so the o-proj's whole weight preload overlaps the core
    (bitwise: the o-proj still waits for MlaValue's grid)."""
    latent = fake((rows, heads, 512), (latent_heads * 512, 512, 1))
    weight = fake(*weight_layout)
    gate = fake((rows, heads * 128), (pitch, 1))
    lengths = fake((rows // q,), (1,), torch.int32, 4)
    output = fake((rows, heads * 128))
    with torch.accelerator.device_index(device):
        return cute.compile(
            MlaValue(heads, rows, q, early_trigger=True),
            latent,
            weight,
            gate,
            lengths,
            output,
            make_fake_stream(),
        )


@dataclass(frozen=True)
class SparseVerifyStep(VerifyStepView):
    causal_lengths: torch.Tensor = None
    score_contexts: torch.Tensor = None
    score_schedule: torch.Tensor = None
    topk_plan: torch.Tensor = None
    sparse: int = 0
    # Decode context parallel (index-K and latent KV token-strided: global position p on
    # rank p % dcp_size at local p // dcp_size). ``local_contexts``: int32[Breq, Q] keys
    # this rank holds per row (the native score's contexts); ``local_slots``: int64[rows]
    # this rank's latent slot per written token, -1 where another rank owns it.
    dcp_size: int = 1
    dcp_rank: int = 0
    local_contexts: torch.Tensor = None
    local_slots: torch.Tensor = None
    # Context-split indexer for this forward: set by the host from the longest causal length
    # (dsa.split.SPLIT_MIN), uniform across ranks, a separate graph variant.
    split: int = 0

    def validate(self, *, auxiliary_width=None):
        super().validate(auxiliary_width=auxiliary_width)
        keys = verify_keys(self.query_width, served_request_batches(self.query_width))
        if (self.batch_requests, self.query_width, self.row_count) not in keys:
            raise ValueError("sparse verify requires a supported request/query key")
        if self.sparse:
            if self.causal_lengths.shape != (self.row_count,):
                raise ValueError("sparse verify needs one causal length per query")
            if self.score_contexts.shape != (self.batch_requests, self.query_width):
                raise ValueError(
                    "sparse verify native score needs int32[Breq,Q] contexts"
                )
            if (
                self.causal_lengths.dtype != torch.int32
                or self.score_contexts.dtype != torch.int32
            ):
                raise ValueError("sparse verify causal lengths/contexts must be int32")
            if self.score_schedule is None or self.topk_plan is None:
                raise ValueError("sparse verify requires current schedule/top-k plan")
            if self.dcp_size > 1:
                if (
                    self.local_contexts is None
                    or self.local_contexts.shape
                    != (self.batch_requests, self.query_width)
                    or self.local_contexts.dtype != torch.int32
                ):
                    raise ValueError(
                        "DCP sparse verify needs int32[Breq,Q] local contexts"
                    )
                if self.local_slots is None or self.local_slots.shape != (
                    self.row_count,
                ):
                    raise ValueError("DCP sparse verify needs one local slot per row")
                if self.split:
                    raise ValueError(
                        "the context-split indexer needs the replicated index-K cache (no DCP)"
                    )
        elif self.split:
            raise ValueError("the context-split indexer runs only in sparse verify")


# DCP sparse verify whose top-k merge is the custom one-shot kernel (dsa_dcp_sharded._TOPK_MERGE usable for the
# rows): the query all-gather and its copies run on the main stream right after the MLA query, overlapping the
# indexer chain on the index stream, instead of on the index stream after the selection. Every rank orders its DCP
# collectives the same way: the selection's cross-rank merge waits for the gather's event. Otherwise (no usable
# merge: a NCCL candidate gather on the index stream) the gather follows the selection.
_EARLY_QAG_LOGGED = []
_DIRECT_QAG_LOGGED = []


def dcp_layout():
    """(DCP size, rank) of the attention group: the index-K writers keep a token only on
    its owning rank (slot % size == rank, at row slot // size)."""
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    try:
        return parallel.attn_dcp_size, parallel.attn_dcp_rank
    except (
        RuntimeError,
        ValueError,
    ):  # no parallel config published (model-free tests): no DCP
        return 1, 0


def dcp_store_args():
    """The index-K store's trailing (dcp_size, dcp_rank) under DCP; none without it, so the
    DCP1 call stays the stock one."""
    size, rank = dcp_layout()
    return (size, rank) if size > 1 else ()


class VerifySelection:
    """Verify top-k: stock topk_transform_paged_v2 (the stock target-verify method, deterministic=False,
    over the stock per-row plan), so the MK matches stock, also on tie-heavy rows where topk_v2 is inexact."""

    def __init__(self, device, rows, split=None, *, scratch=None):
        self.rows = rows
        self.selected = row_buffer(
            scratch, "selection.selected", rows, (2048,), torch.int32, device
        )
        self.row_tables = {}
        # (dsa.split.LocalIndexer, 16-rank merge), shared by the layers; None keeps the replicated selection
        self.split = split

    def row_table(self, table, q):
        """Stock top-k's per-row page table: row r reads request r // q."""
        b, pages = table.shape
        if b == 1:
            return table.expand(q, pages)  # row stride 0: no copy
        rows = self.row_tables.get(pages)
        if rows is None:
            rows = self.row_tables[pages] = table.new_empty((b * q, pages))
        rows.view(b, q, pages).copy_(table.unsqueeze(1).expand(b, q, pages))
        return rows

    def run(self, query, gates, cache, table, step, collective_after=None):
        """``collective_after``: DCP only, an event the cross-rank selection waits for after the scoring is enqueued
        (the early query all-gather on main), so every rank issues its DCP collectives in the same order."""
        import deep_gemm
        from sglang.kernels.ops.attention.dsa.paged_mqa_logits import (
            _restore_row_stride,
        )

        if self.split is not None and step.split:
            local, merge = self.split
            candidates = local.run(query, gates, cache, table, step)
            return merge.run(
                candidates,
                step.causal_lengths,
                self.row_table(table, step.query_width),
                self.selected,
            )

        contexts = step.score_contexts if step.dcp_size == 1 else step.local_contexts
        score_fn = deep_gemm.fp8_paged_mqa_logits
        scores = score_fn(
            query.view(step.batch_requests, step.query_width, 32, 128),
            cache.view(-1, 64, 1, 132),
            gates.squeeze(-1),
            contexts,
            table,
            step.score_schedule,
            table.shape[1] * 64,
            clean_logits=False,
        )
        scores = _restore_row_stride(scores)
        if step.dcp_size > 1:
            if collective_after is not None:
                torch.cuda.current_stream(scores.device).wait_event(collective_after)
            return self.select_dcp(scores, table, step)
        # Native scoring computes through the last row's context (same as stock).
        # Top-k masks each row to P+j+1; no future key can become a selected slot.
        from sglang.kernels.ops.attention.dsv4.topk import topk_transform_paged_v2

        topk_transform_paged_v2(
            scores,
            step.causal_lengths,
            self.row_table(table, step.query_width),
            self.selected,
            64,
            step.topk_plan,
            deterministic=False,
        )
        return self.selected

    def select_dcp(self, scores, table, step):
        """Stock DCP selection: each rank's top-2048 candidates over its own keys, merged
        across the group to the global top-2048 (``deterministic=True``, as stock DCP), then
        this rank's slots compacted to the front: (local rows [rows, 2048], counts).
        The host owns the collective and its workspace."""
        from sglang.srt.layers.attention.dsa.dsa_dcp_sharded import (
            dcp_global_topk,
            dcp_localize_compact,
        )

        slots = dcp_global_topk(
            scores,
            step.local_contexts.view(-1),
            step.causal_lengths,
            self.row_table(table, step.query_width),
            2048,
            64,
            step.topk_plan,
        )
        return dcp_localize_compact(slots)


class PreparedSparseVerifyMla:
    def __init__(self, base, weights, rows, max_context, split=None, *, scratch=None):
        self.base = base
        # Up to 8 rows the index query is decode's one-kernel FusedIndexQuery (same rounding points, head gate from
        # the layer input); it is built for 1-8 rows (sm103/dsa/index_q.py), wider steps run the stock chain.
        self.fused_index = rows <= 8
        if self.fused_index:
            from ..sm103.dsa.index_q import FusedIndexQuery

            self.index_projection = FusedIndexQuery(weights, rows)
        else:
            self.index_projection = IndexProjection(weights, rows, scratch=scratch)
        self.hidden = (
            None  # the layer input; set by the index-K writer for the fused query
        )
        self.selection = VerifySelection(
            base.device, rows, split=split, scratch=scratch
        )
        self.mla = SparseMla(
            base.device, max_context, rows, heads=base.heads, scratch=scratch
        )
        self.index_stream = _index_stream(base.device)
        self.gate_stream = _gate_stream(base.device)
        self.packed_index_gate = False
        self._q_gathered = (
            torch.cuda.Event()
        )  # the early query all-gather (and its copies) done
        self.sparse_values = {}
        self.index_cache = None
        self.step = None
        # DCP: every rank's core runs all TP x heads over the keys it owns, then the rows
        # LSE-merge back to their head owners (built when the attention group is DCP).
        self.dcp_size = dcp_layout()[0]
        self.cp_mla = None
        self.direct_qag = None
        if self.dcp_size > 1:
            self.cp_mla = CpSparseMla(
                base.device,
                max_context,
                rows,
                heads=self.dcp_size * base.heads,
                scratch=scratch,
            )
            # Each rank multicasts its FP8 heads straight into the core's query rows; CpSparseMla
            # bounds the heads and rows to the buffer's.
            self.direct_qag = _direct_query_state(get_parallel().dcp_group, base.device)
            if self.direct_qag is not None:
                # Bind the full padded buffer before capture, not the 96-head
                # slice returned by all_gather_inner. No copy or rebind at run time.
                self.cp_mla.query = (
                    self.direct_qag.comm_buff[:rows]
                    .view(torch.float8_e4m3fn)
                    .view(
                        rows,
                        1,
                        128,
                        576,
                    )
                )
        self.cp_values = {}
        self.cp_latent = None

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def latent(self):
        if self.step is not None and self.step.sparse:
            if self.step.dcp_size > 1:
                return self.cp_latent
            return self.mla.output[:, 0, : self.base.heads]
        return self.base.latent

    def prepare_cache(self, cache, *, prepare_base=True):
        if prepare_base:
            self.base.prepare_cache(cache)
        for rows, q in self.base.kernels:
            if rows != self.selection.rows:
                continue
            self.sparse_values[rows, q] = _padded_value(
                rows,
                self.base.heads,
                self.base.pitches[rows],
                layout(self.base.weights[4]),
                self.base.device.index,
                q,
            )
            if self.cp_mla is not None:  # the merged rows hold this rank's heads only
                self.cp_values[rows, q] = _padded_value(
                    rows,
                    self.base.heads,
                    self.base.pitches[rows],
                    layout(self.base.weights[4]),
                    self.base.device.index,
                    q,
                    latent_heads=self.base.heads,
                )
        return self

    def early_gather(self, rows):
        """The selection's merge is the one-shot kernel for ``rows`` (no NCCL on the index stream that could run
        concurrently with a gather on main)."""
        merge = DCP_SHARDED._TOPK_MERGE
        if merge is None or not merge.usable(rows):
            return False
        if not _EARLY_QAG_LOGGED:  # use-time, once per process
            _EARLY_QAG_LOGGED.append(True)
            logger.info(
                "NVIDIA megakernel: early query gather engaged (DCP sparse verify gathers the query on the "
                "main stream after the MLA query; the selection merge waits for it)"
            )
        return True

    def gather_query(self, query, rows):
        """This rank's FP8 query rows [rows, heads x 576] all-gathered along the rows into
        the DCP core's [rows, DCP x heads, 576] (rank-major heads, the merge's order)."""
        heads = self.dcp_size * self.base.heads
        if self.direct_qag is not None:
            all_gather_inner(
                self.direct_qag,
                query.view(torch.bfloat16).view(rows, self.base.heads * 576 // 2),
                tp_hidden_dim=heads * 576 // 2,
                skip_entry_sync=False,
                safe=False,
            )
            if not _DIRECT_QAG_LOGGED:
                _DIRECT_QAG_LOGGED.append(True)
                logger.info(
                    "NVIDIA megakernel: direct query gather engaged (DCP%d FP8 query multicast "
                    "into the 128-head layout; entry and exit barriers retained)",
                    self.dcp_size,
                )
            return
        own = query.view(torch.uint8).view(rows, self.base.heads * 576)
        gathered = get_parallel().dcp_group.all_gather(own, dim=-1)
        self.cp_mla.query.view(torch.uint8)[:, 0, :heads].copy_(
            gathered.view(rows, heads, 576)
        )

    def merge_heads(self, out, lse):
        """Each head owner's rows from every rank's partials (stock a2a LSE reduce, base-2)."""
        from sglang.srt.layers.dcp.comm import dcp_a2a_lse_reduce
        from sglang.srt.runtime_context import get_parallel

        heads = self.dcp_size * self.base.heads
        parallel = get_parallel()
        # The server's --dcp-comm-backend, as stock MLA: fi_a2a's MNNVL workspace is
        # allocated before capture only when that backend is selected.
        return dcp_a2a_lse_reduce(
            out[:, 0, :heads],
            lse[:, :heads],
            parallel.dcp_group,
            is_lse_base_on_e=False,
            comm_backend=parallel.dcp_comm_backend,
        )

    def prepare_gate(self, hidden):
        if self.step is not None and self.step.sparse and not self.fused_index:
            self.index_projection.prepare_gate(hidden)

    def run(
        self, projected, out, *, cache, slots, block_table, lengths, totals=None, q=1
    ):
        step = self.step
        if step is None or not step.sparse:
            return self.base.run(
                projected,
                out,
                cache=cache,
                slots=slots,
                block_table=block_table,
                lengths=lengths,
                totals=totals,
                q=q,
            )
        b = self.base
        rows = projected.shape[0]
        mq, _, query, _ = b.kernels[rows, q]
        qnorm, kvnorm, qb, uk, uv, qinv, kinv = b.views
        dcp = step.dcp_size > 1
        if dcp and step.dcp_size != self.dcp_size:
            raise RuntimeError(
                f"DCP sparse verify built for {self.dcp_size} ranks, step has {step.dcp_size}"
            )
        # Under DCP the latent row of a written token lives on its owning rank only.
        write_slots = step.local_slots if dcp else slots
        args = (
            to_cute(projected[:, :2112]),
            qnorm,
            kvnorm,
            qb,
            uk,
            to_cute(cache.view(torch.uint8)),
            to_cute(write_slots, 8),
            qinv,
            kinv,
            b.eps,
            query,
        )
        s = stream(b.device)
        main = torch.cuda.current_stream(b.device)
        # The index-K/gate writer is already running on this stream. Wait only
        # for the input projection here; main MLA query is independent of it.
        self.index_stream.wait_stream(main)
        # The packed writer runs ahead of the index query on the index stream; otherwise the gate stream writes the gate.
        if not self.packed_index_gate:
            self.index_stream.wait_stream(self.gate_stream)
        mq(*args, *b.query_views, s)
        early = dcp and self.early_gather(rows)
        if early:  # main: right behind the MLA query it gathers; overlaps the indexer chain below
            self.gather_query(b.query_scratch[:rows], rows)
            self._q_gathered.record()
        with torch.cuda.stream(self.index_stream):
            if self.fused_index:
                # Not run_copy: copying the MLA query would order MlaQuery before the indexer.
                iq, gates = self.index_projection.run(self.hidden, projected)
            else:
                iq, gates = self.index_projection.project_query(projected)
            selected = self.selection.run(
                iq,
                gates,
                self.index_cache,
                block_table,
                step,
                collective_after=self._q_gathered if early else None,
            )
            if dcp and not early:
                # After the selection's candidate gather on this stream: one DCP-group
                # collective at a time (no concurrent NCCL fallbacks on the group).
                self.index_stream.wait_stream(main)
                self.gather_query(b.query_scratch[:rows], rows)
        if not dcp:
            # Copy before joining the indexer, so this independent work overlaps it.
            self.mla.prepare_query(b.query_scratch[:rows])
        main.wait_stream(self.index_stream)
        if dcp:
            local_rows, counts = selected
            partial, lse = self.cp_mla.run_core(
                cache, local_rows, counts, b.stock_scales
            )
            self.cp_latent = latent = self.merge_heads(partial, lse)
            value = self.cp_values[rows, q]
        else:
            self.mla.run_core(cache, selected, step.causal_lengths, b.stock_scales)
            latent = self.mla.output[:, 0, : b.heads]
            value = self.sparse_values[rows, q]
        value(
            to_cute(latent),
            uv,
            to_cute(projected[:, 2112 : 2112 + b.heads * 128]),
            to_cute(lengths, 4),
            to_cute(out),
            s,
        )
        return out


class SparseVerifyIndexKey:
    def __init__(self, base, core):
        self.base, self.core = base, core
        self.packed_weight = None
        if core.selection.rows == 8:
            # Preparation only. Both outputs retain the stock BF16 GEMM rounding.
            self.packed_weight = torch.cat(
                (base.weights.projection, core.index_projection.weights.gate), dim=0
            )
            self.projected = torch.empty(
                (8, 160), device=self.packed_weight.device, dtype=torch.bfloat16
            )
            if (
                not core.fused_index
            ):  # the fused query computes its gate from the layer input
                core.index_projection.gate = self.projected[:, 128:160]
            core.packed_index_gate = True

    def bind(self, cache, page_size, rows=(8,)):
        self.core.index_cache = cache
        return self.base.bind(cache, page_size, rows)

    def run(self, hidden, slots):
        if self.core.step is None or not self.core.step.sparse:
            return self.base.run(hidden, slots)
        side = self.core.index_stream
        gate = self.core.gate_stream
        main = torch.cuda.current_stream(hidden.device)
        side.wait_stream(main)
        # Read on the index stream after its wait on the projection; the executor keeps
        # ``hidden`` until the core has joined that stream.
        self.core.hidden = hidden
        if self.packed_weight is not None:
            from flashinfer.norm import layernorm
            from sglang.kernels.ops.kvcache.dsa_fused_store_index_cache import (
                fused_store_index_k_cache,
            )

            with torch.cuda.stream(side):
                torch.mm(hidden, self.packed_weight.T, out=self.projected)
                w = self.base.weights
                # LayerNorm supports this padded row stride; its output is
                # contiguous, so the stock cache writer needs no input copy.
                key = layernorm(self.projected[:, :128], w.norm, w.bias, w.eps)
                fused_store_index_k_cache(
                    key, self.base.cache, slots, 64, *dcp_store_args()
                )
            return
        gate.wait_stream(main)
        with torch.cuda.stream(side):
            self.base.run(hidden, slots)
        with torch.cuda.stream(gate):
            self.core.prepare_gate(hidden)


class MultiSparseVerifyMla:
    """One fixed scratch profile per row capacity; no step-time allocation."""

    def __init__(self, base, weights, widths, max_context, split=None):
        self.base = base
        split = split or {}
        # The widths replay one at a time: one set of widest-width buffers serves them all.
        scratch = RowScratch(base.device, max(widths))
        self.profiles = {
            rows: PreparedSparseVerifyMla(
                base, weights, rows, max_context, split.get(rows), scratch=scratch
            )
            for rows in widths
        }
        self.active = self.profiles[min(self.profiles)]

    def __getattr__(self, name):
        return getattr(self.active, name)

    @property
    def step(self):
        return self.active.step

    @step.setter
    def step(self, value):
        self.active = self.profiles[value.row_count]
        self.active.step = value

    def prepare_cache(self, cache):
        self.base.prepare_cache(cache)
        for profile in self.profiles.values():
            profile.prepare_cache(cache, prepare_base=False)
        return self

    def run(self, projected, *args, **kwargs):
        if projected.shape[0] != self.active.selection.rows:
            raise ValueError("sparse verify row capacity differs from the bound step")
        return self.active.run(projected, *args, **kwargs)


class MultiSparseVerifyIndexKey:
    def __init__(self, base, core):
        self.base, self.core = base, core
        self.writers = {
            rows: SparseVerifyIndexKey(base, profile)
            for rows, profile in core.profiles.items()
        }

    def bind(self, cache, page_size, rows=(8,)):
        for profile in self.core.profiles.values():
            profile.index_cache = cache
        return self.base.bind(cache, page_size, rows)

    def run(self, hidden, slots):
        return self.writers[hidden.shape[0]].run(hidden, slots)


class BuiltSparseVerify(BuiltK31Verify):
    def __init__(self, weights, deadline_s):
        super().__init__(weights, deadline_s)
        self.sparse_cores = {}
        self._fused_logged = set()
        self._dcp_logged = set()
        self.split = None  # rows -> (LocalIndexer, SplitMerge), shared by the layers
        self.split_device = None
        for record in weights.layers:
            if record.attention_kind != 0:
                continue
            a, sc = record.attention, record.attention_scalars
            if len(a) != 12:
                raise ValueError("sparse verify needs index-Q/head-gate weight tensors")
            attention = self.resources.attention[record.layer_id]
            if self.split is None:
                self.split_device = attention.core.device
                self.split = self._build_split(
                    self.split_device, weights.context_length
                )
            core = MultiSparseVerifyMla(
                attention.core,
                IndexWeights(a[10], a[11], a[1], sc[0]),
                self.widths,
                weights.context_length,
                self.split,
            )
            attention.core = core
            self.sparse_cores[record.layer_id] = core
            self.indexers[record.layer_id] = MultiSparseVerifyIndexKey(
                self.indexers[record.layer_id], core
            )

    def _build_split(self, device, context_length):
        """The context-split indexer (a collective: every TP rank builds the merge's symmetric buffer), or {}
        (an empty split keeps every step on the replicated indexer): a context length below SPLIT_MIN, DCP (it
        scores the replicated index-K cache) or no usable 16-rank merge."""
        from .split import SPLIT_MIN, LocalIndexer, SplitMerge

        if context_length < SPLIT_MIN:  # no forward reaches it
            logger.info(
                "NVIDIA megakernel: sparse verify context split off: context length %d < %d",
                context_length,
                SPLIT_MIN,
            )
            return {}
        if dcp_layout()[0] > 1:
            logger.info(
                "NVIDIA megakernel: sparse verify context split off: DCP (it scores the replicated "
                "index-K cache)"
            )
            return {}
        from sglang.srt.distributed.parallel_state import get_tp_group

        group = get_tp_group()
        merge = SplitMerge(group, max(self.widths))
        if not merge.usable(max(self.widths)):
            logger.info(
                "NVIDIA megakernel: sparse verify context split off: the 16-rank DCP top-k merge is "
                "unavailable (TP %d)",
                group.world_size,
            )
            return {}
        max_pages = -(-context_length // 64)
        logger.info(
            "NVIDIA megakernel: sparse verify context split armed: ctx >= %d, world %d, rows %s",
            SPLIT_MIN,
            group.world_size,
            self.widths,
        )
        return {
            rows: (
                LocalIndexer(
                    device,
                    requests=rows // self.query_width,
                    query_width=self.query_width,
                    max_pages=max_pages,
                    world=group.world_size,
                    rank=group.rank_in_group,
                ),
                merge,
            )
            for rows in self.widths
        }

    def _prepare_split(self, view):
        """The split's once-per-forward prepare on the index stream at forward start: it overlaps the KDA layers
        before the first MLA layer instead of delaying that layer's selection. The cores then find it done
        (``prepared_for``); the executor hands them this same ``view.block_table``."""
        local = self.split[view.row_count][0]
        index_stream = _index_stream(self.split_device)
        index_stream.wait_stream(torch.cuda.current_stream(self.split_device))
        with torch.cuda.stream(index_stream):
            local.prepare(view.block_table, view)

    def summary(self):
        graphs = "dense/sparse/split" if self.split else "dense/sparse"
        return f"TARGET_VERIFY Breq={self.request_batches} Q={self.query_width}; causal sparse DSA; host {graphs} graphs"

    def run_verify(self, embeddings, view):
        if not isinstance(view, SparseVerifyStep):
            raise ValueError("sparse verify requires the host causal-metadata adapter")
        view.validate()
        for core in self.sparse_cores.values():
            core.step = view
        if view.split and self.split:
            self._prepare_split(view)
        result = super().run_verify(embeddings, view)
        key = (view.batch_requests, view.query_width)
        fused = view.sparse and any(
            c.active.fused_index for c in self.sparse_cores.values()
        )
        if fused and key not in self._fused_logged:  # engagement tag for A/B runs
            logger.info(
                "NVIDIA megakernel: sparse verify index query FusedIndexQuery Breq=%d Q=%d",
                *key,
            )
            self._fused_logged.add(key)
        if view.sparse and view.dcp_size > 1 and key not in self._dcp_logged:
            logger.info(
                "NVIDIA megakernel: sparse verify under DCP%d (rank %d): CP sparse core + LSE merge Breq=%d Q=%d",
                view.dcp_size,
                view.dcp_rank,
                *key,
            )
            self._dcp_logged.add(key)
        return result


def build_from_tensors(weights, deadline_s):
    return BuiltSparseVerify(weights, deadline_s)
