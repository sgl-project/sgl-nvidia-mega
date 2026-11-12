# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Request-sharded DSA selection under TP attention (``K31MK_DSA_REQ_SHARD=1``; default off).

Every TP rank holds every request's index-K. Today every rank runs the full selection (DeepGEMM paged MQA logits +
stock topk_v2) for every request. Here rank r runs the same two kernels for its owner block of ceil(B / world)
requests only, and one all-gather over the TP group hands every rank every row's [2048] int32 selection, so the
sparse core and everything after it are unchanged. Owner blocks are contiguous (rank r owns requests r * per ..
r * per + per - 1), so the rank-major gather is in request order; a rank whose block runs past the last request
repeats the last request (static shapes for graph capture) and those rows are gathered and dropped.

The selection per row is the stock selection over that row's context, computed on fewer rows: equal to the full
selection when topk_v2's result does not depend on the batch's row count (a GPU test checks it)."""

import os
from types import SimpleNamespace

import torch

PAGE = 64
TOPK = 2048
# ``K31MK_DSA_REQ_SHARD_AG`` picks the all-gather: "nccl" (default) = the group's all_gather_into_tensor (SGLang's
# GroupCoordinator -> pynccl ncclAllGather); "mm" = one multimem store kernel into a symmetric buffer (the host's
# triton_symm_mem_ag._all_gather_kernel_inner: acq_rel entry and exit barriers over per-block signal pads, fence.proxy.alias
# after each acquire), whose gathered rows this rank reads through its own unicast view of the buffer; "epoch" = the
# barrier-free epoch-flag gather (dsa/epoch_gather.py: multimem rows + one release flag per peer and block, readers acquire
# the flags and read the copy in place; copies alternate by DSA layer so buffer reuse is ordered by the previous layer's flags);
# "tag" = the flag-free gather with epoch-tagged words (dsa/tag_gather.py: unicast stores of tagged slot ids into every rank's
# mailbox, the owner polls each word's tag and decodes; no multicast, no flag, no re-arm; copies alternate as for "epoch");
# "fused" = the "tag" route's push only: the sparse core (dsa/sparse_mla6.cuh, tagged mode; needs K31MK_SPARSE_MLA6=1) polls
# the tagged words of its own rows itself, so there is no poll kernel and no decoded copy; ``select`` returns a TaggedRows.
AG = os.environ.get("K31MK_DSA_REQ_SHARD_AG", "nccl")
# ``K31MK_DSA_REQ_SHARD_SLICE=1`` (default 0 = off = the two index_select gathers): when this rank's owner rows are one
# contiguous range (always at <= world requests, i.e. one request per rank, and whenever no owner request is clamped), the
# owner block's index query and head gates are passed as row slices (views) of the step's query / gates instead of being
# gathered by two torch index_select kernels per DSA layer that run serially before the DeepGEMM logits on the selection
# stream. Same bytes, so the selection is bitwise unchanged; a non-contiguous
# query / gates tensor or owner range falls back to the gathers.
SLICE = os.environ.get("K31MK_DSA_REQ_SHARD_SLICE", "0") == "1"
# ``K31MK_DSA_REQ_SHARD_TKPUSH=1`` (default 0 = off = tag_gather's _push_kernel after the owner top-k): with the "tag" or
# "fused" route and an owner block the x47v2 top-k selects (VerifySelection.x47: K31MK_VERIFY_TOPK=x47v2 / x47960 at <= 16
# rows), that top-k stores its output rows as tagged words straight into every rank's mailbox (dsa/tkpush.py,
# dsa/x47960/topk_v2_tkpush.cuh, dsa/tag_push.cuh), so the push launch (_push_kernel) leaves the selection's
# serial chain. Same words, same mailbox, same poll / fused core. Widths whose selection is
# not the x47 top-k keep the _push_kernel; a call whose top-k launch path has no in-kernel push uses push_ext.
TKPUSH = os.environ.get("K31MK_DSA_REQ_SHARD_TKPUSH", "0") == "1"
# ``K31MK_DSA_REQ_SHARD_TAG_NOFENCE=1`` (default 0; "tag" and "fused" routes; TAGNF): the tag exchange runs without its fence.acq_rel.sys
# (_push_kernel / _poll_kernel / _push_ext_kernel, with TKPUSH the top-k's release fence, and on the "fused" route the sparse core's
# acquire fence: dsa/sparse_mla6.cuh kFence = false). Those fences order only mailbox copy
# reuse (a word is its own flag); review: no compliant credit from the served layer chain between
# DSA layers k-2 and k, accepted as a judgement because it fails closed - an early overwrite carries the next use's tag,
# every reader accepts only the exact tag (T1) and the deadline trap stays armed (T2), so a lost order traps, never a wrong row.
TAG_NOFENCE = os.environ.get("K31MK_DSA_REQ_SHARD_TAG_NOFENCE", "0") == "1"


class TaggedRows:
    """K31MK_DSA_REQ_SHARD_AG=fused: what ``select`` hands the sparse core instead of decoded rows. ``words``: this layer's
    mailbox copy [blocks x block_rows, 2048] int32 tagged words (batch row r = mailbox row r); ``epochs``: the core's
    per-(row, split) use counters of that copy [rows, splits]; ``deadline_ns``: the core's spin limit; ``fence``: False under
    K31MK_DSA_REQ_SHARD_TAG_NOFENCE (the core skips its fence.acq_rel.sys; needs deadline_ns > 0)."""

    __slots__ = ("words", "epochs", "deadline_ns", "block_rows", "blocks", "fence")

    def __init__(self, words, epochs, deadline_ns, block_rows, blocks, fence=True):
        self.words, self.epochs, self.deadline_ns, self.block_rows, self.blocks = words, epochs, deadline_ns, block_rows, blocks
        self.fence = bool(fence)
        if not self.fence and int(deadline_ns) <= 0:
            raise ValueError("fence-free tagged rows need an armed deadline (deadline_ns > 0)")


def owner_requests(b, world, rank):
    """Request indices rank ``rank`` selects for: ceil(b / world) of them (a static count), clamped to b - 1."""
    per = -(-b // world)
    return [min(rank * per + j, b - 1) for j in range(per)]


def owner_rows(b, q, world, rank):
    """Verify rows (request-major, q per request) of ``owner_requests``."""
    return [r * q + j for r in owner_requests(b, world, rank) for j in range(q)]


def owner_slice(b, q, world, rank):
    """First row of this rank's owner rows when they form one contiguous range of ceil(b / world) * q rows, else None."""
    rows = owner_rows(b, q, world, rank)
    return rows[0] if rows == list(range(rows[0], rows[0] + len(rows))) else None


def req_shard_label(seq_lens, min_requests, min_context, sum_context=0):
    """True when a verify step should replay the request-sharded graph: at least ``min_requests`` requests and
    EVERY request's context >= ``min_context`` (the shortest request decides). With ``sum_context`` > 0
    (K31MK_DSA_REQ_SHARD_SUM_CONTEXT) the context test is the SUM of the requests' contexts >= ``sum_context``
    instead: the replicated selection's cost follows the summed context while the sharded one follows the longest
    request plus one all-gather, so one short request should not keep a wide step on the replicated selection."""
    if seq_lens is None or seq_lens.numel() == 0:
        return False
    if seq_lens.numel() < min_requests:
        return False
    if sum_context > 0:
        return int(seq_lens.sum()) >= sum_context
    return int(seq_lens.min()) >= min_context


class SplitSelection:
    """K31MK_DSA_REQ_SHARD_WITH_TSI=1 (default off; dsa/verify.py): the host's third verify graph runs the TSI state at rows <=
    ``split_rows`` and the request-sharded state above. A captured graph has fixed capacity rows, so ``prepare`` (ordinal 0 of
    every forward, on the selection stream) picks the state by rows and every later DSA layer of that forward uses the same
    state; each state therefore sees only whole forwards, which keeps its own copy / epoch alternation. Same interface as
    TsiState and ReqShardState."""

    def __init__(self, tsi, rs, split_rows):
        self.tsi, self.rs, self.split_rows = tsi, rs, int(split_rows)
        self.active = None

    def pick(self, rows):
        return self.tsi if int(rows) <= self.split_rows else self.rs

    def prepare(self, table, contexts, causal):
        self.active = self.pick(causal.shape[0])
        self.active.prepare(table, contexts, causal)

    def select(self, ordinal, query, gates, cache, out):
        return self.active.select(ordinal, query, gates, cache, out)


def pool_slots(cache):
    """Token slots of the index-K pool = pages x PAGE. Dim 0 is the page in every layout the selection sees ([pages, 64 x row]
    or [pages, 64, 1, row]), whatever the row bytes (132: FP8 + scale; 68: the MXFP4 index cache), so the count needs neither
    (numel // 132 under-counted a 68-byte pool ~1.94x; shape[-1] // 64 over-counts a 4-D view)."""
    return int(cache.shape[0]) * PAGE


def check_slot_pool(slots, ag):
    """The tag and fused routes carry slot ids in 25 bits as v + 1 (dsa/tag_gather.py), so every slot of the index-K pool
    (``slots`` token slots from pool_slots, ids 0 .. slots - 1) must satisfy v + 1 <= 2**25 - 1. Raises otherwise: a larger
    id would wrap silently under an intact epoch tag. Called at the first select on every rank (the pool is the same size
    on every rank, so all ranks raise before any gather)."""
    from .tag_gather import VALUE_LIMIT

    if slots > VALUE_LIMIT:
        raise ValueError(f"K31MK_DSA_REQ_SHARD_AG={ag} carries slot ids below {VALUE_LIMIT}; the pool holds {slots} token slots")


class ReqShardState:
    """The request-sharded selection of one model, shared by all DSA layers; static buffers per row capacity.
    Same interface as TsiState: ``prepare`` once per forward (ordinal 0, selection stream), ``select`` per layer.
    ``group``: an object with ``all_gather_into_tensor(output, input)`` over the TP ranks (SGLang's GroupCoordinator); the
    "mm", "epoch" and "tag" all-gathers also need its torch process group (``group.device_group``, else the default group).
    ``layers``: DSA layers per forward (``select`` ordinals 0 .. layers - 1; the epoch and tag routes alternate copies over
    them); ``deadline_ns``: their spin limit (a late or lost peer traps the step)."""

    def __init__(self, group, world, rank, device, widths, query_width, ag=None, layers=None, deadline_ns=None):
        from .verify import VerifySelection

        self.group, self.world, self.rank, self.q = group, int(world), int(rank), int(query_width)
        self.ag = AG if ag is None else ag
        if self.ag not in ("nccl", "mm", "epoch", "tag", "fused"):
            raise ValueError(f"K31MK_DSA_REQ_SHARD_AG must be nccl, mm, epoch, tag or fused, got {self.ag!r}")
        self.layers = layers
        self.slots_checked = False
        self.nofence = TAG_NOFENCE and self.ag in ("tag", "fused")
        if TAG_NOFENCE and not self.nofence:
            raise ValueError(f"K31MK_DSA_REQ_SHARD_TAG_NOFENCE=1 is built for K31MK_DSA_REQ_SHARD_AG=tag / fused only, got {self.ag!r}")
        if self.ag in ("epoch", "tag", "fused"):
            from .epoch_gather import copies_for, copy_for

            copy_for(0, layers)  # raises below 2 layers
            self.copies = copies_for(layers)
            self.deadline_ns = int(deadline_ns) if deadline_ns is not None else EPOCH_DEADLINE_NS
        mm = _multimem_module() if self.ag == "mm" else None
        self.sms = torch.cuda.get_device_properties(device).multi_processor_count
        i32 = dict(dtype=torch.int32, device=device)
        self.widths = {}
        for rows in sorted(set(widths)):
            if rows % self.q:
                raise ValueError(f"request sharding: rows {rows} is not a multiple of the query width {self.q}")
            b = rows // self.q
            per = -(-b // self.world)
            self.widths[rows] = SimpleNamespace(
                b=b, per=per,
                req=torch.tensor(owner_requests(b, self.world, self.rank), dtype=torch.long, device=device),
                rows_idx=torch.tensor(owner_rows(b, self.q, self.world, self.rank), dtype=torch.long, device=device),
                own_lo=owner_slice(b, self.q, self.world, self.rank) if SLICE else None,
                selection=VerifySelection(device, per * self.q),
                gathered=torch.full((self.world * per * self.q, TOPK), -1, **i32),
                contexts=torch.zeros((per, self.q), **i32),
                causal=torch.zeros(per * self.q, **i32),
                plan=torch.zeros((per * self.q + 1) * 2, **i32),
                table=None, step=None, mm=None,
            )
            w = self.widths[rows]
            if mm is not None:  # one symmetric buffer per width: the gathered rows are one bf16 "row" of world x per x q x 2048 int32
                width = self.world * per * self.q * TOPK * 2
                w.mm = mm.create_state(group=_device_group(group), rank_in_group=self.rank, max_tokens=1, hidden_size=width, device=device)
                if w.mm.symm_mem_hdl.multicast_ptr == 0:
                    raise ValueError(f"K31MK_DSA_REQ_SHARD_AG=mm: no multicast for world {self.world}")
                w.gathered = w.mm.comm_buff[:1, :width].view(torch.int32).view(self.world * per * self.q, TOPK)
                w.mm_width = width
            if self.ag == "epoch":  # copies x [world x per x q, 2048] int32; select returns the copy of the layer's ordinal
                from .epoch_gather import EpochGather

                w.eg = EpochGather(_device_group(group), self.rank, self.world, per * self.q * TOPK, self.copies, device,
                                   self.deadline_ns)
                w.eg.self_check(layers)
                w.gathered = None
            w.tkp = None
            if self.ag in ("tag", "fused"):  # copies x [world x per x q, 2048] tagged words; select uses the layer ordinal's copy
                from .tag_gather import TagGather

                tkpush = TKPUSH and w.selection.x47 is not None
                if tkpush:  # JIT-build the push top-k before TagGather's barrier, so no rank compiles while a peer self-checks
                    from .tkpush import load as _tkpush_load

                    _tkpush_load(1, self.nofence)
                w.tg = TagGather(_device_group(group), self.rank, self.world, per * self.q * TOPK, self.copies, device,
                                 self.deadline_ns, ext=tkpush, fence=not self.nofence)
                if tkpush:  # the owner top-k pushes (K31MK_DSA_REQ_SHARD_TKPUSH); module already built above
                    from .tkpush import TkPushSelect

                    w.tkp = TkPushSelect(device, per * self.q, w.tg.peers(), self.world, self.rank, per * self.q * TOPK,
                                         nofence=self.nofence)
                    w.selection.x47 = w.tkp  # with TKPUSH only (979e3d4 had this line under ``if self.nofence``)
                if self.nofence:
                    import logging

                    logging.getLogger(__name__).info("K31MK_DSA_REQ_SHARD_TAG_NOFENCE ENGAGED: %s exchange without fence.acq_rel.sys "
                                                     "(rows %d, world %d, tkpush %d, deadline_ns %d)", self.ag, rows, self.world,
                                                     int(tkpush), self.deadline_ns)
                w.tg.self_check(layers)
                w.gathered = None
            if self.ag == "fused":  # the sparse core's use counters per copy and (row, split); self_check used the last layer's copy once
                from .epoch_gather import copy_for
                from .sparse_mla6 import TOKENS

                w.core_epochs = torch.zeros((self.copies, b * self.q, -(-TOPK // TOKENS)), **i32)
                w.core_epochs[copy_for(layers - 1, layers)] = 1
        self.mm = mm
        # The all-gather runs on its own stream, joined by stream waits (events) on both sides: the edge from the top-k
        # (a PDL primary that triggers early) to the collective is then a full stream dependency whatever launch
        # attributes the collective's kernel uses (memory-model audit E1).
        self.comm_stream = torch.cuda.Stream(device=device)
        self.active = None

    def prepare(self, table, contexts, causal):
        """Per forward: the owner block's page table, contexts, causal lengths, DeepGEMM schedule and top-k plan."""
        import deep_gemm
        from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2

        rows = causal.shape[0]
        w = self.widths.get(rows)
        if w is None or table.shape[0] != w.b or tuple(contexts.shape) != (w.b, self.q):
            raise ValueError(f"request sharding prepare: table {tuple(table.shape)} contexts {tuple(contexts.shape)} rows {rows}")
        w.table = table.index_select(0, w.req)
        torch.index_select(contexts, 0, w.req, out=w.contexts)
        torch.index_select(causal, 0, w.rows_idx, out=w.causal)
        schedule = deep_gemm.get_paged_mqa_logits_metadata(w.contexts, PAGE, self.sms)
        plan = plan_topk_v2(w.causal, out=w.plan.view(w.per * self.q + 1, 2))
        w.step = SimpleNamespace(batch_requests=w.per, query_width=self.q, score_contexts=w.contexts,
                                 causal_lengths=w.causal, score_schedule=schedule, topk_plan=plan)
        self.active = w

    def select(self, ordinal, query, gates, cache, selected=None):
        """This rank's owner rows through the stock selection, then the all-gather; returns [rows, 2048] int32 in
        request order (``ordinal`` and ``selected`` are TsiState's interface; unused)."""
        w = self.active
        if isinstance(query, torch.Tensor) and w.own_lo is not None and query.is_contiguous() and gates.is_contiguous():  # K31MK_DSA_REQ_SHARD_SLICE: views
            q_own = query[w.own_lo : w.own_lo + w.per * self.q]
            g_own = gates[w.own_lo : w.own_lo + w.per * self.q]
        elif isinstance(query, torch.Tensor):
            q_own = query.view(torch.uint8).index_select(0, w.rows_idx).view(query.dtype)
            g_own = gates.index_select(0, w.rows_idx)
        elif (w.own_lo is not None and gates.is_contiguous()
              and (q_own := query.slice_rows(w.own_lo, w.per * self.q)) is not None):  # SLICE x K31MK_DSA_IDX_Q=fp4: views
            g_own = gates[w.own_lo : w.own_lo + w.per * self.q]
        else:  # K31MK_DSA_IDX_FP4: the MXFP4 index query (dsa/idx_fp4.Fp4Query)
            q_own = query.index_select(w.rows_idx)
            g_own = gates.index_select(0, w.rows_idx)
        if w.tkp is not None:  # K31MK_DSA_REQ_SHARD_TKPUSH: this layer's copy and its consumer's epoch counter
            from .epoch_gather import copy_for

            c = copy_for(ordinal, self.layers)
            w.tkp.arm(c, w.core_epochs[c].view(-1)[:1] if self.ag == "fused" else w.tg.tag_epoch(c))
        own = w.selection.run(q_own, g_own, cache, w.table, w.step)
        cur = torch.cuda.current_stream(own.device)
        self.comm_stream.wait_stream(cur)
        if self.ag == "epoch":
            from .epoch_gather import copy_for

            c, p = copy_for(ordinal, self.layers), copy_for((ordinal - 1) % self.layers, self.layers)
            with torch.cuda.stream(self.comm_stream):
                rows = w.eg.gather(c, p, own.view(-1))
            cur.wait_stream(self.comm_stream)
            return rows.view(self.world * w.per * self.q, TOPK)[: w.b * self.q]
        if self.ag in ("tag", "fused"):
            from .epoch_gather import copy_for

            if not self.slots_checked:  # slot ids are below the index-K pool's token count (pages x 64 tokens x 132 B)
                check_slot_pool(pool_slots(cache), self.ag)  # pages x 64 for FP8 (132 B) and MXFP4 (68 B) index-K pages alike
                self.slots_checked = True
            c = copy_for(ordinal, self.layers)
            pushed = w.tkp.take() if w.tkp is not None else None  # K31MK_DSA_REQ_SHARD_TKPUSH: the top-k pushed (armed above)
            if self.ag == "fused":  # push only; the sparse core polls its rows (and every block's first word) itself
                if w.tkp is None:
                    with torch.cuda.stream(self.comm_stream):
                        w.tg.push(c, own.view(-1))
                    cur.wait_stream(self.comm_stream)
                elif not pushed:
                    with torch.cuda.stream(self.comm_stream):
                        w.tg.push_ext(c, own.view(-1), w.core_epochs[c].view(-1)[:1])
                    cur.wait_stream(self.comm_stream)
                return TaggedRows(w.tg.words(c).view(self.world * w.per * self.q, TOPK), w.core_epochs[c], self.deadline_ns,
                                  w.per * self.q, self.world, fence=not self.nofence)
            with torch.cuda.stream(self.comm_stream):
                if w.tkp is None:
                    rows = w.tg.gather(c, own.view(-1))
                else:
                    if not pushed:
                        w.tg.push_ext(c, own.view(-1), w.tg.tag_epoch(c))
                    rows = w.tg.poll(c)
            cur.wait_stream(self.comm_stream)
            return rows.view(self.world * w.per * self.q, TOPK)[: w.b * self.q]
        with torch.cuda.stream(self.comm_stream):
            if self.mm is None:
                self.group.all_gather_into_tensor(w.gathered, own)
            else:  # rank r's [per x q, 2048] block lands at row r x per x q of every rank's buffer (one bf16 row of the buffer)
                self.mm.all_gather_inner(w.mm, own.view(torch.bfloat16).view(1, -1), tp_hidden_dim=w.mm_width,
                                         skip_entry_sync=False, safe=False)
        cur.wait_stream(self.comm_stream)
        return w.gathered[: w.b * self.q]


EPOCH_DEADLINE_NS = 10_000_000_000  # the epoch route's spin limit when the caller passes none (10 s)


def _device_group(group):
    """The torch process group behind ``group`` (SGLang's GroupCoordinator.device_group), else the default group."""
    pg = getattr(group, "device_group", None)
    if pg is None:
        import torch.distributed as dist

        pg = dist.group.WORLD
    return pg


def _multimem_module():
    """The host's multimem all-gather, refused unless its acquire carries the alias fence (the compliant form:
    acq_rel barriers + fence.proxy.alias after the acquire; host triton_symm_mem_ag.py)."""
    from sglang.srt.distributed.device_communicators import triton_symm_mem_ag as mm

    src = getattr(mm._wait_signal_acquire, "src", "")
    if "fence.proxy.alias" not in src or "acq_rel" not in getattr(mm._all_gather_kernel_inner, "src", ""):
        raise ValueError("K31MK_DSA_REQ_SHARD_AG=mm needs the host triton_symm_mem_ag with acq_rel barriers and fence.proxy.alias "
                         "after the acquire (98d17ba0 or later)")
    return mm
