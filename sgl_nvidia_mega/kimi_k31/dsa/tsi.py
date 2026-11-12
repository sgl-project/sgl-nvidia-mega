# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Token-split indexer (TSI) for the K3.1 MK sparse DSA verify at TP16/DCP1.

At DCP1 every rank scores the whole (replicated) index-K context and selects the same top-2048. TSI splits only the
scan: rank r scores the global pages p with p % W == r (W = TP size), takes its local top-2048 with stock topk_v2
(stable ties), and the ranks run one candidate exchange plus the exact certified merge (tsi.cuh). Every rank ends
with the same selected KV slots, sorted by position (``K31MK_TSI_S1R=1``: source-major, see S1R).

Serving: ``TsiComm.from_group`` rendezvouses one receive region per rank in torch symmetric memory; one ``TsiState``
per model (shared by every DSA layer): ``prepare`` once per forward, ``select`` per layer. Tests: ``TsiComm(...,
peers=[region of rank 0..W-1])`` emulates every rank on one GPU, and ``phases`` splits a layer's merge into its push
and poll halves.
"""

from __future__ import annotations

import math
import os
import re
from functools import cache
from pathlib import Path

import torch

# SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/jit/utils/compile/loader.py": ("load_jit",),
    "kernels/ops/attention/dsa/paged_mqa_logits.py": ("_restore_row_stride",),
    "kernels/ops/attention/dsv4/topk.py": ("plan_topk_v2", "topk_transform_paged_v2"),
}

WORLD = 16
TOPK = 2048
PAGE = 64
INDEX_BYTES = 132  # FP8 index-K row (128) + FP32 scale
CAND = int(os.environ.get("K31MK_TSI_CAND", "256"))
DEADLINE_NS = int(float(os.environ.get("K31MK_TSI_DEADLINE_S", "10")) * 1e9)
# plan_topk_v2 static threshold for the local rows. Note: at <= kNumPersistentClusters rows
# (8 or 16 here) topk_plan writes no plan and the small-batch kernel routes by its fixed floor (24,576 tokens on
# Blackwell-class GPUs), so this knob does not change the path at our row counts (bench_tsi.py measures both).
TOPK_CLUSTER_MIN = int(os.environ.get("K31MK_TSI_TOPK_CLUSTER_MIN", "4096"))
# ``K31MK_TSI_S1R=1`` (default 0 = off: the v6 merge): the "s1r" merge, tsi.cuh K31_TSI_EMIT 3 (measured faster per merge,
# rows 8, 131K-786K tokens). The sender ranks its candidates by position and the
# receiver emits non-fallback rows without the block radix sort. Same selected set; the slot ORDER differs from the
# default: source-major (source (pos // 64) % W ascending, position ascending inside a source; certificate-fallback
# rows stay position-sorted), so the sparse core sums in another order (REORDER class vs the default). Read at import
# (one module build per boot); needs 32 * W == 2 * C (W 16, C 256).
_S1R = os.environ.get("K31MK_TSI_S1R", "0")
if _S1R not in ("0", "1"):
    raise ValueError(f"K31MK_TSI_S1R must be 0 or 1, got {_S1R!r}")
S1R = _S1R == "1"
# ``K31MK_TSI_T1L=1`` (default 0 = off; needs K31MK_TSI_S1R=1): "T1-lite", tsi.cuh K31_TSI_T1L. The scan's local top-k
# keeps the local top-C (topk_v2 at k = C) instead of the top-2048 and the merge pushes that list as the candidates; a
# rank computes and publishes its full local top-2048 only for rows whose certificate failed on it, after its poll.
# Same candidates, same t*, same slots in the same order as the s1r merge. Read at import (one module build per boot).
_T1L = os.environ.get("K31MK_TSI_T1L", "0")
if _T1L not in ("0", "1"):
    raise ValueError(f"K31MK_TSI_T1L must be 0 or 1, got {_T1L!r}")
T1L = _T1L == "1"
if T1L and not S1R:
    raise ValueError("K31MK_TSI_T1L=1 needs K31MK_TSI_S1R=1 (the T1L merge is built on the s1r merge)")
MERGE_DEFINES = (("K31_TSI_EMIT=3",) if S1R else ()) + (("K31_TSI_T1L=1",) if T1L else ())
# ``K31MK_TSI_MERGE=tsim`` (default unset = the local top-k + the v6 / s1r / T1L merge above): the TSIM kernel (tsim.cuh)
# replaces topk_v2 + merge with ONE kernel per DSA layer (local threshold filter, one exchange, exact merge with an exact
# fallback). Same selected set, s1r's slot order on every row (fallback rows included); T1L's merge is not used under TSIM.
# Its own symmetric region; the push is UNICAST (strong .sys stores through the peers' IPC VAs) unless ``K31MK_TSIM_MC=1``:
# one multimem.st per 64-bit element through the region's multicast VA, polled through the reader's unicast VA with the
# per-element tag check and no proxy fence (the multicast-store / unicast-read form of the symmetric-memory all-gathers
# SGLang serves). Serves the full TP16 group only.
_MERGE = os.environ.get("K31MK_TSI_MERGE", "")
if _MERGE not in ("", "tsim"):
    raise ValueError(f"K31MK_TSI_MERGE must be unset or 'tsim', got {_MERGE!r}")
TSIM = _MERGE == "tsim"
_TSIM_MC = os.environ.get("K31MK_TSIM_MC", "0")
if _TSIM_MC not in ("0", "1"):
    raise ValueError(f"K31MK_TSIM_MC must be 0 or 1, got {_TSIM_MC!r}")
TSIM_MC = _TSIM_MC == "1"
# ``K31MK_DSA_DG_VENDOR=1`` (default 0 = off; the same switch as verify.py's DSA_DG_VENDOR): the scan's logits run
# dg_logits.py's vendored copy of DeepGEMM's paged MQA logits kernel instead of deep_gemm.fp8_paged_mqa_logits. Bitwise
# equal logits; the copy is launched with PDL, fences the async proxy in its TMA warps after the grid-dependency wait (its
# inputs were written with plain stores) and triggers the local top-k launch right after that wait.
DG_VENDOR = os.environ.get("K31MK_DSA_DG_VENDOR", "0") == "1"
# ``K31MK_DSA_IDX_FP4=1`` (dsa/idx_fp4_knob.py): the index-K pool holds MXFP4 pages [pages, 64 * 68]; the scan's local logits run the
# verify path's MXFP4-page logits (idx_fp4_q.paged_logits: K31MK_DSA_IDX_Q=fp8 -> FP8 query x MXFP4 index-K, fp4 -> MXFP4 query) over this rank's
# pages. Per-token logits do not depend on which pages a launch covers, so a rank's local logits equal the DCP1 call's at the same positions and
# the merge / TSIM selection is unchanged.
from .idx_fp4_knob import ENABLED as IDX_FP4  # noqa: E402
# Merge phases (tsi.cuh): PUSH = the scan and the push half, POLL = the poll half; BOTH = one served launch. Under T1L the
# poll half ends after the certificate and the FULL publish, and FINISH runs the fallback and the output (a served BOTH
# launch adds FINISH); the 1-GPU emulation launches PUSH, POLL and FINISH for every rank in turn.
PUSH, POLL, BOTH, FINISH = 1, 2, 3, 4
KIND_CAND, KIND_FULL = 0, 1
KINDS = 2
_SOURCE = Path(__file__).with_name("tsi.cuh")


@cache
def load(world: int = WORLD, cand: int = CAND, defines: tuple = ()):
    """``defines``: the tsi.cuh -D flags of this build (the served merge: MERGE_DEFINES)."""
    import sglang
    from sglang.kernels.jit.utils import load_jit

    tag = ("k31_tsi", f"w{world}_c{cand}")
    tag += tuple(re.sub(r"[^A-Za-z0-9]+", "_", d).strip("_").lower() for d in defines)
    binding = f"k31_tsi::Binding<{world}, {cand}>"
    return load_jit(
        *tag,
        cuda_files=[str(_SOURCE)],
        cuda_wrappers=[
            ("prep", f"{binding}::prep"),
            ("merge", f"{binding}::merge"),
            ("fill", f"{binding}::fill"),
        ],
        extra_cuda_cflags=[f"-D{d}" for d in defines],
        extra_include_paths=[str(Path(sglang.__file__).parent / "kernels/jit/csrc")],
    )


def tag(epoch: int, ordinal: int, kind: int, layers: int = 24) -> int:
    """The 32-bit tag of a record of (forward ``epoch``, DSA ``ordinal``, ``kind``); never 0 (tsi.cuh make_tag)."""
    value = ((epoch * layers + ordinal) * KINDS + kind + 1) & 0xFFFFFFFF
    return value or 1


def region_records(world: int, cap: int, cand: int) -> int:
    """int64 records of one rank's region: CAND [2][W][cap][C][2] + FULL [2][cap][2048][2]."""
    return 2 * world * cap * cand * 2 + 2 * cap * TOPK * 2


def cand_index(world, cand, cap, parity, src, row, e, half=0) -> int:
    return ((((parity * world + src) * cap + row) * cand) + e) * 2 + half


def full_index(world, cand, cap, parity, row, i, half=0) -> int:
    return 2 * world * cap * cand * 2 + ((parity * cap + row) * TOPK + i) * 2 + half


class TsiComm:
    """The TSI exchange state of one model: one receive region (int64 records) per rank and the forward epoch."""

    def __init__(
        self,
        device,
        world: int = WORLD,
        rank: int = 0,
        *,
        cap: int = 32,
        cand: int = CAND,
        layers: int = 24,
        peers: list[torch.Tensor] | None = None,
        epoch: torch.Tensor | None = None,
        deadline_ns: int = DEADLINE_NS,
        live: int | None = None,
        defines: tuple | None = None,
    ):
        """``peers``: [region tensor of rank 0..world-1] (tests: every rank emulated on one GPU). ``epoch``: an int32
        device tensor whose element 0 is this rank's forward counter. ``defines``: tsi.cuh -D flags of this build;
        None = the served merge (MERGE_DEFINES, from K31MK_TSI_S1R)."""
        defines = MERGE_DEFINES if defines is None else tuple(defines)
        if "K31_TSI_EMIT=3" in defines and 32 * world != 2 * cand:
            raise ValueError(f"TSI s1r merge needs 32 * world == 2 * candidates, got world {world} candidates {cand}")
        if not 2 <= world <= 16 or not 0 <= rank < world:
            raise ValueError(f"TSI world {world} / rank {rank} unsupported (2..16)")
        if cand % 32 or not 32 <= cand <= TOPK:
            raise ValueError(f"TSI candidates {cand} must be a multiple of 32 in [32, 2048]")
        self.device = torch.device(device)
        self.world, self.rank, self.cap, self.cand, self.layers = world, rank, cap, cand, layers
        self.deadline_ns, self.defines = int(deadline_ns), tuple(defines)
        self.t1l = "K31_TSI_T1L=1" in self.defines
        if self.t1l and "K31_TSI_EMIT=3" not in self.defines:
            raise ValueError("TSI T1L merge needs the s1r merge (K31_TSI_EMIT=3)")
        self.live = world if live is None else live
        self.records = region_records(world, cap, cand)
        self.epoch = epoch if epoch is not None else torch.zeros(4, dtype=torch.int32, device=self.device)
        self.regions = peers
        self.pointers = [int(t.data_ptr()) for t in peers] if peers is not None else None
        if peers is not None:
            if len(peers) != world or any(t.numel() < self.records or t.dtype != torch.int64 for t in peers):
                raise ValueError("peers: one int64 region of region_records() per rank")
        self._meta: dict[tuple[int, int], torch.Tensor] = {}
        self.module = None

    @classmethod
    def from_group(cls, group, device, world, rank, *, live=None, **kwargs):
        """Collective over ``group``: allocate, zero and rendezvous this rank's region. ``live`` < world (benchmarks):
        ``group`` holds ranks 0..live-1; stores to virtual ranks live..world-1 land in local scratch, and the caller
        fills this rank's records of virtual sources (``fill``)."""
        import torch.distributed._symmetric_memory as symm_mem

        live = world if live is None else live
        cap = kwargs.get("cap", 32)
        cand = kwargs.get("cand", CAND)
        buffer = symm_mem.empty(region_records(world, cap, cand), dtype=torch.int64, device=device)
        buffer.zero_()  # tag 0 is never valid
        torch.cuda.synchronize(device)
        handle = symm_mem.rendezvous(buffer, group)
        self = cls(device, world, rank, live=live, **kwargs)
        self.owned, self.handle = buffer, handle
        pointers = list(handle.buffer_ptrs)
        if len(pointers) != live:
            raise RuntimeError("TSI rendezvous did not return one region per live rank")
        self.virtual = [torch.zeros_like(buffer) for _ in range(world - live)]
        self.pointers = pointers + [int(t.data_ptr()) for t in self.virtual]
        if TSIM:
            from . import tsim

            if live != world:
                raise ValueError("K31MK_TSI_MERGE=tsim serves the full group only (live == world)")
            tbuf = symm_mem.empty(tsim.region_records(cap), dtype=torch.int64, device=device)
            tbuf.zero_()  # tag 0 is never valid
            torch.cuda.synchronize(device)
            th = symm_mem.rendezvous(tbuf, group)
            self.tsim_owned, self.tsim_handle = tbuf, th
            self.tsim_pointers = list(th.buffer_ptrs)
            # 0 = the unicast push; the multicast base under K31MK_TSIM_MC=1
            self.tsim_mc = int(getattr(th, "multicast_ptr", 0) or 0) if TSIM_MC else 0
            if TSIM_MC and not self.tsim_mc:
                raise RuntimeError("K31MK_TSIM_MC=1 but the TSIM region has no multicast object")
            tsim.load()  # JIT-built here, never inside a graph capture
        torch.distributed.barrier(group=group)
        return self

    def meta(self, ordinal: int, phases: int = BOTH) -> torch.Tensor:
        """The CPU parameter vector of one DSA layer (tsi.cuh Binding)."""
        key = (ordinal, phases)
        if key not in self._meta:
            values = [self.world, self.rank, ordinal, self.layers, self.cap, self.deadline_ns, phases, 0]
            self._meta[key] = torch.tensor(values + list(self.pointers), dtype=torch.int64)
        return self._meta[key]

    def _module(self):
        if self.module is None:
            self.module = load(self.world, self.cand, self.defines)
        return self.module

    # ----------------------------------------------------------------------------------------- launches
    def prep(self, table, contexts, causal, ltab, lctx, lcausal) -> None:
        """epoch += 1; local page table [b, ceil(pages / W)], local contexts [b, q], local causal lengths [rows]."""
        self._module().prep(self.epoch, table, contexts, causal, ltab, lctx, lcausal, self.rank)

    def merge(self, ordinal, raw, scores, causal, table, out, fallback, scratch, rows_per_request, end_bit,
              phases=BOTH) -> None:
        if self.t1l and phases == BOTH:
            phases = BOTH | FINISH
        self._module().merge(self.meta(ordinal, phases), self.epoch, raw, scores, causal, table, out, fallback,
                             scratch, rows_per_request, end_bit)

    def tsim_select(self, ordinal, scores, lcausal, causal, table, out, fallback, scratch, rows_per_request, ncols) -> None:
        """K31MK_TSI_MERGE=tsim: one TSIM launch (local selection + exchange + exact merge) for this DSA layer."""
        from . import tsim

        key = ("tsim", ordinal)
        if key not in self._meta:  # meta[7] = 0: serving (deadline traps, multicast or unicast fan-out)
            values = [self.world, self.rank, ordinal, self.layers, self.cap, self.deadline_ns, self.tsim_mc, 0]
            self._meta[key] = torch.tensor(values + list(self.tsim_pointers), dtype=torch.int64)
        tsim.load().run(self._meta[key], self.epoch, scores, lcausal, causal, table, out, fallback, scratch,
                        rows_per_request, ncols)

    def fill(self, ordinal, payload, region_ptr, kind, src_lo=0) -> None:
        """Benchmarks/tests: records of ``payload`` [nsrc, rows, n, 2] int32 (score bits, position) tagged for
        (current device epoch, ``ordinal``, ``kind``) into the region at ``region_ptr`` (KIND_CAND: sources
        src_lo.., KIND_FULL: that region's own full list, nsrc 1)."""
        self._module().fill(self.meta(ordinal), self.epoch, payload, int(region_ptr), kind, src_lo)


class TsiState:
    """The TSI selection of one model, shared by all DSA layers; static buffers sized at build."""

    def __init__(self, comm: TsiComm, max_rows: int, max_pages: int, rows_per_request: int = 8):
        if max_rows % rows_per_request or max_rows > comm.cap:
            raise ValueError("max_rows must be a multiple of rows_per_request and fit the region capacity")
        dev = comm.device
        self.comm, self.max_rows, self.max_pages, self.rows_per_request = comm, max_rows, max_pages, rows_per_request
        self.max_requests = max_rows // rows_per_request
        self.max_width = -(-max_pages // comm.world)
        i32 = dict(dtype=torch.int32, device=dev)
        self._ltab = torch.zeros(self.max_requests * self.max_width, **i32)
        self._lctx = torch.zeros(self.max_requests * rows_per_request, **i32)
        self._lcausal = torch.zeros(max_rows, **i32)
        self._plan = torch.zeros((max_rows + 1) * 2, **i32)
        self._raw = torch.full((max_rows, comm.cand if comm.t1l else TOPK), -1, **i32)  # T1L: the local top-C
        self._fallback = torch.zeros(max_rows, **i32)
        self.scratch = torch.zeros((max_rows, comm.world * TOPK), dtype=torch.int64, device=dev)
        self.sms = torch.cuda.get_device_properties(dev).multi_processor_count
        self.fallback = None
        self.scores = None
        if IDX_FP4:  # the MXFP4-page logits kernel is JIT-built here, never inside a graph capture
            from .idx_fp4_q import build_logits

            build_logits(dev.index if dev.index is not None else torch.cuda.current_device(), rows_per_request)
        elif DG_VENDOR and rows_per_request == 8:  # the vendored kernel is JIT-built here, never inside a graph capture
            from .dg_logits import build

            build(dev.index if dev.index is not None else torch.cuda.current_device(), rows_per_request)

    def prepare(self, table: torch.Tensor, contexts: torch.Tensor, causal: torch.Tensor) -> None:
        """Per forward: epoch += 1; local table, contexts and causal lengths; DeepGEMM metadata; top-k plan."""
        import deep_gemm
        from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2

        b, pages = table.shape
        q = contexts.shape[1]
        rows = causal.shape[0]
        if b > self.max_requests or pages > self.max_pages or q != self.rows_per_request or rows != b * q:
            raise ValueError(f"TSI prepare: table {tuple(table.shape)} contexts {tuple(contexts.shape)} rows {rows}")
        width = -(-pages // self.comm.world)
        self.ltab = self._ltab[: b * width].view(b, width)
        self.lctx = self._lctx[: b * q].view(b, q)
        self.lcausal = self._lcausal[:rows]
        self.comm.prep(table, contexts, causal, self.ltab, self.lctx, self.lcausal)
        self.schedule = deep_gemm.get_paged_mqa_logits_metadata(self.lctx, PAGE, self.sms)
        self.plan = plan_topk_v2(self.lcausal, TOPK_CLUSTER_MIN, out=self._plan[: (rows + 1) * 2].view(rows + 1, 2))
        self.table, self.causal = table, causal
        self.b, self.q, self.rows, self.width = b, q, rows, width
        self.end_bit = max(1, (pages * PAGE).bit_length())
        self.raw = self._raw[:rows]
        self.fallback = self._fallback[:rows]

    def scan(self, query, gates, cache):
        """DeepGEMM logits over this rank's local pages (under DG_VENDOR: dg_logits.py's bitwise-equal copy) -> topk_v2 raw
        local top-2048 (T1L: top-C; stable ties)."""
        import deep_gemm
        from sglang.kernels.ops.attention.dsa.paged_mqa_logits import _restore_row_stride
        from sglang.kernels.ops.attention.dsv4.topk import topk_transform_paged_v2

        gates = gates.reshape(self.rows, 32)
        if IDX_FP4:  # MXFP4 index-K pages: the verify path's MXFP4-page logits over this rank's local pages
            from .idx_fp4_q import paged_logits

            scores = paged_logits(query, gates, cache, self.lctx, self.ltab, self.schedule, self.width * PAGE, self.b, self.q)
        else:
            logits = deep_gemm.fp8_paged_mqa_logits
            if DG_VENDOR and self.q == 8 and gates.dtype == torch.float32:  # shapes the copy does not serve keep DeepGEMM
                from .dg_logits import paged_mqa_logits as logits
            scores = logits(
                query.view(self.b, self.q, 32, 128), cache.view(-1, PAGE, 1, INDEX_BYTES), gates, self.lctx, self.ltab,
                self.schedule, self.width * PAGE, clean_logits=False,
            )
        self.scores = _restore_row_stride(scores)
        if not TSIM:  # TSIM selects from the local logits directly
            topk_transform_paged_v2(self.scores, self.lcausal, None, self.raw, PAGE, self.plan, deterministic=True)
        return self.scores

    def select(self, ordinal, query, gates, cache, out, *, phases=BOTH):
        """Per DSA layer: ``out`` [rows, 2048] int32 KV slots of the global top-2048 under (score desc, position asc),
        sorted by position (S1R: source-major on rows whose certificate held), -1 after; ``self.fallback`` [rows] 1
        where the certificate failed. ``phases`` PUSH runs the scan and the push half, POLL the poll half, and under T1L
        FINISH the fallback and the output (tests emulating every rank on one GPU)."""
        if TSIM:
            if phases != BOTH:
                raise ValueError("K31MK_TSI_MERGE=tsim runs push and poll in one launch (phases BOTH only)")
            self.scan(query, gates, cache)
            self.comm.tsim_select(ordinal, self.scores, self.lcausal, self.causal, self.table, out, self.fallback,
                                  self.scratch, self.q, self.width * PAGE)
            return out
        if phases & PUSH:
            self.scan(query, gates, cache)
        self.comm.merge(ordinal, self.raw, self.scores, self.causal, self.table, out, self.fallback, self.scratch,
                        self.q, self.end_bit, phases)
        return out
