# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Causal sparse DSA extension of the tensor-only TARGET_VERIFY entry.

The existing executor, KDA rings, MoE, projection and aux consumers are reused.
Index-K is written for all query rows before scoring. Native paged MQA matches
stock's [1,Q,H,D] score layout; per-row causal lengths mask top-k selection.
"""
import logging
import os
from dataclasses import dataclass
from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import fake, layout, stream, to_cute
from sgl_nvidia_mega.kimi_k31.k3.cutedsl.mla.head_gate import G18 as DSA_QF_G18
from sgl_nvidia_mega.kimi_k31.k3.cutedsl.mla.head_gate import INDEX_ROWS, head_gate
from sgl_nvidia_mega.kimi_k31.k3.cutedsl.mla.value import MlaFoldMerge, MlaValue, MlaValueFold
from ..verify_binding import VerifyStepView
from ..verify_spec import BuiltK31Verify
from ..verify_contract import Q8_REQUEST_BATCHES, SHORT_QUERY_WIDTHS, verify_keys
from .dg_logits import paged_mqa_logits as dg_paged_mqa_logits
from .idx_fp4_knob import ENABLED as IDX_FP4
from .idx_fp4_knob import FP8Q as IDX_FP4_FP8Q
from .idx_fp4_knob import FP4Q as IDX_FP4_FP4Q
from .indexer import IndexProjection, IndexWeights
from .mla import SparseMla
from .stream_check import ensure as ensure_stream

logger = logging.getLogger(__name__)


# SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/ops/attention/dsa/paged_mqa_logits.py": ("_restore_row_stride",),
    "kernels/ops/attention/dsv4/topk.py": ("topk_transform_paged_v2",),
    "kernels/ops/kvcache/dsa_fused_store_index_cache.py": ("fused_store_index_k_cache",),
}


@cache
def _index_stream(device):
    return torch.cuda.Stream(device=device)


@cache
def _gate_stream(device):
    return torch.cuda.Stream(device=device)


@cache
def _query_stream(device):
    return torch.cuda.Stream(device=device)


@cache
def _padded_value(rows, pitch, weight_layout, device, q, heads=12, fence=False):
    """Read the first ``heads`` (TP8 12, TP16 6) heads directly from the sparse core's 16-head output. ``fence``:
    MlaValue's fence.acq_rel.gpu before its PDL trigger (part of the cache key)."""
    latent = fake((rows, heads, 512), (16 * 512, 512, 1))
    weight = fake(*weight_layout)
    gate = fake((rows, heads * 128), (pitch, 1))
    lengths = fake((rows // q,), (1,), torch.int32, 4)
    output = fake((rows, heads * 128))
    with torch.accelerator.device_index(device):
        return cute.compile(
            MlaValue(heads, rows, q, fence), latent, weight, gate, lengths, output,
            make_fake_stream(),
        )


@cache
def _fold_value(rows, pitch, weight_layout, device, q, heads, splits, early=False):
    """MlaValueFold over the sparse MLA6 core's split partials; the merged latent also goes to the core's
    16-head latent view (as _padded_value reads it). ``early``: K31MK_MLAVALUE_FOLD_EARLY's mode, False / True /
    "fence" (part of the cache key)."""
    partial = fake((rows, splits, heads, 512), None, torch.float32, 16)
    stats = fake((rows, splits, heads, 2), None, torch.float32, 8)
    weight = fake(*weight_layout)
    gate = fake((rows, heads * 128), (pitch, 1))
    lengths = fake((rows // q,), (1,), torch.int32, 4)
    output = fake((rows, heads * 128))
    latent = fake((rows, heads, 512), (16 * 512, 512, 1))
    with torch.accelerator.device_index(device):
        return cute.compile(
            MlaValueFold(heads, rows, q, splits, early), partial, stats, weight, gate, lengths, cutlass.Float32(1.0),
            output, latent, make_fake_stream(),
        )


class _SplitFold:
    """K31MK_FOLD_SPLIT_ROWS: MlaFoldMerge, then the shipped MlaValue; called with _fold_value's arguments."""

    kind = "split"

    def __init__(self, merge, value):
        self.merge, self.value = merge, value

    def __call__(self, partial, stats, w_uv, gate, lengths, out_scale, output, latent, s):
        self.merge(partial, stats, out_scale, latent, s)
        self.value(latent, w_uv, gate, lengths, output, s)


@cache
def _split_fold_value(rows, pitch, weight_layout, device, q, heads, splits, early=False):
    """MlaValueFold's result in two PDL launches on one stream (K31MK_FOLD_SPLIT_ROWS): MlaFoldMerge writes the merged
    latent into the core's 16-head latent view, then MlaValue (_padded_value) applies W_UV and the gate from it.
    ``early``: K31MK_MLAVALUE_FOLD_EARLY's mode. Under "fence" MlaValueFold fences its trigger for the o-projection's
    pre-wait reads; here MlaValue is the o-projection's PDL primary, so it gets the same fence before its trigger.
    MlaFoldMerge needs none: its dependent (MlaValue) reads only weights before its wait, and it triggers after its
    own wait. The other modes leave MlaValue as shipped."""
    partial = fake((rows, splits, heads, 512), None, torch.float32, 16)
    stats = fake((rows, splits, heads, 2), None, torch.float32, 8)
    view = fake((rows, heads, 512), (16 * 512, 512, 1))  # the 16-head latent view
    with torch.accelerator.device_index(device):
        merge = cute.compile(
            MlaFoldMerge(heads, rows, splits), partial, stats, cutlass.Float32(1.0), view, make_fake_stream()
        )
    return _SplitFold(merge, _padded_value(rows, pitch, weight_layout, device, q, heads, early == "fence"))


@dataclass(frozen=True)
class SparseVerifyStep(VerifyStepView):
    causal_lengths: torch.Tensor = None
    score_contexts: torch.Tensor = None
    score_schedule: torch.Tensor = None
    topk_plan: torch.Tensor = None
    sparse: int = 0

    def validate(self, *, auxiliary_width=None):
        super().validate(auxiliary_width=auxiliary_width)
        keys = verify_keys(self.query_width, (1,) if self.query_width == 16 else Q8_REQUEST_BATCHES)  # Q4-Q7: Q-port
        if (self.batch_requests, self.query_width, self.row_count) not in keys:
            raise ValueError("sparse verify requires a supported request/query key")
        if self.sparse:
            if self.causal_lengths.shape != (self.row_count,):
                raise ValueError("sparse verify needs one causal length per query")
            if self.score_contexts.shape != (self.batch_requests, self.query_width):
                raise ValueError("sparse verify native score needs int32[Breq,Q] contexts")
            if self.causal_lengths.dtype != torch.int32 or self.score_contexts.dtype != torch.int32:
                raise ValueError("sparse verify causal lengths/contexts must be int32")
            if self.score_schedule is None or self.topk_plan is None:
                raise ValueError("sparse verify requires current schedule/top-k plan")
        if self.sparse not in (0, 1, 2):
            raise ValueError(f"sparse verify label must be 0 (dense), 1 (sparse) or 2 (tsi), got {self.sparse}")


# Verify top-k (``K31MK_VERIFY_TOPK``): ``v2`` (default) = stock topk_transform_paged_v2 (the stock target-verify method,
# deterministic=False, over the stock per-row plan). Caveat: topk_v2 is inexact on tie-heavy rows; it is the stock method, so the
# MK matches stock there.
# ``x47960``: SGLang #47960's exact mode (SGL_TOPK_MODE=1; dsa/x47960, vendored from public sgl-project/sglang b2833196d8 with
# the cu134 build fix: the cluster kernel's staging buffer in dynamic smem, no smem-spill pragma) for <= 16 rows, and above
# 16 rows FlashInfer gvr_2 (dsa/gvr2; THIRD_PARTY_NOTICES.txt names its revision): exact top-2048 over the same
# scores and per-row causal lengths, paged output, called in its next_n form: request q's rows share page-table row q and
# row j sees kv_len[q] - Q + j + 1 keys, kv_len[q] = the request's last row's causal length (the per-row lengths here are
# P + j + 1). Its launchers compile on the first eager call at a geometry (rows, logits row stride); a graph capture before
# that raises (gvr_2 refuses to compile under capture). Bake-off (cold chain at W = 1M, not in situ): exact 96 / 96 cells;
# faster than stock at 1-16 rows from 32K up, stock speed at 64 rows (hence gvr_2 there).
# ``x47v2``: x47960's X47960Select for <= 16 rows, stock topk_v2 above (no gvr_2: the > 16-row calls stay the stock
# method). Use: the request-sharded owner block (8-16 rows). Exact where stock v2 drops true keys on tie-heavy rows.
VERIFY_TOPK = os.environ.get("K31MK_VERIFY_TOPK", "v2")
if VERIFY_TOPK not in ("v2", "x47960", "x47v2"):
    raise ValueError(f"K31MK_VERIFY_TOPK must be v2, x47960 or x47v2, got {VERIFY_TOPK!r}")
# rows at or below which x47960 runs #47960's exact mode; above it gvr_2
X47_MAX_ROWS = 16


# ``K31MK_VERIFY_TOPK_CHAIN`` (default 0 = off): keep the gvr_2 top-k (x47960 above 16 rows) on the DSA layer's PDL chain at
# 2+ requests. Values:
#   d1  gvr_select's next_n kv_len vector (each request's last-row causal length) is built ONCE per verify step (RequestLengths.begin,
#       called by BuiltSparseVerify.run_verify on the step's stream before any layer) instead of a one-CTA .contiguous() copy between
#       every layer's logits and its top-k; that copy is not PDL-launched, so it cut the logits -> top-k programmatic edge. Same
#       values -> bitwise.
#   d2  the gvr_2 decode kernels (dsa/gvr2) execute griddepcontrol.launch_dependents right after griddepcontrol.wait, so
#       the dependent sparse_mla6 launches while the top-k runs (stock topk_v2's form); every dependent still waits before reading the
#       selection (sparse_mla6.cuh pdl_wait). Read at import by those modules. Same kernels' outputs -> bitwise.
#   1   d1 + d2.
VERIFY_TOPK_CHAIN = os.environ.get("K31MK_VERIFY_TOPK_CHAIN", "0")
if VERIFY_TOPK_CHAIN not in ("0", "1", "d1", "d2"):
    raise ValueError(f"K31MK_VERIFY_TOPK_CHAIN must be 0, 1, d1 or d2, got {VERIFY_TOPK_CHAIN!r}")
TOPK_CHAIN_D1 = VERIFY_TOPK_CHAIN in ("1", "d1")


class RequestLengths:
    """K31MK_VERIFY_TOPK_CHAIN d1: the current verify step's per-request kv_len (its last row's causal length), shared by every DSA
    layer's VerifySelection. ``begin(view)`` runs once per verify step (eager call or graph capture) on the step's current stream and
    records the step it was built for; gvr_select reads ``kv`` only for that same step object (else it falls back to its per-layer
    copy, counted in ``misses``).

    Ordering: the copy is an ordinary (non-programmatic) launch on the step stream before any layer runs; each layer's selection
    stream joins the step stream after it (side.wait_stream(main) in PreparedSparseVerifyMla.run), so the copy precedes every
    top-k read in stream order. The next step's copy follows this step's last top-k read in the same way (main joins the selection
    stream in every layer before the sparse core), and a captured graph holds the copy as a node re-run on every replay."""

    def __init__(self, device, capacity):
        self.buf = torch.zeros(capacity, dtype=torch.int32, device=device)
        self.view = self.kv = None
        self.hits = self.misses = 0
        self.logged = False

    def begin(self, view):
        self.view = self.kv = None
        if view.sparse != 1 or view.batch_requests < 2:
            return
        b, q = view.batch_requests, view.query_width
        kv = self.buf[:b]
        kv.copy_(view.causal_lengths.view(b, q)[:, q - 1])
        self.view, self.kv = view, kv


# Verify index query at <= 8 rows (Breq 1 x Q8): index_q.FusedIndexQuery (one PDL kernel: rmsnorm, wq_b, act_quant,
# head gate and the weights_proj gate; the stock chain's method) instead of IndexProjection's stock chain on the
# index stream. ``K31MK_FUSED_INDEX_Q_VERIFY=0`` restores the stock chain.
FUSED_INDEX_Q_VERIFY = os.environ.get("K31MK_FUSED_INDEX_Q_VERIFY", "1") != "0"
# ``K31MK_INDEX_Q_OFF_WRITER`` (default 0 = off): the fused index query waits only for the input projection instead of
# queueing behind the index-K writer (projection, LayerNorm, store); the scoring runs on its own stream and joins the
# writer. 1: the index query on that stream too; 2: the index query on the main stream ahead of the MLA query, so
# the MLA query launches only once every index-query CTA has started. Same kernels and inputs: outputs bitwise equal.
INDEX_Q_OFF_WRITER = int(os.environ.get("K31MK_INDEX_Q_OFF_WRITER", "0"))
# ``K31MK_INDEX_Q_OFF_WRITER_WIDE=1`` (default 0 = off): mode 1 above for the verify steps above 8 rows (Breq 2-8 x Q8), which
# run the stock index-query chain (rmsnorm, wq_b mm, quant + head gate) because the fused index query serves at most 8 rows.
# Off, that chain queues on the index stream behind the index-K writer (projection, split-K reduce, LayerNorm, store); on, it
# runs on its own stream once the input projection and the gate GEMM are done, and the scoring joins the writer. Mode 2 is not
# offered here: it issues the index query on main before main joins the gate stream, and the stock chain reads the gate GEMM's
# output. Same kernels and inputs: outputs bitwise equal (up to the stock top-k's run-dependent slot order).
INDEX_Q_OFF_WRITER_WIDE = int(os.environ.get("K31MK_INDEX_Q_OFF_WRITER_WIDE", "0"))
# ``K31MK_INDEX_Q_MODE_ROWS`` (default empty): per-row override of K31MK_INDEX_Q_OFF_WRITER's mode (1 or 2) for the profiles with a
# fused index query (QF at 1-8 rows, QF-wide at 9-32), as 'rows:mode,...' (e.g. 16:1,32:1); needs K31MK_INDEX_Q_OFF_WRITER=1 or 2. Same kernels and
# inputs as the global mode, only the stream placement moves (bitwise). A 1-GPU c=8-env rig (exploratory): QF-wide at mode 1
# ends the layer earlier than at mode 2 at 16 and 32 rows, later at 24 rows 8K: under
# mode 2 the MLA query waits behind the index query and then overlaps the long-context index logits.
INDEX_Q_MODE_ROWS = {}
for _item in filter(None, os.environ.get("K31MK_INDEX_Q_MODE_ROWS", "").replace(" ", "").split(",")):
    _r, _, _m = _item.partition(":")
    if not (_r.isdigit() and _m in ("1", "2") and 1 <= int(_r) <= 32):
        raise ValueError(f"K31MK_INDEX_Q_MODE_ROWS: want 'rows:mode' items with rows 1-32 and mode 1 or 2, got {_item!r}")
    INDEX_Q_MODE_ROWS[int(_r)] = int(_m)
if INDEX_Q_MODE_ROWS and INDEX_Q_OFF_WRITER not in (1, 2):
    raise ValueError("K31MK_INDEX_Q_MODE_ROWS needs K31MK_INDEX_Q_OFF_WRITER=1 or 2 (the fused index query's query stream)")
_iq_mode_logged = set()
# ``K31MK_IDXK_LN_PDL=1`` (default 0 = off): launch the index-K writer's LayerNorm (FlashInfer's CuTe kernel, the same
# kernel and math stock uses) with PDL, so it no longer waits for the split-K reduce to drain before launching. The
# kernel executes griddepcontrol.wait before its first global read and triggers its dependents after its last store;
# the index-K store that follows waits before every read. Outputs are bitwise equal to the default launch.
IDXK_LN_PDL = os.environ.get("K31MK_IDXK_LN_PDL", "0") == "1"
# ``K31MK_DSA_QF=1`` (default 0 = off; query front): the input projection also computes the index-K rows (indexer.wk)
# and the head-gate logits (indexer.weights_proj) as 160 more BF16 columns of the packed MLA projection (the host packs
# [q_a | kv_a | gate | wk | weights_proj] under the same switch: TP16 3040 rows, TP8 3808), and on sparse verify steps
# of 1-8 rows the fused index query reads the gate from those columns and stores the new tokens' index-K rows itself
# (index_q.cuh kQF). The side-stream index-K writer (projection GEMM, LayerNorm, store) and the gate GEMM leave those
# steps; dense verify steps keep the stock writer. Not bitwise to the default: the FP32 summation orders of the
# two projections and of the LayerNorm change; stock's rounding points are kept.
DSA_QF = os.environ.get("K31MK_DSA_QF", "0") == "1"
QF_EXTRA_ROWS = INDEX_ROWS  # wk + weights_proj
# ``K31MK_DSA_QF_MAX_ROWS=16`` (default 8 = off; with K31MK_DSA_QF=1): the query front also serves the sparse verify steps of
# 9-16 rows (Breq 2 x Q8): index_q.cuh kR = 16, one launch, outputs bitwise equal to the 8-row kernel run on the two row
# halves; the 16-row profile's stock index-K writer and gate GEMM leave those steps as they left the 8-row ones.
DSA_QF_MAX_ROWS = int(os.environ.get("K31MK_DSA_QF_MAX_ROWS", "8"))
if DSA_QF_MAX_ROWS not in (8, 16):
    raise ValueError(f"K31MK_DSA_QF_MAX_ROWS must be 8 or 16, got {DSA_QF_MAX_ROWS}")
# ``K31MK_DSA_QFW=1`` (default 0 = off; with K31MK_DSA_QF=1): QF-wide - the query front also serves the sparse verify steps of
# up to 32 rows that K31MK_DSA_QF_MAX_ROWS leaves to the stock chain (9-32 rows at the default 8; 17-32 at 16): index_qfw.cuh,
# one PDL kernel behind the q in-proj (gate from the projection, index-K rows stored in-kernel), so those profiles' stock
# index-K writer and gate GEMM leave as they left the 8-row ones. NOT bitwise to the stock chain or to QF16: the q_a RMSNorm
# weight is folded into wq_b at load (one 12.6 MB copy per DSA layer) and rstd scales the FP32 product (index_qfw.cuh header).
DSA_QFW = os.environ.get("K31MK_DSA_QFW", "0") == "1"
if DSA_QFW and not DSA_QF:
    raise ValueError("K31MK_DSA_QFW=1 needs K31MK_DSA_QF=1 (the packed projection carries the index-K and gate columns)")
# ``K31MK_DSA_QFW_MAX_ROWS`` (default 32 = every 9-32-row profile; 16 or 24; with K31MK_DSA_QFW=1): QF-wide serves only the
# 9 .. N-row profiles; wider profiles keep the stock chain, or take QFP under K31MK_DSA_QFP (set K31MK_DSA_QFP_MIN_ROWS = N + 1). The
# 1-GPU chain rigs in the c=8 bundle env (exploratory): QFP forked + KST ends the DSA layer earlier than QF-wide at 32 rows (8K,
# 32K, 131K) and at 24 rows (32K, 131K), later at 24 rows 8K and at 16 rows 8K.
DSA_QFW_MAX_ROWS = int(os.environ.get("K31MK_DSA_QFW_MAX_ROWS", "32"))
if DSA_QFW_MAX_ROWS not in (16, 24, 32):
    raise ValueError(f"K31MK_DSA_QFW_MAX_ROWS must be 16, 24 or 32, got {DSA_QFW_MAX_ROWS}")
# ``K31MK_DSA_QFX_LOW=1`` (default 0 = off; with K31MK_DSA_QFW=1): the QF-wide profiles (9 .. K31MK_DSA_QFW_MAX_ROWS
# rows) run the QF-cross kernel (index_qfx.cuh, 2-4 n8 row tiles) in place of index_qfw.cuh: each cluster rank loads one 192-column K
# slice of every q_a row instead of all of q_a, and the head's partials meet over DSMEM. Every row's outputs (FP8 codes, head weights,
# stored index-K) are BITWISE QF-wide's (index_qfx.cuh header; test_k31_index_qfx at 9-32 rows). A 1-GPU harness
# (exploratory, behind an in-proj-shaped GEMM): measured faster per DSA layer at 16 / 24 / 32 rows.
DSA_QFX_LOW = os.environ.get("K31MK_DSA_QFX_LOW", "0") == "1"
if DSA_QFX_LOW and not DSA_QFW:
    raise ValueError("K31MK_DSA_QFX_LOW=1 needs K31MK_DSA_QFW=1 (it replaces the QF-wide kernel on the QF-wide profiles)")
# ``K31MK_DSA_QFP=1`` (default 0 = off; with K31MK_DSA_QF=1): the sparse verify profiles WITHOUT a fused index query (rows above the
# QF / QF-wide range, e.g. 40-128) read the index-K rows and the head gate from the packed projection as the query front does:
# the index-K LayerNorm + store run on the query's stream from projected[:, k_col : k_col + 128] (no index-K writer GEMM), the head
# gate is projected[:, g_col : g_col + 32] (no gate GEMM), and the index query is QFN (indexer.py / index_qfn.py: no q_a RMSNorm
# launch; the wq_b GEMM on the QF-wide fold). Numerics-changing (projection GEMM order for index-K / gate; QFN), KL-gated.
# K31MK_DSA_QFP=2: the same, with the index-K LayerNorm + store on the index stream beside the query GEMM (fork / join inside the step)
# instead of ahead of it on the query stream. K31MK_DSA_QFP_MIN_ROWS (default 2): only profiles of at least this many rows take QFP.
DSA_QFP = int(os.environ.get("K31MK_DSA_QFP", "0"))
DSA_QFP_MIN_ROWS = int(os.environ.get("K31MK_DSA_QFP_MIN_ROWS", "2"))
if DSA_QFP not in (0, 1, 2):
    raise ValueError(f"K31MK_DSA_QFP must be 0, 1 or 2, got {DSA_QFP}")
if DSA_QFP and not DSA_QF:
    raise ValueError("K31MK_DSA_QFP=1 needs K31MK_DSA_QF=1 (the packed projection carries the index-K and gate columns)")
# ``K31MK_DSA_QFP_PAR_MAX_ROWS`` (default 0 = no limit; with K31MK_DSA_QFP=2): only QFP profiles of at most this many rows fork the index-K
# LayerNorm + store onto the index stream; wider QFP profiles run it ahead on the query stream (K31MK_DSA_QFP=1's form). A
# 1-GPU chain bake-off: the forked form ends the layer first at 40-96 rows, the serial one at 104-128 rows
# (MlaQuery's 8-CTA cluster count rises from 18 to 24 there). Each profile runs one of the two gated forms.
DSA_QFP_PAR_MAX_ROWS = int(os.environ.get("K31MK_DSA_QFP_PAR_MAX_ROWS", "0"))
if DSA_QFP_PAR_MAX_ROWS < 0:
    raise ValueError(f"K31MK_DSA_QFP_PAR_MAX_ROWS must be >= 0 (0 = no limit), got {DSA_QFP_PAR_MAX_ROWS}")
# ``K31MK_DSA_QFP_KST=1`` (default 0 = off; with K31MK_DSA_QFP): QFP profiles store this step's index-K rows with ONE warp-per-row
# kernel (index_kst.cuh: LayerNorm + FP8 / MXFP4 quant + store, the query front's index_q.cuh store_index_k row math) instead of FlashInfer's
# LayerNorm launch followed by the store launch, on the same stream as without it (the index stream when forked, the query stream when serial).
# Bytes equal the query front's store of the same rows; vs FlashInfer's LayerNorm only the FP32 summation order differs (KL class, as QF).
# Under K31MK_DSA_REQ_SHARD_OWNQ the index-K branch, not the owner query, is the last input of the logits (trace).
DSA_QFP_KST = os.environ.get("K31MK_DSA_QFP_KST", "0") == "1"
if DSA_QFP_KST and not DSA_QFP:
    raise ValueError("K31MK_DSA_QFP_KST=1 needs K31MK_DSA_QFP=1 or 2 (it replaces QFP's index-K LayerNorm + store)")
# ``K31MK_MLAQ_W=1`` (default 0 = off): sparse verify profiles of at least K31MK_MLAQ_W_MIN_ROWS rows (default 72) compute
# the MLA query and the current KV row with mlaq_w.MlaQueryWide (a norm + KV-insert kernel, a cuBLAS Q_B GEMM and a W_UK + quant kernel,
# stream-ordered on the main stream) instead of launching the cluster MlaQuery (192 one-per-SM CTAs at 128 rows), which shares the SMs
# with the index chain. Same roundings as MlaQuery, FP32 summation orders differ (KL class). On a 1-GPU chain rig
# (8K context) the layer ends earlier than stock at 80-128 rows, together with QFP2 more than with QFP2 alone.
MLAQ_W = os.environ.get("K31MK_MLAQ_W", "0") == "1"
MLAQ_W_MIN_ROWS = int(os.environ.get("K31MK_MLAQ_W_MIN_ROWS", "72"))
# ``K31MK_DSA_QF_G18=1`` (default 0 = off; with K31MK_DSA_QF=1 and K31MK_PROJ_TMEM8=1): the plugin reorders the packed MLA
# projection to [q_a | kv_a | wk | weights_proj | gate] at load and splits the 8-row projection into an 18-cluster TM8 and a
# side-stream head-gate GEMM (mla/head_gate.py); outputs bitwise equal to K31MK_DSA_QF's, at other columns.
# ``K31MK_DSA_DG_VENDOR=1`` (default 0 = off): the verify index logits run dg_logits.py's vendored copy of DeepGEMM's
# paged MQA logits kernel (same instantiation, tensor maps and launch, so bitwise equal logits), launched with PDL: its CTAs
# start while the index query is still running and wait for it in every thread before their first global access. After the
# wait its TMA warps fence the async proxy (their inputs were written with plain stores), and thread 0 of each CTA issues
# fence.acq_rel.gpu and then triggers the top-k launch, which waits for the logits grid to complete before reading. Shapes
# the copy does not serve keep DeepGEMM.
# The TSI scan's local logits (dsa/tsi.py DG_VENDOR) follow the same switch.
DSA_DG_VENDOR = os.environ.get("K31MK_DSA_DG_VENDOR", "0") == "1"
# ``K31MK_DSA_ROW_TABLE_ONCE=1`` (default 0 = off): at 2+ requests the stock top-k needs a per-row page table (row r reads
# request r // q), which VerifySelection.row_table copies from the [requests, pages] table. Every DSA layer of a verify
# step reads the same table, so with this switch the first layer of each step copies it and the other layers reuse that
# copy (RowTableShare) instead of each copying it again between its logits and its top-k. Same table values: bitwise.
DSA_ROW_TABLE_ONCE = os.environ.get("K31MK_DSA_ROW_TABLE_ONCE", "0") == "1"


def qf_columns(heads):
    """(index-K column, head-gate column, packed pitch) of the QF packing for ``heads`` local MLA heads; under
    K31MK_DSA_QF_G18 those of the plugin's reordered weight [q_a | kv_a | wk | weights_proj | gate]."""
    k_col = 2112 if DSA_QF_G18 else 2112 + 128 * heads
    return k_col, k_col + 128, 2112 + 128 * heads + QF_EXTRA_ROWS


# ``K31MK_SPARSE_MLA6=1`` (default 0 = off; L38): at 6 heads per rank (TP16) the sparse core is
# sparse_mla6.cuh (no 6 -> 16 head padding, no query copy, 2048 / K31MK_SPARSE_MLA6_TOKENS split partials) and
# MlaValueFold merges the splits inside the W_UV / gate kernel, instead of trtllm-gen's padded sparse MLA + MlaValue.
# Not bitwise: P stays FP16 (stock E4M3), FP32 partials (stock BF16), other summation orders.
SPARSE_MLA6 = os.environ.get("K31MK_SPARSE_MLA6", "0") == "1"
# K31MK_VERIFY_TOPK_CHAIN d2 lets the selection's dependent launch while the top-k runs; that is safe only for a dependent that executes
# griddepcontrol.wait before reading the selection. sparse_mla6 does (sparse_mla6.cuh pdl_wait before every read); the trtllm-gen core
# (K31MK_SPARSE_MLA6=0) is closed source and not shown to: refuse that pairing (fail closed).
if VERIFY_TOPK_CHAIN in ("1", "d2") and not SPARSE_MLA6:
    raise ValueError("K31MK_VERIFY_TOPK_CHAIN=1 / d2 needs K31MK_SPARSE_MLA6=1 (the only sparse core shown to wait before its reads)")
# ``K31MK_MLAVALUE_FOLD_EARLY`` (default 0 = off; lane parity P2): under K31MK_SPARSE_MLA6, MlaValueFold releases its
# PDL dependents (the o-projection) right after its griddepcontrol.wait instead of after the split merge, so the
# o-projection's weight prologue overlaps the merge. "1": the trigger alone. "f": wait -> fence.acq_rel.gpu -> trigger
# (the K-19 fix on this hop: OprojFused prefetches the AttnRes prefix /
# blocks before its own wait, so the trigger releases what the fold's wait acquired). Dependents read the fold's outputs
# only after their own wait. Bitwise in every mode. Any other value fails at import.
MLAVALUE_FOLD_EARLY = {"0": False, "1": True, "f": "fence"}[os.environ.get("K31MK_MLAVALUE_FOLD_EARLY", "0")]
# ``K31MK_FOLD_SPLIT_ROWS=N`` (default 0 = off; needs K31MK_SPARSE_MLA6=1): at N or more verify rows the L38 fold runs
# as two kernels, MlaFoldMerge (MlaValueFold's split merge, once per (row, head)) and then the shipped MlaValue over
# 16-row tiles. W_UV is then read once per 16 rows instead of once per row, and each split partial once instead of
# 4 times, for one more PDL boundary. Bitwise equal to MlaValueFold. Below N rows MlaValueFold runs unchanged.
FOLD_SPLIT_ROWS = int(os.environ.get("K31MK_FOLD_SPLIT_ROWS", "0"))
# ``K31MK_TSI=1`` (default 0 = off): token-split indexer (dsa/tsi.py) at TP16 / DCP1. Each
# rank scores 1/16 of the index-K pages and the ranks merge their local top-2048 into the exact global top-2048 (score
# desc, position asc; slots sorted by position). It runs only on steps the host labels "tsi" (step.sparse == 2), which
# tsi_graph_label picks above the context gate; every other step keeps DCP1's DeepGEMM logits + topk_v2.
TSI_KERNEL = os.environ.get("K31MK_TSI", "0") == "1"
# ``K31MK_DSA_REQ_SHARD=1`` (default 0 = off; form R1): request-sharded selection at DCP1
# (dsa/req_shard.py): rank r runs the stock selection (DeepGEMM logits + topk_v2) for its owner block of ceil(B / TP)
# requests and one all-gather over the TP group hands every rank every row's selection. It uses the host's third verify
# graph (label "tsi", step.sparse == 2; the host captures it when TSI below is true), picked by tsi_graph_label above
# its own gate: >= K31MK_DSA_REQ_SHARD_MIN_REQ requests (default 4) and every request's context >=
# K31MK_DSA_REQ_SHARD_MIN_CONTEXT (default 65,536: mine, from P5's measured cells and BIX-R1's exchange; re-fit on data).
# K31MK_TSI and K31MK_DSA_REQ_SHARD are exclusive (one selection fills the third graph).
REQ_SHARD = os.environ.get("K31MK_DSA_REQ_SHARD", "0") == "1"
REQ_SHARD_MIN_REQ = int(os.environ.get("K31MK_DSA_REQ_SHARD_MIN_REQ", "4"))
REQ_SHARD_MIN_CONTEXT = int(os.environ.get("K31MK_DSA_REQ_SHARD_MIN_CONTEXT", "65536"))
# ``K31MK_DSA_REQ_SHARD_SUM_CONTEXT`` (tokens; default 0 = off = the shortest-request rule above): gate on the SUM of the
# step's contexts instead (at served contexts the shortest-request rule engages R1 on
# only 9-40 % of Breq 9-16 steps; a sum rule near 1,048,576 tokens engages it on ~87-100 %, ESTIMATE; re-fit on data).
REQ_SHARD_SUM_CONTEXT = int(os.environ.get("K31MK_DSA_REQ_SHARD_SUM_CONTEXT", "0"))
# ``K31MK_DSA_REQ_SHARD_WITH_TSI=1`` (default 0 = off = the exclusivity below): K31MK_TSI and K31MK_DSA_REQ_SHARD together. The
# third verify graph then runs TSI at rows <= TSI_RS_SPLIT_ROWS (every TSI_GATE_ROWS / TSI_WIDE_ROWS width) under TSI's label and
# the request-sharded selection above it under R1's label (dsa/req_shard.py SplitSelection; capacity rows are fixed per captured
# graph, so the state is picked by rows). R1 never acts at rows <= TSI_RS_SPLIT_ROWS then, whatever its MIN_REQ.
REQ_SHARD_WITH_TSI = os.environ.get("K31MK_DSA_REQ_SHARD_WITH_TSI", "0") == "1"
TSI_RS_SPLIT_ROWS = 32
if REQ_SHARD_WITH_TSI and not (TSI_KERNEL and REQ_SHARD):
    raise ValueError("K31MK_DSA_REQ_SHARD_WITH_TSI=1 needs K31MK_TSI=1 and K31MK_DSA_REQ_SHARD=1")
if TSI_KERNEL and REQ_SHARD and not REQ_SHARD_WITH_TSI:
    raise ValueError("K31MK_TSI and K31MK_DSA_REQ_SHARD are exclusive (both fill the host's third verify graph) unless "
                     "K31MK_DSA_REQ_SHARD_WITH_TSI=1")
# ``K31MK_DSA_REQ_SHARD_OWNQ=1`` (default 0 = off; needs K31MK_DSA_REQ_SHARD=1): on a request-sharded step the index query (the
# wq_b GEMM on the fold + the quant / head-weight kernel) runs for this rank's owner rows only - the rows its selection reads
# (dsa/req_shard.py owner_rows: ceil(B / TP) requests x Q, 8 rows at Breq <= 16 on TP16) - instead of all rows. Profiles whose
# index query is QFN or QFP (index_qfn.qfn_quant_head_gate) take it when the owner rows are one contiguous range; any other
# profile declines with a log line, and steps that are not request-sharded run the full index query. The owner rows land at
# their own row offsets of fixed per-profile buffers, so the selection's row slices (K31MK_DSA_REQ_SHARD_SLICE) or gathers read
# them unchanged; the other rows of those buffers are not read on such a step. The index-K store stays full (every rank holds
# every request's index-K at DCP1). Numerics: an M = 8 GEMM may pick another cuBLAS kernel than M = rows, so the owner rows'
# accumulation order can change: KL class (FP64 gate + in-boot KL check).
# =2: QFP profiles with <= 8 owner rows run the owner rows through the fused query-front kernel instead (index_q.cuh run_qf, the
# c=1 / c=8 query front: q_a RMSNorm + wq_b product + quant + head gate from the projection in one PDL kernel; every slot -1, so it
# stores no index-K); QFN profiles keep form 1. Form 1 at M = 8 runs cuBLAS split-K + a reduce kernel (chain read).
# Under K31MK_DSA_IDX_Q=fp4 the owner rows take the MXFP4 index query into fixed (idx_fp4.Fp4Query, weights) buffers: form 1 =
# idx_fp4_q.qfn_quantize_query on the owner rows' fold GEMM, form 2 = index_q.cuh's kQ4 query front (run_qf_q4, slots -1); same class (KL).
REQ_SHARD_OWNQ = int(os.environ.get("K31MK_DSA_REQ_SHARD_OWNQ", "0"))
if REQ_SHARD_OWNQ not in (0, 1, 2):
    raise ValueError(f"K31MK_DSA_REQ_SHARD_OWNQ must be 0, 1 or 2, got {REQ_SHARD_OWNQ}")
if REQ_SHARD_OWNQ and not REQ_SHARD:
    raise ValueError("K31MK_DSA_REQ_SHARD_OWNQ needs K31MK_DSA_REQ_SHARD=1")
# ``K31MK_DSA_REQ_SHARD_OWNQ_MIN_ROWS`` (default 0 = every request-sharded QFN / QFP profile): only profiles of at least this many rows take
# OWNQ. On a 1-GPU chain (131K context) form 2 ends the layer earlier than the best full QFP form at 96-128 rows and
# later than the full forked QFP query at 64 rows.
REQ_SHARD_OWNQ_MIN_ROWS = int(os.environ.get("K31MK_DSA_REQ_SHARD_OWNQ_MIN_ROWS", "0"))
if REQ_SHARD_OWNQ_MIN_ROWS < 0:
    raise ValueError(f"K31MK_DSA_REQ_SHARD_OWNQ_MIN_ROWS must be >= 0, got {REQ_SHARD_OWNQ_MIN_ROWS}")
# ``K31MK_DSA_IQ_PIPE=1`` (default 0 = off): the RS-OWNQ form-2 owner query (8 rows, index_q.cuh run_qf) runs on index_qp.cuh, the
# IQ-PIPE form of the same kernel (w_qa read as 16-byte vectors, BF16 packed in pairs, wq_b fragments in registers before griddepcontrol.wait; same
# run_qf contract and output bytes, gate iqp3g). The c=1 / c=8 query front takes the knob through index_q.FusedIndexQuery.
# FP8 query only: under K31MK_DSA_IDX_Q=fp4 the owner query keeps index_q.cuh's kQ4 front (IQ-PIPE has no kQ4 instance; declined, logged).
DSA_IQ_PIPE = os.environ.get("K31MK_DSA_IQ_PIPE", "0") == "1"
# The host reads TSI as "a third verify graph per capacity exists" (kimi_k31_verify_dsa_binding).
TSI = TSI_KERNEL or REQ_SHARD
# ``K31MK_DSA_IDX_FP4=1`` (default 0 = off;, dsa/idx_fp4.py): MXFP4 index-K pages (68 B/token, the host's
# K31HT_PREC_IDX_FP4 pool) + MXFP4 index query; the verify writer, the stock-chain index query and VerifySelection's logits
# switch to idx_fp4. Every path that still reads or writes FP8 index-K bytes is refused here (fail closed).
# With ``K31MK_DSA_IDX_FP4_FP8Q=1`` the index query stays the served FP8 one (stock chain or the fused index query, which writes
# FP8 q and gates only) and the logits run dg_logits_fp4k.py (FP8 query x MXFP4 index-K); only the writer and the logits change.
# The indexer-query toggle ``K31MK_DSA_IDX_Q`` (dsa/idx_fp4_knob.py) picks it: fp8 (the default under K31MK_DSA_IDX_FP4=1)
# = the FP8-query form above; fp4 = the MXFP4 index query FUSED into its producer on every path (dsa/idx_fp4_q.py: the fused index query
# and the query front run index_q.cuh's kQ4 instances, QFN / QFP rows the MXFP4 QFN kernel, the stock chain idx_fp4.quantize_query in place
# of indexer_q_quant_head_gate) and the MXFP4 logits (Q8: dg_logits_fp4.py's vendored PDL kernel when K31MK_DSA_DG_VENDOR=1 or
# K31MK_DSA_IDX_FP4_VENDOR=1, else the stock DeepGEMM fp8_fp4 launch).
if IDX_FP4:
    from . import idx_fp4  # noqa: E402  (imported only when the switch is on)

    # Under K31MK_DSA_IDX_FP4_FP8Q the query front (K31MK_DSA_QF) stays allowed: index_q.cuh run_qf stores MXFP4 index-K rows when
    # handed the 68 B pages (kKFP4), and QFP's LayerNorm + store writes MXFP4 (_index_query_qfp). QF-wide (index_qfw.cuh) stores FP8
    # in-kernel and is refused.
    # Under K31MK_DSA_IDX_Q=fp4 the fused index query and the query front run their kQ4 (MXFP4-query) instances. The TSI scan (K31MK_TSI)
    # runs the MXFP4-page logits over each rank's local pages (dsa/tsi.py TsiState.scan -> idx_fp4_q.paged_logits).
    _fp8_paths = [name for name, on in (("K31MK_FUSED_INDEX_Q_VERIFY", FUSED_INDEX_Q_VERIFY and not (IDX_FP4_FP8Q or IDX_FP4_FP4Q)),
                                        ("K31MK_DSA_QF", DSA_QF and not (IDX_FP4_FP8Q or IDX_FP4_FP4Q)), ("K31MK_DSA_QFW", DSA_QFW)) if on]
    if _fp8_paths:
        raise ValueError(f"K31MK_DSA_IDX_FP4=1: {_fp8_paths} read or write FP8 index-K bytes (not ported); set them to 0")
# ``K31MK_DSA_IDX_FP4_VENDOR=1`` (default 0 = off; needs K31MK_DSA_IDX_FP4=1): the Q8 MXFP4 verify logits run dg_logits_fp4.py, the FP4
# instantiation of dg_logits.cuh's vendored kernel (PDL launch, the same synchronization additions as K31MK_DSA_DG_VENDOR's FP8 copy);
# other query widths keep idx_fp4.paged_logits' stock DeepGEMM launch. Bitwise equal logits (test_k31_dg_logits_fp4.py).
IDX_FP4_VENDOR = os.environ.get("K31MK_DSA_IDX_FP4_VENDOR", "0") == "1"
if IDX_FP4_VENDOR and not IDX_FP4:
    raise ValueError("K31MK_DSA_IDX_FP4_VENDOR=1 needs K31MK_DSA_IDX_FP4=1")
if IDX_FP4_VENDOR and IDX_FP4_FP8Q:
    raise ValueError("K31MK_DSA_IDX_FP4_VENDOR (the MXFP4-query logits) and K31MK_DSA_IDX_FP4_FP8Q are exclusive")
# The fp4 setting's Q8 logits follow the served FP8 path's vendoring (K31MK_DSA_DG_VENDOR): the vendored PDL kernel when it is on.
IDX_FP4_Q8_VENDOR = IDX_FP4_VENDOR or (IDX_FP4_FP4Q and DSA_DG_VENDOR)
# Context gate (tokens). Default = the rows-8 crossover of TSI vs DCP1, measured on the target GPU
# (live 4-GPU TSI chain vs the DCP1 chain at two contexts),
# linearly interpolated and rounded up to a page:
# 469,696. TSI is slower below it. Rows 16 (compute-only) cross between 128K and 256K, so this gate is conservative
# there; rows 32 is unmeasured and never gated on (TSI_GATE_ROWS).
TSI_MIN_CONTEXT = int(os.environ.get("K31MK_TSI_MIN_CONTEXT", "469696"))
TSI_GATE_ROWS = (8, 16)
# ``K31MK_TSI_WIDE=1`` (default 0 = off = the rule above): steps of 2 or more requests
# gate on the SUM of the requests' contexts instead of the shortest one, at rows TSI_WIDE_ROWS (Breq 2-4). The DCP1
# selection cost follows the summed context at Breq 3 / 4 (in-situ round fit, a max-context term adds nothing), and
# TSI splits every request's scan, so the sum decides. Default K31MK_TSI_SUM_CONTEXT = 393,216 is an ESTIMATE
# (best or near-best over the c=8 B3 / B4 rounds); re-fit it from the arm. Breq 1 keeps the rule
# above. A Breq 3 step replays the capacity-4 graph: its padded 4th request runs TSI on its fill-value context.
TSI_WIDE = os.environ.get("K31MK_TSI_WIDE", "0") == "1"
TSI_SUM_CONTEXT = int(os.environ.get("K31MK_TSI_SUM_CONTEXT", "393216"))
# ``K31MK_TSI_WIDE_ROWS`` (default "16,24,32" = the tuple above, the default behaviour): comma list of
# the verify row counts that take the sum gate under K31MK_TSI_WIDE. Breq 5-8 (40 / 48 / 56 / 64 rows) otherwise run the DCP1 selection
# (DeepGEMM logits + topk_v2 over every request's full context on every rank), whose cost follows the summed context.
# TsiState already holds max(widths) rows (cap = max(widths)), so wider rows need no new buffer. Each value: a multiple of 8 in [16, 64].
_TSI_WIDE_ROWS_RAW = os.environ.get("K31MK_TSI_WIDE_ROWS", "16,24,32")
try:
    TSI_WIDE_ROWS = tuple(sorted({int(_v) for _v in _TSI_WIDE_ROWS_RAW.split(",") if _v.strip()}))
except ValueError:
    raise ValueError(f"K31MK_TSI_WIDE_ROWS must be a comma list of integers, got {_TSI_WIDE_ROWS_RAW!r}") from None
if not TSI_WIDE_ROWS or any(_v % 8 or not 16 <= _v <= 64 for _v in TSI_WIDE_ROWS):
    raise ValueError(f"K31MK_TSI_WIDE_ROWS values must be multiples of 8 in [16, 64], got {_TSI_WIDE_ROWS_RAW!r}")


def tsi_graph_label(seq_lens, query_width):
    """Host graph policy hook (kimi_k31_verify_dsa_binding): True when this verify step should replay the "tsi" graph:
    K31MK_TSI on, rows in TSI_GATE_ROWS, and EVERY request's context >= TSI_MIN_CONTEXT (the shortest request decides;
    the crossover was measured per request context). Under K31MK_TSI_WIDE, a step of >= 2 requests instead needs rows
    in TSI_WIDE_ROWS and the sum of its contexts >= TSI_SUM_CONTEXT."""
    if int(query_width) in SHORT_QUERY_WIDTHS:  # Q-port stage 1 (K31MK_VERIFY_QPORT): TSI / request sharding stay at Q8
        return False
    rows = (0 if seq_lens is None else seq_lens.numel()) * int(query_width)
    if REQ_SHARD and not (REQ_SHARD_WITH_TSI and rows <= TSI_RS_SPLIT_ROWS):
        from .req_shard import req_shard_label

        return req_shard_label(seq_lens, REQ_SHARD_MIN_REQ, REQ_SHARD_MIN_CONTEXT, REQ_SHARD_SUM_CONTEXT)
    if not TSI or seq_lens is None or seq_lens.numel() == 0:  # (TSI == TSI_KERNEL here; tests patch TSI)
        return False
    if TSI_WIDE and seq_lens.numel() >= 2:
        return seq_lens.numel() * query_width in TSI_WIDE_ROWS and int(seq_lens.sum()) >= TSI_SUM_CONTEXT
    return seq_lens.numel() * query_width in TSI_GATE_ROWS and int(seq_lens.min()) >= TSI_MIN_CONTEXT


def _build_req_shard(weights, widths, device, query_width, deadline_s):
    """The model's request-sharded selection over the TP group (K31MK_DSA_REQ_SHARD) and the DSA layers in order."""
    from sglang.srt.runtime_context import get_parallel

    from sgl_nvidia_mega.kimi_k31.k3.spec import _deadline_ns

    from .req_shard import ReqShardState

    parallel = get_parallel()
    tp, rank = int(parallel.tp_size), int(parallel.tp_rank)
    if int(getattr(parallel, "attn_dcp_size", 1)) != 1:
        raise ValueError("K31MK_DSA_REQ_SHARD serves DCP1 (every rank holds every request's index-K)")
    from .req_shard import AG

    if AG == "fused" and not (SPARSE_MLA6 and tp == 16):
        raise ValueError("K31MK_DSA_REQ_SHARD_AG=fused: the sparse core polls the slots itself, so it needs K31MK_SPARSE_MLA6=1 "
                         "at 6 heads per rank (TP16)")
    dsa = sorted(record.layer_id for record in weights.layers if record.attention_kind == 0)
    return ReqShardState(parallel.tp_group, tp, rank, device, widths, query_width, layers=len(dsa),
                         deadline_ns=_deadline_ns(deadline_s)), dsa


def _build_tsi(weights, widths, device, query_width):
    """The model's TSI state over the TP group (collective: every TP rank builds it) and the DSA layers in order."""
    from sglang.srt.runtime_context import get_parallel

    from .tsi import TsiComm, TsiState

    parallel = get_parallel()
    tp, rank = int(parallel.tp_size), int(parallel.tp_rank)
    dcp = int(getattr(parallel, "attn_dcp_size", 1))
    if tp != 16 or dcp != 1:
        raise ValueError(f"K31MK_TSI serves TP16 / DCP1 (the measured shape), got TP{tp} / DCP{dcp}")
    if rank == 0 and (TSI_WIDE or _TSI_WIDE_ROWS_RAW != "16,24,32"):
        # One build-time line so a boot's log shows the TSI_WIDE form it serves, and every decline (word inert without
        # K31MK_TSI_WIDE=1; rows above the TSI capacity never replay the tsi graph). Per-step engagement stays a profile read.
        above = [r for r in TSI_WIDE_ROWS if r > max(widths)]
        logger.info(
            "K31MK_TSI_WIDE=%d: sum gate %d at rows %s (K31MK_TSI_WIDE_ROWS=%r), TSI cap %d rows%s%s",
            int(TSI_WIDE), TSI_SUM_CONTEXT, TSI_WIDE_ROWS, _TSI_WIDE_ROWS_RAW, max(widths),
            "" if TSI_WIDE else "; DECLINED: K31MK_TSI_WIDE_ROWS is inert without K31MK_TSI_WIDE=1",
            f"; DECLINED rows above the cap: {above}" if above else "",
        )
    dsa = sorted(record.layer_id for record in weights.layers if record.attention_kind == 0)
    comm = TsiComm.from_group(parallel.tp_group.device_group, device, tp, rank, cap=max(widths), layers=len(dsa))
    return TsiState(comm, max(widths), -(-weights.context_length // 64) + 16, query_width), dsa


def ownq_range(rows, query_width, world, rank):
    """(first row, row count) of rank ``rank``'s owner rows on a request-sharded step of ``rows`` verify rows (req_shard.owner_rows),
    or None when they are not one contiguous range (K31MK_DSA_REQ_SHARD_OWNQ)."""
    from .req_shard import owner_rows

    own = owner_rows(rows // query_width, query_width, world, rank)
    return (own[0], len(own)) if own == list(range(own[0], own[0] + len(own))) else None


_ownq_logged = set()


def _enable_ownq(core, state, mode=None):
    """K31MK_DSA_REQ_SHARD_OWNQ on every profile of ``core`` that the request-sharded selection ``state`` serves (a ReqShardState, or
    the request-sharded half of a SplitSelection: rows above its split) with a QFN / QFP index query (see REQ_SHARD_OWNQ). ``mode``:
    1 or 2 (default: the knob)."""
    from .req_shard import ReqShardState, SplitSelection

    mode = REQ_SHARD_OWNQ if mode is None else int(mode)

    for rows, profile in core.profiles.items():
        rs = state
        if isinstance(state, SplitSelection):
            if rows <= state.split_rows:  # TSI serves these rows: every row's query is scored
                continue
            rs = state.rs
        if not isinstance(rs, ReqShardState) or rows not in rs.widths:
            continue
        ip = profile.index_projection
        if profile.qfp is not None:
            form = "QFP"
        elif profile.fused_index_q is None and profile.qf is None and ip.folded is not None and rows >= 2:
            form = "QFN"
        else:
            form = None
        rng = ownq_range(rows, rs.q, rs.world, rs.rank)
        why = ("below K31MK_DSA_REQ_SHARD_OWNQ_MIN_ROWS" if rows < REQ_SHARD_OWNQ_MIN_ROWS
               else "the index query is not QFN / QFP" if form is None else "owner rows are not one contiguous range" if rng is None
               else "the owner rows are every row" if rng[1] >= rows else None)
        if why is not None:
            if (rows, why) not in _ownq_logged:
                _ownq_logged.add((rows, why))
                logger.info("DSA RS-OWNQ declined: rows %d (%s)", rows, why)
            continue
        lo, n = rng
        device = ip.query.device
        if IDX_FP4_FP4Q:  # K31MK_DSA_IDX_Q=fp4: the MXFP4 owner-rows query (q4 + scale words); rows other than the owner's stay zero
            q8 = idx_fp4.Fp4Query(torch.zeros((rows, 32, 64), device=device, dtype=torch.int8),
                                  torch.zeros((rows, 32), device=device, dtype=torch.int32))
        else:
            q8 = torch.zeros((rows, 32, 128), device=device, dtype=torch.float8_e4m3fn)  # rows other than the owner's stay zero
        weights = torch.zeros((rows, 32, 1), device=device, dtype=torch.float32)
        fused = noslots = None
        if mode == 2 and form == "QFP" and n <= 8:  # the fused query-front kernel; JIT-loaded here, never inside a graph capture
            if IDX_FP4_FP4Q:  # index_q.cuh kQ4 query front (run_qf_q4): same kernel with the MXFP4 quantizer in its epilogue
                from .idx_fp4_q import Fp4FusedIndexQuery

                fused = Fp4FusedIndexQuery(ip.weights, n)
            else:
                from .index_q import FusedIndexQuery

                fused = FusedIndexQuery(ip.weights, n)
            noslots = torch.full((n,), -1, dtype=torch.int64, device=device)  # negative slots: no index-K store
            form = "QFP fused query-front"
            if DSA_IQ_PIPE and not IDX_FP4_FP4Q:  # K31MK_DSA_IQ_PIPE: the owner query on index_qp.cuh (index_q's run_qf contract, same bytes)
                from .index_qp import module as iqp_module

                fused.module = iqp_module()
                form += " (IQ-PIPE)"
            elif DSA_IQ_PIPE:  # IQ-PIPE is the FP8 query only: the fp4 setting keeps Fp4FusedIndexQuery's kQ4 front
                form += " (IQ-PIPE declined: K31MK_DSA_IDX_Q=fp4)"
        if IDX_FP4_FP4Q:
            form += ", MXFP4"
        profile.ownq = (lo, n, q8, weights, fused, noslots)
        if (rows, form) not in _ownq_logged:
            _ownq_logged.add((rows, form))
            logger.info("DSA RS-OWNQ engaged: rows %d, owner rows %d-%d of rank %d / %d (%s index query on %d rows on request-sharded steps)",
                        rows, lo, lo + n - 1, rs.rank, rs.world, form, n)


def _layernorm_pdl(out, x, w):
    """``flashinfer.norm.layernorm(x, ...)`` into ``out``, launched with PDL (CuTe DSL kernel only)."""
    from flashinfer.norm import _use_cuda_norm
    from flashinfer.norm.kernels import layernorm_cute

    if _use_cuda_norm(x.device):
        raise RuntimeError("K31MK_IDXK_LN_PDL=1 needs FlashInfer's CuTe LayerNorm (the CUDA JIT one has no PDL)")
    layernorm_cute(out, x, w.norm, w.bias, w.eps, enable_pdl=True)
    return out


class RowTableShare:
    """The per-row page table of the current verify step, shared by every DSA layer's VerifySelection
    (K31MK_DSA_ROW_TABLE_ONCE). BuiltSparseVerify.run_verify advances ``step`` once per verify step (eager call or graph
    capture); the first layer that needs the table in a step copies it and records (key, rows); later layers whose key
    matches return the recorded rows. The key holds the step number and the table's address, shape and row stride, so a
    new step, or a layer passing another table, copies again.

    Ordering: at 2+ requests every layer's selection runs on one per-device stream (the index stream with index_q_mode 0
    above 8 rows, the query stream under K31MK_INDEX_Q_OFF_WRITER_WIDE=1; a graph runs one row count, so one of them), so the
    owner's copy precedes the later layers' top-k reads in stream order. The copy is a launch without the
    programmatic-serialization attribute: it starts only after all earlier work on that stream has completed, so the next
    step's copy cannot overwrite rows a previous top-k is still reading."""

    def __init__(self):
        self.step, self.key, self.rows = 0, None, None


def idx_fp4_projection():
    """IndexProjection whose query stage ends in the MXFP4 q / head-gate kernel (K31MK_DSA_IDX_FP4): the stock BF16 rmsnorm and
    wq_b GEMM rounding points are kept; idx_fp4.quantize_query replaces act_quant / indexer_q_quant_head_gate."""

    class Fp4IndexProjection(IndexProjection):
        def __init__(self, weights, rows=1):
            super().__init__(weights, rows)
            self.fp4 = idx_fp4.QueryBuffers(rows, weights.query.device)

        def project_query(self, projected):
            if self.folded is not None:  # K31MK_DSA_QFN rows: the norm-folded GEMM, then the MXFP4 QFN kernel (rstd + quantizer)
                from .idx_fp4_q import qfn_quantize_query

                torch.mm(projected[:, :1536], self.folded.T, out=self.query.view(-1, 4096))
                return qfn_quantize_query(self.query, projected, self.gate, self.weights.epsilon, self.fp4)
            from sgl_kernel import rmsnorm

            w = self.weights
            normalized = rmsnorm(projected[:, :1536], w.query_norm, w.epsilon)
            torch.mm(normalized, w.query.T, out=self.query.view(-1, 4096))
            from .idx_fp4_q import quantize_query_served

            return quantize_query_served(self.query, self.gate, self.fp4)

    return Fp4IndexProjection


class VerifySelection:
    def __init__(self, device, rows):
        self.gvr = self.x47 = None
        x47_rows = VERIFY_TOPK == "x47960" and rows <= X47_MAX_ROWS
        # import (no compile) at prepare time; launchers compile on the first eager call
        if VERIFY_TOPK == "x47960" and rows > X47_MAX_ROWS:
            from .gvr2 import gvr2_topk_host

            self.gvr = gvr2_topk_host
        if x47_rows:  # JIT-built here (prepare time), never inside a graph capture
            from .x47960 import X47960Select

            self.x47 = X47960Select(device, rows)
        if VERIFY_TOPK == "x47v2" and rows <= 16:  # X47960Select at <= 16 rows only; stock v2 above
            from .x47960 import X47960Select

            self.x47 = X47960Select(device, rows)
        self.rows = rows
        self.selected = torch.empty(rows, 2048, device=device, dtype=torch.int32)
        self.row_tables = {}
        self.share = None  # a RowTableShare under K31MK_DSA_ROW_TABLE_ONCE, set by BuiltSparseVerify
        self.req_lens = None  # a RequestLengths under K31MK_VERIFY_TOPK_CHAIN d1, set by BuiltSparseVerify
        if IDX_FP4_Q8_VENDOR:  # the Q8 MXFP4 kernel, JIT-built at prepare time, never inside a graph capture
            from .dg_logits_fp4 import build as build_fp4

            build_fp4(device.index if device.index is not None else torch.cuda.current_device(), 8)
        if IDX_FP4_FP4Q:  # the fp4 setting's MXFP4 QFN kernel (QFN / QFP rows), JIT-built at prepare time, never inside a graph capture
            from .idx_fp4_q import build_qfn

            build_qfn()
        if IDX_FP4_FP8Q:  # FP8 query x MXFP4 index-K for every served Q but legacy Q16 (prepare time; a capture-time build fails closed)
            from ..verify_contract import served_query_widths
            from .dg_logits_fp4k import build as build_fp4k

            for width in served_query_widths():
                if width != 16:
                    build_fp4k(device.index if device.index is not None else torch.cuda.current_device(), width)
        if DSA_DG_VENDOR:  # the Q8 kernel, JIT-built at prepare time, never inside a graph capture
            from .dg_logits import build

            build(device.index if device.index is not None else torch.cuda.current_device(), 8)

    def row_table(self, table, q):
        """Stock top-k's per-row page table: row r reads request r // q."""
        b, pages = table.shape
        if b == 1:
            return table.expand(q, pages)  # row stride 0: no copy
        share = self.share
        if share is not None:
            key = (share.step, table.data_ptr(), b, pages, table.stride(0), q)
            if share.key == key:
                return share.rows
        rows = self.row_tables.get(pages)
        if rows is None:
            rows = self.row_tables[pages] = table.new_empty((b * q, pages))
        rows.view(b, q, pages).copy_(table.unsqueeze(1).expand(b, q, pages))
        if share is not None:
            share.key, share.rows = key, rows
        return rows

    def run(self, query, gates, cache, table, step):
        import deep_gemm
        from sglang.kernels.ops.attention.dsa.paged_mqa_logits import _restore_row_stride

        if IDX_FP4:  # MXFP4 pages + MXFP4 query (dsa/idx_fp4.py): DeepGEMM's fp8_fp4 paged logits, same score layout
            if IDX_FP4_FP8Q:  # FP8 query x MXFP4 pages (dg_logits_fp4k.py)
                from .dg_logits_fp4k import verify_logits as fp4_logits
            elif IDX_FP4_Q8_VENDOR and step.query_width == 8:
                from .dg_logits_fp4 import verify_logits as fp4_logits
            else:
                fp4_logits = idx_fp4.paged_logits
            scores = fp4_logits(
                query, gates, cache, step.score_contexts, table, step.score_schedule, table.shape[1] * 64,
                step.batch_requests, step.query_width,
            )
            scores = _restore_row_stride(scores)
        else:
            vendored = DSA_DG_VENDOR and step.query_width == 8 and gates.dtype == torch.float32
            logits = dg_paged_mqa_logits if vendored else deep_gemm.fp8_paged_mqa_logits
            scores = logits(
                query.view(step.batch_requests, step.query_width, 32, 128), cache.view(-1, 64, 1, 132),
                gates.squeeze(-1), step.score_contexts, table, step.score_schedule, table.shape[1] * 64, clean_logits=False,
            )
            scores = _restore_row_stride(scores)
        # Native scoring computes through the last row's context (same as stock).
        # Top-k masks each row to P+j+1; no future key can become a selected slot.
        return self.topk(scores, step, table)

    def topk(self, scores, step, table):
        """The verify top-k over this step's scores (the dispatch run() ends with; tests call it on their own scores)."""
        if self.x47 is not None:
            return self.x47.run(scores, step.causal_lengths, self.row_table(table, step.query_width), self.selected)
        if self.gvr is not None:
            return self.gvr_select(scores, step, table)
        from sglang.kernels.ops.attention.dsv4.topk import topk_transform_paged_v2

        topk_transform_paged_v2(
            scores, step.causal_lengths, self.row_table(table, step.query_width), self.selected, 64,
            step.topk_plan, deterministic=False,
        )
        return self.selected


    def gvr_select(self, scores, step, table):
        """K31MK_VERIFY_TOPK=x47960 above 16 rows: gvr_2 next_n form over the request-level page table (no per-row table copy)."""
        b, q = step.batch_requests, step.query_width
        if not table.is_contiguous():
            table = table.contiguous()
        # row j of request r sees causal_lengths[r * q + j] = kv_len[r] - q + j + 1 keys: kv_len = each request's last row
        rl = self.req_lens
        if rl is not None and b > 1 and rl.view is step:  # K31MK_VERIFY_TOPK_CHAIN d1: built once for this step
            kv = rl.kv
            rl.hits += 1
            if not rl.logged:
                rl.logged = True
                logger.info("K31MK_VERIFY_TOPK_CHAIN=%s ENGAGED: gvr_2 kv_len once per step (Breq %d, rows %d)",
                            VERIFY_TOPK_CHAIN, b, b * q)
        else:
            if rl is not None and b > 1:
                rl.misses += 1
            kv = step.causal_lengths.view(b, q)[:, q - 1]
            if not kv.is_contiguous():  # b > 1: one small copy per layer call
                kv = kv.contiguous()
        # launch plan: the score buffer width
        self.gvr.run_varlen(
            scores, None, kv, self.selected, next_n=q, top_k=2048, max_seq_len=scores.shape[1],
            page_table=table, page_size=64,
        )
        return self.selected


class PreparedSparseVerifyMla:
    def __init__(self, base, weights, rows, max_context, tsi=None):
        self.base = base
        self.tsi = tsi  # (the model's TsiState, this layer's DSA ordinal) under K31MK_TSI, else None
        self.index_projection = (idx_fp4_projection() if IDX_FP4 and not IDX_FP4_FP8Q else IndexProjection)(weights, rows)
        self.fused_index_q = None
        if IDX_FP4_FP4Q:  # K31MK_DSA_IDX_Q=fp4: the fused index query / query front with the MXFP4 query (index_q.cuh kQ4)
            from .idx_fp4_q import Fp4FusedIndexQuery as FusedIndexQuery
        else:
            from .index_q import FusedIndexQuery
        if FUSED_INDEX_Q_VERIFY and rows <= 8:
            self.fused_index_q = FusedIndexQuery(weights, rows)
        elif FUSED_INDEX_Q_VERIFY and DSA_QF and rows <= DSA_QF_MAX_ROWS:  # K31MK_DSA_QF_MAX_ROWS=16: query front only
            self.fused_index_q = FusedIndexQuery(weights, rows, qf_only=True)
        elif FUSED_INDEX_Q_VERIFY and DSA_QFW and 9 <= rows <= DSA_QFW_MAX_ROWS and DSA_QFX_LOW and not IDX_FP4:  # K31MK_DSA_QFX_LOW: QF-cross
            from .index_qfx import FusedIndexQueryCross

            self.fused_index_q = FusedIndexQueryCross(weights, rows)
        elif FUSED_INDEX_Q_VERIFY and DSA_QFW and 9 <= rows <= DSA_QFW_MAX_ROWS and not IDX_FP4:  # K31MK_DSA_QFW: query front only (run_qf)
            from .index_qfw import FusedIndexQueryWide

            self.fused_index_q = FusedIndexQueryWide(weights, rows)
        self.hidden = None  # this layer's input rows; the fused index query computes the head gate from them
        self.selection = VerifySelection(base.device, rows)
        self.mla = SparseMla(base.device, max_context, rows, base.heads)
        self.mla6 = None
        if SPARSE_MLA6 and base.heads == 6:
            from .sparse_mla6 import SparseMla6

            self.mla6 = SparseMla6(base.device, rows, base.heads)
        if VERIFY_TOPK_CHAIN in ("1", "d2") and self.mla6 is None:
            # The import guard sees only K31MK_SPARSE_MLA6; sparse_mla6 is built only at 6 heads per rank (TP16). At any other
            # head count (TP8: 12) this layer would run the trtllm-gen core behind the early-triggering gvr_2: refuse (fail closed).
            raise ValueError(
                f"K31MK_VERIFY_TOPK_CHAIN={VERIFY_TOPK_CHAIN} needs the sparse_mla6 core, which is built only at 6 heads per rank; "
                f"this layer has {base.heads} heads per rank (use K31MK_VERIFY_TOPK_CHAIN=0 or d1)"
            )
        self.index_stream = _index_stream(base.device)
        self.gate_stream = _gate_stream(base.device)
        if INDEX_Q_OFF_WRITER not in (0, 1, 2):
            raise ValueError(f"K31MK_INDEX_Q_OFF_WRITER must be 0, 1 or 2, got {INDEX_Q_OFF_WRITER}")
        if INDEX_Q_OFF_WRITER_WIDE not in (0, 1):
            raise ValueError(f"K31MK_INDEX_Q_OFF_WRITER_WIDE must be 0 or 1, got {INDEX_Q_OFF_WRITER_WIDE}")
        if self.fused_index_q is not None:
            self.index_q_mode = INDEX_Q_MODE_ROWS.get(rows, INDEX_Q_OFF_WRITER)
            if rows in INDEX_Q_MODE_ROWS and rows not in _iq_mode_logged:
                _iq_mode_logged.add(rows)
                logger.info("DSA index-query mode %d at rows %d (K31MK_INDEX_Q_MODE_ROWS; K31MK_INDEX_Q_OFF_WRITER=%d elsewhere)",
                            self.index_q_mode, rows, INDEX_Q_OFF_WRITER)
        else:  # the stock chain; above 8 rows the writer never packs the gate (SparseVerifyIndexKey): it joins the gate stream
            self.index_q_mode = INDEX_Q_OFF_WRITER_WIDE if rows > 8 else 0
        self.query_stream = _query_stream(base.device) if self.index_q_mode else self.index_stream
        self.packed_index_gate = False
        self.sparse_values = {}
        self.index_cache = None
        self.step = None
        # K31MK_DSA_QF: (index-K column, head-gate column, (LayerNorm weight, bias, eps)), set by BuiltSparseVerify
        self.qf = None
        self.qfp = None  # K31MK_DSA_QFP: the same tuple, for a profile without a fused index query (_enable_qfp)
        self.qfp_par = False  # K31MK_DSA_QFP=2: index-K LayerNorm + store on the index stream
        self.qfp_kst = False  # K31MK_DSA_QFP_KST: that LayerNorm + store as one kernel (index_kst.cuh)
        self.mlaq_w = None  # K31MK_MLAQ_W: an mlaq_w.MlaQueryWide that replaces the MlaQuery launch (_enable_mlaq_w)
        self.ownq = None  # K31MK_DSA_REQ_SHARD_OWNQ: (first owner row, owner rows, q8, weights, fused kernel | None, -1 slots | None)

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def latent(self):
        if self.step is not None and self.step.sparse:
            return self.mla.output[:, 0, : self.base.heads]
        return self.base.latent

    def prepare_cache(self, cache, *, prepare_base=True):
        if prepare_base:
            self.base.prepare_cache(cache)
        for rows, q in self.base.kernels:
            if rows != self.selection.rows:
                continue
            if self.mla6 is not None:
                split = FOLD_SPLIT_ROWS > 0 and rows >= FOLD_SPLIT_ROWS
                self.sparse_values[rows, q] = (_split_fold_value if split else _fold_value)(
                    rows, self.base.pitches[rows], layout(self.base.weights[4]),
                    self.base.device.index, q, self.base.heads, self.mla6.splits, MLAVALUE_FOLD_EARLY,
                )
                continue
            self.sparse_values[rows, q] = _padded_value(
                rows, self.base.pitches[rows], layout(self.base.weights[4]),
                self.base.device.index, q, self.base.heads,
            )
        return self

    def prepare_gate(self, hidden):
        if self.step is not None and self.step.sparse:
            if self.fused_index_q is not None:
                self.hidden = hidden
            else:
                self.index_projection.prepare_gate(hidden)

    def index_query(self, projected, slots=None, own=None):
        """(FP8 index query [rows, 32, 128], FP32 head weights [rows, 32, 1]) of this step's rows. Under K31MK_DSA_QF it
        also stores the rows' index-K at ``slots``. ``own`` (K31MK_DSA_REQ_SHARD_OWNQ, a request-sharded step): this profile's
        ``ownq``; only the owner rows of the returned tensors are computed (QFN / QFP profiles only, see _enable_ownq)."""
        if self.qf is not None:
            k_col, g_col, norm = self.qf
            return self.fused_index_q.run_qf(projected, k_col, g_col, slots, self.index_cache, norm)
        if self.qfp is not None:
            return self._index_query_qfp(projected, slots, own)
        if self.fused_index_q is not None:
            return self.fused_index_q.run(self.hidden, projected)
        if own is not None:  # QFN: the head gate from the gate GEMM (all rows, gate stream), the query for the owner rows
            return self._index_query_own(projected, self.index_projection.gate, own)
        return self.index_projection.project_query(projected)

    def _index_query_own(self, projected, gate, own):
        """K31MK_DSA_REQ_SHARD_OWNQ: the QFN index query (fold GEMM + qfn_quant_head_gate) of owner rows lo .. lo + n - 1 only,
        written at those rows of the fixed (q8, weights) buffers that the request-sharded selection reads; ``gate``: [rows, 32]."""
        from .index_qfn import qfn_quant_head_gate
        from .indexer import SCALE_FMT

        lo, n, q8, weights, fused, noslots = own
        ip = self.index_projection
        if fused is not None:  # K31MK_DSA_REQ_SHARD_OWNQ=2 (QFP): q_a RMSNorm + wq_b + quant + head gate in one kernel, no index-K store
            k_col, g_col, (norm, bias, eps) = self.qfp
            w = ip.weights
            if IDX_FP4_FP4Q:  # K31MK_DSA_IDX_Q=fp4: index_q.cuh's kQ4 query front (MXFP4 codes + one exponent word per row and head)
                fused.module.run_qf_q4(projected[lo : lo + n], w.query_norm, w.query, q8.q4[lo : lo + n], q8.sf[lo : lo + n],
                                       weights[lo : lo + n].view(-1, 32), float(w.epsilon), int(k_col), int(g_col), noslots,
                                       self.index_cache, norm, bias, float(eps))
                return q8, weights
            fused.module.run_qf(projected[lo : lo + n], w.query_norm, w.query, q8[lo : lo + n].view(torch.uint8),
                                weights[lo : lo + n].view(-1, 32), float(w.epsilon), int(k_col), int(g_col), noslots,
                                self.index_cache, norm, bias, float(eps))
            return q8, weights
        acc = ip.query[lo : lo + n]
        torch.mm(projected[lo : lo + n, :1536], ip.folded.T, out=acc.view(n, 4096))
        if IDX_FP4_FP4Q:  # K31MK_DSA_IDX_Q=fp4: the MXFP4 QFN kernel (idx_fp4_q.qfn_quantize_query) into the owner rows of the fixed buffers
            from types import SimpleNamespace

            from .idx_fp4_q import qfn_quantize_query

            qfn_quantize_query(acc, projected[lo : lo + n], gate[lo : lo + n], ip.weights.epsilon,
                               SimpleNamespace(q4=q8.q4[lo : lo + n], sf=q8.sf[lo : lo + n], weights=weights[lo : lo + n]))
            return q8, weights
        qfn_quant_head_gate(acc, projected[lo : lo + n], gate[lo : lo + n], ip.weights.epsilon, 128**-0.5, SCALE_FMT,
                            out=(q8[lo : lo + n], weights[lo : lo + n]))
        return q8, weights

    def _index_query_qfp(self, projected, slots, own=None):
        """K31MK_DSA_QFP: this step's index-K rows from the projection (the stock writer's LayerNorm + store, on this stream) and
        the QFN index query with the head gate from the projection."""
        from flashinfer.norm import layernorm
        from sglang.kernels.ops.kvcache.dsa_fused_store_index_cache import fused_store_index_k_cache

        from .index_qfn import qfn_quant_head_gate
        from .indexer import SCALE_FMT

        k_col, g_col, (norm, bias, eps) = self.qfp
        cur = idx = None
        if self.qfp_par:  # fork: the index-K rows are written beside the query GEMM; joined back before this returns
            cur = torch.cuda.current_stream(projected.device)
            idx = self.index_stream
            idx.wait_stream(cur)
            with torch.cuda.stream(idx):
                ensure_stream(projected, idx)
                if self.qfp_kst:  # K31MK_DSA_QFP_KST: LayerNorm + store in one launch (FP8 or MXFP4 pages, as the pool is)
                    from .index_kst import store as kst_store

                    kst_store(projected, k_col, slots, self.index_cache, (norm, bias, eps))
                else:
                    key = layernorm(projected[:, k_col : k_col + 128], norm, bias, eps)
                    if IDX_FP4:  # K31MK_DSA_IDX_FP4_FP8Q: MXFP4 index-K rows (the refusals keep the (k + q) form off QF / QFP)
                        idx_fp4.store_index_k(key, self.index_cache, slots)
                    else:
                        fused_store_index_k_cache(key, self.index_cache, slots, 64)
        elif self.qfp_kst:
            from .index_kst import store as kst_store

            kst_store(projected, k_col, slots, self.index_cache, (norm, bias, eps))
        else:
            key = layernorm(projected[:, k_col : k_col + 128], norm, bias, eps)
            if IDX_FP4:
                idx_fp4.store_index_k(key, self.index_cache, slots)
            else:
                fused_store_index_k_cache(key, self.index_cache, slots, 64)
        ip = self.index_projection
        if own is not None:  # K31MK_DSA_REQ_SHARD_OWNQ (FP8 query, or MXFP4 under K31MK_DSA_IDX_Q=fp4): the owner rows' query only (index-K above stays full)
            out = self._index_query_own(projected, projected[:, g_col : g_col + 32], own)
        elif IDX_FP4_FP4Q:  # K31MK_DSA_IDX_Q=fp4: the same rstd step with the MXFP4 quantizer (idx_fp4_q.qfn_quantize_query)
            from .idx_fp4_q import qfn_quantize_query

            torch.mm(projected[:, :1536], ip.folded.T, out=ip.query.view(-1, 4096))
            out = qfn_quantize_query(ip.query, projected, projected[:, g_col : g_col + 32], ip.weights.epsilon, ip.fp4)
        else:
            torch.mm(projected[:, :1536], ip.folded.T, out=ip.query.view(-1, 4096))
            out = qfn_quant_head_gate(ip.query, projected, projected[:, g_col : g_col + 32], ip.weights.epsilon, 128**-0.5, SCALE_FMT)
        if idx is not None:
            cur.wait_stream(idx)  # the scoring on this stream reads the stored index-K rows
        return out

    def _mla_query(self, mq, args, projected, cache, slots, s):
        """The MLA query bytes + current KV row of this step: MlaQuery, or MlaQueryWide under K31MK_MLAQ_W (same stream)."""
        if self.mlaq_w is not None:
            self.mlaq_w.run(projected, self.base.query_scratch[: projected.shape[0]], cache, slots)
        else:
            mq(*args, *self.base.query_views, s)

    def run(self, projected, out, *, cache, slots, block_table, lengths, totals=None, q=1):
        step = self.step
        if step is None or not step.sparse:
            return self.base.run(projected, out, cache=cache, slots=slots,
                                 block_table=block_table, lengths=lengths, totals=totals, q=q)
        if step.sparse == 2 and self.tsi is None:
            raise ValueError("verify step labelled tsi but this build has no TSI state (K31MK_TSI=0)")
        b = self.base
        rows = projected.shape[0]
        mq, _, query, _ = b.kernels[rows, q]
        qnorm, kvnorm, qb, uk, uv, qinv, kinv = b.views
        own = self.ownq if step.sparse == 2 else None  # K31MK_DSA_REQ_SHARD_OWNQ: set only on profiles the sharded selection serves
        args = (to_cute(projected[:, :2112]), qnorm, kvnorm, qb, uk,
                to_cute(cache.view(torch.uint8)), to_cute(slots, 8),
                qinv, kinv, b.eps, query)
        s = stream(b.device)
        main = torch.cuda.current_stream(b.device)
        # The index-K/gate writer is already running on the index stream. Wait only
        # for the input projection here; main MLA query is independent of it.
        side = self.query_stream
        # K-18 / I5: every stream context below that issues tvm-ffi launches is checked once per (device, capture,
        # stream) before it launches anything (stream_check.ensure; fail closed).
        ensure_stream(projected, main)  # the mode-2 index query launches on main
        with torch.cuda.stream(side):
            ensure_stream(projected, side)  # the mode-0/1 index query launches on side
        if self.index_q_mode == 2:
            iq, gates = self.index_query(projected, slots, own)
            side.wait_stream(main)
            if not self.packed_index_gate:
                side.wait_stream(self.gate_stream)
            self._mla_query(mq, args, projected, cache, slots, s)
        else:
            side.wait_stream(main)
            if not self.packed_index_gate:
                side.wait_stream(self.gate_stream)
            self._mla_query(mq, args, projected, cache, slots, s)
            with torch.cuda.stream(side):
                iq, gates = self.index_query(projected, slots, own)
        if side is not self.index_stream and self.qf is None and self.qfp is None:
            # The scoring reads the index-K rows the writer stores. Under K31MK_DSA_QF the index query stored them on this
            # path and the index stream stays out of the step (joining it inside a capture would join an unforked stream).
            side.wait_stream(self.index_stream)
        with torch.cuda.stream(side):
            ensure_stream(projected, side)  # DCP1 topk_v2 / TSI prep, plan, scan, merge launch here
            if step.sparse == 2:  # TSI: the same exact top-2048 from 1/16 of the pages per rank + one exchange; or, under
                # K31MK_DSA_REQ_SHARD, the stock selection for this rank's owner requests + one all-gather (dsa/req_shard.py)
                state, ordinal = self.tsi
                if ordinal == 0:  # once per forward, on the selection stream ahead of every layer's scan
                    state.prepare(block_table, step.score_contexts, step.causal_lengths)
                selected = state.select(ordinal, iq, gates, self.index_cache, self.selection.selected)
            else:
                selected = self.selection.run(iq, gates, self.index_cache, block_table, step)
        value = self.sparse_values[rows, q]
        h = b.heads
        if self.mla6 is not None:  # L38: no padded query copy; the fold replaces the core's split reduction
            main.wait_stream(side)
            if isinstance(selected, torch.Tensor):
                partial, stats = self.mla6.run(b.query_scratch[:rows], cache, selected, step.causal_lengths,
                                               b.stock_scales, stream=main.cuda_stream)  # main == s: the fold's stream
            else:  # K31MK_DSA_REQ_SHARD_AG=fused: a req_shard.TaggedRows; the core polls the tagged slot words itself
                partial, stats = self.mla6.run_tagged(b.query_scratch[:rows], cache, selected, step.causal_lengths,
                                                      b.stock_scales, stream=main.cuda_stream)
            value(to_cute(partial, 16), to_cute(stats, 8), uv, to_cute(head_gate(projected, h)),
                  to_cute(lengths, 4), cutlass.Float32(b.stock_scales[1]), to_cute(out),
                  to_cute(self.mla.output[:, 0, :h]), s)
            return out
        # Copy before joining the indexer, so this independent work overlaps it.
        self.mla.prepare_query(b.query_scratch[:rows])
        main.wait_stream(side)
        self.mla.run_core(cache, selected, step.causal_lengths, b.stock_scales)
        value(to_cute(self.mla.output[:, 0, :h]), uv, to_cute(head_gate(projected, h)),
              to_cute(lengths, 4), to_cute(out), s)
        return out


class SparseVerifyIndexKey:
    def __init__(self, base, core):
        self.base, self.core = base, core
        self.packed_weight = None
        self.key = None
        if core.qf is not None or core.qfp is not None:
            # K31MK_DSA_QF / QFP: the index query stores the index-K rows and reads the gate from the projection.
            core.packed_index_gate = True
        elif core.selection.rows == 8:
            # Preparation only. Both outputs retain the stock BF16 GEMM rounding. With the fused
            # index query the head gate is computed there, so only the index-K rows are packed.
            fused = core.fused_index_q is not None
            parts = (base.weights.projection,) + (() if fused else (core.index_projection.weights.gate,))
            self.packed_weight = torch.cat(parts, dim=0)
            self.projected = torch.empty(
                (8, self.packed_weight.shape[0]), device=self.packed_weight.device, dtype=torch.bfloat16
            )
            if not fused:
                core.index_projection.gate = self.projected[:, 128:160]
            core.packed_index_gate = True
            if core.index_q_mode and not fused:  # a packed gate is written on the index stream, which the query never joins
                raise RuntimeError("K31MK_INDEX_Q_OFF_WRITER_WIDE: the stock index query off the writer needs the gate stream")
            if IDXK_LN_PDL:  # the PDL LayerNorm writes this fixed buffer (the default allocates per call)
                self.key = torch.empty((8, 128), device=self.packed_weight.device, dtype=torch.bfloat16)

    def bind(self, cache, page_size, rows=(8,)):
        self.core.index_cache = cache
        return self.base.bind(cache, page_size, rows)

    def run(self, hidden, slots):
        if self.core.step is None or not self.core.step.sparse:
            return self.base.run(hidden, slots)
        if self.core.qf is not None or self.core.qfp is not None:
            return  # K31MK_DSA_QF / QFP: index_query stores this step's index-K rows (PreparedSparseVerifyMla.index_query)
        side = self.core.index_stream
        gate = self.core.gate_stream
        main = torch.cuda.current_stream(hidden.device)
        side.wait_stream(main)
        if self.core.fused_index_q is not None:
            self.core.hidden = hidden  # read by the fused index query on the index stream
        if self.packed_weight is not None:
            from flashinfer.norm import layernorm
            from sglang.kernels.ops.kvcache.dsa_fused_store_index_cache import (
                fused_store_index_k_cache,
            )

            with torch.cuda.stream(side):
                ensure_stream(hidden, side)  # K-18 / I5: LayerNorm -> index-K store (PDL) launch here
                torch.mm(hidden, self.packed_weight.T, out=self.projected)
                w = self.base.weights
                # LayerNorm supports this padded row stride; its output is
                # contiguous, so the stock cache writer needs no input copy.
                if self.key is not None:
                    key = _layernorm_pdl(self.key, self.projected[:, :128], w)
                else:
                    key = layernorm(self.projected[:, :128], w.norm, w.bias, w.eps)
                if IDX_FP4:
                    idx_fp4.store_index_k(key, self.base.cache, slots)
                else:
                    fused_store_index_k_cache(key, self.base.cache, slots, 64)
            return
        gate.wait_stream(main)
        with torch.cuda.stream(side):
            ensure_stream(hidden, side)
            self.base.run(hidden, slots)
        with torch.cuda.stream(gate):
            ensure_stream(hidden, gate)
            self.core.prepare_gate(hidden)



class MultiSparseVerifyMla:
    """One fixed scratch profile per row capacity; no step-time allocation."""

    def __init__(self, base, weights, widths, max_context, tsi=None):
        self.base = base
        self.profiles = {
            rows: PreparedSparseVerifyMla(base, weights, rows, max_context, tsi)
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


def _enable_qf(core, key_weights):
    """K31MK_DSA_QF on every 1-8-row profile of ``core`` (1-16 under K31MK_DSA_QF_MAX_ROWS=16, 1-32 under K31MK_DSA_QFW=1, 1-N with
    K31MK_DSA_QFW_MAX_ROWS=N;
    wider profiles keep the stock writer and gate GEMM). The packed
    projection must carry the QF rows: the host packs them under the same switch."""
    for rows, profile in core.profiles.items():
        if profile.fused_index_q is None:
            if DSA_QFP and rows >= max(2, DSA_QFP_MIN_ROWS):
                _enable_qfp(profile, rows, key_weights)
            continue
        k_col, g_col, pitch = qf_columns(profile.base.heads)
        if profile.base.pitches[rows] != pitch:
            raise ValueError(
                f"K31MK_DSA_QF needs the packed MLA projection [q_a | kv_a | gate | wk | weights_proj] ({pitch} rows at "
                f"{profile.base.heads} heads), got pitch {profile.base.pitches[rows]}: set K31MK_DSA_QF=1 for the host too"
            )
        profile.qf = (k_col, g_col, (key_weights.norm, key_weights.bias, key_weights.eps))
        logger.info("DSA query front engaged: rows %d, heads %d, pitch %d (index-K col %d, gate col %d%s)", rows,
                    profile.base.heads, pitch, k_col, g_col, ("; MXFP4 index-K store" + ("; MXFP4 index query" if IDX_FP4_FP4Q else "")) if IDX_FP4 else "")
        if type(profile.fused_index_q).__name__ == "FusedIndexQueryWide":
            logger.info("DSA QF-wide engaged: rows %d (folded wq_b, index_qfw.cuh)", rows)
        elif type(profile.fused_index_q).__name__ == "FusedIndexQueryCross":
            logger.info("DSA QF-cross engaged: rows %d (K31MK_DSA_QFX_LOW, folded wq_b, index_qfx.cuh)", rows)


def _enable_qfp(profile, rows, key_weights, parallel=None, kst=None):
    """K31MK_DSA_QFP on one profile without a fused index query (its writer and gate GEMM stop; see DSA_QFP)."""
    from .index_qfw import folded_query

    k_col, g_col, pitch = qf_columns(profile.base.heads)
    if profile.base.pitches[rows] != pitch:
        raise ValueError(
            f"K31MK_DSA_QFP needs the packed MLA projection ({pitch} at {profile.base.heads} heads), got pitch "
            f"{profile.base.pitches[rows]} at rows {rows}: set K31MK_DSA_QF=1 for the host too"
        )
    if (k_col * 2) % 16 or (pitch * 2) % 16:  # flashinfer's cute LayerNorm needs 16-byte aligned rows (fail closed at build)
        raise ValueError(f"K31MK_DSA_QFP: index-K column {k_col} / pitch {pitch} not 16-byte aligned")
    ip = profile.index_projection
    if ip.folded is None:
        ip.folded = folded_query(ip.weights)  # built before any graph capture; shared per layer (QF-wide / QFN)
    profile.qfp = (k_col, g_col, (key_weights.norm, key_weights.bias, key_weights.eps))
    if parallel is None:  # K31MK_DSA_QFP=2 forks the store, on profiles up to K31MK_DSA_QFP_PAR_MAX_ROWS rows when that is set
        parallel = DSA_QFP == 2 and (DSA_QFP_PAR_MAX_ROWS == 0 or rows <= DSA_QFP_PAR_MAX_ROWS)
    profile.qfp_par = bool(parallel)
    logger.info("DSA QFP engaged: rows %d, heads %d, pitch %d (index-K col %d, gate col %d; no writer / gate GEMM, QFN query; index-K %s)",
                rows, profile.base.heads, pitch, k_col, g_col, "on the index stream" if profile.qfp_par else "ahead on the query stream")
    profile.qfp_kst = DSA_QFP_KST if kst is None else bool(kst)
    if profile.qfp_kst:
        nrm = key_weights.norm, key_weights.bias
        if any(t.dtype != torch.float32 or t.shape != (128,) or not t.is_contiguous() for t in nrm):
            raise ValueError(f"K31MK_DSA_QFP_KST needs FP32 contiguous [128] index-K LayerNorm weight / bias, got {[(t.dtype, tuple(t.shape)) for t in nrm]}")
        logger.info("DSA QFP-KST engaged: rows %d (index-K LayerNorm + %s store in one kernel, index_kst.cuh; %s)", rows,
                    "MXFP4" if IDX_FP4 else "FP8", "index stream" if profile.qfp_par else "query stream")


def _enable_mlaq_w(core):
    """K31MK_MLAQ_W on every profile of ``core`` with at least K31MK_MLAQ_W_MIN_ROWS rows (built before any graph capture; fail closed
    when the base query is not the plain DCP1 MlaQuery, whose only outputs MlaQueryWide reproduces)."""
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.mla.query import MlaQuery

    from .mlaq_w import MlaQueryWide, eps_value

    for rows in sorted(core.profiles, reverse=True):  # widest first: the shared scratch is sized once
        profile = core.profiles[rows]
        if rows < MLAQ_W_MIN_ROWS:
            continue
        b = profile.base
        if b.query_type is not MlaQuery or b.dcp is not None or b.query_buffers:
            raise ValueError(f"K31MK_MLAQ_W replaces the plain DCP1 MlaQuery only (query_type {b.query_type.__name__}, dcp {b.dcp})")
        profile.mlaq_w = MlaQueryWide(b.weights, eps_value(b.eps), rows)
        logger.info("MLAQ-W engaged: rows %d, heads %d (norm + KV insert, cuBLAS Q_B, W_UK + quant; no MlaQuery launch)", rows, b.heads)


class BuiltSparseVerify(BuiltK31Verify):
    def __init__(self, weights, deadline_s):
        super().__init__(weights, deadline_s)
        self.sparse_cores = {}
        self.row_share = RowTableShare() if DSA_ROW_TABLE_ONCE else None
        self.req_lens = None
        if TOPK_CHAIN_D1 and VERIFY_TOPK == "x47960":
            self.req_lens = RequestLengths(self.resources.scratch["hidden.0"].device, max(self.widths))
        self.tsi, dsa = None, ()
        if TSI_KERNEL and REQ_SHARD:  # K31MK_DSA_REQ_SHARD_WITH_TSI (checked at import): TSI at rows <= TSI_RS_SPLIT_ROWS, R1 above
            from .req_shard import SplitSelection

            device = self.resources.scratch["hidden.0"].device
            tsi, dsa = _build_tsi(weights, self.widths, device, self.query_width)
            rs, dsa_rs = _build_req_shard(weights, self.widths, device, self.query_width, deadline_s)
            if list(dsa) != list(dsa_rs):
                raise ValueError(f"TSI and request sharding disagree on the DSA layers: {list(dsa)} vs {list(dsa_rs)}")
            self.tsi = SplitSelection(tsi, rs, TSI_RS_SPLIT_ROWS)
        elif TSI_KERNEL:
            device = self.resources.scratch["hidden.0"].device
            self.tsi, dsa = _build_tsi(weights, self.widths, device, self.query_width)
        elif REQ_SHARD:  # same slot as TSI: (state, ordinal) per layer; state.prepare / state.select
            device = self.resources.scratch["hidden.0"].device
            self.tsi, dsa = _build_req_shard(weights, self.widths, device, self.query_width, deadline_s)
        for record in weights.layers:
            if record.attention_kind != 0:
                continue
            a, sc = record.attention, record.attention_scalars
            if len(a) != 12:
                raise ValueError("sparse verify needs index-Q/head-gate weight tensors")
            attention = self.resources.attention[record.layer_id]
            core = MultiSparseVerifyMla(
                attention.core, IndexWeights(a[10], a[11], a[1], sc[0]),
                self.widths, weights.context_length,
                None if self.tsi is None else (self.tsi, dsa.index(record.layer_id)),
            )
            attention.core = core
            self.sparse_cores[record.layer_id] = core
            if self.row_share is not None:
                for profile in core.profiles.values():
                    profile.selection.share = self.row_share
            if self.req_lens is not None:
                for profile in core.profiles.values():
                    profile.selection.req_lens = self.req_lens
            if DSA_QF:
                _enable_qf(core, self.indexers[record.layer_id].weights)
            if MLAQ_W:
                _enable_mlaq_w(core)
            if REQ_SHARD_OWNQ:  # after _enable_qf: QFP profiles are known
                _enable_ownq(core, self.tsi)
            self.indexers[record.layer_id] = MultiSparseVerifyIndexKey(
                self.indexers[record.layer_id], core
            )

    def summary(self):
        return f"TARGET_VERIFY Breq={self.request_batches} Q={self.query_width}; causal sparse DSA; host dense/sparse graphs"

    def run_verify(self, embeddings, view):
        if not isinstance(view, SparseVerifyStep):
            raise ValueError("sparse verify requires the host causal-metadata adapter")
        view.validate()
        if view.sparse == 2 and self.tsi is None:
            raise ValueError("verify step labelled tsi but K31MK_TSI is off in the plugin")
        for core in self.sparse_cores.values():
            core.step = view
        if self.row_share is not None:
            self.row_share.step += 1  # a new step: its first DSA layer copies the per-row page table again
        if self.req_lens is not None:
            self.req_lens.begin(view)  # K31MK_VERIFY_TOPK_CHAIN d1: this step's kv_len, before any layer
        return super().run_verify(embeddings, view)


def build_from_tensors(weights, deadline_s):
    return BuiltSparseVerify(weights, deadline_s)
