# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/qwen4_exp.py (Qwen4ExpAttentionDecoderLayer.
# _compute_qsa_topk_indices_eager) and
# python/sglang/srt/layers/attention/qsa/qsa_indexer.py (QSAIndexer._forward_impl,
# select_decode_tokens and _compress_decode_cuda_graph on CUDA-graph metadata).
"""The QSA indexer (the top-k block selection) of a covered step's attention layer.

A TP1 decode, target verify or draft extend step on CUDA-graph metadata runs the
indexer here on the plugin's kernels, each op behind its own switch:

* ``prologue_fold``: index-Q norm + RoPE, the raw-key and RoPE-position ring store and
  the graph group compression into the compressed-K cache in one launch
  (``qsa/prologue_fold.py``), for ``project_qk``'s q_prep and
  ``_compress_decode_cuda_graph``'s k_compress;
* ``score``: up to 64 rows the per-row scorer (v5) or, for 4-row verify requests, the
  grouped scorer (v4); 68..576 rows of 4-row requests the grouped4 scorer (v7, or v6
  without the key pipeline; SM107 only); one 4-row draft-extend request the MK scorer
  (v0, or v3 with the short-row variant);
* ``select``: up to 64 rows the cluster-split top-512 (v1); wider target verify (and
  grouped draft extend) the sampled verify top-512 (v2).

An op no kernel serves runs SGLang's own (``project_qk`` +
``update_key_state_and_compress``, ``qsa_mqa_decode``, ``fast_topk``). Prefill / extend,
breakable graphs, CP, batch-invariant mode, eager metadata and any other geometry call
``layer._compute_qsa_topk_indices``. An MTP draft decode step that reuses the draft
extend's captured indices runs SGLang's lookup; a draft extend captures its indices
through SGLang's hooks, in the stock order (slot ``qsa_draft``: the lookup inside its
K/V pack and the capture as one launch, ``runtime/qsa_draft.py``).

Reads: the index projection and norms, the step's graph metadata, the layer's ring,
RoPE positions and compressed-K cache. Writes: the ring rows and RoPE positions at the
step's ring slots and the compressed rows at the graph write slots (rows completing no
group write the inert slot 0), as the stock indexer does. Returns the stock layout:
block indices [rows, 512] when the backend defers expansion, else SGLang's expanded
token indices.

Numerics: the fold's q, ring rows, RoPE positions and compressed rows are
bit-identical to SGLang's two launches; v5 is SGLang's per-row scorer kernel, and v4 /
v6 / v7 give its logits bit for bit (v0 / v3: the source's exact mapping, not
re-qualified here). v1 and v2 select the blocks SGLang's ``fast_topk`` selects wherever
a row's 512th score is not tied (at a tie they take the smaller indices, ``fast_topk``
any); the order within a row differs, as ``fast_topk``'s own order is unspecified.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from sgl_nvidia_mega.qwen3_8.runtime import bf16_gemm_pins

logger = logging.getLogger(__name__)

STOCK = "stock"
OP = "qsa.indexer"
# The plugin kernels' geometry (_providers/sparse.py supported(), capabilities.py).
HEAD_DIM, RATIO, BLOCK_TOPK, PAGE = 128, 4, 512, 16
MAX_HEADS = 16  # index query heads, padded to 8 or 16
MAX_CAPACITY = 131072  # compressed keys one row's page table spans
SPLIT_ROWS = 64  # SPARSE_SCORE v4 / v5, SPARSE_SELECT v1: 1..64 rows
GROUPED4_ROWS = 576  # SPARSE_SCORE v6 / v7: 17..144 four-row requests
# v4: SGLANG_QSA_SCORER_V2_* at the defaults the provider admits (ver 7, 4096 CTAs
# over the groups at these levels, NT 1, no guard or peel)
V4_CTA_LEVELS = (16, 32, 64, 128, 256, 512, 1024)
V4_TARGET_CTAS = 4096
# v1: SGLANG_QSA_TOPK_THREADS / _CLUSTER unset, the only setting the provider admits:
# 1024 threads, a cluster of 1..16 CTAs, at most 8 float4 loads a thread
SPLIT_THREADS, SPLIT_MAX_CLUSTER, SPLIT_MAX_ITERS = 1024, 16, 8
MK_LONG_CTAS = 1024

SGLANG_READS = {
    "srt/models/qwen4_exp.py": (
        "Qwen4ExpAttentionDecoderLayer",
        "_compute_qsa_topk_indices",
        "_shards_rows_over_cp",
        "is_qsa",
        "indexer",
        "layer_id",
    ),
    "srt/layers/attention/qsa/glue.py": (
        "get_qsa_indexer_metadata",
        "resolve_qsa_sparse_backend",
    ),
    "srt/layers/attention/qwen_sparse_attn_backend.py": (
        "should_reuse_mtp_sparse_indices",
        "should_capture_mtp_sparse_indices",
        "capture_mtp_sparse_indices",
    ),
    "srt/layers/attention/qsa/metadata.py": (
        "is_cuda_graph",
        "token_to_kv_pool",
        "req_pool_indices",
        "decode_logical_positions",
        "pending_ring_slots",
        "graph_ring_group_locs",
        "graph_write_locs",
        "defer_block_expansion",
        "get_token_to_batch_idx",
        "get_seqlens_int32",
        "get_decode_mqa_inputs",
    ),
    "srt/layers/attention/qsa/qsa_indexer.py": (
        "_use_fused_prep",
        "_rope_axis_map",
        "apply_rope",
        "index_kv_heads",
        "layer_id",
        "index_n_heads",
        "index_kv_heads",
        "index_head_dim",
        "token_topk",
        "compress_ratio",
        "block_topk",
        "rotary_emb",
        "index_qk_proj",
        "q_layernorm",
        "k_layernorm",
        "_rope_axis_map",
        "_use_fused_prep",
        "_use_fused_compress",
        "project_qk",
        "update_key_state_and_compress",
    ),
    "srt/layers/attention/qsa/mqa.py": (
        "qsa_mqa_decode",
        "_decode_ctas_per_row",
        "_tilelang_qsa_mqa_decode_kernel",
    ),
    "srt/layers/attention/qsa/kernel.py": ("expand_qsa_block_indices",),
    "srt/layers/layernorm.py": ("GemmaRMSNorm", "weight", "variance_epsilon"),
    "srt/layers/rotary_embedding/base.py": (
        "cos_sin_cache",
        "rotary_dim",
        "is_neox_style",
    ),
    "srt/mem_cache/qsa_kv_pool.py": (
        "index_state_dtype",
        "qsa_compressed_dtype",
        "qsa_rope_position_buffer",
        "get_qsa_key_state_buffer",
        "get_qsa_compressed_k_buffer",
    ),
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "srt/model_executor/forward_mode.py": (
        "is_decode",
        "is_target_verify",
        "is_draft_extend_v2",
    ),
    "srt/model_executor/forward_batch_info.py": (
        "forward_mode",
        "req_pool_indices",
        "spec_info",
    ),
    "srt/speculative/eagle_info.py": ("draft_token_num",),
    "srt/model_executor/runner_backend_utils/breakable_cuda_graph/__init__.py": (
        "is_in_breakable_cuda_graph",
    ),
    "srt/batch_invariant_ops/__init__.py": ("is_batch_invariant_mode_enabled",),
    "srt/runtime_context.py": ("get_parallel",),
    "kernels/ops/attention/fast_topk.py": ("fast_topk",),
    "kernels/jit/utils/compile/paths.py": ("KERNEL_PATH",),
    "kernels/ops/attention/qsa_indexer.py": ("qsa_index_q_norm_rope_store",),
}

# _project_qk mirrors project_qk when BF16 pins are on
SGLANG_MIRRORS = {
    "srt/layers/attention/qsa/qsa_indexer.py:QSAIndexer.project_qk": "0df8761ad501b30a",
}

# Reads and mirrors that need an SGLang change newer than the unit job's pin: the
# contract test skips each, naming the change, only where the installed SGLang lacks it.
SGLANG_REQUIRES = {
    "srt/layers/attention/qsa/qsa_indexer.py:QSAIndexer.project_qk": (
        "sglang#48271",
        ("4931b511637d8755",),
    ),
}


@dataclass(frozen=True)
class Options:
    """The indexer's kernel choices. The three op switches are the executor's slots;
    the rest replace the source's environment knobs, with defaults from its
    TP1 tunings (mid concurrency for up to 64 rows, high concurrency above)."""

    prologue_fold: bool = True  # SGLANG_QSA_QPF=1 (mid)
    score: bool = True  # SGLANG_QSA_SPLIT_TOPK=1 + SCORER_V2 defaults (mid), grouped4
    select: bool = True  # split v1 (mid), SGLANG_QSA_VERIFY_TOPK=sampled (high)
    mk_score: bool = True  # SGLANG_QSA_MK_SCORE=1, SGLANG_QSA_Q4_SCORE=1 (mid)
    # SGLANG_QSA_MK_SCORE_SHORT32 / SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_SCORE_SHORT32: unset in every recipe
    short32: bool = False
    grouped4: bool = True  # SGLANG_ENABLE_QSA_GROUPED4_VERIFY(_CHAIN)=1 (high)
    grouped4_ctas: int = 128  # SGLANG_QSA_GROUPED4_VERIFY_CHAIN_CTAS=128 (high)
    # SGLANG_QSA_GROUPED4_VERIFY_PIPELINE_STAGES=2 (high): 2 -> v7, 0 -> v6
    grouped4_stages: int = 2
    grouped_max_requests: int = 144  # SGLANG_QSA_GROUPED_SCORE_MAX_REQUESTS=144
    grouped_draft_extend: bool = True  # SGLANG_ENABLE_QSA_GROUPED4_DRAFT_EXTEND=1
    draft_extend_union: bool = True  # SGLANG_ENABLE_QSA_DRAFT_EXTEND_UNION=1 (high)

    def __post_init__(self):
        if self.grouped4_stages not in (0, 2):
            raise ValueError("grouped4_stages is 0 (v6) or 2 (v7)")
        if self.grouped4_ctas <= 0 or self.grouped_max_requests <= 0:
            raise ValueError("grouped4_ctas and grouped_max_requests are positive")


def layer_reasons(layer) -> str:
    """'' when the kernels take this layer's indexer, else why not."""
    if not layer.is_qsa:
        return "not a QSA layer"
    ix = layer.indexer
    geometry = (ix.index_kv_heads, ix.index_head_dim, ix.compress_ratio, ix.block_topk)
    if geometry != (1, HEAD_DIM, RATIO, BLOCK_TOPK):
        return (
            f"QSA index (kv heads, head dim, ratio, block top-k) {geometry}, the "
            f"kernels take {(1, HEAD_DIM, RATIO, BLOCK_TOPK)}"
        )
    if not 0 < ix.index_n_heads <= MAX_HEADS:
        return f"{ix.index_n_heads} index heads, the kernels take 1..{MAX_HEADS}"
    return ""


def build(
    model, kernels, *, source_dir, context_len=None, compressed_dtype=None
) -> str:
    """Compile ``kernels`` on every rank before capture; '' or why the indexer stays
    stock. ``source_dir``: an absolute node-local directory for the kernel sources the
    JIT builds. ``context_len`` (the server's) also builds the split selects of every
    width; ``compressed_dtype`` (the pool's, BF16 by default) is the fold's output."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return "the QSA indexer kernels need an SM10x GPU"
    from sglang.srt.models.qwen4_exp import Qwen4ExpAttentionDecoderLayer
    from sglang.srt.runtime_context import get_parallel

    tp = get_parallel().tp_size
    if tp != 1:
        return f"TP{tp}: the QSA indexer kernels are admitted at TP1"
    m = model.model
    layers = [
        m.layers[i]
        for i in range(m.start_layer, m.end_layer)
        if isinstance(m.layers[i], Qwen4ExpAttentionDecoderLayer) and m.layers[i].is_qsa
    ]
    if not layers:
        return "no QSA layer"
    for layer in layers:
        reason = layer_reasons(layer)
        if reason:
            return f"layer {layer.layer_id}: {reason}"
    if not Path(source_dir).is_absolute():
        return f"source_dir {source_dir!r} is not an absolute node-local directory"
    kernels.source_dir = Path(source_dir)
    kernels.sm107 = torch.cuda.get_device_capability() == (10, 7)
    try:
        kernels.compile(layers[0].indexer, context_len, compressed_dtype)
    except Exception as e:  # a compile failure keeps the stock indexer
        return f"QSA indexer kernels do not build: {type(e).__name__}: {e}"
    return ""


def select(
    executor, layer, hidden_states, positions, st, step=None, projected_qk=None
) -> torch.Tensor:
    """What ``layer._compute_qsa_topk_indices(hidden_states, positions, st.fb)``
    returns. ``executor.qsa_indexer``: the built ``IndexerKernels``, or None.
    ``step``, ``projected_qk``: a step ``prepare`` made and ``index_qk_proj`` of its rows,
    launched ahead by the caller (the source's ``projected_qk``,
    SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_QSA_PDL_INDEX)."""
    kernels = executor.qsa_indexer
    fb = st.fb
    if step is None and kernels is not None:
        step = kernels.prepare(layer, hidden_states, positions, fb)
    if step is None:
        st.ran(OP, STOCK)
        if executor.qsa_draft is not None and fb.forward_mode.is_draft_extend_v2():
            from sgl_nvidia_mega.qwen3_8.runtime import qsa_draft

            return qsa_draft.indices(executor, layer, hidden_states, positions, st)
        return layer._compute_qsa_topk_indices(hidden_states, positions, fb)
    topk = kernels.run(step, layer.indexer, hidden_states, positions, projected_qk)
    # _compute_qsa_topk_indices_eager: a draft extend seeds the MTP decode steps
    should_capture = getattr(step.backend, "should_capture_mtp_sparse_indices", None)
    if should_capture is not None and should_capture(fb):
        from sgl_nvidia_mega.qwen3_8.runtime import qsa_draft

        qsa_draft.capture(
            executor, step.backend, topk, fb, step.metadata, layer.layer_id, st
        )
    st.ran(OP, step.impl)
    return topk


def choose(
    options,
    *,
    mode,
    rows,
    requests,
    tokens_per_request,
    heads,
    scored,
    capacity,
    row_requests,
    sm107,
):
    """(score, select) of one step: a kernel variant ("v0".."v7") or STOCK each.
    ``scored``: the compressed-K cache is BF16 in 16-key pages, the scorers' layout;
    ``row_requests``: the metadata has each row's request (the MK scorer reads it)."""
    if not 0 < capacity <= MAX_CAPACITY:
        return STOCK, STOCK
    verify4 = mode == "target_verify" and tokens_per_request == 4 and rows % 4 == 0
    grouped = verify4 or (
        mode == "draft_extend" and options.grouped_draft_extend and rows == 4 * requests
    )
    score = STOCK
    if options.score and scored:
        if (
            options.mk_score
            and mode == "draft_extend"
            and rows == 4
            and requests == 1
            and heads == 4
            and row_requests
        ):
            score = "v3" if options.short32 else "v0"
        elif rows <= SPLIT_ROWS:
            score = "v4" if verify4 else "v5"
        elif (
            options.grouped4
            and sm107
            and heads == 4
            and grouped
            and rows <= GROUPED4_ROWS
            and rows // 4 <= options.grouped_max_requests
        ):
            score = "v7" if options.grouped4_stages == 2 else "v6"
    select = STOCK
    if options.select:
        if rows <= SPLIT_ROWS:
            select = "v1"
        elif (
            mode == "target_verify"
            and tokens_per_request > 1
            and rows % tokens_per_request == 0
        ):
            select = "v2"
        elif mode == "draft_extend" and grouped and options.draft_extend_union:
            select = "v2"
    return score, select


def split_config(rows: int, width: int) -> tuple:
    """(threads, cluster, iters) of the split select for [rows, width] scores:
    ``qsa/split_topk.py``'s ``_config`` at its defaults, without the environment."""
    threads = SPLIT_THREADS
    cluster = 1
    while cluster < SPLIT_MAX_CLUSTER and cluster * 2 * rows <= 128:
        cluster *= 2
    while (
        cluster < SPLIT_MAX_CLUSTER and cluster * threads * 4 * SPLIT_MAX_ITERS < width
    ):
        cluster *= 2
    per_thread4 = -(-width // (cluster * threads * 4))
    iters = 1
    while iters < per_thread4:
        iters *= 2
    return threads, cluster, iters


def grouped_ctas(rows: int) -> int:
    """v4's CTAs per 4-row group: the largest level whose CTAs over all groups stay
    within the target."""
    wanted = V4_TARGET_CTAS // max(rows // 4, 1)
    return max([V4_CTA_LEVELS[0]] + [c for c in V4_CTA_LEVELS if c <= wanted])


def compressed_capacity(context_len: int) -> int:
    """The graph page table's width in compressed keys for a context length."""
    blocks = -(-context_len // RATIO)
    return max(1, -(-blocks // PAGE)) * PAGE


def _mode(forward_mode) -> Optional[str]:
    if forward_mode.is_target_verify():
        return "target_verify"
    if forward_mode.is_decode():
        return "decode"
    if forward_mode.is_draft_extend_v2():
        return "draft_extend"
    return None


def _padded(heads: int) -> int:
    return -(-heads // 8) * 8


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


@dataclass
class _Step:
    """One layer's host decisions, made before any launch."""

    rows: int
    fold: bool
    score: str
    select: str
    metadata: object
    backend: object
    inputs: tuple  # get_decode_mqa_inputs: cache, page table, lengths, width

    @property
    def impl(self) -> str:
        fold = "fold" if self.fold else STOCK
        return f"{fold}+score_{self.score}+select_{self.select}"


class IndexerKernels:
    """The plugin's indexer kernels, called directly (no provider offers). A kernel
    layout is warm once it ran eagerly at its (rows, width); a capture warmup runs every
    width first. A cold one met while capturing runs SGLang's op in that graph."""

    def __init__(self, **options):
        self.options = Options(**options)
        self.source_dir: Optional[Path] = None  # set by build()
        self.sm107 = False  # set by build(): grouped4 is qualified on SM107 only
        self._modules: dict = {}
        self._kernels: dict = {}
        self._warm: set = set()
        self._cold_logged: set = set()

    def prepare(self, layer, hidden_states, positions, fb) -> Optional[_Step]:
        """This step's decisions, or None for the stock path. Host state only."""
        from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
        from sglang.srt.layers.attention.qsa.glue import (
            get_qsa_indexer_metadata,
            resolve_qsa_sparse_backend,
        )
        from sglang.srt.model_executor.forward_context import get_attn_backend
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
            is_in_breakable_cuda_graph,
        )
        from sglang.srt.models.qwen4_exp import _shards_rows_over_cp

        o = self.options
        mode = _mode(fb.forward_mode)
        if (
            not (o.prologue_fold or o.score or o.select)
            or mode is None
            or not layer.is_qsa
            or is_in_breakable_cuda_graph()
            or _shards_rows_over_cp(fb)
            or is_batch_invariant_mode_enabled()
        ):
            return None
        backend = get_attn_backend()
        sparse_backend = resolve_qsa_sparse_backend(backend)
        should_reuse = getattr(sparse_backend, "should_reuse_mtp_sparse_indices", None)
        if should_reuse is not None and should_reuse(fb):
            return None  # SGLang looks up the draft extend's captured indices
        md = get_qsa_indexer_metadata(backend, layer.layer_id, fb)
        rows = md.get_token_to_batch_idx().numel()
        logical = md.decode_logical_positions
        if (
            not md.is_cuda_graph
            or logical is None
            or md.pending_ring_slots is None
            or not 0 < rows <= min(hidden_states.shape[0], positions.shape[-1])
            or rows > logical.numel()
        ):
            return None
        ix = layer.indexer
        pool = md.token_to_kv_pool
        group_locs, write_locs = md.graph_ring_group_locs, md.graph_write_locs
        fold = bool(
            o.prologue_fold
            and group_locs is not None
            and write_locs is not None
            and group_locs.dtype == torch.int32
            and write_locs.dtype == torch.int32
            and tuple(group_locs.shape) == (rows, ix.compress_ratio)
            and tuple(write_locs.shape) == (rows,)
            and group_locs.is_contiguous()
            and hidden_states.dtype == pool.index_state_dtype
            and ix._use_fused_prep(hidden_states)
            and ix._use_fused_compress(pool)
        )
        cache, table, lengths, width = md.get_decode_mqa_inputs(layer.layer_id)
        requests = md.req_pool_indices
        score, select = choose(
            o,
            mode=mode,
            rows=rows,
            requests=fb.req_pool_indices.numel(),
            tokens_per_request=(
                int(fb.spec_info.draft_token_num) if mode == "target_verify" else 1
            ),
            heads=ix.index_n_heads,
            scored=(
                cache.dtype == torch.bfloat16
                and tuple(cache.shape[1:]) == (PAGE, 1, HEAD_DIM)
                and tuple(table.shape) == (rows, width // PAGE)
                and tuple(lengths.shape) == (rows,)
            ),
            capacity=width,
            row_requests=requests is not None and requests.numel() == rows,
            sm107=self.sm107,
        )
        if _capturing():
            key = self._fold_key(ix, hidden_states.dtype, pool.qsa_compressed_dtype)
            fold = fold and self._ready(key in self._modules, "fold", rows, width)
            if not self._ready((score, rows, width) in self._warm, score, rows, width):
                score = STOCK
            if not self._ready(
                (select, rows, width) in self._warm, select, rows, width
            ):
                select = STOCK
        if not fold and score == STOCK and select == STOCK:
            return None
        inputs = (cache, table, lengths, width)
        return _Step(rows, fold, score, select, md, sparse_backend, inputs)

    def _ready(self, warm, variant, rows, width) -> bool:
        """Whether a layout may run in the graph being captured."""
        if variant == STOCK or warm:
            return True
        if (variant, rows, width) not in self._cold_logged:
            self._cold_logged.add((variant, rows, width))
            logger.warning(
                "QSA indexer %s is cold at rows=%d width=%d during capture; this graph "
                "runs SGLang's op",
                variant,
                rows,
                width,
            )
        return False

    def run(
        self, step, indexer, hidden_states, positions, projected_qk=None
    ) -> torch.Tensor:
        """``QSAIndexer._forward_impl`` for a decode-like step on graph metadata;
        ``projected_qk``: ``index_qk_proj`` of the step's rows, already launched."""
        md, n = step.metadata, step.rows
        logical = md.decode_logical_positions[:n]
        hidden = hidden_states[:n]
        positions = positions[:, :n] if positions.ndim == 2 else positions[:n]
        slots = md.pending_ring_slots
        if step.fold:
            q = self._fold(indexer, md, hidden, positions, slots[:n], projected_qk)
        else:
            q, token_k, stored = _project_qk(
                indexer,
                hidden,
                positions,
                pool=md.token_to_kv_pool,
                cache_loc=slots,
                q_heads_padded=_padded(indexer.index_n_heads),
                projected_qk=projected_qk,
            )
            indexer.update_key_state_and_compress(
                token_k, logical, positions, md, state_slots=slots, state_stored=stored
            )
        cache, table, lengths, width = step.inputs
        scores = self._score(step, indexer, q, cache, table, lengths, width)
        blocks = self._select(step, scores, lengths, width)
        if md.defer_block_expansion:
            return blocks
        from sglang.srt.layers.attention.qsa.kernel import expand_qsa_block_indices

        return expand_qsa_block_indices(
            blocks,
            logical,
            md.get_seqlens_int32(),
            compress_ratio=indexer.compress_ratio,
            token_topk=indexer.token_topk,
        )

    # -- prologue ---------------------------------------------------------------------

    @staticmethod
    def _fold_key(indexer, dtype, out_dtype) -> tuple:
        neox = bool(indexer.rotary_emb.is_neox_style)
        return ("fold", dtype, out_dtype, indexer.index_head_dim, neox)

    def _fold(self, indexer, md, hidden, positions, slots, qk=None) -> torch.Tensor:
        """``project_qk``'s fused q_prep and ``_compress_decode_cuda_graph``'s fused
        k_compress in one launch, on ``qk`` (``index_qk_proj`` of the rows) if given;
        returns the index Q [rows, padded heads, dim]."""
        pool = md.token_to_kv_pool
        rope = indexer.rotary_emb
        if qk is None:
            qk = bf16_gemm_pins.linear(indexer.index_qk_proj, hidden)
        out_dtype = pool.qsa_compressed_dtype
        module = self._fold_module(self._fold_key(indexer, qk.dtype, out_dtype))
        positions = positions.long()
        if positions.ndim == 1:
            positions = positions.unsqueeze(0)
        ring = pool.get_qsa_key_state_buffer(indexer.layer_id)
        compressed = pool.get_qsa_compressed_k_buffer(indexer.layer_id)
        q = torch.empty(
            (qk.shape[0], _padded(indexer.index_n_heads), indexer.index_head_dim),
            dtype=out_dtype,
            device=qk.device,
        )
        module.fold(
            qk,
            q,
            indexer.q_layernorm.weight.data,
            rope.cos_sin_cache,
            indexer._rope_axis_map(qk.device),
            positions,
            positions.shape[0],
            slots.long(),
            ring.view(ring.shape[0], -1),
            pool.qsa_rope_position_buffer,
            indexer.index_n_heads,
            rope.rotary_dim,
            indexer.q_layernorm.variance_epsilon,
            md.graph_ring_group_locs,
            indexer.k_layernorm.weight.data,
            md.graph_write_locs,
            compressed.view(compressed.shape[0], -1),
            indexer.compress_ratio,
            indexer.k_layernorm.variance_epsilon,
        )
        return q

    def _fold_module(self, key):
        module = self._modules.get(key)
        if module is None:
            from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH

            from sgl_nvidia_mega.qwen3_8 import _jit
            from sgl_nvidia_mega.qwen3_8.qsa import prologue_fold as source

            _, dtype, out_dtype, head_dim, neox = key
            args = _jit.make_cpp_args(
                dtype, out_dtype, head_dim, neox, _jit.is_arch_support_pdl()
            )
            module = _jit.load_jit(
                "qsa_prologue_fold",
                *args,
                "s" + source._SOURCE_DIGEST,
                cuda_files=[self._source_file("qsa_prologue_fold", source._SOURCE)],
                cuda_wrappers=[("fold", f"qsa_index_prologue_fold<{args}>")],
                extra_include_paths=[str(KERNEL_PATH / "csrc")],
            )
            self._modules[key] = module
        return module

    # -- score ------------------------------------------------------------------------

    def _score(self, step, indexer, q, cache, table, lengths, width) -> torch.Tensor:
        """[rows, width] FP32 scores; rows of at most 512 blocks and keys past a row's
        length are left unset (the selects never read them)."""
        if step.score == STOCK:
            from sglang.srt.layers.attention.qsa.mqa import qsa_mqa_decode

            return qsa_mqa_decode(
                q, cache, table, lengths, width, min_scored_len=indexer.block_topk
            )
        rows = q.shape[0]
        scores = torch.empty((rows, width), dtype=torch.float32, device=q.device)
        scale = float(math.sqrt(indexer.index_head_dim))
        table, lengths = table.to(torch.int32), lengths.to(torch.int32)
        if step.score in ("v0", "v3"):
            requests = step.metadata.req_pool_indices
            kernel = self._score_kernel(step.score, rows, 0, q.device, requests.dtype)
            kernel(q, cache, table, lengths, requests, scores, scale)
        else:
            kernel = self._score_kernel(step.score, rows, q.shape[1], q.device, None)
            kernel(q.unsqueeze(1), cache, table, lengths, scores, scale)
        if not _capturing():
            self._warm.add((step.score, rows, width))
        return scores

    def _score_kernel(self, variant, rows, heads, device, request_dtype):
        if variant in ("v0", "v3"):
            key = (variant, request_dtype)
        elif variant == "v4":
            key = (variant, heads, grouped_ctas(rows))
        elif variant == "v5":
            from sglang.srt.layers.attention.qsa.mqa import _decode_ctas_per_row

            key = (variant, heads, _decode_ctas_per_row(rows, device))
        else:
            key = (variant, heads, self.options.grouped4_ctas)
        kernel = self._kernels.get(key)
        if kernel is None:
            kernel = self._kernels[key] = _score_factory(key)
        return kernel

    # -- select -----------------------------------------------------------------------

    def _select(self, step, scores, lengths, width) -> torch.Tensor:
        """[rows, 512] int32 block indices of each row's 512 largest scores; rows of at
        most 512 blocks select 0..length-1, then -1."""
        lengths = lengths.to(torch.int32)
        if step.select == STOCK:
            from sglang.kernels.ops.attention.fast_topk import fast_topk

            return fast_topk(scores, lengths, topk=BLOCK_TOPK, row_starts=None)
        rows = scores.shape[0]
        out = torch.empty((rows, BLOCK_TOPK), dtype=torch.int32, device=scores.device)
        if step.select == "v1":
            self._split_select(scores, lengths, out)
        else:
            self._verify_select(scores, lengths, out)
        if not _capturing():
            self._warm.add((step.select, rows, width))
        return out

    def _split_select(self, scores, lengths, out) -> None:
        """v1: one thread-block cluster a row (``qsa/split_topk.py``)."""
        vec = scores.data_ptr() % 16 == 0 and scores.stride(0) % 4 == 0
        module = self._split_module(*split_config(*scores.shape), vec)
        module.cluster_topk(scores, lengths, out)

    def _verify_select(self, scores, lengths, out) -> None:
        """v2: the sampled top-k, 1024- or 512-thread blocks (``qsa/verify_topk.py``)."""
        from sgl_nvidia_mega.qwen3_8.qsa import verify_topk

        rows = scores.shape[0]
        # int32 [2 * rows]: the zero row starts, then the 512-thread kernel's flags
        scratch = torch.empty(2 * rows, dtype=torch.int32, device=scores.device)
        impl = verify_topk.impl_for(rows, scores.device)
        verify_topk.select_into(scores, lengths, out, scratch, impl)

    def _split_module(self, threads, cluster, iters, vec):
        key = ("split", threads, cluster, iters, vec)
        module = self._modules.get(key)
        if module is None:
            from sgl_nvidia_mega.qwen3_8 import _jit
            from sgl_nvidia_mega.qwen3_8.qsa import split_topk as source

            args = _jit.make_cpp_args(
                threads, cluster, iters, vec, _jit.is_arch_support_pdl()
            )
            module = _jit.load_jit(
                "qsa_cluster_topk",
                *args,
                cuda_files=[self._source_file("qsa_cluster_topk", source._SOURCE)],
                extra_include_paths=[str(_jit.KERNEL_PATH / "csrc")],
                cuda_wrappers=[("cluster_topk", f"QsaClusterTopKKernel<{args}>::run")],
            )
            self._modules[key] = module
        return module

    # -- build ------------------------------------------------------------------------

    def _source_file(self, stem: str, text: str) -> str:
        """The kernel source in the node-local directory, written once per digest."""
        if self.source_dir is None:
            raise RuntimeError("QSA indexer kernels are not built (no source_dir)")
        digest = hashlib.sha256(text.encode()).hexdigest()[:16]
        root = self.source_dir
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{stem}_{digest}.cuh"
        if not path.exists():
            tmp = root / f".{path.name}.{os.getpid()}"
            tmp.write_text(text)
            os.replace(tmp, path)
        return str(path)

    def compile(self, indexer, context_len=None, compressed_dtype=None) -> None:
        """Build what needs no step geometry; the split selects need the width
        (``context_len``). The first eager launch of each layout makes it warm."""
        o = self.options
        compressed_dtype = compressed_dtype or torch.bfloat16
        if o.prologue_fold:
            self._fold_module(self._fold_key(indexer, torch.bfloat16, compressed_dtype))
        if o.score and compressed_dtype == torch.bfloat16:
            device = torch.device("cuda", torch.cuda.current_device())
            heads = _padded(indexer.index_n_heads)
            for rows in range(1, SPLIT_ROWS + 1):
                self._score_kernel("v5", rows, heads, device, None)
                if rows % 4 == 0:
                    self._score_kernel("v4", rows, heads, device, None)
            if indexer.index_n_heads == 4:
                if o.grouped4 and self.sm107:
                    variant = "v7" if o.grouped4_stages == 2 else "v6"
                    self._score_kernel(variant, GROUPED4_ROWS, heads, device, None)
                if o.mk_score:
                    variant = "v3" if o.short32 else "v0"
                    self._score_kernel(variant, 4, heads, device, torch.int32)
        if o.select:
            from sgl_nvidia_mega.qwen3_8.qsa import prefill_select

            prefill_select._topk_launcher("sampled")
            prefill_select._topk_launcher("half")
            if context_len:
                width = compressed_capacity(context_len)
                for rows in range(1, SPLIT_ROWS + 1):
                    self._split_module(*split_config(rows, width), True)


def _project_qk(
    indexer,
    hidden_states,
    positions,
    pool=None,
    cache_loc=None,
    q_heads_padded=None,
    projected_qk=None,
):
    """``indexer.project_qk(...)``; with BF16 pins on, or on an already launched
    ``projected_qk`` (the source's ``project_qk`` argument), here, with ``index_qk_proj``
    through ``bf16_gemm_pins``."""
    if projected_qk is None and not bf16_gemm_pins.routed():
        return indexer.project_qk(
            hidden_states,
            positions,
            pool=pool,
            cache_loc=cache_loc,
            q_heads_padded=q_heads_padded,
        )
    if projected_qk is None:
        qk = bf16_gemm_pins.linear(indexer.index_qk_proj, hidden_states)
    else:
        qk = projected_qk
    token_k = qk[:, indexer.index_n_heads * indexer.index_head_dim :].reshape(
        -1, indexer.index_kv_heads, indexer.index_head_dim
    )
    if (
        pool is not None
        and cache_loc is not None
        and qk.shape[0] > 0
        and indexer._use_fused_prep(qk)
    ):
        from sglang.kernels.ops.attention.qsa_indexer import qsa_index_q_norm_rope_store

        key_state_buffer = pool.get_qsa_key_state_buffer(indexer.layer_id)
        q = qsa_index_q_norm_rope_store(
            qk,
            positions.long(),
            indexer.rotary_emb.cos_sin_cache,
            indexer._rope_axis_map(qk.device),
            indexer.q_layernorm.weight.data,
            cache_loc[: qk.shape[0]].long(),
            key_state_buffer.view(key_state_buffer.shape[0], -1),
            pool.qsa_rope_position_buffer,
            indexer.index_n_heads,
            indexer.rotary_emb.rotary_dim,
            indexer.q_layernorm.variance_epsilon,
            indexer.rotary_emb.is_neox_style,
            q_heads_padded=q_heads_padded,
            out_dtype=pool.qsa_compressed_dtype,
        )
        return q, token_k, True
    q_raw = qk[:, : indexer.index_n_heads * indexer.index_head_dim]
    q = indexer.q_layernorm(q_raw.reshape(-1, indexer.index_head_dim)).reshape(
        -1, indexer.index_n_heads, indexer.index_head_dim
    )
    q = indexer.apply_rope(positions, q)
    if pool is not None:
        q = q.to(pool.qsa_compressed_dtype)
    return q, token_k, False


def _score_factory(key):
    """The TileLang scorer of a ``_score_kernel`` key, as ``_providers/sparse.py``
    builds it (BF16, 16-key pages, rows of at most 512 blocks unscored)."""
    variant = key[0]
    if variant in ("v0", "v3"):
        if variant == "v3":
            from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.qsa_score_short import (
                _perf_score_kernel as factory,
            )
        else:
            from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.qsa_score import (
                _mk_score_kernel as factory,
            )
        request_dtype = {torch.int32: "int32", torch.int64: "int64"}[key[1]]
        return factory(
            min_scored_len=BLOCK_TOPK,
            request_dtype=request_dtype,
            long_ctas=MK_LONG_CTAS,
        )
    from sgl_nvidia_mega.qwen3_8.qsa import mqa

    _, heads, ctas = key
    args = dict(
        heads=heads,
        head_dim=HEAD_DIM,
        page_size=PAGE,
        min_scored_len=BLOCK_TOPK,
        dtype="bfloat16",
    )
    if variant == "v5":
        from sglang.srt.layers.attention.qsa.mqa import _tilelang_qsa_mqa_decode_kernel

        return _tilelang_qsa_mqa_decode_kernel(**args, ctas_per_row=ctas)
    if variant == "v4":
        return mqa._tilelang_qsa_mqa_decode_grouped_v7_kernel(
            **args, rows_per_group=4, ctas_per_group=ctas
        )
    if variant == "v7":
        return mqa._tilelang_qsa_mqa_decode_grouped4_pipe_kernel(
            **args, ctas_per_row=ctas, num_stages=2
        )
    return mqa._tilelang_qsa_mqa_decode_grouped4_chain_kernel(**args, ctas_per_row=ctas)
