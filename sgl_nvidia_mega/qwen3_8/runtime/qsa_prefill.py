# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team): main's
# python/sglang/srt/models/qwen4_exp.py (Qwen4ExpAttentionDecoderLayer.self_attention and
# _compute_qsa_topk_indices_eager), python/sglang/srt/models/qwen3_5.py
# (Qwen3_5AttentionDecoderLayer.forward_prepare_cuda_fused),
# python/sglang/kernels/ops/attention/fused_qk_rmsnorm_rope_gate.py
# (fused_qk_gemma_rmsnorm_rope_gate), python/sglang/srt/layers/attention/qsa/qsa_indexer.py
# (QSAIndexer._forward_impl's prefill branch and select_prefill_tokens) and
# python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py
# (QwenSparseAttnBackend.forward_extend and capture_mtp_sparse_indices), routed as the
# source routes them (union prefill and its output alias, row pack, host lengths, scale
# fold, row-tiled QK norm + RoPE + gate), and the plugin's fast prefill selection and
# TMA packed prefill.
"""A QSA layer's whole extend / prefill attention, outside SGLang's attention module.

``extend(executor, layer, x, st)`` returns what ``layer.self_attention`` returns for an
EXTEND or MIXED step, or None when it does not serve the step (nothing launched; the
caller runs ``self_attention``). It declines: no options (``executor.qsa_prefill`` is
None), a non-QSA layer, any other mode, a stream capture outside a breakable prefill
graph, CP, batch-invariant mode, MLP-sync padded rows, missing host lengths, a non-CUDA
input, and an attention backend that is not SGLang's ``QwenSparseAttnBackend``. Inside a
breakable prefill graph (the executor's prefill body) the indexer and the attention (2
and 3) run as one graph break on the batch's real rows (``prefill_graph.qsa``), the rest
in the captured segments. Otherwise it runs, in self_attention's order (the indexer is
never overlapped at prefill):

1. QKV + gate (``qsa.qkv``): ``qkv_proj``, then ``qk_rows`` (the row-tiled Gluon QK
   norm + RoPE + gate deinterleave, ``fused_qk_rmsnorm_rope_gate_rows``) for gated BF16
   calls of at least 1,024 rows with head 256 / rotary 64, else SGLang's
   ``fused_qk_gemma_rmsnorm_rope_gate``; ungated layers run ``_prepare_qkv_gate``.
2. The indexer (``qsa.indexer``): project, ring store and compression as SGLang's
   indexer does, then the selection: ``fast_select`` (the plugin's request-relative
   scorer and ordered top-k, ``qsa/prefill_select.py``) where the source offered
   it, else ``scale_fold`` (SGLang's chunked selection on the scale-folded relative
   scorer, ``qsa/prefill_scale_fold.py``), else SGLang's ``_compute_qsa_topk_indices``.
   An MTP draft's post-prefill extend captures its seed rows as SGLang does, with the
   seed rows from host lengths under ``host_path``.
3. The K/V store (SGLang's layer router) and the attention (``qsa``, ``qsa.pack``): a
   pass without cached prefixes runs ``union`` (row-group union attention,
   ``qsa/union_prefill.py``, 1,024 rows and up) or SGLang's sparse prefill kernel; a
   prefix-chunk pass packs the context (``pack_rows``, ``qsa/prefill_pack.py``, or
   SGLang's pack) and runs ``union``, else ``tma_prefill`` (``qsa/tma_prefill.py``, on
   passes stock runs with its 16-token, 1-warp tile), else SGLang's chunk kernel. Under
   ``host_path`` the K lengths go through pinned memory and the chunk kernel gets its
   grid from the host lengths.
4. The sigmoid output gate (SGLang's ``fused_sigmoid_mul``) and ``o_proj``.

Reads: the layer's projections and norms, the indexer metadata, the KV pool, the
request table and the step's host lengths. Writes: the new K/V at ``out_cache_loc``, the
indexer ring, RoPE positions and compressed keys (as SGLang's indexer does), and the MTP
seed rows when the backend captures them.

Numerics against SGLang's stock path: ``qk_rows`` is bitwise as the source's Triton
compiles the stock kernel (its claim, not shown here); ``pack_rows``, ``scale_fold``,
``host_path`` and ``tma_prefill`` are bitwise (the source's claims; ``tma_prefill`` only
on the passes stock runs with the same tile); ``fast_select`` selects the blocks the
stock relative selection selects (its scorer stores the stock logits bitwise; with
``scorer_tc`` the scorer is on tensor cores, the hardware's bits, so ties may move);
``union`` is the stock attention up to rounding (128-token union tiles and another MMA
order), a tolerance slot."""

from __future__ import annotations

import dataclasses
import logging
from typing import Optional

import torch
import torch.nn.functional as F

from sgl_nvidia_mega.qwen3_8.runtime import bf16_gemm_pins

logger = logging.getLogger(__name__)

STOCK = "stock"
# The plugin prefill selection's geometry (_providers/prefill_select.py supported()).
SELECT_HEADS, SELECT_HEAD_DIM, SELECT_TOPK, SELECT_RATIO = 4, 128, 512, 4
SELECT_MAX_CAPACITY = 131072
# The TMA packed prefill's geometry (_providers/packed_prefill.py supported()).
TMA_HEAD_DIM, TMA_SELECTED, TMA_MAX_GROUP = 256, 2051, 16
_FAST_SELECT_TOPK = ("sampled", "onepass")
_LOGITS_BUDGET_BYTES = 128 * 1024 * 1024
_ALIAS_LOGGED = [False]

SGLANG_READS = {
    "srt/models/qwen4_exp.py": (
        "Qwen4ExpAttentionDecoderLayer",
        "_shards_rows_over_cp",
        "is_qsa",
        "indexer",
        "self_attention",
        "_compute_qsa_topk_indices",
        "_compute_qsa_topk_indices_eager",
    ),
    "srt/models/qwen3_5.py": (
        "_prepare_qkv_gate",
        "forward_prepare_cuda_fused",
        "attn_output_gate",
        "qkv_proj",
        "q_size",
        "kv_size",
        "q_norm",
        "k_norm",
        "rotary_emb",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "attn",
        "o_proj",
    ),
    "srt/layers/radix_attention.py": (
        "layer_id",
        "tp_q_head_num",
        "tp_k_head_num",
        "tp_v_head_num",
        "qk_head_dim",
        "v_head_dim",
        "head_dim",
        "scaling",
    ),
    "kernels/ops/attention/fused_qk_rmsnorm_rope_gate.py": (
        "fused_qk_gemma_rmsnorm_rope_gate",
    ),
    "kernels/ops/elementwise/elementwise.py": ("fused_sigmoid_mul",),
    "srt/layers/attention/qsa/glue.py": (
        "get_qsa_indexer_metadata",
        "resolve_qsa_sparse_backend",
    ),
    "srt/layers/attention/qsa/qsa_indexer.py": (
        "_qsa_prefill_row_chunk_size",
        "select_prefill_tokens",
        "project_qk",
        "update_key_state_and_compress",
        "_pending_ring_slots",
        "layer_id",
        "index_n_heads",
        "index_kv_heads",
        "index_head_dim",
        "token_topk",
        "compress_ratio",
        "block_topk",
    ),
    "srt/layers/attention/qsa/metadata.py": (
        "get_token_to_batch_idx",
        "pending_ring_slots",
        "compress_member_rows",
        "token_to_kv_pool",
        "get_prefill_mqa_inputs",
        "prefill_max_row_keys",
        "is_cuda_graph",
        "req_pool_indices",
        "get_seqlens_expanded",
    ),
    "srt/layers/attention/qsa/kernel.py": ("expand_qsa_block_indices", "qsa_fast_topk"),
    "srt/layers/attention/qsa/mqa.py": ("HAS_TILELANG",),
    "srt/layers/attention/qsa/sparse_attn.py": (
        "_get_best_config",
        "pack_qsa_prefill_kv",
        "sparse_gqa_fwd_interface_triton",
        "sparse_gqa_fwd_interface_triton_ck",
    ),
    "srt/layers/attention/qwen_sparse_attn_backend.py": (
        "QwenSparseAttnBackend",
        "token_to_kv_pool",
        "req_to_token_pool",
        "req_to_token",
        "compress_ratio",
        "max_context_len",
        "_pad_extend_output",
        "should_capture_mtp_sparse_indices",
        "capture_mtp_sparse_indices",
        "_mtp_shared_sparse_indices",
        "capture",
    ),
    "srt/layers/attention/base_attn_backend.py": ("kv_layers",),
    "srt/mem_cache/qsa_kv_pool.py": ("qsa_compressed_page_size",),
    "srt/mem_cache/kv/router.py": ("layer", "store", "read", "key", "value"),
    "srt/layers/attention/graph_utils.py": ("padded_extend_real_tokens",),
    "srt/layers/cp/utils.py": ("is_cp_active",),
    "srt/batch_invariant_ops/__init__.py": ("is_batch_invariant_mode_enabled",),
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "srt/model_executor/forward_mode.py": ("is_extend_or_draft_extend_or_mixed",),
    "srt/model_executor/forward_batch_info.py": (
        "forward_mode",
        "positions",
        "out_cache_loc",
        "extend_seq_lens",
        "extend_seq_lens_cpu",
        "seq_lens_cpu",
        "req_pool_indices",
        "req_pool_indices_cpu",
        "attn_cp_metadata",
        "_attn_output",
    ),
    "srt/model_executor/runner_backend_utils/breakable_cuda_graph/__init__.py": (
        "is_in_breakable_cuda_graph",
    ),
    "srt/utils/device.py|srt/utils/common.py": ("async_h2d", "is_sm100_supported"),
}

# The SGLang functions whose body this module mirrors, by the digest of their code
# (test/unit/test_contract.py recomputes them on the installed SGLang)
SGLANG_MIRRORS = {
    "srt/models/qwen4_exp.py:Qwen4ExpAttentionDecoderLayer.self_attention": "e88bd51bae6665bc",
    "srt/models/qwen4_exp.py:Qwen4ExpAttentionDecoderLayer._compute_qsa_topk_indices_eager": "2cae5ae3a3167b95",
    "srt/models/qwen3_5.py:Qwen3_5AttentionDecoderLayer.forward_prepare_cuda_fused": "d10e23f5651990d0",
    "kernels/ops/attention/fused_qk_rmsnorm_rope_gate.py:fused_qk_gemma_rmsnorm_rope_gate": "191b6d63a9473398",
    "srt/layers/attention/qsa/qsa_indexer.py:QSAIndexer._forward_impl": "a9ac6848a4d7811c",
    "srt/layers/attention/qsa/qsa_indexer.py:QSAIndexer.select_prefill_tokens": "af28f4f745595f4c",
    "srt/layers/radix_attention.py:RadixAttention.forward": "5973e131d911edc8",
    "srt/layers/attention/qwen_sparse_attn_backend.py:QwenSparseAttnBackend.forward_extend": "529e03bea0a3823c",
    "srt/layers/attention/qwen_sparse_attn_backend.py:QwenSparseAttnBackend.capture_mtp_sparse_indices": "bc39db1cc9b58976",
}

# Reads and mirrors that need an SGLang change newer than the unit job's pin: the
# contract test skips each, naming the change, only where the installed SGLang lacks it.
SGLANG_REQUIRES = {
    "kernels/ops/attention/fused_qk_rmsnorm_rope_gate.py:fused_qk_gemma_rmsnorm_rope_gate": (
        "sglang#48366",
        ("618c8b9510090454",),
    ),
    "srt/layers/radix_attention.py:RadixAttention.forward": (
        "sglang#48980",
        ("79fcc925a1bdb5ee",),
    ),
}


@dataclasses.dataclass(frozen=True)
class Options:
    """The source's prefill knobs as values. Defaults: its high-concurrency
    tuning; ``MID_CONCURRENCY`` is its mid-concurrency one."""

    # SGLANG_OPT_QSA_UNION_PREFILL (high / mid: 1)
    union: bool = True
    # SGLANG_OPT_QSA_UNION_PREFILL_OUT_ALIAS (all 1): the union kernel writes
    # forward_batch._attn_output in place when that buffer is set (only inside a
    # breakable graph's attention break, prefill_graph.qsa)
    union_out_alias: bool = True
    # SGLANG_OPT_QSA_PREFILL_PACK_ROWS (high: 1, mid: 0)
    pack_rows: bool = True
    # SGLANG_OPT_QSA_PREFILL_HOST_PATH (all 1)
    host_path: bool = True
    # SGLANG_OPT_QSA_PREFILL_SCALE_FOLD (all 1)
    scale_fold: bool = True
    # SGLANG_OPT_FUSED_QK_NORM_ROPE_ROWS (high: 1, mid: unset)
    qk_rows: bool = True
    # SGLANG_ENABLE_QSA_PREFILL_FAST_SELECT (all 1); needs scale_fold, as there
    fast_select: bool = True
    # SGLANG_QSA_PREFILL_FAST_SELECT_TOPK (all "onepass")
    fast_select_topk: str = "onepass"
    # SGLANG_QSA_PREFILL_SCORER_TC (high: 1, mid: unset)
    scorer_tc: bool = True
    # SGLANG_QSA_PREFILL_FAST_SELECT_WIDE_LOGITS_MB (high: 256, mid: unset) and
    # _WIDE_MIN_KEYS (unset everywhere: 24576)
    wide_logits_mib: int = 256
    wide_min_keys: int = 24576
    # SGLANG_ENABLE_QSA_TMA_PREFILL (all 1) and SGLANG_QSA_TMA_PREFILL_STAGES (unset: 3)
    tma_prefill: bool = True
    tma_stages: int = 3

    def __post_init__(self):
        if self.fast_select_topk not in _FAST_SELECT_TOPK:
            raise ValueError(f"fast_select_topk is one of {_FAST_SELECT_TOPK}")
        if self.fast_select and not self.scale_fold:
            raise ValueError(
                "fast_select reproduces the scale-folded scores: set scale_fold"
            )
        if not 0 <= self.wide_logits_mib < 4096 or (
            self.wide_logits_mib
            and not (
                0 < self.wide_min_keys < 2048 * 256 and self.wide_min_keys % 256 == 0
            )
        ):
            raise ValueError(
                "wide_logits_mib is in [0, 4096) and wide_min_keys a positive multiple "
                "of 256 below 524288"
            )
        if not 1 <= self.tma_stages <= 8:
            raise ValueError("tma_stages is 1..8")


HIGH_CONCURRENCY = Options()
MID_CONCURRENCY = Options(
    pack_rows=False, qk_rows=False, scorer_tc=False, wide_logits_mib=0
)
DEFAULT = HIGH_CONCURRENCY


def build(model, options: Optional[Options] = DEFAULT) -> str:
    """Load or compile the enabled kernels on this rank before serving; '' or why the
    prefill attention stays stock. Triton and Gluon kernels compile at their first
    (eager) prefill call."""
    if options is None:
        return "no options"
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return "the QSA prefill kernels need an SM10x GPU"
    from sglang.srt.models.qwen4_exp import Qwen4ExpAttentionDecoderLayer

    m = model.model
    layers = [
        m.layers[i]
        for i in range(m.start_layer, m.end_layer)
        if isinstance(m.layers[i], Qwen4ExpAttentionDecoderLayer) and m.layers[i].is_qsa
    ]
    if not layers:
        return "no QSA layer"
    failures = []

    def attempt(name, fn):
        try:
            fn()
        except Exception as e:  # a failed kernel keeps the stock path
            failures.append(f"{name}: {type(e).__name__}: {e}")

    def union():
        from sgl_nvidia_mega.qwen3_8.qsa import union_prefill

        if not union_prefill._GLUON:
            raise RuntimeError("this Triton has no Gluon Blackwell API")
        for layer in layers:
            rows = union_prefill.rows_per_group(
                layer.attn.tp_q_head_num, layer.attn.tp_k_head_num
            )
            if rows not in union_prefill._MODULES:
                union_prefill._MODULES[rows] = union_prefill._build_module(rows)

    def qk_rows():
        from sgl_nvidia_mega.qwen3_8.kernels.ops.attention import (
            fused_qk_rmsnorm_rope_gate_rows as rows,
        )

        if not rows._HAS_GLUON:
            raise RuntimeError("this Triton has no Gluon")

    def pack_rows():
        from sgl_nvidia_mega.qwen3_8.qsa import prefill_pack  # noqa: F401

    def scale_fold():
        from sgl_nvidia_mega.qwen3_8.qsa import prefill_scale_fold

        if not prefill_scale_fold.HAS_TILELANG:
            raise RuntimeError("TileLang is unavailable")

    def fast_select():
        from sgl_nvidia_mega.qwen3_8.qsa import prefill_select

        prefill_select._topk_launcher("stock")
        prefill_select._topk_launcher(options.fast_select_topk)

    def tma_prefill():
        from sgl_nvidia_mega.qwen3_8.qsa import tma_prefill  # noqa: F401

    for name, enabled, fn in (
        ("union_prefill", options.union, union),
        ("qk_rows", options.qk_rows, qk_rows),
        ("pack_rows", options.pack_rows, pack_rows),
        ("scale_fold", options.scale_fold, scale_fold),
        ("fast_select", options.fast_select, fast_select),
        ("tma_prefill", options.tma_prefill, tma_prefill),
    ):
        if enabled:
            attempt(name, fn)
    return "; ".join(failures)


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def decline_reason(layer, x, fb, backend, *, graph=False) -> str:
    """'' when ``extend`` serves this layer's step, else why not. Host state only.
    ``graph``: a breakable prefill graph's capture, whose break reads the batch's host
    lengths and real rows (``prefill_graph.qsa``)."""
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
    from sglang.srt.layers.attention.graph_utils import padded_extend_real_tokens
    from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
        QwenSparseAttnBackend,
    )
    from sglang.srt.layers.cp.utils import is_cp_active
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        is_in_breakable_cuda_graph,
    )
    from sglang.srt.models.qwen4_exp import _shards_rows_over_cp

    if not layer.is_qsa:
        return "not a QSA layer"
    if not fb.forward_mode.is_extend_or_draft_extend_or_mixed():
        return "not an extend step"
    if not graph and (is_in_breakable_cuda_graph() or _capturing()):
        return "graph capture"
    if _shards_rows_over_cp(fb) or (
        fb.attn_cp_metadata is not None and is_cp_active(fb)
    ):
        return "context parallel"
    if is_batch_invariant_mode_enabled():
        return "batch-invariant mode"
    if not x.is_cuda:
        return "not a CUDA input"
    if not graph and (fb.extend_seq_lens_cpu is None or fb.seq_lens_cpu is None):
        return "no host lengths"
    if not graph and padded_extend_real_tokens(x.shape[0], fb) is not None:
        return "MLP-sync padded rows"
    if not isinstance(backend, QwenSparseAttnBackend):
        return f"attention backend {type(backend).__name__}"
    return ""


def extend(executor, layer, x, st):
    """The layer's attention output [rows, hidden] (before the stage's reduction), as
    ``layer.self_attention`` computes it, or None (nothing launched) for a step it does
    not serve. ``executor.qsa_prefill``: this module's ``Options`` once ``build`` passed,
    else None."""
    options = executor.qsa_prefill
    if options is None:
        return None
    from sglang.srt.layers.attention.qsa.glue import resolve_qsa_sparse_backend
    from sglang.srt.model_executor.forward_context import get_attn_backend
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        is_in_breakable_cuda_graph,
    )

    backend = resolve_qsa_sparse_backend(get_attn_backend())
    graph = is_in_breakable_cuda_graph()
    if decline_reason(layer, x, st.fb, backend, graph=graph):
        return None
    q, k, v, gate = qkv_gate(options, layer, x, st.positions, st)
    if graph:
        from sgl_nvidia_mega.qwen3_8.runtime import prefill_graph

        # the indexer and attention read the batch's host lengths: a graph break
        out = prefill_graph.qsa(executor, layer, x, q, k, v, st)
    else:
        out = attend(options, layer, x, q, k, v, backend, st)
    return finish(layer, out, gate)


def attend(options, layer, x, q, k, v, backend, st):
    """``self_attention``'s indexer, then the K/V store and the attention:
    [rows, heads * 256]."""
    indices = indexer(options, layer, x, st.positions, backend, st)
    return attention(options, layer.attn, backend, q, k, v, indices, st)


# -- QKV + gate ---------------------------------------------------------------------


def qkv_gate(options, layer, x, positions, st):
    """``layer._prepare_qkv_gate``: on CUDA with the output gate SGLang runs
    ``forward_prepare_cuda_fused``, whose fused norm + RoPE + gate call takes the
    row-tiled kernel here where it applies (as the source's hook in
    ``fused_qk_gemma_rmsnorm_rope_gate`` does)."""
    if not (options.qk_rows and layer.attn_output_gate):
        st.ran("qsa.qkv", STOCK)
        return layer._prepare_qkv_gate(
            positions=positions, hidden_states=x, forward_batch=st.fb
        )
    from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
        fused_qk_gemma_rmsnorm_rope_gate,
    )

    from sgl_nvidia_mega.qwen3_8.kernels.ops.attention import (
        fused_qk_rmsnorm_rope_gate_rows as rows,
    )

    qkv = bf16_gemm_pins.linear(layer.qkv_proj, x)
    q_gate, k, v = qkv.split([layer.q_size * 2, layer.kv_size, layer.kv_size], dim=-1)
    rope = layer.rotary_emb
    mrope = positions.dim() == 2
    axis_map = rope.axis_map if mrope else None
    tokens = q_gate.shape[0]
    head_dim = layer.head_dim
    # The stock function's [3, T] contract (it asserts it) before the rows hook
    stock_layout = not mrope or (positions.shape[0] == 3 and positions.stride(1) == 1)
    if stock_layout and rows.applies(
        q_gate, k, positions, tokens, head_dim, rope.rotary_dim, True
    ):
        q_out = torch.empty(
            tokens, layer.num_heads * head_dim, dtype=q_gate.dtype, device=q_gate.device
        )
        k_out = torch.empty(
            tokens, layer.num_kv_heads * head_dim, dtype=k.dtype, device=k.device
        )
        gate_out = torch.empty(
            tokens, layer.num_heads, head_dim, dtype=q_gate.dtype, device=q_gate.device
        )
        rows.launch(
            q_gate=q_gate,
            k=k,
            q_out=q_out,
            k_out=k_out,
            gate_out=gate_out,
            q_weight=layer.q_norm.weight.data,
            k_weight=layer.k_norm.weight.data,
            cos_sin_cache=rope.cos_sin_cache,
            positions=positions,
            mrope_axis_map=axis_map,
            eps=layer.q_norm.variance_epsilon,
            num_q_heads=layer.num_heads,
            num_kv_heads=layer.num_kv_heads,
        )
        st.ran("qsa.qkv", "qk_rows")
    else:
        q_out, k_out, gate_out = fused_qk_gemma_rmsnorm_rope_gate(
            q_gate,
            k,
            layer.q_norm.weight.data,
            layer.k_norm.weight.data,
            rope.cos_sin_cache,
            positions,
            layer.q_norm.variance_epsilon,
            layer.num_heads,
            layer.num_kv_heads,
            head_dim,
            rope.rotary_dim,
            has_gate=True,
            mrope_axis_map=axis_map,
        )
        st.ran("qsa.qkv", STOCK)
    return (
        q_out.view(tokens, -1),
        k_out.view(tokens, -1),
        v,
        gate_out.view(tokens, -1),
    )


# -- indexer ------------------------------------------------------------------------


def indexer(options, layer, x, positions, backend, st):
    """``layer._compute_qsa_topk_indices`` at prefill (no CP, no breakable graph):
    SGLang's indexer prologue, then the selection this module routes."""
    fb = st.fb
    if not (options.fast_select or options.scale_fold):
        st.ran("qsa.indexer", STOCK)
        return layer._compute_qsa_topk_indices(x, positions, fb)
    from sglang.srt.layers.attention.qsa.glue import get_qsa_indexer_metadata
    from sglang.srt.model_executor.forward_context import get_attn_backend

    md = get_qsa_indexer_metadata(get_attn_backend(), layer.layer_id, fb)
    topk, impl = _indexer_prefill(options, layer.indexer, x, positions, fb, md, backend)
    if backend.should_capture_mtp_sparse_indices(fb):
        capture_seeds(options, backend, topk, fb, layer.layer_id, md)
    st.ran("qsa.indexer", impl)
    return topk


def _indexer_prefill(options, ix, hidden_states, positions, fb, md, backend):
    """``QSAIndexer._forward_impl``'s prefill branch; returns (top-k, impl)."""
    logical_positions = fb.positions
    if logical_positions is None:
        logical_positions = positions[0] if positions.ndim == 2 else positions
    logical_positions = logical_positions.flatten()
    num_valid_tokens = md.get_token_to_batch_idx().numel()
    if logical_positions.numel() < num_valid_tokens:
        raise ValueError(
            "QSA logical positions are shorter than the request mapping: "
            f"positions={logical_positions.numel()}, mapping={num_valid_tokens}"
        )
    if hidden_states.shape[0] < num_valid_tokens:
        raise ValueError(
            "QSA hidden states are shorter than the request mapping: "
            f"hidden={hidden_states.shape[0]}, mapping={num_valid_tokens}"
        )
    position_tokens = positions.shape[-1] if positions.ndim == 2 else positions.numel()
    if position_tokens < num_valid_tokens:
        raise ValueError(
            "QSA RoPE positions are shorter than the request mapping: "
            f"positions={position_tokens}, mapping={num_valid_tokens}"
        )
    logical_positions = logical_positions[:num_valid_tokens]
    hidden_states = hidden_states[:num_valid_tokens]
    positions = (
        positions[:, :num_valid_tokens]
        if positions.ndim == 2
        else positions[:num_valid_tokens]
    )
    state_slots = md.pending_ring_slots
    if state_slots is None:
        state_slots = ix._pending_ring_slots(
            md, logical_positions, md.compress_member_rows is not None
        )
    q, token_k, state_stored = ix.project_qk(
        hidden_states,
        positions,
        pool=md.token_to_kv_pool,
        cache_loc=state_slots,
        q_heads_padded=None,
    )
    ix.update_key_state_and_compress(
        token_k,
        logical_positions,
        positions,
        md,
        state_slots=state_slots,
        state_stored=state_stored,
    )
    compressed_keys, row_starts, row_ends, sequence_lengths = md.get_prefill_mqa_inputs(
        ix.layer_id, logical_positions
    )
    row_sequence_lengths = sequence_lengths.index_select(
        0, md.get_token_to_batch_idx().long()
    )
    capacity = compressed_capacity(
        backend.max_context_len, md.token_to_kv_pool.qsa_compressed_page_size
    )
    return select_prefill(
        options,
        ix,
        q,
        compressed_keys,
        row_starts,
        row_ends,
        logical_positions,
        row_sequence_lengths,
        md.prefill_max_row_keys(fb.seq_lens_cpu),
        capacity,
    )


def select_prefill(
    options,
    ix,
    q,
    compressed_keys,
    row_starts,
    row_ends,
    query_positions,
    sequence_lengths_for_rows,
    max_row_keys,
    capacity,
):
    """``QSAIndexer.select_prefill_tokens``; returns (top-k, impl). A request-relative
    call runs the fast select where it serves, else the chunked selection on the
    scale-folded scorer; any other call is SGLang's. ``capacity``: the compressed keys
    the fast select is declared for (``compressed_capacity``)."""
    from sglang.srt.layers.attention.qsa.mqa import HAS_TILELANG

    rows = q.shape[0]
    relative = (
        max_row_keys is not None and max_row_keys > 0 and q.is_cuda and HAS_TILELANG
    )
    if rows == 0 or not (relative and options.scale_fold):
        return (
            ix.select_prefill_tokens(
                q,
                compressed_keys,
                row_starts,
                row_ends,
                query_positions,
                sequence_lengths_for_rows,
                max_row_keys=max_row_keys,
            ),
            STOCK,
        )
    output = torch.empty(
        (rows, ix.token_topk + ix.compress_ratio - 1),
        dtype=torch.int32,
        device=q.device,
    )
    if (
        options.fast_select
        and fast_select_reason(ix, q, compressed_keys, output, max_row_keys, capacity)
        == ""
    ):
        fast_select(
            options,
            q,
            compressed_keys,
            row_starts,
            row_ends,
            query_positions,
            sequence_lengths_for_rows,
            output,
            max_row_keys,
        )
        return output, "fast_select"
    _select_scale_fold(
        ix,
        q,
        compressed_keys,
        row_starts,
        row_ends,
        query_positions,
        sequence_lengths_for_rows,
        output,
        max_row_keys,
    )
    return output, "scale_fold"


def _select_scale_fold(
    ix, q, keys, row_starts, row_ends, positions, row_lengths, output, max_row_keys
):
    """``select_prefill_tokens``' relative loop with the scale-folded scorer (the
    source's ``tilelang_qsa_mqa_prefill_relative`` under its scale fold)."""
    from sglang.srt.layers.attention.qsa.kernel import (
        expand_qsa_block_indices,
        qsa_fast_topk,
    )
    from sglang.srt.layers.attention.qsa.qsa_indexer import _qsa_prefill_row_chunk_size

    from sgl_nvidia_mega.qwen3_8.qsa.prefill_scale_fold import (
        tilelang_qsa_mqa_prefill_relative,
    )

    rows = q.shape[0]
    row_chunk_size = _qsa_prefill_row_chunk_size(rows, max_row_keys, q.shape[1])
    for row_start in range(0, rows, row_chunk_size):
        row_end = min(row_start + row_chunk_size, rows)
        chunk_slice = slice(row_start, row_end)
        if keys.shape[0] == 0:
            block_indices = torch.full(
                (row_end - row_start, ix.block_topk),
                -1,
                dtype=torch.int32,
                device=q.device,
            )
            logits = None
        else:
            chunk_starts = row_starts[chunk_slice]
            chunk_lengths = row_ends[chunk_slice] - chunk_starts
            logits = tilelang_qsa_mqa_prefill_relative(
                q[chunk_slice],
                keys,
                chunk_starts,
                row_ends[chunk_slice],
                max_row_keys,
                scale_fold=True,
            )
            block_indices = qsa_fast_topk(
                logits,
                torch.zeros_like(chunk_starts),
                chunk_lengths,
                topk=ix.block_topk,
            )
        selected = expand_qsa_block_indices(
            block_indices,
            positions[chunk_slice],
            row_lengths[chunk_slice],
            compress_ratio=ix.compress_ratio,
            token_topk=ix.token_topk,
        )
        output[chunk_slice].copy_(selected)
        del logits, block_indices, selected


def compressed_capacity(context_len: int, page_size: int) -> int:
    """The compressed keys a row of ``context_len`` tokens spans in whole compressed
    pages, the bound the source declared its prefill select for (its binding
    passes the server's context length and the pool's ``qsa_compressed_page_size``);
    0 when either is unknown."""
    if not (context_len and page_size):
        return 0
    blocks = -(-int(context_len) // SELECT_RATIO)
    return -(-blocks // int(page_size)) * int(page_size)


def fast_select_reason(ix, q, keys, output, width, capacity) -> str:
    """'' when the plugin's fast select takes this call (the source's
    ``_numeric_prefill_select`` admission and the provider's layout checks)."""
    if not 0 < capacity <= SELECT_MAX_CAPACITY:
        return f"capacity {capacity} keys"
    if type(width) is not int or not 0 < width <= capacity:
        return f"width {width}"
    geometry = (
        ix.index_n_heads,
        ix.index_kv_heads,
        ix.index_head_dim,
        ix.block_topk,
        ix.compress_ratio,
    )
    if geometry != (SELECT_HEADS, 1, SELECT_HEAD_DIM, SELECT_TOPK, SELECT_RATIO):
        return f"index geometry {geometry}"
    if (
        q.dtype != torch.bfloat16
        or q.ndim != 3
        or tuple(q.shape[1:]) != (SELECT_HEADS, SELECT_HEAD_DIM)
        or q.shape[0] == 0
    ):
        return "query layout"
    if (
        keys.ndim != 3
        or keys.shape[0] == 0
        or tuple(keys.shape[1:]) != (1, SELECT_HEAD_DIM)
        or keys.dtype != q.dtype
    ):
        return "compressed key layout"
    if tuple(output.shape) != (q.shape[0], ix.token_topk + ix.compress_ratio - 1) or (
        output.shape[1] != SELECT_TOPK * SELECT_RATIO + SELECT_RATIO - 1
    ):
        return "output layout"
    return ""


def logits_budget(options, width: int) -> int:
    """The source budget rule: the wide budget for calls at least wide_min_keys wide."""
    if options.wide_logits_mib and width >= options.wide_min_keys:
        return options.wide_logits_mib << 20
    return _LOGITS_BUDGET_BYTES


@dataclasses.dataclass
class SelectIO:
    """The fields ``qsa/prefill_select.select_into`` reads (SGLang's binding branch's
    ``PrefillSelectIO``, which SGLang main does not have)."""

    query: torch.Tensor
    compressed_keys: torch.Tensor
    row_starts: torch.Tensor
    row_ends: torch.Tensor
    query_positions: torch.Tensor
    sequence_lengths: torch.Tensor
    selected_indices_out: torch.Tensor
    query_scratch_out: torch.Tensor
    starts_scratch_out: torch.Tensor
    ends_scratch_out: torch.Tensor
    row_lengths: torch.Tensor
    zero_offsets: torch.Tensor
    block_indices_out: torch.Tensor
    logits_out: torch.Tensor
    split_tag: torch.Tensor
    chunk_rows: int
    scorer_sms: int


def fast_select(
    options, q, keys, starts, ends, positions, lengths, output, width
) -> None:
    """Fill ``output`` as SGLang's relative prefill selection does, with the plugin's
    scorer and ordered top-k (the source's ``_numeric_prefill_select``)."""
    from sgl_nvidia_mega.qwen3_8.qsa import pool_decode
    from sgl_nvidia_mega.qwen3_8.qsa import prefill_select as raw

    rows = q.shape[0]
    sms = pool_decode._num_sms(q.device)
    chunk = raw.plan_chunk_rows(rows, width, 2 * sms, logits_budget(options, width))
    padded = (chunk + 31) // 32 * 32
    starts, ends, positions, lengths = (
        t.to(device=q.device, dtype=torch.int32).contiguous()
        for t in (starts, ends, positions, lengths)
    )
    i32 = dict(device=q.device, dtype=torch.int32)
    io = SelectIO(
        q.contiguous(),
        keys.contiguous(),
        starts,
        ends,
        positions,
        lengths,
        output,
        torch.empty((padded, *q.shape[1:]), device=q.device, dtype=q.dtype),
        torch.empty(padded, **i32),
        torch.empty(padded, **i32),
        ends - starts,
        torch.zeros(chunk, **i32),
        torch.empty((rows, SELECT_TOPK), **i32),
        torch.empty((padded, width), device=q.device, dtype=torch.float32),
        torch.empty(4096, **i32),
        chunk,
        sms,
    )
    raw.select_into(
        io, width, scorer_tc=options.scorer_tc, topk_impl=options.fast_select_topk
    )


def _last_rows_of_runs(*, extend_lens, pool_ids, num_rows: int) -> list:
    """Host twin of ``is_last.nonzero()`` in capture_mtp_sparse_indices.

    Rows are laid out as ``repeat_interleave(arange(bs), extend_lens)`` cut to the
    first ``num_rows``; a row is last when the next row's request-pool id differs,
    and the final row always is.
    """
    limit = min(int(num_rows), sum(extend_lens))
    rows = []
    start = 0
    # (pool id, last row inside the limit) of the previous non-empty request
    prev = None
    for extend_len, pool_id in zip(extend_lens, pool_ids):
        if extend_len <= 0:
            continue
        if start >= limit:
            break
        if prev is not None and prev[0] != pool_id:
            rows.append(prev[1])
        start += extend_len
        prev = (pool_id, min(start, limit) - 1)
    if prev is not None:
        rows.append(prev[1])
    return rows


def host_anchor_rows(fb, md, num_rows: int):
    """The seed rows of capture_mtp_sparse_indices' eager path from the step's host
    lengths and request-pool ids (the source's ``_mtp_anchor_host_source`` and
    ``_mtp_anchor_host_rows``), or None when the host lists do not line up."""
    batch_size = md.req_pool_indices.shape[0]
    host_lens = fb.extend_seq_lens_cpu
    host_pool = fb.req_pool_indices_cpu
    extend_seq_lens = fb.extend_seq_lens
    if (
        host_lens is None
        or host_pool is None
        or extend_seq_lens is None
        or len(host_lens) != batch_size
        or extend_seq_lens.dim() != 1
        or extend_seq_lens.shape[0] != batch_size
        or host_pool.dim() != 1
        or host_pool.shape[0] < batch_size
    ):
        return None
    return _last_rows_of_runs(
        extend_lens=tuple(int(x) for x in host_lens),
        pool_ids=tuple(int(x) for x in host_pool[:batch_size].tolist()),
        num_rows=num_rows,
    )


def capture_seeds(options, backend, topk_indices, fb, layer_id, md) -> None:
    """``QwenSparseAttnBackend.capture_mtp_sparse_indices`` for a prefill step; under
    ``host_path`` the eager path's seed rows come from host lengths, not a device
    ``nonzero``."""
    if (
        not options.host_path
        or topk_indices.shape[0] == 0
        or md.is_cuda_graph
        or fb.forward_mode.is_draft_extend_v2()
        or md.req_pool_indices is None
        or md.req_pool_indices.numel() == 0
    ):
        backend.capture_mtp_sparse_indices(topk_indices, fb, layer_id, metadata=md)
        return
    rows = host_anchor_rows(fb, md, topk_indices.shape[0])
    if rows is None:
        backend.capture_mtp_sparse_indices(topk_indices, fb, layer_id, metadata=md)
        return
    from sglang.srt.utils import async_h2d

    state = backend._mtp_shared_sparse_indices
    row_to_req = md.get_token_to_batch_idx().long()
    row_req_pool_indices = md.req_pool_indices[row_to_req[: topk_indices.shape[0]]]
    anchor_rows = async_h2d(rows, dtype=torch.long, device=row_req_pool_indices.device)
    req_rows = row_req_pool_indices[anchor_rows]
    captured_lens = md.get_seqlens_expanded()[anchor_rows]
    state.capture(topk_indices[anchor_rows], req_rows, captured_lens, layer_id)


# -- K/V store and attention --------------------------------------------------------


def attention(options, attn, backend, q, k, v, topk_indices, st):
    """``RadixAttention.forward`` + ``QwenSparseAttnBackend.forward_extend`` for an
    eager, non-CP extend: the K/V store, then the attention this module routes; returns
    [rows, heads * 256], zero-padded past the indexer's rows as SGLang pads."""
    from sglang.srt.layers.attention.qsa.sparse_attn import (
        sparse_gqa_fwd_interface_triton,
        sparse_gqa_fwd_interface_triton_ck,
    )

    fb = st.fb
    # RadixAttention.forward's views
    k = k.view(-1, attn.tp_k_head_num, attn.qk_head_dim)
    v = v.view(-1, attn.tp_v_head_num, attn.v_head_dim)
    # forward_extend: the fused K/V attention serves decode and verify only
    backend.kv_layers.layer(attn.layer_id).store(attn, fb.out_cache_loc, k, v)
    q = q.reshape(-1, attn.tp_q_head_num, attn.head_dim)
    num_output_rows = q.shape[0]
    num_valid_rows = topk_indices.shape[0]
    if num_valid_rows > num_output_rows:
        raise ValueError(
            "QSA top-k rows exceed query rows: "
            f"topk={num_valid_rows}, query={num_output_rows}"
        )
    q = q[:num_valid_rows]
    topk_indices = topk_indices.to(torch.int32).contiguous()
    extend_lens = [int(x) for x in fb.extend_seq_lens_cpu]
    sequence_lens = [int(x) for x in fb.seq_lens_cpu]
    prefix_lens = [sequence_lens[i] - extend_lens[i] for i in range(len(extend_lens))]
    cu_seqlens_q = F.pad(
        fb.extend_seq_lens.to(q.device, dtype=torch.int32).cumsum(0), (1, 0)
    ).contiguous()
    union_out = (
        union_output(q, fb, num_output_rows)
        if options.union and options.union_out_alias
        else None
    )
    pad = backend._pad_extend_output
    if not any(prefix_lens):
        if options.union:
            output = _union_cold(
                q=q.contiguous(),
                out=union_out,
                k=k[:num_valid_rows].contiguous(),
                v=v[:num_valid_rows].contiguous(),
                attn=attn,
                fb=fb,
                topk_indices=topk_indices,
                cu_seqlens_q=cu_seqlens_q,
                extend_lens=extend_lens,
                sequence_lens=sequence_lens,
            )
            if output is not None:
                st.ran("qsa", "union")
                return pad(output, num_output_rows)
        output = sparse_gqa_fwd_interface_triton(
            q.contiguous(),
            k[:num_valid_rows].contiguous(),
            v[:num_valid_rows].contiguous(),
            max(sequence_lens, default=1),
            topk_indices,
            cu_seqlens_q,
            attn.scaling,
        )
        st.ran("qsa", STOCK)
        return pad(output, num_output_rows)

    # Current-chunk K/V has already been committed to the cache above.
    max_q = None
    if options.host_path:
        from sglang.srt.utils import async_h2d

        # Staged in pinned memory, so the copy does not drain the stream; max_q is
        # the host max of the lens cu_seqlens_q is built from.
        sequence_lens_tensor = async_h2d(
            sequence_lens, dtype=torch.int32, device=q.device
        )
        if len(extend_lens) == fb.extend_seq_lens.shape[0]:
            max_q = max(extend_lens)
    else:
        sequence_lens_tensor = torch.tensor(
            sequence_lens, dtype=torch.int32, device=q.device
        )
    cu_seqlens_k = F.pad(sequence_lens_tensor.cumsum(0), (1, 0)).contiguous()
    layer_kv = backend.token_to_kv_pool.layer_router.read(attn.layer_id)
    pack_args = dict(
        k=layer_kv.key(),
        v=layer_kv.value(),
        req_to_token=backend.req_to_token_pool.req_to_token,
        req_indices=fb.req_pool_indices,
        cu_k=cu_seqlens_k,
        total_k=sum(sequence_lens),
        max_k=max(sequence_lens, default=1),
        output_dtype=q.dtype,
    )
    if options.union:
        output = _union_chunk(
            options,
            q=q.contiguous(),
            out=union_out,
            attn=attn,
            topk_indices=topk_indices,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            extend_lens=extend_lens,
            sequence_lens=sequence_lens,
            sequence_lens_tensor=sequence_lens_tensor,
            max_q=max_q,
            pack_args=pack_args,
            st=st,
        )
        if output is not None:
            return pad(output, num_output_rows)
    packed_k, packed_v = pack(options, pack_args, st)
    output = None
    if options.tma_prefill:
        output = tma_prefill(
            options,
            q.contiguous(),
            attn,
            fb,
            packed_k,
            packed_v,
            topk_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            sequence_lens_tensor,
            extend_lens,
            backend.compress_ratio,
        )
    if output is None:
        output = sparse_gqa_fwd_interface_triton_ck(
            q.contiguous(),
            packed_k,
            packed_v,
            topk_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            sequence_lens_tensor,
            attn.scaling,
            max_q=max_q,
        )
        st.ran("qsa", STOCK)
    else:
        st.ran("qsa", "tma_prefill")
    return pad(output, num_output_rows)


def pack(options, pack_args, st):
    """``pack_qsa_prefill_kv``: the row-tiled pack where it applies (bitwise), else
    SGLang's."""
    if options.pack_rows:
        from sgl_nvidia_mega.qwen3_8.qsa import prefill_pack

        packed = prefill_pack.pack_rows(**pack_args)
        if packed is not None:
            st.ran("qsa.pack", "pack_rows")
            return packed
    from sglang.srt.layers.attention.qsa.sparse_attn import pack_qsa_prefill_kv

    st.ran("qsa.pack", STOCK)
    return pack_qsa_prefill_kv(**pack_args)


def union_output(q, fb, num_output_rows):
    """The source's ``_union_prefill_out``: ``fb._attn_output`` as the union
    kernel's [rows, heads, 256] output when it is exactly q's rows in q's dtype and
    contiguous; else None. Set only inside a breakable prefill graph's attention
    break (``prefill_graph.qsa``)."""
    out = fb._attn_output
    if (
        out is None
        or q.shape[0] != num_output_rows
        or out.shape[0] != q.shape[0]
        or out.numel() != q.numel()
        or out.dtype != q.dtype
        or out.device != q.device
        or not out.is_contiguous()
    ):
        return None
    if not _ALIAS_LOGGED[0]:
        _ALIAS_LOGGED[0] = True
        logger.info(
            "QSA union prefill writes the attention output buffer in place "
            "(rows=%d heads=%d)",
            q.shape[0],
            q.shape[1],
        )
    return out.view(q.shape)


def _union_chunk(
    options,
    *,
    q,
    attn,
    topk_indices,
    cu_seqlens_q,
    cu_seqlens_k,
    extend_lens,
    sequence_lens,
    sequence_lens_tensor,
    max_q,
    pack_args,
    st,
    out=None,
):
    """The source's ``_qsa_union_chunk_prefill``: the pack, then row-group union
    attention; None (nothing launched) when the union does not take the pass."""
    from sglang.srt.layers.attention.qsa.sparse_attn import (
        sparse_gqa_fwd_interface_triton_ck,
    )

    from sgl_nvidia_mega.qwen3_8.qsa import union_prefill

    kv_max = pack_args["max_k"]
    total_k = pack_args["total_k"]
    if not union_prefill.applies(
        q,
        topk_indices,
        kv_heads=pack_args["k"].shape[1],
        kv_dtype=q.dtype,
        extend_lens=extend_lens,
        kv_max=kv_max,
        total_k=total_k,
    ):
        return None
    packed_k, packed_v = pack(options, pack_args, st)
    kernel_args = (
        q,
        packed_k,
        packed_v,
        topk_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        sequence_lens_tensor,
        attn.scaling,
    )
    if not union_prefill.kv_ok(packed_k, packed_v):
        st.ran("qsa", STOCK)
        return sparse_gqa_fwd_interface_triton_ck(*kernel_args, max_q=max_q)
    union_prefill.log_once(q, packed_k, topk_indices)
    st.ran("qsa", "union")
    return union_prefill.checked(
        lambda: union_prefill.union_prefill(
            *kernel_args, extend_lens=extend_lens, kv_max=kv_max, out=out
        ),
        lambda: sparse_gqa_fwd_interface_triton_ck(*kernel_args, max_q=max_q),
        q.shape[0],
        kv_max,
        "replaced",
    )


def _union_cold(
    *,
    q,
    k,
    v,
    attn,
    fb,
    topk_indices,
    cu_seqlens_q,
    extend_lens,
    sequence_lens,
    out=None,
):
    """The source's ``_qsa_union_cold_prefill``: a pass without cached prefixes,
    the chunk's own K/V as the context; None keeps the stock kernel."""
    from sglang.srt.layers.attention.qsa.sparse_attn import (
        sparse_gqa_fwd_interface_triton,
    )

    from sgl_nvidia_mega.qwen3_8.qsa import union_prefill

    kv_max = max(extend_lens, default=1)
    if not union_prefill.applies(
        q,
        topk_indices,
        kv_heads=k.shape[1],
        kv_dtype=k.dtype,
        extend_lens=extend_lens,
        kv_max=kv_max,
        total_k=k.shape[0],
    ):
        return None
    if k.ndim != 3 or v.dtype != k.dtype or not union_prefill.kv_ok(k, v):
        return None
    union_prefill.log_once(q, k, topk_indices)
    kv_lens = fb.extend_seq_lens.to(q.device, dtype=torch.int32)
    return union_prefill.checked(
        lambda: union_prefill.union_prefill(
            q,
            k,
            v,
            topk_indices,
            cu_seqlens_q,
            cu_seqlens_q,
            kv_lens,
            attn.scaling,
            extend_lens=extend_lens,
            kv_max=kv_max,
            out=out,
        ),
        lambda: sparse_gqa_fwd_interface_triton(
            q,
            k,
            v,
            max(sequence_lens, default=1),
            topk_indices,
            cu_seqlens_q,
            attn.scaling,
        ),
        q.shape[0],
        kv_max,
        "stock",
    )


@dataclasses.dataclass
class PackedIO:
    """The fields ``qsa/tma_prefill.launch`` reads (SGLang's binding branch's
    ``PackedPrefillAttentionIO``, which SGLang main does not have)."""

    query: torch.Tensor
    packed_k: torch.Tensor
    packed_v: torch.Tensor
    selected_indices: torch.Tensor
    cu_query: torch.Tensor
    cu_key: torch.Tensor
    sequence_lengths: torch.Tensor
    scale: float
    max_query_rows: int
    output: torch.Tensor


def tma_reason(q, k, v, indices, cu_q, cu_k, kv_lens, extend_lens, fb, ratio) -> str:
    """'' when the TMA packed prefill takes this pass: the passes stock runs with its
    (16, 1, 2) tile (the source's ``_numeric_packed_prefill`` admission) in the
    provider's layout."""
    from sglang.srt.layers.attention.qsa.sparse_attn import _get_best_config
    from sglang.srt.utils import is_sm100_supported

    if not is_sm100_supported():
        return "not SM100 family"
    if len(extend_lens) != fb.extend_seq_lens.shape[0] or not extend_lens:
        return "host lengths"
    if _get_best_config(q.shape[0]) != (16, 1, 2):
        return "stock tile"
    if not (q.dtype == k.dtype == v.dtype == torch.bfloat16):
        return "dtype"
    if (
        q.ndim != 3
        or k.ndim != 3
        or k.shape != v.shape
        or k.shape[1] == 0
        or q.shape[1] % k.shape[1]
        or q.shape[1] // k.shape[1] > TMA_MAX_GROUP
        or q.shape[2] != k.shape[2]
        or q.shape[2] != TMA_HEAD_DIM
        or q.stride(2) != 1
    ):
        return "heads"
    if (
        not k.is_contiguous()
        or not v.is_contiguous()
        or k.shape[0] >= 2**31 - 1
        or k.data_ptr() % 16
        or v.data_ptr() % 16
        or k.stride(0) * k.element_size() % 16
    ):
        return "K/V layout"
    if (
        indices.dtype != torch.int32
        or indices.ndim != 2
        or indices.stride(1) != 1
        or indices.shape != (q.shape[0], TMA_SELECTED)
        or ratio != 4
    ):
        return "indices"
    if (
        cu_q.ndim != 1
        or cu_q.shape[0] < 2
        or cu_k.shape != cu_q.shape
        or kv_lens.shape != (cu_q.shape[0] - 1,)
        or any(
            t.dtype not in (torch.int32, torch.int64) or not t.is_contiguous()
            for t in (cu_q, cu_k, kv_lens)
        )
    ):
        return "lengths"
    if not 0 < max(extend_lens) <= q.shape[0]:
        return "query rows"
    return ""


def tma_prefill(
    options, q, attn, fb, k, v, indices, cu_q, cu_k, kv_lens, extend_lens, ratio
):
    """The stock prefix-chunk kernel's tile with TMA-gathered K/V
    (``qsa/tma_prefill.py``), or None."""
    if tma_reason(q, k, v, indices, cu_q, cu_k, kv_lens, extend_lens, fb, ratio):
        return None
    from sgl_nvidia_mega.qwen3_8.qsa import tma_prefill as raw

    output = torch.empty_like(q)
    max_q = max(extend_lens)
    io = PackedIO(q, k, v, indices, cu_q, cu_k, kv_lens, attn.scaling, max_q, output)
    raw.launch(io, max_q=max_q, stages=options.tma_stages)
    return output


# -- gate and o_proj ----------------------------------------------------------------


def finish(layer, attn_output, gate):
    """``self_attention``'s tail on CUDA: the sigmoid output gate, then o_proj."""
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

    if gate is not None:
        attn_output = fused_sigmoid_mul(attn_output, gate, inplace=True)
    return bf16_gemm_pins.linear(layer.o_proj, attn_output)
