# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team): main's python/sglang/srt/models/qwen4_exp.py
# (_prepare_ple_batch, _commit_ple_batch, Qwen4ExpLayerExtensionMixin._add_ple_embedding,
# Qwen4ExpPLELayer.forward / _short_conv / start_prefetch, Qwen4ExpNGramEmbedding.forward /
# _embed_ngram_ids and Qwen4ExpPinnedHostEmbedding.forward / gather), with the source's
# kernels and routing: the batched target-verify layout, the one-launch verify conv,
# the gate after its reduction, the extend short conv over the token rows, the chunked
# prefill short conv and several host-table rows per program.
"""The executor's PLE pieces: SGLang main's functions with the source's kernels
where they serve, each falling back to main's own function.

``prepare(st, input_ids)`` replaces ``_prepare_ple_batch``, ``prefetch(layer, batch, st)``
``layer.ple.start_prefetch``, ``add(layer, residual, batch, st)`` the layer's
``_add_ple_embedding`` and ``commit(batch, st)`` ``_commit_ple_batch``. The first three
take the built ``Options`` as ``options`` (None: main's function). Engagement ops:

* ``ple.prepare``: ``verify_layout`` (``qwen4_ple_verify_prepare``: a target verify's
  rows, offsets, valid mask, state slots, N-gram history and windows in one integer-only
  launch, at most 256 requests, N-gram size 3) or stock.
* ``ple.prefetch`` and ``ple.gather``: ``gather_rows`` (``qwen4_ple_gather``: the pinned
  host table read ``gather_rows`` rows a program, at most ``gather_programs`` programs)
  where the table is offloaded to host, or stock.
* ``ple.gate``: ``gate_post`` (``qwen4_gate_post``: the gate after torch's reduction as
  one kernel, outside the decode fast path and the fused verify reduce, once its startup
  self-test passed) or stock (SGLang's chain or its fused verify / decode kernels).
* ``ple.conv``: on a target verify with SGLang's PLE fusion, ``verify_conv`` (the state
  window, the depthwise conv and the finish in one launch, below the 32-bit index limit);
  on EXTEND / MIXED, ahead of SGLang's own varlen kernel, ``extend_conv``
  (``qwen4_ple_fused``: the conv over the token rows, passes stock runs natively, and with
  ``extend_conv_cudnn`` its cuDNN shapes up to 16 sequences of 32,768 tokens) or
  ``chunked`` (``ple_short_conv``: the stock padded conv in chunks of ``conv_chunk_size``
  output tokens); else stock (``_short_conv`` / ``fused_qwen4_verify_conv``).
* ``ple.add`` stock: the whole add ran ``_add_ple_embedding`` (no piece served, an idle
  or breakable-graph step, or CP); ``ple.commit`` is always SGLang's.

Inside a breakable prefill graph (the executor's prefill body) each of the four runs as
a graph break on the live batch (``prefill_graph.py``), these functions in its body.

Reads and writes what SGLang's functions do: the N-gram history and the short-conv
state by state slot (and the verify intermediate windows, the track slots), the PLE
table, the request pool's window cache.

Numerics against SGLang main: ``verify_layout`` writes the torch branch's tensors bit
for bit, with one exception noted by the source rule: its state slot is the request
table's slot through the state id space (int32 under a virtual id space), which main's
torch branch reads from the attention metadata's resolved slots as int64; the values
agree where those slots are that table's translation (SGLang's ``_forward_metadata``).
``gather_rows``, ``gate_post`` (self-tested bitwise over every BF16 gate input),
``verify_conv``, ``extend_conv`` (passes stock runs on the native depthwise kernel) and
``chunked`` are bitwise to the stock code (the source's claims); ``extend_conv_cudnn``
matched cuDNN's bytes on one torch / cuDNN build only (the source's measurement)."""

from __future__ import annotations

import dataclasses
import math
from typing import Optional

import torch
import torch.nn.functional as F

STOCK = "stock"

SGLANG_READS = {
    "srt/models/qwen4_exp.py": (
        "_PLEBatch",
        "_prepare_ple_batch",
        "_commit_ple_batch",
        "_get_ple_forward_mode",
        "_get_processed_token_count",
        "_ple_track_targets",
        "_pad_token_rows",
        "_use_qwen4_varlen_prefill",
        "_shards_rows_over_cp",
        "Qwen4ExpPinnedHostEmbedding",
        "_add_ple_embedding",
        "_ple_forward_batch",
        "_ple_batch",
        "ple",
        "has_ple",
        "ple_ngram_size",
        "ple_ngram_eos_token_id",
        "start_prefetch",
        "forward_idle",
        "_short_conv",
        "_prefetch_stream",
        "_prefetch_state",
        "_get_prefetch_buffer",
        "_consume_prefetched_embeddings",
        "_apply_ple_norm",
        "ple_embedding",
        "ngram_embedding",
        "ngram_heads",
        "gather_dp_tokens",
        "enable_ple_fusion",
        "compute_ngram_ids",
        "_prepare_embedding_lookup",
        "_finish_embedding_lookup",
        "key_proj",
        "value_proj",
        "norm_key",
        "norm_query",
        "norm_conv",
        "conv1d",
        "conv_channels",
        "short_conv_dilation",
        "short_conv_state_len",
        "hidden_size",
        "hc_count",
        "layer_id",
        "_file_prefetcher",
        "_block_d",
        '"weight_scale"',
        "reduce",
        "allocate_output",
        "ple_window_cache",
    ),
    "srt/layers/vocab_parallel_embedding.py": (
        "embedding_dim",
        "shard_indices",
        "org_vocab_start_index",
        "org_vocab_end_index",
    ),
    "srt/mem_cache/ple_state_pool.py": (
        "NGRAM_PART",
        "SHORT_CONV_PART",
        "context",
    ),
    "srt/mem_cache/kv/router.py": ("part_view", "wait_for_part", "read"),
    "srt/mem_cache/kv/requests.py": (
        "RequestTable",
        "req_index_to_mamba_index_mapping",
        "state_sub_pool",
    ),
    "srt/mem_cache/memory_pool.py": ("HybridReqToTokenPool",),
    "srt/mem_cache/kv/alloc/id_space.py": ("write", "read_v2p"),
    "srt/mem_cache/kv/alloc/runner_ids.py": ("state_ids",),
    "srt/layers/attention/base_attn_backend.py": ("runner_ids",),
    "srt/model_executor/forward_context.py": (
        "get_attn_backend",
        "get_req_to_token_pool",
        "get_token_to_kv_pool",
    ),
    "srt/model_executor/runner/__init__.py": ("get_is_capture_mode",),
    "srt/model_executor/runner_backend_utils/breakable_cuda_graph/__init__.py": (
        "is_in_breakable_cuda_graph",
    ),
    "srt/model_executor/forward_mode.py": (
        "is_target_verify",
        "is_extend",
        "is_extend_without_speculative",
        "is_idle",
    ),
    "srt/model_executor/forward_batch_info.py": (
        "forward_mode",
        "input_ids",
        "spec_info",
        "spec_algorithm",
        "tbo_parent_token_range",
        "req_pool_indices",
        "extend_seq_lens",
        "extend_seq_lens_cpu",
        "out_cache_loc",
    ),
    "srt/speculative/eagle_info.py": ("draft_token_num", "topk"),
    "srt/layers/cp/utils.py": ("is_cp_active",),
    "srt/runtime_context.py": ("get_parallel",),
    "kernels/ops/elementwise/qwen4_gate.py": (
        "can_fuse_qwen4_gate_reduce",
        "fused_qwen4_gate_reduce",
        "can_fuse_qwen4_gate_value",
        "fused_qwen4_gate_value",
    ),
    "kernels/ops/mamba/qwen4_short_conv.py": ("can_fuse_qwen4_verify_conv",),
}

# The SGLang functions whose body this module mirrors, by the digest of their code
# (test/unit/test_contract.py recomputes them on the installed SGLang)
SGLANG_MIRRORS = {
    "srt/models/qwen4_exp.py:_prepare_ple_batch": "a477dd61a4b46b4c",
    "srt/models/qwen4_exp.py:Qwen4ExpLayerExtensionMixin._add_ple_embedding": "5e9187867c87d802",
    "srt/models/qwen4_exp.py:Qwen4ExpPLELayer.forward": "2b3ec67e3c1f2f25",
    "srt/models/qwen4_exp.py:Qwen4ExpPLELayer._short_conv": "118efb0441706a59",
    "srt/models/qwen4_exp.py:Qwen4ExpPLELayer.start_prefetch": "439fc52bdbea4660",
    "srt/models/qwen4_exp.py:Qwen4ExpNGramEmbedding.forward": "655a10aade52c518",
    "srt/models/qwen4_exp.py:Qwen4ExpNGramEmbedding._embed_ngram_ids": "6d557fdc52f6d0a9",
    "srt/models/qwen4_exp.py:Qwen4ExpPinnedHostEmbedding.forward": "20c0ac5384c5bf8d",
    "srt/models/qwen4_exp.py:Qwen4ExpPinnedHostEmbedding.gather": "d1b3188e18e1980e",
    "kernels/ops/mamba/qwen4_short_conv.py:fused_qwen4_verify_conv": "4b2ed8834a5a5d0d",
}

# Reads and mirrors that need an SGLang change newer than the unit job's pin: the
# contract test skips each, naming the change, only where the installed SGLang lacks it.
SGLANG_REQUIRES = {
    "srt/models/qwen4_exp.py:_use_qwen4_varlen_prefill": "sglang#43032",
    "srt/models/qwen4_exp.py:_prepare_ple_batch": (
        "sglang#49309",
        ("96091d8d8cb398c7", "22f0ccb94876ccb6"),
    ),
    "srt/models/qwen4_exp.py:Qwen4ExpPLELayer._short_conv": (
        "sglang#43032",
        ("463d39a6efea82fc",),
    ),
}


@dataclasses.dataclass(frozen=True)
class Options:
    """The source's PLE knobs as values. Defaults: its high-concurrency tuning;
    ``HIGH_CONCURRENCY_PLAIN_VERIFY`` and ``MID_CONCURRENCY`` are its other TP1 tunings."""

    # SGLANG_OPT_PLE_VERIFY_PREPARE_BATCHED (high / mid: 1)
    verify_prepare: bool = True
    # SGLANG_OPT_PLE_FUSED_VERIFY_CONV (high: 1, high-alt: 0, mid: 1)
    verify_conv: bool = True
    # SGLANG_OPT_PLE_PREFILL_FUSED_GATE (all 1)
    fused_gate: bool = True
    # SGLANG_OPT_PLE_FUSED_EXTEND_CONV (high: 1, mid: 0)
    extend_conv: bool = True
    # SGLANG_OPT_PLE_FUSED_EXTEND_CONV_CUDNN_SHAPES (high: 1, high-alt / mid: unset);
    # acts only with extend_conv
    extend_conv_cudnn: bool = True
    # SGLANG_PLE_PREFILL_CONV_CHUNK_SIZE (unset in all three: 0, off)
    conv_chunk_size: int = 0
    # SGLANG_PLE_GATHER_ROWS / SGLANG_PLE_GATHER_PROGRAMS (high / mid: 32 / 64, high-alt: unset)
    gather_rows: int = 32
    gather_programs: int = 64

    def __post_init__(self):
        rows = self.gather_rows
        if rows and (rows < 2 or rows > 128 or rows & (rows - 1)):
            raise ValueError(
                f"gather_rows is 0 or a power of two in 2..128, got {rows}"
            )
        if self.extend_conv_cudnn and not self.extend_conv:
            raise ValueError("extend_conv_cudnn acts only with extend_conv")
        if self.conv_chunk_size < 0 or self.gather_programs < 0:
            raise ValueError("conv_chunk_size and gather_programs are non-negative")


HIGH_CONCURRENCY = Options()
HIGH_CONCURRENCY_PLAIN_VERIFY = Options(
    verify_conv=False, extend_conv_cudnn=False, gather_rows=0, gather_programs=0
)
MID_CONCURRENCY = Options(extend_conv=False, extend_conv_cudnn=False)
DEFAULT = HIGH_CONCURRENCY


def build(model, options: Optional[Options] = DEFAULT) -> str:
    """Load the enabled kernels on this rank before capture; '' or why the PLE pieces
    stay stock. The Triton kernels compile at their first eager call (a capture warmup
    runs every graph width first); the gate's self-test runs here, never under capture
    (a failed self-test keeps SGLang's chain, it is not a build failure)."""
    if options is None:
        return "no options"
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return "the PLE kernels need a CUDA GPU"
    m = model.model
    if not m.has_ple:
        return "no PLE layer"
    failures = []

    def attempt(name, fn):
        try:
            fn()
        except Exception as e:  # a failed kernel keeps the stock piece
            failures.append(f"{name}: {type(e).__name__}: {e}")

    def verify_prepare():
        from sgl_nvidia_mega.qwen3_8.kernels.ops.embeddings import (  # noqa: F401
            qwen4_ple_verify_prepare,
        )

    def verify_conv():
        from sgl_nvidia_mega.qwen3_8.kernels.ops.mamba import (  # noqa: F401
            qwen4_verify_conv,
        )

    def fused_gate():
        from sgl_nvidia_mega.qwen3_8.kernels.ops.elementwise.qwen4_gate_post import (
            is_qwen4_gate_post_ready,
        )

        is_qwen4_gate_post_ready(torch.device("cuda", torch.cuda.current_device()))

    def extend_conv():
        from sgl_nvidia_mega.qwen3_8.kernels.ops.mamba import (  # noqa: F401
            qwen4_ple_fused,
        )

    def chunked():
        from sgl_nvidia_mega.qwen3_8.kernels.ops.mamba import (  # noqa: F401
            ple_short_conv,
        )

    def gather():
        from sgl_nvidia_mega.qwen3_8.kernels.ops.embeddings import (  # noqa: F401
            qwen4_ple_gather,
        )

    for name, enabled, fn in (
        ("verify_prepare", options.verify_prepare, verify_prepare),
        ("verify_conv", options.verify_conv, verify_conv),
        ("fused_gate", options.fused_gate, fused_gate),
        ("extend_conv", options.extend_conv, extend_conv),
        ("chunked", options.conv_chunk_size > 0, chunked),
        ("gather_rows", options.gather_rows > 0, gather),
    ):
        if enabled:
            attempt(name, fn)
    return "; ".join(failures)


# -- prepare and commit -------------------------------------------------------------


def prepare(st, input_ids, *, options: Optional[Options] = None):
    """``_prepare_ple_batch(input_ids, st.fb, ngram_size=..., ngram_eos_token_id=...)``
    for the model's PLE (the caller checks ``has_ple``)."""
    from sglang.srt.models.qwen4_exp import _prepare_ple_batch

    m, fb = st.model, st.fb
    batch = None
    if options is not None and options.verify_prepare:
        batch = prepare_verify(
            input_ids, fb, m.ple_ngram_size, m.ple_ngram_eos_token_id
        )
    if batch is not None:
        st.ran("ple.prepare", "verify_layout")
        return batch
    st.ran("ple.prepare", STOCK)
    return _prepare_ple_batch(
        input_ids=input_ids,
        forward_batch=fb,
        ngram_size=m.ple_ngram_size,
        ngram_eos_token_id=m.ple_ngram_eos_token_id,
    )


def prepare_verify(input_ids, fb, ngram_size, ngram_eos_token_id):
    """``_prepare_ple_batch``'s target-verify branch in one launch (the source's
    ``_try_prepare_ple_verify_batched``), or None (nothing launched, nothing written)
    outside its contract or wherever ``_prepare_ple_batch`` would refuse the batch; the
    caller then runs ``_prepare_ple_batch``."""
    from sglang.srt.layers.cp.utils import is_cp_active
    from sglang.srt.mem_cache.kv.requests import RequestTable
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
    from sglang.srt.mem_cache.ple_state_pool import NGRAM_PART
    from sglang.srt.model_executor.forward_context import (
        get_attn_backend,
        get_req_to_token_pool,
        get_token_to_kv_pool,
    )
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        is_in_breakable_cuda_graph,
    )
    from sglang.srt.models.qwen4_exp import (
        _get_ple_forward_mode,
        _get_processed_token_count,
        _PLEBatch,
    )
    from sglang.srt.runtime_context import get_parallel

    from sgl_nvidia_mega.qwen3_8.kernels.ops.embeddings import (
        qwen4_ple_verify_prepare as ple_prepare,
    )

    mode = _get_ple_forward_mode(fb)
    spec_algorithm = fb.spec_algorithm
    # _prepare_ple_batch's refusals: it raises for these, so its own call must run
    if (
        not mode.is_target_verify()
        or fb.tbo_parent_token_range is not None
        or (spec_algorithm is not None and spec_algorithm.is_ngram())
        or (fb.spec_info is not None and getattr(fb.spec_info, "topk", 1) != 1)
    ):
        return None
    if input_ids.dim() > 1:
        input_ids = input_ids.reshape(-1)
    physical_tokens = input_ids.shape[0]
    processed_tokens = _get_processed_token_count(fb, physical_tokens)
    tokens = input_ids[:processed_tokens]
    if not (
        mode.is_target_verify()
        and fb.forward_mode.is_target_verify()
        and not is_in_breakable_cuda_graph()
        and get_parallel().pp_size == 1
        and not is_cp_active(fb)
        and fb.spec_info is not None
        and ngram_size is not None
        and ngram_eos_token_id is not None
    ):
        return None
    width = int(fb.spec_info.draft_token_num)
    if width <= 0 or processed_tokens == 0 or processed_tokens % width:
        return None
    count = processed_tokens // width
    pool = get_req_to_token_pool()
    router = get_token_to_kv_pool().layer_router
    view = router.part_view(NGRAM_PART)
    if type(pool) not in (HybridReqToTokenPool, RequestTable) or view is None:
        return None
    # The state id space's write is a gather through read_v2p (or the identity).
    space = get_attn_backend().runner_ids.state_ids(pool.state_sub_pool)
    slot_v2p = None if space.write is None else space.read_v2p
    if space.write is not None and slot_v2p is None:
        return None
    requests = fb.req_pool_indices[:count]
    lengths = fb.extend_seq_lens
    if lengths is not None:
        lengths = lengths[:count]
    locations = fb.out_cache_loc
    if locations is not None:
        locations = locations[:processed_tokens]
    history = view.context
    if not ple_prepare.can_prepare(
        tokens,
        lengths,
        requests,
        locations,
        pool.req_index_to_mamba_index_mapping,
        slot_v2p,
        history,
        width,
        ngram_size,
    ):
        return None
    # _prepare_ple_batch's reset, then its wait before the history read
    pool.ple_window_cache = None
    router.wait_for_part(NGRAM_PART)
    layout = ple_prepare.prepare_verify_layout(
        tokens,
        lengths,
        requests,
        locations,
        pool.req_index_to_mamba_index_mapping,
        slot_v2p,
        history,
        width,
        ngram_eos_token_id,
    )
    batch = _PLEBatch(
        mode=mode,
        use_decode_fast_path=False,
        physical_tokens=physical_tokens,
        processed_tokens=processed_tokens,
        lengths=layout.lengths,
        row_width=width,
        req_indices=layout.req_indices,
        token_offsets=layout.token_offsets,
        valid_tokens=layout.valid_tokens,
        state_indices=layout.state_indices,
        # _prepare_ple_batch packs query starts for EXTEND / MIXED only
        query_start_loc=None,
        ngram_context=layout.ngram_context,
        ngram_eos_token_id=ngram_eos_token_id,
    )
    # compute_ngram_ids' unfold + index for this batch, already written by the kernel.
    pool.ple_window_cache = (batch, layout.contexts, None)
    return batch


def commit(batch, st) -> None:
    """``_commit_ple_batch``: no kernel here; SGLang's."""
    from sglang.srt.models.qwen4_exp import _commit_ple_batch

    _commit_ple_batch(batch=batch, forward_batch=st.fb)
    st.ran("ple.commit", STOCK)


# -- the host-table gather ----------------------------------------------------------


def _offloaded(ple) -> bool:
    from sglang.srt.models.qwen4_exp import Qwen4ExpPinnedHostEmbedding

    return isinstance(ple.ple_embedding.ngram_embedding, Qwen4ExpPinnedHostEmbedding)


def gather(embedding, input_ids, options, out=None):
    """``Qwen4ExpPinnedHostEmbedding.gather`` with the source's multi-row kernel
    (``options.gather_rows`` > 0)."""
    from sgl_nvidia_mega.qwen3_8.kernels.ops.embeddings.qwen4_ple_gather import (
        gather_rows,
    )

    expected_shape = (*input_ids.shape, embedding.embedding_dim)
    if out is None:
        output = embedding.allocate_output(expected_shape, input_ids.device)
    else:
        if tuple(out.shape) != expected_shape:
            raise ValueError(
                f"invalid PLE prefetch output shape: {tuple(out.shape)} != "
                f"{expected_shape}"
            )
        if out.dtype != torch.bfloat16 or out.device != input_ids.device:
            raise ValueError("PLE prefetch output must be bfloat16 on the id device")
        output = out

    flat_ids = input_ids.reshape(-1).long()
    if flat_ids.numel():
        if embedding._file_prefetcher is not None:
            embedding._file_prefetcher.enqueue(
                flat_ids,
                vocab_start=embedding.shard_indices.org_vocab_start_index,
                vocab_end=embedding.shard_indices.org_vocab_end_index,
            )
        gather_rows(
            weight_ptr=embedding.weight.data_ptr(),
            flat_ids=flat_ids,
            output=output,
            embedding_dim=embedding.embedding_dim,
            tp_vocab_start=embedding.shard_indices.org_vocab_start_index,
            tp_vocab_end=embedding.shard_indices.org_vocab_end_index,
            is_fp8=embedding.weight.dtype == torch.float8_e4m3fn,
            block_d=embedding._block_d,
            rows=options.gather_rows,
            programs=options.gather_programs,
        )
    return output


def prefetch(layer, batch, st, *, options: Optional[Options] = None) -> None:
    """``layer.ple.start_prefetch(batch=batch, forward_batch=st.fb)``: the next PLE
    layer's table rows gathered on the PLE's side stream while this layer runs."""
    ple, fb = layer.ple, st.fb
    if (
        options is None
        or not options.gather_rows
        or ple._prefetch_stream is None
        or not _offloaded(ple)
    ):
        st.ran("ple.prefetch", STOCK)
        ple.start_prefetch(batch=batch, forward_batch=fb)
        return
    if ple._prefetch_state is not None:
        raise RuntimeError("PLE prefetch state was not consumed before reuse")
    if batch is None:
        if not ple.ple_embedding.gather_dp_tokens:
            return
        physical_tokens = fb.input_ids.numel()
        ngram_ids = fb.input_ids.new_zeros(
            (physical_tokens, ple.ple_embedding.ngram_heads)
        )
    else:
        physical_tokens = batch.physical_tokens
        ngram_ids = ple.ple_embedding.compute_ngram_ids(batch)

    lookup_ids, semantic_tokens = ple.ple_embedding._prepare_embedding_lookup(
        ngram_ids, fb, physical_tokens
    )
    lookup_tokens = lookup_ids.shape[0]
    if lookup_tokens == 0:
        return
    prefetched = ple._get_prefetch_buffer(lookup_tokens, lookup_ids)
    output_view = prefetched.view(lookup_tokens, ple.ple_embedding.ngram_heads, -1)
    offloaded_embedding = ple.ple_embedding.ngram_embedding

    stream = ple._prefetch_stream
    stream.wait_stream(torch.get_device_module().current_stream())
    lookup_ids.record_stream(stream)
    with torch.get_device_module().stream(stream):
        gather(offloaded_embedding, lookup_ids, options, out=output_view)
    ple._prefetch_state = prefetched, semantic_tokens, physical_tokens
    st.ran("ple.prefetch", "gather_rows")


def _embed(ple, batch, fb, options):
    """``ple.ple_embedding(batch, fb)`` on an offloaded table, with ``gather``:
    ``Qwen4ExpNGramEmbedding.forward`` through ``_embed_ngram_ids`` and
    ``Qwen4ExpPinnedHostEmbedding.forward``."""
    ngram = ple.ple_embedding
    ngram_ids = ngram.compute_ngram_ids(batch)
    lookup_ids, semantic_tokens = ngram._prepare_embedding_lookup(
        ngram_ids, fb, batch.physical_tokens
    )
    table = ngram.ngram_embedding
    embeddings = table.reduce(gather(table, lookup_ids, options))
    embeddings = embeddings * table.weight_scale
    embeddings = ngram._finish_embedding_lookup(
        embeddings, semantic_tokens, fb, batch.physical_tokens
    )
    return embeddings.flatten(start_dim=-2)


# -- the PLE embedding add ----------------------------------------------------------


def served(options, layer, batch, fb, *, in_break=False) -> str:
    """The pieces of this layer's PLE add a kernel here serves ('' for none: the add
    runs ``_add_ple_embedding``). Host state only. ``in_break``: the add runs eagerly
    in a breakable prefill graph's break (``prefill_graph.ple_add``)."""
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        is_in_breakable_cuda_graph,
    )
    from sglang.srt.models.qwen4_exp import _shards_rows_over_cp

    if (
        options is None
        or batch is None
        or (
            not in_break
            and is_in_breakable_cuda_graph()
            and fb.forward_mode.is_extend()
        )
        or _shards_rows_over_cp(fb)
    ):
        return ""
    ple = layer.ple
    pieces = []
    if options.gather_rows and ple._prefetch_state is None and _offloaded(ple):
        pieces.append("gather")
    mode = batch.mode
    if not batch.use_decode_fast_path and options.fused_gate:
        # ``forward`` keeps SGLang's fused verify reduce where that applies
        pieces.append("gate")
    if mode.is_target_verify():
        if options.verify_conv and ple.ple_embedding.enable_ple_fusion:
            pieces.append("verify_conv")
    elif mode.name in ("EXTEND", "MIXED") and not batch.use_decode_fast_path:
        if options.extend_conv or options.conv_chunk_size > 0:
            pieces.append("conv")
    return "+".join(pieces)


def add(layer, residual, batch, st, *, options: Optional[Options] = None):
    """``layer._add_ple_embedding(residual)`` with the layer's PLE batch ``batch``:
    ``residual`` plus the PLE embedding of the step's tokens."""
    fb = st.fb
    if not served(options, layer, batch, fb):
        st.ran("ple.add", STOCK)
        layer._ple_forward_batch, layer._ple_batch = fb, batch
        try:
            return layer._add_ple_embedding(residual)
        finally:
            layer._ple_forward_batch = layer._ple_batch = None
    return residual + forward(options, layer.ple, residual, fb, batch, st)


def forward(options, ple, hidden_states, fb, batch, st):
    """``Qwen4ExpPLELayer.forward`` with the source's gate, conv and gather."""
    from sglang.kernels.ops.elementwise.qwen4_gate import (
        can_fuse_qwen4_gate_reduce,
        fused_qwen4_gate_reduce,
    )
    from sglang.srt.mem_cache.ple_state_pool import SHORT_CONV_PART
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
    from sglang.srt.models.qwen4_exp import _pad_token_rows

    hidden_states = hidden_states[: batch.processed_tokens]
    if ple._prefetch_state is not None:
        embeddings = ple._consume_prefetched_embeddings(fb)
    elif options.gather_rows and _offloaded(ple):
        embeddings = _embed(ple, batch, fb, options)
        st.ran("ple.gather", "gather_rows")
    else:
        embeddings = ple.ple_embedding(batch, fb)
    key, _ = ple.key_proj(embeddings)
    value, _ = ple.value_proj(embeddings)
    token_count = hidden_states.shape[0]
    hidden_size = ple.hidden_size
    hc_count = ple.hc_count
    if hidden_states.shape[-1] != hc_count * hidden_size:
        raise RuntimeError(
            "PLE hidden size does not match its hyper-connection layout: "
            f"expected {hc_count * hidden_size}, got {hidden_states.shape[-1]}"
        )
    key = key.reshape(token_count, hc_count, hidden_size)
    query = hidden_states.reshape(token_count, hc_count, hidden_size)
    key_normed = ple._apply_ple_norm(ple.norm_key, key)
    query_normed = ple._apply_ple_norm(ple.norm_query, query)

    fused_gate_reduce = (
        ple.ple_embedding.enable_ple_fusion
        and batch.mode.is_target_verify()
        and can_fuse_qwen4_gate_reduce(key_normed, query_normed, value)
    )
    fused_gate_post = False
    if fused_gate_reduce:
        gated_value = fused_qwen4_gate_reduce(key_normed, query_normed, value)
    else:
        gate = (key_normed * query_normed).sum(dim=-1, keepdim=True)
        if not batch.use_decode_fast_path and options.fused_gate:
            from sgl_nvidia_mega.qwen3_8.kernels.ops.elementwise.qwen4_gate_post import (
                can_fuse_qwen4_gate_post,
                is_qwen4_gate_post_ready,
            )

            # The reduction above stays torch's; the rest of the chain is one
            # kernel, bitwise to it (self-tested over every BF16 gate input).
            fused_gate_post = can_fuse_qwen4_gate_post(
                gate, value
            ) and is_qwen4_gate_post_ready(gate.device)
    if fused_gate_post:
        from sgl_nvidia_mega.qwen3_8.kernels.ops.elementwise.qwen4_gate_post import (
            fused_qwen4_gate_post,
        )

        gated_value = fused_qwen4_gate_post(gate, value)
        st.ran("ple.gate", "gate_post")
    else:
        st.ran("ple.gate", STOCK)
    if not fused_gate_reduce and not fused_gate_post:
        gate = gate / math.sqrt(hidden_size)
        fused_gate_value = False
        if batch.use_decode_fast_path:
            from sglang.kernels.ops.elementwise.qwen4_gate import (
                can_fuse_qwen4_gate_value,
                fused_qwen4_gate_value,
            )

            fused_gate_value = can_fuse_qwen4_gate_value(gate, value)
        if fused_gate_value:
            gated_value = fused_qwen4_gate_value(gate, value)
        else:
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = torch.sigmoid(gate)
            gated_value = gate * value.unsqueeze(-2)
    gated_value_normed = ple._apply_ple_norm(ple.norm_conv, gated_value)
    gated_value = gated_value.flatten(-2)
    gated_value_normed = gated_value_normed.flatten(-2)
    if ple.ple_embedding.enable_ple_fusion and batch.mode.is_target_verify():
        from sglang.kernels.ops.mamba.qwen4_short_conv import (
            can_fuse_qwen4_verify_conv,
        )

        from sgl_nvidia_mega.qwen3_8.kernels.ops.mamba.qwen4_verify_conv import (
            fused_qwen4_verify_conv,
        )

        part = get_token_to_kv_pool().layer_router.read(ple.layer_id, SHORT_CONV_PART)
        conv_args = (
            gated_value_normed,
            gated_value,
            ple.conv1d.weight,
            part.view.layer_cache(part.view_layer_id),
            batch.state_indices,
            batch.valid_tokens,
            batch.row_width,
            ple.short_conv_dilation,
            part.view.layer_intermediate_cache(part.view_layer_id),
        )
        if can_fuse_qwen4_verify_conv(*conv_args):
            output = fused_qwen4_verify_conv(*conv_args, one_kernel=options.verify_conv)
            st.ran("ple.conv", "verify_conv" if options.verify_conv else STOCK)
            return _pad_token_rows(output, batch.physical_tokens)
    conv_output = short_conv(options, ple, gated_value_normed, fb, batch, st)
    output = gated_value + conv_output
    if not batch.use_decode_fast_path:
        output = torch.where(
            batch.valid_tokens.unsqueeze(-1),
            output,
            torch.zeros_like(output),
        )
    return _pad_token_rows(output, batch.physical_tokens)


def short_conv(options, ple, x, fb, batch, st):
    """``Qwen4ExpPLELayer._short_conv``: SGLang's decode kernel and target-verify path
    first, then the source's extend conv over the token rows, then SGLang's varlen
    kernel, then the source's chunked conv, else SGLang's padded conv."""
    from sglang.srt.mem_cache.ple_state_pool import SHORT_CONV_PART
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
    from sglang.srt.model_executor.runner import get_is_capture_mode
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        is_in_breakable_cuda_graph,
    )
    from sglang.srt.models.qwen4_exp import (
        _ple_track_targets,
        _use_qwen4_varlen_prefill,
    )

    if x.shape[0] == 0:
        return x
    part = get_token_to_kv_pool().layer_router.read(ple.layer_id, SHORT_CONV_PART)
    conv_state = part.view.layer_cache(part.view_layer_id)
    if batch.use_decode_fast_path or batch.mode.is_target_verify():
        st.ran("ple.conv", STOCK)
        return ple._short_conv(x, fb, batch)

    # The source's extend conv runs ahead of SGLang's varlen kernel (faster at prefill sizes)
    if options.extend_conv:
        from sgl_nvidia_mega.qwen3_8.kernels.ops.mamba import qwen4_ple_fused as fused

        weight = ple.conv1d.weight.to(dtype=x.dtype)
        if fused.conv_engages(
            x,
            batch,
            fb.extend_seq_lens_cpu,
            weight=weight,
            conv_state=conv_state,
            channels=ple.conv_channels,
            state_len=ple.short_conv_state_len,
            cudnn_shapes=options.extend_conv_cudnn,
        ):
            # Stock's conv bytes from the token rows, without the padded [B, C, S + W] input.
            state = conv_state.index_select(0, batch.state_indices).to(dtype=x.dtype)
            track = _ple_track_targets(fb, batch)
            offsets = (batch.lengths,) if track is None else (batch.lengths, track[1])
            conv_tokens, states = fused.fused_short_conv_extend(
                x,
                state,
                weight,
                batch,
                dilation=ple.short_conv_dilation,
                state_offsets=offsets,
            )
            del state
            conv_state[batch.state_indices] = states[0].to(dtype=conv_state.dtype)
            if track is not None:
                conv_state[track[0]] = states[1].to(dtype=conv_state.dtype)
            st.ran("ple.conv", "extend_conv")
            return F.silu(conv_tokens)

    if _use_qwen4_varlen_prefill(
        batch.mode,
        batch.query_start_loc,
        x,
        ple.conv1d.weight,
        conv_state,
        batch.state_indices,
        batch.req_indices,
        batch.token_offsets,
        ple.short_conv_dilation,
    ):
        st.ran("ple.conv", STOCK)
        return ple._short_conv(x, fb, batch)

    conv_chunk_size = options.conv_chunk_size
    conv_lengths = fb.extend_seq_lens_cpu if conv_chunk_size > 0 else None
    chunked = (
        conv_chunk_size > 0
        and batch.row_width > conv_chunk_size
        and x.dtype == torch.bfloat16
        and batch.mode.is_extend_without_speculative()
        # A fully captured graph must not bake in host request boundaries.
        and (not get_is_capture_mode() or is_in_breakable_cuda_graph())
        and conv_lengths is not None
        and len(conv_lengths) == batch.lengths.shape[0]
        and all(0 <= length <= batch.row_width for length in conv_lengths)
        and sum(conv_lengths) == x.shape[0]
    )
    if not chunked:
        st.ran("ple.conv", STOCK)
        return ple._short_conv(x, fb, batch)
    from sgl_nvidia_mega.qwen3_8.kernels.ops.mamba.ple_short_conv import (
        chunked_packed_short_conv,
    )

    # _short_conv's padded path (not a target verify) with the chunked conv
    state = conv_state.index_select(0, batch.state_indices).to(dtype=x.dtype)
    padded_seq = x.new_zeros(
        (batch.lengths.shape[0], batch.row_width, ple.conv_channels)
    )
    padded_seq[batch.req_indices, batch.token_offsets] = x
    conv_input = torch.cat([state, padded_seq.transpose(1, 2)], dim=-1)
    # conv_input owns the history and padding needed by state commits.
    del padded_seq
    conv_output = chunked_packed_short_conv(
        conv_input,
        ple.conv1d.weight.to(dtype=x.dtype),
        conv_lengths,
        dilation=ple.short_conv_dilation,
        chunk_size=conv_chunk_size,
    )
    state_cols = torch.arange(
        ple.short_conv_state_len, device=x.device, dtype=torch.long
    )

    def _gather_at(offsets: torch.Tensor) -> torch.Tensor:
        return conv_input.gather(
            2,
            (offsets.unsqueeze(1) + state_cols.unsqueeze(0))
            .unsqueeze(1)
            .expand(-1, ple.conv_channels, -1),
        )

    next_state = _gather_at(batch.lengths)
    conv_state[batch.state_indices] = next_state.to(dtype=conv_state.dtype)

    # Same boundary mamba uses, into the slot the radix tree reads.
    track = _ple_track_targets(fb, batch)
    if track is not None:
        track_indices, track_offsets = track
        conv_state[track_indices] = _gather_at(track_offsets).to(dtype=conv_state.dtype)
    st.ran("ple.conv", "chunked")
    return F.silu(conv_output)
