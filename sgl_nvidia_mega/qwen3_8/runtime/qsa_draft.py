# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py (QwenSparseAttnBackend.
# _try_fused_kv_attention, _forward_trtllm_sparse, lookup_mtp_sparse_indices,
# capture_mtp_sparse_indices and _capture_mtp_sparse_indices_from_extend_lens) and
# python/sglang/srt/models/qwen4_exp.py (Qwen4ExpAttentionDecoderLayer.
# _compute_qsa_topk_indices_eager).
"""Slot ``qsa_draft``: the MTP draft's QSA glue between its draft extend and its draft
decode steps, inside the draft's graphs.

* A draft decode step whose backend reuses the draft extend's selection
  (``should_reuse_mtp_sparse_indices``) runs its sparse attention as the backend's
  fused K/V path does, with the K/V pack looking each request's captured selection
  up itself (``qsa/fused_kv_lookup.py``) instead of SGLang's torch lookup before it,
  then SGLang's TRT-LLM sparse decode, at any batch (one row a request). The source's
  gate: TP1, hidden 2560, 4 HC streams, a 3-step topk-1 chain.
* A draft extend stores each request's seed (its last accepted row's selection and
  length) in one launch of ``qsa/mtp_capture.py`` instead of SGLang's torch capture,
  for 1-256 requests.

The slot runs at TP1, where the source's recipes turn both on. Each declines,
launching nothing, to SGLang's own path wherever the backend would not take its fused
K/V path or the kernel's input checks fail. The QKV / gate projection, the draft
extend's indexer, the gate and o_proj stay SGLang's (or another slot's).

Numerics: bit-identical; the pack reads the K/V the lookup selects and the capture
writes the rows and lengths SGLang's torch expressions write.

Reads: the backend's MTP selection state and step metadata, the draft's K/V pool and
``req_to_token``. Writes: the new K/V at ``out_cache_loc``, the backend's packed
scratch and valid counts (as its own path does), the seed rows and lengths.
"""

from __future__ import annotations

import torch

STOCK = "stock"
HEAD_DIM = 256
WORKSPACE_BYTES = 128 * 1024 * 1024  # the backend's TRT-LLM workspace size

SGLANG_READS = {
    "srt/layers/attention/qwen_sparse_attn_backend.py": (
        "_TRTLLM_SPARSE_PAGE_SIZE",
        "_resolve_trtllm_sparse_decode",
        "_mtp_shared_sparse_indices",
        "layer_slots",
        "tail_width",
        "trash_row",
        "indices",
        "captured_len",
        "should_reuse_mtp_sparse_indices",
        "should_capture_mtp_sparse_indices",
        "capture_mtp_sparse_indices",
        "_capture_extend_seq_lens",
        "_resolve_metadata",
        "_fused_kv_pool_eligible",
        "_uses_block_indices",
        "_get_trtllm_sparse_tables",
        "_get_fa2_scratch",
        "_cuda_graph_max_tokens",
        "_trtllm_workspace",
        "fa2_valid_counts",
        "row_req_pool_indices",
        "sequence_lengths",
        "is_cuda_graph",
        "indexer_metadata",
        "req_to_token_pool",
        "token_to_kv_pool",
    ),
    "srt/layers/attention/qsa/metadata.py": (
        "decode_logical_positions",
        "sequence_lengths",
        "is_cuda_graph",
        "get_token_to_batch_idx",
    ),
    "srt/layers/attention/qsa/glue.py": (
        "get_qsa_indexer_metadata",
        "resolve_qsa_sparse_backend",
    ),
    "srt/models/qwen4_exp.py": (
        "Qwen4ExpAttentionDecoderLayer",
        "_compute_qsa_topk_indices",
        "is_qsa",
        "indexer",
        "layer_id",
    ),
    "srt/layers/radix_attention.py": (
        "layer_id",
        "scaling",
        "tp_q_head_num",
        "head_dim",
    ),
    "srt/mem_cache/kv/router.py": ("read", "key", "value"),
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "srt/model_executor/forward_batch_info.py": (
        "batch_size",
        "forward_mode",
        "out_cache_loc",
        "req_pool_indices",
        "spec_info",
    ),
    "srt/speculative/eagle_info.py": ("num_accept_tokens", "num_front_tokens"),
    "srt/model_executor/runner_backend_utils/breakable_cuda_graph/__init__.py": (
        "is_in_breakable_cuda_graph",
    ),
    "srt/utils/device.py|srt/utils/common.py": ("is_sm100_supported",),
    "srt/runtime_context.py": ("get_parallel", "get_spec"),
    "srt/arg_groups/fields/spec.py": (
        "speculative_num_steps",
        "speculative_eagle_topk",
    ),
    "srt/models/qwen4_exp_mtp.py": ("hidden_size", "hc_count"),
}


def build(model) -> str:
    """'' when the draft glue can run on this rank, else why not."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return "the draft glue kernels need an SM10x GPU"
    from sglang.srt.models.qwen4_exp import Qwen4ExpAttentionDecoderLayer
    from sglang.srt.runtime_context import get_parallel

    tp = get_parallel().tp_size
    if tp != 1:
        return f"TP{tp}: the draft glue runs at TP1"
    m = model.model
    if not any(
        isinstance(m.layers[i], Qwen4ExpAttentionDecoderLayer) and m.layers[i].is_qsa
        for i in range(m.start_layer, m.end_layer)
    ):
        return "no QSA layer"
    try:
        from sgl_nvidia_mega.qwen3_8.qsa import (  # noqa: F401
            fused_kv_lookup,
            mtp_capture,
        )
    except Exception as e:
        return f"the draft glue kernels do not import: {type(e).__name__}: {e}"
    return ""


def lookup_reason(model) -> str:
    """'' when the draft decode's lookup pack runs for this draft: the source's gate
    of its native lookup beyond TP1 and the pool."""
    from sglang.srt.runtime_context import get_spec

    spec = get_spec()
    if (model.hidden_size, model.hc_count) != (2560, 4):
        return f"hidden {model.hidden_size} x {model.hc_count} HC streams, not 2560 x 4"
    steps, topk = spec.speculative_num_steps, spec.speculative_eagle_topk
    if (steps, topk) != (3, 1):
        return f"a {steps}-step topk-{topk} draft, not a 3-step topk-1 chain"
    return ""


class DraftGlue:
    """Slot qsa_draft's launches, with the TRT-LLM decode scratch its draft decode
    steps share where the backend has none yet (the steps run in order).
    ``lookup``: the draft decode's lookup pack runs (``lookup_reason``)."""

    def __init__(self, lookup: bool = True):
        from sgl_nvidia_mega.qwen3_8.qsa import fused_kv_lookup, mtp_capture

        self.lookup_ready = lookup
        self.pack = fused_kv_lookup
        self.seed = mtp_capture
        self._workspace = None
        self._counters: dict = {}

    def decode(self, layer, backend, q, k, v, fb):
        """``layer.attn(q, k, v, fb, topk_indices=<the looked-up selection>)`` of a
        draft decode step, [rows, heads * 256]; None (nothing launched) where SGLang's
        backend would not run its fused K/V path."""
        from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
            _resolve_trtllm_sparse_decode,
        )

        state = backend._mtp_shared_sparse_indices
        if (
            not self.lookup_ready
            or state is None
            or not backend.should_reuse_mtp_sparse_indices(fb)
        ):
            return None
        attn = layer.attn
        trtllm_decode = _resolve_trtllm_sparse_decode()
        md = backend._resolve_metadata(fb)
        positions = (
            None
            if md.indexer_metadata is None
            else md.indexer_metadata.decode_logical_positions
        )
        loc = fb.out_cache_loc
        rows = q.shape[0]
        if (
            not backend._fused_kv_pool_eligible
            or trtllm_decode is None
            or k is None
            or v is None
            or positions is None
            or q.dtype != torch.bfloat16
            or rows == 0
            or rows != fb.req_pool_indices.numel()
            or positions.numel() != rows
            or not isinstance(loc, torch.Tensor)
            or loc.ndim != 1
            or loc.numel() != rows
            or not loc.is_contiguous()
            or k.shape[0] != rows
            or v.shape[0] != rows
            or md.sequence_lengths.numel() != rows
            or (md.is_cuda_graph and md.fa2_valid_counts is None)
        ):
            return None
        # keyed by the decoder layer's id, as SGLang's lookup is
        slot = state.layer_slots[int(layer.layer_id)]
        lookup = self.pack.MtpLookup(
            indices=state.indices[slot],
            captured_len=state.captured_len[slot],
            requests=fb.req_pool_indices,
            positions=positions,
            tail_width=int(state.tail_width),
        )
        if backend._uses_block_indices(lookup):
            return None  # the backend would expand block indices: not a lookup row
        kv = backend.token_to_kv_pool.layer_router.read(attn.layer_id)
        kc, vc = kv.key(), kv.value()
        if (
            kc.ndim != 3
            or vc.shape != kc.shape
            or not kc.is_contiguous()
            or not vc.is_contiguous()
            or kc.shape[-1] != HEAD_DIM
            or vc.dtype != kc.dtype
        ):
            return None
        k = k.reshape(rows, kc.shape[1], HEAD_DIM).to(kc.dtype)
        v = v.reshape(rows, kc.shape[1], HEAD_DIM).to(vc.dtype)
        if k.stride()[1:] != (HEAD_DIM, 1) or v.stride()[1:] != (HEAD_DIM, 1):
            return None
        q = q.reshape(rows, attn.tp_q_head_num, attn.head_dim)
        return self._sparse_decode(
            backend, attn, q, kc, vc, k, v, loc, md, lookup, fb, trtllm_decode
        )

    def _sparse_decode(
        self, backend, attn, q, kc, vc, k, v, loc, md, lookup, fb, trtllm_decode
    ):
        """``_forward_trtllm_sparse`` with ``new_kv`` at width 1 on token indices: the
        lookup pack into page-aligned rows, then TRT-LLM's paged decode over them."""
        from flashinfer.utils import get_trtllm_gen_multi_ctas_kv_counter_bytes
        from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
            _TRTLLM_SPARSE_PAGE_SIZE,
        )
        from sglang.srt.utils import is_sm100_supported

        rows, topk = lookup.shape
        page = _TRTLLM_SPARSE_PAGE_SIZE
        pages_per_row = -(-topk // page)
        stride = pages_per_row * page
        device = q.device
        if md.is_cuda_graph:
            counts = md.fa2_valid_counts
        else:
            counts = torch.empty(rows, dtype=torch.int32, device=device)
        _, block_tables = backend._get_trtllm_sparse_tables(
            rows, pages_per_row, page, device
        )
        capacity = backend._cuda_graph_max_tokens if md.is_cuda_graph else rows
        packed_dtype = (
            kc.dtype
            if kc.dtype == torch.float8_e4m3fn and is_sm100_supported()
            else q.dtype
        )
        heads, dim = kc.shape[1], kc.shape[2]
        packed_k, packed_v = backend._get_fa2_scratch(
            max(capacity, rows) * stride, heads, dim, packed_dtype, kc.device
        )
        packed_k, packed_v = packed_k[: rows * stride], packed_v[: rows * stride]
        row_requests = md.row_req_pool_indices
        if row_requests is None:
            row_requests = fb.req_pool_indices
        self.pack.fused_kv_lookup(
            kc,
            vc,
            k,
            v,
            loc,
            backend.req_to_token_pool.req_to_token,
            row_requests,
            lookup,
            md.sequence_lengths,
            counts,
            packed_k,
            packed_v,
        )
        paged = lambda t: t.view(-1, page, heads, dim).permute(0, 2, 1, 3)
        workspace = backend._trtllm_workspace
        if workspace is None:
            if self._workspace is None:
                self._workspace = torch.zeros(
                    WORKSPACE_BYTES, dtype=torch.uint8, device=device
                )
            workspace = self._workspace
        key = (rows, q.shape[1], device)
        if key not in self._counters:
            sms = torch.cuda.get_device_properties(device).multi_processor_count
            size = get_trtllm_gen_multi_ctas_kv_counter_bytes(rows, q.shape[1], sms)
            self._counters[key] = torch.zeros(size, dtype=torch.uint8, device=device)
        output = trtllm_decode(
            query=q.contiguous(),
            kv_cache=(paged(packed_k), paged(packed_v)),
            workspace_buffer=workspace,
            block_tables=block_tables,
            seq_lens=counts,
            max_seq_len=stride,
            bmm1_scale=attn.scaling,
            bmm2_scale=1.0,
            multi_ctas_kv_counter_buffer=self._counters[key],
        )
        return output.reshape(rows, -1)

    def capture(self, backend, topk, fb, md, layer_id) -> bool:
        """``backend.capture_mtp_sparse_indices(topk, fb, layer_id, metadata=md)`` of a
        draft extend as one launch; False (nothing launched) where the kernel does not
        take it, and SGLang's capture runs."""
        if not (md.is_cuda_graph or fb.forward_mode.is_draft_extend_v2()):
            return False
        state = backend._mtp_shared_sparse_indices
        bs = int(fb.batch_size)
        if bs == 0 or fb.req_pool_indices.numel() < bs:
            return False
        spec = fb.spec_info
        # only a draft extend's input carries accept counts; other rows end at the seed
        accept = getattr(spec, "num_accept_tokens", None)
        if accept is None or accept.numel() < bs:
            accept, front = None, 0
        else:
            accept, front = accept[:bs], int(spec.num_front_tokens)
        extend = backend._capture_extend_seq_lens(fb)[:bs]
        req = fb.req_pool_indices[:bs]
        slot = state.layer_slots[int(layer_id)]
        seed, captured = state.indices[slot], state.captured_len[slot]
        lengths, rows = md.sequence_lengths, md.get_token_to_batch_idx()
        trash = state.trash_row
        if self.seed.reason(
            topk, extend, accept, req, lengths, rows, seed, captured, trash
        ):
            return False
        self.seed.capture(
            topk, extend, accept, front, req, lengths, rows, seed, captured, trash
        )
        return True


def capture(executor, backend, topk, fb, md, layer_id, st) -> None:
    """The draft extend's seed capture: slot qsa_draft's launch, else SGLang's."""
    glue = executor.qsa_draft
    if glue is not None and glue.capture(backend, topk, fb, md, layer_id):
        st.ran("qsa.mtp_capture", "batched")
        return
    backend.capture_mtp_sparse_indices(topk, fb, layer_id, metadata=md)
    st.ran("qsa.mtp_capture", STOCK)


def indices(executor, layer, hidden_states, positions, st) -> torch.Tensor:
    """``layer._compute_qsa_topk_indices`` of a draft extend: SGLang's indexer, then the
    seed capture (``capture``)."""
    from sglang.srt.layers.attention.qsa.glue import (
        get_qsa_indexer_metadata,
        resolve_qsa_sparse_backend,
    )
    from sglang.srt.model_executor.forward_context import get_attn_backend
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        is_in_breakable_cuda_graph,
    )

    fb = st.fb
    if not fb.forward_mode.is_draft_extend_v2() or is_in_breakable_cuda_graph():
        return layer._compute_qsa_topk_indices(hidden_states, positions, fb)
    backend = get_attn_backend()
    sparse_backend = resolve_qsa_sparse_backend(backend)
    md = get_qsa_indexer_metadata(backend, layer.layer_id, fb)
    topk = layer.indexer(hidden_states, positions, fb, md)
    should_capture = getattr(sparse_backend, "should_capture_mtp_sparse_indices", None)
    if should_capture is not None and should_capture(fb):
        capture(executor, sparse_backend, topk, fb, md, layer.layer_id, st)
    return topk
