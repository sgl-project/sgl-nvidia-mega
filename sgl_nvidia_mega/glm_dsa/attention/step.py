# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team
"""Stock verify attention steps: StepContext, PrologueOut, IndexQ, AttentionDecode."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Hashable, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

class StepContext:
    """Per-forward state shared by the step groups of one verify forward."""

    forward_batch: Any
    positions: torch.Tensor
    zero_allocator: Any
    # Set by a routing hop: (expert ids [rows, 8] int32, weights [rows, 8] fp32);
    # read and cleared by the same layer's MoE step.
    routing: Any = None
    # Per layer, cleared by AttentionDecode.forward: the indexer key / head-gate
    # projections and the FP8 MLA query when a fused step produced them early.
    index_kw: Any = None
    q_fp8: Any = None
    # A deferred input norm (DNORM): (layer id, block sums [rows, 8] fp32); read
    # and cleared by that layer's attention.
    dnorm: Any = None


@dataclass
class PrologueOut:
    """What ``prologue`` hands on; None where a fused override consumed it.
    ``q_lora`` [rows, q_lora_rank] is the indexer input; ``q_nope`` / ``q_pe``
    are per head and ``k_nope`` [rows, 1, kv_lora_rank] / ``k_pe`` before RoPE."""

    q_lora: Optional[torch.Tensor]
    q_nope: Optional[torch.Tensor]
    q_pe: Optional[torch.Tensor]
    k_nope: Optional[torch.Tensor]
    k_pe: Optional[torch.Tensor]


@dataclass
class IndexQ:
    """The indexer query of a layer that runs its own indexer."""

    q_fp8: torch.Tensor  # [rows, index_heads, index_head_dim], FP8
    weights: torch.Tensor  # [rows, index_heads, 1], fp32 head gates


class AttentionDecode:
    """Stock verify attention of one ``DeepseekV2AttentionMLA`` layer, in steps:
    ``qkv_a`` -> ``prologue`` -> (own indexer) ``index_k`` -> ``index_q`` ->
    ``indexer`` -> ``q_absorb`` -> ``rope_cache`` -> ``sparse_mla`` -> ``uv_proj``.

    The bodies copy the verify slices of ``forward_absorb_prepare`` /
    ``forward_absorb_core`` (forward_mla.py), the DSA indexer's ``forward_cuda``
    and the DSA backend's ``_forward_trtllm`` at SGLang main 44b44774 for the one
    configuration served (bf16 W_UK / W_UV, fused indexer projections, FP8 KV
    cache on the TRT-LLM sparse kernel, fused top-k page table, full-graph
    verify); eligibility refuses every other one, and the replica test checks the
    copies bitwise against stock. A kernel covering several steps overrides the
    first and passes the others through, handing state on ``StepContext``.
    """

    def forward(self, layer, x, prev_topk_indices, step: StepContext):
        """The ``o_proj`` input ``[rows, heads * v_head_dim]`` and the top-k
        indices handed to the next layer (None unless it reuses them)."""
        from sglang.srt.layers.layer_boundary import get_attn_tp_context
        from sglang.srt.layers.layer_boundary.adapters.attention import (
            AttentionInputs,
        )

        attn = layer.self_attn
        fb = step.forward_batch
        self.check_mode(attn, fb.forward_mode)
        ctx = get_attn_tp_context()
        # The attention reads its input through the attention-TP context.
        ctx.set_attn_inputs(AttentionInputs(x, fb, attn.prepare_qkv_latent))
        step.index_kw = step.q_fp8 = None
        try:
            with attn.maybe_use_decode_attn_tp(fb):
                qkv_latent = self.qkv_a(attn, x, step)
                pro = self.prologue(attn, qkv_latent, step)
                index_q = None
                if attn.should_run_indexer(prev_topk_indices):
                    if self._stock_index_steps():
                        index_q = self._index_k_and_q(attn, x, pro.q_lora, step)
                    else:
                        self.index_k(attn, x, step.positions, step)
                        index_q = self.index_q(attn, x, pro.q_lora, step)
                topk_indices = self.indexer(attn, index_q, prev_topk_indices, step)
                q_nope_out = self.q_absorb(attn, pro.q_nope, step)
                q_fp8 = self.rope_cache(
                    attn, q_nope_out, pro.q_pe, pro.k_nope, pro.k_pe, step
                )
                attn_output = self.sparse_mla(attn, q_fp8, topk_indices, step)
                out = self.uv_proj(attn, attn_output)
        finally:
            ctx.clear_attn_inputs()
            step.index_kw = step.q_fp8 = None
        return out, (topk_indices if attn.next_skip_topk else None)

    def check_mode(self, attn, mode) -> None:
        # The stock bodies copy the branches of these modes only (the draft's
        # decode passes: draft_attention.DraftAttention).
        if not (mode.is_target_verify() or mode.is_draft_extend_v2()):
            raise RuntimeError(
                f"AttentionDecode serves target-verify and draft-extend-v2 forwards only (got {mode})"
            )

    def qkv_a(self, attn, x, step: StepContext) -> torch.Tensor:
        """``[rows, q_lora_rank + kv_lora_rank + qk_rope_head_dim]``; an override
        may also leave the indexer key / head-gate projections in
        ``step.index_kw``."""
        from sglang.srt.layers.layer_boundary import get_attn_tp_context

        # forward_absorb_prepare (q_lora_rank set).
        return get_attn_tp_context().fetch_qkv_latent()

    def prologue(
        self, attn, qkv_latent: torch.Tensor, step: StepContext
    ) -> PrologueOut:
        """The q_a / kv_a norms, ``q_b_proj`` and the q / key-RoPE split."""
        from sglang.srt.model_executor.runner import get_is_capture_mode

        # forward_absorb_prepare (main 44b44774), verify: the kv_a norm runs on
        # the side stream during capture.
        q, latent_cache = qkv_latent.split(
            [attn.q_lora_rank, attn.kv_lora_rank + attn.qk_rope_head_dim], dim=-1
        )
        k_nope = latent_cache[..., : attn.kv_lora_rank]
        if attn.alt_stream is not None and get_is_capture_mode():
            current_stream = torch.cuda.current_stream()
            attn.alt_stream.wait_stream(current_stream)
            q = attn.q_a_layernorm(q)
            with torch.cuda.stream(attn.alt_stream):
                k_nope = attn.kv_a_layernorm(k_nope)
            current_stream.wait_stream(attn.alt_stream)
        else:
            q = attn.q_a_layernorm(q)
            k_nope = attn.kv_a_layernorm(k_nope)
        q_lora = q
        k_nope = k_nope.unsqueeze(1)
        q = attn.q_b_proj_forward(q)
        q_nope, q_pe, k_pe = attn._split_q_nope_pe(q, latent_cache)
        return PrologueOut(q_lora, q_nope, q_pe, k_nope, k_pe)

    def index_k(self, attn, x, positions: torch.Tensor, step: StepContext) -> None:
        """The indexer key's norm, RoPE, FP8 quantization and index-K cache rows."""
        # _fused_q_prepare_and_store (main 44b44774), single-stream key half.
        key, _ = self._index_kw(attn.indexer, x, step)
        _store_index_k(attn, key, positions, step)

    def index_q(self, attn, x, q_lora: torch.Tensor, step: StepContext) -> IndexQ:
        """The indexer query (``wq_b``, RoPE, FP8) and its head gates."""
        # _fused_q_prepare_and_store (main 44b44774), single-stream query half.
        indexer = attn.indexer
        _, weights_raw = self._index_kw(indexer, x, step)
        q = indexer.wq_b(q_lora)[0].view(-1, indexer.n_heads, indexer.head_dim)
        return IndexQ(*_index_q_quant(indexer, q, weights_raw, step.positions))

    def _stock_index_steps(self) -> bool:
        cls = type(self)
        return (
            cls.index_k is AttentionDecode.index_k
            and cls.index_q is AttentionDecode.index_q
        )

    def _index_k_and_q(
        self, attn, x, q_lora: torch.Tensor, step: StepContext
    ) -> IndexQ:
        """``index_k`` + ``index_q`` when neither is replaced, as stock runs them."""
        indexer = attn.indexer
        if indexer.alt_stream is None:
            AttentionDecode.index_k(self, attn, x, step.positions, step)
            return AttentionDecode.index_q(self, attn, x, q_lora, step)
        # _fused_q_prepare_and_store (main 44b44774), two-stream branch: wq_b
        # (side) || key / head-gate projections (current), then the query kernel
        # (current) || the key kernel + index-K store (side).
        positions = step.positions
        current_stream = torch.cuda.current_stream()
        indexer.alt_stream.wait_stream(current_stream)
        with torch.cuda.stream(indexer.alt_stream):
            q = indexer.wq_b(q_lora)[0].view(-1, indexer.n_heads, indexer.head_dim)
        key, weights_raw = self._index_kw(indexer, x, step)
        current_stream.wait_stream(indexer.alt_stream)
        indexer.alt_stream.wait_stream(current_stream)
        q_fp8, weights = _index_q_quant(indexer, q, weights_raw, positions)
        with torch.cuda.stream(indexer.alt_stream):
            _store_index_k(attn, key, positions, step)
        current_stream.wait_stream(indexer.alt_stream)
        return IndexQ(q_fp8, weights)

    @staticmethod
    def _index_kw(indexer, x, step: StepContext):
        """(key, raw head gates), projecting ``x`` unless a step already did."""
        if step.index_kw is None:
            step.index_kw, _ = indexer.wk_weights_proj(x)
        return step.index_kw.split([indexer.head_dim, indexer.n_heads], dim=-1)

    def indexer(
        self, attn, index_q: Optional[IndexQ], prev_topk_indices, step: StepContext
    ):
        """Top-k cache rows per query row (page-table rows); a layer without its
        own indexer (``index_q`` None) reuses the previous layer's."""
        from sglang.srt.layers.attention.dsa.dsa_indexer import (
            _broadcast_indexer_topk_from_rank0,
        )
        from sglang.srt.model_executor.forward_context import get_attn_backend
        from sglang.srt.state_capturer.indexer_topk import maybe_capture_indexer_topk

        if index_q is None:
            # forward_absorb_prepare: a skip-top-k layer mirrors the reused
            # indices into its own capture slot.
            return maybe_capture_indexer_topk(attn.layer_id, prev_topk_indices)
        # The DSA indexer's forward_cuda (main 44b44774), target verify inside a
        # full graph (non-empty batch, logits not skipped).
        fb = step.forward_batch
        metadata = get_attn_backend().get_indexer_metadata(attn.layer_id, fb)
        if metadata is None:
            raise RuntimeError(
                "fused verify step: the attention backend has no indexer metadata"
            )
        topk_result = attn.indexer._get_topk_paged(
            fb, attn.layer_id, index_q.q_fp8, index_q.weights, metadata
        )
        topk_result = _broadcast_indexer_topk_from_rank0(topk_result)
        return maybe_capture_indexer_topk(attn.layer_id, topk_result)

    def q_absorb(self, attn, q_nope: torch.Tensor, step: StepContext) -> torch.Tensor:
        """``q_nope`` through W_UK: ``[rows, heads, kv_lora_rank]``."""
        # forward_absorb_prepare (main 44b44774), bf16 W_UK.
        return torch.bmm(q_nope.transpose(0, 1), attn.w_kc).transpose(0, 1)

    def rope_cache(
        self, attn, q_nope_out, q_pe, k_nope, k_pe, step: StepContext
    ) -> torch.Tensor:
        """RoPE, the FP8 query ``[rows, heads, kv_lora_rank + rope]`` (returned)
        and the key's MLA cache rows."""
        from sglang.kernels.ops.attention.utils import mla_quantize_and_rope_for_fp8
        from sglang.srt.model_executor.forward_context import get_attn_backend

        backend = get_attn_backend()
        layer = attn.attn_mqa
        fb = step.forward_batch
        # The key as RadixAttention.forward hands it to the backend.
        k = k_nope.view(-1, layer.tp_k_head_num, layer.v_head_dim)
        # The DSA backend's _forward_trtllm (main 44b44774), FP8 KV cache with
        # RoPE fused into the quantization (no prefill CP).
        q, k, k_rope = mla_quantize_and_rope_for_fp8(
            q_nope_out,
            q_pe,
            k.squeeze(1),
            k_pe.squeeze(1),
            fb.positions,
            attn.rotary_emb.cos_sin_cache,
            attn.rotary_emb.is_neox_style,
            backend.kv_lora_rank,
            backend.qk_rope_head_dim,
        )
        backend.token_to_kv_pool.set_mla_kv_buffer(layer, fb.out_cache_loc, k, k_rope)
        return q

    def sparse_mla(
        self, attn, q_fp8: torch.Tensor, topk_indices, step: StepContext, decode=False
    ) -> torch.Tensor:
        """Sparse MLA over the selected cache rows: ``[rows, heads, kv_lora_rank]``.
        ``decode``: a draft DECODE pass."""
        import flashinfer.decode

        from sglang.srt.environ import envs
        from sglang.srt.model_executor.forward_context import get_attn_backend

        backend = get_attn_backend()
        layer = attn.attn_mqa
        metadata = backend.forward_metadata
        # The DSA backend's _forward_trtllm (main 44b44774), FP8 query, fused
        # top-k page table, no DCP; seq_lens as forward_extend passes them for
        # target verify. Decode (forward_decode, is_prefill False): the full cache
        # lengths, and eager DP padding rows beyond the metadata's batch trimmed
        # before the kernel and restored as zeros.
        k_cache = backend.token_to_kv_pool.get_key_buffer(layer.layer_id)
        kv_cache = k_cache.view(
            -1, backend.physical_page_size, backend.kv_cache_dim
        ).unsqueeze(1)
        q_all = q_fp8.view(-1, layer.tp_q_head_num, layer.head_dim)
        if topk_indices is not None:
            topk_indices = backend._pad_topk_indices(topk_indices, q_fp8.shape[0])
        if decode:
            from sglang.srt.layers.attention.dsa_backend import (
                _restore_trtllm_decode_dp_padding,
                _trim_trtllm_decode_dp_padding,
            )

            q_all, topk_indices, n_pad = _trim_trtllm_decode_dp_padding(
                q_all, topk_indices, metadata.cache_seqlens_int32.shape[0]
            )
        page_table_1 = backend._get_fused_topk_page_table(topk_indices)
        page_table_1, sparse_mla_top_k = backend._pad_trtllm_sparse_page_table(
            page_table_1
        )
        k_scale = (
            layer.k_scale_float
            if getattr(layer, "k_scale_float", None) is not None
            else 1.0
        )
        bmm1_scale = 1.0 * k_scale * layer.scaling
        batch_size, num_heads, head_dim = q_all.shape
        multi_ctas_kv_counter_buffer = backend._multi_ctas_kv_counter_for(batch_size)
        seq_lens = None if decode else metadata.dsa_cache_seqlens_int32
        if seq_lens is None:
            seq_lens = metadata.cache_seqlens_int32
        out = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=q_all.view(batch_size, 1, num_heads, head_dim),
            block_tables=page_table_1.unsqueeze(1),
            seq_lens=seq_lens,
            kv_cache=kv_cache.view(
                -1, 1, backend.physical_page_size, backend.kv_cache_dim
            ),
            workspace_buffer=backend.workspace_buffer,
            qk_nope_head_dim=backend.qk_nope_head_dim,
            kv_lora_rank=backend.kv_lora_rank,
            qk_rope_head_dim=backend.qk_rope_head_dim,
            max_seq_len=metadata.max_seq_len_k,
            sparse_mla_top_k=sparse_mla_top_k,
            bmm1_scale=bmm1_scale,
            backend="trtllm-gen",
            skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),
            sparse_mla_top_k_lens=None,
            multi_ctas_kv_counter_buffer=multi_ctas_kv_counter_buffer,
        )
        if decode:
            out = _restore_trtllm_decode_dp_padding(out, n_pad)
        return out.view(-1, attn.num_local_heads, attn.kv_lora_rank)

    def uv_proj(self, attn, attn_output: torch.Tensor) -> torch.Tensor:
        """``[rows, heads, kv_lora]`` -> ``[rows, heads * v]``: one batched GEMM
        over the heads into the ``o_proj`` input buffer."""
        out = torch.empty(
            (attn_output.shape[0], attn.num_local_heads * attn.v_head_dim),
            dtype=attn_output.dtype,
            device=attn_output.device,
        )
        torch.bmm(
            attn_output.transpose(0, 1),
            attn.w_vc,
            out=out.view(-1, attn.num_local_heads, attn.v_head_dim).transpose(0, 1),
        )
        return out


def _store_index_k(attn, key, positions, step: StepContext) -> None:
    """The stock index-K store (``_fused_k_prepare_and_store``) of ``key``."""
    from sglang.kernels.ops.quantization.act_quant_triton import act_quant

    fb = step.forward_batch
    attn.indexer._fused_k_prepare_and_store(
        key, positions, fb, attn.layer_id, act_quant, out_cache_loc=fb.out_cache_loc
    )


def _index_q_quant(
    indexer, q: torch.Tensor, weights_raw: torch.Tensor, positions: torch.Tensor
):
    """The indexer query's RoPE + FP8 quantization and the scaled head gates
    (_fused_q_prepare_and_store, main 44b44774)."""
    from sglang.kernels.ops.attention.dsa.indexer_q import (
        fused_q_indexer_rope_first_quant,
    )

    q_scale_gate = indexer.softmax_scale * indexer.n_heads**-0.5
    return fused_q_indexer_rope_first_quant(
        q.contiguous(),
        weights_raw,
        q_scale_gate,
        indexer._indexer_cos_sin_cache,
        positions,
    )


