# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (forward_absorb_prepare, forward_absorb_core),
# python/sglang/srt/layers/attention/dsa/dsa_indexer.py (Indexer.forward_cuda, Indexer._fused_q_prepare_and_store),
# python/sglang/srt/layers/attention/dsa_backend.py (DeepseekSparseAttnBackend._forward_trtllm, _pad_topk_indices,
# _pad_trtllm_sparse_page_table).
"""The stock verify attention of one GLM-5.3 layer, in steps, and the per-step state
the step groups share.

``AttentionStep`` copies the verify slices of ``forward_absorb_prepare`` /
``forward_absorb_core`` (forward_mla.py), the DSA indexer's ``forward_cuda`` and the
DSA backend's ``_forward_trtllm`` at SGLang main af856717 for the one configuration
the spec serves (bf16 W_UK / W_UV, fused indexer projections, FP8 KV cache on the
TRT-LLM sparse kernel, fused top-k page table, no indexer top-k broadcast, full-graph
verify); the spec refuses every other one. The attention backend's state is read
through the step's ``ForwardView`` (``StepContext.view``: its ``dsa`` view and its KV
pool); the backend's two pure padding helpers are copied here. A kernel covering
several steps overrides the first and passes the others through, handing state on
``StepContext``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/binding.py": (
        "metadata",
        "kv_pool",
        "dsa",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "kv_cache_dim",
        "physical_page_size",
        "workspace_buffer",
        "multi_ctas_kv_counter_buffer",
        "indexer_metadata",
        "get_key_buffer",
        "set_mla_kv_buffer",
        "decode_impl",
        "use_mha",
        "kv_cache_dtype",
        "use_fused_topk",
        "topk_backend",
        "dcp_size",
        "index_kpool",
        "hisparse",
    ),
    "srt/layers/layer_boundary/__init__.py": ("get_attn_tp_context",),
    "srt/layers/layer_boundary/adapters/attention.py": (
        "AttentionInputs",
        "set_attn_inputs",
        "clear_attn_inputs",
        "fetch_qkv_latent",
    ),
    "srt/model_executor/runner/__init__.py": ("get_is_capture_mode",),
    "srt/layers/attention/dsa/dsa_indexer.py": (
        "_get_topk_paged",
        "_fused_k_prepare_and_store",
        "_fused_k_prepare_and_store(out_cache_loc)",
        "wk_weights_proj",
        "wq_b",
        "n_heads",
        "head_dim",
        "softmax_scale",
        "_indexer_cos_sin_cache",
        "alt_stream",
        "rotary_emb",
        "use_dsa_indexer_fusion",
        "use_gfx950_fused_indexer",
        "dsa_enable_prefill_cp",
        "_aiter_fused_fp8_active",
    ),
    "srt/state_capturer/indexer_topk.py": ("maybe_capture_indexer_topk",),
    "kernels/ops/attention/utils.py": ("mla_quantize_and_rope_for_fp8",),
    "kernels/ops/quantization/act_quant_triton.py": ("act_quant",),
    "kernels/ops/attention/dsa/indexer_q.py": ("fused_q_indexer_rope_first_quant",),
    "srt/environ.py": ("envs", "SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR"),
    "srt/models/deepseek_common/attention_forward_methods/forward_methods.py": (
        "AttnForwardMethod",
        "MLA",
    ),
    "srt/layers/attention/dsa/dsa_topk_backend.py": (
        "is_sgl_kernel",
        "is_flashinfer",
    ),
    # the replaced model's attention modules
    "srt/models/deepseek_common/attention_forward_methods/forward_mla.py": (
        "should_run_indexer",
        "_split_q_nope_pe",
        "_fuse_rope_for_trtllm_mla",
        "_can_fuse_bmm_into_attention",
    ),
    "srt/models/deepseek_v2.py": (
        "prepare_qkv_latent",
        "maybe_use_decode_attn_tp",
        "q_b_proj_forward",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_rope_head_dim",
        "q_a_layernorm",
        "kv_a_layernorm",
        "alt_stream",
        "w_kc",
        "w_vc",
        "attn_mqa",
        "rotary_emb",
        "num_local_heads",
        "v_head_dim",
        "next_skip_topk",
        "layer_id",
        "dispatch_attn_forward_method",
        "self_attn",
        "skip_topk",
        "start_layer",
        "end_layer",
        "layers",
    ),
    "srt/layers/radix_attention.py": (
        "tp_k_head_num",
        "tp_q_head_num",
        "v_head_dim",
        "head_dim",
        "scaling",
        "k_scale_float",
    ),
    "srt/model_executor/forward_batch_info.py": ("attn_cp_metadata",),
    "srt/layers/rotary_embedding/base.py": ("is_neox_style",),
}

# The SGLang function bodies this module's stock steps copy, by the digest of their
# code (test/unit/test_contract.py pins them)
SGLANG_MIRRORS = {
    # before and after sgl-project/sglang#46359, which changed only the
    # --dcp-replicate-q-proj lines (decode context parallelism, refused here), and
    # after sgl-project/sglang#45946, which added SAIL-only fp8 conditions and a
    # FlashMLA-backend-only llama_4_scaling step (the DSA path served here is unchanged)
    "srt/models/deepseek_common/attention_forward_methods/forward_mla.py:DeepseekMLAForwardMixin.forward_absorb_prepare": (
        "40202da2cffe9560",
        "4f3bf6125427f3c7",
        "13dadb6e42fe43f5",
    ),
    # before and after sgl-project/sglang#45946, which added a SAIL-only fp8 condition
    "srt/models/deepseek_common/attention_forward_methods/forward_mla.py:DeepseekMLAForwardMixin.forward_absorb_core": (
        "8a3f638957ef5cd7",
        "69ba2c4967bde418",
    ),
    "srt/layers/attention/dsa/dsa_indexer.py:Indexer.forward_cuda": "805ab11bf040ed52",
    "srt/layers/attention/dsa/dsa_indexer.py:Indexer._fused_q_prepare_and_store": "6ac8fd1c71413a93",
    # before and after sgl-project/sglang#46990, which stores and reads the KV through
    # the pool's layer router: on the DSA pool, the same set_mla_kv_buffer and
    # get_key_buffer calls
    "srt/layers/attention/dsa_backend.py:DeepseekSparseAttnBackend._forward_trtllm": (
        "d31092c1f10bc4a1",
        "76c87b0740d2b05d",
    ),
    # the fused top-k page table is the top-k indices on the top-k backends the spec
    # serves (attention_step_reasons); the two padding helpers are copied below
    "srt/layers/attention/dsa_backend.py:DeepseekSparseAttnBackend._get_fused_topk_page_table": "b4cc1d5b7eef0b8d",
    "srt/layers/attention/dsa_backend.py:DeepseekSparseAttnBackend._pad_topk_indices": "a6c6754bca9b3add",
    "srt/layers/attention/dsa_backend.py:DeepseekSparseAttnBackend._pad_trtllm_sparse_page_table": "af41a74e3fc0830b",
}


@dataclass
class StepContext:
    """Per-forward state shared by the step groups of one verify forward."""

    forward_batch: Any
    positions: torch.Tensor
    zero_allocator: Any
    # The running forward's ForwardView (Step.view): the attention metadata, the KV
    # pool and the DSA backend's view.
    view: Any = None
    # Set by a routing hop: (expert ids [rows, 8] int32, weights [rows, 8] fp32);
    # read and cleared by the same layer's MoE step.
    routing: Any = None
    # Per layer, cleared by AttentionStep.forward: the indexer key / head-gate
    # projections and the FP8 MLA query when a fused step produced them early.
    index_kw: Any = None
    q_fp8: Any = None
    # A deferred input norm (slot dnorm): (layer id, block sums [rows, 8] fp32); read
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


# Host-only facts of a verify step that the attention steps need: whether the stock
# attention runs a batch on the absorbed MLA path (``absorbed_mla``), why it would
# leave the branches ``AttentionStep`` copies (``attention_step_reasons``, read at the
# first capture of each width, once the attention backend exists), and why a layer's
# RoPE does not fit the fused MLA prologue (``rope_style_reasons``).
def absorbed_mla(model, forward_batch) -> bool:
    """Whether stock attention runs this batch on the absorbed MLA path (the one
    the attention steps implement)."""
    from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
        AttnForwardMethod,
    )

    attn = model.model.layers[model.model.start_layer].self_attn
    return attn.dispatch_attn_forward_method(forward_batch) == AttnForwardMethod.MLA


def rope_style_reasons(attn) -> List[str]:
    """Why a layer's RoPE does not fit the fused MLA prologue (empty if it does): the
    prologue rotates interleaved pairs (2i, 2i + 1) of q_pe / k_pe and of the
    index-K rope dims, so a NeoX-style (i, i + 32) rotary of the attention or of its
    indexer is refused (``rope_interleave`` / ``indexer_rope_interleave`` false)."""
    why = []
    if attn.rotary_emb is not None and attn.rotary_emb.is_neox_style:
        why.append("RoPE is NeoX style, not interleaved")
    index_rope = getattr(getattr(attn, "indexer", None), "rotary_emb", None)
    if index_rope is not None and index_rope.is_neox_style:
        why.append("the indexer's RoPE is NeoX style, not interleaved")
    return why


def attention_step_reasons(model, forward_batch, view) -> List[str]:
    """Why stock verify attention would leave the branches ``AttentionStep`` copies
    (empty if it would not); ``view`` is the running forward's ForwardView."""
    from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
        AttnForwardMethod,
    )

    dsa = view.dsa
    if dsa is None:
        return ["the attention backend is not the DSA backend (no DSA view)"]
    reasons: List[str] = []
    if dsa.decode_impl != "trtllm" or dsa.use_mha:
        reasons.append(
            f"verify attention is not the TRT-LLM sparse MLA kernel "
            f"(impl {dsa.decode_impl!r}, use_mha {dsa.use_mha})"
        )
    if dsa.kv_cache_dtype != torch.float8_e4m3fn:
        reasons.append(f"KV cache dtype {dsa.kv_cache_dtype} is not fp8_e4m3")
    if not dsa.use_fused_topk:
        reasons.append("the fused DSA top-k page table is off")
    elif not (dsa.topk_backend.is_sgl_kernel() or dsa.topk_backend.is_flashinfer()):
        reasons.append(f"DSA top-k backend {dsa.topk_backend} is not supported")
    if dsa.dcp_size != 1 or dsa.index_kpool > 1:
        reasons.append(
            "decode context parallelism / index K-pool layouts are not supported"
        )
    if dsa.hisparse:
        reasons.append("HiSparse is not supported")
    if forward_batch.attn_cp_metadata is not None:
        reasons.append("attention context parallelism is not supported")

    mm = model.model
    for i in range(mm.start_layer, mm.end_layer):
        attn = mm.layers[i].self_attn
        indexer = getattr(attn, "indexer", None)
        why = []
        if attn.dispatch_attn_forward_method(forward_batch) != AttnForwardMethod.MLA:
            why.append("not absorbed MLA")
        if attn.q_lora_rank is None or attn.rotary_emb is None:
            why.append("no q_lora / RoPE")
        if not attn._fuse_rope_for_trtllm_mla(forward_batch):
            why.append("RoPE is not fused into the FP8 query quantization")
        if attn._can_fuse_bmm_into_attention(forward_batch):
            why.append("the W_UK bmm is fused into attention")
        if getattr(attn, "w_kc", None) is None or attn.w_kc.dtype != torch.bfloat16:
            why.append("W_UK is not bf16")
        if getattr(attn, "_kimi_split_gguf_kv_b", False) or hasattr(
            attn, "prepare_attention_output_gate"
        ):
            why.append("unsupported attention variant")
        if attn.attn_mqa.is_cross_attention:
            why.append("cross attention")
        if attn.skip_topk is None or (not attn.skip_topk and indexer is None):
            why.append("no DSA indexer")
        elif not attn.skip_topk:
            if not indexer.use_dsa_indexer_fusion:
                why.append("the indexer projections are not fused")
            if (
                indexer.use_gfx950_fused_indexer
                or indexer.dsa_enable_prefill_cp
                or indexer._aiter_fused_fp8_active(forward_batch)
            ):
                why.append("unsupported indexer path")
        if why:
            reasons.append(f"layer {i}: " + ", ".join(why))
            if len(reasons) > 8:
                break
    return reasons


class AttentionStep:
    """Stock verify attention of one ``DeepseekV2AttentionMLA`` layer, in steps:
    ``qkv_a`` -> ``prologue`` -> (own indexer) ``index_k`` -> ``index_q`` ->
    ``indexer`` -> ``q_absorb`` -> ``rope_cache`` -> ``sparse_mla`` -> ``uv_proj``.
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
        # The stock bodies copy the branches of target-verify forwards only.
        if not mode.is_target_verify():
            raise RuntimeError(
                f"AttentionStep serves target-verify forwards only (got {mode})"
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

        # forward_absorb_prepare (main af856717), verify: the kv_a norm runs on
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
        # _fused_q_prepare_and_store (main af856717), single-stream key half.
        key, _ = self._index_kw(attn.indexer, x, step)
        _store_index_k(attn, key, positions, step)

    def index_q(self, attn, x, q_lora: torch.Tensor, step: StepContext) -> IndexQ:
        """The indexer query (``wq_b``, RoPE, FP8) and its head gates."""
        # _fused_q_prepare_and_store (main af856717), single-stream query half.
        indexer = attn.indexer
        _, weights_raw = self._index_kw(indexer, x, step)
        q = indexer.wq_b(q_lora)[0].view(-1, indexer.n_heads, indexer.head_dim)
        return IndexQ(*_index_q_quant(indexer, q, weights_raw, step.positions))

    def _stock_index_steps(self) -> bool:
        cls = type(self)
        return (
            cls.index_k is AttentionStep.index_k
            and cls.index_q is AttentionStep.index_q
        )

    def _index_k_and_q(
        self, attn, x, q_lora: torch.Tensor, step: StepContext
    ) -> IndexQ:
        """``index_k`` + ``index_q`` when neither is replaced, as stock runs them."""
        indexer = attn.indexer
        if indexer.alt_stream is None:
            AttentionStep.index_k(self, attn, x, step.positions, step)
            return AttentionStep.index_q(self, attn, x, q_lora, step)
        # _fused_q_prepare_and_store (main 74c08f3dc5), two-stream branch: key /
        # head-gate projections (current) || wq_b (side, issued after them so the
        # join stays on the main lane at graph replay), then the query kernel
        # (current) || the key kernel + index-K store (side).
        positions = step.positions
        current_stream = torch.cuda.current_stream()
        indexer.alt_stream.wait_stream(current_stream)
        key, weights_raw = self._index_kw(indexer, x, step)
        with torch.cuda.stream(indexer.alt_stream):
            q = indexer.wq_b(q_lora)[0].view(-1, indexer.n_heads, indexer.head_dim)
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
        from sglang.srt.state_capturer.indexer_topk import maybe_capture_indexer_topk

        if index_q is None:
            # forward_absorb_prepare: a skip-top-k layer mirrors the reused
            # indices into its own capture slot.
            return maybe_capture_indexer_topk(attn.layer_id, prev_topk_indices)
        # The DSA indexer's forward_cuda (main af856717), target verify inside a
        # full graph (non-empty batch, logits not skipped; its rank-0 top-k
        # broadcast is off: the spec requires SGLANG_DSA_TOPK_BROADCAST unset).
        fb = step.forward_batch
        metadata = step.view.dsa.indexer_metadata(attn.layer_id, fb)
        if metadata is None:
            raise RuntimeError(
                "fused verify step: the attention backend has no indexer metadata"
            )
        topk_result = attn.indexer._get_topk_paged(
            fb, attn.layer_id, index_q.q_fp8, index_q.weights, metadata
        )
        return maybe_capture_indexer_topk(attn.layer_id, topk_result)

    def q_absorb(self, attn, q_nope: torch.Tensor, step: StepContext) -> torch.Tensor:
        """``q_nope`` through W_UK: ``[rows, heads, kv_lora_rank]``."""
        # forward_absorb_prepare (main af856717), bf16 W_UK.
        return torch.bmm(q_nope.transpose(0, 1), attn.w_kc).transpose(0, 1)

    def rope_cache(
        self, attn, q_nope_out, q_pe, k_nope, k_pe, step: StepContext
    ) -> torch.Tensor:
        """RoPE, the FP8 query ``[rows, heads, kv_lora_rank + rope]`` (returned)
        and the key's MLA cache rows."""
        from sglang.kernels.ops.attention.utils import mla_quantize_and_rope_for_fp8

        dsa = step.view.dsa
        layer = attn.attn_mqa
        fb = step.forward_batch
        # The key as RadixAttention.forward hands it to the backend.
        k = k_nope.view(-1, layer.tp_k_head_num, layer.v_head_dim)
        # The DSA backend's _forward_trtllm (main af856717), FP8 KV cache with
        # RoPE fused into the quantization (no prefill CP).
        q, k, k_rope = mla_quantize_and_rope_for_fp8(
            q_nope_out,
            q_pe,
            k.squeeze(1),
            k_pe.squeeze(1),
            fb.positions,
            attn.rotary_emb.cos_sin_cache,
            attn.rotary_emb.is_neox_style,
            dsa.kv_lora_rank,
            dsa.qk_rope_head_dim,
        )
        step.view.kv_pool.set_mla_kv_buffer(layer, fb.out_cache_loc, k, k_rope)
        return q

    def sparse_mla(
        self, attn, q_fp8: torch.Tensor, topk_indices, step: StepContext
    ) -> torch.Tensor:
        """Sparse MLA over the selected cache rows: ``[rows, heads, kv_lora_rank]``."""
        import flashinfer.decode
        from sglang.srt.environ import envs

        view = step.view
        dsa = view.dsa
        layer = attn.attn_mqa
        metadata = view.metadata
        # The DSA backend's _forward_trtllm (main af856717), FP8 query, fused
        # top-k page table (the top-k indices themselves on the served top-k
        # backends), no DCP; seq_lens from sparse_mla_seq_lens.
        k_cache = view.kv_pool.get_key_buffer(layer.layer_id)
        kv_cache = k_cache.view(-1, dsa.physical_page_size, dsa.kv_cache_dim).unsqueeze(
            1
        )
        q_all = q_fp8.view(-1, layer.tp_q_head_num, layer.head_dim)
        if topk_indices is not None:
            topk_indices = pad_topk_indices(topk_indices, q_fp8.shape[0])
        page_table_1, sparse_mla_top_k = pad_trtllm_sparse_page_table(topk_indices)
        k_scale = (
            layer.k_scale_float
            if getattr(layer, "k_scale_float", None) is not None
            else 1.0
        )
        bmm1_scale = 1.0 * k_scale * layer.scaling
        batch_size, num_heads, head_dim = q_all.shape
        multi_ctas_kv_counter_buffer = dsa.multi_ctas_kv_counter_buffer
        seq_lens = self.sparse_mla_seq_lens(metadata, step, batch_size)
        out = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=q_all.view(batch_size, 1, num_heads, head_dim),
            block_tables=page_table_1.unsqueeze(1),
            seq_lens=seq_lens,
            kv_cache=kv_cache.view(-1, 1, dsa.physical_page_size, dsa.kv_cache_dim),
            workspace_buffer=dsa.workspace_buffer,
            qk_nope_head_dim=dsa.qk_nope_head_dim,
            kv_lora_rank=dsa.kv_lora_rank,
            qk_rope_head_dim=dsa.qk_rope_head_dim,
            max_seq_len=metadata.max_seq_len_k,
            sparse_mla_top_k=sparse_mla_top_k,
            bmm1_scale=bmm1_scale,
            backend="trtllm-gen",
            skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),
            sparse_mla_top_k_lens=None,
            multi_ctas_kv_counter_buffer=multi_ctas_kv_counter_buffer,
        )
        return out.view(-1, attn.num_local_heads, attn.kv_lora_rank)

    def sparse_mla_seq_lens(self, metadata, step: StepContext, rows: int):
        """The cache lengths the sparse MLA reads: as forward_extend passes them for
        target verify."""
        seq_lens = metadata.dsa_cache_seqlens_int32
        return metadata.cache_seqlens_int32 if seq_lens is None else seq_lens

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


def pad_topk_indices(topk_indices: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """``DeepseekSparseAttnBackend._pad_topk_indices`` (main af856717): -1 rows up to
    ``num_tokens``."""
    current_tokens = topk_indices.shape[0]
    if current_tokens == num_tokens:
        return topk_indices
    assert current_tokens <= num_tokens, (
        f"topk_indices rows ({current_tokens}) > num_tokens ({num_tokens}); "
        "this indicates a mismatch between indexer output and q layout."
    )
    pad_size = num_tokens - current_tokens
    padding = torch.full(
        (pad_size, topk_indices.shape[1]),
        -1,
        dtype=topk_indices.dtype,
        device=topk_indices.device,
    )
    return torch.cat([topk_indices, padding], dim=0)


def pad_trtllm_sparse_page_table(page_table_1: torch.Tensor):
    """``DeepseekSparseAttnBackend._pad_trtllm_sparse_page_table`` (main af856717):
    the page table padded with -1 to a multiple of 4 columns, and that width."""
    sparse_mla_top_k = page_table_1.shape[1]
    padded_top_k = ((sparse_mla_top_k + 3) // 4) * 4
    if padded_top_k == sparse_mla_top_k:
        return page_table_1, sparse_mla_top_k
    padding = torch.full(
        (page_table_1.shape[0], padded_top_k - sparse_mla_top_k),
        -1,
        dtype=page_table_1.dtype,
        device=page_table_1.device,
    )
    return torch.cat([page_table_1, padding], dim=1), padded_top_k


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
    (_fused_q_prepare_and_store, main af856717)."""
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
