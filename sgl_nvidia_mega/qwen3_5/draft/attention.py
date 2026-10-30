# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/qwen3_5.py (Qwen3_5AttentionDecoderLayer.self_attention) and
# python/sglang/srt/layers/attention/trtllm_mha_backend.py (TRTLLMHAAttnBackend
# forward_decode / forward_extend on the fused FP8 KV path, _run_fixed_q_len_decode).
"""The MTP draft layer's gated full attention inside a draft entry, which gets the
step's attention metadata and the draft's KV pool from SGLang's binding
(``AttnStepMetadata``, ``KvPoolBuffers``) instead of a forward batch.

Stock: SGLang's ops in the order the TRT-LLM MHA backend runs them on an FP8 KV cache:
the qkv projection with the fused q / k Gemma RMSNorm + RoPE + gate split, the fused
FP8 quantization of q / k / v with the paged KV write, trtllm-gen's decode with
``q_len_per_req`` rows a request (1 for a draft step, the verify width for the draft
extend), the sigmoid output gate and ``o_proj``. Slot ``draft_q1`` runs a draft step's
decode as the q1 split-KV kernel (``sm103/draft/q1``).
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# the KV-length graph variants of the draft loop (2.2's graph_variants.Q1_STOCK /
# Q1_LONG): trtllm-gen below Q1_MIN_LENGTH, SKV2-Q1 from it (2.2's
# SGLANG_Q35MF_MKA_Q1_MIN_LENGTH default, which its TP8 serving configuration keeps)
Q1_STOCK, Q1_LONG = "q1_stock", "q1_long"
Q1_MIN_LENGTH = 16384
# slot draft_q1_cp: one request past attention.route.CP_MIN_KV["draft"] runs its steps'
# decode as CP-lite (SGLang's read-split attention over the KV group)
Q1_CP = "q1_cp"
# A draft step's q1 decode and its merge launch as programmatic dependents, the
# one-row front releasing them at entry (same arithmetic)
Q1_PDL = True
# The one-row front launches as a programmatic dependent of the qkv GEMM, loading its
# static operands before the wait (same arithmetic)
QKN_PDL = True
# 1: slot draft_hop's one-row o_proj stages attn * sigmoid(gate) itself, in the gate
# kernel's arithmetic, in place of that kernel; 2: also, under Q1_PDL, the q1 merge
# releases that o_proj at entry
GATE_FOLD = 2


class Q1Variants:
    """The draft loop's graph variants (SGLang's AttentionGraphVariants), as 2.2's
    Q1GraphVariants: SKV2-Q1 from a host-only KV length lower bound, and with
    ``cp_min_length`` (slot draft_q1_cp) CP-lite from that length."""

    # select picks Q1_STOCK past one request (q1 decodes one row), so SGLang captures
    # one graph at every larger batch size
    max_batch_size = 1

    def __init__(self, min_length: int = Q1_MIN_LENGTH, cp_min_length=None):
        self.min_length = min_length
        self.cp_min_length = cp_min_length
        # stock first, as 2.2 captures them (variants share a graph memory pool)
        self.capture_labels = (Q1_STOCK, Q1_LONG) + (
            (Q1_CP,) if cp_min_length is not None else ()
        )

    def select(self, forward_batch) -> str:
        if forward_batch.batch_size != 1:
            return Q1_STOCK
        length = getattr(forward_batch, "attention_kv_len_lower_bound", None)
        if length is None:
            lengths = forward_batch.seq_lens_cpu
            if (
                lengths is not None
                and lengths.device.type == "cpu"
                and lengths.numel() == 1
            ):
                length = int(lengths[0])
        if length is None:
            return Q1_STOCK
        if self.cp_min_length is not None and length >= self.cp_min_length:
            return Q1_CP
        return Q1_LONG if length >= self.min_length else Q1_STOCK


# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/qwen3_5.py": (
        "forward_prepare_cuda_fused",
        "attn_output_gate",
        "o_proj",
        "qkv_proj",
        "q_size",
        "kv_size",
        "q_norm",
        "k_norm",
        "rotary_emb",
        "num_heads",
        "num_kv_heads",
        "head_dim",
    ),
    "srt/layers/radix_attention.py": (
        "tp_q_head_num",
        "tp_k_head_num",
        "tp_v_head_num",
        "head_dim",
        "scaling",
        "sliding_window_size",
        "k_scale",
        "v_scale",
    ),
    "kernels/ops/kvcache/fused_fp8_qkv_kv_cache.py": ("fused_fp8_qkv_kv_cache",),
    "kernels/ops/elementwise/elementwise.py": (
        "fused_sigmoid_mul",
        "fused_sigmoid_mul(inplace)",
    ),
    "kernels/ops/attention/utils.py": ("canonicalize_stride",),
    "srt/mem_cache/layout/paged_view.py": ("paged_kv_view",),
    # TRTLLMHAAttnBackend: its shared workspace, its counters and max_context_len, and
    # _run_fixed_q_len_decode's trtllm-gen call
    "srt/layers/attention/trtllm_mha_backend.py": (
        "DEFAULT_WORKSPACE_SIZE_MB",
        '"trtllm_mha_zero_workspace"',
        "max_context_len",
        "_multi_ctas_kv_counter_buffer",
        "_run_fixed_q_len_decode",
        "flashinfer",
    ),
    "srt/layers/attention/trtllm_mla_backend.py": (
        "make_persistent_multi_ctas_kv_counter_buffer",
    ),
    "srt/environ.py": ("envs", "SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR"),
    "srt/runtime_context.py": ("get_buffer",),
    "srt/model_executor/runner_utils/capture_mode.py": (
        "get_capture_attention_variant",
    ),
    # ModelConfig._derive_context_length writes the backend's max_context_len into the
    # HF config the draft's build gets
    "srt/configs/model_config.py": ("context_len",),
}

# The SGLang functions whose body this module mirrors, by the digest of their code
# (test/unit/test_contract.py recomputes them on the installed SGLang)
SGLANG_MIRRORS = {
    "srt/models/qwen3_5.py:Qwen3_5AttentionDecoderLayer.self_attention": "db0f4d98e0fb1abf",
}


def _context_len(draft) -> int:
    """The max_seq_len SGLang's TRT-LLM MHA backend gives trtllm-gen: the draft's
    ModelConfig.context_len, which it writes into the HF config the build gets."""
    if draft.context_len is None:
        raise RuntimeError(
            "NVIDIA megakernel: the draft's HF config has no context_len"
        )
    return int(draft.context_len)


class Attention:
    """The draft layer's attention buffers: trtllm-gen's zeroed workspace (SGLang's
    shared one) and multi-CTA counters, and the FP8 KV scales."""

    def __init__(self, parts):
        from sglang.srt.layers.attention.trtllm_mha_backend import (
            DEFAULT_WORKSPACE_SIZE_MB,
        )
        from sglang.srt.layers.attention.trtllm_mla_backend import (
            make_persistent_multi_ctas_kv_counter_buffer,
        )
        from sglang.srt.runtime_context import get_buffer

        layer, device = parts.layer, parts.device
        attn = layer.attn
        self.layer = layer
        self.heads = attn.tp_q_head_num
        self.head_dim = attn.head_dim
        self.page_size = parts.binding.page_size
        self.max_context_len = _context_len(parts.draft)
        self.workspace = get_buffer(
            "trtllm_mha_zero_workspace",
            lambda: torch.zeros(
                DEFAULT_WORKSPACE_SIZE_MB * 1024 * 1024,
                dtype=torch.uint8,
                device=device,
            ),
        )
        # the draft's rows a call at most: the verify width of the draft extend
        width = int(parts.binding.tokens_per_request)
        self.counters = make_persistent_multi_ctas_kv_counter_buffer(
            device,
            num_q_heads=parts.lm.config.num_attention_heads,
            max_batch_size=width,
        )
        ones = torch.ones(1, dtype=torch.float32, device=device)
        self.k_scale = attn.k_scale if attn.k_scale is not None else ones
        self.v_scale = attn.v_scale if attn.v_scale is not None else ones
        self.q1 = None  # sm103.draft.q1 decode (slot draft_q1), set by DraftLoop
        self._noted: set = set()  # engagement lines logged (once each, under capture)
        # slot draft_front: 2.2's attention front writes the KV and FP8 q in one launch
        self.front = "draft_front" in parts.slots
        # slot draft_skv2: the draft extend's split-KV decode (DraftExtendSplitKV)
        self.skv2 = (
            DraftExtendSplitKV.build(parts) if "draft_skv2" in parts.slots else None
        )
        # slots draft_q1_cp / draft_skv2_cp: one request's steps / extend in their
        # CP graph variant run CP-lite at one request (DraftCP)
        self.cp_b1 = DraftCP.build(parts)

    def __call__(
        self, x, positions, cache_loc, meta, kv, *, rows_per_request: int, o_proj=None
    ):
        """The TP-partial attention output [rows, hidden] of the normed input ``x``
        [rows, hidden]; writes this step's KV rows at ``cache_loc``. ``o_proj`` (the
        draft hop's GEMV) stands in for SGLang's when given."""
        from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

        layer = self.layer
        if self.front:
            q8, gate = self._front(x, positions, cache_loc, kv)
            o = self.decode(q8, meta, kv, rows_per_request)
        else:
            q, k, v, gate = layer.forward_prepare_cuda_fused(
                positions=positions, hidden_states=x
            )
            o = self.core(q, k, v, cache_loc, meta, kv, rows_per_request)
        if layer.attn_output_gate:
            if (
                o_proj is not None
                and GATE_FOLD
                and gate.shape == o.shape
                and gate.is_contiguous()
            ):
                self._note(
                    "slot draft_hop's one-row o_proj applies the attention output gate"
                )
                return o_proj(o, gate=gate)
            o = fused_sigmoid_mul(o, gate, inplace=True)
        if o_proj is not None:
            return o_proj(o)
        out, _ = layer.o_proj(o)
        return out

    def _front(self, x, positions, cache_loc, kv):
        """Slot draft_front (2.2's ATTN_FRONT): the qkv projection, then one Triton launch for
        the q / k Gemma RMSNorm, RoPE, gate split, FP8 q, and K / V into the FP8 cache
        at ``cache_loc``: (q8 [rows, heads * head_dim], gate)."""
        from sgl_nvidia_mega.qwen3_5.sm103.draft.front import (
            fused_qk_gemma_rmsnorm_rope_gate,
        )

        layer = self.layer
        qkv, _ = layer.qkv_proj(x)
        q_gate_size = layer.q_size * (2 if layer.attn_output_gate else 1)
        q_gate, k, v = qkv.split([q_gate_size, layer.kv_size, layer.kv_size], dim=-1)
        q8, _, gate = fused_qk_gemma_rmsnorm_rope_gate(
            q_gate,
            k,
            layer.q_norm.weight.data,
            layer.k_norm.weight.data,
            layer.rotary_emb.cos_sin_cache,
            positions,
            layer.q_norm.variance_epsilon,
            layer.num_heads,
            layer.num_kv_heads,
            layer.head_dim,
            layer.rotary_emb.rotary_dim,
            has_gate=layer.attn_output_gate,
            mrope_axis_map=(
                layer.rotary_emb.axis_map if positions.dim() == 2 else None
            ),
            v=v,
            kv_cache=(kv.k[0], kv.v[0]),
            cache_loc=cache_loc,
            k_scale=self.k_scale,
            v_scale=self.v_scale,
            early_trigger=x.shape[0] == 1 and self.q1 is not None and Q1_PDL,
            pdl_wait=x.shape[0] == 1 and QKN_PDL,
        )
        if x.shape[0] == 1 and QKN_PDL and layer.attn_output_gate:
            self._note(
                "slot draft_front launches the one-row front as a programmatic "
                "dependent of the qkv GEMM"
            )
        rows = x.shape[0]
        return q8.view(rows, -1), (gate.view(rows, -1) if gate is not None else None)

    def _note(self, what: str) -> None:
        """Log ``what`` once, the first time a graph capture runs it."""
        if what not in self._noted and torch.cuda.is_current_stream_capturing():
            self._noted.add(what)
            logger.info("NVIDIA megakernel: Qwen3.5 MTP draft %s", what)

    def core(self, q, k, v, cache_loc, meta, kv, rows_per_request: int):
        """FP8 q / k / v with the KV write, then the decode: [rows, heads * head_dim]."""
        from sglang.kernels.ops.kvcache.fused_fp8_qkv_kv_cache import (
            fused_fp8_qkv_kv_cache,
        )

        q8 = fused_fp8_qkv_kv_cache(
            q=q,
            k=k,
            v=v,
            k_cache=kv.k[0],
            v_cache=kv.v[0],
            cache_loc=cache_loc,
            k_scale=self.k_scale,
            v_scale=self.v_scale,
        )
        return self.decode(q8, meta, kv, rows_per_request)

    def decode(self, q8, meta, kv, rows_per_request: int):
        """The decode of FP8 q rows over the cache (this step's rows written):
        [rows, heads * head_dim]."""
        import flashinfer
        from sglang.srt.environ import envs

        k_cache, v_cache = kv.k[0], kv.v[0]
        q8 = q8.reshape(-1, self.heads, self.head_dim)
        attn = self.layer.attn
        bmm1_scale = 1.0 * kv.k_scale[0] * attn.scaling
        bmm2_scale = kv.v_scale[0]
        if self.cp_b1 is not None and self.cp_b1.takes(rows_per_request, q8.shape[0]):
            o = self.cp_b1.run(self, q8, meta, kv, rows_per_request)
            if o is not None:
                return o
        # q1 decodes one request
        if (
            self.q1 is not None
            and rows_per_request == 1
            and q8.shape[0] == 1
            and _q1_label() != Q1_STOCK
        ):
            return self.q1(q8, kv, meta).view(-1, self.heads * self.head_dim)
        if (
            self.skv2 is not None
            and rows_per_request > 1
            and q8.shape[0] == rows_per_request
            and self.skv2.long()
        ):
            return self.skv2.run(attn, q8, kv, meta)
        kwargs = {}
        if rows_per_request != 1:
            kwargs["q_len_per_req"] = rows_per_request
        o = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=q8,
            kv_cache=self._paged(k_cache, v_cache),
            workspace_buffer=self.workspace,
            block_tables=meta.page_table,
            seq_lens=meta.cache_seqlens,
            max_seq_len=self.max_context_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            window_left=attn.sliding_window_size,
            sinks=None,
            skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),
            out=None,
            out_dtype=torch.bfloat16,
            kv_cache_sf=None,
            multi_ctas_kv_counter_buffer=self.counters,
            **kwargs,
        )
        return o.view(-1, self.heads * self.head_dim)

    def _paged(self, k_cache, v_cache):
        """The pool buffers as trtllm-gen's HND pages, as the backend views them."""
        from sglang.kernels.ops.attention.utils import canonicalize_stride
        from sglang.srt.mem_cache.layout.paged_view import paged_kv_view

        attn = self.layer.attn
        k = paged_kv_view(k_cache, self.page_size, attn.tp_k_head_num, self.head_dim)
        v = paged_kv_view(v_cache, self.page_size, attn.tp_v_head_num, self.head_dim)
        k, v = k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
        if attn.tp_k_head_num == 1:
            k = canonicalize_stride(k)
        if attn.tp_v_head_num == 1:
            v = canonicalize_stride(v)
        return k, v


class DraftExtendSplitKV:
    """Slot draft_skv2: the draft extend's 7-row decode as 2.2's SKV2 (MKA_SKV2_DRAFT,
    from 32768 tokens of KV), through the target attention package's SKV2 op
    (``qwen3_5.attention.split_kv``, slot skv2's) on the draft's pool."""

    def __init__(self, split_kv, route):
        self.split_kv, self.route = split_kv, route
        self._logged = False

    @classmethod
    def build(cls, parts):
        try:
            from sgl_nvidia_mega.qwen3_5.attention import route, split_kv
        except ImportError as e:
            raise RuntimeError(
                "NVIDIA megakernel: slot draft_skv2 needs the attention package's SKV2 "
                f"(qwen3_5.attention.split_kv): {e}"
            ) from e
        width = int(parts.binding.tokens_per_request)
        why = split_kv.unsupported(parts.layer.attn, width)
        if why:
            raise RuntimeError(f"NVIDIA megakernel: slot draft_skv2 cannot run: {why}")
        return cls(split_kv, route)

    def long(self) -> bool:
        """The draft-extend graph being captured is the split-KV one ("split_kv_long"),
        or SGLang captures no label of ours (then every length, as for the verify)."""
        label = _q1_label()
        return label is None or label == self.route.SPLIT_KV_LONG

    def run(self, attn, q8, kv, meta):
        if not self._logged and torch.cuda.is_current_stream_capturing():
            self._logged = True
            logger.info(
                "NVIDIA megakernel: Qwen3.5 MTP draft slot draft_skv2 runs the draft "
                "extend's %d rows as SKV2",
                q8.shape[0],
            )
        return self.split_kv.skv2(attn, q8, kv, meta)


class DraftCP:
    """Slots draft_q1_cp / draft_skv2_cp: in the draft loop's ``Q1_CP`` and the draft
    extend's ``SPLIT_KV_CP`` graph variants (one request past its measured crossover),
    the decode of the request's rows as the plugin's CP-lite, which then serves one
    request through ``attention.split_kv.cp_core``. Slot
    draft_multi also preserves CP-lite on multi-request graphs where SGLang built it;
    CP-lite's own shape gate decides whether it runs."""

    def __init__(self, split_kv, route, steps: bool, extend: bool, multi: bool = False):
        self.split_kv, self.route = split_kv, route
        self.steps, self.extend, self.multi = steps, extend, multi
        self._logged: set = set()

    @classmethod
    def build(cls, parts):
        steps = "draft_q1_cp" in parts.slots
        extend = "draft_skv2_cp" in parts.slots
        multi = "draft_multi" in parts.slots
        if not (steps or extend or multi):
            return None
        from sgl_nvidia_mega.qwen3_5.attention import route, split_kv

        why = split_kv.cp_reason()
        if why:
            if multi and not (steps or extend):
                return None
            raise RuntimeError(
                f"NVIDIA megakernel: slots draft_q1_cp / draft_skv2_cp cannot run: {why}"
            )
        return cls(split_kv, route, steps, extend, multi)

    def takes(self, rows_per_request: int, rows: int) -> bool:
        """Whether this graph lets CP-lite try the decode."""
        if self.multi and rows > rows_per_request:
            return True
        label = _q1_label()
        if rows_per_request == 1:
            return self.steps and label == Q1_CP
        return self.extend and label == self.route.SPLIT_KV_CP

    def run(self, attention, q8, meta, kv, rows_per_request: int):
        layer = attention.layer
        attn = layer.attn
        multi = q8.shape[0] > rows_per_request
        o = self.split_kv.cp_core(
            attn,
            q8,
            kv,
            meta,
            attention,
            rows_per_request,
            min_bs=None if multi else 1,
            # R11 orders the draft-multi exact partial and merge after their waits.
            # Keep the shared CP-lite default for target and one-request draft calls.
            early_merge=False if multi else None,
        )
        if o is not None and torch.cuda.is_current_stream_capturing():
            key = (rows_per_request, q8.shape[0])
            if key not in self._logged:
                self._logged.add(key)
                if multi:
                    logger.info(
                        "NVIDIA megakernel: Qwen3.5 MTP draft slot draft_multi keeps "
                        "CP-lite for %d requests x %d rows",
                        q8.shape[0] // rows_per_request,
                        rows_per_request,
                    )
                else:
                    logger.info(
                        "NVIDIA megakernel: Qwen3.5 MTP draft slot %s runs the %s's "
                        "%d rows as CP-lite (from %d tokens of KV)",
                        "draft_q1_cp" if rows_per_request == 1 else "draft_skv2_cp",
                        "step" if rows_per_request == 1 else "extend",
                        q8.shape[0],
                        self.route.cp_min_length(
                            "draft" if rows_per_request == 1 else "extend"
                        ),
                    )
        return o


def _q1_label():
    """The KV-length graph variant SGLang is capturing ("q1_stock" / "q1_long"), or
    None where its draft runner captures one graph per batch size (then Q1 serves
    every length; 2.2 switched at 16384 by graph variant)."""
    from sglang.srt.model_executor.runner_utils.capture_mode import (
        get_capture_attention_variant,
    )

    return get_capture_attention_variant()


def build_q1(parts):
    """Slot draft_q1: the draft steps' q1 decode as 2.2's SKV2-Q1 (``sm103/draft/q1``),
    a callable (q8, kv, meta) -> [1, heads, head_dim] bf16; each layout's producer
    compiles at the graph warmup's first call, outside capture, as in 2.2."""
    from sgl_nvidia_mega.qwen3_5.sm103.draft import q1

    attention, device = parts.attention, parts.device
    why = q1.enabled_reason(
        device_index=device.index, max_context_len=attention.max_context_len
    )
    if why is not None:
        raise RuntimeError(f"NVIDIA megakernel: slot draft_q1 cannot run: {why}")
    release = GATE_FOLD == 2
    q1.build(device.index, heads=attention.heads, pdl=Q1_PDL, release=release)
    decode = q1.Q1Decode(pdl=Q1_PDL, release=release)
    if Q1_PDL:
        logger.info(
            "NVIDIA megakernel: Qwen3.5 MTP draft slot draft_q1 launches the q1 decode "
            "and its merge as programmatic dependents"
        )
    attn = parts.layer.attn

    def run(q8, kv, meta):
        return decode.step(q8, kv, attn.layer_id, meta, scaling=attn.scaling)

    return run
