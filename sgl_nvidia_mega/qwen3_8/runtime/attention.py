# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/qwen4_exp.py (Qwen4ExpAttentionDecoderLayer.self_attention).
"""The QSA attention layer step of a covered forward.

Slot ``qsa_pool``: a TP1 target verify of up to 64 rows (16 requests x 4) on a BF16 KV
pool runs the sparse attention as the plugin's split-K pool kernels
(``qsa/pool_decode.py``): the new K/V write at the step's slots, the split-K partials
over the selected blocks (and the request's own uncompressed tail) and the merge, which
also applies the sigmoid output gate. With SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_ATTN_CORE_V1=1
(the source branch's tcgen05-core switch), a one-request verify (4 rows) in a CUDA graph
runs the source's tcgen05 core instead (``kernels/ops/attention/qsa_tcgen05_v1.py``): the
new K/V store, the 33 fixed 64-token tile partials and their merge, which applies the
output gate where the source hands it over
(SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_ATTN_GATE_V2 or SGLANG_ENABLE_QSA_POOL_MERGE_GATE).

Slot ``qsa_pool_fp8``: a TP1 target verify of 56-576 rows (14-144 requests x 4) on an FP8
E4M3 KV pool runs it as the union kernels (``qsa/pool_union_ws.py``): the new BF16 K/V
cast to E4M3 in the pool write, each request's union of selected tokens, the
warp-specialized attention over it (launched with PDL) and the split merge; SGLang's
output gate follows. It replaces SGLang's K/V pack and TRT-LLM decode. The slot also
runs the MTP draft's CUDA-graph draft extend of 56-576 rows the same way: four rows a
request in position order, as a verify lays them out, with the blocks read back from
the indexer's expanded token indices. It replaces the K/V store, ``_compact_kv`` pack
and TRT-LLM decode there.

The QKV / gate projection with its norms and RoPE, the indexer (selection) and o_proj
stay SGLang's (with BF16 GEMM pins on, the projection's CUDA branch runs here so
``qkv_proj`` and ``o_proj`` can take a pin: ``bf16_gemm_pins``; with
SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_QSA_PDL_FIX=1, the source branch's QSA PDL-fix switch, a
4-row step's QK norm is the source's PDL launch). As in the source's
``self_attention``, under CUDA graph capture (not a breakable graph) below 1024 rows the
indexer runs on the layer's side stream beside the QKV / gate projection, joined before
the attention. Every other step runs SGLang's ``self_attention``. Numerics: both
reorder the FP32 softmax reduction against TRT-LLM's sparse decode (the union kernel
also rescales lazily), tolerance slots; the E4M3 pool bytes are torch's cast.

Reads: the layer's K/V pool, ``req_to_token``, the backend's step metadata (row
requests, logical positions). Writes: the new K/V at ``out_cache_loc``."""

from __future__ import annotations

import logging
import os

import torch

from sgl_nvidia_mega.qwen3_8.runtime import bf16_gemm_pins

logger = logging.getLogger(__name__)

STOCK = "stock"
# The engagement op naming the stream the indexer ran on, recorded off the main stream
INDEX_STREAM = "qsa.index_stream"


def _env_bool(name: str) -> bool:
    """SGLang's ``get_bool_env_var(name)`` at its default, as the source reads it."""
    return os.environ.get(name, "false").lower() in ("true", "1")


# The source branch's switches of its QSA attention, under this plugin's names, parsed
# with the source's rules and defaults.
# SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_QSA_EARLY_INDEXER (self_attention): the side-stream
# indexer is enqueued before the QKV
QSA_EARLY_INDEXER = _env_bool("SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_QSA_EARLY_INDEXER")
# SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_QSA_PDL_INDEX (self_attention): a 1-16 row verify /
# draft extend projects the index Q/K on the main stream before the QKV; only the rest
# of the indexer forks to the side stream
QSA_PDL_INDEX = _env_bool("SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_QSA_PDL_INDEX")
# The index projection's (N, K) the source's PDL_INDEX branch takes
INDEX_QK_SHAPE = (640, 2560)
# SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_ATTN_CORE_V1 (qwen_sparse_attn_backend.py, exactly
# "1"): a one-request verify on a BF16 pool runs the tcgen05 core before the pool kernels
# (slot qsa_pool)
ATTN_CORE_V1 = (
    os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_ATTN_CORE_V1", "0") == "1"
)
# SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_ATTN_GATE_V2 (a 4-row verify) or
# SGLANG_ENABLE_QSA_POOL_MERGE_GATE (any verify): the source hands the output gate to the
# attention, whose tcgen05 core merge applies it
ATTN_GATE_V2 = _env_bool("SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_ATTN_GATE_V2")
POOL_MERGE_GATE = _env_bool("SGLANG_ENABLE_QSA_POOL_MERGE_GATE")
# SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_QSA_PDL_FIX (exactly "1"): a 4-row QK norm of 24 / 2
# heads launches with PDL and waits for the QKV in-kernel; the core's gate epilogue waits
# for its partials
QSA_PDL_FIX = os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_QSA_PDL_FIX", "0") == "1"
# Test only: on eager steps, also run SGLang's attention on the same q/k/v and selected
# indices and log how far the pool kernel's output is from it, per row.
OPCHECK = os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_OPCHECK") == "1"
MAX_ROWS = 64
QUERY_HEADS, KV_HEADS, HEAD_DIM, BLOCKS, RATIO = 24, 2, 256, 512, 4
# Slot qsa_pool_fp8's verify rows; other widths run SGLang's pack + TRT-LLM decode. On SM107 the union
# beat it at 56-64 rows at every context and selection overlap measured, not reliably below
UNION_MIN_ROWS_ENV = "SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_QSA_UNION_MIN_ROWS"
UNION_MIN_ROWS = int(os.environ.get(UNION_MIN_ROWS_ENV) or 56)
UNION_MAX_ROWS = 576
# Splits of a request's token list in the union attention (1-16)
UNION_SPLITS_ENV = "SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_QSA_UNION_SPLITS"
UNION_SPLITS = int(os.environ.get(UNION_SPLITS_ENV) or 2)
# Warps of the token-union kernel (its output does not depend on them)
UNION_WARPS = 16

SGLANG_READS = {
    "srt/models/qwen4_exp.py": (
        "_compute_qsa_topk_indices",
        "self_attention",
        "is_qsa",
        "indexer",
        "_QSA_INDEXER_OVERLAP_TOKEN_THRESHOLD",
        "_shards_rows_over_cp",
        "fused_sigmoid_mul",
        "get_attn_backend",
        "get_is_capture_mode",
        "is_in_breakable_cuda_graph",
    ),
    "srt/models/qwen3_5.py": (
        "_prepare_qkv_gate",
        "o_proj",
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
        "alt_stream",
        "attn",
    ),
    "kernels/ops/attention/fused_qk_rmsnorm_rope_gate.py": (
        "fused_qk_gemma_rmsnorm_rope_gate",
    ),
    "srt/model_executor/runner/__init__.py": ("get_is_capture_mode",),
    "srt/model_executor/runner_backend_utils/breakable_cuda_graph/__init__.py": (
        "is_in_breakable_cuda_graph",
    ),
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "kernels/ops/elementwise/elementwise.py": ("fused_sigmoid_mul",),
    "srt/layers/attention/qsa/glue.py": ("resolve_qsa_sparse_backend",),
    "srt/layers/attention/qwen_sparse_attn_backend.py": (
        "_resolve_metadata",
        "_uses_block_indices",
        "_fused_kv_pool_eligible",
        "compress_ratio",
        "qsa_profile",
        "row_req_pool_indices",
        "indexer_metadata",
        "decode_logical_positions",
        "req_to_token_pool",
        "token_to_kv_pool",
        "prefetch_cp_kv",
    ),
    "srt/layers/attention/qsa/config.py": ("budget", "block_topk"),
    "srt/layers/attention/qsa/qsa_indexer.py": ("_use_fused_prep", "index_qk_proj"),
    "srt/layers/quantization/unquant.py": (
        "UnquantizedLinearMethod",
        "get_bf16_gemm_backend",
        "is_cutedsl",
        "use_bf16_splitk_gemm",
    ),
    "srt/layers/linear.py": ("quant_method", "weight"),
    "srt/speculative/eagle_info.py": ("topk",),
    "srt/layers/attention/qsa/metadata.py": ("is_cuda_graph",),
    "srt/mem_cache/kv/router.py": ("layer", "read", "key", "value"),
}

# With BF16 pins on, _self_attention mirrors self_attention and _qkv_gate this branch
# of _prepare_qkv_gate
SGLANG_MIRRORS = {
    "srt/models/qwen4_exp.py:Qwen4ExpAttentionDecoderLayer.self_attention": "e88bd51bae6665bc",
    "srt/models/qwen3_5.py:Qwen3_5AttentionDecoderLayer._prepare_qkv_gate": "1af0524597d94935",
    "srt/models/qwen3_5.py:Qwen3_5AttentionDecoderLayer.forward_prepare_cuda_fused": "d10e23f5651990d0",
}


def layer_reasons(layer) -> str:
    attn = layer.attn
    if not layer.is_qsa:
        return "not a QSA layer"
    if (attn.tp_q_head_num, attn.tp_k_head_num, attn.head_dim) != (
        QUERY_HEADS,
        KV_HEADS,
        HEAD_DIM,
    ):
        return (
            f"QSA heads {(attn.tp_q_head_num, attn.tp_k_head_num, attn.head_dim)}, "
            f"the kernel takes {(QUERY_HEADS, KV_HEADS, HEAD_DIM)} (TP1)"
        )
    return ""


def build() -> str:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return "the pool kernels need an SM10x GPU"
    try:
        from sgl_nvidia_mega.qwen3_8.qsa import pool_decode  # noqa: F401
    except Exception as e:
        return f"pool kernels do not import: {type(e).__name__}: {e}"
    return ""


def core_build() -> str:
    """The tcgen05 core's build reason ('' when it can run here)."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return "the tcgen05 core needs an SM10x GPU"
    try:
        from sgl_nvidia_mega.qwen3_8.kernels.ops.attention import (  # noqa: F401
            qsa_tcgen05_v1,
        )
    except Exception as e:
        return f"the tcgen05 core does not import: {type(e).__name__}: {e}"
    return ""


def union_build() -> str:
    """Slot qsa_pool_fp8's build reason ('' when its kernels can run here)."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return "the union kernels need an SM10x GPU"
    if not 1 <= UNION_SPLITS <= 16:
        return f"{UNION_SPLITS_ENV}={UNION_SPLITS}, the merge takes 1-16 splits"
    try:
        from sgl_nvidia_mega.qwen3_8.qsa import pool_union_ws
    except Exception as e:
        return f"union kernels do not import: {type(e).__name__}: {e}"
    if not pool_union_ws.HAS_GLUON:
        return "the warp-specialized union kernel needs Triton's Gluon Blackwell API"
    return ""


def union_pool_reasons(backend, layer_id: int) -> str:
    """'' when the layer's KV pool is what the union kernels read: SGLang's plain FP8
    E4M3 token pool (no k/v scale) of contiguous [slots, 2, 256] at compress ratio 4,
    with the BF16 -> E4M3 pool write equal to torch's cast. Eager only: the cast check
    launches kernels once per device."""
    from sgl_nvidia_mega.qwen3_8.qsa import pool_decode

    if not backend._fused_kv_pool_eligible:
        return "not SGLang's plain token KV pool (its fused K/V path declines it too)"
    if backend.compress_ratio != RATIO:
        return f"QSA compress ratio {backend.compress_ratio}, the kernels take {RATIO}"
    kv = backend.token_to_kv_pool.layer_router.layer(layer_id)
    k, v = kv.key(), kv.value()
    if k.dtype != torch.float8_e4m3fn:
        return f"KV pool dtype {k.dtype}, the union kernels read FP8 E4M3"
    if (
        k.dim() != 3
        or tuple(k.shape[1:]) != (KV_HEADS, HEAD_DIM)
        or v.shape != k.shape
        or v.dtype != k.dtype
        or not (k.is_contiguous() and v.is_contiguous())
    ):
        return f"KV pool {tuple(k.shape)}, the kernels take contiguous [slots, 2, 256]"
    if not pool_decode.write_new_kv_e4m3_ok(k, v):
        return (
            "the BF16 -> E4M3 pool write differs from torch's cast (or was not checked)"
        )
    return ""


def sparse_backend():
    """The attention backend that owns the QSA layers' pool and step metadata."""
    from sglang.srt.layers.attention.qsa.glue import resolve_qsa_sparse_backend
    from sglang.srt.model_executor.forward_context import get_attn_backend

    return resolve_qsa_sparse_backend(get_attn_backend())


def step(executor, layer, x, st):
    """The layer's attention output [rows, hidden] (before the stage's reduction).

    With slot ``qsa_indexer``, ``qsa_pool``, ``qsa_pool_fp8`` or ``qsa_draft``, a
    decode / verify / draft-extend step runs the attention split into its pieces here:
    SGLang's QKV + gate projection, the indexer (``qsa_indexer.select``: the plugin's
    fold, scorers and selects where they serve the step; a draft decode's lookup inside
    ``qsa_draft``'s pack), the sparse attention (the pool, union or lookup kernels where
    they serve it) and the gate and o_proj. Otherwise SGLang's ``self_attention`` runs
    whole."""
    fb = st.fb
    mode = fb.forward_mode
    if (
        (
            executor.qsa_indexer is not None
            or executor.qsa_pool_ready
            or executor.qsa_union is not None
            or executor.qsa_draft is not None
        )
        and layer.is_qsa
        and (mode.is_target_verify() or mode.is_decode() or mode.is_draft_extend_v2())
    ):
        # only the draft plans a draft extend: slot qsa_draft or qsa_pool_fp8 is on
        if mode.is_draft_extend_v2() and executor.qsa_indexer is None:
            return _draft_extend(executor, layer, x, st)
        return _decomposed(executor, layer, x, st)
    if executor.qsa_prefill is not None and layer.is_qsa and st.step.mode == "prefill":
        from sgl_nvidia_mega.qwen3_8.runtime import qsa_prefill

        out = qsa_prefill.extend(executor, layer, x, st)
        if out is not None:
            return out
    st.ran("qsa", STOCK)
    if bf16_gemm_pins.routed():
        return _self_attention(layer, x, st.positions, fb)
    return layer.self_attention(
        positions=st.positions, hidden_states=x, forward_batch=fb
    )


def _self_attention(layer, hidden_states, positions, forward_batch):
    """``layer.self_attention`` with ``qkv_proj`` and ``o_proj`` through
    ``bf16_gemm_pins`` (the stock QSA step with BF16 pins on)."""
    from sglang.srt.models.qwen4_exp import (
        _QSA_INDEXER_OVERLAP_TOKEN_THRESHOLD,
        _shards_rows_over_cp,
        fused_sigmoid_mul,
        get_attn_backend,
        get_is_capture_mode,
        is_in_breakable_cuda_graph,
    )

    overlap_indexer = (
        layer.is_qsa
        and layer.alt_stream is not None
        and get_is_capture_mode()
        and not is_in_breakable_cuda_graph()
        and hidden_states.shape[0] < _QSA_INDEXER_OVERLAP_TOKEN_THRESHOLD
    )
    attention_kwargs = {}
    if overlap_indexer:
        current_stream = torch.cuda.current_stream()
        layer.alt_stream.wait_stream(current_stream)

    q, k, v, gate = _qkv_gate(layer, hidden_states, positions, forward_batch)

    if overlap_indexer:
        with torch.cuda.stream(layer.alt_stream):
            topk_indices = layer._compute_qsa_topk_indices(
                hidden_states, positions, forward_batch
            )
        current_stream.wait_stream(layer.alt_stream)
        topk_indices.record_stream(current_stream)
        attention_kwargs["topk_indices"] = topk_indices
    elif layer.is_qsa:
        if _shards_rows_over_cp(forward_batch) and not get_is_capture_mode():
            from sglang.srt.layers.attention.qsa.glue import resolve_qsa_sparse_backend

            resolve_qsa_sparse_backend(get_attn_backend()).prefetch_cp_kv(
                layer.attn, k, v, forward_batch
            )
        attention_kwargs["topk_indices"] = layer._compute_qsa_topk_indices(
            hidden_states, positions, forward_batch
        )

    attn_output = layer.attn(q, k, v, forward_batch, **attention_kwargs)
    if gate is not None:
        if attn_output.is_cuda:
            attn_output = fused_sigmoid_mul(attn_output, gate, inplace=True)
        else:
            gate = gate.reshape(gate.shape[0], -1) if gate.ndim == 3 else gate
            attn_output = attn_output * torch.sigmoid(gate)
    return bf16_gemm_pins.linear(layer.o_proj, attn_output)


def _qkv_gate(layer, x, positions, fb):
    """``layer._prepare_qkv_gate``; with BF16 pins on, or where
    SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_QSA_PDL_FIX changes the QK norm's launch, its CUDA
    gated branch (``forward_prepare_cuda_fused``) here, so ``qkv_proj`` can take a pin and
    the QK norm is the source's launch."""
    pdl_fix = (
        QSA_PDL_FIX
        and x.shape[0] == 4
        and (layer.num_heads, layer.num_kv_heads) == (QUERY_HEADS, KV_HEADS)
    )
    if not (
        (bf16_gemm_pins.routed() or pdl_fix) and x.is_cuda and layer.attn_output_gate
    ):
        return layer._prepare_qkv_gate(
            positions=positions, hidden_states=x, forward_batch=fb
        )
    kwargs = {}
    if pdl_fix:  # SGLang's launcher has no PDL wait
        from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
            fused_qk_gemma_rmsnorm_rope_gate,
        )

        kwargs["qsa_pdl_fix"] = True
    else:
        from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
            fused_qk_gemma_rmsnorm_rope_gate,
        )

    qkv = bf16_gemm_pins.linear(layer.qkv_proj, x)
    q_gate, k, v = qkv.split([layer.q_size * 2, layer.kv_size, layer.kv_size], dim=-1)
    rope = layer.rotary_emb
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
        layer.head_dim,
        rope.rotary_dim,
        has_gate=True,
        mrope_axis_map=(rope.axis_map if positions.dim() == 2 else None),
        **kwargs,
    )
    rows = x.shape[0]
    return q_out.view(rows, -1), k_out.view(rows, -1), v, gate_out.view(rows, -1)


def _core_eligible(executor, fb, rows) -> bool:
    return (
        executor.qsa_core is not None
        and fb.forward_mode.is_target_verify()
        and rows == 4
        and int(fb.spec_info.draft_token_num) == 4
        and int(getattr(fb.spec_info, "topk", 1) or 1) == 1
    )


def _pool_eligible(executor, fb, rows) -> bool:
    return (
        executor.qsa_pool_ready
        and fb.forward_mode.is_target_verify()
        and rows <= MAX_ROWS
        and int(fb.spec_info.draft_token_num) == 4
    )


def _union_eligible(executor, fb, rows) -> bool:
    return (
        executor.qsa_union is not None
        and fb.forward_mode.is_target_verify()
        and UNION_MIN_ROWS <= rows <= UNION_MAX_ROWS
        and int(fb.spec_info.draft_token_num) == 4
    )


def _draft_union_eligible(executor, fb, rows) -> bool:
    return (
        executor.qsa_union is not None
        and fb.forward_mode.is_draft_extend_v2()
        and UNION_MIN_ROWS <= rows <= UNION_MAX_ROWS
    )


def _overlap_indexer(layer, x) -> bool:
    """``self_attention``'s ``overlap_indexer``: under CUDA graph capture (not a
    breakable graph), below 1024 rows, the indexer runs on the layer's side stream."""
    if not layer.is_qsa or layer.alt_stream is None:
        return False
    from sglang.srt.model_executor.runner import get_is_capture_mode
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        is_in_breakable_cuda_graph,
    )
    from sglang.srt.models.qwen4_exp import _QSA_INDEXER_OVERLAP_TOKEN_THRESHOLD

    return (
        get_is_capture_mode()
        and not is_in_breakable_cuda_graph()
        and x.shape[0] < _QSA_INDEXER_OVERLAP_TOKEN_THRESHOLD
    )


def _pdl_index_projection(executor, layer, x, positions, fb):
    """The PDL-index branch of the source's ``self_attention``
    (SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_QSA_PDL_INDEX): (slot qsa_indexer's step, the step's
    rows of ``index_qk_proj`` launched here on the current stream) where the
    projection is the native BF16 split-K of 1-16 rows, else (None, None). The step
    hands the projection to the plugin's indexer (SGLang's takes none)."""
    mode = fb.forward_mode
    kernels = executor.qsa_indexer
    if not (
        QSA_PDL_INDEX
        and kernels is not None
        and layer.attn_output_gate
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and (mode.is_target_verify() or mode.is_draft_extend_v2())
    ):
        return None, None
    step = kernels.prepare(layer, x, positions, fb)
    if step is None:
        return None, None
    from sglang.srt.layers.quantization.unquant import (
        UnquantizedLinearMethod,
        get_bf16_gemm_backend,
        use_bf16_splitk_gemm,
    )

    rows = step.rows
    projection = layer.indexer.index_qk_proj
    if not (
        0 < rows <= 16
        and rows <= x.shape[0]
        and layer.indexer._use_fused_prep(x)
        and isinstance(projection.quant_method, UnquantizedLinearMethod)
        and projection.weight.dtype == torch.bfloat16
        and tuple(projection.weight.shape) == INDEX_QK_SHAPE
        # SGLang enables its BF16 split-K with the CuTe DSL backend
        and get_bf16_gemm_backend().is_cutedsl()
        and use_bf16_splitk_gemm(rows, *INDEX_QK_SHAPE)
    ):
        return None, None
    return step, bf16_gemm_pins.linear(projection, x[:rows])


def _qkv_and_indices(executor, layer, x, st, indices):
    """``self_attention`` up to its attention call, in its order and streams: (q, k, v,
    gate, top-k indices), ``indices(step, projected_qk)`` running the indexer. Where the
    source overlaps it, the indexer runs on the layer's side stream beside the QKV /
    gate projection and the current stream joins it before the attention; with
    SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_QSA_PDL_INDEX the index projection stays on the
    current stream before the QKV and the side stream depends on its completion only."""
    fb, positions = st.fb, st.positions
    overlap = _overlap_indexer(layer, x)
    early = (
        overlap
        and QSA_EARLY_INDEXER
        and layer.attn_output_gate
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and layer.indexer._use_fused_prep(x)
    )
    step = projected_qk = nodes = None
    if overlap:
        # the indexer reads only hidden_states / positions and writes QSA-private buffers
        current = torch.cuda.current_stream()
        step, projected_qk = _pdl_index_projection(executor, layer, x, positions, fb)
        if projected_qk is not None:
            from sgl_nvidia_mega.qwen3_8.runtime import capture_fork

            nodes = capture_fork.projection_frontier(current)
        else:
            layer.alt_stream.wait_stream(current)
        if early and projected_qk is None:
            with torch.cuda.stream(layer.alt_stream):
                topk = indices()
    q, k, v, gate = _qkv_gate(layer, x, positions, fb)
    if not overlap:
        return q, k, v, gate, indices()
    if projected_qk is not None:
        # No event between the projection and the QKV keeps their PDL chain. Outside a
        # capture (no frontier) the side stream waits for the whole QKV instead.
        layer.alt_stream.wait_stream(current)
        if nodes is not None:
            capture_fork.fork_after_projection(layer.alt_stream, nodes)
        with torch.cuda.stream(layer.alt_stream):
            topk = indices(step, projected_qk)
        projected_qk.record_stream(layer.alt_stream)
        impl = "pdl_index"
    else:
        if not early:
            with torch.cuda.stream(layer.alt_stream):
                topk = indices()
        impl = "side_early" if early else "side"
    current.wait_stream(layer.alt_stream)
    # allocated on the side stream, read by the attention on this one
    topk.record_stream(current)
    st.ran(INDEX_STREAM, impl)
    return q, k, v, gate, topk


def _indices(executor, layer, x, positions, st, step=None, projected_qk=None):
    """``layer._compute_qsa_topk_indices``: slot qsa_indexer's ``select`` where it is
    on (with the PDL-index branch's step and projection), else SGLang's."""
    if executor.qsa_indexer is not None:
        from sgl_nvidia_mega.qwen3_8.runtime import qsa_indexer

        return qsa_indexer.select(executor, layer, x, positions, st, step, projected_qk)
    st.ran("qsa.indexer", STOCK)
    return layer._compute_qsa_topk_indices(x, positions, st.fb)


def _decomposed(executor, layer, x, st):
    """``self_attention`` from its pieces, the indexer in its streams
    (``_qkv_and_indices``). The BF16 pool's split-K kernels take a verify first; an
    FP8 pool's verify or draft extend goes to the union kernels; a draft decode that
    reuses the draft extend's selection goes to slot qsa_draft's lookup pack."""
    fb, positions = st.fb, st.positions
    if executor.qsa_draft is not None and fb.forward_mode.is_decode():
        # the pack looks the selection up itself; where it declines, SGLang's lookup
        q, k, v, gate = _qkv_gate(layer, x, positions, fb)
        out = executor.qsa_draft.decode(layer, sparse_backend(), q, k, v, fb)
        if out is not None:
            st.ran("qsa.indexer", "mtp_lookup_in_pack")
            return _finish(executor, layer, out, gate, st, "fused_kv_lookup")
        indices = _indices(executor, layer, x, positions, st)
    else:
        q, k, v, gate, indices = _qkv_and_indices(
            executor,
            layer,
            x,
            st,
            lambda step=None, qk=None: _indices(
                executor, layer, x, positions, st, step, qk
            ),
        )
    rows = q.shape[0]
    out = None
    if _core_eligible(executor, fb, rows):
        out = _verify_core(
            executor, layer, sparse_backend(), q, k, v, gate, indices, st
        )
    if out is None and _pool_eligible(executor, fb, rows):
        out = _verify_pool(
            executor, layer, sparse_backend(), q, k, v, gate, indices, st
        )
    if out is None and _union_eligible(executor, fb, rows):
        out = _verify_union(
            executor, layer, sparse_backend(), q, k, v, gate, indices, st
        )
    if out is None and _draft_union_eligible(executor, fb, rows):
        out = _draft_union(
            executor, layer, sparse_backend(), q, k, v, gate, indices, st
        )
    if out is not None:
        return out
    attn_out = layer.attn(q, k, v, fb, topk_indices=indices)
    return _finish(executor, layer, attn_out, gate, st, STOCK)


def _draft_extend(executor, layer, x, st):
    """``self_attention`` of a draft extend with no plugin indexer: SGLang's indexer with
    the seed capture of ``qsa_draft.indices``, in the source's order and streams
    (``_qkv_and_indices``), then the union kernels where they serve it."""
    from sgl_nvidia_mega.qwen3_8.runtime import qsa_draft

    fb, positions = st.fb, st.positions
    q, k, v, gate, indices = _qkv_and_indices(
        executor,
        layer,
        x,
        st,
        lambda: qsa_draft.indices(executor, layer, x, positions, st),
    )
    st.ran("qsa.indexer", STOCK)
    if _draft_union_eligible(executor, fb, q.shape[0]):
        out = _draft_union(
            executor, layer, sparse_backend(), q, k, v, gate, indices, st
        )
        if out is not None:
            return out
    attn_out = layer.attn(q, k, v, fb, topk_indices=indices)
    return _finish(executor, layer, attn_out, gate, st, STOCK)


def _verify_inputs(backend, fb, indices, rows):
    """(step metadata, the rows' logical positions) for the pool kernels, or None when
    this step's layout is not theirs (4-row requests selecting 512 blocks a row)."""
    md = backend._resolve_metadata(fb)
    indexer = md.indexer_metadata
    query_positions = None if indexer is None else indexer.decode_logical_positions
    if (
        not backend._uses_block_indices(indices)
        or tuple(indices.shape) != (rows, BLOCKS)
        or query_positions is None
        or md.row_req_pool_indices is None
        or rows != fb.req_pool_indices.numel() * 4
    ):
        return None
    return md, query_positions


def _verify_pool(executor, layer, backend, q, k, v, gate, indices, st):
    """The pool kernels on a TP1 verify of up to 64 rows, or None (nothing launched)
    when this step's layout is not theirs."""
    fb = st.fb
    pool = backend.token_to_kv_pool
    if pool.layer_router.layer(layer.attn.layer_id).key().dtype != torch.bfloat16:
        return None
    rows = q.shape[0]
    inputs = _verify_inputs(backend, fb, indices, rows)
    if inputs is None:
        return None
    md, query_positions = inputs
    row_requests = md.row_req_pool_indices
    kv = pool.layer_router.read(layer.attn.layer_id)
    out = executor.qsa_pool.run(
        q=q.reshape(rows, QUERY_HEADS * HEAD_DIM),
        key_cache=kv.key(),
        value_cache=kv.value(),
        new_k=k.reshape(rows, KV_HEADS, HEAD_DIM),
        new_v=v.reshape(rows, KV_HEADS, HEAD_DIM),
        locations=fb.out_cache_loc,
        slot_map=backend.req_to_token_pool.req_to_token,
        row_requests=row_requests,
        indices=indices.to(torch.int32).contiguous(),
        sequence_lengths=md.sequence_lengths,
        query_positions=query_positions,
        gate=gate,
        scale=layer.attn.scaling,
    )
    out = out.view(rows, QUERY_HEADS * HEAD_DIM)
    if OPCHECK and not torch.cuda.is_current_stream_capturing():
        _opcheck(
            layer, q, k, v, fb, indices, gate, out, md, query_positions, "pool_splitk"
        )
    return _finish(executor, layer, out, None, st, "pool_splitk")


def _verify_core(executor, layer, backend, q, k, v, gate, indices, st):
    """The source's tcgen05 core (SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_ATTN_CORE_V1) on a
    one-request TP1 verify in a CUDA graph on a BF16 pool, under its ``_fused_kv_prefix``
    and core gates, or None (nothing launched). Its merge applies the output gate where
    the source passes it (to the source's output-gate function), else SGLang's gate
    follows."""
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

    fb = st.fb
    rows = q.shape[0]
    loc = fb.out_cache_loc
    kv = backend.token_to_kv_pool.layer_router.layer(layer.attn.layer_id)
    kc, vc = kv.key(), kv.value()
    md = backend._resolve_metadata(fb)
    indexer = md.indexer_metadata
    row_requests = md.row_req_pool_indices
    if row_requests is None:
        row_requests = fb.req_pool_indices
    if not (
        backend._fused_kv_pool_eligible
        and backend.compress_ratio == RATIO
        and rows == fb.req_pool_indices.numel() * 4
        and isinstance(loc, torch.Tensor)
        and loc.ndim == 1
        and loc.numel() == rows
        and loc.is_contiguous()
        and k.shape[0] == rows
        and v.shape[0] == rows
        and q.dtype == torch.bfloat16
        and kc.ndim == 3
        and vc.shape == kc.shape
        and kc.is_contiguous()
        and vc.is_contiguous()
        and tuple(kc.shape[1:]) == (KV_HEADS, HEAD_DIM)
        and kc.dtype == vc.dtype == torch.bfloat16
        and md.is_cuda_graph
        and md.sequence_lengths.numel() == rows
        and backend._uses_block_indices(indices)
        and tuple(indices.shape) == (rows, BLOCKS)
        and indexer is not None
        and indexer.decode_logical_positions is not None
        and row_requests.numel() == rows
    ):
        return None
    new_k = k.reshape(rows, KV_HEADS, HEAD_DIM)
    new_v = v.reshape(rows, KV_HEADS, HEAD_DIM)
    if any(t.stride(1) != HEAD_DIM or t.stride(2) != 1 for t in (new_k, new_v)):
        return None
    query_positions = indexer.decode_logical_positions
    gated = gate is not None and (ATTN_GATE_V2 or POOL_MERGE_GATE)
    kv = backend.token_to_kv_pool.layer_router.read(layer.attn.layer_id)
    out = executor.qsa_core.run(
        q=q.reshape(rows, QUERY_HEADS, HEAD_DIM),
        key_cache=kv.key(),
        value_cache=kv.value(),
        new_k=new_k,
        new_v=new_v,
        locations=loc,
        slot_map=backend.req_to_token_pool.req_to_token,
        row_requests=row_requests,
        indices=indices.to(torch.int32).contiguous(),
        sequence_lengths=md.sequence_lengths,
        query_positions=query_positions,
        scale=layer.attn.scaling,
        gate=gate if gated else None,
    )
    out = out.reshape(rows, QUERY_HEADS * HEAD_DIM)
    impl = "tcgen05_gate_v2" if gated else "tcgen05_v1"
    if gate is not None and not gated:
        out = fused_sigmoid_mul(out, gate, inplace=True)
    if OPCHECK and not torch.cuda.is_current_stream_capturing():
        _opcheck(layer, q, k, v, fb, indices, gate, out, md, query_positions, impl)
    return _finish(executor, layer, out, None, st, impl)


def _verify_union(executor, layer, backend, q, k, v, gate, indices, st):
    """The union kernels on a TP1 verify of an FP8 pool (checked at the first verify),
    or None (nothing launched) when this step's layout is not theirs."""
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

    fb = st.fb
    rows = q.shape[0]
    inputs = _verify_inputs(backend, fb, indices, rows)
    if inputs is None or q.dtype != torch.bfloat16:
        return None
    md, query_positions = inputs
    kv = backend.token_to_kv_pool.layer_router.read(layer.attn.layer_id)
    out, impl = executor.qsa_union.run(
        q=q.reshape(rows, QUERY_HEADS, HEAD_DIM),
        key_cache=kv.key(),
        value_cache=kv.value(),
        new_k=k.reshape(rows, KV_HEADS, HEAD_DIM),
        new_v=v.reshape(rows, KV_HEADS, HEAD_DIM),
        locations=fb.out_cache_loc,
        slot_map=backend.req_to_token_pool.req_to_token,
        row_requests=md.row_req_pool_indices,
        indices=indices.to(torch.int32).contiguous(),
        sequence_lengths=md.sequence_lengths,
        query_positions=query_positions,
        scale=layer.attn.scaling,
    )
    out = fused_sigmoid_mul(out, gate, inplace=True)
    if OPCHECK and not torch.cuda.is_current_stream_capturing():
        _opcheck(layer, q, k, v, fb, indices, gate, out, md, query_positions, impl)
    return _finish(executor, layer, out, None, st, impl)


def _draft_union(executor, layer, backend, q, k, v, gate, indices, st):
    """The union kernels on a CUDA-graph draft extend of an FP8 pool (checked at the
    first draft extend), or None (nothing launched) when this step's layout is not
    theirs: four rows a request, expanded token indices [rows, 2051]."""
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

    fb = st.fb
    rows = q.shape[0]
    md = backend._resolve_metadata(fb)
    indexer = md.indexer_metadata
    profile = backend.qsa_profile
    loc = fb.out_cache_loc
    if not (
        md.is_cuda_graph
        and q.dtype == torch.bfloat16
        and profile is not None
        and (profile.block_topk, profile.budget) == (BLOCKS, BLOCKS * RATIO)
        and tuple(indices.shape) == (rows, BLOCKS * RATIO + RATIO - 1)
        and rows == fb.req_pool_indices.numel() * 4
        and indexer is not None
        and indexer.decode_logical_positions is not None
        and md.row_req_pool_indices is not None
        and md.row_req_pool_indices.numel() == rows
        and md.sequence_lengths.numel() == rows
        and isinstance(loc, torch.Tensor)
        and loc.ndim == 1
        and loc.numel() == rows
        and loc.is_contiguous()
    ):
        return None
    # column RATIO * c of the expansion holds block c's first token (or -1)
    blocks = indices[:, : BLOCKS * RATIO : RATIO].div(RATIO, rounding_mode="floor")
    query_positions = indexer.decode_logical_positions
    kv = backend.token_to_kv_pool.layer_router.read(layer.attn.layer_id)
    out, impl = executor.qsa_union.run(
        q=q.reshape(rows, QUERY_HEADS, HEAD_DIM),
        key_cache=kv.key(),
        value_cache=kv.value(),
        new_k=k.reshape(rows, KV_HEADS, HEAD_DIM),
        new_v=v.reshape(rows, KV_HEADS, HEAD_DIM),
        locations=loc,
        slot_map=backend.req_to_token_pool.req_to_token,
        row_requests=md.row_req_pool_indices,
        indices=blocks.to(torch.int32).contiguous(),
        sequence_lengths=md.sequence_lengths,
        query_positions=query_positions,
        scale=layer.attn.scaling,
    )
    out = fused_sigmoid_mul(out, gate, inplace=True)
    impl += "_draft"
    if OPCHECK and not torch.cuda.is_current_stream_capturing():
        _opcheck(layer, q, k, v, fb, indices, gate, out, md, query_positions, impl)
    return _finish(executor, layer, out, None, st, impl)


def _opcheck(layer, q, k, v, fb, indices, gate, got, md, query_positions, impl):
    """Same selection, SGLang's attention (it rewrites the same K/V at the same slots),
    then the gate as SGLang applies it: the remaining difference is the kernel's."""
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

    want = layer.attn(q, k, v, fb, topk_indices=indices)
    want = fused_sigmoid_mul(want.reshape(got.shape).clone(), gate, inplace=True)
    diff = (want.float() - got.float()).abs()
    per_row = diff.amax(dim=1)
    worst = int(per_row.argmax())
    rel = float(
        (want.float() - got.float()).norm() / want.float().norm().clamp_min(1e-30)
    )
    valid = int((indices[worst] >= 0).sum())
    logger.info(
        "NVIDIA megakernel opcheck qsa=%s layer=%d rows=%d max_abs=%.4g rel_l2=%.3g "
        "worst_row=%d worst_row_max=%.4g worst_row_valid_blocks=%d worst_row_seq_len=%d "
        "worst_row_pos=%d rows_over_0.05=%d",
        impl,
        layer.attn.layer_id,
        got.shape[0],
        float(diff.max()),
        rel,
        worst,
        float(per_row[worst]),
        valid,
        int(md.sequence_lengths[worst]),
        int(query_positions[worst]),
        int((per_row > 0.05).sum()),
    )


def _finish(executor, layer, attn_out, gate, st, impl):
    """The output gate (unless the merge applied it) and o_proj (on slot smallm_gemm's
    split-K where it serves)."""
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

    from sgl_nvidia_mega.qwen3_8.runtime import smallm_gemm

    if gate is not None:
        attn_out = fused_sigmoid_mul(attn_out, gate, inplace=True)
    st.ran("qsa", impl)
    output = smallm_gemm.out_proj(executor, layer.o_proj, attn_out, st, "qsa.o_proj")
    if output is None:
        output = bf16_gemm_pins.linear(layer.o_proj, attn_out)
    return output


class PoolKernels:
    """The plugin's TP1 split-K pool attention for 4-row verify requests, with the
    sigmoid output gate in its merge. Hardware config is resolved eagerly (a capture
    warmup runs every width before its graph is captured)."""

    def __init__(self):
        from sgl_nvidia_mega.qwen3_8.qsa import pool_decode as raw

        self.raw = raw
        self._configs: dict = {}

    def run(
        self,
        *,
        q,
        key_cache,
        value_cache,
        new_k,
        new_v,
        locations,
        slot_map,
        row_requests,
        indices,
        sequence_lengths,
        query_positions,
        gate,
        scale,
    ):
        raw = self.raw
        rows, dim = q.shape[0], HEAD_DIM
        device = q.device
        config_key = (device, rows)
        if config_key not in self._configs:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("qsa_pool: config must be resolved before capture")
            self._configs[config_key] = raw.choose_config(
                rows, KV_HEADS, BLOCKS * RATIO + RATIO - 1, raw._num_sms(device)
            )
        splits, chunk, bn = self._configs[config_key]
        gp = 16
        stats = rows * KV_HEADS * splits * gp
        scratch = torch.empty(stats * (dim + 2), dtype=torch.float32, device=device)
        maximum, sums = scratch[:stats], scratch[stats : 2 * stats]
        acc = scratch[2 * stats :]
        out = torch.empty(rows, QUERY_HEADS, dim, dtype=q.dtype, device=device)
        out2 = out.view(rows, QUERY_HEADS * dim)
        nk2, nv2 = (
            new_k.reshape(rows, KV_HEADS * dim),
            new_v.reshape(rows, KV_HEADS * dim),
        )
        scale = float(scale) * raw.LOG2E
        raw._qsa_write_new_kv[(rows, KV_HEADS)](
            key_cache,
            value_cache,
            nk2,
            nv2,
            locations,
            nk2.stride(0),
            nv2.stride(0),
            KV_HEADS,
            dim,
            True,
            num_warps=1,
            launch_pdl=True,
        )
        raw._qsa_pool_splitk[(rows, KV_HEADS, splits)](
            q,
            key_cache,
            value_cache,
            nk2,
            nv2,
            locations,
            slot_map,
            row_requests,
            indices,
            sequence_lengths,
            query_positions,
            maximum,
            sums,
            acc,
            maximum,
            out2,
            out2.stride(0),
            scale,
            q.stride(0),
            slot_map.stride(0),
            nk2.stride(0),
            nv2.stride(0),
            indices.stride(0),
            KV_HEADS,
            dim,
            QUERY_HEADS // KV_HEADS,
            gp,
            BLOCKS,
            RATIO,
            4,
            splits,
            chunk,
            bn,
            0,
            False,
            False,
            True,
            num_warps=4,
            num_stages=4,
            launch_pdl=True,
        )
        gate3 = gate.reshape(rows, QUERY_HEADS, dim) if gate.ndim == 2 else gate
        raw._qsa_pool_merge[(rows * QUERY_HEADS,)](
            maximum,
            sums,
            acc,
            out2,
            out2.stride(0),
            KV_HEADS,
            dim,
            QUERY_HEADS // KV_HEADS,
            gp,
            splits,
            min(16, 1 << (splits - 1).bit_length()),
            True,
            num_warps=2,
            GATE=gate3,
            GATE_STRIDE_ROW=gate3.stride(0),
            GATE_STRIDE_HEAD=gate3.stride(1),
            HAS_GATE=True,
            launch_pdl=True,
        )
        return out


class CoreKernels:
    """The source's TP1 tcgen05 attention of a one-request (4-row) verify on a BF16 pool
    (SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_ATTN_CORE_V1): the new K/V store, the tile partials
    and the merge; with a ``gate``, the gate epilogue merge (``qsa_gate_epilogue_v2``)
    applies it."""

    def run(
        self,
        *,
        q,
        key_cache,
        value_cache,
        new_k,
        new_v,
        locations,
        slot_map,
        row_requests,
        indices,
        sequence_lengths,
        query_positions,
        scale,
        gate=None,
    ):
        """[4, QUERY_HEADS, HEAD_DIM] in q's dtype."""
        from sgl_nvidia_mega.qwen3_8.kernels.ops.attention import (
            qsa_gate_epilogue_v2,
            qsa_tcgen05_v1,
        )

        args = (
            q,
            key_cache,
            value_cache,
            new_k,
            new_v,
            locations,
            slot_map,
            row_requests,
            indices,
            sequence_lengths,
            query_positions,
            scale,
        )
        if gate is None:
            return qsa_tcgen05_v1.run(*args, width=4, store_new_kv=True)
        return qsa_gate_epilogue_v2.run_gated(
            *args, gate, width=4, store_new_kv=True, wait_primary=QSA_PDL_FIX
        )


class UnionKernels:
    """The plugin's TP1 union attention of 4-row requests (a verify or a draft extend)
    on an FP8 E4M3 pool: the new K/V's BF16 -> E4M3 pool write, each request's token
    union, the warp-specialized attention (launched with PDL) and the split merge. The
    output is not gated."""

    def __init__(self):
        from sgl_nvidia_mega.qwen3_8.qsa import pool_union_ws

        self.ws = pool_union_ws

    def run(
        self,
        *,
        q,
        key_cache,
        value_cache,
        new_k,
        new_v,
        locations,
        slot_map,
        row_requests,
        indices,
        sequence_lengths,
        query_positions,
        scale,
    ):
        """[rows, QUERY_HEADS * HEAD_DIM] in q's dtype, and the engagement name of the
        attention kernel (the Triton kernel 2 where the warp-specialized one does not
        serve the shapes)."""
        ws = self.ws
        rows, device = q.shape[0], q.device
        requests = rows // 4
        nb, _, nt = ws.union.union_buffers(rows, 4, BLOCKS, RATIO)
        stats = requests * KV_HEADS * UNION_SPLITS * 64
        ints = torch.empty(
            requests * (2 * nt + nb + 1) + rows, dtype=torch.int32, device=device
        )
        floats = torch.empty(stats * (HEAD_DIM + 2), dtype=torch.float32, device=device)
        # token slots, row masks, sort scratch, per-row then per-request counts: every
        # slice starts 16-byte aligned at any request count
        lists, counts = requests * nt, 2 * requests * nt + requests * nb
        workspace = (
            ints[:lists],
            ints[lists : 2 * lists],
            ints[2 * lists : counts],
            ints[counts + rows :],
            ints[counts : counts + rows],
            floats[:stats],
            floats[stats : 2 * stats],
            floats[2 * stats :],
        )
        out = torch.empty(rows, QUERY_HEADS * HEAD_DIM, dtype=q.dtype, device=device)
        ws.qsa_pool_union_decode_ws(
            q,
            key_cache,
            value_cache,
            new_k,
            new_v,
            locations,
            slot_map,
            row_requests,
            indices,
            sequence_lengths,
            query_positions,
            4,
            scale,
            ratio=RATIO,
            out=out,
            workspace=workspace,
            splits=UNION_SPLITS,
            union_warps=UNION_WARPS,
            pdl=True,
            ws_pdl=True,
            bn=128,
            num_warps=8,
            num_stages=3,
        )
        return out, "union_ws_fp8" if ws.supported(q, key_cache, 4) else "union_fp8"
