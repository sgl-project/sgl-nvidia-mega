# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CuTe DSL kernels in the verify attention of the GLM-5.3 megakernel (slot
``attention``).

``FusedAttention`` serves the ``step.AttentionStep`` steps of one verify
row count: ``qkv_a`` on ``qkv_a_proj_small_m`` at up to 8 rows (on layers with
their own indexer together with the indexer's key / head-gate projection, left in
``step.index_kw``; elsewhere the stock fused-A GEMM, and on indexer layers the
stock key / head-gate projection right after it, which the prologue consumes);
``prologue`` on ``mla_prologue`` at SGLang's rounding points (q_a / kv_a norms,
``q_b_proj``, W_UK absorption, RoPE + FP8 query left in ``step.q_fp8``, the MLA
and index-K cache rows), so ``index_k``, ``q_absorb`` and ``rope_cache`` pass
through; ``index_q`` on ``indexer_q_proj``; ``sparse_mla`` on ``sparse_mla_decode``
over the fused top-k page table (16 warps per CTA up to 6 rows, 8 above);
``uv_proj`` on ``mla_uv_proj``. The indexer's logits and top-k stay stock.
Slot ``qkv_a_small_m_shared``: the layers without their own top-k also run
``qkv_a`` on the small-M kernel, and with slot ``dnorm`` on its epilogue-scaled form,
which consumes the fold hop's deferred input norm.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from sgl_nvidia_mega.glm5_3 import HIDDEN
from sgl_nvidia_mega.glm5_3.attention.step import (
    AttentionStep,
    IndexQ,
    PrologueOut,
    StepContext,
    pad_topk_indices,
    rope_style_reasons,
)
from sgl_nvidia_mega.glm5_3.runtime.compile import compile_cached, cu_stream

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/binding.py": (
        "kv_pool",
        "get_key_buffer",
        "get_index_k_with_scale_buffer",
        "get_index_k_with_scale_buffer(layer_id)",
        "index_page_size",
    ),
    "kernels/ops/kvcache/dsa_fused_store_index_cache.py": ("can_use_dsa_fused_store",),
    "kernels/ops/quantization/fp8_kernel.py": ("is_fp8_fnuz",),
    "srt/layers/layernorm.py": ("LayerNorm", "variance_epsilon"),
    # the replaced model's modules and weights
    "srt/models/deepseek_v2.py": (
        "kv_b_proj",
        "q_b_proj",
        "fused_qkv_a_proj_with_mqa",
        "o_proj",
        "q_a_layernorm",
        "kv_a_layernorm",
        "input_layernorm",
        "rotary_emb",
        "indexer",
        "skip_topk",
        "layer_id",
    ),
    "srt/layers/attention/dsa/dsa_indexer.py": (
        "k_norm",
        "wq_b",
        "wk_weights_proj",
        "n_heads",
        "head_dim",
        "softmax_scale",
        "rotary_emb",
        "_indexer_cos_sin_cache",
    ),
    "srt/layers/rotary_embedding/*.py": ("cos_sin_cache",),
    "srt/layers/radix_attention.py": ("scaling", "k_scale_float"),
}

# One tensor-parallel rank (TP 8).
HEADS, Q_LORA, KV_LORA, ROPE, NOPE, VALUE_DIM = 8, 2048, 512, 64, 192, 256
INDEX_HEADS, INDEX_DIM, TOPK, INDEX_PAGE = 32, 128, 2048, 64
KV_B_ROWS = HEADS * (NOPE + VALUE_DIM)
Q_B_ROWS = HEADS * (NOPE + ROPE)
MQA_DIM = KV_LORA + ROPE
QKV_A_ROWS = Q_LORA + KV_LORA + ROPE
INDEX_KW = INDEX_DIM + INDEX_HEADS
LOG2E = 1.4426950408889634

logger = logging.getLogger(__name__)


def _prologue_numerics():
    """The prologue's numeric options: this engine's stock prologue arithmetic."""
    from sgl_nvidia_mega.glm5_3.sm103.attention.mla_prologue import PrologueNumerics

    return PrologueNumerics(
        norm_order="strided_8x256",
        norm_weight_plus_zero=True,
        kv_norm_bf16=True,
        skip_slot0=True,
        index_mean_sum="sequential",
        index_rope_odd="round_x1_cos",
        index_k_bf16=True,
        index_scale="fp32_amax",
        q_rope="one_product_rounded",
    )


# the RoPE arithmetic of this engine's stock indexer query path
INDEX_Q_ROPE_FORM = "mul_sub"

# L2 prefetch hints (evict_last; no output depends on them) of the value
# up-projection while it waits for the attention core: on the target's layers this
# layer's o_proj weight, which the o_proj hop reads next (of each 48-row slab, the
# rows one hop CTA reads, its first 122,880 B: 15 MiB over the 128 slabs), then the
# first 8 MiB of the next layer's q_a / kv_a weight.
NEXT_QKV_A_PREFETCH_BYTES = 8 << 20
O_PROJ_STAGE_SLAB_BYTES = 48 * 2048 * 2
O_PROJ_STAGE_PREFIX_BYTES = 122_880
EVICT_LAST = 0

Slot = Tuple[int, int, int, int, int]
NO_SLOT: Slot = (0, 0, 0, 0, 0)


def kernel_rows(rows: int) -> Optional[int]:
    """The row capacity (8 or 16) of the kernel forms that serve ``rows``."""
    return 8 if 1 <= rows <= 8 else 16 if rows <= 16 else None


def sparse_warps(rows: int) -> int:
    """Warps per CTA of the sparse MLA form: one CTA per SM up to 6 rows (one request
    of up to 6 verify rows), two above."""
    return 16 if rows <= 6 else 8


def _f32(x: float) -> float:
    return float(np.float32(x))


def _i64(*values) -> list:
    from cutlass import Int64

    return [Int64(v) for v in values]


def _addr(t: Optional[torch.Tensor]) -> Tuple[int, int]:
    """(address, row stride) of ``t``; (0, 0) for None."""
    return (0, 0) if t is None else (t.data_ptr(), t.stride(0))


def _fail(what: str, cls=RuntimeError) -> Exception:
    return cls(f"fused verify attention: {what}")


def _prefetch_slot(w, nbytes: int, stride: int = 0, prefix: int = 0) -> Slot:
    """An evict_last L2 prefetch slot over the first ``nbytes`` of ``w`` (with
    ``stride``: the first ``prefix`` bytes of every ``stride`` bytes), or no slot
    unless ``w`` is a contiguous, 128-byte aligned CUDA tensor."""
    if w is None or not w.is_cuda or not w.is_contiguous() or w.data_ptr() % 128:
        return NO_SLOT
    size = min(nbytes, w.numel() * w.element_size())
    return (w.data_ptr(), size, stride, prefix, EVICT_LAST)


def _o_proj_stage_slot(w: torch.Tensor) -> Slot:
    """The strided staging slot over this layer's o_proj weight, or no slot unless
    it is the expected BF16 [hidden, heads * value_dim] tensor."""
    if w.dtype != torch.bfloat16 or tuple(w.shape) != (HIDDEN, HEADS * VALUE_DIM):
        return NO_SLOT
    return _prefetch_slot(
        w,
        w.numel() * w.element_size(),
        O_PROJ_STAGE_SLAB_BYTES,
        O_PROJ_STAGE_PREFIX_BYTES,
    )


def _slot_args(slot: Slot) -> tuple:
    from cutlass import Int32

    return *_i64(*slot[:4]), Int32(slot[4])


def _bf16_weight(linear, shape, what: str) -> torch.Tensor:
    weight = linear.weight
    if weight.dtype != torch.bfloat16 or tuple(weight.shape) != shape:
        raise ValueError(
            f"fused verify attention needs a bf16 {what} weight of shape {shape} per "
            f"rank, got {weight.dtype} {tuple(weight.shape)}"
        )
    return weight


def _check_rows(t: torch.Tensor, rows: int, cols: int, dtype, what: str) -> None:
    """A [rows, cols] tensor with unit column stride and 16-byte aligned rows."""
    if (
        t.dtype != dtype
        or t.dim() != 2
        or tuple(t.shape) != (rows, cols)
        or t.stride(1) != 1
        or t.data_ptr() % 16
        or (t.stride(0) * t.element_size()) % 16
    ):
        raise _fail(f"unexpected {what} {tuple(t.shape)} / {t.stride()} / {t.dtype}")


def _check_index(t: torch.Tensor, rows: int, what: str) -> None:
    if t.dtype != torch.int64 or t.dim() != 1 or t.numel() < rows or t.stride(0) != 1:
        raise _fail(f"unexpected {what} {tuple(t.shape)} {t.dtype}")


@dataclass
class LayerArgs:
    """Per-layer constants of the attention kernels: device addresses of tensors
    owned by the ``DerivedWeights`` passed to ``AttentionGlue`` (kept alive there),
    and the model-owned norm weights / RoPE caches (live as the model)."""

    w_uv: int
    stage0: Slot
    w_qb: int
    w_uk: int
    q_norm_w: torch.Tensor
    kv_norm_w: torch.Tensor
    q_eps: float
    kv_eps: float
    rope: torch.Tensor
    # layers with their own indexer
    w_qkva: int = 0
    w_iq: int = 0
    ik_w: Optional[torch.Tensor] = None
    ik_b: Optional[torch.Tensor] = None
    ik_eps: float = 0.0
    index_rope: Optional[torch.Tensor] = None
    w_scale: float = 0.0
    stage1: Slot = NO_SLOT  # the value up-projection's second staging slot
    # layers without their own indexer: the packed q_a / kv_a weight of the small-M
    # projection (0: not served) and the input RMSNorm epsilon (deferred input norm)
    w_qkva_s: int = 0
    in_eps: float = 0.0


class FusedAttention(AttentionStep):
    """Verify attention of one row count on the attention-family kernels."""

    def __init__(self, glue: AttentionGlue, rows: int) -> None:
        self.glue, fn = glue, glue.fn
        self.max_rows = cap = kernel_rows(rows)
        self.uv = fn[f"uv.{cap}"]
        self.sparse = fn[f"sparse.{sparse_warps(rows)}"]
        self.pro = fn[f"prologue.{cap}"]
        # the target's layers that reuse the previous top-k: the end-trigger prologue
        self.pro_end = fn.get(f"prologue_end.{cap}")
        self.iq = fn[f"index_q.{cap}"]
        # the small-M q_a / kv_a forms (8 rows only), if built:
        self.qa = fn.get(f"qkv_a.{cap}")  # indexer layers
        self.qa_s = fn.get(f"qkv_a_shared.{cap}")  # shared-top-k layers
        self.qa_s_dn = fn.get(f"qkv_a_shared_dn.{cap}")  # + deferred input norm
        # (slot dnorm; the composition wires its producer when ``takes_deferred_norm``)
        self.takes_deferred_norm = self.qa_s_dn is not None

    def runs_indexer(self, attn, step: StepContext) -> bool:
        """Whether this layer's indexer runs in this forward: the layers with their
        own top-k (a step group of other layers may override it)."""
        return not attn.skip_topk

    def qkv_a(self, attn, x, step: StepContext) -> torch.Tensor:
        dn, step.dnorm = step.dnorm, None
        if not self.runs_indexer(attn, step):
            la = self.glue.layers.get(attn.layer_id)
            if dn is not None:
                # x is bf16(residual * gamma): only the epilogue-scaled projection
                # of this very layer may consume it
                if dn[0] != attn.layer_id or self.qa_s_dn is None or not la.w_qkva_s:
                    raise _fail(
                        f"layer {attn.layer_id} got the deferred input norm of "
                        f"layer {dn[0]}"
                    )
                inv_rms = (dn[1], la.in_eps)
                return self._small_m(self.qa_s_dn, la.w_qkva_s, x, inv_rms)[0]
            if self.qa_s is not None and la is not None and la.w_qkva_s:
                return self._small_m(self.qa_s, la.w_qkva_s, x)[0]
            # layers that reuse the previous indexer keep the stock fused-A GEMM
            return super().qkv_a(attn, x, step)
        if dn is not None:
            raise _fail(f"indexer layer {attn.layer_id} got a deferred input norm")
        if self.qa is None:
            qkv = super().qkv_a(attn, x, step)
            step.index_kw, _ = attn.indexer.wk_weights_proj(x)
            return qkv
        w = self.glue.layers[attn.layer_id].w_qkva
        qkv, step.index_kw = self._small_m(self.qa, w, x, index_kw=True)
        return qkv

    def _small_m(self, fn, w: int, x, dn=None, index_kw=False):
        """``(qkv, kw)``: q_a / kv_a of ``x`` over the packed weight at ``w``, and
        with ``index_kw`` the indexer key / head-gate projection (else None); with
        ``dn`` (block sums, eps) ``x`` is ``bf16(residual * gamma)`` and the
        epilogue applies the row's inverse RMS (deferred input norm)."""
        from cutlass import Float32, Int32

        rows = x.shape[0]
        _check_rows(x, rows, HIDDEN, torch.bfloat16, "attention input")
        qkv = torch.empty((rows, QKV_A_ROWS), dtype=torch.bfloat16, device=x.device)
        kw = None
        if index_kw:
            kw = torch.empty((rows, INDEX_KW), dtype=torch.bfloat16, device=x.device)
        tail = ()
        if dn is not None:
            _check_rows(dn[0], rows, 8, torch.float32, "deferred-norm block sums")
            tail = (*_i64(dn[0].data_ptr()), Float32(dn[1]))  # bsum, eps
        fn(
            *_i64(w),  # weights
            *_i64(*_addr(x)),  # x, x_stride
            Int32(rows),
            *_i64(*_addr(qkv)),  # out_main, out_main_stride
            *_i64(*_addr(kw)),  # out_aux, out_aux_stride (0, 0: none)
            *tail,
            cu_stream(x.device),
        )
        return qkv, kw

    def prologue(
        self, attn, qkv_latent: torch.Tensor, step: StepContext
    ) -> PrologueOut:
        from cutlass import Float32, Int32

        rows = qkv_latent.shape[0]
        _check_rows(qkv_latent, rows, QKV_A_ROWS, torch.bfloat16, "q_a / kv_a latent")
        full = self.runs_indexer(attn, step)
        dev = qkv_latent.device
        la = self.glue.layers[attn.layer_id]
        loc = step.forward_batch.out_cache_loc
        pos = step.positions
        _check_index(loc, rows, "cache slots")
        _check_index(pos, rows, "positions")
        pool = step.view.kv_pool
        mla = pool.get_key_buffer(attn.layer_id)
        if (
            mla.dim() != 3
            or tuple(mla.shape[1:]) != (1, MQA_DIM)
            or mla.element_size() != 1
        ):
            raise _fail(f"MLA cache {tuple(mla.shape)}")
        mqa = torch.empty((rows, HEADS, MQA_DIM), dtype=torch.float8_e4m3fn, device=dev)
        # the indexer operands, on the layers that run the indexer only
        kw = qc = None
        idx = idx_stride = ik_w = ik_b = irope = irope_stride = 0
        if full:
            kw = step.index_kw
            if kw is None:
                raise _fail("no indexer key projection")
            _check_rows(kw, rows, INDEX_KW, torch.bfloat16, "indexer key projection")
            # the index-K rows as the stock store writes them (a pool that splits its
            # layers across owners, the only one with host-side bookkeeping there, is
            # refused: enable_dsa_cache_layer_split off)
            buf = pool.get_index_k_with_scale_buffer(layer_id=attn.layer_id)
            idx, idx_stride = buf.data_ptr(), buf.stride(0) * buf.element_size()
            qc = torch.empty((rows, Q_LORA), dtype=torch.bfloat16, device=dev)
            ik_w, ik_b = la.ik_w.data_ptr(), la.ik_b.data_ptr()
            irope, irope_stride = _addr(la.index_rope)
        base, stride = qkv_latent.data_ptr(), qkv_latent.stride(0)
        ones = self.glue.ones.data_ptr()
        pro = self.pro if full or self.pro_end is None else self.pro_end
        pro(
            *_i64(base, stride),  # q_c, q_c_stride
            *_i64(base + Q_LORA * 2, stride),  # kv_c, kv_c_stride
            *_i64(base + (Q_LORA + KV_LORA) * 2, stride),  # k_pe, k_pe_stride
            *_i64(*_addr(kw)),  # idx_k, idx_k_stride (0, 0: no indexer)
            *_i64(pos.data_ptr(), loc.data_ptr()),  # positions, slot_mapping
            *_i64(loc.data_ptr() if full else 0),  # idx_slot_mapping
            Int32(rows),
            Int32(int(full)),  # has_indexer
            *_i64(la.w_qb, la.w_uk),
            *_i64(la.q_norm_w.data_ptr(), la.kv_norm_w.data_ptr()),
            Float32(la.q_eps),
            Float32(la.kv_eps),
            *_i64(*_addr(la.rope)),  # rope_cache, rope_stride
            *_i64(ones, ones),  # q_scale, k_scale
            *_i64(ik_w, ik_b),  # index-K norm weight, bias
            Float32(la.ik_eps),
            *_i64(irope, irope_stride),  # index_rope_cache, index_rope_stride
            *_i64(*_addr(mqa)),  # mqa_q, mqa_stride
            *_i64(mla.data_ptr(), *mla.stride()[:2], 1),  # mla_cache, strides, size
            *_i64(idx, idx_stride, INDEX_PAGE),  # idx_cache, block stride, size
            *_i64(0, 0),  # topk, topk_stride: unused
            Int32(0),  # topk_width: unused
            *_i64(*_addr(qc)),  # qc_out, qc_out_stride
            cu_stream(dev),
        )
        step.q_fp8 = mqa
        return PrologueOut(qc, None, None, None, None)

    def _done(self, *args) -> None:
        """A step the prologue did: the index-K rows, the W_UK absorption.
        ``q_absorb`` returns None: the prologue left the absorbed FP8 query in
        ``step.q_fp8``, which ``rope_cache`` returns."""

    index_k = q_absorb = _done

    def index_q(self, attn, x, q_lora: torch.Tensor, step: StepContext) -> IndexQ:
        from cutlass import Float32, Int32

        rows = q_lora.shape[0]
        _check_rows(q_lora, rows, Q_LORA, torch.bfloat16, "indexer query latent")
        _, weights_raw = self._index_kw(attn.indexer, x, step)
        la = self.glue.layers[attn.layer_id]
        pos = step.positions
        _check_index(pos, rows, "positions")
        dev = q_lora.device
        q = torch.empty(
            (rows, INDEX_HEADS, INDEX_DIM), dtype=torch.float8_e4m3fn, device=dev
        )
        w = torch.empty((rows, INDEX_HEADS), dtype=torch.float32, device=dev)
        self.iq(
            *_i64(la.w_iq),  # weights
            *_i64(*_addr(q_lora)),  # x, x_stride
            *_i64(pos.data_ptr()),  # positions
            *_i64(*_addr(la.index_rope)),  # rope_cache, rope_stride
            *_i64(*_addr(weights_raw)),  # index_weights, index_weights_stride
            Float32(la.w_scale),
            Int32(rows),
            *_i64(*_addr(q)),  # q_fp8, q_stride
            *_i64(*_addr(w)),  # w_out, w_out_stride
            cu_stream(dev),
        )
        return IndexQ(q, w.unsqueeze(-1))

    def rope_cache(self, attn, q_nope_out, q_pe, k_nope, k_pe, step: StepContext):
        if step.q_fp8 is None:
            raise _fail("the prologue left no FP8 query")
        return step.q_fp8

    def sparse_mla(self, attn, q_fp8: torch.Tensor, topk_indices, step: StepContext):
        from cutlass import Float32, Int32

        layer = attn.attn_mqa
        rows = q_fp8.shape[0]
        if topk_indices is None:
            raise _fail("no top-k cache rows")
        # as the stock path: the fused top-k page table (the top-k indices on the
        # served top-k backends) holds physical cache rows
        ids = pad_topk_indices(topk_indices, rows)
        if (
            ids.dtype != torch.int32
            or tuple(ids.shape) != (rows, TOPK)
            or not ids.is_contiguous()
            or q_fp8.dtype != torch.float8_e4m3fn
            or tuple(q_fp8.shape) != (rows, HEADS, MQA_DIM)
            or not q_fp8.is_contiguous()
        ):
            raise _fail(
                "unexpected sparse MLA inputs "
                f"{tuple(q_fp8.shape)} {q_fp8.dtype} / {tuple(ids.shape)} {ids.dtype}"
            )
        kv = step.view.kv_pool.get_key_buffer(layer.layer_id)
        if kv.shape[-1] != MQA_DIM or not kv.is_contiguous() or kv.data_ptr() % 16:
            raise _fail(f"MLA cache {tuple(kv.shape)}")
        k_scale = getattr(layer, "k_scale_float", None)
        bmm1_scale = 1.0 * (1.0 if k_scale is None else float(k_scale)) * layer.scaling
        out = torch.empty(
            (rows, HEADS, KV_LORA), dtype=torch.bfloat16, device=q_fp8.device
        )
        self.sparse(
            *_i64(q_fp8.data_ptr(), kv.data_ptr()),  # q, kv
            *_i64(ids.data_ptr(), out.data_ptr()),  # ids, out
            *_i64(kv.numel() // MQA_DIM),  # kv_rows
            Int32(rows),
            Float32(_f32(bmm1_scale * LOG2E)),  # scale_log2
            Float32(1.0),  # bmm2_scale
            # unused: the ids are physical cache rows (no block-table translation)
            *_i64(0, 0, 0),  # request_of_row, block_table, block_table_stride
            Int32(0),  # max_blocks
            Int32(0),  # block_shift
            *_i64(0, 0),  # block_rows, phys_out
            cu_stream(q_fp8.device),
        )
        return out

    def uv_proj(self, attn, attn_output: torch.Tensor) -> torch.Tensor:
        from cutlass import Int32

        rows = attn_output.shape[0]
        if (
            attn_output.dtype != torch.bfloat16
            or attn_output.dim() != 3
            or tuple(attn_output.shape[1:]) != (HEADS, KV_LORA)
            or attn_output.stride(2) != 1
            or not 1 <= rows <= self.max_rows
            or attn_output.data_ptr() % 16
            or attn_output.stride(0) % 8
            or attn_output.stride(1) % 8
        ):
            raise _fail(
                "unexpected value up-projection input "
                f"{tuple(attn_output.shape)} / {attn_output.stride()} / "
                f"{attn_output.dtype}"
            )
        out = torch.empty(
            (rows, HEADS * VALUE_DIM), dtype=torch.bfloat16, device=attn_output.device
        )
        la = self.glue.layers[attn.layer_id]
        self.uv(
            *_i64(la.w_uv),  # weights
            *_i64(attn_output.data_ptr(), *attn_output.stride()[:2]),  # attn, strides
            *_i64(*_addr(out)),  # out, out_row_stride
            Int32(rows),
            *_slot_args(la.stage0),
            *_slot_args(la.stage1),
            cu_stream(attn_output.device),
        )
        return out


def _rope_cache_fp32(cache: torch.Tensor, what: str) -> bool:
    """Whether a [positions, cos 32 | sin 32] cache is FP32 (else BF16)."""
    if (
        cache.dtype not in (torch.float32, torch.bfloat16)
        or cache.dim() != 2
        or cache.shape[1] != ROPE
        or cache.stride(1) != 1
    ):
        raise _fail(
            f"unsupported {what} {tuple(cache.shape)} {cache.dtype}", ValueError
        )
    return cache.dtype == torch.float32


class AttentionGlue:
    """Derived attention weights, compiled kernels and per-row-count steps."""

    def __init__(self, model, derived, layers=None) -> None:
        from sglang.srt.layers.layernorm import LayerNorm

        from sgl_nvidia_mega.glm5_3.runtime import SLOTS
        from sgl_nvidia_mega.glm5_3.sm103.packing import (
            pack_indexer_q,
            pack_q_b,
            pack_qkv_a,
            pack_uv_proj,
            pack_w_uk_t,
        )

        # the target model's glue, unless a caller passes its own decoder layers
        self.target = layers is None
        layers = list(model.model.layers if layers is None else layers)
        self.decoder_layers = layers
        dev = layers[0].self_attn.kv_b_proj.weight.device
        self.grid = torch.cuda.get_device_properties(dev).multi_processor_count
        self.ones = torch.ones(1, dtype=torch.float32, device=dev)
        self.layers: Dict[int, LayerArgs] = {}
        # the compiled kernel forms by ``attn.api.compile_specs`` name:
        # "uv|prologue|index_q.{8,16}" (row capacity), "prologue_end.{8,16}" (target
        # only), "sparse.{16,8}" (warps per CTA: 16 up to 6 rows, 8 above) and
        # "qkv_a[_shared[_dn]].8"
        self.fn: Dict[str, object] = {}
        rope_f32, index_rope_f32 = set(), set()

        def fused_a(attn) -> torch.Tensor:
            qkv_a = attn.fused_qkv_a_proj_with_mqa
            return _bf16_weight(qkv_a, (QKV_A_ROWS, HIDDEN), "fused q_a / kv_a")

        # the target's layers without their own indexer, on the small-M projection
        # (slot qkv_a_small_m_shared): one packed copy of the fused q_a / kv_a weight
        # per layer
        shared_qa: Dict[int, torch.Tensor] = {}
        if self.target and "qkv_a_small_m_shared" in SLOTS:
            for i, layer in enumerate(layers):
                attn = layer.self_attn
                if getattr(attn, "indexer", None) is None and getattr(
                    attn, "skip_topk", False
                ):
                    packed = pack_qkv_a(fused_a(attn), self.grid)
                    shared_qa[i] = derived.add("attention.qkv_a_shared", packed)
        self.shared_qkv_a_layers = len(shared_qa)
        for i, layer in enumerate(layers):
            attn = layer.self_attn
            kv_b = _bf16_weight(attn.kv_b_proj, (KV_B_ROWS, KV_LORA), "kv_b_proj")
            q_b = _bf16_weight(attn.q_b_proj, (Q_B_ROWS, Q_LORA), "q_b_proj")
            qkv_a = fused_a(attn)
            nxt = None
            if i + 1 < len(layers):
                # the tensor the next layer's projection reads
                nxt = shared_qa.get(
                    i + 1, layers[i + 1].self_attn.fused_qkv_a_proj_with_mqa.weight
                )
            rope = attn.rotary_emb.cos_sin_cache
            rope_f32.add(_rope_cache_fp32(rope, "RoPE cache"))
            args = LayerArgs(
                w_uv=derived.add("attention.uv_proj", pack_uv_proj(kv_b)).data_ptr(),
                stage0=_prefetch_slot(nxt, NEXT_QKV_A_PREFETCH_BYTES),
                w_qb=derived.add("attention.q_b", pack_q_b(q_b)).data_ptr(),
                w_uk=derived.add("attention.w_uk_t", pack_w_uk_t(kv_b)).data_ptr(),
                q_norm_w=attn.q_a_layernorm.weight,
                kv_norm_w=attn.kv_a_layernorm.weight,
                q_eps=float(attn.q_a_layernorm.variance_epsilon),
                kv_eps=float(attn.kv_a_layernorm.variance_epsilon),
                rope=rope,
            )
            self.layers[attn.layer_id] = args
            if i in shared_qa:
                args.w_qkva_s = shared_qa[i].data_ptr()
                args.in_eps = _f32(layer.input_layernorm.variance_epsilon)
            if self.target:
                # slot 0 stages this layer's o_proj weight, slot 1 the next q_a / kv_a
                wo = _o_proj_stage_slot(attn.o_proj.weight)
                if wo != NO_SLOT:
                    args.stage0, args.stage1 = wo, args.stage0
            # every layer with an indexer; the target's layers that reuse the
            # previous top-k have none
            ix = getattr(attn, "indexer", None)
            if ix is None:
                continue
            if (
                not isinstance(ix.k_norm, LayerNorm)
                or ix.k_norm.weight.dtype != torch.float32
                or ix.k_norm.bias is None
            ):
                raise ValueError(
                    "fused verify attention needs the fp32 LayerNorm index-K norm"
                )
            if ix.n_heads != INDEX_HEADS or ix.head_dim != INDEX_DIM:
                raise _fail("unexpected indexer shape", ValueError)
            wq_b = _bf16_weight(
                ix.wq_b, (INDEX_HEADS * INDEX_DIM, Q_LORA), "indexer wq_b"
            )
            wkw = _bf16_weight(
                ix.wk_weights_proj, (INDEX_KW, HIDDEN), "indexer key / head gate"
            )
            args.w_iq = derived.add(
                "attention.index_q", pack_indexer_q(wq_b)
            ).data_ptr()
            args.w_qkva = derived.add(
                "attention.qkv_a_index",
                pack_qkv_a(torch.cat([qkv_a, wkw], dim=0), self.grid),
            ).data_ptr()
            args.ik_w, args.ik_b = ix.k_norm.weight, ix.k_norm.bias
            args.ik_eps = float(ix.k_norm.variance_epsilon)
            args.index_rope = ix.rotary_emb.cos_sin_cache
            index_rope_f32.add(_rope_cache_fp32(args.index_rope, "index RoPE cache"))
            args.w_scale = _f32(ix.softmax_scale * ix.n_heads**-0.5)
        if len(rope_f32) != 1 or len(index_rope_f32) > 1:
            raise _fail("RoPE cache dtypes differ by layer", ValueError)
        self.rope_f32 = rope_f32.pop()
        self.index_rope_f32 = index_rope_f32.pop() if index_rope_f32 else True

    def compile(self, rows: Tuple[int, ...]) -> None:
        import cutlass.cute as cute

        from sgl_nvidia_mega.glm5_3.runtime import SLOTS
        from sgl_nvidia_mega.glm5_3.sm103.attention import api

        shared = QKV_A_ROWS if self.shared_qkv_a_layers else None
        specs = api.compile_specs(
            sorted({kernel_rows(m) for m in rows} - {None}),
            numerics=_prologue_numerics(),
            index_q_rope_form=INDEX_Q_ROPE_FORM,
            rope_cache_fp32=self.rope_f32,
            index_rope_cache_fp32=self.index_rope_f32,
            warps=sorted({sparse_warps(m) for m in rows if kernel_rows(m)}),
            qkv_a=(QKV_A_ROWS + INDEX_KW, QKV_A_ROWS),
            grid=self.grid,
            prologue_end=self.target,
            qkv_a_shared=shared,
            qkv_a_shared_dn=bool(shared) and "dnorm" in SLOTS,
        )
        for name, (kernel, key, args) in specs.items():
            fn = compile_cached(
                kernel, key, lambda k=kernel, a=args: cute.compile(k, *a)
            )
            self.fn[name] = _allow_cluster16(fn) if name.startswith("sparse.") else fn
            if name == "qkv_a_shared.8":
                logger.info(
                    "NVIDIA megakernel: GLM-5.3 / GLM-5.2: q_a / kv_a of %d layers "
                    "without their own indexer on the small-M projection (packed "
                    "copies %.1f MiB)",
                    self.shared_qkv_a_layers,
                    self.shared_qkv_a_layers * QKV_A_ROWS * HIDDEN * 2 / 2**20,
                )
            elif name == "qkv_a_shared_dn.8":
                logger.info(
                    "NVIDIA megakernel: GLM-5.3 / GLM-5.2: deferred input norm: "
                    "epilogue-scaled q_a / kv_a projection built"
                )

    def step_reasons(self, model, forward_batch, view) -> List[str]:
        """Host-only facts the attention kernels need beyond the stock copies
        (read at the first capture of each row count through the running forward's
        ``view``; any reason on any rank keeps that row count on the stock forward)."""
        from sglang.kernels.ops.kvcache.dsa_fused_store_index_cache import (
            can_use_dsa_fused_store,
        )
        from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz

        pool = view.kv_pool
        reasons: List[str] = []
        page = pool.index_page_size
        if page != INDEX_PAGE:
            reasons.append(f"index-K page size {page} is not {INDEX_PAGE}")
        elif is_fp8_fnuz() or not can_use_dsa_fused_store(
            torch.bfloat16, torch.int64, page
        ):
            reasons.append("the fused index-K store is not available")
        for layer in self.decoder_layers:
            attn = layer.self_attn
            lid = attn.layer_id
            reasons += [f"layer {lid}: {why}" for why in rope_style_reasons(attn)]
            mla = pool.get_key_buffer(lid)
            if (
                mla.dim() != 3
                or tuple(mla.shape[1:]) != (1, MQA_DIM)
                or mla.element_size() != 1
                or not mla.is_contiguous()
            ):
                reasons.append(f"layer {lid}: MLA cache {tuple(mla.shape)} {mla.dtype}")
            if getattr(attn, "indexer", None) is None:
                continue
            idx = pool.get_index_k_with_scale_buffer(layer_id=lid)
            if (
                idx.dtype != torch.uint8
                or idx.dim() != 2
                or idx.shape[1] != INDEX_PAGE * (INDEX_DIM + 4)
                or idx.stride(1) != 1
            ):
                reasons.append(f"layer {lid}: index-K cache {tuple(idx.shape)}")
            if len(reasons) > 8:
                break
        return reasons

    def for_rows(self, rows: int) -> Optional[AttentionStep]:
        if f"uv.{kernel_rows(rows)}" not in self.fn:
            return None
        return FusedAttention(self, rows)


def _allow_cluster16(compiled):
    """The sparse MLA kernel's 16-CTA cluster is a non-portable cluster size:
    allow it on every kernel of the compiled module before its first launch."""
    import cuda.bindings.driver as cuda

    lib = getattr(compiled, "library", None)
    if lib is None and hasattr(compiled, "to"):
        compiled.to(torch.cuda.current_device())
        lib = getattr(compiled, "library", None)
    if lib is None:
        raise _fail("no module handle for the cluster size")
    _, n = cuda.cuLibraryGetKernelCount(lib)
    _, kernels = cuda.cuLibraryEnumerateKernels(n, lib)
    dev = torch.cuda.current_device()
    attr = cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED
    for k in kernels:
        res = cuda.cuKernelSetAttribute(attr, 1, k, dev)
        err = res[0] if isinstance(res, tuple) else res
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise _fail(f"cluster size attribute: {err}")
    return compiled


# the family protocol (plan.build_providers); other callers may pass ``layers``
build = AttentionGlue
