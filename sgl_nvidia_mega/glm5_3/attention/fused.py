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
import os
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
# Measurement knobs (read at build; prefetches are hints, so no output changes):
# SGLANG_NVIDIA_MEGAKERNEL_GLM_OPROJ_STAGE_PREFIX bytes of each o_proj slab (0 .. 196,608; "full" = the whole slab) and
# SGLANG_NVIDIA_MEGAKERNEL_GLM_NEXT_QKV_A_PREFETCH bytes of the next layer's q_a / kv_a weight.
OPROJ_STAGE_PREFIX_ENV = "SGLANG_NVIDIA_MEGAKERNEL_GLM_OPROJ_STAGE_PREFIX"
# ... applied from this row count on (fewer rows keep O_PROJ_STAGE_PREFIX_BYTES): at 4 rows the attention core is short,
# and the longer staging delays the up-projection itself (T84: +0.9 us per layer at 4 rows, -1.5 us at 8 rows)
OPROJ_STAGE_MIN_ROWS_ENV = "SGLANG_NVIDIA_MEGAKERNEL_GLM_OPROJ_STAGE_MIN_ROWS"
NEXT_QKV_A_PREFETCH_ENV = "SGLANG_NVIDIA_MEGAKERNEL_GLM_NEXT_QKV_A_PREFETCH"


_OPROJ_STAGE_MIN_ROWS = int(os.environ.get(OPROJ_STAGE_MIN_ROWS_ENV, "1") or "1")
UV_VPC_ENV = "SGLANG_NVIDIA_MEGAKERNEL_GLM_UV_VPC"
_UV_VPC = int(os.environ.get(UV_VPC_ENV, "16") or "16")
if _UV_VPC not in (16, 32, 64):
    raise ValueError(f"{UV_VPC_ENV}={_UV_VPC}: expected 16, 32 or 64")
DRAFT_QKV_A_DECODE_ENV = "SGLANG_NVIDIA_MEGAKERNEL_GLM_DRAFT_QKV_A_DECODE"
_DRAFT_QKV_A_DECODE = os.environ.get(DRAFT_QKV_A_DECODE_ENV, "1") == "1"
# SGLANG_NVIDIA_MEGAKERNEL_GLM_SPARSE_LL_MIN_ROWS=n (0 = off): the target's verify rows n..16 run the sparse MLA core
# on the flash decode of ll.sparse_mla_fd instead of the 16-CTA SparseMlaDecode: its one-tile kernel (16-CTA
# clusters, one CTA per SM, early gather) up to its one-wave rows (capped by ..._SPARSE_LL_ONE_TILE_MAX, default
# ll's measured 7), its multi-tile kernel (clusters of 8 / 6 / 4 CTAs over several 128-position tiles) above, so
# 9-16 rows fit one wave instead of the 16-CTA clusters' two or three. Its P rounding and merge order are
# TRTLLM-Gen's (the stock decode's), not SparseMlaDecode's: numerics change. Co-resident clusters are agreed
# over the TP group (MIN); a row count neither kernel holds in one wave keeps SparseMlaDecode. Read at build.
# SGLANG_NVIDIA_MEGAKERNEL_GLM_SPARSE_MT_MIN_ROWS=n is the same switch from max(n, 9) rows (the multi-tile rows).
SPARSE_LL_MIN_ROWS_ENV = "SGLANG_NVIDIA_MEGAKERNEL_GLM_SPARSE_LL_MIN_ROWS"
SPARSE_LL_ONE_TILE_MAX_ENV = "SGLANG_NVIDIA_MEGAKERNEL_GLM_SPARSE_LL_ONE_TILE_MAX"
SPARSE_MT_MIN_ROWS_ENV = "SGLANG_NVIDIA_MEGAKERNEL_GLM_SPARSE_MT_MIN_ROWS"
_SPARSE_LL_MIN_ROWS = int(os.environ.get(SPARSE_LL_MIN_ROWS_ENV, "0") or "0")
_SPARSE_MT_MIN_ROWS = int(os.environ.get(SPARSE_MT_MIN_ROWS_ENV, "0") or "0")
if _SPARSE_LL_MIN_ROWS <= 0 and _SPARSE_MT_MIN_ROWS > 0:
    _SPARSE_LL_MIN_ROWS = max(_SPARSE_MT_MIN_ROWS, 9)
_SPARSE_LL_ONE_TILE_MAX = int(os.environ.get(SPARSE_LL_ONE_TILE_MAX_ENV, "0") or "0")
# SGLANG_NVIDIA_MEGAKERNEL_GLM_PROLOGUE16_OVERLAP=1: the 16-row MLA prologue (9-16 verify rows) issues every load of
# a phase before that phase's stores (MlaPrologue overlap16), so its row slots' latencies overlap. Same outputs.
# Read at build.
PROLOGUE16_OVERLAP_ENV = "SGLANG_NVIDIA_MEGAKERNEL_GLM_PROLOGUE16_OVERLAP"
_PROLOGUE16_OVERLAP = os.environ.get(PROLOGUE16_OVERLAP_ENV, "0").strip() not in ("", "0")
# SGLANG_NVIDIA_MEGAKERNEL_GLM_PROLOGUE16_KV_SPLIT=1: the 9-16-row prologue writes the MLA / index-K rows 8..15 on
# their own CTAs of the kv cluster (CTAs 2 / 3, idle with the target's top-k width 0) beside rows 0..7, instead of a
# second pass of the same warps; the same outputs. Read at build.
PROLOGUE16_KV_SPLIT_ENV = "SGLANG_NVIDIA_MEGAKERNEL_GLM_PROLOGUE16_KV_SPLIT"
_PROLOGUE16_KV_SPLIT = os.environ.get(PROLOGUE16_KV_SPLIT_ENV, "0").strip() == "1"
# SGLANG_NVIDIA_MEGAKERNEL_GLM_PROLOGUE8_OVERLAP=1: the 8-row MLA prologue (1-8 verify rows) issues each phase's loads
# before that phase's stores and stores its q_pe bytes after the hand-off (MlaPrologue overlap8). Same outputs.
# Read at build.
PROLOGUE8_OVERLAP_ENV = "SGLANG_NVIDIA_MEGAKERNEL_GLM_PROLOGUE8_OVERLAP"
_PROLOGUE8_OVERLAP = os.environ.get(PROLOGUE8_OVERLAP_ENV, "0").strip() not in ("", "0")
# SGLANG_NVIDIA_MEGAKERNEL_GLM_PROLOGUE_PUSH=1: the overlapped prologues (PROLOGUE8 / PROLOGUE16_OVERLAP) hand the
# q_nope rows to the head cluster by st.async pushes completing on each rank's mbarrier, instead of a release
# cluster barrier + remote gather. SGLANG_NVIDIA_MEGAKERNEL_GLM_PROLOGUE_F32X2=1: their q_a normalize branch-free
# with packed f32x2 multiplies. Same outputs. Read at build.
PROLOGUE_PUSH_ENV = "SGLANG_NVIDIA_MEGAKERNEL_GLM_PROLOGUE_PUSH"
PROLOGUE_F32X2_ENV = "SGLANG_NVIDIA_MEGAKERNEL_GLM_PROLOGUE_F32X2"
_PROLOGUE_PUSH = os.environ.get(PROLOGUE_PUSH_ENV, "0").strip() not in ("", "0")
_PROLOGUE_F32X2 = os.environ.get(PROLOGUE_F32X2_ENV, "0").strip() not in ("", "0")
# SGLANG_NVIDIA_MEGAKERNEL_GLM_IDX_SPLIT=1: on the target's layers with their own indexer, the prologue splits in two
# launches: P_front (IndexerFront: the q_a norm the indexer query projection reads and the index-K rows the logits read)
# on the current stream ahead of the indexer, P_back (the stock MlaPrologue with has_indexer 0 and no q_a out, the
# end-trigger form of the shared layers: q_b, W_UK, RoPE + FP8 query, MLA rows) on a side stream forked after the q_a /
# kv_a projection (at 9-16 rows before the stock key / head-gate projection, which moves after the fork) and joined
# before the sparse MLA. =2 forks P_back after the indexer query projection instead. Same outputs, bit for bit. The
# layers without their own indexer and the draft are unchanged. SGLANG_NVIDIA_MEGAKERNEL_GLM_IDX_SPLIT_MIN_ROWS=n
# (default 9): only verify widths of n..16 rows split. Read at build.
IDX_SPLIT_ENV = "SGLANG_NVIDIA_MEGAKERNEL_GLM_IDX_SPLIT"
IDX_SPLIT_MIN_ROWS_ENV = "SGLANG_NVIDIA_MEGAKERNEL_GLM_IDX_SPLIT_MIN_ROWS"
_IDX_SPLIT = (os.environ.get(IDX_SPLIT_ENV, "0").strip() or "0")
if _IDX_SPLIT not in ("0", "1", "2"):
    raise ValueError(f"{IDX_SPLIT_ENV}={_IDX_SPLIT!r}: expected 0, 1 or 2")
_IDX_SPLIT = int(_IDX_SPLIT)
_IDX_SPLIT_MIN_ROWS = int(os.environ.get(IDX_SPLIT_MIN_ROWS_ENV, "9").strip() or "9")
PROLOGUE16_ROW_NORM_ENV = "SGLANG_NVIDIA_MEGAKERNEL_GLM_PROLOGUE16_ROW_NORM"
PROLOGUE16_EARLY_ARM_ENV = "SGLANG_NVIDIA_MEGAKERNEL_GLM_PROLOGUE16_EARLY_ARM"
_PROLOGUE16_ROW_NORM = os.environ.get(PROLOGUE16_ROW_NORM_ENV, "0").strip() not in ("", "0")
_PROLOGUE16_EARLY_ARM = os.environ.get(PROLOGUE16_EARLY_ARM_ENV, "0").strip() not in ("", "0")
# SGLANG_NVIDIA_MEGAKERNEL_GLM_PROLOGUE8_EARLY_ARM=1 (with PROLOGUE8_OVERLAP and PROLOGUE_PUSH): the 8-row prologue's
# head CTA arms its QN barrier with a CTA-scope arrive before the weight copies (MlaPrologue early_arm8), not with a
# cluster-scope release arrive after them. Same outputs. Read at build.
PROLOGUE8_EARLY_ARM_ENV = "SGLANG_NVIDIA_MEGAKERNEL_GLM_PROLOGUE8_EARLY_ARM"
_PROLOGUE8_EARLY_ARM = os.environ.get(PROLOGUE8_EARLY_ARM_ENV, "0").strip() not in ("", "0")


def _env_bytes(name: str, default: int, full: int) -> int:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    value = full if raw == "full" else int(raw)
    if not 0 <= value <= full:
        raise ValueError(f"{name}={raw!r}: expected 0..{full} bytes or 'full'")
    return value
EVICT_LAST = 0

Slot = Tuple[int, int, int, int, int]
NO_SLOT: Slot = (0, 0, 0, 0, 0)


def kernel_rows(rows: int) -> Optional[int]:
    """The row capacity (8 or 16) of the kernel forms that serve ``rows``."""
    return 8 if 1 <= rows <= 8 else 16 if rows <= 16 else None


def sparse_warps(rows: int) -> int:
    """Warps per CTA of the sparse MLA form: one CTA per SM up to 7 rows (7*16=112 CTAs
    <= 152 SMs on GB300, covering K=6 c1), two above."""
    return 16 if rows <= 7 else 8


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
        _env_bytes(
            OPROJ_STAGE_PREFIX_ENV, O_PROJ_STAGE_PREFIX_BYTES, O_PROJ_STAGE_SLAB_BYTES
        ),
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
    # 16-row UV projection weight when cap=8 uses values_per_cta=64 (cap=16 uses 32)
    w_uv16: int = 0


class FusedAttention(AttentionStep):
    """Verify attention of one row count on the attention-family kernels."""

    def __init__(self, glue: AttentionGlue, rows: int) -> None:
        self.glue, fn = glue, glue.fn
        self.max_rows = cap = kernel_rows(rows)
        self.uv = fn[f"uv.{cap}"]
        self.sparse = fn[f"sparse.{sparse_warps(rows)}"]
        # ll flash decode (CTAs per row, one-tile shared memory), or None: the
        # 16-CTA kernel above
        self.ll_fd = getattr(glue, "sparse_ll", {}).get(rows)
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
        # the indexer-layer prologue split (IDX_SPLIT_ENV; target only): P_front, or None
        self.front = None
        if glue.target and glue.idx_split and rows >= glue.idx_split_min_rows:
            self.front = fn.get(f"front.{cap}")
        self.fork_after_iq = glue.idx_split == 2
        self._rows = rows
        self._split_layers: Dict[int, bool] = {}  # first forward: layer id -> split
        self._split_logged = False

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
            if self.front is None:
                step.index_kw, _ = attn.indexer.wk_weights_proj(x)
            # else the split's P_front projects it after the fork (_index_kw)
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
        self,
        attn,
        qkv_latent: torch.Tensor,
        step: StepContext,
        full: Optional[bool] = None,
        stream=None,
    ) -> PrologueOut:
        from cutlass import Float32, Int32

        rows = qkv_latent.shape[0]
        _check_rows(qkv_latent, rows, QKV_A_ROWS, torch.bfloat16, "q_a / kv_a latent")
        if full is None:
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
            cu_stream(dev) if stream is None else _cu(stream),
        )
        step.q_fp8 = mqa
        return PrologueOut(qc, None, None, None, None)

    # ------------------------------------------------ the indexer-layer prologue split
    def indexer_split(self, attn, step: StepContext, run_indexer: bool) -> Optional[str]:
        """``"qkv"`` / ``"iq"`` (where P_back forks) on the layers whose prologue
        splits, else None (called for every layer: the first forward is logged)."""
        if self.front is None:
            return None
        split = run_indexer and self.runs_indexer(attn, step)
        if not self._split_logged:
            # the first forward of this width, logged at its last layer
            self._split_layers[attn.layer_id] = split
            if attn.layer_id == self.glue.last_layer_id:
                self._log_split()
        if not split:
            return None
        return "iq" if self.fork_after_iq else "qkv"

    def _log_split(self) -> None:
        self._split_logged = True
        on = sorted(k for k, v in self._split_layers.items() if v)
        off = sorted(k for k, v in self._split_layers.items() if not v)
        logger.info(
            "NVIDIA megakernel: GLM-5.3 / GLM-5.2: indexer-layer prologue split engaged "
            "at %d rows: %d layers split (P_front + P_back, fork after %s), %d layers "
            "unsplit; split layers %s (%s=%d)",
            self._rows,
            len(on),
            "the indexer query" if self.fork_after_iq else "q_a / kv_a",
            len(off),
            on,
            IDX_SPLIT_ENV,
            self.glue.idx_split,
        )

    def split_stream(self, attn):
        return self.glue.split_stream

    def prologue_front(
        self, attn, x, qkv_latent: torch.Tensor, step: StepContext
    ) -> torch.Tensor:
        """P_front on the current stream: the q_a norm (returned: the indexer query
        projection's input) and the index-K rows of this layer."""
        from cutlass import Float32, Int32

        rows = qkv_latent.shape[0]
        _check_rows(qkv_latent, rows, QKV_A_ROWS, torch.bfloat16, "q_a / kv_a latent")
        self._index_kw(attn.indexer, x, step)  # at 9-16 rows the stock projection
        kw = step.index_kw
        _check_rows(kw, rows, INDEX_KW, torch.bfloat16, "indexer key projection")
        la = self.glue.layers[attn.layer_id]
        loc = step.forward_batch.out_cache_loc
        pos = step.positions
        _check_index(loc, rows, "cache slots")
        _check_index(pos, rows, "positions")
        buf = step.view.kv_pool.get_index_k_with_scale_buffer(layer_id=attn.layer_id)
        dev = qkv_latent.device
        qc = torch.empty((rows, Q_LORA), dtype=torch.bfloat16, device=dev)
        self.front(
            *_i64(*_addr(qkv_latent)),  # q_c, q_c_stride
            *_i64(*_addr(kw)),  # idx_k, idx_k_stride
            *_i64(pos.data_ptr(), loc.data_ptr(), loc.data_ptr()),  # positions, slots
            Int32(rows),
            *_i64(la.q_norm_w.data_ptr()),
            Float32(la.q_eps),
            *_i64(la.ik_w.data_ptr(), la.ik_b.data_ptr()),
            Float32(la.ik_eps),
            *_i64(*_addr(la.index_rope)),  # index_rope_cache, index_rope_stride
            *_i64(buf.data_ptr(), buf.stride(0) * buf.element_size(), INDEX_PAGE),
            *_i64(*_addr(qc)),  # qc_out, qc_out_stride
            cu_stream(dev),
        )
        return qc

    def prologue_back(
        self, attn, qkv_latent: torch.Tensor, step: StepContext, stream
    ) -> PrologueOut:
        """P_back on ``stream``: the prologue of a layer without its own indexer (the
        end-trigger form when built), its FP8 query left in ``step.q_fp8``; outputs
        allocated on the current stream."""
        return self.prologue(attn, qkv_latent, step, full=False, stream=stream)

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
        if self.ll_fd is not None:
            from sgl_nvidia_mega.glm5_3.ll import sparse_mla as fd

            splits, smem = self.ll_fd
            # physical rows, every position used (masks: id < 0, id >= cache rows)
            return fd.flash_decode(
                q_fp8,
                kv.view(-1, MQA_DIM),
                ids,
                self.glue.ll_lens,
                _f32(bmm1_scale * LOG2E),
                out=out,
                smem_bytes=smem,
                splits=splits,
                out_scale=1.0,
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
        stage0 = la.stage0
        if rows < _OPROJ_STAGE_MIN_ROWS and stage0[2] == O_PROJ_STAGE_SLAB_BYTES:
            # below OPROJ_STAGE_MIN_ROWS the o_proj slot stages the default prefix
            stage0 = (*stage0[:3], O_PROJ_STAGE_PREFIX_BYTES, stage0[4])
        w_uv = la.w_uv16 if (self.max_rows == 16 and la.w_uv16) else la.w_uv
        self.uv(
            *_i64(w_uv),  # weights
            *_i64(attn_output.data_ptr(), *attn_output.stride()[:2]),  # attn, strides
            *_i64(*_addr(out)),  # out, out_row_stride
            Int32(rows),
            *_slot_args(stage0),
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
        elif not self.target and _DRAFT_QKV_A_DECODE:
            for i, layer in enumerate(layers):
                packed = pack_qkv_a(fused_a(layer.self_attn), self.grid)
                shared_qa[i] = derived.add("attention.qkv_a_shared", packed)
        self.shared_qkv_a_layers = len(shared_qa)
        self.uv_vpc = _UV_VPC if self.target else 16
        self._derived = derived
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
                w_uv=derived.add(
                    "attention.uv_proj", pack_uv_proj(kv_b, values_per_cta=self.uv_vpc)
                ).data_ptr(),
                stage0=_prefetch_slot(
                    nxt,
                    _env_bytes(
                        NEXT_QKV_A_PREFETCH_ENV, NEXT_QKV_A_PREFETCH_BYTES, 1 << 30
                    ),
                ),
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
        # the indexer-layer prologue split (IDX_SPLIT_ENV): the target's only, and only
        # with layers that run their own indexer
        self.idx_split = _IDX_SPLIT if self.target and index_rope_f32 else 0
        self.idx_split_min_rows = _IDX_SPLIT_MIN_ROWS
        self.rope_f32 = rope_f32.pop()
        self.index_rope_f32 = index_rope_f32.pop() if index_rope_f32 else True
        self.split_stream = (
            torch.cuda.Stream(device=dev) if self.idx_split else None
        )
        self.last_layer_id = max(layer.self_attn.layer_id for layer in layers)

    def compile(self, rows: Tuple[int, ...]) -> None:
        import cutlass.cute as cute

        from sgl_nvidia_mega.glm5_3.runtime import SLOTS
        from sgl_nvidia_mega.glm5_3.sm103.attention import api

        caps = sorted({kernel_rows(m) for m in rows} - {None})
        if self.uv_vpc == 64 and 16 in caps:
            from sgl_nvidia_mega.glm5_3.sm103.packing import pack_uv_proj

            for layer in self.decoder_layers:
                attn = layer.self_attn
                kv_b = _bf16_weight(attn.kv_b_proj, (KV_B_ROWS, KV_LORA), "kv_b_proj")
                self.layers[attn.layer_id].w_uv16 = self._derived.add(
                    "attention.uv_proj16", pack_uv_proj(kv_b, values_per_cta=32)
                ).data_ptr()
        shared = QKV_A_ROWS if self.shared_qkv_a_layers else None
        specs = api.compile_specs(
            caps,
            numerics=_prologue_numerics(),
            index_q_rope_form=INDEX_Q_ROPE_FORM,
            rope_cache_fp32=self.rope_f32,
            index_rope_cache_fp32=self.index_rope_f32,
            warps=sorted({sparse_warps(m) for m in rows if kernel_rows(m)}),
            qkv_a=(QKV_A_ROWS + INDEX_KW, QKV_A_ROWS),
            grid=self.grid,
            prologue_end=self.target,
            qkv_a_shared=shared,
            qkv_a_shared_dn=bool(shared) and self.target and "dnorm" in SLOTS,
            prologue16_overlap=_PROLOGUE16_OVERLAP,
            prologue16_kv_split=_PROLOGUE16_KV_SPLIT,
            prologue8_overlap=_PROLOGUE8_OVERLAP,
            prologue_push=_PROLOGUE_PUSH,
            prologue_f32x2=_PROLOGUE_F32X2,
            idx_front=bool(self.idx_split),
            prologue16_row_norm=_PROLOGUE16_ROW_NORM,
            prologue16_early_arm=_PROLOGUE16_EARLY_ARM,
            prologue8_early_arm=_PROLOGUE8_EARLY_ARM,
            uv_values_per_cta=self.uv_vpc,
        )
        if self.idx_split:
            split_rows = [m for m in sorted(rows) if self.idx_split_min_rows <= m <= 16]
            fronts = {f"front.{kernel_rows(m)}" for m in split_rows}
            for name in [k for k in specs if k.startswith("front.") and k not in fronts]:
                del specs[name]  # no width splits at this capacity
            logger.info(
                "NVIDIA megakernel: GLM-5.3 / GLM-5.2: indexer-layer prologue split built: "
                "P_front %s, P_back = %s on a side stream forked after %s, at rows %s "
                "(%s=%d, %s=%d)",
                sorted(fronts),
                sorted({f"prologue_end.{kernel_rows(m)}" for m in split_rows}),
                "the indexer query" if self.idx_split == 2 else "q_a / kv_a",
                split_rows,
                IDX_SPLIT_ENV,
                self.idx_split,
                IDX_SPLIT_MIN_ROWS_ENV,
                self.idx_split_min_rows,
            )
        if _PROLOGUE16_OVERLAP and "prologue.16" in specs:
            logger.info(
                "NVIDIA megakernel: GLM-5.3 / GLM-5.2: 16-row MLA prologue in the "
                "overlapped issue order (%s=1)",
                PROLOGUE16_OVERLAP_ENV,
            )
        if _PROLOGUE16_KV_SPLIT and "prologue.16" in specs:
            logger.info(
                "NVIDIA megakernel: GLM-5.3 / GLM-5.2: the 9-16-row prologue writes the "
                "kv rows 8-15 on their own CTAs (%s)",
                "target" if self.target else "draft",
            )
        if "prologue.8" in specs and getattr(specs["prologue.8"][0], "early_arm8", False):
            logger.info(
                "NVIDIA megakernel: GLM-5.3 / GLM-5.2: 8-row MLA prologue's QN barrier "
                "armed before its weight copies (%s=1)",
                PROLOGUE8_EARLY_ARM_ENV,
            )
        if _PROLOGUE8_OVERLAP and "prologue.8" in specs:
            logger.info(
                "NVIDIA megakernel: GLM-5.3 / GLM-5.2: 8-row MLA prologue in the "
                "overlapped issue order (%s=1)",
                PROLOGUE8_OVERLAP_ENV,
            )
        for on, env, what in (
            (_PROLOGUE_PUSH, PROLOGUE_PUSH_ENV, "pushed q_nope hand-off"),
            (_PROLOGUE_F32X2, PROLOGUE_F32X2_ENV, "packed branch-free normalize"),
        ):
            forms = [
                c
                for c, flag in ((8, _PROLOGUE8_OVERLAP), (16, _PROLOGUE16_OVERLAP))
                if on and flag and f"prologue.{c}" in specs
            ]
            if forms:
                logger.info(
                    "NVIDIA megakernel: GLM-5.3 / GLM-5.2: MLA prologue %s at max rows %s "
                    "(%s=1)",
                    what,
                    forms,
                    env,
                )
        for key_name, env, what in (
            ("row_norm", PROLOGUE16_ROW_NORM_ENV, "q rows summed and normalized by their loading warp"),
            ("early_arm", PROLOGUE16_EARLY_ARM_ENV, "QN barrier armed before the weight copies"),
        ):
            if "prologue.16" in specs and specs["prologue.16"][1].get(key_name):
                logger.info(
                    "NVIDIA megakernel: GLM-5.3 / GLM-5.2: 16-row MLA prologue: %s (%s=1)",
                    what,
                    env,
                )
        self.sparse_ll = self._sparse_ll_plan(rows)
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

    def _sparse_ll_plan(self, rows: Tuple[int, ...]) -> Dict[int, Tuple[int, Optional[int]]]:
        """{verify rows: (ll flash-decode CTAs per row, one-tile shared memory or
        None)} for the rows SPARSE_LL_MIN_ROWS_ENV moves to ll.sparse_mla_fd: the
        one-tile kernel up to its one-wave (spread) rows, the multi-tile kernel above;
        co-resident clusters agreed over the TP group, so every rank builds one plan."""
        lo = _SPARSE_LL_MIN_ROWS
        # the target's verify only (the draft layer keeps its stock sparse MLA)
        want = sorted(m for m in rows if self.target and lo > 0 and lo <= m <= 16)
        if not want:
            return {}
        from sgl_nvidia_mega.glm5_3.ll import sparse_mla as fd

        caps = err = None
        try:
            caps = fd.device_caps(HEADS, with_mt=True, with_fold=False)
        except Exception as e:  # noqa: BLE001 (agreed: raises on every rank)
            err = e
        caps = fd.agree_caps(caps, err, fd._tp_group(), with_mt=True, with_fold=False)
        one_max = min(caps.spread_rows, _SPARSE_LL_ONE_TILE_MAX or fd.ONE_TILE_MAX_ROWS)
        plan: Dict[int, Tuple[int, Optional[int]]] = {}
        for m in want:
            if m <= one_max:
                plan[m] = (fd.ONE_TILE_SPLITS, caps.spread_smem_bytes)
            else:
                sp = fd.mt_splits_for_rows(m, caps.mt_rows)
                if sp:
                    plan[m] = (sp, None)
        if plan:
            dev = torch.device("cuda", torch.cuda.current_device())
            self.ll_lens = torch.full((16,), TOPK, dtype=torch.int32, device=dev)
        logger.info(
            "NVIDIA megakernel: GLM-5.3 / GLM-5.2: sparse MLA on ll.sparse_mla_fd at "
            "rows %s (CTAs per row %s; one-tile <= %d rows, spread clusters %d, "
            "multi-tile co-resident %s) (target)",
            sorted(plan),
            [plan[m][0] for m in sorted(plan)],
            one_max,
            caps.spread_rows,
            list(zip(fd.MT_SPLITS, caps.mt_rows)),
        )
        return plan

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


def _cu(stream):
    import cuda.bindings.driver as cuda

    return cuda.CUstream(stream.cuda_stream)


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
