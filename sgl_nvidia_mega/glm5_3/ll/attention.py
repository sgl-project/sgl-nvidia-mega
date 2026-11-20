# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""``ll.attention``: the attention front of a TP4 rank (16 heads) of the GLM-5.3 /
GLM-5.2 verify step and of the NextN draft layer, as ``attention.step.AttentionStep`` steps
(the unit ``ll.attention`` of the ``attention`` and ``draft_attention`` groups claims
``qkv_a`` / ``prologue`` / ``index_k`` / ``rope_cache``):

- ``qkv_a``: the fused q_a / kv_a projection on the low-latency fused-A forms of
  ``csrc/gemm/fused_a_ll.cuh``: up to 16 rows SGLang's min-latency kernel with a
  224 KB stage ring, plus a pipelined MMA loop from 9 rows, bit for bit SGLang's
  kernel; 17-24 rows the split-K form (deterministic, within
  ``ll.attention.qkv_a_wide.max_rel`` of fp32, as SGLang's own GEMM there); stock
  above 24 rows.
- ``prologue``: the stock q_a norm; ``q_b`` on the pipelined form at 9-16 rows (bit
  for bit SGLang's kernel; stock elsewhere); the kv_a norm deferred into
  ``rope_cache`` when the fold engages, else stock (on the side stream in capture).
- ``index_k``: stock, with the indexer key / head-gate projection (also read by
  ``index_q``) through ``ll.linear("wk_weights_proj")`` (the registered GEMM once
  ``ll.gemm`` is present, the module's own GEMM until then).
- ``rope_cache``: ``csrc/attention/mla_kv_half.cuh`` in one launch: the kv_a norm
  (when folded), RoPE and FP8 of q and k, and the MLA cache append; bit for bit the
  stock FlashInfer norm, ``mla_rope_quantize_fp8`` and ``set_mla_kv_buffer``. It
  returns the step contract's ``q_fp8 [rows, heads, 576]`` e4m3, contiguous.

The fold (option unit ``ll.attention.kv_norm_fold``, default on) engages only when
``norm_backend.decide`` finds FlashInfer's CuTe DSL norm and a bitwise probe against
the stock kv_a norm module at the step's row counts; the family logs which way it
went. ``q_absorb``, the indexer, ``sparse_mla`` and
``uv_proj`` stay with their providers. Every kernel builds at family compile, never
under capture; a build failure raises.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from sgl_nvidia_mega.glm5_3 import ll
from sgl_nvidia_mega.glm5_3.attention.step import (
    AttentionStep,
    PrologueOut,
    StepContext,
)
from sgl_nvidia_mega.glm5_3.ll import jit, norm_backend
from sgl_nvidia_mega.glm5_3.registry import Candidate

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("is_arch_support_pdl",),
    "srt/model_executor/runner/__init__.py": ("get_is_capture_mode",),
    # the running forward's view (Step.view): its KV pool and DSA backend
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/binding.py": (
        "kv_pool",
        "dsa",
        "pool",
        "get_key_buffer",
        "kv_lora_rank",
        "qk_rope_head_dim",
    ),
    "srt/utils/async_probe.py": ("maybe_detect_oob",),
    "srt/layers/layernorm.py": (
        "RMSNorm",
        "variance_epsilon",
        "variance_size_override",
        "cast_x_before_out_mul",
        "_quant_linear",
        "has_weight",
    ),
    "srt/mem_cache/memory_pool.py": (
        "MLATokenToKVPool",
        "set_mla_kv_buffer",
        "_write_mla_kv_buffer",
        "_scatter_mla_rows",
        "dsa_kv_cache_store_fp8",
        "kv_cache_dim",
        "requires_physical_write_loc",
        "get_key_buffer",
    ),
    "srt/models/deepseek_v2.py": (
        "fused_qkv_a_proj_with_mqa",
        "q_b_proj",
        "q_b_proj_forward",
        "has_fused_proj",
        "is_packed_weight",
        "fused_a_gemm_backend",
        "_q_b_proj_verified_shape",
        "q_a_layernorm",
        "kv_a_layernorm",
        "rotary_emb",
        "num_local_heads",
    ),
    "srt/models/deepseek_common/attention_forward_methods/forward_mla.py": (
        "_split_q_nope_pe",
    ),
}

logger = logging.getLogger(__name__)

# One TP4 rank of GLM-5 DSA.
HEADS = 16
HIDDEN, Q_LORA, KV_LORA, ROPE, NOPE = 6144, 2048, 512, 64, 192
QKV_A_SHAPE = (Q_LORA + KV_LORA + ROPE, HIDDEN)  # fused q_a / kv_a weight [2624, 6144]
Q_B_SHAPE = (HEADS * (NOPE + ROPE), Q_LORA)  # q_b weight [4096, 2048]
MQA_DIM = KV_LORA + ROPE  # one FP8 row of q or of the MLA cache

# Named tolerances (spec NUMERICS); each names the test that holds it.
TOLERANCES = {
    # qkv_a's split-K form at 17-24 rows: max |out - fp32| / max |fp32|. Measured
    # worst 3.3e-3 over 1-48 rows on GLM-5.2 layer 5 (SGLang's own GEMM there has the
    # same max error); test_glm5_3_ll_fused_a.py::test_wide / test_production_forms.
    "ll.attention.qkv_a_wide.max_rel": 8e-3,
}

# Kernel plan, measured on GB300 (branch 09 note §11, per-call CUDA-graph replay
# behind a PDL producer). fused-A: the deep stage ring at <= 16 rows (-0.15..-0.32 us
# at <= 8 rows), the pipelined MMA loop from 9 rows (-0.01..-0.08 us; +0.04..+0.15 at
# <= 8); split-K 64x3 (tile_k 128, 4 MMA warps) at 17-24 rows (-0.6..-0.7 us against
# SGLang's GEMM; it loses from 27 rows). q_b (4096 x 2048): the pipelined loop at
# 9-16 rows (-0.08..-0.11 us); at <= 8 rows one run of five lost, so SGLang's
# kernel; above 16 rows SGLang's GEMM wins (split-K never).
QKV_A_DEEP = True
ILP_MIN_ROWS = 9
QKV_A_WIDE = (64, 3, 128, 4)  # tile_m, split_k, tile_k, mma_warps
QKV_A_WIDE_MAX_ROWS = 24
Q_B_LL_ROWS = (9, 16)
# Rows of the 1-16-row forms and of the split-K form (the kernels' own limits).
LL_MAX_ROWS = 16
WIDE_MAX_ROWS = 48
# Verify rows the family serves (profile ACCEPTED at TP4).
MAX_ROWS = 48
# The GB300 the plan was measured on (SM103); other GPUs decline.
SM = (10, 3)
# The steps the unit claims (atomically: the deferred kv_a norm passes from prologue
# to rope_cache).
STEPS = ("qkv_a", "prologue", "index_k", "rope_cache")
# the option unit that folds the kv_a norm (units.OPTIONS)
FOLD_UNIT = "ll.attention.kv_norm_fold"


# ---------------------------------------------------------------- kernel builds


def _cpp(*values) -> str:
    return ", ".join(
        ("true" if v else "false") if isinstance(v, bool) else str(v) for v in values
    )


def _pdl() -> bool:
    from sglang.kernels.jit.utils import is_arch_support_pdl

    return bool(is_arch_support_pdl())


@functools.cache
def _fused_a_module(hd_in: int, hd_out: int, pdl: bool, ilp: bool, deep: bool):
    args = _cpp(hd_in, hd_out, pdl, ilp, deep)
    return jit.load(
        "glm_ll_fused_a",
        "gemm/fused_a_ll.cuh",
        [("fused_a", f"sglang::glm_ll::DSV3FusedAGemmLLKernel<{args}>::run")],
        marker=(hd_in, hd_out, pdl, ilp, deep),
    )


@functools.cache
def _fused_a_wide_module(hd_in: int, hd_out: int, pdl: bool, wide: Tuple[int, ...]):
    args = _cpp(hd_in, hd_out, pdl, *wide)
    cls = f"sglang::glm_ll::DSV3FusedAGemmWideKernel<{args}>"
    return jit.load(
        "glm_ll_fused_a_wide",
        "gemm/fused_a_ll.cuh",
        [
            ("fused_a_wide", f"{cls}::run"),
            ("max_active_clusters", f"{cls}::max_active_clusters"),
        ],
        marker=(hd_in, hd_out, pdl, *wide),
    )


@functools.cache
def _kv_half_module(
    fuse_norm: bool, write_norm_out: bool, pdl: bool, trigger_at_entry: bool
):
    args = _cpp(fuse_norm, write_norm_out, pdl, trigger_at_entry)
    return jit.load(
        "glm_ll_mla_kv_half",
        "attention/mla_kv_half.cuh",
        [
            (
                "mla_kv_half",
                f"sglang::glm_ll::mla_kv_half::MlaKvHalfKernel<{args}>::run",
            )
        ],
        marker=(fuse_norm, write_norm_out, pdl, trigger_at_entry),
    )


# ---------------------------------------------------------------- launchers


def fused_a_ll(
    x: torch.Tensor, weight: torch.Tensor, *, ilp: bool, deep: bool
) -> torch.Tensor:
    """``x @ weight.T`` for 1-16 rows: ``x`` [M, K] bf16 contiguous, ``weight`` [N, K]
    bf16 contiguous. Bit for bit SGLang's ``dsv3_fused_a_gemm`` JIT kernel."""
    out = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    module = _fused_a_module(weight.shape[1], weight.shape[0], _pdl(), ilp, deep)
    module.fused_a(x, weight.T, out)
    return out


def fused_a_wide(
    x: torch.Tensor,
    weight: torch.Tensor,
    wide: Tuple[int, ...] = QKV_A_WIDE,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``x @ weight.T`` for 1-48 rows on the split-K form ``wide`` = (tile_m,
    split_k, tile_k, mma_warps). ``out`` [M, N] (a view of a larger buffer is fine)."""
    if out is None:
        out = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    module = _fused_a_wide_module(weight.shape[1], weight.shape[0], _pdl(), tuple(wide))
    module.fused_a_wide(x, weight.T, out)
    return out


def fused_a_wide_max_active_clusters(
    weight_shape: Sequence[int], rows: int, wide: Tuple[int, ...] = QKV_A_WIDE
) -> int:
    """Clusters of the split-K launch for ``rows`` that fit on the current device
    at once; one wave needs ``weight_shape[0] // tile_m``."""
    hd_out, hd_in = weight_shape
    module = _fused_a_wide_module(hd_in, hd_out, _pdl(), tuple(wide))
    return int(module.max_active_clusters(rows))


def mla_kv_half(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    loc: torch.Tensor,
    kv_buffer: torch.Tensor,
    *,
    norm_weight: Optional[torch.Tensor] = None,
    norm_eps: float = 0.0,
    q_out: Optional[torch.Tensor] = None,
    k_norm_out: Optional[torch.Tensor] = None,
    trigger_at_entry: bool = False,
) -> torch.Tensor:
    """Write ``kv_buffer[loc]`` and return the FP8 query ``[T, H, 576]``.

    bf16 unless noted, last dim contiguous, other strides free:
    ``q_nope`` [T, H, 512] (absorbed), ``q_pe`` [T, H, 64], ``k_nope`` [T, 512] (the raw
    kv_a latent with ``norm_weight``, else normalized), ``k_pe`` [T, 64];
    ``cos_sin_cache`` [P, 64] fp32 contiguous (cos | sin, GPT-J RoPE); ``positions`` and
    ``loc`` [T] int32 / int64 (rows at ``loc`` 0, the graph padding slot, are not
    written); ``kv_buffer`` [S, >= 576] (or [S, 1, >= 576]) e4m3 / uint8 rows;
    ``k_norm_out`` [T, 512] receives the normalized latent (the norm probe only).
    ``trigger_at_entry`` lets PDL dependents launch at entry (they still wait for the
    outputs). Unit quantization scales, as ``mla_quantize_and_rope_for_fp8``."""
    num_tokens, num_heads = q_pe.shape[0], q_pe.shape[1]
    if q_out is None:
        q_out = q_pe.new_empty(
            (num_tokens, num_heads, MQA_DIM), dtype=torch.float8_e4m3fn
        )
    if num_tokens == 0:
        return q_out
    fuse_norm = norm_weight is not None
    write_norm_out = k_norm_out is not None
    if write_norm_out and not fuse_norm:
        raise ValueError("ll.attention: k_norm_out needs norm_weight")
    if kv_buffer.dim() != 2:
        kv_buffer = kv_buffer.view(kv_buffer.shape[0], -1)
    module = _kv_half_module(fuse_norm, write_norm_out, _pdl(), trigger_at_entry)
    module.mla_kv_half(
        q_nope,
        q_pe,
        k_nope,
        k_pe,
        norm_weight if fuse_norm else k_nope,
        cos_sin_cache,
        positions,
        loc,
        q_out,
        kv_buffer,
        k_norm_out if write_norm_out else k_nope,
        float(norm_eps),
        1.0,
        1.0,
        0,
    )
    return q_out


def kv_half_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """The KV-half kernel's norm stage alone (``k_norm_out``): ``x`` [T, 512] bf16,
    row stride free. Writes nothing (every row targets the skipped slot 0)."""
    t, dev = x.shape[0], x.device
    out = torch.empty((t, KV_LORA), dtype=torch.bfloat16, device=dev)
    zeros = torch.zeros(t, dtype=torch.int64, device=dev)
    mla_kv_half(
        torch.zeros((t, 1, KV_LORA), dtype=torch.bfloat16, device=dev),
        torch.zeros((t, 1, ROPE), dtype=torch.bfloat16, device=dev),
        x,
        torch.zeros((t, ROPE), dtype=torch.bfloat16, device=dev),
        torch.zeros((1, ROPE), dtype=torch.float32, device=dev),
        zeros,
        zeros,
        torch.zeros((1, MQA_DIM), dtype=torch.uint8, device=dev),
        norm_weight=weight,
        norm_eps=eps,
        k_norm_out=out,
    )
    return out


# ---------------------------------------------------------------- the plan


@dataclass(frozen=True)
class Forms:
    """The kernel forms of one verify width (None: the stock body)."""

    qkv_a: Optional[Tuple]  # ("ll", ilp, deep) | ("wide", (tile_m, split_k, ...))
    q_b: Optional[Tuple]  # ("ll", ilp)

    def describe(self) -> str:
        qkv = "stock" if self.qkv_a is None else self.qkv_a[0]
        q_b = "stock" if self.q_b is None else self.q_b[0]
        return f"qkv_a {qkv}, q_b {q_b}"


def plan_forms(rows: int, wide_max_rows: int = QKV_A_WIDE_MAX_ROWS) -> Forms:
    """The forms ``rows`` runs, before the device's occupancy gate (``wide_max_rows``)."""
    if 1 <= rows <= LL_MAX_ROWS:
        qkv_a = ("ll", rows >= ILP_MIN_ROWS, QKV_A_DEEP)
    elif rows <= min(wide_max_rows, WIDE_MAX_ROWS):
        qkv_a = ("wide", QKV_A_WIDE)
    else:
        qkv_a = None
    lo, hi = Q_B_LL_ROWS
    q_b = ("ll", True) if lo <= rows <= hi else None
    return Forms(qkv_a, q_b)


# ---------------------------------------------------------------- the provider


def _fail(what: str, cls=RuntimeError) -> Exception:
    return cls(f"ll.attention: {what}")


class LLAttention(AttentionStep):
    """The TP4 attention front at one verify width (``AttentionFamily.provider``)."""

    def __init__(self, family: AttentionFamily, rows: int) -> None:
        self.family = family
        self.rows = rows
        self.forms = family.forms[rows]
        self.fold = family.fold.fold

    def _check_rows(self, t: torch.Tensor, what: str) -> None:
        if t.shape[0] != self.rows:
            raise _fail(f"{what} has {t.shape[0]} rows at the {self.rows}-row width")

    def qkv_a(self, attn, x, step: StepContext) -> torch.Tensor:
        form = self.forms.qkv_a
        if form is None:
            return super().qkv_a(attn, x, step)
        self._check_rows(x, "the attention input")
        w = attn.fused_qkv_a_proj_with_mqa.weight
        if form[0] == "ll":
            return fused_a_ll(x, w, ilp=form[1], deep=form[2])
        return fused_a_wide(x, w, form[1])

    def _q_b(self, attn, q: torch.Tensor) -> torch.Tensor:
        """``q_b_proj_forward`` (``[rows, heads, qk_head_dim]``)."""
        form = self.forms.q_b
        if form is None:
            return attn.q_b_proj_forward(q)
        out = fused_a_ll(q, attn.q_b_proj.weight, ilp=form[1], deep=False)
        return out.view(-1, attn.num_local_heads, attn.qk_head_dim)

    def prologue(
        self, attn, qkv_latent: torch.Tensor, step: StepContext
    ) -> PrologueOut:
        from sglang.srt.model_executor.runner import get_is_capture_mode

        # AttentionStep.prologue with q_b on _q_b and, when folded, the kv_a norm
        # left to rope_cache (so no side-stream fork / join before q_b).
        q, latent_cache = qkv_latent.split(
            [attn.q_lora_rank, attn.kv_lora_rank + attn.qk_rope_head_dim], dim=-1
        )
        k_nope = latent_cache[..., : attn.kv_lora_rank]
        if self.fold:
            q = attn.q_a_layernorm(q)
        elif attn.alt_stream is not None and get_is_capture_mode():
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
        q = self._q_b(attn, q)
        q_nope, q_pe, k_pe = attn._split_q_nope_pe(q, latent_cache)
        return PrologueOut(q_lora, q_nope, q_pe, k_nope, k_pe)

    def _index_kw(self, indexer, x, step: StepContext):
        """(key, raw head gates), projecting ``x`` through ``ll.linear`` unless a
        step already did. The stock ``index_k`` / ``index_q`` (and their two-stream
        form) read it through ``self._index_kw``."""
        if step.index_kw is None:
            step.index_kw = ll.linear("wk_weights_proj", indexer.wk_weights_proj, x)
        return step.index_kw.split([indexer.head_dim, indexer.n_heads], dim=-1)

    def rope_cache(
        self, attn, q_nope_out, q_pe, k_nope, k_pe, step: StepContext
    ) -> torch.Tensor:
        from sglang.srt.utils.async_probe import maybe_detect_oob

        layer = attn.attn_mqa
        fb = step.forward_batch
        rows = q_pe.shape[0]
        self._check_rows(q_pe, "the query")
        loc = fb.out_cache_loc
        positions = fb.positions
        if not (
            isinstance(loc, torch.Tensor)
            and loc.dim() == 1
            and loc.shape[0] == rows
            and isinstance(positions, torch.Tensor)
            and positions.dim() == 1
            and positions.shape[0] >= rows
        ):
            raise _fail(
                f"unexpected cache slots / positions {type(loc).__name__} "
                f"{tuple(getattr(loc, 'shape', ()))} / {tuple(positions.shape)}"
            )
        kv = step.view.kv_pool.get_key_buffer(layer.layer_id)
        kv = kv.view(kv.shape[0], -1)
        # the probe of the pool's write door (SGLANG_ENABLE_ASYNC_ASSERT)
        maybe_detect_oob(loc, 0, kv.shape[0], "ll.attention KV append")
        norm = attn.kv_a_layernorm
        return mla_kv_half(
            q_nope_out,
            q_pe,
            k_nope.view(rows, KV_LORA),
            k_pe.view(rows, ROPE),
            attn.rotary_emb.cos_sin_cache,
            positions[:rows],
            loc,
            kv,
            norm_weight=norm.weight if self.fold else None,
            norm_eps=norm.variance_epsilon,
            trigger_at_entry=self.family.trigger_at_entry,
        )


# ---------------------------------------------------------------- the family


def _layer_reasons(attn, norm_ref) -> List[str]:
    """Why ``attn`` is not the TP4 attention this family's steps reproduce."""
    from sglang.srt.layers.layernorm import RMSNorm

    why = []
    if attn.num_local_heads != HEADS:
        why.append(f"{attn.num_local_heads} heads per rank (TP4 has {HEADS})")
    for name, value, want in (
        ("q_lora_rank", attn.q_lora_rank, Q_LORA),
        ("kv_lora_rank", attn.kv_lora_rank, KV_LORA),
        ("qk_rope_head_dim", attn.qk_rope_head_dim, ROPE),
        ("qk_nope_head_dim", attn.qk_nope_head_dim, NOPE),
    ):
        if value != want:
            why.append(f"{name} {value} (expected {want})")
    # stock runs SGLang's min-latency JIT kernel on both projections up to 16 rows
    if hasattr(attn, "q_a_proj") or not attn.has_fused_proj or attn.is_packed_weight:
        why.append("the q_a / kv_a projection is not one fused bf16 weight")
    for name, linear, shape in (
        ("fused q_a / kv_a", attn.fused_qkv_a_proj_with_mqa, QKV_A_SHAPE),
        ("q_b", attn.q_b_proj, Q_B_SHAPE),
    ):
        w = linear.weight
        if (
            w.dtype != torch.bfloat16
            or tuple(w.shape) != shape
            or not w.is_contiguous()
        ):
            why.append(
                f"{name} weight {w.dtype} {tuple(w.shape)} (expected bf16 {shape})"
            )
    if (
        attn.fused_a_gemm_backend not in ("auto", "jit")
        or not attn._q_b_proj_verified_shape
    ):
        why.append("stock fused-A / q_b do not run SGLang's JIT fused-A kernel")
    rotary = attn.rotary_emb
    cache = rotary.cos_sin_cache
    if rotary.is_neox_style or cache.dtype != torch.float32 or cache.dim() != 2:
        why.append("RoPE is not GPT-J with an fp32 cos / sin cache")
    elif cache.shape[1] != ROPE or not cache.is_contiguous():
        why.append(f"RoPE cache {tuple(cache.shape)}")
    # the kv_a norm the KV-half kernel can take over (folded or not, it must be the
    # plain bf16 RMSNorm the stock path runs on [rows, 512])
    norm = attn.kv_a_layernorm
    if (
        type(norm) is not RMSNorm
        or not norm.has_weight
        or norm.variance_size_override is not None
        or norm.cast_x_before_out_mul
        or norm._quant_linear is not None
        or norm.weight.dtype != torch.bfloat16
        or tuple(norm.weight.shape) != (KV_LORA,)
        or not norm.weight.is_contiguous()
    ):
        why.append("the kv_a norm is not the plain bf16 RMSNorm")
    elif norm_ref is not None and float(norm.variance_epsilon) != norm_ref:
        why.append("kv_a norm epsilons differ by layer")
    return why


def _model_layers(model) -> list:
    """The decoder layers of the target (``model.model.layers``) or of the NextN
    draft (``model.model.decoder``)."""
    inner = model.model
    if hasattr(inner, "layers"):
        return list(inner.layers)
    return [inner.decoder]


class AttentionFamily:
    """The ``ll.attention`` family, built once per model (the target's layers or the
    NextN draft layer): checks the layers, plans the forms per width, builds the
    kernels and the wk|wp GEMM role, and decides the kv_a norm fold."""

    def __init__(self, model, derived=None, profile=None) -> None:
        layers = _model_layers(model)
        self.layers = layers
        self.profile = profile
        self.attns = [layer.self_attn for layer in layers]
        first = self.attns[0]
        self.eps = float(first.kv_a_layernorm.variance_epsilon)
        reasons = []
        for i, attn in enumerate(self.attns):
            reasons += [f"layer {i}: {r}" for r in _layer_reasons(attn, self.eps)]
            if len(reasons) > 8:
                break
        if reasons:
            raise _fail(
                "unsupported attention:\n  - " + "\n  - ".join(reasons), ValueError
            )
        self.device = first.fused_qkv_a_proj_with_mqa.weight.device
        self.forms: Dict[int, Forms] = {}
        self.fold = norm_backend.NormFold(False, "not decided")
        # PDL dependents of the KV half launch only after its loads (any successor);
        # a successor that gathers earlier steps' KV rows before its own wait may ask
        # for an entry trigger
        self.trigger_at_entry = False
        self.wide_max_rows = QKV_A_WIDE_MAX_ROWS
        self.wide_occupancy: Optional[Tuple[int, int]] = None
        self.kw_gemm = ""
        self._served: Dict[int, LLAttention] = {}

    def compile(self, widths: Sequence[int]) -> None:
        """Every kernel form ``widths`` use, the occupancy gate of the split-K form
        and the fold probe; at build, outside capture. Failures raise."""
        if torch.cuda.is_current_stream_capturing():
            raise _fail("compile inside CUDA-graph capture")
        widths = sorted(set(widths))
        if any(not 1 <= m <= MAX_ROWS for m in widths):
            raise _fail(f"widths {widths} outside 1..{MAX_ROWS}", ValueError)
        pdl = _pdl()
        wide_rows = [m for m in widths if LL_MAX_ROWS < m <= QKV_A_WIDE_MAX_ROWS]
        if wide_rows:
            wave = QKV_A_SHAPE[0] // QKV_A_WIDE[0]
            active = min(
                fused_a_wide_max_active_clusters(QKV_A_SHAPE, m) for m in wide_rows
            )
            self.wide_occupancy = (active, wave)
            if active < wave:
                # the split-K form needs one wave; keep SGLang's GEMM there
                self.wide_max_rows = LL_MAX_ROWS
        self.forms = {m: plan_forms(m, self.wide_max_rows) for m in widths}
        for f in self.forms.values():
            if f.qkv_a is not None and f.qkv_a[0] == "ll":
                _fused_a_module(HIDDEN, QKV_A_SHAPE[0], pdl, f.qkv_a[1], f.qkv_a[2])
            elif f.qkv_a is not None:
                _fused_a_wide_module(HIDDEN, QKV_A_SHAPE[0], pdl, f.qkv_a[1])
            if f.q_b is not None:
                _fused_a_module(Q_LORA, Q_B_SHAPE[0], pdl, f.q_b[1], False)
        # the indexer key / head-gate projection (index_k) at every width
        sm = self.profile.sm if self.profile is not None else None
        sms = self.profile.sms if self.profile is not None else None
        kw = {
            ll.build_linear(
                "wk_weights_proj", a.indexer.wk_weights_proj, widths, sm=sm, sms=sms
            )
            for a in self.attns
            if getattr(a, "indexer", None) is not None
        }
        self.kw_gemm = "; ".join(sorted(d for d in kw if d))
        self.fold = self._decide_fold(widths)
        _kv_half_module(self.fold.fold, False, pdl, self.trigger_at_entry)
        self._log(widths)

    def _decide_fold(self, widths: Sequence[int]) -> norm_backend.NormFold:
        from sgl_nvidia_mega.glm5_3 import units

        if not units.enabled(FOLD_UNIT):
            return norm_backend.NormFold(
                False,
                f"{FOLD_UNIT} switched off (not in SGLANG_NVIDIA_MEGAKERNEL_SLOTS)",
            )
        norm = self.attns[0].kv_a_layernorm
        weight = norm.weight.detach()
        return norm_backend.decide(
            lambda x: norm(x),
            lambda x: kv_half_norm(x, weight, self.eps),
            widths,
            KV_LORA,
            row_stride=QKV_A_SHAPE[0],
            col_offset=Q_LORA,
            device=self.device,
        )

    def _log(self, widths: Sequence[int]) -> None:
        by_form: Dict[str, List[int]] = {}
        for m in widths:
            by_form.setdefault(self.forms[m].describe(), []).append(m)
        occupancy = ""
        if self.wide_occupancy is not None:
            occupancy = (
                f"; split-K fused-A {self.wide_occupancy[0]} clusters co-resident, "
                f"{self.wide_occupancy[1]} per wave"
                + ("" if self.wide_max_rows > LL_MAX_ROWS else ": kept stock")
            )
        logger.info(
            "NVIDIA megakernel: GLM-5.3 / GLM-5.2: ll.attention: TP4 attention front on %d layers: %s%s; wk|wp %s; "
            "KV half at every width; kv_a RMSNorm %s (%s)",
            len(self.attns),
            "; ".join(f"{k} at rows {v}" for k, v in by_form.items()),
            occupancy,
            self.kw_gemm or "on the module's own GEMM",
            "folded into the KV half" if self.fold.fold else "not folded",
            self.fold.reason,
        )

    def provider(self, candidate, rows: int, mode: str) -> LLAttention:
        """The family protocol (``registry``): one ``LLAttention`` per width, shared
        by the step modes (its steps do not depend on the mode)."""
        if rows not in self.forms:
            raise _fail(f"no forms built for {rows} rows")
        if rows not in self._served:
            self._served[rows] = LLAttention(self, rows)
        return self._served[rows]

    def step_reasons(self, model, forward_batch, view) -> List[str]:
        """Host-only facts of the KV pool the KV half writes (read at the first
        capture of each width, once the pool exists); ``view`` is the running
        forward's ForwardView."""
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        dsa = view.dsa
        if dsa is None:
            return ["the attention backend is not the DSA backend (no DSA view)"]
        # the pool itself, for host-only type checks (its rows are written through
        # set_mla_kv_buffer's layout, checked below)
        pool = view.kv_pool.pool
        reasons: List[str] = []
        if dsa.kv_lora_rank != KV_LORA or dsa.qk_rope_head_dim != ROPE:
            reasons.append("backend MLA dims")
        cls = type(pool)
        if not (
            isinstance(pool, MLATokenToKVPool)
            and cls.set_mla_kv_buffer is MLATokenToKVPool.set_mla_kv_buffer
            and cls._write_mla_kv_buffer is MLATokenToKVPool._write_mla_kv_buffer
            and cls._scatter_mla_rows is MLATokenToKVPool._scatter_mla_rows
        ):
            reasons.append(f"KV pool {cls.__name__} writes MLA rows its own way")
        elif (
            pool.dtype != torch.float8_e4m3fn
            or pool.dsa_kv_cache_store_fp8
            or pool.kv_cache_dim != MQA_DIM
            or pool.requires_physical_write_loc
        ):
            reasons.append("the MLA cache rows are not raw 576-byte FP8 rows")
        for attn in self.attns:
            kv = view.kv_pool.get_key_buffer(attn.attn_mqa.layer_id)
            if (
                kv.dim() != 3
                or tuple(kv.shape[1:]) != (1, MQA_DIM)
                or kv.element_size() != 1
                or not kv.is_contiguous()
                or kv.data_ptr() % 16
            ):
                reasons.append(f"layer {attn.layer_id}: MLA cache {tuple(kv.shape)}")
                break
        return reasons


# ---------------------------------------------------------------- the unit


def _covers(profile, rows: int, mode: str) -> Optional[str]:
    """TP4 (16 heads per rank) on SM103, 1-48 rows, every step mode (verify, draft
    extend, and draft decode on the draft group's one-stream bodies)."""
    if profile.heads != HEADS:
        return f"needs {HEADS} attention heads per rank (TP4), not {profile.heads}"
    if tuple(profile.sm) != SM:
        return (
            f"measured and served on SM103 only, not SM{profile.sm[0]}{profile.sm[1]}"
        )
    if not 1 <= rows <= MAX_ROWS:
        return f"serves 1-{MAX_ROWS} rows"
    return None


def _family(model, derived, profile) -> AttentionFamily:
    return AttentionFamily(model, derived, profile)


# the registry's view of this module (registry module docstring); one family per
# model build: the target's layers, or the NextN layer of the draft's build
CANDIDATES = (
    Candidate("ll.attention", "attention", STEPS, "ll.attention", _covers),
    Candidate("ll.attention", "draft_attention", STEPS, "ll.attention", _covers),
)
FAMILIES = {"ll.attention": _family}
