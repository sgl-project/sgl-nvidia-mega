# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team
"""Attention of the GLM-5 DSA NextN draft layer in steps, and the draft layer's
forward with its attention in steps.

``DraftAttention`` is the verify step group (``providers.AttentionDecode``): a
draft forward takes its branches in DRAFT_EXTEND_V2, and in its DECODE passes the
sparse MLA of the DSA backend's ``forward_decode`` (``sparse_mla(decode=True)``).
The stock DECODE prologue also overlaps ``q_b_proj`` with the indexer on a side
stream; the served draft runs the fused prologue instead, and the stock bodies
compute the same values on one stream. ``decoder_layer_forward`` is
``DeepseekV2DecoderLayer.forward`` (main 44b44774) with the attention called
through a step group; on the stock bodies it is bitwise equal to the stock draft
layer (the draft replica test checks it in both modes).

``DraftFusedAttention`` serves the attention family's MLA prologue in every draft
pass and its q_a / kv_a GEMM (with the indexer key / head-gate projection) in the
draft extend at up to 8 rows; the indexer query, the indexer, the sparse MLA and
the value up-projection stay stock. The draft layer has its own weights and KV
pool, hence its own ``AttentionGlue``; compiled forms are shared with the target
through the compile cache.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional

import torch

from sgl_nvidia_mega.glm_dsa.attention import (
    AttentionGlue,
    FusedAttention,
    kernel_rows,
)
from sgl_nvidia_mega.glm_dsa.attention.step import AttentionDecode, StepContext
from sgl_nvidia_mega.glm_dsa.runtime.executor import DerivedWeights


@dataclass
class DraftStepContext(StepContext):
    """``StepContext`` of one draft forward."""

    # Whether this forward's draft layer runs its own indexer (the NextN layer
    # does whenever no top-k carry is passed in: should_run_indexer).
    run_indexer: Optional[bool] = None


class DraftAttention(AttentionDecode):
    """The verify attention steps, plus the DECODE-mode sparse MLA of a draft
    forward. Kernel overrides of the draft layer subclass this group."""

    def forward(self, layer, x, prev_topk_indices, step: DraftStepContext):
        step.run_indexer = layer.self_attn.should_run_indexer(prev_topk_indices)
        return super().forward(layer, x, prev_topk_indices, step)

    def check_mode(self, attn, mode) -> None:
        # Stock tests is_decode_or_idle(); IDLE batches need DP attention, which
        # the fused draft refuses, so is_decode() selects the same branches.
        if mode.is_decode():
            _check_decode_branches(attn)
        else:
            super().check_mode(attn, mode)

    def sparse_mla(self, attn, q_fp8, topk_indices, step: DraftStepContext):
        decode = step.forward_batch.forward_mode.is_decode()
        return super().sparse_mla(attn, q_fp8, topk_indices, step, decode)


def _decode_problems(attn=None) -> List[str]:
    """The configurations the DECODE copy (and the draft prologue) do not cover."""
    from sglang.srt.model_executor.forward_context import get_attn_backend
    from sglang.srt.model_executor.runner_utils.capture_mode import (
        get_capture_attention_variant,
    )

    backend = get_attn_backend()
    variant = get_capture_attention_variant()
    problems = [] if variant is None else [f"capture attention variant {variant!r}"]
    if getattr(backend, "dcp_size", 1) > 1:
        problems.append("decode context parallelism")
    if not getattr(backend, "use_fused_topk", False):
        problems.append("no fused top-k page table")
    if attn is not None and (attn.q_lora_rank is None or not attn.use_dsa):
        problems.append("not a DSA attention with q_lora")
    return problems


def _check_decode_branches(attn) -> None:
    """Any other configuration is a programming error (the fused draft engages
    only on this one)."""
    problems = _decode_problems(attn)
    if problems:
        raise RuntimeError(
            "DraftAttention DECODE steps do not copy this configuration: "
            + "; ".join(problems)
        )


class DraftFusedAttention(FusedAttention, DraftAttention):
    """Draft attention with the MLA prologue on the family's kernel."""

    # The draft rule: these steps stay stock in the draft layer.
    index_q = AttentionDecode.index_q
    sparse_mla = DraftAttention.sparse_mla
    uv_proj = AttentionDecode.uv_proj

    def __init__(self, glue: AttentionGlue, rows: int) -> None:
        super().__init__(glue, rows)
        # qkv_a (per forward): the family's GEMM in the draft extend, else the
        # stock fused-A GEMM (+ the indexer key / head-gate projection)
        self._qa_extend, self.qa = self.qa, None

    def runs_indexer(self, attn, step) -> bool:
        # The draft layer runs its own indexer in the forwards without a top-k
        # carry (DraftAttention.forward), although it is marked skip_topk.
        return bool(step.run_indexer)

    def qkv_a(self, attn, x, step) -> torch.Tensor:
        decode = step.forward_batch.forward_mode.is_decode()
        self.qa = None if decode else self._qa_extend
        return super().qkv_a(attn, x, step)


class DraftAttentionGlue(AttentionGlue):
    """The draft layer's packed attention weights (owned here), compiled forms
    and step groups per row count."""

    def __init__(self, nextn) -> None:
        self.derived = DerivedWeights()
        super().__init__(nextn, self.derived, layers=[nextn.model.decoder])

    def step_reasons(self, nextn, forward_batch) -> List[str]:
        """Host-only facts the draft prologue needs (read at the first capture of
        each draft graph; any reason on any rank keeps it stock)."""
        return super().step_reasons(nextn, forward_batch) + _decode_problems()

    def for_rows(self, rows: int) -> Optional[DraftAttention]:
        if f"prologue.{kernel_rows(rows)}" not in self.fn:
            return None
        return DraftFusedAttention(self, rows)


def decoder_layer_forward(
    layer,
    attention: AttentionDecode,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch,
    zero_allocator,
    prev_topk_indices: Optional[torch.Tensor] = None,
):
    """``DeepseekV2DecoderLayer.forward`` (main 44b44774) for the draft layer,
    with ``self.self_attn(...)`` replaced by the step group's ``forward`` and
    ``o_proj`` (``forward_absorb_core`` ends with ``self.o_proj``). Returns
    ``(hidden_states, topk_indices)`` as the stock layer does."""
    from sglang.srt.layers.layer_boundary import get_attn_tp_context
    from sglang.srt.layers.layer_boundary.residual import access as residual_access
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

    hidden_states_orig = residual_access.buffer(hidden_states)
    # The NextN forward passes no captured outputs, GEMM-output allocator or
    # next full-attention layer (its prefetch is then a no-op).
    hidden_states = layer.attn_boundary.prepare(
        hidden_states,
        forward_batch,
        capture_gathered=None,
        quant_format=layer._resolve_gfx95_quant_format(),
    )
    step = DraftStepContext(
        forward_batch=forward_batch,
        positions=positions,
        zero_allocator=zero_allocator,
    )
    # The step group enters the attention's decode-TP context itself and clears
    # the attention inputs; o_proj then runs in a second entry of that context.
    # Stock runs both in one entry and clears after o_proj. Equivalent here: the
    # context only acts with decode context parallelism (refused), and o_proj
    # reads no attention input.
    o_input, topk_indices = attention.forward(
        layer, hidden_states, prev_topk_indices, step
    )
    with layer.self_attn.maybe_use_decode_attn_tp(forward_batch):
        hidden_states, _ = layer.self_attn.o_proj(o_input)
    get_attn_tp_context().clear_attn_inputs()

    hidden_states = layer.attn_boundary.finish(hidden_states, forward_batch)
    hidden_states = layer.ffn_boundary.prepare(hidden_states, forward_batch)

    if (
        isinstance(layer.mlp, DeepseekV2MoE)
        and not layer.mlp.experts.moe_runner_config.inplace
        and not torch.compiler.is_compiling()
        and isinstance(hidden_states_orig, torch.Tensor)
    ):
        from sglang.srt.layers.moe.moe_runner.base import moe_output_buffer_ctx

        mlp_ctx = moe_output_buffer_ctx(hidden_states_orig)
    else:
        mlp_ctx = nullcontext()

    with layer.ffn_boundary.exit(forward_batch) as ffn_exit, mlp_ctx:
        hidden_states = layer.mlp(hidden_states, forward_batch, None)
    hidden_states = ffn_exit.finish(hidden_states)
    return hidden_states, topk_indices
