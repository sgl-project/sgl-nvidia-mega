# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/deepseek_v2.py (DeepseekV2DecoderLayer.forward),
# python/sglang/srt/layers/attention/dsa_backend.py (DeepseekSparseAttnBackend.forward_decode).
"""Attention of the GLM-5.3 NextN draft layer in steps, and the draft layer's
forward with its attention in steps.

``DraftAttention`` is the verify step group (``attention.step.AttentionStep``): a
draft forward takes its branches in DRAFT_EXTEND_V2, and in its DECODE passes the
sparse MLA of the DSA backend's ``forward_decode`` (the full cache lengths, and no DP
padding rows: the draft spec refuses DP attention, so a DECODE pass whose rows differ
from the attention metadata's batch is a refusal).
The stock DECODE prologue also overlaps ``q_b_proj`` with the indexer on a side
stream; the served draft runs the fused prologue instead, and the stock bodies
compute the same values on one stream. ``decoder_layer_forward`` is
``DeepseekV2DecoderLayer.forward`` (main af856717) with the attention called
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

from sgl_nvidia_mega.glm5_3.attention.fused import (
    AttentionGlue,
    FusedAttention,
    kernel_rows,
)
from sgl_nvidia_mega.glm5_3.attention.step import AttentionStep, StepContext
from sgl_nvidia_mega.glm5_3.runtime import DerivedWeights

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/binding.py": (
        "dsa",
        "dcp_size",
        "use_fused_topk",
    ),
    "srt/model_executor/runner_utils/capture_mode.py": (
        "get_capture_attention_variant",
    ),
    "srt/layers/layer_boundary/__init__.py": ("get_attn_tp_context",),
    "srt/layers/layer_boundary/adapters/attention.py": ("clear_attn_inputs",),
    "srt/layers/layer_boundary/residual/access.py": ("buffer",),
    "srt/layers/moe/moe_runner/base.py": ("moe_output_buffer_ctx", "inplace"),
    "srt/layers/moe/fused_moe_triton/layer.py": ("moe_runner_config",),
    # the replaced model's modules (and the residual module as the layer imports it)
    "srt/models/deepseek_v2.py": (
        "residual_access",
        "DeepseekV2MoE",
        "attn_boundary",
        "ffn_boundary",
        "_resolve_gfx95_quant_format",
        "maybe_use_decode_attn_tp",
        "self_attn",
        "o_proj",
        "mlp",
        "experts",
        "q_lora_rank",
        "use_dsa",
    ),
    "srt/models/deepseek_common/attention_forward_methods/forward_mla.py": (
        "should_run_indexer",
    ),
    "srt/layers/attention/dsa_backend.py": ("cache_seqlens_int32",),
    "srt/layers/layer_boundary/*.py": ("prepare", "finish", "exit"),
}

# The SGLang function bodies this module's stock steps copy, by the digest of their
# code (test/unit/test_contract.py pins them)
SGLANG_MIRRORS = {
    "srt/models/deepseek_v2.py:DeepseekV2DecoderLayer.forward": "228c19b34f667210",
    # before and after sgl-project/sglang#44322 (a ROCm decode context parallel
    # branch) and sgl-project/sglang#46990 (the KV through the pool's layer router:
    # on the DSA pool, the same set_mla_kv_buffer and get_key_buffer calls)
    "srt/layers/attention/dsa_backend.py:DeepseekSparseAttnBackend.forward_decode": (
        "52e50cfcdda18bea",
        "35f75d4c546f4c15",
    ),
}


@dataclass
class DraftStepContext(StepContext):
    """``StepContext`` of one draft forward."""

    # Whether this forward's draft layer runs its own indexer (the NextN layer
    # does whenever no top-k carry is passed in: should_run_indexer).
    run_indexer: Optional[bool] = None


class DraftAttention(AttentionStep):
    """The verify attention steps, plus the DECODE-mode sparse MLA of a draft
    forward. Kernel overrides of the draft layer subclass this group."""

    def forward(self, layer, x, prev_topk_indices, step: DraftStepContext):
        step.run_indexer = layer.self_attn.should_run_indexer(prev_topk_indices)
        if step.forward_batch.forward_mode.is_decode():
            # Stock tests is_decode_or_idle(); IDLE batches need DP attention, which
            # the draft spec refuses, so is_decode() selects the same branches.
            _check_decode_branches(layer.self_attn, step.view)
        return super().forward(layer, x, prev_topk_indices, step)

    def check_mode(self, attn, mode) -> None:
        # the verify branches in DRAFT_EXTEND_V2; a DECODE pass was checked in
        # forward, on the step's view
        if not (mode.is_decode() or mode.is_draft_extend_v2()):
            raise RuntimeError(
                f"DraftAttention serves draft-extend-v2 and decode forwards only "
                f"(got {mode})"
            )

    def sparse_mla_seq_lens(self, metadata, step: DraftStepContext, rows: int):
        """DECODE (forward_decode, is_prefill False): the full cache lengths, and the
        rows equal to the metadata's batch (SGLang trims eager DP padding rows, which
        the draft spec's refusal of DP attention rules out)."""
        if not step.forward_batch.forward_mode.is_decode():
            return super().sparse_mla_seq_lens(metadata, step, rows)
        seq_lens = metadata.cache_seqlens_int32
        if seq_lens.shape[0] != rows:
            raise RuntimeError(
                f"draft decode: {rows} query rows for an attention batch of "
                f"{seq_lens.shape[0]} (DP padding rows are not served)"
            )
        return seq_lens


def _decode_problems(view, attn=None) -> List[str]:
    """The configurations the DECODE copy (and the draft prologue) do not cover;
    ``view`` is the running forward's ForwardView."""
    from sglang.srt.model_executor.runner_utils.capture_mode import (
        get_capture_attention_variant,
    )

    dsa = view.dsa
    variant = get_capture_attention_variant()
    problems = [] if variant is None else [f"capture attention variant {variant!r}"]
    if dsa is not None and dsa.dcp_size > 1:
        problems.append("decode context parallelism")
    if dsa is None or not dsa.use_fused_topk:
        problems.append("no fused top-k page table")
    if attn is not None and (attn.q_lora_rank is None or not attn.use_dsa):
        problems.append("not a DSA attention with q_lora")
    return problems


def _check_decode_branches(attn, view) -> None:
    """Any other configuration is a programming error (the draft step engages
    only on this one)."""
    problems = _decode_problems(view, attn)
    if problems:
        raise RuntimeError(
            "DraftAttention DECODE steps do not copy this configuration: "
            + "; ".join(problems)
        )


class DraftFusedAttention(FusedAttention, DraftAttention):
    """Draft attention with the MLA prologue on the family's kernel."""

    # The draft rule: these steps stay stock in the draft layer (the sparse MLA
    # with the draft's cache lengths, DraftAttention.sparse_mla_seq_lens).
    index_q = AttentionStep.index_q
    sparse_mla = AttentionStep.sparse_mla
    uv_proj = AttentionStep.uv_proj

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

    def step_reasons(self, nextn, forward_batch, view) -> List[str]:
        """Host-only facts the draft prologue needs (read at the first capture of
        each draft graph; any reason on any rank keeps it stock)."""
        return super().step_reasons(nextn, forward_batch, view) + _decode_problems(view)

    def for_rows(self, rows: int) -> Optional[DraftAttention]:
        if f"prologue.{kernel_rows(rows)}" not in self.fn:
            return None
        return DraftFusedAttention(self, rows)


def decoder_layer_forward(
    layer,
    attention: AttentionStep,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch,
    zero_allocator,
    prev_topk_indices: Optional[torch.Tensor] = None,
    view=None,
):
    """``DeepseekV2DecoderLayer.forward`` (main af856717) for the draft layer,
    with ``self.self_attn(...)`` replaced by the step group's ``forward`` and
    ``o_proj`` (``forward_absorb_core`` ends with ``self.o_proj``); ``view`` is
    the running forward's ForwardView. Returns ``(hidden_states, topk_indices)``
    as the stock layer does."""
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
        view=view,
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
