# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""What the draft entries share, built once by the first entry factory SGLang calls
(before it captures the draft graphs, every rank in the same order): the draft layer,
the attention's buffers, and each requested draft slot's build."""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/qwen3_5_mtp.py": (
        "model",
        "fc",
        "pre_fc_norm_embedding",
        "pre_fc_norm_hidden",
        "lm_head",
        "logits_processor",
    ),
    "srt/layers/logits_processor.py": ("logit_scale", "vocab_size"),
    "srt/layers/vocab_parallel_embedding.py": (
        "shard_indices",
        "org_vocab_start_index",
        "org_vocab_end_index",
        "added_vocab_start_index",
        "added_vocab_end_index",
    ),
    "srt/models/qwen3_5.py": (
        "layers",
        "start_layer",
        "end_layer",
        "norm",
        "embed_tokens",
        "input_layernorm",
        "post_attention_layernorm",
        "mlp",
        "attn",
    ),
    "srt/kernel_plugins/nvidia/eagle_binding.py|srt/megakernels/eagle_binding.py": (
        "tp_rank",
        "tp_size",
        "tp_device_group",
        "tp_cpu_group",
        "page_size",
    ),
}


class Parts:
    """The draft model's modules the entries run, and the slot builds."""

    def __init__(self, draft, layer):
        self.draft = draft
        self.lm = draft.lm  # Qwen3_5ForCausalLMMTP
        self.model = draft.lm.model  # its one-layer Qwen3_5ForCausalLM
        self.layer = layer  # the MTP decoder layer (full attention + MoE)
        self.binding = draft.binding
        self.slots = draft.slots
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.attention = None  # draft.attention.Attention
        self.moe: Optional[Any] = None  # draft.moe.DraftMoe (slot draft_moe)
        self.hop: Optional[Any] = None  # draft.hop.DraftHop (slot draft_hop)
        self.mkd: Optional[Any] = None  # draft.xchg.MkdBoundary (slot draft_xchg)
        # sm103.draft.top1.Top1Exchange: the multi-request steps' sharded top-1 (slot
        # draft_multi)
        self.top1x: Optional[Any] = None
        # sm103.draft.embed.EmbedExchange: those steps' embedding rows (slot draft_embed)
        self.embedx: Optional[Any] = None
        # (entry, bs) pairs whose capture logged the engagement line
        self.engaged: set = set()
        # draft.moe_mkm.DraftMkm: compatible quantized draft experts on the target path
        self.moe_mkm: Optional[Any] = None

    @classmethod
    def build(cls, draft) -> Parts | None:
        from sgl_nvidia_mega.qwen3_5.draft.attention import Attention
        from sgl_nvidia_mega.qwen3_5.runtime import unavailable_slots

        m = draft.lm.model
        layers = [m.layers[i] for i in range(m.start_layer, m.end_layer)]
        if len(layers) != 1:
            unavailable_slots(
                draft.slots, f"the MTP draft has {len(layers)} layers, not 1"
            )
            return None
        if getattr(draft, "context_len", 0) is None:
            unavailable_slots(draft.slots, "the draft's HF config has no context_len")
            return None
        parts = cls(draft, layers[0])
        try:
            parts.attention = Attention(parts)
        except ImportError as exc:
            unavailable_slots(
                draft.slots, f"draft attention dependency unavailable: {exc}"
            )
            return None
        slots = parts.slots
        if "draft_fin" in slots:
            from sgl_nvidia_mega.qwen3_5.draft import finalize

            reason = finalize.enable(parts.layer.mlp)
            if reason:
                unavailable_slots("draft_fin", reason)
                slots = parts.slots = slots - {"draft_fin"}
        if "draft_moe" in slots:
            parts.moe = parts._build_moe()
            if parts.moe is None:
                slots = parts.slots = slots - {"draft_moe"}
        if "draft_hop" in slots:
            parts.hop = parts._build_hop()
            if parts.hop is None:
                slots = parts.slots = slots - {"draft_hop", "draft_fin"}
        if "draft_xchg" in slots:
            parts.mkd = parts._build_mkd()
        if "draft_multi" in slots:
            parts.top1x = parts._build_top1x()
            if parts.top1x is None:
                slots = parts.slots = slots - {"draft_multi", "draft_embed"}
        if "draft_embed" in slots:
            parts.embedx = parts._build_embedx()
            if parts.embedx is None:
                slots = parts.slots = slots - {"draft_embed"}
        if "draft_moe" in slots:
            from sgl_nvidia_mega.qwen3_5.draft import moe_mkm

            parts.moe_mkm, why = moe_mkm.DraftMkm.build(parts.layer.mlp)
            if parts.moe_mkm is None:
                logger.info(
                    "NVIDIA megakernel: draft MoE target path unavailable (%s)", why
                )
        logger.info(
            "NVIDIA megakernel: Qwen3.5 MTP draft parts built (slots %s; MoE %s, hop "
            "%s, exchange %s)",
            sorted(slots),
            (
                "single launch at one row, target path from two rows"
                if parts.moe_mkm is not None
                else "single launch"
                if parts.moe is not None
                else "SGLang's"
            ),
            "draft hop" if parts.hop is not None else "SGLang's",
            "MKD" if parts.mkd is not None else "SGLang's",
        )
        return parts

    def _build_moe(self):
        from sgl_nvidia_mega.qwen3_5.draft.moe import DraftMoe

        return DraftMoe.build(self.layer.mlp, self.device.index)

    def _build_hop(self):
        from sgl_nvidia_mega.qwen3_5.draft.hop import GEMV_O_PROJ, DraftHop

        tp = self.binding.tp_size
        hop, why = DraftHop.build(
            tp_size=tp,
            deadline_s=self.draft.deadline_s,
            o_proj_k=(
                int(self.layer.o_proj.weight.shape[1])
                if GEMV_O_PROJ.get(tp, False)
                else None
            ),
            early=True,
            early_head=True,
        )
        if hop is None:
            from sgl_nvidia_mega.qwen3_5.runtime import unavailable_slots

            unavailable_slots("draft_hop", why)
            return None
        logger.info(
            "NVIDIA megakernel: Qwen3.5 MTP draft hop releases at entry: the "
            "single-launch MoE %s, the draft head %s",
            hop.early,
            hop.early_head,
        )
        return hop

    def _build_mkd(self):
        from sgl_nvidia_mega.qwen3_5.draft.xchg import MkdBoundary

        lm, b = self.lm, self.binding
        embed = self.model.embed_tokens
        processor = lm.logits_processor
        # what 2.2's boundary assumed of the draft head and MkdBoundary cannot see
        refused = [
            why
            for bad, why in (
                (lm.config.tie_word_embeddings, "tied embeddings"),
                (processor.logit_scale is not None, "logit scaling"),
                (
                    getattr(processor, "final_logit_softcapping", None) is not None,
                    "logit softcapping",
                ),
            )
            if bad
        ]
        mkd = None
        if not refused:
            mkd = MkdBoundary.maybe_create(
                vocab_size=processor.vocab_size,
                fc_weight=lm.fc.weight,
                norm_embedding=lm.pre_fc_norm_embedding,
                norm_hidden=lm.pre_fc_norm_hidden,
                embed_shard=embed.weight,
                vocab_start=embed.shard_indices.org_vocab_start_index,
                lm_head_shard=lm.lm_head.weight,
                embed=embed,
                rank=b.tp_rank,
                world=b.tp_size,
                device_group=b.tp_device_group,
                cpu_group=b.tp_cpu_group,
                num_steps=b.num_steps,
            )
        if mkd is None:
            from sgl_nvidia_mega.qwen3_5.runtime import unavailable_slots

            unavailable_slots(
                "draft_xchg", "; ".join(refused) or "see the MKD draft boundary warning"
            )
        return mkd

    def _build_top1x(self):
        from sgl_nvidia_mega.qwen3_5.sm103.draft.top1 import Top1Exchange

        lm, b = self.lm, self.binding
        rows = int(lm.lm_head.weight.shape[0])
        start = int(lm.lm_head.shard_indices.org_vocab_start_index)
        vocab = int(lm.logits_processor.vocab_size)
        # the shard layout the all-gathered logits have: rank-major, no padding rows
        reason = ""
        if start != b.tp_rank * rows or rows * b.tp_size != vocab:
            reason = (
                "the LM head's "
                f"{rows}-row shard at {start} is not rank {b.tp_rank}'s slice of "
                f"{vocab} ids"
            )
        weight = lm.lm_head.weight
        if (
            weight.dtype != torch.bfloat16
            or weight.dim() != 2
            or not weight.is_contiguous()
        ):
            reason = f"LM head {weight.dtype} {tuple(weight.shape)}; needs a contiguous BF16 shard"
        from sgl_nvidia_mega.qwen3_5.runtime import slot_supported

        if not slot_supported(
            "draft_multi", reason, cpu_group=getattr(b, "tp_cpu_group", None)
        ):
            return None
        from sgl_nvidia_mega.qwen3_5.runtime import SlotUnavailable, unavailable_slots

        try:
            return Top1Exchange(
                world=b.tp_size,
                rank=b.tp_rank,
                vocab_local=rows,
                vocab_start=start,
                vocab_size=vocab,
                group=b.tp_device_group,
                cpu_group=b.tp_cpu_group,
                device=self.device,
            )
        except SlotUnavailable as exc:
            unavailable_slots("draft_multi", exc.reason)
            return None

    def _build_embedx(self):
        from sgl_nvidia_mega.qwen3_5.sm103.draft.embed import EmbedExchange

        b = self.binding
        embed = self.model.embed_tokens
        shard = embed.shard_indices
        reason = ""
        if shard.added_vocab_end_index != shard.added_vocab_start_index:
            reason = "the embedding has added vocabulary"
        table = embed.weight.data
        local = int(shard.org_vocab_end_index - shard.org_vocab_start_index)
        if (
            table.dtype != torch.bfloat16
            or table.dim() != 2
            or not table.is_contiguous()
            or table.shape[1] % 2
            or table.shape[0] < local
        ):
            reason = f"embedding table {table.dtype} {tuple(table.shape)}, local vocabulary {local}; needs contiguous BF16 rows with an even hidden size"
        from sgl_nvidia_mega.qwen3_5.runtime import slot_supported

        if not slot_supported(
            "draft_embed", reason, cpu_group=getattr(b, "tp_cpu_group", None)
        ):
            return None
        from sgl_nvidia_mega.qwen3_5.runtime import SlotUnavailable, unavailable_slots

        try:
            return EmbedExchange(
                world=b.tp_size,
                table=embed.weight.data,
                vocab_start=int(shard.org_vocab_start_index),
                vocab_local=int(
                    shard.org_vocab_end_index - shard.org_vocab_start_index
                ),
                group=b.tp_device_group,
                cpu_group=b.tp_cpu_group,
                device=self.device,
            )
        except SlotUnavailable as exc:
            unavailable_slots("draft_embed", exc.reason)
            return None

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """The embedding rows of ``token_ids``: slot draft_embed's exchange where it is
        built and the rows fit it, else SGLang's vocabulary-parallel embedding."""
        from sgl_nvidia_mega.qwen3_5.sm103.draft.embed import MAX_ROWS

        if self.embedx is not None and token_ids.shape[0] <= MAX_ROWS:
            return self.embedx(token_ids)
        return self.model.embed_tokens(token_ids)

    def note_engaged(self, entry: str, bs: int) -> None:
        """Log once per (entry, bs) captured: the engagement line every rank prints."""
        if torch.cuda.is_current_stream_capturing() and (entry, bs) not in self.engaged:
            self.engaged.add((entry, bs))
            logger.info(
                "NVIDIA megakernel: Qwen3.5 MTP draft %s captured for batch size %d "
                "(slots %s)",
                entry,
                bs,
                sorted(self.slots),
            )
