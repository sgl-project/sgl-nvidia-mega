# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The GLM-5.3 / GLM-5.2 NextN draft of the kernel-unit profiles (TP4, MTP topk 1): its forward composed from units.

Built by the draft spec (``draft.py``) for a TP4 profile with slot ``draft`` (``draft_step.FusedDraft`` serves TP8).
Each draft graph (draft decode at one row per request, draft extend at the verify rows) is composed from the draft's
step groups (``registry``: draft_front, draft_eh, draft_attention, draft_mlp, draft_top1) by ``units.PREFERENCE``.
Inside the draft-decode and draft-extend CUDA graphs:

* ``draft.front`` (slot ``draft``): the draft step's embedding gather + two eh RMSNorms as one kernel on the
  replicated embedding table (needs ``SGLANG_ENABLE_EMBED_REPLICATION=1``), bitwise equal to
  ``fused_eh_norm(F.embedding(...))``; the kernel does not depend on the TP size;
* ``ll.draft_eh`` (P6a, ``ll.draft``): eh_proj row-parallel, summed in the target's LL hop with the draft layer's
  input norm applied;
* ``draft_rowselect``: in the draft extend, the layer keeps only the ``select_index`` rows (one per request: the rows
  its LM head reads) after the post-attention norm (``draft_attention.select_rows``);
* the draft layer's attention on ``ll.attention`` / ``ll.sparse_mla_fd``; the LM head on ``ll.gemm``'s lm_head role.

The draft top-1 is SGLang's (E-D2: at TP4 its one-tray multimem gather is as fast as the TP8 draft's distributed
argmax or faster). Every eager step, every draft graph outside the profile's widths and every step nobody serves runs
the stock code. The draft is composed and its kernels are built when the target shares its embedding and LM head
(``set_embed_and_head``, before the KV pool is sized): the LM head's role is built on the target's head.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.distributed as dist
from sglang.srt.distributed.parallel_state import get_tp_group

from sgl_nvidia_mega.glm5_3 import HIDDEN, draft_attention, registry, runtime
from sgl_nvidia_mega.glm5_3.draft_step import row_select_supported, select_index_rows
from sgl_nvidia_mega.glm5_3.profile import Profile
from sgl_nvidia_mega.glm5_3.registry import Candidate, Decline
from sgl_nvidia_mega.glm5_3.runtime import DerivedWeights
from sgl_nvidia_mega.glm5_3.runtime.compile import compile_cached, cu_stream

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": ("Step",),
    "srt/distributed/communication_op.py": ("tensor_model_parallel_all_gather",),
    "srt/distributed/parallel_state.py": (
        "get_tp_group",
        "rank_in_group",
        "world_size",
        "device_group",
        "cpu_group",
        "group_name",
    ),
    "srt/model_executor/forward_batch_info.py": (
        "ForwardBatch",
        "forward_mode",
        "spec_info",
        "mm_input_embeds",
        "contains_mm_inputs",
    ),
    "kernels/ops/layernorm/fused_eh_norm.py": ("fused_eh_norm",),
    "srt/eplb/expert_distribution.py": (
        "get_global_expert_distribution_recorder",
        "disable_this_region",
    ),
    "srt/layers/attention/index_topk_share.py": (
        "IndexTopKShareState",
        "from_mtp_carry",
        "topk_indices",
        "update",
        "publish",
    ),
    "srt/models/deepseek_v2.py": ("mlp", "set_embed_and_head"),
    "srt/layers/layer_boundary/residual/batch.py": (
        "start",
        "set_written",
        "complete_output",
        "final_norm",
    ),
    "srt/models/deepseek_nextn.py": (
        "embed_tokens",
        "embed_input_ids",
        "enorm",
        "hnorm",
        "eh_proj",
        "decoder",
        "shared_head",
        "rot_weight",
    ),
    "srt/layers/vocab_parallel_embedding.py": (
        "enable_tp",
        "shard_indices",
        "num_added_elements",
        "org_vocab_start_index",
        "num_org_elements",
    ),
    "srt/utils/common.py": ("BumpAllocator",),
    "srt/layers/logits_processor.py": (
        "LogitsMetadata",
        "LogitsProcessor",
        "from_forward_batch",
        "draft_extend_select_index",
        "_get_logits",
        "_compute_lm_head",
        "_logits_gatherer",
        "_copy_logits_to_buffer",
        "logit_scale",
        "final_logit_softcapping",
        "do_tensor_parallel_all_gather",
        "do_tensor_parallel_all_gather_dp_attn",
        "use_attn_tp_group",
        "vocab_size",
    ),
    "srt/environ.py": ("envs", "SGLANG_ENABLE_EMBED_REPLICATION"),
    "srt/runtime_context.py": ("get_parallel", "get_spec"),
    "srt/arg_groups/fields/*.py": (
        "tp_size",
        "pp_size",
        "enable_dp_attention",
        "enable_dp_lm_head",
        "speculative_eagle_topk",
        "speculative_use_rejection_sampling",
    ),
}

logger = logging.getLogger(__name__)

# tensor-parallel sizes this composed draft serves (draft_step.FusedDraft serves TP8)
TP_SIZES = (4,)
# value table of this engine's eh-norm arithmetic (fused_eh_norm as SGLang JIT-compiles it)
EH_NORM_OPTIONS = dict(
    sum_rule="lane_fma_cta_tree",
    mean_rule="div_rn",
    rsqrt_ftz=False,
    apply_rule="x_r_w",
)
# the draft's step modes (SGLang's Step.mode of the draft graphs)
DECODE, EXTEND = "decode", "draft_extend"


def draft_rows(tokens: int, widths) -> Tuple[int, ...]:
    """The row counts of the draft graphs that pair with the target's verify ``widths`` at ``tokens`` per
    request: draft extend runs the verify rows, draft decode one row per request."""
    return tuple(sorted({rows for _, rows in draft_steps(tokens, widths)}))


def draft_steps(tokens: int, widths) -> Tuple[Tuple[str, int], ...]:
    """The (mode, rows) of the draft graphs that pair with the target's verify ``widths``."""
    if tokens < 1 or any(w % tokens for w in widths):
        raise ValueError(
            f"verify widths {tuple(widths)} are not multiples of {tokens} tokens"
        )
    return tuple((DECODE, w // tokens) for w in widths) + tuple(
        (EXTEND, w) for w in widths
    )


def mode_of(forward_batch: ForwardBatch) -> str:
    return DECODE if forward_batch.forward_mode.is_decode() else EXTEND


def eh_sum(group, eh_input: torch.Tensor, attention) -> Optional[torch.Tensor]:
    """The draft_eh step of a draft forward whose attention is ``attention`` (a step group, or None: the stock
    ``self_attn``). A ``draft_attention.PreparedInput`` (the sum already through the draft layer's input norm, the
    attention prepare skipped) only for a step group: the stock attention reads the attention inputs that the
    layer's prepare sets. None: SGLang's replicated eh_proj."""
    return group.eh_proj(eh_input, prepared=attention is not None)


# ------------------------------------------------------------------ the draft's step groups (stock bodies)


class DraftFront:
    """draft_front: the embedding gather + eh norms. Stock: SGLang's (``fused_eh_norm``)."""

    def front(self, ids, table, prev, enorm_w, hnorm_w, eps) -> Optional[torch.Tensor]:
        return None


class DraftEh:
    """draft_eh: ``eh_proj`` of the eh input. Stock: the NextN's replicated GEMM.

    ``prepared``: the draft attention that reads the result is a step group, which takes the
    draft layer's input already normed; a unit may then return a ``draft_attention.PreparedInput``
    (the sum with the input norm applied, the attention prepare skipped). False (the stock
    attention, which runs the layer's attention prepare itself): only the summed projection."""

    def eh_proj(
        self, eh_input: torch.Tensor, *, prepared: bool = False
    ) -> Optional[torch.Tensor]:
        return None


class DraftMlp:
    """draft_mlp: the rows the NextN MLP runs on (None: every row, the stock layer)."""

    def forward(self, forward_batch: ForwardBatch) -> Optional[torch.Tensor]:
        return None


class DraftTop1:
    """draft_top1: the draft top-1. Stock: SGLang's logits gather."""

    def top1(
        self, logits_shard, n_valid: int, vocab: int, extend: bool
    ) -> Optional[torch.Tensor]:
        return None


class RowSelect(DraftMlp):
    """``draft_rowselect``: the draft extend's MLP on the ``select_index`` rows only."""

    def forward(self, forward_batch: ForwardBatch) -> Optional[torch.Tensor]:
        return select_index_rows(forward_batch)


# group -> its stock step-group factory (registry.build)
DRAFT_STOCK = {
    "draft_front": DraftFront,
    "draft_eh": DraftEh,
    "draft_attention": draft_attention.DraftAttention,
    "draft_mlp": DraftMlp,
    "draft_top1": DraftTop1,
}


class ComposedDraft(DraftFront):
    """The NextN draft forward composed from the draft's units per (mode, rows). It is also the ``draft.front``
    family (the draft step's fused front)."""

    def __init__(self, profile: Optional[Profile] = None) -> None:
        self.profile = profile
        self.steps = (
            draft_steps(profile.tokens_per_request, profile.widths)
            if profile is not None
            else ()
        )
        # the draft graphs' row counts
        self.rows = (
            draft_rows(profile.tokens_per_request, profile.widths)
            if profile is not None
            else ()
        )
        self.armed = False
        self.rank, self.world = 0, 1
        self.derived = DerivedWeights()
        self._front = None
        self._nextn = None
        self.build: Optional[registry.Build] = None
        self._attn_agreed = {}  # (mode, rows) -> every rank serves the draft attention units there
        self._reported = set()
        self._key: Optional[Tuple[str, int]] = (
            None  # the (mode, rows) of the forward being captured
        )

    # ------------------------------------------------------------------ build

    def arm(self, nextn) -> None:
        """After the target's embedding / lm_head are shared: compose every draft graph from the units, build and
        compile what they take, rendezvous."""
        from sgl_nvidia_mega.glm5_3.sm103 import cutlass_supported

        ok, why = cutlass_supported()
        if not ok:
            logger.warning(
                "NVIDIA megakernel: GLM-5.3 / GLM-5.2 draft: %s; the draft runs "
                "SGLang's forward.",
                why,
            )
            return
        tp = get_tp_group()
        self.rank, self.world = tp.rank_in_group, tp.world_size
        self._nextn = nextn
        self._tp = tp
        self.dev = nextn.model.eh_proj.weight.device
        candidates, families = registry.collect(draft_modules())
        # this object is the draft.front family
        families = {**families, "draft.front": lambda model, derived, profile: self}
        self.build = registry.build(
            nextn,
            self.derived,
            self.profile,
            steps=self.steps,
            groups=DRAFT_STOCK,
            candidates=candidates,
            families=families,
            gather=runtime.tp_gather,
            rank=self.rank,
            what="NVIDIA megakernel: GLM-5.3 / GLM-5.2 draft",
        )
        self._arm_lm_head(nextn.lm_head)
        torch.cuda.synchronize(self.dev)
        dist.barrier(group=tp.cpu_group)
        self.armed = True
        self.derived.log(self.rank)
        if self.rank == 0:
            logger.info(
                "NVIDIA megakernel: GLM-5.3 / GLM-5.2 draft armed (TP%d): %s",
                self.world,
                "; ".join(
                    f"draft {'decode' if m == DECODE else 'extend'} {r} rows: {self._units(m, r)}"
                    for m, r in self.steps
                ),
            )

    def _arm_lm_head(self, lm_head) -> None:
        """The draft graphs' LM head rows (one per request: draft decode, and the draft extend's select_index rows)
        on ll.gemm's lm_head role (unit ``ll.gemm.lm_head``), built outside capture; the logits processor
        (``make_draft_logits_processor``) runs it there."""
        from sgl_nvidia_mega.glm5_3 import ll

        if self.profile is None:
            return
        tokens = self.profile.tokens_per_request
        rows = sorted({r if m == DECODE else r // tokens for m, r in self.steps})
        ll.build_linear(
            "lm_head", lm_head, rows, sm=self.profile.sm, sms=self.profile.sms
        )

    def _units(self, mode: str, rows: int) -> str:
        owners = self.build.compositions[(mode, rows)].owners
        names = []
        for group in DRAFT_STOCK:
            taken = list(
                dict.fromkeys(c.name for (g, _), c in owners.items() if g == group)
            )
            names.append(
                f"{group.split('_', 1)[1]} {'+'.join(taken) if taken else 'stock'}"
            )
        return ", ".join(names)

    def compile(self, widths) -> None:
        """The draft.front family protocol: its kernels compile per taken unit and width (``provider``)."""

    def provider(self, candidate, rows: int, mode: str):
        """The draft.front family protocol: compile the front (outside the capture, at arm) and serve it from
        this object."""
        from sgl_nvidia_mega.glm5_3.sm103.draft.api import compile_specs

        if candidate.name != "draft.front":
            raise ValueError(f"the draft.front family has no unit {candidate.name}")
        if self._front is None:
            self._front = _compile(
                *compile_specs(eh_norm=EH_NORM_OPTIONS)["embed_eh_norm"]
            )
        return self

    def engaged(self) -> bool:
        return self.armed and torch.cuda.is_current_stream_capturing()

    def serves(self, mode: str, rows: int) -> bool:
        """Whether the draft graph of (``mode``, ``rows``) runs the composed draft forward."""
        return self.build is not None and (mode, rows) in self.build.groups

    def front(self, ids, table, prev, enorm_w, hnorm_w, eps) -> Optional[torch.Tensor]:
        """[N, 2H] = cat(enorm(table[ids]), hnorm(prev)) as fused_eh_norm(F.embedding(...)), or None."""
        from cutlass import Float32, Int32, Int64

        n = ids.shape[0]
        if (
            not self.engaged()
            or n == 0
            or table.dtype != torch.bfloat16
            or table.shape[1] != HIDDEN
        ):
            return None
        out = torch.empty(n, 2 * HIDDEN, dtype=torch.bfloat16, device=ids.device)
        self._front(
            Int64(ids.data_ptr()),
            Int64(table.data_ptr()),
            Int64(table.stride(0)),
            Int64(table.shape[0]),
            Int64(prev.data_ptr()),
            Int64(prev.stride(0)),
            Int64(enorm_w.data_ptr()),
            Int64(hnorm_w.data_ptr()),
            Int64(out.data_ptr()),
            Int64(out.stride(0)),
            Float32(eps),
            Int32(n),
            cu_stream(),
        )
        return out

    # ------------------------------------------------------------------ the composed draft forward

    def groups(self, mode: str, rows: int):
        """group -> step-group object of the draft graph (``mode``, ``rows``)."""
        return self.build.groups[(mode, rows)]

    def attention_for(self, mode: str, rows: int, forward_batch: ForwardBatch, view):
        """The draft layer's attention step group at (``mode``, ``rows``), or None: the stock attention. None
        when no unit claims a draft_attention step there, or when at the first capture of the graph any rank
        reports a host-only fact its units' families do not serve (``step_reasons``; a host-only gather on the
        CPU group, legal inside the capture); ``view`` is the running forward's ForwardView."""
        comp = self.build.compositions[(mode, rows)]
        taken = [c for (g, _), c in comp.owners.items() if g == "draft_attention"]
        if not taken:
            return None
        key = (mode, rows)
        agreed = self._attn_agreed.get(key)
        if agreed is None:
            reasons = []
            for family in dict.fromkeys(self.build.families[c.family] for c in taken):
                step_reasons = getattr(family, "step_reasons", None)
                if step_reasons is not None:
                    reasons += list(step_reasons(self._nextn, forward_batch, view))
            gathered = runtime.tp_gather(reasons)
            agreed = self._attn_agreed[key] = not any(gathered)
            if self.rank == 0 and not agreed:
                logger.warning(
                    "NVIDIA megakernel: GLM-5.3 / GLM-5.2 draft attention stays stock: "
                    "draft %s graph, %d rows: %s",
                    "decode" if mode == DECODE else "extend",
                    rows,
                    runtime.per_rank(gathered),
                )
        return self.groups(mode, rows)["draft_attention"] if agreed else None

    def run(self, model, input_ids, positions, forward_batch: ForwardBatch, view=None):
        """The served draft forward of an engaged step: ``(hidden_states, logits_metadata)`` for the logits
        processor. With the row select the hidden states hold only the ``select_index`` rows, so the metadata carries no
        row selection (the processor must not select twice). ``view``: the running forward's ForwardView."""
        mode, rows = mode_of(forward_batch), int(positions.shape[0])
        self._key = (mode, rows)
        groups = self.groups(mode, rows)
        attention = self.attention_for(mode, rows, forward_batch, view)
        ffn_rows = groups["draft_mlp"].forward(forward_batch)
        hidden_states = self.forward_steps(
            model,
            input_ids,
            positions,
            forward_batch,
            attention,
            ffn_rows,
            groups,
            view,
        )
        if ffn_rows is None:
            return hidden_states, forward_batch
        from sglang.srt.layers.logits_processor import LogitsMetadata

        metadata = LogitsMetadata.from_forward_batch(forward_batch)
        metadata.draft_extend_select_index = None
        return hidden_states, metadata

    def forward_steps(
        self,
        model,
        input_ids,
        positions,
        forward_batch,
        attention=None,
        ffn_rows=None,
        groups=None,
        view=None,
    ):
        """``DeepseekModelNextN.forward`` (CUDA path) with the front and eh_proj of ``groups`` (stock unless a
        unit serves them), the decoder layer's attention through ``attention`` (a step group; None: the stock
        attention) on the running forward's ``view``, and with ``ffn_rows`` the MLP and the final norm on those
        rows only (the row select)."""
        from sglang.kernels.ops.layernorm.fused_eh_norm import fused_eh_norm
        from sglang.srt.eplb.expert_distribution import (
            get_global_expert_distribution_recorder,
        )
        from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState
        from sglang.srt.layers.layer_boundary.residual.batch import (
            complete_output,
            final_norm,
            set_written,
            start,
        )
        from sglang.srt.utils import BumpAllocator

        if groups is None:
            groups = {"draft_front": self, "draft_eh": self}
        # the zero fill stays the plain launch right before the front kernel (its ids / rows are read after its
        # grid-dependency wait, which orders it only after the immediately preceding grid)
        zero_allocator = BumpAllocator(
            buffer_size=2, dtype=torch.float32, device=input_ids.device
        )
        hidden_states = None
        if input_ids.shape[0] > 0:
            prev = forward_batch.spec_info.hidden_states
            eh_input = None
            # the fused front needs plain token ids (no multimodal embeddings)
            if (
                forward_batch.mm_input_embeds is None
                and not forward_batch.contains_mm_inputs()
                and not model.embed_tokens.enable_tp
            ):
                eh_input = groups["draft_front"].front(
                    input_ids,
                    model.embed_tokens.weight,
                    prev,
                    model.enorm.weight,
                    model.hnorm.weight,
                    model.enorm.variance_epsilon,
                )
            if eh_input is None:
                embeds = model.embed_input_ids(input_ids, forward_batch)
                eh_input = fused_eh_norm(
                    embeds,
                    prev,
                    model.enorm.weight,
                    model.hnorm.weight,
                    model.enorm.variance_epsilon,
                )
            hidden_states = eh_sum(groups["draft_eh"], eh_input, attention)
            if hidden_states is None:
                hidden_states = model.eh_proj(eh_input)
        else:
            hidden_states = model.embed_input_ids(input_ids, forward_batch)

        prepared = isinstance(hidden_states, draft_attention.PreparedInput)
        if prepared:
            # the eh_proj sum already applied the draft layer's input norm (P6a in the LL hop)
            set_written(hidden_states.residual, forward_batch)
            hidden_states = hidden_states.norm
        else:
            start(forward_batch)
        index_topk_share = IndexTopKShareState.from_mtp_carry(forward_batch)
        with get_global_expert_distribution_recorder().disable_this_region():
            if attention is None and ffn_rows is None and not prepared:
                hidden_states, topk_indices = model.decoder(
                    positions,
                    hidden_states,
                    forward_batch,
                    zero_allocator,
                    prev_topk_indices=index_topk_share.topk_indices,
                )
            else:
                hidden_states, topk_indices = draft_attention.decoder_layer_forward(
                    model.decoder,
                    attention,
                    positions,
                    hidden_states,
                    forward_batch,
                    zero_allocator,
                    prev_topk_indices=index_topk_share.topk_indices,
                    view=view,
                    ffn_rows=ffn_rows,
                    prepared=prepared,
                )
        hidden_states = complete_output(hidden_states, forward_batch)
        if not forward_batch.forward_mode.is_idle():
            hidden_states = final_norm(
                hidden_states, forward_batch, model.shared_head.norm
            )
        index_topk_share.update(topk_indices)
        index_topk_share.publish()
        return hidden_states

    def report(self) -> None:
        """Rank 0, once per captured draft graph: the units it runs (the engagement line)."""
        key, self._key = self._key, None
        if (
            key is None
            or key in self._reported
            or not torch.cuda.is_current_stream_capturing()
        ):
            return
        self._reported.add(key)
        if self.rank == 0:
            mode, rows = key
            attention = "stock" if self._attn_agreed.get(key) is False else None
            logger.info(
                "NVIDIA megakernel: GLM-5.3 / GLM-5.2 draft: draft %s graph, %d rows: %s%s",
                "decode" if mode == DECODE else "extend",
                rows,
                self._units(mode, rows),
                "" if attention is None else " (attention kept stock at capture)",
            )


def make_draft_logits_processor(config, fused_draft: ComposedDraft):
    """SGLang's logits processor with the LM head on ``ll.gemm``'s lm_head role at the rows ``ComposedDraft.arm``
    built (TGV), else SGLang's processor as is. The draft top-1 is SGLang's."""
    from sglang.srt.layers.logits_processor import LogitsProcessor

    from sgl_nvidia_mega.glm5_3 import ll

    gemm = ll.module("gemm")
    if gemm is None:
        return LogitsProcessor(config)

    class GlmDraftLogitsProcessor(gemm.LmHeadMixin, LogitsProcessor):
        pass

    return GlmDraftLogitsProcessor(config)


def _compile(kernel, constexprs, args):
    """``cute.compile(kernel, *args)`` through the fused path's compile cache."""
    import cutlass.cute as cute

    return compile_cached(kernel, constexprs, lambda: cute.compile(kernel, *args))


def check_draft_config(model) -> None:
    """Startup checks of the fused draft (an unsupported setting raises ``ValueError``)."""
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_parallel, get_spec

    problems = []

    def need(ok: bool, what: str) -> None:
        if not ok:
            problems.append(what)

    cfg, spec, par = model.config, get_spec(), get_parallel()
    need(
        torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 3),
        "an SM103 GPU",
    )
    need(getattr(cfg, "hidden_size", None) == HIDDEN, f"hidden size {HIDDEN} (GLM-5)")
    need(
        par.tp_size in TP_SIZES,
        f"tensor parallel size {' or '.join(map(str, TP_SIZES))}",
    )
    need(getattr(par, "pp_size", 1) == 1, "pipeline parallel size 1")
    need(not getattr(par, "enable_dp_attention", False), "no DP attention")
    need(not getattr(par, "enable_dp_lm_head", False), "no DP lm_head")
    need(getattr(spec, "speculative_eagle_topk", 1) == 1, "MTP / EAGLE with topk 1")
    need(
        not getattr(spec, "speculative_use_rejection_sampling", False),
        "greedy draft acceptance (no draft probs)",
    )
    need(
        envs.SGLANG_ENABLE_EMBED_REPLICATION.get(),
        "SGLANG_ENABLE_EMBED_REPLICATION=1 (replicated draft table)",
    )
    need(
        getattr(model.model, "rot_weight", None) is None,
        "no rotation of the previous hidden states (the fused front reads them as they are)",
    )
    if problems:
        raise ValueError(
            "NVIDIA megakernel: GLM-5.3 / GLM-5.2: the TP4 draft needs: "
            + "; ".join(problems)
            + "."
        )


def create_composed_draft(model, profile: Profile) -> ComposedDraft:
    check_draft_config(model)
    return ComposedDraft(profile)


def draft_modules() -> tuple:
    """The modules whose units serve the draft's step groups."""
    from sgl_nvidia_mega.glm5_3 import ll

    return (sys.modules[__name__], draft_attention, *ll.modules().values())


# ------------------------------------------------------------------ the units (registry module docstring)


def _covers_front(profile: Profile, rows: int, mode: str) -> Optional[str]:
    if profile.tp not in TP_SIZES:
        return f"TP{profile.tp} (the fused draft serves TP {TP_SIZES})"
    return None


def _covers_row_select(profile: Profile, rows: int, mode: str) -> Optional[str]:
    if mode != EXTEND:
        return "only the draft extend's LM head reads one row per request"
    return None


class _RowSelectFamily:
    """The draft_rowselect unit's family: needs the stock NextN MoE (a fact of the loaded model)."""

    def __init__(self, model, derived, profile) -> None:
        self.supported = row_select_supported(model.model.decoder.mlp)

    def compile(self, widths) -> None:
        pass

    def provider(self, candidate, rows: int, mode: str) -> RowSelect:
        if not self.supported:
            raise Decline(
                "the NextN MLP is not SGLang's DeepseekV2MoE without an all-to-all"
            )
        return RowSelect()


def _armed_draft(model, derived, profile):
    raise RuntimeError(
        "the draft.front family is the armed ComposedDraft (ComposedDraft.arm passes it)"
    )


CANDIDATES = (
    Candidate("draft.front", "draft_front", ("front",), "draft.front", _covers_front),
    Candidate(
        "draft_rowselect",
        "draft_mlp",
        ("forward",),
        "draft.rowselect",
        _covers_row_select,
    ),
)
FAMILIES = {"draft.front": _armed_draft, "draft.rowselect": _RowSelectFamily}


# ------------------------------------------------------------------ the draft spec's build (TP4 profiles)


class ComposedBuiltDraft:
    """The draft spec's built object at a kernel-unit profile: the draft graphs that pair with the target's verify
    widths (draft extend at the verify rows, draft decode at one row per request) run the composed draft."""

    # the draft offers no speculative entries or graph variants (BuiltMegakernel)
    draft_sampler = None
    draft_loop = None
    draft_extend = None
    begin_draft_round = None
    graph_variants = None
    speculative_runtime_hooks = None

    def __init__(self, model, profile: Profile):
        self.model = model
        self.profile = profile
        self.fused = create_composed_draft(model, profile)
        self.widths = self.fused.rows
        model.logits_processor = make_draft_logits_processor(model.config, self.fused)
        share = model.set_embed_and_head

        def set_embed_and_head(embed, head):
            share(embed, head)
            self.fused.arm(model)

        # an instance attribute: the class and every other instance keep theirs
        model.set_embed_and_head = set_embed_and_head

    def summary(self) -> str:
        return (
            f"TP{self.profile.tp} NextN draft composed from the draft units (logged "
            "when armed)"
        )

    def plan(self, input_ids, forward_batch, pp_proxy_tensors=None):
        """A draft step for the captured draft graphs the composed draft serves (draft decode at one row per
        request is off the width grid: SGLang's dispatch does not check widths); None elsewhere (SGLang's
        draft forward)."""
        try:
            from sglang.kernels.plugin_kernels.nvidia.spec import Step
        except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
            from sglang.srt.megakernels.spec import Step

        if pp_proxy_tensors is not None or not self.fused.engaged():
            return None
        mode, rows = mode_of(forward_batch), int(input_ids.shape[0])
        # the draft graphs outside the profile's widths run SGLang's draft forward
        if not self.fused.serves(mode, rows):
            return None
        return Step(
            mode="draft_decode" if mode == DECODE else "draft_extend", rows=rows
        )

    def forward(
        self,
        step,
        original,
        input_ids,
        positions,
        forward_batch,
        pp_proxy_tensors=None,
    ):
        """The composed draft forward of a captured draft step, then SGLang's logits processor (with the LM head
        on ``ll.gemm``'s lm_head role at the rows it was built for)."""
        hidden_states, logits_metadata = self.fused.run(
            self.model.model, input_ids, positions, forward_batch, step.view
        )
        out = self.model.logits_processor(
            input_ids, hidden_states, self.model.lm_head, logits_metadata
        )
        self.fused.report()
        return out


def build(model, profile: Profile) -> ComposedBuiltDraft:
    return ComposedBuiltDraft(model, profile)
