# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/deepseek_nextn.py (DeepseekModelNextN.forward),
# python/sglang/srt/layers/logits_processor.py (LogitsProcessor.forward and _get_logits: the decode / draft-extend
# path and its TP logits gather).
"""The GLM-5.3 NextN draft step (slot ``draft``; SM103, TP8, MTP topk 1): CuTe DSL
kernels inside the draft-decode and draft-extend CUDA graphs.

* the embedding gather and the two eh RMSNorms run as one kernel on the replicated embedding table (needs
  ``SGLANG_ENABLE_EMBED_REPLICATION=1``), bitwise equal to ``fused_eh_norm(F.embedding(...))``;
* ``eh_proj`` runs as a column-sharded GEMV (each rank computes its ``hidden / tp`` output columns) followed by an
  all-gather, instead of the replicated GEMM on every rank; when the target's hops are served (slot ``hops``),
  the gather is the hop family's plain all-reduce of a zero-padded row on the target's mailbox (each column has
  one non-zero contributor; a -0 column comes out +0);
* the draft top-1 runs as a distributed argmax over each rank's logit shard (one cooperative kernel that exchanges a
  (value, index) candidate per row through a symmetric-memory mailbox) instead of the logits all-gather; the stock
  ``draft_topk1_postprocess`` then reads a one-hot logits row, so the token it picks is unchanged.

The kernels, the derived eh_proj shard and the top-1 mailboxes are built with the draft spec. The target's build
registers its hop kernels (with the eh gather's unit and the draft's error-record sites) under the embedding table
the target shares with the draft; the draft looks them up at its first capture and only reads them. Every captured
draft-decode and draft-extend graph runs the draft step, each kernel at the rows it serves (the stock code
elsewhere); eager steps run SGLang's draft forward (the parity switch excepted, ``runtime/parity.py``).
"""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING, Optional

import torch
from sglang.srt.distributed import tensor_model_parallel_all_gather
from sglang.srt.distributed.parallel_state import get_tp_group

try:
    from sglang.kernels.plugin_kernels.nvidia.spec import Step
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.megakernels.spec import Step

from sgl_nvidia_mega.glm5_3 import HIDDEN, draft_attention, runtime
from sgl_nvidia_mega.glm5_3.runtime import parity
from sgl_nvidia_mega.glm5_3.runtime.compile import compile_cached, cu_stream

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/distributed/communication_op.py": ("tensor_model_parallel_all_gather",),
    "srt/distributed/parallel_state.py": (
        "get_tp_group",
        "rank_in_group",
        "world_size",
        "device_group",
    ),
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/runtime.py": (
        "megakernel_agree",
        "megakernel_agree(what)",
    ),
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": (
        "Step",
        "mode",
        "rows",
        "view",
        "widths",
        "draft_sampler",
        "summary",
        "plan",
        "forward",
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
    "srt/layers/layer_boundary/residual/batch.py": (
        "start",
        "complete_output",
        "final_norm",
    ),
    "srt/utils/common.py": ("BumpAllocator",),
    "srt/model_executor/forward_batch_info.py": (
        "ForwardBatch",
        "spec_info",
        "mm_input_embeds",
        "contains_mm_inputs",
        "forward_mode",
    ),
    # the replaced model's modules and its logits processor (and the residual module
    # as the NextN model imports it)
    "srt/models/deepseek_nextn.py": (
        "residual_batch",
        "model",
        "embed_tokens",
        "enorm",
        "hnorm",
        "eh_proj",
        "decoder",
        "shared_head",
        "embed_input_ids",
        "logits_processor",
        "lm_head",
    ),
    "srt/layers/logits_processor.py": (
        "LogitsProcessorOutput",
        "LogitsMetadata",
        "from_forward_batch",
        "extend_return_logprob",
        "is_prefill_only",
        "get_in_autotune_dummy_run",
        "_get_pruned_states",
        "_get_hidden_states_to_store",
        "logit_scale",
        "final_logit_softcapping",
        "do_tensor_parallel_all_gather_dp_attn",
        "do_tensor_parallel_all_gather",
        "use_attn_tp_group",
        "use_tp_lm_head_all_to_all",
        "_compute_lm_head",
        "_logits_gatherer",
        "_copy_logits_to_buffer",
        "vocab_size",
        "next_token_logits",
        "hidden_states",
        "mm_input_embeds",
    ),
    "srt/layers/vocab_parallel_embedding.py": (
        "shard_indices",
        "num_added_elements",
        "org_vocab_start_index",
        "num_org_elements",
        "enable_tp",
    ),
}

# The SGLang function bodies this module's draft step copies, by the digest of their
# code (test/unit/test_contract.py pins them)
SGLANG_MIRRORS = {
    "srt/models/deepseek_nextn.py:DeepseekModelNextN.forward": "d6329c2e1025a6ef",
    # before and after sgl-project/sglang#43449, whose lines run only with
    # --enable-return-aux-hidden-states, which turns speculative decoding off
    "srt/layers/logits_processor.py:LogitsProcessor.forward": (
        "825a8fa668abeec6",
        "813b654dc0599824",
    ),
    "srt/layers/logits_processor.py:LogitsProcessor._get_logits": "0bfb8f2f72aa5755",
}

# draft decode: one row per request (1-4 requests); the draft extend: the covered
# widths, one row per verify token (eh_rows)
DECODE_EH_ROWS = (1, 2, 3, 4)
# the draft-extend graphs select one row per request before the lm_head, so both consumers see B <= 4 rows
TOP1_DECODE_ROWS = 8
TOP1_EXTEND_ROWS = 8
# a peer that never publishes makes the waiting ranks trap after this bound (longer than any first-use
# JIT skew between ranks; far below the collective watchdog), after writing the error record; at most
# SGLang's wait deadline (BuildContext.deadline_s)
TOP1_TIMEOUT_S = 30.0
# call-site ids of the draft's waits in the error record (the target's hops use layer * 4 + kind)
SITE_TOP1_DECODE, SITE_TOP1_EXTEND, SITE_EH_GATHER = 1001, 1002, 1003
SITE_NAMES = {
    SITE_TOP1_DECODE: "draft top-1 (decode)",
    SITE_TOP1_EXTEND: "draft top-1 (extend)",
    SITE_EH_GATHER: "draft eh gather",
}
# rows per hop launch (the target's mailbox instance A); 12 / 16 rows run as two launches
HOP_ROWS = 8
# value table of this engine's eh-norm arithmetic (SGLang's fused_eh_norm kernel)
EH_NORM_OPTIONS = dict(
    sum_rule="lane_fma_cta_tree",
    mean_rule="div_rn",
    rsqrt_ftz=False,
    apply_rule="x_r_w",
)
# draft decode: draft_topk1_postprocess maps NaN to the fp32 constant -1e30 before its argmax;
# draft extend: torch.argmax (NaN is the maximum, the first NaN wins)
NAN_REPLACEMENT_BITS = 0xF149F2CA
ONE_HOT_FILL = -torch.finfo(torch.float32).max


def eh_rows(widths) -> tuple:
    """The eh_proj shard's row counts: the draft decode's (``DECODE_EH_ROWS``) and the
    draft extend's at the covered widths (4 rows a request: 1, 2, 3, 4, 8, 12, 16; 6:
    1, 2, 3, 4, 6, 12)."""
    return tuple(sorted({*DECODE_EH_ROWS, *widths}))


class FusedDraft:
    """The draft spec's built object: kernels, derived weights and top-1 mailboxes of
    the NextN draft layer, and the draft forward of a captured draft step."""

    # the draft offers no speculative entries or graph variants (BuiltMegakernel)
    draft_sampler = None
    draft_loop = None
    draft_extend = None
    begin_draft_round = None
    graph_variants = None

    def __init__(self, nextn, *, widths, deadline_s: float, binding=None) -> None:
        self.armed = False
        self.eh_weight: Optional[torch.Tensor] = None
        self._eh = {}
        self._front = None
        self._top1 = {}
        self.error_record = None
        self._hops = None  # the target's hop kernels (mailboxes, record) when served
        self._hops_checked = False
        self._pad: Optional[torch.Tensor] = None
        self._nextn = nextn
        self._attn_family = None  # the draft layer's attention (draft_attention)
        self._attn_agreed = {}  # (mode, rows) -> every rank serves the kernels there
        self._errors = None
        self.widths = tuple(widths)
        self.eh_rows = eh_rows(self.widths)
        # SGLang's MegakernelBinding for the draft runner (BuildContext.binding)
        self.binding = binding
        # the top-1 wait bound: TOP1_TIMEOUT_S within SGLang's wait deadline
        self.top1_timeout_ns = int(min(TOP1_TIMEOUT_S, deadline_s) * 1e9)
        self._arm(nextn)

    def _arm(self, nextn) -> None:
        """Derive, compile, rendezvous (at build, on every rank)."""
        try:
            from sglang.srt.kernel_plugins.nvidia.binding import megakernel_agree
        except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
            from sglang.srt.megakernels.runtime import megakernel_agree

        from sgl_nvidia_mega.glm5_3.sm103.draft.api import compile_specs
        from sgl_nvidia_mega.glm5_3.sm103.hop.transport import ErrorRecord

        tp = get_tp_group()
        self.rank, self.world = tp.rank_in_group, tp.world_size
        self.n_local = HIDDEN // self.world
        self.col_off = self.rank * self.n_local
        w = nextn.model.eh_proj.weight
        if w.dtype != torch.bfloat16 or tuple(w.shape) != (HIDDEN, 2 * HIDDEN):
            raise ValueError(
                f"GLM-5.3 / GLM-5.2 draft: eh_proj {tuple(w.shape)} {w.dtype} is not "
                "the GLM-5.3 / GLM-5.2 shape"
            )
        self.eh_weight = w[self.col_off : self.col_off + self.n_local].contiguous()
        specs = compile_specs(
            eh_rows=self.eh_rows,
            top1_rows=(TOP1_DECODE_ROWS, TOP1_EXTEND_ROWS),
            eh_norm=EH_NORM_OPTIONS,
        )
        for rows in self.eh_rows:
            spec = specs[f"eh_proj_shard_{rows}"]
            self._eh[rows] = (spec[0], _compile(*spec))
        self._front = _compile(*specs["embed_eh_norm"])
        dev = w.device
        # static: only this rank's columns are ever written, the others stay +0
        self._pad = torch.zeros(
            max(self.eh_rows), HIDDEN, dtype=torch.bfloat16, device=dev
        )
        self._arm_attention(nextn)
        self.error_record = ErrorRecord(dev)
        for name, rows in (("decode", TOP1_DECODE_ROWS), ("extend", TOP1_EXTEND_ROWS)):
            spec = specs[f"distributed_argmax_{rows}"]
            mailbox = _symm_mailbox(spec[0].mailbox_words(self.world), tp, dev)
            ctrl = torch.zeros(spec[0].CTRL_WORDS, dtype=torch.int32, device=dev)
            self._top1[name] = (spec[0], _compile(*spec), mailbox, ctrl)
        torch.cuda.synchronize(dev)
        # every rank has built the same draft kernels (also the barrier before any
        # rank's first use of the mailboxes)
        megakernel_agree(
            (
                self.eh_rows,
                TOP1_DECODE_ROWS,
                TOP1_EXTEND_ROWS,
                self._attn_family is not None,
            ),
            what="GLM-5.3 / GLM-5.2 draft kernels built",
        )
        self.armed = True

    def summary(self) -> str:
        """What was built, for SGLang's startup line."""
        return (
            f"draft: eh_proj shard rows {list(self.eh_rows)}, top-1 rows <= "
            f"{TOP1_DECODE_ROWS} / {TOP1_EXTEND_ROWS}; derived eh_proj shard "
            f"{self.eh_weight.numel() * 2 / 2**20:.1f} MiB; attention kernels: "
            + ("MLA prologue + extend q_a / kv_a GEMM" if self._attn_family else "no")
        )

    def _arm_attention(self, nextn) -> None:
        """Build and compile the draft layer's attention kernels; a reason on any rank
        stops every rank (the draft slot is requested)."""
        family, why = None, None
        try:
            family = draft_attention.DraftAttentionGlue(nextn)
            family.compile(self.eh_rows)
        except ValueError as e:
            family, why = None, str(e)
        whys = runtime.tp_gather([] if why is None else [why])
        if any(whys):
            raise ValueError(
                "NVIDIA megakernel: GLM-5.3 / GLM-5.2 draft attention cannot be built: "
                + runtime.per_rank(whys)
            )
        self._attn_family = family

    def _attach_target_hops(self) -> None:
        """At the first draft capture: the target's hop kernels, registered by the
        target's build under the embedding table it shares with the draft (the eh
        gather's unit compiled and the draft's sites named there; the draft only
        reads them). Every rank must agree; with slot hops requested the target
        serves them, so a miss stops every rank. Without slot hops the draft keeps its
        own error record and SGLang's all-gather."""
        try:
            from sglang.srt.kernel_plugins.nvidia.binding import megakernel_agree
        except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
            from sglang.srt.megakernels.runtime import megakernel_agree

        if self._hops_checked:
            return
        self._hops_checked = True
        hops = runtime.target_hops(self._nextn.model.embed_tokens.weight)
        megakernel_agree(
            hops is not None,
            what="GLM-5.3 / GLM-5.2 draft: target hops found by the draft",
        )
        if hops is None and "hops" in runtime.SLOTS:
            raise ValueError(
                "NVIDIA megakernel: GLM-5.3 / GLM-5.2 draft: slot hops is requested "
                "but the target's hop kernels were not found through the shared "
                "embedding table (the draft and the target do not share it)"
            )
        if hops is not None:
            self._hops = hops
            # the draft's waits write the target's record: one record, one reporter
            self.error_record = hops.err
        else:
            self._errors = runtime.ErrorReporter([self.failure])
        logger.info(
            "NVIDIA megakernel: GLM-5.3 / GLM-5.2 draft: hops on the target's "
            "mailbox: %s (rank %d)",
            "eh gather" if self._hops is not None else "no",
            self.rank,
        )

    def failure(self) -> Optional[dict]:
        """The timeout record of the draft's waits (host memory: readable after the CUDA
        context is lost), or None."""
        rec = self.error_record.read() if self.error_record else None
        if rec is not None and rec["site"] in SITE_NAMES:
            rec = {**rec, "wait": SITE_NAMES[rec["site"]]}
        return rec

    def engaged(self) -> bool:
        return self.armed and torch.cuda.is_current_stream_capturing()

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

    def eh_proj(self, eh_input: torch.Tensor) -> Optional[torch.Tensor]:
        """Column-sharded eh_proj + all-gather, or None (stock path). With the target's
        hops: this rank's columns of the static zero row, then the hop family's plain
        all-reduce, at most 8 rows per launch."""
        from cutlass import Int32, Int64

        rows = eh_input.shape[0]
        if not self.engaged() or rows not in self._eh or not eh_input.is_contiguous():
            return None
        k, fn = self._eh[rows]
        if self._hops is not None:
            part, ld, col = self._pad[:rows], HIDDEN, self.col_off
        else:
            part = torch.empty(
                rows, self.n_local, dtype=eh_input.dtype, device=eh_input.device
            )
            ld, col = part.stride(0), 0
        fn(
            Int64(self.eh_weight.data_ptr()),
            Int64(eh_input.data_ptr()),
            Int64(part.data_ptr()),
            Int32(self.n_local),
            Int32(ld),
            Int32(col),
            Int32(k.grid(self.n_local)),
            cu_stream(),
        )
        if self._hops is None:
            return tensor_model_parallel_all_gather(part, dim=-1)
        out = torch.empty_like(part)
        for lo in range(0, rows, HOP_ROWS):
            hi = min(rows, lo + HOP_ROWS)
            self._hops.allreduce(part[lo:hi], out[lo:hi], site=SITE_EH_GATHER)
        return out

    def top1_one_hot(
        self, logits_shard: torch.Tensor, n_valid: int, vocab: int, extend: bool
    ) -> Optional[torch.Tensor]:
        """A [rows, vocab] fp32 one-hot of the global top-1 (stock NaN rule), or None."""
        from cutlass import Int32, Int64, Uint32, Uint64

        rows, vloc = logits_shard.shape
        k, fn, mailbox, ctrl = self._top1["extend" if extend else "decode"]
        if (
            not self.engaged()
            or rows > k.max_rows
            or logits_shard.dtype != torch.bfloat16
        ):
            return None
        k.check_shapes(rows, vloc, n_valid, self.world, self.rank)
        ids = torch.empty(rows, dtype=torch.int64, device=logits_shard.device)
        vals = torch.empty(rows, dtype=torch.float32, device=logits_shard.device)
        fn(
            Int64(logits_shard.data_ptr()),
            Int64(mailbox.data_ptr()),
            Int64(mailbox.peers.data_ptr()),
            Int64(ctrl.data_ptr()),
            Int64(ids.data_ptr()),
            Int64(vals.data_ptr()),
            Int32(rows),
            Int32(vloc),
            Int32(n_valid),
            Int32(self.world),
            Int32(self.rank),
            Uint64(self.top1_timeout_ns),
            Int32(0 if extend else 1),
            Uint32(NAN_REPLACEMENT_BITS),
            Int64(self.error_record.device_ptr if self.error_record else 0),
            Int32(SITE_TOP1_EXTEND if extend else SITE_TOP1_DECODE),
            cu_stream(),
        )
        one_hot = torch.full(
            (rows, vocab), ONE_HOT_FILL, dtype=torch.float32, device=logits_shard.device
        )
        one_hot.scatter_(1, ids[:, None], 0.0)
        return one_hot

    def attention_for(self, rows: int, forward_batch: ForwardBatch, view):
        """The draft layer's attention step group served at ``rows`` in this forward mode (a
        ``draft_attention.DraftAttention``), or None: the stock decoder layer runs. At the first
        capture of each (mode, rows) with a kernel form, every rank reports the host-only facts the
        kernels need (a host-only gather on the
        CPU group, legal inside the capture) and stops every rank on any reason; ``view`` is the
        running forward's ForwardView."""
        family = self._attn_family
        group = None if family is None else family.for_rows(rows)
        if group is None:
            return None
        mode = "decode" if forward_batch.forward_mode.is_decode() else "extend"
        if (mode, rows) not in self._attn_agreed:
            gathered = runtime.tp_gather(
                list(family.step_reasons(self._nextn, forward_batch, view))
            )
            if any(gathered):
                raise ValueError(
                    f"NVIDIA megakernel: GLM-5.3 / GLM-5.2 draft attention cannot run "
                    f"the draft {mode} graph of {rows} rows: "
                    f"{runtime.per_rank(gathered)}"
                )
            self._attn_agreed[(mode, rows)] = True
            logger.info(
                "NVIDIA megakernel: GLM-5.3 / GLM-5.2 draft attention on: draft %s "
                "graph, %d rows (rank %d)",
                mode,
                rows,
                self.rank,
            )
        return group

    def plan(self, input_ids, forward_batch, pp_proxy_tensors=None) -> Optional[Step]:
        """A draft step for every captured draft-extend and draft-decode graph (each
        kernel then serves the rows it is built for, the stock code the others; draft
        decode has 1 row a request, off the width grid: SGLang's dispatch does not
        check widths); eager steps only under the test-only parity switch; else
        None."""
        if pp_proxy_tensors is not None or not self.armed:
            return None
        mode = forward_batch.forward_mode
        rows = int(input_ids.shape[0])
        if rows < 1:
            return None
        if mode.is_draft_extend_v2():
            step = Step(mode="draft_extend", rows=rows)
        elif mode.is_decode():
            step = Step(mode="draft_decode", rows=rows)
        else:
            return None
        if torch.cuda.is_current_stream_capturing():
            self._attach_target_hops()
            return step
        return step if parity.eager_steps_allowed() else None

    def forward(
        self,
        step: Step,
        original,
        input_ids,
        positions,
        forward_batch,
        pp_proxy_tensors=None,
    ):
        """The draft forward of a captured draft step (``original``: the model's own
        forward, for the parity check of an eager step)."""
        nextn = self._nextn
        if not torch.cuda.is_current_stream_capturing():
            # plan() covers eager steps only under the test-only parity switch
            return parity.run_draft(
                self, step, original, input_ids, positions, forward_batch
            )
        hidden_states = self.run(
            nextn.model, input_ids, positions, forward_batch, step.view
        )
        return draft_logits(self, nextn, input_ids, hidden_states, forward_batch)

    def run_stock(self, step: Step, input_ids, positions, forward_batch):
        """The draft step on its stock bodies (no kernel; the parity check)."""
        nextn = self._nextn
        hidden_states = self.forward_steps(
            nextn.model,
            input_ids,
            positions,
            forward_batch,
            draft_attention.DraftAttention(),
            step.view,
        )
        return draft_logits(self, nextn, input_ids, hidden_states, forward_batch)

    def run(self, model, input_ids, positions, forward_batch: ForwardBatch, view=None):
        """The served draft forward of an engaged step."""
        attention = self.attention_for(int(positions.shape[0]), forward_batch, view)
        return self.forward_steps(
            model, input_ids, positions, forward_batch, attention, view
        )

    def forward_steps(
        self, model, input_ids, positions, forward_batch, attention=None, view=None
    ):
        """``DeepseekModelNextN.forward`` (CUDA path) with the fused front and the sharded eh_proj (each
        stock unless engaged), and the decoder layer's attention through ``attention`` (a step group; None:
        the stock decoder layer) on the running forward's ``view``."""
        from sglang.kernels.ops.layernorm.fused_eh_norm import fused_eh_norm
        from sglang.srt.eplb.expert_distribution import (
            get_global_expert_distribution_recorder,
        )
        from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
        from sglang.srt.utils import BumpAllocator

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
                eh_input = self.front(
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
            hidden_states = self.eh_proj(eh_input)
            if hidden_states is None:
                hidden_states = model.eh_proj(eh_input)
        else:
            hidden_states = model.embed_input_ids(input_ids, forward_batch)

        residual_batch.start(forward_batch)
        index_topk_share = IndexTopKShareState.from_mtp_carry(forward_batch)
        with get_global_expert_distribution_recorder().disable_this_region():
            decoder = model.decoder
            if attention is not None:
                decoder = partial(
                    draft_attention.decoder_layer_forward,
                    model.decoder,
                    attention,
                    view=view,
                )
            hidden_states, topk_indices = decoder(
                positions,
                hidden_states,
                forward_batch,
                zero_allocator,
                prev_topk_indices=index_topk_share.topk_indices,
            )
        hidden_states = residual_batch.complete_output(hidden_states, forward_batch)
        if not forward_batch.forward_mode.is_idle():
            hidden_states = residual_batch.final_norm(
                hidden_states, forward_batch, model.shared_head.norm
            )
        index_topk_share.update(topk_indices)
        index_topk_share.publish()
        return hidden_states


def draft_logits(fd: FusedDraft, nextn, input_ids, hidden_states, forward_batch):
    """``nextn``'s logits on the draft step: SGLang's logits processor, unchanged,
    except in an engaged step that takes its plain TP logits path, where the
    LogitsProcessorOutput is built here as ``LogitsProcessor.forward`` builds it for a
    decode / draft-extend batch, with the draft top-1 served by the distributed argmax
    (a one-hot logits row that SGLang's ``draft_topk1_postprocess`` reads unchanged)."""
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

    lp = nextn.logits_processor
    meta = _plain_tp_logits(fd, lp, forward_batch)
    if meta is None:
        return lp(input_ids, hidden_states, nextn.lm_head, forward_batch)
    # LogitsProcessor.forward (main af856717) for a decode / draft-extend-v2 batch
    # without logprobs: the pruned rows (the draft extend's selected rows), the
    # hidden states it stores, then _get_logits' TP path on the pruned rows
    pruned, pruned_before_norm, aux_pruned, sample_indices, _, _ = (
        lp._get_pruned_states(hidden_states, None, None, meta)
    )
    hidden_states_to_store = lp._get_hidden_states_to_store(
        hidden_states,
        None,
        None,
        pruned,
        pruned_before_norm,
        aux_pruned,
        sample_indices,
        meta,
    )
    lm_head = nextn.lm_head
    shard = lp._compute_lm_head(pruned, lm_head, None)
    si = lm_head.shard_indices
    one_hot = None
    if (
        si.num_added_elements == 0
        and si.org_vocab_start_index == fd.rank * shard.shape[1]
    ):
        one_hot = fd.top1_one_hot(
            shard,
            si.num_org_elements,
            lp.vocab_size,
            extend=meta.forward_mode.is_draft_extend_v2(),
        )
    # same ranks, same decision: everyone takes the stock gather
    if one_hot is None:
        one_hot = lp._logits_gatherer(shard)
    logits = lp._copy_logits_to_buffer(one_hot, meta, use_buffer=True)
    if sample_indices is not None:
        logits = logits[sample_indices]
    return LogitsProcessorOutput(
        next_token_logits=logits,
        hidden_states=hidden_states_to_store,
        mm_input_embeds=meta.mm_input_embeds,
    )


def _plain_tp_logits(fd: FusedDraft, lp, forward_batch):
    """The batch's LogitsMetadata when an engaged draft step's logits take the plain
    TP path that ``draft_logits`` builds (decode or draft extend v2, no logprobs or
    scoring, no DP attention or attention-TP lm_head, no lm_head all-to-all, logit
    scale or softcap, no autotune pass), else None."""
    from sglang.srt.layers.logits_processor import (
        LogitsMetadata,
        get_in_autotune_dummy_run,
    )

    mode = forward_batch.forward_mode
    if (
        not fd.engaged()
        or not (mode.is_decode() or mode.is_draft_extend_v2())
        or lp.logit_scale is not None
        or lp.final_logit_softcapping
        or lp.do_tensor_parallel_all_gather_dp_attn
        or not lp.do_tensor_parallel_all_gather
        or lp.use_attn_tp_group
        or lp.use_tp_lm_head_all_to_all
        or get_in_autotune_dummy_run()
    ):
        return None
    meta = LogitsMetadata.from_forward_batch(forward_batch)
    if (
        not (meta.forward_mode.is_decode() or meta.forward_mode.is_draft_extend_v2())
        or meta.extend_return_logprob
        or meta.is_prefill_only
    ):
        return None
    return meta


def _compile(kernel, constexprs, args):
    """``cute.compile(kernel, *args)`` through the plugin's compile cache."""
    import cutlass.cute as cute

    return compile_cached(kernel, constexprs, lambda: cute.compile(kernel, *args))


def _symm_mailbox(words: int, tp, device) -> torch.Tensor:
    import torch.distributed._symmetric_memory as symm_mem

    t = symm_mem.empty((words,), dtype=torch.int64, device=device)
    t.fill_(-1)
    handle = symm_mem.rendezvous(t, tp.device_group.group_name)
    t.peers = torch.tensor(
        [int(p) for p in handle.buffer_ptrs], dtype=torch.int64, device=device
    )
    t.handle = handle
    return t
