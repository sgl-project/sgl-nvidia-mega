# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Portions Copyright 2023-2024 SGLang Team (Apache-2.0)
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/deepseek_nextn.py (DeepseekModelNextN.forward),
# python/sglang/srt/layers/logits_processor.py (LogitsProcessor.forward and _get_logits: the decode / draft-extend
# path and its TP logits gather).
"""The GLM-5.3 NextN draft step (slot ``draft``; SM103 / SM107, TP8, MTP topk 1): CuTe DSL
kernels inside the draft-decode and draft-extend CUDA graphs.

* the embedding gather and the two eh RMSNorms run as one kernel on the replicated embedding table (needs
  ``SGLANG_ENABLE_EMBED_REPLICATION=1``), bitwise equal to ``fused_eh_norm(F.embedding(...))``;
* ``eh_proj`` runs as a column-sharded GEMV (each rank computes its ``hidden / tp`` output columns) followed by an
  all-gather, instead of the replicated GEMM on every rank; when the target's hops are served (slot ``hops``),
  the gather is the hop family's plain all-reduce of a zero-padded row on the target's mailbox (each column has
  one non-zero contributor; a -0 column comes out +0);
* the draft top-1 runs as a distributed argmax over each rank's logit shard (one cooperative kernel that exchanges a
  (value, index) candidate per row through a symmetric-memory mailbox) instead of the logits all-gather; the stock
  ``draft_topk1_postprocess`` then reads a one-hot logits row, so the token it picks is unchanged;
* ``SGLANG_NVIDIA_MEGAKERNEL_GLM_DRAFT_FRONT=rowpar`` (default off): with the target's hops and a draft attention step
  group, ``eh_proj`` runs row-parallel (each rank multiplies its 1536-column K window of the front's output with the
  same columns of the weight, cuBLAS) and the hop family's residual-free all-reduce + RMSNorm sums the partials and
  applies the draft layer's input norm in one launch (``draft_attention.PreparedInput``: the attention prepare is
  skipped). Each partial rounds to BF16 before the sum and the norm is the hop's, so only the draft's proposals can
  change (acceptance); served at 2-8 rows (1-row and wider steps keep the column shard).
* slot ``draft_rowselect``: in the draft extend the layer keeps only the ``select_index`` rows (one per request:
  the rows its LM head reads) after the post-attention norm, so the MLP, its all-reduce and the final norm run on
  one row per request (the attention still writes every row's KV and indexer key).
* SGLang's draft top-1 sink (``SGLANG_SPEC_DRAFT_TOPK1_SINK``, a ``DraftTopk1Sink`` in the forward flags of a
  draft-decode step): the distributed argmax also writes the step's ``topk_index`` / ``topk_p``, its draft-token
  column and ``positions += 1`` (what SGLang's partials finalizer writes), so SGLang's draft loop does not launch
  the finalizer (and, with ``SGLANG_SPEC_DRAFT_INPUT_NO_COPY``, reads the step inputs in place). Same token, same
  writes: bitwise.

The kernels, the derived eh_proj shard and the top-1 mailboxes are built with the draft spec. The target's build
registers its hop kernels (with the eh gather's unit and the draft's error-record sites) under the embedding table
the target shares with the draft; the draft looks them up at its first capture and only reads them. Every captured
draft-decode and draft-extend graph runs the draft step, each kernel at the rows it serves (the stock code
elsewhere); eager steps run SGLang's draft forward (the parity switch excepted, ``runtime/parity.py``).
"""

from __future__ import annotations

import logging
import os
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
        # DRAFT_FRONT_ENV: the summed eh_proj is the stream's written residual
        "set_written",
    ),
    # SGLANG_SPEC_DRAFT_TOPK1_SINK: the draft-decode step's sink in the forward flags
    "srt/runtime_context.py": ("get_forward", "\"draft_topk1_sink\""),
    "kernels/ops/speculative/topk1.py": (
        "DraftTopk1Sink",
        "topk_p",
        "topk_index",
        "positions",
        "draft_tokens",
        "column",
        "filled",
    ),
    # DRAFT_FRONT_ENV: the draft layer's input norm, applied in the hop
    "srt/layers/layernorm.py": ("weight", "variance_epsilon"),
    "srt/model_executor/forward_batch_info.py": (
        "ForwardBatch",
        "spec_info",
        "mm_input_embeds",
        "contains_mm_inputs",
        "forward_mode",
        # slot draft_rowselect: the draft extend's gate
        "CaptureHiddenMode",
        "return_logprob",
        "capture_hidden_mode",
        "num_token_non_padded",
        "input_ids",
    ),
    "srt/model_executor/forward_mode.py|srt/model_executor/forward_batch_info.py": (
        "FULL",
    ),
    "srt/speculative/eagle_info.py|srt/speculative/*.py": ("select_index",),
    "srt/models/deepseek_v2.py": ("DeepseekV2MoE", "_enable_a2a_moe", "input_layernorm"),
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
        "draft_extend_select_index",
        "do_tensor_parallel_all_gather_dp_attn",
        "do_tensor_parallel_all_gather",
        "use_attn_tp_group",
        "use_tp_lm_head_all_to_all",
        "_compute_lm_head",
        "_logits_gatherer",
        "_copy_logits_to_buffer",
        "vocab_size",
        "next_token_logits",
        # SGLANG_SPEC_DRAFT_ARGMAX_EXCHANGE: the top-1 partials output
        "_draft_argmax_requested",
        "next_token_argmax_partials",
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
    "srt/models/deepseek_nextn.py:DeepseekModelNextN.forward": (
        "d6329c2e1025a6ef",
        # sgl-project/sglang#46784: the eh projection moved into
        # _project_with_target_hidden (same code)
        "c779de58258cf814",
    ),
    # before and after sgl-project/sglang#43449, whose lines run only with
    # --enable-return-aux-hidden-states, which turns speculative decoding off
    "srt/layers/logits_processor.py:LogitsProcessor.forward": (
        "825a8fa668abeec6",
        "813b654dc0599824",
        # sgl-project/sglang#47200: vocab-sharded target-verify logits (a target verify only; the draft
        # steps take the gather as before)
        "48c24aac22ca5303",
        # sgl-project/sglang#44349 and #51899: the beam-search trie head and input
        # candidate selectors (neither serves a draft step)
        "f1604e7bcf4cd8fa",
    ),
    "srt/layers/logits_processor.py:LogitsProcessor._get_logits": (
        "0bfb8f2f72aa5755",
        # sgl-project/sglang#48414: a deferred target-verify gather (a target verify
        # only)
        "835923a6dc3f1a84",
    ),
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
# DRAFT_FRONT_ENV: the row-parallel eh_proj sum + input norm (ll.draft's TP4 sum is 1004)
SITE_EH_NORM = 1005
SITE_NAMES = {
    SITE_TOP1_DECODE: "draft top-1 (decode)",
    SITE_TOP1_EXTEND: "draft top-1 (extend)",
    SITE_EH_GATHER: "draft eh gather",
    SITE_EH_NORM: "draft eh_proj sum + input norm",
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
# lanes per candidate of SGLang's draft top-1 partials (kernels/ops/speculative/topk1.py:
# DRAFT_TOPK1_PARTIAL_LANES)
PARTIAL_LANES = 4
# the draft front (off unless "rowpar"): eh_proj row-parallel + the hop's all-reduce with the input norm
DRAFT_FRONT_ENV = "SGLANG_NVIDIA_MEGAKERNEL_GLM_DRAFT_FRONT"
# the TP8 K window of a rank (2H / 8 input columns: ranks 0-3 the embedding half, 4-7 the hidden half)
ROWPAR_WORLD = 8
ROWPAR_WINDOW = 2 * HIDDEN // ROWPAR_WORLD
# rows from which the row-parallel front serves: at 1 row it measured no faster than the column shard
# (T90a frontbench, GB300 TP8: +0.4 us at 1 row, -2.5 / -4.8 / -7.5 us at 2 / 4 / 8 rows)
ROWPAR_MIN_ROWS = 2


def draft_front_mode() -> Optional[str]:
    """``DRAFT_FRONT_ENV``: "rowpar", or None (the column shard and its gather)."""
    mode = os.environ.get(DRAFT_FRONT_ENV, "").strip().lower()
    if mode in ("", "0", "off"):
        return None
    if mode != "rowpar":
        raise ValueError(
            f"NVIDIA megakernel: GLM-5.3 / GLM-5.2 draft: {DRAFT_FRONT_ENV}={mode!r} "
            "(expected rowpar or unset)"
        )
    return mode


def _topk1_sink():
    """SGLang's draft top-1 sink of the running draft-decode step (``draft_topk1_sink``
    in the forward flags, SGLANG_SPEC_DRAFT_TOPK1_SINK), or None (also on an SGLang
    without the flag)."""
    try:
        from sglang.srt.runtime_context import get_forward
    except ImportError:
        return None
    try:
        return getattr(get_forward(), "draft_topk1_sink", None)
    except Exception:
        return None


def sink_reason(sink, rows: int) -> str:
    """Why the distributed argmax cannot write ``sink`` (a ``DraftTopk1Sink``) at
    ``rows`` rows, or "" (host-only facts, fixed per captured shape)."""
    ti, tp, pos, tok = sink.topk_index, sink.topk_p, sink.positions, sink.draft_tokens
    if ti.dtype != torch.int64 or tuple(ti.shape) != (rows, 1) or not ti.is_contiguous():
        return f"topk_index {tuple(ti.shape)} {ti.dtype}"
    if tp.dtype != torch.float32 or tuple(tp.shape) != (rows, 1) or not tp.is_contiguous():
        return f"topk_p {tuple(tp.shape)} {tp.dtype}"
    if pos.dtype != torch.int64 or tuple(pos.shape) != (rows,) or not pos.is_contiguous():
        return f"positions {tuple(pos.shape)} {pos.dtype}"
    if (
        tok.dtype != torch.int64
        or tok.dim() != 2
        or tok.shape[0] != rows
        or tok.stride(1) != 1
        or not 0 <= int(sink.column) < tok.shape[1]
    ):
        return f"draft tokens {tuple(tok.shape)} {tok.dtype} column {sink.column}"
    return ""


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
    speculative_runtime_hooks = None

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
        # slot draft_rowselect, decided at build (static, rank-uniform)
        self.row_select = "draft_rowselect" in runtime.SLOTS and row_select_supported(
            nextn.model.decoder.mlp
        )
        # (mode, rows) of the draft graphs whose parts rank 0 has logged
        self._reported = set()
        # SGLang's MegakernelBinding for the draft runner (BuildContext.binding)
        self.binding = binding
        # the top-1 wait bound: TOP1_TIMEOUT_S within SGLang's wait deadline
        self.top1_timeout_ns = int(min(TOP1_TIMEOUT_S, deadline_s) * 1e9)
        # DRAFT_FRONT_ENV: this rank's K window of eh_proj, the input norm, and whether the
        # target's hops serve the sum (decided at the first capture, on every rank alike)
        self.front_mode = draft_front_mode()
        self._rp_w: Optional[torch.Tensor] = None
        self._rp_win = 0
        self._rp_gamma: Optional[torch.Tensor] = None
        self._rp_eps = 0.0
        self._rp_ready = False
        self._rp_why = "off"
        # the draft top-1 sink: logged once per rank at its first fill
        self._sink_logged = False
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
        if self.front_mode:
            self._arm_rowpar(nextn, w)
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
        built = (
            self.eh_rows,
            TOP1_DECODE_ROWS,
            TOP1_EXTEND_ROWS,
            self._attn_family is not None,
        )
        if self.front_mode:
            built += (self._rp_w is not None,)
        megakernel_agree(built, what="GLM-5.3 / GLM-5.2 draft kernels built")
        self.armed = True

    def summary(self) -> str:
        """What was built, for SGLang's startup line."""
        return (
            f"draft: eh_proj shard rows {list(self.eh_rows)}, top-1 rows <= "
            f"{TOP1_DECODE_ROWS} / {TOP1_EXTEND_ROWS}; derived eh_proj shard "
            f"{self.eh_weight.numel() * 2 / 2**20:.1f} MiB; attention kernels: "
            + (
                "MLA prologue + extend q_a / kv_a GEMM"
                + (
                    " + sparse MLA + value up-projection"
                    if draft_attention.draft_sparse_mla_on()
                    else ""
                )
                if self._attn_family
                else "no"
            )
            + "; draft-extend MLP on the select_index rows: "
            + ("yes" if self.row_select else "no")
        )

    def _arm_rowpar(self, nextn, w: torch.Tensor) -> None:
        """``DRAFT_FRONT_ENV``: this rank's K window of eh_proj ([H, 2H / 8], 18 MiB) and
        the draft layer's input norm; the hop unit is the target's (``_attach_target_hops``)."""
        norm = nextn.model.decoder.input_layernorm
        gamma = norm.weight
        if self.world != ROWPAR_WORLD:
            self._rp_why = f"TP{self.world} (served at TP{ROWPAR_WORLD})"
        elif gamma.dtype != torch.bfloat16 or tuple(gamma.shape) != (HIDDEN,):
            self._rp_why = f"input norm weight {tuple(gamma.shape)} {gamma.dtype}"
        else:
            self._rp_win = self.rank * ROWPAR_WINDOW
            self._rp_w = w[:, self._rp_win : self._rp_win + ROWPAR_WINDOW].contiguous()
            self._rp_gamma, self._rp_eps = gamma, float(norm.variance_epsilon)
            self._rp_why = "waiting for the target's hops"

    def eh_prepared(self, eh_input: torch.Tensor):
        """``DRAFT_FRONT_ENV``: eh_proj summed through the draft layer's input norm in
        the hop (``draft_attention.PreparedInput``), or None:
        * at ``ROWPAR_MIN_ROWS``-8 rows: row-parallel matmul + ``allreduce_rmsnorm``;
        * at 1 row (when ``1 in self._eh``): column-sharded ``EhProjShard(1)`` into the
          zero-padded buffer + ``allreduce_rmsnorm`` (eliminates the standalone draft
          ``input_layernorm`` launch on every draft-decode step)."""
        rows = eh_input.shape[0]
        if (
            not self._rp_ready
            or not self.engaged()
            or not 1 <= rows <= HOP_ROWS
            or not eh_input.is_contiguous()
        ):
            return None
        if rows >= ROWPAR_MIN_ROWS:
            part = torch.matmul(
                eh_input[:, self._rp_win : self._rp_win + ROWPAR_WINDOW], self._rp_w.t()
            )
        elif rows in self._eh:
            from cutlass import Int32, Int64

            k, fn = self._eh[rows]
            part = self._pad[:rows]
            fn(
                Int64(self.eh_weight.data_ptr()),
                Int64(eh_input.data_ptr()),
                Int64(part.data_ptr()),
                Int32(self.n_local),
                Int32(HIDDEN),
                Int32(self.col_off),
                Int32(k.grid(self.n_local)),
                cu_stream(),
            )
        else:
            return None
        out_res = torch.empty_like(part)
        out_norm = torch.empty_like(part)
        self._hops.allreduce_rmsnorm(
            part, self._rp_gamma, out_res, out_norm, site=SITE_EH_NORM
        )
        return draft_attention.PreparedInput(out_norm, out_res)
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
        if self.front_mode:
            self._attach_rowpar(hops, megakernel_agree)
        logger.info(
            "NVIDIA megakernel: GLM-5.3 / GLM-5.2 draft: hops on the target's "
            "mailbox: %s (rank %d)",
            "eh gather" if self._hops is not None else "no",
            self.rank,
        )

    def _attach_rowpar(self, hops, megakernel_agree) -> None:
        """``DRAFT_FRONT_ENV``: serve the row-parallel front when this rank has its window,
        the target's hops carry the all-reduce + RMSNorm unit and their eps is the input
        norm's; every rank must agree."""
        why = self._rp_why if self._rp_w is None else ""
        if not why and hops is None:
            why = "the target's hops are not served"
        elif not why and "allreduce_rmsnorm" not in hops.compiled:
            why = "the hops have no all-reduce + RMSNorm unit"
        elif not why and float(hops.eps) != self._rp_eps:
            why = f"the hops' eps {hops.eps} is not the input norm's {self._rp_eps}"
        megakernel_agree(
            why == "", what="GLM-5.3 / GLM-5.2 draft: row-parallel eh_proj served"
        )
        self._rp_ready = why == ""
        self._rp_why = why or "served"
        if self.rank == 0:
            logger.info(
                "NVIDIA megakernel: GLM-5.3 / GLM-5.2 draft: %s=rowpar: %s",
                DRAFT_FRONT_ENV,
                "eh_proj row-parallel (K window %d) + the hops' all-reduce + input RMSNorm at %d-%d rows"
                % (ROWPAR_WINDOW, ROWPAR_MIN_ROWS, HOP_ROWS)
                if self._rp_ready
                else f"declined ({why}); the column shard and its gather serve",
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

    def _top1_ids(
        self,
        logits_shard: torch.Tensor,
        n_valid: int,
        extend: bool,
        partials: Optional[torch.Tensor] = None,
        sink=None,
    ) -> Optional[torch.Tensor]:
        """The global top-1 ids ``[rows]`` int64 (stock NaN rule) by the distributed
        argmax, also written as SGLang's top-1 partials into ``partials`` (``[rows, 4]``
        fp32) when given, and with ``sink`` (a checked ``DraftTopk1Sink``) into its
        ``topk_index`` / ``topk_p`` / draft-token column with ``positions += 1``; or
        None (not engaged / not served)."""
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
        if sink is not None:
            ids = sink.topk_index.view(rows)
        else:
            ids = torch.empty(rows, dtype=torch.int64, device=logits_shard.device)
        vals = torch.empty(rows, dtype=torch.float32, device=logits_shard.device)
        fn(
            Int64(logits_shard.data_ptr()),
            Int64(mailbox.data_ptr()),
            Int64(mailbox.peers.data_ptr()),
            Int64(ctrl.data_ptr()),
            Int64(ids.data_ptr()),
            Int64(vals.data_ptr()),
            Int64(0 if partials is None else partials.data_ptr()),
            Int64(0 if sink is None else sink.topk_p.data_ptr()),
            Int64(0 if sink is None else sink.draft_tokens.data_ptr()),
            Int32(1 if sink is None else sink.draft_tokens.stride(0)),
            Int32(0 if sink is None else int(sink.column)),
            Int64(0 if sink is None else sink.positions.data_ptr()),
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
        return ids

    def top1_one_hot(
        self, logits_shard: torch.Tensor, n_valid: int, vocab: int, extend: bool
    ) -> Optional[torch.Tensor]:
        """A [rows, vocab] fp32 one-hot of the global top-1 (stock NaN rule), or None."""
        ids = self._top1_ids(logits_shard, n_valid, extend)
        if ids is None:
            return None
        one_hot = torch.full(
            (ids.shape[0], vocab),
            ONE_HOT_FILL,
            dtype=torch.float32,
            device=logits_shard.device,
        )
        one_hot.scatter_(1, ids[:, None], 0.0)
        return one_hot

    def top1_partials(
        self, logits_shard: torch.Tensor, n_valid: int, extend: bool
    ) -> Optional[torch.Tensor]:
        """The global top-1 (stock NaN rule) as SGLang's draft top-1 partials, one
        candidate per row (``[rows, 4]`` fp32, written by the distributed argmax
        itself), or None. SGLang's partials finalizers then take the token without a
        logits row: no one-hot fill and scatter, no logits-buffer copy, no argmax."""
        rows = logits_shard.shape[0]
        partials = torch.empty(
            (rows, PARTIAL_LANES), dtype=torch.float32, device=logits_shard.device
        )
        if self._top1_ids(logits_shard, n_valid, extend, partials) is None:
            return None
        return partials

    def top1_sink(self, logits_shard: torch.Tensor, n_valid: int, sink) -> bool:
        """SGLang's draft top-1 sink (draft decode): the distributed argmax writes its
        outputs and ``sink.filled`` is set; False (nothing launched) when the sink's
        layout does not fit (the caller then takes the partials)."""
        rows = logits_shard.shape[0]
        why = sink_reason(sink, rows)
        if why:
            if not self._sink_logged:
                self._sink_logged = True
                logger.warning(
                    "NVIDIA megakernel: GLM-5.3 / GLM-5.2 draft: the draft top-1 sink "
                    "is not written by the distributed argmax (%s); the partials "
                    "serve (rank %d)",
                    why,
                    self.rank,
                )
            return False
        if self._top1_ids(logits_shard, n_valid, False, None, sink) is None:
            return False
        sink.filled = True
        if not self._sink_logged:
            self._sink_logged = True
            logger.info(
                "NVIDIA megakernel: GLM-5.3 / GLM-5.2 draft: the distributed argmax "
                "writes the draft-decode top-1 sink (topk_index, topk_p, draft-token "
                "column, positions) at %d rows (rank %d)",
                rows,
                self.rank,
            )
        return True

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
        hidden_states, metadata = self.run(
            nextn.model, input_ids, positions, forward_batch, step.view
        )
        return draft_logits(
            self, nextn, input_ids, hidden_states, forward_batch, metadata
        )

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

    def extend_rows(self, forward_batch: ForwardBatch) -> Optional[torch.Tensor]:
        """Slot draft_rowselect: ``select_index_rows`` when the slot is on and the
        NextN MLP supports it, else None."""
        return select_index_rows(forward_batch) if self.row_select else None

    def run(self, model, input_ids, positions, forward_batch: ForwardBatch, view=None):
        """The served draft forward of an engaged step: ``(hidden_states, metadata)``
        for the logits (``draft_logits``). With the row select the hidden states hold
        only the ``select_index`` rows, so ``metadata`` is the batch's LogitsMetadata
        without the row selection (the logits processor must not select twice);
        otherwise it is None (the forward batch's own)."""
        rows = int(positions.shape[0])
        attention = self.attention_for(rows, forward_batch, view)
        ffn_rows = self.extend_rows(forward_batch)
        hidden_states = self.forward_steps(
            model, input_ids, positions, forward_batch, attention, view, ffn_rows
        )
        self._report(forward_batch, rows, attention is not None, ffn_rows)
        if ffn_rows is None:
            return hidden_states, None
        from sglang.srt.layers.logits_processor import LogitsMetadata

        metadata = LogitsMetadata.from_forward_batch(forward_batch)
        metadata.draft_extend_select_index = None
        return hidden_states, metadata

    def _report(self, forward_batch, rows: int, fused_attention: bool, ffn_rows):
        """Rank 0, once per captured draft graph (mode, rows): its attention and the
        rows its MLP runs on (the engagement line of slot draft_rowselect)."""
        if self.rank != 0 or not torch.cuda.is_current_stream_capturing():
            return
        mode = "decode" if forward_batch.forward_mode.is_decode() else "extend"
        if (mode, rows) in self._reported:
            return
        self._reported.add((mode, rows))
        mlp = (
            f"MLP on all {rows} rows"
            if ffn_rows is None
            else f"MLP on the {int(ffn_rows.shape[0])} select_index rows"
        )
        logger.info(
            "NVIDIA megakernel: GLM-5.3 / GLM-5.2 draft: draft %s graph, %d rows: "
            "%s attention, %s",
            mode,
            rows,
            "fused" if fused_attention else "stock",
            mlp,
        )

    def forward_steps(
        self,
        model,
        input_ids,
        positions,
        forward_batch,
        attention=None,
        view=None,
        ffn_rows=None,
    ):
        """``DeepseekModelNextN.forward`` (CUDA path) with the fused front and the sharded eh_proj (each
        stock unless engaged), and the decoder layer's attention through ``attention`` (a step group; None:
        the stock decoder layer) on the running forward's ``view``. With ``ffn_rows`` (slot
        draft_rowselect) the MLP and the final norm run on those rows only."""
        from sglang.kernels.ops.layernorm.fused_eh_norm import fused_eh_norm
        from sglang.srt.eplb.expert_distribution import (
            get_global_expert_distribution_recorder,
        )
        from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch

        # no eager zero fill (runtime.LazyZeroAllocator): the front kernel's ids / rows are read after its
        # grid-dependency wait, and the launch before it (the graph's input copies or the previous step's
        # _small_copy_kernel) is a plain launch, so the wait orders it after every earlier grid
        zero_allocator = runtime.LazyZeroAllocator(
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
            # DRAFT_FRONT_ENV: only for a step group, which takes the normed input
            hidden_states = (
                self.eh_prepared(eh_input) if attention is not None else None
            )
            if hidden_states is None:
                hidden_states = self.eh_proj(eh_input)
            if hidden_states is None:
                hidden_states = model.eh_proj(eh_input)
        else:
            hidden_states = model.embed_input_ids(input_ids, forward_batch)

        prepared = isinstance(hidden_states, draft_attention.PreparedInput)
        if prepared:
            # the hop applied the draft layer's input norm: the attention prepare is skipped
            residual_batch.set_written(hidden_states.residual, forward_batch)
            hidden_states = hidden_states.norm
        else:
            residual_batch.start(forward_batch)
        index_topk_share = IndexTopKShareState.from_mtp_carry(forward_batch)
        with get_global_expert_distribution_recorder().disable_this_region():
            decoder = model.decoder
            if attention is not None or ffn_rows is not None:
                decoder = partial(
                    draft_attention.decoder_layer_forward,
                    model.decoder,
                    attention,
                    view=view,
                    ffn_rows=ffn_rows,
                    **({"prepared": True} if prepared else {}),
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


def draft_logits(
    fd: FusedDraft, nextn, input_ids, hidden_states, forward_batch, metadata=None
):
    """``nextn``'s logits on the draft step: SGLang's logits processor, unchanged,
    except in an engaged step that takes its plain TP logits path, where the
    LogitsProcessorOutput is built here as ``LogitsProcessor.forward`` builds it for a
    decode / draft-extend batch, with the draft top-1 served by the distributed argmax
    (a one-hot logits row that SGLang's ``draft_topk1_postprocess`` reads unchanged).
    ``metadata``: the LogitsMetadata to use instead of the forward batch's (the row
    select's, without ``draft_extend_select_index``)."""
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

    lp = nextn.logits_processor
    meta = _plain_tp_logits(fd, lp, forward_batch, metadata)
    if meta is None:
        return lp(
            input_ids,
            hidden_states,
            nextn.lm_head,
            forward_batch if metadata is None else metadata,
        )
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
    extend = meta.forward_mode.is_draft_extend_v2()
    served = (
        si.num_added_elements == 0
        and si.org_vocab_start_index == fd.rank * shard.shape[1]
    )
    if served and _argmax_partials_requested(lp):
        # SGLANG_SPEC_DRAFT_TOPK1_SINK: in a draft-decode step the argmax writes the
        # finalizer's outputs itself (SGLang's loop then reads the sink)
        sink = None if extend or sample_indices is not None else _topk1_sink()
        if sink is not None and fd.top1_sink(shard, si.num_org_elements, sink):
            return LogitsProcessorOutput(
                next_token_logits=None,
                hidden_states=hidden_states_to_store,
                mm_input_embeds=meta.mm_input_embeds,
            )
        # SGLang's draft_argmax_only forward (SGLANG_SPEC_DRAFT_ARGMAX_EXCHANGE): the
        # top-1 as its partials, which its draft loop and draft extend finalize
        partials = fd.top1_partials(shard, si.num_org_elements, extend=extend)
        if partials is not None:
            return LogitsProcessorOutput(
                next_token_logits=None,
                next_token_argmax_partials=(
                    partials if sample_indices is None else partials[sample_indices]
                ),
                hidden_states=hidden_states_to_store,
                mm_input_embeds=meta.mm_input_embeds,
            )
    one_hot = None
    if served:
        one_hot = fd.top1_one_hot(
            shard, si.num_org_elements, lp.vocab_size, extend=extend
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


def _argmax_partials_requested(lp) -> bool:
    """Whether SGLang asks this draft LM-head processor for top-1 partials instead of
    logits (its ``draft_argmax_only`` forward flag on an opted-in processor; the same on
    every rank)."""
    requested = getattr(lp, "_draft_argmax_requested", None)
    return bool(requested is not None and requested())


def _plain_tp_logits(fd: FusedDraft, lp, forward_batch, metadata=None):
    """The batch's LogitsMetadata (``metadata`` when given) when an engaged draft
    step's logits take the plain TP path that ``draft_logits`` builds (decode or draft
    extend v2, no logprobs or scoring, no DP attention or attention-TP lm_head, no
    lm_head all-to-all, logit scale or softcap, no autotune pass), else None."""
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
    meta = (
        LogitsMetadata.from_forward_batch(forward_batch)
        if metadata is None
        else metadata
    )
    if (
        not (meta.forward_mode.is_decode() or meta.forward_mode.is_draft_extend_v2())
        or meta.extend_return_logprob
        or meta.is_prefill_only
    ):
        return None
    return meta


def select_index_rows(forward_batch: ForwardBatch) -> Optional[torch.Tensor]:
    """The draft-extend row select's gate: the rows the draft's LM head reads
    (``select_index``, one per request) when this draft-extend forward may run its MLP
    on them alone, else None. Host-only facts, fixed per captured shape and equal on
    every rank."""
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

    if (
        not forward_batch.forward_mode.is_draft_extend_v2()
        or forward_batch.return_logprob
        # LAST (the draft-extend graph) and NULL capture read only the select_index
        # rows; FULL returns every row's hidden state
        or forward_batch.capture_hidden_mode == CaptureHiddenMode.FULL
        # the MoE masks padded rows by position, which a selection moves
        or forward_batch.num_token_non_padded is not None
    ):
        return None
    select_index = getattr(forward_batch.spec_info, "select_index", None)
    if (
        select_index is None
        or select_index.dim() != 1
        or forward_batch.input_ids is None
    ):
        return None
    if not 0 < select_index.shape[0] < forward_batch.input_ids.shape[0]:
        return None
    return select_index


def row_select_supported(mlp) -> bool:
    """Whether the draft layer's MLP input rows are this rank's tokens in order, so a
    draft extend may run the MLP on the rows its LM head reads alone: SGLang's NextN
    MoE without an all-to-all dispatch (the specs already refuse DP attention, context
    parallelism, PP and LoRA)."""
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

    return isinstance(mlp, DeepseekV2MoE) and not mlp._enable_a2a_moe


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
