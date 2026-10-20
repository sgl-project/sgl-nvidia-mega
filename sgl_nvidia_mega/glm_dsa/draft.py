# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team
"""CuTe DSL kernels for the GLM-5 DSA NextN draft layer (SM103 / SM107, TP8, MTP topk 1).

Built by the draft spec (``draft_spec``) unless ``SGLANG_NVIDIA_MEGAKERNEL_GLM_FUSED_DRAFT=0``. Inside the
draft-decode and draft-extend CUDA graphs:

* the embedding gather and the two eh RMSNorms run as one kernel on the replicated embedding table (needs
  ``SGLANG_ENABLE_EMBED_REPLICATION=1``), bitwise equal to ``fused_eh_norm(F.embedding(...))``;
* ``eh_proj`` runs as a column-sharded GEMV (each rank computes its ``hidden / tp`` output columns) followed by an
  all-gather, instead of the replicated GEMM on every rank; when the target's fused hops are served
  (the target spec), the gather is the hop family's plain all-reduce of a zero-padded row on
  the target's mailbox (each column has one non-zero contributor; a -0 column comes out +0);
* the draft top-1 runs as a distributed argmax over each rank's logit shard (one cooperative kernel that exchanges a
  (value, index) candidate per row through a symmetric-memory mailbox) instead of the logits all-gather; the stock
  ``draft_topk1_postprocess`` then reads a one-hot logits row, so the token it picks is unchanged.

Every eager step and every unsupported batch runs the stock code.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist

from sglang.srt.distributed import tensor_model_parallel_all_gather
from sglang.srt.distributed.parallel_state import get_tp_group
from sgl_nvidia_mega.glm_dsa import (
    draft_attention,
    eligibility,
    runtime,
)
from sgl_nvidia_mega.glm_dsa.eligibility import HIDDEN, TP_SIZE
from sgl_nvidia_mega.glm_dsa.providers import compile_cached, cu_stream

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

# draft decode: one row per request; draft extend: 4 per request
EH_ROWS = (1, 2, 3, 4, 8, 12, 16)
# the draft-extend graphs select one row per request before the lm_head, so both consumers see B <= 4 rows
TOP1_DECODE_ROWS = 8
TOP1_EXTEND_ROWS = 8
# a peer that never publishes makes the waiting ranks trap after this bound (longer than any first-use
# JIT skew between ranks; far below the collective watchdog), after writing the error record
TOP1_TIMEOUT_NS = 30_000_000_000
# call-site ids of the draft's waits in the error record (the target's hops use layer * 4 + kind)
SITE_TOP1_DECODE, SITE_TOP1_EXTEND, SITE_EH_GATHER = 1001, 1002, 1003
SITE_NAMES = {
    SITE_TOP1_DECODE: "draft top-1 (decode)",
    SITE_TOP1_EXTEND: "draft top-1 (extend)",
    SITE_EH_GATHER: "draft eh gather",
}
# rows per hop launch (the target's mailbox instance A); 12 / 16 rows run as two launches
HOP_ROWS = 8
# value table of this engine's eh-norm arithmetic (fused_eh_norm as SGLang JIT-compiles it)
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


class FusedDraft:
    """Kernels, derived weights and top-1 mailboxes of the NextN draft layer."""

    def __init__(self) -> None:
        self.armed = False
        self.eh_weight: Optional[torch.Tensor] = None
        self._eh = {}
        self._front = None
        self._top1 = {}
        self.error_record = None
        self._hops = None  # the target's hop kernels (mailboxes, record) when served
        self._pad: Optional[torch.Tensor] = None
        self._nextn = None
        self._attn_family = None  # the draft layer's attention (draft_attention)
        self._attn_agreed = {}  # (mode, rows) -> every rank serves the kernels there

    def arm(self, nextn) -> None:
        """After the target's embedding / lm_head are shared: derive, compile, rendezvous."""
        from sgl_nvidia_mega.glm_dsa.kernels.cute import cutlass_supported

        ok, why = cutlass_supported()
        if not ok:
            logger.warning(
                "GLM-DSA fused draft: %s; the draft runs the stock path.", why
            )
            return
        from sgl_nvidia_mega.glm_dsa.kernels.cute.draft.api import compile_specs

        tp = get_tp_group()
        self.rank, self.world = tp.rank_in_group, tp.world_size
        self.n_local = HIDDEN // self.world
        self.col_off = self.rank * self.n_local
        w = nextn.model.eh_proj.weight
        if w.dtype != torch.bfloat16 or tuple(w.shape) != (HIDDEN, 2 * HIDDEN):
            raise ValueError(
                f"GLM-DSA fused draft: eh_proj {tuple(w.shape)} {w.dtype} is not the GLM-5 shape"
            )
        self.eh_weight = w[self.col_off : self.col_off + self.n_local].contiguous()
        specs = compile_specs(
            eh_rows=EH_ROWS,
            top1_rows=(TOP1_DECODE_ROWS, TOP1_EXTEND_ROWS),
            eh_norm=EH_NORM_OPTIONS,
        )
        for rows in EH_ROWS:
            spec = specs[f"eh_proj_shard_{rows}"]
            self._eh[rows] = (spec[0], _compile(*spec))
        self._front = _compile(*specs["embed_eh_norm"])
        dev = w.device
        self._nextn = nextn
        self._arm_hops(dev)
        self._arm_attention(nextn)
        if self.error_record is None:
            from sgl_nvidia_mega.glm_dsa.kernels.cute.hop.transport import ErrorRecord

            self.error_record = ErrorRecord(dev)
            if runtime.TARGET is not None:
                # the target's reporter logs the draft top-1's record too
                runtime.TARGET.errors.add(self.failure)
        for name, rows in (("decode", TOP1_DECODE_ROWS), ("extend", TOP1_EXTEND_ROWS)):
            spec = specs[f"distributed_argmax_{rows}"]
            mailbox = _symm_mailbox(spec[0].mailbox_words(self.world), tp, dev)
            ctrl = torch.zeros(spec[0].CTRL_WORDS, dtype=torch.int32, device=dev)
            self._top1[name] = (spec[0], _compile(*spec), mailbox, ctrl)
        torch.cuda.synchronize(dev)
        dist.barrier(group=tp.cpu_group)
        self.armed = True
        logger.info(
            "GLM-DSA fused draft armed: eh_proj shard rows %s, top-1 rows <= %d / %d; derived eh_proj shard %.1f MB; "
            "hops on the target's mailbox: %s; attention kernels: %s.",
            EH_ROWS,
            TOP1_DECODE_ROWS,
            TOP1_EXTEND_ROWS,
            self.eh_weight.numel() * 2 / 1e6,
            "eh gather" if self._hops is not None else "no",
            "MLA prologue + extend q_a / kv_a GEMM" if self._attn_family else "no",
        )

    def _arm_attention(self, nextn) -> None:
        """Build and compile the draft layer's attention kernels when every rank can (else the
        draft layer's attention stays stock on every rank)."""
        family, why = None, None
        try:
            family = draft_attention.DraftAttentionGlue(nextn)
            family.compile(EH_ROWS)
        except ValueError as e:
            family, why = None, str(e)
        whys = runtime.tp_gather([] if why is None else [why])
        if any(whys):
            if self.rank == 0:
                logger.warning(
                    "GLM-DSA fused draft attention stays stock on every rank: %s",
                    runtime.per_rank(whys),
                )
            return
        self._attn_family = family

    def _arm_hops(self, dev) -> None:
        """Compile the eh gather when the target serves its hops (the same decision on every rank:
        the target's ranks agreed on its hop family)."""
        family = _target_hop_family()
        if family is None:
            return
        from sgl_nvidia_mega.glm_dsa.kernels.cute.hop import api as hop_api

        hops = family.kernels
        specs = {s.name: s for s in hop_api.unit_specs(plain_allreduce=True)}
        spec = specs["allreduce"]
        if "allreduce" not in hops.compiled:
            # the eh gather is the family's plain all-reduce unit (its own cache key)
            hops.compiled["allreduce"] = _compile(spec.obj, spec.constexprs, spec.args)
        # static: only this rank's columns are ever written, the others stay +0
        self._pad = torch.zeros(max(EH_ROWS), HIDDEN, dtype=torch.bfloat16, device=dev)
        self._hops = hops
        # the draft's waits write the target's record: one record, one reporter per process
        self.error_record = hops.err
        family.register_sites(SITE_NAMES)

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
            Uint64(TOP1_TIMEOUT_NS),
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

    def attention_for(self, rows: int, forward_batch: ForwardBatch):
        """The draft layer's attention step group served at ``rows`` in this forward mode (a
        ``draft_attention.DraftAttention``), or None: the stock decoder layer runs. At the first
        capture of each (mode, rows), every rank reports the host-only facts the kernels need; any
        reason on any rank keeps that draft graph stock on every rank (a host-only gather on the
        CPU group, legal inside the capture)."""
        family = self._attn_family
        if family is None:
            return None
        mode = "decode" if forward_batch.forward_mode.is_decode() else "extend"
        agreed = self._attn_agreed.get((mode, rows))
        if agreed is None:
            gathered = runtime.tp_gather(
                list(family.step_reasons(self._nextn, forward_batch))
            )
            agreed = self._attn_agreed[(mode, rows)] = not any(gathered)
            if self.rank == 0 and agreed:
                logger.info(
                    "GLM-DSA fused draft attention on: draft %s graph, %d rows",
                    mode,
                    rows,
                )
            elif self.rank == 0:
                logger.warning(
                    "GLM-DSA fused draft attention stays stock: draft %s graph, %d rows: %s",
                    mode,
                    rows,
                    runtime.per_rank(gathered),
                )
        return family.for_rows(rows) if agreed else None

    def run(self, model, input_ids, positions, forward_batch: ForwardBatch):
        """The served draft forward of an engaged step."""
        attention = self.attention_for(int(positions.shape[0]), forward_batch)
        return self.forward_steps(model, input_ids, positions, forward_batch, attention)

    def forward_steps(self, model, input_ids, positions, forward_batch, attention=None):
        """``DeepseekModelNextN.forward`` (CUDA path) with the fused front and the sharded eh_proj (each
        stock unless engaged), and the decoder layer's attention through ``attention`` (a step group; None:
        the stock decoder layer)."""
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
                    draft_attention.decoder_layer_forward, model.decoder, attention
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


def make_draft_logits_processor(config, fused_draft: FusedDraft):
    """The stock logits processor whose TP logits path serves the draft top-1 from the distributed argmax."""
    from sglang.srt.layers.logits_processor import LogitsProcessor

    class GlmDsaDraftLogitsProcessor(LogitsProcessor):
        def _get_logits(
            self,
            hidden_states,
            lm_head,
            logits_metadata,
            embedding_bias=None,
            use_logits_buffer=True,
        ):
            fd = fused_draft
            mode = logits_metadata.forward_mode
            if (
                not fd.engaged()
                or embedding_bias is not None
                or self.logit_scale is not None
                or self.final_logit_softcapping
                or self.do_tensor_parallel_all_gather_dp_attn
                or not self.do_tensor_parallel_all_gather
                or self.use_attn_tp_group
                or not (mode.is_decode() or mode.is_draft_extend_v2())
            ):
                return super()._get_logits(
                    hidden_states,
                    lm_head,
                    logits_metadata,
                    embedding_bias,
                    use_logits_buffer,
                )
            si = lm_head.shard_indices
            shard = self._compute_lm_head(hidden_states, lm_head, embedding_bias)
            one_hot = None
            if (
                si.num_added_elements == 0
                and si.org_vocab_start_index == fd.rank * shard.shape[1]
            ):
                one_hot = fd.top1_one_hot(
                    shard,
                    si.num_org_elements,
                    self.vocab_size,
                    extend=mode.is_draft_extend_v2(),
                )
            # same ranks, same decision: everyone takes the stock gather
            if one_hot is None:
                one_hot = self._logits_gatherer(shard)
            return self._copy_logits_to_buffer(
                one_hot, logits_metadata, use_buffer=use_logits_buffer
            )

    return GlmDsaDraftLogitsProcessor(config)


def _compile(kernel, constexprs, args):
    """``cute.compile(kernel, *args)`` through the fused path's compile cache."""
    import cutlass.cute as cute

    return compile_cached(kernel, constexprs, lambda: cute.compile(kernel, *args))


def _target_hop_family():
    """The prepared target's hop family when its fused hops are served on every
    rank, else None."""
    rt = runtime.TARGET
    family = rt.families.get("hops") if rt is not None else None
    return family if getattr(family, "kernels", None) is not None else None


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
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() in eligibility._SM_ACCEPTED,
        "an SM103 or SM107 GPU",
    )
    need(getattr(cfg, "hidden_size", None) == HIDDEN, f"hidden size {HIDDEN} (GLM-5)")
    need(par.tp_size == TP_SIZE, f"tensor parallel size {TP_SIZE}")
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
    if problems:
        raise ValueError(
            "the GLM-5 DSA fused draft needs: " + "; ".join(problems) + "."
        )


def create_fused_draft(model) -> FusedDraft:
    check_draft_config(model)
    return FusedDraft()
