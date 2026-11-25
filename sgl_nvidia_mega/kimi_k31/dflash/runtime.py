# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""The DFlash runtime of the Kimi K3.1 target (``BuiltK31Verify.dflash_runtime``).

SGLang's DFlash worker binds it through
``sglang.srt.kernel_plugins.nvidia.kimi_k31_dflash_binding`` and hands it three whole
phases:

* ``__init__(binding, built)``: the end of the worker's ``__init__``, before the KV pools
  are sized, so the sampled accept's build comes out of the KV budget. Collective.
* ``draft_sampler(lm_head)``: before the draft cuda graph captures, the megakernel head
  that proposes inside it, when the draft chain's head serves every captured graph.
  Collective.
* ``decode_round(batch, ...)``: the whole decode round, on every target TP rank:

      draft graph replay (MK chain + head) -> target verify graph -> sampled accept on
      the kept vocab shard (or SGLang's greedy accept on the target's argmax) ->
      simulated acceptance -> recurrent-state commit (KDA) -> publish -> aux graph
      (main stream)

  It returns None, launching nothing, for a round the megakernel kernels do not serve
  (``_unserved`` names the reason, logged once per reason on TP rank 0); SGLang then
  runs its own round. Every condition is the same on every rank.

It reads the worker only through the names it binds (``bindings``). It declares no
setup guards: what it serves is what the megakernel build offers. The target build
refuses setups its kernels do not support (``SPEC``), the target names this runtime
only once a draft is built on the megakernel chain, and the chain's offers (rows, head
requests, widths) decide which rounds run here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, NamedTuple, Optional

import torch
from sglang.kernels.ops.memory.small_copy import try_small_copy
from sglang.kernels.ops.speculative.dflash import (
    _prepare_dflash_draft_block_unchecked,
)
from sglang.srt.environ import envs
from sglang.srt.kernel_plugins.nvidia.binding import (
    built_megakernel,
    megakernel_agree,
)
from sglang.srt.layers.dp_attention import require_mlp_sync
from sglang.srt.lora.layers import unwrap_lora_layer
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpaceKind
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    enable_num_token_non_padded,
)
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode, ForwardMode
from sglang.srt.runtime_context import get_exec, get_parallel, get_spec
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_utils import (
    apply_dflash_simulated_acceptance,
    apply_dflash_verify_logits_adjustments,
)
from sglang.srt.speculative.dflash_worker_v2 import (
    _commit_accept,
    _is_all_greedy,
    _resolve_dflash_embedding_module,
    _verify_logits_unadjusted,
)
from sglang.srt.speculative.draft_worker_common import make_draft_input_v2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    SIMULATE_ACC_LEN,
    SIMULATE_ACC_METHOD,
    SIMULATE_ACC_TOKEN_MODE,
    draft_tp_context,
)

from .aux_graphs import AuxGraphs
from .commit import KdaCommit
from .protocol import DRAFT_CHAINS, NO_DRAFT_CHAIN, offered
from .sampler import MegakernelSelectorDraftSampler

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch

# The SGLang names this module reads, by file under the sglang package; test/unit/test_contract.py pins them.
SGLANG_READS = {
    "kernels/ops/memory/small_copy.py": ("try_small_copy",),
    "kernels/ops/speculative/dflash.py": ("_prepare_dflash_draft_block_unchecked",),
    "srt/kernel_plugins/nvidia/binding.py": ("built_megakernel", "megakernel_agree"),
    "srt/layers/dp_attention.py": ("require_mlp_sync",),
    "srt/lora/layers.py": ("unwrap_lora_layer",),
    "srt/managers/utils.py": ("GenerationBatchResult",),
    "srt/mem_cache/kv/alloc/id_space.py": ("IdSpaceKind", "SLIDING_WINDOW"),
    "srt/mem_cache/kv/alloc/iteration_ids.py": ("write_ids", "reads_from", "bind"),
    "srt/mem_cache/kv/alloc/runner_ids.py": ("plan", "space"),
    "srt/model_executor/forward_batch_info.py": (
        "ForwardBatch",
        "enable_num_token_non_padded",
    ),
    "srt/model_executor/forward_mode.py": ("CaptureHiddenMode", "ForwardMode"),
    "srt/runtime_context.py": ("get_exec", "get_parallel", "get_spec"),
    "srt/environ.py": ("envs", "SGLANG_OPT_DFLASH_DEFER_PLAIN_GATHER"),
    "srt/arg_groups/fields/*.py": (
        "speculative_draft_window_size",
        "speculative_dflash_draft_kv_ring",
        "speculative_use_block_verification",
    ),
    "srt/speculative/dflash_info.py": ("DFlashVerifyInput", "prepare_for_verify"),
    "srt/speculative/dflash_utils.py": (
        "apply_dflash_simulated_acceptance",
        "apply_dflash_verify_logits_adjustments",
        "is_domino",
    ),
    "srt/speculative/dflash_worker_v2.py": (
        "_commit_accept",
        "_is_all_greedy",
        "_resolve_dflash_embedding_module",
        "_verify_logits_unadjusted",
    ),
    "srt/speculative/draft_worker_common.py": ("make_draft_input_v2",),
    "srt/speculative/spec_info.py": ("SpeculativeAlgorithm",),
    "srt/speculative/spec_utils.py": (
        "SIMULATE_ACC_LEN",
        "SIMULATE_ACC_METHOD",
        "SIMULATE_ACC_TOKEN_MODE",
        "draft_tp_context",
    ),
    "srt/layers/logits_processor.py": (
        "vocab_size",
        "final_logit_softcapping",
        "defer_plain_gather",
        "is_deferred_shard",
        "deferred_argmax",
    ),
    "srt/layers/vocab_parallel_embedding.py": ("num_embeddings_per_partition",),
    "srt/models/dflash.py": ("candidate_selector", "draft_config", "top_k"),
    "srt/managers/schedule_batch.py": (
        "ScheduleBatch",
        "has_grammar",
        "return_logprob",
        "can_run_decode_cuda_graph",
        "out_cache_loc",
        "reqs",
        "kv_allocated_len",
    ),
    "srt/sampling/sampling_batch_info.py": (
        "temperatures",
        "need_top_k_sampling",
        "need_top_p_sampling",
    ),
}
# SGLang code moved here (test/unit/test_contract.py): while SGLang still defines it,
# its digest must match. _plain_deferral_refused, _arm_plain_deferral and
# _prepare_megakernel_sampled_accept are copies; the head attach, the aux offer and the
# sampled-accept gate are draft_sampler, __init__ and _unserved.
SGLANG_MOVED = {
    "srt/speculative/dflash_worker_v2.py:DFlashWorkerV2._prepare_megakernel_sampled_accept": "f1c3d06541d47fdf",
    "srt/speculative/dflash_worker_v2.py:DFlashWorkerV2._plain_deferral_refused": "a926ed410dfdf354",
    "srt/speculative/dflash_worker_v2.py:DFlashWorkerV2._arm_plain_deferral": "a4940f4de693d0f9",
    "srt/speculative/dflash_worker_v2.py:DFlashWorkerV2._maybe_attach_megakernel_head": "7a08ff6f804278e0",
    "srt/speculative/dflash_worker_v2.py:DFlashWorkerV2._maybe_megakernel_aux": "092aaae84c5d6358",
    "srt/speculative/dflash_worker_v2.py:DFlashWorkerV2._can_megakernel_sample": "b4b2f0926f05c18f",
}
# What this runtime pulls from SGLang's DFlash worker (kimi_k31_dflash_binding.BINDINGS)
# and the guard values it serves (kimi_k31_dflash_binding.GUARDS);
# test/unit/test_contract.py pins both against SGLang's binding.
DFLASH_BINDINGS = frozenset(
    {
        "block_size",
        "device",
        "mask_token_id",
        "noise_embed_scale",
        "batched_small_copy",
        "draft_model",
        "draft_model_runner",
        "draft_parallel_layout",
        "draft_block_spec_info",
        "target_worker",
        "target_logits_processor",
        "sim_acc_rng",
        "need_mamba_verify_commit",
        "compute_compact_draft_seq_lens",
        "fill_compact_seq_lens_cpu_bound",
        "rebuild_compact_draft_cache",
        "accept_block",
    }
)
# No setup guards: the build offers what the runtime runs on (see the module docstring).
DFLASH_GUARDS: dict = {}

logger = logging.getLogger(__name__)


class _DraftBlock(NamedTuple):
    """One round's draft block, written by the draft and read by the verify and aux."""

    prefix_lens: torch.Tensor  # [bs], the committed lengths (batch.seq_lens)
    block_ids: torch.Tensor  # [bs, block]: the bonus token, then mask ids
    positions: torch.Tensor  # [bs * block]
    verify_out_cache_loc: torch.Tensor  # [bs * block], the verify's KV slots
    verify_out_cache_loc_2d: torch.Tensor  # [bs, block], the same slots
    draft_tokens: torch.Tensor  # [bs, block], filled by _draft
    iteration_ids: Any  # the target runner's plan of this round's slots
    forward_batch: ForwardBatch  # the draft forward's


class K31DFlashRuntime:
    """One per DFlash worker, over the target's ``BuiltK31Verify``."""

    bindings = DFLASH_BINDINGS
    guards = DFLASH_GUARDS

    def __init__(self, binding: Any, built: Any) -> None:
        pull = binding.pull
        self._target = built
        self._target_tp_rank = get_parallel().tp_rank
        self.device = pull("device")
        self.block_size = pull("block_size")
        self.draft_model = pull("draft_model")
        self.selector = self.draft_model.candidate_selector
        self._target_worker = pull("target_worker")
        self._target_model = self._target_worker.model_runner.model
        self._target_logits_processor = pull("target_logits_processor")
        self._mask_token_id = pull("mask_token_id")
        self._noise_embed_scale = pull("noise_embed_scale")
        self._batched_small_copy = pull("batched_small_copy")
        self._draft_parallel_layout = pull("draft_parallel_layout")
        self._draft_block_spec_info = pull("draft_block_spec_info")
        self._simulated_acc_len_rng = pull("sim_acc_rng")
        self._compute_compact_draft_seq_lens = pull("compute_compact_draft_seq_lens")
        self._fill_compact_seq_lens_cpu_bound = pull("fill_compact_seq_lens_cpu_bound")
        self._rebuild_compact_draft_cache = pull("rebuild_compact_draft_cache")
        self._accept_block = pull("accept_block")
        self._binding = binding
        spec = get_spec()
        # The worker's own sources for these (dflash_worker_v2.py __init__).
        self._compact_draft_cache = spec.speculative_draft_window_size is not None
        self._draft_kv_ring = spec.speculative_dflash_draft_kv_ring
        self._block_verification = spec.speculative_use_block_verification
        self._is_domino = self.draft_model.draft_config.is_domino
        # set by draft_sampler, before the draft graph captures
        self._sampler = None
        self._draft_runner = None
        self._aux = None
        self._kda = None
        self._recurrent_commit = False
        self._round_bs = frozenset()
        # Why no round runs here; draft_sampler clears it when the head attaches.
        self._stock_reason = "no megakernel draft head (no draft cuda graph captured)"
        self._declined = set()
        # SGLang sets a batch's decode-graph flag only under MLP sync.
        self._mlp_sync = require_mlp_sync()
        self._embed_module = None
        self._block_bufs = None
        draft_built = built_megakernel(self.draft_model)
        self._draft_chain = NO_DRAFT_CHAIN
        if any(chain is draft_built for chain in DRAFT_CHAINS):
            self._draft_chain = draft_built
        draft = self._draft_chain
        # Ranks that differ would wait in different collectives below.
        megakernel_agree(
            (draft.draft_aux_rows, draft.draft_head_requests),
            what="DFlash draft chain offers",
        )
        self._aux_chain = offered(draft, draft.draft_aux_rows)
        # None: sampled rounds run SGLang's (_unserved); greedy rounds still run here. A
        # chain without the aux or the head serves no round (_round_reason, draft_sampler),
        # so nothing is built for it (rank-uniform: the offers agree above).
        self._megakernel_sampled = (
            self._prepare_megakernel_sampled_accept()
            if self._aux_chain is not None and draft.draft_head_requests
            else None
        )

    # ---- init phases

    def _prepare_megakernel_sampled_accept(self):
        target_lp = self._target_logits_processor
        built = self._target
        # Declined before the build: it is collective.
        if (
            built is None
            or target_lp is None
            or self._plain_deferral_refused("sampled accept")
        ):
            return None
        lm_head = unwrap_lora_layer(self._target_model.lm_head)
        if not built.prepare_sampled_accept(
            vocab_size=int(target_lp.vocab_size),
            softcap=target_lp.final_logit_softcapping or None,
            shard_width=int(lm_head.num_embeddings_per_partition),
            seed=int(torch.initial_seed()),
            topk=int(self.selector.top_k),
        ):
            return None
        self._arm_plain_deferral("sampled accept")
        return built

    def _plain_deferral_refused(self, consumer: str) -> bool:
        target_lp = self._target_logits_processor
        plain = envs.SGLANG_OPT_DFLASH_DEFER_PLAIN_GATHER
        refused = (
            target_lp is not None
            and not target_lp.final_logit_softcapping
            and plain.is_set()
            and not plain.get()
        )
        if refused and self._target_tp_rank == 0:
            logger.info(
                "DFLASH %s off: SGLANG_OPT_DFLASH_DEFER_PLAIN_GATHER=0 keeps the full "
                "verify gather",
                consumer,
            )
        return refused

    def _arm_plain_deferral(self, consumer: str) -> None:
        """Must run before the graphs capture."""
        target_lp = self._target_logits_processor
        if (
            target_lp is None
            or target_lp.final_logit_softcapping
            or target_lp.defer_plain_gather
        ):
            return
        target_lp.defer_plain_gather = True
        if self._target_tp_rank == 0:
            logger.info(
                "DFLASH plain verify deferral on (the %s reads the shard)", consumer
            )

    def draft_sampler(self, lm_head: Any) -> Optional[MegakernelSelectorDraftSampler]:
        """The megakernel head, which proposes inside the draft forward, or None for
        SGLang's sampler; before the draft graph captures (after the worker's attention
        backends). Collective; every input is the same on every rank."""
        self._draft_runner = self._binding.pull("draft_model_runner")
        self._recurrent_commit = self._binding.pull("need_mamba_verify_commit")
        self._embed_module = unwrap_lora_layer(
            _resolve_dflash_embedding_module(self.draft_model, self._target_model)
        )
        self._aux = AuxGraphs(
            megakernel_aux=self._aux_chain,
            draft_model=self.draft_model,
            draft_kv_pool=self._draft_runner.token_to_kv_pool,
        )
        self._kda = KdaCommit(
            megakernel_kda=self._target,
            target_attn_backend=self._target_worker.model_runner.attn_backend,
            device=self.device,
        )
        chain = self._draft_chain
        head = offered(chain, chain.draft_head_requests)
        graph_bs = set(get_exec().graph.cuda_graph_config.decode.bs)
        block_size = int(self.block_size)
        # The head is folded into every captured draft graph, after the chain's forward:
        # it serves only if the chain runs each graph's rows (B1 x 8 today).
        if (
            head is None
            or not head.draft_head_sampling
            or not graph_bs <= set(head.draft_head_requests)
            or not {bs * block_size for bs in graph_bs} <= set(chain.widths)
        ):
            self._decline_all(
                f"no megakernel draft head for decode graphs {sorted(graph_bs)} "
                f"x block {block_size} (the chain runs rows {list(chain.widths)}, "
                f"its head batch sizes {list(chain.draft_head_requests)})"
            )
            return None
        self._stock_reason = self._round_reason()
        rows = set(chain.draft_aux_rows)
        self._round_bs = frozenset(bs for bs in graph_bs if bs * block_size in rows)
        if self._stock_reason is not None:
            self._decline_all(self._stock_reason)
        elif self._target_tp_rank == 0:
            logger.info(
                "DFLASH rounds run on the NVIDIA megakernel at batch sizes %s "
                "(aux rows %s, on the main stream)",
                sorted(self._round_bs),
                sorted(rows),
            )
        tokens, candidates, q = head.attach_draft_head(
            lm_head=lm_head,
            selector=self.draft_model.candidate_selector,
            draft_config=self.draft_model.draft_config,
        )
        if self._target_tp_rank == 0:
            logger.info(
                "DFLASH selector decode on the NVIDIA megakernel head (%s).",
                "greedy and sampling requests",
            )
        self._sampler = MegakernelSelectorDraftSampler(
            head=head,
            out=tokens,
            candidate_out=candidates,
            q_out=q,
            sampling_enabled=True,
        )
        return self._sampler

    def _round_reason(self) -> Optional[str]:
        """Why no round can run here although the head serves the draft graphs: the
        aux and KDA kernels' layouts."""
        if self._aux_chain is None:
            return "no megakernel aux on the draft chain (see the draft MK build log)"
        # The aux writes the per-slot draft table the ring replaces.
        if self._draft_kv_ring:
            return "the draft KV ring (the megakernel aux writes per-slot draft rows)"
        # The aux graphs hold no sliding-window locations.
        if self._draft_runner.runner_ids.space(IdSpaceKind.SLIDING_WINDOW) is not None:
            return "a sliding-window draft KV sub-pool (the megakernel aux writes none)"
        if self._recurrent_commit and self._kda._kda_state() is None:
            return "a recurrent state the megakernel KDA commit is not laid out for"
        return None

    def _decline_all(self, reason: str) -> None:
        self._stock_reason = reason
        self._round_bs = frozenset()
        self._declined.add(reason)
        if self._target_tp_rank == 0:
            logger.warning(
                "K3.1 DFlash runtime: every round runs SGLang's round: %s", reason
            )

    # ---- the round

    def decode_round(
        self, batch: ScheduleBatch, *, on_publish, grammar_barrier
    ) -> Optional[GenerationBatchResult]:
        """The whole decode round, or None (nothing launched) for SGLang's."""
        sampling_info = batch.sampling_info
        bs = len(batch.seq_lens)
        sampled = not _is_all_greedy(sampling_info)
        unserved = self._unserved(batch, sampling_info, bs=bs, sampled=sampled)
        if unserved is not None:
            if unserved not in self._declined:
                self._declined.add(unserved)
                if self._target_tp_rank == 0:
                    logger.warning(
                        "K3.1 DFlash runtime: a round with %s runs SGLang's round "
                        "(logged once per reason)",
                        unserved,
                    )
            return None
        return self._round(
            batch, sampling_info, bs=bs, sampled=sampled, on_publish=on_publish
        )

    def _unserved(self, batch, sampling_info, *, bs: int, sampled: bool):
        """Why the round runs SGLang's way, or None; rank-uniform, before any launch."""
        if self._stock_reason is not None:
            return self._stock_reason
        if bs not in self._round_bs:
            return (
                f"batch size {bs}: the megakernel chain serves {sorted(self._round_bs)}"
            )
        # Without MLP sync the draft graph runs whenever bs is captured.
        if self._mlp_sync and not batch.can_run_decode_cuda_graph:
            return "no decode cuda graph under MLP sync"
        if batch.has_grammar:
            return "a grammar"
        if batch.return_logprob:
            return "logprobs"
        if sampling_info is None:
            return None
        # Nor penalties, logit bias, custom processors or sampling masks.
        if not _verify_logits_unadjusted(sampling_info):
            return "adjusted logits"
        if not sampled:
            return None
        if sampling_info.need_top_k_sampling or sampling_info.need_top_p_sampling:
            return "top-k/top-p"
        # The megakernel sampled accept accepts per token.
        if self._block_verification:
            return "block verification of a sampled batch"
        if self._megakernel_sampled is None:
            return "no megakernel sampled accept for this target (see its build log)"
        if not self._megakernel_sampled.sampled_accept_fits(bs):
            return "a sampled accept that does not fit"
        return None

    def _round(self, batch, sampling_info, *, bs, sampled, on_publish):
        """Every step launches on the current stream, in this order, on every rank."""
        # `seq_lens` is carried over from the previous overlap iteration and may have been
        # produced on another stream.
        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )
        block = self._prepare_block(batch, bs=bs)
        selector_sample = self._draft(block, sampling_info, bs=bs, sampled=sampled)
        target_out, seq_lens_pre_verify = self._verify(batch, block)
        logits_output = target_out.logits_output
        _, commit_lens, bonus, out_tokens, new_seq_lens = self._accept(
            batch, block, logits_output, sampling_info, selector_sample, bs=bs
        )
        new_seq_lens = self._commit(
            batch, block, seq_lens_pre_verify, commit_lens, new_seq_lens
        )
        # Before the aux: the next round's host reads the lengths without waiting on it.
        if on_publish is not None:
            on_publish(new_seq_lens)
        self._append_aux(block, logits_output, commit_lens, bs=bs)
        return self._result(target_out, out_tokens, commit_lens, bonus, new_seq_lens)

    def _block_buffers(self, bs: int) -> tuple:
        bufs = self._block_bufs
        if bufs is None or bufs[0].shape[0] < bs:
            cap = max(bs, 0 if bufs is None else 2 * bufs[0].shape[0])
            block, device = int(self.block_size), self.device
            bufs = self._block_bufs = (
                torch.empty((cap, block), dtype=torch.long, device=device),
                torch.empty((cap, block), dtype=torch.int64, device=device),
                torch.empty((cap, block), dtype=torch.int64, device=device),
                torch.empty((cap, block), dtype=torch.long, device=device),
                torch.empty((cap,), dtype=torch.int32, device="cpu"),
            )
        return tuple(buf[:bs] for buf in bufs)

    def _prepare_block(self, batch, *, bs) -> _DraftBlock:
        """The draft block's ids, positions, verify slots and embeddings, the draft's
        compact KV table, and the draft forward's batch."""
        device = self.device
        draft_input = batch.spec_info
        target_runner = self._target_worker.model_runner
        block_size = int(self.block_size)
        block_ids, positions_2d, verify_out_cache_loc_2d, draft_tokens, seq_lens_cpu = (
            self._block_buffers(bs)
        )
        prefix_lens = batch.seq_lens
        _prepare_dflash_draft_block_unchecked(
            bonus_tokens=draft_input.bonus_tokens.view(-1),
            prefix_lens=prefix_lens.view(-1),
            req_pool_indices=batch.req_pool_indices.view(-1),
            req_to_token=target_runner.req_to_token_pool.req_to_token,
            block_ids_out=block_ids,
            positions_out=positions_2d,
            cache_loc_out=verify_out_cache_loc_2d,
            mask_token_id=int(self._mask_token_id),
        )
        noise_embedding = self._embed_module(block_ids)
        if self._noise_embed_scale != 1.0:
            noise_embedding = noise_embedding * self._noise_embed_scale
        input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])
        positions = positions_2d.reshape(-1)
        verify_out_cache_loc = verify_out_cache_loc_2d.reshape(-1)
        iteration_ids = target_runner.runner_ids.plan(
            req_pool_indices=batch.req_pool_indices,
            seq_lens=prefix_lens,
            seq_lens_cpu=(
                batch.seq_lens_cpu
                if batch.seq_lens_cpu is not None
                else draft_input.nxt_kv_lens_cpu
            ),
            write_virtual=verify_out_cache_loc,
            read_extent=block_size,
        )
        if self._compact_draft_cache:
            # Rebuild the draft-local sliding-window view from committed target state.
            draft_prefix_lens = self._compute_compact_draft_seq_lens(prefix_lens)
            self._fill_compact_seq_lens_cpu_bound(
                batch_seq_lens_cpu=batch.seq_lens_cpu,
                nxt_kv_lens_cpu=draft_input.nxt_kv_lens_cpu,
                out=seq_lens_cpu,
            )
            self._rebuild_compact_draft_cache(
                req_pool_indices=batch.req_pool_indices,
                prefix_lens=prefix_lens,
                draft_prefix_lens=draft_prefix_lens,
                verify_out_cache_loc_2d=verify_out_cache_loc_2d,
                bs=bs,
                block_size=block_size,
            )
            draft_seq_lens = draft_prefix_lens
            draft_seq_lens_sum = int(seq_lens_cpu.sum().item())
        else:
            # Non-windowed path uses the shared overallocated mapping directly.
            # Backend planning only needs a safe upper bound for the committed
            # prefix lengths, not the full allocator reservation length.
            draft_seq_lens = prefix_lens
            if batch.seq_lens_cpu is not None:
                # Host bound = committed prefix + one verify block.
                seq_lens_cpu.copy_(batch.seq_lens_cpu)
                seq_lens_cpu.add_(block_size)
                draft_seq_lens_sum = int(seq_lens_cpu.sum())
            elif draft_input.nxt_kv_lens_cpu is not None:
                # GPU-only backend: reserved is a safe over-estimate.
                seq_lens_cpu.copy_(draft_input.nxt_kv_lens_cpu)
                draft_seq_lens_sum = int(draft_input.nxt_kv_lens_sum)
            else:
                # Allocated lengths include the in-flight verify reservation.
                for i, req in enumerate(batch.reqs):
                    seq_lens_cpu[i] = req.kv.kv_allocated_len
                draft_seq_lens_sum = int(seq_lens_cpu.sum())
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=block_ids.flatten(),
            req_pool_indices=batch.req_pool_indices,
            seq_lens=draft_seq_lens,
            out_cache_loc=verify_out_cache_loc,
            seq_lens_sum=draft_seq_lens_sum,
            seq_lens_cpu=seq_lens_cpu,
            positions=positions,
            input_embeds=input_embeds,
            spec_algorithm=SpeculativeAlgorithm.DFLASH,
            spec_info=self._draft_block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            num_token_non_padded=(
                torch.full((), bs * block_size, dtype=torch.int32, device=device)
                if enable_num_token_non_padded()
                else None
            ),
            global_num_token_non_padded_cpu=bs * block_size,
            can_run_decode_cuda_graph=batch.can_run_decode_cuda_graph,
        )
        # The draft block writes the whole window. A compact draft reads its
        # own `req_to_token` rows over its own lengths.
        draft_translator = self._draft_runner.runner_ids
        draft_plan = (
            iteration_ids.reads_from(
                draft_translator,
                seq_lens=draft_seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                read_extent=block_size,
            )
            if self._compact_draft_cache
            else iteration_ids
        )
        draft_plan.bind(forward_batch, draft_translator)
        return _DraftBlock(
            prefix_lens=prefix_lens,
            block_ids=block_ids,
            positions=positions,
            verify_out_cache_loc=verify_out_cache_loc,
            verify_out_cache_loc_2d=verify_out_cache_loc_2d,
            draft_tokens=draft_tokens,
            iteration_ids=iteration_ids,
            forward_batch=forward_batch,
        )

    def _draft(self, block: _DraftBlock, sampling_info, *, bs, sampled):
        """The draft graph (megakernel chain + head) proposes the block into
        ``block.draft_tokens``; returns the selector's (candidate ids, q rows) for a
        sampled batch, else None (as SGLang's round, a Domino draft accepts greedily)."""
        block_size = int(self.block_size)
        sampler = self._sampler
        # Consumed by the in-graph sample; must be staged before the replay.
        sampler.stage_sampling_params(bs=bs, sampling_info=sampling_info)
        with (
            torch.inference_mode(),
            draft_tp_context(self._draft_parallel_layout),
        ):
            draft_out = self._draft_runner.forward(block.forward_batch)
        if not draft_out.can_run_graph:
            raise RuntimeError(f"K3.1 DFlash runtime: the draft ran eager at bs={bs}")
        draft_next = sampler.out[: bs * (block_size - 1)].view(bs, block_size - 1)
        selector_sample = (
            (sampler.candidate_out[:bs], sampler.q_out[:bs])
            if sampled and not self._is_domino
            else None
        )
        draft_tokens, block_ids = block.draft_tokens, block.block_ids
        # try_small_copy declines what it cannot copy in one launch.
        if not (
            self._batched_small_copy
            and try_small_copy(
                [draft_tokens[:, 0], draft_tokens[:, 1:]], [block_ids[:, 0], draft_next]
            )
        ):
            draft_tokens[:, 0].copy_(block_ids[:, 0])
            draft_tokens[:, 1:].copy_(draft_next)
        return selector_sample

    def _verify(self, batch, block: _DraftBlock) -> tuple:
        """The target verify graph over the drafted block: (its output, the committed
        lengths before it)."""
        block_size = int(self.block_size)
        draft_input = batch.spec_info
        verify_input = DFlashVerifyInput(
            draft_token=block.draft_tokens.reshape(-1),
            positions=block.positions,
            draft_token_num=block_size,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            iteration_ids=block.iteration_ids,
        )
        batch.out_cache_loc = block.verify_out_cache_loc
        seq_lens_pre_verify = batch.seq_lens.clone() if self._recurrent_commit else None
        seq_lens_cpu_backup = batch.seq_lens_cpu
        seq_lens_sum_backup = batch.seq_lens_sum
        if seq_lens_cpu_backup is not None:
            # Verify host bound = committed prefix + one verify block (matches draft).
            verify_host_seq_lens = seq_lens_cpu_backup + block_size
            batch.seq_lens_cpu = verify_host_seq_lens
            batch.seq_lens_sum = int(verify_host_seq_lens.sum())
        elif draft_input.nxt_kv_lens_cpu is not None:
            batch.seq_lens_cpu = draft_input.nxt_kv_lens_cpu
            batch.seq_lens_sum = int(draft_input.nxt_kv_lens_sum)
        verify_forward_batch, _ = verify_input.prepare_for_verify(
            batch, self._target_worker, defer_metadata_init=False
        )
        batch.seq_lens_cpu = seq_lens_cpu_backup
        batch.seq_lens_sum = seq_lens_sum_backup
        target_out = self._target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        return target_out, seq_lens_pre_verify

    def _accept(
        self, batch, block, logits_output, sampling_info, selector_sample, *, bs
    ) -> tuple:
        """(accept_len, commit_lens, bonus, out_tokens, new_seq_lens or None) from the
        verify's logits: the kept vocab shard, or the gathered logits of a greedy batch
        when the plain deferral is refused."""
        draft_tokens, prefix_lens = block.draft_tokens, block.prefix_lens
        shard = logits_output.next_token_logits
        target_lp = self._target_logits_processor
        deferred = target_lp is not None and target_lp.is_deferred_shard(shard)
        if selector_sample is None:
            # A greedy batch: SGLang's greedy accept on the target's argmax, from the
            # kept vocab shard or (deferral refused) the gathered logits.
            if deferred:
                target_predict = target_lp.deferred_argmax(shard)
                logits_output.next_token_logits = None
            else:
                # SGLang's round adjusts gathered logits before its greedy argmax.
                apply_dflash_verify_logits_adjustments(
                    next_token_logits=shard,
                    sampling_info=sampling_info,
                    draft_token_num=int(self.block_size),
                )
                target_predict = torch.argmax(shard, dim=-1)
            return self._accept_block(
                candidates=draft_tokens,
                next_token_logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
                draft_input=batch.spec_info,
                prefix_lens=prefix_lens,
                bs=bs,
                target_predict=target_predict,
                cutoff_verify_lens=None,
            )[:5]
        # The sampled accept, served only with the deferral armed, reads the shard.
        if not deferred or not self._megakernel_sampled.sampled_accept_fits(bs, shard):
            raise RuntimeError(f"K3.1 DFlash runtime: a {shard.dtype} verify shard")
        logits_output.next_token_logits = None
        # Samples on the vocab shards; every rank gets rank 0's decision.
        selector_candidate_ids, selector_q_rows = selector_sample
        accept_len, bonus, target_predict = self._megakernel_sampled.sampled_accept(
            logits=shard,
            temperatures=sampling_info.temperatures,
            candidates=draft_tokens,
            candidate_ids=selector_candidate_ids,
            q_rows=selector_q_rows,
        )
        out_tokens, commit_lens = _commit_accept(draft_tokens, accept_len, bonus)
        new_seq_lens = None
        if SIMULATE_ACC_LEN > 0:
            new_seq_lens = apply_dflash_simulated_acceptance(
                candidates=draft_tokens,
                target_predict=target_predict,
                accept_len=accept_len,
                commit_lens=commit_lens,
                bonus=bonus,
                out_tokens=out_tokens,
                simulate_acc_len=SIMULATE_ACC_LEN,
                simulate_acc_method=SIMULATE_ACC_METHOD,
                simulate_acc_token_mode=SIMULATE_ACC_TOKEN_MODE,
                next_token_logits=None,
                generator=self._simulated_acc_len_rng,
                prefix_lens=prefix_lens,
                new_seq_lens=new_seq_lens,
            )
        return accept_len, commit_lens, bonus, out_tokens, new_seq_lens

    def _commit(self, batch, block, seq_lens_pre_verify, commit_lens, new_seq_lens):
        """The KDA commit of the accepted recurrent state; returns the new lengths."""
        prefix_lens = block.prefix_lens
        if new_seq_lens is None:
            new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
        if not self._recurrent_commit:
            return new_seq_lens
        _require(
            self._kda._megakernel_kda_commit(
                batch=batch,
                seq_lens_pre_verify=seq_lens_pre_verify,
                commit_lens=commit_lens,
            ),
            "the KDA commit declined the target's recurrent state",
        )
        return new_seq_lens

    def _append_aux(self, block: _DraftBlock, logits_output, commit_lens, *, bs):
        """The committed rows' target hidden states into the draft KV: the aux graph,
        replayed on the main stream (a collective)."""
        block_size = int(self.block_size)
        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError(
                "DFLASH verify requires target hidden states, but got None."
            )
        hidden = hidden.view(bs, block_size, -1)
        # Consume in this step: every decode graph size shares one aux output,
        # which the next target forward overwrites (resolve_aux_hidden_states_width).
        cache_loc = block.iteration_ids.write_ids(self._draft_runner.runner_ids)
        self._aux._append_target_hidden_megakernel(
            target_hidden=hidden.reshape(-1, hidden.shape[-1]),
            positions=block.positions,
            cache_loc=cache_loc,
            cache_loc_2d=cache_loc.view(block.verify_out_cache_loc_2d.shape),
            commit_lens=commit_lens,
        )

    def _result(self, target_out, out_tokens, commit_lens, bonus, new_seq_lens):
        logits_output = target_out.logits_output
        # Avoid copying large hidden-state buffers to CPU in overlap scheduling.
        logits_output.hidden_states = None
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=out_tokens.reshape(-1),
            accept_lens=commit_lens,
            can_run_cuda_graph=target_out.can_run_cuda_graph,
            next_draft_input=make_draft_input_v2(
                bonus_tokens=bonus, new_seq_lens=new_seq_lens
            ),
            side_stream_done=None,
            speculative_num_draft_tokens=int(self.block_size),
            new_seq_lens=new_seq_lens,
            block_accept_lens=None,
            cap_lens=None,
            routed_experts_output=target_out.routed_experts_output,
            indexer_topk_output=target_out.indexer_topk_output,
            kv_weight_version_record=target_out.kv_weight_version_record,
        )


def _require(ok: bool, what: str) -> None:
    if not ok:
        raise RuntimeError(f"K3.1 DFlash runtime: {what}")
