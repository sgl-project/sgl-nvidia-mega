# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# ruff: noqa: F821
"""SGLang functions with the speculative step fusions applied, as they read in
SGLang's tree with them: ``eagle_utils._tree_accept`` (the packed verify broadcast,
the lean and the chunked chain verify sampling) and
``EagleDraftWorker._draft_extend_for_decode`` (the fused topk-1 draft-extend tail).

``hooks`` installs each with its SGLang module's globals (``rebind``), so their free
names resolve there exactly as in the originals; the helpers defined here are added
to that module under the same names. Keep each body in sync with its original: a
diff against SGLang's function shows only the fusion.
"""

from __future__ import annotations

import logging

from sgl_nvidia_mega.spec_step import switches as _spec_switches
from sgl_nvidia_mega.spec_step.ops import draft_extend_tail

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/layers/dp_attention.py": ("is_dp_attention_enabled",),
    "srt/runtime_context.py": ("get_parallel", "get_spec"),
    "srt/speculative/eagle_utils.py": (
        "_PACKED_VERIFY_BCAST",
        "_verify_outputs",
        "_verify_uses_greedy",
        "_can_use_sparse_uno_tree_target_sampling",
        "_sharded_tokens",
        "gather_full_logits",
        "_chain_accept_applies",
        "_chain_accept",
        "_apply_joint_top_k_top_p",
        "_verify_coins",
    ),
    "srt/speculative/spec_utils.py": (
        "SIMULATE_ACC_LEN",
        "SIMULATE_ACC_TOKEN_MODE",
        "generate_simulated_accept_index",
        "generate_simulated_accept_index_device",
    ),
    "srt/utils/async_probe.py": ("maybe_detect_nan",),
    "srt/speculative/uno_utils.py": ("sample_uno_tree_target_tokens",),
    "kernels/ops/speculative/eagle.py": ("prepare_draft_extend_inputs",),
}

# The SGLang function bodies lifted here, by the digest of their code
# (test/unit/test_contract.py pins them): a change there is ported here
SGLANG_MIRRORS = {
    "srt/speculative/eagle_utils.py:_tree_accept": "64c3d92c339b7b56",
}

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- eagle_utils

_chunked_verify_sampling_logged = False


def _log_chunked_verify_sampling_once() -> None:
    global _chunked_verify_sampling_logged
    if not _chunked_verify_sampling_logged:
        _chunked_verify_sampling_logged = True
        logger.info(
            "SGLANG_SPEC_CHUNKED_VERIFY_SAMPLING on: chain verify sampling in two "
            "launches (one CTA per vocabulary chunk)."
        )


def _tree_accept(
    *,
    verify_input: EagleVerifyInput,
    batch: ScheduleBatch,
    logits_output: LogitsProcessorOutput,
    next_token_logits: torch.Tensor,
    vocab_shard: Optional[VocabShard],
    sampling_info: SamplingBatchInfo,
    candidates: torch.Tensor,
    uno_target_max_top_k: Optional[int],
    watermark: bool,
    simulated_acc_len: Optional[torch.Tensor],
    simulated_acc_len_rng: Optional[torch.Generator],
    sparse_bonus_argmax: bool,
    local_keys: Optional[LocalKeysFn],
):
    """A gather it still needs is published to ``logits_output``."""
    from sglang.srt.layers.dp_attention import (
        is_dp_attention_enabled,
    )
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.speculative.spec_utils import (
        SIMULATE_ACC_LEN,
        SIMULATE_ACC_TOKEN_MODE,
        generate_simulated_accept_index,
        generate_simulated_accept_index_device,
    )

    device = batch.device
    bs = len(batch.seq_lens)
    predict_shape = list(next_token_logits.shape)[:-1]
    use_rejection_sampling = get_spec().speculative_use_rejection_sampling
    greedy = _verify_uses_greedy(
        is_all_greedy=sampling_info.is_all_greedy,
        is_cpu=_is_cpu,
        is_hip=_is_hip,
        is_xpu=_is_xpu,
        use_rejection_sampling=use_rejection_sampling,
    )
    sparse = not greedy and _can_use_sparse_uno_tree_target_sampling(
        uno_target_max_top_k,
        sampling_info,
    )
    sharded_predict = None
    if vocab_shard is not None:
        temperatures = None
        if not greedy:
            temperatures = torch.repeat_interleave(
                sampling_info.temperatures.view(-1),
                verify_input.draft_token_num,
                dim=0,
            )
        sharded_predict = _sharded_tokens(
            next_token_logits,
            vocab_shard,
            local_keys=local_keys,
            temperatures=temperatures,
            seeds=sampling_info.sampling_seed,
        )
        if sharded_predict is None:
            # A local policy declines before consuming RNG. Rebuild the exact stock
            # logits, publish that mutation, and let the stock sampler consume RNG
            # in its original order.
            next_token_logits = gather_full_logits(
                next_token_logits, vocab_shard, get_parallel().tp_group
            )
            logits_output.next_token_logits = next_token_logits
            logits_output.next_token_logits_vocab_shard = vocab_shard = None
    # The generic fused chain acceptance writes every verify output itself.
    chain = _chain_accept_applies(
        verify_input=verify_input,
        logits=next_token_logits,
        simulated_acc_len=simulated_acc_len,
        simulate_acc_len=SIMULATE_ACC_LEN,
        simulate_acc_token_mode=SIMULATE_ACC_TOKEN_MODE,
        watermark=watermark,
    )
    fused_verify = chain and (greedy or sparse or vocab_shard is not None)
    predict, accept_index, num_correct_drafts, verify_out = _verify_outputs(
        rows=math.prod(predict_shape),
        bs=bs,
        depth=verify_input.max_tree_depth,
        device=device,
        packed=_PACKED_VERIFY_BCAST
        or _spec_switches.enabled("SGLANG_SPEC_PACKED_VERIFY_BROADCAST"),
        init=not fused_verify,
    )

    # Sample tokens
    target_predict = None
    accept_lens = None
    if greedy:
        target_predict = (
            sampling_argmax(next_token_logits)
            if sharded_predict is None
            else sharded_predict
        )
        target_predict = target_predict.reshape(bs, verify_input.draft_token_num)
        if fused_verify:
            accept_lens, accept_index = _chain_accept(
                predict=predict,
                accept_index=accept_index,
                num_correct_drafts=num_correct_drafts,
                candidates=candidates,
                verify_input=verify_input,
                target_predict=target_predict,
                bonus_target=target_predict,
                simulated_acc_len=simulated_acc_len if SIMULATE_ACC_LEN > 0 else None,
                real_draft_tokens=SIMULATE_ACC_TOKEN_MODE == "real-draft-token",
                finish=True,
            )
        else:
            predict, accept_index, num_correct_drafts = verify_tree_greedy_func(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=num_correct_drafts,  # mutable
                candidates=candidates,
                retrieve_index=verify_input.retrieve_index,
                retrieve_next_token=verify_input.retrieve_next_token,
                retrieve_next_sibling=verify_input.retrieve_next_sibling,
                target_predict=target_predict,
            )

        if _is_hip:
            # On ROCm, the per-rank draft tokens can differ, so ranks accept a
            # different number of drafts, desynchronize the committed seq_lens, and
            # deadlock the next TP collective. Broadcast from rank 0 to ensure
            # consistency, the same way the sampling branch below does.
            tp_group = (
                get_parallel().attn_tp_group
                if is_dp_attention_enabled()
                else get_parallel().tp_group
            )
            if tp_group.world_size > 1:
                tp_group.broadcast(predict, src=0)
                tp_group.broadcast(accept_index, src=0)
                tp_group.broadcast(num_correct_drafts, src=0)
    elif vocab_shard is not None:
        # temperature-only rows: a Gumbel-max target sample per row, then the greedy
        # check against it (exact for a topk-1 chain, whose drafts are deterministic)
        target_predict = sharded_predict.reshape(bs, verify_input.draft_token_num)
        if fused_verify:
            _chain_accept(
                predict=predict,
                accept_index=accept_index,
                num_correct_drafts=num_correct_drafts,
                candidates=candidates,
                verify_input=verify_input,
                target_predict=target_predict,
                bonus_target=None,
                simulated_acc_len=None,
                real_draft_tokens=False,
                finish=False,
            )
        else:
            predict, accept_index, num_correct_drafts = verify_tree_greedy_func(
                predicts=predict,
                accept_index=accept_index,
                accept_token_num=num_correct_drafts,
                candidates=candidates,
                retrieve_index=verify_input.retrieve_index,
                retrieve_next_token=verify_input.retrieve_next_token,
                retrieve_next_sibling=verify_input.retrieve_next_sibling,
                target_predict=target_predict,
            )
    elif sparse:
        from sglang.srt.speculative.uno_utils import (
            sample_uno_tree_target_tokens,
        )

        target_predict = sample_uno_tree_target_tokens(
            next_token_logits=next_token_logits,
            sampling_info=sampling_info,
            batch_size=bs,
            verify_width=verify_input.draft_token_num,
            max_top_k=uno_target_max_top_k,
        )
        if fused_verify:
            _chain_accept(
                predict=predict,
                accept_index=accept_index,
                num_correct_drafts=num_correct_drafts,
                candidates=candidates,
                verify_input=verify_input,
                target_predict=target_predict,
                bonus_target=None,
                simulated_acc_len=None,
                real_draft_tokens=False,
                finish=False,
            )
        else:
            predict, accept_index, num_correct_drafts = verify_tree_greedy_func(
                predicts=predict,
                accept_index=accept_index,
                accept_token_num=num_correct_drafts,
                candidates=candidates,
                retrieve_index=verify_input.retrieve_index,
                retrieve_next_token=verify_input.retrieve_next_token,
                retrieve_next_sibling=verify_input.retrieve_next_sibling,
                target_predict=target_predict,
            )

        tp_group = (
            get_parallel().attn_tp_group
            if is_dp_attention_enabled()
            else get_parallel().tp_group
        )
        if tp_group.world_size > 1:
            if verify_out is not None:
                tp_group.broadcast(verify_out, src=0)
            else:
                tp_group.broadcast(predict, src=0)
                tp_group.broadcast(accept_index, src=0)
                tp_group.broadcast(num_correct_drafts, src=0)
    else:
        if use_rejection_sampling:
            sampling_fn = chain_speculative_sampling_triton
            if get_spec().speculative_use_block_verification:
                sampling_fn = partial(sampling_fn, block_verification=True)
        else:
            sampling_fn = tree_speculative_sampling_target_only

        # Lean form (CUDA, chains, no rejection sampling): no zero-filled
        # draft_probs (the sampling kernel's target-only path).
        lean = (
            _is_cuda
            and not use_rejection_sampling
            and verify_input.tree_topk == 1
            and _spec_switches.enabled("SGLANG_SPEC_LEAN_SAMPLED_VERIFY")
        )
        if lean and _spec_switches.enabled("SGLANG_SPEC_CHUNKED_VERIFY_SAMPLING"):
            _log_chunked_verify_sampling_once()
            sampling_fn = partial(sampling_fn, chunked=True)
        target_probs = sampling_softmax(
            next_token_logits,
            temperatures=sampling_info.temperatures,
            rows_per_request=verify_input.draft_token_num,
        )  # (bs * num_draft_tokens, vocab_size)
        maybe_detect_nan(target_probs, "v2 verify: target_probs after softmax")
        target_probs = _apply_joint_top_k_top_p(
            target_probs,
            sampling_info,
            verify_input.draft_token_num,
            top_k_renorm_prob,
            top_p_renorm_prob,
        )
        if sampling_info.need_min_p_sampling:
            target_probs = min_p_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.min_ps, verify_input.draft_token_num, dim=0
                ),
            )
        target_probs = target_probs.reshape(bs, verify_input.draft_token_num, -1)
        if use_rejection_sampling:
            draft_probs = verify_input.draft_probs
        elif lean:
            draft_probs = None
        else:
            draft_probs = torch.zeros_like(target_probs)
        # Defense-in-depth behind the spec_hook startup allowlist: validate
        # the actual kernel inputs before the Triton kernel.
        if use_rejection_sampling and (
            draft_probs is None or draft_probs.shape[-1] != target_probs.shape[-1]
        ):
            raise ValueError(
                "Rejection sampling requires a target-vocab draft proposal "
                "distribution; the current speculative algorithm/draft worker "
                "does not produce one (draft_probs missing or vocab-mismatched)."
            )

        coins, coins_for_final_sampling = _verify_coins(
            sampling_info=sampling_info,
            seq_lens=batch.seq_lens,
            draft_token_num=verify_input.draft_token_num,
            candidates=candidates,
            device=device,
        )
        sampling_fn(
            predicts=predict,  # mutable
            accept_index=accept_index,  # mutable
            accept_token_num=num_correct_drafts,  # mutable
            candidates=candidates,
            # kwarg LHS retained as `retrive_*` to match sgl_kernel op schema.
            retrive_index=verify_input.retrieve_index,
            retrive_next_token=verify_input.retrieve_next_token,
            retrive_next_sibling=verify_input.retrieve_next_sibling,
            uniform_samples=coins,
            uniform_samples_for_final_sampling=coins_for_final_sampling,
            target_probs=target_probs,
            draft_probs=draft_probs,
            threshold_single=get_spec().speculative_accept_threshold_single,
            threshold_acc=get_spec().speculative_accept_threshold_acc,
            deterministic=True,
        )
        del (
            target_probs,
            draft_probs,
            coins,
            coins_for_final_sampling,
        )

        # Sync sampling results across TP ranks: different GPUs may
        # produce slightly different target_probs due to floating-point
        # non-determinism in softmax/top_k/top_p, causing different
        # sampled tokens. Broadcast from rank 0 to ensure consistency.
        tp_group = (
            get_parallel().attn_tp_group
            if is_dp_attention_enabled()
            else get_parallel().tp_group
        )
        if tp_group.world_size > 1:
            if verify_out is not None:
                tp_group.broadcast(verify_out, src=0)
            else:
                tp_group.broadcast(predict, src=0)
                tp_group.broadcast(accept_index, src=0)
                tp_group.broadcast(num_correct_drafts, src=0)

    if SIMULATE_ACC_LEN > 0 and accept_lens is None:
        # Do simulation. The helper builds (and returns) a replacement
        # accept_index of width spec_steps + 1, so pass max_tree_depth - 1
        # to keep the simulated width identical to the real one.
        if SIMULATE_ACC_TOKEN_MODE not in ("fixed", "real-draft-token"):
            raise ValueError(
                "Invalid SGLANG_SIMULATE_ACC_TOKEN_MODE "
                f"{SIMULATE_ACC_TOKEN_MODE!r}; expected 'fixed' or "
                "'real-draft-token'."
            )

        if SIMULATE_ACC_TOKEN_MODE == "real-draft-token":
            if verify_input.tree_topk != 1:
                raise ValueError(
                    "SGLANG_SIMULATE_ACC_LEN with real draft tokens currently "
                    "requires speculative_eagle_topk=1."
                )

            # Use target argmax as the synthetic bonus for non-greedy requests.
            if target_predict is None or sparse_bonus_argmax or vocab_shard is not None:
                bonus_predict = None
                if vocab_shard is not None:
                    bonus_predict = _sharded_tokens(
                        next_token_logits,
                        vocab_shard,
                        local_keys=local_keys,
                    )
                    if bonus_predict is None:
                        next_token_logits = gather_full_logits(
                            next_token_logits,
                            vocab_shard,
                            get_parallel().tp_group,
                        )
                        logits_output.next_token_logits = next_token_logits
                        logits_output.next_token_logits_vocab_shard = None
                        vocab_shard = None
                if bonus_predict is None:
                    bonus_predict = sampling_argmax(next_token_logits)
                target_predict = bonus_predict.reshape(bs, verify_input.draft_token_num)
        if chain:
            accept_lens, accept_index = _chain_accept(
                predict=predict,
                accept_index=accept_index,
                num_correct_drafts=num_correct_drafts,
                candidates=candidates,
                verify_input=verify_input,
                target_predict=None,
                bonus_target=target_predict,
                simulated_acc_len=simulated_acc_len,
                real_draft_tokens=SIMULATE_ACC_TOKEN_MODE == "real-draft-token",
                finish=True,
            )
        elif simulated_acc_len is not None:
            accept_index = generate_simulated_accept_index_device(
                accept_index=accept_index,
                predict=predict,  # mutable
                num_correct_drafts=num_correct_drafts,  # mutable
                candidates=candidates,
                target_predict=target_predict,
                bs=bs,
                spec_steps=verify_input.max_tree_depth - 1,
                simulate_acc_len=simulated_acc_len,
                simulate_acc_token_mode=SIMULATE_ACC_TOKEN_MODE,
            )
        else:
            accept_index = generate_simulated_accept_index(
                accept_index=accept_index,
                predict=predict,  # mutable
                num_correct_drafts=num_correct_drafts,  # mutable
                candidates=candidates,
                target_predict=target_predict,
                simulate_acc_len=SIMULATE_ACC_LEN,
                simulate_acc_token_mode=SIMULATE_ACC_TOKEN_MODE,
                bs=bs,
                spec_steps=verify_input.max_tree_depth - 1,
                generator=simulated_acc_len_rng,
            )

    # `num_correct_drafts` stays drafts-only inside this function; the returned
    # tensor includes the trailing/bonus token via out-of-place +1 so the
    # name no longer flips semantics mid-function (naming doc C2).
    if accept_lens is None:
        accept_lens = num_correct_drafts + 1
    return predict, accept_lens, accept_index


# ------------------------------------------------------------ eagle_worker_v2

_draft_tail_logged = set()


def _log_draft_tail_once(message: str) -> None:
    if message not in _draft_tail_logged:
        _draft_tail_logged.add(message)
        logger.info(message)


def _fused_draft_extend_tail_eligible(
    self, logits: torch.Tensor, seed_capture: Optional[torch.Tensor]
) -> bool:
    """SGLANG_SPEC_FUSED_DRAFT_EXTEND_TAIL applies to this draft extend.
    (Eager draft extends select their rows after this check; the selected
    copy keeps the dtype and is row-major.)"""
    if not _spec_switches.enabled("SGLANG_SPEC_FUSED_DRAFT_EXTEND_TAIL"):
        return False
    reason = None
    if self.topk != 1 or _is_hip or not logits.is_cuda:
        reason = "not a CUDA topk-1 draft"
    elif get_spec().speculative_use_rejection_sampling:
        reason = "rejection sampling"
    if reason is None and (
        logits.ndim != 2
        or logits.dtype != torch.float32
        or logits.stride(1) != 1
        or logits.shape[1] < 131072
    ):
        reason = "draft logits are not wide FP32 rows"
    if (
        reason is None
        and seed_capture is not None
        and (seed_capture.ndim != 2 or seed_capture.stride(1) != 1)
    ):
        reason = "seed buffer is not 2-D row-major"
    if reason is not None:
        _log_draft_tail_once(f"SGLANG_SPEC_FUSED_DRAFT_EXTEND_TAIL declined: {reason}")
        return False
    _log_draft_tail_once("SGLANG_SPEC_FUSED_DRAFT_EXTEND_TAIL on")
    return True


def _draft_extend_for_decode(
    self,
    batch: ScheduleBatch,
    batch_result: GenerationBatchResult,
    *,
    kv_loc_plan: Optional[KVLocPlan] = None,
):
    # Cast to int64 before entering plan stream to avoid cross-stream
    # synchronization issues with .to() inside the plan stream context.
    if batch_result.prepared_draft_extend_inputs is not None:
        num_correct_drafts, select_index, next_token_ids = (
            batch_result.prepared_draft_extend_inputs
        )
    elif batch_result.accept_lens.is_cuda:
        from sglang.kernels.ops.speculative.eagle import prepare_draft_extend_inputs

        num_correct_drafts, select_index, next_token_ids = prepare_draft_extend_inputs(
            batch_result.accept_lens,
            batch_result.next_token_ids,
            self.speculative_num_draft_tokens,
        )
    else:
        num_correct_drafts = batch_result.accept_lens - 1
        select_index = (
            torch.arange(
                0,
                len(batch.seq_lens) * self.speculative_num_draft_tokens,
                self.speculative_num_draft_tokens,
                device=self.device,
            )
            + batch_result.accept_lens
            - 1
        )

        next_token_ids = batch_result.next_token_ids.to(torch.int64)

    # Batch 2: Draft extend
    draft_extend_input = EagleDraftExtendInput(
        hidden_states=batch_result.logits_output.hidden_states,
        # accept_lens includes the bonus token; correct drafts exclude it.
        num_correct_drafts=num_correct_drafts,
        num_accept_tokens=batch_result.accept_lens,
        # Draft-extend fills the whole tree width (num_draft_tokens) per req,
        # not num_steps + 1, so DP MLP-sync padding stays consistent for topk > 1.
        num_tokens_per_req=self.speculative_num_draft_tokens,
        num_tokens_for_logprob_per_req=self.speculative_num_draft_tokens,
    )
    # Prepare for draft extend in a separate stream
    if self.plan_stream:
        self.plan_stream.wait_stream(
            torch.get_device_module(self.device).current_stream()
        )
    with self.plan_stream_ctx:
        forward_batch = prepare_for_draft_extend(
            draft_extend_input,
            batch,
            next_token_ids,
            self.speculative_num_draft_tokens,
            self.draft_runner,
            self.cuda_graph_runner_for_draft_extend,
            return_hidden_states_before_norm=False,
            kv_loc_plan=kv_loc_plan,
        )

    if self.plan_stream:
        torch.get_device_module(self.device).current_stream().wait_stream(
            self.plan_stream
        )

    # Run draft extend batch in the main compute stream
    can_run_decode_cuda_graph = (
        self.cuda_graph_runner_for_draft_extend
        and self.cuda_graph_runner_for_draft_extend.can_run_graph(forward_batch)
    )

    # Eager path publishes the indexer top-k into a worker buffer (the graph
    # path uses the runner's static buffer). Gathered at select_index below.
    if self.seed_dsa_topk_from_draft_extend and not can_run_decode_cuda_graph:
        forward_batch.spec_info.dsa_seed_topk_capture = self._get_dsa_extend_topk_buf(
            forward_batch.input_ids.shape[0]
        )

    canary_ctx = (
        context_tuple(
            c.with_ops_outside_graph(
                single_forward_indices=[0],
                maybe_inaccurate_forward_batch=forward_batch,
            ),
            c.with_active_single_forward_manager(0),
        )
        if (c := self.draft_runner.canary_manager) is not None
        else contextlib.nullcontext()
    )
    with canary_ctx:
        if can_run_decode_cuda_graph:
            draft_logits_output = self.cuda_graph_runner_for_draft_extend.execute(
                forward_batch, select_index
            )
        else:
            draft_logits_output = self.draft_runner.forward(forward_batch).logits_output

    maybe_detect_nan(
        draft_logits_output.next_token_logits,
        f"draft_extend_for_decode (cuda_graph={can_run_decode_cuda_graph})",
    )
    maybe_detect_inf(
        draft_logits_output.next_token_logits,
        f"draft_extend_for_decode (cuda_graph={can_run_decode_cuda_graph})",
    )

    # Gather the per-request last-position indexer top-k as the next loop's
    # seed (select_index already picks the last accepted position per req).
    dsa_seed_topk_indices = None
    dsa_extend_topk_capture = None
    if self.seed_dsa_topk_from_draft_extend:
        if can_run_decode_cuda_graph:
            dsa_extend_topk_capture = (
                self.cuda_graph_runner_for_draft_extend.buffers.dsa_seed_topk_capture
            )
        else:
            dsa_extend_topk_capture = forward_batch.spec_info.dsa_seed_topk_capture
    fused_tail = _fused_draft_extend_tail_eligible(
        self, draft_logits_output.next_token_logits, dsa_extend_topk_capture
    )
    if dsa_extend_topk_capture is not None and not fused_tail:
        # Fancy indexing returns a fresh tensor (detached from the buffer).
        dsa_seed_topk_indices = dsa_extend_topk_capture[select_index]

    # Reorganize the spec info for the next batch
    if not can_run_decode_cuda_graph:
        draft_logits_output.next_token_logits = draft_logits_output.next_token_logits[
            select_index
        ]
        if draft_logits_output.hidden_states is not None:
            draft_logits_output.hidden_states = draft_logits_output.hidden_states[
                select_index
            ]
    # Selected-row top-k remains worker-owned for both graph and eager
    # paths; the graph runner only moves the row selection before lm_head.
    if get_spec().speculative_use_rejection_sampling:
        ret_draft_probs, ret_topk_p, ret_topk_index = sample_draft_proposal(
            draft_logits_output.next_token_logits,
            batch.sampling_info.temperatures,
            batch.sampling_info.top_ks,
            sampling_seed=batch.sampling_info.sampling_seed,
            positions=batch_result.new_seq_lens,
        )
    elif self.topk == 1 and _is_hip:
        ret_topk_p, ret_topk_index = draft_topk1_argmax_only(
            draft_logits_output.next_token_logits
        )
        ret_draft_probs = None
    elif fused_tail:
        # SGLANG_SPEC_FUSED_DRAFT_EXTEND_TAIL: argmax, unit probs and the
        # seed-row gather below in two launches, same values.
        ret_topk_p, ret_topk_index, dsa_seed_topk_indices = draft_extend_tail(
            draft_logits_output.next_token_logits,
            dsa_extend_topk_capture,
            select_index if dsa_extend_topk_capture is not None else None,
        )
        ret_draft_probs = None
    elif self.topk == 1 and not _is_hip:
        # Gated to CUDA: see #26358 — ROCm's argmax tie-break corrupts
        # MTP draft selection on FP8 logits.
        ret_topk_index = torch.argmax(
            draft_logits_output.next_token_logits, dim=-1, keepdim=True
        )
        ret_topk_p = torch.ones_like(ret_topk_index, dtype=torch.float32)
        ret_draft_probs = None
    else:
        probs = renorm_draft_probs(
            draft_logits_output.next_token_logits,
            batch.sampling_info,
            get_spec().speculative_use_rejection_sampling,
        )
        ret_topk_p, ret_topk_index = fast_topk(probs, self.topk, dim=-1)
        ret_draft_probs = None
    ret_hidden_states = draft_logits_output.hidden_states

    # Construct the return values
    next_draft_input = batch_result.next_draft_input
    (
        next_draft_input.topk_p,
        next_draft_input.topk_index,
        next_draft_input.hidden_states,
    ) = (
        ret_topk_p,
        ret_topk_index,
        ret_hidden_states,
    )
    if get_spec().speculative_use_rejection_sampling:
        next_draft_input.draft_probs = ret_draft_probs
    if self.seed_dsa_topk_from_draft_extend:
        next_draft_input.dsa_topk_indices = dsa_seed_topk_indices
    next_draft_input.cuda_graph_compatible = not (
        self.seed_dsa_topk_from_draft_extend and dsa_seed_topk_indices is None
    )
