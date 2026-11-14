# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py (the fold slots made
# at each capture), python/sglang/srt/speculative/eagle_worker_v2.py (draft_forward's
# finalize of each draft step), python/sglang/srt/models/qwen4_exp_mtp.py
# (_prepare_input_embeds taking the folded embedding) and
# python/sglang/srt/layers/logits_processor.py (LogitsProcessor._get_logits before
# its FP32 copy).
"""Slot ``draft_fold``: an MTP draft step's top-1 finalize with the M-RoPE + 1 and the
next step's input folded in (``kernels/ops/speculative/draft_step_fold.py``), through
SGLang's ``draft_step_fold`` entry:

- before each draft graph of a topk-1 chain is captured, SGLang calls the draft
  executor's ``draft_step_fold(batch_size=, num_steps=)`` (``make``; None: SGLang's
  own top-1);
- in a greedy chain without logprobs, the round's draft forwards see the fold
  (``ForwardBatch.draft_step_fold``): they return the LM head's BF16 logits
  (``bf16_logits``) and the second takes the folded embedding;
- SGLang hands each step's logits to ``StepFold.finalize`` in place of its
  ``draft_topk1_postprocess`` and the M-RoPE + 1.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

SGLANG_READS = {
    "srt/layers/logits_processor.py": (
        "_compute_lm_head",
        "logit_scale",
        "vocab_size",
        "final_logit_softcapping",
        "do_tensor_parallel_all_gather",
        "do_tensor_parallel_all_gather_dp_attn",
        "use_fp32_lm_head",
        "use_fp32_lm_head_output",
    ),
    "srt/models/qwen4_exp_mtp.py": ("logits_processor", "lm_head"),
}


def build() -> str:
    """'' when the fold's kernels import here, else why not."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return "the draft fold needs an SM10x GPU"
    try:
        from sgl_nvidia_mega.qwen3_8.kernels.ops.speculative import (  # noqa: F401
            draft_step_fold,
        )
    except Exception as e:
        return f"the draft fold kernel does not import: {type(e).__name__}: {e}"
    return ""


def bf16_logits_reason(lm) -> str:
    """'' when the draft's logits before the logits processor's FP32 copy are its LM
    head's output alone (no softcap, gather or FP32 head), else why not."""
    lp = lm.logits_processor
    if lp.final_logit_softcapping:
        return "a logit softcap (fused into the FP32 copy)"
    if lp.do_tensor_parallel_all_gather or lp.do_tensor_parallel_all_gather_dp_attn:
        return "a TP / DP logits gather"
    if lp.use_fp32_lm_head or lp.use_fp32_lm_head_output:
        return "an FP32 LM head"
    if lm.lm_head.weight.dtype != torch.bfloat16:
        return f"a {lm.lm_head.weight.dtype} LM head"
    return ""


def bf16_logits(lm, hidden):
    """``LogitsProcessor._get_logits`` of a draft decode up to its FP32 copy: the LM
    head (with its logit scale), cut to the vocabulary; ``bf16_logits_reason`` ''."""
    lp = lm.logits_processor
    logits = lp._compute_lm_head(hidden, lm.lm_head, None)
    if lp.logit_scale is not None:
        logits.mul_(lp.logit_scale)
    return logits[:, : lp.vocab_size]


class StepFold:
    """One captured draft graph's fold: its slot, the chain's length, and the
    executor whose draft forwards take the folded embedding and return BF16 logits
    (``bf16``) while it is the round's."""

    def __init__(self, executor, slot, num_steps, bf16):
        self.executor = executor
        self.slot = slot
        self.num_steps = num_steps
        self.bf16 = bf16

    def finalize(self, logits, positions, draft_tokens, column, mrope_positions=None):
        """``draft_topk1_postprocess(logits, positions, draft_tokens, column)`` and the
        M-RoPE positions' + 1 (when given); before another draft step, with the
        token's embedding row gathered for it. Returns (topk_p, topk_index)."""
        from sgl_nvidia_mega.qwen3_8.kernels.ops.speculative import draft_step_fold

        if column < self.num_steps - 1:
            out = draft_step_fold.finalize_next_input(
                logits, positions, draft_tokens, column, self.slot, mrope_positions
            )
            self.executor.step_embedding = self.slot.embedding
            return out
        self.executor.step_embedding = None
        return draft_step_fold.draft_topk1_postprocess(
            logits, positions, draft_tokens, column, mrope_positions
        )


def make(executor, batch_size: int, num_steps: int):
    """The entry ``draft_step_fold``: a StepFold for a draft graph of ``batch_size``
    requests, or None where the source's gate refuses it."""
    from sgl_nvidia_mega.qwen3_8.kernels.ops.speculative import draft_step_fold

    reason = draft_step_fold.fold_reason(executor.lm, batch_size, num_steps)
    if reason:
        logger.info("Qwen3.8 slot draft_fold: bs %d stock (%s)", batch_size, reason)
        return None
    logits = bf16_logits_reason(executor.lm)
    if logits:
        logger.info("Qwen3.8 slot draft_fold: FP32 logits (%s)", logits)
    weight = executor.lm.model.embed_tokens.weight
    slot = draft_step_fold.make_slot(weight, batch_size)
    return StepFold(executor, slot, num_steps, bf16=not logits)
