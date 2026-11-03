# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/speculative/eagle_worker_v2.py (EagleDraftWorker.draft_forward, the
# topk-1 chain with draft_topk1_postprocess).
"""Entry ``draft_loop``: a served draft graph's whole draft loop for one or two requests.

SGLang calls the factory once before it captures its draft graphs; the step it returns
for bs 1 and 2 runs inside the capture in place of ``draft_forward``. Draft step i (of
``num_steps - 1``) runs the model step at ``positions + i`` on the KV slot
``out_cache_loc[i]`` and writes its top-1 into ``draft_tokens_out[:, i + 1]``
(column 0 is the round's first token); the step advances the positions in place and
restores them before it returns, as the binding asks.

Stock: SGLang's model step and top-1 (``layer``). Slot ``draft_xchg`` ends each
one-request step in the exchange: the next step's fc comes from it, the LM head is the
shard GEMV, and the token is chosen across the shards in the same launch
(``draft/xchg``); a two-request step takes its fc from the exchange's row-split fc (2.2's
DRAFT_FC_SPLIT at two rows) and SGLang's top-1.
``begin_draft_round`` then fills the first step's fc when the round's input did not
come from a served draft extend (2.2's take_fc_buf / fill_fc decision).
"""

from __future__ import annotations

import torch

from sgl_nvidia_mega.qwen3_5 import plan
from sgl_nvidia_mega.qwen3_5.draft import LOOP_BS, SERVED_BS, layer

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/kernel_plugins/nvidia/eagle_binding.py|srt/megakernels/eagle_binding.py": (
        "EagleDraftLoopIO",
        "num_requests",
        "num_steps",
        "topk_index",
        "hidden_states",
        "positions",
        "mrope_positions",
        "out_cache_loc",
        "steps",
        "kv",
        "draft_tokens_out",
        "EagleDraftRoundIO",
        "from_served_extend",
        "tp_size",
    ),
}


class DraftLoop:
    """The draft_loop factory's result: bs -> the step."""

    def __init__(self, parts, *, num_steps: int):
        self.parts = parts
        self.num_steps = num_steps
        if "draft_q1" in parts.slots and parts.attention.q1 is None:
            from sgl_nvidia_mega.qwen3_5.draft.attention import build_q1

            parts.attention.q1 = build_q1(parts)

    def step_for(self, bs: int):
        # two requests where the verify steps of several requests are covered
        from sgl_nvidia_mega.qwen3_5.runtime import SLOTS

        if not plan.multi_request(self.parts.binding.tp_size, SLOTS):
            return self.step if bs == SERVED_BS else None
        return self.step if bs in LOOP_BS else None

    def begin_round(self, io) -> None:
        """On the host before a served draft graph replays: the first step's fc when
        no served draft extend of these requests produced the round's input."""
        mkd = self.parts.mkd
        if (
            mkd is not None
            and mkd.extend_fc
            and io.num_requests == SERVED_BS
            and not io.from_served_extend
        ):
            mkd.fill_fc(io.topk_index, io.hidden_states)

    @torch.no_grad()
    def step(self, io) -> None:
        parts = self.parts
        parts.note_engaged("draft_loop", io.num_requests)
        steps = io.num_steps - 1
        tokens = io.draft_tokens_out
        tokens[:, :1].copy_(io.topk_index)
        positions, mrope = io.positions, io.mrope_positions
        if parts.mkd is not None and io.num_requests == SERVED_BS:
            self._mkd_steps(io, steps)
        else:
            token, hidden = io.topk_index.view(-1), io.hidden_states
            for i in range(steps):
                hidden = self._model_step(io, i, self._fc(token, hidden))
                token = layer.top1(parts, hidden)
                tokens[:, i + 1].copy_(token)
                positions.add_(1)
                if mrope is not None:
                    mrope.add_(1)
        positions.sub_(steps)
        if mrope is not None:
            mrope.sub_(steps)

    def _fc(self, token, hidden):
        """A step's fc output outside the exchange: slot draft_xchg's row-split fc, else
        SGLang's."""
        mkd = self.parts.mkd
        if mkd is None:
            return layer.fc_input(self.parts, token, hidden)
        return mkd.split_fc(self.parts.model.embed_tokens(token), hidden)

    def _model_step(self, io, i: int, x):
        """Draft step i's decoder layer on its fc output ``x``: the final-norm rows."""
        # request-major slots: step i's of several requests are a strided view, which the
        # front and the FP8 KV write read as dense rows
        cache_loc = io.out_cache_loc[i :: io.num_steps]
        if io.num_requests > 1:
            cache_loc = cache_loc.contiguous()
        return layer.layer(
            self.parts,
            x,
            io.positions,
            cache_loc,
            io.steps[i],
            io.kv,
            rows_per_request=1,
        )

    def _mkd_steps(self, io, steps: int) -> None:
        """The steps with slot draft_xchg, in 2.2's order: fc from the exchange, the
        layer, the hidden half of the next fc on a side stream, the shard head, the
        exchange (token, positions and the next fc)."""
        mkd = self.parts.mkd
        for i in range(steps):
            mkd.begin_step(i, io.num_requests)
            hidden = self._model_step(io, i, mkd.fc())
            if mkd.armed:
                mkd.hidden_partial(hidden)
            mkd.finish_step(
                mkd.head(hidden),
                io.positions,
                io.draft_tokens_out,
                i,
                i + 1,
                io.mrope_positions,
            )
        mkd.end()
