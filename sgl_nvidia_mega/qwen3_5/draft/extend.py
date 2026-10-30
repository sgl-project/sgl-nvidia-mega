# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py (the captured
# draft extend with the row selection before the LM head) and
# python/sglang/srt/speculative/eagle_worker_v2.py (EagleDraftWorker
# ._draft_extend_for_decode: the selected rows' top-1).
"""Entry ``draft_extend``: a served draft-extend graph's model step and top-1 for one
request.

SGLang calls the factory once before it captures its draft-extend graphs; the step it
returns for bs 1 runs inside the capture in place of the model's forward. It runs the
model step on the verify's rows (writing their draft KV), and writes the selected
row's top-1 (``torch.argmax``'s choice), 1.0 and final-norm hidden state into the
SGLang-owned outputs.

Stock: SGLang's model step and top-1 (``layer``). Slot ``draft_xchg``: the fc split
of 2.2 and the exchange's top-1, which also writes the next draft round's first fc
(2.2's MKD extend fc).
"""

from __future__ import annotations

import torch

from sgl_nvidia_mega.qwen3_5 import plan
from sgl_nvidia_mega.qwen3_5.draft import MULTI_LOOP_BS, SERVED_BS, layer

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/kernel_plugins/nvidia/eagle_binding.py|srt/megakernels/eagle_binding.py": (
        "EagleDraftExtendIO",
        "num_requests",
        "tokens_per_request",
        "input_ids",
        "positions",
        "hidden_states",
        "out_cache_loc",
        "select_index",
        "attn",
        "kv",
        "topk_index_out",
        "topk_p_out",
        "hidden_out",
    ),
}


class DraftExtend:
    """The draft_extend factory's result: bs -> the step."""

    def __init__(self, parts, *, tokens_per_request: int):
        self.parts = parts
        self.width = tokens_per_request

    def step_for(self, bs: int):
        # more requests with slot draft_multi where the verify steps of several are covered
        from sgl_nvidia_mega.qwen3_5.runtime import SLOTS

        if (
            bs in MULTI_LOOP_BS
            and self.parts.top1x is not None
            and plan.multi_request(self.parts.binding.tp_size, SLOTS)
        ):
            return self.step
        return self.step if bs == SERVED_BS else None

    @torch.no_grad()
    def step(self, io) -> None:
        parts = self.parts
        parts.note_engaged("draft_extend", io.num_requests)
        if io.num_requests > SERVED_BS:
            self._multi_step(io)
            return
        mkd = parts.mkd
        if mkd is not None:
            embeds = parts.model.embed_tokens(io.input_ids)
            x = mkd.split_fc(embeds, io.hidden_states)
        else:
            x = layer.fc_input(parts, io.input_ids, io.hidden_states)
        hidden = layer.layer(
            parts,
            x,
            io.positions,
            io.out_cache_loc,
            io.attn,
            io.kv,
            rows_per_request=io.tokens_per_request,
        )
        # the selected rows and the exchange's token go straight into SGLang's outputs: no
        # copies at the end of the graph
        selected = torch.index_select(hidden, 0, io.select_index, out=io.hidden_out)
        if mkd is not None:
            local = mkd.extend_head(selected)
            if local is not None:
                mkd.exchange_extend(
                    local, selected, io.topk_index_out, io.topk_p_out, mkd_head=True
                )
                return
            lm = parts.lm
            local = lm.logits_processor._compute_lm_head(selected, lm.lm_head, None)
            mkd.exchange_extend(local, selected, io.topk_index_out, io.topk_p_out)
        else:
            io.topk_index_out.copy_(layer.top1(parts, selected).view(-1, 1))
            io.topk_p_out.fill_(1.0)

    def _multi_step(self, io) -> None:
        """More requests (slot draft_multi): SGLang's model step on every verify row,
        then the selected rows' sharded top-1 (torch.argmax's choice, a NaN first)
        straight into SGLang's outputs."""
        from sgl_nvidia_mega.qwen3_5.sm103.draft.top1 import NAN_KEY_TORCH_ARGMAX

        parts = self.parts
        x = layer.fc_input(parts, io.input_ids, io.hidden_states)
        hidden = layer.layer(
            parts,
            x,
            io.positions,
            io.out_cache_loc,
            io.attn,
            io.kv,
            rows_per_request=io.tokens_per_request,
        )
        selected = torch.index_select(hidden, 0, io.select_index, out=io.hidden_out)
        layer.top1_into(
            parts,
            selected,
            io.topk_index_out,
            0,
            nan_key=NAN_KEY_TORCH_ARGMAX,
            topk_p=io.topk_p_out,
        )
