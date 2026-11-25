# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/minimax_m3_vl.py (MiniMaxM3SparseForConditionalGeneration.forward)
# and python/sglang/srt/models/minimax_m3.py (MiniMaxM3Model.forward and its layer loop,
# MiniMaxM3DecoderLayer.forward).
"""The executor: one covered verify step of the M3.1 target.

``MegakernelDecode.plan`` decides from host state alone, alike on every rank, whether
a forward is covered: a DSpark verify step of a built width (8 rows a request, 1-16
requests), while SGLang captures its decode CUDA graphs (or eagerly under the parity
switch). ``MegakernelDecode.forward`` takes and returns what
``MiniMaxM3SparseForConditionalGeneration.forward`` does, by the width's slots:

- slot mega (one request's 8 rows): ``runtime.mega.MegaStep.run``, MegaAttention and
  the token-parallel MoE on every MoE layer (not bitwise; ``spec.NUMERICS``);
- otherwise the stock executor, from the step functions over SGLang's loaded modules,
  never through the model's or a decoder layer's ``forward``:

  1. ``boundary.start``: the embeddings and the residual stream;
  2. per layer, under the expert-distribution recorder:
     ``x = boundary.attn_input(layer, x, st, captured)``,
     ``x = attention.step(layer, x, st)``, ``x = boundary.ffn_input(layer, x, st)``,
     then inside ``layer.ffn_boundary.exit`` ``x = moe.step(layer, x, st)`` (slots
     moe_nofill and moe_swap) and ``x = ffn_exit.finish(x)``;
  3. ``boundary.final`` (the final norm and the last DSpark capture),
     ``logits.logits``.

  With every slot off this is SGLang's forward call for call, so bit for bit;
  moe_nofill and moe_swap keep it so (their kernels and arguments are SGLang's).
  ``SGLANG_MIRRORS`` pins the SGLang code the executor follows.

``SGLANG_NVIDIA_MEGAKERNEL_PARITY`` (test only, eager covered steps) runs SGLang's own
forward on the same step first and logs the difference (``runtime.parity``);
``control`` runs SGLang's forward twice (the run-to-run floor).
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, Callable, Optional

import torch
from sglang.kernels.plugin_kernels.nvidia.spec import Step

from sgl_nvidia_mega.minimax_m31 import ROWS, attention, boundary, logits, moe
from sgl_nvidia_mega.minimax_m31.runtime import parity
from sgl_nvidia_mega.minimax_m31.runtime.state import StepState
from sgl_nvidia_mega.minimax_m31.spec import serves_this_step

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    # the BuiltMegakernel members SGLang reads and the Step it hands forward
    "kernels/plugin_kernels/nvidia/spec.py": (
        "Step",
        "mode",
        "rows",
        "widths",
        "draft_sampler",
        "draft_loop",
        "draft_extend",
        "begin_draft_round",
        "graph_variants",
        "speculative_runtime_hooks",
        "summary",
        "plan",
        "forward",
    ),
    "srt/model_executor/forward_mode.py": ("is_target_verify",),
    "srt/model_executor/forward_batch_info.py": (
        "forward_mode",
        "batch_size",
        "can_run_tbo",
    ),
    "srt/eplb/expert_distribution.py": (
        "get_global_expert_distribution_recorder",
        "with_current_layer",
    ),
    "srt/layers/aux_hidden_states.py": ("AuxHiddenStateList",),
    "srt/model_executor/cuda_graph_config.py": (
        "Backend",
        "Phase",
        "check_cuda_graph_backend",
        "PREFILL",
        "TC_PIECEWISE",
    ),
    "srt/model_executor/runner_backend_utils/breakable_cuda_graph/*.py": (
        "is_in_breakable_cuda_graph",
    ),
    "srt/models/minimax_m3.py": (
        "start_layer",
        "end_layer",
        "layers",
        "ffn_boundary",
        "is_layer_sparse",
    ),
    # MiniMaxM3SparseForConditionalGeneration.forward, which the executor stands in for
    # (plan() mirrors its signature); the wrapper attributes the step and
    # logits/__init__.py read; and the capture flag the wrapper sets
    "srt/models/minimax_m3_vl.py": (
        "forward(input_ids)",
        "forward(positions)",
        "forward(forward_batch)",
        "forward(get_embedding)",
        "forward(pp_proxy_tensors)",
        "model",
        "_is_layer_to_capture",
    ),
    "srt/layers/layer_boundary/*.py": ("exit", "finish"),
    "srt/layers/logits_processor.py": ("next_token_logits", "hidden_states"),
}

# The SGLang functions whose body the stock executor mirrors, by the digest of their
# code (test/unit/test_contract.py recomputes them on the installed SGLang)
SGLANG_MIRRORS = {
    "srt/models/minimax_m3_vl.py:MiniMaxM3SparseForConditionalGeneration.forward": (
        "5153f223c5b8d000"
    ),
    "srt/models/minimax_m3.py:MiniMaxM3Model.forward": "ff1fd0740ed78a69",
    "srt/models/minimax_m3.py:MiniMaxM3DecoderLayer.forward": "2207705a8a666de4",
    "srt/models/minimax_m3.py:_input_rows": "1f2f8be2c5e9ec1c",
}

logger = logging.getLogger(__name__)


class MegakernelDecode:
    """One rank's covered steps: the model, each width's slots, each step package's
    build result and slot mega's step (None when it is off)."""

    # the target offers no draft sampler, EAGLE entries, graph variants or
    # speculative runtime hooks (BuiltMegakernel)
    draft_sampler = None
    draft_loop = None
    draft_extend = None
    begin_draft_round = None
    graph_variants = None
    speculative_runtime_hooks = None

    def __init__(
        self,
        lm,
        *,
        widths: tuple[int, ...],
        plan: dict,
        bound: dict,
        mega: Any = None,
        summary: str = "",
        draft_check: Optional[Callable[[], bool]] = None,
    ):
        self.lm = lm
        self.widths = tuple(sorted(widths))
        self.slot_plan = plan
        self.bound = bound
        self.mega = mega  # runtime.mega.MegaStep, with slot mega built
        self._summary = summary
        # with draft slots requested, their engagement (draft.EngagementCheck): called
        # on every forward until it returns True
        self._draft_check = draft_check

    def summary(self) -> str:
        """What was built, for SGLang's startup line."""
        return self._summary

    def plan(
        self,
        input_ids,
        forward_batch,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> Optional[Step]:
        """The step if this forward is covered, else None (SGLang's forward runs)."""
        if self._draft_check is not None and self._draft_check():
            self._draft_check = None
        if get_embedding or pp_proxy_tensors is not None:
            return None
        if not forward_batch.forward_mode.is_target_verify():
            return None
        # two-batch overlap splits the step across SGLang's stages
        if getattr(forward_batch, "can_run_tbo", False):
            return None
        rows = int(input_ids.shape[0])
        if rows not in self.widths or rows != ROWS * int(forward_batch.batch_size):
            return None
        return Step(mode="target_verify", rows=rows) if serves_this_step() else None

    def forward(
        self,
        step: Step,
        original,
        input_ids,
        positions,
        forward_batch,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ):
        """The covered step (``original``: the model's own forward, bound, for the
        parity check)."""
        mode = parity.mode()
        if mode != "0" and not torch.cuda.is_current_stream_capturing():
            return self._with_parity(
                mode, step, original, input_ids, positions, forward_batch
            )
        return self._forward(step, input_ids, positions, forward_batch)

    def _forward(self, step: Step, input_ids, positions, forward_batch):
        """``MiniMaxM3SparseForConditionalGeneration.forward`` for a covered step."""
        slots = self.slot_plan[step.rows]
        if slots.mega:
            return self.mega.run(step, input_ids, positions, forward_batch)
        return self._stock(step, slots, input_ids, positions, forward_batch)

    def _stock(self, step: Step, slots, input_ids, positions, forward_batch):
        """The stock executor with the width's bitwise slots."""
        from sglang.srt.eplb.expert_distribution import (
            get_global_expert_distribution_recorder,
        )
        from sglang.srt.layers.aux_hidden_states import AuxHiddenStateList
        from sglang.srt.model_executor.cuda_graph_config import (
            Backend,
            Phase,
            check_cuda_graph_backend,
        )
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
            is_in_breakable_cuda_graph,
        )

        st = StepState(
            lm=self.lm,
            forward_batch=forward_batch,
            positions=positions,
            input_ids=input_ids,
            slots=slots,
            bound=self.bound,
        )
        m = st.model
        x = boundary.start(st)
        aux = AuxHiddenStateList()
        for i in range(m.start_layer, m.end_layer):
            # torch dynamo does not support a graph break in a context manager
            ctx = (
                nullcontext()
                if (
                    check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE)
                    or is_in_breakable_cuda_graph()
                )
                else get_global_expert_distribution_recorder().with_current_layer(i)
            )
            with ctx:
                layer = m.layers[i]
                captured = (
                    aux if getattr(layer, "_is_layer_to_capture", False) else None
                )
                x = self._layer(st, layer, x, captured)
        x = boundary.final(st, x, aux)
        return logits.logits(st, x, aux)

    @staticmethod
    def _layer(st: StepState, layer, x, captured):
        """``MiniMaxM3DecoderLayer.forward`` for a covered step."""
        x = boundary.attn_input(layer, x, st, captured)
        if _input_rows(x) != 0:
            x = attention.step(layer, x, st)
        x = boundary.ffn_input(layer, x, st)
        with layer.ffn_boundary.exit(st.fb) as ffn_exit:
            if layer.is_layer_sparse or _input_rows(x) != 0:
                x = moe.step(layer, x, st)
        return ffn_exit.finish(x)

    def _with_parity(self, mode, step, original, input_ids, positions, forward_batch):
        """SGLang's forward, then the executor's (``control``: SGLang's again), on the
        same step; logs how far apart they are (``runtime.parity``)."""
        want = original(input_ids, positions, forward_batch)
        if mode == "control":
            want = (
                parity.clone(want.next_token_logits),
                parity.clone(want.hidden_states),
            )
            got = original(input_ids, positions, forward_batch)
            parity.log_bitwise(step, want, got, control=True)
            return got
        if self.slot_plan[step.rows].mega:
            want = want.next_token_logits.clone()
            got = self._forward(step, input_ids, positions, forward_batch)
            parity.log_parity("target logits", want, got.next_token_logits)
            return got
        want = (parity.clone(want.next_token_logits), parity.clone(want.hidden_states))
        got = self._forward(step, input_ids, positions, forward_batch)
        parity.log_bitwise(step, want, got, control=False)
        return got


def _input_rows(hidden_states) -> int:
    """``minimax_m3._input_rows``: a sublayer input's rows (a norm that quantized for a
    projection returns a tuple, whose first tensor holds them)."""
    if isinstance(hidden_states, tuple):
        return hidden_states[0].shape[0]
    return hidden_states.shape[0]
