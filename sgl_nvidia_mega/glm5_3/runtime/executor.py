# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/deepseek_v2.py (DeepseekV2ForCausalLM.forward and DeepseekV2Model.forward: the layer loop of a verify step).
"""The executor: one covered MTP verify step of the GLM-5.3 target
(``GlmMoeDsaForCausalLM``).

``GlmMegakernelVerify.plan`` decides from host state alone, identical on every rank,
whether a step is covered: a target-verify forward of a built width (whole requests,
up to 16 rows) on the absorbed MLA path, recorded while SGLang's decode graph runner captures a
full graph. The first capture of each width checks, on every rank, that the stock
attention takes the branches the attention steps copy and the facts the kernels need;
a reason on any rank stops every rank, naming each rank's reasons (the gather is also
the barrier before that capture). Eager steps, the graph runner's warm-up runs
included, run SGLang's forward, so every rank has compiled and loaded every stock
kernel before the first replay.

``GlmMegakernelVerify.forward`` takes and returns what ``GlmMoeDsaForCausalLM.forward``
does. It owns the layer loop and the residual stream: per layer ``hops.pre_attention``
(the previous MLP output's all-reduce + residual add + input norm),
``attention.forward`` (up to the ``o_proj`` input), ``hops.post_attention`` (``o_proj``
+ all-reduce + residual add + norm) and ``mlp.forward``; then ``hops.final`` and the
logits. Each step runs on the step groups of the step's width (``plan.Providers``);
with every group stock the result equals SGLang's forward bit for bit.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch

try:
    from sglang.kernels.plugin_kernels.nvidia.spec import Step
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.megakernels.spec import Step

from sgl_nvidia_mega.glm5_3 import logits
from sgl_nvidia_mega.glm5_3.attention.step import (
    StepContext,
    absorbed_mla,
    attention_step_reasons,
)
from sgl_nvidia_mega.glm5_3.plan import Providers
from sgl_nvidia_mega.glm5_3.runtime import parity, per_rank, tp_gather

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    # Step, and the BuiltMegakernel members SGLang reads
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
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/binding.py": (
        "forward_view",
    ),
    "srt/runtime_context.py": ("get_parallel", '"tp_rank"'),
    "srt/layers/layer_boundary/__init__.py": ("get_attn_tp_context",),
    "srt/layers/layer_boundary/adapters/attention.py": ("maybe_input_scattered",),
    "srt/layers/layer_boundary/residual/add_norm.py": ("flashinfer_ar_fusion_applies",),
    "srt/model_executor/runner_backend_utils/breakable_cuda_graph/context.py": (
        "is_in_breakable_cuda_graph",
    ),
    "srt/model_executor/runner_utils/capture_mode.py": ("get_is_capture_mode",),
    "srt/model_executor/forward_batch_info.py": (
        "forward_mode",
        "is_target_verify",
        "can_run_tbo",
    ),
    "srt/eplb/expert_distribution.py": (
        "get_global_expert_distribution_recorder",
        "with_current_layer",
    ),
    "srt/layers/attention/dsa/utils.py": ("maybe_prefetch_next_full_attention_kv",),
    "srt/layers/attention/index_topk_share.py": (
        "IndexTopKShareState",
        "topk_indices",
        "update",
    ),
    "srt/utils/common.py": ("BumpAllocator",),
    # GlmMoeDsaForCausalLM / DeepseekV2Model, which the executor stands in for
    "srt/models/deepseek_v2.py": (
        "capture_aux_hidden_states",
        "model",
        "embed_tokens",
        "start_layer",
        "end_layer",
        "layers",
        "next_full_attention_layer_id",
    ),
}

# The SGLang function bodies this module's stock steps copy, by the digest of their
# code (test/unit/test_contract.py pins them)
SGLANG_MIRRORS = {
    "srt/models/deepseek_v2.py:DeepseekV2Model.forward": "25335aebe84fee87",
    "srt/models/deepseek_v2.py:DeepseekV2ForCausalLM.forward": "bbdd93599bb4ff3d",
}


class GlmMegakernelVerify:
    """One rank's covered verify steps: the model, each width's step groups and the
    built kernel families."""

    # the target offers no speculative entries or graph variants (BuiltMegakernel)
    draft_sampler = None
    draft_loop = None
    draft_extend = None
    begin_draft_round = None
    graph_variants = None

    def __init__(
        self,
        lm,
        providers: Dict[int, Providers],
        families: dict,
        *,
        widths,
        derived=None,
        errors=None,
        summary: str = "",
        binding=None,
    ):
        self.lm, self.providers, self.families = lm, providers, families
        self.widths = tuple(sorted(widths))
        # the derived weight copies the kernels read by address (kept alive here)
        self.derived = derived
        self.errors = errors
        self._summary = summary
        # SGLang's MegakernelBinding (BuildContext.binding)
        self.binding = binding
        # The widths whose first capture every rank has checked
        self._agreed: set = set()

    def summary(self) -> str:
        """What was built, for SGLang's startup line."""
        return self._summary

    def plan(
        self, input_ids, forward_batch, input_embeds=None, pp_proxy_tensors=None
    ) -> Optional[Step]:
        """The step if this forward is covered, else None (SGLang's forward runs)."""
        rows = int(input_ids.shape[0])
        if rows not in self.widths or not self._covered(
            rows, forward_batch, input_embeds, pp_proxy_tensors
        ):
            return None
        if torch.cuda.is_current_stream_capturing():
            if not self._in_full_graph_capture():
                return None
            self._first_capture_agreement(rows, forward_batch)
            return Step(mode="target_verify", rows=rows)
        if parity.eager_steps_allowed():
            return Step(mode="target_verify", rows=rows)
        return None

    def _covered(self, rows, forward_batch, input_embeds, pp_proxy_tensors) -> bool:
        """A verify forward of a covered width that the stock step copies take."""
        from sglang.srt.layers.layer_boundary.residual.add_norm import (
            flashinfer_ar_fusion_applies,
        )

        return (
            forward_batch.forward_mode.is_target_verify()
            and input_embeds is None
            and pp_proxy_tensors is None
            and not getattr(forward_batch, "can_run_tbo", False)
            and not getattr(self.lm, "capture_aux_hidden_states", False)
            and flashinfer_ar_fusion_applies(rows)
            and absorbed_mla(self.lm, forward_batch)
        )

    @staticmethod
    def _in_full_graph_capture() -> bool:
        """Inside the decode graph runner's full-graph capture procedure."""
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
            is_in_breakable_cuda_graph,
        )
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            get_is_capture_mode,
        )

        return get_is_capture_mode() and not is_in_breakable_cuda_graph()

    def _first_capture_agreement(self, rows: int, forward_batch) -> None:
        """At the first capture of ``rows`` every rank checks that the stock
        attention takes the branches the attention steps copy
        (``attention_step_reasons``) and each built family's host-only
        ``step_reasons``; the gather is also the barrier before that capture. A
        reason on any rank stops every rank, naming each rank's reasons."""
        from sglang.srt.runtime_context import get_parallel

        if rows in self._agreed:
            return
        view = self.binding.forward_view()
        reasons = list(attention_step_reasons(self.lm, forward_batch, view))
        for name, family in sorted(self.families.items()):
            extra = getattr(family, "step_reasons", None)
            if extra is not None:
                reasons += [f"{name}: {r}" for r in extra(self.lm, forward_batch, view)]
        gathered = tp_gather(reasons)
        if any(gathered):
            raise ValueError(
                "NVIDIA megakernel: GLM-5.3 / GLM-5.2 cannot run the verify step at "
                f"{rows} rows on this setup: {per_rank(gathered)}"
            )
        self._agreed.add(rows)
        logger.info(
            "NVIDIA megakernel: GLM-5.3 / GLM-5.2: capturing the verify step, %d rows "
            "(rank %d)",
            rows,
            get_parallel().tp_rank,
        )

    def forward(
        self,
        step: Step,
        original,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        pp_proxy_tensors=None,
    ):
        """The covered step (``original``: the model's own forward, bound, for the
        parity check)."""
        if not torch.cuda.is_current_stream_capturing():
            # plan() covers eager steps only under the test-only parity switch
            return parity.run(
                self, step, original, input_ids, positions, forward_batch, input_embeds
            )
        hidden = self._run(
            self.providers[step.rows], input_ids, positions, forward_batch, step.view
        )
        return logits.logits(self.lm, input_ids, hidden, forward_batch)

    def run_stock(self, step: Step, input_ids, positions, forward_batch):
        """The covered step on every step group's stock body (the parity check)."""
        hidden = self._run(Providers(), input_ids, positions, forward_batch, step.view)
        return logits.logits(self.lm, input_ids, hidden, forward_batch)

    @torch.no_grad()
    def _run(self, providers, input_ids, positions, forward_batch, view):
        """The final-normed hidden states of a verify forward."""
        from sglang.srt.layers.layer_boundary import get_attn_tp_context

        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            return self._stack(providers, input_ids, positions, forward_batch, view)

    def _stack(self, providers, input_ids, positions, forward_batch, view):
        """``DeepseekV2Model.forward`` on a covered step, on ``providers``."""
        from sglang.srt.eplb.expert_distribution import (
            get_global_expert_distribution_recorder,
        )
        from sglang.srt.layers.attention.dsa.utils import (
            maybe_prefetch_next_full_attention_kv,
        )
        from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState
        from sglang.srt.utils import BumpAllocator

        mm = self.lm.model
        step = StepContext(
            forward_batch=forward_batch,
            positions=positions,
            zero_allocator=BumpAllocator(
                buffer_size=(mm.end_layer - mm.start_layer) * 2,
                dtype=torch.float32,
                device=positions.device,
            ),
            view=view,
        )
        topk_share = IndexTopKShareState(forward_batch, None)
        recorder = get_global_expert_distribution_recorder()
        attention, hops, mlp = providers.attention, providers.hops, providers.mlp
        mlp_out, residual = mm.embed_tokens(input_ids), None
        for i in range(mm.start_layer, mm.end_layer):
            layer = mm.layers[i]
            with recorder.with_current_layer(i):
                # The consumed MLP output's storage may take this layer's MoE output.
                buffer = mlp_out if isinstance(mlp_out, torch.Tensor) else None
                x, residual = hops.pre_attention(layer, mlp_out, residual, step)
                o_input, topk_indices = attention.forward(
                    layer, x, topk_share.topk_indices, step
                )
                topk_share.update(topk_indices)
                maybe_prefetch_next_full_attention_kv(
                    forward_batch, mm.next_full_attention_layer_id.get(i)
                )
                x, residual = hops.post_attention(layer, o_input, residual, step)
                mlp_out = mlp.forward(layer, x, buffer, step)
        return hops.final(self.lm, mlp_out, residual, step)
