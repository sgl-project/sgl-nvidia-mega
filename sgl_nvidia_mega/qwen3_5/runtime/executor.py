# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/qwen3_vl.py (Qwen3VLForConditionalGeneration.forward) and
# python/sglang/srt/models/qwen3_5.py (Qwen3_5ForCausalLM.forward and its layers' forward).
"""The executor: one covered target-verify step of Qwen3.5.

The build gets SGLang's EAGLE binding (``BuildContext.binding``) and each covered step
the running forward's view (``Step.view``: the attention and GDN metadata, the KV and
mamba pools); the step's batch tensors are the forward's own arguments.
``Qwen35Verify.plan`` decides from host state alone, identical on every rank, whether a
step is covered: a target-verify step of a built width. ``Qwen35Verify.forward`` checks
the pools the slots read at the first covered step, then takes and returns what
``Qwen3_5MoeForConditionalGeneration.forward`` does. It runs the step from the
plugin's step functions over SGLang's loaded modules, never through a model-level or
decoder-layer ``forward``:

1. the positions the layers take (M-RoPE when the model uses it) and the embedding;
2. per layer, under the expert-distribution recorder's layer scope:
   ``x = boundary.pre(layer, h, st)``, ``part = gdn.step(layer, x, st)`` or
   ``attention.step(...)``, ``x = boundary.post_attn(layer, part, st)``, then inside
   ``boundary.ffn_exit(layer, st)``: ``out = moe.step(layer, x, st, defer)``, and
   ``h = exit.finish(out)``;
3. ``boundary.close_stream`` (the final norm) and the logits (``logits``).
"""

from __future__ import annotations

from functools import partial

import torch

try:
    from sglang.kernels.plugin_kernels.nvidia.spec import Step
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.megakernels.spec import Step

from sgl_nvidia_mega.qwen3_5 import attention, boundary, gdn, logits, moe
from sgl_nvidia_mega.qwen3_5.attention import route as _attention_route  # noqa: F401
from sgl_nvidia_mega.qwen3_5.gdn import verify as _gdn_verify  # noqa: F401
from sgl_nvidia_mega.qwen3_5.runtime import embedding
from sgl_nvidia_mega.qwen3_5.runtime.state import StepState

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
    # Qwen3VLForConditionalGeneration.forward, which the executor stands in for
    "srt/models/qwen3_vl.py": (
        "forward(get_embedding)",
        "forward(pp_proxy_tensors)",
        "is_mrope_enabled",
        "capture_aux_hidden_states",
        "model",
    ),
    "srt/model_executor/forward_batch_info.py": ("mrope_positions", "forward_mode"),
    "srt/models/qwen3_5.py": (
        "Qwen3_5LinearDecoderLayer",
        "layers",
        "start_layer",
        "end_layer",
        '"_is_layer_to_capture"',
    ),
    "srt/layers/aux_hidden_states.py": ("AuxHiddenStateList",),
    "srt/eplb/expert_distribution.py": (
        "get_global_expert_distribution_recorder",
        "with_current_layer",
    ),
    "srt/kernel_plugins/nvidia/eagle_binding.py|srt/megakernels/eagle_binding.py": (
        "tokens_per_request",
        "mamba",
    ),
}

# The SGLang functions whose body this module mirrors, by the digest of their code
# (test/unit/test_contract.py recomputes them on the installed SGLang)
SGLANG_MIRRORS = {
    "srt/models/qwen3_vl.py:Qwen3VLForConditionalGeneration.forward": "d394d79ba7fa7b31",
    "srt/models/qwen3_5.py:Qwen3_5ForCausalLM.forward": "9e530b264ce30ad6",
    "srt/models/qwen3_5.py:Qwen3_5LinearDecoderLayer.forward": "2298a57111f9430e",
    "srt/models/qwen3_5.py:Qwen3_5AttentionDecoderLayer.forward": "6473f673a1a0796b",
}


class Qwen35Verify:
    """One rank's covered steps: the model, each width's slots and each package's
    build result."""

    # the target offers no draft sampler (BuiltMegakernel)
    draft_sampler = None
    draft_loop = None
    draft_extend = None
    begin_draft_round = None

    def __init__(
        self,
        lm,
        slots: dict,
        bound: dict,
        widths: tuple[int, ...],
        binding=None,
        summary: str = "",
        parity=None,
    ):
        from sglang.srt.models.qwen3_5 import Qwen3_5LinearDecoderLayer

        self.lm, self.slots, self.bound = lm, slots, bound
        self.widths = tuple(sorted(widths))
        self.binding = binding  # SGLang's EagleBinding (BuildContext.binding)
        self._summary = summary
        self.parity = parity  # runtime.parity.Parity under the test-only switch
        # the attention graph variants per SGLang runner ("target_verify"), which it
        # captures once it takes a megakernel's (attention/route.py)
        self.graph_variants = (
            {"target_verify": attention.route.VERIFY}
            if any(s.skv2 for s in slots.values())
            else {}
        )
        self._pools_checked = False
        m = lm.model
        self.mixers = {
            i: gdn.step
            if isinstance(m.layers[i], Qwen3_5LinearDecoderLayer)
            else attention.step
            for i in range(m.start_layer, m.end_layer)
        }

    def summary(self) -> str:
        """What was built, for SGLang's startup line."""
        return self._summary

    def plan(
        self,
        input_ids,
        forward_batch,
        get_embedding=False,
        pp_proxy_tensors=None,
        **kwargs,
    ) -> Step | None:
        """The step if this forward is covered, else None (SGLang's forward runs).
        An argument this executor does not know also leaves the step to SGLang."""
        if self.parity is not None:
            self.parity.flush()
        if kwargs or get_embedding or pp_proxy_tensors is not None:
            return None
        if not forward_batch.forward_mode.is_target_verify():
            return None
        rows = int(input_ids.shape[0])
        return Step(mode="target_verify", rows=rows) if rows in self.widths else None

    def forward(
        self, step: Step, original, input_ids, positions, forward_batch, *args, **kwargs
    ):
        """The covered step (``original``: the model's own forward, bound, which the
        parity check runs)."""
        if not self._pools_checked:
            self._check_pools(step.view)
        run = partial(self._forward, step, input_ids, positions, forward_batch)
        if self.parity is not None:
            stock = partial(
                original, input_ids, positions, forward_batch, *args, **kwargs
            )
            return self.parity.run(step, stock, run)
        return run()

    def _check_pools(self, view):
        """Refuse a slot whose pools are not the ones it was built for; the pools
        exist from the first covered step on, the same on every rank."""
        if any(s.gdn_verify for s in self.slots.values()):
            reasons = gdn.verify.pool_reasons(
                view.mamba, self.binding.tokens_per_request
            )
            if reasons:
                raise RuntimeError(
                    "NVIDIA megakernel: slot gdn_verify cannot run: "
                    + "; ".join(reasons)
                )
        if any(s.gdn_front for s in self.slots.values()):
            from sgl_nvidia_mega.qwen3_5.gdn import front

            reasons = front.pool_reasons(view.mamba)
            if reasons:
                raise RuntimeError(
                    "NVIDIA megakernel: slot gdn_front cannot run: "
                    + "; ".join(reasons)
                )
        skv2 = any(s.skv2 for s in self.slots.values())
        mka = any(s.mka for s in self.slots.values())
        if skv2 or mka:
            reasons = attention.pool_reasons(view, self.binding, skv2=skv2, mka=mka)
            if reasons:
                raise RuntimeError(
                    "NVIDIA megakernel: the attention slots cannot run: "
                    + "; ".join(reasons)
                )
        self._pools_checked = True

    @torch.no_grad()
    def _forward(self, step, input_ids, positions, forward_batch):
        """``Qwen3VLForConditionalGeneration.forward`` for a covered step."""
        lm = self.lm
        if lm.is_mrope_enabled:
            positions = forward_batch.mrope_positions
        st = StepState(
            lm=lm,
            forward_batch=forward_batch,
            positions=positions,
            input_ids=input_ids,
            slots=self.slots[step.rows],
            bound=self.bound,
            binding=self.binding,
            view=step.view,
        )
        hidden, aux = self._language_model(st, embedding.embed(st))
        return logits.logits(st, hidden, aux if lm.capture_aux_hidden_states else None)

    def _language_model(self, st, hidden):
        """``Qwen3_5ForCausalLM.forward``: the layers and the final norm; returns the
        hidden rows and the captured auxiliary rows (None without a capture layer)."""
        from sglang.srt.eplb.expert_distribution import (
            get_global_expert_distribution_recorder,
        )
        from sglang.srt.layers.aux_hidden_states import AuxHiddenStateList

        m = st.model
        boundary.open_stream(st)
        aux = AuxHiddenStateList()
        for i in range(m.start_layer, m.end_layer):
            layer = m.layers[i]
            capture = aux if getattr(layer, "_is_layer_to_capture", False) else None
            with get_global_expert_distribution_recorder().with_current_layer(i):
                hidden = self._layer(i, layer, hidden, st, capture)
        hidden = boundary.close_stream(st, hidden)
        return hidden, (aux if len(aux) else None)

    def _layer(self, i, layer, hidden, st, capture):
        """A decoder layer's forward: the mixer stage, then the MoE stage."""
        x = boundary.pre(layer, hidden, st, capture)
        part = self.mixers[i](layer, x, st)
        x = boundary.post_attn(layer, part, st)
        with boundary.ffn_exit(layer, st) as ffn_exit:
            out = moe.step(layer, x, st, defer=ffn_exit.defer_moe_finalize)
        return ffn_exit.finish(out)
