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
from sgl_nvidia_mega.qwen3_5.runtime import embedding, slot_failure
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
    "srt/model_executor/forward_mode.py|srt/model_executor/forward_batch_info.py": (
        "is_idle",
    ),
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
        "conv",
        "intermediate_conv_window",
        "ring_d",
        "ring_write_pos",
        "ring_cache_base",
        "ring_is_flush",
    ),
    # SupportsRecurrentCommit / SupportsStateSlotReset of the fold build
    "kernels/plugin_kernels/nvidia/spec.py": (
        "commit_recurrent_state",
        "track_slots",
        "request_rows",
        "accept_lens",
        "last_step",
        "state_slots",
        "src_indices_raw",
        "state_bytes_per_slot",
        "state_bytes_per_request_row",
        "keeps_draft_ssm",
        "tracks_decode_state",
        "backs_up_state",
        "reset_state_slots",
    ),
    "kernels/ops/attention/linear/gdn_replayssm_spec_decode.py": (
        "commit_gdn_replayssm_spec",
    ),
    "kernels/ops/mamba/mamba_state_scatter_triton.py": (
        "fused_conv_window_scatter_with_mask",
    ),
}

# The SGLang functions whose body this module mirrors, by the digest of their code
# (test/unit/test_contract.py recomputes them on the installed SGLang)
SGLANG_MIRRORS = {
    "srt/models/qwen3_vl.py:Qwen3VLForConditionalGeneration.forward": (
        "d394d79ba7fa7b31",
        "dbdcf4b88a2c0d40",
    ),
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

    def plan_body(self, input_ids, forward_batch, *args, **kwargs):
        return None

    def forward_body(self, step, original, *args, **kwargs):
        return original(*args, **kwargs)

    def linear_prefill_graph_seqs(self, backend):
        return None

    def refill_prefill_graph(self, metadata, forward_batch):
        return None

    def __init__(
        self,
        lm,
        slots: dict,
        bound: dict,
        widths: tuple[int, ...],
        binding=None,
        summary: str = "",
        parity=None,
        speculative_runtime_hooks=None,
        recurrent_commit=None,
    ):
        from sglang.srt.models.qwen3_5 import Qwen3_5LinearDecoderLayer

        self.lm, self.slots, self.bound = lm, slots, bound
        self.widths = tuple(sorted(widths))
        self.binding = binding  # SGLang's EagleBinding (BuildContext.binding)
        self._summary = summary
        self.parity = parity  # runtime.parity.Parity under the test-only switch
        # A factory over SGLang's stock strategy, or None.  The binding reads this
        # optional BuiltMegakernel member once when it wires speculative verify.
        self.speculative_runtime_hooks = speculative_runtime_hooks
        self.recurrent_commit = recurrent_commit
        if recurrent_commit is not None:
            self.commit_recurrent_state = recurrent_commit
        # the attention graph variants per SGLang runner ("target_verify"), which it
        # captures once it takes a megakernel's (attention/route.py)
        self.graph_variants = (
            {"target_verify": attention.route.verify_variants()}
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
                raise RuntimeError(slot_failure("gdn_verify", "; ".join(reasons)))
        if any(s.gdn_conv for s in self.slots.values()):
            reasons = gdn.verify.conv_pool_reasons(
                view.mamba, self.binding.tokens_per_request
            )
            if reasons:
                raise RuntimeError(slot_failure("gdn_conv", "; ".join(reasons)))
        if any(s.gdn_core for s in self.slots.values()):
            reasons = gdn.verify.core_pool_reasons(view.mamba)
            if reasons:
                raise RuntimeError(slot_failure("gdn_core", "; ".join(reasons)))
        if any(s.gdn_conv_core for s in self.slots.values()):
            reasons = gdn.verify.conv_core_pool_reasons(view.mamba)
            if reasons:
                raise RuntimeError(slot_failure("gdn_conv_core", "; ".join(reasons)))
        if any(s.gdn_front for s in self.slots.values()):
            from sgl_nvidia_mega.qwen3_5.gdn import front

            reasons = front.pool_reasons(view.mamba)
            if reasons:
                raise RuntimeError(slot_failure("gdn_front", "; ".join(reasons)))
            front.prepare(
                self.lm,
                view.mamba,
                trt=any(s.gdn_trt_front for s in self.slots.values()),
            )
        if any(s.bt_tail for s in self.slots.values()):
            self.bound["boundary"].bt.bind()
        skv2 = any(s.skv2 for s in self.slots.values())
        mka = any(s.mka for s in self.slots.values())
        if skv2 or mka:
            reasons = attention.pool_reasons(view, self.binding, skv2=skv2, mka=mka)
            if reasons:
                raise RuntimeError(
                    slot_failure(
                        {n for n, on in (("skv2", skv2), ("mka", mka)) if on},
                        "; ".join(reasons),
                    )
                )
        if self.recurrent_commit is not None:
            self.recurrent_commit.bind(view.mamba)
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
        # Idle DP ranks still cross the boundaries and run the MoE's collective,
        # but the decoder layers skip their mixer (including empty FP4 GEMMs).
        rows = (x[0] if isinstance(x, tuple) else x).shape[0]
        part = (
            self.mixers[i](layer, x, st)
            if not st.fb.forward_mode.is_idle() and rows > 0
            else x
        )
        x = boundary.post_attn(layer, part, st)
        with boundary.ffn_exit(layer, st) as ffn_exit:
            out = moe.step(layer, x, st, defer=ffn_exit.defer_moe_finalize)
        return ffn_exit.finish(out)


class Qwen35FoldVerify(Qwen35Verify):
    state_bytes_per_slot = 0
    state_bytes_per_request_row = 0
    keeps_draft_ssm = True
    tracks_decode_state = True
    backs_up_state = False

    _mamba = None

    def reset_state_slots(self, slots, request_rows) -> None:
        return None

    def plan(self, input_ids, forward_batch, *args, **kwargs) -> Step | None:
        step = super().plan(input_ids, forward_batch, *args, **kwargs)
        if step is None and forward_batch.forward_mode.is_target_verify():
            raise RuntimeError(
                "NVIDIA megakernel: Qwen3.5 slot gdn_commit_fold: a target verify of "
                f"{int(input_ids.shape[0])} rows is not covered (widths "
                f"{list(self.widths)}), and SGLang's verify cannot read the folded ring"
            )
        return step

    def _check_pools(self, view):
        super()._check_pools(view)
        self._mamba = view.mamba

    def commit_recurrent_state(self, io) -> bool:
        from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import (
            commit_gdn_replayssm_spec,
        )
        from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
            fused_conv_window_scatter_with_mask,
        )

        if io.track_slots is not None:
            return False
        pools = self._mamba
        if pools is None:
            raise RuntimeError(
                "NVIDIA megakernel: Qwen3.5 slot gdn_commit_fold: a commit before any "
                "covered verify"
            )
        commit_gdn_replayssm_spec(
            write_pos=pools.ring_write_pos,
            cache_base=pools.ring_cache_base,
            is_flush=pools.ring_is_flush,
            num_accepted=io.accept_lens,
            replay_indices=io.request_rows,
            state_batch_indices=io.state_slots,
            max_cache_len=pools.ring_d.shape[-2],
            max_spec_len=None,
            fold_every_commit=True,
            null_block_id=-1,
        )
        for conv, windows in zip(pools.conv, pools.intermediate_conv_window):
            fused_conv_window_scatter_with_mask(
                conv, windows, io.state_slots, io.last_step, io.src_indices_raw
            )
        return True
