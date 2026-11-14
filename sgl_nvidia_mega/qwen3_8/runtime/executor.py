# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/qwen4_exp.py (Qwen4ExpForConditionalGeneration.forward,
# Qwen4ExpVLModel.forward, Qwen4ExpModel.forward and its decoder layers' forward),
# python/sglang/srt/models/qwen3_vl.py (Qwen3VLForConditionalGeneration.forward) and
# python/sglang/srt/models/qwen4_exp_mtp.py (Qwen4ExpForCausalLMMTP.forward).
"""The executor: every covered forward of Qwen3.8, target and MTP draft.

``Qwen3_8Executor.plan`` decides from host state alone, identical on every rank, which
forwards it runs: target verify and decode steps of a built width (in CUDA graphs),
draft-extend steps (in the draft's graphs) and prefill / extend steps (eager). It
declines idle batches, embedding calls and pipeline-parallel proxies, which run the
model's own forward.

With breakable prefill CUDA graphs (``prefill_graph``: SGLang's ``SupportsPrefillBody``)
the prefill step of the target and of the MTP draft calls the layer module's forward
(``Qwen4ExpVLModel.forward``, the draft's ``Qwen4ExpModel.forward``), which SGLang
routes to ``plan_body`` / ``forward_body`` (the layer loop below) and a breakable
prefill graph captures per token bucket and replays; the ops that read the batch's host
state run there as graph breaks (``prefill_graph.py``).

``Qwen3_8Executor.forward`` takes and returns what the replaced forward does. It runs
the layer loop itself from the executor's step functions, never through a model-level
or decoder-layer ``forward``:

1. the positions the layers take (M-RoPE when the model uses it) and the embedding
   (the draft: the MTP input fusion of the embedding and the target's streams);
2. the PLE batch, then per layer, under the expert-distribution recorder's layer
   scope and the layer's prepared HC norms: the attention stage's read (HC mix, PLE
   add), the mixer (QSA attention or GDN), the attention stage's write (HC combine),
   and the FFN stage (HC read, MoE, HC write);
3. the PLE commit, the final mix of the streams and the logits, with the HC streams
   handed back as the hidden states.

Each op's implementation is chosen per width at build (``slots``); an op no kernel
serves calls SGLang's own module, and ``engagement`` records which one ran. With BF16
GEMM pins on (``bf16_gemm_pins``), the step functions launch the dense BF16 GEMMs that
SGLang's modules would (projections, router, shared expert, HC mixes, lm_head) on the
pinned kernels.
"""

from __future__ import annotations

import torch

try:
    from sglang.kernels.plugin_kernels.nvidia.spec import Step
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.megakernels.spec import Step

from sgl_nvidia_mega.qwen3_8.runtime import (
    attention,
    bf16_gemm_pins,
    flashinfer_tactic_pins,
    gdn,
    hc,
    hc_wide,
    moe,
    ple,
)
from sgl_nvidia_mega.qwen3_8.runtime.engagement import Engagement
from sgl_nvidia_mega.qwen3_8.runtime.state import StepState

STOCK = "stock"

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py": (
        "Step",
        "mode",
        "rows",
        "view",
        "widths",
        "summary",
        "plan",
        "forward",
    ),
    "srt/models/qwen4_exp.py": (
        "Qwen4ExpAttentionDecoderLayer",
        "Qwen4ExpLinearDecoderLayer",
        "_shards_rows_over_cp",
        "_MixStreams",
        "_prepare_attn_stage",
        "_run_ffn_stage",
        "self_attention",
        "attn_boundary",
        "hc_norms",
        "hyper_connection_mixer",
        "attn_hyper_connection",
        "has_ple",
        "ple_ngram_size",
        "ple_ngram_eos_token_id",
        "last_hc_hidden_states",
        "residual_batch",
        "forward(inputs_embeds)",
    ),
    "srt/models/qwen3_5.py": ("linear_attn",),
    "srt/models/qwen4_exp_mtp.py": (
        "_prepare_input_embeds",
        "_mtp_input_fusion",
        "_set_hc_logits_hidden_states",
    ),
    "srt/models/qwen3_vl.py": (
        "is_mrope_enabled",
        "language_model_only",
        "use_deepstack",
        "logits_processor",
        "lm_head",
    ),
    "srt/managers/mm_utils.py": ("prepare_mm_inputs",),
    "srt/layers/layer_boundary/residual/batch.py": ("start", "final_norm"),
    "srt/layers/logits_processor.py": ("LogitsProcessorOutput", "hidden_states"),
    "srt/eplb/expert_distribution.py": (
        "get_global_expert_distribution_recorder",
        "with_current_layer",
        "disable_this_region",
    ),
    "srt/model_executor/runner_backend_utils/breakable_cuda_graph/__init__.py": (
        "is_in_breakable_cuda_graph",
    ),
}

# The SGLang functions whose body this executor mirrors, by the digest of their code
# (test/unit/test_contract.py recomputes them on the installed SGLang). Filled in from
# the pinned SGLang revision by test/unit/test_contract.py --update.
SGLANG_MIRRORS = {}


class Qwen3_8Executor:
    """One rank's covered forwards of the target or the MTP draft."""

    # EAGLE entries: the draft is covered through plan(), SGLang runs the loop.
    draft_sampler = None
    draft_loop = None
    draft_extend = None
    begin_draft_round = None
    graph_variants = None
    speculative_runtime_hooks = None

    def __init__(
        self, lm, *, widths, draft, slots, binding=None, parity=None, decompose=False
    ):
        from sglang.srt.models.qwen4_exp import Qwen4ExpLinearDecoderLayer

        self.lm = lm
        self.widths = tuple(sorted(widths))
        self._covered = frozenset(self.widths)
        self.draft = draft
        self.slots = slots
        self.binding = binding
        self.parity = parity
        self.engagement = Engagement("draft" if draft else "target")
        # S1: the layer split into its HC read / mixer / HC write / MoE pieces,
        # called from here instead of through the stage boundaries. Only where the
        # boundaries do no collective (TP1, no DP / CP / EP).
        self.decompose = decompose
        # slot hc: set by build once the kernels compiled and every layer fits
        self.hc_ready = False
        # slot hc_wide: set by build; its row limits (hc_wide.Limits)
        self.hc_wide_ready = False
        self.hc_limits = hc_wide.DEFAULT
        m = lm.model
        # slot gdn_native_pdl: set by build; the pools are checked at the first verify
        self.gdn_pdl_ready = False
        # slot gdn_verify_glue: set by build; the pools are checked at the first verify
        self.gdn_glue_ready = False
        # slot gdn_front: set by build; the pools are checked at the first verify
        self.gdn_front_ready = False
        # slot qsa_pool: set by build with its kernels (attention.PoolKernels)
        self.qsa_pool_ready = False
        self.qsa_pool = None
        # slot qsa_pool with SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_ATTN_CORE_V1=1:
        # attention.CoreKernels, set by build
        self.qsa_core = None
        # slot qsa_pool_fp8: attention.UnionKernels, set by build; the pool is checked
        # at the first verify (the draft's first draft extend)
        self.qsa_union = None
        # slot qsa_indexer: qsa_indexer.IndexerKernels, set by build
        self.qsa_indexer = None
        # slot qsa_draft: qsa_draft.DraftGlue, set by build (the draft only)
        self.qsa_draft = None
        # slot draft_fold: the entry draft_step_fold (draft_fold.make), set by build;
        # the embedding its finalize gathered for the round's next draft step
        self.draft_step_fold = None
        self.step_embedding = None
        # slot moe_finalize: set by build
        self.moe_finalize_ready = False
        # slot moe_routing: moe_routing.Options, set by build
        self.moe_routing = None
        # set by build when SGLang cannot run a target verify itself (u-cache pools)
        self.verify_required = False
        # the u-cache verify (gdn_ucache.UcacheVerify), set by prepare_ucache
        self.gdn_ucache = None
        # the plugin's own u-cache rings (gdn_ucache_state.UcacheState), set by build
        self.ucache_state = None
        # slot gdn_prefill: the prefill kernels' gdn_prefill.Options, set by build
        self.gdn_prefill = None
        # slot qsa_prefill: qsa_prefill.Options, set by build
        self.qsa_prefill = None
        # slot ple: ple.Options, set by build
        self.ple_options = None
        # slot smallm_gemm: smallm_gemm.Options, set by build
        self.smallm_gemm = None
        # SGLANG_FLASHINFER_TACTIC_PIN_FILE's MoE table applies, set by build
        self.moe_pins = False
        # set by build: the prefill body runs in breakable prefill graphs
        self.prefill_graph = False
        # gdn_prefill's in-graph core outputs, newest first (gdn_prefill._graph_output)
        self.gdn_graph_outputs = []
        self._pools_checked = False
        self.mixers = {
            i: (
                self._gdn
                if isinstance(m.layers[i], Qwen4ExpLinearDecoderLayer)
                else self._qsa
            )
            for i in range(m.start_layer, m.end_layer)
        }

    def summary(self) -> str:
        what = "draft" if self.draft else "target"
        return f"Qwen3.8 {what} executor; slots {sorted(self.slots) or 'none (stock)'}"

    def plan(
        self,
        input_ids,
        forward_batch,
        get_embedding=False,
        pp_proxy_tensors=None,
        **kwargs,
    ):
        """The step if this forward is covered, else None (the model's forward runs).
        Host state only, so a captured graph replays the decision made at capture."""
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
            is_in_breakable_cuda_graph,
        )
        from sglang.srt.models.qwen4_exp import _shards_rows_over_cp

        if self.parity is not None:
            self.parity.flush()
        if kwargs or get_embedding or pp_proxy_tensors is not None:
            return None
        mode = forward_batch.forward_mode
        if mode.is_idle() or _shards_rows_over_cp(forward_batch):
            return None
        rows = int(
            (input_ids if input_ids is not None else forward_batch.input_ids).shape[0]
        )
        if mode.is_target_verify():
            name = None if self.draft else "target_verify"
        elif mode.is_decode():
            name = "decode"
        elif mode.is_draft_extend_v2():
            name = "draft_extend" if self.draft else None
        elif mode.is_extend_or_draft_extend_or_mixed():
            # a breakable prefill graph's replay runs the outer step eagerly around it
            if is_in_breakable_cuda_graph() and not self.prefill_graph:
                return None
            return Step(mode="prefill", rows=rows)
        else:
            name = None
        if name is None or rows not in self._covered:
            if name == "target_verify" and self.verify_required:
                raise RuntimeError(
                    f"Qwen3.8 executor: a {rows}-row target verify is not covered, and "
                    "SGLang has no verify of its own for this pool (u-cache)"
                )
            return None
        return Step(mode=name, rows=rows)

    def plan_body(
        self,
        input_ids,
        forward_batch,
        input_embeds=None,
        pp_proxy_tensors=None,
        input_deepstack_embeds=None,
        inputs_embeds=None,
        **kwargs,
    ):
        """SupportsPrefillBody: a "prefill" step for the layer module's forward of an
        extend / mixed batch, as ``plan`` covers it but captured or replayed in a
        breakable prefill graph too; None (the module's own forward) otherwise. Host
        state only: a captured bucket replays the decision made at its capture. The
        draft's layer module names its embeddings ``inputs_embeds``."""
        from sglang.srt.models.qwen4_exp import _shards_rows_over_cp

        mode = forward_batch.forward_mode
        if (
            not self.prefill_graph
            or kwargs
            or (inputs_embeds is not None and not self.draft)
            or pp_proxy_tensors is not None
            or input_deepstack_embeds is not None
            or mode.is_idle()
            or mode.is_target_verify()
            or mode.is_decode()
            or mode.is_draft_extend_v2()
            or not mode.is_extend_or_draft_extend_or_mixed()
            or _shards_rows_over_cp(forward_batch)
        ):
            return None
        ids = input_ids if input_ids is not None else forward_batch.input_ids
        return Step(mode="prefill", rows=int(ids.shape[0]))

    @torch.no_grad()
    def forward_body(
        self,
        step,
        original,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        pp_proxy_tensors=None,
        input_deepstack_embeds=None,
        inputs_embeds=None,
        **kwargs,
    ):
        """SupportsPrefillBody: ``Qwen4ExpVLModel.forward`` for a planned prefill body,
        the layer loop on ``input_embeds`` (a capture's static slot); sets
        ``last_hc_hidden_states`` and returns the mixed hidden rows. The draft's:
        ``Qwen4ExpModel.forward`` on the MTP streams (``inputs_embeds``), returning
        the mixed hidden rows and the HC streams."""
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
            is_in_breakable_cuda_graph,
        )

        m = self.lm.model
        if inputs_embeds is not None:
            input_embeds = inputs_embeds
        if not self.draft:
            m.last_hc_hidden_states = None
        if input_ids is None:
            input_ids = forward_batch.input_ids
        st = StepState(
            lm=self.lm,
            step=step,
            forward_batch=forward_batch,
            positions=positions,
            input_ids=input_ids,
            slots=self.slots,
            binding=self.binding,
            engagement=self.engagement,
        )
        # Eager, the outer prefill step records it; a capture (or its warmup) alone.
        st.graph = is_in_breakable_cuda_graph()
        if st.graph:
            self.engagement.begin()
        hidden, hc_hidden = self._language_model(st, input_ids, input_embeds)
        if st.graph:
            self.engagement.end(step.mode, step.rows)
        if self.draft:
            return hidden, hc_hidden
        m.last_hc_hidden_states = hc_hidden
        return hidden

    def forward(
        self, step, original, input_ids, positions, forward_batch, *args, **kwargs
    ):
        """The covered step (``original``: the model's own forward, bound, which the
        parity check runs)."""
        run = lambda: self._forward(step, input_ids, positions, forward_batch)
        if self.parity is not None:
            stock = lambda: original(
                input_ids, positions, forward_batch, *args, **kwargs
            )
            return self.parity.run(step, forward_batch, stock, run)
        return run()

    def _gdn(self, layer, x, st):
        return gdn.step(self, layer, x, st)

    def _qsa(self, layer, x, st):
        return attention.step(self, layer, x, st)

    def _check_pools(self, forward_batch):
        """The pools exist from the first covered verify (the draft's first draft
        extend) on; a capture warmup runs it eagerly before any graph is captured.
        Refuse a slot they cannot serve."""
        self._pools_checked = True
        if self.gdn_ucache is not None:
            # the u-cache verify serves every GDN verify
            self.gdn_pdl_ready = self.gdn_glue_ready = self.gdn_front_ready = False
        if self.gdn_pdl_ready or self.gdn_glue_ready or self.gdn_front_ready:
            from sgl_nvidia_mega.qwen3_8.runtime import unavailable

            m = self.lm.model
            first = next(i for i, f in self.mixers.items() if f == self._gdn)
            args = (
                gdn.linear_backend(),
                m.layers[first].linear_attn.attn.layer_id,
                int(forward_batch.spec_info.draft_token_num),
            )
            if self.gdn_pdl_ready:
                reason = gdn.pool_reasons(*args)
                if reason:
                    unavailable("gdn_native_pdl", reason)
                    self.gdn_pdl_ready = False
            if self.gdn_glue_ready:
                reason = gdn.glue_pool_reasons(*args)
                if reason:
                    unavailable("gdn_verify_glue", reason)
                    self.gdn_glue_ready = False
            if self.gdn_front_ready:
                reason = gdn.front_pool_reasons(*args)
                if reason:
                    unavailable("gdn_front", reason)
                    self.gdn_front_ready = False
        if self.qsa_union is not None:
            from sgl_nvidia_mega.qwen3_8.runtime import unavailable

            first = next(i for i, f in self.mixers.items() if f == self._qsa)
            reason = attention.union_pool_reasons(
                attention.sparse_backend(), self.lm.model.layers[first].attn.layer_id
            )
            if reason:
                unavailable("qsa_pool_fp8", reason)
                self.qsa_union = None

    def _build_ucache_verify(self, mamba, buffers, tokens):
        from sgl_nvidia_mega.qwen3_8.runtime import gdn_ucache

        reason = gdn_ucache.pool_reasons(buffers, tokens, state=mamba.temporal)
        if reason:
            # SGLang has no u-cache verify of its own to fall back to
            raise RuntimeError(f"Qwen3.8 executor: u-cache verify: {reason}")
        device = mamba.temporal.device
        self.gdn_ucache = gdn_ucache.UcacheVerify(
            device=device,
            first_layer_id=min(mamba.layer_ids),
            by_request_row=bool(buffers.by_request_row),
            options=gdn_ucache.default_options(device),
        )

    @torch.no_grad()
    def _forward(self, step, input_ids, positions, forward_batch):
        first = "draft_extend" if self.draft else "target_verify"
        if not self._pools_checked and step.mode == first:
            self._check_pools(forward_batch)
        lm = self.lm
        st = StepState(
            lm=lm,
            step=step,
            forward_batch=forward_batch,
            positions=positions,
            input_ids=input_ids,
            slots=self.slots,
            binding=self.binding,
            engagement=self.engagement,
        )
        self.engagement.begin()
        out = self._draft(st) if self.draft else self._target(st)
        self.engagement.end(step.mode, step.rows)
        if self.moe_pins:
            flashinfer_tactic_pins.after_forward()
        return out

    def _target(self, st):
        """``Qwen4ExpForConditionalGeneration.forward`` (through
        ``Qwen3VLForConditionalGeneration.forward`` and ``Qwen4ExpVLModel.forward``)."""
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        lm, fb = st.lm, st.fb
        if lm.is_mrope_enabled:
            st.positions = fb.mrope_positions
        lm.model.last_hc_hidden_states = None
        if lm.language_model_only:
            ids, embeds = st.input_ids, None
        else:
            from sglang.srt.managers.mm_utils import prepare_mm_inputs

            # general_mm_embed_routine: the language model then gets input_ids=None
            # and reads the batch's own ids (the PLE needs them).
            embeds = prepare_mm_inputs(
                st.input_ids,
                fb,
                st.model,
                {"positions": st.positions, "pp_proxy_tensors": None},
                multimodal_model=lm,
                use_deepstack=lm.use_deepstack,
            )
            ids = fb.input_ids
        if ids is None:
            ids = fb.input_ids
        st.ran("embed", STOCK)
        if self.prefill_graph and st.step.mode == "prefill":
            from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
                is_in_breakable_cuda_graph,
            )

            # Through the attribute: SGLang's body dispatch (plan_body), or a breakable
            # prefill graph's replay of the captured body, which restores the streams.
            hidden = lm.model.forward(ids, st.positions, fb, input_embeds=embeds)
            hc_hidden = lm.model.last_hc_hidden_states
            if is_in_breakable_cuda_graph():
                st.ran("layers", "prefill_graph")
        else:
            hidden, hc_hidden = self._language_model(st, ids, embeds)
            lm.model.last_hc_hidden_states = hc_hidden
        logits = bf16_gemm_pins.logits_processor(lm.logits_processor)
        out = logits(st.input_ids, hidden, lm.lm_head, fb, None)
        st.ran("logits", STOCK)
        if hc_hidden is not None and isinstance(out, LogitsProcessorOutput):
            out.hidden_states = hc_hidden
        return out

    def _draft(self, st):
        """``Qwen4ExpForCausalLMMTP.forward``."""
        from sglang.srt.eplb.expert_distribution import (
            get_global_expert_distribution_recorder,
        )
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
            is_in_breakable_cuda_graph,
        )

        from sgl_nvidia_mega.qwen3_8.runtime import draft_fold

        lm, fb = st.lm, st.fb
        # slot draft_fold: this round's fold (an SGLang with the draft_step_fold entry)
        fold = getattr(fb, "draft_step_fold", None)
        if not isinstance(fold, draft_fold.StepFold) or fold.executor is not self:
            fold = None
        embeds = self.step_embedding if fold is not None else None
        if embeds is not None:
            # the previous step's finalize gathered this embedding
            self.step_embedding = None
            st.ran("embed", "draft_fold")
        else:
            embeds = lm._prepare_input_embeds(st.input_ids, fb, None)
            st.ran("embed", STOCK)
        streams = lm._mtp_input_fusion(embeds, fb.spec_info.hidden_states)
        st.ran("mtp_fusion", STOCK)
        with get_global_expert_distribution_recorder().disable_this_region():
            if self.prefill_graph and st.step.mode == "prefill":
                # Through the attribute, as _target: SGLang's body dispatch, or a
                # breakable prefill graph's replay of the captured body.
                hidden, hc_hidden = lm.model.forward(
                    st.input_ids, st.positions, fb, streams
                )
                if is_in_breakable_cuda_graph():
                    st.ran("layers", "prefill_graph")
            else:
                hidden, hc_hidden = self._language_model(st, st.input_ids, streams)
        if fold is not None and fold.bf16:
            # the fold's top-1 reads the LM head's BF16 logits: no FP32 copy
            logits = draft_fold.bf16_logits(lm, hidden)
            out = LogitsProcessorOutput(next_token_logits=logits)
            st.ran("logits", "lm_head_bf16")
        else:
            logits = bf16_gemm_pins.logits_processor(lm.logits_processor)
            out = logits(st.input_ids, hidden, lm.lm_head, fb)
            st.ran("logits", STOCK)
        lm._set_hc_logits_hidden_states(out, hc_hidden, fb)
        return out

    def _language_model(self, st, input_ids, inputs_embeds):
        """``Qwen4ExpModel.forward`` on one stage holding every layer, without CP row
        sharding (plan declines both): the layers, the PLE commit and the final mix;
        returns the mixed hidden rows and the HC streams."""
        from sglang.srt.eplb.expert_distribution import (
            get_global_expert_distribution_recorder,
        )
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
        from sglang.srt.models.qwen4_exp import _MixStreams

        m, fb = st.model, st.fb
        hidden = (
            inputs_embeds if inputs_embeds is not None else m.embed_tokens(input_ids)
        )
        residual_batch.start(fb)
        if st.graph:
            from sgl_nvidia_mega.qwen3_8.runtime import prefill_graph
        if m.has_ple:
            st.ple_batch = (
                prefill_graph.ple_prepare(self, input_ids, st)
                if st.graph
                else ple.prepare(st, input_ids, options=self.ple_options)
            )
        recorder = get_global_expert_distribution_recorder()
        normalized_input = None
        for i in range(m.start_layer, m.end_layer):
            layer = m.layers[i]
            next_norm = None
            if i + 1 < m.end_layer:
                next_layer = m.layers[i + 1]
                if next_layer.ple is not None and st.graph:
                    prefill_graph.ple_prefetch(self, next_layer, st)
                elif next_layer.ple is not None:
                    ple.prefetch(next_layer, st.ple_batch, st, options=self.ple_options)
                else:
                    next_norm = next_layer.attn_hyper_connection.hc_norm
                st.scratch["hc.next_inject"] = (
                    None
                    if next_layer.ple is not None
                    else next_layer.attn_hyper_connection.block_inject_weight.weight
                )
                # the next read a wide write-back may take over (hc_wide.boundary)
                st.scratch["hc.next_hc"] = (
                    None
                    if next_layer.ple is not None
                    else next_layer.attn_hyper_connection
                )
            elif m.pp_group.is_last_rank:
                next_norm = m.hyper_connection_mixer.hc_norm
                st.scratch["hc.next_inject"] = st.scratch["hc.next_hc"] = None
            with (
                recorder.with_current_layer(i),
                layer.hc_norms(normalized_input, next_norm) as norms,
            ):
                if self.decompose:
                    hidden = self._layer_streams(i, layer, hidden, st)
                else:
                    hidden = self._layer(i, layer, hidden, st)
            normalized_input = norms.output
            if self.parity is not None:
                self.parity.record_layer(i, hidden)
        if st.graph and m.has_ple:
            prefill_graph.ple_commit(self, st)
        else:
            ple.commit(st.ple_batch, st)
        hc_hidden = hidden
        mixer = bf16_gemm_pins.MixStreams if bf16_gemm_pins.routed() else _MixStreams
        mix = mixer(m.hyper_connection_mixer, normalized_input)
        if self.decompose:
            residual_batch.clear(fb)
            hidden = mix(hidden)
        else:
            hidden = residual_batch.final_norm(hidden, fb, mix)
        st.ran("hc.final_mix", STOCK)
        return hidden, hc_hidden

    def _layer(self, i, layer, hidden, st):
        """A decoder layer: the attention stage's read, the mixer and the stage's
        write, then the FFN stage (read, MoE, write)."""
        fb = st.fb
        hidden = layer._prepare_attn_stage(hidden, fb, st.ple_batch)
        st.ran("hc.attn_read", STOCK)
        if not fb.forward_mode.is_idle():
            hidden = self.mixers[i](layer, hidden, st)
        hidden = layer.attn_boundary.finish(hidden, fb)
        st.ran("hc.attn_write", STOCK)
        hidden = layer._run_ffn_stage(hidden, fb)
        st.ran("moe+hc.ffn", STOCK)
        return hidden

    def _ple_add(self, layer, streams, st):
        """``ple.add`` on the layer's streams; a graph break in a prefill graph."""
        if st.graph:
            from sgl_nvidia_mega.qwen3_8.runtime import prefill_graph

            return prefill_graph.ple_add(self, layer, streams, st)
        return ple.add(layer, streams, st.ple_batch, st, options=self.ple_options)

    def _layer_streams(self, i, layer, streams, st):
        """S1: the decoder layer on the HC streams, at TP1. The stage boundaries
        there apply the attention write-back at the FFN read and the FFN
        write-back at the stage's finish, both inside the layer's prepared-norm
        scope, so this is the same sequence of the layer's own HC calls. Slot
        ``hc`` serves the write-backs of up to 32 rows (``hc.py``), slot ``hc_wide``
        the reads and write-backs of more (``hc_wide.py``)."""
        streams = layer._widen_streams(streams)
        rows = streams.shape[0]
        if self.hc_wide_ready and rows > hc.MAX_ROWS:
            return self._layer_streams_wide(i, layer, streams, st)
        fb = st.fb
        norms = layer._hc_norms
        use_hc = self.hc_ready and rows <= hc.MAX_ROWS
        if layer.ple is not None:
            # _attn_mix's PLE add, then its read of the changed streams
            streams = self._ple_add(layer, streams, st)
            x, (res, normed, partials) = bf16_gemm_pins.gated_mix(
                layer.attn_hyper_connection, streams, None
            )
        else:
            x, (res, normed, partials) = bf16_gemm_pins.attn_mix(layer, streams)
        st.ran("hc.attn_read", STOCK)
        if use_hc:
            carried = st.scratch.pop("hc.partials", None)
            if partials is not None:
                # SGLang's mix computed the gate partials on its side stream
                hc.join_gate_stream(layer.attn_hyper_connection, partials)
            if carried is not None and layer.ple is None:
                partials = carried
            elif partials is None:
                partials = hc.gate(
                    normed, layer.attn_hyper_connection.block_inject_weight.weight
                )
        if not fb.forward_mode.is_idle():
            x = self.mixers[i](layer, x, st)
        mlp_hc = layer.mlp_hyper_connection
        if use_hc:
            streams, norms.ffn_input, ffn_partials = hc.write(
                x, res, partials, mlp_hc.hc_norm, mlp_hc.block_inject_weight.weight
            )
            st.ran("hc.attn_write", "hc_norm_gate")
        else:
            streams = layer._attn_combine(x, (res, normed, partials))
            st.ran("hc.attn_write", STOCK)
        x, (res, normed, partials) = bf16_gemm_pins.ffn_mix(layer, streams)
        st.ran("hc.ffn_read", STOCK)
        if use_hc and partials is not None:
            hc.join_gate_stream(mlp_hc, partials)
        x = moe.step(self, layer, x, st)
        if use_hc:
            streams, norms.output, st.scratch["hc.partials"] = hc.write(
                x, res, ffn_partials, norms.next_norm, st.scratch.get("hc.next_inject")
            )
            st.ran("hc.ffn_write", "hc_norm_gate")
        else:
            streams = layer._ffn_combine(x, (res, normed, partials))
            st.ran("hc.ffn_write", STOCK)
        return streams

    def _layer_streams_wide(self, i, layer, streams, st):
        """The S1 layer for more than 32 rows with slot ``hc_wide``: the reads mix on
        the wide split-K kernels and the write-backs combine, normalize for the
        next read and compute its gate totals in one launch where a kernel serves
        the rows (each falls back to SGLang's own read / write-back otherwise)."""
        fb = st.fb
        norms = layer._hc_norms
        attn_hc, mlp_hc = layer.attn_hyper_connection, layer.mlp_hyper_connection
        carried = st.scratch.pop("hc.partials", None)
        # the previous layer's write-back already ran this read (hc_wide.boundary)
        mixed = st.scratch.pop("hc.mixed", None)
        normalized = norms.attn_input
        if layer.ple is not None:
            streams = self._ple_add(layer, streams, st)
            # the PLE embedding changed the streams after the last write-back
            normalized = carried = None
        if mixed is not None:
            x, normed, partials = mixed, normalized, carried
            st.ran("hc.attn_read", "hc_boundary_ks")
        else:
            x, normed, partials = hc_wide.mix(
                attn_hc, streams, normalized, carried, limits=self.hc_limits
            )
            st.ran(
                "hc.attn_read",
                hc_wide.mix_route(attn_hc, normed, limits=self.hc_limits),
            )
        if not fb.forward_mode.is_idle():
            x = self.mixers[i](layer, x, st)
        inject = mlp_hc.block_inject_weight.weight
        if not hc_wide.boundary_reason(
            attn_hc, x, streams, partials, mlp_hc, inject, limits=self.hc_limits
        ):
            # the attention write-back and the FFN read in one launch
            streams, normed, partials, x = hc_wide.boundary(
                attn_hc, x, streams, partials, mlp_hc, inject, limits=self.hc_limits
            )
            norms.ffn_input = normed
            st.ran("hc.attn_write", "hc_boundary_ks")
            st.ran("hc.ffn_read", "hc_boundary_ks")
        else:
            st.ran(
                "hc.attn_write",
                hc_wide.write_route(
                    attn_hc,
                    x,
                    streams,
                    normed,
                    partials,
                    mlp_hc.hc_norm,
                    limits=self.hc_limits,
                ),
            )
            streams, ffn_in, ffn_partials = hc_wide.write(
                attn_hc,
                x,
                streams,
                normed,
                partials,
                mlp_hc.hc_norm,
                inject,
                limits=self.hc_limits,
            )
            norms.ffn_input = ffn_in
            x, normed, partials = hc_wide.mix(
                mlp_hc, streams, ffn_in, ffn_partials, limits=self.hc_limits
            )
            st.ran(
                "hc.ffn_read", hc_wide.mix_route(mlp_hc, normed, limits=self.hc_limits)
            )
        x = moe.step(self, layer, x, st)
        next_norm, next_inject = norms.next_norm, st.scratch.get("hc.next_inject")
        next_hc = st.scratch.get("hc.next_hc")
        if next_hc is not None and not hc_wide.boundary_reason(
            mlp_hc, x, streams, partials, next_hc, next_inject, limits=self.hc_limits
        ):
            # the FFN write-back and the next layer's attention read in one launch
            (
                streams,
                norms.output,
                st.scratch["hc.partials"],
                st.scratch["hc.mixed"],
            ) = hc_wide.boundary(
                mlp_hc,
                x,
                streams,
                partials,
                next_hc,
                next_inject,
                limits=self.hc_limits,
            )
            st.ran("hc.ffn_write", "hc_boundary_ks")
            return streams
        st.ran(
            "hc.ffn_write",
            hc_wide.write_route(
                mlp_hc, x, streams, normed, partials, next_norm, limits=self.hc_limits
            ),
        )
        streams, norms.output, st.scratch["hc.partials"] = hc_wide.write(
            mlp_hc,
            x,
            streams,
            normed,
            partials,
            next_norm,
            next_inject,
            limits=self.hc_limits,
        )
        return streams


class Qwen3_8UcacheExecutor(Qwen3_8Executor):
    """The target executor with the plugin's own GDN u-cache: binding v2's
    SupportsStateSlotReset and SupportsRecurrentCommit, through its UcacheState."""

    @property
    def state_bytes_per_slot(self) -> int:
        return self.ucache_state.state_bytes_per_slot

    @property
    def state_bytes_per_request_row(self) -> int:
        return self.ucache_state.state_bytes_per_request_row

    @property
    def keeps_draft_ssm(self) -> bool:
        return self.ucache_state.keeps_draft_ssm

    @property
    def tracks_decode_state(self) -> bool:
        return self.ucache_state.tracks_decode_state

    @property
    def backs_up_state(self) -> bool:
        return self.ucache_state.backs_up_state

    def reset_state_slots(self, slots, request_rows) -> None:
        self.ucache_state.reset_state_slots(slots, request_rows)

    def commit_recurrent_state(self, io) -> bool:
        return self.ucache_state.commit(io)

    def prepare_ucache(self, view) -> None:
        """spec.prepare: allocate the rings once the pools exist, then the verify."""
        state = self.ucache_state
        state.prepare(view)
        if self.gdn_ucache is None:
            self._build_ucache_verify(
                view.mamba, state.buffers, state.tokens_per_request
            )
