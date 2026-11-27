# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0): Qwen3VLForConditionalGeneration.forward,
# Qwen3_5ForCausalLM.forward, Qwen3_5LinearDecoderLayer.forward and
# Qwen3_5AttentionDecoderLayer.forward.

from __future__ import annotations

import contextlib
import functools
import logging

import torch
from sglang.kernels.plugin_kernels.nvidia.spec import Step

from sgl_nvidia_mega.qwen3_5.attention import prefill_rows
from sgl_nvidia_mega.qwen3_5.gdn import prefill as gdn_prefill
from sgl_nvidia_mega.qwen3_5.gdn import prefill_fi
from sgl_nvidia_mega.qwen3_5.moe import prefill as moe_prefill
from sgl_nvidia_mega.qwen3_5.runtime import parity

logger = logging.getLogger(__name__)

SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py": (
        "Step",
        "prefill_graph",
        "SupportsPrefillBody",
        "plan_body",
        "forward_body",
        "linear_prefill_graph_seqs",
        "refill_prefill_graph",
    ),
    "srt/model_executor/runner_backend_utils/breakable_cuda_graph/__init__.py": (
        "eager_on_graph",
    ),
    "srt/model_executor/runner_utils/forward_batch.py": ("set_forward_batch",),
    "srt/models/qwen3_5.py": (
        "Qwen3_5LinearDecoderLayer",
        "Qwen3_5AttentionDecoderLayer",
        "residual_batch",
        "layers",
        "start_layer",
        "end_layer",
        "pp_group",
        "embed_tokens",
        "attn_boundary",
        "ffn_boundary",
        "linear_attn",
        "self_attention",
        "self_attention(positions)",
        "self_attention(hidden_states)",
        "self_attention(forward_batch)",
        "mlp",
        "norm",
        "flashinfer_mnnvl_cutedsl_fusion",
        '"_is_layer_to_capture"',
    ),
    "srt/models/qwen3_vl.py": (
        "language_model_only",
        "is_mrope_enabled",
        "capture_aux_hidden_states",
        "logits_processor",
        "lm_head",
    ),
    "srt/layers/layer_boundary/residual/batch.py": ("start", "to_pp", "final_norm"),
    "srt/layers/layer_boundary/stage.py": ("from_pp", "prepare", "finish", "exit"),
    "srt/layers/layer_boundary/exit.py": ("defer_moe_finalize", "finish"),
    "srt/layers/aux_hidden_states.py": ("AuxHiddenStateList",),
    "srt/eplb/expert_distribution.py": (
        "get_global_expert_distribution_recorder",
        "with_current_layer",
    ),
    "srt/model_executor/forward_batch_info.py": (
        "mrope_positions",
        "mm_inputs",
        "tensors",
    ),
    "srt/model_executor/forward_mode.py|srt/model_executor/forward_batch_info.py": (
        "is_extend_without_speculative",
        "is_idle",
    ),
    "srt/layers/logits_processor.py": (
        "next_token_logits",
        "hidden_states",
        "last_hidden_states",
    ),
}

SGLANG_MIRRORS = {
    "srt/models/qwen3_vl.py:Qwen3VLForConditionalGeneration.forward": (
        "d394d79ba7fa7b31",
        "dbdcf4b88a2c0d40",
    ),
    "srt/models/qwen3_5.py:Qwen3_5ForCausalLM.forward": "9e530b264ce30ad6",
    "srt/models/qwen3_5.py:Qwen3_5LinearDecoderLayer.forward": "2298a57111f9430e",
    "srt/models/qwen3_5.py:Qwen3_5AttentionDecoderLayer.forward": "6473f673a1a0796b",
}


class Qwen35Prefill:
    draft_sampler = draft_loop = draft_extend = begin_draft_round = None
    graph_variants = None
    speculative_runtime_hooks = None

    def __init__(
        self,
        model,
        ctx,
        enabled,
        label="TP1/PP4",
        split_conv=True,
        front=False,
        track_snapshot=False,
        moe_input_nvfp4=False,
        moe_add_norm=False,
        moe_noclone=False,
        fp8_rows=(),
        qk_rows=False,
    ):
        from sglang.srt.models.qwen3_5 import Qwen3_5LinearDecoderLayer

        self.lm = model
        # split_conv: gdn_prefill_split_conv is declared at this shape
        self.label, self.split_conv = label, split_conv
        self.enabled = enabled
        # SGLang's EagleBinding (BuildContext.binding): the live prefill view
        self.binding = getattr(ctx, "binding", None)
        # front: gdn_prefill_front runs at this shape (_build_front checks it)
        self.front = False
        # gdn_prefill_track_snapshot's state in the front's extend, where it is built
        self.snapshot = None
        # qk_rows: attn_prefill_qk_rows runs at this shape (_build_qk_rows checks it)
        self.qk_rows = False
        self._declined = set()
        self.parity = PrefillParity() if parity.ENABLED else None
        # Widths describe decode-graph installation; P uses the ordinary eager
        # forward wrapper and plans its actual chunk rows independently.
        self.widths = (ctx.tokens_per_request,)
        self._linear = {
            i
            for i in range(model.model.start_layer, model.model.end_layer)
            if isinstance(model.model.layers[i], Qwen3_5LinearDecoderLayer)
        }
        self._engaged = False
        for i in self._linear:
            gdn = model.model.layers[i].linear_attn
            if (
                gdn.attn.q_dim + gdn.attn.k_dim + gdn.attn.v_dim
                <= gdn_prefill.FUSED_SPLIT_LIMIT
                or gdn.attn_tp_size != 1
            ):
                self.enabled = False
        if front:
            self.front = self._build_front(model)
        if track_snapshot:
            self.snapshot = self._build_track_snapshot()
        if qk_rows:
            self.qk_rows = self._build_qk_rows(model)
        self.moe = (
            self._build_moe(model, moe_input_nvfp4, moe_add_norm, moe_noclone)
            if moe_input_nvfp4 or moe_add_norm or moe_noclone
            else None
        )

        self.fp8 = None
        if fp8_rows and self.front:
            from sgl_nvidia_mega.qwen3_5.runtime import prefill_fp8

            self.fp8 = prefill_fp8.PrefillFp8.build(model, fp8_rows, self._linear)
        elif fp8_rows:
            logger.info(
                "NVIDIA megakernel: Qwen3.5 slots %s use stock (they run in the "
                "prefill body, which gdn_prefill_front builds)",
                list(fp8_rows),
            )

    def _build_moe(self, model, nvfp4, add_norm, noclone=False):
        slots = [
            name
            for name, on in (
                (moe_prefill.SLOT, nvfp4),
                (moe_prefill.SLOT_ADD_NORM, add_norm),
                (moe_prefill.SLOT_NOCLONE, noclone),
            )
            if on
        ]
        built, why = None, {}
        if self.front:
            built, why = moe_prefill.build(
                model,
                range(model.model.start_layer, model.model.end_layer),
                nvfp4=nvfp4,
                add_norm=add_norm,
                noclone=noclone,
            )
        else:
            why = dict.fromkeys(slots, "no prefill body (gdn_prefill_front stock)")
        for name in slots:
            if name in why:
                logger.info(
                    "NVIDIA megakernel: Qwen3.5 slot %s uses stock (%s)",
                    name,
                    why[name],
                )
        return built

    def _build_front(self, model) -> bool:
        from sgl_nvidia_mega.qwen3_5 import spec
        from sgl_nvidia_mega.qwen3_5.sm103.gdn import prefill_front

        why = (
            ""
            if spec.prefill_body_offered()
            else "the spec offers no prefill body here"
        ) or prefill_fi.sglang_unsupported(self.binding)
        shapes = {}
        for i in sorted(self._linear) if not why else ():
            gdn = model.model.layers[i].linear_attn
            why = (
                "attention TP"
                if gdn.attn_tp_size != 1
                else prefill_fi.layer_unsupported(gdn.attn, prefill_front)
            )
            if why:
                why = f"layer {i}: {why}"
                break
            attn = gdn.attn
            heads = gdn.num_v_heads // gdn.attn_tp_size
            shape = (attn.num_k_heads, attn.num_v_heads, attn.bias is not None)
            shapes.setdefault((*shape, attn.activation, heads), attn)
        if why:
            logger.info(
                "NVIDIA megakernel: Qwen3.5 slot gdn_prefill_front uses stock (%s)", why
            )
            return False
        for (nk, nv, has_bias, activation, heads), attn in sorted(
            shapes.items(), key=repr
        ):
            prefill_front.precompile(
                torch.cuda.current_device(), nk, nv, has_bias, activation
            )
            prefill_front.warm(attn, heads)
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
            eager_on_graph,
        )

        # SGLang's prefill core is a graph break (RadixLinearAttention); so is this one
        self._core = eager_on_graph(self._run_core)
        return True

    def _build_track_snapshot(self):
        from sgl_nvidia_mega.qwen3_5.gdn import fi_track_snapshot

        why = (
            "no gdn_prefill_front"
            if not self.front
            else "; ".join(fi_track_snapshot.flashinfer_reasons())
        )
        if why:
            logger.info(
                "NVIDIA megakernel: Qwen3.5 slot gdn_prefill_track_snapshot uses stock "
                "(%s)",
                why,
            )
            return None
        return fi_track_snapshot.TrackSnapshot()

    def _build_qk_rows(self, model) -> bool:
        from sglang.srt.models.qwen3_5 import Qwen3_5AttentionDecoderLayer

        from sgl_nvidia_mega.qwen3_5 import spec
        from sgl_nvidia_mega.qwen3_5.sm103.attention import qk_norm_rope_rows as rows

        why = (
            ""
            if spec.prefill_body_offered()
            else "the spec offers no prefill body here"
        )
        layers = [
            (i, model.model.layers[i])
            for i in range(model.model.start_layer, model.model.end_layer)
            if i not in self._linear
        ]
        shapes = {}
        for i, layer in layers if not why else ():
            why = (
                prefill_rows.layer_unsupported(layer, rows)
                if isinstance(layer, Qwen3_5AttentionDecoderLayer)
                else "not a full-attention layer"
            )
            if why:
                why = f"layer {i}: {why}"
                break
            rotary = layer.rotary_emb
            shape = (layer.num_heads, layer.num_kv_heads, layer.head_dim)
            shape += (rotary.rotary_dim, layer.q_norm.variance_epsilon)
            shapes.setdefault((*shape, getattr(rotary, "axis_map", None) is None), i)
        if why:
            logger.info(
                "NVIDIA megakernel: Qwen3.5 slot %s uses stock (%s)",
                prefill_rows.SLOT,
                why,
            )
            return False
        for i in shapes.values():
            prefill_rows.warm(model.model.layers[i], rows)
        return True

    def _body_slots(self) -> list:
        """The prefill body's slots this build runs."""
        return [
            name
            for name, on in (
                ("gdn_prefill_front", self.front),
                (prefill_rows.SLOT, self.qk_rows),
            )
            if on
        ]

    def summary(self):
        if self._body_slots():
            built = ", ".join(self._body_slots())
            moe = "".join(
                f"; {name} built ({len(ids)} layers)"
                for name, ids in (
                    (moe_prefill.SLOT, getattr(self.moe, "nvfp4", ())),
                    (moe_prefill.SLOT_ADD_NORM, getattr(self.moe, "add_norm", ())),
                    (moe_prefill.SLOT_NOCLONE, getattr(self.moe, "noclone", ())),
                )
                if ids
            )
            snapshot = (
                "; gdn_prefill_track_snapshot built"
                if self.snapshot is not None
                else ""
            )
            rows = self.fp8.rows() if self.fp8 is not None else []
            fp8 = f"; {', '.join(rows)} built" if rows else ""
            body = f"{built} built (prefill body){snapshot}"
            return f"prefill {self.label}; {body}{moe}{fp8}"
        if not self.split_conv:
            return f"prefill {self.label}; SGLang's forward"
        return f"prefill {self.label}; gdn_prefill_split_conv " + (
            "built" if self.enabled else "stock"
        )

    def plan(
        self,
        input_ids,
        forward_batch,
        get_embedding=False,
        pp_proxy_tensors=None,
        **kwargs,
    ):
        if (
            self._body_slots()
            or (not self.enabled and self.parity is None)
            or kwargs
            or get_embedding
            or any(item is not None for item in (forward_batch.mm_inputs or ()))
            or not forward_batch.forward_mode.is_extend_without_speculative()
        ):
            return None
        return Step(mode="prefill", rows=input_ids.shape[0])

    def forward(
        self,
        step,
        original,
        input_ids,
        positions,
        forward_batch,
        get_embedding=False,
        pp_proxy_tensors=None,
        **kwargs,
    ):
        if gdn_prefill.unsupported(step.view):
            return original(
                input_ids,
                positions,
                forward_batch,
                get_embedding=get_embedding,
                pp_proxy_tensors=pp_proxy_tensors,
                **kwargs,
            )
        if self.parity is not None:
            stock = lambda: original(
                input_ids,
                positions,
                forward_batch,
                get_embedding=get_embedding,
                pp_proxy_tensors=pp_proxy_tensors,
                **kwargs,
            )
            candidate = (
                (
                    lambda: self._forward(
                        step.view, input_ids, positions, forward_batch, pp_proxy_tensors
                    )
                )
                if self.enabled
                else stock
            )
            return self.parity.run(step.rows, step.view, stock, candidate)
        if not self._engaged:
            logger.info(
                "NVIDIA megakernel: Qwen3.5 slot gdn_prefill_split_conv runs at %d rows",
                step.rows,
            )
            self._engaged = True
        return self._forward(
            step.view, input_ids, positions, forward_batch, pp_proxy_tensors
        )

    def plan_body(
        self,
        input_ids,
        forward_batch,
        input_embeds=None,
        pp_proxy_tensors=None,
        input_deepstack_embeds=None,
        **kwargs,
    ):
        if (
            not self._body_slots()
            or kwargs
            or (input_deepstack_embeds is not None and input_deepstack_embeds.numel())
            or any(item is not None for item in (forward_batch.mm_inputs or ()))
            or not forward_batch.forward_mode.is_extend_without_speculative()
        ):
            return None
        rows = forward_batch.input_ids if input_ids is None else input_ids
        return Step(mode="prefill", rows=rows.shape[0])

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
    ):
        from sglang.srt.model_executor.runner_utils.forward_batch import (
            set_forward_batch,
        )

        def body():
            with set_forward_batch(forward_batch):
                return self._body(
                    input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
                )

        if self.parity is not None and not torch.cuda.is_current_stream_capturing():
            stock = lambda: original(
                input_ids,
                positions,
                forward_batch,
                input_embeds=stage_input_copy(input_embeds),
                pp_proxy_tensors=stage_input_copy(pp_proxy_tensors),
                input_deepstack_embeds=input_deepstack_embeds,
            )
            # the row tiles are numeric: their comparison is logged, not enforced
            return self.parity.run(
                step.rows, step.view, stock, body, bitwise=not self.qk_rows
            )
        if not self._engaged:
            for name in self._body_slots():
                logger.info(
                    "NVIDIA megakernel: Qwen3.5 slot %s runs at %d rows",
                    name,
                    step.rows,
                )
            self._engaged = True
        return body()

    def linear_prefill_graph_seqs(self, backend):
        # the GDN layers' cores stay graph breaks (prefill_fi.core, or SGLang's own)
        return None

    def refill_prefill_graph(self, metadata, forward_batch):
        return None

    def _run_core(self, attn, mixed, a, b, output):
        prefill_fi.core(
            attn, mixed, a, b, output, self.binding, self._declined, self.snapshot
        )

    def _body(self, input_ids, positions, fb, input_embeds, pp_proxy_tensors):
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch

        model = self.lm.model
        if model.pp_group.is_first_rank:
            hidden = (
                model.embed_tokens(input_ids) if input_embeds is None else input_embeds
            )
            residual_batch.start(fb)
        else:
            hidden = model.layers[model.start_layer].attn_boundary.from_pp(
                pp_proxy_tensors, fb
            )
        norm = self.fp8.gdn_norm if self.fp8 is not None and self.fp8.norm else None
        gdn_step = lambda gdn, x: prefill_fi.step(gdn, x, self._core, norm)
        with self.fp8.body() if self.fp8 is not None else contextlib.nullcontext():
            hidden, aux = self._layers(
                hidden,
                positions,
                fb,
                gdn_step if self.front else None,
                self._attention_layer if self.qk_rows else None,
                moe=self.moe,
            )
        if not model.pp_group.is_last_rank:
            return residual_batch.to_pp(hidden, fb)
        hidden = residual_batch.final_norm(
            hidden,
            fb,
            model.norm,
            finalize_norm=model.flashinfer_mnnvl_cutedsl_fusion,
            skip_empty=True,
        )
        return (hidden, aux) if len(aux) else hidden

    def _layers(self, hidden, positions, fb, gdn_step, attn_layer=None, moe=None):
        from sglang.srt.eplb.expert_distribution import (
            get_global_expert_distribution_recorder,
        )
        from sglang.srt.layers.aux_hidden_states import AuxHiddenStateList

        model = self.lm.model
        aux = AuxHiddenStateList()
        # a MoE output owed its gated add (moe_prefill_add_norm), for the next layer
        owed = None
        for i in range(model.start_layer, model.end_layer):
            layer = model.layers[i]
            capture = aux if getattr(layer, "_is_layer_to_capture", False) else None
            with get_global_expert_distribution_recorder().with_current_layer(i):
                if moe is not None and i in moe.ids:
                    hidden, owed = self._moe_layer(
                        moe, i, layer, hidden, positions, fb, capture, gdn_step, owed
                    )
                    continue
                if owed is not None:
                    moe_prefill.complete(owed)
                    owed = None
                if i not in self._linear and attn_layer is not None:
                    hidden = attn_layer(i, layer, positions, hidden, fb, capture)
                    continue
                if self.fp8 is not None and i in self.fp8.gate:
                    hidden = self.fp8.attention_layer(
                        i, layer, positions, hidden, fb, capture
                    )
                    continue
                if i not in self._linear or gdn_step is None:
                    hidden = layer(
                        positions=positions,
                        hidden_states=hidden,
                        forward_batch=fb,
                        captured_last_layer_outputs=capture,
                    )
                    continue
                x = layer.attn_boundary.prepare(hidden, fb, capture_gathered=capture)
                part = gdn_step(layer.linear_attn, x)
                x = layer.attn_boundary.finish(part, fb)
                x = layer.ffn_boundary.prepare(x, fb)
                with layer.ffn_boundary.exit(fb) as ffn_exit:
                    out = layer.mlp(x, fb, defer_finalize=ffn_exit.defer_moe_finalize)
                hidden = ffn_exit.finish(out)
        if owed is not None:
            moe_prefill.complete(owed)
        return hidden, aux

    def _moe_layer(
        self, moe, i, layer, hidden, positions, fb, capture, gdn_step, owed=None
    ):
        x = None
        if owed is not None:
            if capture is None:
                x = moe_prefill.layer_input(layer, hidden, fb, owed, self.fp8)
            if x is None:
                moe_prefill.complete(owed)
        if x is None:
            x = layer.attn_boundary.prepare(hidden, fb, capture_gathered=capture)
        if i in self._linear:
            part = gdn_step(layer.linear_attn, x)
        else:
            # fused AR+quant hands down a (fp8, scale) / (bf16, fp8, scale) tuple
            hs = x[0] if isinstance(x, tuple) else x
            part = x
            if not fb.forward_mode.is_idle() and hs.shape[0] > 0:
                part = self._self_attention(i, layer, positions, x, fb)
        x = layer.attn_boundary.finish(part, fb)
        taken = moe_prefill.ffn_input(moe, i, layer, x, fb) if i in moe.nvfp4 else None
        x, pre = (
            taken if taken is not None else (layer.ffn_boundary.prepare(x, fb), None)
        )
        owing = None
        with layer.ffn_boundary.exit(fb) as ffn_exit:
            if ffn_exit.defer_moe_finalize or i not in moe.blocks:
                out = layer.mlp(x, fb, defer_finalize=ffn_exit.defer_moe_finalize)
            elif i in moe.add_norm and moe_prefill.exit_keeps_output(ffn_exit, fb):
                out, owing = moe_prefill.forward_owing(
                    layer.mlp, x, fb, pre, i in moe.noclone
                )
            else:
                out = moe_prefill.forward(layer.mlp, x, fb, pre, i in moe.noclone)
        return ffn_exit.finish(out), owing

    def _self_attention(self, i, layer, positions, x, fb):
        gate = self._fp8_gate(i)
        if self.qk_rows:
            return prefill_rows.self_attention(layer, positions, x, fb, gate)
        if gate is not None:
            return self.fp8.self_attention(i, layer, positions, x, fb)
        return layer.self_attention(
            positions=positions, hidden_states=x, forward_batch=fb
        )

    def _attention_layer(self, i, layer, positions, hidden, fb, captured=None):
        return prefill_rows.forward(
            layer, positions, hidden, fb, captured, gate=self._fp8_gate(i)
        )

    def _fp8_gate(self, i):
        if self.fp8 is None or i not in self.fp8.gate:
            return None
        return functools.partial(self.fp8.gate_input, i)

    def _forward(self, view, input_ids, positions, fb, pp_proxy_tensors):
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch

        lm, model = self.lm, self.lm.model
        if lm.is_mrope_enabled:
            positions = fb.mrope_positions
        if model.pp_group.is_first_rank:
            hidden = model.embed_tokens(input_ids)
            residual_batch.start(fb)
        else:
            hidden = model.layers[model.start_layer].attn_boundary.from_pp(
                pp_proxy_tensors, fb
            )
        hidden, aux = self._layers(
            hidden,
            positions,
            fb,
            lambda gdn, x: gdn_prefill.step(gdn, x, view, fb),
        )
        if not model.pp_group.is_last_rank:
            return residual_batch.to_pp(hidden, fb)
        hidden = residual_batch.final_norm(
            hidden,
            fb,
            model.norm,
            finalize_norm=model.flashinfer_mnnvl_cutedsl_fusion,
            skip_empty=True,
        )
        return lm.logits_processor(
            input_ids,
            hidden,
            lm.lm_head,
            fb,
            aux if lm.capture_aux_hidden_states and len(aux) else None,
        )


def stage_input_copy(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    tensors = getattr(value, "tensors", None)
    if not isinstance(tensors, dict):
        return value
    return type(value)({name: stage_input_copy(t) for name, t in tensors.items()})


class PrefillParity:
    """Test only: restore every state destination before the second forward.

    Unlike verify, prefill updates committed SSM state as well as conv history.
    PP inputs stay fixed for the local comparison; each PP rank checks its output.
    """

    @staticmethod
    def _outputs(out):
        if isinstance(out, torch.Tensor):
            return {"tensor": out.clone()}
        tensors = getattr(out, "tensors", None)
        if tensors is None:
            tensors = {
                name: getattr(out, name, None)
                for name in ("next_token_logits", "hidden_states", "last_hidden_states")
            }
        return {
            name: value.clone()
            for name, value in tensors.items()
            if isinstance(value, torch.Tensor)
        }

    @staticmethod
    def _indices(view, names):
        rows = [view.state_indices.to(torch.long)]
        if view.has_track:
            rows += [
                value.to(torch.long)
                for name in names
                if (value := getattr(view, name)) is not None
            ]
        indices = torch.cat(rows)
        return indices[indices >= 0].unique()

    def run(self, rows, view, original, candidate, bitwise=True):
        ci = self._indices(view, ("track_conv_dst",))
        si = self._indices(
            view, ("track_ssm_h_dst", "track_ssm_final_dst", "track_ssm_recompute_dst")
        )
        pools = [*((value, ci) for value in view.state.conv), (view.state.temporal, si)]
        before = [value[:, indices].clone() for value, indices in pools]
        want = self._outputs(original())
        state_want = [value[:, indices].clone() for value, indices in pools]
        for (value, indices), saved in zip(pools, before):
            value[:, indices] = saved
        got = candidate()
        outputs = self._outputs(got)
        assert outputs.keys() == want.keys()
        output_differ = sum(int(parity.differing(outputs[n], want[n])) for n in want)
        state_differ = sum(
            int(parity.differing(value[:, indices], saved))
            for (value, indices), saved in zip(pools, state_want)
        )
        logger.info(
            "NVIDIA megakernel prefill parity rows=%d output_differ=%d "
            "state_differ=%d outputs_compared=%d state_compared=%d bitwise=%d",
            rows,
            output_differ,
            state_differ,
            sum(value.numel() for value in want.values()),
            sum(value.numel() for value in state_want),
            bitwise,
        )
        if bitwise and (output_differ or state_differ):
            raise RuntimeError("GDN prefill output/state differs from stock")
        return got
