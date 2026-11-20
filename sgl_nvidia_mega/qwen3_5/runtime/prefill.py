# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0): Qwen3VLForConditionalGeneration.forward,
# Qwen3_5ForCausalLM.forward and Qwen3_5LinearDecoderLayer.forward.
"""The text prefill of one PP stage; native layers except the GDN conv layout."""

from __future__ import annotations

import logging

import torch
from sglang.kernels.plugin_kernels.nvidia.spec import Step

from sgl_nvidia_mega.qwen3_5.gdn import prefill as gdn_prefill
from sgl_nvidia_mega.qwen3_5.runtime import parity

logger = logging.getLogger(__name__)

SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py": ("Step",),
    "srt/models/qwen3_5.py": (
        "Qwen3_5LinearDecoderLayer",
        "residual_batch",
        "layers",
        "start_layer",
        "end_layer",
        "pp_group",
        "embed_tokens",
        "attn_boundary",
        "ffn_boundary",
        "linear_attn",
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
}


class Qwen35Prefill:
    draft_sampler = draft_loop = draft_extend = begin_draft_round = None
    graph_variants = None
    speculative_runtime_hooks = None

    def __init__(self, model, ctx, enabled):
        from sglang.srt.models.qwen3_5 import Qwen3_5LinearDecoderLayer

        self.lm = model
        self.enabled = enabled
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

    def summary(self):
        return "prefill TP1/PP4; gdn_prefill_split_conv " + (
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
            (not self.enabled and self.parity is None)
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

    def _forward(self, view, input_ids, positions, fb, pp_proxy_tensors):
        from sglang.srt.eplb.expert_distribution import (
            get_global_expert_distribution_recorder,
        )
        from sglang.srt.layers.aux_hidden_states import AuxHiddenStateList
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
        aux = AuxHiddenStateList()
        for i in range(model.start_layer, model.end_layer):
            layer = model.layers[i]
            capture = aux if getattr(layer, "_is_layer_to_capture", False) else None
            with get_global_expert_distribution_recorder().with_current_layer(i):
                if i not in self._linear:
                    hidden = layer(
                        positions=positions,
                        hidden_states=hidden,
                        forward_batch=fb,
                        captured_last_layer_outputs=capture,
                    )
                    continue
                x = layer.attn_boundary.prepare(hidden, fb, capture_gathered=capture)
                part = gdn_prefill.step(layer.linear_attn, x, view, fb)
                x = layer.attn_boundary.finish(part, fb)
                x = layer.ffn_boundary.prepare(x, fb)
                with layer.ffn_boundary.exit(fb) as ffn_exit:
                    out = layer.mlp(x, fb, defer_finalize=ffn_exit.defer_moe_finalize)
                hidden = ffn_exit.finish(out)
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

    def run(self, rows, view, original, candidate):
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
            "state_differ=%d outputs_compared=%d state_compared=%d",
            rows,
            output_differ,
            state_differ,
            sum(value.numel() for value in want.values()),
            sum(value.numel() for value in state_want),
        )
        if output_differ or state_differ:
            raise RuntimeError("GDN prefill output/state differs from stock")
        return got
