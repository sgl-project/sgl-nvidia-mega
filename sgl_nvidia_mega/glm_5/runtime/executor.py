# SPDX-License-Identifier: Apache-2.0
"""GLM5 layer composition using the current SGLang stage/residual interfaces.

The framework keeps its embedding, layer loop, top-k/aux publication, final
normalization and logits. Per-layer adapters are active only inside a covered
model forward; all other forwards call the original layer unchanged.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass

import torch
from sglang.srt.layers.layer_boundary import get_attn_tp_context
from sglang.srt.layers.layer_boundary.residual.batch import (
    clear,
    fold,
    set_written,
)
from sglang.srt.mem_cache.memory_pool import unwrap_write_loc

try:
    from sglang.kernels.plugin_kernels.nvidia.spec import Step
except ModuleNotFoundError:
    from sglang.srt.megakernels.spec import Step

from sgl_nvidia_mega.glm_5 import HIDDEN, SELECTED, TOP_K
from sgl_nvidia_mega.glm_5.plan import ExecutionPlan
from sgl_nvidia_mega.glm_5.runtime.state import KernelState
from sgl_nvidia_mega.glm_5.runtime.weights import (
    mla_weights_from_layer,
    moe_weights_from_layer,
    norm_dtype,
)
from sgl_nvidia_mega.glm_5.sm103 import mla, moe

SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": (
        "Step",
        "widths",
        "tokens_per_request",
        "binding",
    ),
    "srt/layers/layer_boundary/__init__.py": ("get_attn_tp_context",),
    "srt/layers/layer_boundary/residual/batch.py": ("clear", "fold", "set_written"),
    "srt/layers/layer_boundary/adapters/attention.py": (
        "get_attn_tp_context",
        "clear_attn_inputs",
    ),
    "srt/mem_cache/memory_pool.py": ("unwrap_write_loc",),
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/binding.py": (
        "metadata",
        "kv_pool",
        "get_key_buffer",
        "use_fused_topk",
    ),
    "srt/models/deepseek_v2.py": (
        "attn_boundary",
        "self_attn",
        "is_layer_sparse",
        "maybe_use_decode_attn_tp",
        "forward(prev_topk_indices)",
        "forward(captured_last_layer_outputs)",
        "layer_id",
        "skip_topk",
        "next_skip_topk",
        "rotary_emb",
    ),
    "srt/layers/aux_hidden_states.py": ("capture",),
    "srt/layers/attention/dsa_backend.py": ("dsa_cache_seqlens_int32",),
    "srt/distributed/parallel_state.py": ("get_attn_tp_group", "get_moe_tp_group"),
    "srt/layers/layer_boundary/stage.py": (
        "prepare",
        "finish",
        "input_on_attn_tp_slices",
    ),
    "srt/layers/rotary_embedding/base.py": ("cos_sin_cache",),
}


@dataclass
class _Buffers:
    plan: ExecutionPlan
    mla_local: torch.Tensor
    mla_output: torch.Tensor
    moe_output: torch.Tensor
    selected_indices: torch.Tensor
    selected_probs: torch.Tensor

    @classmethod
    def allocate(cls, plan, device):
        shape = plan.shape
        return cls(
            plan,
            *(
                torch.empty(shape, device=device, dtype=torch.bfloat16)
                for _ in range(3)
            ),
            torch.empty((*shape[:2], TOP_K), device=device, dtype=torch.int32),
            torch.empty((*shape[:2], TOP_K), device=device, dtype=torch.float32),
        )


class _Layer:
    def __init__(self, owner, layer):
        self.owner, self.layer = owner, layer
        self.original = layer.forward
        self.moe_weights = moe_weights_from_layer(layer)
        self.mla_weights = (
            mla_weights_from_layer(layer) if layer.self_attn.skip_topk else None
        )
        self.moe = {}
        self.mla = {}
        for rows, kernel in owner.moe_kernels.items():
            self.moe[rows] = kernel.bind(self.moe_weights, state=owner.moe_states[rows])
        layer.forward = self.forward

    def forward(
        self,
        positions,
        hidden_states,
        forward_batch,
        zero_allocator,
        gemm_output_zero_allocator=None,
        llama_4_scaling=None,
        prev_topk_indices=None,
        captured_last_layer_outputs=None,
        next_full_attention_layer_id=None,
    ):
        active = self.owner.active.get()
        if active is None or active[1] is not forward_batch:
            return self.original(
                positions,
                hidden_states,
                forward_batch,
                zero_allocator,
                gemm_output_zero_allocator,
                llama_4_scaling,
                prev_topk_indices,
                captured_last_layer_outputs,
                next_full_attention_layer_id,
            )
        step, _ = active
        buffers = self.owner.buffers[step.rows]
        if self.mla_weights is None:
            hidden_states, topk = self._native_attention(
                positions,
                hidden_states,
                forward_batch,
                zero_allocator,
                llama_4_scaling,
                prev_topk_indices,
                captured_last_layer_outputs,
            )
        else:
            hidden_states = fold(hidden_states, forward_batch)
            if captured_last_layer_outputs is not None:
                captured_last_layer_outputs.capture(hidden_states, owned=False)
            hidden_states = self._mla(
                step,
                hidden_states,
                positions,
                forward_batch,
                prev_topk_indices,
                buffers,
            )
            hidden_states = set_written(hidden_states, forward_batch)
            topk = prev_topk_indices if self.layer.self_attn.next_skip_topk else None
        hidden_states = fold(hidden_states, forward_batch)
        x = hidden_states.view(buffers.plan.shape)
        self.moe[step.rows](
            x,
            residual=x,
            final_output=buffers.moe_output,
            selected_indices=buffers.selected_indices,
            selected_probs=buffers.selected_probs,
        )
        output = buffers.moe_output.view(step.rows, HIDDEN)
        return set_written(output, forward_batch), topk

    def _native_attention(
        self,
        positions,
        hidden,
        fb,
        zero_allocator,
        llama_4_scaling,
        prev_topk,
        capture,
    ):
        layer = self.layer
        hidden = layer.attn_boundary.prepare(
            hidden,
            fb,
            capture_gathered=capture,
            quant_format=layer._resolve_gfx95_quant_format(),
        )
        with layer.self_attn.maybe_use_decode_attn_tp(fb):
            result = layer.self_attn(
                positions=positions,
                hidden_states=hidden,
                forward_batch=fb,
                zero_allocator=zero_allocator,
                llama_4_scaling=llama_4_scaling,
                input_on_attn_tp_slices=layer.attn_boundary.input_on_attn_tp_slices,
                prev_topk_indices=prev_topk,
            )
        hidden, topk = result if isinstance(result, tuple) else (result, None)
        get_attn_tp_context().clear_attn_inputs()
        return layer.attn_boundary.finish(hidden, fb), topk

    def _mla(self, step, hidden, positions, fb, topk, buffers):
        if not step.view.use_fused_topk:
            raise ValueError("GLM5 megamla requires native fused physical-slot top-k")
        if topk is None or tuple(topk.shape) != (step.rows, SELECTED):
            raise ValueError("GLM5 shared-indexer layer has no matching native top-k")
        counts = step.view.metadata.dsa_cache_seqlens_int32
        if tuple(counts.shape) != (step.rows,):
            raise ValueError("GLM5 DSA selected-count rows differ from verify rows")
        if step.rows not in self.mla:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("bind GLM5 MLA cache during warmup before capture")
            cache = step.view.kv_pool.get_key_buffer(self.layer.layer_id)
            if cache.ndim != 3 or tuple(cache.shape[1:]) != (1, 576):
                raise ValueError("GLM5 requires native raw [slots,1,576] MLA cache")
            self.mla[step.rows] = self.owner.mla_kernels[step.rows].bind(
                self.mla_weights,
                kv_cache=cache[:, 0, :512],
                pe_cache=cache[:, 0, 512:],
                state=self.owner.mla_states[step.rows],
            )
        x = hidden.view(buffers.plan.shape)
        self.mla[step.rows](
            x,
            selected_indices=topk,
            selected_counts=counts,
            cache_positions=unwrap_write_loc(fb.out_cache_loc)[0],
            rope=self.layer.self_attn.rotary_emb.cos_sin_cache,
            rope_positions=positions,
            residual=x,
            local_output=buffers.mla_local,
            final_output=buffers.mla_output,
        )
        return buffers.mla_output.view(step.rows, HIDDEN)


class MegakernelDecode:
    draft_sampler = None

    def __init__(self, model, context):
        from sglang.srt.distributed import get_attn_tp_group, get_moe_tp_group

        self.model, self.binding = model, context.binding
        self.tokens_per_request = context.tokens_per_request
        self.active = ContextVar("glm5_covered_step", default=None)
        self.widths = tuple(
            width
            for width in context.widths
            if width in (4, 8) and width % self.tokens_per_request == 0
        )
        layers = [
            layer
            for layer in model.model.layers
            if bool(getattr(layer, "is_layer_sparse", False))
        ]
        if not layers or not any(layer.self_attn.skip_topk for layer in layers):
            raise ValueError("GLM5 requires sparse producer/shared-indexer layers")
        tp = get_attn_tp_group().world_size
        device = layers[0].post_attention_layernorm.weight.device
        mla_norm = {
            norm_dtype(mla_weights_from_layer(layer))
            for layer in layers
            if layer.self_attn.skip_topk
        }
        moe_norm = {norm_dtype(moe_weights_from_layer(layer)) for layer in layers}
        if len(mla_norm) != 1 or len(moe_norm) != 1:
            raise ValueError("GLM5 layers must share each operator's norm dtype")
        self.mla_kernels, self.moe_kernels = {}, {}
        self.mla_states, self.moe_states, self.buffers = {}, {}, {}
        for rows in self.widths:
            plan = ExecutionPlan(
                self.tokens_per_request, tp, rows // self.tokens_per_request
            )
            self.buffers[rows] = _Buffers.allocate(plan, device)
            self.mla_kernels[rows] = mla.load(
                draft_tokens=plan.draft_tokens,
                tp_size=tp,
                batch_size=plan.batch_size,
                norm_dtype=next(iter(mla_norm)),
            )
            self.moe_kernels[rows] = moe.load(
                draft_tokens=plan.draft_tokens,
                tp_size=tp,
                batch_size=plan.batch_size,
                norm_dtype=next(iter(moe_norm)),
            )
            self.mla_states[rows] = KernelState.allocate(
                self.mla_kernels[rows].metadata,
                device,
                group=get_attn_tp_group().cpu_group,
            )
            self.moe_states[rows] = KernelState.allocate(
                self.moe_kernels[rows].metadata,
                device,
                group=get_moe_tp_group().cpu_group,
            )
        self.layers = [_Layer(self, layer) for layer in layers]
        self.mla_layer_count = sum(
            layer.mla_weights is not None for layer in self.layers
        )

    def summary(self):
        return (
            f"GLM5 native FP8; {self.mla_layer_count} shared-indexer MLA layers, "
            f"{len(self.layers)} MoE layers; {self.tokens_per_request} tokens/request"
        )

    def plan(
        self,
        input_ids,
        forward_batch,
        input_embeds=None,
        pp_proxy_tensors=None,
    ):
        rows = int(input_ids.shape[0])
        if (
            pp_proxy_tensors is not None
            or not forward_batch.forward_mode.is_target_verify()
            or rows not in self.widths
            or rows != int(forward_batch.batch_size) * self.tokens_per_request
        ):
            return None
        return Step(mode="target_verify", rows=rows)

    def forward(
        self,
        step,
        original,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        pp_proxy_tensors=None,
    ):
        token = self.active.set((step, forward_batch))
        try:
            return original(
                input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
            )
        except BaseException:
            get_attn_tp_context().clear_attn_inputs()
            clear(forward_batch)
            raise
        finally:
            self.active.reset(token)


def build(model, hf_config, context):
    return MegakernelDecode(model, context)
