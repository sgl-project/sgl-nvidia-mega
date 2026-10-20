# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The routed experts of a MoE layer on the GEMM pair (``sm103/moe``), reading SGLang's
loaded MXFP4 experts in place, and the router's top-k on the fast gate kernel
(``gate_topk.cu``) where it reproduces the router bit for bit."""

from __future__ import annotations

import logging

import msgspec
import torch

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/layers/moe/topk.py": (
        "_RENORMALIZE_SUM_EPSILON",
        "topk_config",
        "routed_scaling_factor",
        "apply_routed_scaling_factor_on_output",
        "correction_bias_for_dtype",
        "scoring_func",
        "renormalize",
        "num_shared_topk_slots",
        "use_grouped_topk",
        "custom_routing_function",
        "correction_bias",
        "topk_ids",
        "topk_weights",
    ),
    "srt/layers/moe/moe_runner/flashinfer_trtllm.py": (
        "FlashInferTrtllmDeferredFinalizeOutput",
        "gemm2_out",
        "expert_weights",
        "expanded_idx_to_permuted_idx",
        "top_k",
    ),
    "srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py": (
        "Mxfp4FlashinferTrtllmMoEMethod",
        "routed_hidden_size",
        "quantize_routed_input",
        "_gemm1_clamp_limit_tensor",
        "w13_weight_scale_inv",
        '"w2_weight_scale_inv"',
    ),
    "srt/layers/moe/fused_moe_triton/layer.py": (
        "num_local_experts",
        "num_experts",
        "top_k",
        "quant_method",
    ),
    "srt/models/deepseek_v2.py": (
        "e_score_correction_bias_vl",
        "is_hash",
        "alt_stream",
        "num_fused_shared_experts",
    ),
    "srt/configs/deepseek_v4.py": ("image_token_id",),
    "srt/models/deepseek_v4.py": ("mlp", "layers", "start_layer", "end_layer"),
}


class _GateTopK(msgspec.Struct):
    """The router's logits and routing inputs, for gate_topk.cu inside the routed call."""

    router_logits: torch.Tensor
    bias: torch.Tensor
    scale: float
    num_valid: object  # a 1-element int32 tensor, or None when every row is valid
    alt: object
    eps: float


class MoeSlot:
    """One MoE layer's routed-expert pair: the router's top-k (fast gate) and the
    routed experts, returning SGLang's deferred-finalize output."""

    def __init__(self, mlp: torch.nn.Module, gemm, *, gate: bool):
        from sglang.srt.layers.moe.topk import _RENORMALIZE_SUM_EPSILON

        self.mlp, self.gemm, self.gate = mlp, gemm, gate
        config = mlp.topk.topk_config
        self.scale = (
            float(config.routed_scaling_factor)
            if config.apply_routed_scaling_factor_on_output
            and config.routed_scaling_factor is not None
            else 1.0
        )
        self.bias = config.correction_bias_for_dtype(torch.float32) if gate else None
        self.eps = _RENORMALIZE_SUM_EPSILON

    def select(
        self,
        router_logits: torch.Tensor,
        input_ids_global: torch.Tensor | None,
        num_token_non_padded: torch.Tensor | None,
        use_vision_topk: bool,
    ) -> _GateTopK:
        if not self.gemm.supports_gate_topk(router_logits, self.bias):
            raise RuntimeError(
                f"NVIDIA megakernel: the fast gate cannot take router logits "
                f"{router_logits.dtype} {tuple(router_logits.shape)}"
            )
        alt, eps = None, 0.0
        if use_vision_topk:
            # vision routing: image tokens rank with the alternate bias
            alt = (
                self.mlp.gate.e_score_correction_bias_vl,
                input_ids_global,
                int(self.mlp.config.image_token_id),
            )
            eps = self.eps
        # None (no padded rows) launches no fill kernel, which would break the PDL chain
        return _GateTopK(
            router_logits, self.bias, self.scale, num_token_non_padded, alt, eps
        )

    def run(self, x: torch.Tensor, topk_output, pre_quant_input) -> object:
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            FlashInferTrtllmDeferredFinalizeOutput,
        )
        from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
            routed_hidden_size,
        )

        experts = self.mlp.experts
        ready = None
        if pre_quant_input is not None:
            x_q, x_s, ready = pre_quant_input
        else:
            x_q, x_s = experts.quant_method.quantize_routed_input(
                x, routed_hidden_size(experts)
            )
        if ready is not None:
            # before the top-k, as SGLang joins it: between the route kernel and GEMM1
            # the join would cost GEMM1 its PDL early launch
            torch.cuda.current_stream().wait_event(ready)
        if isinstance(topk_output, _GateTopK):
            rows, weights, permuted, _ = self.gemm.gate_topk_and_gemms(
                x_q,
                x_s,
                topk_output.router_logits,
                topk_output.bias,
                topk_output.scale,
                topk_output.num_valid,
                alt=topk_output.alt,
                renormalize_epsilon=topk_output.eps,
            )
        else:
            rows, weights, permuted = self.gemm(
                x_q, x_s, topk_output.topk_ids, topk_output.topk_weights
            )
        return FlashInferTrtllmDeferredFinalizeOutput(
            gemm2_out=rows,
            expert_weights=weights,
            # flat [rows * top_k] slot -> row map, as FlashInfer's routed MoE returns it
            expanded_idx_to_permuted_idx=permuted.view(-1),
            top_k=self.gemm.top_k,
        )


class Bound(msgspec.Struct):
    """``moe.build``'s result: the slot of each layer whose pair built, by layer id,
    and why none was built ("" when built)."""

    slots: dict = msgspec.field(default_factory=dict)
    reason: str = ""


def gate_applies(mlp: torch.nn.Module) -> bool:
    """The DSV4.1 router gate_topk.cu reproduces bit for bit."""
    config = mlp.topk.topk_config
    bias_vl = getattr(mlp.gate, "e_score_correction_bias_vl", None)
    return (
        not getattr(mlp, "is_hash", False)
        and config.scoring_func == "sqrtsoftplus"
        and getattr(config, "sqrtsoftplus_log1p", False)
        and config.renormalize
        and config.num_shared_topk_slots == 0
        and not config.use_grouped_topk
        and config.custom_routing_function is None
        and config.correction_bias is not None
        and (bias_vl is None or bias_vl.dtype == torch.float32)
    )


def build(model: torch.nn.Module, group, *, gate_topk: bool) -> Bound:
    """A routed-expert GEMM pair per MoE layer; every rank agrees, or none is built."""
    m = model.model
    layers = [m.layers[i] for i in range(m.start_layer, m.end_layer)]
    return build_layers(layers, group, gate_topk=gate_topk)


def build_layers(layers, group, *, gate_topk: bool) -> Bound:
    """``build`` over a list of decoder layers (the target's, or the draft's stages),
    keyed by ``layer_id``."""
    from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
        Mxfp4FlashinferTrtllmMoEMethod,
    )

    from sgl_nvidia_mega.deepseek_v41.sm103.moe.routed_expert_gemm import (
        SERVED_MAX_ROWS,
        RoutedExpertGemm,
    )

    reason, slots = "", {}
    try:
        for layer in layers:
            i = layer.layer_id
            mlp = layer.mlp
            experts = mlp.experts
            method = experts.quant_method
            if not isinstance(method, Mxfp4FlashinferTrtllmMoEMethod):
                raise ValueError(f"layer {i}: experts on {type(method).__name__}")
            if experts.num_local_experts != experts.num_experts:
                raise ValueError(f"layer {i}: expert parallelism")
            if mlp.alt_stream is None or mlp.num_fused_shared_experts:
                continue  # SGLang's dual-stream order does not apply
            gemm = RoutedExpertGemm(
                experts.w13_weight,
                experts.w13_weight_scale_inv,
                experts.w2_weight,
                experts.w2_weight_scale_inv,
                method._gemm1_clamp_limit_tensor,
                top_k=experts.top_k,
                max_rows=SERVED_MAX_ROWS,
                gate_topk=gate_topk,
            )
            gate = gate_topk and gemm.gate_topk_kernel is not None and gate_applies(mlp)
            slots[i] = MoeSlot(mlp, gemm, gate=gate)
    except Exception as e:  # gathered, then the slot is off on every rank
        reason, slots = f"{type(e).__name__}: {e}", {}
    gathered = [None] * group.world_size
    torch.distributed.all_gather_object(
        gathered, (reason, sorted(slots)), group=group.cpu_group
    )
    failed = [f"rank {r}: {e}" for r, (e, _) in enumerate(gathered) if e]
    if failed or any(s != sorted(slots) for _, s in gathered):
        reason = (
            "; ".join(failed) or f"the ranks disagree on the MoE layers: {gathered}"
        )
        slots = {}
    return Bound(slots=slots, reason=reason)
