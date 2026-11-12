# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The routed experts of a MoE layer on the GEMM pair (``sm103/moe``), reading SGLang's
loaded MXFP4 experts in place, and the router's top-k on the fast gate kernel
(``gate_topk.cu``) where it reproduces the router bit for bit. Under expert parallelism
the pair holds the rank's window of experts and drops the slots routed elsewhere, as
SGLang's FlashInfer routed MoE does (``local_expert_offset``)."""

from __future__ import annotations

import logging

import msgspec
import torch

from sgl_nvidia_mega.deepseek_v41.runtime import bench

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
        "moe_ep_rank",
        "num_experts",
        "top_k",
        "quant_method",
    ),
    "srt/models/deepseek_v2.py": (
        "e_score_correction_bias_vl",
        "is_hash",
        "alt_stream",
        "num_fused_shared_experts",
        "layer_id",
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
    routed experts, returning SGLang's deferred-finalize output; with slot shared_ll,
    its shared expert on the low-latency GEMM (``shared``, else None); with slot
    gate_route_early, the router GEMM that releases the top-k once its weight loads are issued (``router``,
    an ``EarlyRouter``, else None)."""

    def __init__(
        self, mlp: torch.nn.Module, gemm, *, gate: bool, shared=None, router=None
    ):
        from sglang.srt.layers.moe.topk import _RENORMALIZE_SUM_EPSILON

        self.mlp, self.gemm, self.gate, self.shared = mlp, gemm, gate, shared
        self.router = router
        config = mlp.topk.topk_config
        self.scale = (
            float(config.routed_scaling_factor)
            if config.apply_routed_scaling_factor_on_output
            and config.routed_scaling_factor is not None
            else 1.0
        )
        self.bias = config.correction_bias_for_dtype(torch.float32) if gate else None
        self.eps = _RENORMALIZE_SUM_EPSILON
        # benchmark only (runtime/bench.py): the layer's fixed routing, else None
        self.fixed = bench.bind_fixed(mlp, gemm, self.bias, self.scale)
        # benchmark only: the fast gate's ids in a buffer the step stamps can read
        # after the graph's replay (None without them)
        self.ids_out = bench.ids_buffer(gemm.max_rows, gemm.top_k, gemm.clamp.device)

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
        if self.fixed is not None:
            router_logits = self.fixed.router_logits(router_logits)
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

    def early_router(self, x: torch.Tensor, input_ids_global, use_vision_topk: bool):
        """Slot gate_route_early serves this call: the layer's early router (else None),
        only where the one-launch top-k + route (gate_route) follows it, which reads the
        logits after its PDL wait."""
        if (
            self.router is None
            or not self.gate
            or not self.router.serves(x)
            or not self.gemm.serves_gate_route(
                x.shape[0], input_ids_global if use_vision_topk else None
            )
        ):
            return None
        return self.router

    def quant_in_gate(
        self, x: torch.Tensor, input_ids_global, use_vision_topk: bool
    ) -> bool:
        """Slot gate_route_quant serves this call: the fast top-k's launch quantizes the
        routed input itself, so the caller issues no routed-input quant."""
        from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
            routed_hidden_size,
        )

        return (
            self.gate
            and routed_hidden_size(self.mlp.experts) == x.shape[-1]
            and self.gemm.serves_gate_route_quant(
                x.shape[0], input_ids_global if use_vision_topk else None, x
            )
        )

    def run(
        self,
        x: torch.Tensor,
        topk_output,
        pre_quant_input,
        quant_in_gate=False,
        early=False,
    ):
        """``quant_in_gate`` (``quant_in_gate``'s answer, no ``pre_quant_input``): the
        top-k launch writes the routed input's MXFP8 (slot gate_route_quant); ``early``:
        the early router (``early_router``) launched the logits (slot gate_route_early)."""
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            FlashInferTrtllmDeferredFinalizeOutput,
        )
        from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
            routed_hidden_size,
        )

        experts = self.mlp.experts
        ready = None
        if quant_in_gate:
            if pre_quant_input is not None or not isinstance(topk_output, _GateTopK):
                raise RuntimeError(
                    "NVIDIA megakernel: slot gate_route_quant takes the fast top-k "
                    "without a pre-quantized input"
                )
            x_q = x_s = None
        elif pre_quant_input is not None:
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
            rows, weights, permuted, ids = self.gemm.gate_topk_and_gemms(
                x_q,
                x_s,
                topk_output.router_logits,
                topk_output.bias,
                topk_output.scale,
                topk_output.num_valid,
                alt=topk_output.alt,
                renormalize_epsilon=topk_output.eps,
                **({"ids_out": self.ids_out} if self.ids_out is not None else {}),
                **({"x": x} if quant_in_gate else {}),
                early=early,
            )
            persistent = self.ids_out is not None
        else:
            ids, w = topk_output.topk_ids, topk_output.topk_weights
            # SGLang's top-k ids live in the graph's pool: the stamps read only a
            # fixed table there
            persistent = self.fixed is not None
            if self.fixed is not None:
                ids, w = self.fixed.topk(ids, w)
            rows, weights, permuted = self.gemm(x_q, x_s, ids, w)
        if persistent:
            bench.note_ids(ids, self.mlp.layer_id, self.gemm.router_experts)
        return FlashInferTrtllmDeferredFinalizeOutput(
            gemm2_out=rows,
            expert_weights=weights,
            # flat [rows * top_k] slot -> row map, as FlashInfer's routed MoE returns it
            expanded_idx_to_permuted_idx=permuted.view(-1),
            top_k=self.gemm.top_k,
        )


class Bound(msgspec.Struct):
    """``moe.build``'s result: the slot of each layer whose pair built, by layer id,
    and why none was built ("" when built); why no shared expert took slot shared_ll;
    why a layer's router did not take slot gate_route_early ("" when every one did)."""

    slots: dict = msgspec.field(default_factory=dict)
    reason: str = ""
    shared_reason: str = ""
    router_reason: str = ""


def shared_topk_slots(config) -> int:
    """The trailing top-k columns SGLang's router fills with fused shared experts:
    ``TopKConfig.num_shared_topk_slots``, named ``num_fused_shared_experts`` before
    SGLang split its two meanings; read by name so both serve."""
    slots = getattr(config, "num_shared_topk_slots", None)
    return getattr(config, "num_fused_shared_experts", 0) if slots is None else slots


def gate_applies(mlp: torch.nn.Module) -> bool:
    """The DSV4.1 router gate_topk.cu reproduces bit for bit."""
    from sgl_nvidia_mega.deepseek_v41.sm103.moe.routed_expert_gemm import (
        warp_bias_layout_ok,
    )

    config = mlp.topk.topk_config
    bias_vl = getattr(mlp.gate, "e_score_correction_bias_vl", None)
    return (
        not getattr(mlp, "is_hash", False)
        and config.scoring_func == "sqrtsoftplus"
        and getattr(config, "sqrtsoftplus_log1p", False)
        and config.renormalize
        and shared_topk_slots(config) == 0
        and not config.use_grouped_topk
        and config.custom_routing_function is None
        and config.correction_bias is not None
        and (bias_vl is None or bias_vl.dtype == torch.float32)
        # SGLang's warp router takes these biases (else it falls back to Triton)
        and warp_bias_layout_ok(config.correction_bias, bias_vl)
    )


def expert_offset(experts: torch.nn.Module) -> int:
    """The router id of the rank's first local expert, as SGLang's FlashInfer routed
    MoE takes it (``local_expert_offset``); 0 without expert parallelism."""
    return experts.moe_ep_rank * experts.num_local_experts


def build(
    model: torch.nn.Module,
    group,
    *,
    gate_topk: bool,
    shared_ll=False,
    widths=(),
    gemm2_pipe: bool = False,
    gemm2_registers: int = 0,
) -> Bound:
    """A routed-expert GEMM pair per MoE layer; every rank agrees, or none is built."""
    m = model.model
    layers = [m.layers[i] for i in range(m.start_layer, m.end_layer)]
    return build_layers(
        layers,
        group,
        gate_topk=gate_topk,
        shared_ll=shared_ll,
        widths=widths,
        gemm2_pipe=gemm2_pipe,
        gemm2_registers=gemm2_registers,
    )


def build_layers(
    layers,
    group,
    *,
    gate_topk: bool,
    shared_ll: bool = False,
    widths=(),
    gate_route: bool | None = None,
    gate_route_quant: bool | None = None,
    gate_route_early: bool | None = None,
    gemm2_pipe: bool = False,
    gemm2_registers: int = 0,
) -> Bound:
    """``build`` over a list of decoder layers (the target's, or the draft's stages),
    keyed by ``layer_id``. With ``shared_ll``, each built layer's shared expert on the
    low-latency GEMM where it applies, compiled for ``widths``; every rank agrees on
    those layers too, or none takes it. ``gate_route`` (default: slot gate_route
    requested): the fast top-k and the route record in one launch at 1-8 rows;
    ``gate_route_quant`` (default: slot gate_route_quant requested): that launch also
    quantizes the routed input; ``gate_route_early`` (default: slot gate_route_early
    requested): the router GEMM ahead of that launch releases it once its weight loads are issued."""
    from sgl_nvidia_mega.deepseek_v41.runtime import SLOTS

    if gate_route is None:
        gate_route = "gate_route" in SLOTS
    if gate_route_quant is None:
        gate_route_quant = "gate_route_quant" in SLOTS
    if gate_route_early is None:
        gate_route_early = "gate_route_early" in SLOTS
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
                router_experts=experts.num_experts,
                expert_offset=expert_offset(experts),
                gate_route=gate_route,
                gate_route_quant=gate_route_quant,
                gemm2_pipe=gemm2_pipe,
                gemm2_registers=gemm2_registers,
            )
            gate = gate_topk and gemm.gate_topk_kernel is not None and gate_applies(mlp)
            slots[i] = MoeSlot(mlp, gemm, gate=gate)
    except Exception as e:  # gathered, then the slot is off on every rank
        reason, slots = f"{type(e).__name__}: {e}", {}
    router_reason = (
        _bind_router(slots) if gate_route_early and gate_route else "not requested"
    )
    shared_reason = _bind_shared(slots, widths) if shared_ll else "not requested"
    shared = sorted(i for i, s in slots.items() if s.shared is not None)
    gathered = [None] * group.world_size
    torch.distributed.all_gather_object(
        gathered, (reason, sorted(slots), shared), group=group.cpu_group
    )
    failed = [f"rank {r}: {e}" for r, (e, _, _) in enumerate(gathered) if e]
    if failed or any(s != sorted(slots) for _, s, _ in gathered):
        reason = (
            "; ".join(failed) or f"the ranks disagree on the MoE layers: {gathered}"
        )
        slots = {}
    elif any(s != shared for _, _, s in gathered):
        shared_reason = f"the ranks disagree on the shared_ll layers: {gathered}"
        for slot in slots.values():
            slot.shared = None
    return Bound(
        slots=slots,
        reason=reason,
        shared_reason=shared_reason,
        router_reason=router_reason,
    )


def _bind_router(slots: dict) -> str:
    """Slot gate_route_early on each built layer whose router takes it (one early router
    per gate weight); why some did not ("" when all did). Bitwise either way, so the
    ranks need not agree."""
    from sgl_nvidia_mega.deepseek_v41.sm103.moe.router_mma import (
        EarlyRouter,
        gate_route_release,
    )

    skipped = {}
    for i, slot in slots.items():
        if not slot.gate or slot.gemm.gate_route_kernel is None:
            skipped[i] = "no gate_route launch follows the router"
            continue
        try:
            slot.router, why = EarlyRouter.build(slot.mlp.gate)
            if slot.router is not None:
                slot.gemm.bind_early_release(gate_route_release())
        except Exception as e:
            slot.router, why = None, f"{type(e).__name__}: {e}"
        if why:
            skipped[i] = why
    return "; ".join(f"layer {i}: {why}" for i, why in sorted(skipped.items()))


def _bind_shared(slots: dict, widths) -> str:
    """Slot shared_ll on each built layer whose shared expert takes it; why some did not
    ("" when all did). A failed build leaves every layer to SGLang's shared expert."""
    from sgl_nvidia_mega.deepseek_v41.moe import shared_ll
    from sgl_nvidia_mega.deepseek_v41.plan import ROUTED_MAX_ROWS

    skipped = {}
    try:
        for i, slot in slots.items():
            why = shared_ll.why_not(slot.mlp)
            if why:
                skipped[i] = why
                continue
            slot.shared = shared_ll.SharedLL(slot.mlp, ROUTED_MAX_ROWS)
            slot.shared.precompile(widths)
    except Exception as e:
        for slot in slots.values():
            slot.shared = None
        return f"{type(e).__name__}: {e}"
    return "; ".join(f"layer {i}: {why}" for i, why in sorted(skipped.items()))
