# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Portions Copyright 2023-2024 SGLang Team (Apache-2.0)
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (fused_experts_none_to_flashinfer_trtllm_fp4: the
# trtllm_fp4_block_scale_moe call), python/sglang/srt/layers/quantization/modelopt_quant.py
# (ModelOptNvFp4FusedMoEMethod.apply: the FlashInferTrtllmFp4MoeQuantInfo fields).
"""``ll.moe``: the MLP step of the GLM low-latency verify step at TP4 (also built at
TP8), for ``ll.hops``.

MoE layers run the stock pieces in the stock order, but leave the finalize to the
next hop (contract token ``mlp:deferred_finalize``):

1. the router GEMM (``ll.linear("router")``: FP32 logits; WP3's registered GEMM when
   present, else ``MoEGate.forward``);
2. the routed experts: FlashInfer's public ``trtllm_fp4_block_scale_moe`` with the
   layer's processed NVFP4 weights and ``do_finalize=False`` (a copy of SGLang's
   ``fused_experts_none_to_flashinfer_trtllm_fp4`` call for this configuration;
   ``test_glm5_3_ll_moe.py`` checks it bitwise against SGLang's runner). The FC1
   input is the NVFP4 quant the layer's post-attention hop wrote (P10,
   ``step.nvfp4_input``), or ``quantize_hidden_states_fp4`` of the input;
3. the shared expert on the layer's alternate stream (or, with the option unit
   ``ll.moe.alt_stream_priority``, on a stream of the greatest priority created at
   build), joined before returning;

and return ``hops.DeferredFinalize``, which the next hop finalizes inside its
all-reduce (P9).

Dense layers (0-2): the stock ``gate_up_proj`` and SiLU-and-mul, then
``ll.linear("dense_down")``; the un-reduced output is summed by the next hop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from sgl_nvidia_mega.glm5_3.ll.hops import DeferredFinalize, Nvfp4Input
from sgl_nvidia_mega.glm5_3.moe.step import MlpSteps
from sgl_nvidia_mega.glm5_3.registry import Candidate, Decline, DeclinedFamily

SGLANG_READS = {
    "srt/layers/moe/moe_runner/flashinfer_trtllm.py": (
        "quantize_hidden_states_fp4",
        "get_activation_type",
        "trtllm_moe_enable_pdl",
        "fused_experts_none_to_flashinfer_trtllm_fp4",
        "FlashInferTrtllmFp4MoeQuantInfo",
        "flashinfer_trtllm_deferred_finalize_context",
    ),
    "srt/layers/moe/utils.py": ("RoutingMethodType",),
    "srt/models/deepseek_v2.py": (
        "DeepseekV2MLP",
        "DeepseekV2MoE",
        "gate",
        "shared_experts",
        "alt_stream",
        "topk",
        "experts",
        "gate_up_proj",
        "down_proj",
        "act_fn",
        "num_fused_shared_experts",
        "_shared_expert_tp1",
        "swiglu_limit",
        "is_replicated",
    ),
}

logger = logging.getLogger(__name__)

# Named tolerances (test_glm5_3_ll_moe.py, the layer test on synthetic NVFP4
# weights): the next layer's normed input and residual of ll.hops + ll.moe against
# the stock MoE + FlashInfer's AR fusion. Bitwise wherever both run FlashInfer's
# one-shot geometry on the same FP4 input; what remains is the P10 quantizer (the
# hop's FlashInfer MNNVL epilogue vs SGLang's cute-dsl fp4_quantize: about 1e-3 of
# the FP4 codes differ by one step) and one-shot vs two-shot rounding above 21 rows.
TOLERANCES = {"ll.moe_layer.max_abs": 1.0, "ll.moe_layer.rel_l2": 5e-2}

# Option unit (units.OPTIONS, an opt-in slot): the shared expert on a stream of the greatest priority,
# so its pending blocks take the SMs freed before FC1's remaining blocks. With
# FlashInfer's small-batch routing kernel, FC1 launches ~1.3 us earlier at 6 rows and
# holds the SMs; on the alternate stream the shared expert's act_and_mul then starts
# ~14 us late, and the next hop, which joins it, waits ~1.25 us per MoE layer.
PRIORITY_UNIT = "ll.moe.alt_stream_priority"
# Widths (verify rows) up to which the option applies. Measured on GB300 at TP4 (8k/1k
# decode step against the alternate stream, small-batch routing on): -2.4% at 5 rows and
# -1.7% at 6; ungated it costs +2.1% at 12 rows and +1.7% at 24, where the shared down
# GEMM preempts FC1 / FC2's later blocks (gated, those widths are unchanged). Without
# the small-batch routing kernel (stock FlashInfer 0.7.2) it loses at 5 rows (+1.2%) and
# 6 rows (+2.4%): the shared expert already wins its race there. Hence opt-in.
PRIORITY_MAX_ROWS = 6


def _next_power_of_2(n: int) -> int:
    return 1 << max(int(n) - 1, 0).bit_length()


@dataclass
class RoutedArgs:
    """What the copied FlashInfer TRT-LLM NVFP4 MoE call reads of one layer (the
    fields of SGLang's ``FlashInferTrtllmFp4MoeQuantInfo`` plus the routing
    configuration), gathered once at build."""

    w13_weight: torch.Tensor
    w13_weight_scale: torch.Tensor  # e4m3 view
    w2_weight: torch.Tensor
    w2_weight_scale: torch.Tensor  # e4m3 view
    g1_scale_c: torch.Tensor
    g1_alphas: torch.Tensor
    g2_alphas: torch.Tensor
    w13_input_scale_quant: torch.Tensor
    num_experts: int
    local_expert_offset: int
    local_num_experts: int
    intermediate_size: int
    routing_bias: Optional[torch.Tensor]
    top_k: int
    n_group: Optional[int]
    topk_group: Optional[int]
    routed_scaling_factor: Optional[float]
    routing_method_type: int
    activation_type: int
    gemm1_alpha: Optional[torch.Tensor] = None
    gemm1_beta: Optional[torch.Tensor] = None
    gemm1_clamp_limit: Optional[torch.Tensor] = None

    @classmethod
    def of_layer(cls, mlp) -> RoutedArgs:
        """From a ``DeepseekV2MoE`` whose experts run FlashInfer TRT-LLM NVFP4 (as
        ``ModelOptNvFp4FusedMoEMethod.apply`` builds its quant info)."""
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            get_activation_type,
        )
        from sglang.srt.layers.moe.utils import RoutingMethodType

        ex = mlp.experts
        cfg = mlp.topk.topk_config
        rcfg = ex.moe_runner_config
        rmt = getattr(ex, "routing_method_type", None)
        opt = lambda name: (  # noqa: E731
            getattr(ex, name).data if getattr(ex, name, None) is not None else None
        )
        return cls(
            w13_weight=ex.w13_weight.data,
            w13_weight_scale=ex.w13_weight_scale.data.view(torch.float8_e4m3fn),
            w2_weight=ex.w2_weight.data,
            w2_weight_scale=ex.w2_weight_scale.data.view(torch.float8_e4m3fn),
            g1_scale_c=ex.g1_scale_c.data,
            g1_alphas=ex.g1_alphas.data,
            g2_alphas=ex.g2_alphas.data,
            w13_input_scale_quant=ex.w13_input_scale_quant,
            num_experts=int(ex.num_experts),
            local_expert_offset=int(ex.moe_ep_rank * ex.num_local_experts),
            local_num_experts=int(ex.num_local_experts),
            intermediate_size=int(ex.intermediate_size_per_partition),
            routing_bias=cfg.correction_bias,
            top_k=int(cfg.top_k),
            n_group=cfg.num_expert_group,
            topk_group=cfg.topk_group,
            routed_scaling_factor=rcfg.routed_scaling_factor,
            routing_method_type=int(
                rmt if rmt is not None else RoutingMethodType.Default
            ),
            activation_type=get_activation_type(
                rcfg.activation, is_gated=rcfg.is_gated
            ),
            gemm1_alpha=opt("gemm1_alpha"),
            gemm1_beta=opt("gemm1_beta"),
            gemm1_clamp_limit=opt("gemm1_clamp_limit"),
        )


def quantize_input(x: torch.Tensor, args: RoutedArgs):
    """The routed experts' NVFP4 input as the stock runner quantizes it."""
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        quantize_hidden_states_fp4,
    )

    return quantize_hidden_states_fp4(x, args.w13_input_scale_quant)


def routed_deferred(
    args: RoutedArgs, logits: torch.Tensor, fp4: torch.Tensor, sf: torch.Tensor
):
    """FlashInfer TRT-LLM NVFP4 MoE with ``do_finalize=False`` on routing logits
    (routing inside the kernel): ``(gemm2_out, expert_weights,
    expanded_idx_to_permuted_idx)``. The call of SGLang's
    ``fused_experts_none_to_flashinfer_trtllm_fp4`` (logits-based, deferred,
    unpadded hidden, static activation scale)."""
    from flashinfer.fused_moe import trtllm_fp4_block_scale_moe
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import trtllm_moe_enable_pdl

    rows = fp4.shape[0]
    hs_scale = sf.view(torch.float8_e4m3fn).reshape(*sf.shape[:-1], -1)
    result = trtllm_fp4_block_scale_moe(
        routing_logits=logits,
        routing_bias=args.routing_bias,
        hidden_states=fp4,
        hidden_states_scale=hs_scale,
        gemm1_weights=args.w13_weight,
        gemm1_weights_scale=args.w13_weight_scale,
        gemm1_bias=None,
        gemm1_alpha=args.gemm1_alpha,
        gemm1_beta=args.gemm1_beta,
        gemm1_clamp_limit=args.gemm1_clamp_limit,
        gemm2_weights=args.w2_weight,
        gemm2_weights_scale=args.w2_weight_scale,
        gemm2_bias=None,
        output1_scale_scalar=args.g1_scale_c,
        output1_scale_gate_scalar=args.g1_alphas,
        output2_scale_scalar=args.g2_alphas,
        per_token_scale=None,
        num_experts=args.num_experts,
        top_k=args.top_k,
        n_group=args.n_group,
        topk_group=args.topk_group,
        intermediate_size=args.intermediate_size,
        local_expert_offset=args.local_expert_offset,
        local_num_experts=args.local_num_experts,
        routed_scaling_factor=args.routed_scaling_factor,
        routing_method_type=args.routing_method_type,
        do_finalize=False,
        activation_type=args.activation_type,
        tune_max_num_tokens=_next_power_of_2(rows),
        enable_pdl=trtllm_moe_enable_pdl(rows),
    )
    gemm2_out, expert_weights, permuted_idx = result[:3]
    if (
        gemm2_out.dtype != torch.bfloat16
        or expert_weights.dtype not in (torch.bfloat16, torch.float32)
        or permuted_idx.dtype != torch.int32
    ):
        raise RuntimeError(
            "FlashInfer deferred finalize ABI changed: "
            f"{gemm2_out.dtype}, {expert_weights.dtype}, {permuted_idx.dtype}"
        )
    return gemm2_out, expert_weights, permuted_idx


def layer_reason(mlp) -> Optional[str]:
    """Why a ``DeepseekV2MoE`` layer cannot run on ``ll.moe`` (None if it can)."""
    from sglang.srt.layers.moe.utils import RoutingMethodType  # noqa: F401 - pinned

    ex = getattr(mlp, "experts", None)
    if ex is None:
        return "no routed experts"
    if not getattr(ex, "supports_deferred_finalize", False):
        return "the routed experts do not support the deferred finalize"
    if type(ex.quant_method).__name__ != "ModelOptNvFp4FusedMoEMethod":
        return f"routed experts quantized by {type(ex.quant_method).__name__}"
    runner = getattr(ex, "runner_backend", None)
    if runner is None or not runner.is_flashinfer_trtllm():
        return f"MoE runner {runner} (needs flashinfer_trtllm)"
    if getattr(ex, "trtllm_padded_hidden_size", None) is not None:
        return "hidden-padded TRT-LLM NVFP4 weights"
    if getattr(
        getattr(ex, "quant_method", None), "quant_config", None
    ) is not None and getattr(
        ex.quant_method.quant_config, "use_per_token_activation", False
    ):
        return "per-token NVFP4 activation scales"
    scale = getattr(ex, "w13_input_scale_quant", None)
    if not isinstance(scale, torch.Tensor) or scale.numel() != 1:
        return "the FC1 input scale is not a single tensor"
    if getattr(mlp, "num_fused_shared_experts", 0) != 0:
        return "fused shared experts"
    if getattr(mlp, "_shared_expert_tp1", False):
        return "a replicated (TP1) shared expert"
    if getattr(mlp, "shared_experts", None) is None:
        return "no shared expert"
    if getattr(mlp, "alt_stream", None) is None:
        return "no alternate stream for the shared expert"
    if getattr(mlp, "_enable_a2a_moe", False) or getattr(mlp, "is_hash", False):
        return "all-to-all or hash routing"
    if getattr(mlp.gate, "e_score_correction_bias_vl", None) is not None:
        return "vision top-k correction"
    return None


def dense_reason(mlp) -> Optional[str]:
    """Why a dense ``DeepseekV2MLP`` cannot run on ``ll.moe`` (None if it can)."""
    for name in ("gate_up_proj", "down_proj"):
        w = getattr(getattr(mlp, name, None), "weight", None)
        if not isinstance(w, torch.Tensor) or w.dtype != torch.bfloat16:
            return f"dense {name} is not a bf16 linear"
    if getattr(mlp, "swiglu_limit", None) is not None:
        return "a clamped SwiGLU"
    if getattr(mlp, "_enable_nvfp4_gemm_swiglu_fusion", False):
        return "the NVFP4 GEMM + SwiGLU fusion"
    if getattr(mlp, "is_replicated", False):
        return "a replicated dense MLP"
    return None


class LlMoeFamily:
    """The per-layer arguments of ``ll.moe`` on one model."""

    def __init__(
        self, layers: Dict[int, Any], dense: Dict[int, Any], sm=None, sms=None
    ) -> None:
        self.layers = layers  # layer id -> (mlp, RoutedArgs)
        self.dense = dense  # layer id -> dense mlp
        self.sm, self.sms = sm, sms  # the GEMM plans' device facts (None: this GPU's)
        self.widths: tuple = ()
        # the shared expert's stream (PRIORITY_UNIT), None: each layer's alt stream
        self.shared_stream: Optional[torch.cuda.Stream] = None

    def compile(self, widths) -> None:
        """Prepare the GEMM roles (``ll.build_linear``) at every width (outside
        capture: a planned row count that was not built raises at run time)."""
        from sgl_nvidia_mega.glm5_3 import ll

        self.widths = tuple(sorted(set(int(w) for w in widths)))
        facts = dict(sm=self.sm, sms=self.sms)
        for mlp, _ in self.layers.values():
            ll.build_linear("router", mlp.gate, self.widths, **facts)
        for mlp in self.dense.values():
            ll.build_linear("dense_down", mlp.down_proj, self.widths, **facts)
        from sgl_nvidia_mega.glm5_3 import units

        # listed in units.KNOWN (OPTIONS) and switched on (an opt-in slot: only when listed)
        if (
            PRIORITY_UNIT in units.KNOWN
            and units.enabled(PRIORITY_UNIT)
            and any(w <= PRIORITY_MAX_ROWS for w in self.widths)
        ):
            self.shared_stream = priority_stream()
            if (
                not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            ):
                logger.info(
                    "GLM LL MoE: shared expert on a stream of priority %d at <= %d rows (%s)",
                    self.shared_stream.priority,
                    PRIORITY_MAX_ROWS,
                    PRIORITY_UNIT,
                )

    def provider(self, candidate: Candidate, rows: int, mode: str) -> LlMlp:
        """The family protocol (``registry``)."""
        if rows not in self.widths:
            raise Decline(f"ll.moe is not built for {rows} rows")
        return LlMlp(self, rows)

    @classmethod
    def of_model(cls, model) -> LlMoeFamily:
        from sglang.srt.models.deepseek_v2 import DeepseekV2MLP, DeepseekV2MoE

        inner = model.model
        layers, dense = {}, {}
        for i in range(inner.start_layer, inner.end_layer):
            mlp = inner.layers[i].mlp
            if isinstance(mlp, DeepseekV2MoE):
                why = layer_reason(mlp)
                if why is not None:
                    raise ValueError(f"layer {i}: {why}")
                layers[i] = (mlp, RoutedArgs.of_layer(mlp))
            elif isinstance(mlp, DeepseekV2MLP):
                why = dense_reason(mlp)
                if why is not None:
                    raise ValueError(f"layer {i}: {why}")
                dense[i] = mlp
            else:
                raise ValueError(f"layer {i}: MLP {type(mlp).__name__}")
        return cls(layers, dense)


def dense_forward(mlp, x: torch.Tensor) -> torch.Tensor:
    """The dense MLP's un-reduced output: ``DeepseekV2MLP.forward``'s bf16 path, its
    ``down_proj`` through ``ll.linear("dense_down")``."""
    from sgl_nvidia_mega.glm5_3 import ll

    gate_up, _ = mlp.gate_up_proj(x)
    h = mlp.act_fn(gate_up)
    return ll.linear("dense_down", mlp.down_proj, h)


def priority_stream(device=None) -> torch.cuda.Stream:
    """A stream of the greatest priority torch offers on ``device`` (lower number =
    higher priority; torch maps an out-of-range request to its highest). Created at
    build: never under capture."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "ll.moe: the shared expert's stream is created inside capture"
        )
    device = (
        torch.device("cuda", torch.cuda.current_device()) if device is None else device
    )
    stream = torch.cuda.Stream(device=device, priority=-(1 << 16))
    if stream.priority >= 0:
        raise RuntimeError(f"ll.moe: no stream priority above 0 on {device}")
    return stream


def moe_forward(
    mlp,
    args: RoutedArgs,
    x: torch.Tensor,
    nvfp4: Optional[Nvfp4Input],
    stream: Optional[torch.cuda.Stream] = None,
):
    """One MoE layer's ``DeferredFinalize`` (router, routed experts, shared expert on
    ``stream``, default the layer's alternate stream), in
    ``forward_normal_dual_stream``'s issue order: the routed chain on the main stream
    first, then the shared expert."""
    from sgl_nvidia_mega.glm5_3 import ll

    side = mlp.alt_stream if stream is None else stream
    current = torch.cuda.current_stream()
    side.wait_stream(current)
    logits = ll.linear("router", mlp.gate, x)
    if nvfp4 is not None:
        fp4, sf = nvfp4.fp4, nvfp4.sf
    else:
        fp4, sf = quantize_input(x, args)
    gemm2_out, weights, idx = routed_deferred(args, logits, fp4, sf)
    with torch.cuda.stream(side):
        shared = mlp.shared_experts(x)
    current.wait_stream(side)
    return DeferredFinalize(gemm2_out, weights, idx, shared, args.top_k)


class LlMlp(MlpSteps):
    """The MLP step of one verify width on ``ll.moe``."""

    def __init__(self, family: LlMoeFamily, rows: int) -> None:
        self.f, self.rows = family, rows
        # the shared expert's stream at this width: the priority stream up to
        # PRIORITY_MAX_ROWS rows (when on), else the layer's alternate stream
        self.shared_stream = family.shared_stream if rows <= PRIORITY_MAX_ROWS else None

    def forward(self, layer, x, buffer: Optional[torch.Tensor], step):
        lid = layer.layer_id
        nvfp4 = getattr(step, "nvfp4_input", None)
        step.nvfp4_input = None
        if lid in self.f.dense:
            return dense_forward(self.f.dense[lid], x)
        mlp, args = self.f.layers[lid]
        if nvfp4 is not None and (nvfp4.layer_id != lid or nvfp4.norm is not x):
            raise RuntimeError(
                f"GLM LL MoE layer {lid}: the NVFP4 input of layer {nvfp4.layer_id} "
                "is not this layer's input"
            )
        return moe_forward(mlp, args, x, nvfp4, self.shared_stream)


# ============================================================== registry
MAX_ROWS = 48  # the finalize hop's widest plan (ll.hops)


def covers(profile, rows: int, mode: str) -> Optional[str]:
    """``ll.moe`` coverage: the TP4 target verify step (the small-M MoE engine serves TP8)."""
    if mode != "target_verify":
        return f"{mode} steps are the draft's"
    if profile.tp != 4:
        return f"TP{profile.tp}: ll.moe serves TP4 (the small-M MoE engine serves TP8)"
    if tuple(profile.sm) != (10, 3):
        return f"sm_{profile.sm[0]}{profile.sm[1]}: measured on GB300 (sm_103) only"
    if rows > MAX_ROWS:
        return f"{rows} rows (the finalize hop serves <= {MAX_ROWS})"
    return None


def _family(model: Any, derived: Any, profile: Any):
    """``FAMILIES["ll.moe"]``: the layers' arguments, or a decline naming the layer
    that does not fit."""
    try:
        family = LlMoeFamily.of_model(model)
    except ValueError as e:
        return DeclinedFamily(str(e))
    family.sm, family.sms = tuple(profile.sm), profile.sms
    return family


CANDIDATES = (
    # P10: ll.moe on the post-attention hop's NVFP4 output as the routed experts'
    # FC1 input (taken first; consuming "mlp_input:nvfp4" wires the hop to write it;
    # "-ll.moe.nvfp4" leaves ll.moe on its own quantize: the D6 leave-one-out)
    Candidate(
        "ll.moe.nvfp4",
        "mlp",
        ("forward",),
        "ll.moe",
        covers,
        provides=("mlp:deferred_finalize",),
        requires=("hops:finalize", "mlp_input:nvfp4"),
    ),
    # the MoE layers' unfinalized output for the next hop (P9), the dense layers'
    # partial; the hop that finalizes it must be taken with it
    Candidate(
        "ll.moe",
        "mlp",
        ("forward",),
        "ll.moe",
        covers,
        provides=("mlp:deferred_finalize",),
        requires=("hops:finalize",),
    ),
)
FAMILIES = {"ll.moe": _family}
