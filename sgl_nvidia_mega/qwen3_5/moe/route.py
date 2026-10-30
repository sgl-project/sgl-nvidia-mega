# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/layers/moe/fused_moe_triton/layer.py (FusedMoE.forward_deferred_finalize).
"""Slot ``route``: a MoE layer's router GEMM and the NVFP4 quantization of its routed
experts' input in one launch (``sm103/moe/router.py``), then SGLang's trtllm routed
experts on that input, with the deferred finalize.

The logits equal the gate's cuBLAS GEMM bit for bit and the quantized input equals the
trtllm runner's own quantizer, so the step's output is the stock one's. The runner
takes the quantized input through its pre-quantized (``hidden_states_scale``) path,
or slot ``mkm`` (``mkm.py``) takes the router front's outputs in its place.

Slot ``route_shared`` (numeric) also computes the shared expert's static-FP8 gate/up
projection in the router front's launch (SGLANG_Q35MF_ROUTE_SHARED on SGLang branch
nvidia/qwen35-agentx-gb300), so the shared expert on the alternate stream starts at its
activation, after that launch only; the gate/up sums in another order than the FP8
linear's GEMM.

Slot ``route_topk`` serves the steps slot ``route`` does not (17 to 896 rows): behind
SGLang's own router GEMM, one launch (``sm103/moe/router.route_topk_quant``) writes each
token's top-10 experts and renormalized weights as the trtllm-gen routing computes them
and the NVFP4 input as the runner's quantizer does; the runner then serves the
materialized routing (FlashInfer's routed MoE), which skips its own top-k kernel. The
logits are SGLang's, so the step's output is the stock one's.

Under TEP8, route_topk or route-big also writes the routing tables of the runner's
trtllm-gen MoE body, which then runs on them (``body.py``) in place of the runner's routed
op and its routing kernel.
The body's FC2 rows equal the op's bit for bit, and under expert parallelism a slot outside
the rank's experts gets permuted index -1 as there.

Slot ``mkm_m32`` takes route_topk's place at 17 to 32 rows: slot mkm's 32-token engine runs
the top-10, FC1, SwiGLU, the NVFP4 requantization and FC2 in one launch on the router
GEMM's logits and the NVFP4 input (with route_gemm the router front's tiles give both;
otherwise route_topk's launch quantizes the input), bit for bit the runner's rows on
those logits, so the step's output is route_topk's. Slots ``mkm_m64`` / ``mkm_m128`` /
``mkm_m256`` / ``mkm_m512`` take the runner's place at 33 to 64 / 65 to 128 / 129 to 256 /
257 to 512 rows: after route_topk's launch (the routing and the NVFP4 input) the 64 / 128 /
256 / 512-token engine runs FC1, SwiGLU, the requantization and FC2 on that routing in one
launch, bit for bit the runner's rows.
"""

from __future__ import annotations

import logging

import msgspec
import torch

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/qwen2_moe.py": (
        "Qwen2MoeSparseMoeBlock",
        "gate",
        "topk",
        "experts",
        "supports_deferred_finalize",
        "layer_id",
        # the shared expert after its gate/up projection (slot route_shared)
        "Qwen2MoeMLP",
        "shared_expert",
        "shared_expert_gate",
        "gate_up_proj",
        "down_proj",
        "act_fn",
        "_enable_silu_fp4_quant_fusion",
        "_silu_fp4_quant_fused",
    ),
    "srt/models/qwen3_5.py": ("layers", "start_layer", "end_layer", "mlp"),
    "srt/layers/moe/topk.py": (
        "TopKOutputChecker",
        "format_is_bypassed",
        "StandardTopKOutput",
    ),
    "srt/layers/moe/fused_moe_triton/layer.py": (
        "dispatcher",
        "run_moe_core",
        "run_moe_core(dispatch_output)",
        "quant_method",
        "moe_runner_config",
    ),
    "srt/layers/moe/token_dispatcher/standard.py": (
        "dispatch(hidden_states)",
        "dispatch(topk_output)",
        "combine(combine_input)",
        "hidden_states_scale",
        "StandardCombineInput",
    ),
    "srt/layers/moe/moe_runner/flashinfer_trtllm.py": (
        "flashinfer_trtllm_deferred_finalize_context",
        "trtllm_padded_hidden_size",
        # TEP8 body: the routed op's deferred-finalize output
        "_make_deferred_finalize_output",
        "_make_deferred_finalize_output(top_k)",
    ),
    "srt/layers/moe/utils.py": (
        "get_moe_runner_backend",
        "is_flashinfer_trtllm",
        "is_flashinfer_trtllm_routed",
    ),
    "srt/layers/moe/moe_runner/base.py": ("activation", "is_gated"),
    "srt/layers/quantization/modelopt_quant.py": (
        '"w13_input_scale_quant"',
        "quant_config",
        "use_per_token_activation",
        "ModelOptFp8LinearMethod",
        "use_marlin",
        "use_sm120_fp8",
        '"input_scale"',
        '"weight_scale"',
    ),
    "srt/layers/activation.py": ("SiluAndMul",),
}

# The SGLang functions whose body this module mirrors, by the digest of their code
# (test/unit/test_contract.py recomputes them on the installed SGLang)
SGLANG_MIRRORS = {
    "srt/layers/moe/fused_moe_triton/layer.py:FusedMoE.forward_deferred_finalize": "af4b474cbaf33493",
    "srt/layers/moe/fused_moe_triton/layer.py:FusedMoE._dispatch_with_pre_quant": "dcdefc28cf643081",
    "srt/models/qwen2_moe.py:Qwen2MoeMLP.forward": (
        "bb4c9fb2d7994e97",  # runtime v2: temporary core shared-act fast path
        "6091ffb613568a10",  # runtime v3: the policy and kernel live in this slot
    ),
}


# the (slot, rows) that have run (each logged once, as engagement evidence)
_ENGAGED: set = set()


class Bound(msgspec.Struct, frozen=True):
    """The MoE package's build result (``st.bound["moe"]``)."""

    route: bool = False  # the router front stands in for every layer's gate
    mkm: dict = msgspec.field(default_factory=dict)  # layer id -> its weight TMA maps
    mkm_wide: bool = False  # slot mkm_wide: mkm's 16-token engine built
    # [160] int32 0, 1, ...: slot mkm's rows are in expert order
    identity: object = None
    shared: bool = False  # slot route_shared: the shared gate/up in the router launch
    topk: bool = False  # slot route_topk: the top-10 + NVFP4 input launch built
    gemm: bool = (
        False  # slot route_gemm: the router front's 16-row tiles stand in for the GEMM
    )
    m32: dict = msgspec.field(
        default_factory=dict
    )  # slot mkm_m32: layer id -> weight TMA maps
    m64: bool = False  # slot mkm_m64: the 64-token engine built (on m32's maps)
    m128: bool = False  # slot mkm_m128: the 128-token engine built (on m32's maps)
    m256: bool = False  # slot mkm_m256: the 256-token engine built (on m32's maps)
    m512: bool = False  # slot mkm_m512: the 512-token engine built (on m32's maps)
    wide_bal: bool = False  # opt-in balanced work split for mkm_m256 / mkm_m512
    # A caller-specific upper bound for the 16-token engine. The draft uses 12 so
    # its alternate-stream shared expert can overlap rows 13-16 with the 32-token engine.
    m16_max_rows: int = 16
    pre: bool = False  # TEP8's body on route_topk/route-big routing tables
    reason: str = ""  # why a requested slot is not built


def build(
    model,
    max_rows: int,
    fused: bool,
    shared: bool = False,
    wide: bool = False,
) -> Bound:
    """The router front for every MoE layer, compiled for 1..max_rows rows, with
    ``fused`` slot mkm's kernel and weight maps (with ``wide`` its 16-token engine too,
    for the rows past 7 up to max_rows), and with ``shared`` slot route_shared's launch;
    or why a layer cannot take them."""
    from sgl_nvidia_mega.qwen3_5.moe import mkm
    from sgl_nvidia_mega.qwen3_5.sm103.moe import mkm as mkm_kernel
    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    m = model.model
    layers = [m.layers[i].mlp for i in range(m.start_layer, m.end_layer)]
    for i, mlp in zip(range(m.start_layer, m.end_layer), layers):
        reason = unsupported(mlp)
        if reason:
            return Bound(reason=f"layer {i}: {reason}")
    if shared:
        for i, mlp in zip(range(m.start_layer, m.end_layer), layers):
            reason = shared_unsupported(mlp)
            if reason:
                return Bound(reason=f"route_shared: layer {i}: {reason}")
    device = torch.cuda.current_device()
    router.build(device, max_rows, shared=shared)
    if shared:
        from sgl_nvidia_mega.qwen3_5.sm103.moe import shared_act_quant

        for columns in sorted(
            {int(mlp.shared_expert.gate_up_proj.weight.shape[1]) for mlp in layers}
        ):
            shared_act_quant.build(device, columns)
    if not fused:
        return Bound(route=True, shared=shared)
    for i, mlp in zip(range(m.start_layer, m.end_layer), layers):
        expert_parallel = mlp.experts.num_local_experts != mlp.experts.num_experts
        reason = (
            mkm.routing_unsupported(mlp) if expert_parallel else mkm.unsupported(mlp)
        )
        if reason:
            return Bound(route=True, reason=f"mkm: layer {i}: {reason}")
    intermediate = layers[0].experts.w2_weight.shape[-1] * 2
    # the 16-token engine serves the rows past 7 where this rank's intermediate has one
    wide = (
        wide
        and max_rows > mkm_kernel.MAX_TOKENS
        and intermediate in mkm_kernel.WIDE_ENGINES
    )
    mkm_kernel.build(device, intermediate, max_rows if wide else mkm_kernel.MAX_TOKENS)
    return Bound(
        route=True,
        mkm={
            mlp.layer_id: mkm_kernel.weight_maps(
                mlp.experts.w13_weight, mlp.experts.w2_weight
            )
            for mlp in layers
        },
        mkm_wide=wide,
        identity=identity_rows(mkm_kernel.M16_MAX_TOKENS * mkm_kernel.TOP_K),
        shared=shared,
    )


def identity_rows(n: int):
    """Return and register an int32 identity row map for the hop finalize."""
    from sgl_nvidia_mega.qwen3_5.sm103.hop import register_identity_rows

    rows = torch.arange(n, dtype=torch.int32, device="cuda")
    register_identity_rows(rows)
    return rows


def build_m32(
    model,
    bound: Bound,
    m32: bool = True,
    m64: bool = False,
    m128: bool = False,
    m256: bool = False,
    m512: bool = False,
    wide_bal: bool = False,
) -> Bound:
    """Slots mkm_m32 / mkm_m64 / mkm_m128 / mkm_m256 / mkm_m512 on ``bound`` (route_topk
    built): the 32 / 64 / 128 / 256 / 512-token engines and every MoE layer's weight maps
    (slot mkm's when built), or why they cannot serve."""
    from sgl_nvidia_mega.qwen3_5.sm103.moe import mkm as mkm_kernel

    m = model.model
    layers = [m.layers[i].mlp for i in range(m.start_layer, m.end_layer)]
    device = torch.cuda.current_device()
    intermediate = layers[0].experts.w2_weight.shape[-1] * 2
    reason = mkm_kernel.m32_unsupported(device, intermediate)
    if not reason and (m256 or m512):
        reason = mkm_kernel.wide_unsupported(device, intermediate)
    if reason:
        return msgspec.structs.replace(bound, reason=f"mkm_m32: {reason}")
    # each requested engine (an engine between two requested ones is built and unused)
    wanted = (m32, m64, m128, m256, m512)
    bounds = mkm_kernel.ENGINE_ROWS[2:]
    lo = bounds[wanted.index(True)]
    hi = bounds[len(wanted) - 1 - wanted[::-1].index(True)]
    mkm_kernel.build(
        device,
        intermediate,
        hi,
        min_rows=mkm_kernel.ENGINE_ROWS[bounds.index(lo) + 1] + 1,
        wide_bal=wide_bal,
    )
    return msgspec.structs.replace(
        bound,
        m64=m64,
        m128=m128,
        m256=m256,
        m512=m512,
        wide_bal=wide_bal and (m256 or m512),
        m32={
            mlp.layer_id: bound.mkm.get(mlp.layer_id)
            or mkm_kernel.weight_maps(mlp.experts.w13_weight, mlp.experts.w2_weight)
            for mlp in layers
        },
        identity=identity_rows(max(hi, mkm_kernel.M128_MAX_TOKENS) * mkm_kernel.TOP_K),
    )


def unsupported(mlp) -> str:
    """Why the router front cannot stand in for ``mlp``'s gate and the runner's input
    quantizer ("" when it can)."""
    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    if not isinstance(mlp, Qwen2MoeSparseMoeBlock):
        return f"{type(mlp).__name__} is not a Qwen2MoeSparseMoeBlock"
    w = mlp.gate.weight
    if w.dtype != torch.bfloat16 or tuple(w.shape) != (router.EXPERTS, router.HIDDEN):
        return f"router gate {w.dtype} {tuple(w.shape)}"
    experts = mlp.experts
    scale = experts.w13_input_scale_quant
    if scale.dtype != torch.float32 or scale.numel() != 1:
        return f"routed input scale {scale.dtype} {tuple(scale.shape)}"
    if experts.trtllm_padded_hidden_size is not None:
        return "the routed experts' hidden dim is padded"
    if experts.quant_method.quant_config.use_per_token_activation:
        return "per-token activation scales"
    if not mlp.supports_deferred_finalize:
        return "no deferred finalize"
    return ""


def topk_unsupported(model) -> str:
    """Why slot route_topk cannot serve every MoE layer of ``model`` ("" when it can):
    the router front's gate and input quantizer, and the trtllm-gen weights with a
    softmax top-10 renormalized over 512 experts. Under expert parallelism only the
    materialized-routing contract applies; the plugin's full-expert engines do not."""
    from sgl_nvidia_mega.qwen3_5.moe import mkm

    m = model.model
    for i in range(m.start_layer, m.end_layer):
        mlp = m.layers[i].mlp
        expert_parallel = mlp.experts.num_local_experts != mlp.experts.num_experts
        reason = unsupported(mlp) or (
            mkm.routing_unsupported(mlp) if expert_parallel else mkm.unsupported(mlp)
        )
        if reason:
            return f"layer {i}: {reason}"
    return ""


def pre_unsupported(model) -> str:
    """Why TEP8's routed body cannot serve ("" when it can)."""
    from sglang.srt.layers.moe.utils import get_moe_runner_backend

    try:
        from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module
        from flashinfer.fused_moe.shared.inputs import (  # noqa: F401
            MoeRunnerInputs,
            alloc_trtllm_moe_output,
        )
        from flashinfer.tllm_enums import RoutingInputMode  # noqa: F401
    except ImportError as e:
        return f"FlashInfer without the MoE body entry ({e})"
    module = get_trtllm_moe_sm100_module()
    if not hasattr(module, "MoERunner") or not hasattr(module, "DABodyRunner"):
        return "FlashInfer without the MoE body entry (MoERunner / DABodyRunner)"
    backend = get_moe_runner_backend()
    if not (backend.is_flashinfer_trtllm() or backend.is_flashinfer_trtllm_routed()):
        return f"MoE runner backend {backend}"
    m = model.model
    for i in range(m.start_layer, m.end_layer):
        experts = m.layers[i].mlp.experts
        cfg = experts.moe_runner_config
        if cfg.activation != "silu" or not cfg.is_gated:
            return f"layer {i}: activation {cfg.activation} (gated {cfg.is_gated})"
        for name in ("gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit"):
            if getattr(experts, name, None) is not None:
                return f"layer {i}: {name}"
        if getattr(experts, "trtllm_padded_hidden_size", None) is not None:
            return f"layer {i}: padded hidden size"
    return ""


def shared_unsupported(mlp) -> str:
    """Why the router front cannot compute ``mlp``'s shared gate/up ("" when it can):
    a static per-tensor FP8 projection of at most 512 rows, then SiLU-and-mul."""
    from sglang.srt.layers.activation import SiluAndMul
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8LinearMethod
    from sglang.srt.models.qwen2_moe import Qwen2MoeMLP

    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    expert = mlp.shared_expert
    if not isinstance(expert, Qwen2MoeMLP) or mlp.shared_expert_gate is None:
        return "no gated Qwen2MoeMLP shared expert"
    gu = expert.gate_up_proj
    method = getattr(gu, "quant_method", None)
    weight = getattr(gu, "weight", None)
    input_scale = getattr(gu, "input_scale", None)
    weight_scale = getattr(gu, "weight_scale", None)
    if (
        not isinstance(method, ModelOptFp8LinearMethod)
        or method.use_marlin
        or method.use_sm120_fp8
        or getattr(gu, "bias", None) is not None
        or weight is None
        or not weight.is_cuda
        or weight.dtype != torch.float8_e4m3fn
        or weight.shape[0] != router.HIDDEN
        or weight.shape[1] % 8
        or weight.shape[1] > router.SHARED_MAX_N
        or not weight.t().is_contiguous()
        or input_scale is None
        or not input_scale.is_cuda
        or input_scale.device != weight.device
        or input_scale.dtype != torch.float32
        or tuple(input_scale.shape) not in ((), (1,))
        or not input_scale.is_contiguous()
        or weight_scale is None
        or not weight_scale.is_cuda
        or weight_scale.device != weight.device
        or weight_scale.dtype != torch.float32
        or tuple(weight_scale.shape) not in ((), (1,))
        or not weight_scale.is_contiguous()
    ):
        return "the shared gate/up is not a static per-tensor FP8 linear of <= 512 rows"
    if not isinstance(expert.act_fn, SiluAndMul):
        return f"shared expert activation {type(expert.act_fn).__name__}"
    if not _down_takes_static_fp8(expert):
        return "the shared down projection does not accept static per-tensor FP8"
    return ""


def shared_expert_tail(expert, gate_up: torch.Tensor) -> torch.Tensor:
    """``Qwen2MoeMLP.forward`` after its gate_up projection."""
    if expert._enable_silu_fp4_quant_fusion and not isinstance(gate_up, tuple):
        x, _ = expert.down_proj(expert._silu_fp4_quant_fused(gate_up))
        return x
    if _down_takes_static_fp8(expert) and gate_up.dtype == torch.bfloat16:
        from sgl_nvidia_mega.qwen3_5.sm103.moe.shared_act_quant import (
            silu_and_mul_static_quant_fp8,
        )

        quantized = silu_and_mul_static_quant_fp8(
            gate_up, expert.down_proj.input_scale.reshape(1)
        )
        key = ("route_shared_act_quant", gate_up.shape[0])
        if key not in _ENGAGED:
            _ENGAGED.add(key)
            logger.info(
                "NVIDIA megakernel: Qwen3.5 slot route_shared shared_act_quant "
                "runs at %d rows",
                gate_up.shape[0],
            )
        x, _ = expert.down_proj(quantized)
        return x
    x = expert.act_fn(gate_up)
    x, _ = expert.down_proj(x)
    return x


def _down_takes_static_fp8(expert) -> bool:
    """Whether this shared expert's down projection accepts static E4M3 input.

    This is the Qwen3.5 slot policy that runtime-v2 temporarily placed on
    ``Qwen2MoeMLP``.  Keeping it beside the plugin-owned activation kernel lets
    runtime-v3 restore the stock model class without leaving a private attribute
    dependency here.
    """
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8LinearMethod

    down = expert.down_proj
    method = getattr(down, "quant_method", None)
    weight = getattr(down, "weight", None)
    scale = getattr(down, "input_scale", None)
    return (
        isinstance(method, ModelOptFp8LinearMethod)
        and not method.use_marlin
        and not method.use_sm120_fp8
        and weight is not None
        and weight.is_cuda
        and weight.dtype == torch.float8_e4m3fn
        and scale is not None
        and scale.is_cuda
        and scale.device == weight.device
        and scale.dtype == torch.float32
        and tuple(scale.shape) in ((), (1,))
        and scale.is_contiguous()
    )


def routed(
    mlp,
    x: torch.Tensor,
    fused: Bound | None = None,
    shared: bool = False,
    pre_topk: bool = False,
):
    """``FusedMoE.forward_deferred_finalize`` of ``mlp``'s routed experts on ``x``
    [rows, hidden], with the router front's logits and quantized input (``fused``: on
    slot mkm's kernel). With ``pre_topk``, a routing-only launch writes TEP8's body
    tables. With ``shared`` (at most 8 rows) the
    router launch also computes the shared expert's gate/up: returns (routed, gate_up, the
    event recorded on the current stream right after that launch)."""
    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    experts = mlp.experts
    width = tables = None
    if pre_topk:
        from sgl_nvidia_mega.qwen3_5.moe import body

        width = body.width(experts, x)
        tables = width.tables
    for slot in ("route" if fused is None else "mkm",) + (
        ("route_shared",) if shared else ()
    ):
        if (slot, x.shape[0]) not in _ENGAGED:
            _ENGAGED.add((slot, x.shape[0]))
            logger.info(
                "NVIDIA megakernel: Qwen3.5 slot %s runs at %d rows", slot, x.shape[0]
            )
    if pre_topk and ("tep8_body", x.shape[0]) not in _ENGAGED:
        _ENGAGED.add(("tep8_body", x.shape[0]))
        logger.info(
            "NVIDIA megakernel: Qwen3.5 TEP8 routed body runs at %d rows",
            x.shape[0],
        )
    if shared:
        gu = mlp.shared_expert.gate_up_proj
        logits, x_fp4, x_sf, gate_up = router.router_gemm_quant_shared(
            x,
            mlp.gate.weight,
            experts.w13_input_scale_quant.reshape(1),
            gu.weight.t(),
            gu.input_scale.reshape(1),
            gu.weight_scale.reshape(1),
        )
        ready = torch.cuda.Event()
        ready.record()
        out = _routed_experts(mlp, x, fused, logits, x_fp4, x_sf)
        return out, gate_up, ready
    scale = experts.w13_input_scale_quant.reshape(1)
    if pre_topk:
        from sgl_nvidia_mega.qwen3_5 import plan

        rows = x.shape[0]
        if ("pre_tier", rows) not in _ENGAGED:
            _ENGAGED.add(("pre_tier", rows))
            logger.info(
                "NVIDIA megakernel: Qwen3.5 MoE front tier %s at %d rows: slot route's front, then route_topk's tables",
                plan.moe_front_tier(rows),
                rows,
            )
        logits, x_fp4, x_sf = router.router_gemm_quant(x, mlp.gate.weight, scale)
        router.route_topk_quant(logits, x, scale, quantize=False, tables=tables)
        return _pre_body(mlp, width, x_fp4, x_sf)
    logits, x_fp4, x_sf = router.router_gemm_quant(x, mlp.gate.weight, scale)
    return _routed_experts(mlp, x, fused, logits, x_fp4, x_sf)


def routed_lin(mlp, x: torch.Tensor, fused: Bound, logits, x_fp4, x_sf):
    """``routed`` with the router front's outputs taken from the MoE input's hop.

    Slot ``route_lin`` is numeric for the logits from the mixer's router partials; its
    NVFP4 input is bitwise equal to the router front's. Slot ``mkm`` runs on them.
    """
    if ("route_lin", x.shape[0]) not in _ENGAGED:
        _ENGAGED.add(("route_lin", x.shape[0]))
        logger.info(
            "NVIDIA megakernel: Qwen3.5 route-lin (router before the all-reduce) runs at %d rows",
            x.shape[0],
        )
    return _routed_experts(mlp, x, fused, logits, x_fp4, x_sf)


def _pre_body(mlp, width, x_fp4, x_sf):
    """TEP8's MoE body on ``width``'s tables, as a deferred-finalize output."""
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        _make_deferred_finalize_output,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    from sgl_nvidia_mega.qwen3_5.moe import body
    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    experts = mlp.experts
    result = body.run(width, experts, x_fp4, x_sf.view(torch.float8_e4m3fn))
    deferred = _make_deferred_finalize_output(result, top_k=router.TOP_K)
    return experts.dispatcher.combine(
        combine_input=StandardCombineInput(hidden_states=deferred)
    )


def _routed_experts(mlp, x, fused, logits, x_fp4, x_sf):
    """The routed experts on the router front's outputs (``fused``: slot mkm's kernel),
    deferred."""
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        flashinfer_trtllm_deferred_finalize_context,
    )
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    from sgl_nvidia_mega.qwen3_5.moe import mkm

    experts = mlp.experts
    if fused is not None:
        maps = fused.mkm[mlp.layer_id]
        return mkm.routed(
            mlp,
            maps,
            fused.identity,
            logits,
            x_fp4,
            x_sf,
            m16_max_rows=fused.m16_max_rows,
        )
    topk_output = mlp.topk(x, logits)
    if not TopKOutputChecker.format_is_bypassed(topk_output):
        raise RuntimeError("the router front requires logits-based bypassed TopK")
    dispatch_output = experts.dispatcher.dispatch(
        hidden_states=x, topk_output=topk_output
    )._replace(hidden_states=x_fp4, hidden_states_scale=x_sf.view(torch.float8_e4m3fn))
    with flashinfer_trtllm_deferred_finalize_context():
        combine_input = experts.run_moe_core(dispatch_output=dispatch_output)
    return experts.dispatcher.combine(combine_input=combine_input)


def routed_topk(mlp, x: torch.Tensor, gemm: bool = False, pre: bool = False):
    """``FusedMoE.forward_deferred_finalize`` of ``mlp``'s routed experts on ``x``
    [rows, hidden] with SGLang's router GEMM, then slot route_topk's launch (the top-10
    and the NVFP4 input), and the runner on the materialized routing. With ``gemm``
    (slot route_gemm, numeric) the router front's GEMM and quantizer in 16-row tiles
    stand in for SGLang's router GEMM, and route_topk's launch only routes. With ``pre``,
    TEP8's internal front also writes the MoE body's routing tables and the body runs on
    them (``body.py``) in place of the runner's routed op."""
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        flashinfer_trtllm_deferred_finalize_context,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    from sgl_nvidia_mega.qwen3_5 import plan
    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    if ("route_topk", x.shape[0]) not in _ENGAGED:
        _ENGAGED.add(("route_topk", x.shape[0]))
        logger.info(
            "NVIDIA megakernel: Qwen3.5 slot route_topk runs at %d rows",
            x.shape[0],
        )
    if pre and ("tep8_body", x.shape[0]) not in _ENGAGED:
        _ENGAGED.add(("tep8_body", x.shape[0]))
        logger.info(
            "NVIDIA megakernel: Qwen3.5 TEP8 routed body runs at %d rows",
            x.shape[0],
        )
    experts = mlp.experts
    scale = experts.w13_input_scale_quant.reshape(1)
    width = tables = None
    if pre:
        from sgl_nvidia_mega.qwen3_5.moe import body

        width = body.width(experts, x)
        tables = width.tables
    if gemm:
        logits, x_fp4, x_sf = router.router_gemm_quant_tiled(x, mlp.gate.weight, scale)
        ids, weights, _, _ = router.route_topk_quant(
            logits, x, scale, quantize=False, tables=tables
        )
    elif pre and x.shape[0] >= plan.ROUTE_BIG_MIN_ROWS:
        rows = x.shape[0]
        if ("route_big", rows) not in _ENGAGED:
            _ENGAGED.add(("route_big", rows))
            cl, _, words = router.big_shape(rows)
            logger.info(
                "NVIDIA megakernel: Qwen3.5 MoE front tier %s at %d rows: "
                "route-big C%d%s",
                plan.moe_front_tier(rows),
                rows,
                cl,
                "W" if words > 1 else "",
            )
        logits, _ = mlp.gate(x)
        ids, weights, x_fp4, x_sf = router.route_big_quant(
            logits,
            x,
            scale,
            tables,
            ctas=(
                plan.ROUTE_BIG_WIDE_CTAS
                if rows >= plan.ROUTE_BIG_WIDE_MIN_ROWS
                else None
            ),
        )
    else:
        logits, _ = mlp.gate(x)
        ids, weights, x_fp4, x_sf = router.route_topk_quant(
            logits, x, scale, tables=tables
        )
    if pre:
        rows = x.shape[0]
        if rows < plan.ROUTE_BIG_MIN_ROWS and ("pre_tier", rows) not in _ENGAGED:
            _ENGAGED.add(("pre_tier", rows))
            logger.info(
                "NVIDIA megakernel: Qwen3.5 MoE front tier %s at %d rows: "
                "route_topk tables%s",
                plan.moe_front_tier(rows),
                rows,
                " (route_gemm front)" if gemm else "",
            )
        return _pre_body(mlp, width, x_fp4, x_sf)
    topk_output = StandardTopKOutput(
        topk_weights=weights, topk_ids=ids, router_logits=logits
    )
    dispatch_output = experts.dispatcher.dispatch(
        hidden_states=x, topk_output=topk_output
    )._replace(hidden_states=x_fp4, hidden_states_scale=x_sf.view(torch.float8_e4m3fn))
    with flashinfer_trtllm_deferred_finalize_context():
        combine_input = experts.run_moe_core(dispatch_output=dispatch_output)
    return experts.dispatcher.combine(combine_input=combine_input)


def routed_m32(mlp, x: torch.Tensor, bound: Bound, gemm: bool = False):
    """``FusedMoE.forward_deferred_finalize`` of ``mlp``'s routed experts on ``x``
    [17-32 rows, hidden] in one launch of the 32-token engine (slot mkm_m32), on SGLang's
    router GEMM and route_topk's quantized input, or with ``gemm`` (slot route_gemm,
    numeric) on the router front's tiles (logits and quantized input in one launch)."""
    from sgl_nvidia_mega.qwen3_5.moe import mkm
    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    if ("mkm_m32", x.shape[0]) not in _ENGAGED:
        _ENGAGED.add(("mkm_m32", x.shape[0]))
        logger.info(
            "NVIDIA megakernel: Qwen3.5 slot mkm_m32 runs at %d rows", x.shape[0]
        )
    scale = mlp.experts.w13_input_scale_quant.reshape(1)
    if gemm:
        logits, x_fp4, x_sf = router.router_gemm_quant_tiled(x, mlp.gate.weight, scale)
    else:
        logits, _ = mlp.gate(x)
        _, _, x_fp4, x_sf = router.route_topk_quant(logits, x, scale)
    return mkm.routed(
        mlp,
        bound.m32[mlp.layer_id],
        bound.identity,
        logits,
        x_fp4,
        x_sf,
        m16_max_rows=bound.m16_max_rows,
    )


def routed_given(mlp, x: torch.Tensor, bound: Bound, slot: str):
    """``FusedMoE.forward_deferred_finalize`` of ``mlp``'s routed experts on ``x``
    [33-512 rows, hidden]: SGLang's router GEMM, route_topk's launch (the routing and the
    NVFP4 input, as slot route_topk runs them), then the 64, 128, 256 or 512-token engine on
    that routing in place of the runner (slots mkm_m64, mkm_m128, mkm_m256, mkm_m512)."""
    from sgl_nvidia_mega.qwen3_5.moe import mkm
    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    if (slot, x.shape[0]) not in _ENGAGED:
        _ENGAGED.add((slot, x.shape[0]))
        logger.info(
            "NVIDIA megakernel: Qwen3.5 slot %s runs at %d rows", slot, x.shape[0]
        )
    logits, _ = mlp.gate(x)
    ids, weights, x_fp4, x_sf = router.route_topk_quant(
        logits, x, mlp.experts.w13_input_scale_quant.reshape(1)
    )
    return mkm.routed(
        mlp,
        bound.m32[mlp.layer_id],
        bound.identity,
        logits,
        x_fp4,
        x_sf,
        routing=(ids, weights),
        wide_bal=bound.wide_bal,
        m16_max_rows=bound.m16_max_rows,
    )
