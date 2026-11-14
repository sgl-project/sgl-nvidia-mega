# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team): python/sglang/srt/models/qwen2_moe.py
# (Qwen2MoeSparseMoeBlock._forward_router_experts, its deferred-finalize branch and the
# dual-stream NVFP4 prequant) and python/sglang/srt/layers/moe/moe_runner/ (the Qwen3.8
# fused routing's prepared-routing run, flashinfer_trtllm.py's logits-bypassed deferred
# call and the source's quantize_nvfp4_routed_input). Changed: the step calls
# FlashInfer's prepared-routing body itself and hands the prequantized input to the
# runner as already quantized; see the docstring.
"""The routed half of the executor's deferred-finalize MoE step (slot ``moe_routing``):
what ``mlp._forward_router_experts(x, defer_finalize=True)`` returns for TP1 ModelOpt
NVFP4 experts on FlashInfer's TRT-LLM runner, inside slot ``moe_finalize``'s step
(``runtime/moe.py``). The router GEMV (``mlp.gate``) and the logits-bypassed
``mlp.topk`` run as stock, then one of:

1. ``fused_routing`` (1..``max_rows`` rows, at most 512): one JIT kernel
   (``kernels/ops/moe/fused_routing.py``) selects each row's top-10 of 512, computes the
   RenormalizeNaive weights, builds the trtllm-gen routing metadata and writes the NVFP4
   activations; FlashInfer's prepared-routing (DA) body then runs only the two GEMMs, at
   the tactic FlashInfer's autotuner returns for the stock call (the same runner, inputs
   and tuning config, so the same lookup key). 1..32 rows take a single-CTA kernel,
   33..512 a thread-block cluster; above 16 rows only with FlashInfer's block-per-token
   routing split off (``FLASHINFER_ROUTING_FORCE_BLOCK_PER_TOKEN=0``), whose single-
   cluster kernel the fused one reproduces.
2. ``prequant`` (under graph capture, rows the fused path does not take): the stock
   quant (``quantize_hidden_states_fp4``) on the block's alternate stream beside the
   router GEMV, then the stock deferred call with that input passed as already quantized
   (the runner's ``hidden_states_scale`` input).
3. otherwise the stock deferred call (``experts.forward_deferred_finalize``) on the same
   routing; ``routed`` returns None before the router GEMV when neither path can run.

Slot ``moe_finalize``'s extend-size step and its prefill tuning pass take the source's
extend order instead: ``quantize`` on the current stream, the router GEMV and
``deferred`` on the block's side stream, its experts after the quant's event.

SGLang main's NVFP4 runner has no prepared-routing entry (its DA body serves MXFP4 only)
and ignores a typed prequantized input, so (1) calls FlashInfer with the stock call's
arguments (``_call_kwargs`` mirrors them) and (2) uses the pre-quantized input the
runner takes from an NVFP4 dispatch.

Numerics: (2) gives the stock call's bits (the same quant function on the same input).
(1) matches FlashInfer 0.7.3's and 0.7.4's routing kernels and CuTe-DSL ``fp4_quantize``
(unchanged between them) by construction: ids, weights, NVFP4 activations and every
metadata field but the row order inside an expert; each permuted row's GEMMs are
unchanged and the finalize reads rows through the returned permutation
(``test/kernels/test_qwen3_8_moe_routing.py``).
"""

from __future__ import annotations

import functools
import logging
import os
from dataclasses import dataclass

import torch

from sgl_nvidia_mega.qwen3_8.runtime import bf16_gemm_pins

logger = logging.getLogger(__name__)

OP = "moe.routing"
STOCK = "stock"
MOE_OP = "flashinfer::trtllm_fp4_block_scale_moe"  # the stock call's autotuner entry
# Rows above DynBlock's 16 take FlashInfer's cluster kernel only when the
# block-per-token split path is forced off; the split path's math is not checked here.
DYNBLOCK_MAX_ROWS = 16

SGLANG_READS = {
    "srt/models/qwen2_moe.py": (
        "gate",
        "topk",
        "experts",
        "alt_stream",
        "enable_shared_expert_fusion",
    ),
    "srt/layers/moe/topk.py": (
        "topk_config",
        "router_logits",
        "top_k",
        "correction_bias",
        "num_expert_group",
        "topk_group",
    ),
    "srt/layers/moe/fused_moe_triton/layer.py": (
        "FusedMoE",
        "forward_deferred_finalize",
        "run_moe_core(dispatch_output)",
        "dispatcher",
        "moe_runner_config",
        "runner_backend",
        "quant_method",
        "num_experts",
        "num_local_experts",
        "moe_ep_rank",
        "intermediate_size_per_partition",
        "routing_method_type",
        "_mega_moe_nvfp4",
    ),
    "srt/layers/moe/moe_runner/base.py": (
        "activation",
        "is_gated",
        "routed_scaling_factor",
    ),
    "srt/layers/moe/moe_runner/flashinfer_trtllm.py": (
        "quantize_hidden_states_fp4",
        "get_activation_type",
        "get_activation_type(is_gated)",
        "trtllm_moe_enable_pdl",
        "flashinfer_trtllm_deferred_finalize_context",
        "FlashInferTrtllmDeferredFinalizeOutput",
        "trtllm_padded_hidden_size",
    ),
    "srt/layers/moe/token_dispatcher/standard.py": (
        "StandardDispatchOutput",
        "combine(combine_input)",
    ),
    "srt/layers/moe/utils.py": (
        "RoutingMethodType",
        "RenormalizeNaive",
        "get_moe_a2a_backend",
        "is_none",
        "is_flashinfer_trtllm",
    ),
    "srt/layers/quantization/modelopt_quant.py": (
        "ModelOptNvFp4FusedMoEMethod",
        "use_per_token_activation",
    ),
    "srt/layers/quantization/fp4_utils.py": ("_flashinfer_fp4_quantize_backend",),
    "srt/model_executor/runner/__init__.py": ("get_is_capture_mode",),
    "srt/runtime_context.py": ("get_parallel",),
    "srt/environ.py": ("envs", "SGLANG_EXPERIMENTAL_LORA_OPTI"),
    "srt/utils/triton_utils.py|srt/utils/common.py": ("next_power_of_2",),
}

# The SGLang functions whose body this module mirrors, by the digest of their code
# (test/unit/test_contract.py recomputes them on the installed SGLang)
SGLANG_MIRRORS = {
    "srt/models/qwen2_moe.py:Qwen2MoeSparseMoeBlock._forward_router_experts": "251e3d343aec92fe",
    "srt/layers/moe/fused_moe_triton/layer.py:FusedMoE.forward_deferred_finalize": "af4b474cbaf33493",
    "srt/layers/moe/moe_runner/flashinfer_trtllm.py:fused_experts_none_to_flashinfer_trtllm_fp4": "9cf153ccc3beee6f",
    "srt/layers/quantization/modelopt_quant.py:ModelOptNvFp4FusedMoEMethod.apply": "1dc4978c61c40ba6",
}

# Reads and mirrors that need an SGLang change newer than the unit job's pin: the
# contract test skips each, naming the change, only where the installed SGLang lacks it.
SGLANG_REQUIRES = {
    "srt/models/qwen2_moe.py:Qwen2MoeSparseMoeBlock._forward_router_experts": (
        "sglang#48366",
        ("32d09f01b77d76b5",),
    ),
    "srt/layers/moe/moe_runner/flashinfer_trtllm.py:fused_experts_none_to_flashinfer_trtllm_fp4": (
        "sglang#48468",
        ("339d95256df67c5a",),
    ),
}

# The FlashInfer releases whose sources the tuner-key mirror, default tile and PDL default
# follow: 0.7.4 leaves every file below, the launcher and the routing kernels as in 0.7.3.
FLASHINFER_VERSIONS = ("0.7.3", "0.7.4")

# The FlashInfer internals the fused path mirrors (trtllm_fp4_block_scale_moe_op)
FLASHINFER_READS = {
    "flashinfer/fused_moe/backends/trtllm/sm100_runner.py": (
        "MoERunner",
        "_make_tuning_config",
        "forward(do_preparation)",
        '"da_routing_metadata"',
        '"da_body_workspace"',
    ),
    "flashinfer/fused_moe/core.py": (
        "get_trtllm_moe_sm100_module",
        "TrtllmMoERoutingMetadataSlot",
    ),
    "flashinfer/fused_moe/shared/inputs.py": (
        "MoeRunnerInputs",
        "alloc_trtllm_moe_output",
    ),
}


@dataclass(frozen=True)
class Options:
    """The source's MoE routing knobs as values, with its tuned defaults."""

    # SGLANG_OPT_MOE_FUSED_ROUTING=1: the fused routing kernel and FlashInfer's
    # prepared-routing body
    fused_routing: bool = True
    # the source's fused-routing row cap (512 in its tuned runs; at most 512): 1..32 rows
    # run the single-CTA kernel, 33..512 the cluster kernel
    max_rows: int = 512
    # SGLANG_OPT_MOE_NVFP4_PREQUANT_OVERLAP=1: under graph capture, quantize the routed
    # input on the alternate stream beside the router GEMV
    prequant_overlap: bool = True


DEFAULT = Options()

# Set by build once the fused path can run in this process (a FLASHINFER_VERSIONS
# release, the CuTe-DSL quant recipe, the kernel compiled)
_fused_built = False


@functools.lru_cache(maxsize=None)
def _log_once(reason: str) -> None:
    logger.info("Qwen3.8 slot moe_routing: fused routing not taken: %s", reason)


def layer_reasons(mlp) -> str:
    """'' when the step takes this sparse MoE block's experts (the source's TP1 NVFP4
    TRT-LLM eligibility), else why not."""
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend
    from sglang.srt.layers.quantization.modelopt_quant import (
        ModelOptNvFp4FusedMoEMethod,
    )

    experts = mlp.experts
    if envs.SGLANG_EXPERIMENTAL_LORA_OPTI.get():
        return "the experimental LoRA MoE runner"
    if not get_moe_a2a_backend().is_none():
        return "an all-to-all MoE backend"
    if type(experts) is not FusedMoE or not isinstance(
        experts.quant_method, ModelOptNvFp4FusedMoEMethod
    ):
        return "the experts are not ModelOpt NVFP4 FusedMoE"
    if not experts.runner_backend.is_flashinfer_trtllm():
        return "the MoE runner is not FlashInfer TRT-LLM"
    if experts.quant_method.quant_config.use_per_token_activation:
        return "per-token NVFP4 activations"
    if experts.trtllm_padded_hidden_size is not None or experts._mega_moe_nvfp4:
        return "hidden-padded or MegaMoE NVFP4 experts"
    if mlp.enable_shared_expert_fusion:
        return "the shared expert is fused into the routed experts"
    return ""


def fused_reasons() -> str:
    """'' when the fused path can run in this process, else why not."""
    import flashinfer
    from sglang.srt.layers.quantization.fp4_utils import (
        _flashinfer_fp4_quantize_backend,
    )

    from sgl_nvidia_mega.qwen3_8.kernels.ops.moe import fused_routing as op

    if flashinfer.__version__ not in FLASHINFER_VERSIONS:
        return (
            f"FlashInfer {flashinfer.__version__}, the fused path follows "
            + " and ".join(FLASHINFER_VERSIONS)
        )
    # The fused quant reproduces the CuTe-DSL fp4_quantize standard recipe only.
    flags = ("FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH", "FLASHINFER_NVFP4_4OVER6")
    if _flashinfer_fp4_quantize_backend != "cute-dsl" or any(
        os.environ.get(flag, "") not in ("", "0") for flag in flags
    ):
        return "the NVFP4 quant is not CuTe-DSL fp4_quantize's standard recipe"
    if not op.available():
        return "the fused routing kernel does not build"
    return ""


def build(model, options: Options = DEFAULT) -> str:
    """'' when the step serves every MoE layer of ``model`` (target or MTP draft), else
    why the routed experts stay stock. With ``options.fused_routing`` it builds the fused
    path once per process; where that cannot run the step keeps the prequant path and
    warns."""
    global _fused_built
    from sglang.srt.runtime_context import get_parallel

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return "the trtllm-gen NVFP4 MoE needs an SM10x GPU"
    tp = get_parallel().tp_size
    if tp != 1:
        return f"TP{tp}: the routed-input paths are qualified at TP1"
    m = model.model
    for i in range(m.start_layer, m.end_layer):
        reason = layer_reasons(m.layers[i].mlp)
        if reason:
            return f"layer {i}: {reason}"
    if options.fused_routing and not _fused_built:
        reason = fused_reasons()
        if reason:
            logger.warning(
                "Qwen3.8 slot moe_routing runs without fused routing: %s", reason
            )
        _fused_built = not reason
    return ""


def routed(executor, mlp, x, st):
    """What ``mlp._forward_router_experts(x, defer_finalize=True)`` returns (the deferred
    output), or None when the slot is off or neither path takes the rows (the caller then
    runs it). ``executor.moe_routing``: the ``Options`` its build succeeded with."""
    from sglang.srt.model_executor.runner import get_is_capture_mode

    options = executor.moe_routing
    if options is None:
        return None
    experts = mlp.experts
    fused = _takes_fused(options, mlp, x)
    prequant = (
        not fused
        and options.prequant_overlap
        and mlp.alt_stream is not None
        and get_is_capture_mode()
    )
    if not (fused or prequant):
        return None
    if prequant:
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            quantize_hidden_states_fp4,
        )

        current = torch.cuda.current_stream()
        mlp.alt_stream.wait_stream(current)
        with torch.cuda.stream(mlp.alt_stream):
            x_fp4, x_sf = quantize_hidden_states_fp4(x, experts.w13_input_scale_quant)
    logits = bf16_gemm_pins.linear(mlp.gate, x)
    topk_output = mlp.topk(x, logits)
    if prequant:
        current.wait_stream(mlp.alt_stream)
        x_fp4.record_stream(current)
        x_sf.record_stream(current)
        st.ran(OP, "prequant")
        return _prequantized(experts, x_fp4, x_sf, topk_output)
    out = _fused(experts, x, topk_output)
    if out is None:
        st.ran(OP, STOCK)
        return experts.forward_deferred_finalize(x, topk_output)
    st.ran(OP, "fused_routing")
    return out


def fused_from_logits(mlp, x, logits, st):
    """``routed``'s fused path on router logits the caller computed."""
    experts = mlp.experts
    topk_output = mlp.topk(x, logits)
    out = _fused(experts, x, topk_output)
    if out is None:
        st.ran(OP, STOCK)
        return experts.forward_deferred_finalize(x, topk_output)
    st.ran(OP, "fused_routing")
    return out


def quantize(mlp, x):
    """The source's ``quantize_nvfp4_routed_input``: SGLang's quant of the routed input on
    the current stream, and the event the routed experts wait for (``deferred``)."""
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        quantize_hidden_states_fp4,
    )

    x_fp4, x_sf = quantize_hidden_states_fp4(x, mlp.experts.w13_input_scale_quant)
    return x_fp4, x_sf, torch.cuda.current_stream().record_event()


def deferred(mlp, x, logits, quant):
    """``mlp.experts.forward_deferred_finalize(x, mlp.topk(x, logits))``; with ``quant``
    (``quantize``'s output) the experts take it as already quantized once its event has
    passed, as the source's runner does with a prequantized input."""
    topk_output = mlp.topk(x, logits)
    if quant is None:
        return mlp.experts.forward_deferred_finalize(x, topk_output)
    x_fp4, x_sf, ready = quant
    torch.cuda.current_stream().wait_event(ready)
    return _prequantized(mlp.experts, x_fp4, x_sf, topk_output)


def _prequantized(experts, x_fp4, x_sf, topk_output):
    """``FusedMoE.forward_deferred_finalize`` on an NVFP4 input; at TP1 / EP1 the
    dispatch passes through."""
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        flashinfer_trtllm_deferred_finalize_context,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput

    dispatch = StandardDispatchOutput(
        hidden_states=x_fp4, hidden_states_scale=x_sf, topk_output=topk_output
    )
    with flashinfer_trtllm_deferred_finalize_context():
        combine_input = experts.run_moe_core(dispatch_output=dispatch)
    return experts.dispatcher.combine(combine_input=combine_input)


def _takes_fused(options, mlp, x) -> bool:
    """Whether the fused path takes these rows, from host state (``_fused`` checks the
    logits' layout once they exist)."""
    from flashinfer.autotuner import AutoTuner
    from sglang.srt.layers.moe.utils import RoutingMethodType

    from sgl_nvidia_mega.qwen3_8.kernels.ops.moe import fused_routing as op

    rows = x.shape[0]
    if not (
        options.fused_routing
        and _fused_built
        and rows <= min(options.max_rows, op.MAX_ROWS)
    ):
        return False
    experts, config = mlp.experts, mlp.topk.topk_config
    checks = (
        (config.top_k == op.TOP_K and config.correction_bias is None, "routing config"),
        (not config.num_expert_group and not config.topk_group, "grouped routing"),
        (
            experts.routing_method_type == RoutingMethodType.RenormalizeNaive,
            "routing method",
        ),
        (
            experts.num_experts == op.NUM_EXPERTS
            and experts.num_local_experts == op.NUM_EXPERTS
            and experts.moe_ep_rank == 0,
            "expert layout",
        ),
        (
            rows <= DYNBLOCK_MAX_ROWS
            or os.environ.get("FLASHINFER_ROUTING_FORCE_BLOCK_PER_TOKEN")
            in ("0", "off", "OFF"),
            "routing kernel above 16 rows",
        ),
        (not AutoTuner.get().is_tuning_mode, "autotuning"),
    )
    reason = next((reason for ok, reason in checks if not ok), None)
    if reason not in (None, "autotuning"):  # the tuning pass runs the stock call
        _log_once(reason)
    return reason is None


def _call_kwargs(experts, topk_output, x_fp4, x_sf) -> dict:
    """The stock call's ``trtllm_fp4_block_scale_moe`` arguments: SGLang's
    ``fused_experts_none_to_flashinfer_trtllm_fp4`` on a logits-bypassed top-k with its
    finalize deferred, with the quant info ``ModelOptNvFp4FusedMoEMethod.apply`` builds."""
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        get_activation_type,
        trtllm_moe_enable_pdl,
    )
    from sglang.srt.utils import next_power_of_2

    def data(name):
        t = getattr(experts, name, None)
        return t.data if t is not None else None

    config = topk_output.topk_config
    runner_config = experts.moe_runner_config
    rows = x_fp4.shape[0]
    return dict(
        routing_logits=topk_output.router_logits,
        routing_bias=config.correction_bias,
        hidden_states=x_fp4,
        hidden_states_scale=x_sf,
        gemm1_weights=experts.w13_weight.data,
        gemm1_weights_scale=experts.w13_weight_scale.data.view(torch.float8_e4m3fn),
        gemm1_bias=None,
        gemm1_alpha=data("gemm1_alpha"),
        gemm1_beta=data("gemm1_beta"),
        gemm1_clamp_limit=data("gemm1_clamp_limit"),
        gemm2_weights=experts.w2_weight.data,
        gemm2_weights_scale=experts.w2_weight_scale.data.view(torch.float8_e4m3fn),
        gemm2_bias=None,
        output1_scale_scalar=experts.g1_scale_c.data,
        output1_scale_gate_scalar=experts.g1_alphas.data,
        output2_scale_scalar=experts.g2_alphas.data,
        per_token_scale=None,
        num_experts=experts.num_experts,
        top_k=config.top_k,
        n_group=config.num_expert_group,
        topk_group=config.topk_group,
        intermediate_size=experts.intermediate_size_per_partition,
        local_expert_offset=experts.moe_ep_rank * experts.num_local_experts,
        local_num_experts=experts.num_local_experts,
        routed_scaling_factor=runner_config.routed_scaling_factor,
        routing_method_type=experts.routing_method_type,
        do_finalize=False,
        activation_type=get_activation_type(
            runner_config.activation, is_gated=runner_config.is_gated
        ),
        tune_max_num_tokens=next_power_of_2(rows),
        enable_pdl=trtllm_moe_enable_pdl(rows),
    )


def _fused(experts, x, topk_output):
    """The stock call's deferred output from the fused kernel and FlashInfer's
    prepared-routing body, or None when the tensors' layout does not fit the kernel."""
    from flashinfer.autotuner import AutoTuner
    from flashinfer.fused_moe.shared.inputs import (
        MoeRunnerInputs,
        alloc_trtllm_moe_output,
    )
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        FlashInferTrtllmDeferredFinalizeOutput,
    )

    from sgl_nvidia_mega.qwen3_8.kernels.ops.moe import fused_routing as op

    router_logits = topk_output.router_logits
    scale = experts.w13_input_scale_quant
    if not op.covered(router_logits, x, scale):
        _log_once("tensor layout")
        return None
    num_tokens = x.shape[0]
    device = x.device
    x_fp4, x_sf = op.alloc_quant_outputs(x)
    moe_kwargs = _call_kwargs(
        experts, topk_output, x_fp4, x_sf.view(torch.float8_e4m3fn)
    )
    # Same shapes and dtypes as the stock op's own buffers: part of the tuner key. The
    # stock deferred call passes no output, so the op profiles its (rows, 0) placeholder.
    ids = torch.empty((num_tokens, op.TOP_K), dtype=torch.int32, device=device)
    weights = torch.empty((num_tokens, op.TOP_K), dtype=torch.bfloat16, device=device)
    inputs = MoeRunnerInputs(
        output=alloc_trtllm_moe_output(num_tokens, op.HIDDEN_SIZE, False, device),
        routing_logits=router_logits,
        topk_ids=ids,
        expert_weights=weights,
        hidden_states=moe_kwargs["hidden_states"],
        hidden_states_scale=moe_kwargs["hidden_states_scale"],
        gemm1_lora_delta=None,
        per_token_scale=None,
    )
    runner, tuning_config, runner_kwargs = _stock_runner(moe_kwargs, inputs)
    _, tactic = AutoTuner.get().choose_one(
        MOE_OP, [runner], tuning_config, inputs.to_list(), **runner_kwargs
    )
    tactic = [-1, -1] if isinstance(tactic, int) else [int(v) for v in tactic]
    # 0.7.3 resolveMoeTileAndConfig keeps an explicit tile whose config is -1.
    if tactic[0] < 0:
        tile_n = _default_tile_n(num_tokens, moe_kwargs["local_num_experts"])
    else:
        tile_n = tactic[0]

    routed = op.fused_routing(
        router_logits,
        x,
        scale,
        x_fp4=moe_kwargs["hidden_states"],
        x_sf=moe_kwargs["hidden_states_scale"],
        ids=ids,
        weights=weights,
        tile_tokens=tile_n,
    )
    metadata = list(routed.metadata.tensors())
    flat_inputs = inputs.to_list()
    workspace = runner.forward(
        flat_inputs,
        tactic=tactic,
        do_preparation=True,
        da_routing_metadata=metadata,
        **runner_kwargs,
    )
    gemm2_out = runner.forward(
        flat_inputs,
        tactic=tactic,
        da_routing_metadata=metadata,
        da_body_workspace=workspace,
        **runner_kwargs,
    )[0]
    if not isinstance(gemm2_out, torch.Tensor):
        gemm2_out = torch.from_dlpack(gemm2_out)
    return FlashInferTrtllmDeferredFinalizeOutput(
        gemm2_out=gemm2_out,
        expert_weights=weights,
        expanded_idx_to_permuted_idx=routed.metadata.expanded_idx_to_permuted_idx,
        top_k=op.TOP_K,
    )


def _default_tile_n(num_tokens: int, num_local_experts: int) -> int:
    # FlashInfer 0.7.3/0.7.4 selectDefaultTileN over FP4BlockScaleLauncher::
    # getSupportedTileNums (E2m1 x E2m1; the 192 tile on every build, Rubin included).
    tiles = (8, 16, 32, 64, 128, 192, 256)
    average = num_tokens * 10 / num_local_experts
    i = next((i for i, t in enumerate(tiles) if t >= average), len(tiles) - 1)
    return tiles[max(i - 1, 0)]


def _stock_runner(moe_kwargs: dict, inputs):
    """The MoERunner, tuning config and runner kwargs trtllm_fp4_block_scale_moe_op
    builds for this call (FlashInfer 0.7.3/0.7.4 core.py), so the tuner key is the same."""
    from flashinfer.fused_moe.backends.trtllm import MoERunner
    from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module
    from flashinfer.tllm_enums import (
        Fp8QuantizationType,
        RoutingInputMode,
        WeightLayout,
        deduce_trtllm_gen_tensor_dtype,
    )
    from flashinfer.utils import device_support_pdl

    hidden_size = inputs.hidden_states.shape[-1] * 2
    runner = MoERunner(
        get_trtllm_moe_sm100_module().moe_op,
        top_k=moe_kwargs["top_k"],
        num_local_experts=moe_kwargs["local_num_experts"],
        dtype_act=deduce_trtllm_gen_tensor_dtype(
            inputs.hidden_states, inputs.hidden_states_scale
        ),
        dtype_weights=deduce_trtllm_gen_tensor_dtype(
            moe_kwargs["gemm1_weights"], moe_kwargs["gemm1_weights_scale"]
        ),
        fp8_quantization_type=Fp8QuantizationType.NoneFp8,
        hidden_size=hidden_size,
        hidden_size_output=hidden_size,
        intermediate_size=moe_kwargs["intermediate_size"],
        activation_type=moe_kwargs["activation_type"],
        weight_layout=WeightLayout.MajorK,
        use_shuffled_weight=True,
        use_per_token_scaling=False,
        num_experts=moe_kwargs["num_experts"],
        num_fused_shared_experts=0,
    )
    tuning_config = runner._make_tuning_config(
        inputs,
        tune_max_num_tokens=moe_kwargs["tune_max_num_tokens"],
        routing_input_mode=RoutingInputMode.FromLogits,
        hidden_states_scale_layout=None,
        use_cold_l2_cache=True,
        use_cuda_graph=True,
    )
    # 0.7.3 drops 0.7.2's SM107 MoE PDL veto: only None defers to the device.
    enable_pdl = moe_kwargs["enable_pdl"]
    if enable_pdl is None:
        enable_pdl = device_support_pdl(inputs.hidden_states.device)
    runner_kwargs = {
        "workspace_buffer": moe_kwargs.get("workspace_buffer"),
        "routing_input_mode": int(RoutingInputMode.FromLogits),
        "num_experts": moe_kwargs["num_experts"],
        "routing_bias": moe_kwargs["routing_bias"],
        "gemm1_weights": moe_kwargs["gemm1_weights"],
        "gemm1_weights_scale": moe_kwargs["gemm1_weights_scale"],
        "gemm1_bias": moe_kwargs["gemm1_bias"],
        "gemm1_alpha": moe_kwargs["gemm1_alpha"],
        "gemm1_beta": moe_kwargs["gemm1_beta"],
        "gemm1_clamp_limit": moe_kwargs["gemm1_clamp_limit"],
        "gemm2_weights": moe_kwargs["gemm2_weights"],
        "gemm2_weights_scale": moe_kwargs["gemm2_weights_scale"],
        "gemm2_bias": moe_kwargs["gemm2_bias"],
        "output1_scale_scalar": moe_kwargs["output1_scale_scalar"],
        "output1_scale_gate_scalar": moe_kwargs["output1_scale_gate_scalar"],
        "output2_scale_scalar": moe_kwargs["output2_scale_scalar"],
        "n_group": moe_kwargs["n_group"],
        "topk_group": moe_kwargs["topk_group"],
        "local_expert_offset": moe_kwargs["local_expert_offset"],
        "routed_scaling_factor": moe_kwargs["routed_scaling_factor"],
        "routing_method_type": moe_kwargs["routing_method_type"],
        "enable_pdl": enable_pdl,
        "do_finalize": False,
        "activation_type": moe_kwargs["activation_type"],
        "num_fused_shared_experts": 0,
        "norm_topk_prob": True,
        "routing_replay_out": None,
        "valid_hidden_size": None,
        "valid_intermediate_size": None,
    }
    return runner, tuning_config, runner_kwargs
