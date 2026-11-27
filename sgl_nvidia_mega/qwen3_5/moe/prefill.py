# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/layers/layer_boundary/stage.py and prepare.py (a stage's input),
# python/sglang/srt/layers/layer_boundary/exit.py (an FFN stage's output),
# python/sglang/kernels/ops/elementwise/elementwise.py (the shared-expert gate launch),
# python/sglang/srt/models/qwen2_moe.py (Qwen2MoeSparseMoeBlock.forward),
# python/sglang/srt/layers/moe/fused_moe_triton/layer.py (FusedMoE.forward),
# python/sglang/srt/layers/quantization/modelopt_quant.py (the CuTe DSL standard path of
# ModelOptNvFp4FusedMoEMethod.apply) and
# python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py
# (fused_experts_none_to_flashinfer_cutedsl_fp4).

from __future__ import annotations

import logging
from functools import partial
from typing import NamedTuple, Optional

import torch

logger = logging.getLogger(__name__)

SLOT = "moe_prefill_input_nvfp4"
SLOT_ADD_NORM = "moe_prefill_add_norm"
SLOT_NOCLONE = "moe_prefill_noclone"
SF_VEC = 16

SGLANG_READS = {
    "srt/models/qwen3_5.py": (
        "attn_boundary",
        "ffn_boundary",
        "mlp",
        "input_layernorm",
        "post_attention_layernorm",
        '"_is_layer_to_capture"',
    ),
    "srt/models/qwen2_moe.py": (
        "Qwen2MoeSparseMoeBlock",
        "_use_mega_moe",
        "alt_stream",
        "gate",
        "topk",
        "experts",
        "shared_expert",
        "shared_expert_gate",
        "enable_shared_expert_fusion",
        "reduce_results",
        "_forward_shared_experts",
        "_forward_shared_experts(apply_gate)",
        "_is_cuda",
        "_use_aiter",
        "_SGLANG_EXPERIMENTAL_LORA_OPTI",
        "use_intel_amx_backend",
    ),
    "srt/layers/layer_boundary/stage.py": ("plan", "declaration"),
    "srt/layers/layer_boundary/exit.py": (
        "FfnExit",
        "boundary",
        "defer_moe_finalize",
        "_complete",
        "_update",
        "_declared_sum",
        "_complete_now",
        "_complete_now(dp_step)",
        "_complete_now(owes_sum)",
    ),
    "srt/layers/layer_boundary/ops.py": ("keep_output",),
    "srt/layers/layer_boundary/layout.py": ("SumGroup", "ATTN_TP", "TP", "MOE_OUTPUT"),
    "srt/layers/layer_boundary/factories.py": ("row_transport",),
    "srt/layers/layer_boundary/construction.py": (
        "BatchVariant",
        "paths",
        "fusions",
        "norm",
        "qkv_latent_func",
        "_publish_lora_layout",
        "variant_for",
        "path_for",
        "enters_stack",
    ),
    "srt/layers/layer_boundary/contracts.py": (
        "ORDINARY",
        "SEQUENCE_PARALLEL",
        "output",
        "group",
        "sum_impl",
        "transform",
        "returns_over_dp",
        "output_move",
        "writes_at_handoff",
        "entry",
        "prepare",
        "input_move",
        "attn_input_adapter",
    ),
    "srt/layers/layer_boundary/prepare.py": (
        "_run_entry",
        "_run_entry(step)",
        "_run_entry(is_plain_add)",
        "_run_entry(carried_fusions)",
        "_run_entry(carried_read_fusions)",
        "_run_entry(read)",
        "_run_entry(expected_sum)",
        "_update_read",
        "_update_read(pre_move)",
        "_update_read(enters_stack)",
        "_update_read(read)",
        "_attn_input_default",
    ),
    "srt/layers/layer_boundary/residual/add_norm.py": (
        "NormReadout",
        "NormQuantReadout",
        "is_plain_add",
        "applied_at_exit",
    ),
    "srt/layers/layer_boundary/residual/batch.py": ("stream_of",),
    "srt/layers/layer_boundary/residual/stream.py": (
        "pending",
        "owed",
        "update",
        "input",
        "write",
    ),
    "srt/layers/layernorm.py": ("GemmaRMSNorm", "variance_epsilon", "forward_cuda"),
    "kernels/fused_op.py": (
        "get_fused_op_backend",
        "_forward_method",
        "_resolve_forward_method",
        "is_torch_compile",
    ),
    "kernels/ops/elementwise/elementwise.py": (
        "fused_gate_sigmoid_mul_add",
        "_fused_gate_sigmoid_mul_add_kernel",
        "_fused_gate_sigmoid_mul_add_kernel(GATE_ONLY)",
        "_is_hip",
        "is_arch_support_pdl",
        "is_batch_invariant_mode_enabled",
    ),
    "srt/model_executor/runner/__init__.py": ("get_is_capture_mode",),
    "srt/model_executor/runner_backend_utils/breakable_cuda_graph/context.py": (
        "is_in_breakable_cuda_graph",
    ),
    "srt/model_executor/runner_backend_utils/tc_piecewise_cuda_graph/__init__.py": (
        "is_in_tc_piecewise_cuda_graph",
    ),
    "srt/runtime_context.py": (
        "get_exec",
        "get_parallel",
        '"tp_group"',
        '"tp_size"',
        '"attn_tp_size"',
        '"moe_ep_size"',
        '"moe_tp_size"',
    ),
    "srt/layers/moe/__init__.py": ("get_moe_a2a_backend", "get_moe_runner_backend"),
    "srt/layers/moe/utils.py": ("is_none", "is_flashinfer_cutedsl"),
    "srt/layers/moe/fused_moe_triton/layer.py": (
        "dispatcher",
        "quant_method",
        "run_moe_core",
        "_use_ascend_fuseep",
        "_dwdp_bound",
        "moe_tp_size",
        "moe_ep_size",
    ),
    "srt/layers/moe/token_dispatcher/standard.py": (
        "StandardDispatcher",
        "StandardCombineInput",
        "hidden_states_scale",
        "topk_output",
        "dispatch",
        "combine",
    ),
    "srt/layers/moe/topk.py": ("TopKOutputChecker", "format_is_standard"),
    "srt/layers/moe/moe_runner/runner.py": ("fused_func", "lora_enabled"),
    "srt/layers/moe/moe_runner/flashinfer_cutedsl.py": (
        "fused_experts_none_to_flashinfer_cutedsl_fp4",
        "ensure_cutedsl_wrapper",
        "_cutedsl_wrapper",
        "_cutedsl_scales",
        "_cutedsl_input_scale",
        "quant_mode",
    ),
    "srt/layers/quantization/modelopt_quant.py": (
        "ModelOptNvFp4FusedMoEMethod",
        "_mega_moe_nvfp4",
        "_moe_runner_backend",
        "_is_cutedsl_v1_deepep",
        "use_per_token_activation",
        "w13_weight",
        "w2_weight",
        "w13_blockscale_swizzled",
        "w2_blockscale_swizzled",
        '"w13_blockscale_mma"',
        '"w2_blockscale_mma"',
    ),
    "srt/layers/dp_attention.py": ("is_allocation_symmetric",),
    "srt/distributed/device_communicators/pynccl_allocator.py": (
        "use_symmetric_memory",
    ),
}

SGLANG_MIRRORS = {
    "srt/layers/layer_boundary/stage.py:StageBoundary.prepare": "f0769bf152eebade",
    "srt/layers/layer_boundary/stage.py:StageBoundary._prepare_input": "53b7f2a5bd394d76",
    "srt/layers/layer_boundary/stage.py:StageBoundary._select": (
        "a1ab3bafed24a06c",
        "ba73d2b3981b6208",
    ),
    "srt/layers/layer_boundary/stage.py:StageBoundary._prepare": "3ade6a2952490616",
    "srt/layers/layer_boundary/prepare.py:_run_entry": "639740496fc09d2d",
    "srt/layers/layer_boundary/prepare.py:_update_read": "7d8519afcd233ad1",
    "srt/layers/layer_boundary/residual/add_norm.py:NormReadout.update_and_read": "d6cb5286ca1304a4",
    "srt/layers/layer_boundary/residual/add_norm.py:NormQuantReadout.update_and_read": "a606319fdf5a2091",
    "srt/layers/layer_boundary/residual/add_norm.py:_update_and_read_residual_plain": "4d97834b2f09f7c1",
    "srt/layers/layer_boundary/residual/add_norm.py:_norm_quant_kernel": "211e37c00fbe7bb2",
    "srt/layers/layer_boundary/residual/stream.py:Contribution.for_boundary": "499696d26f884c3c",
    "srt/layers/layernorm.py:GemmaRMSNorm.forward_cuda": "ef46f7b180a41cfd",
    "srt/layers/layernorm.py:GemmaRMSNorm._forward_impl": "8e9b3a51e75af28c",
    "srt/models/qwen2_moe.py:Qwen2MoeSparseMoeBlock.forward": "80c4b54b62df6e29",
    "srt/models/qwen2_moe.py:Qwen2MoeSparseMoeBlock.forward_normal_dual_stream": "afae959064cc8cd7",
    "srt/models/qwen2_moe.py:Qwen2MoeSparseMoeBlock._forward_router_experts": "251e3d343aec92fe",
    "srt/layers/moe/fused_moe_triton/layer.py:FusedMoE.forward": "8f8872b5ee64da79",
    "srt/layers/moe/fused_moe_triton/layer.py:FusedMoE.forward_impl": "0842f421bde04eed",
    "srt/layers/moe/fused_moe_triton/layer.py:FusedMoE._dispatch_with_pre_quant": "dcdefc28cf643081",
    "srt/layers/quantization/modelopt_quant.py:ModelOptNvFp4FusedMoEMethod.apply": "1dc4978c61c40ba6",
    "srt/layers/moe/moe_runner/runner.py:MoeRunner.run": "23230fc181ea929e",
    "srt/layers/moe/moe_runner/flashinfer_cutedsl.py:fused_experts_none_to_flashinfer_cutedsl_fp4": "b7cd905a9a97f152",
    "srt/layers/layer_boundary/stage.py:StageBoundary._prepare_attention": "ab2768d2e68434a5",
    "srt/layers/layer_boundary/exit.py:FfnExit.__init__": "97ec313564cf0da6",
    "srt/layers/layer_boundary/exit.py:FfnExit.finish": "24808a7f67639234",
    "srt/layers/layer_boundary/exit.py:ExitPolicy._complete_now": "0da409d98e9cba78",
    "srt/layers/layer_boundary/exit.py:ExitPolicy._record_output": "ab96b211f975d353",
    "srt/layers/layer_boundary/ops.py:sum_output": "d43ba78c3b1df15a",
    "kernels/ops/elementwise/elementwise.py:fused_gate_sigmoid_mul_add": "e545ff4a77e8ae86",
    "kernels/ops/elementwise/elementwise.py:_launch_fused_gate_sigmoid_mul": "21b2f9d290468ef0",
}

# the (point, reason) fallbacks and the engagement already logged
_LOGGED: set = set()


def _once(key, msg: str, *args) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        logger.info(msg, *args)


class Layers:
    """The MoE layers of a stage the prefill body runs here (``build``):"""

    def __init__(self, nvfp4=(), add_norm=(), noclone=()):
        self.nvfp4 = frozenset(nvfp4)
        self.add_norm = frozenset(add_norm)
        self.noclone = frozenset(noclone)
        self.entries = frozenset(i + 1 for i in self.add_norm)
        self.blocks = self.nvfp4 | self.add_norm | self.noclone
        self.ids = self.blocks | self.entries
        self.scales: dict = {}


def _log_refused(slot, refused) -> None:
    i, reason = refused[0]
    more = f" (and {len(refused) - 1} more)" if len(refused) > 1 else ""
    logger.info(
        "NVIDIA megakernel: Qwen3.5 slot %s uses stock on layer %d: %s%s",
        slot,
        i,
        reason,
        more,
    )


def build(
    model, ids, nvfp4=True, add_norm=False, noclone=False
) -> tuple[Optional[Layers], dict]:
    layers = model.model.layers
    ids = sorted(ids)
    why: dict = {}
    taken: dict = {}
    for slot, on, refusal in (
        (SLOT, nvfp4, lambda i: layer_refusal(layers[i])),
        (SLOT_ADD_NORM, add_norm, lambda i: add_norm_refusal(layers, i, ids)),
        (SLOT_NOCLONE, noclone, lambda i: noclone_refusal(layers[i])),
    ):
        if not on:
            continue
        refused = [(i, r) for i in ids if (r := refusal(i))]
        taken[slot] = [i for i in ids if i not in dict(refused)]
        if refused and taken[slot]:
            _log_refused(slot, refused)
        if not taken[slot]:
            why[slot] = (
                f"layer {refused[0][0]}: {refused[0][1]}" if refused else "no MoE layer"
            )
    if taken.get(SLOT) or taken.get(SLOT_ADD_NORM):
        _build_kernels(layers, taken, why)
    if not any(taken.values()):
        return None, why
    return (
        Layers(
            taken.get(SLOT, ()),
            taken.get(SLOT_ADD_NORM, ()),
            taken.get(SLOT_NOCLONE, ()),
        ),
        why,
    )


def _build_kernels(layers, taken, why) -> None:
    from sgl_nvidia_mega.qwen3_5.sm103.moe import prefill_input_nvfp4 as nvfp4_kernel
    from sgl_nvidia_mega.qwen3_5.sm103.moe import prefill_output_add_norm as add_kernel

    device = torch.device("cuda", torch.cuda.current_device())
    stock = sglang_refusal(device, nvfp4_kernel)
    sm = nvfp4_kernel.get_sm_version(device)
    for slot, kernel in ((SLOT, nvfp4_kernel), (SLOT_ADD_NORM, add_kernel)):
        hidden = {
            layers[i].input_layernorm.weight.shape[0] for i in taken.get(slot, ())
        }
        bad = [h for h in sorted(hidden) if not kernel.config_supported(h, sm)]
        if hidden and (stock or bad):
            why[slot] = stock or f"hidden size {bad[0]}"
            taken[slot] = []
            continue
        for size in sorted(hidden):
            kernel.precompile(device, size)


def sglang_refusal(device, kernel) -> str:
    from sgl_kernel import elementwise

    if not getattr(elementwise, "_has_flashinfer", False):
        return "SGLang's Gemma norm does not run FlashInfer's"
    if not kernel.stock_runs_cute_dsl(device):
        return "FlashInfer runs its CUDA norm on this GPU"
    return ""


def layer_refusal(layer) -> str:
    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

    mlp = getattr(layer, "mlp", None)
    if not isinstance(mlp, Qwen2MoeSparseMoeBlock):
        return "not a sparse MoE block"
    if _hooked(layer):
        # the body runs the layer's forward without its module call
        return "the layer has module hooks"
    return (
        block_refusal(mlp) or experts_refusal(mlp.experts) or ffn_input_refusal(layer)
    )


def noclone_refusal(layer) -> str:
    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

    mlp = getattr(layer, "mlp", None)
    if not isinstance(mlp, Qwen2MoeSparseMoeBlock):
        return "not a sparse MoE block"
    if _hooked(layer):
        # the body runs the layer's forward without its module call
        return "the layer has module hooks"
    return block_refusal(mlp) or experts_refusal(mlp.experts)


def _hooked(module) -> bool:
    return bool(
        getattr(module, "_forward_hooks", None)
        or getattr(module, "_forward_pre_hooks", None)
    )


def block_refusal(mlp) -> str:
    from sglang.srt.layers.moe import get_moe_a2a_backend
    from sglang.srt.models.qwen2_moe import (
        _SGLANG_EXPERIMENTAL_LORA_OPTI,
        _is_cuda,
        _use_aiter,
        use_intel_amx_backend,
    )

    if mlp._use_mega_moe:
        return "MegaMoE experts"
    if not get_moe_a2a_backend().is_none():
        return "a MoE all-to-all"
    if mlp.shared_expert is None or mlp.shared_expert_gate is None:
        return "no gated shared expert"
    if mlp.enable_shared_expert_fusion:
        return "the shared expert is fused into the routed experts"
    if not _is_cuda or _use_aiter or use_intel_amx_backend(mlp.shared_expert_gate):
        return "not the CUDA path"
    if mlp.reduce_results:
        return "the block sums its own output"
    if _SGLANG_EXPERIMENTAL_LORA_OPTI:
        return "the LoRA shared-add staging"
    if _hooked(mlp) or _hooked(mlp.experts):
        return "module hooks"
    return ""


def experts_refusal(experts) -> str:
    from sglang.srt.layers.moe import get_moe_runner_backend
    from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl import (
        fused_experts_none_to_flashinfer_cutedsl_fp4,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatcher
    from sglang.srt.layers.quantization.modelopt_quant import (
        ModelOptNvFp4FusedMoEMethod,
    )

    method = experts.quant_method
    if type(method) is not ModelOptNvFp4FusedMoEMethod:
        return "experts not ModelOpt NVFP4"
    backend = getattr(method, "_moe_runner_backend", get_moe_runner_backend())
    if not backend.is_flashinfer_cutedsl():
        return "routed runner not flashinfer_cutedsl"
    runner = getattr(method, "runner", None)
    if (
        getattr(experts, "_mega_moe_nvfp4", False)
        or getattr(method, "_is_cutedsl_v1_deepep", False)
        or getattr(runner, "fused_func", None)
        is not fused_experts_none_to_flashinfer_cutedsl_fp4
        or runner.lora_enabled
    ):
        return "not the CuTe DSL standard path"
    if (
        experts._use_ascend_fuseep
        or experts._dwdp_bound
        or type(experts.dispatcher) is not StandardDispatcher
        or (
            experts.reduce_results
            and (experts.moe_tp_size > 1 or experts.moe_ep_size > 1)
        )
    ):
        return "the experts dispatch or reduce"
    return ""


def ffn_input_refusal(layer) -> str:
    return _entry_refusal(
        layer.ffn_boundary, getattr(layer, "post_attention_layernorm", None), "FFN"
    )


def attn_input_refusal(layer) -> str:
    from sglang.srt.layers.layer_boundary.construction import BatchVariant
    from sglang.srt.layers.layer_boundary.prepare import _attn_input_default

    stage = layer.attn_boundary
    why = _entry_refusal(stage, getattr(layer, "input_layernorm", None), "attention")
    if why:
        return why
    plan = stage.plan
    if (
        plan.paths.get(BatchVariant.SEQUENCE_PARALLEL) is not None and plan.enters_stack
    ) or plan.paths[BatchVariant.ORDINARY].entry.attn_input_adapter not in (
        None,
        _attn_input_default,
    ):
        return "its attention input adapts more than the mirrored path"
    return ""


def _entry_refusal(stage, layer_norm, what: str) -> str:
    from sglang.srt.layers.layer_boundary.construction import BatchVariant
    from sglang.srt.layers.layer_boundary.prepare import _run_entry, _update_read
    from sglang.srt.layers.layer_boundary.residual.add_norm import (
        NormQuantReadout,
        NormReadout,
    )
    from sglang.srt.layers.layernorm import GemmaRMSNorm

    plan = stage.plan
    if type(plan.norm) is not GemmaRMSNorm or plan.norm is not layer_norm:
        return f"its {what} input norm is not the layer's GemmaRMSNorm"
    why = norm_refusal(plan.norm)
    if why:
        return why
    if (
        plan.fusions is not None
        or plan.qkv_latent_func is not None
        or plan._publish_lora_layout
        or stage.declaration.row_transport is not None
    ):
        return f"its {what} input fuses or moves more than the mirrored path"
    path = plan.paths.get(BatchVariant.ORDINARY)
    entry = path.entry if path is not None else None
    prepare = entry.prepare if entry is not None else None
    if (
        not isinstance(prepare, partial)
        or prepare.func is not _run_entry
        or prepare.args
        or entry.input_move is not None
        or (what == "FFN" and entry.attn_input_adapter is not None)
    ):
        return f"its {what} input is not the mirrored entry"
    kw = prepare.keywords
    read, step = kw.get("read"), kw.get("step")
    if (
        kw.get("is_plain_add") is not True
        or kw.get("expected_sum") is not None
        or type(read) not in (NormReadout, NormQuantReadout)
        or read.bind is not None
    ):
        return f"its {what} input completes or reads more than a plain norm"
    if (
        not isinstance(step, partial)
        or step.func is not _update_read
        or step.args
        or set(step.keywords) != {"pre_move", "enters_stack", "read"}
        or step.keywords["pre_move"] is not None
        or step.keywords["enters_stack"]
        or step.keywords["read"] is not read
    ):
        return f"its {what} input step is not the residual add and norm read"
    return ""


def add_norm_refusal(layers, i: int, ids) -> str:
    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

    mlp = getattr(layers[i], "mlp", None)
    if not isinstance(mlp, Qwen2MoeSparseMoeBlock):
        return "not a sparse MoE block"
    why = block_refusal(mlp)
    if why:
        return why
    if i + 1 not in ids:
        return "the stage's last layer"
    following = layers[i + 1]
    if not isinstance(getattr(following, "mlp", None), Qwen2MoeSparseMoeBlock):
        return "the next layer is not a sparse MoE layer"
    if _hooked(layers[i]) or _hooked(following):
        # the body runs both layers' forward without their module call
        return "module hooks"
    if getattr(following, "_is_layer_to_capture", False):
        return "the next layer captures its input"
    why = attn_input_refusal(following)
    return f"next layer: {why}" if why else ""


def norm_refusal(norm) -> str:
    from sglang.kernels.fused_op import get_fused_op_backend

    if _hooked(norm):
        return "its norm has module hooks"
    if get_fused_op_backend() is not None:
        return "a forced kernel backend"
    method = norm._forward_method or norm._resolve_forward_method()
    if norm.is_torch_compile or method != norm.forward_cuda:
        return "its norm does not run SGLang's CUDA Gemma norm"
    return ""


def _global_scale(layers: Layers, layer_id: int, experts):
    gs = layers.scales.get(layer_id)
    if gs is None:
        from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl import (
            ensure_cutedsl_wrapper,
        )

        ensure_cutedsl_wrapper(experts)
        scale = getattr(experts, "_cutedsl_input_scale", None)
        method = experts.quant_method
        gs = (
            scale
            if (
                experts._cutedsl_wrapper.quant_mode == "w4a4"
                and not method.quant_config.use_per_token_activation
                and isinstance(scale, torch.Tensor)
                and scale.numel() == 1
                and scale.dtype == torch.float32
                and scale.is_cuda
            )
            else False
        )
        layers.scales[layer_id] = gs
        if gs is False:
            _once(
                ("scale", layer_id),
                "NVIDIA megakernel: Qwen3.5 slot %s uses stock on layer %d (the "
                "runner takes no static NVFP4 input)",
                SLOT,
                layer_id,
            )
    return gs if gs is not False else None


def ffn_input(layers: Layers, i: int, layer, hidden, fb):
    from sglang.srt.layers.layer_boundary.construction import BatchVariant
    from sglang.srt.layers.layer_boundary.residual.batch import stream_of

    from sgl_nvidia_mega.qwen3_5.sm103.moe import prefill_input_nvfp4 as kernel

    plan = layer.ffn_boundary.plan
    stream = stream_of(fb)
    pending = stream.pending
    if (
        pending is None
        or pending.owed is not None
        or pending.value is not hidden
        or not pending.update.is_plain_add
        or plan.variant_for(fb) is not BatchVariant.ORDINARY
        or plan.norm.is_torch_compile
    ):
        return None
    residual = stream.residual
    if residual is None or not kernel.supported(hidden, residual):
        return None
    gs = _global_scale(layers, i, layer.mlp.experts)
    if gs is None:
        return None
    x, residual = stream.input(hidden)
    norm = plan.norm
    q, sf = kernel.gemma_fused_add_rmsnorm_nvfp4(
        x, residual, norm.weight.data, norm.variance_epsilon, gs
    )
    stream.write(residual)
    _once("engaged", "NVIDIA megakernel: Qwen3.5 slot %s runs", SLOT)
    return x, (q, sf)


def forward(mlp, hidden_states, forward_batch, pre, noclone=False):
    from sglang.kernels.ops.elementwise.elementwise import fused_gate_sigmoid_mul_add

    num_tokens, hidden_dim = hidden_states.shape
    if num_tokens == 0:
        return mlp(hidden_states, forward_batch)
    hidden_states, shared_output, final_hidden_states = _block(
        mlp, hidden_states, pre, noclone
    )
    fused_gate_sigmoid_mul_add(
        hidden_states,
        mlp.shared_expert_gate.weight.squeeze(),
        shared_output,
        final_hidden_states,
    )
    return final_hidden_states.view(num_tokens, hidden_dim)


def forward_owing(mlp, hidden_states, forward_batch, pre, noclone=False):
    num_tokens, hidden_dim = hidden_states.shape
    w = mlp.shared_expert_gate.weight
    if not (
        num_tokens > 0
        and hidden_states.dtype == w.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
        and w.numel() == hidden_dim
    ):
        return forward(mlp, hidden_states, forward_batch, pre, noclone), None
    hidden_states, shared_output, final_hidden_states = _block(
        mlp, hidden_states, pre, noclone
    )
    if not (
        final_hidden_states.shape == shared_output.shape == hidden_states.shape
        and final_hidden_states.dtype == shared_output.dtype == torch.bfloat16
        and final_hidden_states.is_contiguous()
        and shared_output.is_contiguous()
    ):
        from sglang.kernels.ops.elementwise.elementwise import (
            fused_gate_sigmoid_mul_add,
        )

        fused_gate_sigmoid_mul_add(
            hidden_states, w.squeeze(), shared_output, final_hidden_states
        )
        return final_hidden_states.view(num_tokens, hidden_dim), None
    w = w.squeeze()
    # the rows the block returns are the ones the next input reads
    out = final_hidden_states.view(num_tokens, hidden_dim)
    owed = Owed(
        routed=out,
        shared=shared_output,
        gate=gate_fp32(hidden_states, w),
        x=hidden_states,
        w=w,
    )
    return out, owed


def _block(mlp, hidden_states, pre, noclone=False):
    from sglang.srt.model_executor.runner import get_is_capture_mode
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
        is_in_breakable_cuda_graph,
    )
    from sglang.srt.runtime_context import get_exec

    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    if (
        mlp.alt_stream is not None
        and get_is_capture_mode()
        and not torch.compiler.is_compiling()
        and not (get_exec().moe.enable_paged_experts and is_in_breakable_cuda_graph())
    ):
        current_stream = torch.cuda.current_stream()
        mlp.alt_stream.wait_stream(current_stream)
        shared_output = mlp._forward_shared_experts(
            hidden_states if noclone else hidden_states.clone(), apply_gate=False
        )
        with torch.cuda.stream(mlp.alt_stream):
            final_hidden_states = _router_experts(mlp, hidden_states, pre)
        current_stream.wait_stream(mlp.alt_stream)
    else:
        shared_output = mlp._forward_shared_experts(hidden_states, apply_gate=False)
        final_hidden_states = _router_experts(mlp, hidden_states, pre)
    return hidden_states, shared_output, final_hidden_states


class Owed(NamedTuple):
    routed: torch.Tensor
    shared: torch.Tensor
    gate: torch.Tensor
    x: torch.Tensor
    w: torch.Tensor


def gate_fp32(x, w):
    import triton
    from sglang.kernels.ops.elementwise.elementwise import (
        _fused_gate_sigmoid_mul_add_kernel,
        _is_hip,
        is_arch_support_pdl,
        is_batch_invariant_mode_enabled,
    )
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    num_tokens, hidden_dim = x.shape
    max_warps = 16 if _is_hip else 32
    config = {
        "BLOCK_SIZE": triton.next_power_of_2(hidden_dim),
        "num_warps": max(
            min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), max_warps), 4
        ),
    }
    if guard_or_false(num_tokens >= 1024) or is_batch_invariant_mode_enabled():
        config["num_warps"] = min(config["num_warps"], 8)
    use_pdl = is_arch_support_pdl()
    pdl_kwargs = {"launch_pdl": True} if use_pdl else {}
    g = torch.empty((num_tokens + 15) // 16 * 16, dtype=torch.float32, device=x.device)
    g = g[:num_tokens]
    _fused_gate_sigmoid_mul_add_kernel[(num_tokens,)](
        x,
        w,
        None,
        g,
        hidden_dim=hidden_dim,
        DO_ADD=False,
        USE_PDL=use_pdl,
        GATE_ONLY=True,
        **config,
        **pdl_kwargs,
    )
    return g


def complete(owed: Owed) -> None:
    from sglang.kernels.ops.elementwise.elementwise import fused_gate_sigmoid_mul_add

    fused_gate_sigmoid_mul_add(owed.x, owed.w, owed.shared, owed.routed)


def _sum_is_identity(group) -> bool:
    """Whether ``sum_output`` over ``group`` returns its input (a one-rank group)."""
    from sglang.srt.layers.layer_boundary.layout import SumGroup
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    if group is SumGroup.ATTN_TP:
        return parallel.attn_tp_size == 1
    if group is SumGroup.TP:
        return parallel.tp_size == 1
    if group is SumGroup.MOE_OUTPUT:
        return parallel.moe_ep_size == 1 and parallel.moe_tp_size == 1
    return False


def exit_keeps_output(ffn_exit, forward_batch) -> bool:
    from sglang.srt.layers.layer_boundary.exit import FfnExit
    from sglang.srt.layers.layer_boundary.ops import keep_output

    if type(ffn_exit) is not FfnExit:
        return False
    complete_now = ffn_exit._complete
    steps = ffn_exit.boundary.plan.path_for(forward_batch)
    update = ffn_exit._update
    return (
        isinstance(complete_now, partial)
        and complete_now.func == ffn_exit.boundary._complete_now
        and complete_now.keywords.get("dp_step") is None
        and (
            not complete_now.keywords.get("owes_sum", True)
            or _sum_is_identity(steps.output.group)
        )
        and not ffn_exit.defer_moe_finalize
        and ffn_exit._declared_sum is None
        and steps.output.sum_impl is None
        and steps.output.transform is None
        and not steps.returns_over_dp
        and steps.output_move is keep_output
        and not steps.writes_at_handoff
        and update.is_plain_add
        and not update.applied_at_exit
    )


def layer_input(layer, hidden, fb, owed: Owed, fp8=None):
    from sglang.srt.layers.layer_boundary.construction import BatchVariant
    from sglang.srt.layers.layer_boundary.residual.batch import stream_of

    from sgl_nvidia_mega.qwen3_5.sm103.moe import prefill_output_add_norm as kernel

    plan = layer.attn_boundary.plan
    stream = stream_of(fb)
    pending = stream.pending
    if (
        hidden is not owed.routed
        or pending is None
        or pending.owed is not None
        or pending.value is not hidden
        or not pending.update.is_plain_add
        or plan.variant_for(fb) is not BatchVariant.ORDINARY
        or plan.norm.is_torch_compile
    ):
        return None
    residual = stream.residual
    if residual is None or not kernel.supported(
        owed.routed, owed.shared, owed.gate, residual
    ):
        return None
    x, residual = stream.input(hidden)
    norm = plan.norm
    scale = fp8.norm_scale(norm, x) if fp8 is not None else None
    q = kernel.gated_add_rmsnorm(
        x,
        owed.shared,
        owed.gate,
        residual,
        norm.weight.data,
        norm.variance_epsilon,
        fp8_scale=scale,
    )
    out = x
    entry = plan.path_for(fb).entry
    if entry.attn_input_adapter is not None:
        out = entry.attn_input_adapter(x, fb, plan.qkv_latent_func)
    stream.write(residual)
    if q is not None and out is x:
        fp8.norm_hand_off(norm, x, q)
    _once("engaged-add-norm", "NVIDIA megakernel: Qwen3.5 slot %s runs", SLOT_ADD_NORM)
    return out


def _router_experts(mlp, hidden_states, pre):
    router_logits, _ = mlp.gate(hidden_states)
    topk_output = mlp.topk(hidden_states, router_logits)
    return routed(mlp.experts, hidden_states, topk_output, pre)


def routed(experts, hidden_states, topk_output, pre):
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        use_symmetric_memory,
    )
    from sglang.srt.layers.dp_attention import is_allocation_symmetric
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        is_in_tc_piecewise_cuda_graph,
    )
    from sglang.srt.runtime_context import get_parallel

    if pre is None or is_in_tc_piecewise_cuda_graph():
        return experts(hidden_states, topk_output)
    origin_hidden_states_dim = hidden_states.shape[-1]
    dispatch_output = experts.dispatcher.dispatch(
        hidden_states=hidden_states, topk_output=topk_output
    )
    if (
        dispatch_output.format.is_standard()
        and dispatch_output.hidden_states is hidden_states
        and dispatch_output.hidden_states_scale is None
        and TopKOutputChecker.format_is_standard(dispatch_output.topk_output)
    ):
        combine_input = _cutedsl_standard(experts, dispatch_output, pre)
    else:
        combine_input = experts.run_moe_core(dispatch_output=dispatch_output)
    with use_symmetric_memory(
        get_parallel().tp_group, disabled=not is_allocation_symmetric()
    ):
        final_hidden_states = experts.dispatcher.combine(combine_input=combine_input)
        final_hidden_states = final_hidden_states[
            ..., :origin_hidden_states_dim
        ].contiguous()
    return final_hidden_states


def _cutedsl_standard(experts, dispatch_output, pre):
    from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl import (
        ensure_cutedsl_wrapper,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    ensure_cutedsl_wrapper(experts)
    w1_alpha, fc2_input_scale, w2_alpha = experts._cutedsl_scales
    topk_output = dispatch_output.topk_output
    topk_ids = topk_output.topk_ids
    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)
    seq_len, hidden_size = dispatch_output.hidden_states.shape
    x_fp4, x_sf = pre
    x_fp4 = x_fp4.reshape(seq_len, hidden_size // 2)
    x_sf = x_sf.view(torch.float8_e4m3fn).reshape(seq_len, hidden_size // SF_VEC)
    output = experts._cutedsl_wrapper.run(
        x=x_fp4,
        x_sf=x_sf,
        token_selected_experts=topk_ids,
        token_final_scales=topk_output.topk_weights,
        w1_weight=experts.w13_weight,
        w1_weight_sf=getattr(
            experts, "w13_blockscale_mma", experts.w13_blockscale_swizzled
        ),
        w1_alpha=w1_alpha,
        fc2_input_scale=fc2_input_scale,
        w2_weight=experts.w2_weight,
        w2_weight_sf=getattr(
            experts, "w2_blockscale_mma", experts.w2_blockscale_swizzled
        ),
        w2_alpha=w2_alpha,
        per_token_scale=None,
    )
    return StandardCombineInput(hidden_states=output)
