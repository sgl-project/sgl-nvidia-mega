# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/qwen2_moe.py (Qwen2MoeSparseMoeBlock.forward and its dual-stream path).
"""The sparse MoE of a covered step, over SGLang's loaded ``Qwen2MoeSparseMoeBlock``.

Stock: what ``Qwen2MoeSparseMoeBlock.forward`` issues on CUDA with no expert
parallelism, no all-to-all and no paged experts (SGLang pages experts on one GPU only),
calling the block's own router gate, top-k, routed
experts and gated shared expert. At CUDA graph capture it takes the dual-stream order
(with a deferred finalize: routed experts on the main stream, shared expert on the
alternate stream), otherwise the eager order. With ``defer`` (the boundary takes the
finalize) it returns SGLang's ``MoeDeferredFinalize`` handoff, otherwise the summed
rows. Slot ``route`` (``route.py``) runs the deferred routed experts on the router
front, and slot ``mkm`` (``mkm.py``) runs them in one launch on its outputs. In the
dual-stream order of up to 8 rows, slot ``route_shared`` computes the shared expert's
gate/up in the router front's launch; the alternate stream waits for that launch only and
runs the rest of the shared expert (numeric).
"""

from __future__ import annotations

import torch

from sgl_nvidia_mega.qwen3_5.moe import route

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/qwen2_moe.py": (
        "mlp",
        "alt_stream",
        "gate",
        "topk",
        "experts",
        "shared_expert",
        "shared_expert_gate",
        "enable_shared_expert_fusion",
        "supports_deferred_finalize",
        "_forward_shared_experts",
        "_forward_shared_experts(apply_gate)",
        "_gate_shared_output_out_of_place",
        "_append_shared_to_topk_output",
    ),
    "srt/layers/moe/topk.py": (
        "TopKOutputChecker",
        "format_is_bypassed",
        "format_is_standard",
    ),
    "srt/layers/moe/fused_moe_triton/layer.py": ("forward_deferred_finalize",),
    "srt/layers/moe/utils.py": ("sum_post_experts_output",),
    "srt/layers/layer_boundary/fusions/moe_finalize.py": (
        "MoeDeferredFinalize",
        "from_flashinfer",
        "from_flashinfer(gated_shared_output)",
        "from_flashinfer(m)",
        "from_flashinfer(reduce)",
    ),
    "kernels/ops/elementwise/elementwise.py": ("fused_gate_sigmoid_mul_add",),
    "srt/model_executor/runner/__init__.py": ("get_is_capture_mode",),
}

# The SGLang functions whose body this module mirrors, by the digest of their code
# (test/unit/test_contract.py recomputes them on the installed SGLang)
SGLANG_MIRRORS = {
    "srt/models/qwen2_moe.py:Qwen2MoeSparseMoeBlock.forward": "633ff9a686d3ea28",
    "srt/models/qwen2_moe.py:Qwen2MoeSparseMoeBlock.forward_normal_dual_stream": "afae959064cc8cd7",
    "srt/models/qwen2_moe.py:Qwen2MoeSparseMoeBlock._forward_router_experts": "32d09f01b77d76b5",
}


def build(model, ctx) -> route.Bound:
    """Slots ``route`` and ``mkm`` for every MoE layer; the result is
    ``st.bound["moe"]``."""
    from sgl_nvidia_mega.qwen3_5 import plan
    from sgl_nvidia_mega.qwen3_5.runtime import SLOTS

    if "route" not in SLOTS:
        return route.Bound(reason="stock (SGLANG_NVIDIA_MEGAKERNEL_SLOTS)")
    rows = plan.widest(ctx.widths, plan.ROUTE_MAX_ROWS)
    if not rows:
        return route.Bound(reason=f"no width of {list(ctx.widths)} within its rows")
    return route.build(
        model,
        rows,
        fused="mkm" in SLOTS,
        shared="route_shared" in SLOTS,
        wide="mkm_wide" in SLOTS,
    )


def step(layer, x: torch.Tensor, st, defer: bool):
    """``layer``'s MoE on the normed input ``x`` [rows, hidden]: the deferred-finalize
    handoff with ``defer``, else the unreduced rows [rows, hidden]."""
    from sglang.kernels.ops.elementwise.elementwise import fused_gate_sigmoid_mul_add
    from sglang.srt.layers.layer_boundary.fusions.moe_finalize import (
        MoeDeferredFinalize,
    )
    from sglang.srt.layers.moe.utils import sum_post_experts_output
    from sglang.srt.model_executor.runner import get_is_capture_mode

    mlp = layer.mlp
    rows, hidden = x.shape
    x = x.view(-1, hidden)
    # the gate is a plain linear on CUDA (no AMX, no NPU)
    use_fused_gate = mlp.shared_expert_gate is not None
    if (
        mlp.alt_stream is not None
        and get_is_capture_mode()
        and not torch.compiler.is_compiling()
    ):
        routed, shared = _dual_stream(mlp, x, use_fused_gate, defer, st)
    else:
        shared = mlp._forward_shared_experts(
            x, apply_gate=not use_fused_gate and not defer
        )
        if defer and shared is not None:
            shared = mlp._gate_shared_output_out_of_place(x, shared)
        routed = _router_experts(mlp, x, defer, st)
    if defer:
        if shared is None:
            raise RuntimeError("Qwen deferred finalize requires shared output")
        return MoeDeferredFinalize.from_flashinfer(
            routed,
            gated_shared_output=shared,
            m=rows,
            reduce=sum_post_experts_output,
        )
    if shared is not None:
        if use_fused_gate:
            fused_gate_sigmoid_mul_add(
                x, mlp.shared_expert_gate.weight.squeeze(), shared, routed
            )
        else:
            routed += shared
    return routed.view(rows, hidden)


def _dual_stream(mlp, x, use_fused_gate: bool, defer: bool, st):
    """``Qwen2MoeSparseMoeBlock.forward_normal_dual_stream`` (no LoRA): (routed,
    shared)."""
    current = torch.cuda.current_stream()
    mlp.alt_stream.wait_stream(current)
    if defer and st.slots.route_shared:
        # the router launch also computes the shared gate/up; the alternate stream
        # waits for that launch only
        routed, gate_up, ready = route.routed(
            mlp, x, st.bound["moe"] if st.slots.mkm else None, shared=True
        )
        gate_up.record_stream(mlp.alt_stream)
        with torch.cuda.stream(mlp.alt_stream):
            mlp.alt_stream.wait_event(ready)
            shared = route.shared_expert_tail(mlp.shared_expert, gate_up)
            shared = mlp._gate_shared_output_out_of_place(x, shared)
        current.wait_stream(mlp.alt_stream)
        return routed, shared
    if defer:
        # routed FC2 stays on the current stream, the finalize's PDL dependency
        routed = _router_experts(mlp, x, True, st)
        with torch.cuda.stream(mlp.alt_stream):
            shared = mlp._forward_shared_experts(x, apply_gate=False)
            if shared is not None:
                shared = mlp._gate_shared_output_out_of_place(x, shared)
        current.wait_stream(mlp.alt_stream)
        return routed, shared
    shared = (
        mlp._forward_shared_experts(x.clone(), apply_gate=not use_fused_gate)
        if mlp.shared_expert is not None
        else None
    )
    with torch.cuda.stream(mlp.alt_stream):
        routed = _router_experts(mlp, x, False, st)
    current.wait_stream(mlp.alt_stream)
    return routed, shared


def _router_experts(mlp, x, defer: bool, st):
    """``Qwen2MoeSparseMoeBlock._forward_router_experts`` on CUDA: the router gate,
    the top-k and the routed experts (slots ``route`` and ``mkm``: theirs, deferred)."""
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    if defer and st.slots.route:
        return route.routed(mlp, x, st.bound["moe"] if st.slots.mkm else None)
    router_logits, _ = mlp.gate(x)
    topk_output = mlp.topk(x, router_logits)
    if defer:
        if not mlp.supports_deferred_finalize:
            raise RuntimeError(
                "Qwen deferred finalize requires a compatible FlashInfer TRTLLM MoE "
                "producer and a separate shared expert"
            )
        if not TopKOutputChecker.format_is_bypassed(topk_output):
            raise RuntimeError(
                "Qwen deferred finalize requires logits-based bypassed TopK"
            )
        return mlp.experts.forward_deferred_finalize(x, topk_output)
    if mlp.enable_shared_expert_fusion and TopKOutputChecker.format_is_standard(
        topk_output
    ):
        topk_output = mlp._append_shared_to_topk_output(topk_output, x)
    return mlp.experts(x, topk_output)
