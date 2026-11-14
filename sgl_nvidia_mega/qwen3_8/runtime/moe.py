# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/qwen2_moe.py (Qwen2MoeSparseMoeBlock.forward), and from the
# source's python/sglang/srt/models/qwen2_moe.py (forward_normal_dual_stream and the eager
# branch of forward at extend sizes) and python/sglang/kernels/ops/elementwise/
# elementwise.py (fused_moe_finalize_gate_sigmoid_mul_add's launch). Changed: the
# finalize launches the plugin's copy of the kernel; the extend switches are constants.
"""The MoE step of a covered forward.

Slot ``moe_finalize`` (TP1, no EP): the router and the routed experts run SGLang's
FlashInfer TRT-LLM MoE with its finalize deferred, the shared expert runs ungated, and
one plugin kernel (``_fused_finalize_gate_sigmoid_mul_add_kernel``) writes
``sigmoid(x . g) * shared + sum_k w_k * expert_k`` from the un-finalized expert rows
(``gemm2_out`` and its permutation). Below ``EXTEND_MIN_ROWS`` rows the step keeps
SGLang's deferred dual stream; from it on it is the source's extend-size MoE
(``_extend``), in breakable prefill graphs too. Stock: FlashInfer's finalize, then
``fused_gate_sigmoid_mul_add``. Every other case runs SGLang's
``Qwen2MoeSparseMoeBlock``."""

from __future__ import annotations

import contextlib
import os

import torch

from sgl_nvidia_mega.qwen3_8.runtime import bf16_gemm_pins, moe_routing

STOCK = "stock"
HIDDEN, TOP_K = 2560, 10
# The source's extend-size finalize at its recipes' defaults: from this many rows the
# extend order, and a launch capped at 64 registers (3 -> 4 CTAs per SM, same math).
EXTEND_MIN_ROWS = 1024
EXTEND_MAX_REGISTERS = 64
# Below it, the source's launch (fused_moe_finalize_gate_sigmoid_mul_add): its warp rule
# max(min(next_pow2(ceil(hidden / 256)), 32), 4) at hidden 2560, PDL, and up to 64 rows the
# (row, chunk-column) grid when the recipe sets the source's column chunk (0: off).
FINALIZE_NUM_WARPS = 16
FINALIZE_COLUMN_CHUNK_ENV = "SGLANG_MOE_FINALIZE_GATE_COLUMN_CHUNK"
FINALIZE_COLUMN_CHUNK = int(os.environ.get(FINALIZE_COLUMN_CHUNK_ENV) or 0)
if FINALIZE_COLUMN_CHUNK not in (0, 128, 256, 512, 1024):
    raise ValueError(
        f"{FINALIZE_COLUMN_CHUNK_ENV} must be 0 or one of (128, 256, 512, 1024), "
        f"got {FINALIZE_COLUMN_CHUNK}"
    )
FINALIZE_COLUMN_CHUNK_MAX_ROWS = 64
# The source's SGLANG_QWEN_MOE_SHARED_NO_CLONE under its recipes' name here: the captured
# dual stream's shared expert reads the block input itself instead of SGLang's clone of it.
SHARED_NO_CLONE_ENV = "SGLANG_OPT_MOE_SHARED_EXPERT_NO_CLONE"
SHARED_NO_CLONE = os.environ.get(SHARED_NO_CLONE_ENV, "false").lower() in ("true", "1")

SGLANG_READS = {
    "srt/models/qwen2_moe.py": (
        "Qwen2MoeSparseMoeBlock",
        "_forward_shared_experts",
        "shared_expert",
        "_forward_router_experts",
        "supports_deferred_finalize",
        "shared_expert_gate",
        "alt_stream",
        "TopKOutputChecker",
        "gate",
        "topk",
        "experts",
    ),
    "srt/layers/moe/topk.py": ("format_is_bypassed",),
    "srt/layers/moe/fused_moe_triton/layer.py": (
        "forward_deferred_finalize",
        "FusedMoE",
        "moe_runner_config",
        "inplace",
    ),
    "srt/model_executor/runner/__init__.py": ("get_is_capture_mode",),
    "srt/layers/moe/moe_runner/flashinfer_trtllm.py": (
        "gemm2_out",
        "expanded_idx_to_permuted_idx",
        "expert_weights",
        "top_k",
    ),
    "kernels/jit/utils/__init__.py": ("is_arch_support_pdl",),
    "srt/batch_invariant_ops/__init__.py": ("is_batch_invariant_mode_enabled",),
}


# With BF16 pins on, _router_experts and _shared_experts mirror these
SGLANG_MIRRORS = {
    "srt/models/qwen2_moe.py:Qwen2MoeSparseMoeBlock._forward_router_experts": "251e3d343aec92fe",
    "srt/models/qwen2_moe.py:Qwen2MoeSparseMoeBlock._forward_shared_experts": "3b55a78638831ffa",
}

# Reads and mirrors that need an SGLang change newer than the unit job's pin: the
# contract test skips each, naming the change, only where the installed SGLang lacks it.
SGLANG_REQUIRES = {
    "srt/models/qwen2_moe.py:Qwen2MoeSparseMoeBlock._forward_router_experts": (
        "sglang#48366",
        ("32d09f01b77d76b5",),
    ),
}


def layer_reasons(layer) -> str:
    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

    mlp = layer.mlp
    if not isinstance(mlp, Qwen2MoeSparseMoeBlock):
        return "not a sparse MoE block"
    if not mlp.supports_deferred_finalize:
        return "the MoE runner cannot defer its finalize"
    if mlp.shared_expert_gate is None:
        return "no shared-expert gate"
    if mlp.shared_expert_gate.weight.dtype != torch.bfloat16:
        return "shared-expert gate is not BF16"
    return ""


def build(parallel) -> str:
    if parallel.tp_size != 1 or getattr(parallel, "moe_ep_size", 1) != 1:
        return "the finalize kernel takes TP1 without expert parallelism"
    try:
        from sgl_nvidia_mega.qwen3_8.kernels.ops.elementwise import (  # noqa: F401
            moe_finalize,
        )
    except Exception as e:
        return f"finalize kernel does not import: {type(e).__name__}: {e}"
    return ""


def _side_stream(mlp):
    """The block's side stream while a graph is captured, as SGLang uses it; else None."""
    from sglang.srt.model_executor.runner import get_is_capture_mode

    return (
        mlp.alt_stream if mlp.alt_stream is not None and get_is_capture_mode() else None
    )


def step(executor, layer, x, st):
    """The MoE output [rows, hidden] (TP1: nothing left to reduce)."""
    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

    mlp = layer.mlp
    rows = x.shape[0]
    if not (executor.moe_finalize_ready and rows > 0 and x.shape[1] == HIDDEN):
        st.ran("moe", STOCK)
        if isinstance(mlp, Qwen2MoeSparseMoeBlock):
            return mlp(x, st.fb, defer_finalize=False)
        return mlp(x)
    if rows >= EXTEND_MIN_ROWS:
        return _extend(executor, mlp, x, st)
    side = _side_stream(mlp)
    options = executor.moe_routing
    if side is None:
        # The source's eager order: the shared expert, then the routed experts.
        shared = _shared(executor, mlp, x, st)
        deferred = moe_routing.routed(executor, mlp, x, st)
        if deferred is None:
            deferred = _router_experts(mlp, x)
    else:
        # The source's forward_normal_dual_stream: the quant (when the fused routing does not
        # quantize) and the shared expert on the current stream, the router GEMM, routing and
        # routed experts on the side stream, no wait between the two before the finalize.
        current = torch.cuda.current_stream()
        side.wait_stream(current)
        fused = options is not None and moe_routing._takes_fused(options, mlp, x)
        quant = None
        if options is not None and not fused and options.prequant_overlap:
            quant = moe_routing.quantize(mlp, x)
        shared = _shared(executor, mlp, _shared_input(mlp, x, st), st)
        with torch.cuda.stream(side):
            if fused or quant is not None:
                logits = bf16_gemm_pins.linear(mlp.gate, x)
                if fused:
                    deferred = moe_routing.fused_from_logits(mlp, x, logits, st)
                else:
                    deferred = moe_routing.deferred(mlp, x, logits, quant)
                    st.ran(moe_routing.OP, "prequant")
            else:
                deferred = _router_experts(mlp, x)
        current.wait_stream(side)
        for t in (
            deferred.gemm2_out,
            deferred.expert_weights,
            deferred.expanded_idx_to_permuted_idx,
        ):
            t.record_stream(current)
    from sglang.kernels.jit.utils import is_arch_support_pdl
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled

    pdl = is_arch_support_pdl()
    out = _finalize(
        mlp,
        x,
        shared,
        deferred,
        chunk=FINALIZE_COLUMN_CHUNK if rows <= FINALIZE_COLUMN_CHUNK_MAX_ROWS else 0,
        USE_PDL=pdl,
        # the source caps the warps at 8 in batch-invariant mode
        num_warps=8 if is_batch_invariant_mode_enabled() else FINALIZE_NUM_WARPS,
        **({"launch_pdl": True} if pdl else {}),
    )
    st.ran("moe", "finalize_permuted")
    return out


def _extend(executor, mlp, x, st):
    """The source's extend-size MoE. Capturing (its forward_normal_dual_stream): the
    shared expert on the current stream, the router and the routed experts on the
    block's side stream, with slot moe_routing the routed input quantized on the current
    stream first. Eager (its forward): the shared, then the routed experts. Then the
    finalize at the source's extend launch: 8 warps, the register cap, PDL."""
    from sglang.kernels.jit.utils import is_arch_support_pdl

    side = _side_stream(mlp)
    options = executor.moe_routing
    quant = None
    if side is not None:
        current = torch.cuda.current_stream()
        side.wait_stream(current)
        if options is not None and options.prequant_overlap:
            quant = moe_routing.quantize(mlp, x)
    shared = _shared_experts(mlp, x)
    with torch.cuda.stream(side) if side is not None else contextlib.nullcontext():
        if quant is None:
            deferred = _router_experts(mlp, x)
        else:
            logits = bf16_gemm_pins.linear(mlp.gate, x)
            deferred = moe_routing.deferred(mlp, x, logits, quant)
            st.ran(moe_routing.OP, "prequant")
    if side is not None:
        current.wait_stream(side)
        for t in (
            deferred.gemm2_out,
            deferred.expert_weights,
            deferred.expanded_idx_to_permuted_idx,
        ):
            t.record_stream(current)
    pdl = is_arch_support_pdl()
    out = _finalize(
        mlp,
        x,
        shared,
        deferred,
        USE_PDL=pdl,
        num_warps=8,
        maxnreg=EXTEND_MAX_REGISTERS,
        **({"launch_pdl": True} if pdl else {}),
    )
    st.ran("moe", "extend_finalize")
    return out


def _shared_input(mlp, x, st):
    """The source's _shared_expert_input: ``x`` itself with SHARED_NO_CLONE when the
    routed runner never writes its input in place, else SGLang's clone. The clone's copy
    node starts the shared expert after the router GEMM; without it they start together."""
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    if (
        SHARED_NO_CLONE
        and isinstance(mlp.experts, FusedMoE)
        and not mlp.experts.moe_runner_config.inplace
    ):
        st.ran("moe.shared_input", "no_clone")
        return x
    st.ran("moe.shared_input", STOCK)
    return x.clone()


def _shared(executor, mlp, x, st):
    """The ungated shared expert: slot smallm_gemm's where it serves, else SGLang's."""
    from sgl_nvidia_mega.qwen3_8.runtime import smallm_gemm

    shared = smallm_gemm.shared_expert(executor, mlp.shared_expert, x, st)
    if shared is None:
        shared = _shared_experts(mlp, x)
    return shared.contiguous()


def _finalize(mlp, x, shared, deferred, chunk=0, **launch):
    """The finalize kernel on the deferred output, one row a program; with ``chunk``,
    one (row, ``chunk`` columns) a program."""
    from sgl_nvidia_mega.qwen3_8.kernels.ops.elementwise import moe_finalize

    rows = x.shape[0]
    top_k = int(deferred.top_k)
    gemm2 = deferred.gemm2_out.view(-1, deferred.gemm2_out.shape[-1])
    perm = deferred.expanded_idx_to_permuted_idx.view(-1, top_k)[:rows].contiguous()
    weights = deferred.expert_weights.view(-1, top_k)[:rows].contiguous()
    if top_k != TOP_K or perm.dtype != torch.int32:
        raise RuntimeError(
            f"moe_finalize: deferred output layout differs (top_k {top_k})"
        )
    out = torch.empty_like(x)
    if chunk:
        kernel = moe_finalize._fused_finalize_gate_sigmoid_mul_add_chunked_kernel
        grid, launch = (rows, -(-HIDDEN // chunk)), dict(launch, CHUNK=chunk)
    else:
        kernel = moe_finalize._fused_finalize_gate_sigmoid_mul_add_kernel
        grid = (rows,)
    kernel[grid](
        x.contiguous(),
        mlp.shared_expert_gate.weight.view(-1),
        shared.contiguous(),
        out,
        gemm2,
        perm,
        weights,
        gemm2.stride(0),
        hidden_dim=HIDDEN,
        TOP_K=TOP_K,
        BLOCK_SIZE=4096,
        **launch,
    )
    return out


def _router_experts(mlp, x):
    """``mlp._forward_router_experts(x, defer_finalize=True)``; with BF16 pins on, here
    with the router GEMM through ``bf16_gemm_pins``."""
    if not bf16_gemm_pins.routed():
        return mlp._forward_router_experts(x, defer_finalize=True)
    from sglang.srt.models.qwen2_moe import TopKOutputChecker

    router_logits = bf16_gemm_pins.linear(mlp.gate, x)
    topk_output = mlp.topk(x, router_logits)
    if not mlp.supports_deferred_finalize:
        raise RuntimeError(
            "Qwen deferred finalize requires a compatible FlashInfer "
            "TRTLLM MoE producer and a separate shared expert"
        )
    if not TopKOutputChecker.format_is_bypassed(topk_output):
        raise RuntimeError("Qwen deferred finalize requires logits-based bypassed TopK")
    return mlp.experts.forward_deferred_finalize(x, topk_output)


def _shared_experts(mlp, x):
    """``mlp._forward_shared_experts(x, apply_gate=False)``; with BF16 pins on, the
    shared expert's projections through ``bf16_gemm_pins``."""
    if not bf16_gemm_pins.routed():
        return mlp._forward_shared_experts(x, apply_gate=False)
    if mlp.shared_expert is None:
        return None
    return bf16_gemm_pins.mlp(mlp.shared_expert, x)
