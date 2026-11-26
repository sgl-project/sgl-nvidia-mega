# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``moe_tail`` (env ``MUSE_VR_MOE_TAIL=1``, default off).

Stock Muse Spark decode runs the MoE tail of a layer as: finalize (+ its fused all-reduce) -> unpooler projection
-> FFN boundary. For row counts up to SGLang's small-projection limit the projection is a column slice per rank
(``gemm_local``) followed by ``all_gather_norm_residual``; above it (on devices outside that path's capability gate)
the full projection runs replicated on every rank followed by ``fused_norm_residual``.

This component keeps the finalize and its all-reduce, and replaces [projection + boundary] with
``moe_tail_ops.moe_tail``: one kernel computes this rank's column slice with the weight rows staged before the
kernel waits on its predecessor and stores the columns straight into every rank's receive buffer; a second kernel
runs SGLang's cluster boundary arithmetic on the gathered rows. Boundary arithmetic is bitwise identical to SGLang's;
the projection's fp32 summation order differs from SGLang's GEMM (bf16 outputs can differ in the last bit;
``tests/dist_moe_tail.py`` measures the error of both against an fp64 reference).

Cross-rank hand-off. Release / acquire at system scope (CTA barrier, ``fence.acq_rel.sys``, relaxed
counter increments; consumers poll with ``ld.acquire.sys``), as the PTX memory model prescribes for
producer/consumer synchronization between GPUs.

Overlap. The pending tensor and the workspace counters are per process; two-batch or single-batch overlap could run
two layers' tails interleaved on different streams, so the component steps aside whenever either overlap mode is on.

Hooks (AROUND): ``MuseSpark1xMoE._reduce_and_unpool`` returns the all-reduced expert output unprojected and marks it;
``FusedPostNormResidualPreNorm.forward_with_ag`` (the layer calls it whenever the MoE output is narrower than the
stream) runs the fused tail for a marked tensor. The workspace (symmetric receive buffers) is allocated collectively
right after ``MuseSpark1xModel.__init__``, before any CUDA graph is captured. Row range:
``MUSE_VR_MOE_TAIL_MIN_M`` .. ``MUSE_VR_MOE_TAIL_MAX_M``. Defaults 9 .. 16 at tensor-parallel size 4 and above; off by
default below size 4.
"""

from __future__ import annotations

import logging
import os

import torch

from sgl_nvidia_mega.muse_spark.components import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/layers/moe/utils.py": ("is_sbo_enabled", "is_tbo_enabled"),
    "srt/distributed/*.py": ("get_tp_group", "tensor_model_parallel_all_reduce"),
    "srt/distributed/device_communicators/custom_all_reduce_v2.py": (
        "CustomAllReduceV2",
    ),
    "srt/layers/dp_attention.py": ("is_dp_attention_enabled",),
    "srt/runtime_context.py": ("get_forward",),
    "srt/models/muse_spark.py": (
        "MuseSpark1xModel",
        "MuseSpark1xMoE",
        "_reduce_and_unpool",
        "_reduce_and_unpool(already_reduced)",
        "post_expert_proj",
        "unpool_gather_boundary",
        "moe_tp_size",
        "tp_size",
    ),
    "kernels/ops/layernorm/muse_spark_norm_residual.py|kernels/ops/muse_spark_v12/fused_norm_residual.py": (
        "FusedPostNormResidualPreNorm",
        "jit_selected",
        "_ensure_cached_weights",
    ),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}

# The SGLang functions whose body this module mirrors, by the digest of their code (test/unit/test_contract.py
# recomputes them on the installed SGLang). One digest: the same in SGLang cf6133a and 306ccf6.
SGLANG_MIRRORS = {
    # the reduction _around_reduce_and_unpool keeps before it hands the unprojected rows to the fused tail
    "srt/models/muse_spark.py:MuseSpark1xMoE._reduce_and_unpool": "8c38fcfb0523752b",
    # the boundary operands moe_tail passes (the boundary's own kwargs, built the same way)
    "kernels/ops/layernorm/muse_spark_norm_residual.py:FusedPostNormResidualPreNorm.jit_kernel_kwargs": "f3ddcac4283845af",
    # the all-gather boundary whose arithmetic the second kernel reproduces (its kernel source is listed in NOTICE)
    "kernels/ops/communication/muse_spark_all_gather_norm_residual.py:all_gather_norm_residual": "c00da7c4e5cd2fea",
}


logger = logging.getLogger(__name__)

NAME = "moe_tail"  # env MUSE_VR_MOE_TAIL
ENV_MIN_M = "MUSE_VR_MOE_TAIL_MIN_M"
ENV_MAX_M = "MUSE_VR_MOE_TAIL_MAX_M"
DEFAULT_MIN_M = 9
DEFAULT_MAX_M = 16
DEFAULT_MAX_M_TP2 = 0

T_MODEL_INIT = "sglang.srt.models.muse_spark.MuseSpark1xModel.__init__"
T_REDUCE_UNPOOL = "sglang.srt.models.muse_spark.MuseSpark1xMoE._reduce_and_unpool"
T_FORWARD_WITH_AG = "sglang.kernels.ops.muse_spark_v12.fused_norm_residual.FusedPostNormResidualPreNorm.forward_with_ag"

_state: dict = {"ws": None, "comm": None, "pending": None}


def enabled() -> bool:
    return bed.enabled(NAME)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _max_m() -> int:
    ws = _state["ws"]
    default = DEFAULT_MAX_M if ws is None or ws.world >= 4 else DEFAULT_MAX_M_TP2
    return _env_int(ENV_MAX_M, default)


def _overlap_enabled() -> bool:
    try:
        from sglang.srt.layers.moe.utils import is_sbo_enabled, is_tbo_enabled

        return bool(is_tbo_enabled() or is_sbo_enabled())
    except Exception:  # noqa: BLE001 - unknown configuration: refuse
        return True


def _after_model_init(result, model, *args, **kwargs):
    """Allocate the workspace once per process, collectively on the tensor-parallel group, before graph capture."""
    if _state["ws"] is not None:
        return None
    try:
        from sglang.srt.distributed import get_tp_group
        from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
            CustomAllReduceV2,
        )
        from sglang.srt.layers.dp_attention import is_dp_attention_enabled

        from .moe_tail_ops import MAX_ROWS, MAX_WORLD, ROWS_PER_CTA, TailWorkspace

        tp = get_tp_group()
        comm = tp.ca_comm
        moe = next((layer.mlp for layer in model.layers if layer.is_moe_layer), None)
        reason = None
        if tp.world_size < 2 or tp.world_size > MAX_WORLD:
            reason = "tensor-parallel size outside 2..8"
        elif not isinstance(comm, CustomAllReduceV2) or comm.disabled:
            reason = "custom all-reduce v2 unavailable"
        elif is_dp_attention_enabled():
            reason = "DP attention"
        elif moe is None or moe.post_expert_proj is None:
            reason = "no MoE unpooler"
        else:
            hidden, k = moe.post_expert_proj.weight.shape
            if hidden % (tp.world_size * ROWS_PER_CTA) or k % 256:
                reason = "unsupported projection shape"
        if reason is not None:
            bed.report(NAME, False, reason=reason)
            return None
        hidden, k = moe.post_expert_proj.weight.shape
        ws = TailWorkspace(
            comm.group,
            comm.device,
            comm.rank,
            comm.world_size,
            hidden,
            k,
            max_rows=MAX_ROWS,
        )
        _state["ws"], _state["comm"] = ws, comm
        lo, hi = _env_int(ENV_MIN_M, DEFAULT_MIN_M), _max_m()
        bed.report(NAME, True, tp=comm.world_size, min_m=lo, max_m=hi)
    except Exception as e:  # noqa: BLE001 - leave SGLang stock if the workspace cannot be built
        bed.report(NAME, False, reason=f"workspace: {type(e).__name__}: {e}")
    return None


def _eligible(moe, output: torch.Tensor) -> str | None:
    from sglang.srt.runtime_context import get_forward

    from .moe_tail_ops import supported

    ws = _state["ws"]
    if ws is None:
        return "no workspace"
    m = output.shape[0] if output.dim() == 2 else -1
    lo, hi = _env_int(ENV_MIN_M, DEFAULT_MIN_M), _max_m()
    if not lo <= m <= hi:
        return "row count outside range"
    if getattr(get_forward(), "sp_active", False):
        return "sequence parallel"
    if _overlap_enabled():
        return "two-batch or single-batch overlap enabled"
    if moe.tp_size <= 1 or moe.moe_tp_size != 1 or moe.post_expert_proj is None:
        return "layer configuration"
    if output.dtype != torch.bfloat16 or not output.is_contiguous():
        return "dtype/layout"
    boundary = moe.unpool_gather_boundary
    if boundary is None or not boundary.jit_selected():
        return "no deferred FFN boundary"
    from sglang.srt.distributed import get_tp_group

    if get_tp_group().ca_comm is not _state["comm"]:
        return "communicator changed"
    w = moe.post_expert_proj.weight
    if (
        w.dtype != torch.bfloat16
        or not w.is_contiguous()
        or tuple(w.shape) != (ws.hidden, ws.k)
    ):
        return "weight layout"
    if output.shape[1] != ws.k or not supported(ws, m, ws.hidden, ws.k):
        return "shape"
    return None


def _around_reduce_and_unpool(original, moe, output, *args, **kwargs):
    already_reduced = (
        bool(kwargs.get("already_reduced", False)) if not args else bool(args[0])
    )
    reason = _eligible(moe, output)
    if reason is not None:
        bed.count(NAME, "stock:" + reason)
        _state["pending"] = None
        return original(moe, output, *args, **kwargs)
    if not already_reduced:
        # the stock reduction of _reduce_and_unpool for this configuration
        from sglang.srt.distributed import tensor_model_parallel_all_reduce

        output = tensor_model_parallel_all_reduce(output)
    _state["pending"] = (output, moe)
    return output


def _around_forward_with_ag(original, boundary, stream, local, comm, *args, **kwargs):
    pending = _state["pending"]
    _state["pending"] = None
    if pending is None or pending[0] is not local or args or kwargs:
        return original(boundary, stream, local, comm, *args, **kwargs)
    _, moe = pending
    if (
        stream.dtype != torch.float32
        or not stream.is_contiguous()
        or stream.shape[0] != local.shape[0]
    ):
        raise RuntimeError(
            "moe_tail: unexpected residual stream layout after the unprojected MoE output"
        )
    from .moe_tail_ops import moe_tail

    boundary._ensure_cached_weights(torch.bfloat16)
    out, new_stream = moe_tail(
        _state["ws"],
        local,
        moe.post_expert_proj.weight,
        stream,
        boundary._post_norm.eps,
        boundary._pre_norm.eps,
        post_norm_is_input_scale=boundary._post_norm_is_input_scale,
        post_weight=boundary._cc_post_weight
        if boundary._post_norm.has_weight
        else None,
        pre_weight=boundary._cc_pre_weight if boundary._pre_norm.has_weight else None,
        alpha=boundary._cc_alpha if boundary._use_residual_gate else None,
        beta=boundary._cc_beta if boundary._use_residual_gate else None,
    )
    bed.count(NAME, f"fused:M={local.shape[0]}:tp={_state['ws'].world}")
    return out, new_stream


def register() -> bool:
    """Register the hooks when enabled; returns whether they were registered."""
    if not enabled():
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(T_MODEL_INIT, _after_model_init, HookType.AFTER)
    HookRegistry.register(T_REDUCE_UNPOOL, _around_reduce_and_unpool, HookType.AROUND)
    HookRegistry.register(T_FORWARD_WITH_AG, _around_forward_with_ag, HookType.AROUND)
    return True
