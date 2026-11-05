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

Cross-rank hand-off. Default: release / acquire at system scope (CTA barrier, ``fence.acq_rel.sys``, relaxed
counter increments; consumers poll with ``ld.acquire.sys``), as the PTX memory model prescribes for
producer/consumer synchronization between GPUs. ``MUSE_VR_MOE_TAIL_UNSAFE=1`` selects a tagged-word hand-off with no
fences: each value travels in a 32-bit word with a per-call tag and the consumer polls each word until the tag
matches. Within a call this relies only on single-copy atomicity of aligned 32-bit relaxed system-scope accesses; the
reuse of a receive half two calls later relies on stream order and on the data dependence between calls rather than
on a release/acquire pair, which is not guaranteed by the PTX memory model. Off by default.

Tagged hand-off and row counts. A call writes rows ``0..M-1`` of the receive half only; nothing clears a half after it
is read, and tags repeat after 65535 calls. If calls with different row counts were mixed, a row last written by an
earlier call could still hold a word whose tag equals a much later call's tag when that later call reads it, and the
stale value would be accepted. The tagged hand-off therefore engages at exactly one row count
(``MUSE_VR_MOE_TAIL_MAX_M``): every engaged call writes every row it reads, and rows above that count are never read.
Calls at other row counts take SGLang's path. The fenced hand-off reads plain values after an acquire and has no tags,
so it keeps the full ``MIN_M`` .. ``MAX_M`` range.

Overlap. The pending tensor and the workspace counters are per process; two-batch or single-batch overlap could run
two layers' tails interleaved on different streams, so the component steps aside whenever either overlap mode is on.

Hooks (AROUND): ``MuseSpark1xMoE._reduce_and_unpool`` returns the all-reduced expert output unprojected and marks it;
``FusedPostNormResidualPreNorm.forward_with_ag`` (the layer calls it whenever the MoE output is narrower than the
stream) runs the fused tail for a marked tensor. The workspace (symmetric receive buffers) is allocated collectively
right after ``MuseSpark1xModel.__init__``, before any CUDA graph is captured. Row range:
``MUSE_VR_MOE_TAIL_MIN_M`` .. ``MUSE_VR_MOE_TAIL_MAX_M``. Defaults 9 .. 16 at tensor-parallel size 4 and above (with the
tagged hand-off, the in-server decode step A/B measured a gain at 16 rows only: at 8 rows SGLang's own slice + gather
path was as fast, at 32 rows the difference was within run-to-run spread; above 32 rows bench/bench_moe_tail.py measured
the fused tail slower). Off by default below size 4 (no in-server measurement there).
"""

from __future__ import annotations

import logging
import os

import torch

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/layers/moe/utils.py": ("is_sbo_enabled", "is_tbo_enabled"),
    "srt/distributed/*.py": ("get_tp_group", "tensor_model_parallel_all_reduce"),
    "srt/distributed/device_communicators/custom_all_reduce_v2.py": (
        "CustomAllReduceV2",
    ),
    "srt/layers/dp_attention.py": ("is_dp_attention_enabled",),
    "srt/runtime_context.py": ("get_forward",),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


logger = logging.getLogger(__name__)

NAME = "moe_tail"  # env MUSE_VR_MOE_TAIL
ENV_UNSAFE = "MUSE_VR_MOE_TAIL_UNSAFE"
ENV_MIN_M = "MUSE_VR_MOE_TAIL_MIN_M"
ENV_MAX_M = "MUSE_VR_MOE_TAIL_MAX_M"
DEFAULT_MIN_M = 9
DEFAULT_MAX_M = 16
DEFAULT_MAX_M_TP2 = 0

T_MODEL_INIT = "sglang.srt.models.muse_spark.MuseSpark1xModel.__init__"
T_REDUCE_UNPOOL = "sglang.srt.models.muse_spark.MuseSpark1xMoE._reduce_and_unpool"
T_FORWARD_WITH_AG = "sglang.kernels.ops.muse_spark_v12.fused_norm_residual.FusedPostNormResidualPreNorm.forward_with_ag"

# sha256 of the SGLang sources (relative to the directory holding the ``sglang`` package) this component mirrors,
# includes or calls.
PINS = {
    "sglang/srt/models/muse_spark.py": (
        "7e5018b3cb2ef41ee6cf09407e91e28dde069e5a33719ea58beb0d7137087355",
        "8bafbb8802fb0d9f81c190e87b9012fca546182be08bef6c8b3db814d27bfe19",
        "c0db42467a960c509b658c3c40570ee889f73d41b52cdbd80b9c291c1792a33f",
    ),
    "sglang/kernels/ops/muse_spark_v12/fused_norm_residual.py": (
        "61be2075b233c687f7b111bae06d4f9494c3032a8b0c2289c9ae10fc8c757493",
        "454f084d20e5f2353a70d260ea4a70c7dfdff3bf45a58619ef72e1eec2646fe3",
    ),
    "sglang/kernels/ops/muse_spark_v12/norm_residual.py": (
        "7b63c2e855371e50d291b684e09135ac466f5a1444f75ce87f591b68dd47223a",
        "76bd0c26ed4d3cf87126619c72a56de908dea0ea1b2c69f6e5ada47318035cbe",
    ),
    "sglang/kernels/jit/csrc/muse_spark_v12/norm_residual.cuh": "c227348c4d705183d377ac623a62b1ed6957eb8832fb0508f4ac99f1aa3cb0c8",
    "sglang/kernels/jit/csrc/muse_spark_v12/all_gather_norm_residual.cuh": "cb53a903365ac8688864c7835253ea8723d777d7ebb49b2c12ba140a6f956186",
    "sglang/kernels/ops/muse_spark_v12/all_gather_norm_residual.py": (
        "d55b841ce940e9c10f7699e8d4bde873734bc8bd4d4a2062eae8cd06e41fb87c",
        "8737d08505a0a53aeda5c0ca20ad0e8ac73a8180a8d1420c4e4a6af799b44c46",
    ),
    "sglang/kernels/ops/muse_spark_v12/gemm_ag.py": (
        "d7d3230691be775ba328ff2d340d39dc0c20d019d8bbf990bfc4e2537c5c33d8",
        "8d134e7cddcf7fdfdda8e5b9b4d93b9acbaece651e34166938545fd065605977",
    ),
    "sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py": (
        "277da88e2837e585f3ebdf4122906d29ea7a96cc30b981f7a028f6f63b6cf001",
        "f7b23a55019734be7b2f1958e20cc1f78a31251a035f2d75885911aa4a46e6b5",
    ),
    "sglang/srt/distributed/parallel_state.py": (
        "49f4ecb8627b469d1e6972ba6eba0c54412ea5eec756da587b3ad375f607a5c7",
        "b6c0cd98a81e884c7c4677bd783b3b0cb4eb1438c93540a520296595e8a2ff2d",
    ),
    "sglang/kernels/jit/include/sgl_kernel/distributed/ptx.cuh": "1ef631431ff74974a5d05758c4af9335cf8937e42ed9f0dc591a8341598be822",
    "sglang/kernels/jit/include/sgl_kernel/mbarrier.cuh": "17e0aa119ce7307f36ce9745f8a311ac29d67aacd442814230d983124e500b8a",
    "sglang/kernels/jit/include/sgl_kernel/utils.cuh": (
        "d7e858444067635f6d52f3958be4341e35e5bd0bc6437a21b085ce1a1b2218b9",
        "18b0bb2e27dccc20d7509e8b6b6c16e2b57a5ce98be648865fb75059be8a13db",
    ),
}

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


def _unsafe() -> bool:
    return os.environ.get(ENV_UNSAFE, "0").strip().lower() in ("1", "true", "yes", "on")


def row_range(tagged: bool, min_m: int, max_m: int) -> tuple[int, int]:
    """Engaged row counts: the tagged hand-off engages at one row count only (see the module docstring)."""
    return (max_m, max_m) if tagged else (min_m, max_m)


def _overlap_enabled() -> bool:
    try:
        from sglang.srt.layers.moe.utils import is_sbo_enabled, is_tbo_enabled

        return bool(is_tbo_enabled() or is_sbo_enabled())
    except Exception:  # noqa: BLE001 - unknown configuration: refuse
        return True


def pins_match() -> tuple[bool, str]:
    for rel, want in PINS.items():
        if not bed.pinned(rel, want):
            return False, f"{rel} differs from the pinned revision"
    return True, "ok"


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
            tagged=_unsafe(),
        )
        _state["ws"], _state["comm"] = ws, comm
        lo, hi = row_range(ws.tagged, _env_int(ENV_MIN_M, DEFAULT_MIN_M), _max_m())
        bed.report(NAME, True, tp=comm.world_size, unsafe=_unsafe(), min_m=lo, max_m=hi)
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
    lo, hi = row_range(ws.tagged, _env_int(ENV_MIN_M, DEFAULT_MIN_M), _max_m())
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
    """Register the hooks when enabled and pinned; returns whether they were registered."""
    if not enabled():
        return False
    ok, why = pins_match()
    if not ok:
        bed.report(NAME, False, reason=why)
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(T_MODEL_INIT, _after_model_init, HookType.AFTER)
    HookRegistry.register(T_REDUCE_UNPOOL, _around_reduce_and_unpool, HookType.AROUND)
    HookRegistry.register(T_FORWARD_WITH_AG, _around_forward_with_ag, HookType.AROUND)
    return True
