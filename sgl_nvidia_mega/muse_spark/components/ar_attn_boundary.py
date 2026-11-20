# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``ar_attn_boundary`` (env ``MUSE_VR_AR_ATTN_BOUNDARY=1``, default off).

Stock decode runs, after each layer's attention, the row-parallel ``o_proj`` GEMM, a custom all-reduce kernel
(``1shot_push`` for every decode size) and then the attention boundary kernel (post-norm + gated residual +
pre-FFN norm + fp32 router copy). This component keeps the GEMM, skips its all-reduce, and runs
``ar_norm_residual`` instead: one kernel that does the same one-shot push reduction and the same boundary
arithmetic (bitwise identical outputs; ``tests/dist_ar_norm_residual.py``).

It is an AROUND hook on ``MuseSpark1xDecoderLayer.forward``. A call takes the fused path only when the stock call
would have taken exactly the pair being replaced: decode or target-verify batch, no sequence-parallel or other
reduce override, the attention group's all-reduce dispatch resolves to the custom all-reduce with ``1shot_push``,
the stock boundary would run its JIT kernel, and the row count is within ``MUSE_VR_AR_ATTN_BOUNDARY_MAX_M``
(default 64: larger caps measured slower in the in-server A/B at tensor-parallel size 4; tunable,
bench/bench_ar_norm_residual.py). Every other call runs SGLang's own forward.

Engagement requires the sha256 of every SGLang source the fused path mirrors (including ``parallel_state.py``, whose
all-reduce dispatch ``_stock_ar_is_one_shot_push`` reproduces) to match the pinned revision. Engagement is recorded
with ``bed.report``; fused call sites seen at CUDA-graph capture or in eager steps are tallied per row count with
``bed.count``.

``MUSE_VR_AR2_BOUNDARY=1`` (default off) extends the component above the one-shot cap: a call with more rows than
``MUSE_VR_AR_ATTN_BOUNDARY_MAX_M`` and at most ``MUSE_VR_AR2_BOUNDARY_MAX_M`` rows (default 512), where the stock
all-reduce dispatch resolves to the custom all-reduce, runs ``ar2_norm_residual`` instead (two-shot push reduce by row
owner + gather of the sums + the same boundary arithmetic, one kernel; bitwise to ``2shot_push`` + the boundary, equal
values to SGLang's ``2shot_pull`` + the boundary). Other calls are unchanged. ``MUSE_VR_AR2_BOUNDARY_MIN_M`` (default 0)
leaves calls with fewer rows than that on the stock pair (a row band between the one-shot cap and the two-shot range).

``MUSE_VR_AR_ATTN_BOUNDARY_ROWS`` (default unset): a comma-separated set of row counts that take the one-shot fused path
instead of the cap (``MUSE_VR_AR_ATTN_BOUNDARY_MAX_M`` is then not used), so a served bucket can take the one-shot form while a
smaller or larger one does not; with ``MUSE_VR_AR2_BOUNDARY=1`` a call whose row count is not in the set takes the two-shot form
inside its own range. Unset = the cap rule above, unchanged.

``MUSE_VR_AR_K47=1`` (default off): calls with ``MUSE_VR_AR_K47_MIN_M`` (default 80) <= rows <= ``MUSE_VR_AR_K47_MAX_M``
(default 1024) that take neither the one-shot form above nor the two-shot form of ``MUSE_VR_AR2_BOUNDARY``'s range run SGLang
pull request 47174's fused push all-reduce + boundary kernel (``vendor/sglang_pr47174``: 1-shot or 2-shot push by its own
rule; the 2-shot gives each rank a column slice of every row, reduces it in rank order and gathers the sums) where the stock
all-reduce dispatch resolves to the custom all-reduce (any algorithm) and that op's own support check passes. Its outputs
are those of the push all-reduce + the boundary kernel at the same launch shape (bitwise to the stock pair at the launch
shapes it uses in this range).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/distributed/device_communicators/custom_all_reduce_v2.py": (
        "CustomAllReduceV2",
    ),
    "srt/layers/moe/utils.py": ("should_skip_mlp_all_reduce",),
    "srt/runtime_context.py": ("get_forward", "get_parallel"),
    "srt/distributed/*.py": ("get_tp_group",),
    "srt/models/muse_spark.py": ("_use_split_bf16_router",),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


logger = logging.getLogger(__name__)

NAME = "ar_attn_boundary"  # env MUSE_VR_AR_ATTN_BOUNDARY
ENV_MAX_M = "MUSE_VR_AR_ATTN_BOUNDARY_MAX_M"
DEFAULT_MAX_M = 64
ENV_AR2 = "MUSE_VR_AR2_BOUNDARY"
ENV_AR2_MAX_M = "MUSE_VR_AR2_BOUNDARY_MAX_M"
DEFAULT_AR2_MAX_M = 512
ENV_AR2_MIN_M = "MUSE_VR_AR2_BOUNDARY_MIN_M"
ENV_ROWS = "MUSE_VR_AR_ATTN_BOUNDARY_ROWS"
ENV_K47 = "MUSE_VR_AR_K47"
ENV_K47_MIN_M = "MUSE_VR_AR_K47_MIN_M"
ENV_K47_MAX_M = "MUSE_VR_AR_K47_MAX_M"
DEFAULT_K47_MIN_M = 80
DEFAULT_K47_MAX_M = 1024
TARGET = "sglang.srt.models.muse_spark.MuseSpark1xDecoderLayer.forward"

# sha256 of the SGLang sources (relative to the directory holding the ``sglang`` package) whose behaviour the fused
# path reproduces.
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
    "sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh": (
        "ec14bbf6b02c481daad053dce6246d684a957cc66aa58a464a15ef117a094258",
        "693fc6e5b73478bab41a6ea93c1636b180c0e2ddb733794d80f370efa9427c9a",
    ),
    "sglang/kernels/jit/include/sgl_kernel/distributed/communicator.cuh": "30d5c75e86e86da7ce99cd6eab85212fdbc71f83a8e3c715c4e9f2c6c1d8986f",
    "sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py": (
        "277da88e2837e585f3ebdf4122906d29ea7a96cc30b981f7a028f6f63b6cf001",
        "f7b23a55019734be7b2f1958e20cc1f78a31251a035f2d75885911aa4a46e6b5",
    ),
    "sglang/srt/distributed/parallel_state.py": (
        "49f4ecb8627b469d1e6972ba6eba0c54412ea5eec756da587b3ad375f607a5c7",
        "b6c0cd98a81e884c7c4677bd783b3b0cb4eb1438c93540a520296595e8a2ff2d",
    ),
    "sglang/srt/layers/linear.py": "3ce31df0a8aab09ed8afa0ada8c0ce30ca207a748c02c3073d7f53648b9b7c3c",
}


def enabled() -> bool:
    return bed.enabled(NAME)


def _max_m() -> int:
    try:
        return int(os.environ.get(ENV_MAX_M, str(DEFAULT_MAX_M)))
    except ValueError:
        return DEFAULT_MAX_M


def _rows() -> Optional[frozenset]:
    """The one-shot row set (MUSE_VR_AR_ATTN_BOUNDARY_ROWS), or None for the cap rule. Malformed values fall back to None."""
    v = os.environ.get(ENV_ROWS, "").strip()
    if not v:
        return None
    try:
        rows = frozenset(int(x) for x in v.split(",") if x.strip())
    except ValueError:
        return None
    return rows if rows and all(r > 0 for r in rows) else None


def _one_shot_row(m: int) -> bool:
    rows = _rows()
    return (m in rows) if rows is not None else m <= _max_m()


def _one_shot_top() -> int:
    rows = _rows()
    return max(rows) if rows is not None else _max_m()


def _ar2_enabled() -> bool:
    return os.environ.get(ENV_AR2, "0") == "1"


def _ar2_min_m() -> int:
    try:
        return int(os.environ.get(ENV_AR2_MIN_M, "0"))
    except ValueError:
        return 0


def _ar2_max_m() -> int:
    try:
        return int(os.environ.get(ENV_AR2_MAX_M, str(DEFAULT_AR2_MAX_M)))
    except ValueError:
        return DEFAULT_AR2_MAX_M


def _k47_enabled() -> bool:
    return os.environ.get(ENV_K47, "0") == "1"


def _k47_min_m() -> int:
    try:
        return int(os.environ.get(ENV_K47_MIN_M, str(DEFAULT_K47_MIN_M)))
    except ValueError:
        return DEFAULT_K47_MIN_M


def _k47_max_m() -> int:
    try:
        return int(os.environ.get(ENV_K47_MAX_M, str(DEFAULT_K47_MAX_M)))
    except ValueError:
        return DEFAULT_K47_MAX_M


def _ar2_row(m: int) -> bool:
    """The call takes the two-shot form's range (MUSE_VR_AR2_BOUNDARY): not a one-shot row and within MIN_M..MAX_M."""
    return _ar2_enabled() and not _one_shot_row(m) and _ar2_min_m() <= m <= _ar2_max_m()


def _k47_row(m: int) -> bool:
    """The call takes pull request 47174's form: neither a one-shot nor a two-shot row, within its MIN_M..MAX_M."""
    return (
        _k47_enabled()
        and not _one_shot_row(m)
        and not _ar2_row(m)
        and _k47_min_m() <= m <= _k47_max_m()
    )


def row_form(m: int) -> str:
    """'one' / 'two' / 'k47' / 'stock': the form the row count selects (before the per-call dispatch checks)."""
    if _one_shot_row(m):
        return "one"
    if _ar2_row(m):
        return "two"
    if _k47_row(m):
        return "k47"
    return "stock"


def pins_match() -> tuple[bool, str]:
    for rel, want in PINS.items():
        if not bed.pinned(rel, want):
            return False, f"{rel} differs from the pinned revision"
    return True, "ok"


def _stock_ar_is_one_shot_push(group, tensor: torch.Tensor) -> bool:
    """True when ``group.all_reduce(tensor)`` resolves to ``ca_comm.custom_all_reduce`` with ``1shot_push``."""
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

    if (
        group is None
        or group.world_size == 1
        or tensor.is_cpu
        or torch.compiler.is_compiling()
    ):
        return False
    for name in ("hpu_communicator", "xpu_communicator", "npu_communicator"):
        c = getattr(group, name, None)
        if c is not None and not c.disabled:
            return False
    ca = group.ca_comm
    if not isinstance(ca, CustomAllReduceV2) or ca.disabled:
        return False
    pym = (
        group.pymscclpp_comm is not None
        and group.pymscclpp_comm.should_mscclpp_allreduce(tensor)
    )
    if pym or not ca.should_custom_ar(tensor):
        return False
    if group._can_use_flashinfer_allreduce(tensor):
        return False
    if (
        group._resolve_outplace_all_reduce_method(
            input_=tensor, should_use_pymscclpp_allreduce=False
        )
        != "ca"
    ):
        return False
    return ca.uses_one_shot_push(tensor.numel() * tensor.element_size())


def _stock_ar_is_custom(group, tensor: torch.Tensor) -> bool:
    """True when ``group.all_reduce(tensor)`` resolves to ``ca_comm.custom_all_reduce`` (any of its algorithms)."""
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

    if (
        group is None
        or group.world_size == 1
        or tensor.is_cpu
        or torch.compiler.is_compiling()
    ):
        return False
    for name in ("hpu_communicator", "xpu_communicator", "npu_communicator"):
        c = getattr(group, name, None)
        if c is not None and not c.disabled:
            return False
    ca = group.ca_comm
    if not isinstance(ca, CustomAllReduceV2) or ca.disabled:
        return False
    pym = (
        group.pymscclpp_comm is not None
        and group.pymscclpp_comm.should_mscclpp_allreduce(tensor)
    )
    if pym or not ca.should_custom_ar(tensor):
        return False
    if group._can_use_flashinfer_allreduce(tensor):
        return False
    return (
        group._resolve_outplace_all_reduce_method(
            input_=tensor, should_use_pymscclpp_allreduce=False
        )
        == "ca"
    )


def _layer_eligible(layer, attn_input: torch.Tensor, forward_batch) -> str | None:
    """Reason the call must run stock, or None. Checks that need only host facts known before the GEMM."""
    from sglang.srt.layers.moe.utils import should_skip_mlp_all_reduce
    from sglang.srt.runtime_context import get_forward

    if layer._megakernel_capable:
        return "megakernel-capable layer"
    mode = forward_batch.forward_mode
    if not (mode.is_decode() or mode.is_target_verify()):
        return "not decode/verify"
    m = attn_input.shape[0]
    if m == 0 or row_form(m) == "stock":
        return "row count outside range"
    o_proj = layer.self_attn.o_proj
    if (
        not (o_proj.reduce_results and o_proj.use_dp_attention_reduce)
        or o_proj.use_decode_attn_tp
    ):
        return "o_proj reduce configuration"
    if get_forward().sp_active or should_skip_mlp_all_reduce():
        return "sequence-parallel / reduce override"
    if (
        layer._fused_attn_boundary is None
        or not layer._fused_attn_boundary.jit_selected()
    ):
        return "stock boundary not on its JIT kernel"
    return None


def _around_forward(original, layer, positions, stream, attn_input, forward_batch):
    reason = _layer_eligible(layer, attn_input, forward_batch)
    if reason is not None:
        bed.count(NAME, "stock:" + reason)
        return original(layer, positions, stream, attn_input, forward_batch)

    from sglang.srt.distributed import get_tp_group
    from sglang.srt.models.muse_spark import _use_split_bf16_router
    from sglang.srt.runtime_context import get_parallel

    from .ar_norm_residual import ar_norm_residual, supported

    attn = layer.self_attn
    # MuseSpark1xAttention.forward with the o_proj all-reduce deferred
    qkv, _ = attn.qkv_proj(attn_input)
    partial, _ = attn.o_proj(
        attn.gated_attention(positions, qkv, forward_batch), skip_all_reduce=True
    )

    router_dtype = layer._router_dtype
    if router_dtype is not None and _use_split_bf16_router(partial.shape[0]):
        router_dtype = None
    boundary = layer._fused_attn_boundary
    group = get_parallel().attn_tp_group
    comm = group.ca_comm
    m_rows = partial.shape[0]
    form = row_form(m_rows)
    k47 = None
    if form == "k47":
        from .vendor.sglang_pr47174 import muse_spark_all_reduce_norm_residual as k47_op

        k47 = (
            k47_op
            if (
                _stock_ar_is_custom(group, partial)
                and layer.operator_dtype == torch.bfloat16
                and k47_op.all_reduce_norm_residual_supported(
                    comm, m_rows, partial.shape[1]
                )
                and boundary._triton_supported(
                    stream, partial, layer.operator_dtype, router_dtype, None
                )
                and partial.is_contiguous()
            )
            else None
        )
    if form == "two":
        from .ar2_norm_residual import ar2_norm_residual
        from .ar2_norm_residual import supported as supported2

        two_shot = (
            _stock_ar_is_custom(group, partial)
            and supported2(comm, m_rows, partial.shape[1])
            and boundary._triton_supported(
                stream, partial, layer.operator_dtype, router_dtype, None
            )
            and partial.is_contiguous()
        )
    else:
        two_shot = False
    if k47 is not None:
        ffn_input, stream_out, router_input = k47.all_reduce_norm_residual(
            comm,
            partial,
            stream,
            **boundary.jit_kernel_kwargs(layer.operator_dtype),
            want_router=router_dtype is not None,
        )
        bed.count(NAME, f"k47:M={m_rows}:tp={comm.world_size}")
        stream = stream_out
    elif two_shot:
        boundary._ensure_cached_weights(layer.operator_dtype)
        ffn_input, stream_out, router_input = ar2_norm_residual(
            comm,
            partial,
            stream,
            boundary._post_norm.eps,
            boundary._pre_norm.eps,
            post_norm_is_input_scale=boundary._post_norm_is_input_scale,
            post_weight=boundary._cc_post_weight
            if boundary._post_norm.has_weight
            else None,
            pre_weight=boundary._cc_pre_weight
            if boundary._pre_norm.has_weight
            else None,
            alpha=boundary._cc_alpha if boundary._use_residual_gate else None,
            beta=boundary._cc_beta if boundary._use_residual_gate else None,
            want_router=router_dtype is not None,
        )
        bed.count(NAME, f"fused2:M={m_rows}:tp={comm.world_size}")
        stream = stream_out
    elif (
        form == "one"
        and _stock_ar_is_one_shot_push(group, partial)
        and supported(comm, partial.shape[0], partial.shape[1])
        and boundary._triton_supported(
            stream, partial, layer.operator_dtype, router_dtype, None
        )
        and partial.is_contiguous()
    ):
        boundary._ensure_cached_weights(layer.operator_dtype)
        ffn_input, stream_out, router_input = ar_norm_residual(
            comm,
            partial,
            stream,
            boundary._post_norm.eps,
            boundary._pre_norm.eps,
            post_norm_is_input_scale=boundary._post_norm_is_input_scale,
            post_weight=boundary._cc_post_weight
            if boundary._post_norm.has_weight
            else None,
            pre_weight=boundary._cc_pre_weight
            if boundary._pre_norm.has_weight
            else None,
            alpha=boundary._cc_alpha if boundary._use_residual_gate else None,
            beta=boundary._cc_beta if boundary._use_residual_gate else None,
            want_router=router_dtype is not None,
        )
        bed.count(NAME, f"fused:M={partial.shape[0]}:tp={comm.world_size}")
        stream = stream_out
    else:
        # the stock pair, as RowParallelLinear.forward and the decoder layer issue them
        bed.count(NAME, "stock:all-reduce dispatch")
        attn_output = group.all_reduce(partial)
        ffn_input, stream, router_input = boundary(
            stream,
            attn_output,
            branch_dtype=layer.operator_dtype,
            router_dtype=router_dtype,
        )

    # remainder of MuseSpark1xDecoderLayer.forward (pinned revision), unchanged
    if layer.is_moe_layer:
        ffn_output = layer.mlp(ffn_input, router_input, forward_batch)
    else:
        ffn_output = layer.mlp(ffn_input)
    if ffn_output.shape[-1] != stream.shape[-1]:
        next_attn_input, stream = layer._fused_ffn_boundary.forward_with_ag(
            stream, ffn_output, get_tp_group().ca_comm
        )
        return stream, next_attn_input
    if layer._fused_ffn_boundary is None:
        return layer.post_feedforward_residual_gate(
            stream, layer.post_feedforward_layernorm(ffn_output)
        ), None
    next_attn_input, stream, _ = layer._fused_ffn_boundary(
        stream, ffn_output, branch_dtype=layer.operator_dtype
    )
    return stream, next_attn_input


def register() -> bool:
    """Register the hook when enabled and pinned; returns whether it was registered (recorded with bed.report)."""
    if not enabled():
        return False
    ok, why = pins_match()
    if not ok:
        bed.report(NAME, False, reason=why)
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    def hook(original, *args, **kwargs):
        if kwargs or len(args) != 5:
            return original(*args, **kwargs)
        return _around_forward(original, *args)

    HookRegistry.register(TARGET, hook, HookType.AROUND)
    bed.report(
        NAME,
        True,
        max_m=_max_m(),
        rows=sorted(_rows()) if _rows() is not None else None,
        ar2=_ar2_enabled(),
        ar2_min_m=_ar2_min_m(),
        ar2_max_m=_ar2_max_m(),
        k47=_k47_enabled(),
        k47_min_m=_k47_min_m(),
        k47_max_m=_k47_max_m(),
    )
    return True
