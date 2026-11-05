# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``fin_2shot`` (env ``MUSE_VR_FIN_2SHOT=1``, default off): wide decode / verify MoE finalize calls fused with a
TWO-SHOT push all-reduce on the tensor-parallel communicator's own push plane, in the form of SGLang pull request 46724
(``vendor/sglang_pr46724``: its finalize op, CuTe-DSL kernel and workspace).

Stock SGLang fuses the routed-MoE finalize (unpermute, post-expert RMSNorm, weighted top-k sum) with a ONE-shot push
all-reduce up to 384 rows (``FinalizePushWorkspace``) and above that runs the finalize and a standalone all-reduce. The
pull request adds a two-shot exchange to the same kernel (reduce-scatter by column slice in source-rank order, then a gather
of the reduced slices) and uses it where the communicator's table picks ``2shot_push``. SGLang 306ccf6's 4-rank table never
picks ``2shot_push``, so this component forces the two-shot form for ``MUSE_VR_FIN_2SHOT_MIN_M <= M <=
MUSE_VR_FIN_2SHOT_MAX_M`` (defaults 176 and 1024) and leaves every other call (in particular every M below MIN_M) on
SGLang's own op unchanged.

Hooks (all on the pinned SGLang revision):
- AFTER ``MuseSpark1xMoE.set_finalize_ar_workspace``: once per stock workspace, a pull-request workspace on the same
  communicator, group, width and multicast setting (model construction, every rank, outside CUDA-graph capture).
- AROUND ``MuseSpark1xMoE._forward_tp``: for a decode / verify call that the stock conditions would fuse except for the row
  cap (non-SP, bf16, expert-TP 1, the workspace's own ``can_fuse``) with M in range, the stock statements of the fused
  branch with that workspace: ``_forward_experts(..., ar_workspace=ws)`` then ``_reduce_and_unpool(output,
  already_reduced=True)``. Other calls run the stock method.
- AROUND ``FusedPostExpertNormWeightedSumTRTLLM.forward_cute_dsl``: only calls that carry this component's workspace run the
  pull request's op (``ar_two_shot`` from the workspace); all others run SGLang's op.

Numerics: rank-order fp32 sums with one bf16 rounding, as the pull request states; bitwise to SGLang's fused one-shot path
where that exists (to 384 rows), value-equal class (a few elements a few ULP) against SGLang's separate finalize +
all-reduce above it. Exclusive with ``fin2`` (same op hook). Decisions are counted under ``fin_2shot`` in ``bed``.
"""

from __future__ import annotations

import logging

import torch

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/runtime_context.py": ("get_forward", "get_parallel"),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


logger = logging.getLogger("muse_vr_plugin")

NAME = "fin_2shot"
OP_MODULE = "sglang.kernels.ops.communication.muse_spark_moe_finalize_all_reduce"
T_OP = OP_MODULE + ".FusedPostExpertNormWeightedSumTRTLLM.forward_cute_dsl"
T_TP = "sglang.srt.models.muse_spark.MuseSpark1xMoE._forward_tp"
T_SET = "sglang.srt.models.muse_spark.MuseSpark1xMoE.set_finalize_ar_workspace"
PINS = {
    "sglang/kernels/ops/communication/muse_spark_moe_finalize_all_reduce_cutedsl.py": (
        "9e503382ec0d904476aa039d77510486cc71563c2e1a8bfcbb8d9234e7e2e689"
    ),
    "sglang/kernels/ops/communication/muse_spark_moe_finalize_all_reduce.py": (
        "b22ba8b4d895fd38c99a0835b5a37b82adaa8e7587379f09807cb072eceb9948"
    ),
    "sglang/srt/layers/communication/muse_spark_finalize_workspace.py": (
        "578c07d454011b0972ff22bb017a015d13ce03bf61fd3445497c313d0ab3de3a"
    ),
    # MuseSpark1xMoE._forward_tp's fused branch is reproduced statement for statement from this revision
    "sglang/srt/models/muse_spark.py": "8bafbb8802fb0d9f81c190e87b9012fca546182be08bef6c8b3db814d27bfe19",
}

_state: dict = {"ws": {}, "cls": None, "op": {}}


def enabled() -> bool:
    return bed.enabled(NAME)


def min_m() -> int:
    return bed.env_int("MUSE_VR_FIN_2SHOT_MIN_M", 176)


def max_m() -> int:
    return bed.env_int("MUSE_VR_FIN_2SHOT_MAX_M", 1024)


def pins_match() -> tuple[bool, str]:
    for rel, want in PINS.items():
        if not bed.pinned(rel, want):
            return False, f"{rel} differs from the pinned revision"
    return True, "ok"


def workspace_class():
    """The pull request's workspace with its fused two-shot forced on in MIN_M..MAX_M (capped at its slot capacity)."""
    if _state["cls"] is None:
        from .vendor.sglang_pr46724 import muse_spark_finalize_workspace as pr_ws

        class TwoShotFinalizeWorkspace(pr_ws.FinalizePushWorkspace):
            def __init__(
                self, communicator, device_group, hidden_dim, *, multicast, lo, hi
            ):
                super().__init__(
                    communicator, device_group, hidden_dim, multicast=multicast
                )
                self.lo = int(lo)
                self.hi = int(min(hi, self.two_shot_max_tokens))
                self.max_tokens = self.hi

            def uses_two_shot_push(self, num_tokens: int) -> bool:
                return self.lo <= num_tokens <= self.hi

        _state["cls"] = TwoShotFinalizeWorkspace
    return _state["cls"]


def two_shot_workspace(stock_ws):
    """This component's workspace for a stock workspace (None when not built)."""
    return None if stock_ws is None else _state["ws"].get(id(stock_ws))


def _after_set_workspace(result, self, workspace, *args, **kwargs):
    """AFTER ``set_finalize_ar_workspace``: build the pull-request workspace once per stock workspace (collective)."""
    if workspace is None or id(workspace) in _state["ws"]:
        return result
    try:
        cls = workspace_class()
        ws = cls(
            workspace.communicator,
            workspace.group,
            workspace.hidden_dim,
            multicast=workspace.multicast,
            lo=min_m(),
            hi=max_m(),
        )
        _state["ws"][id(workspace)] = ws
        bed.report(
            NAME + ":workspace",
            True,
            lo=ws.lo,
            hi=ws.hi,
            world=ws.world_size,
            two_shot_max_tokens=ws.two_shot_max_tokens,
            multicast=ws.multicast,
        )
    except Exception as e:  # noqa: BLE001 - every rank sees the same arguments; stay stock on failure
        _state["ws"][id(workspace)] = None
        bed.report(NAME + ":workspace", False, reason=f"{type(e).__name__}: {e}")
    return result


def _ineligible(moe, hidden_states, forward_batch, ws) -> str | None:
    """Reason a call stays stock, or None. The stock fused branch's conditions (MuseSpark1xMoE._forward_tp), with the
    row cap taken from this component's range."""
    from sglang.srt.runtime_context import get_forward, get_parallel

    m = hidden_states.shape[0]
    if not (ws.lo <= m <= ws.hi):
        return f"M={m}"
    if forward_batch is None or not (
        forward_batch.forward_mode.is_decode()
        or forward_batch.forward_mode.is_target_verify()
    ):
        return "mode"
    if (
        moe.moe_tp_size != 1
        or get_forward().sp_active
        or hidden_states.dtype != torch.bfloat16
    ):
        return "config"
    if not ws.can_fuse(get_parallel().tp_group, m):
        return "can_fuse"
    return None


def _around_forward_tp(
    original, self, hidden_states, router_input=None, forward_batch=None
):
    ws = two_shot_workspace(self.finalize_ar_workspace)
    if ws is None:
        return original(self, hidden_states, router_input, forward_batch)
    why = _ineligible(self, hidden_states, forward_batch, ws)
    if why is not None:
        if not why.startswith("M="):
            bed.count(NAME, "stock:" + why)
        return original(self, hidden_states, router_input, forward_batch)
    # MuseSpark1xMoE._forward_tp (pinned revision), fused branch, with this component's workspace
    output = self._forward_experts(hidden_states, router_input, ar_workspace=ws)
    bed.count(NAME, f"fin_2shot:M={hidden_states.shape[0]}:tp={ws.world_size}")
    return self._reduce_and_unpool(output, already_reduced=True)


def _pr_op(stock_op):
    """The pull request's op with the stock op's configuration (cached per stock op)."""
    op = _state["op"].get(id(stock_op))
    if op is None:
        from .vendor.sglang_pr46724 import muse_spark_moe_finalize_all_reduce as pr_op

        op = pr_op.FusedPostExpertNormWeightedSumTRTLLM(
            topk=stock_op.topk,
            hidden_dim=stock_op.hidden_dim,
            eps=stock_op.eps,
            weight_first=stock_op.weight_first,
            cluster_size=stock_op.cluster_size,
        )
        _state["op"][id(stock_op)] = op
    return op


def _around_forward_cute_dsl(
    original,
    self,
    gemm2_output,
    expert_weights,
    expanded_idx_to_permuted_idx,
    rms_norm_weight,
    output,
    *,
    ar_workspace=None,
):
    if (
        ar_workspace is None
        or _state["cls"] is None
        or not isinstance(ar_workspace, _state["cls"])
    ):
        return original(
            self,
            gemm2_output,
            expert_weights,
            expanded_idx_to_permuted_idx,
            rms_norm_weight,
            output,
            ar_workspace=ar_workspace,
        )
    return _pr_op(self).forward_cute_dsl(
        gemm2_output,
        expert_weights,
        expanded_idx_to_permuted_idx,
        rms_norm_weight,
        output,
        ar_workspace=ar_workspace,
    )


def register() -> bool:
    if not enabled():
        return False
    if bed.enabled("fin2"):
        bed.report(NAME, False, reason="exclusive with fin2 (same finalize op hook)")
        return False
    ok, why = pins_match()
    if not ok:
        bed.report(NAME, False, reason=why)
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(T_SET, _after_set_workspace, HookType.AFTER)
    HookRegistry.register(T_TP, _around_forward_tp, HookType.AROUND)
    HookRegistry.register(T_OP, _around_forward_cute_dsl, HookType.AROUND)
    bed.report(NAME, True, min_m=min_m(), max_m=max_m())
    return True
