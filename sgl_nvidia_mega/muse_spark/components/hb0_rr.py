# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``hb0_rr``: the routed-MoE finalize + all-reduce of wide verify batches as the rows-ready far form.

For an engaged call, MoE-LD-W (``moe_ld``) runs its RING48 build with the early slot-table publish and the rows-ready flags
(``ops/moe_ldw.py`` ``EARLY_SLOTTAB`` + ``ROWS_READY``). The finalize call that follows it,
``FusedPostExpertNormWeightedSumTRTLLM.forward_cute_dsl`` with its all-reduce workspace, is then replaced by one plugin
kernel (``ops/c3_hb0.far`` with ``ROWS_READY``). That kernel reads each token's local expert rows as soon as the expert
kernel's flags for their passes carry its launch epoch, computes the post-expert RMSNorm and the weighted sum in the
engine's order, and exchanges the bf16 partials among the ranks with the zero-word rule of the stock one-shot exchange. It
waits for the expert kernel's completion before it exits.

Numerics: bitwise to the stock fused finalize + one-shot push all-reduce (``tests/dist_hb0_rr.py``: the pair against the
far form after the expert kernel; ``tests/dist_hb0_rr_served.py``: this component against the stock op).

Engagement: ``MUSE_VR_HB0_RR=1`` (default off), and every condition below; anything else runs the stock path, counted in
``bed`` under ``hb0_rr``:
- RING48 calls with ``MUSE_VR_HB0_RR_MIN_M <= M <= MUSE_VR_HB0_RR_MAX_M`` (defaults 224 and 256; at most 256);
- static expert placement (this rank's experts are ``rank * n_local ..``), a 2 / 4 / 8-rank tensor-parallel group;
- the finalize call is the one that consumes exactly the expert kernel's outputs (same tensors), with nothing launched in
  between; a finalize call that does not match a pending expert call turns the component off for the process (logged).
One ring site per process (one call slot per MoE layer, by load order). Per forward, the first layer's call advances the
sequence and the ring before its expert kernel. A wait that runs out sets the arena's sticky error word and writes NaN.
"""

from __future__ import annotations

import logging
import os
import threading

import torch

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/distributed/*.py": ("get_tp_group",),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


logger = logging.getLogger("muse_vr_plugin")

NAME = "hb0_rr"
ENV = "MUSE_VR_HB0_RR"
OP_MODULE = "sglang.kernels.ops.communication.muse_spark_moe_finalize_all_reduce"
TARGET = OP_MODULE + ".FusedPostExpertNormWeightedSumTRTLLM.forward_cute_dsl"
FAR_MAX_M = 256  # the far site's rows per source block
SITE = "far"

_lock = threading.Lock()
_state = {
    "arena": None,
    "arena_err": None,
    "off": "",
    "stash": None,
    "layer_of": {},
    "views": {},
    "flags": None,
    "fused": False,
}
# SGLang's MoE forward decides per call whether the finalize runs fused with the all-reduce (it passes the push workspace
# only for decode / target-verify batches within the workspace capacity); "fused" records that decision for the call in
# progress, so an expert call is only made rows-ready when its finalize will be the fused one this component replaces.
MOE_FORWARD_TARGET = "sglang.srt.models.muse_spark.MuseSpark1xMoE._forward_experts"


def enabled() -> bool:
    return os.environ.get(ENV, "0").strip() == "1"


def min_m() -> int:
    return max(65, int(os.environ.get("MUSE_VR_HB0_RR_MIN_M", "224")))


def max_m() -> int:
    return min(FAR_MAX_M, int(os.environ.get("MUSE_VR_HB0_RR_MAX_M", str(FAR_MAX_M))))


def note_layer(w13) -> None:
    """MoE-LD's load-time site registration: the layer's call slot is its position in load order."""
    lo = _state["layer_of"]
    lo.setdefault(int(w13.data_ptr()), len(lo))


_GROUP_OVERRIDE = None  # tests only: the group coordinator to use instead of SGLang's tensor-parallel group


def _group():
    if _GROUP_OVERRIDE is not None:
        return _GROUP_OVERRIDE
    from sglang.srt.distributed import get_tp_group

    return get_tp_group()


def _arena(device):
    """The far site's arena on the tensor-parallel group (collective; built once, outside graph capture), or None."""
    a = _state["arena"]
    if a is not None or _state["arena_err"] is not None:
        return a
    if torch.cuda.is_current_stream_capturing():
        # never built inside graph capture: refused loudly; the call runs the stock path. SGLang's graph warm-up
        # runs eagerly before capture, so an engaged M meets its first call outside capture.
        if not _state.get("capture_refused"):
            _state["capture_refused"] = True
            bed.report(
                NAME + ":arena",
                False,
                reason="first engaged call inside graph capture: arena build refused",
            )
            logger.error(
                "muse_vr_plugin: hb0_rr arena build refused inside graph capture (stock path for this call)"
            )
        bed.count(NAME, "stock:arena-capture")
        return None
    with _lock:
        if _state["arena"] is None and _state["arena_err"] is None:
            try:
                from .ops import c3_hb0
                from .ops.lb_hop import HopArena

                g = _group()
                world, rank = int(g.world_size), int(g.rank_in_group)
                nl = len(_state["layer_of"])
                if world not in (2, 4, 8) or nl < 1:
                    raise ValueError(f"world {world}, layers {nl}")
                units = (
                    2 * torch.cuda.get_device_properties(device).multi_processor_count
                )
                a = HopArena(
                    g.cpu_group,
                    device,
                    rank,
                    world,
                    [c3_hb0.far_spec(SITE, FAR_MAX_M, world, units, nl)],
                    async_credit=False,
                )
                a._counts = [nl]
                co = a._off[SITE][1]
                a._views[rank][co : co + 4 * world].view(torch.int32).fill_(
                    nl
                )  # sync credits: transfers < nl released
                d = c3_hb0.done_words(a, SITE)
                for tt in (
                    nl - 3,
                    nl - 2,
                    nl - 1,
                ):  # deferred release: those transfers "ran" (the ring starts at nl)
                    if tt >= 0:
                        d[tt % 3] = tt + 1
                torch.cuda.synchronize(device)
                torch.distributed.barrier(group=g.cpu_group)
                c3_hb0._module(
                    world, True, False, False, 0
                )  # the one far build of this process
                _state["arena"] = a
                _state["flags"] = (
                    c3_hb0.ROWS_READY
                    | c3_hb0.PRE_WAIT_TABLE
                    | c3_hb0.DEFER_RELEASE
                    | c3_hb0.EARLY_TRIGGER
                )
                bed.report(NAME + ":arena", True, world=world, layers=nl, units=units)
            except Exception as e:  # noqa: BLE001 - stay on the stock path if the arena cannot be built
                _state["arena_err"] = f"{type(e).__name__}: {e}"
                bed.report(NAME + ":arena", False, reason=_state["arena_err"])
    return _state["arena"]


def before_expert(kw, m: int, form: str) -> tuple:
    """MoE-LD-W, before its launch: the build defines of an engaged call (else ()). For an engaged call it also runs the
    per-forward sequence / ring advance at the first layer, so that nothing is launched between the expert kernel and the
    finalize."""
    if not enabled() or _state["off"]:
        return ()
    if _state["stash"] is not None:  # an engaged expert call whose finalize never came
        _turn_off("an engaged expert call was not followed by its finalize")
        return ()
    if form != "r48" or not min_m() <= m <= max_m():
        bed.count(NAME, f"stock:M={m}:{form}")
        return ()
    if not _state[
        "fused"
    ]:  # this MoE call's finalize will not carry the push workspace: nothing to replace
        bed.count(NAME, f"stock:no-fused-finalize:M={m}")
        return ()
    w13 = kw["gemm1_weights"]
    layer = _state["layer_of"].get(int(w13.data_ptr()))
    if layer is None:
        bed.count(NAME, "stock:no-layer")
        return ()
    p = int(kw["local_num_experts"])
    try:
        g = _group()
        rank = int(g.rank_in_group)
    except Exception:  # noqa: BLE001
        bed.count(NAME, "stock:no-group")
        return ()
    if int(kw["local_expert_offset"]) != rank * p:
        bed.count(NAME, "stock:placement")
        return ()
    a = _arena(w13.device)
    if a is None:
        bed.count(NAME, "stock:arena")
        return ()
    if layer == 0:
        a.forward_head()
        a.end_step()
    from .ops import moe_ldw

    return moe_ldw.EARLY_SLOTTAB + moe_ldw.ROWS_READY


def after_expert(kw, out, scratch) -> None:
    """MoE-LD-W, after an engaged launch: the pending call the next finalize must consume."""
    from .ops import moe_ldw

    v = _state["views"].get(id(scratch))
    if v is None:
        v = _state["views"][id(scratch)] = moe_ldw.rows_ready_views(scratch)
    w13 = kw["gemm1_weights"]
    _state["stash"] = {
        "g2": out[0],
        "ew": out[1],
        "e2p": out[2],
        "packed": kw["topk_ids"],
        "m": int(kw["topk_ids"].shape[0]),
        "n_local": int(kw["local_num_experts"]),
        "layer": _state["layer_of"][int(w13.data_ptr())],
        "views": v,
    }


def _turn_off(reason: str) -> None:
    _state["off"] = reason
    _state["stash"] = None
    bed.report(NAME, False, reason=reason)
    logger.error("muse_vr_plugin: hb0_rr off for this process: %s", reason)


def forward_cute_dsl(
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
    """AROUND ``FusedPostExpertNormWeightedSumTRTLLM.forward_cute_dsl``: the rows-ready far form for the pending call."""
    st = _state["stash"]
    _state["stash"] = None
    if st is None:
        return original(
            self,
            gemm2_output,
            expert_weights,
            expanded_idx_to_permuted_idx,
            rms_norm_weight,
            output,
            ar_workspace=ar_workspace,
        )
    m = int(output.shape[0])
    ok = (
        ar_workspace is not None
        and gemm2_output.data_ptr() == st["g2"].data_ptr()
        and expanded_idx_to_permuted_idx.data_ptr() == st["e2p"].data_ptr()
        and expert_weights.data_ptr() == st["ew"].data_ptr()
        and m == st["m"]
        and bool(getattr(self, "weight_first", False))
        and rms_norm_weight.dtype == torch.bfloat16
        and output.dtype == torch.bfloat16
        and int(ar_workspace.world_size) == int(_state["arena"].world)
    )
    if not ok:
        # the expert kernel already published for a far call that will not run: off, loudly (peers expire loudly too)
        _turn_off(f"finalize call does not match the pending expert call (M={m})")
        return original(
            self,
            gemm2_output,
            expert_weights,
            expanded_idx_to_permuted_idx,
            rms_norm_weight,
            output,
            ar_workspace=ar_workspace,
        )
    from .ops import c3_hb0

    c3_hb0.far(
        _state["arena"],
        SITE,
        st["packed"],
        expanded_idx_to_permuted_idx.view(-1),
        expert_weights,
        gemm2_output,
        rms_norm_weight,
        output,
        float(self.eps),
        m,
        st["n_local"],
        placement="static",
        index=st["layer"],
        step_flags=_state["flags"],
        rows_ready=st["views"],
    )
    bed.count(NAME, f"far:M={m}")
    return None


def around_forward_experts(original, *args, **kwargs):
    """AROUND ``MuseSpark1xMoE._forward_experts``: record whether this call's finalize is the fused one (SGLang passes
    ``ar_workspace`` only then), for the duration of the call; the previous value is restored on every exit."""
    prev = _state["fused"]
    _state["fused"] = kwargs.get("ar_workspace") is not None
    try:
        return original(*args, **kwargs)
    finally:
        _state["fused"] = prev


def error_word() -> int:
    a = _state["arena"]
    return a.error_word() if a is not None else 0


def register() -> bool:
    if not enabled():
        return False
    from . import fin2

    ok, why = fin2.pins_match()  # the same op contract
    if not ok:
        bed.report(NAME, False, reason=why)
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(TARGET, forward_cute_dsl, HookType.AROUND)
    HookRegistry.register(MOE_FORWARD_TARGET, around_forward_experts, HookType.AROUND)
    bed.report(NAME, True, min_m=min_m(), max_m=max_m(), gate="fused finalize only")
    return True
