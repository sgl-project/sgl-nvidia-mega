# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/layers/layer_boundary/stage.py, prepare.py, residual/batch.py and
# fusions/cutedsl.py (the stage boundary's CuTe DSL fusion points).
"""Slot hop: the decode hop (``sm103/hop``) in place of FlashInfer's MNNVL CuTe DSL
kernels at the three points of a covered step where SGLang's stage boundary runs them:

- ``layer_input``: the previous MoE's finalize + all-reduce + residual add + input norm;
- ``moe_input``: the mixer's all-reduce + residual add + post-attention norm;
- ``final_norm``: the last MoE's finalize + all-reduce + residual add + final norm.

Each one mirrors the boundary's path for a covered step, which build checks on every
layer (the ordinary batch rows, the CuTe DSL kernel tried first, nothing moved, sliced
or gathered around it). It runs the hop exactly where that path runs the CuTe DSL
kernel, by SGLang's own eligibility checks, and leaves the residual stream as the
boundary does; anywhere else it returns None and SGLang's boundary runs. At these rows
the CuTe DSL fusion runs FlashInfer's low-latency kernels, whose outputs the hop's
equal bit for bit.
"""

from __future__ import annotations

import logging
from functools import partial
from types import MethodType
from typing import Optional

import msgspec
import torch

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/qwen3_5.py": (
        "flashinfer_mnnvl_cutedsl_fusion",
        "layers",
        "start_layer",
        "end_layer",
        "norm",
        "layer_id",
        "attn_boundary",
        "ffn_boundary",
    ),
    "srt/layers/layer_boundary/stage.py": ("plan", "declaration", "kind"),
    "srt/layers/layer_boundary/factories.py": ("row_transport",),
    "srt/layers/layer_boundary/construction.py": (
        "BatchVariant",
        "paths",
        "fusions",
        "norm",
        "qkv_latent_func",
        "finishes_directly",
        "_publish_lora_layout",
        "variant_for",
    ),
    "srt/layers/layer_boundary/contracts.py": (
        "ORDINARY",
        "entry",
        "output",
        "prepare",
        "input_move",
        "attn_input_adapter",
        "always_partial",
        "sum_impl",
    ),
    "srt/layers/layer_boundary/prepare.py": (
        "_run_entry",
        "_run_entry(step)",
        "_run_entry(carried_fusions)",
        "_run_entry(carried_read_fusions)",
        "_reduce_update_read",
        "_reduce_update_read(gathers_residual)",
        "_reduce_update_read(fusions)",
        "_reduce_update_read(read_fusions)",
        "_reduce_update_read(read_fusion_indices)",
        "_attn_input_default",
    ),
    "srt/layers/layer_boundary/fusions/cutedsl.py": (
        "CuteDSLFusion",
        "_finalize_add_norm",
        "_ffn_input_all_reduce_add_norm",
        "_finalize_eligible",
        "_eligible",
        "hidden_size",
        "top_k",
        "rms_epsilon",
    ),
    "srt/layers/layer_boundary/fusions/moe_finalize.py": (
        "MoeDeferredFinalize",
        "routed_output",
        "expert_weights",
        "permuted_indices",
        "gated_shared_output",
        "m",
    ),
    "srt/layers/layer_boundary/fusions/norm.py": ("fused_norm_gamma",),
    "srt/layers/layer_boundary/residual/batch.py": ("stream_of", "clear"),
    "srt/layers/layer_boundary/residual/stream.py": (
        "DeclaredSum",
        "pending",
        "owed",
        "update",
        "input",
        "write",
        "export",
        "export(takes_handoff)",
    ),
    "srt/layers/layer_boundary/residual/add_norm.py": ("is_plain_add",),
    "srt/layers/layernorm.py": ('"gemma_weight"',),
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/runtime.py": (
        "megakernel_agree",
    ),
    "srt/runtime_context.py": ("get_parallel", '"tp_group"', "get_exec"),
    "srt/distributed/parallel_state.py": ("device_group", "world_size"),
}

# The SGLang functions whose body this module mirrors, by the digest of their code
# (test/unit/test_contract.py recomputes them on the installed SGLang)
SGLANG_MIRRORS = {
    "srt/layers/layer_boundary/stage.py:StageBoundary._prepare": "3ade6a2952490616",
    "srt/layers/layer_boundary/stage.py:StageBoundary._prepare_attention": "ab2768d2e68434a5",
    "srt/layers/layer_boundary/stage.py:StageBoundary._prepare_input": "53b7f2a5bd394d76",
    "srt/layers/layer_boundary/stage.py:StageBoundary._select": "f317ed475a7672cf",
    "srt/layers/layer_boundary/prepare.py:_run_entry": "639740496fc09d2d",
    "srt/layers/layer_boundary/prepare.py:_reduce_update_read": "67badae13f96d80f",
    "srt/layers/layer_boundary/fusions/cutedsl.py:CuteDSLFusion._finalize_add_norm": "51d463a1cecbbb64",
    "srt/layers/layer_boundary/fusions/cutedsl.py:CuteDSLFusion._ffn_input_all_reduce_add_norm": "010e06f02e64affa",
    "srt/layers/layer_boundary/residual/batch.py:final_norm": "13008609a1a7fd8a",
}

# the (point, rows) the slot has run at (each logged once, as engagement evidence)
_ENGAGED: set = set()
# the target's hop once built (active())
_ACTIVE: Optional[object] = None


class Bound(msgspec.Struct, frozen=True):
    """The boundary package's build result (``st.bound["boundary"]``)."""

    hop: Optional[object] = None  # sm103.hop.Hop for every covered width, or None
    reason: str = ""  # why there is no hop


def build(model, ctx) -> Bound:
    """The hop when slot ``hop`` is requested and every layer's boundary is the path
    mirrored here. A collective: every rank builds it or none does."""
    try:
        from sglang.srt.kernel_plugins.nvidia.binding import megakernel_agree
    except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
        from sglang.srt.megakernels.runtime import megakernel_agree
    from sglang.srt.runtime_context import get_parallel

    from sgl_nvidia_mega.qwen3_5 import plan, runtime
    from sgl_nvidia_mega.qwen3_5.sm103.hop import Hop

    if "hop" not in runtime.SLOTS:
        return Bound(reason="slot off")
    # the widest step the hop covers: its mailbox's rows
    rows = plan.widest(ctx.widths, plan.HOP_MAX_ROWS)
    if not rows:
        return Bound(reason=f"no width of {list(ctx.widths)} within its rows")
    m = model.model
    group = get_parallel().tp_group
    reasons = refusals(m, group.world_size)
    # a rank without the hop would leave its peers' consumers waiting
    megakernel_agree(reasons, what="hop refusals")
    if reasons:
        return Bound(reason="; ".join(reasons))
    hop = Hop(
        max_m=rows,
        eps=m.flashinfer_mnnvl_cutedsl_fusion.rms_epsilon,
        group=group.device_group,
        device=torch.device("cuda", torch.cuda.current_device()),
        deadline_s=ctx.deadline_s,
    )
    global _ACTIVE
    _ACTIVE = hop
    return Bound(hop=hop)


def active() -> Optional[object]:
    """The target's built hop (sm103.hop.Hop), or None: the MTP draft's hops share its
    mailbox and stage order (built before the draft, which loads after the target)."""
    return _ACTIVE


def refusals(m, world: int) -> list[str]:
    """Why the hop cannot stand in for the CuTe DSL kernels of ``m``
    (Qwen3_5ForCausalLM) at TP ``world``; [] when it can."""
    from sgl_nvidia_mega.qwen3_5.sm103 import hop as kernels

    service = m.flashinfer_mnnvl_cutedsl_fusion
    if service is None:
        return ["SGLang runs no FlashInfer MNNVL CuTe DSL fusion"]
    out = []
    if service.hidden_size != kernels.HIDDEN or service.top_k > kernels.MAX_TOP_K:
        out.append(
            f"hidden {service.hidden_size}, top-{service.top_k} (the hop serves "
            f"hidden {kernels.HIDDEN}, up to top-{kernels.MAX_TOP_K})"
        )
    if world not in kernels.WORLDS:
        out.append(f"TP {world} (the hop serves TP {kernels.WORLDS})")
    paths = [
        (m.layers[i].layer_id, why)
        for i in range(m.start_layer, m.end_layer)
        for why in path_refusals(m.layers[i])
    ]
    if paths:
        more = f" (and {len(paths) - 1} more)" if len(paths) > 1 else ""
        out.append(f"layer {paths[0][0]}: {paths[0][1]}{more}")
    return out


def path_refusals(layer) -> list[str]:
    """Why the ordinary paths of ``layer``'s two stages are not the ones mirrored."""
    from sglang.srt.layers.layer_boundary.construction import BatchVariant
    from sglang.srt.layers.layer_boundary.fusions.cutedsl import CuteDSLFusion
    from sglang.srt.layers.layer_boundary.fusions.norm import fused_norm_gamma
    from sglang.srt.layers.layer_boundary.prepare import (
        _attn_input_default,
        _reduce_update_read,
        _run_entry,
    )

    out = []
    for stage in (layer.attn_boundary, layer.ffn_boundary):
        plan, name = stage.plan, stage.kind.name.lower()
        if (
            not isinstance(plan.fusions, CuteDSLFusion)
            or fused_norm_gamma(plan.norm) is None
        ):
            out.append(f"its {name} input runs no CuTe DSL fusion")
            continue
        path = plan.paths.get(BatchVariant.ORDINARY)
        entry = path.entry if path is not None else None
        prepare = entry.prepare if entry is not None else None
        if (
            not isinstance(prepare, partial)
            or prepare.func is not _run_entry
            or prepare.keywords["carried_read_fusions"]
            or entry.input_move is not None
            or entry.attn_input_adapter not in (None, _attn_input_default)
            or plan.qkv_latent_func is not None
            or plan._publish_lora_layout
            or stage.declaration.row_transport is not None
        ):
            out.append(f"its {name} input moves or reads more than the mirrored path")
            continue
        if stage is layer.attn_boundary:
            # the finalize handoff; then the mixer's sum, which the MoE input completes
            takes = (
                _first_is(
                    prepare.keywords["carried_fusions"],
                    CuteDSLFusion._finalize_add_norm,
                    plan,
                )
                and plan.finishes_directly
                and path.output.always_partial
                and path.output.sum_impl is None
            )
        else:
            step = prepare.keywords["step"]
            takes = (
                isinstance(step, partial)
                and step.func is _reduce_update_read
                and not step.keywords["gathers_residual"]
                and not step.keywords.get("read_fusions")
                and not step.keywords.get("read_fusion_indices")
                and _first_is(
                    step.keywords["fusions"],
                    CuteDSLFusion._ffn_input_all_reduce_add_norm,
                    plan,
                )
            )
        if not takes:
            out.append(f"its {name} input does not try the CuTe DSL kernel first")
    return out


def _first_is(candidates, method, plan) -> bool:
    """Whether the first of ``candidates`` is ``plan``'s CuTe DSL ``method`` on ``plan``."""
    first = candidates[0] if candidates else None
    return (
        isinstance(first, partial)
        and isinstance(first.func, MethodType)
        and first.func.__func__ is method
        and first.func.__self__ is plan.fusions
        and first.args == (plan,)
    )


def layer_input(hop, layer, hidden, st):
    """``attn_boundary.prepare`` of a layer whose input completes the previous MoE's
    finalize handoff (``hidden``), with the hop as its kernel; None where that
    prepare would not run the CuTe DSL kernel."""
    from sglang.srt.layers.layer_boundary.fusions.moe_finalize import (
        MoeDeferredFinalize,
    )
    from sglang.srt.layers.layer_boundary.fusions.norm import fused_norm_gamma
    from sglang.srt.layers.layer_boundary.residual.batch import stream_of

    stream = stream_of(st.fb)
    pending, plan = stream.pending, layer.attn_boundary.plan
    if not (
        pending is not None
        and isinstance(pending.owed, MoeDeferredFinalize)
        and _ordinary(plan, st, pending)
        and _takes(hop, pending.owed)
        and plan.fusions._finalize_eligible(plan, st.fb, pending.owed.m)
    ):
        return None
    owed, residual = stream.input(hidden)
    norm, residual = _finalize(
        hop, "layer input", owed, residual, fused_norm_gamma(plan.norm)
    )
    stream.write(residual)
    return norm


def takes_moe_input(hop, layer, st, rows: int) -> bool:
    """Whether ``moe_input`` runs the hop for a mixer output of ``rows`` rows, as far as
    the batch decides it (the stream's sum and update are checked there): a mixer may
    push its partial into the hop only then."""
    from sglang.srt.layers.layer_boundary.construction import BatchVariant
    from sglang.srt.runtime_context import get_exec

    plan = layer.ffn_boundary.plan
    return (
        hop.supports(rows)
        and plan.variant_for(st.fb) is BatchVariant.ORDINARY
        and plan.fusions._eligible(plan, st.fb, rows)
        and not get_exec().comm.enable_quant_communications
    )


def moe_input(hop, layer, hidden, st):
    """``ffn_boundary.prepare`` of the mixer's partial output (``hidden``, as
    ``attn_boundary.finish`` recorded it), with the hop as its kernel; None where that
    prepare would not run the CuTe DSL kernel. A partial the mixer already pushed is
    only consumed."""
    from sglang.srt.layers.layer_boundary.fusions.norm import fused_norm_gamma
    from sglang.srt.layers.layer_boundary.residual.batch import stream_of
    from sglang.srt.layers.layer_boundary.residual.stream import DeclaredSum

    stream = stream_of(st.fb)
    pending, plan = stream.pending, layer.ffn_boundary.plan
    if not (
        pending is not None
        and isinstance(pending.owed, DeclaredSum)
        and _ordinary(plan, st, pending)
    ):
        return None
    part, residual = stream.input(hidden)
    rows = part.shape[0]
    if not (
        residual is not None
        and part.dtype == torch.bfloat16
        and takes_moe_input(hop, layer, st, rows)
    ):
        return None
    _engage("MoE input", rows)
    norm, residual = hop.all_reduce_residual_rms_norm(
        local_contribution=part, residual=residual, gamma=fused_norm_gamma(plan.norm)
    )
    stream.write(residual)
    return norm


def final_norm(hop, hidden, st):
    """``residual_batch.final_norm`` of the last MoE's finalize handoff (``hidden``)
    with the hop as its kernel; None when the stream holds no handoff the hop takes."""
    from sglang.srt.layers.layer_boundary.fusions.moe_finalize import (
        MoeDeferredFinalize,
    )
    from sglang.srt.layers.layer_boundary.residual.batch import clear, stream_of

    stream = stream_of(st.fb)
    pending = stream.pending
    if not (
        pending is not None
        and isinstance(pending.owed, MoeDeferredFinalize)
        and pending.update.is_plain_add
        and _takes(hop, pending.owed)
    ):
        return None
    owed, residual = stream.export(hidden, takes_handoff=True)
    clear(st.fb)
    norm, _ = _finalize(hop, "final norm", owed, residual, st.model.norm.gemma_weight)
    return norm


def _ordinary(plan, st, pending) -> bool:
    """A plain residual add on the ordinary batch rows: the path build checked."""
    from sglang.srt.layers.layer_boundary.construction import BatchVariant

    return (
        pending.update.is_plain_add and plan.variant_for(st.fb) is BatchVariant.ORDINARY
    )


def _takes(hop, owed) -> bool:
    """Whether the hop's finalize reads this handoff's rows as SGLang wrote them."""
    shared = owed.gated_shared_output
    return (
        hop.supports(owed.m)
        and owed.routed_output.dtype == torch.bfloat16
        and owed.expert_weights.dtype == torch.bfloat16
        and owed.permuted_indices.dtype == torch.int32
        and (shared is None or shared.dtype == torch.bfloat16)
    )


def _finalize(hop, point, owed, residual, gamma):
    _engage(point, owed.m)
    return hop.moe_finalize_all_reduce_rms_norm(
        routed_output=owed.routed_output,
        expert_weights=owed.expert_weights,
        permuted_indices=owed.permuted_indices,
        gated_shared_output=owed.gated_shared_output,
        residual=residual,
        gamma=gamma,
    )


def _engage(point: str, rows: int) -> None:
    if (point, rows) not in _ENGAGED:
        _ENGAGED.add((point, rows))
        logger.info(
            "NVIDIA megakernel: Qwen3.5 slot hop runs the %s at %d rows", point, rows
        )
