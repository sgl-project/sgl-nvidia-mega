# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``unorm_boundary`` (env ``MUSE_VR_UNORM_BOUNDARY=1``, default off).

SGLang's decoder boundaries (the attention boundary after the attention all-reduce, and the FFN boundary after the
unpooler) run ``FusedPostNormResidualPreNorm.forward_jit``: post-norm, channel-wise residual gate and the next pre-norm in
one JIT kernel. This component replaces that kernel, for the row counts listed in ``MUSE_VR_UNORM_FORMS`` only, by the
boundary norm in ``ops/unorm.py`` (``csrc/unorm.cuh`` / ``csrc/unorm_cl.cuh``): same thread map, same summation orders
and roundings, byte-identical outputs (tests/test_unorm.py, tests/test_unorm_boundary.py), with the static operands
copied before the dependency wait and an earlier dependent-launch trigger.

``MUSE_VR_UNORM_FORMS`` = ``<pos>:<tp>:<rows>=<rows_per_cta>:<cluster>:<trigger>:<prefetch>;...`` where ``pos`` is ``U``
(FFN boundary, input-scale post-norm) or ``A`` (attention boundary), ``tp`` the tensor-parallel size and ``rows`` the
batch row count. Every other call runs SGLang's kernel unchanged; the reason is tallied (``stock:<reason>``) and logged
once per (position, rows, reason). An empty table leaves the component off.

Static operands (norm weights, gate coefficients) are read before the dependency wait only after they were recorded with
``unorm.mark_static_synced`` (a full device sync, done at the first eager call of each boundary, never inside graph
capture). A boundary first reached inside capture uses the form that reads them after the wait (same bytes).

The attention boundary's post-norm has no weight; the op takes a vector of exact 1.0 instead (multiplying by 1.0 is
exact, and the op does not flush subnormals: tests/test_unorm_boundary.py checks subnormal inputs byte for byte).
"""

from __future__ import annotations

import logging
import os

import torch

from sgl_nvidia_mega.muse_spark.components import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/distributed/*.py": ("get_tp_group",),
    "kernels/ops/layernorm/muse_spark_norm_residual.py|kernels/ops/muse_spark_v12/fused_norm_residual.py": (
        "FusedPostNormResidualPreNorm",
        "forward_jit",
        "_triton_supported",
        "jit_selected",
        "_ensure_cached_weights",
    ),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}

# The SGLang functions whose body this module mirrors, by the digest of their code (test/unit/test_contract.py
# recomputes them on the installed SGLang)
SGLANG_MIRRORS = {
    # the boundary call this hook replaces and the operands it passes (SGLang cf6133a and 306ccf6)
    "kernels/ops/layernorm/muse_spark_norm_residual.py:FusedPostNormResidualPreNorm.forward_jit": "3878afc5a00035cb",
    "kernels/ops/layernorm/muse_spark_norm_residual.py:FusedPostNormResidualPreNorm.jit_kernel_kwargs": "f3ddcac4283845af",
    # the launcher whose kernel ops/unorm.py reproduces (csrc/unorm*.cuh; the kernel source is listed in NOTICE).
    # SGLang cf6133a, then 306ccf6: cf6133a reads the cluster capability through current_arch().
    "kernels/ops/layernorm/muse_spark_norm_residual_jit.py:fused_norm_residual": (
        "975cd488a7dc2b2a",
        "2c23bbac7ad02b49",
    ),
}


logger = logging.getLogger(__name__)

NAME = "unorm_boundary"  # env MUSE_VR_UNORM_BOUNDARY
FORMS_ENV = "MUSE_VR_UNORM_FORMS"
T_FORWARD_JIT = "sglang.kernels.ops.muse_spark_v12.fused_norm_residual.FusedPostNormResidualPreNorm.forward_jit"

_ARG_NAMES = ("stream", "branch", "branch_dtype", "router_dtype", "pre_norm_dtype")
_state: dict = {"forms": {}, "tp": None, "ones": {}, "logged": set()}


def parse_forms(text: str) -> dict:
    """``{(pos, tp, rows): (rows_per_cta, cluster, trigger, prefetch)}``; raises ValueError on a malformed entry."""
    out = {}
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        key, val = part.split("=")
        pos, tp, m = key.split(":")
        if pos not in ("U", "A"):
            raise ValueError(f"{FORMS_ENV}: position {pos!r} is not U or A")
        r, kc, tr, pf = (int(v) for v in val.split(":"))
        if r < 1 or kc not in (1, 2, 4, 8) or tr not in (0, 1, 2) or pf not in (0, 1):
            raise ValueError(f"{FORMS_ENV}: bad form {val!r}")
        out[(pos, int(tp), int(m))] = (r, kc, tr, pf)
    return out


def _tp() -> int:
    if _state["tp"] is None:
        from sglang.srt.distributed import get_tp_group

        _state["tp"] = int(get_tp_group().world_size)
    return _state["tp"]


def _stock(original, boundary, args, kwargs, pos, m, why):
    bed.count(NAME, f"stock:{why}")
    key = (pos, m, why)
    if key not in _state["logged"]:
        _state["logged"].add(key)
        logger.warning(
            "muse_vr_plugin: %s steps aside (%s) at position %s, %s rows",
            NAME,
            why,
            pos,
            m,
        )
    return original(boundary, *args, **kwargs)


def _triton_supported(
    boundary, stream, branch, branch_dtype, router_dtype, pre_norm_dtype
) -> bool:
    """SGLang's own eligibility test for its fused kernel (forward_jit falls back to the PyTorch form when it fails)."""
    try:
        return bool(
            boundary._triton_supported(
                stream, branch, branch_dtype, router_dtype, pre_norm_dtype
            )
        )
    except TypeError:  # revisions without the pre-norm dtype argument
        return bool(
            boundary._triton_supported(stream, branch, branch_dtype, router_dtype)
        )


def _static(boundary, branch_dtype, device):
    """The four static operands in the op's order (post weight, pre weight, gate alpha, gate beta) or None."""
    boundary._ensure_cached_weights(branch_dtype)
    post_w = boundary._cc_post_weight if boundary._post_norm.has_weight else None
    pre_w = boundary._cc_pre_weight if boundary._pre_norm.has_weight else None
    if not boundary._use_residual_gate or pre_w is None:
        return None
    if post_w is None:
        if boundary._post_norm_is_input_scale:
            return None
        ones = _state["ones"].get((device.index, pre_w.numel()))
        if ones is None:
            if torch.cuda.is_current_stream_capturing():
                return None  # built only outside capture
            ones = torch.ones(pre_w.numel(), dtype=torch.bfloat16, device=device)
            _state["ones"][(device.index, pre_w.numel())] = ones
        post_w = ones
    return post_w, pre_w, boundary._cc_alpha, boundary._cc_beta


def around_forward_jit(original, boundary, *args, **kwargs):
    from .ops import unorm as U

    if len(args) > len(_ARG_NAMES) or set(kwargs) - set(_ARG_NAMES[len(args) :]):
        return original(boundary, *args, **kwargs)
    kw = dict(zip(_ARG_NAMES, args))
    kw.update(kwargs)
    stream, branch = kw.get("stream"), kw.get("branch")
    branch_dtype = kw.get("branch_dtype", torch.bfloat16)
    router_dtype, pre_norm_dtype = kw.get("router_dtype"), kw.get("pre_norm_dtype")
    pos = "U" if boundary._post_norm_is_input_scale else "A"
    if stream is None or branch is None or branch.dim() != 2:
        return _stock(original, boundary, args, kwargs, pos, -1, "layout")
    m = int(branch.shape[0])
    form = _state["forms"].get((pos, _tp(), m))
    if form is None:
        return _stock(original, boundary, args, kwargs, pos, m, "unlisted")
    if pre_norm_dtype is not None:
        return _stock(original, boundary, args, kwargs, pos, m, "pre_norm_dtype")
    if (
        branch_dtype != torch.bfloat16
        or branch.dtype != torch.bfloat16
        or stream.dtype != torch.float32
    ):
        return _stock(original, boundary, args, kwargs, pos, m, "dtypes")
    if router_dtype not in (None, torch.float32):
        return _stock(original, boundary, args, kwargs, pos, m, "router_dtype")
    if (
        not branch.is_contiguous()
        or not stream.is_contiguous()
        or stream.shape != branch.shape
    ):
        return _stock(original, boundary, args, kwargs, pos, m, "layout")
    jit_selected = getattr(boundary, "jit_selected", None)
    if jit_selected is None or not jit_selected():
        return _stock(original, boundary, args, kwargs, pos, m, "not_jit")
    if branch.numel() == 0 or not _triton_supported(
        boundary, stream, branch, branch_dtype, router_dtype, pre_norm_dtype
    ):
        return _stock(original, boundary, args, kwargs, pos, m, "unsupported")
    static = _static(boundary, branch_dtype, branch.device)
    if static is None:
        return _stock(original, boundary, args, kwargs, pos, m, "operands")
    post_w, pre_w, alpha, beta = static
    if not U.supported(m, int(branch.shape[1]), alpha, beta, post_w, pre_w):
        return _stock(original, boundary, args, kwargs, pos, m, "unsupported")
    a1, b1 = alpha.reshape(-1), beta.reshape(-1)
    synced = U.static_is_synced(a1, b1, post_w, pre_w)
    if not synced and not torch.cuda.is_current_stream_capturing():
        U.mark_static_synced(
            a1, b1, post_w, pre_w
        )  # full device sync; once per boundary
        synced = True
    r, kc, tr, pf = form
    out = U.norm_residual(
        branch,
        stream,
        boundary._post_norm.eps,
        boundary._pre_norm.eps,
        post_norm_is_input_scale=boundary._post_norm_is_input_scale,
        post_weight=post_w,
        pre_weight=pre_w,
        alpha=a1,
        beta=b1,
        want_router=router_dtype is not None,
        rows=r,
        trig=tr,
        static_pre_wait=synced,
        cluster=kc,
        prefetch_stream=bool(pf),
    )
    bed.count(NAME, f"fused:{pos}:tp{_tp()}:M{m}" + ("" if synced else ":unsynced"))
    return out


def register() -> bool:
    if not bed.enabled(NAME):
        return False
    try:
        forms = parse_forms(os.environ.get(FORMS_ENV, ""))
    except ValueError as e:
        bed.report(NAME, False, reason=str(e))
        return False
    if not forms:
        bed.report(NAME, False, reason=f"{FORMS_ENV} is empty")
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    _state["forms"] = forms
    HookRegistry.register(T_FORWARD_JIT, around_forward_jit, HookType.AROUND)
    bed.report(
        NAME,
        True,
        forms=";".join(
            f"{p}:{t}:{m}={'%d:%d:%d:%d' % f}" for (p, t, m), f in sorted(forms.items())
        ),
    )
    return True
