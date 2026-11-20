# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The Muse Spark kernels, registered as hooks on SGLang's own forward.

Each component replaces one call inside SGLang's forward -- a norm, an
attention call, a MoE finalize, a boundary reduction -- through
``sglang.srt.plugins.hook_registry``. Every one is opt-in and default off, so a
server that sets none of the ``MUSE_VR_*`` switches runs stock SGLang.

ORDER IS LOAD-BEARING. Several components hook the same call and must wrap each
other in a fixed order; the comments on COMPONENTS say which and why. Do not
sort this list.

A component that cannot engage -- a switch unset, a pinned SGLang source that
has moved, a model that does not match -- reports why and leaves SGLang stock.
It never half-engages: that is the difference between a run that is slower and
a run whose numbers cannot be trusted.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import os
from pathlib import Path

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/compile/paths.py": ("DEFAULT_INCLUDE",),
}


logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_OVERRIDE_INCLUDE = _HERE / "include"
# sha256 of python/sglang/kernels/jit/include/sgl_kernel/warp.cuh at the pinned SGLang revision
_UPSTREAM_WARP_SHA256 = (
    "a4dbbd6a40780cd5f7b815f001ead3b4dccb8cfdb3532937951ccf71a4a747a6",
    "264e64c3f386a70de938ed831f05f7a5f3c4b3ed9cfea58e1fbad433a8f860db",
)
# sha256 of the same header once upstream carries the fix itself (SGLang PR #44137, on main): byte-identical to this
# plugin's copy without its attribution lines, so there is nothing to override
_UPSTREAM_WARP_FIXED_SHA256 = (
    "264e64c3f386a70de938ed831f05f7a5f3c4b3ed9cfea58e1fbad433a8f860db"
)
_state: dict = {"engaged": None, "warp_fixed_upstream": False}

# Order matters; see the module docstring.
COMPONENTS = (
    "nvfp4_kv",  # first: kv_draftwin's sizing hook wraps its per-token bytes
    "ar_attn_boundary",
    "pull_bounded",
    "ctx_proj_overlap",
    "ctx_proj_prefill",  # after ctx_proj_overlap: its projection hook takes only the tensor its SP exit handed out
    "spec_sample_split",
    "draft_candidates",
    "draft_boundary",
    "final_norm",
    "draft_kv_append",
    "input_norm",
    "verify_attn",
    "kv4_dec",  # after verify_attn: its hook wraps verify_attn's
    "gemm_pick",
    "gemm_lt",  # after gemm_pick: takes only calls the (wrapped) rules leave to cuBLAS
    "sp_gate",
    "sp_combine_inplace",
    "pfin",  # wraps the finalize op's CuTe-DSL call (sp_combine_inplace only picks its destination)
    "small_extend_attn",
    "xattn_b",  # own extend-attention op; refuses to engage next to small_extend_attn
    "sp_fp4_dispatch",
    "prefill_qkn",
    "kv_draftwin",
    "verify_topk_sample",  # last: its hooks wrap the other components' hooks
)

# Components whose entry point is not called ``register``.
ENTRY: dict[str, str] = {
    "pull_bounded": "engage",
    "gemm_pick": "engage",
    "sp_gate": "engage",
    "sp_combine_inplace": "engage",
    "pfin": "engage",
    "small_extend_attn": "engage",
    "xattn_b": "engage",
    "sp_fp4_dispatch": "engage",
}

_registered: dict[str, bool] | None = None


def _disabled(name: str) -> bool:
    return name in {
        x.strip()
        for x in os.environ.get("MUSE_VR_PLUGIN_DISABLE", "").split(",")
        if x.strip()
    }


def _engage_warp_redux_volatile() -> bool:
    if _disabled("warp_redux_volatile"):
        logger.warning(
            "muse_spark: warp_redux_volatile disabled by MUSE_VR_PLUGIN_DISABLE"
        )
        return False
    import sglang.kernels.jit.utils.compile.paths as paths

    upstream = Path(paths.DEFAULT_INCLUDE[-1]) / "sgl_kernel" / "warp.cuh"
    try:
        digest = hashlib.sha256(upstream.read_bytes()).hexdigest()
    except OSError as e:
        logger.error(
            "muse_spark: cannot read %s (%s); warp_redux_volatile NOT engaged",
            upstream,
            e,
        )
        return False
    if digest == _UPSTREAM_WARP_FIXED_SHA256:
        _state["warp_fixed_upstream"] = (
            True  # the installed header already has the fix: no override needed
        )
        return False
    if digest != _UPSTREAM_WARP_SHA256:
        logger.error(
            "muse_spark: %s sha256 %s does not match the pinned %s; warp_redux_volatile NOT engaged",
            upstream,
            digest[:16],
            _UPSTREAM_WARP_SHA256[:16],
        )
        return False
    inc = str(_OVERRIDE_INCLUDE)
    if inc not in paths.DEFAULT_INCLUDE:
        paths.DEFAULT_INCLUDE.insert(
            0, inc
        )  # in place: modules that imported the list see the change
    return True


def register_all() -> dict[str, bool]:
    """Register every component once; returns name -> engaged.

    A component that raises is recorded as not engaged and SGLang stays stock
    for it: a failed opt-in must never take the server down.
    """
    global _registered
    if _registered is not None:
        return _registered
    if _state["engaged"] is None:
        from . import bed, sgl_compat

        # Engine modules that moved between SGLang layouts stay importable under the names components use. Only when
        # some MUSE_VR_* setting is present: a process with no component settings imports nothing extra.
        _infra = {"MUSE_VR_PLUGIN_SITE", "MUSE_VR_ENGAGE_DIR", "MUSE_VR_PLUGIN_DISABLE"}
        if any(k.startswith("MUSE_VR_") and k not in _infra for k in os.environ):
            try:
                aliases = sgl_compat.install_aliases()
                bed.report(
                    "sgl_compat",
                    True,
                    layout=sgl_compat.layout(),
                    aliases=sorted(aliases),
                )
                from . import sgl_shims

                sgl_shims.install()  # main layout at a checked revision only; records its own decision
            except Exception as e:  # noqa: BLE001 - components that need a missing alias step aside on their own
                bed.report(
                    "sgl_compat",
                    False,
                    layout=sgl_compat.layout(),
                    reason=f"{type(e).__name__}: {e}",
                )
        else:
            bed.report(
                "sgl_compat",
                False,
                layout=sgl_compat.layout(),
                reason="no component enabled",
            )
        _state["engaged"] = _engage_warp_redux_volatile()
        if _state["warp_fixed_upstream"]:
            bed.report(
                "warp_redux_volatile",
                False,
                reason="the installed warp.cuh already carries the fix",
            )
        else:
            bed.report("warp_redux_volatile", _state["engaged"])
    from sgl_nvidia_mega.muse_spark.components import bed

    _registered = {}
    for name in COMPONENTS:
        try:
            mod = importlib.import_module(f"{__name__}.{name}")
            ok = bool(getattr(mod, ENTRY.get(name, "register"))())
        except Exception as e:  # noqa: BLE001 - a failed opt-in leaves SGLang stock
            bed.report(
                name, False, reason=f"registration failed: {type(e).__name__}: {e}"
            )
            ok = False
        _registered[name] = ok
    on = sorted(k for k, v in _registered.items() if v)
    logger.info(
        "muse_spark: %d of %d components engaged: %s", len(on), len(COMPONENTS), on
    )
    return _registered
