# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Run-time resolution of the device-specific pieces of the ``moe_x_kernels`` CuTe-DSL kernels.

The block-scaled tensor-core MMA atoms these kernels use are built by a CuTe DSL helper module that depends on the
device generation. This line serves sm_103 (GB300), so the default is CuTe DSL's ``cutlass.utils.blackwell_helpers``;
``MUSE_VR_MOE_X_MMA_HELPERS`` (a dotted module path) overrides it. The architecture string handed to CuTe DSL capacity queries comes from the device's
compute capability at run time.
"""

from __future__ import annotations

import importlib
import os

HELPERS_ENV = "MUSE_VR_MOE_X_MMA_HELPERS"
DEFAULT_HELPERS = "cutlass.utils.blackwell_helpers"  # sm_100 family, sm_103 included

_state: dict = {"module": None}


def helpers_name() -> str:
    return os.environ.get(HELPERS_ENV, "").strip() or DEFAULT_HELPERS


def helpers_configured() -> bool:
    return bool(helpers_name())


def arch_helpers():
    """The configured MMA-builder module (imported once)."""
    mod = _state["module"]
    if mod is None:
        name = helpers_name()
        mod = importlib.import_module(name)
        for fn in ("make_blockscaled_trivial_tiled_mma", "compute_epilogue_tile_shape"):
            if not hasattr(mod, fn):
                raise RuntimeError(f"{name} has no {fn}()")
        _state["module"] = mod
    return mod


def device_arch(index: int | None = None) -> str:
    """CuTe DSL architecture string of a CUDA device (from its compute capability).

    ``MUSE_VR_MOE_X_ARCH`` overrides it (ahead-of-time builds on a machine without the device)."""
    override = os.environ.get("MUSE_VR_MOE_X_ARCH", "").strip()
    if override:
        return override
    import torch

    major, minor = torch.cuda.get_device_capability(index)
    return f"sm_{major}{minor}"
