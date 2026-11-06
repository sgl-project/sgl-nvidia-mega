# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Run-time resolution of the device-specific pieces of the ``moe_x_kernels`` CuTe-DSL kernels.

The block-scaled tensor-core MMA atoms these kernels use are built by a CuTe DSL helper module that depends on the
device generation. The deployment names that module in ``MUSE_VR_MOE_X_MMA_HELPERS`` (a dotted module path); this
package itself names no device. The architecture string handed to CuTe DSL capacity queries comes from the device's
compute capability at run time.
"""

from __future__ import annotations

import importlib
import os

HELPERS_ENV = "MUSE_VR_MOE_X_MMA_HELPERS"

_state: dict = {"module": None}


# Device generations whose tensor cores the kernels are written for: they issue block-scaled MMAs with instruction
# K = 128 and operand-collector hints, and write the activation scale factors to tensor memory in that instruction's
# layout. The sm_100 family (sm_103 / GB300 included) has K = 64 block-scaled MMAs without collector hints, so the
# kernels need a port of that layout before they can run there.
SUPPORTED_ARCHS = frozenset({"sm_107"})


def unsupported_reason(index: int | None = None) -> str | None:
    """Why the kernels cannot run on the device, or None when they can."""
    arch = device_arch(index)
    if arch not in SUPPORTED_ARCHS:
        return (
            f"{arch}: the kernels use K=128 block-scaled MMAs with operand collectors and their tensor-memory "
            "scale layout; not ported to this architecture"
        )
    return None


def helpers_configured() -> bool:
    return bool(os.environ.get(HELPERS_ENV, "").strip())


def arch_helpers():
    """The configured MMA-builder module (imported once)."""
    mod = _state["module"]
    if mod is None:
        name = os.environ.get(HELPERS_ENV, "").strip()
        if not name:
            raise RuntimeError(
                f"{HELPERS_ENV} is not set: no MMA builder module for this device"
            )
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
