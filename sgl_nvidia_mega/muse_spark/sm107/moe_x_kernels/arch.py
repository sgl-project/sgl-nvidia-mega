# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Run-time resolution of the device-specific pieces of the ``moe_x_kernels`` CuTe-DSL kernels.

The block-scaled tensor-core MMA atoms these kernels use are built by CuTe DSL's Rubin helper module,
``cutlass.utils.rubin_helpers`` (nvidia-cutlass-dsl 4.8.0, which SGLang pins). The architecture string handed to CuTe
DSL capacity queries comes from the device's compute capability at run time.
"""

from __future__ import annotations

import os


def arch_helpers():
    """CuTe DSL's Rubin MMA-builder module (imported when the kernels are traced)."""
    import cutlass.utils.rubin_helpers as rubin_helpers

    return rubin_helpers


def device_arch(index: int | None = None) -> str:
    """CuTe DSL architecture string of a CUDA device (from its compute capability).

    ``MUSE_VR_MOE_X_ARCH`` overrides it (ahead-of-time builds on a machine without the device)."""
    override = os.environ.get("MUSE_VR_MOE_X_ARCH", "").strip()
    if override:
        return override
    import torch

    major, minor = torch.cuda.get_device_capability(index)
    return f"sm_{major}{minor}"
