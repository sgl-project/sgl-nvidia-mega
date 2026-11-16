# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The SGLang megakernel contract names the Kimi K3/K3.1 megakernels import, from either host layout:
sglang.kernels.plugin_kernels.nvidia.spec + sglang.srt.kernel_plugins.nvidia.binding (sgl-project/sglang#45389
and later) or sglang.srt.megakernels.{spec,binding} (before it)."""

try:
    from sglang.kernels.plugin_kernels.nvidia.spec import (
        BuildContext,
        BuiltMegakernel,
        Feature,
        MegakernelSpec,
        Numerics,
        NumericsKind,
        Requirements,
        Step,
    )
    from sglang.srt.kernel_plugins.nvidia.binding import (
        HybridPoolView,
        HybridStepView,
        PreparationView,
    )
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.megakernels.binding import HybridPoolView, HybridStepView, PreparationView
    from sglang.srt.megakernels.spec import (
        BuildContext,
        BuiltMegakernel,
        Feature,
        MegakernelSpec,
        Numerics,
        NumericsKind,
        Requirements,
        Step,
    )

__all__ = [
    "BuildContext",
    "BuiltMegakernel",
    "Feature",
    "HybridPoolView",
    "HybridStepView",
    "MegakernelSpec",
    "Numerics",
    "NumericsKind",
    "PreparationView",
    "Requirements",
    "Step",
]


# SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": (
        "BuildContext", "BuiltMegakernel", "Feature", "MegakernelSpec", "Numerics", "NumericsKind", "Requirements", "Step",
    ),
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/binding.py": ("HybridPoolView", "HybridStepView", "PreparationView"),
}
