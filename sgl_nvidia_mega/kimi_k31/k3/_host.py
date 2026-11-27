# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""SGLang megakernel contract and runtime views used by Kimi K3/K3.1."""

from sgl_nvidia_mega._sglang_spec import (
    BuildContext,
    BuiltMegakernel,
    Feature,
    MegakernelSpec,
    Numerics,
    NumericsKind,
    Requirements,
    Step,
)

try:
    from sglang.srt.kernel_plugins.nvidia.binding import (
        HybridPoolView,
        HybridStepView,
        PreparationView,
    )
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.megakernels.binding import (
        HybridPoolView,
        HybridStepView,
        PreparationView,
    )

# Names read from SGLang. A contract test checks these before GPU use.
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": (
        "BuildContext",
        "BuiltMegakernel",
        "Feature",
        "MegakernelSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
        "Step",
    ),
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/binding.py": (
        "HybridPoolView",
        "HybridStepView",
        "PreparationView",
    ),
}

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
