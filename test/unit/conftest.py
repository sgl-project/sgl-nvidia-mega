# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Run the unit tests' ``sglang.srt.megakernels.*`` imports on either host layout.

sgl-project/sglang#45389 moved the megakernel contract to sglang.kernels.plugin_kernels.nvidia.spec and
sglang.srt.kernel_plugins.nvidia.binding (binding + runtime + guards in one module); the K3.1 host adapters
moved next to it. On such a host this registers the old module names as aliases of the new modules."""

import importlib
import sys
import types

_MOVED = {
    "spec": "sglang.kernels.plugin_kernels.nvidia.spec",
    "binding": "sglang.srt.kernel_plugins.nvidia.binding",
    "runtime": "sglang.srt.kernel_plugins.nvidia.binding",
    "guards": "sglang.srt.kernel_plugins.nvidia.binding",
    "kimi_k31_decode_binding": "sglang.srt.kernel_plugins.nvidia.kimi_k31_decode_binding",
    "kimi_k31_verify_binding": "sglang.srt.kernel_plugins.nvidia.kimi_k31_verify_binding",
    "kimi_k31_verify_dsa_binding": "sglang.srt.kernel_plugins.nvidia.kimi_k31_verify_dsa_binding",
}

try:
    importlib.import_module("sglang.srt.megakernels")
except ModuleNotFoundError:
    try:
        importlib.import_module("sglang.srt.kernel_plugins.nvidia.binding")
    except ModuleNotFoundError:
        pass  # no host with either layout: the tests skip or fail as before
    else:
        package = types.ModuleType("sglang.srt.megakernels")
        package.__path__ = []
        sys.modules["sglang.srt.megakernels"] = package
        for old, new in _MOVED.items():
            try:
                module = importlib.import_module(new)
            except ModuleNotFoundError:
                continue
            sys.modules[f"sglang.srt.megakernels.{old}"] = module
            setattr(package, old, module)
