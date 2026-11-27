# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Resolve the SGLang megakernel contract across supported host layouts."""

from functools import cache
from importlib import import_module
from types import ModuleType

_SPEC_MODULES = (
    "sglang.kernels.plugin_kernels.megakernel.spec",
    "sglang.kernels.plugin_kernels.nvidia.spec",
)
_LEGACY_SPEC = "sglang.srt.megakernels.spec"


@cache
def load_spec() -> ModuleType:
    for module in _SPEC_MODULES:
        try:
            return import_module(module)
        except ModuleNotFoundError as error:
            missing = error.name or ""
            if not missing.startswith("sglang.") or not (
                module == missing or module.startswith(f"{missing}.")
            ):
                raise
    return import_module(_LEGACY_SPEC)


def __getattr__(name: str):
    if name.startswith("_"):
        raise AttributeError(name)
    return getattr(load_spec(), name)
