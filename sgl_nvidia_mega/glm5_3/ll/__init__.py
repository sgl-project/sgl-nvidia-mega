# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Low-latency kernel units for GLM-5.3 / GLM-5.2 beside the TP8 families (TP4 verify and draft, TP8
draft decode): sparse MLA flash decode, the TP4 attention front, registered small-M
GEMMs, the TP4 hops and MoE, and the draft's pieces.

Each module in ``CANDIDATE_MODULES`` is optional: present, it is imported (any import
error raises) and its ``CANDIDATES`` / ``FAMILIES`` join the registry; it may define
``TOLERANCES`` (a name -> value dict, each commented with the test that measures it),
which the spec's ``NUMERICS`` states. Modules import their kernels lazily: the spec
imports this package at argument resolution.

``linear(role, module, x)`` is the one GEMM entry point of the units: the registered
small-M GEMM of ``ll.gemm`` when that module is present, else the module's own local
GEMM. Every row count a role will run at is declared at family build
(``build_linear``), since the fused step is first recorded under capture.
"""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType
from typing import Dict, Optional

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/layers/quantization/unquant.py": ("UnquantizedLinearMethod", "apply"),
    "srt/models/deepseek_v2.py": ("MoEGate",),
}

# the optional unit modules, in registration order (their families build in this
# order)
CANDIDATE_MODULES = ("sparse_mla", "attention", "gemm", "hops", "moe", "draft")
# the GEMM roles (ll.gemm); each is the unit "ll.gemm.<role>" (units.py)
GEMM_ROLES = (
    "wk_weights_proj",
    "o_proj",
    "router",
    "dense_down",
    "eh_proj",
    "lm_head",
    "q_b",
)

_modules: Optional[Dict[str, ModuleType]] = None


def modules() -> Dict[str, ModuleType]:
    """The present unit modules by name, imported once (an import error raises)."""
    global _modules
    if _modules is None:
        found = {}
        for name in CANDIDATE_MODULES:
            full = f"{__name__}.{name}"
            if importlib.util.find_spec(full) is not None:
                found[name] = importlib.import_module(full)
        _modules = found
    return _modules


def module(name: str) -> Optional[ModuleType]:
    """The unit module ``name`` if present, else None."""
    return modules().get(name)


def tolerances() -> Dict[str, float]:
    """The named numeric tolerances the present unit modules state."""
    out: Dict[str, float] = {}
    for name, mod in modules().items():
        for key, value in getattr(mod, "TOLERANCES", {}).items():
            if key in out and out[key] != value:
                raise ValueError(f"ll.{name}: tolerance {key} is stated twice")
            out[key] = float(value)
    return out


def linear(role: str, module, x: torch.Tensor, *, out_dtype=None) -> torch.Tensor:
    """``x @ module.weight.T`` for ``x`` [M, K]: no bias, no tensor-parallel reduction
    (the caller's hop reduces). The router returns FP32 logits, every other role BF16.
    ``ll.gemm.linear`` when present (it raises at a planned row count that
    ``build_linear`` never prepared), else the module's own local GEMM."""
    gemm = _gemm()
    if gemm is not None:
        return gemm.linear(role, module, x, out_dtype=out_dtype)
    out = stock_linear(role, module, x)
    if out_dtype is not None and out.dtype != out_dtype:
        out = out.to(out_dtype)
    return out


def build_linear(role: str, module, rows, *, sm=None, sms=None) -> str:
    """At family build (outside capture): prepare ``role`` on ``module`` for every row
    count in ``rows`` it will run at. Returns a description for the build log ("" when
    the role stays on the module's own GEMM)."""
    gemm = _gemm()
    if gemm is not None:
        return gemm.build(role, module, rows, sm=sm, sms=sms)
    return ""


def stock_linear(role: str, module, x: torch.Tensor) -> torch.Tensor:
    """The module's own local GEMM of ``x`` (``ll.gemm.stock`` keeps the same body)."""
    if role == "router":
        return module(x)  # MoEGate.forward: FP32 logits
    if role == "lm_head":
        return torch.matmul(x, module.weight.T)
    qm = getattr(module, "quant_method", None)
    if qm is not None:
        return qm.apply(module, x, bias=None)  # local GEMM, no all-reduce
    return torch.nn.functional.linear(x, module.weight)


def _gemm() -> Optional[ModuleType]:
    return modules().get("gemm")
