# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""SGLang source-layout compatibility for the plugin's imports, hook targets and source pins.

SGLang moved several Muse Spark modules between the branch the plugin was first built on (``kernels/ops/muse_spark_v12``,
``kernels/jit/csrc/muse_spark_v12``, flat ``srt/layers/muse_spark_*.py``) and SGLang main (``kernels/jit/csrc/muse_spark``,
``srt/layers/muse_spark/``, and operators regrouped by computation under ``kernels/ops/``). Components name every engine
module, hook target and pinned source by its original location; this module resolves each name to the location the
installed SGLang actually has, trying the original location first. Nothing here changes behaviour: a name that resolves
nowhere stays unresolved, and the caller steps aside as it does for any other pin mismatch.

``module(name)``      import a module by its original dotted name (``ImportError`` when no location exists)
``target(dotted)``    hook-target string (``module.Class.attr``) rewritten to the installed module location
``relpath(path)``     pinned source path (relative to the directory holding ``sglang``) at the installed location
``layout()``          ``"branch"`` (original locations present) or ``"main"``, for engagement records
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
from pathlib import Path

# original dotted module -> alternatives, in the order tried after the original
MODULE_MOVES: dict[str, tuple[str, ...]] = {
    "sglang.srt.layers.muse_spark_fp4_moe": (
        "sglang.srt.layers.muse_spark.muse_spark_fp4_moe",
    ),
    "sglang.srt.layers.muse_spark_bf16_moe": (
        "sglang.srt.layers.muse_spark.muse_spark_bf16_moe",
    ),
    "sglang.srt.layers.muse_spark_layers": (
        "sglang.srt.layers.muse_spark.muse_spark_layers",
    ),
    "sglang.srt.layers.quantization.muse_spark_nvfp4_moe": (
        "sglang.srt.layers.muse_spark.muse_spark_nvfp4_moe",
    ),
    # operators: branch kernels/ops/muse_spark_v12/<name> -> main's per-computation modules
    "sglang.kernels.ops.muse_spark_v12.norm_residual": (
        "sglang.kernels.ops.layernorm.muse_spark_norm_residual_jit",
    ),
    "sglang.kernels.ops.muse_spark_v12.fused_norm_residual": (
        "sglang.kernels.ops.layernorm.muse_spark_norm_residual",
    ),
    "sglang.kernels.ops.muse_spark_v12.all_gather_norm_residual": (
        "sglang.kernels.ops.communication.muse_spark_all_gather_norm_residual",
    ),
    "sglang.kernels.ops.muse_spark_v12.norm_fp4_quant": (
        "sglang.kernels.ops.layernorm.muse_spark_rmsnorm_nvfp4_quant",
    ),
    "sglang.kernels.ops.muse_spark_v12.norm_gate": (
        "sglang.kernels.ops.layernorm.muse_spark_norm_gate_jit",
    ),
    "sglang.kernels.ops.muse_spark_v12.fused_norm_gate": (
        "sglang.kernels.ops.layernorm.muse_spark_norm_gate",
    ),
    "sglang.kernels.ops.muse_spark_v12.fused_qk_norm": (
        "sglang.kernels.ops.attention.muse_spark_qk_norm_rope_store",
    ),
    "sglang.kernels.ops.muse_spark_v12.gemm_ag": (
        "sglang.srt.layers.communication.muse_spark_gemm_ag",
    ),
    "sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing": (
        "sglang.kernels.ops.moe.muse_spark_routing",
    ),
    "sglang.kernels.ops.muse_spark_v12.router_gemm": (
        "sglang.kernels.ops.gemm.muse_spark_router_gemm",
    ),
    "sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_trtllm": (
        "sglang.kernels.ops.communication.muse_spark_moe_finalize_all_reduce",
    ),
}

# original source path -> alternatives (relative to the directory holding the ``sglang`` package)
PATH_MOVES: dict[str, tuple[str, ...]] = {
    "sglang/srt/layers/muse_spark_fp4_moe.py": (
        "sglang/srt/layers/muse_spark/muse_spark_fp4_moe.py",
    ),
    "sglang/srt/layers/muse_spark_bf16_moe.py": (
        "sglang/srt/layers/muse_spark/muse_spark_bf16_moe.py",
    ),
    "sglang/srt/layers/muse_spark_layers.py": (
        "sglang/srt/layers/muse_spark/muse_spark_layers.py",
    ),
    "sglang/srt/layers/quantization/muse_spark_nvfp4_moe.py": (
        "sglang/srt/layers/muse_spark/muse_spark_nvfp4_moe.py",
    ),
    "sglang/kernels/jit/csrc/muse_spark_v12/all_gather_norm_residual.cuh": (
        "sglang/kernels/jit/csrc/muse_spark/all_gather_norm_residual.cuh",
    ),
    "sglang/kernels/jit/csrc/muse_spark_v12/norm_fp4_quant.cuh": (
        "sglang/kernels/jit/csrc/muse_spark/norm_fp4_quant.cuh",
    ),
    "sglang/kernels/jit/csrc/muse_spark_v12/norm_gate.cuh": (
        "sglang/kernels/jit/csrc/muse_spark/norm_gate.cuh",
    ),
    "sglang/kernels/jit/csrc/muse_spark_v12/norm_residual.cuh": (
        "sglang/kernels/jit/csrc/muse_spark/norm_residual.cuh",
    ),
    "sglang/kernels/jit/csrc/muse_spark_v12/routing.cuh": (
        "sglang/kernels/jit/csrc/muse_spark/routing.cuh",
    ),
    "sglang/kernels/ops/muse_spark_v12/norm_residual.py": (
        "sglang/kernels/ops/layernorm/muse_spark_norm_residual_jit.py",
    ),
    "sglang/kernels/ops/muse_spark_v12/fused_norm_residual.py": (
        "sglang/kernels/ops/layernorm/muse_spark_norm_residual.py",
    ),
    "sglang/kernels/ops/muse_spark_v12/all_gather_norm_residual.py": (
        "sglang/kernels/ops/communication/muse_spark_all_gather_norm_residual.py",
    ),
    "sglang/kernels/ops/muse_spark_v12/norm_fp4_quant.py": (
        "sglang/kernels/ops/layernorm/muse_spark_rmsnorm_nvfp4_quant.py",
    ),
    "sglang/kernels/ops/muse_spark_v12/norm_gate.py": (
        "sglang/kernels/ops/layernorm/muse_spark_norm_gate_jit.py",
    ),
    "sglang/kernels/ops/muse_spark_v12/fused_norm_gate.py": (
        "sglang/kernels/ops/layernorm/muse_spark_norm_gate.py",
    ),
    "sglang/kernels/ops/muse_spark_v12/fused_qk_norm.py": (
        "sglang/kernels/ops/attention/muse_spark_qk_norm_rope_store.py",
    ),
    "sglang/kernels/ops/muse_spark_v12/gemm_ag.py": (
        "sglang/srt/layers/communication/muse_spark_gemm_ag.py",
    ),
    "sglang/kernels/ops/muse_spark_v12/routing.py": (
        "sglang/kernels/ops/moe/muse_spark_routing_jit.py",
    ),
    "sglang/kernels/ops/muse_spark_v12/fused_muse_spark_routing.py": (
        "sglang/kernels/ops/moe/muse_spark_routing.py",
    ),
}


def _root() -> Path | None:
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0]).resolve().parent


def _exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@functools.lru_cache(maxsize=None)
def module_name(name: str) -> str | None:
    """Installed dotted location of the module originally named ``name`` (None when it exists nowhere)."""
    for cand in (name, *MODULE_MOVES.get(name, ())):
        if _exists(cand):
            return cand
    return None


def module(name: str):
    resolved = module_name(name)
    if resolved is None:
        raise ImportError(f"no installed location for {name}")
    return importlib.import_module(resolved)


@functools.lru_cache(maxsize=None)
def target(dotted: str) -> str:
    """``dotted`` with its module part replaced by the installed location (unchanged when nothing matches)."""
    parts = dotted.split(".")
    for k in range(len(parts), 1, -1):
        mod = ".".join(parts[:k])
        if mod in MODULE_MOVES:
            resolved = module_name(mod)
            return dotted if resolved is None else ".".join([resolved, *parts[k:]])
    return dotted


@functools.lru_cache(maxsize=None)
def relpath(path: str) -> str:
    """Installed location of the pinned source originally at ``path`` (``path`` itself when nothing else exists)."""
    root = _root()
    if root is None:
        return path
    for cand in (path, *PATH_MOVES.get(path, ())):
        if (root / cand).is_file():
            return cand
    return path


@functools.lru_cache(maxsize=None)
def layout() -> str:
    """Decided from the installed files (aliases do not change it)."""
    root = _root()
    return (
        "branch"
        if root is not None
        and (root / "sglang/srt/layers/muse_spark_fp4_moe.py").is_file()
        else "main"
    )


_COMPAT_INCLUDE = Path(__file__).resolve().parent / "include" / "compat_main"


def csrc_include_paths(csrc) -> list[str]:
    """Include directories for a plugin JIT build that includes SGLang JIT sources from ``csrc``. On main's layout this
    adds the plugin's forwarding headers, so ``muse_spark_v12/<name>.cuh`` resolves to ``muse_spark/<name>.cuh``."""
    return [str(csrc)] + ([str(_COMPAT_INCLUDE)] if layout() == "main" else [])


_aliases: dict[str, str] = {}


def install_aliases() -> dict[str, str]:
    """Make every original module name in ``MODULE_MOVES`` importable when SGLang has the module elsewhere.

    The alias is the installed module object itself (``sys.modules[original] is sys.modules[installed]``), so imports,
    hook targets and attribute patches made through the original name act on the module SGLang uses. A missing parent
    package of an original name gets an empty namespace module. Names that exist at their original location are left
    alone (no alias on the layout the plugin was first built on). Returns {original: installed}; idempotent.
    """
    import sys
    import types

    for name in MODULE_MOVES:
        if name in _aliases or _exists(name):
            continue
        resolved = module_name(name)
        if resolved is None:
            continue
        mod = importlib.import_module(resolved)
        parent, _, leaf = name.rpartition(".")
        chain = parent.split(".")
        for k in range(1, len(chain) + 1):
            pkg = ".".join(chain[:k])
            if pkg not in sys.modules and not _exists(pkg):
                ns = types.ModuleType(pkg)
                ns.__path__ = []  # namespace-like: holds only the aliases set below
                sys.modules[pkg] = ns
                if k > 1:
                    setattr(sys.modules[".".join(chain[: k - 1])], chain[k - 1], ns)
        sys.modules[name] = mod
        setattr(importlib.import_module(parent), leaf, mod)
        _aliases[name] = resolved
    return dict(_aliases)
