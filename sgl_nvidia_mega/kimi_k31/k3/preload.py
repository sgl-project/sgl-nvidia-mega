# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Load every CuTe DSL kernel of the process into the current context (``K31MK_PRELOAD=1``).

Under lazy module loading (CUDA's default since 12.2) a kernel loads at its first launch, and
that load can wait for the kernels already running on the GPU (CUDA PG 4.8.5.1). A spinning MK
consumer whose producer is first launched after it would then wait for a load that waits for
the consumer. Every spin wait of the MK step sits in a CuTe kernel and waits on CuTe kernels,
so loading those at prepare time removes the hazard without CUDA_MODULE_LOADING=EAGER, which
loads every module of every library in the process. Kernels of TVM-FFI compiled functions load
through their own entry point and are counted, not loaded.
"""

from __future__ import annotations

import gc
import logging
import os
import time

import cuda.bindings.driver as cuda

ENABLED = os.environ.get("K31MK_PRELOAD", "0") == "1"
logger = logging.getLogger(__name__)
_LOADED = cuda.CUfunctionLoadingState.CU_FUNCTION_LOADING_STATE_LOADED


def _ok(result, what):
    status, *values = result
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"K31MK_PRELOAD: {what} failed: {status}")
    return values[0] if len(values) == 1 else tuple(values)


def compiled_functions() -> list:
    from cutlass.cutlass_dsl.cuda_jit_executor import CudaDialectJitCompiledFunction

    return [o for o in gc.get_objects() if isinstance(o, CudaDialectJitCompiledFunction)]


def kernel_functions(compiled, *, materialize: bool):
    """CUfunctions (current context) of ``compiled``'s kernels; None when its library is not
    reachable (TVM-FFI entry point, or not yet materialized and ``materialize`` is False)."""
    if not compiled.has_gpu_module:
        return []
    if materialize:
        compiled.to()  # loads the CUDA library (context-free); its kernels stay lazy
    module = compiled.jit_module
    if module is None or module.is_unloaded():
        return None
    functions = []
    for library in module.cuda_library:
        count = _ok(cuda.cuLibraryGetKernelCount(library), "cuLibraryGetKernelCount")
        for kernel in _ok(cuda.cuLibraryEnumerateKernels(count, library), "cuLibraryEnumerateKernels"):
            functions.append(_ok(cuda.cuKernelGetFunction(kernel), "cuKernelGetFunction"))
    return functions


def is_loaded(function) -> bool:
    return _ok(cuda.cuFuncIsLoaded(function), "cuFuncIsLoaded") == _LOADED


def preload() -> dict:
    """Load every kernel of every CuTe compiled function in the process; raise if one stays
    unloaded. Returns counts and seconds."""
    start = time.perf_counter()
    stats = {"compiled": 0, "tvm_ffi_or_unreachable": 0, "kernels": 0, "loaded_now": 0}
    for compiled in compiled_functions():
        stats["compiled"] += 1
        functions = kernel_functions(compiled, materialize=True)
        if functions is None:
            stats["tvm_ffi_or_unreachable"] += 1
            continue
        for function in functions:
            stats["kernels"] += 1
            if not is_loaded(function):
                _ok(cuda.cuFuncLoad(function), "cuFuncLoad")
                stats["loaded_now"] += 1
            if not is_loaded(function):
                raise RuntimeError("K31MK_PRELOAD: a CuTe kernel is still unloaded after cuFuncLoad")
    stats["seconds"] = round(time.perf_counter() - start, 3)
    return stats


def audit() -> dict:
    """Count CuTe kernels that are not loaded, without loading anything (a compiled function
    whose library is not materialized counts once, as ``unmaterialized``)."""
    stats = {"compiled": 0, "unmaterialized": 0, "kernels": 0, "unloaded": 0}
    for compiled in compiled_functions():
        stats["compiled"] += 1
        functions = kernel_functions(compiled, materialize=False)
        if functions is None:
            stats["unmaterialized"] += 1
            continue
        stats["kernels"] += len(functions)
        stats["unloaded"] += sum(not is_loaded(f) for f in functions)
    return stats


def at_prepare() -> None:
    """MK prepare hook (after the pools are bound, before graph warmup)."""
    if ENABLED:
        logger.info("K31MK_PRELOAD: CuTe kernels preloaded at MK prepare: %s", preload())
