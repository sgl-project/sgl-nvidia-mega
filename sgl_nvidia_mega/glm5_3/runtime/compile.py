# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Kernel compilation through SGLang's persistent CuTe DSL cache.

The compile key carries the kernel class's qualified name, so a test subclass never
shares a cache entry with a served kernel. Everything compiles at build; a step only
launches.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Hashable, Tuple

import torch

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/jit/cute_aot_cache.py": (
        "get_jit_cache",
        "get_jit_cache(source_paths)",
        "get_jit_cache(enable_tvm_ffi)",
    ),
}

# the persistent cache's name (its directory under SGLang's CuTe AOT cache)
CACHE_NAME = "glm5_3"

_jit_cache: Any = None  # created on first use
# Compiles, cache hits and units the persistent cache could not store (startup log).
compile_stats: Dict[str, int] = {"compiled": 0, "cached": 0, "not_stored": 0}


def kernel_source_paths() -> Tuple[str, ...]:
    """Sources whose content keys the persistent cache directory: this model's package
    (its step code and its kernels)."""
    return (os.path.dirname(os.path.dirname(os.path.abspath(__file__))),)


def compile_key(kernel: Any, constexprs: Dict[str, Hashable]) -> Tuple[Hashable, ...]:
    """The cache key: ``kernel``'s class (so a test subclass caches separately)
    and the compile-time constants."""
    cls = type(kernel)
    return (f"{cls.__module__}.{cls.__qualname__}", tuple(sorted(constexprs.items())))


def compile_cached(
    kernel: Any, constexprs: Dict[str, Hashable], compile_fn: Callable[[], Any]
) -> Any:
    """The compiled ``kernel`` for these constants, compiling on a miss (the
    units are compiled without tvm-ffi). A unit the persistent cache cannot store
    warns once and serves from memory."""
    global _jit_cache
    if _jit_cache is None:
        from sglang.kernels.jit.cute_aot_cache import get_jit_cache

        _jit_cache = get_jit_cache(
            CACHE_NAME, source_paths=kernel_source_paths(), enable_tvm_ffi=False
        )
    key = compile_key(kernel, constexprs)
    if key in _jit_cache:
        compile_stats["cached"] += 1
        return _jit_cache[key]
    compiled = compile_fn()
    compile_stats["compiled"] += 1
    try:
        _jit_cache[key] = compiled
    except Exception as e:  # the export to disk; the in-memory entry is set first
        compile_stats["not_stored"] += 1
        detail = " ".join(str(e).split())[:300]
        logger.warning(
            "NVIDIA megakernel: GLM-5.3 / GLM-5.2: the CuTe AOT cache could not "
            "store %s %s (%s: %s); it is compiled again in every new process",
            key[0],
            dict(key[1]),
            type(e).__name__,
            detail,
        )
    return compiled


def cu_stream(device=None):
    """The current torch stream of ``device`` as a kernel launch argument."""
    import cuda.bindings.driver as cuda

    return cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
