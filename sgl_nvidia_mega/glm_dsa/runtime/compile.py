# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team
"""Kernel compilation utilities: AOT cache wrapper and CUDA stream helper."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Hashable, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_jit_cache: Any = None  # created on first use

compile_stats: Dict[str, int] = {"compiled": 0, "cached": 0, "not_stored": 0}


def kernel_source_paths() -> Tuple[str, ...]:
    """Sources whose content keys the persistent cache directory."""
    import sgl_nvidia_mega.glm_dsa.kernels as kernels

    return tuple(
        os.path.dirname(os.path.abspath(f)) for f in (__file__, kernels.__file__)
    )


def compile_key(kernel: Any, constexprs: Dict[str, Hashable]) -> Tuple[Hashable, ...]:
    """The cache key: kernel's class (so test subclasses cache separately)."""
    cls = type(kernel)
    return (f"{cls.__module__}.{cls.__qualname__}", tuple(sorted(constexprs.items())))


def compile_cached(
    kernel: Any, constexprs: Dict[str, Hashable], compile_fn: Callable[[], Any]
) -> Any:
    """The compiled ``kernel`` for these constants, compiling on a cache miss."""
    global _jit_cache
    if _jit_cache is None:
        from sglang.kernels.jit.cute_aot_cache import get_jit_cache

        _jit_cache = get_jit_cache(
            "glm_dsa", source_paths=kernel_source_paths(), enable_tvm_ffi=False
        )
    key = compile_key(kernel, constexprs)
    if key in _jit_cache:
        compile_stats["cached"] += 1
        return _jit_cache[key]
    compiled = compile_fn()
    compile_stats["compiled"] += 1
    try:
        _jit_cache[key] = compiled
    except Exception as e:
        compile_stats["not_stored"] += 1
        detail = " ".join(str(e).split())[:300]
        logger.warning(
            "GLM DSA fused decode: the CuTe AOT cache could not store %s %s (%s: %s); "
            "it is compiled again in every new process",
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
