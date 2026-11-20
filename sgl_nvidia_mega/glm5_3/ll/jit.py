# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CUDA JIT modules of the low-latency units: SGLang's ``load_jit`` (tvm-ffi, cached by
SGLang's JIT cache) on this package's ``csrc/**/*.cuh`` sources.

``load`` compiles at family build only: it refuses to run under CUDA-graph capture,
and a failure raises (the registry gathers it and every rank raises). Each load is
logged with its time, and ``stats`` counts them for the build summary.

The kernels include SGLang's ``sgl_kernel/*`` headers (``SGLANG_HEADERS``), pinned
against the installed SGLang by ``test/unit/test_glm5_3_ll_jit.py``.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple, Union

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("load_jit",),
    "kernels/jit/utils/compile/loader.py": (
        "load_jit",
        "load_jit(cuda_files)",
        "load_jit(cuda_wrappers)",
        "load_jit(extra_cuda_cflags)",
        "load_jit(extra_include_paths)",
    ),
}
# SGLang's C++ headers the .cuh sources include, and the names they use from them;
# the files are under python/sglang/kernels/jit/include (test_glm5_3_ll_jit.py)
SGLANG_HEADERS = {
    "sgl_kernel/tensor.h": ("TensorMatcher",),
    "sgl_kernel/utils.h": (),
    "sgl_kernel/utils.cuh": (),
    "sgl_kernel/runtime.cuh": (),
    "sgl_kernel/tile.cuh": (),
    "sgl_kernel/vec.cuh": (),
    "sgl_kernel/impl/norm.cuh": (),
    "sgl_kernel/warp.cuh": (),
    "sgl_kernel/ffi.h": (),
}

logger = logging.getLogger(__name__)

CSRC = Path(__file__).resolve().parent / "csrc"
# the JIT module family name of every unit kernel (SGLang's cache directory key)
MODULE = "sgl_nvidia_mega_glm5_3"
# Test hook (fail-stop): a comma-separated list of "name" or "name@rank" whose load
# compiles with an invalid nvcc flag, so the build really fails on those ranks.
FORCE_ERROR_ENV = "SGLANG_NVIDIA_MEGAKERNEL_GLM_FORCE_BUILD_ERROR"
FORCED_ERROR_FLAG = "--sgl-nvidia-mega-forced-build-error"

stats: Dict[str, float] = {"loaded": 0, "seconds": 0.0}


def load(
    name: str,
    cuh: Union[str, Path],
    wrappers: Sequence[Tuple[str, str]],
    *,
    extra_cuda_cflags: Iterable[str] = (),
    marker: Iterable = (),
):
    """The tvm-ffi module of ``cuh`` (absolute, or relative to ``csrc/``) exporting
    ``wrappers`` ((export name, kernel name) pairs); ``name`` + ``marker`` key the
    build (they must differ between kernel instances)."""
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(f"ll.jit: compiling {name} inside CUDA-graph capture")
    path = Path(cuh)
    if not path.is_absolute():
        path = CSRC / path
    if not path.is_file():
        raise FileNotFoundError(f"ll.jit: {name}: no source {path}")
    flags = list(extra_cuda_cflags)
    if forced_error(name):
        flags.append(FORCED_ERROR_FLAG)
    marker = tuple(str(m) for m in marker)
    from sglang.kernels.jit.utils import load_jit

    start = time.perf_counter()
    try:
        module = load_jit(
            MODULE,
            name,
            *marker,
            cuda_files=[str(path)],
            cuda_wrappers=list(wrappers),
            extra_cuda_cflags=flags,
            extra_include_paths=[str(CSRC)],
        )
    except Exception as e:
        raise RuntimeError(
            f"ll.jit: building {name}{list(marker) if marker else ''} from "
            f"{path.relative_to(CSRC) if path.is_relative_to(CSRC) else path} failed: "
            f"{type(e).__name__}: {' '.join(str(e).split())[:2000]}"
        ) from e
    seconds = time.perf_counter() - start
    stats["loaded"] += 1
    stats["seconds"] += seconds
    logger.info(
        "NVIDIA megakernel: GLM-5.3 / GLM-5.2: ll: JIT module %s%s ready in %.1f s",
        name,
        list(marker) if marker else "",
        seconds,
    )
    return module


def forced_error(name: str) -> bool:
    """Whether the test hook ``FORCE_ERROR_ENV`` names ``name`` on this rank."""
    raw = os.environ.get(FORCE_ERROR_ENV, "")
    if not raw:
        return False
    rank = _rank()
    for entry in (e.strip() for e in raw.split(",")):
        target, _, at = entry.partition("@")
        if target == name and (not at or (rank is not None and int(at) == rank)):
            return True
    return False


def _rank():
    import torch.distributed as dist

    return dist.get_rank() if dist.is_available() and dist.is_initialized() else None
