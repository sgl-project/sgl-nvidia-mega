# SPDX-License-Identifier: Apache-2.0
"""Use the framework compiler with this package's kernel sources."""

from __future__ import annotations

import os
from pathlib import Path

from sglang.kernels.jit.utils import cache_once as cache_once
from sglang.kernels.jit.utils import is_arch_support_pdl as is_arch_support_pdl
from sglang.kernels.jit.utils import (
    load_jit,
)
from sglang.kernels.jit.utils import make_cpp_args as make_cpp_args

SGLANG_READS = {
    "kernels/jit/utils/__init__.py": (
        "cache_once",
        "is_arch_support_pdl",
        "load_jit",
        "make_cpp_args",
    ),
    "kernels/jit/utils/compile/paths.py": ("KERNEL_PATH",),
}
_framework_load_jit = load_jit

KERNEL_PATH = Path(__file__).resolve().parent / "kernels" / "jit"


def _sglang_csrc() -> Path:
    from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH

    return KERNEL_PATH / "csrc"


def load_jit(name, *args, cuda_files, extra_include_paths=None, **kwargs):
    """Absolute source paths prevent silently compiling a stock namesake. This
    package's csrc comes first on the include path, then SGLang's."""
    sources = []
    for source in cuda_files:
        path = Path(source)
        if not path.is_absolute():
            path = KERNEL_PATH / "csrc" / path
        if not path.is_file():
            raise FileNotFoundError(path)
        sources.append(str(path))
    includes = [
        str(KERNEL_PATH / "csrc"),
        str(_sglang_csrc()),
        *(extra_include_paths or ()),
    ]
    return _framework_load_jit(
        "nvidia_qwen3_8_" + name,
        *args,
        cuda_files=sources,
        extra_include_paths=includes,
        **kwargs,
    )


def source_cache(variable):
    """Deployment supplies an explicit writable node-local source directory."""
    value = os.environ.get(variable)
    if not value or not Path(value).is_absolute():
        raise ValueError(f"{variable} must name an absolute node-local directory")
    return Path(value).resolve()
