# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""nvcc builds of the MiniMax-M3.1 kernels to cubins, cached on disk by the digest of
every source file, the options and the compiler. A translation unit's own host code
compiles and is discarded: the plugin loads the cubin and launches it from Python.
The ranks of a node build one cubin once: the others wait on its lock file."""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import subprocess
from functools import cache
from pathlib import Path
from typing import Optional

import torch

from sgl_nvidia_mega.common.jit import cache_dir, check, driver

# MegaAttention dequantizes NVFP4 with the QMUL4 instruction, which needs nvcc 13.4 or
# newer (the toolkit SGLang's own MiniMax-M3.1 path on Blackwell requires too)
QMUL4_VERSION = (13, 4)


def find_nvcc() -> Optional[Path]:
    """The toolkit SGLang's MiniMax-M3.1 path uses: CUDA_HOME, then CUDACXX, then the
    nvcc on PATH or /usr/local/cuda."""
    home = os.environ.get("CUDA_HOME")
    if home and (Path(home) / "bin" / "nvcc").is_file():
        return Path(home) / "bin" / "nvcc"
    cxx = os.environ.get("CUDACXX")
    if cxx and Path(cxx).is_file():
        return Path(cxx)
    found = shutil.which("nvcc")
    if found:
        return Path(found)
    default = Path("/usr/local/cuda/bin/nvcc")
    return default if default.is_file() else None


@cache
def version_text(nvcc: Path) -> str:
    return subprocess.run(
        [str(nvcc), "--version"], capture_output=True, text=True, check=True
    ).stdout


def version(nvcc: Path) -> tuple[int, int]:
    """(major, minor) from the "release X.Y" line of nvcc --version."""
    for word in version_text(nvcc).replace(",", " ").split():
        if word.count(".") == 1 and word.replace(".", "").isdigit():
            major, minor = word.split(".")
            return int(major), int(minor)
    raise RuntimeError(f"unrecognized nvcc version: {version_text(nvcc)!r}")


def why_not(min_version: Optional[tuple[int, int]] = None) -> str:
    """Why the kernels cannot build here ("" when they can)."""
    nvcc = find_nvcc()
    if nvcc is None:
        return "no CUDA toolkit (nvcc) found in CUDA_HOME, CUDACXX, PATH or /usr/local/cuda"
    try:
        found = version(nvcc)
    except (OSError, subprocess.CalledProcessError, RuntimeError) as e:
        return f"{nvcc}: {e}"
    if min_version is not None and found < min_version:
        return (
            f"{nvcc} is CUDA {found[0]}.{found[1]}; the QMUL4 NVFP4 conversion needs "
            f"{min_version[0]}.{min_version[1]} or newer"
        )
    return ""


@cache
def _root_digest(root: Path) -> bytes:
    """Every file under an include root, hashed once a process."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.digest()


def _digest(source: str, roots: tuple[Path, ...], options: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for value in (source, *options):
        digest.update(value.encode() + b"\0")
    for root in roots:
        digest.update(_root_digest(Path(root).resolve()))
    return digest.hexdigest()


def compile_cubin(
    name: str,
    source: str,
    roots: tuple[Path, ...],
    arch: str,
    options: tuple[str, ...],
    min_version: Optional[tuple[int, int]] = None,
) -> bytes:
    """Compile the translation unit ``source`` (include roots ``roots``) to a cubin for
    ``arch`` with nvcc, cached by the digest of the source, every file under the roots,
    the options and nvcc's version. Refuses to run under CUDA-graph capture."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(f"compiling {name} inside CUDA-graph capture")
    reason = why_not(min_version)
    if reason:
        raise RuntimeError(reason)
    nvcc = find_nvcc()
    arguments = (
        "-cubin",
        f"-gencode=arch=compute_{arch[3:]},code={arch}",
        *options,
        *(f"-I{root}" for root in roots),
    )
    key = _digest(source, roots, (*arguments, version_text(nvcc)))
    cached = cache_dir() / f"{name}-{key[:24]}.cubin"
    if cached.is_file():
        return cached.read_bytes()
    cached.parent.mkdir(parents=True, exist_ok=True)
    with open(cached.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not cached.is_file():
            _compile(nvcc, arguments, source, name, cached)
    return cached.read_bytes()


def _compile(
    nvcc: Path, arguments: tuple, source: str, name: str, cached: Path
) -> None:
    unit = cached.parent / f"{cached.stem}.{os.getpid()}.cu"
    output = cached.parent / f"{cached.stem}.{os.getpid()}.tmp"
    unit.write_text(source)
    try:
        result = subprocess.run(
            [str(nvcc), *arguments, str(unit), "-o", str(output)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"nvcc failed to compile {name}:\n{result.stderr}")
        output.replace(cached)
    finally:
        unit.unlink(missing_ok=True)
        output.unlink(missing_ok=True)


def kernel_names(cubin: bytes) -> list[str]:
    """The names of the kernels in ``cubin``."""
    cu = driver()
    (module,) = check(cu.cuModuleLoadData(cubin), "cuModuleLoadData")
    try:
        (count,) = check(
            cu.cuModuleGetFunctionCount(module), "cuModuleGetFunctionCount"
        )
        (functions,) = check(
            cu.cuModuleEnumerateFunctions(count, module), "cuModuleEnumerateFunctions"
        )
        return [
            check(cu.cuFuncGetName(f), "cuFuncGetName")[0].decode() for f in functions
        ]
    finally:
        cu.cuModuleUnload(module)


def device_arch(device: torch.device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}a"
