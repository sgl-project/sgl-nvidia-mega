# SPDX-License-Identifier: Apache-2.0
"""Shared cache and model-independent NVCC / optional TVM-FFI compilation."""

import fcntl
import hashlib
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path


def cache_dir() -> Path:
    root = os.environ.get("SGLANG_CACHE_DIR") or "~/.cache/sglang"
    return Path(os.path.expanduser(root)) / "sgl_nvidia_mega"


def nvcc_path() -> str:
    root = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    compiler = str(Path(root) / "bin/nvcc") if root else shutil.which("nvcc")
    if not compiler or not Path(compiler).expanduser().is_file():
        raise RuntimeError("NVCC is required to compile CUDA C++ kernels")
    return str(Path(compiler).expanduser().resolve())


def _compiler_version(command: list[str]) -> str:
    result = subprocess.run(
        [*command, "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout + result.stderr


def compile_nvcc(
    source: str,
    name: str,
    arch: str,
    options: tuple[str, ...],
    include_dirs: tuple[str, ...],
    dependency_key: str,
) -> bytes:
    """Compile a cubin; the caller supplies the target, flags and header digest."""
    compiler = nvcc_path()
    key = hashlib.sha256(
        json.dumps(
            [
                compiler,
                _compiler_version([compiler]),
                source,
                arch,
                options,
                include_dirs,
                dependency_key,
                os.environ.get("NVCC_PREPEND_FLAGS", ""),
                os.environ.get("NVCC_APPEND_FLAGS", ""),
            ]
        ).encode()
    ).hexdigest()
    directory = (cache_dir() / f"{name}-{key}").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "compile.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cached = directory / "kernel.cubin"
        if cached.is_file():
            return cached.read_bytes()
        unit = directory / "kernel.cu"
        output = directory / "build.cubin"
        unit.write_text(source)
        command = [
            compiler,
            "--cubin",
            f"--gpu-architecture={arch}",
            *options,
            *(f"-I{path}" for path in include_dirs),
            str(unit),
            "-o",
            str(output),
        ]
        result = subprocess.run(command, cwd=directory, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(
                f"NVCC failed to compile {name}:\n{result.stdout}{result.stderr}"
            )
        cubin = output.read_bytes()
        if not cubin.startswith(b"\x7fELF"):
            raise RuntimeError(f"NVCC produced an invalid cubin for {name}")
        cached.write_bytes(cubin)
        return cubin


def compile_ffi(
    source: str,
    name: str,
    options: tuple[str, ...],
    include_dirs: tuple[str, ...],
    dependency_key: str,
    *,
    link_flags: tuple[str, ...] = (),
):
    """JIT a typed host binding; callers supply any backend-specific link flags."""
    import tvm_ffi
    from tvm_ffi import libinfo

    compiler = shlex.split(os.environ.get("CXX") or "c++")
    executable = shutil.which(compiler[0]) if compiler else None
    if not executable:
        raise RuntimeError("a C++ compiler is required for TVM-FFI bindings")
    compiler[0] = str(Path(executable).resolve())
    library = Path(libinfo.find_libtvm_ffi())
    includes = (*libinfo.include_paths(), *include_dirs)
    flags = ("-std=c++20", "-O3", "-shared", "-fPIC", "-save-temps=obj", *options)
    key = hashlib.sha256(
        json.dumps(
            [
                source,
                flags,
                includes,
                dependency_key,
                compiler,
                _compiler_version(compiler),
                tvm_ffi.__version__,
                str(library),
                link_flags,
            ]
        ).encode()
    ).hexdigest()
    directory = (cache_dir() / f"{name}-{key}").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "compile.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        output = directory / "binding.so"
        complete = directory / "complete"
        if not (
            output.is_file() and complete.is_file() and complete.read_text() == key
        ):
            unit = directory / "binding.cpp"
            unit.write_text(source)
            command = [
                *compiler,
                *flags,
                *(f"-I{path}" for path in includes),
                str(unit),
                str(library),
                f"-Wl,-rpath,{library.parent}",
                *link_flags,
                "-o",
                str(output),
            ]
            result = subprocess.run(
                command, cwd=directory, capture_output=True, text=True
            )
            if result.returncode:
                raise RuntimeError(
                    f"C++ failed to compile {name}:\n{result.stdout}{result.stderr}"
                )
            complete.write_text(key)
    return tvm_ffi.load_module(output)
