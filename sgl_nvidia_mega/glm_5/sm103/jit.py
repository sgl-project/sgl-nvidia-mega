# SPDX-License-Identifier: Apache-2.0
"""Lazy NVCC cubins with typed TVM-FFI host bindings."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from sgl_nvidia_mega.common.build import compile_ffi, compile_nvcc, nvcc_path
from sgl_nvidia_mega.glm_5.plan import Variant

ROOT = Path(__file__).resolve().parent
INCLUDE = ROOT / "include"
_INCLUDE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]', re.MULTILINE)
_FLAGS = (
    "-std=c++20",
    "-O3",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-lineinfo",
)


@dataclass(frozen=True)
class KernelMetadata:
    """Resource sizes exported by the compiled C++ Config, never recomputed here."""

    variant: Variant
    workspace_bytes: int
    epoch_count: int
    peer_bytes: int
    workspace_fill: int
    required_sms: int
    native_cvt: bool
    norm_dtype: str = "float32"

    @classmethod
    def from_module(cls, variant, module, norm_dtype="float32"):
        values = tuple(map(int, module.metadata()))
        if len(values) != 9 or values[:3] != (1, variant.rows, variant.tp_size):
            raise RuntimeError(f"incompatible GLM5 compiled metadata: {values}")
        _, _, _, workspace, epochs, peer, fill, sms, native = values
        if (
            min(workspace, epochs, peer, sms) <= 0
            or fill not in (0, 255)
            or native not in (0, 1)
        ):
            raise RuntimeError(f"invalid GLM5 compiled resources: {values}")
        return cls(
            variant,
            workspace,
            epochs,
            peer,
            fill,
            sms,
            bool(native),
            norm_dtype,
        )


class KernelSourcesMissing(FileNotFoundError):
    """The harness is installed, but its core CUDA input is not installed yet."""

    def __init__(self, variant: Variant, missing: tuple[str, ...]):
        self.variant, self.missing = variant, missing
        super().__init__(
            f"GLM5 {variant.op} TP{variant.tp_size}/MTP{variant.mtp_label} core sources "
            f"are not installed: {', '.join(missing)}"
        )


def source_files(variant: Variant) -> tuple[Path, Path]:
    return (
        ROOT / "src" / variant.source_name,
        INCLUDE / variant.op / "config.cuh",
    )


def source_bundle(variant: Variant, *, host=False) -> dict[str, bytes]:
    """Read the local include closure. Never import torch or initialize CUDA."""
    pending = (
        [INCLUDE / variant.op / "host.cuh"] if host else list(source_files(variant))
    )
    contents: dict[str, bytes] = {}
    missing = set()
    while pending:
        path = pending.pop().resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError(f"GLM5 source escapes the package directory: {path}")
        relative = path.relative_to(ROOT).as_posix()
        if relative in contents or relative in missing:
            continue
        if not path.is_file():
            missing.add(relative)
            continue
        contents[relative] = path.read_bytes()
        for quote, name in _INCLUDE.findall(contents[relative].decode()):
            if name.startswith(("common/", "comm/", "mla/", "moe/")):
                pending.append(INCLUDE / name)
            elif quote == '"':
                pending.append(path.parent / name)
    if missing:
        raise KernelSourcesMissing(variant, tuple(sorted(missing)))
    return contents


def source_reasons(variant: Variant) -> list[str]:
    try:
        source_bundle(variant)
    except KernelSourcesMissing as error:
        return [str(error)]
    return []


@lru_cache(maxsize=1)
def cuda_version() -> tuple[int, int, int]:
    nvcc = nvcc_path()
    result = subprocess.run(
        [nvcc, "--version"], check=True, capture_output=True, text=True, timeout=10
    )
    match = re.search(r"\bV(\d+)\.(\d+)\.(\d+)\b", result.stdout + result.stderr)
    if match is None:
        raise RuntimeError(f"cannot parse the NVCC version from {nvcc}")
    version = tuple(map(int, match.groups()))
    if version < (13, 0, 0):
        raise RuntimeError(f"GLM5 needs NVCC >= 13.0, got {version}")
    return version


def _cutlass_include() -> Path:
    override = os.environ.get("CUTLASS_INCLUDE_DIR")
    candidates = [Path(override).expanduser()] if override else []
    if not override:
        for name, suffix in (
            ("cutlass", "include"),
            ("cutlass_library", "source/include"),
            ("flashinfer", "data/cutlass/include"),
        ):
            spec = importlib.util.find_spec(name)
            for root in (spec.submodule_search_locations or ()) if spec else ():
                candidates.append(Path(root) / suffix)
    probes = (
        "cutlass/float8.h",
        "cute/arch/copy_sm100.hpp",
        "cute/arch/copy_sm100_tma.hpp",
    )
    for directory in candidates:
        if all((directory / probe).is_file() for probe in probes):
            return directory.resolve()
    raise RuntimeError(
        "set CUTLASS_INCLUDE_DIR to a CUTLASS include directory with SM100 headers"
    )


def _flags(variant: Variant, native: bool, norm_dtype="float32") -> list[str]:
    if variant.op == "mla":
        extra = [
            f"-DGLM5_MEGAMLA_TOKENS={variant.rows}",
            f"-DGLM5_MEGAMLA_TP_SIZE={variant.tp_size}",
            "-DGLM5_MEGAMLA_FP8_KV_CACHE=1",
            f"-DMEGAMLA_USE_NATIVE_BF16X2_E4M3X2_CVT={int(native)}",
        ]
    else:
        extra = [
            f"-DGLM5_MEGAMOE_SEQ_LEN={variant.rows}",
            f"-DGLM5_MEGAMOE_TP_SIZE={variant.tp_size}",
            f"-DMEGAMOE_USE_NATIVE_E4M3X2_TO_BF16X2={int(native)}",
        ]
    return [
        *_FLAGS,
        f"-DGLM5_KERNEL_NAME={variant.symbol}",
        f"-DGLM5_NORM_BF16={int(norm_dtype == 'bfloat16')}",
        *extra,
    ]


def load(variant: Variant, *, norm_dtype="float32"):
    """Precompile outside capture; missing source is distinct from a build error."""
    if norm_dtype not in ("float32", "bfloat16"):
        raise ValueError("GLM5 norm dtype must be float32 or bfloat16")
    contents = source_bundle(variant)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GLM5 kernels require a CUDA GPU")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("preload GLM5 kernels before CUDA-graph capture")
    if torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError("the initial GLM5 harness targets SM103")
    version, cutlass = cuda_version(), _cutlass_include()
    flags = _flags(variant, version >= (13, 2, 0), norm_dtype)
    digest = hashlib.sha256()
    for path, body in sorted({**contents, **source_bundle(variant, host=True)}.items()):
        digest.update(path.encode() + b"\0" + body + b"\0")

    digest.update(
        json.dumps(
            [
                version,
                flags,
                str(cutlass),
                "10.3a",
                "ffi_bound_native_fp8_v1",
                torch.__version__,
                _cutlass_fingerprint(str(cutlass)),
            ]
        ).encode()
    )
    module = _compile(
        variant,
        digest.hexdigest(),
        tuple(flags),
        str(cutlass),
        torch.cuda.current_device(),
    )
    metadata = KernelMetadata.from_module(variant, module, norm_dtype)
    if (
        torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        < metadata.required_sms
    ):
        raise RuntimeError(f"GLM5 persistent kernel needs {metadata.required_sms} SMs")
    return module, metadata


@lru_cache(maxsize=None)
def _cutlass_fingerprint(directory: str) -> str:
    digest = hashlib.sha256()
    root = Path(directory)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in (".h", ".hpp", ".cuh"):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


@lru_cache(maxsize=None)
def _compile(variant, digest, flags, cutlass, device):
    from sgl_nvidia_mega.glm_5.sm103.binding import CompiledModule

    source = "\n".join(f'#include "{path}"' for path in source_files(variant))
    cubin = compile_nvcc(
        source,
        f"glm5_{variant.op}",
        "sm_103a",
        flags,
        (str(INCLUDE), cutlass),
        digest,
    )
    cuda_include = Path(nvcc_path()).resolve().parent.parent / "include"
    host_flags = tuple(flag for flag in flags if flag.startswith("-D"))
    native = compile_ffi(
        f"#include <{variant.op}/host.cuh>\n"
        f"TVM_FFI_DLL_EXPORT_TYPED_FUNC(load, glm5::{variant.op}::host::load);\n",
        f"glm5_{variant.op}_ffi",
        host_flags,
        (str(INCLUDE), str(cuda_include)),
        digest,
        link_flags=("-l:libcuda.so.1",),
    )
    import torch

    with torch.cuda.device(device):
        torch.empty(1, dtype=torch.uint8, device=device)
        return CompiledModule(native, cubin, variant, device)
