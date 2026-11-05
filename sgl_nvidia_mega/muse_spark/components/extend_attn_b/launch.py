# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Build and launch the own extend-attention kernels without an engine JIT: NVRTC compiles the device code of
csrc/prefill_attn.cuh (its __CUDACC_RTC__ branch has no host headers), the CUDA driver loads the cubin and launches it.

Imports torch and cuda.bindings only. Cubins are cached on disk by the digest of source, options and NVRTC version.
Every build refuses under CUDA-graph capture (prepare() builds everything before serving).

Kernel option (``MUSE_VR_XATTN_B_PFA_OPT``, default 0 = the original kernel): the attention kernel's kOpt template bits
(csrc/prefill_attn.cuh): 1 = kOptDirectS, 2 = kOptHintWait and 8 = kOptWarpS (exclusive with 1) change only how the
kernel's warps wait for each other; 4 = kOptTcSum32 takes the softmax row sum on the tensor core (the sum of the e4m3 P
values, a different normalization sum, judged against a float64 reference). Any other value refuses the build loudly.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import os
from pathlib import Path

import torch

SOURCE = Path(__file__).resolve().parent.parent / "csrc" / "prefill_attn.cuh"
THREADS = 512
MERGE_THREADS = 256
HEAD_DIM, PAGE = 64, 128
# shared memory per block (bytes): the kernel's own layout constant kSmBytes; checked against the cubin at load
SMEM_BYTES = 2 * 8 * PAGE * 64 + 2 * 128 * 64 + 1024 + 256 + 1024
KERNEL = "muse_vr::pfa::prefill_attn_kernel<true, false, 0>"  # PDL entry wait on, no trace, fp32 exp only
MERGE = "muse_vr::pfa::prefill_attn_merge"
OPT_ENV = "MUSE_VR_XATTN_B_PFA_OPT"
OPTS = tuple(v for v in range(16) if not (v & 1 and v & 8))


def opt_from_env() -> int:
    raw = os.environ.get(OPT_ENV, "0").strip() or "0"
    try:
        opt = int(raw)
    except ValueError:
        opt = -1
    if opt not in OPTS:
        raise ValueError(f"{OPT_ENV}={raw!r}: expected one of {OPTS}")
    return opt


def kernel_expr(opt: int) -> str:
    """The attention kernel's name expression for kernel option ``opt`` (0 = KERNEL, the original instantiation)."""
    if opt not in OPTS:
        raise ValueError(f"kernel option {opt!r}: expected one of {OPTS}")
    return (
        KERNEL
        if opt == 0
        else f"muse_vr::pfa::prefill_attn_kernel<true, false, 0, {opt}>"
    )


# The engine JIT's device flags for the same source (-std=c++20, nvcc's float defaults); stated explicitly so the two builds
# make the same rounding choices (division, flush-to-zero, contraction).
OPTIONS = (
    "-std=c++20",
    "-default-device",
    "--fmad=true",
    "--prec-div=true",
    "--prec-sqrt=true",
    "--ftz=false",
)


def _drv():
    from cuda.bindings import driver

    return driver


def _check(result, what):
    if int(result[0]) != 0:
        raise RuntimeError(f"{what} failed: {result[0]!r}")
    return result[1:] if len(result) > 2 else (result[1] if len(result) == 2 else None)


def _include_dirs():
    found = []
    for root in (
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_PATH"),
        "/usr/local/cuda",
    ):
        if (
            root
            and (Path(root) / "include" / "cuda_bf16.h").is_file()
            and str(Path(root) / "include") not in found
        ):
            found.append(str(Path(root) / "include"))
    return found


def _cache_dir() -> Path:
    root = os.environ.get("SGLANG_CACHE_DIR") or "~/.cache/sglang"
    return Path(os.path.expanduser(root)) / "muse_vr_extend_attn_b"


def compile_cubin(arch: str, opt: int = 0) -> tuple[bytes, dict]:
    """NVRTC build of the attention kernel (kernel option ``opt``) and the merge kernel for ``arch``; returns (cubin,
    lowered names keyed by the name expressions)."""
    from cuda.bindings import nvrtc

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("extend_attn_b: build inside CUDA-graph capture")
    kexpr = kernel_expr(opt)
    source = SOURCE.read_text()
    options = [
        f"--gpu-architecture={arch}",
        *OPTIONS,
        *[f"-I{d}" for d in _include_dirs()],
    ]
    (major, minor) = _check(nvrtc.nvrtcVersion(), "nvrtcVersion")
    key = hashlib.sha256(
        "\0".join([source, *options, kexpr, MERGE, f"{major}.{minor}"]).encode()
    ).hexdigest()[:24]
    cubin_path = _cache_dir() / f"pfa-{key}.cubin"
    names_path = cubin_path.with_suffix(".names")
    if cubin_path.is_file() and names_path.is_file():
        k, m = names_path.read_text().split("\n")[:2]
        return cubin_path.read_bytes(), {kexpr: k, MERGE: m}
    prog = _check(
        nvrtc.nvrtcCreateProgram(source.encode(), b"prefill_attn.cu", 0, [], []),
        "nvrtcCreateProgram",
    )
    try:
        for expr in (kexpr, MERGE):
            _check(
                nvrtc.nvrtcAddNameExpression(prog, expr.encode()),
                "nvrtcAddNameExpression",
            )
        enc = [o.encode() for o in options]
        (status,) = nvrtc.nvrtcCompileProgram(prog, len(enc), enc)
        if int(status) != 0:
            size = _check(nvrtc.nvrtcGetProgramLogSize(prog), "log size")
            log = b" " * size
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError(
                "extend_attn_b: NVRTC failed: " + log.decode(errors="replace")[-4000:]
            )
        lowered = {}
        for expr in (kexpr, MERGE):
            name = _check(
                nvrtc.nvrtcGetLoweredName(prog, expr.encode()), "nvrtcGetLoweredName"
            )
            lowered[expr] = name.decode() if isinstance(name, bytes) else str(name)
        size = _check(nvrtc.nvrtcGetCUBINSize(prog), "nvrtcGetCUBINSize")
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
    finally:
        nvrtc.nvrtcDestroyProgram(prog)
    cubin_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cubin_path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_bytes(cubin)
    tmp.replace(cubin_path)
    tmpn = names_path.with_suffix(f".{os.getpid()}.ntmp")
    tmpn.write_text(lowered[kexpr] + "\n" + lowered[MERGE] + "\n")
    tmpn.replace(names_path)
    return cubin, lowered


class Params(ctypes.Structure):
    """Mirror of muse_vr::pfa::Params (field order and types as in the device code)."""

    _fields_ = [
        ("units", ctypes.c_void_p),
        ("cta_units", ctypes.c_void_p),
        ("ws_o", ctypes.c_void_p),
        ("ws_ml", ctypes.c_void_p),
        ("q", ctypes.c_void_p),
        ("page_table", ctypes.c_void_p),
        ("seq_lens", ctypes.c_void_p),
        ("cu_q", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("q_token_stride", ctypes.c_int64),
        ("page_table_stride", ctypes.c_int64),
        ("out_token_stride", ctypes.c_int64),
        ("scale_log2", ctypes.c_float),
        ("out_scale", ctypes.c_float),
        ("trace", ctypes.c_void_p),
    ]


class Kernels:
    """Loaded functions of one device (built once by prepare())."""

    def __init__(self, device_index: int, opt: int | None = None):
        cu = _drv()
        self.opt = opt_from_env() if opt is None else int(opt)
        major, minor = torch.cuda.get_device_capability(device_index)
        cubin, names = compile_cubin(f"sm_{major}{minor}a", self.opt)
        with torch.cuda.device(device_index):
            torch.zeros(1, device=f"cuda:{device_index}")  # primary context current
            self.module = _check(cu.cuModuleLoadData(cubin), "cuModuleLoadData")
            self.attn = _check(
                cu.cuModuleGetFunction(
                    self.module, names[kernel_expr(self.opt)].encode()
                ),
                "get attention kernel",
            )
            self.merge = _check(
                cu.cuModuleGetFunction(self.module, names[MERGE].encode()),
                "get merge kernel",
            )
            _check(
                cu.cuFuncSetAttribute(
                    self.attn,
                    cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    SMEM_BYTES,
                ),
                "cuFuncSetAttribute(smem)",
            )


_MAPS: dict = {}
_MAPS_MAX = 256


def tensor_map(cache: torch.Tensor):
    """TMA descriptor of a [pages, Hkv, 128, 64] e4m3 cache view: dims {64 B, 128 tokens, Hkv, pages}, box {64, 128, 1, 1},
    64-byte swizzle (the device code's shared-memory layout). The descriptor holds only the address, dims and strides,
    so it is cached by exactly those (a served KV pool keeps its buffers for the server's life)."""
    key = (
        cache.device.index,
        cache.data_ptr(),
        str(cache.dtype),
        tuple(cache.shape),
        tuple(cache.stride()),
    )
    m = _MAPS.get(key)
    if m is None:
        if len(_MAPS) >= _MAPS_MAX:
            _MAPS.clear()
        m = _MAPS[key] = _encode_map(cache)
    return m


def _encode_map(cache: torch.Tensor):
    cu = _drv()
    pages, hkv = cache.shape[0], cache.shape[1]
    dims = [64, PAGE, hkv, pages]
    strides = [cache.stride(2), cache.stride(1), cache.stride(0)]  # bytes (e4m3)
    box = [64, PAGE, 1, 1]
    u64, u32 = cu.cuuint64_t, cu.cuuint32_t
    return _check(
        cu.cuTensorMapEncodeTiled(
            cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
            4,
            cache.data_ptr(),
            [u64(v) for v in dims],
            [u64(v) for v in strides],
            [u32(v) for v in box],
            [u32(1)] * 4,
            cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B,
            cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        ),
        "cuTensorMapEncodeTiled",
    )


def _launch(function, grid, block, smem, args, stream, pdl: bool):
    cu = _drv()
    addrs = [
        a.getPtr() if isinstance(a, cu.CUtensorMap) else ctypes.addressof(a)
        for a in args
    ]
    arr = (ctypes.c_void_p * len(addrs))(*addrs)
    cfg = cu.CUlaunchConfig()
    cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid
    cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = block
    cfg.sharedMemBytes = smem
    cfg.hStream = cu.CUstream(stream)
    if pdl:  # the kernel waits on its predecessor (griddepcontrol.wait) before reading anything
        attr = cu.CUlaunchAttribute()
        attr.id = (
            cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
        )
        attr.value.programmaticStreamSerializationAllowed = 1
        cfg.attrs = [attr]
        cfg.numAttrs = 1
    else:
        cfg.numAttrs = 0
    _check(
        cu.cuLaunchKernelEx(cfg, function, ctypes.addressof(arr), 0), "cuLaunchKernelEx"
    )


def launch(
    k: Kernels,
    *,
    units,
    cta_units,
    munits,
    ws_o,
    ws_ml,
    q,
    k_cache,
    v_cache,
    page_table,
    seq_lens,
    cu_q,
    out,
    bmm1_scale: float,
    bmm2_scale: float,
    k_map=None,
    v_map=None,
    stream=None,
    pdl: bool = True,
) -> None:
    """One attention launch (grid = blocks of cta_units) and, when there are split pieces, the merge launch.
    ``pdl``: the attention launch's programmatic-serialization attribute (tests switch it off)."""
    p = Params(
        units.data_ptr(),
        cta_units.data_ptr(),
        ws_o.data_ptr(),
        ws_ml.data_ptr(),
        q.data_ptr(),
        page_table.data_ptr(),
        seq_lens.data_ptr(),
        cu_q.data_ptr(),
        out.data_ptr(),
        q.stride(0),
        page_table.stride(0),
        out.stride(0),
        float(bmm1_scale * math.log2(math.e)),
        float(bmm2_scale),
        None,
    )
    km = tensor_map(k_cache) if k_map is None else k_map
    vm = tensor_map(v_cache) if v_map is None else v_map
    s = torch.cuda.current_stream().cuda_stream if stream is None else stream
    blocks = cta_units.shape[0] - 1
    _launch(
        k.attn, (blocks, 1, 1), (THREADS, 1, 1), SMEM_BYTES, [p, km, vm], s, pdl=pdl
    )
    if munits.shape[0] > 0:
        args = [
            ctypes.c_void_p(munits.data_ptr()),
            ctypes.c_void_p(cu_q.data_ptr()),
            ctypes.c_void_p(ws_o.data_ptr()),
            ctypes.c_void_p(ws_ml.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int64(out.stride(0)),
            ctypes.c_float(float(bmm2_scale)),
        ]
        # the merge has no programmatic wait: plain stream order after the attention kernel
        _launch(
            k.merge,
            (munits.shape[0], 1, 1),
            (MERGE_THREADS, 1, 1),
            0,
            args,
            s,
            pdl=False,
        )
