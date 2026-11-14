# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team): the launch of its MMA ReplaySSM
# verify core (Params, _kernel, core). Changed: the kernel's name; the
# FLAT_STORES variant and the 16 / 32-row switches are not taken.
"""The ReplaySSM verify core of a GDN target verify with its gated RMSNorm, one NVRTC
launch (``jit/csrc/attention/mkgdn_core.cu``): one CTA per (request, value head), MMA
products, the circular ring's entries of this step and, with ``normalized``, the
BF16 ``RMSNorm(o) * w * sigmoid(z)`` rows the out_proj takes.

The kernel compiles once per (device, heads, dtypes); a compile while a CUDA graph is
being captured raises, so an eager step must run it first.
"""

from __future__ import annotations

import ctypes
from functools import cache
from pathlib import Path

import torch

SOURCE = Path(__file__).parents[3] / "jit/csrc/attention/mkgdn_core.cu"


class Params(ctypes.Structure):
    """Params of mkgdn_core.cu."""

    _fields_ = [
        ("x", ctypes.c_void_p),
        ("x_stride", ctypes.c_int64),
        ("z", ctypes.c_void_p),
        ("z_stride", ctypes.c_int64),
        ("a", ctypes.c_void_p),
        ("b", ctypes.c_void_p),
        ("ab_stride", ctypes.c_int64),
        ("A_log", ctypes.c_void_p),
        ("dt_bias", ctypes.c_void_p),
        ("state", ctypes.c_void_p),
        ("state_slot", ctypes.c_int64),
        ("ring_d", ctypes.c_void_p),
        ("d_slot", ctypes.c_int64),
        ("ring_k", ctypes.c_void_p),
        ("k_slot", ctypes.c_int64),
        ("ring_g", ctypes.c_void_p),
        ("g_slot", ctypes.c_int64),
        ("ring_dlo", ctypes.c_void_p),
        ("dlo_slot", ctypes.c_int64),
        ("ring_klo", ctypes.c_void_p),
        ("klo_slot", ctypes.c_int64),
        ("state_indices", ctypes.c_void_p),
        ("replay_indices", ctypes.c_void_p),
        ("state_index64", ctypes.c_int),
        ("replay_index64", ctypes.c_int),
        ("write_pos", ctypes.c_void_p),
        ("cache_base", ctypes.c_void_p),
        ("is_flush", ctypes.c_void_p),
        ("norm_w", ctypes.c_void_p),
        ("scale", ctypes.c_void_p),
        ("eps", ctypes.c_float),
        ("qscale", ctypes.c_float),
        ("y", ctypes.c_void_p),
        ("y_stride", ctypes.c_int64),
        ("o_out", ctypes.c_void_p),
        ("prof", ctypes.c_void_p),
    ]


def _check(result, what):
    if int(result[0]):
        raise RuntimeError(f"mkgdn_core: {what}: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


@cache
def _kernel(device, hk, hv, dt_bf16, fused_norm, norm_f32, vrows):
    from cuda.bindings import driver as cu
    from cuda.bindings import nvrtc

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("mkgdn_core: compile it in an eager step before capture")
    defs = dict(
        TT=4,
        HK=hk,
        HV=hv,
        DT_BF16=int(dt_bf16),
        FUSED_NORM=int(fused_norm),
        NORM_F32=int(norm_f32),
        VROWS=vrows,
        STORE_O=1,
        TRIGGER=2,
        PROF=0,
        NW=4,
        MINB=1,
    )
    src = "".join(f"#define {k} {v}\n" for k, v in defs.items()) + SOURCE.read_text()
    prog = _check(
        nvrtc.nvrtcCreateProgram(src.encode(), b"mkgdn_core.cu", 0, [], []), "create"
    )
    major, minor = torch.cuda.get_device_capability(device)
    opts = [
        f"--gpu-architecture=sm_{major}{minor}a".encode(),
        b"-std=c++17",
        b"-default-device",
    ]
    try:
        status = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)[0]
        if int(status):
            size = _check(nvrtc.nvrtcGetProgramLogSize(prog), "log size")
            log = b" " * size
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError(log.decode(errors="replace"))
        cubin = b" " * _check(nvrtc.nvrtcGetCUBINSize(prog), "cubin size")
        _check(nvrtc.nvrtcGetCUBIN(prog, cubin), "cubin")
    finally:
        nvrtc.nvrtcDestroyProgram(prog)
    module = _check(cu.cuModuleLoadData(cubin), "module")
    fn = _check(cu.cuModuleGetFunction(module, b"mkgdn_core"), "function")
    return module, fn


def core(
    qkv,
    a,
    b,
    A_log,
    dt_bias,
    checkpoint,
    ring_d,
    ring_k,
    ring_g,
    ring_dlo,
    ring_klo,
    state_indices,
    replay_indices,
    write_pos,
    cache_base,
    is_flush,
    out,
    *,
    z=None,
    norm_weight=None,
    normalized=None,
    eps=1e-6,
    vrows=128,
):
    """Write raw BF16 O; optionally write BF16 RMSNorm*sigmoid(z).

    qkv is convolved [M,(2*HK+HV)*128], a/b are packed-row-strided.
    Cursors/checkpoint belong to completed prior steps; caller must join commit.
    """
    from cuda.bindings import driver as cu

    hk, hv = ring_k.shape[1], checkpoint.shape[1]
    rows = qkv.shape[0]
    if (hk, hv) not in ((4, 12), (16, 48)) or rows not in (4, 8, 12):
        raise ValueError(
            "mkgdn_core takes 4, 8 or 12 rows of (4, 12) or (16, 48) heads"
        )
    if qkv.dtype != torch.bfloat16 or checkpoint.dtype != torch.bfloat16:
        raise ValueError("mkgdn_core requires BF16 activations/checkpoint")
    if qkv.stride(1) != 1 or qkv.stride(0) % 8 or qkv.data_ptr() % 16:
        raise ValueError("mkgdn_core requires 16-byte aligned contiguous qkv rows")
    if a.stride(0) != b.stride(0) or a.stride(1) != 1 or b.stride(1) != 1:
        raise ValueError("mkgdn_core requires matching a/b row strides")
    if out.shape != (rows, hv, 128) or not out.is_contiguous():
        raise ValueError("mkgdn_core raw output must be contiguous [M,HV,128]")
    if vrows not in (64, 128):
        raise ValueError("V slice must be 64 or 128")
    fused = normalized is not None
    if fused and vrows != 128:
        raise ValueError("fused RMS requires a full-head owner")
    if fused and (z is None or norm_weight is None):
        raise ValueError("fused norm needs z and norm_weight")
    ptr = lambda t: 0 if t is None else t.data_ptr()
    stride = lambda t: 0 if t is None else t.stride(0)
    y = out if not fused else normalized
    p = Params(
        x=ptr(qkv),
        x_stride=stride(qkv),
        z=ptr(z),
        z_stride=stride(z),
        a=ptr(a),
        b=ptr(b),
        ab_stride=stride(a),
        A_log=ptr(A_log),
        dt_bias=ptr(dt_bias),
        state=ptr(checkpoint),
        state_slot=stride(checkpoint),
        ring_d=ptr(ring_d),
        d_slot=stride(ring_d),
        ring_k=ptr(ring_k),
        k_slot=stride(ring_k),
        ring_g=ptr(ring_g),
        g_slot=stride(ring_g),
        ring_dlo=ptr(ring_dlo),
        dlo_slot=stride(ring_dlo),
        ring_klo=ptr(ring_klo),
        klo_slot=stride(ring_klo),
        state_indices=ptr(state_indices),
        replay_indices=ptr(replay_indices),
        state_index64=int(state_indices.dtype == torch.int64),
        replay_index64=int(replay_indices.dtype == torch.int64),
        write_pos=ptr(write_pos),
        cache_base=ptr(cache_base),
        is_flush=ptr(is_flush),
        norm_w=ptr(norm_weight),
        scale=0,
        eps=eps,
        qscale=128**-0.5,
        y=ptr(y),
        y_stride=y.stride(0),
        o_out=ptr(out),
        prof=0,
    )
    _, fn = _kernel(
        qkv.device.index,
        hk,
        hv,
        dt_bias.dtype == torch.bfloat16,
        fused,
        norm_weight is not None and norm_weight.dtype == torch.float32,
        vrows,
    )
    cfg = cu.CUlaunchConfig()
    cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = rows // 4 * hv * (128 // vrows), 1, 1
    cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = 128, 1, 1
    cfg.sharedMemBytes = 0
    cfg.hStream = cu.CUstream(torch.cuda.current_stream(qkv.device).cuda_stream)
    attr = cu.CUlaunchAttribute()
    attr.id = (
        cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
    )
    attr.value.programmaticStreamSerializationAllowed = 1
    cfg.attrs, cfg.numAttrs = [attr], 1
    args = (ctypes.c_void_p * 1)(ctypes.addressof(p))
    _check(cu.cuLaunchKernelEx(cfg, fn, ctypes.addressof(args), 0), "launch")
    return out if not fused else normalized
