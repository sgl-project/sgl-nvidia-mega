# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The GatedDeltaNet verify core of a target-verify step in one launch (``src/core.cu``): the
ReplaySSM verify on the circular ring (non-flush rows; this step's ring entries written as
SGLang's kernel writes them) and the gated RMSNorm writing out_proj's static FP8 input. One CTA
per (request, value head); numeric against slot gdn_verify's two launches (FP32 sums in other
orders, the same BF16 rounding points).

It launches as a programmatic dependent: the checkpoint rows, ring cursors and static parameters
load before ``griddepcontrol.wait``, every activation (q / k / v, z, a, b) after it. With
``early_dependents`` the next launch may start at once (it must wait before reading the output).
The testable ``early_hist`` variant can also hold committed history ahead of the wait;
production keeps ordinary history loads. Flat stores compute the ring entries and FP8
outputs first and predicate only their stores, then store update vectors after the
output; the arithmetic is unchanged.
"""

from __future__ import annotations

import ctypes
from functools import cache
from pathlib import Path
from typing import Optional

import torch

from sgl_nvidia_mega.common import jit

SOURCE = Path(__file__).parent / "src" / "core.cu"
THREADS, K, V, RING, MAX_ROWS_PER_REQUEST = 128, 128, 128, 16, 8
# Validated production policy: ordinary history loads, flat stores, and the early
# dependent release only over the measured 35-56-row window.
EARLY = False
RELEASE_MIN_ROWS, RELEASE_MAX_ROWS = 35, 56
FLAT = True


class Params(ctypes.Structure):
    """Params of src/core.cu."""

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


@cache
def _kernel(
    device_index: int,
    rows: int,
    hk: int,
    hv: int,
    norm_f32: bool,
    trigger: int,
    store_o: bool,
    prof: bool = False,
    warps: int = 4,
    min_blocks: int = 1,
    early_hist: bool = False,
    flat: bool = False,
) -> jit.Kernel:
    major, minor = torch.cuda.get_device_capability(device_index)
    defines = dict(
        TT=rows,
        HK=hk,
        HV=hv,
        NORM_F32=int(norm_f32),
        TRIGGER=trigger,
        STORE_O=int(store_o),
        PROF=int(prof),
        NW=warps,
        MINB=min_blocks,
    )
    if early_hist:
        defines["EARLY_HIST"] = 1
    if flat:
        defines["FLAT_STORES"] = 1
    source = "".join(f"#define {k} {v}\n" for k, v in defines.items())
    cubin = jit.compile_source(
        source + SOURCE.read_text(),
        "q35_gdn_core",
        f"sm_{major}{minor}a",
        ("-std=c++17", "-default-device"),
    )
    return jit.load(cubin, "q35_gdn_core", (32 * warps, 1, 1), 0)


def build(device_index: int, rows: int, hk: int, hv: int, norm_f32: bool) -> None:
    """Compile the core for a step shape, both dependent releases and both register caps
    (no compile inside a step: CUDA-graph capture refuses it)."""
    for trigger in (0, 2):
        for min_blocks in (1, 3):
            _kernel(
                device_index,
                rows,
                hk,
                hv,
                norm_f32,
                trigger,
                False,
                min_blocks=min_blocks,
                flat=FLAT,
            )
        if EARLY:
            _kernel(
                device_index,
                rows,
                hk,
                hv,
                norm_f32,
                trigger,
                False,
                early_hist=True,
                flat=FLAT,
            )


@cache
def _sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _min_blocks(device_index: int, ctas: int) -> int:
    """Three CTAs an SM (168 registers) once two an SM take more than two waves (128 requests
    at TP8): past that the third resident CTA wins; below it the uncapped registers do."""
    return 3 if ctas > 4 * _sms(device_index) else 1


def _slot_stride(t: Optional[torch.Tensor]) -> int:
    return 0 if t is None else t.stride(0)


def core(
    qkv: torch.Tensor,
    z: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    checkpoint: torch.Tensor,
    ring_d: torch.Tensor,
    ring_k: torch.Tensor,
    ring_g: torch.Tensor,
    ring_dlo: Optional[torch.Tensor],
    ring_klo: Optional[torch.Tensor],
    state_indices: torch.Tensor,
    replay_indices: torch.Tensor,
    write_pos: torch.Tensor,
    cache_base: torch.Tensor,
    is_flush: torch.Tensor,
    norm_weight: torch.Tensor,
    input_scale: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    *,
    rows_per_request: int,
    num_k_heads: int,
    early_dependents: bool = False,
    o_out: Optional[torch.Tensor] = None,
    _profile: Optional[torch.Tensor] = None,
    warps: int = 4,
    min_blocks: Optional[int] = None,
    early_hist: Optional[bool] = None,
    flat: Optional[bool] = None,
) -> torch.Tensor:
    """Write the gated RMSNorm of the verify output as FP8 into ``out`` [rows, HV * V].

    ``qkv`` [rows, >= (2 H + HV) * 128] BF16 convolved q | k | v rows (unit column stride,
    16-byte aligned rows); ``z`` [rows, HV, V] with z.stride(1) == V; ``a`` / ``b`` [rows, HV]
    row-strided views with one row stride; ``checkpoint`` [slots, HV, V, K] BF16; the ring
    [slots, HV | H, 16, V | K] (low parts optional) with cursors by ``replay_indices``; the first
    ``state_indices.numel()`` requests of ``rows_per_request`` rows each. ``o_out`` [rows, HV, V]
    BF16 also receives the replay output (checks only). ``early_hist`` (default ``EARLY``) loads the
    committed history before the PDL wait when the grid runs one CTA an SM (same bits); ``flat``
    (default ``FLAT``) predicates the stores (same bits)."""
    rows, T = qkv.shape[0], rows_per_request
    n = rows // T
    HV = checkpoint.shape[1]
    H = num_k_heads
    if (
        qkv.dtype != torch.bfloat16
        or qkv.stride(1) != 1
        or qkv.stride(0) % 8
        or qkv.data_ptr() % 16
        or qkv.shape[1] < (2 * H + HV) * K
    ):
        raise ValueError(
            "q | k | v rows must be BF16, unit column stride, 16-byte aligned"
        )
    if not 2 <= T <= MAX_ROWS_PER_REQUEST or rows % T or state_indices.numel() < n:
        raise ValueError(f"{rows} rows of {T} a request")
    if tuple(checkpoint.shape[1:]) != (HV, V, K) or checkpoint.dtype != torch.bfloat16:
        raise ValueError(f"checkpoint {checkpoint.dtype} {tuple(checkpoint.shape)}")
    if checkpoint.stride()[1:] != (V * K, K, 1):
        raise ValueError("checkpoint slots must be [HV, V, K] contiguous")
    for t, heads, dim in (
        (ring_d, HV, V),
        (ring_dlo, HV, V),
        (ring_k, H, K),
        (ring_klo, H, K),
    ):
        if t is not None and (
            tuple(t.shape[1:]) != (heads, RING, dim)
            or t.stride()[1:] != (RING * dim, dim, 1)
        ):
            raise ValueError(f"ring {tuple(t.shape)} {tuple(t.stride())}")
    if tuple(ring_g.shape[1:]) != (HV, RING) or ring_g.stride()[1:] != (RING, 1):
        raise ValueError(f"ring g {tuple(ring_g.shape)}")
    if z.shape != (rows, HV, V) or z.stride(2) != 1 or z.stride(1) != V:
        raise ValueError(f"z {tuple(z.shape)} {tuple(z.stride())}")
    if a.stride(1) != 1 or b.stride(1) != 1 or a.stride(0) != b.stride(0):
        raise ValueError("a / b must be row-strided views with one row stride")
    if (
        out.dtype != torch.float8_e4m3fn
        or out.shape != (rows, HV * V)
        or out.stride(1) != 1
        or out.stride(0) % 16
        or out.data_ptr() % 16
    ):
        raise ValueError(f"out {out.dtype} {tuple(out.shape)}")
    if write_pos.dtype != torch.int32 or cache_base.dtype != torch.int32:
        raise ValueError("ring cursors must be int32")
    if is_flush.dtype != torch.int8:
        raise ValueError("is_flush must be int8")
    p = Params(
        x=qkv.data_ptr(),
        x_stride=qkv.stride(0),
        z=z.data_ptr(),
        z_stride=z.stride(0),
        a=a.data_ptr(),
        b=b.data_ptr(),
        ab_stride=a.stride(0),
        A_log=A_log.data_ptr(),
        dt_bias=dt_bias.data_ptr(),
        state=checkpoint.data_ptr(),
        state_slot=checkpoint.stride(0),
        ring_d=ring_d.data_ptr(),
        d_slot=ring_d.stride(0),
        ring_k=ring_k.data_ptr(),
        k_slot=ring_k.stride(0),
        ring_g=ring_g.data_ptr(),
        g_slot=ring_g.stride(0),
        ring_dlo=0 if ring_dlo is None else ring_dlo.data_ptr(),
        dlo_slot=_slot_stride(ring_dlo),
        ring_klo=0 if ring_klo is None else ring_klo.data_ptr(),
        klo_slot=_slot_stride(ring_klo),
        state_indices=state_indices.data_ptr(),
        replay_indices=replay_indices.data_ptr(),
        state_index64=int(state_indices.dtype == torch.int64),
        replay_index64=int(replay_indices.dtype == torch.int64),
        write_pos=write_pos.data_ptr(),
        cache_base=cache_base.data_ptr(),
        is_flush=is_flush.data_ptr(),
        norm_w=norm_weight.data_ptr(),
        scale=input_scale.data_ptr(),
        eps=eps,
        qscale=K**-0.5,
        y=out.data_ptr(),
        y_stride=out.stride(0),
        o_out=0 if o_out is None else o_out.data_ptr(),
        prof=0 if _profile is None else _profile.data_ptr(),
    )
    min_blocks = (
        _min_blocks(qkv.device.index, n * HV) if min_blocks is None else min_blocks
    )
    early_hist = (EARLY if early_hist is None else early_hist) and min_blocks == 1
    kernel = _kernel(
        qkv.device.index,
        T,
        H,
        HV,
        norm_weight.dtype == torch.float32,
        2 if early_dependents else 0,
        o_out is not None,
        _profile is not None,
        warps,
        min_blocks,
        early_hist,
        FLAT if flat is None else flat,
    )
    jit.launch(
        kernel, (n * HV, 1, 1), [p], torch.cuda.current_stream(qkv.device).cuda_stream
    )
    return out
