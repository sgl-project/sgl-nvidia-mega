# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import ctypes
from functools import cache
from pathlib import Path
from typing import Optional

import torch

from sgl_nvidia_mega.common import jit

SOURCE = Path(__file__).parent / "src" / "commit.cu"
K = V = 128
RING = 16  # the ring length this build serves
BV, STAGES, THREADS = 64, 2, 256


class Params(ctypes.Structure):
    _fields_ = [
        ("h0", ctypes.c_void_p),
        ("d", ctypes.c_void_p),
        ("dr", ctypes.c_void_p),
        ("k", ctypes.c_void_p),
        ("kr", ctypes.c_void_p),
        ("g", ctypes.c_void_p),
        ("state_idx", ctypes.c_void_p),
        ("replay_idx", ctypes.c_void_p),
        ("accept_lens", ctypes.c_void_p),
        ("track_idx", ctypes.c_void_p),
        ("track_steps", ctypes.c_void_p),
        ("write_pos", ctypes.c_void_p),
        ("cache_base", ctypes.c_void_p),
        ("is_flush", ctypes.c_void_p),
        ("s_state_layer", ctypes.c_int64),
        ("s_state_slot", ctypes.c_int64),
        ("s_d_layer", ctypes.c_int64),
        ("s_d_slot", ctypes.c_int64),
        ("s_dr_layer", ctypes.c_int64),
        ("s_dr_slot", ctypes.c_int64),
        ("s_k_layer", ctypes.c_int64),
        ("s_k_slot", ctypes.c_int64),
        ("s_kr_layer", ctypes.c_int64),
        ("s_kr_slot", ctypes.c_int64),
        ("s_g_layer", ctypes.c_int64),
        ("s_g_slot", ctypes.c_int64),
        ("num_layers", ctypes.c_int32),
        ("layers_per_cta", ctypes.c_int32),
        ("hv", ctypes.c_int32),
        ("hk", ctypes.c_int32),
        ("L", ctypes.c_int32),
        ("null_block", ctypes.c_int32),
        ("idx64", ctypes.c_int32),
        ("has_track", ctypes.c_int32),
        ("chunks", ctypes.c_int32),
        ("items", ctypes.c_int32),
    ]


def shared_bytes(ring: int = RING) -> int:
    stage = 2 * (BV * K + 2 * ring * K + 2 * ring * BV) + 4 * ring
    return STAGES * stage


@cache
def _kernels(device_index: int, ring: int) -> tuple[jit.Kernel, jit.Kernel]:
    major, minor = torch.cuda.get_device_capability(device_index)
    defines = dict(MAX_L=ring, BV=BV, STAGES=STAGES)
    source = "".join(f"#define {k} {v}\n" for k, v in defines.items())
    cubin = jit.compile_source(
        source + SOURCE.read_text(),
        "q35_gdn_commit",
        f"sm_{major}{minor}a",
        ("-std=c++17", "-default-device"),
    )
    return (
        jit.load(cubin, "q35_gdn_commit", (THREADS, 1, 1), shared_bytes(ring)),
        jit.load(cubin, "q35_gdn_commit_finish", (128, 1, 1), 0),
    )


def build(device_index: int) -> None:
    """Compile and load both kernels (no compile inside a step)."""
    _kernels(device_index, RING)


def default_layers_per_cta(rows: int) -> int:
    """Layers a CTA walks: fewer at small batches, so the grid has more CTAs."""
    if rows < 2:
        return 3
    return 5 if rows <= 12 else 9


def _unit(t: torch.Tensor) -> torch.Tensor:
    # the kernels read the per-row vectors with unit stride
    return t if t.stride(0) == 1 else t.contiguous()


def _is64(t: torch.Tensor) -> int:
    return int(t.dtype == torch.int64)


def commit(
    *,
    checkpoint: torch.Tensor,
    ring_d: torch.Tensor,
    ring_dlo: torch.Tensor,
    ring_k: torch.Tensor,
    ring_klo: torch.Tensor,
    ring_g: torch.Tensor,
    state_indices: torch.Tensor,
    ring_rows: torch.Tensor,
    write_pos: torch.Tensor,
    cache_base: torch.Tensor,
    is_flush: torch.Tensor,
    accept_lens: torch.Tensor,
    track_indices: Optional[torch.Tensor],
    track_steps: Optional[torch.Tensor],
    null_block: int = -1,
    layers_per_cta: Optional[int] = None,
) -> None:
    rows = state_indices.shape[0]
    if rows == 0:
        return
    has_track = track_indices is not None and track_steps is not None
    state_indices, ring_rows, accept_lens = map(
        _unit, (state_indices, ring_rows, accept_lens)
    )
    if has_track:
        track_indices, track_steps = _unit(track_indices), _unit(track_steps)
    else:
        track_indices, track_steps = state_indices, accept_lens
    num_layers, _, hv = checkpoint.shape[:3]
    hk, ring = ring_k.shape[2], ring_d.shape[-2]
    if layers_per_cta is None:
        layers_per_cta = default_layers_per_cta(rows)
    chunks = -(-num_layers // layers_per_cta)
    items = rows * hv * (V // BV) * chunks
    p = Params(
        h0=checkpoint.data_ptr(),
        d=ring_d.data_ptr(),
        dr=ring_dlo.data_ptr(),
        k=ring_k.data_ptr(),
        kr=ring_klo.data_ptr(),
        g=ring_g.data_ptr(),
        state_idx=state_indices.data_ptr(),
        replay_idx=ring_rows.data_ptr(),
        accept_lens=accept_lens.data_ptr(),
        track_idx=track_indices.data_ptr(),
        track_steps=track_steps.data_ptr(),
        write_pos=write_pos.data_ptr(),
        cache_base=cache_base.data_ptr(),
        is_flush=is_flush.data_ptr(),
        s_state_layer=checkpoint.stride(0),
        s_state_slot=checkpoint.stride(1),
        s_d_layer=ring_d.stride(0),
        s_d_slot=ring_d.stride(1),
        s_dr_layer=ring_dlo.stride(0),
        s_dr_slot=ring_dlo.stride(1),
        s_k_layer=ring_k.stride(0),
        s_k_slot=ring_k.stride(1),
        s_kr_layer=ring_klo.stride(0),
        s_kr_slot=ring_klo.stride(1),
        s_g_layer=ring_g.stride(0),
        s_g_slot=ring_g.stride(1),
        num_layers=num_layers,
        layers_per_cta=layers_per_cta,
        hv=hv,
        hk=hk,
        L=ring,
        null_block=null_block,
        idx64=_is64(state_indices)
        | _is64(ring_rows) << 1
        | _is64(accept_lens) << 2
        | _is64(track_indices) << 3
        | _is64(track_steps) << 4,
        has_track=int(has_track),
        chunks=chunks,
        items=items,
    )
    main, finish = _kernels(checkpoint.device.index, ring)
    stream = torch.cuda.current_stream(checkpoint.device).cuda_stream
    jit.launch(main, (items, 1, 1), [p], stream, pdl=False)
    jit.launch(finish, (1, 1, 1), [p, ctypes.c_int(rows)], stream, pdl=False)
