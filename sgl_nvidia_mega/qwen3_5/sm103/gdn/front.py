# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The GatedDeltaNet verify front of one request of 7 rows at TP8 in one launch
(``src/front.cu``): the static-FP8 ``in_proj_qkvz`` projection of the BF16 rows, the
causal conv update of its q / k / v channels (output, rolling state, deduplicated
intermediate windows), the BF16 ``in_proj_ba`` side projection and an L2 prefetch of the
ReplaySSM verify's checkpoint and history ring. ``gdn_pdl_front_ba_mkm`` of SGLang branch
nvidia/qwen35-agentx-gb300 (the kernel and its launch); the projections sum in another
order than SGLang's GEMMs (numeric), the conv equals SGLang's on the projected rows.

It launches as a programmatic dependent and releases its own dependents at entry: the
next launch must wait (``griddepcontrol.wait``) before it reads any output, b / a
included. By default (``tma``) each projection CTA's packed weights arrive by one bulk
copy into shared memory issued before that wait, instead of register loads; same MMAs,
same bits.
"""

from __future__ import annotations

import ctypes
from functools import cache
from pathlib import Path

import torch

from sgl_nvidia_mega.common import jit

SOURCE = Path(__file__).parent / "src" / "front.cu"
ROWS, HIDDEN, QKVZ, CONV, BA = 7, 4096, 2560, 1536, 16
THREADS, CLUSTER, CTAS = (
    256,
    8,
    88,
)  # 10 projection clusters + 1 side-projection cluster
TIMEOUT = (
    500_000_000  # clock cycles a CTA waits for its cluster's partials before trapping
)
# one projection CTA's packed weights (kWeightBytes)
WEIGHT_BYTES = 8 * 16 * 2 * 32 * 16


class Prefetch(ctypes.Structure):
    """gdn_ba::Prefetch."""

    _fields_ = [
        ("base", ctypes.c_void_p),
        ("slot_stride", ctypes.c_int64),
        ("bytes", ctypes.c_int64),
        ("ring", ctypes.c_int),
    ]


class BaParams(ctypes.Structure):
    """gdn_ba::Params."""

    _fields_ = [
        ("input", ctypes.c_void_p),
        ("weight", ctypes.c_void_p),
        ("output", ctypes.c_void_p),
        ("rows", ctypes.c_int),
        ("input_stride", ctypes.c_int64),
        ("prefetch", Prefetch * 4),
        ("prefetches", ctypes.c_int),
        ("state_indices", ctypes.c_void_p),
        ("ring_indices", ctypes.c_void_p),
        ("state_index64", ctypes.c_bool),
        ("ring_index64", ctypes.c_bool),
        ("pad_slot", ctypes.c_int64),
    ]


class Params(ctypes.Structure):
    """gdn_pdl_front_ba_mkm_detail::Params."""

    _fields_ = [
        ("x", ctypes.c_void_p),
        ("weight", ctypes.c_void_p),
        ("sx", ctypes.c_void_p),
        ("sw", ctypes.c_void_p),
        ("output", ctypes.c_void_p),
        ("error", ctypes.c_void_p),
        ("timeout", ctypes.c_uint64),
        ("m", ctypes.c_int),
        ("conv_weight", ctypes.c_void_p),
        ("conv_state", ctypes.c_void_p),
        ("windows", ctypes.c_void_p),
        ("conv_output", ctypes.c_void_p),
        ("state_indices", ctypes.c_void_p),
        ("replay_indices", ctypes.c_void_p),
        ("state_stride", ctypes.c_int64 * 3),
        ("window_stride", ctypes.c_int64 * 2),
        ("state_index64", ctypes.c_bool),
        ("replay_index64", ctypes.c_bool),
        ("pad_slot", ctypes.c_int64),
        ("ba", BaParams),
    ]


def pack(weight_nk: torch.Tensor) -> torch.Tensor:
    """in_proj_qkvz's FP8 weights [2560, 4096] (row-major [N, K]) packed once, as the
    front's tensor-core fragments take them."""
    if weight_nk.dtype != torch.float8_e4m3fn or tuple(weight_nk.shape) != (
        QKVZ,
        HIDDEN,
    ):
        raise ValueError(
            f"the GDN front packs FP8 E4M3 [{QKVZ}, {HIDDEN}] weights, not "
            f"{weight_nk.dtype} {tuple(weight_nk.shape)}"
        )
    value = weight_nk.contiguous().view(torch.uint8)
    value = value.reshape(10, 8, 2, 2, 8, 8, 16, 2, 4, 4)
    return value.permute(0, 5, 1, 6, 2, 4, 8, 7, 3, 9).contiguous().view(-1)


@cache
def _kernel(device_index: int, tma: bool = True) -> jit.Kernel:
    major, minor = torch.cuda.get_device_capability(device_index)
    cubin = jit.compile_source(
        SOURCE.read_text(),
        "q35_gdn_front",
        f"sm_{major}{minor}a",
        ("-std=c++17", "-default-device"),
    )
    if tma:
        return jit.load(cubin, "q35_gdn_front_tma", (THREADS, 1, 1), WEIGHT_BYTES)
    return jit.load(cubin, "q35_gdn_front", (THREADS, 1, 1), 0)


def build(device_index: int) -> None:
    """Compile the front (no compile inside a step)."""
    _kernel(device_index, True)


def _prefetch(t: torch.Tensor, ring: int):
    """One L2 prefetch region (slot ``index`` of a [slots, ...] tensor), or None when
    a slot is not contiguous or not 16-byte sized and aligned (as the branch skips it)."""
    if t is None or t.dim() < 2:
        return None
    contiguous, expect = True, 1
    for d in range(t.dim() - 1, 0, -1):
        contiguous = contiguous and (t.shape[d] == 1 or t.stride(d) == expect)
        expect *= t.shape[d]
    nbytes = expect * t.element_size()
    if not contiguous or nbytes % 16 or t.data_ptr() % 16:
        return None
    return Prefetch(t.data_ptr(), t.stride(0) * t.element_size(), nbytes, ring)


def front(
    x: torch.Tensor,
    packed: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    windows: torch.Tensor,
    state_indices: torch.Tensor,
    replay_indices: torch.Tensor,
    projected: torch.Tensor,
    conv_out: torch.Tensor,
    error: torch.Tensor,
    *,
    ba_weight: torch.Tensor,
    ba_out: torch.Tensor,
    checkpoint: torch.Tensor,
    history_d: torch.Tensor,
    history_k: torch.Tensor,
    history_g: torch.Tensor,
    ring_indices: torch.Tensor,
    pad_slot: int = -1,
    tma: bool = True,
) -> None:
    """The front of the BF16 rows ``x`` [7, 4096] (contiguous) into the caller's
    buffers: ``projected`` [7, 2560] BF16 (q / k / v / z), ``conv_out`` [7, 1536] BF16,
    ``ba_out`` [7, 16] BF16 (b then a), and the conv update of ``conv_state``
    [slots, 1536, 3] and ``windows`` [slots, 7, 1536, 3] (strides [s, 1, 9, 1]) at
    ``state_indices[0]`` / ``replay_indices[0]`` (one int32 / int64 each; a state index
    of ``pad_slot`` skips the conv writes). ``error`` is an int32 word, zero before
    capture, that a timed-out wait sets before trapping. The ReplaySSM verify's
    ``checkpoint`` (by state index) and ``history_*`` rings (by ``ring_indices[0]``) are
    prefetched into L2: a hint, nothing is written. ``tma`` False takes the weights by
    register loads instead of the bulk copy (same values)."""
    if (
        x.dtype != torch.bfloat16
        or tuple(x.shape) != (ROWS, HIDDEN)
        or not x.is_contiguous()
    ):
        raise ValueError(
            f"the GDN front takes BF16 [7, 4096] rows, not {x.dtype} {tuple(x.shape)}"
        )
    if windows.stride()[1:] != (1, 9, 1) or windows.stride(0) < CONV * 9:
        raise ValueError(f"windows {tuple(windows.stride())} are not deduplicated")
    regions = [
        r
        for r in (
            _prefetch(checkpoint, 0),
            _prefetch(history_d, 1),
            _prefetch(history_k, 1),
            _prefetch(history_g, 1),
        )
        if r is not None
    ]
    ba = BaParams(
        input=x.data_ptr(),
        weight=ba_weight.data_ptr(),
        output=ba_out.data_ptr(),
        rows=ROWS,
        input_stride=x.stride(0),
        prefetches=len(regions),
        state_indices=state_indices.data_ptr(),
        ring_indices=ring_indices.data_ptr(),
        state_index64=state_indices.dtype == torch.int64,
        ring_index64=ring_indices.dtype == torch.int64,
        pad_slot=pad_slot,
    )
    for i, region in enumerate(regions):
        ba.prefetch[i] = region
    p = Params(
        x=x.data_ptr(),
        weight=packed.data_ptr(),
        sx=input_scale.data_ptr(),
        sw=weight_scale.data_ptr(),
        output=projected.data_ptr(),
        error=error.data_ptr(),
        timeout=TIMEOUT,
        m=ROWS,
        conv_weight=conv_weight.data_ptr(),
        conv_state=conv_state.data_ptr(),
        windows=windows.data_ptr(),
        conv_output=conv_out.data_ptr(),
        state_indices=state_indices.data_ptr(),
        replay_indices=replay_indices.data_ptr(),
        state_index64=state_indices.dtype == torch.int64,
        replay_index64=replay_indices.dtype == torch.int64,
        pad_slot=pad_slot,
        ba=ba,
    )
    p.state_stride[:] = list(conv_state.stride())
    p.window_stride[:] = [windows.stride(0), 9]
    jit.launch(
        _kernel(x.device.index, tma),
        (CTAS, 1, 1),
        [p],
        torch.cuda.current_stream(x.device).cuda_stream,
        (CLUSTER, 1, 1),
    )
