# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0): python/sglang/kernels/ops/speculative/draft_moe_gemv.py
# (draft_moe_fused_scratch, draft_moe_fused), on branch nvidia/qwen35-agentx-gb300.
"""The single-launch M = 1 bf16 draft MoE (``src/draft_moe.cu``): router GEMV,
softmax top-k with renormalized weights, the routed and gated shared expert, one grid
of one CTA per SM whose CTAs hand over through tagged words in global scratch.

Its ids, weights and output equal the router + two-pass GEMV path of the same kernels
bit for bit; against SGLang's stock draft MoE only the fp32 dot-product order differs.
"""

from __future__ import annotations

import ctypes
from functools import cache
from pathlib import Path
from typing import Optional, Tuple

import torch

from sgl_nvidia_mega.common import jit

SOURCE = Path(__file__).parent / "src" / "draft_moe.cu"
THREADS = 288  # 8 consumer warps and the copy / selection warp
# Ring stages (16 KB each) of the fused kernel; at least topk, since every routed
# down slice holds a stage until the end.
STAGES = 11
# Copies of every tagged handoff word (kFusedReplicas in src/draft_moe.cu).
REPLICAS = 152
TRACE_SLOTS = 32
# Routed pairs a CTA keeps in flight (I = 128: 4, -0.35 us per call on GB300).
WINDOW_SMALL_I = 4


def fused_window(inter: int) -> int:
    return WINDOW_SMALL_I if inter <= 128 else STAGES


def logit_words(experts: int) -> int:
    """E logits and the gate, padded to whole 128-byte lines per replica."""
    return (experts + 1 + 15) // 16 * 16


@cache
def sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def columns(hidden: int, device_index: int) -> int:
    """Output columns per CTA: one CTA per SM."""
    return -(-hidden // sm_count(device_index))


def shared_bytes(hidden: int, inter: int, topk: int, cols: int, stages: int) -> int:
    """DraftMoeFused::kSmem: the ring, the shared down slice, x and the act values."""
    slice_alloc = (cols * inter * 2 + 127) // 128 * 128
    return stages * 4 * hidden + slice_alloc + 2 * hidden + 2 * (topk + 1) * inter


class Params(ctypes.Structure):
    """DraftMoeFusedParams."""

    _fields_ = [
        (name, ctypes.c_void_p)
        for name in (
            "x",
            "router_w",
            "w13",
            "w2",
            "w13_shared",
            "w2_shared",
            "gate_w",
            "logit_buf",
            "act_buf",
            "epoch",
            "topk_weights",
            "topk_ids",
            "out",
            "trace",
        )
    ]


@cache
def _kernel(
    device_index: int,
    hidden: int,
    inter: int,
    experts: int,
    topk: int,
    cols: int,
    stages: int,
    window: int,
    trace: bool,
) -> jit.Kernel:
    major, minor = torch.cuda.get_device_capability(device_index)
    defines = (
        f"-DQ35_DMOE_H={hidden}",
        f"-DQ35_DMOE_I={inter}",
        f"-DQ35_DMOE_E={experts}",
        f"-DQ35_DMOE_TOPK={topk}",
        f"-DQ35_DMOE_COLS={cols}",
        f"-DQ35_DMOE_STAGES={stages}",
        f"-DQ35_DMOE_WINDOW={window}",
        f"-DQ35_DMOE_TRACE={int(trace)}",
    )
    cubin = jit.compile_source(
        SOURCE.read_text(),
        f"q35_draft_moe_h{hidden}_i{inter}_e{experts}_k{topk}_c{cols}_s{stages}_w{window}_t{int(trace)}",
        f"sm_{major}{minor}a",
        # the flags of SGLang's JIT build of this kernel (draft_moe_gemv.py)
        ("-std=c++17", "-default-device", "--use_fast_math", *defines),
    )
    return jit.load(
        cubin,
        "q35_draft_moe_fused",
        (THREADS, 1, 1),
        shared_bytes(hidden, inter, topk, cols, stages),
    )


def _fit_reasons(
    kernel: jit.Kernel, *, grid: int, experts: int, inter: int, topk: int, stages: int
) -> list[str]:
    """DraftMoeFused::run's launch checks: the whole grid co-resident (the CTAs spin
    on each other) and every CTA's static work within the ring."""
    cu = jit.driver()
    (per_sm,) = jit.check(
        cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            kernel.function, THREADS, kernel.shared_memory_bytes
        ),
        "cuOccupancyMaxActiveBlocksPerMultiprocessor",
    )
    reasons = []
    device = torch.cuda.current_device()
    if per_sm < 1 or grid > per_sm * sm_count(device):
        reasons.append(
            f"fused draft MoE grid {grid} does not fit: {per_sm} CTA/SM x "
            f"{sm_count(device)} SMs"
        )
    max_static = ((experts + grid - 1) // grid + 1) // 2 + (inter + grid - 1) // grid
    max_pairs = (inter + grid - 1) // grid + (topk * inter + grid - 1) // grid
    if max_static > stages or max_pairs > 32 or (experts + grid - 1) // grid > 8:
        reasons.append(f"too much work per CTA for grid {grid}")
    return reasons


def build(
    device_index: int,
    *,
    hidden: int,
    inter: int,
    experts: int,
    topk: int,
    stages: int = STAGES,
    window: Optional[int] = None,
) -> list[str]:
    """Compile the kernel for this shape (no compile inside a step); the reasons it
    cannot launch on this device ([] when it can)."""
    cols = columns(hidden, device_index)
    window = fused_window(inter) if window is None else window
    if not 1 <= window <= stages or stages < topk:
        return [f"window {window} / stages {stages} for top-{topk}"]
    with torch.cuda.device(device_index):
        kernel = _kernel(
            device_index, hidden, inter, experts, topk, cols, stages, window, False
        )
        return _fit_reasons(
            kernel,
            grid=-(-hidden // cols),
            experts=experts,
            inter=inter,
            topk=topk,
            stages=stages,
        )


def draft_moe_fused_scratch(
    experts: int, topk: int, inter: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zeroed scratch of draft_moe_fused, reused across calls: tagged logits and
    gate, tagged act values (both replicated), and the call epoch."""
    return (
        torch.zeros(REPLICAS * logit_words(experts), dtype=torch.int64, device=device),
        torch.zeros(REPLICAS * (topk + 1) * inter, dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
    )


def _check(name: str, t: torch.Tensor, shape: tuple, dtype, device, align: int = 0):
    if (
        tuple(t.shape) != shape
        or t.dtype != dtype
        or t.device != device
        or not t.is_contiguous()
        or (align and t.data_ptr() % align)
    ):
        raise ValueError(
            f"draft_moe_fused: {name} {tuple(t.shape)} {t.dtype} {t.device} "
            f"(contiguous {t.is_contiguous()}), expected {shape} {dtype} {device}"
        )


def draft_moe_fused(
    x: torch.Tensor,
    router_w: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_shared: torch.Tensor,
    w2_shared: torch.Tensor,
    shared_gate_w: torch.Tensor,
    scratch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    out: torch.Tensor,
    trace: Optional[torch.Tensor] = None,
    stages: int = STAGES,
    window: Optional[int] = None,
    hop_site: Optional[torch.Tensor] = None,
    hop_token: int = 0,
) -> torch.Tensor:
    """Router top-k + the routed and gated shared expert of one token in one launch.

    x [1, H]; router_w [E, H]; w13 [E, 2I, H] (gate rows, then up rows); w2 [E, H, I];
    w13_shared [2I, H]; w2_shared [H, I]; shared_gate_w [H] (all bf16); scratch from
    draft_moe_fused_scratch; outputs topk_weights fp32 [1, K], topk_ids int32 [1, K]
    (lowest id on ties) and out bf16 [1, H]. trace: int64 [grid * 32] stamps, or None.
    The hop push (hop_site) is not part of this port."""
    if hop_site is not None:
        raise ValueError(
            "draft_moe_fused: the hop push is not ported (TP8 runs it off)"
        )
    hidden, inter = x.shape[-1], w2.shape[-1]
    experts, topk = router_w.shape[0], topk_ids.shape[-1]
    dev = x.device
    bf16 = torch.bfloat16
    _check("x", x, (1, hidden), bf16, dev, 16)
    _check("router_w", router_w, (experts, hidden), bf16, dev, 16)
    _check("w13", w13, (experts, 2 * inter, hidden), bf16, dev, 16)
    _check("w2", w2, (experts, hidden, inter), bf16, dev, 16)
    _check("w13_shared", w13_shared, (2 * inter, hidden), bf16, dev, 16)
    _check("w2_shared", w2_shared, (hidden, inter), bf16, dev, 16)
    _check("shared_gate_w", shared_gate_w, (hidden,), bf16, dev, 16)
    logit_buf, act_buf, epoch = scratch
    _check("logit_buf", logit_buf, (REPLICAS * logit_words(experts),), torch.int64, dev)
    _check("act_buf", act_buf, (REPLICAS * (topk + 1) * inter,), torch.int32, dev, 16)
    _check("epoch", epoch, (1,), torch.int32, dev)
    _check("topk_weights", topk_weights, (1, topk), torch.float32, dev)
    _check("topk_ids", topk_ids, (1, topk), torch.int32, dev)
    _check("out", out, (1, hidden), bf16, dev)
    cols = columns(hidden, dev.index)
    grid = -(-hidden // cols)
    if trace is not None:
        _check("trace", trace, (grid * TRACE_SLOTS,), torch.int64, dev)
    window = fused_window(inter) if window is None else window
    kernel = _kernel(
        dev.index,
        hidden,
        inter,
        experts,
        topk,
        cols,
        stages,
        window,
        trace is not None,
    )
    p = Params(
        x=x.data_ptr(),
        router_w=router_w.data_ptr(),
        w13=w13.data_ptr(),
        w2=w2.data_ptr(),
        w13_shared=w13_shared.data_ptr(),
        w2_shared=w2_shared.data_ptr(),
        gate_w=shared_gate_w.data_ptr(),
        logit_buf=logit_buf.data_ptr(),
        act_buf=act_buf.data_ptr(),
        epoch=epoch.data_ptr(),
        topk_weights=topk_weights.data_ptr(),
        topk_ids=topk_ids.data_ptr(),
        out=out.data_ptr(),
        # the kernel forms its trace row pointer even when it records nothing
        trace=(epoch if trace is None else trace).data_ptr(),
    )
    jit.launch(kernel, (grid, 1, 1), [p], torch.cuda.current_stream(dev).cuda_stream)
    return out
