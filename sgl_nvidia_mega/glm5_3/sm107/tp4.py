# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Rubin's plans for the TP4 kernel units (``glm5_3/ll``): the ``ll.target.Target``
that ``glm5_3/targets.py`` picks on SM107.

Rubin (VR200, SM107, 212 SMs, two dies) is in the SM100 family: the units run the same
kernel sources there, JIT-compiled for the device (``ll.jit``: SGLang targets
``sm_107a``, or ``sm_100f`` when its CUTLASS lacks 10.7) or CuTe DSL compiled for it.
Their plans were measured on GB300; here they are GB300's, re-derived where they
depend on the SM count, and not measured on Rubin:

- ``ll.attention``: GB300's fused-A / q_b row ranges. The 1-16-row forms are SGLang's
  fused-A kernel bit for bit, the split-K form's occupancy gate is queried on the
  device at compile, and every stage ring fits the 227 KiB per-CTA opt-in both GPUs
  expose.
- ``ll.hops``: GB300's caps, in Lamport bytes per rank. The cluster geometry follows
  the SM count (FlashInfer's rule, so a hop stays bit-identical to FlashInfer's
  one-shot on the same GPU).
- ``ll.moe``: the same pieces (FlashInfer's trtllm-gen MoE has SM107 cubins) under
  GB300's shared-expert stream gate.
- ``ll.sparse_mla_fd``: GB300's row caps as upper bounds, cut to the 16-CTA clusters
  the device co-schedules at one CTA per SM (queried at build); a width it cannot
  launch in one wave declines to TRTLLM-Gen instead of stopping the server
  (``Target.measured`` False).
- ``ll.gemm``: ``GEMM_PLAN``. The split-K source is the vendored copy, since
  FlashInfer's explicit split-K tactics serve SM100 / SM103; the vendored kernel
  reserves CuTe DSL's ``sm_100`` per-CTA shared memory (232,448 bytes), which is also
  Rubin's opt-in.
"""

from __future__ import annotations

from typing import Tuple

from sgl_nvidia_mega.glm5_3.ll import gemm_plan
from sgl_nvidia_mega.glm5_3.ll.target import Target

# GB300's GEMM plan without the router's split-K ranges. The kernels are the same
# (TGV: SGLang's CuTe DSL kernel, which has a 10.7 path; split-K: the vendored copy),
# and every split-K tactic still fits one wave of 212 SMs with 16 GPCs assumed (at most
# 96 CTAs against 164). The router keeps SGLang's gate: its split-K logits passed the
# router-vs-cuBLAS bound on Rubin but missed the separate FP64 comparison (error to
# FP64 at most 1.5x cuBLAS's plus 1e-6) at 18 and 20 rows, where Rubin's cuBLAS is
# closer to FP64 than GB300's. Those ranges start at 17 rows; one or two requests at
# 3-6 tokens never reach them. The crossovers against SGLang's GEMM are GB300
# measurements.
GEMM_PLAN: Tuple[gemm_plan.Entry, ...] = tuple(
    e for e in gemm_plan.PLAN if e.role != "router"
)

RUBIN = Target(
    name="Rubin (VR200)",
    sm=(10, 7),
    sms=212,
    # not published; 16 (8 per die) is the conservative assumption for the split-K
    # one-wave check
    gpcs=16,
    gemm_plan=GEMM_PLAN,
    flashinfer_splitk=False,
    measured=False,
    notes={
        "ll.attention": "GB300's fused-A / q_b row ranges",
        "ll.gemm": "GB300's GEMM ranges, split-K checked for one wave of this GPU",
        "ll.hops": "GB300's caps, cluster geometry for the device's SM count",
        "ll.moe": "GB300's shared-expert stream gate",
    },
)
