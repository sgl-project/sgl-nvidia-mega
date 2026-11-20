# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Which kernel serves each registered small-M BF16 GEMM, per role, per-rank weight
shape and row count, fixed at build (``ll.gemm``).

The decode-side BF16 GEMMs that SGLang leaves on cuBLAS launch without programmatic
dependent launch (PDL), so each one waits for its producer to drain and delays its
consumer. The ranges below move them to a PDL kernel at the row counts where a GB300
crossover bench measured a win with the GEMM between its model neighbours (CUDA-graph
replay, HBM-cold weights; a range is listed only where it beat SGLang's kernel at every
measured row in each of three runs):

- ``splitk``: FlashInfer's cluster split-K BF16 GEMM, with an explicit tactic
  ``(mma_m, mma_n, split_k, ab_stages)``; its weight warp streams the weight before
  ``griddepcontrol.wait``. FP32 output for the router.
- ``tgv``: SGLang's CuTe DSL TGV GEMM (``cutedsl_bf16_gemm``), BF16 output.

Rows a role's ranges leave out keep SGLang's GEMM for that module, bit for bit. The plan
applies only on the GPU it was measured on (GB300: SM103 with 152 SMs); anywhere else
every role keeps SGLang's GEMM.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import msgspec

SPLITK = "splitk"
TGV = "tgv"

# The GPU the plan was measured on: (compute capability, SMs).
MEASURED_DEVICE = ((10, 3), 152)
# Largest row count any range serves.
MAX_M = 96
# A cluster of c CTAs needs c free SMs in one GPC, so up to c - 1 SMs per GPC can sit
# idle: a split-K grid fits one wave with at most SMs - (split_k - 1) * GPCS CTAs. Every
# split-K CTA reserves the full shared memory (one CTA per SM), so a second wave would
# serialize a nearly empty tail and miss the window in which CTAs load weights before
# their wait. 8 GPCs is the B200 / B300 layout (assumed, not queried).
GPCS = 8
# K extent of one split-K tile.
SPLITK_CTA_K = 128

# role -> the output dtype it returns
ROLE_DTYPES = {
    "wk_weights_proj": "bfloat16",
    "o_proj": "bfloat16",
    "router": "float32",
    "dense_down": "bfloat16",
    "eh_proj": "bfloat16",
    "lm_head": "bfloat16",
    "q_b": "bfloat16",
}
ROLES = tuple(ROLE_DTYPES)


class Range(msgspec.Struct, frozen=True):
    """Rows [m_lo, m_hi] of a role's weight shape and the kernel serving them."""

    m_lo: int
    m_hi: int
    kernel: str  # SPLITK or TGV
    # SPLITK: (mma_m, mma_n, split_k, ab_stages); mma_m tiles the output features
    # (public N) and mma_n the rows (public M).
    tactic: Tuple[int, ...] = ()


class Entry(msgspec.Struct, frozen=True):
    """A role's measured ranges at one per-rank weight shape (N, K)."""

    role: str
    shape: Tuple[int, int]
    ranges: Tuple[Range, ...]
    # False: measured, but not served unless the role is listed in
    # SGLANG_NVIDIA_MEGAKERNEL_GLM_GEMM_ROLES (deferred, plan.md F.1)
    preferred: bool = True


def _sk(m_lo: int, m_hi: int, *tactic: int) -> Range:
    return Range(m_lo=m_lo, m_hi=m_hi, kernel=SPLITK, tactic=tactic)


def _tgv(m_lo: int, m_hi: int) -> Range:
    return Range(m_lo=m_lo, m_hi=m_hi, kernel=TGV)


# GLM-5.2 per-rank shapes (hidden 6144). Measured by the crossover bench of
# notes/branches/04-pdl-splitk-gemms.md sections 11-12 (jobs 4485639, 4485738,
# 4485957; wk beside the indexer's wq_b: 4486889) and kept by branch 50 item 14.
PLAN: Tuple[Entry, ...] = (
    # Router logits, FP32 (256 experts). SGLang's tiny GEMM keeps the rows up to the
    # gate's tiny_router_gemm_max_tokens (16); from 17 rows split-K beats cuBLAS by
    # 1.8-2.0 us at 18-48 rows and 1.5 us at 96. Above 48 rows 64x8/k4 reaches 128 CTAs
    # and loses to 64x16/k4.
    Entry(
        "router",
        (256, 6144),
        (_sk(17, 48, 64, 8, 4, 12), _sk(49, 96, 64, 16, 4, 10)),
    ),
    # The indexer's fused wk | weights_proj (128 + 32 outputs), on the main stream
    # beside wq_b (TGV) on the alternate stream. cuBLAS split-K + reduce is the longer
    # leg; split-K with few CTAs wins 0.4-2.2 us at 1-64 rows. Above 64 rows SGLang
    # runs wq_b on cuBLAS, wq_b becomes the longer leg and no tactic wins.
    Entry(
        "wk_weights_proj",
        (160, 6144),
        (
            _sk(1, 16, 64, 8, 4, 12),
            _sk(17, 32, 64, 16, 4, 10),
            _sk(33, 64, 64, 32, 4, 8),
        ),
    ),
    # TP4 o_proj: SGLang runs TGV except at 48 and 64-96 rows (cuBLAS); TGV wins 2.0 us
    # at 48 rows, 1.1 at 72. No split-K tactic beats TGV at any row. At 96 rows 04
    # measured +0.5 us, the 30dc2f9430 image -0.67 us (job 4768671): capped at 72.
    Entry("o_proj", (6144, 4096), (_tgv(1, 72),)),
    # TP4 dense-layer down projection: TGV beats cuBLAS by 0.07-0.46 us at 1-48 rows;
    # cuBLAS wins from 60 rows.
    Entry("dense_down", (6144, 3072), (_tgv(1, 48),)),
    # The NextN eh_proj replicated (2H -> H): TGV wins 1.7-4.5 us at every row.
    Entry("eh_proj", (6144, 12288), (_tgv(1, 96),)),
    # The TP4 row-parallel eh_proj shard (P6a) has TP4 dense_down's shape; the stack
    # (sgl-mega-306ccf6) served it on dense_down's TGV range. Not timed in its own
    # context (behind the fused eh-norm window kernel).
    Entry("eh_proj", (6144, 3072), (_tgv(1, 48),)),
    # The TP4 LM head shard (verify, draft decode, draft extend): TGV wins 2.8-5.7 us at
    # 3-48 rows (04); the sgl-mega stack served it at 1-60. The logits processors of the
    # covered verify step and the fused draft run it (LmHeadMixin).
    Entry("lm_head", (38720, 6144), (_tgv(1, 60),)),
    # Deferred (measured, not served by default):
    # TP8 shapes: the TP8 families own these sections at TP8.
    Entry("o_proj", (6144, 2048), (_tgv(1, 96),), preferred=False),
    Entry("q_b", (2048, 2048), (_tgv(17, 96),), preferred=False),
)

# Named tolerances of the registered kernels against cuBLAS on the same GPU and inputs
# (test/kernels/test_glm5_3_ll_gemm.py, every preferred role at every served row, random
# and checkpoint weights). Split-K and TGV compute the same FP32 dot products in another
# summation order and round once; FlashInfer's split-K (fi-1) and the vendored copy are
# bitwise equal. Bounds are about 1.7x and 2.4x the maxima measured on GB300 (jobs
# 4768671 / 4768843: 3.5e-4 and 2.1e-6).
TOLERANCES: Dict[str, float] = {
    # ||y - y_cublas||_F / ||y_cublas||_F, BF16 outputs (split-K and TGV)
    "ll.gemm.bf16.rel": 6e-4,
    # the same for the FP32 router logits (split-K)
    "ll.gemm.router.rel": 5e-6,
}


def entry(role: str, shape: Tuple[int, int]) -> Optional[Entry]:
    """The plan entry of ``role`` at per-rank weight ``shape``, or None."""
    for e in PLAN:
        if e.role == role and e.shape == tuple(shape):
            return e
    return None


def range_at(e: Entry, m: int) -> Optional[Range]:
    for r in e.ranges:
        if r.m_lo <= m <= min(r.m_hi, MAX_M):
            return r
    return None


def splitk_ctas(tactic: Tuple[int, ...], m: int, n: int) -> int:
    """CTAs of a split-K grid: mma_m tiles N, mma_n tiles M, split_k per tile."""
    mma_m, mma_n, split_k, _ = tactic
    return -(-n // mma_m) * -(-m // mma_n) * split_k


def splitk_fits_one_wave(tactic: Tuple[int, ...], m: int, n: int, sms: int) -> bool:
    return splitk_ctas(tactic, m, n) <= sms - (tactic[2] - 1) * GPCS


def kernel_name(r: Range) -> str:
    if r.kernel == TGV:
        return TGV
    mma_m, mma_n, split_k, stages = r.tactic
    return f"{mma_m}x{mma_n}/k{split_k}/s{stages}"


def m_ranges(ms) -> str:
    """(1, 2, 3, 5) -> "1-3,5"."""
    ms = sorted(ms)
    if not ms:
        return ""
    out, start, prev = [], ms[0], ms[0]
    for m in ms[1:]:
        if m != prev + 1:
            out.append((start, prev))
            start = m
        prev = m
    out.append((start, prev))
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in out)
