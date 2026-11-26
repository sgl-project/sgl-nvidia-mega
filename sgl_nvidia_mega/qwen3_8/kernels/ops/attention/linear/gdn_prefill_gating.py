# SPDX-License-Identifier: Apache-2.0
# From SGLang python/sglang/kernels/ops/attention/linear/fused_gdn_gating.py
# .
"""The GDN gates ``g = -exp(A_log) * softplus(a + dt_bias)`` and ``beta = sigmoid(b)``
with the extend-sized 2D launch: one program per 16 tokens x all heads.

Byte-identical to the source: ``fused_gdn_gating_2d_kernel``,
``_gating_2d_cfg`` and ``fused_gdn_gating``. Changed: SGLANG_OPT_GDN_PREFILL_GATING_2D
is the ``gating_2d`` argument (``_gating_2d_cfg``'s ``enabled``); the unchanged 1D
kernel is SGLang's ``fused_gdn_gating_kernel``."""

import functools
import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from sglang.kernels.ops.attention.linear.fused_gdn_gating import (
    fused_gdn_gating_kernel,
)

logger = logging.getLogger(__name__)

SGLANG_READS = {
    "kernels/ops/attention/linear/fused_gdn_gating.py": ("fused_gdn_gating_kernel",),
}

# Extend-sized batches take one program per BT tokens x all heads instead of one
# 32-thread program per (token, 8 heads). Same per-element math.
GATING_2D_MIN_BATCH = 512
GATING_2D_CFG = (16, 2)  # (BT, num_warps)


@triton.jit(do_not_specialize=["T"])
def fused_gdn_gating_2d_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    T,
    stride_a,
    stride_b,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BT: tl.constexpr,
    BH: tl.constexpr,
):
    rows = tl.program_id(0) * BT + tl.arange(0, BT)
    heads = tl.arange(0, BH)
    hmask = heads < NUM_HEADS
    mask = (rows < T)[:, None] & hmask[None, :]
    rows64 = rows.to(tl.int64)[:, None]
    blk_A_log = tl.load(A_log + heads, mask=hmask)
    blk_bias = tl.load(dt_bias + heads, mask=hmask)
    blk_a = tl.load(a + rows64 * stride_a + heads[None, :], mask=mask)
    blk_b = tl.load(b + rows64 * stride_b + heads[None, :], mask=mask)
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)[None, :]
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32))[None, :] * softplus_x
    off = rows64 * NUM_HEADS + heads[None, :]
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(beta_output + off, blk_beta_output.to(b.dtype.element_ty), mask=mask)


def _gating_2d_cfg(
    batch: int, launch_cfg: Optional[tuple], enabled: bool = False
) -> Optional[tuple]:
    if launch_cfg is not None:
        return launch_cfg
    if enabled and batch >= GATING_2D_MIN_BATCH:
        _log_gating_2d_once()
        return GATING_2D_CFG
    return None


@functools.lru_cache(maxsize=1)
def _log_gating_2d_once() -> None:
    logger.info(
        "GDN gating: 2D launch %s for batches >= %d (SGLANG_OPT_GDN_PREFILL_GATING_2D)",
        GATING_2D_CFG,
        GATING_2D_MIN_BATCH,
    )


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    launch_cfg: Optional[tuple] = None,
    gating_2d: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """launch_cfg: (BT, num_warps) of the 2D launch; None picks it by batch size
    (the 2D launch from GATING_2D_MIN_BATCH rows when gating_2d)."""
    batch, num_heads = a.shape
    seq_len = 1
    stride_a = a.stride(0)
    stride_b = b.stride(0)
    grid = (batch, seq_len, triton.cdiv(num_heads, 8))
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, batch, num_heads, dtype=torch.float32, device=b.device)
    cfg_2d = _gating_2d_cfg(batch, launch_cfg, gating_2d)
    if cfg_2d is not None:
        bt, num_warps = cfg_2d
        fused_gdn_gating_2d_kernel[(triton.cdiv(batch, bt),)](
            g,
            beta_output,
            A_log,
            a,
            b,
            dt_bias,
            batch,
            stride_a,
            stride_b,
            num_heads,
            beta,
            threshold,
            BT=bt,
            BH=triton.next_power_of_2(num_heads),
            num_warps=num_warps,
        )
        return g, beta_output
    fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        stride_a,
        stride_b,
        num_heads,
        beta,
        threshold,
        8,
        num_warps=1,
    )
    return g, beta_output
