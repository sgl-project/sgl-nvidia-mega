# SPDX-License-Identifier: Apache-2.0
# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/l2norm.py
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# From SGLang python/sglang/kernels/ops/attention/fla/l2norm.py.
"""l2norm and the FlashInfer GDN prefill's q / k / v preparation, with the
extend-sized launch (64 rows a program) and q | k stored back to back normalized by
one launch.

Byte-identical to the source: ``_l2norm_launch_cfg``, ``l2norm_fwd``,
``_qk_adjacent`` and ``gdn_prefill_qkv_prepare_fwd``. Changed: the environment
reads are arguments, SGLANG_OPT_GDN_PREFILL_L2NORM_RETILE the ``retile`` of all
three launchers and SGLANG_OPT_GDN_PREFILL_QK_L2NORM_ONE_LAUNCH
``gdn_prefill_qkv_prepare_fwd``'s ``qk_one_launch``; the unchanged kernels are
SGLang's."""

import functools
import logging
from typing import Optional

import torch
import triton
from sglang.kernels.ops.attention.fla.l2norm import (
    gdn_prefill_qkv_prepare_kernel,
    l2norm_fwd_kernel,
    l2norm_fwd_kernel1,
)

logger = logging.getLogger(__name__)

SGLANG_READS = {
    "kernels/ops/attention/fla/l2norm.py": (
        "gdn_prefill_qkv_prepare_kernel",
        "l2norm_fwd_kernel",
        "l2norm_fwd_kernel1",
    ),
}

# 64 rows / 8 warps per program for extend-sized 128-wide inputs instead of 16 rows.
# Every row keeps the stock 16-lane x 8-element reduction, so only the number of
# programs changes.
L2NORM_RETILE_MIN_ROWS = 16384
L2NORM_RETILE_CFG = (64, 8)  # (BT, num_warps)


def _l2norm_launch_cfg(
    T: int, D: int, launch_cfg: Optional[tuple], retile: bool = False
) -> tuple:
    if launch_cfg is not None:
        return launch_cfg
    if retile and D == 128 and T >= L2NORM_RETILE_MIN_ROWS:
        _log_l2norm_retile_once()
        return L2NORM_RETILE_CFG
    return 16, 8


@functools.lru_cache(maxsize=1)
def _log_l2norm_retile_once() -> None:
    logger.info(
        "l2norm: launch %s for >= %d rows of 128 (SGLANG_OPT_GDN_PREFILL_L2NORM_RETILE)",
        L2NORM_RETILE_CFG,
        L2NORM_RETILE_MIN_ROWS,
    )


def l2norm_fwd(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: Optional[torch.dtype] = None,
    launch_cfg: Optional[tuple] = None,
    retile: bool = False,
):
    """launch_cfg: (BT, num_warps) for D <= 512; None picks it by row count (64
    rows from L2NORM_RETILE_MIN_ROWS rows of 128 when retile)."""
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])
    # allocate output
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]
    # rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    if D <= 512:
        bt, num_warps = _l2norm_launch_cfg(T, D, launch_cfg, retile)

        def grid(meta):
            return (triton.cdiv(T, meta["BT"]),)

        l2norm_fwd_kernel[grid](
            x,
            y,
            eps,
            T=T,
            D=D,
            BD=BD,
            BT=bt,
            num_warps=num_warps,
            num_stages=3,
        )
    else:
        l2norm_fwd_kernel1[(T,)](
            x,
            y,
            eps=eps,
            D=D,
            BD=BD,
            num_warps=8,
            num_stages=3,
        )

    return y.view(x_shape_og)


def _qk_adjacent(q: torch.Tensor, k: torch.Tensor) -> bool:
    """Contiguous k starts where contiguous q ends, in the same storage."""
    return (
        q.numel() > 0
        and k.data_ptr() == q.data_ptr() + q.numel() * q.element_size()
        and q.untyped_storage().data_ptr() == k.untyped_storage().data_ptr()
    )


def gdn_prefill_qkv_prepare_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    eps: float = 1e-6,
    *,
    retile: bool = False,
    qk_one_launch: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare Q/K/V for FlashInfer, materializing only strided inputs. retile: the
    l2norm launches' extend-sized tile; qk_one_launch: q and k stored back to back are
    normalized by one launch."""
    if q.ndim != 3 or k.shape != q.shape or v.ndim != 3:
        raise ValueError(
            "GDN fused prepare requires equal Q/K [T, Hqk, D] shapes and "
            "V [T, Hv, D], got "
            f"{q.shape=}, {k.shape=}, {v.shape=}"
        )
    if v.shape[0] != q.shape[0] or v.shape[2] != q.shape[2]:
        raise ValueError(
            "GDN fused prepare requires common token and head-dim axes, got "
            f"{q.shape=}, {v.shape=}"
        )
    if q.device != k.device or q.device != v.device:
        raise ValueError("GDN fused prepare requires Q/K/V on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("GDN fused prepare requires equal Q/K/V dtypes")

    T, H_QK, D = q.shape
    H_V = v.shape[1]
    if D > 512:
        raise ValueError(f"GDN fused prepare supports head dim <= 512, got {D}")
    if q.is_contiguous() and k.is_contiguous() and v.is_contiguous():
        if qk_one_launch and _qk_adjacent(q, k):
            # Rows are normalized independently: one launch over q's and k's rows.
            qk = torch.as_strided(q, (2 * T * H_QK, D), (D, 1))
            y = l2norm_fwd(qk, eps, retile=retile).view(-1)
            n = q.numel()
            return y[:n].view(q.shape), y[n:].view(k.shape), v
        return (
            l2norm_fwd(q, eps, retile=retile),
            l2norm_fwd(k, eps, retile=retile),
            v,
        )

    q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    v_out = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    BT = 16
    BD = triton.next_power_of_2(D)
    grid = (triton.cdiv(T, BT), H_QK + H_V)
    gdn_prefill_qkv_prepare_kernel[grid](
        q,
        k,
        v,
        q_out,
        k_out,
        v_out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        T=T,
        H_QK=H_QK,
        H_V=H_V,
        D=D,
        BT=BT,
        BD=BD,
        num_warps=4,
        num_stages=2,
    )
    return (
        l2norm_fwd(q_out, eps, retile=retile),
        l2norm_fwd(k_out, eps, retile=retile),
        v_out,
    )
