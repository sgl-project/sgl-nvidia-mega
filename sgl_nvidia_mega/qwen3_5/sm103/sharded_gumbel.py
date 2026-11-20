# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/speculative/sharded_verify.py (_shard_key_kernel, shard_keys)
"""Gumbel-max keys for one rank's vocabulary shard.

The generic runtime owns the shard descriptor, cross-rank key exchange, greedy path,
and stock full-logits fallback.  This module owns only the numerical opt-in: for each
row and local vocabulary range it writes the best 64-bit key after temperature scaling
and deterministic Gumbel noise.  Noise is keyed by ``(seed, global token id)``, so a
row has the same winner whether its vocabulary is evaluated whole, in TP shards, or in
multiple programs.

The key layout is the one consumed by SGLang's generic sharded-verify reducer::

    ordered_fp32_bits << 32 | (0xffffffff - global_token_id)

Thus an int64 maximum chooses the largest sampled value and breaks exact ties toward
the smaller token id.  This is a direct Triton migration of the v2 kernel, including
its RNG stream and FP32 operation order.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["build", "gumbel_shard_keys", "key_splits"]

# Key of (-inf, token id 0xffffffff), below every real column's key.  Keep this
# signed representation bit-for-bit identical to runtime-v2's generic reducer.
_KEY_FLOOR = -2139095041 << 32
_KEY_BLOCK = 2048


@triton.jit
def _gumbel_shard_key_kernel(
    logits_ptr,
    stride_row,
    temperature_ptr,
    seed_ptr,
    keys_ptr,
    vocab_start,
    valid,
    vocab_size,
    span,
    KEY_FLOOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Program (row, part) covers [part * span, (part + 1) * span) of this shard.
    row = tl.program_id(0).to(tl.int64)
    part = tl.program_id(1)
    lo = part * span
    hi = tl.minimum(lo + span, valid)
    best = tl.full([BLOCK], KEY_FLOOR, tl.int64)
    temperature = tl.load(temperature_ptr + row)
    inverse_temperature = 1.0 / tl.maximum(temperature, 1e-20)
    seed = tl.load(seed_ptr + row)
    for offset in range(lo, hi, BLOCK):
        columns = offset + tl.arange(0, BLOCK)
        token_ids = vocab_start + columns
        mask = (columns < hi) & (token_ids < vocab_size)
        values = tl.load(
            logits_ptr + row * stride_row + columns,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        # Mixed batches retain runtime-v2's behavior: T <= 0 rows use their plain
        # logits while positive-temperature rows consume the deterministic stream.
        if temperature > 0.0:
            uniform = tl.rand(seed, token_ids)
            values = values * inverse_temperature - tl.log(-tl.log(uniform))
        # NaNs never beat a finite value.  Adding +0 canonicalizes -0 for argmax ties.
        not_nan = values == values  # noqa: PLR0124 -- intentional IEEE NaN test
        values = tl.where(not_nan, values, float("-inf")) + 0.0
        bits = values.to(tl.int32, bitcast=True)
        ordered = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(tl.int64)
        keys = (ordered << 32) | (4294967295 - token_ids.to(tl.int64))
        keys = tl.where(mask, keys, KEY_FLOOR)
        best = tl.maximum(best, keys)
    tl.store(
        keys_ptr + row * tl.num_programs(1) + part,
        tl.max(best, axis=0),
    )


def key_splits(valid: int) -> int:
    """Programs per row, padded even for the runtime's multimem chunk layout."""
    if valid <= 0:
        raise ValueError(f"valid vocabulary columns must be positive, got {valid}")
    count = -(-valid // _KEY_BLOCK)
    return count + (count & 1)


def _launch(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    seeds: torch.Tensor,
    out: torch.Tensor,
    *,
    vocab_start: int,
    valid: int,
    vocab_size: int,
    splits: int,
) -> None:
    rows = logits.shape[0]
    if rows == 0:
        return
    _gumbel_shard_key_kernel[(rows, splits)](
        logits,
        logits.stride(0),
        temperatures,
        seeds,
        out,
        vocab_start,
        valid,
        vocab_size,
        -(-valid // splits),
        KEY_FLOOR=_KEY_FLOOR,
        BLOCK=_KEY_BLOCK,
        num_warps=8,
    )


def build(
    device_index: int,
    *,
    columns: int,
    vocab_start: int,
    valid: int,
    vocab_size: int,
    splits: int,
) -> None:
    """Compile both dtypes for the rank's exact production shard metadata."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("compiling sharded Gumbel inside CUDA-graph capture")
    if not 0 < valid <= columns:
        raise ValueError(
            f"valid vocabulary columns must be in [1, {columns}], got {valid}"
        )
    if vocab_start < 0 or vocab_start + valid > vocab_size:
        raise ValueError(
            f"invalid vocabulary shard: start={vocab_start}, valid={valid}, "
            f"size={vocab_size}"
        )
    if splits <= 0:
        raise ValueError(f"splits must be positive, got {splits}")
    with torch.cuda.device(device_index):
        temperatures = torch.ones(2, dtype=torch.float32, device="cuda")
        seeds = torch.arange(2, dtype=torch.int64, device="cuda")
        shape = (2, splits) if splits > 1 else (2,)
        keys = torch.empty(shape, dtype=torch.int64, device="cuda")
        for dtype in (torch.bfloat16, torch.float32):
            logits = torch.empty((2, columns), dtype=dtype, device="cuda")
            _launch(
                logits,
                temperatures,
                seeds,
                keys,
                vocab_start=vocab_start,
                valid=valid,
                vocab_size=vocab_size,
                splits=splits,
            )
        torch.cuda.synchronize(device_index)


def gumbel_shard_keys(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    seeds: torch.Tensor,
    *,
    vocab_start: int,
    valid: int,
    vocab_size: int,
    splits: int = 1,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return this rank's best Gumbel key for every row and split.

    ``logits`` is contiguous by column and may contain padding after ``valid``.
    ``temperatures`` is FP32 and ``seeds`` is int64, both one value per row.  Positive
    temperatures sample ``logits / T``; nonpositive temperatures retain the plain
    argmax key, which supports mixed graph batches without another launch.  Columns
    at or beyond ``valid`` or ``vocab_size`` are masked even when their padding holds a
    large value.  Pass a preallocated ``out`` to avoid an allocation in a captured
    graph; call :func:`build` before the first capture.
    """
    if not logits.is_cuda or logits.dim() != 2 or logits.stride(1) != 1:
        raise ValueError(
            "sharded Gumbel logits must be a CUDA [rows, columns] tensor with "
            "unit column stride"
        )
    if logits.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(
            f"sharded Gumbel logits must be BF16 or FP32, got {logits.dtype}"
        )
    rows, columns = logits.shape
    for name, tensor, dtype in (
        ("temperatures", temperatures, torch.float32),
        ("seeds", seeds, torch.int64),
    ):
        if (
            tensor.device != logits.device
            or tensor.dtype != dtype
            or tuple(tensor.shape) != (rows,)
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"sharded Gumbel {name} must be contiguous {dtype} [{rows}] "
                f"on {logits.device}, got {tensor.dtype} {tuple(tensor.shape)} "
                f"on {tensor.device}"
            )
    if not 0 < valid <= columns:
        raise ValueError(
            f"valid vocabulary columns must be in [1, {columns}], got {valid}"
        )
    if vocab_start < 0 or vocab_size <= vocab_start:
        raise ValueError(
            f"invalid vocabulary range: start={vocab_start}, size={vocab_size}"
        )
    if splits <= 0:
        raise ValueError(f"splits must be positive, got {splits}")
    shape = (rows, splits) if splits > 1 else (rows,)
    if out is None:
        out = torch.empty(shape, dtype=torch.int64, device=logits.device)
    elif (
        out.device != logits.device
        or out.dtype != torch.int64
        or tuple(out.shape) != shape
        or not out.is_contiguous()
    ):
        raise ValueError(
            f"sharded Gumbel out must be contiguous int64 {shape} on {logits.device}"
        )
    _launch(
        logits,
        temperatures,
        seeds,
        out,
        vocab_start=vocab_start,
        valid=valid,
        vocab_size=vocab_size,
        splits=splits,
    )
    return out
