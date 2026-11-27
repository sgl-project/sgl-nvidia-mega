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

import ctypes
import logging
from pathlib import Path

import torch
import triton
import triton.language as tl

__all__ = [
    "build",
    "cuda_supported",
    "fast_supported",
    "gumbel_shard_keys",
    "gumbel_shard_keys_cuda",
    "gumbel_shard_keys_fast",
    "key_splits",
]

logger = logging.getLogger(__name__)

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


@triton.jit
def _gumbel_shard_key_fast_kernel(
    logits_ptr,
    stride_row,
    temperature_ptr,
    seed_ptr,
    keys_ptr,
    greedy_ptr,
    vocab_start,
    valid,
    vocab_size,
    span,
    KEY_FLOOR: tl.constexpr,
    BLOCK: tl.constexpr,
    GREEDY: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    part = tl.program_id(1)
    lo = part * span
    hi = tl.minimum(lo + span, valid)
    best = tl.full([BLOCK], KEY_FLOOR, tl.int64)
    plain = tl.full([BLOCK], KEY_FLOOR, tl.int64)
    temperature = tl.load(temperature_ptr + row)
    inverse_temperature = 1.0 / tl.maximum(temperature, 1e-20)
    seed = tl.load(seed_ptr + row)
    quads = tl.arange(0, BLOCK // 4)
    for offset in range(lo, hi, BLOCK):
        columns = offset + tl.arange(0, BLOCK)
        token_ids = vocab_start + columns
        mask = (columns < hi) & (token_ids < vocab_size)
        values = tl.load(
            logits_ptr + row * stride_row + columns,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        low = 4294967295 - token_ids.to(tl.int64)
        if GREEDY:
            # sharded_verify._shard_key_kernel: NaN never wins, -0.0 ties +0.0
            x = tl.where(values == values, values, float("-inf")) + 0.0  # noqa: PLR0124
            bits = x.to(tl.int32, bitcast=True)
            ordered = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(tl.int64)
            plain = tl.maximum(plain, tl.where(mask, (ordered << 32) | low, KEY_FLOOR))
        if temperature > 0.0:
            u0, u1, u2, u3 = tl.rand4x(seed, (vocab_start + offset) // 4 + quads)
            uniform = tl.reshape(tl.join(tl.join(u0, u1), tl.join(u2, u3)), [BLOCK])
            uniform = tl.minimum(tl.maximum(uniform, 1e-12), 0.99999994)
            values = values * inverse_temperature - tl.log(-tl.log(uniform))
        not_nan = values == values  # noqa: PLR0124 -- intentional IEEE NaN test
        values = tl.where(not_nan, values, float("-inf")) + 0.0
        bits = values.to(tl.int32, bitcast=True)
        ordered = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(tl.int64)
        keys = (ordered << 32) | low
        keys = tl.where(mask, keys, KEY_FLOOR)
        best = tl.maximum(best, keys)
    tl.store(keys_ptr + row * tl.num_programs(1) + part, tl.max(best, axis=0))
    if GREEDY:
        tl.store(greedy_ptr + row * tl.num_programs(1) + part, tl.max(plain, axis=0))


_FAST_BLOCK = 1024


def fast_supported(*, vocab_start: int, valid: int, splits: int) -> bool:
    """Whether every tile of the fast kernel starts on a multiple of 4 global ids."""
    span = -(-valid // splits)
    return vocab_start % 4 == 0 and span % 4 == 0


def _launch_fast(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    seeds: torch.Tensor,
    out: torch.Tensor,
    greedy_out: torch.Tensor | None,
    *,
    vocab_start: int,
    valid: int,
    vocab_size: int,
    splits: int,
) -> None:
    rows = logits.shape[0]
    if rows == 0:
        return
    _gumbel_shard_key_fast_kernel[(rows, splits)](
        logits,
        logits.stride(0),
        temperatures,
        seeds,
        out,
        out if greedy_out is None else greedy_out,
        vocab_start,
        valid,
        vocab_size,
        -(-valid // splits),
        KEY_FLOOR=_KEY_FLOOR,
        BLOCK=_FAST_BLOCK,
        GREEDY=greedy_out is not None,
        num_warps=4,
    )


def gumbel_shard_keys_fast(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    seeds: torch.Tensor,
    *,
    vocab_start: int,
    valid: int,
    vocab_size: int,
    splits: int,
    out: torch.Tensor | None = None,
    greedy_out: torch.Tensor | None = None,
    pre: tuple[torch.Tensor, int, torch.Tensor] | None = None,
) -> torch.Tensor:
    if not fast_supported(vocab_start=vocab_start, valid=valid, splits=splits):
        raise ValueError("fast sharded Gumbel needs tiles on multiples of 4 global ids")
    rows = logits.shape[0]
    shape = (rows, splits) if splits > 1 else (rows,)
    if out is None:
        out = torch.empty(shape, dtype=torch.int64, device=logits.device)
    if greedy_out is not None and (
        greedy_out.dtype != torch.int64
        or tuple(greedy_out.shape) != shape
        or not greedy_out.is_contiguous()
    ):
        raise ValueError(
            f"fast sharded Gumbel greedy_out must be contiguous int64 {shape}"
        )
    if rows and cuda_supported(
        logits,
        vocab_start=vocab_start,
        valid=valid,
        splits=splits,
        greedy=greedy_out is not None,
        pre=pre is not None,
    ):
        gumbel_shard_keys_cuda(
            logits,
            temperatures,
            seeds,
            out,
            greedy_out,
            vocab_start=vocab_start,
            valid=valid,
            vocab_size=vocab_size,
            splits=splits,
            pre=pre,
        )
    else:
        _launch_fast(
            logits,
            temperatures,
            seeds,
            out,
            greedy_out,
            vocab_start=vocab_start,
            valid=valid,
            vocab_size=vocab_size,
            splits=splits,
        )
    return out


_CUDA_SRC = Path(__file__).parent / "head" / "src" / "gumbel_keys.cu"
_CUDA_THREADS = 128
_CUDA_LOGGED: set = set()
_CUDA_KERNELS: dict = {}


class _GumbelKeysParams(ctypes.Structure):
    _fields_ = [
        ("logits", ctypes.c_void_p),
        ("stride", ctypes.c_int64),
        ("temperatures", ctypes.c_void_p),
        ("seeds", ctypes.c_void_p),
        ("keys", ctypes.c_void_p),
        ("greedy", ctypes.c_void_p),
        ("values", ctypes.c_void_p),
        ("vocab_start", ctypes.c_int32),
        ("valid", ctypes.c_int32),
        ("vocab_size", ctypes.c_int32),
        ("span", ctypes.c_int32),
        ("pre", ctypes.c_void_p),
        ("pre_ctas", ctypes.c_int32),
        ("epoch", ctypes.c_void_p),
    ]


def _cuda_kernel(
    device_index: int,
    fp32: bool,
    greedy: bool,
    debug: bool = False,
    threads: int = _CUDA_THREADS,
    pre: bool = False,
):
    key = (device_index, fp32, greedy, debug, threads, pre)
    kernel = _CUDA_KERNELS.get(key)
    if kernel is None:
        kernel = _CUDA_KERNELS[key] = _cuda_compile(*key)
    return kernel


def _cuda_compile(
    device_index: int, fp32: bool, greedy: bool, debug: bool, threads: int, pre: bool
):
    from sgl_nvidia_mega.common import jit

    device = torch.device("cuda", device_index)
    torch.empty(1, device=device)
    major, minor = torch.cuda.get_device_capability(device)
    cubin = jit.compile_source(
        _CUDA_SRC.read_text(),
        f"q35_gumbel_keys_t{threads}_f{int(fp32)}_g{int(greedy)}_d{int(debug)}_p{int(pre)}",
        f"sm_{major}{minor}a" if major >= 9 else f"sm_{major}{minor}",
        (
            "-std=c++17",
            # libdevice logf as Triton links it; the other fp32 ops are explicit PTX
            "--ftz=true",
            f"-DGK_THREADS={threads}",
            f"-DGK_FP32={int(fp32)}",
            f"-DGK_GREEDY={int(greedy)}",
            f"-DGK_DEBUG={int(debug)}",
            f"-DGK_PRE={int(pre)}",
        ),
    )
    return jit.load(cubin, "gumbel_keys_fast", (threads, 1, 1), 0)


def cuda_supported(
    logits: torch.Tensor,
    *,
    vocab_start: int,
    valid: int,
    splits: int,
    greedy: bool = False,
    pre: bool = False,
) -> bool:
    """Whether the CUDA launch serves this call (else the Triton fast kernel runs)."""
    quad = 8 if logits.dtype == torch.bfloat16 else 16
    return bool(
        logits.is_cuda
        and logits.dim() == 2
        and logits.dtype in (torch.bfloat16, torch.float32)
        and logits.stride(1) == 1
        and logits.shape[1] % 4 == 0
        and logits.stride(0) % 4 == 0
        and logits.data_ptr() % quad == 0
        and fast_supported(vocab_start=vocab_start, valid=valid, splits=splits)
        and logits.shape[0] < 2**31
        and splits < 65536
        # build() loads the variants; one first met inside a capture keeps Triton's
        and (
            not torch.cuda.is_current_stream_capturing()
            or any(
                k[:3] == (logits.device.index, logits.dtype == torch.float32, greedy)
                and k[5] == pre
                for k in _CUDA_KERNELS
            )
        )
    )


def gumbel_shard_keys_cuda(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    seeds: torch.Tensor,
    out: torch.Tensor,
    greedy_out: torch.Tensor | None,
    *,
    vocab_start: int,
    valid: int,
    vocab_size: int,
    splits: int,
    values: torch.Tensor | None = None,
    threads: int = _CUDA_THREADS,
    pre: tuple[torch.Tensor, int, torch.Tensor] | None = None,
) -> None:
    from sgl_nvidia_mega.common import jit

    rows = int(logits.shape[0])
    if rows == 0:
        return
    fp32 = logits.dtype == torch.float32
    key = (
        logits.device.index,
        fp32,
        greedy_out is not None,
        values is not None,
        threads,
        pre is not None,
    )
    kernel = _cuda_kernel(*key)
    if key not in _CUDA_LOGGED:
        _CUDA_LOGGED.add(key)
        logger.info(
            "NVIDIA megakernel: Qwen3.5 slot gumbel_fast runs the CUDA key launch at "
            "%d rows (%s logits%s)",
            rows,
            "fp32" if fp32 else "bf16",
            ", with the verify head's keys" if pre is not None else "",
        )
    p = _GumbelKeysParams(
        logits=logits.data_ptr(),
        stride=logits.stride(0),
        temperatures=temperatures.data_ptr(),
        seeds=seeds.data_ptr(),
        keys=out.data_ptr(),
        greedy=0 if greedy_out is None else greedy_out.data_ptr(),
        values=0 if values is None else values.data_ptr(),
        vocab_start=vocab_start,
        valid=valid,
        vocab_size=vocab_size,
        span=-(-valid // splits),
        pre=0 if pre is None else pre[0].data_ptr(),
        pre_ctas=0 if pre is None else int(pre[1]),
        epoch=0 if pre is None else pre[2].data_ptr(),
    )
    stream = torch.cuda.current_stream(logits.device).cuda_stream
    jit.launch(kernel, (rows, splits, 1), [p], stream)


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
    fast: bool = False,
    pre: bool = False,
) -> None:
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
        if fast:
            for fp32 in (False, True):
                _cuda_kernel(device_index, fp32, False)
                if pre:
                    _cuda_kernel(device_index, fp32, False, pre=True)
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
            if fast:
                # the Triton kernel serves calls the CUDA launch does not take
                _launch_fast(
                    logits,
                    temperatures,
                    seeds,
                    keys,
                    None,
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
