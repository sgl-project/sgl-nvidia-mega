# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax M3.1's LM-head shard as MXFP8, streamed at 1-8 rows (slot lm_head_mxfp8): a
copy of ``deepseek_v41/sm103/lm_head_mxfp8.py`` (the quantizer, the layout and the
variants are that module's) built with this package's nvcc helper (``sm103/nvcc.py``)
and loaded and launched through ``sm103/host.py``. Its kernels build in
:func:`precompile` only: a launch that finds none raises, so no step compiles.

:func:`quantize` turns this rank's BF16 shard [N, K] into E4M3 codes [N, K] and UE8M0
scales of 32 columns in FlashInfer's 128x4 swizzle (rows padded to 128). A block's scale
is the smallest power of two that keeps its largest magnitude within E4M3's 448, so no
code saturates; codes round to nearest even. :func:`matmul` runs lm_head_mxfp8.cu: BF16
rows times the dequantized weight with FP32 accumulation, rounded once to BF16. Its
LH_FP8=0 build reads a BF16 weight through the same fragments and MMAs
(:func:`matmul_bf16`), so ``matmul(x, q) == matmul_bf16(x, dequantize(q))`` bit for bit.

Shapes: 1-8 rows; K a multiple of 128; any N. The codes are not padded: the tiles are
ceil(N / 16), a tile's rows past N read row N - 1 and are not stored, and the kernel
writes [rows, N]. Only the scales' rows are padded, to 128 (M3.1's TP4 shard: N 50,016,
so 32 rows of zero bytes), and no tile reads them.
"""

from __future__ import annotations

import ctypes
from functools import cache
from pathlib import Path
from typing import NamedTuple, Optional

import torch

from sgl_nvidia_mega.common.jit import pointer as _pointer
from sgl_nvidia_mega.minimax_m31.sm103 import host, nvcc

SOURCE = Path(__file__).resolve().parent / "lm_head_mxfp8.cu"
SYMBOL = "m31_lm_head_stream"
# lm_head_mxfp8.cu annotates every function, so it needs no default-device switch; the
# floating-point options stay the compiler's defaults
NVCC_OPTIONS = ("-std=c++17",)

MAX_ROWS = 8  # one MMA's 8 columns
BLOCK = 32  # columns a scale
TILE = 16  # weight rows a warp
CHUNK = 128  # columns of a row a warp step reads (4 lanes x 32)
WARPS = 16
E4M3_MAX = 448.0
# the kernel builds 2^(scale - 7) as a BF16 exponent: scale bytes up to 134
MAX_SCALE = 134
SUPPORTED_CAPABILITIES = ((10, 0), (10, 3))


class Mxfp8Weight(NamedTuple):
    codes: torch.Tensor  # float8_e4m3fn [N, K]
    scales: torch.Tensor  # uint8 [ceil(N / 128) * 128 * K / 32], 128x4 swizzle

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.codes.shape)


def unsupported_reason(weight: torch.Tensor) -> str:
    """Why this weight cannot be served ("" when it can)."""
    if weight.dim() != 2 or weight.dtype != torch.bfloat16:
        return (
            f"the LM head weight is {weight.dtype} {tuple(weight.shape)}, not 2-D BF16"
        )
    if weight.shape[1] % CHUNK:
        return f"K {weight.shape[1]} is not a multiple of {CHUNK}"
    if weight.is_cuda and torch.cuda.get_device_capability(weight.device) not in (
        SUPPORTED_CAPABILITIES
    ):
        return "the stream kernel is built for SM100 / SM103"
    return ""


def _pad128(n: int) -> int:
    return -(-n // 128) * 128


def block_exponents(amax: torch.Tensor) -> torch.Tensor:
    """The smallest e with amax <= 448 * 2^e, exactly (448 = 0.875 * 2^9)."""
    mantissa, exponent = torch.frexp(amax.float())
    e = exponent - 9 + (mantissa > 0.875).to(exponent.dtype)
    return e.clamp(-127, 127)


def swizzle(scales: torch.Tensor) -> torch.Tensor:
    """[Npad, K / 32] scale bytes -> FlashInfer's 128x4 layout, flat."""
    npad, kb = scales.shape
    return (
        scales.view(npad // 128, 4, 32, kb // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .view(-1)
    )


def unswizzle(flat: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """The inverse of :func:`swizzle`, cropped to [n, K / 32]."""
    npad, kb = _pad128(n), k // BLOCK
    return (
        flat.view(npad // 128, kb // 4, 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(npad, kb)[:n]
    )


@torch.no_grad()
def quantize(weight: torch.Tensor, *, rows_at_once: int = 4096) -> Mxfp8Weight:
    """BF16 [N, K] -> MXFP8 (block 32, UE8M0 scales swizzled 128x4); a few rows at a
    time, so the FP32 temporaries stay small."""
    n, k = weight.shape
    assert k % CHUNK == 0, k
    codes = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=weight.device)
    scales = torch.zeros(
        (_pad128(n), k // BLOCK), dtype=torch.uint8, device=weight.device
    )
    for start in range(0, n, rows_at_once):
        w = weight[start : start + rows_at_once].float()
        blocks = w.view(w.shape[0], k // BLOCK, BLOCK)
        e = block_exponents(blocks.abs().amax(dim=-1))
        codes[start : start + w.shape[0]] = (
            torch.ldexp(blocks, -e.unsqueeze(-1).float())
            .view_as(w)
            .to(torch.float8_e4m3fn)
        )
        scales[start : start + w.shape[0]] = (e + 127).to(torch.uint8)
    return Mxfp8Weight(codes, swizzle(scales))


@torch.no_grad()
def dequantize(q: Mxfp8Weight) -> torch.Tensor:
    """The exact BF16 values the kernel multiplies: code x 2^(scale - 127)."""
    n, k = q.shape
    e = unswizzle(q.scales, n, k).to(torch.int32) - 127
    values = torch.ldexp(
        q.codes.float().view(n, k // BLOCK, BLOCK), e.unsqueeze(-1).float()
    )
    return values.view(n, k).to(torch.bfloat16)


def scale_reason(q: Mxfp8Weight) -> str:
    peak = int(q.scales.max().item())
    if peak > MAX_SCALE:
        return f"a block scale of 2^{peak - 127} (the kernel takes up to 2^{MAX_SCALE - 127})"
    return ""


def nbytes(q: Mxfp8Weight) -> int:
    return q.codes.numel() + q.scales.numel()


# --- the stream kernel --------------------------------------------------------


class Variant(NamedTuple):
    k: int
    fp8: bool
    split: int
    depth: int


def split_for(n: int, sms: int) -> int:
    """The K split: two warps a tile (the split deepseek_v41 measured fastest on GB300
    at its TP4 and TP8 shards); four when a CTA has fewer tiles than a quarter of its
    warps."""
    tiles = -(-n // TILE)
    per_cta = -(-tiles // min(sms, tiles))
    return 4 if per_cta <= WARPS // 4 else 2


def variant(n: int, k: int, *, fp8: bool, sms: int) -> Variant:
    split = split_for(n, sms)
    chunks = k // CHUNK // split
    if k % (CHUNK * split):
        raise ValueError(f"K {k} does not split into {split} parts of {CHUNK} columns")
    depths = (4, 5, 2, 1) if fp8 else (2, 1)
    return Variant(k, fp8, split, next(d for d in depths if chunks % d == 0))


def shared_bytes(v: Variant) -> int:
    x = MAX_ROWS * (v.k + 8) * 2
    partials = (WARPS // v.split) * (v.split - 1) * 32 * 4 * 4
    return x + partials


def compile_cubin(v: Variant, arch: str) -> bytes:
    """One variant's module, with nvcc."""
    options = NVCC_OPTIONS + (
        f"-DLH_K={v.k}",
        f"-DLH_FP8={int(v.fp8)}",
        f"-DLH_SPLIT={v.split}",
        f"-DLH_DEPTH={v.depth}",
        f"-DLH_WARPS={WARPS}",
    )
    name = (
        f"minimax_m31_lm_head_{'mxfp8' if v.fp8 else 'bf16'}_k{v.k}_s{v.split}"
        f"_d{v.depth}"
    )
    # the translation unit includes the toolkit's headers only: no include root
    return nvcc.compile_cubin(name, SOURCE.read_text(), (), arch, options)


class _Kernel(NamedTuple):
    function: object  # cuda.bindings.driver.CUfunction
    shared: int  # bytes of dynamic shared memory


# (device index, variant) -> its loaded kernel; filled by precompile only
_KERNELS: dict[tuple[int, Variant], _Kernel] = {}


@cache
def _sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def precompile(n: int, k: int, device, *, fp8: bool = True) -> None:
    """Build (or read from the cache) and load the variant this shape runs: at load,
    before CUDA graph capture and before any launch."""
    device = host.cuda_device(device)
    v = variant(n, k, fp8=fp8, sms=_sms(device.index))
    if (device.index, v) in _KERNELS:
        return
    with torch.cuda.device(device):
        torch.zeros(1, device=device)  # make the primary context current
        module = host.Module(compile_cubin(v, nvcc.device_arch(device)), SYMBOL)
        shared = shared_bytes(v)
        module.set_shared_memory(shared)
    _KERNELS[device.index, v] = _Kernel(module.function, shared)


def _kernel(device: torch.device, v: Variant) -> _Kernel:
    kernel = _KERNELS.get((device.index, v))
    if kernel is None:
        raise RuntimeError(
            f"the LM-head stream {v} is not built on {device}: precompile() builds it "
            "at load, a launch does not compile"
        )
    return kernel


def _run(
    x: torch.Tensor,
    weight: torch.Tensor,
    scales: Optional[torch.Tensor],
    n: int,
    k: int,
    fp8: bool,
    out: Optional[torch.Tensor],
) -> torch.Tensor:
    rows = x.shape[0] if x.dim() == 2 else 0
    if not (
        x.dim() == 2
        and 1 <= rows <= MAX_ROWS
        and x.shape[1] == k
        and x.dtype == torch.bfloat16
        and x.is_cuda
        and x.stride(1) == 1
        and x.stride(0) % 4 == 0
        and x.data_ptr() % 8 == 0
    ):
        raise ValueError(
            f"x must be BF16 [1-{MAX_ROWS}, {k}] on CUDA with unit column stride and "
            f"8-byte aligned rows, got {tuple(x.shape)} {x.dtype} on {x.device} "
            f"strides {x.stride()}"
        )
    device = host.cuda_device(x.device)
    if out is None:
        out = torch.empty((rows, n), dtype=torch.bfloat16, device=device)
    assert out.shape == (rows, n) and out.stride(1) == 1 and out.dtype == torch.bfloat16
    sms = _sms(device.index)
    v = variant(n, k, fp8=fp8, sms=sms)
    kernel = _kernel(device, v)
    tiles = -(-n // TILE)
    host.launch(
        kernel.function,
        (min(sms, tiles), 1, 1),
        (WARPS * 32, 1, 1),
        kernel.shared,
        [
            _pointer(weight),
            _pointer(scales) if scales is not None else ctypes.c_void_p(0),
            _pointer(x),
            _pointer(out),
            ctypes.c_int32(n),
            ctypes.c_int32(rows),
            ctypes.c_int32(x.stride(0)),
            ctypes.c_int32(out.stride(0)),
        ],
        torch.cuda.current_stream(device).cuda_stream,
        # the kernel prefetches its first weight chunks, then waits for the
        # predecessor that writes x
        dependent=True,
    )
    return out


def matmul(
    x: torch.Tensor, q: Mxfp8Weight, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """``x @ dequantize(q).T`` in BF16 for BF16 ``x`` [1-8, K]."""
    n, k = q.shape
    return _run(x, q.codes, q.scales, n, k, True, out)


def matmul_bf16(
    x: torch.Tensor, weight: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """The same kernel on a BF16 weight [N, K]: the order :func:`matmul` sums in."""
    assert weight.dtype == torch.bfloat16 and weight.is_contiguous()
    n, k = weight.shape
    return _run(x, weight, None, n, k, False, out)
