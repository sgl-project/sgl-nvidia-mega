# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The routed experts of up to 7 rows in one launch, on the router front's outputs: the
top-10 of the logits, FC1, SwiGLU, the NVFP4 requantization of the intermediate and FC2,
reading the trtllm-gen weight layout in place. It writes the deferred-finalize rows (row
``t * 10 + k`` is expert ``k`` of token ``t``) and the expert weights, as the trtllm
runner's deferred finalize takes them. One engine per intermediate size a rank holds:
128 at TP8 (``src/mkm_moe.cu``), 256 at TP4 (``src/mkm_moe_tp4.cu``), a 2-CTA cluster per
selectable expert. At TP8, 8 to 16 rows run the 16-token engine (``src/mkm_moe_m16.cu``:
a CTA per (token, top-k) slot; a slot whose expert an earlier slot holds exits), and 17 to
32 rows the 32-token engine (``src/mkm_moe_m32.cu``: the 16-token engine's work per
distinct expert on at most one CTA per SM, each CTA running its share of the experts in
turn and gathering each one's token rows).
"""

from __future__ import annotations

import ctypes
from functools import cache
from pathlib import Path
from typing import NamedTuple

import torch

from sgl_nvidia_mega.common import jit

HIDDEN, EXPERTS, TOP_K, MAX_TOKENS = 4096, 512, 10, 7
W13_ROWS = 256  # an expert's gate and up rows at TP8 (intermediate 128 per rank)
W13_BOX_ROWS = 128  # kW13BoxRows: rows of one W13 TMA box
THREADS = 256
CTAS = (
    2 * MAX_TOKENS * TOP_K
)  # a 2-CTA cluster per selectable expert; the unused ones exit


class Engine(NamedTuple):
    source: str  # under src/
    kernel: str
    smem_bytes: int  # sizeof(Smem) + 1024, checked when the source compiles
    w2_swizzle: str  # the W2 TMA map's swizzle: its box is a row of I / 2 bytes
    defines: tuple = ()  # extra compile options


# The validated TP8 schedule changes only when the weight and scale-factor loads issue;
# the engine's rows remain bitwise identical.
MKM7_DEFINES = (
    "-DSPEC_STAGES=3",
    "-DPACE_D=4",
    "-DFC2SF_PRE=0",
)

# per intermediate size a rank holds (TP8, TP4)
ENGINES = {
    128: Engine(
        "mkm_moe.cu", "q35_mkm_moe", 211968, "CU_TENSOR_MAP_SWIZZLE_64B", MKM7_DEFINES
    ),
    256: Engine(
        "mkm_moe_tp4.cu", "q35_mkm_moe_tp4", 216064, "CU_TENSOR_MAP_SWIZZLE_128B"
    ),
}
INTERMEDIATES = tuple(ENGINES)
SMEM_BYTES = ENGINES[128].smem_bytes
# the 16-token engines, per intermediate size a rank holds (TP8)
M16_MAX_TOKENS = 16
M16_CTAS = M16_MAX_TOKENS * TOP_K  # a CTA per (token, top-k) slot; the unused ones exit
WIDE_ENGINES = {
    128: Engine(
        "mkm_moe_m16.cu", "q35_mkm_moe_m16", 221184, "CU_TENSOR_MAP_SWIZZLE_64B"
    ),
}
# the 32-token engines, per intermediate size a rank holds (TP8)
M32_MAX_TOKENS = 32
M32_ENGINES = {
    128: Engine(
        "mkm_moe_m32.cu", "q35_mkm_moe_m32", 224256, "CU_TENSOR_MAP_SWIZZLE_64B"
    ),
}
# SMs the 32-token engine leaves to the alternate stream's shared expert, whose
# GEMMs do not fit beside an engine CTA on an SM. The wider engines keep every SM.
# each CTA of the 32-token engine runs at most 3 distinct experts (src/mkm_moe_m32.cu
# kMaxItems): it needs 32 * 10 / 3 SMs; the 64-token engine's 8 items fit as well
M32_MIN_SMS = 107
# the 64-token engines (the 32-token engine's work in (expert, row group) units)
M64_MAX_TOKENS = 64
M64_ENGINES = {
    128: Engine(
        "mkm_moe_m64.cu", "q35_mkm_moe_m64", 228352, "CU_TENSOR_MAP_SWIZZLE_64B"
    ),
}
# the 128-token engines (the 64-token engine's units, on a given routing only)
M128_MAX_TOKENS = 128
M128_ENGINES = {
    128: Engine(
        "mkm_moe_m128.cu", "q35_mkm_moe_m128", 231424, "CU_TENSOR_MAP_SWIZZLE_64B"
    ),
}
# the wide engines (src/mkm_moe_wide.cu, the 128-token engine's units on a static work split: CTA c runs experts
# c, c + G, ...), on a given routing only, built per token bound
M256_MAX_TOKENS = 256
M512_MAX_TOKENS = 512
M256_ENGINES = {
    128: Engine(
        "mkm_moe_wide.cu",
        "q35_mkm_moe_wide",
        226304,
        "CU_TENSOR_MAP_SWIZZLE_64B",
        ("-DQ35_MKM_MAX_TOK=256",),
    ),
}
M512_ENGINES = {
    128: Engine(
        "mkm_moe_wide.cu",
        "q35_mkm_moe_wide",
        231424,
        "CU_TENSOR_MAP_SWIZZLE_64B",
        ("-DQ35_MKM_MAX_TOK=512",),
    ),
}
# a wide engine's CTA runs at most 4 own experts (512 / 128) and 3 further row groups (its kMaxItems)
WIDE_MIN_SMS = 128
WIDE_ROWS = (M256_MAX_TOKENS, M512_MAX_TOKENS)
RESERVED_SMS = {
    M32_MAX_TOKENS: 32,
    M64_MAX_TOKENS: 0,
    M128_MAX_TOKENS: 0,
    M256_MAX_TOKENS: 0,
    M512_MAX_TOKENS: 0,
}
ONEPASS_MAX_ROWS = 42
WIDE_BAL_MAX_ROWS = 0  # 0 means every row count served by a wide engine


def engine_options(rows: int, plain: bool = False, wide_bal: bool = False) -> tuple:
    """Validated bitwise schedule defines for a wide engine.

    The 64-token engine keeps its stock schedule past ``ONEPASS_MAX_ROWS``.
    """
    if rows == M32_MAX_TOKENS:
        return (
            "-DTHR_D=2",
            "-DTOPK_NOINLINE=1",
            "-DONEPASS=1",
            "-DMMA_SFONLY=1",
        )
    if rows == M64_MAX_TOKENS:
        return ("-DMMA_SFONLY=1",) + (() if plain else ("-DONEPASS=1",))
    if rows in WIDE_ROWS and wide_bal and not plain:
        return ("-DBAL_SPLIT=1",)
    return ()


def reserved_sms(rows: int, m: int) -> int:
    """The validated SM reserve for an engine bucket at an actual row count."""
    if rows == M32_MAX_TOKENS and m <= 21:
        return 0
    if rows == M64_MAX_TOKENS:
        if m <= 35:
            return 32
        if m <= 42:
            return 16
    return RESERVED_SMS[rows]


class Params(ctypes.Structure):
    """qwen_mkm_moe::Params."""

    _fields_ = [
        (name, ctypes.c_void_p)
        for name in (
            "logits",
            "expert_weights",
            "topk_ids",
            "x_sf",
            "w13_sf",
            "w2_sf",
            "g1_alpha",
            "g1_scale_c",
            "g2_alpha",
            "out",
            "trace",
        )
    ] + [("num_tokens", ctypes.c_int)]


class ParamsM32(Params):
    """qwen_mkm_moe_m32::Params / qwen_mkm_moe_m64::Params: each item gathers its token rows
    from x_fp4; the 64-token engine also takes a given routing (ids_in, null: its own)."""

    _fields_ = [("x_fp4", ctypes.c_void_p), ("ids_in", ctypes.c_void_p)]


# the rows each engine serves: (7-row, 16-token, 32-token, 64-token, 128-token, 256-token, 512-token) engines up to
# these
ENGINE_ROWS = (
    MAX_TOKENS,
    M16_MAX_TOKENS,
    M32_MAX_TOKENS,
    M64_MAX_TOKENS,
    M128_MAX_TOKENS,
    M256_MAX_TOKENS,
    M512_MAX_TOKENS,
)
# the engines that take a given routing (route_topk's ids); those past 64 tokens only that way
ROUTED_ENGINES = (M64_MAX_TOKENS, M128_MAX_TOKENS, M256_MAX_TOKENS, M512_MAX_TOKENS)


def engine_rows(m: int, m16_max_rows: int = M16_MAX_TOKENS) -> int:
    """The bound of the engine that serves ``m`` rows.

    A caller may hand the tail of the 16-token range to the 32-token engine. Both
    consume the same router logits and NVFP4 rows and produce identical values.
    """
    if m16_max_rows < m <= M16_MAX_TOKENS:
        return M32_MAX_TOKENS
    return next(r for r in ENGINE_ROWS if m <= r)


def plain_schedule(rows: int, m: int, wide_bal: bool = False) -> bool:
    """Whether ``m`` rows use an engine's stock schedule.

    The 64-token engine returns to two FC2 passes above its validated bound. An
    opt-in balanced wide engine would likewise return to its static split above a
    nonzero ``WIDE_BAL_MAX_ROWS``; zero covers every row count.
    """
    if rows == M64_MAX_TOKENS:
        return m > ONEPASS_MAX_ROWS
    return rows in WIDE_ROWS and wide_bal and 0 < WIDE_BAL_MAX_ROWS < m


@cache
def _kernel(
    device_index: int,
    intermediate: int,
    rows: int,
    plain: bool = False,
    wide_bal: bool = False,
) -> jit.Kernel:
    """The engine of ``engine_rows`` bound ``rows`` at ``intermediate`` per rank."""
    engine = {
        MAX_TOKENS: ENGINES,
        M16_MAX_TOKENS: WIDE_ENGINES,
        M32_MAX_TOKENS: M32_ENGINES,
        M64_MAX_TOKENS: M64_ENGINES,
        M128_MAX_TOKENS: M128_ENGINES,
        M256_MAX_TOKENS: M256_ENGINES,
        M512_MAX_TOKENS: M512_ENGINES,
    }[rows][intermediate]
    major, minor = torch.cuda.get_device_capability(device_index)
    cubin = jit.compile_source(
        (Path(__file__).parent / "src" / engine.source).read_text(),
        engine.kernel,
        f"sm_{major}{minor}a",
        (
            "-std=c++17",
            "-default-device",
            f"-DQ35_MKM_SMEM_BYTES={engine.smem_bytes}",
        )
        + engine.defines
        + engine_options(rows, plain, wide_bal),
    )
    return jit.load(cubin, engine.kernel, (THREADS, 1, 1), engine.smem_bytes)


def max_tokens(
    intermediate: int,
    m32: bool = False,
    m64: bool = False,
    m128: bool = False,
    m256: bool = False,
    m512: bool = False,
) -> int:
    """The widest rows an engine serves for ``intermediate`` per rank (``m32`` / ``m64`` /
    ``m128`` / ``m256`` / ``m512``: with the 32 / 64 / 128 / 256 / 512-token engine)."""
    if m512 and intermediate in M512_ENGINES:
        return M512_MAX_TOKENS
    if m256 and intermediate in M256_ENGINES:
        return M256_MAX_TOKENS
    if m128 and intermediate in M128_ENGINES:
        return M128_MAX_TOKENS
    if m64 and intermediate in M64_ENGINES:
        return M64_MAX_TOKENS
    if m32 and intermediate in M32_ENGINES:
        return M32_MAX_TOKENS
    return M16_MAX_TOKENS if intermediate in WIDE_ENGINES else MAX_TOKENS


def m32_unsupported(device_index: int, intermediate: int) -> str:
    """Why the 32-token engine cannot serve ``intermediate`` per rank on this GPU ("" when
    it can)."""
    if intermediate not in M32_ENGINES:
        return f"no 32-token engine at intermediate {intermediate} per rank"
    sms = torch.cuda.get_device_properties(device_index).multi_processor_count - max(
        RESERVED_SMS.values()
    )
    if sms < M32_MIN_SMS:
        return f"{sms} SMs (the 32-token engine needs {M32_MIN_SMS})"
    return ""


def wide_unsupported(device_index: int, intermediate: int) -> str:
    """Why the 256 / 512-token engines cannot serve ``intermediate`` per rank on this GPU
    ("" when they can)."""
    if intermediate not in M256_ENGINES:
        return f"no 256 / 512-token engine at intermediate {intermediate} per rank"
    sms = torch.cuda.get_device_properties(device_index).multi_processor_count - max(
        RESERVED_SMS[r] for r in WIDE_ROWS
    )
    if sms < WIDE_MIN_SMS:
        return f"{sms} SMs (the 256 / 512-token engines need {WIDE_MIN_SMS})"
    return ""


def build(
    device_index: int,
    intermediate: int = 128,
    max_rows: int = MAX_TOKENS,
    min_rows: int = 1,
    wide_bal: bool = False,
) -> None:
    """Compile the engines that serve ``intermediate`` per rank on ``min_rows`` to
    ``max_rows`` rows (no compile inside a step)."""
    lo = 0
    for hi in ENGINE_ROWS:
        if min_rows <= hi and max_rows > lo:
            balanced = wide_bal and hi in WIDE_ROWS
            _kernel(device_index, intermediate, hi, wide_bal=balanced)
            if plain_schedule(hi, max(lo + 1, min(max_rows, hi)), wide_bal=balanced):
                _kernel(device_index, intermediate, hi, True, balanced)
        lo = hi


def _map(
    tensor: torch.Tensor,
    rows: int,
    row_bytes: int,
    box_bytes: int,
    box_rows: int,
    swizzle,
):
    """A 2-D uint8 TMA map of ``tensor`` as [rows, row_bytes]."""
    cu = jit.driver()
    u64, u32 = cu.cuuint64_t, cu.cuuint32_t
    (tensor_map,) = jit.check(
        cu.cuTensorMapEncodeTiled(
            cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
            2,
            tensor.data_ptr(),
            [u64(row_bytes), u64(rows)],
            [u64(row_bytes)],
            [u32(box_bytes), u32(box_rows)],
            [u32(1), u32(1)],
            cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            swizzle,
            cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        ),
        "cuTensorMapEncodeTiled",
    )
    return tensor_map


def weight_maps(w13: torch.Tensor, w2: torch.Tensor) -> tuple:
    """The TMA maps of a layer's W13 [512, 2 I, 2048] and W2 [512, 4096, I / 2] (uint8,
    the trtllm-gen layout; I = 128 or 256 per rank); they stay valid as long as the
    weights."""
    i = w2.shape[2] * 2
    swizzle = jit.driver().CUtensorMapSwizzle
    return (
        _map(
            w13,
            EXPERTS * 2 * i,
            HIDDEN // 2,
            128,
            W13_BOX_ROWS,
            swizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        ),
        _map(
            w2,
            EXPERTS * HIDDEN,
            i // 2,
            i // 2,
            256,
            getattr(swizzle, ENGINES[i].w2_swizzle),
        ),
    )


def mkm_moe(
    maps: tuple,
    logits: torch.Tensor,
    x_fp4: torch.Tensor,
    x_sf: torch.Tensor,
    w13_sf: torch.Tensor,
    w2_sf: torch.Tensor,
    g1_alpha: torch.Tensor,
    g1_scale_c: torch.Tensor,
    g2_alpha: torch.Tensor,
    routing: tuple | None = None,
    wide_bal: bool = False,
    m16_max_rows: int = M16_MAX_TOKENS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(gemm2_out bf16 [M * 10, 4096], expert_weights bf16 [M, 10]) for logits [M, 512]
    bf16 (M <= 7, at TP8 M <= 128) and the router front's x_fp4 [M, 2048] / x_sf [M, 256]
    (uint8, linear e4m3); w13_sf [512, 2 I, 256] / w2_sf [512, 4096, I / 16] e4m3 viewed
    as uint8 (the engine follows I); g1_alpha, g1_scale_c, g2_alpha [512] fp32. With
    ``routing`` (route_topk's ids int32 / weights bf16 [M, 10]; 33 to 64 rows, and required
    at 65 to 512) the engine takes that routing instead of ranking the logits, and returns
    those weights. ``wide_bal`` selects the opt-in balanced unit split of the 256 /
    512-token engines."""
    m = logits.shape[0]
    intermediate = w2_sf.shape[2] * 16
    widest = max_tokens(
        intermediate, m32=True, m64=True, m128=True, m256=True, m512=True
    )
    if not 1 <= m <= widest:
        raise ValueError(f"mkm: {m} rows (1-{widest})")
    rows = engine_rows(m, m16_max_rows)
    if routing is not None and rows not in ROUTED_ENGINES:
        raise ValueError(
            f"mkm: a given routing needs a 64-512-token engine, not {m} rows"
        )
    if routing is None and rows > M64_MAX_TOKENS:
        raise ValueError(f"mkm: {m} rows need a given routing (route_topk's)")
    device = logits.device
    gemm2_out = torch.empty((m * TOP_K, HIDDEN), dtype=torch.bfloat16, device=device)
    expert_weights = torch.empty((m, TOP_K), dtype=torch.bfloat16, device=device)
    topk_ids = torch.empty((m, TOP_K), dtype=torch.int32, device=device)
    fields = dict(
        logits=logits.data_ptr(),
        expert_weights=expert_weights.data_ptr(),
        topk_ids=topk_ids.data_ptr(),
        x_sf=x_sf.data_ptr(),
        w13_sf=w13_sf.data_ptr(),
        w2_sf=w2_sf.data_ptr(),
        g1_alpha=g1_alpha.data_ptr(),
        g1_scale_c=g1_scale_c.data_ptr(),
        g2_alpha=g2_alpha.data_ptr(),
        out=gemm2_out.data_ptr(),
        num_tokens=m,
    )
    balanced = wide_bal and rows in WIDE_ROWS
    kernel = _kernel(
        device.index,
        intermediate,
        rows,
        plain_schedule(rows, m, wide_bal=balanced),
        balanced,
    )
    stream = torch.cuda.current_stream().cuda_stream
    if rows in (M32_MAX_TOKENS, M64_MAX_TOKENS, M128_MAX_TOKENS) + WIDE_ROWS:
        # min(M * 10, SMs) CTAs, one per SM, each running its share of the distinct experts
        # (the wide engines: every SM, CTA c running experts c, c + G, ...)
        sms = torch.cuda.get_device_properties(
            device
        ).multi_processor_count - reserved_sms(rows, m)
        ids_in = 0 if routing is None else routing[0].data_ptr()
        if rows in WIDE_ROWS and not (
            routing[0].is_contiguous() and ids_in % 16 == 0 and sms >= WIDE_MIN_SMS
        ):
            raise ValueError(
                "mkm: the wide engines read contiguous 16-B aligned ids on >= 128 SMs"
            )
        p = ParamsM32(x_fp4=x_fp4.data_ptr(), ids_in=ids_in, **fields)
        grid = sms if rows in WIDE_ROWS else min(m * TOP_K, sms)
        jit.launch(kernel, (grid, 1, 1), [*maps, p], stream)
        return gemm2_out, (expert_weights if routing is None else routing[1])
    swizzle = jit.driver().CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
    wide = rows == M16_MAX_TOKENS
    # the x rows past m read as zeros: 8 rows a box, 16 for the 16-token engine
    x_map = _map(x_fp4, m, HIDDEN // 2, 128, 16 if wide else 8, swizzle)
    jit.launch(
        kernel,
        (M16_CTAS if wide else CTAS, 1, 1),
        [*maps, x_map, Params(**fields)],
        stream,
    )
    return gemm2_out, expert_weights
