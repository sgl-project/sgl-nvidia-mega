# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The router GEMM and the NVFP4 quantization of its input in one launch, for the
trtllm MoE runner, which routes from the logits itself (``src/router_front.cu``).

The logits stay within one BF16 ULP of SGLang's cuBLAS router GEMM, and the
quantization equals the runner's own NVFP4 input quantizer (16-element blocks, e4m3
scales, linear layout).
``router_gemm_quant_shared`` also computes the shared expert's static-FP8 gate/up
projection of up to 8 rows in the same launch; it sums in another order than the FP8
linear's GEMM (numeric).
"""

from __future__ import annotations

import ctypes
from functools import cache
from pathlib import Path
from typing import Optional

import torch

from sgl_nvidia_mega.common import jit

SOURCE = Path(__file__).parent / "src" / "router_front.cu"
TABLES_HEADER = Path(__file__).parent / "src" / "route_tables.cuh"
HIDDEN, EXPERTS, MAX_TOKENS = 4096, 512, 16
THREADS, CTAS = 256, 64  # 8 experts a CTA
_ROW_BYTES = HIDDEN * 2 + 16
_PART_BYTES = 2 * 16 * 8 * 4
_SH_ROW_BYTES = (
    HIDDEN + 16
)  # an e4m3 row of the shared stage, padded like the bf16 ones
_SH_PART_BYTES = 4 * 16 * 8 * 4  # fp32 [kShSplits 4][16][8]
SHARED_MAX_M, SHARED_MAX_N = 8, CTAS * 8  # 8 shared rows a CTA
MODE_QUANT = 1
# The validated plain-launch schedule uses four K chains, one per warp, summed in
# chain order. The shared-stage launch keeps two chains for its larger footprint.
KSPLIT = 4


class TablesParams(ctypes.Structure):
    """q35_route_tables::Tables (src/route_tables.cuh)."""

    _fields_ = [
        (name, ctypes.c_void_p)
        for name in (
            "counter",
            "total_padded",
            "expanded_to_permuted",
            "permuted_to_token",
            "tokens_per_expert",
            "cta_expert",
            "cta_row_limit",
            "live_ctas",
        )
    ] + [
        ("tile", ctypes.c_int32),
        ("local_offset", ctypes.c_int32),
        ("local_count", ctypes.c_int32),
    ]


def _headers() -> dict:
    return {"route_tables.cuh": TABLES_HEADER.read_text()}


def max_m_bucket(m: int) -> int:
    """The compiled row capacity that serves m rows."""
    return 1 if m <= 1 else 8 if m <= 8 else 16


def shared_bytes(max_m: int, shared: bool = False, ksplit: int = 2) -> int:
    """GemvSmem<max_m, shared>::kBytes: 8 weight rows, max_m input rows, the fp32 partials
    (``ksplit`` chains) and two mbarriers; with the shared stage also 8 FP8 weight rows,
    max_m FP8 input rows, its partials and a third mbarrier."""
    plain = 8 * _ROW_BYTES + max_m * _ROW_BYTES + _PART_BYTES * ksplit // 2
    if not shared:
        return plain + 16
    return plain + (8 + max_m) * _SH_ROW_BYTES + _SH_PART_BYTES + 24


class Params(ctypes.Structure):
    """qwen_router_front::Params."""

    _fields_ = [
        (name, ctypes.c_void_p)
        for name in (
            "x",
            "weight",
            "global_scale",
            "logits",
            "topk_weights",
            "topk_ids",
            "x_fp4",
            "x_sf",
            "tile_idx_to_expert_idx",
            "tile_idx_to_mn_limit",
            "expanded_idx_to_permuted_idx",
            "permuted_idx_to_expanded_idx",
            "total_num_padded_tokens",
            "num_non_exiting_tiles",
            "arrival",
            "trace",
        )
    ] + [
        ("num_tokens", ctypes.c_uint32),
        ("tile_log2", ctypes.c_uint32),
        ("mode", ctypes.c_uint32),
        ("sh_weight", ctypes.c_void_p),
        ("sh_in_scale", ctypes.c_void_p),
        ("sh_w_scale", ctypes.c_void_p),
        ("sh_out", ctypes.c_void_p),
        ("sh_n", ctypes.c_uint32),
    ]


@cache
def _cubin(device_index: int, max_m: int, ksplit: int = 2) -> bytes:
    major, minor = torch.cuda.get_device_capability(device_index)
    return jit.compile_source(
        SOURCE.read_text(),
        f"q35_router_front_m{max_m}",
        f"sm_{major}{minor}a",
        ("-std=c++17", "-default-device", f"-DQ35_ROUTER_MAX_M={max_m}")
        + ((f"-DQ35_ROUTER_KSPLIT={ksplit}",) if ksplit != 2 else ()),
        headers=_headers(),
    )


@cache
def _kernel(device_index: int, max_m: int) -> jit.Kernel:
    return jit.load(
        _cubin(device_index, max_m, KSPLIT),
        "q35_router_gemm_quant",
        (THREADS, 1, 1),
        shared_bytes(max_m, ksplit=KSPLIT),
    )


@cache
def _shared_kernel(device_index: int, max_m: int) -> jit.Kernel:
    return jit.load(
        _cubin(device_index, max_m),
        "q35_router_gemm_quant_shared",
        (THREADS, 1, 1),
        shared_bytes(max_m, shared=True),
    )


def build(device_index: int, max_rows: int, shared: bool = False) -> None:
    """Compile every row capacity up to max_rows, with ``shared`` the shared gate/up
    ones too (no compile inside a step)."""
    for m in sorted({max_m_bucket(r) for r in range(1, max_rows + 1)}):
        _kernel(device_index, m)
        if shared and m <= SHARED_MAX_M:
            _shared_kernel(device_index, m)


def router_gemm_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x [M, 4096] bf16 (M <= 16), weight [512, 4096] bf16, global_scale [1] fp32:
    (logits bf16 [M, 512], x_fp4 uint8 [M, 2048], x_sf uint8 [M, 256])."""
    m = x.shape[0]
    # the kernel reads x as packed rows (bulk copies) and serves at most 16
    if x.dtype != torch.bfloat16 or x.shape[1:] != (HIDDEN,) or not x.is_contiguous():
        raise ValueError(
            f"router front: x {x.dtype} {tuple(x.stride())} {tuple(x.shape)}"
        )
    if not 1 <= m <= MAX_TOKENS:
        raise ValueError(f"router front: {m} rows (1-{MAX_TOKENS})")
    logits = torch.empty((m, EXPERTS), dtype=torch.bfloat16, device=x.device)
    x_fp4 = torch.empty((m, HIDDEN // 2), dtype=torch.uint8, device=x.device)
    x_sf = torch.empty((m, HIDDEN // 16), dtype=torch.uint8, device=x.device)
    p = Params(
        x=x.data_ptr(),
        weight=weight.data_ptr(),
        global_scale=global_scale.data_ptr(),
        logits=logits.data_ptr(),
        x_fp4=x_fp4.data_ptr(),
        x_sf=x_sf.data_ptr(),
        num_tokens=m,
        mode=MODE_QUANT,
    )
    jit.launch(
        _kernel(x.device.index, max_m_bucket(m)),
        (CTAS, 1, 1),
        [p],
        torch.cuda.current_stream().cuda_stream,
    )
    return logits, x_fp4, x_sf


def router_gemm_quant_shared(
    x: torch.Tensor,
    weight: torch.Tensor,
    global_scale: torch.Tensor,
    sh_weight: torch.Tensor,
    sh_in_scale: torch.Tensor,
    sh_w_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``router_gemm_quant`` of x [M, 4096] (M <= 8) plus the shared expert's gate/up
    projection in the same launch: sh_weight [N, 4096] e4m3 row-major (N % 8 == 0,
    N <= 512), static scales sh_in_scale / sh_w_scale [1] fp32. Returns (logits, x_fp4,
    x_sf, gate_up bf16 [M, N]); gate_up = (q(x) @ sh_weight.T) * sh_in_scale * sh_w_scale
    with q the static e4m3 quantization of x, summed in fp32."""
    m, n = x.shape[0], sh_weight.shape[0]
    if x.dtype != torch.bfloat16 or x.shape[1:] != (HIDDEN,) or not x.is_contiguous():
        raise ValueError(
            f"router front: x {x.dtype} {tuple(x.stride())} {tuple(x.shape)}"
        )
    if not 1 <= m <= SHARED_MAX_M:
        raise ValueError(f"router front with the shared stage: {m} rows (1-8)")
    if (
        sh_weight.shape[1:] != (HIDDEN,)
        or n % 8
        or n > SHARED_MAX_N
        or not sh_weight.is_contiguous()
    ):
        raise ValueError(f"router front: shared weight {tuple(sh_weight.shape)}")
    logits = torch.empty((m, EXPERTS), dtype=torch.bfloat16, device=x.device)
    x_fp4 = torch.empty((m, HIDDEN // 2), dtype=torch.uint8, device=x.device)
    x_sf = torch.empty((m, HIDDEN // 16), dtype=torch.uint8, device=x.device)
    gate_up = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    p = Params(
        x=x.data_ptr(),
        weight=weight.data_ptr(),
        global_scale=global_scale.data_ptr(),
        logits=logits.data_ptr(),
        x_fp4=x_fp4.data_ptr(),
        x_sf=x_sf.data_ptr(),
        num_tokens=m,
        mode=MODE_QUANT,
        sh_weight=sh_weight.data_ptr(),
        sh_in_scale=sh_in_scale.data_ptr(),
        sh_w_scale=sh_w_scale.data_ptr(),
        sh_out=gate_up.data_ptr(),
        sh_n=n,
    )
    kernel = _shared_kernel(x.device.index, max_m_bucket(m))
    jit.launch(kernel, (CTAS, 1, 1), [p], torch.cuda.current_stream().cuda_stream)
    return logits, x_fp4, x_sf, gate_up


# The top-10 and NVFP4 input behind SGLang's router GEMM (slot route_topk, src/route_topk.cu)
TOPK_SOURCE = Path(__file__).parent / "src" / "route_topk.cu"
TOP_K = 10
TOPK_MAX_ROWS = 896  # the widest verify step it is built for (128 requests of 7 rows)


class TopkParams(ctypes.Structure):
    """q35_route_topk::Params."""

    _fields_ = [
        (name, ctypes.c_void_p)
        for name in (
            "logits",
            "x",
            "global_scale",
            "topk_ids",
            "topk_weights",
            "x_fp4",
            "x_sf",
        )
    ] + [("tables", TablesParams), ("num_tokens", ctypes.c_int32)]


def max_ctas(m: int, tile: int) -> int:
    """The MoE body's CTA bound for m tokens (trtllm-gen runner.h getMaxNumCtasInBatchDim)."""
    expanded = m * TOP_K
    filled = min(EXPERTS, expanded)
    return filled + (expanded - filled) // tile


class RouteTables:
    """Graph-stable trtllm-gen routing tables of one width and tile_n for the local experts
    [local_offset, local_offset + local_count) (expert parallelism: the runner's
    local_expert_offset / local_num_experts), which route_topk_quant or route-big writes
    for the MoE body; ``metadata()`` is
    FlashInfer's nine-tensor RoutingMetadataBuffers order. Also the width's top-10 buffers,
    which the tables borrow. A width's launches run one after another (the arrival counter
    is the width's)."""

    def __init__(
        self,
        m: int,
        tile: int,
        device: torch.device,
        local_offset: int = 0,
        local_count: int = EXPERTS,
    ):
        i32 = dict(dtype=torch.int32, device=device)
        ctas = max_ctas(m, tile)
        if not (
            0 <= local_offset
            and 0 < local_count
            and local_offset + local_count <= EXPERTS
        ):
            raise ValueError(
                f"route tables: local experts {local_offset} + {local_count}"
            )
        self.m, self.tile = m, tile
        self.local_offset, self.local_count = local_offset, local_count
        self.ids = torch.empty((m, TOP_K), **i32)
        self.weights = torch.empty((m, TOP_K), dtype=torch.bfloat16, device=device)
        self.counter = torch.zeros(1, **i32)
        self.total_padded = torch.zeros(1, **i32)
        self.expanded_to_permuted = torch.zeros(m * TOP_K, **i32)
        self.permuted_to_token = torch.full((ctas * tile + 1,), -1, **i32)
        self.histogram = torch.zeros(2 * EXPERTS, **i32)  # FlashInfer's scratch; unused
        self.tokens_per_expert = torch.zeros(EXPERTS, **i32)
        self.cta_expert = torch.zeros(ctas, **i32)
        self.cta_row_limit = torch.zeros(ctas, **i32)
        self.live_ctas = torch.zeros(1, **i32)

    def params(self) -> TablesParams:
        return TablesParams(
            counter=self.counter.data_ptr(),
            total_padded=self.total_padded.data_ptr(),
            expanded_to_permuted=self.expanded_to_permuted.data_ptr(),
            permuted_to_token=self.permuted_to_token.data_ptr(),
            tokens_per_expert=self.tokens_per_expert.data_ptr(),
            cta_expert=self.cta_expert.data_ptr(),
            cta_row_limit=self.cta_row_limit.data_ptr(),
            live_ctas=self.live_ctas.data_ptr(),
            tile=self.tile,
            local_offset=self.local_offset,
            local_count=self.local_count,
        )

    def metadata(self) -> list[torch.Tensor]:
        return [
            self.total_padded,
            self.expanded_to_permuted,
            self.permuted_to_token,
            self.weights,
            self.histogram,
            self.tokens_per_expert,
            self.cta_expert,
            self.cta_row_limit,
            self.live_ctas,
        ]


@cache
def _topk_kernel(device_index: int) -> jit.Kernel:
    major, minor = torch.cuda.get_device_capability(device_index)
    cubin = jit.compile_source(
        TOPK_SOURCE.read_text(),
        "q35_route_topk",
        f"sm_{major}{minor}a",
        ("-std=c++17", "-default-device"),
        headers=_headers(),
    )
    return jit.load(cubin, "q35_route_topk_quant", (THREADS, 1, 1), 0)


def build_topk(device_index: int) -> None:
    """Compile slot route_topk's launch (no compile inside a step)."""
    _topk_kernel(device_index)


def route_topk_quant(
    logits: torch.Tensor,
    x: torch.Tensor,
    global_scale: torch.Tensor,
    quantize: bool = True,
    tables: Optional[RouteTables] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """logits [M, 512] bf16 (the router GEMM's), x [M, 4096] bf16, global_scale [1]
    fp32: (topk_ids int32 [M, 10], topk_weights bf16 [M, 10], x_fp4 uint8 [M, 2048],
    x_sf uint8 [M, 256]); the routing equals the trtllm-gen RenormalizeNaive routing's
    and the input the runner's NVFP4 quantizer's, bit for bit. Without ``quantize`` only
    the routing (x_fp4 / x_sf None). With ``tables`` (of M rows) the top-10 goes to its
    buffers and the launch also writes its routing tables."""
    m = x.shape[0]
    for t, shape in ((logits, (m, EXPERTS)), (x, (m, HIDDEN))):
        if (
            t.dtype != torch.bfloat16
            or tuple(t.shape) != shape
            or not t.is_contiguous()
        ):
            raise ValueError(f"route_topk: {t.dtype} {tuple(t.shape)}, want {shape}")
    if not 1 <= m <= TOPK_MAX_ROWS:
        raise ValueError(f"route_topk: {m} rows (1-{TOPK_MAX_ROWS})")
    if tables is None:
        ids = torch.empty((m, TOP_K), dtype=torch.int32, device=x.device)
        weights = torch.empty((m, TOP_K), dtype=torch.bfloat16, device=x.device)
    elif tables.m != m:
        raise ValueError(f"route_topk: tables of {tables.m} rows for {m}")
    else:
        ids, weights = tables.ids, tables.weights
    x_fp4 = x_sf = None
    if quantize:
        x_fp4 = torch.empty((m, HIDDEN // 2), dtype=torch.uint8, device=x.device)
        x_sf = torch.empty((m, HIDDEN // 16), dtype=torch.uint8, device=x.device)
    p = TopkParams(
        logits=logits.data_ptr(),
        x=x.data_ptr(),
        global_scale=global_scale.data_ptr(),
        topk_ids=ids.data_ptr(),
        topk_weights=weights.data_ptr(),
        x_fp4=jit.null_or_pointer(x_fp4).value,
        x_sf=jit.null_or_pointer(x_sf).value,
        num_tokens=m,
    )
    if tables is not None:
        p.tables = tables.params()
    jit.launch(
        _topk_kernel(x.device.index),
        (m, 1, 1),
        [p],
        torch.cuda.current_stream().cuda_stream,
    )
    return ids, weights, x_fp4, x_sf


BIG_SOURCE = Path(__file__).parent / "src" / "route_big.cu"
# TRT-LLM r3's route-only big front: one routing cluster of 8 CTAs (up to
# 256 rows) or 16 (non-portable, up to 1024), plus quantization clusters.
BIG_THREADS, BIG_CTAS = 1024, 32


def big_shape(m: int) -> tuple[int, int, int]:
    """(CTAs a cluster, tokens a routing CTA, token-mask words an expert) of route_big at m rows."""
    cl = 8 if m <= 256 else 16
    tpc = -(-m // cl)
    return cl, tpc, -(-tpc // 32)


class BigParams(ctypes.Structure):
    """q35_route_big::Params."""

    _fields_ = TopkParams._fields_ + [("tpc", ctypes.c_int32)]


@cache
def _big_kernel(device_index: int, cl: int, mw: int, nl: int) -> jit.Kernel:
    major, minor = torch.cuda.get_device_capability(device_index)
    defines = f"#define CL {cl}\n#define MW {mw}\n#define NL {nl}\n"
    cubin = jit.compile_source(
        defines + BIG_SOURCE.read_text(),
        "q35_route_big",
        f"sm_{major}{minor}a",
        ("-std=c++17", "-default-device"),
        headers=_headers(),
    )
    kernel = jit.load(cubin, "q35_route_big_quant", (BIG_THREADS, 1, 1), 0)
    if cl > 8:
        cu = jit.driver()
        jit.check(
            cu.cuFuncSetAttribute(
                kernel.function,
                cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
                1,
            ),
            "cuFuncSetAttribute",
        )
    return kernel


def build_big(device_index: int, local_count: int) -> None:
    """Compile route_big's launches for every wide step shape (no compile inside a step)."""
    for cl, mw in ((8, 1), (16, 1), (16, 2)):
        _big_kernel(device_index, cl, mw, local_count)


def route_big_quant(
    logits: torch.Tensor,
    x: torch.Tensor,
    global_scale: torch.Tensor,
    tables: RouteTables,
    ctas: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """route_topk_quant(logits, x, global_scale, tables=tables) as one launch of thread-block
    clusters (TRT-LLM's route-only big front): the same top-10 and NVFP4 input bit for bit, and
    the same routing tables up to the order of an expert's rows (token order here)."""
    m = x.shape[0]
    for t, shape in ((logits, (m, EXPERTS)), (x, (m, HIDDEN))):
        if (
            t.dtype != torch.bfloat16
            or tuple(t.shape) != shape
            or not t.is_contiguous()
        ):
            raise ValueError(f"route_big: {t.dtype} {tuple(t.shape)}, want {shape}")
    if tables.m != m or not 1 <= m <= 1024:
        raise ValueError(f"route_big: tables of {tables.m} rows for {m}")
    if tables.local_count % 32 or tables.local_count > BIG_THREADS:
        raise ValueError(f"route_big: {tables.local_count} local experts")
    cl, tpc, mw = big_shape(m)
    x_fp4 = torch.empty((m, HIDDEN // 2), dtype=torch.uint8, device=x.device)
    x_sf = torch.empty((m, HIDDEN // 16), dtype=torch.uint8, device=x.device)
    p = BigParams(
        logits=logits.data_ptr(),
        x=x.data_ptr(),
        global_scale=global_scale.data_ptr(),
        topk_ids=tables.ids.data_ptr(),
        topk_weights=tables.weights.data_ptr(),
        x_fp4=x_fp4.data_ptr(),
        x_sf=x_sf.data_ptr(),
        num_tokens=m,
        tpc=tpc,
    )
    p.tables = tables.params()
    launch_ctas = max(2 * cl, (BIG_CTAS if ctas is None else ctas) // cl * cl)
    jit.launch(
        _big_kernel(x.device.index, cl, mw, tables.local_count),
        (launch_ctas, 1, 1),
        [p],
        torch.cuda.current_stream().cuda_stream,
        cluster=(cl, 1, 1),
    )
    return tables.ids, tables.weights, x_fp4, x_sf


GEMM_MAX_ROWS = 56  # slot route_gemm: the router front in tiles of 16 rows up to here


def router_gemm_quant_tiled(
    x: torch.Tensor, weight: torch.Tensor, global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``router_gemm_quant`` past 16 rows (up to GEMM_MAX_ROWS): one launch of the
    16-row front over token tiles (grid y). Each tile's logits sum as the 16-row front's,
    in another order than SGLang's cuBLAS GEMM at these rows (numeric)."""
    m = x.shape[0]
    if x.dtype != torch.bfloat16 or x.shape[1:] != (HIDDEN,) or not x.is_contiguous():
        raise ValueError(
            f"router front: x {x.dtype} {tuple(x.stride())} {tuple(x.shape)}"
        )
    if not 1 <= m <= GEMM_MAX_ROWS:
        raise ValueError(f"tiled router front: {m} rows (1-{GEMM_MAX_ROWS})")
    logits = torch.empty((m, EXPERTS), dtype=torch.bfloat16, device=x.device)
    x_fp4 = torch.empty((m, HIDDEN // 2), dtype=torch.uint8, device=x.device)
    x_sf = torch.empty((m, HIDDEN // 16), dtype=torch.uint8, device=x.device)
    p = Params(
        x=x.data_ptr(),
        weight=weight.data_ptr(),
        global_scale=global_scale.data_ptr(),
        logits=logits.data_ptr(),
        x_fp4=x_fp4.data_ptr(),
        x_sf=x_sf.data_ptr(),
        num_tokens=m,
        mode=MODE_QUANT,
    )
    tiles = -(-m // MAX_TOKENS)
    jit.launch(
        _kernel(x.device.index, MAX_TOKENS),
        (CTAS, tiles, 1),
        [p],
        torch.cuda.current_stream().cuda_stream,
    )
    return logits, x_fp4, x_sf
