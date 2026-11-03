"""Host side of the ``moe_x_kernels`` routed-expert path: compile caches, launches and the full expert pipeline.

Pipeline for one routed-expert call (all on the current stream, no host synchronization, CUDA-graph capturable once
the kernels are compiled):

1. ``moe_sort`` (FlashInfer): expert-grouped, tile-padded row order for the call's local (token, slot) pairs.
2. ``GatherSwigluGroupedGemmKernel``: first expert GEMM on the call's NVFP4 activations (gathered by token) and the
   stock-layout gate/up weights, SwiGLU, per-token and per-expert scales, BF16 output in natural intermediate order.
3. FlashInfer's per-token NVFP4 quantizer (the same entry and recipe as the fused-MoE runner uses between its GEMMs).
4. ``GroupedGemmRowsKernel``: second expert GEMM on the stock-layout down weights, per-row and per-expert scales,
   unweighted BF16 rows in the sorted row order.

Returns the same deferred-finalize triple as ``trtllm_fp4_block_scale_routed_moe(do_finalize=False)``.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

import torch

TOP_K = 8
SF_VEC = 16
# 1 / (e4m3 max * e2m1 max): the per-token NVFP4 global-scale recipe of the activations.
PER_TOKEN_SCALE_INV = 1.0 / (448.0 * 6.0)


@dataclass(frozen=True)
class Tactic:
    """Kernel configuration of one GEMM: (tile M, tile N) of the MMA tiler. Tile M 256 runs either as B-reuse on one
    CTA (two 128-row MMAs) or, with ``two_cta``, as one MMA over a pair of CTAs (cluster of 2 along M). ``cluster_n`` > 1
    groups that many neighbouring N tiles of the same rows into one cluster, which then loads the A tile once (multicast)."""

    tile_m: int = 128
    tile_n: int = 256
    a_path: str = "cpasync"  # first GEMM only: "cpasync" or "tma"
    two_cta: bool = False
    cluster_n: int = 1

    @property
    def mma_tiler(self):
        return (self.tile_m, self.tile_n, 256)

    @property
    def mma_inst_shape(self):
        return (256 if self.two_cta else 128, self.tile_n, 128)

    @property
    def cluster_shape_mn(self):
        return (2 if self.two_cta else 1, self.cluster_n)

    @property
    def cluster_size(self):
        return (2 if self.two_cta else 1) * self.cluster_n


# Chosen by a sweep of the tile configurations on random routing (bench); tunable.
DEFAULT_GEMM1 = Tactic(256, 256, "cpasync", two_cta=True)
DEFAULT_GEMM2 = Tactic(256, 256, "cpasync", two_cta=True)

_lock = threading.Lock()
_cache: dict = {}


def _max_active_clusters(cluster_size: int = 1) -> int:
    override = os.environ.get("MUSE_VR_MOE_X_MAX_CLUSTERS", "").strip()
    if override:
        return int(override) // cluster_size
    from flashinfer.cute_dsl.utils import get_max_active_clusters

    return get_max_active_clusters(cluster_size)


def _ptr(dtype, addr: int, align: int | None = None):
    import cutlass.cute as cute
    from flashinfer.cute_dsl.utils import make_ptr

    return make_ptr(dtype, addr, cute.AddressSpace.gmem, assumed_align=align)


def _stream():
    import cuda.bindings.driver as cuda

    if not torch.cuda.is_available():  # ahead-of-time build without a device
        return cuda.CUstream(0)
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def _gemm1_ptrs(addrs=None):
    import cutlass

    a = addrs or [0x100000] * 12
    return [
        _ptr(cutlass.Float4E2M1FN, a[0], 32),  # a
        _ptr(cutlass.Float4E2M1FN, a[1], 32),  # b
        _ptr(cutlass.Float8E4M3FN, a[2], 16),  # a_sf
        _ptr(cutlass.Float8E4M3FN, a[3], 16),  # b_sf
        _ptr(cutlass.BFloat16, a[4], 32),  # c
        _ptr(cutlass.Float32, a[5]),  # alpha_gate
        _ptr(cutlass.Float32, a[6]),  # alpha_up
        _ptr(cutlass.Float32, a[7]),  # token_scale
        _ptr(cutlass.Int32, a[8]),  # tile -> expert
        _ptr(cutlass.Int32, a[9]),  # tile -> mn limit
        _ptr(cutlass.Int32, a[10]),  # permuted row -> expanded index
        _ptr(cutlass.Int32, a[11]),  # number of tiles
    ]


def _gemm2_ptrs(addrs=None):
    import cutlass

    a = addrs or [0x100000] * 16
    return [
        _ptr(cutlass.Float4E2M1FN, a[0], 32),  # a
        _ptr(cutlass.Float4E2M1FN, a[1], 32),  # b
        _ptr(cutlass.Float8E4M3FN, a[2], 16),  # a_sf
        _ptr(cutlass.Float8E4M3FN, a[3], 16),  # b_sf
        _ptr(cutlass.BFloat16, a[4], 32),  # c
        _ptr(cutlass.Float32, a[5]),  # alpha
        _ptr(cutlass.Float32, a[6]),  # row scale
        _ptr(cutlass.Float32, a[7]),  # second per-expert scale (up, SwiGLU mode)
        _ptr(cutlass.Int32, a[8]),  # row maximum words (SwiGLU mode with row_amax)
        _ptr(cutlass.Uint8, a[9], 16),  # fused conversion: payload
        _ptr(cutlass.Uint8, a[10], 16),  # fused conversion: swizzled block scales
        _ptr(cutlass.Float32, a[11]),  # fused conversion: per-row scales
        _ptr(cutlass.Int32, a[12]),  # fused conversion: ctl words (counters, done flags)
        _ptr(cutlass.Int32, a[13]),  # tile -> expert
        _ptr(cutlass.Int32, a[14]),  # tile -> mn limit
        _ptr(cutlass.Int32, a[15]),  # number of tiles
    ]


def compile_gemm1(tactic: Tactic, *, use_pdl: bool = False, vectorized_f32: bool = True):
    """Compiled first GEMM for a tactic (cached per process; CuTe DSL's file cache applies across processes)."""
    import cutlass.cute as cute

    from .arch import device_arch
    from .bounded_wait import bounded_waits
    from .gather_swiglu_gemm import GatherSwigluGroupedGemmKernel

    key = ("gemm1", tactic, use_pdl, vectorized_f32)
    with _lock:
        hit = _cache.get(key)
        if hit is not None:
            return hit
        kern = GatherSwigluGroupedGemmKernel(
            sf_vec_size=SF_VEC,
            mma_inst_shape=tactic.mma_inst_shape,
            mma_tiler=tactic.mma_tiler,
            cluster_shape_mn=tactic.cluster_shape_mn,
            vectorized_f32=vectorized_f32,
            topk=TOP_K,
            arch=device_arch(),
            raster_along_m=False,
            a_path=tactic.a_path,
            use_pdl=use_pdl,
        )
        with bounded_waits():
            fn = cute.compile(
                kern.wrapper,
                *_gemm1_ptrs(),
                1024,  # orig_m (runtime)
                tactic.tile_m * 8,  # permuted m (runtime)
                256,  # n (runtime)
                256,  # k (runtime)
                1,  # experts (runtime)
                tile_size=tactic.tile_m,
                scaling_vector_size=SF_VEC,
                max_active_clusters=_max_active_clusters(tactic.cluster_size),
                stream=_stream(),
            )
        _cache[key] = fn
        return fn


def compile_gemm2(tactic: Tactic, swiglu: bool = False, row_amax: bool = False, fuse_quant: bool = False):
    """Compiled rows kernel: the second GEMM, or (``swiglu``) the first GEMM over sorted-row activations (with
    ``row_amax``: also reducing each row's largest |output| for the row-block quantizer)."""
    import cutlass.cute as cute

    from .arch import device_arch
    from .bounded_wait import bounded_waits
    from .grouped_gemm_rows import GroupedGemmRowsKernel

    key = ((("gemm1_rows_fq" if fuse_quant else "gemm1_rows_amax") if row_amax else "gemm1_rows") if swiglu
           else "gemm2", tactic)
    # output stores: per-row bulk copies (default; measured faster) or tensor-memory-access tile stores
    tma_store = os.environ.get("MUSE_VR_MOE_X_TMA_STORE", "0") == "1"
    if tma_store:
        key = key + ("tma_store",)
    with _lock:
        hit = _cache.get(key)
        if hit is not None:
            return hit
        kern = GroupedGemmRowsKernel(
            sf_vec_size=SF_VEC,
            mma_inst_shape=tactic.mma_inst_shape,
            mma_tiler=tactic.mma_tiler,
            cluster_shape_mn=tactic.cluster_shape_mn,
            arch=device_arch(),
            raster_along_m=False,
            swiglu=swiglu,
            row_amax=row_amax,
            fuse_quant=fuse_quant,
            tma_store=tma_store,
        )
        with bounded_waits():
            fn = cute.compile(
                kern.wrapper,
                *_gemm2_ptrs(),
                tactic.tile_m * 8,  # m (runtime)
                256,  # n (runtime)
                256,  # k (runtime)
                1,  # experts (runtime)
                tile_size=tactic.tile_m,
                scaling_vector_size=SF_VEC,
                max_active_clusters=_max_active_clusters(tactic.cluster_size),
                stream=_stream(),
            )
        _cache[key] = fn
        return fn


def gemm1_rows_mode() -> bool:
    """First GEMM as the rows kernel over a sorted-row copy of the activations (default), else the gather kernel."""
    return os.environ.get("MUSE_VR_MOE_X_GEMM1", "rows").strip().lower() != "gather"


def quant_form(t1: Tactic) -> str:
    """Where the second GEMM's input is quantized (default recipe only, rows form of the first GEMM only):
    "blocks" (default: separate row-block quantizer from the first GEMM epilogue's row maxima), "fused" (pair-of-CTAs
    256-row first GEMM only: conversion warps inside the first GEMM, plus a pass over what they left) or "rows" (one
    warp per row); MUSE_VR_MOE_X_QUANT selects."""
    if not (gemm1_rows_mode() and own_quantizer_applies()):
        return "stock" if not own_quantizer_applies() else "rows"
    form = os.environ.get("MUSE_VR_MOE_X_QUANT", "blocks").strip().lower()
    if form not in ("fused", "blocks", "rows"):
        form = "blocks"
    if form == "fused" and not (t1.two_cta and t1.tile_m == 256):
        form = "blocks"
    return form


def row_block_quant(t1: Tactic = None) -> bool:
    """The first GEMM's epilogue reduces each row's maximum (fused or row-block forms)."""
    return quant_form(t1 or DEFAULT_GEMM1) in ("fused", "blocks")


def compiled(t1: Tactic, t2: Tactic) -> bool:
    """Kernels for these tactics (and the row-kernel module, if used) are built in this process."""
    form = quant_form(t1)
    g1 = (({"fused": "gemm1_rows_fq", "blocks": "gemm1_rows_amax"}.get(form, "gemm1_rows"), t1) if gemm1_rows_mode()
          else ("gemm1", t1, False, True))
    ok = g1 in _cache and ("gemm2", t2) in _cache
    return ok and (not (own_quantizer_applies() or gemm1_rows_mode()) or "quant_module" in _cache)


def build(t1: Tactic, t2: Tactic) -> None:
    """Compile / load everything one call needs (outside CUDA graph capture)."""
    if gemm1_rows_mode():
        form = quant_form(t1)
        compile_gemm2(t1, swiglu=True, row_amax=form in ("fused", "blocks"), fuse_quant=form == "fused")
    else:
        compile_gemm1(t1)
    compile_gemm2(t2)
    if own_quantizer_applies() or gemm1_rows_mode():
        _quant_module()


def run_gemm1(a_fp4, a_sf, w13, w13_sf, alpha_gate, alpha_up, token_scale, tile_expert, tile_limit, row_to_expanded,
              num_tiles, out, tactic: Tactic):
    """out[:R, :I] = SwiGLU of the first GEMM for the sorted rows (R = out rows, I = w13 rows / 2)."""
    fn = compile_gemm1(tactic)
    num_experts, n, k_half = w13.shape
    addrs = [a_fp4.data_ptr(), w13.data_ptr(), a_sf.data_ptr(), w13_sf.data_ptr(), out.data_ptr(),
             alpha_gate.data_ptr(), alpha_up.data_ptr(), token_scale.data_ptr(), tile_expert.data_ptr(),
             tile_limit.data_ptr(), row_to_expanded.data_ptr(), num_tiles.data_ptr()]
    fn(*_gemm1_ptrs(addrs), a_fp4.shape[0], out.shape[0], n, k_half * 2, num_experts, stream=_stream())


def run_gemm2(a_fp4, a_sf, w2, w2_sf, alpha, row_scale, tile_expert, tile_limit, num_tiles, out, tactic: Tactic,
              alpha_up=None, row_amax=None, fused=None):
    """out[r, :] = alpha[e] * row_scale[r] * (A[r] @ W2[e]^T) for every valid sorted row r; with ``alpha_up`` the first
    GEMM's SwiGLU form (out has half the weight's rows as columns), with ``row_amax`` (SwiGLU form, zeroed int32 [rows])
    also each row's largest |output| as FP32 bits; with ``fused`` = (payload, swizzled scales, per-row scales, ctl)
    also the per-token NVFP4 form of the output where the epilogue can (see the kernel)."""
    swiglu = alpha_up is not None
    fn = compile_gemm2(tactic, swiglu=swiglu, row_amax=row_amax is not None, fuse_quant=fused is not None)
    num_experts, n, k_half = w2.shape
    fq = [t.data_ptr() for t in fused] if fused is not None else [row_scale.data_ptr()] * 4
    addrs = [a_fp4.data_ptr(), w2.data_ptr(), a_sf.data_ptr(), w2_sf.data_ptr(), out.data_ptr(), alpha.data_ptr(),
             row_scale.data_ptr(), (alpha_up if swiglu else alpha).data_ptr(),
             (row_amax if row_amax is not None else row_scale).data_ptr(), *fq, tile_expert.data_ptr(),
             tile_limit.data_ptr(), num_tiles.data_ptr()]
    fn(*_gemm2_ptrs(addrs), out.shape[0], n, k_half * 2, num_experts, stream=_stream())


def permute_rows(hs_fp4, hs_sf, token_scale, row_to_expanded, num_tiles, tile_limit, tile_m, out, out_sf, out_scale,
                 row_amax=None, ctl=None):
    """Sorted-row copy of the call's NVFP4 activations (payload, 128x4 swizzled block scales, per-token scale); zeroes
    row_amax (int32 [rows]) for the valid rows and the ctl words, if given."""
    empty = torch.empty((0,), dtype=torch.int32, device=out.device)
    _quant_module().permute_rows(hs_fp4, hs_sf.view(torch.uint8), token_scale, row_to_expanded, TOP_K, num_tiles,
                                 tile_limit, tile_m, out, out_sf, out_scale,
                                 row_amax if row_amax is not None else empty, ctl if ctl is not None else empty,
                                 _row_blocks(out, "permute"))


def quantize_rows(inter, row_map, out_fp4, out_sf, out_scale):
    """FlashInfer's per-token NVFP4 quantizer over the rows selected by row_map (expanded index -> sorted row).

    ``inter`` must have at least as many rows as the largest sorted row index + 1 and at least row_map.numel() rows.
    """
    from flashinfer import SfLayout
    from flashinfer.quantization.fp4_quantization import get_fp4_quantization_module

    from .arch import device_arch

    mod = get_fp4_quantization_module(device_arch()[3:])
    m = row_map.numel()
    mod.nvfp4_quant_and_per_token_scale_out_sm100(
        inter[:m], PER_TOKEN_SCALE_INV, out_fp4, out_sf, out_scale, row_map, SfLayout.layout_128x4.value)


_QUANT_ENV_OFF = ("FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH", "TRTLLM_DISABLE_FP4_QUANT_FAST_MATH", "FLASHINFER_NVFP4_4OVER6")


def own_quantizer_applies() -> bool:
    """The sorted-row quantizer reproduces the default recipe only (fast math, no 4over6); otherwise use FlashInfer's."""
    for var in _QUANT_ENV_OFF:
        if os.environ.get(var, "0").strip().lower() in ("1", "true", "yes", "on"):
            return False
    return os.environ.get("MUSE_VR_MOE_X_STOCK_QUANT", "0") != "1"


def _quant_module():
    """JIT module of csrc/moe_x_quant.cu, built with the flags FlashInfer uses for its own NVFP4 quantization module on
    this device (same architecture target, fast-math and feature defines)."""
    mod = _cache.get("quant_module")
    if mod is not None:
        return mod
    from pathlib import Path

    import flashinfer.jit.fp4_quantization as fp4jit
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import gen_jit_spec

    from .arch import device_arch

    digits = device_arch()[3:]
    ref = getattr(fp4jit, f"gen_fp4_quantization_sm{digits}_module")()
    flags = [f for f in ref.extra_cuda_cflags if f.startswith("-gencode") or f.startswith("-D")]
    csrc = jit_env.FLASHINFER_CSRC_DIR
    common = csrc / "nv_internal" / "cpp" / "common"
    src = Path(__file__).resolve().parent.parent / "csrc" / "moe_x_quant.cu"
    spec = gen_jit_spec(
        f"muse_vr_moe_x_quant_{digits}",
        [src, common / "envUtils.cpp", common / "logger.cpp", common / "stringUtils.cpp",
         common / "tllmException.cpp"],
        extra_cuda_cflags=flags,
        extra_cflags=["-DENABLE_BF16", "-DENABLE_FP8", "-DENABLE_FP4"],
        extra_include_paths=[csrc / "nv_internal", csrc / "nv_internal" / "include"],
    )
    mod = spec.build_and_load()
    _cache["quant_module"] = mod
    return mod


def _row_blocks(t, kind: str = "quant") -> int:
    """Grid of a row kernel: blocks per SM x SM count. Tunables: MUSE_VR_MOE_X_ROW_BLOCKS_PER_SM (row quantizer),
    MUSE_VR_MOE_X_BLOCKQ_BLOCKS_PER_SM (row-block quantizer) and MUSE_VR_MOE_X_PERMUTE_BLOCKS_PER_SM (activation copy)."""
    key = "row_blocks_" + kind
    blocks = int(_cache.get(key) or 0)
    if not blocks:
        props = torch.cuda.get_device_properties(t.device)
        var, default = {"permute": ("MUSE_VR_MOE_X_PERMUTE_BLOCKS_PER_SM", "8"),
                        "blocks": ("MUSE_VR_MOE_X_BLOCKQ_BLOCKS_PER_SM", "5"),
                        "regions": ("MUSE_VR_MOE_X_REGION_BLOCKS_PER_SM", "4")}.get(
                            kind, ("MUSE_VR_MOE_X_ROW_BLOCKS_PER_SM", "2"))
        blocks = props.multi_processor_count * int(os.environ.get(var, default))
        _cache[key] = blocks
    return blocks


def quantize_sorted_rows(inter, num_tiles, tile_limit, tile_m, out_fp4, out_sf, out_scale):
    """Per-token NVFP4 quantization of the valid sorted rows (one warp per row), default recipe only."""
    _quant_module().quant_sorted_rows(inter, num_tiles, tile_limit, tile_m, PER_TOKEN_SCALE_INV, out_fp4, out_sf,
                                      out_scale, _row_blocks(inter))


def quantize_row_blocks(inter, row_amax, num_tiles, tile_limit, tile_m, out_fp4, out_sf, out_scale):
    """Per-token NVFP4 quantization of the valid sorted rows from given row maxima (independent blocks), default recipe
    only."""
    _quant_module().quant_row_blocks(inter, row_amax, num_tiles, tile_limit, tile_m, PER_TOKEN_SCALE_INV, out_fp4,
                                     out_sf, out_scale, _row_blocks(inter, "blocks"))


def quantize_regions(inter, row_amax, ctl, num_tiles, tile_limit, tile_m, out_fp4, out_sf, out_scale):
    """Per-token NVFP4 conversion of the regions the fused first-GEMM epilogue left (done flag 0)."""
    _quant_module().quant_regions(inter, row_amax, ctl, num_tiles, tile_limit, tile_m, PER_TOKEN_SCALE_INV, out_fp4,
                                  out_sf, out_scale, _row_blocks(inter, "regions"))


def routed_experts(hs_fp4, hs_sf, per_token_scale, topk_ids, w13, w13_sf, w2, w2_sf, g1_gate, g1_up, g2,
                   *, num_experts: int, local_expert_offset: int, t1: Tactic = DEFAULT_GEMM1,
                   t2: Tactic = DEFAULT_GEMM2):
    """Deferred-finalize outputs (gemm2_out, expanded_idx_to_permuted_idx [T * 8]) of one routed-expert call.

    hs_fp4 [T, H/2] uint8, hs_sf [T, H/16] e4m3 (linear), per_token_scale [T] fp32, topk_ids [T, 8] int32 (global ids),
    w13 [E, 2I, H/2] / w2 [E, H, I/2] uint8 and their e4m3 block scales in the stock fused-MoE layout, g1_gate / g1_up /
    g2 [E] fp32 per-expert global scales.
    """
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
    from flashinfer.fused_moe.cute_dsl.moe_utils import moe_sort

    if t1.tile_m != t2.tile_m:
        raise ValueError("both GEMMs must use the same tile M (one sort)")
    tokens = hs_fp4.shape[0]
    num_local, n1, h_half = w13.shape
    hidden = h_half * 2
    inter_size = n1 // 2
    dev = hs_fp4.device
    tile_expert, tile_limit, e2p, p2e, _padded, num_tiles = moe_sort(
        token_selected_experts=topk_ids,
        token_final_scales=torch.empty(topk_ids.shape, dtype=torch.float32, device=dev),
        num_experts=num_experts,
        top_k=TOP_K,
        local_expert_offset=local_expert_offset,
        num_local_experts=num_local,
        tile_tokens_dim=t1.tile_m,
    )
    rows = p2e.shape[0]
    expanded = tokens * TOP_K
    own_quant = own_quantizer_applies()
    inter = torch.empty((rows if own_quant else max(rows, expanded), inter_size), dtype=torch.bfloat16, device=dev)
    form = quant_form(t1)
    blocks_quant = form in ("fused", "blocks")
    row_amax = torch.empty((rows,), dtype=torch.int32, device=dev) if blocks_quant else None
    a2 = torch.empty((rows, inter_size // 2), dtype=torch.uint8, device=dev)
    a2_sf = torch.empty((rows, inter_size // SF_VEC), dtype=torch.uint8, device=dev)
    a2_scale = torch.empty((rows,), dtype=torch.float32, device=dev)
    ctl = None
    if form == "fused":
        # [row-block counters | done flags | error word]
        ctl = torch.empty((rows // 128 * (1 + inter_size // 128) + 1,), dtype=torch.int32, device=dev)
    if gemm1_rows_mode():
        a1 = torch.empty((rows, hidden // 2), dtype=torch.uint8, device=dev)
        a1_sf = torch.empty((rows, hidden // SF_VEC), dtype=torch.uint8, device=dev)
        a1_scale = torch.empty((rows,), dtype=torch.float32, device=dev)
        permute_rows(hs_fp4, hs_sf, per_token_scale, p2e, num_tiles, tile_limit, t1.tile_m, a1, a1_sf, a1_scale,
                     row_amax, ctl)
        a1_sf_mma = convert_sf_to_mma_layout(a1_sf, m=rows, k=hidden, num_groups=1, sf_vec_size=SF_VEC)
        run_gemm2(a1, a1_sf_mma, w13, w13_sf.view(torch.uint8).reshape(num_local, -1), g1_gate, a1_scale, tile_expert,
                  tile_limit, num_tiles, inter[:rows], t1, alpha_up=g1_up, row_amax=row_amax,
                  fused=(a2, a2_sf, a2_scale, ctl) if ctl is not None else None)
    else:
        run_gemm1(hs_fp4, hs_sf, w13, w13_sf.view(torch.uint8).reshape(num_local, -1), g1_gate, g1_up,
                  per_token_scale, tile_expert, tile_limit, p2e, num_tiles, inter[:rows], t1)
    e2p_flat = e2p.view(-1)
    if form == "fused":
        if os.environ.get("MUSE_VR_MOE_X_KEEP_CTL") == "1":  # inspection of the conversion coverage (tests)
            _cache["last_ctl"] = ctl
        quantize_regions(inter[:rows], row_amax, ctl, num_tiles, tile_limit, t1.tile_m, a2, a2_sf, a2_scale)
    elif blocks_quant:
        quantize_row_blocks(inter, row_amax, num_tiles, tile_limit, t1.tile_m, a2, a2_sf, a2_scale)
    elif own_quant:
        quantize_sorted_rows(inter, num_tiles, tile_limit, t1.tile_m, a2, a2_sf, a2_scale)
    else:
        quantize_rows(inter, e2p_flat, a2, a2_sf, a2_scale)
    out = torch.empty((rows, hidden), dtype=torch.bfloat16, device=dev)
    a2_sf_mma = convert_sf_to_mma_layout(a2_sf, m=rows, k=inter_size, num_groups=1, sf_vec_size=SF_VEC)
    run_gemm2(a2, a2_sf_mma, w2, w2_sf.view(torch.uint8).reshape(num_local, -1), g2, a2_scale, tile_expert,
              tile_limit, num_tiles, out, t2)
    return out, e2p_flat
