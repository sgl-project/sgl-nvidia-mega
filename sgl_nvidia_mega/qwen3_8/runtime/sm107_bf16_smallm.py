# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/layers/quantization/sm107_bf16_smallm.py (ROUTES,
# _cluster_smem, _cluster_ok, _tgvc_ok, _install_routes, _cluster_out, _tgvc_out,
# _tgv_out, route_for, gemm, precompile) and its call sites in
# python/sglang/srt/layers/quantization/unquant.py (initialize_bf16_gemm_config,
# precompile_splitk_tactics, _bf16_gemm_dispatch_unpinned).
# Changed: the switch is SGLANG_OPT_SM107_BF16_SMALLM_ROUTES (the source's is
# SGLANG_ENABLE_SM107_BF16_SMALLM_ROUTES). The routes serve the executor's TP1
# projection calls (``bf16_gemm_pins.linear`` and ``in_proj``, after a pin, as the
# source's dispatch consults them after its pins), not SGLang's BF16 dispatch; ``build``
# installs and checks them once a process, at the executor's build. The cluster routes run
# SGLang's dense_bf16_gemm_sm100_splitk_epilogue (no epilogue), which writes the bits of
# the source's vendored FlashInfer 0.7.2 split-K module at every "cl" key (checked on
# VR200); the source gates on that file's SHA-256 instead. The non-bitwise
# routes and the fused act are slot smallm_gemm's (runtime/smallm_gemm.py).
"""Bitwise small-M BF16 GEMM routes for SM107 (default off; SGLANG_OPT_SM107_BF16_SMALLM_ROUTES=1).

On SM107 SGLang's unquantized BF16 dispatch serves some small-M decode and verify GEMMs
of Qwen3.8 at TP1 as a cuBLAS split-K kernel plus a separate split-K reduce, or on a
slower non-split tile. Each key listed in ``ROUTES`` runs instead as one kernel whose
output bits equal the stock route's:

* ``"tgv"``: the CuTe DSL TGV kernel, which never splits K, at keys where stock runs a
  non-split cuBLAS kernel (the same sequential K order); tactic ``None`` is TGV's own pick.
* ``"tgvc"``: the TGV kernel at a tile outside its tactic table (a 2-CTA token tile
  covering every row), same K order.
* ``"cl"``: FlashInfer's cluster split-K kernel at cuBLAS's slicing (equal K slices,
  FP32 partials summed in rank order); tactic ``(mma_m, mma_n, split_k, stages, cta_k)``.

The source checked every key bit-equal to the stock dispatch on SM107 with cuBLAS 13.8
and FlashInfer 0.7.2, eagerly and under CUDA graph replay. ``build`` checks every key
once more against SGLang's live dispatch: a mismatch turns the routes off, and until
that check has passed the routes are not served. Keys where no bit-equal single kernel beat stock keep
the stock route: gdn_in_proj and attn_qkv at M 1-4, the shared expert's gate_up at M
17-32 and out_proj below M 64. Any other shape, a bias or a non-contiguous operand
takes SGLang's dispatch, and a BF16 pin at a key runs before its route
(``bf16_gemm_pins``), as in the source's dispatch.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)

SGLANG_READS = {
    "srt/layers/quantization/unquant.py": (
        "_bf16_gemm_dispatch_impl",
        "_enable_bf16_splitk_gemm",
        "_cutedsl_bf16_gemm",
        "_run_splitk_dense",
        "_splitk_tactic",
    ),
    "srt/runtime_context.py": ("get_parallel",),
    "kernels/ops/gemm/dense_bf16_gemm_sm100_splitk_epilogue.py": (
        "SplitKDenseGemmKernel",
        "SplitKTactic",
        "_SMEM_CAPACITY_BYTES",
        "_to_cute_swap",
        "_make_compile_repr_tensors",
        "_bmm_no_bias",
        "cute_ext",
    ),
    "kernels/ops/gemm/cutedsl_bf16_gemm.py": (
        "cutedsl_bf16_gemm",
        "_to_cute_swap",
        "_get_compiled_cute_ext_kernel",
        "_TGV_CUTE_EXT_CTA_K",
        "_run_tgv",
        "_pick_tactic",
    ),
}

ENV = "SGLANG_OPT_SM107_BF16_SMALLM_ROUTES"

_GDN_IN_PROJ = (16480, 2560)
_ATTN_QKV = (13312, 2560)
_ROUTER = (512, 2560)
_INDEX_QK = (640, 2560)
_SHARED_GATE_UP = (1280, 2560)
_OUT_PROJ = (2560, 6144)

# (M, N, K) -> (kind, tactic)
ROUTES = {
    **{(m, *_GDN_IN_PROJ): ("tgv", None) for m in (20, 24)},
    **{(m, *_GDN_IN_PROJ): ("tgv", 26) for m in (28, 32)},
    **{(m, *_ATTN_QKV): ("tgv", None) for m in (20, 24, 28, 32, 72, 80, 96)},
    **{(m, *_ATTN_QKV): ("tgvc", (64, 80, 8, True)) for m in (68, 76)},
    **{(m, *_ATTN_QKV): ("tgvc", (64, 96, 7, True)) for m in (84, 88, 92)},
    # cuBLAS 13.8 sums 4 K slices of 640 in rank order at M <= 16 and 8 slices of 320 at M 17-64 for these shapes.
    **{
        (m, *nk): ("cl", (64, 8, 4, 5, 128))
        for nk in (_ROUTER, _INDEX_QK)
        for m in (5, 6, 7, *range(9, 17))
    },
    **{
        (m, *nk): ("cl", (64, 16, 8, 5, 64))
        for nk in (_ROUTER, _INDEX_QK)
        for m in range(17, 33)
    },
    **{(m, *_SHARED_GATE_UP): ("cl", (64, 8, 4, 5, 128)) for m in range(1, 17)},
    # Stock runs TGV tactic 18 here; tactic 21 is the same unsplit K order.
    (64, *_OUT_PROJ): ("tgv", 21),
}

# Keys where the source's executed dispatch and SGLang main's differ. The source's dispatch has no low-M autotune: at
# the router's M 4 it runs its split-K path at its tuned tactic (64, 8, 4, 10), 8 x 4 CTAs. SGLang main's dispatch takes
# FlashInfer's direct dense GEMM there (512 CTAs), with other bits. These keys follow the source: each runs the cluster
# split-K kernel at the source's tactic and is checked bit for bit against SGLang's split-K kernel at that tactic
# (``source_gemm`` in ``precompile``), not against SGLang's dispatch.
SOURCE_SPLITK_ROUTES = {
    (4, *_ROUTER): ("cl", (64, 8, 4, 10, 64))
}  # 10 stages need the 64-wide K tile

_routes: Optional[dict] = None
_fi = None  # (the split-K module, CUstream constructor)
_compiled: dict = {}
_logged: set = set()
_checked = False  # set by precompile's check; routes serve only after it
_agreed = False  # set by enable once every rank built the routes
_built: Optional[str] = None  # build's result, once a process


def requested() -> bool:
    """The switch, read as SGLang's EnvBool at default False (an invalid value warns)."""
    value = os.environ.get(ENV)
    if value is None:
        return False
    if value.lower() in ("true", "1", "yes", "y"):
        return True
    if value.lower() not in ("false", "0", "no", "n"):
        logger.warning('Invalid value for %s: "%s", using default "False"', ENV, value)
    return False


def _cluster_smem(mma_m: int, mma_n: int, split_k: int, stages: int, cta_k: int) -> int:
    """FlashInfer 0.7.2 dense_bf16_gemm_sm100_splitk._smem_bytes with the K tile as a parameter."""

    def align(value: int, alignment: int) -> int:
        return (value + alignment - 1) // alignment * alignment

    cursor = align(mma_m * cta_k * 2 * stages, 1024) + mma_n * cta_k * 2 * stages
    cursor = align(align(cursor, 8) + 2 * stages * 8 + 3 * 8, 4) + 4
    if split_k == 1:
        return cursor
    return align(align(cursor, 128) + (split_k - 1) * mma_m * mma_n * 4, 8) + 8


def _cluster_ok(k: int, tactic: tuple, smem_capacity: int) -> bool:
    mma_m, mma_n, split_k, stages, cta_k = tactic
    return (
        k % (split_k * cta_k) == 0
        and 1 <= stages <= k // (split_k * cta_k)
        and _cluster_smem(*tactic) <= smem_capacity
    )


def _tgvc_ok(cta_m: int, cta_n: int, stages: int, use_2cta: bool) -> bool:
    step = 16 if use_2cta else 8
    smem = stages * (cta_m + cta_n // (2 if use_2cta else 1)) * 128 * 2 + 4096
    return (
        cta_m in (64, 128)
        and step <= cta_n <= 256
        and cta_n % step == 0
        and 2 <= stages
        and smem <= 227 * 1024
    )


def install(device_sm: int, splitk_enabled: bool, cutedsl_available: bool) -> bool:
    """Install the routes (served once ``precompile`` checked them); returns whether they are on."""
    global _routes, _fi, _checked
    _routes = None
    _checked = False
    if not (splitk_enabled and cutedsl_available and device_sm == 107):
        logger.warning(
            "%s needs SM107 and the cutedsl BF16 GEMM backend with split-K; the stock dispatch stays",
            ENV,
        )
        return False
    try:
        import cuda.bindings.driver as cuda_driver
        import sglang.kernels.ops.gemm.dense_bf16_gemm_sm100_splitk_epilogue as fi_splitk
    except ImportError as err:
        logger.warning(
            "%s: the split-K module is unavailable (%s); the stock dispatch stays",
            ENV,
            err,
        )
        return False
    table = {**ROUTES, **SOURCE_SPLITK_ROUTES}
    for (m, n, k), (kind, tactic) in table.items():
        if (kind == "tgvc" and not _tgvc_ok(*tactic)) or (
            kind == "cl" and not _cluster_ok(k, tactic, fi_splitk._SMEM_CAPACITY_BYTES)
        ):
            logger.warning(
                "%s: route %s %s at %s is invalid; the stock dispatch stays",
                ENV,
                kind,
                tactic,
                (m, n, k),
            )
            return False
    _fi = (fi_splitk, cuda_driver.CUstream)
    _routes = table
    kinds = [route[0] for route in table.values()]
    logger.info(
        "SM107 bitwise small-M BF16 GEMM routes on: %d keys (cluster split-K %d, TGV %d)",
        len(kinds),
        kinds.count("cl"),
        kinds.count("tgv") + kinds.count("tgvc"),
    )
    return True


def _cluster_out(
    x_2d: torch.Tensor, weight: torch.Tensor, out: torch.Tensor, tactic
) -> None:
    fi_splitk, stream_ctor = _fi
    mma_m, mma_n, split_k, stages, cta_k = tactic
    tensors = fi_splitk._to_cute_swap(x_2d, weight.T, out, None)
    key = (x_2d.get_device(), x_2d.dtype, tactic, tensors[4])
    compiled = _compiled.get(key)
    if compiled is None:
        # The kernel body reads only self.cta_k; the module's _CTA_K is host-side validation.
        kernel = fi_splitk.SplitKDenseGemmKernel(
            tactic=fi_splitk.SplitKTactic(mma_m, mma_n, split_k, stages),
            use_pdl=True,
            has_bias=False,
        )
        kernel.cta_k = cta_k
        with torch.cuda.device(x_2d.device):
            rep = fi_splitk._make_compile_repr_tensors(x_2d.dtype, False, *tensors[4])
            compiled = fi_splitk.cute_ext.compile(
                fi_splitk._bmm_no_bias,
                kernel,
                *rep[:3],
                stream_ctor(torch.cuda.current_stream().cuda_stream),
            )
        _compiled[key] = compiled
    compiled(
        *tensors[:3], stream_ctor(torch.cuda.current_stream(x_2d.device).cuda_stream)
    )


def _tgvc_out(
    x_2d: torch.Tensor, weight: torch.Tensor, out: torch.Tensor, tactic
) -> None:
    import cuda.bindings.driver as cuda_driver
    from sglang.kernels.ops.gemm import cutedsl_bf16_gemm

    cta_m, cta_n, stages, use_2cta = tactic
    a_, b_, c_, _, (a_leading, b_leading, c_leading) = cutedsl_bf16_gemm._to_cute_swap(
        x_2d, weight.t(), out, None
    )
    compiled = cutedsl_bf16_gemm._get_compiled_cute_ext_kernel(
        dtype=x_2d.dtype,
        c_dtype=out.dtype,
        cta_m=cta_m,
        cta_n=cta_n,
        cta_k=cutedsl_bf16_gemm._TGV_CUTE_EXT_CTA_K,
        num_ab_stage=stages,
        use_2cta=use_2cta,
        use_pdl=True,
        has_bias=False,
        a_leading=a_leading,
        b_leading=b_leading,
        c_leading=c_leading,
    )
    compiled(
        a_,
        b_,
        c_,
        cuda_driver.CUstream(torch.cuda.current_stream(x_2d.device).cuda_stream),
    )


def _tgv_out(
    x_2d: torch.Tensor, weight: torch.Tensor, out: torch.Tensor, tactic
) -> None:
    from sglang.kernels.ops.gemm import cutedsl_bf16_gemm

    m, k = x_2d.shape
    cutedsl_bf16_gemm._run_tgv(
        x_2d,
        weight.t(),
        None,
        out,
        True,
        (
            tactic
            if tactic is not None
            else cutedsl_bf16_gemm._pick_tactic(m, weight.shape[0], k)
        ),
    )


_RUNNERS = {"cl": _cluster_out, "tgvc": _tgvc_out, "tgv": _tgv_out}


def route_for(x: torch.Tensor, weight: torch.Tensor) -> Optional[tuple]:
    """The (kind, tactic) that serves ``x @ weight.T``, or None when the stock dispatch does."""
    if _routes is None or not _checked:
        return None
    k = x.shape[-1]
    m = x.numel() // k
    route = _routes.get((m, weight.shape[0], weight.shape[1]))
    if route is None or not (x.is_contiguous() and weight.is_contiguous()):
        return None
    if not (
        x.view(-1, k).stride() == (k, 1)
        and weight.stride() == (k, 1)
        and x.data_ptr() % 32 == 0
        and weight.data_ptr() % 32 == 0
    ):
        return None
    return route


def gemm(x: torch.Tensor, weight: torch.Tensor) -> Optional[torch.Tensor]:
    """``x @ weight.T`` from one kernel for a listed (M, N, K) with the stock route's bits, or None."""
    route = route_for(x, weight)
    if route is None:
        return None
    k = x.shape[-1]
    m = x.numel() // k
    kind, tactic = route
    out = torch.empty((m, weight.shape[0]), dtype=x.dtype, device=x.device)
    _RUNNERS[kind](x.view(-1, k), weight, out, tactic)
    if (kind, tuple(weight.shape)) not in _logged:
        _logged.add((kind, tuple(weight.shape)))
        logger.debug(
            "SM107 small-M route %s: m=%d n=%d k=%d tactic=%s",
            kind,
            m,
            weight.shape[0],
            k,
            tactic,
        )
    return out.view(*x.shape[:-1], weight.shape[0])


def precompile(
    *,
    stock_gemm: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    source_gemm: Optional[
        Callable[[torch.Tensor, torch.Tensor, tuple], torch.Tensor]
    ] = None,
    device=None,
) -> bool:
    """JIT-compile every key at load and check it bit for bit against ``stock_gemm`` (the dispatch with the routes
    off) on random operands, so a change on the stock side (cuBLAS's slicing) cannot go unnoticed; any mismatch turns
    the routes off. Returns whether the routes stay on."""
    global _routes, _checked
    if _routes is None:
        return False
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=device).manual_seed(0)
    _checked = True  # gemm() serves the routes during the check
    weights = {}
    for m, n, k in sorted(_routes):
        if (n, k) not in weights:
            weights[(n, k)] = (
                torch.randn(n, k, generator=gen, device=device) * 0.02
            ).to(torch.bfloat16)
        x = torch.randn(m, k, generator=gen, device=device).to(torch.bfloat16)
        got = gemm(x, weights[(n, k)])
        if (m, n, k) in SOURCE_SPLITK_ROUTES:
            # the source's split-K path at its tactic (the route's first four fields)
            want = (
                None
                if source_gemm is None
                else source_gemm(x, weights[(n, k)], _routes[(m, n, k)][1][:4])
            )
        else:
            want = stock_gemm(x, weights[(n, k)])
        if (
            got is None
            or want is None
            or not torch.equal(got.view(torch.int16), want.view(torch.int16))
        ):
            logger.warning(
                "%s: route %s at %s differs from the stock dispatch on this build; the stock dispatch stays",
                ENV,
                _routes[(m, n, k)],
                (m, n, k),
            )
            _routes = None
            _checked = False
            return False
    return True


# -- the executor's build ----------------------------------------------------------


def served() -> bool:
    """Whether ``gemm`` serves: installed, checked and agreed on every rank."""
    return _agreed and _routes is not None and _checked


def enable() -> None:
    """Every rank built the routes: the executor's projection calls take them."""
    global _agreed
    _agreed = True


def build(call_sites: dict) -> str:
    """'' when the routes are installed and checked against SGLang's dispatch (once a
    process; the first caller's ``call_sites``, (N, K) -> name, name them), else why not."""
    global _built
    if _built is None:
        _built = _build(call_sites)
    return _built


def _build(call_sites: dict) -> str:
    from sglang.srt.layers.quantization.unquant import (
        _bf16_gemm_dispatch_impl,
        _cutedsl_bf16_gemm,
        _enable_bf16_splitk_gemm,
    )
    from sglang.srt.runtime_context import get_parallel

    if get_parallel().tp_size != 1:
        return "TP1 only: a route replaces the whole projection"
    major, minor = torch.cuda.get_device_capability()
    if not install(
        device_sm=major * 10 + minor,
        splitk_enabled=_enable_bf16_splitk_gemm,
        cutedsl_available=_cutedsl_bf16_gemm is not None,
    ):
        return "not installed (see the warning above)"
    # Checked against SGLang's dispatch, which has no routes; a mismatch keeps them off.
    if not precompile(
        stock_gemm=lambda x, w: _bf16_gemm_dispatch_impl(x, w, None),
        source_gemm=_source_splitk,
    ):
        return "a route differs from SGLang's dispatch (see the warning above)"
    logger.info(
        "NVIDIA megakernel: Qwen3.8 SM107 small-M BF16 routes (%s) serve %d keys, each "
        "checked bit-equal to SGLang's dispatch, the source-dispatch keys to SGLang's split-K kernel at the "
        "source's tactic (a BF16 pin at a key runs first): %s",
        ENV,
        len(_routes),
        _summary(call_sites),
    )
    return ""


def _source_splitk(
    x: torch.Tensor, weight: torch.Tensor, tactic: tuple
) -> torch.Tensor:
    """SGLang's split-K kernel at ``tactic``, the path the source's dispatch runs at SOURCE_SPLITK_ROUTES."""
    from sglang.srt.layers.quantization.unquant import _run_splitk_dense, _splitk_tactic

    out = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    _run_splitk_dense(x, weight.T, None, out, True, _splitk_tactic(*tactic))
    return out


def _summary(call_sites: dict) -> str:
    """The served keys by call site and kind, M as ranges."""
    groups: dict = {}
    for m, n, k in sorted(_routes, key=lambda key: (key[1], key[2], key[0])):
        groups.setdefault((n, k), {}).setdefault(_routes[(m, n, k)][0], []).append(m)
    return "; ".join(
        f"{call_sites.get(nk, f'N {nk[0]} K {nk[1]}')} "
        + " / ".join(f"{kind} M {_ranges(ms)}" for kind, ms in kinds.items())
        for nk, kinds in groups.items()
    )


def _ranges(ms: list) -> str:
    spans = []
    for m in ms:
        if spans and m == spans[-1][1] + 1:
            spans[-1][1] = m
        else:
            spans.append([m, m])
    return ",".join(f"{a}-{b}" if b > a else f"{a}" for a, b in spans)
