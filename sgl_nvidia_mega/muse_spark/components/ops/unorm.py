# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Boundary norm with an early dependent release (csrc/unorm.cuh): development form, not wired into the server.

SGLang's fused post-norm + gated residual + pre-norm (its one-CTA-per-row path, used above 64 rows) with the same
per-row arithmetic, bitwise equal to SGLang's kernel at the thread count its launch policy picks for the row count, and
three launch-shape changes:
  - ``rows`` rows per CTA (a 256-thread group per row), so the launch can leave room on the GPU for the dependent grid;
  - the dependent grid is released at kernel entry (``trig=0``; 1 = after griddepcontrol.wait; 2 = at the end, as SGLang);
  - the static operands (channel-wise gate coefficients, norm weights) are copied to shared memory before
    griddepcontrol.wait (``static_pre_wait``). A caller must pass ``static_pre_wait=False`` whenever any of those tensors
    may have been written by work still in flight on the stream (for example the first call after the weight cache is
    built); then they are copied after the wait.
Rows <= 64 (SGLang's cluster path) and other gate / weight forms are not handled: ``supported`` returns False and the
caller keeps SGLang's op.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Optional, Tuple

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": (
        "is_arch_support_pdl",
        "load_jit",
        "make_cpp_args",
    ),
    "kernels/jit/utils/compile/paths.py": ("KERNEL_PATH",),
}


_CSRC = Path(__file__).resolve().parents[1] / "csrc"
_CUH = _CSRC / "unorm.cuh"
_H = 8192
_MAX_ROWS_PER_CTA = 4


@functools.lru_cache(maxsize=None)
def _use_pdl() -> bool:
    from sglang.kernels.jit.utils import is_arch_support_pdl

    return bool(is_arch_support_pdl())


@functools.lru_cache(maxsize=None)
def module(hidden: int, vt: int, rows: int, pdl: bool, trig: int):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args
    from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH

    from sgl_nvidia_mega.muse_spark.components.sgl_compat import csrc_include_paths

    negctl = os.environ.get(
        "MUSE_VR_UNORM_TEST_NEGCTL", ""
    )  # tests only (sanitizer negative controls)
    delay = (
        os.environ.get("MUSE_VR_UNORM_TEST_DELAY_G2", "") == "1"
    )  # tests only (late second copy group)
    flags = ([f"-DMUSE_VR_UNORM_NEGCTL={int(negctl)}"] if negctl else []) + (
        ["-DMUSE_VR_UNORM_TEST_DELAY_G2=1"] if delay else []
    )
    extra = {"extra_cuda_cflags": flags} if flags else {}
    args = make_cpp_args(hidden, vt, rows, pdl, trig)
    t = "sglang::muse_vr_unorm::UnormNormResidual"
    return load_jit(
        "muse_vr_unorm"
        + (f"_negctl{int(negctl)}" if negctl else "")
        + ("_delayg2" if delay else ""),
        *args,
        cuda_files=[str(_CUH)],
        cuda_wrappers=[
            ("run_std", f"{t}<{args}, false, false>::run"),
            ("run_std_router", f"{t}<{args}, false, true>::run"),
            ("run_scale", f"{t}<{args}, true, false>::run"),
            ("run_scale_router", f"{t}<{args}, true, true>::run"),
        ],
        extra_include_paths=csrc_include_paths(KERNEL_PATH / "csrc") + [str(_CSRC)],
        **extra,
    )


@functools.lru_cache(maxsize=None)
def module_cl(hidden: int, vt: int, rows: int, kk: int, pdl: bool, trig: int):
    """Column-split form (csrc/unorm_cl.cuh): a cluster of kk CTAs per group of ``rows`` rows."""
    from sglang.kernels.jit.utils import load_jit, make_cpp_args
    from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH

    from sgl_nvidia_mega.muse_spark.components.sgl_compat import csrc_include_paths

    negctl = os.environ.get(
        "MUSE_VR_UNORM_TEST_NEGCTL", ""
    )  # tests only (negative controls)
    bounded = (
        os.environ.get("MUSE_VR_UNORM_CL_TEST_BOUNDED", "") == "1"
    )  # tests only (bounded exchange wait)
    delay = (
        os.environ.get("MUSE_VR_UNORM_TEST_DELAY_G2", "") == "1"
    )  # tests only (late second copy group)
    flags = (
        ([f"-DMUSE_VR_UNORM_NEGCTL={int(negctl)}"] if negctl else [])
        + (["-DMUSE_VR_UNORM_CL_TEST_BOUNDED=1"] if bounded else [])
        + (["-DMUSE_VR_UNORM_TEST_DELAY_G2=1"] if delay else [])
    )
    extra = {"extra_cuda_cflags": flags} if flags else {}
    args = make_cpp_args(hidden, vt, rows, kk, pdl, trig)
    t = "sglang::muse_vr_unorm::UnormClNormResidual"
    return load_jit(
        "muse_vr_unorm_cl"
        + (f"_negctl{int(negctl)}" if negctl else "")
        + ("_bounded" if bounded else "")
        + ("_delayg2" if delay else ""),
        *args,
        cuda_files=[str(_CSRC / "unorm_cl.cuh")],
        cuda_wrappers=[
            ("run_std", f"{t}<{args}, false, false>::run"),
            ("run_std_router", f"{t}<{args}, false, true>::run"),
            ("run_scale", f"{t}<{args}, true, false>::run"),
            ("run_scale_router", f"{t}<{args}, true, true>::run"),
        ],
        extra_include_paths=csrc_include_paths(KERNEL_PATH / "csrc") + [str(_CSRC)],
        **extra,
    )


def stock_threads(num_tokens: int, hidden: int = _H) -> Optional[int]:
    """SGLang's thread count for this row count on its one-CTA-per-row path, or None when it launches clusters."""
    from sgl_nvidia_mega.muse_spark.components import sgl_compat

    nr = sgl_compat.module("sglang.kernels.ops.muse_spark_v12.norm_residual")
    cluster, threads, _prefetch = nr.norm_residual_launch_config(hidden, num_tokens)
    return int(threads) if cluster == 1 else None


def rows_per_cta(num_tokens: int, num_sms: int, successor_ctas: int) -> int:
    """Smallest rows per CTA that leaves ``successor_ctas`` SMs free for the dependent grid (one CTA per SM each)."""
    free = num_sms - successor_ctas
    if free <= 0:
        return _MAX_ROWS_PER_CTA
    return max(1, min(_MAX_ROWS_PER_CTA, -(-num_tokens // free)))


_synced: dict = {}  # (data_ptr, numel, dtype) -> (the tensor itself, its version when recorded)


def _key(t: torch.Tensor):
    return (t.data_ptr(), t.numel(), t.dtype)


def mark_static_synced(*tensors: torch.Tensor) -> None:
    """Record that these static operands are built AND every write to them has completed (a full device sync is done
    here). Only recorded tensors (same storage, same version) may be read before griddepcontrol.wait. The registry keeps
    a reference to each recorded tensor, so its memory cannot be handed to another tensor while recorded; a later write
    bumps the version (shared by views) and withdraws the permission."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("unorm: mark_static_synced must run outside graph capture")
    torch.cuda.synchronize()
    for t in tensors:
        _synced[_key(t)] = (t, t._version)


def static_is_synced(*tensors: torch.Tensor) -> bool:
    for t in tensors:
        e = _synced.get(_key(t))
        if e is None or e[0]._version != e[1] or t._version != e[1]:
            return False
    return True


def supported(
    num_tokens: int, hidden: int, alpha, beta, post_weight, pre_weight
) -> bool:
    if hidden != _H or num_tokens <= 0:
        return False
    if post_weight is None or pre_weight is None or alpha is None or beta is None:
        return False
    if alpha.numel() != hidden or beta.numel() != hidden:
        return False
    return stock_threads(num_tokens, hidden) is not None


def norm_residual(
    branch: torch.Tensor,
    stream: torch.Tensor,
    post_eps: float,
    pre_eps: float,
    *,
    post_norm_is_input_scale: bool,
    post_weight: torch.Tensor,
    pre_weight: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    want_router: bool = False,
    pre_norm_bf16: bool = False,
    rows: int = 1,
    trig: int = 0,
    static_pre_wait: bool = True,
    cluster: int = 1,
    prefetch_stream: bool = False,
    output: Optional[torch.Tensor] = None,
    new_stream_out: Optional[torch.Tensor] = None,
    router_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """``(output bf16, new_stream fp32, router fp32 or None)``, byte-equal to SGLang's fused_norm_residual."""
    assert branch.dtype == torch.bfloat16 and stream.dtype == torch.float32
    assert (
        branch.is_contiguous()
        and stream.is_contiguous()
        and stream.shape == branch.shape
    )
    m, hidden = branch.shape
    if not supported(m, hidden, alpha, beta, post_weight, pre_weight):
        raise ValueError(f"unorm: unsupported shape / form (M={m}, H={hidden})")
    assert rows >= 1 and trig in (0, 1, 2) and cluster in (1, 2, 4, 8)
    a1, b1 = alpha.reshape(-1), beta.reshape(-1)
    assert (
        a1.is_contiguous()
        and b1.is_contiguous()
        and post_weight.is_contiguous()
        and pre_weight.is_contiguous()
    )
    if static_pre_wait and not static_is_synced(a1, b1, post_weight, pre_weight):
        raise RuntimeError(
            "unorm: static_pre_wait=True needs mark_static_synced() on these operands after their last "
            "write; pass static_pre_wait=False instead"
        )
    vt_ = stock_threads(m, hidden)
    if rows * (256 if cluster == 1 else vt_ // cluster) > 1024 or (
        cluster > 1 and vt_ // cluster < 32
    ):
        raise ValueError(
            f"unorm: rows={rows} cluster={cluster} exceeds the CTA thread limit at M={m}"
        )
    if (
        -(-m // rows) * cluster
        > torch.cuda.get_device_properties(branch.device).multi_processor_count
    ):
        raise ValueError(
            "unorm: the launch must fit one wave (rows per CTA too small for this row count)"
        )
    vt = stock_threads(m, hidden)
    out = output if output is not None else torch.empty_like(branch)
    ns = new_stream_out if new_stream_out is not None else torch.empty_like(stream)
    rt = (
        (router_out if router_out is not None else torch.empty_like(stream))
        if want_router
        else None
    )
    mod = (
        module(hidden, vt, rows, _use_pdl(), trig)
        if cluster == 1
        else module_cl(hidden, vt, rows, cluster, _use_pdl(), trig)
    )
    fn = getattr(
        mod,
        ("run_scale" if post_norm_is_input_scale else "run_std")
        + ("_router" if want_router else ""),
    )
    fn(
        branch,
        stream,
        out,
        ns,
        rt,
        post_weight,
        pre_weight,
        a1,
        b1,
        float(post_eps),
        float(pre_eps),
        bool(pre_norm_bf16),
        bool(static_pre_wait),
        bool(prefetch_stream),
    )
    return out, ns, rt
