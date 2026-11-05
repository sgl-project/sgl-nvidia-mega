# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""One-shot push all-reduce fused with the Muse Spark post-norm + gated residual + pre-norm boundary.

Replaces the pair ``CustomAllReduceV2.custom_all_reduce`` (algorithm ``1shot_push``) followed by
``sglang.kernels.ops.muse_spark_v12.norm_residual.fused_norm_residual`` with one kernel
(``csrc/ar_norm_residual.cuh``). The launch shape (cluster size, threads, weight prefetch) is taken from SGLang's
own ``norm_residual`` policy so each row is computed with the same thread map, and therefore the same floating-point
operation order, as the stock boundary kernel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch

from . import sgl_compat

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": (
        "is_arch_support_pdl",
        "load_jit",
        "make_cpp_args",
    ),
    "kernels/jit/utils/compile/paths.py": ("KERNEL_PATH",),
    "kernels/ops/layernorm/muse_spark_norm_residual_jit.py|kernels/ops/muse_spark_v12/norm_residual.py": (
        "_cluster_launch_config",
        "_cluster_size_for_shape",
    ),
    "srt/distributed/device_communicators/custom_all_reduce_v2.py": (
        "CustomAllReduceV2",
    ),
}


_CUH = Path(__file__).resolve().parent / "csrc" / "ar_norm_residual.cuh"
_MAX_WORLD = 8


def _cache(fn):
    from functools import lru_cache

    return lru_cache(maxsize=None)(fn)


@_cache
def _jit_module(hidden: int, world: int, threads: int, prefetch: bool, cluster: int):
    from sglang.kernels.jit.utils import is_arch_support_pdl, load_jit, make_cpp_args
    from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH

    args = make_cpp_args(
        hidden, world, threads, prefetch, cluster, is_arch_support_pdl()
    )
    return load_jit(
        "muse_vr_ar_norm_residual",
        *args,
        cuda_files=[str(_CUH)],
        cuda_wrappers=[
            ("run_std", f"MuseVrArNormResidual<{args}, false, false>::run"),
            ("run_std_router", f"MuseVrArNormResidual<{args}, false, true>::run"),
            ("run_scale", f"MuseVrArNormResidual<{args}, true, false>::run"),
            ("run_scale_router", f"MuseVrArNormResidual<{args}, true, true>::run"),
        ],
        extra_include_paths=sgl_compat.csrc_include_paths(KERNEL_PATH / "csrc"),
    )


@_cache
def _supports_clusters(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index)[0] >= 10


def launch_config(
    hidden_size: int, num_tokens: int, device_index: int, pre_norm_bf16: bool = False
):
    """(cluster size, threads, prefetch) exactly as SGLang's ``fused_norm_residual`` picks them."""
    from sglang.kernels.ops.muse_spark_v12.norm_residual import (
        _cluster_launch_config,
        _cluster_size_for_shape,
    )

    cluster = (
        _cluster_size_for_shape(hidden_size, num_tokens)
        if _supports_clusters(device_index) and not pre_norm_bf16
        else 1
    )
    threads, prefetch = _cluster_launch_config(hidden_size, num_tokens, cluster)
    return cluster, threads, prefetch


def supported(
    comm, num_tokens: int, hidden_size: int, pre_norm_bf16: bool = False
) -> bool:
    """Shape/communicator gate. The caller separately checks that stock would take ``1shot_push``."""
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

    if (
        not isinstance(comm, CustomAllReduceV2)
        or comm.disabled
        or not 2 <= comm.world_size <= _MAX_WORLD
        or num_tokens <= 0
        or num_tokens * hidden_size * 2 > comm.max_push_size
    ):
        return False
    try:
        cluster, _, _ = launch_config(
            hidden_size, num_tokens, comm.device.index or 0, pre_norm_bf16
        )
    except (RuntimeError, ValueError):
        return False
    return cluster == 1 or num_tokens * cluster <= comm.config.num_push_blocks


def ar_norm_residual(
    comm,
    partial: torch.Tensor,
    stream: torch.Tensor,
    post_eps: float,
    pre_eps: float,
    *,
    post_norm_is_input_scale: bool = False,
    post_weight: Optional[torch.Tensor] = None,
    pre_weight: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    want_router: bool = False,
    pre_norm_bf16: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Return ``(output bf16, new_stream fp32, router fp32 or None)`` for the all-reduced ``partial``."""
    assert partial.dtype == torch.bfloat16 and stream.dtype == torch.float32
    assert partial.dim() == 2 and stream.shape == partial.shape
    assert partial.is_contiguous() and stream.is_contiguous()
    m, hidden = partial.shape
    output = torch.empty_like(partial)
    new_stream = torch.empty_like(stream)
    router = torch.empty_like(stream) if want_router else None
    if m == 0:
        return output, new_stream, router
    cluster, threads, prefetch = launch_config(
        hidden, m, partial.device.index or 0, pre_norm_bf16
    )
    module = _jit_module(hidden, comm.world_size, threads, prefetch, cluster)
    fn = getattr(
        module,
        ("run_scale" if post_norm_is_input_scale else "run_std")
        + ("_router" if want_router else ""),
    )
    fn(
        comm.obj,
        partial,
        stream,
        output,
        new_stream,
        router,
        post_weight,
        pre_weight,
        alpha,
        beta,
        float(post_eps),
        float(pre_eps),
        bool(pre_norm_bf16),
    )
    return output, new_stream, router
