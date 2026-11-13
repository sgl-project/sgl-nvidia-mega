# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Owner-row boundary role (csrc/wfc_c4_norm.cuh): development form, not wired into the server.

SGLang's fused post-norm + gated residual + pre-norm (one CTA per row) as a device function for a 256-thread group that
plays the launch policy's thread count (256 below 192 rows, 512 from 192 rows), bitwise equal to SGLang's kernel at that
count; its outputs can also land in the front roles' input images (ops/wfc_c4f.py: pooler-input chunk image, router-input
A-fragment image). This module builds a test launch (one 256-thread CTA per row).
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("load_jit",),
    "kernels/jit/utils/compile/paths.py": ("KERNEL_PATH",),
    "kernels/ops/layernorm/muse_spark_norm_residual_jit.py": (
        "norm_residual_launch_config",
    ),
}


_CSRC = Path(__file__).resolve().parents[1] / "csrc"
_CUH = _CSRC / "wfc_c4_norm.cuh"


@functools.lru_cache(maxsize=None)
def module():
    import os

    from sglang.kernels.jit.utils import load_jit
    from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH

    from sgl_nvidia_mega.muse_spark.components.sgl_compat import csrc_include_paths

    plant = (
        os.environ.get("MUSE_VR_WFC_C4N_TEST_PLANT_CL", "") == "1"
    )  # tests only: the cluster form's planted race
    extra = {"extra_cuda_cflags": ["-DMUSE_VR_C4N_TEST_PLANT_CL"]} if plant else {}
    return load_jit(
        "muse_vr_wfc_c4_norm" + ("_plantcl" if plant else ""),
        cuda_files=[str(_CUH)],
        cuda_wrappers=[("run", "WfcC4Norm::run"), ("run_emit", "WfcC4Norm::run_emit")],
        extra_include_paths=csrc_include_paths(KERNEL_PATH / "csrc") + [str(_CSRC)],
        **extra,
    )


def served_threads(num_tokens: int) -> int:
    """The thread count SGLang's one-CTA-per-row launch policy picks for a step of num_tokens rows (hidden 8192)."""
    return 256 if num_tokens < 192 else 512


def run(
    branch,
    stream,
    *,
    post_eps,
    pre_eps,
    vt,
    input_scale=False,
    post_weight=None,
    pre_weight=None,
    alpha=None,
    beta=None,
    pre_norm_bf16=False,
    xp_img=None,
    xr_img=None,
    img_row0=0,
    R=0,
):
    """Returns (output bf16, new_stream fp32, router fp32); optionally writes rows [img_row0, img_row0 + R) into the images."""
    m = branch.shape[0]
    out = torch.empty_like(branch)
    new_stream = torch.empty_like(stream)
    router = torch.empty_like(stream)
    dummy = out[0]
    rpad = (R + 15) // 16 * 16
    module().run(
        branch,
        stream,
        new_stream,
        out,
        router,
        post_weight if post_weight is not None else dummy,
        int(post_weight is not None),
        pre_weight if pre_weight is not None else dummy,
        int(pre_weight is not None),
        alpha if alpha is not None else router[0],
        beta if beta is not None else router[0],
        int(alpha is not None),
        float(post_eps),
        float(pre_eps),
        int(bool(pre_norm_bf16)),
        int(bool(input_scale)),
        int(vt),
        xp_img if xp_img is not None else dummy,
        xr_img if xr_img is not None else router[0],
        int(img_row0),
        int(R),
        int(rpad),
        int(rpad // 16),
    )
    return out, new_stream, router


def run_emit(
    branch,
    stream,
    *,
    post_eps,
    pre_eps,
    vt,
    boundary,
    alpha,
    beta,
    post_weight=None,
    pre_weight=None,
    pre_norm_bf16=False,
    cs=1,
):
    """The emit-hook test launch in a model boundary form (boundary 0 = attention: no post weight, router outputs; 1 = FFN:
    input-scale post weight, no router), gate on; cs = 1: the one-CTA form at thread count vt; cs > 1: the cluster-order
    form (cs CTAs of vt threads, SGLang's small-M launch). Returns (output bf16, new_stream fp32, router fp32 or None, staged bf16):
    staged = each row's shared-memory staging copy written through the emit functor."""
    out = torch.empty_like(branch)
    new_stream = torch.empty_like(stream)
    router = torch.empty_like(stream)
    staged = torch.empty_like(branch)
    staged.view(torch.int16).fill_(
        0x7FC1
    )  # a NaN pattern: rows the functor never wrote cannot match out
    dummy = out[0]
    module().run_emit(
        branch,
        stream,
        new_stream,
        out,
        router,
        post_weight if post_weight is not None else dummy,
        pre_weight if pre_weight is not None else dummy,
        int(pre_weight is not None),
        alpha,
        beta,
        float(post_eps),
        float(pre_eps),
        int(bool(pre_norm_bf16)),
        int(vt),
        int(boundary),
        staged,
        int(cs),
    )
    return out, new_stream, (router if boundary == 0 else None), staged


def served_launch(num_tokens: int, hidden: int = 8192):
    """(cluster size, threads) of SGLang's fused norm-residual launch for a step of num_tokens rows, from SGLang's own
    policy function when the installed version has one (else the one-CTA policy)."""
    try:
        from sglang.kernels.ops.layernorm.muse_spark_norm_residual_jit import (
            norm_residual_launch_config,
        )
    except ImportError:
        return 1, served_threads(num_tokens)
    cs, threads, _ = norm_residual_launch_config(hidden, num_tokens)
    return int(cs), int(threads)
