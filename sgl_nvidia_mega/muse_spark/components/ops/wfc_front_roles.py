# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Front roles for a fused MoE front launch (csrc/wfc_front_roles.cuh): dev, not wired into the server.

The header provides device functions that a larger launch calls for the rows it owns:
  - route_row: SGLang's routing arithmetic for one row (one warp), then per-lane tagged ready words;
  - quant_row: SGLang's fused RMS norm + per-token NVFP4 quant for one row (a 256-thread group on a named barrier), then a
    tagged ready word for the row;
  - wait_word: a bounded consumer wait on one tagged word.
This module builds a test launch that runs both roles side by side in one CTA (quant groups + routing warps), a consumer
kernel that waits on every row's words, and an epoch bump (the tag is epoch * 64 + layer, read from device memory).
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("load_jit",),
    "kernels/jit/utils/compile/paths.py": ("KERNEL_PATH",),
}


_CSRC = Path(__file__).resolve().parents[1] / "csrc"
_CUH = _CSRC / "wfc_front_roles.cuh"


@functools.lru_cache(maxsize=None)
def module(q_threads: int = 256, groups: int = 1, route_warps: int = 4):
    from sglang.kernels.jit.utils import load_jit
    from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH

    from sgl_nvidia_mega.muse_spark.components.sgl_compat import csrc_include_paths

    name = f"WfcFrontRoles<{q_threads}, {groups}, {route_warps}>"
    return load_jit(
        "muse_vr_wfc_front_roles",
        str(q_threads),
        str(groups),
        str(route_warps),
        cuda_files=[str(_CUH)],
        cuda_wrappers=[
            ("run", f"{name}::run"),
            ("run_p", f"{name}::run_p"),
            ("wait", f"{name}::wait"),
            ("bump", f"{name}::bump"),
            ("plant", f"{name}::plant"),
            ("kfm_1a", f"{name}::kfm_1a"),
            ("kfm_1b", f"{name}::kfm_1b"),
        ],
        extra_include_paths=csrc_include_paths(KERNEL_PATH / "csrc") + [str(_CSRC)],
    )


class State:
    """Output buffers + tagged words for one row count; the epoch word persists across launches."""

    def __init__(self, m: int, device):
        d = device
        self.m = m
        self.topk_w = torch.zeros(m, 8, dtype=torch.float32, device=d)
        self.topk_ids = torch.zeros(m, 8, dtype=torch.int32, device=d)
        self.packed = torch.zeros(m, 8, dtype=torch.int32, device=d)
        self.rwords = torch.zeros(m * 8, dtype=torch.int64, device=d)
        self.fp4 = torch.zeros(m, 2048, dtype=torch.uint8, device=d)
        self.sf = torch.zeros(m, 256, dtype=torch.uint8, device=d)
        self.pts = torch.zeros(m, dtype=torch.float32, device=d)
        self.qwords = torch.zeros(m, dtype=torch.int64, device=d)
        self.epoch = torch.zeros(1, dtype=torch.int64, device=d)
        self.err = torch.zeros(1, dtype=torch.int64, device=d)
        self.got = torch.zeros(m, 9, dtype=torch.int32, device=d)
        self._dummy_i32 = torch.zeros(256, dtype=torch.int32, device=d)

    def run(
        self,
        logits,
        bias,
        x,
        weight,
        *,
        layer: int = 0,
        grid: int = 8,
        cfg=(256, 1, 4),
        mult: float = 1.0,
        route_scale: float = 1.0,
        div_eps: float = 0.0,
        qeps: float = 1e-6,
        gsi: float = 1.0 / (448.0 * 6.0),
        lem=None,
        nvalid=None,
    ):
        mod = module(*cfg)
        mod.run(
            logits,
            bias,
            self.topk_w,
            self.topk_ids,
            self.packed,
            lem if lem is not None else self._dummy_i32,
            int(lem is not None),
            nvalid if nvalid is not None else self._dummy_i32,
            int(nvalid is not None),
            self.rwords,
            x,
            weight if weight is not None else x[0],
            int(weight is not None),
            self.fp4,
            self.sf,
            self.pts,
            self.qwords,
            self.epoch,
            int(layer),
            self.m,
            int(grid),
            float(mult),
            float(route_scale),
            float(div_eps),
            float(qeps),
            float(gsi),
        )

    def run_p(
        self,
        logits,
        bias,
        partials,
        weight,
        *,
        layer: int = 0,
        grid: int = 8,
        cfg=(256, 1, 4),
        mult: float = 1.0,
        route_scale: float = 1.0,
        div_eps: float = 0.0,
        qeps: float = 1e-6,
        gsi: float = 1.0 / (448.0 * 6.0),
        pooled=None,
    ):
        """Partials form: partials [S, M, 4096] fp32 (contiguous); pooled: optional [M, 4096] bf16 output of the slice sum."""
        mod = module(*cfg)
        S = partials.shape[0]
        mod.run_p(
            logits,
            bias,
            self.topk_w,
            self.topk_ids,
            self.packed,
            self.rwords,
            partials,
            S,
            pooled if pooled is not None else self.fp4,
            int(pooled is not None),
            weight if weight is not None else self.fp4,
            int(weight is not None),
            self.fp4,
            self.sf,
            self.pts,
            self.qwords,
            self.epoch,
            int(layer),
            self.m,
            int(grid),
            float(mult),
            float(route_scale),
            float(div_eps),
            float(qeps),
            float(gsi),
        )

    def wait(self, *, layer: int = 0, max_polls: int = 1 << 20, cfg=(256, 1, 4)):
        module(*cfg).wait(
            self.rwords,
            self.qwords,
            self.m,
            self.epoch,
            int(layer),
            int(max_polls),
            self.err,
            self.got,
        )

    def bump(self, cfg=(256, 1, 4)):
        module(*cfg).bump(self.epoch)

    def plant(self):
        """Sanitizer non-vacuity control: the quant role with its first group barrier removed (a planted race)."""
        x = torch.randn(self.m, 4096, device=self.fp4.device).to(torch.bfloat16)
        module().plant(x, self.fp4, self.sf, self.pts, self.m)

    # K_FM entries (test launches): 1a writes the routed-ids board; 1b writes the expert-major token image.
    def kfm_1a(
        self,
        logits,
        bias,
        ids64,
        *,
        layer: int = 0,
        grid: int = 8,
        mult: float = 1.0,
        route_scale: float = 1.0,
        div_eps: float = 0.0,
    ):
        module().kfm_1a(
            logits,
            bias,
            ids64,
            self.epoch,
            int(layer),
            self.m,
            int(grid),
            float(mult),
            float(route_scale),
            float(div_eps),
        )

    def kfm_1b(
        self,
        src,
        weight,
        hs_perm,
        img1,
        row_img,
        pts_perm,
        row_tok,
        *,
        grid: int = 8,
        qeps: float = 1e-6,
        gsi: float = 1.0 / (448.0 * 6.0),
    ):
        """src: bf16 [M, 4096] pooled rows, or fp32 [S, M, 4096] partials (S = 1, 2)."""
        S = 0 if src.dim() == 2 else src.shape[0]
        module().kfm_1b(
            src,
            S,
            self.m,
            weight if weight is not None else src,
            int(weight is not None),
            hs_perm,
            img1,
            row_img,
            pts_perm,
            row_tok,
            int(row_tok.numel()),
            int(grid),
            float(qeps),
            float(gsi),
        )
