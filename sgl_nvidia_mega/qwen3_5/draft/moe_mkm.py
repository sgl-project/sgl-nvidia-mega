# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The MTP draft's compatible NVFP4 experts on the target MoE engines.

Slot ``draft_moe`` keeps its one-row BF16 launch. From two through 64 rows, a draft
whose experts already have the target NVFP4 layout reuses the router front and the
7/16/32/64-token engines. Incompatible weights simply keep the normal draft path.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/qwen2_moe.py": ("experts", "layer_id"),
    "srt/layers/quantization/modelopt_quant.py": ('"w13_weight"', '"w2_weight"'),
}

MIN_ROWS = 2
MAX_ROWS = 64
# Past 12 rows the 32-token engine leaves room for the BF16 shared expert; its rows
# equal the 16-token engine bit for bit.
M16_MAX_ROWS = 12

# the MoE slots moe.step reads, all off but the ones a width takes
_SLOT_NAMES = (
    "route",
    "mkm",
    "route_topk",
    "route_gemm",
    "route_shared",
    "mkm_m32",
    "mkm_m64",
    "mkm_m128",
    "mkm_m256",
    "mkm_m512",
    "lowm_se",
    "hop_fp8",
)


class DraftMkm:
    """The draft layer's bound for moe.step and the per-width slots it takes."""

    def __init__(self, bound, wide: bool):
        self.bound = bound
        self.wide = wide
        self._noted: set = set()

    @classmethod
    def build(cls, mlp) -> tuple[Optional[DraftMkm], str]:
        from sgl_nvidia_mega.qwen3_5.moe import mkm, route
        from sgl_nvidia_mega.qwen3_5.sm103.moe import mkm as mkm_kernel
        from sgl_nvidia_mega.qwen3_5.sm103.moe import router

        w13 = getattr(getattr(mlp, "experts", None), "w13_weight", None)
        if not isinstance(w13, torch.Tensor) or w13.dtype != torch.uint8:
            return None, "routed experts are not in the target NVFP4 layout"
        reason = route.unsupported(mlp) or mkm.unsupported(mlp)
        if reason:
            return None, reason
        device = torch.cuda.current_device()
        intermediate = mlp.experts.w2_weight.shape[-1] * 2
        reason = mkm_kernel.m32_unsupported(device, intermediate)
        if reason:
            return None, f"32 / 64-token engines: {reason}"
        # compiled kernels are cached per shape: what the target built is reused, the rest added
        router.build(device, router.MAX_TOKENS)
        router.build_topk(device)
        mkm_kernel.build(device, intermediate, MAX_ROWS)
        wide = intermediate in mkm_kernel.WIDE_ENGINES
        maps = mkm_kernel.weight_maps(mlp.experts.w13_weight, mlp.experts.w2_weight)
        bound = route.Bound(
            route=True,
            mkm={mlp.layer_id: maps},
            mkm_wide=wide,
            identity=route.identity_rows(mkm_kernel.M128_MAX_TOKENS * mkm_kernel.TOP_K),
            topk=True,
            m32={mlp.layer_id: maps},
            m64=True,
            m16_max_rows=M16_MAX_ROWS,
        )
        return cls(bound, wide), ""

    def takes(self, rows: int, rows_per_request: int) -> bool:
        """Whether a draft step or extend of ``rows`` takes the target engines."""
        if rows_per_request == 1 and rows < MIN_ROWS:
            return False
        return 1 <= rows <= MAX_ROWS

    def state(self, rows: int):
        """moe.step's step state for ``rows`` rows: the MoE slots of that width, this bound."""
        from sgl_nvidia_mega.qwen3_5 import plan

        route = rows <= plan.ROUTE_MAX_ROWS
        on = dict.fromkeys(_SLOT_NAMES, False)
        on.update(
            route=route,
            mkm=route
            and rows <= (plan.MKM_WIDE_MAX_ROWS if self.wide else plan.MKM_MAX_ROWS),
            route_topk=not route,
            mkm_m32=not route and rows <= plan.MKM_M32_MAX_ROWS,
            mkm_m64=plan.MKM_M32_MAX_ROWS < rows <= plan.MKM_M64_MAX_ROWS,
        )
        if rows not in self._noted and torch.cuda.is_current_stream_capturing():
            self._noted.add(rows)
            logger.info(
                "NVIDIA megakernel: Qwen3.5 MTP draft MoE of %d rows on the "
                "target path: %s",
                rows,
                [k for k, v in on.items() if v],
            )
        return SimpleNamespace(
            slots=plan.Slots(**on), bound={"moe": self.bound}, scratch={}
        )

    def __call__(self, layer, x: torch.Tensor):
        """Return SGLang's deferred-finalize handoff for ``x``."""
        from sgl_nvidia_mega.qwen3_5 import moe

        return moe.step(
            layer, x, self.state(int(x.shape[0])), defer=True, reduce=_unreduced
        )


def _unreduced(rows: torch.Tensor) -> torch.Tensor:
    """The draft layer owns the tensor-parallel reduction after the MoE."""
    return rows
