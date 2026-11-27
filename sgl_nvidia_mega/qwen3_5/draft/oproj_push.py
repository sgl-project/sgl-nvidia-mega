# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

SGLANG_READS = {
    "srt/models/qwen3_5.py": ("o_proj", "post_attention_layernorm"),
    "srt/layers/layernorm.py": ("weight", "variance_epsilon"),
    "srt/kernel_plugins/nvidia/eagle_binding.py|srt/megakernels/eagle_binding.py": (
        "tp_rank",
        "tp_size",
        "tp_cpu_group",
    ),
}

OPTION = "draft_oproj_push"
MIN_ROWS, MAX_ROWS = 2, 16
HIDDEN = 4096  # a hop mailbox row
CANDIDATES = (
    (8, 1, 1, 0, 1),
    (4, 1, 2, 0, 1),
    (8, 1, 2, 0, 1),
    (16, 1, 1, 0, 1),
    (4, 1, 1, 0, 1),
    (8, 2, 1, 1, 1),
)
RACE_COLD_BYTES = 384 << 20
PUSH_MODE = 1
_NOTED: set = set()


class OprojPush:
    """One rank's pushing o_proj launches by width, on the draft hop."""

    def __init__(self, weight: torch.Tensor, hop, specs: dict, desc: torch.Tensor):
        self.weight, self.hop, self.specs, self.desc = weight, hop, specs, desc
        # the pushed partial's stand-in (never written): the hop's key for "pushed"
        self.out = torch.empty(
            (max(specs) if specs else 1, int(weight.shape[0])),
            dtype=torch.bfloat16,
            device=weight.device,
        )

    def takes(self, rows: int) -> bool:
        return rows in self.specs

    def __call__(self, o: torch.Tensor) -> torch.Tensor:
        from sgl_nvidia_mega.qwen3_5.sm103 import trt_lowm

        rows = int(o.shape[0])
        x = o if o.is_contiguous() else o.contiguous()
        out = self.out[:rows]
        trt_lowm.run(
            self.specs[rows], out=out, x16=x, w_bf16=self.weight, push=self.desc
        )
        self.hop.mark_pushed(out)
        if torch.cuda.is_current_stream_capturing() and rows not in _NOTED:
            _NOTED.add(rows)
            s = self.specs[rows]
            logger.info(
                "NVIDIA megakernel: %s: the draft o_proj of %d rows runs on the low-M "
                "GEMM (tactic %s) and pushes its partial into the draft hop",
                OPTION,
                rows,
                (s.k_warps, s.n_tiles_bf16, s.ksplit, int(s.swap_ab)),
            )
        return out


def maybe_build(parts) -> Optional[OprojPush]:
    from sgl_nvidia_mega.qwen3_5 import runtime

    if not runtime.measured_option(OPTION):
        return None
    from sgl_nvidia_mega.qwen3_5.draft import MULTI_LOOP_BS
    from sgl_nvidia_mega.qwen3_5.draft.head_lowm import push_spec, race_push
    from sgl_nvidia_mega.qwen3_5.sm103 import trt_lowm

    b = parts.binding
    hop = parts.hop
    o_proj = parts.layer.o_proj
    w = getattr(o_proj, "weight", None)
    rows = tuple(
        r
        for r in MULTI_LOOP_BS
        if MIN_ROWS <= r <= MAX_ROWS and hop is not None and hop.hop.supports(r)
    )
    why = ""
    if hop is None:
        why = "slot draft_hop is off"
    elif (
        w is None
        or w.dtype != torch.bfloat16
        or w.dim() != 2
        or not w.is_contiguous()
        or int(w.shape[0]) != HIDDEN
        or int(w.shape[1]) % 512
    ):
        why = (
            "the draft o_proj is not a BF16 [4096, K] slice "
            f"({None if w is None else (w.dtype, tuple(w.shape))})"
        )
    elif getattr(o_proj, "bias", None) is not None:
        why = "the draft o_proj has a bias"
    elif not rows:
        why = "the draft hop serves none of the step widths 2-16"
    if not runtime.slot_supported(
        OPTION, why, cpu_group=getattr(b, "tp_cpu_group", None)
    ):
        return None
    site = hop.hop.site
    if int(site.world) != int(b.tp_size) or int(site.rank) != int(b.tp_rank):
        raise ValueError(
            f"draft o_proj push: the hop is rank {site.rank} of {site.world}, the "
            f"layer rank {b.tp_rank} of {b.tp_size}"
        )
    weight = w.detach()
    norm = parts.layer.post_attention_layernorm
    picks = race_push(
        weight,
        rows,
        candidates=CANDIDATES,
        cpu_group=b.tp_cpu_group,
        rank=int(b.tp_rank),
        world=int(b.tp_size),
        label="draft o_proj",
        consume=dict(
            gamma=norm.weight,
            eps=norm.variance_epsilon,
            weight_bias=1.0,
        ),
        cold_copies=max(1, RACE_COLD_BYTES // weight.nbytes),
    )
    n, k = (int(v) for v in weight.shape)
    specs = {}
    for m, (tac, release) in sorted(picks.items()):
        ps = push_spec(m, k, n, tac, release, PUSH_MODE)
        trt_lowm.prepare(ps)
        specs[m] = ps
    if not specs:
        logger.warning(
            "NVIDIA megakernel: %s stays off: no low-M tactic compiled", OPTION
        )
        return None
    desc = trt_lowm.push_descriptor(
        mailbox=int(site.mc),
        ctrl=int(site.ctrl),
        rank=int(site.rank),
        world=int(site.world),
        m_cap=int(site.m_cap),
        row_elems=n,
        col0=0,
        device=weight.device,
    )
    logger.info(
        "NVIDIA megakernel: %s: the draft o_proj ([%d, %d] slice of rank %d) pushes "
        "into the draft hop from its low-M GEMM at %s rows",
        OPTION,
        n,
        k,
        int(site.rank),
        sorted(specs),
    )
    return OprojPush(weight, hop, specs, desc)
