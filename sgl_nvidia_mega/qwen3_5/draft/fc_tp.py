# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)

SGLANG_READS = {
    "srt/models/qwen3_5_mtp.py": ("fc", "pre_fc_norm_embedding", "pre_fc_norm_hidden"),
    "srt/models/qwen3_5.py": ("input_layernorm",),
    "srt/layers/layernorm.py": ('"gemma_weight"', "weight", "variance_epsilon"),
    "srt/kernel_plugins/nvidia/eagle_binding.py|srt/megakernels/eagle_binding.py": (
        "tp_rank",
        "tp_size",
        "tp_cpu_group",
    ),
}

SLOT = "draft_fc_tp"
PUSH = "draft_fc_push"
# Rows the zero residual and the zero output block cover
MAX_ROWS = 128
LOWM_RACE_ROWS = 16
LOWM_CANDIDATES = (
    (8, 1, 8, 0, 1),
    (8, 1, 4, 0, 1),
    (4, 1, 8, 0, 1),
    (8, 1, 2, 0, 1),
    (8, 1, 1, 0, 1),
    (8, 2, 4, 1, 1),
)
LOWM_RACE_COLD_BYTES = 384 << 20


class FcTp:
    def __init__(
        self,
        *,
        weight_slice: torch.Tensor,
        table: torch.Tensor,
        norm_embedding,
        norm_hidden,
        input_norm,
        rank: int,
        world: int,
        hop=None,
        mnnvl: Optional[Callable[[int], object]] = None,
        lowm=None,
        max_rows: int = MAX_ROWS,
        push: int = 0,
        cpu_group=None,
    ):
        cols = int(weight_slice.shape[0]) if weight_slice.dim() == 2 else 0
        hidden = cols * world
        if (
            weight_slice.dtype != torch.bfloat16
            or weight_slice.dim() != 2
            or not weight_slice.is_contiguous()
            or not cols
            or int(weight_slice.shape[1]) != 2 * hidden
            or not 0 <= rank < world
        ):
            raise ValueError(
                f"draft fc TP: weight window {weight_slice.dtype} "
                f"{tuple(weight_slice.shape)} of rank {rank} at world {world}"
            )
        self.weight, self.table = weight_slice, table
        self.norm_embedding, self.norm_hidden = norm_embedding, norm_hidden
        self.input_norm = input_norm
        self.rank, self.world, self.hidden = rank, world, hidden
        self.c0, self.cols = rank * cols, cols
        self.hop = hop  # draft.hop.DraftHop, or None
        self.mnnvl = mnnvl
        # head_lowm.HeadLowM on the window (out_stride H), or None (cuBLAS at every width)
        self.lowm = lowm
        self.max_rows = int(max_rows)
        self.push_specs: dict = {}
        self.push_desc = None
        self.cpu_group = cpu_group
        if push and hop is not None and lowm is not None:
            self._build_push(int(push), weight_slice.device)
        self.zero = torch.zeros(
            (self.max_rows, hidden), dtype=torch.bfloat16, device=weight_slice.device
        )
        # The cross-rank sum's input: only this rank's columns are ever written
        self.block = torch.zeros(
            (self.max_rows, hidden), dtype=torch.bfloat16, device=weight_slice.device
        )
        self._noted: set = set()
        self.qkv_pdl: Optional[Callable[[int], bool]] = None

    def _build_push(self, mode: int, device) -> None:
        from sgl_nvidia_mega.qwen3_5.draft.head_lowm import push_spec, race_push
        from sgl_nvidia_mega.qwen3_5.sm103 import trt_lowm

        site = self.hop.hop.site
        if int(site.world) != self.world or int(site.rank) != self.rank:
            raise ValueError(
                f"draft fc TP push: the hop is rank {site.rank} of {site.world}, the "
                f"fc rank {self.rank} of {self.world}"
            )
        self.push_desc = trt_lowm.push_descriptor(
            mailbox=int(site.mc),
            ctrl=int(site.ctrl),
            rank=self.rank,
            world=self.world,
            m_cap=int(site.m_cap),
            row_elems=self.hidden,
            col0=self.c0,
            device=device,
        )
        picks = race_push(
            self.weight,
            tuple(r for r in self.lowm.specs if self.hop.hop.supports(r)),
            candidates=LOWM_CANDIDATES,
            cpu_group=self.cpu_group,
            rank=self.rank,
            world=self.world,
            label=f"{SLOT} fc window",
            consume=dict(
                gamma=self.input_norm.weight,
                eps=self.input_norm.variance_epsilon,
                weight_bias=1.0,
                cols=True,
            ),
            out_stride=self.hidden,
            col0=self.c0,
            cold_copies=max(1, LOWM_RACE_COLD_BYTES // self.weight.nbytes),
        )
        k = int(self.weight.shape[1])
        for rows, (tac, release) in sorted(picks.items()):
            ps = push_spec(rows, k, self.cols, tac, release, mode, self.hidden)
            trt_lowm.prepare(ps)
            self.push_specs[rows] = ps

    def ar_for(self, rows: int) -> str:
        if not 1 <= rows <= self.max_rows:
            return ""
        if self.hop is not None and self.hop.hop.supports(rows):
            return "hop"
        if self.mnnvl is not None and self.mnnvl(rows) is not None:
            return "mnnvl"
        return ""

    def _note(self, rows: int, line: str, warn: bool = False) -> None:
        # keyed by (rows, warn): a width that declined once still logs its engagement
        if torch.cuda.is_current_stream_capturing() and (rows, warn) not in self._noted:
            self._noted.add((rows, warn))
            (logger.warning if warn else logger.info)(
                "NVIDIA megakernel: slot %s %s", SLOT, line
            )

    def __call__(
        self, token_ids: torch.Tensor, hidden_states: torch.Tensor
    ) -> Optional[tuple[torch.Tensor, Optional[torch.Tensor]]]:
        from sgl_nvidia_mega.qwen3_5.draft.fc_input import mtp_fused_input

        ids = token_ids.reshape(-1)
        rows = int(ids.shape[0])
        how = self.ar_for(rows)
        if not how or int(hidden_states.shape[0]) != rows:
            why = (
                f"{int(hidden_states.shape[0])} hidden rows for {rows} tokens"
                if how
                else f"no fused all-reduce serves {rows} rows (hop 1-"
                f"{self.hop.hop.max_m if self.hop is not None else 0}, MNNVL "
                f"{'on' if self.mnnvl is not None else 'off'}, max {self.max_rows})"
            )
            self._note(
                rows,
                f"declines the MTP fc of {rows} rows: {why}; the replicated / split fc "
                "runs it",
                True,
            )
            return None
        full = mtp_fused_input(
            ids,
            self.table,
            hidden_states,
            self.norm_embedding.weight.data,
            self.norm_hidden.weight.data,
            self.norm_embedding.variance_epsilon,
            self.norm_hidden.variance_epsilon,
            **({"release_next": True} if self._early_input(rows) else {}),
        )
        window = self.block[:rows, self.c0 : self.c0 + self.cols]
        lowm = self.lowm is not None and self.lowm.takes(rows)
        block, zero = self.block[:rows], self.zero[:rows]
        push = how == "hop" and rows in self.push_specs
        if push:
            from sgl_nvidia_mega.qwen3_5.sm103 import trt_lowm

            trt_lowm.run(
                self.push_specs[rows],
                out=window,
                x16=full,
                w_bf16=self.weight,
                push=self.push_desc,
            )
            self.hop.mark_pushed(block)
        elif lowm:
            self.lowm(full, out=window)
        else:
            torch.mm(full, self.weight.t(), out=window)
        if how == "hop":
            normed, residual = self.hop.layer_input(
                block,
                zero,
                self.input_norm,
                cols=push,
                **({"release_next": True} if self._early_qkv(rows) else {}),
            )
        else:
            # the sum only: the layer runs its own input norm
            normed = None
            _, residual = self.mnnvl(rows).all_reduce_residual_rms_norm(
                local_contribution=block,
                residual=zero,
                gamma=self.input_norm.gemma_weight,
            )
        self._note(
            rows,
            f"runs the MTP fc of {rows} rows as the column-parallel fc "
            f"({'low-M' if lowm else 'cuBLAS'} [{rows}, {self.weight.shape[1]}] x "
            f"[{self.cols}, {self.weight.shape[1]}]^T, one bf16 rounding) + the "
            + (
                "draft hop all-reduce + input norm"
                if how == "hop"
                else "MNNVL CuTe DSL all-reduce (sum only) + the layer's input norm"
            )
            + (
                ", the GEMM pushing its columns into the hop (owner-slot consume)"
                if push
                else ""
            ),
        )
        return residual, normed

    def _early_input(self, rows: int) -> bool:
        from sgl_nvidia_mega.qwen3_5.draft import early

        if not early.on() or rows > early.MAX_ROWS:
            return False
        if self.lowm is not None and self.lowm.takes(rows):
            early.note(
                ("fc_input", rows),
                f"the fused input of {rows} rows releases the low-M fc window",
            )
            return True
        early.note(
            ("fc_input", rows), f"the fused input of {rows} rows keeps its release"
        )
        return False

    def _early_qkv(self, rows: int) -> bool:
        from sgl_nvidia_mega.qwen3_5.draft import early

        if not early.on() or rows > early.MAX_ROWS:
            return False
        if self.qkv_pdl is not None and self.qkv_pdl(rows):
            early.note(
                ("layer_input", rows),
                f"the layer-input consume of {rows} rows releases the qkv GEMM",
            )
            return True
        early.note(
            ("layer_input", rows),
            f"the layer-input consume of {rows} rows keeps its release",
        )
        return False


def maybe_build(parts) -> Optional[FcTp]:
    from sgl_nvidia_mega.qwen3_5 import runtime
    from sgl_nvidia_mega.qwen3_5.draft import MULTI_LOOP_BS
    from sgl_nvidia_mega.qwen3_5.draft.head_lowm import HeadLowM
    from sgl_nvidia_mega.qwen3_5.draft.layer import _mnnvl_workspace

    lm, b = parts.lm, parts.binding
    fc = lm.fc
    w = getattr(fc, "weight", None)
    world, rank = int(b.tp_size), int(b.tp_rank)
    table = parts.fc_input_table
    hidden = int(w.shape[0]) if w is not None and w.dim() == 2 else 0
    reason = ""
    if table is None:
        reason = "it reads slot draft_fc_input's replicated table"
    elif (
        w is None
        or w.dtype != torch.bfloat16
        or w.dim() != 2
        or tuple(w.shape) != (hidden, 2 * hidden)
    ):
        reason = (
            "the MTP fc is not a replicated BF16 [H, 2H] linear "
            f"({None if w is None else (w.dtype, tuple(w.shape))})"
        )
    elif getattr(fc, "bias", None) is not None:
        reason = "the MTP fc has a bias"
    elif world < 2 or hidden % world or (hidden // world) % 8:
        reason = (
            f"TP {world} does not split the fc's {hidden} output columns into whole "
            "8-column tiles"
        )
    elif parts.hop is None and _mnnvl_workspace(parts, 1) is None:
        reason = (
            "no fused all-reduce (slot draft_hop off and no MNNVL CuTe DSL workspace)"
        )
    elif not hasattr(parts.layer.input_layernorm, "gemma_weight"):
        reason = "the draft layer's input norm is not a GemmaRMSNorm"
    if not runtime.slot_supported(
        SLOT, reason, cpu_group=getattr(b, "tp_cpu_group", None)
    ):
        return None
    cols = hidden // world
    # this rank's output rows of the fc weight: a view of contiguous rows
    weight_slice = w.detach()[rank * cols : (rank + 1) * cols]
    if not weight_slice.is_contiguous():
        weight_slice = weight_slice.contiguous()
    lowm = HeadLowM(
        weight_slice,
        tuple(r for r in MULTI_LOOP_BS if r <= LOWM_RACE_ROWS),
        cpu_group=b.tp_cpu_group,
        rank=rank,
        max_rows=LOWM_RACE_ROWS,
        candidates=LOWM_CANDIDATES,
        label=f"{SLOT} fc window",
        cold_copies=max(1, LOWM_RACE_COLD_BYTES // weight_slice.nbytes),
        out_stride=hidden,
    )
    front = FcTp(
        weight_slice=weight_slice,
        table=table,
        norm_embedding=lm.pre_fc_norm_embedding,
        norm_hidden=lm.pre_fc_norm_hidden,
        input_norm=parts.layer.input_layernorm,
        rank=rank,
        world=world,
        hop=parts.hop,
        mnnvl=lambda rows: _mnnvl_workspace(parts, rows),
        lowm=lowm if lowm.specs else None,
        push=1 if runtime.measured_option(PUSH) else 0,
        cpu_group=b.tp_cpu_group,
    )
    from sgl_nvidia_mega.qwen3_5.draft import early

    if early.on():
        front.qkv_pdl = lambda rows, layer=parts.layer: early.qkv_takes_pdl(layer, rows)
    logger.info(
        "NVIDIA megakernel: slot %s built: the MTP fc's [%d, %d] output-row window of "
        "rank %d (output columns %d-%d), low-M at %s rows, all-reduce on the %s (hop "
        "1-%d rows, + input norm), up to %d rows; the GEMM pushes into the hop at %s rows",
        SLOT,
        cols,
        2 * hidden,
        rank,
        front.c0,
        front.c0 + cols - 1,
        sorted(lowm.specs),
        "draft hop, then MNNVL CuTe DSL" if parts.hop is not None else "MNNVL CuTe DSL",
        parts.hop.hop.max_m if parts.hop is not None else 0,
        front.max_rows,
        sorted(front.push_specs) if front.push_specs else "no",
    )
    return front
