# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Persistent routed experts for B<=64 (B>8: grouped units of <=8 tokens per expert)
over one shared core and model-owned workspace. The entry wait completes the
predecessor (incl. workspace rearm/counters) before shared state is read; output rows
keep the caller's top16 order."""

import functools
import logging
import os

import cuda.bindings.driver as cuda
import cutlass as c
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from ..common.host import check_residency, fake, stream, tensor_maps, to_cute
from ..common.policy import batch_table
from .experts import (
    MAX_CTAS,
    PLAN_BYTES,
    ExpertsGeometry,
    ExpertsGroupKernel,
    ExpertsKernel,
    ExpertsPlanKernel,
    cooperative,
    fc2_pairs,
    plan_b8,
    prewait,
)

logger = logging.getLogger(__name__)
_PLAN_B8_LOGGED = []
_NONCOOP_LOGGED = []
_PREWAIT_LOGGED = []


def tensor(value, align=16):
    return to_cute(value.reshape(-1), align)


def streamed(batch):
    """B<=8 on the incremental producer: ~1-1.5% at B4-8, neutral at B1-2, which keep
    the previous producer."""
    return bool(batch_table("4-8:1").get(batch, 0))


def pregrouped(tp, batch):
    """B>8 experts read the unit table a one-CTA kernel builds after routing, beside
    the fused shared expert (hidden at B9-32; at B64 its time after routing outlasts
    the shared expert)."""
    table = batch_table(PREGROUP[tp])
    return batch > 8 and bool(table.get(batch, 0))


PREGROUP = {8: "9-32:1", 16: "9-32:1"}
# With the shared join skipped (moe.prepared.JOIN), TP8 B16 releases at entry.
EARLY = {8: "9-16:1", 16: ""}


def early_group(tp, batch):
    """Pregrouped batches whose unit-table kernel releases the experts at its entry."""
    table = batch_table(EARLY[tp])
    return pregrouped(tp, batch) and bool(table.get(batch, 0))


def stealing(batch):
    """B>8 experts balance the FC2(B) tail by claims."""
    return batch > 8 and bool(batch_table("9-64:1").get(batch, 0))


@functools.cache
def compile_experts(
    tp, experts, batch, grid, device, publish=False, *knobs, activation_floor=None
):
    """Only code and geometry are cached; no owner tensors or descriptors. ``knobs``:
    (B<=8 stream, pregrouped, steal, B8 plan)."""
    defaults = (streamed(batch), False, False, False)
    stream, grouped, steal, planned = knobs + defaults[len(knobs) :]
    capacity = 1 << (batch - 1).bit_length()
    moe = ExpertsKernel(
        tp,
        capacity,
        experts,
        grid,
        publish,
        stream,
        grouped,
        steal,
        activation_floor=activation_floor,
        planned=planned,
    )
    flat = lambda n, dtype=c.Uint8, align=16: fake((n,), (1,), dtype, align)
    box = (c.Int64(0), c.Int32(0), c.Int32(0)) if publish else ()
    if publish == "twoshot":  # receive slots, owners' bases, parity x TP + rank
        box = (fake((3 * tp, 64, 3584 // tp)), fake((tp,), None, c.Int64), c.Int32(0))
    dtype = c.Float32 if tp == 8 else c.BFloat16
    x, ids = fake((batch, 3584), (3584, 1), dtype), flat(batch * 16, c.Int32)
    x = flat(PLAN_BYTES) if moe.grouped or moe.planned_x else x  # the MXFP8 plan
    scales = (flat(experts * n) for n in (moe.w13_scale_bytes, moe.w2_scale_bytes))
    out = flat(batch * 16 * (1 if publish else 3584), c.BFloat16)
    maps = flat(768 if moe.paired else 256, align=64)
    tensors = (x, ids, maps, *scales, flat(moe.workspace_bytes), out)
    scalars = (c.Int32(batch), *map(c.Float32, (4, 25, 0)), *box)
    with torch.accelerator.device_index(device):
        return cute.compile(moe, *tensors, *scalars, make_fake_stream())


@functools.cache
def compile_group(tp, experts, batch, grid, device, early=False):
    """The pregrouped experts' unit-table kernel: (ids, workspace, batch); ``early``:
    it releases the experts at entry."""
    capacity = 1 << (batch - 1).bit_length()
    kernel = ExpertsGroupKernel(tp, capacity, experts, grid, early=early)
    flat = lambda n, dtype=c.Uint8: fake((n,), (1,), dtype, 16)
    args = (flat(batch * 16, c.Int32), flat(kernel.workspace_bytes), c.Int32(batch))
    with torch.accelerator.device_index(device):
        return cute.compile(kernel, *args, make_fake_stream())


@functools.cache
def compile_plan(batch, device, fp32=False):
    """Plan writer for callers without the routing kernel's copy role in front."""
    dtype = c.Float32 if fp32 else c.BFloat16
    args = (
        fake((batch, 3584), None, dtype),
        fake((PLAN_BYTES,), (1,), c.Uint8),
        make_fake_stream(),
    )
    with torch.accelerator.device_index(device):
        return cute.compile(ExpertsPlanKernel(batch), *args)


def scratch(tp, device):
    """Model-owned expert buffers: the workspace and the grouped plan (MXFP8 x, written
    by the routing kernel or ``compile_plan``)."""
    size, u8 = (ExpertsGeometry(tp).workspace_bytes, torch.uint8)
    workspace = torch.full((size,), 255, device=device, dtype=u8)
    plan = torch.empty(PLAN_BYTES, device=device, dtype=u8)
    return {"workspace": workspace, "plan": plan}


# GB300 (152 SMs) grids. TP8: B3-4 on 148 CTAs; B5-8 publish on 128-136 CTAs, leaving
# SMs for the one-wave tail; grouped B9-64 on 152. TP16 B8 on 144 leaves SMs for the
# shared expert (8K/1K c1: faster decode than on 152).
GB300_SMS = 152
GB300_GRIDS = {8: (136, "3-4:148,5-6:128,7:132,9-64:152"), 16: (152, "8:144")}


def device_grids(tp, sms):
    """(default grid, per-batch table) for a device with ``sms`` SMs: GB300's tables
    with the same number of SMs left free beside the experts (so a 212-SM VR200 adds
    60 CTAs to every grid). ``K31MK_EXPERTS_GRID_SMS`` overrides the SM count the
    grids are derived for (e.g. 152 reproduces GB300's grids on any device)."""
    sms = int(os.environ.get("K31MK_EXPERTS_GRID_SMS", sms))
    default, spec = GB300_GRIDS[tp]
    extra = sms - GB300_SMS
    table = {b: min(g + extra, MAX_CTAS) for b, g in batch_table(spec).items()}
    return min(default + extra, MAX_CTAS), table


W2_UNPADDED = "K31MK_EXPERTS_W2_UNPADDED"


def fc2_rows(w2, tp):
    """The w2 rows the FC2 map reads. TP16 FC2 reads K 0-191 of each row as two
    overlapping 128-K boxes (chunk stride 64 elements), so ``K31MK_EXPERTS_W2_UNPADDED``
    swaps the padded rows (pitch 128 bytes, K 192 padded to 256) for an unpadded copy
    (pitch 96) the boxes read identically; the stock prefill MoE keeps the padded w2, so
    the copy is a second w2. Other TPs (and the default) read ``w2`` itself."""
    if tp == 16 and os.environ.get(W2_UNPADDED, "0") not in ("", "0"):
        return w2.view(torch.uint8)[..., :96].contiguous()
    return w2


class PersistentExperts:
    def __init__(
        self,
        w13,
        s13,
        w2,
        s2,
        *,
        tp=None,
        beta=4.0,
        linear_beta=25.0,
        activation_floor=None,
    ):
        experts, rows, _ = w13.shape
        # Per-rank w13 rows: 2 x 384 at TP8; TP16's 192 intermediate pads to 256.
        inferred = {768: 8, 512: 16}.get(rows)
        if inferred is None or tp not in (None, inferred):
            raise ValueError(f"experts w13 has {rows} rows per rank, not TP{tp}")
        self.tp = inferred
        self.device, self.experts = w13.device, experts
        self.weights = (w13, s13, w2, s2)
        self.activation_floor = False if activation_floor is None else activation_floor
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.grid, self.grids = device_grids(self.tp, sms)
        scalars = (beta, linear_beta, 0)
        self.scalars = tuple(map(c.Float32, scalars))
        # The experts kernel uses maps 0-1 (L2 promotion on w13); the TP8 B1 cluster
        # experts use maps 2-4 (w13, w2 two-chunk and one-chunk boxes, promoted); TP8
        # FC2 pairs (B4-64) use map 5 (w2 chunk 2 of 256 rows).
        # The FC2 map's rows; an opt-in unpadded copy at TP16 stays alive on the object.
        self.w2_rows = fc2_rows(w2, self.tp)
        w2_tile = (192, 3, 64) if self.tp == 8 else (self.w2_rows.shape[-1], 2, 32)
        specs = [
            (w13, rows, 1792, 28, 64, 2, True),
            (self.w2_rows, 3584, *w2_tile, 2, False),
        ]
        if self.tp == 8:
            specs += [specs[0]] + [(w2, 3584, 192, 3, 64, box, True) for box in (2, 1)]
        dtype = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B
        swizzle = cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
        l2 = cuda.CUtensorMapL2promotion
        l2 = (l2.CU_TENSOR_MAP_L2_PROMOTION_NONE, l2.CU_TENSOR_MAP_L2_PROMOTION_L2_256B)
        maps = [
            (weight.data_ptr(), dtype, (128, experts * n, chunks), (row_bytes, stride))
            + ((128, 128, box), swizzle, l2[promote])
            for weight, n, row_bytes, chunks, stride, box, promote in specs
        ]
        if self.tp == 8:
            shape = (128, experts * 3584, 3), (192, 64), (128, 256, 1)
            maps.append((w2.data_ptr(), dtype, *shape, swizzle, l2[False]))
        self.maps = tensor_maps(maps, self.device)
        scales = (tensor(s.view(torch.uint8)) for s in (s13, s2))
        self.views = (to_cute(self.maps[:256], 64), *scales)
        self.all_maps = to_cute(self.maps, 64)  # TP8 FC2 pairs: maps 0-5
        self.prepared = {}

    def prepare(
        self, batch, buffers, out, publish=(), planned=False, twoshot=False, early=False
    ):
        """``publish``: (multicast, parity, rank) of the WaveMoeTail latent slot to
        publish (``twoshot``: the ``common.twoshot`` push box); ``planned``: B>8 calls
        find the plan written (by the routing kernel), else run writes it from x first;
        ``early``: the caller runs ``group`` (pregrouped batches) before ``run``."""
        mode = "twoshot" if twoshot else bool(publish)
        grouped = pregrouped(self.tp, batch)
        planned_b8 = planned and plan_b8(self.tp, batch)
        if planned_b8 and not _PLAN_B8_LOGGED:  # once, not per layer
            _PLAN_B8_LOGGED.append(True)
            logger.info("NVIDIA megakernel: TP8 B8 routed experts read the plan")
        if not cooperative() and not _NONCOOP_LOGGED:  # engagement tag for A/B runs
            _NONCOOP_LOGGED.append(True)
            logger.info("NVIDIA megakernel: routed experts launch non-cooperatively")
        if prewait() and not grouped and not _PREWAIT_LOGGED:
            _PREWAIT_LOGGED.append(True)
            logger.info(
                "NVIDIA megakernel: routed experts route before the PDL wait (prewait)"
            )
        args: tuple = (batch, self.grid_for(batch), self.device.index, mode)
        args += (streamed(batch), grouped, stealing(batch), planned_b8)
        code = compile_experts(
            self.tp, self.experts, *args, activation_floor=self.activation_floor
        )
        # CTAs wait on other clusters' partials: the whole grid must be resident.
        check_residency(code, self.device.index, args[1], 384, "Kimi K3 MoE experts")
        ws, maps = tensor(buffers["workspace"]), self.views[0]
        plan: tuple | None = None
        capacity = 1 << (batch - 1).bit_length()
        if fc2_pairs(self.tp, capacity, batch > 8 or streamed(batch)):
            maps = self.all_maps
        group = None
        if batch > 8 or planned_b8:
            plan = (
                tensor(buffers["plan"]),
                compile_plan(batch, self.device.index, self.tp == 8),
            )
            plan = plan[:1] if planned else plan
        if grouped:
            release = early_group(self.tp, batch)
            grouping = compile_group(self.tp, self.experts, *args[:3], release)
            group = grouping, ws, early
        self.prepared[batch] = (out, tensor(out), ws, code, publish, plan, maps, group)

    def group(self, ids):
        """Pregrouped batches prepared ``early``: the unit-table kernel (run it on the
        stream after routing; ``run`` then skips it)."""
        group = self.prepared[ids.shape[0]][7]
        if group is not None and group[2]:
            self.launch_group(ids, group)

    def launch_group(self, ids, group):
        code, ws, _ = group
        code(tensor(ids), ws, c.Int32(ids.shape[0]), stream(self.device))

    def grid_for(self, batch):
        return self.grids.get(batch, self.grid)

    def run(self, x, ids, weights=None):
        batch = x.shape[0]
        out, output, workspace, code, box, plan, maps, group = self.prepared[batch]
        output = tensor(weights) if box else output
        source = to_cute(x, 16)
        if plan is not None:
            if len(plan) > 1:
                plan[1](source, plan[0], stream(self.device))
            source = plan[0]
        if group is not None and not group[2]:
            self.launch_group(ids, group)
        args = (source, tensor(ids), maps, *self.views[1:], workspace, output)
        code(*args, c.Int32(batch), *self.scalars, *box, stream(self.device))
        return out
