# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Prepared MoE provider: one model owner serializes its scratch, no tensor pointers
are rebound at execution, and the side stream joins before the tail."""

import os

import torch

from ..common.policy import batch_table
from ..mkv2.experts_w4a4 import enabled as w4a4_enabled
from .experts_runtime import scratch as expert_scratch
from .planned import enabled as planned_enabled
from .planned import scratch as planned_scratch
from .shared import scratch as shared_scratch
from .tail_runtime import CAP as TAIL_CAP


class MoEScratch:
    def __init__(self, tp, device, batches):
        self.tp, self.device, self.batches = tp, torch.device(device), tuple(batches)
        cap, h = max(batches), 6144 // tp
        with torch.accelerator.device_index(self.device.index):
            self.stream = torch.cuda.Stream(device=self.device)
            self.begin, self.shared_done = torch.cuda.Event(), torch.cuda.Event()
            self.gate_done = torch.cuda.Event()  # the split shared expert's front half
            self.front_done = torch.cuda.Event()  # merged batches: the front kernel that leaves h
            f32, i32, bf16 = torch.float32, torch.int32, torch.bfloat16
            empty = lambda s, t=bf16: torch.empty(s, dtype=t, device=self.device)
            self.latent = empty((cap, 3584), f32)  # FP32 at every TP (stock's FP32 front; A4)
            self.front: tuple = (empty((cap, 896), f32), empty((cap, 2 * h), f32))
            self.front += (empty((cap, 16), i32), empty((cap, 16)), empty((cap, h)))
            self.front += (empty((cap, 7168)), self.latent)
            self.expanded = empty((cap * 16, 3584))
            self.experts = expert_scratch(tp, self.device)
            if planned_enabled() or w4a4_enabled():  # the planned / W4A4 engines' claim / call counters appended
                self.experts = planned_scratch(tp, self.device)
            # The experts' plan (MXFP8 x), written by the routing kernel at B>8.
            self.plan = self.experts["plan"]
            self.shared = shared_scratch(self.device)  # fused shared-expert counters
        self.views = {}
        for b in batches:
            self.views[b] = (self.latent[:b], self.expanded[: b * 16], self.experts)


# TP8 B16: the experts start beside the split gate/up half (+1% E2E).
JOIN = {8: "9-16:0", 16: ""}
# Batches whose split-K shared expert runs inline in the main stream's PDL chain, between the
# front's publish and the routing kernel's poll of the other ranks (the latent hop is in flight
# meanwhile), instead of on the side stream with a fork / join (``K31MK_MOE_SHARED_INLINE``,
# a batch table such as "8:1"; default off). Not for fused / split / merged shared experts (B>8).
INLINE = os.environ.get("K31MK_MOE_SHARED_INLINE", "")
# K31MK_MOE_RUBIN8 at B8: the every-SM ``RubinExpertsKernel`` CTAs spin on one another
# (K-split peer rows, staged rows) and the launch is not cooperative, so its liveness
# rests on every kernel that can share an SM with it finishing on its own. That holds
# for the side stream it was read with: the shared expert's gate/up split-K GEMM (or
# the same tactic as ``K31MK_EXP_SHARED_SKS``), SiTU, the down GEMM and the tail push,
# with no prefetch fork. Any other B8 configuration is refused.
RUBIN8_SHARED_TACTICS = ((64, 8, 4, 8), (128, 8, 1, 3))


def check_rubin8_corunners(moe):
    """Raise unless the B8 co-runners of the RUBIN8 experts are the enumerated set."""
    from .front import SHARED_TACTICS  # lazy: front imports the experts modules

    gate_up, down = (batch_table(spec).get(8) for spec in SHARED_TACTICS[384])
    problems = []
    if moe.prefetch is not None:
        problems.append("a prefetch fork beside the experts")
    if 8 in moe.inline:
        problems.append("K31MK_MOE_SHARED_INLINE at B8")
    if moe.front.merged_batch(8) or 8 in moe.front.split:
        problems.append("a fused / split / merged shared expert at B8")
    if (gate_up, down) != RUBIN8_SHARED_TACTICS:
        problems.append(f"shared expert tactics {gate_up} / {down} at B8")
    if problems:
        raise ValueError(
            "K31MK_MOE_RUBIN8: B8 co-runners outside the set its liveness argument covers: " + "; ".join(problems)
        )


class PreparedMoE:
    def __init__(self, front, experts, tail, state, *, prefetch=None, layer_idx=None):
        self.front, self.experts, self.tail, self.state = front, experts, tail, state
        self.prefetch, self.layer_idx = prefetch, layer_idx
        self.small_batch = None  # optional small-batch chain
        self.tap = None  # K31MK_MOE_ROUTE_TAP census (route_tap.RouteTap), set by the provider
        # Split batches whose routed experts launch beside the shared expert's
        # gate/up half instead of after it.
        self.join = batch_table(JOIN[state.tp])
        inline = batch_table(INLINE) if INLINE else {}
        self.inline = {b for b in state.batches if inline.get(b) and b <= 8}
        self.sms = torch.cuda.get_device_properties(state.device).multi_processor_count
        for b, sms in front.split.items():  # the split's down CTAs take the SMs left
            experts.grids[b] = min(experts.grid_for(b), self.sms - sms)
        # The experts publish the latent with a ``box_rows``-row slot stride (the multicast offset and the
        # two-shot receive layout are compiled with it): it must be the tail box's rows.
        slot_rows = getattr(experts, "box_rows", TAIL_CAP)
        if tail.publish and slot_rows != tail.tail_box.cap:
            raise ValueError(
                f"MoE tail capacity {tail.tail_box.cap}: the experts publish with a {slot_rows}-row slot stride "
                f"(batches {sorted(tail.publish)}); build them with box_rows={tail.tail_box.cap}"
            )
        for b in state.batches:
            _, expanded, buffers = state.views[b]
            publish, planned = tail.publish.get(b, ()), front.planned(b)
            twoshot = b in tail.twoshots
            experts.prepare(b, buffers, expanded, publish, planned, twoshot, True)
        if getattr(experts, "rubin8", None) is not None and 8 in state.batches:
            check_rubin8_corunners(self)

    def run(self, x, out, *, consume=None):
        state, b = self.state, x.shape[0]
        if self.small_batch is not None and b in self.small_batch.batches:
            return self.small_batch.run(out, consume)
        main = torch.cuda.current_stream(state.device)
        if b in self.inline:
            pushed = []

            def shared_expert():
                rows, beside = self.front.shared_expert(x, self.tail)
                assert beside is None, "inline shared expert: split-K path only"
                pushed.append(rows)

            ids, weights, latent = self.front.route(x, between=shared_expert)
            self.experts.group(ids)
            return self.finish(b, out, latent, ids, weights, pushed[0], consume, before=True)
        # The shared expert runs on the side stream beside the routed experts, or, when
        # the persistent expert grid spans every SM, beside the front only. Split: its
        # gate/up half beside the front, its down rows beside the experts.
        before = self.experts.grid_for(b) >= self.sms
        # K31MK_MOE_RUBIN8: its every-SM B8 grid keeps the split join (gate/up half first, the tail after all of it):
        # the shared expert's down rows then share the SMs with the experts instead of delaying their launch.
        if b == 8 and getattr(self.experts, "rubin8", None) is not None:
            before = False
        if self.front.merged_batch(b):
            # One kernel publishes the routing logits and leaves h; the down rows then
            # run beside the experts. The routing kernel after it is a PDL secondary that
            # does not wait for it (above 8 rows), so ``begin`` alone does not order h:
            # ``front_done``, recorded right after the front launch, gives the down rows a
            # full edge on the front kernel, which also orders the front's reads of x before
            # the tail (it waits ``shared_done``) and so before the next layer rewrites x.
            ids, weights, latent = self.front.route(x, between=lambda: state.front_done.record(main))
            state.begin.record(main)
            with torch.cuda.stream(state.stream):
                state.stream.wait_event(state.begin)
                state.stream.wait_event(state.front_done)
                shared, beside = self.front.shared_expert(x, self.tail)
                beside()
                state.shared_done.record(state.stream)
            self.experts.group(ids)
            return self.finish(b, out, latent, ids, weights, shared, consume)
        state.begin.record(main)
        with torch.cuda.stream(state.stream):
            state.stream.wait_event(state.begin)
            shared, beside = self.front.shared_expert(x, self.tail)
            if beside is not None:
                state.gate_done.record(state.stream)
                beside()
            state.shared_done.record(state.stream)
        ids, weights, latent = self.front.route(x)
        self.experts.group(ids)  # pregrouped unit table, beside the shared expert
        if before or (beside is not None and self.join.get(b, 1)):
            main.wait_event(state.shared_done if before else state.gate_done)
        return self.finish(b, out, latent, ids, weights, shared, consume, before)

    def finish(self, b, out, latent, ids, weights, shared, consume, before=False):
        """The routed experts, the next layer's prefetch fork, then the tail once the
        shared expert's side stream is done."""
        main = torch.cuda.current_stream(self.state.device)
        if self.tap is not None:  # census only: the copy delays the experts' early launch
            self.tap.record(self.layer_idx, ids)
        expanded = self.experts.run(latent, ids, weights)
        if self.prefetch is not None:
            self.prefetch.fork("next_attention", self.layer_idx, b)
        if not before:
            main.wait_event(self.state.shared_done)
        return self.tail.run(expanded, weights, shared, out, consume=consume)
