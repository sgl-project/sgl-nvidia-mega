# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project

"""Prepared MoE provider: one model owner serializes its scratch, no tensor pointers
are rebound at execution, and the side stream joins before the tail."""

import logging
import os

import torch

from ..common.policy import batch_table
from .experts_runtime import scratch as expert_scratch
from .shared import scratch as shared_scratch


class MoEScratch:
    def __init__(self, tp, device, batches):
        self.tp, self.device, self.batches = tp, torch.device(device), tuple(batches)
        cap, h = max(batches), 6144 // tp
        with torch.accelerator.device_index(self.device.index):
            self.stream = torch.cuda.Stream(device=self.device)
            self.begin, self.shared_done = torch.cuda.Event(), torch.cuda.Event()
            self.gate_done = torch.cuda.Event()  # the split shared expert's front half
            f32, i32, bf16 = torch.float32, torch.int32, torch.bfloat16
            empty = lambda s, t=bf16: torch.empty(s, dtype=t, device=self.device)
            self.latent = empty((cap, 3584), f32 if tp == 8 else bf16)
            self.front: tuple = (empty((cap, 896), f32), empty((cap, 2 * h), f32))
            self.front += (empty((cap, 16), i32), empty((cap, 16)), empty((cap, h)))
            self.front += (empty((cap, 7168)), self.latent)
            self.expanded = empty((cap * 16, 3584))
            self.experts = expert_scratch(tp, self.device)
            # The experts' plan (MXFP8 x), written by the routing kernel at B>8.
            self.plan = self.experts["plan"]
            self.shared = shared_scratch(self.device)  # fused shared-expert counters
        self.views = {}
        for b in batches:
            self.views[b] = (self.latent[:b], self.expanded[: b * 16], self.experts)


# TP8 B16: the experts start beside the split gate/up half (+1% E2E).
JOIN = {8: "9-16:0", 16: ""}
# Opt-in order of the split-K shared-expert chain (gate/up, SiTU, down, push): "side"
# (default) beside the front and the routed experts; "oproj": on the main stream right
# after the o-proj, with the front GEMM and routing beside it (vLLM's GB300 TP16 order);
# "after": gate/up + SiTU beside the front as in "side", the down rows and
# push on the main stream after the routed experts (every SM to the experts, then the
# down alone), before the tail. Same kernels and inputs in every order, so the same bits.
SHARED_ORDER = "SGLANG_NVIDIA_MEGAKERNEL_SHARED_ORDER"
logger = logging.getLogger(__name__)
_ORDER_LOGGED = []


class PreparedMoE:
    def __init__(self, front, experts, tail, state, *, prefetch=None, layer_idx=None):
        self.front, self.experts, self.tail, self.state = front, experts, tail, state
        self.prefetch, self.layer_idx = prefetch, layer_idx
        self.small_batch = None  # optional small-batch chain
        # Split batches whose routed experts launch beside the shared expert's
        # gate/up half instead of after it.
        self.join = batch_table(JOIN[state.tp])
        self.order = os.environ.get(SHARED_ORDER, "side") or "side"
        if self.order not in ("side", "oproj", "after"):
            raise ValueError(f"{SHARED_ORDER}={self.order!r}: unknown order")
        self.sms = torch.cuda.get_device_properties(state.device).multi_processor_count
        for b, sms in front.split.items():  # the split's down CTAs take the SMs left
            experts.grids[b] = min(experts.grid_for(b), self.sms - sms)
        if (
            self.order == "after"
        ):  # the down rows run after the experts: every SM to them
            for b in state.batches:
                if front.chained(b):
                    experts.grids[b] = experts.grid
        if self.order != "side" and not _ORDER_LOGGED:  # once, not per layer
            _ORDER_LOGGED.append(True)
            grids = {b: experts.grid_for(b) for b in state.batches}
            logger.info(
                "NVIDIA megakernel: shared expert order %s (%s), experts grids %s",
                self.order, SHARED_ORDER, grids,
            )  # fmt: skip
        for b in state.batches:
            _, expanded, buffers = state.views[b]
            publish, planned = tail.publish.get(b, ()), front.planned(b)
            twoshot = b in tail.twoshots
            experts.prepare(b, buffers, expanded, publish, planned, twoshot, True)

    def run(self, x, out, *, consume=None):
        state, b = self.state, x.shape[0]
        if self.small_batch is not None and b in self.small_batch.batches:
            return self.small_batch.run(out, consume)
        main = torch.cuda.current_stream(state.device)
        if self.order == "oproj" and self.front.chained(b):  # merged batches are fused
            return self.run_oproj(x, out, consume)
        if self.order == "after" and self.front.chained(b):
            return self.run_after(x, out, consume)
        # The shared expert runs on the side stream beside the routed experts, or, when
        # the persistent expert grid spans every SM, beside the front only. Split: its
        # gate/up half beside the front, its down rows beside the experts.
        before = self.experts.grid_for(b) >= self.sms
        if self.front.merged_batch(b):
            # One kernel publishes the routing logits and leaves h; the down rows then
            # run beside the experts.
            ids, weights, latent = self.front.route(x)
            state.begin.record(main)
            with torch.cuda.stream(state.stream):
                state.stream.wait_event(state.begin)
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

    def run_oproj(self, x, out, consume):
        """The shared expert's split-K chain on the main stream after the o-proj; the front
        GEMM, routing and the experts' unit table beside it, joined before the experts."""
        state, b = self.state, x.shape[0]
        main = torch.cuda.current_stream(state.device)
        state.begin.record(main)
        with torch.cuda.stream(state.stream):
            state.stream.wait_event(state.begin)
            ids, weights, latent = self.front.route(x)
            self.experts.group(ids)
            # gate_done is otherwise only the fused split batches' (never chained ones).
            state.gate_done.record(state.stream)
        shared, _ = self.front.shared_expert(x, self.tail)
        main.wait_event(state.gate_done)
        return self.finish(b, out, latent, ids, weights, shared, consume, True)

    def run_after(self, x, out, consume):
        """The shared expert's gate/up + SiTU on the side stream beside the front; the
        routed experts alone, then its down rows and push on the main stream once SiTU
        is done, then the tail."""
        state, b = self.state, x.shape[0]
        main = torch.cuda.current_stream(state.device)
        shared, gate, down = self.front.shared_halves(x, self.tail)
        state.begin.record(main)
        with torch.cuda.stream(state.stream):
            state.stream.wait_event(state.begin)
            gate()
            state.gate_done.record(state.stream)
        ids, weights, latent = self.front.route(x)
        self.experts.group(ids)
        expanded = self.experts.run(latent, ids, weights)
        if self.prefetch is not None:
            self.prefetch.fork("next_attention", self.layer_idx, b)
        main.wait_event(state.gate_done)
        down()
        return self.tail.run(expanded, weights, shared, out, consume=consume)

    def finish(self, b, out, latent, ids, weights, shared, consume, before=False):
        """The routed experts, the next layer's prefetch fork, then the tail once the
        shared expert's side stream is done."""
        main = torch.cuda.current_stream(self.state.device)
        expanded = self.experts.run(latent, ids, weights)
        if self.prefetch is not None:
            self.prefetch.fork("next_attention", self.layer_idx, b)
        if not before:
            main.wait_event(self.state.shared_done)
        return self.tail.run(expanded, weights, shared, out, consume=consume)
