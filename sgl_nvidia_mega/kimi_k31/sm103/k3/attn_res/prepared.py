# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Prepared seven-CTA AttnRes consumers for ordinary or Lamport mailbox input."""

import logging
import os
from functools import cache
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from ..common.host import (
    architecture,
    dynamic_fake,
    dynamic_view,
    fake,
    prepare_shared_carveout,
    stream,
    to_cute,
)
from .aux import AuxMixture
from .kernel import ClusterAttnRes

# Side-stream aux mixtures not yet joined: (done event, check pair or None). AuxMixture reads
# the prefix and block rows its consumer just wrote; the next writer of those rows (the next
# consumer, or the o-proj's fused post-attention AttnRes) must wait for them (``join_aux``).
_PENDING_AUX: list = []
# ``K31MK_AUX_SIDE_CHECK=N`` (debug, with the side stream): every side mixture is also computed
# in today's serial order (on the main stream, right after the main kernel) into a shadow slice,
# and the two are byte-compared at the join. A mismatch fails the step with a device-side assert,
# so every CUDA-graph replay is checked; the first N eager joins per process log their counts.
AUX_SIDE_CHECK = int(os.environ.get("K31MK_AUX_SIDE_CHECK", "0") or 0)
_CHECKS_LOGGED = [0]
_CHECK_ON_LOGGED: list = []
logger = logging.getLogger(__name__)
# ``K31MK_ATTNRES_CARVEOUT=N`` (off by default; 0 = off): a carve-out of N percent of shared memory for the
# AttnRes consumer kernels, so the next layer's in-projection CTAs can start loading weights beside the AttnRes CTAs
# before their PDL wait. A launch preference only (no change to the kernel or its synchronization). The attribute is
# set on the compiled function, so the knob compiles its own cached copy (``carveout`` in compile_consumer's key): a
# knob-off compile in the same process never carries it.
_CARVEOUT_LOGGED: list = []


def attnres_carveout():
    """K31MK_ATTNRES_CARVEOUT as a percentage, or None when 0 (off)."""
    value = os.environ.get("K31MK_ATTNRES_CARVEOUT", "")
    if value in ("", "0"):
        return None
    percent = int(value)
    if not 0 <= percent <= 100:
        raise ValueError(f"K31MK_ATTNRES_CARVEOUT={value}: want a percentage 0-100")
    return percent


@cache
def _aux_stream(device):
    return torch.cuda.Stream(device=device)


def join_aux(stream):
    """Make ``stream`` wait for every side-stream aux mixture launched so far."""
    for done, check in _PENDING_AUX:
        stream.wait_event(done)
        if check is not None:
            with torch.cuda.stream(stream):
                check_aux(*check)
    _PENDING_AUX.clear()


def check_aux(side, shadow, fail=True):
    """BF16 words where the side-stream mixture ``side`` differs from its serial ``shadow``
    (a device scalar); ``fail``: device-side assert on any difference."""
    bad = (side.view(torch.int16) != shadow.view(torch.int16)).sum()
    if fail:
        torch._assert_async(bad == 0)
    if (
        not torch.cuda.is_current_stream_capturing()
        and _CHECKS_LOGGED[0] < AUX_SIDE_CHECK
    ):
        _CHECKS_LOGGED[0] += 1
        logger.info(
            "NVIDIA megakernel: aux side check %d: %d of %d BF16 words differ",
            _CHECKS_LOGGED[0],
            int(bad),
            side.numel(),
        )
    return bad


def _consumer_args(slots, cap, aux):
    row, weight, eps = dynamic_fake(2), fake((7168,), (1,)), cutlass.Float32(1e-6)
    source = fake((slots, cap, 7168), (cap * 7168, 7168, 1)) if slots else row
    args = (source, row, dynamic_fake(3), weight, weight, weight, row, eps, eps)
    return args + (row,) * aux  # the updated prefix row, also stored to an aux stream


@cache
def compile_consumer(
    nb,
    delta,
    write,
    final,
    slots,
    cap,
    arch,
    deadline_ns,
    aux=False,
    mixture=False,
    aux_launch=True,
    carveout=None,
):
    # ``carveout`` only keys the cache: a K31MK_ATTNRES_CARVEOUT compile is its own copy (its own function attribute).
    kernel = ClusterAttnRes(
        nb,
        slots=slots,
        has_delta=delta,
        write=write,
        final=final,
        deadline_ns=deadline_ns,
        aux_mixture=mixture,
        aux_launch=aux_launch,
    )
    args = _consumer_args(slots, cap, aux)
    return cute.compile(kernel, *args, make_fake_stream(), options=f"--gpu-arch={arch}")


@cache
def compile_aux_mixture(nb, slots, cap, arch):
    """``AuxMixture`` alone (score + combine) over a consumer's argument list."""
    args = _consumer_args(slots, cap, True)
    return cute.compile(
        AuxMixture(nb), *args, make_fake_stream(), options=f"--gpu-arch={arch}"
    )


class PreparedAttnRes:
    """Ordinary-input code plus the Lamport variant of its bound mailbox ring."""

    def __init__(self, weights, key, eps, deadline_ns):
        self.device = weights[0].device
        self.views = tuple(to_cute(w.reshape(-1)) for w in weights)
        self.key, self.eps = key, tuple(map(cutlass.Float32, eps))
        self.deadline_ns = deadline_ns
        self.code = self._compile(0, 0)
        self.ring: Any = None
        self.aux: Any = None
        self.aux_side: Any = (
            None  # (plain, ring) AuxMixture codes when it runs on a side stream
        )

    def _compile(self, slots, cap, aux=False, mixture=False, aux_launch=True):
        arch = architecture(self.device)
        value = attnres_carveout()
        if value is not None:
            code = compile_consumer(
                *self.key,
                slots,
                cap,
                arch,
                self.deadline_ns,
                aux,
                mixture,
                aux_launch,
                carveout=value,
            )
            if (
                not _CARVEOUT_LOGGED
            ):  # use-time (the consumer the layer runs), once per process
                _CARVEOUT_LOGGED.append(True)
                logger.info(
                    "NVIDIA megakernel: K31MK_ATTNRES_CARVEOUT=%d engaged (AttnRes consumers keep a %d %% "
                    "shared-memory carve-out)",
                    value,
                    value,
                )
            return prepare_shared_carveout(code, self.device.index, value)
        return compile_consumer(
            *self.key, slots, cap, arch, self.deadline_ns, aux, mixture, aux_launch
        )

    @classmethod
    def for_target(cls, target, *, deadline_ns, final=False, has_delta=True):
        if final:
            norm, query = target.output_attn_res_norm, target.output_attn_res_proj
            output, nb, write = target.norm, target.num_attn_res_blocks, -1
        else:
            norm = target.self_attention_res_norm
            query, output = target.self_attention_res_proj, target.input_layernorm
            nb = target.prev_valid_blocks
            write = target.block_write_idx if target.is_block_write_layer else -1
        weights = (norm.weight, query.weight, output.weight)
        eps = (norm.variance_epsilon, output.variance_epsilon)
        return cls(weights, (nb, has_delta, write, final), eps, deadline_ns)

    def bind_ring(self, mailbox):
        """Lamport input: sum the [slots, capacity, 7168] mailbox rows, then re-arm."""
        shape = tuple(mailbox.shape[:2])
        self.ring = self._compile(*shape), to_cute(mailbox), shape

    def bind_aux(self, *, mixture=False, side=False):
        """Variants that also store the updated prefix rows (a stock aux stream). ``side``: a
        mixture over saved blocks runs on a side stream after the main kernel (``join_aux``)."""
        side = side and mixture and self.key[0] > 0
        arch, ring, side_ring = architecture(self.device), None, None
        if self.ring is not None:
            slots, cap = self.ring[2]
            ring = self._compile(slots, cap, True, mixture, not side)
            if side:
                side_ring = compile_aux_mixture(self.key[0], slots, cap, arch)
        self.aux = self._compile(0, 0, True, mixture, not side), ring
        if side:
            self.aux_side = compile_aux_mixture(self.key[0], 0, 0, arch), side_ring
            if (
                AUX_SIDE_CHECK and not _CHECK_ON_LOGGED
            ):  # once per process, not per consumer
                _CHECK_ON_LOGGED.append(True)
                logger.info(
                    "NVIDIA megakernel: aux side check on (K31MK_AUX_SIDE_CHECK=%d)",
                    AUX_SIDE_CHECK,
                )

    def run_ring(self, prefix, blocks, out, aux=None):
        code = self.ring[0] if aux is None else self.aux[1]
        side = None if aux is None or self.aux_side is None else self.aux_side[1]
        return self._launch(code, self.ring[1], prefix, blocks, out, aux, side)

    def run(self, delta, prefix, blocks, out, aux=None):
        code = self.code if aux is None else self.aux[0]
        side = None if aux is None or self.aux_side is None else self.aux_side[0]
        return self._launch(code, dynamic_view(delta), prefix, blocks, out, aux, side)

    def _launch(self, code, source, prefix, blocks, out, aux=None, side=None):
        main = torch.cuda.current_stream(self.device)
        join_aux(main)  # this launch rewrites the rows an earlier side mixture reads
        aux_rows = aux
        rows = (prefix, blocks, out) + (() if aux is None else (aux,))
        prefix, blocks, view, *aux = map(dynamic_view, rows)
        args = *self.views, view, *self.eps, *aux
        code(source, prefix, blocks, *args, stream(self.device))
        if side is not None:
            fork, done = torch.cuda.Event(), torch.cuda.Event()
            fork.record(main)
            aux_stream = _aux_stream(self.device)
            aux_stream.wait_event(fork)
            with torch.cuda.stream(aux_stream):
                side(source, prefix, blocks, *args, stream(self.device))
            done.record(aux_stream)
            check = None
            if AUX_SIDE_CHECK:  # today's serial order, on main, into a shadow slice
                shadow = torch.empty_like(aux_rows)
                side(
                    source,
                    prefix,
                    blocks,
                    *args[:-1],
                    dynamic_view(shadow),
                    stream(self.device),
                )
                check = aux_rows, shadow
            _PENDING_AUX.append((done, check))
        return out
