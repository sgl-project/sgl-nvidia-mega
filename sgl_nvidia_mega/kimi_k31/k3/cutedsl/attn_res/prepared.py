# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Prepared seven-CTA AttnRes consumers for ordinary or Lamport mailbox input."""

import os
from functools import cache
from typing import Any

import cutlass
import cutlass.cute as cute
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
from ..common.policy import batch_table
from .aux import AuxMixture
from .kernel import ClusterAttnRes


def carveout_rows():
    """K31MK_ATTNRES_CARVEOUT_ROWS (default unset = the carve-out on every launch, as today): a row table
    ("1-32:1"; rows 1-128, values 0 / 1) of the launches that take K31MK_ATTNRES_CARVEOUT; every other launch (unlisted
    rows, rows above 128) takes the knob-off code. Read only when K31MK_ATTNRES_CARVEOUT is set. The carve-out serves
    the TMEM in-projection tiles (TM8 / TM64) that land beside AttnRes CTAs; at rows without such a tile it only
    shrinks the AttnRes CTAs' L1. Launch preference only: both
    codes are the same kernel, so outputs are bitwise equal."""
    spec = os.environ.get("K31MK_ATTNRES_CARVEOUT_ROWS", "")
    if not spec:
        return None
    table = batch_table(spec)
    bad = {b: v for b, v in table.items() if not 1 <= b <= 128 or v not in (0, 1)}
    if bad:
        raise ValueError(f"K31MK_ATTNRES_CARVEOUT_ROWS: rows 1-128 with values 0 / 1 only, got {bad}")
    return frozenset(b for b, v in table.items() if v)


class RowCarveout:
    """The carve-out code at ``rows``, the knob-off code elsewhere; ``_launch`` picks by the launch's row count."""

    def __init__(self, on, off, rows):
        self.on, self.off, self.rows = on, off, rows

    def pick(self, rows):
        return self.on if rows in self.rows else self.off


def aux_side_mode():
    """K31MK_AUX_FOLD=3: non-final capture consumers store only the updated prefix row, and
    the stock AuxMixture passes run later on a side stream (``aux_side.AuxSideMixture``)."""
    return os.environ.get("K31MK_AUX_FOLD", "0") == "3"


@cache
def compile_aux_mixture(nb, arch):
    """AuxMixture's score and combine passes alone, in the consumer's argument order
    (source, prefix, blocks, norm, query, output norm, out, eps, eps, aux)."""
    row, weight, eps = dynamic_fake(2), fake((7168,), (1,)), cutlass.Float32(1e-6)
    args = (row, row, dynamic_fake(3), weight, weight, weight, row, eps, eps, row)
    return cute.compile(AuxMixture(nb), *args, make_fake_stream(), options=f"--gpu-arch={arch}")


@cache
def compile_consumer(
    nb, delta, write, final, slots, cap, arch, deadline_ns, aux=False, mixture=False,
    fold=False, overlap=False, late=False, one=False, carveout=None, defer=None,
):
    # ``carveout`` only keys the cache: a K31MK_ATTNRES_CARVEOUT compile is its own copy (its own function attribute).
    row, weight, eps = dynamic_fake(2), fake((7168,), (1,)), cutlass.Float32(1e-6)
    source = fake((slots, cap, 7168), (cap * 7168, 7168, 1)) if slots else row
    kernel = ClusterAttnRes(
        nb,
        slots=slots,
        has_delta=delta,
        write=write,
        final=final,
        deadline_ns=deadline_ns,
        aux_mixture=mixture,
        aux_fold=fold,
        aux_overlap=overlap,
        late_rearm=late,
        one_round=one,
        **({"rearm_prev": defer[0], "rearm_own": defer[1]} if defer else {}),
    )
    args = (source, row, dynamic_fake(3), weight, weight, weight, row, eps, eps)
    args += (source,) if defer and defer[0] else ()  # K31MK_ATTNRES_DEFER_REARM: the previous ordinal's parity rows
    args += (row,) * aux  # the updated prefix row, also stored to an aux stream
    return cute.compile(kernel, *args, make_fake_stream(), options=f"--gpu-arch={arch}")


class PreparedAttnRes:
    """Ordinary-input code plus the Lamport variant of its bound mailbox ring."""

    def __init__(self, weights, key, eps, deadline_ns):
        self.device = weights[0].device
        self.views = tuple(to_cute(w.reshape(-1)) for w in weights)
        self.key, self.eps = key, tuple(map(cutlass.Float32, eps))
        self.deadline_ns = deadline_ns
        # K31MK_ATTNRES_DEFER_REARM (DRA): (re-arm the previous parity, re-arm own rows) set by bind_ring, else None.
        self.defer: Any = None
        self.prev_view: Any = None
        self.code = self._compile(0, 0)
        self.ring: Any = None
        self.aux: Any = None
        self.side_mixture: Any = None

    def _compile(self, slots, cap, aux=False, mixture=False):
        arch = architecture(self.device)
        # K31MK_AUX_FOLD=1: the aux mixture runs inside the consumer (bitwise equal to the
        # separate AuxMixture score/combine kernels; read when the aux variant is bound);
        # 2: the same, overlapped with the consumer's last cluster exchange.
        # 3: as 2, except bind_aux moves the non-final capture mixtures to a side stream (the
        # consumer then compiles with mixture=False: the prefix-row store only).
        mode = os.environ.get("K31MK_AUX_FOLD", "0")
        fold = mixture and mode in ("1", "2", "3")
        overlap = mode in ("2", "3")
        # AttnRes knobs (default off; off = today's cache key and call). K31MK_ATTNRES_LATE_REARM=1:
        # ring variants re-arm + fence after the output stores (EF4, bitwise).
        # K31MK_ATTNRES_ONE_ROUND=1: one post-delta exchange (Q, REORDER: needs an accuracy check before it is served).
        late = slots > 0 and os.environ.get("K31MK_ATTNRES_LATE_REARM", "0") == "1"
        one = os.environ.get("K31MK_ATTNRES_ONE_ROUND", "0") == "1"
        extra = (late, one) if late or one else ()
        # K31MK_ATTNRES_CARVEOUT=N (default unset = today's cache key and call): preferred shared-memory
        # carve-out of N percent for the AttnRes kernels, so the next layer's in-projection CTAs (TM64, ~170-200 KB of
        # SMEM) can start loading weights beside the AttnRes CTAs before their PDL wait. A launch preference only (no
        # change to the kernel or its synchronization). The attribute is set on the compiled function, so the knob
        # compiles its own cached copy (``carveout`` in the key): a knob-off compile in the same process never carries it.
        carveout = os.environ.get("K31MK_ATTNRES_CARVEOUT", "")
        # K31MK_ATTNRES_DEFER_REARM=1 (DRA, bitwise): ring variants of a consumer bound with ``previous`` /
        # ``own`` compile with them; passed only when bound, so default cache keys and calls stay.
        dra = {"defer": self.defer} if slots > 0 and self.defer is not None else {}
        if carveout:
            value = int(carveout)
            if not 0 <= value <= 100:
                raise ValueError(f"K31MK_ATTNRES_CARVEOUT={carveout}: want a percentage 0-100")
            code = compile_consumer(
                *self.key, slots, cap, arch, self.deadline_ns, aux, mixture, fold, overlap, *extra, carveout=value, **dra
            )
            code = prepare_shared_carveout(code, self.device.index, value)
            rows = carveout_rows()
            if rows is None:
                return code
            off = compile_consumer(*self.key, slots, cap, arch, self.deadline_ns, aux, mixture, fold, overlap, *extra, **dra)
            return RowCarveout(code, off, rows)
        return compile_consumer(
            *self.key, slots, cap, arch, self.deadline_ns, aux, mixture, fold, overlap, *extra, **dra
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

    def bind_ring(self, mailbox, previous=None, own=True):
        """Lamport input: sum the [slots, capacity, 7168] mailbox rows, then re-arm.
        ``previous`` / ``own`` (K31MK_ATTNRES_DEFER_REARM): the previous MoE ordinal's parity rows (same shape),
        re-armed by this consumer before its poll, or None; ``own`` False leaves this consumer's rows to the next."""
        if previous is not None or not own:
            if previous is not None and tuple(previous.shape) != tuple(mailbox.shape):
                raise ValueError(f"DRA: previous parity {tuple(previous.shape)} != mailbox {tuple(mailbox.shape)}")
            self.defer = (previous is not None, bool(own))
            self.prev_view = to_cute(previous) if previous is not None else None
        shape = tuple(mailbox.shape[:2])
        self.ring = self._compile(*shape), to_cute(mailbox), shape

    def bind_aux(self, *, mixture=False):
        """Variants that also store the updated prefix rows (a stock aux stream).

        Under K31MK_AUX_FOLD=3 a mixture consumer with saved blocks stores the prefix row only;
        ``run_aux_mixture`` then mixes that stored row with the saved blocks (rows 0..nb-1 must
        not change before the join: the executor checks the layer plan). A block write inside
        rows 0..nb-1 keeps the in-line passes (outside the serving plan; conservative). The final
        AttnRes keeps the folded mixture: nothing follows it before the join, so a side
        mixture there would be exposed in full."""
        nb, _, write, final = self.key
        side = mixture and aux_side_mode() and not final and nb > 0 and not 0 <= write < nb
        self.side_mixture = compile_aux_mixture(nb, architecture(self.device)) if side else None
        mixture = mixture and not side
        ring = None
        if self.ring is not None:
            slots, cap = self.ring[2]
            ring = self._compile(slots, cap, True, mixture)
        self.aux = self._compile(0, 0, True, mixture), ring

    def run_ring(self, prefix, blocks, out, aux=None):
        code = self.ring[0] if aux is None else self.aux[1]
        prev = (self.prev_view,) if self.prev_view is not None else ()
        return self._launch(code, self.ring[1], prefix, blocks, out, aux, prev=prev)

    def run(self, delta, prefix, blocks, out, aux=None):
        code = self.code if aux is None else self.aux[0]
        return self._launch(code, dynamic_view(delta), prefix, blocks, out, aux)

    def run_aux_mixture(self, row, blocks, aux):
        """The aux mixture of a stored prefix ``row`` [R, 7168] and saved ``blocks`` rows
        0..nb-1 into ``aux`` [R, 7168], on the current stream (K31MK_AUX_FOLD=3)."""
        row, blocks, aux = map(dynamic_view, (row, blocks, aux))
        args = (row, row, blocks, *self.views, row, *self.eps, aux)
        self.side_mixture(*args, stream(self.device))
        return aux

    def _launch(self, code, source, prefix, blocks, out, aux=None, prev=()):
        if isinstance(code, RowCarveout):  # K31MK_ATTNRES_CARVEOUT_ROWS
            code = code.pick(out.shape[0])
        rows = (prefix, blocks, out) + (() if aux is None else (aux,))
        prefix, blocks, view, *aux = map(dynamic_view, rows)
        args = *self.views, view, *self.eps, *prev, *aux
        code(source, prefix, blocks, *args, stream(self.device))
        return out
