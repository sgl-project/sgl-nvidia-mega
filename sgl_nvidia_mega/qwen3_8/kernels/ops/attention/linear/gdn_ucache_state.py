# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/layers/attention/linear/gdn_replayssm_ucache.py (its runtime half).
"""The state half of the GDN u-cache (deferred-fold ReplaySSM) target verify: the
rings and their cursors, which the plugin owns on SGLang's binding v2.

Every GDN layer keeps, per mamba slot, an fp16 checkpoint ``S0`` plus a 32-entry
ring of ``(k_hat, u, g)`` records; a slot's live window is the ``hist_len`` records
starting at ``cache_base``. One flush launch per layer and verify
(``gdn_ucache_verify.py``):

* outputs every draft position from ``S0`` plus the window (no state write),
* appends the T draft records after the window,
* and, for rows with ``hist_len >= flush_min``, folds the window into ``S0`` in place.

The cursors are shared by all GDN layers of a slot and committed once per verify
from the accept count (bonus token included), with the ``hist_len`` the verify saw:
a folded row restarts its window at the old window's end with the accepted
records, any other row extends it. A fresh slot only needs ``hist_len = 0``.

With request rings (``by_request_row``) the rings and cursors are request-lifetime
scratch indexed by the request's row in the request table, as the ReplaySSM spec
rings are: only live requests verify, so the radix tree's cached slots carry no
rings. The state ``S0`` stays at the Mamba slot; only ring addresses change.

With radix decode inserts, a row whose accepted tokens cross a track point gets
``fold(S0, window up to that point)`` written into its track slot after the
commit (FlashInfer's ``gdn_prefix_materialize``), so every state the radix tree
holds has an empty window. That kernel reads the state and the rings at one slot
id and takes bf16 k / u rings, so under request rings or the fp16 key ring the
u-cache tracks no decode-time state (``track_snapshots``).

Byte-identical to the source: the Triton kernels (cursor commit by state slot or
by request row, cursor reset, track plan), their host launchers, the ring struct and
the track materializer. Changed: SGLang's environment knobs are arguments
(``key_ring_fp16``: SGLANG_OPT_GDN_UCACHE_FP16_KEY_RING; ``by_request_row``:
SGLANG_OPT_GDN_UCACHE_REQUEST_RINGS; ``track_snapshots``: gdn_ucache_track_snapshots),
defaulting to the tuned values; the PD dtype tag and the retract-backup
check are dropped (the build declares ``backs_up_state=False`` instead).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import msgspec
import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# FlashInfer u-cache ring geometry (gdn_replay_mtp_umma.RING_SLOTS / W_RING).
RING_SLOTS = 32
_RING_MASK = RING_SLOTS - 1
_MAX_WINDOW = 16
_CURSOR_BLOCK = 256


class GdnUcacheBuffers(msgspec.Struct, frozen=True, kw_only=True):
    """Per-layer rings and their cursors, indexed by the mamba state slot, or by the
    request row under ``by_request_row``."""

    # [layers, slots, H, 32, K] normalized keys: bf16, or fp16 with the fp16 key ring
    # (gdn_ucache_key_ring_dtype).
    k_ring: torch.Tensor
    # [layers, slots, HV, 32, V] fp16 beta-scaled corrections (FlashInfer takes bf16 or fp16;
    # fp16 keeps 3 more mantissa bits through the deferred fold); bf16 with track snapshots.
    u_ring: torch.Tensor
    # [layers, slots, HV, 32] fp32 window-relative cumulative log decay.
    g_ring: torch.Tensor
    # [slots] int32 committed window length, in [0, 16].
    hist_len: torch.Tensor
    # [slots] int32 window origin in the ring, in [0, 32).
    cache_base: torch.Tensor
    # Rows whose window holds at least this many records fold it in the verify.
    flush_min: int
    # The "slots" dimension above is the request table's rows (row 0 = padding), not Mamba slots.
    by_request_row: bool = False


def gdn_ucache_flush_min(max_num_draft_tokens: int) -> int:
    # The laziest threshold that still fits T appended records in the window.
    return _MAX_WINDOW - max_num_draft_tokens + 1


def gdn_ucache_ring_bytes_per_slot(
    *,
    num_layers: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> int:
    k_bytes = num_k_heads * RING_SLOTS * head_k_dim * 2
    u_bytes = num_v_heads * RING_SLOTS * head_v_dim * 2
    g_bytes = num_v_heads * RING_SLOTS * 4
    return (k_bytes + u_bytes + g_bytes) * num_layers


def gdn_ucache_key_ring_dtype(device, *, fp16: bool = True) -> torch.dtype:
    if not fp16:
        return torch.bfloat16
    # Only the vendored SM107 kernel takes an fp16 k ring; FlashInfer's SM100 one asserts bf16.
    capability = torch.cuda.get_device_capability(device)
    if capability != (10, 7):
        raise ValueError(
            "the GDN u-cache fp16 key ring needs the SM107 u-cache kernel, "
            f"got SM{capability[0]}{capability[1]}"
        )
    return torch.float16


def allocate_gdn_ucache(
    *,
    num_layers: int,
    num_slots: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    max_num_draft_tokens: int,
    device: str,
    key_ring_fp16: bool = True,
    by_request_row: bool = True,
    track_snapshots: bool = False,
) -> GdnUcacheBuffers:
    # Rings are zeroed once; the kernels never read a record outside the window.
    def zeros(shape, dtype):
        return torch.zeros(shape, dtype=dtype, device=device)

    return GdnUcacheBuffers(
        k_ring=zeros(
            (num_layers, num_slots, num_k_heads, RING_SLOTS, head_k_dim),
            gdn_ucache_key_ring_dtype(device, fp16=key_ring_fp16),
        ),
        # gdn_prefix_materialize takes one ring dtype for k and u (bf16).
        u_ring=zeros(
            (num_layers, num_slots, num_v_heads, RING_SLOTS, head_v_dim),
            torch.bfloat16 if track_snapshots else torch.float16,
        ),
        g_ring=zeros((num_layers, num_slots, num_v_heads, RING_SLOTS), torch.float32),
        hist_len=zeros((num_slots,), torch.int32),
        cache_base=zeros((num_slots,), torch.int32),
        flush_min=gdn_ucache_flush_min(max_num_draft_tokens),
        by_request_row=by_request_row,
    )


@triton.jit
def _commit_cursors_kernel(
    slots_ptr,
    num_accept_tokens_ptr,
    hist_pool_ptr,
    base_pool_ptr,
    num_rows,
    flush_min,
    RING_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = offs < num_rows
    slot = tl.load(slots_ptr + offs, mask=in_range, other=-1).to(tl.int64)
    live = in_range & (slot > 0)
    hist = tl.load(hist_pool_ptr + slot, mask=live, other=0)
    base = tl.load(base_pool_ptr + slot, mask=live, other=0)
    num_accept = tl.load(num_accept_tokens_ptr + offs, mask=live, other=0).to(tl.int32)
    folded = hist >= flush_min
    new_base = tl.where(folded, (base + hist) & RING_MASK, base)
    new_hist = tl.where(folded, num_accept, hist + num_accept)
    tl.store(base_pool_ptr + slot, new_base, live)
    tl.store(hist_pool_ptr + slot, new_hist, live)


@triton.jit
def _commit_request_cursors_kernel(
    slots_ptr,
    rows_ptr,
    num_accept_tokens_ptr,
    hist_pool_ptr,
    base_pool_ptr,
    num_rows,
    flush_min,
    RING_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = offs < num_rows
    slot = tl.load(slots_ptr + offs, mask=in_range, other=-1).to(tl.int64)
    live = in_range & (slot > 0)
    row = tl.load(rows_ptr + offs, mask=live, other=0).to(tl.int64)
    hist = tl.load(hist_pool_ptr + row, mask=live, other=0)
    base = tl.load(base_pool_ptr + row, mask=live, other=0)
    num_accept = tl.load(num_accept_tokens_ptr + offs, mask=live, other=0).to(tl.int32)
    folded = hist >= flush_min
    new_base = tl.where(folded, (base + hist) & RING_MASK, base)
    new_hist = tl.where(folded, num_accept, hist + num_accept)
    tl.store(base_pool_ptr + row, new_base, live)
    tl.store(hist_pool_ptr + row, new_hist, live)


@triton.jit
def _reset_cursors_kernel(
    slots_ptr,
    hist_pool_ptr,
    base_pool_ptr,
    num_rows,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = offs < num_rows
    slot = tl.load(slots_ptr + offs, mask=in_range, other=-1).to(tl.int64)
    live = in_range & (slot > 0)
    zero = tl.zeros([BLOCK], dtype=tl.int32)
    tl.store(hist_pool_ptr + slot, zero, live)
    tl.store(base_pool_ptr + slot, zero, live)


# One program per plan layer (each recomputes the cheap row scan). Entry
# e = layer * num_rows + row; the active list holds every layer's live entries
# first, then -1. Reads the post-commit cursors.
@triton.jit
def _plan_track_kernel(
    slots_ptr,
    num_accept_tokens_ptr,
    track_slots_ptr,
    steps_to_track_ptr,
    hist_pool_ptr,
    base_pool_ptr,
    src_out_ptr,
    dst_out_ptr,
    base_out_ptr,
    count_out_ptr,
    active_out_ptr,
    num_active_out_ptr,
    num_rows,
    layer_slot_stride,
    BLOCK: tl.constexpr,
):
    layer = tl.program_id(0)
    num_layers = tl.num_programs(0)
    offs = tl.arange(0, BLOCK)
    in_range = offs < num_rows
    slot = tl.load(slots_ptr + offs, mask=in_range, other=-1).to(tl.int64)
    dst = tl.load(track_slots_ptr + offs, mask=in_range, other=-1).to(tl.int64)
    step = tl.load(steps_to_track_ptr + offs, mask=in_range, other=-1).to(tl.int32)
    live = in_range & (slot > 0) & (dst > 0) & (step >= 0)
    hist = tl.load(hist_pool_ptr + slot, mask=live, other=0)
    base = tl.load(base_pool_ptr + slot, mask=live, other=0)
    num_accept = tl.load(num_accept_tokens_ptr + offs, mask=live, other=0).to(tl.int32)
    # This verify's accepted records start at window index hist - num_accept on
    # folded and unfolded rows alike; the snapshot covers them through `step`.
    count = hist - num_accept + step + 1
    live_i32 = live.to(tl.int32)
    num_live = tl.sum(live_i32, axis=0)
    rank = tl.cumsum(live_i32, axis=0) - 1
    entry = layer * num_rows + offs
    layer_offset = layer.to(tl.int64) * layer_slot_stride
    src_flat = tl.where(live, layer_offset + slot, -1).to(tl.int32)
    dst_flat = tl.where(live, layer_offset + dst, -1).to(tl.int32)
    tl.store(src_out_ptr + entry, src_flat, in_range)
    tl.store(dst_out_ptr + entry, dst_flat, in_range)
    tl.store(base_out_ptr + entry, tl.where(live, base, 0), in_range)
    tl.store(count_out_ptr + entry, tl.where(live, count, -1), in_range)
    tl.store(active_out_ptr + layer * num_live + rank, entry, live)
    tail = num_layers * num_live + layer * (num_rows - num_live) + offs - num_live
    tl.store(
        active_out_ptr + tail,
        tl.full([BLOCK], -1, tl.int32),
        in_range & (offs >= num_live),
    )
    if layer == 0:
        tl.store(num_active_out_ptr, num_live * num_layers)
        # A track slot holds a folded state, so its own window is empty.
        zero = tl.zeros([BLOCK], dtype=tl.int32)
        tl.store(hist_pool_ptr + dst, zero, live)
        tl.store(base_pool_ptr + dst, zero, live)


def commit_gdn_ucache_cursors(
    *,
    buffers: GdnUcacheBuffers,
    slots: torch.Tensor,
    num_accept_tokens: torch.Tensor,
    request_rows: Optional[torch.Tensor] = None,
) -> None:
    """Advance the cursors of the verified slots (their request rows under
    ``by_request_row``); must follow that verify and precede the next one, with no
    other cursor write in between."""
    num_rows = slots.shape[0]
    if num_rows == 0:
        return
    if buffers.by_request_row:
        if request_rows is None or request_rows.shape[0] < num_rows:
            raise RuntimeError(
                "GDN u-cache rings by request row: the commit needs its request rows."
            )
        _commit_request_cursors_kernel[(triton.cdiv(num_rows, _CURSOR_BLOCK),)](
            slots,
            request_rows,
            num_accept_tokens,
            buffers.hist_len,
            buffers.cache_base,
            num_rows,
            buffers.flush_min,
            RING_MASK=_RING_MASK,
            BLOCK=_CURSOR_BLOCK,
        )
        return
    _commit_cursors_kernel[(triton.cdiv(num_rows, _CURSOR_BLOCK),)](
        slots,
        num_accept_tokens,
        buffers.hist_len,
        buffers.cache_base,
        num_rows,
        buffers.flush_min,
        RING_MASK=_RING_MASK,
        BLOCK=_CURSOR_BLOCK,
    )


def reset_gdn_ucache_cursors(*, buffers: GdnUcacheBuffers, slots: torch.Tensor) -> None:
    """Empty the windows of the slots a target extend writes, so the extend and
    the verifies after it read only the checkpoint ``S0``.

    The slot may carry its previous owner's cursors (a recycled slot, a prefix-cache
    copy-on-write or load-back), while the state it receives is always canonical:
    prefix-cached states are prefill-written, decode-time states never enter the
    tree. ``slots`` are state-pool (physical) ids; ``slot <= 0`` rows are padding
    and left untouched. One launch, no host sync."""
    global _reset_logged
    num_rows = slots.shape[0]
    if num_rows == 0:
        return
    if not _reset_logged:
        _reset_logged = True
        logger.info(
            "GDN u-cache: target extends restart their slots' windows (first: %d rows)",
            num_rows,
        )
    _reset_cursors_kernel[(triton.cdiv(num_rows, _CURSOR_BLOCK),)](
        slots,
        buffers.hist_len,
        buffers.cache_base,
        num_rows,
        BLOCK=_CURSOR_BLOCK,
    )


def reset_gdn_ucache_request_rows(
    *, buffers: GdnUcacheBuffers, request_rows: torch.Tensor
) -> None:
    """``reset_gdn_ucache_cursors`` under ``by_request_row``: empty the windows of the
    request rows a target extend writes (row 0 is the padding row and left untouched)."""
    global _reset_logged
    assert buffers.by_request_row
    num_rows = request_rows.shape[0]
    if num_rows == 0:
        return
    if not _reset_logged:
        _reset_logged = True
        logger.info(
            "GDN u-cache: target extends restart their request rows' windows (first: %d rows)",
            num_rows,
        )
    _reset_cursors_kernel[(triton.cdiv(num_rows, _CURSOR_BLOCK),)](
        request_rows,
        buffers.hist_len,
        buffers.cache_base,
        num_rows,
        BLOCK=_CURSOR_BLOCK,
    )


def _load_prefix_materialize() -> Callable:
    # FlashInfer fixes PM's dtypes when flashinfer.gdn_kernels is first imported
    # (GDN_UCACHE_STATE_DTYPE, which the deployment sets); the pool is fp16 state
    # with bf16 rings.
    try:
        from flashinfer.gdn_kernels.gdn_prefix_materialize import (
            gdn_prefix_materialize,
            ring_ty_torch,
            state_ty_torch,
        )
    except ImportError as e:
        raise RuntimeError(
            "GDN u-cache track snapshots (radix decode inserts) need "
            "FlashInfer >= 0.7.3 (flashinfer.gdn_kernels.gdn_prefix_materialize)."
        ) from e
    if (state_ty_torch(), ring_ty_torch()) != (torch.float16, torch.bfloat16):
        raise RuntimeError(
            "gdn_prefix_materialize resolved state/ring dtypes "
            f"{state_ty_torch()}/{ring_ty_torch()}, need float16/bfloat16: set "
            "GDN_UCACHE_STATE_DTYPE=fp16 (and leave GDN_UCACHE_RING_DTYPE unset) "
            "before flashinfer.gdn_kernels is first imported."
        )
    return gdn_prefix_materialize


class _TrackPlan(msgspec.Struct, frozen=True, kw_only=True):
    """Static gdn_prefix_materialize arguments for up to ``rows`` batch rows
    (times the plan layers); every tensor is [rows * layers] int32."""

    rows: int
    src: torch.Tensor
    dst: torch.Tensor
    base: torch.Tensor
    count: torch.Tensor
    active: torch.Tensor
    # [1] int32: live entries in ``active``.
    num_active: torch.Tensor


class GdnUcacheTrackMaterializer:
    """Writes the decode-time radix track snapshots of a u-cache pool.

    After a verify's cursor commit, every row whose accepted tokens cross a track
    point gets ``fold(S0, window up to that point)`` in its track slot, for all GDN
    layers, and the track slot's cursors are zeroed. One plan launch builds the
    FlashInfer arguments on the device (no host sync, graph-capturable); then one
    FlashInfer launch over ``[layers * slots]`` views of the layer-major pools, or
    one per layer if a pool is not contiguous."""

    def __init__(self, *, state: torch.Tensor, buffers: GdnUcacheBuffers):
        # One slot id addresses the state and the rings (no request-row rings).
        assert not buffers.by_request_row
        self._materialize = _load_prefix_materialize()
        self._buffers = buffers
        self._num_layers, self._num_slots = state.shape[:2]
        pools = (state, buffers.k_ring, buffers.u_ring, buffers.g_ring)
        assert all(t.shape[:2] == (self._num_layers, self._num_slots) for t in pools), [
            t.shape for t in pools
        ]
        self._layer_pools = pools
        # Flattening [layers, slots] into one pool dim needs no gap between layers.
        self._flat_pools = (
            tuple(t.view(-1, *t.shape[2:]) for t in pools)
            if all(t.is_contiguous() for t in pools)
            else None
        )
        self._plan_layers = self._num_layers if self._flat_pools is not None else 1
        self._plan: Optional[_TrackPlan] = None
        # Outgrown plans stay alive: captured commit graphs hold their pointers.
        self._retired_plans: list[_TrackPlan] = []

    @property
    def single_launch(self) -> bool:
        return self._flat_pools is not None

    def materialize(
        self,
        *,
        slots: torch.Tensor,
        num_accept_tokens: torch.Tensor,
        track_slots: torch.Tensor,
        steps_to_track: torch.Tensor,
    ) -> None:
        """``slots`` and ``track_slots`` are physical ids (``<= 0`` skips the row),
        ``steps_to_track`` the verify step whose state is tracked (``-1`` skips).
        Must follow ``commit_gdn_ucache_cursors`` of the same verify and precede
        the next verify."""
        num_rows = slots.shape[0]
        if num_rows == 0:
            return
        plan = self._plan_for(num_rows)
        num_entries = num_rows * self._plan_layers
        _plan_track_kernel[(self._plan_layers,)](
            slots,
            num_accept_tokens,
            track_slots,
            steps_to_track,
            self._buffers.hist_len,
            self._buffers.cache_base,
            plan.src,
            plan.dst,
            plan.base,
            plan.count,
            plan.active,
            plan.num_active,
            num_rows,
            self._num_slots,
            BLOCK=triton.next_power_of_2(plan.rows),
        )
        args = dict(
            src_slots=plan.src[:num_entries],
            dst_slots=plan.dst[:num_entries],
            cache_base=plan.base[:num_entries],
            count=plan.count[:num_entries],
            active_request_indices=plan.active[:num_entries],
            num_active=plan.num_active,
        )
        if self._flat_pools is not None:
            self._launch(self._flat_pools, args)
        else:
            for layer in range(self._num_layers):
                self._launch(tuple(t[layer] for t in self._layer_pools), args)

    def warm_up(self, *, max_batch_size: int, max_eager_batch_size: int = 0) -> None:
        """Allocates the plan for the larger of ``max_batch_size`` (graph batches)
        and ``max_eager_batch_size`` (eager verifies), and compiles both kernels on
        an all-padding batch, so neither CUDA graph capture nor a later eager batch
        allocates."""
        self._plan_for(max(max_batch_size, max_eager_batch_size))
        device = self._buffers.hist_len.device
        padded = torch.full((max_batch_size,), -1, dtype=torch.int32, device=device)
        self.materialize(
            slots=padded,
            num_accept_tokens=padded,
            track_slots=padded,
            steps_to_track=padded,
        )

    def _launch(self, pools: tuple, args: dict) -> None:
        state, k_ring, u_ring, g_ring = pools
        self._materialize(state, k_cache=k_ring, u_cache=u_ring, g_cache=g_ring, **args)

    def _plan_for(self, num_rows: int) -> _TrackPlan:
        plan = self._plan
        if plan is not None and plan.rows >= num_rows:
            return plan
        _assert_not_capturing("the track materialize plan (warm it up first)")
        if plan is not None:
            self._retired_plans.append(plan)
        device = self._buffers.hist_len.device
        entries = num_rows * self._plan_layers

        def buf(n):
            return torch.empty(n, dtype=torch.int32, device=device)

        self._plan = _TrackPlan(
            rows=num_rows,
            src=buf(entries),
            dst=buf(entries),
            base=buf(entries),
            count=buf(entries),
            active=buf(entries),
            num_active=buf(1),
        )
        return self._plan


_reset_logged = False


def _assert_not_capturing(what: str) -> None:
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"GDN u-cache verify: {what} must be prepared before CUDA graph capture."
        )
