# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Options described as "ported" follow NVIDIA's Rubin (sm_107) specialized-decode kernels (the rubin package).
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
from .math import situ_lean_enabled
from .experts import (
    MAX_CTAS,
    PLAN_BYTES,
    ExpertsGeometry,
    ExpertsGroupKernel,
    ExpertsKernel,
    ExpertsPlanKernel,
    fc2_pairs,
)


logger = logging.getLogger(__name__)
_ENGAGED: set = set()  # (knob, rows, kernel) already logged by this process


def log_engaged(knob, batch, kernel, dies):
    """One INFO line per (knob, rows, kernel) and process when a vendored engine takes a batch at prepare (inside graph
    capture: the capture's prepare), so a server log shows which row counts run on which engine."""
    key = (knob, batch, kernel)
    if key not in _ENGAGED:
        _ENGAGED.add(key)
        logger.info("%s engaged: rows=%d kernel=%s dies=%s", knob, batch, kernel, "on" if dies else "off")


def tensor(value, align=16):
    return to_cute(value.reshape(-1), align)


def streamed(batch):
    """B<=8 on the incremental producer: ~1-1.5% at B4-8, neutral at B1-2, which keep
    the previous producer."""
    return bool(batch_table("4-8:1").get(batch, 0))


def pregrouped(tp, batch):
    """B>8 experts read the unit table a one-CTA kernel builds after routing, beside
    the fused shared expert (hidden at B9-32; at B64 its ~2.3 us after routing outlast
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


def rubin8_enabled():
    """``K31MK_MOE_RUBIN8=1`` (default unset = off): TP16 B8 calls run the vendored ``RubinExpertsKernel`` (in
    ``..rubin``) on every SM, over the MXFP8 image the routing kernel's copy role writes (``front.RouteSitu(image=True)``:
    our MXFP8 of the FP32 latent in the vendored SMEM image layout), so it needs ``K31MK_EXP_XPLAN=1`` and ``K31MK_EXP_XPLAN_ROUTE=1``.
    The routing kernel does not signal the vendored route mailbox: the experts wait for it by PDL. ``moe.prepared`` keeps the
    split shared expert's join for this call (the experts wait for its gate/up half, the tail for all of it)."""
    return os.environ.get("K31MK_MOE_RUBIN8", "0") == "1"


def route_poll8_enabled():
    """``K31MK_MOE_ROUTE_POLL=8:1`` (default unset = off): TP16 B8 ``K31MK_MOE_RUBIN8`` calls run the vendored
    experts POLLED (the vendored ``POLLED`` {16: "8:1"}): the routing kernel signals their route mailbox (``front.RouteSitu`` signal +
    early_ids, copy halves count the image) and releases them once the front GEMM completed (the ported RELEASE_AFTER_FRONT), so they
    dedupe, partition and stream their first weights while routing runs. Needs K31MK_MOE_RUBIN8, K31MK_EXP_XPLAN(_ROUTE),
    K31MK_MOE_COPY_HALVES with 8 and K31MK_MOE_ROUTE_LEAN8; anything else raises (a polled grid whose routing never signals hangs)."""
    value = os.environ.get("K31MK_MOE_ROUTE_POLL", "")
    if value not in ("", "8:1"):
        raise ValueError(f"K31MK_MOE_ROUTE_POLL={value!r}: only '8:1' (TP16 B8) is defined")
    return value == "8:1"


def w8_enabled():
    """``K31MK_EXP_W8=1`` (default off): TP16 B8 non-publish calls run the W4A8 whole-tile engine
    (``mkv2.experts_w4a8``, REORDER: whole-tile K64 FP32 accumulation, 1-ulp class) through ``PersistentExperts.prepare``
    at grid ``K31MK_EXP_W8_GRID`` (default: every SM but ``W8_FREE_SMS``, which run the shared expert beside it), with this engine's
    K-17 (``strong``), ``hfull_bar`` and SiTU-lean settings and the K-16 consumer fence. VR200 (sm_107) only."""
    return os.environ.get("K31MK_EXP_W8", "0") == "1"


def xplan_enabled():
    """``K31MK_EXP_XPLAN=1`` (default 0; TP16 B<=8): the routed experts take x as the MXFP8 plan, written
    once per call (``planned.EarlyPlanKernel`` before the experts, or the routing kernel's copy role when the caller
    passes ``planned``) with the stock input rule, and each CTA fetches its SMEM x image with two TMAs instead of
    loading the FP32 latent and quantizing it (every CTA's FP32 reads and the per-CTA MXFP8 work, the
    experts prologue's x prework). Same SMEM bytes: bitwise equal rows."""
    return os.environ.get("K31MK_EXP_XPLAN", "0") == "1"


def xplan_route_enabled():
    """``K31MK_EXP_XPLAN_ROUTE=1`` together with ``K31MK_EXP_XPLAN=1`` (default 0): at B<=8 the routing
    kernel's copy role writes the experts' MXFP8 plan as it copies the latent (as at B>8), so the experts are prepared
    ``planned`` and run no plan kernel of their own (``moe.front`` hook)."""
    # Not with the lossy W4A4 engine: it reads an MXFP4 plan of its own.
    w4a4 = os.environ.get("K31MK_EXPERTS_W4A4", "0") == "1"
    return xplan_enabled() and os.environ.get("K31MK_EXP_XPLAN_ROUTE", "0") == "1" and not w4a4


def packed_k64():
    """``K31MK_EXPERTS_PACKED_K64=1`` (default 0; TP16 B<=8 on VR200): the routed weights
    land as packed E2M1 bytes and feed Rubin's K64 block-scaled MMAs (``stage(packed=True)``;
    a K64 instead of K32 accumulation grouping, same inputs). ``K31MK_EXPERTS_PACKED_STAGES``:
    its 16 KB weight slots (default 6). After the vendored RubinExpertsKernel (see ``..rubin``)."""
    if os.environ.get("K31MK_EXPERTS_PACKED_K64", "0") != "1":
        return 0
    return int(os.environ.get("K31MK_EXPERTS_PACKED_STAGES", "6"))


PACKED_STAGES = (5, 6, 7)  # 16 KB slots that fit beside the misc block (stage metadata at misc+4352, ring barriers <= 64)


def packed_slots(value, device):
    """One validated slot count (0 = the shipped engine) drives both the compiled kernel and the map choice (
    fix: a non-positive count used to compile the unpacked kernel while handing it the packed maps, so the
    FC2 expect_tx never completed). Packed K64 MMAs (idesc bit 31, packed E2M1 A) exist on SM107 only."""
    slots = int(value or 0)
    if slots == 0:
        return 0
    if slots not in PACKED_STAGES:
        raise ValueError(f"packed K64 experts: slot count {slots} not in {PACKED_STAGES}")
    if torch.cuda.get_device_capability(device) != (10, 7):
        raise ValueError("packed K64 experts need SM107 (VR200): the K64 packed-E2M1 MMA form is sm_107 only")
    return slots


@functools.cache
def compile_experts(
    tp, experts, batch, grid, device, publish=False, *knobs, activation_floor=None, packed=0, situ_lean=False,
    strong=False, x3_release=False, producer_pace=0, hfull_bar=False, xtma=False, box_rows=64, packed_fc1=False,
    packed_fc2=False,
):
    """Only code and geometry are cached; no owner tensors or descriptors. ``knobs``:
    (B<=8 stream, pregrouped, steal). ``packed``: 0, or the packed K64 engine's slot count. ``xtma``
    (``K31MK_EXP_XPLAN``): x is the MXFP8 plan, fetched by TMA (maps +256 / +384)."""
    defaults = (streamed(batch), False, False)
    stream, grouped, steal = knobs + defaults[len(knobs) :]
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
        packed=packed > 0,
        packed_stages=packed or 6,
        situ_lean=situ_lean,
        strong=strong,
        x3_release=x3_release,
        producer_spin=producer_pace,
        hfull_bar=hfull_bar,
        xtma=xtma,
        packed_fc1=packed_fc1,
        packed_fc2=packed_fc2,
    )
    moe.box_rows = box_rows  # the tail box's rows (``ExpertsKernel.box_rows``)
    flat = lambda n, dtype=c.Uint8, align=16: fake((n,), (1,), dtype, align)
    box = (c.Int64(0), c.Int32(0), c.Int32(0)) if publish else ()
    if publish == "twoshot":  # receive slots, owners' bases, parity x TP + rank
        box = (fake((3 * tp, box_rows, 3584 // tp)), fake((tp,), None, c.Int64), c.Int32(0))
    # x: the front's FP32 latent at every TP, quantized to MXFP8 in the kernel as stock does.
    x, ids = fake((batch, 3584), (3584, 1), c.Float32), flat(batch * 16, c.Int32)
    x = flat(PLAN_BYTES) if moe.grouped or xtma else x  # grouped / xtma: the MXFP8 plan
    scales = (flat(experts * n) for n in (moe.w13_scale_bytes, moe.w2_scale_bytes))
    out = flat(batch * 16 * (1 if publish else 3584), c.BFloat16)
    maps = flat(512 if xtma else 1152 if packed_fc1 or packed_fc2 else 768 if moe.paired else 256, align=64)
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


def dies_enabled():
    """``K31MK_MOE_DIES`` (default unset = off): TP16 MoE layers' routed w13 / w2 re-homed chunk by chunk alternately on the two
    dies at load (the vendored ``place``: same bytes and layout for every reader) and the SM -> die table handed to the vendored
    engines for die matching. Only with ``K31MK_MOE_RUBIN64`` or ``K31MK_MOE_WIDE`` (the engines that use it)."""
    on = os.environ.get("K31MK_MOE_DIES", "0") not in ("", "0")
    if on and not (os.environ.get("K31MK_MOE_RUBIN64") or os.environ.get("K31MK_MOE_WIDE")):
        raise ValueError("K31MK_MOE_DIES needs K31MK_MOE_RUBIN64 or K31MK_MOE_WIDE")
    return on


def dies8_enabled():
    """``K31MK_MOE_DIES8=1`` (default unset = off): the ``K31MK_MOE_RUBIN8`` B8 engine also gets the SM -> die table ``place``
    returned (the vendored M8 form: the FC2 ranks run on CTAs of their w2 chunk's die; ranks, sums and outputs unchanged), so
    ``K31MK_MOE_DIES`` no longer leaves the B8 calls reading placed weights die-blind. Needs ``K31MK_MOE_DIES`` and
    ``K31MK_MOE_RUBIN8``."""
    on = os.environ.get("K31MK_MOE_DIES8", "0") not in ("", "0")
    if on and not (dies_enabled() and rubin8_enabled()):
        raise ValueError("K31MK_MOE_DIES8 needs K31MK_MOE_DIES and K31MK_MOE_RUBIN8")
    return on


def place_routed(routed):
    """``dies_enabled`` layers: the vendored ``place(routed)`` (re-homes ``routed.w13_weight`` / ``w2_weight`` in place and
    returns the SM -> die table, or () when the device has no trusted die map)."""
    from ..rubin.moe.experts_runtime import place

    return tuple(place(routed))


def tp8_packed_fc1_rows():
    """``K31MK_MOE_TP8_PACKED_FC1`` (default unset = off; Tier 0): TP8 rows (9-64, a batch table such as "9-64:1") whose
    grouped calls land the FC1 weights as packed E2M1 bytes and run dense K64 MMAs (``ExpertsGeometry.packed_fc1``, map 6);
    FC2 unchanged. A K64 instead of K32 accumulation grouping inside FC1 (same inputs): not bitwise to the knob-off engine."""
    table = batch_table(os.environ.get("K31MK_MOE_TP8_PACKED_FC1", ""))
    bad = sorted(b for b in table if not 8 < b <= 64)
    if bad:
        raise ValueError(f"K31MK_MOE_TP8_PACKED_FC1 covers 9-64 rows only, got {bad}")
    return table


def tp8_packed_fc2_rows():
    """``K31MK_MOE_TP8_PACKED_FC2`` (default unset = off; Tier 0): TP8 rows (9-64, a batch table) whose grouped FC2 tile pairs land
    packed E2M1 bytes and run dense K64 MMAs (``ExpertsGeometry.packed_fc2``, maps 7-8). Not bitwise to the knob-off engine."""
    table = batch_table(os.environ.get("K31MK_MOE_TP8_PACKED_FC2", ""))
    bad = sorted(b for b in table if not 8 < b <= 64)
    if bad:
        raise ValueError(f"K31MK_MOE_TP8_PACKED_FC2 covers 9-64 rows only, got {bad}")
    return table


def rubin64_w2_depad_enabled():
    """``K31MK_MOE_RUBIN64_W2_DEPAD=1`` (default 0; c=8 16-32-row path): the ``K31MK_MOE_RUBIN64`` engine
    (vendored ``RubinGroupedExpertsKernel``, TP16 9-64 rows) reads FC2 weights from a load-time de-padded copy
    (``depad_w2``: [E, 3584, 96] live bytes; it refuses nonzero padding), re-homed on the dies like the routed weights
    when the layer is die-placed. Its packed w2 map takes the 96-byte row stride with the same 96 B x 128-row box: the
    same bytes land in SMEM, so outputs are bitwise those of the knob-off engine. Why: whole lines move through the memory system
    for the served rows' 96 data bytes (de-padded rows measured faster in A/B on the target GPU; the padding
    is a sizeable share of the weight stream). Cost: one more 308 MB per MoE layer (896 x 3584 x 96 B); the
    original stays for the other readers (B8 engines, stock)."""
    return os.environ.get("K31MK_MOE_RUBIN64_W2_DEPAD", "0") == "1"


def rubin64_w2sib_enabled():
    """``K31MK_MOE_RUBIN64_W2SIB=1`` (default 0; 24-64 rows): the ``K31MK_MOE_RUBIN64``
    engine reads FC2 from load-time RECORDS (``w2sib_records``: [E, 3584, 112], the 96 live bytes of each w2 row plus 16 B
    of its tile's scales in rows 0-63) instead of the de-padded copy, so each FC2 job is ONE 112 B x 128-row bulk load
    (no separate 1 KiB scale copy) and the MMA warp copies the scales to TMEM from that box. The same weight and scale
    bytes reach the same SMEM / TMEM positions: outputs are bitwise those of the knob-off engine. Why: the c=8 grouped FC2
    cost is per bulk operation, not per byte. Cost: the records replace the 96-byte copy,
    +51 MB per MoE layer (896 x 3584 x 16 B)."""
    return os.environ.get("K31MK_MOE_RUBIN64_W2SIB", "0") == "1"


def w2sib_records(w2, s2):
    """TP16 host-layout FC2 weights [E, 3584, 128] + their scales (E x 28 tiles x 1 KiB) -> [E, 3584, 112] uint8 records:
    bytes 0-95 = ``depad_w2``'s live bytes; bytes 96-111 of row 128 t + r = bytes [16 r, 16 r + 16) of tile t's 1 KiB of
    scales for r < 64, zero for r >= 64."""
    live = depad_w2(w2)
    experts = live.shape[0]
    scales = s2.view(torch.uint8)
    if scales.numel() != experts * 28 * 1024:
        raise ValueError(f"w2sib_records: expected {experts} x 28 KiB of FC2 scales, got {scales.numel()} bytes")
    records = live.new_zeros((experts, 28, 128, 112))
    records[..., :W2_LIVE_BYTES] = live.view(experts, 28, 128, W2_LIVE_BYTES)
    records[:, :, :64, W2_LIVE_BYTES:] = scales.reshape(experts, 28, 64, 16)
    return records.view(experts, 3584, 112)


def place_copy(t):
    """The vendored ``place`` steps for ONE tensor: a copy of ``t`` homed chunk by chunk alternately on die 0 / 1
    (``dies.localized_like``) with the sampled ``chunk_homes`` check. Raises if the homes do not verify (the caller asked
    for die-matched streaming of this copy; a silently die-oblivious copy would only be slower)."""
    from ..rubin.common import dies
    from ..rubin.moe.experts_runtime import PLACE_PASSES, PLACE_SAMPLES, PLACE_UNCLEAR

    copy = dies.localized_like(t, [0, 1])
    chunks = -(-copy.numel() * copy.element_size() // dies.CHUNK)
    wrong, sms = tuple(sorted({chunks * i // PLACE_SAMPLES for i in range(PLACE_SAMPLES)})), 4
    for attempt in range(PLACE_PASSES):
        homes = dies.chunk_homes(copy, wrong, sms)
        wrong = tuple(chunk for chunk, home in zip(wrong, homes) if home != chunk % 2)
        if not wrong:
            break
        sms = 8 if attempt < 2 else 16
    if len(wrong) > PLACE_UNCLEAR:
        raise RuntimeError(f"place_copy: chunks {wrong} not homed on their dies")
    return copy


def rubin64_rows():
    """``K31MK_MOE_RUBIN64`` (default unset = off): TP16 rows (9-64, a batch table such as "9-64:1") whose calls run
    the vendored ``RubinGroupedExpertsKernel`` (in ``..rubin``; ``PersistentExperts.prepare_rubin``)."""
    table = batch_table(os.environ.get("K31MK_MOE_RUBIN64", ""))
    bad = sorted(b for b in table if not 8 < b <= 64)
    if bad:
        raise ValueError(f"K31MK_MOE_RUBIN64 covers 9-64 rows only, got {bad}")
    return table


def wide_rows():
    """``K31MK_MOE_WIDE`` (default unset = off): TP16 rows (65-512, a batch table such as "65-512:1") whose calls run
    the vendored wide experts engine (in ``..rubin``; ``PersistentExperts.prepare_wide_rubin``)."""
    table = batch_table(os.environ.get("K31MK_MOE_WIDE", ""))
    bad = sorted(b for b in table if not 64 < b <= 512)
    if bad:
        raise ValueError(f"K31MK_MOE_WIDE covers 65-512 rows only, got {bad}")
    return table


def rubin64_stages_rows():
    """``K31MK_MOE_RUBIN64_STAGES`` (default unset = off): ``K31MK_MOE_RUBIN64`` rows (9-64, a batch table such
    as "41-64:8") whose grouped engine (``RubinGroupedExpertsKernel``) runs that many weight-ring stages (8; 7 = the default).
    Same protocol and TMEM layout family (8 stages: weight scales 128-191, hidden scales 192-207, activation scales 256-319),
    shared memory stays within the engine's SMEM budget; same MMAs in the same order, so outputs are bitwise the knob-off engine's.
    Why: measured on one GPU (bitwise outputs): faster per call at 48-64 rows, neutral at 16-32."""
    table = {b: v for b, v in batch_table(os.environ.get("K31MK_MOE_RUBIN64_STAGES", "")).items() if v}
    bad = sorted(b for b in table if not 8 < b <= 64 or table[b] not in (7, 8))
    if bad:
        raise ValueError(f"K31MK_MOE_RUBIN64_STAGES covers 9-64 rows at 7 or 8 stages only, got {bad}")
    missing = sorted(b for b in table if not rubin64_rows().get(b))
    if missing:
        raise ValueError(f"K31MK_MOE_RUBIN64_STAGES rows {missing} are not K31MK_MOE_RUBIN64 rows")
    return table


def rubin64_pool_fix_rows():
    """``K31MK_MOE_RUBIN64_POOL_FIX`` (default unset = off): ``K31MK_MOE_RUBIN64``
    rows (9-64, a batch table such as "9-64:1") whose grouped engine runs the FC2 tail pool with (a) the static path's accumulator
    acquire-ahead for the pool's jobs (two waits per eight jobs instead of one per job) and (b) the first claim 6 static jobs before
    the end instead of 3. Same MMAs, accumulators and job order, so outputs are bitwise the knob-off engine's; rows whose pool zone is
    0 run as before. Why (one GPU, test rig): one-at-a-time acquires cost measurable time per 6-job pool grain on
    the MMA warp and a claim takes longer than 3 jobs; with both changes the call is measured faster at 24-64 rows."""
    table = {b: v for b, v in batch_table(os.environ.get("K31MK_MOE_RUBIN64_POOL_FIX", "")).items() if v}
    bad = sorted(b for b in table if not 8 < b <= 64 or table[b] != 1)
    if bad:
        raise ValueError(f"K31MK_MOE_RUBIN64_POOL_FIX covers 9-64 rows with value 1 only, got {bad}")
    missing = sorted(b for b in table if not rubin64_rows().get(b))
    if missing:
        raise ValueError(f"K31MK_MOE_RUBIN64_POOL_FIX rows {missing} are not K31MK_MOE_RUBIN64 rows")
    return table


def rubin64_pool_zone_rows():
    """``K31MK_MOE_RUBIN64_POOL_ZONE`` (default unset = the vendored ``POOL_ZONE`` table): ``K31MK_MOE_RUBIN64``
    rows (9-64, a batch table such as "17-32:0") whose grouped engine runs that FC2 tail pool zone (the jobs at the end of every FC2(B)
    range that go into the pool; 0 = no pool, every CTA runs its whole static range) in place of the vendored table's (6 at 17-32, 12
    above). A zone only moves FC2 jobs between CTAs: every job runs exactly once with the same MMAs, accumulators and summation, so
    outputs are bitwise the vendored engine's. Why (one GPU, test rig, with STAGES 8 + POOL_FIX):
    zone 0 is measured faster per call than zone 6 at 24 and 32 rows; a pooled 6-job grain costs a claim
    (issue -> entry) and a producer take wait that the balance it buys does not repay at these rows."""
    table = batch_table(os.environ.get("K31MK_MOE_RUBIN64_POOL_ZONE", ""))
    bad = sorted(b for b in table if not 8 < b <= 64 or not isinstance(table[b], int) or table[b] not in (0, 6, 12, 18, 24))
    if bad:
        raise ValueError(f"K31MK_MOE_RUBIN64_POOL_ZONE covers 9-64 rows with zones 0 / 6 / 12 / 18 / 24 only, got {bad}")
    missing = sorted(b for b in table if not rubin64_rows().get(b))
    if missing:
        raise ValueError(f"K31MK_MOE_RUBIN64_POOL_ZONE rows {missing} are not K31MK_MOE_RUBIN64 rows")
    return table


def rubin64_gcut_rows():
    """``K31MK_MOE_RUBIN64_GCUT`` (default unset = off): ``K31MK_MOE_RUBIN64`` rows (9-64, value 1) whose
    die-matched grouped engine runs the two grouping cuts (``RubinGroupedExpertsKernel.gcut``): the die-census ticket as one predicated
    atomic (no branch around it: warp 2 waits for the value at its use in ``arrive``, not at the branch end) and each list's matched
    rank's own thread publishes its partition ranges, read back instead of four ``partition`` calls. Same atomic, ranks, ranges, owners
    and partial-sum order, so outputs are bitwise the knob-off engine's. Acts only with ``K31MK_MOE_DIES`` (the die grouping). Why
    (one GPU, test rig, c=8 form): measured faster per call at 16 and 32 rows."""
    table = {b: v for b, v in batch_table(os.environ.get("K31MK_MOE_RUBIN64_GCUT", "")).items() if v}
    bad = sorted(b for b in table if not 8 < b <= 64 or table[b] != 1)
    if bad:
        raise ValueError(f"K31MK_MOE_RUBIN64_GCUT covers 9-64 rows with value 1 only, got {bad}")
    missing = sorted(b for b in table if not rubin64_rows().get(b))
    if missing:
        raise ValueError(f"K31MK_MOE_RUBIN64_GCUT rows {missing} are not K31MK_MOE_RUBIN64 rows")
    if table and not dies_enabled():
        raise ValueError("K31MK_MOE_RUBIN64_GCUT needs K31MK_MOE_DIES (it cuts the die grouping only)")
    return table


def rubin64_prestream_rows():
    """``K31MK_MOE_RUBIN64_PRESTREAM`` (default unset = off): ``K31MK_MOE_RUBIN64`` rows (9-64; a
    batch table such as "9-32:7,33-64:3") whose die-matched grouped engine L2-prefetches each routed id's first NB FC1 boxes (row tile
    0) and their scales while the CTAs group the ids (NB = the table value, 1..14; issued right after the ids land, spread over the
    grid), so FC1 starts on L2-warm weights instead of waiting for HBM after the grouping. Cache hints on read-only weights: outputs
    are bitwise the knob-off engine's. Needs ``K31MK_MOE_DIES``. Why (one GPU, test rig): no weight byte moves
    during the grouping; measured faster per call at 24-32 rows with NB 7 and at 48-64 with NB 3."""
    table = {b: v for b, v in batch_table(os.environ.get("K31MK_MOE_RUBIN64_PRESTREAM", "")).items() if v}
    bad = sorted(b for b in table if not 8 < b <= 64 or not 1 <= table[b] <= 14)
    if bad:
        raise ValueError(f"K31MK_MOE_RUBIN64_PRESTREAM covers 9-64 rows with values 1..14 only, got {bad}")
    missing = sorted(b for b in table if not rubin64_rows().get(b))
    if missing:
        raise ValueError(f"K31MK_MOE_RUBIN64_PRESTREAM rows {missing} are not K31MK_MOE_RUBIN64 rows")
    if table and not dies_enabled():
        raise ValueError("K31MK_MOE_RUBIN64_PRESTREAM needs K31MK_MOE_DIES (die-matched grouped engines only)")
    return table


def wide_fold_rows():
    """``K31MK_MOE_WIDE_FOLD`` (default unset = off): ``K31MK_MOE_WIDE`` rows (65-128, a batch table such as
    "65-128:1") whose wide experts read the 64-row plan blocks the routing kernel wrote (``front.RouteSitu`` blocks) instead of
    running ``compile_wide_plan`` first. Every row must also be a ``K31MK_MOE_WIDE`` row."""
    table = {b: v for b, v in batch_table(os.environ.get("K31MK_MOE_WIDE_FOLD", "")).items() if v}
    bad = sorted(b for b in table if not 64 < b <= 128)
    if bad:
        raise ValueError(f"K31MK_MOE_WIDE_FOLD covers 65-128 rows only, got {bad}")
    missing = sorted(b for b in table if not wide_rows().get(b))
    if missing:
        raise ValueError(f"K31MK_MOE_WIDE_FOLD rows {missing} are not K31MK_MOE_WIDE rows")
    return table


def wide_depad_rows():
    """``K31MK_MOE_WIDE_DEPAD`` (default unset = off): ``K31MK_MOE_WIDE`` rows (65-512, a batch table such
    as "65-80:1") whose wide calls run on the vendored object that reads the de-padded, die-homed 96-byte w2 copy
    ``K31MK_MOE_RUBIN64_W2_DEPAD`` already made for the grouped engine (``PersistentExperts.rubin``) instead of the padded
    w2 (``PersistentExperts.rubin_wide``). No new copy, and the padded w2 stays, so the stock forward keeps working (unlike
    ``K31MK_MOE_W2_ONE96``). The 96 B x 128-row FC2 boxes land the same SMEM bytes either way, so outputs are bitwise the
    knob-off engine's (the 96-byte wide engine's md5 equals the padded one's at 72-128 rows). Why: the
    width-8 wide engine (65-80 rows) runs faster per call on 96-byte rows (measured); the width-16 VW
    rows (81-128) do not move. Every row must also be a ``K31MK_MOE_WIDE`` row."""
    table = {b: v for b, v in batch_table(os.environ.get("K31MK_MOE_WIDE_DEPAD", "")).items() if v}
    bad = sorted(b for b in table if not 64 < b <= 512)
    if bad:
        raise ValueError(f"K31MK_MOE_WIDE_DEPAD covers 65-512 rows only, got {bad}")
    missing = sorted(b for b in table if not wide_rows().get(b))
    if missing:
        raise ValueError(f"K31MK_MOE_WIDE_DEPAD rows {missing} are not K31MK_MOE_WIDE rows")
    return table


def fin_pair_rows():
    """``K31MK_MOE_FIN_PAIR`` (default unset = off): ``K31MK_MOE_WIDE`` rows (65-512, a batch table such as
    "65-128:1") whose wide engine finalizes a warp's staged-row units two at a time (``RubinWideExpertsKernel.unit2``: one poll
    loop over both units' 16 slots, then each unit's re-arm, slot-order FMA chain and local / two-shot store as before). The
    same units are finalized once each with the same sums and stores, so outputs are bitwise the knob-off engine's. Why: the
    finalize after the last FC2 epilogue is exposed at 72-128 rows (measured); a warp holds
    ~1.7 units there and polled them one after the other. Every row must also be a ``K31MK_MOE_WIDE`` row."""
    table = {b: v for b, v in batch_table(os.environ.get("K31MK_MOE_FIN_PAIR", "")).items() if v}
    bad = sorted(b for b in table if not 64 < b <= 512)
    if bad:
        raise ValueError(f"K31MK_MOE_FIN_PAIR covers 65-512 rows only, got {bad}")
    missing = sorted(b for b in table if not wide_rows().get(b))
    if missing:
        raise ValueError(f"K31MK_MOE_FIN_PAIR rows {missing} are not K31MK_MOE_WIDE rows")
    return table


def wide_stages_rows():
    """``K31MK_MOE_WIDE_STAGES`` (default unset = off): ``K31MK_MOE_WIDE`` rows (65-128, a batch
    table such as "65-128:8") whose wide engine (width 8 ``RubinWideExpertsKernel`` or width 16 ``RubinVWExpertsKernel``)
    runs that many weight-ring stages (8; 7 = the default). Same producer / MMA / epilogue protocol with one more 16 KiB slot
    (+ its scales and activation slice) in flight; on the width-16 engine the scale TMEM columns move from 384 to right after
    the 16 accumulator big slots (256) so 8 stages fit the 512 columns. Same MMAs in the same order on the same bytes, so
    outputs are bitwise the knob-off engine's. Why: one more stage in flight shortens the FC1 weight stream
    (measured on one GPU at 72-128 rows, bitwise)."""
    table = {b: v for b, v in batch_table(os.environ.get("K31MK_MOE_WIDE_STAGES", "")).items() if v}
    bad = sorted(b for b in table if not 64 < b <= 128 or table[b] not in (7, 8))
    if bad:
        raise ValueError(f"K31MK_MOE_WIDE_STAGES covers 65-128 rows at 7 or 8 stages only, got {bad}")
    missing = sorted(b for b in table if not wide_rows().get(b))
    if missing:
        raise ValueError(f"K31MK_MOE_WIDE_STAGES rows {missing} are not K31MK_MOE_WIDE rows")
    return table


def fc2_pairload_rows():
    """``K31MK_MOE_FC2_PAIRLOAD`` (default unset = off): ``K31MK_MOE_WIDE`` rows (65-128, a batch
    table such as "65-128:1") whose wide engine (width 8 or 16) loads two consecutive static FC2 jobs of one unit that fall in
    two adjacent ring slots as ONE 256-row w2 box and ONE 2 KiB scale copy on the first slot's full barrier (the second
    slot's full barrier gets a plain 33-count arrive; the MMA warp consumes the first job first). Same bytes in the same
    slots, same MMAs in the same order, so outputs are bitwise the knob-off engine's. Why: the producer thread's per-job
    issue loop paces FC2 on the wide engines and each TMA op in it costs FC2 window time at 128 rows; pairs halve the ops per paired job."""
    table = {b: v for b, v in batch_table(os.environ.get("K31MK_MOE_FC2_PAIRLOAD", "")).items() if v}
    bad = sorted(b for b in table if not 64 < b <= 128)
    if bad:
        raise ValueError(f"K31MK_MOE_FC2_PAIRLOAD covers 65-128 rows only, got {bad}")
    missing = sorted(b for b in table if not wide_rows().get(b))
    if missing:
        raise ValueError(f"K31MK_MOE_FC2_PAIRLOAD rows {missing} are not K31MK_MOE_WIDE rows")
    return table


def rubin64_lean_local_rows():
    """``K31MK_MOE_RUBIN64_LEAN_LOCAL`` (8-32 rows; default unset = off): ``K31MK_MOE_RUBIN64``
    rows (9-64, a batch table such as "9-64:1") whose grouped engine writes its LOCAL (non-published) FC2 rows through the
    published modes' per-warp transpose and one 16-byte store per lane instead of eight 4-byte stores per lane. Same bytes at
    the same addresses (the transpose positions are the local row's column order), so outputs are bitwise the knob-off
    engine's; published calls are unaffected. Why: the eight-store epilogue falls behind the FC2 MMAs and drains
    after the last one; the transposed form keeps up (measured on one GPU at 16-48
    rows)."""
    table = {b: v for b, v in batch_table(os.environ.get("K31MK_MOE_RUBIN64_LEAN_LOCAL", "")).items() if v}
    bad = sorted(b for b in table if not 8 < b <= 64 or table[b] != 1)
    if bad:
        raise ValueError(f"K31MK_MOE_RUBIN64_LEAN_LOCAL covers 9-64 rows with value 1 only, got {bad}")
    missing = sorted(b for b in table if not rubin64_rows().get(b))
    if missing:
        raise ValueError(f"K31MK_MOE_RUBIN64_LEAN_LOCAL rows {missing} are not K31MK_MOE_RUBIN64 rows")
    return table


@functools.cache
def compile_wide_plan(batch, device):
    """FP32-latent plan writer above 64 rows (``wide_plan.WidePlanKernel``)."""
    from .wide_plan import WidePlanKernel

    args = (fake((batch, 3584), None, c.Float32), fake((PLAN_BYTES * -(-batch // 64),), (1,), c.Uint8), make_fake_stream())
    with torch.accelerator.device_index(device):
        return cute.compile(WidePlanKernel(batch), *args)


@functools.cache
def compile_plan(batch, device, fp32=False):
    """Plan writer for callers without the routing kernel's copy role in front."""
    dtype = c.Float32 if fp32 else c.BFloat16
    args = (fake((batch, 3584), None, dtype), fake((PLAN_BYTES,), (1,), c.Uint8), make_fake_stream())
    with torch.accelerator.device_index(device):
        return cute.compile(ExpertsPlanKernel(batch), *args)


def scratch(tp, device):
    """Model-owned expert buffers: the workspace and the grouped plan (MXFP8 x, written
    by the routing kernel or ``compile_plan``)."""
    size, u8 = (ExpertsGeometry(tp).workspace_bytes, torch.uint8)
    if tp == 16 and w8_enabled():  # the W8 engine's claim / call counters appended (``planned.scratch``'s size)
        from .planned import planned_workspace_bytes  # lazy: planned imports this module

        size = planned_workspace_bytes(size)
    workspace = torch.full((size,), 255, device=device, dtype=u8)
    plan = torch.empty(PLAN_BYTES, device=device, dtype=u8)
    buffers = {"workspace": workspace, "plan": plan}
    if tp == 16 and rubin64_rows():  # the vendored engine's own workspace (die census, peer slots: its layout, not ours)
        from ..rubin.moe import experts_runtime as rubin_runtime  # lazy: compiled only under the knob

        buffers["rubin_workspace"] = rubin_runtime.scratch(16, device, rows=64)["workspace"]
    if tp == 16 and rubin8_enabled():  # the B8 engine's own workspace (peer slots, generations; its layout), never shared
        from ..rubin.moe import experts_runtime as rubin_runtime

        buffers["rubin8_workspace"] = rubin_runtime.scratch(16, device, rows=64)["workspace"]
    if tp == 16 and wide_rows():  # the wide engine's workspace + ceil(rows / 64) plans (its layout; "homes" added at prepare)
        from ..rubin.moe import experts_runtime as rubin_runtime

        buffers["rubin_wide"] = rubin_runtime.scratch(16, device, rows=max(wide_rows()))
    return buffers


# Persistent grids as the SMs left free beside the experts (grid = SM count - free), tuned on GB300. TP8: B3-4 leave 4;
# B5-8 publish leave 16-24 for the one-wave tail; grouped B9-64 use every SM. TP16: every SM.
FREE_SMS = {8: (16, "3-4:4,5-6:24,7:20,9-64:0"), 16: (0, "")}
# K31MK_EXP_W8 (VR200, sm_107): SMs left free for the shared expert beside the W4A8 whole-tile grid.
W8_FREE_SMS = 52


def device_grids(tp, sms):
    """(default grid, per-batch table) for a device with ``sms`` SMs: ``FREE_SMS`` SMs left
    free beside the experts (so a larger part adds its extra SMs to every grid).
    ``K31MK_EXPERTS_GRID_SMS`` overrides the SM count the grids are derived for."""
    sms = int(os.environ.get("K31MK_EXPERTS_GRID_SMS", sms))
    free, spec = FREE_SMS[tp]
    table = {b: min(sms - f, MAX_CTAS) for b, f in batch_table(spec).items()}
    return min(sms - free, MAX_CTAS), table


# TP16 FC2 rows in the host layout: K = 192 FP4 (96 live bytes) padded to K = 256 (128 B), the padding zero-filled
# (``FusedMoE._load_w2`` loads the checkpoint slice at the start of the padded intermediate and zeroes the rest).
W2_ROW_BYTES, W2_LIVE_BYTES = 128, 96


def strong_sync_enabled():
    """``K31MK_EXPERTS_STRONG_SYNC=1`` (default 0; K-17): the persistent experts' intra-grid hand-offs
    (hidden images, split-K partials, staged FC2 rows, their re-arms) use ``st.relaxed.gpu`` stores at the width they
    are polled; the hidden scales move as one 16-bit store per (tile, token) and two 16-bit polls instead of byte
    stores polled as 32-bit words (a mixed-size data race, PTX 8.7.2). Same bytes, same arithmetic (bitwise).
    Block 1's hidden scale is computed once per thread outside the token loop (the lean form). ``=2`` is an
    alias of ``=1`` (it selected the lean form while the in-loop form was ``=1``)."""
    return os.environ.get("K31MK_EXPERTS_STRONG_SYNC", "0") in ("1", "2")


def strong_mode(value):
    """``PersistentExperts(strong=...)``: False / True (0 / 1; "lean" is an alias of True), or a constructor-only mode
    of the K-17 cost study: "inloop_DEBUG" (the in-loop scale form; see ``experts.STRONG_MODES``)."""
    if isinstance(value, str):
        if value == "lean":
            return True
        if value != "inloop_DEBUG":
            raise ValueError(f"strong={value!r}: not a K-17 mode")
        return value
    return bool(value)


def producer_pace():
    """``K31MK_EXPERTS_PRODUCER_PACE=<cycles>`` (default 0 = off; TP16 B4-8 stream calls of the packed
    K64 engine): after each weight stage's copies the producer lane spins that many SM clocks (``ExpertsKernel(
    producer_spin=...)``). No memory or barrier operation: outputs bitwise equal. Measured with 7 packed slots at 240."""
    value = int(os.environ.get("K31MK_EXPERTS_PRODUCER_PACE", "0"))
    if value < 0:
        raise ValueError(f"K31MK_EXPERTS_PRODUCER_PACE={value}: cycles >= 0")
    return value


def x3_release_enabled():
    """``K31MK_EXPERTS_X3_RELEASE=2`` (default 0; TP16 B4-8 streamed calls that do not publish):
    the compliant form of the inherited X3-WAR edge (K-10). After each weight stage's MMAs the MMA warp runs
    ``bar.sync 12, 32`` and lane 0 arrives (release) on the slot's empty barrier, which then counts 2
    (``ExpertsKernel(x3_release="bar")``). Same loads, MMAs and stores: bitwise equal outputs. ``=1`` (a lane-0 +
    shfl form with an unstable cost) is not shipped and is refused here."""
    value = os.environ.get("K31MK_EXPERTS_X3_RELEASE", "0")
    if value not in ("0", "2", "3"):
        raise ValueError(f"K31MK_EXPERTS_X3_RELEASE={value!r}: 0 (off), 2 (deferred release) or 3 (meta barrier)")
    return {"2": "bar", "3": "meta"}.get(value, False)


def hfull_bar_enabled():
    """``K31MK_EXPERTS_HFULL_BAR=1`` (default 0; the same TP16 B4-8 stream calls as x3): the hidden
    loader warp joins with ``bar.sync 13, 32`` instead of bar.warp.sync before lane 0's release arrive on hfull
    (``ExpertsKernel(hfull_bar=True)``). Same loads, MMAs and stores: bitwise equal outputs."""
    value = os.environ.get("K31MK_EXPERTS_HFULL_BAR", "0")
    if value not in ("0", "1"):
        raise ValueError(f"K31MK_EXPERTS_HFULL_BAR={value!r}: 0 (off) or 1")
    return value == "1"


def w2_depad_enabled():
    """``K31MK_EXPERTS_W2_DEPAD=1`` (default 0): TP16 routed experts read FC2 weights from a load-time
    copy without the K padding (rows of 96 live bytes instead of 128), so the padding bytes stop moving through the
    memory system. Lossless relayout: ``depad_w2`` refuses a nonzero padding byte and
    ``repad_w2`` inverts it bitwise. The loaded tensor stays as it is (stock's FlashInfer path reads it); the copy
    costs 3/4 of the TP16 w2 bytes per MoE layer (896 x 3584 x 96 B = 308 MB)."""
    return os.environ.get("K31MK_EXPERTS_W2_DEPAD", "0") == "1"


def depad_w2(w2):
    """TP16 host-layout FC2 weights [E, 3584, 128] -> their live bytes as contiguous [E, 3584, 96] uint8 rows.
    Raises if any padding byte is nonzero: the relayout must be invertible (``repad_w2``)."""
    rows = w2.view(torch.uint8)
    if rows.dim() != 3 or tuple(rows.shape[1:]) != (3584, W2_ROW_BYTES):
        raise ValueError(f"depad_w2: expected TP16 FC2 weights [E, 3584, {W2_ROW_BYTES}], got {tuple(rows.shape)}")
    if bool(rows[..., W2_LIVE_BYTES:].any()):
        raise ValueError("depad_w2: nonzero FC2 padding bytes; dropping them would not be lossless")
    return rows[..., :W2_LIVE_BYTES].contiguous()


def repad_w2(live):
    """Inverse of ``depad_w2``: [E, 3584, 96] live bytes -> [E, 3584, 128] with zero padding bytes."""
    out = live.new_zeros((*live.shape[:-1], W2_ROW_BYTES))
    out[..., :W2_LIVE_BYTES] = live
    return out


def w2_one96_enabled():
    """``K31MK_MOE_W2_ONE96=1`` (default 0;, wide decode H-MKPREP): ONE resident FC2 copy per TP16 MoE layer.
    Every MK FC2 reader takes the de-padded [E, 3584, 96] copy (``depad_w2``; die-homed like the routed weights when
    the layer was placed): this engine's map 1 and packed K64 map (96-byte stride, the same boxes), the grouped M64
    engine, the wide engine and the RUBIN8 B8 engine (vendored ``w2_row_bytes`` 96: the same boxes, so the same SMEM
    bytes and bitwise outputs). The W4A8 B8 engine (``K31MK_EXP_W8``) is not converted and refuses. ``release_padded_w2``
    (providers) then frees the loaded padded tensor and makes the stock FusedMoE forward of the layer raise: valid only
    in a worker whose every MoE call is a megakernel call (a PD decode worker; fails closed otherwise). Per layer it
    frees 896 x 3584 x 128 B against the two-copy DEPAD form and 896 x 3584 x 32 B against DEPAD off."""
    value = os.environ.get("K31MK_MOE_W2_ONE96", "0")
    if value not in ("0", "1"):
        raise ValueError(f"K31MK_MOE_W2_ONE96={value!r}: 0 (off) or 1")
    return value == "1"


def release_padded_w2(routed, experts):
    """``K31MK_MOE_W2_ONE96``: free ``routed``'s loaded padded FC2 tensor once ``experts`` (a ``PersistentExperts``
    built under the knob over that tensor) holds no tensor on its storage, and make ``routed``'s stock forward raise
    (its w2 is gone). Returns the bytes of the released storage."""
    if type(experts) is not PersistentExperts or not experts.one96:
        raise ValueError(f"K31MK_MOE_W2_ONE96: {type(experts).__name__} was not built with the 96-byte form")
    w2 = routed.w2_weight
    storage = w2.untyped_storage()
    if storage.data_ptr() != experts.padded_w2_storage:
        raise ValueError("K31MK_MOE_W2_ONE96: routed.w2_weight is not the tensor this engine de-padded")
    if storage.data_ptr() in experts.w2_storages():
        raise ValueError("K31MK_MOE_W2_ONE96: an MK engine still holds the padded w2")
    freed = storage.nbytes()
    w2.data = w2.data.new_empty((0,))

    def refused(*args, **kwargs):
        raise RuntimeError("K31MK_MOE_W2_ONE96: this layer's padded w2 was released; the stock FusedMoE path cannot run")

    routed.forward = refused
    torch.accelerator.empty_cache()
    return freed


class PersistentExperts:
    def __init__(
        self, w13, s13, w2, s2, *, beta=4.0, linear_beta=25.0, activation_floor=None, packed=None, w2_depad=None,
        situ_lean=None, strong=None, xplan=None, dies=(), box_rows=64,
    ):
        experts, rows, _ = w13.shape
        # w13 rows = 2 x the per-rank intermediate: TP8 384 -> 768; TP16 192, padded to 256 by FlashInfer -> 512.
        # Anything else (an EP or other layout) must not be read as TP16.
        if rows not in (768, 512):
            raise ValueError(f"packed-MXFP4 expert rows {rows} are neither TP8 (768) nor TP16 (512) at EP1")
        self.tp = 8 if rows == 768 else 16
        self.device, self.experts = w13.device, experts
        self.weights = (w13, s13, w2, s2)
        self.activation_floor = False if activation_floor is None else activation_floor
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.grid, self.grids = device_grids(self.tp, sms)
        scalars = (beta, linear_beta, 0)
        self.scalars = tuple(map(c.Float32, scalars))
        self.situ_lean = situ_lean_enabled() if situ_lean is None else bool(situ_lean)
        self.strong = strong_sync_enabled() if strong is None else strong_mode(strong)  # K-17 (``strong_sync_enabled``)
        self.x3_release = x3_release_enabled()  # x3 release (TP16 B4-8 stream calls without publish, see prepare)
        self.pace = producer_pace()  # producer pace (cycles per weight stage; packed B4-8 stream calls, see prepare)
        self.hfull_bar = hfull_bar_enabled()  # hfull bar (the x3 calls, see prepare)
        # ``xplan`` (None: ``K31MK_EXP_XPLAN``; or True / False): TP16
        # B<=8 calls take x as the MXFP8 plan by TMA (``ExpertsKernel(xtma=...)``). Attribute names differ from
        # ``PlannedExperts.xplan`` (the TP8 planned engine's own x-plan option).
        xtma = xplan_enabled() if xplan is None else xplan
        if xtma not in (False, True):
            raise ValueError(f"xplan must be None, False or True, not {xplan!r}")
        self.xtma = bool(xtma) if self.tp == 16 else False
        self.xtma_maps = []  # every weight + plan maps tensor ever prepared (kept alive: graphs read them by address)
        # ``K31MK_EXP_W8``: TP16 B8 non-publish calls on the W4A8 whole-tile engine (``prepare_w8``).
        self.w8 = w8_enabled() and self.tp == 16
        self.w8_grid = int(os.environ.get("K31MK_EXP_W8_GRID", sms - W8_FREE_SMS)) if self.w8 else None
        # ``K31MK_EXP_W8_PUBLISH`` (default unset = off): "fin" / "pd" let the W8 take the published TP16 B8 call
        # (``K31MK_MOE_TP16_PUBLISH=8:1`` + a WaveMoeTail form); off, a B8 publish call still raises below.
        self.w8_pub = os.environ.get("K31MK_EXP_W8_PUBLISH", "0") if self.w8 else "0"
        if self.w8_pub not in ("0", "fin", "pd", "f2", "f2bo", "f2sp", "finf", "f2spf", "f2spfcw"):  # f2* = measurement forms
            raise ValueError(f"K31MK_EXP_W8_PUBLISH={self.w8_pub!r} not in 0 / fin / pd")
        if self.w8 and (torch.cuda.get_device_capability(self.device) != (10, 7) or not 0 < self.w8_grid <= sms):
            raise ValueError(f"K31MK_EXP_W8 needs VR200 (sm_107) and 0 < grid <= {sms} (grid {self.w8_grid})")
        if self.w8 and (self.strong is not True or self.hfull_bar is not True):  # fail closed: never the
            # weak K-17 (b) hand-off (mixed-size scale race) or the K-10b hfull join on the W8 engine
            raise ValueError("K31MK_EXP_W8 needs K31MK_EXPERTS_STRONG_SYNC=1 and K31MK_EXPERTS_HFULL_BAR=1 "
                             f"(strong={self.strong!r}, hfull_bar={self.hfull_bar!r})")
        # ``w2_depad`` (None: ``K31MK_EXPERTS_W2_DEPAD``): TP16 maps read FC2 from the de-padded copy. Every TP16 FC2
        # reader (this engine, the B1/B2 ClusterExpertsTP16 chain, grouped B>8) goes through map 1, whose two 64-byte
        # chunks start at bytes 0 and 32 of a row: the same SMEM bytes land from 96-byte rows.
        self.w2_depad = (w2_depad_enabled() if w2_depad is None else bool(w2_depad)) and self.tp == 16
        # ``K31MK_MOE_W2_ONE96``: ONE de-padded copy for every FC2 reader below (map 1, the packed map, the vendored
        # engines); only the padded tensor's storage address is kept (``release_padded_w2`` checks it), no reference.
        self.one96 = w2_one96_enabled() and self.tp == 16
        self.padded_w2_storage = None
        if self.one96:
            w2_96 = depad_w2(w2)
            if dies:  # the routed weights were placed: home the copy the same way (FC2 die matching)
                w2_96 = place_copy(w2_96)
            self.padded_w2_storage = w2.untyped_storage().data_ptr()
            w2 = w2_96
            self.weights = (w13, s13, w2, s2)
            self.w2_depad, self.w2_live = True, w2
        else:
            self.w2_live = depad_w2(w2) if self.w2_depad else None
        # The experts kernel uses maps 0-1 (L2 promotion on w13); the TP8 B1 cluster
        # experts use maps 2-4 (w13, w2 two-chunk and one-chunk boxes, promoted); TP8
        # FC2 pairs (B4-64) use map 5 (w2 chunk 2 of 256 rows).
        w2_tile = (192, 3, 64) if self.tp == 8 else (W2_ROW_BYTES, 2, 32)
        fc2 = (w2, 3584, *w2_tile, 2, False)
        if self.w2_depad:
            fc2 = (self.w2_live, 3584, W2_LIVE_BYTES, 2, 32, 2, False)
        specs = [(w13, rows, 1792, 28, 64, 2, True), fc2]
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
        # Packed K64 (TP16, B<=8): plain-byte maps over the same weight bytes, appended after the 16U4 maps
        # (``ClusterExpertsTP16`` keeps maps 0-1): one 128 B x 128-row box per FC1 stage, the 96 data bytes of
        # each w2 row per FC2 tile (the vendored ``RubinExpertsKernel.encode_maps``).
        self.packed = packed_slots(packed_k64() if packed is None else packed, self.device) if self.tp == 16 else 0
        if self.packed:
            u8 = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8
            maps.append((w13.data_ptr(), u8, (128, experts * rows, 14), (1792, 128), (128, 128, 1), swizzle, l2[True]))
            w2_rows = experts * 3584
            pitch = W2_LIVE_BYTES if self.one96 else W2_ROW_BYTES  # K31MK_MOE_W2_ONE96: 96-byte rows, the same box
            maps.append((w2.data_ptr(), u8, (pitch, w2_rows, 1), (pitch, pitch * w2_rows), (96, 128, 1), swizzle, l2[False]))
        # ``K31MK_MOE_TP8_PACKED_FC1``: map 6 = the w13 bytes as plain UINT8, one 128 B x 128-row box per FC1 stage (the TP16 packed
        # FC1 map's form), after the TP8 maps 0-5.
        self.pk1_rows = tp8_packed_fc1_rows() if self.tp == 8 else {}
        self.pk2_rows = tp8_packed_fc2_rows() if self.tp == 8 else {}
        if self.pk1_rows or self.pk2_rows:  # maps 6-8 together: fixed offsets 768 / 896 / 1024
            u8 = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8
            maps.append((w13.data_ptr(), u8, (128, experts * rows, 14), (1792, 128), (128, 128, 1), swizzle, l2[True]))
            w2_rows = experts * 3584  # K31MK_MOE_TP8_PACKED_FC2: map 7 = bytes 0-127 of each 192 B w2 row, map 8 = bytes 128-191
            maps.append((w2.data_ptr(), u8, (128, w2_rows, 1), (192, 192 * w2_rows), (128, 128, 1), swizzle, l2[False]))
            maps.append((w2.data_ptr() + 128, u8, (64, w2_rows, 1), (192, 192 * w2_rows), (64, 256, 1), swizzle, l2[False]))
        self.maps = tensor_maps(maps, self.device)
        scales = (tensor(s.view(torch.uint8)) for s in (s13, s2))
        self.views = (to_cute(self.maps[:256], 64), *scales)
        self.packed_maps = to_cute(self.maps[256:512], 64) if self.packed else None
        self.all_maps = to_cute(self.maps, 64)  # TP8 FC2 pairs: maps 0-5
        self.prepared = {}
        # ``box_rows``: rows per latent slot of the MoE tail box this layer publishes into (``MoeTailBox.cap``); the
        # compile calls take it only when it is not 64, so 64-row boots keep their compile keys.
        self.box_rows = box_rows
        self.box_kw = {} if box_rows == 64 else {"box_rows": box_rows}
        # ``K31MK_MOE_RUBIN64``: 9-64-row calls on the vendored grouped engine over the same weight bytes; ``dies``: the SM -> die
        # table its ``place`` returned when the caller re-homed w13 / w2 before building this object (() = die-oblivious).
        self.rubin_rows = rubin64_rows() if self.tp == 16 else {}
        self.wide_rows = wide_rows() if self.tp == 16 else {}
        self.rubin, self.rubin_plans = None, {}
        self.rubin_wide = None  # the wide rows' vendored engine (``self.rubin`` unless DEPAD x WIDE)
        if self.rubin_rows or self.wide_rows:
            if torch.cuda.get_device_capability(self.device) != (10, 7):
                raise ValueError("K31MK_MOE_RUBIN64 needs VR200 (sm_107)")
            from ..rubin.moe import experts_runtime as rubin_runtime

            u8 = torch.uint8

            def vendored(w2_bytes, kw):
                return rubin_runtime.PersistentExperts(
                    w13.view(u8), s13, w2_bytes, s2, beta=beta, linear_beta=linear_beta, dies=dies, q=2, **self.box_kw,
                    **kw,
                )

            rubin_w2, rubin_kw = w2.view(u8), {}
            if self.one96:  # K31MK_MOE_W2_ONE96: both vendored engines read the one 96-byte copy (one object)
                rubin_kw = {"w2_row_bytes": W2_LIVE_BYTES}
            # K31MK_MOE_RUBIN64_W2_DEPAD: the grouped engine reads 96-byte FC2 rows (copied only when it serves rows)
            elif rubin64_w2_depad_enabled() and (self.rubin_rows or not self.wide_rows):
                rubin_w2, rubin_kw = depad_w2(w2), {"w2_row_bytes": W2_LIVE_BYTES}
                if dies:  # the routed weights were placed: home the copy the same way (FC2 die matching)
                    rubin_w2 = place_copy(rubin_w2)
            if rubin64_w2sib_enabled():  # K31MK_MOE_RUBIN64_W2SIB: FC2 records (live w2 bytes + their scales), one box per FC2 job
                if self.wide_rows:
                    raise ValueError("K31MK_MOE_RUBIN64_W2SIB: the wide engine reads the padded w2 (unset K31MK_MOE_WIDE)")
                rubin_w2, rubin_kw = w2sib_records(w2, s2), {"w2_row_bytes": 112}
                if dies:  # home the records like the de-padded copy (FC2 die matching)
                    rubin_w2 = place_copy(rubin_w2)
            self.rubin = vendored(rubin_w2, rubin_kw)
            # DEPAD x WIDE: the wide engine addresses FC2 rows with the padded 128-byte stride, so under
            # K31MK_MOE_RUBIN64_W2_DEPAD it gets its own vendored object over the padded w2 (no copy; the same bytes the
            # knob-off engine reads, so its outputs are the knob-off wide engine's). Otherwise one object serves both.
            two = rubin_kw and self.wide_rows and not self.one96
            self.rubin_wide = vendored(w2.view(u8), {}) if two else self.rubin
        # ``K31MK_MOE_WIDE_DEPAD``: wide rows that run on the 96-byte ``self.rubin`` object; fail closed when
        # no such object exists (the knob must not silently time the padded engine).
        self.wide_depad = wide_depad_rows() if self.tp == 16 else {}
        if self.wide_depad and (self.rubin is None or self.rubin.w2_row_bytes != W2_LIVE_BYTES):
            raise ValueError(
                "K31MK_MOE_WIDE_DEPAD needs the grouped engine's 96-byte w2 copy (K31MK_MOE_RUBIN64 rows with "
                "K31MK_MOE_RUBIN64_W2_DEPAD=1, or K31MK_MOE_W2_ONE96=1)"
            )
        # ``K31MK_MOE_RUBIN64_STAGES``: the vendored grouped prepare reads ``rubin64_stages_rows`` (absent = 7)
        self.rubin64_stages = rubin64_stages_rows() if self.tp == 16 else {}
        if self.rubin64_stages:
            if self.rubin is None:
                raise ValueError("K31MK_MOE_RUBIN64_STAGES needs the vendored grouped engine (K31MK_MOE_RUBIN64 rows)")
            self.rubin.rubin64_stages_rows = self.rubin64_stages
        # ``K31MK_MOE_RUBIN64_POOL_FIX``: the vendored grouped prepare reads ``rubin64_pool_fix_rows``
        self.rubin64_pool_fix = rubin64_pool_fix_rows() if self.tp == 16 else {}
        if self.rubin64_pool_fix:
            if self.rubin is None:
                raise ValueError("K31MK_MOE_RUBIN64_POOL_FIX needs the vendored grouped engine (K31MK_MOE_RUBIN64 rows)")
            self.rubin.rubin64_pool_fix_rows = self.rubin64_pool_fix
        # ``K31MK_MOE_RUBIN64_POOL_ZONE``: the vendored grouped prepare reads ``zones`` (compile_rubin64's zone)
        self.rubin64_pool_zone = rubin64_pool_zone_rows() if self.tp == 16 else {}
        if self.rubin64_pool_zone:
            if self.rubin is None:
                raise ValueError("K31MK_MOE_RUBIN64_POOL_ZONE needs the vendored grouped engine (K31MK_MOE_RUBIN64 rows)")
            self.rubin.zones = {**self.rubin.zones, **self.rubin64_pool_zone}
        # ``K31MK_MOE_RUBIN64_GCUT``: the vendored grouped prepare reads ``rubin64_gcut_rows``
        self.rubin64_gcut = rubin64_gcut_rows() if self.tp == 16 else {}
        if self.rubin64_gcut:
            if self.rubin is None:
                raise ValueError("K31MK_MOE_RUBIN64_GCUT needs the vendored grouped engine (K31MK_MOE_RUBIN64 rows)")
            self.rubin.rubin64_gcut_rows = self.rubin64_gcut
        # ``K31MK_MOE_RUBIN64_PRESTREAM``: the vendored grouped prepare reads ``rubin64_prestream_rows``
        self.rubin64_prestream = rubin64_prestream_rows() if self.tp == 16 else {}
        if self.rubin64_prestream:
            if self.rubin is None:
                raise ValueError("K31MK_MOE_RUBIN64_PRESTREAM needs the vendored grouped engine (K31MK_MOE_RUBIN64 rows)")
            self.rubin.rubin64_prestream_rows = self.rubin64_prestream
        # ``K31MK_MOE_FIN_PAIR`` (the vendored wide prepare reads ``fin_pair_rows`` (absent = off)
        self.fin_pair = fin_pair_rows() if self.tp == 16 else {}
        if self.fin_pair:
            for engine in (self.rubin, self.rubin_wide):
                if engine is not None:
                    engine.fin_pair_rows = self.fin_pair
        # ``K31MK_MOE_WIDE_STAGES``: the vendored wide prepare reads ``wide_stages_rows`` (absent = 7 stages)
        self.wide_stages = wide_stages_rows() if self.tp == 16 else {}
        if self.wide_stages:
            for engine in (self.rubin, self.rubin_wide):
                if engine is not None:
                    engine.wide_stages_rows = self.wide_stages
        # ``K31MK_MOE_FC2_PAIRLOAD``: the vendored wide prepare reads ``fc2_pairload_rows`` (absent = off)
        self.fc2_pairload = fc2_pairload_rows() if self.tp == 16 else {}
        if self.fc2_pairload:
            for engine in (self.rubin, self.rubin_wide):
                if engine is not None:
                    engine.fc2_pairload_rows = self.fc2_pairload
        # ``K31MK_MOE_RUBIN64_LEAN_LOCAL``: the vendored grouped prepare reads ``rubin64_lean_local_rows``
        self.rubin64_lean_local = rubin64_lean_local_rows() if self.tp == 16 else {}
        if self.rubin64_lean_local:
            if self.rubin is None:
                raise ValueError("K31MK_MOE_RUBIN64_LEAN_LOCAL needs the vendored grouped engine (K31MK_MOE_RUBIN64 rows)")
            self.rubin.rubin64_lean_local_rows = self.rubin64_lean_local
        # ``K31MK_MOE_RUBIN8``: B8 calls on a second vendored engine over the same weight bytes (verify tables, its own
        # workspace); die-oblivious unless ``K31MK_MOE_DIES8`` hands it the same ``dies`` table (FC2 die matching).
        self.rubin8 = None
        if self.tp == 16 and rubin8_enabled():
            if torch.cuda.get_device_capability(self.device) != (10, 7):
                raise ValueError("K31MK_MOE_RUBIN8 needs VR200 (sm_107)")
            # His B8 kernel publishes with this layer's tail-box slot stride (box_rows: 128 in a wide-tier boot).
            from ..rubin.moe import experts_runtime as rubin_runtime

            # B8 is the vendored POLLED table's only TP16 row; our routing signals the route mailbox only under
            # K31MK_MOE_ROUTE_POLL, else the experts wait for it by PDL.
            rubin_runtime.POLLED = {**rubin_runtime.POLLED, 16: "8:1" if route_poll8_enabled() else ""}
            u8 = torch.uint8
            dies8 = tuple(dies) if dies8_enabled() else ()
            self.rubin8 = rubin_runtime.PersistentExperts(
                w13.view(u8), s13, w2.view(u8), s2, beta=beta, linear_beta=linear_beta, dies=dies8, q=2,
                **self.box_kw, **({"w2_row_bytes": W2_LIVE_BYTES} if self.one96 else {}),
            )

    def w2_storages(self):
        """Storage addresses of every FC2 tensor this engine and its vendored engines hold (``release_padded_w2``)."""
        held = [self.weights[2], self.w2_live]
        held += [e.weights[2] for e in (self.rubin, self.rubin_wide, self.rubin8) if e is not None]
        return {t.untyped_storage().data_ptr() for t in held if t is not None}

    def prepare(
        self, batch, buffers, out, publish=(), planned=False, twoshot=False, early=False
    ):
        """``publish``: (multicast, parity, rank) of the WaveMoeTail latent slot to
        publish (``twoshot``: the ``common.twoshot`` push box); ``planned``: B>8 calls
        find the plan written (by the routing kernel), else run writes it from x first;
        ``early``: the caller runs ``group`` (pregrouped batches) before ``run``."""
        if self.rubin8 is not None and batch == 8:
            return self.prepare_rubin8(buffers, out, publish, planned, twoshot, early)
        if self.rubin_rows.get(batch):
            return self.prepare_rubin(batch, buffers, out, publish, planned, twoshot, early)
        if self.wide_rows.get(batch):
            return self.prepare_wide_rubin(batch, buffers, out, publish, planned, twoshot)
        mode = "twoshot" if twoshot else bool(publish)
        if self.w8 and batch == 8:
            if mode is True and self.w8_pub != "0":  # the W8 publish epilogue (producer side)
                return self.prepare_w8(batch, buffers, out, planned, publish)
            if mode:  # fail closed: grid_for(8) is the W8 grid under the knob, so no B8 publish call may fall through
                raise ValueError("K31MK_EXP_W8 covers TP16 B8 non-publish calls only")
            return self.prepare_w8(batch, buffers, out, planned)
        grouped = pregrouped(self.tp, batch)
        args: tuple = (batch, self.grid_for(batch), self.device.index, mode)
        args += (streamed(batch), grouped, stealing(batch))
        packed = self.packed if batch <= 8 else 0
        pk1 = self.tp == 8 and batch > 8 and bool(self.pk1_rows.get(batch))  # K31MK_MOE_TP8_PACKED_FC1
        pk2 = self.tp == 8 and batch > 8 and bool(self.pk2_rows.get(batch))  # K31MK_MOE_TP8_PACKED_FC2
        x3 = self.x3_release if self.tp == 16 and batch <= 8 and streamed(batch) and not mode else False  # x3 release
        hb = self.hfull_bar if self.tp == 16 and batch <= 8 and streamed(batch) and not mode else False  # hfull bar
        pace = self.pace if self.tp == 16 and batch <= 8 and streamed(batch) and packed else 0  # cand-08
        # B8 only: the c=1 verify batch; B1-B7 keep the shipped x path (no out-of-bounds TMA rows).
        xtma = self.xtma if batch == 8 else False
        code = compile_experts(
            self.tp, self.experts, *args, activation_floor=self.activation_floor, packed=packed,
            situ_lean=self.situ_lean, strong=self.strong, x3_release=x3, producer_pace=pace, hfull_bar=hb, xtma=xtma,
            packed_fc1=pk1, packed_fc2=pk2,
            **self.box_kw,
        )
        # CTAs wait on other clusters' partials: the whole grid must be resident.
        check_residency(code, self.device.index, args[1], 384, "Kimi K3 MoE experts")
        ws, maps = tensor(buffers["workspace"]), self.views[0]
        plan: tuple | None = None
        capacity = 1 << (batch - 1).bit_length()
        if fc2_pairs(self.tp, capacity, batch > 8 or streamed(batch)):
            maps = self.all_maps
        if packed:
            maps = self.packed_maps
        group = None
        if batch > 8:
            plan = (tensor(buffers["plan"]), compile_plan(batch, self.device.index, True))
            plan = plan[:1] if planned else plan
        if xtma:
            # x -> MXFP8 plan once per call (``planned.EarlyPlanKernel``: stock rule, releases the experts at entry,
            # fences generic -> async proxy for their TMA reads), unless the caller writes
            # it (``planned``: the routing kernel's copy role). Maps: this engine's two weight maps, then the plan's.
            from ..mkv2.experts_stream import plan_maps  # lazy: mkv2 imports this module
            from .planned import compile_early_plan

            weights = self.maps[256:512] if packed else self.maps[:256]
            both = torch.cat((weights, plan_maps(buffers["plan"], batch, self.device, atoms=28)))
            # The kernel acquires these tensor maps before its dependency wait: their device writes must be complete
            # before any launch, not only before the wait. Setup-time only.
            torch.cuda.synchronize(self.device)
            self.xtma_maps.append(both)
            maps = to_cute(both, 64)
            plan = (tensor(buffers["plan"]), compile_early_plan(batch, self.device.index))
            plan = plan[:1] if planned else plan
        if grouped:
            release = early_group(self.tp, batch)
            grouping = compile_group(self.tp, self.experts, *args[:3], release)
            group = grouping, ws, early
        self.prepared[batch] = (out, tensor(out), ws, code, publish, plan, maps, group)

    def prepare_w8(self, batch, buffers, out, planned, publish=()):
        """``K31MK_EXP_W8``: the W4A8 whole-tile engine for this TP16 B8 call. x -> MXFP8 plan as the xtma path
        (``planned.EarlyPlanKernel``, or the caller's when ``planned``); weight + plan maps ``experts_w4a8.maps8``."""
        from ..mkv2 import experts_w4a8 as W8  # lazy: mkv2 imports this module
        from .planned import compile_early_plan, planned_workspace_bytes

        need = planned_workspace_bytes(ExpertsGeometry(self.tp).workspace_bytes)
        if buffers["workspace"].numel() < need:
            raise ValueError("K31MK_EXP_W8: the workspace lacks the claim counters (experts_runtime.scratch with the knob set)")
        if self.one96:  # its maps (``maps8`` / gc specs) address 128-byte FC2 rows
            raise ValueError("K31MK_EXP_W8 reads the padded w2; K31MK_MOE_W2_ONE96 serves B8 on K31MK_MOE_RUBIN8 only")
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        knobs = (("ablate", ()), ("slots", 3 if torch.cuda.get_device_capability(self.device) == (10, 7) else 2), ("xone", True),
                 ("deadline_ns", 0), ("strong", self.strong), ("hfull_bar", bool(self.hfull_bar)), ("k16", True),
                 ("situ_lean", bool(self.situ_lean)), ("served", True))
        w13, _, w2, _ = self.weights
        gc = os.environ.get("K31MK_EXP_W8_GC", "0")  # default off: the five tensormaps as __grid_constant__
        if gc not in ("0", "1", "warm"):  # parameters, no proxy acquire at init; "warm" adds gc_diag warm
            raise ValueError(f"K31MK_EXP_W8_GC={gc!r} not in 0 / 1 / warm")
        if gc != "0":  # gc_rt (weight bases as runtime arguments: one compile for every layer)
            knobs += (("tm_mode", "gc"), ("gc_specs", W8.gc_weight_specs(w13, w2, pointers=False)), ("gc_rt", True)) + (
                (("gc_diag", "warm"),) if gc == "warm" else ())
            self.w8_extra = {**getattr(self, "w8_extra", {}), batch: W8.gc_weight_pointers(w13, w2)}
        if publish:  # (multicast, parity, rank) follow the scalars; the routing weights replace out (run())
            knobs += (("pub_form", self.w8_pub),)
        code, _ = W8.compile_w4a8(self.tp, self.experts, batch, self.w8_grid, self.device.index, self.activation_floor,
                                  knobs, **self.box_kw)
        check_residency(code, self.device.index, self.w8_grid, 384, "Kimi K3 MoE experts (W8)")
        maps = W8.maps8(w13, w2, buffers["plan"], batch, self.device)
        # The kernel acquires these tensor maps before its dependency wait: setup-time sync.
        torch.cuda.synchronize(self.device)
        self.xtma_maps.append(maps)
        plan = (tensor(buffers["plan"]), compile_early_plan(batch, self.device.index))
        plan = plan[:1] if planned else plan
        ws = tensor(buffers["workspace"])
        if publish:  # typed box (a bare Python int multicast address would be taken as 32 bits)
            if len(publish) != 3:
                raise ValueError(f"K31MK_EXP_W8_PUBLISH: publish box must be (multicast, parity, rank), got {len(publish)} entries")
            publish = tuple(v if isinstance(v, t) else t(v) for t, v in zip((c.Int64, c.Int32, c.Int32), publish))
        self.prepared[batch] = (out, tensor(out), ws, code, tuple(publish), plan, to_cute(maps, 64), None)

    def prepare_rubin(self, batch, buffers, out, publish, planned, twoshot, early):
        """``K31MK_MOE_RUBIN64`` rows: the vendored ``RubinGroupedExpertsKernel`` with the caller's publish box and plan
        buffer (``PLAN_BYTES`` layout in both trees); its own workspace (``scratch``'s "rubin_workspace"). Unplanned
        callers get the plan written from x by ``compile_plan`` (FP32 x) at ``run``."""
        if "rubin_workspace" not in buffers:
            raise ValueError(f"B{batch}: K31MK_MOE_RUBIN64 needs scratch() built under the knob (no rubin_workspace)")
        if batch in self.grids:  # a caller-set grid (split SMs) holds for the vendored engine too
            self.rubin.grids[batch] = self.grids[batch]
        his = {"workspace": buffers["rubin_workspace"], "plan": buffers["plan"]}
        self.rubin.prepare(batch, his, out, publish, True, twoshot, early)
        writer = None if planned else compile_plan(batch, self.device.index, True)
        self.rubin_plans[batch] = (writer, tensor(buffers["plan"]))
        self.prepared[batch] = self.rubin.prepared[batch]
        if batch in self.rubin.rubin:  # the vendored prepare's own branch order
            kernel = "RubinExpertsKernel"
        elif self.rubin.grouped64(batch):
            kernel = "RubinGroupedExpertsKernel"
        else:
            kernel = "ExpertsKernel (vendored; not the grouped engine)"
        kernel += " (ONE96: 96-byte w2)" if self.one96 else ""
        if self.rubin64_stages.get(batch) and kernel.startswith("RubinGroupedExpertsKernel"):
            kernel += f" (RUBIN64_STAGES: {self.rubin64_stages[batch]}-stage weight ring)"
        if self.rubin64_pool_fix.get(batch) and kernel.startswith("RubinGroupedExpertsKernel"):
            kernel += f" (RUBIN64_POOL_FIX: pool zone {self.rubin.zones.get(batch, 12)}, acquire-ahead + lead 6)"
        if batch in self.rubin64_pool_zone and kernel.startswith("RubinGroupedExpertsKernel"):
            kernel += f" (RUBIN64_POOL_ZONE: pool zone {self.rubin.zones.get(batch, 12)})"
        if self.rubin64_gcut.get(batch) and kernel.startswith("RubinGroupedExpertsKernel") and self.rubin.dies:
            kernel += " (RUBIN64_GCUT: predicated census ticket + owner-published ranges)"
        if self.rubin64_prestream.get(batch) and kernel.startswith("RubinGroupedExpertsKernel") and self.rubin.dies:  # K31MK_MOE_RUBIN64_PRESTREAM
            kernel += f" (RUBIN64_PRESTREAM: {self.rubin64_prestream[batch]} FC1 boxes per routed id into L2 during the grouping)"
        if getattr(self.rubin, "w2_row_bytes", 128) == 112 and kernel.startswith("RubinGroupedExpertsKernel"):
            kernel += " (RUBIN64_W2SIB: FC2 records, one 112 B box per FC2 job, scales from the box)"
        if self.rubin64_lean_local.get(batch) and kernel.startswith("RubinGroupedExpertsKernel") and not publish and not twoshot:
            kernel += " (RUBIN64_LEAN_LOCAL: transposed local FC2 rows)"
        log_engaged("K31MK_MOE_RUBIN64", batch, kernel, self.rubin.dies)

    def prepare_wide_rubin(self, batch, buffers, out, publish, planned, twoshot=False):
        """``K31MK_MOE_WIDE`` rows: the vendored wide engine (``publish`` = ("local", partial BF16 [>= batch, 3584]) or the
        two-shot box, as its ``prepare_wide``); its workspace and plans in ``scratch``'s "rubin_wide"; the plans are written
        from the FP32 latent at ``run`` (``compile_wide_plan``). ``run`` then takes the top-k weights as BF16 [batch x 16]."""
        if "rubin_wide" not in buffers:
            raise ValueError(f"B{batch}: K31MK_MOE_WIDE needs scratch() built under the knob (no rubin_wide)")
        fold = bool(wide_fold_rows().get(batch))  # : the routing kernel writes the plan blocks
        if planned != fold:
            raise ValueError(f"B{batch}: K31MK_MOE_WIDE plans come from {'the routing kernel (K31MK_MOE_WIDE_FOLD)' if fold else 'compile_wide_plan at run'}; caller planned={planned}")
        engine = self.wide_engine(batch)
        if batch in self.grids:
            engine.grids[batch] = self.grids[batch]
        his = buffers["rubin_wide"]  # the same dict every call: the wide engine keeps its "homes" there
        if twoshot:  # the tail's two-shot push box (receive slots, owners' bases, this rank's slot: PreparedTail.publish)
            publish = ("twoshot", *publish)
        engine.prepare_wide(batch, his, out, publish, True)
        size = PLAN_BYTES * -(-batch // 64)  # the writer was compiled for exactly the plans this row count fills
        writer = None if fold else compile_wide_plan(batch, self.device.index)
        self.rubin_plans[batch] = (writer, tensor(his["plan"][:size]))
        self.prepared[batch] = engine.prepared[batch]
        kernel = "vendored wide engine" + (" (padded w2 beside DEPAD)" if engine is not self.rubin else "")
        kernel += " (ONE96: 96-byte w2)" if self.one96 else ""
        if self.wide_depad.get(batch) and not self.one96:
            kernel += " (WIDE_DEPAD: the grouped engine's 96-byte w2 copy)"
        if self.fin_pair.get(batch):
            kernel += " (FIN_PAIR: two-unit finalize)"
        if self.wide_stages.get(batch):
            kernel += f" (WIDE_STAGES: {self.wide_stages[batch]}-stage weight ring)"
        if self.fc2_pairload.get(batch):
            kernel += " (FC2_PAIRLOAD: 256-row w2 pair boxes)"
        log_engaged("K31MK_MOE_WIDE", batch, kernel, engine.dies)

    def wide_engine(self, batch):
        """The vendored object that runs ``batch``'s wide call: ``self.rubin`` (96-byte w2) on ``K31MK_MOE_WIDE_DEPAD``
        rows, else ``self.rubin_wide``."""
        return self.rubin if self.wide_depad.get(batch) else self.rubin_wide

    def prepare_rubin8(self, buffers, out, publish, planned, twoshot, early):
        """``K31MK_MOE_RUBIN8`` B8: the vendored ``RubinExpertsKernel`` reads the image the routing kernel wrote into
        ``buffers["plan"]`` and publishes into the caller's WaveMoeTail box."""
        if not planned or twoshot:
            raise ValueError("K31MK_MOE_RUBIN8: B8 needs the routing-written image (K31MK_EXP_XPLAN_ROUTE) and no two-shot")
        if "rubin8_workspace" not in buffers:
            raise ValueError("K31MK_MOE_RUBIN8 needs scratch() built under the knob (no rubin8_workspace)")
        if len(publish) == 3:  # typed box (a bare Python int multicast address would be taken as 32 bits)
            publish = tuple(v if isinstance(v, t) else t(v) for t, v in zip((c.Int64, c.Int32, c.Int32), publish))
        his = {"workspace": buffers["rubin8_workspace"], "plan": buffers["plan"]}
        self.rubin8.prepare(8, his, out, publish, True, False, early)
        if 8 not in self.rubin8.rubin:
            raise ValueError("K31MK_MOE_RUBIN8: the vendored engine did not take RubinExpertsKernel at B8")
        self.prepared[8] = self.rubin8.prepared[8]
        # The vendored prepare compiles with the table only when the grid has one CTA per SM (its ``len(dies)``).
        matched = bool(self.rubin8.dies) and self.rubin8.grid_for(8) == len(self.rubin8.dies)
        if dies8_enabled() and not matched:
            raise ValueError("K31MK_MOE_DIES8: no SM -> die table reached the B8 engine (no trusted die map, or grid != SMs)")
        log_engaged("K31MK_MOE_RUBIN8", 8, "RubinExpertsKernel" + (" (ONE96: 96-byte w2)" if self.one96 else ""), matched)
        from ..rubin.moe import experts_runtime as rubin_runtime

        polled = rubin_runtime.polled(16, 8, self.device)
        if polled != route_poll8_enabled():  # the routing kernel signals iff these experts poll
            raise ValueError(f"K31MK_MOE_ROUTE_POLL: vendored B8 experts polled={polled}, knob={route_poll8_enabled()}")
        log_engaged("K31MK_MOE_ROUTE_POLL", 8, "RubinExpertsKernel polled", polled)

    def group(self, ids):
        """Pregrouped batches prepared ``early``: the unit-table kernel (run it on the
        stream after routing; ``run`` then skips it)."""
        if ids.shape[0] in self.rubin_plans:
            return  # the grouped engine groups in-kernel
        group = self.prepared[ids.shape[0]][7]
        if group is not None and group[2]:
            self.launch_group(ids, group)

    def launch_group(self, ids, group):
        code, ws, _ = group
        code(tensor(ids), ws, c.Int32(ids.shape[0]), stream(self.device))

    def grid_for(self, batch):
        if self.rubin8 is not None and batch == 8:  # the vendored grid: every SM at TP16 B8
            return self.rubin8.grid_for(8)
        if self.wide_rows.get(batch) and self.rubin_wide is not None:
            return self.grids.get(batch, self.wide_engine(batch).grid_for(batch))
        if self.rubin_rows.get(batch) and self.rubin is not None:
            return self.grids.get(batch, self.rubin.grid_for(batch))
        if self.w8 and batch == 8:
            return self.w8_grid
        return self.grids.get(batch, self.grid)

    def run(self, x, ids, weights=None):
        batch = x.shape[0]
        if self.rubin8 is not None and batch == 8:
            return self.rubin8.run(x, ids, weights)
        if batch in self.rubin_plans:
            writer, plan_t = self.rubin_plans[batch]
            if writer is not None:
                writer(to_cute(x, 16), plan_t, stream(self.device))
            engine = self.wide_engine(batch) if self.wide_rows.get(batch) else self.rubin
            return engine.run(x, ids, weights)
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
        # W8 gc_rt's two weight base addresses (empty unless K31MK_EXP_W8_GC is set: the default call is unchanged)
        code(*args, c.Int32(batch), *self.scalars, *box, *getattr(self, "w8_extra", {}).get(batch, ()), stream(self.device))
        return out
