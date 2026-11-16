# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""W4A8 whole-tile routed experts, TP16 B <= 8 (opt-in ``K31MK_EXPERTS_W4A8WT``, default off).

The W4A4 engine's structure (packed E2M1 weights in 64 KB TMA ops of 4 K stages, whole-tile FC1
tiles claimed as tickets, FC2 tile pairs claimed in expert order, the x plan in one TMA) with the shipped W4A8
numerics, so no precision change:
- x is the MXFP8 plan of the FP32 latent (the stock input rule, ``planned.EarlyPlanKernel``), loaded as 28 128-K
  atoms [8 rows][128 B] (128B swizzle) and its scale words in one TMA each (``experts_stream.plan_maps``);
- FC1 / FC2 are ``tcgen05.mma.kind::mxf8f6f4.block_scale`` with K = 64 on packed E2M1 weights (instruction-descriptor
  bit 31; two per-32 scales per MMA, bytes ID and ID + 1) and E4M3 activations;
- SiTU, the MXFP8 hidden quantization and the FC1 -> FC2 hidden hand-off are the shipped TP16 engine's
  (``ExpertsKernel.store_hidden`` / ``poll_hidden``: 2176-byte compact images per unit, 0xFF byte sentinels, the
  shipped re-arm), so the workspace image layout is the shipped engine's.
Only the FP32 accumulation grouping differs from the shipped engine (whole-tile FC1 in one CTA's TMEM over K64 MMAs
instead of K32 MMAs with split-K partials): a REORDER.

Sync: the W4A4 engine's ring / ticket / accumulator / x-ready protocols (edges L1-1..L1-6)
are unchanged; the hidden hand-off is the shipped engine's (Q-edges)."""

import functools
import os

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass.cute.runtime import make_fake_stream
from cutlass.experimental.cuda import tensor_map as TM

from ..common.host import fake
from ..common.primitives import asm, named_barrier, shared_address, smem_desc, store_words, thread, view
from ..common.primitives import bf16, fence_tmem, fma, load_tmem, load_volatile, multicast_words, pack_pair  # pd form
from ..common.primitives import sleep_ns  # f2bo form
from ..moe import experts as EX
from ..moe import experts_runtime as ER
from ..moe import planned as PLN
from . import experts_stream as MV
from . import experts_w4a4 as W4

SLOT, SLOT_SF, MAPS = W4.SLOT, W4.SLOT_SF, W4.MAPS4
X_BYTES = 28 * 1024  # MXFP8 x: 28 atoms of [8 rows][128 B]
# K31MK_EXP_W8_PUBLISH=pd: misc SMEM word of the prepoll hint (free in the W8 layout: meta 4224..4272 (3 slots),
# planned's x_delay stash 4400..4416 unused by the W8, unit queue from 4608).
PD_DONE = 4416
PUB_FORMS = ("", "fin", "pd", "f2", "f2bo", "f2sp", "finf", "f2spf", "f2spfcw")  # f2* = measurement forms
W4A8WT_FREE = W4.W4A4_FREE  # the W4A4 engine's grids (FC1 tile balance); ``K31MK_EXPERTS_W4A8WT_GRID`` overrides


def enabled():
    return os.environ.get("K31MK_EXPERTS_W4A8WT", "0") == "1"


def idesc8(sid):
    """kind::mxf8f6f4 descriptor, E2M1 A x E4M3 B, UE8M0, M128 N8, dense K64 (bit 31), scale IDs."""
    return 142738048 | 1 << 31 | sid << 29 | sid << 4


def _group8(steps, a_cps, b_cps, nomma, commit_done, commit_release, pre_release):
    """``experts_w4a4._group`` with W4A8 MMAs (same operand registers)."""
    cp = "tcgen05.cp.cta_group::1.32x128b.warpx4"
    arrive = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
    code = ["{ .reg .pred e,p0,pt,pc,pr; .reg .b32 c<8>,sa,sb,id; .reg .b64 d<8>,xa,xb;",
            "elect.sync _|e,0xffffffff; setp.eq.b32 pt,0,0; setp.ne.b32 pc,$7,0; and.pred pc,pc,e;",
            "setp.ne.b32 p0,$8,0;"]
    if pre_release:
        code.append(f"setp.ne.b32 pr,$11,0; and.pred pr,pr,e; @pr {arrive} [$11];")
    for i, (col, off) in enumerate(b_cps):
        code.append(f"add.u32 c{i},$5,{col}; add.s64 d{i},$6,{off}; @pc {cp} [c{i}],d{i};")
    for i, (col, off) in enumerate(a_cps):
        code.append(f"add.u32 c{i + 4},$3,{col}; add.s64 d{i + 4},$4,{off}; @e {cp} [c{i + 4}],d{i + 4};")
    for k, (aoff, boff, acol, bcol, sid) in enumerate(steps):
        code.append(f"add.s64 xa,$1,{aoff}; add.s64 xb,$2,{boff}; add.u32 sa,$3,{acol}; add.u32 sb,$5,{bcol};")
        code.append(f"mov.b32 id,{idesc8(sid)};")
        if not nomma:
            code.append("@e tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale "
                        f"[$0],xa,xb,id,[sa],[sb],{'p0' if k == 0 else 'pt'};")
    if commit_done:
        code.append(f"setp.ne.b32 pr,$9,0; and.pred pr,pr,e; @pr {arrive} [$9];")
    if commit_release:
        code.append(f"setp.ne.b32 pr,$10,0; and.pred pr,pr,e; @pr {arrive} [$10];")
    code.append("}")
    return "\n".join(code)


@asm()
def group8(d, a, b, a_scales, b_scales, accumulate, done, release, pre, *, part, nomma=False):
    """One W4A8 MMA group (offsets in 16-byte units). "fc1": one K256 stage, 4 K64 MMAs; A = the W4A4 slot layout
    [chunk (8 KB)][128 rows][64 B] (64B swizzle, K64 = 32 B); B = the x stage [2 atoms (1 KB)][8 rows][128 B] (128B
    swizzle, K64 = 64 B); scales as W4A4. "fc2t16": K 0-127 from A chunk 0 (K64 at 0 / 32 B) with hidden atom 0 (0 /
    64 B), K 128-191 from the upper 32 B of A chunk 1 (W2 elements 128-191) with hidden atom 1."""
    if part == "fc1":
        steps = [((j // 2) * 512 + (j % 2) * 2, (j // 2) * 64 + (j % 2) * 4, 4 * (j // 2), 4 * (j // 2), (j % 2) * 2)
                 for j in range(4)]
    else:
        assert part == "fc2t16", part
        steps = [(j * 2, j * 4, 0, 0, j * 2) for j in range(2)]
        steps += [(1024 + 2, 64, 4, 4, 0)]
    a_cps, b_cps = [(0, 0), (4, 32)], [(0, 0), (4, 8)]
    code = _group8(steps, a_cps, b_cps, nomma, True, True, True)
    kinds = "rllrlrlrrrrr"
    operands = (d, a, b, *a_scales, *b_scales, accumulate, done, release, pre)
    values = [(Uint64 if k == "l" else Uint32)(v) for k, v in zip(kinds, operands)]
    return code, ",".join(kinds) + ",~{memory}", values


class W4A8ExpertsKernel(W4.W4A4ExpertsKernel):
    """TP16 only. SMEM: ``slots`` 64 KB rings + 4 KB scale slots, x 28 KB, hidden images [hbufs x 2 KB] (shipped TP16
    layout), x scales, hidden scales, misc (5120 B); in the large-SMEM mode."""

    TPS = (16,)

    def __init__(self, *args, strong=False, hfull_bar=False, k16=False, situ_lean=False, served=False, fc2_ahead=1,
                 entry_prefetch=False, fc2_quad=False, route_bar=1, tm_mode="", gc_specs=(), gc_diag="", gc_rt=False,
                 pub_form="", **kwargs):
        super().__init__(*args, **kwargs)
        # ``pub_form`` (K31MK_EXP_W8_PUBLISH, default "" = off): the published B8 call (producer side). The
        # experts stage FC2 rows (``ExpertsKernel.fc2``'s publish path) and the epilogue warps of CTAs 0..55 sum each
        # (token, 64-column) line's 16 staged rows in slot order (FP32 FMA of BF16 x weight) and multicast it into latent slot
        # (parity, rank) of the [3, 16, 64, 3584] mailbox: "fin" = ``ExpertsKernel.finalize`` after the end marker (the
        # served TP16 B9+ code); "pd" = the same with the line's words loaded once this CTA's producer claimed its last FC2
        # op (``epilogue_pd`` / ``publish_pd``). A publish compile without a form is refused (no silent publish path).
        if pub_form not in PUB_FORMS or bool(pub_form) != bool(self.publish) or self.publish == "twoshot":
            raise ValueError(f"W4A8ExpertsKernel: pub_form={pub_form!r} with publish={self.publish!r} (a form iff publish)")
        if pub_form and not (self.capacity == 8 and self.grid * 8 >= 8 * 56 and not self.chain2 and not self.grouped):
            raise ValueError("W4A8ExpertsKernel: pub_form needs capacity 8 and grid >= 56 (one line per epilogue warp)")
        assert not pub_form or (4224 + 16 * self.stages <= PD_DONE and PD_DONE + 4 <= 4608), self.stages
        # finf / f2spf = fin / f2sp with the fenced one-thread trigger (experts_stream kernel_body, ``pub_fence``)
        # f2spfcw = f2spf + cheap stage pack + lane-word stage stores (``fc2`` below; measured faster per layer than f2spf)
        self.p6_cheap = self.p6_lane = pub_form == "f2spfcw"
        pub_form = "f2spf" if self.p6_cheap else pub_form
        self.pub_fence = pub_form in ("finf", "f2spf")
        self.pub_form = pub_form[:-1] if self.pub_fence else pub_form
        if self.pub_form == "f2sp" and 3 * self.grid < self.capacity * 56:  # owners k < 3 per CTA
            raise ValueError(f"W4A8ExpertsKernel: pub_form={pub_form!r} needs 3 x grid >= {self.capacity * 56} (grid {self.grid})")
        self.entry_prefetch = bool(entry_prefetch)  # see StreamExpertsKernel.kernel; default off
        if fc2_quad and not (self.tp == 16 and self.jobs % 2 == 0 and 4 * self.fc2_scale_cols <= 32
                             and 4 * self.w2_tile_scale_bytes <= SLOT_SF and 2 * 256 * 64 * self.w2_chunks <= SLOT):
            raise ValueError("W4A8ExpertsKernel: fc2_quad needs TP16 (4 tiles of weights / scales per slot)")
        self.fc2_quad = bool(fc2_quad)  # see W4A4ExpertsKernel.fc2_quad; default off
        if route_bar not in (1, 9):
            raise ValueError(f"W4A8ExpertsKernel: route_bar={route_bar!r} not in (1, 9)")
        self.route_bar = route_bar  # routing's 128-thread join ID (1 = unchanged; 9 = unused elsewhere)
        if fc2_ahead not in (1, 2):
            raise ValueError(f"W4A8ExpertsKernel: fc2_ahead={fc2_ahead!r} not in (1, 2)")
        self.fc2_ahead = fc2_ahead  # see W4A4ExpertsKernel.fc2_ahead; 1 = unchanged
        # Diagnosis (benchmark-only, default ""): "gpu" = .gpu-scope proxy acquires; "pf" = prefetch only (no
        # proxy fence: tensormaps are host-written before launch; diagnosis only); "t128" = the .sys acquires by tid 128.
        # "late" = the producer (tid 0) issues the .sys acquires right before its first TMA instead of at init;
        # "one" = one acquire (map 0) + prefetch of the other four (diagnosis only).
        # "par" = the five .sys acquires by five threads (lane 0 of warps 4-8) at init, used by tid 0 after the CTA barrier.
        # "gc" = the five tensormaps as __grid_constant__ kernel parameters (W4A8GcExpertsKernel; compile_w4a8 picks
        # it), encoded by the host function from ``gc_specs`` (gc_weight_specs: the three weight maps) and the runtime plan
        # pointer; init prefetches them, no proxy acquire (CUDA Programming Guide 4.12.2.2.3).
        if tm_mode not in ("", "gpu", "pf", "t128", "late", "one", "par", "gc"):
            raise ValueError(f"W4A8ExpertsKernel: tm_mode={tm_mode!r}")
        if (tm_mode == "gc") != (len(gc_specs) == 3):
            raise ValueError("W4A8ExpertsKernel: gc needs 3 gc_specs")
        if tm_mode == "gc" and (self.entry_prefetch or self.fc2_quad):
            raise ValueError("W4A8ExpertsKernel: tm_mode gc runs on the served form only")
        # Controls (benchmark-only, gc only): "pf" = the gc launch (five parameter maps encoded and passed) but the
        # kernel uses the GLOBAL map buffer with prefetch only (no acquire: NON-COMPLIANT, diagnosis only); "warm" = lane 0 of warps
        # 4-8 each issue one generic 8-byte load of one parameter descriptor at entry (read-only; no ordering role).
        if gc_diag not in ("", "pf", "warm") or (gc_diag and tm_mode != "gc"):
            raise ValueError(f"W4A8ExpertsKernel: gc_diag={gc_diag!r} needs tm_mode gc")
        self.tm_mode, self.gc_specs = tm_mode, tuple(gc_specs)
        self.gc_diag = gc_diag
        # ``gc_rt`` (gc only): the two weight base addresses (W13, W2) come as two Int64 RUNTIME arguments after the
        # scalars instead of compile-time constants in gc_specs (whose pointer fields must then be 0), so one compiled function
        # serves every MoE layer (one compile and one kernel-cache entry instead of one per layer). Same kernel PTX as gc.
        if gc_rt and (tm_mode != "gc" or any(spec[0] != 0 for spec in self.gc_specs)):
            raise ValueError("W4A8ExpertsKernel: gc_rt needs tm_mode gc with pointer-free gc_specs (pointer fields 0)")
        self.gc_rt = bool(gc_rt)
        assert self.tp == 16, self.tp
        self.x_bytes = X_BYTES
        # The served integration form (``PersistentExperts`` under ``K31MK_EXP_W8``) adds the Run-form fixes that
        # apply to this engine's code; the standalone ``W4A8WTExperts`` keeps all four off (its PTX unchanged):
        # ``strong`` (K-17, ``experts.STRONG_MODES``): the hidden hand-off (``ExpertsKernel.store_hidden`` /
        # ``poll_hidden``, bound to this kernel) and the hidden re-arm (``rearm`` below) in the strong form;
        # ``hfull_bar``: the synchronizing join before lane 0's hfull release in ``poll_hidden``;
        # ``k16`` (K-16): ``fence.proxy.async.global`` by the x-plan TMA issuer after the dependency wait;
        # ``situ_lean``: SiTU reciprocals once per thread (bitwise equal).
        if strong not in EX.STRONG_MODES:
            raise ValueError(f"W4A8ExpertsKernel: strong={strong!r} not in {tuple(EX.STRONG_MODES)}")
        # ``served``: the form PersistentExperts compiles under K31MK_EXP_W8. Fail closed: it
        # needs the compliant hand-off (strong True = K-17c), the hfull join, the K-16 fence and the one-TMA x
        # (xone), and it adds the three edges the verifiers found open on this engine:
        # ``meta_bar`` (A, R4 / K-10d: ring metadata WAR), ``queue_bar`` (B: every loader lane releases its read on
        # queue-empty, count 32, in ``pop_unit``), ``xexit`` (C: a CTA with no FC1 op waits xready[0] at the end marker,
        # so no x TMA is in flight at exit; PTX 9.7.10.28.1, as ``ExpertsKernel.consume`` under xtma).
        # ``hfull_bar`` is set here, after ``ExpertsKernel.__init__`` (whose guard excludes paired / steal engines for
        # their barrier IDs), so the guard for this engine is stated here: the hfull join only in the served form,
        # whose named-barrier IDs (0, 1 / 4 epilogue groups, 12 meta, 13 hfull; each used by one role)
        # ``test_served_ptx`` checks.
        if served not in (False, True):
            raise ValueError(f"W4A8ExpertsKernel: served={served!r} not a bool")
        if served and not (strong is True and hfull_bar is True and k16 is True and self.xone and (not self.publish or pub_form)):
            raise ValueError("W4A8ExpertsKernel(served=True) needs strong=True, hfull_bar=True, k16=True, xone, no "
                             f"publish (got strong={strong!r} hfull_bar={hfull_bar!r} k16={k16!r} xone={self.xone})")
        if hfull_bar and not served:
            raise ValueError("W4A8ExpertsKernel: hfull_bar only in the served form (its barrier-ID map is checked)")
        self.strong, self.hfull_bar, self.k16, self.situ_lean = strong, bool(hfull_bar), bool(k16), bool(situ_lean)
        self.served = self.meta_bar = self.queue_bar = self.xexit = served
        if served:
            self.bar_meta = 8 * (2 * self.stages + 50)  # misc 448: after xready[14] (336 + 112), below accmeta (600)
            assert self.bar_xready + 8 * 14 <= self.bar_meta and self.bar_meta + 8 * self.stages <= 600, self.bar_meta

    def layout(self, stages):
        geometry = ER.ExpertsGeometry(16)
        self.h_image_bytes, self.h_compact_bytes = geometry.h_image_bytes, geometry.h_compact_bytes
        self.h_scale_bytes = geometry.h_scale_bytes
        slots = stages
        assert 2 <= slots <= 3, slots
        self.stages = slots
        self.slot_bytes = SLOT
        self.off_sfa = slots * SLOT
        self.off_x = self.off_sfa + slots * SLOT_SF
        self.off_h = self.off_x + X_BYTES
        self.off_xsf = self.off_h + self.hbufs * self.h_image_bytes
        self.off_hsf = self.off_xsf + 28 * 128 + 384
        self.off_misc = self.off_hsf + self.hbufs * self.h_scale_bytes
        self.smem_bytes = self.off_misc + 5120
        self.bar_full, self.bar_empty, self.bar_accfull = (0, 8 * slots, 16 * slots)
        acc = self.bar_accfull
        self.bar_accempty, self.bar_hfull = (acc + 128, acc + 256)
        self.bar_hempty, self.bar_xready = (acc + 272, acc + 288)
        self.off_meta = self.off_misc + 4224
        self.off_queue = self.off_misc + 4608
        assert self.off_misc <= 1 << 18 and self.smem_bytes <= PLN.large_smem_alloc(), (self.off_misc, self.smem_bytes)
        assert self.off_x % 1024 == 0 and self.off_h % 1024 == 0, (self.off_x, self.off_h)

    @cute.jit
    def fetch_x(self, sm, maps):
        """The whole MXFP8 x plan (28 atoms) and its scale words onto xready[0] (map +768 / +896)."""
        if cutlass.const_expr(self.k16):
            PLN.fence_proxy_async_global()  # K-16: the plan is the predecessor grid's generic stores (after the wait)
        MV.StreamExpertsKernel.fetch_x(self, sm, maps)

    @cute.jit
    def rearm(self, workspace, call_count):
        """The other parity's 128 hidden images; under ``strong`` the shipped engine's K-17 form (``st.relaxed.gpu``
        v4 words, ``ExpertsKernel.rearm`` non-grouped), else ``PlannedExpertsKernel.rearm`` (weak, unchanged)."""
        if cutlass.const_expr(EX.handoff_scope(self) is not None):
            _, _, lane, cta = thread()
            items: cutlass.Constexpr = 128 * self.h_compact_bytes // 16
            lo, hi = (cta * items // self.grid, (cta + 1) * items // self.grid)
            other = self.image(workspace, call_count & 1 ^ 1, 0)
            for i in range(lo + lane, hi, 32):
                store_words(other + i * 16, (Uint32(4294967295),) * 4, scope=EX.handoff_scope(self))
        else:
            PLN.PlannedExpertsKernel.rearm(self, workspace, call_count)

    @cute.jit
    def store_hidden(self, sm, workspace, values, tile, parity, scalars):
        EX.ExpertsKernel.store_hidden(self, sm, workspace, values, tile, parity, scalars)

    @cute.jit
    def poll_hidden(self, sm, source, buffer: Int32):
        EX.ExpertsKernel.poll_hidden(self, sm, source, buffer)

    @cute.jit
    def pop_unit(self, sm, popped):
        """``ExpertsKernel.pop_unit``; under ``queue_bar`` (served) EVERY loader lane releases its own read of units[slot]:
        each of the 32 lanes arrives (``mbarrier.arrive``, .release.cta) on queue-empty[slot], which counts 32 when
        ``queue_bar`` (``W4A4ExpertsKernel.initialize``), so the producer's acquire wait (push + 8) completes only after all
        32 releases (PTX 9.7.15.16) and its rewrite is ordered after every lane's read. Per-lane
        releases, not a bar.sync join + one arrive: racecheck credits this form, not the join."""
        units, full, empty = self.queue(sm)
        slot = popped % 8
        self.sync(full.iterator + slot, popped // 8 & 1)
        unit = units[slot]
        if cutlass.const_expr(self.queue_bar):  # B: every lane's release (queue-empty counts 32 under queue_bar)
            cute.arch.mbarrier_arrive(empty.iterator + slot)
        else:
            cute.arch.sync_warp()
            if thread()[2] == 0:
                cute.arch.mbarrier_arrive(empty.iterator + slot)
        return unit

    @cute.jit
    def initialize(self, sm, maps, batch):
        """``W4A4ExpertsKernel.initialize``; pd: tid 0 zeroes the prepoll hint word (route()'s CTA barrier orders it before
        every epilogue read and before the producer's store of 1)."""
        tmem_base = W4.W4A4ExpertsKernel.initialize(self, sm, maps, batch)
        if cutlass.const_expr(self.pub_form == "pd"):
            if thread()[0] == 0:
                st_relaxed_cta_shared_u32(sm.iterator + self.off_misc + PD_DONE, Uint32(0))
        return tmem_base

    @cute.jit
    def push_unit(self, sm, pushed, unit):
        """``ExpertsKernel.push_unit``; pd: the producer's end-of-claims push (unit -1) also sets the prepoll hint (a
        hint only: the epilogue validates every word it uses by the sentinel, so no ordering rests on it)."""
        if cutlass.const_expr(self.pub_form == "pd"):
            if unit < 0:
                st_relaxed_cta_shared_u32(sm.iterator + self.off_misc + PD_DONE, Uint32(1))
        return EX.ExpertsKernel.push_unit(self, sm, pushed, unit)

    @cute.jit
    def epilogue(self, sm, workspace, output, tmem, ranges, batch, parity, scalars):
        if cutlass.const_expr(self.pub_form in ("pd", "f2", "f2bo", "f2sp")):
            self.epilogue_pd(sm, workspace, output, tmem, ranges, batch, parity, scalars)
        else:
            MV.StreamExpertsKernel.epilogue(self, sm, workspace, output, tmem, ranges, batch, parity, scalars)

    @cute.jit
    def fc2(self, sm, workspace, output, values, unit, tile, ranges):
        """``ExpertsKernel.fc2``; f2spfcw: the stage word is the bare bf16x2 with only the full-word 0xFFFFFFFF sentinel replaced by 0x7FC07FC0 (= pack_pair
        "stage" of two -NaN halves; finite values: the same bits as pack_pair "stage"), stored by the lane holding column c (lane & 8 == 0) as one
        st.relaxed.gpu.u32 at word (row + c / 2): the same words at the same addresses as the gathered 16 B store. Other forms: ExpertsKernel.fc2."""
        if cutlass.const_expr(self.p6_cheap or self.p6_lane):
            _, warp, lane, _ = thread()
            slots = view(sm.iterator, self.off_misc + 2176, cutlass.Int8, 1024)
            for t in cutlass.range_constexpr(self.width):
                slot = Int32(slots[unit * 8 + t])
                if slot >= 0:
                    other = cute.arch.shuffle_sync_down(values[t], 8)
                    if cutlass.const_expr(self.p6_cheap):
                        word = Uint32(pack_pair(values[t], other, policy=None))
                        if word == Uint32(4294967295):
                            word = Uint32(2143322048)  # 0x7FC07FC0
                    else:
                        word = Uint32(pack_pair(values[t], other, policy="stage"))
                    if cutlass.const_expr(self.p6_lane):
                        if lane & 8 == 0:
                            column = tile * 128 + (warp & 3) * 32 + 4 * (lane % 8) + lane // 8
                            dst = workspace.iterator + self.ws_stage + (self.pair(t, slot) * 1792 + column // 2) * 4
                            store_words(dst, (word,), scope=EX.handoff_scope(self))
                    else:
                        q = lane & 3
                        words = [cute.arch.shuffle_sync(word, s + 2 * q) for s in (0, 16, 1, 17)]
                        row = self.pair(t, slot) * 1792 + tile * 64 + (warp & 3) * 16
                        stage = workspace.iterator + self.ws_stage + (row + 4 * q) * 4
                        if lane < 4:
                            store_words(stage, words, scope=EX.handoff_scope(self))
        else:
            EX.ExpertsKernel.fc2(self, sm, workspace, output, values, unit, tile, ranges)

    @cute.jit
    def epilogue_pd(self, sm, workspace, output, tmem, ranges, batch, parity, scalars):
        """``ExpertsKernel.epilogue`` (no chain2) + the prepoll: warp w of CTA c owns line 8c + w - 4 (< batch x 56: CTAs
        0..55); once the hint is set it loads the line's 16 staged words (ld.volatile) and routing weights once, between
        accumulators; after the end marker ``publish_pd``."""
        _, warp, lane, cta = thread()
        ew, eg = (warp & 3, (warp - 4) // 4)
        full = view(sm.iterator, self.off_misc + self.bar_accfull, Uint64, 16)
        empty = view(sm.iterator, self.off_misc + self.bar_accempty, Uint64, 16)
        meta = view(sm.iterator, self.off_misc + 600, Int32, 64)
        line = cta * 8 + warp - 4
        pending = line < batch * 56
        if cutlass.const_expr(self.pub_form == "f2sp"):  # <= 3 owner warps per CTA over all CTAs
            k = 2 * ((warp - 4) % 4) + (warp - 4) // 4
            line = k * self.grid + cta
            pending = (k < 3) & (line < batch * 56)
        token, column = (line // 56, line % 56 * 64)
        source = workspace.iterator + self.ws_stage
        source += (Int64(token) * 16 * 1792 + column // 2 + lane) * 4
        words = cute.make_rmem_tensor(16, Uint32)
        scale = cute.make_rmem_tensor(16, Float32)
        loaded = Int32(0)
        acc, phase, ended = (eg, Int32(0), False)
        while not ended:
            if cutlass.const_expr(self.pub_form == "pd"):  # the f2 forms carry no prepoll (pd's code unchanged)
                if pending & (loaded == 0):
                    if ld_relaxed_cta_shared_u32(sm.iterator + self.off_misc + PD_DONE) != Uint32(0):
                        for slot in cutlass.range_constexpr(16):
                            scale[slot] = Float32(output[token * 16 + slot])
                            words[slot] = load_volatile(source + slot * 7168)[0]
                        loaded = Int32(1)
            self.sync(full.iterator + acc, phase, True)
            kind, a, b = (meta[4 * acc], meta[4 * acc + 1], meta[4 * acc + 2])
            if kind == 2:
                ended = True
            else:
                raw = load_tmem(tmem + (Uint32(32 * ew) << 16) + acc * 8, 8)
                values = cute.make_rmem_tensor(self.width, Float32)
                for t in cutlass.range_constexpr(self.width):
                    values[t] = raw[t]
                fence_tmem(before=True)
                cute.arch.sync_warp()
                if lane == 0:
                    cute.arch.mbarrier_arrive(empty.iterator + acc)
                acc += 2
                if acc >= 16:
                    acc -= 16
                    phase ^= 1
                if kind == 0:
                    if self.partials(workspace, values, a, ranges, batch):
                        self.store_hidden(sm, workspace, values, a, parity, scalars[:3])
                else:
                    self.fc2(sm, workspace, output, values, a, b, ranges)
        if pending:
            self.publish_pd(words, scale, loaded, token, column, source, output, scalars[3:])

    @cute.jit
    def publish_pd(self, words, scale, loaded, token, column, source, weights, box):
        """``ExpertsKernel.finalize`` of one line from the prepolled words: published at once if no word is the 0xFFFFFFFF
        sentinel (warp vote), else polled as finalize; then finalize's re-arm, slot-order FMA, pack and multicast."""
        lane = thread()[2]
        if loaded == 0:
            for slot in cutlass.range_constexpr(16):
                scale[slot] = Float32(weights[token * 16 + slot])
        ready = loaded != 0
        if ready:
            bad = False
            for slot in cutlass.range_constexpr(16):
                bad |= words[slot] == Uint32(4294967295)
            ready = cute.arch.vote_all_sync(not bad)
        first = True
        while not ready:
            if cutlass.const_expr(self.pub_form == "f2bo"):  # back off 256 ns after a failed round
                if not first:
                    sleep_ns(256)
            first = False
            bad = False
            for slot in cutlass.range_constexpr(16):
                words[slot] = load_volatile(source + slot * 7168)[0]
                bad |= words[slot] == Uint32(4294967295)
            ready = cute.arch.vote_all_sync(not bad)
        low, high = Float32(0), Float32(0)
        for slot in cutlass.range_constexpr(16):
            store_words(source + slot * 7168, (Uint32(4294967295),), scope=EX.handoff_scope(self))
            low = fma(bf16(words[slot]), scale[slot], low)
            high = fma(bf16(words[slot], True), scale[slot], high)
        word = pack_pair(low, high, policy="latent")
        packed = [cute.arch.shuffle_sync(word, 4 * (lane & 7) + i) for i in range(4)]
        if lane < 8:
            mc, parity, rank = box
            offset = ((parity * self.tp + rank) * self.box_rows + token) * 3584 + column
            offset += 8 * lane
            multicast_words(mc + Int64(offset) * 2, packed)

    @cute.jit
    def consume(self, sm, tmem, workspace=None):
        """``W4A4ExpertsKernel.consume`` with W4A8 groups and 128B-swizzled x / hidden descriptors."""
        lane = thread()[2]
        slots, H = (self.stages, self.hbufs)
        nomma: cutlass.Constexpr = "mma" in self.ablate
        bars: tuple = (self.bar_full, self.bar_empty, self.bar_accfull, self.bar_accempty, self.bar_xready)
        full, empty, accfull, accempty, xready = (
            view(sm.iterator, self.off_misc + o, Uint64, n) for o, n in zip(bars, (slots, slots, 16, 16, 14))
        )
        hfull, hempty = self.hidden_barriers(sm)
        meta = view(sm.iterator, self.off_meta, Int32, 4 * slots)
        accmeta = view(sm.iterator, self.off_misc + 600, Int32, 64)
        a0 = W4.desc64(sm.iterator)
        x0, h0 = (smem_desc(sm.iterator + o, True) for o in (self.off_x, self.off_h))
        asf0, xsf0, hsf0 = (smem_desc(sm.iterator + o, False) for o in (self.off_sfa, self.off_xsf, self.off_hsf))
        xmask, ended = (Uint32(0), False)
        uses, acc, count = (Int32(0), Int32(0), Int32(0))
        current, ordinal, hbuffer = (Int32(-1), Int32(-1), Int32(0))
        while not ended:
            slot = count % slots
            self.sync(full.iterator + slot, count // slots & 1, True)
            kind, a, b, c = (meta[4 * slot + i] for i in range(4))
            cute.arch.sync_warp()
            if cutlass.const_expr(not self.meta_bar):  # W4A4's U1-W3 generic arrive (empty counts 2); served: R4 below
                if lane == 0:
                    cute.arch.mbarrier_arrive(empty.iterator + slot)
            released = shared_address(empty.iterator + slot)
            wcol = tmem + 128 + slot * 32
            if kind == 2:
                if cutlass.const_expr(self.xexit):  # C: no x TMA in flight when this CTA exits (9.7.10.28.1)
                    if xmask == 0:
                        self.sync(xready.iterator, 0, True)
                if (lane == 0) & (current >= 0):
                    W4.commit(hempty.iterator + hbuffer)
                for _ in cutlass.range(2, unroll=1):
                    self.acquire(accempty, acc, uses)
                    self.signal(accmeta, accfull, acc, (kind, a, b, c), 2)
                    uses += 1
                    acc = uses % 16
                ended = True
            elif kind == 0:  # FC1 op of tile a: K stages b .. b + c - 1
                for s in cutlass.range_constexpr(4):
                    if s < c:
                        ks = b + s
                        first, last = (ks == 0, ks == 13)
                        if first:
                            self.acquire(accempty, acc, uses)
                        copy_x = Uint32(0)
                        if xmask >> ks & 1 == 0:
                            self.sync(xready.iterator + (0 if self.xone else ks), 0, True)
                            xmask |= Uint32(1) << ks
                            copy_x = Uint32(1)
                        done = Uint32(0)
                        if last:
                            done = shared_address(accfull.iterator + acc)
                        release = Uint32(0)
                        if s == c - 1:
                            release = released
                        group8(
                            tmem + acc * 8, a0 + slot * (SLOT // 16) + s * 1024, x0 + ks * 128,
                            (wcol + s * 8, asf0 + slot * (SLOT_SF // 16) + s * 64),
                            (tmem + 256 + ks * 8, xsf0 + ks * 16, copy_x),
                            Uint32(not first), done, release, Uint32(0), part="fc1", nomma=nomma,
                        )
                        cute.arch.sync_warp()
                        if last:
                            self.signal(accmeta, accfull, acc, (Int32(0), a, ks, Int32(2)))
                            uses += 1
                            acc = uses % 16
            else:  # FC2 pair op: unit a, tiles b and b + 1
                copy_h, pre = (Uint32(0), Uint32(0))
                if a != current:
                    if current >= 0:
                        pre = shared_address(hempty.iterator + hbuffer)
                    current = a
                    ordinal += 1
                    hbuffer = ordinal % H
                    self.sync(hfull.iterator + hbuffer, ordinal // H & 1, True)
                    copy_h = Uint32(1)
                hd = h0 + hbuffer * (self.h_image_bytes // 16)
                hs = hsf0 + hbuffer * (self.h_scale_bytes // 16)
                sfbc = tmem + 224 + hbuffer * 12
                quad = False
                if cutlass.const_expr(self.fc2_quad):  # C4b: meta c == 4 marks a quad op (fc2_quad_op); pair ops carry 0
                    quad = c == 4
                if quad:  # C4: four tiles b .. b + 3 from one slot (boxes at 0 / 32 KB)
                    for t in cutlass.range_constexpr(4):
                        target = (uses + t) % 16
                        self.acquire(accempty, target, uses + t)
                        group8(
                            tmem + target * 8, a0 + slot * (SLOT // 16) + (t // 2) * (SLOT // 32) + (t % 2) * 512, hd,
                            (wcol + self.fc2_scale_cols * t, asf0 + slot * (SLOT_SF // 16) + t * self.fc2_scale_units),
                            (sfbc, hs, copy_h if t == 0 else Uint32(0)),
                            Uint32(0), shared_address(accfull.iterator + target), released if t == 3 else Uint32(0),
                            pre if t == 0 else Uint32(0), part="fc2t16", nomma=nomma,
                        )
                        cute.arch.sync_warp()
                    for t in cutlass.range_constexpr(4):
                        self.signal(accmeta, accfull, acc, (Int32(1), a, b + t, Int32(0)))
                        uses += 1
                        acc = uses % 16
                else:
                    self.acquire(accempty, acc, uses)
                    self.acquire(accempty, (uses + 1) % 16, uses + 1)
                    for t in cutlass.range_constexpr(2):
                        target = acc if t == 0 else (uses + 1) % 16
                        group8(
                            tmem + target * 8, a0 + slot * (SLOT // 16) + t * 512, hd,
                            (wcol + self.fc2_scale_cols * t, asf0 + slot * (SLOT_SF // 16) + t * self.fc2_scale_units),
                            (sfbc, hs, copy_h if t == 0 else Uint32(0)),
                            Uint32(0), shared_address(accfull.iterator + target), released if t == 1 else Uint32(0),
                            pre if t == 0 else Uint32(0), part="fc2t16", nomma=nomma,
                        )
                        cute.arch.sync_warp()
                    for t in cutlass.range_constexpr(2):
                        self.signal(accmeta, accfull, acc, (Int32(1), a, b + t, Int32(0)))
                        uses += 1
                        acc = uses % 16
            if cutlass.const_expr(self.meta_bar):  # A (R4): lane 0's release completes meta[slot]'s barrier alone
                named_barrier(12, 32)  # the 32 lanes' metadata reads happen before lane 0's release (PTX 8.9.4)
                if lane == 0:
                    metabar = view(sm.iterator, self.off_misc + self.bar_meta, Uint64, slots)
                    cute.arch.mbarrier_arrive(metabar.iterator + slot)
            count += 1


@asm(Uint32)
def ld_relaxed_cta_shared_u32(pointer):
    """pd form: the prepoll hint read (morally strong with ``st_relaxed_cta_shared_u32``: same scope, one CTA)."""
    return "ld.relaxed.cta.shared.u32 $0, [$1];", "=r,r,~{memory}", (pointer,)


@asm()
def st_relaxed_cta_shared_u32(pointer, value):
    """pd form: the prepoll hint write."""
    return "st.relaxed.cta.shared.u32 [$0], $1;", "r,r,~{memory}", (pointer, Uint32(value))


@asm(Int64)
def ld_generic_u64(pointer):
    """gc_diag "warm": one generic 8-byte load (a ``cvta.param`` kernel-parameter address or a global address)."""
    return "ld.u64 $0, [$1];", "=l,l", (pointer,)


class GcPointers:
    """tm_mode "gc" (benchmark-only): the five tensormaps' generic addresses (``cvta.param`` of the
    ``__grid_constant__`` kernel parameters), indexed by their offsets in the global map buffer (0 W13 8-chunk box, 128 W13
    4-chunk box, 256 W2 pairs, 768 x plan, 896 x scales), so the engine's ``maps.iterator + offset`` sites are unchanged."""

    OFFSETS = (0, 128, 256, 768, 896)

    def __init__(self, addresses):
        self.addresses = dict(zip(self.OFFSETS, addresses))

    @property
    def iterator(self):
        return self

    def __add__(self, offset):
        if isinstance(offset, int):
            return self.addresses[offset]
        # fc1_ops' dynamic ``128 * tail`` (tail 0 / 1): map 0 or map 128, chosen arithmetically on the two addresses
        first, second = self.addresses[0], self.addresses[128]
        return first + (Int64(offset) >> 7) * (second - first)


def gc_weight_specs(w13, w2, pointers=True):
    """``experts_w4a4.maps4``'s three weight-map specs (pointer, dims, byte strides, box, swizzle, L2 promotion;
    the CUtensorMap enum values) as hashable ints, for tm_mode "gc"'s host-side encode. Same arguments as maps4, so the
    descriptors are byte-equal to the global buffer's. ``pointers=False``
    writes 0 in the pointer fields (``gc_rt``: the addresses come at run time, see ``gc_weight_pointers``)."""
    e, rows13 = w13.shape[0], w13.shape[1]
    w2_row, w2_chunks, w2_stride = (192, 3, 64) if rows13 == 768 else (128, 2, 32)
    p13, p2 = (w13.data_ptr(), w2.data_ptr()) if pointers else (0, 0)
    return ((p13, (64, e * rows13, 28), (1792, 64), (64, 128, 8), 2, 3),
            (p13, (64, e * rows13, 28), (1792, 64), (64, 128, 4), 2, 3),
            (p2, (64, e * 3584, w2_chunks), (w2_row, w2_stride), (64, 256, w2_chunks), 2, 0))


def gc_weight_pointers(w13, w2):
    """``gc_rt``: the two runtime arguments (W13, W2 base addresses) the gc_rt host function takes after the scalars."""
    return Int64(w13.data_ptr()), Int64(w2.data_ptr())


class W4A8GcExpertsKernel(W4A8ExpertsKernel):
    """tm_mode "gc" (benchmark-only, default off): ``W4A8ExpertsKernel`` with its five TMA descriptors passed as
    ``__grid_constant__`` kernel parameters. The host function encodes them at each launch (``create_tensor_map_tiled``:
    cuTensorMapEncodeTiled with maps8's arguments; a CUDA graph keeps the encoded parameter bytes), the kernel takes their
    ``cvta.param`` addresses, and its init only prefetches them. Sync: a grid-constant tensormap is a kernel parameter, not
    memory written through the generic proxy, so the guide's acquire.sys per block (needed for the host-copied global
    buffer) does not apply (CUDA Programming Guide 4.12.2.2.3)."""

    @cute.jit
    def __call__(self, *args):
        """``ExpertsKernel.__call__``'s launch (grid, 384 threads, cluster 1, cooperative, PDL) plus the five maps."""
        x, batch = args[0], args[7]
        base = x.iterator.toint()
        weights = self.gc_specs
        kargs = args[:-1]
        if cutlass.const_expr(self.gc_rt):  # (..., scalars, w13 base, w2 base, stream)
            p13, p2 = args[-3], args[-2]
            weights = [(p13 if i < 2 else p2, *spec[1:]) for i, spec in enumerate(self.gc_specs)]
            kargs = args[:-3]
        specs = [*weights,
                 (base, (128, batch, 28), (3584, 128), (128, 8, 28), 3, 0),  # experts_stream.plan_maps(atoms=28) as maps8
                 (base + PLN.PLAN_XSF, (16, batch, 28), (448, 16), (16, 8, 28), 0, 0)]
        tms = []
        for i, (pointer, dims, strides, box, swizzle, l2) in enumerate(specs):
            assert all(s % 16 == 0 for s in strides), strides
            tms.append(TM.create_tensor_map_tiled(pointer, cutlass.Uint8, dims, [s // 16 for s in strides], box,
                                                  swizzle=TM.TensorMapSwizzle(swizzle),
                                                  l2_promotion=TM.TensorMapL2Promotion(l2)))
        self.kernel(kargs, *tms).launch(grid=(self.grid, 1, 1), block=(384, 1, 1), stream=args[-1], use_pdl=True,
                                            cluster=(1, 1, 1), cooperative=True)

    @cute.kernel
    def kernel(self, args, m0: cutlass.GridConstant[TM.TensorMap], m1: cutlass.GridConstant[TM.TensorMap],
               m2: cutlass.GridConstant[TM.TensorMap], m3: cutlass.GridConstant[TM.TensorMap],
               m4: cutlass.GridConstant[TM.TensorMap]):
        x, ids, maps, w13_scales, w2_scales, workspace, *rest = args
        pointers = GcPointers(tuple(m.get_ptr().toint() for m in (m0, m1, m2, m3, m4)))
        if cutlass.const_expr(self.gc_diag == "warm"):
            for i in cutlass.range_constexpr(5):
                if thread()[0] == 128 + 32 * i:
                    ld_generic_u64(pointers + GcPointers.OFFSETS[i])
        self.kernel_body((x, ids, maps if self.gc_diag == "pf" else pointers, w13_scales, w2_scales, workspace, *rest))


def maps8(w13, w2, plan, batch, device):
    """W4A4's packed-U8 weight maps (W13 8- / 4-chunk boxes, W2 tile pairs) + the MXFP8 plan's x (28 atoms, 128B
    swizzle) and scale words, in the W4A4 map buffer layout (0, 128, 256, 768, 896)."""
    full = W4.maps4(w13, w2, plan, batch, device, 2)
    x = MV.plan_maps(plan, batch, device, atoms=28)
    full[768:1024] = x[:256]
    return full


@functools.cache
def compile_w4a8(tp, experts, batch, grid, device, activation_floor=False, knobs=(), box_rows=64):
    capacity = 1 << (batch - 1).bit_length()
    kind = W4A8GcExpertsKernel if dict(knobs).get("tm_mode") == "gc" else W4A8ExpertsKernel  # default: unchanged
    publish = bool(dict(knobs).get("pub_form"))  # published B8 (routing weights replace out; box after the scalars)
    moe = kind(tp, capacity, experts, grid, publish, activation_floor, **dict(knobs))
    moe.box_rows = box_rows  # the tail box's rows (``ExpertsKernel.box_rows``)
    flat = lambda n, dtype=cutlass.Uint8, align=16: fake((n,), (1,), dtype, align)  # noqa: E731
    ids = flat(batch * 16, Int32)
    scales = (flat(experts * n) for n in (moe.w13_scale_bytes, moe.w2_scale_bytes))
    out = flat(batch * 16 * (1 if publish else 3584), cutlass.BFloat16)
    tensors = (flat(PLN.PLAN_BYTES), ids, flat(MAPS, align=64), *scales, flat(moe.workspace_bytes), out)
    scalars = (Int32(batch), *map(cutlass.Float32, (4, 25, 0)))
    if publish:  # (latent multicast, parity, rank), as compile_experts' publish box
        scalars += (Int64(0), Int32(0), Int32(0))
    if moe.tm_mode == "gc" and moe.gc_rt:  # the two runtime weight base addresses (compile-time placeholders)
        scalars += (Int64(0), Int64(0))
    with torch.accelerator.device_index(device):
        code = cute.compile(moe, *tensors, *scalars, make_fake_stream())
        if moe.smem_bytes > 227 * 1024:
            PLN.set_oversized(code, device)
        return code, moe.smem_bytes


class W4A8WTExperts(W4.W4A4Experts):
    """TP16 non-publish batches <= 8 on the W4A8 whole-tile engine; everything else as ``PlannedExperts`` /
    ``PersistentExperts``. Same hidden image layout as the shipped engine, so a per-batch subset is safe."""

    def __init__(self, *args, xone=True, **kwargs):
        super().__init__(*args, xone=xone, **kwargs)
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        grid = os.environ.get("K31MK_EXPERTS_W4A8WT_GRID")
        free = W4A8WT_FREE.get(torch.cuda.get_device_capability(self.device))
        self.w4_grid = (int(grid) if grid else None if free is None else sms - free) if self.tp == 16 else None
        if (self.tp == 16) and torch.cuda.get_device_capability(self.device) != (10, 7):
            raise ValueError("W4A8 whole-tile experts need SM107 (packed-E2M1 K64 MMAs)")

    def prepare(self, batch, buffers, out, publish=(), planned=False, twoshot=False, early=False):
        """``planned``: the caller writes the MXFP8 plan (``planned.compile_early_plan``) before ``run``."""
        if self.tp != 16 or publish or twoshot or batch > 8:
            return PLN.PlannedExperts.prepare(self, batch, buffers, out, publish, planned, twoshot, early)
        need = PLN.planned_workspace_bytes(ER.ExpertsGeometry(self.tp).workspace_bytes)
        if buffers["workspace"].numel() < need:
            raise ValueError("W4A8 whole-tile experts need planned.scratch() workspace")
        grid = self.grid_for(batch)
        knobs = tuple((k, v) for k, v in self.knobs if k != "sv")
        code, smem = compile_w4a8(self.tp, self.experts, batch, grid, self.device.index, self.activation_floor, knobs)
        if grid > torch.cuda.get_device_properties(self.device).multi_processor_count:
            raise RuntimeError(f"W4A8 whole-tile experts: grid {grid} > SMs")
        w13, _, w2, _ = self.weights
        maps = maps8(w13, w2, buffers["plan"], batch, self.device)
        self.batch_maps[batch] = maps
        plan = (ER.tensor(buffers["plan"]), PLN.compile_early_plan(batch, self.device.index))
        if planned:
            plan = plan[:1]
        ws = ER.tensor(buffers["workspace"])
        self.prepared[batch] = (out, ER.tensor(out), ws, code, (), plan, ER.to_cute(maps, 64), None)
