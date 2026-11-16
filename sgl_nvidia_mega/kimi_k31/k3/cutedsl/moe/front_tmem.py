# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Our changes (K31MK_MOE_FRONT_TMEM): (1) the latent columns leave as FP32 words like every other publisher
# of our front mailbox (``Fp32FrontGemm`` / ``SharedUmma.publish``: latent at 2 * (base + column), router logits at
# 7168 + 2 * (slot + row)), not the source's BF16 latent; (2) ``fenced_trigger`` is SplitKTmemGemm64's (2 = our served fix A
# mode: warp 3 waits, fences and releases the dependents); (3) ``MoeFrontRows`` (the > 64-token form) is not ported;
# (4) (K31MK_MOE_FRONT_N32, default off, read here: ``front_n32_enabled`` / ``front_n32_for``): 32-token N tiles
# at <= 64 rows (half the MMA work per tile, every K tile in tensor memory, half the split partials; two tiles at 33-64 rows);
# same K partition, MMA order and rank-order sum as N64; (5) 65-112 rows (batch-table rows 65-128): two
# N-exact tiles of N = rows / 2 rounded up to 8 (N40 / N48 / N56) instead of two N64 tiles; 113-128 rows stay N64; (6)
# (K31MK_MOE_FRONT_LAMX, default off): ``MoeFrontTmem64Lamx`` polls x from the o-proj consumer's Lamport box.
# (7)  (K31MK_MOE_FRONT_L2PF / _LEAN / _TX576, default off): an L2 prefetch of the A tiles past the weight
# ring's first round; a 116 KB shared-memory floor; one exclusive 576-column tensor-memory allocation (every K tile in TMEM).
# (8)  (K31MK_MOE_FRONT_XSEP, default off): x does not wait for the whole weight drain (``front_xsep``);
# (9)  (K31MK_MOE_FRONT_RFIRST, default off): the first clusters take every N tile's front tiles (``front_rfirst``).
"""MoE front at up to 64 tokens per tile: the front GEMM rows and the shared expert's
gate/up in one ``SplitKTmemGemm64`` grid (``K31MK_MOE_FRONT_TMEM``, TP16 verify 16-32 rows).

The merged gate kernel (``SharedUmma`` with the front rows) streams x after the PDL wait
through a ring that still holds weights. Here each 128-row tile's K slice is in tensor
memory before x lands (clusters of 8, one CTA per SM, 14 K tiles each): tiles 0 .. 2 are
this rank's packed front rows [router (``valid``, padded to ``split``); latent-down
shard], tiles 3 .. are 64 gate rows and the matching 64 up rows each. Front owners
publish like ``Fp32FrontGemm.epilogue`` (FP32 logits and FP32 latent columns, -0 made
+0, multicast into the front mailbox); gate/up owners take 8 gate rows and their 8 up
rows, round both to BF16, apply SiTU like the merged kernel and store h for
``SharedDown``. The 8 split partials are summed in rank order (an FP32 reorder of the
merged / split kernels' sums: not bitwise to them)."""

import os
import sys

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64

from ..common.policy import batch_table
from ..common.primitives import asm, bits_float, multicast_words, named_barrier
from ..gemm.splitk import SplitKTactic
from ..gemm.splitk import _acquire_tensor_map
from ..gemm.splitk_tmem64 import N_TILES, SplitKTmemGemm64, _ld_shared_v4, _tma_box, _tma_prefetch_box
from ..gemm.ws_gemm import positive_zero32
from .shared import situ

__all__ = [
    "FRONT_TILES", "LOGIT_BASE", "NX_TILES", "SMALL_TILES", "MoeFrontTmem64", "MoeFrontTmem64Lamx", "MoeFrontTmem64LamxC", "MoeFrontTmem64LamxD", "front_lamx_for",
    "front_l2pf", "front_lean", "front_n32_enabled", "front_n32_for", "front_rfirst", "front_tx576", "front_xsep",
]  # fmt: skip

FRONT_TILES = 3  # 288 packed front rows in 128-row tiles
LOGIT_BASE = 7168  # BF16 elements before a box row's FP32 logit words (3584 FP32 latent words first)


# N32 (K31MK_MOE_FRONT_N32; default unset / 0 = served N64): 1 = MoeFrontTmem64 runs a 32-token N tile instead
# of 64 at every K31MK_MOE_FRONT_TMEM batch (moe/front.py compiles ONE front per process for all FTM batches, so the knob is
# all-or-nothing and needs every FTM batch <= 32 rows). Half the tensor-core work per K tile (the M128 N64 chain costs the same at
# every row count, measured), all 14 K tiles of a split in tensor memory (two 32-column accumulators under
# PROMOTE 2), half the DSMEM partial pushes and owner sums. Same K partition, per-element MMA order and rank-order FP32 sum as
# N64 (bitwise expected; the GPU rig decides). 2 = the same N32 tile with the MMAs' N field set at run time to the call's rows
# rounded up to 8 (16 / 24 / 32: a 16-row call does half the N32 MMA work; ``runtime_n``, register instruction descriptor).
# One stderr line 'K31MK_MOE_FRONT_N32 engaged mode=<1|2> ftm=<batches> pid=<p>' per process.
# 33-64 rows run TWO N32 tiles (2 x 72 CTAs, each with the c=8 tile's MMA / reduce / finish) instead of
# one N64 tile, so the scalar form accepts FTM batches up to 64 rows. A boot whose FTM table also lists 65-128 rows
# needs N32 per batch: the batch-table form (e.g. "16-64:1", rows 9-64, value 0 / 1 / 2) is read by ``front_n32_for(batch)``,
# which only a caller that compiles one front per batch can use (moe/front.py); one stderr line
# 'K31MK_MOE_FRONT_N32 engaged mode=<m> batch=<b> pid=<p>' per engaged batch per process.
# 65-128 rows (table form only): N32 tiles there would need 3-4 x 72 CTAs (two waves), so value 1 at a
# 65-128-row batch means TWO N-exact tiles of N = ceil(rows / 2) rounded up to 8 (N40 at 65-80 rows, N48 at 81-96, N56 at 97-112;
# 144 CTAs like the served two N64 tiles, MMA work and partials in proportion to N); 113-128 rows give N64 = the served form (0).
# ``front_n32_for`` returns that N (40 / 48 / 56) for MoeFrontTmem64's ``n32``; one stderr line
# 'K31MK_MOE_FRONT_N32 engaged mode=1 batch=<b> ntile=<N> pid=<p>' per engaged batch per process.
# Value 3 (table form): ONE N-exact tile of N = rows rounded up to 8 at 9-64 rows (N16 at 9-16, N24 at 17-24,
# N32 = mode 1 at 25-32, N40 / N48 / N56 at 33-56, N64 = served at 57-64): the MMAs are issue-bound at N <= 32 (the 56-MMA chain does
# not shrink), but the DSMEM partials, owner sums and finish scale with N; at 65-128 rows 3 = 1 (two N-exact tiles). Line
# 'K31MK_MOE_FRONT_N32 engaged mode=3 batch=<b> ntile=<N> pid=<p>'.
NX_TILES = (40, 48, 56)
SMALL_TILES = (16, 24)
_n32_engaged: list = []
_n32_batches: set = set()
_l2pf_engaged: list = []
_flag_engaged: set = set()


# L2PF (K31MK_MOE_FRONT_L2PF; default unset / 0 = off): 1 = before its first ring load, the weight warp
# prefetches to L2 the A tiles past the ring's first round (``SplitKTmemGemm64.l2pf``), at every FTM batch (scalar, all-or-nothing:
# the served front compile is keyed on the whole K31MK_* env). A cache hint: same loads, data and order (bitwise by construction).
# One stderr line 'K31MK_MOE_FRONT_L2PF engaged pid=<p>' per process.
def front_l2pf():
    """K31MK_MOE_FRONT_L2PF: 0 (unset / empty / 0) or 1; anything else raises."""
    value = os.environ.get("K31MK_MOE_FRONT_L2PF", "") or "0"
    if value not in ("0", "1"):
        raise ValueError(f"K31MK_MOE_FRONT_L2PF={value!r}: 0 (off) or 1 (L2 prefetch of the A tiles past the ring's first round)")
    if value == "1" and not _l2pf_engaged:
        print(f"K31MK_MOE_FRONT_L2PF engaged pid={os.getpid()}", file=sys.stderr, flush=True)
        _l2pf_engaged.append(1)
    return int(value)


# LEAN / TX576 (scalar, default unset / 0 = off; one stderr line 'K31MK_MOE_FRONT_<NAME> engaged pid=<p>'):
# K31MK_MOE_FRONT_LEAN=1: the per-CTA shared-memory floor is 116 KB instead of 170 KB (two front CTAs still cannot share a 228 KB SM),
#   so a front CTA fits beside up to six (not three) of its PDL predecessor's 15.7 KB o-proj consumer CTAs. Launch size only: bitwise.
# K31MK_MOE_FRONT_TX576=1 (sm_107 only): one exclusive 576-column tensor-memory allocation, so all 14 K tiles of a split sit in
#   tensor memory up to N64 under PROMOTE 2 (no shared-memory A slots, A-from-TMEM MMAs only; same K order and per-element MMA
#   math: bitwise expected, the GPU rig decides). No other CTA can allocate tensor memory on that SM until the front's dealloc.
def _front_flag(name, what):
    value = os.environ.get(name, "") or "0"
    if value not in ("0", "1"):
        raise ValueError(f"{name}={value!r}: 0 (off) or 1 ({what})")
    if value == "1" and name not in _flag_engaged:
        print(f"{name} engaged pid={os.getpid()}", file=sys.stderr, flush=True)
        _flag_engaged.add(name)
    return int(value)


def front_lean():
    """K31MK_MOE_FRONT_LEAN: 0 or 1; anything else raises."""
    return _front_flag("K31MK_MOE_FRONT_LEAN", "116 KB shared-memory floor")


def front_tx576():
    """K31MK_MOE_FRONT_TX576: 0 or 1; anything else raises."""
    return _front_flag("K31MK_MOE_FRONT_TX576", "exclusive 576-column tensor memory")


# XSEP (scalar, default unset / 0 = off; one stderr line 'K31MK_MOE_FRONT_XSEP engaged value=<v> pid=<p>'):
# the MID front's last clusters enter after the o-proj consumer has ended and then drained 12 weight tiles
# before x loaded and the 56 MMAs ran. 1 = x tile k loads into the ring slot that frees earliest once its stages had
# their final tensor-memory copy, warp 3 issues the MMAs as A and x tiles become ready (SplitKTmemGemm64 ``xsep``); 2 = 1 plus
# two x slots of their own for the last two K tiles (+2 x tiles of shared memory). Same MMAs and sums: bitwise expected.
_xsep_engaged: list = []


# RFIRST (scalar, default unset / 0 = off; one stderr line 'K31MK_MOE_FRONT_RFIRST engaged pid=<p>'): behind the MID
# o-proj consumer only 64 front CTAs (8 clusters, SM ids 0-63) enter before it ends; the GPU launches clusters in grid order (tile fastest:
# measured), so at 65-128 rows the router / latent tiles of the second N tile entered AFTER the consumer
# ended and their publish (what RouteSitu polls) came last. 1 = cluster c (grid order) computes front tile c % 3 of N tile c // 3 for
# c < 3 x N tiles, then the gate / up tiles (3 + c' % 6 of N tile c' // 6, c' = c - 3 x N tiles): the same tiles, each once, in another
# launch order (bitwise by construction; the GPU rig decides). Identity at one N tile (<= 64 rows). h (gate / up) then lands last; the
# shared down rows wait for the whole front through moe/prepared.py's begin event, not for RouteSitu.
def front_rfirst():
    """K31MK_MOE_FRONT_RFIRST: 0 or 1; anything else raises."""
    return _front_flag("K31MK_MOE_FRONT_RFIRST", "front tiles of every N tile first")


def front_xsep():
    """K31MK_MOE_FRONT_XSEP: 0 (unset / empty / 0), 1 or 2; anything else raises."""
    value = os.environ.get("K31MK_MOE_FRONT_XSEP", "") or "0"
    if value not in ("0", "1", "2"):
        raise ValueError(f"K31MK_MOE_FRONT_XSEP={value!r}: 0 (off), 1 (x per ring stage) or 2 (1 + two x slots of their own)")
    if value != "0" and not _xsep_engaged:
        print(f"K31MK_MOE_FRONT_XSEP engaged value={value} pid={os.getpid()}", file=sys.stderr, flush=True)
        _xsep_engaged.append(int(value))
    return int(value)


def front_n32_enabled():
    """K31MK_MOE_FRONT_N32 scalar form: 0 (unset / 0: N64), 1 (N32 tile) or 2 (N32 tile, run-time MMA N) at every
    K31MK_MOE_FRONT_TMEM batch; 1 and 2 need every FTM batch at most 64 rows (one N32 tile up to 32 rows, two at 33-64).
    The batch-table form is per batch (``front_n32_for``) and raises here."""
    value = os.environ.get("K31MK_MOE_FRONT_N32", "") or "0"
    if ":" in value:
        raise ValueError(f"K31MK_MOE_FRONT_N32={value!r}: the batch-table form needs a per-batch front compile (front_n32_for(batch), "
                         "moe/front.py); the served one-front compile takes 0, 1 or 2")
    if value not in ("0", "1", "2"):
        raise ValueError(f"K31MK_MOE_FRONT_N32={value!r}: 0 (off), 1 (N32 tile) or 2 (N32 tile, run-time MMA N)")
    if value == "0":
        return 0
    ftm = sorted(b for b, v in batch_table(os.environ.get("K31MK_MOE_FRONT_TMEM", "")).items() if v == 1)
    if not ftm or ftm[-1] > 64:
        raise ValueError(f"K31MK_MOE_FRONT_N32={value}: needs K31MK_MOE_FRONT_TMEM batches, all <= 64 rows (two N32 tiles), got {ftm}")
    if not _n32_engaged:
        print(f"K31MK_MOE_FRONT_N32 engaged mode={value} ftm={ftm[0]}-{ftm[-1]} pid={os.getpid()}", file=sys.stderr, flush=True)
        _n32_engaged.append((int(value), tuple(ftm)))
    return int(value)


def front_n32_for(batch):
    """K31MK_MOE_FRONT_N32 batch-table form for a caller that compiles one front per batch: this batch's mode (0 / 1 / 2;
    0 for a batch the table does not list), or None for the scalar form / unset (``MoeFrontTmem64`` then reads the
    scalar). Table rows 9-128: 9-64 take 0 / 1 / 2 (N32 tiles; above 64 rows they would need a second wave of CTAs);
    65-128 take 0 / 1, and 1 returns the N-exact tile N (``NX_TILES``: two tiles of ceil(rows / 2) rounded up to 8), or 0 where
    that is 64 (113-128 rows: the served N64 form). Value 3: one N-exact tile of rows rounded up to 8 at 9-64 rows (returns 16 / 24 /
    40 / 48 / 56, 1 for N32, 0 for N64); at 65-128 rows the same as 1."""
    value = os.environ.get("K31MK_MOE_FRONT_N32", "")
    if ":" not in value:
        return None
    table = batch_table(value)
    bad = {b: v for b, v in table.items() if not 9 <= b <= 128 or v not in ((0, 1, 2, 3) if b <= 64 else (0, 1, 3))}
    if bad:
        raise ValueError(f"K31MK_MOE_FRONT_N32={value!r}: table rows 9-64 with value 0, 1, 2 or 3, rows 65-128 with 0, 1 or 3, got {bad}")
    mode = int(table.get(int(batch), 0))
    if mode == 3 and int(batch) <= 64:  # one N-exact tile (value 3)
        ntile = (int(batch) + 7) // 8 * 8
        if ntile in (32, 64):  # N32 = mode 1's tile (same compile key); N64 = the served tile
            ntile_mode = 1 if ntile == 32 else 0
        else:
            assert ntile in SMALL_TILES + NX_TILES, (batch, ntile)
            ntile_mode = ntile
        if ntile_mode and int(batch) not in _n32_batches:
            print(f"K31MK_MOE_FRONT_N32 engaged mode=3 batch={int(batch)} ntile={ntile} pid={os.getpid()}", file=sys.stderr, flush=True)
            _n32_batches.add(int(batch))
        return ntile_mode
    if mode and int(batch) > 64:  # 65-128 rows: two N-exact tiles
        ntile = (-(-int(batch) // 2) + 7) // 8 * 8
        if ntile >= 64:
            return 0
        assert ntile in NX_TILES, (batch, ntile)
        if int(batch) not in _n32_batches:
            print(f"K31MK_MOE_FRONT_N32 engaged mode={mode} batch={int(batch)} ntile={ntile} pid={os.getpid()}", file=sys.stderr, flush=True)
            _n32_batches.add(int(batch))
        return ntile
    if mode and int(batch) not in _n32_batches:
        print(f"K31MK_MOE_FRONT_N32 engaged mode={mode} batch={int(batch)} pid={os.getpid()}", file=sys.stderr, flush=True)
        _n32_batches.add(int(batch))
    return mode


@asm()
def _st_shared_f32(address, value):
    return "st.shared.f32 [$0], $1;", "r,f,~{memory}", (Int32(address), Float32(value))


@asm()
def _tma_half(dst, barrier, tensor_map, k0, row0):
    """One {64, 64} box at (k0, row0) to shared address ``dst`` (half an A tile)."""
    code = "{ .reg .b64 pol; createpolicy.fractional.L2::evict_first.b64 pol, 1.0; "
    code += "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx"
    code += "::bytes.L2::cache_hint [$0], [$1, {$2, $3}], [$4], pol; }"
    operands = (Int32(dst), tensor_map, Int32(k0), Int32(row0), barrier)
    return code, "r,l,r,r,r,~{memory}", operands


class MoeFrontTmem64(SplitKTmemGemm64):
    """Front rows (``valid`` router rows padded to ``split_rows``, then ``shard``
    latent rows) and ``hidden`` gate/up pairs at up to 64 tokens per N tile; split K 8."""

    n_allowed = N_TILES + (32,) + NX_TILES + SMALL_TILES  # N16-N56 (``finish`` below handles 16-64-token tiles)
    any_width = True  # N16 / N24 / N40 / N48 / N56: 2 / 3 / 5 / 6 / 7 tokens per owner thread (``finish`` takes N / 8)
    extra_a_maps = True  # gate / up tiles' ``issue_a`` reads ``gate_map`` (``acquire_a_maps``)

    def __init__(
        self, *, valid, split_rows, shard, hidden, fenced_trigger=2, promote=0, mbox=0, n32=None, l2pf=None, lean=None, tx576=None,
        xsep=None, rfirst=None,
    ):  # fmt: skip
        n32 = front_n32_enabled() if n32 is None else int(n32)  # None: the K31MK_MOE_FRONT_N32 knob (served path)
        assert n32 in (0, 1, 2) + NX_TILES + SMALL_TILES, n32  # 16-56: an N-exact tile (``front_n32_for`` values 1 above 64 rows, 3)
        self.runtime_n = n32 == 2  # read at trace time by SplitKTmemGemm64.body / staged_mma_warp64
        l2pf = front_l2pf() if l2pf is None else int(l2pf)  # None: the K31MK_MOE_FRONT_L2PF knob (served path)
        lean = front_lean() if lean is None else int(lean)  # None: K31MK_MOE_FRONT_LEAN
        tx576 = front_tx576() if tx576 is None else int(tx576)  # None: K31MK_MOE_FRONT_TX576
        xsep = front_xsep() if xsep is None else int(xsep)  # None: K31MK_MOE_FRONT_XSEP
        self.rfirst = front_rfirst() if rfirst is None else int(rfirst)  # None: K31MK_MOE_FRONT_RFIRST (read at trace time by body)
        assert self.rfirst in (0, 1), self.rfirst
        super().__init__(
            tactic=SplitKTactic(128, {0: 64, 1: 32, 2: 32}.get(n32, n32), 8, 14), k=7168, owners=8,
            fenced_trigger=fenced_trigger, promote=promote, mbox=mbox, l2pf=l2pf, lean=lean, tx576=tx576, xsep=xsep,
        )  # fmt: skip
        assert split_rows % 8 == 0 and valid <= split_rows and (split_rows + shard) <= FRONT_TILES * 128
        assert hidden % 64 == 0
        self.valid, self.split_rows, self.shard, self.hidden = (
            valid, split_rows, shard, hidden,
        )  # fmt: skip
        self.gate_tiles = hidden // 64
        self.tiles = FRONT_TILES + self.gate_tiles

    @cute.experimental.jit
    def __call__(
        self, packed: cute.Tensor, x: cute.Tensor, h: cute.Tensor, box: cute.Tensor,
        mc: cutlass.Int64, slot: cutlass.Int32, base: cutlass.Int32,
        beta: cutlass.Float32, linear_beta: cutlass.Float32, front_map: cutlass.Int64,
        gate_map: cutlass.Int64, stream: _cuda.CUstream,
    ):  # fmt: skip
        self.kernel(
            packed, x, h, box, mc, slot, base, beta, linear_beta, front_map, gate_map
        ).launch(
            grid=(self.tiles, cute.ceil_div(x.layout.shape[0], self.n) * self.split_k, 1),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(self.smem_bytes),
            stream=stream,
            use_pdl=True,
        )

    @cute.experimental.kernel
    def kernel(
        self, mA: cute.Tensor, mB: cute.Tensor, h: cute.Tensor, box: cute.Tensor,
        mc: cutlass.Int64, slot: cutlass.Int32, base: cutlass.Int32,
        beta: cutlass.Float32, linear_beta: cutlass.Float32, front_map: cutlass.Int64,
        gate_map: cutlass.Int64,
    ):  # fmt: skip
        tokens = mB.layout.shape[0]
        self.gate_map = gate_map
        out = (box, mc, slot, base, tokens, h, beta, linear_beta)
        self.body(mA, mB, box, front_map, out)

    @cute.experimental.jit
    def cluster_tile(self, bidx: cutlass.Int32, n_idx: cutlass.Int32):
        """RFIRST: cluster c = bidx + tiles x n_idx (the GPU's launch order) -> front tile c % 3 of N tile c // 3 while c < 3 x N tiles,
        then gate / up tile 3 + c' % gate_tiles of N tile c' // gate_tiles (c' = c - 3 x N tiles)."""
        _, grid_y, _ = cute.arch.grid_dim()
        n_tiles = Int32(grid_y) // Int32(self.split_k)
        c = Int32(bidx) + Int32(self.tiles) * Int32(n_idx)
        n_front = Int32(FRONT_TILES) * n_tiles
        f = Int32(c < n_front)
        g = c - n_front + (n_front - c) * f  # 0 for front clusters (keeps the division operands non-negative)
        tile = f * (c % FRONT_TILES) + (1 - f) * (FRONT_TILES + g % self.gate_tiles)
        n_new = f * (c // FRONT_TILES) + (1 - f) * (g // self.gate_tiles)
        return tile, n_new

    @cute.experimental.jit
    def acquire_a_maps(self, a_map, row0):
        """``issue_a`` / ``prefetch_a`` read one tensor map per CTA, chosen by ``row0`` with the test below (front tiles
        ``a_map``; gate / up tiles ``gate_map``, also a tensor map in global memory written by the host, PTX tensormap
        proxy): acquire that map before any TMA reads through it. One acquire per CTA, as ``a_map`` alone took before."""
        if row0 // 128 < FRONT_TILES:
            _acquire_tensor_map(a_map)
        else:
            _acquire_tensor_map(self.gate_map)

    @cute.experimental.jit
    def issue_a(self, dst, barrier, a_map, k0, row0):
        """Front tiles: one {64, 128} box of the packed rows; gate/up tile g: gate
        rows [64 g, + 64) and up rows [hidden + 64 g, + 64) as two {64, 64} boxes."""
        tile = row0 // 128
        if tile < FRONT_TILES:
            _tma_box(dst, barrier, a_map, k0, row0)
        else:
            gate = (tile - FRONT_TILES) * 64
            _tma_half(dst, barrier, self.gate_map, k0, gate)
            _tma_half(dst + 8192, barrier, self.gate_map, k0, self.hidden + gate)

    @cute.experimental.jit
    def prefetch_a(self, a_map, k0, row0):
        """K31MK_MOE_FRONT_L2PF: the boxes ``issue_a`` loads, to L2 only (front tiles one {64, 128} box, gate/up two {64, 64})."""
        tile = row0 // 128
        if tile < FRONT_TILES:
            _tma_prefetch_box(a_map, k0, row0)
        else:
            gate = (tile - FRONT_TILES) * 64
            _tma_prefetch_box(self.gate_map, k0, gate)
            _tma_prefetch_box(self.gate_map, k0, self.hidden + gate)

    def owner_of(self, row, row0):
        """Front tiles: 16-row shares; gate/up tiles: owner o takes gate rows
        [8 o, + 8) and their up rows (share rows 0-7 gate, 8-15 up)."""
        gate = Int32(row0 >= FRONT_TILES * 128)
        owner = gate * (row % 64 // 8) + (1 - gate) * (row // 16)
        share_row = gate * (row // 64 * 8 + row % 8) + (1 - gate) * (row % 16)
        return owner, share_row

    @cute.experimental.jit
    def finish(
        self, epi_tid: cutlass.Int32, totals: cute.Tensor, tile: Int32, out,
        row0: cutlass.Int32, col0: cutlass.Int32, l_idx: cutlass.Int32,
    ):  # fmt: skip
        """An owner's FP32 share [16 rows] x N tokens (thread: row rr, tokens
        w g .. w g + w - 1, w = N / 8; ``totals`` None: already staged by ``stage_quad``), transposed through ``tile`` [N tokens][16];
        front tiles publish one 8-row chunk per thread, gate/up tiles store 4 SiTU
        columns of h per thread (N32: threads 0-63 of the 128)."""
        box, mc, slot, base, tokens, h, beta, linear_beta = out
        width: cutlass.Constexpr = self.n // 8  # tokens per thread (8 at N64, 4 at N32; N16-N56: None, staged by ``stage_quad``)
        rr, g = epi_tid % 16, epi_tid // 16
        if cutlass.const_expr(totals is not None):
            for t in cutlass.range_constexpr(width):
                _st_shared_f32(tile + ((width * g + t) * 16 + rr) * 4, totals[t])
        named_barrier(3, 128)
        m, owner = row0 // 128, row0 % 128 // 16
        token, part = epi_tid // 2, epi_tid % 2
        if cutlass.const_expr(self.n < 64):  # N32: tile tokens 0 .. 31 only
            live = (col0 + token < tokens) & (token < self.n)
        else:
            live = col0 + token < tokens
        if live:
            chunk = tile + token * 64
            if m < FRONT_TILES:
                self.publish(box, mc, slot, base, chunk, part, row0, col0 + token)
            else:
                column = (m - FRONT_TILES) * 64 + 8 * owner + 4 * part
                gates = [bits_float(w) for w in _ld_shared_v4(chunk + 16 * part)]
                ups = [bits_float(w) for w in _ld_shared_v4(chunk + 32 + 16 * part)]
                for c in cutlass.range_constexpr(4):
                    gate, up = Float32(BFloat16(gates[c])), Float32(BFloat16(ups[c]))
                    value = BFloat16(situ(gate, up, beta, linear_beta))
                    h[col0 + token, column + c] = value

    @cute.experimental.jit
    def stage_quad(self, tile: Int32, q, rr, totals: cute.Tensor):
        """``any_width`` (N16-N56): owner row ``rr``'s FP32 sums of tokens 4 q .. 4 q + 3 into ``tile`` [N tokens][16] (``finish``'s layout)."""
        for j in cutlass.range_constexpr(4):
            _st_shared_f32(tile + ((4 * q + j) * 16 + rr) * 4, totals[j])

    @cute.experimental.jit
    def publish(
        self, box: cute.Tensor, mc: cutlass.Int64, slot: cutlass.Int32,
        base: cutlass.Int32, chunk: Int32, part: Int32, row0: cutlass.Int32,
        token: Int32,
    ):  # fmt: skip
        """Front rows ``row0 + 8 part`` .. + 7 of ``token`` (FP32 at shared address
        ``chunk``) into its front-box row: router rows as FP32 logit words, latent rows
        as FP32 words (A4, as ``Fp32FrontGemm``), -0 made +0, multicast."""
        pitch = box.layout.shape[2]
        row = row0 + 8 * part
        values = []
        for v in cutlass.range_constexpr(2):
            values += [bits_float(w) for w in _ld_shared_v4(chunk + 32 * part + 16 * v)]
        if row < self.valid:
            for v in cutlass.range_constexpr(2):
                quad = tuple(positive_zero32(values[4 * v + i]) for i in range(4))
                offset = token * pitch + LOGIT_BASE + 2 * (slot + row + 4 * v)
                multicast_words(mc + Int64(offset) * 2, quad, kind="f32")
        elif (row >= self.split_rows) & (row - self.split_rows < self.shard):
            for v in cutlass.range_constexpr(2):
                quad = tuple(positive_zero32(values[4 * v + i]) for i in range(4))
                offset = token * pitch + 2 * (base + row - self.split_rows + 4 * v)
                multicast_words(mc + Int64(offset) * 2, quad, kind="f32")


# F-L (K31MK_MOE_FRONT_LAMX; default unset = served code): a batch table (rows 9-32, value 0, 1 or 2; 2 = the canary (only with K31MK_MOE_FRONT_LAMX_MEAS=1)
# measurement form; measured faster than served only at 16 rows, so the c=8 table is 16:1) of
# K31MK_MOE_FRONT_TMEM batches whose front polls x from the o-proj consumer's Lamport box (SplitKTmemGemm64 ``lamx``) instead
# of its PDL wait + x TMA: the consumer stores x a second time, as Lamport words (NaN halves canonical, so non-NaN x is
# bitwise x) into box[parity] of a [2][32][7168] BF16 box (parity = the MoE ordinal % 2; an even ordinal count per forward
# keeps the parities aligned across forwards); the front re-arms box[1 - parity] at its entry (c=1 chain form). Rows 9-32 only (the
# box rows, LAMX_ROWS). One stderr line 'K31MK_MOE_FRONT_LAMX engaged batch=<b> pid=<p>' per engaged batch per process.
_lamx_batches: set = set()


def front_lamx_for(batch):
    """K31MK_MOE_FRONT_LAMX: this batch's mode, 0 (off, also unset / unlisted), 1 (F-L), 2 (F-L with the canary spin) or 3 (F-L per-tile form)."""
    value = os.environ.get("K31MK_MOE_FRONT_LAMX", "")
    if not value:
        return False
    table = batch_table(value)
    bad = {b: v for b, v in table.items() if not 9 <= b <= 32 or v not in (0, 1, 2, 3)}
    if bad:
        raise ValueError(f"K31MK_MOE_FRONT_LAMX={value!r}: rows 9-32 with value 0, 1, 2 or 3, got {bad}")
    if 2 in table.values() and os.environ.get("K31MK_MOE_FRONT_LAMX_MEAS", "") != "1":  # a typo cannot enable it
        raise ValueError(f"K31MK_MOE_FRONT_LAMX={value!r}: value 2 (canary measurement form, slower) needs K31MK_MOE_FRONT_LAMX_MEAS=1")
    mode = int(table.get(int(batch), 0))
    if mode and int(batch) not in _lamx_batches:
        print(f"K31MK_MOE_FRONT_LAMX engaged batch={int(batch)} mode={mode} pid={os.getpid()}", file=sys.stderr, flush=True)
        _lamx_batches.add(int(batch))
    return mode


class MoeFrontTmem64Lamx(MoeFrontTmem64):
    """F-L (K31MK_MOE_FRONT_LAMX): ``MoeFrontTmem64`` whose x comes from the consumer's Lamport box (``lamx``); two more
    arguments: ``xbox`` (Int64 byte address of the [2][LAMX_ROWS][7168] BF16 box) and ``parity`` (Int32, 0 or 1). ``x``
    still gives the token count (never read)."""

    lamx = True

    @cute.experimental.jit
    def __call__(
        self, packed: cute.Tensor, x: cute.Tensor, h: cute.Tensor, box: cute.Tensor,
        mc: cutlass.Int64, slot: cutlass.Int32, base: cutlass.Int32,
        beta: cutlass.Float32, linear_beta: cutlass.Float32, front_map: cutlass.Int64,
        gate_map: cutlass.Int64, xbox: cutlass.Int64, parity: cutlass.Int32, stream: _cuda.CUstream,
    ):  # fmt: skip
        self.kernel(
            packed, x, h, box, mc, slot, base, beta, linear_beta, front_map, gate_map, xbox, parity
        ).launch(
            grid=(self.tiles, cute.ceil_div(x.layout.shape[0], self.n) * self.split_k, 1),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(self.smem_bytes),
            stream=stream,
            use_pdl=True,
        )

    @cute.experimental.kernel
    def kernel(
        self, mA: cute.Tensor, mB: cute.Tensor, h: cute.Tensor, box: cute.Tensor,
        mc: cutlass.Int64, slot: cutlass.Int32, base: cutlass.Int32,
        beta: cutlass.Float32, linear_beta: cutlass.Float32, front_map: cutlass.Int64,
        gate_map: cutlass.Int64, xbox: cutlass.Int64, parity: cutlass.Int32,
    ):  # fmt: skip
        tokens = mB.layout.shape[0]
        self.gate_map = gate_map
        self.xbox = (xbox, parity)
        out = (box, mc, slot, base, tokens, h, beta, linear_beta)
        self.body(mA, mB, box, front_map, out)


class MoeFrontTmem64LamxD(MoeFrontTmem64Lamx):
    """F-L per-tile form (K31MK_MOE_FRONT_LAMX value 3; SplitKTmemGemm64 ``lamx_prog`` 2): a poller writes and arrives each K tile as soon
    as its own words are published, and a pass reloads only the tiles still pending (measured: front tiles stored earlier than value 1
    at 16 / 24 / 32 rows)."""

    lamx_prog = 2


class MoeFrontTmem64LamxC(MoeFrontTmem64Lamx):
    """F-L with the canary spin first (``lamx_canary`` True; K31MK_MOE_FRONT_LAMX value 2, measurement form)."""

    lamx_canary = True
