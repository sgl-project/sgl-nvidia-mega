# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""The DFLASH T > 0 sampled accept (chain verification) on the vocab-sharded target logits.

Today (dflash_worker_v2, selector draft, no top-k / top-p): ``finish_deferred_gather`` (multimem
all-gather of the FP32 shard + softcap -> [R, V]), ``_selector_sampling_accept`` (the selector's q
scattered into a dense [B, 7, V]), DSpark ``accept_sampling`` (SoftmaxTemp over V, torch.rand x2,
``chain_speculative_sampling_triton``, bonus gather), then the dflash-selector site's two NCCL
broadcasts of rank 0's (accept_len, bonus). Here, distributionally exact (not bitwise):

``SampledStats``, grid (R, W / 1024), R = 8 B rows of this rank's shard [R, W]. Per 1024-column
slice, z = softcap(x) / T and
  * the online max and sum of exp(z) (the shard's log-sum-exp),
  * the first argmax of softcap(x) (target_predict, which the simulated acceptance reads),
  * the Gumbel-max max(z_v + G_v) over the slice's v outside the row's 16 selector candidates S
    (row 7, the bonus row, has no S); G_v = -log E_v, E_v ~ Exp(1) from one counter-based
    Philox-4x32-10 word keyed by (seed; v / 4, row, round, stream).
The row's last slice CTA merges the slices (fixed order) and multicasts the row's record (those
stats, the round, and z at the S ids this shard owns) into every rank's box [TP, rows, 24] at [rank].

Every box element is a u64 (datum | epoch << 32) written by one b64 multimem store, epoch = the call's
round: a reader takes an element only when its epoch is the round it expects (ctr[2], which the stats
grid sets), so nothing is re-armed and no record of another round can be read. A naturally aligned b64
access is single-copy atomic, so a datum never pairs with another round's epoch. Lamport sentinels with
re-arm stores left records behind after the last call on Rubin: a multicast write applied
after the reader's re-arm (unicast and multicast aliases of one word have no same-location order), and
mid-run such a stale fragment would have been read as the next same-parity record.

``SampledDecide``, grid (B,): polls request b's 8 x TP records for this round, and on every rank (the same
records): LSE over the ranks; p(v) = exp(z_v - LSE) at S; draft x_i is accepted iff
u_i q_i(x_i) < p_i(x_i) (u_i Philox, chain order; q outside (0, 1] rejects, as the Triton kernel); at
the first rejection n the residual max(p_n - q_n, 0) is sampled by one Gumbel-max over
{log(p_n(v) - q_n(v)) + G_v : v in S} and the out-of-S group, whose key max_{v not in S}(z_v + G_v) - LSE
is a Gumbel at the log of that group's mass with its argmax distributed as p_n off S; with every draft
accepted, the bonus row's Gumbel-max over all v (a sample of p_7). Rank 0's (accept_len, bonus) then
reaches every rank in one multicast hop, so ranks cannot diverge. The round
ctr[0] advances once per call, in SampledDecide after the stats grid has completed; the stats grid copies
the round it used to ctr[2], which SampledDecide's CTAs read (ctr[0] changes under them)."""

import ctypes
from functools import cache

import cutlass
import cutlass.cute as cute
import numpy as np
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream

from ..k3.common.host import resolve_kernel, stream
from ..k3.common.primitives import (
    alloc,
    asm,
    at,
    bf16,
    bits_float,
    div_rn,
    float_bits,
    launch,
    load_words,
    store_words,
    wait_deadline_check,
    wait_deadline_start,
    warp_sum,
)

BLOCK = 8


@cute.jit
def _canon(value):
    """FP32 bits with -0.0 as +0.0 (the mailbox sentinel is 0x80000000)."""
    word = Uint32(float_bits(Float32(value)))
    if word == Uint32(0x80000000):
        word = Uint32(0)
    return word


GAMMA, TOPK = (
    BLOCK - 1,
    16,
)  # the defaults (block 8, selector K 16); each build passes its own block / topk
BLOCKS, TOPKS = (
    (5, 8),
    (16, 32),
)  # block 5 = Q5 (1 anchor + 4 drafts), block 8 = Q8; selector K 16 / 32
SHARD_DTYPES = {"f32": torch.float32, "bf16": torch.bfloat16}
SLICE, STATS_THREADS, DECIDE_THREADS = 1024, 128, 256
STATS_WARPS = STATS_THREADS // 32
REC = 24  # m, sum, gumbel key, gumbel id, max, argmax id, round lo16, round hi15, z at the 16 S ids (8 + topk)
ROUND_MASK = (
    0x7FFFFFFF  # rounds are 31-bit: the all-ones epoch of a fresh box is never a round
)
NEG_INF, NO_ID = float("-inf"), 0x7FFFFFFF
PHILOX_M = (0xD2511F53, 0xCD9E8D57)
PHILOX_W = (0x9E3779B9, 0xBB67AE85)
GUMBEL_STREAM, COIN_STREAM = 0x6A09E667, 0x3C6EF372


@asm((Uint32, Uint32), pure=True)
def mulhilo(a, b):
    return (
        "mul.hi.u32 $0,$2,$3; mul.lo.u32 $1,$2,$3;",
        "=r,=r,r,r",
        (Uint32(a), Uint32(b)),
    )


@asm()
def shared_or(pointer, value):
    return "red.shared.or.b32 [$0],$1;", "r,r,~{memory}", (pointer, Uint32(value))


@asm(Uint32)
def atomic_add(pointer, value):
    return (
        "atom.relaxed.gpu.global.add.u32 $0,[$1],$2;",
        "=r,l,r,~{memory}",
        (pointer, Uint32(value)),
    )


@asm()
def multicast_element(address, datum, epoch):
    """One box element (datum | epoch << 32) on every rank: a single relaxed b64 multimem store."""
    code = "{ .reg .b64 e; mov.b64 e, {$1, $2}; multimem.st.relaxed.sys.global.b64 [$0], e; }"
    return code, "l,r,r,~{memory}", (Int64(address), Uint32(datum), Uint32(epoch))


@asm()
def store_element(pointer, datum, epoch):
    """multicast_element's local twin (TP 1)."""
    code = "{ .reg .b64 e; mov.b64 e, {$1, $2}; st.relaxed.sys.global.b64 [$0], e; }"
    return code, "l,r,r,~{memory}", (pointer, Uint32(datum), Uint32(epoch))


@asm((Uint32, Uint32, Uint32, Uint32))
def load_elements(pointer):
    """Two box elements as (datum0, epoch0, datum1, epoch1): one volatile v2.b64 load, each element single-copy
    atomic (a v4.b32 load could pair a datum with another round's epoch)."""
    code = (
        "{ .reg .b64 a, b; ld.volatile.global.v2.b64 {a, b}, [$4]; mov.b64 {$0, $1}, a; "
        "mov.b64 {$2, $3}, b; }"
    )
    return code, "=r,=r,=r,=r,l,~{memory}", (pointer,)


@cute.jit
def fetch_elements(values, i: cutlass.Constexpr[int], pointer, epoch):
    """Elements 2i and 2i + 1 into values[i]; returns bit i set when either is not yet this round's."""
    got = load_elements(pointer)
    values[i, 0] = got[0]
    values[i, 1] = got[2]
    stale = Uint32(0)
    if (got[1] != epoch) | (got[3] != epoch):
        stale = Uint32(1)
    return stale << i


@cute.jit
def poll_round(
    base, count: cutlass.Constexpr[int], epoch, deadline_ns: cutlass.Constexpr[int]
):
    """The data of the 2 ``count`` box elements at ``base`` once every one carries ``epoch`` ([count, 2] words).
    All loads issue before any check and only stale pairs re-poll (as mailbox.poll); a missed deadline
    fail-stops. Nothing is re-armed: the next same-parity round's writes carry a newer epoch."""
    values = cute.make_rmem_tensor((count, 2), Uint32)
    pending = Uint32(0)
    for i in cutlass.range_constexpr(count):
        pending = pending | fetch_elements(values, i, base + 2 * i, epoch)
    started, polls = Int64(0), Int32(0)
    if pending != 0:
        started = wait_deadline_start(deadline_ns)
    while pending != 0:
        polls = wait_deadline_check(started, polls, deadline_ns)
        again = Uint32(0)
        for i in cutlass.range_constexpr(count):
            if (pending >> i) & 1:
                again = again | fetch_elements(values, i, base + 2 * i, epoch)
        pending = again
    return values


@cute.jit
def philox(c0, c1, c2, c3, k0, k1):
    """Philox-4x32-10 (Salmon et al., SC'11): four random words of counter (c0..c3) under key (k0, k1)."""
    for i in cutlass.range_constexpr(10):
        hi0, lo0 = mulhilo(PHILOX_M[0], c0)
        hi1, lo1 = mulhilo(PHILOX_M[1], c2)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        if cutlass.const_expr(i < 9):
            k0 = k0 + Uint32(PHILOX_W[0])
            k1 = k1 + Uint32(PHILOX_W[1])
    return c0, c1, c2, c3


@cute.jit
def pick(words, index):
    """words[index & 3] of a Philox result."""
    lane = index & Int32(3)
    word = words[0]
    if lane == Int32(1):
        word = words[1]
    if lane == Int32(2):
        word = words[2]
    if lane == Int32(3):
        word = words[3]
    return word


@cute.jit
def gumbel(word):
    """G = -log E, E = -log(1 - v) ~ Exp(1) for v = (word + 1/2) / 2^32: the series near 0 keeps E's
    relative precision where the row maxima live (small E = large G)."""
    v = (
        Float32(Int32(word >> Uint32(16))) * Float32(65536.0)
        + Float32(Int32(word & Uint32(0xFFFF)))
        + Float32(0.5)
    ) * Float32(2.0**-32)
    if v > Float32(1.0 - 2.0**-24):
        v = Float32(1.0 - 2.0**-24)
    e = v * (
        Float32(1.0)
        + v
        * (
            Float32(1 / 2)
            + v
            * (
                Float32(1 / 3)
                + v * (Float32(1 / 4) + v * (Float32(1 / 5) + v * Float32(1 / 6)))
            )
        )
    )
    if v >= Float32(2.0**-6):
        e = -cute.math.log(Float32(1.0) - v, fastmath=False)
    return -cute.math.log(e, fastmath=True)


@cute.jit
def uniform(word):
    """[0, 1) on a 2^-24 grid (as torch.rand)."""
    return Float32(Int32(word >> Uint32(8))) * Float32(2.0**-24)


@cute.jit
def lse_merge(m1, s1, m2, s2):
    """(max, sum of exp(z - max)) of two parts; symmetric, so butterfly lanes agree bit for bit."""
    m = cute.arch.fmax(m1, m2)
    total = Float32(0.0)
    if m1 > Float32(NEG_INF):
        total = total + s1 * cute.math.exp(m1 - m, fastmath=True)
    if m2 > Float32(NEG_INF):
        total = total + s2 * cute.math.exp(m2 - m, fastmath=True)
    return m, total


def gtensor(dtype, address, shape, stride, align):
    """``dtype`` tensor at global ``address`` (Int64) with ``layout(shape, stride)``. Every argument of the two
    kernels is a scalar: per-call dlpack views cost host time on every call, and 3-D views with dynamic strides
    indexed every row as row 0 (each row read request 0 / row 0's S and q)."""
    pointer = cute.make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=align)
    return cute.make_tensor(pointer, cute.make_layout(shape, stride=stride))


def load_four(logits, row, j, bf16_shard):
    """logits[row, j:j + 4] as FP32 (a BF16 shard is widened exactly)."""
    if bf16_shard:
        words = load_words(at(logits, row, j), count=2, mutable=True)
        return tuple(bf16(words[k // 2], k % 2 == 1) for k in range(4))
    words = load_words(at(logits, row, j), count=4, mutable=True)
    return tuple(bits_float(word) for word in words)


@cute.jit
def takes(v_new, i_new, v_best, i_best):
    """The larger value; equal values keep the smaller id (the first argmax)."""
    return (v_new > v_best) | ((v_new == v_best) & (i_new < i_best))


def check_shape(block, topk):
    """(block, gamma, topk, rec) of a build: block 5 or 8 rows per request, selector K 16 or 32."""
    if block not in BLOCKS or topk not in TOPKS:
        raise ValueError(
            f"sampled accept serves block {BLOCKS} x selector K {TOPKS}, got block {block} K {topk}"
        )
    return block, block - 1, topk, 8 + topk


class SampledStats:
    """Grid (R, W / 1024), 128 threads, R = block B rows. Arguments (addresses, contiguous): logits [R, W] FP32 or
    BF16 (``dtype``; this rank's raw shard, before the softcap if the model has one; K3.1 has none), temperatures
    [B] FP32, candidate_ids [B, block - 1, K] int64 (S per row), ctr [3] int64 (round, seed, the round in flight:
    set here), scratch [rows, W / 1024, 8] FP32, count [rows] int32 (zero; each row's last CTA re-zeroes it), box
    [TP, rows, 8 + K] u64 elements (this GPU's view of the call's parity box); then mc, rank, col_offset (this
    shard's first vocab id), valid (its real columns), R. block / K: the build's (8 / 16 by default; 5 / 32)."""

    def __init__(
        self,
        tp,
        width,
        softcap,
        rows,
        *,
        dtype="f32",
        local=False,
        block=BLOCK,
        topk=TOPK,
    ):
        self.block, self.gamma, self.topk, self.rec = check_shape(block, topk)
        if width % SLICE or dtype not in SHARD_DTYPES:
            raise ValueError(
                f"SampledStats needs a shard width divisible by {SLICE} and FP32 / BF16, got "
                f"{width} {dtype}"
            )
        self.tp, self.width, self.split, self.local = tp, width, width // SLICE, local
        self.rows, self.bf16_shard = rows, dtype == "bf16"
        self.softcap = None if softcap is None else float(np.float32(softcap))
        self.inverse_softcap = (
            None if softcap is None else float(np.float32(1) / np.float32(softcap))
        )

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, (args[11], self.split, 1), STATS_THREADS)

    @cute.jit
    def logit(self, x):
        if cutlass.const_expr(self.softcap is not None):
            x = cute.math.tanh(
                x * Float32(self.inverse_softcap), fastmath=False
            ) * Float32(self.softcap)
        return x

    @cute.kernel
    def kernel(self, args):
        (
            logits_a,
            temperatures_a,
            ids_a,
            ctr_a,
            scratch_a,
            count_a,
            box_a,
            mc,
            rank,
            col_offset,
            valid,
            rows,
        ) = args
        block, gamma, topk, rec_words = self.block, self.gamma, self.topk, self.rec
        requests = rows // block
        element = cutlass.BFloat16 if self.bf16_shard else Float32
        logits = gtensor(
            element,
            logits_a,
            (rows, self.width),
            (self.width, 1),
            8 if self.bf16_shard else 16,
        )
        temperatures = gtensor(Float32, temperatures_a, requests, 1, 4)
        candidate_ids = gtensor(
            Int64, ids_a, (requests, gamma, topk), (gamma * topk, topk, 1), 8
        )
        ctr = gtensor(Int64, ctr_a, 3, 1, 8)
        scratch = gtensor(
            Float32, scratch_a, (self.rows, self.split, 8), (self.split * 8, 8, 1), 16
        )
        count = gtensor(Int32, count_a, self.rows, 1, 4)
        box = gtensor(
            Int64,
            box_a,
            (self.tp, self.rows, rec_words),
            (self.rows * rec_words, rec_words, 1),
            16,
        )
        row, piece, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane, warp = tid % 32, tid // 32
        req, pos = row // block, row % block
        smem = cutlass.utils.SmemAllocator()
        bitmap = alloc(smem, Uint32, SLICE // 32)
        part_v = alloc(smem, Float32, (STATS_WARPS, 4))
        part_i = alloc(smem, Int32, (STATS_WARPS, 2))
        rec = alloc(smem, Uint32, rec_words)
        last = alloc(smem, Int32, 1)
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        base = piece * SLICE
        if tid < SLICE // 32:
            bitmap[tid] = Uint32(0)
        cute.arch.sync_threads()
        if (pos < gamma) & (
            tid < topk
        ):  # S ids in this slice: excluded from the out-of-S Gumbel-max
            local_id = Int32(candidate_ids[req, pos, tid]) - col_offset - base
            if (local_id >= 0) & (local_id < SLICE) & (local_id + base < valid):
                shared_or(
                    bitmap.iterator + (local_id >> 5),
                    Uint32(1) << Uint32(local_id & 31),
                )
        cute.arch.sync_threads()
        rd, seed = ctr[0], ctr[1]
        round_word = Uint32(rd & Int64(ROUND_MASK))
        if (row == 0) & (piece == 0) & (tid == 0):
            ctr[2] = rd  # for SampledDecide, which reads it after this grid completes
        k0, k1 = (
            Uint32(seed & Int64(0xFFFFFFFF)),
            Uint32((seed >> Int64(32)) & Int64(0xFFFFFFFF)),
        )
        inv_t = div_rn(Float32(1.0), Float32(temperatures[req]))
        m, total = Float32(NEG_INF), Float32(0.0)
        best_l, best_li = Float32(NEG_INF), Int32(NO_ID)
        best_g, best_gi = Float32(NEG_INF), Int32(NO_ID)
        for i in cutlass.range_constexpr(SLICE // (4 * STATS_THREADS)):
            j = base + 4 * (tid + STATS_THREADS * i)
            xs = load_four(logits, row, j, self.bf16_shard)
            noise = philox(
                Uint32((col_offset + j) >> 2),
                Uint32(row),
                round_word,
                Uint32(GUMBEL_STREAM),
                k0,
                k1,
            )
            bits = bitmap[(j - base) >> 5] >> Uint32((j - base) & 31)
            for k in cutlass.range_constexpr(4):
                if j + k < valid:
                    l = self.logit(xs[k])
                    z = l * inv_t
                    m_new = cute.arch.fmax(m, z)
                    total = total * cute.math.exp(
                        m - m_new, fastmath=True
                    ) + cute.math.exp(z - m_new, fastmath=True)
                    m = m_new
                    if l > best_l:
                        best_l = l
                        best_li = col_offset + j + k
                    key = z + gumbel(noise[k])
                    if ((bits >> Uint32(k)) & Uint32(1)) == Uint32(0):
                        if key > best_g:
                            best_g = key
                            best_gi = col_offset + j + k
        for d in cutlass.range_constexpr(5):
            om = cute.arch.shuffle_sync_bfly(m, 1 << d)
            ot = cute.arch.shuffle_sync_bfly(total, 1 << d)
            m, total = lse_merge(m, total, om, ot)
            ol = cute.arch.shuffle_sync_bfly(best_l, 1 << d)
            oli = cute.arch.shuffle_sync_bfly(best_li, 1 << d)
            if takes(ol, oli, best_l, best_li):
                best_l = ol
                best_li = oli
            og = cute.arch.shuffle_sync_bfly(best_g, 1 << d)
            ogi = cute.arch.shuffle_sync_bfly(best_gi, 1 << d)
            if takes(og, ogi, best_g, best_gi):
                best_g = og
                best_gi = ogi
        if lane == 0:
            part_v[warp, 0] = m
            part_v[warp, 1] = total
            part_v[warp, 2] = best_l
            part_v[warp, 3] = best_g
            part_i[warp, 0] = best_li
            part_i[warp, 1] = best_gi
        cute.arch.sync_threads()
        if tid == 0:  # this slice's partial; the row's last slice CTA merges them all
            for w in cutlass.range_constexpr(1, STATS_WARPS):
                merged = lse_merge(m, total, part_v[w, 0], part_v[w, 1])
                m = merged[0]
                total = merged[1]
                if takes(part_v[w, 2], part_i[w, 0], best_l, best_li):
                    best_l = part_v[w, 2]
                    best_li = part_i[w, 0]
                if takes(part_v[w, 3], part_i[w, 1], best_g, best_gi):
                    best_g = part_v[w, 3]
                    best_gi = part_i[w, 1]
            store_words(
                at(scratch, row, piece, 0),
                tuple(Uint32(float_bits(x)) for x in (m, total, best_l, best_g)),
            )
            store_words(at(scratch, row, piece, 4), (Uint32(best_li), Uint32(best_gi)))
            cute.arch.fence_acq_rel_gpu()
            is_last = Int32(0)
            if atomic_add(count.iterator + row, 1) == Uint32(self.split - 1):
                is_last = Int32(1)
            last[0] = is_last
        cute.arch.sync_threads()
        if last[0] != 0:
            if tid == 0:
                cute.arch.fence_acq_rel_gpu()
                m = Float32(NEG_INF)
                total = Float32(0.0)
                best_l = Float32(NEG_INF)
                best_li = Int32(NO_ID)
                best_g = Float32(NEG_INF)
                best_gi = Int32(NO_ID)
                for p in cutlass.range_constexpr(self.split):
                    v = load_words(at(scratch, row, p, 0), count=4, mutable=True)
                    ids = load_words(at(scratch, row, p, 4), count=2, mutable=True)
                    merged = lse_merge(m, total, bits_float(v[0]), bits_float(v[1]))
                    m = merged[0]
                    total = merged[1]
                    if takes(bits_float(v[2]), Int32(ids[0]), best_l, best_li):
                        best_l = bits_float(v[2])
                        best_li = Int32(ids[0])
                    if takes(bits_float(v[3]), Int32(ids[1]), best_g, best_gi):
                        best_g = bits_float(v[3])
                        best_gi = Int32(ids[1])
                count[row] = Int32(0)
                rec[0] = _canon(m)
                rec[1] = _canon(total)
                rec[2] = _canon(best_g)
                rec[3] = _canon(Float32(best_gi))
                rec[4] = _canon(best_l)
                rec[5] = _canon(Float32(best_li))
                rec[6] = _canon(Float32(Int32(round_word & Uint32(0xFFFF))))
                rec[7] = _canon(Float32(Int32(round_word >> Uint32(16))))
            if (
                tid < topk
            ):  # z at the S ids this shard owns (0.0 elsewhere and on the bonus row)
                value = Float32(0.0)
                if pos < gamma:
                    owned = Int32(candidate_ids[req, pos, tid]) - col_offset
                    if (owned >= 0) & (owned < valid):
                        value = self.logit(Float32(logits[row, owned])) * inv_t
                rec[8 + tid] = _canon(value)
            cute.arch.sync_threads()
            if (
                tid < rec_words
            ):  # one (datum, epoch) element per thread (rec_words <= 40 < STATS_THREADS)
                if cutlass.const_expr(self.local):
                    store_element(at(box, rank, row, tid), rec[tid], round_word)
                else:
                    offset = cute.crd2idx((rank, row, tid), box.layout)
                    multicast_element(mc + Int64(offset) * 8, rec[tid], round_word)


class SampledDecide:
    """Grid (B,), 32 block threads (warp w: row block b + w). Arguments (addresses, contiguous): box [TP, rows,
    8 + K] u64 elements (the call's parity), bbox [rows, 2] u64 elements (the decision box, same parity), bmc,
    rank, ctr [3] int64, candidates [B, block] int64 (the verified chain: the root, then the block - 1 drafts),
    candidate_ids [B, block - 1, K] int64, q_rows [B, block - 1, K] FP32, then outputs accept_len [B] int32, bonus
    [B] int64, target [B, block] int64 (the first argmax of each row's softcapped logits), err [1] int32 (set when
    a record's round words differ from the expected round's), B; with ``debug`` also dbg [B, block, 8 + TP (8 + K)]
    FP32 (per row: lse, p and q at the draft, coin, accepted, token, round, hit lanes; then the row's TP records).
    block / K: the build's (8 / 16 by default; 5 / 32). At K = 32 the S candidates fill the warp: the out-of-S
    group's key joins after the butterfly (one compare, the same order)."""

    def __init__(
        self,
        tp,
        width,
        rows,
        *,
        local=False,
        deadline_ns=0,
        debug=False,
        block=BLOCK,
        topk=TOPK,
    ):
        self.block, self.gamma, self.topk, self.rec = check_shape(block, topk)
        self.threads = (
            32 * block
        )  # warp w: row w of the request (no idle warps: block 5 -> 160 threads)
        if tp > 32 - 1:
            raise ValueError(
                f"SampledDecide merges at most 31 ranks in a warp, got tp={tp}"
            )
        self.tp, self.width, self.local, self.deadline_ns = (
            tp,
            width,
            local,
            deadline_ns,
        )
        self.debug, self.rows = debug, rows

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, (args[12], 1, 1), self.threads)

    @cute.kernel
    def kernel(self, args):
        box_a, bbox_a, bmc, rank, ctr_a, candidates_a, ids_a, q_a = args[0:8]
        accept_a, bonus_a, target_a, err_a, requests = args[8:13]
        block, gamma, topk, rec_words = self.block, self.gamma, self.topk, self.rec
        box = gtensor(
            Int64,
            box_a,
            (self.tp, self.rows, rec_words),
            (self.rows * rec_words, rec_words, 1),
            16,
        )
        bbox = gtensor(Int64, bbox_a, (self.rows, 2), (2, 1), 16)
        ctr = gtensor(Int64, ctr_a, 3, 1, 8)
        candidates = gtensor(Int64, candidates_a, (requests, block), (block, 1), 8)
        candidate_ids = gtensor(
            Int64, ids_a, (requests, gamma, topk), (gamma * topk, topk, 1), 8
        )
        q_rows = gtensor(
            Float32, q_a, (requests, gamma, topk), (gamma * topk, topk, 1), 4
        )
        accept_len = gtensor(Int32, accept_a, requests, 1, 4)
        bonus = gtensor(Int64, bonus_a, requests, 1, 8)
        target = gtensor(Int64, target_a, (requests, block), (block, 1), 8)
        err = gtensor(Int32, err_a, 1, 1, 4)
        width = 8 + self.tp * rec_words
        dbg = (
            gtensor(
                Float32,
                args[13],
                (requests, block, width),
                (block * width, width, 1),
                4,
            )
            if self.debug
            else None
        )
        req, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane, w = tid % 32, tid // 32
        smem = cutlass.utils.SmemAllocator()
        recs = alloc(smem, Float32, (block, self.tp, rec_words))
        accepted = alloc(smem, Int32, block)
        tokens = alloc(smem, Int32, block)
        cute.arch.griddepcontrol_wait()  # the stats grid has completed: every read of the round is done
        cute.arch.griddepcontrol_launch_dependents()
        round_word = Uint32(
            ctr[2] & Int64(ROUND_MASK)
        )  # this call's round (the epoch every element must carry)
        if (req == 0) & (tid == 0):
            ctr[0] = ctr[0] + Int64(1)
        if (
            tid < block * self.tp
        ):  # poll rank s's record of row r for this round (block * tp <= 32 * block threads)
            r, s = tid // self.tp, tid % self.tp
            got = poll_round(
                at(box, s, req * block + r, 0),
                rec_words // 2,
                round_word,
                self.deadline_ns,
            )
            for i in cutlass.range_constexpr(rec_words // 2):
                for k in cutlass.range_constexpr(2):
                    recs[r, s, 2 * i + k] = bits_float(got[i, k])
        cute.arch.sync_threads()
        if cutlass.const_expr(self.debug):
            for i in range(tid, block * self.tp * rec_words, self.threads):
                dbg[req, i // (self.tp * rec_words), 8 + i % (self.tp * rec_words)] = (
                    recs[
                        i // (self.tp * rec_words),
                        (i // rec_words) % self.tp,
                        i % rec_words,
                    ]
                )
        seed = ctr[1]
        k0, k1 = (
            Uint32(seed & Int64(0xFFFFFFFF)),
            Uint32((seed >> Int64(32)) & Int64(0xFFFFFFFF)),
        )
        m, total = Float32(NEG_INF), Float32(0.0)
        best_l, best_li = Float32(NEG_INF), Int32(NO_ID)
        best_g, best_gi = Float32(NEG_INF), Int32(NO_ID)
        if lane < self.tp:
            m = recs[w, lane, 0]
            total = recs[w, lane, 1]
            best_g = recs[w, lane, 2]
            best_gi = Int32(recs[w, lane, 3])
            best_l = recs[w, lane, 4]
            best_li = Int32(recs[w, lane, 5])
            lo = Float32(Int32(round_word & Uint32(0xFFFF)))
            hi = Float32(Int32(round_word >> Uint32(16)))
            if (recs[w, lane, 6] != lo) | (
                recs[w, lane, 7] != hi
            ):  # a cross-check: the epochs already matched
                err[0] = Int32(1)
        for d in cutlass.range_constexpr(5):
            om = cute.arch.shuffle_sync_bfly(m, 1 << d)
            ot = cute.arch.shuffle_sync_bfly(total, 1 << d)
            m, total = lse_merge(m, total, om, ot)
            ol = cute.arch.shuffle_sync_bfly(best_l, 1 << d)
            oli = cute.arch.shuffle_sync_bfly(best_li, 1 << d)
            if takes(ol, oli, best_l, best_li):
                best_l = ol
                best_li = oli
            og = cute.arch.shuffle_sync_bfly(best_g, 1 << d)
            ogi = cute.arch.shuffle_sync_bfly(best_gi, 1 << d)
            if takes(og, ogi, best_g, best_gi):
                best_g = og
                best_gi = ogi
        lse = m + cute.math.log(total, fastmath=False)
        if lane == 0:
            target[req, w] = Int64(best_li)
        accept, token = Int32(0), best_gi  # the bonus row: its Gumbel-max over every v
        if w < gamma:
            gid = Int32(-1) - lane
            p = Float32(0.0)
            q = Float32(0.0)
            if lane < topk:
                gid = Int32(candidate_ids[req, w, lane])
                z = Float32(NEG_INF)
                owner = gid // self.width
                if (gid >= 0) & (owner < self.tp):
                    z = recs[w, owner, 8 + lane]
                p = cute.math.exp(z - lse, fastmath=False)
                q = Float32(q_rows[req, w, lane])
                if (q < Float32(0.0)) | (q > Float32(1.0)) | (q != q):
                    q = Float32(0.0)
            q_sum = Float32(0.0)  # a repeated id: its q summed, kept once
            first = Int32(1)
            for j in cutlass.range_constexpr(topk):
                other = cute.arch.shuffle_sync(gid, j)
                other_q = cute.arch.shuffle_sync(q, j)
                if other == gid:
                    q_sum = q_sum + other_q
                    if Int32(j) < lane:
                        first = Int32(0)
            hit = (lane < topk) & (first != 0) & (gid == Int32(candidates[req, w + 1]))
            p_draft = Float32(0.0)
            q_draft = Float32(0.0)
            hit_f = Float32(0.0)
            if hit:
                p_draft = p
                q_draft = q_sum
                hit_f = Float32(1.0)
            p_draft = warp_sum(p_draft)
            q_draft = warp_sum(q_draft)
            coin = uniform(
                philox(Uint32(w), Uint32(req), round_word, Uint32(COIN_STREAM), k0, k1)[
                    0
                ]
            )
            if (
                (q_draft > Float32(0.0))
                & (q_draft <= Float32(1.0))
                & (coin * q_draft < p_draft)
            ):
                accept = Int32(1)
            if cutlass.const_expr(self.debug):
                hits = warp_sum(hit_f)
                if lane == 0:
                    dbg[req, w, 1] = p_draft
                    dbg[req, w, 2] = q_draft
                    dbg[req, w, 3] = coin
                    dbg[req, w, 7] = hits
            key = Float32(
                NEG_INF
            )  # the residual max(p - q, 0): S keys + the out-of-S group
            choice = Int32(NO_ID)
            residual = p - q_sum
            if (lane < topk) & (first != 0) & (residual > Float32(0.0)):
                noise = philox(
                    Uint32(gid >> 2),
                    Uint32(req * block + w),
                    round_word,
                    Uint32(GUMBEL_STREAM),
                    k0,
                    k1,
                )
                key = cute.math.log(residual, fastmath=False) + gumbel(pick(noise, gid))
                choice = gid
            for d in cutlass.range_constexpr(5):
                ok = cute.arch.shuffle_sync_bfly(key, 1 << d)
                oc = cute.arch.shuffle_sync_bfly(choice, 1 << d)
                if takes(ok, oc, key, choice):
                    key = ok
                    choice = oc
            # The out-of-S group's key after the butterfly (K = 32 fills every lane with S): one compare under
            # the same total order (larger key, then smaller id), so the argmax equals the old lane-K entry's.
            # best_g / best_gi / lse are warp-uniform (their butterflies above), so every lane agrees.
            if takes(best_g - lse, best_gi, key, choice):
                key = best_g - lse
                choice = best_gi
            token = choice
        if lane == 0:
            accepted[w] = accept
            tokens[w] = token
            if cutlass.const_expr(self.debug):
                dbg[req, w, 0] = lse
                dbg[req, w, 4] = Float32(accept)
                dbg[req, w, 5] = Float32(token)
                dbg[req, w, 6] = Float32(Int32(round_word))
        cute.arch.sync_threads()
        if tid == 0:
            n = Int32(0)
            live = Int32(1)
            for i in cutlass.range_constexpr(gamma):
                live = live & accepted[i]
                n = n + live
            chosen = tokens[n]
            if cutlass.const_expr(self.local):
                accept_len[req] = n
                bonus[req] = Int64(chosen)
            else:
                # rank 0's decision on every rank: (n, e), (token, e)
                if rank == Int32(0):
                    offset = cute.crd2idx((req, 0), bbox.layout)
                    multicast_element(bmc + Int64(offset) * 8, Uint32(n), round_word)
                    multicast_element(
                        bmc + (Int64(offset) + Int64(1)) * 8, Uint32(chosen), round_word
                    )
                got = poll_round(at(bbox, req, 0), 1, round_word, self.deadline_ns)
                accept_len[req] = Int32(got[0, 0])
                bonus[req] = Int64(got[0, 1])  # a token id: below 2^31


@cache
def compile_sampled_stats(
    tp, width, softcap, rows, local, dtype="f32", block=BLOCK, topk=TOPK
):
    address, value = cutlass.Int64(0), cutlass.Int32(0)
    args = (address,) * 8 + (value,) * 4
    return cute.compile(
        SampledStats(
            tp, width, softcap, rows, dtype=dtype, local=local, block=block, topk=topk
        ),
        *args,
        make_fake_stream(),
    )


@cache
def compile_sampled_decide(
    tp, width, rows, local, deadline_ns, debug=False, block=BLOCK, topk=TOPK
):
    address, value = cutlass.Int64(0), cutlass.Int32(0)
    args = (
        (address, address, address, value)
        + (address,) * 8
        + (value,)
        + ((address,) if debug else ())
    )
    return cute.compile(
        SampledDecide(
            tp,
            width,
            rows,
            local=local,
            deadline_ns=deadline_ns,
            debug=debug,
            block=block,
            topk=topk,
        ),
        *args,
        make_fake_stream(),
    )


class _LaunchAttribute(
    ctypes.Structure
):  # CUlaunchAttribute: id, pad to 8, a 64-byte value union
    _fields_ = [
        ("id", ctypes.c_uint32),
        ("pad", ctypes.c_uint8 * 4),
        ("value", ctypes.c_uint8 * 64),
    ]


class _LaunchConfig(ctypes.Structure):  # CUlaunchConfig
    _fields_ = [
        (name, ctypes.c_uint)
        for name in (
            "gridDimX",
            "gridDimY",
            "gridDimZ",
            "blockDimX",
            "blockDimY",
            "blockDimZ",
            "sharedMemBytes",
        )
    ]
    _fields_ += [
        ("hStream", ctypes.c_void_p),
        ("attrs", ctypes.POINTER(_LaunchAttribute)),
        ("numAttrs", ctypes.c_uint),
    ]


@cache
def _libcuda_launch():
    function = ctypes.CDLL("libcuda.so.1").cuLaunchKernelEx
    function.restype = ctypes.c_int
    function.argtypes = (
        ctypes.POINTER(_LaunchConfig),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    )
    return function


class _DriverLaunch:
    """A compiled kernel launched with cuLaunchKernelEx (PDL, as ``launch``) from preallocated ctypes arguments: the
    CuTe call path costs far more host time per launch. Built
    only when cuFuncGetParamInfo shows exactly ``sizes``; otherwise the caller keeps the CuTe call. ``shared``
    bounds the kernel's dynamic shared memory from above (its SmemAllocator carves from the base). ``mode`` "raw":
    libcuda's cuLaunchKernelEx through ctypes on a prebuilt CUlaunchConfig (the cuda-python call alone kept
    run() slow on the host); "cuda-python" when that struct cannot be checked against cuda-python's."""

    def __init__(self, compiled, device, sizes, threads, shared):
        from cuda.bindings import driver as cu

        _, function = resolve_kernel(compiled, device.index)
        got = []
        while len(got) <= len(sizes):
            result = cu.cuFuncGetParamInfo(function, len(got))
            if result[0] != cu.CUresult.CUDA_SUCCESS:
                break
            got.append(int(result[2]))
        if got != list(sizes):
            raise RuntimeError(f"kernel parameter sizes {got}, expected {list(sizes)}")
        self.cu, self.function = cu, function
        self.values = [
            ctypes.c_int64() if size == 8 else ctypes.c_int32() for size in sizes
        ]
        self.pointers = (ctypes.c_void_p * len(sizes))(
            *(ctypes.addressof(v) for v in self.values)
        )
        self.address = ctypes.addressof(self.pointers)
        config = cu.CUlaunchConfig()
        config.blockDimX, config.blockDimY, config.blockDimZ = threads, 1, 1
        config.sharedMemBytes = shared
        attribute = cu.CUlaunchAttribute()
        attribute.id = (
            cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
        )
        attribute.value.programmaticStreamSerializationAllowed = 1
        config.attrs = [attribute]
        config.numAttrs = 1
        self.config, self.raw, self.mode = config, None, "cuda-python"
        try:
            self.raw = self._raw_config(cu, function, config, threads, shared)
            self.mode = "raw"
        except Exception:  # noqa: BLE001 (keeps the cuda-python launch)
            self.raw = None

    @staticmethod
    def _raw_config(cu, function, config, threads, shared):
        """The same launch as ``config`` as a ctypes CUlaunchConfig; its plain fields must equal cuda-python's
        struct byte for byte (grid / block / shared memory at 0-27, numAttrs at 48), the PDL id must be 6."""
        pdl = int(
            cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
        )
        if (
            pdl != 6
            or ctypes.sizeof(_LaunchConfig) != 56
            or ctypes.sizeof(_LaunchAttribute) != 72
        ):
            raise RuntimeError("CUlaunchConfig layout")
        attribute = _LaunchAttribute()
        attribute.id = pdl
        attribute.value[0] = 1
        raw = _LaunchConfig()
        raw.blockDimX, raw.blockDimY, raw.blockDimZ = threads, 1, 1
        raw.gridDimX, raw.gridDimY, raw.gridDimZ = 3, 5, 7
        raw.sharedMemBytes = shared
        raw.attrs = ctypes.pointer(attribute)
        raw.numAttrs = 1
        config.gridDimX, config.gridDimY, config.gridDimZ = 3, 5, 7
        theirs = ctypes.string_at(int(config.getPtr()), 56)
        ours = ctypes.string_at(ctypes.addressof(raw), 56)
        if theirs[:28] != ours[:28] or theirs[48:52] != ours[48:52]:
            raise RuntimeError("CUlaunchConfig bytes differ from cuda-python's")
        # Not the probe's grid: with it, every kernel launched with gridDimZ 7 (7 copies of each CTA) and trapped.
        raw.gridDimX = raw.gridDimY = raw.gridDimZ = 1
        config.gridDimX = config.gridDimY = config.gridDimZ = 1
        return _libcuda_launch(), raw, attribute, ctypes.c_void_p(int(function))

    def __call__(self, grid, stream_handle, values):
        for slot, value in zip(self.values, values):
            slot.value = value
        if self.raw is not None:
            launch_raw, raw, _, function = self.raw
            raw.gridDimX, raw.gridDimY = grid
            raw.gridDimZ = 1
            raw.hStream = stream_handle
            code = launch_raw(ctypes.byref(raw), function, self.address, None)
            if code:
                raise RuntimeError(f"cuLaunchKernelEx failed: {code}")
            return
        config = self.config
        config.gridDimX, config.gridDimY = grid
        config.gridDimZ = 1
        config.hStream = self.cu.CUstream(stream_handle)
        result = self.cu.cuLaunchKernelEx(config, self.function, self.address, 0)
        if result[0] != self.cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuLaunchKernelEx failed: {result[0]!r}")


# Output ring depth per batch size on the driver path. The outputs are read only on the forward stream after the
# call that wrote them: bonus by the next round's draft (or the overlap scheduler's FutureMap stash copy), accept_len
# by _commit_accept / the simulated acceptance, target by the simulated acceptance; the scheduler gets fresh
# tensors (_commit_accept's out_tokens / commit_lens). The overlap scheduler runs one round ahead, so every read of
# call k's outputs is enqueued before call k + 2 writes; 4 slots leave a margin.
RING = 4
STATS_SIZES = (8,) * 8 + (4,) * 4
DECIDE_SIZES = (8, 8, 8, 4) + (8,) * 8 + (4,)


class SampledAccept:
    """Host of SampledStats + SampledDecide with its own symm-mem boxes on ``group`` (records [2 parity, TP,
    rows, 24], decisions [2 parity, rows, 2], u64 (datum | epoch) elements); construction and calls are
    collective (every rank, same order).
    ``run`` alternates the parity unless one is given. ``seed`` must be the same on every rank (SGLang seeds
    every rank's torch generator from server_args.random_seed). ``local``: one GPU, TP 1, no multicast. ``debug``
    (tests only): ``run`` also returns SampledDecide's dbg tensor."""

    def __init__(
        self,
        group,
        device,
        rank,
        tp,
        *,
        width,
        vocab,
        softcap,
        seed,
        rows=None,
        deadline_ns=0,
        local=False,
        debug=False,
        block=BLOCK,
        topk=TOPK,
    ):
        self.block, self.gamma, self.topk, self.rec = check_shape(block, topk)
        if (
            rows is None
        ):  # 64 requests of the block: 512 at block 8 (as before), 320 at block 5 (512 % 5 != 0)
            rows = 64 * block
        if (
            width % SLICE
            or tp * width >= 2**24
            or not 0 < vocab <= tp * width
            or rows % block
        ):
            raise ValueError(
                f"sampled accept of vocab {vocab} over {tp} x {width} (rows {rows}, block {block})"
            )
        if local != (tp == 1):
            raise ValueError("the sampled accept runs locally exactly at TP 1")
        self.tp, self.rank, self.rows, self.width, self.device = (
            tp,
            rank,
            rows,
            width,
            device,
        )
        self.col_offset = rank * width
        self.valid = max(0, min(width, vocab - self.col_offset))
        if local:
            self.box = torch.empty(
                (2, tp, rows, self.rec), dtype=torch.int64, device=device
            )
            self.bbox = torch.empty((2, rows, 2), dtype=torch.int64, device=device)
            self.mc = self.bmc = 0
        else:
            import torch.distributed as dist
            import torch.distributed._symmetric_memory as symm_mem

            self.box = symm_mem.empty(
                (2, tp, rows, self.rec), dtype=torch.int64, device=device
            )
            self.mc = int(symm_mem.rendezvous(self.box, group).multicast_ptr or 0)
            self.bbox = symm_mem.empty((2, rows, 2), dtype=torch.int64, device=device)
            self.bmc = int(symm_mem.rendezvous(self.bbox, group).multicast_ptr or 0)
            if not (self.mc and self.bmc):
                raise RuntimeError("the sampled accept needs multicast memory")
        self.box.fill_(-1)  # every epoch all-ones: never a (31-bit) round
        self.bbox.fill_(-1)
        self.scratch = torch.empty(
            (rows, width // SLICE, 8), dtype=torch.float32, device=device
        )
        self.count = torch.zeros(rows, dtype=torch.int32, device=device)
        self.ctr = torch.tensor(
            [0, seed & (2**63 - 1), 0], dtype=torch.int64, device=device
        )
        self.err = torch.zeros(1, dtype=torch.int32, device=device)
        torch.cuda.synchronize(device)
        if not local:
            dist.barrier(
                group=group
            )  # every box initialized before any rank's first publish
        # Both shard dtypes at build time (the verify head's output dtype is the model's: BF16 for K3.1).
        self.stats = {
            SHARD_DTYPES[d]: compile_sampled_stats(
                tp, width, softcap, rows, local, d, block, topk
            )
            for d in SHARD_DTYPES
        }
        self.decide = compile_sampled_decide(
            tp, width, rows, local, deadline_ns, debug, block, topk
        )
        self.debug = debug
        self.parity, self.calls = 0, 0
        address = lambda t: cutlass.Int64(t.data_ptr())
        self.fixed = (
            address(self.ctr),
            address(self.scratch),
            address(self.count),
            address(self.err),
        )
        multicast = lambda base, t, p: cutlass.Int64(
            base + p * t[p].numel() * t.element_size() if base else 0
        )
        self.boxes = tuple(
            (
                address(self.box[p]),
                multicast(self.mc, self.box, p),
                address(self.bbox[p]),
                multicast(self.bmc, self.bbox, p),
            )
            for p in (0, 1)
        )
        self.scalars = (
            cutlass.Int32(rank),
            cutlass.Int32(self.col_offset),
            cutlass.Int32(self.valid),
        )
        # The driver launch (not with debug): the same arguments as plain ints.
        self.raw_fixed = tuple(
            int(t.data_ptr()) for t in (self.ctr, self.scratch, self.count, self.err)
        )
        raw_mc = lambda base, t, p: (
            base + p * t[p].numel() * t.element_size() if base else 0
        )
        self.raw_boxes = tuple(
            (
                int(self.box[p].data_ptr()),
                raw_mc(self.mc, self.box, p),
                int(self.bbox[p].data_ptr()),
                raw_mc(self.bmc, self.bbox, p),
            )
            for p in (0, 1)
        )
        self.driver, self.launch_mode, self.rings = None, "cute", {}
        if not debug:
            try:
                stats = {
                    dt: _DriverLaunch(code, device, STATS_SIZES, STATS_THREADS, 1024)
                    for dt, code in self.stats.items()
                }
                # SampledDecide's smem: recs (block, tp, rec) FP32 + accepted / tokens; block 8 / K 16: 768 tp + 1024
                decide = _DriverLaunch(
                    self.decide,
                    device,
                    DECIDE_SIZES,
                    32 * block,
                    4 * block * self.rec * tp + 1024,
                )
                self.driver = (stats, decide)
                self.launch_mode = "driver/" + "/".join(
                    sorted({k.mode for k in (*stats.values(), decide)})
                )
            except Exception as error:  # noqa: BLE001 (any mismatch keeps the CuTe call)
                self.launch_mode = f"cute ({type(error).__name__}: {str(error)[:160]})"

    def ring_outputs(self, b, device):
        """The next of RING preallocated (accept_len, bonus, target) slots for batch size b, with their addresses."""
        ring = self.rings.get(b)
        if ring is None:
            block = self.block
            store = torch.empty(
                (RING, b * (block + 2)), dtype=torch.int64, device=device
            )
            slots = []
            for row in store:
                bonus, target = row[:b], row[b : b * (block + 1)].view(b, block)
                accept_len = row[b * (block + 1) :].view(torch.int32)[:b]
                slots.append(
                    (
                        accept_len,
                        bonus,
                        target,
                        accept_len.data_ptr(),
                        bonus.data_ptr(),
                        target.data_ptr(),
                    )
                )
            ring = self.rings[b] = [slots, 0]
        slots, i = ring
        ring[1] = (i + 1) % RING
        return slots[i]

    def driver_stream(self, device):
        """The current stream's raw handle (an int)."""
        raw = getattr(torch._C, "_cuda_getCurrentRawStream", None)
        if raw is not None:
            return raw(device.index)
        return torch.cuda.current_stream(device).cuda_stream

    def fits(self, requests, logits=None):
        return 0 < requests * self.block <= self.rows and (
            logits is None or logits.dtype in self.stats
        )

    def run(self, logits, temperatures, candidates, candidate_ids, q_rows, parity=None):
        """logits [block B, W] FP32 or BF16 (this rank's raw shard), temperatures [B] (or [B, 1]), candidates
        [B, block] int64, candidate_ids [B, block - 1, K] int64, q_rows [B, block - 1, K]: returns accept_len [B]
        int32, bonus [B] int64 (rank 0's decision on every rank) and target [B, block] int64 (each row's first
        argmax). block / K: this build's (8 / 16 by default)."""
        b = candidates.shape[0]
        block, gamma, topk = self.block, self.gamma, self.topk
        if (
            not self.fits(b, logits)
            or tuple(logits.shape) != (b * block, self.width)
            or not logits.is_contiguous()
            or tuple(candidates.shape) != (b, block)
            or tuple(candidate_ids.shape) != (b, gamma, topk)
            or tuple(q_rows.shape) != (b, gamma, topk)
            or temperatures.numel() != b
        ):
            raise ValueError(
                f"sampled accept of logits {tuple(logits.shape)} {logits.dtype}, candidates "
                f"{tuple(candidates.shape)}, candidate_ids {tuple(candidate_ids.shape)}, q_rows "
                f"{tuple(q_rows.shape)}, temperatures {tuple(temperatures.shape)} (width "
                f"{self.width}, rows {self.rows})"
            )
        # Conversions only when needed: each no-op call still costs host time on the eager path.
        if temperatures.dtype != torch.float32 or not temperatures.is_contiguous():
            temperatures = temperatures.reshape(b).to(torch.float32).contiguous()
        if candidates.dtype != torch.int64 or not candidates.is_contiguous():
            candidates = candidates.to(torch.int64).contiguous()
        if candidate_ids.dtype != torch.int64 or not candidate_ids.is_contiguous():
            candidate_ids = candidate_ids.to(torch.int64).contiguous()
        if q_rows.dtype != torch.float32 or not q_rows.is_contiguous():
            q_rows = q_rows.to(torch.float32).contiguous()
        logits_address = logits.data_ptr()
        if logits_address % (16 if logits.dtype == torch.float32 else 8):
            raise ValueError(
                "sampled accept needs 16-byte (FP32) / 8-byte (BF16) aligned logits"
            )
        if parity is None:
            parity, self.parity = self.parity, self.parity ^ 1
        if (
            self.driver is not None
        ):  # outputs from the per-bs ring (RING), no allocation or view per call
            accept_len, bonus, target, accept_address, bonus_address, target_address = (
                self.ring_outputs(b, logits.device)
            )
            stats, decide = self.driver
            box, mc, bbox, bmc = self.raw_boxes[parity]
            ctr, scratch, count, err = self.raw_fixed
            handle = self.driver_stream(logits.device)
            ids = candidate_ids.data_ptr()
            stats[logits.dtype](
                (b * block, self.width // SLICE),
                handle,
                (
                    logits_address,
                    temperatures.data_ptr(),
                    ids,
                    ctr,
                    scratch,
                    count,
                    box,
                    mc,
                    self.rank,
                    self.col_offset,
                    self.valid,
                    b * block,
                ),
            )
            decide(
                (b, 1),
                handle,
                (
                    box,
                    bbox,
                    bmc,
                    self.rank,
                    ctr,
                    candidates.data_ptr(),
                    ids,
                    q_rows.data_ptr(),
                    accept_address,
                    bonus_address,
                    target_address,
                    err,
                    b,
                ),
            )
            self.calls += 1
            return accept_len, bonus, target
        out = torch.empty(
            b * (block + 2), dtype=torch.int64, device=logits.device
        )  # one allocation for the three
        bonus, target = out[:b], out[b : b * (block + 1)].view(b, block)
        accept_len = out[b * (block + 1) :].view(torch.int32)[:b]
        box, mc, bbox, bmc = self.boxes[parity]
        ctr, scratch, count, err = self.fixed
        rank, col_offset, valid = self.scalars
        address = lambda t: cutlass.Int64(t.data_ptr())
        ids = address(candidate_ids)
        s = stream(logits.device)
        extra = ()
        if self.debug:
            dbg = torch.zeros(
                (b, block, 8 + self.tp * self.rec),
                dtype=torch.float32,
                device=logits.device,
            )
            extra = (address(dbg),)
        self.stats[logits.dtype](
            address(logits),
            address(temperatures),
            ids,
            ctr,
            scratch,
            count,
            box,
            mc,
            rank,
            col_offset,
            valid,
            cutlass.Int32(b * block),
            s,
        )
        self.decide(
            box,
            bbox,
            bmc,
            rank,
            ctr,
            address(candidates),
            ids,
            address(q_rows),
            address(accept_len),
            address(bonus),
            address(target),
            err,
            cutlass.Int32(b),
            *extra,
            s,
        )
        self.calls += 1
        if self.debug:
            return accept_len, bonus, target, dbg
        return accept_len, bonus, target
