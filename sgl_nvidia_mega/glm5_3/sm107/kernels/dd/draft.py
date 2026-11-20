# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.memory import SmemAllocator

from . import bf16_moe as moe
from . import draft_s0 as s0
from . import lm_head as lm
from .preparation import (
    ARENA_WORDS,
    COPIES,
    COUNTER_WORDS,
    HOP_WORDS,
    LG_WORDS,
    TK_WORDS,
    TRANSPORT_WORDS,
    Y_WORDS,
)

HIDDEN = 6144
RANKS = 16
VOCAB_SHARD = 9680
SHARE = HIDDEN // RANKS
TOKEN = 7
SHARED_TOKEN = 6

PK_HEAD = 0
PK_ROW = 4
PK_HID = 4 + HIDDEN // 2
PK_SLOT = PK_HID + SHARE
PK_COPY = RANKS * PK_SLOT
PK_BASE = TRANSPORT_WORDS
EH_BASE = PK_BASE + COPIES * PK_COPY
EMB_BASE = EH_BASE + s0.EH_WORDS
DRAFT_TRANSPORT_WORDS = EMB_BASE + s0.EMB_WORDS

C_KEYS = 768
C_ARR = 784
C_XREADY = 785
C_FRONT = 786

P1_WORD = 306944 // 4
D_NEXTX = P1_WORD
D_NEXTSSQ = P1_WORD + 24576
D_LOGITS = P1_WORD + 24608
D_ACT = P1_WORD + 24864
D_NW = P1_WORD + 26112
D_XN = P1_WORD + 32768
D_NS = P1_WORD + 57344

LM_FP8_STAGES = 3
if LM_FP8_STAGES != lm.STAGES:
    if not (3 <= LM_FP8_STAGES <= 5 and lm.REL_SYNC):
        raise TypeError("LM_FP8_STAGES must be 3..5 (REL_SYNC ring)")
    lm.STAGES8 = LM_FP8_STAGES
    lm.BARS = lm.STAGES + lm.X_STAGES + (LM_FP8_STAGES - lm.STAGES)
B_ROUTER, B_FC1, B_FC2, B_SD, B_LM, B_HH = (
    0,
    1,
    1 + moe.FC1_MAX_ROWS,
    2 + moe.FC1_MAX_ROWS,
    3 + moe.FC1_MAX_ROWS,
    3 + moe.FC1_MAX_ROWS + lm.BARS,
)
HH_SLOTS = 16
B_EH = B_HH + HH_SLOTS
DSM_BARS = B_EH + s0.EH_BARS
DSM_BIAS = moe.SM_WORDS + 2 * DSM_BARS
DSM_WORDS = DSM_BIAS + 256
BIAS_SMEM = True
ROUTER_LG = True
LG_W_WORDS = 16 * 38 * 512
W_ROUTER_ELEMS = 256 * 6144 + (2 * LG_W_WORDS if ROUTER_LG else 0)
ROUTE_EARLY_ISSUE = False
LM_L2_KEEP_CHUNKS = 0
if LM_L2_KEEP_CHUNKS:
    lm.KEEP_CHUNKS = LM_L2_KEEP_CHUNKS
H2_XAMAX = False
SHORT_CTX = True
LM_FP8 = False
LM_FP8_KEEP = False
if LM_FP8_KEEP:
    lm.FP8_KEEP = True
    lm.W_EVICT_FIRST = True
LM_FP8X = False
LM_FP8_RESCORE = True
LM_FP4 = True
LM_FP4_TILES = False
if LM_FP4_TILES:
    lm.FP4_RESCORE_TILES = True
LM_FP4_DOT = True
if LM_FP4_DOT:
    lm.FP4_DOT = True
    lm.FP4_UNROLL = 4
W_LM_ELEMS = (
    VOCAB_SHARD * HIDDEN
    + ((VOCAB_SHARD * HIDDEN) // 2 + 2 * VOCAB_SHARD if LM_FP8 else 0)
    + ((VOCAB_SHARD * lm.FP4_ROW_BYTES + 4 * VOCAB_SHARD) // 2 if LM_FP4 else 0)
)
LM_TABLE_PF = False
HOP_LEAN = True
# Idle H1 / H2 units run one pre-pass of the hop on scratch, so its run-once code is instruction-cache warm when the
# shards land.
WARM_HOP = True
WARM_H2 = True
# The Y-copy clear waits for the local H1, so it still follows every rank's O shard of the row.
SEL_H1 = True
H2_IDLE = True
WARM_L2_PROBE = False
H2_CLEAR_LATE = True
WARM_O = True
WARM_MERGE = True

KIND_SK, KIND_T, KIND_S0 = 0, 1, 2
KIND_TA = 3
TA_CTAS = 80
KIND_TS = 4
TS_BRIDGE = False
TS_BRIDGE_FLAG = 760
TS_BRIDGE_ZERO_CTAS = (2 * COUNTER_WORDS + 255) // 256
# Rolled loops: a few hundred instructions instead of the unrolled bodies, whose cold instruction fetch dominated in
# serving.
TS_ROLLED = True
TS_LATE_DEP = False
TS_PREDICT_ZERO = True
TS_ROW = 260
TS_R3 = 8 * TS_ROW
TS_RES = 8 * TS_ROW
TS_PRUNE = 1.0 / (1 << 29)
TS_LANE_HIST = 32 * 257
TS_RES_ROW = 24
L2_PREFETCH = True
D_XAMAX = P1_WORD + 60544
D_SH = P1_WORD + 57472
ONE_ROW_H1 = False
ROW_HOP = False
ROW_CTA = 211
C_H1 = 787
H2_CLEAR_CURRENT = True
T_CLEAR_PREVIOUS_HOP2 = True
LM_PF_FC2_CHUNKS = 0
LM_PF_CHUNKS = 0
SHARED_IN_FC2 = True
T_HEADS = 8 * RANKS * 4
FC1_CTAS = 208 if H2_IDLE else 212
if WARM_L2_PROBE and not (SEL_H1 and HOP_LEAN and not WARM_HOP):
    raise TypeError("WARM_L2_PROBE needs SEL_H1 + HOP_LEAN and WARM_HOP off")
SD_CTAS = 20
HH_CTAS = 10


@dsl_user_op
def fence_proxy_global(*, loc=None, ip=None):
    """Generic-proxy writes made visible by an acquire are then read by bulk copies, which use the async proxy."""
    cute.arch.fence_proxy("async.global", loc=loc, ip=ip)


lds64 = lm.lds64w


@cute.jit
def shared_rows32(
    unit: cutlass.Int32,
    act: cute.Tensor,
    w_sdn: cute.Tensor,
    base: cutlass.Uint32,
    rows_out: cute.Tensor,
    bar: cute.Pointer,
):
    tid, _, _ = cute.arch.thread_idx()
    warp, lane = tid >> 5, tid & 31
    if tid == 0:
        cute.arch.mbarrier_arrive_and_expect_tx(bar, 32 * moe.INTER * 2)
        bulk_copy(
            base,
            w_sdn.iterator + unit * 32 * moe.INTER,
            cutlass.Uint32(32 * moe.INTER * 2),
            bar,
            evict_first(),
        )
    a0, a1, a2, a3 = moe.poll_words(act, moe.TOPK * moe.INTER + 4 * lane)
    av = (
        bits_f32(a0 & cutlass.Uint32(0xFFFF0000)),
        bits_f32(a1 & cutlass.Uint32(0xFFFF0000)),
        bits_f32(a2 & cutlass.Uint32(0xFFFF0000)),
        bits_f32(a3 & cutlass.Uint32(0xFFFF0000)),
    )
    cute.arch.mbarrier_wait(bar, 0)
    for i in cutlass.range_constexpr(4):
        r = 4 * warp + i
        lo, hi = lds64(base + ((r * moe.INTER + 4 * lane) * 2).to(cutlass.Uint32))
        v = bf16_from_word(lo, False) * av[0]
        v = fma_rn(bf16_from_word(lo, True), av[1], v)
        v = fma_rn(bf16_from_word(hi, False), av[2], v)
        v = fma_rn(bf16_from_word(hi, True), av[3], v)
        for sh in cutlass.range_constexpr(5):
            v = add_rn(v, cute.arch.shuffle_sync_bfly(v, 16 >> sh))
        if lane == 0:
            rows_out[r] = v
    cute.arch.sync_threads()


@cute.jit
def lm_prefetch_chunks(
    cta: cutlass.Int32, w_lm: cute.Tensor, chunks: cutlass.Constexpr = LM_PF_CHUNKS
):
    tid, _, _ = cute.arch.thread_idx()
    if tid < 3 * 16:
        row = (cta * lm.TILES_PER_CTA + tid // 16) * 16 + tid % 16
        if row < VOCAB_SHARD:
            for c in cutlass.range_constexpr(1, 1 + chunks):
                if cutlass.const_expr(c < lm.CHUNKS):
                    prefetch_bulk_l2(
                        w_lm.iterator + row * HIDDEN + c * lm.CHUNK_K,
                        cutlass.Uint32(lm.ROW_BYTES),
                    )


@cute.jit
def row_hop(
    second: cutlass.Constexpr,
    sel: cutlass.Int32,
    hop1: cute.Tensor,
    hop2: cute.Tensor,
    previous_hop2: cute.Tensor,
    residual_input: cute.Tensor,
    gamma: cute.Tensor,
    keep: cute.Tensor,
    scalars: cute.Tensor,
    out_main: cute.Tensor,
    rout: cute.Tensor,
    done: cute.Tensor,
):
    tid, _, _ = cute.arch.thread_idx()
    warp, lane = tid >> 5, tid & 31
    xs = cute.make_rmem_tensor((24,), cutlass.Float32)
    ssq = cutlass.Float32(0)
    for j in cutlass.range_constexpr(3):
        col = 2048 * j + 8 * tid
        if cutlass.const_expr(second):
            self_clear(hop1, previous_hop2, sel, col)
            sums = reduce_rank16(hop2, sel, col, False)
            if cutlass.const_expr(H2_CLEAR_CURRENT):
                for r in cutlass.range_constexpr(RANKS):
                    store_empty(
                        hop2.iterator + (sel * RANKS + r) * 3072 + col // 2,
                        evict_first(),
                    )
        else:
            sums = reduce_rank16(hop1, sel, col, True)
        prev = cute.make_rmem_tensor((4,), cutlass.Uint32)
        if cutlass.const_expr(second):
            for q in cutlass.range_constexpr(4):
                prev[q] = f32_bits(keep[(j * 256 + tid) * 4 + q])
        else:
            r0, r1 = global_pair(residual_input.iterator + sel * 3072 + col // 2)
            r2, r3 = global_pair(residual_input.iterator + sel * 3072 + col // 2 + 2)
            prev[0], prev[1], prev[2], prev[3] = r0, r1, r2, r3
        for q in cutlass.range_constexpr(4):
            xs[8 * j + 2 * q] = round_bf16(
                round_bf16(sums[2 * q]) + bf16_from_word(prev[q], False)
            )
            xs[8 * j + 2 * q + 1] = round_bf16(
                round_bf16(sums[2 * q + 1]) + bf16_from_word(prev[q], True)
            )
        for q in cutlass.range_constexpr(8):
            ssq += round_bf16(xs[8 * j + q] * xs[8 * j + q])
        words = cute.make_rmem_tensor((4,), cutlass.Uint32)
        for q in cutlass.range_constexpr(4):
            words[q] = bf16_pair(xs[8 * j + 2 * q], xs[8 * j + 2 * q + 1])
        if cutlass.const_expr(second):
            store_vector(
                rout.iterator + sel * 3072 + col // 2,
                words[0],
                words[1],
                words[2],
                words[3],
            )
        else:
            for q in cutlass.range_constexpr(4):
                keep[(j * 256 + tid) * 4 + q] = bits_f32(words[q])
    for sh in cutlass.range_constexpr(5):
        ssq += cute.arch.shuffle_sync_bfly(ssq, 16 >> sh)
    if lane == 0:
        scalars[warp] = ssq
    cute.arch.sync_threads()
    total = cutlass.Float32(0)
    for w in cutlass.range_constexpr(8):
        total += scalars[w]
    rstd = cute.math.rsqrt(total / 6144.0 + 1.0e-5, fastmath=True)
    for j in cutlass.range_constexpr(3):
        col = 2048 * j + 8 * tid
        g0, g1, g2, g3 = readonly_vector(gamma.iterator + col // 2)
        gw = (g0, g1, g2, g3)
        out = cute.make_rmem_tensor((4,), cutlass.Uint32)
        for q in cutlass.range_constexpr(4):
            lo = round_bf16((xs[8 * j + 2 * q] * rstd) * bf16_from_word(gw[q], False))
            hi = round_bf16(
                (xs[8 * j + 2 * q + 1] * rstd) * bf16_from_word(gw[q], True)
            )
            out[q] = bf16_pair(lo, hi)
            if cutlass.const_expr(not second):
                out[q] = published_word(out[q])
        store_vector(
            out_main.iterator + sel * 3072 + col // 2, out[0], out[1], out[2], out[3]
        )
    cute.arch.sync_threads()
    if tid == 0:
        for _ in cutlass.range_constexpr(4 if second else 1):
            release_increment(done.iterator)


def hop_warm(passes, scratch, scratch_ssq, second=False):
    if WARM_HOP and (WARM_H2 or not second):
        if not (HOP_LEAN and SHARED_IN_FC2):
            raise TypeError("WARM_HOP needs HOP_LEAN and SHARED_IN_FC2")
        return dict(warm=(passes, scratch, scratch_ssq))
    return {}


def o_warm(passes):
    return dict(passes=passes) if WARM_O else {}


@cute.jit
def prefetch_slice(
    src: cute.Tensor, total_bytes: cutlass.Constexpr, cta: cutlass.Int32
):
    tid, _, _ = cute.arch.thread_idx()
    chunk = ((total_bytes // 212 + 15) // 16) * 16
    lo = cta * chunk
    if (tid == 0) & (lo < total_bytes):
        n = cutlass.Int32(chunk)
        if lo + n > total_bytes:
            n = cutlass.Int32(total_bytes) - lo
        prefetch_bulk_l2(src.iterator + lo // 2, n.to(cutlass.Uint32))


@cute.jit
def head_words(gkey: cutlass.Uint64, gidx: cutlass.Int32):
    hi = (gkey >> cutlass.Uint64(32)).to(cutlass.Uint32)
    return (
        (gkey & cutlass.Uint64(0xFFFFFFFF)).to(cutlass.Uint32),
        (hi & cutlass.Uint32(0xFFFF)) | cutlass.Uint32(0x10000),
        gidx.to(cutlass.Uint32),
        (hi >> cutlass.Uint32(16)) | cutlass.Uint32(0x10000),
    )


@cute.jit
def head_key(w0: cutlass.Uint32, w1: cutlass.Uint32, w3: cutlass.Uint32):
    hi = ((w3 & cutlass.Uint32(0xFFFF)) << cutlass.Uint32(16)) | (
        w1 & cutlass.Uint32(0xFFFF)
    )
    return (hi.to(cutlass.Uint64) << cutlass.Uint64(32)) | w0.to(cutlass.Uint64)


@cute.jit
def wait_count(counter: cute.Tensor, target: cutlass.Uint32):
    tid, _, _ = cute.arch.thread_idx()
    if tid == 0:
        value = acquire_word(counter.iterator)
        while value < target:
            value = acquire_word(counter.iterator)
    cute.arch.sync_threads()


@cute.jit
def poll_vector(pointer: cute.Pointer):
    a, b, c, d = volatile_vector(pointer)
    waiting = (
        (a == cutlass.Uint32(EMPTY))
        | (b == cutlass.Uint32(EMPTY))
        | (c == cutlass.Uint32(EMPTY))
        | (d == cutlass.Uint32(EMPTY))
    )
    while waiting:
        a, b, c, d = volatile_vector(pointer)
        waiting = (
            (a == cutlass.Uint32(EMPTY))
            | (b == cutlass.Uint32(EMPTY))
            | (c == cutlass.Uint32(EMPTY))
            | (d == cutlass.Uint32(EMPTY))
        )
    return a, b, c, d


@cute.jit
def global_key(key: cutlass.Uint64, rank: cutlass.Int32):
    gidx = rank * VOCAB_SHARD + lm.key_index(key)
    low = cutlass.Uint32(0xFFFFFFFF) - gidx.to(cutlass.Uint32)
    return (key & cutlass.Uint64(0xFFFFFFFF00000000)) | low.to(cutlass.Uint64), gidx


@cute.jit
def draft_front_unit(
    unit: cutlass.Int32,
    packets: cute.Tensor,
    gamma_in: cute.Tensor,
    residual0: cute.Tensor,
    x_frag: cute.Tensor,
    x_ssq: cute.Tensor,
    scalars: cute.Tensor,
    front_done: cute.Tensor,
    draft_tokens: cute.Tensor,
    step: cutlass.Int32,
    xn=None,
    ns=None,
):
    tid, _, _ = cute.arch.thread_idx()
    token, quarter = unit // 4, unit % 4
    col = 1536 * quarter + 8 * tid
    live = token == TOKEN
    xw = cute.make_rmem_tensor((4,), cutlass.Uint32)
    for j in cutlass.range_constexpr(4):
        xw[j] = cutlass.Uint32(0)
    if live:
        if tid < RANKS:
            w0, w1, gidx, w3 = poll_vector(packets.iterator + tid * PK_SLOT + PK_HEAD)
            key = head_key(w0, w1, w3)
            scalars[16 + 2 * tid] = bits_f32(
                (key & cutlass.Uint64(0xFFFFFFFF)).to(cutlass.Uint32)
            )
            scalars[17 + 2 * tid] = bits_f32(
                (key >> cutlass.Uint64(32)).to(cutlass.Uint32)
            )
            scalars[48 + tid] = bits_f32(gidx)
        cute.arch.sync_threads()
        if tid == 0:
            best = cutlass.Uint64(0)
            winner = cutlass.Int32(0)
            for r in cutlass.range_constexpr(RANKS):
                key = (
                    f32_bits(scalars[17 + 2 * r]).to(cutlass.Uint64)
                    << cutlass.Uint64(32)
                ) | f32_bits(scalars[16 + 2 * r]).to(cutlass.Uint64)
                if key > best:
                    best = key
                    winner = cutlass.Int32(r)
            scalars[64] = bits_f32(winner.to(cutlass.Uint32))
            if quarter == 0:
                draft_tokens[step - 1] = f32_bits(scalars[48 + winner]).to(
                    cutlass.Int32
                )
        cute.arch.sync_threads()
        winner = f32_bits(scalars[64]).to(cutlass.Int32)
        if tid < 192:
            rows = cute.make_rmem_tensor((RANKS, 4), cutlass.Uint32)
            for r in cutlass.range_constexpr(RANKS):
                a, b, c, d = volatile_vector(
                    packets.iterator + r * PK_SLOT + PK_ROW + col // 2
                )
                rows[r, 0], rows[r, 1], rows[r, 2], rows[r, 3] = a, b, c, d
            waiting = cutlass.Boolean(True)
            while waiting:
                full = cutlass.Boolean(True)
                for r in cutlass.range_constexpr(RANKS):
                    for j in cutlass.range_constexpr(4):
                        full = full & (rows[r, j] != cutlass.Uint32(EMPTY))
                if full:
                    waiting = cutlass.Boolean(False)
                else:
                    for r in cutlass.range_constexpr(RANKS):
                        ready = cutlass.Boolean(True)
                        for j in cutlass.range_constexpr(4):
                            ready = ready & (rows[r, j] != cutlass.Uint32(EMPTY))
                        if not ready:
                            a, b, c, d = volatile_vector(
                                packets.iterator + r * PK_SLOT + PK_ROW + col // 2
                            )
                            rows[r, 0], rows[r, 1], rows[r, 2], rows[r, 3] = a, b, c, d
            qw = cute.make_rmem_tensor((4,), cutlass.Uint32)
            for j in cutlass.range_constexpr(4):
                qw[j] = rows[0, j]
            for r in cutlass.range_constexpr(1, RANKS):
                if winner == r:
                    for j in cutlass.range_constexpr(4):
                        qw[j] = rows[r, j]
            owner = col // SHARE
            offset = col - owner * SHARE
            h0, h1, h2, h3 = poll_vector(
                packets.iterator + owner * PK_SLOT + PK_HID + offset
            )
            h4, h5, h6, h7 = poll_vector(
                packets.iterator + owner * PK_SLOT + PK_HID + offset + 4
            )
            hw = (h0, h1, h2, h3, h4, h5, h6, h7)
            for j in cutlass.range_constexpr(4):
                xw[j] = bf16_pair(
                    add_rn(bf16_from_word(qw[j], False), bits_f32(hw[2 * j])),
                    add_rn(bf16_from_word(qw[j], True), bits_f32(hw[2 * j + 1])),
                )
    if cutlass.const_expr(xn is not None):
        xv = cute.make_rmem_tensor((8,), cutlass.Float32)
        for j in cutlass.range_constexpr(4):
            xv[2 * j] = bf16_from_word(xw[j], False)
            xv[2 * j + 1] = bf16_from_word(xw[j], True)
        if tid < 192:
            store_vector(
                residual0.iterator + token * 3072 + col // 2, xw[0], xw[1], xw[2], xw[3]
            )
        s0.xnorm_publish(unit, xv, gamma_in, xn, ns, scalars, False)
        if tid == 0:
            if live:
                release_increment(front_done.iterator)
        cute.arch.sync_threads()
    else:
        ssq = cutlass.Float32(0)
        if tid < 192:
            store_vector(
                residual0.iterator + token * 3072 + col // 2, xw[0], xw[1], xw[2], xw[3]
            )
            g0, g1, g2, g3 = readonly_vector(gamma_in.iterator + col // 2)
            gw = (g0, g1, g2, g3)
            prime = cute.make_rmem_tensor((4,), cutlass.Uint32)
            for j in cutlass.range_constexpr(4):
                lo = bf16_from_word(xw[j], False)
                hi = bf16_from_word(xw[j], True)
                ssq += round_bf16(lo * lo)
                ssq += round_bf16(hi * hi)
                prime[j] = published_word(
                    bf16_pair(
                        round_bf16(lo * bf16_from_word(gw[j], False)),
                        round_bf16(hi * bf16_from_word(gw[j], True)),
                    )
                )
            store_vector(
                x_frag.iterator + xfrag_word_offset(token, col),
                prime[0],
                prime[1],
                prime[2],
                prime[3],
            )
        for sh in cutlass.range_constexpr(5):
            ssq += cute.arch.shuffle_sync_bfly(ssq, 16 >> sh)
        if (tid & 31) == 0:
            scalars[tid >> 5] = ssq
        cute.arch.sync_threads()
        if tid == 0:
            partial = cutlass.Float32(0)
            for w in cutlass.range_constexpr(8):
                partial += scalars[w]
            store_word(
                x_ssq.iterator + 4 * token + quarter, published_word(f32_bits(partial))
            )
            if live:
                release_increment(front_done.iterator)
        cute.arch.sync_threads()


@cute.jit
def clear_packets(index: cutlass.Int32, packets: cute.Tensor, front_done: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    wait_count(front_done, cutlass.Uint32(4))
    policy = evict_first()
    for s in cutlass.range_constexpr(2):
        word = 4 * (index * 256 + tid + s * 28 * 256)
        if word < PK_COPY:
            store_empty(packets.iterator + word, policy)


@cute.jit
def shared_down(
    index: cutlass.Int32,
    act: cute.Tensor,
    w_sdn: cute.Tensor,
    region: cute.Tensor,
    red: cute.Tensor,
    out: cute.Tensor,
    out_off: cutlass.Int32,
    bar: cute.Pointer,
    clear_counter=None,
):
    tid, _, _ = cute.arch.thread_idx()
    warp, lane = tid >> 5, tid & 31
    r0 = 2 * ((index * (HIDDEN // 2)) // SD_CTAS)
    r1 = 2 * (((index + 1) * (HIDDEN // 2)) // SD_CTAS)
    base = region.iterator.toint().to(cutlass.Uint32)
    if tid == 0:
        cute.arch.mbarrier_arrive_and_expect_tx(bar, (r1 - r0) * moe.INTER * 2)
        bulk_copy(
            base,
            w_sdn.iterator + r0 * moe.INTER,
            ((r1 - r0) * moe.INTER * 2).to(cutlass.Uint32),
            bar,
            evict_first(),
        )
    a0, a1, a2, a3 = moe.poll_words(act, moe.TOPK * moe.INTER + 4 * lane)
    av = (
        bits_f32(a0 & cutlass.Uint32(0xFFFF0000)),
        bits_f32(a1 & cutlass.Uint32(0xFFFF0000)),
        bits_f32(a2 & cutlass.Uint32(0xFFFF0000)),
        bits_f32(a3 & cutlass.Uint32(0xFFFF0000)),
    )
    cute.arch.mbarrier_wait(bar, 0)
    for r in cutlass.range(r0 + warp, r1, 8, unroll=1):
        lo, hi = lds64(
            base + (((r - r0) * moe.INTER + 4 * lane) * 2).to(cutlass.Uint32)
        )
        v = bf16_from_word(lo, False) * av[0]
        v = fma_rn(bf16_from_word(lo, True), av[1], v)
        v = fma_rn(bf16_from_word(hi, False), av[2], v)
        v = fma_rn(bf16_from_word(hi, True), av[3], v)
        for sh in cutlass.range_constexpr(5):
            v = add_rn(v, cute.arch.shuffle_sync_bfly(v, 16 >> sh))
        if lane == 0:
            red[r - r0] = v
    cute.arch.sync_threads()
    if cutlass.const_expr(clear_counter is not None):
        if tid == 0:
            ready = acquire_word(clear_counter.iterator)
            while ready < cutlass.Uint32(COUNTED_CLEARS):
                ready = acquire_word(clear_counter.iterator)
        cute.arch.sync_threads()
    for p in cutlass.range(tid, (r1 - r0) // 2, 256, unroll=1):
        store_word(
            out.iterator + out_off + r0 // 2 + p,
            published_word(bf16_pair(red[2 * p], red[2 * p + 1])),
        )
    cute.arch.sync_threads()


@cute.jit
def issue_hidden_rows(
    index: cutlass.Int32, rows: cute.Tensor, region: cute.Tensor, bars: cute.Tensor
):
    tid, _, _ = cute.arch.thread_idx()
    r0 = (index * SHARE) // HH_CTAS
    r1 = ((index + 1) * SHARE) // HH_CTAS
    base = region.iterator.toint().to(cutlass.Uint32)
    if tid == 0:
        for s in cutlass.range_constexpr(HH_SLOTS):
            if r0 + s < r1:
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bars.iterator + s, moe.ROW_BYTES
                )
                bulk_copy(
                    base + cutlass.Uint32(s * moe.ROW_BYTES),
                    rows.iterator + (r0 + s) * HIDDEN,
                    cutlass.Uint32(moe.ROW_BYTES),
                    bars.iterator + s,
                    evict_first(),
                )


@cute.jit
def hidden_half(
    index: cutlass.Int32,
    xn: cute.Tensor,
    gamma_h: cute.Tensor,
    rows: cute.Tensor,
    region: cute.Tensor,
    red: cute.Tensor,
    bars: cute.Tensor,
    dst: cute.Tensor,
):
    tid, _, _ = cute.arch.thread_idx()
    warp, lane = tid >> 5, tid & 31
    r0 = (index * SHARE) // HH_CTAS
    r1 = ((index + 1) * SHARE) // HH_CTAS
    hv = cute.make_rmem_tensor((24,), cutlass.Float32)
    ssq = cutlass.Float32(0)
    for i in cutlass.range_constexpr(3):
        a, b, c, d = cached_vector(xn.iterator + 12 * tid + 4 * i)
        w = (a, b, c, d)
        for q in cutlass.range_constexpr(4):
            hv[8 * i + 2 * q] = bf16_from_word(w[q], False)
            hv[8 * i + 2 * q + 1] = bf16_from_word(w[q], True)
    for e in cutlass.range_constexpr(24):
        ssq = fma_rn(hv[e], hv[e], ssq)
    for sh in cutlass.range_constexpr(5):
        ssq += cute.arch.shuffle_sync_bfly(ssq, 16 >> sh)
    if lane == 0:
        red[2048 + warp] = ssq
    cute.arch.sync_threads()
    total = cutlass.Float32(0)
    for w in cutlass.range_constexpr(8):
        total += red[2048 + w]
    rstd = cute.math.rsqrt(total / 6144.0 + 1.0e-5, fastmath=True)
    for i in cutlass.range_constexpr(3):
        g0, g1, g2, g3 = readonly_vector(gamma_h.iterator + 12 * tid + 4 * i)
        g = (g0, g1, g2, g3)
        for q in cutlass.range_constexpr(4):
            hv[8 * i + 2 * q] = round_bf16(
                (hv[8 * i + 2 * q] * rstd) * bf16_from_word(g[q], False)
            )
            hv[8 * i + 2 * q + 1] = round_bf16(
                (hv[8 * i + 2 * q + 1] * rstd) * bf16_from_word(g[q], True)
            )
    base = region.iterator.toint().to(cutlass.Uint32)
    for r in cutlass.range(r0, r1, unroll=1):
        s = (r - r0) % HH_SLOTS
        cute.arch.mbarrier_wait(bars.iterator + s, ((r - r0) // HH_SLOTS) % 2)
        acc = cutlass.Float32(0)
        for i in cutlass.range_constexpr(3):
            w0, w1, w2, w3 = moe.lds128(
                base + (s * moe.ROW_BYTES + 48 * tid + 16 * i).to(cutlass.Uint32)
            )
            ww = (w0, w1, w2, w3)
            for q in cutlass.range_constexpr(4):
                acc = fma_rn(bf16_from_word(ww[q], False), hv[8 * i + 2 * q], acc)
                acc = fma_rn(bf16_from_word(ww[q], True), hv[8 * i + 2 * q + 1], acc)
        for sh in cutlass.range_constexpr(5):
            acc = acc + cute.arch.shuffle_sync_bfly(acc, 16 >> sh)
        if lane == 0:
            red[warp] = acc
        cute.arch.sync_threads()
        if tid == 0:
            v = cutlass.Float32(0)
            for w in cutlass.range_constexpr(8):
                v += red[w]
            store_word(dst.iterator + r, published_word(f32_bits(v)))
            if r + HH_SLOTS < r1:
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bars.iterator + s, moe.ROW_BYTES
                )
                bulk_copy(
                    base + (s * moe.ROW_BYTES).to(cutlass.Uint32),
                    rows.iterator + (r + HH_SLOTS) * HIDDEN,
                    cutlass.Uint32(moe.ROW_BYTES),
                    bars.iterator + s,
                    evict_first(),
                )
        cute.arch.sync_threads()


@cute.jit
def publish_packet(
    rank: cutlass.Int32,
    step: cutlass.Int32,
    last: cutlass.Boolean,
    keys: cute.Tensor,
    arrivals: cute.Tensor,
    token_table: cute.Tensor,
    slot_mc: cute.Tensor,
    local_copy: cute.Tensor,
    scalars: cute.Tensor,
    draft_tokens: cute.Tensor,
    next_new_tokens: cute.Tensor,
    accepted: cute.Tensor,
    num_accepted: cute.Tensor,
):
    tid, _, _ = cute.arch.thread_idx()
    wait_count(arrivals, cutlass.Uint32(lm.LM_CTAS))
    if tid == 0:
        k0, k1, _a, _b = volatile_vector(keys.iterator)
        scalars[16] = bits_f32(k0)
        scalars[17] = bits_f32(k1)
    cute.arch.sync_threads()
    key = (f32_bits(scalars[17]).to(cutlass.Uint64) << cutlass.Uint64(32)) | f32_bits(
        scalars[16]
    ).to(cutlass.Uint64)
    gkey, gidx = global_key(key, rank)
    local = gidx - rank * VOCAB_SHARD
    if not last:
        for i in cutlass.range_constexpr(3):
            word = 12 * tid + 4 * i
            a, b, c, d = readonly_vector(
                token_table.iterator + local * (HIDDEN // 2) + word
            )
            store_vector(
                slot_mc.iterator + PK_ROW + word,
                published_word(a),
                published_word(b),
                published_word(c),
                published_word(d),
            )
    if tid == 0:
        h0, h1, h2, h3 = head_words(gkey, gidx)
        store_vector(slot_mc.iterator + PK_HEAD, h0, h1, h2, h3)
    if last:
        if tid < RANKS:
            w0, w1, g, w3 = poll_vector(local_copy.iterator + tid * PK_SLOT + PK_HEAD)
            k = head_key(w0, w1, w3)
            scalars[32 + 2 * tid] = bits_f32(
                (k & cutlass.Uint64(0xFFFFFFFF)).to(cutlass.Uint32)
            )
            scalars[33 + 2 * tid] = bits_f32(
                (k >> cutlass.Uint64(32)).to(cutlass.Uint32)
            )
            scalars[64 + tid] = bits_f32(g)
        cute.arch.sync_threads()
        if tid < RANKS:
            store_empty(local_copy.iterator + tid * PK_SLOT + PK_HEAD, evict_first())
        if tid == 0:
            best = cutlass.Uint64(0)
            winner = cutlass.Int32(0)
            for r in cutlass.range_constexpr(RANKS):
                k = (
                    f32_bits(scalars[33 + 2 * r]).to(cutlass.Uint64)
                    << cutlass.Uint64(32)
                ) | f32_bits(scalars[32 + 2 * r]).to(cutlass.Uint64)
                if k > best:
                    best = k
                    winner = cutlass.Int32(r)
            token = f32_bits(scalars[64 + winner]).to(cutlass.Int32)
            draft_tokens[6] = token
            next_new_tokens[0] = accepted[num_accepted[0] - 1]
            for j in cutlass.range_constexpr(6):
                next_new_tokens[1 + j] = draft_tokens[j]
            next_new_tokens[7] = token
        cute.arch.sync_threads()


@cute.jit
def target_tail(
    rank: cutlass.Int32,
    x8: cute.Tensor,
    w_lm: cute.Tensor,
    region: cute.Tensor,
    red: cute.Tensor,
    lm_bars: cute.Tensor,
    keys: cute.Tensor,
    key_words: cute.Tensor,
    arrivals: cute.Tensor,
    heads_mc: cute.Tensor,
    heads_uc: cute.Tensor,
    draft_in: cute.Tensor,
    accepted_out: cute.Tensor,
    num_accepted_out: cute.Tensor,
    next_arena: cute.Tensor,
    next_tk: cute.Tensor,
    clear_counter: cute.Tensor,
    next_counters: cute.Tensor,
):
    tid, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    cute.arch.griddepcontrol_wait()
    cute.arch.griddepcontrol_launch_dependents()
    reset_next_counters(next_counters)
    if cta < lm.LM_CTAS:
        lm.issue_lm_prologue(cta, w_lm, region, lm_bars)
        fence_proxy_global()
        lm.issue_lm_x(x8, cutlass.Int32(8), region, lm_bars)
        lm.lm_head_stage(
            cta,
            x8,
            w_lm,
            region,
            red,
            lm_bars,
            keys,
            arrivals,
            cutlass.Int32(8),
            red,
            False,
            False,
        )
    clear_assigned(next_arena, next_tk, clear_counter)
    if cta == lm.LM_CTAS - 1:
        wait_count(arrivals, cutlass.Uint32(lm.LM_CTAS))
        if tid < 8:
            a, b, c, d = volatile_vector(key_words.iterator + 4 * (tid // 2))
            k0, k1 = a, b
            if tid % 2 == 1:
                k0, k1 = c, d
            key = (k1.to(cutlass.Uint64) << cutlass.Uint64(32)) | k0.to(cutlass.Uint64)
            gkey, gidx = global_key(key, rank)
            h0, h1, h2, h3 = head_words(gkey, gidx)
            store_vector(heads_mc.iterator + (tid * RANKS + rank) * 4, h0, h1, h2, h3)
        if tid < 8 * RANKS:
            w0, w1, g, w3 = poll_vector(heads_uc.iterator + 4 * tid)
            k = head_key(w0, w1, w3)
            red[3 * tid] = bits_f32((k & cutlass.Uint64(0xFFFFFFFF)).to(cutlass.Uint32))
            red[3 * tid + 1] = bits_f32((k >> cutlass.Uint64(32)).to(cutlass.Uint32))
            red[3 * tid + 2] = bits_f32(g)
        cute.arch.sync_threads()
        if tid < 8 * RANKS:
            store_empty(heads_uc.iterator + 4 * tid, evict_first())
        if tid < 8:
            best = cutlass.Uint64(0)
            token = cutlass.Uint32(0)
            for r in cutlass.range_constexpr(RANKS):
                j = tid * RANKS + r
                kk = (
                    f32_bits(red[3 * j + 1]).to(cutlass.Uint64) << cutlass.Uint64(32)
                ) | f32_bits(red[3 * j]).to(cutlass.Uint64)
                if kk > best:
                    best = kk
                    token = f32_bits(red[3 * j + 2])
            red[1024 + tid] = bits_f32(token)
        cute.arch.sync_threads()
        if tid == 0:
            accepted = cutlass.Int32(1)
            going = cutlass.Boolean(True)
            for j in cutlass.range_constexpr(7):
                if going:
                    if draft_in[j] == f32_bits(red[1024 + j]).to(cutlass.Int32):
                        accepted = accepted + 1
                    else:
                        going = cutlass.Boolean(False)
            num_accepted_out[0] = accepted
            for j in cutlass.range_constexpr(8):
                accepted_out[j] = f32_bits(red[1024 + j]).to(cutlass.Int32)
        cute.arch.sync_threads()


@cute.jit
def ts_bridge_reset(
    tgt_counters: cute.Tensor, cnt_counters: cute.Tensor, flag: cute.Tensor
):
    tid, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    if cta < TS_BRIDGE_ZERO_CTAS:
        index = cta * 256 + tid
        if index < COUNTER_WORDS:
            tgt_counters[index] = cutlass.Uint32(0)
        elif index < 2 * COUNTER_WORDS:
            cnt_counters[index - COUNTER_WORDS] = cutlass.Uint32(0)
        cute.arch.sync_threads()
        if tid == 0:
            lm.release_add_u32(flag.iterator, cutlass.Uint32(1))


@cute.jit
def ts_bridge_clear(
    next_arena: cute.Tensor,
    next_topk_exchange: cute.Tensor,
    current_clear_counter: cute.Tensor,
    flag: cute.Tensor,
):
    tid, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    clears = (cta < 151) | ((cta >= 192) & (cta < 196))
    if cta < 151:
        policy = evict_first()
        for step in cutlass.range_constexpr(4):
            word = cta * 4096 + 4 * tid + step * 1024
            if word < ARENA_WORDS:
                store_empty(next_arena.iterator + word, policy)
    if (cta >= 192) & (cta < 196):
        policy = evict_first()
        for step in cutlass.range_constexpr(4):
            word = (cta - 192) * 4096 + 4 * tid + step * 1024
            store_empty(next_topk_exchange.iterator + word, policy)
    if clears:
        cute.arch.sync_threads()
        if tid == 0:
            value = acquire_word(flag.iterator)
            while value < cutlass.Uint32(TS_BRIDGE_ZERO_CTAS):
                value = acquire_word(flag.iterator)
            release_increment(current_clear_counter.iterator)


@cute.jit
def target_argmax(
    rank: cutlass.Int32,
    logits: cute.Tensor,
    red: cute.Tensor,
    keys: cute.Tensor,
    key_words: cute.Tensor,
    arrivals: cute.Tensor,
    heads_mc: cute.Tensor,
    heads_uc: cute.Tensor,
    draft_in: cute.Tensor,
    accepted_out: cute.Tensor,
    num_accepted_out: cute.Tensor,
    argmax_out: cute.Tensor,
    next_arena: cute.Tensor,
    next_tk: cute.Tensor,
    clear_counter: cute.Tensor,
    next_counters: cute.Tensor,
    bridge=None,
):
    tid, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    cute.arch.griddepcontrol_wait()
    cute.arch.griddepcontrol_launch_dependents()
    if cutlass.const_expr(bridge is None):
        reset_next_counters(next_counters)
    else:
        ts_bridge_reset(bridge[0], bridge[1], bridge[2])
    cols = VOCAB_SHARD // TA_CTAS
    if cta < TA_CTAS:
        if tid < 16:
            red[tid] = bits_f32(cutlass.Uint32(0))
        cute.arch.sync_threads()
        kaddr = red.iterator.toint().to(cutlass.Uint32)
        for j in cutlass.range_constexpr((8 * (VOCAB_SHARD // TA_CTAS) + 255) // 256):
            e = tid + 256 * j
            if e < 8 * cols:
                row = e // cols
                col = cta * cols + e % cols
                bits = logits[row * VOCAB_SHARD + col].to(
                    cutlass.Uint32
                ) << cutlass.Uint32(16)
                lm.shared_max_u64(
                    kaddr + (row * 8).to(cutlass.Uint32), lm.argmax_key(bits, col)
                )
        cute.arch.sync_threads()
        if tid == 0:
            for r in cutlass.range_constexpr(8):
                k = (
                    f32_bits(red[2 * r + 1]).to(cutlass.Uint64) << cutlass.Uint64(32)
                ) | f32_bits(red[2 * r]).to(cutlass.Uint64)
                lm.atom_max_u64(keys.iterator + r * lm.KEY_STRIDE, k)
            lm.release_add_u32(arrivals.iterator, cutlass.Uint32(1))
    if cutlass.const_expr(bridge is None):
        clear_assigned(next_arena, next_tk, clear_counter)
    else:
        ts_bridge_clear(next_arena, next_tk, clear_counter, bridge[2])
    if cta == TA_CTAS - 1:
        wait_count(arrivals, cutlass.Uint32(TA_CTAS))
        if tid < 8:
            a, b, c, d = volatile_vector(key_words.iterator + 4 * (tid // 2))
            k0, k1 = a, b
            if tid % 2 == 1:
                k0, k1 = c, d
            key = (k1.to(cutlass.Uint64) << cutlass.Uint64(32)) | k0.to(cutlass.Uint64)
            gkey, gidx = global_key(key, rank)
            h0, h1, h2, h3 = head_words(gkey, gidx)
            store_vector(heads_mc.iterator + (tid * RANKS + rank) * 4, h0, h1, h2, h3)
        if tid < 8 * RANKS:
            w0, w1, g, w3 = poll_vector(heads_uc.iterator + 4 * tid)
            k = head_key(w0, w1, w3)
            red[64 + 3 * tid] = bits_f32(
                (k & cutlass.Uint64(0xFFFFFFFF)).to(cutlass.Uint32)
            )
            red[64 + 3 * tid + 1] = bits_f32(
                (k >> cutlass.Uint64(32)).to(cutlass.Uint32)
            )
            red[64 + 3 * tid + 2] = bits_f32(g)
        cute.arch.sync_threads()
        if tid < 8 * RANKS:
            store_empty(heads_uc.iterator + 4 * tid, evict_first())
        if tid < 8:
            best = cutlass.Uint64(0)
            token = cutlass.Uint32(0)
            for r in cutlass.range_constexpr(RANKS):
                jj = tid * RANKS + r
                kk = (
                    f32_bits(red[64 + 3 * jj + 1]).to(cutlass.Uint64)
                    << cutlass.Uint64(32)
                ) | f32_bits(red[64 + 3 * jj]).to(cutlass.Uint64)
                if kk > best:
                    best = kk
                    token = f32_bits(red[64 + 3 * jj + 2])
            red[1024 + tid] = bits_f32(token)
        cute.arch.sync_threads()
        if tid == 0:
            accepted = cutlass.Int32(1)
            going = cutlass.Boolean(True)
            for j in cutlass.range_constexpr(7):
                if going:
                    if draft_in[j] == f32_bits(red[1024 + j]).to(cutlass.Int32):
                        accepted = accepted + 1
                    else:
                        going = cutlass.Boolean(False)
            num_accepted_out[0] = accepted
            for j in cutlass.range_constexpr(8):
                tok_j = f32_bits(red[1024 + j]).to(cutlass.Int32)
                if cutlass.const_expr(TS_PREDICT_ZERO):
                    if j >= accepted:
                        tok_j = cutlass.Int32(0)
                accepted_out[j] = tok_j
                argmax_out[j] = f32_bits(red[1024 + j]).to(cutlass.Int32)
        cute.arch.sync_threads()


@dsl_user_op
def ts_ex2(x: cutlass.Float32, *, loc=None, ip=None):
    return cutlass.Float32(
        f32_intrinsic("llvm.nvvm.ex2.approx.ftz.f", x, loc=loc, ip=ip)
    )


@dsl_user_op
def ts_shared_add_u32(
    address: cutlass.Uint32, value: cutlass.Uint32, *, loc=None, ip=None
):
    llvm.atomicrmw(
        llvm.AtomicBinOp.add,
        as_ptr(3, address, loc, ip),
        value.ir_value(loc=loc, ip=ip),
        llvm.AtomicOrdering.monotonic,
        syncscope="block",
        loc=loc,
        ip=ip,
    )


TS_LFIX = 8388608.0


@cute.jit
def ts_lfix(e: cutlass.Float32):
    return (e * cutlass.Float32(TS_LFIX) + cutlass.Float32(0.5)).to(cutlass.Uint32)


@cute.jit
def ts_philox(
    c0: cutlass.Uint32, c1: cutlass.Uint32, k0: cutlass.Uint32, k1: cutlass.Uint32
):
    x0, x1, x2, x3 = c0, c1, cutlass.Uint32(0), cutlass.Uint32(0)
    a, b = k0, k1
    for _ in cutlass.range_constexpr(10):
        p0 = x0.to(cutlass.Uint64) * cutlass.Uint64(0xD2511F53)
        p1 = x2.to(cutlass.Uint64) * cutlass.Uint64(0xCD9E8D57)
        hi0, lo0 = (p0 >> cutlass.Uint64(32)).to(cutlass.Uint32), p0.to(cutlass.Uint32)
        hi1, lo1 = (p1 >> cutlass.Uint64(32)).to(cutlass.Uint32), p1.to(cutlass.Uint32)
        x0, x1, x2, x3 = hi1 ^ x1 ^ a, lo1, hi0 ^ x3 ^ b, lo0
        a = a + cutlass.Uint32(0x9E3779B9)
        b = b + cutlass.Uint32(0xBB67AE85)
    return x0, x1, x2, x3


@cute.jit
def ts_uniform(x: cutlass.Uint32):
    return (x >> cutlass.Uint32(8)).to(cutlass.Float32) * cutlass.Float32(
        1.0 / 16777216.0
    )


@cute.jit
def ts_key16(bits: cutlass.Uint32):
    k = bits | cutlass.Uint32(0x8000)
    if (bits & cutlass.Uint32(0x8000)) != cutlass.Uint32(0):
        k = (~bits) & cutlass.Uint32(0xFFFF)
    return k


@cute.jit
def ts_poll(pointer: cute.Pointer):
    w = volatile_word(pointer)
    while w == cutlass.Uint32(EMPTY):
        w = volatile_word(pointer)
    return w


@cute.jit
def ts_poll16(base: cute.Pointer, stride: cutlass.Constexpr, out: cute.Tensor):
    for r in cutlass.range_constexpr(RANKS):
        out[r] = volatile_word(base + r * stride)
    missing = cutlass.Boolean(True)
    while missing:
        missing = cutlass.Boolean(False)
        for r in cutlass.range_constexpr(RANKS):
            if out[r] == cutlass.Uint32(EMPTY):
                out[r] = volatile_word(base + r * stride)
                missing = cutlass.Boolean(True)


@cute.jit
def ts_block_reduce(
    v: cutlass.Float32,
    red: cute.Tensor,
    base: cutlass.Constexpr,
    is_max: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    warp, lane = tid >> 5, tid & 31
    x = v
    for step in cutlass.range_constexpr(5):
        o = cute.arch.shuffle_sync_bfly(x, 1 << step)
        if cutlass.const_expr(is_max):
            x = max_num(x, o)
        else:
            x = x + o
    if lane == 0:
        red[base + warp] = x
    cute.arch.sync_threads()
    r = red[base]
    for w in cutlass.range_constexpr(1, 8):
        if cutlass.const_expr(is_max):
            r = max_num(r, red[base + w])
        else:
            r = r + red[base + w]
    cute.arch.sync_threads()
    return r


@cute.jit
def ts_block_scan(v: cutlass.Float32, red: cute.Tensor, base: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    warp, lane = tid >> 5, tid & 31
    x = v
    for step in cutlass.range_constexpr(5):
        o = cute.arch.shuffle_sync_up(x, 1 << step)
        if lane >= (1 << step):
            x = x + o
    if lane == 31:
        red[base + warp] = x
    cute.arch.sync_threads()
    before = cutlass.Float32(0)
    for w in cutlass.range_constexpr(8):
        if w < warp:
            before = before + red[base + w]
    cute.arch.sync_threads()
    return before + x


@cute.jit
def ts_value(bits: cutlass.Uint32):
    return bits_f32(bits << cutlass.Uint32(16))


@cute.jit
def ts_vj(vals: cute.Tensor, rowbuf: cute.Tensor, index, j):
    if cutlass.const_expr(TS_ROLLED):
        return rowbuf[index].to(cutlass.Uint32)
    else:
        return vals[j]


@cute.jit
def target_sample(
    rank: cutlass.Int32,
    logits: cute.Tensor,
    params: cute.Tensor,
    red: cute.Tensor,
    hist: cute.Tensor,
    arrivals: cute.Tensor,
    r1_mc: cute.Tensor,
    r1_uc: cute.Tensor,
    r2_mc: cute.Tensor,
    r2_uc: cute.Tensor,
    res: cute.Tensor,
    draft_in: cute.Tensor,
    accepted_out: cute.Tensor,
    num_accepted_out: cute.Tensor,
    argmax_out: cute.Tensor,
    next_arena: cute.Tensor,
    next_tk: cute.Tensor,
    clear_counter: cute.Tensor,
    next_counters: cute.Tensor,
    bridge=None,
):
    tid, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    warp, lane = tid >> 5, tid & 31
    cute.arch.griddepcontrol_wait()
    if cutlass.const_expr(not TS_LATE_DEP):
        cute.arch.griddepcontrol_launch_dependents()
    if cutlass.const_expr(bridge is None):
        reset_next_counters(next_counters)
    else:
        ts_bridge_reset(bridge[0], bridge[1], bridge[2])
    if tid < 11:
        red[3008 + tid] = params[tid]
    elif tid < 18:
        red[3024 + tid - 11] = bits_f32(draft_in[tid - 11].to(cutlass.Uint32))
    cute.arch.sync_threads()
    phil = f32_bits(params[11]) != cutlass.Uint32(0)
    step_w = f32_bits(params[14])
    if phil & (tid == 0):
        k0, k1 = f32_bits(params[12]), f32_bits(params[13])
        for blk in cutlass.range_constexpr(3):
            y0, y1, y2, y3 = ts_philox(step_w, cutlass.Uint32(blk), k0, k1)
            ys = (y0, y1, y2, y3)
            for q in cutlass.range_constexpr(4):
                if 4 * blk + q < 9:
                    red[3008 + 4 * blk + q] = ts_uniform(ys[q])
    cute.arch.sync_threads()
    temperature = red[3008 + 9]
    top_p = red[3008 + 10]
    scale_c = cutlass.Float32(1.4426950408889634) / temperature
    per = (VOCAB_SHARD + 255) // 256
    raddr = red.iterator.toint().to(cutlass.Uint32)
    if cta < 8:
        t = cta
        haddr = hist.iterator.toint().to(cutlass.Uint32)
        for i in cutlass.range_constexpr((2 * TS_LANE_HIST + 255) // 256):
            if tid + 256 * i < 2 * TS_LANE_HIST:
                hist[tid + 256 * i] = cutlass.Float32(0)
        cute.arch.sync_threads()
        vals = cute.make_rmem_tensor((per,), cutlass.Uint32)
        rowbuf = cute.make_tensor(
            cute.recast_ptr(hist.iterator + 2 * TS_LANE_HIST, dtype=cutlass.Uint16),
            cute.make_layout((per * 256,)),
        )
        mx = bits_f32(cutlass.Uint32(0xFF800000))
        if cutlass.const_expr(TS_ROLLED):
            for i in range((per * 256) // 256):
                q = tid + 256 * i
                v16 = cutlass.Uint16(0xFF80)
                if q < VOCAB_SHARD:
                    v16 = logits[t * VOCAB_SHARD + q].to(cutlass.Uint16)
                rowbuf[q] = v16
            cute.arch.sync_threads()
            col = cutlass.Int32(0)
            for j in range(per):
                mx = max_num(mx, ts_value(rowbuf[per * tid + j].to(cutlass.Uint32)))
        else:
            for j in cutlass.range_constexpr(per):
                col = per * tid + j
                vals[j] = cutlass.Uint32(0xFF80)
                if col < VOCAB_SHARD:
                    vals[j] = logits[t * VOCAB_SHARD + col].to(cutlass.Uint32)
                mx = max_num(mx, ts_value(vals[j]))
        mx = ts_block_reduce(mx, red, 560, True)
        first = cutlass.Float32(1.0e9)
        if cutlass.const_expr(TS_ROLLED):
            col = cutlass.Int32(0)
            for j in range(per):
                if (per * tid + j < VOCAB_SHARD) & (first > cutlass.Float32(1.0e8)):
                    if ts_value(ts_vj(vals, rowbuf, per * tid + j, j)) == mx:
                        first = cutlass.Float32(per * tid + j)
        else:
            for j in cutlass.range_constexpr(per):
                if (per * tid + j < VOCAB_SHARD) & (first > cutlass.Float32(1.0e8)):
                    if ts_value(ts_vj(vals, rowbuf, per * tid + j, j)) == mx:
                        first = cutlass.Float32(per * tid + j)
        amx_col = -ts_block_reduce(-first, red, 560, True)
        total = cutlass.Float32(0)
        if cutlass.const_expr(TS_ROLLED):
            col = cutlass.Int32(0)
            for j in range(per):
                e = ts_ex2(
                    (ts_value(ts_vj(vals, rowbuf, per * tid + j, j)) - mx) * scale_c
                )
                total = total + e
                if e >= cutlass.Float32(TS_PRUNE):
                    ts_shared_add_u32(
                        haddr
                        + 4
                        * (
                            lane.to(cutlass.Uint32) * cutlass.Uint32(257)
                            + (
                                ts_key16(ts_vj(vals, rowbuf, per * tid + j, j))
                                >> cutlass.Uint32(8)
                            )
                        ),
                        ts_lfix(e),
                    )
        else:
            for j in cutlass.range_constexpr(per):
                e = ts_ex2(
                    (ts_value(ts_vj(vals, rowbuf, per * tid + j, j)) - mx) * scale_c
                )
                total = total + e
                if e >= cutlass.Float32(TS_PRUNE):
                    ts_shared_add_u32(
                        haddr
                        + 4
                        * (
                            lane.to(cutlass.Uint32) * cutlass.Uint32(257)
                            + (
                                ts_key16(ts_vj(vals, rowbuf, per * tid + j, j))
                                >> cutlass.Uint32(8)
                            )
                        ),
                        ts_lfix(e),
                    )
        total = ts_block_reduce(total, red, 560, False)
        hsum = cutlass.Uint64(0)
        for l in cutlass.range_constexpr(32):
            hsum = hsum + f32_bits(hist[l * 257 + tid]).to(cutlass.Uint64)
        red[tid] = hsum.to(cutlass.Float32) * cutlass.Float32(1.0 / TS_LFIX)
        cute.arch.sync_threads()
        if tid < TS_ROW // 4:
            w = cute.make_rmem_tensor((4,), cutlass.Uint32)
            for k in cutlass.range_constexpr(4):
                i = 4 * tid + k
                v = cutlass.Uint32(0x10000)
                if i == 0:
                    v = published_word(f32_bits(mx))
                elif i == 1:
                    v = published_word(f32_bits(total))
                elif i < 258:
                    v = published_word(f32_bits(red[i - 2]))
                elif i == 259:
                    v = amx_col.to(cutlass.Uint32)
                elif i == 258:
                    if t < 7:
                        d = f32_bits(red[3024 + t]).to(cutlass.Int32)
                        if d // VOCAB_SHARD == rank:
                            v = logits[t * VOCAB_SHARD + d - rank * VOCAB_SHARD].to(
                                cutlass.Uint32
                            ) | cutlass.Uint32(0x10000)
                w[k] = v
            store_vector(
                r1_mc.iterator + rank * PK_SLOT + t * TS_ROW + 4 * tid,
                w[0],
                w[1],
                w[2],
                w[3],
            )
        if tid < RANKS:
            a, b, _c, _d = poll_vector(r1_uc.iterator + tid * PK_SLOT + t * TS_ROW)
            red[512 + tid] = bits_f32(a)
            red[528 + tid] = bits_f32(b)
        cute.arch.sync_threads()
        gm = red[512]
        for r in cutlass.range_constexpr(1, RANKS):
            gm = max_num(gm, red[512 + r])
        z = cutlass.Float32(0)
        scales = cute.make_rmem_tensor((RANKS,), cutlass.Float32)
        for r in cutlass.range_constexpr(RANKS):
            scales[r] = ts_ex2((red[512 + r] - gm) * scale_c)
            z = z + red[528 + r] * scales[r]
        coarse = cute.make_rmem_tensor((RANKS,), cutlass.Float32)
        bin_c = 255 - tid
        csum = cutlass.Float32(0)
        cw = cute.make_rmem_tensor((RANKS,), cutlass.Uint32)
        ts_poll16(r1_uc.iterator + t * TS_ROW + 2 + bin_c, PK_SLOT, cw)
        for r in cutlass.range_constexpr(RANKS):
            coarse[r] = bits_f32(cw[r]) * scales[r]
            csum = csum + coarse[r]
        incl = ts_block_scan(csum, red, 576)
        if tid == 255:
            red[604] = incl
        cute.arch.sync_threads()
        hist_total = red[604]
        target = top_p * hist_total
        if target > hist_total:
            target = hist_total
        if (incl >= target) & (incl - csum < target):
            red[600] = bits_f32(bin_c.to(cutlass.Uint32))
            red[601] = incl - csum
        cute.arch.sync_threads()
        bstar = f32_bits(red[600])
        c_above = red[601]
        if tid == 0:
            owner_a = cutlass.Int32(-1)
            for r in cutlass.range_constexpr(RANKS):
                if (owner_a < 0) & (red[512 + r] == gm):
                    owner_a = cutlass.Int32(r)
            acol = ts_poll(r1_uc.iterator + owner_a * PK_SLOT + t * TS_ROW + 259)
            store_vector(
                res.iterator + t * TS_RES_ROW + 20,
                (owner_a * VOCAB_SHARD).to(cutlass.Uint32) + acol,
                cutlass.Uint32(0x10000),
                cutlass.Uint32(0x10000),
                cutlass.Uint32(0x10000),
            )
        draft_word = cutlass.Uint32(0x10000)
        if t < 7:
            draft_word = ts_poll(
                r1_uc.iterator
                + (f32_bits(red[3024 + t]).to(cutlass.Int32) // VOCAB_SHARD) * PK_SLOT
                + t * TS_ROW
                + 258
            )
        if cutlass.const_expr(TS_ROLLED):
            col = cutlass.Int32(0)
            for j in range(per):
                kk = ts_key16(ts_vj(vals, rowbuf, per * tid + j, j))
                if (kk >> cutlass.Uint32(8)) == bstar:
                    ts_shared_add_u32(
                        haddr
                        + 4
                        * (
                            TS_LANE_HIST
                            + lane.to(cutlass.Uint32) * cutlass.Uint32(257)
                            + (kk & cutlass.Uint32(255))
                        ),
                        ts_lfix(
                            ts_ex2(
                                (ts_value(ts_vj(vals, rowbuf, per * tid + j, j)) - gm)
                                * scale_c
                            )
                        ),
                    )
        else:
            for j in cutlass.range_constexpr(per):
                kk = ts_key16(ts_vj(vals, rowbuf, per * tid + j, j))
                if (kk >> cutlass.Uint32(8)) == bstar:
                    ts_shared_add_u32(
                        haddr
                        + 4
                        * (
                            TS_LANE_HIST
                            + lane.to(cutlass.Uint32) * cutlass.Uint32(257)
                            + (kk & cutlass.Uint32(255))
                        ),
                        ts_lfix(
                            ts_ex2(
                                (ts_value(ts_vj(vals, rowbuf, per * tid + j, j)) - gm)
                                * scale_c
                            )
                        ),
                    )
        cute.arch.sync_threads()
        hsum2 = cutlass.Uint64(0)
        for l in cutlass.range_constexpr(32):
            hsum2 = hsum2 + f32_bits(hist[TS_LANE_HIST + l * 257 + tid]).to(
                cutlass.Uint64
            )
        red[256 + tid] = hsum2.to(cutlass.Float32) * cutlass.Float32(1.0 / TS_LFIX)
        cute.arch.sync_threads()
        if tid < TS_ROW // 4:
            w2 = cute.make_rmem_tensor((4,), cutlass.Uint32)
            for k in cutlass.range_constexpr(4):
                i = 4 * tid + k
                v2 = cutlass.Uint32(0x10000)
                if i < 256:
                    v2 = published_word(f32_bits(red[256 + i]))
                w2[k] = v2
            store_vector(
                r2_mc.iterator + rank * PK_SLOT + t * TS_ROW + 4 * tid,
                w2[0],
                w2[1],
                w2[2],
                w2[3],
            )
        fine = cute.make_rmem_tensor((RANKS,), cutlass.Float32)
        bin_f = 255 - tid
        fsum = cutlass.Float32(0)
        fw = cute.make_rmem_tensor((RANKS,), cutlass.Uint32)
        ts_poll16(r2_uc.iterator + t * TS_ROW + bin_f, PK_SLOT, fw)
        for r in cutlass.range_constexpr(RANKS):
            fine[r] = bits_f32(fw[r])
            fsum = fsum + fine[r]
        incl2 = c_above + ts_block_scan(fsum, red, 576)
        if tid == 255:
            red[602] = bits_f32(cutlass.Uint32(0))
            red[603] = incl2
        cute.arch.sync_threads()
        if (incl2 >= target) & (incl2 - fsum < target):
            red[602] = bits_f32(bin_f.to(cutlass.Uint32))
            red[603] = incl2
        cute.arch.sync_threads()
        vstar = f32_bits(red[602])
        kept = red[603]
        for r in cutlass.range_constexpr(RANKS):
            x = cutlass.Float32(0)
            if bin_c.to(cutlass.Uint32) > bstar:
                x = coarse[r]
            if bin_f.to(cutlass.Uint32) >= vstar:
                x = x + fine[r]
            for step in cutlass.range_constexpr(5):
                x = x + cute.arch.shuffle_sync_bfly(x, 1 << step)
            if lane == 0:
                red[640 + 16 * warp + r] = x
        cute.arch.sync_threads()
        if tid == 0:
            store_vector(
                res.iterator + t * TS_RES_ROW,
                f32_bits(gm),
                f32_bits(kept),
                (bstar << cutlass.Uint32(8)) | vstar,
                draft_word,
            )
        elif tid < 5:
            ms = cute.make_rmem_tensor((4,), cutlass.Float32)
            for k in cutlass.range_constexpr(4):
                r = 4 * (tid - 1) + k
                acc = red[640 + r]
                for w8 in cutlass.range_constexpr(1, 8):
                    acc = acc + red[640 + 16 * w8 + r]
                ms[k] = acc
            store_vector(
                res.iterator + t * TS_RES_ROW + 4 * tid,
                f32_bits(ms[0]),
                f32_bits(ms[1]),
                f32_bits(ms[2]),
                f32_bits(ms[3]),
            )
        cute.arch.sync_threads()
        for i in cutlass.range_constexpr((RANKS * (TS_ROW // 4) + 255) // 256):
            q = tid + 256 * i
            if q < RANKS * (TS_ROW // 4):
                off = (
                    (q // (TS_ROW // 4)) * PK_SLOT
                    + t * TS_ROW
                    + 4 * (q % (TS_ROW // 4))
                )
                store_empty(r1_uc.iterator + off, evict_first())
                store_empty(r2_uc.iterator + off, evict_first())
        dloc = cutlass.Int32(-1)
        if t < 7:
            if f32_bits(red[3024 + t]).to(cutlass.Int32) // VOCAB_SHARD == rank:
                dloc = f32_bits(red[3024 + t]).to(cutlass.Int32) - rank * VOCAB_SHARD
        lam_t = (bstar << cutlass.Uint32(8)) | vstar
        rmass = cutlass.Float32(0)
        lastv = cutlass.Int32(-1)
        if cutlass.const_expr(TS_ROLLED):
            col = cutlass.Int32(0)
            for j in range(per):
                col = per * tid + j
                if (
                    (ts_key16(ts_vj(vals, rowbuf, per * tid + j, j)) >= lam_t)
                    & (col != dloc)
                    & (col < VOCAB_SHARD)
                ):
                    rmass = rmass + ts_ex2(
                        (ts_value(ts_vj(vals, rowbuf, per * tid + j, j)) - gm) * scale_c
                    )
                    lastv = cutlass.Int32(col)
        else:
            for j in cutlass.range_constexpr(per):
                col = per * tid + j
                if (
                    (ts_key16(ts_vj(vals, rowbuf, per * tid + j, j)) >= lam_t)
                    & (col != dloc)
                    & (col < VOCAB_SHARD)
                ):
                    rmass = rmass + ts_ex2(
                        (ts_value(ts_vj(vals, rowbuf, per * tid + j, j)) - gm) * scale_c
                    )
                    lastv = cutlass.Int32(col)
        rincl = ts_block_scan(rmass, red, 576)
        cute.arch.sync_threads()
        if tid == 0:
            lm.release_add_u32(arrivals.iterator, cutlass.Uint32(1))
        if cutlass.const_expr(bridge is None):
            clear_assigned(next_arena, next_tk, clear_counter)
        else:
            ts_bridge_clear(next_arena, next_tk, clear_counter, bridge[2])
        if t == 7:
            wait_count(arrivals, cutlass.Uint32(8))
            if tid < 8 * TS_RES_ROW // 4:
                a4, b4, c4, d4 = volatile_vector(res.iterator + 4 * tid)
                red[2304 + 4 * tid] = bits_f32(a4)
                red[2304 + 4 * tid + 1] = bits_f32(b4)
                red[2304 + 4 * tid + 2] = bits_f32(c4)
                red[2304 + 4 * tid + 3] = bits_f32(d4)
            cute.arch.sync_threads()
            if tid == 0:
                n = cutlass.Int32(0)
                going = cutlass.Boolean(True)
                for r in cutlass.range_constexpr(7):
                    if going:
                        m_r = red[2304 + r * TS_RES_ROW]
                        k_r = cutlass.Float32(0)
                        for s2 in cutlass.range_constexpr(RANKS):
                            k_r = k_r + red[2304 + r * TS_RES_ROW + 4 + s2]
                        lam = f32_bits(red[2304 + r * TS_RES_ROW + 2])
                        lb = f32_bits(red[2304 + r * TS_RES_ROW + 3]) & cutlass.Uint32(
                            0xFFFF
                        )
                        qv = cutlass.Float32(0)
                        if ts_key16(lb) >= lam:
                            qv = ts_ex2((ts_value(lb) - m_r) * scale_c) / k_r
                        if red[3008 + r] <= qv:
                            n = n + 1
                        else:
                            going = cutlass.Boolean(False)
                f = n
                m_f = red[2304 + f * TS_RES_ROW]
                k_f = cutlass.Float32(0)
                for s3 in cutlass.range_constexpr(RANKS):
                    k_f = k_f + red[2304 + f * TS_RES_ROW + 4 + s3]
                lam_f = f32_bits(red[2304 + f * TS_RES_ROW + 2])
                e_d = cutlass.Float32(0)
                d_f = cutlass.Int32(-1)
                if f < 7:
                    d_f = f32_bits(red[3024 + f]).to(cutlass.Int32)
                    lbf = f32_bits(red[2304 + f * TS_RES_ROW + 3]) & cutlass.Uint32(
                        0xFFFF
                    )
                    if ts_key16(lbf) >= lam_f:
                        e_d = ts_ex2((ts_value(lbf) - m_f) * scale_c)
                u = red[3008 + 8] * (k_f - e_d)
                owner_d = cutlass.Int32(-1)
                if d_f >= 0:
                    owner_d = d_f // VOCAB_SHARD
                cum = cutlass.Float32(0)
                sstar = cutlass.Int32(-1)
                ulocal = cutlass.Float32(0)
                last_s = cutlass.Int32(0)
                last_m = cutlass.Float32(0)
                for sidx in cutlass.range_constexpr(RANKS):
                    msv = red[2304 + f * TS_RES_ROW + 4 + sidx]
                    if owner_d == sidx:
                        msv = msv - e_d
                    if msv > cutlass.Float32(0):
                        last_s, last_m = cutlass.Int32(sidx), msv
                    if (sstar < 0) & (cum + msv > u) & (msv > cutlass.Float32(0)):
                        sstar = cutlass.Int32(sidx)
                        ulocal = u - cum
                    cum = cum + msv
                if sstar < 0:
                    sstar, ulocal = last_s, last_m
                store_vector(
                    res.iterator + 8 * TS_RES_ROW,
                    n.to(cutlass.Uint32),
                    sstar.to(cutlass.Uint32),
                    f32_bits(ulocal),
                    cutlass.Uint32(1),
                )
        if tid == 0:
            dn, ds, du, _d1 = poll_vector(res.iterator + 8 * TS_RES_ROW)
            red[2048] = bits_f32(dn)
            red[2049] = bits_f32(ds)
            red[2050] = bits_f32(du)
            red[2054] = bits_f32(cutlass.Uint32(0xFFFFFFFF))
        cute.arch.sync_threads()
        n_s = f32_bits(red[2048]).to(cutlass.Int32)
        sstar_s = f32_bits(red[2049]).to(cutlass.Int32)
        if (n_s == t) & (sstar_s == rank):
            ulocal_s = red[2050]
            red[2560 + tid] = bits_f32(lastv.to(cutlass.Uint32))
            if (rincl > ulocal_s) & (rincl - rmass <= ulocal_s):
                run = rincl - rmass
                pick = cutlass.Int32(-1)
                if cutlass.const_expr(TS_ROLLED):
                    col = cutlass.Int32(0)
                    for j in range(per):
                        col = per * tid + j
                        if (
                            (pick < 0)
                            & (ts_key16(ts_vj(vals, rowbuf, per * tid + j, j)) >= lam_t)
                            & (col != dloc)
                            & (col < VOCAB_SHARD)
                        ):
                            run = run + ts_ex2(
                                (ts_value(ts_vj(vals, rowbuf, per * tid + j, j)) - gm)
                                * scale_c
                            )
                            if run > ulocal_s:
                                pick = cutlass.Int32(col)
                else:
                    for j in cutlass.range_constexpr(per):
                        col = per * tid + j
                        if (
                            (pick < 0)
                            & (ts_key16(ts_vj(vals, rowbuf, per * tid + j, j)) >= lam_t)
                            & (col != dloc)
                            & (col < VOCAB_SHARD)
                        ):
                            run = run + ts_ex2(
                                (ts_value(ts_vj(vals, rowbuf, per * tid + j, j)) - gm)
                                * scale_c
                            )
                            if run > ulocal_s:
                                pick = cutlass.Int32(col)
                if pick < 0:
                    pick = lastv
                red[2054] = bits_f32(pick.to(cutlass.Uint32))
            cute.arch.sync_threads()
            if tid == 0:
                pk = f32_bits(red[2054]).to(cutlass.Int32)
                if pk < 0:
                    for w9 in range(256):
                        lv = f32_bits(red[2560 + w9]).to(cutlass.Int32)
                        if lv >= 0:
                            pk = lv
                store_vector(
                    r1_mc.iterator + rank * PK_SLOT + TS_R3,
                    (rank * VOCAB_SHARD + pk).to(cutlass.Uint32),
                    n_s.to(cutlass.Uint32),
                    cutlass.Uint32(1),
                    cutlass.Uint32(1),
                )
        cute.arch.sync_threads()
        if tid == 0:
            lm.release_add_u32(arrivals.iterator, cutlass.Uint32(1))
        if t == 7:
            wait_count(arrivals, cutlass.Uint32(16))
            if tid == 0:
                tok, _n2, _a, _b = poll_vector(
                    r1_uc.iterator + sstar_s * PK_SLOT + TS_R3
                )
                for j in cutlass.range_constexpr(8):
                    out = cutlass.Int32(-1)
                    if cutlass.const_expr(TS_PREDICT_ZERO):
                        out = cutlass.Int32(0)
                    if j < n_s:
                        out = f32_bits(red[3024 + j]).to(cutlass.Int32)
                    elif j == n_s:
                        out = tok.to(cutlass.Int32)
                    accepted_out[j] = out
                num_accepted_out[0] = n_s + 1
                for j in cutlass.range_constexpr(8):
                    argmax_out[j] = f32_bits(red[2304 + j * TS_RES_ROW + 20]).to(
                        cutlass.Int32
                    )
                store_empty(r1_uc.iterator + sstar_s * PK_SLOT + TS_R3, evict_first())
                if phil:
                    params[14] = bits_f32(step_w + cutlass.Uint32(1))
            if tid < (8 * TS_RES_ROW + 4) // 4:
                store_empty(res.iterator + 4 * tid, evict_first())
            cute.arch.sync_threads()
    else:
        if cutlass.const_expr(bridge is None):
            clear_assigned(next_arena, next_tk, clear_counter)
        else:
            ts_bridge_clear(next_arena, next_tk, clear_counter, bridge[2])
    if cutlass.const_expr(TS_LATE_DEP):
        cute.arch.griddepcontrol_launch_dependents()


@cute.kernel
def draft_kernel(
    x_raw_ptr: cute.Pointer,
    residual_raw: cute.Tensor,
    residual_out_raw: cute.Tensor,
    normalized_raw: cute.Tensor,
    wfa_raw_ptr: cute.Pointer,
    wqb_raw: cute.Tensor,
    wo_raw: cute.Tensor,
    w_router: cute.Tensor,
    bias: cute.Tensor,
    w3w1: cute.Tensor,
    w2: cute.Tensor,
    w_sgu: cute.Tensor,
    w_sdn: cute.Tensor,
    gamma_kv: cute.Tensor,
    gamma_in_raw: cute.Tensor,
    gamma_post_raw: cute.Tensor,
    gamma_head_raw: cute.Tensor,
    gamma_hnorm_raw: cute.Tensor,
    rope_raw: cute.Tensor,
    length: cute.Tensor,
    table_raw: cute.Tensor,
    cache_raw: cute.Tensor,
    topk_raw: cute.Tensor,
    arena_raw: cute.Tensor,
    partials_raw: cute.Tensor,
    stats_raw: cute.Tensor,
    counters_raw: cute.Tensor,
    uc_raw: cute.Tensor,
    mc_raw: cute.Tensor,
    lm_w: cute.Tensor,
    token_table: cute.Tensor,
    hidden_rows: cute.Tensor,
    draft_tokens: cute.Tensor,
    next_new_tokens: cute.Tensor,
    accepted: cute.Tensor,
    num_accepted: cute.Tensor,
    copy: cutlass.Int32,
    rank: cutlass.Int32,
    step: cutlass.Int32,
    last: cutlass.Boolean,
    stride_rows: cutlass.Int64,
    layer_rows: cutlass.Int64,
    max_blocks: cutlass.Int32,
    clear_first: cutlass.Boolean,
    kind: cutlass.Int32,
    wfa_leader_ptr: cute.Pointer,
    widx_raw_ptr: cute.Pointer,
    idx_kn_w_ptr: cute.Pointer,
    idx_kn_b_ptr: cute.Pointer,
    idx_rope_raw_ptr: cute.Pointer,
    idx_pool_raw_ptr: cute.Pointer,
    idx_blocks_raw_ptr: cute.Pointer,
    idx_stride_bytes: cutlass.Int64,
    idx_pages: cutlass.Int64,
    idx_candidates_raw: cute.Tensor,
    key_atom: cute.CopyAtom,
    embed_raw: cute.Tensor,
    embed_sharded: cutlass.Boolean,
    gamma_enorm_raw: cute.Tensor,
    eh_atom: cute.CopyAtom,
    scale: cutlass.Constexpr,
    with_target_argmax: cutlass.Constexpr,
    with_target_sample: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    allocator = SmemAllocator()
    storage = allocator.allocate_tensor(
        cutlass.Uint32, cute.make_layout((SHARED_WORDS,)), byte_alignment=1024
    )
    dsm = allocator.allocate_tensor(
        cutlass.Uint32, cute.make_layout((DSM_WORDS,)), byte_alignment=16
    )
    sm = cute.make_tensor(dsm.iterator, cute.make_layout((moe.SM_WORDS,)))
    dbars = cute.make_tensor(
        cute.recast_ptr(dsm.iterator + moe.SM_WORDS, dtype=cutlass.Uint64),
        cute.make_layout((DSM_BARS,)),
    )
    bias_sm = cute.make_tensor(
        cute.recast_ptr(dsm.iterator + DSM_BIAS, dtype=cutlass.Float32),
        cute.make_layout((256,)),
    )
    if tid < DSM_BARS:
        cute.arch.mbarrier_init(dbars.iterator + tid, 1)
    if cutlass.const_expr(BIAS_SMEM):
        bias_sm[tid] = bias[tid]
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()
    region, red, residual, scalars, _moe, all_bars, tmem_slot = shared_views(storage)

    current = copy * ARENA_WORDS
    next_copy, previous_copy = (copy + 1) % COPIES, (copy + COPIES - 1) % COPIES
    following = next_copy * ARENA_WORDS
    x_raw = cute.make_tensor(x_raw_ptr, cute.make_layout((24576,)))
    x = word_view(x_raw, 0, (8, 3072), (3072, 1))
    wfa = word_view(
        cute.make_tensor(wfa_raw_ptr, cute.make_layout((8060928,))), 0, (8060928,), (1,)
    )
    wqb = word_view(wqb_raw, 0, (2359296,), (1,))
    wo = word_view(wo_raw, 0, (6291456,), (1,))
    w_lg = word_view(w_router, moe.EXPERTS * HIDDEN // 2, (LG_W_WORDS,), (1,))
    rope = cute.make_tensor(
        rope_raw.iterator,
        cute.make_layout((cute.size(rope_raw) // 128, 64), stride=(128, 1)),
    )
    table = cute.make_tensor(
        table_raw.iterator, cute.make_layout((cute.size(table_raw),))
    )
    cache = cute.make_tensor(
        cute.recast_ptr(cache_raw.iterator, dtype=cutlass.Uint32),
        cute.make_layout((cute.size(cache_raw) // 576, 144), stride=(144, 1)),
    )
    topk = cute.make_tensor(
        topk_raw.iterator, cute.make_layout((8, 2048), stride=(2048, 1))
    )
    residual_input = word_view(residual_raw, 0, (8, 3072), (3072, 1))
    xout = word_view(normalized_raw, 0, (8, 3072), (3072, 1))
    rout = word_view(residual_out_raw, 0, (8, 3072), (3072, 1))
    gamma_in = word_view(gamma_in_raw, 0, (3072,), (1,))
    gamma_post = word_view(gamma_post_raw, 0, (3072,), (1,))
    gamma_head = word_view(gamma_head_raw, 0, (3072,), (1,))
    gamma_hnorm = word_view(gamma_hnorm_raw, 0, (3072,), (1,))
    q8 = word_view(arena_raw, current + 41984 // 4, (8, 576), (576, 1))
    kv_arena = word_view(arena_raw, current + 60416 // 4, (8, 144), (144, 1))
    o = word_view(arena_raw, current + 65024 // 4, (8, 4, 256), (1024, 256, 1))
    replicas = word_view(
        arena_raw, current + 2273024 // 4, (3, 8, 4, 256), (8192, 1024, 256, 1)
    )
    o_logits = word_view(arena_raw, current + 105600 // 4, (8, 256), (256, 1))
    h = word_view(arena_raw, current + 114176 // 4, (24576,), (1,))
    hp = word_view(arena_raw, current + 2371328 // 4, (24576,), (1,))
    hfp4 = word_view(arena_raw, current + 212480 // 4, (6912,), (1,))
    ssq1 = word_view(arena_raw, current + 240128 // 4, (8, 4), (4, 1))
    ssq2 = word_view(arena_raw, current + 240256 // 4, (8, 4), (4, 1))
    x_frag = word_view(arena_raw, current + 2174720 // 4, (24576,), (1,))
    x_ssq = word_view(arena_raw, current + 2469632 // 4, (8, 4), (4, 1))
    dummy_x = word_view(arena_raw, current + D_NEXTX, (24576,), (1,))
    dummy_ssq = word_view(arena_raw, current + D_NEXTSSQ, (8, 4), (4, 1))
    router_logits = word_view(arena_raw, current + D_LOGITS, (moe.LOGIT_WORDS,), (1,))
    act = word_view(arena_raw, current + D_ACT, (9 * moe.INTER,), (1,))
    next_arena = word_view(arena_raw, following, (ARENA_WORDS,), (1,))
    partials = word_view(partials_raw, 0, (262144,), (1,))
    stats = word_view(stats_raw, 0, (1024,), (1,))
    clear_counter = word_view(counters_raw, copy * COUNTER_WORDS, (1,), (1,))
    previous_clear = word_view(counters_raw, previous_copy * COUNTER_WORDS, (1,), (1,))
    next_counters = word_view(
        counters_raw, next_copy * COUNTER_WORDS, (COUNTER_WORDS,), (1,)
    )
    keys = cute.make_tensor(
        cute.recast_ptr(
            counters_raw.iterator + (copy * COUNTER_WORDS + C_KEYS),
            dtype=cutlass.Uint64,
        ),
        cute.make_layout((8,)),
    )
    key_words = word_view(counters_raw, copy * COUNTER_WORDS + C_KEYS, (16,), (1,))
    arrivals = word_view(counters_raw, copy * COUNTER_WORDS + C_ARR, (1,), (1,))
    x_ready = word_view(counters_raw, copy * COUNTER_WORDS + C_XREADY, (1,), (1,))
    front_done = word_view(counters_raw, copy * COUNTER_WORDS + C_FRONT, (1,), (1,))
    hop1 = word_view(uc_raw, copy * HOP_WORDS, (HOP_WORDS,), (1,))
    h1_mc = word_view(mc_raw, copy * HOP_WORDS, (HOP_WORDS,), (1,))
    hop2 = word_view(
        uc_raw, (COPIES + copy) * HOP_WORDS, (8, 16, 3072), (49152, 3072, 1)
    )
    previous_hop2 = word_view(
        uc_raw, (COPIES + previous_copy) * HOP_WORDS, (8, 16, 3072), (49152, 3072, 1)
    )
    h2_flat = word_view(mc_raw, (COPIES + copy) * HOP_WORDS, (HOP_WORDS,), (1,))
    next_tk = word_view(
        uc_raw, 2 * COPIES * HOP_WORDS + next_copy * TK_WORDS, (TK_WORDS,), (1,)
    )
    lg_offset = 2 * COPIES * HOP_WORDS + COPIES * TK_WORDS + copy * LG_WORDS
    lg_uc = word_view(uc_raw, lg_offset, (LG_WORDS,), (1,))
    lg_mc = word_view(mc_raw, lg_offset, (LG_WORDS,), (1,))
    exchange = word_view(
        uc_raw, 2 * COPIES * HOP_WORDS + copy * TK_WORDS, (TK_WORDS,), (1,)
    )
    y_offset = (
        2 * COPIES * HOP_WORDS + COPIES * TK_WORDS + COPIES * LG_WORDS + copy * Y_WORDS
    )
    y = word_view(uc_raw, y_offset, (8, 1312), (1312, 1))
    y_mc = word_view(mc_raw, y_offset, (8, 1312), (1312, 1))
    y_clear = word_view(uc_raw, y_offset, (8 * 1312,), (1,))
    read_packets = word_view(
        uc_raw, PK_BASE + previous_copy * PK_COPY, (PK_COPY,), (1,)
    )
    own_packets = word_view(uc_raw, PK_BASE + copy * PK_COPY, (PK_COPY,), (1,))
    slot_mc = word_view(
        mc_raw, PK_BASE + copy * PK_COPY + rank * PK_SLOT, (PK_SLOT,), (1,)
    )
    w_lm = cute.make_tensor(lm_w.iterator, cute.make_layout((VOCAB_SHARD * HIDDEN,)))
    w_lm8 = cute.make_tensor(
        cute.recast_ptr(lm_w.iterator, dtype=cutlass.Uint8) + 2 * VOCAB_SHARD * HIDDEN,
        cute.make_layout((VOCAB_SHARD * HIDDEN,)),
    )
    lm_scales = cute.make_tensor(
        cute.recast_ptr(lm_w.iterator, dtype=cutlass.Float32)
        + (3 * VOCAB_SHARD * HIDDEN) // 4,
        cute.make_layout((VOCAB_SHARD,)),
    )
    w_lm_words = word_view(lm_w, 0, (VOCAB_SHARD * HIDDEN // 2,), (1,))
    w_lm4 = cute.make_tensor(
        cute.recast_ptr(lm_w.iterator, dtype=cutlass.Uint8) + 2 * VOCAB_SHARD * HIDDEN,
        cute.make_layout((VOCAB_SHARD * lm.FP4_ROW_BYTES,)),
    )
    lm_scales4 = cute.make_tensor(
        cute.recast_ptr(lm_w.iterator, dtype=cutlass.Float32)
        + (2 * VOCAB_SHARD * HIDDEN + VOCAB_SHARD * lm.FP4_ROW_BYTES) // 4,
        cute.make_layout((VOCAB_SHARD,)),
    )
    loff = cutlass.Int32(0)
    if kind == KIND_SK:
        loff = step
    lview = cute.make_tensor(length.iterator + loff, cute.make_layout((1,)))
    tt = word_view(token_table, 0, (VOCAB_SHARD * HIDDEN // 2,), (1,))

    tb_arena, tb_tk, tb_clear, tb = next_arena, next_tk, clear_counter, None
    if cutlass.const_expr(TS_BRIDGE):
        b_tgt, b_cnt = (copy + 2) % COPIES, (copy + 1) % COPIES
        tb_arena = word_view(arena_raw, b_tgt * ARENA_WORDS, (ARENA_WORDS,), (1,))
        tb_tk = word_view(
            uc_raw, 2 * COPIES * HOP_WORDS + b_tgt * TK_WORDS, (TK_WORDS,), (1,)
        )
        tb_clear = word_view(counters_raw, b_cnt * COUNTER_WORDS, (1,), (1,))
        tb = (
            word_view(counters_raw, b_tgt * COUNTER_WORDS, (COUNTER_WORDS,), (1,)),
            word_view(counters_raw, b_cnt * COUNTER_WORDS, (COUNTER_WORDS,), (1,)),
            word_view(counters_raw, copy * COUNTER_WORDS + TS_BRIDGE_FLAG, (1,), (1,)),
        )
    is_ta = False
    if cutlass.const_expr(with_target_argmax):
        is_ta = kind == KIND_TA
    is_ts = False
    if cutlass.const_expr(with_target_sample):
        is_ts = kind == KIND_TS
    if is_ta:
        if cutlass.const_expr(with_target_argmax):
            if cutlass.const_expr(T_CLEAR_PREVIOUS_HOP2):
                cute.arch.griddepcontrol_wait()
                if cta >= 180:
                    if tid < 192:
                        self_clear(
                            hop1,
                            previous_hop2,
                            (cta - 180) // 4,
                            1536 * ((cta - 180) % 4) + 8 * tid,
                        )
            target_argmax(
                rank,
                cute.make_tensor(
                    cute.recast_ptr(normalized_raw.iterator, dtype=cutlass.Uint16),
                    cute.make_layout((8 * VOCAB_SHARD,)),
                ),
                red,
                keys,
                key_words,
                arrivals,
                word_view(mc_raw, PK_BASE + copy * PK_COPY, (T_HEADS,), (1,)),
                word_view(uc_raw, PK_BASE + copy * PK_COPY, (T_HEADS,), (1,)),
                draft_tokens,
                accepted,
                num_accepted,
                next_new_tokens,
                tb_arena,
                tb_tk,
                tb_clear,
                next_counters,
                bridge=tb,
            )
    elif is_ts:
        if cutlass.const_expr(with_target_sample):
            if cutlass.const_expr(T_CLEAR_PREVIOUS_HOP2):
                cute.arch.griddepcontrol_wait()
                if cta >= 180:
                    if tid < 192:
                        self_clear(
                            hop1,
                            previous_hop2,
                            (cta - 180) // 4,
                            1536 * ((cta - 180) % 4) + 8 * tid,
                        )
            target_sample(
                rank,
                cute.make_tensor(
                    cute.recast_ptr(normalized_raw.iterator, dtype=cutlass.Uint16),
                    cute.make_layout((8 * VOCAB_SHARD,)),
                ),
                cute.make_tensor(
                    cute.recast_ptr(x_raw.iterator, dtype=cutlass.Float32),
                    cute.make_layout((16,)),
                ),
                red,
                cute.make_tensor(
                    cute.recast_ptr(region.iterator, dtype=cutlass.Float32),
                    cute.make_layout((2 * TS_LANE_HIST,)),
                ),
                arrivals,
                word_view(mc_raw, PK_BASE, (PK_COPY,), (1,)),
                word_view(uc_raw, PK_BASE, (PK_COPY,), (1,)),
                word_view(mc_raw, PK_BASE + 2 * PK_COPY, (PK_COPY,), (1,)),
                word_view(uc_raw, PK_BASE + 2 * PK_COPY, (PK_COPY,), (1,)),
                word_view(
                    uc_raw,
                    PK_BASE + 2 * PK_COPY + rank * PK_SLOT + TS_RES,
                    (8 * TS_RES_ROW + 4,),
                    (1,),
                ),
                draft_tokens,
                accepted,
                num_accepted,
                next_new_tokens,
                tb_arena,
                tb_tk,
                tb_clear,
                next_counters,
                bridge=tb,
            )
    elif kind == KIND_T:
        if cutlass.const_expr(T_CLEAR_PREVIOUS_HOP2):
            cute.arch.griddepcontrol_wait()
            if cta >= 180:
                if tid < 192:
                    self_clear(
                        hop1,
                        previous_hop2,
                        (cta - 180) // 4,
                        1536 * ((cta - 180) % 4) + 8 * tid,
                    )
        target_tail(
            rank,
            word_view(normalized_raw, 0, (24576,), (1,)),
            w_lm,
            region,
            red,
            cute.make_tensor(dbars.iterator + B_LM, cute.make_layout((lm.BARS,))),
            keys,
            key_words,
            arrivals,
            word_view(mc_raw, PK_BASE + copy * PK_COPY, (T_HEADS,), (1,)),
            word_view(uc_raw, PK_BASE + copy * PK_COPY, (T_HEADS,), (1,)),
            draft_tokens,
            accepted,
            num_accepted,
            next_arena,
            next_tk,
            clear_counter,
            next_counters,
        )
    else:
        is_s0 = kind == KIND_S0
        xn_view = word_view(arena_raw, current + D_XN, (24576,), (1,))
        xn_args = (
            (xn_view, word_view(arena_raw, current + D_NS, (64,), (1,)))
            if s0.FA_STOCK
            else (None, None)
        )
        fa_kw = (
            dict(x_carry=xn_view, scale_carry=cutlass.Boolean(False), fa_stock=True)
            if s0.FA_STOCK
            else {}
        )
        if is_s0:
            s0.s0_front(
                cta,
                rank,
                accepted,
                word_view(embed_raw, 0, (cute.size(embed_raw) // 2,), (1,)),
                embed_sharded,
                word_view(uc_raw, EMB_BASE, (s0.EMB_WORDS,), (1,)),
                word_view(mc_raw, EMB_BASE, (s0.EMB_WORDS,), (1,)),
                word_view(x_raw, 0, (24576,), (1,)),
                word_view(gamma_enorm_raw, 0, (3072,), (1,)),
                gamma_hnorm,
                word_view(arena_raw, current + D_NW, (s0.NW_WORDS,), (1,)),
                word_view(uc_raw, EH_BASE, (s0.EH_WORDS,), (1,)),
                word_view(mc_raw, EH_BASE, (s0.EH_WORDS,), (1,)),
                gamma_in,
                residual_input,
                x_frag,
                x_ssq,
                region,
                red,
                scalars,
                cute.make_tensor(
                    dbars.iterator + B_EH, cute.make_layout((s0.EH_BARS,))
                ),
                eh_atom,
                tmem_slot,
                *xn_args,
            )
            rope_rows = cute.size(rope_raw) // 128
            leader_front_body(
                x_raw,
                cute.make_tensor(wfa_leader_ptr, cute.make_layout((174 * 196608,))),
                wqb_raw,
                cute.make_tensor(widx_raw_ptr, cute.make_layout((256 * 65536,))),
                gamma_kv,
                rope_raw,
                lview,
                table_raw,
                cache_raw,
                arena_raw,
                counters_raw,
                copy,
                rank,
                cutlass.Boolean(True),
                stride_rows,
                layer_rows,
                cute.make_tensor(idx_kn_w_ptr, cute.make_layout((128,))),
                cute.make_tensor(idx_kn_b_ptr, cute.make_layout((128,))),
                cute.make_tensor(idx_rope_raw_ptr, cute.make_layout((rope_rows * 64,))),
                cute.make_tensor(
                    idx_pool_raw_ptr,
                    cute.make_layout(((idx_pages - 1) * idx_stride_bytes + 8448,)),
                ),
                cute.make_tensor(
                    idx_blocks_raw_ptr, cute.make_layout((rope_rows // 64,))
                ),
                idx_stride_bytes,
                idx_pages,
                idx_candidates_raw,
                mc_raw,
                uc_raw,
                topk_raw,
                partials_raw,
                stats_raw,
                max_blocks,
                clear_first,
                key_atom,
                residual_raw,
                residual_out_raw,
                normalized_raw,
                wo_raw,
                wo_raw,
                w_sgu,
                w_sdn,
                w_sgu,
                w_sdn,
                gamma_post_raw,
                gamma_head_raw,
                gamma_post_raw,
                partials_raw,
                partials_raw,
                partials_raw,
                cutlass.Boolean(False),
                storage,
                **fa_kw,
            )
        else:
            if cta >= 180:
                cute.arch.griddepcontrol_wait()
                draft_front_unit(
                    cta - 180,
                    read_packets,
                    gamma_in,
                    residual_input,
                    x_frag,
                    x_ssq,
                    scalars,
                    front_done,
                    draft_tokens,
                    step,
                    *xn_args,
                )
            plain_front_body(
                x,
                x_frag,
                x_ssq,
                previous_clear,
                cutlass.Boolean(True),
                wfa,
                wqb,
                y,
                q8,
                gamma_kv,
                rope,
                lview,
                table,
                cache,
                kv_arena,
                stride_rows,
                layer_rows,
                next_counters,
                y_mc,
                storage,
                append_from=cutlass.Int32(TOKEN),
                **(
                    {}
                    if not s0.FA_STOCK
                    else dict(
                        fa_carry=xn_view,
                        fa_scale_carry=cutlass.Boolean(False),
                        fa_stock=True,
                    )
                ),
                short_ctx=SHORT_CTX,
            )
        sel = cutlass.Int32(TOKEN)
        extra = cutlass.Int32(SHARED_TOKEN)
        window = cutlass.Int32(1)
        if is_s0:
            sel = num_accepted[0] - 1
            if sel < 0:
                sel = cutlass.Int32(0)
            if sel > 7:
                sel = cutlass.Int32(7)
            extra = cutlass.Int32(TOKEN)
            if sel == TOKEN:
                extra = cutlass.Int32(SHARED_TOKEN)
            window = cutlass.Int32(8)
            if (cta == 0) & (tid >= 1) & (tid <= 6):
                length[tid] = length[0] - 8 + (sel + 1) + tid

        if cta < 128:
            attention_stage(
                cta,
                q8,
                kv_arena,
                lview,
                table,
                topk,
                cache,
                stride_rows,
                layer_rows,
                max_blocks,
                storage,
                all_bars,
                partials,
                stats,
                wo,
                all_bars.iterator + 1,
                exchange,
                is_s0,
                fresh_window=window,
                topk_select=sel,
            )
            proxy_fence()
            cute.arch.sync_threads()
        elif cta < 192:
            if tid == 0:
                if cutlass.const_expr(ROUTER_LG):
                    if (cta >= 148) & (cta < 164):
                        issue_o(cta, wo, w_lg, storage, all_bars.iterator + 2, True)
                    else:
                        issue_o(cta, wo, wo, storage, all_bars.iterator + 1, False)
                else:
                    issue_o(cta, wo, wo, storage, all_bars.iterator + 1, False)
        if cutlass.const_expr(L2_PREFETCH):
            if (cta < moe.ROUTER_UNITS) & (tid == 0) & (not ROUTER_LG):
                prefetch_bulk_l2(
                    w_router.iterator + 2 * cta * HIDDEN, cutlass.Uint32(2 * HIDDEN * 2)
                )
            prefetch_slice(w_sgu, 2 * moe.INTER * HIDDEN * 2, cta)
            prefetch_slice(w_sdn, HIDDEN * moe.INTER * 2, cta)
        o_passes = cutlass.Int32(1)
        if cutlass.const_expr(WARM_O):
            if (cta >= 128) & (cta < (148 if ROUTER_LG else 164)):
                o_passes = cutlass.Int32(2)
        if (cta >= 164) & (cta < 196):
            merge_stage(
                cta - 164,
                partials,
                stats,
                o,
                replicas,
                red,
                clear_first,
                **(dict(passes=cutlass.Int32(2)) if WARM_MERGE else {}),
            )
            proxy_fence()
            cute.arch.sync_threads()
        if cta < 192:
            if cutlass.const_expr(ROUTER_LG):
                if (cta >= 148) & (cta < 164):
                    o_stage(
                        cta,
                        rank,
                        o,
                        replicas,
                        residual_input,
                        w_lg,
                        region,
                        red,
                        all_bars.iterator + 2,
                        h1_mc,
                        lg_uc,
                        lg_mc,
                        o_logits,
                        tmem_slot[0],
                        all_bars.iterator + 27,
                        True,
                        only_token=sel if (ONE_ROW_H1 or SEL_H1) else None,
                        warm=False,
                    )
                else:
                    o_stage(
                        cta,
                        rank,
                        o,
                        replicas,
                        residual_input,
                        wo,
                        region,
                        red,
                        all_bars.iterator + 1,
                        h1_mc,
                        lg_uc,
                        lg_mc,
                        o_logits,
                        tmem_slot[0],
                        all_bars.iterator + 27,
                        False,
                        only_token=sel if (ONE_ROW_H1 or SEL_H1) else None,
                        **o_warm(o_passes),
                    )
            else:
                o_stage(
                    cta,
                    rank,
                    o,
                    replicas,
                    residual_input,
                    wo,
                    region,
                    red,
                    all_bars.iterator + 1,
                    h1_mc,
                    lg_uc,
                    lg_mc,
                    o_logits,
                    tmem_slot[0],
                    all_bars.iterator + 27,
                    False,
                    only_token=sel if (ONE_ROW_H1 or ROW_HOP or SEL_H1) else None,
                    **o_warm(o_passes),
                )
            proxy_fence()
            cute.arch.sync_threads()

        clear_assigned(next_arena, next_tk, clear_counter)
        h1_done = word_view(counters_raw, copy * COUNTER_WORDS + C_H1, (1,), (1,))
        h2_passes = cutlass.Int32(1)
        if cutlass.const_expr(WARM_HOP and H2_IDLE):
            if cta >= FC1_CTAS:
                h2_passes = cutlass.Int32(2)
        warm_passes = cutlass.Int32(1)
        if cutlass.const_expr(WARM_HOP):
            if cta >= 192:
                warm_passes = cutlass.Int32(2)
        if cutlass.const_expr(ROW_HOP):
            if cta == ROW_CTA:
                row_hop(
                    False,
                    sel,
                    hop1,
                    hop2,
                    previous_hop2,
                    residual_input,
                    gamma_post,
                    red,
                    scalars,
                    h,
                    rout,
                    h1_done,
                )
        elif cta >= 180:
            is_sel = (cta - 180) // 4 == sel
            probe = cutlass.Boolean(False)
            if cutlass.const_expr(WARM_L2_PROBE):
                probe = (cta >= 192) & ((cta - 180) // 4 != sel)
            if is_sel | probe | (not (ONE_ROW_H1 or SEL_H1)):
                if cutlass.const_expr(WARM_L2_PROBE):
                    stage_hop(
                        cta - 180,
                        False,
                        cutlass.Boolean(False),
                        hop1,
                        hop2,
                        previous_hop2,
                        residual_input,
                        gamma_post,
                        cutlass.Float32(1),
                        residual,
                        scalars,
                        ssq1,
                        ssq2,
                        h,
                        hp,
                        hfp4,
                        dummy_x,
                        dummy_ssq,
                        xout,
                        rout,
                        lean=HOP_LEAN,
                        warm=(cutlass.Int32(1), dummy_x, dummy_ssq, probe),
                    )
                else:
                    stage_hop(
                        cta - 180,
                        False,
                        cutlass.Boolean(False),
                        hop1,
                        hop2,
                        previous_hop2,
                        residual_input,
                        gamma_post,
                        cutlass.Float32(1),
                        residual,
                        scalars,
                        ssq1,
                        ssq2,
                        h,
                        hp,
                        hfp4,
                        dummy_x,
                        dummy_ssq,
                        xout,
                        rout,
                        lean=HOP_LEAN,
                        **hop_warm(warm_passes, dummy_x, dummy_ssq),
                    )
                if cutlass.const_expr(SEL_H1):
                    cute.arch.sync_threads()
                    if (tid == 0) & is_sel:
                        release_increment(h1_done.iterator)
        base = region.iterator.toint().to(cutlass.Uint32)
        fc2_base = base + cutlass.Uint32(moe.FC1_REGION)
        shared_out = word_view(arena_raw, current + D_SH, (3072,), (1,))
        if cutlass.const_expr(ROUTER_LG):
            pass
        elif cta < moe.ROUTER_UNITS:
            moe.issue_router(cta, w_router, fc2_base, dbars.iterator + B_ROUTER)
            moe.router_unit(
                cta,
                h,
                sel * 3072,
                fc2_base,
                sm,
                router_logits,
                dbars.iterator + B_ROUTER,
                True,
            )
        rbias = bias_sm if BIAS_SMEM else bias
        route_fn = moe.route_moe_parallel if moe.ROUTE_PARALLEL else moe.route_moe
        rlog = router_logits
        lgk = {}
        if cutlass.const_expr(ROUTER_LG):
            rlog = word_view(arena_raw, current + 105600 // 4 + sel * 256, (256,), (1,))
            lgk = dict(
                ssq=word_view(arena_raw, current + 240128 // 4 + sel * 4, (4,), (1,))
            )
        if cutlass.const_expr(ROUTE_EARLY_ISSUE):

            def early_issue():
                proxy_fence()
                moe.issue_fc1(
                    cta, FC1_CTAS, w3w1, sm, base, dbars.iterator + B_FC1, w_sgu
                )
                if cta < moe.FC2_UNITS:
                    moe.issue_fc2(cta, w2, sm, fc2_base, dbars.iterator + B_FC2)

            lgk = dict(lgk, after_select=early_issue)
        in_moe = cutlass.Boolean(True)
        if cutlass.const_expr(H2_IDLE):
            in_moe = cta < FC1_CTAS
        if in_moe:
            route_fn(rlog, rbias, sm, scale, **lgk)
            if cutlass.const_expr(LM_PF_CHUNKS > 0):
                if cta < lm.LM_CTAS:
                    lm_prefetch_chunks(cta, w_lm)
            proxy_fence()
            if cutlass.const_expr(not ROUTE_EARLY_ISSUE):
                moe.issue_fc1(
                    cta, FC1_CTAS, w3w1, sm, base, dbars.iterator + B_FC1, w_sgu
                )
                if cta < moe.FC2_UNITS:
                    moe.issue_fc2(cta, w2, sm, fc2_base, dbars.iterator + B_FC2)
            moe.fc1_unit(
                cta,
                FC1_CTAS,
                h,
                sel * 3072,
                base,
                sm,
                act,
                dbars.iterator + B_FC1,
                True,
                9 * (moe.INTER // 2),
            )
        if cta < moe.FC2_UNITS:
            if tid == 0:
                ready = acquire_word(clear_counter.iterator)
                while ready < cutlass.Uint32(COUNTED_CLEARS):
                    ready = acquire_word(clear_counter.iterator)
            cute.arch.sync_threads()
            if cutlass.const_expr(SHARED_IN_FC2):
                proxy_fence()
                shared_rows32(cta, act, w_sdn, base, scalars, dbars.iterator + B_SD)
                moe.fc2_unit(
                    cta,
                    act,
                    fc2_base,
                    sm,
                    h2_flat,
                    (sel * 16 + rank) * 3072,
                    dbars.iterator + B_FC2,
                    addend_rows=scalars,
                )
            else:
                moe.fc2_unit(
                    cta,
                    act,
                    fc2_base,
                    sm,
                    h2_flat,
                    (sel * 16 + rank) * 3072,
                    dbars.iterator + B_FC2,
                    addend=shared_out,
                )
        else:
            if cutlass.const_expr(not SHARED_IN_FC2):
                proxy_fence()
                shared_down(
                    cta - moe.FC2_UNITS,
                    act,
                    w_sdn,
                    region,
                    red,
                    shared_out,
                    0,
                    dbars.iterator + B_SD,
                    None,
                )
        tc_before()
        cute.arch.sync_threads()
        if tid // 32 == 0:
            tc_after()
            deallocate_tmem(tmem_slot[0])
        proxy_fence()
        lm_bars = cute.make_tensor(dbars.iterator + B_LM, cute.make_layout((lm.BARS,)))
        hh_bars = cute.make_tensor(dbars.iterator + B_HH, cute.make_layout((HH_SLOTS,)))
        if cta < lm.LM_CTAS:
            if cutlass.const_expr(LM_FP4):
                lm.issue_lm_fp4(cta, w_lm4, region, lm_bars.iterator)
            elif cutlass.const_expr(LM_FP8):
                lm.issue_lm_prologue(cta, w_lm8, region, lm_bars, True)
            else:
                lm.issue_lm_prologue(cta, w_lm, region, lm_bars)
            if cutlass.const_expr(LM_PF_FC2_CHUNKS > 0):
                lm_prefetch_chunks(cta, w_lm, LM_PF_FC2_CHUNKS)
        else:
            if not last:
                issue_hidden_rows(cta - lm.LM_CTAS, hidden_rows, region, hh_bars)

        if cutlass.const_expr(ROW_HOP):
            if cta >= 180:
                unit = cta - 180
                if cta == ROW_CTA:
                    row_hop(
                        True,
                        sel,
                        hop1,
                        hop2,
                        previous_hop2,
                        residual_input,
                        gamma_head,
                        red,
                        scalars,
                        xout,
                        rout,
                        x_ready,
                    )
                if unit // 4 != sel:
                    if tid < 192:
                        self_clear(
                            hop1, previous_hop2, unit // 4, 1536 * (unit % 4) + 8 * tid
                        )
                if not is_s0:
                    if unit < 28:
                        clear_packets(unit, read_packets, front_done)
                    wait_count(h1_done, cutlass.Uint32(1))
                    index = unit * 256 + tid
                    if index < 2624:
                        store_empty(y_clear.iterator + 4 * index, evict_first())
                cute.arch.sync_threads()
        elif cta >= 180:
            unit = cta - 180
            if unit // 4 == sel:
                stage_hop(
                    unit,
                    True,
                    cutlass.Boolean(False),
                    hop1,
                    hop2,
                    previous_hop2,
                    residual_input,
                    gamma_head,
                    cutlass.Float32(1),
                    residual,
                    scalars,
                    ssq1,
                    ssq2,
                    h,
                    hp,
                    hfp4,
                    dummy_x,
                    dummy_ssq,
                    xout,
                    rout,
                    lean=HOP_LEAN,
                    **hop_warm(
                        h2_passes if H2_IDLE else warm_passes, dummy_x, dummy_ssq, True
                    ),
                    amax_out=word_view(arena_raw, current + D_XAMAX, (32,), (1,))
                    if H2_XAMAX
                    else None,
                )
                if cutlass.const_expr(H2_CLEAR_CURRENT and not H2_CLEAR_LATE):
                    if tid < 192:
                        for r in cutlass.range_constexpr(RANKS):
                            store_empty(
                                hop2.iterator
                                + (sel * RANKS + r) * 3072
                                + (1536 * (unit % 4) + 8 * tid) // 2,
                                evict_first(),
                            )
                cute.arch.sync_threads()
                if tid == 0:
                    release_increment(x_ready.iterator)
                if cutlass.const_expr(H2_CLEAR_CURRENT and H2_CLEAR_LATE):
                    if tid < 192:
                        for r in cutlass.range_constexpr(RANKS):
                            store_empty(
                                hop2.iterator
                                + (sel * RANKS + r) * 3072
                                + (1536 * (unit % 4) + 8 * tid) // 2,
                                evict_first(),
                            )
            else:
                if tid < 192:
                    self_clear(
                        hop1, previous_hop2, unit // 4, 1536 * (unit % 4) + 8 * tid
                    )
                if not is_s0:
                    clear_packets(unit, read_packets, front_done)
            if not is_s0:
                if cutlass.const_expr(SEL_H1):
                    wait_count(h1_done, cutlass.Uint32(4))
                index = unit * 256 + tid
                if index < 2624:
                    store_empty(y_clear.iterator + 4 * index, evict_first())
            cute.arch.sync_threads()

        xrow = word_view(normalized_raw, sel * 3072, (3072,), (1,))
        if cta < lm.LM_CTAS:
            wait_count(x_ready, cutlass.Uint32(4))
            fence_proxy_global()
            if cutlass.const_expr(LM_FP4):
                lm.lm_head_fp4_stage(
                    cta,
                    xrow,
                    region,
                    red,
                    lm_bars.iterator,
                    lm_bars.iterator + 1,
                    keys,
                    arrivals,
                    lm_scales4,
                    w_lm_words,
                )
            else:
                lm.issue_lm_x(xrow, cutlass.Int32(1), region, lm_bars)
                xsc = None
                if cutlass.const_expr(LM_FP8X):
                    s0w, s1w, s2w, s3w = volatile_vector(ssq2.iterator + 4 * sel)
                    m0, m1, m2, m3 = volatile_vector(
                        word_view(
                            arena_raw, current + D_XAMAX + 4 * sel, (4,), (1,)
                        ).iterator
                    )
                    tot = (
                        (cutlass.Float32(0) + bits_f32(s0w)) + bits_f32(s1w)
                    ) + bits_f32(s2w)
                    tot += bits_f32(s3w)
                    rstd_x = cute.math.rsqrt(tot / 6144.0 + 1.0e-5, fastmath=True)
                    mx = (
                        max_num(
                            max_num(bits_f32(m0), bits_f32(m1)),
                            max_num(bits_f32(m2), bits_f32(m3)),
                        )
                        * rstd_x
                    )
                    sx = max_num(mx, cutlass.Float32(1e-12)) / 448.0
                    xsc = (sx, 1.0 / sx)
                lm.lm_head_stage(
                    cta,
                    xrow,
                    w_lm8 if LM_FP8 else w_lm,
                    region,
                    red,
                    lm_bars,
                    keys,
                    arrivals,
                    cutlass.Int32(1),
                    red,
                    False,
                    False,
                    table_prefetch=tt if LM_TABLE_PF else None,
                    **(
                        dict(
                            fp8=True,
                            scales=lm_scales,
                            rescore=w_lm_words if LM_FP8_RESCORE else None,
                            xscale=xsc,
                        )
                        if LM_FP8
                        else {}
                    ),
                )
            if cta == lm.LM_CTAS - 1:
                publish_packet(
                    rank,
                    step,
                    last,
                    key_words,
                    arrivals,
                    tt,
                    slot_mc,
                    own_packets,
                    scalars,
                    draft_tokens,
                    next_new_tokens,
                    accepted,
                    num_accepted,
                )
        else:
            if not last:
                wait_count(x_ready, cutlass.Uint32(4))
                hidden_half(
                    cta - lm.LM_CTAS,
                    xrow,
                    gamma_hnorm,
                    hidden_rows,
                    region,
                    red,
                    hh_bars,
                    word_view(slot_mc, PK_HID, (SHARE,), (1,)),
                )


@cute.jit
def draft_entry(
    x_raw: cute.Tensor,
    residual_raw: cute.Tensor,
    residual_out_raw: cute.Tensor,
    normalized_raw: cute.Tensor,
    wfa_raw: cute.Tensor,
    wqb_raw: cute.Tensor,
    wo_raw: cute.Tensor,
    w_router: cute.Tensor,
    bias: cute.Tensor,
    w3w1: cute.Tensor,
    w2: cute.Tensor,
    w_sgu: cute.Tensor,
    w_sdn: cute.Tensor,
    gamma_kv: cute.Tensor,
    gamma_in_raw: cute.Tensor,
    gamma_post_raw: cute.Tensor,
    gamma_head_raw: cute.Tensor,
    gamma_hnorm_raw: cute.Tensor,
    rope_raw: cute.Tensor,
    length: cute.Tensor,
    table_raw: cute.Tensor,
    cache_raw: cute.Tensor,
    topk_raw: cute.Tensor,
    arena_raw: cute.Tensor,
    partials_raw: cute.Tensor,
    stats_raw: cute.Tensor,
    counters_raw: cute.Tensor,
    uc_raw: cute.Tensor,
    mc_raw: cute.Tensor,
    lm_w: cute.Tensor,
    token_table: cute.Tensor,
    hidden_rows: cute.Tensor,
    draft_tokens: cute.Tensor,
    next_new_tokens: cute.Tensor,
    accepted: cute.Tensor,
    num_accepted: cute.Tensor,
    copy: cutlass.Int32,
    rank: cutlass.Int32,
    step: cutlass.Int32,
    last: cutlass.Boolean,
    stride_rows: cutlass.Int64,
    layer_rows: cutlass.Int64,
    max_blocks: cutlass.Int32,
    clear_first: cutlass.Boolean,
    kind: cutlass.Int32,
    wfa_leader_raw: cute.Tensor,
    widx_raw: cute.Tensor,
    idx_kn_w: cute.Tensor,
    idx_kn_b: cute.Tensor,
    idx_rope_raw: cute.Tensor,
    idx_pool_raw: cute.Tensor,
    idx_blocks_raw: cute.Tensor,
    idx_stride_bytes: cutlass.Int64,
    idx_pages: cutlass.Int64,
    idx_candidates_raw: cute.Tensor,
    embed_raw: cute.Tensor,
    embed_sharded: cutlass.Boolean,
    eh_w: cute.Tensor,
    gamma_enorm_raw: cute.Tensor,
    scale: cutlass.Constexpr,
    stream,
    use_pdl: cutlass.Constexpr,
    with_target_argmax: cutlass.Constexpr,
    with_target_sample: cutlass.Constexpr,
):
    sizes = (
        cute.size(w_router),
        cute.size(bias),
        cute.size(w3w1),
        cute.size(w2),
        cute.size(w_sgu),
        cute.size(w_sdn),
        cute.size(lm_w),
        cute.size(token_table),
        cute.size(hidden_rows),
        cute.size(mc_raw) >= DRAFT_TRANSPORT_WORDS,
        cute.size(eh_w),
        cute.size(gamma_enorm_raw),
        cute.size(length),
    )
    if cutlass.const_expr(LM_FP8X and not (LM_FP8 and H2_XAMAX)):
        raise TypeError("LM_FP8X needs LM_FP8 and H2_XAMAX")
    if cutlass.const_expr(ROUTE_EARLY_ISSUE and not moe.ROUTE_PARALLEL):
        raise TypeError("ROUTE_EARLY_ISSUE needs route_moe_parallel (ROUTE_PARALLEL)")
    if cutlass.const_expr(ROUTER_LG and (ROW_HOP or not moe.ROUTE_PARALLEL)):
        raise TypeError(
            "ROUTER_LG needs stage_hop H1 (its ssq1 words) and ROUTE_PARALLEL"
        )
    if cutlass.const_expr(
        LM_FP4
        and (
            LM_FP8
            or LM_PF_FC2_CHUNKS
            or LM_PF_CHUNKS
            or LM_L2_KEEP_CHUNKS
            or LM_TABLE_PF
        )
    ):
        raise TypeError(
            "LM_FP4 does not combine with LM_FP8 / the BF16 lm knobs / LM_TABLE_PF"
        )
    if cutlass.const_expr(
        LM_FP8 and (LM_PF_FC2_CHUNKS or LM_PF_CHUNKS or LM_L2_KEEP_CHUNKS)
    ):
        raise TypeError(
            "LM_FP8 does not combine with the BF16 lm prefetch / keep / ablation knobs"
        )
    want = (
        W_ROUTER_ELEMS,
        moe.EXPERTS,
        moe.EXPERTS * 2 * moe.INTER * HIDDEN,
        moe.EXPERTS * HIDDEN * moe.INTER,
        2 * moe.INTER * HIDDEN,
        HIDDEN * moe.INTER,
        W_LM_ELEMS,
        VOCAB_SHARD * HIDDEN,
        SHARE * HIDDEN,
        True,
        HIDDEN * s0.WINDOW,
        HIDDEN,
        7,
    )
    if cutlass.const_expr(sizes != want):
        raise TypeError(f"draft_entry: operand sizes {sizes} != {want}")
    pool_rows = cute.make_tensor(
        idx_pool_raw.iterator,
        cute.make_layout(
            (128, ((idx_pages - 1) * idx_stride_bytes + 8448) // 128), stride=(1, 128)
        ),
    )
    key_layout = cute.make_composed_layout(
        cute.make_swizzle(3, 4, 3), 0, cute.make_layout((128, 64), stride=(1, 128))
    )
    key_atom, _ = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), pool_rows, key_layout, (128, 64)
    )
    eh_atom = s0.make_eh_atom(eh_w)
    x_words = cute.make_ptr(
        cutlass.Uint32, x_raw.iterator.toint(), cute.AddressSpace.gmem, assumed_align=16
    )
    wfa_words = cute.make_ptr(
        cutlass.Uint32,
        wfa_raw.iterator.toint(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    draft_kernel(
        x_words,
        residual_raw,
        residual_out_raw,
        normalized_raw,
        wfa_words,
        wqb_raw,
        wo_raw,
        w_router,
        bias,
        w3w1,
        w2,
        w_sgu,
        w_sdn,
        gamma_kv,
        gamma_in_raw,
        gamma_post_raw,
        gamma_head_raw,
        gamma_hnorm_raw,
        rope_raw,
        length,
        table_raw,
        cache_raw,
        topk_raw,
        arena_raw,
        partials_raw,
        stats_raw,
        counters_raw,
        uc_raw,
        mc_raw,
        lm_w,
        token_table,
        hidden_rows,
        draft_tokens,
        next_new_tokens,
        accepted,
        num_accepted,
        copy,
        rank,
        step,
        last,
        stride_rows,
        layer_rows,
        max_blocks,
        clear_first,
        kind,
        wfa_leader_raw.iterator,
        widx_raw.iterator,
        idx_kn_w.iterator,
        idx_kn_b.iterator,
        idx_rope_raw.iterator,
        idx_pool_raw.iterator,
        idx_blocks_raw.iterator,
        idx_stride_bytes,
        idx_pages,
        idx_candidates_raw,
        key_atom,
        embed_raw,
        embed_sharded,
        gamma_enorm_raw,
        eh_atom,
        scale,
        with_target_argmax,
        with_target_sample,
    ).launch(
        grid=(212, 1, 1),
        block=(256, 1, 1),
        min_blocks_per_mp=1,
        stream=stream,
        use_pdl=use_pdl,
    )


from .attention import (
    EMPTY,
    attention_stage,
    bf16_from_word,
    bf16_pair,
    issue_o,
    merge_stage,
    o_stage,
    published_word,
    volatile_vector,
)

# Imported last: the role modules import each other, and their names are used only when a kernel is traced.
from .bf16_moe import add_rn
from .common import (
    COUNTED_CLEARS,
    as_ptr,
    bits_f32,
    bulk_copy,
    clear_assigned,
    deallocate_tmem,
    evict_first,
    f32_bits,
    f32_intrinsic,
    fma_rn,
    max_num,
    prefetch_bulk_l2,
    proxy_fence,
    reduce_rank16,
    release_increment,
    reset_next_counters,
    self_clear,
    stage_hop,
    store_empty,
    tc_after,
    tc_before,
    xfrag_word_offset,
)
from .gpu_publication import (
    acquire_word,
    cached_vector,
    global_pair,
    readonly_vector,
    store_vector,
    store_word,
)
from .indexer import round_bf16
from .moe import SHARED_WORDS, shared_views, volatile_word
from .plain import leader_front_body, plain_front_body, word_view
