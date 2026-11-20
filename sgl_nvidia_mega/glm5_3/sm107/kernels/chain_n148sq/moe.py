# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import cutlass
from cutlass._mlir.dialects import llvm, nvvm
import cutlass.cute as cute
from cutlass.cutlass_dsl import T, dsl_user_op
from .schedule import CTAS, CUT_TOTAL, CUT_RATE, CUT_DISCOUNTS, FC1_SPLIT_EXPERTS


@dsl_user_op
def volatile_word(pointer: cute.Pointer, *, loc=None, ip=None):
    return cutlass.Uint32(llvm.load(T.i32(), as_ptr(1, pointer, loc, ip), volatile_=True, alignment=4, loc=loc, ip=ip))


@dsl_user_op
def redux_max(value: cutlass.Uint32, *, loc=None, ip=None):
    return cutlass.Uint32(cute.arch.warp_redux_sync(value, "umax", loc=loc, ip=ip))


@dsl_user_op
def redux_min(value: cutlass.Uint32, *, loc=None, ip=None):
    return cutlass.Uint32(cute.arch.warp_redux_sync(value, "umin", loc=loc, ip=ip))


@cute.jit
def route_tokens(
    logits: cute.Tensor, ssq_input: cute.Tensor, bias_bits: cute.Tensor,
    moe: cute.Tensor, scratch: cute.Tensor, route: cute.Tensor, publish: cutlass.Boolean,
    h_reload=None,
):
    tid, _, _ = cute.arch.thread_idx()
    token = tid // 32
    lane = tid % 32
    scratch[512 + tid] = cutlass.Uint32(0)
    bias = cute.make_rmem_tensor((8,), cutlass.Float32)
    payload = cute.make_rmem_tensor((8,), cutlass.Uint32)
    ssq = cute.make_rmem_tensor((4,), cutlass.Uint32)
    for i in cutlass.range_constexpr(8):
        bias[i] = bf16_from_word(bias_bits[32 * i + lane].to(cutlass.Uint32), False)
        payload[i] = volatile_word(logits.iterator + token * 256 + 32 * i + lane)
    for j in cutlass.range_constexpr(4):
        ssq[j] = cutlass.Uint32(0)
    if lane == 0:
        s0, s1, s2, s3 = volatile_vector(ssq_input.iterator + token * 4)
        ssq[0], ssq[1], ssq[2], ssq[3] = s0, s1, s2, s3
    waiting = cutlass.Boolean(True)
    while waiting:
        full = cutlass.Boolean(True)
        for i in cutlass.range_constexpr(8):
            full = full & (payload[i] != cutlass.Uint32(0x80000000))
        ssq_full = cutlass.Boolean(True)
        for j in cutlass.range_constexpr(4):
            ssq_full = ssq_full & (ssq[j] != cutlass.Uint32(0x80000000))
        full = full & ((lane != 0) | ssq_full)
        if cute.arch.vote_all_sync(full):
            waiting = cutlass.Boolean(False)
        else:
            if (lane == 0) & (not ssq_full):
                s0, s1, s2, s3 = volatile_vector(ssq_input.iterator + token * 4)
                ssq[0], ssq[1], ssq[2], ssq[3] = s0, s1, s2, s3
            for i in cutlass.range_constexpr(8):
                if payload[i] == cutlass.Uint32(0x80000000):
                    payload[i] = volatile_word(logits.iterator + token * 256 + 32 * i + lane)
    if cutlass.const_expr(h_reload is not None):
        h, values = h_reload
        for v in cutlass.range_constexpr(7):
            complete = cutlass.Boolean(True)
            for j in cutlass.range_constexpr(4):
                complete = complete & (values[v, j] != cutlass.Uint32(0x80000000))
            if (not complete) & (tid + 256 * v < 1728):
                a, b, c, d = volatile_vector(h.iterator + 4 * (tid + 256 * v))
                values[v, 0], values[v, 1], values[v, 2], values[v, 3] = a, b, c, d
    rstd = cutlass.Float32(1.0)
    if lane == 0:
        total = cutlass.Float32(0.0)
        for j in cutlass.range_constexpr(4):
            total += bits_f32(ssq[j])
        rstd = cute.math.rsqrt(total / 6144.0 + 1.0e-5, fastmath=True)
        moe[MOE_RCP + token] = f32_bits(rstd)
    rstd = cute.arch.shuffle_sync(rstd, 0)
    sigmoid = cute.make_rmem_tensor((8,), cutlass.Float32)
    keys = cute.make_rmem_tensor((8,), cutlass.Uint32)
    for i in cutlass.range_constexpr(8):
        logit = (cutlass.Float32(0.0) + bits_f32(payload[i])) * rstd
        sigmoid[i] = 0.5 * cute.math.tanh(0.5 * logit, fastmath=False) + 0.5
        score_bits = f32_bits(sigmoid[i] + bias[i])
        sign_mask = cutlass.Uint32(0x80000000)
        if (score_bits >> 31) != cutlass.Uint32(0):
            sign_mask = cutlass.Uint32(0xFFFFFFFF)
        keys[i] = score_bits ^ sign_mask
    taken = cutlass.Uint32(0)
    top_sig = cutlass.Float32(0.0)
    top_expert = cutlass.Uint32(0)
    for k in range(8):
        best = cutlass.Uint32(0)
        bi = cutlass.Int32(0)
        for step in cutlass.range_constexpr(8):
            i = 7 - step
            if ((taken & (cutlass.Uint32(1) << i)) == cutlass.Uint32(0)) & (keys[i] >= best):
                best = keys[i]
                bi = cutlass.Int32(i)
        wmax = redux_max(best)
        expert = (32 * bi + lane).to(cutlass.Uint32)
        candidate = cutlass.Uint32(0xFFFF)
        if best == wmax:
            candidate = expert
        winner = redux_min(candidate)
        selected_sigmoid = cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(8):
            if bi == i:
                selected_sigmoid = sigmoid[i]
        weight_sigmoid = cute.arch.shuffle_sync(selected_sigmoid, (winner & cutlass.Uint32(31)).to(cutlass.Int32))
        if lane.to(cutlass.Uint32) == (winner & cutlass.Uint32(31)):
            taken |= cutlass.Uint32(1) << (winner >> 5)
        if lane == k:
            top_sig = weight_sigmoid
            top_expert = winner
    sig = cutlass.Float32(0.0)
    if lane < 8:
        sig = top_sig
    total_sigmoid = sig
    for step in cutlass.range_constexpr(3):
        total_sigmoid += cute.arch.shuffle_sync_bfly(total_sigmoid, 4 >> step)
    if lane < 8:
        weight = round_bf16(sig * 2.5 / (total_sigmoid + 0.0))
        scratch[token * 8 + lane] = top_expert
        scratch[64 + token * 8 + lane] = f32_bits(weight)
    cute.arch.sync_threads()
    if lane < 8:
        cute.arch.atomic_or(scratch.iterator + 512 + top_expert,
                            cutlass.Uint32(1) << token, sem='relaxed', scope='cta')
    cute.arch.sync_threads()
    token_mask = scratch[512 + tid]
    selected = cute.arch.vote_ballot_sync(token_mask != cutlass.Uint32(0))
    if lane == 0:
        scratch[128 + token] = selected
    cute.arch.sync_threads()
    slot = cute.arch.popc(selected & ((cutlass.Uint32(1) << lane) - cutlass.Uint32(1)))
    count = cutlass.Uint32(0)
    for warp in cutlass.range_constexpr(8):
        n = cute.arch.popc(scratch[128 + warp])
        if warp < token:
            slot += n
        count += n
    if token_mask != cutlass.Uint32(0):
        moe[MOE_SLOT_EXPERT + slot] = tid.to(cutlass.Uint32)
        scratch[256 + slot] = token_mask
        scratch[3072 + tid] = slot
    if tid == 0:
        moe[MOE_E] = count
    cute.arch.sync_threads()
    if tid < 64:
        moe[MOE_SLOT_OF_TK + tid] = scratch[3072 + scratch[tid]]
        moe[MOE_WEIGHT + tid] = scratch[64 + tid]
    cute.arch.sync_threads()
    if publish & (tid < 64):
        if tid < moe[MOE_E]:
            route[ROUTE_SLOT_EXPERT + tid] = moe[MOE_SLOT_EXPERT + tid]
        route[ROUTE_SLOT_OF_TK + tid] = moe[MOE_SLOT_OF_TK + tid]
        route[ROUTE_WEIGHT + tid] = moe[MOE_WEIGHT + tid]
        if tid == 0:
            route[ROUTE_E] = moe[MOE_E]


@cute.jit
def pack_slot_masks(moe: cute.Tensor, scratch: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    if tid < 16:
        word = cutlass.Uint32(0)
        for j in cutlass.range_constexpr(4):
            slot = 4 * tid + j
            mask = cutlass.Uint32(0)
            if slot < moe[MOE_E]:
                mask = scratch[256 + slot] & cutlass.Uint32(0xFF)
            word |= mask << (8 * j)
        moe[MOE_SLOT_MASK + tid] = word


PIECE_BYTES = 55296
PIECE_WORDS = PIECE_BYTES // 4
FC2_WORKERS = 164


@cute.jit
def clipped_segment(c: cutlass.Int32, start: cutlass.Int32, end: cutlass.Int32):
    n = c - start
    if c <= start:
        n = cutlass.Int32(0)
    elif c >= end:
        n = end - start
    return n


@cute.jit
def cut_weight(c: cutlass.Int32):
    weight = CUT_RATE * c
    for i in cutlass.range_constexpr(len(CUT_DISCOUNTS)):
        start, end, discount = CUT_DISCOUNTS[i]
        segment = clipped_segment(c, start, end)
        weight = weight - (segment if discount == 1 else discount * segment)
    return weight


@cute.jit
def aligned_f2(c: cutlass.Int32, experts: cutlass.Int32,
               small_width: cutlass.Constexpr = False):
    result = 8 * experts
    if c < CTAS:
        slot = ((c + 1) * experts - 1) // CTAS
        if slot >= experts:
            slot = experts - 1
        c0 = slot * CTAS // experts
        width = (slot + 1) * CTAS // experts - c0
        numerator = 8 * (c - c0)
        if cutlass.const_expr(small_width):
            reciprocal = cutlass.Int32(65536 // 8)
            for divisor in cutlass.range_constexpr(3, 8):
                if width == divisor:
                    reciprocal = cutlass.Int32((65536 + divisor - 1) // divisor)
            fraction = (numerator * reciprocal) >> 16
        else:
            fraction = numerator // width
        result = 8 * slot + fraction
    return result


@cute.jit
def worker_fc2_start(worker: cutlass.Int32, experts: cutlass.Int32):
    result = 8 * experts
    if worker < FC2_WORKERS:
        slot = ((worker + 1) * experts - 1) // FC2_WORKERS
        if slot >= experts:
            slot = experts - 1
        first_worker = slot * FC2_WORKERS // experts
        width = (slot + 1) * FC2_WORKERS // experts - first_worker
        result = 8 * slot + 8 * (worker - first_worker) // width
    return result


@cute.jit
def precompute_cuts(cta: cutlass.Int32, scratch: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    experts = 1 + (tid >> 2)
    c = cta + (tid & 1)
    value = cutlass.Int32(0)
    if experts >= FC1_SPLIT_EXPERTS:
        aligned = aligned_f2(c, experts, True)
        value = aligned
        if (tid & 2) == 0:
            value = cut_weight(c) * (24 * experts) // CUT_TOTAL - aligned
    else:
        pieces = 16 * experts
        if (tid & 2) != 0:
            pieces = 8 * experts
        value = cut_weight(c) * pieces // CUT_TOTAL
    if (tid & 2) != 0:
        worker = cutlass.Int32(FC2_WORKERS)
        if cta < 100:
            worker = cta + (tid & 1)
        elif cta >= 148:
            worker = cta - 48 + (tid & 1)
        value = worker_fc2_start(worker, experts)
    scratch[1536 + tid] = value
    return value


@cute.jit
def _issue_piece_body(w1_a: cutlass.Int64, w2_a: cutlass.Int64, moe_a: cutlass.Uint32,
                      piece: cutlass.Int32, first1: cutlass.Int32, count1: cutlass.Int32,
                      first2: cutlass.Int32, destination: cutlass.Uint32, bar_a: cutlass.Uint32):
    moe = cute.make_tensor(cute.make_ptr(cutlass.Uint32, cutlass.Int64(moe_a), cute.AddressSpace.smem, assumed_align=4),
                           cute.make_layout((240,)))
    bar = cute.make_ptr(cutlass.Uint64, cutlass.Int64(bar_a), cute.AddressSpace.smem, assumed_align=8)
    cute.arch.mbarrier_arrive_and_expect_tx(bar, 55296)
    policy = evict_first()
    if piece < count1:
        g = first1 + piece
        expert = moe[MOE_SLOT_EXPERT + (g >> 4)].to(cutlass.Int32)
        index = (2 * expert + ((g >> 3) & 1)) * 8 + (g & 7)
        w1 = cute.make_ptr(cutlass.Uint32, w1_a, cute.AddressSpace.gmem, assumed_align=16)
        bulk_copy(destination, w1 + index * PIECE_WORDS, cutlass.Uint32(55296), bar, policy)
    else:
        g = first2 + piece - count1
        expert = moe[MOE_SLOT_EXPERT + (g >> 3)].to(cutlass.Int32)
        index = expert * 8 + (g & 7)
        w2 = cute.make_ptr(cutlass.Uint32, w2_a, cute.AddressSpace.gmem, assumed_align=16)
        bulk_copy(destination, w2 + index * PIECE_WORDS, cutlass.Uint32(55296), bar, policy)


@cute.jit
def issue_piece(w1: cute.Tensor, w2: cute.Tensor, moe: cute.Tensor,
                 piece: cutlass.Int32, first1: cutlass.Int32, count1: cutlass.Int32,
                 first2: cutlass.Int32, destination: cutlass.Uint32, bar: cute.Pointer):
    call_outlined_typed("glm52_issue_piece", _issue_piece_body,
                        cutlass.Int64(w1.iterator.toint()), cutlass.Int64(w2.iterator.toint()),
                        moe.iterator.toint().to(cutlass.Uint32), cutlass.Int32(piece), cutlass.Int32(first1),
                        cutlass.Int32(count1), cutlass.Int32(first2), cutlass.Uint32(destination),
                        bar.toint().to(cutlass.Uint32))


@cute.jit
def load_h(h: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    values = cute.make_rmem_tensor((7, 4), cutlass.Uint32)
    for i in cutlass.range_constexpr(7):
        for j in cutlass.range_constexpr(4):
            values[i, j] = cutlass.Uint32(0)
        if tid + 256 * i < 1728:
            a, b, c, d = volatile_vector(h.iterator + 4 * (tid + 256 * i))
            values[i, 0], values[i, 1], values[i, 2], values[i, 3] = a, b, c, d
    return values


@cute.jit
def finish_h(h: cute.Tensor, values: cute.Tensor, region: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    waiting = cutlass.Boolean(True)
    while waiting:
        full = cutlass.Boolean(True)
        for i in cutlass.range_constexpr(7):
            for j in cutlass.range_constexpr(4):
                full = full & (values[i, j] != cutlass.Uint32(0x80000000))
        if cute.arch.vote_all_sync(full):
            waiting = cutlass.Boolean(False)
        else:
            for i in cutlass.range_constexpr(7):
                complete = cutlass.Boolean(True)
                for j in cutlass.range_constexpr(4):
                    complete = complete & (values[i, j] != cutlass.Uint32(0x80000000))
                if not complete:
                    a, b, c, d = volatile_vector(h.iterator + 4 * (tid + 256 * i))
                    values[i, 0], values[i, 1], values[i, 2], values[i, 3] = a, b, c, d
    for i in cutlass.range_constexpr(7):
        vector = tid + 256 * i
        dst = cutlass.Int32(0)
        if vector < 1536:
            token, c16 = vector // 192, vector % 192
            dst = 41472 + (c16 >> 2) * 128 + token * 16 + (((c16 & 3) ^ ((token >> 1) & 3)) << 2)
        else:
            dst = 41472 + 6144 + 4 * (vector - 1536)
        if vector < 1728:
            store_shared_v4((region.iterator + dst).toint().to(cutlass.Uint32),
                            values[i, 0], values[i, 1], values[i, 2], values[i, 3])


@cute.jit
def issue_fc1_piece(weights: cute.Tensor, moe: cute.Tensor, piece: cutlass.Int32,
                    first: cutlass.Int32, destination: cutlass.Uint32, bar: cute.Pointer):
    g = first + piece
    expert = moe[MOE_SLOT_EXPERT + (g >> 4)].to(cutlass.Int32)
    index = (2 * expert + ((g >> 3) & 1)) * 8 + (g & 7)
    cute.arch.mbarrier_arrive_and_expect_tx(bar, 55296)
    bulk_piece(destination, weights.iterator + index * PIECE_WORDS, bar)


@cute.jit
def stage_h_scales(region: cute.Tensor, scratch: cute.Tensor,
                     tmem: cutlass.Uint32, work: cutlass.Boolean):
    tid, _, _ = cute.arch.thread_idx()
    warp = tid // 32
    if work:
        for iteration in cutlass.range_constexpr(2):
            vector = tid + 256 * iteration
            if vector < 384:
                pair, token = vector // 8, vector % 8
                src = 41472 + 6144 + token * 96 + 2 * pair
                store_shared_v4((scratch.iterator + 512 + 4 * vector).toint().to(cutlass.Uint32),
                                region[src], cutlass.Uint32(0), region[src + 1], cutlass.Uint32(0))
        proxy_fence()
        cute.arch.sync_threads()
        if warp == 1:
            tc_after()
            with cute.arch.elect_one():
                for pair in cutlass.range_constexpr(48):
                    copy_scales(tmem + cutlass.Uint32(256 + 4 * pair),
                                (scratch.iterator + 512 + 32 * pair).toint().to(cutlass.Uint32), True)
            cute.arch.sync_warp()


@cute.jit
def _fc1_piece_body(tmem: cutlass.Uint32, ring_base: cutlass.Uint32,
                    hbq: cutlass.Uint32, sfa: cutlass.Uint32, sfb: cutlass.Uint32,
                    accumulate_first: cutlass.Uint32, bar: cutlass.Uint32):
    tc_after()
    with cute.arch.elect_one():
        for step in cutlass.range_constexpr(12):
            copy_scales(sfa + cutlass.Uint32(4 * step), ring_base + cutlass.Uint32(49152 + 512 * step))
        for step in cutlass.range_constexpr(12):
            accumulate = cutlass.Uint32(1)
            if (step == 0) & (accumulate_first == cutlass.Uint32(0)):
                accumulate = cutlass.Uint32(0)
            mma_nvfp4(tmem, ring_base + cutlass.Uint32((step >> 1) * 8192 + 32 * (step & 1)),
                      hbq + cutlass.Uint32((step >> 1) * 512 + 32 * (step & 1)),
                      sfa + cutlass.Uint32(4 * step), sfb + cutlass.Uint32(2 * step), accumulate)
        commit_mma_addr(bar)
    cute.arch.sync_warp()


@cute.jit
def fc1_piece(tmem: cutlass.Uint32, ring_base: cutlass.Uint32,
                hbq: cutlass.Uint32, sfa: cutlass.Uint32, sfb: cutlass.Uint32,
                accumulate_first: cutlass.Boolean, bar: cute.Pointer):
    flag = cutlass.Uint32(0)
    if accumulate_first:
        flag = cutlass.Uint32(1)
    call_outlined("glm52_fc1_piece", _fc1_piece_body, tmem, ring_base, hbq, sfa, sfb, flag,
                  bar.toint().to(cutlass.Uint32))


@cute.jit
def fc1_produce(weights: cute.Tensor, moe: cute.Tensor, region: cute.Tensor,
                 scratch: cute.Tensor, bars: cute.Tensor, tmem: cutlass.Uint32,
                 p1: cute.Tensor, first: cutlass.Int32, end: cutlass.Int32,
                 sparse: cutlass.Boolean, weights2=None, first2=None, end2=None,
                 scale_staged: cutlass.Constexpr = False,
                 first_in_flight: cutlass.Boolean = False):
    tid, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    warp, lane = tid // 32, tid % 32
    base = region.iterator.toint().to(cutlass.Uint32)
    hb = base + cutlass.Uint32(165888)
    if cutlass.const_expr(not scale_staged):
        stage_h_scales(region, scratch, tmem, first < end)
    phase = cutlass.Int32(0)
    run = cutlass.Int32(0)
    for g in range(first, end):
        piece = g - first
        rs = piece % 3
        ring_base = base + (rs * 55296).to(cutlass.Uint32)
        cute.arch.mbarrier_wait(bars.iterator + rs, (piece // 3) & 1)
        if not (first_in_flight & (piece == 0)):
            if warp == 1:
                fc1_piece(tmem, ring_base, hb + (6 * (g & 7) * 512).to(cutlass.Uint32),
                          tmem + (64 + 48 * rs).to(cutlass.Uint32),
                          tmem + (256 + 24 * (g & 7)).to(cutlass.Uint32),
                          (g != first) & ((g & 7) != 0), bars.iterator + 3)
        cute.arch.mbarrier_wait(bars.iterator + 3, phase)
        phase ^= 1
        cute.arch.sync_threads()
        tc_after()
        if cutlass.const_expr(weights2 is None):
            if (tid == 0) & (piece + 3 < end - first):
                issue_fc1_piece(weights, moe, piece + 3, first, ring_base, bars.iterator + rs)
        else:
            if (tid == 0) & (piece + 3 < end - first + end2 - first2):
                issue_piece(weights, weights2, moe, piece + 3, first, end - first,
                            first2, ring_base, bars.iterator + rs)
        if (g + 1 == end) | ((g & 7) == 7):
            if warp < 4:
                words = load_tmem8(tmem + ((32 * warp).to(cutlass.Uint32) << 16))
                feature = 2 * (lane & 7) + (lane >> 4)
                gate = (lane >> 3) & 1
                mask = cutlass.Uint32(255)
                if sparse:
                    packed = moe[MOE_SLOT_MASK + ((g >> 4) >> 2)]
                    mask = (packed >> (8 * ((g >> 4) & 3))) & cutlass.Uint32(255)
                for token in cutlass.range_constexpr(8):
                    if (mask & (cutlass.Uint32(1) << token)) != cutlass.Uint32(0):
                        index = (cta * 2 + run) * 1024 + token * 128 + 32 * warp + 2 * feature + gate
                        p1[index] = published_word(words[token])
            run += 1
            tc_before()
            cute.arch.sync_threads()
            tc_after()
    return phase


@dsl_user_op
def load_tmem3(address: cutlass.Uint32, *, loc=None, ip=None):
    words = []
    for offset in (0, 8, 16):
        column = as_ptr(6, address + cutlass.Uint32(offset), loc, ip)
        words.append(nvvm.tcgen05_ld(T.i32(), nvvm.Tcgen05LdStShape.SHAPE_32X32B, column, loc=loc, ip=ip))
    nvvm.tcgen05_wait(nvvm.Tcgen05WaitKind.LOAD, loc=loc, ip=ip)
    return tuple(cutlass.Uint32(w) for w in words)


@cute.jit
def slot_mask(moe: cute.Tensor, slot: cutlass.Int32):
    return (moe[MOE_SLOT_MASK + (slot >> 2)] >> (8 * (slot & 3))) & cutlass.Uint32(255)


@cute.jit
def prefetch_fc2_lists(packed: cute.Tensor, alpha1: cute.Tensor, alpha2: cute.Tensor, cut_word: cutlass.Int32):
    tid, _, _ = cute.arch.thread_idx()
    lane = tid % 32
    first = cute.arch.shuffle_sync(cut_word, (lane & 28) | 2)
    end = cute.arch.shuffle_sync(cut_word, (lane & 28) | 3)
    experts = 1 + (tid >> 2)
    words = cute.make_rmem_tensor((8,), cutlass.Uint32)
    for k in cutlass.range_constexpr(8):
        words[k] = cutlass.Uint32(0)
    groups = cutlass.Int32(0)
    if first < end:
        groups = ((end - 1) >> 3) - (first >> 3) + 1
    for k in cutlass.range_constexpr(2):
        slot = (tid & 3) + 4 * k
        if slot < 2 * groups:
            index = experts * (experts - 1) + 2 * ((first >> 3) + (slot >> 1)) + (slot & 1)
            words[3 * k] = packed[index, 0].to(cutlass.Uint32)
            words[3 * k + 1] = packed[index, 1].to(cutlass.Uint32)
            words[3 * k + 2] = packed[index, 2].to(cutlass.Uint32)
    words[6] = alpha1[tid].bitcast(cutlass.Uint32)
    words[7] = alpha2[tid].bitcast(cutlass.Uint32)
    return words


@cute.jit
def _store_list(scratch: cute.Tensor, slot: cutlass.Int32, a: cutlass.Uint32, b: cutlass.Uint32, c: cutlass.Uint32):
    count = a & cutlass.Uint32(15)
    scratch[3584 + 9 * slot] = count
    for q in cutlass.range_constexpr(8):
        if q < count:
            bit = 4 + 9 * q
            word, shift = bit >> 5, bit & 31
            low, high = a, b
            if cutlass.const_expr(word == 1):
                low, high = b, c
            elif cutlass.const_expr(word == 2):
                low, high = c, c
            code = low.to(cutlass.Uint32) >> cutlass.Uint32(shift)
            if cutlass.const_expr(shift > 23):
                code |= high.to(cutlass.Uint32) << cutlass.Uint32(32 - shift)
            scratch[3584 + 9 * slot + 1 + q] = code & cutlass.Uint32(511)


@cute.jit
def store_prefetched_fc2_lists(scratch: cute.Tensor, moe: cute.Tensor, words: cute.Tensor,
                               first: cutlass.Int32, end: cutlass.Int32):
    tid, _, _ = cute.arch.thread_idx()
    groups = cutlass.Int32(0)
    if first < end:
        groups = ((end - 1) >> 3) - (first >> 3) + 1
    if 1 + (tid >> 2) == moe[MOE_E].to(cutlass.Int32):
        for k in cutlass.range_constexpr(2):
            slot = (tid & 3) + 4 * k
            if slot < 2 * groups:
                _store_list(scratch, slot, words[3 * k], words[3 * k + 1], words[3 * k + 2])
    for g in cutlass.range_constexpr(4):
        if g < groups:
            if moe[MOE_SLOT_EXPERT + (first >> 3) + g].to(cutlass.Int32) == tid:
                scratch[3584 + 72 + g] = words[6]
                scratch[3584 + 76 + g] = words[7]


@cute.jit
def poll_runs(p1: cute.Tensor, scratch: cute.Tensor, list_offset: cutlass.Int32,
               offset: cutlass.Int32, count: cutlass.Int32, n: cutlass.Constexpr):
    words = cute.make_rmem_tensor((n, 8), cutlass.Uint32)
    for q in cutlass.range_constexpr(n):
        for j in cutlass.range_constexpr(8):
            words[q, j] = cutlass.Uint32(0)
        if q < count:
            code = scratch[list_offset + 1 + q].to(cutlass.Int32)
            a, b, c, d = volatile_vector(p1.iterator + code * 1024 + offset)
            e, f, g, h = volatile_vector(p1.iterator + code * 1024 + offset + 4)
            words[q, 0], words[q, 1], words[q, 2], words[q, 3] = a, b, c, d
            words[q, 4], words[q, 5], words[q, 6], words[q, 7] = e, f, g, h
    sums = cute.make_rmem_tensor((8,), cutlass.Float32)
    waiting = cutlass.Boolean(True)
    while waiting:
        full = cutlass.Boolean(True)
        for q in cutlass.range_constexpr(n):
            for j in cutlass.range_constexpr(8):
                full = full & (words[q, j] != cutlass.Uint32(0x80000000))
        done = cute.arch.vote_all_sync(full)
        for j in cutlass.range_constexpr(8):
            sums[j] = cutlass.Float32(0)
        for q in cutlass.range_constexpr(n):
            for j in cutlass.range_constexpr(8):
                sums[j] += bits_f32(words[q, j])
        if done:
            waiting = cutlass.Boolean(False)
        else:
            for q in cutlass.range_constexpr(n):
                for vector in cutlass.range_constexpr(2):
                    ready = cutlass.Boolean(True)
                    for j in cutlass.range_constexpr(4):
                        ready = ready & (words[q, 4 * vector + j] != cutlass.Uint32(0x80000000))
                    if not ready:
                        code = scratch[list_offset + 1 + q].to(cutlass.Int32)
                        a, b, c, d = volatile_vector(p1.iterator + code * 1024 + offset + 4 * vector)
                        words[q, 4 * vector], words[q, 4 * vector + 1] = a, b
                        words[q, 4 * vector + 2], words[q, 4 * vector + 3] = c, d
    return sums


@cute.jit
def activation(up: cutlass.Float32, gate: cutlass.Float32, alpha: cutlass.Float32):
    return fma_rn(up, alpha, cutlass.Float32(0)) * (gate * cute.arch.rcp_approx(
        1.0 + cute.arch.exp2(-(gate * (1.4426950408889634 * alpha)))))


@cute.jit
def quant_scale(amax: cutlass.Float32, alpha: cutlass.Float32, in_scale: cutlass.Float32):
    sf = e4m3_pair(in_scale * (amax * alpha * 0.16666667), cutlass.Float32(0)) & cutlass.Uint32(255)
    scale = e4m3_float(sf)
    reciprocal = cutlass.Float32(0)
    if scale != 0.0:
        reciprocal = cute.arch.rcp_approx(scale * cute.arch.rcp_approx(in_scale))
    return sf, reciprocal


@cute.jit
def make_fc2_operand(p1: cute.Tensor, region: cute.Tensor, scratch: cute.Tensor,
                      group: cutlass.Int32, mask: cutlass.Uint32,
                      alpha: cutlass.Float32, in_scale: cutlass.Float32,
                      sparse: cutlass.Boolean):
    tid, _, _ = cute.arch.thread_idx()
    lane = tid & 31
    tokens = cute.arch.popc(mask)
    use_sparse_wide = sparse & (tokens >= 1) & (tokens <= 2)
    if tid < 32:
        vector_address = region.iterator.toint().to(cutlass.Uint32) + cutlass.Uint32(16 * tid)
        if use_sparse_wide:
            zero_shared_vector(vector_address + cutlass.Uint32(198656))
        zero_shared_vector(vector_address + cutlass.Uint32(199168))
    cute.arch.sync_threads()
    byte_view = cute.make_tensor(cute.recast_ptr(region.iterator, dtype=cutlass.Uint8), cute.make_layout((200704,)))
    half_view = cute.make_tensor(cute.recast_ptr(region.iterator, dtype=cutlass.Uint16), cute.make_layout((100352,)))
    if use_sparse_wide:
        active_group = tid >> 7
        if active_group < tokens:
            token_bits = mask
            if active_group != 0:
                token_bits &= token_bits - cutlass.Uint32(1)
            token = cutlass.Int32(0)
            while ((token_bits >> token) & cutlass.Uint32(1)) == cutlass.Uint32(0):
                token += 1
            feature = tid & 127
            tile = feature >> 6
            list_offset = 3584 + 18 * group + 9 * tile
            count = scratch[list_offset].to(cutlass.Int32)
            up, gate = cutlass.Float32(0), cutlass.Float32(0)
            for q0 in range(0, count, 4):
                words = cute.make_rmem_tensor((4, 2), cutlass.Uint32)
                for q in cutlass.range_constexpr(4):
                    words[q, 0], words[q, 1] = cutlass.Uint32(0), cutlass.Uint32(0)
                    if q0 + q < count:
                        code = scratch[list_offset + 1 + q0 + q].to(cutlass.Int32)
                        a, b = volatile_pair(p1.iterator + code * 1024 + token * 128 + 2 * (feature & 63))
                        words[q, 0], words[q, 1] = a, b
                waiting = cutlass.Boolean(True)
                while waiting:
                    full = cutlass.Boolean(True)
                    for q in cutlass.range_constexpr(4):
                        full = full & (words[q, 0] != cutlass.Uint32(0x80000000)) & (words[q, 1] != cutlass.Uint32(0x80000000))
                    if cute.arch.vote_all_sync(full):
                        waiting = cutlass.Boolean(False)
                    else:
                        for q in cutlass.range_constexpr(4):
                            if (words[q, 0] == cutlass.Uint32(0x80000000)) | (words[q, 1] == cutlass.Uint32(0x80000000)):
                                code = scratch[list_offset + 1 + q0 + q].to(cutlass.Int32)
                                a, b = volatile_pair(p1.iterator + code * 1024 + token * 128 + 2 * (feature & 63))
                                words[q, 0], words[q, 1] = a, b
                for q in cutlass.range_constexpr(4):
                    if q0 + q < count:
                        up += bits_f32(words[q, 0])
                        gate += bits_f32(words[q, 1])
            gv = activation(up, gate, alpha)
            amax = cute.math.abs(gv)
            for step in cutlass.range_constexpr(4):
                amax = max_num(amax, cute.arch.shuffle_sync_bfly(amax, 8 >> step))
            sf, reciprocal = quant_scale(amax, alpha, in_scale)
            scaled = (gv * reciprocal) * alpha
            adjacent = cute.arch.shuffle_sync_bfly(scaled, 1)
            if (feature & 1) == 0:
                byte_view[198656 + token * 64 + (((feature >> 5) ^ ((token >> 1) & 3)) << 4) + ((feature >> 1) & 15)] = fp4_pair(scaled, adjacent).to(cutlass.Uint8)
            if (feature & 15) == 0:
                b = feature >> 4
                sf_offset = b
                if b >= 4:
                    sf_offset = 4 + b
                byte_view[199168 + 16 * token + sf_offset] = sf.to(cutlass.Uint8)
    else:
        token, b, sub = tid // 32, (tid // 4) % 8, tid % 4
        active = (not sparse) | (((mask >> token) & cutlass.Uint32(1)) != cutlass.Uint32(0))
        if active:
            tile = b >> 2
            list_offset = 3584 + 18 * group + 9 * tile
            count = scratch[list_offset].to(cutlass.Int32)
            max_count = scratch[3584 + 18 * group]
            if scratch[3584 + 18 * group + 9] > max_count:
                max_count = scratch[3584 + 18 * group + 9]
            offset = token * 128 + 32 * (b & 3) + 8 * sub
            sums = cute.make_rmem_tensor((8,), cutlass.Float32)
            if max_count > 4:
                values = poll_runs(p1, scratch, list_offset, offset, count, 8)
                for j in cutlass.range_constexpr(8):
                    sums[j] = values[j]
            else:
                values = poll_runs(p1, scratch, list_offset, offset, count, 4)
                for j in cutlass.range_constexpr(8):
                    sums[j] = values[j]
            gv = cute.make_rmem_tensor((4,), cutlass.Float32)
            for j in cutlass.range_constexpr(4):
                gv[j] = activation(sums[2 * j], sums[2 * j + 1], alpha)
            amax = max_num(max_num(cute.math.abs(gv[0]), cute.math.abs(gv[1])), max_num(cute.math.abs(gv[2]), cute.math.abs(gv[3])))
            for step in cutlass.range_constexpr(2):
                amax = max_num(amax, cute.arch.shuffle_sync_bfly(amax, 1 << step))
            sf, reciprocal = quant_scale(amax, alpha, in_scale)
            q01 = fp4_pair((gv[0] * reciprocal) * alpha, (gv[1] * reciprocal) * alpha)
            q23 = fp4_pair((gv[2] * reciprocal) * alpha, (gv[3] * reciprocal) * alpha)
            half_view[(198656 + token * 64 + (((b >> 1) ^ ((token >> 1) & 3)) << 4) + 8 * (b & 1) + 2 * sub) // 2] = (q01 | (q23 << 8)).to(cutlass.Uint16)
            if sub == 0:
                sf_offset = b
                if b >= 4:
                    sf_offset = 4 + b
                byte_view[199168 + 16 * token + sf_offset] = sf.to(cutlass.Uint8)
        else:
            half_view[(198656 + token * 64 + (((b >> 1) ^ ((token >> 1) & 3)) << 4) + 8 * (b & 1) + 2 * sub) // 2] = cutlass.Uint16(0)
            if sub == 0:
                sf_offset = b
                if b >= 4:
                    sf_offset = 4 + b
                byte_view[199168 + 16 * token + sf_offset] = cutlass.Uint8(0)
    proxy_fence()
    cute.arch.sync_threads()


@cute.jit
def _fc2_piece_body(tmem: cutlass.Uint32, acc: cutlass.Uint32, ring_base: cutlass.Uint32, fc2b: cutlass.Uint32,
                    fc2sfb: cutlass.Uint32, sfa: cutlass.Uint32, bar: cutlass.Uint32):
    tc_after()
    with cute.arch.elect_one():
        copy_scales(tmem + cutlass.Uint32(448), fc2sfb)
        for step in cutlass.range_constexpr(12):
            copy_scales(sfa + cutlass.Uint32(4 * step), ring_base + cutlass.Uint32(49152 + 512 * step))
        for step in cutlass.range_constexpr(12):
            mma_nvfp4(acc + cutlass.Uint32(8 * (step >> 1)), ring_base + cutlass.Uint32((step >> 1) * 8192 + 32 * (step & 1)),
                      fc2b + cutlass.Uint32(32 * (step & 1)), sfa + cutlass.Uint32(4 * step),
                      tmem + cutlass.Uint32(448 + 2 * (step & 1)), cutlass.Uint32(step & 1))
        commit_mma_addr(bar)
    cute.arch.sync_warp()


@cute.jit
def fc2_piece(tmem: cutlass.Uint32, ring_base: cutlass.Uint32, fc2b: cutlass.Uint32,
              fc2sfb: cutlass.Uint32, sfa: cutlass.Uint32, bar: cute.Pointer, acc=None):
    """One shared function for the real pieces and warm_moe, so the warm-up fetches the code the first real FC2
    piece runs."""
    if cutlass.const_expr(acc is None):
        acc = tmem
    call_outlined("glm52_fc2_piece", _fc2_piece_body, tmem, cutlass.Uint32(acc), ring_base, fc2b, fc2sfb, sfa,
                  bar.toint().to(cutlass.Uint32))


@cute.jit
def fc2_produce(w1: cute.Tensor, w2: cute.Tensor, moe: cute.Tensor,
                 region: cute.Tensor, scratch: cute.Tensor, bars: cute.Tensor,
                 tmem: cutlass.Uint32, p1: cute.Tensor, fc2: cute.Tensor,
                 alpha2: cute.Tensor, first1: cutlass.Int32, end1: cutlass.Int32,
                 first2: cutlass.Int32, end2: cutlass.Int32, phase: cutlass.Int32,
                 sparse: cutlass.Boolean):
    tid, _, _ = cute.arch.thread_idx()
    warp, lane = tid >> 5, tid & 31
    base = region.iterator.toint().to(cutlass.Uint32)
    piece = end1 - first1
    j = first2
    group = cutlass.Int32(0)
    in_scale = alpha2[256]
    while j < end2:
        slot = j >> 3
        j_end = 8 * slot + 8
        if end2 < j_end:
            j_end = end2
        mask = slot_mask(moe, slot)
        alpha_up = bits_f32(scratch[3584 + 72 + group])
        alpha_down = bits_f32(scratch[3584 + 76 + group])
        make_fc2_operand(p1, region, scratch, group, mask, alpha_up, in_scale, sparse)
        k = cutlass.Int32(0)
        if warp == 1:
            rs0 = piece % 3
            cute.arch.mbarrier_wait(bars.iterator + rs0, (piece // 3) & 1)
            fc2_piece(tmem, base + (rs0 * 55296).to(cutlass.Uint32), base + cutlass.Uint32(198656),
                      base + cutlass.Uint32(199168), tmem + (64 + 48 * rs0).to(cutlass.Uint32), bars.iterator + 3)
        while j < j_end:
            rs = piece % 3
            ring_base = base + (rs * 55296).to(cutlass.Uint32)
            acc = tmem + (256 * (k & 1)).to(cutlass.Uint32)
            cute.arch.mbarrier_wait(bars.iterator + 3, phase)
            phase ^= 1
            if piece + 3 < end1 - first1 + end2 - first2:
                cute.arch.sync_threads()
            tc_after()
            if (tid == 0) & (piece + 3 < end1 - first1 + end2 - first2):
                issue_piece(w1, w2, moe, piece + 3, first1, end1 - first1, first2, ring_base, bars.iterator + rs)
            if (j + 1 < j_end) & (warp == 1):
                nrs = (piece + 1) % 3
                cute.arch.mbarrier_wait(bars.iterator + nrs, ((piece + 1) // 3) & 1)
                fc2_piece(tmem, base + (nrs * 55296).to(cutlass.Uint32), base + cutlass.Uint32(198656),
                          base + cutlass.Uint32(199168), tmem + (64 + 48 * nrs).to(cutlass.Uint32), bars.iterator + 3,
                          tmem + (256 * ((k + 1) & 1)).to(cutlass.Uint32))
            quarter, half = warp & 3, warp >> 2
            columns = acc + (24 * half).to(cutlass.Uint32) + ((32 * quarter).to(cutlass.Uint32) << 16)
            for token in cutlass.range(8, unroll=1):
                if ((mask >> token.to(cutlass.Uint32)) & cutlass.Uint32(1)) != cutlass.Uint32(0):
                    values = load_tmem3(columns + token.to(cutlass.Uint32))
                    for i in cutlass.range_constexpr(3):
                        value = (bits_f32(values[i]) + 0.0) * alpha_down
                        x1 = cute.arch.shuffle_sync_down(value, 8)
                        x2 = cute.arch.shuffle_sync_down(value, 16)
                        x3 = cute.arch.shuffle_sync_down(value, 24)
                        if lane < 8:
                            col = 128 * (6 * (j & 7) + 3 * half + i) + 32 * quarter + 4 * (lane & 7)
                            offset = (slot * 8 + token) * 3072 + col // 2
                            store_pair(fc2.iterator + offset, published_word(bf16_pair(value, x1)), published_word(bf16_pair(x2, x3)))
            tc_before()
            cute.arch.sync_threads()
            tc_after()
            piece += 1
            j += 1
            k += 1
        group += 1
    return phase


@cute.jit
def warm_moe(region: cute.Tensor, tmem: cutlass.Uint32, bar: cute.Pointer):
    tid, _, _ = cute.arch.thread_idx()
    warp = tid // 32
    base = region.iterator.toint().to(cutlass.Uint32)
    proxy_fence()
    tc_before()
    cute.arch.sync_threads()
    tc_after()
    if warp == 1:
        fc1_piece(tmem, base, base + cutlass.Uint32(165888), tmem + cutlass.Uint32(64),
                    tmem + cutlass.Uint32(256), cutlass.Boolean(False), bar)
    cute.arch.mbarrier_wait(bar, 0)
    cute.arch.sync_threads()
    tc_after()
    if warp == 1:
        fc2_piece(tmem, base + cutlass.Uint32(55296), base + cutlass.Uint32(198656), base + cutlass.Uint32(199168),
                  tmem + cutlass.Uint32(112), bar)
    cute.arch.mbarrier_wait(bar, 1)
    tc_before()
    cute.arch.sync_threads()
    tc_after()


@cute.jit
def owner_stage(unit: cutlass.Int32, rank: cutlass.Int32, clear_target: cutlass.Uint32,
                clear_first: cutlass.Boolean, moe: cute.Tensor,
                shared_output: cute.Tensor, fc2: cute.Tensor,
                clear_counter: cute.Tensor, h2_multicast: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    if tid == 128:
        ready = acquire_word(clear_counter.iterator)
        while ready < clear_target:
            ready = acquire_word(clear_counter.iterator)
    shard = cute.make_rmem_tensor((4,), cutlass.Uint32)
    slots = cute.make_rmem_tensor((8,), cutlass.Int32)
    for j in cutlass.range_constexpr(4):
        shard[j] = cutlass.Uint32(0)
    if tid < 128:
        token, vector = tid // 16, tid % 16
        pair_col = 64 * unit + 4 * vector
        weights = cute.make_rmem_tensor((8,), cutlass.Float32)
        payload = cute.make_rmem_tensor((8, 4), cutlass.Uint32)
        for k in cutlass.range_constexpr(8):
            slots[k] = moe[MOE_SLOT_OF_TK + 8 * token + k].to(cutlass.Int32)
            weights[k] = bits_f32(moe[MOE_WEIGHT + 8 * token + k])
            a, b, c, d = volatile_vector(fc2.iterator + (slots[k] * 8 + token) * 3072 + pair_col)
            payload[k, 0], payload[k, 1], payload[k, 2], payload[k, 3] = a, b, c, d
        waiting = cutlass.Boolean(True)
        while waiting:
            full = cutlass.Boolean(True)
            for k in cutlass.range_constexpr(8):
                for j in cutlass.range_constexpr(4):
                    full = full & (payload[k, j] != cutlass.Uint32(0x80000000))
            if cute.arch.vote_all_sync(full):
                waiting = cutlass.Boolean(False)
            else:
                for k in cutlass.range_constexpr(8):
                    complete = cutlass.Boolean(True)
                    for j in cutlass.range_constexpr(4):
                        complete = complete & (payload[k, j] != cutlass.Uint32(0x80000000))
                    if not complete:
                        a, b, c, d = volatile_vector(fc2.iterator + (slots[k] * 8 + token) * 3072 + pair_col)
                        payload[k, 0], payload[k, 1], payload[k, 2], payload[k, 3] = a, b, c, d
        if clear_first:
            for k in cutlass.range_constexpr(8):
                empty = cutlass.Uint32(0x80000000)
                store_vector(fc2.iterator + (slots[k] * 8 + token) * 3072 + pair_col, empty, empty, empty, empty)
        routed = cute.make_rmem_tensor((8,), cutlass.Float32)
        for j in cutlass.range_constexpr(8):
            routed[j] = cutlass.Float32(0)
        for k in cutlass.range_constexpr(8):
            for j in cutlass.range_constexpr(4):
                routed[2 * j] = fma_rn(weights[k], bf16_from_word(payload[k, j], False), routed[2 * j])
                routed[2 * j + 1] = fma_rn(weights[k], bf16_from_word(payload[k, j], True), routed[2 * j + 1])
        shared_words = cute.make_rmem_tensor((4,), cutlass.Uint32)
        s0, s1, s2, s3 = shared_vector(shared_output.iterator + token * 64 + 4 * vector)
        shared_words[0], shared_words[1], shared_words[2], shared_words[3] = s0, s1, s2, s3
        for j in cutlass.range_constexpr(4):
            shared = shared_words[j]
            s0, s1 = bf16_from_word(shared, False), bf16_from_word(shared, True)
            shard[j] = published_word(bf16_pair(s0 + round_bf16(routed[2 * j]), s1 + round_bf16(routed[2 * j + 1])))
    cute.arch.sync_threads()
    if tid < 128:
        token, vector = tid // 16, tid % 16
        pair_col = 64 * unit + 4 * vector
        store_vector(h2_multicast.iterator + (token * 16 + rank) * 3072 + pair_col,
                     shard[0], shard[1], shard[2], shard[3])
        if not clear_first:
            for k in cutlass.range_constexpr(8):
                empty = cutlass.Uint32(0x80000000)
                store_vector(fc2.iterator + (slots[k] * 8 + token) * 3072 + pair_col, empty, empty, empty, empty)


@cute.jit
def issue_gu(unit: cutlass.Int32, weights: cute.Tensor, region: cute.Tensor, bar: cute.Pointer):
    ft, quarter = unit // 4, unit % 4
    base = region.iterator.toint().to(cutlass.Uint32)
    cute.arch.mbarrier_arrive_and_expect_tx(bar, 98304)
    policy = evict_first()
    bulk_copy(base, weights.iterator + (ft * 96 + 24 * quarter) * 512, cutlass.Uint32(49152), bar, policy)
    bulk_copy(base + cutlass.Uint32(49152), weights.iterator + ((8 + ft) * 96 + 24 * quarter) * 512, cutlass.Uint32(49152), bar, policy)


@cute.jit
def shared_gu(unit: cutlass.Int32, hp: cute.Tensor, region: cute.Tensor,
                red: cute.Tensor, sh: cute.Tensor, bar: cute.Pointer):
    tid, _, _ = cute.arch.thread_idx()
    warp, lane = tid >> 5, tid & 31
    ft, quarter = unit // 4, unit % 4
    b = cute.make_rmem_tensor((3, 8), cutlass.Uint32)
    for i in cutlass.range_constexpr(3):
        chunk = warp * 3 + i
        offset = (lane >> 2) * 3072 + 768 * quarter + chunk * 32 + (lane & 3) * 8
        a, c, d, e = volatile_vector(hp.iterator + offset)
        f, g, h, word7 = volatile_vector(hp.iterator + offset + 4)
        b[i, 0], b[i, 1], b[i, 2], b[i, 3] = a, c, d, e
        b[i, 4], b[i, 5], b[i, 6], b[i, 7] = f, g, h, word7
    waiting = cutlass.Boolean(True)
    while waiting:
        full = cutlass.Boolean(True)
        for i in cutlass.range_constexpr(3):
            for j in cutlass.range_constexpr(8):
                full = full & (b[i, j] != cutlass.Uint32(0x80000000))
        if cute.arch.vote_all_sync(full):
            waiting = cutlass.Boolean(False)
        else:
            for i in cutlass.range_constexpr(3):
                chunk = warp * 3 + i
                offset = (lane >> 2) * 3072 + 768 * quarter + chunk * 32 + (lane & 3) * 8
                for vector in cutlass.range_constexpr(2):
                    ready = cutlass.Boolean(True)
                    for j in cutlass.range_constexpr(4):
                        ready = ready & (b[i, 4 * vector + j] != cutlass.Uint32(0x80000000))
                    if not ready:
                        a, c, d, e = volatile_vector(hp.iterator + offset + 4 * vector)
                        b[i, 4 * vector], b[i, 4 * vector + 1] = a, c
                        b[i, 4 * vector + 2], b[i, 4 * vector + 3] = d, e
    cute.arch.mbarrier_wait(bar, 0)
    acc = cute.make_rmem_tensor((2, 4), cutlass.Float32)
    a = cute.make_rmem_tensor((16,), cutlass.Uint32)
    for tile in cutlass.range_constexpr(2):
        for r in cutlass.range_constexpr(4):
            acc[tile, r] = cutlass.Float32(0)
    for i in cutlass.range_constexpr(3):
        chunk = warp * 3 + i
        for tile in cutlass.range_constexpr(2):
            for piece in cutlass.range_constexpr(4):
                for j in cutlass.range_constexpr(4):
                    a[4 * piece + j] = region[(tile * 24 + chunk) * 512 + 128 * piece + 4 * lane + j]
            for step in cutlass.range_constexpr(4):
                index = (step // 2) * 4 + (step % 2) * 2
                c0, c1, c2, c3 = mma_bf16(acc[tile, 0], acc[tile, 1], acc[tile, 2], acc[tile, 3],
                                         a[index], a[index + 8], a[index + 1], a[index + 9], b[i, 2 * step], b[i, 2 * step + 1])
                acc[tile, 0], acc[tile, 1], acc[tile, 2], acc[tile, 3] = c0, c1, c2, c3
    for tile in cutlass.range_constexpr(2):
        for r in cutlass.range_constexpr(4):
            red[((warp * 2 + tile) * 32 + lane) * 4 + r] = acc[tile, r]
    cute.arch.sync_threads()
    tile, owner, r = tid // 128, (tid // 4) % 32, tid % 4
    value = cutlass.Float32(0)
    for w in cutlass.range_constexpr(8):
        value += red[((w * 2 + tile) * 32 + owner) * 4 + r]
    row = 16 * tile + (owner >> 2) + (r // 2) * 8
    token = 2 * (owner & 3) + (r & 1)
    red[2048 + row * 8 + token] = value
    cute.arch.sync_threads()
    store_word(sh.iterator + ((quarter * 8 + ft) * 32) * 8 + tid, published_word(f32_bits(red[2048 + tid])))


@cute.jit
def owner_shared(unit: cutlass.Int32, sh: cute.Tensor, weights: cute.Tensor,
                   moe: cute.Tensor, red: cute.Tensor, output: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    warp, lane = tid >> 5, tid & 31
    wa = cute.make_rmem_tensor((2, 16), cutlass.Uint32)
    for chunk in cutlass.range_constexpr(2):
        for piece in cutlass.range_constexpr(4):
            a, b, c, d = cached_vector(weights.iterator + (unit * 8 + warp) * 1024 + 512 * chunk + 128 * piece + 4 * lane)
            wa[chunk, 4 * piece], wa[chunk, 4 * piece + 1] = a, b
            wa[chunk, 4 * piece + 2], wa[chunk, 4 * piece + 3] = c, d
    feature, half = tid // 2, tid % 2
    ft, row = feature // 16, feature % 16
    payload = cute.make_rmem_tensor((8, 4), cutlass.Uint32)
    for i in cutlass.range_constexpr(8):
        offset = (((i // 2) * 8 + ft) * 32 + row + 16 * (i % 2)) * 8 + 4 * half
        a, b, c, d = volatile_vector(sh.iterator + offset)
        payload[i, 0], payload[i, 1], payload[i, 2], payload[i, 3] = a, b, c, d
    waiting = cutlass.Boolean(True)
    while waiting:
        full = cutlass.Boolean(True)
        for i in cutlass.range_constexpr(8):
            for j in cutlass.range_constexpr(4):
                full = full & (payload[i, j] != cutlass.Uint32(0x80000000))
        if cute.arch.vote_all_sync(full):
            waiting = cutlass.Boolean(False)
        else:
            for i in cutlass.range_constexpr(8):
                ready = cutlass.Boolean(True)
                for j in cutlass.range_constexpr(4):
                    ready = ready & (payload[i, j] != cutlass.Uint32(0x80000000))
                if not ready:
                    offset = (((i // 2) * 8 + ft) * 32 + row + 16 * (i % 2)) * 8 + 4 * half
                    a, b, c, d = volatile_vector(sh.iterator + offset)
                    payload[i, 0], payload[i, 1], payload[i, 2], payload[i, 3] = a, b, c, d
    gate = cute.make_rmem_tensor((4,), cutlass.Float32)
    up = cute.make_rmem_tensor((4,), cutlass.Float32)
    for j in cutlass.range_constexpr(4):
        gate[j], up[j] = cutlass.Float32(0), cutlass.Float32(0)
    for quarter in cutlass.range_constexpr(4):
        for j in cutlass.range_constexpr(4):
            gate[j] += bits_f32(payload[2 * quarter, j])
            up[j] += bits_f32(payload[2 * quarter + 1, j])
    activation_bits = cute.make_tensor(cute.recast_ptr(red.iterator, dtype=cutlass.Uint16), cute.make_layout((8192,)))
    for j in cutlass.range_constexpr(4):
        rstd = bits_f32(moe[MOE_RCP + 4 * half + j])
        gb, ub = round_bf16(gate[j] * rstd), round_bf16(up[j] * rstd)
        exponential = cute.math.exp(-gb, fastmath=True)
        sigmoid = cute.math.div(cutlass.Float32(1), cutlass.Float32(1) + exponential,
                                 rounding=cute.math.RoundingMode.NEAREST_EVEN)
        activation_bits[(4 * half + j) * 128 + feature] = (bf16_pair(gb * sigmoid * ub, cutlass.Float32(0)) & cutlass.Uint32(0xFFFF)).to(cutlass.Uint16)
    cute.arch.sync_threads()
    b = cute.make_rmem_tensor((2, 8), cutlass.Uint32)
    for chunk in cutlass.range_constexpr(2):
        for j in cutlass.range_constexpr(8):
            b[chunk, j] = red[(lane >> 2) * 64 + chunk * 32 + (lane & 3) * 8 + j]
    acc = cute.make_rmem_tensor((4,), cutlass.Float32)
    for r in cutlass.range_constexpr(4):
        acc[r] = cutlass.Float32(0)
    for chunk in cutlass.range_constexpr(2):
        for step in cutlass.range_constexpr(4):
            index = (step // 2) * 4 + (step % 2) * 2
            c0, c1, c2, c3 = mma_bf16(acc[0], acc[1], acc[2], acc[3],
                                     wa[chunk, index], wa[chunk, index + 8], wa[chunk, index + 1], wa[chunk, index + 9],
                                     b[chunk, 2 * step], b[chunk, 2 * step + 1])
            acc[0], acc[1], acc[2], acc[3] = c0, c1, c2, c3
    output_bits = cute.make_tensor(cute.recast_ptr(output.iterator, dtype=cutlass.Uint16), cute.make_layout((1024,)))
    for r in cutlass.range_constexpr(4):
        token = 2 * (lane & 3) + (r & 1)
        col = 16 * warp + (lane >> 2) + (r // 2) * 8
        output_bits[token * 128 + col] = (bf16_pair(acc[r], cutlass.Float32(0)) & cutlass.Uint32(0xFFFF)).to(cutlass.Uint16)


SHARED_BYTES = 223232
SHARED_WORDS = SHARED_BYTES // 4


@cute.jit
def shared_views(storage: cute.Tensor):
    region = cute.make_tensor(storage.iterator, cute.make_layout((50176,)))
    red = cute.make_tensor(cute.recast_ptr(storage.iterator + 50176, dtype=cutlass.Float32), cute.make_layout((4096,)))
    residual = cute.make_tensor(storage.iterator + 54272, cute.make_layout((768,)))
    scalars = cute.make_tensor(cute.recast_ptr(storage.iterator + 55168, dtype=cutlass.Float32), cute.make_layout((128,)))
    moe = cute.make_tensor(storage.iterator + 55296, cute.make_layout((240,)))
    bars = cute.make_tensor(cute.recast_ptr(storage.iterator + 55040, dtype=cutlass.Uint64), cute.make_layout((32,)))
    tmem_slot = cute.make_tensor(storage.iterator + 55288, cute.make_layout((1,)))
    return region, red, residual, scalars, moe, bars, tmem_slot


# Imported last: the role modules import each other, and their names are used only when a kernel is traced.
from .indexer import round_bf16, volatile_pair
from .attention import bf16_from_word, volatile_vector, e4m3_pair, published_word, bf16_pair, mma_bf16
from .common import (
    as_ptr,
    f32_bits,
    bulk_copy,
    evict_first,
    bulk_piece,
    load_tmem8,
    store_shared_v4,
    max_num,
    copy_scales,
    mma_nvfp4,
    commit_mma_addr,
    fp4_pair,
    e4m3_float,
    zero_shared_vector,
    tc_before,
    tc_after,
    proxy_fence,
    bits_f32,
    fma_rn,
)
from .plain import (
    ROUTE_E,
    ROUTE_SLOT_EXPERT,
    ROUTE_SLOT_OF_TK,
    ROUTE_WEIGHT,
    call_outlined_typed,
    call_outlined,
    MOE_E,
    MOE_SLOT_EXPERT,
    MOE_SLOT_MASK,
    MOE_SLOT_OF_TK,
    MOE_WEIGHT,
    MOE_RCP,
)
from .gpu_publication import store_pair, acquire_word, store_vector, shared_vector, cached_vector, store_word
