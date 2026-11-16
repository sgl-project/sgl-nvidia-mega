# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Source variants of the JIT CUDA kernels for tests that must be able to fail.

Each entry turns the shipped source into one deliberately broken variant: an ordered list of (old, new) text
replacements on the file, applied to a COPY of the plugin by ``_harness.load_plugin`` (K31_MUTATION=<name>); the
shipped sources carry no test hooks. Variant numbers are the test modules' keys; each test module names the test
that must fail on each variant (and the few variants that must still pass)."""

# Jitter form of the cluster-exchange index-query kernels (soak variants): poison this CTA's exchange slots, a RELEASE arrive
# instead of the relaxed one, and random delays before the DSMEM stores. Shipped sources carry no jitter switch.
_QFW_JITTER = [
    ('  // Every peer has started (its shared memory exists) before anyone stores into it: arrive now, wait before the first\n  // DSMEM store (CUDA PG, distributed shared memory).\n  cluster_arrive_relaxed();\n',
     "  // Test variant: poison this CTA's exchange slots, then a RELEASE arrive orders the poison before any peer's store.\n  for (int i = tid; i < kSplit * kR; i += kThreads) (&s.ex_ss[0][0])[i] = __int_as_float(0x7fc00000);\n  __syncthreads();\n  cluster_arrive_release();\n"),
    ('    const float ssl = s.ssl[b];\n',
     '    const float ssl = s.ssl[b];\n    __nanosleep(static_cast<unsigned>((clock64() ^ (blockIdx.x * 2654435761u) ^ (b * 40503u)) & 4095u));\n'),
]
_QFX_JITTER = [
    ('  // Every peer has started (its shared memory exists) before anyone stores into it: arrive now, wait before the first\n  // DSMEM store (CUDA PG, distributed shared memory).\n  cluster_arrive_relaxed();\n',
     "  // Test variant: poison this CTA's slots, then a RELEASE arrive orders the poison before any peer's store.\n  for (int i = tid; i < (kSplit - 1) * kO * kDim * 8; i += kThreads) (&s.slot[0][0][0][0])[i] = __int_as_float(0x7fc00000);\n  for (int i = tid; i < kSplit * kO * 8; i += kThreads) (&s.ssx[0][0])[i] = __int_as_float(0x7fc00000);\n  __syncthreads();\n  cluster_arrive_release();\n"),
    ('    if (pp == 0) cluster_wait_acquire();  // pairs with the arrive at kernel start: every peer is running\n',
     '    if (pp == 0) cluster_wait_acquire();  // pairs with the arrive at kernel start: every peer is running\n    __nanosleep(static_cast<unsigned>((clock64() ^ (blockIdx.x * 2654435761u) ^ ((warp + 8 * pp) * 40503u)) & 4095u));\n'),
]

VARIANTS = {
    'kimi_k31/dcp/comm.cuh': {
        1: [
            ('  return seen == want;\n',
             '  return seen != 0;\n'),
        ],
        2: [
            ('      const int block = s * p.heads;\n',
             '      const int block = s * p.heads + 1;\n'),
        ],
        3: [
            ('        const float wi = exp2f(sanitize_lse(s.lse[i]) - lse_max);\n',
             '        const float wi = expf(sanitize_lse(s.lse[i]) - lse_max);\n'),
        ],
        4: [
            ('    const float y0 = bf16_round(d[0]), y1 = bf16_round(d[1]), y2 = bf16_round(d[2]), y3 = bf16_round(d[3]);\n',
             '    const float y0 = d[0], y1 = d[1], y2 = d[2], y3 = d[3];\n'),
        ],
        5: [
            ('          }\n          if (pos >= 0) row[pos] = __int_as_float(score);\n',
             '          }\n          pos += 1;\n          if (pos >= 0) row[pos] = __int_as_float(score);\n'),
        ],
        6: [
            ('    const bool own = w >= 0 && w % p.dcp == p.rank;\n',
             '    const bool own = w >= 0 && (w / 64) % p.dcp == p.rank;\n'),
        ],
        7: [
            ('      const int h = d * p.heads + j;\n',
             '      const int h = j * c.dcp + d;\n'),
        ],
    },
    'kimi_k31/dsa/index_q.cuh': {
        1: [
            ('  const float x = fmaxf(amax, 1e-4f) * (1.0f / 448.0f);\n',
             '  const float x = fmaxf(amax, 1e-4f) * (1.0f / 440.0f);  // mutation: wrong FP8 range\n'),
        ],
        2: [
            ('      p.w[b * kHeads + head] = g2 * s.scale[b] * 0.08838834764831845f;  // 128^-0.5\n',
             '      p.w[b * kHeads + head] = g2 * s.scale[b];  // mutation: softmax scale dropped\n'),
        ],
        3: [
            ('  const float e = ceilf(log2f(x));\n',
             '  const float e = rintf(log2f(x));  // mutation: log2 rounded to nearest instead of up\n'),
        ],
        4: [
            ('  const float x = fmaxf(amax, 1e-4f) * (1.0f / 448.0f);\n',
             '  const float x = amax * (1.0f / 448.0f);  // mutation: 1e-4 amax floor dropped\n'),
        ],
        5: [
            ('  const float x = fmaxf(amax, 1e-4f) * (1.0f / 448.0f);\n  const float e = ceilf(log2f(x));\n',
             '  const float x = fmaxf(amax, 1e-4f) * (1.0f / 448.0f);\n  return x;  // mutation: the exact amax/448 scale of an earlier release\n  const float e = ceilf(log2f(x));\n'),
        ],
        6: [
            ('      const int gcol = p.g_col + head;\n',
             "      const int gcol = p.g_col + (head + 1) % kHeads;  // mutation: the next head's gate column\n"),
        ],
        7: [
            ('  reinterpret_cast<uint32_t*>(page + off * 128)[lane] = packed;\n',
             '  reinterpret_cast<uint32_t*>(page + off * 132)[lane] = packed;  // mutation: row pitch 132 instead of 128\n'),
        ],
        8: [
            ('    v[j] = bf16_round(v[j] * rstd * p.kgamma[col] + p.kbeta[col]);\n',
             '    v[j] = bf16_round(v[j] * rstd * p.kgamma[col]);  // mutation: LayerNorm bias dropped\n'),
        ],
        9: [
            ('  const float mean = sum * (1.0f / 128.0f);\n',
             '  const float mean = 0.f;  // mutation: mean not subtracted\n'),
        ],
        10: [
            ('      mma_bf16(d1, a0, a1, a2, a3, xs(g + 8, k0 + 2 * qd), xs(g + 8, k0 + 8 + 2 * qd));\n',
             '      mma_bf16(d1, a0, a1, a2, a3, xs(g, k0 + 2 * qd), xs(g, k0 + 8 + 2 * qd));  // mutation: tile 2 reads rows 0-7\n'),
        ],
        11: [
            ('    const float rr = rsqrtf(ss / static_cast<float>(kLora) + p.eps);\n',
             "    const float rr = jr == 0 ? rsqrtf(ss / static_cast<float>(kLora) + p.eps) : rr0;  // mutation: row w + 8 uses row w's\n"),
        ],
        12: [
            ('  const bool k_owner = kQF && rank == kSplit - 1 && warp == 7 && head < p.rows;\n',
             '  const bool k_owner = kQF && rank == kSplit - 1 && warp == 7 && head < p.rows && head < 8;  // mutation: rows >= 8 not stored\n'),
        ],
    },
    'kimi_k31/dsa/index_qfw.cuh': {
        1: [
            ('    for (int c = 0; c < kSplit; ++c) {\n      ss += s.ex_ss[c][b];\n',
             "    for (int c = 0; c < kSplit; ++c) {\n      if (c == kSplit - 1) continue;  // mutation: rank 7's partial dropped\n      ss += s.ex_ss[c][b];\n"),
        ],
        2: [
            ('    const float rr = rsqrtf(ss / static_cast<float>(kLora) + p.eps);\n',
             '    const float rr = 1.f;  // mutation: rstd := 1\n'),
        ],
        3: [
            ('  const bool k_owner = rank == kSplit - 1 && warp == 7 && head < p.rows;\n',
             '  const bool k_owner = rank == kSplit - 1 && warp == 7 && head < p.rows && head < 16;  // mutation: rows >= 16 not stored\n'),
        ],
        4: [
            ('  const int gcol = p.g_col + head;\n',
             "  const int gcol = p.g_col + (head + 1) % kHeads;  // mutation: the next head's gate column\n"),
        ],
        5: [
            ('      const int xr = 8 * j + g;\n',
             "      const int xr = 8 * (j == 2 ? 0 : j) + g;  // mutation: tile 2 reads tile 0's rows\n"),
        ],
        6: _QFW_JITTER + [
            ('  if (k_owner) store_index_k(p, kraw, kslot, lane);\n  cluster_wait_acquire();\n',
             "  if (k_owner) store_index_k(p, kraw, kslot, lane);\n  float ss_early = 0.f;  // mutation: the peers' partials read before the wait\n  if (tid < kR) {\n#pragma unroll\n    for (int c = 0; c < kSplit; ++c) ss_early += s.ex_ss[c][tid];\n  }\n  cluster_wait_acquire();\n"),
            ('#pragma unroll\n    for (int c = 0; c < kSplit; ++c) {\n      ss += s.ex_ss[c][b];\n    }\n',
             '    ss = ss_early;\n'),
        ],
        7: _QFW_JITTER,
    },
    'kimi_k31/dsa/index_qfx.cuh': {
        1: [
            ('      if (static_cast<int>(rank) + 8 * m < kT) {\n#pragma unroll\n',
             '      if (static_cast<int>(rank) + 8 * m < kT) {\n        acc[m][0] = own[m][0];  // mutation: own partial first\n        acc[m][1] = own[m][1];\n        acc[m][2] = own[m][2];\n        acc[m][3] = own[m][3];\n#pragma unroll\n'),
            ('            v0 = own[m][0];\n            v1 = own[m][1];\n            v2 = own[m][2];\n            v3 = own[m][3];\n',
             '            continue;\n'),
        ],
        2: [
            ('      for (int c = 0; c < kSplit; ++c) {\n        ss += s.ssx[c][tid];\n',
             "      for (int c = 0; c < kSplit; ++c) {\n        if (c == kSplit - 1) continue;  // mutation: rank 7's partial dropped\n        ss += s.ssx[c][tid];\n"),
        ],
        3: [
            ('  const bool k_owner = warp == 7 && krow < p.rows;\n',
             '  const bool k_owner = warp == 7 && krow < p.rows && krow < 64;  // mutation: rows >= 64 not stored\n'),
        ],
        4: [
            ('          const uint32_t dst = static_cast<uint32_t>(j);\n',
             '          const uint32_t dst = pp == 1 ? static_cast<uint32_t>((j + 1) % kSplit) : static_cast<uint32_t>(j);  // mutation\n'),
        ],
        5: _QFX_JITTER + [
            ('  cluster_arrive_release();\n  // The index-K row of this CTA (rows < p.rows), off the exchange: between arrive and wait.\n',
             "  cluster_arrive_release();\n  reduce();  // mutation: the slots read before the exchange's wait\n  // The index-K row of this CTA (rows < p.rows), off the exchange: between arrive and wait.\n"),
            ('  reduce();\n',
             ''),
        ],
        6: _QFX_JITTER,
        7: [
            ('  const int gcol = p.g_col + head;\n',
             "  const int gcol = p.g_col + (head + 1) % kHeads;  // mutation: the next head's gate column\n"),
        ],
        8: [
            ('            v0 = own[m][0];\n            v1 = own[m][1];\n            v2 = own[m][2];\n            v3 = own[m][3];\n',
             '            v0 = v1 = v2 = v3 = 0.f;  // mutation: own partial dropped\n'),
        ],
    },
    'kimi_k31/dsa/index_qp.cuh': {
        1: [
            ('      *reinterpret_cast<uint4*>(&s.x[warp][k0 * 2]) = o;\n',
             "      *reinterpret_cast<uint4*>(&s.x[warp ^ 1][k0 * 2]) = o;  // mutation: row w's values into SMEM row w ^ 1\n"),
        ],
        2: [
            ('  const float w0 = __uint_as_float(w << 16), w1 = __uint_as_float(w & 0xffff0000u);\n',
             '  const float w1 = __uint_as_float(w << 16), w0 = __uint_as_float(w & 0xffff0000u);\n'),
        ],
        3: [
            ('  const uint4* xsrc = reinterpret_cast<const uint4*>(p.projected + static_cast<int64_t>(warp) * p.pitch);\n  pdl_wait();\n',
             '  const uint4* xsrc = reinterpret_cast<const uint4*>(p.projected + static_cast<int64_t>(warp) * p.pitch);\n#pragma unroll\n  for (int i = 0; i < kQaChunks; ++i) xa[i] = warp < p.rows ? xsrc[i * 32 + lane] : make_uint4(0, 0, 0, 0);  // mutation: before the wait\n  pdl_wait();\n'),
            ("#pragma unroll\n  for (int i = 0; i < kQaChunks; ++i) xa[i] = warp < p.rows ? xsrc[i * 32 + lane] : make_uint4(0, 0, 0, 0);  // warp w's q_a row, index_q.cuh's lane order\n",
             ''),
        ],
        4: [
            ('    af[ks][0] = wq(g, k0 + 2 * qd);\n',
             '    af[ks][0] = ks == 0 ? wq(g + 8, k0 + 2 * qd) : wq(g, k0 + 2 * qd);\n'),
        ],
    },
    # tsi.cuh variants (each must make test_k31_tsi.py fail):
    #   1 a dropped source: warp (rank + 1) % W skips its poll and treats the source as empty
    #   2 position remap off by one page                 3 any nonzero tag accepted (stale records)
    #   4 certificate never fails                         5 output not sorted (compaction order)
    #   6 local_len formula off by one (full local pages rounded up one early)
    #   7 (EMIT 3) the sender's position ranking skipped (candidates pushed in compaction order)
    #   8 (T1L) the published FULL list holds one key too few     9 (T1L) the FULL list is never published (peers trap)
    #   10 (T1L) the sender reads every candidate one local token off
    'kimi_k31/dsa/tsi.cuh': {
        1: [
            ('    for (int i = 0; i < kPerLane; ++i) cand[i] = s.ckey[TSI_ORD(lane + 32 * i)];\n  } else {\n',
             '    for (int i = 0; i < kPerLane; ++i) cand[i] = s.ckey[TSI_ORD(lane + 32 * i)];\n  } else if (warp == (p.rank + 1) % kWorld) {\n#pragma unroll\n    for (int i = 0; i < kPerLane; ++i) cand[i] = 0;\n  } else {\n'),
        ],
        2: [
            ('  return (world * (l / kPage) + r) * kPage + l % kPage;\n',
             '  return (world * (l / kPage) + r + 1) * kPage + l % kPage;\n'),
        ],
        3: [
            ('  return seen == want;\n',
             '  return seen != 0;\n'),
        ],
        4: [
            ('#pragma unroll\n  for (int q = 0; q < kWorld; ++q)\n    if (want > 0 && s.kept[q] == s.sent[q] && local_len(n, kWorld, q) > s.sent[q]) fail |= 1u << q;\n',
             ''),
        ],
        5: [
            ("#if !K31_TSI_EMIT_BITMAP\n  bool done = false;\n#if K31_TSI_EMIT == 2\n  // ---- bucket counting sort by position: bucket = pos >> shift (<= 2048 buckets over the row's prefix); a key's slot\n  // is its bucket's start plus the number of keys in its bucket with a smaller position (positions are unique), so the\n  // order is the sorted one and independent of atomic order. Rows with a bucket of > 32 keys take the radix sort. ----\n  {\n    const int shift = max(0, 32 - __clz(max(n - 1, 1)) - 11);\n    const uint64_t* keys = p.scratch + static_cast<int64_t>(row) * kWorld * kTopK;\n    auto each = [&](auto f) {  // every selected position held by this thread\n      if (t == 0) return;\n      if (!fail) {\n#pragma unroll\n        for (int i = 0; i < kPerLane; ++i)\n          if (cand[i] >= t) f(key_pos(cand[i]));\n      } else {\n        for (int e = tid; e < kWorld * kTopK; e += kThreads)\n          if (keys[e] >= t) f(key_pos(keys[e]));\n      }\n    };\n    for (int b = tid; b < S::kBuckets; b += kThreads) s.bk.count[b] = 0;\n    if (tid == 0) s.max_count = 0;\n    __syncthreads();\n    each([&](uint32_t pos) { atomicAdd(&s.bk.count[min(static_cast<int>(pos >> shift), S::kBuckets - 1)], 1); });\n    __syncthreads();\n    constexpr int kPerT = S::kBuckets / kThreads;\n    int c[kPerT], sum = 0, mx = 0;\n#pragma unroll\n    for (int j = 0; j < kPerT; ++j) {\n      c[j] = s.bk.count[tid * kPerT + j];\n      sum += c[j];\n      mx = max(mx, c[j]);\n    }\n    int total;\n    int run = block_exclusive_sum<kThreads>(sum, total, s);\n    if (mx > 0) atomicMax(&s.max_count, mx);\n#pragma unroll\n    for (int j = 0; j < kPerT; ++j) {\n      s.bk.end[tid * kPerT + j] = run;\n      run += c[j];\n    }\n    __syncthreads();\n    if (s.max_count <= 32) {\n      each([&](uint32_t pos) {\n        const int b = min(static_cast<int>(pos >> shift), S::kBuckets - 1);\n        s.bk.arr[atomicAdd(&s.bk.end[b], 1)] = pos;\n      });\n      __syncthreads();\n      each([&](uint32_t pos) {\n        const int b = min(static_cast<int>(pos >> shift), S::kBuckets - 1);\n        const int cnt = s.bk.count[b], start = s.bk.end[b] - cnt;\n        int r = 0;\n        for (int k = 0; k < cnt; ++k) r += s.bk.arr[start + k] < pos;\n        if (start + r < kTopK) out[start + r] = table[pos / kPage] * kPage + static_cast<int32_t>(pos % kPage);\n      });\n      for (int e = min(total, kTopK) + tid; e < kTopK; e += kThreads) out[e] = -1;\n      done = true;\n    }\n    __syncthreads();  // the union is reused by the radix-sort path\n  }\n#endif\n#if K31_TSI_EMIT == 3\n  // ---- s1r, non-fallback rows: every source's candidates arrive position-ascending (sender ranking); warp q compacts\n  // its kept keys in element order after the kept keys of sources < q (s.kept, complete since the barrier above) ----\n  if (!fail) {\n    int bnd[kWorld + 1];\n    bnd[0] = 0;\n#pragma unroll\n    for (int q = 0; q < kWorld; ++q) bnd[q + 1] = bnd[q] + s.kept[q];\n    int base = 0;\n#pragma unroll\n    for (int q = 0; q < kWorld; ++q) base = q == warp ? bnd[q] : base;\n    const unsigned lt = (1u << lane) - 1u;\n#pragma unroll\n    for (int i = 0; i < kPerLane; ++i) {\n      const bool keep = t != 0 && cand[i] >= t;\n      const unsigned m = __ballot_sync(0xffffffffu, keep);\n      const int slot = base + __popc(m & lt);\n      if (keep && slot < kTopK) {\n        const uint32_t pos = key_pos(cand[i]);\n        out[slot] = table[pos / kPage] * kPage + static_cast<int32_t>(pos % kPage);\n      }\n      base += __popc(m);\n    }\n    for (int e = min(bnd[kWorld], kTopK) + tid; e < kTopK; e += kThreads) out[e] = -1;\n    done = true;\n  }\n#endif\n  if (!done) {\n  // ---- every key >= t*, positions compacted into SMEM (kEmptyPos padded), block radix sort, KV slots ----\n",
             '  // (mutation) slots in compaction order, unsorted\n'),
            ("  uint32_t items[kPer];\n#pragma unroll\n  for (int j = 0; j < kPer; ++j) items[j] = s.pos[tid * kPer + j];\n  __syncthreads();  // the SMEM union is reused as the sort's temporary storage\n  typename S::Sort sorter(s.sort);\n  sorter.SortBlockedToStriped(items, 0, p.end_bit);\n#pragma unroll\n  for (int j = 0; j < kPer; ++j) {\n    const uint32_t pos = items[j];\n    out[j * kThreads + tid] = pos == kEmptyPos ? -1 : table[pos / kPage] * kPage + static_cast<int32_t>(pos % kPage);\n",
             '  for (int e = tid; e < kTopK; e += kThreads) {\n    const uint32_t pos = s.pos[e];\n    out[e] = pos == kEmptyPos ? -1 : table[pos / kPage] * kPage + static_cast<int32_t>(pos % kPage);\n'),
            ("  }\n#else\n  // ---- every key >= t* -> one bit per position in a SMEM bitmap (one u64 word per page); thread q owns the\n  // contiguous pages [q * chunk, (q + 1) * chunk) at SMEM words q * stride + j (stride odd: conflict-free walks);\n  // a block scan of the per-thread popcounts gives each thread its output offset, and every thread emits its pages'\n  // slots in position order: the sorted, deterministic order of the sparse core's sum with no comparison sort ----\n  extern __shared__ __align__(16) unsigned long long bm[];\n  const int words = (n + kPage - 1) / kPage;  // pages of this row's causal prefix (every selected position is < n)\n  const int chunk = p.bm_chunk, stride = chunk | 1;\n  for (int j = 0; j < chunk; ++j)\n    if (tid * chunk + j < words) bm[tid * stride + j] = 0ull;\n  __syncthreads();\n  auto mark = [&](uint32_t pos) {\n    const int w = static_cast<int>(pos / kPage);\n    if (w < words) {\n      const int q = w / chunk;\n      atomicOr(&bm[q * stride + (w - q * chunk)], 1ull << (pos % kPage));\n    }\n  };\n  if (t != 0) {\n    if (!fail) {\n#pragma unroll\n      for (int i = 0; i < kPerLane; ++i)\n        if (cand[i] >= t) mark(key_pos(cand[i]));\n    } else {\n      const uint64_t* keys = p.scratch + static_cast<int64_t>(row) * kWorld * kTopK;\n      for (int e = tid; e < kWorld * kTopK; e += kThreads)\n        if (keys[e] >= t) mark(key_pos(keys[e]));\n    }\n  }\n  __syncthreads();\n  int mine = 0, total;\n  for (int j = 0; j < chunk; ++j)\n    if (tid * chunk + j < words) mine += __popcll(bm[tid * stride + j]);\n  int at = block_exclusive_sum<kThreads>(mine, total, s);\n  for (int j = 0; j < chunk; ++j) {\n    const int w = tid * chunk + j;\n    if (w >= words) break;\n    unsigned long long m = bm[tid * stride + j];\n    if (m == 0ull) continue;\n    const int base = table[w] * kPage;\n    do {\n      const int b = __ffsll(static_cast<long long>(m)) - 1;\n      m &= m - 1;\n      if (at < kTopK) out[at] = base + b;\n      ++at;\n    } while (m);\n  }\n  for (int e = min(total, kTopK) + tid; e < kTopK; e += kThreads) out[e] = -1;\n#endif\n",
             ''),
        ],
        6: [
            ('  const int full = (pages + world - 1 - r) / world;\n',
             '  const int full = (pages + world - r) / world;\n'),
        ],
        7: [
            ('  r += __shfl_xor_sync(0xffffffffu, r, 1);\n  if (h == 0) s.order[r] = static_cast<uint16_t>(e);\n',
             '  r += __shfl_xor_sync(0xffffffffu, r, 1);\n  r = e;  // (mutation) sort skipped\n  if (h == 0) s.order[r] = static_cast<uint16_t>(e);\n'),
        ],
        8: [
            ('  const int want = min(nl, kTopK);\n',
             '  const int want = min(nl, kTopK) - 1;\n'),
        ],
        9: [
            ('        st_rec2(reinterpret_cast<uint64_t>(own_region + full_index<kWorld, kCand>(parity, row, at, p.cap)), rec(sb, ftag),\n                rec(static_cast<uint32_t>(ps), ftag));\n',
             ''),
            ('    st_rec2(reinterpret_cast<uint64_t>(own_region + full_index<kWorld, kCand>(parity, row, e, p.cap)), rec(0u, ftag),\n            rec(kEmptyPos, ftag));\n',
             ''),
        ],
        10: [
            ('      int l = p.raw[static_cast<int64_t>(row) * p.raw_stride + tid];\n      if (l >= 0) {\n',
             '      int l = p.raw[static_cast<int64_t>(row) * p.raw_stride + tid];\n      l = l >= 0 ? l + 1 : l;\n      if (l >= 0) {\n'),
        ],
    },
    # tsim.cuh variants (each must make test_k31_tsim.py fail):
    #   1 drop_source   the receiver treats source 7 as empty       2 tie_rank     boundary rank one too deep
    #   3 emit_reversed emission reversed inside each ballot round  4 no_fallback  flagged rows emit the fast path
    #   5 full_trunc    FULL lists keep only their first 1536 keys  6 no_canon     the fallback keys keep -0.0 below +0.0
    #   7 pos_page      global position one page off                8 stash_group  cp.async stages the wrong group (V > 8)
    #   9 skip_last     the uniform group skip drops a partial last group (ve = floor(n4 / 1024))
    #  12 long_drop     a local row longer than 16 groups takes the fast path on its first 16 groups (no forced overflow)
    #  13 push_dest15   the unicast push skips its 16th destination ((rank + 15) % 16 never receives this rank's records)
    #  14 push_src_shift the unicast push writes into the NEXT source's slots of every destination (cand_index source rank + 1)
    #  15 disp_plus1    rows of 9 groups run the 8-group body (DISP boundary one group too high)
    #  18 any_tag       a poll accepts an element carrying ANY non-zero tag (stale records of an earlier layer pass; the multi-GPU
    #                   multicast soak must fail under it)
    # Sanitizer POSITIVE controls (run ONLY under compute-sanitizer; never in a plain test run):
    #  10 san_race      the sender's S2a barrier dropped (racecheck must report)  11 san_heap_oob  a device-heap read 24 B past a 16-B block (memcheck)
    #  16 san_sync      warp 0 of block (0, 0) skips one CTA barrier the other 31 warps execute (synccheck must report)
    #  17 san_uninit    thread 0 of block (0, 0) reads a fresh device-heap block and scratch[0] before any write (initcheck must report;
    #                   the test passes an uninitialised scratch under this mutation)
    'kimi_k31/dsa/tsim.cuh': {
        1: [
            ('  int nsrc = lane < kW && !src_ovf ? static_cast<int>(hd0 & 0x3FFFFFFFu) : 0;\n  const float tau_s = lane < kW ? __uint_as_float(static_cast<uint32_t>(hb)) : -INFINITY;\n',
             '  int nsrc = lane < kW && !src_ovf ? static_cast<int>(hd0 & 0x3FFFFFFFu) : 0;\n  if (lane == 7) nsrc = 0;\n  const float tau_s = lane < kW ? __uint_as_float(static_cast<uint32_t>(hb)) : -INFINITY;\n'),
        ],
        2: [
            ('  const int rsel = rstar;\n',
             '  const int rsel = rstar + 1;\n'),
        ],
        3: [
            ('        const int at = off + k;\n',
             '        const int at = off + nr - 1 - k;\n'),
        ],
        4: [
            ('  const bool fast = fl == 0;\n',
             '  const bool fast = true;\n'),
        ],
        5: [
            ('      const bool keep = true;\n',
             '      const bool keep = at < 1536;\n'),
        ],
        6: [
            ('  return __fadd_rn(x, 0.0f);  // -0 + +0 = +0 (round to nearest): equal scores get equal keys\n',
             '  return x;\n'),
        ],
        7: [
            ('  return (kW * (l / kPage) + r) * kPage + l % kPage;\n',
             '  return (kW * (l / kPage) + r + 1) * kPage + l % kPage;\n'),
        ],
        8: [
            ('        if (i < n4) cp_async16(stash + k * kT + tid, src + i);\n',
             '        if (i + kT < n4) cp_async16(stash + k * kT + tid, src + i + kT);\n'),
        ],
        9: [
            ('  const int vall = (n4 + kT - 1) / kT;\n',
             '  const int vall = n4 / kT;\n'),
        ],
        10: [
            ('  }\n  __syncthreads();\n  const int nv = static_cast<int>(__reduce_add_sync(kFull, static_cast<unsigned>(sm.nv[lane])));\n',
             '  }\n  const int nv = static_cast<int>(__reduce_add_sync(kFull, static_cast<unsigned>(sm.nv[lane])));\n'),
        ],
        11: [
            ('  const uint32_t epoch = ld_gpu_u32(p.epoch);\n  const int n = min(ld_gpu_i32(p.lcausal + rank * p.lc_rank_stride + row), p.ncols);\n',
             '  const uint32_t epoch = ld_gpu_u32(p.epoch);\n  if (tid == 0 && blockIdx.x == 0 && blockIdx.y == 0) {\n    unsigned char* q = static_cast<unsigned char*>(malloc(16));\n    if (q != nullptr) {\n      s.fb.min_key = *reinterpret_cast<volatile unsigned char*>(q + 40);\n      free(q);\n    }\n  }\n  const int n = min(ld_gpu_i32(p.lcausal + rank * p.lc_rank_stride + row), p.ncols);\n'),
        ],
        12: [
            ('  const bool too_long = vall > kVMax;  // uniform; only when the table is wider than kVMax groups (ncols > 65,536 for kernel<16>)\n',
             '  const bool too_long = false;\n'),
        ],
        13: [
            ('  for (int idx = threadIdx.x; idx < ns * kW; idx += kT) {\n',
             '  for (int idx = threadIdx.x; idx < ns * (kW - 1); idx += kT) {\n'),
        ],
        14: [
            ('    unicast_push(p, rank, cand_index(parity, row, rank, 0, p.cap), N, ovf, sm);\n',
             '    unicast_push(p, rank, cand_index(parity, row, (rank + 1) & (kW - 1), 0, p.cap), N, ovf, sm);\n'),
        ],
        15: [
            ('    if (veff <= kVReg) {\n',
             '    if (veff <= kVReg + 1) {\n'),
        ],
        16: [
            ('  const uint32_t epoch = ld_gpu_u32(p.epoch);\n  const int n = min(ld_gpu_i32(p.lcausal + rank * p.lc_rank_stride + row), p.ncols);\n',
             '  const uint32_t epoch = ld_gpu_u32(p.epoch);\n  if (blockIdx.x == 0 && blockIdx.y == 0 && tid >= 32) __syncthreads();\n  const int n = min(ld_gpu_i32(p.lcausal + rank * p.lc_rank_stride + row), p.ncols);\n'),
        ],
        17: [
            ('  const uint32_t epoch = ld_gpu_u32(p.epoch);\n  const int n = min(ld_gpu_i32(p.lcausal + rank * p.lc_rank_stride + row), p.ncols);\n',
             '  const uint32_t epoch = ld_gpu_u32(p.epoch);\n  if (tid == 0 && blockIdx.x == 0 && blockIdx.y == 0) {\n    unsigned char* q = static_cast<unsigned char*>(malloc(16));\n    if (q != nullptr) {\n      s.fb.min_key = *reinterpret_cast<volatile unsigned char*>(q + 4);\n      free(q);\n    }\n    s.fb.min_key += *reinterpret_cast<volatile const unsigned long long*>(p.scratch);\n  }\n  const int n = min(ld_gpu_i32(p.lcausal + rank * p.lc_rank_stride + row), p.ncols);\n'),
        ],
        18: [
            ('__device__ __forceinline__ bool tagged(uint64_t a, uint64_t b, uint32_t tag) { return hi32(a) == tag && hi32(b) == tag; }\n',
             '__device__ __forceinline__ bool tagged(uint64_t a, uint64_t b, uint32_t tag) { return hi32(a) != 0u && hi32(b) != 0u; }\n'),
        ],
    },
}


def edits(rel: str, n: int) -> list:
    """The (rel, old, new) replacements of variant ``n`` of ``rel`` (a path under sgl_nvidia_mega/)."""
    return [(rel, old, new) for old, new in VARIANTS[rel][n]]
