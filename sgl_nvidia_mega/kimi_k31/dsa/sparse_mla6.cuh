// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// K3.1 sparse MLA core at the TP16 per-rank head count (6 heads, no padding to 16): split-KV partials for the fold
// in MlaValueFold (kimi_k31/k3/cutedsl/mla/value.py). Replaces trtllm-gen's sparse MLA decode (heads padded 6 -> 16, its
// own split reduction) behind K31MK_SPARSE_MLA6.
//
// Per query row r (one verify token) and split c (tokens [c*T, c*T + T) of the row's 2048 selected slots):
//   s[h, j] = (q[r, h, :576] . kv[slot_j, :576]) * scale_log2       FP8 x FP8 -> FP32 (mma.sync e4m3, FP32 acc)
//   valid j: j < min(len[r], 2048) and slot_j >= 0                   (else s = -inf, the KV row is zeroed)
//   m[h]    = max_j s[h, j]                                           (-inf when the split has no valid token)
//   p[h, j] = ex2.approx(s[h, j] - m[h])                              (m[h] = -inf -> 0 is used)
//   l[h]    = sum_j p[h, j]                                           (FP32)
//   U[h, :] = sum_j f16(p[h, j]) * kv[slot_j, :512]                   (FP8 -> FP16 exact, mma.sync f16, FP32 acc)
// Written: partial[r, c, h, :512] = U (unnormalized FP32), stats[r, c, h] = (m, l). The fold computes
//   M = max_c m_c, w_c = ex2(m_c - M), latent = (sum_c w_c U_c) / (sum_c w_c l_c) * bmm2_scale  -> BF16.
// Differences from the stock core (not bitwise): P stays FP16 (stock: E4M3), partials stay FP32 (stock: BF16),
// summation orders differ. scale_log2 = bmm1_scale * log2(e) as stock's mScaleSoftmaxLog2.
//
// Grid (2048 / T splits, rows), 128 threads. Warp w gathers and scores tokens [w*T/4, (w+1)*T/4) of the split; for
// P.V warp w owns latent dims [128w, 128w + 128) over all T tokens. Synchronization: PDL only (griddepcontrol.wait
// before the first read of any predecessor output; launch_dependents at entry, no memory effect needed because the
// dependent waits for this grid's completion); KV rows arrive by cp.async (generic proxy) + wait_group + bar.sync.
#include <cuda_fp16.h>
#include <cstdint>

namespace k31_sparse_mla6 {

constexpr int kTopK = 2048, kDim = 576, kLatent = 512;
constexpr int kPitch = kDim + 16;  // SMEM row bytes: 8 consecutive rows hit 8 distinct 16-byte bank groups

struct Params {
  const uint8_t* q;         // [rows, H, 576] FP8 E4M3 (row stride q_stride bytes)
  const uint8_t* kv;        // [slots, 576] FP8 E4M3 (the page-64 pool viewed flat)
  const int32_t* selected;  // [rows, 2048] physical slots, -1 = none (row stride sel_stride)
  const int32_t* lengths;   // [rows] causal lengths
  float* partial;           // [rows, C, H, 512]
  float2* stats;            // [rows, C, H] (m, l)
  int64_t q_stride, sel_stride;
  int64_t num_slots;  // slots in the pool: a slot outside [0, num_slots) is treated as no slot
  float scale_log2;
  // Tagged mode only (K31MK_DSA_REQ_SHARD_AG=fused; zero otherwise): the slots arrive as epoch-tagged words in this rank's
  // copy of the request-sharded selection's mailbox (dsa/tag_gather.py encoding: (tag << 25) | (slot + 1)), not in
  // ``selected``. Each CTA polls its own words; CTA (0, 0) also polls word 0 of every rank's block, so every rank observes
  // every peer's push of this use before its own next push (buffer-reuse order; see the kernel).
  const uint32_t* tagged;  // [rank blocks x heart_rows, 2048]; row r of the batch = mailbox row r
  int32_t* epochs;         // [rows, C] uses of this copy so far, per (row, split); advanced by the CTA after its polls
  int64_t deadline_ns;     // spin limit per CTA; then trap (fail closed)
  int32_t heart_rows;      // rows per rank block (requests per rank x query width)
  int32_t heart_blocks;    // rank blocks (TP world)
};

template <int T, int W>
struct Smem {
  static constexpr int kPPitch = T * 2 + 16;  // FP16 P row bytes (conflict-free 32-bit fragment loads)
  uint8_t kv[T][kPitch];
  uint8_t q[8][kPitch];  // heads H..7 zero
  uint8_t p[8][kPPitch];
  int32_t slot[T];
  float max[W][8];
  float sum[W][8];
};

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}
// Wait until at most n of this thread's commit groups are pending (n is a constant after unrolling).
__device__ __forceinline__ void cp_async_wait_pending(int n) {
  if (n <= 0) cp_async_wait<0>();
  else if (n == 1) cp_async_wait<1>();
  else if (n == 2) cp_async_wait<2>();
  else cp_async_wait<3>();
}
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ int ld_coherent(const int32_t* p) {
  int v;
  asm volatile("ld.global.cg.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint64_t globaltimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
// Tagged mode: strong relaxed sys-scope load of a word peers store with st.relaxed.sys (tag_gather.py's push).
__device__ __forceinline__ uint32_t ld_relaxed_sys(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.relaxed.sys.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void fence_acq_rel_sys() { asm volatile("fence.acq_rel.sys;" ::: "memory"); }
// Spin until the word carries tag ``tg`` (bits 25-31); trap once ``deadline_ns`` has passed since ``t0``.
__device__ __forceinline__ uint32_t poll_tag(const uint32_t* p, uint32_t tg, uint64_t t0, int64_t deadline_ns) {
  uint32_t v = ld_relaxed_sys(p);
  while ((v >> 25) != tg) {
    if (globaltimer() - t0 > static_cast<uint64_t>(deadline_ns)) __trap();
    v = ld_relaxed_sys(p);
  }
  return v;
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_addr(p))
               : "memory");
}
__device__ __forceinline__ void ldsm_x4_trans(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_addr(p))
               : "memory");
}
__device__ __forceinline__ void mma_e4m3(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_f16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// One ldmatrix.trans b16 word of FP8 V = bytes (V[k][d], V[k][d+1], V[k+1][d], V[k+1][d+1]) -> the FP16 A words of
// latent dims d and d+1 at tokens (k, k+1): lo = (V[k][d], V[k+1][d]), hi = (V[k][d+1], V[k+1][d+1]). Exact.
__device__ __forceinline__ void fp8_pairs_to_f16(uint32_t word, uint32_t& lo, uint32_t& hi) {
  asm("{\n.reg .b32 t; .reg .b16 a, b;\n"
      "prmt.b32 t, %2, 0, 0x3120;\n"
      "mov.b32 {a, b}, t;\n"
      "cvt.rn.f16x2.e4m3x2 %0, a;\n"
      "cvt.rn.f16x2.e4m3x2 %1, b;\n}"
      : "=r"(lo), "=r"(hi)
      : "r"(word));
}
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ uint32_t pack_f16(float lo, float hi) {
  const __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&h);
}

// W warps gather and score T / W tokens each (a multiple of 16); warps 0-3 run P.V (128 latent dims each).
// kTag (K31MK_DSA_REQ_SHARD_AG=fused): the slots come from the tagged mailbox (Params, tagged mode) instead of ``selected``.
// Protocol = tag_gather.py's: each polling thread spins with ld.relaxed.sys until its word carries the epoch
// tag of this use, then executes fence.acq_rel.sys (acquire pattern: the observed words precede this rank's next push of
// the request-sharded selection, which is stream-ordered after this kernel); a missing word traps after deadline_ns
// (nothing used, no counter advanced). CTA (0, 0) also polls word 0 of every rank's block: a rank whose block holds only
// repeat rows (Breq < world) is never read by a core otherwise, and its peers must still observe its push of this use
// before they overwrite their own block in its copy of this mailbox two uses later.
// kFence = false (tagged mode only; selected by the host's tagged mode (TAGNF); not selectable in this release; fails
// closed): no fence.acq_rel.sys after the polls. The fence orders only the copy reuse (WAR: this rank's reads of copy c
// before its next push); each word is its own flag and carries its own data, so the slots need no acquire. Without it the
// reuse order rests on the served layer chain between uses of a copy; a lost order shows the next use's tag, which no CTA
// accepts (exact tag), so the CTA traps at its deadline (armed: run_tagged refuses deadline_ns <= 0) - never a wrong row.
// kFeed (K31MK_SPARSE_MLA6_FEED, T 128 only; TMA-FEED): the same SMEM bytes arrive by cp.async.bulk (async proxy, one 576 B
// copy per KV row issued by the thread that owns the row, the query by threads 0..H-1) with completion on mbarriers (one per (warp, m-tile),
// count 16 = one arrival per row thread: arrive.expect_tx(576) before its copy, or a plain arrive after zero-filling an invalid row; one for
// the query, count H) instead of cp.async groups + wait_group + bar.sync. Readers wait on the barrier of the rows they read (try_wait.parity
// acquire: the copies' bytes and the zero-fill stores are visible; SGLang hicache_tma.cuh form); P.V, which reads every warp's rows, first
// waits on every barrier. Math, addressing and summation order are unchanged (bitwise to kFeed = false).
// W = 8 at T 128 (K31MK_SPARSE_MLA6_FEED_W8, only with kFeed): each warp scores ONE 16-token m-tile and all 8 warps run P.V
// on 64 latent dims (each dim keeps its K order); threads < T issue the KV rows. The per-thread softmax sums of warps 2v (tile 0) and
// 2v + 1 (tile 1) are added in SMEM in the 4-warp order before the same shuffle tree, and the stats sum the 4 pair sums in order (kPair),
// so partials and stats are bitwise to W = 4. The pair exchange is ordered by bar.sync (CTA scope).
template <int T, int W>
constexpr int feed_bars() { return W * (T / W / 16) + 1; }
template <int T, int W, bool kFeed>
constexpr size_t smem_bytes() { return kFeed ? (sizeof(Smem<T, W>) + 7) / 8 * 8 + 8 * feed_bars<T, W>() : sizeof(Smem<T, W>); }
__device__ __forceinline__ void mbar_init(uint64_t* b, uint32_t n) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(smem_addr(b)), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void mbar_arrive_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(smem_addr(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];" ::"r"(smem_addr(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "SM6W_%=: mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n\t"
      "@!p bra SM6W_%=;\n\t}\n" ::"r"(smem_addr(b)),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(smem_addr(dst)),
               "l"(src), "r"(bytes), "r"(smem_addr(b))
               : "memory");
}

template <int H, int T, int W, bool kTag = false, bool kFence = true, bool kFeed = false>
__global__ void __launch_bounds__(32 * W, W == 8 ? 2 : 0) sparse_mla6_kernel(const __grid_constant__ Params p) {
  static_assert(H >= 1 && H <= 8 && W >= 4 && T % (16 * W) == 0, "heads <= 8; >= 4 warps; 16-token m-tiles per warp");
  static_assert(!kFeed || T == 32 * W || (W == 8 && T == 128), "kFeed: one KV row per thread (W8: threads < T)");
  constexpr int kThreads = 32 * W, kSplits = (kTopK + T - 1) / T, kTw = T / W, kMt = kTw / 16;
  // W = 8 (FEED_W8, see above): P.V on all 8 warps (64 dims each); kPair keeps the 4-warp summation order of l.
  constexpr bool kPair = W == 8;
  constexpr int kPvW = W == 8 ? 8 : 4, kDw = kLatent / kPvW, kJ = kDw / 32, kSw = kPair ? W / 2 : W;
  extern __shared__ __align__(128) uint8_t smem_raw[];
  Smem<T, W>& s = *reinterpret_cast<Smem<T, W>*>(smem_raw);
  const int split = blockIdx.x, row = blockIdx.y;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, t = lane % 4;

  pdl_trigger();  // MlaValueFold launches and stages W_UV; it waits for this grid before reading its outputs
  for (int i = tid; i < (8 - H) * kPitch / 16; i += kThreads) {  // pad heads: no predecessor data
    reinterpret_cast<uint4*>(s.q[H])[i] = make_uint4(0, 0, 0, 0);
  }
  uint64_t* const bars = reinterpret_cast<uint64_t*>(smem_raw + (sizeof(Smem<T, W>) + 7) / 8 * 8);  // kFeed only
  if constexpr (kFeed) {
    if (tid == 0) {
      for (int b = 0; b < W * kMt; ++b) mbar_init(&bars[b], 16);
      mbar_init(&bars[W * kMt], H);
      mbar_fence_init();
    }  // ordered before every arrive by the bar.sync after the slots
  }
  pdl_wait(); if constexpr (kFeed) asm volatile("fence.proxy.async.global;" ::: "memory");  // kFeed: async-proxy reads of generic writes

  const int len = min(ld_coherent(p.lengths + row), kTopK);
  if constexpr (kTag) {
    const uint32_t epoch = static_cast<uint32_t>(p.epochs[static_cast<int64_t>(row) * kSplits + split]) + 1u;
    const uint32_t tg = epoch & 0x7Fu;
    const uint32_t* mb = p.tagged + static_cast<int64_t>(row) * kTopK + split * T;
    const uint64_t t0 = globaltimer();
    for (int j = tid; j < T; j += kThreads) {
      const int slot = split * T + j < kTopK ? static_cast<int>(poll_tag(mb + j, tg, t0, p.deadline_ns) & 0x1FFFFFFu) - 1 : -1;
      s.slot[j] = (split * T + j < len && slot >= 0 && slot < p.num_slots) ? slot : -1;
    }
    if (row == 0 && split == 0 && tid < p.heart_blocks) {  // every rank's block, word 0 (see the comment above the kernel)
      poll_tag(p.tagged + static_cast<int64_t>(tid) * p.heart_rows * kTopK, tg, t0, p.deadline_ns);
    }
    if constexpr (kFence) fence_acq_rel_sys();  // every polling thread, after its observing loads
    __syncthreads();      // every thread read this CTA's counter before it advances
    if (tid == 0) p.epochs[static_cast<int64_t>(row) * kSplits + split] = static_cast<int32_t>(epoch);
  } else {
    const int32_t* sel = p.selected + row * p.sel_stride + split * T;
    for (int j = tid; j < T; j += kThreads) {
      const int slot = split * T + j < kTopK ? ld_coherent(sel + j) : -1;  // the last split may end past 2048
      s.slot[j] = (split * T + j < len && slot >= 0 && slot < p.num_slots) ? slot : -1;
    }
  }
  // Query (group 0 with the first m-tile of every warp's tokens).
  const uint8_t* q = p.q + row * p.q_stride;
  if constexpr (!kFeed) {
    for (int c = tid; c < H * (kDim / 16); c += kThreads) {
      cp_async16(&s.q[c / (kDim / 16)][(c % (kDim / 16)) * 16], q + c * 16);
    }
  }
  __syncthreads();  // slots visible to every warp
  if constexpr (kFeed) {
    if (tid < H) {
      mbar_arrive_tx(&bars[W * kMt], kDim);
      bulk_g2s(&s.q[tid][0], q + tid * kDim, kDim, &bars[W * kMt]);
    }
    if constexpr (W == 8) {  // FEED_W8: threads < T issue one KV row each (bar = the row's m-tile = its warp's)
      if (tid < T) {
        const int r = tid;
        uint64_t* const bar = &bars[(r / kTw) * kMt + (r % kTw) / 16];
        const int slot = s.slot[r];
        if (slot >= 0) {
          mbar_arrive_tx(bar, kDim);
          bulk_g2s(&s.kv[r][0], p.kv + static_cast<int64_t>(slot) * kDim, kDim, bar);
        } else {
#pragma unroll
          for (int c = 0; c < kDim / 16; ++c) reinterpret_cast<uint4*>(&s.kv[r][0])[c] = make_uint4(0, 0, 0, 0);
          mbar_arrive(bar);
        }
      }
    } else {
    const int r = tid;  // this thread's KV row (T == threads)
    uint64_t* const bar = &bars[(r / kTw) * kMt + (r % kTw) / 16];
    const int slot = s.slot[r];
    if (slot >= 0) {
      mbar_arrive_tx(bar, kDim);
      bulk_g2s(&s.kv[r][0], p.kv + static_cast<int64_t>(slot) * kDim, kDim, bar);
    } else {
#pragma unroll
      for (int c = 0; c < kDim / 16; ++c) reinterpret_cast<uint4*>(&s.kv[r][0])[c] = make_uint4(0, 0, 0, 0);
      mbar_arrive(bar);  // release: the zero row before the waiters' acquire
    }
    }
  } else {
    // KV rows of this warp's tokens, one commit group per m-tile (16 rows x 36 chunks; lanes walk the chunks).
#pragma unroll
    for (int mt = 0; mt < kMt; ++mt) {
      const int r0 = warp * kTw + mt * 16;
#pragma unroll
      for (int i = 0; i < 16 * (kDim / 16) / 32; ++i) {
        const int c = lane + 32 * i, r = r0 + c / (kDim / 16), col = (c % (kDim / 16)) * 16;
        const int slot = s.slot[r];
        if (slot >= 0) {
          cp_async16(&s.kv[r][col], p.kv + static_cast<int64_t>(slot) * kDim + col);
        } else {
          *reinterpret_cast<uint4*>(&s.kv[r][col]) = make_uint4(0, 0, 0, 0);
        }
      }
      cp_async_commit();
    }
  }  // !kFeed

  // S^T = K Q^T per m-tile: A = 16 KV rows x 32 bytes (ldmatrix), B = Q^T (heads g < 8, 4 bytes per k-step half).
  float acc[kMt][2][4];
#pragma unroll
  for (int mt = 0; mt < kMt; ++mt)
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[mt][0][e] = acc[mt][1][e] = 0.f;
  uint32_t qb[kDim / 32][2];
#pragma unroll
  for (int mt = 0; mt < kMt; ++mt) {
    if constexpr (kFeed) {
      if (mt == 0) mbar_wait(&bars[W * kMt], 0);  // the query
      mbar_wait(&bars[warp * kMt + mt], 0);       // this warp's 16 rows of this m-tile
    } else {
      cp_async_wait_pending(kMt - 1 - mt);  // this m-tile's group (group 0 also carries the query)
      // Every thread waited for its own groups; bar.sync makes all of them visible CTA-wide (CUDA PG: __syncthreads
      // after the cp.async completion wait; a warp barrier is not documented to order another lane's cp.async writes).
      __syncthreads();
    }
    if (mt == 0) {
#pragma unroll
      for (int ks = 0; ks < kDim / 32; ++ks) {
        qb[ks][0] = *reinterpret_cast<const uint32_t*>(&s.q[g][32 * ks + 4 * t]);
        qb[ks][1] = *reinterpret_cast<const uint32_t*>(&s.q[g][32 * ks + 16 + 4 * t]);
      }
    }
    const uint8_t* a_row = &s.kv[warp * kTw + mt * 16 + (lane % 8) + 8 * ((lane / 8) & 1)][16 * (lane / 16)];
#pragma unroll
    for (int ks = 0; ks < kDim / 32; ++ks) {
      uint32_t a[4];
      ldsm_x4(a, a_row + 32 * ks);
      mma_e4m3(acc[mt][ks & 1], a, qb[ks][0], qb[ks][1]);
    }
  }

  // Softmax statistics over the split (heads 2t, 2t+1; tokens g and g+8 of every m-tile).
  float sc[kMt][4];
  float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
  for (int mt = 0; mt < kMt; ++mt) {
    const int r = warp * kTw + mt * 16 + g;
    const bool v0 = s.slot[r] >= 0, v1 = s.slot[r + 8] >= 0;
    sc[mt][0] = v0 ? (acc[mt][0][0] + acc[mt][1][0]) * p.scale_log2 : -INFINITY;
    sc[mt][1] = v0 ? (acc[mt][0][1] + acc[mt][1][1]) * p.scale_log2 : -INFINITY;
    sc[mt][2] = v1 ? (acc[mt][0][2] + acc[mt][1][2]) * p.scale_log2 : -INFINITY;
    sc[mt][3] = v1 ? (acc[mt][0][3] + acc[mt][1][3]) * p.scale_log2 : -INFINITY;
    mx0 = fmaxf(mx0, fmaxf(sc[mt][0], sc[mt][2]));
    mx1 = fmaxf(mx1, fmaxf(sc[mt][1], sc[mt][3]));
  }
#pragma unroll
  for (int o = 4; o < 32; o <<= 1) {
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, o));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, o));
  }
  if (g == 0) {
    s.max[warp][2 * t] = mx0;
    s.max[warp][2 * t + 1] = mx1;
  }
  __syncthreads();  // also: every warp's KV rows have landed (each thread waited for its own groups)
  float m0 = s.max[0][2 * t], m1 = s.max[0][2 * t + 1];
#pragma unroll
  for (int w = 1; w < W; ++w) {
    m0 = fmaxf(m0, s.max[w][2 * t]);
    m1 = fmaxf(m1, s.max[w][2 * t + 1]);
  }
  const float u0 = m0 == -INFINITY ? 0.f : m0, u1 = m1 == -INFINITY ? 0.f : m1;
  float l0 = 0.f, l1 = 0.f;
  __half* ph = reinterpret_cast<__half*>(&s.p[0][0]);
  constexpr int kPh = Smem<T, W>::kPPitch / 2;
#pragma unroll
  for (int mt = 0; mt < kMt; ++mt) {
    const int r = warp * kTw + mt * 16 + g;
    const float p0 = ex2(sc[mt][0] - u0), p1 = ex2(sc[mt][1] - u1);
    const float p2 = ex2(sc[mt][2] - u0), p3 = ex2(sc[mt][3] - u1);
    l0 += p0 + p2;
    l1 += p1 + p3;
    ph[(2 * t) * kPh + r] = __float2half_rn(p0);
    ph[(2 * t + 1) * kPh + r] = __float2half_rn(p1);
    ph[(2 * t) * kPh + r + 8] = __float2half_rn(p2);
    ph[(2 * t + 1) * kPh + r + 8] = __float2half_rn(p3);
  }
  if constexpr (kPair) {
    // own region after Smem (+ the feed bars) = smem_bytes<T, W, kFeed>() on the host side
    constexpr size_t kLscOff = kFeed ? (sizeof(Smem<T, W>) + 7) / 8 * 8 + 8 * (W * kMt + 1) : sizeof(Smem<T, W>);
    float* const lsc = reinterpret_cast<float*>(smem_raw + kLscOff);
    if (warp & 1) {
      lsc[((warp / 2) * 32 + lane) * 2] = l0;
      lsc[((warp / 2) * 32 + lane) * 2 + 1] = l1;
    }
    __syncthreads();
    if (!(warp & 1)) {
      l0 = l0 + lsc[((warp / 2) * 32 + lane) * 2];
      l1 = l1 + lsc[((warp / 2) * 32 + lane) * 2 + 1];
    }
  }
#pragma unroll
  for (int o = 4; o < 32; o <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, o);
    l1 += __shfl_xor_sync(0xffffffffu, l1, o);
  }
  if (g == 0 && (!kPair || !(warp & 1))) {
    s.sum[kPair ? warp / 2 : warp][2 * t] = l0;
    s.sum[kPair ? warp / 2 : warp][2 * t + 1] = l1;
  }
  __syncthreads();  // P and the partial sums visible to every warp

  if (warp >= kPvW) return;  // P.V and the outputs: warps 0-3 (W8: 0-7) (no barrier follows)
  // U^T = V^T P^T: warp w, latent dims [128w, 128w+128) as 8 m-tiles; m-row i <-> dim 2i, m-row i+8 <-> dim 2i+1
  // inside each 16-dim block (the ldmatrix.trans pair order); K = the split's T tokens.
  float o[kJ][2][4];
#pragma unroll
  for (int j = 0; j < kJ; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) o[j][0][e] = o[j][1][e] = 0.f;
  if constexpr (kFeed) {
    for (int b = 0; b < W * kMt; ++b) mbar_wait(&bars[b], 0);  // P.V reads every warp's rows (all complete by now)
  }
  const uint8_t* v_row = &s.kv[(lane % 8) + 8 * ((lane / 8) & 1)][kDw * warp + 16 * (lane / 16)];
#pragma unroll
  for (int kk = 0; kk < T / 16; ++kk) {
    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&s.p[g][(16 * kk + 2 * t) * 2]);
    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&s.p[g][(16 * kk + 8 + 2 * t) * 2]);
#pragma unroll
    for (int j = 0; j < kJ; ++j) {
      uint32_t r[4];
      ldsm_x4_trans(r, v_row + 16 * kk * kPitch + 32 * j);
      uint32_t a0, a1, a2, a3, c0, c1, c2, c3;
      fp8_pairs_to_f16(r[0], a0, a1);  // tokens 0-7: dims 2g (a0), 2g+1 (a1) of the low 16-dim block
      fp8_pairs_to_f16(r[1], a2, a3);  // tokens 8-15
      fp8_pairs_to_f16(r[2], c0, c1);  // high 16-dim block
      fp8_pairs_to_f16(r[3], c2, c3);
      mma_f16(o[j][0], a0, a1, a2, a3, b0, b1);
      mma_f16(o[j][1], c0, c1, c2, c3, b0, b1);
    }
  }

  // Outputs: U[h][dim] (h = 2t, 2t+1 < H), stats (m, l) per head.
  const int64_t base = (static_cast<int64_t>(row) * kSplits + split) * H;
#pragma unroll
  for (int j = 0; j < kJ; ++j)
#pragma unroll
    for (int hb = 0; hb < 2; ++hb) {
      const int dim = kDw * warp + 32 * j + 16 * hb + 2 * g;
      if (2 * t < H) {
        *reinterpret_cast<float2*>(p.partial + (base + 2 * t) * kLatent + dim) = make_float2(o[j][hb][0], o[j][hb][2]);
      }
      if (2 * t + 1 < H) {
        *reinterpret_cast<float2*>(p.partial + (base + 2 * t + 1) * kLatent + dim) =
            make_float2(o[j][hb][1], o[j][hb][3]);
      }
    }
  if (tid < H) {
    float m = s.max[0][tid], l = s.sum[0][tid];
    if constexpr (W == 8) {
#pragma unroll
      for (int w = 1; w < W; ++w) m = fmaxf(m, s.max[w][tid]);
#pragma unroll
      for (int w = 1; w < kSw; ++w) l += s.sum[w][tid];
    } else {
#pragma unroll
      for (int w = 1; w < W; ++w) {
        m = fmaxf(m, s.max[w][tid]);
        l += s.sum[w][tid];
      }
    }
    p.stats[base + tid] = make_float2(m, l);
  }
}

}  // namespace k31_sparse_mla6

#ifdef K31_SMLA6_PTX_INSTANTIATE
// Device-only build for PTX inspection (nvcc -ptx -DK31_SMLA6_PTX_INSTANTIATE -DK31_SMLA6_DEVICE_ONLY).
template __global__ void k31_sparse_mla6::sparse_mla6_kernel<6, 64, 4>(const __grid_constant__ k31_sparse_mla6::Params);
template __global__ void k31_sparse_mla6::sparse_mla6_kernel<6, 80, 5>(const __grid_constant__ k31_sparse_mla6::Params);
template __global__ void k31_sparse_mla6::sparse_mla6_kernel<6, 96, 6>(const __grid_constant__ k31_sparse_mla6::Params);
template __global__ void k31_sparse_mla6::sparse_mla6_kernel<6, 128, 4>(const __grid_constant__ k31_sparse_mla6::Params);
template __global__ void k31_sparse_mla6::sparse_mla6_kernel<6, 256, 4>(const __grid_constant__ k31_sparse_mla6::Params);
template __global__ void k31_sparse_mla6::sparse_mla6_kernel<6, 128, 4, true>(const __grid_constant__ k31_sparse_mla6::Params);
template __global__ void k31_sparse_mla6::sparse_mla6_kernel<6, 128, 4, true, false>(const __grid_constant__ k31_sparse_mla6::Params);
template __global__ void k31_sparse_mla6::sparse_mla6_kernel<6, 128, 4, false, true, true>(const __grid_constant__ k31_sparse_mla6::Params);
template __global__ void k31_sparse_mla6::sparse_mla6_kernel<6, 128, 4, true, true, true>(const __grid_constant__ k31_sparse_mla6::Params);
template __global__ void k31_sparse_mla6::sparse_mla6_kernel<6, 128, 4, true, false, true>(const __grid_constant__ k31_sparse_mla6::Params);
#ifdef K31_SMLA6_PTX_W8  // separate build: in one translation unit with the W = 4 feed kernels ptxas renumbers their registers
template __global__ void k31_sparse_mla6::sparse_mla6_kernel<6, 128, 8, false, true, true>(const __grid_constant__ k31_sparse_mla6::Params);
#endif
#endif

#ifndef K31_SMLA6_DEVICE_ONLY
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#ifndef K31_SMLA6_FEED
#define K31_SMLA6_FEED 0
#endif

namespace k31_sparse_mla6 {

// K31MK_SPARSE_MLA6_FEED: sparse_mla6.py builds a second module with -DK31_SMLA6_FEED=1; its T 128 launches take the kFeed kernels.
constexpr bool kFeedBuild = K31_SMLA6_FEED != 0;
#ifndef K31_SMLA6_W8
#define K31_SMLA6_W8 0
#endif
// K31MK_SPARSE_MLA6_FEED_W8: sparse_mla6.py builds the feed module with -DK31_SMLA6_W8=1 too; its untagged T 128 launches take the
// 8-warp feed kernel (the tagged route keeps the 4-warp kernels).
constexpr bool kW8Build = K31_SMLA6_W8 != 0 && kFeedBuild;
constexpr size_t kW8Smem = smem_bytes<128, 8, true>() + 4 * 32 * 2 * sizeof(float);  // + the l pair region

struct Binding {
  static void run(const tvm::ffi::TensorView q, const tvm::ffi::TensorView kv, const tvm::ffi::TensorView selected,
                  const tvm::ffi::TensorView lengths, const tvm::ffi::TensorView partial,
                  const tvm::ffi::TensorView stats, const double scale_log2, const int64_t tokens) {
    using namespace sglang::host;
    const int64_t rows = selected.size(0);
    const int64_t heads = partial.size(2);
    RuntimeCheck(heads == 6, "sparse MLA6 serves 6 heads per rank (TP16)");
    RuntimeCheck(tokens == 64 || tokens == 80 || tokens == 96 || tokens == 128 || tokens == 256,
                 "tokens per split: 64, 80, 96, 128 or 256");
    const int64_t splits = (kTopK + tokens - 1) / tokens;
    RuntimeCheck(rows >= 1 && rows <= 65535, "rows");
    RuntimeCheck(q.size(0) >= rows && q.stride(0) % 16 == 0 && q.stride(0) >= heads * kDim, "query [rows, H, 576]");
    RuntimeCheck(selected.size(1) == kTopK && selected.stride(1) == 1, "selected [rows, 2048] int32");
    RuntimeCheck(lengths.size(0) >= rows, "lengths [rows]");
    RuntimeCheck(kv.ndim() == 1 && kv.stride(0) == 1 && kv.size(0) % kDim == 0, "kv pool: flat uint8 [slots * 576]");
    RuntimeCheck(partial.ndim() == 4 && partial.size(0) == rows && partial.size(1) == splits &&
                     partial.size(3) == kLatent && partial.stride(3) == 1 && partial.stride(2) == kLatent &&
                     partial.stride(1) == heads * kLatent && partial.stride(0) == splits * heads * kLatent,
                 "partial [rows, C, H, 512] contiguous");
    RuntimeCheck(stats.ndim() == 4 && stats.size(0) == rows && stats.size(1) == splits && stats.size(2) == heads &&
                     stats.size(3) == 2 && stats.stride(3) == 1 && stats.stride(2) == 2 && stats.stride(1) == heads * 2 &&
                     stats.stride(0) == splits * heads * 2,
                 "stats [rows, C, H, 2] contiguous");
    RuntimeCheck(q.ndim() == 2 && q.stride(1) == 1 && lengths.stride(0) == 1, "query rows / lengths layouts");
    RuntimeCheck(reinterpret_cast<uintptr_t>(kv.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0,
                 "16-byte aligned query / pool");
    Params p{static_cast<const uint8_t*>(q.data_ptr()),
             static_cast<const uint8_t*>(kv.data_ptr()),
             static_cast<const int32_t*>(selected.data_ptr()),
             static_cast<const int32_t*>(lengths.data_ptr()),
             static_cast<float*>(partial.data_ptr()),
             static_cast<float2*>(stats.data_ptr()),
             q.stride(0),
             selected.stride(0),
             kv.size(0) / kDim,
             static_cast<float>(scale_log2)};
    auto launch = [&](auto kernel, size_t bytes, int warps) {
      CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
      LaunchKernel(dim3(static_cast<unsigned>(splits), static_cast<unsigned>(rows)), 32 * warps, kv.device(), bytes)
          .config({.use_pdl = true})(kernel, p);
    };
    if (tokens == 64) launch(sparse_mla6_kernel<6, 64, 4>, sizeof(Smem<64, 4>), 4);
    if (tokens == 80) launch(sparse_mla6_kernel<6, 80, 5>, sizeof(Smem<80, 5>), 5);
    if (tokens == 96) launch(sparse_mla6_kernel<6, 96, 6>, sizeof(Smem<96, 6>), 6);
    if (tokens == 128) {
      if constexpr (kW8Build) launch(sparse_mla6_kernel<6, 128, 8, false, true, true>, kW8Smem, 8);
      else launch(sparse_mla6_kernel<6, 128, 4, false, true, kFeedBuild>, smem_bytes<128, 4, kFeedBuild>(), 4);
    }
    if (tokens == 256) launch(sparse_mla6_kernel<6, 256, 4>, sizeof(Smem<256, 4>), 4);
  }

  // KNOWN K-13 (fail closed): MlaValueFold waits on this grid through PDL, which orders kernels of ONE stream only.
  // run() launches on tvm-ffi's env stream (LaunchKernel resolves it from kv's device); the fold launches on
  // ``expected_stream`` (verify.py: torch's current stream). This variant compares the two first and launches
  // nothing when they differ, so no partial is written and no dependent is released. sparse_mla6.py calls it on the
  // first launch per device and per graph capture; every other launch is run().
  static void run_checked(const tvm::ffi::TensorView q, const tvm::ffi::TensorView kv,
                          const tvm::ffi::TensorView selected, const tvm::ffi::TensorView lengths,
                          const tvm::ffi::TensorView partial, const tvm::ffi::TensorView stats,
                          const double scale_log2, const int64_t tokens, const int64_t expected_stream) {
    using namespace sglang::host;
    const int64_t launch_stream = reinterpret_cast<int64_t>(LaunchKernel::resolve_device(kv.device()));
    RuntimeCheck(launch_stream == expected_stream, "K-13 stream check: tvm-ffi launch stream ", launch_stream,
                 " != dependent stream ", expected_stream, "; sparse MLA6 core not launched");
    run(q, kv, selected, lengths, partial, stats, scale_log2, tokens);
  }

  // Tagged mode (K31MK_DSA_REQ_SHARD_AG=fused): the slots of batch row r are row r of ``tagged`` ([blocks x heart_rows,
  // 2048] int32 tagged words, this rank's mailbox copy of this DSA layer), polled by the core itself (see the kernel).
  // ``epochs`` [rows, C] int32: uses of this copy so far per (row, split). rows = partial.size(0) (the real batch rows).
  // ``fence`` 0: the fence-free form (kFence = false, TAGNF; see the kernel); 1: today's form.
  static void run_tagged(const tvm::ffi::TensorView q, const tvm::ffi::TensorView kv, const tvm::ffi::TensorView tagged,
                         const tvm::ffi::TensorView epochs, const tvm::ffi::TensorView lengths,
                         const tvm::ffi::TensorView partial, const tvm::ffi::TensorView stats, const double scale_log2,
                         const int64_t tokens, const int64_t deadline_ns, const int64_t heart_rows,
                         const int64_t heart_blocks, const int64_t fence, const int64_t expected_stream) {
    using namespace sglang::host;
    const int64_t launch_stream = reinterpret_cast<int64_t>(LaunchKernel::resolve_device(kv.device()));
    RuntimeCheck(expected_stream == 0 || launch_stream == expected_stream, "K-13 stream check: tvm-ffi launch stream ",
                 launch_stream, " != dependent stream ", expected_stream, "; sparse MLA6 core not launched");
    const int64_t rows = partial.size(0);
    const int64_t heads = partial.size(2);
    RuntimeCheck(heads == 6, "sparse MLA6 serves 6 heads per rank (TP16)");
    RuntimeCheck(tokens == 64 || tokens == 80 || tokens == 96 || tokens == 128 || tokens == 256,
                 "tokens per split: 64, 80, 96, 128 or 256");
    const int64_t splits = (kTopK + tokens - 1) / tokens;
    RuntimeCheck(rows >= 1 && rows <= 65535, "rows");
    RuntimeCheck(heart_rows >= 1 && heart_blocks >= 1 && heart_blocks <= 32 * 4 && heart_rows * heart_blocks >= rows,
                 "tagged: blocks x rows per block must cover the batch rows (and at most 128 blocks)");
    RuntimeCheck(tagged.ndim() == 2 && tagged.size(0) >= heart_rows * heart_blocks && tagged.size(1) == kTopK &&
                     tagged.stride(1) == 1 && tagged.stride(0) == kTopK,
                 "tagged [blocks x rows per block, 2048] int32 contiguous");
    RuntimeCheck(epochs.ndim() == 2 && epochs.size(0) == rows && epochs.size(1) == splits && epochs.stride(1) == 1 &&
                     epochs.stride(0) == splits,
                 "epochs [rows, C] int32 contiguous");
    RuntimeCheck(deadline_ns > 0, "tagged: deadline_ns > 0");
    RuntimeCheck(q.size(0) >= rows && q.stride(0) % 16 == 0 && q.stride(0) >= heads * kDim, "query [rows, H, 576]");
    RuntimeCheck(lengths.size(0) >= rows, "lengths [rows]");
    RuntimeCheck(kv.ndim() == 1 && kv.stride(0) == 1 && kv.size(0) % kDim == 0, "kv pool: flat uint8 [slots * 576]");
    RuntimeCheck(partial.ndim() == 4 && partial.size(1) == splits && partial.size(3) == kLatent && partial.stride(3) == 1 &&
                     partial.stride(2) == kLatent && partial.stride(1) == heads * kLatent &&
                     partial.stride(0) == splits * heads * kLatent,
                 "partial [rows, C, H, 512] contiguous");
    RuntimeCheck(stats.ndim() == 4 && stats.size(0) == rows && stats.size(1) == splits && stats.size(2) == heads &&
                     stats.size(3) == 2 && stats.stride(3) == 1 && stats.stride(2) == 2 && stats.stride(1) == heads * 2 &&
                     stats.stride(0) == splits * heads * 2,
                 "stats [rows, C, H, 2] contiguous");
    RuntimeCheck(q.ndim() == 2 && q.stride(1) == 1 && lengths.stride(0) == 1, "query rows / lengths layouts");
    RuntimeCheck(reinterpret_cast<uintptr_t>(kv.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0,
                 "16-byte aligned query / pool");
    Params p{static_cast<const uint8_t*>(q.data_ptr()),
             static_cast<const uint8_t*>(kv.data_ptr()),
             nullptr,
             static_cast<const int32_t*>(lengths.data_ptr()),
             static_cast<float*>(partial.data_ptr()),
             static_cast<float2*>(stats.data_ptr()),
             q.stride(0),
             0,
             kv.size(0) / kDim,
             static_cast<float>(scale_log2),
             static_cast<const uint32_t*>(tagged.data_ptr()),
             static_cast<int32_t*>(epochs.data_ptr()),
             deadline_ns,
             static_cast<int32_t>(heart_rows),
             static_cast<int32_t>(heart_blocks)};
    auto launch = [&](auto kernel, size_t bytes, int warps) {
      RuntimeCheck(heart_blocks <= 32 * warps, "tagged: one heartbeat poll per thread of CTA (0, 0)");
      CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
      LaunchKernel(dim3(static_cast<unsigned>(splits), static_cast<unsigned>(rows)), 32 * warps, kv.device(), bytes)
          .config({.use_pdl = true})(kernel, p);
    };
    RuntimeCheck(fence == 0 || fence == 1, "tagged: fence 0 or 1");
    if (fence) {
      if (tokens == 64) launch(sparse_mla6_kernel<6, 64, 4, true>, sizeof(Smem<64, 4>), 4);
      if (tokens == 80) launch(sparse_mla6_kernel<6, 80, 5, true>, sizeof(Smem<80, 5>), 5);
      if (tokens == 96) launch(sparse_mla6_kernel<6, 96, 6, true>, sizeof(Smem<96, 6>), 6);
      if (tokens == 128) launch(sparse_mla6_kernel<6, 128, 4, true, true, kFeedBuild>, smem_bytes<128, 4, kFeedBuild>(), 4);
      if (tokens == 256) launch(sparse_mla6_kernel<6, 256, 4, true>, sizeof(Smem<256, 4>), 4);
    } else {
      if (tokens == 64) launch(sparse_mla6_kernel<6, 64, 4, true, false>, sizeof(Smem<64, 4>), 4);
      if (tokens == 80) launch(sparse_mla6_kernel<6, 80, 5, true, false>, sizeof(Smem<80, 5>), 5);
      if (tokens == 96) launch(sparse_mla6_kernel<6, 96, 6, true, false>, sizeof(Smem<96, 6>), 6);
      if (tokens == 128) launch(sparse_mla6_kernel<6, 128, 4, true, false, kFeedBuild>, smem_bytes<128, 4, kFeedBuild>(), 4);
      if (tokens == 256) launch(sparse_mla6_kernel<6, 256, 4, true, false>, sizeof(Smem<256, 4>), 4);
    }
  }
};

}  // namespace k31_sparse_mla6
#endif
