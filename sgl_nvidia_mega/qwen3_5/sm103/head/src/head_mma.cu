// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#if !defined(HEAD_N_TILE) || !defined(HEAD_N_CHUNKS) || !defined(HEAD_STAGES) || !defined(HEAD_SMEM_BYTES) || \
    !defined(HEAD_CLUSTER)
#error "head_mma.cu: the launcher passes HEAD_N_TILE, HEAD_N_CHUNKS, HEAD_STAGES, HEAD_SMEM_BYTES, HEAD_CLUSTER"
#endif
#ifndef HEAD_KEYS
#define HEAD_KEYS 1
#endif
#ifndef HEAD_GUMBEL1
#define HEAD_GUMBEL1 0
#endif
#ifndef HEAD_NOISE
#define HEAD_NOISE 1
#endif

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef signed int int32_t;
typedef unsigned long long uint64_t;
typedef signed long long int64_t;
typedef unsigned long long uintptr_t;

typedef struct __align__(64) {
  uint64_t opaque[16];
} CUtensorMap;

#include <cuda_bf16.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000 || __CUDA_ARCH__ >= 1100)
#error "head_mma.cu targets SM100-family GPUs (tcgen05)"
#endif

#define HEAD_DEVICE __forceinline__ __device__

namespace head_mma {

inline constexpr int kHidden = 4096;
inline constexpr int kBlockK = 64;
inline constexpr int kKBlocks = kHidden / kBlockK;
inline constexpr int kRowsTile = 128;  // MMA M
inline constexpr int kNTile = HEAD_N_TILE;
inline constexpr int kNChunks = HEAD_N_CHUNKS;
inline constexpr int kStages = HEAD_STAGES;
inline constexpr int kCluster = HEAD_CLUSTER;
inline constexpr bool kKeys = HEAD_KEYS != 0;
inline constexpr bool kGumbel1 = HEAD_GUMBEL1 != 0;
inline constexpr int kKinds = 1;  // the Gumbel build folds through the idle ring instead (see its epilogue)
inline constexpr int kXRows = HEAD_N_TILE / HEAD_CLUSTER;  // x rows a cluster CTA loads per tile
inline constexpr int kTokens = kNTile * kNChunks;
inline constexpr int kTmemCols = kTokens <= 32 ? 32 : kTokens <= 64 ? 64 : kTokens <= 128 ? 128 : kTokens <= 256 ? 256 : 512;
inline constexpr int kWTileBytes = kRowsTile * kBlockK * 2;
inline constexpr int kXTileBytes = kNTile * kBlockK * 2;
inline constexpr int kWarps = 6;
inline constexpr int kThreads = kWarps * 32;
inline constexpr int kEpiWarp0 = 2;
inline constexpr int kEpiThreads = 4 * 32;
static_assert(kNTile % 16 == 0 && kNTile >= 16 && kNTile <= 256, "M = 128 needs N in 16..256 step 16");
static_assert(kNChunks == 1 || kNChunks == 2, "one or two token tiles");
static_assert(kTokens <= 512, "TMEM holds 512 fp32 columns");
inline constexpr int kTmemAlloc = kGumbel1 ? 2 * kTmemCols : kTmemCols;
static_assert(!kGumbel1 || (kKeys && kNChunks == 1 && kCluster == 1 && kTokens <= 128), "Gumbel build: <= 128 tokens");
static_assert(kCluster == 1 || kCluster == 2 || kCluster == 4, "cluster of 1, 2 or 4");
static_assert(kXRows % 8 == 0, "an x slice is whole 8-row swizzle atoms");

struct alignas(1024) Smem {
  alignas(1024) uint8_t w[kStages][kWTileBytes];
  alignas(1024) uint8_t x[kStages][kNChunks * kXTileBytes];
  int32_t top[kKinds][4][kTokens];
  uint32_t row[kKinds][4][kTokens];
  uint64_t full[kStages];
  uint64_t empty[kStages];
  uint64_t mma_done;
  uint32_t taddr;
};
static_assert(sizeof(Smem) == HEAD_SMEM_BYTES, "the launcher's dynamic smem must equal sizeof(Smem)");

HEAD_DEVICE uint32_t to_shared(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
HEAD_DEVICE void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(to_shared(bar)), "r"(count));
}
HEAD_DEVICE void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(to_shared(bar)), "r"(bytes));
}
HEAD_DEVICE void mbar_wait_parity(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "WAIT_%=: mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n\t"
      "@!p bra WAIT_%=;\n\t}\n" ::"r"(to_shared(bar)),
      "r"(parity));
}
HEAD_DEVICE void fence_mbarrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
HEAD_DEVICE void prefetch_tensormap(const void* tmap) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(tmap) : "memory");
}
HEAD_DEVICE void tma_2d(uint32_t dst, const CUtensorMap* tmap, int32_t x, int32_t y, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];" ::"r"(dst),
      "l"(tmap), "r"(x), "r"(y), "r"(to_shared(bar))
      : "memory");
}
HEAD_DEVICE void tma_2d_mc(uint32_t dst, const CUtensorMap* tmap, int32_t x, int32_t y, uint64_t* bar,
                           uint16_t mask) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
      " [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(dst),
      "l"(tmap), "r"(x), "r"(y), "r"(to_shared(bar)), "h"(mask)
      : "memory");
}
HEAD_DEVICE uint32_t cluster_rank() {
  uint32_t r;
  asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
HEAD_DEVICE void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
HEAD_DEVICE bool elect_one() {
  uint32_t pred;
  asm volatile(
      "{\n .reg .pred p;\n .reg .b32 r;\n elect.sync r|p, 0xFFFFFFFF;\n selp.b32 %0, 1, 0, p;\n}\n"
      : "=r"(pred));
  return pred != 0;
}
HEAD_DEVICE void named_barrier(uint32_t id, uint32_t n) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}
HEAD_DEVICE void pdl_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}
HEAD_DEVICE void pdl_trigger() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
HEAD_DEVICE void tmem_alloc(uint32_t dst, uint32_t cols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(dst), "r"(cols));
}
HEAD_DEVICE void tmem_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}
HEAD_DEVICE void tmem_dealloc(uint32_t taddr, uint32_t cols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(cols));
}
HEAD_DEVICE void mma_commit(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" ::"r"(to_shared(bar)));
}
HEAD_DEVICE void mma_commit_mc(uint64_t* bar, uint16_t mask) {
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" ::"r"(
          to_shared(bar)),
      "h"(mask));
}
HEAD_DEVICE void tc_fence_after() {
  asm volatile("tcgen05.fence::after_thread_sync;");
}
HEAD_DEVICE void tc_fence_before() {
  asm volatile("tcgen05.fence::before_thread_sync;");
}
HEAD_DEVICE void tmem_wait_ld() {
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
HEAD_DEVICE void tmem_ld_x16(uint32_t taddr, uint32_t* d) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]), "=r"(d[4]), "=r"(d[5]), "=r"(d[6]), "=r"(d[7]),
        "=r"(d[8]), "=r"(d[9]), "=r"(d[10]), "=r"(d[11]), "=r"(d[12]), "=r"(d[13]), "=r"(d[14]), "=r"(d[15])
      : "r"(taddr));
}
HEAD_DEVICE void tmem_st_x16(uint32_t taddr, const uint32_t* d) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], "
      "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};" ::"r"(taddr),
      "r"(d[0]), "r"(d[1]), "r"(d[2]), "r"(d[3]), "r"(d[4]), "r"(d[5]), "r"(d[6]), "r"(d[7]), "r"(d[8]), "r"(d[9]),
      "r"(d[10]), "r"(d[11]), "r"(d[12]), "r"(d[13]), "r"(d[14]), "r"(d[15])
      : "memory");
}
HEAD_DEVICE void tmem_wait_st() {
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}
HEAD_DEVICE float g_mul(float a, float b) {
  float d;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
HEAD_DEVICE float g_sub(float a, float b) {
  float d;
  asm("sub.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
HEAD_DEVICE float g_add(float a, float b) {
  float d;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
HEAD_DEVICE float g_max(float a, float b) {
  float d;
  asm("max.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
HEAD_DEVICE float g_min(float a, float b) {
  float d;
  asm("min.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
HEAD_DEVICE float g_div_full(float a, float b) {
  float d;
  asm("div.full.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
HEAD_DEVICE float g_i2f(int32_t a) {
  float d;
  asm("cvt.rn.f32.s32 %0, %1;" : "=f"(d) : "r"(a));
  return d;
}
HEAD_DEVICE uint32_t g_philox_word(uint32_t c0, uint32_t k0, uint32_t k1, int w) {
  uint32_t c1 = 0u, c2 = 0u, c3 = 0u;
#pragma unroll
  for (int i = 0; i < 10; ++i) {
    const uint32_t a0 = c0, a2 = c2;
    c0 = __umulhi(0xCD9E8D57u, a2) ^ c1 ^ k0;
    c2 = __umulhi(0xD2511F53u, a0) ^ c3 ^ k1;
    c1 = 0xCD9E8D57u * a2;
    c3 = 0xD2511F53u * a0;
    k0 += 0x9E3779B9u;
    k1 += 0xBB67AE85u;
  }
  return w == 0 ? c0 : w == 1 ? c1 : w == 2 ? c2 : c3;
}
HEAD_DEVICE float g_uniform(uint32_t w) {
  int32_t x = static_cast<int32_t>(w);
  x = x < 0 ? static_cast<int32_t>(~w) : x;
  return g_min(g_max(g_mul(g_i2f(x), 4.6566127342e-10f), 1e-12f), 0.99999994f);
}
HEAD_DEVICE bool g_gt(float a, float b) {
  uint32_t r;
  asm("{ .reg .pred q; setp.gt.f32 q, %1, %2; selp.u32 %0, 1, 0, q; }" : "=r"(r) : "f"(a), "f"(b));
  return r != 0u;
}
HEAD_DEVICE int32_t g_order(float v) {
  v = (v == v) ? v : __int_as_float(0xFF800000);
  v = g_add(v, 0.0f);
  const int32_t b = __float_as_int(v);
  return b < 0 ? (b ^ 0x7FFFFFFF) : b;
}
HEAD_DEVICE void store_logits_chunk(const __nv_bfloat16* stage, __nv_bfloat16* logits, long long stride, int m, int t0,
                                    int row0, int rows_here, int et) {
  const int jt = et >> 3, r0 = (et & 7) * 16;
  const int t = t0 + jt;
  if (t >= m || r0 >= rows_here) return;
  const __nv_bfloat16* src = stage + jt * 128 + r0;
  __nv_bfloat16* dst = logits + static_cast<long long>(t) * stride + row0 + r0;
  if (r0 + 16 <= rows_here && (reinterpret_cast<uintptr_t>(dst) & 15) == 0) {
    reinterpret_cast<uint4*>(dst)[0] = reinterpret_cast<const uint4*>(src)[0];
    reinterpret_cast<uint4*>(dst)[1] = reinterpret_cast<const uint4*>(src)[1];
  } else {
    for (int i = 0; i < 16 && r0 + i < rows_here; ++i) dst[i] = src[i];
  }
}
HEAD_DEVICE void mma_bf16(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t accumulate) {
  asm volatile(
      "{ .reg .pred p; setp.ne.b32 p, %4, 0;"
      "  tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p; }" ::"r"(d),
      "l"(a), "l"(b), "r"(idesc), "r"(accumulate));
}
HEAD_DEVICE uint64_t smem_desc(uint32_t addr) {
  auto enc = [](uint32_t v) -> uint64_t { return uint64_t((v & 0x3FFFFu) >> 4); };
  return enc(addr) | (enc(8u * 128u) << 32) | (uint64_t(1) << 46) | (uint64_t(2) << 61);
}
__host__ __device__ constexpr uint32_t idesc_bf16(uint32_t m, uint32_t n) {
  return (1u << 4) | (1u << 7) | (1u << 10) | (((n >> 3) & 0x3Fu) << 17) | (((m >> 4) & 0x1Fu) << 24);
}
}  // namespace head_mma

struct HeadMmaParams {
  long long* keys;          // folded keys (KEY_FLOOR between calls): row t at keys[t * keys_stride]
  unsigned int* done;       // [1] finished CTAs, or null
  __nv_bfloat16* logits;
  long long logits_stride;
  int m;
  int valid;                // shard rows that are real ids
  int vocab_start;          // global id of shard row 0
  int box_rows;             // rows a CTA owns (the W box height)
  int nan_hi;               // order key of a NaN logit
  long long keys_stride;    // int64 elements between rows' keys
  const long long* seeds;
  long long* pre;
  unsigned int* epoch;
};
namespace head_mma {
inline constexpr unsigned long long kSeedBase = 0x5EED6A3Bull << 20;
HEAD_DEVICE void epoch_ticket(unsigned int* epoch, long long* tag, unsigned int seen) {
  __threadfence();
  const unsigned int old = atomicAdd(epoch + 1, 1u);
  if (old == gridDim.x - 1) {
    __threadfence();
    *reinterpret_cast<volatile long long*>(tag) = static_cast<long long>(seen);
    epoch[1] = 0u;
    __threadfence();
    atomicAdd(epoch, 1u);
  }
}
}  // namespace head_mma

extern "C" __global__ void __launch_bounds__(head_mma::kThreads, HEAD_GUMBEL1 ? 2 : 1)
    head_mma_keys(const __grid_constant__ CUtensorMap w_map, const __grid_constant__ CUtensorMap x_map,
                  const HeadMmaParams __grid_constant__ p) {
  using namespace head_mma;
  extern __shared__ Smem smem_raw[];
  Smem& s = smem_raw[0];
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int row0 = static_cast<int>(blockIdx.x) * p.box_rows;
  const int rows_here = min(p.box_rows, p.valid - row0);

  if (kCluster == 1 && rows_here <= 0) {
    pdl_wait();
    pdl_trigger();
    if (threadIdx.x == 0 && p.done != nullptr) {
      __threadfence();
      atomicAdd(p.done, 1u);
    }
    if constexpr (kGumbel1) {
      constexpr long long kFloor = -2139095041LL * 4294967296LL;
      for (int i = static_cast<int>(threadIdx.x); i < 2 * p.m; i += kThreads) {
        p.pre[static_cast<long long>(i) * gridDim.x + blockIdx.x] = kFloor;
      }
      if (threadIdx.x == 0 && p.seeds == nullptr) {
        const unsigned int seen = *reinterpret_cast<volatile unsigned int*>(p.epoch);
        epoch_ticket(p.epoch, p.pre + 2LL * p.m * gridDim.x, seen);
      }
    }
    return;
  }

  constexpr uint32_t kXBytes = kNChunks * kXTileBytes;
  constexpr uint16_t kMask = static_cast<uint16_t>((1u << kCluster) - 1u);
  const uint32_t crank = kCluster > 1 ? cluster_rank() : 0u;
  const int k_off = (static_cast<int>(blockIdx.x) / kCluster * 37) & (kKBlocks - 1);
  const auto kcol = [&](int kb) { return ((kb + k_off) & (kKBlocks - 1)) * kBlockK; };
  const auto load_x = [&](int st, int kb) {
#pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
      if constexpr (kCluster > 1) {
        const int r0 = static_cast<int>(crank) * kXRows;
        tma_2d_mc(to_shared(s.x[st] + c * kXTileBytes + r0 * kBlockK * 2), &x_map, kcol(kb), c * kNTile + r0,
                  &s.full[st], kMask);
      } else {
        tma_2d(to_shared(s.x[st] + c * kXTileBytes), &x_map, kcol(kb), c * kNTile, &s.full[st]);
      }
    }
  };
  constexpr int kPre = kStages < kKBlocks ? kStages : kKBlocks;
  const uint32_t stage_bytes = static_cast<uint32_t>(p.box_rows) * kBlockK * 2 + kXBytes;

  if (warp == 1) {
    tmem_alloc(to_shared(&s.taddr), kTmemAlloc);
    tmem_relinquish();
  } else if (warp == 0) {
    if (lane == 0) {
      prefetch_tensormap(&w_map);
      prefetch_tensormap(&x_map);
#pragma unroll
      for (int i = 0; i < kStages; ++i) {
        mbar_init(&s.full[i], 1);
        mbar_init(&s.empty[i], kCluster);
      }
      mbar_init(&s.mma_done, 1);
      fence_mbarrier_init();
      for (int kb = 0; kb < kPre; ++kb) {
        mbar_arrive_expect_tx(&s.full[kb], stage_bytes);
        tma_2d(to_shared(s.w[kb]), &w_map, kcol(kb), row0, &s.full[kb]);
      }
    }
  }
  tc_fence_before();
  if constexpr (kCluster > 1) {
    cluster_sync();
  } else {
    __syncthreads();
  }
  tc_fence_after();

  if (warp == 0) {
    if (elect_one()) {
      pdl_wait();
      pdl_trigger();
#pragma unroll
      for (int kb = 0; kb < kPre; ++kb) load_x(kb, kb);
      for (int kb = kPre; kb < kKBlocks; ++kb) {
        const int st = kb % kStages;
        const uint32_t use = static_cast<uint32_t>(kb / kStages);
        mbar_wait_parity(&s.empty[st], (use - 1u) & 1u);
        mbar_arrive_expect_tx(&s.full[st], stage_bytes);
        tma_2d(to_shared(s.w[st]), &w_map, kcol(kb), row0, &s.full[st]);
        load_x(st, kb);
      }
    } else {
      pdl_wait();
      pdl_trigger();
    }
  } else if (warp == 1) {
    pdl_wait();
    pdl_trigger();
    const uint32_t taddr = s.taddr;
    if (elect_one()) {
      constexpr uint32_t idesc = idesc_bf16(kRowsTile, kNTile);
      const uint64_t w0 = smem_desc(to_shared(s.w[0]));
      const uint64_t x0 = smem_desc(to_shared(s.x[0]));
      for (int kb = 0; kb < kKBlocks; ++kb) {
        const int st = kb % kStages;
        mbar_wait_parity(&s.full[st], static_cast<uint32_t>(kb / kStages) & 1u);
        tc_fence_after();
        const uint64_t wd = w0 + ((static_cast<uint32_t>(st) * kWTileBytes) >> 4);
#pragma unroll
        for (int c = 0; c < kNChunks; ++c) {
          const uint64_t xd = x0 + ((static_cast<uint32_t>(st) * kXBytes + c * kXTileBytes) >> 4);
#pragma unroll
          for (int k = 0; k < kBlockK / 16; ++k) {
            mma_bf16(taddr + c * kNTile, wd + ((k * 32) >> 4), xd + ((k * 32) >> 4), idesc, (kb | k) != 0);
          }
        }
        if constexpr (kCluster > 1) {
          mma_commit_mc(&s.empty[st], kMask);
        } else {
          mma_commit(&s.empty[st]);
        }
      }
      mma_commit(&s.mma_done);
    }
  } else if constexpr (kGumbel1) {
    pdl_wait();
    pdl_trigger();
    const int band = warp & 3;
    const int r = band * 32 + lane;
    const bool row_ok = r < rows_here;
    const uint32_t grow = static_cast<uint32_t>(row0 + r);
    const uint32_t gid = static_cast<uint32_t>(p.vocab_start) + grow;
    const int word = (gid & 3u) == 1u ? 2 : (gid & 3u) == 2u ? 1 : static_cast<int>(gid & 3u);
    const float inv_t = g_div_full(1.0f, g_max(1.0f, 1e-20f));
    const uint32_t acc_base = s.taddr + (static_cast<uint32_t>(band * 32) << 16);
    const uint32_t noise_base = acc_base + static_cast<uint32_t>(kTmemCols);
    const int k = lane & 3;
    const int qbase = lane & ~3;
    const unsigned int seen = p.seeds != nullptr ? 0u : *reinterpret_cast<volatile unsigned int*>(p.epoch);
    const unsigned long long ebase = kSeedBase + static_cast<unsigned long long>(seen) * 65536ull;
    for (int t0 = 0; t0 < p.m && t0 < kTokens; t0 += 16) {
      uint32_t nz[16];
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        const int tb = t0 + 4 * jj;
        float lgw[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#if HEAD_NOISE
        if (tb + k < p.m) {
          const uint64_t seed = p.seeds != nullptr ? static_cast<uint64_t>(p.seeds[tb + k])
                                                   : static_cast<uint64_t>(ebase + static_cast<unsigned long long>(tb + k));
          uint32_t c0 = gid >> 2, c1 = 0u, c2 = 0u, c3 = 0u;
          uint32_t k0 = static_cast<uint32_t>(seed & 0xFFFFFFFFull), k1 = static_cast<uint32_t>(seed >> 32);
#pragma unroll
          for (int i = 0; i < 10; ++i) {
            const uint32_t a0 = c0, a2 = c2;
            c0 = __umulhi(0xCD9E8D57u, a2) ^ c1 ^ k0;
            c2 = __umulhi(0xD2511F53u, a0) ^ c3 ^ k1;
            c1 = 0xCD9E8D57u * a2;
            c3 = 0xD2511F53u * a0;
            k0 += 0x9E3779B9u;
            k1 += 0xBB67AE85u;
          }
          lgw[0] = logf(-logf(g_uniform(c0)));
          lgw[1] = logf(-logf(g_uniform(c1)));
          lgw[2] = logf(-logf(g_uniform(c2)));
          lgw[3] = logf(-logf(g_uniform(c3)));
        }
#endif
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float v0 = __shfl_sync(0xffffffffu, lgw[0], qbase + i);
          const float v1 = __shfl_sync(0xffffffffu, lgw[1], qbase + i);
          const float v2 = __shfl_sync(0xffffffffu, lgw[2], qbase + i);
          const float v3 = __shfl_sync(0xffffffffu, lgw[3], qbase + i);
          nz[4 * jj + i] = __float_as_uint(word == 0 ? v0 : word == 1 ? v1 : word == 2 ? v2 : v3);
        }
      }
      tmem_st_x16(noise_base + static_cast<uint32_t>(t0), nz);
    }
    (void)k;
    tmem_wait_st();
    mbar_wait_parity(&s.mma_done, 0);
    tc_fence_after();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // the ring's last async-proxy use precedes
    // Rows fold in ascending order (a strict > keeps the lowest row); the 8 threads of a token combine by shuffles.
    __nv_bfloat16* const lbuf = reinterpret_cast<__nv_bfloat16*>(s.w[0]);
    uint32_t* const nbuf = reinterpret_cast<uint32_t*>(s.w[0] + 2 * 16 * 128 * 2);
    static_assert(kStages * kWTileBytes >= 2 * 16 * 128 * 2 + 2 * 16 * 128 * 4, "two stages fit the ring");
    const int et = static_cast<int>(threadIdx.x) - kEpiWarp0 * 32;
    const int jt = et >> 3, r0 = (et & 7) * 16;
    const int32_t none = static_cast<int32_t>(0x80000000u);
    int chunk = 0;
    for (int t0 = 0; t0 < p.m && t0 < kTokens; t0 += 16, ++chunk) {
      uint32_t acc[16], nz[16];
      tmem_ld_x16(acc_base + static_cast<uint32_t>(t0), acc);
      tmem_ld_x16(noise_base + static_cast<uint32_t>(t0), nz);
      tmem_wait_ld();
      __nv_bfloat16* const lst = lbuf + (chunk & 1) * (16 * 128);
      uint32_t* const nst = nbuf + (chunk & 1) * (16 * 128);
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        lst[j * 128 + r] = __float2bfloat16_rn(__uint_as_float(acc[j]));
        nst[j * 128 + r] = nz[j];
      }
      named_barrier(1, kEpiThreads);
      store_logits_chunk(lst, p.logits, p.logits_stride, p.m, t0, row0, rows_here, et);
      const uint4 l0 = reinterpret_cast<const uint4*>(lst + jt * 128 + r0)[0];
      const uint4 l1 = reinterpret_cast<const uint4*>(lst + jt * 128 + r0)[1];
      const uint32_t lw[8] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w};
      uint32_t nzr[16];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const uint4 n4 = reinterpret_cast<const uint4*>(nst + jt * 128 + r0)[q];
        nzr[4 * q] = n4.x;
        nzr[4 * q + 1] = n4.y;
        nzr[4 * q + 2] = n4.z;
        nzr[4 * q + 3] = n4.w;
      }
      // the thread's 16 rows ascending: fp32 compares without FTZ (g_gt); a strict > keeps the lowest row on ties
      int32_t bg = none, bp = none;
      uint32_t rg = 0xFFFFFFFFu, rp = 0xFFFFFFFFu;
      const int nrows = min(16, rows_here - r0);
      if (nrows > 0) {
        const float ninf = __int_as_float(0xFF800000);
        float v0 = __uint_as_float(lw[0] << 16);
        float sg = g_sub(g_mul(v0, inv_t), __uint_as_float(nzr[0]));
        sg = (sg == sg) ? sg : ninf;
        float sp = (v0 == v0) ? v0 : ninf;
        uint32_t ig = 0u, ip = 0u;
#pragma unroll
        for (int i = 1; i < 16; ++i) {
          const float v = __uint_as_float((i & 1) ? (lw[i >> 1] & 0xFFFF0000u) : (lw[i >> 1] << 16));
          const float g = g_sub(g_mul(v, inv_t), __uint_as_float(nzr[i]));
          const bool ok = i < nrows;
          if (ok && g_gt(g, sg)) {
            sg = g;
            ig = static_cast<uint32_t>(i);
          }
          if (ok && g_gt(v, sp)) {
            sp = v;
            ip = static_cast<uint32_t>(i);
          }
        }
        bg = g_order(sg);
        bp = g_order(sp);
        rg = static_cast<uint32_t>(r0) + ig;
        rp = static_cast<uint32_t>(r0) + ip;
      }
#pragma unroll
      for (int x = 1; x <= 4; x <<= 1) {
        const int32_t obg = __shfl_xor_sync(0xffffffffu, bg, x);
        const uint32_t org = __shfl_xor_sync(0xffffffffu, rg, x);
        const int32_t obp = __shfl_xor_sync(0xffffffffu, bp, x);
        const uint32_t orp = __shfl_xor_sync(0xffffffffu, rp, x);
        if (obg > bg || (obg == bg && org < rg)) {
          bg = obg;
          rg = org;
        }
        if (obp > bp || (obp == bp && orp < rp)) {
          bp = obp;
          rp = orp;
        }
      }
      const int t = t0 + jt;
      if ((et & 7) == 0 && t < p.m) {
        const auto key = [&](int32_t o, uint32_t rr) -> long long {
          return o == none ? -2139095041LL * 4294967296LL
                           : static_cast<long long>((static_cast<uint64_t>(static_cast<uint32_t>(o)) << 32) |
                                                    (0xFFFFFFFFull - (static_cast<uint64_t>(p.vocab_start) + row0 + rr)));
        };
        p.pre[static_cast<long long>(t) * gridDim.x + blockIdx.x] = key(bg, rg);
        p.pre[(static_cast<long long>(p.m) + t) * gridDim.x + blockIdx.x] = key(bp, rp);
      }
    }
    tc_fence_before();
    named_barrier(1, kEpiThreads);  // also: every epilogue thread read the epoch before the ticket below
    if (warp == kEpiWarp0) tmem_dealloc(s.taddr, kTmemAlloc);
    if (et == 0 && p.seeds == nullptr) epoch_ticket(p.epoch, p.pre + 2LL * p.m * gridDim.x, seen);
  } else {
    pdl_wait();
    pdl_trigger();
    const int band = warp & 3;
    const int r = band * 32 + lane;
    const bool row_ok = r < rows_here;
    const uint32_t grow = static_cast<uint32_t>(row0 + r);
    mbar_wait_parity(&s.mma_done, 0);
    tc_fence_after();
    if constexpr (!kKeys) asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // the ring is reused below
    const uint32_t base = s.taddr + (static_cast<uint32_t>(band * 32) << 16);
    for (int t0 = 0; t0 < p.m && t0 < kTokens; t0 += 16) {
      uint32_t acc[16];
      tmem_ld_x16(base + static_cast<uint32_t>(t0), acc);
      tmem_wait_ld();
      if constexpr (!kKeys) {
        __nv_bfloat16* const lst = reinterpret_cast<__nv_bfloat16*>(s.w[0]) + ((t0 >> 4) & 1) * (16 * 128);
#pragma unroll
        for (int j = 0; j < 16; ++j) lst[j * 128 + r] = __float2bfloat16_rn(__uint_as_float(acc[j]));
        named_barrier(1, kEpiThreads);
        store_logits_chunk(lst, p.logits, p.logits_stride, p.m, t0, row0, rows_here,
                           static_cast<int>(threadIdx.x) - kEpiWarp0 * 32);
        continue;
      }
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const int t = t0 + j;
        const __nv_bfloat16 lb = __float2bfloat16_rn(__uint_as_float(acc[j]));
        const float v = __bfloat162float(lb);
        uint32_t bits = __float_as_uint(v);
        bits = bits == 0x80000000u ? 0u : bits;
        int32_t order = static_cast<int32_t>((bits & 0x80000000u) ? (bits ^ 0x7FFFFFFFu) : bits);
        order = v != v ? p.nan_hi : order;
        order = row_ok ? order : static_cast<int32_t>(0x80000000u);
        if (p.logits != nullptr && row_ok && t < p.m) p.logits[t * p.logits_stride + grow] = lb;
        const int32_t top = __reduce_max_sync(0xffffffffu, order);
        const uint32_t low = __reduce_min_sync(0xffffffffu, row_ok && order == top ? grow : 0xFFFFFFFFu);
        if (lane == 0 && t < p.m) {
          s.top[0][band][t] = top;
          s.row[0][band][t] = low;
        }
      }
    }
    tc_fence_before();
    named_barrier(1, kEpiThreads);
    if (warp == kEpiWarp0) tmem_dealloc(s.taddr, kTmemAlloc);
    const int et = static_cast<int>(threadIdx.x) - kEpiWarp0 * 32;
    for (int t = et; kKeys && rows_here > 0 && t < p.m && t < kTokens; t += kEpiThreads) {
      int32_t top = s.top[0][0][t];
      uint32_t low = s.row[0][0][t];
#pragma unroll
      for (int b = 1; b < 4; ++b) {
        const int32_t tb = s.top[0][b][t];
        const uint32_t lb = s.row[0][b][t];
        if (tb > top || (tb == top && lb < low)) {
          top = tb;
          low = lb;
        }
      }
      const long long key = static_cast<long long>((static_cast<uint64_t>(static_cast<uint32_t>(top)) << 32) |
                                                   (0xFFFFFFFFull - (static_cast<uint64_t>(p.vocab_start) + low)));
      atomicMax(p.keys + t * p.keys_stride, key);
    }
    if (p.done != nullptr) {
      named_barrier(1, kEpiThreads);
      if (et == 0) {
        __threadfence();
        atomicAdd(p.done, 1u);
      }
    }
  }
  if constexpr (kCluster > 1) {
    __syncwarp();
    cluster_sync();
  }
}
