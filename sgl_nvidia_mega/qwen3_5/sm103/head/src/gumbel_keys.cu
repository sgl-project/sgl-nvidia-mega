// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#ifndef GK_PRE
#define GK_PRE 0  // 1: the build that reads the verify head's keys (p.pre, p.epoch)
#endif
#if !defined(GK_THREADS) || !defined(GK_FP32) || !defined(GK_GREEDY) || !defined(GK_DEBUG)
#error "gumbel_keys.cu: the launcher passes GK_THREADS, GK_FP32, GK_GREEDY, GK_DEBUG"
#endif

typedef unsigned int uint32_t;
typedef signed int int32_t;
typedef unsigned long long uint64_t;
typedef signed long long int64_t;

#include <cuda_bf16.h>

#define GK_DEVICE __forceinline__ __device__

namespace gumbel_keys {

inline constexpr int kThreads = GK_THREADS;
inline constexpr int kWarps = kThreads / 32;
inline constexpr int32_t kOrderNone = static_cast<int32_t>(0x80000000u);  // below ordered(-inf)
inline constexpr int64_t kKeyFloor = static_cast<int64_t>(-2139095041LL) * 4294967296LL;
static_assert(kThreads % 32 == 0 && kThreads >= 32 && kThreads <= 1024, "whole warps");

GK_DEVICE float mul_rn(float a, float b) {
  float d;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
GK_DEVICE float sub_rn(float a, float b) {
  float d;
  asm("sub.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
GK_DEVICE float add_rn(float a, float b) {
  float d;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
GK_DEVICE float max_f(float a, float b) {
  float d;
  asm("max.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
GK_DEVICE float min_f(float a, float b) {
  float d;
  asm("min.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
GK_DEVICE float div_full(float a, float b) {
  float d;
  asm("div.full.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
GK_DEVICE float i2f_rn(int32_t a) {
  float d;
  asm("cvt.rn.f32.s32 %0, %1;" : "=f"(d) : "r"(a));
  return d;
}

GK_DEVICE void philox(uint32_t c0, uint32_t k0, uint32_t k1, uint32_t (&out)[4]) {
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
  out[0] = c0;
  out[1] = c1;
  out[2] = c2;
  out[3] = c3;
}

// uint_to_uniform_float, then the fast kernel's clamp into (0, 1)
GK_DEVICE float uniform(uint32_t w) {
  int32_t x = static_cast<int32_t>(w);
  x = x < 0 ? static_cast<int32_t>(~w) : x;
  const float u = mul_rn(i2f_rn(x), 4.6566127342e-10f);
  return min_f(max_f(u, 1e-12f), 0.99999994f);
}

GK_DEVICE int32_t order_of(float v) {
  v = (v == v) ? v : __int_as_float(0xFF800000);
  v = add_rn(v, 0.0f);
  const int32_t b = __float_as_int(v);
  return b < 0 ? (b ^ 0x7FFFFFFF) : b;
}

GK_DEVICE void fold(int32_t& top, uint32_t& id, int32_t o, uint32_t g) {
  if (o > top || (o == top && g < id)) {
    top = o;
    id = g;
  }
}

GK_DEVICE int64_t pre_max(const long long* pre, int ctas, long long row, int rows, int kind, int64_t* s_max) {
  int64_t best = kKeyFloor;
  const long long* src = pre + (static_cast<long long>(kind) * rows + row) * ctas;
  for (int c = static_cast<int>(threadIdx.x); c < ctas; c += kThreads) best = max(best, static_cast<int64_t>(src[c]));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) best = max(best, static_cast<int64_t>(__shfl_xor_sync(0xffffffffu, best, o)));
  if ((threadIdx.x & 31) == 0) s_max[threadIdx.x >> 5] = best;
  __syncthreads();
  best = s_max[0];
#pragma unroll
  for (int w = 1; w < kWarps; ++w) best = max(best, s_max[w]);
  __syncthreads();
  return best;
}

GK_DEVICE int64_t cta_key(int32_t top, uint32_t id, int32_t (*s_top)[kWarps], uint32_t (*s_id)[kWarps], int slot) {
  const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
  const int32_t wt = __reduce_max_sync(0xffffffffu, top);
  const uint32_t wi = __reduce_min_sync(0xffffffffu, top == wt ? id : 0xFFFFFFFFu);
  if (lane == 0) {
    s_top[slot][warp] = wt;
    s_id[slot][warp] = wi;
  }
  __syncthreads();
  int32_t t = s_top[slot][0];
  uint32_t i = s_id[slot][0];
#pragma unroll
  for (int w = 1; w < kWarps; ++w) fold(t, i, s_top[slot][w], s_id[slot][w]);
  if (t == kOrderNone) return kKeyFloor;
  return static_cast<int64_t>((static_cast<uint64_t>(static_cast<uint32_t>(t)) << 32) |
                              (0xFFFFFFFFull - static_cast<uint64_t>(i)));
}

}  // namespace gumbel_keys

struct GumbelKeysParams {
  const void* logits;
  long long stride;            // elements between rows
  const float* temperatures;   // [rows]
  const long long* seeds;      // [rows]
  long long* keys;             // [rows, parts]
  long long* greedy;           // [rows, parts] (GK_GREEDY)
  float* values;               // [rows, stride] fp32 (GK_DEBUG)
  int vocab_start;             // global id of column 0 (a multiple of 4)
  int valid;                   // real columns of this shard
  int vocab_size;
  int span;                    // columns a part covers (a multiple of 4)
  const long long* pre;
  int pre_ctas;
  unsigned int* epoch;         // with pre: the head's {epoch, head ticket, keys ticket, 0}
};

extern "C" __global__ void __launch_bounds__(gumbel_keys::kThreads)
    gumbel_keys_fast(const GumbelKeysParams __grid_constant__ p) {
  using namespace gumbel_keys;
  __shared__ int32_t s_top[2][kWarps];
  __shared__ uint32_t s_id[2][kWarps];
  __shared__ int64_t s_max[kWarps];
  const long long row = blockIdx.x;
  const int part = static_cast<int>(blockIdx.y);
  const int parts = static_cast<int>(gridDim.y);
  const int lo = part * p.span;
  const int hi = min(lo + p.span, p.valid);
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const float temperature = p.temperatures[row];
  uint32_t gt;  // setp.gt.f32 without FTZ, as Triton compares (a denormal temperature samples)
  asm("{ .reg .pred q; setp.gt.f32 q, %1, 0f00000000; selp.u32 %0, 1, 0, q; }" : "=r"(gt) : "f"(temperature));
  const bool sampled = gt != 0u;
  const bool pre_mode = GK_PRE && p.pre != nullptr;
  const int rows_all = static_cast<int>(gridDim.x);
  const bool pre_ok =
      pre_mode &&
      *reinterpret_cast<const volatile long long*>(p.pre + 2LL * rows_all * p.pre_ctas) ==
          static_cast<long long>(*reinterpret_cast<volatile unsigned int*>(p.epoch) - 1u);
  const bool g_pre = pre_ok && (!sampled || __float_as_uint(temperature) == 0x3F800000u);
  const float inv_t = div_full(1.0f, max_f(temperature, 1e-20f));
  const uint64_t seed = static_cast<uint64_t>(p.seeds[row]);
  const uint32_t k0 = static_cast<uint32_t>(seed & 0xFFFFFFFFull);
  const uint32_t k1 = static_cast<uint32_t>(seed >> 32);
  int32_t g_top = kOrderNone, p_top = kOrderNone;
  uint32_t g_id = 0xFFFFFFFFu, p_id = 0xFFFFFFFFu;
  for (int c0 = lo + 4 * static_cast<int>(threadIdx.x); !g_pre && c0 < hi; c0 += 4 * kThreads) {
    float v[4];
#if GK_FP32
    const float4 q = *reinterpret_cast<const float4*>(static_cast<const float*>(p.logits) + row * p.stride + c0);
    v[0] = q.x;
    v[1] = q.y;
    v[2] = q.z;
    v[3] = q.w;
#else
    const uint2 q = *reinterpret_cast<const uint2*>(static_cast<const __nv_bfloat16*>(p.logits) + row * p.stride + c0);
    v[0] = __uint_as_float(q.x << 16);
    v[1] = __uint_as_float(q.x & 0xFFFF0000u);
    v[2] = __uint_as_float(q.y << 16);
    v[3] = __uint_as_float(q.y & 0xFFFF0000u);
#endif
    const int g0 = p.vocab_start + c0;
    uint32_t w[4] = {0u, 0u, 0u, 0u};
    if (sampled) philox(static_cast<uint32_t>(g0 >> 2), k0, k1, w);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int c = c0 + j;
      const int g = g0 + j;
      if (c >= hi || g >= p.vocab_size) continue;
      float s = v[j];
      if (sampled) {
        const float u = uniform(w[(j == 1) ? 2 : (j == 2) ? 1 : j]);
        s = sub_rn(mul_rn(s, inv_t), logf(-logf(u)));
      }
#if GK_DEBUG
      p.values[row * p.stride + c] = s;
#endif
      fold(g_top, g_id, order_of(s), static_cast<uint32_t>(g));
#if GK_GREEDY
      fold(p_top, p_id, order_of(v[j]), static_cast<uint32_t>(g));
#endif
    }
  }
  int64_t gk = cta_key(g_top, g_id, s_top, s_id, 0);
#if GK_GREEDY
  int64_t pk = cta_key(p_top, p_id, s_top, s_id, 1);
#endif
  if (pre_ok) {
    const int64_t pre_g = part == 0 ? pre_max(p.pre, p.pre_ctas, row, rows_all, sampled ? 0 : 1, s_max) : kKeyFloor;
    if (g_pre) gk = pre_g;
#if GK_GREEDY
    pk = part == 0 ? (sampled ? pre_max(p.pre, p.pre_ctas, row, rows_all, 1, s_max) : pre_g) : kKeyFloor;
#endif
  }
  if (threadIdx.x == 0) {
    p.keys[row * parts + part] = gk;
#if GK_GREEDY
    p.greedy[row * parts + part] = pk;
#endif
    if (pre_mode) {
      __threadfence();
      const unsigned int old = atomicAdd(p.epoch + 2, 1u);
      if (old == gridDim.x * gridDim.y - 1) {
        p.epoch[2] = 0u;
        __threadfence();
        atomicAdd(p.epoch, 1u);
      }
    }
  }
}
