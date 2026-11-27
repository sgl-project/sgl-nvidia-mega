// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#include <cuda_bf16.h>

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef long long int64_t;

#ifndef GDN_UNROLL
#define GDN_UNROLL 4
#endif
#ifndef GDN_MIN_BLOCKS
#define GDN_MIN_BLOCKS 1
#endif
#ifndef GDN_NK
#error "define GDN_NK GDN_NV GDN_D GDN_CHUNK GDN_HAS_BIAS GDN_SILU"
#endif

namespace gdn_prefill_front {

constexpr int kVec = 8;
constexpr int kUnroll = GDN_UNROLL;
constexpr int kThreads = 128;
constexpr int kNKD = GDN_NK * GDN_D;
constexpr int kNVD = GDN_NV * GDN_D;
constexpr int kConvDim = 2 * kNKD + kNVD;
constexpr int kQkBlocks = 2 * kNKD / (kThreads * kVec);
static_assert(GDN_D == 128, "16 lanes x 8 channels a head");
static_assert((2 * kNKD) % (kThreads * kVec) == 0, "q / k and v in whole CTAs");
static_assert(kConvDim % (kThreads * kVec) == 0, "whole CTAs");

struct Params {
    const __nv_bfloat16* x;      // [T, >= conv_dim] token-major, row stride sx
    const __nv_bfloat16* w;      // [conv_dim, 4] contiguous
    const __nv_bfloat16* bias;   // [conv_dim] or null
    const __nv_bfloat16* cs;
    const int* qsl;              // [num_seqs + 1]
    const int* cidx;             // [num_seqs]
    const uint8_t* hinit;        // [num_seqs] bool
    __nv_bfloat16* q;            // [T, NK, D]
    __nv_bfloat16* k;            // [T, NK, D]
    __nv_bfloat16* v;            // [T, NV, D]
    int64_t sx;
    int64_t cs_b;
    int64_t cs_c;
    int64_t cs_l;
    int num_tokens;
    int num_seqs;
    int pad_slot;
    float eps;
};

__device__ __forceinline__ float lo_f(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float hi_f(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

__device__ __forceinline__ uint32_t mul_bf16x2(uint32_t a, uint32_t b) {
    uint32_t d;
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}
__device__ __forceinline__ float add_rn(float a, float b) {
    float d;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}
__device__ __forceinline__ float mul_rn(float a, float b) {
    float d;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}
__device__ __forceinline__ float ex2_approx(float a) {
    float d;
    asm("ex2.approx.f32 %0, %1;" : "=f"(d) : "f"(a));
    return d;
}
__device__ __forceinline__ float ex2_approx_ftz(float a) {
    float d;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a));
    return d;
}
__device__ __forceinline__ float rcp_approx_ftz(float a) {
    float d;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a));
    return d;
}
__device__ __forceinline__ float div_full(float a, float b) {
    float d;
    asm("div.full.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}
__device__ __forceinline__ float sqrt_approx_ftz(float a) {
    float d;
    asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a));
    return d;
}
__device__ __forceinline__ uint32_t pack_rn(float lo, float hi) {
    uint32_t d;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
    return d;
}

__device__ __forceinline__ uint32_t word(const uint4& r, int i) {
    return i == 0 ? r.x : i == 1 ? r.y : i == 2 ? r.z : r.w;
}

__device__ __forceinline__ uint4 state_row(const __nv_bfloat16* sp, int64_t cs_c) {
    uint16_t h[kVec];
#pragma unroll
    for (int e = 0; e < kVec; ++e) h[e] = __bfloat16_as_ushort(sp[(int64_t)e * cs_c]);
    return make_uint4(h[0] | ((uint32_t)h[1] << 16), h[2] | ((uint32_t)h[3] << 16),
                      h[4] | ((uint32_t)h[5] << 16), h[6] | ((uint32_t)h[7] << 16));
}

// the raw input the conv of a row sees at position pos (< the row) of its sequence
__device__ __forceinline__ uint4 window_tap(const Params& p, int c0, int pos, int start, int slot,
                                            bool conv_ok, bool hinit) {
    if (conv_ok && pos >= start)
        return __ldg(reinterpret_cast<const uint4*>(p.x + (int64_t)pos * p.sx + c0));
    if (conv_ok && hinit)
        return state_row(p.cs + (int64_t)slot * p.cs_b + (int64_t)c0 * p.cs_c + (int64_t)(pos - start + 3) * p.cs_l,
                         p.cs_c);
    return make_uint4(0, 0, 0, 0);
}

template <bool QK>
__device__ __forceinline__ void body(const Params& p) {
    const int c0 = (blockIdx.y * kThreads + threadIdx.x) * kVec;
    __nv_bfloat16* out;
    int64_t out_row;
    if (QK) {
        out = c0 < kNKD ? p.q + c0 : p.k + (c0 - kNKD);
        out_row = kNKD;
    } else {
        out = p.v + (c0 - 2 * kNKD);
        out_row = kNVD;
    }
    uint32_t wt[4][4];
    {
        const uint4* wp = reinterpret_cast<const uint4*>(p.w + (int64_t)c0 * 4);
        uint16_t h[32];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint4 r = __ldg(wp + i);
            const uint32_t ws[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                h[8 * i + 2 * j] = (uint16_t)(ws[j] & 0xffffu);
                h[8 * i + 2 * j + 1] = (uint16_t)(ws[j] >> 16);
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i)
                wt[j][i] = h[(2 * i) * 4 + j] | ((uint32_t)h[(2 * i + 1) * 4 + j] << 16);
    }
    float bv[kVec];
#pragma unroll
    for (int e = 0; e < kVec; ++e) bv[e] = GDN_HAS_BIAS ? __bfloat162float(p.bias[c0 + e]) : 0.0f;

    asm volatile("griddepcontrol.wait;" ::: "memory");

    const int t0 = blockIdx.x * GDN_CHUNK;
    if (t0 >= p.num_tokens) return;
    const int t_end = min(t0 + GDN_CHUNK, p.num_tokens);

    int lo = 0, hi = p.num_seqs;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (__ldg(p.qsl + mid) <= t0) lo = mid; else hi = mid - 1;
    }
    int seq = lo;
    bool in_seq = seq < p.num_seqs;
    int start = __ldg(p.qsl + (in_seq ? seq : p.num_seqs - 1));
    int next_start = in_seq ? __ldg(p.qsl + seq + 1) : 0x7fffffff;
    int slot = in_seq ? __ldg(p.cidx + seq) : p.pad_slot;
    bool hinit = in_seq && __ldg(p.hinit + seq) != 0;
    bool conv_ok = in_seq && slot != p.pad_slot;

    const uint4 wa0 = window_tap(p, c0, t0 - 3, start, slot, conv_ok, hinit);
    const uint4 wb0 = window_tap(p, c0, t0 - 2, start, slot, conv_ok, hinit);
    const uint4 wc0 = window_tap(p, c0, t0 - 1, start, slot, conv_ok, hinit);
    uint4 wa = wa0, wb = wb0, wc = wc0;

    const __nv_bfloat16* xrow = p.x + (int64_t)t0 * p.sx + c0;
    __nv_bfloat16* orow = out + (int64_t)t0 * out_row;
    uint4 cur[kUnroll], nxt[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u)
        cur[u] = t0 + u < t_end ? __ldg(reinterpret_cast<const uint4*>(xrow + u * p.sx)) : make_uint4(0, 0, 0, 0);
    for (int tb = t0; tb < t_end; tb += kUnroll) {
        const __nv_bfloat16* nrow = xrow + kUnroll * p.sx;
#pragma unroll
        for (int u = 0; u < kUnroll; ++u)  // the next block's rows in flight while this block computes
            nxt[u] = tb + kUnroll + u < t_end ? __ldg(reinterpret_cast<const uint4*>(nrow + u * p.sx))
                                              : make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            const int t = tb + u;
            if (t < t_end) {
                if (t == next_start) {
                    // a new sequence (or the padding past the last one) starts at t
                    do { ++seq; } while (seq < p.num_seqs && __ldg(p.qsl + seq + 1) <= t);
                    in_seq = seq < p.num_seqs;
                    start = t;
                    next_start = in_seq ? __ldg(p.qsl + seq + 1) : 0x7fffffff;
                    slot = in_seq ? __ldg(p.cidx + seq) : p.pad_slot;
                    hinit = in_seq && __ldg(p.hinit + seq) != 0;
                    conv_ok = in_seq && slot != p.pad_slot;
                    if (conv_ok && hinit) {
                        const __nv_bfloat16* sp = p.cs + (int64_t)slot * p.cs_b + (int64_t)c0 * p.cs_c;
                        wa = state_row(sp, p.cs_c);
                        wb = state_row(sp + p.cs_l, p.cs_c);
                        wc = state_row(sp + 2 * p.cs_l, p.cs_c);
                    } else {
                        wa = make_uint4(0, 0, 0, 0);
                        wb = wa;
                        wc = wa;
                    }
                }
                const uint4 xr = cur[u];
                uint4 o;
                if (conv_ok) {
                    uint32_t ow[4];
                    float acc[kVec];
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const uint32_t p0 = mul_bf16x2(word(wa, i), wt[0][i]);
                        const uint32_t p1 = mul_bf16x2(word(wb, i), wt[1][i]);
                        const uint32_t p2 = mul_bf16x2(word(wc, i), wt[2][i]);
                        const uint32_t p3 = mul_bf16x2(word(xr, i), wt[3][i]);
                        float a_lo = add_rn(bv[2 * i], lo_f(p0));
                        float a_hi = add_rn(bv[2 * i + 1], hi_f(p0));
                        a_lo = add_rn(a_lo, lo_f(p1));
                        a_hi = add_rn(a_hi, hi_f(p1));
                        a_lo = add_rn(a_lo, lo_f(p2));
                        a_hi = add_rn(a_hi, hi_f(p2));
                        a_lo = add_rn(a_lo, lo_f(p3));
                        a_hi = add_rn(a_hi, hi_f(p3));
                        acc[2 * i] = a_lo;
                        acc[2 * i + 1] = a_hi;
                    }
                    if (GDN_SILU) {
                        bool fast = true;
#pragma unroll
                        for (int e = 0; e < kVec; ++e) fast &= fabsf(acc[e]) < 87.0f;
                        const float m = __uint_as_float(0xBFB8AA3Bu);
                        if (__all_sync(0xffffffffu, fast)) {
#pragma unroll
                            for (int e = 0; e < kVec; ++e)
                                acc[e] = mul_rn(acc[e], rcp_approx_ftz(add_rn(ex2_approx_ftz(mul_rn(acc[e], m)), 1.0f)));
                        } else {
#pragma unroll
                            for (int e = 0; e < kVec; ++e)
                                acc[e] = div_full(acc[e], add_rn(ex2_approx(mul_rn(acc[e], m)), 1.0f));
                        }
                    }
#pragma unroll
                    for (int i = 0; i < 4; ++i) ow[i] = pack_rn(acc[2 * i], acc[2 * i + 1]);
                    o = make_uint4(ow[0], ow[1], ow[2], ow[3]);
                } else {
                    o = xr;  // raw input
                }
                if (QK) {
                    float f[kVec];
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        f[2 * i] = lo_f(word(o, i));
                        f[2 * i + 1] = hi_f(word(o, i));
                    }
                    float s[kVec];
#pragma unroll
                    for (int e = 0; e < kVec; ++e) s[e] = mul_rn(f[e], f[e]);
                    const float c_lo = add_rn(add_rn(s[0], s[2]), add_rn(s[4], s[6]));
                    const float c_hi = add_rn(add_rn(s[1], s[3]), add_rn(s[5], s[7]));
                    float r = add_rn(c_lo, c_hi);
#pragma unroll
                    for (int m = 8; m >= 1; m >>= 1) r = add_rn(r, __shfl_xor_sync(0xffffffffu, r, m));
                    const float den = sqrt_approx_ftz(add_rn(r, p.eps));
                    if (den <= 8.50705917302346158658e+37f && den >= 1.17549435082228750797e-38f) {
                        const float inv = rcp_approx_ftz(den);
#pragma unroll
                        for (int e = 0; e < kVec; ++e) f[e] = mul_rn(f[e], inv);
                    } else {
#pragma unroll
                        for (int e = 0; e < kVec; ++e) f[e] = div_full(f[e], den);
                    }
                    o = make_uint4(pack_rn(f[0], f[1]), pack_rn(f[2], f[3]), pack_rn(f[4], f[5]),
                                   pack_rn(f[6], f[7]));
                }
                *reinterpret_cast<uint4*>(orow + u * out_row) = o;
                wa = wb;
                wb = wc;
                wc = xr;
            }
        }
        xrow = nrow;
        orow += kUnroll * out_row;
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) cur[u] = nxt[u];
    }
}

}  // namespace gdn_prefill_front

extern "C" __global__ void __launch_bounds__(gdn_prefill_front::kThreads, GDN_MIN_BLOCKS)
    q35_gdn_prefill_front(const gdn_prefill_front::Params p) {
    // A CTA covers kThreads * kVec channels: q / k heads only, or v heads only.
    if (blockIdx.y < gdn_prefill_front::kQkBlocks)
        gdn_prefill_front::body<true>(p);
    else
        gdn_prefill_front::body<false>(p);
}
