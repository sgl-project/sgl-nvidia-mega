// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// MoE front roles of one GPU's owned rows in ONE launch (development form; not wired into the server).
//
// The launch runs one CTA per SM. Every CTA owns one 512-column K chunk of the hidden width and a contiguous range of
// that chunk's 8-column pooler tiles and 8-expert router tiles (ranges split evenly over the CTAs of each chunk). Roles,
// phase-multiplexed per CTA by warp:
//   warp 0     control: CTA 0 publishes the go word (a tag in device memory); every CTA waits for it.
//   warp 1     pooler producer: bulk copies of the CTA's pooler weight tiles into a shared-memory ring (weights are static,
//              so stages may be issued before the go word) and of the CTA's pooler input chunk after it.
//   warps 2-5  pooler MMA: for each 8-column tile and 16-row tile, one bf16 mma.sync m16n8k16 accumulator chain over the
//              chunk's 512 columns from +0 in ascending k (the chunk-partial form); the fp32 chunk partials go to
//              part[chunk][row][column]; after the CTA's last tile, one release of the CTA's partials word. Then every
//              CTA sums, for its own slice of pooled columns and every owned row, the 16 chunk partials in chunk order
//              (after acquiring the words of the CTAs that produced them), rounds once to bf16 into the pooled rows and
//              releases its pooled-slice word.
//   warps 6-9  router: the TF32 router contract of wfc_k1s.cuh (per 512-column chunk one mma.sync m16n8k8 TF32 chain from
//              +0 in ascending k, operands rounded with cvt.rna) for the CTA's expert tiles; partials published as tagged
//              64-bit words (launch tag in the high half). The CTA that owns routed row m then sums that row's 16 x 256
//              partials in chunk order and runs the routing arithmetic (muse_vr_front::route_row). The balance CTA instead
//              computes the per-step expert assignment over the replica pool (muse_vr::bal) once every owned row is routed.
//   warps 2-9  on the CTA that owns quantized row m, after their own work and every CTA's pooled-slice word: the norm +
//              NVFP4 quant group of muse_vr_front::quant_row on that pooled bf16 row.
//
// Ordering (PTX ISA memory consistency model):
//   - go: CTA 0 publishes it with st.release.gpu after every CTA's arrival (atom.add.release.gpu); waiters ld.acquire.gpu.
//   - router partials and routing / quant ready words carry the tag in the same 64-bit word as the data (or are released
//     after the data with st.release.gpu); readers retry until every word carries this launch's tag.
//   - pooler partials and pooled slices (wide data): plain stores by the four pooler warps -> bar.sync over exactly those
//     warps -> one st.release.gpu of the CTA's tagged word; each consumer thread acquires one producer word, then bar.sync
//     over the consuming group, then the group reads. No counters.
//   - bulk copies: completion through mbarrier complete_tx; before a ring slot is refilled the producer waits on the slot's
//     empty barrier (released by every MMA warp after its last read of the slot) and issues fence.proxy.async.shared::cta;
//     before the input-chunk copy (data written before the go release) it issues fence.proxy.async.global.
//   - every wait is bounded (iteration counts); a timeout sets a sticky error bit in ctl[3].
// Co-residency: CTAs wait on each other (go word, partials, ready words), so the launch must have every CTA resident:
// the host launches it cooperatively.

#pragma once

#include "wfc_k1s.cuh"
#include "wfc_front_roles.cuh"
#include "bal_assign.cuh"
#include "wfc_c4_norm.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

// Timing builds include a private stamp header (not part of this package); production builds compile the stamps out.
#ifdef MUSE_VR_WFC_C4F_STAMPS
#include "wfc_c4f_stamps.cuh"
#else
#define C4F_STAMP(lv) ((void)0)
#define C4F_PREGO_WAIT(t_start, ns) ((void)0)
#define C4F_NOW(v) ((v) = 0ull)
#endif

namespace sglang {
namespace muse_vr_c4f {

namespace k1 = ::sglang::muse_vr_k1s;
namespace fr = ::sglang::muse_vr_front;
namespace bl = ::muse_vr::bal;

inline constexpr int kK = 8192;
inline constexpr int kKC = 512;
inline constexpr int kChunks = 16;
inline constexpr int kHid = 4096;                    // pooled width
inline constexpr int kNE = 256;                      // routed experts
inline constexpr int kRowsMax = 64;                  // owned rows per launch (4 row tiles)
inline constexpr int kPTiles = kHid / 8;             // 8-column pooler tiles per chunk
inline constexpr int kPTileBytes = 32 * 32 * 8;      // 32 k16 steps x 32 lanes x 2 bf16x2 registers
inline constexpr int kStageTilesMax = 8;            // pooler tiles per ring stage: 4 or 8 (runtime)
inline constexpr int kNSMax = 3;
inline constexpr int kRTiles = kNE / 8;              // 8-expert router tiles per chunk
inline constexpr int kRTileBytes = 64 * 32 * 8;      // 64 k8 steps x 32 lanes x 2 TF32 registers
inline constexpr int kRTilesMax = 3;
inline constexpr int kXpPad = 520;                   // bf16 per pooler-input row in shared memory (conflict-free ldmatrix)
inline constexpr int kXpRowBytes = kXpPad * 2;
inline constexpr int kWarps = 10;
inline constexpr int kThreads = kWarps * 32;
inline constexpr int kBalRowsMax = 256;              // rows of the whole step the assignment covers
inline constexpr int kMinGrid = kChunks * ((kRTiles + kRTilesMax - 1) / kRTilesMax);  // <= kRTilesMax router tiles per CTA
inline constexpr int kMaxGrid = 256;
// role mask
inline constexpr int kRolePool = 1, kRoleRouter = 2, kRoleBal = 4, kRoleNoPrestage = 8;
// named barrier ids (disjoint per group; 0 = __syncthreads)
inline constexpr int kBarPool = 1, kBarRoute = 2, kBarBal = 3, kBarQuantIn = 4, kBarQuant = 5;
inline constexpr int kBarSlot0 = 6;                 // ring slot s released to the producer: ids 6 .. 6 + kNSMax - 1
inline constexpr int kSlotThreads = 5 * 32;          // the four pooler MMA warps + the producer warp
// error bits (ctl[3])
inline constexpr unsigned long long kErrArrive = 1ull, kErrGo = 2ull, kErrRing = 4ull, kErrBoard = 8ull, kErrReady = 16ull,
                                    kErrRouted = 32ull, kErrStage = 64ull, kErrBal = 128ull, kErrPart = 256ull;
// stamp slots (timing builds): [kSGo] go publish, [kSAsg] assignment published, [kSBal0] assignment start, [kSBalCnt]
// counted, [kSBalAsg] assigned, [kSBalVid] virtual ids stored,
// [kSIds + m] row m routed, [kSTok + m] row m quantized, [kSPend + b] CTA b partials released, [kSGoSeen + b] CTA b saw go,
// [kSRtrEnd + b] CTA b router chains published, [kSSum + b] CTA b pooled slice released.
inline constexpr int kSGo = 0, kSAsg = 1, kSBal0 = 2, kSBalCnt = 3, kSBalAsg = 4, kSBalVid = 5, kSIds = 8, kSTok = kSIds + kRowsMax, kSPend = kSTok + kRowsMax,
                     kSGoSeen = kSPend + kMaxGrid, kSRtrEnd = kSGoSeen + kMaxGrid, kSSum = kSRtrEnd + kMaxGrid,
                     kNumStamps = kSSum + kMaxGrid;
inline constexpr unsigned kSpin = 1u << 24;
inline constexpr unsigned kPoll = 1u << 22;

struct Params {
  // inputs (static weights in the launch's layouts; the activation images written before the go word)
  const float4* xr_img;          // router input, A-fragment order [16][mt][64][32] float4 (raw fp32)
  const uint2* wr_frag;          // router weights, B-fragment order [16][32 tiles][64][32] (TF32-rounded)
  const float* bias;             // [256]
  const uint2* wp_frag;          // pooler weights, B-fragment order [16][512 tiles][32][32] (bf16 pairs)
  const uint16_t* xp_img;        // pooler input [16][Rpad][520] bf16
  const void* qweight;           // [4096] bf16 norm weight (kHasW)
  // router outputs
  unsigned long long* board;     // [16][R][256] tagged chunk partials
  float* topk_w;                 // [R][8]
  int32_t* topk_ids;             // [R][8]
  int32_t* packed;               // [R][8]
  unsigned long long* rwords;    // [R][8] tagged packed ids
  // pooler outputs
  float* part;                   // [16][R][4096] fp32 chunk partials
  unsigned long long* pflag;     // [G] tag: CTA b's partials are released
  unsigned long long* sflag;     // [G] tag: CTA b's pooled slice is released
  uint8_t* fp4;                  // [R][2048]
  uint8_t* sf;                   // [R][256]
  float* pts;                    // [R]
  unsigned long long* qwords;    // [R] tagged row words
  uint16_t* pooled;              // [R][4096] bf16 pooled rows (the CTAs' slices)
  // balance
  const int32_t* bal_ids;        // [Mtot][8] packed ids of the whole step (rows of the other ranks land before the go word)
  const uint32_t* bal_table;     // [256] placement table
  int32_t* bal_vids;             // [Mtot][8] packed virtual ids (rank * pool + slot)
  uint8_t* bal_asg;              // [256] holder rank per expert (0xff: not routed)
  unsigned long long* bal_word;  // [1] tag << 32 | assignment digest
  // owner-row boundary (front-of-MoE mode; nbranch == nullptr: off, the input images are written before the go word)
  const uint16_t* nbranch;       // [R][8192] bf16 owned rows' attention output (the collected rows)
  const float* nstream;          // [R][8192] fp32 owned rows' residual stream
  float* nnew_stream;            // [R][8192] fp32 new residual stream
  const uint16_t* npre_w;        // [8192] bf16 pre-norm effective weight (nullable)
  const float* nalpha;           // [1] or [8192] fp32 residual gates
  const float* nbeta;
  float npost_eps, npre_eps;
  int nchannel;                  // 1: alpha / beta per channel
  unsigned long long* nwords;    // [R] tag << 32 | row: row's images written
  float* xr_mut;                 // = xr_img (written by the boundary role in this mode)
  uint16_t* xp_mut;              // = xp_img
  // control
  unsigned long long* ctl;       // [0] go, [1] arrivals, [2] epoch, [3] sticky error bits
  unsigned long long* stamps;    // [kNumStamps] (timing builds)
  int R, Rpad, G, NS, stile, roles, prego_ns, tmo_scale;
  int Mtot, bal_pool, bal_cta;
  uint32_t bal_pass_rows, bal_pass_inv;
  float mult, route_scale, div_eps, qeps, gsi;
};

// Shared-memory layout (bytes from a 1024-aligned base).
struct Layout {
  int ring, xp, wr, bal, ids, lg, qsm, bars, total;
};
__host__ __device__ inline int align_up(int x, int a) { return (x + a - 1) / a * a; }
__host__ __device__ inline Layout layout(int Rpad, int NS, int stile, int Mtot) {
  Layout L{};
  L.ring = 0;
  L.xp = align_up(L.ring + NS * stile * kPTileBytes, 128);
  L.wr = align_up(L.xp + Rpad * kXpRowBytes, 128);
  L.bal = align_up(L.wr + kRTilesMax * kRTileBytes, 128);
  L.ids = align_up(L.bal + static_cast<int>(sizeof(bl::Smem)), 128);
  L.lg = align_up(L.ids + Mtot * 8 * 4, 128);
  L.qsm = align_up(L.lg + kNE * 4, 128);
  L.bars = align_up(L.qsm + 32 * 4, 128);
  L.total = L.bars + (2 * kNSMax + 2) * 8;
  return L;
}
inline constexpr int kSmemSlack = 1024;

__device__ __forceinline__ void fail(const Params& p, unsigned long long bit) { atomicOr(p.ctl + 3, bit); }

__device__ __forceinline__ bool wait_mbar(const Params& p, uint32_t bar, unsigned parity, unsigned long long bit) {
  unsigned it = 0;
  while (!k1::mbar_try(bar, parity)) {
    if (++it > kSpin * static_cast<unsigned>(p.tmo_scale)) {
      fail(p, bit);
      return false;
    }
  }
  return true;
}

__device__ __forceinline__ bool wait_tag(const Params& p, const unsigned long long* w, uint32_t tag, unsigned long long bit,
                                         unsigned long long* val = nullptr) {
  unsigned it = 0;
  unsigned long long v;
  while (static_cast<uint32_t>((v = k1::ld_acq(w)) >> 32) != tag) {
    if (++it > kPoll * static_cast<unsigned>(p.tmo_scale)) {
      fail(p, bit);
      return false;
    }
    __nanosleep(32);
  }
  if (val != nullptr) *val = v;
  return true;
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float4 ld_f4(const float4* p) {
  float4 v;
  asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
  return v;
}

__device__ __forceinline__ void st_f2(float* p, float a, float b) {
  asm volatile("st.global.v2.f32 [%0], {%1, %2};" ::"l"(p), "f"(a), "f"(b) : "memory");
}

__device__ __forceinline__ void red_add_release(unsigned long long* p, unsigned long long v) {
  asm volatile("red.release.gpu.global.add.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

// Static per-CTA work split.
struct Plan {
  int c, pt0, npt, rt0, nrt, nstg, route_m, quant_m;
};
__device__ __forceinline__ Plan plan_of(int b, int G, int R, int stile) {
  Plan q{};
  const int base = G / kChunks, extra = G % kChunks;
  int s0 = 0, c = 0;
  for (c = 0; c < kChunks; ++c) {
    const int n = base + (c < extra ? 1 : 0);
    if (b < s0 + n) break;
    s0 += n;
  }
  const int n = base + (c < extra ? 1 : 0), j = b - s0;
  q.c = c;
  q.pt0 = kPTiles * j / n;
  q.npt = kPTiles * (j + 1) / n - q.pt0;
  q.rt0 = kRTiles * j / n;
  q.nrt = kRTiles * (j + 1) / n - q.rt0;
  q.nstg = (q.npt + stile - 1) / stile;
  q.route_m = -1;
  q.quant_m = -1;
  for (int m = 0; m < R; ++m) {
    if ((m * G) / R == b) q.route_m = m;
    if (((2 * m + 1) * G) / (2 * R) == b) q.quant_m = m;
  }
  return q;
}

// ---------------------------------------------------------------------------------------------------------------------
// Pooler MMA warp: for every stage of the ring, this warp's (row tile, tile subset) chains over the 32 k16 steps of the
// chunk, then the fp32 partials of its tiles. kMT row tiles; warps per row tile = 4 / kMT (kMT = 3: warp 3 idle). The A and
// B fragments of step k + 2 are loaded while the MMAs of step k issue (three register buffers, steps fully unrolled), so the
// shared-memory latency overlaps the MMA chains.
template <int kMT>
__device__ __forceinline__ void pool_warp(const Params& p, const Plan& q, unsigned char* smem, const Layout& L,
                                          unsigned long long* bars, int pw, int lane) {
  constexpr int kWpm = 4 / kMT;               // warps per row tile
  constexpr int kNT = kStageTilesMax / kWpm;   // tiles per warp per stage (at most)
  const int ntw = p.stile / kWpm;              // tiles per warp per stage (this launch)
  const int stage_bytes = p.stile * kPTileBytes;
  const int mi = pw % kMT, part = pw / kMT;
  const bool active = part < kWpm;
  const int t_lo = part * ntw;
  const uint32_t ring = k1::smem_u32(smem + L.ring);
  const uint32_t a_lane = k1::smem_u32(smem + L.xp) + static_cast<uint32_t>((mi * 16 + (lane & 15)) * kXpRowBytes +
                                                                           (lane >> 4) * 16);
  const int gq = lane >> 2, t4 = lane & 3;
  const int r0 = mi * 16 + gq, r1 = r0 + 8;
  // on a failed wait the warp still walks every stage (bounded waits, slot releases) so the producer never blocks
  const bool have_x = wait_mbar(p, k1::smem_u32(bars + 2 * kNSMax), 0, kErrStage);
  for (int s = 0; s < q.nstg; ++s) {
    const int slot = s % p.NS;
    const unsigned parity = static_cast<unsigned>(s / p.NS) & 1u;
    const bool have_w = wait_mbar(p, k1::smem_u32(bars + slot), parity, kErrRing);
    const int tiles_here = min(p.stile, q.npt - s * p.stile);
    const int nt = active ? max(0, min(ntw, tiles_here - t_lo)) : 0;  // tiles this warp computes in this stage
    if (nt > 0 && have_x && have_w) {
      float acc[kNT][4];
#pragma unroll
      for (int t = 0; t < kNT; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
      const uint32_t b_lane = ring + static_cast<uint32_t>(slot * stage_bytes + t_lo * kPTileBytes + lane * 8);
      uint32_t fa[3][4];
      uint2 fb[3][kNT];
      auto load = [&](int buf, int ks) {
        k1::ldsm_x4(fa[buf], a_lane + static_cast<uint32_t>(ks * 32));
#pragma unroll
        for (int t = 0; t < kNT; ++t)
          if (t < nt) fb[buf][t] = k1::lds64(b_lane + static_cast<uint32_t>(t * kPTileBytes + ks * 256));
      };
      load(0, 0);
      load(1, 1);
#pragma unroll
      for (int ks = 0; ks < 32; ++ks) {
        if (ks + 2 < 32) load((ks + 2) % 3, ks + 2);
        const int cur = ks % 3;
#pragma unroll
        for (int t = 0; t < kNT; ++t)
          if (t < nt) mma16816(acc[t], fa[cur], fb[cur][t].x, fb[cur][t].y);
      }
#pragma unroll
      for (int t = 0; t < kNT; ++t) {
        if (t < nt) {
          const int col = (q.pt0 + s * p.stile + t_lo + t) * 8 + t4 * 2;
          if (r0 < p.R) st_f2(p.part + (static_cast<long long>(q.c) * p.R + r0) * kHid + col, acc[t][0], acc[t][1]);
          if (r1 < p.R) st_f2(p.part + (static_cast<long long>(q.c) * p.R + r1) * kHid + col, acc[t][2], acc[t][3]);
        }
      }
    }
    // every shared-memory read of the slot by this warp precedes the arrive (the MMAs consumed the registers); the slot's
    // named barrier is used only when the slot is refilled later (arrivals and the producer's waits match one to one)
    __syncwarp();
    if (s + p.NS < q.nstg) k1::bar_arrive_named(kBarSlot0 + slot, kSlotThreads);
  }
}

// CTA index that produced chunk c's tile t (the static split of plan_of).
__device__ __forceinline__ int producer_of(int c, int t, int G) {
  const int base = G / kChunks, extra = G % kChunks;
  const int n = base + (c < extra ? 1 : 0);
  const int s0 = c * base + min(c, extra);
  int j = (t * n) / kPTiles;
  while (j + 1 < n && kPTiles * (j + 1) / n <= t) ++j;
  while (j > 0 && kPTiles * j / n > t) --j;
  return s0 + j;
}

// Pooled slice of CTA b: columns [b * 4096 / G, (b + 1) * 4096 / G) of every owned row. The four pooler warps (128 threads)
// acquire the partials words of the CTAs that produced those columns (at most 2 per chunk), then sum the 16 chunk partials
// of each element in chunk order (fp32, round to nearest), round once to bf16 and store the pooled values; then one
// release of the CTA's pooled-slice word.
__device__ __forceinline__ void slice_sum(const Params& p, int b, int tp, uint32_t tag) {
  const int c0 = b * kHid / p.G, c1 = (b + 1) * kHid / p.G;
  // producers: per chunk the CTAs covering tiles c0 / 8 .. (c1 - 1) / 8 (consecutive CTA ids)
  if (tp < kChunks) {
    const int lo = producer_of(tp, c0 / 8, p.G), hi = producer_of(tp, (c1 - 1) / 8, p.G);
    for (int pb = lo; pb <= hi; ++pb) wait_tag(p, p.pflag + pb, tag, kErrPart);
  }
  k1::bar_named(kBarPool, 128);  // every producer word acquired by one thread of the group before any read below
  const int w = c1 - c0, n = p.R * w;
  for (int e = tp; e < n; e += 128) {
    const int r = e / w, col = c0 + e % w;
    const float* src = p.part + static_cast<long long>(r) * kHid + col;
    float a = src[0];
#pragma unroll
    for (int cc = 1; cc < kChunks; ++cc) a = __fadd_rn(a, src[static_cast<long long>(cc) * p.R * kHid]);
    p.pooled[static_cast<long long>(r) * kHid + col] = __bfloat16_as_ushort(__float2bfloat16_rn(a));
  }
  k1::bar_named(kBarPool, 128);  // every pooled store of the slice precedes the release below
  if (tp == 0) {
    k1::st_rel(p.sflag + b, static_cast<unsigned long long>(tag) << 32);
    C4F_STAMP(p.stamps[kSSum + b]);
  }
}

// Router chain warp: this warp's row tile x up to kRTilesMax expert tiles of the CTA's chunk; A fragments straight from the
// fragment-order image (prefetched 8 k8 steps ahead), B fragments from the staged weights; tagged partials published.
template <int kMT>
__device__ __forceinline__ void router_warp(const Params& p, const Plan& q, unsigned char* smem, const Layout& L,
                                            unsigned long long* bars, int rw, int lane, uint32_t tag) {
  constexpr int kWpm = 4 / kMT;
  const int mi = rw % kMT, part = rw / kMT;
  if (part >= kWpm || q.nrt == 0) return;
  if (!wait_mbar(p, k1::smem_u32(bars + 2 * kNSMax + 1), 0, kErrStage)) return;
  if (p.nbranch != nullptr) {
    // front-of-MoE mode: the 16 rows of this row tile are written by the boundary role (one acquire per row word, then
    // the warp barrier before any lane reads the image)
    if (lane < 16 && mi * 16 + lane < p.R) wait_tag(p, p.nwords + mi * 16 + lane, tag, kErrStage);
    __syncwarp();
  }
  float acc[kRTilesMax][4];
#pragma unroll
  for (int u = 0; u < kRTilesMax; ++u) acc[u][0] = acc[u][1] = acc[u][2] = acc[u][3] = 0.f;
  const float4* ax = p.xr_img + (static_cast<long long>(q.c * kMT + mi) * 64) * 32 + lane;
  const uint32_t w_lane = k1::smem_u32(smem + L.wr) + static_cast<uint32_t>(lane * 8);
  float4 abuf[8];
#pragma unroll
  for (int u = 0; u < 8; ++u) abuf[u] = ld_f4(ax + u * 32);
#pragma unroll 1
  for (int ks0 = 0; ks0 < 64; ks0 += 8) {
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int ks = ks0 + u;
      const float4 av = abuf[u];
      if (ks + 8 < 64) abuf[u] = ld_f4(ax + (ks + 8) * 32);
      const uint32_t a[4] = {k1::to_tf32(__float_as_uint(av.x)), k1::to_tf32(__float_as_uint(av.y)),
                             k1::to_tf32(__float_as_uint(av.z)), k1::to_tf32(__float_as_uint(av.w))};
#pragma unroll
      for (int u2 = 0; u2 < kRTilesMax; ++u2) {
        const int t = part + u2 * kWpm;
        if (t < q.nrt) {
          const uint2 w = k1::lds64(w_lane + static_cast<uint32_t>((t * 64 + ks) * 256));
          k1::mma1688(acc[u2], a, w.x, w.y);
        }
      }
    }
  }
  const int gq = lane >> 2, t4 = lane & 3;
  const int r0 = mi * 16 + gq, r1 = r0 + 8;
#pragma unroll
  for (int u2 = 0; u2 < kRTilesMax; ++u2) {
    const int t = part + u2 * kWpm;
    if (t < q.nrt) {
      const int col = (q.rt0 + t) * 8 + t4 * 2;
      if (r0 < p.R)
        k1::st_rlx2(p.board + (static_cast<long long>(q.c) * p.R + r0) * kNE + col, k1::tagged(tag, acc[u2][0]),
                    k1::tagged(tag, acc[u2][1]));
      if (r1 < p.R)
        k1::st_rlx2(p.board + (static_cast<long long>(q.c) * p.R + r1) * kNE + col, k1::tagged(tag, acc[u2][2]),
                    k1::tagged(tag, acc[u2][3]));
    }
  }
}

// Routed row m on this CTA: the four router warps sum the row's 16 x 256 tagged partials in chunk order (two 32-expert
// groups per warp), then warp 0 of the group runs the routing arithmetic and releases the row's ready words.
__device__ __forceinline__ void route_job(const Params& p, float* lg, int m, int rw, int lane, uint32_t tag) {
  for (int gi = rw; gi < kNE / 32; gi += 4) {
    const int e = gi * 32 + lane;
    unsigned long long v[kChunks];
    unsigned pend = 0xffffu;
    unsigned it = 0;
    while (pend) {
#pragma unroll
      for (int cc = 0; cc < kChunks; ++cc)
        if (pend & (1u << cc)) v[cc] = k1::ld_rlx(p.board + (static_cast<long long>(cc) * p.R + m) * kNE + e);
      unsigned left = 0;
#pragma unroll
      for (int cc = 0; cc < kChunks; ++cc)
        if ((pend & (1u << cc)) && static_cast<uint32_t>(v[cc] >> 32) != tag) left |= 1u << cc;
      pend = left;
      if (pend) {
        if (++it > kPoll * static_cast<unsigned>(p.tmo_scale)) {
          fail(p, kErrBoard);
          break;
        }
        __nanosleep(32);
      }
    }
    float a = __uint_as_float(static_cast<uint32_t>(v[0]));
#pragma unroll
    for (int cc = 1; cc < kChunks; ++cc) a = __fadd_rn(a, __uint_as_float(static_cast<uint32_t>(v[cc])));
    lg[e] = a;
  }
#ifndef MUSE_VR_C4F_TEST_PLANT_RACE
  k1::bar_named(kBarRoute, 128);  // the row's 256 logits are in shared memory
#endif  // test-only build: the barrier dropped, a planted shared-memory race (sanitizer non-vacuity control)
  if (rw == 0) {
    fr::RouteArgs ra{};
    ra.logits = lg;
    ra.bias = p.bias;
    ra.topk_w = p.topk_w;
    ra.topk_ids = p.topk_ids;
    ra.packed = p.packed;
    ra.local_expert_mapping = nullptr;
    ra.num_valid_tokens = nullptr;
    ra.words = p.rwords;
    ra.num_tokens = p.R;
    ra.num_experts = kNE;
    ra.stride_gm = 0;
    ra.router_output_multiplier = p.mult;
    ra.route_scale = p.route_scale;
    ra.division_eps = p.div_eps;
    fr::route_row<8, 8>(ra, m, lane, tag);
    __syncwarp();
    if (lane == 0) C4F_STAMP(p.stamps[kSIds + m]);
  }
}

// Balance role (one CTA, the four router warps after their chains): wait until every owned row is routed (its ready
// words carry this tag), then the bal assignment over the step's ids (count with all 128 threads, the one-warp assignment,
// virtual ids with all threads), then one release of tag << 32 | digest.
__device__ __forceinline__ void bal_job(const Params& p, bl::Smem& bs, const int32_t* ids_s, int tb, int rw, int lane,
                                        uint32_t tag) {
  for (int i = tb; i < p.R * 8; i += 128) wait_tag(p, p.rwords + i, tag, kErrRouted);
  k1::bar_named(kBarBal, 128);
  if (tb == 0) C4F_STAMP(p.stamps[kSBal0]);
  uint32_t err = bl::count_ids(ids_s, p.Mtot * 8, bs, tb, 128);
  k1::bar_named(kBarBal, 128);
  if (tb == 0) C4F_STAMP(p.stamps[kSBalCnt]);
  if (rw == 0) err |= bl::assign_warp<4>(p.bal_pass_rows, p.bal_pass_inv, bs, static_cast<uint32_t>(lane), nullptr);
  k1::bar_named(kBarBal, 128);
  if (tb == 0) C4F_STAMP(p.stamps[kSBalAsg]);
  const uint32_t pool = static_cast<uint32_t>(p.bal_pool);
  for (int i = tb; i < p.Mtot * 8; i += 128) {
    const uint32_t w = static_cast<uint32_t>(ids_s[i]);
    const uint32_t e = (w >> 16) & 0xffffu;
    uint32_t out = w;
    if (e < static_cast<uint32_t>(kNE)) {
      const uint32_t r = bs.asg[e];
      const uint32_t t = bs.tab[e];
      const uint32_t slot = r == (t & 0xfu) ? ((t >> 8) & 0xffu) : ((t >> 16) & 0xffu);
      out = ((r * pool + slot) << 16) | (w & 0xffffu);
    }
    p.bal_vids[i] = static_cast<int32_t>(out);
  }
  for (int i = tb; i < kNE; i += 128) p.bal_asg[i] = bs.asg[i];
  if (err != 0u) fail(p, kErrBal);
  k1::bar_named(kBarBal, 128);  // every virtual-id store precedes the release below
  if (tb == 0) C4F_STAMP(p.stamps[kSBalVid]);
  if (rw == 0) {
    const uint32_t dg = bl::digest_warp(bs, static_cast<uint32_t>(lane));
    if (lane == 0) {
      k1::st_rel(p.bal_word, (static_cast<unsigned long long>(tag) << 32) | dg);
      C4F_STAMP(p.stamps[kSAsg]);
    }
  }
}

template <int kMT, bool kHasW, int kNVT>
__global__ void __launch_bounds__(kThreads, 1) c4f_kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  unsigned char* smem = smem_raw + ((1024u - (k1::smem_u32(smem_raw) & 1023u)) & 1023u);
  const Layout L = layout(p.Rpad, p.NS, p.stile, p.Mtot);
  unsigned long long* bars = reinterpret_cast<unsigned long long*>(smem + L.bars);  // full[3], empty[3], x, w
  bl::Smem& bs = *reinterpret_cast<bl::Smem*>(smem + L.bal);
  int32_t* ids_s = reinterpret_cast<int32_t*>(smem + L.ids);
  float* lg = reinterpret_cast<float*>(smem + L.lg);
  float* qsm = reinterpret_cast<float*>(smem + L.qsm);
  __shared__ unsigned long long s_epoch;

  const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
  const int b = static_cast<int>(blockIdx.x);
  const Plan q = plan_of(b, p.G, p.R, p.stile);
  const bool do_pool = (p.roles & kRolePool) != 0, do_rtr = (p.roles & kRoleRouter) != 0;
  const bool do_bal = (p.roles & kRoleBal) != 0 && do_rtr && b == p.bal_cta;
  const bool npl = (p.roles & kRoleNoPrestage) != 0;
  const uint32_t ring = k1::smem_u32(smem + L.ring);

  unsigned long long go_start = 0, t_start = 0;
  if (tid == 0) {
    go_start = k1::ld_rlx(p.ctl + 0);
    C4F_NOW(t_start);
    for (int i = 0; i < p.NS; ++i) {
      k1::mbar_init(k1::smem_u32(bars + i), 1);
      k1::mbar_init(k1::smem_u32(bars + kNSMax + i), 4);
    }
    k1::mbar_init(k1::smem_u32(bars + 2 * kNSMax), 1);
    k1::mbar_init(k1::smem_u32(bars + 2 * kNSMax + 1), 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  auto issue_stage = [&](int s) {
    const int slot = s % p.NS;
    const int tiles = min(p.stile, q.npt - s * p.stile);
    const uint32_t full = k1::smem_u32(bars + slot);
    k1::mbar_expect_tx(full, static_cast<unsigned>(tiles * kPTileBytes));
    const char* src = reinterpret_cast<const char*>(p.wp_frag) +
                      (static_cast<long long>(q.c) * kPTiles + q.pt0 + s * p.stile) * kPTileBytes;
    for (int t = 0; t < tiles; ++t)
      k1::bulk_g2s(ring + static_cast<uint32_t>(slot * p.stile * kPTileBytes + t * kPTileBytes), src + t * kPTileBytes, kPTileBytes,
                   full);
  };

  // ---- before the go word: static weights only ----
  if (warp == 1 && lane == 0) {
    if (do_rtr && q.nrt > 0) {
      const uint32_t wb = k1::smem_u32(bars + 2 * kNSMax + 1);
      k1::mbar_expect_tx(wb, static_cast<unsigned>(q.nrt * kRTileBytes));
      k1::bulk_g2s(k1::smem_u32(smem + L.wr),
                   reinterpret_cast<const char*>(p.wr_frag) +
                       (static_cast<long long>(q.c) * kRTiles + q.rt0) * kRTileBytes,
                   static_cast<unsigned>(q.nrt * kRTileBytes), wb);
    }
    if (do_pool && !npl)
      for (int s = 0; s < min(p.NS, q.nstg); ++s) issue_stage(s);
  }
  if (do_bal && warp >= 6) {
    const int tb = tid - 192;
    if (warp == 6) bl::stage_table(p.bal_table, bs, static_cast<uint32_t>(lane));
    for (int i = tb; i < kNE; i += 128) bs.cnt[i] = 0u;
    for (int i = tb; i < p.Mtot * 8; i += 128) ids_s[i] = p.bal_ids[i];
  }
  __syncthreads();

  // ---- go ----
  if (tid == 0) {
    k1::atom_add_rel(p.ctl + 1, 1ull);
    if (b == 0) {
      const unsigned long long e = p.ctl[2] + 1;  // the previous launch's epoch, ordered by the kernel boundary
      unsigned it = 0;
      while (k1::ld_acq(p.ctl + 1) < static_cast<unsigned long long>(p.G) * e) {
        if (++it > kPoll * static_cast<unsigned>(p.tmo_scale)) {
          fail(p, kErrArrive);
          break;
        }
      }
      C4F_PREGO_WAIT(t_start, p.prego_ns);
      C4F_STAMP(p.stamps[kSGo]);
      k1::st_rel(p.ctl + 0, e);
      p.ctl[2] = e;
      s_epoch = e;
    } else {
      unsigned it = 0;
      unsigned long long v;
      while ((v = k1::ld_acq(p.ctl + 0)) == go_start) {
        if (++it > kPoll * static_cast<unsigned>(p.tmo_scale)) {
          fail(p, kErrGo);
          break;
        }
      }
      s_epoch = v;
    }
    C4F_STAMP(p.stamps[kSGoSeen + b]);
  }
  __syncthreads();
  const unsigned long long epoch = s_epoch;
  const uint32_t tag = static_cast<uint32_t>(epoch);

  // ---- owner-row boundary (front-of-MoE mode): warps 2-9 of the CTA that owns row quant_m write that row's images ----
  if constexpr (kNVT > 0) {
    if (p.nbranch != nullptr && q.quant_m >= 0 && warp >= 2) {
      const int gtid = tid - 64, m = q.quant_m;
      muse_vr_c4n::NormArgs na{};
      na.branch = p.nbranch;
      na.stream = p.nstream;
      na.new_stream = p.nnew_stream;
      na.xp_img = p.xp_mut;
      na.xr_img = p.xr_mut;
      na.Rpad = p.Rpad;
      na.mt = kMT;
      na.pre_weight = p.npre_w;
      na.alpha = p.nalpha;
      na.beta = p.nbeta;
      na.post_eps = p.npost_eps;
      na.pre_eps = p.npre_eps;
      na.channel_wise = p.nchannel != 0;
      na.pre_norm_bf16 = false;
      // the served attention -> MoE boundary: no post-norm weight, gated residual, router input in fp32
      muse_vr_c4n::owner_norm_row<8192, 256, kNVT, false, true, false, true>(na, m, m, gtid, kBarQuant, qsm);
      k1::bar_named(kBarQuant, 256);  // every image / stream store of the row precedes the release below
      if (gtid == 0) k1::st_rel(p.nwords + m, (static_cast<unsigned long long>(tag) << 32) | static_cast<uint32_t>(m));
    }
  }

  // ---- roles ----
  if (warp == 1) {
    if (do_pool) {
      if (p.nbranch != nullptr) {
        // front-of-MoE mode: every owned row's image is written by the boundary role (one acquire per row word)
        for (int i = lane; i < p.R; i += 32) wait_tag(p, p.nwords + i, tag, kErrStage);
        __syncwarp();
      }
      if (lane == 0) {
        // the input chunk was written by generic-proxy stores ordered before the go release or the row words (acquired
        // above), so the copy issue is fenced against the async proxy
        asm volatile("fence.proxy.async.global;" ::: "memory");
        const uint32_t xb = k1::smem_u32(bars + 2 * kNSMax);
        const unsigned xbytes = static_cast<unsigned>(p.Rpad * kXpRowBytes);
        k1::mbar_expect_tx(xb, xbytes);
        k1::bulk_g2s(k1::smem_u32(smem + L.xp), p.xp_img + static_cast<long long>(q.c) * p.Rpad * kXpPad, xbytes, xb);
      }
      for (int s = npl ? 0 : min(p.NS, q.nstg); s < q.nstg; ++s) {
        if (s >= p.NS) {
          // slot reuse: every pooler MMA warp released the slot (whole warp on the slot's named barrier)
          k1::bar_named(kBarSlot0 + s % p.NS, kSlotThreads);
          // the slot's generic-proxy reads (ordered by the barrier) before this async-proxy write
          if (lane == 0) asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        if (lane == 0) issue_stage(s);
        __syncwarp();
      }
    }
  } else if (warp >= 2 && warp < 6) {
    if (do_pool) {
      pool_warp<kMT>(p, q, smem, L, bars, warp - 2, lane);
      k1::bar_named(kBarPool, 128);  // the four pooler warps' partial stores precede the release below
      if (warp == 2 && lane == 0) {
        k1::st_rel(p.pflag + b, static_cast<unsigned long long>(tag) << 32);
        C4F_STAMP(p.stamps[kSPend + b]);
      }
      slice_sum(p, b, tid - 64, tag);
    }
  } else if (warp >= 6) {
    if (do_rtr) {
      const int rw = warp - 6;
      router_warp<kMT>(p, q, smem, L, bars, rw, lane, tag);
      __syncwarp();
      k1::bar_named(kBarRoute, 128);
      if (rw == 0 && lane == 0) C4F_STAMP(p.stamps[kSRtrEnd + b]);
      if (q.route_m >= 0) route_job(p, lg, q.route_m, rw, lane, tag);
      if (do_bal) bal_job(p, bs, ids_s, tid - 192, rw, lane, tag);
    }
  }

  // ---- norm + quant of this CTA's quantized row (warps 2-9 after their own work) ----
  if (do_pool && q.quant_m >= 0 && warp >= 2) {
    const int gtid = tid - 64;
    if (gtid < p.G) wait_tag(p, p.sflag + gtid, tag, kErrReady);  // every CTA's pooled slice of this launch
    k1::bar_named(kBarQuantIn, 256);  // the whole pooled row is visible to the group after the acquires above
    fr::QuantArgs qa{};
    qa.x = p.pooled;
    qa.weight = kHasW ? p.qweight : nullptr;
    qa.fp4_out = p.fp4;
    qa.sf_out = p.sf;
    qa.per_token_scale = p.pts;
    qa.words = p.qwords;
    qa.eps = p.qeps;
    qa.global_scale_inv = p.gsi;
    fr::quant_row<kHid, 256, kHasW, false, 0, 0>(qa, q.quant_m, gtid, kBarQuant, qsm, tag);
    if (gtid == 0) C4F_STAMP(p.stamps[kSTok + q.quant_m]);
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Image builders (the layouts above) and test-only references.

// router input x [R, 8192] fp32 -> A-fragment order [16][mt][64][32] float4 (rows >= R zero)
__global__ void c4f_prep_xr(const float* __restrict__ x, int R, int mt, float4* __restrict__ img) {
  const long long n = static_cast<long long>(kChunks) * mt * 64 * 32;
  for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<long long>(gridDim.x) * blockDim.x) {
    const int lane = static_cast<int>(i & 31);
    const int ks = static_cast<int>((i >> 5) & 63);
    const long long cm = i >> 11;
    const int mi = static_cast<int>(cm % mt), c = static_cast<int>(cm / mt);
    const int g = lane >> 2, t4 = lane & 3;
    const int r0 = mi * 16 + g, r1 = r0 + 8, k = c * kKC + ks * 8 + t4;
    float4 v;
    v.x = r0 < R ? x[static_cast<long long>(r0) * kK + k] : 0.f;
    v.y = r1 < R ? x[static_cast<long long>(r1) * kK + k] : 0.f;
    v.z = r0 < R ? x[static_cast<long long>(r0) * kK + k + 4] : 0.f;
    v.w = r1 < R ? x[static_cast<long long>(r1) * kK + k + 4] : 0.f;
    img[i] = v;
  }
}

// router weights w [256, 8192] fp32 -> B-fragment order [16][32][64][32] (TF32-rounded with cvt.rna)
__global__ void c4f_prep_wr(const float* __restrict__ w, uint2* __restrict__ img) {
  const long long n = static_cast<long long>(kChunks) * kRTiles * 64 * 32;
  for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<long long>(gridDim.x) * blockDim.x) {
    const int lane = static_cast<int>(i & 31);
    const int ks = static_cast<int>((i >> 5) & 63);
    const int tile = static_cast<int>((i >> 11) % kRTiles);
    const int c = static_cast<int>((i >> 11) / kRTiles);
    const int e = tile * 8 + (lane >> 2), k = c * kKC + ks * 8 + (lane & 3);
    const float* wr = w + static_cast<long long>(e) * kK + k;
    img[i] = make_uint2(k1::to_tf32(__float_as_uint(wr[0])), k1::to_tf32(__float_as_uint(wr[4])));
  }
}

// Test-only reference of the pooler chunk partials: the same chains with fragments read straight from the row-major
// tensors (x [R, 8192] bf16, W [4096, 8192] bf16) -> part [16][R][4096] fp32. Grid (512 / 4, 16, mt), 4 warps per CTA.
__global__ void c4f_ref_pool(const uint16_t* __restrict__ x, int R, const uint16_t* __restrict__ w, float* __restrict__ part) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t4 = lane & 3;
  const int tile = blockIdx.x * 4 + warp, c = blockIdx.y, m0 = blockIdx.z * 16;
  const int r0 = m0 + g, r1 = r0 + 8, n = tile * 8 + g;
  auto pair = [&](const uint16_t* base, int row, bool ok, int k) -> uint32_t {
    if (!ok) return 0u;
    const uint16_t* q = base + static_cast<long long>(row) * kK + k;
    return static_cast<uint32_t>(q[0]) | (static_cast<uint32_t>(q[1]) << 16);
  };
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int ks = 0; ks < 32; ++ks) {
    const int k = c * kKC + ks * 16 + t4 * 2;
    const uint32_t a[4] = {pair(x, r0, r0 < R, k), pair(x, r1, r1 < R, k), pair(x, r0, r0 < R, k + 8),
                           pair(x, r1, r1 < R, k + 8)};
    mma16816(acc, a, pair(w, n, true, k), pair(w, n, true, k + 8));
  }
  const int col = tile * 8 + t4 * 2;
  float* pp = part + static_cast<long long>(c) * R * kHid;
  if (r0 < R) {
    pp[static_cast<long long>(r0) * kHid + col] = acc[0];
    pp[static_cast<long long>(r0) * kHid + col + 1] = acc[1];
  }
  if (r1 < R) {
    pp[static_cast<long long>(r1) * kHid + col] = acc[2];
    pp[static_cast<long long>(r1) * kHid + col + 1] = acc[3];
  }
}

// Test-only reference of the quant of each row from the 16 partial slices (one 256-thread CTA per row, no words).
template <bool kHasW>
__global__ void __launch_bounds__(256) c4f_ref_quant(const fr::QuantArgs qa, int R) {
  __shared__ float smem[16];
  const int row = static_cast<int>(blockIdx.x);
  if (row < R) fr::quant_row<kHid, 256, kHasW, false, kChunks, 0>(qa, row, static_cast<int>(threadIdx.x), 1, smem, 0u);
}

}  // namespace muse_vr_c4f

// ---------------------------------------------------------------------------------------------------------------------
struct WfcC4F {
  static void run(tvm::ffi::TensorView xr_img, tvm::ffi::TensorView wr_frag, tvm::ffi::TensorView bias,
                  tvm::ffi::TensorView wp_frag, tvm::ffi::TensorView xp_img, tvm::ffi::TensorView qweight, int64_t has_w,
                  tvm::ffi::TensorView board, tvm::ffi::TensorView topk_w, tvm::ffi::TensorView topk_ids,
                  tvm::ffi::TensorView packed, tvm::ffi::TensorView rwords, tvm::ffi::TensorView part,
                  tvm::ffi::TensorView pflag, tvm::ffi::TensorView sflag, tvm::ffi::TensorView fp4, tvm::ffi::TensorView sf,
                  tvm::ffi::TensorView pts, tvm::ffi::TensorView qwords, tvm::ffi::TensorView pooled,
                  tvm::ffi::TensorView bal_ids, tvm::ffi::TensorView bal_table, tvm::ffi::TensorView bal_vids,
                  tvm::ffi::TensorView bal_asg, tvm::ffi::TensorView bal_word, tvm::ffi::TensorView ctl,
                  tvm::ffi::TensorView stamps, int64_t R, int64_t G, int64_t NS, int64_t stile, int64_t oversize, int64_t roles, int64_t prego_ns,
                  int64_t tmo_scale, int64_t Mtot, int64_t bal_pool, int64_t bal_cta, int64_t bal_pass_rows, double mult,
                  double route_scale, double div_eps, double qeps, double gsi, int64_t nvt, tvm::ffi::TensorView nbranch,
                  tvm::ffi::TensorView nstream, tvm::ffi::TensorView nnew_stream, tvm::ffi::TensorView npre_w,
                  int64_t has_npre_w, tvm::ffi::TensorView nalpha, tvm::ffi::TensorView nbeta, double npost_eps,
                  double npre_eps, tvm::ffi::TensorView nwords) {
    using namespace host;
    namespace c = muse_vr_c4f;
    RuntimeCheck(R >= 1 && R <= c::kRowsMax, "wfc_c4f: owned rows 1..64");
    RuntimeCheck(G >= c::kMinGrid && G <= c::kMaxGrid && 2 * R < G, "wfc_c4f: grid size");
    RuntimeCheck(NS >= 1 && NS <= c::kNSMax, "wfc_c4f: ring depth 1..3");
    RuntimeCheck(stile == 4 || stile == 8, "wfc_c4f: 4 or 8 pooler tiles per stage");
    RuntimeCheck(Mtot >= R && Mtot <= c::kBalRowsMax, "wfc_c4f: step rows");
    const int Rpad = static_cast<int>((R + 15) / 16 * 16), mt = Rpad / 16;
    RuntimeCheck(xp_img.numel() == static_cast<int64_t>(c::kChunks) * Rpad * c::kXpPad, "wfc_c4f: pooler input image");
    RuntimeCheck(xr_img.numel() == static_cast<int64_t>(c::kChunks) * mt * 64 * 32 * 4, "wfc_c4f: router input image");
    RuntimeCheck(part.numel() == static_cast<int64_t>(c::kChunks) * R * c::kHid, "wfc_c4f: partials");
    RuntimeCheck(stamps.numel() >= c::kNumStamps, "wfc_c4f: stamps");
    RuntimeCheck(pflag.numel() >= G && sflag.numel() >= G && pooled.numel() == R * c::kHid, "wfc_c4f: slice words / pooled rows");
    c::Params p{};
    p.xr_img = static_cast<const float4*>(xr_img.data_ptr());
    p.wr_frag = static_cast<const uint2*>(wr_frag.data_ptr());
    p.bias = static_cast<const float*>(bias.data_ptr());
    p.wp_frag = static_cast<const uint2*>(wp_frag.data_ptr());
    p.xp_img = static_cast<const uint16_t*>(xp_img.data_ptr());
    p.qweight = has_w ? qweight.data_ptr() : nullptr;
    p.board = static_cast<unsigned long long*>(board.data_ptr());
    p.topk_w = static_cast<float*>(topk_w.data_ptr());
    p.topk_ids = static_cast<int32_t*>(topk_ids.data_ptr());
    p.packed = static_cast<int32_t*>(packed.data_ptr());
    p.rwords = static_cast<unsigned long long*>(rwords.data_ptr());
    p.part = static_cast<float*>(part.data_ptr());
    p.pflag = static_cast<unsigned long long*>(pflag.data_ptr());
    p.sflag = static_cast<unsigned long long*>(sflag.data_ptr());
    p.fp4 = static_cast<uint8_t*>(fp4.data_ptr());
    p.sf = static_cast<uint8_t*>(sf.data_ptr());
    p.pts = static_cast<float*>(pts.data_ptr());
    p.qwords = static_cast<unsigned long long*>(qwords.data_ptr());
    p.pooled = static_cast<uint16_t*>(pooled.data_ptr());
    p.bal_ids = static_cast<const int32_t*>(bal_ids.data_ptr());
    p.bal_table = static_cast<const uint32_t*>(bal_table.data_ptr());
    p.bal_vids = static_cast<int32_t*>(bal_vids.data_ptr());
    p.bal_asg = static_cast<uint8_t*>(bal_asg.data_ptr());
    p.bal_word = static_cast<unsigned long long*>(bal_word.data_ptr());
    p.ctl = static_cast<unsigned long long*>(ctl.data_ptr());
    p.stamps = static_cast<unsigned long long*>(stamps.data_ptr());
    p.R = static_cast<int>(R);
    p.Rpad = Rpad;
    p.G = static_cast<int>(G);
    p.NS = static_cast<int>(NS);
    p.stile = static_cast<int>(stile);
    p.roles = static_cast<int>(roles);
    p.prego_ns = static_cast<int>(prego_ns);
    p.tmo_scale = static_cast<int>(tmo_scale < 1 ? 1 : tmo_scale);
    p.Mtot = static_cast<int>(Mtot);
    p.bal_pool = static_cast<int>(bal_pool);
    p.bal_cta = static_cast<int>(bal_cta);
    p.bal_pass_rows = static_cast<uint32_t>(bal_pass_rows);
    p.bal_pass_inv = static_cast<uint32_t>((65536 + bal_pass_rows - 1) / bal_pass_rows);
    p.mult = static_cast<float>(mult);
    p.route_scale = static_cast<float>(route_scale);
    p.div_eps = static_cast<float>(div_eps);
    p.qeps = static_cast<float>(qeps);
    p.gsi = static_cast<float>(gsi);
    RuntimeCheck(nvt == 0 || nvt == 256 || nvt == 512, "wfc_c4f: boundary thread count 0 / 256 / 512");
    if (nvt != 0) {
      RuntimeCheck(nbranch.numel() == R * 8192 && nstream.numel() == R * 8192 && nnew_stream.numel() == R * 8192 &&
                       nwords.numel() >= R && (nalpha.numel() == 1 || nalpha.numel() == 8192) &&
                       nbeta.numel() == nalpha.numel(),
                   "wfc_c4f: boundary buffers");
      p.nbranch = static_cast<const uint16_t*>(nbranch.data_ptr());
      p.nstream = static_cast<const float*>(nstream.data_ptr());
      p.nnew_stream = static_cast<float*>(nnew_stream.data_ptr());
      p.npre_w = has_npre_w ? static_cast<const uint16_t*>(npre_w.data_ptr()) : nullptr;
      p.nalpha = static_cast<const float*>(nalpha.data_ptr());
      p.nbeta = static_cast<const float*>(nbeta.data_ptr());
      p.npost_eps = static_cast<float>(npost_eps);
      p.npre_eps = static_cast<float>(npre_eps);
      p.nchannel = nalpha.numel() == 8192 ? 1 : 0;
      p.nwords = static_cast<unsigned long long*>(nwords.data_ptr());
      p.xr_mut = static_cast<float*>(xr_img.data_ptr());
      p.xp_mut = static_cast<uint16_t*>(xp_img.data_ptr());
    }
    if (roles & c::kRoleBal) {
      RuntimeCheck(bal_pass_rows >= 1 && bal_pass_rows <= 128, "wfc_c4f: pass rows");
      for (int64_t x = 0; x <= c::kBalRowsMax + bal_pass_rows; ++x)  // a count is at most the step rows
        RuntimeCheck(((uint64_t(x) * p.bal_pass_inv) >> 16) == uint64_t(x / bal_pass_rows), "wfc_c4f: pass reciprocal");
      RuntimeCheck(bal_cta >= 0 && bal_cta < G && bal_pool >= c::kNE / 4 && bal_pool <= 128, "wfc_c4f: balance CTA / pool");
      RuntimeCheck(bal_ids.numel() >= Mtot * 8 && bal_vids.numel() >= Mtot * 8 && bal_table.numel() == c::kNE,
                   "wfc_c4f: balance buffers");
    }
    const c::Layout L = c::layout(Rpad, p.NS, p.stile, p.Mtot);
    const int smem = L.total + c::kSmemSlack;
    const cudaStream_t stream = LaunchKernel::resolve_device(xp_img.device());
    int dev = 0, optin = 0;
    RuntimeDeviceCheck(cudaGetDevice(&dev));
    RuntimeDeviceCheck(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    RuntimeCheck(oversize != 0 || smem <= optin, "wfc_c4f: shared memory above the per-block opt-in (use oversize)");
    auto launch = [&](auto kernel) {
      if (oversize) {
        // the larger per-block shared memory configuration (CUDA Programming Guide, shared memory capacity per compute
        // capability) is enabled per kernel through the shared-memory mode attribute; the launch then requests the larger
        // dynamic size directly (the dynamic-size attribute is not raised in that mode)
        RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeSharedMemoryMode,
                                                static_cast<int>(cudaSharedMemoryModeAllowOversizedSharedMemory)));
      } else {
        RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
      }
      cudaLaunchConfig_t cfg{};
      cfg.gridDim = dim3(static_cast<unsigned>(G));
      cfg.blockDim = dim3(c::kThreads);
      cfg.dynamicSmemBytes = static_cast<size_t>(smem);
      cfg.stream = stream;
      cudaLaunchAttribute attr[1];
      attr[0].id = cudaLaunchAttributeCooperative;
      attr[0].val.cooperative = 1;
      cfg.attrs = attr;
      cfg.numAttrs = 1;
      RuntimeDeviceCheck(cudaLaunchKernelEx(&cfg, kernel, p));
    };
#define C4F_W(MT, NV) \
  if (has_w) launch(c::c4f_kernel<MT, true, NV>); else launch(c::c4f_kernel<MT, false, NV>);
#define C4F_CASE(MT)                                                                         \
  case MT:                                                                                   \
    if (nvt == 256) { C4F_W(MT, 256) } else if (nvt == 512) { C4F_W(MT, 512) } else { C4F_W(MT, 0) } \
    break;
    switch (mt) {
      C4F_CASE(1)
      C4F_CASE(2)
      C4F_CASE(3)
      C4F_CASE(4)
      default:
        RuntimeCheck(false, "wfc_c4f: row tiles");
    }
#undef C4F_CASE
#undef C4F_W
  }

  static void prep(tvm::ffi::TensorView xr, int64_t R, tvm::ffi::TensorView xr_img, tvm::ffi::TensorView wr,
                   tvm::ffi::TensorView wr_frag) {
    using namespace host;
    const int mt = static_cast<int>((R + 15) / 16);
    const cudaStream_t stream = LaunchKernel::resolve_device(xr.device());
    muse_vr_c4f::c4f_prep_xr<<<256, 256, 0, stream>>>(static_cast<const float*>(xr.data_ptr()), static_cast<int>(R), mt,
                                                      static_cast<float4*>(xr_img.data_ptr()));
    muse_vr_c4f::c4f_prep_wr<<<512, 256, 0, stream>>>(static_cast<const float*>(wr.data_ptr()),
                                                      static_cast<uint2*>(wr_frag.data_ptr()));
  }

  static void ref_pool(tvm::ffi::TensorView x, int64_t R, tvm::ffi::TensorView w, tvm::ffi::TensorView part) {
    using namespace host;
    const cudaStream_t stream = LaunchKernel::resolve_device(x.device());
    const dim3 grid(muse_vr_c4f::kPTiles / 4, muse_vr_c4f::kChunks, static_cast<unsigned>((R + 15) / 16));
    muse_vr_c4f::c4f_ref_pool<<<grid, 128, 0, stream>>>(static_cast<const uint16_t*>(x.data_ptr()), static_cast<int>(R),
                                                        static_cast<const uint16_t*>(w.data_ptr()),
                                                        static_cast<float*>(part.data_ptr()));
  }

  static void ref_quant(tvm::ffi::TensorView part, int64_t R, tvm::ffi::TensorView qweight, int64_t has_w,
                        tvm::ffi::TensorView fp4, tvm::ffi::TensorView sf, tvm::ffi::TensorView pts,
                        tvm::ffi::TensorView pooled, double qeps, double gsi) {
    using namespace host;
    muse_vr_front::QuantArgs qa{};
    qa.weight = has_w ? qweight.data_ptr() : nullptr;
    qa.fp4_out = fp4.data_ptr();
    qa.sf_out = sf.data_ptr();
    qa.per_token_scale = static_cast<float*>(pts.data_ptr());
    qa.words = nullptr;
    qa.eps = static_cast<float>(qeps);
    qa.global_scale_inv = static_cast<float>(gsi);
    qa.partials = static_cast<const float*>(part.data_ptr());
    qa.pstride = R * muse_vr_c4f::kHid;
    qa.pooled_out = pooled.data_ptr();
    const cudaStream_t stream = LaunchKernel::resolve_device(part.device());
    if (has_w)
      muse_vr_c4f::c4f_ref_quant<true><<<static_cast<unsigned>(R), 256, 0, stream>>>(qa, static_cast<int>(R));
    else
      muse_vr_c4f::c4f_ref_quant<false><<<static_cast<unsigned>(R), 256, 0, stream>>>(qa, static_cast<int>(R));
  }
};

}  // namespace sglang
