// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// K3.1 MK TSIM: the token-split indexer's local selection, exchange and exact merge in ONE kernel per DSA layer.
// select_kth and the (score desc, position asc) key are adapted from SGLang's
// python/sglang/kernels/jit/csrc/dsa/dcp_topk_merge.cuh (Apache-2.0; commit f3532057e15f), as in tsi.cuh.
// The fast path is the measured kernel TF3 with fence form (d).
//
// Page striping as tsi.cuh: rank r scores the global pages p with p % 16 == r; local token l <-> global position
// pos = (16 (l / 64) + r) 64 + l % 64 (monotonic in l).
//
// One CTA of 1024 threads per row (serving: grid [rows], rank = this GPU; tests: grid [rows, 16] emulates the 16 ranks on
// one GPU with a cooperative launch and unicast pushes into 16 regions):
//   sender    the local row (local logits, nloc = local causal length) in float4 groups (groups 0..7 in registers, 8..15
//             staged in SMEM by cp.async), 4 running maxima per thread; tau = the value-linear 2048-bin histogram of the
//             finite maxima at rank 256, lowered by M 2^-19 (>= 256 distinct local keys are >= tau); every local key
//             >= tau is a record {score bits, global position}, in local-index order, after one ordered block scan;
//             header {n | ovf | complete, tau, hi, 0}. Pushed to every rank's CAND slot [parity][row][r]: multimem.st
//             through the region's multicast VA (mc != 0), else a unicast store to each of the 16 regions.
//   receiver  polls the 16 sources' headers and records in its own region; lo = max tau over the sources that sent;
//             value-linear 4096-bin fine + 128 coarse histograms of the union keys >= lo find the bin b* holding rank
//             want = min(n, 2048); the boundary bin's keys are ranked by (score desc, position asc) -> t*; certificate:
//             every complete source sent everything, every other source every key >= its tau <= lo <= t* (the slow
//             pass, lo = union min, checks tau_s <= t* for each incomplete source). Emits every key >= t* SOURCE-MAJOR
//             (source = (pos / 64) % 16 ascending), position-ascending inside a source (= the s1r merge's order).
//   fallback  any flag (1 a source sent > 384 keys (ovf), 4 certificate, 32 the union holds < want keys, 64 the boundary
//             bin holds > 1024 keys, 128 emitted != want), identical on every rank (same records): every rank writes its
//             EXACT local top-min(nloc, 2048) by the 64-bit key (canonical score, -0 == +0; position asc) in local-index
//             order to its OWN FULL slot; every rank polls the 16 FULL lists through the peers' unicast VAs into scratch,
//             selects the want-th key t* of their union (it holds the global top-want) and emits the keys >= t*
//             source-major, position-ascending. fallback[row] = the flag bits (0 = fast path).
//
// Synchronization (memory-model audit):
//   E-R  every cross-GPU datum is a 64-bit element {payload32 low, tag32 high}; multimem.st.relaxed.sys.global.b64 (two per
//        record) or st.relaxed.sys.global.v2.u64 (unicast); polled with ld.relaxed.sys.global.v2.u64 through the reader's
//        own unicast VA until both elements carry the expected tag. Multicast writes (K31MK_TSIM_MC=1) are read through the
//        unicast VA with no proxy fence, as SGLang's symmetric-memory multicast all-gathers do; the per-element tag is what
//        makes a stale or partial element unacceptable. K31_TSIM_ALIAS_FENCE=1 (default 0) restores the fenced form: ONE
//        fence.proxy.alias by thread 0 after the first CTA barrier that follows every thread's last poll of the CAND region
//        (the pass-0 histogram barrier); no CAND element is read after it.
//   E-T  tag = make_tag(device epoch, DSA ordinal, kind) as tsi.cuh (never 0); CAND / FULL slots alternate parity by
//        ordinal; a rank cannot push layer l + 2 before every rank finished reading layer l (its layer l + 1 receive
//        needs every rank's layer l + 1 records, written only after that rank's layer l kernel completed).
//   E-F  FULL lists: written by their owner with st.relaxed.sys through its own unicast VA, read by peers with
//        ld.relaxed.sys through the owner's unicast VA (same proxy; R2 as v6's FULL kind).
//   E-P  griddepcontrol.wait before any load of the local logits, local / global lengths, table or epoch; plain (weak,
//        non-.nc) loads; launch_dependents at the end.
//   E-Q  every poll traps %globaltimer deadline_ns after its loop starts (fail closed: no output on that path).
//   E-L  SMEM phases separated by CTA barriers only (no cluster, no DSMEM).
#pragma once
#ifndef K31_TSIM_ALIAS_FENCE
#define K31_TSIM_ALIAS_FENCE 0
#endif
// Unicast push (the production push): the record loop stages every record (and thread 0 the header) in SMEM; after a
// CTA barrier all 1024 threads store the 16 x (N + 2) (slot, destination) pairs, own region included (STG; the direct form,
// 16 stores per record from the record loop, was measured slower and is dropped).
#include <cstdint>
#include <type_traits>

namespace k31_tsim {

constexpr int kT = 1024, kWarps = 32, kTauRank = 256, kCap = 384, kSlots = kCap + 2, kSBins = 2048;
constexpr int kW = 16, kPage = 64, kTopK = 2048, kRBins = 4096, kPer = 6, kHalf = 192, kSpec = 320, kTie = 1024;
constexpr int kSCo = kSBins / 32, kRCo = kRBins / 32;  // coarse bins (32 fine bins each)
constexpr int kVReg = 8;                              // float4 groups per thread held in registers (groups 0..7; 8..15 staged
                                                      // in SMEM; one sender body per kernel, VR8 / ONEV)
constexpr int kVMaxAll = 16;                          // 65,536 local keys = 1,048,576 tokens
constexpr int kKinds = 2, kKindCand = 0, kKindFull = 1;
constexpr unsigned kFull = 0xffffffffu;
constexpr uint32_t kEmptyPos = 0xFFFFFFFFu;
constexpr int kMeta = 8;

// ------------------------------------------------------------------------------------------------ primitives
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void fence_alias() { asm volatile("fence.proxy.alias;" ::: "memory"); }
__device__ __forceinline__ uint64_t gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ uint64_t rec(uint32_t payload, uint32_t tag) {
  return (static_cast<uint64_t>(tag) << 32) | payload;
}
__device__ __forceinline__ uint32_t lo32(uint64_t r) { return static_cast<uint32_t>(r); }
__device__ __forceinline__ uint32_t hi32(uint64_t r) { return static_cast<uint32_t>(r >> 32); }
__device__ __forceinline__ void st_rec2(uint64_t addr, uint64_t a, uint64_t b) {
  asm volatile("st.relaxed.sys.global.v2.u64 [%0], {%1, %2};" ::"l"(addr), "l"(a), "l"(b) : "memory");
}
__device__ __forceinline__ void mm_st(uint64_t addr, uint64_t a) {
  asm volatile("multimem.st.relaxed.sys.global.b64 [%0], %1;" ::"l"(addr), "l"(a) : "memory");
}
__device__ __forceinline__ void ld_rec2(const uint64_t* p, uint64_t& a, uint64_t& b) {
  asm volatile("ld.relaxed.sys.global.v2.u64 {%0, %1}, [%2];" : "=l"(a), "=l"(b) : "l"(p) : "memory");
}
// A value a previous grid of this stream wrote (read after griddepcontrol.wait; never ld.global.nc).
__device__ __forceinline__ int ld_gpu_i32(const int32_t* p) {
  int v;
  asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t ld_gpu_u32(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
// plain weak loads of the predecessor grid's logits (after griddepcontrol.wait); the PTX must hold no ld.global.nc
__device__ __forceinline__ float4 ld_f4(const float4* p) { return *p; }
__device__ __forceinline__ float ld_f(const float* p) { return *p; }
__device__ __forceinline__ void cp_async16(void* smem_dst, const void* gsrc) {
  const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d), "l"(gsrc) : "memory");
}
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }
__device__ __forceinline__ bool tagged(uint64_t a, uint64_t b, uint32_t tag) { return hi32(a) == tag && hi32(b) == tag; }
// A poll's deadline: serving traps (fail closed: nothing after it runs); test_mode 2 records flag 8 and stops polling.
__device__ __forceinline__ bool expired(int test_mode, int* dl, bool& dead) {
  if (!(test_mode & 2)) __trap();
  atomicOr(dl, 1);
  dead = true;
  return true;
}
__device__ __forceinline__ unsigned lanemask_lt() {
  unsigned m;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(m));
  return m;
}
// float <-> int with the same order (redux.sync min / max)
__device__ __forceinline__ int f2o(float f) {
  const int b = __float_as_int(f);
  return b >= 0 ? b : b ^ 0x7FFFFFFF;
}
__device__ __forceinline__ float o2f(int o) { return __int_as_float(o >= 0 ? o : o ^ 0x7FFFFFFF); }
// value-linear bin, unclamped (monotone non-decreasing in x: correctly rounded sub / mul, never contracted)
__device__ __forceinline__ int sbin(float x, float lo, float scale) {
  return __float2int_rd(__fmul_rn(__fsub_rn(x, lo), scale));
}
__device__ __forceinline__ int warp_incl(int v, int lane) {
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int x = __shfl_up_sync(kFull, v, o);
    if (lane >= o) v += x;
  }
  return v;
}
__device__ __forceinline__ float comp(const float4& v, int j) { return j == 0 ? v.x : j == 1 ? v.y : j == 2 ? v.z : v.w; }
__device__ __forceinline__ int bytesum(uint32_t x) {
  return static_cast<int>((x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + (x >> 24));
}

__host__ __device__ __forceinline__ uint32_t make_tag(uint32_t epoch, int ordinal, int kind, int layers) {
  // = tsi.py tag(): ((epoch * layers + ordinal) * 2 + kind + 1) mod 2^32, never 0.
  const uint32_t v = (epoch * static_cast<uint32_t>(layers) + static_cast<uint32_t>(ordinal)) * kKinds +
                     static_cast<uint32_t>(kind) + 1u;
  return v ? v : 1u;
}
__device__ __forceinline__ int gpos(int l, int r) {
  return (kW * (l / kPage) + r) * kPage + l % kPage;
}
// keys: (canonical score desc, position asc) -> larger is better; 0 = empty
__device__ __forceinline__ uint32_t ordered_bits(uint32_t bits) {
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}
__device__ __forceinline__ uint64_t make_key(uint32_t score_bits, int32_t pos) {
  if (pos < 0) return 0;
  return (static_cast<uint64_t>(ordered_bits(score_bits)) << 32) | static_cast<uint32_t>(~static_cast<uint32_t>(pos));
}
__device__ __forceinline__ uint32_t key_pos(uint64_t k) { return ~static_cast<uint32_t>(k); }
__device__ __forceinline__ float canon(float x) {
  return __fadd_rn(x, 0.0f);  // -0 + +0 = +0 (round to nearest): equal scores get equal keys
}

// ------------------------------------------------------------------------------------------------ region layout
// Records (int64) of one rank's region:
//   CAND [2 parity][cap rows][16 src][386 slots][2]   slots 0..383 records, 384 / 385 the header
//   FULL [2 parity][cap rows][2048][2]                 this rank's own exact local top-2048 (fallback rows only)
__host__ __device__ __forceinline__ int64_t cand_index(int parity, int row, int src, int e, int cap) {
  return ((((static_cast<int64_t>(parity) * cap + row) * kW + src) * kSlots) + e) * 2;
}
__host__ __device__ __forceinline__ int64_t full_index(int parity, int row, int i, int cap) {
  return 2LL * cap * kW * kSlots * 2 + ((static_cast<int64_t>(parity) * cap + row) * kTopK + i) * 2;
}
__host__ __device__ __forceinline__ int64_t region_records(int cap) {
  return 2LL * cap * kW * kSlots * 2 + 2LL * cap * kTopK * 2;
}

// ------------------------------------------------------------------------------------------------ params, SMEM
struct Params {
  const float* scores;  // [R][rows][score_stride] local logits; columns [0, ncols) readable
  int64_t score_stride, score_rank_stride;
  const int32_t* lcausal;  // [R][rows] local causal lengths
  int64_t lc_rank_stride;
  const int32_t* causal;  // [rows] global causal lengths
  const int32_t* table;   // [rows / rows_per_request][table_stride] global page table
  int64_t table_stride;
  int32_t* out;  // [R][rows][2048] KV slots
  int64_t out_rank_stride;
  int32_t* fallback;  // [R][rows] flag bits (0 = fast path)
  int64_t fb_rank_stride;
  uint64_t* scratch;  // [R][rows][16 * 2048] fallback union keys
  int64_t scratch_rank_stride;
  const uint32_t* epoch;  // device forward counter (prep_kernel advances it)
  uint64_t peers[kW];     // unicast region base (bytes) of every rank
  uint64_t mc;            // multicast region base (bytes); 0 = unicast push to every peer
  uint64_t deadline_ns;
  int ncols, rank, ordinal, layers, cap, rows_per_request, row0;
  int test_mode;  // meta[7] bits, tests / benchmarks only (0 in serving): 1 one store per record into the own region (mc 0,
                  // no fan-out: the multicast cost model on one GPU); 2 a poll deadline sets flag 8 and the kernel finishes
                  // (no trap: a test bug must not fault a shared GPU)
};

struct SendSmem {
  int hist[kSBins];
  alignas(16) int coarse[kSCo];
  int lo[kWarps], hi[kWarps], nv[kWarps];
  uint32_t t[kVMaxAll / 4][kWarps];
  alignas(16) uint64_t stg[kSlots][2];  // unicast push: this row's records + header, staged
};
struct RecvSmem {
  int hist[kRBins];
  alignas(16) int coarse[kRCo];
  int4 tie[kTie];  // {score bits, global position, holder warp, 0}
  int tsel[kTie];  // slow tie path only: entry selected
  int wmin[kWarps], wc[kWarps];
  int m;
  float ts;
};
struct FbSmem {  // fallback: select_kth + block scans
  int hist[2][256];
  int warp_sum[kWarps];
  int digit, above, cnt;
  unsigned long long min_key;
  int dl;  // test_mode 2: a poll deadline expired
};
struct Smem {
  SendSmem snd;
  RecvSmem rcv;
  FbSmem fb;
};
constexpr size_t kSmemHead = (sizeof(Smem) + 15) / 16 * 16;
__host__ __device__ constexpr size_t smem_bytes(int vmax) {
  return kSmemHead + (vmax > kVReg ? static_cast<size_t>(vmax - kVReg) * kT * 16 : 0);
}

// ------------------------------------------------------------------------------------------------ block helpers
__device__ __forceinline__ int block_exclusive_sum(int v, int& total, FbSmem& s) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int incl = warp_incl(v, lane);
  if (lane == 31) s.warp_sum[warp] = incl;
  __syncthreads();
  const int ws = s.warp_sum[lane];
  total = static_cast<int>(__reduce_add_sync(kFull, static_cast<unsigned>(ws)));
  const int before = static_cast<int>(__reduce_add_sync(kFull, lane < warp ? static_cast<unsigned>(ws) : 0u));
  __syncthreads();
  return before + incl - v;
}
// The rem-th largest nonzero key (rem >= 1) among the keys visited by visit(f) (tsi.cuh / stock select_kth: 8-bit radix
// rounds from the top byte, double-buffered histograms, one-warp scan, early exit once the bucket is exactly the quota).
template <class Visit>
__device__ uint64_t select_kth(Visit visit, int rem, FbSmem& s) {
  if (rem < 1) return 0;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid < 256) {
    s.hist[0][tid] = 0;
    s.hist[1][tid] = 0;
  }
  __syncthreads();
  uint64_t prefix = 0;
  int b = 0;
  for (int shift = 56; shift >= 0; shift -= 8, b ^= 1) {
    int* h = s.hist[b];
    visit([&](uint64_t k) {
      if (k && (shift == 56 || (k >> (shift + 8)) == prefix)) atomicAdd(&h[(k >> shift) & 255], 1);
    });
    if (tid < 256) s.hist[b ^ 1][tid] = 0;  // its last reader finished before the previous round's last barrier
    __syncthreads();
    if (warp == 0) {
      int c[8], tot = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        c[i] = h[255 - (lane * 8 + i)];
        tot += c[i];
      }
      int run = warp_incl(tot, lane) - tot;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int excl = run;
        run += c[i];
        if (run >= rem && excl < rem) {
          s.digit = 255 - (lane * 8 + i);
          s.above = excl;
          s.cnt = c[i];
          s.min_key = ~0ull;
        }
      }
    }
    __syncthreads();
    rem -= s.above;
    prefix = (prefix << 8) | static_cast<uint64_t>(s.digit);
    if (s.cnt == rem) {
      const uint64_t pre = prefix;
      uint64_t mk = ~uint64_t{0};
      visit([&](uint64_t k) {
        if (k && (k >> shift) == pre) mk = k < mk ? k : mk;
      });
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        const uint64_t x = __shfl_xor_sync(kFull, mk, o);
        mk = x < mk ? x : mk;
      }
      if (lane == 0) atomicMin(&s.min_key, static_cast<unsigned long long>(mk));
      __syncthreads();
      return s.min_key;
    }
  }
  return prefix;
}

// ------------------------------------------------------------------------------------------------ push
// One store per record (header in the record loop when not staged): through the multicast VA (every rank's copy; K31MK_TSIM_MC=1), or
// (bench self-push, test_mode 1) into this rank's own region only.
__device__ __forceinline__ void push2(const Params& p, int rank, int64_t idx, uint64_t a, uint64_t b) {
  if (p.mc != 0) {
    const uint64_t at = p.mc + 8ull * static_cast<uint64_t>(idx);
    mm_st(at, a);
    mm_st(at + 8, b);
  } else {  // test_mode 1 (benches): one store into the own region (the multicast cost model on one GPU)
    st_rec2(p.peers[rank] + 8ull * static_cast<uint64_t>(idx), a, b);
  }
}
// Unicast push from SMEM (mc == 0, not the bench self-push): after the CTA barrier that follows the staging, every (slot, destination)
// pair is one st.relaxed.sys.global.v2.u64 of register values (E-R; no slot byte is read back, no E-U edge). Destinations rotate from
// this rank so the 16 ranks do not all hit the same peer first.
__device__ __forceinline__ void unicast_push(const Params& p, int rank, int64_t at0, int N, bool ovf, const SendSmem& sm) {
  const int ns = (ovf ? 0 : N) + 2;  // records (none on overflow) + the 2 header elements
  for (int idx = threadIdx.x; idx < ns * kW; idx += kT) {
    const int d = idx / ns, e0 = idx - d * ns;
    const int e = e0 < ns - 2 ? e0 : kCap + (e0 - (ns - 2));
    st_rec2(p.peers[(rank + d) & (kW - 1)] + 8ull * static_cast<uint64_t>(at0 + 2 * e), sm.stg[e][0], sm.stg[e][1]);
  }
}

// ------------------------------------------------------------------------------------------------ sender
// ONE body per kernel instantiation (a per-length dispatch over several unrolled bodies measured slower):
// V = kVMax float4 groups per thread, groups 0..min(V, kVReg)-1 in registers, kVReg..V-1 staged in SMEM. ve = the row's group
// count (CTA-uniform: one CTA = one row); every phase skips groups k >= ve with a uniform branch (they hold no valid key: for
// k >= ve every thread's index k kT + tid >= n4, so their loads are predicated off, their maxima are -inf, their filter bits
// and counts are 0: skipping them changes no value). v0 / v1 = groups 0 / 1, issued before the local length was known
// (bounded by the readable width only; PF2).
template <int V>
__device__ __forceinline__ void send_body(const Params& p, const float4* src, int n, int n4, int ve, float4 v0, float4 v1,
                                          uint32_t ctag, SendSmem& sm, float4* stash, bool too_long) {
  constexpr int VR = V <= kVReg ? V : kVReg;
  constexpr int VS = V - VR;
  constexpr int NP = (V + 3) / 4;
  static_assert(V >= 2 && V <= kVMaxAll && NP <= kVMaxAll / 4, "V");
  using Bits = typename std::conditional<(V > 8), uint64_t, uint32_t>::type;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const float4 ninf = make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
  // ---- S1: every load issued before any use; elements at or past n are masked to -inf below ----
  if constexpr (VS > 0) {
    if (ve > VR) {  // uniform
#pragma unroll
      for (int k = 0; k < VS; ++k) {
        const int i = (VR + k) * kT + tid;
        if (i < n4) cp_async16(stash + k * kT + tid, src + i);
      }
    }
  }
  float4 v[VR];
  v[0] = v0;
  v[1] = v1;
#pragma unroll
  for (int k = 2; k < VR; ++k) {
    const int i = k * kT + tid;
    v[k] = i < n4 ? ld_f4(src + i) : ninf;  // i < n4 implies k < ve
  }
  if constexpr (VS > 0) {
    if (ve > VR) cp_async_wait_all();  // uniform; this thread's own copies (it reads back only its own entries)
  }
  float m[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
#pragma unroll
  for (int k = 0; k < V; ++k) {
    if (k >= ve) break;  // uniform
    const int i = k * kT + tid;
    float4 x;
    if (k < VR) {
      x = v[k < VR ? k : 0];
    } else {
      x = i < n4 ? stash[(k < VR ? 0 : k - VR) * kT + tid] : ninf;
    }
    if (4 * i + 3 >= n) {  // the one group that crosses n (and groups past it)
      if (4 * i + 0 >= n) x.x = -INFINITY;
      if (4 * i + 1 >= n) x.y = -INFINITY;
      if (4 * i + 2 >= n) x.z = -INFINITY;
      x.w = -INFINITY;
    }
    if (k < VR) {
      v[k < VR ? k : 0] = x;
    } else {
      stash[(k < VR ? 0 : k - VR) * kT + tid] = x;
    }
    m[0] = fmaxf(m[0], x.x);
    m[1] = fmaxf(m[1], x.y);
    m[2] = fmaxf(m[2], x.z);
    m[3] = fmaxf(m[3], x.w);
  }
  // ---- S2a: lo / hi / count of the finite maxima ----
  {
    int olo = 0x7FFFFFFF, ohi = static_cast<int>(0x80000000u), nv = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j)
      if (m[j] > -INFINITY) {
        olo = min(olo, f2o(m[j]));
        ohi = max(ohi, f2o(m[j]));
        ++nv;
      }
    olo = __reduce_min_sync(kFull, olo);
    ohi = __reduce_max_sync(kFull, ohi);
    nv = static_cast<int>(__reduce_add_sync(kFull, static_cast<unsigned>(nv)));
    if (lane == 0) {
      sm.lo[warp] = olo;
      sm.hi[warp] = ohi;
      sm.nv[warp] = nv;
    }
  }
  __syncthreads();
  const int nv = static_cast<int>(__reduce_add_sync(kFull, static_cast<unsigned>(sm.nv[lane])));
  const float lo = o2f(__reduce_min_sync(kFull, sm.lo[lane]));
  const float hi = o2f(__reduce_max_sync(kFull, sm.hi[lane]));
  // ---- S2b: tau ----
  float tau = -INFINITY;  // nv <= rank: send every valid key (complete)
  if (nv > kTauRank && hi > lo) {
    const float scale = __fdiv_rn(static_cast<float>(kSBins), __fsub_rn(hi, lo));
#pragma unroll
    for (int j = 0; j < 4; ++j)
      if (m[j] > -INFINITY) {
        const int b = min(sbin(m[j], lo, scale), kSBins - 1);
        atomicAdd(&sm.hist[b], 1);
        atomicAdd(&sm.coarse[b >> 5], 1);
      }
    __syncthreads();
    const int2 cq = *reinterpret_cast<const int2*>(&sm.coarse[kSCo - 2 - 2 * lane]);
    const int a0 = cq.y, a1 = cq.x;
    const int cc = warp_incl(a0 + a1, lane);
    const int ls = __ffs(__ballot_sync(kFull, cc >= kTauRank)) - 1;  // exists: total = nv > kTauRank
    int r0 = __shfl_sync(kFull, cc - a0 - a1, ls);
    const int a0s = __shfl_sync(kFull, a0, ls);
    int cst = kSCo - 1 - 2 * ls;
    if (r0 + a0s < kTauRank) {
      r0 += a0s;
      --cst;
    }
    const int fv = sm.hist[32 * cst + 31 - lane];
    const int fc = warp_incl(fv, lane) + r0;
    const int lf = __ffs(__ballot_sync(kFull, fc >= kTauRank)) - 1;
    const int bstar = 32 * cst + 31 - lf;
    const float M = fmaxf(fabsf(lo), fabsf(hi));
    const float tk = __fadd_rn(lo, __fdiv_rn(static_cast<float>(bstar), scale));
    tau = __fsub_rn(tk, __fmul_rn(M, 0x1p-19f));
  } else if (nv > kTauRank) {
    tau = lo;  // hi == lo: every finite maximum equals lo
  }
  const bool complete = !(tau > -INFINITY);
  // ---- S3: filter (values stay in registers / own SMEM entries), ordered offsets ----
  const float tc = complete ? -3.4028234663852886e38f : tau;
  Bits bits = 0;
  uint32_t P[NP], I[NP];
#pragma unroll
  for (int w = 0; w < NP; ++w) P[w] = 0;
#pragma unroll
  for (int k = 0; k < V; ++k) {
    if (k >= ve) break;  // uniform
    const float4 x = k < VR ? v[k < VR ? k : 0] : stash[(k < VR ? 0 : k - VR) * kT + tid];
    unsigned b4 = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) b4 |= static_cast<unsigned>(comp(x, j) >= tc) << j;
    bits |= static_cast<Bits>(b4) << (4 * k);
    P[k >> 2] |= static_cast<uint32_t>(__popc(b4)) << (8 * (k & 3));
  }
  const int nw = (ve + 3) >> 2;  // uniform: words past it hold only skipped groups (P = I = Wd = 0)
#pragma unroll
  for (int w = 0; w < NP; ++w) I[w] = P[w];
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
#pragma unroll
    for (int w = 0; w < NP; ++w) {
      if (w >= nw) break;  // uniform
      const uint32_t a = __shfl_up_sync(kFull, I[w], o);
      if (lane >= o) I[w] += a;
    }
  }
  if (lane == 31) {
#pragma unroll
    for (int w = 0; w < NP; ++w)
      if (w < nw) sm.t[w][warp] = I[w];
  }
  __syncthreads();
  uint32_t Wd[NP];
  int tsum = 0;
#pragma unroll
  for (int w = 0; w < NP; ++w) {
    Wd[w] = w < nw ? sm.t[w][lane] : 0u;
    tsum += bytesum(Wd[w]);
  }
  const int N = static_cast<int>(__reduce_add_sync(kFull, static_cast<unsigned>(tsum)));
  const bool ovf = N > kCap || too_long;  // too_long: the row exceeds this kernel's groups -> every rank takes the exact fallback
  // ---- S4: header, records ----
  const int row = p.row0 + static_cast<int>(blockIdx.x), rank = p.rank + static_cast<int>(blockIdx.y), parity = p.ordinal & 1;
  const bool staged = p.mc == 0 && !(p.test_mode & 1);  // uniform: the unicast push (serving, tests)
  if (tid == 0) {
    const uint32_t hdr = static_cast<uint32_t>(min(N, 0x3FFFFFFF)) | (ovf ? 0x80000000u : 0u) | (complete ? 0x40000000u : 0u);
    const uint64_t h0 = rec(hdr, ctag), h1 = rec(__float_as_uint(tau), ctag);
    const uint64_t h2 = rec(__float_as_uint(N > 0 ? hi : -INFINITY), ctag), h3 = rec(0u, ctag);
    if (staged) {
      sm.stg[kCap][0] = h0;
      sm.stg[kCap][1] = h1;
      sm.stg[kCap + 1][0] = h2;
      sm.stg[kCap + 1][1] = h3;
    } else {
      const int64_t at = cand_index(parity, row, rank, kCap, p.cap);
      push2(p, rank, at, h0, h1);
      push2(p, rank, at + 2, h2, h3);
    }
  }
  if (!ovf) {  // uniform
    const int64_t at0 = cand_index(parity, row, rank, 0, p.cap);
    int Nb = 0;
#pragma unroll
    for (int k = 0; k < V; ++k) {
      if (k >= ve) break;  // uniform
      const int sh = 8 * (k & 3), w = k >> 2;
      const unsigned c = (Wd[w] >> sh) & 0xFFu;
      const int ex = static_cast<int>(__reduce_add_sync(kFull, lane < warp ? c : 0u));
      const int tot = static_cast<int>(__reduce_add_sync(kFull, c));
      const unsigned b4 = static_cast<unsigned>(bits >> (4 * k)) & 0xFu;
      if (b4 != 0u) {
        const int base = Nb + ex + static_cast<int>(((I[w] >> sh) & 0xFFu) - ((P[w] >> sh) & 0xFFu));
        const float4 x = k < VR ? v[k < VR ? k : 0] : stash[(k < VR ? 0 : k - VR) * kT + tid];
        const int i = k * kT + tid;
#pragma unroll
        for (int j = 0; j < 4; ++j)
          if (b4 >> j & 1u) {
            const int off = base + __popc(b4 & ((1u << j) - 1u));
            const int pos = gpos(4 * i + j, rank);
            const uint64_t r0 = rec(__float_as_uint(comp(x, j)), ctag), r1 = rec(static_cast<uint32_t>(pos), ctag);
            if (staged) {
              sm.stg[off][0] = r0;
              sm.stg[off][1] = r1;
            } else {
              push2(p, rank, at0 + 2 * off, r0, r1);
            }
          }
      }
      Nb += tot;
    }
  }
  if (staged) {  // uniform
    __syncthreads();  // every staged record / header element written (CTA scope, SMEM)
    unicast_push(p, rank, cand_index(parity, row, rank, 0, p.cap), N, ovf, sm);
  }
}

// ------------------------------------------------------------------------------------------------ fallback
// Exact selection of a flagged row (identical decision on every rank). Every thread of the CTA calls it.
__device__ __noinline__ void fallback_row(const Params& p, int want, FbSmem& fb) {
  const int tid = threadIdx.x;
  const int row = p.row0 + static_cast<int>(blockIdx.x), rank = p.rank + static_cast<int>(blockIdx.y), parity = p.ordinal & 1;
  const uint32_t ftag = make_tag(ld_gpu_u32(p.epoch), p.ordinal, kKindFull, p.layers);
  const int nloc = min(ld_gpu_i32(p.lcausal + rank * p.lc_rank_stride + row), p.ncols);
  const float* const srow = p.scores + rank * p.score_rank_stride + static_cast<int64_t>(row) * p.score_stride;
  int32_t* const o = p.out + rank * p.out_rank_stride + static_cast<int64_t>(row) * kTopK;
  const int32_t* const trow = p.table + static_cast<int64_t>(row / p.rows_per_request) * p.table_stride;
  const uint64_t own = p.peers[rank];
  uint64_t* const keys = p.scratch + rank * p.scratch_rank_stride + static_cast<int64_t>(row) * kW * kTopK;
  // (1) this rank's exact local top-kl by the 64-bit key, in local-index order -> its own FULL slot
  const int kl = min(nloc, kTopK);
  auto lkey = [&](int l) { return make_key(__float_as_uint(canon(ld_f(srow + l))), gpos(l, rank)); };
  const uint64_t tl = select_kth(
      [&](auto f) {
        for (int l = tid; l < nloc; l += kT) f(lkey(l));
      },
      kl, fb);
  const int ch = (nloc + kT - 1) / kT, l0 = min(tid * ch, nloc), l1 = min(l0 + ch, nloc);
  int mine = 0;
  for (int l = l0; l < l1; ++l) mine += tl != 0 && lkey(l) >= tl;
  int tot;
  int at = block_exclusive_sum(mine, tot, fb);
  for (int l = l0; l < l1; ++l) {
    const float x = canon(ld_f(srow + l));
    const int pos = gpos(l, rank);
    if (tl != 0 && make_key(__float_as_uint(x), pos) >= tl) {
      const bool keep = true;
      if (at < kTopK)
        st_rec2(own + 8ull * full_index(parity, row, at, p.cap), rec(keep ? __float_as_uint(x) : 0u, ftag),
                rec(keep ? static_cast<uint32_t>(pos) : kEmptyPos, ftag));
      ++at;
    }
  }
  for (int e = tot + tid; e < kTopK; e += kT)
    st_rec2(own + 8ull * full_index(parity, row, e, p.cap), rec(0u, ftag), rec(kEmptyPos, ftag));
  // (2) the 16 FULL lists (2 entries per thread per source) -> union keys in scratch, segment q = source q
  const uint64_t start = gtime();
  bool dead = false;
#pragma unroll 1
  for (int qb = 0; qb < kW; qb += 4) {
    uint64_t a[8], b[8];
    uint32_t pending = 0xFFu, spins = 0;
    while (true) {
#pragma unroll
      for (int j = 0; j < 8; ++j)
        if (pending >> j & 1u)
          ld_rec2(reinterpret_cast<const uint64_t*>(p.peers[qb + (j >> 1)]) + full_index(parity, row, tid + (j & 1) * kT, p.cap),
                  a[j], b[j]);
#pragma unroll
      for (int j = 0; j < 8; ++j)
        if ((pending >> j & 1u) && tagged(a[j], b[j], ftag)) pending &= ~(1u << j);
      if (pending == 0u || dead) break;
      if ((++spins & 63u) == 0 && gtime() - start > p.deadline_ns && expired(p.test_mode, &fb.dl, dead)) break;
    }
#pragma unroll
    for (int j = 0; j < 8; ++j)
      keys[(qb + (j >> 1)) * kTopK + tid + (j & 1) * kT] = make_key(lo32(a[j]), static_cast<int32_t>(lo32(b[j])));
  }
  __syncthreads();  // the union in scratch is complete (CTA-scope order through the barrier)
  // (3) t* = the want-th key of the union (it holds every global top-want key: each is in its source's local top-2048)
  constexpr int kU = kW * kTopK / kT;  // 32 union entries per thread
  const uint64_t ts = select_kth(
      [&](auto f) {
#pragma unroll 4
        for (int e = tid; e < kW * kTopK; e += kT) f(keys[e]);
      },
      want, fb);
  // (4) emission: contiguous chunks of the union (source-major, local-index = position order inside a source)
  mine = 0;
#pragma unroll 4
  for (int e = tid * kU; e < tid * kU + kU; ++e) {
    const uint64_t k = keys[e];
    mine += k != 0 && k >= ts;
  }
  at = block_exclusive_sum(mine, tot, fb);
#pragma unroll 4
  for (int e = tid * kU; e < tid * kU + kU; ++e) {
    const uint64_t k = keys[e];
    if (k != 0 && k >= ts) {
      const uint32_t pos = key_pos(k);
      if (at < kTopK) o[at] = trow[pos / kPage] * kPage + static_cast<int32_t>(pos % kPage);
      ++at;
    }
  }
  for (int e = tot + tid; e < kTopK; e += kT) o[e] = -1;
}

// ------------------------------------------------------------------------------------------------ receiver
__device__ __forceinline__ void recv_body(const Params& p, uint32_t ctag, RecvSmem& rs, FbSmem& fb) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int row = p.row0 + static_cast<int>(blockIdx.x), rank = p.rank + static_cast<int>(blockIdx.y), parity = p.ordinal & 1;
  const int s = warp >> 1, h = warp & 1;
  const uint64_t* const base = reinterpret_cast<const uint64_t*>(p.peers[rank]) + cand_index(parity, row, 0, 0, p.cap);
  const uint64_t* const sb = base + static_cast<int64_t>(s) * kSlots * 2;
  const int want = min(ld_gpu_i32(p.causal + row), kTopK);
  if (want <= 0) {  // uniform; no row has an empty causal prefix in serving (nothing polled, nothing to order)
    int32_t* const o = p.out + rank * p.out_rank_stride + static_cast<int64_t>(row) * kTopK;
    for (int i = tid; i < kTopK; i += kT) o[i] = -1;
    if (tid == 0) p.fallback[rank * p.fb_rank_stride + row] = 0;
    return;
  }
  const uint64_t start = gtime();
  // ---- R1: one batch of polls (headers of all 16 sources on lanes 0..15, this warp's first slots) ----
  uint64_t ha = 0, hb = 0, hc = 0, hd = 0;
  const uint64_t* const hp = base + (static_cast<int64_t>(lane & (kW - 1)) * kSlots + kCap) * 2;
  if (lane < kW) {
    ld_rec2(hp, ha, hb);
    ld_rec2(hp + 2, hc, hd);
  }
  uint64_t ra[kPer], rb[kPer];
#pragma unroll
  for (int r = 0; r < kPer; ++r) {
    const int e = h * kHalf + r * 32 + lane;
    ra[r] = 0;
    rb[r] = 0;
    if (e < kSpec) ld_rec2(sb + 2 * e, ra[r], rb[r]);
  }
  bool dead = false;
  if (lane < kW) {
    for (uint32_t it = 1; !(tagged(ha, hb, ctag) && tagged(hc, hd, ctag)); ++it) {
      if ((it & 63u) == 0 && gtime() - start > p.deadline_ns && expired(p.test_mode, &fb.dl, dead)) {
        ha = hb = hc = hd = 0;
        break;
      }
      ld_rec2(hp, ha, hb);
      ld_rec2(hp + 2, hc, hd);
    }
  }
  const uint32_t hd0 = static_cast<uint32_t>(ha);
  const bool src_ovf = lane < kW && (hd0 & 0x80000000u);
  const bool src_cpl = (hd0 >> 30) & 1u;
  int nsrc = lane < kW && !src_ovf ? static_cast<int>(hd0 & 0x3FFFFFFFu) : 0;
  const float tau_s = lane < kW ? __uint_as_float(static_cast<uint32_t>(hb)) : -INFINITY;
  const float hi_s = lane < kW && nsrc > 0 ? __uint_as_float(static_cast<uint32_t>(hc)) : -INFINITY;
  const bool any_ovf = __ballot_sync(kFull, src_ovf) != 0u;
  const float lo0 = o2f(__reduce_max_sync(kFull, f2o(nsrc > 0 ? tau_s : -INFINITY)));
  const float hi = o2f(__reduce_max_sync(kFull, f2o(hi_s)));
  const int ns = __shfl_sync(kFull, nsrc, s);
  bool valid[kPer];
#pragma unroll
  for (int r = 0; r < kPer; ++r) {
    const int e = h * kHalf + r * 32 + lane;
    valid[r] = e < ns;
    if (valid[r] && e >= kSpec) ld_rec2(sb + 2 * e, ra[r], rb[r]);
  }
#pragma unroll
  for (int r = 0; r < kPer; ++r) {
    if (!valid[r]) continue;
    const int e = h * kHalf + r * 32 + lane;
    for (uint32_t it = 1; !dead && !tagged(ra[r], rb[r], ctag); ++it) {
      if ((it & 63u) == 0 && gtime() - start > p.deadline_ns && expired(p.test_mode, &fb.dl, dead)) break;
      ld_rec2(sb + 2 * e, ra[r], rb[r]);
    }
  }
  float sc[kPer];
  int sp[kPer];
#pragma unroll
  for (int r = 0; r < kPer; ++r) {
    sc[r] = __uint_as_float(static_cast<uint32_t>(ra[r]));
    sp[r] = static_cast<int>(static_cast<uint32_t>(rb[r]));
  }
  // ---- R2: value-linear fine + coarse histograms over [lo, hi] (second pass with lo = the union min if needed) ----
  float lo = lo0;
  bool slow = !(lo0 > -INFINITY);
  bool short_union = false;
  int bin[kPer];
  int bstar = -1, above = 0;
  for (int pass = 0; pass < 2; ++pass) {
    if (slow) {  // uniform: every warp derived lo0 from the same headers, and the counts below are block-wide
      int om = 0x7FFFFFFF;
#pragma unroll
      for (int r = 0; r < kPer; ++r)
        if (valid[r]) om = min(om, f2o(sc[r]));
      om = __reduce_min_sync(kFull, om);
      if (lane == 0) rs.wmin[warp] = om;
      __syncthreads();  // also: every warp has finished reading pass 0's histograms before they are cleared below
      lo = o2f(__reduce_min_sync(kFull, rs.wmin[lane]));
      if (pass == 1) {
        for (int i = tid; i < kRBins; i += kT) rs.hist[i] = 0;
        if (tid < kRCo) rs.coarse[tid] = 0;
        __syncthreads();
      }
    }
    const float scale = hi > lo ? __fdiv_rn(static_cast<float>(kRBins), __fsub_rn(hi, lo)) : 0.f;
#pragma unroll
    for (int r = 0; r < kPer; ++r) {
      bin[r] = -1;
      if (valid[r] && sc[r] >= lo) {
        const int b = min(sbin(sc[r], lo, scale), kRBins - 1);
        bin[r] = b;
        atomicAdd(&rs.hist[b], 1);
        atomicAdd(&rs.coarse[b >> 5], 1);
      }
    }
    __syncthreads();
    // E-R fence, form (d), MULTICAST push only and only when built with K31_TSIM_ALIAS_FENCE=1: every thread's last CAND poll is
    // before this barrier and no CAND element is read after it. The unicast push is all strong .sys ops: no alias proxy.
    if (K31_TSIM_ALIAS_FENCE && pass == 0 && tid == 0 && p.mc != 0) fence_alias();
    const int4 cq = *reinterpret_cast<const int4*>(&rs.coarse[kRCo - 4 - 4 * lane]);
    const int a0 = cq.w, a1 = cq.z, a2 = cq.y, a3 = cq.x;
    const int cs = a0 + a1 + a2 + a3;
    const int cc = warp_incl(cs, lane);
    const unsigned bw = __ballot_sync(kFull, cc >= want);
    if (bw == 0u) {  // fewer than want keys >= lo (uniform)
      if (!slow) {
        slow = true;
        continue;
      }
      short_union = true;  // lo was already the union min: the union holds fewer than want keys
      above = __shfl_sync(kFull, cc, 31);
      bstar = -1;
      break;
    }
    const int ls = __ffs(bw) - 1;
    int r0 = __shfl_sync(kFull, cc - cs, ls);
    const int b0 = __shfl_sync(kFull, a0, ls), b1 = __shfl_sync(kFull, a1, ls), b2 = __shfl_sync(kFull, a2, ls);
    int cst = kRCo - 1 - 4 * ls;
    if (r0 + b0 < want) {
      r0 += b0;
      --cst;
      if (r0 + b1 < want) {
        r0 += b1;
        --cst;
        if (r0 + b2 < want) {
          r0 += b2;
          --cst;
        }
      }
    }
    const int fv = rs.hist[32 * cst + 31 - lane];
    const int fc = warp_incl(fv, lane) + r0;
    const int lf = __ffs(__ballot_sync(kFull, fc >= want)) - 1;
    above = __shfl_sync(kFull, fc - fv, lf);
    bstar = 32 * cst + 31 - lf;
    break;
  }
  // ---- per-key state after b* (2 bits per round: 1 above b*, 2 in the boundary bin) ----
  unsigned st = 0;
#pragma unroll
  for (int r = 0; r < kPer; ++r) {
    const bool ab = bin[r] >= 0 && bin[r] > bstar;
    const bool ti = bstar >= 0 && bin[r] == bstar;
    st |= (ab ? 1u : 0u) << (2 * r) | (ti ? 2u : 0u) << (2 * r);
  }
  // ---- KV-slot gathers for the keys that can be emitted (consumed at emission only) ----
  const int32_t* const trow = p.table + static_cast<int64_t>(row / p.rows_per_request) * p.table_stride;
  int tb[kPer];
#pragma unroll
  for (int r = 0; r < kPer; ++r) tb[r] = trow[(st >> (2 * r)) & 3u ? sp[r] >> 6 : 0];
  // ---- R3: tie list (boundary bin) + per-warp counts of the keys above b*, one barrier ----
  int cnt = 0;
  unsigned tixp = 0;  // 5 bits per round: the key's tie-list index (fast path, < 32)
#pragma unroll
  for (int r = 0; r < kPer; ++r) {
    cnt += __popc(__ballot_sync(kFull, (st >> (2 * r)) & 1u));
    if ((st >> (2 * r)) & 2u) {
      const int k = atomicAdd(&rs.m, 1);
      tixp |= static_cast<unsigned>(k & 31) << (5 * r);
      if (k < kTie) rs.tie[k] = make_int4(__float_as_int(sc[r]), sp[r], warp, 0);
    }
  }
  if (lane == 0) rs.wc[warp] = cnt;
  __syncthreads();
  const int mt = rs.m;
  const int rstar = want - above - 1;  // 0-based rank of t* inside the boundary bin
  const int rsel = rstar;
  int tie_before = 0, tie_total = 0;
  float tstar = INFINITY;  // score of t* (certificate)
  bool tsl = false;        // fast path: lane j holds tie entry j; selected?
  if (mt <= 32) {          // uniform; every warp ranks the boundary-bin entries lane-resident (full key: score desc, position asc)
    const int4 te = lane < mt ? rs.tie[lane] : make_int4(0, 0, kWarps, 0);
    const float tsc = __int_as_float(te.x);
    int rank_ = 0;
    for (int j = 0; j < mt; ++j) {
      const float sj = __shfl_sync(kFull, tsc, j);
      const int pj = __shfl_sync(kFull, te.y, j);
      rank_ += (sj > tsc) || (sj == tsc && pj < te.y);
    }
    tsl = lane < mt && rank_ <= rsel;
    tie_before = __popc(__ballot_sync(kFull, tsl && te.z < warp));
    tie_total = __popc(__ballot_sync(kFull, tsl));
    const unsigned bt = __ballot_sync(kFull, lane < mt && rank_ == rstar);
    if (bt != 0u) tstar = __shfl_sync(kFull, tsc, __ffs(bt) - 1);
  } else {  // rare (never on the real rows): entries ranked strided over the warps; counts after one more barrier
    const int mm = min(mt, kTie);
    for (int j = tid; j < mm; j += kT) {
      const int4 t = rs.tie[j];
      const float tj = __int_as_float(t.x);
      int rank_ = 0;
      for (int i = 0; i < mm; ++i) {
        const int4 u = rs.tie[i];
        const float si = __int_as_float(u.x);
        rank_ += (si > tj) || (si == tj && u.y < t.y);
      }
      rs.tsel[j] = rank_ <= rsel ? 1 : 0;
      if (rank_ == rstar) rs.ts = tj;
    }
    __syncthreads();
    int cb = 0, ct = 0;
    for (int j = lane; j < mm; j += 32) {
      const int sj = rs.tsel[j];
      ct += sj;
      cb += sj && rs.tie[j].z < warp;
    }
    tie_before = static_cast<int>(__reduce_add_sync(kFull, static_cast<unsigned>(cb)));
    tie_total = static_cast<int>(__reduce_add_sync(kFull, static_cast<unsigned>(ct)));
    tstar = rs.ts;
  }
  const int wcl = rs.wc[lane];
  const int total = static_cast<int>(__reduce_add_sync(kFull, static_cast<unsigned>(wcl))) + tie_total;
  // ---- flags (uniform: every warp derives them from the same headers and SMEM) ----
  const bool cert_bad = __ballot_sync(kFull, slow && lane < kW && nsrc > 0 && !src_cpl && tau_s > tstar) != 0u;
  const int fl = (any_ovf ? 1 : 0) | (cert_bad ? 4 : 0) | (short_union ? 32 : 0) | (mt > kTie ? 64 : 0) |
                 (total != want ? 128 : 0);
  const bool fast = fl == 0;
  if (fast) {
    // ---- R4: emission (source-major, position-ascending) ----
    int32_t* const o = p.out + rank * p.out_rank_stride + static_cast<int64_t>(row) * kTopK;
    int off = static_cast<int>(__reduce_add_sync(kFull, lane < warp ? static_cast<unsigned>(wcl) : 0u)) + tie_before;
    const unsigned lt = lanemask_lt();
#pragma unroll
    for (int r = 0; r < kPer; ++r) {
      const unsigned sr = (st >> (2 * r)) & 3u;
      bool sel = sr == 1u;
      if (__ballot_sync(kFull, sr == 2u) != 0u) {  // uniform
        bool ts = false;
        if (mt <= 32) {
          ts = __shfl_sync(kFull, tsl ? 1 : 0, (tixp >> (5 * r)) & 31u) != 0;
        } else if (sr == 2u) {  // rare path: find this key's entry by its (unique) global position
          const int mm = min(mt, kTie);
          for (int j = 0; j < mm; ++j)
            if (rs.tie[j].y == sp[r]) {
              ts = rs.tsel[j] != 0;
              break;
            }
        }
        if (sr == 2u) sel = ts;
      }
      const unsigned bal = __ballot_sync(kFull, sel);
      const int nr = __popc(bal);
      if (sel) {
        const int k = __popc(bal & lt);
        const int at = off + k;
        if (at < kTopK) o[at] = tb[r] * kPage + (sp[r] & (kPage - 1));
      }
      off += nr;
    }
    for (int i = total + tid; i < kTopK; i += kT) o[i] = -1;
  } else {
    fallback_row(p, want, fb);
  }
  int fl_out = fl;
  if (p.test_mode & 2) {  // uniform
    __syncthreads();
    fl_out |= fb.dl ? 8 : 0;
  }
  if (tid == 0) p.fallback[(p.rank + static_cast<int>(blockIdx.y)) * p.fb_rank_stride + p.row0 + static_cast<int>(blockIdx.x)] = fl_out;
}

// ------------------------------------------------------------------------------------------------ kernel
template <int kVMax>
__global__ void __launch_bounds__(kT, 1) tsim_kernel(const __grid_constant__ Params p) {
  static_assert(kVMax == 8 || kVMax == 16, "kVMax");
  extern __shared__ __align__(16) unsigned char smem_raw[];
  Smem& s = *reinterpret_cast<Smem*>(smem_raw);
  float4* const stash = reinterpret_cast<float4*>(smem_raw + kSmemHead);
  const int tid = threadIdx.x;
  const int row = p.row0 + static_cast<int>(blockIdx.x), rank = p.rank + static_cast<int>(blockIdx.y);
  for (int i = tid; i < kSBins; i += kT) s.snd.hist[i] = 0;
  if (tid < kSCo) s.snd.coarse[tid] = 0;
  for (int i = tid; i < kRBins; i += kT) s.rcv.hist[i] = 0;
  if (tid < kRCo) s.rcv.coarse[tid] = 0;
  if (tid == 0) {
    s.rcv.m = 0;
    s.rcv.ts = INFINITY;
    s.fb.dl = 0;
  }
  __syncthreads();
  pdl_wait();
  const float4* const src =
      reinterpret_cast<const float4*>(p.scores + rank * p.score_rank_stride + static_cast<int64_t>(row) * p.score_stride);
  const int n4cols = p.ncols >> 2;
  const float4 ninf = make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
  const float4 v0 = tid < n4cols ? ld_f4(src + tid) : ninf;            // groups 0 / 1: no wait for nloc
  const float4 v1 = tid + kT < n4cols ? ld_f4(src + tid + kT) : ninf;  // (masked to -inf past n in the sender)
  const uint32_t epoch = ld_gpu_u32(p.epoch);
  const int n = min(ld_gpu_i32(p.lcausal + rank * p.lc_rank_stride + row), p.ncols);
  const int n4 = min((n + 3) >> 2, n4cols);
  const int vall = (n4 + kT - 1) / kT;
  const bool too_long = vall > kVMax;  // uniform; only when the table is wider than kVMax groups (ncols > 65,536 for kernel<16>)
  const int veff = min(vall, kVMax);
  const uint32_t ctag = make_tag(epoch, p.ordinal, kKindCand, p.layers);
  // one sender body for every row length (ve = the row's own group count, CTA-uniform)
  if constexpr (kVMax > kVReg) {  // rows of <= 8 groups (the served band below ~524K) run the 8-group body (DISP)
    if (veff <= kVReg) {
      send_body<kVReg>(p, src, n, n4, veff, v0, v1, ctag, s.snd, stash, too_long);
    } else {
      send_body<kVMax>(p, src, n, n4, veff, v0, v1, ctag, s.snd, stash, too_long);
    }
  } else {
    send_body<kVMax>(p, src, n, n4, veff, v0, v1, ctag, s.snd, stash, too_long);
  }
  __syncthreads();  // (measured TF3 form) the sender's SMEM phases end before the receiver's
  recv_body(p, ctag, s.rcv, s.fb);
  pdl_trigger();
}

}  // namespace k31_tsim

#ifdef K31_TSIM_PTX_INSTANTIATE
template __global__ void k31_tsim::tsim_kernel<8>(const __grid_constant__ k31_tsim::Params);
template __global__ void k31_tsim::tsim_kernel<16>(const __grid_constant__ k31_tsim::Params);
#endif

#ifndef K31_TSIM_DEVICE_ONLY
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

namespace k31_tsim {

struct Binding {
  // meta (CPU int64 [8 + 16]): world (16), rank, ordinal, layers, cap, deadline_ns, mc (multicast base, 0 = unicast push),
  // bench_self_push (benchmarks only: 1 = one store per record into the own region, the multicast cost model on one
  // GPU; 0 in serving and tests), then the 16 regions' unicast bases.
  static const int64_t* meta_ptr(const tvm::ffi::TensorView meta) {
    using namespace sglang::host;
    const int64_t* m = static_cast<const int64_t*>(meta.data_ptr());
    RuntimeCheck(meta.device().device_type == kDLCPU && meta.size(0) == kMeta + kW, "TSIM meta: CPU int64 [8 + 16]");
    RuntimeCheck(m[0] == kW && m[1] >= 0 && m[1] < kW, "TSIM meta: world / rank");
    return m;
  }
  static void check_i32(const tvm::ffi::TensorView t, const char* what) {
    using namespace sglang::host;
    RuntimeCheck(t.dtype().code == kDLInt && t.dtype().bits == 32 && t.device().device_type == kDLCUDA, what,
                 ": CUDA int32");
  }
  template <int kVMax>
  static void set_smem() {
    static bool done = false;
    if (done) return;
    sglang::host::RuntimeDeviceCheck(cudaFuncSetAttribute(tsim_kernel<kVMax>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                          static_cast<int>(smem_bytes(kVMax))));
    done = true;
  }

  // R = 1 (serving: this rank) or 16 (emulation): scores [R, rows, >= ncols] (or [rows, >= ncols] when R = 1),
  // lcausal [R, rows], out [R, rows, 2048], fallback [R, rows], scratch [R, rows, 16 * 2048].
  static Params params(const int64_t* m, const tvm::ffi::TensorView epoch, const tvm::ffi::TensorView scores,
                       const tvm::ffi::TensorView lcausal, const tvm::ffi::TensorView causal,
                       const tvm::ffi::TensorView table, const tvm::ffi::TensorView out,
                       const tvm::ffi::TensorView fallback, const tvm::ffi::TensorView scratch, int64_t rows_per_request,
                       int64_t ncols, int R, int rows) {
    using namespace sglang::host;
    check_i32(lcausal, "lcausal");
    check_i32(causal, "causal");
    check_i32(table, "table");
    check_i32(out, "out");
    check_i32(fallback, "fallback");
    RuntimeCheck(scores.dtype().code == kDLFloat && scores.dtype().bits == 32, "scores: float32");
    RuntimeCheck(scratch.dtype().bits == 64, "scratch: int64");
    const int sd = scores.ndim();
    RuntimeCheck((sd == 2 && R == 1) || (sd == 3 && scores.size(0) == R), "scores [R, rows, cols]");
    RuntimeCheck(scores.size(sd - 2) >= rows && scores.stride(sd - 1) == 1 && scores.size(sd - 1) >= ncols, "scores rows / cols");
    RuntimeCheck(ncols % 4 == 0 && ncols >= 4 && scores.stride(sd - 2) % 4 == 0 &&
                     reinterpret_cast<uintptr_t>(scores.data_ptr()) % 16 == 0,
                 "scores: 16-B aligned rows, ncols % 4 == 0");
    // ncols may exceed 16 groups (a 16,400-page table at context 1,048,576): kernel<16> runs; a row longer than 16 groups takes the
    // exact fallback (sender header ovf), so no row is refused or truncated.
    RuntimeCheck(lcausal.size(lcausal.ndim() - 1) >= rows && lcausal.stride(lcausal.ndim() - 1) == 1, "lcausal");
    RuntimeCheck(causal.ndim() == 1 && causal.size(0) >= rows, "causal [rows]");
    RuntimeCheck(table.ndim() == 2 && table.stride(1) == 1 && table.size(0) * rows_per_request >= rows,
                 "table [rows / rows_per_request, pages]");
    RuntimeCheck(out.size(out.ndim() - 1) == kTopK && out.stride(out.ndim() - 1) == 1 &&
                     out.stride(out.ndim() - 2) == kTopK && out.size(out.ndim() - 2) >= rows,
                 "out [R, rows, 2048] contiguous rows");
    RuntimeCheck(fallback.size(fallback.ndim() - 1) >= rows && fallback.stride(fallback.ndim() - 1) == 1, "fallback");
    RuntimeCheck(scratch.size(scratch.ndim() - 1) == kW * kTopK && scratch.stride(scratch.ndim() - 1) == 1 &&
                     scratch.stride(scratch.ndim() - 2) == kW * kTopK && scratch.size(scratch.ndim() - 2) >= rows,
                 "scratch [R, rows, 16 * 2048]");
    RuntimeCheck(rows <= m[4], "TSIM: rows exceed the region capacity");
    Params p{};
    p.scores = static_cast<const float*>(scores.data_ptr());
    p.score_stride = scores.stride(sd - 2);
    p.score_rank_stride = sd == 3 ? scores.stride(0) : 0;
    p.lcausal = static_cast<const int32_t*>(lcausal.data_ptr());
    p.lc_rank_stride = lcausal.ndim() == 2 ? lcausal.stride(0) : 0;
    p.causal = static_cast<const int32_t*>(causal.data_ptr());
    p.table = static_cast<const int32_t*>(table.data_ptr());
    p.table_stride = table.stride(0);
    p.out = static_cast<int32_t*>(out.data_ptr());
    p.out_rank_stride = out.ndim() == 3 ? out.stride(0) : 0;
    p.fallback = static_cast<int32_t*>(fallback.data_ptr());
    p.fb_rank_stride = fallback.ndim() == 2 ? fallback.stride(0) : 0;
    p.scratch = static_cast<uint64_t*>(scratch.data_ptr());
    p.scratch_rank_stride = scratch.ndim() == 3 ? scratch.stride(0) : 0;
    p.epoch = static_cast<const uint32_t*>(epoch.data_ptr());
    for (int i = 0; i < kW; ++i) p.peers[i] = static_cast<uint64_t>(m[kMeta + i]);
    p.mc = static_cast<uint64_t>(m[6]);
    p.deadline_ns = static_cast<uint64_t>(m[5]);
    p.ncols = static_cast<int>(ncols);
    p.rank = static_cast<int>(m[1]);
    p.ordinal = static_cast<int>(m[2]);
    p.layers = static_cast<int>(m[3]);
    p.cap = static_cast<int>(m[4]);
    p.rows_per_request = static_cast<int>(rows_per_request);
    p.row0 = 0;
    p.test_mode = static_cast<int>(m[7] & 3);
    return p;
  }

  // Serving: one launch per DSA layer, grid [rows], PDL. meta[7] (test_mode) must be 0 here (G2V.2 point 3: the deadline traps,
  // fail closed; the no-trap test mode and the bench self-push exist only through bench() / emulate(), which serving never calls).
  static void run(const tvm::ffi::TensorView meta, const tvm::ffi::TensorView epoch, const tvm::ffi::TensorView scores,
                  const tvm::ffi::TensorView lcausal, const tvm::ffi::TensorView causal,
                  const tvm::ffi::TensorView table, const tvm::ffi::TensorView out,
                  const tvm::ffi::TensorView fallback, const tvm::ffi::TensorView scratch, int64_t rows_per_request,
                  int64_t ncols) {
    using namespace sglang::host;
    const int64_t* m = meta_ptr(meta);
    RuntimeCheck(m[7] == 0, "TSIM run (serving): meta[7] test_mode must be 0 (tests / benches use emulate / bench)");
    launch(m, epoch, scores, lcausal, causal, table, out, fallback, scratch, rows_per_request, ncols);
  }
  // Benchmarks only (never called by tsi.py): as run() with meta[7] test_mode bits (1 self-push, 2 no trap).
  static void bench(const tvm::ffi::TensorView meta, const tvm::ffi::TensorView epoch, const tvm::ffi::TensorView scores,
                    const tvm::ffi::TensorView lcausal, const tvm::ffi::TensorView causal,
                    const tvm::ffi::TensorView table, const tvm::ffi::TensorView out,
                    const tvm::ffi::TensorView fallback, const tvm::ffi::TensorView scratch, int64_t rows_per_request,
                    int64_t ncols) {
    launch(meta_ptr(meta), epoch, scores, lcausal, causal, table, out, fallback, scratch, rows_per_request, ncols);
  }
  static void launch(const int64_t* m, const tvm::ffi::TensorView epoch, const tvm::ffi::TensorView scores,
                     const tvm::ffi::TensorView lcausal, const tvm::ffi::TensorView causal,
                     const tvm::ffi::TensorView table, const tvm::ffi::TensorView out,
                     const tvm::ffi::TensorView fallback, const tvm::ffi::TensorView scratch, int64_t rows_per_request,
                     int64_t ncols) {
    using namespace sglang::host;
    const int rows = static_cast<int>(out.size(out.ndim() - 2));
    if (rows == 0) return;
    Params p = params(m, epoch, scores, lcausal, causal, table, out, fallback, scratch, rows_per_request, ncols, 1, rows);
    const int groups = static_cast<int>((ncols / 4 + kT - 1) / kT);
    if (groups <= kVReg) {
      set_smem<8>();
      LaunchKernel(dim3(rows), dim3(kT), out.device(), smem_bytes(8)).enable_pdl(true)(tsim_kernel<8>, p);
    } else {
      set_smem<16>();
      LaunchKernel(dim3(rows), dim3(kT), out.device(), smem_bytes(16)).enable_pdl(true)(tsim_kernel<16>, p);
    }
  }

  // Tests: all 16 ranks on this GPU, grid [nrows, 16] (blockIdx.y = rank), cooperative launch (every CTA resident: the
  // ranks poll each other), unicast pushes into the 16 regions of meta (meta mc must be 0, meta rank 0).
  static void emulate(const tvm::ffi::TensorView meta, const tvm::ffi::TensorView epoch,
                      const tvm::ffi::TensorView scores, const tvm::ffi::TensorView lcausal,
                      const tvm::ffi::TensorView causal, const tvm::ffi::TensorView table,
                      const tvm::ffi::TensorView out, const tvm::ffi::TensorView fallback,
                      const tvm::ffi::TensorView scratch, int64_t rows_per_request, int64_t ncols, int64_t row0,
                      int64_t nrows) {
    using namespace sglang::host;
    const int64_t* m = meta_ptr(meta);
    RuntimeCheck(m[1] == 0 && m[6] == 0 && (m[7] & 1) == 0, "TSIM emulate: meta rank 0, unicast push");
    const int rows = static_cast<int>(out.size(1));
    RuntimeCheck(out.ndim() == 3 && out.size(0) == kW && row0 >= 0 && nrows >= 1 && row0 + nrows <= rows, "emulate rows");
    Params p = params(m, epoch, scores, lcausal, causal, table, out, fallback, scratch, rows_per_request, ncols, kW, rows);
    p.row0 = static_cast<int>(row0);
    p.test_mode |= 2;  // never trap in the emulation
    const int groups = static_cast<int>((ncols / 4 + kT - 1) / kT);
    const bool big = groups > kVReg;
    void (*kernel)(const Params) = big ? tsim_kernel<16> : tsim_kernel<8>;
    const size_t smem = smem_bytes(big ? 16 : 8);
    if (big) set_smem<16>(); else set_smem<8>();
    int dev = 0, sms = 0, per_sm = 0;
    RuntimeDeviceCheck(cudaGetDevice(&dev));
    RuntimeDeviceCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    RuntimeDeviceCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kT, smem));
    RuntimeCheck(static_cast<int64_t>(nrows) * kW <= static_cast<int64_t>(sms) * per_sm, "TSIM emulate: ", nrows * kW,
                 " CTAs exceed ", sms * per_sm, " co-resident");
    cudaLaunchConfig_t cfg{};
    cudaLaunchAttribute at[1];
    cfg.gridDim = dim3(static_cast<unsigned>(nrows), kW);
    cfg.blockDim = dim3(kT);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = LaunchKernel::resolve_device(out.device());
    at[0].id = cudaLaunchAttributeCooperative;
    at[0].val.cooperative = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    RuntimeDeviceCheck(cudaLaunchKernelEx(&cfg, kernel, p));
  }

  static void max_coresident(const tvm::ffi::TensorView out_i32, int64_t big) {  // tests: [sms, per_sm]
    using namespace sglang::host;
    int dev = 0, sms = 0, per_sm = 0;
    if (big) set_smem<16>(); else set_smem<8>();
    RuntimeDeviceCheck(cudaGetDevice(&dev));
    RuntimeDeviceCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    RuntimeDeviceCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, big ? tsim_kernel<16> : tsim_kernel<8>, kT,
                                                                     smem_bytes(big ? 16 : 8)));
    int32_t* o = static_cast<int32_t*>(out_i32.data_ptr());
    RuntimeCheck(out_i32.device().device_type == kDLCPU && out_i32.size(0) >= 2, "max_coresident: CPU int32 [2]");
    o[0] = sms;
    o[1] = per_sm;
  }
};

}  // namespace k31_tsim
#endif
