// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// K3.1 MK decode context parallelism (DCP = TP, e.g. TP16/DCP16): the cross-rank exchanges of one MLA/DSA layer.
// Layout and index math: dcp/layout.py (the stock SGLang DCP layout).
//
// Kernels (all PDL: griddepcontrol.wait before any predecessor output is read, launch_dependents after this grid's
// remote stores are issued):
//   prologue_kernel  once per forward: epoch += 1; widened write slots -> this rank's local rows (-1: other owner)
//   qpush_kernel     this rank's FP8 query rows [rows, H, 576] -> every peer's q region (records); optional -inf fill
//                    of the global-top-k score rows; optional all-visible compact list (dense graphs, no indexer)
//   cand_kernel      local top-k candidates (score bits, global position) -> every peer's cand region; poll every
//                    source's candidates and scatter the scores at their global positions (stock
//                    _scatter_candidates_kernel); poll every source's query heads into the padded query buffer
//   merge_kernel     attention partials of every peer's heads -> that peer's part region; the owner polls its heads'
//                    N partials, projects each by W_UV (BF16 MMA, FP32 accumulation, one BF16 rounding: stock's
//                    dcp_v_preproject bmm), LSE-combines them in rank order (stock dcp_push_combine: exp2 weights,
//                    FP32, x * (1 / max(sum, 1))), then the output gate (BF16(BF16(x) * BF16(sigmoid(g)))).
//
// Synchronization (memory-model audit): every cross-GPU datum is a 64-bit
// record {payload32 low, tag32 high} written with st.relaxed.sys.global.v2.u64 through the destination's unicast VA
// and polled with ld.relaxed.sys.global.v2.u64 until both elements carry the expected tag (each u64 element is one
// single-copy-atomic access, PTX ISA 8.2.4 / vector accesses). No sentinel, no re-arm, no fence (uc_epoch,
// see dcp/layout.py). The tag is unique per (forward epoch, MLA ordinal, exchange kind);
// regions alternate parity by MLA ordinal. A rank never pushes to itself: its own source is read from its local
// input, so no CTA ever waits on another CTA of its own grid (forward progress depends only on remote grids).
// Every poll fails stop (trap) after ``deadline_ns`` instead of hanging.
#pragma once

#include <cuda_bf16.h>
#include <cstdint>


namespace k31_dcp {

constexpr int kMaxDcp = 16;
constexpr int kHeadBytes = 576;
constexpr int kLatent = 512;
constexpr int kPartWords = 258;  // 256 BF16 pairs + LSE + pad (16-byte aligned head blocks)
constexpr int kExchanges = 3;
constexpr int kKindQ = 0, kKindCand = 1, kKindPart = 2;
constexpr int kThreads = 128;

// ------------------------------------------------------------------------------------------------ primitives
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
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
// Two records (16 B aligned) to a peer (or local) unicast address; system scope (the reader is another GPU).
__device__ __forceinline__ void st_rec2(uint64_t addr, uint64_t a, uint64_t b) {
  asm volatile("st.relaxed.sys.global.v2.u64 [%0], {%1, %2};" ::"l"(addr), "l"(a), "l"(b) : "memory");
}
__device__ __forceinline__ void ld_rec2(const uint64_t* p, uint64_t& a, uint64_t& b) {
  asm volatile("ld.relaxed.sys.global.v2.u64 {%0, %1}, [%2];" : "=l"(a), "=l"(b) : "l"(p) : "memory");
}
// A device-memory value another kernel of this stream wrote (read after griddepcontrol.wait; never ld.global.nc).
__device__ __forceinline__ uint32_t ld_gpu_u32(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ bool tag_ok(uint32_t seen, uint32_t want) {
  return seen == want;
}
// Poll two records at ``p`` until both carry ``want``; trap at the deadline.
__device__ __forceinline__ void poll2(const uint64_t* p, uint32_t want, uint64_t start, uint64_t deadline_ns,
                                      uint64_t& a, uint64_t& b) {
  ld_rec2(p, a, b);
  uint32_t spins = 0;
  while (!(tag_ok(hi32(a), want) && tag_ok(hi32(b), want))) {
    if ((++spins & 255u) == 0 && gtime() - start > deadline_ns) __trap();
    ld_rec2(p, a, b);
  }
}

__host__ __device__ __forceinline__ uint32_t make_tag(uint32_t epoch, int ordinal, int kind, int layers) {
  // Must equal dcp/layout.py tag(): ((epoch * layers + ordinal) * 3 + kind + 1) mod 2^32, never 0.
  const uint32_t v = (epoch * static_cast<uint32_t>(layers) + static_cast<uint32_t>(ordinal)) * kExchanges +
                     static_cast<uint32_t>(kind) + 1u;
  return v ? v : 1u;
}
__host__ __device__ __forceinline__ int local_len(int length, int dcp, int rank) {
  return length <= 0 ? 0 : (length + dcp - 1 - rank) / dcp;
}

struct Common {
  uint64_t peers[kMaxDcp];  // every rank's receive region of this exchange (records), indexed by rank
  const uint64_t* box;      // this rank's receive region (== peers[rank])
  const uint32_t* epoch;    // device epoch counter (advanced by prologue_kernel)
  uint64_t deadline_ns;
  int dcp, rank, parity, ordinal, layers, cap;
  int phases;  // bit 0: push, bit 1: poll (serving: 3; single-GPU multi-rank tests run every rank's push, then polls)
};

// ------------------------------------------------------------------------------------------------ prologue
struct PrologueParams {
  uint32_t* epoch;
  const int64_t* write_slots;  // [rows] widened slots from the host (-1: padding)
  int64_t* local_slots;        // [rows] this rank's physical row or -1
  int rows, dcp, rank;
};

__global__ void __launch_bounds__(kThreads) prologue_kernel(const __grid_constant__ PrologueParams p) {
  pdl_wait();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0) *p.epoch = *p.epoch + 1u;  // read by successors only after their griddepcontrol.wait
  if (i < p.rows) {
    const int64_t w = p.write_slots[i];
    const bool own = w >= 0 && w % p.dcp == p.rank;
    p.local_slots[i] = own ? w / p.dcp : -1;
  }
  pdl_trigger();
}

// ------------------------------------------------------------------------------------------------ q push
struct QPushParams {
  Common c;
  const uint8_t* q;   // [rows, heads, 576] FP8 bytes of this rank's heads
  int64_t q_stride;   // bytes between rows (multiple of 8)
  int rows, heads;
  // Optional -inf fill of the global top-k score rows over [0, glens[row]) (nullptr: none).
  float* dense;
  int64_t dense_stride;
  const int32_t* glens;
  // Optional all-visible compact list (dense graphs): local rows of positions 0..local_len-1 of each row's request.
  int32_t* visible;            // [rows, visible_stride] (nullptr: none)
  int32_t* visible_count;      // [rows]
  const int32_t* table;        // [requests, pages] widened page ids
  int64_t table_stride, visible_stride;
  int rows_per_request;        // verify: query width (row r reads request r / q)
};

__global__ void __launch_bounds__(kThreads) qpush_kernel(const __grid_constant__ QPushParams p) {
  const Common& c = p.c;
  pdl_wait();
  const uint32_t tag = make_tag(ld_gpu_u32(c.epoch), c.ordinal, kKindQ, c.layers);
  const int words = p.heads * kHeadBytes / 4, pairs = words / 2;
  const int64_t gtid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t nthreads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  // Push: pair i of row r -> region [parity][rank][r][2i] on every peer but this rank.
  for (int64_t i = gtid; (c.phases & 1) && i < static_cast<int64_t>(p.rows) * pairs; i += nthreads) {
    const int r = static_cast<int>(i / pairs), k = static_cast<int>(i % pairs);
    const uint2 v = *reinterpret_cast<const uint2*>(p.q + r * p.q_stride + 8 * k);
    const uint64_t a = rec(v.x, tag), b = rec(v.y, tag);
    const int64_t slot = ((static_cast<int64_t>(c.parity) * c.dcp + c.rank) * c.cap + r) * words + 2 * k;
    for (int j = 1; j < c.dcp; ++j) {
      const int d = (c.rank + j) % c.dcp;
      st_rec2(c.peers[d] + 8 * slot, a, b);
    }
  }
  pdl_trigger();
  if (p.dense != nullptr) {
    for (int r = 0; r < p.rows; ++r) {
      const int n = p.glens[r];
      float* row = p.dense + r * p.dense_stride;
      for (int64_t j = gtid; j < n; j += nthreads) row[j] = -INFINITY;
    }
  }
  if (p.visible != nullptr) {
    for (int r = 0; r < p.rows; ++r) {
      const int n = local_len(p.glens[r], c.dcp, c.rank);
      const int32_t* table = p.table + (r / p.rows_per_request) * p.table_stride;
      for (int64_t l = gtid; l < p.visible_stride; l += nthreads) {
        p.visible[r * p.visible_stride + l] = l < n ? table[l / 64] * 64 + static_cast<int>(l % 64) : -1;
      }
      if (gtid == 0) p.visible_count[r] = n;
    }
  }
}

// ------------------------------------------------------------------------------------------------ candidates
struct CandParams {
  Common c;                 // cand region
  const int32_t* cand;      // [rows, topk, 2] (score bits, global position or -1), this rank's local candidates
  int64_t cand_stride;      // int32 elements between rows
  const int32_t* glens;     // [rows] global lengths
  float* dense;             // [rows, dense_stride] global score rows (filled with -inf over [0, glen) before)
  int64_t dense_stride;
  int rows, topk;
  int push_cand;            // 0: skip the candidate exchange (dense graphs)
  // q poll (q region of the same parity; nullptr: skip)
  const uint64_t* q_box;
  uint8_t* q_out;           // [rows, padded_heads, 576]
  int64_t q_out_stride;     // bytes
  const uint8_t* q_local;   // [rows, heads, 576] this rank's own heads (copied into its block)
  int64_t q_local_stride;
  int heads;
  uint32_t q_tag_kind;      // kKindQ
};

__global__ void __launch_bounds__(kThreads) cand_kernel(const __grid_constant__ CandParams p) {
  const Common& c = p.c;
  pdl_wait();
  const uint32_t epoch = ld_gpu_u32(c.epoch);
  const uint32_t tag = make_tag(epoch, c.ordinal, kKindCand, c.layers);
  const int64_t gtid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t nthreads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t row_records = static_cast<int64_t>(p.topk) * 2;
  if (p.push_cand && (c.phases & 1)) {
    for (int r = 0; r < p.rows; ++r) {
      const int n = min(local_len(p.glens[r], c.dcp, c.rank), p.topk);
      const int64_t slot = ((static_cast<int64_t>(c.parity) * c.dcp + c.rank) * c.cap + r) * row_records;
      for (int64_t i = gtid; i < n; i += nthreads) {
        const int2 v = *reinterpret_cast<const int2*>(p.cand + r * p.cand_stride + 2 * i);
        const uint64_t a = rec(static_cast<uint32_t>(v.x), tag), b = rec(static_cast<uint32_t>(v.y), tag);
        for (int j = 1; j < c.dcp; ++j) {
          const int d = (c.rank + j) % c.dcp;
          st_rec2(c.peers[d] + 8 * (slot + 2 * i), a, b);
        }
      }
    }
  }
  pdl_trigger();
  if (!(c.phases & 2)) return;
  const uint64_t start = gtime();
  if (p.push_cand) {
    // Scatter every source's candidates at their global positions (stock _scatter_candidates_kernel: positions
    // < 0 are skipped; a source sends min(local_len, topk) candidates).
    for (int s = 0; s < c.dcp; ++s) {
      for (int r = 0; r < p.rows; ++r) {
        const int n = min(local_len(p.glens[r], c.dcp, s), p.topk);
        float* row = p.dense + r * p.dense_stride;
        const int64_t slot = ((static_cast<int64_t>(c.parity) * c.dcp + s) * c.cap + r) * row_records;
        for (int64_t i = gtid; i < n; i += nthreads) {
          int32_t score, pos;
          if (s == c.rank) {
            const int2 v = *reinterpret_cast<const int2*>(p.cand + r * p.cand_stride + 2 * i);
            score = v.x, pos = v.y;
          } else {
            uint64_t a, b;
            poll2(c.box + slot + 2 * i, tag, start, c.deadline_ns, a, b);
            score = static_cast<int32_t>(lo32(a)), pos = static_cast<int32_t>(lo32(b));
          }
          if (pos >= 0) row[pos] = __int_as_float(score);
        }
      }
    }
  }
  if (p.q_box != nullptr) {
    const uint32_t qtag = make_tag(epoch, c.ordinal, kKindQ, c.layers);
    const int words = p.heads * kHeadBytes / 4, pairs = words / 2;
    for (int64_t i = gtid; i < static_cast<int64_t>(c.dcp) * p.rows * pairs; i += nthreads) {
      const int s = static_cast<int>(i / (static_cast<int64_t>(p.rows) * pairs));
      const int64_t rem = i % (static_cast<int64_t>(p.rows) * pairs);
      const int r = static_cast<int>(rem / pairs), k = static_cast<int>(rem % pairs);
      const int block = s * p.heads;
      uint2 v;
      if (s == c.rank) {
        v = *reinterpret_cast<const uint2*>(p.q_local + r * p.q_local_stride + 8 * k);
      } else {
        const int64_t slot = ((static_cast<int64_t>(c.parity) * c.dcp + s) * c.cap + r) * words + 2 * k;
        uint64_t a, b;
        poll2(p.q_box + slot, qtag, start, c.deadline_ns, a, b);
        v = make_uint2(lo32(a), lo32(b));
      }
      *reinterpret_cast<uint2*>(p.q_out + r * p.q_out_stride + static_cast<int64_t>(block) * kHeadBytes + 8 * k) = v;
    }
  }
}

// ------------------------------------------------------------------------------------------------ merge
struct MergeParams {
  Common c;                        // part region
  const __nv_bfloat16* part;       // [rows, padded_heads, 512] this rank's partials of every gathered head
  int64_t part_row_stride, part_head_stride;  // elements
  const float* lse;                // [rows, padded_heads] (log2)
  int64_t lse_row_stride, lse_head_stride;
  const int32_t* counts;           // [rows] keys this rank attended (0: the partial is empty)
  const int32_t* totals;           // [rows] global lengths (0: padding row -> zero output)
  const __nv_bfloat16* w_uv;       // this rank's heads: element (h, k, n) at h * w_head_stride + n * w_n_stride + k
  int64_t w_head_stride, w_n_stride;
  const __nv_bfloat16* gate;       // [rows, heads * 128] unsigmoided output gate (projected[:, 2112:])
  int64_t gate_stride;
  __nv_bfloat16* out;              // [rows, heads * 128]
  int64_t out_stride;
  int rows, heads;
};

__device__ __forceinline__ void mma_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float bf16_round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float sanitize_lse(float x) { return (x != x || x == INFINITY) ? -INFINITY : x; }

constexpr int kAStride = kLatent + 8;  // padded SMEM row (BF16): conflict-free fragment loads

struct MergeSmem {
  __nv_bfloat16 a[16][kAStride];  // source partials of one (row, head): M = 16 sources (dcp <= 16)
  float lse[16];
  float y[16][32];                // projected partials of this CTA's 32 columns, BF16-rounded
};

// Work items (row, head, column quarter), grid-strided over at most ``blocks`` CTAs (one wave: every CTA is resident
// before any waits on a remote grid; the push phase is spread over all CTAs and precedes every poll). 128 threads =
// 4 warps; warp w owns 8 of the item's 32 columns.
__global__ void __launch_bounds__(kThreads) merge_kernel(const __grid_constant__ MergeParams p) {
  const Common& c = p.c;
  __shared__ __align__(16) MergeSmem s;
  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  const int group = lane / 4, tig = lane % 4;
  const int items = p.rows * p.heads * 4;
  pdl_wait();
  const uint32_t tag = make_tag(ld_gpu_u32(c.epoch), c.ordinal, kKindPart, c.layers);
  // Push: every CTA takes a share of (destination, row, destination head, word pair) items.
  {
    const int64_t gtid = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
    const int64_t nthreads = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const int pairs = kPartWords / 2;  // 129: 128 data pairs + (LSE, pad)
    const int64_t per_dest = static_cast<int64_t>(p.rows) * p.heads * pairs;
    for (int64_t i = gtid; (c.phases & 1) && i < static_cast<int64_t>(c.dcp - 1) * per_dest; i += nthreads) {
      const int d = (c.rank + 1 + static_cast<int>(i / per_dest)) % c.dcp;
      const int64_t rem = i % per_dest;
      const int r = static_cast<int>(rem / (p.heads * pairs));
      const int j = static_cast<int>((rem / pairs) % p.heads), k = static_cast<int>(rem % pairs);
      if (p.totals[r] <= 0) continue;  // padding rows are neither sent nor read (totals are replicated)
      const bool empty = p.counts[r] <= 0;  // stock fixup_zero_kv_rows: zero output, LSE -inf
      const int h = d * p.heads + j;
      uint32_t w0, w1;
      if (k < kLatent / 4) {
        if (empty) {
          w0 = w1 = 0u;
        } else {
          const uint2 v =
              *reinterpret_cast<const uint2*>(p.part + r * p.part_row_stride + h * p.part_head_stride + 4 * k);
          w0 = v.x, w1 = v.y;
        }
      } else {
        const float l = empty ? -INFINITY : p.lse[r * p.lse_row_stride + h * p.lse_head_stride];
        w0 = __float_as_uint(l), w1 = 0u;
      }
      const int64_t slot =
          (((static_cast<int64_t>(c.parity) * c.dcp + c.rank) * c.cap + r) * p.heads + j) * kPartWords + 2 * k;
      st_rec2(c.peers[d] + 8 * slot, rec(w0, tag), rec(w1, tag));
    }
  }
  pdl_trigger();
  if (!(c.phases & 2)) return;
  const uint64_t start = gtime();
  for (int item = blockIdx.x; item < items; item += gridDim.x) {
    const int quarter = item % 4, head = (item / 4) % p.heads, row = item / (4 * p.heads);
    __nv_bfloat16* const out = p.out + row * p.out_stride + head * 128 + quarter * 32;
    if (p.totals[row] <= 0) {  // padding rows may hold a NaN gate: keep them zero
      if (tid < 32) out[tid] = __float2bfloat16_rn(0.f);
      continue;
    }
    // B fragments of this lane's column n over the 512 latent values (k pairs 2*tig, +8): static weights.
    const int n = quarter * 32 + warp * 8 + group;
    const __nv_bfloat16* wcol = p.w_uv + head * p.w_head_stride + n * p.w_n_stride;
    // Gather the N source partials of (row, head) into SMEM rows 0..dcp-1 (rows dcp..15 zero, LSE -inf).
    const int me = c.rank * p.heads + head;
    for (int i = tid; i < 16 * (kPartWords / 2); i += kThreads) {
      const int src = i / (kPartWords / 2), k = i % (kPartWords / 2);
      uint32_t w0 = 0u, w1 = 0u;
      if (src < c.dcp) {
        if (src == c.rank) {
          const bool empty = p.counts[row] <= 0;
          if (k < kLatent / 4) {
            if (!empty) {
              const uint2 v = *reinterpret_cast<const uint2*>(p.part + row * p.part_row_stride +
                                                              me * p.part_head_stride + 4 * k);
              w0 = v.x, w1 = v.y;
            }
          } else {
            w0 = __float_as_uint(empty ? -INFINITY : p.lse[row * p.lse_row_stride + me * p.lse_head_stride]);
          }
        } else {
          const int64_t slot =
              (((static_cast<int64_t>(c.parity) * c.dcp + src) * c.cap + row) * p.heads + head) * kPartWords + 2 * k;
          uint64_t a, b;
          poll2(c.box + slot, tag, start, c.deadline_ns, a, b);
          w0 = lo32(a), w1 = lo32(b);
        }
      } else if (k == kLatent / 4) {
        w0 = __float_as_uint(-INFINITY);
      }
      if (k < kLatent / 4) {
        *reinterpret_cast<uint2*>(&s.a[src][4 * k]) = make_uint2(w0, w1);
      } else {
        s.lse[src] = __uint_as_float(w0);
      }
    }
    __syncthreads();
    // y[src][n] = BF16(sum_k a[src][k] * W_UV[k][n]): M = 16 sources, N = this warp's 8 columns, K = 512.
    float d[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 8
    for (int ks = 0; ks < 32; ++ks) {
      const int k0 = 16 * ks + 2 * tig;
      const uint32_t b0 = *reinterpret_cast<const uint32_t*>(wcol + 16 * ks + 2 * tig);
      const uint32_t b1 = *reinterpret_cast<const uint32_t*>(wcol + 16 * ks + 8 + 2 * tig);
      const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&s.a[group][k0]);
      const uint32_t a1 = *reinterpret_cast<const uint32_t*>(&s.a[group + 8][k0]);
      const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&s.a[group][k0 + 8]);
      const uint32_t a3 = *reinterpret_cast<const uint32_t*>(&s.a[group + 8][k0 + 8]);
      mma_bf16(d, a0, a1, a2, a3, b0, b1);
    }
    // D fragment: (row group, cols 2*tig, 2*tig+1) and (row group+8, same cols) of the warp's 8-column tile.
    const float y0 = bf16_round(d[0]), y1 = bf16_round(d[1]), y2 = bf16_round(d[2]), y3 = bf16_round(d[3]);
    s.y[group][warp * 8 + 2 * tig] = y0;
    s.y[group][warp * 8 + 2 * tig + 1] = y1;
    s.y[group + 8][warp * 8 + 2 * tig] = y2;
    s.y[group + 8][warp * 8 + 2 * tig + 1] = y3;
    __syncthreads();
    if (tid < 32) {
      // stock dcp_push_combine: lse sanitized, max over ranks (0 if all -inf), w_i = exp2(lse_i - max), acc and the
      // weight sum in rank order, x * (1 / max(sum, 1)); then the gate BF16(BF16(x) * BF16(sigmoid(g))).
      float lse_max = -INFINITY;
      for (int i = 0; i < c.dcp; ++i) lse_max = fmaxf(lse_max, sanitize_lse(s.lse[i]));
      if (lse_max == -INFINITY) lse_max = 0.f;
      float acc = 0.f, weight_sum = 0.f;
      for (int i = 0; i < c.dcp; ++i) {
        const float wi = exp2f(sanitize_lse(s.lse[i]) - lse_max);
        weight_sum += wi;
        acc = __fmaf_rn(s.y[i][tid], wi, acc);  // stock: acc += x * w, contracted to one FMA (nvcc -O3)
      }
      const float inv = 1.f / fmaxf(weight_sum, 1.f);
      const float x = bf16_round(acc * inv);
      const int col = head * 128 + quarter * 32 + tid;
      const float g = __bfloat162float(p.gate[row * p.gate_stride + col]);
      const float sig = bf16_round(1.f / (1.f + expf(-g)));
      out[tid] = __float2bfloat16_rn(x * sig);
    }
    __syncthreads();  // SMEM is reused by the next item
  }
}

}  // namespace k31_dcp

#ifndef K31_DCP_DEVICE_ONLY
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

namespace k31_dcp {

// Host binding. ``meta`` is a CPU int64 tensor: [dcp, rank, parity, ordinal, layers, cap, deadline_ns, blocks,
// phases, peer_0 .. peer_{dcp-1}] (peer region base addresses of this exchange, own included). Everything a captured graph
// must not bake as a changing value (the epoch) is read from device memory.
struct Binding {
  static Common common(const tvm::ffi::TensorView meta, const tvm::ffi::TensorView epoch) {
    using namespace sglang::host;
    const int64_t* m = static_cast<const int64_t*>(meta.data_ptr());
    RuntimeCheck(meta.device().device_type == kDLCPU && meta.size(0) >= 9 + m[0], "DCP meta: CPU int64 [9 + dcp]");
    Common c{};
    c.dcp = static_cast<int>(m[0]);
    c.rank = static_cast<int>(m[1]);
    c.parity = static_cast<int>(m[2]);
    c.ordinal = static_cast<int>(m[3]);
    c.layers = static_cast<int>(m[4]);
    c.cap = static_cast<int>(m[5]);
    c.deadline_ns = static_cast<uint64_t>(m[6]);
    c.phases = static_cast<int>(m[8]);
    RuntimeCheck(c.dcp >= 2 && c.dcp <= kMaxDcp && c.rank >= 0 && c.rank < c.dcp, "DCP size/rank");
    for (int i = 0; i < c.dcp; ++i) c.peers[i] = static_cast<uint64_t>(m[9 + i]);
    c.box = reinterpret_cast<const uint64_t*>(c.peers[c.rank]);
    c.epoch = static_cast<const uint32_t*>(epoch.data_ptr());
    return c;
  }
  static int blocks(const tvm::ffi::TensorView meta) { return static_cast<int>(static_cast<const int64_t*>(meta.data_ptr())[7]); }

  static void prologue(const tvm::ffi::TensorView epoch, const tvm::ffi::TensorView write_slots,
                       const tvm::ffi::TensorView local_slots, int64_t dcp, int64_t rank) {
    using namespace sglang::host;
    const int rows = static_cast<int>(write_slots.size(0));
    RuntimeCheck(local_slots.size(0) >= rows, "local slots");
    PrologueParams p{static_cast<uint32_t*>(epoch.data_ptr()), static_cast<const int64_t*>(write_slots.data_ptr()),
                     static_cast<int64_t*>(local_slots.data_ptr()), rows, static_cast<int>(dcp),
                     static_cast<int>(rank)};
    LaunchKernel((rows + kThreads - 1) / kThreads, kThreads, epoch.device()).enable_pdl(true)(prologue_kernel, p);
  }

  static void qpush(const tvm::ffi::TensorView meta, const tvm::ffi::TensorView epoch, const tvm::ffi::TensorView q,
                    const tvm::ffi::TensorView dense, const tvm::ffi::TensorView glens,
                    const tvm::ffi::TensorView visible, const tvm::ffi::TensorView visible_count,
                    const tvm::ffi::TensorView table, int64_t rows_per_request, int64_t flags) {
    using namespace sglang::host;
    QPushParams p{};
    p.c = common(meta, epoch);
    p.q = static_cast<const uint8_t*>(q.data_ptr());
    p.q_stride = q.stride(0);
    p.rows = static_cast<int>(q.size(0));
    p.heads = static_cast<int>(q.size(1));
    RuntimeCheck(q.size(2) == kHeadBytes && q.stride(1) == kHeadBytes && p.q_stride % 8 == 0, "q [rows, H, 576]");
    RuntimeCheck(p.rows <= p.c.cap, "rows exceed the mailbox capacity");
    p.glens = static_cast<const int32_t*>(glens.data_ptr());
    if (flags & 1) {
      p.dense = static_cast<float*>(dense.data_ptr());
      p.dense_stride = dense.stride(0);
    }
    if (flags & 2) {
      p.visible = static_cast<int32_t*>(visible.data_ptr());
      p.visible_count = static_cast<int32_t*>(visible_count.data_ptr());
      p.visible_stride = visible.stride(0);
      p.table = static_cast<const int32_t*>(table.data_ptr());
      p.table_stride = table.stride(0);
      p.rows_per_request = static_cast<int>(rows_per_request);
    }
    LaunchKernel(blocks(meta), kThreads, q.device()).enable_pdl(true)(qpush_kernel, p);
  }

  static void cand(const tvm::ffi::TensorView meta, const tvm::ffi::TensorView q_meta,
                   const tvm::ffi::TensorView epoch, const tvm::ffi::TensorView cand, const tvm::ffi::TensorView glens,
                   const tvm::ffi::TensorView dense, const tvm::ffi::TensorView q_local,
                   const tvm::ffi::TensorView q_out, int64_t push_cand) {
    using namespace sglang::host;
    CandParams p{};
    p.c = common(meta, epoch);
    p.rows = static_cast<int>(q_local.size(0));
    RuntimeCheck(p.rows <= p.c.cap, "rows exceed the mailbox capacity");
    p.glens = static_cast<const int32_t*>(glens.data_ptr());
    p.push_cand = static_cast<int>(push_cand);
    if (p.push_cand) {
      RuntimeCheck(cand.size(0) >= p.rows && cand.size(2) == 2 && cand.stride(1) == 2, "candidates [rows, topk, 2]");
      p.cand = static_cast<const int32_t*>(cand.data_ptr());
      p.cand_stride = cand.stride(0);
      p.topk = static_cast<int>(cand.size(1));
      p.dense = static_cast<float*>(dense.data_ptr());
      p.dense_stride = dense.stride(0);
    }
    const Common qc = common(q_meta, epoch);
    RuntimeCheck(qc.parity == p.c.parity && qc.ordinal == p.c.ordinal && qc.dcp == p.c.dcp, "q/cand meta");
    p.q_box = qc.box;
    p.q_out = static_cast<uint8_t*>(q_out.data_ptr());
    p.q_out_stride = q_out.stride(0);
    p.q_local = static_cast<const uint8_t*>(q_local.data_ptr());
    p.q_local_stride = q_local.stride(0);
    p.heads = static_cast<int>(q_local.size(1));
    RuntimeCheck(q_out.size(1) * kHeadBytes == q_out.stride(0) && q_out.size(1) >= p.c.dcp * p.heads, "padded q");
    LaunchKernel(blocks(meta), kThreads, q_local.device()).enable_pdl(true)(cand_kernel, p);
  }

  static void merge(const tvm::ffi::TensorView meta, const tvm::ffi::TensorView epoch,
                    const tvm::ffi::TensorView part, const tvm::ffi::TensorView lse,
                    const tvm::ffi::TensorView counts, const tvm::ffi::TensorView totals,
                    const tvm::ffi::TensorView w_uv, const tvm::ffi::TensorView gate, const tvm::ffi::TensorView out) {
    using namespace sglang::host;
    MergeParams p{};
    p.c = common(meta, epoch);
    p.rows = static_cast<int>(out.size(0));
    p.heads = static_cast<int>(w_uv.size(0));
    RuntimeCheck(p.rows <= p.c.cap && out.size(1) == p.heads * 128 && out.stride(1) == 1, "out [rows, H * 128]");
    RuntimeCheck(part.size(-1) == kLatent && part.stride(-1) == 1, "partials [rows, heads, 512]");
    RuntimeCheck(w_uv.size(1) == kLatent && w_uv.size(2) == 128 && w_uv.stride(1) == 1, "W_UV (k contiguous)");
    p.part = static_cast<const __nv_bfloat16*>(part.data_ptr());
    p.part_row_stride = part.stride(0);
    p.part_head_stride = part.stride(part.ndim() - 2);
    p.lse = static_cast<const float*>(lse.data_ptr());
    p.lse_row_stride = lse.stride(0);
    p.lse_head_stride = lse.stride(lse.ndim() - 1);
    p.counts = static_cast<const int32_t*>(counts.data_ptr());
    p.totals = static_cast<const int32_t*>(totals.data_ptr());
    p.w_uv = static_cast<const __nv_bfloat16*>(w_uv.data_ptr());
    p.w_head_stride = w_uv.stride(0);
    p.w_n_stride = w_uv.stride(2);
    p.gate = static_cast<const __nv_bfloat16*>(gate.data_ptr());
    p.gate_stride = gate.stride(0);
    p.out = static_cast<__nv_bfloat16*>(out.data_ptr());
    p.out_stride = out.stride(0);
    const int items = p.rows * p.heads * 4;
    LaunchKernel(std::min(items, blocks(meta)), kThreads, out.device()).enable_pdl(true)(merge_kernel, p);
  }
};

}  // namespace k31_dcp
#endif
