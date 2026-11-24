// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Launch wrappers for the hop device API v2 (lb_hop.cuh): the per-forward sequence bump, the ring counter advance,
// standalone row transfers, and the self-test kernels used by tests/dist_lb_hop.py.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>

#include "lb_hop.cuh"

#include <cstdint>

namespace sglang {

namespace muse_vr_lb_hop_ops {

namespace hop = muse_vr::lb_hop;

inline constexpr int kMaxHops = 16;

struct Counts {
  unsigned long long add[kMaxHops];
};

// Forward head: raises the per-forward sequence word by one. The first chain kernel after it must be launched without
// programmatic stream serialization, so that every later kernel of the forward reads the new value.
__global__ void lbhop_bump_kernel(unsigned long long* seq64) {
  if (threadIdx.x == 0 && blockIdx.x == 0) *seq64 += 1ull;
}

// Forward head with the forward's step_release: raises seq64 and releases this rank's step token to every rank. It
// has the full, non-programmatic stream dependency on the previous forward that the first hop kernel would need, so the
// forward's hop kernels only acquire.
template <int W>
__global__ void lbhop_bump_release_kernel(unsigned long long* seq64, const __grid_constant__ hop::StepTok<W> t) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    const unsigned long long s = *seq64 + 1ull;
    *seq64 = s;
    hop::step_release(t, s);
  }
}

// Raises every ring hop's transfer counter by the number of transfers the step performed (after the step's kernels).
template <bool kUsePDL>
__global__ void lbhop_advance_kernel(unsigned long long* base, uint32_t n, const __grid_constant__ Counts counts) {
  device::PDLWaitPrimary<kUsePDL>();
  if (threadIdx.x == 0 && blockIdx.x == 0)
    for (uint32_t i = 0; i < n; ++i) base[i] += counts.add[i];
}

// ------------------------------------------------------------------------------------------------- test patterns
// Payload value of element (src rank, transfer, row, col): a finite bf16 with magnitude below 4.
SGL_DEVICE uint32_t pattern(uint32_t src, unsigned long long t, uint32_t row, uint32_t col) {
  uint32_t h = uint32_t(t) * 2654435761u ^ uint32_t(t >> 32) * 40503u ^ src * 97u ^ row * 2246822519u ^ col * 3266489917u;
  h ^= h >> 15;
  h *= 2246822519u;
  h ^= h >> 13;
  const uint32_t sign = (h & 1u) << 15;
  const uint32_t exp = 120u + (h >> 1) % 8u;
  const uint32_t man = (h >> 4) & 0x7fu;
  return sign | (exp << 7) | man;
}

// Spins a pseudo-random 0..max_units delay (a unit is 1024 clock64() ticks) chosen by hashing the key.
SGL_DEVICE void fuzz_wait(uint32_t key, uint32_t max_units) {
  if (!max_units) return;
  uint32_t h = key * 2654435761u;
  h ^= h >> 15;
  h *= 2246822519u;
  h ^= h >> 13;
  const long long until = clock64() + 1024ll * (h % (max_units + 1));
  while (clock64() < until) {
  }
}

inline constexpr uint32_t kModeSkipPush = 1u;     // producer: do not push (a dead peer)
inline constexpr uint32_t kModeEarlyStore = 2u;   // consumer: store outputs before the dependency wait (mutant)
inline constexpr uint32_t kModeBench = 4u;        // producer: constant payload; consumer: no pattern check
inline constexpr uint32_t kModeTile = 8u;         // producer: column tiles of tile_cols, one tile per CTA (writer audit)
inline constexpr uint32_t kModeNoGuard = 16u;     // consumer: skip the missed-advance guard (mutant)
inline constexpr uint32_t kModeStepTok = 32u;     // producer: step_release + step_acquire before pushing
inline constexpr uint32_t kModeRingScrub = 64u;   // consumer (kRing): scrub rows above `rows` after the wait
inline constexpr uint32_t kModeNoRelease = 128u;  // producer with kModeStepTok: skip step_release (negative twin)
inline constexpr uint32_t kModeFuzzReads = 256u;  // consumer: a seeded delay per warp before each read round (race fuzz)
inline constexpr uint32_t kModeEarlyCredit = 512u;  // consumer (kRing + Ctl kAsyncCredit): release the credit BEFORE the
                                                    // reads (negative twin of the async-credit form; must be caught)
inline constexpr uint32_t kModeReplicate = 1024u;   // consumer: EVERY CTA collects all rows (full-row landing in each
                                                    // consumer group, a bench mode); outputs are written by CTA 0

template <int W>
SGL_DEVICE void audit_line(uint32_t* audit, uint32_t* bad, const hop::Site<W>& s, uint32_t dst, uint32_t row, uint32_t col) {
  if (audit == nullptr) return;
  const uint32_t lines = s.pitch / hop::kLineWords;
  const uint32_t line = (col / 3u) / hop::kLineWords;
  uint32_t* a = audit + (uint64_t(dst) * s.rows_max + row) * lines + line;
  const uint32_t me = blockIdx.x + 1u;
  const uint32_t old = atomicCAS(a, 0u, me);
  if (old != 0u && old != me) atomicAdd(bad, 1u);
}

// Pushes rows 0..rows-1 of this rank's block (pattern values). Default: element groups spread over every thread of
// the grid; kModeTile: CTA b pushes column tiles of tile_cols elements (rows x tiles assigned round-robin), recording
// the writer CTA of every 128-byte line in `audit` (a multi-writer line increments *bad).
template <int W, bool kUsePDL>
__global__ void lbhop_test_push_kernel(const __grid_constant__ hop::Site<W> s, const __grid_constant__ hop::Ctl c,
                                       const __grid_constant__ hop::StepTok<W> tk, uint32_t rows, uint32_t fuzz,
                                       uint32_t mode, uint32_t tile_cols, uint32_t* audit, uint32_t* bad) {
  __shared__ int ok_s;
  device::PDLTriggerSecondary<kUsePDL>();
  const hop::Xfer x = hop::xfer(s);
  if (threadIdx.x < 32) {
    bool ok = true;
    if (s.policy == hop::kRing) ok = hop::credit_acquire(s, c, x);
    if (ok && (mode & kModeStepTok)) {
      if (blockIdx.x == 0 && threadIdx.x == 0 && !(mode & kModeNoRelease)) hop::step_release(tk, x.seq);
      __syncwarp();
      ok = hop::step_acquire(tk, c, x.seq);
    }
    if (threadIdx.x == 0) ok_s = ok ? 1 : 0;
  }
  __syncthreads();
  device::PDLWaitPrimary<kUsePDL>();
  if (!ok_s || (mode & kModeSkipPush)) return;
  if (fuzz) {
    if (threadIdx.x == 0) fuzz_wait(uint32_t(x.t) * 7919u + blockIdx.x * 131u + s.rank * 31u, fuzz);
    __syncthreads();
  }
  const uint32_t groups = s.width / 6u;
  if (mode & kModeTile) {
    const uint32_t tiles_c = (s.width + tile_cols - 1u) / tile_cols;
    for (uint32_t tile = blockIdx.x; tile < rows * tiles_c; tile += gridDim.x) {
      const uint32_t row = tile / tiles_c, c0 = (tile % tiles_c) * tile_cols;
      const uint32_t c1 = min(c0 + tile_cols, s.width);
      for (uint32_t col = c0 + threadIdx.x * 6u; col + 6u <= c1; col += blockDim.x * 6u) {
        uint32_t e[6];
#pragma unroll
        for (int k = 0; k < 6; ++k) e[k] = (mode & kModeBench) ? 0x3f80u : pattern(s.rank, x.t, row, col + k);
        hop::push6(s, x, row, col, e);
        for (int r = 0; r < W; ++r) audit_line(audit, bad, s, uint32_t(r), row, col);
      }
    }
    return;
  }
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * groups; i += gridDim.x * blockDim.x) {
    const uint32_t row = i / groups, col = (i % groups) * 6u;
    uint32_t e[6];
#pragma unroll
    for (int k = 0; k < 6; ++k) e[k] = (mode & kModeBench) ? 0x3f80u : pattern(s.rank, x.t, row, col + k);
    hop::push6(s, x, row, col, e);
  }
}

// Collects rows 0..rows-1 from every source (each CTA = one consumer unit of the missed-advance guard), checks every
// source's elements against the pattern (exact), writes the fp32 rank-order sums to out [rows][width] (NaN when the
// unit's wait failed or its guard refused), after the dependency wait. kRing + kModeRingScrub: scrubs rows above rows.
template <int W, bool kUsePDL>
__global__ void lbhop_test_collect_kernel(const __grid_constant__ hop::Site<W> s, const __grid_constant__ hop::Ctl c,
                                          uint32_t rows, uint32_t fuzz, uint32_t mode, float* out, uint32_t* mismatch) {
  constexpr int N = W >= 16 ? 1 : (W >= 8 ? 2 : 4);
  __shared__ int go_s;
  device::PDLTriggerSecondary<kUsePDL>();
  const hop::Xfer x = hop::xfer(s);
  if (threadIdx.x == 0) go_s = (mode & kModeNoGuard) ? 1 : (hop::guard_enter(s, c, blockIdx.x, x.seq) ? 1 : 0);
  __syncthreads();
  const bool go = go_s != 0;
  const bool async_credit = s.policy == hop::kRing && (c.flags & hop::kAsyncCredit);
  if (async_credit && (mode & kModeEarlyCredit) && threadIdx.x == 0) hop::credit_release_async(s, x, rows);
  if (fuzz) {
    if (threadIdx.x == 0) fuzz_wait(uint32_t(x.t) * 104729u + blockIdx.x * 17u + s.rank * 7u, fuzz);
    __syncthreads();
  }
  const uint32_t groups = s.width / 6u;
  const uint32_t total = rows * groups;
  const uint32_t lane = threadIdx.x & 31u;
  const bool repl = (mode & kModeReplicate) != 0u;
  const uint32_t warp = repl ? (threadIdx.x >> 5) : ((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  const uint32_t nwarps = repl ? (blockDim.x >> 5) : ((gridDim.x * blockDim.x) >> 5);
  const bool writer = !repl || blockIdx.x == 0;
  const float nan = __uint_as_float(0x7fc00000u);
  uint32_t badc = 0;
  bool waited = false;
  for (uint32_t base = warp * 32u * N; base < total; base += nwarps * 32u * N) {  // warp-uniform trip count
    if (fuzz && (mode & kModeFuzzReads)) {
      if (lane == 0) fuzz_wait(uint32_t(x.t) * 2654435761u + warp * 40503u + base + s.rank * 97u, fuzz);
      __syncwarp();
    }
    uint32_t rw[N], cl[N];
    int n = 0;
#pragma unroll
    for (int k = 0; k < N; ++k) {
      const uint32_t i = base + k * 32u + lane;
      rw[k] = i < total ? i / groups : 0u;
      cl[k] = i < total ? (i % groups) * 6u : 0u;
      if (i < total) n = k + 1;
    }
    ulonglong2 raw[N][W];
    float sum[N][6];
    bool ok = false;
    if (go) ok = hop::collect6<W, N>(s, c, x, rw, cl, n, raw, sum);
    if (ok && !(mode & kModeBench)) {
#pragma unroll
      for (int k = 0; k < N; ++k)
        if (k < n) {
#pragma unroll
          for (int r = 0; r < W; ++r)
#pragma unroll
            for (int e = 0; e < 6; ++e)
              if (hop::elem6(raw[k][r], e) != pattern(uint32_t(r), x.t, rw[k], cl[k] + e)) ++badc;
        }
    }
    if (!waited && !(mode & kModeEarlyStore)) {
      device::PDLWaitPrimary<kUsePDL>();
      waited = true;
    }
    if (writer)
#pragma unroll
      for (int k = 0; k < N; ++k)
        if (k < n) {
#pragma unroll
          for (int e = 0; e < 6; ++e) out[uint64_t(rw[k]) * s.width + cl[k] + e] = ok ? sum[k][e] : nan;
        }
  }
  device::PDLWaitPrimary<kUsePDL>();
  if (badc) atomicAdd(mismatch, badc);
  if (s.policy == hop::kRing && (mode & kModeRingScrub)) {
    const uint32_t hwm = hop::hwm_read(s, x);
    hop::ring_scrub(s, x, rows, hwm, blockIdx.x * blockDim.x + threadIdx.x, gridDim.x * blockDim.x);
  }
  __syncthreads();  // ends this CTA's reads (and scrub stores) of the slot
  if (threadIdx.x == 0 && go && !(mode & kModeNoGuard)) hop::guard_exit(s, blockIdx.x, x.seq);
  if (threadIdx.x == 0 && async_credit && !(mode & kModeEarlyCredit)) hop::credit_release_async(s, x, rows);
}

// kRing: credit release for the transfer just consumed (a later kernel, after its dependency wait).
template <int W, bool kUsePDL>
__global__ void lbhop_test_release_kernel(const __grid_constant__ hop::Site<W> s, uint32_t rows) {
  device::PDLWaitPrimary<kUsePDL>();
  if (threadIdx.x == 0 && blockIdx.x == 0) hop::credit_release(s, hop::xfer(s), rows);
}

// Predecessor for the store-after-wait test: fills out[0:n) with `value` after a delay, triggering its dependents at
// entry so that the next kernel can run concurrently with it.
template <bool kUsePDL>
__global__ void lbhop_test_late_write_kernel(float* out, uint32_t n, uint32_t delay_units, float value) {
  device::PDLTriggerSecondary<kUsePDL>();
  device::PDLWaitPrimary<kUsePDL>();
  if (threadIdx.x == 0) {
    const long long until = clock64() + 1024ll * delay_units;
    while (clock64() < until) {
    }
  }
  __syncthreads();
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) out[i] = value;
}

// ------------------------------------------------------------------------------------------------- row transfers
// Standalone producer / consumers of one transfer for callers whose own kernels do not push or collect (stock-shaped
// stand-ins and units). push_rows stores this rank's in [rows, width] (bf16) into every rank's copy; collect_rows
// writes out [rows, width] (bf16) = round-to-nearest-even of the fp32 rank-order sum (one-shot all-reduce arithmetic);
// gather_rows writes out [W, rows, width] (bf16) = every source's block as sent. Consumers keep the missed-advance
// guard (one unit per CTA). Sync credit form: neither releases credits (a later kernel does); with Ctl kAsyncCredit
// every collect CTA releases its own credit after its last read (credit_release_async).
template <int W, bool kUsePDL>
__global__ void lbhop_push_rows_kernel(const __grid_constant__ hop::Site<W> s, const __grid_constant__ hop::Ctl c,
                                       uint32_t rows, const uint16_t* in) {
  __shared__ int ok_s;
  const hop::Xfer x = hop::xfer(s);
  if (threadIdx.x < 32) {
    const bool ok = s.policy == hop::kRing ? hop::credit_acquire(s, c, x) : true;
    if (threadIdx.x == 0) ok_s = ok ? 1 : 0;
  }
  __syncthreads();
  device::PDLWaitPrimary<kUsePDL>();
  if (!ok_s) return;
  device::PDLTriggerSecondary<kUsePDL>();
  const uint32_t groups = s.width / 6u;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * groups; i += gridDim.x * blockDim.x) {
    const uint32_t row = i / groups, col = (i % groups) * 6u;
    const uint16_t* src = in + uint64_t(row) * s.width + col;
    const uint32_t e[6] = {src[0], src[1], src[2], src[3], src[4], src[5]};
    hop::push6(s, x, row, col, e);
  }
}

SGL_DEVICE uint32_t bf16_rne(float v) {
  const __nv_bfloat16 b = __float2bfloat16_rn(v);
  return uint32_t(*reinterpret_cast<const uint16_t*>(&b));
}

template <int W, bool kUsePDL, bool kGather>
__global__ void lbhop_collect_rows_kernel(const __grid_constant__ hop::Site<W> s, const __grid_constant__ hop::Ctl c,
                                          uint32_t rows, uint16_t* out) {
  constexpr int N = W >= 16 ? 1 : (W >= 8 ? 2 : 4);
  __shared__ int go_s;
  const hop::Xfer x = hop::xfer(s);
  if (threadIdx.x == 0) go_s = hop::guard_enter(s, c, blockIdx.x, x.seq) ? 1 : 0;
  __syncthreads();
  const bool go = go_s != 0;
  const uint32_t groups = s.width / 6u;
  const uint32_t total = rows * groups;
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const uint32_t nwarps = (gridDim.x * blockDim.x) >> 5;
  bool waited = false;
  for (uint32_t base = warp * 32u * N; base < total; base += nwarps * 32u * N) {
    uint32_t rw[N], cl[N];
    int n = 0;
#pragma unroll
    for (int k = 0; k < N; ++k) {
      const uint32_t i = base + k * 32u + lane;
      rw[k] = i < total ? i / groups : 0u;
      cl[k] = i < total ? (i % groups) * 6u : 0u;
      if (i < total) n = k + 1;
    }
    ulonglong2 raw[N][W];
    float sum[N][6];
    bool ok = false;
    if (go) ok = hop::collect6<W, N>(s, c, x, rw, cl, n, raw, kGather ? nullptr : sum);
    if (!waited) {
      device::PDLWaitPrimary<kUsePDL>();
      waited = true;
    }
#pragma unroll
    for (int k = 0; k < N; ++k) {
      if (k >= n) continue;
      if constexpr (kGather) {
#pragma unroll
        for (int r = 0; r < W; ++r) {
          uint16_t* dst = out + (uint64_t(r) * rows + rw[k]) * s.width + cl[k];
#pragma unroll
          for (int e = 0; e < 6; ++e) dst[e] = ok ? uint16_t(hop::elem6(raw[k][r], e)) : hop::kPoisonBf16;
        }
      } else {
        uint16_t* dst = out + uint64_t(rw[k]) * s.width + cl[k];
#pragma unroll
        for (int e = 0; e < 6; ++e) dst[e] = ok ? uint16_t(bf16_rne(sum[k][e])) : hop::kPoisonBf16;
      }
    }
  }
  device::PDLWaitPrimary<kUsePDL>();
  __syncthreads();  // ends this CTA's reads of the slot
  if (threadIdx.x == 0 && go) hop::guard_exit(s, blockIdx.x, x.seq);
  if (threadIdx.x == 0 && s.policy == hop::kRing && (c.flags & hop::kAsyncCredit)) hop::credit_release_async(s, x, rows);
}

// ------------------------------------------------------------------------------------------------- host side
template <int W>
inline hop::Site<W> make_site(const tvm::ffi::TensorView table, int64_t rank, int64_t rows_max, int64_t width,
                              int64_t pitch, int64_t policy, int64_t parity, int64_t index, int64_t site, int64_t units) {
  using namespace host;
  RuntimeCheck(rank >= 0 && rank < W, "lb_hop: bad rank");
  RuntimeCheck(rows_max >= 1 && width >= 6 && width % 6 == 0, "lb_hop: rows_max >= 1, width a positive multiple of 6");
  RuntimeCheck(pitch % hop::kLineWords == 0 && pitch * 3 >= width, "lb_hop: pitch must hold the row in whole lines");
  const int64_t pol = policy & 0xff;
  RuntimeCheck(pol == hop::kUnique || pol == hop::kRing, "lb_hop: policy");
  RuntimeCheck((policy & ~int64_t(0xff | hop::kLoopbackDev)) == 0, "lb_hop: policy bits");
  RuntimeCheck(index >= 0 && units >= 1, "lb_hop: index / units");
  RuntimeCheck(table.numel() == 2 * W + 4 || table.numel() == 2 * W + 5, "lb_hop: site table size");
  // table (int64, host): recv[W], credit[W], seq64, ring_base, hwm, last_seq [, multicast address of the region]
  const auto* v = static_cast<const int64_t*>(table.data_ptr());
  hop::Site<W> s{};
  for (int r = 0; r < W; ++r) {
    s.recv[r] = reinterpret_cast<unsigned long long*>(v[r]);
    s.credit[r] = reinterpret_cast<uint32_t*>(v[W + r]);
  }
  s.seq64 = reinterpret_cast<const unsigned long long*>(v[2 * W]);
  s.ring_base = reinterpret_cast<unsigned long long*>(v[2 * W + 1]);
  s.hwm = reinterpret_cast<uint32_t*>(v[2 * W + 2]);
  s.last_seq = reinterpret_cast<unsigned long long*>(v[2 * W + 3]);
  s.mc = table.numel() == 2 * W + 5 ? reinterpret_cast<unsigned long long*>(v[2 * W + 4]) : nullptr;
  s.copy_words = static_cast<unsigned long long>(W) * rows_max * pitch;
  s.rank = uint32_t(rank);
  s.rows_max = uint32_t(rows_max);
  s.width = uint32_t(width);
  s.pitch = uint32_t(pitch);
  s.policy = uint32_t(pol);
  s.loop = (policy & hop::kLoopbackDev) ? 1u : 0u;
  s.format = hop::kTag64;
  s.parity = uint32_t(parity != 0);
  s.index = uint32_t(index);
  s.site = uint32_t(site);
  s.units = uint32_t(units);
  return s;
}

inline hop::Ctl make_ctl(const tvm::ffi::TensorView ctl_table, int64_t max_rounds, int64_t spin_rounds, int64_t flags,
                         int64_t backoff_ns) {
  const auto* v = static_cast<const int64_t*>(ctl_table.data_ptr());
  hop::Ctl c{};
  c.err_dev = reinterpret_cast<uint32_t*>(v[0]);
  c.err_host = reinterpret_cast<uint32_t*>(v[1]);
  c.max_rounds = static_cast<uint32_t>(max_rounds);
  c.spin_rounds = static_cast<uint32_t>(spin_rounds);
  c.flags = static_cast<uint32_t>(flags);
  c.backoff_ns = static_cast<uint32_t>(backoff_ns);
  return c;
}

template <int W>
inline hop::StepTok<W> make_tok(const tvm::ffi::TensorView tok_table, int64_t rank, int64_t site) {
  using namespace host;
  RuntimeCheck(tok_table.numel() == W + 1, "lb_hop: token table size");
  // table (int64, host): tok[W], seq64
  const auto* v = static_cast<const int64_t*>(tok_table.data_ptr());
  hop::StepTok<W> t{};
  for (int r = 0; r < W; ++r) t.tok[r] = reinterpret_cast<unsigned long long*>(v[r]);
  t.seq64 = reinterpret_cast<const unsigned long long*>(v[W]);
  t.rank = uint32_t(rank);
  t.site = uint32_t(site);
  return t;
}

}  // namespace muse_vr_lb_hop_ops

struct MuseVrLbHopSeq {
  // Forward head (no programmatic serialization).
  static void bump(const tvm::ffi::TensorView seq64) {
    using namespace host;
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({1}).with_dtype<int64_t>().with_device(device_).verify(seq64);
    LaunchKernel(dim3(1), dim3(32), device_.unwrap())(muse_vr_lb_hop_ops::lbhop_bump_kernel,
                                                      static_cast<unsigned long long*>(seq64.data_ptr()));
  }
};

template <int W>
struct MuseVrLbHopSeqRel {
  // Forward head + step_release (no programmatic serialization).
  static void bump_release(const tvm::ffi::TensorView seq64, const tvm::ffi::TensorView tok, int64_t rank) {
    using namespace host;
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({1}).with_dtype<int64_t>().with_device(device_).verify(seq64);
    const auto t = muse_vr_lb_hop_ops::make_tok<W>(tok, rank, 0xff);
    RuntimeCheck(reinterpret_cast<const void*>(t.seq64) == seq64.data_ptr(), "lb_hop: token table / seq64 mismatch");
    LaunchKernel(dim3(1), dim3(32), device_.unwrap())(muse_vr_lb_hop_ops::lbhop_bump_release_kernel<W>,
                                                      static_cast<unsigned long long*>(seq64.data_ptr()), t);
  }
};

template <bool kUsePDL>
struct MuseVrLbHopAdvance {
  static void run(const tvm::ffi::TensorView base, const tvm::ffi::TensorView counts) {
    using namespace host;
    namespace ho = muse_vr_lb_hop_ops;
    auto N = SymbolicSize{"num_hops"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({N}).with_dtype<int64_t>().with_device(device_).verify(base);
    const auto n = N.unwrap();
    RuntimeCheck(n >= 1 && n <= ho::kMaxHops, "hop count outside 1..16");
    RuntimeCheck(counts.numel() == n, "one count per hop");
    const auto* cv = static_cast<const int64_t*>(counts.data_ptr());
    ho::Counts k{};
    for (int64_t i = 0; i < n; ++i) k.add[i] = static_cast<unsigned long long>(cv[i]);
    LaunchKernel(dim3(1), dim3(32), device_.unwrap())
        .config({.use_pdl = kUsePDL})(ho::lbhop_advance_kernel<kUsePDL>,
                                      static_cast<unsigned long long*>(base.data_ptr()), uint32_t(n), k);
  }
};

template <int W, bool kUsePDL>
struct MuseVrLbHopRows {
  static void push(const tvm::ffi::TensorView table, const tvm::ffi::TensorView ctl, int64_t rank, int64_t rows_max,
                   int64_t width, int64_t pitch, int64_t policy, int64_t parity, int64_t index, int64_t site,
                   int64_t units, int64_t max_rounds, int64_t spin_rounds, int64_t flags, int64_t backoff_ns,
                   const tvm::ffi::TensorView in, int64_t blocks) {
    using namespace host;
    namespace ho = muse_vr_lb_hop_ops;
    auto R = SymbolicSize{"rows"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({R, width}).with_dtype<bf16_t>().with_device(device_).verify(in);
    RuntimeCheck(R.unwrap() >= 1 && R.unwrap() <= rows_max, "rows outside 1..rows_max");
    const auto s = ho::make_site<W>(table, rank, rows_max, width, pitch, policy, parity, index, site, units);
    const auto c = ho::make_ctl(ctl, max_rounds, spin_rounds, flags, backoff_ns);
    LaunchKernel(dim3(uint32_t(blocks)), dim3(256), device_.unwrap())
        .config({.use_pdl = kUsePDL})(ho::lbhop_push_rows_kernel<W, kUsePDL>, s, c, uint32_t(R.unwrap()),
                                      static_cast<const uint16_t*>(in.data_ptr()));
  }
  static void collect(const tvm::ffi::TensorView table, const tvm::ffi::TensorView ctl, int64_t rank, int64_t rows_max,
                      int64_t width, int64_t pitch, int64_t policy, int64_t parity, int64_t index, int64_t site,
                      int64_t units, int64_t max_rounds, int64_t spin_rounds, int64_t flags, int64_t backoff_ns,
                      const tvm::ffi::TensorView out, int64_t gather) {
    using namespace host;
    namespace ho = muse_vr_lb_hop_ops;
    auto R = SymbolicSize{"rows"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    if (gather) {
      TensorMatcher({int64_t(W), R, width}).with_dtype<bf16_t>().with_device(device_).verify(out);
    } else {
      TensorMatcher({R, width}).with_dtype<bf16_t>().with_device(device_).verify(out);
    }
    RuntimeCheck(R.unwrap() >= 1 && R.unwrap() <= rows_max, "rows outside 1..rows_max");
    const auto s = ho::make_site<W>(table, rank, rows_max, width, pitch, policy, parity, index, site, units);
    const auto c = ho::make_ctl(ctl, max_rounds, spin_rounds, flags, backoff_ns);
    const uint32_t blocks = uint32_t(units);  // one consumer unit per CTA
    if (gather) {
      LaunchKernel(dim3(blocks), dim3(256), device_.unwrap())
          .config({.use_pdl = kUsePDL})(ho::lbhop_collect_rows_kernel<W, kUsePDL, true>, s, c, uint32_t(R.unwrap()),
                                        static_cast<uint16_t*>(out.data_ptr()));
    } else {
      LaunchKernel(dim3(blocks), dim3(256), device_.unwrap())
          .config({.use_pdl = kUsePDL})(ho::lbhop_collect_rows_kernel<W, kUsePDL, false>, s, c, uint32_t(R.unwrap()),
                                        static_cast<uint16_t*>(out.data_ptr()));
    }
  }
};

template <int W, bool kUsePDL>
struct MuseVrLbHopTest {
  static void push(const tvm::ffi::TensorView table, const tvm::ffi::TensorView ctl, const tvm::ffi::TensorView tok,
                   int64_t rank, int64_t rows_max, int64_t width, int64_t pitch, int64_t policy, int64_t parity,
                   int64_t index, int64_t site, int64_t units, int64_t max_rounds, int64_t spin_rounds, int64_t flags,
                   int64_t backoff_ns, int64_t rows, int64_t fuzz, int64_t mode, int64_t tile_cols,
                   const tvm::ffi::TensorView audit, const tvm::ffi::TensorView bad, int64_t blocks, int64_t pdl) {
    using namespace host;
    namespace ho = muse_vr_lb_hop_ops;
    RuntimeCheck(rows >= 1 && rows <= rows_max, "rows outside 1..rows_max");
    RuntimeCheck(!(mode & ho::kModeTile) || tile_cols >= 6, "tile_cols >= 6");
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device_).verify(bad);
    const auto s = ho::make_site<W>(table, rank, rows_max, width, pitch, policy, parity, index, site, units);
    const auto c = ho::make_ctl(ctl, max_rounds, spin_rounds, flags, backoff_ns);
    const auto t = ho::make_tok<W>(tok, rank, site);
    uint32_t* au = audit.numel() > 0 ? static_cast<uint32_t*>(audit.data_ptr()) : nullptr;
    // pdl = 0: launched with a full stream dependency (the first kernel after the forward head)
    LaunchKernel(dim3(uint32_t(blocks)), dim3(256), device_.unwrap())
        .config({.use_pdl = kUsePDL && pdl != 0})(ho::lbhop_test_push_kernel<W, kUsePDL>, s, c, t, uint32_t(rows),
                                                  uint32_t(fuzz), uint32_t(mode), uint32_t(tile_cols), au,
                                                  static_cast<uint32_t*>(bad.data_ptr()));
  }
  static void collect(const tvm::ffi::TensorView table, const tvm::ffi::TensorView ctl, int64_t rank, int64_t rows_max,
                      int64_t width, int64_t pitch, int64_t policy, int64_t parity, int64_t index, int64_t site,
                      int64_t units, int64_t max_rounds, int64_t spin_rounds, int64_t flags, int64_t backoff_ns,
                      int64_t rows, int64_t fuzz, int64_t mode, const tvm::ffi::TensorView out,
                      const tvm::ffi::TensorView mismatch) {
    using namespace host;
    namespace ho = muse_vr_lb_hop_ops;
    RuntimeCheck(rows >= 1 && rows <= rows_max, "rows outside 1..rows_max");
    auto R = SymbolicSize{"rows"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({R, width}).with_dtype<fp32_t>().with_device(device_).verify(out);
    RuntimeCheck(R.unwrap() >= rows, "output has fewer rows than the transfer");
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device_).verify(mismatch);
    const auto s = ho::make_site<W>(table, rank, rows_max, width, pitch, policy, parity, index, site, units);
    const auto c = ho::make_ctl(ctl, max_rounds, spin_rounds, flags, backoff_ns);
    LaunchKernel(dim3(uint32_t(units)), dim3(256), device_.unwrap())
        .config({.use_pdl = kUsePDL})(ho::lbhop_test_collect_kernel<W, kUsePDL>, s, c, uint32_t(rows), uint32_t(fuzz),
                                      uint32_t(mode), static_cast<float*>(out.data_ptr()),
                                      static_cast<uint32_t*>(mismatch.data_ptr()));
  }
  static void release(const tvm::ffi::TensorView table, int64_t rank, int64_t rows_max, int64_t width, int64_t pitch,
                      int64_t policy, int64_t parity, int64_t index, int64_t site, int64_t units, int64_t rows,
                      const tvm::ffi::TensorView anchor) {
    using namespace host;
    namespace ho = muse_vr_lb_hop_ops;
    auto A = SymbolicSize{"anchor"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({A}).with_dtype<int32_t>().with_device(device_).verify(anchor);
    const auto s = ho::make_site<W>(table, rank, rows_max, width, pitch, policy, parity, index, site, units);
    LaunchKernel(dim3(1), dim3(32), device_.unwrap())
        .config({.use_pdl = kUsePDL})(ho::lbhop_test_release_kernel<W, kUsePDL>, s, uint32_t(rows));
  }
  static void late_write(const tvm::ffi::TensorView out, int64_t delay_units, double value) {
    using namespace host;
    namespace ho = muse_vr_lb_hop_ops;
    auto R = SymbolicSize{"rows"};
    auto C = SymbolicSize{"width"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({R, C}).with_dtype<fp32_t>().with_device(device_).verify(out);
    LaunchKernel(dim3(64), dim3(256), device_.unwrap())
        .config({.use_pdl = kUsePDL})(ho::lbhop_test_late_write_kernel<kUsePDL>, static_cast<float*>(out.data_ptr()),
                                      uint32_t(R.unwrap() * C.unwrap()), uint32_t(delay_units), float(value));
  }
};

}  // namespace sglang
