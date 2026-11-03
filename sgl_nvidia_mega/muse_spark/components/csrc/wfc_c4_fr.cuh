// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// FR: the split (RS) form's front launch for one GPU's owned rows (development form; not wired into the server).
//
// Inputs come from library calls launched before it: the router logits of the owned rows (fp32 [own_n, 256], the library
// router GEMM) and the pooled rows (bf16 [own_n, 4096], the library pooler GEMM). One CTA per owned row:
//   warp 0     routing of the row (muse_vr_front::route_row: SGLang's routing arithmetic), the 8 packed ids published
//              ids-first: word k = tag << 32 | packed (the expert stage's ids64 format) through the transport's ids path
//              (the ids travel only there, not in the token row);
//   warps 1-8  norm + NVFP4 quant of the pooled row (muse_vr_front::quant_row, the 256-thread group) into shared memory;
//   all        after a CTA barrier, the token row (fp4 payload, block scales, per-token scale) packed into 48-bit payload
//              words (the hop roles' byte format: payload in bits 47..0, a 16-bit tag in bits 63..48), each word stored once
//              through the transport's token path.
// Optional balance CTA (the run_bal entry; one extra CTA after the row CTAs): the per-step expert assignment over the replica
// pool (bal_pw.cuh, pairwise form) from the step's ids words of ALL rows (this launch's rows and the other ranks' rows, read
// where the expert stage reads them), published as 33 tagged words for the expert stage of this GPU. It is the only CTA of
// the launch that waits (bounded) on other CTAs' words; it stores no hop word.
// Memory model: every published word carries its tag in the same aligned 64-bit word as its payload (single-copy atomic per
// the PTX ISA memory consistency model) and has exactly one producer thread; consumers accept a word on its own matching load.
// The shared-memory staging is ordered by bar.sync before it is read. The row CTAs do not wait on other CTAs.
//
// Token row layout (bytes; proposal for the expert stage's gather): [0, 2048) the fp4 payload (the quant's linear order),
// [2048, 2304) the 256 e4m3 block scales (linear order), [2304, 2308) the per-token scale (fp32), [2308, 2320) zero -
// 2320 bytes (a 16-byte multiple) = 387 words.

#pragma once

#include "wfc_front_roles.cuh"
#include "wfc_c4_route.cuh"
#include "bal_pw.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace sglang {
namespace muse_vr_c4fr {

namespace fr = ::sglang::muse_vr_front;

inline constexpr int kHid = 4096;
inline constexpr int kOffSf = kHid / 2, kOffPts = kHid / 2 + kHid / 16;
inline constexpr int kRowBytes = 2320;                           // kOffPts + 4, padded to 16 bytes
inline constexpr int kRowWords = (kRowBytes + 5) / 6;             // 387
inline constexpr int kThreads = 32 + 256;
inline constexpr int kBarQ = 1, kBarAll = 0;
// stamp slots (timing builds only): [kSGo] CTA 0 start, [kSBalCnt] balance CTA counted, [kSBalAsg] assigned, [kSBalPub]
// published, [kSIds + row] row's ids words stored, [kSTok + row] row's token words stored, [kSQnt + row] row's quant group
// done (staged in shared memory; the group's last warp), [kSBeg + row] the row CTA's start, [kSBar + row] the staged-row
// barrier passed.
inline constexpr int kStampRows = 1024;
inline constexpr int kSGo = 0, kSBalCnt = 1, kSBalAsg = 2, kSBalPub = 3, kSIds = 8, kSTok = kSIds + kStampRows,
                     kSQnt = kSTok + kStampRows, kSBeg = kSQnt + kStampRows, kSBar = kSBeg + kStampRows,
                     kNumStamps = kSBar + kStampRows;
// Routing form (build flag; default 0 = route_row): MUSE_VR_C4FR_ROUTE_SEL = 1 / 2 selects route_row_sel<.., 1 / 2>, 3
// selects route_row_sel_sm (wfc_c4_route.cuh); their outputs equal route_row's for every input.
#ifndef MUSE_VR_C4FR_ROUTE_SEL
#define MUSE_VR_C4FR_ROUTE_SEL 0
#endif
// Token-word assembly (build flag; default 0 = byte loads): 1 = two aligned 8-byte shared loads per word, same words.
#ifndef MUSE_VR_C4FR_PUSH_VEC
#define MUSE_VR_C4FR_PUSH_VEC 0
#endif
// Quant group (build flag; default 0): 1 = the norm weight's loads issued with the row's loads (quant_row kPreW), same bytes.
#ifndef MUSE_VR_C4FR_QUANT_PREW
#define MUSE_VR_C4FR_QUANT_PREW 0
#endif

// Local transport (one GPU): the ids board [rows][8] and the token board [rows][pitch] of 64-bit words, relaxed stores at gpu
// scope. The hop roles' transport (multicast site) replaces it in the served form; the word formats are the same.
// Transport interface: tags(seq_word) runs once per thread at kernel start and returns the transfer's tags, derived on the
// device (here from the forward sequence word; never a kernel argument); ids_push / tok_push take them. needs_enter() (grid-
// uniform) asks for enter(tags), run by warp 0 of every CTA and followed by a CTA barrier before any push; a false result
// means this CTA stores no word.
struct LocalTransport {
  unsigned long long* ids;    // [rows][8] (row = own0 + owned-row index)
  unsigned long long* tok;    // [rows][pitch] (owned-row index)
  int pitch;
  int own0;                   // first owned row of the ids board (0 unless the board also holds other ranks' rows)
  struct Tags {
    uint32_t t32, t16;
  };
  __device__ __forceinline__ bool needs_enter() const { return false; }
  __device__ __forceinline__ bool enter(const Tags&) const { return true; }
  __device__ __forceinline__ Tags tags(const unsigned long long* seq_word) const {
    const unsigned long long seq = *seq_word;  // written before this launch (kernel boundary)
    return Tags{static_cast<uint32_t>(seq % 0xFFFFFFFFull) + 1u, static_cast<uint32_t>(seq % 65535ull) + 1u};
  }
  __device__ __forceinline__ void ids_push(const Tags& g, int row, int k, uint32_t packed) const {
    const unsigned long long v = (static_cast<unsigned long long>(g.t32) << 32) | packed;
    asm volatile("st.relaxed.gpu.global.b64 [%0], %1;" ::"l"(ids + (own0 + row) * 8 + k), "l"(v) : "memory");
  }
  // balance CTA: ONE load of word k of step row `row` (global index), no compiler memory barrier (the payload is in the word)
  __device__ __forceinline__ unsigned long long ids_load(const Tags&, int row, int k) const {
    unsigned long long v;
    asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];" : "=l"(v) : "l"(ids + row * 8 + k));
    return v;
  }
  __device__ __forceinline__ uint32_t ids_tag32(const Tags& g) const { return g.t32; }
  __device__ __forceinline__ void tok_push(const Tags& g, int row, int w, unsigned long long payload48) const {
    const unsigned long long v = (static_cast<unsigned long long>(g.t16) << 48) | (payload48 & 0xFFFFFFFFFFFFull);
    asm volatile("st.relaxed.gpu.global.b64 [%0], %1;" ::"l"(tok + static_cast<long long>(row) * pitch + w), "l"(v) : "memory");
  }
};

// The hop roles' transport (served form; compiled only where the hop headers are present): the ids site (kT32) and the token
// site (kB48) of the C3 hop API, rows addressed by their GLOBAL index (own0(M, W, rank) + owned-row index; the site's rows_max
// >= M). The transfers (copy, tags) are derived on the device from each site's sequence word by the hop API's xfer() at kernel
// start; the sequence-word argument of tags() is not used. Cross-forward order (the hop API's step tokens): with acquire = 0
// the caller orders FR after a kernel of the same forward that acquired the step tokens (through kernel completion, the hop
// API's precondition (2)), or uses parity sites; with acquire = 1 warp 0 of every CTA acquires them itself (step_acquire,
// then the CTA barrier) and a CTA whose acquire expired stores nothing (the sticky error word is set).
#if __has_include("c3_hops.cuh")
}  // namespace muse_vr_c4fr
}  // namespace sglang
#include "c3_hops.cuh"
namespace sglang {
namespace muse_vr_c4fr {
template <int W>
struct C3Transport {
  ::muse_vr::lb_hop::Site<W> ids_site, tok_site;
  ::muse_vr::lb_hop::StepTok<W> step;
  ::muse_vr::lb_hop::Ctl ctl;
  uint32_t M;        // rows of the step (fixed per captured shape)
  uint32_t acquire;  // 1: every CTA acquires the forward's step tokens before its first hop store
  struct Tags {
    ::muse_vr::lb_hop::Xfer ids, tok;
    uint32_t own0;
  };
  __device__ __forceinline__ Tags tags(const unsigned long long*) const {
    return Tags{::muse_vr::lb_hop::xfer<W>(ids_site), ::muse_vr::lb_hop::xfer<W>(tok_site),
                ::muse_vr::c3::own0(M, static_cast<uint32_t>(W), ids_site.rank)};
  }
  __device__ __forceinline__ bool needs_enter() const { return acquire != 0u; }
  __device__ __forceinline__ bool enter(const Tags& g) const {
    return ::muse_vr::lb_hop::step_acquire<W>(step, ctl, g.ids.seq);
  }
  __device__ __forceinline__ void ids_push(const Tags& g, int row, int k, uint32_t packed) const {
    ::muse_vr::c3::ids_push<W>(ids_site, g.ids, g.own0 + static_cast<uint32_t>(row), static_cast<uint32_t>(k), packed);
  }
  __device__ __forceinline__ void tok_push(const Tags& g, int row, int w, unsigned long long payload48) const {
    ::muse_vr::c3::agTok_push_word<W>(tok_site, g.tok, g.own0 + static_cast<uint32_t>(row), static_cast<uint32_t>(w),
                                      payload48);
  }
  // balance CTA: ONE load of word k of step row `row` from this GPU's copy of the ids site (where the expert stage reads it),
  // relaxed at the producers' scope, no compiler memory barrier (the payload is in the word)
  __device__ __forceinline__ unsigned long long ids_load(const Tags& g, int row, int k) const {
    const uint32_t src = ::muse_vr::c3::owner_of(M, static_cast<uint32_t>(W), static_cast<uint32_t>(row));
    const unsigned long long* p = ::muse_vr::lb_hop::word_ptr<W>(ids_site, ids_site.rank, g.ids.copy, src,
                                                                static_cast<uint32_t>(row), static_cast<uint32_t>(k));
    unsigned long long v;
    asm volatile("ld.relaxed.sys.global.b64 %0, [%1];" : "=l"(v) : "l"(p));
    return v;
  }
  __device__ __forceinline__ uint32_t ids_tag32(const Tags& g) const { return ::muse_vr::c3::ids_tag(g.ids); }
};
#endif

struct Params {
  const float* logits;          // [R][256] fp32 (library router GEMM output)
  const float* bias;            // [256]
  const void* pooled;           // [R][4096] bf16 (library pooler GEMM output)
  const void* qweight;          // [4096] bf16 norm weight or nullptr
  float* topk_w;                // [R][8] (local copies, as the served routing op writes them)
  int32_t* topk_ids;            // [R][8]
  int32_t* packed;              // [R][8]
  const unsigned long long* tag_word;  // local transport: the forward sequence word (device-incremented by the chain's
                                       // forward head; never a kernel argument): tag32 = seq mod (2^32 - 1) + 1,
                                       // tag16 = seq mod 65535 + 1 (the hop API's formulas); the hop transport derives its own
  int R;
  float mult, route_scale, div_eps, qeps, gsi;
  // balance CTA (run_bal only)
  const uint32_t* bal_tbl;         // [bal::kPwTableWords] static pair table
  unsigned long long* bal_board;   // [bal::kPwWords] this GPU's assignment words
  uint32_t* bal_err;               // [1] sticky
  int bal_M, bal_world, bal_sweeps, bal_variant, bal_maxm;
  uint32_t bal_pass_rows, bal_pass_inv, bal_spin, bal_max, bal_backoff;
  unsigned long long* stamps;      // timing builds only: [kNumStamps]
  // PDL form and the logits ready word (both default off: pdl = 0, lready = nullptr)
  int pdl;                             // 1: launched with programmatic stream serialization; every producer-written read
                                       //    (the sequence word, logits, pooled row) comes after griddepcontrol.wait
  const unsigned long long* lready;    // logits ready word or nullptr: written once per forward by the router's producer side
                                       //    (release) as (*tag_word) + 1; the routing warp polls it (acquire) before the logits
  uint32_t* err;                       // [1] sticky error word (bit 1: the logits ready word expired) or nullptr
  uint32_t lready_max;                 // poll rounds before expiry (0: a default)
};

#ifdef MUSE_VR_C4FR_STAMPS
#include "wfc_c4fr_stamps.cuh"  // timing builds only (not part of the package)
#else
#define FR_STAMP(lv) ((void)0)
#define FR_STAMP_WARP(lv) ((void)0)
#define FR_STAMP_CTA(lv) ((void)0)
#define FR_STAMP_MAX(lv) ((void)0)
#endif

// The balance CTA: count the step's ids words (all rows), assign, publish (bal_pw.cuh).
template <int kWorld, class Transport, class Tags>
__device__ __forceinline__ void bal_cta(const Params& p, const Transport& tr, const Tags& tg, ::muse_vr::bal::PwSmem& s) {
  namespace bl = ::muse_vr::bal;
  const int tid = static_cast<int>(threadIdx.x), nt = static_cast<int>(blockDim.x);
  bl::pw_stage(p.bal_tbl, s, tid, nt);
  __syncthreads();
  const auto ld = [&tr, &tg](int row, int k) { return tr.ids_load(tg, row, k); };
  const uint32_t tag32 = tr.ids_tag32(tg);
  bl::pw_count(ld, tag32, p.bal_M, s, tid, nt, p.bal_spin, p.bal_max, p.bal_backoff);
  __syncthreads();
  if (tid == 0) FR_STAMP(p.stamps[kSBalCnt]);
  const bool ok = (s.err & bl::kErrWait) == 0u;  // CTA-uniform (read after the barrier)
  if (ok) {
    unsigned long long* board = p.bal_board;
    const auto wr = [board](int i, unsigned long long v) {
      asm volatile("st.relaxed.gpu.global.b64 [%0], %1;" ::"l"(board + i), "l"(v) : "memory");
    };
    bl::pw_assign_any<kWorld>(p.bal_variant, p.bal_maxm, p.bal_pass_rows, p.bal_pass_inv, p.bal_sweeps, s, tid, wr, tag32);
  }
  if (tid < 32) {
    __syncwarp();
    if (tid == 0) {
      FR_STAMP(p.stamps[kSBalAsg]);
      FR_STAMP(p.stamps[kSBalPub]);
    }
  }
  if (tid == 0 && s.err != 0u) atomicOr(p.bal_err, s.err);
}

template <bool kHasW, int kBalW, class Transport>  // kBalW: 0 = no balance CTA, 4 / 8 = its world
__global__ void __launch_bounds__(kThreads, 1) fr_kernel(const __grid_constant__ Params p, const __grid_constant__ Transport tr) {
  __shared__ __align__(16) uint8_t row[kRowBytes + 16];
  __shared__ float qsm[16];
  __shared__ int s_ok;
  const int m = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
  if (m == 0 && tid == 0) FR_STAMP(p.stamps[kSGo]);
  if (p.pdl) {
    // PDL form: everything above is static; the producers' writes (sequence word, logits, pooled row) are read only after
    // the wait (PTX ISA griddepcontrol: the prerequisite grids completed and their memory operations are visible).
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }
  const auto tg = tr.tags(p.tag_word);  // device-derived transfer tags (see the transport interface)
  if constexpr (kBalW != 0) {
    if (m == p.R) {  // the balance CTA (CTA-uniform); one world per instantiation keeps the row CTAs' register budget
      __shared__ __align__(16) ::muse_vr::bal::PwSmem bs;
      bal_cta<kBalW>(p, tr, tg, bs);
      return;
    }
  }
  if (tid == 0) FR_STAMP(p.stamps[kSBeg + m]);
  bool ok = true;
  if (tr.needs_enter()) {  // grid-uniform
    if (warp == 0) {
      const bool e = tr.enter(tg);  // warp-uniform
      if (lane == 0) s_ok = e ? 1 : 0;
    }
    __syncthreads();  // the transport's entry (a step-token acquire) precedes every hop store of this CTA
    ok = s_ok != 0;
  }
  bool rok = true;  // routing warp: the logits are ready (always true without a ready word)
  if (warp == 0 && p.lready != nullptr) {
    int r = 1;
    if (lane == 0) {
      const unsigned long long want = *p.tag_word + 1ull;  // the forward's sequence word (read after the wait in PDL form)
      const uint32_t lim = p.lready_max ? p.lready_max : (1u << 22);
      uint32_t it = 0;
      unsigned long long v;
      while (true) {
        asm volatile("ld.acquire.gpu.global.b64 %0, [%1];" : "=l"(v) : "l"(p.lready) : "memory");
        if (v == want) break;
        if (++it >= lim) {
          r = 0;
          if (p.err != nullptr) atomicOr(p.err, 2u);
          break;
        }
        __nanosleep(32);
      }
    }
    rok = __shfl_sync(0xffffffffu, r, 0) != 0;
    __syncwarp();  // lane 0's acquire precedes every lane's logits loads (bar.warp.sync memory ordering)
  }
  if (warp == 0) {
    if (rok) {  // an expired ready word: no routing, no ids words (their consumers expire loudly; err bit 2 is set)
      fr::RouteArgs ra{};
      ra.logits = p.logits;
      ra.bias = p.bias;
      ra.topk_w = p.topk_w;
      ra.topk_ids = p.topk_ids;
      ra.packed = p.packed;
      ra.words = nullptr;
      ra.num_tokens = p.R;
      ra.num_experts = 256;
      ra.stride_gm = 256;
      ra.router_output_multiplier = p.mult;
      ra.route_scale = p.route_scale;
      ra.division_eps = p.div_eps;
      if constexpr (MUSE_VR_C4FR_ROUTE_SEL == 0) {
        fr::route_row<8, 8>(ra, m, lane, 0u);
      } else if constexpr (MUSE_VR_C4FR_ROUTE_SEL == 3) {
        __shared__ __align__(16) float rsc[256];  // the routing warp's score table (route_row_sel_sm)
        fr::route_row_sel_sm<8, 8>(ra, m, lane, 0u, rsc);
      } else {
        fr::route_row_sel<8, 8, MUSE_VR_C4FR_ROUTE_SEL>(ra, m, lane, 0u);
      }
      __syncwarp();  // the routing role's stores of this row (its own lanes) precede the reads below
      if (ok && lane < 8) tr.ids_push(tg, m, lane, static_cast<uint32_t>(p.packed[m * 8 + lane]));
      FR_STAMP_WARP(p.stamps[kSIds + m]);
    }
    if (lane < kRowBytes + 16 - kOffPts - 4) row[kOffPts + 4 + lane] = 0;  // the row's zero tail + the staging padding
  } else {
    // the input pointer is offset to this row and the role runs as row 0, so its outputs land at the staging buffer
    fr::QuantArgs qa{};
    qa.x = static_cast<const uint16_t*>(p.pooled) + static_cast<int64_t>(m) * kHid;
    qa.weight = kHasW ? p.qweight : nullptr;
    qa.fp4_out = row;                                         // [2048] fp4 payload (shared memory, generic pointer)
    qa.sf_out = row + kOffSf;                                 // [256] e4m3 block scales
    qa.per_token_scale = reinterpret_cast<float*>(row + kOffPts);
    qa.words = nullptr;
    qa.eps = p.qeps;
    qa.global_scale_inv = p.gsi;
    fr::quant_row<kHid, 256, kHasW, false, 0, 0, MUSE_VR_C4FR_QUANT_PREW != 0>(qa, 0, tid - 32, kBarQ, qsm, 0u);
    if (lane == 0) FR_STAMP_MAX(p.stamps[kSQnt + m]);  // timing builds: the group's last warp after its staging stores
  }
#ifndef MUSE_VR_C4FR_TEST_PLANT_RACE
  __syncthreads();  // the staged row (zero tail from warp 0, quant outputs from the group) is complete
#endif  // test-only build: the barrier dropped, a planted shared-memory race (sanitizer non-vacuity control)
  if (tid == 0) FR_STAMP(p.stamps[kSBar + m]);
  for (int w = tid; ok && w < kRowWords; w += kThreads) {
    unsigned long long v = 0ull;
    if constexpr (MUSE_VR_C4FR_PUSH_VEC != 0) {
      // bytes [6w, 6w + 6) from the two aligned 8-byte words around them (the staging row is 16-byte aligned and zero
      // past kRowBytes, so the last word reads only zero padding beyond the row)
      const int o = 6 * w, a8 = o & ~7, sh = (o & 7) * 8;
      const unsigned long long lo = *reinterpret_cast<const unsigned long long*>(row + a8);
      const unsigned long long hi = *reinterpret_cast<const unsigned long long*>(row + a8 + 8);
      v = (sh == 0 ? lo : ((lo >> sh) | (hi << (64 - sh)))) & 0xFFFFFFFFFFFFull;
    } else {
#pragma unroll
      for (int b = 0; b < 6; ++b) {
        const int i = 6 * w + b;
        if (i < kRowBytes) v |= static_cast<unsigned long long>(row[i]) << (8 * b);
      }
    }
    tr.tok_push(tg, m, w, v);
  }
  FR_STAMP_CTA(p.stamps[kSTok + m]);
}

__global__ void ready_kernel(unsigned long long* word, const unsigned long long* seq_word) {
  const unsigned long long v = *seq_word + 1ull;
  asm volatile("st.release.gpu.global.b64 [%0], %1;" ::"l"(word), "l"(v) : "memory");
}

// Host launch of the row grid: plain, or with programmatic stream serialization (the PDL form; p.pdl must be 1 then).
template <class Kern, class Tr>
inline void launch_rows(Kern kern, unsigned grid, cudaStream_t stream, const Params& p, const Tr& tr) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = p.pdl ? 1 : 0;
  host::RuntimeCheck(cudaLaunchKernelEx(&cfg, kern, p, tr) == cudaSuccess, "wfc_c4_fr: launch");
}

}  // namespace muse_vr_c4fr

struct WfcC4Fr {
  static void run(tvm::ffi::TensorView logits, tvm::ffi::TensorView bias, tvm::ffi::TensorView pooled,
                  tvm::ffi::TensorView qweight, int64_t has_w, tvm::ffi::TensorView topk_w, tvm::ffi::TensorView topk_ids,
                  tvm::ffi::TensorView packed, tvm::ffi::TensorView tag_word, tvm::ffi::TensorView ids_board,
                  tvm::ffi::TensorView tok_board, int64_t R, double mult, double route_scale, double div_eps, double qeps,
                  double gsi, int64_t pdl, tvm::ffi::Optional<tvm::ffi::TensorView> lready,
                  tvm::ffi::Optional<tvm::ffi::TensorView> err, int64_t lready_max) {
    using namespace host;
    namespace f = muse_vr_c4fr;
    RuntimeCheck(R >= 1 && logits.size(1) == 256 && pooled.size(1) == f::kHid, "wfc_c4_fr: shapes");
    RuntimeCheck(ids_board.numel() >= R * 8 && tok_board.size(1) >= f::kRowWords, "wfc_c4_fr: boards");
    f::Params p{};
    p.logits = static_cast<const float*>(logits.data_ptr());
    p.bias = static_cast<const float*>(bias.data_ptr());
    p.pooled = pooled.data_ptr();
    p.qweight = has_w ? qweight.data_ptr() : nullptr;
    p.topk_w = static_cast<float*>(topk_w.data_ptr());
    p.topk_ids = static_cast<int32_t*>(topk_ids.data_ptr());
    p.packed = static_cast<int32_t*>(packed.data_ptr());
    p.tag_word = static_cast<const unsigned long long*>(tag_word.data_ptr());
    p.R = static_cast<int>(R);
    p.mult = static_cast<float>(mult);
    p.route_scale = static_cast<float>(route_scale);
    p.div_eps = static_cast<float>(div_eps);
    p.qeps = static_cast<float>(qeps);
    p.gsi = static_cast<float>(gsi);
    p.pdl = pdl ? 1 : 0;
    p.lready = lready.has_value() ? static_cast<const unsigned long long*>(lready.value().data_ptr()) : nullptr;
    p.err = err.has_value() ? static_cast<uint32_t*>(err.value().data_ptr()) : nullptr;
    p.lready_max = static_cast<uint32_t>(lready_max);
    f::LocalTransport tr{static_cast<unsigned long long*>(ids_board.data_ptr()),
                         static_cast<unsigned long long*>(tok_board.data_ptr()), static_cast<int>(tok_board.size(1))};
    const cudaStream_t stream = LaunchKernel::resolve_device(logits.device());
    if (has_w)
      f::launch_rows(f::fr_kernel<true, 0, f::LocalTransport>, static_cast<unsigned>(R), stream, p, tr);
    else
      f::launch_rows(f::fr_kernel<false, 0, f::LocalTransport>, static_cast<unsigned>(R), stream, p, tr);
  }

  // The logits ready word's producer side (tests / C1's side stream, after the router GEMM): one thread stores
  // (*seq_word) + 1 with release semantics; the router GEMM's writes precede it by stream order.
  static void ready(tvm::ffi::TensorView word, tvm::ffi::TensorView seq_word) {
    using namespace host;
    const cudaStream_t stream = LaunchKernel::resolve_device(word.device());
    muse_vr_c4fr::ready_kernel<<<1, 1, 0, stream>>>(static_cast<unsigned long long*>(word.data_ptr()),
                                                    static_cast<const unsigned long long*>(seq_word.data_ptr()));
  }
};

// Development launch with the balance CTA (local transport): the ids board holds all M rows of the step (this launch's rows
// at [own0, own0 + R), the other rows written by the caller with the step's tag); the token board holds the R owned rows.
// grid = R row CTAs + 1 balance CTA. stamps (timing builds only) [kNumStamps] int64.
struct WfcC4FrBal {
  static void run(tvm::ffi::TensorView logits, tvm::ffi::TensorView bias, tvm::ffi::TensorView pooled,
                  tvm::ffi::TensorView qweight, int64_t has_w, tvm::ffi::TensorView topk_w, tvm::ffi::TensorView topk_ids,
                  tvm::ffi::TensorView packed, tvm::ffi::TensorView tag_word, tvm::ffi::TensorView ids_board,
                  tvm::ffi::TensorView tok_board, int64_t R, int64_t own0, double mult, double route_scale, double div_eps,
                  double qeps, double gsi, tvm::ffi::TensorView bal_tbl, tvm::ffi::TensorView bal_board,
                  tvm::ffi::TensorView bal_err, tvm::ffi::Optional<tvm::ffi::TensorView> stamps, int64_t M, int64_t world,
                  int64_t sweeps, int64_t pass_rows, int64_t spin_rounds, int64_t max_rounds, int64_t backoff_ns,
                  int64_t bal_cta, int64_t variant, int64_t coop, int64_t maxm) {
    using namespace host;
    namespace f = muse_vr_c4fr;
    namespace bl = ::muse_vr::bal;
    RuntimeCheck(R >= 1 && logits.size(1) == 256 && pooled.size(1) == f::kHid, "wfc_c4_fr bal: shapes");
    RuntimeCheck(M >= R && M <= bl::kPwMaxRows && own0 >= 0 && own0 + R <= M && M * 8 <= 32 * f::kThreads,
                 "wfc_c4_fr bal: M / own0");
    RuntimeCheck(ids_board.numel() >= M * 8 && tok_board.size(0) >= R && tok_board.size(1) >= f::kRowWords,
                 "wfc_c4_fr bal: boards");
    RuntimeCheck(bal_tbl.numel() >= bl::kPwTableWords && bal_board.numel() >= bl::kPwWords && bal_err.numel() >= 1,
                 "wfc_c4_fr bal: pair table / board / err");
    RuntimeCheck((world == 4 || world == 8) && sweeps >= 0 && sweeps <= 8 && pass_rows >= 1 && pass_rows <= 128,
                 "wfc_c4_fr bal: world / sweeps / pass_rows");
    f::Params p{};
    p.logits = static_cast<const float*>(logits.data_ptr());
    p.bias = static_cast<const float*>(bias.data_ptr());
    p.pooled = pooled.data_ptr();
    p.qweight = has_w ? qweight.data_ptr() : nullptr;
    p.topk_w = static_cast<float*>(topk_w.data_ptr());
    p.topk_ids = static_cast<int32_t*>(topk_ids.data_ptr());
    p.packed = static_cast<int32_t*>(packed.data_ptr());
    p.tag_word = static_cast<const unsigned long long*>(tag_word.data_ptr());
    p.R = static_cast<int>(R);
    p.mult = static_cast<float>(mult);
    p.route_scale = static_cast<float>(route_scale);
    p.div_eps = static_cast<float>(div_eps);
    p.qeps = static_cast<float>(qeps);
    p.gsi = static_cast<float>(gsi);
    p.bal_tbl = static_cast<const uint32_t*>(bal_tbl.data_ptr());
    p.bal_board = static_cast<unsigned long long*>(bal_board.data_ptr());
    p.bal_err = static_cast<uint32_t*>(bal_err.data_ptr());
    p.bal_M = static_cast<int>(M);
    p.bal_world = static_cast<int>(world);
    p.bal_sweeps = static_cast<int>(sweeps);
    RuntimeCheck(variant >= 0 && variant <= 2 && maxm >= 1 && maxm <= bl::kPwMaxMembers && (variant != 2 || maxm <= 16),
                 "wfc_c4_fr bal: variant / maxm");
    p.bal_variant = static_cast<int>(variant);
    p.bal_maxm = static_cast<int>(maxm);
    p.bal_pass_rows = static_cast<uint32_t>(pass_rows);
    p.bal_pass_inv = static_cast<uint32_t>((65536 + pass_rows - 1) / pass_rows);
    for (int64_t x = 0; x <= bl::kPwMaxRows + pass_rows; ++x) {
      RuntimeCheck(((uint64_t(x) * p.bal_pass_inv) >> 16) == uint64_t(x / pass_rows),
                   "wfc_c4_fr bal: pass_rows reciprocal not exact");
    }
    p.bal_spin = static_cast<uint32_t>(spin_rounds);
    p.bal_max = static_cast<uint32_t>(max_rounds);
    p.bal_backoff = static_cast<uint32_t>(backoff_ns);
    p.stamps = nullptr;
    if (stamps.has_value()) {
      RuntimeCheck(stamps.value().numel() >= f::kNumStamps && M <= f::kStampRows, "wfc_c4_fr bal: stamps");
      p.stamps = static_cast<unsigned long long*>(stamps.value().data_ptr());
    }
#ifdef MUSE_VR_C4FR_STAMPS
    RuntimeCheck(p.stamps != nullptr, "wfc_c4_fr bal: timing build needs stamps");
#endif
    f::LocalTransport tr{static_cast<unsigned long long*>(ids_board.data_ptr()),
                         static_cast<unsigned long long*>(tok_board.data_ptr()), static_cast<int>(tok_board.size(1)),
                         static_cast<int>(own0)};
    const cudaStream_t stream = LaunchKernel::resolve_device(logits.device());
    // the balance CTA waits on the row CTAs of the same grid: the whole grid must be co-resident
    auto* kern = world == 8 ? (has_w ? f::fr_kernel<true, 8, f::LocalTransport> : f::fr_kernel<false, 8, f::LocalTransport>)
                            : (has_w ? f::fr_kernel<true, 4, f::LocalTransport> : f::fr_kernel<false, 4, f::LocalTransport>);
    int per_sm = 0, sms = 0, dev = 0;
    RuntimeCheck(cudaGetDevice(&dev) == cudaSuccess &&
                     cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, f::kThreads, 0) == cudaSuccess &&
                     cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess,
                 "wfc_c4_fr bal: occupancy query");
    const int64_t grid = R + (bal_cta != 0 ? 1 : 0);  // bal_cta = 0: the same build without the balance CTA (A/B arms)
    RuntimeCheck(static_cast<int64_t>(per_sm) * sms >= grid, "wfc_c4_fr bal: grid not co-resident");
    // cooperative launch: the runtime guarantees co-residency of the whole grid (or fails the launch)
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(static_cast<unsigned>(grid));
    cfg.blockDim = dim3(f::kThreads);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeCooperative;
    attr[0].val.cooperative = 1;
    cfg.attrs = attr;
    cfg.numAttrs = coop != 0 ? 1 : 0;  // coop = 0: a plain launch after the co-residency check above
    RuntimeCheck(cudaLaunchKernelEx(&cfg, kern, p, tr) == cudaSuccess, "wfc_c4_fr bal: launch");
  }
};

// Host launch with the hop roles' transport (compiled only where the hop host header is present): the ids and token sites are
// unique sites of the caller's hop arena (tables and site arguments as HopArena.push_table / site_args give them), the step
// tokens and Ctl as the arena gives them. M = the step's rows; this rank's owned rows are [own0, own1) of M (R = own1 - own0).
#if __has_include("lb_hop_ops.cuh")
}  // namespace sglang
#include "lb_hop_ops.cuh"  // make_site / make_ctl / make_tok
namespace sglang {
template <int W>
struct WfcC4FrC3 {
  static void run(tvm::ffi::TensorView logits, tvm::ffi::TensorView bias, tvm::ffi::TensorView pooled,
                  tvm::ffi::TensorView qweight, int64_t has_w, tvm::ffi::TensorView topk_w, tvm::ffi::TensorView topk_ids,
                  tvm::ffi::TensorView packed, tvm::ffi::TensorView ids_table, tvm::ffi::TensorView tok_table,
                  tvm::ffi::TensorView step_table, tvm::ffi::TensorView ctl_table, int64_t rank, int64_t ids_rows_max,
                  int64_t ids_width, int64_t ids_pitch, int64_t ids_policy, int64_t ids_parity, int64_t ids_site,
                  int64_t tok_rows_max, int64_t tok_width, int64_t tok_pitch, int64_t tok_policy, int64_t tok_parity,
                  int64_t tok_site, int64_t max_rounds, int64_t spin_rounds, int64_t flags, int64_t backoff_ns,
                  int64_t acquire, int64_t M, double mult, double route_scale, double div_eps, double qeps, double gsi,
                  int64_t pdl, tvm::ffi::Optional<tvm::ffi::TensorView> lready, tvm::ffi::Optional<tvm::ffi::TensorView> err,
                  int64_t lready_max) {
    using namespace host;
    namespace f = muse_vr_c4fr;
    namespace lho = muse_vr_lb_hop_ops;
    const int64_t own0 = (M * rank) / W, own1 = (M * (rank + 1)) / W, R = own1 - own0;
    RuntimeCheck(M >= 1 && rank >= 0 && rank < W, "wfc_c4_fr c3: M / rank");
    RuntimeCheck(M <= ids_rows_max && M <= tok_rows_max, "wfc_c4_fr c3: M <= the sites' rows_max (rows are global)");
    RuntimeCheck(ids_pitch >= 8 && tok_pitch >= f::kRowWords, "wfc_c4_fr c3: site pitch (8 ids words, 387 token words)");
    RuntimeCheck((ids_policy & 0xff) == muse_vr::lb_hop::kUnique && (tok_policy & 0xff) == muse_vr::lb_hop::kUnique,
                 "wfc_c4_fr c3: unique sites");
    RuntimeCheck(logits.size(0) == R && pooled.size(0) == R && logits.size(1) == 256 && pooled.size(1) == f::kHid,
                 "wfc_c4_fr c3: logits / pooled hold the owned rows");
    if (R == 0) return;
    f::C3Transport<W> tr{};
    tr.ids_site = lho::make_site<W>(ids_table, rank, ids_rows_max, ids_width, ids_pitch, ids_policy, ids_parity, 0, ids_site, 1);
    tr.tok_site = lho::make_site<W>(tok_table, rank, tok_rows_max, tok_width, tok_pitch, tok_policy, tok_parity, 0, tok_site, 1);
    tr.step = lho::make_tok<W>(step_table, rank, ids_site);
    tr.ctl = lho::make_ctl(ctl_table, max_rounds, spin_rounds, flags, backoff_ns);
    tr.M = static_cast<uint32_t>(M);
    tr.acquire = acquire ? 1u : 0u;
    f::Params p{};
    p.logits = static_cast<const float*>(logits.data_ptr());
    p.bias = static_cast<const float*>(bias.data_ptr());
    p.pooled = pooled.data_ptr();
    p.qweight = has_w ? qweight.data_ptr() : nullptr;
    p.topk_w = static_cast<float*>(topk_w.data_ptr());
    p.topk_ids = static_cast<int32_t*>(topk_ids.data_ptr());
    p.packed = static_cast<int32_t*>(packed.data_ptr());
    p.tag_word = tr.ids_site.seq64;  // the hop transport derives its transfers from the sites; the ready word uses this
    p.R = static_cast<int>(R);
    p.mult = static_cast<float>(mult);
    p.route_scale = static_cast<float>(route_scale);
    p.div_eps = static_cast<float>(div_eps);
    p.qeps = static_cast<float>(qeps);
    p.gsi = static_cast<float>(gsi);
    p.pdl = pdl ? 1 : 0;
    p.lready = lready.has_value() ? static_cast<const unsigned long long*>(lready.value().data_ptr()) : nullptr;
    p.err = err.has_value() ? static_cast<uint32_t*>(err.value().data_ptr()) : nullptr;
    p.lready_max = static_cast<uint32_t>(lready_max);
    const cudaStream_t stream = LaunchKernel::resolve_device(logits.device());
    if (has_w)
      f::launch_rows(f::fr_kernel<true, 0, f::C3Transport<W>>, static_cast<unsigned>(R), stream, p, tr);
    else
      f::launch_rows(f::fr_kernel<false, 0, f::C3Transport<W>>, static_cast<unsigned>(R), stream, p, tr);
  }
};

// The same launch plus the balance CTA (bal_pw.cuh): R row CTAs + 1 (the balance CTA runs even when R == 0).
template <int W>
struct WfcC4FrC3Bal {
  static void run(tvm::ffi::TensorView logits, tvm::ffi::TensorView bias, tvm::ffi::TensorView pooled,
                  tvm::ffi::TensorView qweight, int64_t has_w, tvm::ffi::TensorView topk_w, tvm::ffi::TensorView topk_ids,
                  tvm::ffi::TensorView packed, tvm::ffi::TensorView ids_table, tvm::ffi::TensorView tok_table,
                  tvm::ffi::TensorView step_table, tvm::ffi::TensorView ctl_table, int64_t rank, int64_t ids_rows_max,
                  int64_t ids_width, int64_t ids_pitch, int64_t ids_policy, int64_t ids_parity, int64_t ids_site,
                  int64_t tok_rows_max, int64_t tok_width, int64_t tok_pitch, int64_t tok_policy, int64_t tok_parity,
                  int64_t tok_site, int64_t max_rounds, int64_t spin_rounds, int64_t flags, int64_t backoff_ns,
                  int64_t acquire, int64_t M, double mult, double route_scale, double div_eps, double qeps, double gsi,
                  tvm::ffi::TensorView bal_tbl, tvm::ffi::TensorView bal_board, tvm::ffi::TensorView bal_err,
                  int64_t sweeps, int64_t pass_rows, int64_t bal_spin_rounds, int64_t bal_max_rounds,
                  int64_t bal_backoff_ns, int64_t bal_world, int64_t maxm) {
    using namespace host;
    namespace f = muse_vr_c4fr;
    namespace lho = muse_vr_lb_hop_ops;
    namespace bl = ::muse_vr::bal;
    const int64_t own0 = (M * rank) / W, own1 = (M * (rank + 1)) / W, R = own1 - own0;
    RuntimeCheck(M >= 1 && rank >= 0 && rank < W, "wfc_c4_fr c3: M / rank");
    RuntimeCheck(M <= ids_rows_max && M <= tok_rows_max, "wfc_c4_fr c3: M <= the sites' rows_max (rows are global)");
    RuntimeCheck(ids_pitch >= 8 && tok_pitch >= f::kRowWords, "wfc_c4_fr c3: site pitch (8 ids words, 387 token words)");
    RuntimeCheck((ids_policy & 0xff) == muse_vr::lb_hop::kUnique && (tok_policy & 0xff) == muse_vr::lb_hop::kUnique,
                 "wfc_c4_fr c3: unique sites");
    RuntimeCheck(logits.size(0) == R && pooled.size(0) == R && logits.size(1) == 256 && pooled.size(1) == f::kHid,
                 "wfc_c4_fr c3: logits / pooled hold the owned rows");
    RuntimeCheck(M <= bl::kPwMaxRows && M * 8 <= 32 * f::kThreads, "wfc_c4_fr c3 bal: M");
    RuntimeCheck(bal_tbl.numel() >= bl::kPwTableWords && bal_board.numel() >= bl::kPwWords && bal_err.numel() >= 1,
                 "wfc_c4_fr c3 bal: pair table / board / err");
    RuntimeCheck(sweeps >= 0 && sweeps <= 8 && pass_rows >= 1 && pass_rows <= 128, "wfc_c4_fr c3 bal: sweeps / pass_rows");
    // the assignment's world = the hop world in served use; a world-1 arena may emulate a 4- or 8-rank assignment (tests)
    RuntimeCheck((bal_world == 4 || bal_world == 8) && (W == 1 || bal_world == W), "wfc_c4_fr c3 bal: assignment world");
    // R == 0 still launches the balance CTA: every rank publishes its own copy of the assignment
    f::C3Transport<W> tr{};
    tr.ids_site = lho::make_site<W>(ids_table, rank, ids_rows_max, ids_width, ids_pitch, ids_policy, ids_parity, 0, ids_site, 1);
    tr.tok_site = lho::make_site<W>(tok_table, rank, tok_rows_max, tok_width, tok_pitch, tok_policy, tok_parity, 0, tok_site, 1);
    tr.step = lho::make_tok<W>(step_table, rank, ids_site);
    tr.ctl = lho::make_ctl(ctl_table, max_rounds, spin_rounds, flags, backoff_ns);
    tr.M = static_cast<uint32_t>(M);
    tr.acquire = acquire ? 1u : 0u;
    f::Params p{};
    p.logits = static_cast<const float*>(logits.data_ptr());
    p.bias = static_cast<const float*>(bias.data_ptr());
    p.pooled = pooled.data_ptr();
    p.qweight = has_w ? qweight.data_ptr() : nullptr;
    p.topk_w = static_cast<float*>(topk_w.data_ptr());
    p.topk_ids = static_cast<int32_t*>(topk_ids.data_ptr());
    p.packed = static_cast<int32_t*>(packed.data_ptr());
    p.tag_word = tr.ids_site.seq64;  // not read by the hop transport (it derives its transfers from the sites)
    p.R = static_cast<int>(R);
    p.mult = static_cast<float>(mult);
    p.route_scale = static_cast<float>(route_scale);
    p.div_eps = static_cast<float>(div_eps);
    p.qeps = static_cast<float>(qeps);
    p.gsi = static_cast<float>(gsi);
    p.bal_tbl = static_cast<const uint32_t*>(bal_tbl.data_ptr());
    p.bal_board = static_cast<unsigned long long*>(bal_board.data_ptr());
    p.bal_err = static_cast<uint32_t*>(bal_err.data_ptr());
    p.bal_M = static_cast<int>(M);
    p.bal_world = static_cast<int>(bal_world);
    p.bal_sweeps = static_cast<int>(sweeps);
    RuntimeCheck(maxm >= 1 && maxm <= 16, "wfc_c4_fr c3 bal: maxm");
    p.bal_variant = 2;
    p.bal_maxm = static_cast<int>(maxm);
    p.bal_pass_rows = static_cast<uint32_t>(pass_rows);
    p.bal_pass_inv = static_cast<uint32_t>((65536 + pass_rows - 1) / pass_rows);
    for (int64_t x = 0; x <= bl::kPwMaxRows + pass_rows; ++x) {
      RuntimeCheck(((uint64_t(x) * p.bal_pass_inv) >> 16) == uint64_t(x / pass_rows),
                   "wfc_c4_fr c3 bal: pass_rows reciprocal not exact");
    }
    p.bal_spin = static_cast<uint32_t>(bal_spin_rounds);
    p.bal_max = static_cast<uint32_t>(bal_max_rounds);
    p.bal_backoff = static_cast<uint32_t>(bal_backoff_ns);
    const cudaStream_t stream = LaunchKernel::resolve_device(logits.device());
    // the balance CTA waits on the row CTAs of the same grid: the whole grid must be co-resident
    auto* kern = bal_world == 8
                     ? (has_w ? f::fr_kernel<true, 8, f::C3Transport<W>> : f::fr_kernel<false, 8, f::C3Transport<W>>)
                     : (has_w ? f::fr_kernel<true, 4, f::C3Transport<W>> : f::fr_kernel<false, 4, f::C3Transport<W>>);
    int per_sm = 0, sms = 0, dev = 0;
    RuntimeCheck(cudaGetDevice(&dev) == cudaSuccess &&
                     cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, f::kThreads, 0) == cudaSuccess &&
                     cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess,
                 "wfc_c4_fr c3 bal: occupancy query");
    RuntimeCheck(static_cast<int64_t>(per_sm) * sms >= R + 1, "wfc_c4_fr c3 bal: grid not co-resident");
    // cooperative launch: the runtime guarantees co-residency of the whole grid (or fails the launch)
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(static_cast<unsigned>(R + 1));
    cfg.blockDim = dim3(f::kThreads);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeCooperative;
    attr[0].val.cooperative = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    RuntimeCheck(cudaLaunchKernelEx(&cfg, kern, p, tr) == cudaSuccess, "wfc_c4_fr c3 bal: cooperative launch");
  }
};
#endif

}  // namespace sglang
