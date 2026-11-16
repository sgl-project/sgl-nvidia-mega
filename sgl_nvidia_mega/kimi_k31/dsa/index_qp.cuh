// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// K3.1 DSA index query, query-front form, 1-8 rows: the IQ-PIPE form of index_q.cuh's index_q_kernel<8, 12, kQF = true, 8, kKFP4, false>
// (K31MK_DSA_IQ_PIPE). Same structure, inputs, outputs and arithmetic, bit for bit (warp w normalizes q_a row w in index_q.cuh's
// lane order into SMEM; 12 m16n8k16 steps per warp in the same K order; fixed-order sum over the 8 K slices; cluster amax exchange; ue8m0 scale;
// FP8 codes; index-K store). What differs is only how the operands reach the instructions:
//   (1) w_qa is read as one 16-byte vector per 8 elements (index_q.cuh: eight 2-byte loads, 4-way bank conflicted);
//   (2) the normalized BF16 values are packed in pairs (one cvt per two elements; same round-to-nearest results);
//   (3) the wq_b A fragments of each warp's K slice are read from the staged SMEM rows into registers BEFORE griddepcontrol.wait (weights only),
//       so after the wait the MMA loop issues only the B-fragment SMEM loads.
//  A profile of index_q put a large share of its time in the normalize + SMEM store loop and a smaller one in
// the MMA loop; this form ends the 8-row query earlier, standalone and in the MID OWNQ chain. A variant
// that loaded each lane's B-fragment elements of q_a straight from global (no SMEM round trip) was bitwise but not faster.
// 9-16 rows (K31MK_DSA_QF_MAX_ROWS=16) run index_q.cuh's kR = 16 kernel unchanged (run_qf below).
#include "index_q.cuh"

namespace k31_index_qp {
using namespace k31_index_q;


// bf16(float(x) * rr * float(w)) for the two BF16 values of x (low half = lower column), as index_q.cuh's RMSNorm store does per element.
__device__ __forceinline__ uint32_t norm_pair(uint32_t x, uint32_t w, float rr) {
  const float x0 = __uint_as_float(x << 16), x1 = __uint_as_float(x & 0xffff0000u);
  const float w0 = __uint_as_float(w << 16), w1 = __uint_as_float(w & 0xffff0000u);
  const __nv_bfloat162 o = __floats2bfloat162_rn(x0 * rr * w0, x1 * rr * w1);  // .x (low half) = first argument
  return *reinterpret_cast<const uint32_t*>(&o);
}

struct SmemP {
  alignas(16) uint8_t wq[16][kStride];
  alignas(16) uint8_t x[kRows][kStride];
  alignas(16) __nv_bfloat16 wqa[kLora];
  float part[8][16][kRows];
  float val[16][kRows];
  float ex_amax[8][kRows];
  float scale[kRows];
};

template <bool kKFP4>
__global__ void __launch_bounds__(kThreads, 2) iqp_kernel(const __grid_constant__ Params p) {
  constexpr int kSplit = 8, kOwn = 16, kR = kRows, kKs = kLora / 8 / 16;
  extern __shared__ __align__(16) uint8_t smem_raw[];
  SmemP& s = *reinterpret_cast<SmemP*>(smem_raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, qd = lane & 3;
  const int head = blockIdx.x / kSplit;
  const uint32_t rank = cluster_rank();
  const int row0 = head * kDim + static_cast<int>(rank) * kOwn;
  cluster_arrive_relaxed();
  for (int i = tid; i < kOwn * (kLora / 8); i += kThreads) {
    const int r = i / (kLora / 8), c = i % (kLora / 8);
    cp_async16(&s.wq[r][c * 16], p.wq_b + static_cast<int64_t>(row0 + r) * kLora + c * 8);
  }
  for (int i = tid; i < kLora / 8; i += kThreads) cp_async16(&s.wqa[i * 8], p.w_qa + i * 8);
  cp_async_commit();
  cp_async_wait_all();
  __syncthreads();
  auto wq = [&](int r, int k) { return *reinterpret_cast<const uint32_t*>(&s.wq[r][k * 2]); };
  uint32_t af[kKs][4];
#pragma unroll
  for (int ks = 0; ks < kKs; ++ks) {
    const int k0 = warp * (kLora / 8) + ks * 16;
    af[ks][0] = wq(g, k0 + 2 * qd);
    af[ks][1] = wq(g + 8, k0 + 2 * qd);
    af[ks][2] = wq(g, k0 + 8 + 2 * qd);
    af[ks][3] = wq(g + 8, k0 + 8 + 2 * qd);
  }
  constexpr int kQaChunks = kLora / 8 / 32;
  uint4 xa[kQaChunks];
  const uint4* xsrc = reinterpret_cast<const uint4*>(p.projected + static_cast<int64_t>(warp) * p.pitch);
  pdl_wait();
  pdl_trigger();
#pragma unroll
  for (int i = 0; i < kQaChunks; ++i) xa[i] = warp < p.rows ? xsrc[i * 32 + lane] : make_uint4(0, 0, 0, 0);  // warp w's q_a row, index_q.cuh's lane order
  float gq = 0.f;
  uint2 kraw = {0u, 0u};
  int64_t kslot = -1;
  const bool k_owner = rank == kSplit - 1 && warp == 7 && head < p.rows;
  if (rank == 0 && tid < p.rows) gq = __bfloat162float(p.projected[static_cast<int64_t>(tid) * p.pitch + p.g_col + head]);
  if (k_owner) {
    kraw = reinterpret_cast<const uint2*>(p.projected + static_cast<int64_t>(head) * p.pitch + p.k_col)[lane];
    kslot = p.slots[head];
  }
  {  // RMSNorm of row `warp` (index_q.cuh's lane order and rounding), stored to SMEM
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < kQaChunks; ++i) {
      const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&xa[i]);
#pragma unroll
      for (int j = 0; j < 8; ++j) ss = fmaf(__bfloat162float(e[j]), __bfloat162float(e[j]), ss);
    }
#pragma unroll
    for (int d = 16; d; d >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, d);
    const float rr = rsqrtf(ss / static_cast<float>(kLora) + p.eps);
    const bool rlive = warp < p.rows;
#pragma unroll
    for (int i = 0; i < kQaChunks; ++i) {
      const int k0 = (i * 32 + lane) * 8;
      const uint4 wv = *reinterpret_cast<const uint4*>(&s.wqa[k0]);
      const uint32_t* xw = reinterpret_cast<const uint32_t*>(&xa[i]);
      const uint32_t* ww = reinterpret_cast<const uint32_t*>(&wv);
      uint4 o;
      uint32_t* ow = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
      for (int j = 0; j < 4; ++j) ow[j] = rlive ? norm_pair(xw[j], ww[j], rr) : 0u;
      *reinterpret_cast<uint4*>(&s.x[warp][k0 * 2]) = o;
    }
  }
  __syncthreads();
  float d[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int ks = 0; ks < kKs; ++ks) {
    const int k0 = warp * (kLora / 8) + ks * 16;
    auto xs = [&](int r, int k) { return *reinterpret_cast<const uint32_t*>(&s.x[r][k * 2]); };
    mma_bf16(d, af[ks][0], af[ks][1], af[ks][2], af[ks][3], xs(g, k0 + 2 * qd), xs(g, k0 + 8 + 2 * qd));
  }
  s.part[warp][g][2 * qd] = d[0];
  s.part[warp][g][2 * qd + 1] = d[1];
  s.part[warp][g + 8][2 * qd] = d[2];
  s.part[warp][g + 8][2 * qd + 1] = d[3];
  __syncthreads();
  if (tid < kOwn * kR) {
    const int r = tid / kR, b = tid % kR;
    float acc = 0.f;
#pragma unroll
    for (int w8 = 0; w8 < 8; ++w8) acc += s.part[w8][r][b];
    s.val[r][b] = bf16_round(acc);
  }
  __syncthreads();
  cluster_wait_acquire();
  if (tid < kR) {
    const int b = tid;
    float amax = 0.f;
#pragma unroll
    for (int r = 0; r < kOwn; ++r) amax = fmaxf(amax, fabsf(s.val[r][b]));
#pragma unroll
    for (int c = 0; c < kSplit; ++c) st_peer_f32(&s.ex_amax[rank][b], c, amax);
  }
  cluster_arrive_release();
  if (k_owner) store_index_k<kKFP4>(p, kraw, kslot, lane);
  cluster_wait_acquire();
  if (tid < kR) {
    const int b = tid;
    float amax = 0.f;
#pragma unroll
    for (int c = 0; c < kSplit; ++c) amax = fmaxf(amax, s.ex_amax[c][b]);
    s.scale[b] = ue8m0_scale(amax);
    if (rank == 0 && b < p.rows) {
      const float g2 = bf16_round(bf16_round(gq) * 0.17677669529663687f);
      p.w[b * kHeads + head] = g2 * s.scale[b] * 0.08838834764831845f;
    }
  }
  __syncthreads();
  if (tid < kOwn * kR) {
    const int r = tid % kOwn, b = tid / kOwn;
    if (b < p.rows) {
      const float y = fminf(fmaxf(div_full(s.val[r][b], s.scale[b]), -448.f), 448.f);
      p.q8[(static_cast<int64_t>(b) * kHeads + head) * kDim + rank * kOwn + r] = to_e4m3(y);
    }
  }
}

struct BindingP {
  // index_q.cuh Binding::run_qf's contract (same argument list and checks); 1-8 rows take iqp_kernel, 9-16 rows index_q.cuh's kR = 16 kernel.
  static void run_qf(const tvm::ffi::TensorView projected, const tvm::ffi::TensorView w_qa, const tvm::ffi::TensorView wq_b,
                     const tvm::ffi::TensorView q8, const tvm::ffi::TensorView w, const double eps, const int64_t k_col, const int64_t g_col,
                     const tvm::ffi::TensorView slots, const tvm::ffi::TensorView kcache, const tvm::ffi::TensorView kgamma,
                     const tvm::ffi::TensorView kbeta, const double keps) {
    using namespace sglang::host;
    const int64_t rows = projected.size(0);
    if (rows > kRows) {
      Binding::run_qf(projected, w_qa, wq_b, q8, w, eps, k_col, g_col, slots, kcache, kgamma, kbeta, keps);
      return;
    }
    RuntimeCheck(rows >= 1, "query-front index-Q kernel serves 1-16 rows");
    RuntimeCheck(projected.stride(1) == 1 && projected.stride(0) % 8 == 0, "16-byte aligned projected rows");
    RuntimeCheck(k_col >= kLora && k_col % 4 == 0 && k_col + kDim <= projected.size(1), "index-K columns");
    RuntimeCheck(g_col >= kLora && g_col + kHeads <= projected.size(1), "head-gate columns");
    RuntimeCheck(wq_b.size(0) == kHeads * kDim && wq_b.size(1) == kLora && wq_b.stride(0) == kLora, "wq_b");
    RuntimeCheck(w_qa.size(0) == kLora, "w_qa");
    RuntimeCheck(q8.size(0) == rows && w.size(0) == rows, "outputs");
    RuntimeCheck(slots.size(0) >= rows && slots.stride(0) == 1 && slots.dtype().bits == 64, "int64 write slots");
    const bool kfp4 = kcache.ndim() == 2 && kcache.size(1) == 64 * 68 && kcache.stride(0) == 64 * 68 && kcache.stride(1) == 1;
    RuntimeCheck(kfp4 || (kcache.ndim() == 2 && kcache.size(1) == 64 * 132 && kcache.stride(0) == 64 * 132 && kcache.stride(1) == 1),
                 "index-K pages [pages, 64 * 132] (or MXFP4 [pages, 64 * 68])");
    RuntimeCheck(kgamma.size(0) == kDim && kbeta.size(0) == kDim && kgamma.dtype().bits == 32 && kbeta.dtype().bits == 32,
                 "FP32 index-K LayerNorm");
    Params p{};
    p.projected = static_cast<const __nv_bfloat16*>(projected.data_ptr());
    p.w_qa = static_cast<const __nv_bfloat16*>(w_qa.data_ptr());
    p.wq_b = static_cast<const __nv_bfloat16*>(wq_b.data_ptr());
    p.q8 = static_cast<uint8_t*>(q8.data_ptr());
    p.w = static_cast<float*>(w.data_ptr());
    p.pitch = projected.stride(0);
    p.rows = static_cast<int>(rows);
    p.eps = static_cast<float>(eps);
    p.k_col = static_cast<int>(k_col);
    p.g_col = static_cast<int>(g_col);
    p.slots = static_cast<const int64_t*>(slots.data_ptr());
    p.kcache = static_cast<uint8_t*>(kcache.data_ptr());
    p.kgamma = static_cast<const float*>(kgamma.data_ptr());
    p.kbeta = static_cast<const float*>(kbeta.data_ptr());
    p.keps = static_cast<float>(keps);
    constexpr int kSplit = 8;
    auto kernel = kfp4 ? iqp_kernel<true> : iqp_kernel<false>;
    const size_t bytes = sizeof(SmemP);
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
    LaunchKernel(kHeads * kSplit, kThreads, projected.device(), bytes)
        .config({.use_pdl = true, .cluster_dim = dim3(kSplit, 1, 1)})(kernel, p);
  }
};

}  // namespace k31_index_qp
