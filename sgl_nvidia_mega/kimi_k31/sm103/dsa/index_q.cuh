// SPDX-License-Identifier: Apache-2.0
// K3.1 DSA index query as one PDL kernel (replaces the stock decode chain sgl_kernel rmsnorm -> torch.mm wq_b ->
// act_quant -> bf16_head_gate_q_scale, plus the weights_proj torch.mm of prepare_gate). Same method and rounding
// points as stock (kimi_k31/dsa/indexer.py, sglang head_gate.py / act_quant_triton.py):
//   xn    = bf16(x * rsqrt(mean(x^2) + eps) * w_qa)           (x = projected[:, :1536], FP32 math)
//   q     = bf16(xn @ wq_b^T)                                  (FP32 accumulation; one BF16 rounding)
//   scale = max(amax_128(q), 1e-4) * (1/448);  q8 = e4m3_rn(clamp(q / scale, +-448))   (per row and head)
//   gate  = bf16(hidden @ weights_proj^T)                      (FP32 accumulation)
//   w     = float(bf16(gate * 32^-0.5)) * scale * 128^-0.5
// Only FP32 summation orders differ from stock.
//
// Grid: 32 index heads x kSplit CTAs (one cluster per head). Each CTA owns 128/kSplit rows of its head's wq_b
// block (staged into SMEM before griddepcontrol.wait: weights only) and 7168/kSplit columns of the gate dot. The
// head's amax and gate partials meet through DSMEM with one cluster barrier.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cstdint>

namespace k31_index_q {

constexpr int kHeads = 32, kDim = 128, kLora = 1536, kHidden = 7168, kRows = 8;
constexpr int kThreads = 256;
constexpr int kStride = kLora * 2 + 16;  // padded SMEM row (bytes): conflict-free fragment loads

#ifndef K31_INDEXQ_MUTATION
#define K31_INDEXQ_MUTATION 0
#endif

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cluster_arrive_release() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void cluster_wait_acquire() {
  asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void st_peer_f32(const void* p, uint32_t rank, float v) {
  uint32_t a;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(a) : "r"(smem_addr(p)), "r"(rank));
  asm volatile("st.relaxed.cluster.shared::cluster.f32 [%0], %1;\n" ::"r"(a), "f"(v) : "memory");
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint8_t to_e4m3(float x) {
  uint16_t pair;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(pair) : "f"(0.f), "f"(x));
  return static_cast<uint8_t>(pair & 0xff);
}
__device__ __forceinline__ float bf16_round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

struct Params {
  const __nv_bfloat16* projected;  // [rows, pitch]; q_a = columns [0, 1536)
  const __nv_bfloat16* hidden;     // [rows, hidden_stride]
  const __nv_bfloat16* w_qa;       // [1536] q_a RMSNorm weight
  const __nv_bfloat16* wq_b;       // [4096, 1536]
  const __nv_bfloat16* w_proj;     // [32, 7168]
  uint8_t* q8;                     // [rows, 32, 128] FP8 E4M3
  float* w;                        // [rows, 32]
  int64_t pitch, hidden_stride;
  int rows;
  float eps;
  const uint8_t* mq_src;  // optional: MLA query [rows, heads, 576] FP8 -> sparse core's padded [rows, 16, 576]
  uint8_t* mq_dst;        // (null: off)
};

template <int kSplit>
struct Smem {
  static constexpr int kOwn = kDim / kSplit;  // wq_b rows per CTA (16 for kSplit 8)
  static constexpr int kGate = kHidden / kSplit;
  alignas(16) uint8_t wq[kOwn][kStride];
  alignas(16) uint8_t x[kRows][kStride];
  alignas(16) __nv_bfloat16 wqa[kLora];
  alignas(16) __nv_bfloat16 wp[kGate];
  float part[8][kOwn][kRows];  // per-warp K-slice partials
  float val[kOwn][kRows];
  float red[8][kRows];
  float ex_amax[kSplit][kRows];
  float ex_gate[kSplit][kRows];
  float scale[kRows];
};

// kMqHeads: the rank's real MLA query heads (12 at TP8, 6 at TP16) for the optional padded-query copy.
template <int kSplit, int kMqHeads>
__global__ void __launch_bounds__(kThreads, 1) index_q_kernel(const __grid_constant__ Params p) {
  using S = Smem<kSplit>;
  constexpr int kOwn = S::kOwn, kGate = S::kGate;
  static_assert(kOwn == 16, "one m16 tile of wq_b rows per CTA");
  extern __shared__ __align__(16) uint8_t smem_raw[];
  S& s = *reinterpret_cast<S*>(smem_raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, qd = lane & 3;
  const int head = blockIdx.x / kSplit;
  const uint32_t rank = cluster_rank();
  const int row0 = head * kDim + static_cast<int>(rank) * kOwn;  // first wq_b row of this CTA

  // Every peer has started (its shared memory exists) before anyone stores into it: arrive now, wait before the
  // first DSMEM store (CUDA PG, distributed shared memory).
  cluster_arrive_relaxed();
  // Weights only before the wait.
  for (int i = tid; i < kOwn * (kLora / 8); i += kThreads) {
    const int r = i / (kLora / 8), c = i % (kLora / 8);
    cp_async16(&s.wq[r][c * 16], p.wq_b + static_cast<int64_t>(row0 + r) * kLora + c * 8);
  }
  for (int i = tid; i < kLora / 8; i += kThreads) cp_async16(&s.wqa[i * 8], p.w_qa + i * 8);
  for (int i = tid; i < kGate / 8; i += kThreads)
    cp_async16(&s.wp[i * 8], p.w_proj + static_cast<int64_t>(head) * kHidden + rank * kGate + i * 8);
  cp_async_commit();
  pdl_wait();
  pdl_trigger();

  // Optional side job: the sparse core's kMqHeads -> 16 head-padded query copy (replaces SparseMla.prepare_query's
  // separate copy kernel); the MLA query kernel wrote mq_src before our wait. Pad heads stay zero.
  if (p.mq_src != nullptr) {
    constexpr int kMqChunks = kMqHeads * 36;  // 16-B chunks per row
    const int chunks = p.rows * kMqChunks;
    for (int i = blockIdx.x * kThreads + tid; i < chunks; i += gridDim.x * kThreads) {
      const int r = i / kMqChunks, rem = i % kMqChunks;
      reinterpret_cast<uint4*>(p.mq_dst + static_cast<int64_t>(r) * 16 * 576)[rem] =
          reinterpret_cast<const uint4*>(p.mq_src + static_cast<int64_t>(r) * kMqHeads * 576)[rem];
    }
  }
  // Issue every activation load at once: warp w's q_a row (6 x 16 B per lane) and this CTA's hidden columns.
  constexpr int kQaChunks = kLora / 8 / 32;  // 16-B chunks per lane
  constexpr int kGateChunks = kGate / 8;     // 16-B chunks per row
  constexpr int kGatePer = (kRows * kGateChunks + kThreads - 1) / kThreads;
  uint4 xa[kQaChunks];
  uint4 hv[kGatePer];
  {
    const int r = warp;
    const uint4* src = reinterpret_cast<const uint4*>(p.projected + static_cast<int64_t>(r) * p.pitch);
#pragma unroll
    for (int i = 0; i < kQaChunks; ++i) xa[i] = r < p.rows ? src[i * 32 + lane] : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int i = 0; i < kGatePer; ++i) {
      const int c = tid + i * kThreads, b = c / kGateChunks, cc = c % kGateChunks;
      hv[i] = b < p.rows ? reinterpret_cast<const uint4*>(p.hidden + b * p.hidden_stride + rank * kGate)[cc]
                         : make_uint4(0, 0, 0, 0);
    }
  }
  cp_async_wait_all();
  __syncthreads();  // w_qa, wq_b and wp staged
  // RMSNorm of q_a (rows >= p.rows are zero): warp w normalizes row w.
  {
    const int r = warp;
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
#pragma unroll
    for (int i = 0; i < kQaChunks; ++i) {
      const int k0 = (i * 32 + lane) * 8;
      const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&xa[i]);
      __align__(16) __nv_bfloat16 o[8];
#pragma unroll
      for (int j = 0; j < 8; ++j)
        o[j] = r < p.rows ? __float2bfloat16_rn(__bfloat162float(e[j]) * rr * __bfloat162float(s.wqa[k0 + j]))
                          : __float2bfloat16_rn(0.f);
      *reinterpret_cast<uint4*>(&s.x[r][k0 * 2]) = *reinterpret_cast<const uint4*>(o);
    }
  }
  // Gate partial: this CTA's kGate columns of hidden @ weights_proj[head]^T, fixed reduction order.
  {
    float acc[kRows];
#pragma unroll
    for (int b = 0; b < kRows; ++b) acc[b] = 0.f;
#pragma unroll
    for (int i = 0; i < kGatePer; ++i) {
      const int c = tid + i * kThreads, b = c / kGateChunks, cc = c % kGateChunks;
      if (b < kRows) {
        const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&hv[i]);
        float t = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) t = fmaf(__bfloat162float(e[j]), __bfloat162float(s.wp[cc * 8 + j]), t);
#pragma unroll
        for (int bb = 0; bb < kRows; ++bb)
          if (bb == b) acc[bb] += t;
      }
    }
#pragma unroll
    for (int b = 0; b < kRows; ++b) {
#pragma unroll
      for (int d = 16; d; d >>= 1) acc[b] += __shfl_xor_sync(0xffffffffu, acc[b], d);
      if (lane == 0) s.red[warp][b] = acc[b];
    }
  }
  __syncthreads();
  // wq_b slice: warp w reduces K columns [192 w, 192 w + 192) of the 16 x 8 product.
  {
    float d[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int ks = 0; ks < kLora / 8 / 16; ++ks) {
      const int k0 = warp * (kLora / 8) + ks * 16;
      auto wq = [&](int r, int k) { return *reinterpret_cast<const uint32_t*>(&s.wq[r][k * 2]); };
      auto xs = [&](int r, int k) { return *reinterpret_cast<const uint32_t*>(&s.x[r][k * 2]); };
      mma_bf16(d, wq(g, k0 + 2 * qd), wq(g + 8, k0 + 2 * qd), wq(g, k0 + 8 + 2 * qd), wq(g + 8, k0 + 8 + 2 * qd),
               xs(g, k0 + 2 * qd), xs(g, k0 + 8 + 2 * qd));
    }
    s.part[warp][g][2 * qd] = d[0];
    s.part[warp][g][2 * qd + 1] = d[1];
    s.part[warp][g + 8][2 * qd] = d[2];
    s.part[warp][g + 8][2 * qd + 1] = d[3];
  }
  __syncthreads();
  if (tid < kOwn * kRows) {  // fixed-order sum over the 8 K slices, one BF16 rounding
    const int r = tid / kRows, b = tid % kRows;
    float acc = 0.f;
#pragma unroll
    for (int w8 = 0; w8 < 8; ++w8) acc += s.part[w8][r][b];
    s.val[r][b] = bf16_round(acc);
  }
  __syncthreads();
  cluster_wait_acquire();  // pairs with the arrive at kernel start: every peer is running
  if (tid < kRows) {  // local amax over this CTA's 16 rows, gate partial: to the head's cluster
    const int b = tid;
    float amax = 0.f, gate = 0.f;
#pragma unroll
    for (int r = 0; r < kOwn; ++r) amax = fmaxf(amax, fabsf(s.val[r][b]));
#pragma unroll
    for (int w8 = 0; w8 < 8; ++w8) gate += s.red[w8][b];
#pragma unroll
    for (int c = 0; c < kSplit; ++c) st_peer_f32(&s.ex_amax[rank][b], c, amax);
    st_peer_f32(&s.ex_gate[rank][b], 0, gate);
  }
  cluster_arrive_release();
  cluster_wait_acquire();
  if (tid < kRows) {
    const int b = tid;
    float amax = 0.f;
#pragma unroll
    for (int c = 0; c < kSplit; ++c) amax = fmaxf(amax, s.ex_amax[c][b]);
#if K31_INDEXQ_MUTATION == 1
    s.scale[b] = fmaxf(amax, 1e-4f) * (1.0f / 440.0f);  // mutation: wrong FP8 range
#else
    s.scale[b] = fmaxf(amax, 1e-4f) * (1.0f / 448.0f);
#endif
    if (rank == 0 && b < p.rows) {
      float gate = 0.f;
#pragma unroll
      for (int c = 0; c < kSplit; ++c) gate += s.ex_gate[c][b];
      const float g2 = bf16_round(bf16_round(gate) * 0.17677669529663687f);  // 32^-0.5
#if K31_INDEXQ_MUTATION == 2
      p.w[b * kHeads + head] = g2 * s.scale[b];  // mutation: softmax scale dropped
#else
      p.w[b * kHeads + head] = g2 * s.scale[b] * 0.08838834764831845f;  // 128^-0.5
#endif
    }
  }
  __syncthreads();
  if (tid < kOwn * kRows) {
    const int r = tid % kOwn, b = tid / kOwn;
    if (b < p.rows) {
      const float y = fminf(fmaxf(__fdiv_rn(s.val[r][b], s.scale[b]), -448.f), 448.f);
      p.q8[(static_cast<int64_t>(b) * kHeads + head) * kDim + rank * kOwn + r] = to_e4m3(y);
    }
  }
}

struct Binding {
  static void run(const tvm::ffi::TensorView projected, const tvm::ffi::TensorView hidden,
                  const tvm::ffi::TensorView w_qa, const tvm::ffi::TensorView wq_b, const tvm::ffi::TensorView w_proj,
                  const tvm::ffi::TensorView q8, const tvm::ffi::TensorView w, const double eps) {
    // No copy: the head count only picks an instantiation.
    run_impl(projected, hidden, w_qa, wq_b, w_proj, q8, w, eps, nullptr, nullptr, 12);
  }
  static void run_copy(const tvm::ffi::TensorView projected, const tvm::ffi::TensorView hidden,
                       const tvm::ffi::TensorView w_qa, const tvm::ffi::TensorView wq_b,
                       const tvm::ffi::TensorView w_proj, const tvm::ffi::TensorView q8, const tvm::ffi::TensorView w,
                       const double eps, const tvm::ffi::TensorView mq_src, const tvm::ffi::TensorView mq_dst,
                       const int64_t mq_heads) {
    using namespace sglang::host;
    RuntimeCheck(mq_heads == 12 || mq_heads == 6, "MLA query heads: 12 (TP8) or 6 (TP16)");
    RuntimeCheck(mq_src.size(0) >= projected.size(0) && mq_src.stride(0) == mq_heads * 576,
                 "MLA query [rows, heads, 576]");
    RuntimeCheck(mq_dst.size(0) >= projected.size(0) && mq_dst.stride(0) == 16 * 576, "padded query [rows, 16, 576]");
    run_impl(projected, hidden, w_qa, wq_b, w_proj, q8, w, eps, static_cast<const uint8_t*>(mq_src.data_ptr()),
             static_cast<uint8_t*>(mq_dst.data_ptr()), mq_heads);
  }
  static void run_impl(const tvm::ffi::TensorView projected, const tvm::ffi::TensorView hidden,
                       const tvm::ffi::TensorView w_qa, const tvm::ffi::TensorView wq_b,
                       const tvm::ffi::TensorView w_proj, const tvm::ffi::TensorView q8, const tvm::ffi::TensorView w,
                       const double eps, const uint8_t* mq_src, uint8_t* mq_dst, const int64_t mq_heads) {
    using namespace sglang::host;
    const int64_t rows = projected.size(0);
    RuntimeCheck(rows >= 1 && rows <= kRows, "index-Q kernel serves 1-8 rows");
    RuntimeCheck(projected.stride(1) == 1 && hidden.stride(1) == 1 && hidden.size(0) == rows, "row layouts");
    RuntimeCheck(projected.stride(0) % 8 == 0 && hidden.stride(0) % 8 == 0, "16-byte aligned rows");
    RuntimeCheck(wq_b.size(0) == kHeads * kDim && wq_b.size(1) == kLora && wq_b.stride(0) == kLora, "wq_b");
    RuntimeCheck(w_proj.size(0) == kHeads && w_proj.size(1) == kHidden && w_proj.stride(0) == kHidden, "w_proj");
    RuntimeCheck(w_qa.size(0) == kLora, "w_qa");
    RuntimeCheck(q8.size(0) == rows && w.size(0) == rows, "outputs");
    Params p{static_cast<const __nv_bfloat16*>(projected.data_ptr()),
             static_cast<const __nv_bfloat16*>(hidden.data_ptr()),
             static_cast<const __nv_bfloat16*>(w_qa.data_ptr()),
             static_cast<const __nv_bfloat16*>(wq_b.data_ptr()),
             static_cast<const __nv_bfloat16*>(w_proj.data_ptr()),
             static_cast<uint8_t*>(q8.data_ptr()),
             static_cast<float*>(w.data_ptr()),
             projected.stride(0),
             hidden.stride(0),
             static_cast<int>(rows),
             static_cast<float>(eps),
             mq_src,
             mq_dst};
    constexpr int kSplit = 8;
    auto kernel = mq_heads == 6 ? &index_q_kernel<kSplit, 6> : &index_q_kernel<kSplit, 12>;
    const size_t bytes = sizeof(Smem<kSplit>);
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
    LaunchKernel(kHeads * kSplit, kThreads, projected.device(), bytes)
        .config({.use_pdl = true, .cluster_dim = dim3(kSplit, 1, 1)})(kernel, p);
  }
};

}  // namespace k31_index_q
