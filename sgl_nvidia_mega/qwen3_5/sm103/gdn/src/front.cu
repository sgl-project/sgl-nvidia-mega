// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// The GatedDeltaNet verify front of Qwen3.5 at TP8 for one request of 7 rows, in one launch of 11
// clusters of 8 CTAs: the static-FP8 in_proj_qkvz projection (BF16 input quantized in-kernel, e4m3
// tensor-core split-K reduced in fixed cluster-rank order), the causal conv update of its q / k / v
// channels (output, rolling state, deduplicated intermediate windows), the BF16 in_proj_ba side
// projection on tensor cores (cluster 10), and an L2 prefetch of the ReplaySSM verify's checkpoint
// and history ring. Ported from gdn_pdl_front_ba_mkm.cuh of SGLang branch nvidia/qwen35-agentx-gb300
// (SGLANG_Q35MF_MKG_FUSED + MKG_BA_TC + MKM_GDN_FRONT); the device code is unchanged.
//
// Built with NVRTC: front_kernel is a device function, and the extern "C" entries at the end
// instantiate its BF16-input form with the fused conv and the inline side projection, with the
// projection weights in registers (q35_gdn_front) or by bulk copy into shared memory (q35_gdn_front_tma).

#include <cuda_bf16.h>

// NVRTC has no <cstdint>
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;

namespace sglang {
namespace gdn_pdl_front_ba_mkm_detail {

namespace gdn_ba {

// One L2 prefetch region per recurrent operand: slot `index` of [slots, ...] with contiguous slots.
struct Prefetch {
  const char* base;
  int64_t slot_stride;  // bytes
  int64_t bytes;        // bytes per slot, multiple of 16
  int ring;             // 0: physical state index, 1: request ring index
};
struct Params {
  const __nv_bfloat16* input;
  const __nv_bfloat16* weight;
  __nv_bfloat16* output;
  int rows;
  int64_t input_stride;
  Prefetch prefetch[4];
  int prefetches;
  const void* state_indices;
  const void* ring_indices;
  bool state_index64;
  bool ring_index64;
  int64_t pad_slot;
};

// A hint only: the consumer still reads these operands after its own dependency wait.
__device__ __forceinline__ void prefetch_l2(const char* p, int64_t bytes) {
  for (int64_t offset = 0; offset < bytes; offset += 32768) {
    const int64_t chunk = bytes - offset < 32768 ? bytes - offset : 32768;
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p + offset), "r"((unsigned)chunk) : "memory");
  }
}

__device__ __forceinline__ unsigned shared_address(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ unsigned peer_address(unsigned address, int rank) {
  unsigned result;
  asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(result) : "r"(address), "r"(rank));
  return result;
}
__device__ __forceinline__ void expect(uint64_t* barrier, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               :: "r"(shared_address(barrier)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void push(unsigned target, float x, float y, float z,
                                     float w, unsigned barrier) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1,%2,%3,%4}, [%5];"
               :: "r"(target), "f"(x), "f"(y), "f"(z), "f"(w), "r"(barrier) : "memory");
}
__device__ __forceinline__ void wait(uint64_t* barrier, unsigned* error, uint64_t timeout) {
  const uint64_t start = clock64();
  unsigned polls = 0, ready = 0;
  do {
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }"
                 : "=r"(ready) : "r"(shared_address(barrier)) : "memory");
    if (!ready && (clock64() - start > timeout || ++polls >= (1u << 24))) {
      if (atomicCAS(error, 0u, 1u) == 0u) printf("side projection inbox timeout block=%u\n", blockIdx.x);
      asm volatile("trap;");
    }
  } while (!ready);
}
__device__ __forceinline__ void mma(float (&d)[4], unsigned a0, unsigned a1,
                                    unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// All eight CTAs enter this body. Scratch belongs exclusively to this role.
// The receiver acknowledges each source only after its final shared read.
template <int Threads = 256>
__device__ __forceinline__ void ba_cluster_stage(
    const Params& p, int rank, int tid, float* partial, float* inbox,
    uint64_t* barrier, unsigned* error, uint64_t timeout) {
  static_assert(Threads == 256);
  const int lane = tid & 31, warp = tid >> 5;
  const int group = lane >> 2, pair = 2 * (lane & 3);
  const int split = warp >> 1, column_tile = warp & 1;
  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(shared_address(barrier)) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (tid == 0) expect(barrier, rank == 0 ? 8 * p.rows * 16 * sizeof(float) : 16);
  __syncthreads();
  asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");

  unsigned weight[8][2];
#pragma unroll
  for (int step = 0; step < 8; ++step) {
    const int k = rank * 512 + split * 128 + step * 16 + pair;
    const auto* row = p.weight + (column_tile * 8 + group) * 4096;
    weight[step][0] = *reinterpret_cast<const unsigned*>(row + k);
    weight[step][1] = *reinterpret_cast<const unsigned*>(row + k + 8);
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (tid == 0 && *reinterpret_cast<volatile unsigned*>(error)) asm volatile("trap;");
  float acc[4] = {};
#pragma unroll
  for (int step = 0; step < 8; ++step) {
    unsigned low = 0, high = 0;
    if (group < p.rows) {
      const auto* row = p.input + group * p.input_stride;
      const int k = rank * 512 + split * 128 + step * 16 + pair;
      low = *reinterpret_cast<const unsigned*>(row + k);
      high = *reinterpret_cast<const unsigned*>(row + k + 8);
    }
    mma(acc, low, 0, high, 0, weight[step][0], weight[step][1]);
  }
  if (group < p.rows) {
    const int output = group * 16 + column_tile * 8 + pair;
    partial[split * 112 + output] = acc[0];
    partial[split * 112 + output + 1] = acc[1];
  }
  __syncthreads();
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
  if (tid < p.rows * 4) {
    const int offset = tid * 4;
    float values[4] = {};
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) values[j] = __fadd_rn(values[j], partial[k * 112 + offset + j]);
    push(peer_address(shared_address(inbox + rank * 256 + offset), 0),
         values[0], values[1], values[2], values[3], peer_address(shared_address(barrier), 0));
  }
  if (tid == 0) wait(barrier, error, timeout);
  __syncthreads();
  if (rank == 7 && tid < p.prefetches) {
    const int64_t state = p.state_index64 ? static_cast<const int64_t*>(p.state_indices)[0]
                                          : static_cast<const int32_t*>(p.state_indices)[0];
    if (state != p.pad_slot) {
      const Prefetch f = p.prefetch[tid];
      const int64_t slot = f.ring ? (p.ring_index64 ? static_cast<const int64_t*>(p.ring_indices)[0]
                                                    : static_cast<const int32_t*>(p.ring_indices)[0])
                                  : state;
      if (slot >= 0) prefetch_l2(f.base + slot * f.slot_stride, f.bytes);
    }
  }
  if (rank == 0) {
    if (tid < p.rows * 16) {
      float value = 0.f;
#pragma unroll
      for (int source = 0; source < 8; ++source) value = __fadd_rn(value, inbox[source * 256 + tid]);
      p.output[tid] = __float2bfloat16_rn(value);
    }
    __syncthreads();
    if (tid > 0 && tid < 8)
      push(peer_address(shared_address(partial), tid), 0.f, 0.f, 0.f, 0.f,
           peer_address(shared_address(barrier), tid));
  }
}

}  // namespace gdn_ba

constexpr int kThreads = 256;
constexpr int kCluster = 8;
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void cluster_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
__device__ __forceinline__ void cluster_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }
__device__ __forceinline__ uint4 ldg_nc_v4(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}

__device__ __forceinline__ uint32_t smem_u32(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_init_fence() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes)
               : "memory");
}
// Bulk copy of `bytes` from global memory into this CTA's shared memory, completing on `bar` (TMA engine).
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar))
               : "memory");
}
__device__ __forceinline__ void mbar_wait_parity0(uint64_t* bar) {
  unsigned ready = 0;
  do {
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }"
                 : "=r"(ready)
                 : "r"(smem_u32(bar))
                 : "memory");
  } while (!ready);
}
__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
// low byte = e4m3(lo), high byte = e4m3(hi)  (cvt d, a=hi, b=lo)
__device__ __forceinline__ uint32_t e4m3x2(float lo, float hi) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ uint8_t e4m3(float v) { return (uint8_t)(e4m3x2(v, 0.f) & 0xff); }

__device__ __forceinline__ void mma_e4m3(float (&d)[4], const uint4& a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

// stock _static_quant_fp8: y * div.full(1, s), clamp +-448, cvt.rn.satfinite.e4m3
__device__ __forceinline__ uint32_t quant4_static(const __nv_bfloat16* p, float inv) {
  const uint2 raw = *reinterpret_cast<const uint2*>(p);
  const __nv_bfloat162 v01 = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
  const __nv_bfloat162 v23 = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
  float f[4] = {__low2float(v01), __high2float(v01), __low2float(v23), __high2float(v23)};
#pragma unroll
  for (int i = 0; i < 4; ++i) f[i] = fminf(fmaxf(__fmul_rn(f[i], inv), -448.f), 448.f);
  return e4m3x2(f[0], f[1]) | (e4m3x2(f[2], f[3]) << 16);
}


__device__ __forceinline__ unsigned cluster_rank() {
  unsigned value;
  asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(value));
  return value;
}
__device__ __forceinline__ unsigned peer_address(unsigned value, unsigned rank) {
  unsigned result;
  asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(result) : "r"(value), "r"(rank));
  return result;
}
__device__ __forceinline__ void push4(unsigned dst, float a, float b, float c, float d, unsigned bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1,%2,%3,%4}, [%5];"
      :: "r"(dst), "f"(a), "f"(b), "f"(c), "f"(d), "r"(bar) : "memory");
}
__device__ __forceinline__ void wait_inbox(uint64_t* bar, unsigned* error, uint64_t timeout) {
  const uint64_t start = clock64();
  unsigned polls = 0, ready = 0;
  do {
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }"
                 : "=r"(ready) : "r"(smem_u32(bar)) : "memory");
    if (!ready && (clock64() - start > timeout || ++polls >= (1u << 24))) {
      if (atomicCAS(error, 0u, 1u) == 0u) printf("projection inbox timeout block=%u\n", blockIdx.x);
      asm volatile("trap;");
    }
  } while (!ready);
}
struct Params {
  const void* x;
  const uint4* weight;
  const float* sx;
  const float* sw;
  __nv_bfloat16* output;
  unsigned* error;
  uint64_t timeout;
  int m;
  const __nv_bfloat16* conv_weight;
  __nv_bfloat16* conv_state;
  __nv_bfloat16* windows;
  __nv_bfloat16* conv_output;
  const void* state_indices;
  const void* replay_indices;
  int64_t state_stride[3];
  int64_t window_stride[2];
  bool state_index64;
  bool replay_index64;
  int64_t pad_slot;
  gdn_ba::Params ba;
};

// Weight bytes per projection CTA: [8 warps][16 steps][2 blocks][32 lanes] x 16 B, contiguous in the packed layout.
constexpr uint32_t kWeightBytes = 8 * 16 * 2 * 32 * 16;

// Each cluster reduces its split-K partials in fixed rank order.
// Tma: the CTA's weights arrive by one bulk copy into shared memory issued before the PDL wait (instead of 32
// register loads per thread), then each thread reads the same A fragments; same MMAs, same bits.
template <bool Quantized, bool FuseConv = false, bool InlineBa = false, bool Tma = false>
__device__ __forceinline__ void front_kernel(const Params& p) {
  if (threadIdx.x == 0) pdl_trigger();
  __shared__ __align__(16) float inbox[kCluster][256];
  __shared__ __align__(16) uint2 fragments[16][32];
  __shared__ __align__(8) uint64_t barrier;
  __shared__ __align__(8) uint64_t weight_barrier;
  extern __shared__ __align__(128) uint4 weight_smem[];  // kWeightBytes when Tma
  __shared__ float conv_stage[32][8];  // per channel: state taps 0..2, weights 3..6
  __shared__ int64_t conv_meta[3];     // valid, state slot, replay slot
  const int rank = cluster_rank(), unit = blockIdx.x / kCluster;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  if constexpr (InlineBa) {
    // The side projection uses the original BF16 activation independently of FP8 input.
    if (unit == 10) {
      gdn_ba::ba_cluster_stage<kThreads>(p.ba,rank,tid,
          reinterpret_cast<float*>(fragments),reinterpret_cast<float*>(inbox),
          &barrier,p.error,p.timeout);
      return;
    }
  }
  if (tid == 0) {
    mbar_init(&barrier, 1);
    if constexpr (Tma) mbar_init(&weight_barrier, 1);
    mbar_init_fence();
  }
  __syncthreads();
  cluster_arrive();
  const uint4* block_weights = p.weight + (size_t)(unit * kCluster + rank) * (8 * 16 * 2 * 32);
  if constexpr (Tma) {
    if (tid == 0) {
      mbar_expect_tx(&weight_barrier, kWeightBytes);
#pragma unroll
      for (int chunk = 0; chunk < 4; ++chunk)
        bulk_g2s(weight_smem + chunk * (kWeightBytes / 64), block_weights + chunk * (kWeightBytes / 64),
                 kWeightBytes / 4, &weight_barrier);
    }
  }
  // Convolution owners: warp 0 reads the weights, indices and rolling state off the projection path.
  const bool conv_reader = FuseConv && unit < 6 && warp == 0;
  const int conv_channel = unit * 256 + rank * 32 + (tid >> 2) + 8 * (tid & 3);
  // Raw BF16 until the staging write, so no thread waits on these reads before the projection.
  __nv_bfloat16 conv_tap[4] = {};
  if (conv_reader) {
#pragma unroll
    for (int tap = 0; tap < 4; ++tap) conv_tap[tap] = p.conv_weight[conv_channel * 4 + tap];
  }
  uint4 weights[16][2];
  if constexpr (!Tma) {
    const uint4* wp = block_weights + warp * 1024 + lane;
#pragma unroll
    for (int step = 0; step < 16; ++step)
#pragma unroll
      for (int block = 0; block < 2; ++block)
        weights[step][block] = ldg_nc_v4(wp + (step * 2 + block) * 32);
  }
  const float sx = __ldg(p.sx), alpha = __fmul_rn(sx, __ldg(p.sw));
  pdl_wait();
  // Issued now, tested before the first push: no store precedes the check.
  const unsigned sticky = tid == 0 ? *reinterpret_cast<volatile unsigned*>(p.error) : 0u;
  // Both indices are read now (independently) and used only after the projection.
  int64_t conv_slot = 0, conv_replay = 0;
  if (conv_reader) {
    conv_slot = p.state_index64 ? static_cast<const int64_t*>(p.state_indices)[0]
                                : static_cast<const int32_t*>(p.state_indices)[0];
    conv_replay = p.replay_index64 ? static_cast<const int64_t*>(p.replay_indices)[0]
                                   : static_cast<const int32_t*>(p.replay_indices)[0];
  }
  const float inverse = div_full(1.0f, sx);
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int step = 2 * warp + i;
    uint2 value = make_uint2(0u, 0u);
    if (g < p.m) {
      const int offset = g * 4096 + rank * 512 + step * 32 + 4 * t;
      if constexpr (Quantized) {
        const uint8_t* x = static_cast<const uint8_t*>(p.x) + offset;
        value = make_uint2(*reinterpret_cast<const uint32_t*>(x),
                           *reinterpret_cast<const uint32_t*>(x + 16));
      } else {
        const auto* x = static_cast<const __nv_bfloat16*>(p.x) + offset;
        value = make_uint2(quant4_static(x, inverse), quant4_static(x + 16, inverse));
      }
    }
    fragments[step][lane] = value;
  }
  __syncthreads();
  if constexpr (Tma) {
    mbar_wait_parity0(&weight_barrier);
    const uint4* ws = weight_smem + warp * 1024 + lane;
#pragma unroll
    for (int step = 0; step < 16; ++step)
#pragma unroll
      for (int block = 0; block < 2; ++block)
        weights[step][block] = ws[(step * 2 + block) * 32];
  }
  float chains[2][4][4] = {};
#pragma unroll
  for (int step = 0; step < 16; ++step) {
    const uint2 b = fragments[step][lane];
    mma_e4m3(chains[0][step & 3], weights[step][0], b.x, b.y);
    mma_e4m3(chains[1][step & 3], weights[step][1], b.x, b.y);
  }
  float acc[2][4];
#pragma unroll
  for (int block = 0; block < 2; ++block)
#pragma unroll
    for (int i = 0; i < 4; ++i)
      acc[block][i] = __fadd_rn(__fadd_rn(chains[block][0][i], chains[block][1][i]),
                               __fadd_rn(chains[block][2][i], chains[block][3][i]));
  const bool feature_owner = FuseConv && unit < 6;
  const bool owner = feature_owner || rank < p.m;
  const bool conv_valid = conv_reader && conv_slot != p.pad_slot;
  __nv_bfloat16 conv_old[3] = {};
  if (conv_valid) {
    const int64_t base = conv_slot * p.state_stride[0] + conv_channel * p.state_stride[1];
#pragma unroll
    for (int j = 0; j < 3; ++j) conv_old[j] = p.conv_state[base + j * p.state_stride[2]];
  }
  const unsigned bytes = feature_owner ? kCluster * p.m * 32 * sizeof(float)
                                        : kCluster * 256 * sizeof(float);
  if (tid == 0 && owner) mbar_expect_tx(&barrier, bytes);
  if (sticky) asm volatile("trap;");
  cluster_wait();
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int token = 2 * t + j;
    if (token < p.m) {
      const int slot = feature_owner ? token * 32 + g * 4 : warp * 32 + g * 4;
      const unsigned destination = smem_u32(&inbox[rank][slot]);
      const int peer = feature_owner ? warp : token;
      push4(peer_address(destination, peer), acc[0][j], acc[0][2+j], acc[1][j], acc[1][2+j],
            peer_address(smem_u32(&barrier), peer));
    }
  }
  if (!owner) return;
  if (conv_reader) {
#pragma unroll
    for (int j = 0; j < 3; ++j) conv_stage[lane][j] = __bfloat162float(conv_old[j]);
#pragma unroll
    for (int j = 0; j < 4; ++j) conv_stage[lane][3 + j] = __bfloat162float(conv_tap[j]);
    if (lane == 0) {
      conv_meta[0] = conv_slot != p.pad_slot;
      conv_meta[1] = conv_slot;
      conv_meta[2] = conv_replay;
    }
  }
  if (tid == 0) wait_inbox(&barrier, p.error, p.timeout);
  __syncthreads();
  if (feature_owner) {
    // Token `warp` of 32 channels; neighbours' projected values are recomputed with the same arithmetic.
    if (warp >= 7) return;
    const int token = warp;
    const int channel = unit * 256 + rank * 32 + (lane >> 2) + 8 * (lane & 3);
    const bool valid = conv_meta[0] != 0;
    float value[4];  // projected BF16 values of tokens token-3 .. token (state taps before token 0)
    __nv_bfloat16 rounded = __float2bfloat16_rn(0.f);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int t = token - 3 + j;
      if (t < 0) {
        value[j] = conv_stage[lane][t + 3];
      } else {
        float sum = 0.f;
#pragma unroll
        for (int source = 0; source < kCluster; ++source)
          sum = __fadd_rn(sum, inbox[source][t * 32 + lane]);
        const __nv_bfloat16 r = __float2bfloat16_rn(__fmul_rn(sum, alpha));
        value[j] = __bfloat162float(r);
        if (j == 3) rounded = r;
      }
    }
    p.output[token * 2560 + channel] = rounded;
    if (valid) {
      const int64_t state_base = conv_meta[1] * p.state_stride[0] + channel * p.state_stride[1];
      __nv_bfloat16* window = p.windows + conv_meta[2] * p.window_stride[0] + channel * p.window_stride[1];
      const float weight[4] = {conv_stage[lane][3], conv_stage[lane][4], conv_stage[lane][5], conv_stage[lane][6]};
      if (token == 0) {
        // The overlapping logical windows have one physical writer per channel.
        window[0] = __float2bfloat16_rn(value[1]);
        window[1] = __float2bfloat16_rn(value[2]);
      }
      // Preserve the BF16 product boundary before the stock FP32 additions.
      const float p0 = __bfloat162float(__float2bfloat16_rn(__fmul_rn(value[0],weight[0])));
      const float p1 = __bfloat162float(__float2bfloat16_rn(__fmul_rn(value[1],weight[1])));
      const float p2 = __bfloat162float(__float2bfloat16_rn(__fmul_rn(value[2],weight[2])));
      const float p3 = __bfloat162float(__float2bfloat16_rn(__fmul_rn(value[3],weight[3])));
      float accumulator = __fadd_rn(p0,0.f);
      accumulator = __fadd_rn(accumulator,p1);
      accumulator = __fadd_rn(accumulator,p2);
      accumulator = __fadd_rn(accumulator,p3);
      float exponential;
      const float exponent = __fmul_rn(accumulator,__int_as_float(0xBFB8AA3B));
      asm("ex2.approx.f32 %0, %1;" : "=f"(exponential) : "f"(exponent));
      const float activated = div_full(accumulator, __fadd_rn(1.f, exponential));
      p.conv_output[token * 1536 + channel] = __float2bfloat16_rn(activated);
      window[token + 2] = rounded;
      if (token == 6) {
        p.conv_state[state_base] = __float2bfloat16_rn(value[1]);
        p.conv_state[state_base + p.state_stride[2]] = __float2bfloat16_rn(value[2]);
        p.conv_state[state_base + 2 * p.state_stride[2]] = __float2bfloat16_rn(value[3]);
      }
    }
  } else {
    float sum = 0.f;
#pragma unroll
    for (int source = 0; source < kCluster; ++source) sum = __fadd_rn(sum, inbox[source][tid]);
    const int feature = 32 * warp + ((lane >> 2) & 7) + 8 * (lane & 3);
    p.output[rank * 2560 + unit * 256 + feature] = __float2bfloat16_rn(__fmul_rn(sum, alpha));
  }
}

}  // namespace gdn_pdl_front_ba_mkm_detail

}  // namespace sglang

// The verify front of 7 BF16 rows: grid 88 (11 clusters of 8), 256 threads, with PDL.
extern "C" __global__ __launch_bounds__(sglang::gdn_pdl_front_ba_mkm_detail::kThreads, 1) void q35_gdn_front(
    const __grid_constant__ sglang::gdn_pdl_front_ba_mkm_detail::Params p) {
  sglang::gdn_pdl_front_ba_mkm_detail::front_kernel<false, true, true>(p);
}

// The same with each projection CTA's weights bulk-copied into kWeightBytes of dynamic shared memory.
extern "C" __global__ __launch_bounds__(sglang::gdn_pdl_front_ba_mkm_detail::kThreads, 1) void q35_gdn_front_tma(
    const __grid_constant__ sglang::gdn_pdl_front_ba_mkm_detail::Params p) {
  sglang::gdn_pdl_front_ba_mkm_detail::front_kernel<false, true, true, true>(p);
}
