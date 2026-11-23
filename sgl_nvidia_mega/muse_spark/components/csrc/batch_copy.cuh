// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Batched device-to-device byte copy for the step graphs' input staging: up to kMaxCopies contiguous byte ranges
// (destination, source, length) in ONE launch, instead of one launch per dtype / word-width group plus one per large
// tensor. Bytes are copied as they are (16-byte vector accesses when both addresses of a block are 16-byte aligned,
// single bytes otherwise), so the result is bitwise the source whatever the dtype.
//
// Host side: the table is a CPU int64 tensor [3, n] (destination addresses, source addresses, lengths); it is read on
// the host and passed to the kernel by value, so no host-to-device transfer is issued. ``anchor`` is any CUDA tensor on
// the ranges' device: it gives the device and makes the call run on that device's current stream.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace muse_vr_batch_copy {

constexpr int kMaxCopies = 64;
constexpr uint32_t kThreads = 256;
constexpr int64_t kChunk = 16384;  // bytes per block (tunable)

struct Params {
  int n;
  int64_t dst[kMaxCopies];
  int64_t src[kMaxCopies];
  int64_t bytes[kMaxCopies];
  int32_t first_block[kMaxCopies + 1];  // prefix of the blocks per copy
};

__global__ void batch_copy_kernel(const __grid_constant__ Params p) {
  const int b = static_cast<int>(blockIdx.x);
  int lo = 0, hi = p.n - 1;  // the copy of this block: the last i with first_block[i] <= b
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (p.first_block[mid] <= b) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  const int i = lo;
  const int64_t off = int64_t(b - p.first_block[i]) * kChunk;
  const int64_t rest = p.bytes[i] - off;
  const int64_t len = rest < kChunk ? rest : kChunk;
  const char* s = reinterpret_cast<const char*>(p.src[i]) + off;
  char* d = reinterpret_cast<char*>(p.dst[i]) + off;
  int64_t done = 0;
  if (((reinterpret_cast<uintptr_t>(s) | reinterpret_cast<uintptr_t>(d)) & 15) == 0) {
    const int64_t n16 = len >> 4;
    const uint4* s16 = reinterpret_cast<const uint4*>(s);
    uint4* d16 = reinterpret_cast<uint4*>(d);
    for (int64_t k = threadIdx.x; k < n16; k += blockDim.x) d16[k] = s16[k];
    done = n16 << 4;
  }
  for (int64_t k = done + threadIdx.x; k < len; k += blockDim.x) d[k] = s[k];
}

}  // namespace muse_vr_batch_copy

struct MuseVrBatchCopy {
  using TensorView = tvm::ffi::TensorView;

  // table: CPU int64 [3, n] (dst addresses, src addresses, byte lengths); anchor: a CUDA tensor on the ranges' device
  static void run(TensorView table, TensorView anchor) {
    using namespace host;
    namespace bc = muse_vr_batch_copy;
    auto N = SymbolicSize{"n"};
    auto cpu = SymbolicDevice{};
    cpu.set_options<kDLCPU>();
    TensorMatcher({3, N}).with_dtype<int64_t>().with_device(cpu).verify(table);
    const int64_t n = N.unwrap();
    RuntimeCheck(n >= 1 && n <= bc::kMaxCopies, "batch copy: 1 to 64 ranges");
    RuntimeCheck(table.stride(1) == 1 && table.stride(0) == n, "batch copy: the table must be contiguous");
    const int64_t* t = static_cast<const int64_t*>(table.data_ptr());
    bc::Params p{};
    p.n = static_cast<int>(n);
    int64_t blocks = 0;
    for (int64_t i = 0; i < n; ++i) {
      p.dst[i] = t[i];
      p.src[i] = t[n + i];
      p.bytes[i] = t[2 * n + i];
      RuntimeCheck(p.bytes[i] > 0, "batch copy: empty range");
      p.first_block[i] = static_cast<int32_t>(blocks);
      blocks += (p.bytes[i] + bc::kChunk - 1) / bc::kChunk;
    }
    p.first_block[n] = static_cast<int32_t>(blocks);
    RuntimeCheck(blocks < (int64_t(1) << 30), "batch copy: too many blocks");
    const DLDevice device = anchor.device();
    RuntimeCheck(device.device_type == kDLCUDA, "batch copy: the anchor must be a CUDA tensor");
    LaunchKernel(static_cast<uint32_t>(blocks), bc::kThreads, device)(bc::batch_copy_kernel, p);
  }
};

}  // namespace sglang
