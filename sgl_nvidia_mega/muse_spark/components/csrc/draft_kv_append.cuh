// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Writes the draft's per-layer context K/V rows of one verify step into the draft KV cache for every layer in one
// launch: for request b and block position j < commit_lens[b], row (b * block + j) of layer l goes to slot
// loc_2d[b, j] of that layer's K and V buffers (plain copies of the bf16 rows, as SGLang's per-layer prefix-valid
// writer does them).

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace muse_vr::draft_kv_append {

using namespace sglang;

struct Params {
  const uint8_t* k_src;
  const uint8_t* v_src;
  int64_t src_layer_bytes;  // byte distance between consecutive layers' rows
  int64_t src_row_bytes;    // byte distance between rows of one layer
  const int64_t* k_dst;     // [layers] buffer base addresses
  const int64_t* v_dst;
  int64_t dst_row_bytes;
  const int64_t* loc;     // [bs, block]
  const int32_t* commit;  // [bs]
  int block;
  int rows;
  int row_bytes;
};

template <bool kUsePDL>
__global__ __launch_bounds__(64) void kv_append_kernel(const __grid_constant__ Params p) {
  device::PDLWaitPrimary<kUsePDL>();
  const int r = blockIdx.x, l = blockIdx.y;
  const int b = r / p.block, j = r % p.block;
  if (j < p.commit[b]) {
    const int64_t slot = p.loc[r];
    const uint8_t* ks = p.k_src + l * p.src_layer_bytes + r * p.src_row_bytes;
    const uint8_t* vs = p.v_src + l * p.src_layer_bytes + r * p.src_row_bytes;
    uint8_t* kd = reinterpret_cast<uint8_t*>(p.k_dst[l]) + slot * p.dst_row_bytes;
    uint8_t* vd = reinterpret_cast<uint8_t*>(p.v_dst[l]) + slot * p.dst_row_bytes;
    for (int c = threadIdx.x; c < p.row_bytes / 16; c += blockDim.x) {
      reinterpret_cast<uint4*>(kd)[c] = reinterpret_cast<const uint4*>(ks)[c];
      reinterpret_cast<uint4*>(vd)[c] = reinterpret_cast<const uint4*>(vs)[c];
    }
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL>
struct KvAppend {
  static void run(const tvm::ffi::TensorView k_first, const tvm::ffi::TensorView v_first, int64_t src_layer_bytes,
                  const tvm::ffi::TensorView k_dst, const tvm::ffi::TensorView v_dst, int64_t dst_row_bytes,
                  const tvm::ffi::TensorView loc, const tvm::ffi::TensorView commit, int64_t layers) {
    using namespace host;
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    auto B = SymbolicSize{"bs"};
    auto K = SymbolicSize{"block"};
    TensorMatcher({B, K}).with_dtype<int64_t>().with_device(device_).verify(loc);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(commit);
    const int64_t rows = B.unwrap() * K.unwrap();
    RuntimeCheck(k_first.size(0) == rows && v_first.size(0) == rows, "kv rows must match loc_2d");
    const int64_t row_bytes = k_first.stride(0) * 2;
    RuntimeCheck(v_first.stride(0) == k_first.stride(0) && row_bytes % 16 == 0, "row layout");
    Params p{};
    p.k_src = static_cast<const uint8_t*>(k_first.data_ptr());
    p.v_src = static_cast<const uint8_t*>(v_first.data_ptr());
    p.src_layer_bytes = src_layer_bytes;
    p.src_row_bytes = row_bytes;
    p.k_dst = static_cast<const int64_t*>(k_dst.data_ptr());
    p.v_dst = static_cast<const int64_t*>(v_dst.data_ptr());
    p.dst_row_bytes = dst_row_bytes;
    p.loc = static_cast<const int64_t*>(loc.data_ptr());
    p.commit = static_cast<const int32_t*>(commit.data_ptr());
    p.block = static_cast<int>(K.unwrap());
    p.rows = static_cast<int>(rows);
    p.row_bytes = static_cast<int>(row_bytes);
    if (rows == 0 || layers == 0) return;
    LaunchKernel(dim3(static_cast<unsigned>(rows), static_cast<unsigned>(layers)), 64, device_.unwrap())
        .enable_pdl(kUsePDL)(kv_append_kernel<kUsePDL>, p);
  }
};

}  // namespace muse_vr::draft_kv_append
