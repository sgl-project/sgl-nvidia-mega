// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// KNOWN K-18 (I5): the stream-ordered and PDL edges of the verify path's tvm-ffi launches (DCP1 topk_v2 / plan, TSI
// prep / scan / merge, the fused index query, the index-K store) hold only if tvm-ffi launches them on the stream the
// surrounding event joins assume. tvm-ffi resolves that stream per call (TVMFFIEnvGetStream, set from the caller's
// framework stream); this probe resolves it exactly as LaunchKernel does, for a tensor of the launch's device, and
// fails when it differs from the expected stream. No kernel, no device work.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace k31_stream_check {

struct Probe {
  static void check(const tvm::ffi::TensorView like, const int64_t expected_stream) {
    using namespace sglang::host;
    const int64_t launch_stream = reinterpret_cast<int64_t>(LaunchKernel::resolve_device(like.device()));
    RuntimeCheck(launch_stream == expected_stream, "K-18 stream check: tvm-ffi launch stream ", launch_stream,
                 " != expected stream ", expected_stream, "; nothing launched");
  }
};

}  // namespace k31_stream_check
