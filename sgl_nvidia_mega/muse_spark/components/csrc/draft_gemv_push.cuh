// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Host entry of draft_gemv's push epilogue (kEpiPush): the GEMV that produces a draft layer's row-parallel partial (attention
// output projection or MLP down projection) stores its output tile as the plain epilogue does and also pushes every 16-byte
// vector into the peers' slots of the custom all-reduce push plane, exactly where the layer boundary kernel
// (draft_ar_norm.cuh, one-shot push protocol) would have pushed it. The boundary then runs with kConsumerOnly (polls, sums,
// clears and flips the phase counters as before, without pushing). Same vectors in the same slots: the boundary's outputs are
// bit-identical to the plain GEMV followed by the full boundary kernel.
//
// kMcPush: every vector is stored once through the push plane's multicast address instead of once per peer; that store
// also writes this rank's own slot, so the boundary must then run with kConsumerOnly and kMcPush (it polls and scrubs the
// own slot as for its own multicast push). Same vectors in the same slots: bit-identical outputs.
//
// Requirements (checked): rows are whole 8-token request blocks; the boundary would use one cluster of `consumer_cluster`
// CTAs per request block without looping (requests x consumer_cluster <= the plane's block counters); the output fits a slot.

#pragma once

#include "draft_gemv.cuh"

#include <sgl_kernel/distributed/communicator.cuh>

namespace sglang {

template <int kTokTile, int kRows, int kStepK, int kStages, int kWarps, int kCluster, uint32_t kWorld, bool kUsePDL,
          bool kEarlyTrigger = false, int kPfSteps = 0, bool kMcPush = false>
struct MuseVrDraftGemvPush {
  using G = muse_vr::draft_gemv::DraftGemv<kTokTile, kRows, kStepK, kStages, kWarps, kCluster, 1, muse_vr::draft_gemv::kEpiPush,
                                           kUsePDL, kEarlyTrigger, kPfSteps>;

  static void run(host::distributed::CommunicatorRef ref, const tvm::ffi::TensorView x, const tvm::ffi::TensorView w,
                  const tvm::ffi::TensorView out, int64_t consumer_cluster) {
    using namespace host;
    static_assert(kWorld >= 2 && kWorld <= 8, "world size");
    const auto& push = ref.get()->get_push_obj();
    muse_vr::draft_gemv::Params params{};
    auto device_ = SymbolicDevice{};
    auto M = SymbolicSize{"m"};
    auto N = SymbolicSize{"n"};
    device_.set_options<kDLCUDA>();
    TensorMatcher({M, N}).with_strides({N, 1}).with_dtype<bf16_t>().with_device(device_).verify(out);
    const int64_t m = M.unwrap(), n = N.unwrap();
    RuntimeCheck(push.world_size == kWorld, "draft_gemv push: the kernel is compiled for TP", kWorld);
    RuntimeCheck(m % 8 == 0, "draft_gemv push: rows must be whole request blocks");
    RuntimeCheck(consumer_cluster >= 1 && n % (8 * consumer_cluster) == 0, "draft_gemv push: column slices");
    RuntimeCheck(m * n * 2 <= int64_t(push.slot_bytes), "draft_gemv push: rows exceed the push slot size");
    RuntimeCheck((m / 8) * consumer_cluster <= int64_t(push.num_blocks), "draft_gemv push: the boundary would loop");
    const auto ws = push.get_workspace<kWorld>(m * n * 2);
    for (uint32_t i = 0; i < kWorld; ++i) params.push_ws[i] = ws.workspaces[i];
    params.push_counter = reinterpret_cast<const uint32_t*>(ws.counter);
    params.push_slot_bytes = ws.slot_bytes;
    params.push_world = static_cast<int>(kWorld);
    params.push_rank = static_cast<int>(push.rank);
    params.push_cluster = static_cast<int>(consumer_cluster);
    if constexpr (kMcPush) {
      RuntimeCheck(ws.mc_workspace != nullptr, "draft_gemv push: the multicast push needs a push plane with a multicast address");
      params.push_mc = ws.mc_workspace;
    }
    params.out = static_cast<bf16_t*>(out.data_ptr());
    params.ldo = n;
    int64_t mm = 0;
    G::launch(params, x, w, device_, mm);
    RuntimeCheck(mm == m, "draft_gemv push: x rows must equal out rows");
  }
};

}  // namespace sglang
