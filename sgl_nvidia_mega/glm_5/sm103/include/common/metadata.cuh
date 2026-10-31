// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace glm5 {
// Shared ABI between a device-only cubin and its typed host binding.
// Resource and launch values are populated exclusively from the device Config.
struct KernelMetadata {
  std::uint64_t version, rows, tp_size, workspace_bytes, epoch_count, peer_bytes;
  std::uint64_t workspace_fill, required_sms, native_cvt;
  std::uint64_t grid, threads, shared_bytes, args_size, args_alignment;
};
static_assert(sizeof(KernelMetadata) == 14 * sizeof(std::uint64_t));
}  // namespace glm5
