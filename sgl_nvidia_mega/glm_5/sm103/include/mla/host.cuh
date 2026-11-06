// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <limits>
#include <mla/args.h>
#include <common/host_helpers.cuh>

namespace glm5::mla::host {
using tvm::ffi::TensorView;
using namespace ::glm5::host_detail;
void check_mla_matrix(TensorView tensor, char const* name, int rows, int cols, bool fp8) {
  require(tensor.device().device_type == kDLCUDA &&
      (fp8 ? is_type<fp8_e4m3_t>(tensor.dtype()) : is_type<float>(tensor.dtype())),
      name, " has an invalid device/dtype");
  require(shape_matches(tensor, {rows, cols}) && tensor.stride(1) == 1 &&
      tensor.stride(0) >= cols && tensor.stride(0) <= std::numeric_limits<std::int32_t>::max() &&
      tensor.stride(0) % (fp8 ? 16 : 4) == 0, name, " has an invalid shape/row stride");
  check_alignment(tensor, name, 16);
}

void check_mla_scale(TensorView tensor, char const* name, int rows, int cols) {
  if (is_type<std::int32_t>(tensor.dtype())) {
    require(tensor.device().device_type == kDLCUDA &&
        shape_matches(tensor, {rows, (cols / 128 + 3) / 4}) &&
        tensor.stride(0) == 1 && tensor.stride(1) >= rows &&
        tensor.stride(1) <= std::numeric_limits<std::int32_t>::max(),
        name, " has an invalid packed UE8M0 layout");
    check_alignment(tensor, name, 16);
  } else {
    check_mla_matrix(tensor, name, (rows + 127) / 128, cols / 128, false);
  }
}

NativeWeightSource native_weight(TensorView weight, TensorView scales,
                                int box_rows, int k_atoms, bool transposed) {
  NativeWeightSource result{};
  if (transposed) {
    encode_tma_2d(&result.tensor_map, CU_TENSOR_MAP_DATA_TYPE_UINT8, weight.data_ptr(),
        weight.size(1), weight.size(0), weight.stride(0), 64, box_rows,
        CU_TENSOR_MAP_SWIZZLE_64B);
  } else {
    encode_tma_3d(&result.tensor_map, CU_TENSOR_MAP_DATA_TYPE_UINT8, weight.data_ptr(),
        128, weight.size(0), weight.size(1) / 128, weight.stride(0), 128,
        128, box_rows, k_atoms, CU_TENSOR_MAP_SWIZZLE_128B);
  }
  result.scales = static_cast<float const*>(scales.data_ptr());
  result.scale_row_stride = scales.stride(0);
  result.packed_scale_col_stride =
      is_type<std::int32_t>(scales.dtype()) ? scales.stride(1) : 0;
  return result;
}

void check_positions(TensorView tensor, char const* name, bool legacy) {
  require(tensor.device().device_type == kDLCUDA && tensor.is_contiguous() &&
      (is_type<std::int32_t>(tensor.dtype()) ||
       (!legacy && is_type<std::int64_t>(tensor.dtype()))) &&
      shape_matches(tensor, {kTokens}), name, " has an invalid position dtype/shape");
  check_alignment(tensor, name, tensor.dtype().bits / 8);
}

tvm::ffi::Function bind(std::shared_ptr<Module> module,
    TensorView x_gamma, TensorView q_a_weight, TensorView q_a_scale_inv,
    TensorView kv_a_with_mqa_weight, TensorView kv_a_with_mqa_scale_inv,
    TensorView q_gamma, TensorView q_b_weight, TensorView q_b_scale_inv,
    TensorView kv_gamma, TensorView kv_b_weight, TensorView kv_b_scale_inv,
    TensorView o_proj_weight, TensorView o_proj_scale_inv,
    TensorView kv_cache, TensorView pe_cache, std::int64_t layout,
    TensorView workspace, TensorView epochs, tvm::ffi::Array<std::int64_t> peers,
    std::int64_t rank) {
  check_tensor<NormGain>(x_gamma, "x_gamma", kNormDtype, {kHidden});
  check_tensor<NormGain>(q_gamma, "q_gamma", kNormDtype, {2048});
  check_tensor<NormGain>(kv_gamma, "kv_gamma", kNormDtype, {kKvLatentDim});
  check_mla_matrix(q_a_weight, "q_a_weight", 2048, kHidden, true);
  check_mla_scale(q_a_scale_inv, "q_a_scale_inv", 2048, kHidden);
  check_mla_matrix(kv_a_with_mqa_weight, "kv_a_with_mqa_weight", 576, kHidden, true);
  check_mla_scale(kv_a_with_mqa_scale_inv, "kv_a_with_mqa_scale_inv", 576, kHidden);
  check_mla_matrix(q_b_weight, "q_b_weight", kQueryHeads * 256, 2048, true);
  check_mla_scale(q_b_scale_inv, "q_b_scale_inv", kQueryHeads * 256, 2048);
  check_mla_matrix(kv_b_weight, "kv_b_weight", kQueryHeads * 448, 512, true);
  check_mla_scale(kv_b_scale_inv, "kv_b_scale_inv", kQueryHeads * 448, 512);
  check_mla_matrix(o_proj_weight, "o_proj_weight", kHidden, kQueryHeads * 256, true);
  check_mla_scale(o_proj_scale_inv, "o_proj_scale_inv", kHidden, kQueryHeads * 256);
  require(layout == 0 || layout == 1, "unknown cache/RoPE layout");
  require(kv_cache.ndim() == 2 && pe_cache.ndim() == 2 &&
      kv_cache.size(0) >= kTokens && kv_cache.size(0) <= std::numeric_limits<int>::max() &&
      pe_cache.size(0) == kv_cache.size(0) && kv_cache.size(1) == 512 && pe_cache.size(1) == 64 &&
      is_type<fp8_e4m3_t>(kv_cache.dtype()) &&
      is_type<fp8_e4m3_t>(pe_cache.dtype()) &&
      kv_cache.stride(1) == 1 && pe_cache.stride(1) == 1,
      "invalid E4M3 cache views");
  if (layout == 1) {
    require(kv_cache.stride(0) == 576 && pe_cache.stride(0) == 576 &&
        static_cast<char*>(pe_cache.data_ptr()) - static_cast<char*>(kv_cache.data_ptr()) == 512,
        "combined cache must alias one 576-byte row");
  } else {
    require(kv_cache.is_contiguous() && pe_cache.is_contiguous(),
                       "separate caches must be contiguous");
  }
  TensorView fixed[] = {q_gamma, kv_gamma, q_a_weight, q_a_scale_inv,
      kv_a_with_mqa_weight, kv_a_with_mqa_scale_inv, q_b_weight, q_b_scale_inv,
      kv_b_weight, kv_b_scale_inv, o_proj_weight, o_proj_scale_inv, kv_cache, pe_cache};
  for (auto tensor : fixed) check_same_device(tensor, x_gamma, "bound tensor");
  check_alignment(x_gamma, "x_gamma", 16);
  check_alignment(q_gamma, "q_gamma", 16);
  check_alignment(kv_gamma, "kv_gamma", 16);
  check_alignment(kv_cache, "kv_cache", 128);
  check_alignment(pe_cache, "pe_cache", 16);
  ContextGuard guard(module->context);
  Glm5MegaMlaArgs args{};
  bind_state(args, x_gamma, workspace, epochs, peers, rank,
             *module, kTpWorldSize);
  args.phase_epoch_counters = static_cast<std::uint32_t*>(epochs.data_ptr());
  args.x_gamma = static_cast<NormGain const*>(x_gamma.data_ptr());
  args.q_gamma = static_cast<NormGain const*>(q_gamma.data_ptr());
  args.kv_gamma = static_cast<NormGain const*>(kv_gamma.data_ptr());
  args.q_a_weight = native_weight(q_a_weight, q_a_scale_inv, 32, 8, false);
  args.kv_a_weight = native_weight(kv_a_with_mqa_weight, kv_a_with_mqa_scale_inv, 32, 8, false);
  args.q_b_weight = native_weight(q_b_weight, q_b_scale_inv, 32, 8, false);
  args.k_absorb_weight = native_weight(kv_b_weight, kv_b_scale_inv, 192, 0, true);
  args.v_absorb_weight = native_weight(kv_b_weight, kv_b_scale_inv, 32, 4, false);
  args.o_proj_weight = native_weight(o_proj_weight, o_proj_scale_inv, 48, 4, false);
  args.kv_cache = kv_cache.data_ptr();
  args.pe_cache = pe_cache.data_ptr();
  args.kv_cache_row_stride = kv_cache.stride(0);
  args.pe_cache_row_stride = pe_cache.stride(0);
  args.max_seq_len = kv_cache.size(0);
  args.rope_layout = layout;
  encode_tma_2d(&args.kv_cache_tma, CU_TENSOR_MAP_DATA_TYPE_UINT8, args.kv_cache,
      kKvLatentDim, args.max_seq_len, args.kv_cache_row_stride, 128, 1,
      CU_TENSOR_MAP_SWIZZLE_128B);
  auto const device = x_gamma.device();
  return tvm::ffi::Function::FromTyped(
      [args, device, module](TensorView x, TensorView selected_indices, TensorView selected_counts,
          TensorView cache_positions, TensorView rope, TensorView rope_positions,
          TensorView residual, TensorView local_output, TensorView final_output) {
    auto launch_args = args;
    check_token_matrix<bf16_t>(x, "x", kTokens, kHidden);
    check_token_matrix<bf16_t>(residual, "residual", kTokens, kHidden);
    check_token_matrix<bf16_t>(local_output, "local_output", kTokens, kHidden);
    check_token_matrix<bf16_t>(final_output, "final_output", kTokens, kHidden);
    check_tensor<std::int32_t>(selected_indices, "selected_indices", "int32", {kTokens, 2048});
    check_tensor<std::int32_t>(selected_counts, "selected_counts", "int32", {kTokens});
    check_positions(cache_positions, "cache_positions", args.rope_layout == 0);
    check_positions(rope_positions, "rope_positions", args.rope_layout == 0);
    require(is_type<float>(rope.dtype()) && rope.is_contiguous() &&
        rope.ndim() == 2 && rope.size(1) == 64 &&
        (args.rope_layout == 1 ? rope.size(0) > 0 : rope.size(0) == kTokens),
        "invalid RoPE table");
    require(x.device().device_type == device.device_type &&
        x.device().device_id == device.device_id, "input device differs from binding");
    for (auto tensor : {selected_indices, selected_counts, cache_positions, rope,
                         rope_positions, residual, local_output, final_output})
      check_same_device(tensor, x, "dynamic tensor");
    check_alignment(selected_indices, "selected_indices", 4);
    check_alignment(selected_counts, "selected_counts", 4);
    check_alignment(rope, "rope", 4);
    launch_args.x = x.data_ptr();
    launch_args.selected_indices = static_cast<std::int32_t const*>(selected_indices.data_ptr());
    launch_args.selected_counts = static_cast<std::int32_t const*>(selected_counts.data_ptr());
    launch_args.cache_positions = cache_positions.data_ptr();
    launch_args.cache_position_bytes = cache_positions.dtype().bits / 8;
    launch_args.rope_freqs = static_cast<float const*>(rope.data_ptr());
    launch_args.rope_positions = rope_positions.data_ptr();
    launch_args.rope_position_bytes = rope_positions.dtype().bits / 8;
    launch_args.residual = residual.data_ptr();
    launch_args.local_output = local_output.data_ptr();
    launch_args.final_output = final_output.data_ptr();
    module->launch(launch_args);
  });
}

tvm::ffi::Array<tvm::ffi::Any> load(
    tvm::ffi::Bytes cubin, tvm::ffi::String symbol, std::int64_t device) {
  auto module = Module::load<Glm5MegaMlaArgs>(cubin, symbol, device, kTokens, kTpWorldSize);
  auto bind_weights = bind_module(module, &bind);
  return {module->values(), bind_weights};
}
}  // namespace glm5::mla::host
