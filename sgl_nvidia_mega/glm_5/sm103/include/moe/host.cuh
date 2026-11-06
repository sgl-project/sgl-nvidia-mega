// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <limits>
#include <moe/launch.cuh>
#include <common/host_helpers.cuh>

namespace glm5::moe::host {
using tvm::ffi::TensorView;
using namespace ::glm5::host_detail;
tvm::ffi::Function bind(std::shared_ptr<Module> module, TensorView rms_gamma, TensorView router_weight,
    TensorView router_bias, TensorView up_gate_weight, TensorView up_gate_scale,
    TensorView down_weight, TensorView down_scale, TensorView workspace,
    TensorView epochs, tvm::ffi::Array<std::int64_t> peers, std::int64_t rank) {
  constexpr int h = 6144, inter = kFusedMoeLocalInter, tokens = kFusedMoeSeqLen;
  check_tensor<NormGain>(rms_gamma, "rms_gamma", kNormDtype, {h});
  check_tensor<bf16_t>(router_weight, "router_weight", "bfloat16", {256, h});
  check_tensor<float>(router_bias, "router_bias", "float32", {256});
  check_tensor<fp8_e4m3_t>(up_gate_weight, "up_gate_weight", "float8_e4m3fn", {257, 2 * inter, h});
  check_tensor<float>(up_gate_scale, "up_gate_scale", "float32", {257, 2 * inter / 128, 48});
  check_tensor<fp8_e4m3_t>(down_weight, "down_weight", "float8_e4m3fn", {257, h, inter});
  check_tensor<float>(down_scale, "down_scale", "float32", {257, 48, inter / 128});
  for (auto tensor : {rms_gamma, router_weight, router_bias, up_gate_weight,
                       up_gate_scale, down_weight, down_scale}) {
    check_same_device(tensor, rms_gamma, "bound weight");
    check_alignment(tensor, "bound weight", 16);
  }
  ContextGuard guard(module->context);
  Glm5MegaMoeArgs args{};
  bind_state(args, rms_gamma, workspace, epochs, peers, rank,
             *module, kFusedMoeTpSize);
  args.ar_epoch_counters = static_cast<std::uint32_t*>(epochs.data_ptr());
  args.rms_gamma = static_cast<NormGain const*>(rms_gamma.data_ptr());
  args.router_weight = router_weight.data_ptr();
  args.router_bias = static_cast<float const*>(router_bias.data_ptr());
  args.up_gate_weight = up_gate_weight.data_ptr();
  args.up_gate_scale = static_cast<float const*>(up_gate_scale.data_ptr());
  args.down_weight = down_weight.data_ptr();
  args.down_scale = static_cast<float const*>(down_scale.data_ptr());
  encode_expert_tma_descriptors(&args);
  auto const device = rms_gamma.device();
  return tvm::ffi::Function::FromTyped(
      [args, device, module](TensorView x, TensorView residual,
          tvm::ffi::Optional<TensorView> local_output, TensorView final_output,
          TensorView selected_indices, TensorView selected_probs) {
    auto launch_args = args;
    check_token_matrix<bf16_t>(x, "x", tokens, h);
    check_token_matrix<bf16_t>(residual, "residual", tokens, h);
    check_token_matrix<bf16_t>(final_output, "final_output", tokens, h);
    check_token_matrix<std::int32_t>(selected_indices, "selected_indices", tokens, 8);
    check_token_matrix<float>(selected_probs, "selected_probs", tokens, 8);
    require(x.device().device_type == device.device_type &&
        x.device().device_id == device.device_id, "input device differs from binding");
    for (auto tensor : {residual, final_output, selected_indices, selected_probs})
      check_same_device(tensor, x, "dynamic tensor");
    if (local_output.has_value()) {
      check_token_matrix<bf16_t>(local_output.value(), "local_output", tokens, h);
      check_same_device(local_output.value(), x, "local_output");
      launch_args.local_output = local_output.value().data_ptr();
    } else {
      launch_args.local_output = nullptr;
    }
    launch_args.x = x.data_ptr();
    launch_args.residual = residual.data_ptr();
    launch_args.final_output = final_output.data_ptr();
    launch_args.selected_indices = static_cast<std::int32_t*>(selected_indices.data_ptr());
    launch_args.selected_probs = static_cast<float*>(selected_probs.data_ptr());
    module->launch(launch_args);
  });
}

tvm::ffi::Array<tvm::ffi::Any> load(
    tvm::ffi::Bytes cubin, tvm::ffi::String symbol, std::int64_t device) {
  auto module = Module::load<Glm5MegaMoeArgs>(cubin, symbol, device, kFusedMoeSeqLen, kFusedMoeTpSize);
  auto bind_weights = bind_module(module, &bind);
  return {module->values(), bind_weights};
}
}  // namespace glm5::moe::host
