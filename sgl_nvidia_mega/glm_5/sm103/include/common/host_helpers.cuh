// SPDX-License-Identifier: Apache-2.0
#pragma once

// Host-only FFI, validation and CUDA Driver helpers. Device stages never include this.
#include <cuda.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>

#include <array>
#include <initializer_list>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <common/metadata.cuh>
#include <common/norm.h>

namespace glm5::host_detail {
using tvm::ffi::TensorView;
using bf16_t = __nv_bfloat16;
struct fp8_e4m3_t {};

template <class... Message>
inline void require(bool condition, Message const&... message) {
  if (condition) return;
  std::ostringstream text;
  (text << ... << message);
  throw std::runtime_error(text.str());
}

template <class T>
inline bool is_type(DLDataType dtype) {
  int code = std::is_same_v<T, bf16_t> ? kDLBfloat :
      std::is_same_v<T, fp8_e4m3_t> ? kDLFloat8_e4m3fn :
      std::is_floating_point_v<T> ? kDLFloat : std::is_signed_v<T> ? kDLInt : kDLUInt;
  return dtype.code == code && dtype.bits == sizeof(T) * 8 && dtype.lanes == 1;
}

inline void check_driver(CUresult status, char const* operation) {
  if (status == CUDA_SUCCESS) return;
  char const* description = nullptr;
  static_cast<void>(cuGetErrorString(status, &description));
  require(false, operation, ": ", description ? description : "unknown CUDA error");
}

struct ContextGuard {
  explicit ContextGuard(CUcontext context) {
    CUcontext current = nullptr;
    check_driver(cuCtxGetCurrent(&current), "cuCtxGetCurrent");
    changed = current != context;
    if (changed) check_driver(cuCtxPushCurrent(context), "cuCtxPushCurrent");
  }
  ~ContextGuard() {
    if (changed) { CUcontext previous; static_cast<void>(cuCtxPopCurrent(&previous)); }
  }
  bool changed;
};

inline bool shape_matches(TensorView tensor, std::initializer_list<std::int64_t> shape) {
  if (tensor.ndim() != static_cast<int>(shape.size())) return false;
  int dim = 0;
  for (auto extent : shape) if (tensor.size(dim++) != extent) return false;
  return true;
}

inline void check_alignment(TensorView tensor, char const* name, std::size_t alignment) {
  require(reinterpret_cast<std::uintptr_t>(tensor.data_ptr()) % alignment == 0,
          name, ": invalid pointer alignment (expected ", alignment, " bytes)");
}

inline void check_device(TensorView tensor, int device, char const* name) {
  require(tensor.device().device_type == kDLCUDA && tensor.device().device_id == device,
          name, ": device differs from binding");
}

inline void check_same_device(TensorView tensor, TensorView reference, char const* name) {
  check_device(tensor, reference.device().device_id, name);
}

template <class Element>
inline void check_tensor(TensorView tensor, char const* name, char const*,
                         std::initializer_list<std::int64_t> shape) {
  require(tensor.device().device_type == kDLCUDA && is_type<Element>(tensor.dtype()) &&
          tensor.is_contiguous() && shape_matches(tensor, shape),
          name, ": invalid device, dtype, shape or stride");
}

template <class Element>
inline void check_token_matrix(TensorView tensor, char const* name, int rows, int width) {
  bool shape = tensor.ndim() == 2 && tensor.size(0) == rows && tensor.size(1) == width;
  shape = shape || (tensor.ndim() == 3 && tensor.size(0) > 0 &&
      tensor.size(0) * tensor.size(1) == rows && tensor.size(2) == width);
  require(tensor.device().device_type == kDLCUDA && is_type<Element>(tensor.dtype()) &&
          tensor.is_contiguous() && shape, name, ": invalid token matrix");
  check_alignment(tensor, name, 16);
}

inline void encode_tma_2d(CUtensorMap* map, CUtensorMapDataType type,
    void const* address, std::uint64_t inner, std::uint64_t outer, std::uint64_t stride,
    std::uint32_t box_inner, std::uint32_t box_outer, CUtensorMapSwizzle swizzle) {
  cuuint64_t dims[] = {inner, outer}, strides[] = {stride};
  cuuint32_t box[] = {box_inner, box_outer}, elements[] = {1, 1};
  check_driver(cuTensorMapEncodeTiled(map, type, 2, const_cast<void*>(address),
      dims, strides, box, elements, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      "cuTensorMapEncodeTiled");
}

inline void encode_tma_3d(CUtensorMap* map, CUtensorMapDataType type,
    void const* address, std::uint64_t inner, std::uint64_t middle, std::uint64_t outer,
    std::uint64_t middle_stride, std::uint64_t outer_stride,
    std::uint32_t box_inner, std::uint32_t box_middle, std::uint32_t box_outer,
    CUtensorMapSwizzle swizzle) {
  cuuint64_t dims[] = {inner, middle, outer}, strides[] = {middle_stride, outer_stride};
  cuuint32_t box[] = {box_inner, box_middle, box_outer}, elements[] = {1, 1, 1};
  check_driver(cuTensorMapEncodeTiled(map, type, 3, const_cast<void*>(address),
      dims, strides, box, elements, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      "cuTensorMapEncodeTiled");
}

struct Module {
  CUmodule module = nullptr;
  CUfunction function = nullptr;
  CUcontext context = nullptr;
  int device = -1;
  KernelMetadata metadata{};

  ~Module() {
    // No synchronization is introduced on the invocation path.
    if (module) {
      CUcontext previous = nullptr;
      if (cuCtxPushCurrent(context) == CUDA_SUCCESS) {
        static_cast<void>(cuModuleUnload(module));
        static_cast<void>(cuCtxPopCurrent(&previous));
      }
    }
  }

  template <class Args>
  static std::shared_ptr<Module> load(tvm::ffi::Bytes cubin, tvm::ffi::String symbol,
                                      int device, int rows, int tp) {
    auto result = std::make_shared<Module>();
    result->device = device;
    CUdevice current = -1;
    check_driver(cuCtxGetCurrent(&result->context), "cuCtxGetCurrent");
    require(result->context != nullptr, "initialize the CUDA primary context before load");
    check_driver(cuCtxGetDevice(&current), "cuCtxGetDevice");
    require(current == device, "FFI module belongs to a different CUDA device");
    check_driver(cuModuleLoadData(&result->module, cubin.data()), "cuModuleLoadData");
    CUdeviceptr pointer;
    std::size_t size;
    check_driver(cuModuleGetGlobal(&pointer, &size, result->module, "glm5_metadata"),
                 "cuModuleGetGlobal");
    require(size == sizeof(KernelMetadata), "invalid GLM5 metadata size");
    auto& meta = result->metadata;
    check_driver(cuMemcpyDtoH(&meta, pointer, size), "cuMemcpyDtoH");
    require(meta.version == 1 && meta.rows == rows && meta.tp_size == tp &&
            meta.args_size == sizeof(Args) && meta.args_alignment == alignof(Args),
            "incompatible GLM5 cubin/FFI ABI");
    check_driver(cuModuleGetFunction(&result->function, result->module, symbol.c_str()),
                 "cuModuleGetFunction");
    check_driver(cuFuncSetAttribute(result->function,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, meta.shared_bytes),
        "cuFuncSetAttribute");
    check_driver(cuFuncSetAttribute(result->function,
        CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100), "cuFuncSetAttribute");
    int occupancy = 0;
    check_driver(cuOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, result->function, meta.threads, meta.shared_bytes), "cuOccupancy");
    require(occupancy == 1, "GLM5 requires one CTA per SM");
    return result;
  }

  tvm::ffi::Array<std::int64_t> values() const {
    auto const& m = metadata;
    tvm::ffi::Array<std::int64_t> result;
    for (auto value : {m.version, m.rows, m.tp_size, m.workspace_bytes, m.epoch_count,
        m.peer_bytes, m.workspace_fill, m.required_sms, m.native_cvt, m.grid, m.threads,
        m.shared_bytes, m.args_size, m.args_alignment}) result.push_back(value);
    return result;
  }

  template <class Args>
  void launch(Args& args) const {
    ContextGuard guard(context);
    CUlaunchConfig config{};
    config.gridDimX = metadata.grid; config.gridDimY = config.gridDimZ = 1;
    config.blockDimX = metadata.threads; config.blockDimY = config.blockDimZ = 1;
    config.sharedMemBytes = metadata.shared_bytes;
    config.hStream = reinterpret_cast<CUstream>(TVMFFIEnvGetStream(kDLCUDA, device));
    // numAttrs remains zero: GLM5 does not enable PDL.
    void* parameters[] = {&args};
    check_driver(cuLaunchKernelEx(&config, function, parameters, nullptr), "cuLaunchKernelEx");
  }
};

template <class... Args>
inline tvm::ffi::Function bind_module(std::shared_ptr<Module> module,
    tvm::ffi::Function (*binder)(std::shared_ptr<Module>, Args...)) {
  return tvm::ffi::Function::FromTyped(
      [module, binder](Args... values) { return binder(module, values...); });
}

template <class Args>
inline void bind_state(Args& args, TensorView reference, TensorView workspace,
    TensorView epochs, tvm::ffi::Array<std::int64_t> peers, std::int64_t rank,
    Module const& module, int tp_size) {
  auto const& meta = module.metadata;
  check_device(reference, module.device, "weight");
  require(workspace.ndim() == 1 && workspace.is_contiguous() &&
      is_type<std::uint8_t>(workspace.dtype()) &&
      workspace.size(0) >= static_cast<std::int64_t>(meta.workspace_bytes),
      "workspace is smaller than compiled Config metadata");
  check_tensor<std::int32_t>(epochs, "epochs", "int32",
                            {static_cast<std::int64_t>(meta.epoch_count)});
  check_same_device(workspace, reference, "workspace");
  check_same_device(epochs, reference, "epochs");
  check_alignment(workspace, "workspace", 128);
  check_alignment(epochs, "epochs", 4);
  int world = static_cast<int>(peers.size());
  require(world == 1 || world == tp_size, "expected one local peer or tp_size peers");
  require(rank >= 0 && rank < world, "invalid peer rank");
  for (int peer = 0; peer < world; ++peer) {
    auto address = static_cast<std::uintptr_t>(peers[peer]);
    require(address && address % 128 == 0, "invalid peer pointer alignment");
    for (int previous = 0; previous < peer; ++previous)
      require(peers[peer] != peers[previous], "peer pointers must be unique");
    args.peer_workspaces[peer] = reinterpret_cast<void*>(address);
  }
  args.workspace = workspace.data_ptr();
  args.workspace_bytes = workspace.size(0);
  args.rank = rank;
  args.world_size = world;
}
}  // namespace glm5::host_detail
