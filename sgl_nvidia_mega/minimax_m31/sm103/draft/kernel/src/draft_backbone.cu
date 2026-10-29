// MiniMax-M3.1 DSpark draft backbone, TP4, one request's seven-row block: the
// five dense decoder layers and the final norm in one cooperative launch.
//
// Every layer runs SGLang's operators in order, with their rounding points:
//   Gemma RMSNorm -> MXFP8 QKV -> per-head Gemma q/k norm + NeoX RoPE (K/V to
//   the cache) -> SWA attention over the last 4096 keys (bidirectional inside
//   the block) -> MXFP8 o_proj -> TP4 sum + residual + Gemma RMSNorm -> MXFP8
//   gate_up -> SwiGLU-OAI -> MXFP8 down -> TP4 sum + residual + the next norm.
//
// One CTA per SM. The last warp of every CTA streams the CTA's weight tiles for
// the whole call (every layer and GEMM, in consumption order) through an mbarrier
// ring, one tensor-map TMA op (16 rows x 2048 K bytes, 128-byte swizzle) plus one
// bulk copy of the tile's scales per stage; it never waits for
// activations, so the weight stream overlaps the dependent phases. Warps 0-7
// run the phases. Every GEMM is cut into 16-row full-K units (gate_up: 8 gate
// rows + the matching 8 up rows); the eight warps split K and add their sums in
// warp order. qkv units store BF16 rows, gate_up units apply SwiGLU, o_proj and
// down units push their BF16 rows (this rank's TP4 partial) straight to every
// rank. The attention CTAs apply the q/k norm + RoPE themselves and use the
// block's fresh K/V rows directly; the down CTAs quantize the SwiGLU output.
// The GEMM inner loop converts e4m3 pairs to f16 for two exact m16n8k16 HMMAs per
// 32-wide group and applies the two UE8M0 scales to the FP32 group result with
// packed f32x2 multiplies/FMAs (activation scales decoded once per phase).
//
// Hand-overs are self-validating 8-byte words that carry the call generation, so
// every reader polls the data itself (no release fence, no counter round trip):
// the normed MXFP8 activations, the qkv rows, the SwiGLU rows, the o_proj input
// and the TP4 partials; only the FP32 attention partials use a release counter.
// Every rank adds the four TP4 partials in rank order, so all ranks hold
// identical bits.

#include "include/sm100/collective/op_attention.cuh"
#include "include/sm100/collective/op_combine.cuh"
#include "include/sm100/collective/op_embed.cuh"
#include "include/sm100/collective/op_prefetch.cuh"
#include "include/sm100/collective/op_tp_norm.cuh"
#include "include/sm100/collective/op_weight_producer.cuh"

namespace m31_draft {
namespace {

__device__ void consumer_main(Ctx &c) {
  AttnStaging const staging = stage_attention_issue(c);
  embedding_phase(c);
  consumer_sync();
  if (c.tid == 0)
    mbar_arrive(smem_u32(&c.misc->go));
  stage_attention_store(c, staging);
  consumer_sync();
  // Norm stage s: the embedding's (s = 0), then per layer the post-attention (2l + 1) and the next
  // input / final (2l + 2) TP4 boundaries. One call site per OP keeps a single inlined copy of
  // each.
  for (int s = 0;; ++s) {
    int const l = s / 2;
    bool const last = s == 2 * kLayers;
    uint16_t const *weight = last           ? c.a.final_norm
                             : (s & 1) != 0 ? c.a.layer[l].post_norm
                                            : c.a.layer[l].input_norm;
    norm_phase(c, s, weight, last);
    if (last)
      break;
    if ((s & 1) == 0) {
      qkv_phase(c, l);
      attention_phase(c, l);
      combine_phase(c, l);
      o_phase(c, l);
    } else {
      gate_up_phase(c, l);
      down_phase(c, l);
    }
  }
  // Every CTA read this call's generation at entry; the next call starts after this one ends.
  if (c.cta == 0 && c.tid == 0) {
    wait_count(at<uint32_t>(c.arena, kStartWord), c.epoch * uint32_t(c.grid));
    *reinterpret_cast<uint32_t volatile *>(c.arena + kCallWord) = c.epoch;
  }
}

__global__ void __launch_bounds__(kThreads, 1)
    draft_backbone_kernel(const __grid_constant__ Args a, const __grid_constant__ WeightMaps maps) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  // The swizzled TMA boxes need a 1024-byte aligned ring.
  uint8_t *smem = smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u);
  SmemMisc *misc = reinterpret_cast<SmemMisc *>(smem + kSmemMisc);
  uint8_t *arena = a.arena[a.rank];
  int const tid = int(threadIdx.x);
  if (tid == 0) {
    init_ring(misc);
    mbar_init(smem_u32(&misc->go), 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    misc->epoch = *reinterpret_cast<uint32_t volatile *>(arena + kCallWord) + 1u;
    misc->kv_len = *a.kv_len;
    atomicAdd(at<uint32_t>(arena, kStartWord), 1u);
  }
  if (tid < kRows) {
    misc->slot_mapping[tid] = a.slot_mapping[tid];
    misc->positions[tid] = a.positions[tid];
    misc->input_ids[tid] = a.input_ids[tid];
  }
  __syncthreads();
  Ring const ring{smem_u32(smem + kSmemRing), misc};
  if (tid >= kConsumerThreads) {
    entry_prefetch(a, misc, int(blockIdx.x), tid - kConsumerThreads);
    producer(a, maps, Producer{ring}, misc, int(blockIdx.x), int(gridDim.x),
             tid - kConsumerThreads);
    return;
  }
  Ctx c{a,   arena,    smem,     misc,          misc->epoch, int(blockIdx.x), int(gridDim.x),
        tid, tid / 32, tid % 32, Consumer{ring}};
  consumer_main(c);
}

// The plan state is 64-byte aligned; the maps sit at the first 128-byte boundary inside it
// (the state's address is fixed for the plan, so prepare_plan and launch find the same copy).
WeightMaps *plan_maps(void const *state) {
  uintptr_t const raw =
      reinterpret_cast<uintptr_t>(static_cast<PlanState const *>(state)->maps_raw);
  return reinterpret_cast<WeightMaps *>((raw + 127) & ~uintptr_t(127));
}
} // namespace

cudaError_t configure_kernel(int device, int *needed_ctas, int *resident_ctas) {
  cudaError_t status = cudaFuncSetAttribute(
      draft_backbone_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
  if (status != cudaSuccess)
    return status;
  cudaDeviceProp properties{};
  if ((status = cudaGetDeviceProperties(&properties, device)) != cudaSuccess)
    return status;
  int per_sm = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, draft_backbone_kernel, kThreads,
                                                         kSmemBytes);
  if (status != cudaSuccess)
    return status;
  *needed_ctas = properties.multiProcessorCount;
  *resident_ctas = properties.cooperativeLaunch ? per_sm * properties.multiProcessorCount : 0;
  return cudaSuccess;
}

// The grid (one CTA per SM) and the weight tensor maps (pointers and shapes only).
cudaError_t prepare_plan(Args const &args, void *state) {
  PlanState *plan = static_cast<PlanState *>(state);
  int device = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess)
    return status;
  if ((status = cudaDeviceGetAttribute(&plan->grid, cudaDevAttrMultiProcessorCount, device)) !=
      cudaSuccess)
    return status;
  void *entry = nullptr;
  cudaDriverEntryPointQueryResult query = cudaDriverEntryPointSymbolNotFound;
  if ((status = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &entry, 12000,
                                                 cudaEnableDefault, &query)) != cudaSuccess)
    return status;
  if (entry == nullptr || query != cudaDriverEntryPointSuccess)
    return cudaErrorNotSupported;
  auto const encode = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(entry);
  WeightMaps *maps = plan_maps(state);
  for (int l = 0; l < kLayers; ++l) {
    LayerArgs const &layer = args.layer[l];
    if (!GemmQkv::encode(encode, layer, l, maps) || !GemmO::encode(encode, layer, l, maps) ||
        !GemmGateUp::encode(encode, layer, l, maps) || !GemmDown::encode(encode, layer, l, maps))
      return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

cudaError_t launch(Args const &args, void const *state, cudaStream_t stream) {
  PlanState const *plan = static_cast<PlanState const *>(state);
  void *parameters[] = {const_cast<Args *>(&args), plan_maps(state)};
  return cudaLaunchCooperativeKernel(reinterpret_cast<void *>(draft_backbone_kernel),
                                     dim3(plan->grid), dim3(kThreads), parameters, kSmemBytes,
                                     stream);
}
} // namespace m31_draft
