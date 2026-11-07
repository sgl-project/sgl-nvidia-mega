# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""BF16 circular checkpoint materialization; the runtime retains fold policy and cursors.

The FP32 MMA and scan accumulation order differs from Triton; this is a
numerical, not bitwise path.
"""

from __future__ import annotations

import ctypes
import logging
from functools import cache
from typing import Optional

import torch
from sglang.kernels.plugin_kernels.nvidia.spec import (
    CheckpointCommitView,
    HooksFactory,
    SpeculativeRuntimeHooks,
)

from sgl_nvidia_mega.common import jit

SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py": (
        "CheckpointCommitView",
        "HooksFactory",
        "SpeculativeRuntimeHooks",
    ),
}
logger = logging.getLogger(__name__)

_SRC = r"""
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;
#define KD 128
#define VD 128
#define NW 4
#define FULL 0xffffffffu
#ifndef EARLY
#define EARLY 0  // read the cursors before staging: a row that neither folds nor tracks leaves first
#endif

struct alignas(16) U4 { u32 x, y, z, w; };

struct Params {
  u16* h0; long long h0_layer, h0_slot;          // [layers, slots, HV, V, K]
  const u16* d; long long d_layer, d_slot;       // [layers, slots, HV, L, V]
  const u16* k; long long k_layer, k_slot;       // [layers, slots, H, L, K]
  const float* g; long long g_layer, g_slot;     // [layers, slots, HV, L]
  const u16* dr; long long dr_layer, dr_slot;    // low parts (layout of d), or null
  const u16* kr; long long kr_layer, kr_slot;    // low parts (layout of k), or null
  const void* sidx; const void* ridx; const int* write_pos; const int* cache_base;
  const signed char* is_flush; const void* acc; const void* tidx; const void* tstep;
  int sidx64, ridx64, acc64, tidx64, tstep64;
  int B;
  long long null_block;
};

// shared memory: the checkpoint [V][K], then the ring's keys, key low parts, updates, update low parts
#define SH_STATE 0
#define SH_K (VD * KD)
#define SH_KR (SH_K + RING * KD)
#define SH_D (SH_KR + RING * KD)
#define SH_DR (SH_D + RING * VD)
#define SH_END (SH_DR + RING * VD)

__device__ __forceinline__ float bflo(u32 x) { return __int_as_float(x << 16); }
__device__ __forceinline__ float bfhi(u32 x) { return __int_as_float(x & 0xffff0000u); }
__device__ __forceinline__ float bf2f(u16 x) { return __int_as_float(((u32)x) << 16); }
__device__ __forceinline__ u32 pack2(float lo, float hi) {
  u32 r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ u32 word(const U4& v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }
__device__ __forceinline__ void mma16816(float (&d)[4], u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma1688(float (&d)[4], u32 a0, u32 a1, u32 b0) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(b0));
}
__device__ __forceinline__ long long ld_idx(const void* p, int i, int i64) {
  return i64 ? reinterpret_cast<const long long*>(p)[i] : (long long)reinterpret_cast<const int*>(p)[i];
}
__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, u32 bytes, u64* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait0(u64* bar) {
  u32 ready = 0;
  do {
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }"
                 : "=r"(ready) : "r"(smem_u32(bar)) : "memory");
  } while (!ready);
}

// A fragments of bf16 hi / lo of (update[t][v] * decay[t]): rows v = vb + 16 m + r (+ 8), entries t = 2q, 2q + 1
// (a0 / a1) and 2q + 8, 2q + 9 (a2 / a3)
__device__ __forceinline__ void a_frags(const float (&u)[2][2][4], const float (&dec)[4], u32 (&AH)[2][4],
                                        u32 (&AL)[2][4]) {
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int h = i & 1, jp = i >> 1;
      const float lo = u[m][h][2 * jp] * dec[2 * jp], hi = u[m][h][2 * jp + 1] * dec[2 * jp + 1];
      AH[m][i] = pack2(lo, hi);
      AL[m][i] = pack2(lo - bflo(AH[m][i]), hi - bfhi(AH[m][i]));
    }
}

// out = bf16(old e^total + K^T D) for this warp's rows, from the shared checkpoint and keys; B column n = r holds
// k = 32c + 8 (r >> 1) + 2e + (r & 1), so the C fragment of lane q is word e of its 16-byte chunk c
template <bool K16>
__device__ __forceinline__ void fold_store(const u16* sm, const int (&ph)[4], const u32 (&AH)[2][4],
                                           const u32 (&AL)[2][4], float etot, bool resid, u16* dst, int vb, int r,
                                           int q) {
  const u16* sk = sm + SH_K;
  const u16* skr = sm + SH_KR;
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    U4 H[2][2];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int h = 0; h < 2; ++h)
        H[m][h] = *reinterpret_cast<const U4*>(sm + SH_STATE + (vb + 16 * m + 8 * h + r) * KD + 32 * c + 8 * q);
    u32 out[2][2][4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int kk = 32 * c + 8 * (r >> 1) + 2 * e + (r & 1);
      const u32 b0 = (u32)sk[ph[0] * KD + kk] | ((u32)sk[ph[1] * KD + kk] << 16);
      const u32 b1 = K16 ? ((u32)sk[ph[2] * KD + kk] | ((u32)sk[ph[3] * KD + kk] << 16)) : 0u;
      u32 r0 = 0u, r1 = 0u;
      if (resid) {
        r0 = (u32)skr[ph[0] * KD + kk] | ((u32)skr[ph[1] * KD + kk] << 16);
        if (K16) r1 = (u32)skr[ph[2] * KD + kk] | ((u32)skr[ph[3] * KD + kk] << 16);
      }
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        float a[4] = {0.f, 0.f, 0.f, 0.f};
        if (K16) {
          mma16816(a, AH[m][0], AH[m][1], AH[m][2], AH[m][3], b0, b1);
          mma16816(a, AL[m][0], AL[m][1], AL[m][2], AL[m][3], b0, b1);
          if (resid) {
            mma16816(a, AH[m][0], AH[m][1], AH[m][2], AH[m][3], r0, r1);
            mma16816(a, AL[m][0], AL[m][1], AL[m][2], AL[m][3], r0, r1);
          }
        } else {  // entries 8-15 carry no weight: the k8 half of the same products
          mma1688(a, AH[m][0], AH[m][1], b0);
          mma1688(a, AL[m][0], AL[m][1], b0);
          if (resid) {
            mma1688(a, AH[m][0], AH[m][1], r0);
            mma1688(a, AL[m][0], AL[m][1], r0);
          }
        }
        const u32 w0 = word(H[m][0], e), w1 = word(H[m][1], e);
        out[m][0][e] = pack2(fmaf(bflo(w0), etot, a[0]), fmaf(bfhi(w0), etot, a[1]));
        out[m][1][e] = pack2(fmaf(bflo(w1), etot, a[2]), fmaf(bfhi(w1), etot, a[3]));
      }
    }
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        U4 v4;
        v4.x = out[m][h][0]; v4.y = out[m][h][1]; v4.z = out[m][h][2]; v4.w = out[m][h][3];
        *reinterpret_cast<U4*>(dst + (vb + 16 * m + 8 * h + r) * KD + 32 * c + 8 * q) = v4;
      }
  }
}

// the window's decay: masked inclusive prefix of the log-decays (lane t holds entry t), e^{total - prefix_j}
__device__ __forceinline__ float window(float gt, int len, int lane, const int (&jj)[4], float (&dec)[4]) {
  float pf = lane < len ? gt : 0.f;
#pragma unroll
  for (int dd = 1; dd < 32; dd <<= 1) {
    const float o = __shfl_up_sync(FULL, pf, dd);
    if (lane >= dd) pf += o;
  }
  const float tot = __shfl_sync(FULL, pf, 31);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float pj = __shfl_sync(FULL, pf, jj[i]);
    dec[i] = jj[i] < len ? __expf(tot - pj) : 0.f;
  }
  return __expf(tot);
}

extern "C" __global__ void __launch_bounds__(NW * 32) gdn_replayssm_commit(const Params p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  __shared__ u64 bar;
  constexpr int GRP = HV / HK;
  const int cta = blockIdx.x;
  const int hv = cta % HV;
  const int n = (cta / HV) % p.B;
  const int layer = cta / (HV * p.B);
  const int hk = hv / GRP;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, r = lane >> 2, q = lane & 3;
  const int vb = warp * 32;

  const long long slot = ld_idx(p.sidx, n, p.sidx64);
  const long long rid = ld_idx(p.ridx, n, p.ridx64);
  if (slot <= p.null_block) return;
  const bool resid = p.dr != nullptr && p.kr != nullptr;
  u16* sm = reinterpret_cast<u16*>(smem_raw);
#if EARLY
  const int n_hist = p.write_pos[rid];
  const bool fold = p.is_flush[rid] != 0;
  const int base = p.cache_base[rid];
  long long tslot = p.null_block;
  int track_len = 0;
  if (HAS_TRACK) {
    tslot = ld_idx(p.tidx, n, p.tidx64);
    const long long ts = ld_idx(p.tstep, n, p.tstep64);
    if (!(tslot > p.null_block && ts >= 0)) tslot = p.null_block;
    track_len = n_hist - (int)ld_idx(p.acc, n, p.acc64) + (int)ts + 1;
  }
  const bool track = tslot > p.null_block;
  if (!fold && !track) return;
#endif
  // the checkpoint and the whole ring slice need only the two indices: copies first, cursors meanwhile
  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_u32(&bar)) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    const u32 bytes = 2 * (VD * KD + (resid ? 4 : 2) * RING * KD);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(&bar)), "r"(bytes)
                 : "memory");
    bulk_g2s(sm + SH_STATE, p.h0 + layer * p.h0_layer + slot * p.h0_slot + (long long)hv * VD * KD, 2 * VD * KD,
             &bar);
    bulk_g2s(sm + SH_K, p.k + layer * p.k_layer + rid * p.k_slot + (long long)hk * RING * KD, 2 * RING * KD, &bar);
    bulk_g2s(sm + SH_D, p.d + layer * p.d_layer + rid * p.d_slot + (long long)hv * RING * VD, 2 * RING * VD, &bar);
    if (resid) {
      bulk_g2s(sm + SH_KR, p.kr + layer * p.kr_layer + rid * p.kr_slot + (long long)hk * RING * KD, 2 * RING * KD,
               &bar);
      bulk_g2s(sm + SH_DR, p.dr + layer * p.dr_layer + rid * p.dr_slot + (long long)hv * RING * VD, 2 * RING * VD,
               &bar);
    }
  }
#if !EARLY
  const int n_hist = p.write_pos[rid];
  const bool fold = p.is_flush[rid] != 0;
  const int base = p.cache_base[rid];
  long long tslot = p.null_block;
  int track_len = 0;
  if (HAS_TRACK) {
    tslot = ld_idx(p.tidx, n, p.tidx64);
    const long long ts = ld_idx(p.tstep, n, p.tstep64);
    if (!(tslot > p.null_block && ts >= 0)) tslot = p.null_block;
    track_len = n_hist - (int)ld_idx(p.acc, n, p.acc64) + (int)ts + 1;
  }
  const bool track = tslot > p.null_block;
#endif
  const float gt = lane < RING ? p.g[layer * p.g_layer + rid * p.g_slot + hv * RING + ((base + lane) & (RING - 1))] : 0.f;
  const int jj[4] = {2 * q, 2 * q + 1, 2 * q + 8, 2 * q + 9};
  int ph[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) ph[i] = (base + jj[i]) & (RING - 1);
  __syncthreads();  // the barrier's init before anyone waits on it
  mbar_wait0(&bar);  // also before an early exit: no copy may land after the CTA is gone
  if (!fold && !track) return;
  // windows of at most 8 entries (every fold of a step of up to 8 draft rows) take the k8 products
  const bool k16 = (fold && n_hist > 8) || (track && track_len > 8);

  float u[2][2][4];  // updates (high + low, FP32): [m][half][entry jj]
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int v = vb + 16 * m + 8 * h + r;
        float x = 0.f;
        if (i < 2 || k16) {
          x = bf2f(sm[SH_D + ph[i] * VD + v]);
          if (resid) x += bf2f(sm[SH_DR + ph[i] * VD + v]);
        }
        u[m][h][i] = x;
      }
  if (fold) {
    float dec[4];
    const float etot = window(gt, n_hist, lane, jj, dec);
    u32 AH[2][4], AL[2][4];
    a_frags(u, dec, AH, AL);
    if (k16) fold_store<true>(sm, ph, AH, AL, etot, resid, p.h0 + layer * p.h0_layer + slot * p.h0_slot + (long long)hv * VD * KD,
               vb, r, q);
    else fold_store<false>(sm, ph, AH, AL, etot, resid, p.h0 + layer * p.h0_layer + slot * p.h0_slot + (long long)hv * VD * KD,
               vb, r, q);
  }
  if (track) {
    float dec[4];
    const float etot = window(gt, track_len, lane, jj, dec);
    u32 AH[2][4], AL[2][4];
    a_frags(u, dec, AH, AL);
    if (k16) fold_store<true>(sm, ph, AH, AL, etot, resid, p.h0 + layer * p.h0_layer + tslot * p.h0_slot + (long long)hv * VD * KD,
               vb, r, q);
    else fold_store<false>(sm, ph, AH, AL, etot, resid, p.h0 + layer * p.h0_layer + tslot * p.h0_slot + (long long)hv * VD * KD,
               vb, r, q);
  }
}
"""


class _Params(ctypes.Structure):
    _fields_ = [
        ("h0", ctypes.c_void_p),
        ("h0_layer", ctypes.c_int64),
        ("h0_slot", ctypes.c_int64),
        ("d", ctypes.c_void_p),
        ("d_layer", ctypes.c_int64),
        ("d_slot", ctypes.c_int64),
        ("k", ctypes.c_void_p),
        ("k_layer", ctypes.c_int64),
        ("k_slot", ctypes.c_int64),
        ("g", ctypes.c_void_p),
        ("g_layer", ctypes.c_int64),
        ("g_slot", ctypes.c_int64),
        ("dr", ctypes.c_void_p),
        ("dr_layer", ctypes.c_int64),
        ("dr_slot", ctypes.c_int64),
        ("kr", ctypes.c_void_p),
        ("kr_layer", ctypes.c_int64),
        ("kr_slot", ctypes.c_int64),
        ("sidx", ctypes.c_void_p),
        ("ridx", ctypes.c_void_p),
        ("write_pos", ctypes.c_void_p),
        ("cache_base", ctypes.c_void_p),
        ("is_flush", ctypes.c_void_p),
        ("acc", ctypes.c_void_p),
        ("tidx", ctypes.c_void_p),
        ("tstep", ctypes.c_void_p),
        ("sidx64", ctypes.c_int),
        ("ridx64", ctypes.c_int),
        ("acc64", ctypes.c_int),
        ("tidx64", ctypes.c_int),
        ("tstep64", ctypes.c_int),
        ("B", ctypes.c_int),
        ("null_block", ctypes.c_int64),
    ]


def _smem_bytes(ring: int) -> int:
    # the checkpoint [128][128] and four [ring][128] rings, BF16
    return 2 * (128 * 128 + 4 * ring * 128)


@cache
def _kernel(device: int, hk: int, hv: int, ring: int, has_track: bool):
    major, minor = torch.cuda.get_device_capability(device)
    defines = f"#define HK {hk}\n#define HV {hv}\n#define RING {ring}\n#define HAS_TRACK {int(has_track)}\n"
    cubin = jit.compile_source(
        defines + _SRC,
        "checkpoint_commit",
        f"sm_{major}{minor}",
        ("-std=c++17", "-default-device"),
    )
    kernel = jit.load(cubin, "gdn_replayssm_commit", (128, 1, 1), _smem_bytes(ring))
    cu = jit.driver()
    jit.check(
        cu.cuFuncSetAttribute(
            kernel.function,
            cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            _smem_bytes(ring),
        ),
        "checkpoint commit shared memory",
    )
    return kernel


def _kwargs(view: CheckpointCommitView) -> dict:
    return dict(
        checkpoint_state=view.checkpoint,
        d_cache=view.updates,
        k_cache=view.keys,
        g_cache=view.log_decays,
        d_residual_cache=view.update_residuals,
        k_residual_cache=view.key_residuals,
        state_batch_indices=view.state_indices,
        replay_indices=view.history_indices,
        write_pos=view.history_lengths,
        cache_base=view.history_offsets,
        is_flush=view.fold_mask,
        accept_lens=view.accepted_lengths,
        mamba_track_indices=view.track_state_indices,
        mamba_steps_to_track=view.track_steps,
        null_block_id=view.null_index,
    )


def supported(view: CheckpointCommitView) -> bool:
    st, d, k, g = view.checkpoint, view.updates, view.keys, view.log_decays
    if not (st.is_cuda and st.ndim == 5 and d.ndim == k.ndim == 5 and g.ndim == 4):
        return False
    layers, _, hv, v, width = st.shape
    batch = view.state_indices.numel()
    if not (batch and (v, width) == (128, 128) and k.shape[2] and hv % k.shape[2] == 0):
        return False
    history = d.shape[1]
    expected = ((st, (hv, 128, 128)), (d, (hv, 16, 128)), (k, (k.shape[2], 16, 128)))
    for tensor, tail in expected:
        if not (
            tensor.dtype == torch.bfloat16
            and tensor.shape[0] == layers
            and tuple(tensor.shape[2:]) == tail
            and tensor.stride()[2:] == (tail[1] * tail[2], tail[2], 1)
            and tensor.data_ptr() % 16 == 0
        ):
            return False
    if not (
        k.shape[1] == history
        and g.shape == (layers, history, hv, 16)
        and g.dtype == torch.float32
        and g.stride()[2:] == (16, 1)
    ):
        return False
    tensors = [st, d, k, g]
    if view.update_residuals is not None and view.key_residuals is not None:
        for residual, original in ((view.update_residuals, d), (view.key_residuals, k)):
            if not (
                residual.shape == original.shape
                and residual.dtype == original.dtype
                and residual.stride()[2:] == original.stride()[2:]
                and residual.data_ptr() % 16 == 0
            ):
                return False
            tensors.append(residual)
    for tensor, dtype, count in (
        (view.history_lengths, torch.int32, history),
        (view.history_offsets, torch.int32, history),
        (view.fold_mask, torch.int8, history),
    ):
        if (
            tensor.dtype != dtype
            or not tensor.is_contiguous()
            or tensor.numel() < count
        ):
            return False
        tensors.append(tensor)
    indices = [view.state_indices, view.history_indices, view.accepted_lengths]
    if view.track_state_indices is not None and view.track_steps is not None:
        indices += [view.track_state_indices, view.track_steps]
    for tensor in indices:
        if (
            tensor.ndim != 1
            or tensor.numel() != batch
            or not tensor.is_contiguous()
            or tensor.dtype not in (torch.int32, torch.int64)
        ):
            return False
        tensors.append(tensor)
    return all(t.is_cuda and t.device == st.device for t in tensors)


class _CheckpointCommit:
    def __init__(self, device: int, hk: int, hv: int):
        self.kernels = {
            track: _kernel(device, hk, hv, 16, track) for track in (False, True)
        }
        self.shape = (device, hk, hv)
        self.engaged = False
        logger.info(
            "NVIDIA megakernel: CUDA checkpoint commit built (BF16 fold policy unchanged)"
        )

    def __call__(self, view: CheckpointCommitView) -> bool:
        if not supported(view):
            return False
        st = view.checkpoint
        if (st.device.index, view.keys.shape[2], st.shape[2]) != self.shape:
            return False
        # All capability checks precede writes; launch failures propagate to the owner.
        has_track = (
            view.track_state_indices is not None and view.track_steps is not None
        )
        params = _params(**_kwargs(view))
        _launch(
            self.kernels[has_track],
            st.shape[0] * view.state_indices.numel() * st.shape[2],
            16,
            st.device,
            [params],
        )
        if not self.engaged:
            logger.info(
                "NVIDIA megakernel: CUDA checkpoint commit engaged (%d layers, batch %d)",
                st.shape[0],
                view.state_indices.numel(),
            )
            self.engaged = True
        return True


def bind(factory: HooksFactory | None, device: int, world: int) -> HooksFactory:
    # This model's head counts are pinned by its spec; build before graph capture.
    commit = _CheckpointCommit(device, max(1, 16 // world), 64 // world)

    def checkpoint_hooks(defaults: SpeculativeRuntimeHooks) -> SpeculativeRuntimeHooks:
        hooks = defaults if factory is None else factory(defaults)
        return SpeculativeRuntimeHooks(
            post_verify=hooks.post_verify, checkpoint_commit=commit
        )

    return checkpoint_hooks


def _strides(t: Optional[torch.Tensor]):
    return (0, 0) if t is None else (t.stride(0), t.stride(1))


def _params(
    *,
    checkpoint_state: torch.Tensor,
    d_cache: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    d_residual_cache: Optional[torch.Tensor],
    k_residual_cache: Optional[torch.Tensor],
    state_batch_indices: torch.Tensor,
    replay_indices: torch.Tensor,
    write_pos: torch.Tensor,
    cache_base: torch.Tensor,
    is_flush: torch.Tensor,
    accept_lens: torch.Tensor,
    mamba_track_indices: Optional[torch.Tensor] = None,
    mamba_steps_to_track: Optional[torch.Tensor] = None,
    null_block_id: int = -1,
) -> _Params:
    B = state_batch_indices.shape[0]
    resid = d_residual_cache is not None and k_residual_cache is not None
    assert write_pos.dtype == torch.int32 and cache_base.dtype == torch.int32
    assert is_flush.dtype == torch.int8
    for t in (
        state_batch_indices,
        replay_indices,
        accept_lens,
        mamba_track_indices,
        mamba_steps_to_track,
    ):
        assert t is None or (
            t.is_contiguous() and t.dtype in (torch.int32, torch.int64)
        )
    i64 = lambda t: int(t is not None and t.dtype == torch.int64)  # noqa: E731
    ptr = lambda t: 0 if t is None else t.data_ptr()  # noqa: E731
    return _Params(
        checkpoint_state.data_ptr(),
        *_strides(checkpoint_state),
        d_cache.data_ptr(),
        *_strides(d_cache),
        k_cache.data_ptr(),
        *_strides(k_cache),
        g_cache.data_ptr(),
        *_strides(g_cache),
        ptr(d_residual_cache) if resid else 0,
        *(_strides(d_residual_cache) if resid else (0, 0)),
        ptr(k_residual_cache) if resid else 0,
        *(_strides(k_residual_cache) if resid else (0, 0)),
        state_batch_indices.data_ptr(),
        replay_indices.data_ptr(),
        write_pos.data_ptr(),
        cache_base.data_ptr(),
        is_flush.data_ptr(),
        accept_lens.data_ptr(),
        ptr(mamba_track_indices),
        ptr(mamba_steps_to_track),
        i64(state_batch_indices),
        i64(replay_indices),
        i64(accept_lens),
        i64(mamba_track_indices),
        i64(mamba_steps_to_track),
        B,
        null_block_id,
    )


def _launch(fn, grid: int, ring: int, device: torch.device, args) -> None:
    from cuda.bindings import driver as cu

    cfg = cu.CUlaunchConfig()
    cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid, 1, 1
    cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = fn.block
    cfg.sharedMemBytes = _smem_bytes(ring)
    cfg.hStream = cu.CUstream(torch.cuda.current_stream(device).cuda_stream)
    cfg.attrs, cfg.numAttrs = [], 0
    addresses = (ctypes.c_void_p * len(args))(*[ctypes.addressof(a) for a in args])
    jit.check(
        cu.cuLaunchKernelEx(cfg, fn.function, ctypes.addressof(addresses), 0), "launch"
    )
