# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 by FlashInfer team.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# This file has been modified from its original version (a static FP8 store).
# Adapted from FlashInfer (Apache-2.0): flashinfer/norm/kernels/fused_add_rmsnorm.py
# (FusedAddRMSNormKernel, _get_compiled_fused_add_rmsnorm_kernel).

import functools
import math

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from flashinfer.norm.kernels.fused_add_rmsnorm import FusedAddRMSNormKernel
from flashinfer.norm.kernels.rmsnorm import RMSNormKernel
from flashinfer.norm.utils import (
    cvt_and_store_8xf32_to_fp8_hw,
    get_cutlass_dtype,
    get_ptr_as_int64,
    get_sm_version,
    predicate_k,
    row_reduce_sum_multirow,
)

FLASHINFER_SOURCES = {
    "norm/kernels/fused_add_rmsnorm.py": (
        "8519f660705e44da4feb5c8d0945fe8ecbc2d19938c6d7f4e98992680e0e0697",
    ),
    "norm/kernels/rmsnorm.py": (
        "837a4c785cda9cedf8ee6179a0f98f91af291847fff605188db0c81e9b392d7f",
    ),
    "norm/utils.py": (
        "39b30af1c8160fcb01f855fd9c7ecde5076231764b87bb100e2011781a9152cb",
    ),
    "norm/__init__.py": (
        "a9cfee06cd8e02cca5922ddd1fc8134329d8ea740adefa81ae472798d5412f87",
    ),
}


@dsl_user_op
def div_full_rcp_f32(a: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "div.full.f32 $0, 0f3F800000, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


class FusedAddRMSNormDualKernel(FusedAddRMSNormKernel):
    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        mQ: cute.Tensor,
        mS: cute.Tensor,
        M: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        tv_shape, tv_stride = RMSNormKernel._make_tv_layout(
            self.threads_per_row,
            self.rows_per_block,
            self.vec_size,
            self.num_vec_blocks,
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (self.rows_per_block, self.cols_per_tile)

        cluster_n = self.cluster_n

        self.kernel(mX, mR, mW, mQ, mS, M, eps, enable_pdl, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(M, self.rows_per_block), cluster_n, 1],
            block=[self.num_threads, 1, 1],
            cluster=[1, cluster_n, 1] if cutlass.const_expr(cluster_n > 1) else None,
            smem=self._smem_size_in_bytes(),
            stream=stream,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        mQ: cute.Tensor,
        mS: cute.Tensor,
        M: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        if enable_pdl:
            cute.arch.griddepcontrol_wait()

        H = self.H
        cluster_n = self.cluster_n
        weight_bias = self.weight_bias
        copy_bits = self.copy_bits
        threads_per_row = tv_layout.shape[0][0]
        rows_per_block = tiler_mn[0]
        warps_per_row = max(threads_per_row // 32, 1)

        if cutlass.const_expr(cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = cutlass.const_expr(0)

        smem = cutlass.utils.SmemAllocator()

        if cutlass.const_expr(self.use_async_copy):
            sX = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )
            sR = smem.allocate_tensor(
                mR.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )

        if cutlass.const_expr(cluster_n == 1):
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, warps_per_row)),
                byte_alignment=4,
            )
            mbar_ptr = None
        else:
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, (warps_per_row, cluster_n))),
                byte_alignment=4,
            )
            mbar_ptr = smem.allocate_array(cutlass.Int64, num_elems=1)

        if cutlass.const_expr(cluster_n > 1):
            if tidx == 0:
                cute.arch.mbarrier_init(mbar_ptr, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

        idX = cute.make_identity_tensor(mX.shape)

        gX = cute.local_tile(mX, tiler_mn, (bidx, cluster_y))
        gR = cute.local_tile(mR, tiler_mn, (bidx, cluster_y))
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

        mW_expanded_layout = cute.prepend(
            mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
        )
        mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
        gW = cute.local_tile(mW_2d, tiler_mn, (0, cluster_y))

        copy_atom_sync = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=copy_bits,
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=copy_bits,
        )

        if cutlass.const_expr(self.use_async_copy):
            copy_atom_async = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mX.element_type,
                num_bits_per_copy=copy_bits,
            )
            tiled_copy_load = cute.make_tiled_copy(copy_atom_async, tv_layout, tiler_mn)
        else:
            tiled_copy_load = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)

        tiled_copy_W = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)
        tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)

        thr_copy_X = tiled_copy_load.get_slice(tidx)
        thr_copy_W = tiled_copy_W.get_slice(tidx)
        thr_copy_O = tiled_copy_store.get_slice(tidx)

        tXgX = thr_copy_X.partition_S(gX)
        tXcX = thr_copy_X.partition_S(cX)
        tXrX = cute.make_fragment_like(tXgX)

        tRgR = thr_copy_X.partition_S(gR)
        tRrR = cute.make_fragment_like(tRgR)

        if cutlass.const_expr(self.use_async_copy):
            tXsX = thr_copy_X.partition_D(sX)
            tRsR = thr_copy_X.partition_D(sR)

        tWgW = thr_copy_W.partition_S(gW)
        tWrW = cute.make_fragment_like(tWgW)
        tXrW = thr_copy_X.retile(tWrW)

        tXgO = thr_copy_O.partition_D(gX)
        tRgO = thr_copy_O.partition_D(gR)
        tXrO = cute.make_fragment_like(tXgO)

        tXpX = predicate_k(tXcX, limit=H)
        tWpW = predicate_k(thr_copy_W.partition_S(cX), limit=H)
        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

        if cutlass.const_expr(self.use_async_copy):
            if row_in_bounds:
                cute.copy(copy_atom_async, tXgX, tXsX, pred=tXpX)
                cute.copy(copy_atom_async, tRgR, tRsR, pred=tXpX)
            cute.arch.cp_async_commit_group()

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

            cute.arch.cp_async_wait_group(0)

            cute.autovec_copy(tXsX, tXrX)
            cute.autovec_copy(tRsR, tRrR)
        else:
            tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
            tRrR.store(cute.zeros_like(tRrR, dtype=mR.element_type))
            if row_in_bounds:
                cute.copy(copy_atom_sync, tXgX, tXrX, pred=tXpX)
                cute.copy(copy_atom_sync, tRgR, tRrR, pred=tXpX)

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

        x_in = tXrX.load().to(Float32)
        r_in = tRrR.load().to(Float32)
        h = x_in + r_in

        tXrO.store(h.to(mR.element_type))
        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tRgO, pred=tXpX)

        h_sq = h * h
        sum_sq = row_reduce_sum_multirow(
            h_sq, threads_per_row, reduction_buffer, mbar_ptr, cluster_n
        )

        mean_sq = sum_sq / Float32(H)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        if cutlass.const_expr(cluster_n > 1):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier()

        w = tXrW.load().to(Float32)
        y = h * rstd * (w + Float32(weight_bias))

        tXrO.store(y.to(mX.element_type))

        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tXgO, pred=tXpX)

        inv_scale = div_full_rcp_f32(mS[0])
        vec_size = self.vec_size
        cols_per_tile = self.cols_per_tile
        tQrQ_f32 = cute.make_rmem_tensor(tXrX.shape, Float32)
        tQrQ_f32.store(tXrO.load().to(Float32) * inv_scale)

        lane_in_row = tidx % threads_per_row
        row_in_block = tidx // threads_per_row
        actual_row = Int64(bidx) * rows_per_block + row_in_block
        col_offset = lane_in_row * vec_size
        for v in cutlass.range_constexpr(self.num_vec_blocks):
            local_col = col_offset + v * threads_per_row * vec_size
            abs_col = cluster_y * cols_per_tile + local_col
            if abs_col + 8 <= H and actual_row < M:
                base = v * 8
                cvt_and_store_8xf32_to_fp8_hw(
                    tQrQ_f32[base],
                    tQrQ_f32[base + 1],
                    tQrQ_f32[base + 2],
                    tQrQ_f32[base + 3],
                    tQrQ_f32[base + 4],
                    tQrQ_f32[base + 5],
                    tQrQ_f32[base + 6],
                    tQrQ_f32[base + 7],
                    get_ptr_as_int64(
                        mQ,
                        cute.crd2idx((Int64(actual_row), Int32(abs_col)), mQ.layout),
                    ),
                    mQ.element_type,
                )

        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


@functools.cache
def _compiled(H: int, weight_bias: float, enable_pdl: bool, sm_version: int):
    dtype = get_cutlass_dtype("bfloat16")
    kernel_obj = FusedAddRMSNormDualKernel(dtype, H, weight_bias, sm_version=sm_version)
    if kernel_obj.vec_size != 8:
        raise ValueError(
            f"gemma_norm_fp8: vec_size {kernel_obj.vec_size} != 8 at H={H}"
        )
    sym_m = cute.sym_int(64)
    align = math.gcd(128, H * (dtype.width // 8))
    x_fake = cute.runtime.make_fake_compact_tensor(
        dtype, (sym_m, H), stride_order=(1, 0), assumed_align=align
    )
    r_fake = cute.runtime.make_fake_compact_tensor(
        dtype, (sym_m, H), stride_order=(1, 0), assumed_align=align
    )
    w_fake = cute.runtime.make_fake_compact_tensor(dtype, (H,), assumed_align=16)
    q_fake = cute.runtime.make_fake_compact_tensor(
        get_cutlass_dtype("float8_e4m3fn"),
        (sym_m, H),
        stride_order=(1, 0),
        assumed_align=math.gcd(128, H),
    )
    s_fake = cute.runtime.make_fake_compact_tensor(Float32, (1,), assumed_align=4)
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel_obj,
        x_fake,
        r_fake,
        w_fake,
        q_fake,
        s_fake,
        Int64(1),
        Float32(1e-6),
        enable_pdl,
        stream_fake,
        options="--enable-tvm-ffi",
    )


def cute_route(device) -> bool:
    from flashinfer.norm import _use_cuda_norm

    return not _use_cuda_norm(device)


def supported(x: torch.Tensor, residual: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and residual.is_cuda
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and residual.dtype == torch.bfloat16
        and x.shape == residual.shape
        and x.is_contiguous()
        and residual.is_contiguous()
        and x.shape[1] % 8 == 0
        and 0 < x.shape[0] * x.shape[1] <= 2**31 - 1
    )


def build(device_index: int, hidden: int) -> None:
    """Compile the kernel for ``hidden`` at load, outside any capture."""
    from flashinfer.utils import device_support_pdl

    device = torch.device("cuda", device_index)
    _compiled(hidden, 1.0, bool(device_support_pdl(device)), get_sm_version(device))


def gemma_fused_add_rmsnorm_fp8(x, residual, weight, eps, scale):
    from flashinfer.utils import device_support_pdl

    if not supported(x, residual):
        raise ValueError("gemma_fused_add_rmsnorm_fp8: unsupported input")
    enable_pdl = device_support_pdl(x.device)
    q = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    kernel = _compiled(x.shape[1], 1.0, bool(enable_pdl), get_sm_version(x.device))
    kernel(x, residual, weight, q, scale.reshape(1), x.shape[0], eps)
    return q
