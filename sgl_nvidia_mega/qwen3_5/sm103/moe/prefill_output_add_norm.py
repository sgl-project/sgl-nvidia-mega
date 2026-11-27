# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 by FlashInfer team.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# This file has been modified from its original version (the shared-expert gate add).
# Adapted from FlashInfer (Apache-2.0): flashinfer/norm/kernels/fused_add_rmsnorm.py
# (FusedAddRMSNormKernel.kernel).

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

SGLANG_MIRRORS = {
    "kernels/ops/elementwise/elementwise.py:_fused_gate_sigmoid_mul_add_kernel": "655b0ef65fe4c485",
    "kernels/ops/elementwise/elementwise.py:_launch_fused_gate_sigmoid_mul": "21b2f9d290468ef0",
}

FLASHINFER_SOURCES = {
    "norm/kernels/fused_add_rmsnorm.py": (
        "8519f660705e44da4feb5c8d0945fe8ecbc2d19938c6d7f4e98992680e0e0697",
    ),
    "norm/__init__.py": (
        "a9cfee06cd8e02cca5922ddd1fc8134329d8ea740adefa81ae472798d5412f87",
    ),
}


@dsl_user_op
def fma_rn_f32(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    """a * b + c rounded once (fma.rn.f32): the contraction of SGLang's ``gate * s + f``."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                Float32(c).ir_value(loc=loc, ip=ip),
            ],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


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


class GatedAddRMSNormKernel(FusedAddRMSNormKernel):
    def __init__(self, *args, with_fp8: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_fp8 = with_fp8

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mSh: cute.Tensor,
        mG: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        mQ8: cute.Tensor,
        mS8: cute.Tensor,
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

        self.kernel(
            mX, mSh, mG, mR, mW, mQ8, mS8, M, eps, enable_pdl, tv_layout, tiler_mn
        ).launch(
            grid=[cute.ceil_div(M, self.rows_per_block), 1, 1],
            block=[self.num_threads, 1, 1],
            smem=self._smem_size_in_bytes(),
            stream=stream,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mSh: cute.Tensor,
        mG: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        mQ8: cute.Tensor,
        mS8: cute.Tensor,
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
        weight_bias = self.weight_bias
        copy_bits = self.copy_bits
        threads_per_row = tv_layout.shape[0][0]
        rows_per_block = tiler_mn[0]
        warps_per_row = max(threads_per_row // 32, 1)
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

        reduction_buffer = smem.allocate_tensor(
            Float32,
            cute.make_layout((rows_per_block, warps_per_row)),
            byte_alignment=4,
        )
        mbar_ptr = None

        idX = cute.make_identity_tensor(mX.shape)

        gX = cute.local_tile(mX, tiler_mn, (bidx, cluster_y))
        gS = cute.local_tile(mSh, tiler_mn, (bidx, cluster_y))
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

        tSgS = thr_copy_W.partition_S(gS)
        tSrS = cute.make_fragment_like(tSgS)
        tXrS = thr_copy_X.retile(tSrS)

        tXgO = thr_copy_O.partition_D(gX)
        tRgO = thr_copy_O.partition_D(gR)
        tXrO = cute.make_fragment_like(tXgO)

        tXpX = predicate_k(tXcX, limit=H)
        tWpW = predicate_k(thr_copy_W.partition_S(cX), limit=H)
        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

        tSrS.store(cute.zeros_like(tSrS, dtype=mSh.element_type))
        if cutlass.const_expr(self.use_async_copy):
            if row_in_bounds:
                cute.copy(copy_atom_async, tXgX, tXsX, pred=tXpX)
                cute.copy(copy_atom_async, tRgR, tRsR, pred=tXpX)
            cute.arch.cp_async_commit_group()

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)
            if row_in_bounds:
                cute.copy(copy_atom_sync, tSgS, tSrS, pred=tWpW)

            cute.arch.cp_async_wait_group(0)

            cute.autovec_copy(tXsX, tXrX)
            cute.autovec_copy(tRsR, tRrR)
        else:
            tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
            tRrR.store(cute.zeros_like(tRrR, dtype=mR.element_type))
            if row_in_bounds:
                cute.copy(copy_atom_sync, tXgX, tXrX, pred=tXpX)
                cute.copy(copy_atom_sync, tRgR, tRrR, pred=tXpX)
                cute.copy(copy_atom_sync, tSgS, tSrS, pred=tWpW)

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

        g = mG[row_coord[0]]
        tF = cute.make_rmem_tensor(tXrX.shape, Float32)
        tF.store(tXrX.load().to(Float32))
        tSf = cute.make_rmem_tensor(tXrX.shape, Float32)
        tSf.store(tXrS.load().to(Float32))
        for i in cutlass.range_constexpr(cute.size(tF)):
            tF[i] = fma_rn_f32(g, tSf[i], tF[i])
        tXrX.store(tF.load().to(mX.element_type))

        x_in = tXrX.load().to(Float32)
        r_in = tRrR.load().to(Float32)
        h = x_in + r_in

        tXrO.store(h.to(mR.element_type))
        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tRgO, pred=tXpX)

        h_sq = h * h
        sum_sq = row_reduce_sum_multirow(
            h_sq, threads_per_row, reduction_buffer, mbar_ptr, 1
        )

        mean_sq = sum_sq / Float32(H)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        cute.arch.barrier()

        w = tXrW.load().to(Float32)
        y = h * rstd * (w + Float32(weight_bias))

        tXrO.store(y.to(mX.element_type))

        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tXgO, pred=tXpX)

        if cutlass.const_expr(self.with_fp8):
            inv_scale = div_full_rcp_f32(mS8[0])
            vec_size = self.vec_size
            tQf = cute.make_rmem_tensor(tXrX.shape, Float32)
            tQf.store(tXrO.load().to(Float32) * inv_scale)
            lane_in_row = tidx % threads_per_row
            row_in_block = tidx // threads_per_row
            actual_row = Int64(bidx) * rows_per_block + row_in_block
            col_offset = lane_in_row * vec_size
            for v in cutlass.range_constexpr(self.num_vec_blocks):
                abs_col = col_offset + v * threads_per_row * vec_size
                if abs_col + 8 <= H and actual_row < M:
                    base = v * 8
                    cvt_and_store_8xf32_to_fp8_hw(
                        tQf[base],
                        tQf[base + 1],
                        tQf[base + 2],
                        tQf[base + 3],
                        tQf[base + 4],
                        tQf[base + 5],
                        tQf[base + 6],
                        tQf[base + 7],
                        get_ptr_as_int64(
                            mQ8,
                            cute.crd2idx(
                                (Int64(actual_row), Int32(abs_col)), mQ8.layout
                            ),
                        ),
                        mQ8.element_type,
                    )

        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


@functools.cache
def _kernel_obj(H: int, weight_bias: float, sm_version: int, with_fp8: bool = False):
    return GatedAddRMSNormKernel(
        get_cutlass_dtype("bfloat16"),
        H,
        weight_bias,
        sm_version=sm_version,
        with_fp8=with_fp8,
    )


@functools.cache
def config_supported(H: int, sm_version: int) -> bool:
    if H % 8 != 0:
        return False
    k = _kernel_obj(H, 1.0, sm_version)
    return k.cluster_n == 1 and k.vec_size == 8 and ROW_PAD % k.rows_per_block == 0


@functools.cache
def _compiled(
    H: int, weight_bias: float, enable_pdl: bool, sm_version: int, with_fp8: bool
):
    dtype = get_cutlass_dtype("bfloat16")
    kernel_obj = _kernel_obj(H, weight_bias, sm_version, with_fp8)
    sym_m = cute.sym_int(64)
    align = math.gcd(128, H * 2)
    x_fake = cute.runtime.make_fake_compact_tensor(
        dtype, (sym_m, H), stride_order=(1, 0), assumed_align=align
    )
    s_fake = cute.runtime.make_fake_compact_tensor(
        dtype, (sym_m, H), stride_order=(1, 0), assumed_align=align
    )
    g_fake = cute.runtime.make_fake_compact_tensor(Float32, (sym_m,), assumed_align=4)
    r_fake = cute.runtime.make_fake_compact_tensor(
        dtype, (sym_m, H), stride_order=(1, 0), assumed_align=align
    )
    w_fake = cute.runtime.make_fake_compact_tensor(dtype, (H,), assumed_align=16)
    q8_fake = cute.runtime.make_fake_compact_tensor(
        get_cutlass_dtype("float8_e4m3fn"),
        (sym_m if with_fp8 else cute.sym_int(), H),
        stride_order=(1, 0),
        assumed_align=math.gcd(128, H),
    )
    s8_fake = cute.runtime.make_fake_compact_tensor(Float32, (1,), assumed_align=4)
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel_obj,
        x_fake,
        s_fake,
        g_fake,
        r_fake,
        w_fake,
        q8_fake,
        s8_fake,
        Int64(1),
        Float32(1e-6),
        enable_pdl,
        stream_fake,
        options="--enable-tvm-ffi",
    )


def precompile(device, H: int) -> None:
    from flashinfer.utils import device_support_pdl

    _compiled(H, 1.0, bool(device_support_pdl(device)), get_sm_version(device), False)


ROW_PAD = 16


def gate_storage_ok(gate: torch.Tensor) -> bool:
    m = gate.shape[0]
    need = (gate.storage_offset() + (m + ROW_PAD - 1) // ROW_PAD * ROW_PAD) * 4
    return gate.untyped_storage().nbytes() >= need


def supported(
    x: torch.Tensor, shared: torch.Tensor, gate: torch.Tensor, residual: torch.Tensor
) -> bool:
    return (
        x.is_cuda
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and shared.dtype == torch.bfloat16
        and residual.dtype == torch.bfloat16
        and gate.dtype == torch.float32
        and x.shape == shared.shape == residual.shape
        and gate.shape == (x.shape[0],)
        and shared.device == x.device == residual.device == gate.device
        and x.is_contiguous()
        and shared.is_contiguous()
        and residual.is_contiguous()
        and gate.is_contiguous()
        and gate_storage_ok(gate)
        and 0 < x.shape[0] * x.shape[1] <= 2**31 - 1
        and config_supported(x.shape[1], get_sm_version(x.device))
    )


def gated_add_rmsnorm(
    x, shared, gate, residual, weight, eps, fp8_scale=None, enable_pdl=None
):
    from flashinfer.utils import device_support_pdl

    if not supported(x, shared, gate, residual):
        raise ValueError("gated_add_rmsnorm: unsupported input")
    if enable_pdl is None or enable_pdl:
        enable_pdl = device_support_pdl(x.device)
    M, H = x.shape
    with_fp8 = fp8_scale is not None
    if with_fp8:
        s8 = fp8_scale.reshape(1)
        if s8.dtype != torch.float32:
            raise ValueError("gated_add_rmsnorm: fp8_scale must be float32")
        q8 = torch.empty((M, H), dtype=torch.float8_e4m3fn, device=x.device)
    else:
        s8, q8 = (
            gate[:1],
            torch.empty((0, H), dtype=torch.float8_e4m3fn, device=x.device),
        )
    kernel = _compiled(H, 1.0, bool(enable_pdl), get_sm_version(x.device), with_fp8)
    kernel(x, shared, gate, residual, weight, q8, s8, M, eps)
    return q8 if with_fp8 else None
