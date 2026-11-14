# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/layers/quantization/sm107_shared_expert_act.py (SILU_MUL_PTX,
# permute_gate_up, _kernel_module, run); its GEMM and parts of its epilogue come from
# SGLang's split-K epilogue GEMM (python/sglang/kernels/ops/gemm/
# dense_bf16_gemm_sm100_splitk_epilogue.py with sgl-project/sglang#50194), which it
# subclasses as the source does. Changed: the switch, the eligibility checks, the
# permuted-weight cache and the load-time check moved to runtime/smallm_gemm.py.
"""The shared expert's gate_up GEMM with act_and_mul in its epilogue: one kernel for
``silu(gate) * up`` of ``x @ gate_up.T`` at 1-16 rows (N 1280, K 2560).

The GEMM is the cluster split-K kernel at tactic (64, 8, 4, 5): four K slices of 640
whose FP32 partials the owner rank sums in rank order, cuBLAS's slicing at these rows.
Only the epilogue warp is overridden, after that reduction:

* One CTA owns the gate and up tiles of the same columns: the kernel reads a
  row-permuted copy of the weight (``permute_gate_up``) in which kernel-M tile t holds
  gate rows 32t..32t+31, then up rows 640+32t..640+32t+31. A row's dot product does
  not depend on where its row sits in the tile (same MMA instruction, same K order).
* The epilogue rounds the FP32 sum to BF16 as the stock store does, widens it back
  (exact), stages the tile in shared memory as the vendored "gate" epilogue does, and
  computes ``float(bf16(gate / (1 + expf(-gate)) * up))`` with the instructions nvcc
  emits for SGLang's act_and_mul kernel as built before SGLang #48942 under
  --use_fast_math: mul.ftz by -log2(e), ex2.approx.ftz, add.ftz, div.approx.ftz,
  mul.ftz, then cvt.rn.bf16 (``SILU_MUL_PTX``). The source checked every finite
  (gate, up) BF16 pair equal to that kernel's output. Since #48942 SGLang's kernel
  computes ``gate * rcp.approx.ftz(1 + ex2.approx.ftz(-gate * log2(e))) * up``, which
  differs from it on 87,313,176 of the 4,261,478,400 finite pairs.
"""

import torch

SGLANG_READS = {
    "kernels/ops/gemm/dense_bf16_gemm_sm100_splitk_epilogue.py": (
        "OWNER_RANK",
        "SplitKDenseGemmKernel",
        "SplitKTactic",
        "_from_dlpack_dynamic",
        "_make_compile_repr_tensors",
        "_mbarrier_wait_acquire_cluster",
        "_store_shared_remote_v4",
        "_tcgen05_fence",
        "_to_cute_swap",
    ),
}
# ActMulSplitKKernel.epilog_warp copies this body up to the reduction
# (sgl-project/sglang#50194's version)
SGLANG_MIRRORS = {
    "kernels/ops/gemm/dense_bf16_gemm_sm100_splitk_epilogue.py:SplitKDenseGemmKernel.epilog_warp": "d317579e743e7dd2",
}

# Reads and mirrors that need an SGLang change newer than the unit job's pin: the
# contract test skips each, naming the change, only where the installed SGLang lacks it.
SGLANG_REQUIRES = {
    "kernels/ops/gemm/dense_bf16_gemm_sm100_splitk_epilogue.py:_mbarrier_wait_acquire_cluster": "sglang#50194",
    "kernels/ops/gemm/dense_bf16_gemm_sm100_splitk_epilogue.py:_tcgen05_fence": "sglang#48616",
    "kernels/ops/gemm/dense_bf16_gemm_sm100_splitk_epilogue.py:SplitKDenseGemmKernel.epilog_warp": (
        "sglang#50194",
        ("e6f914864abef611", "070a3e085d4ddd5d"),
    ),
}

# (mma_m, mma_n, split_k, ab_stages): the source's gate_up route at 1-16 rows
TACTIC = (64, 8, 4, 5)
_TILE_M = 64
_HALF = (
    _TILE_M // 2
)  # gate rows per kernel-M tile; the other half are the up rows of the same columns
_BARRIER_ID = 7  # named barrier among the 4 epilogue warps (the vendored gate epilogue uses the same id)

# act_and_mul_kernel's silu(g) * u under --use_fast_math, one PTX instruction per C operation:
# expf(-g) -> ex2.approx.ftz(g * -log2(e)); 1 + e; g / d -> div.approx.ftz; * u.
SILU_MUL_PTX = (
    "{\n\t.reg .f32 ar_t, ar_e, ar_d, ar_s;\n\t"
    "mul.ftz.f32 ar_t, $1, 0fBFB8AA3B;\n\t"
    "ex2.approx.ftz.f32 ar_e, ar_t;\n\t"
    "add.ftz.f32 ar_d, ar_e, 0f3F800000;\n\t"
    "div.approx.ftz.f32 ar_s, $1, ar_d;"
    "\n\tmul.ftz.f32 $0, ar_s, $2;\n\t}"
)

_km = None  # the CuTe DSL side, built on first use
_compiled: dict = {}


def permute_gate_up(weight: torch.Tensor) -> torch.Tensor:
    """(2I, K) [gate; up] -> (2I, K) with 64-row tile t = gate rows 32t..32t+31, then up rows I+32t..I+32t+31."""
    n, k = weight.shape
    inter = n // 2
    if n % 2 or inter % _HALF:
        raise ValueError(f"gate_up rows {n} must be 2 x a multiple of {_HALF}")
    return (
        weight.view(2, inter // _HALF, _HALF, k).transpose(0, 1).contiguous().view(n, k)
    )


def _kernel_module():
    """Define the CuTe DSL kernel on first use (cutlass loads only when it is used)."""
    global _km
    if _km is not None:
        return _km
    from types import SimpleNamespace

    import cuda.bindings.driver as cuda_driver
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils.blackwell_helpers as sm100_utils
    import sglang.kernels.ops.gemm.dense_bf16_gemm_sm100_splitk_epilogue as epi
    from cutlass import Int32
    from cutlass._mlir.dialects import llvm
    from cutlass.cute import experimental as cute_ext
    from cutlass.cutlass_dsl import T, dsl_user_op

    def make_silu_mul(asm: str):
        @dsl_user_op
        def silu_mul(g, u, *, loc=None, ip=None):
            return cutlass.Float32(
                llvm.inline_asm(
                    T.f32(),
                    [g.ir_value(loc=loc, ip=ip), u.ir_value(loc=loc, ip=ip)],
                    asm,
                    "=f,f,f",
                    has_side_effects=False,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                )
            )

        return silu_mul

    class ActMulSplitKKernel(epi.SplitKDenseGemmKernel):
        """The vendored cluster split-K GEMM; the owner rank's epilogue writes silu(gate) * up, not gate_up."""

        def __init__(self, tactic, silu_mul_ptx: str):
            super().__init__(tactic=tactic, use_pdl=True, has_bias=False)
            if tactic.mma_m != _TILE_M:
                raise ValueError(f"act_and_mul epilogue needs mma_m={_TILE_M}")
            if (_HALF * tactic.mma_n) % self.epilog_threads:
                raise ValueError(
                    f"{_HALF} x {tactic.mma_n} outputs do not split over {self.epilog_threads} threads"
                )
            # kernel() allocates the (cta_m, cta_n) FP32 staging tile for mode "gate" only; epilog_warp is ours.
            self.epilogue_mode = "gate"
            self.silu_mul = make_silu_mul(silu_mul_ptx)

        @cute.experimental.jit
        def epilog_warp(
            self,
            bar_tma_epilog,
            bar_mma_epilog,
            bar_tmem_alloc,
            tmem_base_ptr,
            acc_layout: cutlass.Constexpr,
            gD_tile: cute.Tensor,
            gBias_tile: cute.Tensor,
            epi_tid: cutlass.Int32,
            c_dtype: cutlass.Constexpr,
            d_layout: cutlass.Constexpr,
            mailbox,
            bar_reduce,
            split_rank: cutlass.Int32,
            gate_tile,
            mX: cute.Tensor,
            mOut: cute.Tensor,
            bidx: cutlass.Int32,
            n_idx: cutlass.Int32,
        ):
            # Verbatim from the vendored epilog_warp (has_bias=False): TMEM -> RMEM, then the rank-ordered reduce.
            cute.arch.mbarrier_arrive(bar_tmem_alloc)
            cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
            epi._tcgen05_fence(before=False)

            acc_view = cute.make_tensor(
                cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr),
                acc_layout,
            )[((None, None), 0, 0, 0)]

            epi_tile = (self.cta_m, self.cta_n)
            tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(
                sm100_utils.get_tmem_load_op(
                    (self.cta_m, self.cta_n, self.cta_k),
                    d_layout,
                    c_dtype,
                    self.acc_dtype,
                    epi_tile,
                    False,
                ),
                acc_view,
            )
            gD_epi = cute.flat_divide(gD_tile, epi_tile)
            rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, gD_epi, epi_tid)
            rAcc = cute_ext.allocate(
                self.acc_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32
            )
            rD = cute_ext.allocate(
                c_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32
            )
            thr_t2r = tiled_copy_t2r.get_slice(epi_tid)

            cute.arch.mbarrier_wait(bar_mma_epilog, 0)
            epi._tcgen05_fence(before=False)
            cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
            cute.arch.fence_view_async_tmem_load()
            epi._tcgen05_fence(before=True)
            cute.arch.mbarrier_arrive(bar_tmem_alloc)

            if cutlass.const_expr(self.split_k > 1):
                values_per_thread = cutlass.const_expr(cute.size(rmem_layout))
                values_per_peer = cutlass.const_expr(
                    self.epilog_threads * values_per_thread
                )
                if split_rank != epi.OWNER_RANK:
                    for value_idx in cutlass.range_constexpr(0, values_per_thread, 4):
                        epi._store_shared_remote_v4(
                            rAcc[value_idx],
                            rAcc[value_idx + 1],
                            rAcc[value_idx + 2],
                            rAcc[value_idx + 3],
                            mailbox.iterator
                            + (split_rank - Int32(1)) * values_per_peer
                            + epi_tid * values_per_thread
                            + value_idx,
                            bar_reduce,
                            Int32(epi.OWNER_RANK),
                        )
                else:
                    if epi_tid == 0:
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            bar_reduce, self.expected_transaction_bytes
                        )
                    epi._mbarrier_wait_acquire_cluster(bar_reduce, Int32(0))
                    for peer in cutlass.range_constexpr(self.split_k - 1):
                        for value_idx in cutlass.range_constexpr(values_per_thread):
                            rAcc[value_idx] = (
                                rAcc[value_idx]
                                + mailbox[
                                    peer * values_per_peer
                                    + epi_tid * values_per_thread
                                    + value_idx
                                ]
                            )

            # The act_and_mul epilogue (owner rank).
            if split_rank == epi.OWNER_RANK:
                # The stock store's BF16 rounding of gate_up, widened back exactly (act_and_mul reads that BF16).
                rD.store(rAcc.load().to(c_dtype))
                rAcc.store(rD.load().to(self.acc_dtype))
                sStage_epi = cute.flat_divide(gate_tile, epi_tile)
                cute_ext.partition_and_copy(thr_t2r, rAcc, sStage_epi[None, None, 0, 0])
                cute.arch.barrier(
                    barrier_id=_BARRIER_ID, number_of_threads=self.epilog_threads
                )

                rows = cute.size(mOut, mode=[0])
                for it in cutlass.range_constexpr(
                    (_HALF * self.cta_n) // self.epilog_threads
                ):
                    elem = it * self.epilog_threads + epi_tid
                    j_local = elem % _HALF
                    m_local = elem // _HALF
                    m_global = n_idx * self.cta_n + m_local
                    if m_global < rows:
                        gate = gate_tile[j_local, m_local]
                        up = gate_tile[_HALF + j_local, m_local]
                        mOut[m_global, bidx * _HALF + j_local] = self.silu_mul(
                            gate, up
                        ).to(c_dtype)

    @cute.experimental.jit
    def bmm_actmul(
        gemm_op: cutlass.Constexpr,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        out: cute.Tensor,
        stream: cuda_driver.CUstream,
    ):
        # As the vendored _bmm_gate: C is only the grid's shape anchor here; out is the (rows, I) act output.
        c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
        gemm_op(
            cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
            cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
            c,
            cute.make_tensor(c.iterator, cute.select(c.layout, mode=[0, 1, 2])),
            c,
            out,
            stream,
        )

    _km = SimpleNamespace(
        epi=epi,
        cute_ext=cute_ext,
        stream=cuda_driver.CUstream,
        kernel=ActMulSplitKKernel,
        bmm=bmm_actmul,
    )
    return _km


def run(
    x: torch.Tensor,
    w_perm: torch.Tensor,
    out: torch.Tensor,
    anchor: torch.Tensor,
    *,
    silu_mul_ptx: str = SILU_MUL_PTX,
) -> torch.Tensor:
    """out[m, j] = bf16(silu(g) * u), g/u = bf16(x @ w_perm.T) at the gate/up rows of column j; anchor is (m, 2I).

    ``silu_mul_ptx`` exists for tests, which build a deliberately different epilogue as a control.
    """
    km = _kernel_module()
    m = x.shape[0]
    tensors = km.epi._to_cute_swap(x, w_perm.t(), anchor, None)
    key = (x.get_device(), tensors[4], silu_mul_ptx)
    compiled = _compiled.get(key)
    if compiled is None:
        with torch.cuda.device(x.device):
            kernel = km.kernel(km.epi.SplitKTactic(*TACTIC), silu_mul_ptx)
            rep = km.epi._make_compile_repr_tensors(x.dtype, False, *tensors[4])
            out_rep = km.epi._from_dlpack_dynamic(
                torch.empty((8, _TILE_M), dtype=x.dtype, device=x.device), 1
            )
            compiled = km.cute_ext.compile(
                km.bmm,
                kernel,
                *rep[:3],
                out_rep,
                km.stream(torch.cuda.current_stream().cuda_stream),
            )
        _compiled[key] = compiled
    assert (
        out.shape == (m, w_perm.shape[0] // 2)
        and out.stride(1) == 1
        and out.data_ptr() % 32 == 0
    )
    compiled(
        *tensors[:3],
        km.epi._from_dlpack_dynamic(out, 1),
        km.stream(torch.cuda.current_stream(x.device).cuda_stream),
    )
    return out
