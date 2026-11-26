# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/kernels/ops/elementwise/elementwise.py (_fused_gate_sigmoid_mul_add_kernel), and the
# source's _fused_finalize_gate_sigmoid_mul_add_chunked_kernel there.
"""The MoE finalize of slot ``moe_finalize``: one row a program writes
``sigmoid(x . g) * shared + sum_k w_k * expert_k`` from the un-finalized expert rows
(``gemm2_out`` through its permutation), the expert sum in a fixed order with fused
multiply-adds and rounded to BF16 before the gate, as FlashInfer's finalize then
``fused_gate_sigmoid_mul_add`` would."""

import triton
import triton.language as tl


@triton.jit
def _fused_finalize_gate_sigmoid_mul_add_kernel(
    hidden_states_ptr,
    gate_weight_ptr,
    shared_output_ptr,
    output_ptr,
    gemm2_out_ptr,
    permuted_idx_ptr,
    expert_weights_ptr,
    gemm2_stride,
    hidden_dim: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_PDL: tl.constexpr = False,
):

    pid = tl.program_id(axis=0).to(tl.int64)
    row_offset = pid * hidden_dim

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim

    w = tl.load(gate_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    h = tl.load(hidden_states_ptr + row_offset + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    s = tl.load(shared_output_ptr + row_offset + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for k in tl.static_range(TOP_K):
        p = tl.load(permuted_idx_ptr + pid * TOP_K + k)
        scale = tl.load(expert_weights_ptr + pid * TOP_K + k).to(tl.float32)
        valid = p >= 0
        x = tl.load(
            gemm2_out_ptr + p.to(tl.int64) * gemm2_stride + offsets,
            mask=mask & valid,
            other=0.0,
        ).to(tl.float32)

        fma = tl.inline_asm_elementwise(
            "fma.rn.ftz.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            [scale + tl.zeros_like(x), x, acc],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        acc = tl.where(valid, fma, acc)

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    f = acc.to(tl.bfloat16).to(tl.float32)
    gate_val = tl.sigmoid(tl.sum(h * w, axis=0))
    result = gate_val * s
    result += f
    tl.store(output_ptr + row_offset + offsets, result, mask=mask)


@triton.jit
def _fused_finalize_gate_sigmoid_mul_add_chunked_kernel(
    hidden_states_ptr,  # [num_tokens, hidden_dim]
    gate_weight_ptr,  # [hidden_dim]
    shared_output_ptr,  # [num_tokens, hidden_dim]
    output_ptr,  # [num_tokens, hidden_dim]
    gemm2_out_ptr,  # [num_permuted_rows, gemm2_stride] BF16
    permuted_idx_ptr,  # [num_tokens * TOP_K] int32, -1 = not local
    expert_weights_ptr,  # [num_tokens, TOP_K]
    gemm2_stride,
    hidden_dim: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CHUNK: tl.constexpr,
    USE_PDL: tl.constexpr = False,
):
    # _fused_finalize_gate_sigmoid_mul_add_kernel on a (token, column chunk) grid.
    # Each program recomputes the row's gate with the full-row load and tl.sum at the
    # same BLOCK_SIZE and num_warps (one reduction tree), then writes its CHUNK
    # columns with the same per-element FMA chain, rounding and mul-add.
    pid = tl.program_id(axis=0).to(tl.int64)
    cid = tl.program_id(axis=1)
    row_offset = pid * hidden_dim

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim
    cols = cid * CHUNK + tl.arange(0, CHUNK)
    cmask = cols < hidden_dim

    w = tl.load(gate_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        # Early trigger: a PDL successor must not write global memory before its
        # griddepcontrol.wait, which also orders its reads after this grid ends.
        tl.extra.cuda.gdc_launch_dependents()

    h = tl.load(hidden_states_ptr + row_offset + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    s = tl.load(shared_output_ptr + row_offset + cols, mask=cmask, other=0.0).to(
        tl.float32
    )
    acc = tl.zeros([CHUNK], dtype=tl.float32)
    for k in tl.static_range(TOP_K):
        p = tl.load(permuted_idx_ptr + pid * TOP_K + k)
        scale = tl.load(expert_weights_ptr + pid * TOP_K + k).to(tl.float32)
        valid = p >= 0
        x = tl.load(
            gemm2_out_ptr + p.to(tl.int64) * gemm2_stride + cols,
            mask=cmask & valid,
            other=0.0,
        ).to(tl.float32)
        fma = tl.inline_asm_elementwise(
            "fma.rn.ftz.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            [scale + tl.zeros_like(x), x, acc],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        acc = tl.where(valid, fma, acc)

    f = acc.to(tl.bfloat16).to(tl.float32)
    gate_val = tl.sigmoid(tl.sum(h * w, axis=0))
    result = gate_val * s
    result += f
    tl.store(output_ptr + row_offset + cols, result, mask=cmask)
