# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import ctypes
from functools import cache
from pathlib import Path

import torch
import triton
import triton.language as tl

from sgl_nvidia_mega.common import jit

SGLANG_MIRRORS = {
    "kernels/ops/mamba/causal_conv1d_triton.py:_causal_conv1d_fwd_kernel": (
        "a33cda998f11221e",
        "40ebba3bbe758aad",
    ),
    "srt/layers/attention/mamba/causal_conv1d.py:causal_conv1d_fn": "b8f0d3930a23f2b4",
    "kernels/ops/attention/fla/l2norm.py:l2norm_fwd_kernel": "e9cd6f991e96ba96",
    "kernels/ops/attention/fla/l2norm.py:l2norm_fwd": "d0895bbd4c6dd77d",
    "kernels/ops/attention/fla/l2norm.py:gdn_prefill_qkv_prepare_kernel": "4eb758f28fc8c319",
    "kernels/ops/attention/fla/l2norm.py:gdn_prefill_qkv_prepare_fwd": "152dd1336a614cb0",
    "kernels/ops/attention/linear/fused_gdn_gating.py:_gdn_gating_values": "545795b09ac68d3e",
    "kernels/ops/attention/linear/fused_gdn_gating.py:fused_gdn_gating_kernel": (
        "3d55a03cb36ecfa3",
        "8deccec111ff2cc7",
    ),
    "kernels/ops/attention/linear/fused_gdn_gating.py:fused_gdn_gating": (
        "23465554d0642aa7",
        "085be51c1a00596a",
    ),
}

SOURCE = Path(__file__).parent / "src" / "prefill_front.cu"
THREADS, VEC, CHUNK, HEAD_DIM, WIDTH = 128, 8, 64, 128, 4
UNROLL, MIN_BLOCKS = 2, 8
PAD_SLOT_ID = -1


class Params(ctypes.Structure):
    """gdn_prefill_front::Params."""

    _fields_ = [
        (name, ctypes.c_void_p)
        for name in ("x", "w", "bias", "cs", "qsl", "cidx", "hinit", "q", "k", "v")
    ] + [
        ("sx", ctypes.c_int64),
        ("cs_b", ctypes.c_int64),
        ("cs_c", ctypes.c_int64),
        ("cs_l", ctypes.c_int64),
        ("num_tokens", ctypes.c_int32),
        ("num_seqs", ctypes.c_int32),
        ("pad_slot", ctypes.c_int32),
        ("eps", ctypes.c_float),
    ]


@cache
def _kernel(
    device_index: int, nk: int, nv: int, has_bias: bool, silu: bool
) -> jit.Kernel:
    major, minor = torch.cuda.get_device_capability(device_index)
    defines = (
        f"-DGDN_NK={nk}",
        f"-DGDN_NV={nv}",
        f"-DGDN_D={HEAD_DIM}",
        f"-DGDN_CHUNK={CHUNK}",
        f"-DGDN_HAS_BIAS={int(has_bias)}",
        f"-DGDN_SILU={int(silu)}",
        f"-DGDN_MIN_BLOCKS={MIN_BLOCKS}",
        f"-DGDN_UNROLL={UNROLL}",
    )
    cubin = jit.compile_source(
        SOURCE.read_text(),
        "q35_gdn_prefill_front",
        f"sm_{major}{minor}a",
        ("-std=c++17", "-default-device", *defines),
    )
    return jit.load(cubin, "q35_gdn_prefill_front", (THREADS, 1, 1), 0)


@triton.jit
def _state_value(
    x_ptr, base, cols, cmask, start, end, hinit, w, stride_x_tok, stride_cs_l
):
    pos = end - 3 + w
    from_x = pos >= start
    xv = tl.load(
        x_ptr + pos.to(tl.int64) * stride_x_tok + cols,
        mask=cmask & from_x,
        other=0.0,
    )
    sv = tl.load(
        base + (pos - start + 3) * stride_cs_l,
        mask=cmask & (from_x == 0) & hinit,
        other=0.0,
    )
    return tl.where(from_x, xv, sv)


@triton.jit
def _conv_state_update_kernel(
    x_ptr,
    cs_ptr,
    qsl_ptr,
    cidx_ptr,
    hinit_ptr,
    pad_slot_id,
    conv_dim,
    stride_x_tok,
    stride_cs_b,
    stride_cs_c,
    stride_cs_l,
    BLOCK_C: tl.constexpr,
):
    s = tl.program_id(0)
    slot = tl.load(cidx_ptr + s)
    start = tl.load(qsl_ptr + s)
    end = tl.load(qsl_ptr + s + 1)
    if (slot != pad_slot_id) & (end > start):
        hinit = tl.load(hinit_ptr + s) != 0
        cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        cmask = cols < conv_dim
        base = cs_ptr + slot.to(tl.int64) * stride_cs_b + cols * stride_cs_c
        v0 = _state_value(
            x_ptr, base, cols, cmask, start, end, hinit, 0, stride_x_tok, stride_cs_l
        )
        v1 = _state_value(
            x_ptr, base, cols, cmask, start, end, hinit, 1, stride_x_tok, stride_cs_l
        )
        v2 = _state_value(
            x_ptr, base, cols, cmask, start, end, hinit, 2, stride_x_tok, stride_cs_l
        )
        tl.store(base, v0, mask=cmask)
        tl.store(base + stride_cs_l, v1, mask=cmask)
        tl.store(base + 2 * stride_cs_l, v2, mask=cmask)


def _activation(activation):
    if isinstance(activation, bool):
        return "silu" if activation else None
    return activation


def supported(
    *,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_width: int,
    activation,
) -> bool:
    activation = _activation(activation)
    conv_dim = (2 * num_k_heads + num_v_heads) * HEAD_DIM
    return (
        head_k_dim == head_v_dim == HEAD_DIM
        and conv_width == WIDTH
        and activation in (None, "silu", "swish")
        and (2 * num_k_heads * HEAD_DIM) % (THREADS * VEC) == 0
        and conv_dim % (THREADS * VEC) == 0
    )


def eligible(
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
) -> str:
    """Run-time layout / dtype conditions; the reason it cannot run, or "" when it can."""
    if (
        not mixed_qkv.is_cuda
        or mixed_qkv.dtype != torch.bfloat16
        or mixed_qkv.dim() != 2
    ):
        return "mixed_qkv dtype / rank"
    if (
        mixed_qkv.stride(1) != 1
        or mixed_qkv.stride(0) % VEC
        or mixed_qkv.data_ptr() % 16
    ):
        return "mixed_qkv layout"
    w = conv_weight.reshape(conv_weight.shape[0], -1)
    if (
        w.dtype != torch.bfloat16
        or w.shape[1] != WIDTH
        or not w.is_contiguous()
        or w.data_ptr() % 16
    ):
        return "conv weight layout"
    if conv_bias is not None and (
        conv_bias.dtype != torch.bfloat16 or not conv_bias.is_contiguous()
    ):
        return "conv bias"
    if (
        conv_states.dtype != torch.bfloat16
        or conv_states.dim() != 3
        or conv_states.shape[1] != mixed_qkv.shape[1]
        or conv_states.shape[2] < WIDTH - 1
    ):
        return "conv state dtype / shape"
    if query_start_loc.dtype != torch.int32 or cache_indices.dtype != torch.int32:
        return "index dtype"
    if not query_start_loc.is_contiguous() or not cache_indices.is_contiguous():
        return "index layout"
    return ""


def gdn_prefill_front(
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    *,
    num_k_heads: int,
    num_v_heads: int,
    activation="silu",
    eps: float = 1e-6,
    pad_slot_id: int = PAD_SLOT_ID,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    activation = _activation(activation)
    T, conv_dim = mixed_qkv.shape
    nk, nv, d = num_k_heads, num_v_heads, HEAD_DIM
    dev = mixed_qkv.device
    q = torch.empty((T, nk, d), dtype=torch.bfloat16, device=dev)
    k = torch.empty((T, nk, d), dtype=torch.bfloat16, device=dev)
    v = torch.empty((T, nv, d), dtype=torch.bfloat16, device=dev)
    num_seqs = query_start_loc.shape[0] - 1
    if num_seqs <= 0:
        return q, k, v
    if has_initial_state.dtype != torch.bool:
        has_initial_state = has_initial_state != 0
    w = conv_weight.reshape(conv_dim, WIDTH)
    stream = torch.cuda.current_stream(dev).cuda_stream
    if T > 0:
        kernel = _kernel(
            dev.index, nk, nv, conv_bias is not None, activation is not None
        )
        params = Params(
            x=mixed_qkv.data_ptr(),
            w=w.data_ptr(),
            bias=conv_bias.data_ptr() if conv_bias is not None else 0,
            cs=conv_states.data_ptr(),
            qsl=query_start_loc.data_ptr(),
            cidx=cache_indices.data_ptr(),
            hinit=has_initial_state.data_ptr(),
            q=q.data_ptr(),
            k=k.data_ptr(),
            v=v.data_ptr(),
            sx=mixed_qkv.stride(0),
            cs_b=conv_states.stride(0),
            cs_c=conv_states.stride(1),
            cs_l=conv_states.stride(2),
            num_tokens=T,
            num_seqs=num_seqs,
            pad_slot=pad_slot_id,
            eps=eps,
        )
        grid = (triton.cdiv(T, CHUNK), conv_dim // (THREADS * VEC), 1)
        jit.launch(kernel, grid, [params], stream)
    block_c = 1024
    _conv_state_update_kernel[(num_seqs, triton.cdiv(conv_dim, block_c))](
        mixed_qkv,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        pad_slot_id,
        conv_dim,
        mixed_qkv.stride(0),
        conv_states.stride(0),
        conv_states.stride(1),
        conv_states.stride(2),
        BLOCK_C=block_c,
        num_warps=4,
    )
    return q, k, v


@triton.jit
def _gates_kernel(
    g_ptr,
    beta_ptr,
    A_log_ptr,
    a_ptr,
    b_ptr,
    dt_bias_ptr,
    num_tokens,
    stride_a,
    stride_b,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """fused_gdn_gating_kernel's expressions on a [BLOCK_T, NUM_HEADS] tile."""
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    h = tl.arange(0, NUM_HEADS)
    t64 = t.to(tl.int64)
    mask = (t < num_tokens)[:, None]
    blk_A_log = tl.load(A_log_ptr + h)
    blk_a = tl.load(a_ptr + t64[:, None] * stride_a + h[None, :], mask=mask)
    blk_b = tl.load(b_ptr + t64[:, None] * stride_b + h[None, :], mask=mask)
    blk_bias = tl.load(dt_bias_ptr + h)
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)[None, :]
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32))[None, :] * softplus_x
    off = t64[:, None] * NUM_HEADS + h[None, :]
    tl.store(g_ptr + off, blk_g.to(g_ptr.dtype.element_ty), mask=mask)
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(beta_ptr + off, blk_beta_output.to(b_ptr.dtype.element_ty), mask=mask)


def gates_eligible(A_log, a, b, dt_bias) -> bool:
    """The layouts gdn_prefill_gates takes (else keep fused_gdn_gating)."""
    heads = a.shape[-1]
    return (
        a.dim() == 2
        and b.shape == a.shape
        and a.stride(1) == 1
        and b.stride(1) == 1
        and heads & (heads - 1) == 0
        and A_log.shape == (heads,)
        and dt_bias.shape == (heads,)
        and A_log.is_contiguous()
        and dt_bias.is_contiguous()
    )


def gdn_prefill_gates(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    T, heads = a.shape
    g = torch.empty(1, T, heads, dtype=torch.float32, device=a.device)
    beta_out = torch.empty(1, T, heads, dtype=torch.float32, device=b.device)
    if T == 0:
        return g, beta_out
    block_t = 32
    _gates_kernel[(triton.cdiv(T, block_t),)](
        g,
        beta_out,
        A_log,
        a,
        b,
        dt_bias,
        T,
        a.stride(0),
        b.stride(0),
        NUM_HEADS=heads,
        beta=beta,
        threshold=threshold,
        BLOCK_T=block_t,
        num_warps=4,
    )
    return g, beta_out


def precompile(
    device_index: int, num_k_heads: int, num_v_heads: int, has_bias: bool, activation
) -> None:
    """Build the CUDA kernel now (NVRTC + module load), outside any capture."""
    _kernel(device_index, num_k_heads, num_v_heads, has_bias, activation is not None)


def warm(attn, heads: int) -> None:
    activation = _activation(attn.activation)
    nk, nv, d = attn.num_k_heads, attn.num_v_heads, HEAD_DIM
    conv_dim = (2 * nk + nv) * d
    dev = attn.conv_weights.device
    rows = torch.zeros((CHUNK, conv_dim + nv * d), dtype=torch.bfloat16, device=dev)
    states = torch.zeros((2, conv_dim, WIDTH - 1), dtype=torch.bfloat16, device=dev)
    gdn_prefill_front(
        rows[:, :conv_dim],
        attn.conv_weights,
        attn.bias,
        states,
        torch.tensor([0, CHUNK], dtype=torch.int32, device=dev),
        torch.ones(1, dtype=torch.int32, device=dev),
        torch.zeros(1, dtype=torch.bool, device=dev),
        num_k_heads=nk,
        num_v_heads=nv,
        activation=activation,
    )
    # token counts of each alignment class, as Triton may specialize integer arguments
    for tokens in (1, 16, 17):
        ba = torch.zeros((tokens, 2 * heads), dtype=torch.bfloat16, device=dev)
        a, b = ba[:, heads:], ba[:, :heads]
        if gates_eligible(attn.A_log, a, b, attn.dt_bias):
            gdn_prefill_gates(attn.A_log, a, b, attn.dt_bias)
