# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/kernels/ops/attention/fused_qk_rmsnorm_rope_gate.py, as SGLang branch
# nvidia/qwen35-agentx-gb300 extends it with the FP8 KV-cache write (attention-front path;
# 98573b3e69, 0d86b61136, 4115dd1469): copied verbatim.
"""Fused Q/K GemmaRMSNorm + NeoX RoPE + gate deinterleave (Triton), and with
``kv_cache`` 2.2's attention front (ATTN_FRONT): FP8 q, and K / V written straight
into the FP8 KV cache with the cache writer's round-to-nearest reciprocal scales, bit
for bit what SGLang's separate fused FP8 q/k/v + KV write produces.

Single kernel launch fusing per-head GemmaRMSNorm, partial NeoX RoPE over 1-D or
mrope positions, and gate deinterleave for Qwen3.5's interleaved Q+Gate layout.

2D grid (T, num_q_heads + num_kv_heads) — each program handles one
(token, head) pair. Q programs also copy the gate slice.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from sgl_nvidia_mega.qwen3_5.sm103.attention.q35_cp_lite import _fail_late, _t_end


def _pdl_supported() -> bool:
    """Check if Programmatic Dependent Launch is supported (NVIDIA SM >= 90)."""
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
        return major >= 9
    except Exception:
        return False


_ENABLE_PDL = _pdl_supported()


@triton.jit
def _st_relaxed_sys_b64(addr, val):
    tl.inline_asm_elementwise(
        "st.relaxed.sys.global.b64 [$1], $2; mov.u32 $0, 0;",
        "=r,l,l",
        [addr, val],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _st_relaxed_sys_b32(addr, val):
    tl.inline_asm_elementwise(
        "st.relaxed.sys.global.b32 [$1], $2; mov.u32 $0, 0;",
        "=r,l,r",
        [addr, val],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _opaque_i32(x):
    return tl.inline_asm_elementwise(
        "mov.b32 $0, $1;", "=r,r", [x], dtype=tl.int32, is_pure=True, pack=1
    )


@triton.jit
def _st_lam_pair_owner(a0, v0, a1, v1, k):
    tl.inline_asm_elementwise(
        "{ .reg .pred p; .reg .u32 t; mov.u32 t, %tid.x; setp.eq.u32 p, t, $5; "
        "@p st.relaxed.sys.global.b32 [$1], $2; @p st.relaxed.sys.global.b32 [$3], $4; } "
        "mov.u32 $0, 0;",
        "=r,l,r,l,r,r",
        [a0, v0, a1, v1, k],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _fused_qk_rmsnorm_rope_gate_kernel(
    q_gate_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    gate_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    mrope_axis_map_ptr,
    stride_qg_t,
    stride_k_t,
    stride_qo_t,
    stride_ko_t,
    stride_gate_t,
    stride_cos_t,
    stride_pos_axis,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    HALF_ROTARY: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    ROT_HALF_BLOCK: tl.constexpr,
    EPS: tl.constexpr,
    FP16: tl.constexpr,
    HAS_PASS: tl.constexpr,
    HAS_GATE: tl.constexpr,
    MROPE: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    WEIGHT_SHIFT: tl.constexpr = 1.0,
    EARLY_TRIGGER: tl.constexpr = False,
    EXPLICIT_ROPE_FMA: tl.constexpr = False,
    v_ptr=None,
    v_cache_ptr=None,
    cache_loc_ptr=None,
    k_scale_ptr=None,
    v_scale_ptr=None,
    stride_v_t: tl.constexpr = 0,
    stride_v_cache_t: tl.constexpr = 0,
    FP8_CACHE: tl.constexpr = False,
    PUSH_Q: tl.constexpr = False,
    push_peers_ptr=None,
    push_me=0,
    push_off=0,
    push_cnt_ptr=None,
    PUSH_CP: tl.constexpr = 4,
    PUSH_WORDS: tl.constexpr = 256,
    PUSH_LAM: tl.constexpr = False,
    GATHER_Q: tl.constexpr = False,
    gq_all_ptr=None,
    gq_seq_lens_ptr=None,
    gq_pre_lens_ptr=None,
    GQ_QL: tl.constexpr = 1,
    gq_tmo=0,
    GQ_DEADLINE: tl.constexpr = False,
):
    # EARLY_TRIGGER: release the dependents at entry; they must wait before reading this kernel's outputs.
    if ENABLE_PDL and EARLY_TRIGGER:
        tl.extra.cuda.gdc_launch_dependents()
    token = tl.program_id(0)
    head = tl.program_id(1)
    is_k = head >= NUM_Q_HEADS
    local_head = tl.where(is_k, head - NUM_Q_HEADS, head)
    out_dtype = tl.float16 if FP16 else tl.bfloat16
    if FP8_CACHE:
        slot = tl.load(cache_loc_ptr + token).to(tl.int64)
        # Match the cache writer's round-to-nearest reciprocal.
        inv_k = tl.div_rn(1.0, tl.load(k_scale_ptr).to(tl.float32))

    if is_k:
        in_base = k_ptr + token * stride_k_t + local_head * HEAD_DIM
        w_ptr = k_weight_ptr
        out_base = k_out_ptr + token * stride_ko_t + local_head * HEAD_DIM
        if FP8_CACHE:
            out_base = k_out_ptr + slot * stride_ko_t + local_head * HEAD_DIM
    else:
        if HAS_GATE:
            in_base = q_gate_ptr + token * stride_qg_t + local_head * 2 * HEAD_DIM
        else:
            in_base = q_gate_ptr + token * stride_qg_t + local_head * HEAD_DIM
        w_ptr = q_weight_ptr
        out_base = q_out_ptr + token * stride_qo_t + local_head * HEAD_DIM

    # Full load -> RMSNorm variance
    head_offs = tl.arange(0, HEAD_BLOCK)
    head_mask = head_offs < HEAD_DIM
    x = tl.load(in_base + head_offs, mask=head_mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + head_offs, mask=head_mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / HEAD_DIM
    inv_rms = tl.rsqrt(var + EPS)
    x_norm = (x * inv_rms * (w + WEIGHT_SHIFT)).to(out_dtype).to(tl.float32)

    # Pass-through tail [rotary_dim, head_dim)
    if HAS_PASS:
        pass_mask = head_mask & (head_offs >= ROTARY_DIM)
        tail = x_norm
        if FP8_CACHE and is_k:
            tail *= inv_k
        tl.store(out_base + head_offs, tail, mask=pass_mask)

    # Reload rotary portion from L1 -> re-norm -> RoPE
    rot_offs = tl.arange(0, ROT_HALF_BLOCK)
    rot_mask = rot_offs < HALF_ROTARY
    xr1 = tl.load(in_base + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
    xr2 = tl.load(in_base + HALF_ROTARY + rot_offs, mask=rot_mask, other=0.0).to(
        tl.float32
    )
    wr1 = tl.load(w_ptr + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
    wr2 = tl.load(w_ptr + HALF_ROTARY + rot_offs, mask=rot_mask, other=0.0).to(
        tl.float32
    )
    xr1 = (xr1 * inv_rms * (wr1 + WEIGHT_SHIFT)).to(out_dtype).to(tl.float32)
    xr2 = (xr2 * inv_rms * (wr2 + WEIGHT_SHIFT)).to(out_dtype).to(tl.float32)

    if MROPE:
        axis = tl.load(mrope_axis_map_ptr + rot_offs, mask=rot_mask, other=0)
        pos = tl.load(
            positions_ptr + axis * stride_pos_axis + token, mask=rot_mask, other=0
        )
    else:
        pos = tl.load(positions_ptr + token)
    cache_off = pos.to(tl.int64) * stride_cos_t
    cos = tl.load(
        cos_sin_cache_ptr + cache_off + rot_offs, mask=rot_mask, other=0.0
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_cache_ptr + cache_off + HALF_ROTARY + rot_offs, mask=rot_mask, other=0.0
    ).to(tl.float32)
    if EXPLICIT_ROPE_FMA:
        first = tl.fma(xr1, cos, -libdevice.mul_rn(xr2, sin))
        second = tl.fma(xr2, cos, libdevice.mul_rn(xr1, sin))
    else:
        first = xr1 * cos - xr2 * sin
        second = xr2 * cos + xr1 * sin
    if FP8_CACHE:
        first = first.to(out_dtype).to(tl.float32)
        second = second.to(out_dtype).to(tl.float32)
        if is_k:
            first *= inv_k
            second *= inv_k
    tl.store(out_base + rot_offs, first, mask=rot_mask)
    tl.store(out_base + HALF_ROTARY + rot_offs, second, mask=rot_mask)

    if FP8_CACHE and is_k:
        value = tl.load(
            v_ptr + token * stride_v_t + local_head * HEAD_DIM + head_offs,
            head_mask,
            0.0,
        ).to(tl.float32)
        inv_v = tl.div_rn(1.0, tl.load(v_scale_ptr).to(tl.float32))
        value_out = v_cache_ptr + slot * stride_v_cache_t + local_head * HEAD_DIM
        tl.store(value_out + head_offs, value * inv_v, head_mask)

    # Gate copy (Q heads only)
    if HAS_GATE and not is_k:
        gate_in = in_base + HEAD_DIM
        gate_out = gate_out_ptr + token * stride_gate_t + local_head * HEAD_DIM
        g = tl.load(gate_in + head_offs, mask=head_mask, other=0.0)
        tl.store(gate_out + head_offs, g, mask=head_mask)

    if PUSH_Q and PUSH_LAM:
        if not is_k:
            tl.debug_barrier()
            NWQ: tl.constexpr = HEAD_DIM // 4
            tl.static_assert(PUSH_CP * NWQ == 256, "two words a thread of 128")
            k = _opaque_i32(tl.arange(0, 128))
            f1 = k + 128
            qb = (q_out_ptr + token * stride_qo_t + local_head * HEAD_DIM).to(
                tl.pointer_type(tl.int32)
            )
            q0 = tl.load(qb + k % NWQ)
            q1 = tl.load(qb + f1 % NWQ)
            row = (token * PUSH_CP + push_me) * PUSH_WORDS + local_head * NWQ
            d0 = (
                tl.load(push_peers_ptr + (push_me + k // NWQ) % PUSH_CP)
                + push_off
                + (row + k % NWQ).to(tl.int64) * 4
            )
            d1 = (
                tl.load(push_peers_ptr + (push_me + f1 // NWQ) % PUSH_CP)
                + push_off
                + (row + f1 % NWQ).to(tl.int64) * 4
            )
            _st_lam_pair_owner(
                d0,
                tl.where(q0 == -1, 0x7FFFFFFF, q0),
                d1,
                tl.where(q1 == -1, 0x7FFFFFFF, q1),
                k,
            )
            if GATHER_Q:
                cc = tl.arange(0, PUSH_CP)
                jw = tl.arange(0, NWQ)
                own = tl.load(push_peers_ptr + push_me) + push_off
                widx = ((token * PUSH_CP + cc) * PUSH_WORDS + local_head * NWQ).to(
                    tl.int64
                )
                laddr = own + (widx[:, None] + jw[None, :].to(tl.int64)) * 4
                src = laddr.to(tl.pointer_type(tl.int32))
                if GQ_DEADLINE:
                    t_end = _t_end(gq_tmo)
                v = tl.load(src, volatile=True)
                done = tl.min(tl.min((v != -1).to(tl.int32), axis=1), axis=0)
                while done == 0:
                    if GQ_DEADLINE:
                        _fail_late(t_end, done)
                    v = tl.load(src, volatile=True)
                    done = tl.min(tl.min((v != -1).to(tl.int32), axis=1), axis=0)
                slot_q = ((token // GQ_QL) * PUSH_CP + cc) * GQ_QL + token % GQ_QL
                tl.store(
                    gq_all_ptr
                    + (slot_q * PUSH_WORDS + local_head * NWQ)[:, None]
                    + jw[None, :],
                    v,
                )
                if local_head == 0:
                    if token % GQ_QL == 0:
                        tl.store(
                            gq_pre_lens_ptr + token // GQ_QL,
                            tl.maximum(
                                tl.load(gq_seq_lens_ptr + token // GQ_QL) - GQ_QL, 0
                            ),
                        )
                    tl.store(push_cnt_ptr + token, tl.load(push_cnt_ptr + token) + 1)
                _st_relaxed_sys_b32(laddr, (laddr & 0).to(tl.int32) - 1)
    elif PUSH_Q:
        if not is_k:
            tl.debug_barrier()
            e = tl.load(push_cnt_ptr + token) + 1
            flag = e.to(tl.int64) << 32
            wq = tl.arange(0, HEAD_DIM // 4)
            qw = tl.load(
                (q_out_ptr + token * stride_qo_t + local_head * HEAD_DIM).to(
                    tl.pointer_type(tl.int32)
                )
                + wq
            )
            xw = flag | (qw.to(tl.int64) & 0xFFFFFFFF)
            for c in tl.static_range(PUSH_CP):
                dst = (
                    tl.load(push_peers_ptr + (push_me + c) % PUSH_CP)
                    + push_off
                    + (
                        (token * PUSH_CP + push_me) * PUSH_WORDS
                        + local_head * (HEAD_DIM // 4)
                    )
                    * 8
                )
                _st_relaxed_sys_b64(dst + wq.to(tl.int64) * 8, xw)

    # PDL: signal dependent kernels (attention/allreduce) can start early.
    # Only available on NVIDIA Hopper+ (sm_90+); guarded for AMD/other backends.
    if ENABLE_PDL and not EARLY_TRIGGER:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _fused_qk_rmsnorm_rope_gate_pdl_kernel(
    q_gate_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    gate_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    mrope_axis_map_ptr,
    v_ptr,
    v_cache_ptr,
    cache_loc_ptr,
    k_scale_ptr,
    v_scale_ptr,
    stride_qg_t,
    stride_k_t,
    stride_qo_t,
    stride_ko_t,
    stride_gate_t,
    stride_cos_t,
    stride_pos_axis,
    stride_v_t: tl.constexpr,
    stride_v_cache_t: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    HALF_ROTARY: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    ROT_HALF_BLOCK: tl.constexpr,
    EPS: tl.constexpr,
    FP16: tl.constexpr,
    HAS_PASS: tl.constexpr,
    MROPE: tl.constexpr,
    EARLY_TRIGGER: tl.constexpr,
    WEIGHT_SHIFT: tl.constexpr = 1.0,
):
    # The kernel above with an FP8 cache and a gate, launched as a programmatic dependent of the qkv GEMM:
    # static operands load before griddepcontrol.wait, all other loads are issued right after it; same arithmetic.
    if EARLY_TRIGGER:
        tl.extra.cuda.gdc_launch_dependents()
    token = tl.program_id(0)
    head = tl.program_id(1)
    is_k = head >= NUM_Q_HEADS
    local_head = tl.where(is_k, head - NUM_Q_HEADS, head)
    out_dtype = tl.float16 if FP16 else tl.bfloat16
    head_offs = tl.arange(0, HEAD_BLOCK)
    head_mask = head_offs < HEAD_DIM
    rot_offs = tl.arange(0, ROT_HALF_BLOCK)
    rot_mask = rot_offs < HALF_ROTARY

    if is_k:
        w_ptr = k_weight_ptr
    else:
        w_ptr = q_weight_ptr
    w = tl.load(w_ptr + head_offs, mask=head_mask, other=0.0).to(tl.float32)
    wr1 = tl.load(w_ptr + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
    wr2 = tl.load(w_ptr + HALF_ROTARY + rot_offs, mask=rot_mask, other=0.0).to(
        tl.float32
    )
    # Match the cache writer's round-to-nearest reciprocal.
    inv_k = tl.div_rn(1.0, tl.load(k_scale_ptr).to(tl.float32))
    inv_v = tl.div_rn(1.0, tl.load(v_scale_ptr).to(tl.float32))
    if MROPE:
        axis = tl.load(mrope_axis_map_ptr + rot_offs, mask=rot_mask, other=0)

    tl.extra.cuda.gdc_wait()
    slot = tl.load(cache_loc_ptr + token).to(tl.int64)
    if MROPE:
        pos = tl.load(
            positions_ptr + axis * stride_pos_axis + token, mask=rot_mask, other=0
        )
    else:
        pos = tl.load(positions_ptr + token)
    if is_k:
        in_base = k_ptr + token * stride_k_t + local_head * HEAD_DIM
        out_base = k_out_ptr + slot * stride_ko_t + local_head * HEAD_DIM
    else:
        in_base = q_gate_ptr + token * stride_qg_t + local_head * 2 * HEAD_DIM
        out_base = q_out_ptr + token * stride_qo_t + local_head * HEAD_DIM
    x = tl.load(in_base + head_offs, mask=head_mask, other=0.0).to(tl.float32)
    xr1 = tl.load(in_base + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
    xr2 = tl.load(in_base + HALF_ROTARY + rot_offs, mask=rot_mask, other=0.0).to(
        tl.float32
    )
    value = tl.load(
        v_ptr + token * stride_v_t + local_head * HEAD_DIM + head_offs,
        head_mask & is_k,
        0.0,
    ).to(tl.float32)
    g = tl.load(
        in_base + HEAD_DIM + head_offs, mask=head_mask & (head < NUM_Q_HEADS), other=0.0
    )
    cache_off = pos.to(tl.int64) * stride_cos_t
    cos = tl.load(
        cos_sin_cache_ptr + cache_off + rot_offs, mask=rot_mask, other=0.0
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_cache_ptr + cache_off + HALF_ROTARY + rot_offs, mask=rot_mask, other=0.0
    ).to(tl.float32)

    var = tl.sum(x * x, axis=0) / HEAD_DIM
    inv_rms = tl.rsqrt(var + EPS)
    x_norm = (x * inv_rms * (w + WEIGHT_SHIFT)).to(out_dtype).to(tl.float32)

    if HAS_PASS:
        pass_mask = head_mask & (head_offs >= ROTARY_DIM)
        tail = x_norm
        if is_k:
            tail *= inv_k
        tl.store(out_base + head_offs, tail, mask=pass_mask)

    xr1 = (xr1 * inv_rms * (wr1 + WEIGHT_SHIFT)).to(out_dtype).to(tl.float32)
    xr2 = (xr2 * inv_rms * (wr2 + WEIGHT_SHIFT)).to(out_dtype).to(tl.float32)
    first = xr1 * cos - xr2 * sin
    second = xr2 * cos + xr1 * sin
    first = first.to(out_dtype).to(tl.float32)
    second = second.to(out_dtype).to(tl.float32)
    if is_k:
        first *= inv_k
        second *= inv_k
    tl.store(out_base + rot_offs, first, mask=rot_mask)
    tl.store(out_base + HALF_ROTARY + rot_offs, second, mask=rot_mask)

    if is_k:
        value_out = v_cache_ptr + slot * stride_v_cache_t + local_head * HEAD_DIM
        tl.store(value_out + head_offs, value * inv_v, head_mask)
    else:
        gate_out = gate_out_ptr + token * stride_gate_t + local_head * HEAD_DIM
        tl.store(gate_out + head_offs, g, mask=head_mask)
    if not EARLY_TRIGGER:
        tl.extra.cuda.gdc_launch_dependents()


def fused_qk_gemma_rmsnorm_rope_gate(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    has_gate: bool = True,
    mrope_axis_map: Optional[torch.Tensor] = None,
    *,
    v: Optional[torch.Tensor] = None,
    kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    cache_loc: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    early_trigger: bool = False,
    pdl_wait: bool = False,
    q_push: Optional[dict] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Fused QK GemmaRMSNorm + NeoX RoPE + gate deinterleave.

    early_trigger: release programmatic dependents at entry (they must wait before reading the outputs).
    pdl_wait: with an FP8 cache and a gate, launch as a programmatic dependent of the producer of q_gate / k / v
    (static operands load before the wait); same outputs.

    Args:
        q_gate: [T, num_q_heads * (1 + has_gate) * head_dim] — interleaved Q+Gate if has_gate
        k: [T, num_kv_heads * head_dim]
        q_weight, k_weight: [head_dim] — raw GemmaRMSNorm weights (kernel adds +1.0)
        cos_sin_cache: [max_seq_len, rotary_dim] — [cos..., sin...]
        positions: [T] token positions, or [3, T] mrope rows (temporal, height, width)
        mrope_axis_map: [rotary_dim // 2] — the axis owning each rotary lane, from
            MRotaryEmbedding
    """
    assert positions.dim() in (1, 2), f"want [T] or [3, T], got {positions.shape}"
    mrope = positions.dim() == 2
    assert mrope == (mrope_axis_map is not None), "mrope_axis_map needs [3, T]"
    if mrope:
        assert positions.shape[0] == 3 and positions.stride(1) == 1, (
            f"want [3, T] contiguous over T, got {positions.shape} "
            f"stride {positions.stride()}"
        )
        lanes = rotary_dim // 2
        assert mrope_axis_map.shape == (lanes,), f"want one axis per lane ({lanes})"
    T = q_gate.shape[0]
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim

    fp8_cache = kv_cache is not None
    q_out = torch.empty(
        T,
        q_size,
        dtype=torch.float8_e4m3fn if fp8_cache else q_gate.dtype,
        device=q_gate.device,
    )
    k_out = (
        kv_cache[0]
        if fp8_cache
        else torch.empty(T, kv_size, dtype=k.dtype, device=k.device)
    )
    gate_out = (
        torch.empty(T, num_q_heads, head_dim, dtype=q_gate.dtype, device=q_gate.device)
        if has_gate
        else q_out
    )

    half_rotary = rotary_dim // 2
    head_block = triton.next_power_of_2(head_dim)
    rot_half_block = triton.next_power_of_2(half_rotary)

    grid = (T, num_q_heads + num_kv_heads)
    kwargs = {}
    if fp8_cache:
        assert v is not None and cache_loc is not None
        assert k_scale is not None and v_scale is not None
        assert kv_cache[0].dtype == kv_cache[1].dtype == torch.float8_e4m3fn
        kwargs = dict(
            v_ptr=v,
            v_cache_ptr=kv_cache[1],
            cache_loc_ptr=cache_loc,
            k_scale_ptr=k_scale,
            v_scale_ptr=v_scale,
            stride_v_t=v.stride(0),
            stride_v_cache_t=kv_cache[1].stride(0),
            FP8_CACHE=True,
        )
    if q_push is not None:
        assert fp8_cache and not pdl_wait, "q_push rides the FP8-cache front"
        kwargs.update(
            PUSH_Q=True,
            push_peers_ptr=q_push["peers"],
            push_me=q_push["me"],
            push_off=q_push["off"],
            push_cnt_ptr=q_push["cnt"],
            PUSH_CP=q_push["cp"],
            PUSH_WORDS=q_push["words"],
        )
        if q_push.get("lam", False):
            kwargs.update(PUSH_LAM=True, num_warps=4)
        gq = q_push.get("gather")
        if gq is not None:
            assert q_push.get("lam", False), "the in-front q gather polls Lamport words"
            assert (
                gq["seq_lens"].dtype == torch.int32 and gq["q_all"].dtype == torch.int32
            )
            kwargs.update(
                GATHER_Q=True,
                gq_all_ptr=gq["q_all"],
                gq_seq_lens_ptr=gq["seq_lens"],
                gq_pre_lens_ptr=gq["pre_lens"],
                GQ_QL=int(gq["ql"]),
                **(
                    {"gq_tmo": int(gq["tmo"]), "GQ_DEADLINE": True}
                    if gq.get("tmo")
                    else {}
                ),
            )
            early_trigger = True
    if pdl_wait and fp8_cache and has_gate and _ENABLE_PDL:
        _fused_qk_rmsnorm_rope_gate_pdl_kernel[grid](
            q_gate,
            k,
            q_out,
            k_out,
            gate_out,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            mrope_axis_map,
            v,
            kv_cache[1],
            cache_loc,
            k_scale,
            v_scale,
            q_gate.stride(0),
            k.stride(0),
            q_out.stride(0),
            k_out.stride(0),
            gate_out.stride(0),
            cos_sin_cache.stride(0),
            positions.stride(0),
            v.stride(0),
            kv_cache[1].stride(0),
            NUM_Q_HEADS=num_q_heads,
            HEAD_DIM=head_dim,
            ROTARY_DIM=rotary_dim,
            HALF_ROTARY=half_rotary,
            HEAD_BLOCK=head_block,
            ROT_HALF_BLOCK=rot_half_block,
            EPS=eps,
            FP16=q_gate.dtype == torch.float16,
            HAS_PASS=rotary_dim < head_dim,
            MROPE=mrope,
            EARLY_TRIGGER=early_trigger,
            launch_pdl=True,
        )
        return q_out, None, gate_out
    _fused_qk_rmsnorm_rope_gate_kernel[grid](
        q_gate,
        k,
        q_out,
        k_out,
        gate_out,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        mrope_axis_map,
        q_gate.stride(0),
        k.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        gate_out.stride(0),
        cos_sin_cache.stride(0),
        positions.stride(0),
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        ROTARY_DIM=rotary_dim,
        HALF_ROTARY=half_rotary,
        HEAD_BLOCK=head_block,
        ROT_HALF_BLOCK=rot_half_block,
        EPS=eps,
        FP16=q_gate.dtype == torch.float16,
        HAS_PASS=rotary_dim < head_dim,
        HAS_GATE=has_gate,
        MROPE=mrope,
        ENABLE_PDL=_ENABLE_PDL,
        EARLY_TRIGGER=early_trigger,
        **kwargs,
    )

    return q_out, None if fp8_cache else k_out, gate_out if has_gate else None
