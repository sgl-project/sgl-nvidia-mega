# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Ported from TensorRT-LLM (Apache-2.0): tensorrt_llm/_torch/kernels/blackwell/mnnvl_slot_allreduce_norm.py and the
# helpers of tensorrt_llm/_torch/kernels/blackwell/mnnvl_allreduce_norm.py it imports.
# This file has been modified from its original version (the opt-in ``bt_numerics`` rounding points and two
# quant helpers).
"""One-shot all-reduce + residual + RMSNorm on a flag-free slot protocol (CuTe DSL)."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass import Float32, Int32, Int64, const_expr
from cutlass._mlir import ir  # noqa: F401  (the helpers' module imported it)
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from sgl_nvidia_mega.qwen3_5.sm103.moe.router_ptx import _asm, _smem_addr, _v
from sgl_nvidia_mega.qwen3_5.sm103.trt_lowm.utils import (
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
)

VEC = 8
_NEG_ZERO = -2147483648
_FC2_SENT = -1


@dsl_user_op
def _ld_v4(addr: Int64, *, loc=None, ip=None):
    words = _asm(
        ir.Type.parse("!llvm.struct<(i32,i32,i32,i32)>"),
        [_v(Int64(addr), loc, ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l,~{memory}",
        loc=loc,
        ip=ip,
    )
    return tuple(
        Int32(llvm.extractvalue(T.i32(), words, [i], loc=loc, ip=ip)) for i in range(4)
    )


@dsl_user_op
def _ld_volatile_v4(addr: Int64, *, loc=None, ip=None):
    words = _asm(
        ir.Type.parse("!llvm.struct<(i32,i32,i32,i32)>"),
        [_v(Int64(addr), loc, ip)],
        "ld.volatile.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l,~{memory}",
        loc=loc,
        ip=ip,
    )
    return tuple(
        Int32(llvm.extractvalue(T.i32(), words, [i], loc=loc, ip=ip)) for i in range(4)
    )


@dsl_user_op
def _st_v4(
    addr: Int64, w0: Int32, w1: Int32, w2: Int32, w3: Int32, *, loc=None, ip=None
) -> None:
    _asm(
        None,
        [_v(Int64(addr), loc, ip)] + [_v(Int32(w), loc, ip) for w in (w0, w1, w2, w3)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_v2(addr: Int64, w0: Int32, w1: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [_v(Int64(addr), loc, ip), _v(Int32(w0), loc, ip), _v(Int32(w1), loc, ip)],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_s64(addr: Int64, *, loc=None, ip=None) -> Int64:
    return Int64(
        _asm(
            T.i64(),
            [_v(Int64(addr), loc, ip)],
            "ld.global.b64 $0, [$1];",
            "=l,l,~{memory}",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_u16(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_v(Int64(addr), loc, ip)],
            "{\n\t.reg .b16 h;\n\tld.global.b16 h, [$1];\n\tcvt.u32.u16 $0, h;\n\t}",
            "=r,l,~{memory}",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_s32(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_v(Int64(addr), loc, ip)],
            "ld.global.b32 $0, [$1];",
            "=r,l,~{memory}",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_s32(addr: Int64, value: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [_v(Int64(addr), loc, ip), _v(Int32(value), loc, ip)],
        "st.global.b32 [$0], $1;",
        "l,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sanitize(word: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_v(Int32(word), loc, ip)],
            "{\n\t.reg .b32 t, r;\n\t.reg .pred p;\n\tmov.b32 r, $1;\n\t"
            "and.b32 t, r, 0xffff0000;\n\tsetp.eq.b32 p, t, 0x80000000;\n\t@p and.b32 r, r, 0x0000ffff;\n\t"
            "and.b32 t, r, 0x0000ffff;\n\tsetp.eq.b32 p, t, 0x00008000;\n\t@p and.b32 r, r, 0xffff0000;\n\t"
            "mov.b32 $0, r;\n\t}",
            "=r,r",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _lo(word: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(Int32(word), loc, ip)],
            "{\n\t.reg .b32 t;\n\tshl.b32 t, $1, 16;\n\tmov.b32 $0, t;\n\t}",
            "=f,r",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _hi(word: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(Int32(word), loc, ip)],
            "{\n\t.reg .b32 t;\n\tand.b32 t, $1, 0xffff0000;\n\tmov.b32 $0, t;\n\t}",
            "=f,r",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _pack_bf16(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Int32:
    """Two FP32 rounded to BF16 (nearest even), ``lo`` in the low half."""
    return Int32(
        _asm(
            T.i32(),
            [_v(Float32(lo), loc, ip), _v(Float32(hi), loc, ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bf16r(x: Float32, *, loc=None, ip=None) -> Float32:
    """FP32 -> BF16 (nearest even) -> FP32."""
    return Float32(
        _asm(
            T.f32(),
            [_v(Float32(x), loc, ip)],
            "{\n\t.reg .b16 h;\n\t.reg .b32 t;\n\tcvt.rn.bf16.f32 h, $1;\n\tmov.b32 t, {0, h};\n\t"
            "mov.b32 $0, t;\n\t}",
            "=f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _add(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(Float32(a), loc, ip), _v(Float32(b), loc, ip)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _mul(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(Float32(a), loc, ip), _v(Float32(b), loc, ip)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sel_ge0(p: Int32, a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """``a if p >= 0 else b`` (selp, no branch)."""
    return Float32(
        _asm(
            T.f32(),
            [_v(Int32(p), loc, ip), _v(Float32(a), loc, ip), _v(Float32(b), loc, ip)],
            "{\n\t.reg .pred q;\n\tsetp.ge.s32 q, $1, 0;\n\tselp.f32 $0, $2, $3, q;\n\t}",
            "=f,r,f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _max0(p: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_v(Int32(p), loc, ip)],
            "max.s32 $0, $1, 0;",
            "=r,r",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fma(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(Float32(a), loc, ip), _v(Float32(b), loc, ip), _v(Float32(c), loc, ip)],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _div_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(Float32(a), loc, ip), _v(Float32(b), loc, ip)],
            "div.rn.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _div_full_bt(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(Float32(a), loc, ip), _v(Float32(b), loc, ip)],
            "div.full.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _clamp448_bt(x: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(Float32(x), loc, ip)],
            "{\n\t.reg .f32 t;\n\tmax.f32 t, $1, 0fC3E00000;\n\tmin.f32 $0, t, 0f43E00000;\n\t}",
            "=f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _rsqrt(x: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(Float32(x), loc, ip)],
            "rsqrt.approx.f32 $0, $1;",
            "=f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sigmoid_triton(g: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(Float32(g), loc, ip)],
            "{\n\t.reg .f32 t, e;\n\tmul.f32 t, $1, 0fBFB8AA3B;\n\tex2.approx.f32 e, t;\n\t"
            "add.f32 e, e, 0f3F800000;\n\tdiv.full.f32 $0, 0f3F800000, e;\n\t}",
            "=f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _e4m3x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_v(Float32(lo), loc, ip), _v(Float32(hi), loc, ip)],
            "{\n\t.reg .b16 h;\n\tcvt.rn.satfinite.e4m3x2.f32 h, $2, $1;\n\tcvt.u32.u16 $0, h;\n\t}",
            "=r,f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shfl_bfly(x: Float32, mask: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(Float32(x), loc, ip), _v(Int32(mask), loc, ip)],
            "shfl.sync.bfly.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=f,f,r",
            True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sink4(
    addr: Int32, a: Int32, b: Int32, c: Int32, d: Int32, *, loc=None, ip=None
) -> None:
    _asm(
        None,
        [_v(Int32(addr), loc, ip)] + [_v(Int32(w), loc, ip) for w in (a, b, c, d)],
        "{\n\t.reg .b32 t;\n\txor.b32 t, $1, $2;\n\txor.b32 t, t, $3;\n\txor.b32 t, t, $4;\n\t"
        "st.shared.b32 [$0], t;\n\t}",
        "r,r,r,r,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_shared_f32(addr: Int32, value: Float32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [_v(Int32(addr), loc, ip), _v(Float32(value), loc, ip)],
        "st.shared.f32 [$0], $1;",
        "r,f,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_shared_f32(addr: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(Int32(addr), loc, ip)],
            "ld.shared.f32 $0, [$1];",
            "=f,r,~{memory}",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _cluster_sync(*, loc=None, ip=None) -> None:
    _asm(
        None,
        [],
        "barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;",
        "~{memory}",
        loc=loc,
        ip=ip,
    )


SLOT_STATE_STRIDE = 128
_FC2_SENT = -1


@dsl_user_op
def _ld_cg_s32(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.cg.b32 $0, [$1];",
            "=r,l,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _nanosleep(ns: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Int32(ns).ir_value(loc=loc, ip=ip)],
        "nanosleep.u32 $0;",
        "r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mbar_init(addr: Int32, count: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Int32(count).ir_value(loc=loc, ip=ip)],
        "mbarrier.init.shared::cta.b64 [$0], $1;\n\tfence.mbarrier_init.release.cluster;",
        "r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mbar_expect_tx(addr: Int32, nbytes: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Int32(nbytes).ir_value(loc=loc, ip=ip)],
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1;",
        "r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mbar_try_wait(addr: Int32, parity: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Int32(addr).ir_value(loc=loc, ip=ip),
                Int32(parity).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2;\n\t"
            "selp.u32 $0, 1, 0, p;\n\t}",
            "=r,r,r,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_async_f32(
    addr: Int32, mbar: Int32, cta_rank: Int32, value: Float32, *, loc=None, ip=None
) -> None:
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Int32(mbar).ir_value(loc=loc, ip=ip),
            Int32(cta_rank).ir_value(loc=loc, ip=ip),
            Float32(value).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .b32 ra, rb, v;\n\tmapa.shared::cluster.u32 ra, $0, $2;\n\t"
        "mapa.shared::cluster.u32 rb, $1, $2;\n\tmov.b32 v, $3;\n\t"
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [ra], v, [rb];\n\t}",
        "r,r,r,f,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _tree_sum_smem(red_addr, n: int):
    vals = [_ld_shared_f32(red_addr + Int32(4 * i)) for i in range(n)]
    s = n // 2
    while s >= 1:
        vals = [_add(vals[i], vals[i ^ s]) for i in range(n)]
        s //= 2
    return vals[0]


class MnnvlSlotAllReduceNormKernel:
    def __init__(
        self,
        *,
        hidden: int,
        norm_eps: float,
        nranks: int,
        ctas: int,
        use_pdl: bool,
        se_fold: bool,
        norm_out: bool,
        quant: bool,
        region_bytes: int,
        moe_topk: int = 0,
        fc2_row: int = 0,
        fc2_lamport: bool = False,
        test_no_poll: bool = False,
        test_no_end_wait: bool = False,
        test_store_early: bool = False,
        ts_ctas: int = 0,
        poll_ns: int = 0,
        bt_numerics: bool = False,
    ):
        if (
            hidden % (VEC * ctas)
            or (hidden // VEC // ctas) % 32
            or hidden // VEC // ctas > 1024
        ):
            raise ValueError(
                f"unsupported hidden size {hidden} for {ctas} CTAs per token"
            )
        if nranks not in (2, 4, 8, 16) or ctas not in (1, 2, 4, 8):
            raise ValueError("unsupported group / cluster size")
        if region_bytes <= 0:
            raise ValueError("the slot protocol needs region_bytes")
        token_warps = hidden // VEC // 32
        if ts_ctas and (
            ts_ctas not in (1, 2, 4, 8)
            or token_warps % ts_ctas
            or not 1 <= token_warps // ts_ctas <= 32
            or (token_warps // ts_ctas) & (token_warps // ts_ctas - 1)
        ):
            raise ValueError(
                f"unsupported two-shot CTA split {ts_ctas} for hidden size {hidden}"
            )
        self.ts_ctas = ts_ctas
        self.bt_numerics = bool(bt_numerics)
        if self.bt_numerics and (
            hidden // VEC // 32 > 32
            or (hidden // VEC // 32) & (hidden // VEC // 32 - 1)
        ):
            raise ValueError(
                f"bt_numerics: the warp-sum butterfly takes 1-32 warps a token (a power of 2): {hidden}"
            )
        self.poll_ns = int(poll_ns)
        self.hidden = hidden
        self.norm_eps = float(norm_eps)
        self.nranks = nranks
        self.ctas = ctas
        self.threads = hidden // VEC // ctas
        self.num_warps = self.threads // 32
        self.use_pdl = use_pdl
        self.se_fold = se_fold
        self.moe_topk = moe_topk
        self.fc2_row = fc2_row or hidden
        self.norm_out = norm_out
        self.quant = quant
        self.region_bytes = region_bytes
        self.fc2_lamport = fc2_lamport
        self.test_no_poll = test_no_poll
        self.test_no_end_wait = test_no_end_wait
        self.test_store_early = test_store_early
        self.store_wait = bool(
            use_pdl and fc2_lamport and not test_store_early and not test_no_end_wait
        )
        if fc2_lamport and not (se_fold and moe_topk > 0):
            raise ValueError("fc2_lamport needs the in-kernel MoE finalize")

    @cute.jit
    def __call__(
        self,
        gX: cute.Tensor,
        gRouted: cute.Tensor,
        gShared: cute.Tensor,
        gGate: cute.Tensor,
        gRes: cute.Tensor,
        gW: cute.Tensor,
        gScale: cute.Tensor,
        gResOut: cute.Tensor,
        gOut: cute.Tensor,
        gQuant: cute.Tensor,
        own: Int64,
        mcast: Int64,
        ctl: Int64,
        fc2_ptr: Int64,
        topw_ptr: Int64,
        perm_ptr: Int64,
        rank: Int32,
        stream,
    ) -> None:
        m = cute.size(gRes, mode=[0])
        self.kernel(
            gX,
            gRouted,
            gShared,
            gGate,
            gRes,
            gW,
            gScale,
            gResOut,
            gOut,
            gQuant,
            own,
            mcast,
            ctl,
            fc2_ptr,
            topw_ptr,
            perm_ptr,
            rank,
        ).launch(
            grid=[m * self.ctas, 1, 1],
            block=[self.threads, 1, 1],
            cluster=[self.ctas, 1, 1],
            smem=512,
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        gX: cute.Tensor,
        gRouted: cute.Tensor,
        gShared: cute.Tensor,
        gGate: cute.Tensor,
        gRes: cute.Tensor,
        gW: cute.Tensor,
        gScale: cute.Tensor,
        gResOut: cute.Tensor,
        gOut: cute.Tensor,
        gQuant: cute.Tensor,
        own: Int64,
        mcast: Int64,
        ctl: Int64,
        fc2_ptr: Int64,
        topw_ptr: Int64,
        perm_ptr: Int64,
        rank: Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        ctas: cutlass.Constexpr = self.ctas
        nranks: cutlass.Constexpr = self.nranks
        vecs: cutlass.Constexpr = self.hidden // VEC
        num_warps: cutlass.Constexpr = self.num_warps
        threads: cutlass.Constexpr = self.threads
        tok = bid // ctas
        part = bid % ctas
        vi = part * threads + tidx
        warp = tidx // 32
        lane = tidx % 32
        row_off = (Int64(tok) * Int64(vecs) + Int64(vi)) * Int64(16)

        ts_ctas: cutlass.Constexpr = self.ts_ctas
        bt: cutlass.Constexpr = self.bt_numerics
        nsum: cutlass.Constexpr = ctas * num_warps if (ts_ctas or bt) else ctas
        smem = cutlass.utils.SmemAllocator()
        red = smem.allocate_tensor(
            Float32, cute.make_layout((num_warps,)), byte_alignment=16
        )
        csum = smem.allocate_tensor(
            Float32, cute.make_layout((nsum,)), byte_alignment=16
        )
        fl = smem.allocate_tensor(Int32, cute.make_layout((8,)), byte_alignment=16)
        red_addr = _smem_addr(red.iterator)
        csum_addr = _smem_addr(csum.iterator)

        mbar_addr = Int32(0)
        if const_expr(ctas > 1):
            mb = smem.allocate_tensor(Int64, cute.make_layout((1,)), byte_alignment=8)
            mbar_addr = _smem_addr(mb.iterator)
            if tidx == 0:
                _mbar_init(mbar_addr, Int32(1))
            _cluster_sync()
            if tidx == 0:
                _mbar_expect_tx(mbar_addr, Int32(4 * nsum))

        # ---- before the dependency wait: only the norm weight ----
        w0, w1, w2, w3 = _ld_v4(Int64(gW.iterator.toint()) + Int64(vi) * Int64(16))
        topk: cutlass.Constexpr = self.moe_topk
        pk = cute.make_rmem_tensor((max(topk, 1),), Int32)
        wk = cute.make_rmem_tensor((max(topk, 1),), Int32)

        if const_expr(self.use_pdl and not self.fc2_lamport):
            griddepcontrol_wait()

        r0, r1, r2, r3 = _ld_v4(Int64(gRes.iterator.toint()) + row_off)
        if const_expr(self.se_fold):
            if const_expr(topk > 0):
                # routing outputs of this step: only after the dependency wait (RAW)
                kbase = Int64(tok) * Int64(topk)
                for k in cutlass.range_constexpr(topk):
                    pk[k] = _ld_s32(perm_ptr + (kbase + Int64(k)) * Int64(4))
                    wk[k] = _ld_u16(topw_ptr + (kbase + Int64(k)) * Int64(2))
            else:
                f0, f1, f2, f3 = _ld_v4(Int64(gRouted.iterator.toint()) + row_off)
            h0, h1, h2, h3 = _ld_v4(Int64(gShared.iterator.toint()) + row_off)
            gw = _ld_u16(Int64(gGate.iterator.toint()) + Int64(tok) * Int64(2))
        else:
            x0, x1, x2, x3 = _ld_v4(Int64(gX.iterator.toint()) + row_off)

        if tidx == 0:
            fl[0] = _ld_cg_s32(ctl + Int64(bid) * Int64(SLOT_STATE_STRIDE))
        cute.arch.sync_threads()

        if const_expr(self.se_fold):
            if const_expr(topk > 0):
                ev = cute.make_rmem_tensor((topk * 4,), Int32)
                for k in cutlass.range_constexpr(topk):
                    fa = fc2_ptr + (
                        Int64(_max0(pk[k])) * Int64(self.fc2_row)
                        + Int64(vi) * Int64(VEC)
                    ) * Int64(2)
                    if const_expr(self.fc2_lamport):
                        e0 = Int32(0)
                        e1 = Int32(0)
                        e2 = Int32(0)
                        e3 = Int32(0)
                        if pk[k] >= 0:
                            e0, e1, e2, e3 = _ld_volatile_v4(fa)
                    else:
                        e0, e1, e2, e3 = _ld_v4(fa)
                    ev[4 * k + 0] = e0
                    ev[4 * k + 1] = e1
                    ev[4 * k + 2] = e2
                    ev[4 * k + 3] = e3
                if const_expr(self.fc2_lamport and not self.test_no_poll):
                    pending = Int32(1)
                    while pending != 0:
                        pending = Int32(0)
                        for k in cutlass.range_constexpr(topk):
                            if pk[k] >= 0:
                                bad = Int32(0)
                                for i in cutlass.range_constexpr(4):
                                    if ev[4 * k + i] == Int32(_FC2_SENT):
                                        bad = Int32(1)
                                if bad != 0:
                                    fb = fc2_ptr + (
                                        Int64(pk[k]) * Int64(self.fc2_row)
                                        + Int64(vi) * Int64(VEC)
                                    ) * Int64(2)
                                    g0, g1, g2, g3 = _ld_volatile_v4(fb)
                                    ev[4 * k + 0] = g0
                                    ev[4 * k + 1] = g1
                                    ev[4 * k + 2] = g2
                                    ev[4 * k + 3] = g3
                                    pending = Int32(1)
                        if const_expr(self.poll_ns > 0):
                            if pending != 0:
                                _nanosleep(Int32(self.poll_ns))
                acc = cute.make_rmem_tensor((VEC,), Float32)
                for e in cutlass.range_constexpr(VEC):
                    acc[e] = Float32(0.0)
                for k in cutlass.range_constexpr(topk):
                    wf = _lo(wk[k])
                    for i in cutlass.range_constexpr(4):
                        acc[2 * i] = _sel_ge0(
                            pk[k], _fma(wf, _lo(ev[4 * k + i]), acc[2 * i]), acc[2 * i]
                        )
                        acc[2 * i + 1] = _sel_ge0(
                            pk[k],
                            _fma(wf, _hi(ev[4 * k + i]), acc[2 * i + 1]),
                            acc[2 * i + 1],
                        )
                if const_expr(not bt):
                    f0 = _pack_bf16(acc[0], acc[1])
                    f1 = _pack_bf16(acc[2], acc[3])
                    f2 = _pack_bf16(acc[4], acc[5])
                    f3 = _pack_bf16(acc[6], acc[7])
                if const_expr(self.fc2_lamport):
                    for k in cutlass.range_constexpr(topk):
                        if pk[k] >= 0:
                            fr = fc2_ptr + (
                                Int64(pk[k]) * Int64(self.fc2_row)
                                + Int64(vi) * Int64(VEC)
                            ) * Int64(2)
                            _st_v4(
                                fr,
                                Int32(_FC2_SENT),
                                Int32(_FC2_SENT),
                                Int32(_FC2_SENT),
                                Int32(_FC2_SENT),
                            )
            g = _lo(gw)
            s = _sigmoid_triton(g)
            if const_expr(bt and topk > 0):
                x0 = _pack_bf16(_fma(s, _lo(h0), acc[0]), _fma(s, _hi(h0), acc[1]))
                x1 = _pack_bf16(_fma(s, _lo(h1), acc[2]), _fma(s, _hi(h1), acc[3]))
                x2 = _pack_bf16(_fma(s, _lo(h2), acc[4]), _fma(s, _hi(h2), acc[5]))
                x3 = _pack_bf16(_fma(s, _lo(h3), acc[6]), _fma(s, _hi(h3), acc[7]))
            else:
                x0 = _pack_bf16(_fma(s, _lo(h0), _lo(f0)), _fma(s, _hi(h0), _hi(f0)))
                x1 = _pack_bf16(_fma(s, _lo(h1), _lo(f1)), _fma(s, _hi(h1), _hi(f1)))
                x2 = _pack_bf16(_fma(s, _lo(h2), _lo(f2)), _fma(s, _hi(h2), _hi(f2)))
                x3 = _pack_bf16(_fma(s, _lo(h3), _lo(f3)), _fma(s, _hi(h3), _hi(f3)))

        neg0 = Int32(_NEG_ZERO)
        slot_cnt = fl[0]
        blk_bytes: cutlass.Constexpr = nranks * threads * 16  # [W slots] of this CTA
        data_off = Int64(slot_cnt % Int32(3)) * Int64(self.region_bytes) + Int64(
            bid
        ) * Int64(blk_bytes)
        _st_v4(
            mcast + data_off + (Int64(rank) * Int64(threads) + Int64(tidx)) * Int64(16),
            _sanitize(x0),
            _sanitize(x1),
            _sanitize(x2),
            _sanitize(x3),
        )
        poll_base = own + data_off + Int64(tidx) * Int64(16)

        if const_expr(self.use_pdl):
            _sink4(red_addr, r0, r1, r2, r3)
            cute.arch.sync_threads()
            griddepcontrol_launch_dependents()

        if tidx == 0:
            _st_s32(ctl + Int64(bid) * Int64(SLOT_STATE_STRIDE), slot_cnt + Int32(1))
        pq = (slot_cnt + Int32(2)) % Int32(3)
        pbase = (
            own
            + Int64(pq) * Int64(self.region_bytes)
            + Int64(bid) * Int64(blk_bytes)
            + Int64(tidx) * Int64(16)
        )
        for r in cutlass.range_constexpr(nranks):
            _st_v4(pbase + Int64(r * threads * 16), neg0, neg0, neg0, neg0)

        # ---- wait for this vector of every rank ----
        rstride = Int64(threads * 16)
        got = cute.make_rmem_tensor((nranks * 4,), Int32)
        pending = Int32(1)
        while pending != 0:
            pending = Int32(0)
            for r in cutlass.range_constexpr(nranks):
                a, b, cc, d = _ld_volatile_v4(poll_base + Int64(r) * rstride)
                got[r * 4 + 0] = a
                got[r * 4 + 1] = b
                got[r * 4 + 2] = cc
                got[r * 4 + 3] = d
                if a == neg0:
                    pending = Int32(1)
                if b == neg0:
                    pending = Int32(1)
                if cc == neg0:
                    pending = Int32(1)
                if d == neg0:
                    pending = Int32(1)

        # ---- FP32 sum in rank order -> BF16; + residual (BF16) ----
        rw = (r0, r1, r2, r3)
        ww = (w0, w1, w2, w3)
        res = cute.make_rmem_tensor((VEC,), Float32)
        for i in cutlass.range_constexpr(4):
            lo = Float32(0.0)
            hi = Float32(0.0)
            for r in cutlass.range_constexpr(nranks):
                lo = _add(lo, _lo(got[r * 4 + i]))
                hi = _add(hi, _hi(got[r * 4 + i]))
            if const_expr(bt):
                pw = _sanitize(_pack_bf16(_add(lo, _lo(rw[i])), _add(hi, _hi(rw[i]))))
                res[2 * i] = _lo(pw)
                res[2 * i + 1] = _hi(pw)
            elif const_expr(ts_ctas > 0):
                sw = _sanitize(_pack_bf16(lo, hi))
                res[2 * i] = _bf16r(_add(_lo(sw), _lo(rw[i])))
                res[2 * i + 1] = _bf16r(_add(_hi(sw), _hi(rw[i])))
            else:
                res[2 * i] = _bf16r(_add(_bf16r(lo), _lo(rw[i])))
                res[2 * i + 1] = _bf16r(_add(_bf16r(hi), _hi(rw[i])))
        out_off = Int64(gResOut.iterator.toint()) + row_off
        if const_expr(not self.store_wait):
            _st_v4(
                out_off,
                _pack_bf16(res[0], res[1]),
                _pack_bf16(res[2], res[3]),
                _pack_bf16(res[4], res[5]),
                _pack_bf16(res[6], res[7]),
            )

        tsum = Float32(0.0)
        for e in cutlass.range_constexpr(VEC):
            if const_expr(bt):
                tsum = _add(tsum, _mul(res[e], res[e]))
            else:
                tsum = _add(tsum, _bf16r(_mul(res[e], res[e])))
        for k in cutlass.range_constexpr(5):
            tsum = _add(tsum, _shfl_bfly(tsum, Int32(16 >> k)))
        if lane == 0:
            _st_shared_f32(red_addr + Int32(4) * warp, tsum)
        cute.arch.sync_threads()
        if const_expr(ts_ctas > 0 or bt):
            tsn: cutlass.Constexpr = 1 if bt else ts_ctas
            wpc: cutlass.Constexpr = (ctas * num_warps) // tsn
            src = red_addr
            if const_expr(ctas > 1):
                src = csum_addr
                if tidx < ctas * num_warps:
                    _st_async_f32(
                        csum_addr
                        + Int32(4)
                        * (part * Int32(num_warps) + tidx % Int32(num_warps)),
                        mbar_addr,
                        tidx // Int32(num_warps),
                        _ld_shared_f32(red_addr + Int32(4) * (tidx % Int32(num_warps))),
                    )
                ready = _mbar_try_wait(mbar_addr, Int32(0))
                while ready == 0:
                    ready = _mbar_try_wait(mbar_addr, Int32(0))
            if const_expr(tsn == 1):
                total = _tree_sum_smem(src, wpc)
            else:
                total = Float32(0.0)
                for q in cutlass.range_constexpr(tsn):
                    total = _add(total, _tree_sum_smem(src + Int32(4 * q * wpc), wpc))
        else:
            total = _tree_sum_smem(red_addr, num_warps)
            if const_expr(ctas > 1):
                if tidx < ctas:
                    _st_async_f32(
                        csum_addr + Int32(4) * part, mbar_addr, Int32(tidx), total
                    )
                ready = _mbar_try_wait(mbar_addr, Int32(0))
                while ready == 0:
                    ready = _mbar_try_wait(mbar_addr, Int32(0))
                total = Float32(0.0)
                for p in cutlass.range_constexpr(ctas):
                    total = _add(total, _ld_shared_f32(csum_addr + Int32(4 * p)))
        rstd = _rsqrt(
            _add(_div_rn(total, Float32(float(self.hidden))), Float32(self.norm_eps))
        )
        y = cute.make_rmem_tensor((VEC,), Float32)
        if const_expr(ts_ctas > 0 and not bt):
            for i in cutlass.range_constexpr(4):  # the two-shot's (w * x) * rstd
                y[2 * i] = _bf16r(_mul(_mul(_lo(ww[i]), res[2 * i]), rstd))
                y[2 * i + 1] = _bf16r(_mul(_mul(_hi(ww[i]), res[2 * i + 1]), rstd))
        else:
            for i in cutlass.range_constexpr(4):  # (x * rstd) * w
                y[2 * i] = _bf16r(_mul(_mul(res[2 * i], rstd), _lo(ww[i])))
                y[2 * i + 1] = _bf16r(_mul(_mul(res[2 * i + 1], rstd), _hi(ww[i])))
        if const_expr(self.store_wait):
            griddepcontrol_wait()
            _st_v4(
                out_off,
                _pack_bf16(res[0], res[1]),
                _pack_bf16(res[2], res[3]),
                _pack_bf16(res[4], res[5]),
                _pack_bf16(res[6], res[7]),
            )
        if const_expr(self.norm_out):
            _st_v4(
                Int64(gOut.iterator.toint()) + row_off,
                _pack_bf16(y[0], y[1]),
                _pack_bf16(y[2], y[3]),
                _pack_bf16(y[4], y[5]),
                _pack_bf16(y[6], y[7]),
            )
        if const_expr(self.quant):
            if const_expr(bt):
                inv = _div_full_bt(Float32(1.0), gScale[0])
                q01 = _e4m3x2(
                    _clamp448_bt(_mul(y[0], inv)), _clamp448_bt(_mul(y[1], inv))
                )
                q23 = _e4m3x2(
                    _clamp448_bt(_mul(y[2], inv)), _clamp448_bt(_mul(y[3], inv))
                )
                q45 = _e4m3x2(
                    _clamp448_bt(_mul(y[4], inv)), _clamp448_bt(_mul(y[5], inv))
                )
                q67 = _e4m3x2(
                    _clamp448_bt(_mul(y[6], inv)), _clamp448_bt(_mul(y[7], inv))
                )
            else:
                inv = _div_rn(Float32(1.0), gScale[0])
                q01 = _e4m3x2(_mul(y[0], inv), _mul(y[1], inv))
                q23 = _e4m3x2(_mul(y[2], inv), _mul(y[3], inv))
                q45 = _e4m3x2(_mul(y[4], inv), _mul(y[5], inv))
                q67 = _e4m3x2(_mul(y[6], inv), _mul(y[7], inv))
            _st_v2(
                Int64(gQuant.iterator.toint())
                + (Int64(tok) * Int64(vecs) + Int64(vi)) * Int64(8),
                q01 | (q23 << 16),
                q45 | (q67 << 16),
            )

        if const_expr(
            self.use_pdl
            and self.fc2_lamport
            and self.test_store_early
            and not self.test_no_end_wait
        ):
            griddepcontrol_wait()
