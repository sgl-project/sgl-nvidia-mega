# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tensor-core, tensor-map, NVFP4 and routing-scalar helpers.

* tcgen05 issue: the MMAs, scale-factor copies and commits of one ring chunk are one asm
  block (a "burst"), predicated on a leader lane that the caller elects once and keeps
  in a register. Every tcgen05 operation of a role is then issued by the same lane,
  which ``tcgen05.commit`` needs (it tracks the executing thread's operations).
* TMEM loads: the loads and their ``tcgen05.wait::ld`` are one asm block.
* Arithmetic uses fixed PTX forms (routing scores and weights, the SiLU of the routed
  and shared experts, NVFP4 block quantization)."""

from __future__ import annotations

from cutlass import Float32, Int32, Uint32, Uint64
from cutlass.experimental import primitives as nvvm

from .ptx_core import ptx_arith, ptx_op

# ---------------------------------------------------- descriptors (host-side constants)


def idesc_nvf4(m: int, n: int) -> int:
    """Instruction descriptor, kind::mxf4nvf4: E2M1 x E2M1, K-major, UE4M3 scales."""
    return (1 << 7) | (1 << 10) | ((n >> 3) << 17) | ((m >> 4) << 24)


def idesc_bf16(m: int, n: int) -> int:
    """Instruction descriptor, kind::f16: BF16 x BF16 -> FP32, K-major."""
    return (1 << 4) | (1 << 7) | (1 << 10) | ((n >> 3) << 17) | ((m >> 4) << 24)


def smem_desc(addr, lbo: int, sbo: int, layout: int) -> Uint64:
    """Shared-memory matrix descriptor (version 1) of the 32-bit shared address
    ``addr``; ``layout``: 0 no swizzle, 2 128-byte swizzle."""
    static = (
        ((lbo & 0x3FFFF) >> 4) << 16
        | ((sbo & 0x3FFFF) >> 4) << 32
        | 1 << 46
        | layout << 61
    )
    return Uint64((Uint32(addr) & Uint32(0x3FFFF)) >> Uint32(4)) | Uint64(static)


# ---------------------------------------------------------------- leader election


def elect_leader() -> Int32:
    """1 in the one lane ``elect.sync`` picks for the full warp, else 0 (call
    converged)."""
    return Int32(nvvm.elect_sync())


# ---------------------------------------------------------------- tcgen05 bursts

_MMA_NVF4 = "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16"
_MMA_BF16 = "tcgen05.mma.cta_group::1.kind::f16"
_CP_SF = "tcgen05.cp.cta_group::1.32x128b.warpx4"
_COMMIT = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"


def _burst(mmas, cps, block_scaled, idesc, second_group, commit):
    """Asm text of one burst. Operands: $0 lead, $1 d, $2 a, $3 b, $4 sfa, $5 sfb, $6
    first (accumulate flag of MMAs marked "first"), $7 second (gate of ops marked group
    1), $8 commit barrier, $9 cp TMEM base, $10 cp source descriptor. ``mmas``: (d, a,
    b, sfa, sfb, accumulate, group) with byte offsets for a / b (descriptor units added
    as bytes >> 4) and column offsets for d / sfa / sfb; accumulate is "first", 1 or 0.
    ``cps``: (tmem column offset, source byte offset, group)."""
    nm, nc = max(len(mmas), 1), max(len(cps), 1)
    lines = [
        "{",
        ".reg .pred e, e2, pf, pt, p0;",
        f".reg .b32 d<{nm}>, sa<{nm}>, sb<{nm}>;",
        f".reg .b64 a<{nm}>, b<{nm}>, c<{nc}>;",
        f".reg .b32 t<{nc}>, id;",
        f"mov.b32 id, {idesc};",
        "setp.ne.b32 e, $0, 0;",
        "setp.ne.b32 pf, $6, 0;",
        "setp.eq.b32 pt, 0, 0;",
        "setp.ne.b32 p0, 0, 0;",
    ]
    if second_group:
        lines.append("setp.ne.b32 e2, $7, 0; and.pred e2, e2, e;")
    for i, (_t, src, _g) in enumerate(cps):
        lines.append(f"add.u32 t{i}, $9, {_t}; add.s64 c{i}, $10, {src >> 4};")
    for i, (_t, _src, g) in enumerate(cps):
        lines.append(f"@{'e2' if g else 'e'} {_CP_SF} [t{i}], c{i};")
    for i, (d, a, b, sfa, sfb, acc, _g) in enumerate(mmas):
        lines.append(
            f"add.u32 d{i}, $1, {d}; add.s64 a{i}, $2, {a >> 4}; "
            f"add.s64 b{i}, $3, {b >> 4};"
        )
        if block_scaled:
            lines.append(f"add.u32 sa{i}, $4, {sfa}; add.u32 sb{i}, $5, {sfb};")
    for i, (d, a, b, sfa, sfb, acc, g) in enumerate(mmas):
        p = {"first": "pf", 1: "pt", 0: "p0"}[acc]
        gate = "e2" if g else "e"
        if block_scaled:
            lines.append(
                f"@{gate} {_MMA_NVF4} [d{i}], a{i}, b{i}, id, [sa{i}], [sb{i}], {p};"
            )
        else:
            lines.append(f"@{gate} {_MMA_BF16} [d{i}], a{i}, b{i}, id, {p};")
    if commit:
        lines.append(f"@e {_COMMIT} [$8];")
    lines.append("}")
    return "\n".join(lines)


@ptx_op()
def tc_burst(
    lead,
    d,
    a,
    b,
    sfa,
    sfb,
    first,
    second,
    bar,
    cp_base,
    cp_desc,
    *,
    mmas=(),
    cps=(),
    block_scaled=True,
    idesc=0,
    commit=True,
):
    """One chunk's tcgen05 issue (see ``_burst``): scale-factor copies, then MMAs, then
    an optional commit to ``bar``, all by the leader lane."""
    second_group = any(op[-1] for op in mmas) or any(op[-1] for op in cps)
    code = _burst(tuple(mmas), tuple(cps), block_scaled, idesc, second_group, commit)
    ops = (
        Int32(lead),
        Uint32(d),
        Uint64(a),
        Uint64(b),
        Uint32(sfa),
        Uint32(sfb),
        Int32(first),
        Int32(second),
        Uint32(bar),
        Uint32(cp_base),
        Uint64(cp_desc),
    )
    return code, "r,r,l,l,r,r,r,r,r,r,l", ops


@ptx_op()
def tc_commit(lead, bar):
    """Arrive once on ``bar`` when the leader's prior tcgen05 operations complete."""
    code = "{ .reg .pred e; setp.ne.b32 e, $0, 0; @e " + _COMMIT + " [$1]; }"
    return code, "r,r", (Int32(lead), Uint32(bar))


@ptx_op((Float32, Float32, Float32, Float32))
def tmem_load4(taddr):
    """Four FP32 TMEM columns of this lane (32x32b.x4), waited."""
    code = (
        "tcgen05.ld.sync.aligned.32x32b.x4.b32 {$0, $1, $2, $3}, [$4];\n"
        "tcgen05.wait::ld.sync.aligned;"
    )
    return code, "=f,=f,=f,=f,r", (Uint32(taddr),)


@ptx_op(tuple([Float32] * 8))
def _tmem_load4_pair(taddr_a, taddr_b, second):
    code = (
        "{ .reg .pred p; setp.ne.b32 p, $10, 0;\n"
        "mov.f32 $4, 0f00000000; mov.f32 $5, 0f00000000; "
        "mov.f32 $6, 0f00000000; mov.f32 $7, 0f00000000;\n"
        "tcgen05.ld.sync.aligned.32x32b.x4.b32 {$0, $1, $2, $3}, [$8];\n"
        "@p tcgen05.ld.sync.aligned.32x32b.x4.b32 {$4, $5, $6, $7}, [$9];\n"
        "tcgen05.wait::ld.sync.aligned; }"
    )
    return (
        code,
        ",".join(["=f"] * 8 + ["r", "r", "r"]),
        (Uint32(taddr_a), Uint32(taddr_b), Int32(second)),
    )


def tmem_load4_pair(taddr_a, taddr_b, second):
    """Four FP32 columns at ``taddr_a`` and, if ``second`` (warp-uniform), four at
    ``taddr_b`` (else zeros), with one wait."""
    v = _tmem_load4_pair(taddr_a, taddr_b, second)
    return v[:4], v[4:]


@ptx_op(tuple([Float32] * 16))
def _tmem_load4_masked(taddr, mask):
    lines = ["{", ".reg .pred p<4>;", ".reg .b32 m<4>, t<4>;"]
    for q in range(4):
        lines.append(
            f"and.b32 m{q}, $17, {1 << q}; setp.ne.b32 p{q}, m{q}, 0; "
            f"add.u32 t{q}, $16, {4 * q};"
        )
    for q in range(4):
        lines.append(" ".join(f"mov.f32 ${4 * q + j}, 0f00000000;" for j in range(4)))
    for q in range(4):
        regs = ", ".join(f"${4 * q + j}" for j in range(4))
        lines.append(f"@p{q} tcgen05.ld.sync.aligned.32x32b.x4.b32 {{{regs}}}, [t{q}];")
    lines.append("tcgen05.wait::ld.sync.aligned;")
    lines.append("}")
    return (
        "\n".join(lines),
        ",".join(["=f"] * 16 + ["r", "r"]),
        (Uint32(taddr), Uint32(mask)),
    )


def tmem_load4_masked(taddr, mask):
    """For each bit q of the warp-uniform ``mask`` (q < 4): four FP32 columns at
    ``taddr + 4 q`` (zeros for a clear bit), with one wait. Returns four 4-tuples."""
    v = _tmem_load4_masked(taddr, mask)
    return tuple(tuple(v[4 * q : 4 * q + 4]) for q in range(4))


@ptx_op(tuple([Float32] * 48))
def tmem_load4x12(taddr, stride: int = 8):
    """Twelve 32x32b.x4 TMEM loads at column offsets 0, stride, ..., 11 stride, then one
    wait."""
    lines = ["{", ".reg .b32 t<12>;"]
    for t in range(12):
        lines.append(f"add.u32 t{t}, $48, {stride * t};")
    for t in range(12):
        regs = ", ".join(f"${4 * t + j}" for j in range(4))
        lines.append(f"tcgen05.ld.sync.aligned.32x32b.x4.b32 {{{regs}}}, [t{t}];")
    lines.append("tcgen05.wait::ld.sync.aligned;")
    lines.append("}")
    return "\n".join(lines), ",".join(["=f"] * 48 + ["r"]), (Uint32(taddr),)


@ptx_op()
def tmem_alloc(dst, columns: int):
    """Allocate ``columns`` TMEM columns (warp-wide), writing the base address to the
    shared word ``dst``, then give up the allocation permit."""
    code = (
        f"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [$0], {columns};\n"
        "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
    )
    return code, "r", (Int32(dst),)


@ptx_op()
def tmem_free(taddr, columns: int):
    """Free a TMEM allocation (warp-wide)."""
    return (
        f"tcgen05.dealloc.cta_group::1.sync.aligned.b32 $0, {columns};",
        "r",
        (Uint32(taddr),),
    )


def tc_fence_before():
    nvvm.tcgen05_fence("before_thread_sync")


def tc_fence_after():
    nvvm.tcgen05_fence("after_thread_sync")


# ---------------------------------------------------------------- tensor-map copy


@ptx_op()
def tma_load_3d(dst, tmap, x, y, z, bar, policy):
    """3D tile copy global -> shared (``tmap``: generic address of the tensor map)."""
    code = (
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
        ".mbarrier::complete_tx::bytes.L2::cache_hint "
        "[$0], [$1, {$2, $3, $4}], [$5], $6;"
    )
    return (
        code,
        "r,l,r,r,r,r,l",
        (dst, tmap, Int32(x), Int32(y), Int32(z), bar, Uint64(policy)),
    )


# ---------------------------------------------------------------- NVFP4 conversions


@ptx_arith(Uint32)
def e2m1x8(v0, v1, v2, v3, v4, v5, v6, v7):
    """Eight FP32 -> eight E2M1 nibbles (RN, satfinite), element i in bits 4i..4i+3."""
    code = (
        "{ .reg .b8 b0, b1, b2, b3; "
        "cvt.rn.satfinite.e2m1x2.f32 b0, $2, $1; "
        "cvt.rn.satfinite.e2m1x2.f32 b1, $4, $3; "
        "cvt.rn.satfinite.e2m1x2.f32 b2, $6, $5; "
        "cvt.rn.satfinite.e2m1x2.f32 b3, $8, $7; "
        "mov.b32 $0, {b0, b1, b2, b3}; }"
    )
    return (
        code,
        "=r,f,f,f,f,f,f,f,f",
        tuple(Float32(v) for v in (v0, v1, v2, v3, v4, v5, v6, v7)),
    )


@ptx_arith(Uint32)
def e2m1x2(lo, hi):
    """Two FP32 -> one E2M1 pair (RN, satfinite): ``lo`` in bits 0-3, ``hi`` in bits
    4-7."""
    code = (
        "{ .reg .b8 b; .reg .b16 h; .reg .b32 w; "
        "cvt.rn.satfinite.e2m1x2.f32 b, $2, $1; mov.b16 h, {b, b}; "
        "cvt.u32.u16 w, h; and.b32 $0, w, 255; }"
    )
    return code, "=r,f,f", (Float32(lo), Float32(hi))


@ptx_arith((Float32, Uint32))
def nvfp4_block_scale(amax, gs):
    """NVFP4 block scale of a 16-element block with absolute maximum ``amax`` and global
    scale ``gs``: the E4M3 scale byte ``sf = e4m3(gs * (amax / 6))`` and the factor that
    maps the block onto E2M1, ``1 / (float(sf) / gs)`` (0 for a zero scale)."""
    code = (
        "{ .reg .f32 r6, a, s, f, rg, q; .reg .b16 t, u; .reg .b32 h; .reg .pred z;\n"
        "mov.f32 r6, 0f40C00000; rcp.approx.ftz.f32 r6, r6;\n"
        "mul.f32 a, $2, r6; mul.f32 s, $3, a;\n"
        "cvt.rn.satfinite.e4m3x2.f32 t, s, s; and.b16 u, t, 255;\n"
        "cvt.rn.f16x2.e4m3x2 h, u; cvt.u16.u32 t, h; cvt.f32.f16 f, t;\n"
        "setp.eq.f32 z, f, 0f00000000;\n"
        "rcp.approx.ftz.f32 rg, $3; mul.f32 q, f, rg; rcp.approx.ftz.f32 q, q;\n"
        "selp.f32 $0, 0f00000000, q, z; cvt.u32.u16 $1, u; }"
    )
    return code, "=f,=r,f,f", (Float32(amax), Float32(gs))


# ---------------------------------------------------------------- activations


@ptx_arith(Float32)
def silu_fast(g):
    """SiLU, g / (1 + exp(-g)), with the fast exponential (ex2.approx of -g log2 e)
    and an IEEE divide: the routed experts' SiLU."""
    code = (
        "{ .reg .f32 t, e, d; mul.f32 t, $1, 0fBFB8AA3B; ex2.approx.f32 e, t; "
        "add.f32 d, e, 0f3F800000; div.rn.f32 $0, $1, d; }"
    )
    return code, "=f,f", (Float32(g),)


@ptx_arith(Float32)
def silu_accurate(g):
    """SiLU, g / (1 + exp(-g)), with the accurate single-precision exponential (range
    reduction by 2^k, two-term log2 e) and an IEEE divide: the shared expert's SiLU."""
    code = (
        "{ .reg .f32 t, s, k, kb, r, e, d; .reg .b32 sh;\n"
        "fma.rn.f32 t, $1, 0fBBBB989D, 0f3F000000; cvt.sat.f32.f32 s, t;\n"
        "fma.rm.f32 k, s, 0f437C0000, 0f4B400001; "
        "add.f32 kb, k, 0fCB40007F; neg.f32 kb, kb;\n"
        "fma.rn.f32 r, $1, 0fBFB8AA3B, kb; fma.rn.f32 r, $1, 0fB2A57060, r;\n"
        "mov.b32 sh, k; shl.b32 sh, sh, 23; ex2.approx.ftz.f32 e, r;\n"
        "mov.b32 d, sh; fma.rn.f32 d, e, d, 0f3F800000; div.rn.f32 $0, $1, d; }"
    )
    return code, "=f,f", (Float32(g),)


@ptx_arith(Float32)
def abs_f32(a):
    return "abs.f32 $0, $1;", "=f,f", (Float32(a),)


@ptx_arith(Float32)
def max_f32(a, b):
    """``max.f32`` (a NaN operand yields the other operand)."""
    return "max.f32 $0, $1, $2;", "=f,f,f", (Float32(a), Float32(b))


# --------------------------------------------------------------- routing scalars (pure)


@ptx_arith(Float32)
def routing_sigmoid(x):
    """0.5 * tanh(0.5 * x) + 0.5 with the approximate tanh, flush-to-zero."""
    code = (
        "{ .reg .f32 h,t; mul.ftz.f32 h,$1,0f3F000000; tanh.approx.f32 t,h; "
        "fma.rn.ftz.f32 $0,t,0f3F000000,0f3F000000; }"
    )
    return code, "=f,f", (Float32(x),)


@ptx_arith(Float32)
def routing_add(a, b):
    return "add.ftz.f32 $0,$1,$2;", "=f,f,f", (Float32(a), Float32(b))


@ptx_arith(Uint32)
def routing_key(score, bias):
    """Order-preserving u32 of ``score + bias`` (flush-to-zero add): NaN ranks as -inf,
    -0 as +0, so a larger key is a better candidate."""
    code = (
        "{ .reg .f32 s,t; .reg .b32 u,nu,pu; .reg .pred pn,pz,ps; "
        "add.ftz.f32 s,$1,$2; setp.nan.f32 pn,s,s; selp.f32 t,0fFF800000,s,pn; "
        "setp.eq.f32 pz,t,0f00000000; selp.b32 u,0,t,pz; not.b32 nu,u; "
        "or.b32 pu,u,0x80000000; setp.lt.s32 ps,u,0; selp.b32 $0,nu,pu,ps; }"
    )
    return code, "=r,f,f", (Float32(score), Float32(bias))


@ptx_arith(Float32)
def routing_weight(scale, score, total):
    """BF16-rounded ``scale * score / (total + 1e-20)`` (approximate divide,
    flush-to-zero), returned as FP32."""
    code = (
        "{ .reg .f32 n,d,q; .reg .b16 h; mul.ftz.f32 n,$1,$2; "
        "add.ftz.f32 d,$3,0f1E3CE508; "
        "div.approx.ftz.f32 q,n,d; cvt.rn.bf16.f32 h,q; cvt.f32.bf16 $0,h; }"
    )
    return code, "=f,f,f,f", (Float32(scale), Float32(score), Float32(total))
