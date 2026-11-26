# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Inline-PTX, memory, synchronization and math helpers of the decode kernels."""

import math
from functools import partial, wraps

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


def ptx(assembly, constraints, operands=(), result=None, *, effects, loc=None, ip=None):
    """One verbatim asm block; a tuple ``result`` (even a 1-tuple) returns a tuple."""
    if isinstance(result, tuple) and len(result) == 1:
        args = (assembly, constraints, operands, result[0])
        return (ptx(*args, effects=effects, loc=loc, ip=ip),)
    if result is None:
        ir_type = None
    elif isinstance(result, tuple):
        ir_type = llvm.StructType.get_literal([dtype.mlir_type for dtype in result])
    else:
        ir_type = result.mlir_type
    value = llvm.inline_asm(
        ir_type,
        [operand.ir_value(loc=loc, ip=ip) for operand in operands],
        assembly,
        constraints,
        has_side_effects=effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    if result is None:
        return None
    if isinstance(result, tuple):
        return tuple(
            dtype(llvm.extractvalue(dtype.mlir_type, value, [i], loc=loc, ip=ip))
            for i, dtype in enumerate(result)
        )
    return result(value)


def asm(result=None, *, pure=False):
    """``f(*args) -> (code, constraints, operands[, result, pure])`` as one inline-PTX
    op. Pointer operands pass their address: 32-bit for "r" (shared), 64-bit for "l"."""

    def wrap(fn):
        @dsl_user_op
        @wraps(fn)
        def op(*args, loc=None, ip=None, **kwargs):
            code, constraints, operands, *rest = fn(*args, **kwargs)
            res, fx = (*rest, pure)[:2] if rest else (result, pure)
            kinds = [c for c in constraints.split(",") if c[:1] not in ("=", "~", "")]
            values = []
            for kind, v in zip(kinds, operands):
                if hasattr(v, "toint"):
                    v = v.toint(loc=loc, ip=ip)
                    v = Int32(v) if kind == "r" else v
                values.append(v)
            return ptx(code, constraints, values, res, effects=not fx, loc=loc, ip=ip)

        return op

    return wrap


def _vector(count, first=0):
    """(".vN", "{$first,...}") of an N-word access; ("", "$first") for one word."""
    names = ",".join(f"${first + i}" for i in range(count))
    return (f".v{count}", "{" + names + "}") if count > 1 else ("", names)


# A remote Lamport wait samples the device timer only once per this many retry
# passes.  The timer is started lazily, after the first read proves that a peer's
# publication is still pending, so the completed fast path pays no timer cost.
DEADLINE_POLL_INTERVAL = 1024


@asm(Int64)
def global_timer_ns():
    return "mov.u64 $0, %globaltimer;", "=l", ()


@asm()
def fail_stop():
    """Poison the CUDA context on a missed cross-GPU publication deadline."""
    code = "{ .reg .u64 address; mov.u64 address, 0; "
    code += "st.volatile.global.u32 [address], 0; trap; }"
    return code, "", ()


@cute.jit
def wait_deadline_start(deadline_ns: cutlass.Constexpr[int]):
    """Start a remote-wait deadline only after an initial read found pending data."""
    started = Int64(0)
    if cutlass.const_expr(deadline_ns > 0):
        started = global_timer_ns()
    return started


@cute.jit
def wait_deadline_check(started, polls, deadline_ns: cutlass.Constexpr[int]):
    """Advance one retry and fail-stop after a sampled deadline violation."""
    polls += Int32(1)
    if cutlass.const_expr(deadline_ns > 0):
        if (polls & Int32(DEADLINE_POLL_INTERVAL - 1)) == 0:
            if global_timer_ns() - started > Int64(deadline_ns):
                fail_stop()
    return polls


@asm(Float32, pure=True)
def add_rn(a, b):
    return "add.rn.f32 $0,$1,$2;", "=f,f,f", (Float32(a), Float32(b))


@asm(Float32, pure=True)
def fma(a, b, c, *, ftz=False):
    return f"fma.rn{'.ftz' if ftz else ''}.f32 $0, $1, $2, $3;", "=f,f,f,f", (a, b, c)


def dot(a, b, acc=None, **kwargs):
    """``fma`` chain over ``zip(a, b)`` from ``acc``, or from ``a[0] * b[0]``."""
    for x, y in zip(a, b):
        acc = x * y if acc is None else fma(x, y, acc, **kwargs)
    return acc


def bits_float(bits):
    return Float32(llvm.bitcast(T.f32(), Uint32(bits).ir_value()))


def float_bits(value: Float32) -> Int32:
    return Int32(llvm.bitcast(T.i32(), value.ir_value()))


def bf16(word, high=False):
    """FP32 of the low (or high) BF16 half of ``word``."""
    return bits_float(word & 0xFFFF0000 if high else word << 16)


@asm(Uint32, pure=True)
def pack_pair(low, high, *, policy="plain"):
    """BF16x2 of (low, high); ``policy=None`` is the bare conversion."""
    code = "cvt.rn.bf16x2.f32 $0,$2,$1; "
    if policy == "stage":
        # NaN halves become canonical, so no word is the 0xffffffff staging sentinel.
        code += (
            "and.b32 bits,$0,0x7fff; setp.gt.u32 p,bits,0x7f80; "
            "@p and.b32 $0,$0,0xffff0000; @p or.b32 $0,$0,0x7fc0; "
            "and.b32 bits,$0,0x7fff0000; setp.gt.u32 p,bits,0x7f800000; "
            "@p and.b32 $0,$0,0xffff; @p or.b32 $0,$0,0x7fc00000; "
        )
    if policy == "latent":
        # Negative BF16 zero is the downstream mailbox's unpublished sentinel.
        code += (
            "and.b32 bits,$0,0xffff; setp.eq.u32 p,bits,0x8000; "
            "@p and.b32 $0,$0,0xffff0000; shr.u32 bits,$0,16; "
            "setp.eq.u32 p,bits,0x8000; @p and.b32 $0,$0,0xffff; "
        )
    if policy is not None:
        code = "{ .reg .b32 bits; .reg .pred p; " + code + "}"
    return code, "=r,f,f", (Float32(low), Float32(high))


bf16x2 = partial(pack_pair, policy=None)


def published(words, ready=True, *, sentinel=0x80000000):
    """``ready`` and no word is the mailbox ``sentinel``."""
    for word in words:
        ready &= word != Uint32(sentinel)
    return ready


@cute.jit
def warp_sum(value: Float32, lanes: cutlass.Constexpr[int] = 32):
    for shift in cutlass.range_constexpr(lanes.bit_length() - 2, -1, -1):
        value += cute.arch.shuffle_sync_bfly(value, 1 << shift)
    return value


def at(tensor, *coord):
    """Pointer to ``tensor[coord]``."""
    return tensor.iterator + cute.crd2idx(coord, tensor.layout)


def zeros(shape, dtype):
    tensor = cute.make_rmem_tensor(shape, dtype)
    tensor.fill(0)
    return tensor


def alloc(allocator, dtype, shape, align=16):
    """Row-major ``shape`` tensor from a ``SmemAllocator``."""
    stride = None
    if isinstance(shape, tuple):
        stride = tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))
    layout = cute.make_layout(shape, stride=stride)
    return allocator.allocate_tensor(dtype, layout, align)


@asm()
def load_words(
    pointer,
    *,
    count=2,
    shared=False,
    evict_first=False,
    mutable=False,
    volatile=False,
    scope=None,
    cache_policy="cg",
    word_type="b32",
    result_type=Uint32,
):
    """``count`` words (a tuple); ``scope`` is a relaxed load at that scope.
    ``mutable`` (a side-effecting load) for data written during the step (activations,
    tables, refills). Only static weights use ``evict_first``."""
    space = "global" + (f".{cache_policy}" if cache_policy else "")
    if shared:
        space = "shared"
    if volatile:
        space = "volatile.global"
    if scope:
        space = f"relaxed.{scope}.global"
    policy = hint = tail = ""
    if evict_first:
        assert not shared
        space = "global.nc.L1::no_allocate"
        policy = ".reg .b64 pol; createpolicy.fractional.L2::evict_first.b64 pol, 1.0; "
        hint, tail = ".L2::cache_hint", ", pol"
    vector, registers = _vector(count)
    address = f"[${count}]{tail}"
    code = f"{{ {policy}ld.{space}{hint}{vector}.{word_type} {registers}, {address}; }}"
    constraints = ["=f" if result_type is Float32 else "=r"] * count
    constraints += ["r" if shared else "l"] + (["~{memory}"] if scope else [])
    # Shared data is written by other threads between reads: never pure.
    pure = not (shared or mutable or volatile or scope)
    return code, ",".join(constraints), (pointer,), (result_type,) * count, pure


smem_words = partial(load_words, count=4, shared=True, mutable=True)


@asm()
def store_words(pointer, words, *, shared=False, scope=None):
    """``scope`` is a relaxed global store at that scope."""
    vector, registers = _vector(len(words), 1)
    if shared:
        # Other threads read it: a memory side effect, like the global stores.
        space, constraints = (
            f"shared{vector}.b32",
            "r" + ",r" * len(words) + ",~{memory}",
        )
    else:
        space = (
            f"relaxed.{scope}.global{vector}.u32" if scope else f"global{vector}.b32"
        )
        constraints = "l" + ",r" * len(words) + ",~{memory}"
        words = tuple(Uint32(v) for v in words)
    return f"st.{space} [$0], {registers};", constraints, (pointer, *words)


store_relaxed = partial(store_words, scope="sys")


@asm()
def push_float(pointer, value, barrier, peer, when=None):
    """DSMEM store of one FP32 ``value`` (or a 4-tuple) completing on ``barrier``
    (``when``: predicated on it)."""
    values = tuple(map(Float32, value if isinstance(value, tuple) else (value,)))
    vector, registers = _vector(len(values), 1)
    b, p, w = (f"${len(values) + i}" for i in (1, 2, 3))
    code = f"{{ .reg .b32 dst,bar; mapa.shared::cluster.u32 dst,$0,{p}; "
    code += f"mapa.shared::cluster.u32 bar,{b},{p}; "
    guard = "" if when is None else f".reg .pred w; setp.ne.b32 w,{w},0; @w "
    code += (
        guard + f"st.async.shared::cluster.mbarrier::complete_tx::bytes{vector}.b32 "
    )
    code += f"[dst],{registers},[bar]; }}"
    constraints = "r" + ",f" * len(values) + ",r,r" + ",r" * (when is not None)
    operands = (pointer, *values, barrier, Int32(peer))
    operands += () if when is None else (Int32(when),)
    return code, constraints + ",~{memory}", operands


# The word a checked bulk copy looks for in word 0 of every 16-byte chunk it reads.
CHECKED_SENTINEL = 0x80000000


@asm()
def copy_g2s_checked(dst, src, size, barrier):
    """Checked bulk copy (an SM107 instruction: the assembler takes it for sm_107a
    only): expect ``size`` bytes (static or dynamic; a multiple of 16, ``dst`` 16-byte
    aligned) on the layout::v1 ``barrier`` (``init_barrier_v1``) and copy them. The
    phase the copy completes reports whether word 0 of any 16-byte chunk it read was
    ``CHECKED_SENTINEL`` (``try_wait_checked``)."""
    operands = [dst, src, barrier]
    if not isinstance(size, int):
        operands.append(Int32(size))
        size = "$3"
    code = f"mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [$2], {size}; "
    code += "cp.async.bulk.relaxed.gpu.shared::cta.global.mbarrier::complete_tx::bytes"
    code += f".mbarrier::report::validity::per_16bytes::{CHECKED_SENTINEL:08x}.b128 "
    code += f"[$0], [$1], {size}, [$2];"
    constraints = ",".join(["r", "l"] + ["r"] * (len(operands) - 2))
    return "{ " + code + " }", constraints + ",~{memory}", operands


@asm(Int32)
def try_wait(barrier, phase=0, *, test=False, scope="cluster"):
    """Acquire parity probe at ``scope`` ("cluster", or "cta" when every arrival is
    CTA-local); ``test`` is the non-blocking probe of phase 0."""
    op, parity = ("test_wait", "0") if test else ("try_wait", "$2")
    code = f"{{ .reg .pred p; mbarrier.{op}.parity.acquire.{scope}.shared::cta.b64 "
    code += f"p,[$1],{parity}; selp.u32 $0,1,0,p; }}"
    if test:
        return code, "=r,r", (barrier,)
    return code, "=r,r,r,~{memory}", (barrier, Int32(phase))


@cute.jit
def wait(barrier, phase=0, test: cutlass.Constexpr[bool] = False, scope="cluster"):
    done = try_wait(barrier, phase, test=test, scope=scope)
    while done == 0:
        done = try_wait(barrier, phase, test=test, scope=scope)


@asm()
def init_barrier_v1(barrier, count=1):
    """``mbarrier.init`` of a layout::v1 barrier (SM107 only), the kind a checked bulk
    copy (``copy_g2s_checked``) reports through; ``count`` is static."""
    code = f"mbarrier.init.layout::v1.shared.b64 [$0], {count};"
    return code, "r,~{memory}", (barrier,)


@asm((Int32, Int32))
def try_wait_checked(barrier, parity):
    """Acquire probe of the ``parity`` phase of a layout::v1 barrier (SM107 only):
    (done, report); ``report`` (read it once done) is nonzero when a checked copy of
    that phase saw the sentinel. The parity alternates per completed phase, as on an
    ordinary barrier."""
    code = "{ .reg .pred done,report; mbarrier.try_wait.parity.phase_type::primary"
    code += ".acquire.cta.shared::cta.b64 done|report,[$2],$3; "
    code += "selp.b32 $0,1,0,done; selp.b32 $1,1,0,report; }"
    return code, "=r,=r,r,r,~{memory}", (barrier, Int32(parity))


@cute.jit
def wait_checked(barrier, parity):
    """Wait for the ``parity`` phase of a layout::v1 barrier (SM107 only); returns its
    report."""
    done, report = try_wait_checked(barrier, parity)
    while done == 0:
        done, report = try_wait_checked(barrier, parity)
    return report


def group_barrier(group):
    cute.arch.barrier(barrier_id=1 + 3 * group, number_of_threads=128)


def launch(kernel, args, grid, block, cluster=None, pdl=True, **options):
    """Launch kernel(args[:-1]) on stream args[-1]; an int dimension means (n, 1, 1)."""
    dim = lambda d: d if isinstance(d, tuple) else (d, 1, 1)
    if cluster is not None:
        options["cluster"] = dim(cluster)
    kernel(args[:-1]).launch(
        grid=dim(grid), block=dim(block), stream=args[-1], use_pdl=pdl, **options
    )


@cute.jit
def scatter_reduce(values, lane, count: cutlass.Constexpr[int]):
    """Warp-reduce ``count`` values; lane groups end up owning one value each."""
    selected = Int32(0)
    for stage in cutlass.range_constexpr(5):
        offset = 16 >> stage
        remaining = max(count >> stage, 1)
        if cutlass.const_expr(remaining > 1):
            half = remaining // 2
            upper = (lane & offset) != 0
            for i in cutlass.range_constexpr(half):
                send = values[i] if upper else values[i + half]
                keep = values[i + half] if upper else values[i]
                values[i] = keep + cute.arch.shuffle_sync_bfly(send, offset)
            if upper:
                selected += half
        else:
            values[0] += cute.arch.shuffle_sync_bfly(values[0], offset)
    return values[0], selected
