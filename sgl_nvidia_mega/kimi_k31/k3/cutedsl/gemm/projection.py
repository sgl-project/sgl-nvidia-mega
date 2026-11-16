# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Split-K projection on the FlashInfer CuTe core vendored in ``splitk.py`` (with an
evict_first weight TMA; policies map padded rows to (M tile, N tile, split K,
stages); attention owns live slot masks)."""

import logging
import os
import re
from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import torch
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
from cutlass.cutlass_dsl import CompileCallable

from ..common.host import architecture
from ..common.host import stream as current_stream
from ..common.host import tensor_maps, to_cute
from .splitk import SplitKGemm, SplitKTactic

logger = logging.getLogger(__name__)

# FX-4 reads the host IR of the compile it checks, so projections compile through the DSL's own entry point (the
# object cute.experimental.compile is as shipped), never through a wrapper of that attribute: with K31MK_KERNEL_CACHE
# set, common/kernel_cache.py routes this module's cute.experimental.compile calls through its store, and a hit is an
# AOT-loaded function without host IR (ir_module None) that FX-4 refused at every boot.
# The object launched is the object whose IR was checked; the cost is one compile per tactic per process.
_DSL_COMPILE = CompileCallable()


def matrix_view(tensor, *, output=False):
    """Keep packed row pitches and FlashInfer's singleton-safe dynamic major axis."""
    rows, cols = tensor.shape
    shape = (cols, rows, 1) if output else (rows, cols, 1)
    stride = (1, tensor.stride(0), 0) if output else (tensor.stride(0), 1, 0)
    result = to_cute(tensor.as_strided(shape, stride))
    return result.mark_layout_dynamic(leading_dim=0 if output else 1)


# Packed attention input projections (TP8: MLA K3.1 [3776, 7168], K3 [3648, 7168]; KDA
# [6288, 7168]): (M tile, N tile, split K, stages) per batch of rows, from GB300 sweeps.
def _gb300_tactic(kind, batch):
    if kind == "mla":
        return {1: (64, 8, 1, 12), 64: (64, 64, 2, 5), 32: (64, 32, 2, 8), 16: (64, 16, 2, 10)}.get(
            batch, (64, 8, 4, 8)
        )
    return {64: (64, 64, 1, 6), 32: (128, 32, 2, 5), 16: (128, 16, 2, 6)}.get(batch, (128, 8, 2, 6))


# VR200 (SM107) overrides where a sweep (cold L2, same gate)
# measured faster than the GB300 tactic by a clear margin: MLA B1-B16 and B64; KDA B32 and B64.
# ``K31MK_PROJ_SM107_RETUNE=0`` keeps GB300's tactics on SM107.
# B128 (Breq 16, TP16 per-rank shapes, 8 tactics per kind, gate passed by all): (64, 64, 2, 6) is best for both
# kinds (cold L2), ahead of the next best and far ahead of GB300's fallback at 128 rows (N tile 8).
SM107_TACTICS = {
    "mla": {**{b: (128, 8, 4, 6) for b in range(1, 9)}, 16: (128, 16, 4, 5), 64: (64, 64, 2, 6), 128: (64, 64, 2, 6)},
    "kda": {32: (64, 32, 2, 9), 64: (64, 64, 2, 6), 128: (64, 64, 2, 6)},
}
# ``K31MK_PROJ_SM107_ENT=1`` (default off): these entries replace or extend SM107_TACTICS (with the retune on). Measured at
# TP16 per-rank shapes only.
# 16 rows (Breq 2, TP16 shapes, 1 GPU, in our AttnRes -> projection -> KdaSplit chain, cold, per
# layer vs the served tactic): (64, 16, 2, 11) measured faster for both kinds (KDA: GB300's (128, 16, 2, 6) otherwise, same split,
# bitwise; MLA: (128, 16, 4, 5) otherwise, split 4 -> 2 reorders the FP32 sum); also faster isolated.
# 256 rows (Breq 32, isolated cold L2, 32 tactics gated): (128, 64, 2, 4) measured much faster for KDA (GB300's
# (128, 8, 2, 6), 1664 CTAs, otherwise; same split, bitwise) and MLA ((64, 8, 4, 8), 6016 CTAs, otherwise).
SM107_TACTICS_ENT = {
    "mla": {16: (64, 16, 2, 11), 256: (128, 64, 2, 4)},
    "kda": {16: (64, 16, 2, 11), 256: (128, 64, 2, 4)},
}

# ``K31MK_PROJ_SM107_TP8=1`` (default off): entries for the TP8 per-rank projections only, keyed on the packed
# width (KDA 6288 = 12 heads, MLA 3776); other widths (TP16: 3216 / 3008) keep the tables above. At TP8 the N tiles of the
# TP16-tuned 128- and 256-row tactics double, so (64, 64, 2, 6) and (128, 64, 2, 4) run two waves on the target GPU;
# these one-wave forms need the large-SMEM mode (``K31MK_SPLITK_OVERSIZED=1``, as served).
# In the AttnRes -> projection -> KdaSplit chain (TP8 shapes, 1 GPU, cold, us per layer vs the served tactic, served fences):
# 128 rows (Breq 16): KDA (64, 128, 2, 6) measured faster (same split, bitwise); MLA (64, 64, 1, 10) faster.
# 256 rows (Breq 32), vs (128, 64, 2, 4): KDA (64, 128, 1, 6) and MLA (128, 64, 1, 6) measured faster.
# Split 2 -> 1 reorders the FP32 sum (one K chain instead of two halves added).
SM107_TP8_WIDTHS = {"kda": 6288, "mla": 3776}
SM107_TACTICS_TP8 = {
    "kda": {128: (64, 128, 2, 6), 256: (64, 128, 1, 6)},
    "mla": {128: (64, 64, 1, 10), 256: (128, 64, 1, 6)},
}

# TM64 (``K31MK_PROJ_TMEM64=1``, default off): SplitKTmemGemm64 (splitk_tmem64.py) at 64 rows on SM107,
# 5-field tactics (128, N tile, split, x slots, tensor-memory A tiles). TP16 per-rank shapes: KDA 3216 rows = 26 M tiles in
# clusters of 6 (one wave on the target GPU), MLA 3008 = 24 tiles in clusters of 7 (also one wave).
# Sums the split partials in rank order (an FP32 accumulation reorder of split 2).
# Rows: only 64 (Breq 8). In the AttnRes -> projection -> KdaSplit chain (TP16 shapes, 1 GPU, cold, us per layer vs the
# served tactic) it pays only when AttnRes runs with a 100 % shared-memory carve-out (its CTAs then leave room for these
# ~170-200 KB CTAs before the PDL wait): measured faster for KDA and MLA at 64 rows; slower (KDA) with the default carve-out, and slower
# at 16 / 32 rows either way. So K31MK_PROJ_TMEM64=1 is meant to run with that AttnRes carve-out.
SM107_TMEM64 = {"kda": (128, 64, 6, 14, 14), "mla": (128, 64, 7, 16, 14)}
TMEM64_ROWS = range(64, 65)

# TM8 (``K31MK_PROJ_TMEM8=1``, default off): SplitKTmemGemm8 (splitk_tmem8.py) for the MLA in-projection at 8 rows
# (Breq 1, the c=1 verify) on SM107 at the TP16 widths: 2880 rows (the TARGET_VERIFY build: k3 pitch, 23 M tiles), 3040 (the same
# + index-K + head gate under K31MK_DSA_QF) and 3008 (the decode build's pitch); clusters of 7; 7-field tactics
# (128, 8, split, ring stages, tensor-memory A tiles, A tiles requested before the PDL wait = all of them, owners).
# K 7168 / 64 / 7 = 16 tiles per rank, 15 staged in tensor memory. Its CTAs use 227,328 B, so they land beside an AttnRes
# CTA only when the AttnRes runs at a 100 % shared-memory carve-out: K31MK_PROJ_TMEM8=1 is meant to run with
# K31MK_ATTNRES_CARVEOUT=100. Same MMAs and rank-order sum as SplitKGemm (128, 8, 7, *): bitwise equal to it (unit test).
SM107_TMEM8 = {"mla": (128, 8, 7, 12, 15, 16, 4)}
TMEM8_WIDTH = {"mla": (2880, 3008, 3040)}


# Widths above 8 rows with a measured tactic. Any other width above 8 (the K3.1 verify widths 24 / 40 / 48 / 56 of
# Breq 3 / 5 / 6 / 7) takes the tactic of the next one: the tactic its batch ran with when padded to that graph.
# K31MK_PROJ_TACTICS entries stay exact-width (looked up before this round-up).
TACTIC_WIDTHS = (16, 32, 64, 128)


@cache
def tactic_overrides(spec):
    """Parse ``K31MK_PROJ_TACTICS`` ("kda:8=64.8.2.12,mla:1=128.8.4.6") into {(kind,
    rows): tactic}; default off. TP16 A/B knob (more CTAs on the 52-CTA KDA stream).
    Another M tile at the same split is tested bitwise-equal; another split reorders the sum.
    A 5-field entry (m.n.split.slots.tmem_tiles) selects SplitKTmemGemm64 (TM64), a 7-field entry
    (m.n.split.stages.tmem_tiles.early_tiles.owners) SplitKTmemGemm8 (TM8), a 6-field entry
    (128.n.split.stages.owners.cta_k) SplitKRsGemm (RS, splitk_rs.py)."""
    table = {}
    for item in filter(None, spec.split(",")):
        key, value = item.split("=")
        kind, rows = key.split(":")
        tactic = tuple(int(v) for v in value.split("."))
        if kind not in ("mla", "kda") or len(tactic) not in (4, 5, 6, 7):
            raise ValueError(
                f"K31MK_PROJ_TACTICS entry {item!r}: want kind:rows=m.n.split.stages[.tmem_tiles], a 6-field RS or a 7-field TM8 entry"
            )
        # K-42: fix A modes 0 / 3 trigger from the weight warp after its bar_empty wait on tile ``stages``, which follows
        # the MMA warp's first commit and so its TMEM alloc; that order needs more K tiles per split (K 7168 / 128 = 56
        # in these projections) than stages, as every table tactic has.
        if len(tactic) == 4 and 56 // max(tactic[2], 1) <= tactic[3]:
            raise ValueError(f"K31MK_PROJ_TACTICS entry {item!r}: 56 // split K tiles per split must exceed the stages (K-42)")
        table[kind, int(rows)] = tactic
    return table


def projection_tactic(kind, batch, device, width=None):
    """The split-K tactic of the packed ``kind`` ("mla" / "kda") projection at ``batch``; ``width`` = the packed weight's
    rows (the projection's N), which selects the TP8 entries (``K31MK_PROJ_SM107_TP8``); None keeps the shared tables."""
    override = tactic_overrides(os.environ.get("K31MK_PROJ_TACTICS", "")).get((kind, batch))
    if override is not None:
        return override
    sm107 = torch.cuda.get_device_capability(device) == (10, 7)
    if sm107 and os.environ.get("K31MK_PROJ_TMEM8", "0") == "1" and batch == 8 and kind in SM107_TMEM8:
        if width is None or width in TMEM8_WIDTH[kind]:  # other widths (TP8) keep their tables
            return SM107_TMEM8[kind]
    if sm107 and os.environ.get("K31MK_PROJ_TMEM64", "0") == "1" and batch in TMEM64_ROWS:
        return SM107_TMEM64[kind]
    # Padded widths (after TM8 / TM64, which stay exact-width: 8 and 64 rows only): see TACTIC_WIDTHS.
    if 8 < batch < max(TACTIC_WIDTHS) and batch not in TACTIC_WIDTHS:
        batch = next(w for w in TACTIC_WIDTHS if w >= batch)
    retune = os.environ.get("K31MK_PROJ_SM107_RETUNE", "1") != "0"
    if retune and sm107:
        table = SM107_TACTICS[kind]
        if os.environ.get("K31MK_PROJ_SM107_ENT", "0") == "1":
            table = {**table, **SM107_TACTICS_ENT[kind]}
        if os.environ.get("K31MK_PROJ_SM107_TP8", "0") == "1" and width == SM107_TP8_WIDTHS[kind]:
            if os.environ.get("K31MK_SPLITK_OVERSIZED", "0") != "1":
                raise ValueError("K31MK_PROJ_SM107_TP8=1 needs K31MK_SPLITK_OVERSIZED=1 (its tactics exceed the normal SMEM allocation)")
            table = {**table, **SM107_TACTICS_TP8[kind]}
        tactic = table.get(batch)
        if tactic is not None:
            return tactic
    return _gb300_tactic(kind, batch)


def splitk_smem_estimate(tactic):
    """SplitKGemm's SMEM use for (M tile, N tile, split K, stages): the A + B rings, the split-K mailbox and the
    barriers (equal to the DSL's count for 128.8.7.6 = 233,608 and 128.16.4.6 = 245,896 B).
    Used only to pick the launch allocation; FX-4 still checks the DSL's own count against it after compile."""
    m, n, split, stages = tactic
    return stages * (m + n) * 256 + (split - 1) * m * n * 4 + 136


def oversized_for(tactic):
    """OV (``K31MK_SPLITK_OVERSIZED=1``, default off): a tactic whose SMEM exceeds the normal
    allocation launches in the large-SMEM mode (a larger per-CTA allocation, smaller L1; cluster CTAs one per SM), so the
    weight ring holds more stages before the PDL wait. Off (default) every tactic keeps the normal allocation and an
    over-cap tactic fails FX-4 as before."""
    if os.environ.get("K31MK_SPLITK_OVERSIZED", "0") != "1":
        return False
    return splitk_smem_estimate(tactic) > cutlass_utils.get_smem_capacity_in_bytes("sm_100")


def set_oversized(code, device):
    """CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE (17) = ALLOW_OVERSIZED (3) on the compiled projection's function before its
    first launch (same call as moe/planned.set_oversized; the driver then allows a larger per-CTA allocation)."""
    from ..common.host import resolve_kernel

    _, function = resolve_kernel(code, device)
    attr = getattr(cuda.CUfunction_attribute, "CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE", None)
    attr = attr if attr is not None else cuda.CUfunction_attribute(17)
    (status,) = cuda.cuFuncSetAttribute(function, attr, 3)
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"large-SMEM mode: {status}")


@cache
def compile_projection(tactic, output_dtype, arch, fenced_trigger=0, proxy_fences=False, act_fuse=False, oversized=False):
    """Code-only cache; weights, descriptors and activation pointers stay private.
    ``fenced_trigger``: SplitKGemm's fix A mode (0 = shipped; 1-3: the trigger follows a
    PDL wait behind a release fence; see splitk.py), for a projection whose dependent
    reads before its own wait. ``oversized``: launch with the oversized allocation (OV); the
    caller sets the function's large-SMEM mode on its device (``set_oversized``)."""

    def fake(leading, dtype=cutlass.BFloat16):
        shape = tuple(cute.sym_int() for _ in range(3))
        stride = (1, cute.sym_int(64), 0) if leading == 0 else (cute.sym_int(64), 1, 0)
        return make_fake_tensor(dtype, shape, stride, assumed_align=16)

    y = fake(0, cutlass.Float32 if output_dtype == torch.float32 else cutlass.BFloat16)
    split = SplitKTactic(*tactic)
    kernel = SplitKGemm(
        tactic=split, fenced_trigger=fenced_trigger, proxy_fences=proxy_fences, act_fuse=act_fuse, oversized=bool(oversized)
    )
    args = (fake(1), fake(1), y, y, cutlass.Int64(0), make_fake_stream())
    compiled = _DSL_COMPILE(kernel, *args, options=f"--gpu-arch={arch}")
    overflow = smem_overflow(compiled)
    if overflow is not None:  # FX-4: fail closed before any launch (the launch would overrun SMEM and fault)
        raise ValueError(
            f"SplitKGemm tactic {tuple(tactic)} uses more shared memory than its launch allocates "
            f"(allocated, used = {overflow} B, the DSL's own count); pick a tactic with fewer stages"
        )
    return compiled


@cache
def compile_projection_tmem64(tactic, k, output_dtype, arch, fenced_trigger=2):
    """TM64: ``SplitKTmemGemm64`` for ``tactic`` = (128, N tile, split, x slots, tensor-memory A tiles) and reduction
    length ``k``. ``fenced_trigger``: 2 = warp 3 waits, fences and triggers (served fix A mode 2); 0 = the weight warp
    triggers before any wait (the original form; a measurement control). FX-4 checks its SMEM use as for SplitKGemm."""
    from .splitk_tmem64 import N_TILES, SplitKTmemGemm64, acc_columns

    def fake(leading, dtype=cutlass.BFloat16):
        shape = tuple(cute.sym_int() for _ in range(3))
        stride = (1, cute.sym_int(64), 0) if leading == 0 else (cute.sym_int(64), 1, 0)
        return make_fake_tensor(dtype, shape, stride, assumed_align=16)

    m, n, split, slots, tmem_tiles = tactic
    if m != 128 or n not in N_TILES or tmem_tiles != (512 - acc_columns(n)) // 32:
        raise ValueError(f"TM64 tactic {tuple(tactic)}: want (128, N in {N_TILES}, split, slots, {(512 - acc_columns(n)) // 32})")
    y = fake(0, cutlass.Float32 if output_dtype == torch.float32 else cutlass.BFloat16)
    kernel = SplitKTmemGemm64(tactic=SplitKTactic(m, n, split, slots), k=k, fenced_trigger=fenced_trigger)
    args = (fake(1), fake(1), y, y, cutlass.Int64(0), make_fake_stream())
    compiled = _DSL_COMPILE(kernel, *args, options=f"--gpu-arch={arch}")
    overflow = smem_overflow(compiled)
    if overflow is not None:
        raise ValueError(f"TM64 tactic {tuple(tactic)} uses more shared memory than its launch allocates ({overflow} B)")
    return compiled


@cache
def compile_projection_rs(tactic, k, output_dtype, arch, fenced_trigger=0, proxy_fences=False):
    """RS (default off): ``SplitKRsGemm`` for ``tactic`` = (128, N tile, split, stages, owners,
    cta_k) and reduction length ``k``: SplitKGemm's pipeline with a reduce-scatter epilogue, always in the oversized
    SMEM mode (the caller sets it, ``set_oversized``). BF16 output only. ``fenced_trigger`` as SplitKGemm.
    FX-4 checks its SMEM use."""
    from .splitk_rs import SplitKRsGemm, rs_check

    def fake(leading, dtype=cutlass.BFloat16):
        shape = tuple(cute.sym_int() for _ in range(3))
        stride = (1, cute.sym_int(64), 0) if leading == 0 else (cute.sym_int(64), 1, 0)
        return make_fake_tensor(dtype, shape, stride, assumed_align=16)

    if output_dtype != torch.bfloat16:
        raise ValueError(f"RS tactic {tuple(tactic)}: BF16 output only, got {output_dtype}")
    rs_check(tactic, k)
    m, n, split, stages, owners, cta_k = tactic
    y = fake(0)
    kernel = SplitKRsGemm(
        tactic=SplitKTactic(m, n, split, stages), k=k, owners=owners, cta_k=cta_k, fenced_trigger=fenced_trigger,
        proxy_fences=proxy_fences,
    )  # fmt: skip
    args = (fake(1), fake(1), y, y, cutlass.Int64(0), make_fake_stream())
    compiled = _DSL_COMPILE(kernel, *args, options=f"--gpu-arch={arch}")
    overflow = smem_overflow(compiled)
    if overflow is not None:
        raise ValueError(f"RS tactic {tuple(tactic)} uses more shared memory than its launch allocates ({overflow} B)")
    return compiled


def tmem8_check(tactic, k):
    """Raise unless ``tactic`` is a valid TM8 entry (128, 8, split, stages, tmem_tiles, early_tiles, owners) for
    reduction length ``k``: whole 64-wide K tiles per rank, every tile requested before the PDL wait (early_tiles =
    the rank's tile count), at least as many tiles as ring stages, the staged tiles within tensor memory."""
    m, n, split, stages, tmem_tiles, early, owners = tactic
    tiles = k // 64 // max(split, 1)
    ok = m == 128 and n == 8 and 1 <= split <= 8 and tiles * 64 * split == k and early == tiles
    ok = ok and 2 <= stages <= tiles and 0 < tmem_tiles <= tiles and 8 + 32 * tmem_tiles <= 512
    ok = ok and 1 <= owners <= split and 128 % (4 * owners) == 0
    if not ok:
        raise ValueError(f"TM8 tactic {tuple(tactic)} is invalid for K {k}")


@cache
def compile_projection_tmem8(tactic, k, output_dtype, arch, fenced_trigger=2):
    """TM8: ``SplitKTmemGemm8`` for a 7-field ``tactic`` and reduction length ``k``. ``fenced_trigger``: 2 = warp 3
    waits, fences and triggers; 0 = the weight warp triggers once every A tile is requested. FX-4 checks its SMEM use."""
    from .splitk_tmem8 import SplitKTmemGemm8

    def fake(leading, dtype=cutlass.BFloat16):
        shape = tuple(cute.sym_int() for _ in range(3))
        stride = (1, cute.sym_int(64), 0) if leading == 0 else (cute.sym_int(64), 1, 0)
        return make_fake_tensor(dtype, shape, stride, assumed_align=16)

    tmem8_check(tactic, k)
    m, n, split, stages, tmem_tiles, _early, owners = tactic
    y = fake(0, cutlass.Float32 if output_dtype == torch.float32 else cutlass.BFloat16)
    kernel = SplitKTmemGemm8(
        tactic=SplitKTactic(m, n, split, stages), k=k, tmem_tiles=tmem_tiles, owners=owners, fenced_trigger=fenced_trigger
    )
    args = (fake(1), fake(1), y, y, cutlass.Int64(0), make_fake_stream())
    compiled = _DSL_COMPILE(kernel, *args, options=f"--gpu-arch={arch}")
    overflow = smem_overflow(compiled)
    if overflow is not None:
        raise ValueError(f"TM8 tactic {tuple(tactic)} uses more shared memory than its launch allocates ({overflow} B)")
    return compiled


def partial_view(partials, rows):
    """The kernel's view of ``partials`` [8, >= rows, M] FP32 (rows token-major, M contiguous): (M, rows, 8) M-major."""
    slices, _, m = partials.shape
    view = partials.as_strided((m, rows, slices), (1, partials.stride(1), partials.stride(0)))
    return to_cute(view).mark_layout_dynamic(leading_dim=0)


_SMEM_CHECK = re.compile(
    r'llvm\.mlir\.global internal constant @(\w+)\("[^"]*may exceed available memory set in kernel launch'
)


def smem_overflow(compiled):
    """FX-4. SplitKGemm launches with a fixed SMEM allocation
    (the arch capacity) and the CuTe DSL emits, in the host launch code, its own exact comparison of the kernel's
    SMEM use against that allocation (a printf when use > allocation). After constant folding the compiled host IR
    still references that message only when the kernel does not fit. Returns (allocated, used) bytes then (the
    printf's operands, or None when they cannot be read), else None. A DSL build without the message cannot be
    checked, nor can an object without host IR (an AOT-loaded one, e.g. a K31MK_KERNEL_CACHE hit): raise rather
    than launch unchecked."""
    module = getattr(compiled, "ir_module", None)
    if module is None:
        raise RuntimeError("FX-4: this SplitKGemm has no host IR to read (an AOT-loaded object); refusing an unchecked SplitKGemm")
    ir = str(module)
    found = _SMEM_CHECK.search(ir)
    if found is None:
        raise RuntimeError("FX-4: this CuTe DSL build has no launch SMEM check to read; refusing an unchecked SplitKGemm")
    ref = re.search(rf"= llvm\.mlir\.addressof @{found.group(1)}\b", ir)
    if ref is None:
        return None
    start, end = ir.rfind("llvm.func", 0, ref.start()), ir.find("\n  }", ref.start())
    func = ir[start:end]  # the host function that launches the kernel
    call = re.search(r"llvm\.call @printf\(%\w+, (%\w+), (%\w+)\)", ir[ref.start() : end])
    values = []
    for name in call.groups() if call else ():
        const = re.search(rf"{re.escape(name)} = llvm\.mlir\.constant\((\d+) : i\d+\)", func)
        values.append(int(const.group(1)) if const else None)
    return tuple(values) if len(values) == 2 else (None, None)


def weight_map(weight, rows):
    """Device TMA map of ``weight`` [N, K] (row pitch kept): box {64, rows}, SW128."""
    types = cuda.CUtensorMapDataType
    dtype = {
        torch.bfloat16: types.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        torch.float16: types.CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    }[weight.dtype]
    swizzle = cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
    l2 = cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B
    n, k = weight.shape
    pitch = weight.stride(0) * weight.element_size()
    spec = (weight.data_ptr(), dtype, (k, n), (pitch,), (64, rows), swizzle, l2)
    return tensor_maps([spec], weight.device)


class PreparedProjection:
    """Split-K projection with a per-batch tactic table and caller-owned outputs."""

    def __init__(
        self, weight, policies, *, output_dtype=torch.bfloat16, fenced_trigger=0, proxy_fences=False, act_fuse=False,
        oversized=None,
    ):  # fmt: skip
        # ``oversized``: None = OV knob decides per tactic (``oversized_for``); True / False forces the allocation
        # (measurement control: the mode's own cost at an unchanged tactic).
        self.oversized = oversized
        if weight.dim() != 2 or weight.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(
                f"projection weight must be a BF16/FP16 [N, K] matrix, got "
                f"{tuple(weight.shape)} {weight.dtype}"
            )
        pitch = weight.stride(0) * weight.element_size()
        if weight.stride(1) != 1 or pitch % 16 or weight.data_ptr() % 16:
            raise ValueError(
                "projection weight rows need unit stride and a 16 B TMA pitch"
            )
        if output_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"unsupported projection output dtype {output_dtype}")
        # act_fuse: weight = merged [gate; up] [2I, K]; output = SiLU(gate) * up [rows, I] in BF16.
        self.act_fuse = bool(act_fuse)
        if self.act_fuse:
            if output_dtype != torch.bfloat16 or weight.shape[0] % 2:
                raise ValueError("act_fuse needs a BF16 output and an even merged gate_up row count")
            for batch, (m, *_rest) in dict(policies).items():
                if (weight.shape[0] // 2) % (m // 2):
                    raise ValueError(f"act_fuse tactic M tile {m} at B{batch} does not tile I={weight.shape[0] // 2}")
        k_tiles = weight.shape[1] // 128
        for batch, tactic in dict(policies).items():
            if len(tactic) == 5:  # TM64: uneven K slices are allowed (rank r < (K / 64) % split takes one more tile)
                if self.act_fuse or weight.shape[1] % 64 or not 2 <= tactic[2] <= 8:
                    raise ValueError(f"TM64 tactic {tactic} at B{batch} is invalid for K {weight.shape[1]} (act_fuse {act_fuse})")
                continue
            if len(tactic) == 7:  # TM8
                if self.act_fuse:
                    raise ValueError(f"TM8 tactic {tactic} at B{batch} does not take act_fuse")
                tmem8_check(tactic, weight.shape[1])
                continue
            if len(tactic) == 6:  # RS: BF16 out, whole 8-row output words, uneven K slices allowed (rs_check)
                from .splitk_rs import rs_check

                if self.act_fuse or output_dtype != torch.bfloat16 or weight.shape[0] % 8:
                    raise ValueError(f"RS tactic {tactic} at B{batch}: needs BF16 output, no act_fuse, rows % 8 == 0")
                rs_check(tactic, weight.shape[1])
                continue
            m, n, split, stages = tactic
            # SplitKGemm gives each split K // 128 // split tiles: a split that does not divide
            # the K tiles would drop the remainder silently.
            if m not in (64, 128) or n % 8 or not 1 <= split <= 8 or k_tiles % split or stages < 2:
                raise ValueError(f"projection tactic {(m, n, split, stages)} at B{batch} is invalid for K {weight.shape[1]}")
        self.weight, self.device = weight, weight.device
        self.output_dtype, self.policies = output_dtype, dict(policies)
        self.fenced_trigger = int(fenced_trigger)
        self.proxy_fences = proxy_fences
        self.weight_view = matrix_view(weight)
        self.kernels, self.extra, self.maps = {}, {}, {}

    def prepare(self):
        arch, dtype = architecture(self.device), self.output_dtype
        with torch.accelerator.device_index(self.device.index):
            for batch, tactic in self.policies.items():
                if len(tactic) == 5:  # TM64; fix A: any fenced mode (1-3) -> its warp-3 mode 2, 0 -> the original early trigger
                    fenced = 2 if self.fenced_trigger else 0
                    self.kernels[batch] = compile_projection_tmem64(tactic, self.weight.shape[1], dtype, arch, fenced)
                    self.extra[batch] = self.weight_map(tactic[0])
                    continue
                if len(tactic) == 6:  # RS: SplitKGemm's fix A modes as is; always the large-SMEM mode
                    code = compile_projection_rs(
                        tactic, self.weight.shape[1], dtype, arch, self.fenced_trigger, self.proxy_fences
                    )  # fmt: skip
                    set_oversized(code, self.device)
                    self.kernels[batch] = code
                    self.extra[batch] = self.weight_map(tactic[0])
                    # One line per prepared RS projection: shows the tactic engaged at this width.
                    logger.info(
                        "RS in-projection engaged: batch %d, rows %d, K %d, tactic %s, trigger mode %d",
                        batch, self.weight.shape[0], self.weight.shape[1], tactic, self.fenced_trigger,
                    )
                    continue
                if len(tactic) == 7:  # TM8; fix A: any fenced mode (1-3) -> its warp-3 mode 2, 0 -> the weight warp
                    fenced = 2 if self.fenced_trigger else 0
                    self.kernels[batch] = compile_projection_tmem8(tactic, self.weight.shape[1], dtype, arch, fenced)
                    self.extra[batch] = self.weight_map(tactic[0])
                    # One line per prepared TM8 projection: shows the knob engaged at this width.
                    logger.info(
                        "TM8 in-projection engaged: batch %d, rows %d, K %d, tactic %s, trigger mode %d",
                        batch, self.weight.shape[0], self.weight.shape[1], tactic, fenced,
                    )
                    continue
                over = oversized_for(tactic) if self.oversized is None else bool(self.oversized)
                if over:
                    code = compile_projection(
                        tactic, dtype, arch, self.fenced_trigger, self.proxy_fences, self.act_fuse, True
                    )  # fmt: skip
                    set_oversized(code, self.device)
                else:
                    code = compile_projection(tactic, dtype, arch, self.fenced_trigger, self.proxy_fences, self.act_fuse)
                self.kernels[batch] = code
                # act_fuse: half-height boxes (the gate half and the up half of one A tile).
                self.extra[batch] = self.weight_map(tactic[0] // 2 if self.act_fuse else tactic[0])
        return self

    def weight_map(self, rows):
        """The kernel's extra argument: A's map for ``rows``-row tiles."""
        if rows not in self.maps:
            self.maps[rows] = weight_map(self.weight, rows)
        return (cutlass.Int64(self.maps[rows].data_ptr()),)

    def run(self, x, out):
        y = matrix_view(out, output=True)
        b, stream = x.shape[0], current_stream(self.device)
        if self.act_fuse and (out.shape != (b, self.weight.shape[0] // 2) or out.stride(1) != 1):
            raise ValueError(f"act_fuse output {tuple(out.shape)} stride {out.stride()}: want ({b}, I) with unit inner stride")
        if b not in self.kernels or x.shape[1] != self.weight.shape[1]:
            raise ValueError(
                f"projection input {tuple(x.shape)}: prepared batches "
                f"{sorted(self.kernels)}, K {self.weight.shape[1]}"
            )
        code, extra = self.kernels[b], self.extra[b]
        code(self.weight_view, matrix_view(x), y, y, *extra, stream)
        return out
