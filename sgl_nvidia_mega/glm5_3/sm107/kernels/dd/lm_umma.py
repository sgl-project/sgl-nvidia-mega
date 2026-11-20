# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import cute as cute_ir
from cutlass._mlir.dialects import cute_nvgpu as nvgpu_ir
from cutlass._mlir.dialects import nvvm
from cutlass.cutlass_dsl import dsl_user_op

HIDDEN = 6144
VOCAB_SHARD = 9680
KEY_STRIDE = 1
TILE_M = 64
TILES = (VOCAB_SHARD + TILE_M - 1) // TILE_M
K_TILES = HIDDEN // 128
STAGES = 12
A_BYTES = TILE_M * 256
B_BYTES = 8 * 256
STAGE_BYTES = A_BYTES + B_BYTES
REGION_BYTES = STAGES * STAGE_BYTES
TMEM_COLS = 32
_LMU = [int(v) for v in "0,0,1".split(",")]
TMA_HINT = _LMU[0] != 0
MMA_FENCE = len(_LMU) < 2 or _LMU[1] != 0
ELECT = len(_LMU) < 3 or _LMU[2] != 0
TMA3D = len(_LMU) > 3 and _LMU[3] != 0
IDESC = (1 << 4) | (1 << 7) | (1 << 10) | ((8 >> 3) << 17) | ((TILE_M >> 4) << 24)
SDESC = (1 << 16) | (64 << 32) | (1 << 46) | (2 << 61)


@dsl_user_op
def tma_2d_elect(
    atom: cute.CopyAtom,
    dst: cutlass.Uint32,
    x: cutlass.Int32,
    y: cutlass.Int32,
    bar: cutlass.Uint32,
    *,
    loc=None,
    ip=None,
):
    """Issued by one elected lane, so the operands stay warp-uniform and no per-lane loop is generated around it."""
    executable = nvgpu_ir.atom_make_exec_tma(atom._trait.value, loc=loc, ip=ip)
    ptr_type = cute_ir.PtrType.get(
        nvgpu_ir.TmaDescriptorTiledType.get(), cutlass.AddressSpace.generic, 64
    )
    pointer = nvgpu_ir.get_tma_desc_addr(ptr_type, executable, loc=loc, ip=ip)
    descriptor = as_ptr(0, pointer.toint().ir_value(loc=loc, ip=ip), loc, ip)
    destination = as_ptr(7, dst, loc, ip)
    coordinates = [x.ir_value(loc=loc, ip=ip), y.ir_value(loc=loc, ip=ip)]
    with cute.arch.elect_one(loc=loc, ip=ip):
        nvvm.cp_async_bulk_tensor_shared_cluster_global(
            destination, descriptor, coordinates, as_ptr(3, bar, loc, None), [], loc=loc
        )


@dsl_user_op
def expect_tx_elect(
    bar: cutlass.Uint32, nbytes: cutlass.Constexpr, *, loc=None, ip=None
):
    with cute.arch.elect_one(loc=loc, ip=ip):
        nvvm.mbarrier_arrive_expect_tx(
            as_ptr(3, bar, loc, None), cutlass.Int32(nbytes).ir_value(loc=loc), loc=loc
        )


@dsl_user_op
def umma_bf16_elect(
    d: cutlass.Uint32,
    a: cutlass.Uint32,
    b: cutlass.Uint32,
    accumulate: cutlass.Uint32,
    *,
    loc=None,
    ip=None,
):
    common = cutlass.Uint64(SDESC)
    da = common | ((a & cutlass.Uint32(0x3FFFF)) >> 4).to(cutlass.Uint64)
    db = common | ((b & cutlass.Uint32(0x3FFFF)) >> 4).to(cutlass.Uint64)
    idesc = cutlass.Uint32(IDESC)
    enable = (accumulate != cutlass.Uint32(0)).ir_value(loc=loc, ip=ip)
    tmem = as_ptr(6, d, loc, ip)
    with cute.arch.elect_one(loc=loc, ip=ip):
        nvvm.tcgen05_mma(
            nvvm.Tcgen05MMAKind.F16,
            nvvm.CTAGroupKind.CTA_1,
            tmem,
            da.ir_value(loc=loc),
            db.ir_value(loc=loc),
            idesc.ir_value(loc=loc),
            enable,
            loc=loc,
        )


@dsl_user_op
def commit_elect(bar: cutlass.Uint32, *, loc=None, ip=None):
    with cute.arch.elect_one(loc=loc, ip=ip):
        nvvm.tcgen05_commit(
            as_ptr(3, bar, loc, None), group=nvvm.CTAGroupKind.CTA_1, loc=loc
        )


@dsl_user_op
def evict_last(*, loc=None, ip=None):
    return cutlass.Uint64(0x14F0000000000000)


@dsl_user_op
def tmem_dealloc32(address: cutlass.Uint32, *, loc=None, ip=None):
    nvvm.tcgen05_dealloc(
        as_ptr(6, address, loc, ip),
        cutlass.Int32(TMEM_COLS).ir_value(loc=loc, ip=ip),
        group=nvvm.CTAGroupKind.CTA_1,
        loc=loc,
        ip=ip,
    )


from .common import as_ptr
