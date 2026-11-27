# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compile units of the attention family, with one source of example arguments.

``compile_specs`` returns ``{name: (kernel, constexprs, example_args)}`` for every
attention kernel form the decode / verify glue compiles: the kernel object, the
constexpr options that identify the compiled form (the compile-cache key), and typed
example arguments for ``cute.compile`` (zero pointers, unit sizes, a ``CUstream``).
The engine glue compiles exactly these units, and tests compile and reload them, so
the two never hold separate argument tuples.

Names: ``uv.<cap>``, ``prologue.<cap>``, ``index_q.<cap>`` (``cap`` = max rows of the
form, 8 or 16), ``sparse.<warps>`` (8 or 16 warps) and ``qkv_a.8`` (only when
``qkv_a`` is given; the skinny projection has an 8-row form only).
"""

import dataclasses
from collections.abc import Iterable


def example_args() -> dict[str, tuple]:
    """Typed example arguments per kernel kind for ``cute.compile`` (pointers 0,
    sizes 1, scalars 1e-5)."""
    import cuda.bindings.driver as cuda
    from cutlass import Float32, Int32, Int64

    z, f, s = Int64(0), Float32(1e-5), cuda.CUstream(0)
    one = Int32(1)
    return {
        "uv": (z,) * 6 + (one,) + (z,) * 4 + (Int32(0),) + (z,) * 4 + (Int32(0), s),
        "sparse": (z,) * 5 + (one, f, f, z, z, z, Int32(0), Int32(6), z, z, s),
        "index_q": (z,) * 8 + (f, one, z, z, z, z, s),
        "qkv_a": (z, z, z, one, z, z, z, z, s),
        "qkv_a_dn": (z, z, z, one, z, z, z, z, z, f, s),
        "front": (z,) * 7 + (one, z, f, z, z, f) + (z,) * 5 + (z, z, s),
        "prologue": (z,) * 11
        + (one, one)
        + (z,) * 4
        + (f, f)
        + (z,) * 6
        + (f,)
        + (z,) * 13
        + (Int32(0), z, z, s),
    }


def compile_specs(
    caps: Iterable[int] = (8, 16),
    *,
    numerics,
    index_q_rope_form: str,
    rope_cache_fp32: bool = True,
    index_rope_cache_fp32: bool = True,
    index_scale_ue8m0x4: bool = False,
    warps: Iterable[int] = (8, 16),
    qkv_a: tuple[int, int] | None = None,
    grid: int | None = None,
    prologue_end: bool = False,
    qkv_a_shared: int | None = None,
    qkv_a_shared_dn: bool = False,
    prologue16_overlap: bool = False,
    prologue16_kv_split: bool = False,
    prologue8_overlap: bool = False,
    prologue_push: bool = False,
    prologue_f32x2: bool = False,
    idx_front: bool = False,
    prologue16_row_norm: bool = False,
    prologue16_early_arm: bool = False,
    prologue8_early_arm: bool = False,
    uv_values_per_cta: int = 16,
) -> dict[str, tuple]:
    """The attention units for row capacities ``caps`` and sparse-MLA warp counts
    ``warps``. ``numerics`` is the engine's ``PrologueNumerics`` value table and
    ``index_q_rope_form`` its index-query RoPE layout; ``index_scale_ue8m0x4`` selects
    the prologue's index-K scale layout of four UE8M0 bytes.
    ``qkv_a = (n_rows, n_main)`` adds the skinny q_a / kv_a (+ indexer key) projection
    at 8 rows on ``grid`` CTAs (default: the current device's multiprocessor count).
    ``prologue_end`` adds ``prologue_end.<cap>``: the prologue that triggers its
    dependents after its last output store.
    ``prologue16_overlap`` builds the 16-row prologues in the overlapped issue order
    (``MlaPrologue(overlap16=True)``: same outputs). ``qkv_a_shared = n_rows`` adds
    ``qkv_a_shared.8``: the skinny projection of the q_a / kv_a rows alone (the layers
    without their own indexer), at 8 rows on ``grid`` CTAs; ``qkv_a_shared_dn`` adds
    its epilogue-scaled form ``qkv_a_shared_dn.8`` (deferred input norm).
    ``prologue16_kv_split`` builds ``prologue[_end].16`` with the kv cluster's rows
    8..15 on their own CTAs (same outputs).
    ``prologue8_overlap`` builds the 8-row prologues in the overlapped issue order
    (``MlaPrologue(overlap8=True)``: same outputs).
    ``prologue_push`` / ``prologue_f32x2`` add the pushed q_nope hand-off / the packed
    branch-free normalize to the overlapped forms (``MlaPrologue(push=True)`` /
    ``f32x2=True``; same outputs).
    ``idx_front`` adds ``front.<cap>``: the indexer's share of the prologue as its own
    launch (``IndexerFront``: the q_a norm out and the index-K rows, the prologue's
    arithmetic), for the indexer-layer prologue split.
    ``prologue16_row_norm`` (with ``prologue16_overlap`` and the strided q_a sums) builds
    the 16-row prologues with the q rows summed and normalized by the warp that loads
    them (``MlaPrologue(row_norm=True)``); ``prologue16_early_arm`` (with
    ``prologue16_overlap`` and ``prologue_push``) arms their QN barrier before the weight
    copies (``MlaPrologue(early_arm=True)``). ``prologue8_early_arm`` arms the 8-row overlapped
    pushed form's QN barrier before its weight copies (``early_arm8=True``; same
    outputs; only with ``prologue8_overlap`` and ``prologue_push``)."""
    from .indexer_front import IndexerFront
    from .indexer_q_proj import IndexerQProj
    from .mla_prologue import MlaPrologue
    from .mla_uv_proj import MlaUvProj
    from .qkv_a_proj_small_m import QkvAProjSmallM, QkvAProjSmallMDn
    from .sparse_mla_decode import SparseMlaDecode

    args = example_args()
    specs: dict[str, tuple] = {}
    caps = sorted(set(caps))
    for cap in caps:
        vpc = uv_values_per_cta if cap == 8 else min(uv_values_per_cta, 32)
        uv_key = {"max_rows": cap}
        if vpc != 16:
            uv_key["values_per_cta"] = vpc
        specs[f"uv.{cap}"] = (
            MlaUvProj(cap, values_per_cta=vpc),
            uv_key,
            args["uv"],
        )
        key = {
            "max_rows": cap,
            **dataclasses.asdict(numerics),
            "rope_cache_fp32": rope_cache_fp32,
            "index_rope_cache_fp32": index_rope_cache_fp32,
        }
        if index_scale_ue8m0x4:
            key["index_scale_ue8m0x4"] = True
        overlap16 = prologue16_overlap and cap == 16
        if overlap16:
            key["overlap16"] = True
        kvs = bool(prologue16_kv_split and cap == 16)
        if kvs:
            key["kv_split"] = True
        overlap8 = prologue8_overlap and cap == 8
        if overlap8:
            key["overlap8"] = True
        ov = overlap8 or overlap16
        push, f32x2 = bool(prologue_push and ov), bool(prologue_f32x2 and ov)
        if push:
            key["push"] = True
        if f32x2:
            key["f32x2"] = True
        row_norm = bool(
            prologue16_row_norm
            and overlap16
            and numerics.norm_order == "strided_8x256"
        )
        early_arm = bool(prologue16_early_arm and overlap16 and push)
        arm8 = bool(prologue8_early_arm and overlap8 and push)
        if row_norm:
            key["row_norm"] = True
        if early_arm:
            key["early_arm"] = True
        if arm8:
            key["early_arm8"] = True
        specs[f"prologue.{cap}"] = (
            MlaPrologue(
                cap,
                numerics,
                rope_cache_fp32,
                index_rope_cache_fp32,
                index_scale_ue8m0x4,
                overlap16=overlap16,
                kv_split=kvs,
                overlap8=overlap8,
                push=push,
                f32x2=f32x2,
                row_norm=row_norm,
                early_arm=early_arm,
                early_arm8=arm8,
            ),
            key,
            args["prologue"],
        )
        if prologue_end:
            specs[f"prologue_end.{cap}"] = (
                MlaPrologue(
                    cap,
                    numerics,
                    rope_cache_fp32,
                    index_rope_cache_fp32,
                    index_scale_ue8m0x4,
                    trigger_at_end=True,
                    overlap16=overlap16,
                    kv_split=kvs,
                    overlap8=overlap8,
                    push=push,
                    f32x2=f32x2,
                    row_norm=row_norm,
                    early_arm=early_arm,
                    early_arm8=arm8,
                ),
                {**key, "trigger_at_end": True},
                args["prologue"],
            )
        if idx_front:
            fkey = {
                "max_rows": cap,
                **dataclasses.asdict(numerics),
                "index_rope_cache_fp32": index_rope_cache_fp32,
            }
            if index_scale_ue8m0x4:
                fkey["index_scale_ue8m0x4"] = True
            specs[f"front.{cap}"] = (
                IndexerFront(cap, numerics, index_rope_cache_fp32, index_scale_ue8m0x4),
                fkey,
                args["front"],
            )
        key = {
            "max_rows": cap,
            "rope_form": index_q_rope_form,
            "rope_cache_fp32": index_rope_cache_fp32,
        }
        specs[f"index_q.{cap}"] = (
            IndexerQProj(cap, index_q_rope_form, index_rope_cache_fp32),
            key,
            args["index_q"],
        )
    for w in sorted(set(warps)):
        key = {"warps": w, "logical_ids": False, "early_trigger": True}
        specs[f"sparse.{w}"] = (SparseMlaDecode(w, False, True), key, args["sparse"])
    if (qkv_a is not None or qkv_a_shared is not None) and 8 in caps:
        if grid is None:
            import torch

            grid = torch.cuda.get_device_properties(
                torch.accelerator.current_device_index()
            ).multi_processor_count
    if qkv_a is not None and 8 in caps:
        n_rows, n_main = qkv_a
        kernel = QkvAProjSmallM(n_rows, n_main, grid)
        key = {"n_rows": n_rows, "n_main": n_main, "grid": grid, "slots": kernel.slots}
        specs["qkv_a.8"] = (kernel, key, args["qkv_a"])
    if qkv_a_shared is not None and 8 in caps:
        kernel = QkvAProjSmallM(qkv_a_shared, qkv_a_shared, grid)
        key = {
            "n_rows": qkv_a_shared,
            "n_main": qkv_a_shared,
            "grid": grid,
            "slots": kernel.slots,
        }
        specs["qkv_a_shared.8"] = (kernel, key, args["qkv_a"])
        if qkv_a_shared_dn:
            kernel = QkvAProjSmallMDn(qkv_a_shared, qkv_a_shared, grid)
            specs["qkv_a_shared_dn.8"] = (
                kernel,
                {**key, "epilogue_scale": True},
                args["qkv_a_dn"],
            )
    return specs
