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
) -> dict[str, tuple]:
    """The attention units for row capacities ``caps`` and sparse-MLA warp counts
    ``warps``. ``numerics`` is the engine's ``PrologueNumerics`` value table and
    ``index_q_rope_form`` its index-query RoPE layout; ``index_scale_ue8m0x4`` selects
    the prologue's index-K scale layout of four UE8M0 bytes.
    ``qkv_a = (n_rows, n_main)`` adds the skinny q_a / kv_a (+ indexer key) projection
    at 8 rows on ``grid`` CTAs (default: the current device's multiprocessor count).
    ``prologue_end`` adds ``prologue_end.<cap>``: the prologue that triggers its
    dependents after its last output store. ``qkv_a_shared = n_rows`` adds
    ``qkv_a_shared.8``: the skinny projection of the q_a / kv_a rows alone (the layers
    without their own indexer), at 8 rows on ``grid`` CTAs; ``qkv_a_shared_dn`` adds
    its epilogue-scaled form ``qkv_a_shared_dn.8`` (deferred input norm)."""
    from .indexer_q_proj import IndexerQProj
    from .mla_prologue import MlaPrologue
    from .mla_uv_proj import MlaUvProj
    from .qkv_a_proj_small_m import QkvAProjSmallM, QkvAProjSmallMDn
    from .sparse_mla_decode import SparseMlaDecode

    args = example_args()
    specs: dict[str, tuple] = {}
    caps = sorted(set(caps))
    for cap in caps:
        specs[f"uv.{cap}"] = (MlaUvProj(cap), {"max_rows": cap}, args["uv"])
        key = {
            "max_rows": cap,
            **dataclasses.asdict(numerics),
            "rope_cache_fp32": rope_cache_fp32,
            "index_rope_cache_fp32": index_rope_cache_fp32,
        }
        if index_scale_ue8m0x4:
            key["index_scale_ue8m0x4"] = True
        specs[f"prologue.{cap}"] = (
            MlaPrologue(
                cap,
                numerics,
                rope_cache_fp32,
                index_rope_cache_fp32,
                index_scale_ue8m0x4,
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
                ),
                {**key, "trigger_at_end": True},
                args["prologue"],
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
