# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compile units of the draft kernels: for each unit, the kernel object, its compile-time constants (the compile-cache
key next to the class name) and example launch arguments. These are the compile units the draft glue and the export /
reload test of the persistent compile cache use.

Each caller asks only for the units it ships (the eh shard GEMV rows, the top-1, and the embedding + eh norm front with
its value table).
"""

import cuda.bindings.driver as cuda
from cutlass import Float32, Int32, Int64, Uint32, Uint64

HIDDEN = 6144


def compile_specs(
    *,
    eh_rows=(),
    top1_rows=(),
    eh_norm=None,
    stream=None,
):
    """``{name: (kernel, constexprs, example_args)}`` for the requested units.

    ``eh_rows``: row counts of ``EhProjShard``; ``top1_rows``: ``max_rows`` of
    ``DistributedArgmax``; ``eh_norm``: the ``EmbedEhNorm`` value table (None: no
    front). The example arguments are typed placeholders for ``cute.compile``.
    """
    s = stream if stream is not None else cuda.CUstream(0)
    z = Int64(0)
    out = {}
    if eh_rows:
        from .mtp_eh_proj_shard import EhProjShard

        for rows in eh_rows:
            out[f"eh_proj_shard_{rows}"] = (
                EhProjShard(rows),
                {"rows": rows},
                (z, z, z, Int32(1), Int32(1), Int32(1), Int32(1), s),
            )
    if top1_rows:
        from .distributed_argmax import DistributedArgmax

        for max_rows in top1_rows:
            out[f"distributed_argmax_{max_rows}"] = (
                DistributedArgmax(max_rows),
                {"max_rows": max_rows},
                (
                    *(z,) * 6,
                    *(Int32(1),) * 5,
                    Uint64(1),
                    Int32(0),
                    Uint32(0),
                    z,
                    Int32(0),
                    s,
                ),
            )
    if eh_norm is not None:
        from .mtp_embed_eh_norm import EmbedEhNorm

        out["embed_eh_norm"] = (
            EmbedEhNorm(HIDDEN, **eh_norm),
            {"hidden": HIDDEN, **eh_norm},
            (*(z,) * 10, Float32(1e-6), Int32(1), s),
        )
    return out
