# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Projection publish: a replicated projection computed as column slices that every rank's GEMM epilogue publishes.

For a projection ``y = x @ W.T`` whose input ``x`` is identical on every tensor-parallel rank (the MoE pooler after the
attention boundary, the unpooler after the expert all-reduce), each rank multiplies only its contiguous ``N / world``
block of ``W`` rows. The GEMM (``opp_tgv_gemm`` push_mode 7, SGLang's TGV kernel) stores each output vector, through
the multicast address of the tensor-parallel push workspace, into this rank's block of a gather slot in every rank's
copy: the slot layout ``[source rank][M][N / world]``, the phase halves and the empty-marker rule (+0.0 in either 16-bit
half = not arrived; an exact +0.0 half is sent as -0.0) are those of SGLang's own all-gather (``gemm_ag`` /
``all_gather_norm_residual``). The consumers take every vector from the local copy, reset it, and run the arithmetic of
the stock kernel they replace (``csrc/proj_publish.cuh``):

- ``pub_ag_norm``: the FFN boundary of SGLang's ``all_gather_norm_residual`` without its own push of the local slice;
- ``pub_norm_fp4_quant``: SGLang's pre-expert RMS norm + per-token NVFP4 quantization (quantize-only form).

A producer and its consumer form one use of the push plane: the producer reads the phase after its dependency wait and
never advances it; the consumer, launched next on the same stream, reads the same phase and advances it at its end, as
every user of the plane does. Results equal the stock pair "replicated GEMM + stock kernel" up to (a) the GEMM's
summation order where the slice GEMM and the replicated GEMM differ, and (b) the sign of exact-zero projection outputs
(the empty-marker remap), which SGLang's all-gather path shares.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from .. import sgl_compat

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/compile/paths.py": ("KERNEL_PATH",),
    "kernels/jit/utils/__init__.py": (
        "is_arch_support_pdl",
        "load_jit",
        "make_cpp_args",
    ),
    "srt/distributed/device_communicators/custom_all_reduce_v2.py": (
        "CustomAllReduceV2",
    ),
}


_CUH = Path(__file__).resolve().parents[1] / "csrc" / "proj_publish.cuh"
_WORLDS = (2, 4, 8)
NVFP4_PER_TOKEN_SCALE_INV = 1.0 / (448.0 * 6.0)


def _cache(fn):
    from functools import lru_cache

    return lru_cache(maxsize=None)(fn)


def _include_paths():
    from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH

    return sgl_compat.csrc_include_paths(KERNEL_PATH / "csrc")


@_cache
def _ag_norm_module(
    hidden: int,
    world: int,
    config: Tuple[int, int, bool],
    scale: bool,
    has_post: bool,
    gate: bool,
):
    from sglang.kernels.jit.utils import is_arch_support_pdl, load_jit, make_cpp_args

    cluster, threads, prefetch = config
    args = make_cpp_args(
        hidden,
        world,
        threads,
        cluster,
        is_arch_support_pdl(),
        scale,
        prefetch,
        has_post,
        gate,
    )
    return load_jit(
        "muse_vr_pub_ag_norm",
        *args,
        cuda_files=[str(_CUH)],
        cuda_wrappers=[("run", f"MuseVrPubAgNorm<{args}>::run")],
        extra_include_paths=_include_paths(),
        extra_cuda_cflags=["-O3"],
    )


@_cache
def _quant_module(hidden: int, world: int, threads: int):
    from sglang.kernels.jit.utils import is_arch_support_pdl, load_jit, make_cpp_args

    args = make_cpp_args(hidden, world, threads, is_arch_support_pdl())
    return load_jit(
        "muse_vr_pub_norm_fp4_quant",
        *args,
        cuda_files=[str(_CUH)],
        cuda_wrappers=[("run", f"MuseVrPubNormFp4Quant<{args}>::run")],
        extra_include_paths=_include_paths(),
        extra_cuda_cflags=["-O3"],
    )


def step_aside_reason(comm, num_tokens: int, n_full: int, k: int) -> Optional[str]:
    """None when a publish of an [num_tokens, k] x [n_full, k]^T projection can run on ``comm``; else the reason."""
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

    if not isinstance(comm, CustomAllReduceV2) or comm.disabled:
        return "no custom all-reduce"
    if comm.world_size not in _WORLDS:
        return "world"
    if (
        not getattr(comm, "has_multicast", False)
        or getattr(comm, "_push_multicast_ptr", None) is None
    ):
        return "no multicast"
    if num_tokens <= 0:
        return "rows"
    if n_full % (comm.world_size * 128) or k % 128:
        return "shape"
    if num_tokens * n_full * 2 > comm.max_push_size:
        return "push slot size"
    return None


_meta: Dict[Tuple[int, int, int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
_carriers: Dict[Tuple[int, int], torch.Tensor] = {}
_MAX_ROWS = 512


def _carrier(device: torch.device, n_slice: int, m: int) -> torch.Tensor:
    """A [m, n_slice] view of one shared shape-carrier buffer per (device, n_slice): the publish GEMM never writes its
    plain output, so every row count shares one allocation."""
    key = (device.index or 0, n_slice)
    buf = _carriers.get(key)
    if buf is None:
        buf = torch.empty(_MAX_ROWS, n_slice, dtype=torch.bfloat16, device=device)
        _carriers[key] = buf
    if m > _MAX_ROWS:
        raise RuntimeError(f"proj_publish: {m} rows > {_MAX_ROWS}")
    return buf[:m]


# Call-uniform pairing: a publish must be followed by its consumer as the very next use of the push plane, and a
# consumer must have a publish to take, or a half would be left holding words (or a consumer would wait forever).
# Every rank makes the same calls (the inputs of every decision are rank-uniform); this host check refuses an
# unpaired call on the rank that makes it.
_pending: Dict[int, Tuple[str, int, int]] = {}


def _begin(comm, site: str, m: int, n_full: int) -> None:
    key = id(comm)
    if key in _pending:
        raise RuntimeError(
            f"proj_publish: publish {site} while {_pending[key]} has no consumer yet"
        )
    _pending[key] = (site, m, n_full)


def _end(comm, site: str, m: int, n_full: int) -> None:
    got = _pending.get(id(comm))
    if got != (site, m, n_full):
        raise RuntimeError(
            f"proj_publish: {site} consumer for (M={m}, N={n_full}) without its publish (pending: {got})"
        )
    del _pending[id(comm)]


def pending(comm):
    """The unconsumed publish on ``comm`` (site, M, N), or None."""
    return _pending.get(id(comm))


def _publish_meta(comm, m: int, n_slice: int, device: torch.device):
    """(dummy output [m, n_slice], int64 argument block) of a push_mode 7 GEMM, cached per (device, m, n_slice, plane)."""
    from .. import opp_tgv_gemm as g

    key = (device.index or 0, m, n_slice, int(comm._push_multicast_ptr))
    hit = _meta.get(key)
    if hit is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "proj_publish: first use of a row count inside graph capture (warm-up missing)"
            )
        w = comm.world_size
        out = _carrier(device, n_slice, m)  # shape carrier only (no plain stores)
        meta = g.make_push_meta(
            counter_addr=comm._push_counter.data_ptr(),
            phase_stride=w * comm.max_push_size,
            rank=comm.rank,
            rows=m,
            chunk_rows=0,
            out=out,
            peer_slot_bases=[
                int(comm._push_multicast_ptr) + comm.rank * m * n_slice * 2
            ],
        )
        hit = (out, meta)
        _meta[key] = hit
    return hit


def publish_gemm(
    comm, x: torch.Tensor, weight: torch.Tensor, site: str, tactic: Optional[int] = None
) -> None:
    """Publish this rank's column block of ``x @ weight.T`` (``weight`` = the full replicated [N, K]) into the gather slot
    of the current phase on every rank. ``site`` ("ag_norm" or "norm_fp4_quant") names the consumer, which must be the
    next user of the plane."""
    from .. import opp_tgv_gemm as g

    assert (
        x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.is_contiguous()
    )
    m, (n, k) = x.shape[0], weight.shape
    r = step_aside_reason(comm, m, n, k)
    if r is not None:
        raise RuntimeError(
            f"proj_publish: cannot publish [{m}, {k}] x [{n}, {k}]^T: {r}"
        )
    w = comm.world_size
    n_slice = n // w
    w_slice = weight[comm.rank * n_slice : (comm.rank + 1) * n_slice]
    out, meta = _publish_meta(comm, m, n_slice, x.device)
    _begin(comm, site, m, n)
    g.run_tgv(x, w_slice, out, tactic=tactic, push_mode=7, push_world=w, meta=meta)


def pub_ag_norm(
    comm,
    stream: torch.Tensor,
    post_eps: float,
    pre_eps: float,
    *,
    post_norm_is_input_scale: bool = False,
    post_weight: Optional[torch.Tensor] = None,
    pre_weight: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The FFN boundary of the published projection: ``(output bf16 [M, H], new_stream fp32 [M, H])``, as
    ``all_gather_norm_residual`` returns them, with SGLang's launch shape for ``(H, M)``."""
    norm_residual_launch_config = sgl_compat.module(
        "sglang.kernels.ops.muse_spark_v12.norm_residual"
    ).norm_residual_launch_config

    num_tokens, hidden = stream.shape
    _end(comm, "ag_norm", num_tokens, hidden)
    config = norm_residual_launch_config(hidden_size=hidden, num_tokens=num_tokens)
    output = torch.empty(num_tokens, hidden, dtype=torch.bfloat16, device=stream.device)
    new_stream = torch.empty_like(stream)
    module = _ag_norm_module(
        hidden,
        comm.world_size,
        tuple(config),
        bool(post_norm_is_input_scale),
        post_weight is not None,
        alpha is not None,
    )
    module.run(
        comm.obj,
        stream,
        output,
        new_stream,
        post_weight,
        pre_weight,
        alpha,
        beta,
        float(post_eps),
        float(pre_eps),
    )
    return output, new_stream


def pub_norm_fp4_quant(
    comm,
    num_tokens: int,
    hidden: int,
    weight: Optional[torch.Tensor],
    eps: float,
    device: torch.device,
    global_scale_inv: float = NVFP4_PER_TOKEN_SCALE_INV,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(fp4 [M, K/2] u8, sf [M, K/16] u8, per_token_scale [M] fp32)`` of the published pooler output, as
    ``fused_norm_fp4_quant(quantize=True)`` returns them."""
    _num_threads = sgl_compat.module(
        "sglang.kernels.ops.muse_spark_v12.norm_fp4_quant"
    )._num_threads

    _end(comm, "norm_fp4_quant", num_tokens, hidden)
    fp4 = torch.empty(num_tokens, hidden // 2, dtype=torch.uint8, device=device)
    sf = torch.empty(num_tokens, hidden // 16, dtype=torch.uint8, device=device)
    pts = torch.empty(num_tokens, dtype=torch.float32, device=device)
    module = _quant_module(hidden, comm.world_size, _num_threads(hidden))
    module.run(comm.obj, weight, fp4, sf, pts, float(eps), float(global_scale_inv))
    return fp4, sf, pts
