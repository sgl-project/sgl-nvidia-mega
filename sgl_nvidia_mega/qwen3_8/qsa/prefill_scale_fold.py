# SPDX-License-Identifier: Apache-2.0
# From SGLang python/sglang/srt/layers/attention/qsa/mqa.py: the request-relative prefill scorer with the 1/sqrt(128) scale folded
# into its store (the inline-PTX prelude and the scaled TileLang kernel, unchanged) and
# that commit's ``tilelang_qsa_mqa_prefill_relative``, whose SGLANG_OPT_QSA_PREFILL_SCALE_FOLD
# read became the ``scale_fold`` argument. The unscaled kernel and the helpers it shares
# with that function are SGLang main's mqa.py, which defines them identically.
"""QSA prefill scoring into request-relative columns with the score scale folded into
the store: the logits are bitwise the stock kernel's followed by its ``div_`` by
sqrt(128) (CUDA divides a tensor by a CPU scalar as a multiply by the FP32 reciprocal,
0x3DB504F3, which the store applies with ``mul.rn``), without the divide pass."""

import math
from typing import Optional

import torch
from sglang.srt.layers.attention.qsa.mqa import (
    _TILELANG_DTYPES,
    HAS_TILELANG,
    _prefill_key_splits,
    _scoring_dtype,
    _validate_k,
    _validate_q,
)
from sglang.srt.utils import is_hip

if HAS_TILELANG:
    import tilelang
    from tilelang import language as T

SGLANG_READS = {
    "srt/layers/attention/qsa/mqa.py": (
        "_TILELANG_DTYPES",
        "HAS_TILELANG",
        "_prefill_key_splits",
        "_scoring_dtype",
        "_validate_k",
        "_validate_q",
        "_tilelang_qsa_mqa_prefill_relative_kernel",
    ),
    "srt/utils/device.py|srt/utils/common.py": ("is_hip",),
}

# CUDA's div_ by a CPU scalar multiplies by the FP32 reciprocal: for sqrt(128)
# that is 0x3DB504F3. Inline PTX keeps the multiply IEEE (.rn, no .ftz) under
# fast math, for the scaled relative prefill kernel.
_QSA_INV_SQRT128_PRELUDE = r"""
__device__ __forceinline__ float qsa_mul_inv_sqrt128(float x) {
  float y;
  asm("mul.rn.f32 %0, %1, 0f3DB504F3;" : "=f"(y) : "f"(x));
  return y;
}
"""

if HAS_TILELANG:
    # The relative kernel above with the 1/sqrt(128) scale folded into its store;
    # GEMM, ReLU, head sum and written positions are unchanged.
    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        }
    )
    def _tilelang_qsa_mqa_prefill_relative_scaled_kernel(
        heads: int,
        head_dim: int,
        block_n: int = 64,
        block_q: int = 32,
        num_stages: int = 3,
        threads: int = 512,
        key_splits: int = 1,
        dtype: str = "bfloat16",
    ):
        rows = T.dynamic("rows")
        keys = T.dynamic("keys")
        width = T.dynamic("width")

        @T.prim_func
        def kernel(
            Q: T.Tensor([rows * heads, head_dim], dtype),  # type: ignore
            K: T.Tensor([keys, head_dim], dtype),  # type: ignore
            Logits: T.Tensor([rows, width], T.float32),  # type: ignore
            Starts: T.Tensor([rows], T.int32),  # type: ignore
            Ends: T.Tensor([rows], T.int32),  # type: ignore
        ):
            with T.Kernel(
                T.ceildiv(rows, block_q),
                key_splits,
                threads=threads,
                prelude=_QSA_INV_SQRT128_PRELUDE,
            ) as (
                bx,
                by,
            ):
                q_shared = T.alloc_shared([block_q * heads, head_dim], dtype)
                k_shared = T.alloc_shared([block_n, head_dim], dtype)
                scores = T.alloc_fragment([block_n, block_q * heads], T.float32)
                scores_3d = T.reshape(scores, (block_n, block_q, heads))
                reduced = T.alloc_fragment([block_n, block_q], T.float32)
                row_base = bx * block_q
                start_min = T.alloc_var(T.int32)
                end_max = T.alloc_var(T.int32)
                start_min = 2147483647
                end_max = -2147483648
                for qi in T.serial(block_q):
                    start_min = T.min(start_min, T.min(Starts[row_base + qi], keys))
                    end_max = T.max(end_max, T.min(Ends[row_base + qi], keys))
                num_tiles = T.ceildiv(end_max - start_min, block_n)
                tiles_per_split = T.ceildiv(num_tiles, key_splits)
                tile_begin = by * tiles_per_split
                tile_count = T.max(T.min(num_tiles - tile_begin, tiles_per_split), 0)

                T.copy(Q[row_base * heads, 0], q_shared)
                for ti in T.Pipelined(tile_count, num_stages=num_stages):
                    key_base = start_min + (tile_begin + ti) * block_n
                    T.copy(K[key_base, 0], k_shared)
                    T.gemm(
                        k_shared,
                        q_shared,
                        scores,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
                    for n, qi, head in T.Parallel(block_n, block_q, heads):
                        scores_3d[n, qi, head] = T.max(scores_3d[n, qi, head], 0.0)
                    T.reduce_sum(scores_3d, reduced, dim=-1, clear=True)
                    for qi, n in T.Parallel(block_q, block_n):
                        column = key_base + n - Starts[row_base + qi]
                        # A row block can straddle requests; keep each row to
                        # its own window.
                        if column >= 0:
                            if column < width:
                                Logits[row_base + qi, column] = T.call_extern(
                                    "float32", "qsa_mul_inv_sqrt128", reduced[n, qi]
                                )

        return kernel


def tilelang_qsa_mqa_prefill_relative(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    width: int,
    score_scale: Optional[float] = None,
    *,
    scale_fold: bool = True,
) -> torch.Tensor:
    """Packed prefill scoring into request-relative columns.

    Returns fp32 [rows, width] where ``[r, :row_ends[r] - row_starts[r]]`` holds
    the same values ``tilelang_qsa_mqa_prefill`` puts at
    ``[r, row_starts[r]:row_ends[r]]``. Columns past a row's length are left
    undefined: callers must read only each row's own section (the fast top-k
    reads ``[start, start + length)``). ``width`` must cover the longest row
    and comes from host lengths, so no device sync is needed to size it."""

    if not HAS_TILELANG:
        raise RuntimeError("TileLang is unavailable")
    _validate_q(q)
    _validate_k(k)
    rows, keys = q.shape[0], k.shape[0]
    width = max(int(width), 1)
    if not rows or not keys:
        return torch.full(
            (rows, width), -float("inf"), dtype=torch.float32, device=q.device
        )
    heads, head_dim = q.shape[1:]
    block_q = max(1, 128 // heads)
    block_n = 64
    padding = (-rows) % block_q
    padded_rows = rows + padding
    logits = torch.empty((padded_rows, width), dtype=torch.float32, device=q.device)
    scoring_dtype = _scoring_dtype(q, k)
    q_padded = q.to(scoring_dtype).contiguous()
    starts = row_starts.to(device=q.device, dtype=torch.int32).contiguous()
    ends = row_ends.to(device=q.device, dtype=torch.int32).contiguous()
    if padding:
        q_padded = torch.cat([q_padded, q_padded.new_zeros(padding, heads, head_dim)])
        starts = torch.cat([starts, starts[-1:].expand(padding)])
        ends = torch.cat([ends, ends[-1:].expand(padding)])

    # The store's constant is the FP32 reciprocal of sqrt(128); any other scale
    # keeps the divide. The helper is inline PTX, so ROCm keeps the divide too.
    fold = scale_fold and score_scale is None and head_dim == 128 and not is_hip()
    (_tilelang_qsa_mqa_prefill_relative_scaled_kernel if fold else _unscaled_kernel())(
        heads=heads,
        head_dim=head_dim,
        block_n=block_n,
        block_q=block_q,
        key_splits=_prefill_key_splits(
            padded_rows // block_q, width, block_n, q.device
        ),
        dtype=_TILELANG_DTYPES[scoring_dtype],
    )(
        q_padded.reshape(-1, head_dim),
        k[:, 0].to(scoring_dtype).contiguous(),
        logits,
        starts,
        ends,
    )
    logits = logits[:rows]
    if not fold:
        logits.div_(score_scale or math.sqrt(head_dim))
    return logits


def _unscaled_kernel():
    """SGLang main's unscaled relative kernel (the fallback when the fold does not apply)."""
    from sglang.srt.layers.attention.qsa.mqa import (
        _tilelang_qsa_mqa_prefill_relative_kernel,
    )

    return _tilelang_qsa_mqa_prefill_relative_kernel
