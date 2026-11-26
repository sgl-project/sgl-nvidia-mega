# SPDX-License-Identifier: Apache-2.0
"""Audited BF16 scorer and exact ordered sampled / one-pass top-k; source math retained."""

import functools

from .._jit import is_arch_support_pdl, load_jit, make_cpp_args
from .mqa import HAS_TILELANG

SGLANG_READS = {
    "kernels/ops/attention/fast_topk.py": ("_jit_fast_topk_module",),
    "srt/layers/attention/qsa/kernel.py": ("_expand_qsa_block_indices_kernel",),
}

_BLOCK_Q = 32
_STOCK_TILE = 64
_SAMPLED_MIN_KEYS = 4096
_SCORER_CTAS_PER_SM = 1
CONFIG = (128, 512, 2, "Square", 1)
_QSA_INV_SQRT128_PRELUDE = r"""
__device__ __forceinline__ float qsa_mul_inv_sqrt128(float x) {
  float y;
  asm("mul.rn.f32 %0, %1, 0f3DB504F3;" : "=f"(y) : "f"(x));
  return y;
}
"""

if HAS_TILELANG:
    import tilelang
    from tilelang import language as T

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        }
    )
    def _prefill_scorer_kernel(
        heads: int,
        head_dim: int,
        block_n: int = 128,
        block_q: int = 32,
        num_stages: int = 2,
        threads: int = 512,
        policy: str = "Square",
        dtype: str = "bfloat16",
    ):
        """``_tilelang_qsa_mqa_prefill_relative_scaled_kernel`` (mqa.py) with a
        configurable key tile and warp split, and a runtime key split (the
        length of ``SplitTag``). Per (row, key) the arithmetic is the stock
        kernel's; the store is cut at the stock kernel's 64-key coverage."""
        rows = T.dynamic("rows")
        keys = T.dynamic("keys")
        width = T.dynamic("width")
        splits = T.dynamic("splits")
        warp_policy = getattr(T.GemmWarpPolicy, policy)

        @T.prim_func
        def kernel(
            Q: T.Tensor([rows * heads, head_dim], dtype),  # type: ignore
            K: T.Tensor([keys, head_dim], dtype),  # type: ignore
            Logits: T.Tensor([rows, width], T.float32),  # type: ignore
            Starts: T.Tensor([rows], T.int32),  # type: ignore
            Ends: T.Tensor([rows], T.int32),  # type: ignore
            SplitTag: T.Tensor([splits], T.int32),  # type: ignore
        ):
            with T.Kernel(
                T.ceildiv(rows, block_q),
                splits,
                threads=threads,
                prelude=_QSA_INV_SQRT128_PRELUDE,
            ) as (bx, by):
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
                # Stock walks 64-key tiles from start_min; never write past them.
                limit = start_min + T.ceildiv(end_max - start_min, 64) * 64
                num_tiles = T.ceildiv(end_max - start_min, block_n)
                tiles_per_split = T.ceildiv(num_tiles, splits)
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
                        policy=warp_policy,
                    )
                    for n, qi, head in T.Parallel(block_n, block_q, heads):
                        scores_3d[n, qi, head] = T.max(scores_3d[n, qi, head], 0.0)
                    T.reduce_sum(scores_3d, reduced, dim=-1, clear=True)
                    for qi, n in T.Parallel(block_q, block_n):
                        column = key_base + n - Starts[row_base + qi]
                        if key_base + n < limit:
                            if column >= 0:
                                if column < width:
                                    Logits[row_base + qi, column] = T.call_extern(
                                        "float32", "qsa_mul_inv_sqrt128", reduced[n, qi]
                                    )

        return kernel


def plan_chunk_rows(rows: int, width: int, topk_slots: int, budget_bytes: int) -> int:
    """Rows per logits chunk: stock's budget rule (``_qsa_prefill_row_chunk_size``),
    then rounded down to whole top-k waves when the budget holds at least one."""
    if rows <= 0 or width <= 0:
        return max(rows, 1)
    max_rows = max(_BLOCK_Q, budget_bytes // (width * 4))
    max_rows = max(_BLOCK_Q, max_rows // _BLOCK_Q * _BLOCK_Q)
    if rows <= max_rows:
        return rows
    unit = topk_slots // _BLOCK_Q * _BLOCK_Q
    if unit >= _BLOCK_Q and max_rows >= unit:
        return max_rows // unit * unit
    return max_rows


def plan_key_splits(row_blocks: int, width: int, block_n: int, target_ctas: int) -> int:
    """Key splits for about ``target_ctas`` CTAs (whole waves), at least 1 and
    never more than the longest row's tiles."""
    max_useful = max(1, -(-width // block_n))
    return max(1, min(max_useful, target_ctas // max(row_blocks, 1), 4096))


def score_chunk(
    q_rows, keys_2d, logits, starts, ends, width, config=None, *, split_tag, sms
):
    """Score ``q_rows`` [padded rows, heads, dim] (rows a multiple of 32) into
    ``logits`` [padded rows, width] fp32, stock's request-relative layout."""
    block_n, threads, stages, policy, waves = config or CONFIG
    heads, head_dim = q_rows.shape[1:]
    row_blocks = q_rows.shape[0] // _BLOCK_Q
    target = waves * _SCORER_CTAS_PER_SM * sms
    splits = plan_key_splits(row_blocks, width, block_n, target)
    _prefill_scorer_kernel(
        heads=heads,
        head_dim=head_dim,
        block_n=block_n,
        block_q=_BLOCK_Q,
        num_stages=stages,
        threads=threads,
        policy=policy,
    )(
        q_rows.reshape(-1, head_dim),
        keys_2d,
        logits,
        starts,
        ends,
        split_tag[:splits],
    )
    return splits


def _topk_impl_for(width: int, impl: str) -> str:
    # No row is longer than the width; rows of 4096 or fewer keys take fast_topk's own
    # select inside qsa_prefill_topk anyway, under its 32-register cap (about 5% slower).
    if impl in ("sampled", "onepass") and width <= _SAMPLED_MIN_KEYS:
        return "stock"
    return impl


@functools.lru_cache(maxsize=None)
def _topk_launcher(impl: str):
    """(score, row_starts, indices, lengths) -> None for top-k 512, writing indices."""
    if impl == "stock":
        from sglang.kernels.ops.attention.fast_topk import _jit_fast_topk_module

        return _jit_fast_topk_module(512).fast_topk

    args = make_cpp_args(is_arch_support_pdl())
    if impl == "half":
        # The sampled select on 512-thread blocks, the same output; also takes the
        # per-row flags (int32 [rows] scratch) of the rows its 1024-thread launch serves.
        return load_jit(
            "qsa_prefill_topk_half",
            *args,
            cuda_files=["elementwise/qsa_prefill_topk_half.cuh"],
            cuda_wrappers=[("topk", f"QsaPrefillTopKHalfKernel<{args}>::run")],
        ).topk
    if impl == "onepass":
        # Also takes per-row flags (int32 [rows] scratch) of the rows its second,
        # 1024-thread launch serves.
        return load_jit(
            "qsa_topk_onepass",
            *args,
            cuda_files=["elementwise/qsa_topk_onepass.cuh"],
            cuda_wrappers=[("topk", f"QsaTopKOnePassKernel<{args}>::run")],
        ).topk
    return load_jit(
        "qsa_prefill_topk",
        *args,
        cuda_files=["elementwise/qsa_prefill_topk.cuh"],
        cuda_wrappers=[("topk", f"QsaPrefillTopKKernel<{args}>::run")],
    ).topk


def _expand_into(
    kernel,
    block_indices,
    query_positions,
    sequence_lengths,
    output,
    compress_ratio,
    token_topk,
):
    """``triton_expand_qsa_block_indices`` writing ``output`` in place: the same
    kernel, constexprs and per-row values, one launch for every row."""
    import triton

    rows, block_topk = block_indices.shape
    final_topk = token_topk + compress_ratio - 1
    assert output.shape == (rows, final_topk) and output.stride(1) == 1
    kernel[(rows,)](
        block_indices,
        query_positions.to(device=block_indices.device).contiguous(),
        sequence_lengths.to(device=block_indices.device).contiguous(),
        output,
        block_indices.stride(0),
        output.stride(0),
        BLOCK_TOPK=block_topk,
        COMPRESS_RATIO=compress_ratio,
        TOKEN_TOPK=token_topk,
        FINAL_TOPK=final_topk,
        OUTPUT_BLOCK_SIZE=triton.next_power_of_2(final_topk),
        num_warps=8,
    )


def _score_tc(q, keys_2d, logits, starts, ends, width, sms):
    """The tensor-core scorer into ``logits``; False when it does not cover the chunk."""
    from . import prefill_scorer_tc

    if not prefill_scorer_tc.applies(q, keys_2d):
        return False
    prefill_scorer_tc.score(q, keys_2d, logits, starts, ends, width, sms)
    return True


def select_into(
    io, width, block_topk=512, ratio=4, scorer_tc=False, topk_impl="sampled"
):
    from sglang.srt.layers.attention.qsa.kernel import _expand_qsa_block_indices_kernel

    chunk = io.chunk_rows
    rows = io.query.shape[0]
    impl = _topk_impl_for(width, topk_impl)
    topk = _topk_launcher(impl)
    for r0 in range(0, rows, chunk):
        r1 = min(rows, r0 + chunk)
        n = r1 - r0
        pad = (-n) % 32
        q = io.query[r0:r1]
        starts = io.row_starts[r0:r1]
        ends = io.row_ends[r0:r1]
        if pad:
            qpad = io.query_scratch_out[: n + pad]
            spad = io.starts_scratch_out[: n + pad]
            epad = io.ends_scratch_out[: n + pad]
            qpad[:n].copy_(q)
            qpad[n:].zero_()
            spad[:n].copy_(starts)
            spad[n:].copy_(starts[-1:].expand(pad))
            epad[:n].copy_(ends)
            epad[n:].copy_(ends[-1:].expand(pad))
            q, starts, ends = qpad, spad, epad
        keys, logits = io.compressed_keys[:, 0], io.logits_out[: n + pad]
        if not (
            scorer_tc and _score_tc(q, keys, logits, starts, ends, width, io.scorer_sms)
        ):
            score_chunk(
                q,
                keys,
                logits,
                starts,
                ends,
                width,
                CONFIG,
                split_tag=io.split_tag,
                sms=io.scorer_sms,
            )
        # The one-pass flags go to the starts scratch: the chunk's scorer, queued
        # before it on this stream, is its only reader.
        flags = (io.starts_scratch_out[:n],) if impl == "onepass" else ()
        topk(
            io.logits_out[:n],
            io.zero_offsets[:n],
            io.block_indices_out[r0:r1],
            io.row_lengths[r0:r1],
            *flags,
        )
    _expand_into(
        _expand_qsa_block_indices_kernel,
        io.block_indices_out,
        io.query_positions,
        io.sequence_lengths,
        io.selected_indices_out,
        ratio,
        block_topk * ratio,
    )
